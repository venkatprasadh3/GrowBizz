# slack_bot.py
import os
import re
import logging
import concurrent.futures
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
import pandas as pd
from fpdf import FPDF
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import datetime
import smtplib
import seaborn as sns
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from textblob import TextBlob
from pytrends.request import TrendReq
import numpy as np
from scipy.stats import norm
import uuid
from io import BytesIO
from twilio.rest import Client
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from PIL import Image, ImageDraw
import image_gen  # Import promotion image generation

# Configuration and Constants
sns.set_style("darkgrid")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Environment Variables for API Keys and Tokens
SLACK_BOT_TOKEN = os.environ["SLACK_BOT_TOKEN"]
SLACK_APP_TOKEN = os.environ["SLACK_APP_TOKEN"]
GENAI_API_KEY = os.environ["GENAI_API_KEY"]  # Used in image_gen.py
TWILIO_ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN", "")
TWILIO_PHONE_NUMBER = os.environ.get("TWILIO_PHONE_NUMBER", "")

# Other Constants
SALES_DATA_PATH = os.getenv("SALES_DATA_PATH", "sales_data.csv")
CUSTOMER_DATA_PATH = os.getenv("CUSTOMER_DATA_PATH", "customer_shopping_data.csv")
LOGO_PATH = os.getenv("LOGO_PATH", "psg_logo_blue.png")
WHATSAPP_NUMBER = os.getenv("WHATSAPP_NUMBER", "+919944934545")
EMAIL_FROM = os.getenv("EMAIL_FROM", "21z268@psgtech.ac.in")
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD", "")
DEFAULT_LANGUAGE = os.getenv("DEFAULT_LANGUAGE", "English")

# Global Variables
user_states = {}  # Tracks user-specific conversation state
client = WebClient(token=SLACK_BOT_TOKEN)
pytrends = TrendReq(hl='en-IN', tz=330)
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN) if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN else None

# Static Fallback Responses
FALLBACK_RESPONSES = {
    "register": "Sorry, registration failed. Please try again later.",
    "purchase": "Purchase could not be processed. Please check back later.",
    "weekly analysis": "Weekly analysis unavailable. Data might be missing.",
    "insights": "Insights generation failed. Please try again.",
    "simple insights": "Simple insights unavailable. Check back soon!",
    "promotion": "Promotion image generation failed. Try again later.",
    "whatsapp": "Failed to send WhatsApp message. Please try again.",
    "email": "Email sending failed. Please retry later.",
    "invoice": "Sorry, invoice generation failed. Please try again.",
    "default": "Oops! Something went wrong. Try: register, purchase, weekly analysis, insights, simple insights, promotion, whatsapp, email, invoice",
    "not_in_channel": "I can't respond here. Please invite me to this channel with `/invite @YourBotName` or send me a direct message!",
    "channel_not_found": "Specified channel not found. File sent to your DM instead."
}

# Helper to Get DM Channel
def get_dm_channel(user_id):
    try:
        response = client.conversations_open(users=user_id)
        dm_channel = response['channel']['id']
        logging.info(f"Opened DM channel {dm_channel} for user {user_id}")
        return dm_channel
    except SlackApiError as e:
        logging.error(f"Failed to open DM channel for {user_id}: {e}")
        return None

# Messaging Functions
def send_whatsapp_message(phone_number, message):
    if not twilio_client:
        return "WhatsApp not configured."
    try:
        msg = twilio_client.messages.create(
            body=message,
            from_=f"whatsapp:{TWILIO_PHONE_NUMBER}",
            to=f"whatsapp:{phone_number}"
        )
        return f"WhatsApp message sent to {phone_number}!"
    except Exception as e:
        logging.error(f"WhatsApp error: {e}")
        return "Failed to send WhatsApp message."

def send_email(subject, body, to_emails, from_email, password, images=None, attachments=None):
    if not EMAIL_PASSWORD:
        return FALLBACK_RESPONSES["email"] + " (Email not configured)"
    if isinstance(to_emails, str):
        to_emails = [to_emails]
    to_emails = [re.sub(r'mailto:|\|.*$', '', email.strip()) for email in to_emails]
    logging.info(f"Sending email to: {to_emails}")

    msg = MIMEMultipart()
    msg["Subject"] = subject
    msg["From"] = from_email
    msg["To"] = ", ".join(to_emails)
    msg.attach(MIMEText(body, "plain"))

    if images:
        images = [images] if not isinstance(images, list) else images
        for image_path in images:
            try:
                with open(image_path, 'rb') as img_file:
                    img = MIMEImage(img_file.read(), name=os.path.basename(image_path))
                    msg.attach(img)
            except FileNotFoundError:
                logging.error(f"Image not found: {image_path}")

    if attachments:
        attachments = [attachments] if not isinstance(attachments, list) else attachments
        for attachment_path in attachments:
            try:
                with open(attachment_path, 'rb') as file:
                    part = MIMEApplication(file.read(), name=os.path.basename(attachment_path))
                    part['Content-Disposition'] = f'attachment; filename="{os.path.basename(attachment_path)}"'
                    msg.attach(part)
            except FileNotFoundError:
                logging.error(f"Attachment not found: {attachment_path}")

    smtp = None
    try:
        smtp = smtplib.SMTP("smtp.gmail.com", 587)
        smtp.starttls()
        smtp.login(from_email, password)
        smtp.sendmail(from_email, to_emails, msg.as_string())
        logging.info("Email sent successfully!")
        return "Email sent successfully!"
    except Exception as e:
        logging.error(f"Failed to send email: {e}")
        return FALLBACK_RESPONSES["email"]
    finally:
        if smtp:
            smtp.quit()

# Load Sales Data
def load_sales_data():
    try:
        if os.path.exists(SALES_DATA_PATH):
            df = pd.read_csv(SALES_DATA_PATH, low_memory=False)
            df['Order Date'] = pd.to_datetime(df['Order Date'], format='%m/%d/%y %H:%M', errors='coerce')
            df.dropna(subset=['Order Date'], inplace=True)
            df['Quantity Ordered'] = pd.to_numeric(df['Quantity Ordered'], errors='coerce').fillna(0).astype('Int64')
            df['Price Each'] = pd.to_numeric(df['Price Each'], errors='coerce').fillna(0.0)
            return df
        elif os.path.exists(CUSTOMER_DATA_PATH):
            df = pd.read_csv(CUSTOMER_DATA_PATH, low_memory=False)
            df['invoice_date'] = pd.to_datetime(df['invoice_date'], errors='coerce')
            df.dropna(subset=['invoice_date'], inplace=True)
            df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce').fillna(0).astype('Int64')
            df['price'] = pd.to_numeric(df['price'], errors='coerce').fillna(0.0)
            df = df.rename(columns={
                'invoice_date': 'Order Date',
                'category': 'Product',
                'quantity': 'Quantity Ordered',
                'price': 'Price Each',
                'shopping_mall': 'Purchase Address'
            })
            return df
        else:
            raise FileNotFoundError("No sales or customer data file found.")
    except Exception as e:
        logging.error(f"Error loading sales data: {e}")
        return None

# Customer Registration with State Management
def handle_customer_registration(user_id, text):
    if user_id not in user_states:
        user_states[user_id] = {'last_message': '', 'context': 'idle'}

    state = user_states[user_id]
    if 'customer_id' in state:
        return f"You are already registered with Customer ID: `{state['customer_id']}`."

    if "register:" in text.lower():
        try:
            _, details = text.lower().split("register:", 1)
            name, email, phone, language, address = [x.strip() for x in details.split(',', 4)]
            
            # Clean phone number
            phone = re.sub(r'<tel:|\|.*$', '', phone).strip().replace('+', '')
            phone = '+' + phone if not phone.startswith('+') else phone
            
            customer_id = str(uuid.uuid4())
            user_states[user_id] = {
                'customer_id': customer_id,
                'name': name,
                'email': email,
                'phone': phone,
                'language': language,
                'address': address,
                'context': 'registered'
            }
            
            welcome_msg = f"Welcome {name}! Registration successful."
            whatsapp_response = send_whatsapp_message(phone, welcome_msg)
            return f"Registration successful! Customer ID: `{customer_id}`\n{whatsapp_response}"
        except Exception as e:
            logging.error(f"Registration error: {e}")
            return "Sorry, registration failed."
    return "Please use: `register: name, email, phone, language, address`"

# Promotion Generation
def generate_promotion(prompt, user_id):
    img_byte_arr = image_gen.generate_promotion_image(prompt)
    if img_byte_arr:
        size = img_byte_arr.getbuffer().nbytes
        logging.info(f"Generated promotion image, size: {size} bytes")
        dm_channel = get_dm_channel(user_id)
        if dm_channel:
            client.files_upload_v2(
                channels=dm_channel,
                file=img_byte_arr,
                filename=f"promotion_{str(uuid.uuid4())[:8]}.png",
                title=f"Promotion: {prompt}"
            )
            logging.info(f"Uploaded promotion image to DM {dm_channel}")
            return "Promotion image sent to your DM!"
        else:
            return "Failed to upload image: DM channel not available."
    else:
        logging.error("Promotion image generation failed")
        return "Promotion generation failed."

# Weekly Sales Analysis
def generate_weekly_sales_analysis(user_id):
    df = load_sales_data()
    if df is None:
        return "Weekly analysis unavailable. Data might be missing."

    plt.figure(figsize=(10, 5))
    weekly_sales = df.resample('W', on='Order Date')['Price Each'].sum()
    plt.plot(weekly_sales.index, weekly_sales.values, marker='o')
    plt.title("Weekly Sales Trend")
    img_byte_arr = BytesIO()
    plt.savefig(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    plt.close()

    size = img_byte_arr.getbuffer().nbytes
    logging.info(f"Generated weekly sales image, size: {size} bytes")

    if size > 0:
        dm_channel = get_dm_channel(user_id)
        if dm_channel:
            client.files_upload_v2(
                channels=dm_channel,
                file=img_byte_arr,
                filename="weekly_sales.png",
                title="Weekly Sales Trend"
            )
            logging.info(f"Uploaded weekly sales image to DM {dm_channel}")
            return "Weekly analysis sent to your DM!"
        else:
            return "Failed to upload image: DM channel not available."
    else:
        logging.error("Weekly sales image is empty")
        return "Weekly analysis failed. (Image was empty)"

# Invoice Generation
def generate_invoice(customer_id=None, product=None, user_id=None):
    class InvoicePDF(FPDF):
        def header(self):
            self.set_font("Arial", "B", 16)
            self.cell(0, 10, "INVOICE", align="C", ln=True)
            self.ln(10)

        def add_table(self, header, data):
            self.set_font("Arial", "B", 10)
            for col in header:
                self.cell(40, 10, col, border=1)
            self.ln()
            self.set_font("Arial", "", 10)
            for row in data:
                for col in row:
                    self.cell(40, 10, str(col), border=1)
                self.ln()

    pdf = InvoicePDF()
    pdf.add_page()

    try:
        logging.info(f"Generating invoice for user {user_id}")
        header = ["Product", "Qty", "Price", "Total"]
        if customer_id and product:
            invoice_items = [[product['product_name'], 1, product['price'], product['price']]]
        else:
            invoice_items = [["Sample Product", 1, 100.00, 100.00]]  # Default data

        pdf.add_table(header, invoice_items)
        pdf_bytes = pdf.output(dest='S').encode('latin-1')
        pdf_byte_arr = BytesIO(pdf_bytes)
        pdf_byte_arr.seek(0)
        filename = f"invoice_{customer_id or 'sample'}_{str(uuid.uuid4())[:8]}.pdf"

        size = pdf_byte_arr.getbuffer().nbytes
        logging.info(f"Generated PDF size: {size} bytes")

        if size > 0:
            dm_channel = get_dm_channel(user_id)
            if dm_channel:
                client.files_upload_v2(
                    channels=dm_channel,
                    file=pdf_byte_arr,
                    filename=filename,
                    title="Invoice"
                )
                logging.info(f"Uploaded invoice to DM {dm_channel}")
                return "Invoice generated and sent to your DM!"
            else:
                return "Failed to upload invoice: DM channel not available."
        else:
            logging.error("Generated invoice is empty")
            return "Sorry, invoice generation failed. (File was empty)"
    except Exception as e:
        logging.error(f"Invoice generation error: {e}")
        return "Sorry, invoice generation failed."

# Sales Insights
def generate_sales_insights():
    df = load_sales_data()
    if df is None:
        return FALLBACK_RESPONSES["simple insights"]
    
    try:
        total_sales = df["Price Each"].sum()
        avg_sale = df["Price Each"].mean()
        best_selling_product = df.groupby("Product")["Quantity Ordered"].sum().idxmax()
        return f"📊 Sales Insights:\n" \
               f"🔹 Total Sales: INR {total_sales:,.2f}\n" \
               f"🔹 Average Sale: INR {avg_sale:,.2f}\n" \
               f"🔥 Best Selling Product: {best_selling_product}"
    except Exception as e:
        logging.error(f"Insights error: {e}")
        return FALLBACK_RESPONSES["simple insights"]

# Query Processing with State Management
def process_query(text, user_id, event_channel):
    text = text.lower().strip()
    if user_id not in user_states:
        user_states[user_id] = {'last_message': '', 'context': 'idle'}
    
    state = user_states[user_id]
    state['last_message'] = text
    logging.info(f"Processing query: '{text}' for user {user_id}, context: {state['context']}, event_channel: {event_channel}")

    # Determine target_channel
    parts = text.split(" in ")
    if len(parts) == 2:
        command = parts[0].strip()
        target_channel = parts[1].strip()
    else:
        command = text.strip()
        target_channel = event_channel

    try:
        if "register" in command:
            return handle_customer_registration(user_id, command)
        elif "purchase:" in command:
            product_id = command.split(":")[1].strip()
            return process_purchase(product_id, user_id, target_channel)
        elif "weekly analysis" in command:
            state['context'] = 'idle'
            return generate_weekly_sales_analysis(user_id)
        elif "insights" in command or "simple insights" in command:
            state['context'] = 'idle'
            return generate_sales_insights()
        elif "promotion:" in command:
            prompt = command.replace("promotion:", "").strip()
            state['context'] = 'idle'
            return generate_promotion(prompt, user_id)
        elif "whatsapp" in command:
            message = "Hello, this is a test message from the bot!"
            if "in " in command:
                lang = command.split("in ")[-1].strip()
                message = translate_message(message, lang)
            state['context'] = 'idle'
            return send_whatsapp_message(WHATSAPP_NUMBER, message)
        elif "email" in command:
            df = load_sales_data()
            if df is None:
                return FALLBACK_RESPONSES["email"]
            top_customers = df.groupby("Purchase Address")["Price Each"].sum().nlargest(3)
            subject = "Top Customers Report"
            body = f"Top 3 customers:\n{top_customers.to_string()}"
            if "in " in command:
                lang = command.split("in ")[-1].strip()
                body = translate_message(body, lang)
            state['context'] = 'idle'
            return send_email(subject, body, [EMAIL_FROM], EMAIL_FROM, EMAIL_PASSWORD)
        elif "invoice" in command or "generate invoice" in command:
            state['context'] = 'idle'
            return generate_invoice(user_id=user_id)
        elif state['context'] == 'purchase_prompt':
            return "Please specify a product with: `purchase: product_name`"
        elif state['context'] == 'promotion_prompt':
            return "Please specify a promotion with: `promotion: item discount%`"
        else:
            state['context'] = 'idle'
            return FALLBACK_RESPONSES["default"]
    except Exception as e:
        logging.error(f"Query processing error: {e}")
        state['context'] = 'idle'
        return FALLBACK_RESPONSES["default"]

# Minimal HTTP Server for Render
class DummyHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/plain")
        self.end_headers()
        self.wfile.write(b"Slack bot is running")

    def do_HEAD(self):
        self.send_response(200)
        self.send_header("Content-type", "text/plain")
        self.end_headers()

def run_http_server():
    server = HTTPServer(("", int(os.getenv("PORT", 3000))), DummyHandler)
    server.serve_forever()

# Start the HTTP server in a background thread
threading.Thread(target=run_http_server, daemon=True).start()

# Slack Bot Setup
if __name__ == "__main__":
    from slack_bolt import App
    from slack_bolt.adapter.socket_mode import SocketModeHandler

    app = App(token=SLACK_BOT_TOKEN)
    
    @app.message(".*")
    def handle_message(event, say):
        user_id = event['user']
        text = event['text']
        channel = event['channel']
        logging.info(f"Received message: '{text}' from user {user_id} in channel {channel}")
        response = process_query(text, user_id, channel)
        try:
            say(response)
        except SlackApiError as e:
            if e.response["error"] == "not_in_channel":
                try:
                    client.chat_postMessage(
                        channel=user_id,
                        text=FALLBACK_RESPONSES["not_in_channel"]
                    )
                except SlackApiError as dm_error:
                    logging.error(f"Failed to send DM: {dm_error}")
            else:
                logging.error(f"Slack API error: {e}")
                raise

    SocketModeHandler(app, SLACK_APP_TOKEN).start()
