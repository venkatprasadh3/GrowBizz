import os
import re
import logging
import concurrent.futures
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
import google.generativeai as genai
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
import image_gen  # Import the new image generation module

# Configuration and Constants
sns.set_style("darkgrid")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Application Constants (Load from Environment Variables)
SALES_DATA_PATH = os.getenv("SALES_DATA_PATH", "sales_data.csv")
CUSTOMER_DATA_PATH = os.getenv("CUSTOMER_DATA_PATH", "customer_shopping_data.csv")
LOGO_PATH = os.getenv("LOGO_PATH", "psg_logo_blue.png")
WHATSAPP_NUMBER = os.getenv("WHATSAPP_NUMBER", "+919944934545")
EMAIL_FROM = os.getenv("EMAIL_FROM", "21z268@psgtech.ac.in")
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD", "")  # Optional
SLACK_BOT_TOKEN = os.environ["SLACK_BOT_TOKEN"]  # Required
SLACK_APP_TOKEN = os.environ["SLACK_APP_TOKEN"]  # Required
GENAI_API_KEY = os.environ["GENAI_API_KEY"]  # Required
TWILIO_ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN", "")
TWILIO_PHONE_NUMBER = os.environ.get("TWILIO_PHONE_NUMBER", "")
DEFAULT_LANGUAGE = os.getenv("DEFAULT_LANGUAGE", "English")

# Global Variables
user_states = {}
client = WebClient(token=SLACK_BOT_TOKEN)
model = genai.GenerativeModel("gemini-1.5-flash")
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
    "default": "Oops! Something went wrong. Try: register, purchase, weekly analysis, insights, simple insights, promotion, whatsapp, email, invoice"
}

# Messaging Functions
def send_whatsapp_message(phone_number, message):
    if not twilio_client:
        return FALLBACK_RESPONSES["whatsapp"] + " (Twilio not configured)"
    try:
        twilio_client.messages.create(
            body=message,
            from_=f"whatsapp:{TWILIO_PHONE_NUMBER}",
            to=f"whatsapp:{phone_number}"
        )
        logging.info(f"WhatsApp message sent to {phone_number}")
        return f"WhatsApp message sent to {phone_number} successfully!"
    except Exception as e:
        logging.error(f"Error sending WhatsApp message: {e}")
        return FALLBACK_RESPONSES["whatsapp"]

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
            # Rename columns to match sales_data.csv for consistency
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

# Customer Registration (Simplified without DB)
def handle_customer_registration(user_id, text):
    if user_id not in user_states:
        user_states[user_id] = {'step': 0}

    state = user_states[user_id]
    
    if state['step'] == 0:
        state['step'] = 1
        return "Please provide registration details in format:\n`register: name, email@example.com, +1234567890, English, 123 Street Name, City`"
    
    try:
        _, details = text.split("register:", 1)
        parts = [x.strip() for x in details.split(',', 5)]
        if len(parts) != 5:
            return "Invalid format. Need: name, email, phone, language, address"
        
        name, email, phone, language, address = parts
        
        email = re.sub(r'mailto:|\|.*$', '', email)
        phone = re.sub(r'tel:|\|.*$', '', phone)
        
        if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
            return "Invalid email format."
        if len(phone) > 20:
            return "Phone number too long."

        customer_id = str(uuid.uuid4())
        user_states[user_id] = {
            'customer_id': customer_id,
            'name': name,
            'email': email,
            'phone': phone,
            'language': language,
            'address': address
        }
        
        welcome_msg = translate_message(f"Welcome {name}! Registration successful.", language)
        whatsapp_response = send_whatsapp_message(phone, welcome_msg)
        return f"Registration successful! Customer ID: `{customer_id}`\n{whatsapp_response}"
    except Exception as e:
        logging.error(f"Registration error: {e}")
        return FALLBACK_RESPONSES["register"]

# Promotion Generation
def generate_promotion_image(prompt, channel):
    try:
        discount_percentage = "50"  # Default discount
        if " " in prompt:
            parts = prompt.split()
            for part in parts:
                if part.endswith("%"):
                    discount_percentage = part[:-1]
                    prompt = " ".join(p for p in parts if p != part)
                    break
        
        img_byte_arr = image_gen.generate_promotion_image(prompt, discount_percentage)
        if img_byte_arr:
            client.files_upload_v2(
                channels=channel,
                file=img_byte_arr,
                filename=f"promotion_{uuid.uuid4()[:8]}.png",
                title="Promotion Image"
            )
            return True
        return False
    except Exception as e:
        logging.error(f"Promotion image generation failed: {e}")
        return False

def generate_promotion(prompt, channel):
    try:
        text_msg = f"🚀 Promotion Alert! Get discount on {prompt}!"
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(generate_promotion_image, prompt, channel)
            if future.result():
                return f"{text_msg}\nPromotional image uploaded above!"
            return FALLBACK_RESPONSES["promotion"]
    except Exception as e:
        logging.error(f"Promotion generation failed: {e}")
        return FALLBACK_RESPONSES["promotion"]

# Weekly Sales Analysis
def generate_weekly_sales_analysis(channel):
    df = load_sales_data()
    if df is None:
        return FALLBACK_RESPONSES["weekly analysis"]
    
    try:
        weekly_sales = df.resample('W', on='Order Date')['Price Each'].sum()
        plt.figure(figsize=(12, 6))
        plt.plot(weekly_sales.index, weekly_sales.values, marker='o', linestyle='-', color='#4CAF50')
        plt.title('Weekly Sales Trend')
        plt.xlabel('Week')
        plt.ylabel('Sales (INR)')
        plt.grid(True)
        weekly_byte_arr = BytesIO()
        plt.savefig(weekly_byte_arr, format='PNG')
        weekly_byte_arr.seek(0)
        plt.close()

        client.files_upload_v2(channels=channel, file=weekly_byte_arr, filename="weekly_trend.png", title="Weekly Sales Trend")
        return "Weekly sales trend uploaded above!"
    except Exception as e:
        logging.error(f"Analysis error: {e}")
        return FALLBACK_RESPONSES["weekly analysis"]

# Multilingual Support
def translate_message(text, target_lang):
    try:
        response = model.generate_content(f"Translate this to {target_lang}: {text}")
        return response.text.strip()
    except Exception as e:
        logging.error(f"Translation error: {e}")
        return text

# Purchase Processing (Simplified without DB)
def process_purchase(product_id, user_id, channel):
    try:
        if user_id not in user_states or 'customer_id' not in user_states[user_id]:
            return "Please register first using: `register: name, email, phone, language, address`"
        
        user = user_states[user_id]
        df = load_sales_data()
        if df is None:
            return FALLBACK_RESPONSES["purchase"]
        
        product = df[df['Product'].str.lower() == product_id.lower()].iloc[0] if not df[df['Product'].str.lower() == product_id.lower()].empty else None
        if not product:
            return "Product not found."
        
        sale_id = str(uuid.uuid4())
        invoice_response = generate_invoice(user['customer_id'], {'product_name': product['Product'], 'price': product['Price Each']}, channel)
        message = f"Purchase confirmed: {product['Product']} for INR {product['Price Each']}"
        translated_msg = translate_message(message, user.get('language', DEFAULT_LANGUAGE))
        
        whatsapp_response = send_whatsapp_message(user['phone'], translated_msg)
        return f"Purchase completed! Sale ID: `{sale_id}`\n{invoice_response}\n{whatsapp_response}"
    except Exception as e:
        logging.error(f"Purchase error: {e}")
        return FALLBACK_RESPONSES["purchase"]

# Invoice Generation
def generate_invoice(customer_id=None, product=None, channel=None):
    class InvoicePDF(FPDF):
        def header(self):
            try:
                self.image(LOGO_PATH, 140, 8, 33)
            except:
                self.cell(0, 10, "Logo not found", align="R", ln=True)
            self.set_font("Arial", "B", 16)
            self.cell(0, 10, "INVOICE", align="C", ln=True)
            self.ln(10)

        def footer(self):
            self.set_y(-15)
            self.set_font("Arial", "I", 8)
            self.cell(0, 10, "Thank you for your business!", align="C")

        def add_company_and_client_info(self, company, client):
            self.set_font("Arial", "", 10)
            self.multi_cell(0, 10, company)
            self.ln(5)
            self.set_font("Arial", "B", 8)
            self.cell(0, 10, "INVOICE TO:", align="L")
            self.ln(5)
            self.set_font("Arial", "", 10)
            self.multi_cell(0, 10, client)
            self.ln(10)

        def add_table(self, header, data):
            self.set_fill_color(200, 220, 255)
            self.set_font("Arial", "B", 10)
            col_widths = [60, 30, 40, 40]
            for i, col in enumerate(header):
                self.cell(col_widths[i], 10, col, border=1, align="C", fill=True)
            self.ln()
            self.set_font("Arial", "", 10)
            for row in data:
                for i, col in enumerate(row):
                    self.cell(col_widths[i], 10, str(col), border=1, align="C")
                self.ln()

        def add_summary(self, summary):
            self.set_font("Arial", "B", 10)
            self.ln(10)
            for label, value in summary:
                self.cell(80, 10, label, border=0)
                self.cell(30, 10, value, border=0, align="R")
                self.ln()

    pdf = InvoicePDF()
    pdf.add_page()

    try:
        if customer_id and product:
            user = user_states.get(next((uid for uid, data in user_states.items() if data['customer_id'] == customer_id), None), {})
            company_info = "PSG College of Technology\nCoimbatore, Tamil Nadu\nPhone: 123-456-7890"
            client_info = f"{user.get('name', 'Unknown')}\n{user.get('address', 'N/A')}\n{user.get('email', 'N/A')}\n{user.get('phone', 'N/A')}"
            header = ["Product", "Qty", "Unit Price", "Total"]
            invoice_items = [[product['product_name'], 1, float(product['price']), float(product['price'])]]
            
            total_before_gst = float(product['price'])
            gst_rate = 0.18
            gst_amount = total_before_gst * gst_rate
            total_with_gst = total_before_gst + gst_amount
            
            pdf.add_company_and_client_info(company_info, client_info)
            pdf.add_table(header, invoice_items)
            pdf.add_summary([
                ("Subtotal", f"INR {total_before_gst:,.2f}"),
                ("GST (18%)", f"INR {gst_amount:,.2f}"),
                ("Total Amount", f"INR {total_with_gst:,.2f}")
            ])
            
            pdf_bytes = pdf.output(dest='S').encode('latin-1')
            pdf_byte_arr = BytesIO(pdf_bytes)
            pdf_byte_arr.seek(0)
            filename = f"invoice_{customer_id}_{uuid.uuid4()[:8]}.pdf"
        else:
            company_info = "PSG College of Technology\nCoimbatore, Tamil Nadu\nPhone: 123-456-7890"
            client_info = "Akil K\nCSE\nCoimbatore"
            header = ["Product", "Qty", "Unit Price", "Total"]
            invoice_items = [
                ["iPhone", 1, 700.00, 700.00],
                ["Lightning Charging Cable", 1, 14.95, 14.95]
            ]
            
            total_before_gst = sum(item[3] for item in invoice_items)
            gst_rate = 0.18
            gst_amount = total_before_gst * gst_rate
            total_with_gst = total_before_gst + gst_amount
            
            pdf.add_company_and_client_info(company_info, client_info)
            pdf.add_table(header, invoice_items)
            pdf.add_summary([
                ("Subtotal", f"INR {total_before_gst:,.2f}"),
                ("GST (18%)", f"INR {gst_amount:,.2f}"),
                ("Total Amount", f"INR {total_with_gst:,.2f}")
            ])
            
            pdf_bytes = pdf.output(dest='S').encode('latin-1')
            pdf_byte_arr = BytesIO(pdf_bytes)
            pdf_byte_arr.seek(0)
            filename = "invoice_dynamic.pdf"

        if channel:
            client.files_upload_v2(
                channels=channel,
                file=pdf_byte_arr,
                filename=filename,
                title="Invoice"
            )
        return "Invoice uploaded above!"
    except Exception as e:
        logging.error(f"Invoice generation error: {e}")
        return FALLBACK_RESPONSES["invoice"]

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

# Query Processing
def process_query(text, user_id, channel):
    text = text.lower().strip()
    try:
        if text.startswith("register:"):
            return handle_customer_registration(user_id, text)
        elif text.startswith("purchase:"):
            product_id = text.split(":")[1].strip()
            return process_purchase(product_id, user_id, channel)
        elif "weekly analysis" in text:
            return generate_weekly_sales_analysis(channel)
        elif "insights" in text or "simple insights" in text:
            return generate_sales_insights()
        elif "promotion" in text:
            prompt = text.replace("promotion:", "").strip()
            return generate_promotion(prompt, channel)
        elif "whatsapp" in text:
            message = "Hello, this is a test message from the bot!"
            if "in " in text:
                lang = text.split("in ")[-1].strip()
                message = translate_message(message, lang)
            return send_whatsapp_message(WHATSAPP_NUMBER, message)
        elif "email" in text:
            df = load_sales_data()
            if df is None:
                return FALLBACK_RESPONSES["email"]
            top_customers = df.groupby("Purchase Address")["Price Each"].sum().nlargest(3)
            subject = "Top Customers Report"
            body = f"Top 3 customers:\n{top_customers.to_string()}"
            if "in " in text:
                lang = text.split("in ")[-1].strip()
                body = translate_message(body, lang)
            return send_email(subject, body, [EMAIL_FROM], EMAIL_FROM, EMAIL_PASSWORD)
        elif "invoice" in text:
            return generate_invoice(channel=channel)
        else:
            return FALLBACK_RESPONSES["default"]
    except Exception as e:
        logging.error(f"Query processing error: {e}")
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
        response = process_query(text, user_id, channel)
        say(response)

    SocketModeHandler(app, SLACK_APP_TOKEN).start()
