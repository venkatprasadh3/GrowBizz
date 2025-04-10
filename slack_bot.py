import os
import re
import logging
import pandas as pd
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import uuid
from io import BytesIO
from twilio.rest import Client
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
import google.generativeai as genai
from fpdf import FPDF
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from scipy.stats import norm
from pytrends.request import TrendReq
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
import plotly.express as px
import datetime

# Configuration and Constants
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Environment Variables for Render
SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN")
SLACK_APP_TOKEN = os.environ.get("SLACK_APP_TOKEN")
GENAI_API_KEY = os.environ.get("GENAI_API_KEY", "your-default-api-key")
TWILIO_ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN", "")
TWILIO_PHONE_NUMBER = os.environ.get("TWILIO_PHONE_NUMBER", "")
EMAIL_FROM = os.environ.get("EMAIL_FROM", "your-email@example.com")
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD", "")
WHATSAPP_NUMBER = os.environ.get("WHATSAPP_NUMBER", "+1234567890")
DEFAULT_LANGUAGE = "English"

# File Paths Based on Project Structure
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SALES_DATA_PATH = os.path.join(BASE_DIR, "sales_data.csv")
INVENTORY_PATH = os.path.join(BASE_DIR, "inventory.csv")
USERS_PATH = os.path.join(BASE_DIR, "users.csv")
CUSTOMER_SHOPPING_DATA_PATH = os.path.join(BASE_DIR, "customer_shopping_data.csv")
LOGO_PATH = os.path.join(BASE_DIR, "psg_logo_blue.png")

# Global Variables
user_states = {}
client = WebClient(token=SLACK_BOT_TOKEN)
genai.configure(api_key=GENAI_API_KEY)
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN) if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN else None
pytrends = TrendReq(hl='en-US', tz=360)
app = FastAPI()

# Fallback Responses
FALLBACK_RESPONSES = {
    "register": "Sorry, registration failed. Please try again later.",
    "purchase": "Purchase could not be processed. Please check back later.",
    "weekly analysis": "Weekly analysis unavailable. Data might be missing.",
    "insights": "Insights generation failed. Please try again.",
    "promotion": "Promotion image generation failed. Try again later.",
    "whatsapp": "Failed to send WhatsApp message. Please try again.",
    "invoice": "Sorry, invoice generation failed. Please try again.",
    "default": "Oops! I didn’t understand that. Try: register, purchase, weekly analysis, insights, promotion, whatsapp, invoice, chart, or ask me anything!",
    "not_in_channel": "I can't respond here. Please invite me to this channel with `/invite @GrowBizz` or send me a DM!"
}

# Helper Functions
def get_dm_channel(user_id):
    try:
        response = client.conversations_open(users=user_id)
        return response['channel']['id']
    except SlackApiError as e:
        logging.error(f"Failed to open DM channel for {user_id}: {e}")
        return None

def translate_message(text, target_lang):
    try:
        response = genai.generate_text(prompt=f"Translate this to {target_lang}: {text}")
        return response.result.strip()
    except Exception as e:
        logging.error(f"Translation error: {e}")
        return text

def send_whatsapp_message(user_id, message):
    if not twilio_client:
        return "WhatsApp not configured."
    if user_id not in user_states or 'phone' not in user_states[user_id]:
        return "Please register first."
    phone = user_states[user_id]['phone']
    lang = user_states[user_id].get('language', DEFAULT_LANGUAGE)
    translated_msg = translate_message(message, lang)
    try:
        twilio_client.messages.create(
            body=translated_msg,
            from_=f"whatsapp:{TWILIO_PHONE_NUMBER}",
            to=f"whatsapp:{phone}"
        )
        return f"WhatsApp message sent to {phone}!"
    except Exception as e:
        logging.error(f"WhatsApp error: {e}")
        return FALLBACK_RESPONSES["whatsapp"]

def send_email(user_id, subject, body, attachment=None):
    if user_id not in user_states or 'email' not in user_states[user_id]:
        return "Please register first."
    email = user_states[user_id]['email']
    lang = user_states[user_id].get('language', DEFAULT_LANGUAGE)
    translated_body = translate_message(body, lang)
    msg = MIMEMultipart()
    msg['From'] = EMAIL_FROM
    msg['To'] = email
    msg['Subject'] = subject
    msg.attach(MIMEText(translated_body, 'plain'))
    if attachment and os.path.exists(attachment):
        with open(attachment, 'rb') as f:
            part = MIMEApplication(f.read(), Name=os.path.basename(attachment))
            part['Content-Disposition'] = f'attachment; filename="{os.path.basename(attachment)}"'
            msg.attach(part)
    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(EMAIL_FROM, EMAIL_PASSWORD)
            server.send_message(msg)
        return f"Email sent to {email}!"
    except Exception as e:
        logging.error(f"Email error: {e}")
        return "Failed to send email."

# Data Management
def load_inventory():
    if os.path.exists(INVENTORY_PATH):
        return pd.read_csv(INVENTORY_PATH)
    else:
        df = pd.DataFrame(columns=["Product", "Stock", "Price"])
        df.to_csv(INVENTORY_PATH, index=False)
        return df

def update_inventory(product, quantity):
    df = load_inventory()
    if product in df['Product'].values:
        df.loc[df['Product'] == product, 'Stock'] -= quantity
        df.to_csv(INVENTORY_PATH, index=False)
        return True
    return False

def load_users():
    if os.path.exists(USERS_PATH):
        return pd.read_csv(USERS_PATH)
    else:
        df = pd.DataFrame(columns=["customer_id", "name", "email", "phone", "language", "address"])
        df.to_csv(USERS_PATH, index=False)
        return df

def load_sales_data():
    if os.path.exists(SALES_DATA_PATH):
        df = pd.read_csv(SALES_DATA_PATH)
        df['Order Date'] = pd.to_datetime(df['Order Date'], format='%m/%d/%Y', errors='coerce')  # Fixed format
        return df
    else:
        df = pd.DataFrame(columns=["Order Date", "Product", "Quantity Ordered", "Price Each"])
        df.to_csv(SALES_DATA_PATH, index=False)
        return df

def update_sales_data(product, quantity, price):
    df = load_sales_data()
    new_sale = pd.DataFrame({
        "Order Date": [pd.Timestamp.now()],
        "Product": [product],
        "Quantity Ordered": [quantity],
        "Price Each": [price]
    })
    df = pd.concat([df, new_sale], ignore_index=True)
    df.to_csv(SALES_DATA_PATH, index=False)

# Customer Registration
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
            phone = re.sub(r'[^0-9+]', '', phone)
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
            users_df = load_users()
            new_user = pd.DataFrame([user_states[user_id]])
            users_df = pd.concat([users_df, new_user], ignore_index=True)
            users_df.to_csv(USERS_PATH, index=False)
            welcome_msg = f"Welcome {name} to GrowBizz!"
            whatsapp_response = send_whatsapp_message(user_id, welcome_msg)
            return f"Registration successful! Customer ID: `{customer_id}`\n{whatsapp_response}"
        except Exception as e:
            logging.error(f"Registration error: {e}")
            return FALLBACK_RESPONSES["register"]
    return "Please use: `register: name, email, phone, language, address`"

# Purchase Processing
def process_purchase(user_id, text):
    if user_id not in user_states or 'customer_id' not in user_states[user_id]:
        return "Please register first using: `register: name, email, phone, language, address`"
    if "purchase:" not in text.lower():
        return "Please use: `purchase: product, quantity`"
    try:
        _, details = text.lower().split("purchase:", 1)
        product, quantity = [x.strip() for x in details.split(',', 1)]
        quantity = int(quantity)
        inventory = load_inventory()
        if product not in inventory['Product'].values:
            return f"Product '{product}' not found in inventory."
        stock = inventory[inventory['Product'] == product]['Stock'].iloc[0]
        if stock < quantity:
            return f"Insufficient stock for '{product}'. Available: {stock}"
        price = inventory[inventory['Product'] == product]['Price'].iloc[0]
        update_inventory(product, quantity)
        update_sales_data(product, quantity, price)
        invoice_msg = generate_invoice(user_states[user_id]['customer_id'], {'product_name': product, 'price': price * quantity}, user_id)
        purchase_msg = f"Purchase confirmed: {quantity} x {product} for ${price * quantity}"
        whatsapp_response = send_whatsapp_message(user_id, purchase_msg)
        email_response = send_email(user_id, "Purchase Confirmation", purchase_msg, "invoice.pdf" if os.path.exists("invoice.pdf") else None)
        return f"{purchase_msg}\n{invoice_msg}\n{whatsapp_response}\n{email_response}"
    except Exception as e:
        logging.error(f"Purchase error: {e}")
        return FALLBACK_RESPONSES["purchase"]

# Invoice Generation (Integrated)
def generate_invoice(customer_id=None, product=None, user_id=None):
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
        if customer_id and product and user_id:
            users_df = load_users()
            customer = users_df[users_df['customer_id'] == customer_id].iloc[0] if customer_id in users_df['customer_id'].values else None
            if not customer:
                return "Customer not found in users.csv."

            company_info = "PSG College of Technology\nCoimbatore, Tamil Nadu\nPhone: 123-456-7890"
            client_info = f"{customer['name']}\n{customer['address']}\n{customer['email']}\n{customer['phone']}"
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

            invoice_path = os.path.join(BASE_DIR, f"invoice_{customer_id}_{uuid.uuid4().hex[:8]}.pdf")
        else:
            company_info = "PSG College of Technology\nCoimbatore, Tamil Nadu\nPhone: 123-456-7890"
            client_info = "Akil K\nCSE\nCoimbatore"
            header = ["Product", "Qty", "Unit Price", "Total"]
            invoice_items = [
                ["iPhone", 1, 700.00, 700.00],
                ["Lightning Charging Cable", 1, 14.95, 14.95],
                ["Wired Headphones", 2, 11.99, 23.98],
                ["27in FHD Monitor", 1, 149.99, 149.99],
                ["Wired Headphones", 1, 11.99, 11.99]
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

            invoice_path = os.path.join(BASE_DIR, "invoice_dynamic.pdf")

        pdf.output(invoice_path)
        dm_channel = get_dm_channel(user_id) if user_id else None
        if dm_channel:
            with open(invoice_path, 'rb') as f:
                client.files_upload_v2(
                    channel=dm_channel,
                    file=f,
                    filename=os.path.basename(invoice_path),
                    title="Invoice"
                )
            return f"Invoice generated and sent to your DM! Saved as {os.path.basename(invoice_path)}"
        return f"Invoice generated at {invoice_path} but not uploaded (no DM channel)."
    except Exception as e:
        logging.error(f"Invoice error: {e}")
        return FALLBACK_RESPONSES["invoice"]

# Promotion Generation
def generate_promotion(prompt, user_id):
    try:
        img = Image.new('RGB', (400, 200), color='white')
        d = ImageDraw.Draw(img)
        font = ImageFont.load_default()
        d.text((10, 10), f"Promotion: {prompt}", fill='black', font=font)
        promo_file = os.path.join(BASE_DIR, "promotion_image.png")
        img.save(promo_file)
        dm_channel = get_dm_channel(user_id)
        if dm_channel:
            with open(promo_file, 'rb') as f:
                client.files_upload_v2(
                    channel=dm_channel,
                    file=f,
                    filename="promotion_image.png",
                    title=f"Promotion: {prompt}"
                )
            return f"Promotion generated and sent to your DM!\nText: {prompt}"
        return "Failed to upload promotion image."
    except Exception as e:
        logging.error(f"Promotion error: {e}")
        return FALLBACK_RESPONSES["promotion"]

# Chart Generation
def generate_chart(user_id, query):
    df = load_sales_data()
    if "bar chart" in query.lower() and "sales by" in query.lower():
        fig = px.bar(df, x="Product", y="Price Each", title="Sales by Product", color="Product",
                     labels={"Price Each": "Total Sales ($)"}, height=500)
        img_byte_arr = BytesIO()
        fig.write_image(img_byte_arr, format="png")
        img_byte_arr.seek(0)
        dm_channel = get_dm_channel(user_id)
        if dm_channel:
            client.files_upload_v2(
                channel=dm_channel,
                file=img_byte_arr,
                filename=f"chart_{uuid.uuid4().hex[:8]}.png",
                title="Sales by Product"
            )
            return "Bar chart sent to your DM!"
        return "Failed to upload chart."
    return "Please specify a valid chart type, e.g., 'create a bar chart for the sales by shoe brand'."

# Weekly Sales Analysis
def generate_weekly_sales_analysis(user_id):
    df = load_sales_data()
    if df.empty:
        return FALLBACK_RESPONSES["weekly analysis"]
    
    # Weekly Sales Graph
    weekly_sales = df.resample('W', on='Order Date')['Price Each'].sum()
    fig1 = px.line(x=weekly_sales.index, y=weekly_sales.values, title="Weekly Sales Trend", labels={"y": "Sales ($)", "x": "Week"})
    img_byte_arr1 = BytesIO()
    fig1.write_image(img_byte_arr1, format="png")
    img_byte_arr1.seek(0)

    # Overall Sales Graph
    fig2 = px.line(x=df['Order Date'], y=df['Price Each'].cumsum(), title="Overall Sales Trend", labels={"y": "Cumulative Sales ($)", "x": "Date"})
    img_byte_arr2 = BytesIO()
    fig2.write_image(img_byte_arr2, format="png")
    img_byte_arr2.seek(0)

    # Normal Distribution Graph
    sales = df['Price Each']
    mu, sigma = sales.mean(), sales.std()
    x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
    fig3 = px.line(x=x, y=norm.pdf(x, mu, sigma), title="Sales Distribution", labels={"y": "Density", "x": "Sales ($)"})
    img_byte_arr3 = BytesIO()
    fig3.write_image(img_byte_arr3, format="png")
    img_byte_arr3.seek(0)

    dm_channel = get_dm_channel(user_id)
    if dm_channel:
        for i, img in enumerate([img_byte_arr1, img_byte_arr2, img_byte_arr3], 1):
            client.files_upload_v2(
                channel=dm_channel,
                file=img,
                filename=f"weekly_analysis_{i}_{uuid.uuid4().hex[:8]}.png",
                title=f"Weekly Analysis Graph {i}"
            )
        return "Weekly sales analysis (3 graphs) sent to your DM!"
    return "Failed to upload images."

# Sales Insights and Recommendations
def generate_sales_insights():
    df = load_sales_data()
    inventory = load_inventory()
    if df.empty or inventory.empty:
        return FALLBACK_RESPONSES["insights"]
    try:
        total_sales = df["Price Each"].sum()
        avg_sale = df["Price Each"].mean()
        best_selling = df.groupby("Product")["Quantity Ordered"].sum().idxmax()
        weekly_sales = df.resample('W', on='Order Date')['Price Each'].sum()
        trend = "increasing" if weekly_sales.diff().mean() > 0 else "decreasing"
        pytrends.build_payload([best_selling], timeframe='today 3-m')
        trends = pytrends.interest_over_time()
        seasonal_factor = trends[best_selling].mean() / 100 if best_selling in trends else 1

        recommendations = []
        for product in inventory['Product']:
            stock = inventory[inventory['Product'] == product]['Stock'].iloc[0]
            price = inventory[inventory['Product'] == product]['Price'].iloc[0]
            sales = df[df['Product'] == product]['Quantity Ordered'].sum()
            demand_rate = sales / len(weekly_sales) if sales > 0 else 0.1
            if stock < demand_rate * seasonal_factor:
                recommendations.append(f"Stock up {product} (current: {stock}, suggested: {int(demand_rate * seasonal_factor * 2)})")
            elif stock > demand_rate * 5:
                recommendations.append(f"Decrease price of {product} (current stock: {stock}, low demand)")
            elif trend == "increasing" and sales > avg_sale:
                recommendations.append(f"Increase price of {product} (high demand)")
        return f"📊 Sales Insights:\n🔹 Total Sales: ${total_sales:,.2f}\n🔹 Average Sale: ${avg_sale:,.2f}\n🔥 Best Selling: {best_selling}\n📈 Trend: {trend}\nRecommendations:\n" + "\n".join(recommendations)
    except Exception as e:
        logging.error(f"Insights error: {e}")
        return FALLBACK_RESPONSES["insights"]

# Query Processing
def process_query(text, user_id, event_channel):
    text = text.lower().strip()
    if user_id not in user_states:
        user_states[user_id] = {'last_message': '', 'context': 'idle'}
    state = user_states[user_id]
    state['last_message'] = text
    logging.info(f"Processing query: '{text}' from {user_id} in {event_channel}")
    try:
        if "register" in text:
            return handle_customer_registration(user_id, text)
        elif "purchase:" in text:
            return process_purchase(user_id, text)
        elif "weekly analysis" in text:
            return generate_weekly_sales_analysis(user_id)
        elif "insights" in text:
            return generate_sales_insights()
        elif "promotion:" in text:
            prompt = text.replace("promotion:", "").strip()
            return generate_promotion(prompt, user_id)
        elif "whatsapp" in text:
            message = text.replace("whatsapp", "").strip() or "Hello from GrowBizz!"
            return send_whatsapp_message(user_id, message)
        elif "invoice" in text or "generate invoice" in text:
            return generate_invoice(None, None, user_id)
        elif "chart" in text:
            return generate_chart(user_id, text)
        else:
            response = genai.generate_text(prompt=f"Respond to this user query: {text}")
            return response.result.strip()
    except Exception as e:
        logging.error(f"Query processing error: {e}")
        return FALLBACK_RESPONSES["default"]

# FastAPI Endpoints for Mobile Access
@app.post("/register")
async def api_register(request: Request):
    data = await request.json()
    text = data.get("text", "")
    user_id = data.get("user_id", "api_user")
    return JSONResponse({"response": handle_customer_registration(user_id, text)})

@app.post("/purchase")
async def api_purchase(request: Request):
    data = await request.json()
    text = data.get("text", "")
    user_id = data.get("user_id", "api_user")
    return JSONResponse({"response": process_purchase(user_id, text)})

@app.post("/query")
async def api_query(request: Request):
    data = await request.json()
    text = data.get("text", "")
    user_id = data.get("user_id", "api_user")
    return JSONResponse({"response": process_query(text, user_id, "api_channel")})

# HTTP Server for Render
class DummyHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/plain")
        self.end_headers()
        self.wfile.write(b"GrowBizz Slack bot is running")

def run_http_server():
    server = HTTPServer(("", int(os.getenv("PORT", 3000))), DummyHandler)
    server.serve_forever()

# Slack Bot Setup
if __name__ == "__main__":
    from slack_bolt import App as SlackApp
    from slack_bolt.adapter.socket_mode import SocketModeHandler
    slack_app = SlackApp(token=SLACK_BOT_TOKEN)

    @slack_app.message(".*")
    def handle_message(event, say):
        user_id = event['user']
        text = event['text']
        channel = event['channel']
        response = process_query(text, user_id, channel)
        try:
            say(response)
        except SlackApiError as e:
            if e.response["error"] == "not_in_channel":
                try:
                    client.chat_postMessage(channel=user_id, text=FALLBACK_RESPONSES["not_in_channel"])
                except SlackApiError as dm_error:
                    logging.error(f"Failed to send DM: {dm_error}")
            else:
                logging.error(f"Slack API error: {e}")

    threading.Thread(target=run_http_server, daemon=True).start()
    SocketModeHandler(slack_app, SLACK_APP_TOKEN).start()
