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
from google.api_core import exceptions
import time
from pytrends.request import TrendReq
from fpdf import FPDF
from PIL import Image
import numpy as np
from scipy.stats import norm
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
import plotly.express as px
import datetime
import seaborn as sns
from googletrans import Translator
import requests
import html2text
from weasyprint import HTML

# Configuration and Constants
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN")
SLACK_APP_TOKEN = os.environ.get("SLACK_APP_TOKEN")
GENAI_API_KEY = os.environ.get("GENAI_API_KEY")
TWILIO_ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN", "")
TWILIO_PHONE_NUMBER = os.environ.get("TWILIO_PHONE_NUMBER", "")
EMAIL_FROM = os.environ.get("EMAIL_FROM", "your-email@example.com")
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD", "")
WHATSAPP_NUMBER = os.environ.get("WHATSAPP_NUMBER", "+1234567890")
DEFAULT_LANGUAGE = "English"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SALES_DATA_PATH = os.path.join(BASE_DIR, "sales_data.csv")
INVENTORY_PATH = os.path.join(BASE_DIR, "inventory.csv")
USERS_PATH = os.path.join(BASE_DIR, "users.csv")
LOGO_PATH = os.path.join(BASE_DIR, "smart_shoes_logo.png")
DEFAULT_PROMO_IMG = os.path.join(BASE_DIR, "promotion_image.png")

user_states = {}
client = WebClient(token=SLACK_BOT_TOKEN)
genai.configure(api_key=GENAI_API_KEY)
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN) if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN else None
pytrends = TrendReq(hl='en-US', tz=360, retries=2, backoff_factor=0.1)
app = FastAPI()
translator = Translator()

FALLBACK_RESPONSES = {
    "register": "Sorry, registration failed. Please try again later. 🙁",
    "purchase": "Purchase could not be processed. Please check back later. 🙁",
    "weekly analysis": "Weekly analysis unavailable. Data might be missing. 📉",
    "insights": "Insights generation failed. Please try again. 🙁",
    "insights_api_quota": "Sales insights unavailable due to API rate limits. Please wait and retry. ⏳",
    "insights_no_data": "Sales insights unavailable because no sales data was found. 📊",
    "promotion": "Promotion image generation failed. Try again later. 🙁",
    "promotion_no_image": "Couldn’t generate promotion image due to missing default image. 🖼️",
    "whatsapp": "Failed to send WhatsApp message. Please try again. 📱",
    "invoice": "Sorry, invoice generation failed. Please try again. 📄",
    "chart": "Chart generation failed. Please try again later. 📊",
    "chart_api_quota": "Chart generation failed due to API rate limits. Using default chart. 📉",
    "chart_no_data": "Chart generation failed because no sales data was found. 📊",
    "chart_invalid_query": "Chart generation failed due to an invalid query. Please specify a valid chart type. ❓",
    "default": "Oops! I didn’t understand that. Try: register, purchase, weekly analysis, insights, promotion, whatsapp, invoice, chart, or ask me anything! 🤔",
    "not_in_channel": "I can't respond here. Please invite me to this channel with `/invite @GrowBizz` or send me a DM! 🚪",
    "audio": "Audio processing failed. Please try again later. 🎙️"
}

# Cache for API results
trend_cache = {}
chart_cache = {}

# Helper Functions
def get_dm_channel(user_id):
    try:
        response = client.conversations_open(users=user_id)
        return response['channel']['id']
    except SlackApiError as e:
        logging.error(f"Failed to open DM channel for {user_id}: {e}")
        return None

def get_user_language(user_id):
    users_df = load_users()
    user = users_df[users_df['slack_id'] == user_id].iloc[0] if user_id in users_df['slack_id'].values else None
    return user['language'] if user else DEFAULT_LANGUAGE

def translate_message(text, target_lang):
    try:
        if target_lang.lower() == "english":
            return text
        translated = translator.translate(text, dest=target_lang.lower()).text
        return translated
    except Exception as e:
        logging.error(f"Translation error: {e}")
        return text

def send_whatsapp_message(user_id, message):
    if not twilio_client:
        logging.error("WhatsApp not configured: Twilio credentials missing.")
        return FALLBACK_RESPONSES["whatsapp"]
    if user_id not in user_states or 'phone' not in user_states[user_id]:
        logging.error(f"WhatsApp failed for {user_id}: User not registered.")
        return FALLBACK_RESPONSES["whatsapp"]
    phone = user_states[user_id]['phone']
    lang = user_states[user_id].get('language', DEFAULT_LANGUAGE)
    translated_msg = translate_message(message, lang)
    try:
        twilio_client.messages.create(
            body=translated_msg,
            from_=f"whatsapp:{TWILIO_PHONE_NUMBER}",
            to=f"whatsapp:{phone}"
        )
        return f"WhatsApp message sent to {phone}! 📱"
    except Exception as e:
        logging.error(f"WhatsApp error for {user_id}: {e}")
        return FALLBACK_RESPONSES["whatsapp"]

def send_whatsapp_media_message(user_id, message, media_url=None, media_path=None):
    if not twilio_client:
        logging.error("WhatsApp not configured: Twilio credentials missing.")
        return FALLBACK_RESPONSES["whatsapp"]
    if user_id not in user_states or 'phone' not in user_states[user_id]:
        logging.error(f"WhatsApp failed for {user_id}: User not registered.")
        return FALLBACK_RESPONSES["whatsapp"]
    phone = user_states[user_id]['phone']
    lang = user_states[user_id].get('language', DEFAULT_LANGUAGE)
    translated_msg = translate_message(message, lang)
    try:
        if media_path and os.path.exists(media_path):
            # Upload media to a temporary hosting service or use a public URL
            # For simplicity, assuming media_path is accessible; Twilio requires a URL
            media_url = media_url or "https://via.placeholder.com/600x400"  # Placeholder for demo
        twilio_client.messages.create(
            body=translated_msg,
            from_=f"whatsapp:{TWILIO_PHONE_NUMBER}",
            to=f"whatsapp:{ protips://media_url=media_url if media_url else None,
            media=media_url if media_url else None
        )
        return f"WhatsApp message sent to {phone}! 📱"
    except Exception as e:
        logging.error(f"WhatsApp media message error for {user_id}: {e}")
        return FALLBACK_RESPONSES["whatsapp"]

def send_email(user_id, subject, body, attachment=None):
    if user_id not in user_states or 'email' not in user_states[user_id]:
        logging.error(f"Email failed for {user_id}: User not registered.")
        return "Please register first. 📧"
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
        return f"Email sent to {email}! 📧"
    except Exception as e:
        logging.error(f"Email error for {user_id}: {e}")
        return "Failed to send email. 🙁"

# Audio Processing
def process_audio(audio_file_path, prompt, user_id):
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        with open(audio_file_path, 'rb') as audio_file:
            audio_data = audio_file.read()
        contents = [
            prompt,
            {"mime_type": "audio/mp3", "data": audio_data}
        ]
        response = model.generate_content(contents)
        return response.text.strip()
    except Exception as e:
        logging.error(f"Audio processing error for {user_id}: {e}")
        return FALLBACK_RESPONSES["audio"]

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
        df = pd.read_csv(USERS_PATH)
        if 'slack_id' not in df.columns:
            df['slack_id'] = ""
        return df
    else:
        df = pd.DataFrame(columns=["customer_id", "name", "email", "phone", "language", "address", "slack_id"])
        df.to_csv(USERS_PATH, index=False)
        return df

def load_sales_data():
    try:
        if os.path.exists(SALES_DATA_PATH):
            df = pd.read_csv(SALES_DATA_PATH)
            df['Price Each'] = pd.to_numeric(df['Price Each'], errors='coerce')
            df['Order Date'] = pd.to_datetime(df['Order Date'], format='%m/%d/%y %H:%M', errors='coerce').fillna(
                pd.to_datetime(df['Order Date'], format='%m-%d-%Y %H:%M', errors='coerce')
            )
            logging.info(f"Loaded sales data: {df.head().to_string()}")
            return df.dropna(subset=['Order Date', 'Price Each'])
        else:
            df = pd.DataFrame(columns=["Order ID", "Product", "Quantity Ordered", "Price Each", "Order Date", "Purchase Address"])
            df.to_csv(SALES_DATA_PATH, index=False)
            return df
    except Exception as e:
        logging.error(f"Error loading sales data: {e}")
        return pd.DataFrame()

def update_sales_data(product, quantity, price):
    df = load_sales_data()
    new_sale = pd.DataFrame({
        "Order ID": [int(df['Order ID'].max() + 1) if not df.empty else 141234],
        "Product": [product],
        "Quantity Ordered": [quantity],
        "Price Each": [price],
        "Order Date": [pd.Timestamp.now()],
        "Purchase Address": ["Unknown"]
    })
    df = pd.concat([df, new_sale], ignore_index=True)
    df.to_csv(SALES_DATA_PATH, index=False)

# Customer Registration
def handle_customer_registration(user_id, text):
    if user_id not in user_states:
        user_states[user_id] = {'last_message': '', 'context': 'idle'}
    state = user_states[user_id]
    if 'customer_id' in state:
        return f"**User is already registered** 😊\nYou are already registered with Customer ID: `{state['customer_id']}`."
    if "register:" in text.lower():
        try:
            _, details = text.lower().split("register:", 1)
            name, email, phone, language, address = [x.strip() for x in details.split(',', 4)]
            phone = re.sub(r'<tel:(\+\d+)\|.*>', r'\1', phone)
            phone = re.sub(r'[^0-9+]', '', phone)
            phone = '+' + phone if not phone.startswith('+') else phone
            email = re.sub(r'<mailto:([^|]+)\|.*>', r'\1', email)
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
            new_user = pd.DataFrame([{**user_states[user_id], 'slack_id': user_id}])
            users_df = pd.concat([users_df, new_user], ignore_index=True)
            users_df.to_csv(USERS_PATH, index=False)
            welcome_msg = f"Welcome to GrowBizz, {name}! Enjoy your shopping! 🛒\n📍 Address: {address}\n📞 Phone: {phone}"
            whatsapp_response = send_whatsapp_message(user_id, welcome_msg)
            return f"**Registration Successful** ✅\nCustomer ID: `{customer_id}`\n{whatsapp_response}"
        except Exception as e:
            logging.error(f"Registration error for {user_id}: {e}")
            return FALLBACK_RESPONSES["register"]
    return "Please use: `register: name, email, phone, language, address` 📝"

# Purchase Processing
def process_purchase(user_id, text):
    if user_id not in user_states or 'customer_id' not in user_states[user_id]:
        return "Please register first using: `register: name, email, phone, language, address` 📝"
    if "purchase:" not in text.lower():
        return "Please use: `purchase: product, quantity` 🛍️"
    try:
        _, details = text.lower().split("purchase:", 1)
        product, quantity = [x.strip() for x in details.split(',', 1)]
        quantity = int(quantity)
        inventory = load_inventory()
        if product not in inventory['Product'].values:
            return f"Product '{product}' not found in inventory. 🙁"
        stock = inventory[inventory['Product'] == product]['Stock'].iloc[0]
        if stock < quantity:
            return f"Insufficient stock for '{product}'. Available: {stock} 📦"
        price = inventory[inventory['Product'] == product]['Price'].iloc[0]
        update_inventory(product, quantity)
        update_sales_data(product, quantity, price)
        invoice_msg = generate_invoice(user_states[user_id]['customer_id'], {'product_name': product, 'price': price * quantity, 'quantity': quantity}, user_id)
        purchase_msg = f"Purchase confirmed: {quantity} x {product} for ₹{price * quantity:,.2f} 🛒"
        whatsapp_response = send_whatsapp_message(user_id, purchase_msg)
        email_response = send_email(user_id, "Purchase Confirmation", purchase_msg, "invoice_A35432.pdf" if os.path.exists("invoice_A35432.pdf") else None)
        return f"{purchase_msg}\n{invoice_msg}\n{whatsapp_response}\n{email_response}"
    except Exception as e:
        logging.error(f"Purchase error for {user_id}: {e}")
        return FALLBACK_RESPONSES["purchase"]

# Invoice Generation
def generate_invoice(customer_id=None, product=None, user_id=None):
    try:
        if customer_id and product and user_id:
            users_df = load_users()
            customer = users_df[users_df['customer_id'] == customer_id].iloc[0] if customer_id in users_df['customer_id'].values else None
            if not customer:
                return "Customer not found in users.csv. 🙁"

            # HTML template for invoice
            html_content = f"""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Shoe Shop Invoice</title>
                <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap" rel="stylesheet">
                <style>
                    body {{
                        font-family: 'Inter', sans-serif;
                        margin: 0;
                        padding: 0;
                        background-color: #f9f9f9;
                        color: #333;
                        line-height: 1.7;
                    }}
                    .container {{
                        max-width: 900px;
                        margin: 50px auto;
                        padding: 50px;
                        background-color: #fff;
                        border-radius: 15px;
                        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.1);
                    }}
                    header {{
                        text-align: center;
                        margin-bottom: 40px;
                        border-bottom: 2px solid #e0e0e0;
                        padding-bottom: 30px;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        flex-direction: column;
                    }}
                    header h1 {{
                        color: #2c3e50;
                        font-weight: 800;
                        margin: 0 0 10px 0;
                        font-size: 2.5em;
                        text-align: center;
                    }}
                    header p {{
                        font-size: 1.1em;
                        color: #7f8c8d;
                        margin-top: 0;
                        text-align: center;
                    }}
                    .invoice-logo {{
                        width: 120px;
                        height: 120px;
                        border-radius: 50%;
                        margin-bottom: 20px;
                        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
                        background-color: #ffffff;
                    }}
                    .invoice-details {{
                        margin-bottom: 40px;
                        display: flex;
                        justify-content: space-between;
                        font-size: 1.1em;
                    }}
                    .invoice-details .info, .invoice-details .date {{
                        text-align: left;
                    }}
                    .billing-shipping {{
                        display: flex;
                        justify-content: space-between;
                        margin-bottom: 40px;
                        font-size: 1.1em;
                    }}
                    .billing-shipping div {{
                        flex: 1;
                        text-align: left;
                        padding: 20px;
                        border-radius: 12px;
                        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.08);
                        background: linear-gradient(135deg, #ffffff, #f0f0f0);
                        border: 1px solid #dcdcdc;
                    }}
                    .billing-shipping strong {{
                        color: #3498db;
                    }}
                    .table-responsive {{
                        overflow-x: auto;
                        margin-bottom: 50px;
                    }}
                    table {{
                        width: 100%;
                        border-collapse: separate;
                        border-spacing: 0;
                        background-color: #fff;
                        margin-bottom: 30px;
                        border-radius: 12px;
                        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.05);
                        border: none;
                    }}
                    th, td {{
                        padding: 12px 18px;
                        text-align: left;
                        border-bottom: 1px solid #e0e0e0;
                    }}
                    th {{
                        background-color: #f0f0f0;
                        color: #2c3e50;
                        font-weight: 600;
                        text-transform: uppercase;
                        letter-spacing: 0.08em;
                        border-bottom: 2px solid #b0bec5;
                        font-size: 1.1em;
                    }}
                    td {{
                        font-size: 1.1em;
                        color: #555;
                    }}
                    tbody tr:last-child td {{
                        border-bottom: none;
                    }}
                    tfoot {{
                        text-align: right;
                        font-weight: bold;
                        font-size: 1.2em;
                        color: #2c3e50;
                    }}
                    .total-section {{
                        text-align: right;
                        margin-top: 50px;
                        font-size: 1.7em;
                        background: linear-gradient(to right, #FFB74D, #F57C00);
                        -webkit-background-clip: text;
                        color: transparent;
                    }}
                    .payment-terms, .comments {{
                        margin-top: 40px;
                        font-size: 0.95em;
                        border-top: 2px solid #dcdcdc;
                        padding-top: 25px;
                    }}
                    .payment-terms {{
                        color: #e74c3c;
                        font-weight: 600;
                    }}
                    .comments {{
                        color: #2c3e50;
                        font-style: italic;
                    }}
                    .signature-container {{
                        margin-top: 50px;
                        display: flex;
                        justify-content: center;
                        align-items: center;
                        flex-direction: column;
                    }}
                    .signature-line {{
                        border-bottom: 1px solid #90a4ae;
                        width: 300px;
                        margin-top: 10px;
                    }}
                    .insights-section {{
                        margin-top: 40px;
                        padding: 20px;
                        background-color: #f8f8f8;
                        border-radius: 10px;
                    }}
                    .insights-section h2 {{
                        color: #2c3e50;
                        font-size: 1.5em;
                        margin-bottom: 15px;
                    }}
                </style>
            </head>
            <body>
                <div class="container">
                    <header>
                        <img src="{LOGO_PATH}" class="invoice-logo">
                        <h1>Smart Shoes</h1>
                        <p>15, MG Road, Bengaluru, Karnataka, India</p>
                        <p>Phone: +91 9876543210 | Email: sales@smartshoes.in | Website: www.smartshoes.in</p>
                    </header>
                    <div class="invoice-details">
                        <div class="info">
                            <p>Invoice Number: A35432</p>
                        </div>
                        <div class="date">
                            <p>Date: April 04, 2025</p>
                        </div>
                    </div>
                    <div class="billing-shipping">
                        <div>
                            <p><strong>Bill To:</strong></p>
                            <p>Customer Name: {customer['name']}</p>
                            <p>Customer Address: {customer['address']}</p>
                        </div>
                        <div>
                            <p><strong>Ship To:</strong></p>
                            <p>Customer Name: {customer['name']}</p>
                            <p>Customer Address: {customer['address']}</p>
                        </div>
                    </div>
                    <div class="table-responsive">
                        <table>
                            <thead>
                                <tr>
                                    <th>Qty</th>
                                    <th>Product Description</th>
                                    <th>Size</th>
                                    <th>Unit Price</th>
                                    <th>Total</th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr>
                                    <td>{product['quantity']}</td>
                                    <td>{product['product_name']}</td>
                                    <td>10</td>
                                    <td>₹{product['price'] / product['quantity']:,.2f}</td>
                                    <td>₹{product['price']:,.2f}</td>
                                </tr>
                            </tbody>
                            <tfoot>
                                <tr>
                                    <td colspan="4">Subtotal:</td>
                                    <td>₹{product['price']:,.2f}</td>
                                </tr>
                                <tr>
                                    <td colspan="4">Tax (CGST 6% + SGST 6%):</td>
                                    <td>₹{product['price'] * 0.12:,.2f}</td>
                                </tr>
                                <tr>
                                    <td colspan="4">Shipping:</td>
                                    <td>₹500.00</td>
                                </tr>
                            </tfoot>
                        </table>
                    </div>
                    <div class="total-section">
                        <p>Total: ₹{product['price'] * 1.12 + 500:,.2f}</p>
                    </div>
                    <div class="payment-terms">
                        <p>Payment Terms: Net 30 days. Please make checks payable to Smart Shoes.</p>
                    </div>
                    <div class="comments">
                        <p>Comments: Thank you for your business! We appreciate your support and hope you enjoy your new shoes.</p>
                    </div>
                    <div class="insights-section">
                        <h2>Sales Insights</h2>
                        <p><strong>Total Sales This Month:</strong> [Calculated dynamically below]</p>
                        <p><strong>Average Daily Sales:</strong> [Calculated dynamically below]</p>
                        <p><strong>Best Selling Product:</strong> [Calculated dynamically below]</p>
                    </div>
                    <div class="signature-container">
                        <p>Shop Signature:</p>
                        <div class="signature-line">Smart Shoes</div>
                    </div>
                </div>
            </body>
            </html>
            """

            # Calculate insights for April
            df = load_sales_data()
            april_df = df[(df['Order Date'].dt.month == 4) & (df['Order Date'].dt.year == 2025)]
            total_sales = april_df["Price Each"].sum() if not april_df.empty else 0
            days_in_april = 30
            avg_daily_sales = total_sales / days_in_april if total_sales > 0 else 0
            best_selling_product = april_df.groupby("Product")["Quantity Ordered"].sum().idxmax() if not april_df.empty else "N/A"

            # Inject insights into HTML
            html_content = html_content.replace("[Calculated dynamically below]", 
                f"₹{total_sales:,.2f}<br>₹{avg_daily_sales:,.2f}<br>{best_selling_product}")

            # Convert HTML to PDF
            invoice_path = os.path.join(BASE_DIR, "invoice_A35432.pdf")
            HTML(string=html_content).write_pdf(invoice_path)

            # Send to WhatsApp
            whatsapp_msg = f"Hey {customer['name']} 👋, thank you for your latest purchase with Smart Shoes 👟. Here's your invoice against your order A35432. Visit us again. We have exciting discounts only for you! 🎉"
            whatsapp_response = send_whatsapp_media_message(user_id, whatsapp_msg, media_path=invoice_path)

            # Upload to DM
            dm_channel = get_dm_channel(user_id)
            if dm_channel:
                with open(invoice_path, 'rb') as f:
                    client.files_upload_v2(
                        channel=dm_channel,
                        file=f,
                        filename="invoice_A35432.pdf",
                        title="Smart Shoes Invoice"
                    )
                return f"**Invoice Generated** 📄\nInvoice sent to your DM as `invoice_A35432.pdf`!\n{whatsapp_response}"
            return f"Invoice generated at {invoice_path} but not uploaded (no DM channel). 🙁"
        else:
            return "Invalid parameters for invoice generation. 🙁"
    except Exception as e:
        logging.error(f"Invoice error for {user_id}: {e}")
        return FALLBACK_RESPONSES["invoice"]

# Promotion Generation
def generate_promotion(prompt, user_id, event_channel):
    try:
        if not os.path.exists(DEFAULT_PROMO_IMG):
            logging.error(f"Promotion image not found at {DEFAULT_PROMO_IMG} for {user_id}")
            return FALLBACK_RESPONSES["promotion_no_image"]
        try:
            with open(DEFAULT_PROMO_IMG, 'rb') as f:
                client.files_upload_v2(
                    channel=event_channel,
                    file=f,
                    filename="promotion_image.png",
                    title="Promotion Image"
                )
            logging.info(f"Promotion image uploaded to channel {event_channel} for {user_id}")
            return "**Promotion Generated** 🖼️\nPromotion image uploaded to this channel!"
        except SlackApiError as e:
            logging.error(f"Slack upload failed for {user_id}: {e}")
            return FALLBACK_RESPONSES["promotion"]
    except Exception as e:
        logging.error(f"Promotion generation failed for {user_id}: {e}")
        return FALLBACK_RESPONSES["promotion"]

# Chart Generation
def generate_chart(user_id, query):
    df = load_sales_data()
    if df.empty:
        logging.error(f"Chart failed for {user_id}: No sales data")
        return FALLBACK_RESPONSES["chart_no_data"]
    cache_key = f"{query}_{df.to_string()[:100]}"
    if cache_key in chart_cache:
        logging.info(f"Using cached chart code for {user_id}: {chart_cache[cache_key]}")
        code = chart_cache[cache_key]
    else:
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"""
        Generate Plotly Express code for: {query}. Use this data:
        {df.to_string()}
        Return just the code, use columns: 'Product', 'Quantity Ordered', 'Price Each', 'Order Date'.
        """
        retries = 3
        for attempt in range(retries):
            try:
                response = model.generate_content(prompt)
                code = response.text.strip()
                chart_cache[cache_key] = code
                logging.info(f"Generated Plotly code for {user_id}: {code}")
                break
            except exceptions.ResourceExhausted as e:
                if attempt < retries - 1:
                    wait_time = 22 * (2 ** attempt)
                    logging.warning(f"Gemini API quota exceeded for {user_id}, retrying in {wait_time}s: {e}")
                    time.sleep(wait_time)
                else:
                    logging.error(f"Gemini API quota exhausted for {user_id} after {retries} attempts: {e}")
                    code = "fig = px.bar(df, x='Product', y='Price Each', title='Sales by Product')"
                    chart_cache[cache_key] = code
                    logging.info(f"Using fallback chart code for {user_id}: {code}")
            except Exception as e:
                logging.error(f"Chart code generation failed for {user_id}: {e}")
                code = "fig = px.bar(df, x='Product', y='Price Each', title='Sales by Product')"
                chart_cache[cache_key] = code
                break
    try:
        local_vars = {"df": df, "px": px}
        exec(code, globals(), local_vars)
        fig = local_vars.get("fig")
        if not fig:
            logging.error(f"Chart execution failed for {user_id}: No figure generated")
            return FALLBACK_RESPONSES["chart"]
        img_byte_arr = BytesIO()
        fig.write_image(img_byte_arr, format="png", engine="kaleido")
        img_byte_arr.seek(0)
        dm_channel = get_dm_channel(user_id)
        if dm_channel:
            try:
                client.files_upload_v2(
                    channel=dm_channel,
                    file=img_byte_arr,
                    filename=f"chart_{uuid.uuid4().hex[:8]}.png",
                    title="Sales Chart"
                )
                logging.info(f"Chart uploaded to DM for {user_id}")
                return "**Chart Generated** 📊\nChart sent to your DM!"
            except SlackApiError as e:
                logging.error(f"Chart upload failed for {user_id}: {e}")
                return FALLBACK_RESPONSES["chart"]
        logging.warning(f"No DM channel for {user_id}")
        return FALLBACK_RESPONSES["chart"]
    except Exception as e:
        logging.error(f"Chart rendering/upload failed for {user_id}: {e}")
        return FALLBACK_RESPONSES["chart"]

# Weekly Sales Analysis
def generate_weekly_sales_analysis(user_id, event_channel):
    df = load_sales_data()
    if df.empty:
        logging.warning(f"Weekly analysis failed for {user_id}: No sales data.")
        return FALLBACK_RESPONSES["weekly analysis"]
    try:
        weekly_sales = df.resample('W', on='Order Date')['Price Each'].sum()
        total_sales = weekly_sales.sum()
        avg_weekly_sales = weekly_sales.mean()
        best_selling_product = df.groupby('Product')['Quantity Ordered'].sum().idxmax()
        
        plt.figure(figsize=(12, 6))
        plt.plot(weekly_sales.index, weekly_sales.values, marker='o', linestyle='-', color='#4CAF50')
        plt.title('Weekly Sales Trend', fontsize=14, fontweight='bold')
        plt.xlabel('Week', fontsize=12)
        plt.ylabel('Sales (₹)', fontsize=12)
        plt.grid(True)
        weekly_trend_file = os.path.join(BASE_DIR, f"weekly_trend_{uuid.uuid4().hex[:8]}.png")
        plt.savefig(weekly_trend_file)
        plt.close()

        insights = (
            f"**Weekly Sales Analysis** 📈\n"
            f"🔹 **Total Sales**: ₹{total_sales:,.2f}\n"
            f"🔹 **Average Weekly Sales**: ₹{avg_weekly_sales:,.2f}\n"
            f"🔹 **Best Selling Product**: {best_selling_product} 🔥\n"
            f"📊 The chart above shows the sales trend over recent weeks."
        )

        try:
            with open(weekly_trend_file, 'rb') as f:
                client.files_upload_v2(
                    channel=event_channel,
                    file=f,
                    filename=os.path.basename(weekly_trend_file),
                    title="Weekly Sales Trend"
                )
            os.remove(weekly_trend_file)
            return f"{insights}\n**Chart Uploaded** 📊\nWeekly sales trend uploaded to this channel!"
        except SlackApiError as e:
            logging.error(f"Weekly analysis upload failed for {user_id}: {e}")
            return f"{insights}\nFailed to upload chart. 🙁"
    except Exception as e:
        logging.error(f"Weekly analysis error for {user_id}: {e}")
        return FALLBACK_RESPONSES["weekly analysis"]

# Sales Insights and Recommendations
def generate_sales_insights(user_id=None):
    df = load_sales_data()
    inventory_df = load_inventory()
    if df.empty:
        logging.warning(f"Insights failed for {user_id}: No sales data.")
        return FALLBACK_RESPONSES["insights_no_data"]
    try:
        april_df = df[(df['Order Date'].dt.month == 4) & (df['Order Date'].dt.year == 2025)]
        if april_df.empty:
            return "**Sales Insights for April** 📊\nNo sales data available for April 2025. 🙁"

        total_sales = april_df["Price Each"].sum()
        days_in_april = 30
        weeks_in_april = 4.29  # Approx weeks
        avg_daily_sales = total_sales / days_in_april
        avg_weekly_sales = total_sales / weeks_in_april
        best_selling_product = april_df.groupby("Product")["Quantity Ordered"].sum().idxmax()

        trend_score = 0
        if user_id and best_selling_product in trend_cache:
            trend_score = trend_cache[best_selling_product]
            logging.info(f"Using cached trend score for {user_id}: {trend_score}")
        else:
            retries = 3
            for attempt in range(retries):
                try:
                    pytrends.build_payload(kw_list=[best_selling_product], timeframe='now 7-d')
                    trends = pytrends.interest_over_time()
                    trend_score = trends[best_selling_product].mean() / 100 if best_selling_product in trends else 0
                    if user_id:
                        trend_cache[best_selling_product] = trend_score
                    break
                except Exception as e:
                    if "429" in str(e) and attempt < retries - 1:
                        wait_time = 10 * (2 ** attempt)
                        logging.warning(f"Pytrends quota exceeded for {user_id}, retrying in {wait_time}s: {e}")
                        time.sleep(wait_time)
                    else:
                        logging.error(f"Pytrends failed for {user_id} after {retries} attempts: {e}")
                        trend_score = 0.00
                        break

        recommendations = {"decrease": [], "increase": [], "restock": []}
        if not inventory_df.empty:
            for _, product in inventory_df.iterrows():
                product_sales = april_df[april_df['Product'] == product['Product']]['Price Each'].sum()
                sales_score = product_sales / total_sales if total_sales > 0 else 0
                stock_score = 1 - (product['Stock'] / 100)
                weighted_score = 0.4 * sales_score + 0.3 * stock_score + 0.3 * trend_score

                if weighted_score < 0.3:
                    price_decrease = product['Price'] * 0.1
                    sales_increase = 0.15 * product_sales
                    recommendations["decrease"].append(
                        f"Decreasing price of {product['Product']} by ₹{price_decrease:,.2f} may increase sales by ₹{sales_increase:,.2f} 📉"
                    )
                elif weighted_score > 0.7:
                    price_increase = product['Price'] * 0.05
                    recommendations["increase"].append(
                        f"Increasing price of {product['Product']} by ₹{price_increase:,.2f} is recommended due to high demand 📈"
                    )
                if product['Stock'] < 10:
                    recommendations["restock"].append(
                        f"Restock {product['Product']} (Current: {product['Stock']}) 📦"
                    )

        insights = (
            f"**Sales Insights for April** 📊\n"
            f"🔹 **Total Sales**: ₹{total_sales:,.2f}\n"
            f"🔹 **Average Daily Sales**: ₹{avg_daily_sales:,.2f}\n"
            f"🔹 **Average Weekly Sales**: ₹{avg_weekly_sales:,.2f}\n"
            f"🔹 **Best Selling Product**: {best_selling_product} 🔥\n"
            f"🔹 **GrowBizz Trend Score**: {trend_score:.2f}\n\n"
        )
        if recommendations["decrease"] or recommendations["increase"] or recommendations["restock"]:
            insights += "**Recommendations** 📋\n"
            if recommendations["decrease"]:
                insights += "🔽 **Price Decreases**:\n" + "\n".join(recommendations["decrease"]) + "\n"
            if recommendations["increase"]:
                insights += "🔼 **Price Increases**:\n" + "\n".join(recommendations["increase"]) + "\n"
            if recommendations["restock"]:
                insights += "📦 **Restock**:\n" + "\n".join(recommendations["restock"]) + "\n"
        else:
            insights += "**Recommendations** 📋\nNo specific recommendations available. 🙁\n"
        logging.info(f"Generated insights for {user_id}: {insights}")
        return insights
    except Exception as e:
        logging.error(f"Insights error for {user_id}: {e}")
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
            return generate_weekly_sales_analysis(user_id, event_channel)
        elif "insights" in text or "sales insights" in text:
            return generate_sales_insights(user_id)
        elif "promotion:" in text or "generate promotion" in text or "promotion poster" in text:
            prompt = text.replace("promotion:", "").replace("generate promotion", "").replace("promotion poster", "").strip()
            return generate_promotion(prompt, user_id, event_channel)
        elif "whatsapp" in text or "send whatsapp message" in text:
            message = text.replace("whatsapp", "").replace("send whatsapp message", "").strip() or "Hello from GrowBizz! 😊"
            return send_whatsapp_message(user_id, message)
        elif "invoice" in text or "generate invoice" in text:
            return generate_invoice(None, None, user_id)
        elif "chart" in text:
            return generate_chart(user_id, text)
        elif "audio:" in text:
            audio_path = text.replace("audio:", "").strip()
            if os.path.exists(audio_path):
                prompt = "Transcribe and summarize the key points from this audio."
                return process_audio(audio_path, prompt, user_id)
            return "Audio file not found. Please upload a valid file. 🎙️"
        else:
            model = genai.GenerativeModel('gemini-1.5-flash')
            retries = 3
            for attempt in range(retries):
                try:
                    response = model.generate_content(f"Respond to this user query: {text}")
                    return response.text.strip()
                except exceptions.ResourceExhausted as e:
                    if attempt < retries - 1:
                        wait_time = 22 * (2 ** attempt)
                        logging.warning(f"Gemini API quota exceeded for {user_id}, retrying in {wait_time}s: {e}")
                        time.sleep(wait_time)
                    else:
                        logging.error(f"Default query failed for {user_id} due to API quota: {e}")
                        return FALLBACK_RESPONSES["default"]
                except Exception as e:
                    logging.error(f"Default query error for {user_id}: {e}")
                    return FALLBACK_RESPONSES["default"]
    except Exception as e:
        logging.error(f"Query processing error for {user_id}: {e}")
        return FALLBACK_RESPONSES["default"]

# FastAPI Endpoints
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

    def do_HEAD(self):
        self.send_response(200)
        self.send_header("Content-type", "text/plain")
        self.end_headers()

def run_http_server():
    server = HTTPServer(("", int(os.getenv("PORT", 3000))), DummyHandler)
    server.serve_forever()

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
                    logging.error(f"Failed to send DM for {user_id}: {dm_error}")
            else:
                logging.error(f"Slack API error for {user_id}: {e}")

    threading.Thread(target=run_http_server, daemon=True).start()
    SocketModeHandler(slack_app, SLACK_APP_TOKEN).start()
