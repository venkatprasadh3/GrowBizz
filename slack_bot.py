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
import numpy as np
from scipy.stats import norm
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
import plotly.express as px
import datetime
import seaborn as sns
from googletrans import Translator
import cloudinary
import cloudinary.uploader
import json

# Configuration and Constants
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN")
SLACK_APP_TOKEN = os.environ.get("SLACK_APP_TOKEN")
GENAI_API_KEY = os.environ.get("GENAI_API_KEY")
TWILIO_ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN", "")
TWILIO_PHONE_NUMBER = os.environ.get("TWILIO_PHONE_NUMBER", "")
CLOUDINARY_CLOUD_NAME = os.environ.get("CLOUDINARY_CLOUD_NAME")
CLOUDINARY_API_KEY = os.environ.get("CLOUDINARY_API_KEY")
CLOUDINARY_API_SECRET = os.environ.get("CLOUDINARY_API_SECRET")
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
cloudinary.config(
    cloud_name=CLOUDINARY_CLOUD_NAME,
    api_key=CLOUDINARY_API_KEY,
    api_secret=CLOUDINARY_API_SECRET,
    secure=True
) if all([CLOUDINARY_CLOUD_NAME, CLOUDINARY_API_KEY, CLOUDINARY_API_SECRET]) else None
pytrends = TrendReq(hl='en-US', tz=360, retries=2, backoff_factor=0.1)
translator = Translator()
processed_events = set()
response_cache = {}
file_url_cache = {}

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
    "default": "Oops! I didn’t understand that. Try: register, purchase, weekly analysis, insights, promotion, whatsapp, invoice, chart, summarize call, bengali voice, or ask me anything! 🤔",
    "not_in_channel": "I can't respond here. Please invite me to this channel with `/invite @GrowBizz` or send me a DM! 🚪",
    "audio": "Audio processing failed. Please try again later. 🎙️",
    "whatsapp_media": "Failed to send media via WhatsApp. Please try again. 📱"
}

trend_cache = {}
chart_cache = {}

# Helper Functions
def get_user_language(user_id):
    users_df = load_users()
    user = users_df[users_df['slack_id'] == user_id].iloc[0] if not users_df[users_df['slack_id'] == user_id].empty else None
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

def upload_to_cloudinary(file_path=None, file_bytes=None, public_id=None, resource_type="auto"):
    try:
        if file_path and os.path.exists(file_path):
            upload_result = cloudinary.uploader.upload(
                file_path,
                resource_type=resource_type,
                public_id=public_id or f"growbizz_{uuid.uuid4().hex[:8]}"
            )
        elif file_bytes:
            upload_result = cloudinary.uploader.upload(
                file_bytes,
                resource_type=resource_type,
                public_id=public_id or f"growbizz_{uuid.uuid4().hex[:8]}"
            )
        else:
            raise ValueError("No file or bytes provided")
        url = upload_result["secure_url"]
        file_url_cache[file_path or public_id] = url
        logging.info(f"File uploaded to Cloudinary: {url}")
        return url
    except Exception as e:
        logging.error(f"Cloudinary upload failed: {e}")
        return None

def send_whatsapp_message(user_id, message):
    if not twilio_client:
        logging.error("WhatsApp not configured: Twilio credentials missing.")
        return FALLBACK_RESPONSES["whatsapp"]
    if user_id not in user_states or 'phone' not in user_states[user_id]:
        logging.error(f"WhatsApp failed for {user_id}: User not registered.")
        return FALLBACK_RESPONSES["whatsapp"]
    phone = user_states[user_id]['phone']
    if not re.match(r'^\+\d{10,15}$', phone):
        logging.error(f"Invalid phone number for {user_id}: {phone}")
        return "Invalid phone number format. Please register with a valid number. 📱"
    lang = user_states[user_id].get('language', DEFAULT_LANGUAGE)
    translated_msg = translate_message(message, lang)
    try:
        response = twilio_client.messages.create(
            body=translated_msg,
            from_=f"whatsapp:{TWILIO_PHONE_NUMBER}",
            to=f"whatsapp:{phone}"
        )
        logging.info(f"WhatsApp message SID: {response.sid}, Status: {response.status}, To: {phone}")
        if response.status in ['queued', 'sent', 'delivered']:
            return f"WhatsApp message sent to {phone}! 📱"
        else:
            logging.warning(f"WhatsApp message to {phone} failed with status: {response.status}")
            return "WhatsApp message queued but may not have been delivered. Please check the number. 📱"
    except Exception as e:
        logging.error(f"WhatsApp error for {user_id}: {e}")
        return f"Failed to send WhatsApp message: {str(e)}. 🙁"

def send_whatsapp_media_message(user_id, message, media_url=None, media_path=None):
    if not twilio_client:
        logging.error("WhatsApp not configured: Twilio credentials missing.")
        return FALLBACK_RESPONSES["whatsapp"]
    if user_id not in user_states or 'phone' not in user_states[user_id]:
        logging.error(f"WhatsApp failed for {user_id}: User not registered.")
        return FALLBACK_RESPONSES["whatsapp"]
    phone = user_states[user_id]['phone']
    if not re.match(r'^\+\d{10,15}$', phone):
        logging.error(f"Invalid phone number for {user_id}: {phone}")
        return "Invalid phone number format. Please register with a valid number. 📱"
    lang = user_states[user_id].get('language', DEFAULT_LANGUAGE)
    translated_msg = translate_message(message, lang)
    try:
        if media_path and os.path.exists(media_path):
            media_url = upload_to_cloudinary(file_path=media_path, resource_type="raw")
        if not media_url:
            media_url = "https://via.placeholder.com/600x400"
        response = twilio_client.messages.create(
            body=translated_msg,
            from_=f"whatsapp:{TWILIO_PHONE_NUMBER}",
            to=f"whatsapp:{phone}",
            media_url=[media_url]
        )
        logging.info(f"WhatsApp media message SID: {response.sid}, Status: {response.status}, To: {phone}")
        return f"WhatsApp message with media sent to {phone}! 📱"
    except Exception as e:
        logging.error(f"WhatsApp media message error for {user_id}: {e}")
        return FALLBACK_RESPONSES["whatsapp_media"]

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
        logging.info(f"Email sent to {email} for {user_id}")
        return f"Email sent to {email}! 📧"
    except Exception as e:
        logging.error(f"Email error for {user_id}: {e}")
        return "Failed to send email. 🙁"

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
        return {
            "blocks": [
                {"type": "header", "text": {"type": "plain_text", "text": "Registration Status 📝"}},
                {"type": "section", "text": {"type": "mrkdwn", "text": f"*You are already registered* with Customer ID: `{state['customer_id']}`."}}
            ],
            "text": "User is already registered."
        }
    if "register:" in text.lower():
        try:
            _, details = text.lower().split("register:", 1)
            name, email, phone, language, address = [x.strip() for x in details.split(',', 4)]
            phone = re.sub(r'<tel:(\+\d+)\|.*>', r'\1', phone)
            phone = re.sub(r'[^0-9+]', '', phone)
            phone = '+' + phone if not phone.startswith('+') else phone
            email = re.sub(r'<mailto:([^|]+)\|.*>', r'\1', email)
            if not re.match(r'^\+\d{10,15}$', phone):
                raise ValueError("Invalid phone number format")
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
            welcome_msg = f"Welcome to GrowBizz, {name}!\nEnjoy your shopping! 🛒\n📍 Address: {address}\n📞 Phone: {phone}"
            whatsapp_response = send_whatsapp_message(user_id, welcome_msg)
            response = {
                "blocks": [
                    {"type": "header", "text": {"type": "plain_text", "text": "Registration Successful ✅"}},
                    {"type": "section", "text": {"type": "mrkdwn", "text": f"Customer ID: `{customer_id}`\n{whatsapp_response}"}}
                ],
                "text": f"Registered {name} with ID {customer_id}"
            }
            logging.info(f"Registration successful for {user_id}: {response}")
            return response
        except Exception as e:
            logging.error(f"Registration error for {user_id}: {e}")
            return {
                "blocks": [
                    {"type": "header", "text": {"type": "plain_text", "text": "Registration Failed 🙁"}},
                    {"type": "section", "text": {"type": "mrkdwn", "text": FALLBACK_RESPONSES["register"]}}
                ],
                "text": "Registration failed."
            }
    return {
        "blocks": [
            {"type": "header", "text": {"type": "plain_text", "text": "Registration Help 📝"}},
            {"type": "section", "text": {"type": "mrkdwn", "text": "Please use: `register: name, email, phone, language, address`"}}
        ],
        "text": "Registration help provided."
    }

# Purchase Processing
def process_purchase(user_id, text):
    if user_id not in user_states or 'customer_id' not in user_states[user_id]:
        return {
            "blocks": [
                {"type": "header", "text": {"type": "plain_text", "text": "Purchase Failed 🙁"}},
                {"type": "section", "text": {"type": "mrkdwn", "text": "Please register first using: `register: name, email, phone, language, address` 📝"}}
            ],
            "text": "Purchase failed: not registered."
        }
    if "purchase:" not in text.lower():
        return {
            "blocks": [
                {"type": "header", "text": {"type": "plain_text", "text": "Purchase Help 🛍️"}},
                {"type": "section", "text": {"type": "mrkdwn", "text": "Please use: `purchase: product, quantity`"}}
            ],
            "text": "Purchase help provided."
        }
    try:
        _, details = text.lower().split("purchase:", 1)
        product, quantity = [x.strip() for x in details.split(',', 1)]
        quantity = int(quantity)
        inventory = load_inventory()
        if product not in inventory['Product'].values:
            return {
                "blocks": [
                    {"type": "header", "text": {"type": "plain_text", "text": "Purchase Failed 🙁"}},
                    {"type": "section", "text": {"type": "mrkdwn", "text": f"Product '{product}' not found in inventory."}}
                ],
                "text": f"Purchase failed: {product} not found."
            }
        stock = inventory[inventory['Product'] == product]['Stock'].iloc[0]
        if stock < quantity:
            return {
                "blocks": [
                    {"type": "header", "text": {"type": "plain_text", "text": "Purchase Failed 🙁"}},
                    {"type": "section", "text": {"type": "mrkdwn", "text": f"Insufficient stock for '{product}'. Available: {stock} 📦"}}
                ],
                "text": f"Purchase failed: insufficient stock for {product}."
            }
        price = inventory[inventory['Product'] == product]['Price'].iloc[0]
        update_inventory(product, quantity)
        update_sales_data(product, quantity, price)
        invoice_msg = generate_invoice(user_states[user_id]['customer_id'], {'product_name': product, 'price': price * quantity, 'quantity': quantity}, user_id, None)
        purchase_msg = f"Purchase confirmed: {quantity} x {product} for ₹{price * quantity:,.2f} 🛒"
        whatsapp_response = send_whatsapp_message(user_id, purchase_msg)
        email_response = send_email(user_id, "Purchase Confirmation", purchase_msg, "/tmp/invoice_A35432.html" if os.path.exists("/tmp/invoice_A35432.html") else None)
        response = {
            "blocks": [
                {"type": "header", "text": {"type": "plain_text", "text": "Purchase Confirmed 🛒"}},
                {"type": "section", "text": {"type": "mrkdwn", "text": f"{purchase_msg}\n{invoice_msg['blocks'][1]['text']['text']}\n{whatsapp_response}\n{email_response}"}}
            ],
            "text": f"Purchase confirmed: {quantity} x {product}"
        }
        logging.info(f"Purchase successful for {user_id}: {response}")
        return response
    except Exception as e:
        logging.error(f"Purchase error for {user_id}: {e}")
        return {
            "blocks": [
                {"type": "header", "text": {"type": "plain_text", "text": "Purchase Failed 🙁"}},
                {"type": "section", "text": {"type": "mrkdwn", "text": FALLBACK_RESPONSES["purchase"]}}
            ],
            "text": "Purchase failed."
        }

# Invoice Generation
def generate_invoice(customer_id=None, product=None, user_id=None, event_channel=None):
    try:
        if not user_id:
            return {
                "blocks": [
                    {"type": "header", "text": {"type": "plain_text", "text": "Invoice Generation 📄"}},
                    {"type": "section", "text": {"type": "mrkdwn", "text": "User ID required. 🙁"}}
                ],
                "text": "Invoice failed: missing user ID."
            }
        users_df = load_users()
        if users_df[users_df['slack_id'] == user_id].empty:
            return {
                "blocks": [
                    {"type": "header", "text": {"type": "plain_text", "text": "Invoice Generation 📄"}},
                    {"type": "section", "text": {"type": "mrkdwn", "text": "Please register first. 🙁"}}
                ],
                "text": "Invoice failed: user not registered."
            }
        customer = users_df[users_df['slack_id'] == user_id].iloc[0]
        product = product or {'product_name': 'Sample Product', 'price': 1000.00, 'quantity': 1}
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head><title>Invoice</title></head>
        <body>
            <h1>Smart Shoes Invoice</h1>
            <p>Invoice Number: A35432</p>
            <p>Date: {datetime.datetime.now().strftime('%Y-%m-%d')}</p>
            <p>Bill To: {customer['name']}</p>
            <table border="1">
                <tr><th>Qty</th><th>Product</th><th>Price</th></tr>
                <tr><td>{product['quantity']}</td><td>{product['product_name']}</td><td>₹{product['price']:,.2f}</td></tr>
            </table>
            <p>Total: ₹{product['price'] * product['quantity'] * 1.12:,.2f}</p>
        </body>
        </html>
        """
        invoice_path = "/tmp/invoice_A35432.html"
        with open(invoice_path, 'w') as f:
            f.write(html_content)
        public_id = f"invoice_A35432_{uuid.uuid4().hex[:8]}"
        invoice_url = upload_to_cloudinary(file_path=invoice_path, public_id=public_id, resource_type="raw")
        if not invoice_url:
            raise Exception("Failed to upload invoice to Cloudinary")
        if event_channel:
            with open(invoice_path, 'rb') as f:
                client.files_upload_v2(
                    channel=event_channel,
                    file=f,
                    filename="invoice_A35432.html",
                    title="Invoice"
                )
        if os.path.exists(invoice_path):
            os.remove(invoice_path)
        return {
            "blocks": [
                {"type": "header", "text": {"type": "plain_text", "text": "Invoice Generated 📄"}},
                {"type": "section", "text": {"type": "mrkdwn", "text": f"Invoice uploaded to this channel!\nURL: {invoice_url}"}}
            ],
            "text": "Invoice generated."
        }
    except Exception as e:
        logging.error(f"Invoice error for {user_id}: {e}")
        return {
            "blocks": [
                {"type": "header", "text": {"type": "plain_text", "text": "Invoice Generation 📄"}},
                {"type": "section", "text": {"type": "mrkdwn", "text": FALLBACK_RESPONSES["invoice"]}}
            ],
            "text": "Invoice generation failed."
        }

# Promotion Generation
def generate_promotion(prompt, user_id, event_channel):
    try:
        if not os.path.exists(DEFAULT_PROMO_IMG):
            logging.error(f"Promotion image not found at {DEFAULT_PROMO_IMG} for {user_id}")
            return {
                "blocks": [
                    {"type": "header", "text": {"type": "plain_text", "text": "Promotion Generation 🖼️"}},
                    {"type": "section", "text": {"type": "mrkdwn", "text": FALLBACK_RESPONSES["promotion_no_image"]}}
                ],
                "text": "Promotion failed: no image."
            }
        context = "Promotion poster for your shoe shop" if not prompt else f"Promotion poster for {prompt}"
        with threading.Lock():
            with open(DEFAULT_PROMO_IMG, 'rb') as f:
                client.files_upload_v2(
                    channel=event_channel,
                    file=f,
                    filename="promotion_image.png",
                    title="Smart Shoes Promotion Poster",
                    initial_comment=context
                )
        logging.info(f"Promotion image uploaded to channel {event_channel} for {user_id}")
        return {
            "blocks": [],
            "text": context
        }
    except Exception as e:
        logging.error(f"Promotion generation failed for {user_id}: {e}")
        return {
            "blocks": [
                {"type": "header", "text": {"type": "plain_text", "text": "Promotion Generation 🖼️"}},
                {"type": "section", "text": {"type": "mrkdwn", "text": FALLBACK_RESPONSES["promotion"]}}
            ],
            "text": "Promotion generation failed."
        }

# Chart Generation
def generate_chart(user_id, query, event_channel):
    df = load_sales_data()
    if df.empty:
        logging.error(f"Chart failed for {user_id}: No sales data")
        return {
            "blocks": [
                {"type": "header", "text": {"type": "plain_text", "text": "Chart Generation 📊"}},
                {"type": "section", "text": {"type": "mrkdwn", "text": FALLBACK_RESPONSES["chart_no_data"]}}
            ],
            "text": "Chart failed: no sales data."
        }
    if "bar chart for the sales by product" in query.lower():
        try:
            fig = px.bar(
                df.groupby('Product')['Price Each'].sum().reset_index(),
                x='Product',
                y='Price Each',
                title='Sales by Product',
                labels={'Price Each': 'Total Sales (₹)'}
            )
            img_byte_arr = BytesIO()
            fig.write_image(img_byte_arr, format="png", engine="kaleido", width=800, height=600)
            img_byte_arr.seek(0)
            with threading.Lock():
                client.files_upload_v2(
                    channel=event_channel,
                    file=img_byte_arr,
                    filename=f"chart_{uuid.uuid4().hex[:8]}.png",
                    title="Sales by Product Bar Chart",
                    initial_comment="Bar chart showing total sales by product."
                )
            logging.info(f"Bar chart uploaded to channel {event_channel} for {user_id}")
            return {
                "blocks": [
                    {"type": "header", "text": {"type": "plain_text", "text": "Chart Generated 📊"}},
                    {"type": "section", "text": {"type": "mrkdwn", "text": "Bar chart uploaded to this channel!"}}
                ],
                "text": "Bar chart uploaded."
            }
        except Exception as e:
            logging.error(f"Plotly bar chart failed for {user_id}: {e}")
            return generate_fallback_chart(df, user_id, event_channel, chart_type="bar")
    return generate_fallback_chart(df, user_id, event_channel, chart_type="bar")

def generate_fallback_chart(df, user_id, event_channel, chart_type="bar"):
    try:
        plt.figure(figsize=(10, 6))
        product_sales = df.groupby('Product')['Price Each'].sum()
        if chart_type == "bar":
            product_sales.plot(kind='bar', color='skyblue')
            plt.title('Sales by Product')
            plt.xlabel('Product')
            plt.ylabel('Total Sales (₹)')
            plt.xticks(rotation=45, ha='right')
        else:
            plt.plot(product_sales.index, product_sales.values, marker='o', linestyle='-', color='skyblue')
            plt.title('Sales Trend by Product')
            plt.xlabel('Product')
            plt.ylabel('Total Sales (₹)')
            plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        img_byte_arr = BytesIO()
        plt.savefig(img_byte_arr, format='png')
        plt.close()
        img_byte_arr.seek(0)
        with threading.Lock():
            client.files_upload_v2(
                channel=event_channel,
                file=img_byte_arr,
                filename=f"fallback_chart_{uuid.uuid4().hex[:8]}.png",
                title="Fallback Sales Chart",
                initial_comment="Fallback chart for sales by product."
            )
        logging.info(f"Fallback chart uploaded to channel {event_channel} for {user_id}")
        return {
            "blocks": [
                {"type": "header", "text": {"type": "plain_text", "text": "Chart Generated 📊"}},
                {"type": "section", "text": {"type": "mrkdwn", "text": "Chart uploaded to this channel (using fallback method)!"}}
            ],
            "text": "Fallback chart uploaded."
        }
    except Exception as e:
        logging.error(f"Fallback chart failed for {user_id}: {e}")
        return {
            "blocks": [
                {"type": "header", "text": {"type": "plain_text", "text": "Chart Generation 📊"}},
                {"type": "section", "text": {"type": "mrkdwn", "text": FALLBACK_RESPONSES["chart"]}}
            ],
            "text": "Chart generation failed."
        }

# Weekly Sales Analysis
def generate_weekly_sales_analysis(user_id, event_channel):
    df = load_sales_data()
    if df.empty:
        logging.warning(f"Weekly analysis failed for {user_id}: No sales data.")
        return {
            "blocks": [
                {"type": "header", "text": {"type": "plain_text", "text": "Weekly Sales Analysis 📈"}},
                {"type": "section", "text": {"type": "mrkdwn", "text": FALLBACK_RESPONSES["weekly analysis"]}}
            ],
            "text": "Weekly analysis failed: no sales data."
        }
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

        plt.figure(figsize=(12, 6))
        plt.plot(df['Order Date'], df['Price Each'].cumsum(), color='#2196F3')
        plt.title('Overall Sales Trend', fontsize=14, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Cumulative Sales (₹)', fontsize=12)
        plt.grid(True)
        overall_trend_file = os.path.join(BASE_DIR, f"overall_trend_{uuid.uuid4().hex[:8]}.png")
        plt.savefig(overall_trend_file)
        plt.close()

        mu, std = norm.fit(df['Price Each'].dropna())
        plt.figure(figsize=(10, 6))
        sns.histplot(df['Price Each'], kde=True, stat="density", color='#2196F3')
        x = np.linspace(df['Price Each'].min(), df['Price Each'].max(), 100)
        plt.plot(x, norm.pdf(x, mu, std), 'r-', lw=2, label=f'Normal fit (μ={mu:.2f}, σ={std:.2f})')
        plt.title('Sales Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('Sales Amount (₹)', fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.legend()
        sales_dist_file = os.path.join(BASE_DIR, f"sales_distribution_{uuid.uuid4().hex[:8]}.png")
        plt.savefig(sales_dist_file)
        plt.close()

        insights = (
            f"*Total Sales*: ₹{total_sales:,.2f}\n"
            f"*Average Weekly Sales*: ₹{avg_weekly_sales:,.2f}\n"
            f"*Best Selling Product*: {best_selling_product} 🔥\n\n"
            f"*Weekly Sales Trend*: Shows sales fluctuations week by week. 📈\n"
            f"*Overall Sales Trend*: Tracks total sales growth over time. 📊\n"
            f"*Sales Distribution*: Displays the spread of sale amounts with a normal fit. 📉"
        )

        for i, file in enumerate([weekly_trend_file, overall_trend_file, sales_dist_file], 1):
            with open(file, 'rb') as f:
                client.files_upload_v2(
                    channel=event_channel,
                    file=f,
                    filename=os.path.basename(file),
                    title=f"Graph {i}: {'Weekly Sales Trend' if i == 1 else 'Overall Sales Trend' if i == 2 else 'Sales Distribution'}"
                )
            os.remove(file)

        response = {
            "blocks": [
                {"type": "header", "text": {"type": "plain_text", "text": "Weekly Sales Analysis 📈"}},
                {"type": "section", "text": {"type": "mrkdwn", "text": insights}},
                {"type": "section", "text": {"type": "mrkdwn", "text": "Graphs uploaded to this channel! 📊"}}
            ],
            "text": f"Weekly sales analysis: Total ₹{total_sales:,.2f}, Best Product: {best_selling_product}"
        }
        logging.info(f"Weekly analysis generated for {user_id}: {response}")
        return response
    except Exception as e:
        logging.error(f"Weekly analysis error for {user_id}: {e}")
        return {
            "blocks": [
                {"type": "header", "text": {"type": "plain_text", "text": "Weekly Sales Analysis 📈"}},
                {"type": "section", "text": {"type": "mrkdwn", "text": FALLBACK_RESPONSES["weekly analysis"]}}
            ],
            "text": "Weekly analysis failed."
        }

# Sales Insights
def generate_sales_insights(user_id=None):
    df = load_sales_data()
    inventory_df = load_inventory()
    if df.empty:
        logging.warning(f"Insights failed for {user_id}: No sales data.")
        return {
            "blocks": [
                {"type": "header", "text": {"type": "plain_text", "text": "Sales Insights for April 2019 📊"}},
                {"type": "section", "text": {"type": "mrkdwn", "text": FALLBACK_RESPONSES["insights_no_data"]}}
            ],
            "text": "Sales insights failed: no data."
        }
    try:
        april_df = df[(df['Order Date'].dt.month == 4) & (df['Order Date'].dt.year == 2019)]
        if april_df.empty:
            latest_month = df['Order Date'].dt.to_period('M').max()
            april_df = df[df['Order Date'].dt.to_period('M') == latest_month]
            logging.info(f"No data for April 2019 for {user_id}, using {latest_month}")
            month_text = f"{latest_month.strftime('%B %Y')} (No April 2019 data)"
        else:
            month_text = "April 2019"
            logging.info(f"April 2019 data for {user_id}: {april_df.shape[0]} rows")

        total_sales = april_df["Price Each"].sum()
        days_in_month = 30
        weeks_in_month = 4.29
        avg_daily_sales = total_sales / days_in_month
        avg_weekly_sales = total_sales / weeks_in_month
        best_selling_product = april_df.groupby("Product")["Quantity Ordered"].sum().idxmax() if not april_df.empty else "N/A"

        trend_score = 0
        if user_id and best_selling_product in trend_cache:
            trend_score = trend_cache[best_selling_product]
            logging.info(f"Using cached trend score for {user_id}: {trend_score}")
        else:
            try:
                pytrends.build_payload(kw_list=[best_selling_product], timeframe='now 7-d')
                trends = pytrends.interest_over_time()
                trend_score = trends[best_selling_product].mean() / 100 if best_selling_product in trends else 0
                if user_id:
                    trend_cache[best_selling_product] = trend_score
            except Exception as e:
                logging.warning(f"Pytrends failed for {user_id}: {e}")
                trend_score = 0.00

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
                        f"Decrease price of {product['Product']} by ₹{price_decrease:,.2f} to boost sales by ~₹{sales_increase:,.2f} 📉"
                    )
                elif weighted_score > 0.7:
                    price_increase = product['Price'] * 0.05
                    recommendations["increase"].append(
                        f"Increase price of {product['Product']} by ₹{price_increase:,.2f} due to high demand 📈"
                    )
                if product['Stock'] < 10:
                    recommendations["restock"].append(
                        f"Restock {product['Product']} (Current: {product['Stock']}) 📦"
                    )

        insights = (
            f"*Total Sales*: ₹{total_sales:,.2f}\n"
            f"*Average Daily Sales*: ₹{avg_daily_sales:,.2f}\n"
            f"*Average Weekly Sales*: ₹{avg_weekly_sales:,.2f}\n"
            f"*Best Selling Product*: {best_selling_product} 🔥\n"
            f"*GrowBizz Trend Score*: {trend_score:.2f}"
        )
        recommendations_text = ""
        if recommendations["decrease"] or recommendations["increase"] or recommendations["restock"]:
            recommendations_text += "*Recommendations* 📋\n"
            if recommendations["decrease"]:
                recommendations_text += "🔽 *Price Decreases*:\n" + "\n".join(f"• {r}" for r in recommendations["decrease"][:3]) + "\n"
            if recommendations["increase"]:
                recommendations_text += "🔼 *Price Increases*:\n" + "\n".join(f"• {r}" for r in recommendations["increase"][:3]) + "\n"
            if recommendations["restock"]:
                recommendations_text += "📦 *Restock*:\n" + "\n".join(f"• {r}" for r in recommendations["restock"][:3]) + "\n"
        else:
            recommendations_text += "*Recommendations* 📋\nNo specific recommendations available. 🙁"

        if len(recommendations_text) > 2900:
            recommendations_text = recommendations_text[:2900] + "... (truncated)"
        logging.info(f"Insights for {user_id}: {insights}\nRecommendations length: {len(recommendations_text)}")

        response = {
            "blocks": [
                {"type": "header", "text": {"type": "plain_text", "text": f"Sales Insights for {month_text} 📊"}},
                {"type": "section", "text": {"type": "mrkdwn", "text": insights}},
                {"type": "section", "text": {"type": "mrkdwn", "text": recommendations_text}}
            ],
            "text": f"Sales insights for {month_text}: Total ₹{total_sales:,.2f}"
        }
        return response
    except Exception as e:
        logging.error(f"Insights error for {user_id}: {e}")
        return {
            "blocks": [
                {"type": "header", "text": {"type": "plain_text", "text": "Sales Insights for April 2019 📊"}},
                {"type": "section", "text": {"type": "mrkdwn", "text": FALLBACK_RESPONSES["insights"]}}
            ],
            "text": "Sales insights failed."
        }

# Audio Processing
def process_audio_query(text, user_id):
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        if "summarize call" in text.lower():
            audio_path = text.replace("summarize call", "").strip()
            if not os.path.exists(audio_path):
                return {
                    "blocks": [
                        {"type": "header", "text": {"type": "plain_text", "text": "Call Summary 🎙️"}},
                        {"type": "section", "text": {"type": "mrkdwn", "text": "Audio file not found. Please upload a valid file. 🙁"}}
                    ],
                    "text": "Call summary failed: no audio file."
                }
            prompt = "Transcribe and summarize the key points from this audio call."
            with open(audio_path, 'rb') as audio_file:
                audio_data = audio_file.read()
            contents = [
                prompt,
                {"mime_type": "audio/mp3", "data": audio_data}
            ]
            response = model.generate_content(contents)
            return {
                "blocks": [
                    {"type": "header", "text": {"type": "plain_text", "text": "Call Summary 🎙️"}},
                    {"type": "section", "text": {"type": "mrkdwn", "text": response.text.strip()}}
                ],
                "text": "Call summarized."
            }
        elif "bengali voice" in text.lower():
            parts = text.lower().split("bengali voice")
            audio_path = parts[1].strip().split()[0] if len(parts) > 1 else ""
            query = " ".join(parts[1].strip().split()[1:]) if len(parts[1].strip().split()) > 1 else "Transcribe and translate to English"
            if not os.path.exists(audio_path):
                return {
                    "blocks": [
                        {"type": "header", "text": {"type": "plain_text", "text": "Bengali Voice Message 🎙️"}},
                        {"type": "section", "text": {"type": "mrkdwn", "text": "Audio file not found. Please upload a valid file. 🙁"}}
                    ],
                    "text": "Bengali voice failed: no audio file."
                }
            prompt = f"Process this Bengali audio: {query}"
            with open(audio_path, 'rb') as audio_file:
                audio_data = audio_file.read()
            contents = [
                prompt,
                {"mime_type": "audio/mp3", "data": audio_data}
            ]
            response = model.generate_content(contents)
            return {
                "blocks": [
                    {"type": "header", "text": {"type": "plain_text", "text": "Bengali Voice Message 🎙️"}},
                    {"type": "section", "text": {"type": "mrkdwn", "text": response.text.strip()}}
                ],
                "text": "Bengali voice processed."
            }
        else:
            audio_path = text.replace("audio:", "").strip()
            if not os.path.exists(audio_path):
                return {
                    "blocks": [
                        {"type": "header", "text": {"type": "plain_text", "text": "Audio Processing 🎙️"}},
                        {"type": "section", "text": {"type": "mrkdwn", "text": "Audio file not found. Please upload a valid file. 🙁"}}
                    ],
                    "text": "Audio processing failed: no file."
                }
            prompt = "Transcribe and summarize the key points from this audio."
            with open(audio_path, 'rb') as audio_file:
                audio_data = audio_file.read()
            contents = [
                prompt,
                {"mime_type": "audio/mp3", "data": audio_data}
            ]
            response = model.generate_content(contents)
            return {
                "blocks": [
                    {"type": "header", "text": {"type": "plain_text", "text": "Audio Processing 🎙️"}},
                    {"type": "section", "text": {"type": "mrkdwn", "text": response.text.strip()}}
                ],
                "text": "Audio processed."
            }
    except Exception as e:
        logging.error(f"Audio processing error for {user_id}: {e}")
        return {
            "blocks": [
                {"type": "header", "text": {"type": "plain_text", "text": "Audio Processing 🎙️"}},
                {"type": "section", "text": {"type": "mrkdwn", "text": FALLBACK_RESPONSES["audio"]}}
            ],
            "text": "Audio processing failed."
        }

# Query Processing
def process_query(text, user_id, event_channel, event_ts):
    text = text.lower().strip()
    if user_id not in user_states:
        user_states[user_id] = {'last_message': '', 'context': 'idle', 'last_response_time': 0}
    state = user_states[user_id]
    state['last_message'] = text
    logging.info(f"Processing query: '{text}' from {user_id} in {event_channel}")

    cache_key = f"{user_id}_{text}_{event_channel}_{event_ts}"
    current_time = time.time()
    if cache_key in response_cache and (current_time - response_cache[cache_key]['time'] < 10):
        logging.info(f"Skipping duplicate query: '{text}' from {user_id}")
        return response_cache[cache_key]['response']

    try:
        if "weekly sales" in text:
            text = "weekly analysis"
        if "register" in text:
            response = handle_customer_registration(user_id, text)
        elif "purchase:" in text:
            response = process_purchase(user_id, text)
        elif "weekly analysis" in text:
            response = generate_weekly_sales_analysis(user_id, event_channel)
        elif "insights" in text or "sales insights" in text:
            response = generate_sales_insights(user_id)
        elif "promotion:" in text or "generate promotion" in text or "promotion poster" in text:
            prompt = text.replace("promotion:", "").replace("generate promotion", "").replace("promotion poster", "").strip()
            response = generate_promotion(prompt, user_id, event_channel)
        elif "whatsapp" in text or "send whatsapp message" in text:
            message = text.replace("whatsapp", "").replace("send whatsapp message", "").strip() or "Hello from GrowBizz! 😊"
            whatsapp_response = send_whatsapp_message(user_id, message)
            response = {
                "blocks": [
                    {"type": "header", "text": {"type": "plain_text", "text": "WhatsApp Message 📱"}},
                    {"type": "section", "text": {"type": "mrkdwn", "text": whatsapp_response}}
                ],
                "text": whatsapp_response
            }
        elif "invoice" in text or "generate invoice" in text:
            response = generate_invoice(None, None, user_id, event_channel)
        elif "chart" in text:
            response = generate_chart(user_id, text, event_channel)
        elif "audio:" in text or "summarize call" in text or "bengali voice" in text:
            response = process_audio_query(text, user_id)
        else:
            model = genai.GenerativeModel('gemini-1.5-flash')
            retries = 5
            for attempt in range(retries):
                try:
                    gen_response = model.generate_content(f"Respond to this user query: {text}")
                    response = {
                        "blocks": [
                            {"type": "header", "text": {"type": "plain_text", "text": "Response 🤔"}},
                            {"type": "section", "text": {"type": "mrkdwn", "text": gen_response.text.strip()}}
                        ],
                        "text": gen_response.text.strip()[:100] + "..."
                    }
                    break
                except exceptions.ResourceExhausted as e:
                    if attempt < retries - 1:
                        wait_time = 30 * (2 ** attempt)
                        logging.warning(f"Gemini API quota exceeded for {user_id}, retrying in {wait_time}s: {e}")
                        time.sleep(wait_time)
                    else:
                        logging.error(f"Default query failed for {user_id} due to API quota: {e}")
                        response = {
                            "blocks": [
                                {"type": "header", "text": {"type": "plain_text", "text": "Error 🙁"}},
                                {"type": "section", "text": {"type": "mrkdwn", "text": FALLBACK_RESPONSES["default"]}}
                            ],
                            "text": "Query failed due to API limits."
                        }
                except Exception as e:
                    logging.error(f"Default query error for {user_id}: {e}")
                    response = {
                        "blocks": [
                            {"type": "header", "text": {"type": "plain_text", "text": "Error 🙁"}},
                            {"type": "section", "text": {"type": "mrkdwn", "text": FALLBACK_RESPONSES["default"]}}
                        ],
                        "text": "Query failed."
                    }
                    break

        response_cache[cache_key] = {'response': response, 'time': current_time}
        try:
            client.chat_postMessage(
                channel=event_channel,
                blocks=response["blocks"],
                text=response.get("text", "GrowBizz response")
            )
            logging.info(f"Response sent to {event_channel} for {user_id}: {response['text']}")
        except SlackApiError as e:
            logging.error(f"Slack post failed for {user_id}: {e}")
            response = {
                "blocks": [
                    {"type": "header", "text": {"type": "plain_text", "text": "Error 🙁"}},
                    {"type": "section", "text": {"type": "mrkdwn", "text": "Failed to post response. Please try again. 🙁"}}
                ],
                "text": "Slack post failed."
            }
        return response
    except Exception as e:
        logging.error(f"Query processing error for {user_id}: {e}")
        response = {
            "blocks": [
                {"type": "header", "text": {"type": "plain_text", "text": "Error 🙁"}},
                {"type": "section", "text": {"type": "mrkdwn", "text": FALLBACK_RESPONSES["default"]}}
            ],
            "text": "Query processing failed."
        }
        try:
            client.chat_postMessage(
                channel=event_channel,
                blocks=response["blocks"],
                text=response["text"]
            )
        except SlackApiError as e:
            logging.error(f"Slack error post failed for {user_id}: {e}")
        return response

# HTTP Server for Slack Events
class SlackEventHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/plain")
        self.end_headers()
        self.wfile.write(b"GrowBizz Slack bot is running")

    def do_HEAD(self):
        self.send_response(200)
        self.send_header("Content-type", "text/plain")
        self.end_headers()

    def do_POST(self):
        if self.path == "/slack/events":
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length).decode('utf-8')
            try:
                data = json.loads(post_data)
                if data.get("type") == "url_verification":
                    self.send_response(200)
                    self.send_header("Content-type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"challenge": data.get("challenge")}).encode('utf-8'))
                    return
                event = data.get("event", {})
                user_id = event.get("user")
                text = event.get("text", "")
                event_channel = event.get("channel")
                event_ts = event.get("ts")
                if user_id and text and event_channel:
                    response = process_query(text, user_id, event_channel, event_ts)
                    self.send_response(200)
                    self.send_header("Content-type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"status": "ok", "response": response}).encode('utf-8'))
                else:
                    self.send_response(200)
                    self.send_header("Content-type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"status": "ignored"}).encode('utf-8'))
            except Exception as e:
                logging.error(f"Error processing Slack event: {e}")
                self.send_response(400)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode('utf-8'))
        else:
            self.send_response(404)
            self.send_header("Content-type", "text/plain")
            self.end_headers()
            self.wfile.write(b"Not found")

def run_http_server():
    server = HTTPServer(("", int(os.getenv("PORT", 10000))), SlackEventHandler)
    server.serve_forever()

if __name__ == "__main__":
    from slack_bolt import App as SlackApp
    from slack_bolt.adapter.socket_mode import SocketModeHandler
    slack_app = SlackApp(token=SLACK_BOT_TOKEN)

    @slack_app.message(".*")
    def handle_message(event, say):
        event_id = f"{event['event_ts']}_{event['channel']}_{event['user']}"
        if event_id in processed_events:
            logging.info(f"Skipping duplicate event {event_id}")
            return
        processed_events.add(event_id)
        if len(processed_events) > 10000:
            processed_events.clear()
        user_id = event['user']
        text = event['text']
        event_channel = event['channel']
        event_ts = event['event_ts']
        try:
            response = process_query(text, user_id, event_channel, event_ts)
        except SlackApiError as e:
            if e.response["error"] == "not_in_channel":
                try:
                    client.chat_postMessage(
                        channel=user_id,
                        blocks=[
                            {"type": "header", "text": {"type": "plain_text", "text": "Channel Issue 🚪"}},
                            {"type": "section", "text": {"type": "mrkdwn", "text": FALLBACK_RESPONSES["not_in_channel"]}}
                        ],
                        text=FALLBACK_RESPONSES["not_in_channel"]
                    )
                except SlackApiError as dm_error:
                    logging.error(f"Failed to send DM for {user_id}: {dm_error}")
            else:
                logging.error(f"Slack API error for {user_id}: {e}")

    threading.Thread(target=run_http_server, daemon=True).start()
    SocketModeHandler(slack_app, SLACK_APP_TOKEN).start()
