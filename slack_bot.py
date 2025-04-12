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
processed_events = set()  # Track processed Slack events

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
    "audio": "Audio processing failed. Please try again later. 🎙️"
}

# Cache for API results
trend_cache = {}
chart_cache = {}

# Helper Functions
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
            media_url = media_url or "https://via.placeholder.com/600x400"  # Replace with actual URL in production
        response = twilio_client.messages.create(
            body=translated_msg,
            from_=f"whatsapp:{TWILIO_PHONE_NUMBER}",
            to=f"whatsapp:{phone}",
            media_url=[media_url] if media_url else None
        )
        logging.info(f"WhatsApp media message SID: {response.sid}, Status: {response.status}, To: {phone}")
        return f"WhatsApp message with media sent to {phone}! 📱"
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
                {
                    "type": "header",
                    "text": {"type": "plain_text", "text": "User Already Registered 😊"}
                },
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": f"You are already registered with Customer ID: `{state['customer_id']}`."}
                }
            ],
            "text": "User already registered."
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
            welcome_msg = f"Welcome to GrowBizz, {name}! Enjoy your shopping! 🛒\n📍 Address: {address}\n📞 Phone: {phone}"
            whatsapp_response = send_whatsapp_message(user_id, welcome_msg)
            response = {
                "blocks": [
                    {
                        "type": "header",
                        "text": {"type": "plain_text", "text": "Registration Successful ✅"}
                    },
                    {
                        "type": "section",
                        "text": {"type": "mrkdwn", "text": f"Customer ID: `{customer_id}`\n{whatsapp_response}"}
                    }
                ],
                "text": f"Registered {name} with ID {customer_id}"
            }
            logging.info(f"Registration successful for {user_id}: {response}")
            return response
        except Exception as e:
            logging.error(f"Registration error for {user_id}: {e}")
            return {
                "blocks": [
                    {
                        "type": "header",
                        "text": {"type": "plain_text", "text": "Registration Failed 🙁"}
                    },
                    {
                        "type": "section",
                        "text": {"type": "mrkdwn", "text": FALLBACK_RESPONSES["register"]}
                    }
                ],
                "text": "Registration failed."
            }
    return {
        "blocks": [
            {
                "type": "header",
                "text": {"type": "plain_text", "text": "Registration Help 📝"}
            },
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": "Please use: `register: name, email, phone, language, address`"}
            }
        ],
        "text": "Registration help provided."
    }

# Purchase Processing
def process_purchase(user_id, text):
    if user_id not in user_states or 'customer_id' not in user_states[user_id]:
        return {
            "blocks": [
                {
                    "type": "header",
                    "text": {"type": "plain_text", "text": "Purchase Failed 🙁"}
                },
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": "Please register first using: `register: name, email, phone, language, address` 📝"}
                }
            ],
            "text": "Purchase failed: not registered."
        }
    if "purchase:" not in text.lower():
        return {
            "blocks": [
                {
                    "type": "header",
                    "text": {"type": "plain_text", "text": "Purchase Help 🛍️"}
                },
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": "Please use: `purchase: product, quantity`"}
                }
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
                    {
                        "type": "header",
                        "text": {"type": "plain_text", "text": "Purchase Failed 🙁"}
                    },
                    {
                        "type": "section",
                        "text": {"type": "mrkdwn", "text": f"Product '{product}' not found in inventory."}
                    }
                ],
                "text": f"Purchase failed: {product} not found."
            }
        stock = inventory[inventory['Product'] == product]['Stock'].iloc[0]
        if stock < quantity:
            return {
                "blocks": [
                    {
                        "type": "header",
                        "text": {"type": "plain_text", "text": "Purchase Failed 🙁"}
                    },
                    {
                        "type": "section",
                        "text": {"type": "mrkdwn", "text": f"Insufficient stock for '{product}'. Available: {stock} 📦"}
                    }
                ],
                "text": f"Purchase failed: insufficient stock for {product}."
            }
        price = inventory[inventory['Product'] == product]['Price'].iloc[0]
        update_inventory(product, quantity)
        update_sales_data(product, quantity, price)
        invoice_msg = generate_invoice(user_states[user_id]['customer_id'], {'product_name': product, 'price': price * quantity, 'quantity': quantity}, user_id)
        purchase_msg = f"Purchase confirmed: {quantity} x {product} for ₹{price * quantity:,.2f} 🛒"
        whatsapp_response = send_whatsapp_message(user_id, purchase_msg)
        email_response = send_email(user_id, "Purchase Confirmation", purchase_msg, "invoice_A35432.pdf" if os.path.exists("invoice_A35432.pdf") else None)
        response = {
            "blocks": [
                {
                    "type": "header",
                    "text": {"type": "plain_text", "text": "Purchase Confirmed 🛒"}
                },
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": f"{purchase_msg}\n{invoice_msg['blocks'][1]['text']['text']}\n{whatsapp_response}\n{email_response}"}
                }
            ],
            "text": f"Purchase confirmed: {quantity} x {product}"
        }
        logging.info(f"Purchase successful for {user_id}: {response}")
        return response
    except Exception as e:
        logging.error(f"Purchase error for {user_id}: {e}")
        return {
            "blocks": [
                {
                    "type": "header",
                    "text": {"type": "plain_text", "text": "Purchase Failed 🙁"}
                },
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": FALLBACK_RESPONSES["purchase"]}
                }
            ],
            "text": "Purchase failed."
        }

# Invoice Generation
def generate_invoice(customer_id=None, product=None, user_id=None, event_channel=None):
    try:
        if not user_id or not event_channel:
            return {
                "blocks": [
                    {
                        "type": "header",
                        "text": {"type": "plain_text", "text": "Invoice Generation 📄"}
                    },
                    {
                        "type": "section",
                        "text": {"type": "mrkdwn", "text": "User ID or channel required for invoice generation. 🙁"}
                    }
                ],
                "text": "Invoice failed: missing user ID or channel."
            }
        users_df = load_users()
        customer = users_df[users_df['slack_id'] == user_id].iloc[0] if user_id in users_df['slack_id'].values else None
        if not customer:
            return {
                "blocks": [
                    {
                        "type": "header",
                        "text": {"type": "plain_text", "text": "Invoice Generation 📄"}
                    },
                    {
                        "type": "section",
                        "text": {"type": "mrkdwn", "text": "Please register first to generate an invoice. 🙁"}
                    }
                ],
                "text": "Invoice failed: user not registered."
            }
        # Default product for standalone invoice
        product = product or {'product_name': 'Sample Product', 'price': 1000.00, 'quantity': 1}
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Shoe Shop Invoice</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 0; padding: 0; background-color: #f9f9f9; color: #333; }}
                .container {{ max-width: 800px; margin: 20px auto; padding: 20px; background-color: #fff; border-radius: 10px; }}
                header {{ text-align: center; margin-bottom: 20px; }}
                header img {{ width: 100px; }}
                header h1 {{ color: #2c3e50; }}
                table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; }}
                th, td {{ padding: 10px; border: 1px solid #ddd; text-align: left; }}
                th {{ background-color: #f0f0f0; }}
                .total {{ text-align: right; font-weight: bold; }}
            </style>
        </head>
        <body>
            <div class="container">
                <header>
                    <img src="{LOGO_PATH}" alt="Smart Shoes Logo">
                    <h1>Smart Shoes</h1>
                    <p>15, MG Road, Bengaluru, Karnataka, India</p>
                </header>
                <p>Invoice Number: A35432</p>
                <p>Date: April 04, 2019</p>
                <p>Bill To: {customer['name']}, {customer['address']}</p>
                <table>
                    <thead>
                        <tr>
                            <th>Qty</th>
                            <th>Product</th>
                            <th>Unit Price</th>
                            <th>Total</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td>{product['quantity']}</td>
                            <td>{product['product_name']}</td>
                            <td>₹{product['price'] / product['quantity']:,.2f}</td>
                            <td>₹{product['price']:,.2f}</td>
                        </tr>
                    </tbody>
                </table>
                <div class="total">
                    <p>Subtotal: ₹{product['price']:,.2f}</p>
                    <p>Tax (12%): ₹{product['price'] * 0.12:,.2f}</p>
                    <p>Total: ₹{product['price'] * 1.12:,.2f}</p>
                </div>
            </div>
        </body>
        </html>
        """
        invoice_path = os.path.join(BASE_DIR, "invoice_A35432.pdf")
        HTML(string=html_content).write_pdf(invoice_path)
        logging.info(f"Invoice PDF generated at {invoice_path} for {user_id}")
        try:
            with open(invoice_path, 'rb') as f:
                client.files_upload_v2(
                    channel=event_channel,
                    file=f,
                    filename="invoice_A35432.pdf",
                    title="Smart Shoes Invoice"
                )
            whatsapp_msg = f"Hey {customer['name']} 👋, your invoice A35432 is ready! Check it out."
            whatsapp_response = send_whatsapp_media_message(user_id, whatsapp_msg, media_path=invoice_path)
            response = {
                "blocks": [
                    {
                        "type": "header",
                        "text": {"type": "plain_text", "text": "Invoice Generated 📄"}
                    },
                    {
                        "type": "section",
                        "text": {"type": "mrkdwn", "text": f"Invoice uploaded to this channel as `invoice_A35432.pdf`!\n{whatsapp_response}"}
                    }
                ],
                "text": "Invoice A35432 generated and uploaded."
            }
            logging.info(f"Invoice uploaded for {user_id}: {response}")
            return response
        except SlackApiError as e:
            logging.error(f"Invoice upload failed for {user_id}: {e}")
            return {
                "blocks": [
                    {
                        "type": "header",
                        "text": {"type": "plain_text", "text": "Invoice Generation 📄"}
                    },
                    {
                        "type": "section",
                        "text": {"type": "mrkdwn", "text": f"Invoice generated at {invoice_path} but upload failed. 🙁\n{whatsapp_response}"}
                    }
                ],
                "text": "Invoice generated but upload failed."
            }
    except Exception as e:
        logging.error(f"Invoice error for {user_id}: {e}")
        return {
            "blocks": [
                {
                    "type": "header",
                    "text": {"type": "plain_text", "text": "Invoice Generation 📄"}
                },
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": FALLBACK_RESPONSES["invoice"]}
                }
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
                    {
                        "type": "header",
                        "text": {"type": "plain_text", "text": "Promotion Generation 🖼️"}
                    },
                    {
                        "type": "section",
                        "text": {"type": "mrkdwn", "text": FALLBACK_RESPONSES["promotion_no_image"]}
                    }
                ],
                "text": "Promotion failed: no image."
            }
        model = genai.GenerativeModel('gemini-1.5-flash')
        description = model.generate_content(f"Create a short description for a promotion poster based on: {prompt}").text.strip()
        try:
            with open(DEFAULT_PROMO_IMG, 'rb') as f:
                client.files_upload_v2(
                    channel=event_channel,
                    file=f,
                    filename="promotion_image.png",
                    title="Promotion Image"
                )
            logging.info(f"Promotion image uploaded to channel {event_channel} for {user_id}")
            response = {
                "blocks": [
                    {
                        "type": "header",
                        "text": {"type": "plain_text", "text": "Promotion Generated 🖼️"}
                    },
                    {
                        "type": "section",
                        "text": {"type": "mrkdwn", "text": f"{description}\n\nImage uploaded to this channel!"}
                    }
                ],
                "text": f"Promotion poster generated: {description[:100]}..."
            }
            return response
        except SlackApiError as e:
            logging.error(f"Slack upload failed for {user_id}: {e}")
            return {
                "blocks": [
                    {
                        "type": "header",
                        "text": {"type": "plain_text", "text": "Promotion Generation 🖼️"}
                    },
                    {
                        "type": "section",
                        "text": {"type": "mrkdwn", "text": FALLBACK_RESPONSES["promotion"]}
                    }
                ],
                "text": "Promotion upload failed."
            }
    except Exception as e:
        logging.error(f"Promotion generation failed for {user_id}: {e}")
        return {
            "blocks": [
                {
                    "type": "header",
                    "text": {"type": "plain_text", "text": "Promotion Generation 🖼️"}
                },
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": FALLBACK_RESPONSES["promotion"]}
                }
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
                {
                    "type": "header",
                    "text": {"type": "plain_text", "text": "Chart Generation 📊"}
                },
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": FALLBACK_RESPONSES["chart_no_data"]}
                }
            ],
            "text": "Chart failed: no sales data."
        }
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
        retries = 5
        for attempt in range(retries):
            try:
                response = model.generate_content(prompt)
                code = response.text.strip()
                chart_cache[cache_key] = code
                logging.info(f"Generated Plotly code for {user_id}: {code}")
                break
            except exceptions.ResourceExhausted as e:
                if attempt < retries - 1:
                    wait_time = 30 * (2 ** attempt)
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
            return {
                "blocks": [
                    {
                        "type": "header",
                        "text": {"type": "plain_text", "text": "Chart Generation 📊"}
                    },
                    {
                        "type": "section",
                        "text": {"type": "mrkdwn", "text": FALLBACK_RESPONSES["chart"]}
                    }
                ],
                "text": "Chart failed: no figure generated."
            }
        img_byte_arr = BytesIO()
        fig.write_image(img_byte_arr, format="png", engine="kaleido", width=800, height=600)
        img_byte_arr.seek(0)
        try:
            client.files_upload_v2(
                channel=event_channel,
                file=img_byte_arr,
                filename=f"chart_{uuid.uuid4().hex[:8]}.png",
                title="Sales Chart"
            )
            logging.info(f"Chart uploaded to channel {event_channel} for {user_id}")
            return {
                "blocks": [
                    {
                        "type": "header",
                        "text": {"type": "plain_text", "text": "Chart Generated 📊"}
                    },
                    {
                        "type": "section",
                        "text": {"type": "mrkdwn", "text": "Chart uploaded to this channel!"}
                    }
                ],
                "text": "Chart uploaded."
            }
        except SlackApiError as e:
            logging.error(f"Chart upload failed for {user_id}: {e}")
            return {
                "blocks": [
                    {
                        "type": "header",
                        "text": {"type": "plain_text", "text": "Chart Generation 📊"}
                    },
                    {
                        "type": "section",
                        "text": {"type": "mrkdwn", "text": FALLBACK_RESPONSES["chart"]}
                    }
                ],
                "text": "Chart upload failed."
            }
    except Exception as e:
        logging.error(f"Chart rendering/upload failed for {user_id}: {e}")
        return {
            "blocks": [
                {
                    "type": "header",
                    "text": {"type": "plain_text", "text": "Chart Generation 📊"}
                },
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": FALLBACK_RESPONSES["chart"]}
                }
            ],
            "text": "Chart rendering failed."
        }

# Weekly Sales Analysis
def generate_weekly_sales_analysis(user_id, event_channel):
    df = load_sales_data()
    if df.empty:
        logging.warning(f"Weekly analysis failed for {user_id}: No sales data.")
        return {
            "blocks": [
                {
                    "type": "header",
                    "text": {"type": "plain_text", "text": "Weekly Sales Analysis 📈"}
                },
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": FALLBACK_RESPONSES["weekly analysis"]}
                }
            ],
            "text": "Weekly analysis failed: no sales data."
        }
    try:
        weekly_sales = df.resample('W', on='Order Date')['Price Each'].sum()
        total_sales = weekly_sales.sum()
        avg_weekly_sales = weekly_sales.mean()
        best_selling_product = df.groupby('Product')['Quantity Ordered'].sum().idxmax()

        # Plot 1: Weekly Sales Trend
        plt.figure(figsize=(12, 6))
        plt.plot(weekly_sales.index, weekly_sales.values, marker='o', linestyle='-', color='#4CAF50')
        plt.title('Weekly Sales Trend', fontsize=14, fontweight='bold')
        plt.xlabel('Week', fontsize=12)
        plt.ylabel('Sales (₹)', fontsize=12)
        plt.grid(True)
        weekly_trend_file = os.path.join(BASE_DIR, f"weekly_trend_{uuid.uuid4().hex[:8]}.png")
        plt.savefig(weekly_trend_file)
        plt.close()

        # Plot 2: Overall Sales Trend
        plt.figure(figsize=(12, 6))
        plt.plot(df['Order Date'], df['Price Each'].cumsum(), color='#2196F3')
        plt.title('Overall Sales Trend', fontsize=14, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Cumulative Sales (₹)', fontsize=12)
        plt.grid(True)
        overall_trend_file = os.path.join(BASE_DIR, f"overall_trend_{uuid.uuid4().hex[:8]}.png")
        plt.savefig(overall_trend_file)
        plt.close()

        # Plot 3: Sales Distribution
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
                {
                    "type": "header",
                    "text": {"type": "plain_text", "text": "Weekly Sales Analysis 📈"}
                },
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": insights}
                },
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": "Graphs uploaded to this channel! 📊"}
                }
            ],
            "text": f"Weekly sales analysis: Total ₹{total_sales:,.2f}, Best Product: {best_selling_product}"
        }
        logging.info(f"Weekly analysis generated for {user_id}: {response}")
        return response
    except Exception as e:
        logging.error(f"Weekly analysis error for {user_id}: {e}")
        return {
            "blocks": [
                {
                    "type": "header",
                    "text": {"type": "plain_text", "text": "Weekly Sales Analysis 📈"}
                },
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": FALLBACK_RESPONSES["weekly analysis"]}
                }
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
                {
                    "type": "header",
                    "text": {"type": "plain_text", "text": "Sales Insights for April 2019 📊"}
                },
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": FALLBACK_RESPONSES["insights_no_data"]}
                }
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
            f"*Trend Score*: {trend_score:.2f}"
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

        # Truncate to avoid Slack block limit
        if len(recommendations_text) > 2900:
            recommendations_text = recommendations_text[:2900] + "... (truncated)"
        logging.info(f"Insights for {user_id}: {insights}\nRecommendations length: {len(recommendations_text)}")

        response = {
            "blocks": [
                {
                    "type": "header",
                    "text": {"type": "plain_text", "text": f"Sales Insights for {month_text} 📊"}
                },
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": insights}
                },
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": recommendations_text}
                }
            ],
            "text": f"Sales insights for {month_text}: Total ₹{total_sales:,.2f}"
        }
        return response
    except Exception as e:
        logging.error(f"Insights error for {user_id}: {e}")
        return {
            "blocks": [
                {
                    "type": "header",
                    "text": {"type": "plain_text", "text": "Sales Insights for April 2019 📊"}
                },
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": FALLBACK_RESPONSES["insights"]}
                }
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
                        {
                            "type": "header",
                            "text": {"type": "plain_text", "text": "Call Summary 🎙️"}
                        },
                        {
                            "type": "section",
                            "text": {"type": "mrkdwn", "text": "Audio file not found. Please upload a valid file. 🙁"}
                        }
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
                    {
                        "type": "header",
                        "text": {"type": "plain_text", "text": "Call Summary 🎙️"}
                    },
                    {
                        "type": "section",
                        "text": {"type": "mrkdwn", "text": response.text.strip()}
                    }
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
                        {
                            "type": "header",
                            "text": {"type": "plain_text", "text": "Bengali Voice Message 🎙️"}
                        },
                        {
                            "type": "section",
                            "text": {"type": "mrkdwn", "text": "Audio file not found. Please upload a valid file. 🙁"}
                        }
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
                    {
                        "type": "header",
                        "text": {"type": "plain_text", "text": "Bengali Voice Message 🎙️"}
                    },
                    {
                        "type": "section",
                        "text": {"type": "mrkdwn", "text": response.text.strip()}
                    }
                ],
                "text": "Bengali voice processed."
            }
        else:
            audio_path = text.replace("audio:", "").strip()
            if not os.path.exists(audio_path):
                return {
                    "blocks": [
                        {
                            "type": "header",
                            "text": {"type": "plain_text", "text": "Audio Processing 🎙️"}
                        },
                        {
                            "type": "section",
                            "text": {"type": "mrkdwn", "text": "Audio file not found. Please upload a valid file. 🙁"}
                        }
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
                    {
                        "type": "header",
                        "text": {"type": "plain_text", "text": "Audio Processing 🎙️"}
                    },
                    {
                        "type": "section",
                        "text": {"type": "mrkdwn", "text": response.text.strip()}
                    }
                ],
                "text": "Audio processed."
            }
    except Exception as e:
        logging.error(f"Audio processing error for {user_id}: {e}")
        return {
            "blocks": [
                {
                    "type": "header",
                    "text": {"type": "plain_text", "text": "Audio Processing 🎙️"}
                },
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": FALLBACK_RESPONSES["audio"]}
                }
            ],
            "text": "Audio processing failed."
        }

# Query Processing
def process_query(text, user_id, event_channel):
    text = text.lower().strip()
    if user_id not in user_states:
        user_states[user_id] = {'last_message': '', 'context': 'idle'}
    state = user_states[user_id]
    state['last_message'] = text
    logging.info(f"Processing query: '{text}' from {user_id} in {event_channel}")
    try:
        # Normalize query
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
                    {
                        "type": "header",
                        "text": {"type": "plain_text", "text": "WhatsApp Message 📱"}
                    },
                    {
                        "type": "section",
                        "text": {"type": "mrkdwn", "text": whatsapp_response}
                    }
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
                            {
                                "type": "header",
                                "text": {"type": "plain_text", "text": "Response 🤔"}
                            },
                            {
                                "type": "section",
                                "text": {"type": "mrkdwn", "text": gen_response.text.strip()}
                            }
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
                                {
                                    "type": "header",
                                    "text": {"type": "plain_text", "text": "Error 🙁"}
                                },
                                {
                                    "type": "section",
                                    "text": {"type": "mrkdwn", "text": FALLBACK_RESPONSES["default"]}
                                }
                            ],
                            "text": "Query failed due to API limits."
                        }
                except Exception as e:
                    logging.error(f"Default query error for {user_id}: {e}")
                    response = {
                        "blocks": [
                            {
                                "type": "header",
                                "text": {"type": "plain_text", "text": "Error 🙁"}
                            },
                            {
                                "type": "section",
                                "text": {"type": "mrkdwn", "text": FALLBACK_RESPONSES["default"]}
                                }
                            ],
                            "text": "Query failed."
                        }
        # Post response to Slack
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
                    {
                        "type": "header",
                        "text": {"type": "plain_text", "text": "Error 🙁"}
                    },
                    {
                        "type": "section",
                        "text": {"type": "mrkdwn", "text": "Failed to post response. Please try again. 🙁"}
                    }
                ],
                "text": "Slack post failed."
            }
        return response
    except Exception as e:
        logging.error(f"Query processing error for {user_id}: {e}")
        response = {
            "blocks": [
                {
                    "type": "header",
                    "text": {"type": "plain_text", "text": "Error 🙁"}
                },
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": FALLBACK_RESPONSES["default"]}
                }
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

# FastAPI Endpoints
@app.post("/register")
async def api_register(request: Request):
    data = await request.json()
    text = data.get("text", "")
    user_id = data.get("user_id", "api_user")
    response = handle_customer_registration(user_id, text)
    return JSONResponse({"response": response["blocks"]})

@app.post("/purchase")
async def api_purchase(request: Request):
    data = await request.json()
    text = data.get("text", "")
    user_id = data.get("user_id", "api_user")
    response = process_purchase(user_id, text)
    return JSONResponse({"response": response["blocks"]})

@app.post("/query")
async def api_query(request: Request):
    data = await request.json()
    text = data.get("text", "")
    user_id = data.get("user_id", "api_user")
    response = process_query(text, user_id, "api_channel")
    return JSONResponse({"response": response["blocks"]})

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
    server = HTTPServer(("", int(os.getenv("PORT", 10000))), DummyHandler)
    server.serve_forever()

if __name__ == "__main__":
    from slack_bolt import App as SlackApp
    from slack_bolt.adapter.socket_mode import SocketModeHandler
    slack_app = SlackApp(token=SLACK_BOT_TOKEN)

    @slack_app.message(".*")
    def handle_message(event, say):
        event_id = f"{event['event_ts']}_{event['channel']}"
        if event_id in processed_events:
            logging.info(f"Skipping duplicate event {event_id}")
            return
        processed_events.add(event_id)
        user_id = event['user']
        text = event['text']
        event_channel = event['channel']
        response = process_query(text, user_id, event_channel)
        try:
            say(blocks=response["blocks"], text=response.get("text", "GrowBizz response"))
        except SlackApiError as e:
            if e.response["error"] == "not_in_channel":
                try:
                    client.chat_postMessage(
                        channel=user_id,
                        blocks=[
                            {
                                "type": "header",
                                "text": {"type": "plain_text", "text": "Channel Issue 🚪"}
                            },
                            {
                                "type": "section",
                                "text": {"type": "mrkdwn", "text": FALLBACK_RESPONSES["not_in_channel"]}
                            }
                        ],
                        text=FALLBACK_RESPONSES["not_in_channel"]
                    )
                except SlackApiError as dm_error:
                    logging.error(f"Failed to send DM for {user_id}: {dm_error}")
            else:
                logging.error(f"Slack API error for {user_id}: {e}")

    threading.Thread(target=run_http_server, daemon=True).start()
    SocketModeHandler(slack_app, SLACK_APP_TOKEN).start()
