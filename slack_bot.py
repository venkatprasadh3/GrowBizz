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
from google.api_core import exceptions  # Added for retry handling
import time
from pytrends.request import TrendReq
from fpdf import FPDF
from PIL import Image, ImageDraw, ImageFont
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
LOGO_PATH = os.path.join(BASE_DIR, "psg_logo_blue.png")
DEFAULT_PROMO_IMG = os.path.join(BASE_DIR, "promotion_image.png")

user_states = {}
client = WebClient(token=SLACK_BOT_TOKEN)
genai.configure(api_key=GENAI_API_KEY)
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN) if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN else None
pytrends = TrendReq(hl='en-US', tz=360, retries=2, backoff_factor=0.1)
app = FastAPI()
translator = Translator()

# Expanded Fallback Responses
FALLBACK_RESPONSES = {
    "register": "Sorry, registration failed. Please try again later.",
    "purchase": "Purchase could not be processed. Please check back later.",
    "weekly analysis": "Weekly analysis unavailable. Data might be missing.",
    "insights": "Insights generation failed. Please try again.",
    "insights_api_quota": "Sales insights unavailable due to API rate limits. Please wait and retry.",
    "insights_no_data": "Sales insights unavailable because no sales data was found.",
    "promotion": "Promotion image generation failed. Try again later.",
    "promotion_no_image": "Couldn’t generate promotion image due to missing default image or network issues.",
    "whatsapp": "Failed to send WhatsApp message. Please try again.",
    "invoice": "Sorry, invoice generation failed. Please try again.",
    "chart": "Chart generation failed. Please try again later.",
    "chart_api_quota": "Chart generation failed due to API rate limits. Please wait and retry.",
    "chart_no_data": "Chart generation failed because no sales data was found.",
    "chart_invalid_query": "Chart generation failed due to an invalid query. Please specify a valid chart type.",
    "default": "Oops! I didn’t understand that. Try: register, purchase, weekly analysis, insights, promotion, whatsapp, invoice, chart, or ask me anything!",
    "not_in_channel": "I can't respond here. Please invite me to this channel with `/invite @GrowBizz` or send me a DM!"
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
        logging.error(f"WhatsApp not configured for {user_id}: Twilio credentials missing.")
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
        return f"WhatsApp message sent to {phone}!"
    except Exception as e:
        logging.error(f"WhatsApp error for {user_id}: {e}")
        return FALLBACK_RESPONSES["whatsapp"]

def send_email(user_id, subject, body, attachment=None):
    if user_id not in user_states or 'email' not in user_states[user_id]:
        logging.error(f"Email failed for {user_id}: User not registered.")
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
        logging.error(f"Email error for {user_id}: {e}")
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
        return f"You are already registered with Customer ID: `{state['customer_id']}`."
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
            welcome_msg = f"Welcome {name} to GrowBizz!"
            whatsapp_response = send_whatsapp_message(user_id, welcome_msg)
            return f"Registration successful! Customer ID: `{customer_id}`\n{whatsapp_response}"
        except Exception as e:
            logging.error(f"Registration error for {user_id}: {e}")
            return FALLBACK_RESPONSES["register"]
    logging.warning(f"Registration failed for {user_id}: Invalid format.")
    return "Please use: `register: name, email, phone, language, address`"

# Purchase Processing
def process_purchase(user_id, text):
    if user_id not in user_states or 'customer_id' not in user_states[user_id]:
        logging.error(f"Purchase failed for {user_id}: User not registered.")
        return "Please register first using: `register: name, email, phone, language, address`"
    if "purchase:" not in text.lower():
        logging.warning(f"Purchase failed for {user_id}: Invalid format.")
        return "Please use: `purchase: product, quantity`"
    try:
        _, details = text.lower().split("purchase:", 1)
        product, quantity = [x.strip() for x in details.split(',', 1)]
        quantity = int(quantity)
        inventory = load_inventory()
        if product not in inventory['Product'].values:
            logging.error(f"Purchase failed for {user_id}: Product '{product}' not in inventory.")
            return f"Product '{product}' not found in inventory."
        stock = inventory[inventory['Product'] == product]['Stock'].iloc[0]
        if stock < quantity:
            logging.error(f"Purchase failed for {user_id}: Insufficient stock for '{product}'.")
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
        logging.error(f"Purchase error for {user_id}: {e}")
        return FALLBACK_RESPONSES["purchase"]

# Invoice Generation
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
                logging.error(f"Invoice failed for {user_id}: Customer not found.")
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
        logging.warning(f"Invoice not uploaded for {user_id}: No DM channel.")
        return f"Invoice generated at {invoice_path} but not uploaded (no DM channel)."
    except Exception as e:
        logging.error(f"Invoice error for {user_id}: {e}")
        return FALLBACK_RESPONSES["invoice"]

# Promotion Generation (Fixed to Always Send Image)
def generate_promotion(prompt, user_id):
    try:
        # Parse prompt
        discount_match = re.search(r'(\d+%?|free|buy \d+ get \d+)', prompt, re.IGNORECASE)
        discount = discount_match.group(0) if discount_match else "Special Offer"
        product_match = re.search(r'(shoe|phone|headphone|monitor|cable)', prompt, re.IGNORECASE)
        product = product_match.group(0) if product_match else "Product"
        shop_match = re.search(r'(?:for|at)\s+([\w\s]+)$', prompt, re.IGNORECASE)
        shop_name = shop_match.group(1) if shop_match else "Our Store"

        # Try Unsplash, fall back to local default or gray image
        placeholder_urls = {
            "shoe": "https://images.unsplash.com/photo-1542291026-7eec264c27ff",
            "phone": "https://images.unsplash.com/photo-1511707171634-5f897ff02aa9",
            "headphone": "https://images.unsplash.com/photo-1505740420928-5e560c06d30e",
            "monitor": "https://images.unsplash.com/photo-1593642532973-d31b6557fa68",
            "cable": "https://images.unsplash.com/photo-1610056490484-256a73c68d65"
        }
        img_url = placeholder_urls.get(product.lower(), "https://images.unsplash.com/photo-1505740420928-5e560c06d30e")
        try:
            response = requests.get(img_url, timeout=5)
            response.raise_for_status()
            product_img = Image.open(BytesIO(response.content)).resize((200, 200))
            logging.info(f"Loaded Unsplash image for {product} for {user_id}.")
        except Exception as e:
            logging.warning(f"Failed to fetch Unsplash image {img_url} for {user_id}: {e}")
            if os.path.exists(DEFAULT_PROMO_IMG):
                product_img = Image.open(DEFAULT_PROMO_IMG).resize((200, 200))
                logging.info(f"Using default image {DEFAULT_PROMO_IMG} for {user_id}.")
            else:
                product_img = Image.new('RGB', (200, 200), color='gray')
                logging.warning(f"No default image found at {DEFAULT_PROMO_IMG} for {user_id}, using gray placeholder.")

        # Create promotion image
        img = Image.new('RGB', (600, 400), color=(255, 215, 0))  # Gold background
        d = ImageDraw.Draw(img)
        d.rectangle([(10, 10), (590, 390)], outline="black", width=5)

        try:
            title_font = ImageFont.truetype("arial.ttf", 50)
            shop_font = ImageFont.truetype("arial.ttf", 40)
        except:
            logging.warning(f"Font loading failed for {user_id}, using default.")
            title_font = ImageFont.load_default()
            shop_font = ImageFont.load_default()

        d.text((50, 50), discount.upper(), fill='red', font=title_font)
        d.rectangle([(45, 45), (45 + d.textlength(discount.upper(), title_font), 95)], fill=None, outline='red', width=3)
        img.paste(product_img, (350, 100))
        shop_text = f"At {shop_name}"
        shop_x = (600 - d.textlength(shop_text, shop_font)) / 2
        d.text((shop_x, 300), shop_text, fill='blue', font=shop_font)

        promo_file = os.path.join(BASE_DIR, f"promotion_{uuid.uuid4().hex[:8]}.png")
        img.save(promo_file)
        logging.info(f"Promotion image saved at {promo_file} for {user_id}.")

        dm_channel = get_dm_channel(user_id)
        if dm_channel:
            with open(promo_file, 'rb') as f:
                client.files_upload_v2(
                    channel=dm_channel,
                    file=f,
                    filename=os.path.basename(promo_file),
                    title=f"Promotion: {prompt}"
                )
            logging.info(f"Promotion image sent to DM for {user_id}.")
            return f"Promotion image generated and sent to your DM!\nText: {prompt}"
        logging.error(f"Promotion upload failed for {user_id}: No DM channel.")
        return FALLBACK_RESPONSES["promotion"]
    except Exception as e:
        logging.error(f"Promotion error for {user_id}: {e}")
        return FALLBACK_RESPONSES["promotion"]

# Chart Generation (Fixed with Retry and Cache)
def generate_chart(user_id, query):
    df = load_sales_data()
    if df.empty:
        logging.error(f"Chart failed for {user_id}: No sales data in {SALES_DATA_PATH}.")
        return FALLBACK_RESPONSES["chart_no_data"]

    # Check cache first
    cache_key = f"{query}_{df.to_string()[:100]}"
    if cache_key in chart_cache:
        logging.info(f"Using cached chart code for {user_id}.")
        code = chart_cache[cache_key]
    else:
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"""
        Generate Plotly Express code for: {query}. Use this data:
        {df.to_string()}
        Return just the code, make it professional and sleek with proper bar spacing if applicable.
        Use columns: 'Product', 'Quantity Ordered', 'Price Each', 'Order Date'.
        """
        retries = 3
        for attempt in range(retries):
            try:
                response = model.generate_content(prompt)
                code = response.text.strip()
                chart_cache[cache_key] = code  # Cache the result
                logging.info(f"Generated Plotly code for {user_id}: {code}")
                break
            except exceptions.ResourceExhausted as e:
                if attempt < retries - 1:
                    wait_time = 22 * (2 ** attempt)  # Exponential backoff: 22s, 44s, 88s
                    logging.warning(f"Gemini API quota exceeded for {user_id}, retrying in {wait_time}s: {e}")
                    time.sleep(wait_time)
                else:
                    logging.error(f"Gemini API quota exhausted for {user_id} after {retries} attempts: {e}")
                    return FALLBACK_RESPONSES["chart_api_quota"]
            except Exception as e:
                logging.error(f"Chart generation error for {user_id}: {e}")
                return FALLBACK_RESPONSES["chart"]

    try:
        if "px." not in code.lower():
            logging.error(f"Chart failed for {user_id}: Invalid Plotly code - {code}")
            return FALLBACK_RESPONSES["chart_invalid_query"]

        local_vars = {"df": df, "px": px}
        exec(code, globals(), local_vars)
        fig = local_vars.get("fig")
        if not fig:
            logging.error(f"Chart failed for {user_id}: No figure generated from code - {code}")
            return FALLBACK_RESPONSES["chart"]

        img_byte_arr = BytesIO()
        fig.write_image(img_byte_arr, format="png")
        img_byte_arr.seek(0)

        dm_channel = get_dm_channel(user_id)
        if dm_channel:
            client.files_upload_v2(
                channel=dm_channel,
                file=img_byte_arr,
                filename=f"chart_{uuid.uuid4().hex[:8]}.png",
                title="Sales Chart"
            )
            logging.info(f"Chart sent to DM for {user_id}.")
            return "Chart sent to your DM!"
        logging.error(f"Chart upload failed for {user_id}: No DM channel.")
        return FALLBACK_RESPONSES["chart"]
    except Exception as e:
        logging.error(f"Chart rendering error for {user_id}: {e}")
        return FALLBACK_RESPONSES["chart"]

# Weekly Sales Analysis
def generate_weekly_sales_analysis(user_id):
    df = load_sales_data()
    if df.empty:
        logging.error(f"Weekly analysis failed for {user_id}: No sales data.")
        return FALLBACK_RESPONSES["weekly analysis"]
    try:
        weekly_sales = df.resample('W', on='Order Date')['Price Each'].sum()
        plt.figure(figsize=(12, 6))
        plt.plot(weekly_sales.index, weekly_sales.values, marker='o', linestyle='-', color='#4CAF50')
        plt.title('Weekly Sales Trend')
        plt.xlabel('Week')
        plt.ylabel('Sales (INR)')
        plt.grid(True)
        weekly_trend_file = os.path.join(BASE_DIR, f"weekly_trend_{uuid.uuid4().hex[:8]}.png")
        plt.savefig(weekly_trend_file)
        plt.close()

        plt.figure(figsize=(12, 6))
        plt.plot(df['Order Date'], df['Price Each'].cumsum(), color='#2196F3')
        plt.title('Overall Sales Trend')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Sales (INR)')
        plt.grid(True)
        overall_trend_file = os.path.join(BASE_DIR, f"overall_trend_{uuid.uuid4().hex[:8]}.png")
        plt.savefig(overall_trend_file)
        plt.close()

        mu, std = norm.fit(df['Price Each'].dropna())
        plt.figure(figsize=(10, 6))
        sns.histplot(df['Price Each'], kde=True, stat="density", color='#2196F3')
        x = np.linspace(df['Price Each'].min(), df['Price Each'].max(), 100)
        plt.plot(x, norm.pdf(x, mu, std), 'r-', lw=2, label=f'Normal fit (μ={mu:.2f}, σ={std:.2f})')
        plt.title('Sales Distribution with Normal Fit')
        plt.xlabel('Sales Amount (INR)')
        plt.ylabel('Density')
        plt.legend()
        sales_dist_file = os.path.join(BASE_DIR, f"sales_distribution_{uuid.uuid4().hex[:8]}.png")
        plt.savefig(sales_dist_file)
        plt.close()

        dm_channel = get_dm_channel(user_id)
        if dm_channel:
            for i, file in enumerate([weekly_trend_file, overall_trend_file, sales_dist_file], 1):
                with open(file, 'rb') as f:
                    client.files_upload_v2(
                        channel=dm_channel,
                        file=f,
                        filename=os.path.basename(file),
                        title=f"Weekly Analysis Graph {i}"
                    )
            logging.info(f"Weekly analysis sent to DM for {user_id}.")
            return "Weekly sales analysis (3 graphs) sent to your DM!"
        logging.error(f"Weekly analysis upload failed for {user_id}: No DM channel.")
        return FALLBACK_RESPONSES["weekly analysis"]
    except Exception as e:
        logging.error(f"Weekly analysis error for {user_id}: {e}")
        return FALLBACK_RESPONSES["weekly analysis"]

# Sales Insights (Fixed with Retry and Cache)
def generate_sales_insights(user_id=None):
    df = load_sales_data()
    inventory_df = load_inventory()
    if df.empty:
        logging.error(f"Insights failed for {user_id}: No sales data in {SALES_DATA_PATH}.")
        return FALLBACK_RESPONSES["insights_no_data"]
    try:
        total_sales = df["Price Each"].sum()
        avg_sale = df["Price Each"].mean()
        best_selling_product = df.groupby("Product")["Quantity Ordered"].sum().idxmax()

        # Use cached trend score if available
        if user_id and best_selling_product in trend_cache:
            trend_score = trend_cache[best_selling_product]
            logging.info(f"Using cached trend score for {user_id} - {best_selling_product}: {trend_score}")
        else:
            retries = 3
            for attempt in range(retries):
                try:
                    pytrends.build_payload(kw_list=[best_selling_product], timeframe='now 7-d')
                    trends = pytrends.interest_over_time()
                    trend_score = trends[best_selling_product].mean() / 100 if best_selling_product in trends else 0
                    if user_id:
                        trend_cache[best_selling_product] = trend_score  # Cache only if user_id present
                    logging.info(f"Trend score for {best_selling_product}: {trend_score} for {user_id}")
                    break
                except Exception as e:
                    if "429" in str(e) and attempt < retries - 1:
                        wait_time = 10 * (2 ** attempt)  # Exponential backoff: 10s, 20s, 40s
                        logging.warning(f"Pytrends quota exceeded for {user_id}, retrying in {wait_time}s: {e}")
                        time.sleep(wait_time)
                    else:
                        logging.error(f"Pytrends failed for {user_id} after {retries} attempts: {e}")
                        trend_score = 0.00  # Fallback
                        break

        recommendations = []
        if not inventory_df.empty:
            for _, product in inventory_df.iterrows():
                product_sales = df[df['Product'] == product['Product']]['Price Each'].sum()
                sales_score = product_sales / total_sales if total_sales > 0 else 0
                stock_score = 1 - (product['Stock'] / 100)
                weighted_score = 0.4 * sales_score + 0.3 * stock_score + 0.3 * trend_score

                if weighted_score < 0.3:
                    recommendations.append(f"Decrease price or promote {product['Product']} (Score: {weighted_score:.2f})")
                elif weighted_score > 0.7:
                    recommendations.append(f"Increase price for {product['Product']} (Score: {weighted_score:.2f})")
                if product['Stock'] < 10:
                    recommendations.append(f"Restock {product['Product']} (Current: {product['Stock']})")

        insights = (
            f"📊 Sales Insights:\n"
            f"🔹 Total Sales: INR {total_sales:,.2f}\n"
            f"🔹 Average Sale: INR {avg_sale:,.2f}\n"
            f"🔥 Best Selling Product: {best_selling_product}\n"
            f"📈 Trend Score: {trend_score:.2f}\n\n"
        )
        if recommendations:
            insights += f"📦 Inventory Recommendations:\n" + "\n".join(recommendations)
        else:
            insights += "📦 No inventory data available for recommendations."
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
            return generate_weekly_sales_analysis(user_id)
        elif "insights" in text or "sales insights" in text:
            return generate_sales_insights(user_id)
        elif "promotion:" in text or "generate promotion" in text:
            prompt = text.replace("promotion:", "").replace("generate promotion", "").strip()
            return generate_promotion(prompt, user_id)
        elif "whatsapp" in text or "send whatsapp message" in text:
            message = text.replace("whatsapp", "").replace("send whatsapp message", "").strip() or "Hello from GrowBizz!"
            return send_whatsapp_message(user_id, message)
        elif "invoice" in text or "generate invoice" in text:
            return generate_invoice(None, None, user_id)
        elif "chart" in text:
            return generate_chart(user_id, text)
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
