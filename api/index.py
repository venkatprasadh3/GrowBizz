import os
import re
import logging
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import uuid
from io import BytesIO
from twilio.rest import Client
import google.generativeai as genai
from fpdf import FPDF
from PIL import Image, ImageDraw, ImageFont
import image_gen
import numpy as np
from scipy.stats import norm
from pytrends.request import TrendReq
from flask import Flask, request, jsonify, Response
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

app = Flask(__name__)

# Configuration and Constants
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Environment Variables (Set in Vercel Dashboard)
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
GENAI_API_KEY = os.getenv("GENAI_API_KEY")
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER", "")
SALES_DATA_PATH = "sales_data.csv"
INVENTORY_PATH = "inventory.csv"
USERS_PATH = "users.csv"
DEFAULT_LANGUAGE = "English"

# Global Variables
user_states = {}
client = WebClient(token=SLACK_BOT_TOKEN)
genai.configure(api_key=GENAI_API_KEY)
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN) if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN else None
pytrends = TrendReq(hl='en-IN', tz=330)

# Static Fallback Responses
FALLBACK_RESPONSES = {
    "register": "Sorry, registration failed. Please try again later.",
    "purchase": "Purchase could not be processed. Please check back later.",
    "weekly analysis": "Weekly analysis unavailable. Data might be missing.",
    "insights": "Insights generation failed. Please try again.",
    "promotion": "Promotion image generation failed. Try again later.",
    "whatsapp": "Failed to send WhatsApp message. Please try again.",
    "invoice": "Sorry, invoice generation failed. Please try again.",
    "default": "Oops! I didn’t understand that. Try: register, purchase, weekly analysis, insights, promotion, whatsapp, invoice, or ask me anything!",
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
    else:
        logging.error(f"Product {product} not found in inventory")
        return False

def load_users():
    if os.path.exists(USERS_PATH):
        return pd.read_csv(USERS_PATH)
    else:
        df = pd.DataFrame(columns=["customer_id", "business_name", "owner_name", "phone", "business_type", "email", "language", "address"])
        df.to_csv(USERS_PATH, index=False)
        return df

def load_sales_data():
    if os.path.exists(SALES_DATA_PATH):
        df = pd.read_csv(SALES_DATA_PATH, low_memory=False)
        df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')
        df.dropna(subset=['Order Date'], inplace=True)
        df['Quantity Ordered'] = pd.to_numeric(df['Quantity Ordered'], errors='coerce').fillna(0).astype('Int64')
        df['Price Each'] = pd.to_numeric(df['Price Each'], errors='coerce').fillna(0.0)
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

# User Registration
def handle_customer_registration(user_id, data):
    if user_id in user_states and 'customer_id' in user_states[user_id]:
        return f"You are already registered with ID: `{user_states[user_id]['customer_id']}`."

    try:
        customer_id = str(uuid.uuid4())
        user_states[user_id] = {
            'customer_id': customer_id,
            'business_name': data['business_name'],
            'owner_name': data['owner_name'],
            'phone': '+' + data['phone'].strip().replace('+', ''),
            'business_type': data['business_type'],
            'email': data.get('email', ''),
            'language': data.get('language', DEFAULT_LANGUAGE),
            'address': data.get('address', '')
        }
        
        users_df = load_users()
        new_user = pd.DataFrame([user_states[user_id]])
        users_df = pd.concat([users_df, new_user], ignore_index=True)
        users_df.to_csv(USERS_PATH, index=False)
        
        welcome_msg = f"Welcome {user_states[user_id]['owner_name']} to GrowBizz! Your ID: {customer_id}"
        whatsapp_response = send_whatsapp_message(user_id, welcome_msg)
        return f"Registration successful! ID: `{customer_id}`\n{whatsapp_response}"
    except Exception as e:
        logging.error(f"Registration error: {e}")
        return FALLBACK_RESPONSES["register"]

# Purchase Processing
def process_purchase(user_id, data):
    if user_id not in user_states or 'customer_id' not in user_states[user_id]:
        return "Please register first."

    try:
        product = data['product']
        quantity = int(data['quantity'])

        inventory = load_inventory()
        if product not in inventory['Product'].values:
            return f"Product '{product}' not found in inventory."
        
        stock = inventory[inventory['Product'] == product]['Stock'].iloc[0]
        if stock < quantity:
            return f"Insufficient stock for '{product}'. Available: {stock}"

        price = inventory[inventory['Product'] == product]['Price'].iloc[0]
        total_price = price * quantity
        
        # Update inventory and sales
        if update_inventory(product, quantity):
            update_sales_data(product, quantity, price)
        else:
            return FALLBACK_RESPONSES["purchase"]

        # Generate invoice
        invoice_msg = generate_invoice(user_states[user_id]['customer_id'], {'product_name': product, 'quantity': quantity, 'price': price, 'total': total_price}, user_id)
        purchase_msg = f"Purchase confirmed: {quantity} x {product} for INR {total_price}"
        
        # Send WhatsApp message
        whatsapp_response = send_whatsapp_message(user_id, purchase_msg)
        return f"{purchase_msg}\n{invoice_msg}\n{whatsapp_response}"
    except Exception as e:
        logging.error(f"Purchase error: {e}")
        return FALLBACK_RESPONSES["purchase"]

# Invoice Generation
def generate_invoice(customer_id, purchase, user_id):
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
        header = ["Product", "Qty", "Price", "Total"]
        invoice_items = [[purchase['product_name'], purchase['quantity'], purchase['price'], purchase['total']]] if purchase else [["Sample Product", 1, 100.00, 100.00]]

        pdf.add_table(header, invoice_items)
        pdf_bytes = pdf.output(dest='S').encode('latin-1')
        pdf_byte_arr = BytesIO(pdf_bytes)
        pdf_byte_arr.seek(0)
        filename = f"invoice_{customer_id or 'sample'}_{str(uuid.uuid4())[:8]}.pdf"

        dm_channel = get_dm_channel(user_id)
        if dm_channel:
            client.files_upload_v2(
                channels=dm_channel,
                file=pdf_byte_arr,
                filename=filename,
                title="Invoice"
            )
            return "Invoice sent to your DM!"
        return "Invoice generated but DM failed."
    except Exception as e:
        logging.error(f"Invoice error: {e}")
        return FALLBACK_RESPONSES["invoice"]

# Promotion Generation
def generate_promotion(prompt, user_id):
    try:
        img_byte_arr = image_gen.generate_promotion_image(prompt)
        if img_byte_arr:
            dm_channel = get_dm_channel(user_id)
            if dm_channel:
                client.files_upload_v2(
                    channels=dm_channel,
                    file=img_byte_arr,
                    filename=f"promotion_{str(uuid.uuid4())[:8]}.png",
                    title=f"Promotion: {prompt}"
                )
                return f"Promotion sent to your DM!\nText: {prompt}"
        return FALLBACK_RESPONSES["promotion"]
    except Exception as e:
        logging.error(f"Promotion error: {e}")
        return FALLBACK_RESPONSES["promotion"]

# Weekly Sales Analysis
def generate_weekly_sales_analysis(user_id):
    df = load_sales_data()
    if df.empty:
        return FALLBACK_RESPONSES["weekly analysis"]

    # Weekly Sales Graph
    plt.figure(figsize=(10, 5))
    weekly_sales = df.resample('W', on='Order Date')['Price Each'].sum()
    plt.plot(weekly_sales.index, weekly_sales.values, marker='o')
    plt.title("Weekly Sales Trend")
    img_byte_arr1 = BytesIO()
    plt.savefig(img_byte_arr1, format='PNG')
    img_byte_arr1.seek(0)
    plt.close()

    # Overall Sales Graph
    plt.figure(figsize=(10, 5))
    plt.plot(df['Order Date'], df['Price Each'].cumsum(), marker='o')
    plt.title("Overall Sales Trend")
    img_byte_arr2 = BytesIO()
    plt.savefig(img_byte_arr2, format='PNG')
    img_byte_arr2.seek(0)
    plt.close()

    # Normal Distribution Graph
    plt.figure(figsize=(10, 5))
    sales = df['Price Each']
    mu, sigma = sales.mean(), sales.std()
    x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
    plt.plot(x, norm.pdf(x, mu, sigma))
    plt.title("Sales Distribution")
    img_byte_arr3 = BytesIO()
    plt.savefig(img_byte_arr3, format='PNG')
    img_byte_arr3.seek(0)
    plt.close()

    dm_channel = get_dm_channel(user_id)
    if dm_channel:
        for i, img in enumerate([img_byte_arr1, img_byte_arr2, img_byte_arr3], 1):
            client.files_upload_v2(
                channels=dm_channel,
                file=img,
                filename=f"weekly_analysis_{i}_{str(uuid.uuid4())[:8]}.png",
                title=f"Weekly Analysis Graph {i}"
            )
        return "Weekly sales analysis (3 graphs) sent to your DM!"
    return FALLBACK_RESPONSES["weekly analysis"]

# Sales Insights and Recommendations
def generate_sales_insights():
    df = load_sales_data()
    inventory = load_inventory()
    if df.empty or inventory.empty:
        return FALLBACK_RESPONSES["insights"]

    total_sales = df["Price Each"].sum()
    avg_sale = df["Price Each"].mean()
    top_product = df.groupby("Product")["Quantity Ordered"].sum().idxmax()
    weekly_sales = df.resample('W', on='Order Date')['Price Each'].sum()
    trend = "increasing" if weekly_sales.diff().mean() > 0 else "decreasing"
    
    pytrends.build_payload([top_product], timeframe='today 3-m')
    trends = pytrends.interest_over_time()
    seasonal_factor = trends[top_product].mean() / 100 if top_product in trends else 1

    recommendations = []
    for product in inventory['Product']:
        stock = inventory[inventory['Product'] == product]['Stock'].iloc[0]
        price = inventory[inventory['Product'] == product]['Price'].iloc[0]
        sales = df[df['Product'] == product]['Quantity Ordered'].sum()
        weekly_avg = weekly_sales.mean()
        demand_rate = (sales / len(weekly_sales)) * seasonal_factor if sales > 0 else 0.1
        
        if stock < demand_rate:
            recommendations.append(f"Stock up {product} (current: {stock}, suggested: {int(demand_rate * 2)})")
        elif stock > demand_rate * 5:
            recommendations.append(f"Decrease price of {product} (current stock: {stock}, low demand)")
        elif trend == "increasing" and sales > weekly_avg:
            recommendations.append(f"Increase price of {product} (high demand)")

    return f"📊 Sales Insights:\nTotal Sales: INR {total_sales:,.2f}\nAverage Sale: INR {avg_sale:,.2f}\nTop Product: {top_product}\nTrend: {trend}\nRecommendations:\n" + "\n".join(recommendations)

# API Endpoints
@app.route('/slack/events', methods=['POST'])
def slack_events():
    try:
        raw_data = request.data.decode('utf-8')
        logging.info(f"Raw request data: {raw_data}")

        data = request.get_json(silent=True)
        if not data:
            logging.error("Failed to parse JSON from request")
            return Response("Invalid JSON", status=400)

        logging.info(f"Parsed Slack event: {data}")

        if data.get('type') == 'url_verification':
            challenge = data.get('challenge')
            if not challenge:
                logging.error("Challenge parameter missing")
                return Response("Missing challenge", status=400)
            logging.info(f"Returning challenge: {challenge}")
            return jsonify({'challenge': challenge}), 200, {'Content-Type': 'application/json'}

        if 'event' in data and data['event'].get('type') == 'message' and 'text' in data['event']:
            user_id = data['event']['user']
            text = data['event']['text'].lower().strip()
            logging.info(f"Processing message: '{text}' from {user_id}")

            if 'bot_id' in data['event']:
                logging.info(f"Ignoring bot message from {user_id}")
                return Response(status=200)

            if "register:" in text:
                try:
                    _, details = text.split("register:", 1)
                    business_name, owner_name, phone, business_type, email, language, address = [x.strip() for x in details.split(',', 6)]
                    response = handle_customer_registration(user_id, {
                        'business_name': business_name, 'owner_name': owner_name, 'phone': phone, 'business_type': business_type,
                        'email': email, 'language': language, 'address': address
                    })
                except:
                    response = FALLBACK_RESPONSES["register"]
            elif "purchase:" in text:
                try:
                    _, details = text.split("purchase:", 1)
                    product, quantity = [x.strip() for x in details.split(',', 1)]
                    response = process_purchase(user_id, {'product': product, 'quantity': quantity})
                except:
                    response = FALLBACK_RESPONSES["purchase"]
            elif "weekly analysis" in text:
                response = generate_weekly_sales_analysis(user_id)
            elif "insights" in text:
                response = generate_sales_insights()
            elif "promotion:" in text:
                prompt = text.replace("promotion:", "").strip()
                response = generate_promotion(prompt, user_id)
            elif "whatsapp" in text:
                message = text.replace("whatsapp", "").strip() or "Hello from GrowBizz!"
                response = send_whatsapp_message(user_id, message)
            elif "invoice" in text:
                response = generate_invoice(None, None, user_id)
            else:
                try:
                    response = genai.generate_text(prompt=f"Respond to this: {text}").result.strip()
                except:
                    response = FALLBACK_RESPONSES["default"]

            client.chat_postMessage(channel=user_id, text=response)
        return Response(status=200)
    except Exception as e:
        logging.error(f"Slack event processing error: {e}")
        return Response(f"Internal error: {str(e)}", status=500)

if __name__ == "__main__":
    app.run()
