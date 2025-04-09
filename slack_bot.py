# slack_bot.py
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
import image_gen
import numpy as np
from scipy.stats import norm
from pytrends.request import TrendReq

# Configuration and Constants
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Environment Variables
SLACK_BOT_TOKEN = os.environ["SLACK_BOT_TOKEN"]
SLACK_APP_TOKEN = os.environ["SLACK_APP_TOKEN"]
GENAI_API_KEY = os.environ["GENAI_API_KEY"]
TWILIO_ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN", "")
TWILIO_PHONE_NUMBER = os.environ.get("TWILIO_PHONE_NUMBER", "")
SALES_DATA_PATH = os.getenv("SALES_DATA_PATH", "sales_data.csv")
CUSTOMER_DATA_PATH = os.getenv("CUSTOMER_DATA_PATH", "customer_shopping_data.csv")
WHATSAPP_NUMBER = os.getenv("WHATSAPP_NUMBER", "+919944934545")
EMAIL_FROM = os.getenv("EMAIL_FROM", "21z268@psgtech.ac.in")
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD", "")
DEFAULT_LANGUAGE = os.getenv("DEFAULT_LANGUAGE", "English")
INVENTORY_PATH = "inventory.csv"
USERS_PATH = "users.csv"

# Global Variables
user_states = {}
client = WebClient(token=SLACK_BOT_TOKEN)
genai.configure(api_key=GENAI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN) if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN else None
pytrends = TrendReq(hl='en-IN', tz=330)

# Static Fallback Responses
FALLBACK_RESPONSES = {
    "register": "Sorry, registration failed. Please try again later.",
    "purchase": "Purchase could not be processed. Please check back later.",
    "weekly analysis": "Weekly analysis unavailable. Data might be missing.",
    "insights": "Insights generation failed. Please try again.",
    "simple insights": "Simple insights unavailable. Check back soon!",
    "promotion": "Promotion image generation failed. Try again later.",
    "whatsapp": "Failed to send WhatsApp message. Please try again.",
    "invoice": "Sorry, invoice generation failed. Please try again.",
    "default": "Oops! I didn’t understand that. Try: register, purchase, weekly analysis, insights, promotion, whatsapp, invoice, or ask me anything!",
    "not_in_channel": "I can't respond here. Please invite me to this channel with `/invite @YourBotName` or send me a direct message!",
}

# Helper Functions
def get_dm_channel(user_id):
    try:
        response = client.conversations_open(users=user_id)
        dm_channel = response['channel']['id']
        logging.info(f"Opened DM channel {dm_channel} for user {user_id}")
        return dm_channel
    except SlackApiError as e:
        logging.error(f"Failed to open DM channel for {user_id}: {e}")
        return None

def translate_message(text, target_lang):
    try:
        response = model.generate_content(f"Translate this to {target_lang}: {text}")
        return response.text.strip()
    except Exception as e:
        logging.error(f"Translation error: {e}")
        return text

def send_whatsapp_message(user_id, message):
    if not twilio_client:
        return "WhatsApp not configured."
    if user_id not in user_states or 'phone' not in user_states[user_id]:
        return "Please register first to receive WhatsApp messages."
    phone = user_states[user_id]['phone']
    lang = user_states[user_id].get('language', DEFAULT_LANGUAGE)
    translated_msg = translate_message(message, lang)
    try:
        msg = twilio_client.messages.create(
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
    else:
        logging.error(f"Product {product} not found in inventory")

def load_users():
    if os.path.exists(USERS_PATH):
        return pd.read_csv(USERS_PATH)
    else:
        df = pd.DataFrame(columns=["customer_id", "name", "email", "phone", "language", "address"])
        df.to_csv(USERS_PATH, index=False)
        return df

def load_sales_data():
    if os.path.exists(SALES_DATA_PATH):
        df = pd.read_csv(SALES_DATA_PATH, low_memory=False)
        df['Order Date'] = pd.to_datetime(df['Order Date'], format='%m/%d/%y %H:%M', errors='coerce')
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
        purchase_msg = f"Purchase confirmed: {quantity} x {product} for INR {price * quantity}"
        whatsapp_response = send_whatsapp_message(user_id, purchase_msg)
        return f"{purchase_msg}\n{invoice_msg}\n{whatsapp_response}"
    except Exception as e:
        logging.error(f"Purchase error: {e}")
        return FALLBACK_RESPONSES["purchase"]

# Invoice Generation
def generate_invoice(customer_id, product, user_id):
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
        invoice_items = [[product['product_name'], 1, product['price'], product['price']]]

        pdf.add_table(header, invoice_items)
        pdf_bytes = pdf.output(dest='S').encode('latin-1')
        pdf_byte_arr = BytesIO(pdf_bytes)
        pdf_byte_arr.seek(0)
        filename = f"invoice_{customer_id}_{str(uuid.uuid4())[:8]}.pdf"

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
            return FALLBACK_RESPONSES["invoice"]
    except Exception as e:
        logging.error(f"Invoice generation error: {e}")
        return FALLBACK_RESPONSES["invoice"]

# Promotion Generation
def generate_promotion(prompt, user_id):
    try:
        img = Image.new('RGB', (400, 200), color='white')
        d = ImageDraw.Draw(img)
        font = ImageFont.load_default()
        d.text((10, 10), f"Promotion: {prompt}", fill='black', font=font)
        img_byte_arr = BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)

        size = img_byte_arr.getbuffer().nbytes
        logging.info(f"Generated promotion image, size: {size} bytes")

        if size > 0:
            dm_channel = get_dm_channel(user_id)
            if dm_channel:
                client.files_upload_v2(
                    channels=dm_channel,
                    file=img_byte_arr,
                    filename=f"promotion_{str(uuid.uuid4())[:8]}.png",
                    title=f"Promotion: {prompt}"
                )
                logging.info(f"Uploaded promotion image to DM {dm_channel}")
                return f"Promotion generated and sent to your DM!\nText: {prompt}"
            else:
                return "Failed to upload image: DM channel not available."
        else:
            logging.error("Promotion image is empty")
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

    sizes = [img_byte_arr1.getbuffer().nbytes, img_byte_arr2.getbuffer().nbytes, img_byte_arr3.getbuffer().nbytes]
    logging.info(f"Generated weekly sales images, sizes: {sizes}")

    dm_channel = get_dm_channel(user_id)
    if dm_channel:
        for i, img in enumerate([img_byte_arr1, img_byte_arr2, img_byte_arr3], 1):
            client.files_upload_v2(
                channels=dm_channel,
                file=img,
                filename=f"weekly_analysis_{i}_{str(uuid.uuid4())[:8]}.png",
                title=f"Weekly Analysis Graph {i}"
            )
        logging.info(f"Uploaded weekly sales images to DM {dm_channel}")
        return "Weekly sales analysis (3 graphs) sent to your DM!"
    else:
        return "Failed to upload images: DM channel not available."

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
        
        # Weekly trend analysis
        weekly_sales = df.resample('W', on='Order Date')['Price Each'].sum()
        trend = "increasing" if weekly_sales.diff().mean() > 0 else "decreasing"
        
        # Seasonal trends (simplified using pytrends)
        pytrends.build_payload([best_selling], timeframe='today 3-m')
        trends = pytrends.interest_over_time()
        seasonal_factor = trends[best_selling].mean() / 100 if best_selling in trends else 1

        # Mathematical model for recommendations
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

        return f"📊 Sales Insights:\n" \
               f"🔹 Total Sales: INR {total_sales:,.2f}\n" \
               f"🔹 Average Sale: INR {avg_sale:,.2f}\n" \
               f"🔥 Best Selling: {best_selling}\n" \
               f"📈 Trend: {trend}\n" \
               f"Recommendations:\n" + "\n".join(recommendations) if recommendations else "No specific recommendations."
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
    logging.info(f"Processing query: '{text}' for user {user_id}, context: {state['context']}, event_channel: {event_channel}")

    try:
        if "register" in text:
            return handle_customer_registration(user_id, text)
        elif "purchase:" in text:
            return process_purchase(user_id, text)
        elif "weekly analysis" in text:
            state['context'] = 'idle'
            return generate_weekly_sales_analysis(user_id)
        elif "insights" in text or "simple insights" in text:
            state['context'] = 'idle'
            return generate_sales_insights()
        elif "promotion:" in text:
            prompt = text.replace("promotion:", "").strip()
            state['context'] = 'idle'
            return generate_promotion(prompt, user_id)
        elif "whatsapp" in text:
            message = text.replace("whatsapp", "").strip() or "Hello from GrowBizz!"
            state['context'] = 'idle'
            return send_whatsapp_message(user_id, message)
        elif "invoice" in text or "generate invoice" in text:
            state['context'] = 'idle'
            return generate_invoice(None, None, user_id)
        else:
            state['context'] = 'idle'
            return model.generate_content(f"Respond to this user query: {text}").text.strip()
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
