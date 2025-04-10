import os
import logging
import pandas as pd
import uuid
from io import BytesIO
from twilio.rest import Client
import google.generativeai as genai
from google.generativeai.types import GenerateContentConfig
from fpdf import FPDF
from PIL import Image
import numpy as np
from pytrends.request import TrendReq
from flask import Flask, request, jsonify, Response
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
import plotly.express as px

# Set up Flask app
app = Flask(__name__)

# Configure logging for debugging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load environment variables (set these in Vercel dashboard)
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
GENAI_API_KEY = os.getenv("GENAI_API_KEY")
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER", "")

# File paths for CSV data
SALES_DATA_PATH = "sales_data.csv"
INVENTORY_PATH = "inventory.csv"
USERS_PATH = "users.csv"
DEFAULT_LANGUAGE = "English"

# Initialize global clients and states
user_states = {}  # Store user data in memory
slack_client = WebClient(token=SLACK_BOT_TOKEN)
genai.configure(api_key=GENAI_API_KEY)
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN) if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN else None
pytrends = TrendReq(hl='en-IN', tz=330)

# Fallback responses for error handling
FALLBACK_RESPONSES = {
    "register": "Registration failed. Try again later!",
    "purchase": "Purchase failed. Please check back later.",
    "weekly": "Weekly analysis unavailable. No data found.",
    "insights": "Insights generation failed. Try again!",
    "promotion": "Promotion image generation failed. Try again!",
    "whatsapp": "WhatsApp message failed. Try again!",
    "invoice": "Invoice generation failed. Try again!",
    "chart": "Chart generation failed. Try again!",
    "default": "Oops! I didn’t get that. Try: register, purchase, weekly, insights, promotion, whatsapp, invoice, chart"
}

# --- Helper Functions ---

def get_dm_channel(user_id):
    """Open a direct message channel with the user."""
    try:
        response = slack_client.conversations_open(users=user_id)
        return response['channel']['id']
    except SlackApiError as e:
        logging.error(f"Failed to open DM for {user_id}: {e}")
        return None

def translate_message(text, target_lang):
    """Translate text to the target language using Gemini."""
    try:
        response = genai.generate_text(prompt=f"Translate this to {target_lang}: {text}")
        return response.result.strip()
    except Exception as e:
        logging.error(f"Translation error: {e}")
        return text

def send_whatsapp_message(user_id, message):
    """Send a WhatsApp message to the user's registered phone."""
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
        return f"WhatsApp sent to {phone}!"
    except Exception as e:
        logging.error(f"WhatsApp error: {e}")
        return FALLBACK_RESPONSES["whatsapp"]

# --- Data Management ---

def load_csv(file_path, columns):
    """Load or initialize a CSV file with given columns."""
    if os.path.exists(file_path):
        return pd.read_csv(file_path, low_memory=False)
    df = pd.DataFrame(columns=columns)
    df.to_csv(file_path, index=False)
    return df

def load_inventory():
    """Load inventory data."""
    df = load_csv(INVENTORY_PATH, ["Product", "Stock", "Price"])
    df['Stock'] = pd.to_numeric(df['Stock'], errors='coerce').fillna(0).astype(int)
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce').fillna(0.0)
    return df

def load_users():
    """Load user data."""
    return load_csv(USERS_PATH, ["customer_id", "business_name", "owner_name", "phone", "business_type", "email", "language", "address"])

def load_sales_data():
    """Load sales data and ensure proper data types."""
    df = load_csv(SALES_DATA_PATH, ["Order Date", "Product", "Quantity Ordered", "Price Each"])
    df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')
    df.dropna(subset=['Order Date'], inplace=True)
    df['Quantity Ordered'] = pd.to_numeric(df['Quantity Ordered'], errors='coerce').fillna(0).astype(int)
    df['Price Each'] = pd.to_numeric(df['Price Each'], errors='coerce').fillna(0.0)
    return df

def update_inventory(product, quantity):
    """Reduce stock for a product in inventory."""
    df = load_inventory()
    if product in df['Product'].values:
        df.loc[df['Product'] == product, 'Stock'] -= quantity
        df.to_csv(INVENTORY_PATH, index=False)
        return True
    logging.error(f"Product {product} not in inventory")
    return False

def update_sales_data(product, quantity, price):
    """Add a new sale to sales data."""
    df = load_sales_data()
    new_sale = pd.DataFrame({
        "Order Date": [pd.Timestamp.now()],
        "Product": [product],
        "Quantity Ordered": [quantity],
        "Price Each": [price]
    })
    df = pd.concat([df, new_sale], ignore_index=True)
    df.to_csv(SALES_DATA_PATH, index=False)

# --- Feature Functions ---

def handle_registration(user_id, data):
    """Register a new user and store their details."""
    if user_id in user_states and 'customer_id' in user_states[user_id]:
        return f"Already registered with ID: `{user_states[user_id]['customer_id']}`."

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
        
        welcome_msg = f"Welcome {user_states[user_id]['owner_name']} to GrowBizz! ID: {customer_id}"
        whatsapp_response = send_whatsapp_message(user_id, welcome_msg)
        return f"Registration successful! ID: `{customer_id}`\n{whatsapp_response}"
    except Exception as e:
        logging.error(f"Registration error: {e}")
        return FALLBACK_RESPONSES["register"]

def process_purchase(user_id, data):
    """Handle a product purchase, update inventory and sales."""
    if user_id not in user_states or 'customer_id' not in user_states[user_id]:
        return "Please register first."

    try:
        product, quantity = data['product'], int(data['quantity'])
        inventory = load_inventory()
        
        if product not in inventory['Product'].values:
            return f"Product '{product}' not found."
        stock = inventory[inventory['Product'] == product]['Stock'].iloc[0]
        if stock < quantity:
            return f"Only {stock} of '{product}' available."
        
        price = inventory[inventory['Product'] == product]['Price'].iloc[0]
        total_price = price * quantity
        
        if update_inventory(product, quantity):
            update_sales_data(product, quantity, price)
            invoice_msg = generate_invoice(user_states[user_id]['customer_id'], 
                                         {'product_name': product, 'quantity': quantity, 'price': price, 'total': total_price}, 
                                         user_id)
            purchase_msg = f"Purchase confirmed: {quantity} x {product} for INR {total_price}"
            whatsapp_response = send_whatsapp_message(user_id, purchase_msg)
            return f"{purchase_msg}\n{invoice_msg}\n{whatsapp_response}"
        return FALLBACK_RESPONSES["purchase"]
    except Exception as e:
        logging.error(f"Purchase error: {e}")
        return FALLBACK_RESPONSES["purchase"]

def generate_invoice(customer_id, purchase, user_id):
    """Generate and upload an invoice PDF to Slack."""
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
        items = [[purchase['product_name'], purchase['quantity'], purchase['price'], purchase['total']]] if purchase else [["Sample", 1, 100.0, 100.0]]
        pdf.add_table(header, items)
        pdf_bytes = pdf.output(dest='S').encode('latin-1')
        pdf_byte_arr = BytesIO(pdf_bytes)
        pdf_byte_arr.seek(0)
        
        dm_channel = get_dm_channel(user_id)
        if dm_channel:
            slack_client.files_upload_v2(
                channels=dm_channel,
                file=pdf_byte_arr,
                filename=f"invoice_{customer_id or 'sample'}_{uuid.uuid4().hex[:8]}.pdf",
                title="Invoice"
            )
            return "Invoice sent to your DM!"
        return "Invoice generated but DM failed."
    except Exception as e:
        logging.error(f"Invoice error: {e}")
        return FALLBACK_RESPONSES["invoice"]

def generate_promotion(prompt, user_id):
    """Generate a promotion image with Gemini 2.0 and upload to Slack."""
    try:
        response = genai.generate_content(
            model="gemini-2.0-flash-exp-image-generation",
            prompt=prompt,
            generation_config=GenerateContentConfig(response_mime_type="image/png")
        )
        for candidate in response.candidates:
            for part in candidate.content.parts:
                if hasattr(part, 'inline_data') and part.inline_data:
                    img_data = part.inline_data.data
                    img = Image.open(BytesIO(img_data))
                    img.save('promotion_image.png')  # Save locally
                    
                    img_byte_arr = BytesIO()
                    img.save(img_byte_arr, format='PNG')
                    img_byte_arr.seek(0)
                    
                    dm_channel = get_dm_channel(user_id)
                    if dm_channel:
                        slack_client.files_upload_v2(
                            channels=dm_channel,
                            file=img_byte_arr,
                            filename="promotion_image.png",
                            title=f"Promotion: {prompt}"
                        )
                        return f"Promotion image sent to your DM!\nText: {prompt}"
                    return "Promotion generated but DM failed."
        return FALLBACK_RESPONSES["promotion"]
    except Exception as e:
        logging.error(f"Promotion error: {e}")
        return FALLBACK_RESPONSES["promotion"]

def generate_weekly_analysis(user_id):
    """Generate and upload weekly sales analysis charts to Slack."""
    df = load_sales_data()
    if df.empty:
        return FALLBACK_RESPONSES["weekly"]

    try:
        # Weekly Sales Trend
        weekly_sales = df.resample('W', on='Order Date')['Price Each'].sum().reset_index()
        fig1 = px.line(weekly_sales, x='Order Date', y='Price Each', title="Weekly Sales Trend",
                       labels={'Price Each': 'Sales (INR)'}, template='plotly_white', color_discrete_sequence=['#4CAF50'])
        img_byte_arr1 = BytesIO()
        fig1.write_image(img_byte_arr1, format='png')
        img_byte_arr1.seek(0)

        # Overall Sales Trend
        overall_sales = df.copy()
        overall_sales['Cumulative'] = overall_sales['Price Each'].cumsum()
        fig2 = px.line(overall_sales, x='Order Date', y='Cumulative', title="Overall Sales Trend",
                       labels={'Cumulative': 'Total Sales (INR)'}, template='plotly_white', color_discrete_sequence=['#2196F3'])
        img_byte_arr2 = BytesIO()
        fig2.write_image(img_byte_arr2, format='png')
        img_byte_arr2.seek(0)

        # Sales Distribution
        fig3 = px.histogram(df, x='Price Each', nbins=30, title="Sales Distribution",
                            labels={'Price Each': 'Sales (INR)'}, template='plotly_white', color_discrete_sequence=['#FF9800'])
        fig3.update_traces(opacity=0.75)
        img_byte_arr3 = BytesIO()
        fig3.write_image(img_byte_arr3, format='png')
        img_byte_arr3.seek(0)

        dm_channel = get_dm_channel(user_id)
        if dm_channel:
            for i, img in enumerate([img_byte_arr1, img_byte_arr2, img_byte_arr3], 1):
                slack_client.files_upload_v2(
                    channels=dm_channel,
                    file=img,
                    filename=f"weekly_{i}_{uuid.uuid4().hex[:8]}.png",
                    title=f"Weekly Analysis {i}"
                )
            return "Weekly analysis (3 charts) sent to your DM!"
        return FALLBACK_RESPONSES["weekly"]
    except Exception as e:
        logging.error(f"Weekly analysis error: {e}")
        return FALLBACK_RESPONSES["weekly"]

def generate_sales_insights():
    """Generate sales insights and recommendations."""
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
        demand_rate = (sales / len(weekly_sales)) * seasonal_factor if sales > 0 else 0.1
        if stock < demand_rate:
            recommendations.append(f"Stock up {product} (current: {stock})")
        elif stock > demand_rate * 5:
            recommendations.append(f"Decrease price of {product} (stock: {stock})")

    return f"📊 Insights:\nTotal Sales: INR {total_sales:,.2f}\nAvg Sale: INR {avg_sale:,.2f}\nTop Product: {top_product}\nTrend: {trend}\nRecommendations:\n" + "\n".join(recommendations)

def generate_bar_chart(user_query, user_id):
    """Generate a bar chart based on user query and upload to Slack."""
    df = load_sales_data()
    if df.empty:
        return FALLBACK_RESPONSES["chart"]

    try:
        if "shoe brand" in user_query.lower():
            sales_by_brand = df.groupby('Product')['Price Each'].sum().reset_index()
            fig = px.bar(sales_by_brand, x='Product', y='Price Each', title="Sales by Shoe Brand",
                         labels={'Price Each': 'Sales (INR)'}, template='plotly_white', color='Product', bargap=0.2)
            img_byte_arr = BytesIO()
            fig.write_image(img_byte_arr, format='png')
            img_byte_arr.seek(0)
            
            dm_channel = get_dm_channel(user_id)
            if dm_channel:
                slack_client.files_upload_v2(
                    channels=dm_channel,
                    file=img_byte_arr,
                    filename=f"chart_{uuid.uuid4().hex[:8]}.png",
                    title="Sales by Shoe Brand"
                )
                return "Bar chart sent to your DM!"
            return FALLBACK_RESPONSES["chart"]
        return "Use 'shoe brand' in your chart request (e.g., 'chart create a bar chart for the sales by shoe brand')."
    except Exception as e:
        logging.error(f"Chart error: {e}")
        return FALLBACK_RESPONSES["chart"]

# --- Slack Event Handler ---

@app.route('/slack/events', methods=['POST'])
def slack_events():
    """Handle Slack events and commands."""
    try:
        data = request.get_json(silent=True)
        if not data:
            logging.error("Invalid JSON")
            return Response("Invalid JSON", status=400)

        if data.get('type') == 'url_verification':
            return jsonify({'challenge': data.get('challenge')}), 200

        if 'event' in data and data['event'].get('type') == 'message' and 'text' in data['event']:
            user_id = data['event']['user']
            text = data['event']['text'].lower().strip()
            if 'bot_id' in data['event']:
                return Response(status=200)

            if "register:" in text:
                try:
                    _, details = text.split("register:", 1)
                    parts = [x.strip() for x in details.split(',', 6)]
                    response = handle_registration(user_id, {
                        'business_name': parts[0], 'owner_name': parts[1], 'phone': parts[2], 'business_type': parts[3],
                        'email': parts[4], 'language': parts[5], 'address': parts[6]
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
            elif "weekly" in text:
                response = generate_weekly_analysis(user_id)
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
            elif "chart" in text:
                response = generate_bar_chart(text, user_id)
            else:
                response = genai.generate_text(prompt=f"Respond to: {text}").result.strip() or FALLBACK_RESPONSES["default"]

            slack_client.chat_postMessage(channel=user_id, text=response)
        return Response(status=200)
    except Exception as e:
        logging.error(f"Slack event error: {e}")
        return Response(f"Error: {str(e)}", status=500)

if __name__ == "__main__":
    app.run()
