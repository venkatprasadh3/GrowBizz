    import os
    import re
    import logging
    import pandas as pd
    from slack_sdk import WebClient
    from slack_sdk.errors import SlackApiError
    import uuid
    import matplotlib
    matplotlib.use('Agg')
    import threading
    from http.server import HTTPServer, BaseHTTPRequestHandler
    import requests
    import json
    import time
    import tempfile
    import cloudinary.uploader
    from deep_translator import GoogleTranslator
    from google import genai
    from google.genai import types
    from PIL import Image
    from io import BytesIO
    import os
    import cloudinary.uploader
    import cloudinary
    from twilio.rest import Client
    import io
    import plotly.express as px
    import datetime
    import mimetypes
    import pytrends
    from pytrends.request import TrendReq
    import numpy as np
    from scipy.stats import norm
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Configuration
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN")
    SLACK_APP_TOKEN = os.environ.get("SLACK_APP_TOKEN")
    TWILIO_ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID")
    TWILIO_AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN")
    TWILIO_PHONE_NUMBER = os.environ.get("TWILIO_PHONE_NUMBER", "+14155238886")
    CLOUDINARY_CLOUD_NAME = os.environ.get("CLOUDINARY_CLOUD_NAME")
    CLOUDINARY_API_KEY = os.environ.get("CLOUDINARY_API_KEY")
    CLOUDINARY_API_SECRET = os.environ.get("CLOUDINARY_API_SECRET")
    GENAI_API_KEY = os.environ.get("GENAI_API_KEY")
    DEFAULT_LANGUAGE = "English"

    # Validate environment variables
    required_vars = [
        "SLACK_BOT_TOKEN", "SLACK_APP_TOKEN",
        "TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN",
        "CLOUDINARY_CLOUD_NAME", "CLOUDINARY_API_KEY", "CLOUDINARY_API_SECRET",
        "GENAI_API_KEY"
    ]
    for var in required_vars:
        if not os.environ.get(var):
            logging.error(f"Missing environment variable: {var}")
            raise ValueError(f"Environment variable {var} is required")

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    USERS_PATH = os.path.join(BASE_DIR, "users.csv")
    SALES_DATA_PATH = os.path.join(BASE_DIR, "sales_data.csv")
    SALES_DATA_RAW_PATH = os.path.join(BASE_DIR, "sales_data_raw.csv")
    INVENTORY_PATH = os.path.join(BASE_DIR, "inventory.csv")

    user_states = {}
    client = WebClient(token=SLACK_BOT_TOKEN)
    twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN) if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN else None
    processed_events = set()
    response_cache = {}

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
        "default": "Oops! I didn’t understand that. Try: register, purchase, weekly analysis, insights, promotion, whatsapp, invoice, chart, summarize call, or ask me anything! 🤔",
        "not_in_channel": "I can't respond here. Please invite me to this channel with `/invite @GrowBizz` or send me a DM! 🚪",
        "audio": "Audio processing failed. Please try again later. 🎙️",
        "whatsapp_media": "Failed to send media via WhatsApp. Please try again. 📱"
    }

    # Helper Functions
    def get_user_language(user_id):
        users_df = load_users()
        user = users_df[users_df['slack_id'] == user_id].iloc[0] if not users_df[
            users_df['slack_id'] == user_id].empty else None
        return user['language'] if user else DEFAULT_LANGUAGE


    def translate_message(text, target_lang): 
        try:
            if target_lang.lower() == "english":
                return text
            translated = GoogleTranslator(source='auto', target=target_lang.lower()).translate(text)
            return translated
        except Exception as e:
            logging.error(f"Translation error: {e}")
            return text

    def send_whatsapp_message(user_id, invoice_url):
        if not twilio_client:
            logging.error("WhatsApp not configured: Twilio credentials missing.")
            return FALLBACK_RESPONSES["whatsapp"]
        if user_id not in user_states or 'phone' not in user_states[user_id]:
            logging.error(f"WhatsApp failed for {user_id}: User not registered.")
            return FALLBACK_RESPONSES["whatsapp"]
        customer = user_states[user_id]
        phone = customer['phone']
        name = customer['name']
        if not re.match(r'^\+\d{10,15}$', phone):
            logging.error(f"Invalid phone number for {user_id}: {phone}")
            return "Invalid phone number format. Please register with a valid number. 📱"
        recipients = [f"whatsapp:{phone}"]
        twilio_whatsapp_number = f"whatsapp:{TWILIO_PHONE_NUMBER}"
        caption_text = f"Hey {name} 👋, thank you for your latest purchase with Smart Shoes 👟. Heres your invoice against your order A35432. Visit us again. We have exciting discounts only for you!"
        try:
            results = []
            for recipient in recipients:
                message = twilio_client.messages.create(
                    media_url=[invoice_url],
                    from_=twilio_whatsapp_number,
                    to=recipient,
                    body=caption_text
                )
                results.append(f"WhatsApp message SID: {message.sid} to {recipient}")
                logging.info(f"WhatsApp message SID: {message.sid}, Status: {message.status}, To: {recipient}")
            return "WhatsApp message sent successfully! 📱 " + "; ".join(results)
        except Exception as e:
            logging.error(f"WhatsApp error for {user_id}: {e}")
            return FALLBACK_RESPONSES["whatsapp"]


    # **Customer Registration 📝**
    def handle_customer_registration(user_id, text):
        if user_id not in user_states:
            user_states[user_id] = {'last_message': '', 'context': 'idle'}
        state = user_states[user_id]
        if 'customer_id' in state:
            return {
                "blocks": [
                    {"type": "header", "text": {"type": "plain_text", "text": "Registration Status 📝"}},
                    {"type": "section", "text": {"type": "mrkdwn",
                                                "text": f"*User is already registered* with Customer ID: `{state['customer_id']}`."}}
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
                response = {
                    "blocks": [
                        {"type": "header", "text": {"type": "plain_text", "text": "Registration Successful ✅"}},
                        {"type": "section",
                        "text": {"type": "mrkdwn", "text": f"Welcome, {name}! Customer ID: `{customer_id}`"}}
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
                {"type": "section",
                "text": {"type": "mrkdwn", "text": "Please use: `register: name, email, phone, language, address`"}}
            ],
            "text": "Registration help provided."
        }


    # **Invoice Generation 📄**
    def generate_invoice(user_id, event_channel):
        try:
            if user_id not in user_states or 'customer_id' not in user_states[user_id]:
                return {
                    "blocks": [
                        {"type": "header", "text": {"type": "plain_text", "text": "Invoice Generation 📄"}},
                        {"type": "section", "text": {"type": "mrkdwn", "text": "Please register first. 🙁"}}
                    ],
                    "text": "Invoice failed: user not registered."
                }
            invoice_url = "https://res.cloudinary.com/dnnj6hykk/image/upload/v1744547940/ivoice-test-GED_2_gmxls7.pdf"
            invoice_data = requests.get(invoice_url).content
            client.files_upload_v2(
                channel=event_channel,
                file=BytesIO(invoice_data),
                filename="invoice_A35432.pdf",
                title="Invoice A35432",
                initial_comment="Your invoice has been generated."
            )
            whatsapp_response = send_whatsapp_message(user_id, invoice_url)
            response = {
                "blocks": [
                    {"type": "header", "text": {"type": "plain_text", "text": "Invoice Generated 📄"}},
                    {"type": "section", "text": {"type": "mrkdwn",
                                                "text": f"Invoice uploaded to this channel!\nURL: {invoice_url}\n{whatsapp_response}"}}
                ],
                "text": "Invoice generated."
            }
            logging.info(f"Invoice generated for {user_id}: {invoice_url}")
            return response
        except Exception as e:
            logging.error(f"Invoice error for {user_id}: {e}")
            return {
                "blocks": [
                    {"type": "header", "text": {"type": "plain_text", "text": "Invoice Generation 📄"}},
                    {"type": "section", "text": {"type": "mrkdwn", "text": FALLBACK_RESPONSES["invoice"]}}
                ],
                "text": "Invoice generation failed."
            }


    def generate_promotion(user_id, event_channel, text):
        try:
            if user_id not in user_states or 'customer_id' not in user_states[user_id]:
                return {
                    "blocks": [
                        {"type": "header", "text": {"type": "plain_text", "text": "Promotion Generation 🖼️"}},
                        {"type": "section", "text": {"type": "mrkdwn", "text": "Please register first. 🙁"}}
                    ],
                    "text": "Promotion failed: user not registered."
                }
            customer = user_states[user_id]
            image_client = genai.Client(api_key=GENAI_API_KEY)
            contents = text
            response = image_client.models.generate_content(
                model="gemini-2.0-flash-exp-image-generation",
                contents=contents,
                config=types.GenerateContentConfig(
                    response_modalities=['Text', 'Image']
                )
            )
            generated_image_data = None
            for part in response.candidates[0].content.parts:
                if part.inline_data is not None:
                    generated_image_data = part.inline_data.data
                    break

            if generated_image_data:
                image = Image.open(BytesIO(generated_image_data))
                # Cloudinary Upload
                cloudinary.config(
                    cloud_name=CLOUDINARY_CLOUD_NAME,
                    api_key=CLOUDINARY_API_KEY,
                    api_secret=CLOUDINARY_API_SECRET,
                    secure=True
                )
                public_id = f"Smart_Shoes_test_{uuid.uuid4().hex[:8]}"
                upload_result = cloudinary.uploader.upload(
                    BytesIO(generated_image_data),
                    resource_type="image",
                    public_id=public_id
                )
                image_url = upload_result["secure_url"]
                # Upload to Slack
                client.files_upload_v2(
                    channel=event_channel,
                    file=BytesIO(generated_image_data),
                    filename="promotion_poster.png",
                    title="Smart Shoes Promotion",
                    initial_comment="Your promotion poster has been generated."
                )
                # Twilio WhatsApp
                twilio_client_specific = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
                recipients = [f"whatsapp:{customer['phone']}"]
                twilio_whatsapp_number = f"whatsapp:{TWILIO_PHONE_NUMBER}"
                caption_text = "Sure, Here's the poster that you requested"
                results = []
                for recipient in recipients:
                    message = twilio_client_specific.messages.create(
                        media_url=[image_url],
                        from_=twilio_whatsapp_number,
                        to=recipient,
                        body=caption_text
                    )
                    results.append(f"WhatsApp message SID: {message.sid} to {recipient}")
                    logging.info(f"WhatsApp message SID: {message.sid}, Status: {message.status}, To: {recipient}")
                whatsapp_response = "WhatsApp message sent successfully! 📱 " + "; ".join(results)
                response = {
                    "blocks": [
                        {"type": "header", "text": {"type": "plain_text", "text": "Promotion Generated 🖼️"}},
                        {"type": "section", "text": {"type": "mrkdwn",
                                                    "text": f"Promotion poster uploaded to this channel!\nURL: {image_url}\n{whatsapp_response}"}}
                    ],
                    "text": "Promotion generated."
                }
                logging.info(f"Promotion generated for {user_id}: {image_url}")
                return response
            else:
                logging.error(f"No image data received from Gemini for {user_id}")
                return {
                    "blocks": [
                        {"type": "header", "text": {"type": "plain_text", "text": "Promotion Generation 🖼️"}},
                        {"type": "section", "text": {"type": "mrkdwn", "text": FALLBACK_RESPONSES["promotion"]}}
                    ],
                    "text": "Promotion generation failed."
                }
        except Exception as e:
            logging.error(f"Promotion error for {user_id}: {e}")
            return {
                "blocks": [
                    {"type": "header", "text": {"type": "plain_text", "text": "Promotion Generation 🖼️"}},
                    {"type": "section", "text": {"type": "mrkdwn", "text": FALLBACK_RESPONSES["promotion"]}}
                ],
                "text": "Promotion generation failed."
            }


    def process_audio(audio_file_path: str, prompt: str) -> str:
        try:
            audio_client = genai.Client(api_key=GENAI_API_KEY)

            # Guess the MIME type
            mime_type, _ = mimetypes.guess_type(audio_file_path)
            if not mime_type:
                mime_type = "audio/m4a"

            myfile = audio_client.files.upload(
                file=audio_file_path,
                mime_type=mime_type 
            )

            contents = [prompt, myfile]

            response = audio_client.models.generate_content(
                model="gemini-2.0-flash",
                contents=contents
            )

            return response.text
        except Exception as e:
            return f"An error occurred: {e}"

    trend_cache = {}

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
            df = pd.DataFrame(columns=["customer_id", "name", "email", "phone", "language", "address"])
            df.to_csv(USERS_PATH, index=False)
            return df

    def load_sales_raw_data():
        try:
            if os.path.exists(SALES_DATA_RAW_PATH):
                df = pd.read_csv(SALES_DATA_RAW_PATH)
                df['Price Each'] = pd.to_numeric(df['Price Each'], errors='coerce')
                df['Order Date'] = pd.to_datetime(df['Order Date'], format='%m/%d/%y %H:%M', errors='coerce').fillna(
                    pd.to_datetime(df['Order Date'], format='%m-%d-%Y %H:%M', errors='coerce')
                )
                logging.info(f"Loaded sales data: {df.head().to_string()}")
                return df.dropna(subset=['Order Date', 'Price Each'])
            else:
                df = pd.DataFrame(columns=["Order ID", "Product", "Quantity Ordered", "Price Each", "Order Date", "Purchase Address"])
                df.to_csv(SALES_DATA_RAW_PATH, index=False)
                return df
        except Exception as e:
            logging.error(f"Error loading sales data: {e}")
            return pd.DataFrame()
            
    def load_sales_data():
        try:
            if os.path.exists(SALES_DATA_PATH):
                df = pd.read_csv(SALES_DATA_PATH)
                df['Price Each'] = pd.to_numeric(df['Price Each'], errors='coerce')
                df['Order Date'] = pd.to_datetime(
                    df['Order Date'],
                    errors='coerce',
                    format='%Y-%m-%d %H:%M'
                ).fillna(
                    pd.to_datetime(df['Order Date'], format='%m/%d/%y %H:%M', errors='coerce')
                ).fillna(
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
            april_df = df[(df['Order Date'].dt.month == 4) & (df['Order Date'])]
            if april_df.empty:
                latest_month = df['Order Date'].dt.to_period('M').max()
                april_df = df[df['Order Date'].dt.to_period('M') == latest_month]
                logging.info(f"No data for April 2019 for {user_id}, using {latest_month}")
                month_text = f"{latest_month.strftime('%B %Y')} (No April 2019 data)"
            else:
                month_text = "April"
                logging.info(f"April data for {user_id}: {april_df.shape[0]} rows")

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
                f"Total Sales: ₹{total_sales:,.2f}\n"
                f"Average Daily Sales: ₹{avg_daily_sales:,.2f}\n"
                f"Average Weekly Sales: ₹{avg_weekly_sales:,.2f}\n"
                f"Best Selling Product: {best_selling_product} 🔥\n"
                f"GrowBizz Trend Score: {trend_score:.2f}"
            )
            recommendations_text = ""
            if recommendations["decrease"] or recommendations["increase"] or recommendations["restock"]:
                recommendations_text += "Recommendations 📋\n"
                if recommendations["decrease"]:
                    recommendations_text += "🔽 Price Decreases:\n" + "\n".join(f"• {r}" for r in recommendations["decrease"][:3]) + "\n"
                if recommendations["increase"]:
                    recommendations_text += "🔼 Price Increases:\n" + "\n".join(f"• {r}" for r in recommendations["increase"][:3]) + "\n"
                if recommendations["restock"]:
                    recommendations_text += "📦 Restock:\n" + "\n".join(f"• {r}" for r in recommendations["restock"][:3]) + "\n"
            else:
                recommendations_text += "Recommendations 📋\nNo specific recommendations available. 🙁"

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
                    {"type": "header", "text": {"type": "plain_text", "text": "Sales Insights for April 📊"}},
                    {"type": "section", "text": {"type": "mrkdwn", "text": FALLBACK_RESPONSES["insights"]}}
                ],
                "text": "Sales insights failed."
            }

    # Weekly Sales Analysis
    def generate_weekly_sales_analysis(user_id, event_channel):
        df = load_sales_raw_data()
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

    PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")

    def extract_python_code(llm_response):
        match = re.search(r"```python\n(.*?)\n```", llm_response, re.DOTALL)
        return match.group(1) if match else llm_response

    def fetch_csv_content(csv_path):
        try:
            with open(csv_path, 'r') as f:
                return f.read()
        except FileNotFoundError:
            print(f"Error: CSV file not found at path: {csv_path}")
            return None
        except Exception as e:
            print(f"Error reading CSV file: {e}")
            return None

    def process_csv_and_query(csv_path, user_query):
        
        try:
            csv_content = fetch_csv_content(csv_path)
            if not csv_content:
                return None
            df = pd.read_csv(io.StringIO(csv_content))
            df_summary = df.describe().to_string()
            df_head = df.to_string()

            prompt = f"""
            You are a data analysis assistant. CSV data:

            Summary:
            {df_summary}

            All Rows:
            {df_head}

            User query: {user_query}

            Return plotly express code for a visualization. Make plots professional, sleek, and attractive with proper bar spacing for bar graphs. Return only the code.
            """
            response = client.models.generate_content(
                model='gemini-2.0-flash',
                contents=prompt,
            )
            llm_response = response.text

            if "px." in llm_response.lower():
                code_to_execute = extract_python_code(llm_response)
                local_vars = {"df": df, "px": px}
                exec(code_to_execute, globals(), local_vars)
                fig = local_vars.get("fig")
                if fig:
                    os.makedirs(PLOTS_DIR, exist_ok=True)
                    plot_filename = os.path.join(PLOTS_DIR, f"plot_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
                    fig.write_image(plot_filename, format="png")
                    image_url = cloudinary.uploader.upload(
                        plot_filename,
                        resource_type="image",
                        public_id=f"plot_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    )["secure_url"]
                    print(f"Plot saved as: {plot_filename}, Uploaded to: {image_url}")
                    return image_url
            return None
        except Exception as e:
            print(f"Error processing visualization: {e}")
            return None
        
    csv_path = SALES_DATA_PATH
    # Query Processing
    def generate_plots(user_id, event_channel, text):
        try:
            if user_id not in user_states or 'customer_id' not in user_states[user_id]:
                return {
                    "blocks": [
                        {"type": "header", "text": {"type": "plain_text", "text": "Plots Generation 🖼️"}},
                        {"type": "section", "text": {"type": "mrkdwn", "text": "Please register first. 🙁"}}
                    ],
                    "text": "Plot failed: user not registered."
                }
            customer = user_states[user_id]
            plots_url=process_csv_and_query(csv_path, text)
            if plots_url:
                client.files_upload_v2(
                    channel=event_channel,
                    file=BytesIO(plots_url),
                    filename="visualization.png",
                    title="Visualization",
                    initial_comment="Your visualization has been generated."
                )
        except Exception as e:
            return {}
    

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
            if "register" in text:
                response = handle_customer_registration(user_id, text)
            elif "invoice" in text:
                response = generate_invoice(user_id, event_channel)
            elif "promotion" in text:
                response = generate_promotion(user_id, event_channel, text)
            elif "chart" in text:
                response = generate_plots(user_id, event_channel, text)
            elif "insights" in text:
                response = generate_sales_insights(user_id)
            elif "weekly analysis" in text:
                response = generate_weekly_sales_analysis(user_id, event_channel)
            elif any(greeting in text for greeting in ["hello", "hi", "hey", "how are you", "good morning"]):
                try:
                    genai.configure(api_key=GENAI_API_KEY)
                    model = genai.GenerativeModel("gemini-2.0-flash")
                    prompt = f"Respond to this casual message in a friendly, professional tone: '{text}'"
                    gen_response = model.generate_content(prompt).text
                    response = {
                        "blocks": [
                            {"type": "section", "text": {"type": "mrkdwn", "text": gen_response}}
                        ],
                        "text": gen_response
                    }
                except Exception as e:
                    logging.error(f"Casual message processing error for {user_id}: {e}")
                    response = {
                        "blocks": [
                            {"type": "section", "text": {"type": "mrkdwn", "text": "Hey there! I'm here to help with your shopping needs. What's up? 😊"}}
                        ],
                        "text": "Friendly greeting response"
                    }
            else:
                response = {
                    "blocks": [
                        {"type": "section", "text": {"type": "mrkdwn", "text": FALLBACK_RESPONSES["default"]}}
                    ],
                    "text": "Query not recognized."
                }

            response_cache[cache_key] = {'response': response, 'time': current_time}
            client.chat_postMessage(
                channel=event_channel,
                blocks=response["blocks"],
                text=response.get("text", "GrowBizz response")
            )
            logging.info(f"Response sent to {event_channel} for {user_id}: {response['text']}")
            return response
        except Exception as e:
            logging.error(f"Query processing error for {user_id}: {e}")
            response = {
                "blocks": [
                    {"type": "section", "text": {"type": "mrkdwn", "text": FALLBACK_RESPONSES["default"]}}
                ],
                "text": "Query processing failed."
            }
            client.chat_postMessage(
                channel=event_channel,
                blocks=response["blocks"],
                text=response["text"]
            )
            return response

    # HTTP Server
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
            
        @slack_app.event("app_mention")
        def handle_message_with_mention_and_file(event, client, logger):
            text = event.get("text", "")
            channel_id = event["channel"]
            user_id = event["user"]
            files = event.get("files", [])
            bot_user_id = client.auth_test().get("user_id")

            if bot_user_id in text:  # Check if the bot is mentioned
                if files:
                    for file in files:
                        if file.get("mimetype", "").startswith("audio/"):
                            file_id = file["id"]
                            try:
                                file_info = client.files_info(file=file_id)
                                file_details = file_info["file"]
                                download_url = file_details["url_private_download"]
                                headers = {"Authorization": f"Bearer {SLACK_BOT_TOKEN}"}
                                response = requests.get(download_url, headers=headers, stream=True)
                                response.raise_for_status()

                                with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                                    for chunk in response.iter_content(chunk_size=8192):
                                        tmp_file.write(chunk)
                                    local_audio_path = tmp_file.name

                                # Try to get a prompt from the message text
                                prompt = text.replace(f"<@{bot_user_id}>", "").strip()

                                if prompt:
                                    client.chat_postMessage(
                                        channel=channel_id,
                                        text=f"Processing audio with the prompt: '{prompt}'...",
                                        thread_ts=event.get("thread_ts")
                                    )
                                    processed_text = process_audio(local_audio_path, prompt)
                                    client.chat_postMessage(
                                        channel=channel_id,
                                        text=f"Processed audio output:\n{processed_text}",
                                        thread_ts=event.get("thread_ts")
                                    )
                                else:
                                    client.chat_postMessage(
                                        channel=channel_id,
                                        text="Audio file received with a mention, but no specific prompt was provided.",
                                        thread_ts=event.get("thread_ts")
                                    )
                                os.remove(local_audio_path)
                            except requests.exceptions.RequestException as e:
                                logger.error(f"Error downloading file {file_id}: {e}")
                                client.chat_postMessage(channel=channel_id, text=f"Error downloading the audio file.",
                                                        thread_ts=event.get("thread_ts"))
                            except SlackApiError as e:
                                logger.error(f"Slack API error for file {file_id}: {e}")
                                client.chat_postMessage(channel=channel_id, text=f"Error accessing file information.",
                                                        thread_ts=event.get("thread_ts"))
                            except Exception as e:
                                logger.error(f"An error occurred processing file {file_id}: {e}")
                                client.chat_postMessage(channel=channel_id,
                                                        text=f"An error occurred while processing the audio file.",
                                                        thread_ts=event.get("thread_ts"))
                        # You can add 'else if' conditions here to handle other file types
                else:
                    # If the bot is mentioned but no files are attached, you can respond to the text mention
                    client.chat_postMessage(
                        channel=channel_id,
                        text=f"Hi <@{user_id}>! You mentioned me. If you want me to process an audio file, please attach it to your message.",
                        thread_ts=event.get("thread_ts")
                    )


        @slack_app.event("file_shared")
        def handle_file_shared(event, client, logger):
            file_id = event["file_id"]
            channel_id = event["channel_id"]
            try:
                file_info = client.files_info(file=file_id)
                file = file_info["file"]
                if file.get("mimetype", "").startswith("audio/"):
                    download_url = file["url_private_download"]
                    headers = {"Authorization": f"Bearer {SLACK_BOT_TOKEN}"}
                    response = requests.get(download_url, headers=headers, stream=True)
                    response.raise_for_status()
                    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                        for chunk in response.iter_content(chunk_size=8192):
                            tmp_file.write(chunk)
                        local_audio_path = tmp_file.name
                    prompt = file.get("initial_comment", {}).get("comment", "")
                    if prompt:
                        client.chat_postMessage(
                            channel=channel_id,
                            text=f"Processing audio with the prompt: '{prompt}'...",
                            thread_ts=event.get("thread_ts")
                        )
                        processed_text = process_audio(local_audio_path, prompt)
                        client.chat_postMessage(
                            channel=channel_id,
                            text=f"Processed audio output:\n{processed_text}",
                            thread_ts=event.get("thread_ts")
                        )
                    else:
                        client.chat_postMessage(
                            channel=channel_id,
                            text="Audio file received, but no prompt was provided in the caption.",
                            thread_ts=event.get("thread_ts")
                        )
                    os.remove(local_audio_path)
                else:
                    logger.info(f"Received a non-audio file: {file.get('mimetype')}")
            except Exception as e:
                logger.error(f"Error responding to file shared: {e}")


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
                logging.error(f"Slack API error for {user_id}: {e}")


        threading.Thread(target=run_http_server, daemon=True).start()
        SocketModeHandler(slack_app, SLACK_APP_TOKEN).start()
