import os
import logging
import json
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from twilio.rest import Client
import cloudinary
import cloudinary.uploader
import pandas as pd
import plotly.express as px
import io
import datetime
import re
from urllib import request
from PIL import Image
from io import BytesIO
import google.generativeai as genai
from google.generativeai.types import GenerateContentConfig

# Configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN")
TWILIO_ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.environ.get("TWILIO_PHONE_NUMBER")
CLOUDINARY_CLOUD_NAME = os.environ.get("CLOUDINARY_CLOUD_NAME")
CLOUDINARY_API_KEY = os.environ.get("CLOUDINARY_API_KEY")
CLOUDINARY_API_SECRET = os.environ.get("CLOUDINARY_API_SECRET")
GENAI_API_KEY = os.environ.get("GENAI_API_KEY")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
SALES_DATA_PATH = os.path.join(BASE_DIR, "sales_data.csv")
USERS_PATH = os.path.join(BASE_DIR, "users.csv")

os.makedirs(PLOTS_DIR, exist_ok=True)

slack_client = WebClient(token=SLACK_BOT_TOKEN)
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
cloudinary.config(
    cloud_name=CLOUDINARY_CLOUD_NAME,
    api_key=CLOUDINARY_API_KEY,
    api_secret=CLOUDINARY_API_SECRET,
    secure=True
)

# Initialize two Gemini clients
client = genai.Client(api_key=GENAI_API_KEY)  # For text queries
img_client = genai.Client(api_key=GENAI_API_KEY)  # For image generation

user_states = {}
processed_events = set()

# Helper Functions
def load_users():
    if os.path.exists(USERS_PATH):
        return pd.read_csv(USERS_PATH)
    else:
        df = pd.DataFrame(columns=["slack_id", "name", "phone"])
        df.to_csv(USERS_PATH, index=False)
        return df

def save_user(slack_id, name, phone):
    users_df = load_users()
    if slack_id not in users_df["slack_id"].values:
        new_user = pd.DataFrame([{"slack_id": slack_id, "name": name, "phone": phone}])
        users_df = pd.concat([users_df, new_user], ignore_index=True)
        users_df.to_csv(USERS_PATH, index=False)

def get_user_info(slack_id):
    users_df = load_users()
    user = users_df[users_df["slack_id"] == slack_id]
    return user.iloc[0] if not user.empty else None

def send_whatsapp_message(phone, message, media_url=None):
    try:
        msg = twilio_client.messages.create(
            body=message,
            from_=f"whatsapp:{TWILIO_PHONE_NUMBER}",
            to=f"whatsapp:{phone}",
            media_url=[media_url] if media_url else []
        )
        logging.info(f"WhatsApp sent to {phone}: SID {msg.sid}")
        return True
    except Exception as e:
        logging.error(f"WhatsApp failed for {phone}: {e}")
        return False

# Invoice Generation
def generate_invoice(slack_id, channel):
    try:
        user = get_user_info(slack_id)
        if not user:
            return {
                "blocks": [
                    {"type": "header", "text": {"type": "plain_text", "text": ":pencil2: *Invoice Generation Failed* :pencil2:"}},
                    {"type": "section", "text": {"type": "mrkdwn", "text": "Please register with: `register: name, phone`"}}
                ],
                "text": "Invoice failed: not registered."
            }

        name = user["name"]
        phone = user["phone"]
        invoice_url = "https://res.cloudinary.com/dnnj6hykk/image/upload/v1744547940/ivoice-test-GED_2_gmxls7.pdf"
        
        # Send PDF to Slack
        with request.urlopen(invoice_url) as response:
            pdf_data = response.read()
            pdf_io = BytesIO(pdf_data)
            slack_client.files_upload_v2(
                channel=channel,
                file=pdf_io,
                filename="invoice_A35432.pdf",
                title="Invoice A35432",
                initial_comment=":pencil2: *Invoice Generated* :pencil2:\nYour invoice has been uploaded!"
            )
        
        # Send WhatsApp message
        message = f"Hey {name} 👋, thank you for your latest purchase with Smart Shoes 👟. Here's your invoice against your order A35432. Visit us again. We have exciting discounts only for you!"
        send_whatsapp_message(phone, message, invoice_url)
        
        logging.info(f"Invoice sent for {slack_id}: {invoice_url}")
        return {
            "blocks": [
                {"type": "header", "text": {"type": "plain_text", "text": ":pencil2: *Invoice Generated* :pencil2:"}},
                {"type": "section", "text": {"type": "mrkdwn", "text": f"Invoice sent to this channel and WhatsApp!\nURL: {invoice_url}"}}
            ],
            "text": "Invoice generated."
        }
    except Exception as e:
        logging.error(f"Invoice error for {slack_id}: {e}")
        return {
            "blocks": [
                {"type": "header", "text": {"type": "plain_text", "text": ":pencil2: *Invoice Generation Failed* :pencil2:"}},
                {"type": "section", "text": {"type": "mrkdwn", "text": "Sorry, invoice generation failed. Try again later. 🙁"}}
            ],
            "text": "Invoice failed."
        }

# Promotion Generation
def generate_promotion(slack_id, channel):
    try:
        user = get_user_info(slack_id)
        if not user:
            return {
                "blocks": [
                    {"type": "header", "text": {"type": "plain_text", "text": ":frame_with_picture: *Promotion Generation Failed* :frame_with_picture:"}},
                    {"type": "section", "text": {"type": "mrkdwn", "text": "Please register with: `register: name, phone`"}}
                ],
                "text": "Promotion failed: not registered."
            }

        name = user["name"]
        phone = user["phone"]
        
        # Generate promotion image using img_client
        contents = 'Hi, can you create a 50 percent offer poster for my shoe shop named "Smart Shoes". I need a colorful and attractive shoe image and my shop name "Smart Shoes" in centre and the text "50 percent discount" highlighted'
        response = img_client.models.generate_content(
            model="gemini-2.0-flash-exp-image-generation",
            contents=contents,
            config=GenerateContentConfig(response_modalities=['Text', 'Image'])
        )
        
        image_data = None
        for part in response.candidates[0].content.parts:
            if part.inline_data is not None:
                image_data = part.inline_data.data
                break
        
        if not image_data:
            raise Exception("No image generated by Gemini.")
        
        image = Image.open(BytesIO(image_data))
        image_io = BytesIO()
        image.save(image_io, format="PNG")
        image_io.seek(0)
        
        # Upload to Cloudinary
        public_id = f"promotion_smart_shoes_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        upload_result = cloudinary.uploader.upload(
            image_io,
            resource_type="image",
            public_id=public_id
        )
        image_url = upload_result["secure_url"]
        
        # Send image to Slack
        image_io.seek(0)
        slack_client.files_upload_v2(
            channel=channel,
            file=image_io,
            filename="promotion.png",
            title="Smart Shoes Promotion",
            initial_comment=":frame_with_picture: *Promotion Generated* :frame_with_picture:\nYour promotion poster is ready!"
        )
        
        # Send WhatsApp message
        message = f"Hey {name} 👋, check out this exclusive 50% off promotion from Smart Shoes 👟! Grab your deal now. Visit us again for more exciting offers!"
        send_whatsapp_message(phone, message, image_url)
        
        logging.info(f"Promotion sent for {slack_id}: {image_url}")
        return {
            "blocks": [
                {"type": "header", "text": {"type": "plain_text", "text": ":frame_with_picture: *Promotion Generated* :frame_with_picture:"}},
                {"type": "section", "text": {"type": "mrkdwn", "text": f"Promotion poster sent to this channel and WhatsApp!\nURL: {image_url}"}}
            ],
            "text": "Promotion generated."
        }
    except Exception as e:
        logging.error(f"Promotion error for {slack_id}: {e}")
        return {
            "blocks": [
                {"type": "header", "text": {"type": "plain_text", "text": ":frame_with_picture: *Promotion Generation Failed* :frame_with_picture:"}},
                {"type": "section", "text": {"type": "mrkdwn", "text": "Sorry, promotion generation failed. Try again later. 🙁"}}
            ],
            "text": "Promotion failed."
        }

# Chart Generation (Price Category Pie Chart)
def generate_chart(slack_id, channel):
    try:
        user = get_user_info(slack_id)
        if not user:
            return {
                "blocks": [
                    {"type": "header", "text": {"type": "plain_text", "text": ":chart_with_upwards_trend: *Chart Generation Failed* :chart_with_upwards_trend:"}},
                    {"type": "section", "text": {"type": "mrkdwn", "text": "Please register with: `register: name, phone`"}}
                ],
                "text": "Chart failed: not registered."
            }

        # Load sales data
        if not os.path.exists(SALES_DATA_PATH):
            return {
                "blocks": [
                    {"type": "header", "text": {"type": "plain_text", "text": ":chart_with_upwards_trend: *Chart Generation Failed* :chart_with_upwards_trend:"}},
                    {"type": "section", "text": {"type": "mrkdwn", "text": "No sales data available. 🙁"}}
                ],
                "text": "Chart failed: no data."
            }
        
        df = pd.read_csv(SALES_DATA_PATH)
        df["Price Each"] = pd.to_numeric(df["Price Each"], errors="coerce")
        
        # Create pie chart
        df["Price Category"] = df["Price Each"].apply(lambda x: "Less than 500" if x < 500 else "500 or More")
        price_counts = df["Price Category"].value_counts().reset_index()
        price_counts.columns = ["Price Category", "Count"]
        
        fig = px.pie(
            price_counts,
            names="Price Category",
            values="Count",
            title="Products Sold by Price Category",
            color_discrete_sequence=["#FF6F61", "#6B728E"],
            template="plotly_white"
        )
        fig.update_traces(textinfo="percent+label", pull=[0.1, 0])
        fig.update_layout(
            title_font_size=20,
            font=dict(size=14),
            margin=dict(t=50, b=50),
            showlegend=True
        )
        
        # Save plot
        plot_filename = os.path.join(PLOTS_DIR, f"pie_chart_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        fig.write_image(plot_filename, width=800, height=600, engine="kaleido")
        
        # Upload to Cloudinary
        upload_result = cloudinary.uploader.upload(
            plot_filename,
            resource_type="image",
            public_id=f"chart_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        chart_url = upload_result["secure_url"]
        
        # Send to Slack
        with open(plot_filename, "rb") as f:
            slack_client.files_upload_v2(
                channel=channel,
                file=f,
                filename="price_category_pie_chart.png",
                title="Price Category Pie Chart",
                initial_comment=":chart_with_upwards_trend: *Chart Generated* :chart_with_upwards_trend:\nPie chart showing products sold by price category."
            )
        
        # Clean up
        os.remove(plot_filename)
        
        logging.info(f"Chart sent for {slack_id}: {chart_url}")
        return {
            "blocks": [
                {"type": "header", "text": {"type": "plain_text", "text": ":chart_with_upwards_trend: *Chart Generated* :chart_with_upwards_trend:"}},
                {"type": "section", "text": {"type": "mrkdwn", "text": f"Chart sent to this channel!\nURL: {chart_url}"}}
            ],
            "text": "Chart generated."
        }
    except Exception as e:
        logging.error(f"Chart error for {slack_id}: {e}")
        return {
            "blocks": [
                {"type": "header", "text": {"type": "plain_text", "text": ":chart_with_upwards_trend: *Chart Generation Failed* :chart_with_upwards_trend:"}},
                {"type": "section", "text": {"type": "mrkdwn", "text": "Sorry, chart generation failed. Try again later. 🙁"}}
            ],
            "text": "Chart failed."
        }

# Weekly Analysis
def generate_weekly_analysis(slack_id, channel):
    try:
        user = get_user_info(slack_id)
        if not user:
            return {
                "blocks": [
                    {"type": "header", "text": {"type": "plain_text", "text": ":calendar: *Weekly Analysis Failed* :calendar:"}},
                    {"type": "section", "text": {"type": "mrkdwn", "text": "Please register with: `register: name, phone`"}}
                ],
                "text": "Weekly analysis failed: not registered."
            }

        name = user["name"]
        phone = user["phone"]
        
        # Load sales data
        if not os.path.exists(SALES_DATA_PATH):
            return {
                "blocks": [
                    {"type": "header", "text": {"type": "plain_text", "text": ":calendar: *Weekly Analysis Failed* :calendar:"}},
                    {"type": "section", "text": {"type": "mrkdwn", "text": "No sales data available. 🙁"}}
                ],
                "text": "Weekly analysis failed: no data."
            }
        
        df = pd.read_csv(SALES_DATA_PATH)
        df["Price Each"] = pd.to_numeric(df["Price Each"], errors="coerce")
        df["Quantity Ordered"] = pd.to_numeric(df["Quantity Ordered"], errors="coerce")
        df["Order Date"] = pd.to_datetime(df["Order Date"])
        
        # Calculate weekly sales
        df["Total Sales"] = df["Quantity Ordered"] * df["Price Each"]
        df["Week"] = df["Order Date"].dt.isocalendar().week
        df["Year"] = df["Order Date"].dt.year
        weekly_sales = df.groupby(["Year", "Week"])["Total Sales"].sum().reset_index()
        weekly_sales["Week Label"] = weekly_sales.apply(lambda x: f"{x['Year']}-W{x['Week']:02d}", axis=1)
        
        # Create bar chart
        fig = px.bar(
            weekly_sales,
            x="Week Label",
            y="Total Sales",
            title="Weekly Sales Analysis",
            color="Total Sales",
            color_continuous_scale="Viridis",
            template="plotly_white"
        )
        fig.update_layout(
            title_font_size=20,
            font=dict(size=14),
            margin=dict(t=50, b=50),
            xaxis_title="Week",
            yaxis_title="Total Sales ($)",
            xaxis_tickangle=45,
            bargap=0.2
        )
        
        # Save plot
        plot_filename = os.path.join(PLOTS_DIR, f"weekly_sales_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        fig.write_image(plot_filename, width=800, height=600, engine="kaleido")
        
        # Upload to Cloudinary
        upload_result = cloudinary.uploader.upload(
            plot_filename,
            resource_type="image",
            public_id=f"weekly_sales_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        chart_url = upload_result["secure_url"]
        
        # Generate summary
        total_weeks = len(weekly_sales)
        avg_sales = weekly_sales["Total Sales"].mean()
        max_sales_week = weekly_sales.loc[weekly_sales["Total Sales"].idxmax(), "Week Label"]
        summary = f"Analyzed {total_weeks} weeks. Average weekly sales: ${avg_sales:.2f}. Highest sales in {max_sales_week}."
        
        # Send to Slack
        with open(plot_filename, "rb") as f:
            slack_client.files_upload_v2(
                channel=channel,
                file=f,
                filename="weekly_sales.png",
                title="Weekly Sales Analysis",
                initial_comment=f":calendar: *Weekly Analysis Generated* :calendar:\n{summary}"
            )
        
        # Send WhatsApp message
        message = f"Hey {name} 👋, here's your weekly sales analysis for Smart Shoes 👟! Check out the trends and plan your next big sale. More insights await!"
        send_whatsapp_message(phone, message, chart_url)
        
        # Clean up
        os.remove(plot_filename)
        
        logging.info(f"Weekly analysis sent for {slack_id}: {chart_url}")
        return {
            "blocks": [
                {"type": "header", "text": {"type": "plain_text", "text": ":calendar: *Weekly Analysis Generated* :calendar:"}},
                {"type": "section", "text": {"type": "mrkdwn", "text": f"{summary}\nChart sent to this channel and WhatsApp!\nURL: {chart_url}"}}
            ],
            "text": "Weekly analysis generated."
        }
    except Exception as e:
        logging.error(f"Weekly analysis error for {slack_id}: {e}")
        return {
            "blocks": [
                {"type": "header", "text": {"type": "plain_text", "text": ":calendar: *Weekly Analysis Failed* :calendar:"}},
                {"type": "section", "text": {"type": "mrkdwn", "text": "Sorry, weekly analysis failed. Try again later. 🙁"}}
            ],
            "text": "Weekly analysis failed."
        }

# Sales Insights
def generate_sales_insights(slack_id, channel):
    try:
        user = get_user_info(slack_id)
        if not user:
            return {
                "blocks": [
                    {"type": "header", "text": {"type": "plain_text", "text": ":bar_chart: *Sales Insights Failed* :bar_chart:"}},
                    {"type": "section", "text": {"type": "mrkdwn", "text": "Please register with: `register: name, phone`"}}
                ],
                "text": "Sales insights failed: not registered."
            }

        name = user["name"]
        phone = user["phone"]
        
        # Load sales data
        if not os.path.exists(SALES_DATA_PATH):
            return {
                "blocks": [
                    {"type": "header", "text": {"type": "plain_text", "text": ":bar_chart: *Sales Insights Failed* :bar_chart:"}},
                    {"type": "section", "text": {"type": "mrkdwn", "text": "No sales data available. 🙁"}}
                ],
                "text": "Sales insights failed: no data."
            }
        
        df = pd.read_csv(SALES_DATA_PATH)
        df["Price Each"] = pd.to_numeric(df["Price Each"], errors="coerce")
        df["Quantity Ordered"] = pd.to_numeric(df["Quantity Ordered"], errors="coerce")
        df["Order Date"] = pd.to_datetime(df["Order Date"])
        
        # Calculate metrics
        df["Total Sales"] = df["Quantity Ordered"] * df["Price Each"]
        total_revenue = df["Total Sales"].sum()
        top_product = df.groupby("Product")["Quantity Ordered"].sum().idxmax()
        avg_order_value = df["Total Sales"].mean()
        
        # Create line chart
        df["Date"] = df["Order Date"].dt.date
        daily_sales = df.groupby("Date")["Total Sales"].sum().reset_index()
        fig = px.line(
            daily_sales,
            x="Date",
            y="Total Sales",
            title="Sales Trend Over Time",
            template="plotly_white",
            markers=True
        )
        fig.update_layout(
            title_font_size=20,
            font=dict(size=14),
            margin=dict(t=50, b=50),
            xaxis_title="Date",
            yaxis_title="Total Sales ($)"
        )
        
        # Save plot
        plot_filename = os.path.join(PLOTS_DIR, f"sales_trend_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        fig.write_image(plot_filename, width=800, height=600, engine="kaleido")
        
        # Upload to Cloudinary
        upload_result = cloudinary.uploader.upload(
            plot_filename,
            resource_type="image",
            public_id=f"sales_trend_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        chart_url = upload_result["secure_url"]
        
        # Generate summary
        summary = f"Total Revenue: ${total_revenue:.2f}\nTop Product: {top_product}\nAverage Order Value: ${avg_order_value:.2f}"
        
        # Send to Slack
        with open(plot_filename, "rb") as f:
            slack_client.files_upload_v2(
                channel=channel,
                file=f,
                filename="sales_trend.png",
                title="Sales Trend Over Time",
                initial_comment=f":bar_chart: *Sales Insights Generated* :bar_chart:\n{summary}"
            )
        
        # Send WhatsApp message
        message = f"Hey {name} 👋, dive into your Smart Shoes sales insights 👟! See key metrics and trends to boost your business. More data awaits!"
        send_whatsapp_message(phone, message, chart_url)
        
        # Clean up
        os.remove(plot_filename)
        
        logging.info(f"Sales insights sent for {slack_id}: {chart_url}")
        return {
            "blocks": [
                {"type": "header", "text": {"type": "plain_text", "text": ":bar_chart: *Sales Insights Generated* :bar_chart:"}},
                {"type": "section", "text": {"type": "mrkdwn", "text": f"{summary}\nChart sent to this channel and WhatsApp!\nURL: {chart_url}"}}
            ],
            "text": "Sales insights generated."
        }
    except Exception as e:
        logging.error(f"Sales insights error for {slack_id}: {e}")
        return {
            "blocks": [
                {"type": "header", "text": {"type": "plain_text", "text": ":bar_chart: *Sales Insights Failed* :bar_chart:"}},
                {"type": "section", "text": {"type": "mrkdwn", "text": "Sorry, sales insights failed. Try again later. 🙁"}}
            ],
            "text": "Sales insights failed."
        }

# User Registration
def handle_registration(slack_id, text):
    try:
        if "register:" not in text.lower():
            return {
                "blocks": [
                    {"type": "header", "text": {"type": "plain_text", "text": ":bust_in_silhouette: *Registration Help* :bust_in_silhouette:"}},
                    {"type": "section", "text": {"type": "mrkdwn", "text": "Use: `register: name, phone`"}}
                ],
                "text": "Registration help."
            }
        
        _, details = text.lower().split("register:", 1)
        name, phone = [x.strip() for x in details.split(",", 1)]
        phone = re.sub(r"[^0-9+]", "", phone)
        if not phone.startswith("+"):
            phone = "+" + phone
        if not re.match(r"^\+\d{10,15}$", phone):
            return {
                "blocks": [
                    {"type": "header", "text": {"type": "plain_text", "text": ":bust_in_silhouette: *Registration Failed* :bust_in_silhouette:"}},
                    {"type": "section", "text": {"type": "mrkdwn", "text": "Invalid phone number. Use format: `+1234567890`"}}
                ],
                "text": "Registration failed: invalid phone."
            }
        
        save_user(slack_id, name, phone)
        message = f"Hey {name} 👋, welcome to Smart Shoes! You're now registered. Start shopping! 👟"
        send_whatsapp_message(phone, message)
        
        return {
            "blocks": [
                {"type": "header", "text": {"type": "plain_text", "text": ":bust_in_silhouette: *Registration Successful* :bust_in_silhouette:"}},
                {"type": "section", "text": {"type": "mrkdwn", "text": f"Welcome, {name}! Check your WhatsApp for a confirmation. 😊"}}
            ],
            "text": "Registration successful."
        }
    except Exception as e:
        logging.error(f"Registration error for {slack_id}: {e}")
        return {
            "blocks": [
                {"type": "header", "text": {"type": "plain_text", "text": ":bust_in_silhouette: *Registration Failed* :bust_in_silhouette:"}},
                {"type": "section", "text": {"type": "mrkdwn", "text": "Sorry, registration failed. Try again later. 🙁"}}
            ],
            "text": "Registration failed."
        }

# Query Processing
def process_query(text, slack_id, channel):
    text = text.lower().strip()
    logging.info(f"Query from {slack_id}: {text}")
    
    # Casual greetings
    if any(g in text for g in ["hello", "hi", "how are you"]):
        return {
            "blocks": [
                {"type": "section", "text": {"type": "mrkdwn", "text": "Hey there! 😊 I'm ready to help with invoices, promotions, charts, weekly analysis, sales insights, or registration. What's up?"}}
            ],
            "text": "Greeting response."
        }
    
    # Main functionalities
    if "generate invoice" in text:
        return generate_invoice(slack_id, channel)
    elif "generate promotion" in text:
        return generate_promotion(slack_id, channel)
    elif "generate chart" in text:
        return generate_chart(slack_id, channel)
    elif "weekly analysis" in text:
        return generate_weekly_analysis(slack_id, channel)
    elif "sales insights" in text:
        return generate_sales_insights(slack_id, channel)
    elif "register:" in text:
        return handle_registration(slack_id, text)
    else:
        # Use text client for general queries
        try:
            model = client.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(f"Respond to: {text}")
            return {
                "blocks": [
                    {"type": "section", "text": {"type": "mrkdwn", "text": response.text.strip()}}
                ],
                "text": response.text.strip()[:100]
            }
        except Exception as e:
            logging.error(f"Text query error for {slack_id}: {e}")
            return {
                "blocks": [
                    {"type": "section", "text": {"type": "mrkdwn", "text": "Oops! Try: `generate invoice`, `generate promotion`, `generate chart`, `weekly analysis`, `sales insights`, or `register: name, phone` 🤔"}}
                ],
                "text": "Unknown command."
            }

# HTTP Server
class SlackEventHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/plain")
        self.end_headers()
        self.wfile.write(b"Smart Shoes Slack Bot")

    def do_POST(self):
        if self.path == "/slack/events":
            content_length = int(self.headers.get("Content-Length", 0))
            post_data = self.rfile.read(content_length).decode("utf-8")
            try:
                data = json.loads(post_data)
                if data.get("type") == "url_verification":
                    self.send_response(200)
                    self.send_header("Content-type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"challenge": data.get("challenge")}).encode("utf-8"))
                    return
                
                event = data.get("event", {})
                slack_id = event.get("user")
                text = event.get("text", "")
                channel = event.get("channel")
                event_ts = event.get("ts")
                
                if slack_id and text and channel:
                    event_id = f"{event_ts}_{channel}_{slack_id}"
                    if event_id in processed_events:
                        logging.info(f"Skipping duplicate event {event_id}")
                        return
                    processed_events.add(event_id)
                    if len(processed_events) > 1000:
                        processed_events.clear()
                    
                    response = process_query(text, slack_id, channel)
                    try:
                        slack_client.chat_postMessage(
                            channel=channel,
                            blocks=response["blocks"],
                            text=response["text"]
                        )
                    except SlackApiError as e:
                        logging.error(f"Slack post failed for {slack_id}: {e}")
                
                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "ok"}).encode("utf-8"))
            except Exception as e:
                logging.error(f"Slack event error: {e}")
                self.send_response(400)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

def run_server():
    server = HTTPServer(("", int(os.getenv("PORT", 10000))), SlackEventHandler)
    server.serve_forever()

if __name__ == "__main__":
    threading.Thread(target=run_server, daemon=True).start()
    logging.info("Slack bot started on port 10000")
    while True:
        threading.Event().wait()
