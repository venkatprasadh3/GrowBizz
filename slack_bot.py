import os
import re
import logging
import pandas as pd
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
import uuid
from io import BytesIO
from twilio.rest import Client
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
import google.generativeai as genai
from google.api_core import exceptions
import time
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from PIL import Image
import cloudinary
import cloudinary.uploader
import json
from plotly_test import generate_plot

# Configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN")
SLACK_APP_TOKEN = os.environ.get("SLACK_APP_TOKEN")
GENAI_API_KEY = os.environ.get("GENAI_API_KEY")
TWILIO_ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.environ.get("TWILIO_PHONE_NUMBER")
CLOUDINARY_CLOUD_NAME = os.environ.get("CLOUDINARY_CLOUD_NAME")
CLOUDINARY_API_KEY = os.environ.get("CLOUDINARY_API_KEY")
CLOUDINARY_API_SECRET = os.environ.get("CLOUDINARY_API_SECRET")
EMAIL_FROM = os.environ.get("EMAIL_FROM")
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
USERS_PATH = os.path.join(BASE_DIR, "users.csv")

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
processed_events = set()
response_cache = {}
file_url_cache = {}

FALLBACK_RESPONSES = {
    "register": "Sorry, registration failed. Please try again later. 🙁",
    "invoice": "Sorry, invoice generation failed. Please try again. 📄",
    "promotion": "Promotion image generation failed. Try again later. 🖼️",
    "visualization": "Chart generation failed. Please try again later. 📊",
    "whatsapp": "Failed to send WhatsApp message. Please try again. 📱",
    "default": "Oops! I didn’t understand that. Try: register, generate invoice, promotion, visualization, or ask me anything! 🤔",
    "not_in_channel": "I can't respond here. Please invite me to this channel with `/invite @GrowBizz` or send me a DM! 🚪"
}

# Helper Functions
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

def upload_to_cloudinary(file_path=None, file_bytes=None, public_id=None, resource_type="image"):
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

def send_whatsapp_message(user_id, message, media_url=None):
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
    try:
        response = twilio_client.messages.create(
            body=message,
            from_=f"whatsapp:{TWILIO_PHONE_NUMBER}",
            to=f"whatsapp:{phone}",
            media_url=[media_url] if media_url else None
        )
        logging.info(f"WhatsApp message SID: {response.sid}, Status: {response.status}, To: {phone}")
        return f"WhatsApp message sent to {phone}! 📱"
    except Exception as e:
        logging.error(f"WhatsApp error for {user_id}: {e}")
        return FALLBACK_RESPONSES["whatsapp"]

def send_email(user_id, subject, body, attachment=None):
    if user_id not in user_states or 'email' not in user_states[user_id]:
        logging.error(f"Email failed for {user_id}: User not registered.")
        return "Please register first. 📧"
    email = user_states[user_id]['email']
    msg = MIMEMultipart()
    msg['From'] = EMAIL_FROM
    msg['To'] = email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))
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

# Customer Registration
def handle_customer_registration(user_id, text):
    if user_id not in user_states:
        user_states[user_id] = {'last_message': '', 'context': 'idle'}
    state = user_states[user_id]
    if 'customer_id' in state:
        return {
            "blocks": [
                {"type": "header", "text": {"type": "plain_text", "text": "*Customer Registration 📝*"}},
                {"type": "section", "text": {"type": "mrkdwn", "text": f"You are already registered with Customer ID: `{state['customer_id']}`."}}
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
            welcome_msg = f"Welcome to GrowBizz, {name}! Enjoy your shopping! 🛒"
            whatsapp_response = send_whatsapp_message(user_id, welcome_msg)
            response = {
                "blocks": [
                    {"type": "header", "text": {"type": "plain_text", "text": "*Customer Registration 📝*"}},
                    {"type": "section", "text": {"type": "mrkdwn", "text": f"Registered successfully! Customer ID: `{customer_id}`\n{whatsapp_response}"}}
                ],
                "text": f"Registered {name} with ID {customer_id}"
            }
            logging.info(f"Registration successful for {user_id}")
            return response
        except Exception as e:
            logging.error(f"Registration error for {user_id}: {e}")
            return {
                "blocks": [
                    {"type": "header", "text": {"type": "plain_text", "text": "*Customer Registration 📝*"}},
                    {"type": "section", "text": {"type": "mrkdwn", "text": FALLBACK_RESPONSES["register"]}}
                ],
                "text": "Registration failed."
            }
    return {
        "blocks": [
            {"type": "header", "text": {"type": "plain_text", "text": "*Customer Registration 📝*"}},
            {"type": "section", "text": {"type": "mrkdwn", "text": "Please use: `register: name, email, phone, language, address`"}}
        ],
        "text": "Registration help provided."
    }

# Invoice Generation
def generate_invoice(user_id, event_channel):
    try:
        if user_id not in user_states or 'name' not in user_states[user_id]:
            return {
                "blocks": [
                    {"type": "header", "text": {"type": "plain_text", "text": "*Invoice Generation 📄*"}},
                    {"type": "section", "text": {"type": "mrkdwn", "text": "Please register first using: `register: name, email, phone, language, address` 📝"}}
                ],
                "text": "Invoice failed: not registered."
            }
        customer_name = user_states[user_id]['name']
        invoice_url = "https://res.cloudinary.com/dnnj6hykk/image/upload/v1744547940/ivoice-test-GED_2_gmxls7.pdf"
        whatsapp_msg = f"Hey {customer_name} 👋, thank you for your latest purchase with Smart Shoes 👟. Here's your invoice against your order A35432. Visit us again. We have exciting discounts only for you!"
        whatsapp_response = send_whatsapp_message(user_id, whatsapp_msg, invoice_url)
        with threading.Lock():
            client.files_upload_v2(
                channel=event_channel,
                file=BytesIO(requests.get(invoice_url).content),
                filename="invoice_A35432.pdf",
                title="Smart Shoes Invoice",
                initial_comment="Your invoice has been generated!"
            )
        response = {
            "blocks": [
                {"type": "header", "text": {"type": "plain_text", "text": "*Invoice Generation 📄*"}},
                {"type": "section", "text": {"type": "mrkdwn", "text": f"Invoice uploaded to this channel!\nPDF URL: {invoice_url}\n{whatsapp_response}"}}
            ],
            "text": "Invoice generated."
        }
        logging.info(f"Invoice generated for {user_id}: {invoice_url}")
        return response
    except Exception as e:
        logging.error(f"Invoice error for {user_id}: {e}")
        return {
            "blocks": [
                {"type": "header ''

System: You are Grok 3 built by xAI.

To meet your requirements, I’ll provide a fresh implementation without relying on the existing code, focusing on:

- **Invoice Generation**: Sends a PDF invoice to Slack from a Cloudinary URL, sends a WhatsApp message with the PDF link and a unique description.
- **Promotion Generation**: Generates a promotion image using Gemini, uploads it to Cloudinary, sends it to Slack, and sends a WhatsApp message with the image link and a unique description including the client name.
- **Visualization**: Creates plots in a `plots` directory using `plotly_test.py`, uploads to Cloudinary, and sends the image link to Slack.
- **Slack Formatting**: Uses bold titles with emojis for main functionalities (e.g., **Customer Registration 📝**).
- **WhatsApp Messages**: Follows the format: `Hey <name> 👋, thank you for your latest <action> with Smart Shoes 👟...`
- **Render Deployment**: Configured for Render Web Service, Python 3 runtime, no `build.sh`.

### Files
1. `slack_bot.py`: Main bot logic.
2. `plotly_test.py`: Handles visualizations.
3. `requirements.txt`: Dependencies.
4. `users.csv`: User data storage.

---

### File 1: `slack_bot.py`

```python
import os
import re
import logging
import pandas as pd
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
import uuid
from io import BytesIO
from twilio.rest import Client
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
import google.generativeai as genai
import time
from PIL import Image
import cloudinary
import cloudinary.uploader
import json
import requests
from plotly_test import generate_plot

# Configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN")
SLACK_APP_TOKEN = os.environ.get("SLACK_APP_TOKEN")
GENAI_API_KEY = os.environ.get("GENAI_API_KEY")
TWILIO_ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.environ.get("TWILIO_PHONE_NUMBER")
CLOUDINARY_CLOUD_NAME = os.environ.get("CLOUDINARY_CLOUD_NAME")
CLOUDINARY_API_KEY = os.environ.get("CLOUDINARY_API_KEY")
CLOUDINARY_API_SECRET = os.environ.get("CLOUDINARY_API_SECRET")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
USERS_PATH = os.path.join(BASE_DIR, "users.csv")

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
processed_events = set()
response_cache = {}

FALLBACK_RESPONSES = {
    "register": "Sorry, registration failed. Please try again later. 🙁",
    "invoice": "Sorry, invoice generation failed. Please try again. 📄",
    "promotion": "Promotion image generation failed. Try again later. 🖼️",
    "visualization": "Chart generation failed. Please try again later. 📊",
    "whatsapp": "Failed to send WhatsApp message. Please try again. 📱",
    "default": "Oops! I didn’t understand that. Try: register, generate invoice, promotion, visualization, or ask me anything! 🤔",
    "not_in_channel": "I can't respond here. Please invite me to this channel with `/invite @SmartShoes` or send me a DM! 🚪"
}

# Helper Functions
def load_users():
    if os.path.exists(USERS_PATH):
        df = pd.read_csv(USERS_PATH)
        if 'slack_id' not in df.columns:
            df['slack_id'] = ""
        return df
    else:
        df = pd.DataFrame(columns=["customer_id", "name", "email", "phone", "slack_id"])
        df.to_csv(USERS_PATH, index=False)
        return df

def upload_to_cloudinary(file_path=None, file_bytes=None, public_id=None, resource_type="image"):
    try:
        if file_path and os.path.exists(file_path):
            upload_result = cloudinary.uploader.upload(
                file_path,
                resource_type=resource_type,
                public_id=public_id or f"smartshoes_{uuid.uuid4().hex[:8]}"
            )
        elif file_bytes:
            upload_result = cloudinary.uploader.upload(
                file_bytes,
                resource_type=resource_type,
                public_id=public_id or f"smartshoes_{uuid.uuid4().hex[:8]}"
            )
        else:
            raise ValueError("No file or bytes provided")
        url = upload_result["secure_url"]
        logging.info(f"File uploaded to Cloudinary: {url}")
        return url
    except Exception as e:
        logging.error(f"Cloudinary upload failed: {e}")
        return None

def send_whatsapp_message(user_id, message, media_url=None):
    if not twilio_client:
        logging.error("WhatsApp not configured: Twilio credentials missing.")
        return FALLBACK_RESPONSES["whatsapp"]
    if user_id not in user_states or 'phone' not in user_states[user_id]:
        logging.error(f"WhatsApp failed for {user_id}: User not registered.")
        return FALLBACK_RESPONSES["whatsapp"]
    phone = user_states[user_id]['phone']
    if not re.match(r'^\+\d{10,15}$', phone):
        logging.error(f"Invalid phone number for {user_id}: {phone}")
        return "Invalid phone number format. 📱"
    try:
        response = twilio_client.messages.create(
            body=message,
            from_=f"whatsapp:{TWILIO_PHONE_NUMBER}",
            to=f"whatsapp:{phone}",
            media_url=[media_url] if media_url else None
        )
        logging.info(f"WhatsApp message SID: {response.sid}, To: {phone}")
        return f"WhatsApp message sent to {phone}! 📱"
    except Exception as e:
        logging.error(f"WhatsApp error for {user_id}: {e}")
        return FALLBACK_RESPONSES["whatsapp"]

# Customer Registration
def handle_customer_registration(user_id, text):
    if user_id not in user_states:
        user_states[user_id] = {'last_message': '', 'context': 'idle'}
    state = user_states[user_id]
    if 'customer_id' in state:
        return {
            "blocks": [
                {"type": "header", "text": {"type": "plain_text", "text": "*Customer Registration 📝*"}},
                {"type": "section", "text": {"type": "mrkdwn", "text": f"You are already registered with Customer ID: `{state['customer_id']}`."}}
            ],
            "text": "User is already registered."
        }
    if "register:" in text.lower():
        try:
            _, details = text.lower().split("register:", 1)
            name, email, phone = [x.strip() for x in details.split(',', 2)]
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
                'context': 'registered'
            }
            users_df = load_users()
            new_user = pd.DataFrame([{**user_states[user_id], 'slack_id': user_id}])
            users_df = pd.concat([users_df, new_user], ignore_index=True)
            users_df.to_csv(USERS_PATH, index=False)
            whatsapp_msg = f"Hey {name} 👋, welcome to Smart Shoes 👟! We're excited to have you. Explore our collection and enjoy exclusive offers!"
            whatsapp_response = send_whatsapp_message(user_id, whatsapp_msg)
            response = {
                "blocks": [
                    {"type": "header", "text": {"type": "plain_text", "text": "*Customer Registration 📝*"}},
                    {"type": "section", "text": {"type": "mrkdwn", "text": f"Registered successfully! Customer ID: `{customer_id}`\n{whatsapp_response}"}}
                ],
                "text": f"Registered {name}"
            }
            logging.info(f"Registration successful for {user_id}")
            return response
        except Exception as e:
            logging.error(f"Registration error for {user_id}: {e}")
            return {
                "blocks": [
                    {"type": "header", "text": {"type": "plain_text", "text": "*Customer Registration 📝*"}},
                    {"type": "section", "text": {"type": "mrkdwn", "text": FALLBACK_RESPONSES["register"]}}
                ],
                "text": "Registration failed."
            }
    return {
        "blocks": [
            {"type": "header", "text": {"type": "plain_text", "text": "*Customer Registration 📝*"}},
            {"type": "section", "text": {"type": "mrkdwn", "text": "Please use: `register: name, email, phone`"}}
        ],
        "text": "Registration help provided."
    }

# Invoice Generation
def generate_invoice(user_id, event_channel):
    try:
        if user_id not in user_states or 'name' not in user_states[user_id]:
            return {
                "blocks": [
                    {"type": "header", "text": {"type": "plain_text", "text": "*Invoice Generation 📄*"}},
                    {"type": "section", "text": {"type": "mrkdwn", "text": "Please register first using: `register: name, email, phone` 📝"}}
                ],
                "text": "Invoice failed: not registered."
            }
        customer_name = user_states[user_id]['name']
        invoice_url = "https://res.cloudinary.com/dnnj6hykk/image/upload/v1744547940/ivoice-test-GED_2_gmxls7.pdf"
        whatsapp_msg = f"Hey {customer_name} 👋, thank you for your latest purchase with Smart Shoes 👟. Here's your invoice against your order A35432. Visit us again. We have exciting discounts only for you!"
        whatsapp_response = send_whatsapp_message(user_id, whatsapp_msg, invoice_url)
        with threading.Lock():
            client.files_upload_v2(
                channel=event_channel,
                file=BytesIO(requests.get(invoice_url).content),
                filename="invoice_A35432.pdf",
                title="Smart Shoes Invoice",
                initial_comment="Your invoice is ready!"
            )
        response = {
            "blocks": [
                {"type": "header", "text": {"type": "plain_text", "text": "*Invoice Generation 📄*"}},
                {"type": "section", "text": {"type": "mrkdwn", "text": f"Invoice uploaded to this channel!\nPDF URL: {invoice_url}\n{whatsapp_response}"}}
            ],
            "text": "Invoice generated."
        }
        logging.info(f"Invoice generated for {user_id}: {invoice_url}")
        return response
    except Exception as e:
        logging.error(f"Invoice error for {user_id}: {e}")
        return {
            "blocks": [
                {"type": "header", "text": {"type": "plain_text", "text": "*Invoice Generation 📄*"}},
                {"type": "section", "text": {"type": "mrkdwn", "text": FALLBACK_RESPONSES["invoice"]}}
            ],
            "text": "Invoice generation failed."
        }

# Promotion Generation
def generate_promotion(user_id, event_channel, client_name="Customer"):
    try:
        if user_id not in user_states or 'name' not in user_states[user_id]:
            return {
                "blocks": [
                    {"type": "header", "text": {"type": "plain_text", "text": "*Promotion Generation 🖼️*"}},
                    {"type": "section", "text": {"type": "mrkdwn", "text": "Please register first using: `register: name, email, phone` 📝"}}
                ],
                "text": "Promotion failed: not registered."
            }
        customer_name = user_states[user_id]['name']
        prompt = f'Create a 50 percent offer poster for my shoe shop named "Smart Shoes". I need a colorful and attractive shoe image and my shop name "Smart Shoes" in center and the text "50 percent discount" highlighted for {client_name}.'
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        generated_image_data = None
        for part in response.parts:
            if hasattr(part, 'inline_data') and part.inline_data:
                generated_image_data = part.inline_data.data
                break
        if not generated_image_data:
            raise Exception("No image generated")
        image = Image.open(BytesIO(generated_image_data))
        img_byte_arr = BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        public_id = f"promotion_smartshoes_{uuid.uuid4().hex[:8]}"
        image_url = upload_to_cloudinary(file_bytes=img_byte_arr, public_id=public_id, resource_type="image")
        if not image_url:
            raise Exception("Failed to upload promotion image to Cloudinary")
        whatsapp_msg = f"Hey {customer_name} 👋, check out our latest promotion from Smart Shoes 👟 for {client_name}! Grab this 50% discount now. Visit us for exclusive deals just for you!"
        whatsapp_response = send_whatsapp_message(user_id, whatsapp_msg, image_url)
        with threading.Lock():
            client.files_upload_v2(
                channel=event_channel,
                file=BytesIO(requests.get(image_url).content),
                filename="promotion.png",
                title="Smart Shoes Promotion",
                initial_comment="Your promotion poster is ready!"
            )
        response = {
            "blocks": [
                {"type": "header", "text": {"type": "plain_text", "text": "*Promotion Generation 🖼️*"}},
                {"type": "section", "text": {"type": "mrkdwn", "text": f"Promotion poster uploaded to this channel!\nImage URL: {image_url}\n{whatsapp_response}"}}
            ],
            "text": "Promotion generated."
        }
        logging.info(f"Promotion generated for {user_id}: {image_url}")
        return response
    except Exception as e:
        logging.error(f"Promotion error for {user_id}: {e}")
        return {
            "blocks": [
                {"type": "header", "text": {"type": "plain_text", "text": "*Promotion Generation 🖼️*"}},
                {"type": "section", "text": {"type": "mrkdwn", "text": FALLBACK_RESPONSES["promotion"]}}
            ],
            "text": "Promotion generation failed."
        }

# Visualization
def generate_visualization(user_id, event_channel):
    try:
        if user_id not in user_states or 'name' not in user_states[user_id]:
            return {
                "blocks": [
                    {"type": "header", "text": {"type": "plain_text", "text": "*Sales Visualization 📊*"}},
                    {"type": "section", "text": {"type": "mrkdwn", "text": "Please register first using: `register: name, email, phone` 📝"}}
                ],
                "text": "Visualization failed: not registered."
            }
        plot_path = generate_plot()
        if not plot_path:
            raise Exception("Failed to generate plot")
        public_id = f"plot_smartshoes_{uuid.uuid4().hex[:8]}"
        plot_url = upload_to_cloudinary(file_path=plot_path, public_id=public_id, resource_type="image")
        if not plot_url:
            raise Exception("Failed to upload plot to Cloudinary")
        with threading.Lock():
            client.files_upload_v2(
                channel=event_channel,
                file=open(plot_path, 'rb'),
                filename=os.path.basename(plot_path),
                title="Sales Visualization",
                initial_comment="Your sales chart is ready!"
            )
        os.remove(plot_path)
        response = {
            "blocks": [
                {"type": "header", "text": {"type": "plain_text", "text": "*Sales Visualization 📊*"}},
                {"type": "section", "text": {"type": "mrkdwn", "text": f"Chart uploaded to this channel!\nImage URL: {plot_url}"}}
            ],
            "text": "Visualization generated."
        }
        logging.info(f"Visualization generated for {user_id}: {plot_url}")
        return response
    except Exception as e:
        logging.error(f"Visualization error for {user_id}: {e}")
        return {
            "blocks": [
                {"type": "header", "text": {"type": "plain_text", "text": "*Sales Visualization 📊*"}},
                {"type": "section", "text": {"type": "mrkdwn", "text": FALLBACK_RESPONSES["visualization"]}}
            ],
            "text": "Visualization generation failed."
        }

# Query Processing
def process_query(text, user_id, event_channel, event_ts):
    text = text.lower().strip()
    if user_id not in user_states:
        user_states[user_id] = {'last_message': '', 'context': 'idle'}
    state = user_states[user_id]
    state['last_message'] = text
    logging.info(f"Processing query: '{text}' from {user_id} in {event_channel}")

    cache_key = f"{user_id}_{text}_{event_ts}"
    current_time = time.time()
    if cache_key in response_cache and (current_time - response_cache[cache_key]['time'] < 10):
        logging.info(f"Skipping duplicate query: '{text}' from {user_id}")
        return response_cache[cache_key]['response']

    try:
        if any(greeting in text for greeting in ["hello", "hi", "how are you"]):
            response = {
                "blocks": [
                    {"type": "section", "text": {"type": "mrkdwn", "text": "Hey there! What's up? 😊 Try: register, generate invoice, promotion, visualization."}}
                ],
                "text": "Casual greeting response."
            }
        elif "register" in text:
            response = handle_customer_registration(user_id, text)
        elif "generate invoice" in text or "invoice" in text:
            response = generate_invoice(user_id, event_channel)
        elif "promotion" in text:
            client_name = re.search(r"for\s+(.+)", text, re.IGNORECASE)
            client_name = client_name.group(1) if client_name else "Customer"
            response = generate_promotion(user_id, event_channel, client_name)
        elif "visualization" in text or "chart" in text:
            response = generate_visualization(user_id, event_channel)
        else:
            model = genai.GenerativeModel('gemini-1.5-flash')
            gen_response = model.generate_content(f"Respond to: {text}")
            response = {
                "blocks": [
                    {"type": "header", "text": {"type": "plain_text", "text": "*General Query ❓*"}},
                    {"type": "section", "text": {"type": "mrkdwn", "text": gen_response.text.strip()}}
                ],
                "text": gen_response.text.strip()[:100] + "..."
            }

        response_cache[cache_key] = {'response': response, 'time': current_time}
        client.chat_postMessage(
            channel=event_channel,
            blocks=response["blocks"],
            text=response.get("text", "Smart Shoes response")
        )
        logging.info(f"Response sent to {event_channel} for {user_id}")
        return response
    except SlackApiError as e:
        logging.error(f"Slack post failed for {user_id}: {e}")
        if e.response["error"] == "not_in_channel":
            response = {
                "blocks": [
                    {"type": "header", "text": {"type": "plain_text", "text": "*Channel Issue 🚪*"}},
                    {"type": "section", "text": {"type": "mrkdwn", "text": FALLBACK_RESPONSES["not_in_channel"]}}
                ],
                "text": FALLBACK_RESPONSES["not_in_channel"]
            }
            client.chat_postMessage(channel=user_id, blocks=response["blocks"], text=response["text"])
        return response
    except Exception as e:
        logging.error(f"Query processing error for {user_id}: {e}")
        response = {
            "blocks": [
                {"type": "header", "text": {"type": "plain_text", "text": "*Error 🙁*"}},
                {"type": "section", "text": {"type": "mrkdwn", "text": FALLBACK_RESPONSES["default"]}}
            ],
            "text": "Query processing failed."
        }
        client.chat_postMessage(channel=event_channel, blocks=response["blocks"], text=response["text"])
        return response

# HTTP Server
class SlackEventHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/plain")
        self.end_headers()
        self.wfile.write(b"Smart Shoes Slack bot is running")

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
                    self.wfile.write(json.dumps({"status": "ok"}).encode('utf-8'))
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
    from slack_bolt import App
    from slack_bolt.adapter.socket_mode import SocketModeHandler
    slack_app = App(token=SLACK_BOT_TOKEN)

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
        process_query(text, user_id, event_channel, event_ts)

    threading.Thread(target=run_http_server, daemon=True).start()
    SocketModeHandler(slack_app, SLACK_APP_TOKEN).start()
