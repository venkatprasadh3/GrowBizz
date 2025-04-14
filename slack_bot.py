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
import requests
import json
from googletrans import Translator

# Configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN")
SLACK_APP_TOKEN = os.environ.get("SLACK_APP_TOKEN")
TWILIO_ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.environ.get("TWILIO_PHONE_NUMBER", "+14155238886")
CLOUDINARY_CLOUD_NAME = os.environ.get("CLOUDINARY_CLOUD_NAME", "dnnj6hykk")
CLOUDINARY_API_KEY = os.environ.get("CLOUDINARY_API_KEY", "991754979222148")
CLOUDINARY_API_SECRET = os.environ.get("CLOUDINARY_API_SECRET", "u-C4hv1OBts-wGDrkfDeGRv4OCk")
DEFAULT_LANGUAGE = "English"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
USERS_PATH = os.path.join(BASE_DIR, "users.csv")
SALES_DATA_PATH = os.path.join(BASE_DIR, "sales_data.csv")

user_states = {}
client = WebClient(token=SLACK_BOT_TOKEN)
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN) if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN else None
translator = Translator()
processed_events = set()
response_cache = {}

FALLBACK_RESPONSES = {
    "register": "Sorry, registration failed. Try again later. 🙁",
    "invoice": "Sorry, invoice generation failed. Please try again. 📄",
    "whatsapp": "Failed to send WhatsApp message. Please try again. 📱",
    "default": "Oops! I didn’t understand that. Try: register, generate invoice. 🤔"
}

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
    lang = user_states[user_id].get('language', DEFAULT_LANGUAGE)
    translated_msg = translate_message(message, lang)
    try:
        msg_params = {
            "body": translated_msg,
            "from_": f"whatsapp:{TWILIO_PHONE_NUMBER}",
            "to": f"whatsapp:{phone}"
        }
        if media_url:
            msg_params["media_url"] = [media_url]
        response = twilio_client.messages.create(**msg_params)
        logging.info(f"WhatsApp message SID: {response.sid}, Status: {response.status}, To: {phone}")
        return f"WhatsApp message sent to {phone}! 📱"
    except Exception as e:
        logging.error(f"WhatsApp error for {user_id}: {e}")
        return FALLBACK_RESPONSES["whatsapp"]

# Data Management
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

# **Customer Registration 📝**
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
            response = {
                "blocks": [
                    {"type": "header", "text": {"type": "plain_text", "text": "Registration Successful ✅"}},
                    {"type": "section", "text": {"type": "mrkdwn", "text": f"Welcome, {name}! Customer ID: `{customer_id}`"}}
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
        customer = user_states[user_id]
        invoice_url = "https://res.cloudinary.com/dnnj6hykk/image/upload/v1744547940/ivoice-test-GED_2_gmxls7.pdf"
        invoice_data = requests.get(invoice_url).content
        client.files_upload_v2(
            channel=event_channel,
            file=BytesIO(invoice_data),
            filename="invoice_A35432.pdf",
            title="Invoice A35432",
            initial_comment="Your invoice has been generated."
        )
        whatsapp_msg = f"Hey {customer['name']} 👋, thank you for your latest purchase with Smart Shoes 👟. Here's your invoice against your order A35432. Visit us again. We have exciting discounts *only for you*!"
        whatsapp_response = send_whatsapp_message(user_id, whatsapp_msg, media_url=invoice_url)
        response = {
            "blocks": [
                {"type": "header", "text": {"type": "plain_text", "text": "Invoice Generated 📄"}},
                {"type": "section", "text": {"type": "mrkdwn", "text": f"Invoice uploaded to this channel!\nURL: {invoice_url}\n{whatsapp_response}"}}
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
        if "register" in text:
            response = handle_customer_registration(user_id, text)
        elif "generate invoice" in text:
            response = generate_invoice(user_id, event_channel)
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
