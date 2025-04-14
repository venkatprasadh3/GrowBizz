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
import time
import tempfile
import google.generativeai as genai
from google.generativeai.types import GenerateContentConfig
from PIL import Image
import cloudinary
import cloudinary.uploader
from googletrans import Translator

# Configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN")
SLACK_APP_TOKEN = os.environ.get("SLACK_APP_TOKEN")
TWILIO_ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN")
TWILIO_PROMO_ACCOUNT_SID = os.environ.get("TWILIO_PROMO_ACCOUNT_SID")
TWILIO_PROMO_AUTH_TOKEN = os.environ.get("TWILIO_PROMO_AUTH_TOKEN")
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
    "TWILIO_PROMO_ACCOUNT_SID", "TWILIO_PROMO_AUTH_TOKEN",
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

user_states = {}
client = WebClient(token=SLACK_BOT_TOKEN)
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN) if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN else None
translator = Translator()
processed_events = set()
response_cache = {}

FALLBACK_RESPONSES = {
    "register": "Sorry, registration failed. Try again later. 🙁",
    "invoice": "Sorry, invoice generation failed. Please try again. 📄",
    "promotion": "Sorry, promotion generation failed. Please try again. 🖼️",
    "whatsapp": "Failed to send WhatsApp message. Please try again. 📱",
    "default": "Oops! I didn’t understand that. Try: register, generate invoice, generate promotion. 🤔"
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

# **Promotion Generation 🖼️**
def generate_promotion(user_id, event_channel):
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
        # Gemini Image Generation
        genai.configure(api_key=GENAI_API_KEY)
        image_client = genai.GenerativeModel("gemini-2.0-flash")
        contents = ('Hi, can you create a 50 percent offer poster for my shoe shop named "Smart Shoes". I need a colorful and attractive shoe image and my shop name "Smart Shoes" in centre and the text "50 percent discount" highlighted')
        response = image_client.generate_content(
            contents,
            generation_config=GenerateContentConfig(
                response_mime_type="image/png"
            )
        )
        generated_image_data = None
        for part in response.parts:
            if hasattr(part, 'inline_data') and part.inline_data:
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
            twilio_client_specific = Client(TWILIO_PROMO_ACCOUNT_SID, TWILIO_PROMO_AUTH_TOKEN)
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
                    {"type": "section", "text": {"type": "mrkdwn", "text": f"Promotion poster uploaded to this channel!\nURL: {image_url}\n{whatsapp_response}"}}
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
        genai.configure(api_key=GENAI_API_KEY)
        client = genai.GenerativeModel("gemini-2.0-flash")
        with open(audio_file_path, 'rb') as audio_file:
            audio_data = audio_file.read()
        contents = [
            prompt,
            {"inline_data": {"mime_type": "audio/mp3", "data": audio_data}}
        ]
        response = client.generate_content(contents)
        return response.text
    except Exception as e:
        return f"An error occurred: {e}"

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
        elif "generate promotion" in text:
            response = generate_promotion(user_id, event_channel)
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
    def handle_app_mention(event, say, client, logger):
        text = event["text"]
        channel_id = event["channel"]
        user_id = event["user"]
        # Respond to the mention
        try:
            client.chat_postMessage(
                channel=channel_id,
                text=f"Hi <@{user_id}>! You mentioned me. How can I help you with text or audio files today? Please upload an audio file with your request in the caption, or just send a text message.",
                thread_ts=event.get("thread_ts")
            )
        except Exception as e:
            logger.error(f"Error responding to app mention: {e}")

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
