import os
import re
import logging
import concurrent.futures
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
import google.generativeai as genai
import pandas as pd
from fpdf import FPDF
import matplotlib
from diffusers import StableDiffusionPipeline
import torch
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pywhatkit
import datetime
import smtplib
import seaborn as sns
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from textblob import TextBlob
from pytrends.request import TrendReq
import numpy as np
from scipy.stats import norm
import mysql.connector
from mysql.connector import Error
import uuid
import json
from sqlalchemy import create_engine
from decimal import Decimal

# Configuration and Constants
sns.set_style("darkgrid")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Application Constants
SALES_DATA_PATH = "sales_data.csv"
LOGO_PATH = "psg_logo_blue.png"
WHATSAPP_NUMBER = "+919944934545"
EMAIL_FROM = "21z268@psgtech.ac.in"
EMAIL_PASSWORD = "zlcd yfnp fhbl hvka"  # Ensure this is an App Password if 2FA is enabled
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN", "xoxb-8187318282212-8181982234117-HOs0CDRM1fO6H3KfLgqBW9ER")
SLACK_APP_TOKEN = os.getenv("SLACK_APP_TOKEN", "xapp-1-A085TESEZEV-8470325493031-a6d5494e5a64cd9c970f8acc9df1346e788a48a7f4315a8393bca927eec134a1")
GENAI_API_KEY = os.getenv("GENAI_API_KEY", "AIzaSyBjeahE-KUNTXCpd42RMeQ_IUVjHrvk9U0")
DEFAULT_LANGUAGE = "English"

# Database Configuration
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': 'root',
    'database': 'slack_user_db',
    'auth_plugin': 'mysql_native_password'
}
DB_URI = f"mysql+mysqlconnector://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}/{DB_CONFIG['database']}"

# Global Variables
user_states = {}
pipe = None
client = WebClient(token=SLACK_BOT_TOKEN)
model = genai.GenerativeModel("gemini-1.5-flash")
pytrends = TrendReq(hl='en-IN', tz=330)

# Messaging Functions
def send_whatsapp_message(phone_number, message):
    try:
        now = datetime.datetime.now()
        send_time_hour = now.hour
        send_time_minute = now.minute + 2
        pywhatkit.sendwhatmsg(phone_number, message, send_time_hour, send_time_minute)
        return f"WhatsApp message scheduled to {phone_number} successfully!"
    except Exception as e:
        logging.error(f"Error sending WhatsApp message: {e}")
        return f"An error occurred: {e}"

def send_email(subject, body, to_emails, from_email, password, images=None, attachments=None):
    if isinstance(to_emails, str):
        to_emails = [to_emails]
    to_emails = [re.sub(r'mailto:|\|.*$', '', email.strip()) for email in to_emails]
    logging.info(f"Sending email to: {to_emails}")

    msg = MIMEMultipart()
    msg["Subject"] = subject
    msg["From"] = from_email
    msg["To"] = ", ".join(to_emails)
    msg.attach(MIMEText(body, "plain"))

    if images:
        images = [images] if not isinstance(images, list) else images
        for image_path in images:
            with open(image_path, 'rb') as img_file:
                img = MIMEImage(img_file.read(), name=os.path.basename(image_path))
                msg.attach(img)

    if attachments:
        attachments = [attachments] if not isinstance(attachments, list) else attachments
        for attachment_path in attachments:
            with open(attachment_path, 'rb') as file:
                part = MIMEApplication(file.read(), name=os.path.basename(attachment_path))
                part['Content-Disposition'] = f'attachment; filename="{os.path.basename(attachment_path)}"'
                msg.attach(part)

    smtp = None
    try:
        smtp = smtplib.SMTP("smtp.gmail.com", 587)
        smtp.starttls()
        smtp.login(from_email, password)
        smtp.sendmail(from_email, to_emails, msg.as_string())
        logging.info("Email sent successfully via port 587!")
        return "Email sent successfully!"
    except Exception as e:
        logging.error(f"Failed to send email via port 587: {e}")
        try:
            smtp = smtplib.SMTP_SSL("smtp.gmail.com", 465)
            smtp.login(from_email, password)
            smtp.sendmail(from_email, to_emails, msg.as_string())
            logging.info("Email sent successfully via port 465!")
            return "Email sent successfully!"
        except Exception as e2:
            logging.error(f"Failed to send email via port 465: {e2}")
            return f"Failed to send email: {e2}"
    finally:
        if smtp:
            smtp.quit()

# Initialize AI Models
def initialize_models():
    global pipe
    genai.configure(api_key=GENAI_API_KEY)
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    ).to("cuda" if torch.cuda.is_available() else "cpu")

# Database Functions
def create_db_connection():
    try:
        return mysql.connector.connect(**DB_CONFIG)
    except Error as e:
        logging.error(f"Database connection error: {e}")
        return None

def initialize_databases():
    conn = create_db_connection()
    if not conn:
        return False
    try:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                customer_id VARCHAR(36) PRIMARY KEY,
                name VARCHAR(255),
                email VARCHAR(255) UNIQUE,
                phone VARCHAR(20),
                address TEXT,
                communication_language VARCHAR(50) DEFAULT 'English',
                slack_id VARCHAR(255) UNIQUE,
                registration_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sales (
                sale_id VARCHAR(36) PRIMARY KEY,
                customer_id VARCHAR(36),
                product_details JSON,
                sale_amount DECIMAL(10,2),
                sale_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (customer_id) REFERENCES users(customer_id)
            );
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS inventory (
                product_id VARCHAR(36) PRIMARY KEY,
                product_name VARCHAR(255),
                price DECIMAL(10,2),
                stock_count INT DEFAULT 0,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        conn.commit()
        return True
    except Error as e:
        logging.error(f"Database initialization error: {e}")
        return False
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

# Load Sales Data
def load_sales_data():
    try:
        df = pd.read_csv(SALES_DATA_PATH, low_memory=False)
        df['Order Date'] = pd.to_datetime(df['Order Date'], format='%m/%d/%y %H:%M', errors='coerce')
        df.dropna(subset=['Order Date'], inplace=True)
        df['Quantity Ordered'] = pd.to_numeric(df['Quantity Ordered'], errors='coerce').fillna(0).astype('Int64')
        df['Price Each'] = pd.to_numeric(df['Price Each'], errors='coerce').fillna(0.0)
        return df
    except Exception as e:
        return f"Error loading sales data: {e}"

# Customer Registration
def handle_customer_registration(user_id, text):
    if user_id not in user_states:
        user_states[user_id] = {'step': 0}

    state = user_states[user_id]
    
    if state['step'] == 0:
        state['step'] = 1
        return "Please provide registration details in format:\n`register: name, email@example.com, +1234567890, English, 123 Street Name, City`"
    
    conn = None
    try:
        _, details = text.split("register:", 1)
        parts = [x.strip() for x in details.split(',', 5)]
        if len(parts) != 5:
            return "Invalid format. Need: name, email, phone, language, address"
        
        name, email, phone, language, address = parts
        
        email = re.sub(r'mailto:|\|.*$', '', email)
        phone = re.sub(r'tel:|\|.*$', '', phone)
        
        if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
            return "Invalid email format."
        if len(phone) > 20:
            return "Phone number too long. Maximum 20 characters allowed (e.g., +1234567890123456789)."

        customer_id = str(uuid.uuid4())
        conn = create_db_connection()
        if conn is None:
            return "Database connection failed. Please try later."

        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT customer_id FROM users WHERE slack_id = %s", (user_id,))
        existing_user = cursor.fetchone()
        
        if existing_user:
            return f"User with Slack ID {user_id} is already registered with Customer ID: `{existing_user['customer_id']}`. Please use a different Slack account or contact support."

        cursor.execute(
            """INSERT INTO users 
            (customer_id, name, email, phone, address, communication_language, slack_id) 
            VALUES (%s, %s, %s, %s, %s, %s, %s)""",
            (customer_id, name, email, phone, address, language, user_id)
        )
        conn.commit()
        del user_states[user_id]
        
        welcome_msg = translate_message(f"Welcome {name}! Registration successful.", language)
        send_whatsapp_message(phone, welcome_msg)
        send_email("Registration Confirmation", welcome_msg, [email], EMAIL_FROM, EMAIL_PASSWORD)
        
        return f"Registration successful! Customer ID: `{customer_id}`"
    except Error as e:
        logging.error(f"Registration error: {e}")
        return f"Registration failed due to: {str(e)}"
    finally:
        if conn and conn.is_connected():
            cursor.close()
            conn.close()

# Promotion Generation with Parallel Processing
def generate_image_in_parallel(prompt, image_path):
    try:
        enhanced_prompt = f"{prompt} | vibrant and attractive product promotion, clear text, professional design"
        image = pipe(enhanced_prompt).images[0]
        image.save(image_path)
        logging.info(f"Image saved at: {image_path}")
        return image_path
    except Exception as e:
        logging.error(f"Parallel image generation failed: {e}")
        return None

def generate_promotion(prompt):
    try:
        logging.info("Starting promotion image generation in parallel...")
        image_path = "promotion_image.png"
        text_msg = f"🚀 Promotion Alert! Get 20% off on {prompt.split(':')[-1].strip()}!"

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(generate_image_in_parallel, prompt, image_path)
            return f"{text_msg}\nPromotional image is being generated and will be saved at: {image_path}"
    except Exception as e:
        logging.error(f"Image generation setup failed: {e}")
        return "Image generation failed. Please try again."

# Weekly Sales Analysis
def generate_weekly_sales_analysis():
    df = load_sales_data()
    if isinstance(df, str):
        return df
    
    try:
        weekly_sales = df.resample('W', on='Order Date')['Price Each'].sum()
        plt.figure(figsize=(12, 6))
        plt.plot(weekly_sales.index, weekly_sales.values, marker='o', linestyle='-', color='#4CAF50')
        plt.title('Weekly Sales Trend')
        plt.xlabel('Week')
        plt.ylabel('Sales (INR)')
        plt.grid(True)
        plt.savefig("weekly_trend.png")
        plt.close()
        
        plt.figure(figsize=(12, 6))
        plt.plot(df['Order Date'], df['Price Each'].cumsum(), color='#2196F3')
        plt.title('Overall Sales Trend')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Sales (INR)')
        plt.grid(True)
        plt.savefig("overall_trend.png")
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
        plt.savefig("sales_distribution.png")
        plt.close()
        
        return ("Weekly analysis generated:\n"
                "- Weekly Trend: weekly_trend.png\n"
                "- Overall Trend: overall_trend.png\n"
                "- Sales Distribution: sales_distribution.png")
    except Exception as e:
        logging.error(f"Analysis error: {e}")
        return f"Analysis error: {str(e)}"

# Multilingual Support
def get_user_language(user_id):
    conn = create_db_connection()
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT communication_language FROM users WHERE slack_id = %s", (user_id,))
        result = cursor.fetchone()
        return result['communication_language'] if result else DEFAULT_LANGUAGE
    except Error as e:
        logging.error(f"Language fetch error: {e}")
        return DEFAULT_LANGUAGE
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

def translate_message(text, target_lang):
    try:
        response = model.generate_content(f"Translate this to {target_lang}: {text}")
        return response.text.strip()
    except Exception as e:
        logging.error(f"Translation error: {e}")
        return text

# Purchase Processing
def process_purchase(product_id, user_id):
    conn = create_db_connection()
    if not conn:
        return "Database connection failed."
        
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM users WHERE slack_id = %s", (user_id,))
        user = cursor.fetchone()
        if not user:
            return "Please register first using: register: name, email, phone, language, address"
        
        cursor.execute("SELECT * FROM inventory WHERE product_id = %s", (product_id,))
        product = cursor.fetchone()
        if not product or product['stock_count'] < 1:
            return "Product out of stock."
            
        sale_id = str(uuid.uuid4())  # Ensure this is a string
        sale_details = {
            "product_id": product['product_id'],
            "product_name": product['product_name'],
            "price": float(product['price'])  # Convert Decimal to float
        }
        cursor.execute("""
            INSERT INTO sales (sale_id, customer_id, product_details, sale_amount)
            VALUES (%s, %s, %s, %s)
        """, (sale_id, user['customer_id'], json.dumps(sale_details), float(product['price'])))
        
        cursor.execute("UPDATE inventory SET stock_count = stock_count - 1 WHERE product_id = %s", (product_id,))
        conn.commit()
        
        # Pass customer_id as a string explicitly
        invoice_path = generate_invoice(str(user['customer_id']), product)
        user_lang = get_user_language(user_id)
        
        message = f"Purchase confirmed: {product['product_name']} for INR {float(product['price'])}"
        translated_msg = translate_message(message, user_lang)
        
        send_whatsapp_message(user['phone'], translated_msg)
        send_email(
            subject=translate_message("Purchase Confirmation", user_lang),
            body=f"{translated_msg}\nInvoice attached at: {invoice_path}",
            to_emails=[user['email']],
            from_email=EMAIL_FROM,
            password=EMAIL_PASSWORD,
            attachments=[invoice_path]
        )
        
        return f"Purchase completed! Sale ID: {sale_id}\nInvoice: {invoice_path}"
    except Error as e:
        logging.error(f"Purchase error: {e}")
        return f"Purchase failed: {str(e)}"
    except Exception as e:
        logging.error(f"Unexpected purchase error: {e}")
        return f"Purchase failed due to: {str(e)}"
    finally:
        if conn and conn.is_connected():
            cursor.close()
            conn.close()

# Invoice Generation
def generate_invoice(customer_id=None, product=None):
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

    if customer_id and product:
        conn = create_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM users WHERE customer_id = %s", (customer_id,))
        customer = cursor.fetchone()
        if not customer:
            return "Customer not found in database."
        
        company_info = "PSG College of Technology\nCoimbatore, Tamil Nadu\nPhone: 123-456-7890"
        client_info = f"{customer['name']}\n{customer['address']}\n{customer['email']}\n{customer['phone']}"
        header = ["Product", "Qty", "Unit Price", "Total"]
        invoice_items = [[product['product_name'], 1, float(product['price']), float(product['price'])]]
        
        total_before_gst = float(product['price'])  # Convert Decimal to float
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
        
        invoice_path = f"invoice_{customer_id}_{uuid.uuid4()[:8]}.pdf"
        if conn.is_connected():
            cursor.close()
            conn.close()
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
        
        invoice_path = "invoice_dynamic.pdf"
    
    pdf.output(invoice_path)
    return invoice_path

# Sales Insights and Recommendations
def generate_sales_recommendations():
    df = load_sales_data()
    if isinstance(df, str):
        return df
    
    try:
        engine = create_engine(DB_URI)
        inventory_df = pd.read_sql("SELECT * FROM inventory", engine)
        
        weekly_sales = df.resample('W', on='Order Date')['Price Each'].sum()
        total_sales = df['Price Each'].sum()
        top_product = df.groupby("Product")['Quantity Ordered'].sum().idxmax()
        
        pytrends.build_payload(kw_list=[top_product], timeframe='now 7-d')
        trends = pytrends.interest_over_time()
        trend_score = trends[top_product].mean() / 100 if top_product in trends else 0
        
        recommendations = []
        for _, product in inventory_df.iterrows():
            product_sales = df[df['Product'] == product['product_name']]['Price Each'].sum()
            sales_score = product_sales / total_sales if total_sales > 0 else 0
            stock_score = 1 - (product['stock_count'] / 100)
            weighted_score = 0.4 * sales_score + 0.3 * stock_score + 0.3 * trend_score
            
            if weighted_score < 0.3:
                recommendations.append(f"Decrease price or promote {product['product_name']} (Score: {weighted_score:.2f})")
            elif weighted_score > 0.7:
                recommendations.append(f"Increase price for {product['product_name']} (Score: {weighted_score:.2f})")
            if product['stock_count'] < 10:
                recommendations.append(f"Restock {product['product_name']} (Current: {product['stock_count']})")
        
        insights = (
            f"📊 Sales Insights:\n"
            f"• Total Sales: INR {total_sales:,.2f}\n"
            f"• Top Product: {top_product}\n"
            f"• Trend Score: {trend_score:.2f}\n\n"
            f"📦 Inventory Recommendations:\n" + "\n".join(recommendations)
        )
        return insights
    except Exception as e:
        logging.error(f"Recommendations error: {e}")
        return f"Error generating recommendations: {str(e)}"

def generate_sales_insights():
    df = load_sales_data()
    if isinstance(df, str):
        return df
    
    total_sales = df["Price Each"].sum()
    avg_sale = df["Price Each"].mean()
    best_selling_product = df.groupby("Product")["Quantity Ordered"].sum().idxmax()
    
    return f"📊 Sales Insights:\n" \
           f"🔹 Total Sales: INR {total_sales:,.2f}\n" \
           f"🔹 Average Sale: INR {avg_sale:,.2f}\n" \
           f"🔥 Best Selling Product: {best_selling_product}"

# Query Processing
def process_query(text, user_id):
    text = text.lower()
    
    if text.startswith("register:"):
        return handle_customer_registration(user_id, text)
    elif text.startswith("purchase:"):
        product_id = text.split(":")[1].strip()
        return process_purchase(product_id, user_id)
    elif "weekly analysis" in text:
        return generate_weekly_sales_analysis()
    elif "insights" in text:
        return generate_sales_recommendations()
    elif "simple insights" in text:
        return generate_sales_insights()
    elif "promotion" in text:
        return generate_promotion(text.replace("promotion:", "").strip())
    elif "whatsapp" in text:
        message = "Hello, this is a test message from the bot!"
        if "in " in text:
            lang = text.split("in ")[-1].strip()
            message = translate_message(message, lang)
        return send_whatsapp_message(WHATSAPP_NUMBER, message)
    elif "email" in text:
        df = load_sales_data()
        if isinstance(df, str):
            return df
        top_customers = df.groupby("Purchase Address")["Price Each"].sum().nlargest(3)
        subject = "Top Customers Report"
        body = f"""
Hello,
Here are the top 3 customers with the highest purchase amounts:

{top_customers.to_string()}

Thank you for your business!

Best Regards,
PSG College of Technology
"""
        if "in " in text:
            lang = text.split("in ")[-1].strip()
            body = translate_message(body, lang)
        return send_email(subject, body, [EMAIL_FROM], EMAIL_FROM, EMAIL_PASSWORD, images=[LOGO_PATH], attachments=["invoice_dynamic.pdf"])
    elif "invoice" in text:
        return generate_invoice()
    else:
        return "Command not recognized. Available: register, purchase, weekly analysis, insights, simple insights, promotion, whatsapp, email, invoice"

# Slack Bot Setup
if __name__ == "__main__":
    initialize_databases()
    initialize_models()
    
    from slack_bolt import App
    from slack_bolt.adapter.socket_mode import SocketModeHandler

    app = App(token=SLACK_BOT_TOKEN)
    
    @app.message(".*")
    def handle_message(event, say):
        user_id = event['user']
        text = event['text']
        response = process_query(text, user_id)
        say(response)

    SocketModeHandler(app, SLACK_APP_TOKEN).start()