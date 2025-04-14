import os
import re
import pandas as pd
import io
import plotly.express as px
import datetime
import google.generativeai as genai
import cloudinary
import cloudinary.uploader

# Configuration
GENAI_API_KEY = os.environ.get("GENAI_API_KEY")
CLOUDINARY_CLOUD_NAME = os.environ.get("CLOUDINARY_CLOUD_NAME", "dnnj6hykk")
CLOUDINARY_API_KEY = os.environ.get("CLOUDINARY_API_KEY", "991754979222148")
CLOUDINARY_API_SECRET = os.environ.get("CLOUDINARY_API_SECRET", "u-C4hv1OBts-wGDrkfDeGRv4OCk")

genai.configure(api_key=GENAI_API_KEY)
cloudinary.config(
    cloud_name=CLOUDINARY_CLOUD_NAME,
    api_key=CLOUDINARY_API_KEY,
    api_secret=CLOUDINARY_API_SECRET,
    secure=True
)

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
        response = genai.GenerativeModel('gemini-1.5-flash').generate_content(prompt)
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
