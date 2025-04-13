import os
import re
from urllib import request

import pandas as pd
import io
import plotly.express as px
import base64
import json
from google import genai
import random
import datetime
from fastapi import FastAPI, HTTPException, Request, APIRouter
from fastapi.responses import JSONResponse
# Configure your Gemini API key (replace with your actual API key)
client = genai.Client(api_key=")

PLOTS_DIR = os.path.dirname(os.path.abspath(__file__))
SALES_DATA_PATH = os.path.join(BASE_DIR, "sales_data.csv")

def extract_python_code(llm_response):
    """Extracts Python code from the LLM's response, handling code blocks."""
    match = re.search(r"```python\n(.*?)\n```", llm_response, re.DOTALL)
    if match:
        return match.group(1)
    else:
        return llm_response

def fetch_csv_content(csv_path):
    """
    Fetches the content of a CSV file from the given path.

    Args:
        csv_path (str): The path to the CSV file.

    Returns:
        str or None: The content of the CSV file as a string, or None if an error occurs.
    """
    try:
        with open(csv_path, 'r') as f:
            csv_content = f.read()
        return csv_content
    except FileNotFoundError:
        print(f"Error: CSV file not found at path: {csv_path}")
        return None
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None
    
def process_csv_and_query(csv_content, user_query):
    """
    Processes a CSV file, answers user questions, or generates visualizations using Gemini.
    """
    try:
        df = pd.read_csv(io.StringIO(csv_content))
        df_summary = df.describe().to_string()
        df_head = df.to_string()

        prompt = f"""
        You are a data analysis assistant. You have access to the following CSV data:

        Summary:
        {df_summary}

        All Rows:
        {df_head}

        The user asked: {user_query}

        Respond with the answer or visualization code. If the user asks for a visualization, return plotly express code.Return just the code do not add any sentence in your response.Make the plots look professional sleek and very much attractive as if it is created by an expert.Give proper spacing between the bars in bargraph. If the user asks for a number, return just the number.
        """
        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=prompt,
        )
        llm_response = response.text
        print(llm_response)

        if "px." in llm_response.lower():
            try:
                code_to_execute = extract_python_code(llm_response)
                local_vars = {"df": df, "px": px}
                exec(code_to_execute, globals(), local_vars)
                fig = local_vars.get("fig")
                if fig is not None:
                    os.makedirs(PLOTS_DIR, exist_ok=True)

                    # Save the plot as a PNG file
                    plot_filename = os.path.join(PLOTS_DIR,
                                                 f"plot_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
                    try:
                        fig.write_image(plot_filename)
                        print(f"Plot saved as: {plot_filename}")
                    except Exception as e:
                        print(f"Error saving plot: {e}")

                    # Encode the image to base64
                    img_data = fig.to_image(format="png")
                    img_base64 = base64.b64encode(img_data).decode('utf-8')
                    return {"response": f"data:image/png;base64,{img_base64}", "type": "image"}

                else:
                    return {"response": "The Gemini model returned code, but a figure was not created.", "type": "text"}

            except Exception as e:
                return {"response": f"Error generating visualization: {e}", "type": "text"}
        else:
            return {"response": llm_response, "type": "text"}

    except Exception as e:
        return {"response": f"Error processing CSV: {e}", "type": "text"}
    
csv_file_path = os.path.join(PLOTS_DIR, "sales_data.csv")

csv_data = fetch_csv_content(csv_file_path)

prompt = "create a pie chart showing how many products sold were having prize less than 500 and greater than 500"


response = process_csv_and_query(csv_data, prompt)
print(f"\nUser Query: {prompt}")
print(json.dumps(response, indent=2))
