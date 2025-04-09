# image_gen.py
import google.generativeai as genai
import os

# Configure Google Generative AI with environment variable
genai.configure(api_key=os.environ["GENAI_API_KEY"])
client = genai.GenerativeModel("gemini-1.5-flash")

def generate_promotion_image(prompt):
    contents = f"Create a promotional poster for {prompt}"
    try:
        response = client.generate_content(contents)
        for candidate in response.candidates:
            if hasattr(candidate.content, 'parts'):
                for part in candidate.content.parts:
                    if hasattr(part, 'inline_data') and part.inline_data:
                        return BytesIO(part.inline_data.data)
        return None
    except Exception as e:
        print(f"Error generating promotion image: {e}")
        return None
