# image_gen.py
from google.generativeai import genai
from google.genai import types
from PIL import Image
from io import BytesIO
import base64
import os

# Get API key from environment variable
api_key = os.environ["GENAI_API_KEY"]
client = genai.Client(api_key=api_key)

def generate_promotion_image(prompt):
    contents = f"Create a promotional poster for {prompt}"
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash-exp-image-generation",
            contents=contents,
            config=types.GenerateContentConfig(
                response_modalities=['Text', 'Image']
            )
        )
        for part in response.candidates[0].content.parts:
            if part.inline_data is not None:
                image = Image.open(BytesIO(base64.b64decode(part.inline_data.data)))
                img_byte_arr = BytesIO()
                image.save(img_byte_arr, format='PNG')
                img_byte_arr.seek(0)
                return img_byte_arr
        return None
    except Exception as e:
        print(f"Error generating promotion image: {e}")
        return None
