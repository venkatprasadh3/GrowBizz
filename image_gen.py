# image_gen.py
import google.generativeai as genai
import os
from io import BytesIO

# Configure Google Generative AI with environment variable
genai.configure(api_key=os.environ["GENAI_API_KEY"])

def generate_promotion_image(prompt):
    # Fallback to PIL since Google API may not generate images in this version
    try:
        from PIL import Image, ImageDraw, ImageFont
        img = Image.new('RGB', (400, 200), color='white')
        d = ImageDraw.Draw(img)
        font = ImageFont.load_default()
        d.text((10, 10), f"Promotion: {prompt}", fill='black', font=font)
        img_byte_arr = BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        return img_byte_arr
    except Exception as e:
        print(f"Error generating promotion image: {e}")
        return None
