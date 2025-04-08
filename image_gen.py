# image_gen.py
import google.generativeai as genai
from google.generativeai import types
from PIL import Image
from io import BytesIO

# Configure the client with the API key
client = genai.Client(api_key="AIzaSyBjeahE-KUNTXCpd42RMeQ_IUVjHrvk9U0")

def generate_promotion_image(prompt, discount_percentage, shop_name="Biggies Burger"):
    """
    Generate a promotion image based on the prompt, discount percentage, and shop name.
    Returns a BytesIO object containing the image.
    """
    try:
        # Construct the content string for Gemini
        content = (
            f"Create a {discount_percentage}% discount poster for my shop '{shop_name}'. "
            f"Include an image related to '{prompt}', the shop name '{shop_name}' in the center, "
            f"and highlight the {discount_percentage}% discount."
        )

        # Generate content with text and image modalities
        response = client.models.generate_content(
            model="gemini-2.0-flash-exp-image-generation",
            contents=content,
            config=types.GenerateContentConfig(
                response_modalities=['Text', 'Image']
            )
        )

        # Extract the image from the response
        for part in response.candidates[0].content.parts:
            if part.inline_data is not None:
                image = Image.open(BytesIO(part.inline_data.data))
                img_byte_arr = BytesIO()
                image.save(img_byte_arr, format='PNG')
                img_byte_arr.seek(0)
                return img_byte_arr
        return None  # No image found in response
    except Exception as e:
        print(f"Error generating promotion image: {e}")
        return None
