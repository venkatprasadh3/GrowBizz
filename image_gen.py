# image_gen.py
import google.generativeai as genai
from PIL import Image
from io import BytesIO

def generate_promotion_image(prompt, discount_percentage, shop_name="Biggies Burger", api_key=None):
    """
    Generate a promotion image based on the prompt, discount percentage, and shop name.
    Returns a BytesIO object containing the image or None if generation fails.
    
    Args:
        prompt (str): Description of the promotion (e.g., "burgers").
        discount_percentage (str): Discount amount (e.g., "50").
        shop_name (str): Name of the shop (default: "Biggies Burger").
        api_key (str): Google Generative AI API key (required).
    
    Returns:
        BytesIO: Image data if successful, None otherwise.
    """
    try:
        # Check for API key
        if not api_key:
            raise ValueError("API key is required for image generation.")

        # Configure the API key
        genai.configure(api_key=api_key)

        # Construct the content string for Gemini
        content = (
            f"Create a {discount_percentage}% discount poster for my shop '{shop_name}'. "
            f"Include an image related to '{prompt}', the shop name '{shop_name}' in the center, "
            f"and highlight the {discount_percentage}% discount."
        )

        # Initialize the generative model
        # Note: Verify the correct model name in Google Generative AI documentation
        model = genai.GenerativeModel("gemini-1.5-flash")  # Adjust model name as needed

        # Generate content (assuming text + image output)
        response = model.generate_content(
            content,
            generation_config={
                "response_mime_type": "image/png"  # Adjust based on actual API capability
            }
        )

        # Extract the image from the response
        # Note: Response structure may vary; adjust based on actual API output
        for candidate in response.candidates:
            for part in candidate.content.parts:
                if hasattr(part, 'inline_data') and part.inline_data is not None:
                    image = Image.open(BytesIO(part.inline_data.data))
                    img_byte_arr = BytesIO()
                    image.save(img_byte_arr, format='PNG')
                    img_byte_arr.seek(0)
                    return img_byte_arr
        print("No image found in response.")
        return None
    except Exception as e:
        print(f"Error generating promotion image: {e}")
        return None
