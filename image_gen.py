# image_gen.py
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont

def generate_promotion_image(prompt):
    try:
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
