from io import BytesIO
from PIL import Image, ImageDraw, ImageFont

def generate_promotion_image(prompt):
    try:
        # Create a larger image for better visibility
        img = Image.new('RGB', (800, 400), color='white')
        d = ImageDraw.Draw(img)
        
        # Use a larger, bold font
        try:
            font = ImageFont.truetype("arial.ttf", 40)  # Try system font
        except:
            font = ImageFont.load_default()  # Fallback to default if arial.ttf unavailable
        
        # Calculate text size and position for centering
        text_bbox = d.textbbox((0, 0), f"Promotion: {prompt}", font=font)
        text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
        x = (800 - text_width) / 2
        y = (400 - text_height) / 2
        
        # Draw text with black fill for visibility
        d.text((x, y), f"Promotion: {prompt}", fill='black', font=font)
        
        img_byte_arr = BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        return img_byte_arr
    except Exception as e:
        logging.error(f"Error generating promotion image: {e}")
        return None
