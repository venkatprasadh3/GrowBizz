from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO
import base64

client = genai.Client(api_key="AIzaSyBjeahE-KUNTXCpd42RMeQ_IUVjHrvk9U0")

contents = ('Hi, can you create a 50 percent offer poster for my burger shop named "Biggies Burger".I need a burger image and my shop name "Biggies Burger" in centre and the number 50 percent discount highlighted')

response = client.models.generate_content(
    model="gemini-2.0-flash-exp-image-generation",
    contents=contents,
    config=types.GenerateContentConfig(
      response_modalities=['Text', 'Image']
    )
)

for part in response.candidates[0].content.parts:
  if part.text is not None:
    print(part.text)
  elif part.inline_data is not None:
    image = Image.open(BytesIO((part.inline_data.data)))
    image.save('gemini-native-image.png')
    image.show()