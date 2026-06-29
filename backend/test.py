import os
from dotenv import load_dotenv
from google import genai

load_dotenv()  # Loads variables from .env

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Say hello in one sentence."
)

print(response.text)