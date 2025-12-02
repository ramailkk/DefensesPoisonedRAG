import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()

GEMINI_KEY = os.getenv("SECRET_GEMINI_KEY")
# assert GEMINI_KEY is not None, "GEMINI_API_KEY is not set"

class GPT():
    def __init__(self, model_name):
        self.model_name = model_name
        self.max_output_tokens = 1024
        genai.configure(api_key=GEMINI_KEY)
        self.model = genai.GenerativeModel(model_name)

    def query(self, msg, temperature=0.1):
        try:
            response = self.model.generate_content(
                msg,
                generation_config={
                    "temperature": temperature,
                    "max_output_tokens": self.max_output_tokens,
                }
            )
            return response.text
        except Exception as e:
            print(e)
            return ""
