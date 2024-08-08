import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

def get_google_response(prompt, system_prompt):
    model = genai.GenerativeModel(model_name="gemini-1.5-flash", system_instruction=system_prompt)
    response = model.generate_content([prompt])
    return response.text