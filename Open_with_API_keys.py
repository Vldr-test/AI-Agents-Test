import langchain
import openai
from openai import OpenAI
import google.generativeai as genai
import dotenv
import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

openai_api_key = os.getenv("OPENAI_API_KEY")
#print(os.environ)

if not openai_api_key:
    print("Error: OPENAI_API_KEY environment variable not set.")
else:
    client = OpenAI(api_key=openai_api_key)  # Initialize the OpenAI client
    try:
        models = client.models.list()
        print("OpenAI API Key test successful!")
        print("Successfully retrieved OpenAI models list (using new API).")
        # You can uncomment the line below to print the list of models if you want
        # print(models)
    except Exception as e:
        print(f"Error testing OpenAI API Key: {e}")

print("\n--- Test script finished ---")

gemini_api_key = os.getenv("GOOGLE_API_KEY")

if not gemini_api_key:
    print("Error: GOOGLE_API_KEY environment variable not set.")
else:
    genai.configure(api_key=gemini_api_key)
    try:
        # Example: List available models (a simple API call to test key)
        models = genai.list_models()
        print("Gemini API Key test successful!")
        print("Successfully retrieved Gemini models list.")
        # You can uncomment the line below to print the list of models if you want
        print(models)
    except Exception as e:
        print(f"Error testing Gemini API Key: {e}")

print("\n--- Test script finished ---")