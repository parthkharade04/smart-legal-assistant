import os
import google.generativeai as genai

GOOGLE_API_KEY = "AIzaSyB0Tf5waTTwUjjt3huf61IyXjeq4zcskyo"
genai.configure(api_key=GOOGLE_API_KEY)

print("Listing available models...")
try:
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"- {m.name}")
except Exception as e:
    print(f"Error: {e}")
