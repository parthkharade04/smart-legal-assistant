from google import genai
import os

API_KEY = "AIzaSyB0Tf5waTTwUjjt3huf61IyXjeq4zcskyo"
# 2. Initialize the Client
client = genai.Client(api_key=API_KEY)

# 3. Call the model (using Gemini 2.5 Flash)
response = client.models.generate_content(
    model="gemini-2.5-flash", 
    contents="Write a short poem about coding."
)

# 4. Print the result
print(response.text)
