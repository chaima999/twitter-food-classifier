import pandas as pd
from deep_translator import GoogleTranslator
import time

# Load the dataset
df = pd.read_csv('submission.csv')  # Replace 'tweets.csv' with your file path

# Initialize the translator
translator = GoogleTranslator(source='auto', target='en')

# Function to translate text to English with error handling and delays
def translate_to_english(text):
    try:
        # Add a delay to avoid being blocked (1 second delay per request)
        time.sleep(1)
        # Translate the text
        translation = translator.translate(text)
        return translation
    except Exception as e:
        print(f"Error translating text: {e}")
        return text  # Return the original text if translation fails

# Apply translation to the 'tweet' column
print("Starting translation...")
df['text'] = df['text'].apply(translate_to_english)

# Save the translated data to a new CSV file
df.to_csv('submission_data.csv', index=False)
print("Translation completed! Saved to 'submission_data.csv'.")
