from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import openai
import re
import requests
import json


# Load the CSV file
df = pd.read_csv("wbw.csv")

# Preprocess the data (replace 'embedding' with actual column name if different)
df['embedding'] = df['embedding'].apply(lambda x: np.array(eval(x)))
df['content_clean'] = df['content'].apply(lambda x: re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', '', x.strip())))

# Function to find the closest match
def find_closest_match(query_embedding, threshold=0.5):
    similarities = cosine_similarity([query_embedding], df['embedding'].tolist())
    index = np.argmax(similarities)
    if similarities[0][index] < threshold:
        return None
    return df.iloc[index]

# Function to generate human-like response using the Chat API
def generate_response(post_title, post_url, content):
    openai.api_key = "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"
  
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are Tim Urban's helpful assistant that is in a chatbot. You generate Tim Urban (waitbutwhy) like responses for given information",
            },
            {
                "role": "user",
                "content": f"Tell in detail in a neat readable format about your post titled {post_title} with its content'{content}' and URL {post_url}.",
            },
        ],
    )

    response = re.sub(r'\s+', ' ', response["choices"][0]["message"]["content"]).strip()
    return response

# Function to get the embedding for a given input text
def get_embedding(text):
    API_KEY = "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"
    url = "https://api.openai.com/v1/embeddings"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    data = {
        "input": text,
        "model": "text-embedding-ada-002"
    }
    response = requests.post(url, headers=headers, data=json.dumps(data))
    if response.status_code == 200:
        embedding = response.json()["data"][0]["embedding"]
        return np.array(embedding)
    else:
        print(f"Error: {response.status_code}")
        print(response.json())
        return None


app = Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    message = request.json['message']
    query_embedding = get_embedding(message)
    result = find_closest_match(query_embedding)
    if result is None:
        response = "No relevant match found."
    else:
        response = generate_response(result['post_title'], result['post_url'], result['content_clean'])
    return jsonify(response=response)

if __name__ == '__main__':
    app.run(debug=True)
