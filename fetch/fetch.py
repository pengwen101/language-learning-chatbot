import os
import json
import time
import requests
from flask import Flask, jsonify, request
from dotenv import load_dotenv
import logging

# Load environment variables from .env
load_dotenv()

# Initialize Flask app and logging
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

directory = './cache'
if not os.path.exists(directory):
    os.makedirs(directory)
CACHE_FILE = './cache/cached_data.json'
CACHE_EXPIRATION = 1382400

def is_cache_valid():
    """Check if the cached file exists and is still valid."""
    if not os.path.exists(CACHE_FILE):
        return False
    return (time.time() - os.path.getmtime(CACHE_FILE)) < CACHE_EXPIRATION

def load_cache():
    """Load data from the cache file."""
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_cache(data, params_key):
    """Save data to the cache file with a unique params_key."""
    cache_data = load_cache()
    cache_data[params_key] = {
        'data': data,
        'timestamp': time.time()
    }
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache_data, f)

def get_cached_data(params_key):
    """Retrieve cached data based on the parameters key."""
    if not is_cache_valid():
        return None
    
    cache_data = load_cache()
    return cache_data.get(params_key, {}).get('data')

# Define the API endpoint
@app.route('/api/fetch', methods=['POST'])
def fetch_from_api():
    # Extract query parameters from the JSON request body
    params = request.get_json() or {}
    query_params = {
        'q': params.get('q', '')
    }
    
    # Create a unique key based on query parameters for caching
    params_key = json.dumps(query_params, sort_keys=True)

    # Check the cache
    cached_data = get_cached_data(params_key)
    if cached_data:
        logging.info("Using cached data")
        return jsonify(cached_data)

    # Set up headers for the external API request
    # print(os.getenv("APIJOB_API_KEY"))
    headers = {
        'apikey': os.getenv("APIJOB_API_KEY"),
        'Content-Type': 'application/json',
    }

    # Make the request to the external API
    try:
        response = requests.post('https://api.apijobs.dev/v1/job/search', headers=headers, json=query_params)
        response.raise_for_status()

        # Parse and cache the response data
        data = response.json()
        save_cache(data, params_key)
        return jsonify(data)
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching data from external API: {e}")
        return jsonify({"error": "Failed to fetch data from external API", "details": str(e)}), 500

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
