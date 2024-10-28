# Job Search API Caching Application
This is a Flask application that allows you to search for job listings using the API from apijobs.dev with caching functionality to enhance performance and reduce redundant API calls.

## Features
Fetch job listings based on a query parameter (q).
Cache responses to minimize API calls and improve performance.
Configurable caching expiration time.
## Requirements
Python 3.x
Flask
Requests
Python-dotenv
## Setup
1. Install the required packages:
```bash
# go to main folder and install if haven't
pip install -r requirements.txt
```
2. Copy .env.example to .env file in the root directory and add your API key:
```bash
APIJOB_API_KEY=your_api_key_here
DEBUG=False
```
## Usage
1. Run the Flask application:
```bash
python app.py
```
By default, the application runs on http://127.0.0.1:5000/.
2. Send a POST request to fetch job listings:

Use a tool like Postman or curl to send a request to the API endpoint.

Example request using curl:
```bash
curl -X POST http://127.0.0.1:5000/api/fetch \
-H "Content-Type: application/json" \
-d '{
    "q": "software engineer"
}'
```
Or u can use python
```python
url = 'http://localhost:5000/api/fetch'
payload = {
    "q": "software engineer",
}
# Send the POST request with the payload as JSON
response = requests.post(url, json=payload)

# Print the response
if response.status_code == 200:
    print("Response data:", response.json())
else:
    print("Error:", response.status_code, response.json())
```

3. Response:
The API will return job listings matching the query. If the same query has been made recently, it will return cached data.

## Cache Management
The cache is stored in the ./cache/cached_data.json file.
The cache expiration is set to 16 days (1382400 seconds) by default. You can modify CACHE_EXPIRATION in the code to change this duration.



