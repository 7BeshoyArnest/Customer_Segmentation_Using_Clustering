import requests
import json

# API endpoint
url = "http://127.0.0.1:5000/predict"

# Example payload (features in correct order)
data = {
    "Annual Income (k$)": 60,
    "Spending Score (1-100)": 50
}

# Send POST request
response = requests.post(url, json=data)

# Print response
print("Status Code:", response.status_code)
print("Response JSON:", json.dumps(response.json(), indent=2))
