from flask import Flask, request, jsonify, Response
import json
import joblib
import numpy as np
from flasgger import Swagger
from collections import OrderedDict

# Initialize Flask app
app = Flask(__name__)
swagger = Swagger(app)

# Load the trained model
model = joblib.load('Customer Segmentation')


@app.route('/')
def home():
    return " Customer Segmentation API is running! Visit /apidocs for Swagger UI."


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict A Customer Segment
    ---
    tags:
      - Customer Segmentation
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          required:
            -Annual Income (k$)
            -Spending Score (1-100)
          properties:
            Annual Income (k$):
              type: number
              example: 60
            Spending Score (1-100):
              type: number
              example: 50
    responses:
      200:
        description: Prediction result
        schema:
          type: object
          properties:
            predicted Cluster:
              type: integer
              example: 0
    """

    data = request.get_json()

    # List of required fields
    required_fields = [
        'Annual Income (k$)',
        'Spending Score (1-100)'
    ]

    # Check for missing fields
    for field in required_fields:
        if field not in data:
            return jsonify({"error": f"Missing field: {field}"}), 400

        # Check for empty or null values
        if data[field] is None or (isinstance(data[field], str) and not data[field].strip()):
            return jsonify({"error": f"Field '{field}' cannot be empty"}), 400

    # Type validation
    numeric_fields = [
        'Annual Income (k$)',
        'Spending Score (1-100)'
    ]

    for field in numeric_fields:
        if not isinstance(data[field], (int, float)):
            return jsonify({"error": f"Field '{field}' must be a number"}), 400


    # Logical range validation

    # Arrange features exactly as the model was trained
    features = np.array([[
        data['Annual Income (k$)'],
        data['Spending Score (1-100)']
    ]])

    # Make prediction safely
    try:
        prediction = model.predict(features)
    except Exception as e:
        return jsonify({"error": f"Model prediction failed: {str(e)}"}), 500
    

    if int(prediction[0]) == 0:
        data = OrderedDict([
            ('predicted Cluster', int(prediction[0])+1),
            ('Description', 'the customer belongs to Cluster 1: Average Income, Average Spending')
        ])
    elif int(prediction[0]) == 1:
        data = OrderedDict([
            ('predicted Cluster', int(prediction[0])+1),
            ('Description', 'the customer belongs to Cluster 2: High Income, High Spending')
        ])
    elif int(prediction[0]) == 2:
        data = OrderedDict([
            ('predicted Cluster', int(prediction[0])+1),
            ('Description', 'the customer belongs to Cluster 3: Low Income, High Spending')
        ])
    elif int(prediction[0]) == 3:
        data = OrderedDict([
            ('predicted Cluster', int(prediction[0])+1),
            ('Description', 'the customer belongs to Cluster 4: High Income, Low Spending')
        ])
    elif int(prediction[0]) == 4:
        data = OrderedDict([
            ('predicted Cluster', int(prediction[0])+1),
            ('Description', 'the customer belongs to Cluster 5: Low Income, Low Spending')
        ])

    return Response(json.dumps(data), mimetype='application/json')


if __name__ == "__main__":
    app.run(port=5000, debug=True)
