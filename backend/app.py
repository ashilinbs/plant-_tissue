from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd


app = Flask(__name__)
CORS(app)

# Load the trained model and the fitted preprocessor
model = joblib.load('plant_tissue_culture_model.joblib')
preprocessor = joblib.load('fitted_preprocessor.joblib')

# Define the mapping from predicted class to percentage range
class_to_percentage = {
    0: "0% - 25%",
    1: "25% - 50%",
    2: "50% - 75%",
    3: "75% - 100%"
}

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # Get data from the request

    # Create a DataFrame from the input data
    input_df = pd.DataFrame([data])

    # Apply preprocessing to transform the data
    processed_input = preprocessor.transform(input_df)

    # Make the prediction
    predicted_class = model.predict(processed_input)[0]

    # Map the predicted class to its corresponding percentage range
    predicted_percentage_range = class_to_percentage.get(predicted_class, "Unknown")

    # Return the prediction as JSON
    return jsonify({
        'predicted_class': int(predicted_class),
        'predicted_percentage_range': predicted_percentage_range
    })



if __name__ == '__main__':
    app.run(debug=True)
