from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np
import yaml

# --------------------------------------------------
# Load config
# --------------------------------------------------
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

MODEL_PATH = config["model_name"]
SCALER_PATH = config["scaler_name"]
ENCODER_PATH = config["encoder_name"]
DEBUG = config["app"]["debug"]

# --------------------------------------------------
# Flask app
# --------------------------------------------------
app = Flask(__name__)

# --------------------------------------------------
# Load artifacts
# --------------------------------------------------
model = pickle.load(open(MODEL_PATH, "rb"))
scaler = pickle.load(open(SCALER_PATH, "rb"))
name_encoder = pickle.load(open(ENCODER_PATH, "rb"))

# --------------------------------------------------
# Routes
# --------------------------------------------------
@app.route("/")
def home():
    car_names = list(name_encoder.classes_)
    return render_template("index.html", car_names=car_names)

@app.route("/predict", methods=["POST"])
def predict():
    car_names = list(name_encoder.classes_)

    try:
        car_name_input = request.form["car_name"]

        if car_name_input not in name_encoder.classes_:
            return render_template(
                "index.html",
                prediction_text="Error: Car name not found.",
                car_names=car_names
            )

        # Input values
        year = int(request.form["year"])
        present_price = float(request.form["present_price"])
        kms_driven = float(request.form["kms_driven"])
        owner = int(request.form["owner"])

        fuel_type = request.form["fuel_type"]
        seller_type = request.form["seller_type"]
        transmission = request.form["transmission"]

        # Encoding
        car_name_encoded = name_encoder.transform([car_name_input])[0]
        fuel_diesel = 1 if fuel_type == "Diesel" else 0
        fuel_petrol = 1 if fuel_type == "Petrol" else 0
        seller_individual = 1 if seller_type == "Individual" else 0
        trans_manual = 1 if transmission == "Manual" else 0

        # DataFrame
        df = pd.DataFrame({
            "Car_Name": [car_name_encoded],
            "Year": [year],
            "Present_Price": [present_price],
            "Kms_Driven": [kms_driven],
            "Owner": [owner],
            "Fuel_Type_Diesel": [fuel_diesel],
            "Fuel_Type_Petrol": [fuel_petrol],
            "Seller_Type_Individual": [seller_individual],
            "Transmission_Manual": [trans_manual]
        })

        # Predict
        final_features = scaler.transform(df)
        prediction = model.predict(final_features)[0]
        output = round(prediction, 2)

        result = (
            "Valuation is negative."
            if output < 0
            else f"Estimated Value: ₹ {output} Lakhs"
        )

        return render_template(
            "index.html",
            prediction_text=result,
            car_names=car_names
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction_text=f"Error: {str(e)}",
            car_names=car_names
        )

# --------------------------------------------------
# Run app
# --------------------------------------------------
if __name__ == "__main__":
    app.run(debug=DEBUG)
