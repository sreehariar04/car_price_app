from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load artifacts
model = pickle.load(open("best_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
name_encoder = pickle.load(open("name_encoder.pkl", "rb"))

@app.route("/")
def home():
    car_names = list(name_encoder.classes_)
    return render_template("index.html", car_names=car_names)

@app.route("/predict", methods=["POST"])
def predict():
    # Helper to ensure dropdown persists on reload
    car_names = list(name_encoder.classes_)

    if request.method == "POST":
        car_name_input = request.form["car_name"]
        
        # Validation
        if car_name_input not in name_encoder.classes_:
            return render_template('index.html', 
                                   prediction_text="Error: Car name not found.", 
                                   car_names=car_names)

        # Retrieve form data
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

        # Construct DataFrame
        data_dict = {
            'Car_Name': [car_name_encoded],
            'Year': [year],
            'Present_Price': [present_price],
            'Kms_Driven': [kms_driven],
            'Owner': [owner],
            'Fuel_Type_Diesel': [fuel_diesel],
            'Fuel_Type_Petrol': [fuel_petrol],
            'Seller_Type_Individual': [seller_individual],
            'Transmission_Manual': [trans_manual]
        }
        df = pd.DataFrame(data_dict)
        
        # Preprocessing & Inference
        final_features = scaler.transform(df)
        prediction = model.predict(final_features)
        output = round(prediction[0], 2)

        if output < 0:
            return render_template('index.html', 
                                   prediction_text="Valuation is negative.", 
                                   car_names=car_names)
        else:
            return render_template('index.html', 
                                   prediction_text=f"Estimated Value: ₹ {output} Lakhs", 
                                   car_names=car_names)

    return render_template("index.html", car_names=car_names)

if __name__ == "__main__":
    app.run(debug=True)