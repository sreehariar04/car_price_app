from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# 1. Load Model, Scaler, and the Name Encoder

model = pickle.load(open("best_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
name_encoder = pickle.load(open("name_encoder.pkl", "rb"))

@app.route("/")
def home():
    # Get the list of car names to populate the autocomplete list
    car_names = list(name_encoder.classes_)
    return render_template("index.html", car_names=car_names)

@app.route("/predict", methods=["POST"])
def predict():
    # Always have the list ready for re-rendering the page
    car_names = list(name_encoder.classes_)

    if request.method == "POST":
       #user input
        car_name_input = request.form["car_name"]
        
        
        # If the user typed a name that isn't in our training data, stop here.
        if car_name_input not in name_encoder.classes_:
            return render_template('index.html', 
                                   prediction_text="Error: Car name not found. Please pick from the list.", 
                                   car_names=car_names)

        year = int(request.form["year"])
        present_price = float(request.form["present_price"])
        kms_driven = float(request.form["kms_driven"])
        owner = int(request.form["owner"])
        
        fuel_type = request.form["fuel_type"]
        seller_type = request.form["seller_type"]
        transmission = request.form["transmission"]

        
        # Encode the Car Name using the encoder
        car_name_encoded = name_encoder.transform([car_name_input])[0]

        # Manual One-Hot Encoding 
        fuel_diesel = 1 if fuel_type == "Diesel" else 0
        fuel_petrol = 1 if fuel_type == "Petrol" else 0
        seller_individual = 1 if seller_type == "Individual" else 0
        trans_manual = 1 if transmission == "Manual" else 0

        #DataFrame with EXACT column order from training
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
        
        # Scale the data 
        final_features = scaler.transform(df)

        #Prediction 
        prediction = model.predict(final_features)
        output = round(prediction[0], 2)

        if output < 0:
            return render_template('index.html', 
                                   prediction_text="Sorry, you cannot sell this car (Negative Value)", 
                                   car_names=car_names)
        else:
            return render_template('index.html', 
                                   prediction_text=f"Estimated Value: ₹ {output} Lakhs", 
                                   car_names=car_names)

    return render_template("index.html", car_names=car_names)

if __name__ == "__main__":
    app.run(debug=True)