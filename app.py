from flask import Flask, render_template, request
import pickle
import pandas as pd  # Import pandas

app = Flask(__name__)

# Load Model and Scaler
model = pickle.load(open("best_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        # Get input values
        year = int(request.form["year"])
        present_price = float(request.form["present_price"])
        kms_driven = float(request.form["kms_driven"])
        owner = int(request.form["owner"])
        
        fuel_type = request.form["fuel_type"]
        seller_type = request.form["seller_type"]
        transmission = request.form["transmission"]

        # Manual Encoding
        fuel_diesel = 1 if fuel_type == "Diesel" else 0
        fuel_petrol = 1 if fuel_type == "Petrol" else 0
        seller_individual = 1 if seller_type == "Individual" else 0
        trans_manual = 1 if transmission == "Manual" else 0

        # Create a Dictionary with EXACT column names from your notebook
        data_dict = {
            'Year': [year],
            'Present_Price': [present_price],
            'Kms_Driven': [kms_driven],
            'Owner': [owner],
            'Fuel_Type_Diesel': [fuel_diesel],
            'Fuel_Type_Petrol': [fuel_petrol],
            'Seller_Type_Individual': [seller_individual],
            'Transmission_Manual': [trans_manual]
        }

        # Convert to DataFrame 
        df = pd.DataFrame(data_dict)
        
        # Scale the data using the DataFrame
        final_features = scaler.transform(df)

        # Predict
        prediction = model.predict(final_features)
        
        output = round(prediction[0], 2)

        if output < 0:
            return render_template('index.html', prediction_text="Sorry, you cannot sell this car")
        else:
            return render_template('index.html', prediction_text=f"Predicted Selling Price: ₹ {output} Lakhs")

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)