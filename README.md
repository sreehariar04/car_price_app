## Car Price Prediction 🚗💰
## Car Price Prediction Project

### Overview

This project predicts the **selling price of used cars** based on various features such as car age, mileage, fuel type, transmission, and ownership details.
It uses **Machine Learning** to learn patterns from historical car data and estimate a fair market price.

## Dataset

The dataset contains information such as:

* Year of manufacture
* Present price
* Kilometers driven
* Fuel type
* Seller type
* Transmission
* Number of previous owners

Target variable:

* **Selling Price**

## Steps

1. Data cleaning and preprocessing
2. Handling missing values
3. Encoding categorical features
4. Feature scaling
5. Model training using **Random Forest**
6. Model evaluation
7. Prediction on new/unseen data

## Model Used

* **Random Forest Regressor / Classifier**
* Handles non-linear relationships well
* Robust to outliers
* Performs well on tabular data

## Evaluation Metrics

Depending on setup:

* R² Score
* Mean Absolute Error (MAE)
* Mean Squared Error (MSE)

## Project Structure

```bash
Car_Price/
│
├── static/
│   └── style.css
│
├── templates/
│   └── index.html
│
├── venv/                  # Virtual environment
│
├── app.py                 # Flask application
├── best_model.pkl         # Trained ML model
├── scaler.pkl             # StandardScaler object
├── name_encoder.pkl       # LabelEncoder for car names
├── README.md
└── pyvenv.cfg
```


## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/sreehariar04/car_price_app.git
cd car_price_app
```

### 2. Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
python app.py
```
## Results

The model provides reliable car price predictions and performs well on unseen data after tuning.

## Future Improvements

* Try XGBoost / LightGBM
* Add more feature engineering
* Improve UI
* Deploy the model online

##DEMO : 
<img src="" alt="Alt text" width="100"/>

## Author

**Sreehari AR**
