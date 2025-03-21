


import streamlit as st
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load trained model
model_path = r"C:\Users\LENOVO\Desktop\dataset for Reg.zip.pkl"  # Update this if needed
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# UI Title
st.title("🚘 Car Price Prediction Application")

with st.sidebar.expander('Who is the developer of this App ?'):#'Who is the developer of the App ?'):
    st.write('**This Application is developed and deployed by Cheikh Ahmed,MedS, Anda (a Data Scientist)**')
  

st.header("Enter car details below to predict the price of a car.")

# Define categorical options (ensure alignment with training data)
brands = ['Audi', 'BMW', 'Chevrolet', 'Ford', 'Honda', 'Hyundai', 'Kia', 'Mercedes', 'Toyota', 'Volkswagen']
models = ['3 Series', '5 Series', 'A3', 'A4', 'Accord', 'C-Class', 'CR-V', 'Camry', 'Civic', 'Corolla', 
          'E-Class', 'Elantra', 'Equinox', 'Explorer', 'Fiesta', 'Focus', 'GLA', 'Golf', 'Impala', 'Malibu', 
          'Optima', 'Passat', 'Q5', 'RAV4', 'Rio', 'Sonata', 'Sportage', 'Tiguan', 'Tucson', 'X5']
fuel_types = ['Diesel', 'Electric', 'Hybrid', 'Petrol']
transmissions = ['Automatic', 'Manual', 'Semi-Automatic']

# User Inputs
selected_brand = st.selectbox("Select Brand", brands)
selected_model = st.selectbox("Select Model", models)  # ✅ FIXED: 'Model' selection added
selected_fuel = st.selectbox("Select Fuel Type", fuel_types)
selected_transmission = st.selectbox("Select Transmission", transmissions)

year = st.slider("Year", 1990, 2025, 2020)
engine_size = st.number_input("Engine Size (L)", min_value=0.5, max_value=6.0, step=0.1, value=2.0)
mileage = st.number_input("Mileage", min_value=0, max_value=300000, step=1000, value=50000)
doors = st.slider("Number of Doors", 2, 5, 4)
owner_count = st.slider("Number of Previous Owners", 1, 5, 1)

# One-hot encode categorical features
brand_encoded = [1 if b == selected_brand else 0 for b in brands]
model_encoded = [1 if m == selected_model else 0 for m in models]  # ✅ FIXED: Encoding for 'Model' column
fuel_encoded = [1 if f == selected_fuel else 0 for f in fuel_types]
transmission_encoded = [1 if t == selected_transmission else 0 for t in transmissions]

# Create feature vector (ensure all 52 features are represented)
features = np.zeros(52)  # Initialize with zeros

# Fill in categorical features
features[:len(brand_encoded)] = brand_encoded
features[len(brands):len(brands) + len(model_encoded)] = model_encoded  # ✅ FIXED: Models added
features[len(brands) + len(models):len(brands) + len(models) + len(fuel_encoded)] = fuel_encoded
features[len(brands) + len(models) + len(fuel_encoded):len(brands) + len(models) + len(fuel_encoded) + len(transmission_encoded)] = transmission_encoded

# Append numerical features at the correct positions
features[-5:] = [year, engine_size, mileage, doors, owner_count]

# Reshape for model input
features = features.reshape(1, -1)

# Predict button
if st.button("🔮 Predict Price"):
    prediction = model.predict(features)[0]
    st.success(f"💰 Predicted Car Price: ${prediction:,.2f}")

    # Visualization - Show relationship of features with price
with st.sidebar.expander('Visualization Shows relationship of features with price'):
    st.write('Input Feature Comparision')
       

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=['Year', 'Engine Size', 'Mileage', 'Doors', 'Owner Count'], 
                y=[year, engine_size, mileage, doors, owner_count], ax=ax, palette="Blues")
    ax.set_ylabel("Feature Values")
    ax.set_title("📊 Input Feature Comparison")
    st.pyplot(fig)
    
