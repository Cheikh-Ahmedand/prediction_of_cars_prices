
import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ✅ Load the pre-trained model from the server
MODEL_PATH = "model.pkl"
model = None  # Initialize the model variable

try:
    with open(MODEL_PATH, 'rb') as file:
        model = pickle.load(file)
    
    # Ensure model has a predict method
    if not hasattr(model, "predict"):
        raise ValueError("Loaded model is not valid! Ensure it supports .predict() method.")
    
    st.success("✅ Pre-trained Model Loaded Successfully!")
except FileNotFoundError:
    st.warning("⚠️ Model file not found! Please upload 'model.pkl' to the server.")
except Exception as e:
    st.error(f"❌ Failed to load the model: {e}")

# 🎨 UI Title
st.title("🚘 Car Price Prediction Application")

with st.sidebar.expander('ℹ️ Who is the developer of this App?'):
    st.write('**Developed & deployed by Cheikh Ahmed, MedS, Anda (Data Scientist)**')

st.header("📝 Enter car details below to predict the price.")

# 🔹 Define categorical options (ensure alignment with training data)
brands = ['Audi', 'BMW', 'Chevrolet', 'Ford', 'Honda', 'Hyundai', 'Kia', 'Mercedes', 'Toyota', 'Volkswagen']
models = ['3 Series', '5 Series', 'A3', 'A4', 'Accord', 'C-Class', 'CR-V', 'Camry', 'Civic', 'Corolla', 
          'E-Class', 'Elantra', 'Equinox', 'Explorer', 'Fiesta', 'Focus', 'GLA', 'Golf', 'Impala', 'Malibu', 
          'Optima', 'Passat', 'Q5', 'RAV4', 'Rio', 'Sonata', 'Sportage', 'Tiguan', 'Tucson', 'X5']
fuel_types = ['Diesel', 'Electric', 'Hybrid', 'Petrol']
transmissions = ['Automatic', 'Manual', 'Semi-Automatic']

# 📌 User Inputs
selected_brand = st.selectbox("🏷️ Select Brand", brands)
selected_model = st.selectbox("🚗 Select Model", models)
selected_fuel = st.selectbox("⛽ Select Fuel Type", fuel_types)
selected_transmission = st.selectbox("⚙️ Select Transmission", transmissions)

year = st.slider("📅 Year", 1990, 2025, 2020)
engine_size = st.number_input("🛠️ Engine Size (L)", min_value=0.5, max_value=6.0, step=0.1, value=2.0)
mileage = st.number_input("🛣️ Mileage (KM)", min_value=0, max_value=300000, step=1000, value=50000)
doors = st.slider("🚪 Number of Doors", 2, 5, 4)
owner_count = st.slider("👤 Number of Previous Owners", 1, 5, 1)

# 🔄 Function for One-Hot Encoding
def one_hot_encode(options, selected_option):
    return [1 if option == selected_option else 0 for option in options]

# Encode categorical features
brand_encoded = one_hot_encode(brands, selected_brand)
model_encoded = one_hot_encode(models, selected_model)
fuel_encoded = one_hot_encode(fuel_types, selected_fuel)
transmission_encoded = one_hot_encode(transmissions, selected_transmission)

# 🧩 Create feature vector dynamically
features = np.array(brand_encoded + model_encoded + fuel_encoded + transmission_encoded + 
                    [year, engine_size, mileage, doors, owner_count])

# Reshape for model input
features = features.reshape(1, -1)

# 🚀 Prediction Button
if st.button("🔮 Predict Price"):
    if model is not None:
        try:
            prediction = model.predict(features)[0]
            st.success(f"💰 **Estimated Car Price: ${prediction:,.2f}**")
        except Exception as e:
            st.error(f"❌ Prediction Error: {e}")
    else:
        st.error("⚠️ Model not loaded! Please upload 'model.pkl' to the server.")

# 📊 Feature Visualization
with st.sidebar.expander('📈 Feature Comparison with Price'):
    st.write("Here is a visualization of your input features:")
    
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=['Year', 'Engine Size', 'Mileage', 'Doors', 'Owner Count'], 
                y=[year, engine_size, mileage, doors, owner_count], ax=ax, palette="Blues")
    ax.set_ylabel("Feature Values")
    ax.set_title("📊 Input Feature Comparison")
    
    st.pyplot(fig)
