"""SDI

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1Io9gXJLDl6NvqTwphs8T19KNJ96jrXFo
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
#from xgboost import XGBRegressor
#from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import streamlit as st

# Step 1: Load Data
df = pd.read_csv("FINAL_SDI CSV FILE.csv")

# Step 2: Preprocessing
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df = df.sort_index()

#df.dtypes

# Encode categorical columns
categorical_cols = ['Industry', 'Chemical_Type', 'Weather_Condition', 'Production_Scale']
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Store encoders for future use

# Selecting relevant features
features = ['Effluent_Volume_Liters', 'Chemical_Concentration_ppm', 'pH', 'TDS_ppm', 'Conductivity_uS', 'Production_Scale']
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df[features])

# Convert back to DataFrame
df_scaled = pd.DataFrame(df_scaled, columns=features)

print("Preprocessing complete. No categorical conversion errors.")

#Step 3: Train-Test Split
train_size = int(len(df) * 0.8)
train, test = df_scaled[:train_size], df_scaled[train_size:]

# Step 4: LSTM Model
X_train, y_train = train[:-1], train[1:]
X_test, y_test = test[:-1], test[1:]
#X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
#X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

X_train = np.array(X_train)  # Convert to NumPy array
y_train = np.array(y_train)  # Convert to NumPy array
X_test = np.array(X_test)    # Convert to NumPy array
y_test = np.array(y_test)    # Convert to NumPy array

# Reshape the NumPy arrays
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))



# Now reshape
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))



model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    LSTM(50),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=10, batch_size=16)

# Step 5: Future Predictions
def predict_future(years=5):
    future_dates = pd.date_range(df.index[-1], periods=years * 12, freq='M')
    future_preds = model.predict(X_test[-len(future_dates):])
    #future_preds = scaler.inverse_transform(future_preds.reshape(-1, 1))

    # Assuming you had 6 features, you should pass an array with 6 features for inverse_transform
    future_preds = future_preds.reshape(-1, 6)  # Reshape to have 6 features, matching the original training data

# Now apply inverse_transform
    future_preds_inversed = scaler.inverse_transform(future_preds)


    return pd.DataFrame({'Date': future_dates, 'Predicted_Concentration': future_preds.flatten()})

future_trends = predict_future(6)

# Step 6: Streamlit Deployment
def generate_report(industry, year, concentration, reg_limit, effluent_volume, tds, conductivity):
    output = []
    if concentration > reg_limit:
        output.append(f"The chemical concentration for {industry} in {year} is predicted to be {concentration:.2f} ppm, which is above the regulatory limit of {reg_limit} ppm. Immediate action is recommended.")
    else:
        output.append(f"The chemical concentration for {industry} in {year} is predicted to be {concentration:.2f} ppm, which is within the safe limit.")

    if effluent_volume > 50000:
        output.append(f"The effluent volume for {industry} in {year} is expected to be {effluent_volume:.2f} liters, indicating a significant increase in production, potentially leading to higher environmental impact.")

    if tds > 1000:
        output.append(f"The Total Dissolved Solids (TDS) level is projected to be {tds:.2f} ppm, suggesting a potential issue with water quality that may require further treatment.")

    if conductivity > 1500:
        output.append(f"The conductivity level is expected to be {conductivity:.2f} µS, indicating high ion concentration, which could impact aquatic life.")

    return "\n".join(output)

st.title("Effluent Prediction Dashboard")
industry = st.text_input("Enter Industry Name")
year = st.selectbox("Select Year", [2025, 2026, 2027, 2028, 2029, 2030])
predicted_value = future_trends[future_trends['Date'].dt.year == year]['Predicted_Concentration'].values[0]
effluent_volume = np.random.uniform(30000, 60000)  # Simulated value for demonstration
regulatory_limit = 100  # Example threshold
tds = np.random.uniform(500, 2000)  # Simulated value for demonstration
conductivity = np.random.uniform(800, 2000)  # Simulated value for demonstration
st.write(generate_report(industry, year, predicted_value, regulatory_limit, effluent_volume, tds, conductivity))