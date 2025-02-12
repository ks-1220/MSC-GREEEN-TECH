import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Set Streamlit page config
st.set_page_config(
    page_title="Effluent Prediction Dashboard",
    page_icon="🌊",
    layout="wide"
)

# ----------------------------------------------------------------------------
# Load and preprocess data

@st.cache_data
def load_data():
    """Load effluent data from a CSV file."""
    DATA_FILENAME = Path(__file__).parent/'FINAL_SDI CSV FILE.csv'
    df = pd.read_csv(DATA_FILENAME)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df = df.sort_index()
    return df

df = load_data()

# Encode categorical columns
categorical_cols = ['Industry', 'Chemical_Type', 'Weather_Condition', 'Production_Scale']
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Feature selection and scaling
features = ['Effluent_Volume_Liters', 'Chemical_Concentration_ppm', 'pH', 'TDS_ppm', 'Conductivity_uS', 'Production_Scale']
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df[features])
df_scaled = pd.DataFrame(df_scaled, columns=features)

# Train-Test Split
train_size = int(len(df) * 0.8)
train, test = df_scaled[:train_size], df_scaled[train_size:]

# Prepare LSTM input data
X_train, y_train = train[:-1], train[1:]
X_test, y_test = test[:-1], test[1:]
X_train = np.array(X_train).reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = np.array(X_test).reshape((X_test.shape[0], X_test.shape[1], 1))

# Build LSTM Model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    LSTM(50),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=10, batch_size=16)

# Future Prediction Function
def predict_future(years=5):
    future_dates = pd.date_range(df.index[-1], periods=years * 12, freq='M')
    future_preds = model.predict(X_test[-len(future_dates):])
    future_preds = future_preds.reshape(-1, 6)
    future_preds_inversed = scaler.inverse_transform(future_preds)
    return pd.DataFrame({'Date': future_dates, 'Predicted_Concentration': future_preds.flatten()})

future_trends = predict_future(6)

# ----------------------------------------------------------------------------
# Streamlit UI Elements
st.title("🌊 Effluent Prediction Dashboard")

# Sidebar Inputs
st.sidebar.header("User Inputs")
industry = st.sidebar.selectbox("Select Industry", df['Industry'].unique())
year = st.sidebar.slider("Select Year", 2025, 2030, 2027)

# Fetch prediction for selected industry and year
filtered_trends = future_trends.copy()
filtered_trends = filtered_trends[filtered_trends['Date'].dt.year == year]

# ----------------------------------------------------------------------------
# Data Visualization
st.subheader("📊 Effluent Concentration Trends")
chart = (
    alt.Chart(filtered_trends)
    .mark_line()
    .encode(
        x=alt.X("Date:T", title="Date"),
        y=alt.Y("Predicted_Concentration:Q", title="Predicted Chemical Concentration (ppm)"),
    )
    .properties(height=320)
)
st.altair_chart(chart, use_container_width=True)

# Correlation Heatmap
st.subheader("🔥 Feature Correlation Heatmap")
fig, ax = plt.subplots()
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
st.pyplot(fig)
