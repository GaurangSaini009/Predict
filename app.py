import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="🚗",
    layout="wide"
)
model = joblib.load("best_model.pkl")
encoder = joblib.load("encoder.pkl")

df = pd.read_csv("Cardetails.csv")
st.markdown("""
<style>

.main{
background-color:#f5f5f5;
}

h1{
color:#0E76A8;
text-align:center;
}

.stButton>button{
width:100%;
background:#0E76A8;
color:white;
font-size:18px;
border-radius:10px;
height:50px;
}

.stButton>button:hover{
background:#064663;
}

.pred{
background:#ffffff;
padding:20px;
border-radius:15px;
box-shadow:0px 0px 10px gray;
text-align:center;
}

</style>
""",unsafe_allow_html=True)
st.title("🚗 Car Price Prediction System")

st.write("Predict the resale value of your car using Machine Learning.")
st.sidebar.header("Car Details")
name = st.sidebar.selectbox(
"Car Name",
sorted(df["name"].unique())
)

year = st.sidebar.number_input(
"Year",
2000,
2025,
2020
)

km_driven = st.sidebar.number_input(
"KM Driven",
0,
500000,
50000
)

fuel = st.sidebar.selectbox(
"Fuel",
df["fuel"].unique()
)

seller_type = st.sidebar.selectbox(
"Seller Type",
df["seller_type"].unique()
)

transmission = st.sidebar.selectbox(
"Transmission",
df["transmission"].unique()
)

owner = st.sidebar.selectbox(
"Owner",
df["owner"].unique()
)

mileage = st.sidebar.number_input(
"Mileage",
0.0,
50.0,
20.0
)

engine = st.sidebar.number_input(
"Engine",
500,
5000,
1200
)

max_power = st.sidebar.number_input(
"Max Power",
20.0,
500.0,
80.0
)

seats = st.sidebar.number_input(
"Seats",
2,
10,
5
)
name = encoder["name"].transform([name])[0]

fuel = encoder["fuel"].transform([fuel])[0]

seller_type = encoder["seller_type"].transform([seller_type])[0]

transmission = encoder["transmission"].transform([transmission])[0]

owner = encoder["owner"].transform([owner])[0]
input_data = pd.DataFrame({

"name":[name],

"year":[year],

"km_driven":[km_driven],

"fuel":[fuel],

"seller_type":[seller_type],

"transmission":[transmission],

"owner":[owner],

"mileage":[mileage],

"engine":[engine],

"max_power":[max_power],

"seats":[seats]

})
if st.button("Predict Price"):

    prediction = model.predict(input_data)

    st.markdown(
    f"""
    <div class='pred'>

    <h2>Estimated Selling Price</h2>

    <h1 style="color:green;">
    ₹ {prediction[0]:,.0f}
    </h1>

    </div>
    """,
    unsafe_allow_html=True
    )
    