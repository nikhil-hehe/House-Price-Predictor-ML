import gradio as gr
import pandas as pd
import joblib

# Load the saved pipeline
# Ensure this file is in the same folder as your gui.py
try:
    model = joblib.load("house_model.pkl")
except FileNotFoundError:
    print("Error: 'house_model.pkl' not found. Please run the save cell in your notebook first.")

def predict_house_value(longitude, latitude, age, rooms, bedrooms, population, households, income, ocean):
    input_data = pd.DataFrame([{
        "longitude": longitude,
        "latitude": latitude,
        "housing_median_age": age,
        "total_rooms": rooms,
        "total_bedrooms": bedrooms,
        "population": population,
        "households": households,
        "median_income": income,
        "ocean_proximity": ocean
    }])
    
    # Make prediction (the pipeline handles scaling/encoding automatically)
    prediction = model.predict(input_data)[0]
    
    return f"${prediction:,.2f}"

inputs = [
    gr.Number(label="Longitude", value=-122.23),
    gr.Number(label="Latitude", value=37.88),
    gr.Slider(1, 52, label="Median House Age", value=25),
    gr.Number(label="Total Rooms", value=2000),
    gr.Number(label="Total Bedrooms", value=400),
    gr.Number(label="Population", value=1000),
    gr.Number(label="Households", value=400),
    gr.Number(label="Median Income (in $10k)", value=3.5),
    gr.Dropdown(
        choices=["<1H OCEAN", "INLAND", "NEAR OCEAN", "NEAR BAY", "ISLAND"], 
        label="Ocean Proximity", 
        value="NEAR BAY"
    )
]

app = gr.Interface(
    fn=predict_house_value, 
    inputs=inputs, 
    outputs=gr.Textbox(label="Predicted Median House Value"),
    title="California House Price Predictor",
    description="Enter neighborhood stats to get an estimated house value based on the trained model."
)

if __name__ == "__main__":
    app.launch()