# main.py
from fastapi import FastAPI, Request
from pydantic import BaseModel
import joblib

app = FastAPI()

# Load your model once at startup
model = joblib.load('hybrid_model_1.pkl')

# Define the input schema
class TextInput(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "Hate Speech Detection API"}

@app.post("/predict/")
def predict(input: TextInput):
    prediction = model.predict([input.text])
    label = "Hate Speech" if prediction[0] == 0 else "Non-Hate Speech"
    return {
        "input": input.text,
        "prediction": label
    }
