# main.py
from fastapi import FastAPI, Request
from pydantic import BaseModel
import joblib
from sklearn.base import BaseEstimator, ClassifierMixin
from scipy.stats import mode
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

class HybridModel(BaseEstimator, ClassifierMixin):
    def __init__(self, model1, model2, model3):
        self.model1 = model1
        self.model2 = model2
        self.model3 = model3

    def fit(self, X, y):
        self.model1.fit(X, y)
        self.model2.fit(X, y)
        self.model3.fit(X, y)
        return self

    def predict(self, X):
        pred1 = self.model1.predict(X)
        pred2 = self.model2.predict(X)
        pred3 = self.model3.predict(X)
        final_pred = mode([pred1, pred2, pred3], axis=0)[0].flatten()
        return final_pred

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
