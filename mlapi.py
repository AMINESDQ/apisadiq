from fastapi import FastAPI
import pickle
import pandas as pd
from pydantic import BaseModel

app = FastAPI()

class Item(BaseModel):
    BALANCE: float
    BALANCE_FREQUENCY: float
    PURCHASES: float
    ONEOFF_PURCHASES: float
    INSTALLMENTS_PURCHASES: float
    CASH_ADVANCE: float
    PURCHASES_FREQUENCY: float
    ONEOFF_PURCHASES_FREQUENCY: float
    PURCHASES_INSTALLMENTS_FREQUENCY: float
    CASH_ADVANCE_FREQUENCY: float
    CASH_ADVANCE_TRX: float
    PURCHASES_TRX: float
    CREDIT_LIMIT: float
    PAYMENTS: float
    MINIMUM_PAYMENTS: float
    PRC_FULL_PAYMENT: float
    TENURE: float

@app.get("/")
def read_root():
    return {"Hello": "World"}

pickle_in = open("Kmeans_model.pkl", "rb")
model = pickle.load(pickle_in)

@app.post("/predict/")
async def pppp(item: Item):
    item_dict = item.dict()
    prediction = model.predict(pd.DataFrame([item_dict]))
    return {"prediction": prediction.tolist()}
