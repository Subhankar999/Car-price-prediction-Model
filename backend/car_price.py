from fastapi import FastAPI,HTTPException, UploadFile,File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import joblib
import io
import pandas as pd
import torch
import torch.nn as nn

app=FastAPI()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class my_nn(nn.Module): 
    def __init__(self,input_features):
        super().__init__()
        self.model=nn.Sequential(
            nn.Linear(input_features,128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            nn.Linear(128,64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            nn.Linear(64,32),
            nn.ReLU(),
            
            nn.Linear(32,1)
            
        )
    def forward(self,x):
        return self.model(x)

features = joblib.load("car_price_features.joblib")

print("Features:", features)
print("Number of features:", len(features))

# =========================================================
# CREATE MODEL
# =========================================================

model = my_nn(314)
# =========================================================
# LOAD TRAINED WEIGHTS
# =========================================================

state_dict = torch.load(
    "car_price.pth",
    map_location=device,
    weights_only=False
)

model.load_state_dict(state_dict)

model.to(device)

model.eval()




# [['seller', 'abtest', 'vehicleType', 'yearOfRegistration', 'gearbox', 'powerPS', 'model', 'kilometer', 'fuelType', 'brand', 'notRepairedDamage']
class car_price(BaseModel):
    seller :str= Field(min_length=1,max_length=50, description="Type of seller")
    ABtest : str =Field(min_length=1,max_length=50, description="Whether the listing was part of an A/B test")
    vehicle_type :str =Field(min_length=1,max_length=50, description="Type/body style of the vehicle")
    year_of_registration :float =Field(gt=0, description="Year in which the vehicle was registered")
    gearbox :str =Field(min_length=1,max_length=50,description="Type of transmission")
    powerPS :float =Field(gt=0, description="Engine power measured in PS (metric horsepower)")
    model :str =Field(min_length=1,max_length=50, description="Specific car model")
    kilometer :float =Field(gt=0, description="Approximate kilometers driven")
    fuelType : str= Field(min_length=1,max_length=50, description="Type of fuel used")
    brand : str= Field(min_length=1,max_length=50, description="Manufacturer/brand of the car")
    notRepairedDamage :str =Field(min_length=1,max_length=50, description="Whether the car has unrepaired damage")
    
@app.get("/")
def home():
    return{
        "Project":"Used Car Price Prediction",
        "Message":"Running",
        "Endpoint":"Send post request to predict"
    }

@app.post("/predict")
def predict(car:car_price):
    try:
        input_data=pd.DataFrame([{
           'seller': car.seller,
            "ABtest":car.ABtest,
            "vehicle_type":car.vehicle_type,
            "year_of_registration": car.year_of_registration,
            "gearbox":car.gearbox,
            "powerPS":car.powerPS,
            "model":car.model,
            "kilometer": car.kilometer,
            "fueltype": car.fuelType,
            "brand": car.brand,
            "Not Repaired damage": car.notRepairedDamage
            
        }])
        input_data=torch.tensor(input_data)
        
        prediction=model(input_data)[0]
        
        return{
            "predicted price":f"${prediction}"
        }
    except Exception as e:
        raise HTTPException(
               status_code=500,
               detail=f"Prediction Failed: {str(e)}" 
            )

