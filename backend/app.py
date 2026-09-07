from fastapi import FastAPI,HTTPException, UploadFile,File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import joblib
import io
import pandas as pd
import torch
import torch.nn as nn
import numpy as np

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class my_nn(nn.Module):
    def __init__(self, input_features):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.model(x)



features = joblib.load("car_price_features.joblib")             # raw column order used to build X
encoder = joblib.load("car_price_encoder.joblib")                # fitted OneHotEncoder
scaler = joblib.load("car_price_scaler.joblib")                  # fitted StandardScaler
categorical_cols = joblib.load("car_price_categorical_cols.joblib")
numeric_cols = joblib.load("car_price_numeric_cols.joblib")

n_input_features = len(encoder.get_feature_names_out(categorical_cols)) + len(numeric_cols)
print("Raw features:", features)
print("Model input features:", n_input_features)


model = my_nn(n_input_features)

state_dict = torch.load(
    "car_price.pth",
    map_location=device,
    weights_only=True,  # this is a plain state_dict, no need to disable safety checks
)
model.load_state_dict(state_dict)
model.to(device)
model.eval()



class car_price(BaseModel):
    seller: str = Field(min_length=1, max_length=50, description="Type of seller")
    ABtest: str = Field(min_length=1, max_length=50, description="Whether the listing was part of an A/B test")
    vehicle_type: str = Field(min_length=1, max_length=50, description="Type/body style of the vehicle")
    year_of_registration: float = Field(gt=0, description="Year in which the vehicle was registered")
    gearbox: str = Field(min_length=1, max_length=50, description="Type of transmission")
    powerPS: float = Field(gt=0, description="Engine power measured in PS (metric horsepower)")
    model: str = Field(min_length=1, max_length=50, description="Specific car model")
    kilometer: float = Field(gt=0, description="Approximate kilometers driven")
    fuelType: str = Field(min_length=1, max_length=50, description="Type of fuel used")
    brand: str = Field(min_length=1, max_length=50, description="Manufacturer/brand of the car")
    notRepairedDamage: str = Field(min_length=1, max_length=50, description="Whether the car has unrepaired damage")


@app.get("/")
def home():
    return {
        "Project": "Used Car Price Prediction",
        "Message": "Running",
        "Endpoint": "Send post request to predict",
    }


@app.post("/predict")
def predict(car: car_price):
    try:
        
        input_data = pd.DataFrame([{
            "seller": car.seller,
            "abtest": car.ABtest,
            "vehicleType": car.vehicle_type,
            "yearOfRegistration": car.year_of_registration,
            "gearbox": car.gearbox,
            "powerPS": car.powerPS,
            "model": car.model,
            "kilometer": car.kilometer,
            "fuelType": car.fuelType,
            "brand": car.brand,
            "notRepairedDamage": car.notRepairedDamage,
        }])

        encoded = encoder.transform(input_data[categorical_cols])
        scaled = scaler.transform(input_data[numeric_cols])

 
        processed = np.hstack([encoded, scaled])

        input_tensor = torch.tensor(processed, dtype=torch.float32).to(device)

        with torch.no_grad():
            prediction = model(input_tensor)[0].item()

        return {"predicted_price": round(prediction, 2)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction Failed: {str(e)}")
    

@app.post("/upload-file")
async def uplod_file(file: UploadFile=File(...))  :
    if not file.filename.endswith(".csv") :
        raise HTTPException(
            status_code=400,
            detail="please upload a csv file"
        )  
    contents=await file.read()
    
    df=pd.read_csv(io.BytesIO(contents))
    required_columns=['seller', 'abtest', 'vehicleType', 'yearOfRegistration', 'gearbox', 'powerPS', 'model', 'kilometer', 'fuelType', 'brand', 'notRepairedDamage','price']
    
    missing_cols=[cols for cols in required_columns if cols not in df.columns]
    
    if missing_cols:
        raise HTTPException(
            status_code=400,
                        detail=f'This columns are missing from your file{missing_cols}'
        )
    
    if len(df)==0:
            raise HTTPException(
                status_code=400,
                detail='The uploaded file has no data rows'
                
                )
    
    try:
        encode=encoder.transform(df[categorical_cols])
        scaled=scaler.transform(df[numeric_cols])
        final= np.hstack([encode,scaled])
        t1=torch.tensor(final, dtype=torch.float32).to(device)

        with torch.no_grad():
            model.to(device)
            predictions=model(t1)
        predictions = predictions.cpu().numpy().flatten()

        df["predicated_price"] = predictions

        df["predicated_price"] = df["predicated_price"].apply(lambda x: f"${x:,.0f}")

        output = df.to_csv(index=False)
        return StreamingResponse(io.StringIO(output),media_type="text/csv",headers={"Content-Disposition":"attachment ; filename=car_prices.csv"})
    except Exception as e:
        raise HTTPException(
                    status_code=500,
                    detail=f"prediction failed :{str(e)}"
                )
    


