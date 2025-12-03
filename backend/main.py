from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime, timedelta
import joblib
import os
from pathlib import Path
import numpy as np
import pandas as pd
import hashlib

# Initialize FastAPI app
app = FastAPI(
    title="Temperature Forecaster API",
    description="ML-powered temperature forecasting API for Cagayan de Oro City",
    version="1.0.0"
)

# Configure CORS to allow frontend requests

# Consolidated list of ALL allowed origins, including your Vercel domain
origins = [
    # VERCEL PRODUCTION DOMAIN (REQUIRED FIX)
    "https://temp-cdo.vercel.app", 
    # VERCEL PREVIEW DEPLOYMENTS (Useful for testing PRs)
    "https://temp-cdo-git-*.vercel.app", 
    # Local Development Environments
    "http://localhost:3000",
    "http://localhost:5173",
    "http://localhost:8080",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:8080",
    # Add your Render domain if you need to access it directly (usually not needed)
    "https://tempcdo.onrender.com",
]

app.add_middleware(
    CORSMiddleware,
    # Use the consolidated 'origins' list here:
    allow_origins=origins, 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variable to store the loaded model
model = None

# Global variable to store the historical temperature data
historical_data = None

# Get the path to the model file (assuming it's in the project root)
MODEL_PATH = Path(__file__).parent.parent / "temp_forecaster_model.joblib"

# Get the path to the historical data file
HISTORICAL_DATA_PATH = Path(__file__).parent.parent / "hourly_data_new.csv"

# Feature names will be extracted from the model after loading
EXPECTED_FEATURES = None

def load_historical_data():
    """Load the historical temperature dataset from CSV"""
    global historical_data
    if historical_data is None:
        if not HISTORICAL_DATA_PATH.exists():
            raise FileNotFoundError(f"Historical data file not found at {HISTORICAL_DATA_PATH}")
        try:
            # Read CSV file
            df = pd.read_csv(HISTORICAL_DATA_PATH)
            
            # Parse datetime column (format: "1/1/2024 0:00")
            df['datetime'] = pd.to_datetime(df['datetime'], format='%m/%d/%Y %H:%M')
            
            # Sort by datetime to ensure proper ordering
            df = df.sort_values('datetime').reset_index(drop=True)
            
            # Create a datetime index for fast lookups
            df.set_index('datetime', inplace=True)
            
            # Ensure temperature_2m column exists and is numeric
            if 'temperature_2m' not in df.columns:
                raise ValueError("temperature_2m column not found in historical data")
            df['temperature_2m'] = pd.to_numeric(df['temperature_2m'], errors='coerce')
            
            historical_data = df
            print(f"Historical data loaded successfully from {HISTORICAL_DATA_PATH}")
            print(f"Data range: {df.index.min()} to {df.index.max()}")
            print(f"Total records: {len(df)}")
        except Exception as e:
            raise Exception(f"Error loading historical data: {str(e)}")
    return historical_data

def get_historical_temperature(target_datetime: datetime):
    """
    Get historical temperature for a specific datetime.
    Returns None if data is not available.
    """
    if historical_data is None:
        return None
    
    # Round to nearest hour
    target_datetime = target_datetime.replace(minute=0, second=0, microsecond=0)
    
    try:
        # Try to get exact match
        if target_datetime in historical_data.index:
            return historical_data.loc[target_datetime, 'temperature_2m']
        
        # If exact match not found, try to find nearest (within 1 hour)
        # Get the closest datetime
        idx = historical_data.index.get_indexer([target_datetime], method='nearest')[0]
        closest_datetime = historical_data.index[idx]
        
        # Only return if within 1 hour
        if abs((closest_datetime - target_datetime).total_seconds()) <= 3600:
            return historical_data.iloc[idx]['temperature_2m']
        
        return None
    except Exception as e:
        print(f"Error getting historical temperature for {target_datetime}: {str(e)}")
        return None

def get_lag_features_and_rolling_stats(target_datetime: datetime, hour: int):
    """
    Get lag features and rolling statistics from historical data.
    
    Returns:
        dict with keys: lag_1, lag_3, lag_24, rolling_mean_24, rolling_std_24
        Values are None if data is not available
    """
    result = {
        'lag_1': None,
        'lag_3': None,
        'lag_24': None,
        'rolling_mean_24': None,
        'rolling_std_24': None
    }
    
    if historical_data is None:
        return result
    
    # Round target datetime to nearest hour
    target_datetime = target_datetime.replace(minute=0, second=0, microsecond=0)
    
    # Get lag_1: 1 hour ago
    lag_1_datetime = target_datetime - timedelta(hours=1)
    result['lag_1'] = get_historical_temperature(lag_1_datetime)
    
    # Get lag_3: 3 hours ago
    lag_3_datetime = target_datetime - timedelta(hours=3)
    result['lag_3'] = get_historical_temperature(lag_3_datetime)
    
    # Get lag_24: 24 hours ago (same hour yesterday)
    lag_24_datetime = target_datetime - timedelta(hours=24)
    result['lag_24'] = get_historical_temperature(lag_24_datetime)
    
    # Get rolling statistics: last 24 hours
    # Get temperatures for the last 24 hours (including current hour)
    rolling_temps = []
    for i in range(24):
        check_datetime = target_datetime - timedelta(hours=i)
        temp = get_historical_temperature(check_datetime)
        if temp is not None:
            rolling_temps.append(temp)
    
    if len(rolling_temps) >= 1:
        result['rolling_mean_24'] = np.mean(rolling_temps)
        if len(rolling_temps) > 1:
            result['rolling_std_24'] = np.std(rolling_temps)
        else:
            result['rolling_std_24'] = 2.8  # Default std if only one value
    
    return result

def load_model():
    """Load the ML model from the joblib file and extract feature names"""
    global model, EXPECTED_FEATURES
    if model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
        try:
            model = joblib.load(MODEL_PATH)
            # Extract feature names from the model (LightGBM stores them in feature_name_)
            if hasattr(model, 'feature_name_'):
                EXPECTED_FEATURES = model.feature_name_
                print(f"Model loaded successfully from {MODEL_PATH}")
                print(f"Model expects {len(EXPECTED_FEATURES)} features")
            else:
                raise Exception("Model does not have feature_name_ attribute. Cannot determine expected features.")
        except Exception as e:
            raise Exception(f"Error loading model: {str(e)}")
    return model

# Load model and historical data on startup
@app.on_event("startup")
async def startup_event():
    """Load the model and historical data when the application starts"""
    try:
        load_historical_data()
        load_model()
        print("FastAPI server started and model loaded successfully")
    except Exception as e:
        print(f"Warning: Could not load model or historical data on startup: {str(e)}")

# Request/Response models
class PredictionRequest(BaseModel):
    date: str  # ISO format date string (e.g., "2024-01-15")
    hour: int  # Hour in 24-hour format (0-23)

class SinglePredictionRequest(BaseModel):
    date: str  # ISO format date string (e.g., "2024-01-15")
    time: str  # Time string (e.g., "12:00:00" or "12:00")

class DailyForecastRequest(BaseModel):
    date: str  # ISO format date string (e.g., "2024-01-15")

class HourlyPrediction(BaseModel):
    hour: int
    time: str  # Formatted as "HH:00"
    temperature: float

class PredictionResponse(BaseModel):
    requested_date: str
    requested_hour: int
    requested_temperature: float
    hourly_forecast: list[HourlyPrediction]

class SinglePredictionResponse(BaseModel):
    date: str
    time: str
    temperature: float

class DailyForecastResponse(BaseModel):
    date: str
    forecast: list[HourlyPrediction]

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool

@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint"""
    return {
        "status": "ok",
        "model_loaded": model is not None
    }

@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint"""
    return {
        "status": "ok",
        "model_loaded": model is not None
    }

@app.post("/api/predict/single", response_model=SinglePredictionResponse)
async def predict_single(request: SinglePredictionRequest):
    """
    Get temperature prediction for a specific date and time.
    Returns a single temperature prediction.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please check server logs.")
    
    try:
        # Parse the date
        date_obj = datetime.fromisoformat(request.date)
        
        # Parse time string (accepts "HH:00:00" or "HH:00" or just "HH")
        time_parts = request.time.split(':')
        hour = int(time_parts[0])
        
        # Validate hour
        if not (0 <= hour <= 23):
            raise HTTPException(status_code=400, detail="Hour must be between 0 and 23")
        
        # Set the hour on the date object
        date_obj = date_obj.replace(hour=hour, minute=0, second=0, microsecond=0)
        
        # Create features for the requested prediction
        features_df = create_features(date_obj, hour)
        
        # Make prediction
        try:
            # features_df is already a DataFrame with correct column names
            prediction = model.predict(features_df)
            temperature = float(prediction[0]) if hasattr(prediction, '__iter__') else float(prediction)
        except Exception as e:
            print(f"Model prediction error: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Model prediction failed: {str(e)}"
            )
        
        # Format time as HH:00:00
        formatted_time = f"{hour:02d}:00:00"
        
        return {
            "date": request.date,
            "time": formatted_time,
            "temperature": round(temperature, 2)
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid date or time format: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/api/forecast/daily", response_model=DailyForecastResponse)
async def forecast_daily(request: DailyForecastRequest):
    """
    Get 24-hour temperature forecast for a specific date.
    Returns predictions for all hours from 00:00 to 23:00.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please check server logs.")
    
    try:
        # Parse the date
        date_obj = datetime.fromisoformat(request.date)
        
        # Generate 24-hour forecast (00:00 to 23:00)
        hourly_forecast = []
        temp_history = {}
        
        for hour in range(24):
            # Calculate the datetime for this hour
            forecast_datetime = date_obj.replace(hour=hour, minute=0, second=0, microsecond=0)
            
            # Get historical temperatures for lag features from the dataset
            hist_features = get_lag_features_and_rolling_stats(forecast_datetime, hour)
            
            # Prepare historical temperatures dictionary for create_features
            historical_temps = {}
            if hist_features['lag_1'] is not None:
                historical_temps[1] = hist_features['lag_1']
            if hist_features['lag_3'] is not None:
                historical_temps[3] = hist_features['lag_3']
            if hist_features['lag_24'] is not None:
                historical_temps[24] = hist_features['lag_24']
            if hist_features['rolling_mean_24'] is not None:
                historical_temps['mean_24'] = hist_features['rolling_mean_24']
            if hist_features['rolling_std_24'] is not None:
                historical_temps['std_24'] = hist_features['rolling_std_24']
            
            # For hours where we have predictions from previous hours in this forecast,
            # prefer those over historical data for lag_1 and lag_3
            if hour >= 1 and hour - 1 in temp_history:
                historical_temps[1] = temp_history[hour - 1]
            if hour >= 3 and hour - 3 in temp_history:
                historical_temps[3] = temp_history[hour - 3]
            
            # For rolling stats, combine historical data with predictions from this forecast
            if len(temp_history) > 0:
                # Get recent historical temperatures (for hours before current forecast)
                recent_hist_temps = []
                for i in range(1, min(24, hour + 1)):
                    check_datetime = forecast_datetime - timedelta(hours=i)
                    hist_temp = get_historical_temperature(check_datetime)
                    if hist_temp is not None:
                        recent_hist_temps.append(hist_temp)
                
                # Combine with predictions from this forecast
                combined_temps = recent_hist_temps + list(temp_history.values())
                if len(combined_temps) >= 1:
                    historical_temps['mean_24'] = np.mean(combined_temps[-24:]) if len(combined_temps) >= 24 else np.mean(combined_temps)
                    if len(combined_temps) > 1:
                        historical_temps['std_24'] = np.std(combined_temps[-24:]) if len(combined_temps) >= 24 else np.std(combined_temps)
                    else:
                        historical_temps['std_24'] = 2.8
            
            # Create features for this hour
            forecast_features_df = create_features(forecast_datetime, hour, historical_temps)
            
            try:
                # forecast_features_df is already a DataFrame with correct column names
                forecast_prediction = model.predict(forecast_features_df)
                forecast_temp = float(forecast_prediction[0]) if hasattr(forecast_prediction, '__iter__') else float(forecast_prediction)
                temp_history[hour] = forecast_temp
            except Exception as e:
                print(f"Forecast prediction error for hour {hour}: {str(e)}")
                # Use estimated temperature if prediction fails
                forecast_temp = 28.0 + np.sin(2 * np.pi * hour / 24) * 3.0
                temp_history[hour] = forecast_temp
            
            hourly_forecast.append({
                "hour": hour,
                "time": f"{hour:02d}:00",
                "temperature": round(forecast_temp, 2)
            })
        
        return {
            "date": request.date,
            "forecast": hourly_forecast
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid date format: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forecast error: {str(e)}")

def create_features(date_obj: datetime, hour: int, historical_temps: dict = None):
    """
    Create all 50 features required by the model.
    Uses real historical data from the dataset for lag features and rolling statistics.
    
    Args:
        date_obj: Datetime object for the prediction
        hour: Hour of day (0-23)
        historical_temps: Dictionary with historical temperatures for lag features
                          Format: {offset_hours: temperature} or {'mean_24': value, 'std_24': value}
    
    Returns:
        DataFrame with features in the correct order
    """
    month = date_obj.month
    day_of_week = date_obj.weekday()
    day_of_year = date_obj.timetuple().tm_yday
    year = date_obj.year
    
    # Create a deterministic seed based on the date to make random values consistent for the same date
    # This ensures the same date always gets the same "random" weather values
    date_str = f"{year}-{month:02d}-{date_obj.day:02d}"
    seed = int(hashlib.md5(date_str.encode()).hexdigest()[:8], 16) % (2**31)
    rng = np.random.RandomState(seed)
    
    # Determine if summer - matches training notebook: months 6, 7, 8 (June-August)
    is_summer = 1 if month in [6, 7, 8] else 0
    
    # Add seasonal temperature variation based on month and day of year
    seasonal_temp_adjustment = np.sin(2 * np.pi * (day_of_year - 80) / 365.0) * 2.5
    daily_temp_variation = np.sin(2 * np.pi * day_of_year / 7.0) * 0.8
    seasonal_temp_adjustment += daily_temp_variation
    
    # Time-based features
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    
    # Typical weather patterns for Cagayan de Oro (tropical climate)
    # Precipitation (mm) - higher in afternoon/evening, lower at night
    is_wet_season = 152 <= day_of_year <= 304
    if is_wet_season:
        wet_season_intensity = np.sin(np.pi * (day_of_year - 152) / 152.0)
        base_precipitation = (1.0 if hour >= 12 else 0.5) * wet_season_intensity
    else:
        base_precipitation = 0.3 if hour >= 12 else 0.0
    
    daily_precip_variation = np.sin(2 * np.pi * day_of_year / 5.0) * 0.3
    precipitation = base_precipitation + daily_precip_variation + (rng.uniform(0, 1.5) if hour >= 12 else 0.0)
    rain = 1.0 if precipitation > 0.5 else 0.0
    
    # Weather code (simplified: 0=clear, 1=partly cloudy, 2=cloudy, 3=rainy)
    if hour < 6 or hour > 18:  # Night
        weather_code = 0  # Clear
    elif hour < 12:  # Morning
        weather_code = 1  # Partly cloudy
    else:  # Afternoon/Evening
        weather_code = 2 if precipitation < 1.0 else 3
    
    # Cloud cover (0-100%) - varies by season and hour
    base_cloud = 20.0
    if hour < 6 or hour > 18:
        base_cloud = 20.0
    elif hour < 12:
        base_cloud = 40.0
    else:
        base_cloud = 60.0
    
    is_wet_season = 152 <= day_of_year <= 304
    if is_wet_season:
        wet_season_intensity = np.sin(np.pi * (day_of_year - 152) / 152.0)
        base_cloud += 15.0 * wet_season_intensity
    
    daily_cloud_variation = np.sin(2 * np.pi * day_of_year / 7.0) * 5.0
    cloud_cover = base_cloud + daily_cloud_variation + rng.uniform(-10, 20)
    cloud_cover = max(0, min(100, cloud_cover))
    
    cloud_cover_low = cloud_cover * 0.3
    cloud_cover_mid = cloud_cover * 0.4
    cloud_cover_high = cloud_cover * 0.3
    
    # Evapotranspiration (mm/day) - typical for tropical climate
    et0_fao_evapotranspiration = 4.0 + np.sin(2 * np.pi * hour / 24) * 2.0
    
    # Wind features (typical for Cagayan de Oro)
    day_variation = np.sin(2 * np.pi * day_of_year / 365.0) * 1.5
    wind_base = 2.0 + day_variation + (day_of_year / 365.0) * 0.5
    wind_speed_10m = wind_base + rng.uniform(-0.5, 1.5)  # m/s
    wind_speed_100m = wind_speed_10m * 1.5
    wind_direction_10m = 180.0 + np.sin(2 * np.pi * day_of_year / 365.0) * 30
    wind_direction_100m = wind_direction_10m + rng.uniform(-10, 10)
    wind_gusts_10m = wind_speed_10m * 1.3
    
    # Soil temperature (typical for tropical climate, varies with air temp)
    base_temp_soil = 28.0 + seasonal_temp_adjustment + np.sin(2 * np.pi * hour / 24) * 5.0
    soil_temperature_0_to_7cm = base_temp_soil + rng.uniform(-1, 1)
    soil_temperature_7_to_28cm = base_temp_soil - 1.0
    soil_temperature_28_to_100cm = base_temp_soil - 2.0
    soil_temperature_100_to_255cm = base_temp_soil - 3.0
    
    # Soil moisture (typical for tropical climate) - higher in wet season
    base_moisture = 0.6
    if month in [6, 7, 8, 9, 10]:  # Wet season
        base_moisture = 0.75
    soil_moisture_0_to_7cm = base_moisture + rng.uniform(-0.1, 0.1)
    soil_moisture_7_to_28cm = base_moisture + 0.05
    soil_moisture_28_to_100cm = base_moisture + 0.1
    soil_moisture_100_to_255cm = base_moisture + 0.15
    
    # Radiation features (W/m²) - depends on time of day
    if 6 <= hour <= 18:  # Daytime
        max_radiation = 800.0
        radiation_factor = np.sin(np.pi * (hour - 6) / 12)
        shortwave_radiation = max_radiation * radiation_factor
        direct_radiation = shortwave_radiation * 0.7
        diffuse_radiation = shortwave_radiation * 0.3
        direct_normal_irradiance = direct_radiation / max(0.1, radiation_factor)
        global_tilted_irradiance = shortwave_radiation * 0.9
    else:  # Nighttime
        shortwave_radiation = 0.0
        direct_radiation = 0.0
        diffuse_radiation = 0.0
        direct_normal_irradiance = 0.0
        global_tilted_irradiance = 0.0
    
    # Terrestrial radiation varies by day of year with seasonal pattern
    terrestrial_radiation = 300.0 + np.sin(2 * np.pi * day_of_year / 365.0) * 20 + rng.uniform(-10, 10)
    
    # Instantaneous radiation (same as regular for this use case)
    shortwave_radiation_instant = shortwave_radiation
    direct_radiation_instant = direct_radiation
    diffuse_radiation_instant = diffuse_radiation
    direct_normal_irradiance_instant = direct_normal_irradiance
    global_tilted_irradiance_instant = global_tilted_irradiance
    terrestrial_radiation_instant = terrestrial_radiation
    
    # Lag features - use historical temps from dataset if available, otherwise estimate
    base_temp = 28.0 + seasonal_temp_adjustment
    
    if historical_temps:
        # Use provided historical temperatures (from dataset or previous predictions)
        temp_lag_1 = historical_temps.get(1)
        temp_lag_3 = historical_temps.get(3)
        temp_lag_24 = historical_temps.get(24)
        temp_rolling_mean_24 = historical_temps.get('mean_24')
        temp_rolling_std_24 = historical_temps.get('std_24')
        
        # Fallback to estimated values if not provided
        if temp_lag_1 is None:
            prev_hour = (hour - 1) % 24
            temp_lag_1 = base_temp + np.sin(2 * np.pi * prev_hour / 24) * 5.0
        if temp_lag_3 is None:
            prev_3_hour = (hour - 3) % 24
            temp_lag_3 = base_temp + np.sin(2 * np.pi * prev_3_hour / 24) * 5.0
        if temp_lag_24 is None:
            hourly_var = np.sin(2 * np.pi * hour / 24) * 5.0
            day_var = np.sin(2 * np.pi * day_of_year / 365.0) * 0.8
            temp_lag_24 = base_temp + hourly_var + day_var
        if temp_rolling_mean_24 is None:
            temp_rolling_mean_24 = base_temp + np.sin(2 * np.pi * hour / 24) * 2.0
        if temp_rolling_std_24 is None:
            temp_rolling_std_24 = 2.8 + np.abs(np.sin(2 * np.pi * hour / 24)) * 1.7
    else:
        # Enhanced synthetic history so model can explore 22‑36 °C range
        seasonal_offset = 0.0
        if month in (12, 1, 2):
            seasonal_offset -= 3.0
        elif month in (3, 4, 5):
            seasonal_offset += 3.0
        elif month in (6, 7, 8):
            seasonal_offset += 1.5
        else:
            seasonal_offset -= 1.0

        daily_random_variation = rng.uniform(-2.0, 2.0)
        hourly_amp = 6.0 + rng.uniform(-1.0, 2.0)  # ~±6–8 °C range

        temp_baseline = 28.0 + seasonal_temp_adjustment + seasonal_offset + daily_random_variation

        def temp_at_hour(h: int) -> float:
            wave = np.sin(2 * np.pi * h / 24.0)
            return temp_baseline + wave * hourly_amp

        prev_hour = (hour - 1) % 24
        prev3_hour = (hour - 3) % 24
        yesterday_hour = hour

        temp_lag_1 = temp_at_hour(prev_hour) + rng.uniform(-0.8, 0.8)
        temp_lag_3 = temp_at_hour(prev3_hour) + rng.uniform(-1.0, 1.0)
        temp_lag_24 = temp_at_hour(yesterday_hour) + rng.uniform(-1.5, 1.5)

        simulated_series = [
            temp_at_hour((hour - i) % 24) + rng.uniform(-1.0, 1.0)
            for i in range(1, 25)
        ]
        temp_rolling_mean_24 = float(np.mean(simulated_series))
        rolling_std_raw = float(np.std(simulated_series))
        temp_rolling_std_24 = float(np.clip(rolling_std_raw + rng.uniform(-0.3, 0.4), 1.5, 4.0))
    
    # Other lag features (estimated) - vary by date
    base_humidity = 75.0
    if month in [6, 7, 8, 9, 10]:  # Wet season
        base_humidity = 85.0
    humidity_variation = np.sin(2 * np.pi * day_of_year / 365.0) * 5.0
    relative_humidity_2m_lag_1 = base_humidity + humidity_variation + (day_of_year / 10.0) % 3 - 1.5 + rng.uniform(-3, 3)
    dew_point_2m_lag_1 = temp_lag_1 - 5.0
    apparent_temperature_lag_1 = temp_lag_1 + 1.0
    pressure_variation = np.sin(2 * np.pi * day_of_year / 365.0) * 8.0
    pressure_msl_lag_1 = 1013.0 + pressure_variation + (day_of_year / 5.0) % 4 - 2.0 + rng.uniform(-3, 3)
    surface_pressure_lag_1 = pressure_msl_lag_1 - 10.0
    vapour_pressure_deficit_lag_1 = 1.5 + np.sin(2 * np.pi * day_of_year / 365.0) * 0.3 + rng.uniform(-0.2, 0.2)
    
    # Interaction feature
    temp_x_humidity = temp_lag_1 * relative_humidity_2m_lag_1 / 100.0
    
    # Ensure model is loaded
    if EXPECTED_FEATURES is None:
        raise ValueError("Model not loaded. EXPECTED_FEATURES is None. Call load_model() first.")
    
    # Create feature dictionary matching the exact model feature names
    feature_dict = {
        'precipitation': precipitation,
        'rain': rain,
        'weather_code': weather_code,
        'cloud_cover': cloud_cover,
        'cloud_cover_low': cloud_cover_low,
        'cloud_cover_mid': cloud_cover_mid,
        'cloud_cover_high': cloud_cover_high,
        'et0_fao_evapotranspiration': et0_fao_evapotranspiration,
        'wind_speed_10m': wind_speed_10m,
        'wind_speed_100m': wind_speed_100m,
        'wind_direction_10m': wind_direction_10m,
        'wind_direction_100m': wind_direction_100m,
        'wind_gusts_10m': wind_gusts_10m,
        'soil_temperature_0_to_7cm': soil_temperature_0_to_7cm,
        'soil_temperature_7_to_28cm': soil_temperature_7_to_28cm,
        'soil_temperature_28_to_100cm': soil_temperature_28_to_100cm,
        'soil_temperature_100_to_255cm': soil_temperature_100_to_255cm,
        'soil_moisture_0_to_7cm': soil_moisture_0_to_7cm,
        'soil_moisture_7_to_28cm': soil_moisture_7_to_28cm,
        'soil_moisture_28_to_100cm': soil_moisture_28_to_100cm,
        'soil_moisture_100_to_255cm': soil_moisture_100_to_255cm,
        'shortwave_radiation': shortwave_radiation,
        'direct_radiation': direct_radiation,
        'diffuse_radiation': diffuse_radiation,
        'direct_normal_irradiance': direct_normal_irradiance,
        'global_tilted_irradiance': global_tilted_irradiance,
        'terrestrial_radiation': terrestrial_radiation,
        'shortwave_radiation_instant': shortwave_radiation_instant,
        'direct_radiation_instant': direct_radiation_instant,
        'diffuse_radiation_instant': diffuse_radiation_instant,
        'direct_normal_irradiance_instant': direct_normal_irradiance_instant,
        'global_tilted_irradiance_instant': global_tilted_irradiance_instant,
        'terrestrial_radiation_instant': terrestrial_radiation_instant,
        'hour_sin': hour_sin,
        'hour_cos': hour_cos,
        'dayofweek': day_of_week,
        'month': month,
        'is_summer': is_summer,
        'relative_humidity_2m_lag_1': relative_humidity_2m_lag_1,
        'dew_point_2m_lag_1': dew_point_2m_lag_1,
        'apparent_temperature_lag_1': apparent_temperature_lag_1,
        'pressure_msl_lag_1': pressure_msl_lag_1,
        'surface_pressure_lag_1': surface_pressure_lag_1,
        'vapour_pressure_deficit_lag_1': vapour_pressure_deficit_lag_1,
        'temperature_2m_lag_1': temp_lag_1,
        'temperature_2m_lag_3': temp_lag_3,
        'temperature_2m_lag_24': temp_lag_24,
        'temperature_2m_rolling_mean_24': temp_rolling_mean_24,
        'temperature_2m_rolling_std_24': temp_rolling_std_24,
        'temp_x_humidity': temp_x_humidity
    }
    
    # Convert to DataFrame using model's feature order (ensures exact match)
    if EXPECTED_FEATURES is None:
        raise ValueError("Model not loaded. EXPECTED_FEATURES is None.")
    
    features_df = pd.DataFrame([[feature_dict[feat] for feat in EXPECTED_FEATURES]], columns=EXPECTED_FEATURES)
    
    return features_df

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Get temperature prediction for a specific date and hour.
    Also returns a 24-hour forecast starting from the requested hour.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please check server logs.")
    
    try:
        # Parse the date
        date_obj = datetime.fromisoformat(request.date)
        date_obj = date_obj.replace(hour=request.hour, minute=0, second=0, microsecond=0)
        
        # Validate hour
        if not (0 <= request.hour <= 23):
            raise HTTPException(status_code=400, detail="Hour must be between 0 and 23")
        
        # Get historical temperatures for lag features from the dataset
        hist_features = get_lag_features_and_rolling_stats(date_obj, request.hour)
        
        # Prepare historical temperatures dictionary
        historical_temps = {}
        if hist_features['lag_1'] is not None:
            historical_temps[1] = hist_features['lag_1']
        if hist_features['lag_3'] is not None:
            historical_temps[3] = hist_features['lag_3']
        if hist_features['lag_24'] is not None:
            historical_temps[24] = hist_features['lag_24']
        if hist_features['rolling_mean_24'] is not None:
            historical_temps['mean_24'] = hist_features['rolling_mean_24']
        if hist_features['rolling_std_24'] is not None:
            historical_temps['std_24'] = hist_features['rolling_std_24']
        
        # Create features for the requested prediction
        features_df = create_features(date_obj, request.hour, historical_temps)
        
        # Make prediction for the requested hour
        try:
            prediction = model.predict(features_df)
            requested_temp = float(prediction[0]) if hasattr(prediction, '__iter__') else float(prediction)
        except Exception as e:
            print(f"Model prediction error: {str(e)}")
            print(f"Features shape: {features_df.shape}")
            raise HTTPException(
                status_code=500,
                detail=f"Model prediction failed: {str(e)}"
            )
        
        # Generate 24-hour forecast starting from requested hour
        hourly_forecast = []
        temp_history = {0: requested_temp}
        
        for i in range(24):
            forecast_hour = (request.hour + i) % 24
            # Calculate the actual datetime for this forecast hour
            forecast_datetime = date_obj + timedelta(hours=i)
            
            # Get historical temperatures for lag features from the dataset
            hist_features = get_lag_features_and_rolling_stats(forecast_datetime, forecast_hour)
            
            # Prepare historical temperatures dictionary
            historical_temps = {}
            if hist_features['lag_1'] is not None:
                historical_temps[1] = hist_features['lag_1']
            if hist_features['lag_3'] is not None:
                historical_temps[3] = hist_features['lag_3']
            if hist_features['lag_24'] is not None:
                historical_temps[24] = hist_features['lag_24']
            if hist_features['rolling_mean_24'] is not None:
                historical_temps['mean_24'] = hist_features['rolling_mean_24']
            if hist_features['rolling_std_24'] is not None:
                historical_temps['std_24'] = hist_features['rolling_std_24']
            
            # Prefer predictions from previous hours in this forecast for lag_1 and lag_3
            if i >= 1 and i - 1 in temp_history:
                historical_temps[1] = temp_history[i - 1]
            if i >= 3 and i - 3 in temp_history:
                historical_temps[3] = temp_history[i - 3]
            
            # For rolling stats, combine historical data with predictions from this forecast
            if len(temp_history) > 0:
                recent_hist_temps = []
                for j in range(1, min(24, i + 1)):
                    check_datetime = forecast_datetime - timedelta(hours=j)
                    hist_temp = get_historical_temperature(check_datetime)
                    if hist_temp is not None:
                        recent_hist_temps.append(hist_temp)
                
                combined_temps = recent_hist_temps + list(temp_history.values())
                if len(combined_temps) >= 1:
                    historical_temps['mean_24'] = np.mean(combined_temps[-24:]) if len(combined_temps) >= 24 else np.mean(combined_temps)
                    if len(combined_temps) > 1:
                        historical_temps['std_24'] = np.std(combined_temps[-24:]) if len(combined_temps) >= 24 else np.std(combined_temps)
                    else:
                        historical_temps['std_24'] = 2.8
            
            # Create features for this forecast hour
            forecast_features_df = create_features(forecast_datetime, forecast_hour, historical_temps)
            
            try:
                forecast_prediction = model.predict(forecast_features_df)
                forecast_temp = float(forecast_prediction[0]) if hasattr(forecast_prediction, '__iter__') else float(forecast_prediction)
                temp_history[i] = forecast_temp
            except Exception as e:
                print(f"Forecast prediction error for hour {forecast_hour}: {str(e)}")
                forecast_temp = requested_temp
                temp_history[i] = forecast_temp
            
            hourly_forecast.append({
                "hour": forecast_hour,
                "time": f"{forecast_hour:02d}:00",
                "temperature": round(forecast_temp, 2)
            })
        
        return {
            "requested_date": request.date,
            "requested_hour": request.hour,
            "requested_temperature": round(requested_temp, 2),
            "hourly_forecast": hourly_forecast
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid date format: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

