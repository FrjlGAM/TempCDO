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
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite default port
        "http://localhost:8080",  # Vite configured port
        "http://localhost:3000",  # Alternative port
        "http://127.0.0.1:5173",
        "http://127.0.0.1:8080",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variable to store the loaded model
model = None

# Get the path to the model file (assuming it's in the project root)
MODEL_PATH = Path(__file__).parent.parent / "temp_forecaster_model.joblib"

# Feature names will be extracted from the model after loading
EXPECTED_FEATURES = None

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

# Load model on startup
@app.on_event("startup")
async def startup_event():
    """Load the model when the application starts"""
    try:
        load_model()
        print("FastAPI server started and model loaded successfully")
    except Exception as e:
        print(f"Warning: Could not load model on startup: {str(e)}")

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
        
        # Calculate seasonal adjustment once for this date (used in fallback estimates)
        day_of_year = date_obj.timetuple().tm_yday
        seasonal_temp_adjustment = np.sin(2 * np.pi * (day_of_year - 80) / 365.0) * 2.5
        daily_temp_variation = np.sin(2 * np.pi * day_of_year / 7.0) * 0.8
        seasonal_temp_adjustment += daily_temp_variation
        
        for hour in range(24):
            # Calculate the datetime for this hour
            forecast_datetime = date_obj.replace(hour=hour, minute=0, second=0, microsecond=0)
            
            # Prepare historical temperatures for lag features
            # Use actual predicted temperatures from previous hours when available
            historical_temps = {}
            if hour >= 1:
                # Use the actual predicted temperature from the previous hour
                prev_temp = temp_history.get(hour-1)
                if prev_temp is None:
                    # Estimate based on hour pattern if not available
                    base_est = 28.0 + seasonal_temp_adjustment
                    prev_hour = hour - 1
                    prev_temp = base_est + np.sin(2 * np.pi * prev_hour / 24) * 5.0
                historical_temps[1] = prev_temp
            if hour >= 3:
                # Use the actual predicted temperature from 3 hours ago
                prev_3_temp = temp_history.get(hour-3)
                if prev_3_temp is None:
                    base_est = 28.0 + seasonal_temp_adjustment
                    prev_3_hour = hour - 3
                    prev_3_temp = base_est + np.sin(2 * np.pi * prev_3_hour / 24) * 5.0
                historical_temps[3] = prev_3_temp
            # For lag_24, we'd need previous day's data, so estimate based on hour pattern
            if hour == 0:
                # Estimate previous day midnight - use typical night temp
                base_est = 28.0 + seasonal_temp_adjustment
                historical_temps[24] = base_est - 3.0  # Night is cooler
            else:
                # Use midnight temp as base, then adjust for current hour pattern
                midnight_temp = temp_history.get(0)
                if midnight_temp is None:
                    base_est = 28.0 + seasonal_temp_adjustment
                    midnight_temp = base_est - 3.0
                # Estimate what temp would be at this hour yesterday
                hour_factor = np.sin(2 * np.pi * hour / 24) * 5.0
                historical_temps[24] = midnight_temp + hour_factor
            
            # Calculate rolling statistics if we have enough history
            if len(temp_history) >= 24:
                recent_temps = list(temp_history.values())[-24:]
                historical_temps['mean_24'] = np.mean(recent_temps)
                historical_temps['std_24'] = np.std(recent_temps) if len(recent_temps) > 1 else 2.8
            elif len(temp_history) > 0:
                # Use available history for rolling stats
                recent_temps = list(temp_history.values())
                historical_temps['mean_24'] = np.mean(recent_temps)
                historical_temps['std_24'] = np.std(recent_temps) if len(recent_temps) > 1 else 2.8
            else:
                # Use seasonal-adjusted base for initial estimates with hourly variation
                base_est = 28.0 + seasonal_temp_adjustment
                # Add hourly variation to the mean (represents average of last 24 hours)
                hourly_mean_factor = np.sin(2 * np.pi * hour / 24) * 2.0
                historical_temps['mean_24'] = base_est + hourly_mean_factor
                historical_temps['std_24'] = 2.8 + np.abs(np.sin(2 * np.pi * hour / 24)) * 1.7  # 2.8-4.5°C
            
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
    Since we don't have real-time weather data, we'll use reasonable defaults
    based on date/time and typical patterns for Cagayan de Oro.
    
    Args:
        date_obj: Datetime object for the prediction
        hour: Hour of day (0-23)
        historical_temps: Dictionary with historical temperatures for lag features
                          Format: {offset_hours: temperature}
    
    Returns:
        numpy array with 50 features in the correct order
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
    # Note: This matches the training data definition, not typical Philippines summer
    is_summer = 1 if month in [6, 7, 8] else 0
    
    # Add seasonal temperature variation based on month and day of year
    # Philippines has wet season (June-October) and dry season (November-May)
    # Temperature varies by season with smooth transitions
    # Use day_of_year for smooth seasonal curve instead of discrete month buckets
    # Increased to ±2.5°C for more realistic seasonal variation
    seasonal_temp_adjustment = np.sin(2 * np.pi * (day_of_year - 80) / 365.0) * 2.5  # Peak in April (day ~90-120)
    # Add additional day-specific variation for more diversity
    daily_temp_variation = np.sin(2 * np.pi * day_of_year / 7.0) * 0.8  # Weekly pattern, increased
    seasonal_temp_adjustment += daily_temp_variation
    
    # Time-based features
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    
    # Typical weather patterns for Cagayan de Oro (tropical climate)
    # These are reasonable defaults - in production, you'd fetch real weather data
    
    # Precipitation (mm) - higher in afternoon/evening, lower at night
    # Vary by day of year for wet season (June-October) with smooth transitions
    # Wet season roughly days 152-304 (June 1 - Oct 31)
    is_wet_season = 152 <= day_of_year <= 304
    if is_wet_season:
        # Smooth transition into and out of wet season
        wet_season_intensity = np.sin(np.pi * (day_of_year - 152) / 152.0)  # 0 to 1
        base_precipitation = (1.0 if hour >= 12 else 0.5) * wet_season_intensity
    else:
        base_precipitation = 0.3 if hour >= 12 else 0.0
    
    # Add day-specific variation
    daily_precip_variation = np.sin(2 * np.pi * day_of_year / 5.0) * 0.3  # 5-day cycle
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
    
    # More clouds in wet season with smooth transition
    is_wet_season = 152 <= day_of_year <= 304
    if is_wet_season:
        wet_season_intensity = np.sin(np.pi * (day_of_year - 152) / 152.0)
        base_cloud += 15.0 * wet_season_intensity
    
    # Add day-specific variation
    daily_cloud_variation = np.sin(2 * np.pi * day_of_year / 7.0) * 5.0  # Weekly pattern
    cloud_cover = base_cloud + daily_cloud_variation + rng.uniform(-10, 20)
    cloud_cover = max(0, min(100, cloud_cover))
    
    cloud_cover_low = cloud_cover * 0.3
    cloud_cover_mid = cloud_cover * 0.4
    cloud_cover_high = cloud_cover * 0.3
    
    # Evapotranspiration (mm/day) - typical for tropical climate
    et0_fao_evapotranspiration = 4.0 + np.sin(2 * np.pi * hour / 24) * 2.0
    
    # Wind features (typical for Cagayan de Oro)
    # Vary by day to make predictions different - use full day_of_year for unique values
    # Add sine wave based on day of year to create smooth seasonal variation
    day_variation = np.sin(2 * np.pi * day_of_year / 365.0) * 1.5
    wind_base = 2.0 + day_variation + (day_of_year / 365.0) * 0.5  # Varies throughout year
    wind_speed_10m = wind_base + rng.uniform(-0.5, 1.5)  # m/s
    wind_speed_100m = wind_speed_10m * 1.5
    wind_direction_10m = 180.0 + np.sin(2 * np.pi * day_of_year / 365.0) * 30  # Seasonal variation
    wind_direction_100m = wind_direction_10m + rng.uniform(-10, 10)
    wind_gusts_10m = wind_speed_10m * 1.3
    
    # Soil temperature (typical for tropical climate, varies with air temp)
    # Use consistent base temperature calculation with lag features
    # This ensures soil temp aligns with air temp estimates
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
    
    # Lag features - use historical temps if available, otherwise estimate
    # Base temperature with seasonal adjustment (Cagayan de Oro: ~24-32°C range)
    base_temp = 28.0 + seasonal_temp_adjustment  # Seasonal: ±2.5°C
    
    if historical_temps:
        # Use provided historical temperatures, but ensure they're realistic
        temp_lag_1 = historical_temps.get(1)
        temp_lag_3 = historical_temps.get(3)
        temp_lag_24 = historical_temps.get(24)
        temp_rolling_mean_24 = historical_temps.get('mean_24')
        temp_rolling_std_24 = historical_temps.get('std_24')
        
        # Fallback to estimated values if not provided (use same variation as main estimation)
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
        # Estimate based on typical patterns with realistic hourly and seasonal variation
        # Hourly variation: cooler at night (lowest ~4-6 AM), warmer during day (peak ~2-3 PM)
        # Using sine wave: sin(0) at midnight = 0, peaks at noon, back to 0 at midnight
        # Increased to ±5°C for more realistic daily temperature swings
        hourly_variation = np.sin(2 * np.pi * hour / 24) * 5.0  # ±5°C variation
        
        # Lag 1 hour: previous hour's temperature
        # Previous hour will be slightly different based on time of day
        prev_hour = (hour - 1) % 24
        temp_lag_1 = base_temp + np.sin(2 * np.pi * prev_hour / 24) * 5.0
        
        # Lag 3 hours: 3 hours ago temperature
        # This captures earlier in the day/night cycle
        prev_3_hour = (hour - 3) % 24
        temp_lag_3 = base_temp + np.sin(2 * np.pi * prev_3_hour / 24) * 5.0
        
        # Lag 24 hours: same hour yesterday (with seasonal adjustment)
        # Should be similar to current hour but may have slight day-to-day variation
        day_variation = np.sin(2 * np.pi * day_of_year / 365.0) * 0.8
        temp_lag_24 = base_temp + hourly_variation + day_variation
        
        # Rolling mean: average of last 24 hours
        # Should be close to base temperature with some smoothing
        # The mean of a full day's sine wave is approximately the base
        temp_rolling_mean_24 = base_temp + np.sin(2 * np.pi * hour / 24) * 2.0  # Increased variation for mean
        
        # Rolling std: standard deviation of last 24 hours
        # Higher during day (more variation), lower at night (more stable)
        # Typical daily std for Cagayan de Oro: 2.8-4.5°C
        temp_rolling_std_24 = 2.8 + np.abs(np.sin(2 * np.pi * hour / 24)) * 1.7  # 2.8-4.5°C range
    
    # Other lag features (estimated) - vary by date
    # Higher humidity in wet season, with day-specific variation
    base_humidity = 75.0
    if month in [6, 7, 8, 9, 10]:  # Wet season
        base_humidity = 85.0
    # Add day-specific variation using sine wave for smooth seasonal transition
    humidity_variation = np.sin(2 * np.pi * day_of_year / 365.0) * 5.0
    relative_humidity_2m_lag_1 = base_humidity + humidity_variation + (day_of_year / 10.0) % 3 - 1.5 + rng.uniform(-3, 3)
    dew_point_2m_lag_1 = temp_lag_1 - 5.0
    apparent_temperature_lag_1 = temp_lag_1 + 1.0
    # Pressure varies by day with seasonal pattern
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
    # This ensures we match the training notebook's feature engineering exactly
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
        'dayofweek': day_of_week,  # Note: model expects 'dayofweek' not 'day_of_week'
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
    
    # Create DataFrame with features in the exact order expected by the model
    # This ensures features match the training notebook exactly
    if EXPECTED_FEATURES is None:
        raise ValueError("Model not loaded. EXPECTED_FEATURES is None. Call load_model() first.")
    
    features_df = pd.DataFrame([[feature_dict[feat] for feat in EXPECTED_FEATURES]], columns=EXPECTED_FEATURES)
    
    return features_df  # Return DataFrame directly for better compatibility with model

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
        
        # Validate hour
        if not (0 <= request.hour <= 23):
            raise HTTPException(status_code=400, detail="Hour must be between 0 and 23")
        
        # Create features for the requested prediction
        # Note: For lag features, we'll estimate them since we don't have historical data
        features_df = create_features(date_obj, request.hour)
        
        # Make prediction for the requested hour
        try:
            # features_df is already a DataFrame with correct column names
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
        # Store predictions for lag features
        temp_history = {0: requested_temp}
        
        for i in range(24):
            forecast_hour = (request.hour + i) % 24
            # Calculate the actual datetime for this forecast hour
            forecast_datetime = date_obj + timedelta(hours=i)
            
            # Prepare historical temperatures for lag features
            historical_temps = {}
            if i >= 1:
                historical_temps[1] = temp_history.get(i-1, requested_temp)
            if i >= 3:
                historical_temps[3] = temp_history.get(i-3, requested_temp)
            if i >= 24:
                historical_temps[24] = temp_history.get(i-24, requested_temp)
            
            # Calculate rolling statistics if we have enough history
            if len(temp_history) >= 24:
                recent_temps = list(temp_history.values())[-24:]
                historical_temps['mean_24'] = np.mean(recent_temps)
                historical_temps['std_24'] = np.std(recent_temps) if len(recent_temps) > 1 else 2.0
            
            # Create features for this forecast hour
            forecast_features_df = create_features(forecast_datetime, forecast_hour, historical_temps)
            
            try:
                # forecast_features_df is already a DataFrame with correct column names
                forecast_prediction = model.predict(forecast_features_df)
                forecast_temp = float(forecast_prediction[0]) if hasattr(forecast_prediction, '__iter__') else float(forecast_prediction)
                temp_history[i] = forecast_temp
            except Exception as e:
                print(f"Forecast prediction error for hour {forecast_hour}: {str(e)}")
                # Use a fallback temperature if prediction fails
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

