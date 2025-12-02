# Temperature Forecaster Backend API

FastAPI backend service for temperature forecasting in Cagayan de Oro City.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure the model file `temp_forecaster_model.joblib` is in the project root directory.

3. Run the server:
```bash
python main.py
```

Or using uvicorn directly:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

## API Endpoints

### GET `/`
Health check endpoint. Returns API status and model loading status.

### GET `/health`
Same as `/` - health check endpoint.

### POST `/predict`
Get temperature prediction for a specific date and hour.

**Request Body:**
```json
{
  "date": "2024-01-15",
  "hour": 12
}
```

**Response:**
```json
{
  "requested_date": "2024-01-15",
  "requested_hour": 12,
  "requested_temperature": 28.5,
  "hourly_forecast": [
    {
      "hour": 12,
      "time": "12:00",
      "temperature": 28.5
    },
    ...
  ]
}
```

## API Documentation

Once the server is running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

