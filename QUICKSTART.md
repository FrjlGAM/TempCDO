# Quick Start Guide

## Prerequisites

- **Node.js** (v16 or higher) - [Download](https://nodejs.org/)
- **Python** (3.8 or higher) - [Download](https://www.python.org/downloads/)
- **npm** (comes with Node.js)
- **pip** (comes with Python)

## Step-by-Step Setup

### 1. Install Frontend Dependencies

Open a terminal in the project root directory and run:

```bash
npm install
```

This will install all React/TypeScript dependencies.

### 2. Install Backend Dependencies

```bash
cd backend
pip install -r requirements.txt
cd ..
```

**Note:** If you get permission errors, try:
- Windows: `python -m pip install -r requirements.txt`
- Mac/Linux: `pip3 install -r requirements.txt` or `python3 -m pip install -r requirements.txt`

### 3. Verify Model File

Make sure `temp_forecaster_model.joblib` is in the project root (same level as `backend/` folder).

### 4. Start the Application

You need **TWO terminal windows** - one for the backend and one for the frontend.

#### Terminal 1 - Backend Server

```bash
cd backend
python main.py
```

You should see:
```
Model loaded successfully from ...
FastAPI server started and model loaded successfully
INFO:     Uvicorn running on http://0.0.0.0:8000
```

**Alternative (Windows):**
```bash
cd backend
start.bat
```

**Alternative (Mac/Linux):**
```bash
cd backend
chmod +x start.sh
./start.sh
```

#### Terminal 2 - Frontend Server

```bash
npm run dev
```

You should see:
```
VITE v5.x.x  ready in xxx ms

➜  Local:   http://localhost:8080/
```

### 5. Access the Application

- **Frontend:** Open your browser and go to `http://localhost:8080`
- **Backend API Docs:** `http://localhost:8000/docs`

## Troubleshooting

### Backend Issues

**"Model not found" error:**
- Make sure `temp_forecaster_model.joblib` is in the project root directory
- Check the file path in the error message

**"Module not found" errors:**
- Run `pip install -r requirements.txt` again
- Make sure you're using the correct Python version

**Port 8000 already in use:**
- Change the port in `backend/main.py` (last line): `uvicorn.run(app, host="0.0.0.0", port=8001)`
- Or stop the process using port 8000

### Frontend Issues

**"Cannot find module" errors:**
- Run `npm install` again
- Delete `node_modules` folder and `package-lock.json`, then run `npm install`

**Port 8080 already in use:**
- Vite will automatically use the next available port
- Or change the port in `vite.config.ts`

**TypeScript errors:**
- These are usually false positives if `node_modules` exists
- The app should still run despite TypeScript warnings in the IDE

## Testing the Application

1. Open `http://localhost:8080` in your browser
2. Select a date using the calendar picker
3. Select an hour (00-23)
4. Click "Get Prediction"
5. You should see:
   - Temperature displayed in the large banner
   - The queried date shown in the top-right
   - 24-hour forecast in the hourly strip below

## API Endpoints

Once the backend is running, you can test the API:

**Single Prediction:**
```bash
curl -X POST "http://localhost:8000/api/predict/single" \
  -H "Content-Type: application/json" \
  -d '{"date": "2024-06-15", "time": "12:00:00"}'
```

**Daily Forecast:**
```bash
curl -X POST "http://localhost:8000/api/forecast/daily" \
  -H "Content-Type: application/json" \
  -d '{"date": "2024-06-15"}'
```

Or visit `http://localhost:8000/docs` for interactive API documentation.

## Stopping the Servers

- **Backend:** Press `Ctrl+C` in Terminal 1
- **Frontend:** Press `Ctrl+C` in Terminal 2

## Development Tips

- The backend auto-reloads when you change Python files (if using `--reload`)
- The frontend auto-reloads when you change React/TypeScript files
- Check browser console (F12) for frontend errors
- Check terminal output for backend errors

