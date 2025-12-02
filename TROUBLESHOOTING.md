# Troubleshooting Guide

## "Failed to fetch" or "Cannot connect to backend" Error

This error means the frontend cannot reach the backend server. Follow these steps:

### Step 1: Check if Backend is Running

**Windows:**
```bash
netstat -ano | findstr :8000
```

**Mac/Linux:**
```bash
lsof -i :8000
```

If nothing shows up, the backend is **not running**.

### Step 2: Start the Backend Server

1. Open a **new terminal window**
2. Navigate to the backend folder:
   ```bash
   cd backend
   ```
3. Start the server:
   ```bash
   python main.py
   ```

You should see output like:
```
Model loaded successfully from ...
FastAPI server started and model loaded successfully
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

### Step 3: Verify Backend is Working

Open your browser and go to:
- `http://localhost:8000/docs` - Should show API documentation
- `http://localhost:8000/health` - Should return `{"status":"ok","model_loaded":true}`

### Step 4: Try the Frontend Again

Go back to `http://localhost:8080` and click "Get Prediction" again.

## Common Issues

### Issue: "Module not found" when starting backend

**Solution:**
```bash
cd backend
pip install -r requirements.txt
```

### Issue: "Model file not found"

**Solution:**
- Make sure `temp_forecaster_model.joblib` is in the project root (same level as `backend/` folder)
- Check the file path in the error message

### Issue: Port 8000 already in use

**Solution 1:** Find and stop the process using port 8000

**Windows:**
```bash
netstat -ano | findstr :8000
# Note the PID (last number)
taskkill /PID <PID> /F
```

**Mac/Linux:**
```bash
lsof -ti:8000 | xargs kill -9
```

**Solution 2:** Change the backend port

Edit `backend/main.py`, last line:
```python
uvicorn.run(app, host="0.0.0.0", port=8001)  # Changed to 8001
```

Then update frontend `src/pages/Index.tsx`:
```typescript
const API_BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8001";
```

### Issue: CORS errors in browser console

**Solution:**
- Make sure the frontend URL is in the CORS allowed origins in `backend/main.py`
- Check that you're accessing frontend on `http://localhost:8080` (not `127.0.0.1:8080`)

### Issue: Frontend shows "Loading..." forever

**Possible causes:**
1. Backend is not responding (check backend terminal for errors)
2. Network issue
3. Backend crashed

**Solution:**
- Check backend terminal for error messages
- Restart the backend server
- Check browser console (F12) for detailed errors

## Quick Diagnostic Commands

**Check if backend is running:**
```bash
curl http://localhost:8000/health
```

Should return: `{"status":"ok","model_loaded":true}`

**Test backend API directly:**
```bash
curl -X POST "http://localhost:8000/api/predict/single" \
  -H "Content-Type: application/json" \
  -d "{\"date\":\"2024-06-15\",\"time\":\"12:00:00\"}"
```

## Still Having Issues?

1. Check both terminal windows (backend and frontend) for error messages
2. Open browser Developer Tools (F12) and check the Console tab
3. Check the Network tab to see if requests are being made
4. Verify the backend URL in the frontend code matches where backend is running

