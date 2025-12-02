# Welcome to your Lovable project

## Project info

**URL**: https://lovable.dev/projects/cc087b4a-b66a-435e-8aa0-4e3559f95f88

## How can I edit this code?

There are several ways of editing your application.

**Use Lovable**

Simply visit the [Lovable Project](https://lovable.dev/projects/cc087b4a-b66a-435e-8aa0-4e3559f95f88) and start prompting.

Changes made via Lovable will be committed automatically to this repo.

**Use your preferred IDE**

If you want to work locally using your own IDE, you can clone this repo and push changes. Pushed changes will also be reflected in Lovable.

The only requirement is having Node.js & npm installed - [install with nvm](https://github.com/nvm-sh/nvm#installing-and-updating)

Follow these steps:

```sh
# Step 1: Clone the repository using the project's Git URL.
git clone <YOUR_GIT_URL>

# Step 2: Navigate to the project directory.
cd <YOUR_PROJECT_NAME>

# Step 3: Install the necessary dependencies.
npm i

# Step 4: Start the development server with auto-reloading and an instant preview.
npm run dev
```

## Backend Setup

The project includes a FastAPI backend service for ML model predictions.

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Setup Steps

1. **Navigate to the backend directory:**
```sh
cd backend
```

2. **Install Python dependencies:**
```sh
pip install -r requirements.txt
```

3. **Ensure the model file is in the project root:**
   - The model file `temp_forecaster_model.joblib` should be in the project root directory (same level as `backend/` folder)

4. **Start the backend server:**
```sh
python main.py
```

Or on Windows:
```sh
start.bat
```

Or on Linux/Mac:
```sh
chmod +x start.sh
./start.sh
```

The API will be available at `http://localhost:8000`

### API Documentation

Once the server is running, you can access:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### API Endpoints

- `GET /` or `GET /health` - Health check endpoint
- `POST /predict` - Get temperature prediction
  - Request body: `{ "date": "2024-01-15", "hour": 12 }`
  - Returns: Prediction data with requested temperature and 24-hour forecast

### Running Both Frontend and Backend

1. **Terminal 1 - Start Backend:**
```sh
cd backend
python main.py
```

2. **Terminal 2 - Start Frontend:**
```sh
npm run dev
```

The frontend will connect to the backend API automatically.

**Edit a file directly in GitHub**

- Navigate to the desired file(s).
- Click the "Edit" button (pencil icon) at the top right of the file view.
- Make your changes and commit the changes.

**Use GitHub Codespaces**

- Navigate to the main page of your repository.
- Click on the "Code" button (green button) near the top right.
- Select the "Codespaces" tab.
- Click on "New codespace" to launch a new Codespace environment.
- Edit files directly within the Codespace and commit and push your changes once you're done.

## What technologies are used for this project?

This project is built with:

**Frontend:**
- Vite
- TypeScript
- React
- shadcn-ui
- Tailwind CSS

**Backend:**
- FastAPI
- Python
- scikit-learn (via joblib for model loading)
- NumPy

## How can I deploy this project?

Simply open [Lovable](https://lovable.dev/projects/cc087b4a-b66a-435e-8aa0-4e3559f95f88) and click on Share -> Publish.

## Can I connect a custom domain to my Lovable project?

Yes, you can!

To connect a domain, navigate to Project > Settings > Domains and click Connect Domain.

Read more here: [Setting up a custom domain](https://docs.lovable.dev/features/custom-domain#custom-domain)
