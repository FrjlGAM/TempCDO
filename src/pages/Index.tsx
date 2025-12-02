import { useState } from "react";
import { DateTimeInput } from "@/components/DateTimeInput";
import { WeatherHeader } from "@/components/WeatherHeader";
import { HourlyForecast } from "@/components/HourlyForecast";

interface SinglePrediction {
  date: string;
  time: string;
  temperature: number;
}

interface HourlyForecastItem {
  hour: number;
  time: string;
  temperature: number;
}

interface DailyForecast {
  date: string;
  forecast: HourlyForecastItem[];
}

const API_BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

const Index = () => {
  const [singlePrediction, setSinglePrediction] = useState<SinglePrediction | null>(null);
  const [dailyForecast, setDailyForecast] = useState<DailyForecast | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [queriedDate, setQueriedDate] = useState<Date | null>(null);
  const [queriedHour, setQueriedHour] = useState<number | null>(null);

  const handlePredict = async (date: Date | undefined, hour: string) => {
    if (!date) {
      setError("Please select a date");
      return;
    }

    setLoading(true);
    setError(null);
    setQueriedDate(date);
    setQueriedHour(parseInt(hour, 10));

    try {
      // Format date as YYYY-MM-DD
      const dateStr = date.toISOString().split('T')[0];
      // Format time as HH:00:00
      const hourNum = parseInt(hour, 10);
      const timeStr = `${hourNum.toString().padStart(2, '0')}:00:00`;

      // Call both APIs in parallel
      const [singleResponse, dailyResponse] = await Promise.all([
        // API Call 1: Single Prediction
        fetch(`${API_BASE_URL}/api/predict/single`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            date: dateStr,
            time: timeStr,
          }),
        }),
        // API Call 2: Daily Forecast
        fetch(`${API_BASE_URL}/api/forecast/daily`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            date: dateStr,
          }),
        }),
      ]);

      // Parse responses and check for errors
      let singleData: SinglePrediction;
      let dailyData: DailyForecast;

      if (!singleResponse.ok) {
        const errorData = await singleResponse.json().catch(() => ({ detail: "Unknown error" }));
        throw new Error(`Single prediction failed: ${errorData.detail || `HTTP ${singleResponse.status}`}`);
      } else {
        singleData = await singleResponse.json();
      }

      if (!dailyResponse.ok) {
        const errorData = await dailyResponse.json().catch(() => ({ detail: "Unknown error" }));
        throw new Error(`Daily forecast failed: ${errorData.detail || `HTTP ${dailyResponse.status}`}`);
      } else {
        dailyData = await dailyResponse.json();
      }

      // Update state with results
      // dailyData contains the 24-hour forecast for the selected date
      setSinglePrediction(singleData);
      setDailyForecast(dailyData);
      
      // Log for debugging (can be removed in production)
      console.log("Daily forecast for date:", dateStr, dailyData);
    } catch (err) {
      let errorMessage = "Failed to get prediction";
      
      if (err instanceof TypeError && err.message.includes("fetch")) {
        errorMessage = `Cannot connect to backend server at ${API_BASE_URL}. Please make sure the backend is running on port 8000.`;
      } else if (err instanceof Error) {
        errorMessage = err.message;
      }
      
      setError(errorMessage);
      console.error("Prediction error:", err);
      console.error("API Base URL:", API_BASE_URL);
      // Clear previous data on error
      setSinglePrediction(null);
      setDailyForecast(null);
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="min-h-screen bg-background py-8">
      <div className="container max-w-6xl mx-auto">
        <DateTimeInput onPredict={handlePredict} loading={loading} />
        {error && (
          <div className="mx-4 mb-4 p-4 bg-red-500/10 border border-red-500/50 rounded-lg text-red-500">
            <p className="font-semibold mb-2">Error: {error}</p>
            {error.includes("Cannot connect to backend") && (
              <div className="mt-2 text-sm">
                <p className="font-medium mb-1">To fix this:</p>
                <ol className="list-decimal list-inside space-y-1 ml-2">
                  <li>Open a terminal and navigate to the <code className="bg-red-500/20 px-1 rounded">backend</code> folder</li>
                  <li>Run: <code className="bg-red-500/20 px-1 rounded">python main.py</code></li>
                  <li>Wait for "Uvicorn running on http://0.0.0.0:8000"</li>
                  <li>Then try again</li>
                </ol>
                <p className="mt-2 text-xs opacity-75">Backend URL: {API_BASE_URL}</p>
              </div>
            )}
          </div>
        )}
        {loading && (
          <div className="mx-4 mb-4 p-4 bg-blue-500/10 border border-blue-500/50 rounded-lg text-blue-500 text-center">
            Loading prediction...
          </div>
        )}
        {singlePrediction && queriedDate && queriedHour !== null && (
          <>
            <WeatherHeader 
              temperature={singlePrediction.temperature} 
              queriedDate={queriedDate}
              queriedHour={queriedHour}
            />
            <HourlyForecast forecastData={dailyForecast?.forecast} />
          </>
        )}
      </div>
    </main>
  );
};

export default Index;
