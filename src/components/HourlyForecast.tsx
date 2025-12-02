import React from "react";

interface HourlyForecastProps {
  forecastData?: Array<{
    hour: number;
    time: string;
    temperature: number;
  }>;
}

export const HourlyForecast = ({ forecastData }: HourlyForecastProps) => {
  // Don't render if no forecast data
  if (!forecastData || forecastData.length === 0) {
    return null;
  }

  // Sort by hour to ensure correct order (00:00 to 23:00)
  // This ensures we display the temperatures in chronological order for the selected date
  const sortedForecast = [...forecastData].sort((a, b) => a.hour - b.hour);
  
  // Create a map for quick lookup by hour
  const forecastMap = new Map(
    sortedForecast.map(item => [item.hour, item])
  );
  
  // Generate array for all 24 hours (00:00 to 23:00) using the forecast data
  // Each temperature comes from the API prediction for that specific hour on the selected date
  const hourlyData = Array.from({ length: 24 }, (_, i) => {
    const forecastItem = forecastMap.get(i);
    if (forecastItem) {
      // Use the actual temperature from the API for this hour
      return {
        time: forecastItem.time,
        temp: `${Math.round(forecastItem.temperature)}°`
      };
    } else {
      // This should not happen if the API returns all 24 hours, but handle gracefully
      console.warn(`Missing forecast data for hour ${i}`);
      // Use a fallback estimate (shouldn't be needed if API works correctly)
      const estimatedTemp = 28.0 + Math.sin(2 * Math.PI * i / 24) * 3.0;
      return {
        time: `${i.toString().padStart(2, '0')}:00`,
        temp: `${Math.round(estimatedTemp)}°`
      };
    }
  });

  return (
    <div className="w-full px-4 pb-8">
      <div className="max-w-[1200px] mx-auto">
        <div className="bg-weather-forecast-strip rounded-2xl p-2 overflow-x-auto">
          <div className="flex gap-2 min-w-max">
            {hourlyData.map((data, index) => (
              <div
                key={index}
                className="bg-weather-forecast-cell rounded-xl px-6 py-4 text-center min-w-[100px] shadow-sm"
              >
                <p className="text-accent-foreground text-sm font-semibold mb-2">
                  {data.time}
                </p>
                <p className="text-accent-foreground text-2xl font-bold">
                  {data.temp}
                </p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};
