import React from "react";
import dayImage from "@/assets/day.png";
import nightImage from "@/assets/night.png";
import { cn } from "@/lib/utils";

interface WeatherHeaderProps {
  temperature: number;
  queriedDate: Date;
  queriedHour: number;
}

export const WeatherHeader = ({ temperature, queriedDate, queriedHour }: WeatherHeaderProps) => {
  // Determine day/night mode based on queried hour
  // Day mode: 6:00 AM (6) to 5:59 PM (17)
  // Night mode: 6:00 PM (18) to 5:59 AM (5)
  const isDay = queriedHour >= 6 && queriedHour < 18;
  
  // Debug: Log which image is being used
  console.log(`WeatherHeader: Hour ${queriedHour}, isDay: ${isDay}, using image: ${isDay ? 'day.png' : 'night.png'}`);

  const formatTime = (hour: number) => {
    // Format hour as HH:00 AM/PM
    const date = new Date();
    date.setHours(hour, 0, 0, 0);
    return date.toLocaleTimeString("en-US", {
      hour: "2-digit",
      minute: "2-digit",
      hour12: true,
    });
  };

  const formatDate = (date: Date) => {
    return date.toLocaleDateString("en-US", {
      weekday: "long",
      year: "numeric",
      month: "long",
      day: "numeric",
    });
  };

  return (
    <div className="w-full px-4 pb-8">
      <header
        className="relative w-full max-w-[1200px] mx-auto min-h-[400px] rounded-3xl overflow-hidden transition-all duration-700"
        style={{
          backgroundImage: `url(${isDay ? dayImage : nightImage})`,
          backgroundSize: "cover",
          backgroundPosition: "center",
          backgroundRepeat: "no-repeat",
        }}
      >
        <div 
          className={`absolute inset-0 bg-gradient-to-b ${
            isDay 
              ? "from-black/30 via-black/20 to-black/50" 
              : "from-black/20 via-black/10 to-black/30"
          }`}
        />
        
        <div className="relative z-10 p-8">
          <div className="flex justify-between items-start mb-8">
            <h1
              className={cn(
                "text-xl font-medium transition-colors duration-700 drop-shadow-lg",
                isDay ? "text-white" : "text-white"
              )}
              style={{ textShadow: '2px 2px 4px rgba(0,0,0,0.8)' }}
            >
              Cagayan de Oro City, 9000
            </h1>
            <div className={cn(
              "text-right transition-colors duration-700 drop-shadow-lg",
              isDay ? "text-white" : "text-white"
            )}
            style={{ textShadow: '2px 2px 4px rgba(0,0,0,0.8)' }}
            >
              <p className="text-lg font-semibold">{formatTime(queriedHour)}</p>
              <p className="text-sm opacity-90">
                {formatDate(queriedDate)}
              </p>
            </div>
          </div>

        <div className="flex items-center gap-6 mt-16">
          <span
            className={cn(
              "text-9xl font-bold transition-colors duration-700 drop-shadow-lg",
              isDay ? "text-white" : "text-white"
            )}
            style={{ textShadow: "3px 3px 6px rgba(0,0,0,0.9)" }}
          >
            {Math.round(temperature)}°
          </span>
        </div>

        <div className="mt-6">
          <span className="inline-block bg-weather-temp-label text-white px-4 py-2 rounded-lg text-sm font-semibold shadow-lg">
            Predicted Temperature
          </span>
        </div>
      </div>
      </header>
    </div>
  );
};
