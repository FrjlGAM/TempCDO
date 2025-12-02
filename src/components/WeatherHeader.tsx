import { useEffect, useState } from "react";
import dayImage from "@/assets/day.png";
import nightImage from "@/assets/night.png";
import { cn } from "@/lib/utils";

export const WeatherHeader = () => {
  const [isDay, setIsDay] = useState(true);
  const [currentTime, setCurrentTime] = useState(new Date());

  useEffect(() => {
    const updateTime = () => {
      const now = new Date();
      setCurrentTime(now);
      
      const hour = now.getHours();
      // Day mode: 6:00 AM (6) to 6:00 PM (18)
      // Night mode: 6:01 PM (18:01) to 5:59 AM (5:59)
      setIsDay(hour >= 6 && hour < 18);
    };

    updateTime();
    const interval = setInterval(updateTime, 1000);

    return () => clearInterval(interval);
  }, []);

  const formatTime = (date: Date) => {
    return date.toLocaleTimeString("en-US", {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
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
        }}
      >
        <div className="absolute inset-0 bg-gradient-to-b from-black/30 via-black/20 to-black/50" />
        
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
              <p className="text-lg font-semibold">{formatTime(currentTime)}</p>
              <p className="text-sm opacity-90">{formatDate(currentTime)}</p>
            </div>
          </div>

        <div className="flex items-center gap-6 mt-16">
          <div className="flex items-baseline">
            <span
              className={cn(
                "text-9xl font-bold transition-colors duration-700 drop-shadow-lg",
                isDay ? "text-white" : "text-white"
              )}
              style={{ textShadow: '3px 3px 6px rgba(0,0,0,0.9)' }}
            >
              26°
            </span>
          </div>
          <CloudIcon className={cn(
            "w-24 h-24 transition-colors duration-700 drop-shadow-lg",
            isDay ? "text-white/80" : "text-white/80"
          )} />
        </div>

        <div className="mt-6">
          <span className="inline-block bg-weather-temp-label text-white px-4 py-2 rounded-lg text-sm font-semibold shadow-lg">
            Current Temperature
          </span>
        </div>
      </div>
      </header>
    </div>
  );
};

const CloudIcon = ({ className }: { className?: string }) => (
  <svg
    className={className}
    viewBox="0 0 24 24"
    fill="none"
    xmlns="http://www.w3.org/2000/svg"
  >
    <path
      d="M18.5 19H7C4.79086 19 3 17.2091 3 15C3 12.7909 4.79086 11 7 11C7 7.68629 9.68629 5 13 5C15.7286 5 18 6.87429 18.5 9.5C20.433 9.5 22 11.067 22 13C22 14.933 20.433 16.5 18.5 16.5"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      fill="currentColor"
      fillOpacity="0.3"
    />
  </svg>
);
