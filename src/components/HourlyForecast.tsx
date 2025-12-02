export const HourlyForecast = () => {
  // Generate 24-hour forecast data (00:00 to 23:00)
  const hourlyData = Array.from({ length: 24 }, (_, i) => {
    const temp = 24 + Math.floor(Math.random() * 8); // Random temps 24-32°C
    return {
      time: `${i.toString().padStart(2, '0')}:00`,
      temp: `${temp}°`
    };
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
