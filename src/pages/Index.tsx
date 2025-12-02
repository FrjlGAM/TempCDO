import { DateTimeInput } from "@/components/DateTimeInput";
import { WeatherHeader } from "@/components/WeatherHeader";
import { HourlyForecast } from "@/components/HourlyForecast";

const Index = () => {
  return (
    <main className="min-h-screen bg-background py-8">
      <div className="container max-w-6xl mx-auto">
        <DateTimeInput onPredict={() => {}} />
        <WeatherHeader />
        <HourlyForecast />
      </div>
    </main>
  );
};

export default Index;
