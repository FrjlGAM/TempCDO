interface PredictionDisplayProps {
  temperature: string | null;
}

export const PredictionDisplay = ({ temperature }: PredictionDisplayProps) => {
  return (
    <div className="w-full max-w-4xl mx-auto px-4 pb-8">
      <div className="bg-weather-card-dark rounded-2xl p-8 border border-border">
        <h2 className="text-accent text-lg font-semibold mb-4">
          Predicted Temperature
        </h2>
        <div className="text-center">
          {temperature ? (
            <p className="text-6xl font-bold text-foreground">{temperature}°</p>
          ) : (
            <p className="text-2xl text-muted-foreground">
              Select a date and time to get prediction
            </p>
          )}
        </div>
      </div>
    </div>
  );
};
