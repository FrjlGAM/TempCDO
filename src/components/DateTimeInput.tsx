import { useState } from "react";
import { Calendar } from "@/components/ui/calendar";
import { Button } from "@/components/ui/button";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Calendar as CalendarIcon } from "lucide-react";
import { format } from "date-fns";
import { cn } from "@/lib/utils";

interface DateTimeInputProps {
  onPredict: (date: Date | undefined, hour: string) => void;
}

export const DateTimeInput = ({ onPredict }: DateTimeInputProps) => {
  const [date, setDate] = useState<Date>();
  const [hour, setHour] = useState<string>("12");

  const hours = Array.from({ length: 24 }, (_, i) => i.toString().padStart(2, "0"));

  const handlePredict = () => {
    onPredict(date, hour);
  };

  return (
    <div className="w-full max-w-4xl mx-auto px-4 py-8">
      <div className="flex flex-col md:flex-row items-center gap-4 mb-6">
        <div className="flex-1 w-full">
          <label className="text-accent text-sm font-medium mb-2 block">
            Select Date
          </label>
          <Popover>
            <PopoverTrigger asChild>
              <Button
                variant="outline"
                className={cn(
                  "w-full justify-start text-left font-normal bg-muted border-border hover:bg-muted/80",
                  !date && "text-muted-foreground"
                )}
              >
                <CalendarIcon className="mr-2 h-4 w-4" />
                {date ? format(date, "PPP") : <span>Pick a date</span>}
              </Button>
            </PopoverTrigger>
            <PopoverContent className="w-auto p-0 bg-card border-border" align="start">
              <Calendar
                mode="single"
                selected={date}
                onSelect={setDate}
                initialFocus
                className="pointer-events-auto"
              />
            </PopoverContent>
          </Popover>
        </div>

        <div className="flex-1 w-full">
          <label className="text-accent text-sm font-medium mb-2 block">
            Select Hour
          </label>
          <Select value={hour} onValueChange={setHour}>
            <SelectTrigger className="w-full bg-muted border-border">
              <SelectValue placeholder="Select hour" />
            </SelectTrigger>
            <SelectContent className="bg-card border-border">
              {hours.map((h) => (
                <SelectItem key={h} value={h} className="hover:bg-muted">
                  {h}:00
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        <div className="flex-1 w-full md:pt-6">
          <Button
            onClick={handlePredict}
            className="w-full bg-primary hover:bg-primary/90 text-primary-foreground font-semibold"
          >
            Get Prediction
          </Button>
        </div>
      </div>

      <p className="text-muted-foreground text-sm italic">
        * Note: Time should be in 24-hour format
      </p>
    </div>
  );
};
