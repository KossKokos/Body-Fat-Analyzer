import { FAT_CLASS_INFO } from "../utils/constants";
import { Activity } from "lucide-react";

// Displays the clickable star rating shown next to the prediction result.
// Can be disabled after feedback is submitted to prevent repeated interaction.
const StarRating = ({ value = 0, onSelect, disabled = false }) => {
  return (
    <div className="flex items-center gap-1">
      {[1, 2, 3, 4, 5].map((star) => (
        <button
          key={star}
          type="button"
          disabled={disabled}
          onClick={() => onSelect?.(star)}
          className={`text-2xl transition ${
            star <= value ? "text-yellow-400" : "text-gray-300"
          } ${disabled ? "cursor-default" : "hover:scale-110"}`}
          aria-label={`Rate ${star} out of 5`}
        >
          ★
        </button>
      ))}
    </div>
  );
};

const ResultDisplay = ({
  prediction,
  selectedStars,
  onStarSelect,
  feedbackSubmitted,
  onNewPrediction,
}) => {
  const { fat_class, fat_percentage, timestamp } = prediction;
  const classInfo = FAT_CLASS_INFO[fat_class];

  return (
    <div className="space-y-6">
      <div
        className={`${classInfo.bgColor} border-2 ${classInfo.textColor.replace("text", "border")} rounded-xl p-6`}
      >
        <div className="flex flex-col md:flex-row md:items-center justify-between mb-4">
          <div>
            <h3 className="text-2xl font-bold">Comprehensive Analysis</h3>
            {timestamp && (
              <p className="text-sm opacity-75">
                Generated: {new Date(timestamp).toLocaleString()}
              </p>
            )}
          </div>

          <div className="flex items-center mt-2 md:mt-0">
            <div
              className={`${classInfo.color} w-10 h-10 rounded-full flex items-center justify-center mr-3`}
            >
              <Activity className="h-5 w-5 text-white" />
            </div>
            <div>
              <div className={`text-lg font-semibold ${classInfo.textColor}`}>
                {classInfo.label}
              </div>
            </div>
          </div>
        </div>

        <div className="text-center py-6">
          <div className="text-5xl font-bold mb-2">
            {Number(fat_percentage).toFixed(1)}%
          </div>
          <p className="text-gray-600">Body Fat Percentage</p>
          <p className="text-sm text-gray-500 mt-2">{classInfo.description}</p>
        </div>

        <div className="mt-4 border-t border-black/5 pt-4">
          <p className="mb-2 text-sm text-gray-600">
            {feedbackSubmitted
              ? "Thanks for your feedback."
              : "Want to rate this prediction?"}
          </p>
          <StarRating
            value={selectedStars}
            onSelect={onStarSelect}
            disabled={feedbackSubmitted}
          />
        </div>
      </div>

      <div className="flex flex-col sm:flex-row gap-4 pt-6 border-t">
        <button onClick={onNewPrediction} className="btn-secondary flex-1 py-3">
          ↻ New Analysis
        </button>

        <button
          onClick={() => {
            const dataStr = JSON.stringify(prediction, null, 2);
            const blob = new Blob([dataStr], { type: "application/json" });
            const url = URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            a.download = `fat-prediction-${new Date().toISOString().split("T")[0]}.json`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
          }}
          className="btn-primary flex-1 py-3"
        >
          💾 Download Results
        </button>

        <button
          onClick={() => window.print()}
          className="bg-gray-100 text-gray-800 px-6 py-3 rounded-lg hover:bg-gray-200 transition-colors font-medium flex-1"
        >
          🖨️ Print Report
        </button>
      </div>
    </div>
  );
};

export default ResultDisplay;