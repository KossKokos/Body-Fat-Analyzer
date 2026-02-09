import { FAT_CLASS_INFO } from '../utils/constants';
import { TrendingUp, TrendingDown, Activity, Heart } from 'lucide-react';

const ResultDisplay = ({ prediction }) => {
  const { fat_class, fat_percentage, confidence, timestamp } = prediction;
  const classInfo = FAT_CLASS_INFO[fat_class];
  
  const getRecommendation = () => {
    switch(fat_class) {
      case 'low':
        return "Maintain your healthy lifestyle with balanced nutrition and regular exercise.";
      case 'mid':
        return "Consider adding more cardio and strength training to your routine.";
      case 'high':
        return "Focus on creating a calorie deficit through diet and increased activity.";
      default:
        return "";
    }
  };

  return (
    <div className="space-y-6">
      {/* Main Result Card */}
      <div className={`${classInfo.bgColor} border-2 ${classInfo.textColor.replace('text', 'border')} rounded-xl p-6`}>
        <div className="flex items-center justify-between mb-4">
          <div>
            <h3 className="text-2xl font-bold">Prediction Result</h3>
            {timestamp && (
              <p className="text-sm opacity-75">
                {new Date(timestamp).toLocaleString()}
              </p>
            )}
          </div>
          <div className={`${classInfo.color} w-12 h-12 rounded-full flex items-center justify-center`}>
            <Activity className="h-6 w-6 text-white" />
          </div>
        </div>
        
        <div className="text-center py-6">
          <div className="text-5xl font-bold mb-2">{fat_percentage.toFixed(1)}%</div>
          <div className={`text-xl font-semibold ${classInfo.textColor}`}>
            {classInfo.label}
          </div>
          <p className="text-gray-600 mt-2">{classInfo.description}</p>
        </div>

        {confidence && (
          <div className="mt-4 pt-4 border-t border-gray-200">
            <p className="text-gray-600">
              <span className="font-medium">Confidence:</span> {(confidence * 100).toFixed(1)}%
            </p>
          </div>
        )}
      </div>

      {/* Recommendation */}
      <div className="bg-white border border-gray-200 rounded-xl p-6">
        <div className="flex items-center mb-4">
          <Heart className="h-5 w-5 text-primary-600 mr-2" />
          <h4 className="text-lg font-semibold">Health Recommendation</h4>
        </div>
        <p className="text-gray-700">{getRecommendation()}</p>
        
        <div className="mt-4 grid grid-cols-2 gap-4">
          <div className="bg-blue-50 p-3 rounded-lg">
            <TrendingDown className="h-5 w-5 text-blue-600 mb-1" />
            <p className="text-sm text-blue-700">
              <span className="font-medium">Calorie Balance:</span> {prediction.calories_burned > prediction.calories_eaten ? 'Deficit' : 'Surplus'}
            </p>
          </div>
          <div className="bg-green-50 p-3 rounded-lg">
            <TrendingUp className="h-5 w-5 text-green-600 mb-1" />
            <p className="text-sm text-green-700">
              <span className="font-medium">Status:</span> {fat_class === 'low' ? 'Optimal' : 'Needs Attention'}
            </p>
          </div>
        </div>
      </div>

      {/* Share/Reset Buttons */}
      <div className="flex space-x-4">
        <button
          onClick={() => window.location.reload()}
          className="btn-secondary flex-1"
        >
          New Prediction
        </button>
        <button
          onClick={() => window.print()}
          className="btn-primary flex-1"
        >
          Save Results
        </button>
      </div>
    </div>
  );
};

export default ResultDisplay;