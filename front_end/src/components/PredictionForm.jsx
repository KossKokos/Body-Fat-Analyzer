import { useForm } from 'react-hook-form';
import { yupResolver } from '@hookform/resolvers/yup';
import { predictionSchema, calculateBMI, ACTIVITY_LEVELS } from '../utils/validation';
import { DEFAULT_FORM_VALUES } from '../utils/constants';

const PredictionForm = ({ onSubmit, loading }) => {
  const { register, handleSubmit, watch, formState: { errors } } = useForm({
    resolver: yupResolver(predictionSchema),
    defaultValues: DEFAULT_FORM_VALUES,
  });

  const weight = watch('weight');
  const height = watch('height');
  const bmi = weight && height ? calculateBMI(weight, height) : null;

  return (
    <form onSubmit={handleSubmit(onSubmit)} className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Weight */}
        <div>
          <label className="label">Weight (kg)</label>
          <input
            type="number"
            step="0.1"
            {...register('weight')}
            className="input-field"
            placeholder="e.g., 70.5"
          />
          {errors.weight && (
            <p className="text-red-500 text-sm mt-1">{errors.weight.message}</p>
          )}
        </div>

        {/* Height */}
        <div>
          <label className="label">Height (cm)</label>
          <input
            type="number"
            {...register('height')}
            className="input-field"
            placeholder="e.g., 175"
          />
          {errors.height && (
            <p className="text-red-500 text-sm mt-1">{errors.height.message}</p>
          )}
        </div>

        {/* BMI Display */}
        <div className="md:col-span-2">
          {bmi && (
            <div className="bg-gray-50 p-3 rounded-lg">
              <p className="text-gray-700">
                <span className="font-medium">Calculated BMI:</span> {bmi}
                <span className="text-gray-500 text-sm ml-2">
                  ({bmi < 18.5 ? 'Underweight' : bmi < 25 ? 'Normal' : bmi < 30 ? 'Overweight' : 'Obese'})
                </span>
              </p>
            </div>
          )}
        </div>

        {/* Age */}
        <div>
          <label className="label">Age</label>
          <input
            type="number"
            {...register('age')}
            className="input-field"
            placeholder="e.g., 30"
          />
          {errors.age && (
            <p className="text-red-500 text-sm mt-1">{errors.age.message}</p>
          )}
        </div>

        {/* Gender */}
        <div>
          <label className="label">Gender</label>
          <select {...register('gender')} className="input-field">
            <option value="male">Male</option>
            <option value="female">Female</option>
          </select>
        </div>

        {/* Calories Burned */}
        <div>
          <label className="label">Daily Calories Burned</label>
          <input
            type="number"
            {...register('calories_burned')}
            className="input-field"
            placeholder="e.g., 2000"
          />
          {errors.calories_burned && (
            <p className="text-red-500 text-sm mt-1">{errors.calories_burned.message}</p>
          )}
        </div>

        {/* Calories Eaten */}
        <div>
          <label className="label">Daily Calories Eaten</label>
          <input
            type="number"
            {...register('calories_eaten')}
            className="input-field"
            placeholder="e.g., 2500"
          />
          {errors.calories_eaten && (
            <p className="text-red-500 text-sm mt-1">{errors.calories_eaten.message}</p>
          )}
        </div>

        {/* Activity Level */}
        <div className="md:col-span-2">
          <label className="label">Activity Level</label>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            {ACTIVITY_LEVELS.map((level) => (
              <label key={level.value} className="flex items-center p-3 border rounded-lg cursor-pointer hover:bg-gray-50">
                <input
                  type="radio"
                  value={level.value}
                  {...register('activity_level')}
                  className="h-4 w-4 text-primary-600"
                />
                <span className="ml-2 text-gray-700">{level.label}</span>
              </label>
            ))}
          </div>
          {errors.activity_level && (
            <p className="text-red-500 text-sm mt-1">{errors.activity_level.message}</p>
          )}
        </div>
      </div>

      {/* Submit Button */}
      <div className="pt-4">
        <button
          type="submit"
          disabled={loading}
          className={`btn-primary w-full py-3 text-lg ${loading ? 'opacity-70 cursor-not-allowed' : ''}`}
        >
          {loading ? (
            <span className="flex items-center justify-center">
              <svg className="animate-spin h-5 w-5 mr-2 text-white" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
              </svg>
              Calculating...
            </span>
          ) : (
            'Predict Body Fat Percentage'
          )}
        </button>
      </div>
    </form>
  );
};

export default PredictionForm;