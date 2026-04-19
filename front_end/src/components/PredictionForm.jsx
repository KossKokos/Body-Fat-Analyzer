import { useForm } from "react-hook-form";
import { yupResolver } from "@hookform/resolvers/yup";
import { predictionSchema, calculateBMI } from "../utils/validation";
import {
  DEFAULT_FORM_VALUES,
  WORKOUT_TYPES,
  DIET_TYPES,
  EXPERIENCE_LEVELS,
} from "../utils/constants";

const PredictionForm = ({ onSubmit, loading }) => {
  const {
    register,
    handleSubmit,
    watch,
    formState: { errors },
  } = useForm({
    resolver: yupResolver(predictionSchema),
    defaultValues: DEFAULT_FORM_VALUES,
  });

  const weight = watch("weight");
  const height = watch("height");
  const calories = watch("calories");
  const caloriesBurned = watch("calories_burned");

  const bmi = weight && height ? calculateBMI(weight, height) : null;
  const calorieBalance =
    calories && caloriesBurned ? calories - caloriesBurned : null;

  return (
    <form onSubmit={handleSubmit(onSubmit)} className="space-y-8">
      {/* Demographics & Body Metrics Section */}
      <div className="card">
        <h3 className="text-xl font-bold text-gray-900 mb-6 border-b pb-3">
          📋 Personal Information
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <div>
            <label className="label">Age (years)</label>
            <input
              type="number"
              {...register("age", {
                setValueAs: (value) => (value === "" ? "" : Number(value)),
              })}
              className="input-field"
              placeholder="e.g., 30"
            />
            {errors.age && (
              <p className="text-red-500 text-xs mt-1">{errors.age.message}</p>
            )}
          </div>

          <div>
            <label className="label">Gender</label>
            <select {...register("gender")} className="input-field">
              <option value="male">Male</option>
              <option value="female">Female</option>
            </select>
          </div>

          <div>
            <label className="label">Weight (kg)</label>
            <input
              type="number"
              step="0.1"
              {...register("weight", {
                setValueAs: (value) => (value === "" ? "" : Number(value)),
              })}
              className="input-field"
              placeholder="e.g., 70.5"
            />
            {errors.weight && (
              <p className="text-red-500 text-xs mt-1">
                {errors.weight.message}
              </p>
            )}
          </div>

          <div>
            <label className="label">Height (m)</label>
            <input
              type="number"
              step="0.01"
              {...register("height", {
                setValueAs: (value) => (value === "" ? "" : Number(value)),
              })}
              className="input-field"
              placeholder="e.g., 1.75"
            />
            {errors.height && (
              <p className="text-red-500 text-xs mt-1">
                {errors.height.message}
              </p>
            )}
          </div>
        </div>

        {/* BMI Display */}
        {bmi && (
          <div className="mt-4 p-3 bg-gray-50 rounded-lg">
            <p className="text-gray-700 text-sm">
              <span className="font-medium">BMI: {bmi}</span>
              <span className="text-gray-500 text-xs ml-2">
                (
                {bmi < 18.5
                  ? "Underweight"
                  : bmi < 25
                    ? "Normal"
                    : bmi < 30
                      ? "Overweight"
                      : "Obese"}
                )
              </span>
            </p>
          </div>
        )}
      </div>

      {/* Heart Rate Metrics */}
      <div className="card">
        <h3 className="text-xl font-bold text-gray-900 mb-6 border-b pb-3">
          ❤️ Heart Rate Metrics
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div>
            <label className="label">Max Heart Rate (BPM)</label>
            <input
              type="number"
              {...register("max_bpm", {
                setValueAs: (value) => (value === "" ? "" : Number(value)),
              })}
              className="input-field"
              placeholder="e.g., 185"
            />
            {errors.max_bpm && (
              <p className="text-red-500 text-xs mt-1">
                {errors.max_bpm.message}
              </p>
            )}
          </div>

          <div>
            <label className="label">Avg Workout BPM</label>
            <input
              type="number"
              {...register("avg_bpm", {
                setValueAs: (value) => (value === "" ? "" : Number(value)),
              })}
              className="input-field"
              placeholder="e.g., 140"
            />
            {errors.avg_bpm && (
              <p className="text-red-500 text-xs mt-1">
                {errors.avg_bpm.message}
              </p>
            )}
          </div>

          <div>
            <label className="label">Resting BPM</label>
            <input
              type="number"
              {...register("resting_bpm", {
                setValueAs: (value) => (value === "" ? "" : Number(value)),
              })}
              className="input-field"
              placeholder="e.g., 65"
            />
            {errors.resting_bpm && (
              <p className="text-red-500 text-xs mt-1">
                {errors.resting_bpm.message}
              </p>
            )}
          </div>
        </div>
      </div>

      {/* Training Section */}
      <div className="card">
        <h3 className="text-xl font-bold text-gray-900 mb-6 border-b pb-3">
          🏋️ Training Information
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          <div>
            <label className="label">Session Duration (hours)</label>
            <input
              type="number"
              step="0.1"
              {...register("session_duration", {
                setValueAs: (value) => (value === "" ? "" : Number(value)),
              })}
              className="input-field"
              placeholder="e.g., 1.5"
            />
            {errors.session_duration && (
              <p className="text-red-500 text-xs mt-1">
                {errors.session_duration.message}
              </p>
            )}
          </div>

          <div>
            <label className="label">Calories Burned</label>
            <input
              type="number"
              {...register("calories_burned", {
                setValueAs: (value) => (value === "" ? "" : Number(value)),
              })}
              className="input-field"
              placeholder="e.g., 500"
            />
            {errors.calories_burned && (
              <p className="text-red-500 text-xs mt-1">
                {errors.calories_burned.message}
              </p>
            )}
          </div>

          <div>
            <label className="label">Workout Type</label>
            <select {...register("workout_type")} className="input-field">
              {WORKOUT_TYPES.map((type) => (
                <option key={type.value} value={type.value}>
                  {type.label}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="label">Workouts per Week</label>
            <input
              type="text"
              inputMode="decimal"
              {...register("workout_frequency")}
              className="input-field"
              placeholder="e.g., 3.5"
            />
            {errors.workout_frequency && (
              <p className="text-red-500 text-xs mt-1">
                {errors.workout_frequency.message}
              </p>
            )}
          </div>

          <div>
            <label className="label">Experience Level</label>
            <select {...register("experience_level")} className="input-field">
              {EXPERIENCE_LEVELS.map((level) => (
                <option key={level.value} value={level.value}>
                  {level.label}
                </option>
              ))}
            </select>
          </div>
        </div>
      </div>

      {/* Nutrition Section */}
      <div className="card">
        <h3 className="text-xl font-bold text-gray-900 mb-6 border-b pb-3">
          🍎 Nutrition Information
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          <div>
            <label className="label">Daily Calories</label>
            <input
              type="number"
              {...register("calories", {
                setValueAs: (value) => (value === "" ? "" : Number(value)),
              })}
              className="input-field"
              placeholder="e.g., 2000"
            />
            {errors.calories && (
              <p className="text-red-500 text-xs mt-1">
                {errors.calories.message}
              </p>
            )}
          </div>

          <div>
            <label className="label">Carbs (g)</label>
            <input
              type="text"
              inputMode="decimal"
              {...register('carbs')}
              className="input-field"
              placeholder="e.g., 250"
            />
            {errors.carbs && (
              <p className="text-red-500 text-xs mt-1">
                {errors.carbs.message}
              </p>
            )}
          </div>

          <div>
            <label className="label">Protein (g)</label>
            <input
              type="text"
              inputMode="decimal"
              {...register('proteins')}
              className="input-field"
              placeholder="e.g., 120"
            />
            {errors.proteins && (
              <p className="text-red-500 text-xs mt-1">
                {errors.proteins.message}
              </p>
            )}
          </div>

          <div>
            <label className="label">Fat (g)</label>
            <input
              type="text"
              inputMode="decimal"
              {...register('fats')}
              className="input-field"
              placeholder="e.g., 70"
            />
            {errors.fats && (
              <p className="text-red-500 text-xs mt-1">{errors.fats.message}</p>
            )}
          </div>

          <div>
            <label className="label">Sugar (g)</label>
            <input
              type="text"
              inputMode="decimal"
              {...register('sugar_g')}
              className="input-field"
              placeholder="e.g., 50"
            />
            {errors.sugar_g && (
              <p className="text-red-500 text-xs mt-1">
                {errors.sugar_g.message}
              </p>
            )}
          </div>

          <div>
            <label className="label">Diet Type</label>
            <select {...register("diet_type")} className="input-field">
              {DIET_TYPES.map((diet) => (
                <option key={diet.value} value={diet.value}>
                  {diet.label}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="label">Meals per Day</label>
            <input
              type="text"
              inputMode="decimal"
              {...register("daily_meals_frequency")}
              className="input-field"
              placeholder="e.g., 3"
            />
            {errors.daily_meals_frequency && (
              <p className="text-red-500 text-xs mt-1">
                {errors.daily_meals_frequency.message}
              </p>
            )}
          </div>

          <div>
            <label className="label">Water (liters)</label>
            <input
              type="text"
              inputMode="decimal"
              {...register("water_intake")}
              className="input-field"
              placeholder="e.g., 2.5"
            />
            {errors.water_intake && (
              <p className="text-red-500 text-xs mt-1">
                {errors.water_intake.message}
              </p>
            )}
          </div>
        </div>

        {/* Calorie Balance Display */}
        {calorieBalance !== null && (
          <div className="mt-6 p-4 rounded-lg bg-gradient-to-r from-blue-50 to-gray-50 border">
            <div className="flex items-center justify-between">
              <div>
                <h4 className="font-semibold text-gray-900">
                  Calorie Analysis
                </h4>
                <p
                  className={`text-sm font-medium ${
                    calorieBalance > 0
                      ? "text-red-600"
                      : calorieBalance < -300
                        ? "text-green-600"
                        : "text-yellow-600"
                  }`}
                >
                  {calorieBalance > 0 ? "Surplus" : "Deficit"}:{" "}
                  {Math.abs(calorieBalance).toFixed(0)} calories
                </p>
              </div>
              <div className="text-right">
                <p className="text-xs text-gray-500">Intake: {calories} cal</p>
                <p className="text-xs text-gray-500">
                  Burned: {caloriesBurned} cal
                </p>
              </div>
            </div>
            <div className="mt-2 h-2 bg-gray-200 rounded-full overflow-hidden">
              <div
                className={`h-full rounded-full ${
                  calorieBalance > 0
                    ? "bg-red-500"
                    : calorieBalance < -300
                      ? "bg-green-500"
                      : "bg-yellow-500"
                }`}
                style={{
                  width: `${Math.min(100, Math.abs(calorieBalance) / 20)}%`,
                  marginLeft: calorieBalance > 0 ? "0" : "auto",
                  marginRight: calorieBalance > 0 ? "auto" : "0",
                }}
              ></div>
            </div>
          </div>
        )}
      </div>

      {/* Submit Section */}
      <div className="sticky bottom-6 z-10">
        <div className="bg-white rounded-xl shadow-2xl border border-gray-200 p-6">
          <div className="flex flex-col md:flex-row items-center justify-between gap-4">
            <div>
              <h4 className="text-lg font-bold text-gray-900">
                Ready for Analysis?
              </h4>
              <p className="text-sm text-gray-600">
                Click below to process all 20+ metrics through our AI model
              </p>
            </div>

            <button
              type="submit"
              disabled={loading}
              className={`btn-primary px-8 py-3 text-lg min-w-[200px] ${loading ? "opacity-70 cursor-not-allowed" : ""}`}
            >
              {loading ? (
                <span className="flex items-center justify-center">
                  <svg
                    className="animate-spin h-6 w-6 mr-3 text-white"
                    fill="none"
                    viewBox="0 0 24 24"
                  >
                    <circle
                      className="opacity-25"
                      cx="12"
                      cy="12"
                      r="10"
                      stroke="currentColor"
                      strokeWidth="4"
                    />
                    <path
                      className="opacity-75"
                      fill="currentColor"
                      d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
                    />
                  </svg>
                  Processing...
                </span>
              ) : (
                <>
                  🔬 Predict Body Fat
                </>
              )}
            </button>
          </div>
        </div>
      </div>
    </form>
  );
};

export default PredictionForm;
