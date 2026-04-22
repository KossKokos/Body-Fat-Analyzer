export const WORKOUT_TYPES = [
  { value: 'cardio', label: 'Cardio' },
  { value: 'hiit', label: 'HIIT' },
  { value: 'strength', label: 'Strength Training' },
  { value: 'yoga', label: 'Yoga/Pilates' },
];

export const DIET_TYPES = [
  { value: 'vegan', label: 'Vegan' },
  { value: 'vegetarian', label: 'Vegetarian' },
  { value: 'paleo', label: 'Paleo' },
  { value: 'keto', label: 'Keto' },
  { value: 'low-carb', label: 'Low-Carb' },
  { value: 'balanced', label: 'Balanced' },
];

export const EXPERIENCE_LEVELS = [
  { value: 1, label: 'Beginner' },
  { value: 2, label: 'Intermediate' },
  { value: 3, label: 'Advanced' },
];

export const FAT_CLASS_INFO = {
  low: {
    label: 'Low Fat',
    color: 'bg-green-500',
    textColor: 'text-green-700',
    bgColor: 'bg-green-50',
    description: 'Healthy fat percentage range'
  },
  mid: {
    label: 'Moderate Fat',
    color: 'bg-yellow-500',
    textColor: 'text-yellow-700',
    bgColor: 'bg-yellow-50',
    description: 'Average fat percentage range'
  },
  high: {
    label: 'High Fat',
    color: 'bg-red-500',
    textColor: 'text-red-700',
    bgColor: 'bg-red-50',
    description: 'Above average fat percentage'
  }
};

export const DEFAULT_FORM_VALUES = {
  // Demographics
  age: '',
  gender: 'male',
  
  // Body metrics
  weight: '',
  height: '',
  
  // Heart rate metrics
  max_bpm: '',
  avg_bpm: '',
  resting_bpm: '',
  
  // Training
  session_duration: '',
  calories_burned: '',
  workout_type: 'cardio',
  workout_frequency: '',
  experience_level: 1,
  
  // Nutrition
  calories: '',
  carbs: '',
  proteins: '',
  fats: '',
  sugar_g: '',
  diet_type: 'balanced',
  daily_meals_frequency: '',
  water_intake: '',
};

export const DEFAULT_FEEDBACK_VALUES = {
  rating: 0,
  is_prediction_close: null,
  actual_fat_percentage: "",
  comment: "",
  consent_to_retrain: false,
};

export const STAR_RATING_TO_API_RATING = {
  1: 2,
  2: 4,
  3: 6,
  4: 8,
  5: 10,
};