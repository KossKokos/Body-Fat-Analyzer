export const ACTIVITY_LEVELS = [
  { value: 'sedentary', label: 'Sedentary (little or no exercise)' },
  { value: 'moderate', label: 'Moderate (exercise 3-4 times/week)' },
  { value: 'active', label: 'Active (exercise 5+ times/week)' },
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
  weight: '',
  height: '',
  age: '',
  calories_burned: '',
  calories_eaten: '',
  gender: 'male',
  activity_level: 'moderate',
};