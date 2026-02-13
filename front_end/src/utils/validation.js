import * as yup from 'yup';

export const predictionSchema = yup.object({
  // Demographics
  age: yup
    .number()
    .min(1, 'Age must be at least 1')
    .max(100, 'Age must be less than 100')
    .required('Age is required')
    .integer('Age must be a whole number'),
    
  gender: yup
    .string()
    .oneOf(['male', 'female'], 'Please select a valid gender')
    .required('Gender is required'),

  // Body metrics
  weight: yup
    .number()
    .min(2, 'Weight must be at least 2kg')
    .max(635, 'Weight must be less than 635kg')
    .required('Weight is required'),
    
  height: yup
    .number()
    .min(0.2, 'Height must be at least 0.2m')
    .max(2.72, 'Height must be less than 2.72m')
    .required('Height is required'),

  // Heart rate metrics
  max_bpm: yup
    .number()
    .min(60, 'Max BPM must be at least 60')
    .max(230, 'Max BPM must be less than 230')
    .required('Max BPM is required')
    .integer('Max BPM must be a whole number'),
    
  avg_bpm: yup
    .number()
    .min(40, 'Average BPM must be at least 40')
    .max(200, 'Average BPM must be less than 200')
    .required('Average BPM is required')
    .integer('Average BPM must be a whole number'),
    
  resting_bpm: yup
    .number()
    .min(30, 'Resting BPM must be at least 30')
    .max(120, 'Resting BPM must be less than 120')
    .required('Resting BPM is required')
    .integer('Resting BPM must be a whole number'),

  // Training
  session_duration: yup
    .number()
    .min(0.1, 'Session duration must be at least 0.1 hours')
    .max(3, 'Session duration must be less than 3 hours')
    .required('Session duration is required'),
    
  calories_burned: yup
    .number()
    .min(10, 'Calories burned must be at least 10')
    .max(5000, 'Calories burned must be less than 5000')
    .required('Calories burned is required'),
    
  workout_type: yup
    .string()
    .oneOf(['cardio', 'hiit', 'strength', 'yoga'], 'Please select workout type')
    .required('Workout type is required'),
    
  workout_frequency: yup
    .number()
    .min(0, 'Workout frequency must be at least 0')
    .max(14, 'Workout frequency must be less than 14')
    .required('Workout frequency is required'),
    
  experience_level: yup
    .number()
    .min(1, 'Experience level must be at least 1')
    .max(3, 'Experience level must be at most 3')
    .required('Experience level is required')
    .integer('Experience level must be a whole number'),

  // Nutrition
  calories: yup
    .number()
    .min(500, 'Daily calories must be at least 500')
    .max(10000, 'Daily calories must be less than 10000')
    .required('Daily calories is required'),
    
  carbs: yup
    .number()
    .min(0, 'Carbs must be at least 0g')
    .max(1500, 'Carbs must be less than 1500g')
    .required('Carbs is required'),
    
  proteins: yup
    .number()
    .min(0, 'Proteins must be at least 0g')
    .max(500, 'Proteins must be less than 500g')
    .required('Proteins is required'),
    
  fats: yup
    .number()
    .min(0, 'Fats must be at least 0g')
    .max(500, 'Fats must be less than 500g')
    .required('Fats is required'),
    
  sugar_g: yup
    .number()
    .min(0, 'Sugar must be at least 0g')
    .max(1000, 'Sugar must be less than 1000g')
    .required('Sugar is required'),
    
  diet_type: yup
    .string()
    .oneOf(['vegan', 'vegetarian', 'paleo', 'keto', 'low-carb', 'balanced'], 'Please select diet type')
    .required('Diet type is required'),
    
  daily_meals_frequency: yup
    .number()
    .min(1, 'Meals frequency must be at least 1')
    .max(10, 'Meals frequency must be less than 10')
    .required('Meals frequency is required'),
    
  water_intake: yup
    .number()
    .min(0, 'Water intake must be at least 0L')
    .max(20, 'Water intake must be less than 20L')
    .required('Water intake is required'),
});

export const calculateBMI = (weight, height) => {
  // height in meters
  return (weight / (height * height)).toFixed(1);
};

export const calculateTDEE = (age, gender, weight, height, activityLevel) => {
  // Basic TDEE calculation (Mifflin-St Jeor Equation)
  let bmr;
  if (gender === 'male') {
    bmr = 10 * weight + 6.25 * (height * 100) - 5 * age + 5;
  } else {
    bmr = 10 * weight + 6.25 * (height * 100) - 5 * age - 161;
  }
  
  const activityMultipliers = {
    sedentary: 1.2,
    moderate: 1.55,
    active: 1.9
  };
  
  return Math.round(bmr * (activityMultipliers[activityLevel] || 1.2));
};