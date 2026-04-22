import * as yup from "yup";

const numericStringField = ({
  label,
  min,
  max,
  integer = false,
  forbidOnlyRepeatedZeros = true,
}) => {
  let schema = yup
    .string()
    .required(`${label} is required`)
    .test(
      "valid-number",
      `${label} must be ${integer ? "a whole number" : "a number"}`,
      (value) => {
        if (value === undefined || value === null || value === "") return false;
        return !Number.isNaN(Number(value));
      },
    );

  if (forbidOnlyRepeatedZeros) {
    schema = schema.test(
      "no-repeated-zero-only",
      "Enter 0 only once, not multiple zeros",
      (value) => {
        if (value === undefined || value === null || value === "") return false;
        return !/^0{2,}$/.test(value);
      },
    );
  }

  if (integer) {
    schema = schema.test(
      "integer",
      `${label} must be a whole number`,
      (value) => {
        if (value === undefined || value === null || value === "") return false;
        return Number.isInteger(Number(value));
      },
    );
  }

  schema = schema.test(
    "range",
    `${label} must be between ${min} and ${max}`,
    (value) => {
      if (value === undefined || value === null || value === "") return false;
      const num = Number(value);
      return num >= min && num <= max;
    },
  );

  return schema;
};

const numericField = ({ label, min, max, integer = false }) =>
  yup
    .number()
    .transform((value, originalValue) =>
      originalValue === "" ? undefined : value,
    )
    .typeError(`${label} must be ${integer ? "a whole number" : "a number"}`)
    .min(min, `${label} must be at least ${min}`)
    .max(max, `${label} must be less than or equal to ${max}`)
    .required(`${label} is required`)
    .test(
      "integer",
      `${label} must be a whole number`,
      (value) => !integer || value === undefined || Number.isInteger(value),
    );

export const predictionSchema = yup.object({
  // Demographics
  age: numericField({
    label: "Age",
    min: 1,
    max: 100,
    integer: true,
  }),

  gender: yup
    .string()
    .oneOf(["male", "female"], "Please select a valid gender")
    .required("Gender is required"),

  // Body metrics
  weight: numericField({
    label: "Weight",
    min: 2,
    max: 635,
  }),

  height: numericField({
    label: "Height",
    min: 0.2,
    max: 2.72,
  }),

  // Heart rate metrics
  max_bpm: numericField({
    label: "Max BPM",
    min: 60,
    max: 230,
    integer: true,
  }),

  avg_bpm: numericField({
    label: "Average BPM",
    min: 40,
    max: 200,
    integer: true,
  }),

  resting_bpm: numericField({
    label: "Resting BPM",
    min: 30,
    max: 120,
    integer: true,
  }),

  // Training
  session_duration: numericField({
    label: "Session duration",
    min: 0.1,
    max: 3,
  }),

  calories_burned: numericField({
    label: "Calories burned",
    min: 10,
    max: 5000,
  }),

  workout_type: yup
    .string()
    .oneOf(["cardio", "hiit", "strength", "yoga"], "Please select workout type")
    .required("Workout type is required"),

  workout_frequency: numericStringField({
    label: "Workout frequency",
    min: 0,
    max: 14,
    forbidOnlyRepeatedZeros: true,
  }),

  experience_level: numericField({
    label: "Experience level",
    min: 1,
    max: 3,
    integer: true,
  }),

  // Nutrition
  calories: numericField({
    label: "Daily calories",
    min: 500,
    max: 10000,
  }),

  carbs: numericStringField({
    label: "Carbs",
    min: 0,
    max: 1500,
    forbidOnlyRepeatedZeros: true,
  }),

  proteins: numericStringField({
    label: "Proteins",
    min: 0,
    max: 500,
    forbidOnlyRepeatedZeros: true,
  }),

  fats: numericStringField({
    label: "Fats",
    min: 0,
    max: 500,
    forbidOnlyRepeatedZeros: true,
  }),

  sugar_g: numericStringField({
    label: "Sugar",
    min: 0,
    max: 1000,
    forbidOnlyRepeatedZeros: true,
  }),

  diet_type: yup
    .string()
    .oneOf(
      ["vegan", "vegetarian", "paleo", "keto", "low-carb", "balanced"],
      "Please select diet type",
    )
    .required("Diet type is required"),

  daily_meals_frequency: numericStringField({
    label: "Daily meals frequency",
    min: 1,
    max: 10,
    forbidOnlyRepeatedZeros: true,
  }),

  water_intake: numericStringField({
    label: "Water intake",
    min: 0,
    max: 20,
    forbidOnlyRepeatedZeros: true,
  }),
});


export const feedbackSchema = yup.object({
  rating: yup
    .number()
    .min(0, "Rating must be at least 0")
    .max(10, "Rating must be at most 10")
    .required("Rating is required"),

  is_prediction_close: yup
    .mixed()
    .oneOf([true, false, null], "Please choose yes or no, or leave it blank")
    .nullable(),

  actual_fat_percentage: yup
    .number()
    .transform((value, originalValue) =>
      originalValue === "" ? null : value
    )
    .nullable()
    .min(0, "Actual body fat percentage must be at least 0")
    .max(100, "Actual body fat percentage must be at most 100"),

  comment: yup
    .string()
    .max(2000, "Comment must be 2000 characters or less")
    .nullable(),

  consent_to_retrain: yup
    .boolean()
    .default(false),
});