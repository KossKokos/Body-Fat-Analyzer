import * as yup from "yup";

export const predictionSchema = yup.object({
  weight: yup
    .number()
    .min(30, "Weight must be at least 30kg")
    .max(200, "Weight must be less than 200kg")
    .required("Weight is required"),
  height: yup
    .number()
    .min(1, "Height must be at least 1m")
    .max(2.5, "Height must be less than 2.5m")
    .required("Height is required"),
  age: yup
    .number()
    .min(15, "Age must be at least 15")
    .max(100, "Age must be less than 100")
    .required("Age is required"),
  calories_burned: yup
    .number()
    .min(500, "Calories burned must be at least 500")
    .max(5000, "Calories burned must be less than 5000")
    .required("Calories burned is required"),
  calories_eaten: yup
    .number()
    .min(500, "Calories eaten must be at least 500")
    .max(5000, "Calories eaten must be less than 5000")
    .required("Calories eaten is required"),
  gender: yup
    .string()
    .oneOf(["male", "female"], "Please select a valid gender")
    .required("Gender is required"),
  activity_level: yup
    .string()
    .oneOf(["sedentary", "moderate", "active"], "Please select activity level")
    .required("Activity level is required"),
});

export const calculateBMI = (weight, height) => {
  // height in cm to meters
  const heightInMeters = height / 100;
  return (weight / (heightInMeters * heightInMeters)).toFixed(1);
};
