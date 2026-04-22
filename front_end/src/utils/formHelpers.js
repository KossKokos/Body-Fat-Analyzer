export const calculateBMI = (weight, height) => {
  return (weight / (height * height)).toFixed(1);
};