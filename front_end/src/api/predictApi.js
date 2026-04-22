import { postRequest } from "../utils/apiHelpers";

/**
 * Sends prediction input data to the backend and returns the prediction result.
 */
export const predictFatPercentage = async (data) => {
  return postRequest("/api/predict/", data, "Prediction failed");
};