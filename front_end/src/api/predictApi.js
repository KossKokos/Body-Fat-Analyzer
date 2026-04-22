// front_end/src/api/predictApi.js
import apiClient from "./apiClient";
import { getApiErrorMessage } from "../utils/apiHelpers";

// Sends prediction form data to the backend prediction endpoint.
// Returns the prediction result payload from the API.
// If the request fails, throws a cleaned error message for UI display.
export const predictFatPercentage = async (data) => {
  try {
    const response = await apiClient.post("/api/predict/", data);
    return response.data;
  } catch (error) {
    throw new Error(
      getApiErrorMessage(error, "Prediction failed")
    );
  }
};