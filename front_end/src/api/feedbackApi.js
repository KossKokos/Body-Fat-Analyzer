import apiClient from "./apiClient";
import { getApiErrorMessage } from "../utils/apiHelpers";

// Sends the user's feedback to the backend feedback endpoint.
// Returns the saved feedback response data.
// If the request fails, converts the backend/API error into a user-friendly Error.
export const createPredictionFeedback = async (data) => {
  try {
    const response = await apiClient.post("/api/feedback/", data);
    return response.data;
  } catch (error) {
    throw new Error(
      getApiErrorMessage(error, "Failed to save feedback")
    );
  }
};