import apiClient from "./apiClient";

export const createPredictionFeedback = async (data) => {
  try {
    const response = await apiClient.post("/api/feedback/", data);
    return response.data;
  } catch (error) {
    throw new Error(
      error.response?.data?.detail ||
      error.response?.data?.message ||
      "Failed to save feedback"
    );
  }
};