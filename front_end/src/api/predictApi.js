// front_end/src/api/predictApi.js
import apiClient from "./apiClient";

export const predictFatPercentage = async (data) => {
  try {
    console.log(data);
    const response = await apiClient.post("/api/predict", data);
    return response.data;
  } catch (error) {
    throw new Error(
      error.response?.data?.detail ||
      error.response?.data?.message ||
      "Prediction failed"
    );
  }
};

// Example of another API call
export const getModelInfo = async () => {
  try {
    const response = await apiClient.get("/api/model-info");
    return response.data;
  } catch (error) {
    throw new Error(
      error.response?.data?.detail ||
      error.response?.data?.message ||
      "Failed to get model info"
    );
  }
};

// Health check (if you want to verify connection)
export const checkHealth = async () => {
  try {
    const response = await apiClient.get("/api/health");
    return response.data;
  } catch (error) {
    throw new Error(
      error.response?.data?.detail ||
      "Backend connection failed"
    );
  }
};
