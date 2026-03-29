// import axios from 'axios';

// const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// console.log(API_BASE_URL);

// export const predictFatPercentage = async (data) => {
//   try {
//     const response = await axios.post(
//       `${API_BASE_URL}/api/predict`,
//       data,
//       {
//         headers: {
//           'Content-Type': 'application/json',
//         },
//       }
//     );
//     return response.data;
//   } catch (error) {
//     if (axios.isAxiosError(error)) {
//       throw new Error(error.response?.data?.detail || 'Prediction failed');
//     }
//     throw error;
//   }
// };

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
