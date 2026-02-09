import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export const predictFatPercentage = async (data) => {
  try {
    const response = await axios.post(
      `${API_BASE_URL}/api/predict`,
      data,
      {
        headers: {
          'Content-Type': 'application/json',
        },
      }
    );
    return response.data;
  } catch (error) {
    if (axios.isAxiosError(error)) {
      throw new Error(error.response?.data?.detail || 'Prediction failed');
    }
    throw error;
  }
};