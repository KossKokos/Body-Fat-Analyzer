import apiClient from "../api/apiClient";

const getApiErrorMessage = (error, fallbackMessage) =>
  error?.response?.data?.detail ||
  error?.response?.data?.message ||
  fallbackMessage;

/**
 * Sends a POST request and returns response data.
 * Wraps transport-level API error handling in one place.
 */
export const postRequest = async (url, data, fallbackMessage) => {
  try {
    const response = await apiClient.post(url, data);
    return response.data;
  } catch (error) {
    throw new Error(getApiErrorMessage(error, fallbackMessage));
  }
};
