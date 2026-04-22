export const getApiErrorMessage = (error, fallbackMessage) =>
  error?.response?.data?.detail ||
  error?.response?.data?.message ||
  fallbackMessage;