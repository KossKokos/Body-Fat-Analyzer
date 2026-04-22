// front_end/src/api/apiClient.js
import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL;
const API_KEY = import.meta.env.VITE_API_KEY; 

if (!API_BASE_URL) {
  throw new Error("Missing VITE_API_BASE_URL");
}
if (!API_KEY) {
  throw new Error("Missing VITE_API_KEY");
}

// Create axios instance
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 10000, // 10 seconds timeout
  headers: {
    'X-API-Key': API_KEY,
    'Content-Type': 'application/json',
  },
});

// Request interceptor
apiClient.interceptors.request.use(
  (config) => {
    return config;
  },
  (error) => {  
    return Promise.reject(error);
  }
);
  
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    return Promise.reject(error);
  }
);

export default apiClient;