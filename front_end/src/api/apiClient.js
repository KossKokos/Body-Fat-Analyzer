// front_end/src/api/apiClient.js
import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL // || 'http://localhost:8000';
const API_KEY = import.meta.env.VITE_API_KEY //|| "your-super-secret-api-key-here-change-this-in-production";//import.meta.env.VITE_API_KEY;

// Create axios instance
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 10000, // 10 seconds timeout
  headers: {
    'Content-Type': 'application/json',
    'X-API-Key': API_KEY,  // Automatically add API key to every request
  },
  // withCredentials: true,
});

// Request interceptor
apiClient.interceptors.request.use(
  (config) => {
    // You can add logging here in development
    if (import.meta.env.DEV) {
      console.log(`🚀 API Request: ${config.method.toUpperCase()} ${config.url}`);
    }
    return config;
  },
  (error) => {  
    return Promise.reject(error);
  }
);

// Response interceptor
// apiClient.interceptors.response.use(
//   (response) => {
//     // You can add logging here in development
//     if (import.meta.env.DEV) {
//       console.log(`✅ API Response: ${response.status}`, response.data);
//     }
//     return response;
//   },
//   (error) => {
//     if (error.response) {
//       // The request was made and the server responded with a status code
//       // that falls out of the range of 2xx
//       console.error('API Error Response:', {
//         status: error.response.status,
//         data: error.response.data,
//         headers: error.response.headers,
//       });
  
//     } else if (error.request) {
//       console.error('No response received from server');
//     } else {
//       console.error('Request setup error:', error.message);
//     }

//       return Promise.reject(error);      
//   }
// );
  
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    return Promise.reject(error);
  }
);

export default apiClient;