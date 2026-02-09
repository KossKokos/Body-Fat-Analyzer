import { useState } from 'react';
import PredictionForm from '../components/PredictionForm';
import ResultDisplay from '../components/ResultDisplay';
import { predictFatPercentage } from '../api/predictApi';
import toast from 'react-hot-toast';

const Home = () => {
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handlePrediction = async (formData) => {
    setLoading(true);
    setError(null);
    
    try {
      const result = await predictFatPercentage(formData);
      setPrediction(result);
      toast.success('Prediction completed successfully!');
    } catch (err) {
      setError(err.message);
      toast.error('Prediction failed: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-6xl mx-auto">
      {/* Hero Section */}
      <div className="text-center mb-12">
        <h1 className="text-4xl font-bold text-gray-900 mb-4">
          AI Body Fat Percentage Predictor
        </h1>
        <p className="text-xl text-gray-600 max-w-3xl mx-auto">
          Get an accurate estimate of your body fat percentage using machine learning.
          Input your details below to get started.
        </p>
      </div>

      {/* Main Content */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Form Section */}
        <div className="card">
          <h2 className="text-2xl font-bold text-gray-900 mb-6">
            Enter Your Details
          </h2>
          <PredictionForm onSubmit={handlePrediction} loading={loading} />
        </div>

        {/* Results Section */}
        <div className="card">
          <h2 className="text-2xl font-bold text-gray-900 mb-6">
            Prediction Results
          </h2>
          {loading ? (
            <div className="text-center py-12">
              <div className="inline-block animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600"></div>
              <p className="mt-4 text-gray-600">Analyzing your data...</p>
            </div>
          ) : error ? (
            <div className="bg-red-50 border border-red-200 rounded-lg p-4">
              <p className="text-red-700 font-medium">Error: {error}</p>
              <p className="text-red-600 text-sm mt-2">
                Please check your inputs and try again.
              </p>
            </div>
          ) : prediction ? (
            <ResultDisplay prediction={prediction} />
          ) : (
            <div className="text-center py-12 text-gray-500">
              <p className="text-lg">Fill out the form to get your prediction</p>
              <p className="text-sm mt-2">Results will appear here</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Home;