import { useState } from 'react';
import PredictionForm from '../components/PredictionForm';
import ResultDisplay from '../components/ResultDisplay';
import { predictFatPercentage } from '../api/predictApi';

const Home = () => {
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [showResults, setShowResults] = useState(false);

  const handlePrediction = async (formData) => {
    setLoading(true);
    setError(null);
    
    try {
      const result = await predictFatPercentage(formData);
      setPrediction(result);
      setShowResults(true);
      
      // Scroll to results
      setTimeout(() => {
        document.getElementById('results-section')?.scrollIntoView({ 
          behavior: 'smooth' 
        });
      }, 100);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleNewPrediction = () => {
    setShowResults(false);
    setPrediction(null);
    setError(null);
    
    // Scroll back to top
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  return (
    <div className="max-w-6xl mx-auto">
      {/* Hero Section */}
      <div className="text-center mb-8">
        <h1 className="text-4xl font-bold text-gray-900 mb-4">
          🏥 Advanced Body Fat Percentage Predictor
        </h1>
        <p className="text-lg text-gray-600 max-w-3xl mx-auto">
          Get a comprehensive body fat analysis using our advanced machine learning model. 
          Fill in your complete health and fitness profile for the most accurate prediction.
        </p>
      </div>

      {/* Info Box */}
      <div className=" border border-blue-200 rounded-xl p-4 mb-8">
        <div className="flex items-start">
          <div className="flex-shrink-0">
            <svg className="h-5 w-5 text-blue-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          </div>
          <div className="ml-3">
            <p className="text-sm text-blue-700">
              <strong>Note:</strong> All fields are required for accurate prediction. 
              Our model analyzes 20+ metrics including heart rate, nutrition, and training data.
            </p>
          </div>
        </div>
      </div>

      {/* Form Section */}
      <div className="mb-12">
        <div className="card mb-6">
          <div className="flex items-center justify-between mb-6">
            <div>
              <h2 className="text-2xl font-bold text-gray-900">
              Complete Health Profile
              </h2>
              <p className="text-gray-600 mt-1">
                Fill in all sections below for the most accurate prediction
              </p>
            </div>
            <span className="text-sm bg-primary-100 text-primary-700 px-3 py-1 rounded-full font-medium">
              20 metrics • 5 sections
            </span>
          </div>
          <PredictionForm onSubmit={handlePrediction} loading={loading} />
        </div>
      </div>

      {/* Results Section */}
      <div id="results-section" className={`transition-all duration-300 ${showResults ? 'opacity-100' : 'opacity-50'}`}>
        {showResults && (
          <div className="mb-8">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-2xl font-bold text-gray-900">
                📊 Prediction Results
              </h2>
              <button
                onClick={handleNewPrediction}
                className="text-sm bg-gray-100 text-gray-700 px-4 py-2 rounded-lg hover:bg-gray-200 transition-colors flex items-center"
              >
                <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                </svg>
                New Prediction
              </button>
            </div>

            {loading ? (
              <div className="text-center py-12">
                <div className="inline-block animate-spin rounded-full h-16 w-16 border-4 border-primary-600 border-t-transparent"></div>
                <p className="mt-4 text-gray-600 font-medium">Analyzing your comprehensive health data...</p>
                <p className="text-sm text-gray-500 mt-2">
                  Processing 20+ metrics through our machine learning model
                </p>
                <div className="mt-6 max-w-md mx-auto">
                  <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
                    <div className="h-full bg-primary-600 rounded-full animate-pulse" style={{ width: '75%' }}></div>
                  </div>
                  <p className="text-xs text-gray-500 mt-2">Calculating body fat percentage...</p>
                </div>
              </div>
            ) : error ? (
              <div className="bg-red-50 border border-red-200 rounded-xl p-6">
                <div className="flex items-center mb-3">
                  <svg className="h-6 w-6 text-red-600 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  <h3 className="text-lg font-semibold text-red-700">Prediction Error</h3>
                </div>
                <p className="text-red-600 mb-4">{error}</p>
                <div className="flex space-x-3">
                  <button
                    onClick={handleNewPrediction}
                    className="px-4 py-2 bg-red-100 text-red-700 rounded-lg hover:bg-red-200 transition-colors"
                  >
                    Try Again
                  </button>
                  <button
                    onClick={() => setError(null)}
                    className="px-4 py-2 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 transition-colors"
                  >
                    Dismiss
                  </button>
                </div>
              </div>
            ) : prediction ? (
              <ResultDisplay prediction={prediction} onNewPrediction={handleNewPrediction} />
            ) : null}
          </div>
        )}

        {/* Placeholder when no results */}
        {!showResults && !loading && (
          <div className="text-center py-12 border-2 border-dashed border-gray-300 rounded-xl">
            <div className="inline-flex items-center justify-center w-20 h-20 rounded-full bg-gray-100 mb-4">
              <svg className="h-10 w-10 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            </div>
            <h3 className="text-xl font-semibold text-gray-700 mb-2">Results will appear here</h3>
            <p className="text-gray-600 max-w-md mx-auto mb-6">
              Fill out the form above and click "Predict Body Fat Percentage" to see your comprehensive analysis.
            </p>
            <div className="inline-flex items-center text-primary-600 font-medium">
              <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 14l-7 7m0 0l-7-7m7 7V3" />
              </svg>
              Fill the form above to get started
            </div>
          </div>
        )}
      </div>

      {/* Quick Tips Section */}
      <div className="mt-12">
        <div className="card">
          <h3 className="text-xl font-bold text-gray-900 mb-6">
            💡 Tips for Accurate Results
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            <div className="bg-blue-50 p-4 rounded-lg">
              <div className="flex items-center mb-3">
                <div className="bg-blue-100 p-2 rounded-lg mr-3">
                  <svg className="h-5 w-5 text-blue-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
                  </svg>
                </div>
                <h4 className="font-semibold text-blue-800">Accurate Measurements</h4>
              </div>
              <p className="text-sm text-blue-700">
                Use a scale and measuring tape for weight and height. Measure in the morning for consistency.
              </p>
            </div>

            <div className="bg-green-50 p-4 rounded-lg">
              <div className="flex items-center mb-3">
                <div className="bg-green-100 p-2 rounded-lg mr-3">
                  <svg className="h-5 w-5 text-green-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                  </svg>
                </div>
                <h4 className="font-semibold text-green-800">Heart Rate Tips</h4>
              </div>
              <p className="text-sm text-green-700">
                Resting BPM is best measured in the morning before getting up. Use a fitness tracker for accuracy.
              </p>
            </div>

            <div className="bg-purple-50 p-4 rounded-lg">
              <div className="flex items-center mb-3">
                <div className="bg-purple-100 p-2 rounded-lg mr-3">
                  <svg className="h-5 w-5 text-purple-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                  </svg>
                </div>
                <h4 className="font-semibold text-purple-800">Nutrition Tracking</h4>
              </div>
              <p className="text-sm text-purple-700">
                Use apps like MyFitnessPal to track calories and macros accurately for a few days before inputting.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Home;