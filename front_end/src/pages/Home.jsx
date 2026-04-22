import { useEffect, useRef, useState } from "react";
import PredictionForm from "../components/PredictionForm";
import PredictionResultsSection from "../components/PredictionResultsSection";
import { predictFatPercentage } from "../api/predictApi";

const Home = () => {
  const [predictionResult, setPredictionResult] = useState(null);
  const [showFeedbackPrompt, setShowFeedbackPrompt] = useState(false);
  const [showFeedbackForm, setShowFeedbackForm] = useState(false);
  const [selectedStars, setSelectedStars] = useState(0);
  const [feedbackSubmitted, setFeedbackSubmitted] = useState(false);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [showResults, setShowResults] = useState(false);

  const feedbackPromptTimerRef = useRef(null);

  const handlePredictionSuccess = (result) => {
    setPredictionResult(result);
    setShowFeedbackPrompt(false);
    setShowFeedbackForm(false);
    setSelectedStars(0);
    setFeedbackSubmitted(false);

    if (feedbackPromptTimerRef.current) {
      clearTimeout(feedbackPromptTimerRef.current);
    }

    feedbackPromptTimerRef.current = setTimeout(() => {
      setShowFeedbackPrompt(true);
    }, 5000);
  };

  // Cleans up the delayed feedback prompt timer when the component unmounts.
  // Prevents stale timers from firing after navigation or component removal.
  useEffect(() => {
    return () => {
      if (feedbackPromptTimerRef.current) {
        clearTimeout(feedbackPromptTimerRef.current);
      }
    };
  }, []);

  const handleOpenFeedbackFromPrompt = () => {
    setShowFeedbackPrompt(false);
    setShowFeedbackForm(true);
  };

  const handleDismissFeedbackPrompt = () => {
    setShowFeedbackPrompt(false);
  };

  const handleStarSelect = (stars) => {
    if (feedbackSubmitted) return;
    setSelectedStars(stars);
    setShowFeedbackPrompt(false);
    setShowFeedbackForm(true);
  };

  const handleFeedbackSubmitted = () => {
    setFeedbackSubmitted(true);
    setShowFeedbackForm(false);
    setShowFeedbackPrompt(false);
  };

  // Submits the prediction form to the backend.
  // Converts string-based numeric fields into numbers before sending,
  // stores the result on success, shows the results section, and handles API errors.
  const handlePrediction = async (formData) => {
    setLoading(true);
    setError(null);

    // Normalize fields that are stored as strings in the form UI
    // but expected as numeric values by the backend.
    try {
      const payload = {
        ...formData,
        workout_frequency: Number(formData.workout_frequency),
        daily_meals_frequency: Number(formData.daily_meals_frequency),
        water_intake: Number(formData.water_intake),
        carbs: Number(formData.carbs),
        proteins: Number(formData.proteins),
        fats: Number(formData.fats),
        sugar_g: Number(formData.sugar_g),
      };

      const result = await predictFatPercentage(payload);

      // Save the successful prediction and trigger post-prediction UI flow.
      handlePredictionSuccess(result);
      setShowResults(true);

      // Scroll smoothly to the results section after render completes.
      setTimeout(() => {
        document.getElementById("results-section")?.scrollIntoView({
          behavior: "smooth",
        });
      }, 100);
    } catch (err) {
      // Show the error in the results area and clear any stale prediction data.
      setError(err.message);
      setPredictionResult(null);
      setShowResults(true);
    } finally {
      setLoading(false);
    }
  };

  // Resets the prediction and feedback UI so the user can start a fresh prediction.
  // Also clears any pending feedback prompt timer and scrolls back to the top.
  const handleNewPrediction = () => {
    if (feedbackPromptTimerRef.current) {
      clearTimeout(feedbackPromptTimerRef.current);
    }

    setShowResults(false);
    setPredictionResult(null);
    setError(null);
    setShowFeedbackPrompt(false);
    setShowFeedbackForm(false);
    setSelectedStars(0);
    setFeedbackSubmitted(false);

    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  return (
    <div className="max-w-6xl mx-auto">
      <div className="text-center mb-8">
        <h1 className="text-4xl font-bold text-gray-900 mb-4">
          🏥 Body Fat Percentage Predictor Portfolio Project
        </h1>
        <p className="text-lg text-gray-600 max-w-3xl mx-auto">
          This is a portfolio project built to demonstrate full-stack
          development, machine learning integration, and frontend/backend
          engineering. It provides estimated body fat predictions based on
          user-entered data and is intended for demonstration purposes only.
        </p>
      </div>

      <div className="border border-blue-200 rounded-xl p-4 mb-8">
        <div className="flex items-start">
          <div className="flex-shrink-0">
            <svg
              className="h-5 w-5 text-blue-600"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
              />
            </svg>
          </div>
          <div className="ml-3">
            <p className="text-sm text-blue-700">
              <strong>Note:</strong> All fields are required for accurate
              prediction. Our model analyzes 20+ metrics including heart rate,
              nutrition, and training data.
            </p>
          </div>
        </div>
      </div>

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
      <PredictionResultsSection
        showResults={showResults}
        loading={loading}
        error={error}
        predictionResult={predictionResult}
        selectedStars={selectedStars}
        feedbackSubmitted={feedbackSubmitted}
        showFeedbackPrompt={showFeedbackPrompt}
        showFeedbackForm={showFeedbackForm}
        onStarSelect={handleStarSelect}
        onRatingChange={setSelectedStars}
        onNewPrediction={handleNewPrediction}
        onDismissError={() => setError(null)}
        onOpenFeedbackFromPrompt={handleOpenFeedbackFromPrompt}
        onDismissFeedbackPrompt={handleDismissFeedbackPrompt}
        onCloseFeedbackForm={() => setShowFeedbackForm(false)}
        onFeedbackSubmitted={handleFeedbackSubmitted}
      />
    </div>
  );
};

export default Home;
