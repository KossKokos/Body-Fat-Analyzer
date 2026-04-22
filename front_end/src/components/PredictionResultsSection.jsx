import ResultDisplay from "./ResultDisplay";
import FeedbackPrompt from "./FeedbackPrompt";
import FeedbackFormModal from "./FeedbackFormModal";

const PredictionResultsSection = ({
  showResults,
  loading,
  error,
  predictionResult,
  selectedStars,
  feedbackSubmitted,
  showFeedbackPrompt,
  showFeedbackForm,
  onStarSelect,
  onRatingChange,
  onNewPrediction,
  onDismissError,
  onOpenFeedbackFromPrompt,
  onDismissFeedbackPrompt,
  onCloseFeedbackForm,
  onFeedbackSubmitted,
}) => {
  return (
    <div
      id="results-section"
      className={`transition-all duration-300 ${
        showResults ? "opacity-100" : "opacity-50"
      }`}
    >
      {showResults && (
        <div className="mb-8">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-2xl font-bold text-gray-900">
              📊 Prediction Results
            </h2>
            <button
              onClick={onNewPrediction}
              className="text-sm bg-gray-100 text-gray-700 px-4 py-2 rounded-lg hover:bg-gray-200 transition-colors flex items-center"
            >
              <svg
                className="w-4 h-4 mr-2"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"
                />
              </svg>
              New Prediction
            </button>
          </div>

          {loading ? (
            <div className="text-center py-12">
              <div className="inline-block animate-spin rounded-full h-16 w-16 border-4 border-primary-600 border-t-transparent"></div>
              <p className="mt-4 text-gray-600 font-medium">
                Analyzing your comprehensive health data...
              </p>
              <p className="text-sm text-gray-500 mt-2">
                Processing 20+ metrics through our machine learning model
              </p>
            </div>
          ) : error ? (
            <div className="bg-red-50 border border-red-200 rounded-xl p-6">
              <div className="flex items-center mb-3">
                <svg
                  className="h-6 w-6 text-red-600 mr-2"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                  />
                </svg>
                <h3 className="text-lg font-semibold text-red-700">
                  Prediction Error
                </h3>
              </div>
              <p className="text-red-600 mb-4">{error}</p>
              <div className="flex space-x-3">
                <button
                  onClick={onNewPrediction}
                  className="px-4 py-2 bg-red-100 text-red-700 rounded-lg hover:bg-red-200 transition-colors"
                >
                  Try Again
                </button>
                <button
                  onClick={onDismissError}
                  className="px-4 py-2 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 transition-colors"
                >
                  Dismiss
                </button>
              </div>
            </div>
          ) : predictionResult ? (
            <>
              {feedbackSubmitted && (
                <div className="mb-4 rounded-xl border border-green-200 bg-green-50 px-4 py-3">
                  <div className="flex items-start gap-3">
                    <div className="mt-0.5 text-green-600">✓</div>
                    <div>
                      <p className="font-medium text-green-800">
                        Feedback submitted successfully
                      </p>
                      <p className="text-sm text-green-700">
                        Thank you. Your feedback has been saved and will help
                        improve future predictions.
                      </p>
                    </div>
                  </div>
                </div>
              )}
              <ResultDisplay
                prediction={predictionResult}
                selectedStars={selectedStars}
                onStarSelect={onStarSelect}
                feedbackSubmitted={feedbackSubmitted}
                onNewPrediction={onNewPrediction}
              />

              <FeedbackPrompt
                isOpen={showFeedbackPrompt && !feedbackSubmitted}
                onYes={onOpenFeedbackFromPrompt}
                onNo={onDismissFeedbackPrompt}
                onClose={onDismissFeedbackPrompt}
              />

              <FeedbackFormModal
                isOpen={showFeedbackForm && !feedbackSubmitted}
                predictionId={predictionResult.prediction_id}
                selectedStars={selectedStars}
                onRatingChange={onRatingChange}
                onClose={onCloseFeedbackForm}
                onSubmitted={onFeedbackSubmitted}
              />
            </>
          ) : null}
        </div>
      )}

      {!showResults && !loading && (
        <div className="text-center py-12 border-2 border-dashed border-gray-300 rounded-xl">
          <div className="inline-flex items-center justify-center w-20 h-20 rounded-full bg-gray-100 mb-4">
            <svg
              className="h-10 w-10 text-gray-400"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
              />
            </svg>
          </div>
          <h3 className="text-xl font-semibold text-gray-700 mb-2">
            Results will appear here
          </h3>
          <p className="text-gray-600 max-w-md mx-auto mb-6">
            Fill out the form above and click "Predict Body Fat Percentage" to
            see your comprehensive analysis.
          </p>
        </div>
      )}
    </div>
  );
};

export default PredictionResultsSection;
