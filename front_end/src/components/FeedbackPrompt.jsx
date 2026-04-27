const FeedbackPrompt = ({ isOpen, onYes, onNo, onClose }) => {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 px-3 py-4">
      <div className="w-full max-w-sm rounded-2xl bg-white p-4 shadow-xl sm:max-w-md sm:p-6 max-h-[calc(100dvh-2rem)] overflow-y-auto">
        <div className="mb-4 flex items-start justify-between gap-3">
          <div className="min-w-0">
            <h3 className="text-base font-semibold text-gray-900 sm:text-lg">
              Would you like to leave feedback?
            </h3>

            <p className="mt-2 text-sm text-gray-600">
              Your feedback helps us improve future predictions.
            </p>
          </div>

          <button
            type="button"
            onClick={onClose}
            className="flex h-9 w-9 flex-shrink-0 items-center justify-center rounded-full text-gray-400 hover:bg-gray-100 hover:text-gray-600 focus:outline-none focus:ring-2 focus:ring-blue-500"
            aria-label="Close feedback prompt"
          >
            ✕
          </button>
        </div>

        <div className="flex gap-2 sm:gap-3">
          <button
            type="button"
            onClick={onYes}
            className="min-h-11 flex-1 rounded-xl bg-blue-600 px-3 py-2 text-sm font-medium text-white hover:bg-blue-700 sm:px-4 sm:text-base"
          >
            Yes
          </button>

          <button
            type="button"
            onClick={onNo}
            className="min-h-11 flex-1 rounded-xl border border-gray-300 px-3 py-2 text-sm font-medium text-gray-700 hover:bg-gray-50 sm:px-4 sm:text-base"
          >
            No
          </button>
        </div>
      </div>
    </div>
  );
};

export default FeedbackPrompt;