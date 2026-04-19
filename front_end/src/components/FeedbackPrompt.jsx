import React from "react";

const FeedbackPrompt = ({ isOpen, onYes, onNo, onClose }) => {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 px-4">
      <div className="w-full max-w-md rounded-2xl bg-white p-6 shadow-xl">
        <div className="mb-4 flex items-start justify-between gap-4">
          <div>
            <h3 className="text-lg font-semibold text-gray-900">
              Would you like to leave feedback?
            </h3>
            <p className="mt-2 text-sm text-gray-600">
              Your feedback helps us improve the prediction experience.
            </p>
          </div>

          <button
            type="button"
            onClick={onClose}
            className="text-gray-400 hover:text-gray-600"
            aria-label="Close feedback prompt"
          >
            ✕
          </button>
        </div>

        <div className="flex gap-3">
          <button
            type="button"
            onClick={onYes}
            className="rounded-xl bg-blue-600 px-4 py-2 text-white hover:bg-blue-700"
          >
            Yes
          </button>

          <button
            type="button"
            onClick={onNo}
            className="rounded-xl border border-gray-300 px-4 py-2 text-gray-700 hover:bg-gray-50"
          >
            No
          </button>
        </div>
      </div>
    </div>
  );
};

export default FeedbackPrompt;