import { useMemo, useState } from "react";
import { createPredictionFeedback } from "../api/feedbackApi";
import {
  DEFAULT_FEEDBACK_VALUES,
  STAR_RATING_TO_API_RATING,
} from "../utils/constants";

// Reusable star rating input for the feedback modal.
// Displays 5 clickable stars and calls onChange with the selected value.
// Uses the current value to visually highlight the selected rating.
const StarRatingInput = ({ value = 0, onChange }) => {
  return (
    <div className="flex items-center gap-1">
      {[1, 2, 3, 4, 5].map((star) => (
        <button
          key={star}
          type="button"
          onClick={() => onChange?.(star)}
          className={`text-2xl transition sm:text-3xl ${
            star <= value ? "text-yellow-400" : "text-gray-300"
          } hover:scale-110`}
          aria-label={`Select ${star} out of 5 stars`}
        >
          ★
        </button>
      ))}
    </div>
  );
};

const sanitizeDecimalInput = (value) => {
  const cleanedValue = value.replace(/[^\d.]/g, "");
  const [integerPart, ...decimalParts] = cleanedValue.split(".");

  return decimalParts.length > 0
    ? `${integerPart}.${decimalParts.join("")}`
    : integerPart;
};

const FeedbackChoiceButton = ({ selected, onClick, children }) => {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`min-h-10 flex-1 rounded-xl border px-3 py-2 text-sm font-medium transition-colors ${
        selected
          ? "border-blue-600 bg-blue-50 text-blue-700"
          : "border-gray-300 text-gray-700 hover:bg-gray-50"
      }`}
    >
      {children}
    </button>
  );
};

const FeedbackFormModal = ({
  isOpen,
  onClose,
  predictionId,
  selectedStars,
  onRatingChange,
  onSubmitted,
}) => {
  const [form, setForm] = useState(DEFAULT_FEEDBACK_VALUES);
  const [errors, setErrors] = useState({});
  const [isSubmitting, setIsSubmitting] = useState(false);

  // Converts the UI star rating (1-5) into the backend rating scale (0-10).
  // Memoized so the value only recalculates when selectedStars changes.
  const apiRating = useMemo(
    () => STAR_RATING_TO_API_RATING[selectedStars] ?? 0,
    [selectedStars],
  );

  // Updates a single feedback form field.
  // Also clears the field-specific error and any general submission error
  // so the UI reacts immediately when the user edits the form.
  const handleChange = (field, value) => {
    setForm((prev) => ({ ...prev, [field]: value }));
    setErrors((prev) => ({ ...prev, [field]: undefined, general: undefined }));
  };

  const handleRatingChange = (star) => {
    onRatingChange?.(star);
    setErrors((prev) => ({
      ...prev,
      rating: undefined,
      general: undefined,
    }));
  };

  const handleActualFatPercentageChange = (e) => {
    handleChange("actual_fat_percentage", sanitizeDecimalInput(e.target.value));
  };

  // Validates the feedback form before submission.
  // Builds an errors object for missing/invalid values and stores it in state.
  // Returns true when the form is valid, otherwise false.
  const validate = () => {
    const nextErrors = {};

    // Feedback must always be linked to a valid prediction record.
    if (!predictionId) {
      nextErrors.general = "Missing prediction reference.";
    }

    // A star rating is required before feedback can be submitted.
    if (!apiRating) {
      nextErrors.rating = "Please choose a star rating first.";
    }

    // If the user allows their feedback to be used for improving predictions,
    // require their actual body fat percentage as supporting data.
    if (form.consent_to_retrain && form.actual_fat_percentage === "") {
      nextErrors.actual_fat_percentage =
        "Please enter your actual body fat percentage if you allow us to use this feedback to improve future predictions.";
    }

    // Validate actual fat percentage range when the field is filled in.
    if (
      form.actual_fat_percentage !== "" &&
      (Number(form.actual_fat_percentage) < 0 ||
        Number(form.actual_fat_percentage) > 100)
    ) {
      nextErrors.actual_fat_percentage =
        "Actual body fat percentage must be between 0 and 100.";
    }

    // Limit comment length to match backend validation and keep payload reasonable.
    if (form.comment && form.comment.length > 2000) {
      nextErrors.comment = "Comment must be 2000 characters or less.";
    }

    setErrors(nextErrors);
    return Object.keys(nextErrors).length === 0;
  };

  // Resets the feedback form back to its default state and closes the modal.
  // Used when the user cancels or closes the feedback form without submitting.
  const resetAndClose = () => {
    setForm(DEFAULT_FEEDBACK_VALUES);
    setErrors({});
    onClose?.();
  };

  // Handles feedback form submission.
  // Prevents the default form refresh, validates input, sends the payload to the API,
  // resets local form state on success, and stores a general error on failure.
  const handleSubmit = async (e) => {
    e.preventDefault();

    // Stop submission immediately if the form is invalid.
    if (!validate()) return;

    setIsSubmitting(true);

    try {
      await createPredictionFeedback({
        prediction_id: predictionId,
        rating: apiRating,
        is_prediction_close: form.is_prediction_close,
        actual_fat_percentage:
          form.actual_fat_percentage === ""
            ? null
            : Number(form.actual_fat_percentage),
        comment: form.comment?.trim() || null,
        consent_to_retrain: form.consent_to_retrain,
      });

      // Clear the form after a successful submission.
      setForm(DEFAULT_FEEDBACK_VALUES);
      setErrors({});
      onSubmitted?.();
    } catch (error) {
      // Store a user-friendly submission error for display in the modal.
      setErrors({
        general: error.message || "Failed to submit feedback",
      });
    } finally {
      setIsSubmitting(false);
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-end justify-center bg-black/40 p-4 sm:items-center">
      <div className="flex max-h-[90dvh] w-full max-w-sm flex-col rounded-2xl bg-white shadow-2xl sm:max-w-lg">
        {/* Fixed header */}
        <div className="flex flex-shrink-0 items-start justify-between gap-3 border-b border-gray-200 px-4 py-3 sm:px-6 sm:py-4">
          <div>
            <h3 className="text-base font-semibold text-gray-900 sm:text-lg">
              Share your feedback
            </h3>

            <p className="mt-1 text-xs text-gray-600 sm:text-sm">
              Selected rating:{" "}
              <span className="font-medium">
                {selectedStars > 0 ? `${selectedStars}/5` : "Not selected yet"}
              </span>
            </p>
          </div>

          <button
            type="button"
            onClick={resetAndClose}
            className="flex h-9 w-9 flex-shrink-0 items-center justify-center rounded-full text-gray-500 hover:bg-gray-100 hover:text-gray-700 focus:outline-none focus:ring-2 focus:ring-blue-500 sm:h-10 sm:w-10"
            aria-label="Close feedback form"
          >
            ✕
          </button>
        </div>

        <form onSubmit={handleSubmit} className="flex min-h-0 flex-1 flex-col">
          {/* Scrollable body */}
          <div className="min-h-0 flex-1 space-y-4 overflow-y-auto px-4 py-4 sm:px-6">
            {errors.general && (
              <div className="rounded-lg bg-red-50 px-3 py-2 text-sm text-red-700">
                {errors.general}
              </div>
            )}

            <div>
              <label className="mb-2 block text-sm font-medium text-gray-800">
                Rate this prediction <span className="text-red-500">*</span>
              </label>

              <StarRatingInput
                value={selectedStars}
                onChange={handleRatingChange}
              />

              <p className="mt-1 text-xs text-gray-500 sm:text-sm">
                Choose from 1 to 5 stars.
              </p>

              {errors.rating && (
                <p className="mt-1 text-sm text-red-600">{errors.rating}</p>
              )}
            </div>

            <div>
              <label className="mb-2 block text-sm font-medium text-gray-800">
                Was this prediction close to what you expected?
              </label>

              <div className="flex gap-2">
                <FeedbackChoiceButton
                  selected={form.is_prediction_close === true}
                  onClick={() => handleChange("is_prediction_close", true)}
                >
                  Yes
                </FeedbackChoiceButton>

                <FeedbackChoiceButton
                  selected={form.is_prediction_close === false}
                  onClick={() => handleChange("is_prediction_close", false)}
                >
                  No
                </FeedbackChoiceButton>
              </div>
            </div>

            <div>
              <label className="mb-2 block text-sm font-medium text-gray-800">
                Actual body fat percentage
                {form.consent_to_retrain ? (
                  <span className="text-red-500"> *</span>
                ) : (
                  <span className="text-gray-500"> (optional)</span>
                )}
              </label>

              <input
                type="text"
                inputMode="decimal"
                pattern="[0-9]*[.]?[0-9]*"
                value={form.actual_fat_percentage}
                onChange={handleActualFatPercentageChange}
                className="w-full rounded-xl border border-gray-300 px-3 py-2.5 text-sm outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-100"
                placeholder="e.g. 18.4"
              />

              {errors.actual_fat_percentage && (
                <p className="mt-1 text-sm text-red-600">
                  {errors.actual_fat_percentage}
                </p>
              )}
            </div>

            <div>
              <label className="mb-2 block text-sm font-medium text-gray-800">
                Comment <span className="text-gray-500">(optional)</span>
              </label>

              <textarea
                rows={3}
                value={form.comment}
                onChange={(e) => handleChange("comment", e.target.value)}
                className="min-h-[88px] w-full resize-y rounded-xl border border-gray-300 px-3 py-2.5 text-sm outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-100"
                placeholder="Tell us anything useful about this result..."
              />

              {errors.comment && (
                <p className="mt-1 text-sm text-red-600">{errors.comment}</p>
              )}
            </div>

            <div className="rounded-xl border border-gray-200 bg-gray-50 p-3">
              <label className="flex items-start gap-3">
                <input
                  type="checkbox"
                  checked={form.consent_to_retrain}
                  onChange={(e) =>
                    handleChange("consent_to_retrain", e.target.checked)
                  }
                  className="mt-1 h-4 w-4 flex-shrink-0"
                />

                <span className="text-xs leading-5 text-gray-700 sm:text-sm">
                  Allow us to use this feedback to improve future predictions.
                  If you turn this on, please also enter your actual body fat
                  percentage above. This is optional and turned off by default.
                </span>
              </label>
            </div>
          </div>

          {/* Fixed footer */}
          <div className="flex flex-shrink-0 gap-2 border-t border-gray-200 bg-white px-4 py-3 sm:gap-3 sm:px-6">
            <button
              type="button"
              onClick={resetAndClose}
              className="min-h-10 flex-1 rounded-xl border border-gray-300 px-3 py-2 text-sm font-medium text-gray-700 hover:bg-gray-50 sm:min-h-11"
            >
              Cancel
            </button>

            <button
              type="submit"
              disabled={isSubmitting}
              className="min-h-10 flex-1 rounded-xl bg-blue-600 px-3 py-2 text-sm font-medium text-white hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-60 sm:min-h-11"
            >
              {isSubmitting ? "Submitting..." : "Confirm"}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default FeedbackFormModal;