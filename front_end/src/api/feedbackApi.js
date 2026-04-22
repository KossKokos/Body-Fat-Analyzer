import { postRequest } from "../utils/apiHelpers";

/**
 * Sends user feedback for a previously generated prediction.
 */
export const createPredictionFeedback = async (data) => {
  return postRequest("/api/feedback/", data, "Failed to save feedback");
};