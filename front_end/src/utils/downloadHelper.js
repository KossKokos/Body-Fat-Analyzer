export const downloadPredictionReport = ({ fatClassLabel, fatPercentage, timestamp }) => {
  const formattedTimestamp = timestamp
    ? new Date(timestamp).toLocaleString("en-GB", {
        dateStyle: "long",
        timeStyle: "short",
      })
    : "Not available";

  const report = [
    "Body Fat Prediction Report",
    "==========================",
    `Fat class: ${fatClassLabel}`,
    `Body fat percentage: ${Number(fatPercentage).toFixed(1)}%`,
    `Generated: ${formattedTimestamp}`,
  ].join("\n");

  const blob = new Blob([report], { type: "text/plain;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");

  a.href = url;
  a.download = `body-fat-report-${new Date().toISOString().split("T")[0]}.txt`;

  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
};