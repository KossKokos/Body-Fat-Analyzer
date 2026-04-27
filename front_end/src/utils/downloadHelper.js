const escapeCsvValue = (value) => {
  if (value === null || value === undefined) {
    return "";
  }

  const stringValue = String(value);

  if (/[",\n\r]/.test(stringValue)) {
    return `"${stringValue.replace(/"/g, '""')}"`;
  }

  return stringValue;
};

const formatTimestampForDisplay = (timestamp) => {
  if (!timestamp) {
    return "Not available";
  }

  return new Date(timestamp).toLocaleString("en-GB", {
    dateStyle: "long",
    timeStyle: "short",
  });
};

const formatDateForFileName = () => {
  return new Date().toISOString().split("T")[0];
};

export const downloadPredictionReport = ({
  fatClassLabel,
  fatPercentage,
  timestamp,
}) => {
  const formattedTimestamp = formatTimestampForDisplay(timestamp);

  const formattedFatPercentage = Number.isFinite(Number(fatPercentage))
    ? `${Number(fatPercentage).toFixed(1)}%`
    : "Not available";

  const rows = [
    ["Body Fat Prediction Report", ""],
    ["Fat class", fatClassLabel],
    ["Body fat percentage", formattedFatPercentage],
    ["Generated", formattedTimestamp],
  ];

  const csvContent = rows
    .map((row) => row.map(escapeCsvValue).join(","))
    .join("\r\n");

  const blob = new Blob([`\uFEFF${csvContent}`], {
    type: "text/csv;charset=utf-8",
  });

  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");

  a.href = url;
  a.download = `body-fat-report-${formatDateForFileName()}.csv`;

  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);

  URL.revokeObjectURL(url);
};