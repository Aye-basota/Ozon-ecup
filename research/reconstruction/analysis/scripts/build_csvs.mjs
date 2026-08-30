import fs from "node:fs/promises";
import path from "node:path";
import { Workbook } from "@oai/artifact-tool";

const root = process.cwd();
const analysisDir = path.join(root, "analysis");
const intermediateDir = path.join(analysisDir, "intermediate");

const registry = JSON.parse(
  await fs.readFile(path.join(intermediateDir, "compatibility.json"), "utf8"),
);
const predictions = JSON.parse(
  await fs.readFile(path.join(intermediateDir, "prediction_audit.json"), "utf8"),
);

function csvCell(value) {
  if (value === null || value === undefined) return "";
  const text = typeof value === "number" ? String(value) : String(value);
  return /[",\r\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

function toCsv(headers, rows) {
  return [
    headers.map(csvCell).join(","),
    ...rows.map((row) => headers.map((header) => csvCell(row[header])).join(",")),
  ].join("\r\n") + "\r\n";
}

function columnName(index) {
  let n = index + 1;
  let out = "";
  while (n > 0) {
    n -= 1;
    out = String.fromCharCode(65 + (n % 26)) + out;
    n = Math.floor(n / 26);
  }
  return out;
}

async function validateAndWrite(filename, csvText, expectedRows, expectedColumns) {
  const workbook = await Workbook.fromCSV(csvText, { sheetName: "Sheet1" });
  const lastColumn = columnName(expectedColumns - 1);
  const inspect = await workbook.inspect({
    kind: "table",
    sheetId: "Sheet1",
    range: `A1:${lastColumn}${Math.min(expectedRows + 1, 6)}`,
    include: "values,formulas",
    tableMaxRows: 6,
    tableMaxCols: Math.min(expectedColumns, 12),
    maxChars: 5000,
  });
  if (!inspect.ndjson || !inspect.ndjson.includes("Sheet1")) {
    throw new Error(`artifact-tool validation failed for ${filename}`);
  }
  const preview = await workbook.render({
    sheetName: "Sheet1",
    range: `A1:${columnName(Math.min(expectedColumns, 8) - 1)}${Math.min(expectedRows + 1, 8)}`,
    scale: 1,
    format: "png",
  });
  await fs.writeFile(
    path.join(intermediateDir, `${path.parse(filename).name}_preview.png`),
    new Uint8Array(await preview.arrayBuffer()),
  );
  await fs.writeFile(path.join(analysisDir, filename), csvText, "utf8");
  return { filename, rows: expectedRows, columns: expectedColumns };
}

const metadataHeaders = [
  "experiment_id",
  "canonical_name",
  "namespace",
  "role",
  "relationship_to_EXP-037",
  "normalized_primary_delta",
  "folds_positive",
  "folds_total",
  "comparison_class",
  "evidence_strength",
];
const experimentHeaders = registry.nodes.map((node) => node.experiment_id);
const compatibilityRows = registry.matrix.map((row) => {
  const out = {
    experiment_id: row.experiment_id,
    canonical_name: row.canonical_name,
    namespace: row.namespace,
    role: row.role,
    "relationship_to_EXP-037": row.relationship,
    normalized_primary_delta: row.delta,
    folds_positive: row.folds_positive,
    folds_total: row.folds_total,
    comparison_class: row.comparison_class,
    evidence_strength: row.evidence_strength,
  };
  for (const experimentId of experimentHeaders) {
    out[experimentId] = row.compatibility[experimentId];
  }
  return out;
});
const compatibilityHeaders = [...metadataHeaders, ...experimentHeaders];
const compatibilityCsv = toCsv(compatibilityHeaders, compatibilityRows);

const testMap = new Map();
for (const row of predictions.test_diversity) {
  testMap.set(`${row.source_a}|||${row.source_b}`, row);
  testMap.set(`${row.source_b}|||${row.source_a}`, row);
}
const baseWcv = predictions.models.STRONGEST_CURRENT.wcv;
const diversityRows = predictions.diversity.map((row) => {
  const test = testMap.get(`${row.source_a}|||${row.source_b}`) || {};
  return {
    source_a: row.source_a,
    source_b: row.source_b,
    oof_rows: row.oof_rows,
    source_a_wcv: predictions.models[row.source_a].wcv,
    source_b_wcv: predictions.models[row.source_b].wcv,
    source_a_delta_vs_strongest: predictions.models[row.source_a].wcv - baseWcv,
    source_b_delta_vs_strongest: predictions.models[row.source_b].wcv - baseWcv,
    prediction_corr: row.prediction_corr,
    residual_corr: row.residual_corr,
    error_covariance: row.error_covariance,
    disagreement_variance: row.disagreement_variance,
    mean_abs_disagreement: row.mean_abs_disagreement,
    test_rows: test.test_rows ?? "",
    test_prediction_corr: test.test_prediction_corr ?? "",
    test_disagreement_variance: test.test_disagreement_variance ?? "",
    test_mean_abs_disagreement: test.test_mean_abs_disagreement ?? "",
  };
});
const diversityHeaders = [
  "source_a",
  "source_b",
  "oof_rows",
  "source_a_wcv",
  "source_b_wcv",
  "source_a_delta_vs_strongest",
  "source_b_delta_vs_strongest",
  "prediction_corr",
  "residual_corr",
  "error_covariance",
  "disagreement_variance",
  "mean_abs_disagreement",
  "test_rows",
  "test_prediction_corr",
  "test_disagreement_variance",
  "test_mean_abs_disagreement",
];
const diversityCsv = toCsv(diversityHeaders, diversityRows);

await fs.mkdir(intermediateDir, { recursive: true });
const results = [];
results.push(
  await validateAndWrite(
    "COMPATIBILITY_MATRIX.csv",
    compatibilityCsv,
    compatibilityRows.length,
    compatibilityHeaders.length,
  ),
);
results.push(
  await validateAndWrite(
    "PREDICTION_DIVERSITY.csv",
    diversityCsv,
    diversityRows.length,
    diversityHeaders.length,
  ),
);
console.log(JSON.stringify(results));
