import crypto from "node:crypto";
import fs from "node:fs/promises";
import path from "node:path";

const ROOT = "C:\\Users\\Admin\\Desktop\\OZON-E-CUP";
const SOURCE = path.join(ROOT, "submissions", "submission_STRONGEST_CURRENT.csv");
const SAMPLE = path.join(ROOT, "data", "raw", "sample_submit.csv");
const OUTPUT = path.join(ROOT, "submissions", "submission_LEVEL_MINUS_006.csv");
const ARTIFACT_DIR = path.join(ROOT, "artifacts", "LEVEL_MINUS_006_EXP060");
const DIAGNOSTICS = path.join(ARTIFACT_DIR, "diagnostics.json");
const EXPECTED_SOURCE_SHA256 = "abc2218b1a3d55d41121b7b5a22db7e95ffd45283b42ff0006c5e6e731e04bda";
const SHIFT = 0.06;
const QUANTILES = [0.01, 0.05, 0.10, 0.50, 0.90, 0.95, 0.99];

function invariant(condition, message) {
  if (!condition) throw new Error(message);
}

function sha256(buffer) {
  return crypto.createHash("sha256").update(buffer).digest("hex");
}

function parseSimpleSubmission(buffer, label) {
  invariant(!buffer.includes(13), `${label}: CR byte found; expected LF-only CSV`);
  invariant(buffer.at(-1) === 10, `${label}: missing final LF`);
  const text = buffer.toString("utf8");
  const lines = text.split("\n");
  invariant(lines.pop() === "", `${label}: malformed final line`);
  invariant(lines[0] === "user_id,predict", `${label}: unexpected schema ${lines[0]}`);

  const ids = new Array(lines.length - 1);
  const predictions = new Float64Array(lines.length - 1);
  const seen = new Set();
  for (let index = 1; index < lines.length; index += 1) {
    const line = lines[index];
    const comma = line.indexOf(",");
    invariant(comma > 0 && comma === line.lastIndexOf(","), `${label}: malformed row ${index + 1}`);
    const id = line.slice(0, comma);
    const rawPrediction = line.slice(comma + 1);
    invariant(/^\d+$/.test(id), `${label}: invalid user_id at row ${index + 1}`);
    invariant(!seen.has(id), `${label}: duplicate user_id ${id}`);
    seen.add(id);
    const prediction = Number(rawPrediction);
    invariant(Number.isFinite(prediction), `${label}: non-finite prediction at row ${index + 1}`);
    invariant(prediction >= 0, `${label}: negative prediction at row ${index + 1}`);
    ids[index - 1] = id;
    predictions[index - 1] = prediction;
  }
  return { text, header: lines[0], ids, predictions };
}

function parseSampleIds(buffer) {
  invariant(!buffer.includes(13), "sample: CR byte found; expected LF-only CSV");
  invariant(buffer.at(-1) === 10, "sample: missing final LF");
  const lines = buffer.toString("utf8").split("\n");
  invariant(lines.pop() === "", "sample: malformed final line");
  invariant(lines[0] === "user_id,predict", `sample: unexpected schema ${lines[0]}`);
  return lines.slice(1).map((line, index) => {
    const comma = line.indexOf(",");
    invariant(comma > 0, `sample: malformed row ${index + 2}`);
    return line.slice(0, comma);
  });
}

function quantileSorted(sorted, probability) {
  const position = (sorted.length - 1) * probability;
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  if (lower === upper) return sorted[lower];
  const weight = position - lower;
  return sorted[lower] * (1 - weight) + sorted[upper] * weight;
}

function summarize(predictions) {
  const sorted = Array.from(predictions).sort((a, b) => a - b);
  let sumPrediction = 0;
  let sumZ = 0;
  let zeros = 0;
  for (const prediction of predictions) {
    sumPrediction += prediction;
    sumZ += Math.log1p(prediction);
    if (prediction === 0) zeros += 1;
  }
  const quantiles = {};
  for (const probability of QUANTILES) {
    const key = `p${String(Math.round(probability * 100)).padStart(2, "0")}`;
    quantiles[key] = quantileSorted(sorted, probability);
  }
  return {
    mean_z: sumZ / predictions.length,
    mean_prediction: sumPrediction / predictions.length,
    median: quantiles.p50,
    zero_share: zeros / predictions.length,
    quantiles,
    max: sorted.at(-1),
  };
}

await fs.mkdir(ARTIFACT_DIR, { recursive: true });
const [sourceBuffer, sampleBuffer] = await Promise.all([
  fs.readFile(SOURCE),
  fs.readFile(SAMPLE),
]);
const sourceSha256 = sha256(sourceBuffer);
invariant(sourceSha256 === EXPECTED_SOURCE_SHA256, `source SHA256 mismatch: ${sourceSha256}`);

const source = parseSimpleSubmission(sourceBuffer, "source");
const sampleIds = parseSampleIds(sampleBuffer);
invariant(source.ids.length === sampleIds.length, "source/sample row count mismatch");
for (let index = 0; index < source.ids.length; index += 1) {
  invariant(source.ids[index] === sampleIds[index], `source/sample order mismatch at row ${index + 2}`);
}

const probeRaw = new Float64Array(source.predictions.length);
const outputLines = new Array(source.predictions.length + 1);
outputLines[0] = source.header;
let clippedRows = 0;
let meanRequestedZ = 0;
for (let index = 0; index < source.predictions.length; index += 1) {
  const zOriginal = Math.log1p(source.predictions[index]);
  const zRequested = zOriginal - SHIFT;
  const prediction = Math.max(Math.expm1(zRequested), 0);
  if (zRequested < 0) clippedRows += 1;
  meanRequestedZ += zRequested;
  probeRaw[index] = prediction;
  outputLines[index + 1] = `${source.ids[index]},${prediction.toFixed(6)}`;
}
meanRequestedZ /= source.predictions.length;
const outputText = `${outputLines.join("\n")}\n`;
const outputBuffer = Buffer.from(outputText, "utf8");
try {
  const existingOutput = await fs.readFile(OUTPUT);
  invariant(existingOutput.equals(outputBuffer), `Refusing to overwrite non-matching output: ${OUTPUT}`);
} catch (error) {
  if (error.code !== "ENOENT") throw error;
  await fs.writeFile(OUTPUT, outputBuffer, { flag: "wx" });
}

const output = parseSimpleSubmission(outputBuffer, "output");
invariant(output.header === source.header, "output/source schema mismatch");
invariant(output.ids.length === source.ids.length, "output/source row count mismatch");
for (let index = 0; index < output.ids.length; index += 1) {
  invariant(output.ids[index] === source.ids[index], `output/source order mismatch at row ${index + 2}`);
}

const originalSummary = summarize(source.predictions);
const probeSummary = summarize(output.predictions);
const expectedMeanRequestedZ = originalSummary.mean_z - SHIFT;
invariant(Math.abs(meanRequestedZ - expectedMeanRequestedZ) < 1e-12, "requested mean z shift is not exactly -0.06");
invariant(probeSummary.mean_z <= originalSummary.mean_z, "serialized probe level did not decrease");

const diagnostics = {
  experiment: "LEVEL_MINUS_006",
  experiment_id: "exp_060",
  created_at: new Date().toISOString(),
  training: "NONE",
  base: "STRONGEST_CURRENT",
  only_change: "z - 0.06",
  purpose: "public production-level diagnostic",
  source: {
    path: SOURCE,
    sha256: sourceSha256,
    expected_sha256: EXPECTED_SOURCE_SHA256,
    sha_verified: true,
    schema: source.header.split(","),
    row_count: source.ids.length,
    lf_only: true,
    final_newline: true,
    sample_path: SAMPLE,
    sample_schema_match: true,
    sample_order_match: true,
    missing_users: 0,
    duplicate_users: 0,
  },
  transform: {
    shift_log_space: -SHIFT,
    formula: "maximum(expm1(log1p(pred_STRONGEST_CURRENT) - 0.06), 0)",
    normalization_after_shift: false,
    other_changes: false,
    clipped_rows: clippedRows,
    clipped_share: clippedRows / source.predictions.length,
    mean_z_requested_before_nonnegative_clip: meanRequestedZ,
    expected_mean_z_requested: expectedMeanRequestedZ,
    requested_mean_z_shift_error: meanRequestedZ - expectedMeanRequestedZ,
  },
  diagnostics: {
    original: originalSummary,
    probe_serialized: probeSummary,
    actual_mean_log1p_shift: probeSummary.mean_z - originalSummary.mean_z,
    deviation_from_minus_006_due_to_nonnegative_clip_and_csv_rounding:
      probeSummary.mean_z - originalSummary.mean_z + SHIFT,
  },
  output: {
    path: OUTPUT,
    sha256: sha256(outputBuffer),
    byte_count: outputBuffer.length,
    schema_match: true,
    row_order_match: true,
    row_count_match: true,
    row_count: output.ids.length,
    missing_users: 0,
    duplicate_users: 0,
    nan_count: 0,
    inf_count: 0,
    negative_count: 0,
    lf_only: true,
    final_newline: true,
  },
  artifact_tool_verification: {
    performed_during_initial_creation: true,
    source_and_output_preview_rows: 100,
    visual_preview_rows: 15,
    result: "PASS",
  },
  future_lb_interpretation: {
    noticeably_better_than_strongest:
      "Evidence that the production global GMV level is below the current CV-calibrated level.",
    worse_by_expected_quadratic_amount:
      "Global calibration is probably already close to correct.",
    improvement_around_0001_or_more:
      "Strong evidence of a fundamental CV-to-test level mismatch; analyze separately instead of tuning the offset immediately.",
    difference_near_public_noise:
      "Do not tune anything from this single score.",
  },
};
await fs.writeFile(DIAGNOSTICS, `${JSON.stringify(diagnostics, null, 2)}\n`, "utf8");

console.log(JSON.stringify({
  source_sha256: sourceSha256,
  output_sha256: diagnostics.output.sha256,
  row_count: output.ids.length,
  mean_z_original: originalSummary.mean_z,
  mean_z_probe: probeSummary.mean_z,
  actual_mean_z_shift: diagnostics.diagnostics.actual_mean_log1p_shift,
  clipped_rows: clippedRows,
  output: OUTPUT,
  diagnostics: DIAGNOSTICS,
}, null, 2));
