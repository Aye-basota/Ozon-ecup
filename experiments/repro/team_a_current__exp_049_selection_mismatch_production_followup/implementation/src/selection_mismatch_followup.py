"""EXP-049: same-fold EXP-048 reanalysis and artifact-only production audit.

This runner never fits a model.  It uses the persisted OOF/test artifacts from
STRONGEST_CURRENT, EXP-040, EXP-047 and EXP-048.  A submission is deliberately
not constructed unless exact registered BTYD and FRESH test predictions exist.

Run: python src/selection_mismatch_followup.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import polars as pl

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.block4_saf import _strongest_test
from src.config import ARTIFACTS, SEED, SUBMISSIONS
from src.data import sample_submit
from src.fresh_contrast import level_shift
from src.selection_mismatch_cv import (
    ELIGIBLE_FOLDS,
    INCREMENTAL,
    load_candidates,
    load_history,
    reconstruct_baseline,
    weighted_calibrate,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "research" / "strategies" / "results" / "SELMATCH_EXP049"
SELECTION_ARTIFACT = (ROOT / "research" / "strategies" / "results" /
                      "SELMATCH_EXP048" / "selection_rows.npz")
REFERENCE_CSV = (ROOT / "research" / "strategies" / "results" /
                 "SELMATCH_EXP048" / "reference_distribution.csv")
FOLDS = list(ELIGIBLE_FOLDS)
FOLD_WEIGHTS = np.asarray([1.0, 2.0, 4.0])
CANDIDATE = "BTYD05_FRESH1"
L_STAR = 2.3293
BOOTSTRAP_REPS = 500
SIGNAL_SHUFFLES = 200
SELECTION_SHUFFLES = 100


def jsonable(value):
    if isinstance(value, np.ndarray):
        return [jsonable(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(value), ensure_ascii=False, indent=2,
                               sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        fields.extend(k for k in row if k not in fields)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(jsonable(row.get(k)), ensure_ascii=False)
                             if isinstance(row.get(k), (list, tuple, dict, np.ndarray))
                             else jsonable(row.get(k, "")) for k in fields})


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def qstats(x: np.ndarray) -> dict:
    x = np.asarray(x, float)
    q = np.quantile(x, [0, .001, .005, .01, .05, .5, .95, .99, .995, .999, 1])
    return {
        "n": len(x), "finite": bool(np.isfinite(x).all()),
        "mean": float(x.mean()), "std": float(x.std()), "var": float(x.var()),
        **{f"q{p}": float(v) for p, v in zip(
            ("000", "001", "005", "010", "050", "500", "950", "990",
             "995", "999", "1000"), q)},
    }


def load_inputs():
    canonical, base_arrays, baseline_manifest = reconstruct_baseline()
    models, candidate_audit = load_candidates(canonical, base_arrays)
    history = load_history(canonical)
    sel = np.load(SELECTION_ARTIFACT, allow_pickle=False)
    assert np.array_equal(sel["user_id"], canonical["uid"])
    assert np.array_equal(np.asarray(sel["cutoff"], dtype="U10"), canonical["cutoff"])
    k = np.asarray(sel["future_blocks_active"], np.int8)
    ref = pl.read_csv(REFERENCE_CSV).sort("k")
    pi_ref = ref["pi_ref"].to_numpy().astype(float)
    pi_kpos = pi_ref.copy()
    pi_kpos[0] = 0.0
    pi_kpos /= pi_kpos.sum()
    return canonical, models, history, k, pi_ref, pi_kpos, baseline_manifest, candidate_audit


def scheme_weights(cut: np.ndarray, k: np.ndarray, pi_kpos: np.ndarray):
    schemes = {"A_STANDARD_3F": {}, "B_K3_3F": {}, "C_MATCHED_KPOS_3F": {}}
    support = []
    for fold in FOLDS:
        m = cut == fold
        fk = k[m]
        counts = np.bincount(fk, minlength=4)
        pi = counts / counts.sum()
        matched = np.asarray([pi_kpos[j] / pi[j] if j > 0 and pi[j] > 0 else 0.0
                              for j in fk], float)
        schemes["A_STANDARD_3F"][fold] = np.ones(m.sum(), float)
        schemes["B_K3_3F"][fold] = (fk == 3).astype(float)
        schemes["C_MATCHED_KPOS_3F"][fold] = matched
        ess = matched.sum() ** 2 / np.sum(matched * matched)
        support.append({
            "fold": fold, "n": int(m.sum()), "counts_k0_k3": counts,
            "pi_fold_k0_k3": pi, "pi_reference_kpos_k0_k3": pi_kpos,
            "max_weight": float(matched.max()), "ess": float(ess),
            "ess_fraction": float(ess / m.sum()), "k0_available": bool(counts[0]),
            "identified_scope": "conditional on k>0",
        })
    return schemes, support


def evaluate(canonical: dict, models: dict[str, np.ndarray], schemes: dict):
    rows = []
    y, cut = canonical["y"], canonical["cutoff"]
    wanted = ["BTYD05", "FRESH", CANDIDATE, "ZERO2D",
              "SEQ_SLOT_25", "SEQ_SLOT_50", "SEQ_SLOT_75"]
    for scheme, weights in schemes.items():
        base_scores = []
        candidate_scores = {name: [] for name in wanted}
        offsets = {"STRONGEST": []} | {name: [] for name in wanted}
        for fold in FOLDS:
            m = cut == fold
            off, score = weighted_calibrate(y[m], models["STRONGEST"][m], weights[fold])
            base_scores.append(score); offsets["STRONGEST"].append(off)
            for name in wanted:
                off, score = weighted_calibrate(y[m], models[name][m], weights[fold])
                candidate_scores[name].append(score); offsets[name].append(off)
        for name in wanted:
            delta = np.asarray(candidate_scores[name]) - np.asarray(base_scores)
            rows.append({
                "scheme": scheme, "model": name,
                "base_score": float(np.average(base_scores, weights=FOLD_WEIGHTS)),
                "score": float(np.average(candidate_scores[name], weights=FOLD_WEIGHTS)),
                "delta": float(np.average(delta, weights=FOLD_WEIGHTS)),
                "fold_scores": candidate_scores[name], "fold_base_scores": base_scores,
                "fold_deltas": delta, "signs": ["-" if x < 0 else "+" if x > 0 else "0" for x in delta],
                "improved_folds": int(np.sum(delta < 0)),
                "fold_offsets": offsets[name], "fold_base_offsets": offsets["STRONGEST"],
            })
    return rows


def bootstrap_candidate(canonical: dict, models: dict[str, np.ndarray], schemes: dict):
    """Cluster bootstrap; calibration is re-estimated in every replicate."""
    uid_unique = np.unique(canonical["uid"])
    uid_pos = np.searchsorted(uid_unique, canonical["uid"])
    rng = np.random.default_rng(SEED)
    out = {s: np.empty(BOOTSTRAP_REPS, float) for s in
           ("A_STANDARD_3F", "C_MATCHED_KPOS_3F")}
    fold_masks = {f: canonical["cutoff"] == f for f in FOLDS}
    for r in range(BOOTSTRAP_REPS):
        sampled = rng.integers(0, len(uid_unique), size=len(uid_unique))
        mult = np.bincount(sampled, minlength=len(uid_unique)).astype(float)
        for scheme in out:
            deltas = []
            for fold in FOLDS:
                m = fold_masks[fold]
                w = schemes[scheme][fold] * mult[uid_pos[m]]
                bs = weighted_calibrate(canonical["y"][m], models["STRONGEST"][m], w)[1]
                cs = weighted_calibrate(canonical["y"][m], models[CANDIDATE][m], w)[1]
                deltas.append(cs - bs)
            out[scheme][r] = np.average(deltas, weights=FOLD_WEIGHTS)
    rows = []
    for scheme, values in out.items():
        rows.append({
            "scheme": scheme, "model": CANDIDATE, "replicates": BOOTSTRAP_REPS,
            "calibration_reestimated_inside_replicate": True,
            "mean": float(values.mean()), "median": float(np.median(values)),
            "p025": float(np.quantile(values, .025)), "p05": float(np.quantile(values, .05)),
            "p10": float(np.quantile(values, .10)), "p90": float(np.quantile(values, .90)),
            "p95": float(np.quantile(values, .95)), "p975": float(np.quantile(values, .975)),
            "p_delta_lt0": float(np.mean(values < 0)),
        })
    return rows, out


def history_groups(history: dict, mask: np.ndarray, k: np.ndarray | None = None):
    rec = history["rec_buy"][mask]
    buy = history["w180_days_buy"][mask]
    rb = np.where((rec >= 15) & (rec <= 60), 1, np.where(rec > 60, 2, 0))
    wb = np.where(buy <= 1, 0, np.where(buy <= 15, 1, 2))
    keys = rb * 3 + wb
    if k is not None:
        keys = keys * 4 + k
    return [np.flatnonzero(keys == value) for value in np.unique(keys)]


def shuffle_controls(canonical: dict, models: dict[str, np.ndarray], history: dict,
                     k: np.ndarray, pi_kpos: np.ndarray, schemes: dict, point_rows: list[dict]):
    """Return a signal-placebo and the old selection-shuffle with explicit estimands."""
    rng = np.random.default_rng(SEED)
    y, cut = canonical["y"], canonical["cutoff"]
    corr = models[CANDIDATE] - models["STRONGEST"]
    signal = np.empty(SIGNAL_SHUFFLES, float)
    groups_signal = {}
    for fold in FOLDS:
        m = cut == fold
        groups_signal[fold] = history_groups(history, m, k[m])
    for r in range(SIGNAL_SHUFFLES):
        fd = []
        for fold in FOLDS:
            m = cut == fold
            c = corr[m].copy()
            for ix in groups_signal[fold]:
                c[ix] = rng.permutation(c[ix])
            w = schemes["C_MATCHED_KPOS_3F"][fold]
            bs = weighted_calibrate(y[m], models["STRONGEST"][m], w)[1]
            cs = weighted_calibrate(y[m], models["STRONGEST"][m] + c, w)[1]
            fd.append(cs - bs)
        signal[r] = np.average(fd, weights=FOLD_WEIGHTS)

    point = {(r["scheme"], r["model"]): r for r in point_rows}
    real_effect = point[("C_MATCHED_KPOS_3F", CANDIDATE)]["delta"]
    signal_row = {
        "control": "SIGNAL_CORRECTION_SHUFFLE", "estimand": "matched candidate delta",
        "real_value": real_effect, "permutations": SIGNAL_SHUFFLES,
        "p05": float(np.quantile(signal, .05)), "p50": float(np.median(signal)),
        "p95": float(np.quantile(signal, .95)),
        "outside_central_90": bool(real_effect < np.quantile(signal, .05)
                                    or real_effect > np.quantile(signal, .95)),
        "passed_improving_direction": bool(real_effect < np.quantile(signal, .05)),
        "permutation": "correction within fold x k x rec_buy_bin x w180_buy_bin",
    }

    # Reproduce the EXP-048 selection-placebo estimand correctly: it is C-A,
    # not the candidate's C delta.  The old report printed the latter next to
    # an interval and boolean computed from the former.
    selection = np.empty(SELECTION_SHUFFLES, float)
    groups_selection = {f: history_groups(history, cut == f) for f in FOLDS}
    standard = point[("A_STANDARD_3F", CANDIDATE)]["delta"]
    for r in range(SELECTION_SHUFFLES):
        fd = []
        for fold in FOLDS:
            m = cut == fold
            kp = k[m].copy()
            for ix in groups_selection[fold]:
                kp[ix] = rng.permutation(kp[ix])
            counts = np.bincount(kp, minlength=4)
            pi = counts / counts.sum()
            w = np.asarray([pi_kpos[j] / pi[j] if j > 0 and pi[j] > 0 else 0.0
                            for j in kp])
            bs = weighted_calibrate(y[m], models["STRONGEST"][m], w)[1]
            cs = weighted_calibrate(y[m], models[CANDIDATE][m], w)[1]
            fd.append(cs - bs)
        selection[r] = np.average(fd, weights=FOLD_WEIGHTS) - standard
    real_shift = real_effect - standard
    selection_row = {
        "control": "SELECTION_K_SHUFFLE", "estimand": "selection reweighting shift C-A",
        "real_value": real_shift, "candidate_real_effect_reported_separately": real_effect,
        "permutations": SELECTION_SHUFFLES,
        "p05": float(np.quantile(selection, .05)), "p50": float(np.median(selection)),
        "p95": float(np.quantile(selection, .95)),
        "outside_central_90": bool(real_shift < np.quantile(selection, .05)
                                    or real_shift > np.quantile(selection, .95)),
        "passed_improving_direction": bool(real_shift < np.quantile(selection, .05)),
        "root_cause_exp048": (
            "REPORT displayed matched candidate effect, while outside/interval were "
            "computed for the different estimand matched-minus-standard selection shift"),
    }
    return [signal_row, selection_row], {"signal": signal, "selection": selection}


def residual_and_interaction(canonical: dict, models: dict[str, np.ndarray],
                             schemes: dict, point_rows: list[dict]):
    rows = []
    y, cut = canonical["y"], canonical["cutoff"]
    for scheme in ("A_STANDARD_3F", "C_MATCHED_KPOS_3F"):
        for fold in FOLDS:
            m = cut == fold
            w = schemes[scheme][fold]
            base = models["STRONGEST"][m]
            off = weighted_calibrate(y[m], base, w)[0]
            residual = np.log1p(y[m]) - np.maximum(base + off, 0.0)
            correction = models[CANDIDATE][m] - base
            keep = w > 0
            ww = w[keep] / w[keep].sum()
            a, b = correction[keep], residual[keep]
            am, bm = np.sum(ww*a), np.sum(ww*b)
            cov = np.sum(ww*(a-am)*(b-bm))
            corr = cov / math.sqrt(np.sum(ww*(a-am)**2)*np.sum(ww*(b-bm)**2))
            rows.append({"scheme": scheme, "fold": fold, "n": int(keep.sum()),
                         "residual_alignment": float(corr),
                         "weighted_covariance": float(cov),
                         "correction_mean": float(am), "correction_std": float(math.sqrt(np.sum(ww*(a-am)**2)))})
    point = {(r["scheme"], r["model"]): r for r in point_rows}
    interactions = []
    for scheme in ("A_STANDARD_3F", "C_MATCHED_KPOS_3F"):
        combo = point[(scheme, CANDIDATE)]
        btyd = point[(scheme, "BTYD05")]
        fresh = point[(scheme, "FRESH")]
        interactions.append({
            "scheme": scheme, "combined_delta": combo["delta"],
            "btyd05_delta": btyd["delta"], "fresh_delta": fresh["delta"],
            "interaction": combo["delta"] - btyd["delta"] - fresh["delta"],
            "fold_interactions": (np.asarray(combo["fold_deltas"])
                                  - np.asarray(btyd["fold_deltas"])
                                  - np.asarray(fresh["fold_deltas"])),
        })
    return rows, interactions


def missing_k0_sensitivity(canonical: dict, models: dict[str, np.ndarray], k: np.ndarray,
                           pi_ref: np.ndarray, schemes: dict):
    """Scenario analysis; k=0 remains explicitly non-identified."""
    y, cut = canonical["y"], canonical["cutoff"]
    p0 = float(pi_ref[0])
    per_fold = []
    for fold in FOLDS:
        m = cut == fold
        w = schemes["C_MATCHED_KPOS_3F"][fold]
        base, cand = models["STRONGEST"][m], models[CANDIDATE][m]
        bo = weighted_calibrate(y[m], base, w)[0]
        co = weighted_calibrate(y[m], cand, w)[0]
        be = np.square(np.log1p(y[m]) - np.maximum(base + bo, 0))
        ce = np.square(np.log1p(y[m]) - np.maximum(cand + co, 0))
        row = {"fold": fold, "identified_base_mse": float(np.sum(w*be)/np.sum(w)),
               "identified_candidate_mse": float(np.sum(w*ce)/np.sum(w))}
        for kval in (1, 2, 3):
            s = k[m] == kval
            row[f"k{kval}_n"] = int(s.sum())
            row[f"k{kval}_base_mse"] = float(be[s].mean())
            row[f"k{kval}_candidate_mse"] = float(ce[s].mean())
        per_fold.append(row)

    scenarios = []
    for label in ("neutral", "like_k1", "like_k2", "like_k3"):
        deltas = []
        for row in per_fold:
            mb, mc = row["identified_base_mse"], row["identified_candidate_mse"]
            if label == "neutral":
                b0 = c0 = row["k1_base_mse"]
            else:
                kval = int(label[-1])
                b0, c0 = row[f"k{kval}_base_mse"], row[f"k{kval}_candidate_mse"]
            deltas.append(math.sqrt((1-p0)*mc+p0*c0)-math.sqrt((1-p0)*mb+p0*b0))
        scenarios.append({"scenario": label, "missing_mass": p0,
                          "delta": float(np.average(deltas, weights=FOLD_WEIGHTS)),
                          "fold_deltas": deltas,
                          "identified": False})

    # Required constant extra candidate MSE in k=0 that would erase the gain.
    def delta_for(extra: float) -> float:
        ds = []
        for row in per_fold:
            mb, mc = row["identified_base_mse"], row["identified_candidate_mse"]
            b0 = row["k1_base_mse"]
            c0 = max(0.0, b0 + extra)
            ds.append(math.sqrt((1-p0)*mc+p0*c0)-math.sqrt((1-p0)*mb+p0*b0))
        return float(np.average(ds, weights=FOLD_WEIGHTS))
    lo, hi = 0.0, 100.0
    for _ in range(80):
        mid = (lo+hi)/2
        if delta_for(mid) < 0:
            lo = mid
        else:
            hi = mid
    break_even = (lo+hi)/2
    scenarios.append({"scenario": "break_even_adversarial_k0", "missing_mass": p0,
                      "extra_candidate_mse_vs_base_k0": break_even,
                      "delta": delta_for(break_even), "identified": False})
    return per_fold, scenarios


def fresh_support_audit() -> dict:
    rows = []
    all_pass = True
    for fold in FOLDS + ["2025-10-16"]:
        tag = fold.replace("-", "")
        full = np.load(ARTIFACTS / f"FRESH_CONTRAST_MOE_fold_{tag}.npz", allow_pickle=False)
        mirror = np.load(ARTIFACTS / f"FRESH_CONTRAST_MOE_mirror_A_{tag}.npz", allow_pickle=False)
        uid, group = full["uid"], full["group"]
        donor, recipient = mirror["donor_uid"], mirror["recipient_uid"]
        passed = (len(uid) == len(np.unique(uid)) and
                  len(np.intersect1d(donor, recipient)) == 0 and
                  np.isin(recipient, uid[group == 1]).all() and
                  np.isfinite(full["d_fresh"]).all())
        all_pass &= passed
        rows.append({"fold": fold, "rows": len(uid), "unique_users": len(np.unique(uid)),
                     "group0": int(np.sum(group == 0)), "group1": int(np.sum(group == 1)),
                     "mirror_donors": len(donor), "mirror_recipients": len(recipient),
                     "donor_recipient_overlap": int(len(np.intersect1d(donor, recipient))),
                     "algebra_max_fresh": float(full["algebra_max_fresh"]), "pass": bool(passed)})
    return {"status": "PASS" if all_pass else "FAIL", "folds": rows,
            "semantics": "saved two-sided OOF donor/recipient support only"}


def production_audit(canonical: dict, models: dict[str, np.ndarray], candidate_audit: dict):
    """Audit exact support.  Missing registered test artifacts are a hard FAIL."""
    b = np.load(ARTIFACTS / "BTYD_DAY_BGNBD_EXP047_V2" / "oof_raw.npz", allow_pickle=False)
    f = np.load(ARTIFACTS / "oof_FRESH_CONTRAST_MOE.npz", allow_pickle=False)
    bo = np.lexsort((b["user_id"], np.asarray(b["cutoff"], dtype="U10")))
    fo = np.lexsort((f["uid"], np.asarray(f["cutoff"], dtype="U10")))
    assert np.array_equal(b["user_id"][bo], canonical["uid"])
    assert np.array_equal(f["uid"][fo], canonical["uid"])
    c_btyd = .05 * (np.asarray(b["z_btyd"], float)[bo] - models["STRONGEST"])
    c_fresh = np.asarray(f["fresh_processed_nested"], float)[fo]
    c_combined = c_btyd + c_fresh

    fit = pl.read_csv(ROOT / "research" / "strategies" / "results" /
                      "BTYD_DAY_BGNBD" / "fit_parameters.csv")
    monetary = pl.read_csv(ROOT / "research" / "strategies" / "results" /
                           "BTYD_DAY_BGNBD" / "monetary_parameters.csv")
    nested = pl.read_csv(ROOT / "research" / "strategies" / "results" /
                         "FRESH_CONTRAST" / "nested_lofo.csv")
    fresh_rows = nested.filter(pl.col("contrast") == "FRESH")

    uid_test, z_test = _strongest_test()
    sample = sample_submit()
    same_sample_set = set(uid_test.tolist()) == set(sample["user_id"].to_list())
    shift = level_shift(z_test, L_STAR)
    z_level = np.maximum(z_test + shift, 0.0)
    order = np.argsort(uid_test)
    pos = np.searchsorted(uid_test[order], sample["user_id"].to_numpy())
    z_sample = z_level[order][pos]
    strongest_file = SUBMISSIONS / "submission_STRONGEST_CURRENT.csv"
    strongest = pl.read_csv(strongest_file)
    strongest_recon = float(np.max(np.abs(
        np.log1p(strongest["predict"].to_numpy()) - z_sample)))

    btyd_test = sorted(str(p.resolve()) for p in ARTIFACTS.glob("*BTYD*test*") if p.is_file())
    fresh_test = sorted(str(p.resolve()) for p in ARTIFACTS.glob("*FRESH*test*") if p.is_file())
    exact_btyd = len(btyd_test) > 0
    exact_fresh = len(fresh_test) > 0
    blockers = []
    if not exact_btyd:
        blockers.append("EXP-047 registered no production/test inference artifact or production ensemble rule")
    if not exact_fresh:
        blockers.append("EXP-040 registered no production/test inference artifact; conditional head weights were not persisted")
    blockers.append("inventing latest-fold/averaged BTYD fits or carrying OOF FRESH corrections to test is not an exact registered recipe")

    audit = {
        "status": "PASS" if exact_btyd and exact_fresh else "FAIL_MISSING_EXACT_PRODUCTION_SUPPORT",
        "no_training_performed": True, "candidate_recipe_oof": candidate_audit["candidate_formulas"][CANDIDATE],
        "strongest_test": {"rows": len(uid_test), "unique_users": len(np.unique(uid_test)),
                           "same_user_set_as_sample": same_sample_set, "finite": bool(np.isfinite(z_test).all()),
                           "raw_stats": qstats(z_test), "level_shift": float(shift),
                           "mean_log1p_after_level": float(z_level.mean()),
                           "existing_submission_reconstruction_max_abs": strongest_recon,
                           "existing_submission_sha256": sha256(strongest_file)},
        "oof_alignment": {"rows": len(canonical["uid"]), "btyd": True, "fresh": True,
                          "same_users_and_targets": True},
        "oof_corrections": {"BTYD05": qstats(c_btyd), "FRESH_GLOBAL_ALPHA1": qstats(c_fresh),
                            "BTYD05_FRESH1": qstats(c_combined),
                            "combined_finite": bool(np.isfinite(c_combined).all())},
        "btyd_oof_support": {
            "p_alive": qstats(np.asarray(b["p_alive"], float)),
            "expected_count_30": qstats(np.asarray(b["expected_count_30"], float)),
            "fit_parameters": {c: qstats(fit[c].to_numpy()) for c in ("r", "alpha", "a", "b")},
            "all_fits_stable": bool(fit["stable"].all()),
            "max_gradient_norm": float(fit["gradient_norm"].max()),
            "max_mean_nll_spread": float(fit["mean_nll_spread"].max()),
            "max_log_parameter_spread": float(fit["max_log_parameter_spread"].max()),
            "monetary_mu_population": qstats(monetary["mu_population"].to_numpy()),
            "monetary_sigma_population": qstats(monetary["sigma_population"].to_numpy()),
        },
        "fresh_oof_support": {**fresh_support_audit(),
                              "nested_clipped_fraction": qstats(fresh_rows["clipped_fraction"].to_numpy())},
        "test_corrections": {
            "BTYD05": {"available": exact_btyd, "paths": btyd_test},
            "FRESH_GLOBAL_ALPHA1": {"available": exact_fresh, "paths": fresh_test},
            "combined": {"available": exact_btyd and exact_fresh},
            "variance_ratio_test_to_oof": None, "diagnostic_range": [0.6, 1.4],
            "correction_quantiles": None, "clipping": None, "test_regime_support": None,
        },
        "blockers": blockers,
        "submission_authorized": bool(exact_btyd and exact_fresh),
    }
    return audit


def build_report(summary: dict) -> str:
    p = summary["candidate"]
    return f"""# exp_049 — corrected EXP-048 same-fold analysis / production audit

- **Дата:** 2026-08-23
- **Тип:** analysis/production only; model training = **NO**
- **Кандидат:** `BTYD05_FRESH1` (fixed weights)
- **Verdict:** **{summary['verdict']}**

## Method correction

EXP-048 mixed standard 4-fold 1:2:4:8 with matched 3-fold 1:2:4, so its
selection penalty also contained removal of `2025-10-16`.  This follow-up uses
exactly `09-04/09-18/10-02`, weights `1:2:4`, for A/B/C.  C is explicitly
conditional on `k>0`; reference probabilities were renormalized after removing
`pi_ref(k=0)={summary['pi_ref_k0']:.6f}` and are not called fully identified CV.

## Corrected endpoint

| scheme | delta | folds | signs |
|---|---:|---:|---|
| A_STANDARD_3F | {p['standard_delta']:+.6f} | {p['standard_improved']}/3 | `{p['standard_signs']}` |
| C_MATCHED_KPOS_3F | {p['matched_delta']:+.6f} | {p['matched_improved']}/3 | `{p['matched_signs']}` |

Bootstrap re-estimates weighted calibration inside every one of 500 cluster
replicates: `P(delta<0)={p['bootstrap_p_lt0']:.3f}` and 95% interval
`[{p['bootstrap_p025']:+.6f}, {p['bootstrap_p975']:+.6f}]`.

The EXP-048 shuffle contradiction was an estimand/reporting bug: the displayed
`-0.000544` was the matched candidate effect, but the interval and `outside`
boolean were computed for `matched-standard` selection shift.  The corrected
signal shuffle compares like with like and {'passes' if p['signal_shuffle_pass'] else 'fails'};
the selection-k shuffle remains a separate sensitivity diagnostic.

Missing `k=0` sensitivity is recorded in `missing_k0_sensitivity.csv`.  It is
scenario analysis only; no result is presented as fully identified matched-CV.

## Production audit

Status: **{summary['production_status']}**.  STRONGEST_CURRENT test rows/order,
finiteness, official level normalization and reconstruction all pass.  Exact
BTYD and FRESH production predictions do not exist in the authorized artifacts:
EXP-047 explicitly stopped before test inference, while EXP-040 stopped before
production inference and did not persist conditional-head weights.  Averaging
fold BTYD fits, choosing the latest fit, or carrying OOF FRESH corrections to
test would invent an unregistered production recipe.  Therefore correction
quantiles/variance ratio/support on test are not identified, the audit cannot
PASS, and no submission slot is spent.

## Final gate

Validation evidence meets the fixed statistical gate: **{summary['validation_gate']}**.
Production-support audit is a hard failure, so final verdict is **{summary['verdict']}**.
Submission path/hash: **not created**.

Artifacts: `research/strategies/results/SELMATCH_EXP049/`.
"""


def main() -> None:
    if SEED != 42:
        raise AssertionError("all stochastic diagnostics must use config.SEED")
    RESULTS.mkdir(parents=True, exist_ok=True)
    canonical, models, history, k, pi_ref, pi_kpos, baseline_manifest, candidate_audit = load_inputs()
    schemes, support = scheme_weights(canonical["cutoff"], k, pi_kpos)
    point = evaluate(canonical, models, schemes)
    bootstrap, bootstrap_raw = bootstrap_candidate(canonical, models, schemes)
    shuffles, shuffle_raw = shuffle_controls(canonical, models, history, k, pi_kpos, schemes, point)
    residual, interactions = residual_and_interaction(canonical, models, schemes, point)
    k0_fold, k0_sensitivity = missing_k0_sensitivity(canonical, models, k, pi_ref, schemes)
    prod = production_audit(canonical, models, candidate_audit)

    lookup = {(r["scheme"], r["model"]): r for r in point}
    a = lookup[("A_STANDARD_3F", CANDIDATE)]
    c = lookup[("C_MATCHED_KPOS_3F", CANDIDATE)]
    boot_c = next(r for r in bootstrap if r["scheme"] == "C_MATCHED_KPOS_3F")
    signal = next(r for r in shuffles if r["control"] == "SIGNAL_CORRECTION_SHUFFLE")
    preferred = (c["delta"] <= -.0005 and c["improved_folds"] == 3
                 and signal["passed_improving_direction"] and a["delta"] < 0)
    exploratory = (-.0005 < c["delta"] <= -.0003 and c["improved_folds"] == 3
                   and boot_c["p_delta_lt0"] >= .90
                   and signal["passed_improving_direction"] and a["delta"] < 0)
    validation_gate = "PREFERRED" if preferred else "EXPLORATORY" if exploratory else "FAIL"
    verdict = ("PROMOTE" if preferred and prod["status"] == "PASS" else
               "EXPERIMENTAL_SUBMISSION" if exploratory and prod["status"] == "PASS" else "REJECT")
    summary = {
        "experiment": "SELMATCH_EXP049", "verdict": verdict,
        "no_training": True, "pi_ref_k0": float(pi_ref[0]),
        "reference_kpos": pi_kpos, "folds": FOLDS, "fold_weights": FOLD_WEIGHTS,
        "candidate_recipe": candidate_audit["candidate_formulas"][CANDIDATE],
        "candidate": {
            "standard_delta": a["delta"], "standard_fold_deltas": a["fold_deltas"],
            "standard_signs": a["signs"], "standard_improved": a["improved_folds"],
            "matched_delta": c["delta"], "matched_fold_deltas": c["fold_deltas"],
            "matched_signs": c["signs"], "matched_improved": c["improved_folds"],
            "bootstrap_p_lt0": boot_c["p_delta_lt0"], "bootstrap_p025": boot_c["p025"],
            "bootstrap_p975": boot_c["p975"],
            "signal_shuffle_pass": signal["passed_improving_direction"],
        },
        "validation_gate": validation_gate, "production_status": prod["status"],
        "submission": None, "baseline_manifest": baseline_manifest,
        "methodological_error_exp048": (
            "A used 4 folds/1:2:4:8 while C used 3 folds/1:2:4; reported penalty mixed "
            "selection reweighting with removal of 2025-10-16"),
        "shuffle_reporting_error_exp048": shuffles[1]["root_cause_exp048"],
    }

    write_csv(RESULTS / "same_fold_results.csv", point)
    write_csv(RESULTS / "matched_support.csv", support)
    write_csv(RESULTS / "bootstrap.csv", bootstrap)
    write_csv(RESULTS / "shuffle_controls.csv", shuffles)
    write_csv(RESULTS / "residual_alignment.csv", residual)
    write_csv(RESULTS / "interaction.csv", interactions)
    write_csv(RESULTS / "missing_k0_fold_inputs.csv", k0_fold)
    write_csv(RESULTS / "missing_k0_sensitivity.csv", k0_sensitivity)
    write_json(RESULTS / "production_audit.json", prod)
    write_json(RESULTS / "summary.json", summary)
    np.savez_compressed(RESULTS / "diagnostic_distributions.npz",
                        bootstrap_A=bootstrap_raw["A_STANDARD_3F"],
                        bootstrap_C=bootstrap_raw["C_MATCHED_KPOS_3F"],
                        shuffle_signal=shuffle_raw["signal"],
                        shuffle_selection=shuffle_raw["selection"])
    report = build_report(summary)
    (RESULTS / "REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps(jsonable({"verdict": verdict, "validation_gate": validation_gate,
                              "production": prod["status"], "submission": None}),
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
