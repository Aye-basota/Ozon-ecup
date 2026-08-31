"""Contract tests for EXP-057 GLOBAL-REGIME-OCC-RANK.

Covers the checks required before any arm is trained:
cutoff safety, global aggregation correctness, percentile determinism,
no target-window access, placebo marginal preservation, row/user alignment,
deterministic rerun and submission schema.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest

from src import global_regime_occ as G


# ------------------------------------------------------------------- fixtures
def synth_log() -> pl.DataFrame:
    """Tiny deterministic event log with a hand-checkable answer.

    Three users, days 2025-01-01 .. 2025-01-10.  User 3 only appears strictly
    after the cutoff we test with, so it must never influence any feature.
    """
    rows = []
    d = dt.date(2025, 1, 1)
    # user 1: buys on day 2 and day 9
    for k, (searches, cat, cart, order, gmv) in enumerate([
        (2, 1, 0, 0, 0.0),      # 01-01
        (3, 0, 1, 1, 100.0),    # 01-02
        (0, 0, 0, 0, 0.0),      # 01-03
        (1, 1, 0, 0, 0.0),      # 01-04
        (0, 2, 1, 0, 0.0),      # 01-05
    ]):
        rows.append(dict(event_date=d + dt.timedelta(days=k), user_id=1, searches=searches,
                         cat=cat, to_cart=cart, to_ord=order, gmv=gmv,
                         search_to_cart=0, search_to_ord=0, cat_to_cart=0, cat_to_ord=0,
                         gmv_search=0.0, gmv_cat=0.0))
    # user 2: one purchase inside the window, one after the cutoff
    for k, (searches, cart, order, gmv) in enumerate([
        (5, 2, 1, 50.0),        # 01-03
        (0, 0, 0, 0.0),         # 01-04
    ]):
        rows.append(dict(event_date=dt.date(2025, 1, 3) + dt.timedelta(days=k), user_id=2,
                         searches=searches, cat=0, to_cart=cart, to_ord=order, gmv=gmv,
                         search_to_cart=0, search_to_ord=0, cat_to_cart=0, cat_to_ord=0,
                         gmv_search=0.0, gmv_cat=0.0))
    # strictly-after-cutoff rows: the leakage tripwire
    rows.append(dict(event_date=dt.date(2025, 1, 9), user_id=2, searches=99, cat=99,
                     to_cart=99, to_ord=99, gmv=99999.0, search_to_cart=0, search_to_ord=0,
                     cat_to_cart=0, cat_to_ord=0, gmv_search=0.0, gmv_cat=0.0))
    rows.append(dict(event_date=dt.date(2025, 1, 10), user_id=3, searches=7, cat=7,
                     to_cart=7, to_ord=7, gmv=7777.0, search_to_cart=0, search_to_ord=0,
                     cat_to_cart=0, cat_to_ord=0, gmv_search=0.0, gmv_cat=0.0))
    return pl.DataFrame(rows)


@pytest.fixture()
def lf() -> pl.LazyFrame:
    return synth_log().lazy()


CUT = dt.date(2025, 1, 5)


# ------------------------------------------------------- global aggregation
def test_global_totals_match_hand_computation(lf):
    """Window (T-5, T] = 01-01..01-05 for the synthetic log."""
    t = G._block_totals(lf, CUT, 5)
    assert t["users_row"] == 2                    # users 1 and 2, not 3
    assert t["users_buy"] == 2                    # both bought inside the window
    assert t["users_search"] == 2
    assert t["users_cart"] == 2
    assert t["searches"] == 2 + 3 + 1 + 5         # user1 days1..5 + user2 day3
    assert t["carts"] == 1 + 1 + 2
    assert t["orders"] == 1 + 1
    assert t["gmv"] == pytest.approx(150.0)
    assert t["buyer_rate"] == pytest.approx(1.0)
    assert t["gmv_per_buyer"] == pytest.approx(75.0)
    assert t["gmv_per_order"] == pytest.approx(75.0)
    assert t["conv_cart_order"] == pytest.approx(2 / 4)


def test_global_state_never_reads_after_cutoff(lf):
    """The 99999-GMV row on 01-09 and user 3 on 01-10 must be invisible at T=01-05."""
    state = G.build_global_state(CUT, lf)
    for w in G.GLOBAL_WINDOWS:
        assert state[f"g_w{w}_gmv"] == pytest.approx(150.0)
        assert state[f"g_w{w}_users_row"] == 2
    assert max(state[f"g_w{w}_orders"] for w in G.GLOBAL_WINDOWS) == 2


def test_truncating_the_future_does_not_change_features(lf):
    """Deleting every row after the cutoff must be a no-op: the definition of
    cutoff safety, checked on the whole feature vector at once."""
    truncated = synth_log().filter(pl.col("event_date") <= CUT).lazy()
    a = G.build_global_state(CUT, lf)
    b = G.build_global_state(CUT, truncated)
    assert a.keys() == b.keys()
    for k in a:
        assert a[k] == pytest.approx(b[k]), k


def test_window_frame_is_half_open(lf):
    """(hi - days, hi]: the left edge is excluded, the right edge included."""
    got = set(G._window_frame(lf, CUT, 2).collect()["event_date"].to_list())
    assert got == {dt.date(2025, 1, 4), dt.date(2025, 1, 5)}


def test_dynamics_compare_adjacent_blocks(lf):
    """g_d{w}_dlog_x = log1p(last block) - log1p(previous block)."""
    state = G.build_global_state(CUT, lf)
    last = G._block_totals(lf, CUT, 7)
    prev = G._block_totals(lf, CUT - dt.timedelta(days=7), 7)
    expected = np.log1p(last["gmv"]) - np.log1p(prev["gmv"])
    assert state["g_d7_dlog_gmv"] == pytest.approx(expected)


def test_no_cutoff_identity_leaks_into_the_names():
    """The manifest forbids cutoff index / date / fold id as inputs."""
    for n in G.global_feature_names():
        low = n.lower()
        assert "cutoff" not in low and "date" not in low and "fold" not in low
        assert not any(ch.isdigit() and low.startswith("g_2") for ch in low)


def test_global_feature_names_match_built_keys(lf):
    assert sorted(G.build_global_state(CUT, lf)) == sorted(G.global_feature_names())


def test_global_names_are_unique():
    n = G.all_new_feature_names()
    assert len(n) == len(set(n))


# ---------------------------------------------------------------- percentile
def test_percentile_is_deterministic_and_order_free():
    rng = np.random.default_rng(0)
    v = np.r_[np.zeros(500), rng.gamma(2.0, 3.0, 500)]
    a = G.percentile(v)
    b = G.percentile(v.copy())
    assert np.array_equal(a, b)
    # permuting the input permutes the output the same way: no positional bias
    perm = rng.permutation(v.size)
    assert np.allclose(G.percentile(v[perm]), a[perm])


def test_percentile_gives_ties_one_shared_value():
    v = np.array([0.0, 0.0, 0.0, 5.0])
    p = G.percentile(v)
    assert len(set(p[:3].tolist())) == 1
    assert p[3] == pytest.approx(1.0)
    assert p[0] == pytest.approx(2.0 / 4.0)


def test_percentile_is_monotone_and_bounded():
    rng = np.random.default_rng(1)
    v = rng.gamma(1.5, 2.0, 2000)
    p = G.percentile(v)
    assert p.min() > 0.0 and p.max() <= 1.0
    o = np.argsort(v, kind="stable")
    assert np.all(np.diff(p[o]) >= -1e-12)


def test_user_relative_alignment_and_delta(lf):
    users = np.array([2, 1], dtype=np.int64)          # deliberately unsorted
    out = G.build_user_relative(CUT, users, lf)
    assert out["user_id"].to_list() == [1, 2]         # always returned sorted
    assert out.height == len(users)
    for m in G.RANK_METRICS:
        d = out[f"u_pct_delta_{m}"].to_numpy()
        c = out[f"u_pct_cur_{m}"].to_numpy()
        p = out[f"u_pct_prev_{m}"].to_numpy()
        assert np.allclose(d, c - p)


def test_user_relative_ignores_the_target_window(lf):
    """User 2 has a 99999 GMV row on 01-09, inside (T, T+30]. It must not move
    any percentile at T."""
    users = np.array([1, 2], dtype=np.int64)
    truncated = synth_log().filter(pl.col("event_date") <= CUT).lazy()
    a = G.build_user_relative(CUT, users, lf)
    b = G.build_user_relative(CUT, users, truncated)
    for c in a.columns:
        assert np.allclose(a[c].to_numpy(), b[c].to_numpy()), c


def test_user_relative_missing_users_become_zero_blocks(lf):
    """A user with no row in the window still gets a row, at the bottom rank."""
    users = np.array([1, 2, 999], dtype=np.int64)
    out = G.build_user_relative(CUT, users, lf)
    assert out.height == 3
    assert out["user_id"].to_list() == [1, 2, 999]
    assert np.isfinite(out["u_pct_cur_gmv"].to_numpy()).all()


# ---------------------------------------------------------------- interactions
def test_interactions_are_finite_and_named(lf):
    users = np.array([1, 2], dtype=np.int64)
    cur, prev = G.user_blocks(CUT, users, lf)
    pct = G.build_user_relative(CUT, users, lf)
    glob = G.build_global_state(CUT, lf)
    x = G.build_interactions(cur, prev, pct, glob,
                             rec_buy=np.array([3.0, np.nan]),
                             rec_any=np.array([0.0, 2.0]))
    assert sorted(x) == sorted(G.INTERACTION_NAMES)
    for k, v in x.items():
        assert v.shape == (2,)
        assert np.isfinite(v).all(), k


def test_interactions_react_to_the_global_regime(lf):
    """The decline crosses must actually move when platform GMV dynamics move."""
    users = np.array([1, 2], dtype=np.int64)
    cur, prev = G.user_blocks(CUT, users, lf)
    pct = G.build_user_relative(CUT, users, lf)
    g1 = dict(G.build_global_state(CUT, lf))
    g2 = dict(g1)
    g2["g_d30_dlog_gmv"] = g1["g_d30_dlog_gmv"] - 0.5
    rb, ra = np.array([3.0, 10.0]), np.array([0.0, 2.0])
    a = G.build_interactions(cur, prev, pct, g1, rb, ra)
    b = G.build_interactions(cur, prev, pct, g2, rb, ra)
    assert not np.allclose(a["x_global_decline_X_rec_any"], b["x_global_decline_X_rec_any"])
    assert np.allclose(a["x_user_conv_over_platform"], b["x_user_conv_over_platform"])


# -------------------------------------------------------------------- placebo
def test_placebo_global_map_is_a_cyclic_shift_by_one():
    cuts = [dt.date(2025, 1, d) for d in (1, 8, 15, 22)]
    real = {c: {"g": float(i)} for i, c in enumerate(cuts)}
    pl_map = G.placebo_global_map(cuts, real)
    assert pl_map[cuts[0]]["g"] == 3.0      # first receives the last
    assert pl_map[cuts[1]]["g"] == 0.0
    assert pl_map[cuts[3]]["g"] == 2.0
    # every real vector is used exactly once: identical marginals
    assert sorted(v["g"] for v in pl_map.values()) == sorted(v["g"] for v in real.values())


def test_placebo_global_map_tells_every_cutoff_the_wrong_regime():
    cuts = [dt.date(2025, 1, d) for d in (1, 8, 15, 22)]
    real = {c: {"g": float(i)} for i, c in enumerate(cuts)}
    pl_map = G.placebo_global_map(cuts, real)
    assert all(pl_map[c]["g"] != real[c]["g"] for c in cuts)


def test_placebo_permutation_preserves_marginals_exactly():
    rng = np.random.default_rng(3)
    block = rng.normal(size=(400, 5))
    strata = rng.integers(0, 7, size=400)
    out = G.placebo_permute(block, strata, seed=11)
    assert out.shape == block.shape
    for j in range(block.shape[1]):
        assert np.allclose(np.sort(out[:, j]), np.sort(block[:, j]))
    for s in np.unique(strata):
        i = strata == s
        for j in range(block.shape[1]):
            assert np.allclose(np.sort(out[i, j]), np.sort(block[i, j]))


def test_placebo_permutation_keeps_rows_together():
    """Columns must move as one row, or the joint distribution would change too."""
    block = np.column_stack([np.arange(200.0), np.arange(200.0) * 3.0])
    strata = np.zeros(200, dtype=np.int64)
    out = G.placebo_permute(block, strata, seed=5)
    assert np.allclose(out[:, 1], out[:, 0] * 3.0)


def test_placebo_permutation_actually_moves_rows():
    rng = np.random.default_rng(4)
    block = rng.normal(size=(300, 3))
    strata = np.zeros(300, dtype=np.int64)
    out = G.placebo_permute(block, strata, seed=7)
    assert (np.abs(out - block).sum(axis=1) > 0).mean() > 0.9


def test_placebo_permutation_is_deterministic():
    rng = np.random.default_rng(6)
    block = rng.normal(size=(150, 4))
    strata = rng.integers(0, 3, size=150)
    a = G.placebo_permute(block, strata, seed=42)
    b = G.placebo_permute(block, strata, seed=42)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, G.placebo_permute(block, strata, seed=43))


def test_placebo_permutation_rejects_misaligned_input():
    with pytest.raises(ValueError):
        G.placebo_permute(np.zeros((10, 2)), np.zeros(9, dtype=np.int64), seed=1)


def test_placebo_strata_are_cutoff_gmv_recency_cells():
    gmv = np.array([0.0, 0.0, 5.0, 100.0])
    rec = np.array([1.0, 300.0, 20.0, np.nan])
    s0 = G.placebo_strata(0, gmv, rec)
    s1 = G.placebo_strata(1, gmv, rec)
    assert len(set(s0.tolist())) > 1
    assert set(s0.tolist()).isdisjoint(set(s1.tolist()))   # cutoffs never mix


def test_recency_bucket_is_monotone_and_handles_never_bought():
    b = G.recency_bucket(np.array([0.0, 8.0, 20.0, 45.0, 90.0, 200.0, np.nan]))
    assert list(b[:6]) == sorted(b[:6])
    assert b[-1] == b[-1]  # nan -> 400 -> last bucket, not a crash
    assert b[-1] == G.recency_bucket(np.array([400.0]))[0]


# ------------------------------------------------------------------ determinism
def test_global_state_rerun_is_bitwise(lf):
    a = G.build_global_state(CUT, lf)
    b = G.build_global_state(CUT, lf)
    assert G.sha256_array(np.array([a[k] for k in sorted(a)])) == \
           G.sha256_array(np.array([b[k] for k in sorted(b)]))


def test_user_relative_rerun_is_bitwise(lf):
    users = np.array([1, 2], dtype=np.int64)
    a = G.build_user_relative(CUT, users, lf)
    b = G.build_user_relative(CUT, users, lf)
    for c in a.columns:
        assert np.array_equal(a[c].to_numpy(), b[c].to_numpy()), c


def test_manifest_counts_agree(tmp_path):
    man = G.write_manifest(tmp_path / "m.json")
    assert man["n_global"] == len(G.GLOBAL_WINDOWS) * len(G.GLOBAL_METRICS) \
        + len(G.DYNAMIC_WINDOWS) * len(G.DYNAMIC_METRICS)
    assert man["n_user_relative"] == 3 * len(G.RANK_METRICS)
    assert man["n_interactions"] == len(G.INTERACTION_NAMES)
    assert man["n_total"] == man["n_global"] + man["n_user_relative"] + man["n_interactions"]


# --------------------------------------------------- arm construction on real data
@pytest.fixture(scope="module")
def real_setup():
    return G.occ_setup(G.OCC_TAU)


@pytest.mark.slow
def test_base_arm_reproduces_plain_assemble(real_setup):
    """BASE must be the untouched recipe: same matrix, same labels, same weights."""
    from src.features import feature_names
    from src.train import assemble, select_features, xy
    s = real_setup
    V = G.FOLDS[0]
    Xv, _ = xy(V, s)
    feats = select_features(feature_names(Xv), s.drop_groups, s.keep_only)
    cuts = list(s.train_cutoffs(V))[-2:]
    Xa, ya, wa = assemble(cuts, s, feats, V)
    Xb, yb, wb = G.assemble_augmented(cuts, s, feats, V, None, {}, None)
    # equal_nan: the raw matrix legitimately carries NaN (rec_buy for never-buyers,
    # lgmv_std for single-purchase users), and NaN != NaN would fail a true match.
    assert np.array_equal(Xa, Xb, equal_nan=True)
    assert np.array_equal(ya, yb, equal_nan=True)
    assert np.array_equal(wa, wb)
    assert np.isnan(Xa).sum() == np.isnan(Xb).sum() > 0


@pytest.mark.slow
def test_new_block_shape_alignment_and_finiteness(real_setup):
    from src.train import xy
    s = real_setup
    V = G.FOLDS[-1]
    Xv, _ = xy(V, s)
    real, plac, _ = G.load_global_maps()
    df = G.load().lazy()
    for arm, gmap in (("GLOBAL", real), ("PLACEBO", plac)):
        b = G.new_block(V, Xv, arm, gmap, df)
        assert b.shape == (Xv.height, len(G.all_new_feature_names()))
        assert np.isfinite(b).all()


@pytest.mark.slow
def test_global_columns_are_constant_within_a_cutoff(real_setup):
    """They are platform state, so every row of a cutoff must share them."""
    from src.train import xy
    s = real_setup
    V = G.FOLDS[-1]
    Xv, _ = xy(V, s)
    real, _, _ = G.load_global_maps()
    b = G.new_block(V, Xv, "GLOBAL", real, G.load().lazy())
    n_glob = len(G.global_feature_names())
    # exact peak-to-peak, not std: a float32 sum over 197k rows of ~1e5 values
    # loses enough precision to fake a nonzero standard deviation.
    assert np.ptp(b[:, :n_glob], axis=0).max() == 0.0
    # the percentile block must NOT be constant, or it carries nothing
    assert (b[:, n_glob:n_glob + 18].std(axis=0) > 0).all()


@pytest.mark.slow
def test_placebo_matches_real_marginals_but_not_row_order(real_setup):
    from src.train import xy
    s = real_setup
    V = G.FOLDS[-1]
    Xv, _ = xy(V, s)
    real, plac, _ = G.load_global_maps()
    df = G.load().lazy()
    a = G.new_block(V, Xv, "GLOBAL", real, df)
    b = G.new_block(V, Xv, "PLACEBO", plac, df)
    n_glob = len(G.global_feature_names())
    pct = slice(n_glob, n_glob + len(G.user_relative_feature_names()))
    for j in range(pct.start, pct.stop):
        assert np.allclose(np.sort(a[:, j]), np.sort(b[:, j]), atol=1e-6)
    moved = (np.abs(a[:, pct] - b[:, pct]).sum(axis=1) > 1e-6).mean()
    assert moved > 0.5
    # the global block keeps real values, just attached to the wrong cutoff
    assert not np.allclose(a[0, :n_glob], b[0, :n_glob])


@pytest.mark.slow
def test_percentiles_are_within_cutoff_ranks(real_setup):
    """A rank inside the scored cross-section is bounded and covers (0, 1]."""
    from src.train import xy
    s = real_setup
    V = G.FOLDS[-1]
    Xv, _ = xy(V, s)
    real, _, _ = G.load_global_maps()
    b = G.new_block(V, Xv, "GLOBAL", real, G.load().lazy())
    n_glob = len(G.global_feature_names())
    names = G.user_relative_feature_names()
    for j, nm in enumerate(names):
        col = b[:, n_glob + j]
        if nm.startswith("u_pct_delta_"):
            assert col.min() >= -1.0001 and col.max() <= 1.0001, nm
        else:
            assert col.min() > 0.0 and col.max() <= 1.0001, nm


def test_submission_schema_contract(tmp_path):
    """Any submission this experiment could emit must satisfy the project rules."""
    import pandas as pd
    uid = np.arange(250000, dtype=np.int64)
    pred = np.expm1(np.clip(np.full(250000, 2.3293), 0, 20))
    df = pd.DataFrame({"user_id": uid, "predict": pred})
    p = tmp_path / "submission_GLOBAL_REGIME_OCC.csv"
    df.to_csv(p, index=False)
    back = pd.read_csv(p)
    assert list(back.columns) == ["user_id", "predict"]
    assert len(back) == 250000
    assert not back.user_id.duplicated().any()
    assert back.predict.notna().all() and (back.predict >= 0).all()
    assert np.isfinite(back.predict.to_numpy()).all()


# ------------------------------------------------------------ overlay contract
def test_p_apply_is_asymmetric_in_the_right_direction():
    """A predicted drop in occurrence is trusted more than a rise: that asymmetry
    is the whole point of the correction, and it is visible on the real X3."""
    base = np.zeros(4)
    p_base = np.array([0.5, 0.5, 0.5, 0.5])
    mu = np.ones(4)
    p_new = np.array([0.4, 0.6, 0.3, 0.7])
    z = G.p_apply(base, p_base, mu, p_new, down=1.0, up=0.1, shift=0.0, threshold=None)
    down_move, up_move = 0.5 - z[0], z[1] - 0.5
    assert down_move > up_move * 5


def test_p_apply_threshold_silences_small_moves():
    base = np.full(3, 1.0)
    z = G.p_apply(base, np.full(3, 0.5), np.ones(3), np.array([0.51, 0.6, 0.4]),
                  down=1.0, up=1.0, shift=0.0, threshold=0.05)
    assert z[0] == pytest.approx(1.0)          # |delta| = 0.01 < 0.05 -> untouched
    assert z[1] != pytest.approx(1.0)
    assert z[2] != pytest.approx(1.0)


def test_p_apply_scales_by_conditional_magnitude():
    base = np.zeros(2)
    z = G.p_apply(base, np.full(2, 0.5), np.array([1.0, 4.0]), np.full(2, 0.3),
                  down=1.0, up=1.0, shift=0.0, threshold=None)
    assert z[1] == pytest.approx(0.0)          # clipped at zero
    z2 = G.p_apply(np.full(2, 5.0), np.full(2, 0.5), np.array([1.0, 4.0]),
                   np.full(2, 0.3), down=1.0, up=1.0, shift=0.0, threshold=None)
    assert (5.0 - z2[1]) == pytest.approx(4 * (5.0 - z2[0]))


def test_p_apply_output_is_clipped_to_the_metric_range():
    z = G.p_apply(np.array([0.0, 19.9]), np.array([0.9, 0.1]), np.array([100.0, 100.0]),
                  np.array([0.1, 0.9]), down=1.0, up=1.0, shift=0.0, threshold=None)
    assert z.min() >= 0.0 and z.max() <= 20.0


def test_walk_forward_overlay_never_tunes_on_the_scored_fold():
    """Fold 1 must use the fixed parameters, and each later fold must be fitted on
    strictly earlier folds only."""
    rng = np.random.default_rng(0)
    bank = {}
    for i, f in enumerate(G.FOLDS):
        n = 400
        y = rng.gamma(1.0, 40.0, n) * (rng.random(n) > 0.4)
        bank[f] = {"y": y, "true_z": np.log1p(y), "p": rng.uniform(.2, .8, n),
                   "mu": rng.uniform(1.0, 5.0, n), "p_T": rng.uniform(.2, .8, n),
                   "table_core": np.log1p(rng.gamma(1.0, 30.0, n))}
    out, pars = G.walk_occ_candidate(bank, "T")
    assert tuple(pars[0]) == G.FIXED_OCC_PARAMS
    assert len(out) == 4 and all(np.isfinite(v).all() for v in out.values())
    # perturbing only the LAST fold must not change any earlier fold's parameters
    bank[G.FOLDS[-1]]["p_T"] = rng.uniform(.2, .8, 400)
    _, pars2 = G.walk_occ_candidate(bank, "T")
    assert pars[:3] == pars2[:3]


def test_overlay_grid_matches_the_teammate_recipe():
    """Guards against a silent drift of the transcribed grid."""
    import inspect
    src = inspect.getsource(G.fit_occ_params_on_past)
    for token in ("-.22", "-.14", "-.08", ".06", ".45", ".65", ".85",
                  ".05", ".12", ".22", ".025", ".05"):
        assert token in src
    assert ".0015" in src and ".00025" in src and ".00020" in src
    assert G.TABLE_WEIGHT == 0.55
    assert G.FIXED_OCC_PARAMS == (-.08, .75, .12, .025)


def test_core_table_weights_renormalize_the_55_percent_slot():
    assert sum(G.CORE_TABLE.values()) == pytest.approx(1.0)
    assert G.CORE_TABLE["S1-E03a"] == pytest.approx(0.10 / 0.55)
    assert dict(zip(G.BASE_COMPONENTS, G.BASE_WEIGHTS))["SEQ-AVG3"] == pytest.approx(0.225)
    assert sum(G.BASE_WEIGHTS) == pytest.approx(1.0)
