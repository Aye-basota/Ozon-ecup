import json
from pathlib import Path

import numpy as np
import torch

from src.buyctrl_det import (ARMS, EPOCHS, EXP_NUM, LAMBDA_AUX, SEEDS,
                             OUT, ARMS_DIR, BASE_DIR, arm_name, base_name,
                             build_aux_head, decision_rule, forward_outputs,
                             label_plan_paths, sha256_array,
                             shuffle_within_cutoff)
from src.config import SEED
from src.seq import build_model


def test_preregistered_configuration():
    assert SEEDS == (SEED, SEED + 1, SEED + 2)
    assert EPOCHS == 4
    assert LAMBDA_AUX == 0.1
    assert ARMS == ("BUYTRUE", "BUYSHUF")
    assert EXP_NUM == 45


def test_shuffle_is_strictly_within_cutoff_and_preserves_prevalence():
    ci = np.repeat(np.arange(4), [101, 109, 113, 127]).astype(np.int16)
    labels = np.concatenate([
        np.r_[np.ones(31, np.uint8), np.zeros(70, np.uint8)],
        np.r_[np.ones(47, np.uint8), np.zeros(62, np.uint8)],
        np.r_[np.ones(71, np.uint8), np.zeros(42, np.uint8)],
        np.r_[np.ones(83, np.uint8), np.zeros(44, np.uint8)],
    ])
    shuffled = shuffle_within_cutoff(labels, ci, SEED)
    replay = shuffle_within_cutoff(labels, ci, SEED)
    assert np.array_equal(shuffled, replay)
    assert not np.array_equal(shuffled, labels)
    for k in np.unique(ci):
        mask = ci == k
        assert int(shuffled[mask].sum()) == int(labels[mask].sum())


def test_auxiliary_head_is_linear_and_prevalence_initialized():
    p = 0.61
    head = build_aux_head(192, p)
    assert tuple(head.weight.shape) == (1, 192)
    assert torch.count_nonzero(head.weight) == 0
    assert torch.allclose(torch.sigmoid(head.bias), torch.tensor([p]))


def test_auxiliary_head_does_not_enter_direct_prediction():
    torch.manual_seed(SEED)
    cfg = dict(hidden=8, blocks=2, kernel=3, dropout=0.0, z0=2.0)
    model = build_model(cfg).eval()
    aux = build_aux_head(24, 0.6).eval()
    x = torch.randn(5, 17, 31)
    z1, a1 = forward_outputs(model, aux, x)
    with torch.no_grad():
        aux.weight.fill_(3.0)
        aux.bias.fill_(-7.0)
    z2, a2 = forward_outputs(model, aux, x)
    assert torch.equal(z1, z2)
    assert not torch.equal(a1, a2)


def test_decision_boundaries_are_preregistered():
    assert decision_rule(-0.0008, 2, -0.0004)[0] == "PASS"
    assert decision_rule(-0.00029, 3, -0.001)[0] == "FAIL"
    assert decision_rule(-0.0005, 3, -0.001)[0] == "INCONCLUSIVE"
    assert decision_rule(-0.0008, 1, -0.001)[0] == "INCONCLUSIVE"
    assert decision_rule(-0.0008, 3, -0.0001)[0] == "INCONCLUSIVE"


def test_completed_artifacts_are_strictly_paired_when_present():
    analysis_path = OUT / "analysis.json"
    if not analysis_path.exists():
        return
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    assert analysis["decision"] == "FAIL"
    assert analysis["promote_other_folds"] is False
    assert analysis["scope"]["leaderboard_submission_created"] is False
    for seed in SEEDS:
        plan = np.load(label_plan_paths(seed)[0], allow_pickle=False)
        assert len(plan["val_y"]) == 197_379
        assert int(plan["labels_true"].sum()) == int(plan["labels_shuf"].sum())
        true = json.loads((ARMS_DIR / arm_name(seed, "BUYTRUE") / "result.json").read_text())
        shuf = json.loads((ARMS_DIR / arm_name(seed, "BUYSHUF") / "result.json").read_text())
        base = json.loads((BASE_DIR / base_name(seed) / "result.json").read_text())
        assert true["initial"] == shuf["initial"]
        assert true["batch_index_arrays"] == shuf["batch_index_arrays"]
        assert true["n_steps"] == shuf["n_steps"] == base["n_steps"] == 19_368
        assert true["validation_order_sha256"] == shuf["validation_order_sha256"]
        assert [x["step"] for x in true["snapshots"]] == [x["step"] for x in shuf["snapshots"]]
        assert [x["rng_sha256"] for x in true["snapshots"]] == [
            x["rng_sha256"] for x in shuf["snapshots"]]
        for result in (base, true, shuf):
            z = np.load(Path(result["prediction"]["file"]))
            assert sha256_array(z) == result["prediction"]["sha256"]
            assert result["prediction"]["inference_uses_auxiliary_head"] is False
