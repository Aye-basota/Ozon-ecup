"""D3A-AVG3: усреднение сидов 42/43/44 `SEQ-D3A` В ЛОГ-ПРОСТРАНСТВЕ.

`src.seq avg` не подходит напрямую: его схема имён — `{src}-S{seed}-{tag}`, а
сиды 43/44 из `exp_030c` считались на разных арендованных картах и лежат под
`SEQ-D3A-G1-S43-*` / `SEQ-D3A-G2-S44-*`. Переименовывать нельзя: локальное имя
`SEQ-D3A-S43-V0918` уже занято ДРУГИМ прогоном того же сида (`exp_030b`, eager),
и перезапись затёрла бы наблюдение.

Логика усреднения побитово повторяет `src.seq.cmd_avg`: среднее z (то есть
log1p), а не сырого GMV.

Запуск: PYTHONPATH=. python research/strategies/results/MIX9/build_d3a_avg3.py
"""
from __future__ import annotations

import numpy as np

from src.config import VAL_FOLDS_S1
from src.merge_oof import auc_positive, load_parts, merge_arrays
from src.report import format_report, save_report
from src.tracking import load_oof, save_oof

VARIANTS = {
    # приём: `--depth-aug 0.5`
    "SEQ-D3A-AVG3": {42: "SEQ-D3A-S42", 43: "SEQ-D3A-G1-S43", 44: "SEQ-D3A-G2-S44"},
    # парная база тех же 12 прогонов: тот же сид, та же машина, без `--depth-aug`.
    # Нужна как контроль среды: сиды 43/44 считались на A10, а `SEQ-AVG3` (exp_026)
    # целиком локальный, и абсолютные уровни машин расходятся на 0.0010-0.0015.
    "SEQ-D3A-BASE-AVG3": {42: "SEQ-D3A-BASE-S42", 43: "SEQ-D3A-G1-BASE-S43",
                          44: "SEQ-D3A-G2-BASE-S44"},
}


def build(OUT: str, SRC: dict) -> None:
    tags = [f"V{d.strftime('%m%d')}" for d in VAL_FOLDS_S1]
    for t in tags:
        ds = [load_oof(f"{p}-{t}") for p in SRC.values()]
        u0 = np.asarray(ds[0]["user_id"])
        for d in ds[1:]:
            assert np.array_equal(np.asarray(d["user_id"]), u0), f"{t}: разные строки"
            assert np.allclose(d["y"], ds[0]["y"]), f"{t}: разные таргеты"
        z = np.mean([d["z"] for d in ds], axis=0)
        save_oof(f"{OUT}-{t}", u0, ds[0]["cutoff"], z, ds[0]["y"])
        spread = np.mean([np.var(d["z"] - z) for d in ds])
        print(f"{t}: усреднено {len(ds)} сидов, Var(z_i - z_avg) = {spread:.5f}")

    parts = [f"{OUT}-{t}" for t in tags]
    uid, cut, z, y = load_parts(parts)
    rep = merge_arrays(uid, cut, z, y)
    print(format_report(rep))
    print(f"  AUC(1[y>0]) = {auc_positive(y, z):.5f}")
    save_oof(OUT, uid, cut, z, y)
    save_report(OUT, rep, extra=dict(description="EXP-030c: log-space avg сидов 42/43/44",
                                     parts=parts, seeds=sorted(SRC)))
    print(f"\nOOF: artifacts/oof_{OUT}.npz")


def merge_seed(name: str) -> None:
    """Склейка 4 пофолдовых OOF одного сида в один — для пер-сидовых замеров.

    Сиды 43/44 считались пофолдово на арендованных картах, и склеенного OOF у них
    нет; `src.seq merge` тоже опирается на схему имён `{exp}-V{MMDD}`, которая
    здесь совпадает, но пишет ещё и отчёт с описанием — этого достаточно.
    """
    tags = [f"V{d.strftime('%m%d')}" for d in VAL_FOLDS_S1]
    uid, cut, z, y = load_parts([f"{name}-{t}" for t in tags])
    rep = merge_arrays(uid, cut, z, y)
    save_oof(name, uid, cut, z, y)
    print(f"{name}: склеено 4 фолда, wCV={rep['wcv']:.5f}")


def main() -> None:
    import sys
    todo = sys.argv[1:] or list(VARIANTS)
    for name in todo:
        if name in VARIANTS:
            print(f"\n===== {name} =====")
            build(name, VARIANTS[name])
        else:
            merge_seed(name)


if __name__ == "__main__":
    main()
