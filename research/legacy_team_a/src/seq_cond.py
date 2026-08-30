"""EXP-032 (S04) — conditional intensity head поверх ЗАМОРОЖЕННОГО SEQ-энкодера.

Вопрос эксперимента ровно один (`experiments/EXP_032_S04_conditional_fresh_seq.md`):

    даёт ли свежая supervision с поздних cutoff'ов `2025-10-22..2026-01-14`
    дополнительный сигнал о ВЕЛИЧИНЕ покупки, не затрагивая экстенсив?

Почему это вообще законно. `exp_028` показал: у всех `T >= 2025-10-17` панель
отобрана по будущей активности, поэтому `P(активность) = 0.98..1.00` вместо
естественных ~0.90 — отравлена ВЕРОЯТНОСТЬ. Правило отбора не смотрит ни на
покупку, ни на её величину, значит на строках `y > 0` ограничение не связывающее
(`STRATEGY_04` §2-3). В лог-пространстве разложение точное:

    z = log1p(y),  z = 0 при y = 0   =>   E[z|x] = P(y>0|x) * E[z | y>0, x]

Первый множитель обучается ТОЛЬКО на CLEAN, второй — на CLEAN или CLEAN+EXTRA.

## Что здесь зафиксировано и почему

* **Энкодер заморожен и обучен только на CLEAN.** Берётся готовый чекпойнт фолда
  (`SEQ-D3A-BASE-S42-V1016` = штатный BASE seed 42, aug=none, epochs 4). Через
  него EXTRA не может изменить представление: эмбеддинги считаются один раз в
  `torch.no_grad()`, градиента в энкодер нет по построению, контрольная сумма
  параметров сверяется до и после.
* **Расщепление по пользователям.** EXTRA-cutoff'ы лежат в БУДУЩЕМ относительно
  любого фолда, поэтому строки EXTRA дают только пользователи группы B, а
  метрика считается только на группе A (`STRATEGY_04`, «Training»). Канал, ради
  которого это делается, — корреляция таргетов ОДНОГО человека во времени
  (`N9`: 0.498 на сдвиге 60 дней); он работает всегда, независимо от того,
  пересекаются ли окна таргета. На поздних фолдах (10-16, 10-02) окна ещё и
  пересекаются буквально, на ранних — нет, но окно ПРИЗНАКОВ строки EXTRA
  накрывает валидационное окно фолда на всех четырёх. Контроль `COND-CLEAN`
  считается на тех же строках группы A, иначе варианты несравнимы.
* **Центрирование цели по cutoff'у обязательно.** Уровень интенсива сезонно
  дрейфует, а в EXTRA сидит новогодний пик (`e09b`, `e05`). Голова учит ФОРМУ
  `z+ - c(T)`, уровень восстанавливается средним `c(T)` по CLEAN.
* **Глубина входа EXTRA обрезается до 289** — это боевая политика теста
  (`exp_027`, «Не повторять»). Без обрезки поздние cutoff'ы дают `avail = 1` на
  всём окне и глубину до 378 дней: непрожитый энкодером режим, который смешал бы
  вопрос об EXTRA с вопросом о глубине.
* **Равный шаговый бюджет у всех вариантов.** Иначе «больше данных» означало бы
  ещё и «больше шагов оптимизатора», и выигрыш нельзя было бы отнести к данным.
* **Контроль объёма `COND-VOL`** (вариант C из `STRATEGY_04`): к CLEAN
  добавляется столько же строк, сколько дал бы EXTRA, но из САМЫХ РАННИХ CLEAN
  cutoff'ов. Отделяет «данные ближе к тесту» от «данных просто больше».

Запуск:
  python -m src.seq_cond audit                      # только анти-лукап, без GPU
  python -m src.seq_cond pilot --ckpt SEQ-D3A-BASE-S42-V1016
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import time

import numpy as np

from src.config import ARTIFACTS, CORRIDOR_END, DATA_END, SEED, TARGET_DAYS
from src.features import panel_users
from src.seq import (N_CH_STORED, day_index, fold_cutoffs, gather, load_ckpt, log,
                     panel, target_at, user_rows)
from src.validation import bias_z, calibrate, rmsle_z

# 13 физически наблюдаемых поздних cutoff'ов, шаг 7 назад от 2026-01-14.
# У самого позднего таргет-окно (01-14, 02-13] заканчивается ровно на границе
# данных, поэтому относительно теста лукапа нет по построению.
EXTRA_CUTOFFS = [DATA_END - dt.timedelta(days=30 + 7 * k) for k in range(13)][::-1]

EMB_DIM_MULT = 3          # pooled = [last, mean, max]
POS_ONLY = "y>0"
SEED_VAR_FLOOR = 0.00712  # проектный пол Var(dz) по сидам (exp_016)


# --------------------------------------------------------------- расщепление панели
def user_group(uids: np.ndarray) -> np.ndarray:
    """0 = группа A (на ней метрика), 1 = группа B (только она даёт строки EXTRA).

    Хеш splitmix64, а не `user_id % 2`: идентификаторы могут нести структуру
    (порядок регистрации), и делить по младшему биту значило бы делить по ней.
    Функция чистая и детерминированная — тот же пользователь всегда в той же
    группе на всех cutoff'ах, иначе расщепление не закрывало бы канал.
    """
    x = np.asarray(uids).astype(np.uint64)
    x = x + np.uint64(0x9E3779B97F4A7C15)
    z = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    z = z ^ (z >> np.uint64(31))
    return (z & np.uint64(1)).astype(np.int8)


def audit_extra(V: dt.date) -> dict:
    """Анти-лукап EXTRA: печатает и возвращает проверки, GPU не нужен."""
    out = {}
    assert all(T > CORRIDOR_END for T in EXTRA_CUTOFFS), "EXTRA обязан лежать вне коридора"
    assert all(T > V for T in EXTRA_CUTOFFS), "EXTRA обязан лежать в будущем от фолда"
    last = EXTRA_CUTOFFS[-1] + dt.timedelta(days=TARGET_DAYS)
    assert last <= DATA_END, f"таргет-окно {EXTRA_CUTOFFS[-1]} выходит за данные ({last})"
    out["n_extra"] = len(EXTRA_CUTOFFS)
    out["extra_first"] = EXTRA_CUTOFFS[0].isoformat()
    out["extra_last"] = EXTRA_CUTOFFS[-1].isoformat()
    out["extra_target_end"] = last.isoformat()
    out["data_end"] = DATA_END.isoformat()
    out["val_target_window"] = f"({V}, {V + dt.timedelta(days=TARGET_DAYS)}]"
    out["overlap_with_val_window"] = bool(
        EXTRA_CUTOFFS[0] < V + dt.timedelta(days=TARGET_DAYS))
    u = panel_users(V, 3)["user_id"].to_numpy()
    g = user_group(u)
    out["val_users"] = int(len(u))
    out["val_share_group_B"] = float(g.mean())
    return out


# ------------------------------------------------------------------------ эмбеддинги
def _pool(h):
    import torch
    return torch.cat([h[:, -1], h.mean(1), h.amax(1)], dim=1)


def embed(model, cfg, dev, T: dt.date, rows: np.ndarray, depth_clip=None,
          batch: int = 1024) -> np.ndarray:
    """(n, 3H) fp16 — ровно тот pooled-вектор, который видит боевая голова.

    `model.encode` плюс та же конкатенация `[last, mean, max]`, что в `TCN.forward`.
    Всё под `no_grad`: EXTRA физически не может изменить энкодер.
    """
    import torch
    model.eval()
    _, _, _, sc_np = panel()
    sc = torch.from_numpy(sc_np).to(dev).view(1, N_CH_STORED, 1)
    H = cfg["hidden"] * EMB_DIM_MULT
    out = np.empty((len(rows), H), np.float16)
    with torch.no_grad():
        for i in range(0, len(rows), batch):
            x = gather(T, rows[i:i + batch], depth_clip=depth_clip)
            t = torch.from_numpy(x).to(dev).permute(0, 2, 1).contiguous().float()
            t[:, :N_CH_STORED] *= sc
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                e = _pool(model.encode(t))
            out[i:i + batch] = e.float().cpu().numpy().astype(np.float16)
    return out


def collect(model, cfg, dev, cuts, blocks: int, keep=None, group_keep=None,
            depth_clip=None, tag: str = "", cache: str | None = None):
    """Эмбеддинги и таргет по списку cutoff'ов.

    `keep="y>0"` — оставить только покупающих (интенсивная голова);
    `group_keep` — оставить только эту группу расщепления (для EXTRA это 1).
    Фильтры применяются ДО инференса, поэтому лишние строки даже не считаются.
    """
    if cache:
        fx, fm = ARTIFACTS / f"{cache}_X.npy", ARTIFACTS / f"{cache}_meta.npz"
        if fx.exists() and fm.exists():
            m = np.load(fm)
            if len(m["cuts"]) == len(cuts) and all(
                    str(a) == b.isoformat() for a, b in zip(m["cuts"], cuts)):
                log(f"  [{tag}] кэш: {fx.name} ({len(m['z']):,} строк)")
                return np.load(fx), m["z"], m["u"], m["c"]
            log(f"  [{tag}] кэш {fx.name} собран на другой сетке cutoff'ов — пересчёт")
    E, Y, U, C = [], [], [], []
    for k, T in enumerate(cuts):
        u = panel_users(T, blocks)["user_id"].to_numpy()
        if group_keep is not None:
            u = u[user_group(u) == group_keep]
        r = user_rows(u)
        y = target_at(T, r)
        share = float((y > 0).mean())
        if keep == POS_ONLY:
            m = y > 0
            u, r, y = u[m], r[m], y[m]
        E.append(embed(model, cfg, dev, T, r, depth_clip=depth_clip))
        Y.append(np.log1p(y).astype(np.float32))
        U.append(u.astype(np.int64))
        C.append(np.full(len(u), k, np.int16))
        log(f"  [{tag}] {T} -> {len(u):,} строк, доля y>0 {share:.4f}")
    X, z, u, c = (np.concatenate(E), np.concatenate(Y), np.concatenate(U),
                  np.concatenate(C))
    if cache:
        np.save(ARTIFACTS / f"{cache}_X.npy", X)
        np.savez(ARTIFACTS / f"{cache}_meta.npz", z=z, u=u, c=c,
                 cuts=np.array([T.isoformat() for T in cuts], dtype="U10"))
        log(f"  [{tag}] кэш записан: {cache}_X.npy ({X.nbytes / 1e9:.2f} ГБ)")
    return X, z, u, c


# ----------------------------------------------------------------------------- головы
def build_head(dim: int, hidden: int, dropout: float, out_bias: float):
    """Та же форма, что боевая голова TCN: Linear -> GELU -> Dropout -> Linear."""
    from torch import nn
    net = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout),
                        nn.Linear(hidden, 1))
    nn.init.zeros_(net[-1].weight)
    nn.init.constant_(net[-1].bias, out_bias)
    return net


def fit_head(X, t, *, steps, batch, lr, wd, hidden, dropout, seed, binary, dev,
             out_bias, rows=None):
    """Обучение головы на готовых эмбеддингах. Бюджет шагов ФИКСИРОВАН извне.

    Равное число шагов у всех вариантов — обязательное условие сравнимости:
    иначе «CLEAN+EXTRA лучше» могло бы означать просто «дольше учили».

    `rows` — индексы разрешённых строк общего пула. Варианты отличаются ТОЛЬКО
    этим списком, поэтому пул не копируется (иначе три копии по ~1.4 ГБ) и
    вариант нельзя случайно обучить на чужой матрице.
    """
    import torch
    torch.manual_seed(seed)
    net = build_head(X.shape[1], hidden, dropout, out_bias).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.98))
    rng = np.random.default_rng(seed)
    Xt = torch.from_numpy(X)                    # fp16 на CPU, батч уходит на GPU
    tt = torch.from_numpy(t)
    rows = np.arange(len(X)) if rows is None else np.asarray(rows)
    lossf = (torch.nn.functional.binary_cross_entropy_with_logits if binary
             else torch.nn.functional.mse_loss)
    net.train()
    run, seen = 0.0, 0
    for s in range(steps):
        for g in opt.param_groups:
            g["lr"] = lr * min(1.0, (s + 1) / 200) * 0.5 * (1 + np.cos(np.pi * s / steps))
        idx = torch.from_numpy(rows[rng.integers(0, len(rows), batch)])
        xb = Xt[idx].to(dev).float()
        yb = tt[idx].to(dev).float()
        loss = lossf(net(xb).squeeze(1), yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        run += float(loss.detach()) * batch
        seen += batch
    net.eval()
    return net, run / seen


def head_predict(net, X, dev, batch: int = 65536, sigmoid: bool = False):
    import torch
    out = np.empty(len(X), np.float32)
    Xt = torch.from_numpy(X)
    with torch.no_grad():
        for i in range(0, len(X), batch):
            o = net(Xt[i:i + batch].to(dev).float()).squeeze(1)
            if sigmoid:
                o = torch.sigmoid(o)
            out[i:i + batch] = o.float().cpu().numpy()
    return out


# -------------------------------------------------------------------------- сегменты
def segments(V: dt.date, rows: np.ndarray) -> dict:
    """`rec_buy` и `w180_days_buy` прямо из панели GMV — те же оси, что в exp_030."""
    _, g, _, _ = panel()
    d = day_index(V)
    lo = max(0, d - 364)
    win = g[rows, lo:d + 1] > 0                       # (n, <=365) дни с покупкой
    n_days = win.shape[1]
    any_buy = win.any(1)
    last = n_days - 1 - np.argmax(win[:, ::-1], axis=1)
    rec = np.where(any_buy, n_days - 1 - last, 10 ** 4)
    w180 = win[:, max(0, n_days - 180):].sum(1)
    return dict(rec_buy=rec, w180_days_buy=w180)


def seg_masks(seg: dict) -> dict:
    rec, w180 = seg["rec_buy"], seg["w180_days_buy"]
    return {
        "rec_buy 15-60": (rec >= 15) & (rec <= 60),
        "w180_days_buy 2-15": (w180 >= 2) & (w180 <= 15),
        "пересечение": (rec >= 15) & (rec <= 60) & (w180 >= 2) & (w180 <= 15),
        "никогда не покупал": rec > 365,
        "w180_days_buy 0-1": w180 <= 1,
        "w180_days_buy 16+": w180 >= 16,
    }


def _auc(lab, s) -> float:
    lab = np.asarray(lab).astype(bool)
    s = np.asarray(s, float)
    o = np.argsort(s, kind="mergesort")
    sv = s[o]
    r = np.empty(len(s), float)
    r[o] = np.arange(1, len(s) + 1)
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            r[o[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    n1, n0 = int(lab.sum()), int((~lab).sum())
    return float((r[lab].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


# ------------------------------------------------- развёртка по сидам головы
def _run_heads(Xc, zc, pos_c, Xp, tp, heads, Xv, c_hat, seed, batch, hidden, drop,
               a, dev, p_steps, steps):
    """Полный набор голов на одном сиде. Энкодер и эмбеддинги общие — меняется
    только инициализация голов и порядок батчей, то есть ровно шум приёма."""
    pr = float(pos_c.mean())
    net_p, _ = fit_head(Xc, (zc > 0).astype(np.float32), steps=p_steps, batch=batch,
                        lr=a.lr, wd=a.wd, hidden=hidden, dropout=drop, seed=seed,
                        binary=True, dev=dev, out_bias=float(np.log(pr / (1 - pr))))
    p_val = head_predict(net_p, Xv, dev, sigmoid=True)
    zs, ms = {}, {}
    for name, idx in heads.items():
        net_m, _ = fit_head(Xp, tp, steps=steps, batch=batch, lr=a.lr, wd=a.wd,
                            hidden=hidden, dropout=drop, seed=seed, binary=False,
                            dev=dev, out_bias=0.0, rows=idx)
        mu = head_predict(net_m, Xv, dev) + c_hat
        ms[name] = mu
        zs[name] = np.maximum(p_val * np.maximum(mu, 0.0), 0.0)
    return zs, ms


def _deltas(zs, ms, yv, A) -> dict:
    pm = A & (yv > 0)
    ly = np.log1p(yv[pm])
    cal = {k: calibrate(yv[A], v[A])[1] for k, v in zs.items()}
    mu = {k: float(np.sqrt(np.mean((ly - v[pm]) ** 2))) for k, v in ms.items()}
    return dict(d_cal=cal["COND-FRESH"] - cal["COND-CLEAN"],
                d_cal_vol=cal["COND-VOL"] - cal["COND-CLEAN"],
                d_mu=mu["COND-FRESH"] - mu["COND-CLEAN"],
                cal_clean=cal["COND-CLEAN"], cal_fresh=cal["COND-FRESH"])


# ----------------------------------------------------------------------------- пилот
def cmd_audit(a):
    V = dt.date.fromisoformat(a.val)
    print(json.dumps(audit_extra(V), indent=1, ensure_ascii=False))


def cmd_pilot(a):
    model, cfg, Vc, dev = load_ckpt(a.ckpt)
    V = dt.date.fromisoformat(a.val) if a.val else Vc
    assert V == Vc, f"чекпойнт обучен на фолде {Vc}, запрошен {V}"
    for p_ in model.parameters():
        p_.requires_grad_(False)
    enc_sig = float(sum(float(p_.double().sum()) for p_ in model.parameters()))
    log(f"энкодер {a.ckpt}: фолд {Vc}, hidden={cfg['hidden']}, aug={cfg.get('aug')}, "
        f"depth_aug={cfg.get('depth_aug')}, заморожен, чек-сумма {enc_sig:.6f}")

    aud = audit_extra(V)
    log("анти-лукап EXTRA: " + json.dumps(aud, ensure_ascii=False))
    # Пересечение окон таргета есть только у поздних фолдов (10-16, 10-02).
    # Расщепление по пользователям применяется ВСЕГДА и от него не зависит:
    # канал утечки — корреляция таргетов ОДНОГО человека во времени (`N9`: 0.498
    # на сдвиге 60 дней), а она работает и без пересечения окон. Плюс признаки
    # строки EXTRA всегда содержат валидационное окно фолда целиком.
    log("  пересечение окон таргета EXTRA и валидации: "
        + ("ЕСТЬ" if aud["overlap_with_val_window"] else "НЕТ (ранний фолд); "
           "расщепление по пользователям применяется всё равно"))

    clean_cuts = fold_cutoffs(V)
    extra_cuts = list(EXTRA_CUTOFFS)
    if a.n_cutoffs:                      # только отладка, боевой прогон без него
        clean_cuts, extra_cuts = clean_cuts[-a.n_cutoffs:], extra_cuts[-a.n_cutoffs:]
        log(f"ОТЛАДКА: только последние {a.n_cutoffs} cutoff'ов каждого набора")
    log(f"CLEAN: {len(clean_cuts)} cutoff'ов {clean_cuts[0]}..{clean_cuts[-1]}")
    log(f"EXTRA: {len(extra_cuts)} cutoff'ов {extra_cuts[0]}..{extra_cuts[-1]}, "
        f"глубина входа обрезана до {a.extra_depth_clip}")

    cpre = None if a.no_cache else f"S04SEQ_emb_{a.ckpt}"
    t0 = time.time()
    Xc, zc, uc, cc = collect(model, cfg, dev, clean_cuts, 1, tag="CLEAN",
                             cache=None if cpre is None else f"{cpre}_clean")
    log(f"CLEAN: {len(zc):,} строк, доля y>0 {float((zc > 0).mean()):.4f} "
        f"[{time.time() - t0:.0f}s]")
    t0 = time.time()
    Xe, ze, ue, ce = collect(model, cfg, dev, extra_cuts, 1, keep=POS_ONLY,
                             group_keep=1, depth_clip=a.extra_depth_clip, tag="EXTRA",
                             cache=None if cpre is None else f"{cpre}_extra")
    log(f"EXTRA: {len(ze):,} положительных строк группы B [{time.time() - t0:.0f}s]")

    uv = panel_users(V, 3)["user_id"].to_numpy()
    if a.val_frac < 1.0:                 # только отладка
        uv = uv[::int(round(1 / a.val_frac))]
    rv = user_rows(uv)
    yv = target_at(V, rv)
    gv = user_group(uv)
    fv = ARTIFACTS / f"{cpre}_val_X.npy" if cpre else None
    if fv is not None and fv.exists():
        Xv = np.load(fv)
        log(f"  [VAL] кэш: {fv.name}")
    else:
        Xv = embed(model, cfg, dev, V, rv)
        if fv is not None:
            np.save(fv, Xv)
    A = gv == 0
    log(f"VAL {V}: {len(uv):,} пользователей, группа A {int(A.sum()):,} "
        f"({A.mean():.3f}), доля y>0 {(yv > 0).mean():.4f}")

    # ---- анти-лукап: ни один пользователь группы A не дал строк EXTRA
    assert len(ue) and int(user_group(ue).min()) == 1, "в EXTRA попала группа A"
    inter = np.intersect1d(np.unique(ue), uv[A])
    assert inter.size == 0, f"{inter.size} пользователей группы A нашлись в EXTRA"
    log(f"анти-лукап OK: пересечение EXTRA x (группа A валидации) = {inter.size}")

    # ---- центрирование цели по cutoff'у
    pos_c = zc > 0
    c_clean = np.array([zc[pos_c & (cc == k)].mean() for k in range(len(clean_cuts))])
    c_extra = np.array([ze[ce == k].mean() for k in range(len(extra_cuts))])
    c_hat = float(c_clean.mean())
    log(f"уровень c(T): CLEAN {c_clean.min():.4f}..{c_clean.max():.4f}, chat = {c_hat:.4f}; "
        f"EXTRA {c_extra.min():.4f}..{c_extra.max():.4f} "
        f"(макс. на {extra_cuts[int(c_extra.argmax())]})")

    batch, hidden, drop = a.batch, cfg["hidden"], a.dropout

    # ---- экстенсив: одна голова на всех вариантах, обучена ТОЛЬКО на CLEAN.
    # Считается ПЕРВОЙ, пока цела полная матрица CLEAN: дальше она режется до
    # покупающих, и полная больше не нужна.
    pr = float(pos_c.mean())
    p_steps = int(np.ceil(len(Xc) / batch)) * a.epochs
    net_p, l_p = fit_head(Xc, (zc > 0).astype(np.float32), steps=p_steps, batch=batch,
                          lr=a.lr, wd=a.wd, hidden=hidden, dropout=drop, seed=a.seed,
                          binary=True, dev=dev, out_bias=float(np.log(pr / (1 - pr))))
    p_val = head_predict(net_p, Xv, dev, sigmoid=True)
    log(f"голова P(y>0): {p_steps:,} шагов, BCE {l_p:.5f}, mean p = {p_val.mean():.4f}")

    Xcp = Xc[pos_c]
    tcp = (zc[pos_c] - c_clean[cc[pos_c]]).astype(np.float32)
    tep = (ze - c_extra[ce]).astype(np.float32)
    ci_pos = cc[pos_c]
    # `Xc` НЕ удаляется: развёртка по сидам головы переобучает и экстенсивную
    # голову, а она живёт на полной матрице CLEAN.

    n_cp, n_extra = len(tcp), len(tep)
    steps = int(np.ceil(n_cp / batch)) * a.epochs
    log(f"бюджет головы: {steps:,} шагов x batch {batch} (одинаков у всех вариантов)")

    # Общий пул: [CLEAN-положительные | EXTRA-положительные группы B]. Варианты
    # различаются только списком разрешённых строк.
    Xp = np.concatenate([Xcp, Xe])
    tp = np.concatenate([tcp, tep])
    del Xcp, Xe
    i_clean = np.arange(n_cp)
    i_extra = np.arange(n_cp, n_cp + n_extra)
    rng = np.random.default_rng(a.seed)
    early = np.flatnonzero(ci_pos < max(1, len(clean_cuts) // 3))
    heads = {
        "COND-CLEAN": i_clean,
        "COND-VOL": np.concatenate([i_clean, rng.choice(early, size=n_extra, replace=True)]),
        "COND-FRESH": np.concatenate([i_clean, i_extra]),
    }
    for k, idx in heads.items():
        log(f"  {k}: {len(idx):,} обучающих строк "
            f"(из них EXTRA {int((idx >= n_cp).sum()):,})")

    seeds = a.head_seeds or [a.seed]
    assert a.seed in seeds, "основной сид обязан входить в развёртку"
    sweep = []
    for hs in seeds:
        if hs == a.seed:
            continue
        zs, ms = _run_heads(Xc, zc, pos_c, Xp, tp, heads, Xv, c_hat, hs, batch,
                            hidden, drop, a, dev, p_steps, steps)
        sweep.append(dict(seed=hs, **_deltas(zs, ms, yv, A)))
        log(f"  сид головы {hs}: Δ RMSLE_cal FRESH−CLEAN = {sweep[-1]['d_cal']:+.5f}, "
            f"Δ RMSLE_mu = {sweep[-1]['d_mu']:+.5f}, "
            f"VOL−CLEAN = {sweep[-1]['d_cal_vol']:+.5f}")

    zmap, mumap, extra = {}, {}, {}
    for name, idx in heads.items():
        net_m, l_m = fit_head(Xp, tp, steps=steps, batch=batch, lr=a.lr, wd=a.wd,
                              hidden=hidden, dropout=drop, seed=a.seed, binary=False,
                              dev=dev, out_bias=0.0, rows=idx)
        mu = head_predict(net_m, Xv, dev) + c_hat
        mumap[name] = mu
        zmap[name] = np.maximum(p_val * np.maximum(mu, 0.0), 0.0)
        if name == "COND-CLEAN":
            # остаток CLEAN-головы на EXTRA — прямая проверка допущения §2 STRATEGY_04:
            # систематический сдвиг вниз означал бы, что правило панели затягивает
            # в {y>0} маргинальных покупателей, и тогда интенсив тоже отравлен.
            r_ex = tp[i_extra] - head_predict(net_m, Xp[i_extra], dev)
            per = [float(r_ex[ce == k].mean()) for k in range(len(extra_cuts))]
            extra["poison_resid_mean"] = float(r_ex.mean())
            extra["poison_resid_sd"] = float(r_ex.std())
            extra["poison_resid_per_cutoff"] = per
            log(f"  отравление интенсива: остаток CLEAN-головы на EXTRA "
                f"mean {r_ex.mean():+.4f}, sd {r_ex.std():.4f}, "
                f"по cutoff'ам {min(per):+.4f}..{max(per):+.4f}")
        log(f"голова {name}: train {l_m:.5f}, mean mu+chat = {mu.mean():.4f}")

    zb = None
    oof = ARTIFACTS / f"oof_{a.ckpt}.npz"
    if oof.exists():
        d = np.load(oof, allow_pickle=True)
        ub, zbv = np.asarray(d["user_id"]).astype(np.int64), np.asarray(d["z"], float)
        o = np.argsort(ub)
        pos = np.searchsorted(ub[o], uv)
        ok = bool(np.array_equal(ub[o][np.minimum(pos, len(o) - 1)], uv))
        zb = zbv[o][pos] if ok else None
        if zb is None:
            log("OOF базы не покрывает эту панель (отладочный прогон?) — BASE-1HEAD пропущен")

    def score(name, z):
        off, cal = calibrate(yv[A], z[A])
        pm = A & (yv > 0)
        return dict(variant=name, n=int(A.sum()), rmsle=rmsle_z(yv[A], z[A]),
                    rmsle_cal=cal, bias=bias_z(yv[A], z[A]), offset=off,
                    mean_z=float(z[A].mean()), rmsle_pos=rmsle_z(yv[pm], z[pm]),
                    rmsle_pos_cal=calibrate(yv[pm], z[pm])[1],
                    auc=_auc(yv[A] > 0, z[A]))

    rows_out = []
    if zb is not None:
        rows_out.append(score("BASE-1HEAD", zb))
    for name in ("COND-CLEAN", "COND-VOL", "COND-FRESH"):
        rows_out.append(score(name, zmap[name]))

    pm = A & (yv > 0)
    lyp = np.log1p(yv[pm])
    for r in rows_out:
        if r["variant"] in mumap:
            mu = mumap[r["variant"]][pm]
            r["rmsle_mu"] = float(np.sqrt(np.mean((lyp - mu) ** 2)))
            r["bias_mu"] = float(lyp.mean() - mu.mean())

    zc_, zf_ = zmap["COND-CLEAN"][A], zmap["COND-FRESH"][A]
    d_var = float(np.var(zf_ - zc_))
    ly = np.log1p(yv[A])
    corr = float(np.corrcoef(ly - zc_, ly - zf_)[0, 1])
    auc_p = _auc(yv[A] > 0, p_val[A])
    log(f"FRESH - CLEAN: Var(dz) = {d_var:.5f} ({d_var / SEED_VAR_FLOOR:.2f}x пола сидов), "
        f"corr остатков {corr:.5f}")
    log(f"AUC(y>0) по p (общая для всех вариантов) = {auc_p:.5f}")

    hdr = ["variant", "n", "rmsle", "rmsle_cal", "bias", "offset", "mean_z",
           "rmsle_pos_cal", "rmsle_mu", "bias_mu", "auc"]
    print("\n" + " | ".join(f"{h:>13}" for h in hdr))
    for r in rows_out:
        cells = []
        for h in hdr:
            v = r.get(h)
            cells.append(f"{v:>13.5f}" if isinstance(v, float) else f"{str(v):>13}")
        print(" | ".join(cells))

    base = next(r for r in rows_out if r["variant"] == "COND-CLEAN")
    fresh = next(r for r in rows_out if r["variant"] == "COND-FRESH")
    vol = next(r for r in rows_out if r["variant"] == "COND-VOL")
    print(f"\nD RMSLE_cal  FRESH - CLEAN = {fresh['rmsle_cal'] - base['rmsle_cal']:+.5f}"
          "   <- главный гейт (общий скор)")
    print(f"D RMSLE_cal  VOL   - CLEAN = {vol['rmsle_cal'] - base['rmsle_cal']:+.5f}"
          "   <- контроль объёма")
    print(f"D RMSLE_mu   FRESH - CLEAN = {fresh['rmsle_mu'] - base['rmsle_mu']:+.5f}"
          "   <- интенсив на y>0")
    print(f"D RMSLE_mu   VOL   - CLEAN = {vol['rmsle_mu'] - base['rmsle_mu']:+.5f}")

    seg = segments(V, rv)
    seg_rows = []
    for nm, m in seg_masks(seg).items():
        mm = m & A
        if mm.sum() < 100:
            continue
        a_ = calibrate(yv[mm], zmap["COND-CLEAN"][mm])[1]
        b_ = calibrate(yv[mm], zmap["COND-FRESH"][mm])[1]
        seg_rows.append(dict(segment=nm, share=float(mm.sum() / A.sum()),
                             cond_clean=a_, cond_fresh=b_, delta=b_ - a_))
    print("\nсегменты (RMSLE_cal, группа A):")
    for r in seg_rows:
        print(f"  {r['segment']:<22} доля {r['share']:.3f}  CLEAN {r['cond_clean']:.5f}  "
              f"FRESH {r['cond_fresh']:.5f}  D {r['delta']:+.5f}")

    if sweep:
        allr = [dict(seed=a.seed, d_cal=fresh["rmsle_cal"] - base["rmsle_cal"],
                     d_cal_vol=vol["rmsle_cal"] - base["rmsle_cal"],
                     d_mu=fresh["rmsle_mu"] - base["rmsle_mu"],
                     cal_clean=base["rmsle_cal"], cal_fresh=fresh["rmsle_cal"])] + sweep
        dc = np.array([r["d_cal"] for r in allr])
        dm = np.array([r["d_mu"] for r in allr])
        dv = np.array([r["d_cal_vol"] for r in allr])
        print()
        print(f"развёртка по сидам головы ({len(allr)} шт., энкодер и эмбеддинги общие):")
        for r in allr:
            print(f"  сид {r['seed']}: FRESH−CLEAN {r['d_cal']:+.5f} | "
                  f"VOL−CLEAN {r['d_cal_vol']:+.5f} | интенсив {r['d_mu']:+.5f}")
        print(f"  СРЕДНЕЕ FRESH−CLEAN = {dc.mean():+.5f} (sd {dc.std(ddof=1):.5f}), "
              f"улучшение на {int((dc < 0).sum())}/{len(dc)} сидах")
        print(f"  СРЕДНЕЕ VOL−CLEAN   = {dv.mean():+.5f} (sd {dv.std(ddof=1):.5f}) "
              "— контроль объёма")
        print(f"  СРЕДНЕЕ интенсив    = {dm.mean():+.5f} (sd {dm.std(ddof=1):.5f})")
        extra["sweep"] = allr

    out = dict(ckpt=a.ckpt, val=V.isoformat(), seed=a.seed, audit=aud, c_hat=c_hat,
               c_clean=c_clean.tolist(), c_extra=c_extra.tolist(), steps=steps,
               p_steps=p_steps, auc_p=auc_p, var_delta=d_var, corr_resid=corr,
               rows=rows_out, segments=seg_rows, n_extra_pos=int(n_extra),
               n_clean_pos=int(len(tcp)), encoder_checksum=enc_sig,
               extra_depth_clip=a.extra_depth_clip, **extra)
    p = ARTIFACTS / f"S04SEQ_{a.exp}.json"
    p.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    np.savez_compressed(
        ARTIFACTS / f"S04SEQ_{a.exp}_z.npz", uid=uv, y=yv, group=gv, p=p_val,
        **{k.replace("-", "_"): v for k, v in zmap.items()},
        **{"mu_" + k.replace("-", "_"): v for k, v in mumap.items()})
    log(f"записано: artifacts/{p.name}, artifacts/S04SEQ_{a.exp}_z.npz")

    enc_after = float(sum(float(p_.double().sum()) for p_ in model.parameters()))
    assert enc_after == enc_sig, "энкодер изменился — заморозка нарушена"
    log(f"энкодер после обучения голов: чек-сумма {enc_after:.6f} — не изменилась")


# ======================================================================== EXP-032B
# Замена переобученной головы `P(y>0)` на боевой экстенсив.
#
# `exp_032` намерил, что вся цена двухчастной пересборки (+0.00088 wCV(A)) сидит
# в экстенсиве: `COND-CLEAN` хуже одноголового `BASE-1HEAD` на всех фолдах, а
# сам приём (`FRESH − CLEAN` = −0.00128, 4/4) от этого не зависит. Здесь `μ̂`
# берётся тот же самый, а `p̂` — из уже существующей CLEAN-модели проекта.
# Энкодер не переобучается: эмбеддинги читаются из кэша `exp_032`.

# Кандидаты в `P_prod`. Все обучены ТОЛЬКО на чистом коридоре (`T + 30 <= V`),
# ни один не видел EXTRA-cutoff'ов. Порядок — приоритет при выборе основного.
PPROD_SOURCES = {
    # боевой член смеси (0.25 в SEQ-01-MIX). exp_014: экстенсив головы
    # распределения читается через p0, AUC(1-p0)=0.84689 против 0.84368 у ẑ.
    "DIST": ("PACT_dist_{v}.npz", "p_act"),
    # бинарная голова табличного S04 (`src/calval.py`): 227 боевых признаков,
    # 600 раундов, только CLEAN. Не член смеси, поэтому — контроль, а не основной.
    "S04LGB": ("S04_fold_{v}_s42.npz", "p_hat"),
    # OOF-головы `exp_024` (кросс-фиттинг по пользователям). Приём отвергнут,
    # но сами вероятности чистые и лежат на диске — берутся как контроль.
    "MHZ-B30": ("mhz_val_{v}.npz", "b30_p"),
    "MHZ-CNT": ("mhz_val_{v}.npz", "!cnt_p0"),
}
PPROD_PRIMARY = "DIST"


def load_pprod(V: dt.date, uv: np.ndarray) -> dict:
    """`P(y>0)` всех доступных CLEAN-источников, выровненные по строкам панели.

    Выравнивание — по `user_id`, а не по позиции: совпадение порядка проверяется,
    а не предполагается. Отсутствующий файл пропускается с записью в лог, потому
    что контрольные источники не обязаны быть посчитаны.
    """
    out = {}
    for name, (tpl, col) in PPROD_SOURCES.items():
        f = ARTIFACTS / tpl.format(v=V.isoformat())
        if not f.exists():
            log(f"  P_prod [{name}]: нет {f.name} — пропущен")
            continue
        d = np.load(f, allow_pickle=True)
        uid = np.asarray(d["user_id"]).astype(np.int64)
        inv, key = col.startswith("!"), col.lstrip("!")
        if "aux_cols" in d.files:                 # артефакты exp_024: колонка в матрице aux
            v = d["aux"][:, [str(c) for c in d["aux_cols"]].index(key)].astype(np.float64)
        else:
            v = np.asarray(d[key]).astype(np.float64)
        if inv:
            v = 1.0 - v
        if not np.array_equal(uid, uv):
            o = np.argsort(uid)
            pos = np.searchsorted(uid[o], uv)
            assert pos.max() < len(o) and np.array_equal(uid[o][pos], uv), (
                f"{name}: панель не совпадает с валидационной")
            v = v[o][pos]
        assert np.isfinite(v).all() and (0.0 <= v).all() and (v <= 1.0).all(), (
            f"{name}: значения вне [0, 1]")
        out[name] = v
        log(f"  P_prod [{name}]: {f.name}, mean p = {v.mean():.4f}")
    assert out, "ни одного источника P_prod на диске"
    return out


def cmd_prod(a):
    """`P_prod × μ_CLEAN` против `P_prod × μ_FRESH` на группе A одного фолда."""
    model, cfg, Vc, dev = load_ckpt(a.ckpt)
    V = dt.date.fromisoformat(a.val) if a.val else Vc
    assert V == Vc, f"чекпойнт обучен на фолде {Vc}, запрошен {V}"
    for p_ in model.parameters():
        p_.requires_grad_(False)
    enc_sig = float(sum(float(p_.double().sum()) for p_ in model.parameters()))
    log(f"энкодер {a.ckpt}: фолд {Vc}, заморожен, чек-сумма {enc_sig:.6f}")

    aud = audit_extra(V)
    log("анти-лукап EXTRA: " + json.dumps(aud, ensure_ascii=False))

    clean_cuts, extra_cuts = fold_cutoffs(V), list(EXTRA_CUTOFFS)
    cpre = f"S04SEQ_emb_{a.ckpt}"
    Xc, zc, uc, cc = collect(model, cfg, dev, clean_cuts, 1, tag="CLEAN",
                             cache=f"{cpre}_clean")
    Xe, ze, ue, ce = collect(model, cfg, dev, extra_cuts, 1, keep=POS_ONLY, group_keep=1,
                             depth_clip=a.extra_depth_clip, tag="EXTRA", cache=f"{cpre}_extra")

    uv = panel_users(V, 3)["user_id"].to_numpy()
    rv = user_rows(uv)
    yv = target_at(V, rv)
    gv = user_group(uv)
    fv = ARTIFACTS / f"{cpre}_val_X.npy"
    Xv = np.load(fv) if fv.exists() else embed(model, cfg, dev, V, rv)
    A = gv == 0
    log(f"VAL {V}: {len(uv):,} пользователей, группа A {int(A.sum()):,}, "
        f"доля y>0 {(yv > 0).mean():.4f}")

    # ---- те же анти-лукап проверки, что в пилоте: состав выборок не изменился
    assert len(ue) and int(user_group(ue).min()) == 1, "в EXTRA попала группа A"
    inter = np.intersect1d(np.unique(ue), uv[A])
    assert inter.size == 0, f"{inter.size} пользователей группы A нашлись в EXTRA"
    assert all(T > CORRIDOR_END and T > V for T in extra_cuts)
    assert all(T + dt.timedelta(days=TARGET_DAYS) <= V for T in clean_cuts), \
        "CLEAN нарушает T+30<=V"
    log(f"анти-лукап OK: пересечение EXTRA x (группа A валидации) = {inter.size}")

    pos_c = zc > 0
    c_clean = np.array([zc[pos_c & (cc == k)].mean() for k in range(len(clean_cuts))])
    c_extra = np.array([ze[ce == k].mean() for k in range(len(extra_cuts))])
    c_hat = float(c_clean.mean())

    batch, hidden, drop = a.batch, cfg["hidden"], a.dropout
    p_steps = int(np.ceil(len(Xc) / batch)) * a.epochs
    tcp = (zc[pos_c] - c_clean[cc[pos_c]]).astype(np.float32)
    tep = (ze - c_extra[ce]).astype(np.float32)
    ci_pos = cc[pos_c]
    n_cp, n_extra = len(tcp), len(tep)
    steps = int(np.ceil(n_cp / batch)) * a.epochs
    # Общий пул собирается СРАЗУ в конечный буфер: `Xc[pos_c]` плюс `concatenate`
    # держали бы в пике лишнюю копию положительных строк (+1 ГБ на фолде 10-16),
    # а `Xc` умереть не может — на нём учится экстенсивная голова. Значения
    # побитово те же, что у прежней сборки.
    Xp = np.empty((n_cp + n_extra, Xc.shape[1]), Xc.dtype)
    np.take(Xc, np.flatnonzero(pos_c), axis=0, out=Xp[:n_cp])
    Xp[n_cp:] = Xe
    tp = np.concatenate([tcp, tep])
    del Xe
    i_clean = np.arange(n_cp)
    rng = np.random.default_rng(a.seed)
    early = np.flatnonzero(ci_pos < max(1, len(clean_cuts) // 3))
    heads = {
        "CLEAN": i_clean,
        "VOL": np.concatenate([i_clean, rng.choice(early, size=n_extra, replace=True)]),
        "FRESH": np.concatenate([i_clean, np.arange(n_cp, n_cp + n_extra)]),
    }
    log(f"бюджет: интенсив {steps:,} шагов, экстенсив {p_steps:,} шагов, batch {batch}; "
        f"CLEAN+ {n_cp:,}, EXTRA {n_extra:,}, c^ = {c_hat:.4f}")

    pprod = load_pprod(V, uv)
    pp = getattr(a, "primary", None) or PPROD_PRIMARY
    assert pp in pprod, (
        f"нет основного источника {pp}: для DIST сначала `python -m src.dist_pact`")

    def _aligned_z(f, key="z"):
        """`ẑ` из чужого артефакта, выровненный по строкам панели; None если не тот набор."""
        if not f.exists():
            return None
        d = np.load(f, allow_pickle=True)
        ub, zz = np.asarray(d["user_id"]).astype(np.int64), np.asarray(d[key], float)
        o = np.argsort(ub)
        pos = np.searchsorted(ub[o], uv)
        return zz[o][pos] if bool(
            np.array_equal(ub[o][np.minimum(pos, len(o) - 1)], uv)) else None

    zb = _aligned_z(ARTIFACTS / f"oof_{a.ckpt}.npz")
    # табличная опора: сам `S1-DIST`, из которого взят экстенсив. Нужен, чтобы
    # видеть, добавляет ли интенсив SEQ хоть что-то поверх модели-донора.
    z_tab = _aligned_z(ARTIFACTS / f"PACT_dist_{V.isoformat()}.npz")

    pm = A & (yv > 0)
    lyp = np.log1p(yv[pm])

    def score(name, z, mu=None):
        off, cal = calibrate(yv[A], z[A])
        r = dict(variant=name, n=int(A.sum()), rmsle=rmsle_z(yv[A], z[A]), rmsle_cal=cal,
                 bias=bias_z(yv[A], z[A]), offset=off, mean_z=float(z[A].mean()),
                 auc=_auc(yv[A] > 0, z[A]))
        if mu is not None:
            r["rmsle_mu"] = float(np.sqrt(np.mean((lyp - mu[pm]) ** 2)))
            r["bias_mu"] = float(lyp.mean() - mu[pm].mean())
        return r

    seeds = a.head_seeds or [a.seed]
    assert a.seed in seeds, "основной сид обязан входить в развёртку"

    # Кэш голов. `μ̂` и `p̂_SEQ` не зависят от того, какой боевой экстенсив на них
    # умножается, поэтому обучение голов и композиция разделены: пересобрать
    # таблицу под другой `P_prod` стоит секунды, а не ещё один проход по GPU.
    fmu = ARTIFACTS / f"S04PROD_{a.exp}_mu.npz"
    cache = dict(np.load(fmu)) if fmu.exists() else {}
    have = all(f"mu_{m}_s{s}" in cache and f"pseq_{m}_s{s}" in cache
               for s in seeds for m in heads)
    if have:
        log(f"головы взяты из кэша {fmu.name} (сиды {seeds})")

    per_seed, zref, fresh_cache = [], {}, {}
    for hs in seeds:
        if have:
            ms = {m: cache[f"mu_{m}_s{hs}"].astype(np.float64) for m in heads}
            zs = {m: cache[f"pseq_{m}_s{hs}"].astype(np.float64) for m in heads}
        else:
            zs, ms = _run_heads(Xc, zc, pos_c, Xp, tp, heads, Xv, c_hat, hs, batch,
                                hidden, drop, a, dev, p_steps, steps)
            # `_run_heads` называет варианты как в пилоте — короткие имена
            ms = {k.replace("COND-", ""): v for k, v in ms.items()}
            zs = {k.replace("COND-", ""): v for k, v in zs.items()}
            for m in heads:
                fresh_cache[f"mu_{m}_s{hs}"] = np.asarray(ms[m], np.float32)
                fresh_cache[f"pseq_{m}_s{hs}"] = np.asarray(zs[m], np.float32)
        rows, zmap = [], {}
        for pname, pv in pprod.items():
            for hname, mu in ms.items():
                z = np.maximum(pv * np.maximum(mu, 0.0), 0.0)
                zmap[f"{pname}x{hname}"] = z
                rows.append(score(f"{pname}x{hname}", z, mu))
        for hname, z in zs.items():                       # референс из exp_032
            zmap[f"SEQx{hname}"] = z
            rows.append(score(f"SEQx{hname}", z, ms[hname]))
        if zb is not None:
            zmap["BASE-1HEAD"] = zb
            rows.append(score("BASE-1HEAD", zb))
        if z_tab is not None:
            # `S1-DIST` — сам двухчастная модель: центроид нулевого бина равен нулю,
            # поэтому `ẑ = Σ_{k≥1} p_k·m_k = (1 − p0)·μ_DIST`. Значит `DIST-TAB` и
            # `DISTxFRESH` делят ОДИН экстенсив и различаются ровно интенсивом —
            # это прямое сравнение «интенсив донора против интенсива SEQ».
            zmap["DIST-TAB"] = z_tab
            rows.append(score("DIST-TAB", z_tab, mu=z_tab / np.maximum(pprod["DIST"], 1e-9)))
        d = {r["variant"]: r["rmsle_cal"] for r in rows}
        log(f"  сид головы {hs}: {pp}xFRESH−{pp}xCLEAN = "
            f"{d[f'{pp}xFRESH'] - d[f'{pp}xCLEAN']:+.5f} | "
            + (f"{pp}xFRESH−BASE = {d[f'{pp}xFRESH'] - d['BASE-1HEAD']:+.5f} | "
               if "BASE-1HEAD" in d else "")
            + f"SEQxFRESH−SEQxCLEAN = {d['SEQxFRESH'] - d['SEQxCLEAN']:+.5f}")
        per_seed.append(dict(seed=hs, rows=rows))
        if hs == a.seed:
            zref = dict(zmap=zmap, mus=ms, p_seq=zs)
    if fresh_cache:
        np.savez_compressed(fmu, uid=uv, **fresh_cache)
        log(f"головы сохранены: artifacts/{fmu.name} ({len(fresh_cache)} векторов)")

    # ---- сверка с пилотом: та же подготовка данных, тот же результат
    pilot = ARTIFACTS / f"S04SEQ_PILOT-S{a.seed}-V{V.strftime('%m%d')}_z.npz"
    recheck = {}
    if pilot.exists() and zref:
        d = np.load(pilot)
        assert np.array_equal(d["uid"], uv), "пилот считался на другой панели"
        for k, nm in (("COND_CLEAN", "CLEAN"), ("COND_FRESH", "FRESH")):
            dd = float(np.max(np.abs(d[k].astype(np.float64) - zref["p_seq"][nm])))
            recheck[nm] = dd
        log(f"сверка с пилотом ({pilot.name}): max|Δz| CLEAN {recheck['CLEAN']:.2e}, "
            f"FRESH {recheck['FRESH']:.2e}")

    zc_, zf_ = zref["zmap"][f"{pp}xCLEAN"][A], zref["zmap"][f"{pp}xFRESH"][A]
    ly = np.log1p(yv[A])
    d_var = float(np.var(zf_ - zc_))
    corr = float(np.corrcoef(ly - zc_, ly - zf_)[0, 1])
    auc_p = {k: _auc(yv[A] > 0, v[A]) for k, v in pprod.items()}
    log(f"{pp}: FRESH−CLEAN Var(dz) = {d_var:.5f} ({d_var / SEED_VAR_FLOOR:.2f}x пола сидов), "
        f"corr остатков {corr:.5f}")
    log("AUC(y>0) по P (группа A): "
        + ", ".join(f"{k} {v:.5f}" for k, v in auc_p.items()))

    hdr = ["variant", "rmsle", "rmsle_cal", "bias", "mean_z", "rmsle_mu", "auc"]
    print("\n" + " | ".join(f"{h:>13}" for h in hdr))
    for r in per_seed[0]["rows"]:
        print(" | ".join(f"{r[h]:>13.5f}" if isinstance(r.get(h), float)
                         else f"{str(r.get(h, '')):>13}" for h in hdr))

    seg = segments(V, rv)
    seg_rows = []
    for nm, m in seg_masks(seg).items():
        mm = m & A
        if mm.sum() < 100:
            continue
        row = dict(segment=nm, share=float(mm.sum() / A.sum()))
        for key in (f"{pp}xCLEAN", f"{pp}xFRESH", "SEQxCLEAN", "SEQxFRESH"):
            row[key] = calibrate(yv[mm], zref["zmap"][key][mm])[1]
        if zb is not None:
            row["BASE-1HEAD"] = calibrate(yv[mm], zb[mm])[1]
        row["delta"] = row[f"{pp}xFRESH"] - row[f"{pp}xCLEAN"]
        row["delta_base"] = (row[f"{pp}xFRESH"] - row["BASE-1HEAD"]) if zb is not None else None
        seg_rows.append(row)
    print("\nсегменты (RMSLE_cal, группа A):")
    for r in seg_rows:
        print(f"  {r['segment']:<22} доля {r['share']:.3f}  "
              f"{pp}xCLEAN {r[f'{pp}xCLEAN']:.5f}  {pp}xFRESH {r[f'{pp}xFRESH']:.5f}  "
              f"Δ {r['delta']:+.5f}  Δ к BASE {('%+.5f' % r['delta_base']) if r['delta_base'] is not None else 'н/д'}")

    out = dict(exp="EXP-032B", ckpt=a.ckpt, val=V.isoformat(), seed=a.seed,
               head_seeds=list(seeds), primary=pp, audit=aud, c_hat=c_hat,
               steps=steps, p_steps=p_steps, n_clean_pos=int(n_cp),
               n_extra_pos=int(n_extra), encoder_checksum=enc_sig,
               extra_depth_clip=a.extra_depth_clip, auc_p=auc_p, var_delta=d_var,
               corr_resid=corr, pilot_recheck=recheck, segments=seg_rows,
               per_seed=per_seed, n_groupA=int(A.sum()))
    p = ARTIFACTS / f"S04PROD_{a.exp}.json"
    p.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    np.savez_compressed(ARTIFACTS / f"S04PROD_{a.exp}_z.npz", uid=uv, y=yv, group=gv,
                        **{f"P_{k.replace('-', '_')}": v for k, v in pprod.items()},
                        **{"mu_" + k: v for k, v in zref["mus"].items()},
                        **{"z_" + k.replace("-", "_").replace("x", "_X_"): v
                           for k, v in zref["zmap"].items()})
    log(f"записано: artifacts/{p.name}, artifacts/S04PROD_{a.exp}_z.npz")

    enc_after = float(sum(float(p_.double().sum()) for p_ in model.parameters()))
    assert enc_after == enc_sig, "энкодер изменился — заморозка нарушена"
    log(f"энкодер после обучения голов: чек-сумма {enc_after:.6f} — не изменилась")


def main():
    ap = argparse.ArgumentParser(description="EXP-032 S04: conditional intensity head")
    sub = ap.add_subparsers(dest="cmd", required=True)

    q = sub.add_parser("audit", help="анти-лукап EXTRA без GPU")
    q.add_argument("--val", default="2025-10-16")

    p = sub.add_parser("pilot", help="COND-CLEAN vs COND-VOL vs COND-FRESH")
    p.add_argument("--ckpt", default="SEQ-D3A-BASE-S42-V1016")
    p.add_argument("--val", default=None)
    p.add_argument("--exp", default="PILOT-S42-V1016")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--batch", type=int, default=8192)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-2)
    p.add_argument("--dropout", type=float, default=0.10)
    p.add_argument("--extra-depth-clip", type=int, default=289)
    p.add_argument("--n-cutoffs", type=int, default=None, help="только последние N (отладка)")
    p.add_argument("--val-frac", type=float, default=1.0, help="доля val-панели (отладка)")
    p.add_argument("--head-seeds", type=int, nargs="*", default=None,
                   help="развёртка головы по сидам; эмбеддинги считаются один раз")
    p.add_argument("--no-cache", action="store_true", help="не кэшировать эмбеддинги")

    r = sub.add_parser("prod", help="EXP-032B: P_prod x mu_CLEAN против P_prod x mu_FRESH")
    r.add_argument("--ckpt", default="SEQ-D3A-BASE-S42-V1016")
    r.add_argument("--val", default=None)
    r.add_argument("--exp", default=None)
    r.add_argument("--seed", type=int, default=SEED)
    r.add_argument("--batch", type=int, default=8192)
    r.add_argument("--epochs", type=int, default=4)
    r.add_argument("--lr", type=float, default=1e-3)
    r.add_argument("--wd", type=float, default=1e-2)
    r.add_argument("--dropout", type=float, default=0.10)
    r.add_argument("--extra-depth-clip", type=int, default=289)
    r.add_argument("--primary", default=PPROD_PRIMARY, choices=list(PPROD_SOURCES),
                   help="какой источник P считать основным; по умолчанию боевой DIST")
    r.add_argument("--head-seeds", type=int, nargs="*", default=None,
                   help="развёртка головы по сидам; эмбеддинги читаются из кэша exp_032")

    a = ap.parse_args()
    if a.cmd == "prod" and not a.exp:
        a.exp = f"S{a.seed}-{a.ckpt.rsplit('-', 1)[-1]}"      # SEQ-...-V1016 -> S42-V1016
    {"audit": cmd_audit, "pilot": cmd_pilot, "prod": cmd_prod}[a.cmd](a)


if __name__ == "__main__":
    main()
