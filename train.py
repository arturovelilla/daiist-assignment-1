"""
Assignment 1 - training pipeline.

Business question: 2 hours before an inbound flight leaves its origin, will it
arrive at Detroit (DTW) 15+ minutes late? If yes, the hub operations centre
activates connection protection for its passengers.

Run with:  uv run python main.py train

Stages (one function each, in this order):
  1. load       national BTS file for March 2026 + airport time zones
  2. times      convert every schedule/actual time to UTC
  3. features   built on the NATIONAL data (inbound aircraft, day-so-far rates)
  4. target     keep DTW arrivals, drop cancelled/diverted
  5. split      chronological by day: 1-16 train, 17-21 validation, 22-31 test
  6. encode     fit encoders/scaler on train only, apply to all three sets
  7. models     baseline + logistic regression three ways
  8. evaluate   metrics, agreement checks, cost-based threshold
  9. save       everything app.py needs, so the app never retrains
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

# ---------------------------------------------------------------------------
# Configuration: every decision that is a "knob" lives here
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "data" / "march2026.zip"
TZ_FILE = ROOT / "data" / "airport_timezones.csv"
ART = ROOT / "artifacts"

HUB = "DTW"
HUB_TZ = "America/Detroit"
LEAD_HOURS = 2                 # prediction moment t0 = scheduled departure - 2h
LATE_MIN = 15                  # "late" means 15+ minutes, the industry standard

TRAIN_DAYS = range(1, 17)      # 1-16 March (contains the mid-month storm peak)
VAL_DAYS = range(17, 22)       # 17-21 March (storm tail -> bad days in validation)
TEST_DAYS = range(22, 32)      # 22-31 March (touched once, at the end)

PRIOR_STRENGTH = 10            # pseudo-flights added to day-so-far rates
TOP_ORIGINS = 15               # origins kept as their own category
TOP_CARRIERS = ["DL", "OO", "NK", "YX"]
HOUR_BINS = [0, 9, 13, 17, 21, 24]
HOUR_LABELS = ["before_09", "09_13", "13_17", "17_21", "after_21"]

# Placeholder business costs in USD per flight (to be replaced with sourced figures)
COST_FP = 500    # protection activated, flight arrives on time (wasted staff/rebooking)
COST_FN = 2000   # flight arrives late, nothing was done (missed connections)

C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0]   # sklearn regularisation (inverse strength)
C_TOL = 0.001    # Cs within this val log loss of the best count as "equally good"
LR_GRID = [0.05, 0.1, 0.5, 1.0]          # learning rates for both PyTorch versions
EPOCHS = 20000                           # full-batch gradient descent steps
SEED = 0


# ---------------------------------------------------------------------------
# 1. Load
# ---------------------------------------------------------------------------
def load():
    df = pd.read_csv(DATA_FILE, low_memory=False)
    df["date"] = pd.to_datetime(df["FL_DATE"], format="%m/%d/%Y %I:%M:%S %p")
    df["day"] = df["date"].dt.day
    tz = pd.read_csv(TZ_FILE).set_index("iata")["tz"]
    df["origin_tz"] = df["ORIGIN"].map(tz)
    missing = df["origin_tz"].isna().sum()
    if missing:
        raise ValueError(f"{missing} rows have an origin without a time zone")
    print(f"[load] {len(df):,} national flights, {df['day'].nunique()} days")
    return df


# ---------------------------------------------------------------------------
# 2. Times -> UTC
# ---------------------------------------------------------------------------
def hhmm_to_minutes(hhmm):
    """1435 -> 875 minutes after midnight. 2400 becomes 1440 (next midnight)."""
    hhmm = hhmm.astype(int)
    return (hhmm // 100) * 60 + hhmm % 100


def add_utc_times(df):
    """
    Only the scheduled departure needs a time-zone conversion. Everything else is
    derived from it with durations, which avoids the overnight-wrap problem:
      sched_arr = sched_dep + scheduled elapsed time (BTS computes it across zones)
      act_dep   = sched_dep + departure delay
      act_arr   = sched_arr + arrival delay
    March 2026 includes the US switch to summer time (8 March), so local times
    from 2:00 to 2:59 that morning do not exist; they are shifted forward.
    """
    local = df["date"] + pd.to_timedelta(hhmm_to_minutes(df["CRS_DEP_TIME"]), unit="min")
    sched_dep = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns, UTC]")
    for zone, idx in df.groupby("origin_tz").groups.items():
        sched_dep.loc[idx] = (
            local.loc[idx]
            .dt.tz_localize(zone, nonexistent="shift_forward", ambiguous="NaT")
            .dt.tz_convert("UTC")
        )
    df["sched_dep"] = sched_dep
    df["sched_arr"] = df["sched_dep"] + pd.to_timedelta(df["CRS_ELAPSED_TIME"], unit="min")
    df["act_dep"] = df["sched_dep"] + pd.to_timedelta(df["DEP_DELAY"], unit="min")
    df["act_arr"] = df["sched_arr"] + pd.to_timedelta(df["ARR_DELAY"], unit="min")
    df["t0"] = df["sched_dep"] - pd.Timedelta(hours=LEAD_HOURS)
    return df


# ---------------------------------------------------------------------------
# 3. Features (computed on national data, before filtering to DTW)
# ---------------------------------------------------------------------------
def add_inbound_features(df):
    """
    The inbound leg is the previous flight of the same aircraft (tail number)
    that lands at this flight's origin.
      turn_buffer   minutes between inbound scheduled arrival and our scheduled departure
      inbound_delay inbound departure delay, ONLY if it left before t0, else 0
      inbound_status 'departed' / 'not_departed' / 'none'
      propagated    max(0, inbound_delay - turn_buffer): the part of the delay
                    the turnaround buffer cannot absorb
    """
    df = df.sort_values(["TAIL_NUM", "sched_dep"]).copy()
    prev = df.groupby("TAIL_NUM", sort=False)[["DEST", "sched_arr", "act_dep", "DEP_DELAY"]].shift(1)

    buffer = (df["sched_dep"] - prev["sched_arr"]).dt.total_seconds() / 60
    # A valid inbound leg lands where we depart, within the previous 12 hours.
    # Anything else (first flight of the day, tail swap, missing tail, 28 Feb
    # leg not in the file) is treated as "no inbound leg".
    has_inbound = (
        df["TAIL_NUM"].notna()
        & (prev["DEST"] == df["ORIGIN"])
        & buffer.between(0, 12 * 60)
    )
    departed = has_inbound & prev["act_dep"].notna() & (prev["act_dep"] <= df["t0"])

    df["turn_buffer"] = np.where(has_inbound, buffer, 0.0)
    df["inbound_delay"] = np.where(departed, prev["DEP_DELAY"].clip(lower=0), 0.0)
    df["inbound_status"] = np.select(
        [departed, has_inbound], ["departed", "not_departed"], default="none"
    )
    df["propagated"] = np.where(
        departed, (df["inbound_delay"] - df["turn_buffer"]).clip(lower=0), 0.0
    )
    # Both delay features are mostly 0 with a long right tail (up to 9+ hours).
    # log1p compresses the tail so a handful of extreme flights don't dominate
    # the fit, and keeps 0 at 0. It also makes gradient descent converge: on the
    # raw scale the rare large values make the loss surface very elongated.
    df["log_inbound_delay"] = np.log1p(df["inbound_delay"])
    df["log_propagated"] = np.log1p(df["propagated"])
    return df.sort_index()


def so_far_rate(events, targets, by, event_time, late_col, prior):
    """
    For each target row: among `events` in the same group (`by`) whose
    `event_time` is strictly before the target's t0, the share that were late.
    Shrunk toward `prior` with PRIOR_STRENGTH pseudo-flights:
        rate = (late + K * prior) / (n + K)
    """
    ev = events[by + [event_time, late_col]].dropna(subset=[event_time])
    ev = ev.sort_values(event_time)
    ev["n_cum"] = ev.groupby(by).cumcount() + 1
    ev["late_cum"] = ev.groupby(by)[late_col].cumsum()

    tg = targets[by + ["t0"]].reset_index().sort_values("t0")
    merged = pd.merge_asof(
        tg, ev[by + [event_time, "n_cum", "late_cum"]],
        left_on="t0", right_on=event_time, by=by, allow_exact_matches=False,
    ).set_index("index").sort_index()
    n = merged["n_cum"].fillna(0)
    late = merged["late_cum"].fillna(0)
    return (late + PRIOR_STRENGTH * prior) / (n + PRIOR_STRENGTH)


def add_day_so_far_features(df, hub):
    """Three 'how is today going' rates, using only what happened before t0."""
    departed = df[df["act_dep"].notna()].copy()
    departed["dep_late"] = (departed["DEP_DELAY"] >= LATE_MIN).astype(float)
    arrived = df[(df["DEST"] == HUB) & df["act_arr"].notna()].copy()
    arrived["arr_late"] = arrived["ARR_DEL15"].astype(float)

    # Priors come from TRAINING days only, so validation/test never shape them
    prior_dep = departed.loc[departed["day"].isin(TRAIN_DAYS), "dep_late"].mean()
    prior_arr = arrived.loc[arrived["day"].isin(TRAIN_DAYS), "arr_late"].mean()

    hub = hub.copy()
    hub["origin_rate"] = so_far_rate(departed, hub, ["ORIGIN", "day"], "act_dep", "dep_late", prior_dep)
    hub["hub_rate"] = so_far_rate(arrived, hub, ["day"], "act_arr", "arr_late", prior_arr)
    hub["national_rate"] = so_far_rate(departed, hub, ["day"], "act_dep", "dep_late", prior_dep)
    priors = {"departure_late_rate": prior_dep, "hub_arrival_late_rate": prior_arr}
    return hub, priors


def add_schedule_features(df, hub):
    """Arrival-hour bin (Detroit local time) and DTW arrivals scheduled in the same hour."""
    hub = hub.copy()
    local_arr = hub["sched_arr"].dt.tz_convert(HUB_TZ)
    hub["arr_hour_bin"] = pd.cut(
        local_arr.dt.hour, bins=HOUR_BINS, labels=HOUR_LABELS, right=False
    ).astype(str)
    # Count every scheduled DTW arrival, including later-cancelled ones: the
    # schedule is known in advance, cancellations are not.
    all_hub = df[df["DEST"] == HUB]
    slot = all_hub["sched_arr"].dt.floor("h").value_counts()
    hub["hub_slot_arrivals"] = hub["sched_arr"].dt.floor("h").map(slot).astype(float)
    return hub


def build_dataset(df):
    df = add_inbound_features(df)
    hub = df[df["DEST"] == HUB].copy()
    n_all = len(hub)
    dropped = hub[(hub["CANCELLED"] == 1) | (hub["DIVERTED"] == 1)]
    hub = hub[(hub["CANCELLED"] == 0) & (hub["DIVERTED"] == 0)]
    print(f"[target] {n_all:,} DTW arrivals, dropped {len(dropped)} cancelled/diverted "
          f"-> {len(hub):,} rows, {hub['ARR_DEL15'].mean():.1%} late")
    hub, priors = add_day_so_far_features(df, hub)
    hub = add_schedule_features(df, hub)
    hub["y"] = hub["ARR_DEL15"].astype(int)
    hub["split"] = np.select(
        [hub["day"].isin(TRAIN_DAYS), hub["day"].isin(VAL_DAYS)], ["train", "val"], "test"
    )
    dropped_by_day = dropped.groupby("day").size().reindex(range(1, 32), fill_value=0)
    return hub, priors, dropped_by_day


# ---------------------------------------------------------------------------
# 6. Encode (fit on train only)
# ---------------------------------------------------------------------------
NUMERIC = ["CRS_ELAPSED_TIME", "turn_buffer", "log_inbound_delay", "log_propagated",
           "origin_rate", "hub_rate", "national_rate", "hub_slot_arrivals"]


class Encoder:
    """One-hot categoricals (with an explicit reference level dropped) + standardised numerics."""

    def fit(self, train):
        top = train["ORIGIN"].value_counts().index[:TOP_ORIGINS].tolist()
        self.categories = {
            # first entry of each list is the reference level and gets no column
            "origin": ["other"] + top,
            "carrier": ["other"] + TOP_CARRIERS,
            "arr_hour_bin": HOUR_LABELS,
            "inbound_status": ["none", "not_departed", "departed"],
        }
        self.mean = train[NUMERIC].mean()
        self.std = train[NUMERIC].std().replace(0, 1)
        self.feature_names = NUMERIC + [
            f"{col}={lvl}" for col, lvls in self.categories.items() for lvl in lvls[1:]
        ]
        return self

    def raw_categoricals(self, df):
        top = self.categories["origin"][1:]
        return pd.DataFrame({
            "origin": df["ORIGIN"].where(df["ORIGIN"].isin(top), "other"),
            "carrier": df["OP_UNIQUE_CARRIER"].where(df["OP_UNIQUE_CARRIER"].isin(TOP_CARRIERS), "other"),
            "arr_hour_bin": df["arr_hour_bin"],
            "inbound_status": df["inbound_status"],
        }, index=df.index)

    def transform(self, df):
        num = ((df[NUMERIC] - self.mean) / self.std).to_numpy(dtype=np.float64)
        cats = self.raw_categoricals(df)
        onehot = [
            (cats[col] == lvl).to_numpy(dtype=np.float64)[:, None]
            for col, lvls in self.categories.items() for lvl in lvls[1:]
        ]
        return np.hstack([num] + onehot)

    def to_dict(self):
        return {"categories": self.categories, "numeric": NUMERIC,
                "mean": self.mean.to_dict(), "std": self.std.to_dict(),
                "feature_names": self.feature_names}


# ---------------------------------------------------------------------------
# 7. Models
# ---------------------------------------------------------------------------
def penalty_lambda(C, n):
    """
    sklearn minimises   C * sum(logloss) + 0.5 * ||w||^2
    dividing by C*n:    mean(logloss) + 0.5 * lambda * ||w||^2,  lambda = 1/(C*n)
    The PyTorch versions minimise the second form, so this lambda makes all
    three solve the same problem. The intercept is never penalised.
    """
    return 1.0 / (C * n)


def train_sklearn(X, y, C):
    model = LogisticRegression(C=C, max_iter=20000, tol=1e-10)
    return model.fit(X, y)


def train_manual(X, y, lam, lr, epochs=EPOCHS):
    """Session 5 style: raw tensors, autograd for the gradient, manual update step."""
    Xt = torch.tensor(X, dtype=torch.float64)
    yt = torch.tensor(y, dtype=torch.float64)
    w = torch.zeros(X.shape[1], dtype=torch.float64, requires_grad=True)
    b = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    history = []
    for epoch in range(epochs):
        z = Xt @ w + b
        # Binary cross-entropy written from logits: log(1+e^z) - y*z.
        # softplus is the numerically stable log(1+e^z); log(sigmoid(z)) would
        # produce -inf when the sigmoid saturates.
        loss = (torch.nn.functional.softplus(z) - yt * z).mean() + 0.5 * lam * (w ** 2).sum()
        loss.backward()
        with torch.no_grad():
            w -= lr * w.grad
            b -= lr * b.grad
        w.grad.zero_()
        b.grad.zero_()
        if epoch % 100 == 0:
            history.append(loss.item())
    return w.detach().numpy().copy(), float(b.detach()), history


class LogReg(torch.nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.linear = torch.nn.Linear(n_features, 1, dtype=torch.float64)
        torch.nn.init.zeros_(self.linear.weight)   # same start point as the manual loop
        torch.nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return self.linear(x).squeeze(-1)          # logits; sigmoid lives in the loss


def train_standard(X, y, lam, lr, epochs=EPOCHS):
    """Standard workflow: nn.Module + torch.optim. Full-batch SGD = plain gradient descent."""
    Xt = torch.tensor(X, dtype=torch.float64)
    yt = torch.tensor(y, dtype=torch.float64)
    model = LogReg(X.shape[1])
    # No weight_decay here: it would also penalise the bias. L2 on weights only
    # is added to the loss explicitly instead.
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    bce = torch.nn.BCEWithLogitsLoss()
    history = []
    for epoch in range(epochs):
        opt.zero_grad()
        loss = bce(model(Xt), yt) + 0.5 * lam * (model.linear.weight ** 2).sum()
        loss.backward()
        opt.step()
        if epoch % 100 == 0:
            history.append(loss.item())
    return model, history


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


# ---------------------------------------------------------------------------
# 8. Evaluate
# ---------------------------------------------------------------------------
def cost(y, pred):
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    return fp * COST_FP + fn * COST_FN, fp, fn


def best_threshold(y, p):
    """Threshold minimising business cost on VALIDATION data."""
    grid = np.round(np.arange(0.01, 1.0, 0.01), 2)
    costs = [cost(y, (p >= t).astype(int))[0] for t in grid]
    return float(grid[int(np.argmin(costs))])


def evaluate(y, p, threshold):
    pred = (p >= threshold).astype(int)
    total, fp, fn = cost(y, pred)
    tp = int(((pred == 1) & (y == 1)).sum())
    return {
        "roc_auc": roc_auc_score(y, p) if len(np.unique(p)) > 1 else 0.5,
        "pr_auc": average_precision_score(y, p),
        "log_loss": log_loss(y, np.clip(p, 1e-12, 1 - 1e-12)),
        "accuracy": float((pred == y).mean()),
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(int(y.sum()), 1),
        "threshold": threshold,
        "fp": fp, "fn": fn, "cost_usd": total, "cost_per_flight": total / len(y),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    ART.mkdir(exist_ok=True)

    df = add_utc_times(load())
    data, priors, dropped_by_day = build_dataset(df)

    parts = {s: data[data["split"] == s] for s in ["train", "val", "test"]}
    for s, d in parts.items():
        print(f"[split] {s:5s} days {d['day'].min()}-{d['day'].max()}: "
              f"{len(d):,} rows, {d['y'].mean():.1%} late")

    enc = Encoder().fit(parts["train"])
    X = {s: enc.transform(d) for s, d in parts.items()}
    y = {s: d["y"].to_numpy() for s, d in parts.items()}
    n_train, n_feat = X["train"].shape
    print(f"[encode] {n_feat} features")

    # --- tune C on validation log loss (sklearn), reuse the same lambda in torch
    c_scores = {}
    for C in C_GRID:
        m = train_sklearn(X["train"], y["train"], C)
        c_scores[C] = log_loss(y["val"], m.predict_proba(X["val"])[:, 1])
    # Among Cs that are equally good on validation (within C_TOL of the best
    # log loss), take the smallest C = the strongest regularisation. A weaker
    # penalty lets the weight of the rare, near-separating `propagated` feature
    # run away, which only gradient descent's slow convergence then exposes.
    best_score = min(c_scores.values())
    best_C = min(C for C, s in c_scores.items() if s <= best_score + C_TOL)
    lam = penalty_lambda(best_C, n_train)
    print(f"[tune] val log loss by C: { {k: round(v, 4) for k, v in c_scores.items()} } -> C={best_C}")

    # --- tune learning rate: largest lr whose loss decreases without diverging,
    # judged on validation log loss after the fixed epoch budget
    lr_scores = {}
    for lr in LR_GRID:
        w, b, hist = train_manual(X["train"], y["train"], lam, lr, epochs=5000)
        diverged = not np.isfinite(hist[-1]) or hist[-1] > hist[0]
        lr_scores[lr] = np.inf if diverged else log_loss(y["val"], sigmoid(X["val"] @ w + b))
    best_lr = min(lr_scores, key=lr_scores.get)
    print(f"[tune] val log loss by lr (5k epochs): { {k: round(v, 4) for k, v in lr_scores.items()} } -> lr={best_lr}")

    # --- final models
    sk = train_sklearn(X["train"], y["train"], best_C)
    w_man, b_man, hist_man = train_manual(X["train"], y["train"], lam, best_lr)
    nn_model, hist_std = train_standard(X["train"], y["train"], lam, best_lr)
    w_std = nn_model.linear.weight.detach().numpy().ravel()
    b_std = float(nn_model.linear.bias.detach())

    coefs = {
        "sklearn": (sk.coef_.ravel(), float(sk.intercept_[0])),
        "torch_manual": (w_man, b_man),
        "torch_nn": (w_std, b_std),
    }
    probs = {name: {s: sigmoid(X[s] @ w + b) for s in X} for name, (w, b) in coefs.items()}
    # Baseline: predict the training late rate for every flight (never protects anyone)
    base_rate = y["train"].mean()
    probs["baseline"] = {s: np.full(len(y[s]), base_rate) for s in X}

    # --- agreement between implementations
    agreement = {}
    for a, b_ in [("sklearn", "torch_manual"), ("sklearn", "torch_nn"), ("torch_manual", "torch_nn")]:
        agreement[f"{a} vs {b_}"] = {
            "max_abs_coef_diff": float(np.abs(np.append(*coefs[a]) - np.append(*coefs[b_])).max()),
            "max_abs_prob_diff_test": float(np.abs(probs[a]["test"] - probs[b_]["test"]).max()),
        }
    print("[agree]", json.dumps(agreement, indent=2))

    # --- thresholds chosen on validation, applied once to test
    metrics = {}
    for name, p in probs.items():
        thr = best_threshold(y["val"], p["val"]) if name != "baseline" else 1.0
        metrics[name] = {s: evaluate(y[s], p[s], thr) for s in ["val", "test"]}
    # Two naive policies for the cost comparison
    for name, pred in [("always_protect", 1), ("never_protect", 0)]:
        metrics[name] = {s: dict(zip(["cost_usd", "fp", "fn"], cost(y[s], np.full(len(y[s]), pred))))
                         for s in ["val", "test"]}

    print("\n[results] test set")
    print(f"{'model':14s} {'ROC AUC':>8s} {'PR AUC':>7s} {'logloss':>8s} {'thr':>5s} {'cost $':>10s}")
    for name in ["baseline", "sklearn", "torch_manual", "torch_nn"]:
        m = metrics[name]["test"]
        print(f"{name:14s} {m['roc_auc']:8.3f} {m['pr_auc']:7.3f} {m['log_loss']:8.4f} "
              f"{m['threshold']:5.2f} {m['cost_usd']:10,.0f}")
    for name in ["always_protect", "never_protect"]:
        print(f"{name:14s} {'':8s} {'':7s} {'':8s} {'':5s} {metrics[name]['test']['cost_usd']:10,.0f}")

    # -----------------------------------------------------------------------
    # 9. Save: the app loads these and never retrains
    # -----------------------------------------------------------------------
    joblib.dump(sk, ART / "sklearn_model.joblib")
    torch.save({"w": torch.tensor(w_man), "b": torch.tensor(b_man)}, ART / "torch_manual.pt")
    torch.save(nn_model.state_dict(), ART / "torch_nn.pt")

    keep = ["day", "split", "y", "ORIGIN", "OP_UNIQUE_CARRIER", "arr_hour_bin",
            "inbound_status", "ARR_DELAY", "inbound_delay", "propagated"] + NUMERIC
    preds = data[keep].copy()
    for name in ["baseline", "sklearn", "torch_manual", "torch_nn"]:
        preds[f"p_{name}"] = np.nan
        for s in ["train", "val", "test"]:
            preds.loc[parts[s].index, f"p_{name}"] = probs[name][s]
    preds.to_csv(ART / "predictions.csv", index_label="row_id")

    meta = {
        "encoder": enc.to_dict(),
        "priors": priors,
        "prior_strength": PRIOR_STRENGTH,
        "costs": {"fp": COST_FP, "fn": COST_FN},
        "tuning": {"C_scores": c_scores, "best_C": best_C, "lambda": lam,
                   "lr_scores": {str(k): v for k, v in lr_scores.items()}, "best_lr": best_lr,
                   "epochs": EPOCHS},
        "coefficients": {k: {"w": dict(zip(enc.feature_names, map(float, w))), "b": b}
                         for k, (w, b) in coefs.items()},
        "agreement": agreement,
        "metrics": metrics,
        "loss_history": {"torch_manual": hist_man, "torch_nn": hist_std},
        "dropped_cancelled_diverted_by_day": dropped_by_day.to_dict(),
        "split_days": {"train": [min(TRAIN_DAYS), max(TRAIN_DAYS)],
                       "val": [min(VAL_DAYS), max(VAL_DAYS)],
                       "test": [min(TEST_DAYS), max(TEST_DAYS)]},
    }
    (ART / "meta.json").write_text(json.dumps(meta, indent=2, default=float))
    print(f"\n[save] artifacts written to {ART}")


if __name__ == "__main__":
    main()
