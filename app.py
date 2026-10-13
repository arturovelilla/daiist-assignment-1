"""
Assignment 1 - Gradio dashboard.

Run with:  uv run python main.py app      (after `main.py train`)

Loads the three trained models from artifacts/ and NEVER retrains anything.
The models are re-applied to the saved, already-engineered feature rows, using
the encoder parameters that train.py fitted on the training days. As a
safeguard, those fresh predictions are checked against the ones train.py saved.

Tabs
  1. Models      prediction vs actual (calibration), ROC curves, agreement, metrics
  2. Threshold   slider -> confusion matrix and business cost (editable costs)
  3. Data        delay rate by day (target + split), feature distributions
  4. Weights     coefficients of the three implementations side by side
"""

import json
from pathlib import Path

import gradio as gr
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
from sklearn.metrics import roc_auc_score, roc_curve

from train import LogReg  # same nn.Module definition as training (import does not train)

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"

# Fixed colour per implementation (never re-assigned when a filter changes)
COLORS = {"sklearn": "#2a78d6", "torch_manual": "#eb6834", "torch_nn": "#1baf7a", "baseline": "#8a8985"}
DASH = {"sklearn": "solid", "torch_manual": "dash", "torch_nn": "dot", "baseline": "solid"}
LABELS = {"sklearn": "scikit-learn", "torch_manual": "PyTorch manual loop",
          "torch_nn": "PyTorch nn.Module + optim", "baseline": "Baseline (train late rate)"}
MODELS = ["sklearn", "torch_manual", "torch_nn"]
LATE, ON_TIME = "#eb6834", "#2a78d6"
SPLIT_COLORS = {"train": "#2a78d6", "val": "#eda100", "test": "#1baf7a"}


# ---------------------------------------------------------------------------
# Load artifacts (fails loudly if training has not been run)
# ---------------------------------------------------------------------------
if not (ART / "meta.json").exists():
    raise SystemExit("artifacts/ not found - run `uv run python main.py train` first.")

META = json.loads((ART / "meta.json").read_text())
ENC = META["encoder"]
ROWS = pd.read_csv(ART / "predictions.csv", index_col="row_id")


def encode(df):
    """Re-create the training-time design matrix from saved encoder parameters."""
    mean, std = pd.Series(ENC["mean"]), pd.Series(ENC["std"])
    num = ((df[ENC["numeric"]] - mean[ENC["numeric"]]) / std[ENC["numeric"]]).to_numpy(np.float64)
    top_origins = ENC["categories"]["origin"][1:]
    carriers = ENC["categories"]["carrier"][1:]
    cats = {
        "origin": df["ORIGIN"].where(df["ORIGIN"].isin(top_origins), "other"),
        "carrier": df["OP_UNIQUE_CARRIER"].where(df["OP_UNIQUE_CARRIER"].isin(carriers), "other"),
        "arr_hour_bin": df["arr_hour_bin"],
        "inbound_status": df["inbound_status"],
    }
    onehot = [(cats[col] == lvl).to_numpy(np.float64)[:, None]
              for col, lvls in ENC["categories"].items() for lvl in lvls[1:]]
    return np.hstack([num] + onehot)


def load_models_and_predict():
    X = encode(ROWS)
    Xt = torch.tensor(X, dtype=torch.float64)
    probs = {}

    sk = joblib.load(ART / "sklearn_model.joblib")
    probs["sklearn"] = sk.predict_proba(X)[:, 1]

    manual = torch.load(ART / "torch_manual.pt")
    probs["torch_manual"] = torch.sigmoid(Xt @ manual["w"] + manual["b"]).numpy()

    nn_model = LogReg(X.shape[1])
    nn_model.load_state_dict(torch.load(ART / "torch_nn.pt"))
    nn_model.eval()
    with torch.no_grad():
        probs["torch_nn"] = torch.sigmoid(nn_model(Xt)).numpy()

    probs["baseline"] = ROWS["p_baseline"].to_numpy()

    for name in MODELS:
        gap = np.abs(probs[name] - ROWS[f"p_{name}"].to_numpy()).max()
        if gap > 1e-6:
            raise SystemExit(f"{name}: reloaded model disagrees with training output ({gap:.2e})")
    return probs


PROBS = load_models_and_predict()
DATA = ROWS.assign(**{f"p_{k}": v for k, v in PROBS.items()})
COST_FP, COST_FN = META["costs"]["fp"], META["costs"]["fn"]


def subset(split):
    return DATA if split == "all" else DATA[DATA["split"] == split]


def style(fig, title, x, y, height=420):
    fig.update_layout(
        title=title, xaxis_title=x, yaxis_title=y, height=height,
        template="plotly_white", font=dict(size=13),
        legend=dict(orientation="h", y=-0.2), margin=dict(l=60, r=20, t=60, b=60),
        hovermode="closest",
    )
    fig.update_xaxes(gridcolor="#ececea", zeroline=False)
    fig.update_yaxes(gridcolor="#ececea", zeroline=False)
    return fig


# ---------------------------------------------------------------------------
# Tab 1: model comparison
# ---------------------------------------------------------------------------
def calibration_plot(split):
    """Prediction vs actual for a classifier: mean predicted probability vs
    observed late rate, in 10 equal-count bins. Points on the diagonal = the
    predicted probabilities can be taken at face value."""
    d = subset(split)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect calibration",
                             line=dict(color="#c3c2b7", width=1, dash="dot"), hoverinfo="skip"))
    for name in MODELS:
        bins = pd.qcut(d[f"p_{name}"], 10, duplicates="drop")
        g = d.groupby(bins, observed=True).agg(pred=(f"p_{name}", "mean"), obs=("y", "mean"), n=("y", "size"))
        fig.add_trace(go.Scatter(
            x=g["pred"], y=g["obs"], mode="lines+markers", name=LABELS[name],
            line=dict(color=COLORS[name], width=2, dash=DASH[name]), marker=dict(size=9),
            customdata=g["n"],
            hovertemplate="predicted %{x:.2f}<br>actual %{y:.2f}<br>%{customdata} flights<extra>" + LABELS[name] + "</extra>",
        ))
    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, 1])
    return style(fig, f"Predicted vs actual late rate ({split})", "Mean predicted probability", "Observed share late")


def roc_plot(split):
    d = subset(split)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random / baseline",
                             line=dict(color="#c3c2b7", width=1, dash="dot"), hoverinfo="skip"))
    for name in MODELS:
        fpr, tpr, _ = roc_curve(d["y"], d[f"p_{name}"])
        auc = roc_auc_score(d["y"], d[f"p_{name}"])
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{LABELS[name]} (AUC {auc:.3f})",
                                 line=dict(color=COLORS[name], width=2, dash=DASH[name])))
    return style(fig, f"ROC curves ({split})", "False positive rate", "True positive rate")


def agreement_plot(split):
    """Each implementation's probability against scikit-learn's, flight by flight."""
    d = subset(split)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Identical",
                             line=dict(color="#c3c2b7", width=1, dash="dot"), hoverinfo="skip"))
    for name in ["torch_manual", "torch_nn"]:
        fig.add_trace(go.Scattergl(
            x=d["p_sklearn"], y=d[f"p_{name}"], mode="markers", name=LABELS[name],
            marker=dict(color=COLORS[name], size=8 if name == "torch_manual" else 5, opacity=0.6),
            hovertemplate="sklearn %{x:.4f}<br>torch %{y:.4f}<extra>" + LABELS[name] + "</extra>",
        ))
    return style(fig, f"Agreement with scikit-learn ({split})", "scikit-learn probability", "PyTorch probability")


def metrics_table(split):
    rows = []
    for name in ["baseline"] + MODELS:
        if split in ("val", "test"):
            m = META["metrics"][name][split]
            rows.append([LABELS[name], round(m["roc_auc"], 4), round(m["pr_auc"], 4),
                         round(m["log_loss"], 4), m["threshold"], f"${m['cost_usd']:,.0f}"])
        else:
            d = subset(split)
            rows.append([LABELS[name], round(roc_auc_score(d["y"], d[f"p_{name}"]), 4) if name != "baseline" else 0.5,
                         "", "", "", ""])
    return pd.DataFrame(rows, columns=["Model", "ROC AUC", "PR AUC", "Log loss", "Threshold (from val)", "Cost"])


def agreement_table():
    return pd.DataFrame([
        [pair, f"{v['max_abs_coef_diff']:.2e}", f"{v['max_abs_prob_diff_test']:.2e}"]
        for pair, v in META["agreement"].items()
    ], columns=["Pair", "Max |coefficient diff|", "Max |probability diff| (test)"])


# ---------------------------------------------------------------------------
# Tab 2: threshold and business cost
# ---------------------------------------------------------------------------
def threshold_view(model, split, threshold, cost_fp, cost_fn):
    d = subset(split)
    y, p = d["y"].to_numpy(), d[f"p_{model}"].to_numpy()
    pred = (p >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    total = fp * cost_fp + fn * cost_fn
    never = int(y.sum()) * cost_fn
    always = int((y == 0).sum()) * cost_fp

    cm = go.Figure(go.Heatmap(
        z=[[tn, fp], [fn, tp]], x=["Predicted on time", "Predicted late"],
        y=["Actually on time", "Actually late"], colorscale=[[0, "#e8f1fb"], [1, "#1c5aa6"]],
        showscale=False, text=[[f"TN<br>{tn:,}", f"FP<br>{fp:,}"], [f"FN<br>{fn:,}", f"TP<br>{tp:,}"]],
        texttemplate="%{text}", textfont=dict(size=18), hovertemplate="%{y}, %{x}: %{z}<extra></extra>",
    ))
    cm.update_yaxes(autorange="reversed")
    cm = style(cm, f"Confusion matrix at threshold {threshold:.2f}", "", "", height=380)

    grid = np.round(np.arange(0.01, 1.0, 0.01), 2)
    costs = [((p >= t) & (y == 0)).sum() * cost_fp + ((p < t) & (y == 1)).sum() * cost_fn for t in grid]
    curve = go.Figure()
    curve.add_trace(go.Scatter(x=grid, y=costs, mode="lines", name="Model policy",
                               line=dict(color=COLORS[model], width=2),
                               hovertemplate="threshold %{x:.2f}<br>$%{y:,.0f}<extra></extra>"))
    curve.add_hline(y=never, line=dict(color="#8a8985", dash="dot", width=1),
                    annotation_text="Never protect", annotation_position="top right")
    curve.add_hline(y=always, line=dict(color="#8a8985", dash="dash", width=1),
                    annotation_text="Protect every flight", annotation_position="bottom right")
    curve.add_vline(x=threshold, line=dict(color="#0b0b0b", width=1))
    curve = style(curve, "Total cost by threshold", "Threshold", "Cost (USD)", height=380)

    best_t = float(grid[int(np.argmin(costs))])
    saving = never - total
    summary = (
        f"### Cost at threshold {threshold:.2f}: **${total:,.0f}**  ({len(y):,} flights, ${total / len(y):,.0f} per flight)\n"
        f"- {fp:,} unnecessary protections × ${cost_fp:,.0f}  +  {fn:,} unprotected late arrivals × ${cost_fn:,.0f}\n"
        f"- vs never protecting: **${saving:,.0f} saved**  (never = ${never:,.0f}, protect all = ${always:,.0f})\n"
        f"- Precision {tp / max(tp + fp, 1):.1%} · Recall {tp / max(tp + fn, 1):.1%}\n"
        f"- Cheapest threshold on this split: {best_t:.2f}. "
        f"Cost-ratio rule of thumb: cost_FP / (cost_FP + cost_FN) = {cost_fp / (cost_fp + cost_fn):.2f}\n\n"
        f"_The official threshold is chosen on validation ({META['metrics'][model]['val']['threshold']:.2f}) "
        f"and only then applied to test; picking it on test would be peeking._"
    )
    return cm, curve, summary


# ---------------------------------------------------------------------------
# Tab 3: data
# ---------------------------------------------------------------------------
def daily_plot():
    g = DATA.groupby(["day", "split"]).agg(late=("y", "mean"), n=("y", "size")).reset_index()
    dropped = META["dropped_cancelled_diverted_by_day"]
    fig = go.Figure()
    for split, color in SPLIT_COLORS.items():
        s = g[g["split"] == split]
        fig.add_trace(go.Bar(
            x=s["day"], y=s["late"], name=split, marker=dict(color=color, cornerradius=4),
            customdata=np.c_[s["n"], [dropped[str(d)] for d in s["day"]]],
            hovertemplate="March %{x}<br>%{y:.0%} late<br>%{customdata[0]} flights kept"
                          "<br>%{customdata[1]} cancelled/diverted (dropped)<extra>" + split + "</extra>",
        ))
    fig.update_yaxes(tickformat=".0%")
    fig.update_layout(bargap=0.15)
    return style(fig, "Target: share of DTW arrivals 15+ min late, by day (colour = split)", "Day of March 2026", "Share late")


FEATURES = {
    "Inbound departure delay (min, known before t0)": "inbound_delay",
    "Delay the turnaround cannot absorb (min)": "propagated",
    "Turnaround buffer (min)": "turn_buffer",
    "Origin day-so-far late rate": "origin_rate",
    "DTW day-so-far late rate": "hub_rate",
    "National day-so-far late rate": "national_rate",
    "DTW arrivals scheduled in the same hour": "hub_slot_arrivals",
    "Scheduled elapsed time (min)": "CRS_ELAPSED_TIME",
    "Actual arrival delay (min) - the raw target, not a feature": "ARR_DELAY",
}
CATEGORICAL = {"Arrival hour (DTW local)": "arr_hour_bin", "Inbound aircraft status": "inbound_status",
               "Carrier": "OP_UNIQUE_CARRIER", "Origin airport": "ORIGIN"}


def feature_plot(label, split):
    d = subset(split)
    fig = go.Figure()
    if label in FEATURES:
        col = FEATURES[label]
        hi = d[col].quantile(0.99)
        for value, name, color in [(0, "On time", ON_TIME), (1, "Late (15+ min)", LATE)]:
            fig.add_trace(go.Histogram(
                x=d.loc[d["y"] == value, col].clip(upper=hi), name=name, marker_color=color,
                opacity=0.65, histnorm="probability", nbinsx=40,
            ))
        fig.update_layout(barmode="overlay")
        return style(fig, f"{label} - distribution by outcome ({split}, top 1% clipped)", label, "Share of flights")
    col = CATEGORICAL[label]
    g = d.groupby(col).agg(late=("y", "mean"), n=("y", "size")).sort_values("n", ascending=False).head(20)
    fig.add_trace(go.Bar(x=g.index.astype(str), y=g["late"], marker=dict(color=ON_TIME, cornerradius=4),
                         customdata=g["n"], name="Share late",
                         hovertemplate="%{x}<br>%{y:.0%} late<br>%{customdata} flights<extra></extra>"))
    fig.update_yaxes(tickformat=".0%")
    return style(fig, f"Share late by {label.lower()} ({split}{', top 20' if len(g) == 20 else ''})", label, "Share late")


# ---------------------------------------------------------------------------
# Tab 4: coefficients
# ---------------------------------------------------------------------------
def coef_plot():
    names = ENC["feature_names"]
    sk = META["coefficients"]["sklearn"]["w"]
    order = sorted(names, key=lambda n: sk[n])
    fig = go.Figure()
    for name in MODELS:
        w = META["coefficients"][name]["w"]
        fig.add_trace(go.Bar(y=order, x=[w[n] for n in order], orientation="h", name=LABELS[name],
                             marker=dict(color=COLORS[name], cornerradius=3),
                             hovertemplate="%{y}: %{x:.5f}<extra>" + LABELS[name] + "</extra>"))
    fig.update_layout(barmode="group", bargap=0.25)
    return style(fig, "Learned weights (standardised features; bars should be identical)",
                 "Weight (log-odds per 1 std / per category vs reference)", "", height=900)


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
INTRO = f"""
# DTW connection-protection model
Two hours before an inbound flight leaves its origin, predict whether it will reach **Detroit (DTW) 15+ minutes late**,
so the hub can protect its connecting passengers. Logistic regression trained three ways on March 2026
(train days {META['split_days']['train'][0]}-{META['split_days']['train'][1]},
validation {META['split_days']['val'][0]}-{META['split_days']['val'][1]},
test {META['split_days']['test'][0]}-{META['split_days']['test'][1]}).
Models are loaded from `artifacts/`; nothing is retrained here.
"""

with gr.Blocks(title="DTW delay model") as demo:
    gr.Markdown(INTRO)
    with gr.Tab("Models"):
        split1 = gr.Radio(["test", "val", "train"], value="test", label="Data split")
        metrics = gr.Dataframe(metrics_table("test"), label="Metrics (threshold chosen on validation)")
        with gr.Row():
            calib = gr.Plot(calibration_plot("test"))
            roc = gr.Plot(roc_plot("test"))
        with gr.Row():
            agree = gr.Plot(agreement_plot("test"))
            gr.Dataframe(agreement_table(), label="Implementation agreement")
        split1.change(lambda s: (metrics_table(s), calibration_plot(s), roc_plot(s), agreement_plot(s)),
                      split1, [metrics, calib, roc, agree])

    with gr.Tab("Threshold & cost"):
        with gr.Row():
            model = gr.Dropdown([(LABELS[m], m) for m in MODELS], value="sklearn", label="Model")
            split2 = gr.Radio(["test", "val"], value="test", label="Data split")
        thr = gr.Slider(0.01, 0.99, value=META["metrics"]["sklearn"]["val"]["threshold"], step=0.01,
                        label="Decision threshold: protect if P(late) ≥ threshold")
        with gr.Row():
            cfp = gr.Number(COST_FP, label="Cost of a false positive (USD): unnecessary protection")
            cfn = gr.Number(COST_FN, label="Cost of a false negative (USD): unprotected late arrival")
        summary = gr.Markdown()
        with gr.Row():
            cm = gr.Plot()
            curve = gr.Plot()
        inputs = [model, split2, thr, cfp, cfn]
        for comp in inputs:
            comp.change(threshold_view, inputs, [cm, curve, summary])
        demo.load(threshold_view, inputs, [cm, curve, summary])

    with gr.Tab("Data"):
        gr.Plot(daily_plot())
        with gr.Row():
            feat = gr.Dropdown(list(FEATURES) + list(CATEGORICAL), value=list(FEATURES)[0], label="Feature")
            split3 = gr.Radio(["train", "val", "test", "all"], value="train", label="Data split")
        fplot = gr.Plot(feature_plot(list(FEATURES)[0], "train"))
        feat.change(feature_plot, [feat, split3], fplot)
        split3.change(feature_plot, [feat, split3], fplot)

    with gr.Tab("Weights"):
        gr.Plot(coef_plot())


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)
