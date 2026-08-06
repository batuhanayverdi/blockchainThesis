"""
Human validation report with visualizations
===========================================
Replaces evaluate_validation.py. Compares the student raters' labels against the
model labels in master_key.xlsx and writes ONE self-contained HTML report
(validation_report.html) with figures and plain-language explanations that can
be opened in any browser and shared with a supervisor directly.

Because the two raters see disjoint posts, everything here is model-human
agreement. No inter-rater reliability is possible with this design. If more
than one rater file is present, the report shows pooled results plus a small
per-rater comparison table.

Reads
-----
  master_key.xlsx                 written by build_validation.py
  results/validation_rater2.csv   rater exports (rater 1 file: uncomment below)

Writes (into results/)
----------------------
  validation_report.html      the shareable report (figures embedded, no
                              external files needed)
  validation_merged.csv       row-level merge for manual inspection
  fig_*.png                   the individual figures, if you need them for
                              slides later

Run:  python evaluate_validation.py
Deps: pip install pandas numpy openpyxl matplotlib
"""

import os
import io
import csv
import base64
import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================ CONFIG ========================================
BASE = r"C:\Users\batuh\PycharmProjects\Thesis\fact_checked_hive_pipeline\8. human validation"

KEY_XLSX = os.path.join(BASE, "master_key.xlsx")
RATER_FILES = [
    os.path.join(BASE, "results", "validation_rater2.csv"),
    # os.path.join(BASE, "results", "validation_rater1.csv"),  # add when it arrives
]
OUT_DIR = os.path.join(BASE, "results")
OUT_HTML = os.path.join(OUT_DIR, "validation_report.html")

CATEGORIES = ["pants_on_fire", "false", "mostly_false", "mostly_true", "true"]
ORD = {c: i for i, c in enumerate(CATEGORIES)}
MISLEADING = {"pants_on_fire", "false", "mostly_false"}
UNVERIFIED = "unverified"

N_BOOT = 2000
BOOT_SEED = 20260723

DISPLAY = {
    "pants_on_fire": "Pants on Fire",
    "false": "False",
    "mostly_false": "Mostly False",
    "mostly_true": "Mostly True",
    "true": "True",
    "unverified": "Unverified",
}

# TUM-style palette
NAVY = "#072140"
BLUE = "#3070B3"
GOLD = "#CBAB01"
LIGHT = "#EAF1F8"
RED = "#C4071B"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.edgecolor": NAVY,
    "axes.labelcolor": NAVY,
    "text.color": NAVY,
    "xtick.color": NAVY,
    "ytick.color": NAVY,
    "figure.facecolor": "white",
})


# ============================ IO HELPERS ====================================
# Excel rewrites the exported CSV: "true"/"false" become TRUE/FALSE (or
# WAHR/FALSCH in a German locale) and the delimiter becomes ";". Reversed here.
LABEL_FIX = {
    "true": "true", "TRUE": "true", "True": "true", "WAHR": "true",
    "wahr": "true",
    "false": "false", "FALSE": "false", "False": "false",
    "FALSCH": "false", "falsch": "false",
    "pants_on_fire": "pants_on_fire", "mostly_false": "mostly_false",
    "mostly_true": "mostly_true", "unverified": "unverified",
}
BOOL_FIX = {
    "true": True, "TRUE": True, "True": True, "WAHR": True, "1": True,
    "false": False, "FALSE": False, "False": False, "FALSCH": False,
    "0": False,
}


def sniff_sep(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        head = fh.read(4096)
    try:
        return csv.Sniffer().sniff(head, delimiters=",;\t").delimiter
    except csv.Error:
        return ";" if head.count(";") > head.count(",") else ","


def read_rater(path):
    sep = sniff_sep(path)
    df = pd.read_csv(path, sep=sep, dtype=str, encoding="utf-8-sig").fillna("")
    df.columns = [c.strip() for c in df.columns]
    raw = df["human_label"].astype(str).str.strip()
    df["human_label"] = raw.map(LABEL_FIX)
    unmapped = raw[df["human_label"].isna()].unique().tolist()
    if unmapped:
        print(f"  WARNING [{os.path.basename(path)}]: unrecognised labels "
              f"{unmapped}; those rows are dropped.")
        df = df[df["human_label"].notna()].copy()
    if "is_checkable_claim" in df.columns:
        df["is_checkable_claim"] = (df["is_checkable_claim"].astype(str)
                                    .str.strip().map(BOOL_FIX))
    for c in ("seconds_spent", "order_shown"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["post_id"] = df["post_id"].astype(str).str.strip()
    df["source_file"] = os.path.basename(path)
    print(f"  Read {os.path.basename(path)}: {len(df)} rows (delimiter '{sep}').")
    return df


# ============================ STATISTICS ====================================
def cohen_kappa(a, b, labels, weights=None):
    a, b = list(a), list(b)
    n = len(a)
    if n == 0:
        return np.nan
    k = len(labels)
    idx = {lab: i for i, lab in enumerate(labels)}
    O = np.zeros((k, k))
    for x, y in zip(a, b):
        O[idx[x], idx[y]] += 1
    O /= n
    E = np.outer(O.sum(axis=1), O.sum(axis=0))
    if weights is None:
        W = 1.0 - np.eye(k)
    else:
        i, j = np.indices((k, k))
        d = np.abs(i - j).astype(float)
        W = d / (k - 1) if weights == "linear" else (d / (k - 1)) ** 2
    denom = (W * E).sum()
    return np.nan if denom == 0 else 1.0 - (W * O).sum() / denom


def boot_ci(fn, n, seed=BOOT_SEED):
    if n == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        v = fn(idx)
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            vals.append(v)
    if not vals:
        return (np.nan, np.nan)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def kappa_word(k):
    """Descriptive band for a kappa value, used in the plain-language text."""
    if np.isnan(k):
        return "not computable"
    if k < 0.20:
        return "slight"
    if k < 0.40:
        return "fair"
    if k < 0.60:
        return "moderate"
    if k < 0.80:
        return "substantial"
    return "almost perfect"


def compute_stats(df):
    """All headline numbers for one group of rows. Returns a dict."""
    ok = df[df["human_label"] != UNVERIFIED]
    n = len(ok)
    m_lab = ok["model_label"].to_numpy()
    h_lab = ok["human_label"].to_numpy()
    m_ord = ok["model_ord"].to_numpy(dtype=float)
    h_ord = ok["human_ord"].to_numpy(dtype=float)

    s = {}
    s["n_total"] = len(df)
    s["n_ok"] = n
    s["n_unverified"] = int((df["human_label"] == UNVERIFIED).sum())
    s["exact"] = float((m_lab == h_lab).mean()) if n else np.nan
    s["within1"] = float((np.abs(h_ord - m_ord) <= 1).mean()) if n else np.nan
    s["k_lin"] = cohen_kappa(m_lab, h_lab, CATEGORIES, weights="linear")
    s["k_quad"] = cohen_kappa(m_lab, h_lab, CATEGORIES, weights="quadratic")
    s["bias"] = float(np.mean(h_ord - m_ord)) if n else np.nan
    s["ci_within1"] = boot_ci(
        lambda i: float(np.mean(np.abs(h_ord[i] - m_ord[i]) <= 1)), n)
    s["ci_k_lin"] = boot_ci(
        lambda i: cohen_kappa(m_lab[i], h_lab[i], CATEGORIES,
                              weights="linear"), n)
    s["ci_bias"] = boot_ci(lambda i: float(np.mean(h_ord[i] - m_ord[i])), n)

    okb = df[df["human_bin"] != "unverified"]
    mb = okb["model_bin"].to_numpy()
    hb = okb["human_bin"].to_numpy()
    tp = int(((mb == "misleading") & (hb == "misleading")).sum())
    fp = int(((mb == "misleading") & (hb == "accurate")).sum())
    fn = int(((mb == "accurate") & (hb == "misleading")).sum())
    tn = int(((mb == "accurate") & (hb == "accurate")).sum())
    nb = tp + fp + fn + tn
    s.update({"tp": tp, "fp": fp, "fn": fn, "tn": tn, "n_bin": nb})
    s["b_acc"] = (tp + tn) / nb if nb else np.nan
    s["b_prec"] = tp / (tp + fp) if (tp + fp) else np.nan
    s["b_rec"] = tp / (tp + fn) if (tp + fn) else np.nan
    s["b_kappa"] = cohen_kappa(mb, hb, ["accurate", "misleading"])
    s["ci_b_acc"] = boot_ci(
        lambda i: float(np.mean(mb[i] == hb[i])), nb)
    s["ci_b_kappa"] = boot_ci(
        lambda i: cohen_kappa(mb[i], hb[i], ["accurate", "misleading"]), nb)

    if "is_checkable_claim" in df.columns and df["is_checkable_claim"].notna().any():
        cc = df["is_checkable_claim"].astype(bool)
        s["checkable_share"] = float(cc.mean())
        s["checkable_n"] = int(cc.sum())
    else:
        s["checkable_share"] = np.nan
        s["checkable_n"] = 0
    return s


# ============================ FIGURES =======================================
def fig_to_b64(fig, path):
    fig.savefig(path, dpi=160, bbox_inches="tight")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def make_fig_distribution(df, path):
    counts = pd.DataFrame({
        "Model": df["model_label"].value_counts(),
        "Rater": df["human_label"].value_counts(),
    }).reindex(CATEGORIES + [UNVERIFIED]).fillna(0)
    x = np.arange(len(counts))
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    ax.bar(x - 0.19, counts["Model"], width=0.38, label="Model", color=BLUE)
    ax.bar(x + 0.19, counts["Rater"], width=0.38, label="Rater", color=GOLD)
    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY[c] for c in counts.index], fontsize=9)
    ax.set_ylabel("Number of posts")
    ax.set_title("How often each label was used", fontsize=12, pad=10)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    for xi, (mv, hv) in enumerate(zip(counts["Model"], counts["Rater"])):
        if mv:
            ax.text(xi - 0.19, mv + 0.3, int(mv), ha="center", fontsize=9)
        if hv:
            ax.text(xi + 0.19, hv + 0.3, int(hv), ha="center", fontsize=9)
    return fig_to_b64(fig, path)


def make_fig_confusion(df, path):
    conf = pd.crosstab(df["model_label"], df["human_label"]).reindex(
        index=CATEGORIES, columns=CATEGORIES + [UNVERIFIED]).fillna(0).astype(int)
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    im = ax.imshow(conf.values, cmap="Blues", vmin=0)
    ax.set_xticks(range(conf.shape[1]))
    ax.set_xticklabels([DISPLAY[c] for c in conf.columns], rotation=30,
                       ha="right", fontsize=9)
    ax.set_yticks(range(conf.shape[0]))
    ax.set_yticklabels([DISPLAY[c] for c in conf.index], fontsize=9)
    ax.set_xlabel("Rater's label")
    ax.set_ylabel("Model's label")
    ax.set_title("Where the model and the rater agree and disagree",
                 fontsize=12, pad=10)
    vmax = conf.values.max()
    for i in range(conf.shape[0]):
        for j in range(conf.shape[1]):
            v = conf.values[i, j]
            if v == 0:
                continue
            color = "white" if v > vmax * 0.55 else NAVY
            weight = "bold" if (j < 5 and i == j) else "normal"
            ax.text(j, i, v, ha="center", va="center", color=color,
                    fontsize=11, fontweight=weight)
    for d in range(5):
        ax.add_patch(plt.Rectangle((d - 0.5, d - 0.5), 1, 1, fill=False,
                                   edgecolor=GOLD, lw=2))
    fig.colorbar(im, ax=ax, shrink=0.75, label="Posts")
    return fig_to_b64(fig, path)


def make_fig_distance(df, path):
    ok = df[df["human_label"] != UNVERIFIED]
    diff = (ok["human_ord"] - ok["model_ord"]).astype(int)
    counts = diff.value_counts().reindex(range(-4, 5), fill_value=0)
    colors = [BLUE if d == 0 else (GOLD if abs(d) == 1 else RED)
              for d in counts.index]
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    ax.bar(counts.index, counts.values, color=colors)
    ax.set_xticks(range(-4, 5))
    ax.set_xlabel("Rater's label minus model's label (in categories)")
    ax.set_ylabel("Number of posts")
    ax.set_title("How far apart the two labels are", fontsize=12, pad=10)
    ax.spines[["top", "right"]].set_visible(False)
    for xi, v in zip(counts.index, counts.values):
        if v:
            ax.text(xi, v + 0.3, int(v), ha="center", fontsize=9)
    ax.text(0.02, 0.95,
            "0 = same label   right of 0 = rater judged the post MORE accurate",
            transform=ax.transAxes, fontsize=9, va="top")
    return fig_to_b64(fig, path)


def make_fig_binary(df, path):
    okb = df[df["human_bin"] != "unverified"]
    conf = pd.crosstab(okb["model_bin"], okb["human_bin"]).reindex(
        index=["misleading", "accurate"],
        columns=["misleading", "accurate"]).fillna(0).astype(int)
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    im = ax.imshow(conf.values, cmap="Blues", vmin=0)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Misleading", "Accurate"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Misleading", "Accurate"])
    ax.set_xlabel("Rater says")
    ax.set_ylabel("Model says")
    ax.set_title("The two-way split used in the regression",
                 fontsize=12, pad=10)
    vmax = conf.values.max()
    notes = [["agree", "model flags,\nrater does not"],
             ["model misses,\nrater flags", "agree"]]
    for i in range(2):
        for j in range(2):
            v = conf.values[i, j]
            color = "white" if v > vmax * 0.55 else NAVY
            ax.text(j, i, f"{v}\n{notes[i][j]}", ha="center", va="center",
                    color=color, fontsize=10)
    for d in range(2):
        ax.add_patch(plt.Rectangle((d - 0.5, d - 0.5), 1, 1, fill=False,
                                   edgecolor=GOLD, lw=2))
    return fig_to_b64(fig, path)


def make_fig_per_category(df, path):
    ok = df[df["human_label"] != UNVERIFIED]
    rows = []
    for c in CATEGORIES:
        sub = ok[ok["model_label"] == c]
        if not len(sub):
            rows.append((c, 0.0, 0.0))
            continue
        rows.append((c,
                     float((sub["model_label"] == sub["human_label"]).mean()),
                     float((np.abs(sub["human_ord"] - sub["model_ord"]) <= 1)
                           .mean())))
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    ax.bar(x - 0.19, [r[1] for r in rows], width=0.38,
           label="Same label", color=BLUE)
    ax.bar(x + 0.19, [r[2] for r in rows], width=0.38,
           label="Within one category", color=GOLD)
    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY[r[0]] for r in rows], fontsize=9)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Share of posts")
    ax.set_title("Agreement by model label", fontsize=12, pad=10)
    ax.legend(frameon=False, ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    for xi, r in enumerate(rows):
        ax.text(xi - 0.19, r[1] + 0.02, f"{r[1]:.0%}", ha="center", fontsize=8)
        ax.text(xi + 0.19, r[2] + 0.02, f"{r[2]:.0%}", ha="center", fontsize=8)
    return fig_to_b64(fig, path)


# ============================ HTML ==========================================
def pct(v):
    return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) \
        else f"{v:.0%}"


def num(v, d=2):
    return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) \
        else f"{v:.{d}f}"


def ci_txt(ci):
    if np.isnan(ci[0]):
        return ""
    return f" (95% CI {ci[0]:.2f} to {ci[1]:.2f})"


def build_html(s, figs, worst_rows, per_rater, meta):
    bias_dir = ("the rater tends to judge posts as MORE accurate than the model"
                if s["bias"] > 0 else
                "the rater tends to judge posts as LESS accurate than the model"
                if s["bias"] < 0 else "there is no systematic direction")

    cards = f"""
    <div class="cards">
      <div class="card"><div class="big">{pct(s['within1'])}</div>
        <div class="lbl">of posts land within one category of the model's label
        {ci_txt(s['ci_within1'])}</div></div>
      <div class="card"><div class="big">{num(s['k_lin'])}</div>
        <div class="lbl">weighted kappa on the 5-point scale, which counts as
        {kappa_word(s['k_lin'])} agreement beyond chance{ci_txt(s['ci_k_lin'])}</div></div>
      <div class="card"><div class="big">{pct(s['b_acc'])}</div>
        <div class="lbl">agreement on the misleading vs accurate split used in
        the regression{ci_txt(s['ci_b_acc'])}</div></div>
      <div class="card"><div class="big">{num(s['b_kappa'])}</div>
        <div class="lbl">kappa for that split, which counts as
        {kappa_word(s['b_kappa'])} agreement beyond chance{ci_txt(s['ci_b_kappa'])}</div></div>
    </div>"""

    rater_table = ""
    if len(per_rater) > 1:
        rows = "".join(
            f"<tr><td>{name}</td><td>{st['n_total']}</td>"
            f"<td>{pct(st['exact'])}</td><td>{pct(st['within1'])}</td>"
            f"<td>{num(st['k_lin'])}</td><td>{pct(st['b_acc'])}</td>"
            f"<td>{num(st['b_kappa'])}</td><td>{num(st['bias'])}</td></tr>"
            for name, st in per_rater)
        rater_table = f"""
    <h2>Per rater</h2>
    <p>The two raters reviewed different posts, so their numbers describe
       different samples and are not a reliability comparison. Large
       differences would still suggest the raters apply the scale with
       different strictness.</p>
    <table>
      <tr><th>Rater</th><th>Posts</th><th>Same label</th><th>Within one</th>
          <th>Weighted kappa</th><th>Binary agreement</th><th>Binary kappa</th>
          <th>Signed difference</th></tr>
      {rows}
    </table>"""

    worst_html = "".join(
        f"<tr><td>{int(r['abs_err'])}</td><td>{DISPLAY[r['model_label']]}</td>"
        f"<td>{DISPLAY[r['human_label']]}</td>"
        f"<td class='mono'>{r['post_id']}</td></tr>"
        for _, r in worst_rows.iterrows())

    checkable_line = ""
    if not np.isnan(s["checkable_share"]):
        checkable_line = (f"<li>In {pct(s['checkable_share'])} of posts "
                          f"({s['checkable_n']} of {s['n_total']}), the rater "
                          f"confirmed that the highlighted text is a checkable "
                          f"factual claim. This supports the claim extraction "
                          f"step of the pipeline, independently of the "
                          f"accuracy verdicts.</li>")

    unv_line = ""
    if s["n_unverified"]:
        unv_line = (f"<li>The rater marked {s['n_unverified']} post(s) as "
                    f"unverified. These cannot be placed on the accuracy scale "
                    f"and are excluded from the agreement numbers.</li>")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Human validation of the model's accuracy labels</title>
<style>
  body {{ font-family: Georgia, "Times New Roman", serif; color: {NAVY};
         max-width: 880px; margin: 0 auto; padding: 28px 20px 60px;
         line-height: 1.55; }}
  h1 {{ font-size: 26px; border-bottom: 3px solid {GOLD};
        padding-bottom: 8px; }}
  h2 {{ font-size: 19px; margin-top: 34px; color: {BLUE}; }}
  .meta {{ color: #566b82; font-size: 13px; }}
  .cards {{ display: flex; gap: 12px; flex-wrap: wrap; margin: 18px 0; }}
  .card {{ flex: 1 1 190px; background: {LIGHT}; border-radius: 8px;
           padding: 14px 16px; }}
  .big {{ font-size: 30px; font-weight: bold; color: {BLUE}; }}
  .lbl {{ font-size: 12.5px; margin-top: 4px; }}
  .box {{ background: {LIGHT}; border-left: 4px solid {BLUE};
          padding: 10px 16px; border-radius: 0 6px 6px 0; margin: 14px 0;
          font-size: 14.5px; }}
  .warn {{ border-left-color: {GOLD}; }}
  img {{ max-width: 100%; display: block; margin: 10px auto; }}
  .cap {{ font-size: 13px; color: #566b82; margin: 2px 0 26px; }}
  table {{ border-collapse: collapse; font-size: 13.5px; width: 100%; }}
  th, td {{ border: 1px solid #c9d6e4; padding: 5px 9px; text-align: left; }}
  th {{ background: {LIGHT}; }}
  .mono {{ font-family: Consolas, monospace; font-size: 11.5px; }}
  ul {{ padding-left: 20px; }}
  li {{ margin-bottom: 7px; }}
</style>
</head>
<body>

<h1>Human validation of the model's accuracy labels</h1>
<p class="meta">Generated {meta['date']} from {meta['files']}.
   {meta['nrater']} of the two raters returned results so far.</p>

<h2>What was done</h2>
<p>A balanced sample of posts was drawn from the analysis pool, with equal
   numbers from each of the five model accuracy categories. A student rater,
   blind to the model's labels, read each post with the fact-checked claims
   highlighted, searched public sources, and assigned an accuracy label using
   the same five-point scale. This report compares the two sets of labels for
   the {s['n_total']} posts returned so far.</p>

<div class="box">In short: the model and the human rater usually place a post
   in the same region of the accuracy scale, and they reach
   {kappa_word(s['b_kappa'])} chance-corrected agreement on the misleading vs
   accurate distinction that the regressions actually use. Exact category
   matches are less common, and the model labels posts somewhat more harshly
   than the rater does.</div>

{cards}

<h2>1. Which labels each side used</h2>
<img src="data:image/png;base64,{figs['dist']}">
<p class="cap">The sample contains an equal number of posts per model category
   by design. The rater's bars show how the same posts were judged by a human.
   The rater used the True label most often, which already hints at the
   direction of the disagreement.</p>

<h2>2. Where agreement and disagreement happen</h2>
<img src="data:image/png;base64,{figs['conf']}">
<p class="cap">Each cell counts posts. The gold boxes on the diagonal are exact
   matches. Cells near the diagonal are one-category differences, which are
   common and expected on a five-point accuracy scale. Cells far from the
   diagonal are genuine disagreements, and those posts are listed at the end of
   this report.</p>

<h2>3. How far apart the labels are</h2>
<img src="data:image/png;base64,{figs['dist2']}">
<p class="cap">Most posts sit at 0 or one step away. The bars lean to one side
   of zero: {bias_dir}, on average by {num(abs(s['bias']))}
   categories{ci_txt(s['ci_bias'])}.</p>

<h2>4. The split that matters for the thesis</h2>
<img src="data:image/png;base64,{figs['bin']}">
<p class="cap">The regressions do not use the five categories directly. They
   rest on a two-way split, with Pants on Fire, False, and Mostly False counted
   as misleading. On this split the model and the rater agree on
   {pct(s['b_acc'])} of posts (kappa {num(s['b_kappa'])},
   {kappa_word(s['b_kappa'])} agreement). When the model flags a post as
   misleading, the rater agrees in {pct(s['b_prec'])} of cases. When the rater
   considers a post misleading, the model has flagged it in
   {pct(s['b_rec'])} of cases, so the model rarely misses what the human
   flags.</p>

<h2>5. Agreement by category</h2>
<img src="data:image/png;base64,{figs['percat']}">
<p class="cap">Agreement tends to be strongest at the ends of the scale and
   weakest in the middle categories, where the label definitions overlap the
   most. This is the usual pattern for fact-checking scales.</p>

<h2>What this suggests for the thesis</h2>
<ul>
  <li>The validation supports the pipeline at the level that matters most. The
      misleading vs accurate distinction used in the regressions reaches
      {kappa_word(s['b_kappa'])} chance-corrected agreement with an independent
      human rater.</li>
  <li>The model appears stricter than the human rater. Disagreements are not
      random: {bias_dir}. Some posts the model labels as misleading may
      therefore be borderline cases rather than clear misinformation.</li>
  <li>This measurement error works against finding an effect, not for it. If
      some posts counted as misleading are borderline, the contrast between
      misleading and accurate posts in the regressions is diluted, so
      estimates are, if anything, attenuated toward zero. The null results
      should be read with that in mind.</li>
  {checkable_line}
  {unv_line}
</ul>

<div class="box warn"><b>Caveats.</b> {meta['caveat_rater']} The sample is
   balanced across model categories by construction, so the shares above
   describe this stratified sample, not the full corpus, where accurate posts
   are far more common. The two raters review disjoint posts, so no
   inter-rater reliability can be computed. The validation targets the
   post-level five-category label, while the regressions use a claim-level
   misleading count, so it supports the labelling pipeline in general rather
   than the exact regression variable.</div>

{rater_table}

<h2>Appendix: the largest disagreements</h2>
<p>Distance is the number of categories between the two labels. These posts
   are worth reading by hand.</p>
<table>
  <tr><th>Distance</th><th>Model says</th><th>Rater says</th><th>Post</th></tr>
  {worst_html}
</table>

</body>
</html>"""


# ============================ MAIN ==========================================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    key = pd.read_excel(KEY_XLSX, dtype=str).fillna("")
    key["post_id"] = key["post_id"].astype(str).str.strip()
    key["model_label"] = key["model_label"].astype(str).str.strip()
    print(f"  Read master_key.xlsx: {len(key)} rows.")

    frames = []
    for p in RATER_FILES:
        if not os.path.exists(p):
            print(f"  Skipping missing file: {p}")
            continue
        frames.append(read_rater(p))
    if not frames:
        raise SystemExit("No rater files found. Check RATER_FILES.")
    hum = pd.concat(frames, ignore_index=True)

    df = hum.merge(key, on="post_id", how="left", suffixes=("", "_key"))
    missing = df["model_label"].isna() | (df["model_label"] == "")
    if missing.any():
        print(f"  WARNING: {int(missing.sum())} rated posts absent from the "
              f"key were dropped.")
        df = df[~missing].copy()

    # check the rater picked the button matching their assignment in the key
    if "rater" in df.columns and "rater_key" in df.columns:
        mism = df["rater"].astype(str).str.strip() != \
            df["rater_key"].astype(str).str.strip()
        if mism.any():
            print(f"  WARNING: {int(mism.sum())} posts were answered under a "
                  f"different rater name than assigned in the key. The rater "
                  f"may have clicked the wrong button on entry. Results are "
                  f"still usable, but check before pooling.")

    df["model_ord"] = df["model_label"].map(ORD)
    df["human_ord"] = df["human_label"].map(ORD)
    df["model_bin"] = np.where(df["model_label"].isin(MISLEADING),
                               "misleading", "accurate")
    df["human_bin"] = np.where(df["human_label"].isin(MISLEADING),
                               "misleading", "accurate")
    df.loc[df["human_label"] == UNVERIFIED, "human_bin"] = "unverified"
    df["abs_err"] = (df["human_ord"] - df["model_ord"]).abs()

    # ---- stats ----
    s = compute_stats(df)
    per_rater = []
    if df["source_file"].nunique() > 1:
        for name, sub in df.groupby("source_file"):
            per_rater.append((name, compute_stats(sub)))

    # ---- figures ----
    figs = {
        "dist": make_fig_distribution(df, os.path.join(OUT_DIR, "fig_distribution.png")),
        "conf": make_fig_confusion(df, os.path.join(OUT_DIR, "fig_confusion.png")),
        "dist2": make_fig_distance(df, os.path.join(OUT_DIR, "fig_distance.png")),
        "bin": make_fig_binary(df, os.path.join(OUT_DIR, "fig_binary.png")),
        "percat": make_fig_per_category(df, os.path.join(OUT_DIR, "fig_per_category.png")),
    }

    worst = (df[df["human_label"] != UNVERIFIED]
             .sort_values("abs_err", ascending=False)
             .loc[lambda d: d["abs_err"] >= 2]
             .head(12)[["abs_err", "model_label", "human_label", "post_id"]])

    nrater = df["source_file"].nunique()
    meta = {
        "date": datetime.date.today().isoformat(),
        "files": ", ".join(sorted(df["source_file"].unique())),
        "nrater": "One" if nrater == 1 else "Both",
        "caveat_rater": ("These results come from a single rater reviewing 50 "
                         "posts, so all numbers carry wide uncertainty and may "
                         "shift when the second rater's results arrive."
                         if nrater == 1 else
                         "These results pool two raters who reviewed 50 posts "
                         "each."),
    }

    html = build_html(s, figs, worst, per_rater, meta)
    with open(OUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)

    keep = [c for c in ["post_id", "rater", "source_file", "model_label",
                        "human_label", "abs_err", "model_bin", "human_bin",
                        "is_checkable_claim", "is_political", "n_claims",
                        "order_shown", "seconds_spent", "title", "author",
                        "date"] if c in df.columns]
    df[keep].to_csv(os.path.join(OUT_DIR, "validation_merged.csv"),
                    index=False, encoding="utf-8-sig")

    print(f"\n  Wrote {OUT_HTML}")
    print(f"  Wrote validation_merged.csv and fig_*.png to {OUT_DIR}")
    print("  Open validation_report.html in a browser. It is self-contained "
          "and can be sent as a single file.")


if __name__ == "__main__":
    main()
