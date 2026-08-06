"""
Build the regression dataset for Study 1 (Hive)  -- with extended engagement
============================================================================
Joins four outputs into one analysis table keyed on doc_id (author/permlink):
  1. model4/posts.csv                accuracy labels (final_label, ordinal, source)
  2. hive_sample_master.parquet      NON-engagement controls only: category,
                                     is_political, sample_stratum, month, created,
                                     body length. (Its engagement columns are no
                                     longer used; the extended pull supersedes them.)
  3. hive_emotion_scores.parquet     7 emotions + emotional_language
  4. hive_engagement_extended.parquet   the authoritative, finalized engagement
                                     panel from HiveSQL: payout split, net_votes,
                                     comments, up/down votes, distinct voters,
                                     reblogs, net_rshares + curation weights,
                                     promoted, and author reputation + stake.

Why engagement now comes from the extended pull: it was queried after every
post's 7-day payout window closed, so payouts are final (pending ~ 0) and the
finalization caveat is resolved. If the extended file is absent, the script
falls back to the sample's engagement columns.

Engagement outcomes available downstream (logged in R as needed):
  payout (reward), net_votes / distinct_voters / upvotes (breadth),
  downvotes (pushback), comments (discussion), reblogs (diffusion),
  net_rshares / total_vote_weight (stake-weighted curation).
Author standing (controls, or absorbed by author FE):
  reputation_ui, effective_vesting (Hive Power), post_count.

Run:  python build_regression_dataset.py
Deps: pip install pandas numpy pyarrow openpyxl
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL4_CSV   = r"C:\Users\batuh\PycharmProjects\Thesis\fact_checked_hive_pipeline\4.4. model cascade (model3,1)\hive_outputs\model4\posts.csv"
SAMPLE_FILE  = r"C:\Users\batuh\PycharmProjects\Thesis\fact_checked_hive_pipeline\hive sample preparation (politic-non-politic)\hive_sample_master.parquet"
EMOTION_FILE = r"C:\Users\batuh\PycharmProjects\Thesis\fact_checked_hive_pipeline\hive_emotion_scores.parquet"
EXTENDED_FILE = r"C:\Users\batuh\PycharmProjects\Thesis\fact_checked_hive_pipeline\7. hive new dataset\hive_engagement_extended.parquet"
OUTPUT_DIR   = r"C:\Users\batuh\PycharmProjects\Thesis\fact_checked_hive_pipeline\6. regression"

FALSE_SIDE = {"pants_on_fire", "false", "mostly_false"}
TRUE_SIDE = {"mostly_true", "true"}

# Sample engagement fallback (used only if the extended file is missing).
VOTES_CANDS = ["net_votes", "vote_count", "num_votes"]
COMMENT_CANDS = ["children", "num_comments", "comment_count", "replies"]
PAYOUT_CANDS = ["payout_combined", "total_payout_value", "payout", "author_payout"]

EMOTIONS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]

# Numeric columns to pull from the extended engagement pull.
EXT_NUMERIC = [
    "net_votes", "comments", "upvotes", "downvotes", "distinct_voters", "n_votes_raw",
    "reblogs", "net_rshares", "abs_rshares", "vote_rshares", "total_vote_weight",
    "total_payout_value", "curator_payout_value", "pending_payout_value", "author_rewards",
    "promoted", "percent_hbd",
    "reputation", "reputation_ui", "vesting_shares", "received_vesting_shares",
    "delegated_vesting_shares", "post_count",
]

FINAL_COLS = [
    # keys & bookkeeping
    "doc_id", "author", "analysis_sample", "label_source", "final_label",
    # accuracy
    "misinformation", "misinformation_strict", "accuracy_ordinal", "inaccuracy_severity",
    # reward
    "payout", "total_payout_value", "curator_payout_value", "pending_payout_value", "author_rewards",
    # breadth / pushback / discussion / diffusion
    "net_votes", "upvotes", "downvotes", "distinct_voters", "n_votes_raw",
    "comments", "reblogs",
    # stake-weighted curation
    "net_rshares", "abs_rshares", "vote_rshares", "total_vote_weight",
    # promotion
    "promoted", "percent_hbd",
    # author standing (controls)
    "reputation", "reputation_ui", "vesting_shares", "received_vesting_shares",
    "delegated_vesting_shares", "effective_vesting", "post_count",
    # emotion
    "emotional_language", *EMOTIONS, "dominant_emotion",
    # controls / fixed effects / clustering
    "is_political", "category", "sample_stratum", "month", "created", "log_len",
]

try:
    from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE
except Exception:                                    # pragma: no cover
    ILLEGAL_CHARACTERS_RE = re.compile(r"[\000-\010]|[\013-\014]|[\016-\037]")


# ============================================================================
# HELPERS
# ============================================================================

def ensure_doc_id(df: pd.DataFrame, name: str) -> pd.DataFrame:
    if "doc_id" in df.columns:
        return df
    if "post_key" in df.columns:
        df = df.copy(); df["doc_id"] = df["post_key"].astype(str); return df
    if {"author", "permlink"} <= set(df.columns):
        df = df.copy()
        df["doc_id"] = df["author"].astype(str) + "/" + df["permlink"].astype(str)
        return df
    raise KeyError(f"{name}: need doc_id, post_key, or author+permlink. Columns: {list(df.columns)}")


def first_present(df, cands):
    for c in cands:
        if c in df.columns:
            return c
    return None


def to_misinfo(label):
    l = str(label).strip().lower()
    if l in FALSE_SIDE or l == "misinformation":
        return 1.0
    if l in TRUE_SIDE or l == "true":
        return 0.0
    return np.nan


def to_num(series):
    """Numeric from ints/floats, or from strings like '1.234 HBD' / '1,234 VESTS'."""
    s = pd.to_numeric(series, errors="coerce")
    need = s.isna() & series.notna()
    if need.any():
        ext = (series[need].astype(str).str.replace(",", "", regex=False)
               .str.extract(r"(-?\d+(?:\.\d+)?)")[0])
        s.loc[need] = pd.to_numeric(ext, errors="coerce")
    return s


def to01(v):
    if pd.isna(v):
        return np.nan
    if isinstance(v, (bool, np.bool_)):
        return float(v)
    s = str(v).strip().lower()
    if s in ("1", "true", "t", "yes"):
        return 1.0
    if s in ("0", "false", "f", "no"):
        return 0.0
    try:
        return float(float(s) != 0)
    except Exception:
        return np.nan


def sanitize_for_excel(df):
    out = df.copy()
    for col in out.columns:
        out[col] = out[col].map(
            lambda v: ILLEGAL_CHARACTERS_RE.sub("", v) if isinstance(v, str) else v)
    return out


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    has_ext = Path(EXTENDED_FILE).exists()

    # ---- 1. accuracy labels from Model 4 -----------------------------------
    m4_full = ensure_doc_id(pd.read_csv(MODEL4_CSV, dtype=str).fillna(""), "model4")
    keep4 = [c for c in ["doc_id", "final_label", "accuracy_ordinal", "label_source"]
             if c in m4_full.columns]
    m4 = m4_full[keep4].copy()
    m4["misinformation"] = m4["final_label"].map(to_misinfo)
    m4["misinformation_strict"] = np.where(
        m4.get("label_source", "") == "model3", m4["misinformation"], np.nan)
    m4["accuracy_ordinal"] = pd.to_numeric(
        m4.get("accuracy_ordinal", pd.Series(index=m4.index, dtype=object)), errors="coerce")
    m4["inaccuracy_severity"] = 6 - m4["accuracy_ordinal"]
    base = m4
    print(f"Model 4 posts: {len(base):,}")
    print(m4_full.columns.tolist())
    print(m4_full.head(3).to_dict("records"))

    # ---- 2. controls from the sample (engagement only as fallback) ---------
    samp = ensure_doc_id(pd.read_parquet(SAMPLE_FILE), "sample")
    want = ["doc_id", "author", "category", "is_political", "sample_stratum",
            "month", "created", "body_for_analysis_len"]
    s = samp[[c for c in want if c in samp.columns]].copy()
    if not has_ext:
        vcol, ccol, pcol = (first_present(samp, VOTES_CANDS),
                            first_present(samp, COMMENT_CANDS),
                            first_present(samp, PAYOUT_CANDS))
        if vcol: s["net_votes"] = to_num(samp[vcol])
        if ccol: s["comments"] = to_num(samp[ccol])
        if pcol: s["payout"] = to_num(samp[pcol])
        print(f"(no extended file; engagement from sample -> "
              f"votes:{vcol} comments:{ccol} payout:{pcol})")
    if "month" not in s.columns and "created" in s.columns:
        s["month"] = pd.to_datetime(s["created"], errors="coerce").dt.strftime("%Y-%m")
    if "body_for_analysis_len" in s.columns:
        s["log_len"] = np.log1p(
            pd.to_numeric(s["body_for_analysis_len"], errors="coerce").clip(lower=0))
        s = s.drop(columns=["body_for_analysis_len"])
    if "is_political" in s.columns:
        s["is_political"] = s["is_political"].map(to01)
    base = base.merge(s, on="doc_id", how="left")

    # ---- 3. emotion scores -------------------------------------------------
    emo = ensure_doc_id(pd.read_parquet(EMOTION_FILE), "emotion")
    emo_keep = ["doc_id"] + [c for c in EMOTIONS + ["emotional_language", "dominant_emotion"]
                             if c in emo.columns]
    base = base.merge(emo[emo_keep], on="doc_id", how="left")

    # ---- 4. extended engagement panel (authoritative) ----------------------
    if has_ext:
        ext = ensure_doc_id(pd.read_parquet(EXTENDED_FILE), "extended")
        if "children" in ext.columns and "comments" not in ext.columns:
            ext = ext.rename(columns={"children": "comments"})
        for c in EXT_NUMERIC:
            if c in ext.columns:
                ext[c] = to_num(ext[c])
        # payout (reward) = total + pending; finalized posts have pending ~ 0
        tp = ext["total_payout_value"] if "total_payout_value" in ext.columns else 0.0
        pp = ext["pending_payout_value"] if "pending_payout_value" in ext.columns else 0.0
        ext["payout"] = pd.to_numeric(tp, errors="coerce").fillna(0) + \
                        pd.to_numeric(pp, errors="coerce").fillna(0)
        # effective Hive Power = own + received - delegated vesting
        vs  = ext.get("vesting_shares", 0)
        rvs = ext.get("received_vesting_shares", 0)
        dvs = ext.get("delegated_vesting_shares", 0)
        ext["effective_vesting"] = (pd.to_numeric(vs, errors="coerce").fillna(0)
                                    + pd.to_numeric(rvs, errors="coerce").fillna(0)
                                    - pd.to_numeric(dvs, errors="coerce").fillna(0))
        if "promoted" in ext.columns:
            ext["promoted"] = ext["promoted"].fillna(0)
        for c in ["reblogs", "upvotes", "downvotes", "distinct_voters", "n_votes_raw"]:
            if c in ext.columns:
                ext[c] = ext[c].fillna(0)
        ext_take = ["doc_id", "payout", "effective_vesting"] + EXT_NUMERIC
        ext = ext[[c for c in ext_take if c in ext.columns]]
        base = base.merge(ext, on="doc_id", how="left")
        print(f"Extended engagement merged for "
              f"{int(base['payout'].notna().sum()):,} posts")
    else:
        print("Extended engagement file NOT found; using sample engagement only.")

    # ---- 5. analysis flag + fixed final schema -----------------------------
    base["analysis_sample"] = base["misinformation"].notna()
    present = [c for c in FINAL_COLS if c in base.columns]
    missing = [c for c in FINAL_COLS if c not in base.columns]
    base = base[present]

    # ---- 6. report ---------------------------------------------------------
    print("\n" + "=" * 64)
    print(f"Merged rows:                 {len(base):,}")
    print(f"Analysis sample (labelled):  {int(base['analysis_sample'].sum()):,}")
    print(f"  misinformation 1/0/NA:     {base['misinformation'].value_counts(dropna=False).to_dict()}")
    if base["misinformation_strict"].notna().any():
        print(f"  strict (model3 only):      {base['misinformation_strict'].value_counts(dropna=False).to_dict()}")
    for col in ["payout", "net_votes", "comments", "upvotes", "downvotes",
                "distinct_voters", "reblogs", "net_rshares", "reputation_ui"]:
        if col in base.columns:
            x = pd.to_numeric(base[col], errors="coerce")
            print(f"  {col:<16} mean/med/max: {x.mean():.2f} / {x.median():.2f} / {x.max():.2f}")
    if missing:
        print(f"  NOTE missing from output:  {missing}")

    # correlation among the candidate engagement outcomes (analysis sample)
    eng_for_corr = ["payout", "net_votes", "distinct_voters", "upvotes",
                    "downvotes", "comments", "reblogs", "net_rshares"]
    chk = pd.DataFrame()
    for c in eng_for_corr:
        if c in base.columns:
            chk["log_" + c] = np.log1p(pd.to_numeric(base[c], errors="coerce").clip(lower=0))
    chk = chk.loc[base["analysis_sample"].values]
    if chk.shape[1] > 1:
        print("\nEngagement-outcome correlations (logged, analysis sample):")
        print(chk.corr().round(2).to_string())

    # ---- 7. write ----------------------------------------------------------
    parquet_path = out_dir / "hive_regression_dataset.parquet"
    xlsx_path = out_dir / "hive_regression_dataset.xlsx"
    base.to_parquet(parquet_path, index=False)
    try:
        sanitize_for_excel(base).to_excel(xlsx_path, index=False)
        print(f"\nWrote: {parquet_path}\nWrote: {xlsx_path}")
    except Exception as e:                            # pragma: no cover
        print(f"\nWrote: {parquet_path}")
        print(f"(Excel write skipped: {e}. Read the parquet in R via arrow.)")



if __name__ == "__main__":
    main()
