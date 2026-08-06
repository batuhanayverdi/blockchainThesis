"""
Hive engagement-variable extraction for the existing Study 1 sample
===================================================================
Pulls richer reward / curation / diffusion fields for the SAME 6,242 posts and
the SAME window (2024-09-01 to 2025-01-21). Everything joins on (author,
permlink), so the dataset and date range are unchanged; this only adds columns.

What it adds, by source:
  Comments  -> net_rshares, abs_rshares, total_vote_weight, curator_payout_value,
               author_rewards, promoted, author_reputation, percent_hbd
  TxVotes   -> upvotes, downvotes, distinct_voters, sum_rshares (per post)
  Accounts  -> author reputation + vesting (Hive Power proxy)
  TxCustoms -> reblog count per post (diffusion)   [best-effort, verify schema]

Outputs (in OUTPUT_DIR), each checkpointed so reruns skip finished steps:
  eng_comments.parquet, eng_votes.parquet, eng_accounts.parquet,
  eng_reblogs.parquet, and the merged hive_engagement_extended.parquet

Usage:
  1. Fill HIVESQL_PASSWORD and set POSTS_FILE / OUTPUT_DIR
  2. Run. It prints each table's real columns first; confirm the names look
     right, then it runs the queries. Rerun to resume.
Deps: pip install pyodbc pandas pyarrow openpyxl
"""

import os
import time
import pyodbc
import pandas as pd

# ============================================================================
# CONFIGURATION
# ============================================================================

HIVESQL_PASSWORD = "password"  # fill in

# The existing sample. Either the merged dataset or hive_sample_master.parquet;
# both carry author + permlink (and the dataset carries doc_id = author/permlink).
POSTS_FILE = r"C:\Users\batuh\PycharmProjects\Thesis\fact_checked_hive_pipeline\6. regression\hive_regression_dataset.xlsx"
OUTPUT_DIR = r"C:\Users\batuh\PycharmProjects\Thesis\fact_checked_hive_pipeline\7. hive new dataset"

# Same window as the scrape. Reblogs of late posts can land a few days after, so
# the reblog scan runs to a slightly later bound.
SCOPE_START = "2024-09-01 00:00:00"
SCOPE_END   = "2025-01-21 00:00:00"
REBLOG_END  = "2025-01-28 00:00:00"

DO_REBLOGS = True   # set False to skip the best-effort reblog step

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================================
# CONNECTION
# ============================================================================

def open_conn():
    conn = pyodbc.connect(
        "DRIVER={SQL Server};"
        "SERVER=vip.hivesql.io;"
        "DATABASE=DBHive;"
        "UID=Hive-batuhanayverdi01;"
        f"PWD={HIVESQL_PASSWORD};",
        timeout=30,
    )
    conn.timeout = 3600
    return conn


def col_map(conn, table):
    """Lower-cased name -> actual column name, so the SELECTs adapt to whatever
    the schema really calls things."""
    df = pd.read_sql(
        "SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = ?",
        conn, params=[table])
    return {c.lower(): c for c in df["COLUMN_NAME"]}


def pick(cmap, wanted):
    """Return the actual column names for the wanted (lower-cased) list that
    actually exist, preserving order, plus the list of missing ones."""
    have = [cmap[w] for w in wanted if w in cmap]
    missing = [w for w in wanted if w not in cmap]
    return have, missing


# ============================================================================
# POSTS LIST  ->  #posts temp table
# ============================================================================

def load_posts():
    if POSTS_FILE.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(POSTS_FILE)
    else:
        df = pd.read_parquet(POSTS_FILE)
    df.columns = [c.lower() for c in df.columns]
    if not {"author", "permlink"} <= set(df.columns):
        if "doc_id" in df.columns:
            sp = df["doc_id"].astype(str).str.split("/", n=1, expand=True)
            df["author"], df["permlink"] = sp[0], sp[1]
        else:
            raise KeyError("POSTS_FILE needs author+permlink or doc_id")
    posts = df[["author", "permlink"]].dropna().drop_duplicates()
    print(f"Posts to enrich: {len(posts):,}")
    return posts


def make_temp(conn, posts):
    cur = conn.cursor()
    cur.execute("IF OBJECT_ID('tempdb..#posts') IS NOT NULL DROP TABLE #posts;")
    cur.execute("CREATE TABLE #posts (author NVARCHAR(20), permlink NVARCHAR(300));")
    cur.fast_executemany = True
    cur.executemany("INSERT INTO #posts (author, permlink) VALUES (?, ?)",
                    list(posts.itertuples(index=False, name=None)))
    cur.execute("CREATE INDEX ix_posts ON #posts(author, permlink);")
    conn.commit()
    n = pd.read_sql("SELECT COUNT(*) n FROM #posts", conn)["n"][0]
    print(f"#posts temp table loaded: {n:,} rows")


# ============================================================================
# QUERIES (each checkpointed)
# ============================================================================

def step_comments(conn):
    out = os.path.join(OUTPUT_DIR, "eng_comments.parquet")
    if os.path.exists(out):
        print("eng_comments.parquet exists, skipping"); return pd.read_parquet(out)
    cmap = col_map(conn, "Comments")
    wanted = ["author", "permlink", "net_votes", "children", "net_rshares",
              "abs_rshares", "total_vote_weight", "total_payout_value",
              "curator_payout_value", "pending_payout_value", "author_rewards",
              "promoted", "author_reputation", "percent_hbd", "created"]
    cols, missing = pick(cmap, wanted)
    if missing:
        print(f"  Comments: not found, skipped -> {missing}")
    sel = ", ".join(f"c.[{c}]" for c in cols)
    q = (f"SELECT {sel} FROM Comments c "
         f"INNER JOIN #posts p ON p.author = c.author AND p.permlink = c.permlink "
         f"WHERE c.depth = 0")
    t0 = time.time()
    df = pd.read_sql(q, conn)
    print(f"  Comments enrichment: {len(df):,} rows in {time.time()-t0:.0f}s")
    df.to_parquet(out, index=False)
    return df


def step_votes(conn):
    out = os.path.join(OUTPUT_DIR, "eng_votes.parquet")
    if os.path.exists(out):
        print("eng_votes.parquet exists, skipping"); return pd.read_parquet(out)
    cmap = col_map(conn, "TxVotes")
    # sign of the vote: prefer percent, else rshares, else weight
    sign_col = next((cmap[c] for c in ("percent", "rshares", "weight") if c in cmap), None)
    rsh = cmap.get("rshares")
    if sign_col is None:
        print("  TxVotes: no percent/rshares/weight column found; skipping"); return None
    rsh_sel = f"SUM(CAST(v.[{rsh}] AS float)) AS sum_rshares," if rsh else ""
    q = (f"SELECT v.author, v.permlink, COUNT(*) AS n_votes_raw, "
         f"SUM(CASE WHEN v.[{sign_col}] > 0 THEN 1 ELSE 0 END) AS upvotes, "
         f"SUM(CASE WHEN v.[{sign_col}] < 0 THEN 1 ELSE 0 END) AS downvotes, "
         f"COUNT(DISTINCT v.voter) AS distinct_voters, {rsh_sel} "
         f"MAX(0) AS _pad "
         f"FROM TxVotes v INNER JOIN #posts p "
         f"ON p.author = v.author AND p.permlink = v.permlink "
         f"GROUP BY v.author, v.permlink")
    t0 = time.time()
    df = pd.read_sql(q, conn).drop(columns=["_pad"], errors="ignore")
    print(f"  Votes breakdown: {len(df):,} posts in {time.time()-t0:.0f}s")
    df.to_parquet(out, index=False)
    return df


def step_accounts(conn):
    out = os.path.join(OUTPUT_DIR, "eng_accounts.parquet")
    if os.path.exists(out):
        print("eng_accounts.parquet exists, skipping"); return pd.read_parquet(out)
    cmap = col_map(conn, "Accounts")
    name_col = cmap.get("name")
    wanted = ["reputation", "vesting_shares", "received_vesting_shares",
              "delegated_vesting_shares", "post_count"]
    cols, missing = pick(cmap, wanted)
    if missing:
        print(f"  Accounts: not found, skipped -> {missing}")
    sel = ", ".join(f"a.[{c}]" for c in cols)
    q = (f"SELECT a.[{name_col}] AS author, {sel} FROM Accounts a "
         f"INNER JOIN (SELECT DISTINCT author FROM #posts) p ON p.author = a.[{name_col}]")
    t0 = time.time()
    df = pd.read_sql(q, conn)
    print(f"  Author standing: {len(df):,} authors in {time.time()-t0:.0f}s")
    df.to_parquet(out, index=False)
    return df


def step_reblogs(conn):
    out = os.path.join(OUTPUT_DIR, "eng_reblogs.parquet")
    if os.path.exists(out):
        print("eng_reblogs.parquet exists, skipping"); return pd.read_parquet(out)
    cmap = col_map(conn, "TxCustoms")
    id_col = next((cmap[c] for c in ("tid", "id") if c in cmap), None)
    ts_col = next((cmap[c] for c in ("timestamp", "expiration", "time") if c in cmap), None)
    json_col = cmap.get("json")
    if not (id_col and json_col):
        print("  TxCustoms: expected id/json columns not found; skipping reblogs")
        print(f"  (TxCustoms columns seen: {sorted(cmap.values())})")
        return None
    ts_clause = (f"AND t.[{ts_col}] >= '{SCOPE_START}' AND t.[{ts_col}] < '{REBLOG_END}'"
                 if ts_col else "")
    # reblog json is either an object {account,author,permlink} or the array form
    # ["reblog",{...}]; COALESCE covers both.
    q = (f"SELECT r.author, r.permlink, COUNT(*) AS reblogs FROM ("
         f"  SELECT COALESCE(JSON_VALUE(t.[{json_col}],'$.author'), "
         f"                  JSON_VALUE(t.[{json_col}],'$[1].author')) AS author, "
         f"         COALESCE(JSON_VALUE(t.[{json_col}],'$.permlink'), "
         f"                  JSON_VALUE(t.[{json_col}],'$[1].permlink')) AS permlink "
         f"  FROM TxCustoms t WHERE t.[{id_col}] = 'reblog' {ts_clause}"
         f") r INNER JOIN #posts p ON p.author = r.author AND p.permlink = r.permlink "
         f"GROUP BY r.author, r.permlink")
    t0 = time.time()
    try:
        df = pd.read_sql(q, conn)
    except Exception as e:
        print(f"  Reblog query failed ({e}). Verify TxCustoms schema / JSON support; "
              f"the rest of the extraction is unaffected.")
        return None
    print(f"  Reblogs: {len(df):,} posts had >=1 reblog in {time.time()-t0:.0f}s")
    df.to_parquet(out, index=False)
    return df


# ============================================================================
# MAIN: run, merge, verify
# ============================================================================

def main():
    posts = load_posts()
    conn = open_conn()
    try:
        # Show the real schema first so the column choices can be confirmed.
        for tbl in ("Comments", "TxVotes", "Accounts", "TxCustoms"):
            try:
                print(f"\n{tbl} columns: {sorted(col_map(conn, tbl).values())}")
            except Exception as e:
                print(f"\n{tbl}: could not read schema ({e})")
        print()

        make_temp(conn, posts)
        comments = step_comments(conn)
        votes    = step_votes(conn)
        accounts = step_accounts(conn)
        reblogs  = step_reblogs(conn) if DO_REBLOGS else None
    finally:
        try: conn.close()
        except Exception: pass

    # ---- merge everything onto the post list ------------------------------
    merged = posts.copy()
    for name, df, on in [("comments", comments, ["author", "permlink"]),
                         ("votes", votes, ["author", "permlink"]),
                         ("reblogs", reblogs, ["author", "permlink"]),
                         ("accounts", accounts, ["author"])]:
        if df is not None and len(df):
            merged = merged.merge(df, on=on, how="left", suffixes=("", f"_{name}"))

    if "reblogs" in merged.columns:
        merged["reblogs"] = merged["reblogs"].fillna(0)

    out = os.path.join(OUTPUT_DIR, "hive_engagement_extended.parquet")
    merged.to_parquet(out, index=False)

    # ---- verification report ----------------------------------------------
    print("\n" + "=" * 64)
    print("VERIFICATION")
    print("=" * 64)
    print(f"Posts in sample:        {len(posts):,}")
    print(f"Rows in merged output:  {len(merged):,}")
    print("\nCoverage (non-null) per new field:")
    new_cols = [c for c in merged.columns if c not in ("author", "permlink")]
    for c in new_cols:
        nn = merged[c].notna().sum()
        print(f"  {c:<26s} {nn:>6,} / {len(merged):,}  ({100*nn/len(merged):.1f}%)")
    print(f"\nWrote: {out}")
    print("\nNext: join hive_engagement_extended.parquet onto the regression "
          "dataset on doc_id (= author/permlink) and we add the new outcomes.")


if __name__ == "__main__":
    main()
