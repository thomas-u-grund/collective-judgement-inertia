import duckdb

con = duckdb.connect()
con.execute("INSTALL httpfs; LOAD httpfs;")
con.execute("SET threads=4;")
con.execute("SET http_retries=10;")
con.execute("SET http_retry_wait_ms=3000;")
con.execute("SET http_retry_backoff=2;")

con.execute("""
    CREATE TABLE resolved AS
    SELECT id AS market_id, winner_idx, end_date
    FROM read_parquet('resolved_binary_markets.parquet')
""")
con.execute("""
    CREATE TABLE active_wallets AS
    SELECT wallet FROM read_parquet('wallet_counts.parquet') WHERE n_trades >= 50
""")
print("active wallets:", con.execute("SELECT count(*) FROM active_wallets").fetchone())

url = "https://huggingface.co/datasets/SII-WANGZJ/Polymarket_data/resolve/main/trades.parquet"
q = f"""
COPY (
    SELECT
        t.taker AS wallet,
        t.market_id,
        t.timestamp,
        r.end_date,
        t.price,
        t.nonusdc_side,
        r.winner_idx,
        CASE WHEN (t.nonusdc_side = 'token1' AND r.winner_idx = 0)
               OR (t.nonusdc_side = 'token2' AND r.winner_idx = 1)
             THEN 1 ELSE 0 END AS correct
    FROM read_parquet('{url}') t
    JOIN resolved r ON t.market_id = r.market_id
    JOIN active_wallets w ON t.taker = w.wallet
    WHERE t.taker_direction = 'BUY' AND t.maker != t.taker
) TO 'poly_panel_raw.parquet' (FORMAT PARQUET)
"""
con.execute(q)
print("done")
print(con.execute("SELECT count(*) FROM read_parquet('poly_panel_raw.parquet')").fetchone())
