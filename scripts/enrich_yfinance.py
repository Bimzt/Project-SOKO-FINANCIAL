import time
import pandas as pd
import yfinance as yf

df_raw = pd.read_csv("combined_financial_data_idx.csv", usecols=["symbol"])
symbols = sorted(df_raw["symbol"].unique())

#symbols = symbols[:15]
records = []
total = len(symbols)

for i, sym in enumerate(symbols, start=1):
    ticker_code = f"{sym}.JK"
    row = {
        "symbol": sym,
        "company_name": None,
        "sector": "Unknown",
        "industry": "Unknown",
        "price_2020": None,
        "price_2021": None,
        "price_2022": None,
        "price_2023": None,
    }
    try:
        tk = yf.Ticker(ticker_code)
        info = tk.info or {}
        row["company_name"] = info.get("longName") or info.get("shortName")
        row["sector"] = info.get("sector") or "Unknown"
        row["industry"] = info.get("industry") or "Unknown"

        hist = tk.history(start="2020-01-01", end="2024-01-05")
        if not hist.empty:
            hist.index = hist.index.tz_localize(None)
            for yr in ["2020", "2021", "2022", "2023"]:
                yr_data = hist.loc[f"{yr}-01-01":f"{yr}-12-31", "Close"]
                if not yr_data.empty:
                    row[f"price_{yr}"] = round(float(yr_data.iloc[-1]), 2)
    except Exception as e:
        print(f"[{i}/{total}] {sym} gagal: {e}")

    records.append(row)
    if i % 25 == 0 or i == total:
        print(f"Progress: {i}/{total} emiten selesai")
    time.sleep(0.1)  

df_meta = pd.DataFrame(records)
df_meta.to_csv("idx_market_metadata.csv", index=False)
print("Selesai! Tersimpan ke idx_market_metadata.csv")