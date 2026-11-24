import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

@st.cache_data
def load_enriched_data():
    df_raw = pd.read_csv("combined_financial_data_idx.csv")
    df_meta = pd.read_csv("idx_market_metadata.csv")

    melted = df_raw.melt(
        id_vars=["symbol", "account", "type"],
        value_vars=["2020", "2021", "2022", "2023"],
        var_name="year",
        value_name="value",
    )
    piv = melted.pivot_table(index=["symbol", "year"], columns="account", values="value").reset_index()

    price_long = df_meta.melt(
        id_vars=["symbol", "company_name", "sector", "industry"],
        value_vars=["price_2020", "price_2021", "price_2022", "price_2023"],
        var_name="year_col",
        value_name="close_price",
    )
    price_long["year"] = price_long["year_col"].str.replace("price_", "")

    df = piv.merge(
        price_long[["symbol", "year", "company_name", "sector", "industry", "close_price"]],
        on=["symbol", "year"],
        how="inner",
    )

    shares = df.get("Ordinary Shares Number", pd.Series(np.nan, index=df.index)).fillna(
        df.get("Basic Average Shares", np.nan)
    )
    df["eps"] = df["Net Income"] / shares.replace(0, np.nan)
    df["bvps"] = df["Stockholders Equity"] / shares.replace(0, np.nan)

    df["per"] = np.where(df["eps"] > 0, df["close_price"] / df["eps"], np.nan)
    df["pbv"] = np.where(df["bvps"] > 0, df["close_price"] / df["bvps"], np.nan)
    df["roe_pct"] = (df["Net Income"] / df["Stockholders Equity"].replace(0, np.nan)) * 100
    df["net_margin_pct"] = (df["Net Income"] / df["Total Revenue"].replace(0, np.nan)) * 100

    df = df[df["sector"] != "Unknown"].copy()
    return df

def render_peer_value_screener():
    st.subheader("💎 Sector Enrichment, Value Screener & Peer Comparison")
    st.caption("Menggabungkan data fundamental IDX (2020–2023) dengan sektor & harga saham historis dari Yahoo Finance.")

    df = load_enriched_data()

   
    f1, f2, f3, f4, f5 = st.columns(5)
    with f1:
        selected_year = st.selectbox("Tahun Fiskal", ["2023", "2022", "2021", "2020"], index=0, key="pvs_year")
    with f2:
        sectors = ["Semua Sektor"] + sorted(df["sector"].dropna().unique().tolist())
        selected_sector = st.selectbox("Sektor", sectors, key="pvs_sector")
    with f3:
        max_pbv = st.slider("PBV Maksimal", 0.5, 10.0, 3.0, 0.5, key="pvs_pbv")
    with f4:
        max_per = st.slider("PER Maksimal", 1.0, 50.0, 20.0, 1.0, key="pvs_per")
    with f5:
        min_roe = st.slider("ROE Minimal (%)", -10.0, 40.0, 10.0, 1.0, key="pvs_roe")

    df_yr = df[df["year"] == selected_year].copy()
    if selected_sector != "Semua Sektor":
        df_yr = df_yr[df_yr["sector"] == selected_sector]

    st.markdown(f"#### 1. Value & Quality Screener ({selected_year})")
    screened = df_yr[
        (df_yr["pbv"] <= max_pbv)
        & (df_yr["per"] <= max_per)
        & (df_yr["roe_pct"] >= min_roe)
    ].sort_values("roe_pct", ascending=False)

    c1, c2, c3 = st.columns(3)
    c1.metric("Emiten Lolos Filter", len(screened))
    c2.metric("Median PBV (Terfilter)", f"{screened['pbv'].median():.2f}x" if not screened.empty else "-")
    c3.metric("Median PER (Terfilter)", f"{screened['per'].median():.2f}x" if not screened.empty else "-")

    cols_show = ["symbol", "company_name", "sector", "industry", "close_price", "per", "pbv", "roe_pct", "net_margin_pct"]
    st.dataframe(
        screened[cols_show].style.format({
            "close_price": "{:,.0f}",
            "per": "{:.2f}x",
            "pbv": "{:.2f}x",
            "roe_pct": "{:.2f}%",
            "net_margin_pct": "{:.2f}%",
        }),
        use_container_width=True,
    )

    st.divider()
    st.markdown("#### 2. Head-to-Head Peer Comparison")
    available_syms = sorted(df_yr["symbol"].unique().tolist())
    default_peers = [s for s in ["AALI", "LSIP", "BBCA", "BBRI"] if s in available_syms][:3]
    peers = st.multiselect("Pilih 2–5 Emiten untuk Dibandingkan", available_syms, default=default_peers, key="pvs_peers")

    if peers:
        df_peers = df[(df["symbol"].isin(peers)) & (df["year"] == selected_year)]
        st.dataframe(df_peers[cols_show], use_container_width=True)

        metric_choice = st.selectbox("Pilih Metrik Grafik Tren (2020–2023)", ["roe_pct", "net_margin_pct", "pbv", "per"], key="pvs_metric")
        df_trend = df[df["symbol"].isin(peers)]
        chart = (
            alt.Chart(df_trend)
            .mark_line(point=True)
            .encode(
                x=alt.X("year:O", title="Tahun"),
                y=alt.Y(f"{metric_choice}:Q", title=metric_choice.upper()),
                color="symbol:N",
                tooltip=["symbol", "company_name", "year", metric_choice],
            )
            .properties(height=350)
        )
        st.altair_chart(chart, use_container_width=True)