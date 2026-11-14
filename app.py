import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="IDX Corporate Financial Intelligence Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS STYLING ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .metric-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        margin-bottom: 15px;
    }
    .metric-title {
        font-size: 0.82rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
    }
    .metric-val {
        font-size: 1.65rem;
        font-weight: 700;
        color: #0F172A;
    }
    .metric-sub {
        font-size: 0.78rem;
        color: #64748B;
        margin-top: 4px;
    }
    .badge-safe {
        background-color: #DEF7EC;
        color: #03543F;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.75rem;
    }
    .badge-grey {
        background-color: #FEF08A;
        color: #713F12;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.75rem;
    }
    .badge-distress {
        background-color: #FDE8E8;
        color: #9B1C1C;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.75rem;
    }
</style>
""", unsafe_allow_html=True)

# --- DATA INGESTION & PIPELINE ENGINE ---
@st.cache_data
def load_and_process_idx_data():
    df_raw = pd.read_csv('combined_financial_data_idx.csv')
    melted = df_raw.melt(
        id_vars=['symbol', 'account', 'type'],
        value_vars=['2020', '2021', '2022', '2023'],
        var_name='year',
        value_name='value'
    )
    piv = melted.pivot_table(index=['symbol', 'year'], columns='account', values='value').reset_index()
    
    # 1. Structural Segmentation: Financial Institutions vs Non-Financial Corporates
    piv['is_financial'] = piv['Working Capital'].isna()
    piv['sector_type'] = np.where(piv['is_financial'], 'Financials & Banking', 'Non-Financial Corporates')
    
    # 2. Clean EBIT with hierarchical fallback
    piv['ebit_clean'] = piv['EBIT'].fillna(piv.get('Operating Income', np.nan)).fillna(
        piv['Pretax Income'] + piv.get('Interest Expense', 0).fillna(0)
    )
    
    # 3. DuPont 3-Way Components
    piv['net_margin_pct'] = (piv['Net Income'] / piv['Total Revenue'].replace(0, np.nan)) * 100
    piv['asset_turnover'] = piv['Total Revenue'] / piv['Total Assets'].replace(0, np.nan)
    piv['equity_multiplier'] = piv['Total Assets'] / piv['Stockholders Equity'].replace(0, np.nan)
    piv['roe_pct'] = (piv['Net Income'] / piv['Stockholders Equity'].replace(0, np.nan)) * 100
    piv['roa_pct'] = (piv['Net Income'] / piv['Total Assets'].replace(0, np.nan)) * 100
    
    # 4. Solvency & Debt Ratios
    piv['debt_to_equity'] = piv['Total Liabilities Net Minority Interest'] / piv['Stockholders Equity'].replace(0, np.nan)
    piv['debt_to_assets'] = piv['Total Liabilities Net Minority Interest'] / piv['Total Assets'].replace(0, np.nan)
    
    # 5. Altman Z''-Score Model for Emerging Markets (Non-Financials)
    x1 = piv['Working Capital'] / piv['Total Assets']
    x2 = piv['Retained Earnings'] / piv['Total Assets']
    x3 = piv['ebit_clean'] / piv['Total Assets']
    x4 = piv['Stockholders Equity'] / piv['Total Liabilities Net Minority Interest'].replace(0, np.nan)
    piv['altman_z'] = np.where(~piv['is_financial'], 6.56*x1 + 3.26*x2 + 6.72*x3 + 1.05*x4, np.nan)
    
    def classify_z_score(z):
        if pd.isna(z): return 'Financials / NA'
        if z > 2.6: return 'Safe Zone'
        if z >= 1.1: return 'Grey Zone'
        return 'Distress Zone'
        
    piv['solvency_zone'] = piv['altman_z'].apply(classify_z_score)
    
    # 6. Forensic Earnings Quality (Sloan Accrual Ratio)
    cfo = piv.get('Cash Flowsfromusedin Operating Activities Direct', pd.Series(np.nan, index=piv.index))
    piv['sloan_accrual'] = (piv['Net Income'] - cfo) / piv['Total Assets'].replace(0, np.nan)
    piv['cfo_to_ni'] = cfo / piv['Net Income'].replace(0, np.nan)
    
    # Scaled metrics in Trillions IDR for clear presentation
    piv['Revenue_Tril'] = piv['Total Revenue'] / 1e12
    piv['NetIncome_Tril'] = piv['Net Income'] / 1e12
    piv['Assets_Tril'] = piv['Total Assets'] / 1e12
    piv['CFO_Tril'] = cfo / 1e12
    piv['FCF_Tril'] = piv.get('Free Cash Flow', pd.Series(np.nan, index=piv.index)) / 1e12
    
    return piv

df = load_and_process_idx_data()

# --- SIDEBAR INTERACTIVE FILTERS ---
st.sidebar.image("https://img.icons8.com/fluency/96/combo-chart.png", width=56)
st.sidebar.title("Kontrol Analitik")
st.sidebar.caption("Filter data keuangan 604 emiten BEI (2020-2023)")

# Year Selector
selected_year = st.sidebar.selectbox(
    "Pilih Tahun Fiskal:",
    options=["Semua Tahun (2020-2023)", "2023", "2022", "2021", "2020"],
    index=1
)

# Sector Classification Filter
selected_sector = st.sidebar.radio(
    "Segmentasi Sektor:",
    options=["Semua Emiten", "Non-Financial Corporates (538)", "Financials & Banking (66)"],
    index=0
)

# Solvency Filter
selected_zone = st.sidebar.multiselect(
    "Zona Risiko Solvabilitas (Altman Z):",
    options=["Safe Zone", "Grey Zone", "Distress Zone", "Financials / NA"],
    default=["Safe Zone", "Grey Zone", "Distress Zone", "Financials / NA"]
)

# Filter Data according to controls
filtered_df = df.copy()

if selected_year != "Semua Tahun (2020-2023)":
    filtered_df = filtered_df[filtered_df['year'] == selected_year]

if selected_sector == "Non-Financial Corporates (538)":
    filtered_df = filtered_df[~filtered_df['is_financial']]
elif selected_sector == "Financials & Banking (66)":
    filtered_df = filtered_df[filtered_df['is_financial']]

filtered_df = filtered_df[filtered_df['solvency_zone'].isin(selected_zone)]

# --- MAIN DASHBOARD HEADER ---
st.title("📈 IDX Corporate Financial Intelligence Dashboard")
st.markdown("""
Analisis kesehatan finansial komprehensif atas **604 Emiten Terdaftar di Bursa Efek Indonesia (BEI)**.
Mengintegrasikan diagnostik **DuPont 3-Way**, **Altman Z''-Score (Emerging Markets)**, dan **Forensic Earnings Quality (Sloan Accrual)** guna mendukung mitigasi risiko dan alokasi modal strategis.
""")

# --- EXECUTIVE KPI METRIC CARDS ---
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

total_rev = filtered_df['Total Revenue'].sum() / 1e12
total_ni = filtered_df['Net Income'].sum() / 1e12
med_roe = filtered_df['roe_pct'].median()
safe_pct = (filtered_df['solvency_zone'] == 'Safe Zone').mean() * 100 if len(filtered_df) > 0 else 0
med_z = filtered_df[~filtered_df['is_financial']]['altman_z'].median()

with kpi1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Agregat Revenue</div>
        <div class="metric-val">Rp {total_rev:,.1f} T</div>
        <div class="metric-sub">{len(filtered_df):,} Observasi Terpilih</div>
    </div>
    """, unsafe_allow_html=True)

with kpi2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Agregat Laba Bersih</div>
        <div class="metric-val" style="color: {'#059669' if total_ni >= 0 else '#DC2626'}">Rp {total_ni:,.1f} T</div>
        <div class="metric-sub">Margin Bersih: {(total_ni/total_rev*100) if total_rev else 0:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

with kpi3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Median ROE</div>
        <div class="metric-val">{med_roe:.2f}%</div>
        <div class="metric-sub">Efisiensi Ekuitas Pasar</div>
    </div>
    """, unsafe_allow_html=True)

with kpi4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Porsi Safe Zone</div>
        <div class="metric-val" style="color: #059669;">{safe_pct:.1f}%</div>
        <div class="metric-sub">Emiten Solven Tinggi</div>
    </div>
    """, unsafe_allow_html=True)

with kpi5:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Median Altman Z''</div>
        <div class="metric-val">{med_z:.2f}</div>
        <div class="metric-sub">Ambang Aman > 2.60</div>
    </div>
    """, unsafe_allow_html=True)

# --- MULTI-TAB EXECUTIVE WORKBENCH ---
tab_macro, tab_dupont, tab_altman, tab_accrual, tab_drilldown, tab_table = st.tabs([
    "🌐 Trajektori Makro (2020-2023)",
    "🎯 Matriks Profitabilitas DuPont",
    "🛡️ Solvabilitas & Altman Z''-Score",
    "🔍 Forensic Earnings Quality (Sloan)",
    "🏢 Single-Company Drill-Down",
    "📋 Master Data Explorer"
])

# TAB 1: MACRO DYNAMICS & POST-PANDEMIC CYCLE
with tab_macro:
    st.subheader("Siklus Pemulihan Ekonomi BEI: Dari Krisis Pandemi Menuju Normalisasi")
    
    col_m1, col_m2 = st.columns([3, 2])
    
    macro_agg = df.groupby('year').agg(
        Revenue=('Revenue_Tril', 'sum'),
        Net_Income=('NetIncome_Tril', 'sum'),
        Operating_Cash_Flow=('CFO_Tril', 'sum')
    ).reset_index()
    
    macro_melted = macro_agg.melt(id_vars='year', var_name='Metrik', value_name='Nilai_Tril')
    
    chart_macro = alt.Chart(macro_melted).mark_line(point=True, strokeWidth=3).encode(
        x=alt.X('year:O', title='Tahun Fiskal'),
        y=alt.Y('Nilai_Tril:Q', title='Nilai Agregat (Rp Triliun)'),
        color=alt.Color('Metrik:N', scale=alt.Scale(
            domain=['Revenue', 'Operating_Cash_Flow', 'Net_Income'],
            range=['#2563EB', '#0D9488', '#059669']
        )),
        tooltip=['year:O', 'Metrik:N', alt.Tooltip('Nilai_Tril:Q', format=',.1f')]
    ).properties(height=380).interactive()
    
    with col_m1:
        st.altair_chart(chart_macro, use_container_width=True)
        
    with col_m2:
        st.markdown("""
        **💡 Strategic Macro Insights:**
        1. **2020 (The COVID-19 Trough):** Penjualan dan arus kas operasi mengalami tekanan likuiditas akut akibat pembatasan mobilitas, dengan rasio Altman Z median anjlok ke **2.65** (batas krisis).
        2. **2021-2022 (Commodity & Reopening Boom):** Terjadi lompatan agresif laba bersih berkat lonjakan harga komoditas (batu bara, CPO, nikel) dan pemulihan konsumsi domestik. Total laba bersih melesat lebih dari 200%.
        3. **2023 (Normalization & Tight Liquidity):** Normalisasi harga komoditas global menyebabkan laju pertumbuhan laba melambat, namun neraca emiten BEI terbukti jauh lebih kokoh dengan median Z-score mencapai **3.71**.
        """)
        st.dataframe(macro_agg.rename(columns={
            'year': 'Tahun',
            'Revenue': 'Revenue (Rp T)',
            'Net_Income': 'Laba Bersih (Rp T)',
            'Operating_Cash_Flow': 'Arus Kas Operasi (Rp T)'
        }).set_index('Tahun').style.format("{:,.1f}"), use_container_width=True)

# TAB 2: DUPONT PROFITABILITY MATRIX
with tab_dupont:
    st.subheader("Dekomposisi DuPont 3-Way: Net Profit Margin vs Asset Turnover")
    st.caption("Membedah sumber keunggulan kompetitif emiten: Efisiensi Margin Operasional vs Kecepatan Perputaran Aset.")
    
    # Filter valid for DuPont scatter
    dup_df = filtered_df[
        (filtered_df['net_margin_pct'].between(-40, 60)) &
        (filtered_df['asset_turnover'].between(0.05, 3.0))
    ].copy()
    
    dup_chart = alt.Chart(dup_df).mark_circle(opacity=0.75).encode(
        x=alt.X('net_margin_pct:Q', title='Net Profit Margin (%)', scale=alt.Scale(domain=[-30, 50])),
        y=alt.Y('asset_turnover:Q', title='Asset Turnover (Perputaran Aset x)', scale=alt.Scale(domain=[0, 2.5])),
        size=alt.Size('Assets_Tril:Q', title='Total Aset (Rp T)', scale=alt.Scale(range=[30, 900])),
        color=alt.Color('solvency_zone:N', scale=alt.Scale(
            domain=['Safe Zone', 'Grey Zone', 'Distress Zone', 'Financials / NA'],
            range=['#059669', '#D97706', '#DC2626', '#6366F1']
        ), title='Zona Solvabilitas'),
        tooltip=['symbol:N', 'year:O', 'sector_type:N',
                 alt.Tooltip('net_margin_pct:Q', format='.2f'),
                 alt.Tooltip('asset_turnover:Q', format='.2f'),
                 alt.Tooltip('roe_pct:Q', format='.2f'),
                 alt.Tooltip('Assets_Tril:Q', format=',.1f')]
    ).properties(height=450).interactive()
    
    st.altair_chart(dup_chart, use_container_width=True)
    
    c_q1, c_q2 = st.columns(2)
    with c_q1:
        st.info("""
        **Quadrant Strategic Framework:**
        * **Top-Right (Quality Compounders):** Margin tebal & turnover tinggi. Model bisnis defensif dengan moat kuat (cth. consumer leaders & tambang efisien).
        * **Top-Left (Volume Drivers):** Margin tipis namun perputaran aset sangat cepat (cth. ritel AMRT, logistik/distribusi).
        """)
    with c_q2:
        st.warning("""
        * **Bottom-Right (Margin Specialists):** Margin tinggi dengan siklus aset lambat / padat modal (cth. telekomunikasi TLKM, infrastruktur).
        * **Bottom-Left (Lagging / Restructuring):** Margin tertekan & aset menganggur. Menghadapi risiko penurunan ekuitas jangka panjang.
        """)

# TAB 3: ALTMAN Z''-SCORE MONITOR
with tab_altman:
    st.subheader("Diagnostik Kesehatan Solvabilitas & Risiko Kebangkrutan (Altman Z''-Score)")
    st.caption("Khusus dihitung untuk emiten non-keuangan menggunakan model Altman Z-Score Pasar Berkembang (1993/2000).")
    
    col_z1, col_z2 = st.columns([3, 2])
    
    non_fin = df[~df['is_financial']].copy()
    zone_by_year = pd.crosstab(non_fin['year'], non_fin['solvency_zone'], normalize='index').reset_index()
    zone_melt = zone_by_year.melt(id_vars='year', var_name='Zona', value_name='Proporsi')
    zone_melt['Persentase'] = zone_melt['Proporsi'] * 100
    
    chart_zone = alt.Chart(zone_melt).mark_bar().encode(
        x=alt.X('year:O', title='Tahun Fiskal'),
        y=alt.Y('Persentase:Q', title='Distribusi Emiten (%)', stack='normalize'),
        color=alt.Color('Zona:N', scale=alt.Scale(
            domain=['Safe Zone', 'Grey Zone', 'Distress Zone'],
            range=['#059669', '#D97706', '#DC2626']
        )),
        tooltip=['year:O', 'Zona:N', alt.Tooltip('Persentase:Q', format='.1f')]
    ).properties(height=380)
    
    with col_z1:
        st.altair_chart(chart_zone, use_container_width=True)
        
    with col_z2:
        st.markdown("""
        **Analisis Transisi Kesehatan Neraca:**
        * Pada tahun 2020, hampir **30%** emiten terpuruk di **Distress Zone** ($Z < 1.10$).
        * Terjadi perbaikan struktural masif hingga tahun 2023, di mana proporsi **Safe Zone** melesat menjadi **63.3%**.
        * Pengurangan utang (deleveraging) dan penumpukan laba ditahan pasca pemulihan komoditas berhasil memperkuat bantalan modal emiten BEI.
        """)
        
    st.markdown("#### 🚨 Emiten dengan Risiko Solvabilitas Tertinggi (Tahun Terpilih)")
    distress_watchlist = filtered_df[
        (~filtered_df['is_financial']) & (filtered_df['solvency_zone'] == 'Distress Zone')
    ][['symbol', 'year', 'altman_z', 'debt_to_equity', 'Revenue_Tril', 'NetIncome_Tril', 'Assets_Tril']].sort_values(by='altman_z').head(10)
    
    st.dataframe(
        distress_watchlist.rename(columns={
            'symbol': 'Ticker',
            'year': 'Tahun',
            'altman_z': 'Altman Z-Score',
            'debt_to_equity': 'Debt/Equity (x)',
            'Revenue_Tril': 'Revenue (Rp T)',
            'NetIncome_Tril': 'Net Income (Rp T)',
            'Assets_Tril': 'Assets (Rp T)'
        }).style.format({
            'Altman Z-Score': '{:.2f}',
            'Debt/Equity (x)': '{:.2f}',
            'Revenue (Rp T)': '{:,.2f}',
            'Net Income (Rp T)': '{:,.2f}',
            'Assets (Rp T)': '{:,.2f}'
        }), use_container_width=True
    )

# TAB 4: FORENSIC EARNINGS QUALITY (SLOAN)
with tab_accrual:
    st.subheader("Forensic Accounting: Laba Akuntansi vs Realisasi Arus Kas (Sloan Accrual)")
    st.caption("Mendeteksi anomali laba di atas kertas (paper profits) yang tidak didukung oleh arus kas operasional nyata.")
    
    acc_df = filtered_df[
        (filtered_df['NetIncome_Tril'].between(-5, 20)) &
        (filtered_df['CFO_Tril'].between(-5, 20))
    ].copy()
    
    chart_acc = alt.Chart(acc_df).mark_circle(size=80, opacity=0.7).encode(
        x=alt.X('NetIncome_Tril:Q', title='Laba Bersih Akuntansi (Rp Triliun)'),
        y=alt.Y('CFO_Tril:Q', title='Arus Kas Operasi Nyata (Rp Triliun)'),
        color=alt.condition(
            alt.datum.CFO_Tril >= alt.datum.NetIncome_Tril,
            alt.value('#059669'),
            alt.value('#DC2626')
        ),
        tooltip=['symbol:N', 'year:O',
                 alt.Tooltip('NetIncome_Tril:Q', format=',.2f'),
                 alt.Tooltip('CFO_Tril:Q', format=',.2f'),
                 alt.Tooltip('sloan_accrual:Q', format='.3f')]
    ).properties(height=420).interactive()
    
    line_data = pd.DataFrame({'x': [-5, 20], 'y': [-5, 20]})
    parity_line = alt.Chart(line_data).mark_line(color='#94A3B8', strokeDash=[4, 4]).encode(
        x='x:Q', y='y:Q'
    )
    
    st.altair_chart(chart_acc + parity_line, use_container_width=True)
    
    col_a1, col_a2 = st.columns(2)
    with col_a1:
        st.success("""
        **🟢 Zona Hijau (Arus Kas > Laba Bersih):**
        Emiten memiliki *high-grade earnings quality*. Laba yang dilaporkan bersifat konservatif dan didukung konversi kas tunai yang prima.
        """)
    with col_a2:
        st.error("""
        **🔴 Zona Merah (Laba Bersih > Arus Kas):**
        *Accrual Warning*. Emiten mengakui pendapatan atau laba akuntansi tinggi namun belum mampu menarik uang kas dari pelanggan, mengindikasikan potensi risiko piutang macet atau restrukturisasi.
        """)

# TAB 5: SINGLE-COMPANY DRILL-DOWN SCORECARD
with tab_drilldown:
    st.subheader("🏢 Drill-Down Diagnostik Individual Emiten")
    
    all_symbols = sorted(df['symbol'].unique())
    selected_ticker = st.selectbox(
        "Cari atau Pilih Kode Emiten (Ticker):",
        options=all_symbols,
        index=all_symbols.index("ASII") if "ASII" in all_symbols else 0
    )
    
    company_data = df[df['symbol'] == selected_ticker].sort_values(by='year')
    latest_rec = company_data.iloc[-1]
    
    col_d1, col_d2, col_d3, col_d4 = st.columns(4)
    with col_d1:
        st.metric("Kategori Sektor", latest_rec['sector_type'])
    with col_d2:
        st.metric("Total Aset Terakhir", f"Rp {latest_rec['Assets_Tril']:,.2f} T")
    with col_d3:
        st.metric("ROE Terakhir", f"{latest_rec['roe_pct']:.2f}%")
    with col_d4:
        zone_color = {
            'Safe Zone': 'badge-safe',
            'Grey Zone': 'badge-grey',
            'Distress Zone': 'badge-distress'
        }.get(latest_rec['solvency_zone'], 'badge-safe')
        st.markdown(f"**Status Solvabilitas:**<br><span class='{zone_color}'>{latest_rec['solvency_zone']}</span>", unsafe_allow_html=True)
        
    st.markdown("#### Perkembangan Finansial 4-Tahun (2020-2023)")
    st.dataframe(company_data[[
        'year', 'Revenue_Tril', 'NetIncome_Tril', 'CFO_Tril', 'FCF_Tril',
        'net_margin_pct', 'asset_turnover', 'equity_multiplier', 'roe_pct', 'altman_z'
    ]].rename(columns={
        'year': 'Tahun',
        'Revenue_Tril': 'Revenue (Rp T)',
        'NetIncome_Tril': 'Net Profit (Rp T)',
        'CFO_Tril': 'Arus Kas Operasi (Rp T)',
        'FCF_Tril': 'Free Cash Flow (Rp T)',
        'net_margin_pct': 'Net Margin (%)',
        'asset_turnover': 'Asset Turnover (x)',
        'equity_multiplier': 'Leverage (x)',
        'roe_pct': 'ROE (%)',
        'altman_z': 'Altman Z'
    }).set_index('Tahun').style.format({
        'Revenue (Rp T)': '{:,.2f}',
        'Net Profit (Rp T)': '{:,.2f}',
        'Arus Kas Operasi (Rp T)': '{:,.2f}',
        'Free Cash Flow (Rp T)': '{:,.2f}',
        'Net Margin (%)': '{:.2f}%',
        'Asset Turnover (x)': '{:.2f}',
        'Leverage (x)': '{:.2f}',
        'ROE (%)': '{:.2f}%',
        'Altman Z': '{:.2f}'
    }), use_container_width=True)

# TAB 6: MASTER DATA EXPLORER & CSV EXPORT
with tab_table:
    st.subheader("📋 Master Panel Data Finansial 604 Emiten")
    st.caption("Akses penuh ke dataset panel yang telah diperkaya dengan rasio analitik tingkat lanjut.")
    
    export_cols = [
        'symbol', 'year', 'sector_type', 'Revenue_Tril', 'NetIncome_Tril',
        'Assets_Tril', 'net_margin_pct', 'asset_turnover', 'equity_multiplier',
        'roe_pct', 'altman_z', 'solvency_zone', 'sloan_accrual'
    ]
    
    table_view = filtered_df[export_cols].rename(columns={
        'symbol': 'Ticker',
        'year': 'Tahun',
        'sector_type': 'Sektor',
        'Revenue_Tril': 'Revenue (Rp T)',
        'NetIncome_Tril': 'Laba Bersih (Rp T)',
        'Assets_Tril': 'Total Aset (Rp T)',
        'net_margin_pct': 'Net Margin (%)',
        'asset_turnover': 'Asset Turnover (x)',
        'equity_multiplier': 'Equity Multiplier (x)',
        'roe_pct': 'ROE (%)',
        'altman_z': 'Altman Z-Score',
        'solvency_zone': 'Zona Risiko',
        'sloan_accrual': 'Sloan Accrual'
    })
    
    st.dataframe(table_view.style.format({
        'Revenue (Rp T)': '{:,.2f}',
        'Laba Bersih (Rp T)': '{:,.2f}',
        'Total Aset (Rp T)': '{:,.2f}',
        'Net Margin (%)': '{:.2f}%',
        'Asset Turnover (x)': '{:.2f}',
        'Equity Multiplier (x)': '{:.2f}',
        'ROE (%)': '{:.2f}%',
        'Altman Z-Score': '{:.2f}',
        'Sloan Accrual': '{:.3f}'
    }), use_container_width=True)
    
    csv = table_view.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Data Finansial Terfilter (CSV)",
        data=csv,
        file_name='idx_financial_intelligence_panel.csv',
        mime='text/csv'
    )
