import streamlit as st
import numpy as np
from scipy.stats import norm

# ==========================================
# ファイル名: finance_calc.py
# 実行コマンド: streamlit run finance_calc.py
# ==========================================

# --- 1. 設定 & 言語データ ---
st.set_page_config(
    page_title="Quant Calculator Pro",
    page_icon="📈",
    layout="centered"
)

# A8.netの広告コード（ここに入力されたコードが入っています）
A8_AD_HTML = """
<div style="text-align: center; margin-top: 20px;">
<a href="https://px.a8.net/svt/ejp?a8mat=4AV8S7+B2WG36+ONS+TT69D" rel="nofollow">
<img border="0" width="320" height="50" alt="" src="https://www22.a8.net/svt/bgt?aid=260118583670&wid=001&eno=01&mid=s00000003196005007000&mc=1"></a>
<img border="0" width="1" height="1" src="https://www16.a8.net/0.gif?a8mat=4AV8S7+B2WG36+ONS+TT69D" alt="">
</div>
"""

# 翻訳辞書
TRANS = {
    "JP": {
        "title": "高度金融計算機 (Quant Calculator)",
        "sidebar_title": "計算モデル選択",
        "calc_bs": "ブラック・ショールズ (オプション)",
        "calc_kelly": "ケリー基準 (資金管理)",
        "calc_var": "VaR (バリュー・アット・リスク)",
        "calc_port": "ポートフォリオ分散 (行列計算)",
        "calc_garch": "GARCHモデル (ボラティリティ予測)",
        "calc_hurst": "ハースト指数 (トレンド判定)",
        "calc_btn": "計算する",
        "desc_bs": "コールオプションの理論価格とグリークスを計算します。",
        "desc_kelly": "破産を避けつつ資産を最大化する最適な投資比率を計算します。",
        "desc_var": "特定の確率で発生しうる最大損失額を計算します。",
        "desc_port": "相関係数を考慮したポートフォリオ全体のリスク（分散・標準偏差）を行列演算で算出します。",
        "desc_garch": "GARCH(1,1)モデルを用いて、翌日のボラティリティを予測します。",
        "desc_hurst": "時系列データからハースト指数を算出し、相場の性質（トレンド/平均回帰）を判定します。",
        # Inputs & Results
        "bs_s": "現在株価 (S)", "bs_k": "行使価格 (K)", "bs_t": "満期 (年)", "bs_r": "金利 (%)", "bs_v": "ボラティリティ (%)", "bs_call": "コール価格",
        "kelly_p": "勝率 (%)", "kelly_rr": "リスクリワード", "kelly_res": "推奨レバレッジ",
        "var_amt": "投資元本", "var_vol": "ボラティリティ (%)", "var_day": "期間 (日)", "var_res": "最大損失 (VaR)",
        "port_w": "投資比率 (カンマ区切り)", "port_v": "ボラティリティ (%, カンマ区切り)", "port_corr": "相関係数", "port_res_vol": "ポートフォリオ標準偏差",
        "garch_omega": "オメガ (ω)", "garch_alpha": "アルファ (α)", "garch_beta": "ベータ (β)", "garch_res": "予測ボラティリティ",
        "hurst_data": "時系列データ", "hurst_gen_btn": "データ生成", "hurst_res": "ハースト指数", "hurst_interp": "判定",
        "disclaimer": "免責事項: 投資は自己責任で行ってください。"
    },
    "EN": {
        "title": "Quant Calculator Pro",
        "sidebar_title": "Select Model",
        "calc_bs": "Black-Scholes (Option Pricing)",
        "calc_kelly": "Kelly Criterion (Money Mgmt)",
        "calc_var": "Value at Risk (VaR)",
        "calc_port": "Portfolio Variance (Matrix)",
        "calc_garch": "GARCH Model (Volatility)",
        "calc_hurst": "Hurst Exponent (Trend)",
        "calc_btn": "Calculate",
        "desc_bs": "Calculate theoretical call option price using Black-Scholes.",
        "desc_kelly": "Calculate optimal bet size to maximize wealth.",
        "desc_var": "Estimate maximum potential loss (VaR).",
        "desc_port": "Calculate portfolio risk using matrix algebra.",
        "desc_garch": "Predict next-day volatility using GARCH(1,1).",
        "desc_hurst": "Estimate Hurst Exponent to determine trend.",
        # Inputs & Results
        "bs_s": "Spot Price (S)", "bs_k": "Strike Price (K)", "bs_t": "Maturity (Y)", "bs_r": "Rate (%)", "bs_v": "Volatility (%)", "bs_call": "Call Price",
        "kelly_p": "Win Rate (%)", "kelly_rr": "Risk/Reward", "kelly_res": "Optimal Leverage",
        "var_amt": "Portfolio Value", "var_vol": "Volatility (%)", "var_day": "Days", "var_res": "Max Loss (VaR)",
        "port_w": "Weights (comma separated)", "port_v": "Vols (%, comma separated)", "port_corr": "Correlation", "port_res_vol": "Portfolio Risk",
        "garch_omega": "Omega (ω)", "garch_alpha": "Alpha (α)", "garch_beta": "Beta (β)", "garch_res": "Forecasted Vol",
        "hurst_data": "Time Series Data", "hurst_gen_btn": "Generate Data", "hurst_res": "Hurst Exponent", "hurst_interp": "Interpretation",
        "disclaimer": "Disclaimer: Trading involves risk."
    }
}

# --- 2. 計算ロジック ---

def black_scholes(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0: return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def kelly_criterion(win_rate, risk_reward):
    if risk_reward == 0: return 0.0
    return ((win_rate * (risk_reward + 1) - 1) / risk_reward) * 100

def calculate_var(amount, volatility_annual, confidence, days):
    vol_period = (volatility_annual) * np.sqrt(days / 252)
    return amount * vol_period * norm.ppf(confidence)

def portfolio_risk_matrix(weights, vols, corr):
    w = np.array(weights)
    v = np.array(vols)
    n = len(w)
    corr_matrix = np.full((n, n), corr)
    np.fill_diagonal(corr_matrix, 1.0)
    D = np.diag(v)
    sigma_matrix = D @ corr_matrix @ D
    port_var = w.T @ sigma_matrix @ w
    return port_var, np.sqrt(port_var)

def garch_forecast(omega, alpha, beta, prev_ret, prev_vol):
    p_ret = prev_ret / 100
    p_vol = prev_vol / 100
    next_var = omega + alpha * (p_ret ** 2) + beta * (p_vol ** 2)
    return np.sqrt(next_var) * 100

def calculate_hurst(ts):
    ts = np.array(ts)
    if len(ts) < 10: return 0.5
    lags = range(2, min(len(ts)//2, 20))
    tau = []
    lagvec = []
    for lag in lags:
        pp = np.subtract(ts[lag:], ts[:-lag])
        tau.append(np.sqrt(np.std(pp)))
        lagvec.append(lag)
    m = np.polyfit(np.log(lagvec), np.log(tau), 1)
    return m[0] * 2

# --- 3. UI構築 ---

def main():
    lang_opt = st.sidebar.radio("Language / 言語", ["日本語", "English"])
    lang = "JP" if lang_opt == "日本語" else "EN"
    txt = TRANS[lang]

    st.title(txt["title"])
    
    # メニュー選択
    menu = [
        txt["calc_bs"], txt["calc_kelly"], txt["calc_var"],
        txt["calc_port"], txt["calc_garch"], txt["calc_hurst"]
    ]
    choice = st.sidebar.selectbox(txt["sidebar_title"], menu)
    
    # === A8.net 広告表示 (サイドバー) ===
    st.sidebar.markdown("---")
    st.sidebar.markdown(A8_AD_HTML, unsafe_allow_html=True)
    # ===================================

    st.markdown("---")

    # 1. Black-Scholes
    if choice == txt["calc_bs"]:
        st.subheader(txt["calc_bs"])
        st.info(txt["desc_bs"])
        c1, c2 = st.columns(2)
        s = c1.number_input(txt["bs_s"], value=100.0)
        k = c2.number_input(txt["bs_k"], value=100.0)
        t = c1.number_input(txt["bs_t"], value=1.0)
        r = c2.number_input(txt["bs_r"], value=5.0)
        v = c1.number_input(txt["bs_v"], value=20.0)
        if st.button(txt["calc_btn"]):
            res = black_scholes(s, k, t, r/100, v/100)
            st.success(f"**{txt['bs_call']}: {res:.2f}**")

    # 2. Kelly Criterion
    elif choice == txt["calc_kelly"]:
        st.subheader(txt["calc_kelly"])
        st.info(txt["desc_kelly"])
        c1, c2 = st.columns(2)
        p = c1.number_input(txt["kelly_p"], value=50.0, max_value=100.0)
        rr = c2.number_input(txt["kelly_rr"], value=1.5)
        if st.button(txt["calc_btn"]):
            res = kelly_criterion(p/100, rr)
            st.metric(txt["kelly_res"], f"{res:.2f}%")
            if res > 0: st.write(f"Half Kelly: {res/2:.2f}%")
            else: st.error("Do not trade.")

    # 3. VaR
    elif choice == txt["calc_var"]:
        st.subheader(txt["calc_var"])
        st.info(txt["desc_var"])
        amt = st.number_input(txt["var_amt"], value=1000000)
        c1, c2 = st.columns(2)
        vol = c1.number_input(txt["var_vol"], value=20.0)
        conf = c2.selectbox("Confidence", [0.99, 0.95, 0.90])
        days = st.slider(txt["var_day"], 1, 100, 10)
        if st.button(txt["calc_btn"]):
            res = calculate_var(amt, vol/100, conf, days)
            st.error(f"**{txt['var_res']}: -{res:,.0f}**")

    # 4. Portfolio Variance
    elif choice == txt["calc_port"]:
        st.subheader(txt["calc_port"])
        st.info(txt["desc_port"])
        w_input = st.text_input(txt["port_w"], "0.5, 0.3, 0.2")
        v_input = st.text_input(txt["port_v"], "20, 15, 30")
        corr = st.slider(txt["port_corr"], -1.0, 1.0, 0.5)
        if st.button(txt["calc_btn"]):
            try:
                w_list = [float(x) for x in w_input.split(",")]
                v_list = [float(x)/100 for x in v_input.split(",")]
                if len(w_list) != len(v_list): st.error("Error: Length mismatch.")
                else:
                    var_p, vol_p = portfolio_risk_matrix(w_list, v_list, corr)
                    st.success(f"**{txt['port_res_vol']}: {vol_p*100:.2f}%**")
            except: st.error("Input Error.")

    # 5. GARCH
    elif choice == txt["calc_garch"]:
        st.subheader(txt["calc_garch"])
        st.info(txt["desc_garch"])
        c1, c2, c3 = st.columns(3)
        omega = c1.number_input(txt["garch_omega"], value=0.000002, format="%.6f")
        alpha = c2.number_input(txt["garch_alpha"], value=0.10)
        beta = c3.number_input(txt["garch_beta"], value=0.85)
        c4, c5 = st.columns(2)
        p_ret = c4.number_input("Prev Ret (%)", value=1.5)
        p_vol = c5.number_input("Prev Vol (%)", value=15.0)
        if st.button(txt["calc_btn"]):
            res_vol = garch_forecast(omega, alpha, beta, p_ret, p_vol)
            st.metric(txt["garch_res"], f"{res_vol:.2f}%")

    # 6. Hurst
    elif choice == txt["calc_hurst"]:
        st.subheader(txt["calc_hurst"])
        st.info(txt["desc_hurst"])
        if st.button(txt["hurst_gen_btn"]):
            rw = np.cumsum(np.random.randn(200)) + 100
            st.session_state["hurst_data"] = ",".join([f"{x:.2f}" for x in rw])
        default_data = st.session_state.get("hurst_data", "100, 101, 102, 101, 100")
        data_input = st.text_area(txt["hurst_data"], default_data)
        if st.button(txt["calc_btn"]):
            try:
                ts = [float(x) for x in data_input.split(",")]
                h = calculate_hurst(ts)
                st.metric(txt["hurst_res"], f"{h:.4f}")
                if 0.45 <= h <= 0.55: interp = "Random Walk"
                elif h > 0.55: interp = "Trending"
                else: interp = "Mean Reverting"
                st.info(f"{txt['hurst_interp']}: {interp}")
                st.line_chart(ts)
            except: st.error("Input Error.")

    st.markdown("---")
    
    # === A8.net 広告表示 (ページ下部にも表示したい場合) ===
    st.markdown("#### Sponsored Link")
    st.markdown(A8_AD_HTML, unsafe_allow_html=True)
    # =================================================
    
    st.caption(txt["disclaimer"])

if __name__ == "__main__":
    main()
