import streamlit as st
import numpy as np
from scipy.stats import norm

# ==========================================
# ファイル名: finance_calc.py
# 実行コマンド: streamlit run finance_calc.py
# ==========================================

# --- 1. 設定 & 言語データ (Configuration & Localization) ---
st.set_page_config(
    page_title="Quant Calculator Pro",
    page_icon="📈",
    layout="centered"
)

# 翻訳辞書
TRANS = {
    "JP": {
        "title": "高度金融計算機 (Quant Calculator)",
        "sidebar_title": "計算モデル選択",
        "lang_select": "言語 / Language",
        "calc_bs": "ブラック・ショールズ (オプション)",
        "calc_kelly": "ケリー基準 (資金管理)",
        "calc_var": "VaR (バリュー・アット・リスク)",
        "calc_btn": "計算する",
        "result": "計算結果",
        "desc_bs": "コールオプションの理論価格とグリークスを計算します。",
        "desc_kelly": "破産を避けつつ資産を最大化する最適な投資比率を計算します。",
        "desc_var": "特定の確率で発生しうる最大損失額を計算します。",
        # Black-Scholes Inputs
        "bs_s": "現在株価 (S)",
        "bs_k": "行使価格 (K)",
        "bs_t": "満期までの期間 (年)",
        "bs_r": "無リスク金利 (%)",
        "bs_v": "ボラティリティ (%)",
        "bs_call": "コール価格",
        # Kelly Inputs
        "kelly_p": "勝率 (%)",
        "kelly_rr": "リスクリワードレシオ (利益/損失)",
        "kelly_res": "推奨レバレッジ (資金の%)",
        "kelly_note": "※実務では計算結果の半分(ハーフケリー)を使うことが一般的です。",
        # VaR Inputs
        "var_amt": "投資元本",
        "var_vol": "年率ボラティリティ (%)",
        "var_conf": "信頼区間 (%)",
        "var_day": "保有期間 (日)",
        "var_res": "推定最大損失額",
        # Disclaimer
        "disclaimer": "免責事項: 本ツールの計算結果は参考値であり、投資勧誘や利益を保証するものではありません。投資は自己責任で行ってください。"
    },
    "EN": {
        "title": "Quant Calculator Pro",
        "sidebar_title": "Select Model",
        "lang_select": "Language",
        "calc_bs": "Black-Scholes (Option Pricing)",
        "calc_kelly": "Kelly Criterion (Money Mgmt)",
        "calc_var": "Value at Risk (VaR)",
        "calc_btn": "Calculate",
        "result": "Result",
        "desc_bs": "Calculate theoretical call option price and Greeks.",
        "desc_kelly": "Calculate optimal bet size to maximize wealth while avoiding ruin.",
        "desc_var": "Estimate the maximum potential loss with a given confidence level.",
        # Black-Scholes Inputs
        "bs_s": "Spot Price (S)",
        "bs_k": "Strike Price (K)",
        "bs_t": "Time to Maturity (Years)",
        "bs_r": "Risk-Free Rate (%)",
        "bs_v": "Volatility (%)",
        "bs_call": "Call Price",
        # Kelly Inputs
        "kelly_p": "Win Rate (%)",
        "kelly_rr": "Risk/Reward Ratio",
        "kelly_res": "Optimal Leverage (% of Equity)",
        "kelly_note": "*It is common practice to use half of this value (Half-Kelly).",
        # VaR Inputs
        "var_amt": "Portfolio Value",
        "var_vol": "Annual Volatility (%)",
        "var_conf": "Confidence Level (%)",
        "var_day": "Holding Period (Days)",
        "var_res": "Estimated Max Loss (VaR)",
        # Disclaimer
        "disclaimer": "Disclaimer: Results are for informational purposes only. Trading involves risk."
    }
}

# --- 2. 計算ロジック (Calculation Logic) ---

def black_scholes(S, K, T, r, sigma):
    # r and sigma should be in decimal (e.g., 0.05 for 5%)
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
