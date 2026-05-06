# Scalp2: Quantitative Trading Framework Reference

This document serves as the definitive reference and skill set for AI agents operating within the Scalp2 project. Adhere strictly to these rules and conventions to maintain system integrity, prevent look-ahead bias, and ensure consistency.

## 🧠 System Architecture & Context

- **Objective:** High-frequency scalping/swing trading of BTC/USDT perpetual futures.
- **Data Flow:** 15m/1H (Primary) + 4H (MTF) -> Wavelet Denoising -> TCN+GRU Hybrid -> (Optional XGBoost) -> Execution.
- **Labeling:** Triple Barrier (`tb_label_cls`: 0=Short, 1=Hold, 2=Long). `tb_return` represents raw price change (negative for short drops).
- **Current Model State:** `bypass_xgboost=true`. Using raw TCN+GRU logits. The model currently exhibits a SHORT bias, but config forces `long_only`, leading to potential trade starvation (`direction_blocked`).
- **Loss Functions:** Implements advanced objective functions including LogMDD, Sharpe, Rank IC, Contrastive, Focal, and Center loss.

## ⚠️ CRITICAL RULES (NEVER VIOLATE)

### 1. Data Leakage & Look-Ahead Bias Prevention
- **Wavelets:** USE `wavelet_denoise` (causal, 256-bar rolling window) in `features/builder.py`. NEVER use `wavelet_denoise_fast` for modeling/live (causes look-ahead).
- **HMM Regime:** 
  - Training: `predict_proba()` (forward-backward) is allowed.
  - Live/Validation/Test: MUST use `predict_proba_online()` (forward-only).
- **Scaling:** RobustScaler must be fitted ONLY on the training fold.

### 2. Live Bot State & Execution Constraints
- **Execution Mode:** Currently strictly `long_only` (`config.execution.direction_filter`). 
- **Signal Thresholds:** `confidence_threshold` (0.48), `min_adx` (20), `min_atr_percentile` (0.10), `choppy_adx_override` (22).
- **HMM Online Updates:** Bot dynamically updates HMM stats. DO NOT overwrite `regime_online_stats.json` manually unless resetting a collapsed HMM.
- **State Files:** `bot_state.json`, `cycle_history.csv`, `protection_state.json` must be preserved during restarts.

### 3. Coding Conventions
- **Imports:** Use absolute imports (`from scalp2.features.builder import ...`). No relative imports.
- **Config Management:** Centralized in `config.yaml`. Parsed via `scalp2/config.py` into strict Dataclasses. Do not hardcode parameters.
- **Token Efficiency:** When writing code or markdown, be concise. Omit redundant comments.

## 📂 Project Structure & Notebook Pipeline

Code resides locally/GitHub, executed primarily on Google Colab for training, and deployed to a VPS for live trading.

| Notebook / Module | Purpose & Dependencies |
| :--- | :--- |
| `01_data_prep.ipynb` | Cleans OHLCV, resamples to 1H/4H. Run if `preprocessing.py` changes. |
| `02_feature_engineering.ipynb` | TA, Wavelet, Volatility, Smart Money. Run if `features/*` changes. |
| `03_labeling.ipynb` | Generates Triple Barrier targets. Run if `config.yaml` labeling changes. |
| `04_train_stage1.ipynb` | Trains HybridEncoder (TCN+GRU) per fold using Walk-Forward CV. |
| `05_train_stage2.ipynb` | Trains XGBoost (currently bypassed). Run if HMM/XGB changes. |
| `06_backtest.ipynb` | Walk-forward backtest simulation with Risk/Trade managers. |
| `scalp2/live/bot.py` | Production daemon for live inference and exchange execution. |
| `scalp2/regime/hmm.py` | 3-state Gaussian HMM. Handles online sufficient statistics. |
| `scalp2/execution/*` | Pipeline for Signal Generation, Trade Management, and Risk. |

## 🧪 Validation & Walk-Forward CV
- **Folds:** 52 folds (210d train, 45d val, 30d test, 30d step).
- **Purge/Embargo:** 48-bar purge + 12-bar embargo between splits to ensure temporal independence.
- **Cost Model:** 8 bps round-trip (2 bps fee/side + 2 bps slippage/side).

## 🛠️ Typical Agent Workflows
- **Debugging Bot Stops:** Check `cycle_history.csv` for rejection reasons (`choppy`, `direction_blocked`, `low_confidence`). Check `regime_online_stats.json` for HMM state collapse (N-weight imbalances).
- **Updating Config:** Modify `config.yaml`, ensure Dataclasses in `scalp2/config.py` reflect structural changes.
- **Adding Features:** Implement in `scalp2/features/builder.py`, register in pipeline, rebuild datasets via NB 01/02.
