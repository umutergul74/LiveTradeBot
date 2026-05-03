# SKILLS.md — yeniBot AI Agent Operational Manual

> Read this file AND the initialization prompt in full before writing any code.
> Every rule here was learned from a real, measured failure across 8 development cycles.

---

## 1. PROJECT IDENTITY

**Goal:** Build a professional, bias-free ML pipeline that trains a TCN+GRU model to identify BTC/USDT trend direction using **market microstructure features** — NOT classical TA indicators.

**Priority order (strictly enforced):**
1. Correct data download (full Binance kline with taker data) → microstructure features
2. Correct labeling → model learns real market structure
3. Robust, bias-free training → model generalizes
4. Regime detection → understand market state
5. Execution logic → only after model is validated (Phase 2)

**Never skip steps. Never mix concerns across phases.**

---

## 2. HARD RULES — NEVER VIOLATE

### 2.1 No Look-Ahead Bias
- **Wavelets:** Causal rolling window (256-bar) only. NEVER `pywt.wavedec(full_series)`.
- **HMM:** `predict_proba_online()` (forward-only) for val/test/live. `predict_proba()` only on training fold.
- **Scalers:** `RobustScaler.fit()` on train split only. Transform val/test with train-fitted scaler.
- **Features:** Rolling windows must use `min_periods`. No future reference.
- **Labels:** Triple barrier ATR computed from train fold only. No future volatility.
- **MTF Alignment:** Shift HTF timestamps forward by one full period before `merge_asof`. A 4H bar at 08:00 covers 08:00-11:59 and must NOT be available until 12:00.

### 2.2 Feature Requirements
- **Classical TA (RSI, EMA, MACD, Bollinger) = ZERO predictive edge.** Proven: IC dropped from 0.31 to 0.004 when MTF leakage was removed. Remaining IC with TA features alone: 0.013 (statistically noise).
- **Required features:** True order flow (taker_buy_ratio, true CVD), whale detection (vol_per_trade zscore), volatility/structure (ATR, ADX, VWAP distance).
- **Data download:** Must include Binance's full 12-column kline data. CCXT `fetch_ohlcv()` only returns 5 columns by default — use Binance REST API directly if needed.

### 2.3 Labeling Must Reflect Execution Reality
- **Symmetric triple barrier in uptrend = poisonous short labels.** Use long-only binary (Long vs Not-Long) for Phase 1.
- Always validate: mean forward return by class, class balance (Long 20-50%).
- `tb_return` = sign-corrected realized return.

### 2.4 No Unnecessary Complexity
- **XGBoost meta-learner = zero added value.** Stage 1 IC = Stage 2 IC (both 0.013). Don't add it.
- **3-class softmax creates deadlocks** in long_only mode. Use binary P(Long) output.
- **Don't reduce label horizon to "fix" weak features.** Tested: 10→4 bars, IC unchanged. Fix features instead.

### 2.5 Confidence Check Must Be Direction-Aware
- If `direction_filter = "long_only"`, confidence = `prob_long` only.
- NEVER use `max(prob_short, prob_long)` — creates structural deadlock.
- In `long_only` mode: direction = LONG always. Let confidence threshold decide.

### 2.6 HMM Online Stats Anti-Collapse
- Gamma floor: 0.02 per state per bar.
- State weight floor: 8% of total N.
- N-ratio alarm: 15:1.
- Redistribution: from **trained parameter snapshot**, NEVER from dominant state.
- Persist snapshots across restarts.
- After `reset_online_stats()`: use `self._online_stats.bars = 0` (not stale local ref).

### 2.7 Colab Workflow
- After `git pull`, Python does NOT reload modules. **Always restart runtime** after pulling.
- All data saved to Google Drive (persistent across sessions).
- Code cloned from GitHub into `/content/yenibot_repo/`.
- GPU (T4) required for training notebooks.

### 2.8 Git Discipline
- Commit after every logical unit. Never batch unrelated changes.
- Format: `<type>: <what>` (feat, fix, docs, data, model)
- Never commit: `.json` state files, `.env`, `data/`, `checkpoints/*.pt`
- Never push to `main` during experiments. Use `experiment/` branches.

---

## 3. KNOWN FAILURE MODES (COMPLETE LIST)

| # | Failure | Root Cause | Fix | Conversation |
|---|---|---|---|---|
| 1 | IC = 0.31 (fake) | MTF `merge_asof(backward)` leaked 3h44m of future data | Shift HTF timestamps forward by 1 period | decdce16 |
| 2 | IC = 0.004 (real) after fix | Classical TA features have zero predictive power | Use microstructure features | decdce16 |
| 3 | XGBoost = zero value | Stage1 IC = Stage2 IC (both 0.013) | Don't add meta-learner | decdce16 |
| 4 | Reducing holding bars didn't help | Problem at feature level, not label level | Fix features, not labels | decdce16 |
| 5 | Bot never trades (long_only) | `max(prob_short, prob_long)` for confidence | Use `prob_long` only | 47f54506 |
| 6 | `direction_blocked` loop | `argmax` gates direction in long_only mode | Force direction=LONG | 47f54506 |
| 7 | HMM collapse (34:1 N ratio) | Gamma floor missing, 2% state floor too low, no snapshots | Gamma 0.02, floor 8%, snapshot restore | 47f54506 |
| 8 | HMM states converge | Redistribution copies dominant state | Reconstruct from trained snapshot | 47f54506 |
| 9 | Stale stats after reset | Local variable cached before `reset_online_stats()` | Use `self._online_stats` directly | 47f54506 |
| 10 | Backtest halts at 2021 | `drawdown_halt_pct: 8.0` triggers, never resets peak PnL | Add `backtest_mode` flag | decdce16 |
| 11 | Colab doesn't see new code | Python caches imported modules in RAM | Restart runtime after git pull | decdce16 |
| 12 | 0/53 folds generate trades | Confidence threshold too high for weak signals | Lower threshold or improve model | decdce16 |
| 13 | SHAP shows wrong features | Feature name order [latent,regime,hc] ≠ data order [latent,hc,regime] | Match name order to data order | decdce16 |
| 14 | Latent analysis on train data | PCA/t-SNE on last 5000 bars included train | Use OOS test data only | decdce16 |
| 15 | Short labels poison model | Symmetric barrier in uptrend: short TP rarely hit | Long-only binary labels | 47f54506 |
| 16 | CCXT missing taker data | `fetch_ohlcv()` returns only 5 of 12 kline columns | Use Binance REST API directly | decdce16 |
| 17 | Config/code mismatch | Changed config.yaml but not Python dataclass | Always update both together | decdce16 |
| 18 | Rank IC used tb_return | tb_return is future-derived, circular correlation | Use realized forward returns | decdce16 |

---

## 4. DIAGNOSTIC CHEAT SHEET

```
IC ≈ 0.01          → Features inadequate. Don't tune model. Fix features.
IC > 0.10           → Check for look-ahead bias (MTF alignment? scaler leak?)
One-class output    → FocalLoss alpha wrong, or model collapsed (check grad clip)
No trades           → Confidence gate direction-aware? Threshold too high?
HMM N ratio > 15:1  → State collapse. Delete online_stats, restart.
Colab stale code    → Restart runtime after git pull.
```

---

## 5. FILE STRUCTURE

```
yeniBot/
├── SKILLS.md              # This file
├── config.yaml            # All hyperparameters
├── requirements.txt
├── yenibot/               # Source code
│   ├── data/              # Download, preprocessing, MTF alignment
│   ├── features/          # Microstructure, volatility, wavelet
│   ├── labeling/          # Triple barrier (Numba)
│   ├── models/            # TCN, GRU, HybridEncoder
│   ├── losses/            # FocalLoss, RankICLoss
│   ├── regime/            # HMM with anti-collapse
│   ├── training/          # Trainer, WalkForwardCV
│   ├── execution/         # Signal generator, trade/risk manager (Phase 2)
│   └── live/              # Bot, exchange adapter (Phase 2)
├── notebooks/             # 01-06, run on Colab
├── checkpoints/           # Model artifacts (gitignored)
├── data/                  # Raw + processed (gitignored)
└── tests/                 # Unit tests
```
