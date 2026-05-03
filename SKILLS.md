# SKILLS.md — Scalp2 AI Agent Operational Manual

> Read this file in full before writing any code. Every rule here was learned from a real failure.

---

## 1. PROJECT IDENTITY

**Goal:** Build a professional, bias-free ML pipeline that trains a TCN+GRU model to accurately identify BTC/USDT trend direction, regime, and entry opportunities — before adding any execution or trade management complexity.

**Priority order (strictly enforced):**
1. Correct labeling → model learns real market structure
2. Robust, bias-free training → model generalizes
3. Regime detection → understand market state
4. Execution logic → only after model is validated
5. Trade management → only after execution is validated

**Never skip steps. Never mix concerns across phases.**

---

## 2. HARD RULES — NEVER VIOLATE

### 2.1 No Look-Ahead Bias
- Wavelets: Use **causal-only** rolling window implementation. NEVER apply wavelet to the full series.
- HMM: `predict_proba_online()` (forward-only) for val/test/live. `predict_proba()` (forward-backward) only on training fold data.
- Scalers: `RobustScaler.fit()` on train split only. Transform val/test with the train-fitted scaler.
- Feature engineering: Any rolling window calculation must use `min_periods` and never reference future bars.
- Labels: Triple barrier must be computed **bar by bar** using only past ATR. Never use future volatility to size barriers.

### 2.2 Labeling Must Reflect Execution Reality
- **Symmetric triple barrier in a directional market (e.g. BTC 2020-2026 uptrend) produces poisonous short labels.** Short TP is hit rarely; short SL (above entry) is hit constantly. This teaches the model phantom short patterns.
- Solution: Use **long-only labels** (binary: will this be a profitable long entry?) OR use **asymmetric barriers** with tighter SL for shorts.
- Always log and inspect label distribution. If short labels > 35% in a long-term uptrend dataset, the labeling is suspect.
- `tb_return` must store the **sign-corrected** realized return. For short labels, return = `(entry - exit) / entry`. Never store raw price delta for both sides.

### 2.3 Confidence Check Must Be Direction-Aware
- If `direction_filter = "long_only"`, confidence = `prob_long` only. NEVER use `max(prob_short, prob_long)` — this creates a structural deadlock where the irrelevant class blocks all trades.
- In `long_only` mode: direction = LONG always. Let the confidence threshold decide.
- In `both` mode: `argmax(prob_short, prob_long)` determines direction.

### 2.4 HMM Online Stats Anti-Collapse
- State weight floor: minimum **8%** per state (not 2%).
- N-ratio alarm threshold: **15:1** (not 50:1).
- Gamma floor: Apply `max(gamma, 0.02)` per state per bar before accumulating stats. This breaks the positive feedback loop.
- State redistribution: Reconstruct starving state from **trained parameter snapshot** (mean, covariance). NEVER copy from dominant state — this collapses all states to the same distribution.
- Trained parameter snapshots must be persisted across bot restarts. Initialize in `_init_online_stats()`, re-initialize in `set_online_stats_dict()` and lazily in `update_online()`.
- `reset_online_stats()` creates a new `_online_stats` object. Always use `self._online_stats.bars_since_update = 0` after reset (not a stale local reference).

### 2.5 Git Discipline
- Commit after every logical unit of work. Never batch unrelated changes.
- Commit message format: `<type>: <what changed>` (e.g. `fix: direction-aware confidence check`)
- Never commit: state files (`*.json`, `*.csv` runtime output), model weights unless explicitly versioned, `.env`.
- Always verify `git status` before committing. Separate model code from data/state.

---

## 3. ARCHITECTURE

```
Data (OHLCV, 1H primary + 4H MTF)
    ↓ Causal wavelet denoising
    ↓ Feature engineering (TA, volatility, order flow, smart money)
    ↓ Purged Walk-Forward CV split
    ↓ RobustScaler (fit on train only)
    ↓ Triple Barrier Labeling (long-only or asymmetric)
    ↓ TCN + GRU HybridEncoder → latent embeddings
    ↓ (Optional) XGBoost meta-learner on [latent + handcrafted + HMM regime]
    ↓ Signal Generator (direction-aware confidence, regime filters)
    ↓ Trade Manager + Risk Manager
    ↓ Live Bot (paper → live)
```

**Current production state:**
- `bypass_xgboost: true` — raw TCN+GRU softmax is used directly
- `direction_filter: "long_only"`
- HMM: 3-state Gaussian, full covariance, online update enabled

---

## 4. LABELING SYSTEM

### Triple Barrier Parameters (current)
```yaml
tp_multiplier: 2.0   # ATR multiples for take-profit
sl_multiplier: 5.0   # ATR multiples for stop-loss
max_holding_bars: 10 # 1H bars = 10 hours
atr_period: 14
```

### Label Classes
| Class | Value | Meaning |
|---|---|---|
| Short | 0 | Short TP hit before Long TP |
| Hold  | 1 | Time barrier or SL hit |
| Long  | 2 | Long TP hit first |

**Important:** In long-only backtests, Short labels (class 0) are simply not traded. Their quality affects the model's probability calibration — poor short labels inflate prob_short and suppress prob_long.

---

## 5. MODEL ARCHITECTURE

### TCN + GRU HybridEncoder
- **TCN branch:** Dilated causal convolutions, captures local patterns
- **GRU branch:** Captures sequential dependencies
- **Fusion:** Concatenate → linear head
- **Output:** 3-class softmax [P(short), P(hold), P(long)]

### Loss Functions Available
- `FocalLoss`: Handles class imbalance
- `CenterLoss`: Compact latent clusters per class
- `RankICLoss`: Maximize rank correlation with forward returns
- `SharpeRatioLoss`: Portfolio-aware training objective
- `LogMDDLoss`: Penalizes drawdown during training

### Walk-Forward CV
- 52 folds: 7560 train / 1080 val / 720 test bars (1H)
- Purge: 12 bars, Embargo: 3 bars between splits
- Each fold: fresh RobustScaler + fresh HMM

---

## 6. REGIME DETECTION (HMM)

- 3 states: Bull / Bear / Choppy (mapped at fit time, never changed)
- Features: `log_return`, `gk_vol_14`, `adx`, `cvd_delta_zscore`, `vwap_dist_atr`
- Covariance: `full`
- Online update: `decay_factor=0.998`, `update_interval=24`, `min_samples=48`
- Choppy threshold: 0.50 (P(choppy) > 0.50 AND ADX < 22 → skip)

---

## 7. SIGNAL FILTER CHAIN (in order)

1. Choppy regime + ADX check
2. Time-of-day filter (disabled by default)
3. ADX minimum (20)
4. ATR percentile minimum (0.10)
5. **Direction-aware confidence check** (prob_long ≥ 0.48 in long_only mode)
6. Regime-direction alignment (optional)
7. Direction assignment (forced LONG in long_only mode)
8. SL protection, risk manager checks
9. Kelly position sizing

---

## 8. FILE STRUCTURE

```
scalp2/
├── config.py              # Typed dataclass config loader
├── data/
│   ├── preprocessing.py   # OHLCV cleaning
│   └── mtf_builder.py     # Multi-timeframe alignment (shift forward to prevent leakage)
├── features/
│   ├── builder.py         # Full feature engineering pipeline
│   └── wavelet.py         # wavelet_denoise (causal) — NEVER use wavelet_denoise_fast
├── labeling/
│   └── triple_barrier.py  # Numba-accelerated barrier labeling
├── models/
│   ├── hybrid.py          # HybridEncoder (TCN+GRU)
│   ├── tcn.py             # TCN backbone
│   ├── gru.py             # GRU backbone
│   ├── attention.py       # Attention mechanism
│   └── meta_learner.py    # XGBoost wrapper
├── losses/                # Custom loss functions
├── regime/
│   └── hmm.py             # Online Gaussian HMM with anti-collapse mechanisms
├── training/
│   ├── trainer.py         # Stage 1 training loop
│   ├── stage2_trainer.py  # Stage 2 (XGBoost) training
│   └── walk_forward.py    # PurgedWalkForwardCV
├── execution/
│   ├── signal_generator.py # 14-step signal pipeline
│   ├── trade_manager.py    # Partial TP, trailing stop, breakeven
│   └── risk_manager.py     # Daily/weekly loss limits, drawdown halts
└── live/
    ├── bot.py             # Main live trading daemon
    ├── data_pipeline.py   # Real-time feature computation
    ├── exchange.py        # Binance CCXT adapter
    └── notifier.py        # Telegram notifications
```

---

## 9. TYPICAL DEBUGGING WORKFLOWS

### Bot not opening trades
1. Check `cycle_history.csv` for rejection reason column
2. `direction_blocked` → check `direction_filter` config and signal_generator step 9b
3. `low_confidence` → check `confidence_threshold` and whether check is direction-aware (step 8)
4. `choppy` → check `regime_online_stats.json` for HMM N-weight imbalance (ratio > 15:1 = problem)
5. `low_adx` / `low_volatility` → market genuinely ranging, may be correct behavior

### HMM collapse diagnosis
- Open `regime_online_stats.json`, check `"N"` array
- If ratio > 15:1 (e.g. [117, 3.4, 3.4]) → collapsed
- Fix: Delete `regime_online_stats.json` and restart bot
- Root cause: Missing gamma floor, too-low state weight floor, or restart without snapshot restore

### Backtest stops generating trades
- `RiskManager` has `backtest_mode` flag — enable it to disable live-only safeguards
- Check daily/weekly loss limit config values

---

## 10. KNOWN FAILURE MODES (NEVER REPEAT)

| Failure | Root Cause | Fix |
|---|---|---|
| Bot never opens trades in long_only mode | `max(prob_short, prob_long)` used for confidence check instead of `prob_long` | Direction-aware confidence check |
| `direction_blocked` infinite loop | `argmax` of 3-class softmax used to gate direction in long_only mode | Force direction=LONG in long_only, skip argmax gate |
| HMM state collapse (34:1 N ratio) | Gamma floor missing + 2% state weight floor too low + restart without snapshots | Gamma floor 0.02, floor 8%, reconstructive redistribution |
| Short labels poisoning model | Symmetric triple barrier in uptrending market | Use long-only labels or asymmetric barriers |
| Look-ahead in wavelets | `wavelet_denoise_fast` applied to full series | Use causal `wavelet_denoise` only |
| Stale reference after reset_online_stats | `stats.bars_since_update = 0` on local copy | Use `self._online_stats.bars_since_update = 0` |
| Regression after "both" mode | Model's short labels are low quality (trained on poisonous data) | Long-only or retrain with proper labels |
