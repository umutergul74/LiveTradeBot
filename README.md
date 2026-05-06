# Scalp2: Quantitative Trading Engine

<div align="center">
  <h3>Machine Learning-Driven Market Microstructure & Trading Framework</h3>
</div>

---

**Scalp2** is a production-grade algorithmic trading framework designed for mid-to-high frequency automated trading of the `BTC/USDT` perpetual futures market. 

Moving beyond classical technical analysis, Scalp2 leverages deep learning (TCN + GRU), robust market microstructure feature engineering, wavelet denoising, and dynamic regime detection (Gaussian HMM) to capture predictive edges without look-ahead bias.

## 🚀 Key Architectural Features

- **Microstructure-First Feature Engineering**: Relies strictly on order flow, true CVD (Cumulative Volume Delta), volatility distribution, and smart money concepts rather than lagging classical indicators.
- **Wavelet Denoising Pipeline**: Utilizes causal, rolling-window wavelet transforms (Symlets) to separate signal from market noise while strictly preventing temporal data leakage.
- **Hybrid Sequence Modeling**: Employs a custom deep learning architecture combining **Temporal Convolutional Networks (TCN)** for local feature extraction and **Gated Recurrent Units (GRU)** for long-range dependency modeling.
- **Dynamic Regime Detection**: Implements an online, updating 3-State Gaussian Hidden Markov Model (HMM) equipped with anti-collapse mechanics to dynamically adjust to changing market volatility and trend states.
- **Triple Barrier Labeling**: Targets are generated using symmetric/asymmetric volatility-adjusted triple barrier methods, processed via Numba for extreme performance.
- **Walk-Forward Cross-Validation**: Backtesting and model validation strictly utilize walk-forward methodology with explicit purge/embargo windows to ensure unbiased out-of-sample performance evaluation.
- **Production Execution Engine**: Includes a live trading daemon built with CCXT, featuring fractional Kelly position sizing, adaptive trailing stops, and robust asynchronous execution management.

## 📂 Repository Structure

```text
scalp2/
├── scalp2/                     # Core Framework
│   ├── data/                   # Asynchronous data acquisition & MTF alignment
│   ├── features/               # Microstructure, Wavelet, & Volatility modules
│   ├── labeling/               # Numba-accelerated Triple Barrier labeling
│   ├── models/                 # PyTorch architectures (TCN, GRU, Hybrid)
│   ├── losses/                 # Custom objectives (Focal, Rank IC, Contrastive)
│   ├── regime/                 # Gaussian HMM with online updates
│   ├── execution/              # Risk management & trade execution logic
│   └── live/                   # Production daemon & Exchange adapters
├── notebooks/                  # Colab/Jupyter research pipeline (01_ to 06_)
├── tests/                      # Unit testing suite
├── config.yaml                 # Centralized hyperparameters & system config
├── requirements.txt            # Python dependencies
└── pyproject.toml              # Build & dependency specifications
```

## 🧠 Research to Production Pipeline

The system is designed with a strict barrier between experimentation and execution. The research pipeline is divided into modular Jupyter Notebooks for execution in GPU-accelerated environments (e.g., Google Colab):

1. **`01_data_prep.ipynb`**: Fetching, cleaning, and Multi-Timeframe (MTF) alignment.
2. **`02_feature_engineering.ipynb`**: Microstructure generation and wavelet transforms.
3. **`03_labeling.ipynb`**: Triple barrier target generation.
4. **`04_train_stage1.ipynb`**: TCN+GRU Walk-Forward cross-validation and training.
5. **`05_train_stage2.ipynb`**: Meta-learner/XGBoost layer (Optional/Bypassed).
6. **`06_backtest.ipynb`**: Rigorous OOS walk-forward backtest simulation with fractional Kelly risk modeling.

The live execution engine (`scalp2.live.bot`) consumes the finalized artifacts (state dicts, scaler fits, HMM statistics) without requiring codebase modifications.

## 🛠️ Quick Start

### Installation

Requires **Python 3.10+**.

```bash
# Clone the repository
git clone https://github.com/your-org/Scalp2.git
cd Scalp2

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -e .
```

### Configuration

All system parameters are centrally managed in `config.yaml`. Update API keys via environment variables or a `.env` file before initiating live execution.

```env
BINANCE_API_KEY=your_api_key
BINANCE_SECRET_KEY=your_secret_key
```

## ⚠️ Development Principles & Data Integrity

This framework strictly enforces quantitative integrity:
1. **Zero Look-Ahead Bias**: Causal modeling only. MTF data is strictly shifted forward. `RobustScaler` is fitted exclusively on the training fold.
2. **Real-World Execution Modeling**: Backtesting incorporates realistic slippage, maker/taker fee structures, and holding-time decay.
3. **Execution-Aware Confidence**: Model confidence gating adapts dynamically based on the current regime and long/short block conditions.

Please consult `CLAUDE.md` and `SKILLS.md` for in-depth system architecture details and specific engineering guidelines.

---
*Built for robustness, engineered for precision.*
