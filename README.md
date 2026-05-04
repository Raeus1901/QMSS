# QMSS — Columbia University Graduate Portfolio

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python)](https://www.python.org/)
[![FinBERT](https://img.shields.io/badge/NLP-FinBERT-orange?style=flat-square)](https://huggingface.co/yiyanghkust/finbert-tone)
[![SARIMAX](https://img.shields.io/badge/Time%20Series-SARIMAX-green?style=flat-square)](https://www.statsmodels.org/)
[![Columbia](https://img.shields.io/badge/Columbia%20University-QMSS%20M.A-lightblue?style=flat-square)](https://qmss.columbia.edu/)

Coursework and master thesis from the **Quantitative Methods in Social Sciences (QMSS)** M.A. at Columbia University (GPA: 3.92). The repository covers applied work across machine learning, NLP, Bayesian statistics, time series analysis, and data science.

---

## Master Thesis — Earnings Sentiment & Renewable Energy Stock Returns (2014–2023)

**`/Master_Thesis`**

> *"The Cognitive of Finances: Applying Sentiment Analysis to Time Series in Understanding Shifts in the Renewable Energy Stock Market"*  
> Columbia University, GSAS — QMSS 5999, December 2023

### Research Question

Does investor sentiment derived from quarterly earnings reports (10-Q) significantly predict stock price movements in renewable energy equities — and does this relationship hold before and after COVID-19?

The study tests whether cognitive biases (anchoring, loss aversion) leave measurable traces in stock prices, challenging the **Efficient Market Hypothesis (EMH)**.

### Companies Studied

| Ticker | Company | Sector |
|--------|---------|--------|
| FSLR | First Solar | Solar Power |
| GE | General Electric | Hydro / Wind |
| NEE | NextEra Energy | Wind Power |
| TSLA | Tesla | Electric Vehicles |
| PLUG | Plug Power | Hydrogen / Fuel Cells |

Data period: **Q1 2014 — Q3 2023** (40 quarterly observations per company)

---

### Pipeline Overview

```
Earnings Call Text
  → Web crawler (BeautifulSoup + Google Search top-10 results)
  → FinBERT-tone sentiment scoring (yiyanghkust/finbert-tone)
      · Chunked tokenization (512-token windows with CLS/SEP)
      · Weighted score: Positive (+1) / Negative (-1) / Neutral (0)
      · Output: continuous sentiment_score ∈ [-1, 1]
        ↓
Financial Data (yfinance + Financial Modeling Prep API)
  → 10-year daily OHLCV prices
  → Quarterly Surprise EPS = actual − estimated earnings
        ↓
Stationarity Tests (ADF + KPSS) per ticker
        ↓
SARIMAX Regression (auto_arima order selection)
  → Long-run: full 2014–2023 period per ticker
  → Short-run: ±5 days window around each earnings date
  → Final model: pre-COVID (2014–2020) vs post-COVID (2020–2023)
        ↓
Model Diagnostics
  → Durbin-Watson · Shapiro-Wilk · Ljung-Box · Jarque-Bera
```

---

### Key Results

#### Stationarity Tests
All five tickers showed **non-stationarity** under ADF (unit root not rejected) and **trend-stationarity** under KPSS, requiring first differencing before modelling.

| Ticker | ADF p-value | KPSS p-value |
|--------|-------------|--------------|
| GE | 0.687 | 0.032 |
| NEE | 0.075 | 0.010 |
| TSLA | 0.998 | 0.015 |
| PLUG | 0.315 | 0.048 |
| FSLR | 0.999 | 0.023 |

#### Long-Run SARIMAX — Surprise EPS as predictor

Auto-ARIMA selected **ARIMA(0,1,0)** for all five tickers — a pure random walk, consistent with weak-form EMH.

| Ticker | SEPS Coeff | p-value | Durbin-Watson | Shapiro-Wilk p |
|--------|------------|---------|---------------|----------------|
| FSLR | −1.001 | 0.855 | 1.302 | 0.118 |
| GE | −2.717 | 0.381 | 1.153 | ~0.000 |
| NEE | −14.311 | 0.618 | 1.276 | 0.001 |
| TSLA | +27.840 | 0.697 | 1.902 | ~0.000 |
| **PLUG** | **+37.118** | **0.008** | **2.182** | **0.001** |

**PLUG is the sole exception**: SEPS is statistically significant (p=0.008) with a positive impact on stock price, suggesting the nascent hydrogen market was less informationally efficient than its peers.

#### Final Model — Sentiment + SEPS, SARIMAX(0,2,1)

| Period | Sentiment Coeff | p-value | SEPS Coeff | p-value |
|--------|-----------------|---------|------------|---------|
| Pre-COVID (2014–Q1 2020) | **+79.55** | **0.000** | +1.001 | 0.941 |
| Post-COVID (Q2 2020–2023) | **−90.23** | **0.039** | +75.38 | 0.120 |

**Core finding:**
- **Pre-COVID**: Sentiment was the primary driver of renewable stock prices (p=0.000). The renewable market's growth was linked to positive cognitive beliefs around the green transition, not earnings fundamentals (SEPS p=0.941).
- **Post-COVID**: Sentiment **inverted** sign (−90.23, p=0.039) — pandemic-induced macroeconomic disruption altered investor psychology. SEPS gained significance (p improved from 0.941 → 0.120), suggesting a gradual shift toward fundamental-driven investing.

#### Model Diagnostics

| Model | Ljung-Box (Q) | Jarque-Bera p | Heterosked. p |
|-------|---------------|----------------|----------------|
| Short-run + sentiment | 66.47 (p=0.00) | 0.01 | 0.00 |
| Pre-COVID final | 36.68 (p=0.00) | 0.63 | 0.09 |
| Post-COVID final | 23.53 (p=0.00) | **0.97** | **0.38** |

Progressive improvement: the post-COVID model achieves near-normal residual distribution (JB p=0.97) and non-significant heteroskedasticity.

---

### Conclusions

- All five tickers follow a **random walk** in the long run — consistent with weak-form EMH
- **Sentiment significantly drove prices pre-COVID** (coeff +79.55, p=0.000), consistent with cognitive biases (anchoring, loss aversion) shaping early renewable market growth
- **Post-COVID sentiment inverted** (coeff −90.23, p=0.039): positive sentiment now associates with lower prices, reflecting market skepticism and post-pandemic uncertainty
- **PLUG Power** deviates from EMH with a statistically significant SEPS relationship (p=0.008) — consistent with its nascent stage and low investor coverage pre-2020
- The increasing SEPS significance post-COVID supports a long-run convergence toward **market efficiency** (Graham's weighing machine, 1965)

---

### Tech Stack

| Component | Library / Tool |
|-----------|---------------|
| Data ingestion | `yfinance`, `requests` (Financial Modeling Prep API) |
| Web crawling | Custom `BeautifulSoup` + Google Search crawler |
| NLP / Sentiment | `transformers` — FinBERT (`yiyanghkust/finbert-tone`, 4.9B token corpus) |
| Time series modelling | `statsmodels` SARIMAX, `pmdarima` auto_arima |
| Statistical tests | `scipy.stats` Shapiro-Wilk, `statsmodels` ADF / KPSS / Ljung-Box / Durbin-Watson |
| Data manipulation | `pandas`, `numpy` |
| Visualisation | `matplotlib`, `seaborn` |

---

### Getting Started

```bash
git clone https://github.com/Raeus1901/QMSS.git
cd QMSS/Master_Thesis

pip install -r requirements.txt

# Set your Financial Modeling Prep API key
export FMP_API_KEY=your_key_here

python Master_Thesis.py
```

**Requirements:** Python 3.9+, ~4GB RAM for FinBERT inference

---

## Coursework Modules

### Bayesian Statistics
| Lab | Topic |
|-----|-------|
| Lab 1 | Normal and Gaussian Density, Probability Mass Functions |
| Lab 2 | Posterior Inference and Beta-Binomial Distribution |
| Lab 3 | Zero-Inflated Poisson and Hierarchical Models with LOO-CV benchmarking |

### Data Science
| Lab | Topic |
|-----|-------|
| Lab 1 | Data Visualisation and Manipulation |
| Lab 2 | Variable Subcategories and Data Labelling |
| Lab 3 | Multivariate Regression with Dummy Variables |
| Lab 4 | Regression with Double-Interaction Variables |
| Lab 5 | Multiple Linear and Logarithmic Probability Models |
| Lab 6 | First Differences Regression on Pooled OLS |

### Machine Learning
| Lab | Topic |
|-----|-------|
| Lab 1 | Pandas / Seaborn Data Manipulation |
| Lab 2 | Scikit-learn Penalized Regression and Classifiers |
| Lab 3 | K-Means and Hierarchical Clustering + PCA |
| Lab 4 | Keras Text Recognition and Neural Networks |
| Final | End-to-end ML pipeline |

### Natural Language Processing
| Lab | Topic |
|-----|-------|
| Lab 1 | Python Fundamentals |
| Lab 2 | VADER Token Sentiment Analysis |
| Lab 3 | NLTK Token Probability Classifier |
| Lab 4 | Scikit-learn Real-Time Reddit Data Classifier |

### Social Network Analysis
| Lab | Topic |
|-----|-------|
| Lab 1 | Ego-Network Measures with Regression |
| Lab 2 | Degree Centrality and Node Analysis |
| Lab 3 | Community Detection Models and Advanced Visualisation |

### Time Series Analysis
| Lab | Topic |
|-----|-------|
| Lab 1 | Unpooled Regression and Panel Data |
| Lab 2 | Survival Analysis with Cox Proportional Hazard |
| Lab 3 | ARIMA with First Differencing and Trend Decomposition |

---

## Author

**Jean Trèves** — M.A. QMSS, Columbia University (GPA: 3.92)  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/jean-treves-bbaa91257)
[![GitHub](https://img.shields.io/badge/GitHub-Raeus1901-black?style=flat-square&logo=github)](https://github.com/Raeus1901)
