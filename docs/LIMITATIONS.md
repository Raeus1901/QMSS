# Limitations & Robustness Analysis

## Sample Size Constraints

The pre-COVID SARIMAX model uses N=114 pooled observations (5 firms × ~23 quarters, Q1 2014–Q1 2020) and the post-COVID model uses N=77 (5 firms × ~16 quarters, Q2 2020–Q3 2023). Both are below conventional thresholds for stable SARIMAX inference (Hyndman & Athanasopoulos 2021, §9 recommend N≥200 for reliable MA estimation). Coefficients should be interpreted with appropriate caution.

## Multiple Testing

Results are reported across 5 tickers × multiple model specifications (long-run, short-run, pre/post-COVID) without Bonferroni or FDR correction. The probability of at least one spurious significant result across this family of tests is non-trivial.

## Bootstrap Stability

The sentiment coefficient estimates (+79.55 pre-COVID, −90.23 post-COVID) warrant validation via block bootstrap (Politis & Romano 1994, B=1000 iterations) before causal interpretation, given the small-sample SARIMAX context.

## Structural Break Detection

The COVID-19 split is exogenously imposed at Q1/Q2 2020. Bai-Perron (2003) endogenous break detection would allow the data to identify the structural break date rather than assuming it, and could reveal additional regime changes within the sample.

## Crawler Reliability

Sentiment scores are derived from Google News search results scraped via BeautifulSoup. Google News URLs are non-deterministic across runs (results vary by date, region, and algorithm updates), meaning full reproducibility of the original sentiment scores requires archiving the raw HTML at collection time.
