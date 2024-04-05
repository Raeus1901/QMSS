# 🌴 Hello 🌴
 I appreciate you stumbled upon my page and chose to take a look. I graduated from Columbia University with a M.A in  Quantitative Methods in Social Sciences (QMSS) departement, and currently in New-York. This git page will provide you with the coding I did as part of my program along with my final thesis. You can find me on [![LinkedIn][3.2]][2]. 


## 📖 Master Thesis

[Code](Master%20Thesis)

This code was developed for my Master's Thesis, which aims to predict price fluctuations in renewable energy stocks through sentiment analysis in time series regression. The study explores how cognitive biases—unquantified variables—impact investment decisions, independent of the available factual information. Examples include anchoring, where an investor hesitates to buy a stock deemed overvalued based on current data, and loss aversion, where there's reluctance to invest in a stock due to perceived inadequate returns based on historical data.

The hypothesis validation encompasses three critical components:

1) **Sentiment Quantification**: Utilizing a web crawler, the project collects textual data from reliable investment reports for each company every quarter from 2014 to 2024. The FinBERT model is then used to categorize the nuanced financial language in these reports into floating-point sentiment scores ranging from -1 to 1.

2) **Statistical Significance and Output**: The research applies a frequentist statistical method to ascertain the influence of sentiment scores on stock prices, specifically focusing on the closing prices on the days the quarterly earnings reports are published. Significance is determined when sentiment scores yield a p-value greater than 0.95. Additionally, the study employs ARIMA modeling to investigate stochastic effects, considering the model significant if it presents an MA score of 1 or higher with a p-value over 0.95, thus indicating a substantial impact of cognitive biases on stock prices.

3) **Model Statistical Diagnosis**: Ensuring the model's realism involves confirming that its error variance is low and its residuals are white noise, indicating all information has been adequately extracted and modeled. The model also undergoes thorough checks for hidden variables, autocorrelation, and heteroskedasticity, ensuring its structural integrity. Furthermore, the analysis includes examining the residual distribution to confirm that the sample used represents the entire data set accurately over time, suggesting that unquantifiable variables likely influence stock prices.

The completed model meets all these prerequisites, confirming a significant impact of cognitive biases on stock prices before the COVID-19 pandemic and identifying a subsequent attenuation of this effect.


## 📊 Data Science 

[Lab 1: Data Visualisation and Manipulation](Data%20Science/Lab%201)

[Lab 2: Variables Subcategories and Data Labelling](Data%20Science/Lab%202)

[Lab 3: Multivariate Regression with Dummy Variables](Data%20Science/Lab%203)

[Lab 4: Regression with Double-Interaction Variables](Data%20Science/Lab%204)

[Lab 5: Multiple Linear and Logarithmic Probability Models.](Data%20Science/Lab%205)

[Lab 6: First Differences Regression applied on Naive ("pooled") OLS Model](Data%20Science/Lab%206)

[Midterm](Data%20Science/Midterm)

## ⏱️ Time series analysis 

[Lab 1: Unpooled Regression and Panel Data Analysis ](Time%20Series%20Analysis/Lab%20A)

[Lab 2: Multiple Variable Survival Analysis with Cox Hazard](Time%20Series%20Analysis/Lab%202)

[Lab 3: ARIMA Regression with First Differenciation and Trends](Time%20Series%20Analysis/Lab%203)

## 🕸️ Social network analysis

[Lab 1: Ego-Network Measures with Regression](Social%20Network%20Analysis/Lab%201)

[Lab 2: Degree Centrality Measures and Nodes Analysis ](Social%20Network%20Analysis/Lab%202)

[Lab 2: Community Detection Models and Advanced Vizualisation](Social%20Network%20Analysis/Lab%203)


## 💡 Machine Learning 

[Lab 1: Pandas-Seaborne Data Manipulation ](Machine%20Learning/Lab%201.ipynb)

[Lab 2: SkLearn Penalized Regression and Classifiers ](Machine%20Learning/Lab%202)

[Lab 3: SkLearn-SciPy K Means and Hierchal Clusters + PCA ](Machine%20Learning/Lab%203.ipynb)

[Lab 4: Sklearn-Keras Text Recognition Models and Neural Networks](Machine%20Learning/Lab%204.ipynb)

[Final: Sklearn-Keras Text Recognition Models and Neural Networks](Machine%20Learning/Final.ipynb)


## 🤖 Natural Language Processing 

[Lab 1: Python Basics](Natural%20Language%20Processing/Lab%201)

[Lab 2: VADER Token Sentiment Analysis](Natural%20Language%20Processing/Lab%202)

[Lab 3: NLTK Token Probabililty Classifier](Natural%20Language%20Processing/Lab%203)

[Lab 4: SkLearn Real-Time Reddit Data Classifier](Natural%20Language%20Processing/Lab%204)


## 📜 Bayesian Statistics

[Lab 1: Normal and Gaussian Density Probability Mass](Bayesian%20Stastics%20/Lab%201)

[Lab 2: Posterior Information and Beta-Binomial Distribution](Bayesian%20Stastics%20/Lab%202)

[Lab 3: Zero-Inflated Poisson and Hierarchal Models with loo-compare benchmarking](Bayesian%20Stastics%20/Lab%203)



[3.2]: https://raw.githubusercontent.com/MartinHeinz/MartinHeinz/master/linkedin-3-16.png (LinkedIn icon without padding)
[2]: https://www.linkedin.com/in/jean-treves-bbaa91257
