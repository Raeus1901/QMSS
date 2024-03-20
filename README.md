# 🌴 Hello 🌴
 I appreciate you stumbled upon my page and chose to take a look. I graduated from Columbia University with a M.A in  Quantitative Methods in Social Sciences (QMSS) departement, and currently in New-York. This git page will provide you with the coding I did as part of my program along with my final thesis. You can find me on [![LinkedIn][3.2]][2]. 



## 📊 Data Science 

[Lab 1](Data%20Science/Lab%201)

[Lab 2](Data%20Science/Lab%202)

[Lab 3](Data%20Science/Lab%203)

[Lab 4](Data%20Science/Lab%204)

[Lab 5](Data%20Science/Lab%205)

[Lab 6](Data%20Science/Lab%206)

[Midterm](Data%20Science/Midterm)

## ⏱️ Time series analysis 

[Lab 2](Time%20Series/Lab%202)

[Lab 3](Time%20Series/Lab%203)

## 🕸️ Social network analysis

[Lab 1](Social%20Network%20Analysis/Lab%201)

[Lab 2](Social%20Network%20Analysis/Lab%202)

## 💡 Machine Learning 

## 🤖 Natural Language Processing 

## 📖 Master Thesis

[Code](Master%20Thesis)

This code was developed for my Master's Thesis, which aims to predict price fluctuations in renewable energy stocks through sentiment analysis in time series regression. The study explores how cognitive biases—unquantified variables—impact investment decisions, independent of the available factual information. Examples include anchoring, where an investor hesitates to buy a stock deemed overvalued based on current data, and loss aversion, where there's reluctance to invest in a stock due to perceived inadequate returns based on historical data.

The hypothesis validation encompasses three critical components:

1) **Sentiment Quantification**: Utilizing a web crawler, the project collects textual data from reliable investment reports for each company every quarter from 2014 to 2024. The FinBERT model is then used to categorize the nuanced financial language in these reports into floating-point sentiment scores ranging from -1 to 1.

2) **Statistical Significance and Output**: The research applies a frequentist statistical method to ascertain the influence of sentiment scores on stock prices, specifically focusing on the closing prices on the days the quarterly earnings reports are published. Significance is determined when sentiment scores yield a p-value greater than 0.95. Additionally, the study employs ARIMA modeling to investigate stochastic effects, considering the model significant if it presents an MA score of 1 or higher with a p-value over 0.95, thus indicating a substantial impact of cognitive biases on stock prices.

3) **Model Statistical Diagnosis**: Ensuring the model's realism involves confirming that its error variance is low and its residuals are white noise, indicating all information has been adequately extracted and modeled. The model also undergoes thorough checks for hidden variables, autocorrelation, and heteroskedasticity, ensuring its structural integrity. Furthermore, the analysis includes examining the residual distribution to confirm that the sample used represents the entire data set accurately over time, suggesting that unquantifiable variables likely influence stock prices.

The completed model meets all these prerequisites, confirming a significant impact of cognitive biases on stock prices before the COVID-19 pandemic and identifying a subsequent attenuation of this effect.


[3.2]: https://raw.githubusercontent.com/MartinHeinz/MartinHeinz/master/linkedin-3-16.png (LinkedIn icon without padding)
[2]: https://www.linkedin.com/in/jean-treves-bbaa91257
