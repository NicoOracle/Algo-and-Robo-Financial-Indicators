# Algo-and-Robo-Financial-Indicators

1. Use the AlphaVantage API to obtain the data. Use any data from 01-01-2022 onwards.
Select three different stocks and an index. Some strategies may require high-frequency
data or more granular data than daily. If that is the case, let me know your data needs.
2. You should have a function that downloads the data, computes the returns, if needed, and
exports the data to a CSV. The ticker and date HAVE to be exported for tractability
purposes, as we have done in class.
3. You should have a function for each indicator. The function should use the data, process
the data, and return an indicator: 0/1 or -1,0,1 depending on your setting. For example, =1
indicates the signal is triggered, 0 if not. The idea is that for every date in your data, you
run the function and produce an indicator. Repeat four times (i.e., 4 functions that produce
4 indicators given a date).
4. A function that uses the inputs from 7), and produces a recommendation. For example:
buy/sell or could be more granular: strong buy, buy, nothing, sell, strong sell. The function
should be executed at each date.
5. A function that implements the trade and reports its log-return given the output in 8).
Store all the log-returns in a vector. Only store the returns where there is a trade.
6. A function that computes and prints:
  a. Number of trades per month in your testing sample.
  b. Average return and its statistical significance
  c. Average return for the longs, if any
  d. Average return for the shorts, if any.
  e. Cumulative return, annualized.
  f. Sharpe Ratio, annualized. Note: if the trades are not that frequent, there is no need
    to account for serial correlation
  g. Sortino Ratio, annualized
  h. Jensen’s Alpha. To compute the covariance and the variance, only use the market
    returns when a trade is closed. If there is no trade, you should not use the market
    return on that date.
  i. The VaR at the 5%

The goal is to work on 7, 8 and 9. Feel free to modify the strategies at will. You need to target:
  ➔ A yearly cumulative return greater than 11% (i.e., beating the SPY)
  ➔ A statistically significant return (if you match the 11%, it is likely the t-stat will be large
    enough)
  ➔ A strategy that survives a walk forward backtesting. I give you a lot of degrees of freedom
    here in the choice of the window, the gap, and the length of the data.
    
Second Set of Deliverables:
  ➔ A properly formatted code on .py or notebook that runs! Specify the dates that you used to
    run the analysis and hard-code them.
  ➔ A PowerPoint presentation that produces a high-level description of the strategy, what
    needs to be implemented, the performance, and the limitations. The slides must also discuss
    which of the four signals is the strongest predictor of your strategy, and why.
  ➔ Delivered via Canvas.
  ➔ Deadline: 04/12/2026, 5 PM EST (strictly enforced)
  
Grading:
First Deliverable: 50%. Based on consistency, and accuracy of the report.
Second Deliverable: 50%:
  ➔ Code: 15% (must include all items/functions to get full 15% credit)
  ➔ PPTX: 10%
  ➔ Success in data mining the strategy: 15%
  ➔ Presentation: 10%
