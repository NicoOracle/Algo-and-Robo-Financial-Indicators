"""
Recommendation_Bot.py
=====================
Combined multi-indicator trading strategy.

All indicators use the shared list + integer index interface matching every
notebook in this project (prices_series: list[float], date_idx: int).

Indicator stack
---------------
  1. Permutation Entropy  (PE)   — chaos / order filter       [0 or 1]
     Source: Permutation_Entropy.ipynb  (indicator_1_pe)
  2. MACD                        — trend direction            [-1, 0, +1]
     Source: MACD.ipynb  (indicator_2_macd)
  3. Fibonacci Retracement       — key support / resistance   [-1, 0, +1]
     Source: Fibonacci_Retracement.ipynb  (indicator_5_fibonacci)
  4. Moving Average Crossover    — trend confirmation         [-1, 0, +1]
     Source: Permutation_Entropy.ipynb  (indicator_4_ma_crossover)

Recommendation logic
--------------------
  PE = 0  (chaotic)  →  hold  (step aside regardless of other signals)
  PE = 1  (orderly)  →  majority vote on MACD, Fibonacci, MA Crossover:
      3 bullish  →  strong_buy
      2 bullish  →  buy
      3 bearish  →  strong_sell
      2 bearish  →  sell
      mixed      →  hold

Deliverables satisfied (README items 7 – 10)
--------------------------------------------
  7.  get_recommendation()   — 5-level signal per date
  8.  execute_trade()        — log-return for a single trade
  9.  execute_trades()       — full trade log (position-based, one log-return per closed trade)
  10. compute_performance()  — metrics a – i
  +   walk_forward_backtest() — OOS validation

Usage
-----
  python Recommendation_Bot.py          # full run
  from Recommendation_Bot import execute_trades, get_recommendation
"""

import os
import csv
import math
import time
import warnings
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import requests
from scipy import stats

warnings.filterwarnings('ignore')

# ---- Import all indicator functions from the shared module ------------------
from indicators import (
    indicator_1_pe,
    indicator_2_macd,
    indicator_3_rsi,
    indicator_5_fibonacci,
    indicator_4_ma_crossover,
)


# =============================================================================
#  SECTION 0 — CONFIGURATION
# =============================================================================

API_KEY    = 'ZKMMTO1ATDBLXH2K'
TICKERS    = ['NVDA', 'MSFT', 'JPM', 'META']
INDEX      = 'SPY'
START_DATE = '2022-01-01'
END_DATE   = '2024-12-31'
RISK_FREE  = 0.05          # annual risk-free rate

TRAIN_DAYS = 252            # walk-forward: in-sample window  (~1 year)
TEST_DAYS  = 126            # walk-forward: out-of-sample window (~6 months)


# =============================================================================
#  SECTION 1 — DATA
#  Reproduced from Permutation_Entropy.ipynb (cell c04).
#  Data format: {ticker: [[ticker, date, price, log_return], ...]}
# =============================================================================

def _download_prices(ticker, start_date, end_date, api_key):
    """Download daily adjusted close prices from AlphaVantage.
    Returns list of [ticker, date, adjusted_close, log_return].
    """
    url = ('https://www.alphavantage.co/query'
           '?function=TIME_SERIES_DAILY_ADJUSTED'
           '&symbol=' + ticker +
           '&outputsize=full'
           '&apikey=' + api_key)
    response = requests.get(url)
    data     = response.json()

    ts = data.get('Time Series (Daily)', {})
    if not ts:
        print('  Warning: no data for ' + ticker + '. Keys: ' + str(list(data.keys())))
        return []

    raw = []
    for date_str, vals in ts.items():
        if start_date <= date_str <= end_date:
            raw.append([ticker, date_str, float(vals['5. adjusted close'])])
    raw.sort(key=lambda x: x[1])

    result = []
    for i, rec in enumerate(raw):
        lr = 0.0 if i == 0 else math.log(rec[2] / raw[i - 1][2])
        result.append([rec[0], rec[1], rec[2], lr])
    return result


def download_all_data(tickers, index, start_date, end_date, api_key):
    """Download + cache all ticker data. One CSV per ticker.
    Skips download if CSV already exists (respects AlphaVantage rate limits).

    Returns
    -------
    dict  {ticker: [[ticker, date, price, log_return], ...]}
    """
    all_data = {}
    for ticker in tickers + [index]:
        csv_file = ticker + '_data.csv'
        if os.path.exists(csv_file):
            print('  ' + ticker + ': loading from ' + csv_file)
            records = []
            with open(csv_file, newline='') as f:
                reader = csv.reader(f)
                next(reader)
                for row in reader:
                    records.append([row[0], row[1], float(row[2]), float(row[3])])
        else:
            print('  ' + ticker + ': downloading from AlphaVantage...')
            records = _download_prices(ticker, start_date, end_date, api_key)
            if records:
                with open(csv_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['ticker', 'date', 'adjusted_close', 'log_return'])
                    writer.writerows(records)
                print('    Saved ' + str(len(records)) + ' rows to ' + csv_file)
            else:
                print('  *** ' + ticker + ': API returned no data — will retry next run ***')
            time.sleep(15)  # AlphaVantage free tier: 5 req/min
        all_data[ticker] = records
    return all_data


# =============================================================================
#  SECTION 2 — INDICATORS
#  All four indicator functions are imported from indicators.py at the top.
#  See indicators.py for full implementations and source notebook references.
# =============================================================================


# =============================================================================
#  SECTION 3 — RECOMMENDATION  (README item 7)
# =============================================================================

def get_recommendation(pe_sig, macd_sig, fib_sig, ma_sig, rsi_sig=0):
    """Combine indicator signals into a 5-level recommendation.

    Decision tree
    -------------
    PE = 0  → 'hold'  (chaotic regime — all other signals ignored)
    PE = 1  → majority vote on {MACD, Fibonacci, MA Crossover, RSI}:
        bullish  recommendation
        ≥ 3      strong_buy / strong_sell
        2        buy / sell
        else     hold

    RSI (momentum mode, threshold=50) fires earlier in a trend than the
    slower MA crossover, allowing entry before the full MA lag resolves.

    Parameters
    ----------
    pe_sig   : int  0 or 1    (Permutation Entropy regime filter)
    macd_sig : int  -1, 0, 1  (MACD above / below signal line)
    fib_sig  : int  -1, 0, 1  (Fibonacci level proximity)
    ma_sig   : int  -1, 0, 1  (MA crossover direction)
    rsi_sig  : int  -1, 0, 1  (RSI above / below 50)

    Returns
    -------
    str : 'strong_buy' | 'buy' | 'hold' | 'sell' | 'strong_sell'
    """
    if pe_sig == 0:
        return 'hold'

    bullish = sum(1 for s in [macd_sig, fib_sig, ma_sig, rsi_sig] if s ==  1)
    bearish = sum(1 for s in [macd_sig, fib_sig, ma_sig, rsi_sig] if s == -1)

    if   bullish >= 3: return 'strong_buy'
    elif bullish == 2: return 'buy'
    elif bearish >= 3: return 'strong_sell'
    elif bearish == 2: return 'sell'
    return 'hold'


# =============================================================================
#  SECTION 4 — TRADE EXECUTION  (README items 8 / 9)
# =============================================================================

def execute_trade(recommendation, log_return_next_day):
    """Execute a single trade based on a recommendation.

    Parameters
    ----------
    recommendation      : str    output of get_recommendation()
    log_return_next_day : float  next bar's log-return (execution price)

    Returns
    -------
    (log_return, direction) — or (None, None) if recommendation is 'hold'
    """
    if recommendation in ('strong_buy', 'buy'):
        return log_return_next_day, 'long'
    elif recommendation in ('strong_sell', 'sell'):
        return -log_return_next_day, 'short'
    return None, None


def execute_trades(data_dict, tickers, index_ticker='SPY', long_only=True,
                   force_close_at_end=False):
    """Run the full indicator pipeline on each trading day for each ticker.

    Opens a long position on bullish signals; closes on bearish signals.
    When long_only=True (default), bearish signals exit existing longs but
    do NOT open short positions.  Equity stocks have an upward bias and
    lagging momentum indicators mis-time short entries during bull markets,
    producing net-negative short returns — long-only avoids this drag.

    Parameters
    ----------
    data_dict         : dict  {ticker: [[ticker, date, price, log_return], ...]}
    tickers           : list[str]  tickers to trade
    index_ticker      : str        benchmark ticker for market returns
    long_only         : bool       if True, never open short positions (default True)
    force_close_at_end: bool       if True, close any open position at the last bar.
                                   Use True in walk-forward folds so that capital
                                   deployed within the window is fully accounted for.

    Returns
    -------
    trade_log_returns        : list[float]
    trade_dates              : list[tuple]  (entry_date, exit_date, ticker, direction)
    market_returns_at_trades : list[float]  market log-return on each exit day
    """
    trade_log_returns        = []
    trade_dates              = []
    market_returns_at_trades = []
    spy_ret = {r[1]: r[3] for r in data_dict.get(index_ticker, [])}

    # max indicator warmup: Fibonacci=60, MA=50, MACD=35, PE=32 → 60
    WARMUP = 60

    for ticker in tickers:
        rows   = data_dict[ticker]
        prices = [r[2] for r in rows]
        dates  = [r[1] for r in rows]

        position    = 0      # 0 = flat, 1 = long, -1 = short
        entry_price = None
        entry_date  = None

        for i in range(WARMUP, len(prices)):
            pe  = indicator_1_pe(prices, i)
            mac = indicator_2_macd(prices, i)
            fib = indicator_5_fibonacci(prices, i, proximity_pct=0.02)
            ma  = indicator_4_ma_crossover(prices, i)
            rsi = indicator_3_rsi(prices, i)
            rec = get_recommendation(pe, mac, fib, ma, rsi)

            desired = (1  if rec in ('strong_buy',  'buy')
                  else -1 if rec in ('strong_sell', 'sell')
                  else 0)

            # In long_only mode a bearish signal (-1) closes an existing long
            # but is treated as flat (0) for entry purposes — never open short.
            entry_signal = desired if not long_only else max(desired, 0)

            # Close on active reversal: bearish signal closes long, bullish closes short.
            if position != 0 and desired == -position:
                log_ret = math.log(prices[i] / entry_price)
                if position == -1:
                    log_ret = -log_ret
                trade_log_returns.append(log_ret)
                direction = 'long' if position == 1 else 'short'
                trade_dates.append((entry_date, dates[i], ticker, direction))
                market_returns_at_trades.append(spy_ret.get(dates[i], 0.0))
                position    = 0
                entry_price = None

            # Open new position (long_only: only enter on bullish signal)
            if position == 0 and entry_signal != 0:
                position    = entry_signal
                entry_price = prices[i]
                entry_date  = dates[i]

        # Force-close any open position at the last bar in the window.
        # Required in walk-forward folds so that capital deployed within the
        # test period is fully accounted for rather than silently discarded.
        if force_close_at_end and position != 0 and entry_price is not None:
            log_ret = math.log(prices[-1] / entry_price)
            if position == -1:
                log_ret = -log_ret
            trade_log_returns.append(log_ret)
            direction = 'long' if position == 1 else 'short'
            trade_dates.append((entry_date, dates[-1], ticker, direction))
            market_returns_at_trades.append(spy_ret.get(dates[-1], 0.0))

    return trade_log_returns, trade_dates, market_returns_at_trades


# =============================================================================
#  SECTION 5 — PERFORMANCE ANALYTICS  (README item 10)
#  Reproduced from Permutation_Entropy.ipynb cell c14.
# =============================================================================

def compute_performance(trade_log_returns, trade_dates, market_returns_at_trades,
                        rf_annual=RISK_FREE, label='Strategy',
                        benchmark_ann_return=None):
    """Compute and print all required performance metrics.

    Parameters
    ----------
    trade_log_returns        : list[float]   log-return for each closed trade
    trade_dates              : list[tuple]   (entry_date, exit_date, ticker, direction)
    market_returns_at_trades : list[float]   market log-return on each exit day
    rf_annual                : float         annual risk-free rate (default 5%)
    label                    : str           report header label

    Metrics
    -------
    a. Number of trades per month
    b. Average return + t-test for statistical significance
    c. Average return for long trades
    d. Average return for short trades
    e. Cumulative return, annualised
    f. Sharpe Ratio, annualised
    g. Sortino Ratio, annualised
    h. Jensen's Alpha
    i. VaR at the 5% level
    """
    if not trade_log_returns:
        print('No trades executed.')
        return {}

    rets    = np.array(trade_log_returns)
    mkt     = np.array(market_returns_at_trades)
    n       = len(rets)
    long_r  = [r for r, (_, _, _, d) in zip(trade_log_returns, trade_dates) if d == 'long']
    short_r = [r for r, (_, _, _, d) in zip(trade_log_returns, trade_dates) if d == 'short']

    # a. Trades per month
    months = {}
    for _, exit_d, _, _ in trade_dates:
        months[exit_d[:7]] = months.get(exit_d[:7], 0) + 1
    avg_tpm = n / max(len(months), 1)

    # b. Average return + t-test
    mean_ret      = float(np.mean(rets))
    t_stat, p_val = stats.ttest_1samp(rets, 0)

    # c/d. Long / short averages
    avg_long  = float(np.mean(long_r))  if long_r  else float('nan')
    avg_short = float(np.mean(short_r)) if short_r else float('nan')

    # e. Cumulative & annualised return
    cum_ret = float(np.sum(rets))
    d0      = datetime.strptime(trade_dates[0][0],  '%Y-%m-%d')
    d1      = datetime.strptime(trade_dates[-1][1], '%Y-%m-%d')
    years   = max((d1 - d0).days / 365.25, 1e-6)
    ann_ret = cum_ret / years

    # f. Sharpe ratio (annualised)
    rf_d   = rf_annual / 252
    excess = rets - rf_d
    sharpe = float(np.mean(excess) / np.std(excess, ddof=1) * np.sqrt(252)) \
             if np.std(excess) > 0 else 0.0

    # g. Sortino ratio (annualised)
    down    = rets[rets < 0]
    d_std   = float(np.std(down, ddof=1)) if len(down) > 1 else 1e-9
    sortino = float((mean_ret / d_std) * np.sqrt(252))

    # h. Jensen's Alpha — only market returns on trade close dates
    if len(mkt) > 1 and np.var(mkt) > 0:
        cov   = np.cov(rets, mkt)
        beta  = cov[0, 1] / np.var(mkt, ddof=1)
        alpha = (mean_ret - (rf_d + beta * (np.mean(mkt) - rf_d))) * 252
    else:
        beta  = float('nan')
        alpha = float('nan')

    # i. VaR 5%
    var5 = float(np.percentile(rets, 5))

    # ---- Print report -------------------------------------------------------
    SEP = '=' * 56
    sig = ('***' if p_val < 0.01 else
           '**'  if p_val < 0.05 else
           '*'   if p_val < 0.10 else 'n.s.')
    print(SEP)
    print('  PERFORMANCE REPORT — ' + label)
    print(SEP)
    print('Total Trades           : ' + str(n) +
          '  (Long: ' + str(len(long_r)) + ', Short: ' + str(len(short_r)) + ')')
    print('')
    print('a. Avg Trades / Month  : ' + str(round(avg_tpm, 2)))
    print('')
    print('b. Mean Log-Return     : ' + str(round(mean_ret, 6)))
    print('   t-stat              : ' + str(round(float(t_stat), 4)))
    print('   p-value             : ' + str(round(float(p_val), 4)) + '  (' + sig + ')')
    print('')
    print('c. Avg Return (Longs)  : ' + str(round(avg_long,  6)))
    print('d. Avg Return (Shorts) : ' + str(round(avg_short, 6)))
    print('')
    print('e. Cumulative Return   : ' + str(round(cum_ret * 100, 2)) + '%')
    print('   Annualised Return   : ' + str(round(ann_ret * 100, 2)) + '%')
    if benchmark_ann_return is not None:
        diff = ann_ret - benchmark_ann_return
        sign = '+' if diff >= 0 else ''
        print('   SPY Annualised      : ' + str(round(benchmark_ann_return * 100, 2)) + '%')
        print('   Alpha vs SPY        : ' + sign + str(round(diff * 100, 2)) + '%')
    print('')
    print('f. Sharpe Ratio        : ' + str(round(sharpe,  4)))
    print('')
    print('g. Sortino Ratio       : ' + str(round(sortino, 4)))
    print('')
    print('h. Beta                : ' + str(round(float(beta),  4)))
    print('   Jensens Alpha       : ' + str(round(float(alpha), 6)))
    print('')
    print('i. VaR (5%)            : ' + str(round(var5 * 100, 2)) + '%')
    print(SEP)

    return dict(n_trades=n, avg_tpm=avg_tpm, mean_return=mean_ret,
                t_stat=float(t_stat), p_value=float(p_val),
                avg_long=avg_long, avg_short=avg_short,
                cum_return=cum_ret, ann_return=ann_ret,
                sharpe=sharpe, sortino=sortino,
                beta=float(beta), jensens_alpha=float(alpha), var5=var5)


# =============================================================================
#  SECTION 6 — WALK-FORWARD BACKTEST
#  Adapted from Permutation_Entropy.ipynb cell c17 (day-based windows).
# =============================================================================

def walk_forward_backtest(data_dict, tickers, index_ticker='SPY',
                          train_days=TRAIN_DAYS, test_days=TEST_DAYS):
    """Walk-forward out-of-sample validation.

    Slides a training window forward by test_days at each step.
    Strategy is applied only to the out-of-sample window.
    No parameter refitting — tests stability with fixed hyperparameters.

    Parameters
    ----------
    data_dict    : dict  {ticker: [[ticker, date, price, log_return], ...]}
    tickers      : list[str]
    index_ticker : str
    train_days   : int  in-sample window (default 252 ≈ 1 year)
    test_days    : int  out-of-sample window (default 126 ≈ 6 months)
    """
    print('=' * 60)
    print('           WALK-FORWARD BACKTEST RESULTS')
    print('  Train: ' + str(train_days) + ' days  |  Test: ' + str(test_days) + ' days')
    print('-' * 60)

    n_dates = min(len(v) for v in data_dict.values())
    folds   = []
    start   = train_days
    while start + test_days <= n_dates:
        folds.append((start, start + test_days))
        start += test_days

    print('  Total folds: ' + str(len(folds)))
    print('-' * 60)
    print('  Fold   Test Start     Test End    Ann.Ret%   Sharpe   Win%')
    print('-' * 60)

    all_oos = []
    for k, (ts, te) in enumerate(folds):
        fold_data = {t: data_dict[t][ts:te] for t in tickers + [index_ticker]}
        tr, td, mr = execute_trades(fold_data, tickers, index_ticker,
                                    force_close_at_end=True)
        if not tr:
            print('   ' + str(k + 1) + '   (no trades in this fold)')
            continue

        all_oos.extend(tr)
        r   = np.array(tr)
        d0  = datetime.strptime(td[0][0],  '%Y-%m-%d')
        d1  = datetime.strptime(td[-1][1], '%Y-%m-%d')
        yrs = max((d1 - d0).days / 365.25, 1e-6)
        ann = float(np.sum(r)) / yrs
        exc = r - RISK_FREE / 252
        sh  = (float(np.mean(exc) / np.std(exc, ddof=1) * np.sqrt(252))
               if np.std(exc) > 0 else 0.0)
        win = float(np.mean(r > 0)) * 100
        s   = fold_data[tickers[0]][0][1]  if fold_data[tickers[0]] else '---'
        e   = fold_data[tickers[0]][-1][1] if fold_data[tickers[0]] else '---'
        line = ('  ' + str(k + 1).rjust(4) + '   ' + s + '   ' + e +
                '   ' + str(round(ann * 100, 2)).rjust(8) + '%' +
                '   ' + str(round(sh, 3)).rjust(6) +
                '   ' + str(round(win, 1)).rjust(5) + '%')
        print(line)

    if all_oos:
        r_all  = np.array(all_oos)
        t, p   = stats.ttest_1samp(r_all, 0)
        print('-' * 60)
        print('  OOS avg return : ' + str(round(float(np.mean(r_all)), 6)) +
              '   t=' + str(round(float(t), 3)) +
              '   p=' + str(round(float(p), 4)))
    print('=' * 60)
    return all_oos if all_oos else []


# =============================================================================
#  SECTION 7 — MAIN
# =============================================================================

def main():
    print('\n' + '=' * 60)
    print('  RECOMMENDATION BOT — COMBINED STRATEGY')
    print('  Period     : ' + START_DATE + '  ->  ' + END_DATE)
    print('  Tickers    : ' + str(TICKERS))
    print('  Benchmark  : ' + INDEX)
    print('  Indicators : PE (filter)  |  MACD  |  Fibonacci  |  MA Crossover')
    print('=' * 60 + '\n')

    # ---- Download / load data -----------------------------------------------
    print('Loading data...')
    data = download_all_data(TICKERS, INDEX, START_DATE, END_DATE, API_KEY)
    missing = [t for t in TICKERS + [INDEX] if not data.get(t)]
    for ticker, records in data.items():
        if records:
            print('  ' + ticker + ': ' + str(len(records)) + ' records  (' +
                  records[0][1] + ' to ' + records[-1][1] + ')')
        else:
            print('  *** ' + ticker + ': NO DATA — re-run to retry download ***')
    if missing:
        print('\nSkipping backtest — missing data for: ' + str(missing))
        return

    # ---- Compute SPY annualised return as benchmark -------------------------
    spy_records = data[INDEX]
    spy_prices  = [r[2] for r in spy_records]
    spy_cum     = math.log(spy_prices[-1] / spy_prices[0])
    spy_d0      = datetime.strptime(spy_records[0][1],  '%Y-%m-%d')
    spy_d1      = datetime.strptime(spy_records[-1][1], '%Y-%m-%d')
    spy_years   = max((spy_d1 - spy_d0).days / 365.25, 1e-6)
    spy_ann     = spy_cum / spy_years

    # ---- Full-sample backtest ------------------------------------------------
    print('\n' + '─' * 60)
    print('  FULL-SAMPLE BACKTEST  —  ' + str(TICKERS))
    print('─' * 60)
    trade_returns, trade_info, mkt_rets = execute_trades(data, TICKERS, INDEX)
    compute_performance(trade_returns, trade_info, mkt_rets,
                        label='All Tickers Combined',
                        benchmark_ann_return=spy_ann)

    # ---- Per-ticker breakdown ------------------------------------------------
    for ticker in TICKERS:
        tr, td, mr = execute_trades({ticker: data[ticker], INDEX: data[INDEX]},
                                    [ticker], INDEX)
        if tr:
            compute_performance(tr, td, mr, label=ticker,
                                benchmark_ann_return=spy_ann)
        else:
            print('  [' + ticker + ']  No trades generated.')

    # ---- Walk-forward backtest -----------------------------------------------
    walk_forward_backtest(data, TICKERS, INDEX)

    # ---- Equity curve --------------------------------------------------------
    if trade_returns:
        cum_curve  = np.cumsum(trade_returns)
        exit_dates = [td[1] for td in trade_info]
        dt_exits   = [datetime.strptime(d, '%Y-%m-%d') for d in exit_dates]

        fig, ax = plt.subplots(figsize=(13, 5))
        ax.plot(dt_exits, cum_curve, linewidth=1.5, color='steelblue',
                label='Strategy cumulative log-return')
        ax.axhline(0, color='black', linewidth=0.6, linestyle='--')
        ax.set_title('Cumulative Log-Return — PE + MACD + Fibonacci + MA Crossover',
                     fontsize=13)
        ax.set_xlabel('Trade Exit Date')
        ax.set_ylabel('Cumulative Log-Return')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('recommendation_bot_equity_curve.png', dpi=150)
        plt.show()
        print('\nEquity curve saved to recommendation_bot_equity_curve.png')


if __name__ == '__main__':
    main()
