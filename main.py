import yfinance as yf
import pandas as pd
import numpy as np
import cvxpy as cp
import gc
from typing import Dict, Optional, List, Tuple

np.random.seed(42)


def get_returns(stocks: list, period: str = '2y') -> pd.DataFrame:
    """Download historical data (supports US + HK stocks)"""
    print(f"📥 Downloading data for {len(stocks)} stocks...")
    
    data = yf.download(stocks, period=period, threads=False, progress=False, auto_adjust=False)
    
    if isinstance(data.columns, pd.MultiIndex):
        adj_close = data['Adj Close'] if 'Adj Close' in data.columns.get_level_values(0) else data['Close']
    else:
        adj_close = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
    
    if isinstance(adj_close, pd.Series):
        adj_close = adj_close.to_frame()
    
    adj_close = adj_close.ffill().dropna(axis=1, thresh=len(adj_close) * 0.6)
    returns = adj_close.pct_change().dropna()
    
    print(f"✓ Data: {len(returns)} days, {len(returns.columns)} stocks")
    return returns


def get_dividend_yield(stocks: list) -> Dict[str, float]:
    """
    Get trailing dividend yield for each stock
    Supports US (.US) and HK (.HK) stocks
    """
    print(f"\n📊 Fetching dividend data...")
    dividend_yields = {}
    
    for stock in stocks:
        try:
            ticker = yf.Ticker(stock)
            info = ticker.info
            
            # Get dividend yield (already annualized %)
            div_yield = info.get('trailingAnnualDividendYield', 0)
            
            # Fallback: calculate from dividend rate and price
            if div_yield == 0 or div_yield is None:
                div_rate = info.get('trailingAnnualDividendRate', 0)
                price = info.get('currentPrice', info.get('previousClose', 0))
                if div_rate > 0 and price > 0:
                    div_yield = div_rate / price
            
            dividend_yields[stock] = div_yield if div_yield else 0.0
            
        except Exception as e:
            dividend_yields[stock] = 0.0
    
    # Display dividend info
    print(f"{'Stock':<15} {'Div Yield':>12}")
    print("-" * 30)
    for stock, yield_val in dividend_yields.items():
        print(f"{stock:<15} {yield_val:>11.2%}")
    print("-" * 30)
    
    return dividend_yields


def get_capm_returns(stocks: list, returns: pd.DataFrame, risk_free: float = 0.04,
                     dividend_yields: Dict[str, float] = None,
                     market_ticker: str = '^GSPC', period: str = '2y') -> pd.Series:
    """
    CAPM Expected Returns WITH DIVIDEND YIELD
    
    Total Expected Return = CAPM Price Return + Dividend Yield
    E[R_total] = Rf + β × (E[Rm] - Rf) + Dividend_Yield
    """
    print(f"\n📥 Downloading market data ({market_ticker})...")
    
    market_data = yf.download(market_ticker, period=period, threads=False, progress=False, auto_adjust=False)
    
    if isinstance(market_data.columns, pd.MultiIndex):
        market_close = market_data['Adj Close'] if 'Adj Close' in market_data.columns.get_level_values(0) else market_data['Close']
    else:
        market_close = market_data['Adj Close'] if 'Adj Close' in market_data.columns else market_data['Close']
    
    if isinstance(market_close, pd.DataFrame):
        market_close = market_close.iloc[:, 0]
    
    market_close = market_close.ffill().dropna()
    market_returns = market_close.pct_change().dropna()
    market_returns = market_returns.reindex(returns.index).ffill().dropna()
    returns_aligned = returns.reindex(market_returns.index).dropna()
    
    market_mean = float(market_returns.mean() * 252)
    market_excess = market_mean - risk_free
    
    capm_mu = {}
    print(f"\n📊 EXPECTED RETURNS (CAPM + Dividends):")
    print(f"{'Stock':<15} {'Beta':>8} {'CAPM':>10} {'Div':>8} {'Total':>10}")
    print("-" * 55)
    
    for stock in stocks:
        if stock in returns_aligned.columns:
            cov = float(returns_aligned[stock].cov(market_returns) * 252)
            var = float(market_returns.var() * 252)
            beta = cov / var if var > 0 else 1.0
        else:
            beta = 1.0
        
        # CAPM price return
        capm_price_return = risk_free + beta * market_excess
        
        # Add dividend yield
        div_yield = dividend_yields.get(stock, 0.0) if dividend_yields else 0.0
        total_return = capm_price_return + div_yield
        
        capm_mu[stock] = total_return
        
        print(f"{stock:<15} {beta:>7.2f} {capm_price_return:>9.2%} {div_yield:>7.2%} {total_return:>9.2%}")
    
    print("-" * 55)
    
    return pd.Series(capm_mu)


def solve_tangency_sharpe(mu: np.ndarray, Sigma: np.ndarray, risk_free: float) -> Tuple[np.ndarray, float, float, float]:
    """Solve tangency portfolio"""
    n = len(mu)
    
    if Sigma.shape != (n, n):
        Sigma = np.eye(n) * 0.04
    
    if np.any(np.isnan(Sigma)) or np.any(np.isinf(Sigma)):
        Sigma = np.nan_to_num(Sigma, nan=0.04, posinf=0.04, neginf=0.04)
    Sigma = (Sigma + Sigma.T) / 2
    min_eig = np.min(np.linalg.eigvalsh(Sigma))
    if min_eig < 1e-8:
        Sigma = Sigma + (1e-6 - min_eig) * np.eye(n)
    
    x = cp.Variable(n)
    k = cp.Variable()
    constraints = [(mu - risk_free) @ x == 1, cp.sum(x) == k, x >= 0, k >= 0]
    prob = cp.Problem(cp.Minimize(cp.quad_form(x, Sigma)), constraints)
    
    try:
        prob.solve(solver=cp.SCS, verbose=False)
        if prob.status in ['optimal', 'optimal_inaccurate'] and k.value is not None and k.value > 1e-10:
            weights = x.value / k.value
        else:
            weights = np.ones(n) / n
    except:
        weights = np.ones(n) / n
    
    port_ret = float(mu @ weights)
    port_vol = float(np.sqrt(weights @ Sigma @ weights))
    sharpe = (port_ret - risk_free) / port_vol if port_vol > 0 else 0
    
    return weights, sharpe, port_ret, port_vol


def solve_efficient_frontier_target(mu: np.ndarray, Sigma: np.ndarray, risk_free: float, target_return: float) -> Tuple[np.ndarray, float, float, float]:
    """Solve for minimum variance at target return"""
    n = len(mu)
    
    if Sigma.shape != (n, n):
        Sigma = np.eye(n) * 0.04
    
    if np.any(np.isnan(Sigma)) or np.any(np.isinf(Sigma)):
        Sigma = np.nan_to_num(Sigma, nan=0.04, posinf=0.04, neginf=0.04)
    Sigma = (Sigma + Sigma.T) / 2
    min_eig = np.min(np.linalg.eigvalsh(Sigma))
    if min_eig < 1e-8:
        Sigma = Sigma + (1e-6 - min_eig) * np.eye(n)
    
    w = cp.Variable(n)
    constraints = [mu @ w == target_return, cp.sum(w) == 1, w >= 0]
    prob = cp.Problem(cp.Minimize(cp.quad_form(w, Sigma)), constraints)
    
    try:
        prob.solve(solver=cp.SCS, verbose=False)
        weights = w.value if prob.status in ['optimal', 'optimal_inaccurate'] else np.ones(n) / n
    except:
        weights = np.ones(n) / n
    
    port_ret = float(mu @ weights)
    port_vol = float(np.sqrt(weights @ Sigma @ weights))
    sharpe = (port_ret - risk_free) / port_vol if port_vol > 0 else 0
    
    return weights, sharpe, port_ret, port_vol


def select_top_k_with_covariance(mu: np.ndarray, Sigma: np.ndarray, risk_free: float,
                                  stocks: list, max_k: Optional[int] = None) -> Tuple[list, np.ndarray, np.ndarray, list]:
    """Forward stepwise selection with covariance"""
    n = len(mu)
    
    if max_k is None or max_k >= n:
        return stocks, mu, Sigma, list(range(n))
    
    selected_indices = []
    remaining_indices = list(range(n))
    
    for step in range(1, max_k + 1):
        if len(remaining_indices) == 0:
            break
        
        best_sharpe = -np.inf
        best_candidate = None
        
        for candidate_idx in remaining_indices:
            test_indices = selected_indices + [candidate_idx]
            test_mu = mu[test_indices]
            test_Sigma = Sigma[np.ix_(test_indices, test_indices)]
            _, test_sharpe, _, _ = solve_tangency_sharpe(test_mu, test_Sigma, risk_free)
            
            if test_sharpe > best_sharpe:
                best_sharpe = test_sharpe
                best_candidate = candidate_idx
        
        if best_candidate is None:
            break
        
        selected_indices.append(best_candidate)
        remaining_indices.remove(best_candidate)
    
    final_mu = mu[selected_indices]
    final_Sigma = Sigma[np.ix_(selected_indices, selected_indices)]
    final_stocks = [stocks[i] for i in selected_indices]
    
    return final_stocks, final_mu, final_Sigma, selected_indices


def solve_portfolio(mu: np.ndarray, Sigma: np.ndarray, risk_free: float, budget: float, 
                    stocks: list, target_return: float = None,
                    original_stocks: list = None, original_indices: list = None) -> Dict:
    """Solve portfolio with target return logic"""
    n = len(mu)
    
    if Sigma.ndim == 1:
        Sigma = Sigma.reshape(1, 1)
    if Sigma.shape != (n, n):
        Sigma = np.eye(n) * 0.04 if Sigma.shape[0] != n else Sigma[:n, :n]
    if np.any(np.isnan(Sigma)) or np.any(np.isinf(Sigma)):
        Sigma = np.nan_to_num(Sigma, nan=0.04, posinf=0.04, neginf=0.04)
    Sigma = (Sigma + Sigma.T) / 2
    min_eig = np.min(np.linalg.eigvalsh(Sigma))
    if min_eig < 1e-8:
        Sigma = Sigma + (1e-6 - min_eig) * np.eye(n)
    
    tangency_weights, tangency_sharpe, tangency_return, tangency_vol = solve_tangency_sharpe(mu, Sigma, risk_free)
    
    weight_rf = 0.0
    final_weights = tangency_weights.copy()
    final_return = tangency_return
    final_vol = tangency_vol
    final_sharpe = tangency_sharpe
    
    if target_return is not None:
        if target_return < tangency_return:
            w_risky = max(0, min(1, (target_return - risk_free) / (tangency_return - risk_free))) if tangency_return != risk_free else 0
            weight_rf = 1.0 - w_risky
            final_weights = w_risky * tangency_weights
            final_return = target_return
            final_vol = w_risky * tangency_vol
            strategy = f"CML ({weight_rf:.0%} RF)"
        elif abs(target_return - tangency_return) < 0.001:
            strategy = "Tangency (Max Sharpe)"
        else:
            ef_weights, ef_sharpe, ef_ret, ef_vol = solve_efficient_frontier_target(mu, Sigma, risk_free, target_return)
            final_weights, final_sharpe, final_return, final_vol = ef_weights, ef_sharpe, ef_ret, ef_vol
            strategy = f"Efficient Frontier ({target_return:.0%})"
    else:
        strategy = "Tangency (Max Sharpe)"
    
    if original_stocks is not None and original_indices is not None:
        full_weights = np.zeros(len(original_stocks))
        for i, idx in enumerate(original_indices):
            full_weights[idx] = final_weights[i]
        final_weights = full_weights
    
    return {
        'tangency_weights': tangency_weights if original_stocks is None else np.zeros(len(original_stocks)),
        'final_weights': final_weights,
        'weight_rf': weight_rf,
        'strategy': strategy,
        'tangency_return': tangency_return,
        'tangency_vol': tangency_vol,
        'tangency_sharpe': tangency_sharpe,
        'final_return': final_return,
        'final_vol': final_vol,
        'final_sharpe': final_sharpe,
        'active_stocks': int(np.sum(final_weights > 0.001))
    }


def run_monte_carlo(weights: np.ndarray, mu: np.ndarray, Sigma: np.ndarray, 
                    budget: float, risk_free: float, weight_rf: float,
                    days: int = 252, simulations: int = 5000) -> Dict:
    """Monte Carlo simulation"""
    dt = 1.0 / days
    
    nonzero_mask = np.abs(weights) > 1e-6
    if np.sum(nonzero_mask) > 0:
        risky_weights = weights[nonzero_mask]
        risky_weights = risky_weights / np.sum(risky_weights)
        port_mu = float(np.dot(risky_weights, mu[nonzero_mask]))
        port_var = float(np.dot(risky_weights, Sigma[np.ix_(nonzero_mask, nonzero_mask)] @ risky_weights))
        port_vol = np.sqrt(port_var)
    else:
        port_mu, port_vol = risk_free, 0
    
    final_values = []
    for _ in range(simulations):
        if port_vol > 0:
            daily_returns = (port_mu - 0.5 * port_var) * dt + port_vol * np.random.normal(0, 1, days) * np.sqrt(dt)
            risky_cumulative = np.prod(1 + daily_returns)
        else:
            risky_cumulative = np.exp(risk_free)
        
        final_values.append(budget * (weight_rf * np.exp(risk_free) + (1 - weight_rf) * risky_cumulative))
    
    final_values = np.array(final_values)
    
    var_95 = budget - np.percentile(final_values, 5)
    worst_5pct = final_values[final_values <= np.percentile(final_values, 5)]
    cvar_95 = budget - np.mean(worst_5pct) if len(worst_5pct) > 0 else var_95
    
    return {
        'median': np.percentile(final_values, 50),
        'var_95': var_95,
        'cvar_95': cvar_95,
        'prob_profit': np.sum(final_values > budget) / simulations
    }


def main():
    # Configuration - SUPPORTS US AND HK STOCKS!
    # US Stocks: 'AAPL', 'MSFT', etc.
    # HK Stocks: '0700.HK' (Tencent), '9988.HK' (Alibaba), '0005.HK' (HSBC)
    all_stocks = sorted([
    # 🇺🇸 美股 (12檔)
    "MSFT",   # Microsoft - 雲端與AI
    "AAPL",   # Apple - 消費電子
    "NVDA",   # Nvidia - AI晶片
    "GOOGL",  # Alphabet - 搜尋廣告
    "AMZN",   # Amazon - 電商雲端
    "META",   # Meta - 社群媒體
    "JPM",    # JPMorgan - 銀行
    "V",      # Visa - 支付網路
    "LLY",    # Eli Lilly - 生技醫療
    "BRK.B",  # Berkshire Hathaway - 多元控股
    "KO",     # Coca-Cola - 防禦型消費
    "XOM",    # Exxon Mobil - 能源

    # 🇭🇰 港股 (8檔)
    "0700.HK",  # 騰訊 Tencent - 科技龍頭
    "1810.HK",  # 小米 Xiaomi - 消費電子/EV
    "3690.HK",  # 美團 Meituan - 本地生活
    "1299.HK",  # AIA - 保險龍頭
    "0005.HK",  # 匯豐 HSBC - 銀行
    "0883.HK",  # 中海油 CNOOC - 能源
    "0941.HK",  # 中國移動 China Mobile - 電訊
    "0388.HK",  # 港交所 HKEX - 金融基建
])
    risk_free_rate = 0.04
    budget = 100000
    target_return = 0.12
    max_k = 6
    market_ticker = '^GSPC'  # Use ^HSI for HK market benchmark

    print(f"\n{'='*60}")
    print(f"PORTFOLIO OPTIMIZATION (US + HK STOCKS)")
    print(f"{'='*60}")
    print(f"Stocks: {', '.join(all_stocks)}")
    print(f"Budget: ${budget:,.0f} | Target: {target_return:.0%} | Rf: {risk_free_rate:.0%}")
    print(f"Max Stocks: {max_k} | Market: {market_ticker}")
    print(f"{'='*60}\n")

    # 1. Get Historical Returns
    returns = get_returns(all_stocks)
    stocks = list(returns.columns)
    
    # 2. Get Dividend Yields
    dividend_yields = get_dividend_yield(stocks)
    
    # 3. Get CAPM Returns (WITH DIVIDENDS)
    mu = get_capm_returns(stocks, returns, risk_free_rate, dividend_yields, market_ticker).values
    
    # 4. Calculate Covariance
    Sigma = returns.cov().values * 252
    
    # 5. Get Current Prices
    print(f"\n📥 Fetching current prices...")
    prices_data = yf.download(stocks, period='5d', threads=False, progress=False, auto_adjust=False)
    if isinstance(prices_data.columns, pd.MultiIndex):
        col = 'Adj Close' if 'Adj Close' in prices_data.columns.get_level_values(0) else 'Close'
        prices = prices_data[col].iloc[-1].ffill().values
    else:
        col = 'Adj Close' if 'Adj Close' in prices_data.columns else 'Close'
        prices = prices_data[col].iloc[-1].ffill().values
    
    # Cleanup
    del returns, prices_data
    gc.collect()

    # 6. Stock Selection
    sel_stocks, sel_mu, sel_Sigma, sel_idx = select_top_k_with_covariance(mu, Sigma, risk_free_rate, stocks, max_k)
    
    # 7. Solve Portfolio
    portfolio = solve_portfolio(sel_mu, sel_Sigma, risk_free_rate, budget, sel_stocks, 
                                 target_return, stocks, sel_idx)
    
    # 8. Calculate Shares
    weights = portfolio['final_weights']
    shares = np.floor((weights * budget) / prices).astype(int)
    
    # 9. Monte Carlo
    mc = run_monte_carlo(weights, mu, Sigma, budget, risk_free_rate, portfolio['weight_rf'])
    
    # Output: Tangency Portfolio
    print(f"\n{'='*60}")
    print(f"TANGENCY PORTFOLIO (Maximum Sharpe Ratio)")
    print(f"{'='*60}")
    print(f"Return: {portfolio['tangency_return']:.2%} | Vol: {portfolio['tangency_vol']:.2%} | Sharpe: {portfolio['tangency_sharpe']:.4f}")
    print(f"{'-'*60}")
    
    # Output: Final Portfolio
    print(f"\n{'='*60}")
    print(f"FINAL PORTFOLIO")
    print(f"{'='*60}")
    print(f"Strategy: {portfolio['strategy']}")
    print(f"Return: {portfolio['final_return']:.2%} | Vol: {portfolio['final_vol']:.2%} | Sharpe: {portfolio['final_sharpe']:.4f}")
    print(f"Risk-Free: {portfolio['weight_rf']:.0%} | Active Stocks: {portfolio['active_stocks']}")
    print(f"{'-'*60}")
    print(f"{'Stock':<15} {'Weight':>10} {'Shares':>10} {'Price':>12} {'Value':>14} {'Div':>8}")
    print(f"{'-'*60}")
    
    total_value = 0
    for i, stock in enumerate(stocks):
        if weights[i] > 0.001 or shares[i] > 0:
            value = shares[i] * prices[i]
            total_value += value
            div = dividend_yields.get(stock, 0.0)
            print(f"{stock:<15} {weights[i]:>9.2%} {shares[i]:>10} ${prices[i]:>10.2f} ${value:>12,.2f} {div:>7.2%}")
    
    print(f"{'-'*60}")
    print(f"Total Stocks: ${total_value:,.2f} | Cash (RF): ${budget - total_value:,.2f}")
    
    # Output: Risk Metrics
    print(f"\n{'='*60}")
    print(f"RISK METRICS (1-Year Monte Carlo)")
    print(f"{'='*60}")
    print(f"Median Value: ${mc['median']:,.2f} | Profit Prob: {mc['prob_profit']:.0%}")
    print(f"VaR (95%): ${mc['var_95']:,.2f} | CVaR (95%): ${mc['cvar_95']:,.2f}")
    print(f"{'='*60}\n")
    
    # Output: Dividend Summary
    print(f"{'='*60}")
    print(f"DIVIDEND SUMMARY")
    print(f"{'='*60}")
    portfolio_div_yield = sum(weights[i] * dividend_yields.get(stock, 0.0) for i, stock in enumerate(stocks))
    annual_div_income = portfolio_div_yield * total_value
    print(f"Portfolio Dividend Yield: {portfolio_div_yield:.2%}")
    print(f"Estimated Annual Dividend Income: ${annual_div_income:,.2f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main() 
