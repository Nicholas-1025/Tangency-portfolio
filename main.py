import yfinance as yf
import pandas as pd
import numpy as np
import cvxpy as cp
import gc
from typing import Dict, Optional, List, Tuple

np.random.seed(42)


def get_returns(stocks: list, period: str = '2y') -> pd.DataFrame:
    """Download historical data and return cleaned returns"""
    data = yf.download(stocks, period=period, threads=False, progress=False, auto_adjust=False)
    
    if isinstance(data.columns, pd.MultiIndex):
        adj_close = data['Adj Close'] if 'Adj Close' in data.columns.get_level_values(0) else data['Close']
    else:
        adj_close = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
    
    if isinstance(adj_close, pd.Series):
        adj_close = adj_close.to_frame()
    
    adj_close = adj_close.ffill().dropna(axis=1, thresh=len(adj_close) * 0.8)
    return adj_close.pct_change().dropna()


def get_capm_returns(stocks: list, returns: pd.DataFrame, risk_free: float = 0.04, 
                     market_ticker: str = '^GSPC', period: str = '2y') -> pd.Series:
    """CAPM Expected Returns"""
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
    for stock in stocks:
        if stock in returns_aligned.columns:
            cov = float(returns_aligned[stock].cov(market_returns) * 252)
            var = float(market_returns.var() * 252)
            beta = cov / var if var > 0 else 1.0
            capm_mu[stock] = risk_free + beta * market_excess
        else:
            capm_mu[stock] = risk_free + market_excess
    
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
    
    # Validate Sigma
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
    
    # Tangency Portfolio
    tangency_weights, tangency_sharpe, tangency_return, tangency_vol = solve_tangency_sharpe(mu, Sigma, risk_free)
    
    # Determine Strategy
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
    
    # Map back to original stocks
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
                    days: int = 252, simulations: int = 100000) -> Dict:
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
    # Configuration
    all_stocks = sorted([
    "MSFT",  # Microsoft
    "AAPL",  # Apple
    "GOOGL", # Alphabet
    "AMZN",  # Amazon
    "NVDA",  # Nvidia
    "META",  # Meta Platforms
    "JPM",   # JPMorgan Chase
    "V",     # Visa 
    "UNH",   # UnitedHealth
    "JNJ",   # Johnson & Johnson

    "XOM",   # Exxon Mobil 
    "PG",    # Procter & Gamble
    "HD",    # Home Depot 
    "KO",    # Coca-Cola
    "WMT",   # Walmart 
    "LLY",   # Eli Lilly 
    "AVGO",  # Broadcom
    "CRM",   # Salesforce 
    "HD"     # Home Depot 
])
    risk_free_rate = 0.04
    budget = 100000
    target_return = 0.18
    max_k = 6

    print(f"\n{'='*60}")
    print(f"PORTFOLIO OPTIMIZATION (CAPM + Mean-Variance)")
    print(f"{'='*60}")
    print(f"Stocks: {', '.join(all_stocks)} | Max K: {max_k}")
    print(f"Budget: ${budget:,.0f} | Target: {target_return:.0%} | Rf: {risk_free_rate:.0%}")
    print(f"{'='*60}\n")

    # Get Data
    returns = get_returns(all_stocks)
    stocks = list(returns.columns)
    mu = get_capm_returns(stocks, returns, risk_free_rate).values
    Sigma = returns.cov().values * 252
    
    # Get Prices
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

    # Stock Selection
    sel_stocks, sel_mu, sel_Sigma, sel_idx = select_top_k_with_covariance(mu, Sigma, risk_free_rate, stocks, max_k)
    
    # Solve Portfolio
    portfolio = solve_portfolio(sel_mu, sel_Sigma, risk_free_rate, budget, sel_stocks, 
                                 target_return, stocks, sel_idx)
    
    # Calculate Shares
    weights = portfolio['final_weights']
    shares = np.floor((weights * budget) / prices).astype(int)
    
    # Monte Carlo
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
    print(f"{'Stock':<10} {'Weight':>10} {'Shares':>10} {'Price':>12} {'Value':>14}")
    print(f"{'-'*60}")
    
    total_value = 0
    for i, stock in enumerate(stocks):
        if weights[i] > 0.001 or shares[i] > 0:
            value = shares[i] * prices[i]
            total_value += value
            print(f"{stock:<10} {weights[i]:>9.2%} {shares[i]:>10} ${prices[i]:>10.2f} ${value:>12,.2f}")
    
    print(f"{'-'*60}")
    print(f"Total Stocks: ${total_value:,.2f} | Cash (RF): ${budget - total_value:,.2f}")
    
    # Output: Risk Metrics
    print(f"\n{'='*60}")
    print(f"RISK METRICS (1-Year Monte Carlo)")
    print(f"{'='*60}")
    print(f"Median Value: ${mc['median']:,.2f} | Profit Prob: {mc['prob_profit']:.0%}")
    print(f"VaR (95%): ${mc['var_95']:,.2f} | CVaR (95%): ${mc['cvar_95']:,.2f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
