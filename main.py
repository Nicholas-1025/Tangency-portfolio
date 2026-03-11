import yfinance as yf
import pandas as pd
import numpy as np
import cvxpy as cp
import gc
from typing import List

# Set random seed for reproducibility
np.random.seed(42)


def get_returns(stocks: List[str], period: str = '2y') -> pd.DataFrame:
    """Download historical data and return cleaned returns"""
    print(f"📥 Downloading historical data for {stocks}...")
    
    data = yf.download(stocks, period=period, threads=False, progress=False, auto_adjust=False)
    
    # Handle MultiIndex columns
    if isinstance(data.columns, pd.MultiIndex):
        if 'Adj Close' in data.columns.get_level_values(0):
            adj_close = data['Adj Close']
        else:
            adj_close = data['Close']
    else:
        if 'Adj Close' in data.columns:
            adj_close = data['Adj Close']
        else:
            adj_close = data['Close']
    
    # Clean data
    adj_close = adj_close.ffill().dropna(axis=1, thresh=len(adj_close) * 0.8)
    returns = adj_close.pct_change().dropna()
    
    print(f"✓ Historical data: {len(returns)} days, {len(returns.columns)} stocks")
    return returns


def get_capm_returns(stocks: List[str], returns: pd.DataFrame, 
                     risk_free: float = 0.04, 
                     market_ticker: str = '^GSPC',
                     period: str = '2y') -> pd.Series:
    """
    CAPM Expected Returns: E[R] = Rf + β × (E[Rm] - Rf)
    """
    print(f"\n📥 Downloading market data ({market_ticker}) for CAPM...")
    
    # Download market data
    market_data = yf.download(market_ticker, period=period, 
                              threads=False, progress=False, auto_adjust=False)
    
    # Handle market data columns
    if isinstance(market_data.columns, pd.MultiIndex):
        if 'Adj Close' in market_data.columns.get_level_values(0):
            market_close = market_data['Adj Close']
        else:
            market_close = market_data['Close']
    else:
        if 'Adj Close' in market_data.columns:
            market_close = market_data['Adj Close']
        else:
            market_close = market_data['Close']
    
    # Handle potential MultiIndex on rows (date)
    if isinstance(market_close, pd.DataFrame):
        market_close = market_close.iloc[:, 0]  # Take first column if DataFrame
    
    market_close = market_close.ffill().dropna()
    market_returns = market_close.pct_change().dropna()
    
    # Align dates between stock returns and market returns
    market_returns = market_returns.reindex(returns.index).ffill().dropna()
    returns_aligned = returns.reindex(market_returns.index).dropna()
    
    # Calculate annualized market excess return (extract scalar value)
    market_mean = float(market_returns.mean() * 252)
    market_excess = market_mean - risk_free
    
    print(f"\n📊 CAPM Parameters:")
    print(f"   Risk-Free Rate: {risk_free:.2%}")
    print(f"   Market Expected Return: {market_mean:.2%}")
    print(f"   Market Risk Premium: {market_excess:.2%}")
    
    # Calculate beta and CAPM expected return for each stock
    capm_mu = {}
    print(f"\n📊 CAPM Expected Returns:")
    print(f"{'Stock':<10} {'Beta':<10} {'CAPM Return':<12}")
    print("-" * 35)
    
    for stock in stocks:
        if stock in returns_aligned.columns:
            stock_returns = returns_aligned[stock]
            
            # Calculate beta: Cov(Ri, Rm) / Var(Rm)
            cov = float(stock_returns.cov(market_returns) * 252)
            var = float(market_returns.var() * 252)
            beta = cov / var if var > 0 else 1.0
            
            # CAPM formula
            expected_return = risk_free + beta * market_excess
            capm_mu[stock] = expected_return
            
            print(f"{stock:<10} {beta:>9.2f} {expected_return:>11.2%}")
        else:
            # Fallback if stock not in aligned data
            expected_return = risk_free + 1.0 * market_excess
            capm_mu[stock] = expected_return
            print(f"{stock:<10} {'1.00':>9} {expected_return:>11.2%} (estimated)")
    
    print("-" * 35)
    
    return pd.Series(capm_mu)


def solve_cvxpy_portfolio(mu, Sigma, prices, risk_free, target_return, max_k, budget, stocks):
    """
    CVXPY Solver with Cardinality Constraint (Max K stocks)
    """
    n = len(mu)
    
    # 1. Find Tangency Portfolio (DCP-compliant)
    x = cp.Variable(n)
    k = cp.Variable()
    
    constraints_tangency = [
        (mu - risk_free) @ x == 1,
        cp.sum(x) == k,
        x >= 0,
        k >= 0
    ]
    
    prob_tangency = cp.Problem(cp.Minimize(cp.quad_form(x, Sigma)), constraints_tangency)
    
    try:
        prob_tangency.solve(solver=cp.SCS, verbose=False)
        
        if prob_tangency.status in ['optimal', 'optimal_inaccurate']:
            x_val = x.value
            k_val = k.value
            
            if k_val is not None and k_val > 1e-10:
                tangency_weights = x_val / k_val
            else:
                tangency_weights = np.ones(n) / n
        else:
            print(f"⚠ Tangency solve status: {prob_tangency.status}")
            tangency_weights = np.ones(n) / n
    except Exception as e:
        print(f"⚠ Tangency solve error: {e}")
        tangency_weights = np.ones(n) / n
    
    # Calculate tangency portfolio metrics
    tangency_ret = float(mu @ tangency_weights)
    tangency_vol = float(np.sqrt(tangency_weights @ Sigma @ tangency_weights))
    tangency_sharpe = (tangency_ret - risk_free) / tangency_vol if tangency_vol > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"🎯 TANGENCY PORTFOLIO (Maximum Sharpe Ratio)")
    print(f"{'='*60}")
    print(f"Expected Return:  {tangency_ret:>10.2%}")
    print(f"Volatility:       {tangency_vol:>10.2%}")
    print(f"Sharpe Ratio:     {tangency_sharpe:>10.2f}")
    print(f"\n{'Stock':<10} {'Weight':<12} {'Allocation':<15}")
    print("-" * 40)
    for i in range(n):
        if tangency_weights[i] > 0.001:
            alloc = tangency_weights[i] * budget
            print(f"{stocks[i]:<10} {tangency_weights[i]:>11.2%} ${alloc:>12,.2f}")
    print(f"{'='*60}")
    
    final_weights_risky = np.zeros(n)
    weight_risk_free = 0.0
    strategy = ""
    
    # 2. Logic Branching based on Target Return
    if target_return < tangency_ret:
        strategy = "CML (Risk-Free + Tangency)"
        
        if tangency_ret - risk_free == 0:
            w_risky = 0
        else:
            w_risky = (target_return - risk_free) / (tangency_ret - risk_free)
        
        w_risky = max(0, min(1, w_risky))
        weight_risk_free = 1.0 - w_risky
        final_weights_risky = w_risky * tangency_weights
        
        print(f"\n📉 Target ({target_return:.2%}) < Tangency ({tangency_ret:.2%})")
        print(f"   → Using Capital Market Line")
        print(f"   → Risk-Free: {weight_risk_free:.2%}, Risky: {w_risky:.2%}")
        
    else:
        strategy = "Efficient Frontier (Constrained)"
        weight_risk_free = 0.0
        
        z = cp.Variable(n, boolean=True) 
        w = cp.Variable(n)
        
        M = 1.0 
        constraints = [
            mu @ w == target_return,
            cp.sum(w) == 1,
            w >= 0,
            w <= M * z,
            cp.sum(z) <= max_k
        ]
        
        prob = cp.Problem(cp.Minimize(cp.quad_form(w, Sigma)), constraints)
        
        try:
            prob.solve(solver=cp.CBC, verbose=False) 
            
            if prob.status not in ['optimal', 'optimal_inaccurate']:
                print("⚠ MIP Solver not available, relaxing integer constraint.")
                constraints_relaxed = [
                    mu @ w == target_return,
                    cp.sum(w) == 1,
                    w >= 0,
                ]
                prob_relaxed = cp.Problem(cp.Minimize(cp.quad_form(w, Sigma)), constraints_relaxed)
                prob_relaxed.solve(solver=cp.SCS, verbose=False)
                
                if prob_relaxed.status in ['optimal', 'optimal_inaccurate']:
                    final_weights_risky = w.value
                    top_k_idx = np.argsort(final_weights_risky)[-max_k:]
                    mask = np.zeros(n)
                    mask[top_k_idx] = 1
                    final_weights_risky = final_weights_risky * mask
                    if final_weights_risky.sum() > 0:
                        final_weights_risky = final_weights_risky / final_weights_risky.sum()
                else:
                    print(f"❌ Relaxed optimization failed: {prob_relaxed.status}")
                    final_weights_risky = tangency_weights
            else:
                final_weights_risky = w.value
                final_weights_risky[final_weights_risky < 1e-4] = 0
                print(f"\n📈 Target ({target_return:.2%}) >= Tangency ({tangency_ret:.2%})")
                print(f"   → Using Constrained Efficient Frontier")
                print(f"   → Active Stocks: {np.sum(final_weights_risky > 0)}")
        except Exception as e:
            print(f"❌ Solver Error: {e}")
            final_weights_risky = tangency_weights

    risky_budget = budget * (1 - weight_risk_free)
    dollar_amounts = final_weights_risky * risky_budget
    shares = np.floor(dollar_amounts / prices).astype(int)
    
    return {
        'weights_risky': final_weights_risky,
        'weight_rf': weight_risk_free,
        'strategy': strategy,
        'shares': shares,
        'tangency_ret': tangency_ret,
        'tangency_sharpe': tangency_sharpe,
        'tangency_weights': tangency_weights
    }


def solve_dp_portfolio(mu, Sigma, prices, risk_free, target_return, max_k, budget, stocks):
    """
    Discrete Optimization (Recursive Search)
    """
    n = len(mu)
    best_solution = None
    best_diff = float('inf')
    best_vol = float('inf')
    
    est_tangency_ret = float(mu @ (np.ones(n)/n))
    use_risk_free = target_return < est_tangency_ret
    
    if use_risk_free:
        target_risky_return = est_tangency_ret 
        print(f"\n📉 DP: Target < Est. Tangency. Using Risk-Free Mix.")
    else:
        target_risky_return = target_return
        print(f"\n📈 DP: Target >= Est. Tangency. Targeting Frontier.")

    def search(idx, current_shares, current_cost, current_count):
        nonlocal best_solution, best_diff, best_vol
        
        if current_cost > budget or current_count > max_k:
            return

        if idx == n:
            if current_cost == 0 or np.sum(current_shares) == 0:
                return
                
            shares_arr = np.array(current_shares)
            values = shares_arr * prices
            total_val = np.sum(values)
            if total_val == 0: return
            
            w = values / total_val
            port_ret = float(mu @ w)
            port_vol = float(np.sqrt(w @ Sigma @ w))
            
            if use_risk_free:
                score = (port_ret - risk_free) / port_vol
                current_metric = -score 
                diff = 0
            else:
                diff = abs(port_ret - target_risky_return)
                current_metric = diff
            
            is_better = False
            if best_solution is None:
                is_better = True
            elif use_risk_free:
                if current_metric < best_diff:
                    is_better = True
            else:
                if diff < best_diff or (diff == best_diff and port_vol < best_vol):
                    is_better = True
            
            if is_better:
                best_diff = current_metric
                best_vol = port_vol if not use_risk_free else best_vol
                best_solution = {
                    'shares': shares_arr,
                    'weights': w,
                    'ret': port_ret,
                    'vol': port_vol
                }
            return

        max_possible = int((budget - current_cost) / prices[idx])
        limit = min(max_possible, 50)
        
        start_share = 0
        if current_count >= max_k:
            start_share = 0
            limit = 0
            
        for s in range(start_share, limit + 1):
            new_count = current_count + (1 if s > 0 else 0)
            if new_count > max_k and s > 0: 
                continue
                
            search(idx + 1, 
                   current_shares + [s], 
                   current_cost + s * prices[idx], 
                   new_count)

    search(0, [], 0, 0)
    
    if best_solution is None:
        return {'weights_risky': np.ones(n)/n, 'weight_rf': 0, 'shares': np.ones(n), 
                'strategy': 'Fallback', 'tangency_ret': 0, 'tangency_sharpe': 0, 'tangency_weights': np.ones(n)/n}

    final_weight_rf = 0.0
    final_weights_risky = best_solution['weights']
    
    if use_risk_free:
        actual_tangency_ret = best_solution['ret']
        if actual_tangency_ret - risk_free == 0:
            w_risky = 0
        else:
            w_risky = (target_return - risk_free) / (actual_tangency_ret - risk_free)
        final_weight_rf = 1.0 - w_risky
        final_weights_risky = w_risky * best_solution['weights']
        strategy = "DP CML (Risk-Free + Tangency)"
    else:
        strategy = "DP Efficient Frontier"

    risky_budget = budget * (1 - final_weight_rf)
    final_shares = np.floor((final_weights_risky * budget) / prices).astype(int)

    return {
        'weights_risky': final_weights_risky,
        'weight_rf': final_weight_rf,
        'strategy': strategy,
        'shares': final_shares,
        'tangency_ret': best_solution['ret'] if use_risk_free else target_return,
        'tangency_sharpe': 0,
        'tangency_weights': best_solution['weights'] if use_risk_free else np.zeros(n)
    }


def main():
    # Configuration
    global stocks
    stocks = sorted(['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA'])
    risk_free_rate = 0.04
    budget = 100000
    max_k = 3
    target_return_input = 0.15
    market_ticker = '^GSPC'  # S&P 500 as market benchmark

    print(f"\n{'='*60}")
    print(f"PORTFOLIO OPTIMIZATION WITH CAPM EXPECTED RETURNS")
    print(f"{'='*60}")
    print(f"Stocks: {', '.join(stocks)}")
    print(f"Budget: ${budget:,.0f}")
    print(f"Max Stocks (K): {max_k}")
    print(f"Target Return: {target_return_input:.2%}")
    print(f"Risk-Free Rate: {risk_free_rate:.2%}")
    print(f"Market Benchmark: {market_ticker}")
    print(f"{'='*60}\n")

    # 1. Get Historical Data
    returns = get_returns(stocks)
    
    # 2. Get Expected Returns using CAPM
    mu_series = get_capm_returns(stocks, returns, risk_free=risk_free_rate, 
                                  market_ticker=market_ticker)
    mu = mu_series.values
    
    # 3. Calculate Covariance Matrix (annualized)
    Sigma = returns.cov().values * 252
    
    # 4. Get Current Prices
    print(f"\n📥 Downloading current prices...")
    prices_data = yf.download(stocks, period='5d', threads=False, progress=False, auto_adjust=False)
    
    if isinstance(prices_data.columns, pd.MultiIndex):
        if 'Adj Close' in prices_data.columns.get_level_values(0):
            col_level = 'Adj Close'
        else:
            col_level = 'Close'
        last_prices_series = prices_data[col_level].iloc[-1].ffill()
        prices = last_prices_series.values
    else:
        if 'Adj Close' in prices_data.columns:
            prices = prices_data['Adj Close'].iloc[-1].ffill().values
        else:
            prices = prices_data['Close'].iloc[-1].ffill().values

    print(f"✓ Latest prices: {dict(zip(stocks, [f'{p:.2f}' for p in prices]))}")
    
    # 5. CLEAN UP DATA (Free memory)
    print(f"\n🧹 Cleaning up downloaded data from memory...")
    del returns
    del mu_series
    del prices_data
    gc.collect()
    print(f"✓ Data cleanup complete")

    print(f"\n{'='*60}")
    print(f"OPTIMIZATION SETTINGS")
    print(f"{'='*60}")
    print(f"Budget: ${budget:,.0f}")
    print(f"Max Stocks (K): {max_k}")
    print(f"Target Return: {target_return_input:.2%}")
    print(f"Risk-Free Rate: {risk_free_rate:.2%}")
    print(f"{'='*60}\n")

    # 6. Run CVXPY
    print("🔄 Running CVXPY (Continuous + MIP)...")
    res_cvx = solve_cvxpy_portfolio(mu, Sigma, prices, risk_free_rate, target_return_input, max_k, budget, stocks)
    
    # 7. Run DP
    print("\n🔄 Running DP (Discrete Shares)...")
    res_dp = solve_dp_portfolio(mu, Sigma, prices, risk_free_rate, target_return_input, max_k, budget, stocks)

    # 8. Display Final Portfolio Results
    def print_results(name, res):
        print(f"\n{'='*60}")
        print(f"📋 {name} FINAL PORTFOLIO")
        print(f"{'='*60}")
        print(f"Strategy: {res['strategy']}")
        print(f"Risk-Free Weight: {res['weight_rf']:.2%}")
        print(f"Risky Asset Weight: {1 - res['weight_rf']:.2%}")
        
        print(f"\n{'Stock':<10} {'Weight':<12} {'Shares':<10} {'Price':<12} {'Value':<12}")
        print("-" * 56)
        
        total_stock_value = 0
        for i, stock in enumerate(stocks):
            if res['weights_risky'][i] > 0.001 or res['shares'][i] > 0:
                value = res['shares'][i] * prices[i]
                total_stock_value += value
                print(f"{stock:<10} {res['weights_risky'][i]:>11.2%} {res['shares'][i]:>8} ${prices[i]:>10.2f} ${value:>10,.2f}")
        
        print("-" * 56)
        print(f"Total Invested in Stocks: ${total_stock_value:,.2f}")
        print(f"Cash Remaining (Risk-Free): ${budget - total_stock_value:,.2f}")
        print(f"{'='*60}")

    print_results("CVXPY", res_cvx)
    print_results("DP", res_dp)

    # 9. Summary Comparison
    print(f"\n{'='*60}")
    print(f"📊 METHOD COMPARISON")
    print(f"{'='*60}")
    print(f"{'Metric':<30} {'CVXPY':<15} {'DP':<15}")
    print("-" * 60)
    print(f"{'Risk-Free Weight':<30} {res_cvx['weight_rf']:>13.2%} {res_dp['weight_rf']:>13.2%}")
    print(f"{'Active Stocks':<30} {np.sum(res_cvx['weights_risky'] > 0):>13.0f} {np.sum(res_dp['weights_risky'] > 0):>13.0f}")
    print(f"{'Strategy':<30} {res_cvx['strategy']:>15} {res_dp['strategy']:>15}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
