import yfinance as yf
import pandas as pd
import numpy as np
import cvxpy as cp
import gc
from typing import Dict

# Set random seed for reproducibility
np.random.seed(42)


def get_returns(stocks: list, period: str = '2y') -> pd.DataFrame:
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
    
    # Handle potential MultiIndex on rows (date)
    if isinstance(adj_close, pd.DataFrame):
        adj_close = adj_close.iloc[:, 0] if adj_close.shape[1] == 1 else adj_close
    
    # Clean data
    adj_close = adj_close.ffill().dropna(axis=1, thresh=len(adj_close) * 0.8)
    returns = adj_close.pct_change().dropna()
    
    print(f"✓ Historical  {len(returns)} days, {len(returns.columns)} stocks")
    return returns


def get_capm_returns(stocks: list, returns: pd.DataFrame, 
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
        market_close = market_close.iloc[:, 0]
    
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


def solve_tangency_portfolio(mu: np.ndarray, Sigma: np.ndarray, 
                              risk_free: float, budget: float, 
                              stocks: list, target_return: float = None) -> Dict:
    """
    Find Tangency Portfolio (Maximum Sharpe Ratio)
    With optional CML allocation based on target return
    
    Mathematical Foundation:
    max (w'μ - rf) / √(w'Σw)  s.t.  w'1 = 1, w ≥ 0
    
    Reformulated as:
    min x'Σx  s.t.  (μ - rf·1)'x = 1,  then w = x / sum(x)
    """
    print(f"\n{'='*60}")
    print(f"🎯 SOLVING TANGENCY PORTFOLIO (Maximum Sharpe Ratio)")
    print(f"{'='*60}")
    
    n = len(mu)
    
    # DCP-compliant tangency portfolio formulation
    x = cp.Variable(n)
    k = cp.Variable()
    
    constraints = [
        (mu - risk_free) @ x == 1,  # Excess return constraint
        cp.sum(x) == k,              # Sum of scaled weights
        x >= 0,                      # Long-only
        k >= 0                       # Positive scaling
    ]
    
    prob = cp.Problem(cp.Minimize(cp.quad_form(x, Sigma)), constraints)
    
    try:
        prob.solve(solver=cp.SCS, verbose=False)
        
        if prob.status in ['optimal', 'optimal_inaccurate']:
            x_val = x.value
            k_val = k.value
            
            if k_val is not None and k_val > 1e-10:
                tangency_weights = x_val / k_val
            else:
                tangency_weights = np.ones(n) / n
                print("⚠ Degenerate solution, using equal weights")
        else:
            print(f"⚠ Optimization status: {prob.status}, using equal weights")
            tangency_weights = np.ones(n) / n
    except Exception as e:
        print(f"⚠ Optimization error: {e}, using equal weights")
        tangency_weights = np.ones(n) / n
    
    # Calculate tangency portfolio metrics
    tangency_return = float(mu @ tangency_weights)
    tangency_volatility = float(np.sqrt(tangency_weights @ Sigma @ tangency_weights))
    tangency_sharpe = (tangency_return - risk_free) / tangency_volatility if tangency_volatility > 0 else 0
    
    print(f"\n📊 TANGENCY PORTFOLIO METRICS:")
    print(f"   Expected Return:  {tangency_return:>10.2%}")
    print(f"   Volatility:       {tangency_volatility:>10.2%}")
    print(f"   Sharpe Ratio:     {tangency_sharpe:>10.2f}")
    print(f"   Risk-Free Rate:   {risk_free:>10.2%}")
    
    print(f"\n{'Stock':<10} {'Weight':<12} {'Expected Return':<15}")
    print("-" * 40)
    
    for i, stock in enumerate(stocks):
        if tangency_weights[i] > 0.001:
            print(f"{stock:<10} {tangency_weights[i]:>11.2%} {mu[i]:>14.2%}")
    
    print("-" * 40)
    print(f"{'Total':<10} {tangency_weights.sum():>11.2%}")
    print(f"{'='*60}")
    
    # CML Allocation: Mix Risk-Free + Tangency based on target return
    weight_rf = 0.0
    final_weights = tangency_weights.copy()
    strategy = "100% Tangency Portfolio"
    
    if target_return is not None:
        if target_return < tangency_return:
            # Use CML: mix risk-free + tangency
            if tangency_return - risk_free != 0:
                w_risky = (target_return - risk_free) / (tangency_return - risk_free)
                w_risky = max(0, min(1, w_risky))  # Clamp to [0, 1]
                weight_rf = 1.0 - w_risky
                final_weights = w_risky * tangency_weights
                strategy = f"CML: {weight_rf:.1%} Risk-Free + {w_risky:.1%} Tangency"
                
                print(f"\n📉 Target Return ({target_return:.2%}) < Tangency ({tangency_return:.2%})")
                print(f"   → Using Capital Market Line (CML)")
                print(f"   → Strategy: {strategy}")
        else:
            print(f"\n📈 Target Return ({target_return:.2%}) >= Tangency ({tangency_return:.2%})")
            print(f"   → Target exceeds max Sharpe; using 100% tangency portfolio")
            strategy = "100% Tangency (Target > Max Sharpe)"
    
    # Calculate dollar allocation
    dollar_amounts = final_weights * budget
    
    return {
        'tangency_weights': tangency_weights,
        'final_weights': final_weights,
        'weight_rf': weight_rf,
        'strategy': strategy,
        'tangency_return': tangency_return,
        'tangency_volatility': tangency_volatility,
        'tangency_sharpe': tangency_sharpe,
        'dollar_amounts': dollar_amounts
    }


def run_monte_carlo_scenarios(weights: np.ndarray, mu: np.ndarray, Sigma: np.ndarray, 
                               budget: float, risk_free: float, weight_rf: float,
                               days: int = 252, simulations: int = 10000) -> Dict:
    """
    Monte Carlo Simulation with Scenario Analysis
    Shows: Best 5%, 25th percentile, median, average, worst cases
    
    Mathematical Foundation:
    dS_t = μS_t dt + σS_t dW_t  (Geometric Brownian Motion)
    """
    print(f"\n🎲 Running Monte Carlo Simulation ({simulations:,} paths)...")
    print(f"   Mathematical Model: Geometric Brownian Motion (GBM)")
    print(f"   Time Horizon: {days} trading days (1 year)")
    print(f"   Portfolio: {100*(1-weight_rf):.1f}% Risky + {100*weight_rf:.1f}% Risk-Free")
    
    dt = 1.0 / days  # Daily time step
    
    # Calculate portfolio-level parameters (risky portion only)
    if np.sum(weights) > 0:
        risky_weights = weights / np.sum(weights)  # Normalize risky weights
        port_mu = float(np.dot(risky_weights, mu))
        port_var = float(np.dot(risky_weights, Sigma @ risky_weights))
        port_vol = np.sqrt(port_var)
    else:
        # 100% risk-free
        port_mu = risk_free
        port_vol = 0
        risky_weights = weights
    
    print(f"   Risky Portfolio Expected Return: {port_mu:.2%}")
    print(f"   Risky Portfolio Volatility: {port_vol:.2%}")
    
    final_values = []
    
    for _ in range(simulations):
        if port_vol > 0:
            # Generate random shocks (Wiener process increments)
            daily_shocks = np.random.normal(0, 1, days) * np.sqrt(dt)
            
            # GBM solution for risky portion
            daily_returns = (port_mu - 0.5 * port_var) * dt + port_vol * daily_shocks
            risky_cumulative = np.prod(1 + daily_returns)
        else:
            # 100% risk-free: deterministic growth
            risky_cumulative = np.exp(risk_free)
        
        # Combine risky + risk-free portions
        # Total return = w_rf * Rf + (1-w_rf) * R_risky
        total_return = weight_rf * np.exp(risk_free) + (1 - weight_rf) * risky_cumulative
        final_value = budget * total_return
        final_values.append(final_value)
    
    final_values = np.array(final_values)
    
    # Calculate Scenario Percentiles
    scenarios = {
        'best_5%': np.percentile(final_values, 95),
        'q3_75%': np.percentile(final_values, 75),
        'median_50%': np.percentile(final_values, 50),
        'q1_25%': np.percentile(final_values, 25),
        'worst_5%': np.percentile(final_values, 5),
        'mean': np.mean(final_values),
        'std': np.std(final_values),
        'prob_profit': np.sum(final_values > budget) / simulations,
        'all_values': final_values
    }
    
    # Calculate VaR and CVaR
    var_95 = budget - scenarios['worst_5%']
    worst_5pct = final_values[final_values <= scenarios['worst_5%']]
    cvar_95 = budget - np.mean(worst_5pct) if len(worst_5pct) > 0 else var_95
    
    print(f"\n{'='*60}")
    print(f"📊 MONTE CARLO SCENARIO ANALYSIS (1-Year Horizon)")
    print(f"{'='*60}")
    print(f"Simulations:    {simulations:,} paths")
    print(f"Initial Budget: ${budget:,.2f}")
    print(f"\n📈 UPSIDE SCENARIOS:")
    print(f"   Best 5% Case:     ${scenarios['best_5%']:>12,.2f}  (+{(scenarios['best_5%']/budget-1)*100:>6.1f}%)")
    print(f"   75th Percentile:  ${scenarios['q3_75%']:>12,.2f}  (+{(scenarios['q3_75%']/budget-1)*100:>6.1f}%)")
    print(f"\n📊 CENTRAL TENDENCY:")
    print(f"   Median (50%):     ${scenarios['median_50%']:>12,.2f}  (+{(scenarios['median_50%']/budget-1)*100:>6.1f}%)")
    print(f"   Mean (Average):   ${scenarios['mean']:>12,.2f}  (+{(scenarios['mean']/budget-1)*100:>6.1f}%)")
    print(f"\n📉 DOWNSIDE SCENARIOS:")
    print(f"   25th Percentile:  ${scenarios['q1_25%']:>12,.2f}  ({(scenarios['q1_25%']/budget-1)*100:>6.1f}%)")
    print(f"   Worst 5% Case:    ${scenarios['worst_5%']:>12,.2f}  ({(scenarios['worst_5%']/budget-1)*100:>6.1f}%)")
    print(f"\n⚠️  RISK METRICS:")
    print(f"   VaR (95%):        ${var_95:>12,.2f}  (Max loss in worst 5%)")
    print(f"   CVaR (95%):       ${cvar_95:>12,.2f}  (Avg loss in worst 5%)")
    print(f"   Probability Profit: {scenarios['prob_profit']:.2%}")
    print(f"   Return Std Dev:   {scenarios['std']/budget:.2%}")
    print(f"{'='*60}")
    
    # Text-based distribution histogram
    print(f"\n📊 RETURN DISTRIBUTION (Text Histogram):")
    print(f"{'='*60}")
    
    returns_pct = (final_values / budget - 1) * 100  # Convert to percentage
    bins = np.linspace(np.percentile(returns_pct, 1), np.percentile(returns_pct, 99), 20)
    hist, bin_edges = np.histogram(returns_pct, bins=bins)
    max_count = np.max(hist)
    
    for i in range(len(hist)):
        if hist[i] > 0:
            bar_len = int(40 * hist[i] / max_count)
            bin_mid = (bin_edges[i] + bin_edges[i+1]) / 2
            print(f"   {bin_mid:>+6.1f}% │ {'█' * bar_len} {hist[i]}")
    
    print(f"{'='*60}")
    
    return scenarios


def main():
    # Configuration
    stocks = sorted(['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA'])
    risk_free_rate = 0.04
    budget = 100000
    target_return = 0.12  # Optional: Set to None for 100% tangency
    market_ticker = '^GSPC'

    print(f"\n{'='*60}")
    print(f"🎯 TANGENCY PORTFOLIO WITH SCENARIO ANALYSIS")
    print(f"{'='*60}")
    print(f"Stocks: {', '.join(stocks)}")
    print(f"Budget: ${budget:,.0f}")
    print(f"Target Return: {target_return:.2%}" if target_return else "Target Return: None (100% Tangency)")
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

    # 6. Solve Tangency Portfolio (with optional CML allocation)
    print("\n🔄 Solving Tangency Portfolio...")
    portfolio = solve_tangency_portfolio(mu, Sigma, risk_free_rate, budget, stocks, target_return)
    
    # 7. Calculate Shares
    weights = portfolio['final_weights']
    weight_rf = portfolio['weight_rf']
    dollar_amounts = weights * budget
    shares = np.floor(dollar_amounts / prices).astype(int)
    invested_value = np.sum(shares * prices)
    cash_remaining = budget - invested_value + (weight_rf * budget - np.sum(weights * budget - shares * prices))
    
    # 8. Run Monte Carlo Scenario Analysis
    scenarios = run_monte_carlo_scenarios(
        weights=portfolio['tangency_weights'],  # Use pure tangency for simulation
        mu=mu, 
        Sigma=Sigma, 
        budget=budget,
        risk_free=risk_free_rate,
        weight_rf=weight_rf
    )
    
    # 9. Display Final Portfolio
    print(f"\n{'='*60}")
    print(f"📋 FINAL PORTFOLIO ALLOCATION")
    print(f"{'='*60}")
    print(f"Strategy: {portfolio['strategy']}")
    print(f"Risk-Free Proportion: {weight_rf:.2%}")
    print(f"Risky Asset Proportion: {1-weight_rf:.2%}")
    
    print(f"\n{'Stock':<10} {'Weight':<12} {'Shares':<10} {'Price':<12} {'Value':<12}")
    print("-" * 56)
    
    total_stock_value = 0
    for i, stock in enumerate(stocks):
        if weights[i] > 0.001 or shares[i] > 0:
            value = shares[i] * prices[i]
            total_stock_value += value
            print(f"{stock:<10} {weights[i]:>11.2%} {shares[i]:>8} ${prices[i]:>10.2f} ${value:>10,.2f}")
    
    print("-" * 56)
    print(f"Total Invested in Stocks: ${total_stock_value:,.2f}")
    print(f"Cash (Risk-Free Asset):   ${budget - total_stock_value:,.2f}")
    print(f"{'='*60}")
    
    # 10. Scenario Summary Table
    print(f"\n{'='*60}")
    print(f"📊 SCENARIO SUMMARY TABLE")
    print(f"{'='*60}")
    print(f"{'Scenario':<20} {'Portfolio Value':>15} {'Return vs Budget':>18}")
    print("-" * 56)
    
    scenario_labels = [
        ('Best 5% (95th pct)', 'best_5%'),
        ('75th Percentile', 'q3_75%'),
        ('Median (50%)', 'median_50%'),
        ('Mean (Average)', 'mean'),
        ('25th Percentile', 'q1_25%'),
        ('Worst 5% (5th pct)', 'worst_5%'),
    ]
    
    for label, key in scenario_labels:
        value = scenarios[key]
        ret_pct = (value / budget - 1) * 100
        sign = '+' if ret_pct >= 0 else ''
        print(f"{label:<20} ${value:>14,.2f} {sign}{ret_pct:>16.1f}%")
    
    print("-" * 56)
    print(f"Probability of Profit: {scenarios['prob_profit']:.2%}")
    print(f"VaR (95%): ${budget - scenarios['worst_5%']:,.2f}")
    print(f"{'='*60}")
    
    # 11. Mathematical Finance Summary
    print(f"\n{'='*60}")
    print(f"📐 MATHEMATICAL FOUNDATIONS")
    print(f"{'='*60}")
    print(f"1. Expected Returns: CAPM")
    print(f"   E[Rᵢ] = Rf + βᵢ × (E[Rm] - Rf)")
    print(f"\n2. Optimization: Mean-Variance (Markowitz)")
    print(f"   max (w'μ - Rf) / √(w'Σw)  s.t.  w'1 = 1, w ≥ 0")
    print(f"\n3. Capital Market Line (CML)")
    print(f"   E[Rp] = Rf + [(E[Rt]-Rf)/σt] × σp")
    print(f"\n4. Risk Simulation: Geometric Brownian Motion")
    print(f"   dSₜ = μSₜdt + σSₜdWₜ")
    print(f"\n5. Scenario Analysis: Percentile-based outcomes")
    print(f"   Best 5% = 95th percentile of simulated returns")
    print(f"   Worst 5% = 5th percentile (VaR threshold)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
