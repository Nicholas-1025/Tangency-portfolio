# Tangency-portfolio
# 📊 Portfolio Optimization with CAPM & Monte Carlo Simulation

A Python-based portfolio optimization tool using Modern Portfolio Theory (MPT), 
CAPM for expected returns, and Monte Carlo simulation for risk analysis.

## 🎯 Features

- ✅ **CAPM Expected Returns** - Theoretically-grounded return estimation
- ✅ **Tangency Portfolio** - Maximum Sharpe Ratio optimization
- ✅ **Capital Market Line (CML)** - Risk-free asset allocation
- ✅ **Monte Carlo Simulation** - 10,000+ path scenario analysis
- ✅ **Risk Metrics** - VaR, CVaR, percentiles
- ✅ **Data Cleanup** - Automatic memory management

## 📐 Mathematical Foundations

1. **CAPM**: E[Rᵢ] = Rf + βᵢ × (E[Rm] - Rf)
2. **Mean-Variance Optimization**: max (w'μ - Rf) / √(w'Σw)
3. **Geometric Brownian Motion**: dSₜ = μSₜdt + σSₜdWₜ

## 📦 Installation

```bash
pip install yfinance pandas numpy cvxpy
