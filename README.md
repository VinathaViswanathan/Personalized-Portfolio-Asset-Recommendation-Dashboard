# Personalized Portfolio & Asset Recommendation Dashboard


## Overview
This project is a Streamlit-based financial analytics dashboard that empowers investors with portfolio optimization and personalized asset recommendations. The application integrates mean-variance optimization, collaborative filtering, and heuristic fallback strategies to deliver data-driven insights. It supports filtering by customer profiles, simulates risk-return tradeoffs, and visualizes investment behavior.

The app combines techniques from quantitative finance, machine learning, and recommender systems, offering both transparency and interactivity.

## Dataset
The app uses 4 key CSV datasets located in the FAR-Trans-Data/ directory:

asset_information.csv – Metadata including sector and category for each asset.
close_prices.csv – Time-series closing prices by ISIN.
customer_information.csv – Customer profiles (type, capacity).
transactions.csv – Buy transaction records linking customers to assets.

These datasets simulate a structured financial system that allows historical analysis and personalized recommendations.


## Key Features
### Portfolio Optimization
Uses mean historical returns and sample covariance to optimize weights.
Supports target volatility optimization using pypfopt.EfficientFrontier.
Falls back to minimum volatility portfolios if optimization fails.

Visualizes:

Asset allocation weights
Expected return vs volatility (risk-return scatter)
Asset volatilities and correlation matrix

## Asset Recommendation Engine
### 1. Collaborative Filtering (SVD-based)
Leverages Truncated SVD on customer-asset interaction matrix
Computes cosine similarity to recommend assets based on similar investors
Provides confidence score and qualitative reasoning (e.g., peer ownership, sector match)

### 2. Fallback Strategy
Recommends popular assets not held by user
Filters for sector alignment to ensure contextual relevance
Used when user has too few assets or data is sparse

### 3. Hybrid Recommendation
Automatically switches between SVD-based and fallback depending on data availability
