# Personalized Portfolio & Asset Recommendation Dashboard


## Overview
This project is a Streamlit-based financial analytics dashboard that empowers investors with portfolio optimization and personalized asset recommendations. The application integrates mean-variance optimization, collaborative filtering, and heuristic fallback strategies to deliver data-driven insights. It supports filtering by customer profiles, simulates risk-return tradeoffs, and visualizes investment behavior.

The app combines techniques from quantitative finance, machine learning, and recommender systems, offering both transparency and interactivity.

<img width="2940" height="1596" alt="Image" src="https://github.com/user-attachments/assets/cc6a6209-fee1-4728-8192-664bc850cbb7" />


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
<img width="1990" height="498" alt="Image" src="https://github.com/user-attachments/assets/17b1945d-e68a-4f7d-ab78-2ee0db4e8de5" />
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


## Implementation Details
### Portfolio Optimization
Library: pypfopt
Methods: efficient_risk, min_volatility
Handles edge cases (e.g., low asset count, infeasible volatility targets)

### Recommender System
SVD via sklearn.decomposition.TruncatedSVD
Similarity: sklearn.metrics.pairwise.cosine_similarity
Fallback logic ensures robust output even for cold-start users

### Front-End: Streamlit
Fully interactive layout with sidebar filtering
Context-aware visualizations, including heatmaps and pie/bar charts
Embedded FAQ logic to assist users


## Requirements
Ensure the following packages are installed (via requirements.txt or pip install):

```
streamlit
pandas
numpy
matplotlib
seaborn
scikit-learn
scipy
PyPortfolioOpt
```

## Applcation Startup

```
stream run <location of app1.py>
```
## Future Developments
1. Real-time market data integration via financial APIs
2. Transaction simulation and automated portfolio rebalancing
3. Integration of advanced risk models (e.g., Monte Carlo, Black-Litterman, VaR)
4. Enhanced recommender systems using neural collaborative filtering and autoencoders
5. User authentication and session management with persistent dashboards
6. Mobile and multi-device UI optimization
7. Containerization and deployment using Docker and cloud platforms
8. Explainable AI features for transparent recommendation insights
