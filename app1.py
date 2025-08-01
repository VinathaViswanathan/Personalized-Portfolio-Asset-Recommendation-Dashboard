#streamlit run /Users/vinatha/PycharmProjects/PythonProject/PythonProject1/app1.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pypfopt import EfficientFrontier, risk_models, expected_returns, plotting
from collections import OrderedDict
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from pypfopt.exceptions import OptimizationError


# 1. DATA LOADING & PREPROCESSING
def load_data():
    asset_df = pd.read_csv("FAR-Trans-Data/asset_information.csv")
    price_df = pd.read_csv("FAR-Trans-Data/close_prices.csv")
    customer_df = pd.read_csv("FAR-Trans-Data/customer_information.csv")
    transactions_df = pd.read_csv("FAR-Trans-Data/transactions.csv")
    return asset_df, price_df, customer_df, transactions_df

def get_user_assets(customer_id, transactions_df):
    return transactions_df[transactions_df['customerID'] == customer_id]['ISIN'].unique().tolist()

def build_price_matrix(selected_assets, price_df):
    df = price_df[price_df['ISIN'].isin(selected_assets)]
    matrix = df.pivot(index='timestamp', columns='ISIN', values='closePrice').fillna(method='ffill').dropna()
    return matrix

def suggest_portfolio(customer_id, transactions_df, price_df, user_target_vol, st=None):
    user_assets = get_user_assets(customer_id, transactions_df)
    if len(user_assets) < 2:
        return None, None, None, None, True

    price_matrix = build_price_matrix(user_assets, price_df)
    mu = expected_returns.mean_historical_return(price_matrix)
    S = risk_models.sample_cov(price_matrix)
    ef = EfficientFrontier(mu, S)

    # Get minimum achievable volatility
    try:
        min_vol = ef.min_volatility()
        if isinstance(min_vol, dict) or isinstance(min_vol, OrderedDict):
            min_vol = list(min_vol.values())[0]
    except OptimizationError as e:
        if st:
            st.error(f"Optimization failed: {e}")
        return None, None, price_matrix, None, True

    target_vol = max(user_target_vol, min_vol)

    if st and min_vol > user_target_vol:
        st.warning(f"Minimum achievable volatility is {min_vol:.2%}. Using this instead.")

    # Try target volatility optimization
    try:
        ef1 = EfficientFrontier(mu, S)
        ef1.efficient_risk(target_volatility=target_vol)
        cleaned_weights = ef1.clean_weights()
        perf = ef1.portfolio_performance(verbose=False)
    except (OptimizationError, ValueError) as e1:
        if st:
            st.warning(f"Portfolio optimization failed at target_vol {target_vol:.2%}: {e1}")

        # Fallback to minimum volatility optimization
        try:
            ef2 = EfficientFrontier(mu, S)
            ef2.min_volatility()
            cleaned_weights = ef2.clean_weights()
            perf = ef2.portfolio_performance(verbose=False)
        except (OptimizationError, ValueError) as e2:
            if st:
                st.error(f"Fallback optimization also failed: {e2}")
            return None, None, price_matrix, min_vol, True

    # Convert weights to float
    weights_float = {}
    for asset, weight in cleaned_weights.items():
        try:
            weights_float[asset] = float(weight)
        except:
            weights_float[asset] = 0.0

    return weights_float, perf, price_matrix, min_vol, False

def recommend_popular_assets(customer_id, transactions_df, top_n=3):
    owned = set(get_user_assets(customer_id, transactions_df))
    popular_assets = transactions_df['ISIN'].value_counts().index
    recs = [isin for isin in popular_assets if isin not in owned][:top_n]
    return recs

def filter_customers(customer_df, customer_type=None, investment_capacity=None, num_assets=None):
    df = customer_df.copy()
    if num_assets is not None:
        df = df[df['num_assets'] == num_assets]
    if customer_type and customer_type != "All":
        df = df[df['customerType'] == customer_type]
    if investment_capacity and investment_capacity != "All":
        df = df[df['investmentCapacity'] == investment_capacity]
    return df

def simulate_portfolio_performance(price_matrix, weights):
    if price_matrix is None or weights is None:
        return None, None
    returns = price_matrix.pct_change().dropna()
    w = np.array([weights.get(col, 0.0) for col in price_matrix.columns])
    port_returns = returns.dot(w)
    perf = (1 + port_returns).cumprod() * 100
    return perf.index, perf.values


def fallback_recommendation(customer_id, transactions_df, asset_df, top_n=5):
    owned = set(get_user_assets(customer_id, transactions_df))
    customer_sector = asset_df[asset_df['ISIN'].isin(owned)]['sector'].dropna().unique()

    popular_assets = transactions_df['ISIN'].value_counts().index.tolist()
    recommendations = []
    confidences = []
    reasons = []

    for isin in popular_assets:
        if isin in owned:
            continue
        asset_row = asset_df[asset_df['ISIN'] == isin]
        if asset_row.empty:
            continue
        asset = asset_row.iloc[0]

        reason = "Popular among many investors."
        if asset['sector'] in customer_sector:
            reason += " Matches your preferred sector."

        recommendations.append({
            'ISIN': asset['ISIN'],
            'assetName': asset['assetName'],
            'assetCategory': asset['assetCategory'],
            'sector': asset['sector'],
            'Confidence (%)': 80.0,
            'Reason': reason,
            'Recommendation Method': "Fallback (Popular + Sector Match)"
        })

        if len(recommendations) >= top_n:
            break

    rec_df = pd.DataFrame(recommendations)

    method_used = "Fallback (Popular + Sector Match)"
    method_reason = "Collaborative filtering was skipped due to insufficient user data. Popular assets and sector alignment used instead."

    return rec_df, method_used, method_reason



def hybrid_recommend_assets(customer_id, transactions_df, asset_df, top_n=5, popularity_threshold=2):
    interaction = pd.crosstab(transactions_df['customerID'], transactions_df['ISIN'])

    # Use fallback if not enough data
    if customer_id not in interaction.index or interaction.shape[0] < 5 or interaction.shape[1] < 10:
        fallback_df, method, reason = fallback_recommendation(customer_id, transactions_df, asset_df, top_n)
        return fallback_df, method, reason

    sparse_matrix = csr_matrix(interaction.values)
    n_users, n_assets = sparse_matrix.shape
    n_components = min(15, n_users - 1, n_assets - 1)

    try:
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        latent_matrix = svd.fit_transform(sparse_matrix)
    except Exception:
        fallback_df, method, reason = fallback_recommendation(customer_id, transactions_df, asset_df, top_n)
        return fallback_df, method, reason

    user_idx = interaction.index.get_loc(customer_id)
    user_vector = latent_matrix[user_idx]
    similarities = cosine_similarity(latent_matrix, user_vector.reshape(1, -1)).flatten()
    similarities[user_idx] = 0  # exclude self
    top_peers_idx = similarities.argsort()[::-1][:10]
    peer_ids = interaction.index[top_peers_idx]

    user_assets = set(interaction.columns[interaction.iloc[user_idx] > 0])
    user_sectors = asset_df[asset_df['ISIN'].isin(user_assets)]['sector'].dropna().unique()

    asset_scores = svd.components_.T.dot(user_vector)
    min_score = np.min(asset_scores)
    max_score = np.max(asset_scores)

    results = []

    for idx in np.argsort(asset_scores)[::-1]:
        isin = interaction.columns[idx]
        if isin in user_assets:
            continue

        asset_row = asset_df[asset_df['ISIN'] == isin]
        if asset_row.empty:
            continue
        asset = asset_row.iloc[0]

        reason_parts = []
        if asset['sector'] in user_sectors:
            reason_parts.append("Matches your preferred sector.")
        peer_owners = interaction.loc[peer_ids, isin].sum()
        if peer_owners > popularity_threshold:
            reason_parts.append("Popular among similar investors.")
        if not reason_parts:
            reason_parts.append("Diversifies your current holdings.")

        confidence = round(100 * (asset_scores[idx] - min_score) / (max_score - min_score + 1e-8), 2)

        results.append({
            'ISIN': asset['ISIN'],
            'assetName': asset['assetName'],
            'assetCategory': asset['assetCategory'],
            'sector': asset['sector'],
            'Confidence (%)': confidence,
            'Reason': " ".join(reason_parts),
            'Recommendation Method': "Collaborative Filtering (SVD)"
        })

        if len(results) >= top_n:
            break

    rec_df = pd.DataFrame(results)

    method_used = "Collaborative Filtering (SVD)"
    method_reason = "Collaborative filtering based on investor similarity using TruncatedSVD was successfully applied."

    return rec_df, method_used, method_reason, top_peers_idx, latent_matrix, interaction


def show_collab_filtering_explanation(customer_id, transactions_df, asset_df, top_peers_idx, interaction, latent_matrix):
    user_idx = interaction.index.get_loc(customer_id)
    peers_idx = top_peers_idx
    selected_users = [user_idx] + list(peers_idx)
    subset = interaction.iloc[selected_users, :]

    # Prepare similarities for bar chart
    user_vector = latent_matrix[user_idx]
    similarities = cosine_similarity(latent_matrix, user_vector.reshape(1, -1)).flatten()
    similarities[user_idx] = 0
    top_similarities = similarities[peers_idx]

    # Prepare sector data
    user_assets = set(interaction.columns[interaction.iloc[user_idx] > 0])
    user_sectors = asset_df[asset_df['ISIN'].isin(user_assets)]['sector'].value_counts()

    peers_assets = interaction.iloc[peers_idx].sum(axis=0)
    peers_top_assets = peers_assets[peers_assets > 0].index
    peers_sectors = asset_df[asset_df['ISIN'].isin(peers_top_assets)]['sector'].value_counts()

    sector_df = pd.DataFrame({
        'User': user_sectors,
        'Peers': peers_sectors
    }).fillna(0)

    # Create 3 columns
    col1, col2 = st.columns([1.08, 1])

    # 2. Similarity Scores Bar Chart
    with col1:
        st.subheader("Similarity Scores")
        sim_df = pd.DataFrame({
            'CustomerID': interaction.index[peers_idx],
            'Similarity': top_similarities
        }).sort_values('Similarity', ascending=False)
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        sns.barplot(data=sim_df, x='CustomerID', y='Similarity', palette="Blues_d", ax=ax2)
        ax2.set_xlabel("Peer Customer IDs")
        ax2.set_ylabel("Cosine Similarity")
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig2)

    # 3. Sector Distribution Bar Chart
    with col2:
        st.subheader("Sector Distribution")
        fig3, ax3 = plt.subplots(figsize=(5, 4))
        sector_df.plot(kind='bar', ax=ax3)
        ax3.set_ylabel("Number of Assets")
        ax3.set_xlabel("Sector")
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig3)

    col3, col4, col5 = st.columns([1, 5, 1])
    # 1. Interaction Matrix Heatmap
    with col4:
        st.subheader("User-Peer Interaction")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(subset.replace(0, np.nan), cmap="cubehelix", cbar=True, ax=ax)
        ax.set_xlabel("Assets (ISIN)")
        ax.set_ylabel("Users (CustomerID)")
        st.pyplot(fig)

# --- FAQ Help Button Logic ---
if 'show_faq' not in st.session_state:
    st.session_state['show_faq'] = False
if 'show_answer' not in st.session_state:
    st.session_state['show_answer'] = False

def toggle_faq():
    st.session_state['show_faq'] = not st.session_state['show_faq']
    if not st.session_state['show_faq']:
        st.session_state['show_answer'] = False

# --- Streamlit App ---
def main():
    st.set_page_config(layout="wide")
    st.title("Personalized Portfolio & Asset Recommendation Dashboard")
    st.markdown("""
        This dashboard helps you optimize your investment portfolio and discover new asset recommendations based on your selected risk level, investment profile, and past behavior. 
        You can adjust your target volatility, view your portfolio allocation, analyze risk, and get transparent explanations for each recommendation.
        """)

    st.markdown("---")
    asset_df, price_df, customer_df, transactions_df = load_data()
    transactions_df = transactions_df[transactions_df['transactionType'] == "Buy"]

    # --- Number of Assets per Customer ---
    assets_per_customer = (
        transactions_df.groupby('customerID')['ISIN']
        .nunique()
        .reset_index()
        .rename(columns={'ISIN': 'num_assets'})
    )
    customer_df = customer_df.merge(assets_per_customer, on='customerID', how='left')
    customer_df['num_assets'] = customer_df['num_assets'].fillna(0).astype(int)

    # Sidebar filters
    st.sidebar.header("Customer Filters")
    asset_counts = sorted(customer_df['num_assets'].unique())
    selected_num_assets = st.sidebar.selectbox("Number of Assets Owned", asset_counts)

    # Dynamically filter customer types and investment capacities based on asset count
    filtered_df = customer_df[customer_df['num_assets'] == selected_num_assets]
    customer_types = ["All"] + sorted(filtered_df['customerType'].dropna().unique().tolist())
    investment_caps = ["All"] + sorted(filtered_df['investmentCapacity'].dropna().unique().tolist())
    customer_type = st.sidebar.selectbox("Customer Type", customer_types)
    investment_capacity = st.sidebar.selectbox("Investment Capacity", investment_caps)

    filtered_customers = filter_customers(filtered_df, customer_type, investment_capacity, num_assets=selected_num_assets)
    customer_list = filtered_customers['customerID'].unique()
    customer_id = st.sidebar.selectbox("Select Customer", customer_list)

    risk_levels = ["Conservative", "Balanced", "Aggressive"]
    risk_targets = {"Conservative": 0.10, "Balanced": 0.15, "Aggressive": 0.25}
    risk_level = st.sidebar.selectbox("Risk Level", risk_levels)
    desired_vol = risk_targets.get(risk_level, 0.15)

    user_assets = get_user_assets(customer_id, transactions_df)
    min_vol_display = 0.01
    price_matrix = None

    if len(user_assets) >= 2:
        price_matrix = build_price_matrix(user_assets, price_df)
        mu = expected_returns.mean_historical_return(price_matrix)
        S = risk_models.sample_cov(price_matrix)
        ef_tmp = EfficientFrontier(mu, S)
        min_vol_tmp = ef_tmp.min_volatility()
        if isinstance(min_vol_tmp, dict) or isinstance(min_vol_tmp, OrderedDict):
            min_vol_tmp = list(min_vol_tmp.values())[0]
        min_vol_display = float(min_vol_tmp)
        st.sidebar.info(f"Minimum achievable volatility for your assets: {min_vol_display:.2%}")
        user_target_vol = st.sidebar.slider(
            "Set your target volatility (%)",
            min_value=int(min_vol_display*100),
            max_value=100,
            value=int(desired_vol*100),
            step=1
        ) / 100
    else:
        user_target_vol = desired_vol

    if st.sidebar.checkbox("Show asset volatility and correlation") and price_matrix is not None:
        col1, col2 = st.columns([2.7, 2.04])
        with col1:
            st.write("Asset Volatility Distribution:")
            asset_vols = price_matrix.pct_change().std()
            vol_pct = (asset_vols * 100).round(2)
            fig, ax = plt.subplots(figsize=(8, 4))
            bars = ax.bar(vol_pct.index, vol_pct.values, color='skyblue')
            ax.set_ylabel('Volatility (%)')
            ax.set_title('Asset Volatility (Std Dev of Returns)')
            plt.xticks(rotation=45)
            for bar in bars:
                height = bar.get_height()
                ax.annotate(
                    f'{height:.2f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=7,
                )
            st.pyplot(fig)
        with col2:
            st.write("Asset Correlation Matrix:")
            corr_matrix = price_matrix.pct_change().corr().round(2)
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax, annot_kws={"size": 6})
            ax.set_title('Asset Correlation Matrix Heatmap')
            st.pyplot(fig)

    st.markdown("---")

    if st.sidebar.checkbox("Show Investment Type Insights"):
        st.header("Investment Type Insights Across All Assets")
        col1, col2, col3= st.columns(3)

        user_asset_df = asset_df[asset_df['ISIN'].isin(user_assets)]
        category_counts = user_asset_df['assetCategory'].value_counts()

        with col1:
            category_counts = user_asset_df['assetCategory'].value_counts()
            category_counts_df = category_counts.rename_axis('Investment Category').reset_index(
                name='Number of Assets Owned')
            st.subheader("Your Portfolio: Distribution by Investment Category")
            fig1, ax1 = plt.subplots()
            ax1.pie(category_counts, labels=category_counts.index, autopct='%1.1f%%', startangle=90)
            ax1.axis('equal')
            st.pyplot(fig1)
            st.write(category_counts_df)

        user_transactions = transactions_df[transactions_df['customerID'] == customer_id]
        merged = user_transactions.merge(asset_df[['ISIN', 'assetCategory']], on='ISIN', how='left')

        with col2:
            type_counts = merged['assetCategory'].value_counts()
            type_counts_df = type_counts.rename_axis('Investment Category').reset_index(
                name='Number of Buy Transactions')
            st.subheader("Your Transactions: Popularity by Investment Type")
            fig2, ax2 = plt.subplots()
            ax2.bar(type_counts.index, type_counts.values, color='skyblue')
            ax2.set_ylabel('Number of Buy Transactions')
            ax2.set_title('Buy Transactions per Investment Type (You)')
            plt.xticks(rotation=30)
            st.pyplot(fig2)
            st.write(type_counts_df)

        with col3:
            sector_counts = user_asset_df['sector'].value_counts()
            sector_counts_df = sector_counts.rename_axis('Sector').reset_index(name='Number of Assets Owned')
            st.subheader("Your Portfolio: Distribution by Sector")
            fig3, ax3 = plt.subplots()
            ax3.bar(sector_counts.index, sector_counts.values, color='lightgreen')
            ax3.set_ylabel('Number of Assets Owned')
            ax3.set_title('Assets per Sector (You)')
            plt.xticks(rotation=30)
            st.pyplot(fig3)
            st.write(sector_counts_df)

    if len(user_assets) < 2:
        st.warning("Not enough assets in portfolio for optimization. Suggesting popular assets to diversify:")
        #recs = recommend_popular_assets(customer_id, transactions_df, top_n=5)
        #rec_details = asset_df[asset_df['ISIN'].isin(recs)][['ISIN', 'assetName', 'assetCategory', 'sector']]
        #st.dataframe(rec_details)
    else:
        st.write(f"Customer {customer_id} owns {len(user_assets)} unique assets.")

    weights_float, perf, price_matrix, min_vol, opt_failed = suggest_portfolio(
        customer_id, transactions_df, price_df, user_target_vol, st=st
    )

    if opt_failed:
        st.error("Portfolio optimization failed. Cannot proceed with recommendations.")
        fallback_df, method, reason = fallback_recommendation(customer_id, transactions_df, asset_df, 5)
        st.dataframe(fallback_df)
        return  # or skip portfolio-dependent logic

    if st.sidebar.checkbox('Show Portfolio Weights, Risk and Return') and weights_float and perf:
        st.markdown("---")
        st.subheader("📊 Portfolio Optimization Visualizations")

        col1, col2 = st.columns([1.07, 1])

        with col1:
            st.markdown("#### Optimized Portfolio Weights")
            weights_series = pd.Series(weights_float).sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.bar(weights_series.index, weights_series.values, color='skyblue')
            ax.set_ylabel('Weight')
            ax.set_title('Asset Allocation')
            plt.xticks(rotation=45)
            st.pyplot(fig)

        with col2:
            exp_return, volatility, sharpe = perf
            st.markdown("#### Risk vs Return")
            st.markdown(f"**Expected Annual Return**: `{exp_return:.2%}`")
            st.markdown(f"**Annual Volatility**: `{volatility:.2%}`")
            st.markdown(f"**Sharpe Ratio**: `{sharpe:.2f}`")

            fig2, ax2 = plt.subplots(figsize=(5, 4))
            ax2.scatter(volatility, exp_return, color='green', s=100)
            ax2.set_xlabel('Volatility (Risk)')
            ax2.set_ylabel('Expected Return')
            ax2.set_title('Optimized Portfolio Position')
            ax2.grid(True)
            st.pyplot(fig2)

    if st.sidebar.checkbox("Show Interaction Matrix Sparsity"):
        st.markdown("---")
        st.subheader("User-Asset Interaction Matrix (Full View)")

        # Build full interaction matrix (user x asset)
        interaction = pd.crosstab(transactions_df['customerID'], transactions_df['ISIN'])

        # Calculate sparsity
        n_users, n_assets = interaction.shape
        total_entries = n_users * n_assets
        non_zero_entries = interaction.values.sum()
        sparsity = 1.0 - (non_zero_entries / total_entries)

        # Display matrix shape and sparsity
        st.markdown(f"**Matrix Shape:** `{interaction.shape}`")
        st.markdown(f"**Sparsity:** `{sparsity:.2%}` (percentage of empty entries)")

        # Convert to binary (1 if user has asset, 0 otherwise)
        binary_matrix = interaction.astype(bool).astype(int)

        # Plot heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(binary_matrix, cmap="Greys", cbar=False, xticklabels=False, yticklabels=False)
        ax.set_xlabel("Assets (ISINs)", fontsize=12)
        ax.set_ylabel("Customers", fontsize=12)
        ax.set_title("User-Asset Interaction Matrix (1 = Owned, 0 = Not Owned)", fontsize=14)
        st.pyplot(fig)

    if not opt_failed:
        if st.sidebar.button("Get Recommendation"):
            st.markdown("---")
            st.subheader("🔍 Asset Recommendations")

            rec_df, method_used, method_reason, top_peers_idx, latent_matrix, interaction = hybrid_recommend_assets(
                customer_id, transactions_df, asset_df, top_n=5)

            st.write(f"Recommendation method used: {method_used}")
            st.write(method_reason)

            if method_used == "Collaborative Filtering (SVD)":
                show_collab_filtering_explanation(customer_id, transactions_df, asset_df, top_peers_idx, interaction,
                                                  latent_matrix)

            #st.dataframe(rec_df)

            if not rec_df.empty:
                st.dataframe(rec_df.rename(columns={
                    'assetName': 'Asset Name',
                    'assetCategory': 'Asset Category',
                    'sector': 'Sector',
                    'Confidence (%)': 'Confidence (%)',
                    'Reason': 'Why This Asset',
                    'Recommendation Method': 'Method'
                }))
            else:
                st.warning("No suitable recommendations found.")
    else:
        st.warning(
            "Cannot generate recommendations because portfolio optimization failed. Please increase your target volatility.")

    # --- FAQ Help Button Logic ---
    if 'show_faq' not in st.session_state:
        st.session_state['show_faq'] = False
    if 'show_answer' not in st.session_state:
        st.session_state['show_answer'] = False

    def toggle_faq():
        st.session_state['show_faq'] = not st.session_state['show_faq']
        if not st.session_state['show_faq']:
            st.session_state['show_answer'] = False

    def toggle_answer():
        st.session_state['show_answer'] = not st.session_state['show_answer']

    col1, col2 = st.columns([9, 1])
    with col2:
        if st.button("FAQ Help", key="faq_button_main", use_container_width=True, on_click=toggle_faq):
            pass

    if st.session_state['show_faq']:
        if st.button("What is target volatility?", key="faq_question_main", on_click=toggle_answer):
            pass

    if st.session_state['show_answer']:
        st.markdown("---")
        st.write("**What is target volatility?**")
        st.info(
            "This is the level of risk (how much your portfolio value can fluctuate) you want in your portfolio. "
            "If your assets are volatile or highly correlated, you may not be able to achieve a low target volatility. "
            "The dashboard will guide you to set a realistic target based on your assets."
        )
        if st.button("Cancel", key="faq_cancel_main", on_click=toggle_answer):
            pass

if __name__ == '__main__':
    main()

