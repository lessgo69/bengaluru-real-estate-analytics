import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# 1. Page Configuration
st.set_page_config(
    page_title="Bengaluru Real Estate Analytics",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Data Loading & Caching
@st.cache_data
def load_data():
    data = pd.read_excel('Bengaluru_House_Data_Cleaned.xlsx')
    # Remove top 1% extreme price_per_sqft outliers so charts don't get squished
    q99 = data['price_per_sqft'].quantile(0.99)
    data = data[data['price_per_sqft'] <= q99].reset_index(drop=True)
    return data

df = load_data()

# 3. Sidebar Filters
st.sidebar.header("🔍 Filter Properties")

all_locations = sorted([loc for loc in df['location'].unique() if loc != 'Other'])
locations_list = ['All Locations'] + all_locations
selected_loc = st.sidebar.selectbox("Select Micro-Market", locations_list)

bhk_options = sorted(df['bhk'].unique())
selected_bhk = st.sidebar.multiselect(
    "Select Bedroom Configuration (BHK)", 
    options=bhk_options, 
    default=[1, 2, 3, 4]
)

min_price = float(df['price'].min())
max_slider = float(df['price'].quantile(0.98)) # Cap slider at 98th percentile (~400 Lakhs) for better chart detail
price_range = st.sidebar.slider(
    "Budget Range (₹ Lakhs)", 
    min_value=min_price, 
    max_value=max_slider, 
    value=(min_price, 250.0), 
    step=5.0
)

# Apply Filters
filtered_df = df.copy()

if selected_loc != 'All Locations':
    filtered_df = filtered_df[filtered_df['location'] == selected_loc]

if selected_bhk:
    filtered_df = filtered_df[filtered_df['bhk'].isin(selected_bhk)]

filtered_df = filtered_df[
    (filtered_df['price'] >= price_range[0]) & 
    (filtered_df['price'] <= price_range[1])
]

# 4. Header & Executive KPIs
st.title("🏢 Bengaluru Real Estate Property Price Analytics")
st.markdown("*Interactive Business Intelligence Dashboard for Property Valuation & Market Dynamics*")
st.markdown("---")

col1, col2, col3, col4 = st.columns(4)

total_listings = len(filtered_df)
median_price = filtered_df['price'].median() if total_listings > 0 else 0
median_pps = filtered_df['price_per_sqft'].median() if total_listings > 0 else 0
avg_sqft = filtered_df['total_sqft'].mean() if total_listings > 0 else 0

col1.metric("Total Listings Available", f"{total_listings:,}")
col2.metric("Median Price", f"₹ {median_price:.1f} Lakhs")
col3.metric("Median Price / Sq. Ft.", f"₹ {median_pps:,.0f}")
col4.metric("Average Built-up Area", f"{avg_sqft:,.0f} sq. ft.")

st.markdown("---")

if total_listings == 0:
    st.warning("No properties match the selected filter criteria. Please broaden your selections.")
    st.stop()

# 5. Core Visual Analytics (Tabs)
tab1, tab2, tab3 = st.tabs([
    "📊 Market Overview & Pricing", 
    "📍 Micro-Market Analysis", 
    "💡 Property Value Estimator"
])

with tab1:
    st.subheader("Property Size vs. Acquisition Price")
    # Convert bhk to string so colors appear as distinct categories
    scatter_df = filtered_df[filtered_df['total_sqft'] <= 5000].copy()
    scatter_df['BHK Tier'] = scatter_df['bhk'].astype(str) + " BHK"
    
    fig_scatter = px.scatter(
        scatter_df, 
        x="total_sqft", 
        y="price", 
        color="BHK Tier",
        hover_data=["location", "bath", "balcony"],
        title="Total Built-up Area vs. Price (₹ Lakhs)",
        labels={"total_sqft": "Square Footage", "price": "Price (₹ Lakhs)"},
        opacity=0.7
    )
    fig_scatter.update_layout(height=450)
    st.plotly_chart(fig_scatter, width="stretch")

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Price Distribution")
        fig_hist = px.histogram(
            filtered_df, 
            x="price", 
            nbins=35, 
            marginal="box",
            title="Distribution of Property Prices (₹ Lakhs)",
            labels={"price": "Price (₹ Lakhs)", "count": "Number of Properties"},
            color_discrete_sequence=['#2b5c8f']
        )
        fig_hist.update_layout(height=400, bargap=0.05)
        st.plotly_chart(fig_hist, width="stretch")

    with col_b:
        st.subheader("Valuation by Room Configuration")
        box_df = filtered_df.copy()
        box_df['BHK'] = box_df['bhk'].astype(str) + " BHK"
        fig_box = px.box(
            box_df, 
            x="BHK", 
            y="price_per_sqft",
            title="Price per Sq. Ft. Distribution across BHK Tiers", 
            labels={"price_per_sqft": "Price per Sq. Ft. (₹)", "BHK": "Bedroom Configuration"},
            color="BHK",
            points="outliers"
        )
        # Zoom Y-axis to 15,000 so boxes are large and readable
        fig_box.update_layout(height=400, showlegend=False, yaxis=dict(range=[1500, 16000]))
        st.plotly_chart(fig_box, width="stretch")

with tab2:
    st.subheader("Top 15 Micro-Markets by Valuation (₹ / Sq. Ft.)")

    loc_metrics = df[df['location'] != 'Other'].groupby('location').agg(
        median_pps=('price_per_sqft', 'median'),
        listing_count=('price', 'count')
    ).reset_index()
    loc_metrics = loc_metrics[loc_metrics['listing_count'] >= 15].sort_values(
        by='median_pps', ascending=False
    ).head(15)

    fig_bar = px.bar(
        loc_metrics, 
        x='median_pps', 
        y='location', 
        orientation='h', 
        color='median_pps',
        color_continuous_scale='Bluyl', 
        title="Ranked Top 15 Micro-Markets by Median Price per Sq. Ft.",
        labels={'median_pps': 'Median Price / Sq. Ft. (₹)', 'location': 'Location'}
    )
    fig_bar.update_layout(yaxis={'categoryorder': 'total ascending'}, height=500)
    st.plotly_chart(fig_bar, width="stretch")

    st.markdown(
        "**Business Insight:** Northern corridors (Hebbal) and prime IT corridors "
        "command higher per-square-foot valuation markups compared to Southern peripheral clusters."
    )

with tab3:
    st.subheader("Real Estate Price Predictor / Estimator")
    st.write("Estimate baseline market valuation for a property based on historical medians.")

    c1, c2, c3 = st.columns(3)
    with c1:
        pred_loc = st.selectbox("Property Location", all_locations, key="pred_loc")
    with c2:
        pred_sqft = st.number_input(
            "Total Square Footage", 
            min_value=300.0, 
            max_value=10000.0, 
            value=1200.0, 
            step=50.0
        )
    with c3:
        pred_bhk = st.selectbox("BHK Configuration", options=[1, 2, 3, 4, 5], index=1)

    loc_median_rate = df[df['location'] == pred_loc]['price_per_sqft'].median()
    if np.isnan(loc_median_rate):
        loc_median_rate = df['price_per_sqft'].median()

    estimated_price = (loc_median_rate * pred_sqft) / 100000.0

    st.success(
        f"### Estimated Valuation: **₹ {estimated_price:.2f} Lakhs**\n"
        f"*Based on a median rate of **₹ {loc_median_rate:,.0f} / sq. ft.** in {pred_loc}*"
    )

st.markdown("---")
st.caption("CA4 Capstone Project: Real Estate Property Price Analytics | Business Analytics")
