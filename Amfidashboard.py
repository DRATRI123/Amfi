import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime
from scipy import stats

# Page configuration
st.set_page_config(
    page_title="AMFI Mutual Fund Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data(file):
    """Load and cache the Excel data"""
    try:
        df = pd.read_excel(file, sheet_name='All_Data')
        # Create proper date column
        month_map = {'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
                     'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12}
        df['Month_Num'] = df['Month'].map(month_map)
        df['Date'] = pd.to_datetime(df['Year'].astype(int).astype(str) + '-' + 
                                     df['Month_Num'].astype(int).astype(str) + '-01')
        df = df.sort_values('Date')
        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

def calculate_advanced_metrics(df):
    """Calculate all advanced metrics"""
    df = df.sort_values(['Scheme_ID', 'Date'])
    
    # Growth metrics - AUM
    df['MoM_Growth_AUM'] = df.groupby('Scheme_ID')['AUM'].pct_change() * 100
    df['YoY_Growth_AUM'] = df.groupby('Scheme_ID')['AUM'].pct_change(12) * 100
    df['QoQ_Growth_AUM'] = df.groupby('Scheme_ID')['AUM'].pct_change(3) * 100
    df['6M_Growth_AUM'] = df.groupby('Scheme_ID')['AUM'].pct_change(6) * 100
    
    # Growth metrics - Folios
    df['MoM_Growth_Folios'] = df.groupby('Scheme_ID')['No_of_Folios'].pct_change() * 100
    df['YoY_Growth_Folios'] = df.groupby('Scheme_ID')['No_of_Folios'].pct_change(12) * 100
    df['QoQ_Growth_Folios'] = df.groupby('Scheme_ID')['No_of_Folios'].pct_change(3) * 100
    
    # Growth metrics - Net Inflow
    df['MoM_Growth_NetInflow'] = df.groupby('Scheme_ID')['Net_Inflow'].pct_change() * 100
    df['YoY_Growth_NetInflow'] = df.groupby('Scheme_ID')['Net_Inflow'].pct_change(12) * 100
    
    # Rolling averages (3-month, 6-month, 12-month)
    df['AUM_3M_Avg'] = df.groupby('Scheme_ID')['AUM'].transform(lambda x: x.rolling(3, min_periods=1).mean())
    df['AUM_6M_Avg'] = df.groupby('Scheme_ID')['AUM'].transform(lambda x: x.rolling(6, min_periods=1).mean())
    df['AUM_12M_Avg'] = df.groupby('Scheme_ID')['AUM'].transform(lambda x: x.rolling(12, min_periods=1).mean())
    
    df['NetInflow_3M_Avg'] = df.groupby('Scheme_ID')['Net_Inflow'].transform(lambda x: x.rolling(3, min_periods=1).mean())
    df['NetInflow_6M_Avg'] = df.groupby('Scheme_ID')['Net_Inflow'].transform(lambda x: x.rolling(6, min_periods=1).mean())
    
    df['Folios_3M_Avg'] = df.groupby('Scheme_ID')['No_of_Folios'].transform(lambda x: x.rolling(3, min_periods=1).mean())
    df['Folios_6M_Avg'] = df.groupby('Scheme_ID')['No_of_Folios'].transform(lambda x: x.rolling(6, min_periods=1).mean())
    
    # Volatility (standard deviation)
    df['AUM_Volatility_3M'] = df.groupby('Scheme_ID')['AUM'].transform(lambda x: x.rolling(3, min_periods=1).std())
    df['AUM_Volatility_6M'] = df.groupby('Scheme_ID')['AUM'].transform(lambda x: x.rolling(6, min_periods=1).std())
    df['AUM_Volatility_12M'] = df.groupby('Scheme_ID')['AUM'].transform(lambda x: x.rolling(12, min_periods=1).std())
    
    df['NetInflow_Volatility_3M'] = df.groupby('Scheme_ID')['Net_Inflow'].transform(lambda x: x.rolling(3, min_periods=1).std())
    df['NetInflow_Volatility_6M'] = df.groupby('Scheme_ID')['Net_Inflow'].transform(lambda x: x.rolling(6, min_periods=1).std())
    
    # Rolling sum metrics
    df['Net_Inflow_3M_Sum'] = df.groupby('Scheme_ID')['Net_Inflow'].transform(lambda x: x.rolling(3, min_periods=1).sum())
    df['Net_Inflow_6M_Sum'] = df.groupby('Scheme_ID')['Net_Inflow'].transform(lambda x: x.rolling(6, min_periods=1).sum())
    df['Net_Inflow_12M_Sum'] = df.groupby('Scheme_ID')['Net_Inflow'].transform(lambda x: x.rolling(12, min_periods=1).sum())
    
    df['Funds_Mobilized_3M_Sum'] = df.groupby('Scheme_ID')['Funds_Mobilized'].transform(lambda x: x.rolling(3, min_periods=1).sum())
    df['Funds_Mobilized_6M_Sum'] = df.groupby('Scheme_ID')['Funds_Mobilized'].transform(lambda x: x.rolling(6, min_periods=1).sum())
    
    df['Repurchase_3M_Sum'] = df.groupby('Scheme_ID')['Repurchase'].transform(lambda x: x.rolling(3, min_periods=1).sum())
    df['Repurchase_6M_Sum'] = df.groupby('Scheme_ID')['Repurchase'].transform(lambda x: x.rolling(6, min_periods=1).sum())
    
    # Cumulative metrics
    df['Cumulative_Net_Inflow'] = df.groupby('Scheme_ID')['Net_Inflow'].cumsum()
    df['Cumulative_Funds_Mobilized'] = df.groupby('Scheme_ID')['Funds_Mobilized'].cumsum()
    df['Cumulative_Repurchase'] = df.groupby('Scheme_ID')['Repurchase'].cumsum()
    
    # Market share
    df['Market_Share'] = df.groupby('Date')['AUM'].transform(lambda x: (x / x.sum() * 100))
    df['Market_Share_Change'] = df.groupby('Scheme_ID')['Market_Share'].diff()
    
    # Redemption and retention metrics
    df['Redemption_Ratio'] = (df['Repurchase'] / df['Funds_Mobilized'] * 100).replace([np.inf, -np.inf], 0)
    df['Retention_Rate'] = (100 - df['Redemption_Ratio']).clip(0, 100)
    df['Gross_Sales_Rate'] = (df['Funds_Mobilized'] / df['AUM'] * 100).replace([np.inf, -np.inf], 0)
    df['Redemption_Rate'] = (df['Repurchase'] / df['AUM'] * 100).replace([np.inf, -np.inf], 0)
    
    # Average AUM per folio
    df['AUM_Per_Folio'] = (df['AUM'] / df['No_of_Folios']).replace([np.inf, -np.inf], 0)
    df['AUM_Per_Folio_Growth'] = df.groupby('Scheme_ID')['AUM_Per_Folio'].pct_change() * 100
    
    # Net flow per folio
    df['NetInflow_Per_Folio'] = (df['Net_Inflow'] / df['No_of_Folios']).replace([np.inf, -np.inf], 0)
    
    # Rank within category by AUM
    df['Rank_in_Category'] = df.groupby(['Date', 'Category_ID'])['AUM'].rank(ascending=False, method='min')
    df['Rank_in_Subcategory'] = df.groupby(['Date', 'Subcategory_ID'])['AUM'].rank(ascending=False, method='min')
    
    # Percentile ranking
    df['AUM_Percentile'] = df.groupby('Date')['AUM'].rank(pct=True) * 100
    
    # Momentum indicators
    df['AUM_Momentum_3M'] = df.groupby('Scheme_ID')['AUM'].transform(lambda x: x.pct_change(3) * 100)
    df['AUM_Momentum_6M'] = df.groupby('Scheme_ID')['AUM'].transform(lambda x: x.pct_change(6) * 100)
    
    # Acceleration (rate of change of growth)
    df['Growth_Acceleration'] = df.groupby('Scheme_ID')['MoM_Growth_AUM'].diff()
    
    # Asset gathering efficiency
    df['Asset_Gathering_Ratio'] = (df['Net_Inflow'] / df['Funds_Mobilized'] * 100).replace([np.inf, -np.inf], 0)
    
    # Scheme size category
    df['Size_Category'] = pd.cut(df['AUM'], 
                                  bins=[0, 100, 500, 1000, 5000, float('inf')],
                                  labels=['Micro (<100Cr)', 'Small (100-500Cr)', 'Medium (500-1000Cr)', 
                                         'Large (1000-5000Cr)', 'Very Large (>5000Cr)'])
    
    # Flow consistency score (lower volatility = higher consistency)
    df['Flow_Consistency_Score'] = df.groupby('Scheme_ID')['NetInflow_Volatility_6M'].transform(
        lambda x: 100 - ((x / x.max() * 100) if x.max() > 0 else 0)
    )
    
    # Compound Annual Growth Rate (CAGR) approximation
    df['CAGR_Approx'] = ((1 + df['YoY_Growth_AUM']/100) - 1) * 100
    
    # Trend direction indicator
    df['Trend_Direction'] = np.where(df['AUM_3M_Avg'] > df['AUM_6M_Avg'], 'Uptrend', 
                                     np.where(df['AUM_3M_Avg'] < df['AUM_6M_Avg'], 'Downtrend', 'Neutral'))
    
    return df

def plot_parametric_analysis(df, metric, group_by, top_n=10, agg_func='sum'):
    """Generic parametric plotting function"""
    if agg_func == 'sum':
        grouped = df.groupby(['Date', group_by])[metric].sum().reset_index()
    elif agg_func == 'mean':
        grouped = df.groupby(['Date', group_by])[metric].mean().reset_index()
    elif agg_func == 'median':
        grouped = df.groupby(['Date', group_by])[metric].median().reset_index()
    else:
        grouped = df.groupby(['Date', group_by])[metric].sum().reset_index()
    
    # Get top N entities
    latest_date = grouped['Date'].max()
    top_entities = grouped[grouped['Date'] == latest_date].nlargest(top_n, metric)[group_by].unique()
    filtered = grouped[grouped[group_by].isin(top_entities)]
    
    fig = px.line(
        filtered,
        x='Date',
        y=metric,
        color=group_by,
        title=f'{metric} by {group_by} (Top {top_n})',
        labels={metric: metric.replace('_', ' ').title()}
    )
    
    fig.update_layout(
        hovermode='x unified',
        template='plotly_white',
        height=500
    )
    return fig

def plot_correlation_matrix(df, metrics):
    """Correlation matrix for selected metrics"""
    latest_date = df['Date'].max()
    latest_data = df[df['Date'] == latest_date][metrics].corr()
    
    fig = px.imshow(
        latest_data,
        labels=dict(color="Correlation"),
        x=latest_data.columns,
        y=latest_data.columns,
        color_continuous_scale='RdBu',
        zmin=-1,
        zmax=1,
        title="Correlation Matrix of Selected Metrics"
    )
    
    fig.update_layout(height=600, template='plotly_white')
    return fig

def plot_distribution(df, metric):
    """Distribution plot for any metric"""
    latest_date = df['Date'].max()
    latest_data = df[df['Date'] == latest_date][metric].dropna()
    
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=latest_data,
        nbinsx=50,
        name='Distribution',
        marker_color='lightblue'
    ))
    
    fig.update_layout(
        title=f'Distribution of {metric.replace("_", " ").title()} (Latest Month)',
        xaxis_title=metric.replace('_', ' ').title(),
        yaxis_title='Frequency',
        template='plotly_white',
        height=400
    )
    return fig

def plot_box_plot(df, metric, group_by):
    """Box plot for metric across groups"""
    latest_date = df['Date'].max()
    latest_data = df[df['Date'] == latest_date]
    
    fig = px.box(
        latest_data,
        x=group_by,
        y=metric,
        title=f'{metric.replace("_", " ").title()} Distribution by {group_by}',
        color=group_by
    )
    
    fig.update_layout(
        template='plotly_white',
        height=500,
        showlegend=False
    )
    return fig

def plot_scatter(df, x_metric, y_metric, color_by, size_by=None):
    """Scatter plot with multiple parameters"""
    latest_date = df['Date'].max()
    latest_data = df[df['Date'] == latest_date]
    
    fig = px.scatter(
        latest_data,
        x=x_metric,
        y=y_metric,
        color=color_by,
        size=size_by if size_by else None,
        hover_data=['Scheme_Name'],
        title=f'{y_metric} vs {x_metric}',
        labels={
            x_metric: x_metric.replace('_', ' ').title(),
            y_metric: y_metric.replace('_', ' ').title()
        }
    )
    
    fig.update_layout(
        template='plotly_white',
        height=500
    )
    return fig

def plot_time_series_decomposition(df, scheme_name, metric='AUM'):
    """Time series analysis for a specific scheme"""
    scheme_data = df[df['Scheme_Name'] == scheme_name].sort_values('Date')
    
    if len(scheme_data) < 12:
        st.warning("Not enough data for decomposition (minimum 12 months required)")
        return None
    
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('Original', 'Trend (12-month MA)', 'Seasonality'),
        vertical_spacing=0.1
    )
    
    # Original
    fig.add_trace(
        go.Scatter(x=scheme_data['Date'], y=scheme_data[metric], name='Original', line=dict(color='blue')),
        row=1, col=1
    )
    
    # Trend
    trend = scheme_data[metric].rolling(window=12, center=True).mean()
    fig.add_trace(
        go.Scatter(x=scheme_data['Date'], y=trend, name='Trend', line=dict(color='red')),
        row=2, col=1
    )
    
    # Seasonality (detrended)
    seasonality = scheme_data[metric] - trend
    fig.add_trace(
        go.Scatter(x=scheme_data['Date'], y=seasonality, name='Seasonality', line=dict(color='green')),
        row=3, col=1
    )
    
    fig.update_layout(height=800, title_text=f"Time Series Decomposition - {scheme_name}")
    return fig

def plot_rankings_over_time(df, metric, top_n=10):
    """Show how rankings change over time"""
    rankings = []
    
    for date in df['Date'].unique():
        date_data = df[df['Date'] == date].nlargest(top_n, metric)
        for rank, row in enumerate(date_data.itertuples(), 1):
            rankings.append({
                'Date': date,
                'Scheme_Name': row.Scheme_Name,
                'Rank': rank,
                metric: getattr(row, metric)
            })
    
    rankings_df = pd.DataFrame(rankings)
    
    # Get schemes that appear most frequently in top N
    top_schemes = rankings_df['Scheme_Name'].value_counts().head(top_n).index
    filtered_rankings = rankings_df[rankings_df['Scheme_Name'].isin(top_schemes)]
    
    fig = px.line(
        filtered_rankings,
        x='Date',
        y='Rank',
        color='Scheme_Name',
        title=f'Top {top_n} Schemes Ranking Over Time ({metric})',
        labels={'Rank': 'Rank (1 = Top)'}
    )
    
    fig.update_layout(
        height=500, 
        template='plotly_white',
        yaxis=dict(autorange="reversed")
    )
    return fig

def plot_heatmap_calendar(df, metric):
    """Calendar heatmap for metric"""
    monthly_data = df.groupby(['Year', 'Month'])[metric].sum().reset_index()
    
    pivot = monthly_data.pivot(index='Year', columns='Month', values=metric)
    month_order = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
    pivot = pivot[month_order]
    
    fig = px.imshow(
        pivot,
        labels=dict(x="Month", y="Year", color=metric),
        x=month_order,
        y=pivot.index,
        color_continuous_scale='Viridis',
        title=f'Calendar Heatmap - {metric.replace("_", " ").title()}',
        aspect="auto"
    )
    
    fig.update_layout(height=400, template='plotly_white')
    return fig

def calculate_statistics(df, metric):
    """Calculate comprehensive statistics"""
    latest_date = df['Date'].max()
    data = df[df['Date'] == latest_date][metric].dropna()
    
    stats_dict = {
        'Count': len(data),
        'Mean': data.mean(),
        'Median': data.median(),
        'Std Dev': data.std(),
        'Min': data.min(),
        'Max': data.max(),
        '25th Percentile': data.quantile(0.25),
        '75th Percentile': data.quantile(0.75),
        'Skewness': data.skew(),
        'Kurtosis': data.kurtosis()
    }
    
    return pd.DataFrame(stats_dict.items(), columns=['Statistic', 'Value'])

def display_key_metrics(df):
    """Display key metrics"""
    latest_date = df['Date'].max()
    latest_data = df[df['Date'] == latest_date]
    
    prev_date = df[df['Date'] < latest_date]['Date'].max()
    prev_data = df[df['Date'] == prev_date]
    
    total_aum = latest_data['AUM'].sum()
    prev_aum = prev_data['AUM'].sum()
    aum_change = ((total_aum - prev_aum) / prev_aum * 100) if prev_aum > 0 else 0
    
    total_folios = latest_data['No_of_Folios'].sum()
    prev_folios = prev_data['No_of_Folios'].sum()
    folio_change = ((total_folios - prev_folios) / prev_folios * 100) if prev_folios > 0 else 0
    
    net_inflow = latest_data['Net_Inflow'].sum()
    avg_aum_per_folio = total_aum / total_folios if total_folios > 0 else 0
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total AUM", f"₹{total_aum:,.0f} Cr", f"{aum_change:+.2f}% MoM")
    with col2:
        st.metric("Total Folios", f"{total_folios:,.0f}", f"{folio_change:+.2f}% MoM")
    with col3:
        st.metric("Net Inflow", f"₹{net_inflow:,.0f} Cr", "Latest Month")
    with col4:
        st.metric("Active Schemes", f"{latest_data['No_of_Schemes'].sum():,.0f}", "Total")
    with col5:
        st.metric("Avg AUM/Folio", f"₹{avg_aum_per_folio:,.2f} Cr", "Per Account")

# Main App
def main():
    st.markdown('<p class="main-header">📊 AMFI Mutual Fund Data Analysis Dashboard</p>', 
                unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("📁 Upload Data")
        uploaded_file = st.file_uploader(
            "Upload AMFI Excel File",
            type=['xlsx', 'xls'],
            help="Upload the Excel file generated by the AMFI downloader script"
        )
        
        if uploaded_file:
            st.success("✅ File uploaded successfully!")
        
        st.markdown("---")
        st.header("ℹ️ About")
        st.info("""
        Comprehensive analysis with:
        - 15+ visualization types
        - 20+ calculated metrics
        - Statistical analysis
        - Correlation studies
        - Time series decomposition
        - Custom parametric analysis
        """)
    
    if not uploaded_file:
        st.warning("⬆️ Please upload an AMFI Excel file to begin analysis")
        st.markdown("""
        ### Comprehensive Analysis Features:
        1. **Advanced Metrics**: Growth rates, volatility, rolling averages
        2. **Parametric Analysis**: Customize any metric combination
        3. **Statistical Analysis**: Distributions, correlations, rankings
        4. **Time Series**: Decomposition, trends, seasonality
        5. **Comparative Studies**: Multi-scheme, multi-category analysis
        6. **Custom Filters**: Drill down any dimension
        """)
        return
    
    # Load data
    with st.spinner("Loading and calculating metrics..."):
        df = load_data(uploaded_file)
        if df is not None:
            df = calculate_advanced_metrics(df)
    
    if df is None:
        return
    
    # Display key metrics
    st.subheader("📈 Key Performance Indicators")
    display_key_metrics(df)
    st.markdown("---")
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
        "🎯 Parametric Analysis",
        "📊 Statistical Analysis", 
        "🔍 Growth & Trends",
        "💰 Flow Analysis",
        "📈 Advanced Metrics",
        "🏆 Rankings & Comparison",
        "⏱️ Time Series",
        "📑 Aggregated Analysis",
        "🎪 Comprehensive Metrics",
        "📋 Data Explorer"
    ])
    
    # Tab 1: Parametric Analysis
    with tab1:
        st.subheader("🎯 Custom Parametric Analysis")
        st.markdown("Select any combination of metrics and groupings to analyze")
        
        col1, col2, col3, col4 = st.columns(4)
        
        numeric_cols = ['AUM', 'No_of_Schemes', 'No_of_Folios', 'Funds_Mobilized', 
                       'Repurchase', 'Net_Inflow', 'Average_AUM', 'MoM_Growth_AUM',
                       'YoY_Growth_AUM', 'QoQ_Growth_AUM', '6M_Growth_AUM',
                       'MoM_Growth_Folios', 'YoY_Growth_Folios', 'QoQ_Growth_Folios',
                       'AUM_3M_Avg', 'AUM_6M_Avg', 'AUM_12M_Avg',
                       'NetInflow_3M_Avg', 'NetInflow_6M_Avg',
                       'Net_Inflow_3M_Sum', 'Net_Inflow_6M_Sum', 'Net_Inflow_12M_Sum',
                       'AUM_Volatility_3M', 'AUM_Volatility_6M', 'AUM_Volatility_12M',
                       'Market_Share', 'Market_Share_Change', 'AUM_Per_Folio', 'NetInflow_Per_Folio',
                       'Redemption_Ratio', 'Retention_Rate', 'Gross_Sales_Rate', 'Redemption_Rate',
                       'AUM_Per_Folio_Growth', 'AUM_Momentum_3M', 'AUM_Momentum_6M',
                       'Growth_Acceleration', 'Asset_Gathering_Ratio', 'Flow_Consistency_Score',
                       'CAGR_Approx', 'AUM_Percentile', 'Rank_in_Category', 'Rank_in_Subcategory']
        
        grouping_cols = ['Category_Name', 'Subcategory_Name', 'Scheme_Name']
        
        with col1:
            selected_metric = st.selectbox("Select Metric", numeric_cols)
        with col2:
            selected_group = st.selectbox("Group By", grouping_cols)
        with col3:
            top_n = st.slider("Top N", 5, 20, 10)
        with col4:
            agg_function = st.selectbox("Aggregation", ['sum', 'mean', 'median'])
        
        st.plotly_chart(
            plot_parametric_analysis(df, selected_metric, selected_group, top_n, agg_function),
            use_container_width=True,
            key='param_main_chart'
        )
        
        # Additional parametric views
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(
                plot_distribution(df, selected_metric),
                use_container_width=True,
                key='param_dist_chart'
            )
        
        with col2:
            st.plotly_chart(
                plot_box_plot(df, selected_metric, selected_group),
                use_container_width=True,
                key='param_box_chart'
            )
        
        # Heatmap calendar
        st.plotly_chart(
            plot_heatmap_calendar(df, selected_metric),
            use_container_width=True,
            key='param_heatmap_chart'
        )
    
    # Tab 2: Statistical Analysis
    with tab2:
        st.subheader("📊 Statistical Analysis")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            stat_metric = st.selectbox("Select Metric for Statistics", numeric_cols, key='stat_metric')
            
            stats_df = calculate_statistics(df, stat_metric)
            st.markdown("#### Descriptive Statistics")
            st.dataframe(stats_df.style.format({'Value': '{:.2f}'}), use_container_width=True)
        
        with col2:
            st.markdown("#### Distribution Analysis")
            st.plotly_chart(
                plot_distribution(df, stat_metric),
                use_container_width=True,
                key='stat_dist_chart'
            )
        
        # Correlation Analysis
        st.markdown("#### Correlation Analysis")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            available_metrics = [col for col in numeric_cols if col in df.columns]
            selected_corr_metrics = st.multiselect(
                "Select Metrics for Correlation",
                available_metrics,
                default=available_metrics[:5]
            )
        
        with col2:
            if len(selected_corr_metrics) >= 2:
                st.plotly_chart(
                    plot_correlation_matrix(df, selected_corr_metrics),
                    use_container_width=True,
                    key='stat_corr_chart'
                )
            else:
                st.info("Select at least 2 metrics for correlation analysis")
        
        # Scatter plot analysis
        st.markdown("#### Scatter Plot Analysis")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            x_metric = st.selectbox("X-Axis", numeric_cols, index=0, key='scatter_x')
        with col2:
            y_metric = st.selectbox("Y-Axis", numeric_cols, index=1, key='scatter_y')
        with col3:
            color_by = st.selectbox("Color By", grouping_cols, key='scatter_color')
        with col4:
            size_by = st.selectbox("Size By (optional)", ['None'] + numeric_cols, key='scatter_size')
        
        size_metric = None if size_by == 'None' else size_by
        st.plotly_chart(
            plot_scatter(df, x_metric, y_metric, color_by, size_metric),
            use_container_width=True,
            key='stat_scatter_chart'
        )
    
    # Tab 3: Growth & Trends
    with tab3:
        st.subheader("🔍 Growth & Trend Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            growth_metric = st.selectbox(
                "Select Growth Metric",
                ['MoM_Growth_AUM', 'YoY_Growth_AUM', 'QoQ_Growth_AUM', 
                 'MoM_Growth_Folios', 'YoY_Growth_Folios']
            )
        
        with col2:
            growth_group = st.selectbox("Group By", grouping_cols, key='growth_group')
        
        st.plotly_chart(
            plot_parametric_analysis(df, growth_metric, growth_group, 10, 'mean'),
            use_container_width=True,
            key='growth_main_chart'
        )
        
        # Growth comparison table
        st.markdown("#### Latest Growth Metrics")
        latest_date = df['Date'].max()
        growth_data = df[df['Date'] == latest_date].nlargest(15, 'AUM')[[
            'Scheme_Name', 'Category_Name', 'AUM', 'MoM_Growth_AUM', 
            'YoY_Growth_AUM', 'QoQ_Growth_AUM', 'MoM_Growth_Folios'
        ]]
        
        st.dataframe(
            growth_data.style.format({
                'AUM': '₹{:,.0f}',
                'MoM_Growth_AUM': '{:+.2f}%',
                'YoY_Growth_AUM': '{:+.2f}%',
                'QoQ_Growth_AUM': '{:+.2f}%',
                'MoM_Growth_Folios': '{:+.2f}%'
            }),
            use_container_width=True,
            height=400
        )
        
        # Rolling averages
        st.markdown("#### Rolling Averages Comparison")
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(
                plot_parametric_analysis(df, 'AUM_3M_Avg', growth_group, 10, 'mean'),
                use_container_width=True,
                key='growth_3m_chart'
            )
        
        with col2:
            st.plotly_chart(
                plot_parametric_analysis(df, 'AUM_12M_Avg', growth_group, 10, 'mean'),
                use_container_width=True,
                key='growth_12m_chart'
            )
    
    # Tab 4: Flow Analysis
    with tab4:
        st.subheader("💰 Comprehensive Flow Analysis")
        
        # Net inflow trends
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(
                plot_parametric_analysis(df, 'Net_Inflow', 'Category_Name', 5, 'sum'),
                use_container_width=True,
                key='flow_net_inflow_chart'
            )
        
        with col2:
            st.plotly_chart(
                plot_parametric_analysis(df, 'Net_Inflow_3M_Sum', 'Category_Name', 5, 'sum'),
                use_container_width=True,
                key='flow_3m_inflow_chart'
            )
        
        # Cumulative flows
        st.markdown("#### Cumulative Flow Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(
                plot_parametric_analysis(df, 'Cumulative_Net_Inflow', 'Scheme_Name', 10, 'sum'),
                use_container_width=True,
                key='flow_cumulative_chart'
            )
        
        with col2:
            st.plotly_chart(
                plot_parametric_analysis(df, 'Redemption_Ratio', 'Category_Name', 5, 'mean'),
                use_container_width=True,
                key='flow_redemption_chart'
            )
        
        # Flow metrics table
        st.markdown("#### Flow Metrics Summary")
        latest_date = df['Date'].max()
        flow_data = df[df['Date'] == latest_date].nlargest(15, 'Net_Inflow')[[
            'Scheme_Name', 'Funds_Mobilized', 'Repurchase', 'Net_Inflow',
            'Net_Inflow_3M_Sum', 'Net_Inflow_6M_Sum', 'Redemption_Ratio'
        ]]
        
        st.dataframe(
            flow_data.style.format({
                'Funds_Mobilized': '₹{:,.0f}',
                'Repurchase': '₹{:,.0f}',
                'Net_Inflow': '₹{:,.0f}',
                'Net_Inflow_3M_Sum': '₹{:,.0f}',
                'Net_Inflow_6M_Sum': '₹{:,.0f}',
                'Redemption_Ratio': '{:.2f}%'
            }),
            use_container_width=True,
            height=400
        )
    
    # Tab 5: Advanced Metrics
    with tab5:
        st.subheader("📈 Advanced Metrics Dashboard")
        
        # Market share analysis
        st.markdown("#### Market Share Trends")
        st.plotly_chart(
            plot_parametric_analysis(df, 'Market_Share', 'Scheme_Name', 10, 'mean'),
            use_container_width=True,
            key='advanced_market_share_chart'
        )
        
        # Volatility analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### AUM Volatility (3M)")
            st.plotly_chart(
                plot_parametric_analysis(df, 'AUM_Volatility_3M', 'Category_Name', 5, 'mean'),
                use_container_width=True,
                key='advanced_volatility_3m_chart'
            )
        
        with col2:
            st.markdown("#### AUM Volatility (6M)")
            st.plotly_chart(
                plot_parametric_analysis(df, 'AUM_Volatility_6M', 'Category_Name', 5, 'mean'),
                use_container_width=True,
                key='advanced_volatility_6m_chart'
            )
        
        # AUM per folio
        st.markdown("#### Average AUM per Folio")
        st.plotly_chart(
            plot_parametric_analysis(df, 'AUM_Per_Folio', 'Category_Name', 5, 'mean'),
            use_container_width=True,
            key='advanced_aum_per_folio_chart'
        )
        
        # Advanced metrics table
        st.markdown("#### Advanced Metrics Summary")
        latest_date = df['Date'].max()
        advanced_data = df[df['Date'] == latest_date].nlargest(15, 'AUM')[[
            'Scheme_Name', 'Market_Share', 'AUM_Volatility_3M', 'AUM_Volatility_6M',
            'AUM_Per_Folio', 'Redemption_Ratio'
        ]]
        
        st.dataframe(
            advanced_data.style.format({
                'Market_Share': '{:.2f}%',
                'AUM_Volatility_3M': '{:,.2f}',
                'AUM_Volatility_6M': '{:,.2f}',
                'AUM_Per_Folio': '₹{:,.4f}',
                'Redemption_Ratio': '{:.2f}%'
            }),
            use_container_width=True,
            height=400
        )
    
    # Tab 6: Rankings & Comparison
    with tab6:
        st.subheader("🏆 Rankings & Multi-Dimensional Comparison")
        
        # Rankings over time
        st.markdown("#### How Top Schemes Rankings Changed Over Time")
        ranking_metric = st.selectbox(
            "Select Metric for Ranking",
            ['AUM', 'Net_Inflow', 'No_of_Folios', 'Market_Share'],
            key='ranking_metric'
        )
        
        st.plotly_chart(
            plot_rankings_over_time(df, ranking_metric, top_n=10),
            use_container_width=True,
            key='ranking_chart'
        )
        
        # Multi-scheme comparison
        st.markdown("#### Multi-Scheme Comparison Tool")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            schemes = df['Scheme_Name'].unique()
            selected_schemes = st.multiselect(
                "Select Schemes to Compare (max 8)",
                schemes,
                default=list(schemes[:4]),
                max_selections=8,
                key='compare_schemes'
            )
        
        with col2:
            compare_metric = st.selectbox(
                "Metric to Compare",
                numeric_cols,
                key='compare_metric'
            )
        
        if selected_schemes:
            scheme_data = df[df['Scheme_Name'].isin(selected_schemes)]
            
            fig = px.line(
                scheme_data,
                x='Date',
                y=compare_metric,
                color='Scheme_Name',
                title=f'{compare_metric.replace("_", " ").title()} Comparison',
                labels={compare_metric: compare_metric.replace('_', ' ').title()}
            )
            
            fig.update_layout(
                hovermode='x unified',
                template='plotly_white',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True, key='ranking_comparison_chart')
            
            # Comparison table
            st.markdown("#### Detailed Comparison (Latest Month)")
            latest_date = df['Date'].max()
            comparison_table = df[
                (df['Scheme_Name'].isin(selected_schemes)) & 
                (df['Date'] == latest_date)
            ][[
                'Scheme_Name', 'Category_Name', 'AUM', 'MoM_Growth_AUM', 'YoY_Growth_AUM',
                'Net_Inflow', 'No_of_Folios', 'Market_Share', 'AUM_Per_Folio'
            ]]
            
            st.dataframe(
                comparison_table.style.format({
                    'AUM': '₹{:,.0f}',
                    'MoM_Growth_AUM': '{:+.2f}%',
                    'YoY_Growth_AUM': '{:+.2f}%',
                    'Net_Inflow': '₹{:,.0f}',
                    'No_of_Folios': '{:,.0f}',
                    'Market_Share': '{:.2f}%',
                    'AUM_Per_Folio': '₹{:,.4f}'
                }),
                use_container_width=True
            )
        else:
            st.info("Select at least one scheme to compare")
        
        # Category comparison
        st.markdown("#### Category Performance Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            cat_metric1 = st.selectbox("Primary Metric", numeric_cols, index=0, key='cat_m1')
        
        with col2:
            cat_metric2 = st.selectbox("Secondary Metric", numeric_cols, index=1, key='cat_m2')
        
        latest_date = df['Date'].max()
        cat_comparison = df[df['Date'] == latest_date].groupby('Category_Name').agg({
            cat_metric1: 'sum',
            cat_metric2: 'sum'
        }).reset_index()
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=cat_comparison['Category_Name'],
            y=cat_comparison[cat_metric1],
            name=cat_metric1.replace('_', ' ').title(),
            yaxis='y',
            offsetgroup=1
        ))
        fig.add_trace(go.Bar(
            x=cat_comparison['Category_Name'],
            y=cat_comparison[cat_metric2],
            name=cat_metric2.replace('_', ' ').title(),
            yaxis='y2',
            offsetgroup=2
        ))
        
        fig.update_layout(
            title='Category Comparison - Dual Metrics',
            yaxis=dict(title=cat_metric1.replace('_', ' ').title()),
            yaxis2=dict(title=cat_metric2.replace('_', ' ').title(), overlaying='y', side='right'),
            barmode='group',
            template='plotly_white',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True, key='ranking_category_comparison_chart')
    
    # Tab 7: Time Series Analysis
    with tab7:
        st.subheader("⏱️ Time Series Analysis & Forecasting")
        
        # Scheme selector for time series
        ts_scheme = st.selectbox(
            "Select Scheme for Time Series Analysis",
            df['Scheme_Name'].unique(),
            key='ts_scheme'
        )
        
        ts_metric = st.selectbox(
            "Select Metric",
            ['AUM', 'Net_Inflow', 'No_of_Folios', 'Funds_Mobilized'],
            key='ts_metric'
        )
        
        # Time series decomposition
        st.markdown("#### Time Series Decomposition")
        decomp_fig = plot_time_series_decomposition(df, ts_scheme, ts_metric)
        if decomp_fig:
            st.plotly_chart(decomp_fig, use_container_width=True, key='ts_decomposition_chart')
        
        # Detailed scheme analysis
        st.markdown("#### Detailed Historical Analysis")
        scheme_history = df[df['Scheme_Name'] == ts_scheme].sort_values('Date')
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Original vs Moving Averages
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=scheme_history['Date'],
                y=scheme_history[ts_metric],
                name='Original',
                line=dict(color='blue', width=2)
            ))
            fig.add_trace(go.Scatter(
                x=scheme_history['Date'],
                y=scheme_history[f'{ts_metric}_3M_Avg'] if f'{ts_metric}_3M_Avg' in scheme_history.columns else scheme_history[ts_metric],
                name='3M Average',
                line=dict(color='red', width=2, dash='dash')
            ))
            fig.add_trace(go.Scatter(
                x=scheme_history['Date'],
                y=scheme_history[f'{ts_metric}_12M_Avg'] if f'{ts_metric}_12M_Avg' in scheme_history.columns else scheme_history[ts_metric],
                name='12M Average',
                line=dict(color='green', width=2, dash='dot')
            ))
            
            fig.update_layout(
                title=f'{ts_metric} with Moving Averages',
                template='plotly_white',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True, key='ts_moving_avg_chart')
        
        with col2:
            # Growth rates
            fig = go.Figure()
            if 'MoM_Growth_AUM' in scheme_history.columns and ts_metric == 'AUM':
                fig.add_trace(go.Scatter(
                    x=scheme_history['Date'],
                    y=scheme_history['MoM_Growth_AUM'],
                    name='MoM Growth',
                    line=dict(color='orange')
                ))
                fig.add_trace(go.Scatter(
                    x=scheme_history['Date'],
                    y=scheme_history['YoY_Growth_AUM'],
                    name='YoY Growth',
                    line=dict(color='purple')
                ))
                fig.update_layout(
                    title='Growth Rates',
                    yaxis_title='Growth (%)',
                    template='plotly_white',
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True, key='ts_growth_rates_chart')
        
        # Summary statistics for the scheme
        st.markdown("#### Historical Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                f"Average {ts_metric}",
                f"{scheme_history[ts_metric].mean():,.2f}"
            )
        with col2:
            st.metric(
                f"Max {ts_metric}",
                f"{scheme_history[ts_metric].max():,.2f}"
            )
        with col3:
            st.metric(
                f"Min {ts_metric}",
                f"{scheme_history[ts_metric].min():,.2f}"
            )
        with col4:
            st.metric(
                "Volatility (Std Dev)",
                f"{scheme_history[ts_metric].std():,.2f}"
            )
        
        # Year-over-Year comparison
        st.markdown("#### Year-over-Year Comparison")
        
        yoy_data = scheme_history.copy()
        yoy_data['Year_Label'] = yoy_data['Year'].astype(int).astype(str)
        yoy_data['Month_Name'] = yoy_data['Month']
        
        fig = px.line(
            yoy_data,
            x='Month_Name',
            y=ts_metric,
            color='Year_Label',
            title=f'{ts_metric} by Month Across Years',
            category_orders={
                'Month_Name': ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 
                              'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
            }
        )
        
        fig.update_layout(
            template='plotly_white',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True, key='ts_yoy_comparison_chart')
    
    # Tab 8: Aggregated Analysis
    with tab8:
        st.subheader("📑 Aggregated Analysis - Category & Subcategory Level")
        
        st.markdown("""
        This section shows aggregated metrics at Category and Subcategory levels over time.
        Perfect for understanding broader market trends and segments.
        """)
        
        # Category level aggregation
        st.markdown("### 📊 Category-Level Aggregations")
        
        category_agg = df.groupby(['Date', 'Category_ID', 'Category_Name']).agg({
            'No_of_Schemes': 'sum',
            'No_of_Folios': 'sum',
            'Funds_Mobilized': 'sum',
            'Repurchase': 'sum',
            'Net_Inflow': 'sum',
            'AUM': 'sum',
            'Average_AUM': 'sum'
        }).reset_index()
        
        # Calculate category growth
        category_agg = category_agg.sort_values(['Category_ID', 'Date'])
        category_agg['AUM_MoM_Growth'] = category_agg.groupby('Category_ID')['AUM'].pct_change() * 100
        category_agg['AUM_YoY_Growth'] = category_agg.groupby('Category_ID')['AUM'].pct_change(12) * 100
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Category AUM over time
            fig = px.area(
                category_agg,
                x='Date',
                y='AUM',
                color='Category_Name',
                title='Category-wise Total AUM Over Time',
                labels={'AUM': 'Total AUM (₹ Crores)'}
            )
            fig.update_layout(template='plotly_white', height=450)
            st.plotly_chart(fig, use_container_width=True, key='agg_cat_aum')
        
        with col2:
            # Category Net Inflow
            fig = px.bar(
                category_agg,
                x='Date',
                y='Net_Inflow',
                color='Category_Name',
                title='Category-wise Net Inflow Over Time',
                labels={'Net_Inflow': 'Net Inflow (₹ Crores)'}
            )
            fig.update_layout(template='plotly_white', height=450)
            st.plotly_chart(fig, use_container_width=True, key='agg_cat_inflow')
        
        # Category growth rates
        st.markdown("#### Category Growth Rates")
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.line(
                category_agg,
                x='Date',
                y='AUM_MoM_Growth',
                color='Category_Name',
                title='Category MoM Growth Rate (%)',
                labels={'AUM_MoM_Growth': 'MoM Growth (%)'}
            )
            fig.update_layout(template='plotly_white', height=400)
            st.plotly_chart(fig, use_container_width=True, key='agg_cat_mom')
        
        with col2:
            fig = px.line(
                category_agg,
                x='Date',
                y='AUM_YoY_Growth',
                color='Category_Name',
                title='Category YoY Growth Rate (%)',
                labels={'AUM_YoY_Growth': 'YoY Growth (%)'}
            )
            fig.update_layout(template='plotly_white', height=400)
            st.plotly_chart(fig, use_container_width=True, key='agg_cat_yoy')
        
        # Latest category metrics
        st.markdown("#### Latest Category Metrics")
        latest_date = category_agg['Date'].max()
        latest_cat = category_agg[category_agg['Date'] == latest_date][[
            'Category_Name', 'No_of_Schemes', 'No_of_Folios', 'AUM', 
            'Net_Inflow', 'AUM_MoM_Growth', 'AUM_YoY_Growth'
        ]]
        
        st.dataframe(
            latest_cat.style.format({
                'No_of_Schemes': '{:.0f}',
                'No_of_Folios': '{:,.0f}',
                'AUM': '₹{:,.0f}',
                'Net_Inflow': '₹{:,.0f}',
                'AUM_MoM_Growth': '{:+.2f}%',
                'AUM_YoY_Growth': '{:+.2f}%'
            }),
            use_container_width=True
        )
        
        st.markdown("---")
        
        # Subcategory level aggregation
        st.markdown("### 📊 Subcategory-Level Aggregations")
        
        subcategory_agg = df.groupby(['Date', 'Category_ID', 'Subcategory_ID', 'Subcategory_Name']).agg({
            'No_of_Schemes': 'sum',
            'No_of_Folios': 'sum',
            'Funds_Mobilized': 'sum',
            'Repurchase': 'sum',
            'Net_Inflow': 'sum',
            'AUM': 'sum',
            'Average_AUM': 'sum'
        }).reset_index()
        
        # Calculate subcategory growth
        subcategory_agg = subcategory_agg.sort_values(['Subcategory_ID', 'Date'])
        subcategory_agg['AUM_MoM_Growth'] = subcategory_agg.groupby('Subcategory_ID')['AUM'].pct_change() * 100
        subcategory_agg['AUM_YoY_Growth'] = subcategory_agg.groupby('Subcategory_ID')['AUM'].pct_change(12) * 100
        
        # Filter selector
        selected_category_agg = st.selectbox(
            "Select Category to View Subcategories",
            subcategory_agg['Category_ID'].unique(),
            format_func=lambda x: f"{x} - {subcategory_agg[subcategory_agg['Category_ID']==x]['Subcategory_Name'].iloc[0].split('/')[0]}"
        )
        
        filtered_subcat = subcategory_agg[subcategory_agg['Category_ID'] == selected_category_agg]
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Subcategory AUM trend
            fig = px.line(
                filtered_subcat,
                x='Date',
                y='AUM',
                color='Subcategory_Name',
                title=f'Subcategory AUM Trends - Category {selected_category_agg}',
                labels={'AUM': 'Total AUM (₹ Crores)'}
            )
            fig.update_layout(template='plotly_white', height=450)
            st.plotly_chart(fig, use_container_width=True, key='agg_subcat_aum')
        
        with col2:
            # Subcategory market share within category
            latest_subcat = filtered_subcat[filtered_subcat['Date'] == latest_date]
            fig = px.pie(
                latest_subcat,
                values='AUM',
                names='Subcategory_Name',
                title=f'Subcategory Distribution - Category {selected_category_agg} (Latest)',
                hole=0.4
            )
            fig.update_layout(template='plotly_white', height=450)
            st.plotly_chart(fig, use_container_width=True, key='agg_subcat_pie')
        
        # Subcategory growth comparison
        st.markdown("#### Subcategory Growth Comparison")
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(
                filtered_subcat,
                x='Date',
                y='Net_Inflow',
                color='Subcategory_Name',
                title=f'Subcategory Net Inflow - Category {selected_category_agg}',
                labels={'Net_Inflow': 'Net Inflow (₹ Crores)'}
            )
            fig.update_layout(template='plotly_white', height=400)
            st.plotly_chart(fig, use_container_width=True, key='agg_subcat_inflow')
        
        with col2:
            fig = px.line(
                filtered_subcat,
                x='Date',
                y='No_of_Folios',
                color='Subcategory_Name',
                title=f'Subcategory Investor Accounts - Category {selected_category_agg}',
                labels={'No_of_Folios': 'Number of Folios'}
            )
            fig.update_layout(template='plotly_white', height=400)
            st.plotly_chart(fig, use_container_width=True, key='agg_subcat_folios')
        
        # Latest subcategory metrics
        st.markdown("#### Latest Subcategory Metrics")
        latest_subcat_full = filtered_subcat[filtered_subcat['Date'] == latest_date][[
            'Subcategory_Name', 'No_of_Schemes', 'No_of_Folios', 'AUM',
            'Net_Inflow', 'AUM_MoM_Growth', 'AUM_YoY_Growth'
        ]].sort_values('AUM', ascending=False)
        
        st.dataframe(
            latest_subcat_full.style.format({
                'No_of_Schemes': '{:.0f}',
                'No_of_Folios': '{:,.0f}',
                'AUM': '₹{:,.0f}',
                'Net_Inflow': '₹{:,.0f}',
                'AUM_MoM_Growth': '{:+.2f}%',
                'AUM_YoY_Growth': '{:+.2f}%'
            }),
            use_container_width=True
        )
        
        st.markdown("---")
        
        # All subcategories comparison
        st.markdown("### 📊 All Subcategories Comparison")
        
        comparison_metric = st.selectbox(
            "Select Metric for Comparison",
            ['AUM', 'Net_Inflow', 'No_of_Folios', 'No_of_Schemes', 'AUM_YoY_Growth'],
            key='subcat_comparison_metric'
        )
        
        # Get top 10 subcategories by selected metric
        latest_all_subcat = subcategory_agg[subcategory_agg['Date'] == latest_date].nlargest(10, comparison_metric)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(
                latest_all_subcat.sort_values(comparison_metric),
                x=comparison_metric,
                y='Subcategory_Name',
                orientation='h',
                title=f'Top 10 Subcategories by {comparison_metric}',
                color='Category_ID',
                labels={comparison_metric: comparison_metric.replace('_', ' ').title()}
            )
            fig.update_layout(template='plotly_white', height=500)
            st.plotly_chart(fig, use_container_width=True, key='agg_top_subcat')
        
        with col2:
            # Time series for top subcategories
            top_subcat_names = latest_all_subcat['Subcategory_Name'].tolist()
            top_subcat_ts = subcategory_agg[subcategory_agg['Subcategory_Name'].isin(top_subcat_names)]
            
            fig = px.line(
                top_subcat_ts,
                x='Date',
                y=comparison_metric,
                color='Subcategory_Name',
                title=f'Top 10 Subcategories {comparison_metric} Over Time',
                labels={comparison_metric: comparison_metric.replace('_', ' ').title()}
            )
            fig.update_layout(template='plotly_white', height=500)
            st.plotly_chart(fig, use_container_width=True, key='agg_top_subcat_ts')
        
        # Export aggregated data
        st.markdown("#### 📥 Export Aggregated Data")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            cat_csv = category_agg.to_csv(index=False)
            st.download_button(
                label="Download Category Data (CSV)",
                data=cat_csv,
                file_name=f"category_aggregated_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        
        with col2:
            subcat_csv = subcategory_agg.to_csv(index=False)
            st.download_button(
                label="Download Subcategory Data (CSV)",
                data=subcat_csv,
                file_name=f"subcategory_aggregated_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        
        with col3:
            # Combined Excel export
            from io import BytesIO
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                category_agg.to_excel(writer, sheet_name='Category_Aggregated', index=False)
                subcategory_agg.to_excel(writer, sheet_name='Subcategory_Aggregated', index=False)
            
            excel_data = output.getvalue()
            st.download_button(
                label="Download Both (Excel)",
                data=excel_data,
                file_name=f"aggregated_analysis_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    
    # Tab 9: Comprehensive Metrics Dashboard
    with tab9:
        st.subheader("🎪 Comprehensive Metrics Dashboard")
        
        st.markdown("""
        ### All-in-One Metrics Overview
        This tab provides a comprehensive view of all calculated metrics including efficiency, consistency, 
        momentum, and behavioral indicators.
        """)
        
        latest_date = df['Date'].max()
        latest_data = df[df['Date'] == latest_date]
        
        # Scheme selector
        selected_scheme_metrics = st.selectbox(
            "Select Scheme for Detailed Metrics",
            df['Scheme_Name'].unique(),
            key='comp_metrics_scheme'
        )
        
        scheme_metrics = df[df['Scheme_Name'] == selected_scheme_metrics].sort_values('Date')
        latest_scheme = latest_data[latest_data['Scheme_Name'] == selected_scheme_metrics].iloc[0]
        
        # Display comprehensive metrics in organized sections
        st.markdown(f"### 📊 Metrics for: **{selected_scheme_metrics}**")
        st.markdown(f"**Category**: {latest_scheme['Category_Name']} | **Subcategory**: {latest_scheme['Subcategory_Name']}")
        
        # Section 1: Size & Position Metrics
        st.markdown("#### 📏 Size & Position Metrics")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Current AUM", f"₹{latest_scheme['AUM']:,.0f} Cr")
            st.caption(f"Size: {latest_scheme['Size_Category']}")
        with col2:
            st.metric("Market Share", f"{latest_scheme['Market_Share']:.3f}%")
            st.caption(f"Change: {latest_scheme['Market_Share_Change']:.3f}%")
        with col3:
            st.metric("Rank in Category", f"{int(latest_scheme['Rank_in_Category'])}")
            st.caption(f"Percentile: {latest_scheme['AUM_Percentile']:.1f}%")
        with col4:
            st.metric("Rank in Subcategory", f"{int(latest_scheme['Rank_in_Subcategory'])}")
        with col5:
            st.metric("Total Folios", f"{latest_scheme['No_of_Folios']:,.0f}")
            st.caption(f"AUM/Folio: ₹{latest_scheme['AUM_Per_Folio']:.2f} Cr")
        
        # Section 2: Growth Metrics
        st.markdown("#### 📈 Growth Metrics")
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        with col1:
            st.metric("MoM Growth", f"{latest_scheme['MoM_Growth_AUM']:.2f}%")
        with col2:
            st.metric("QoQ Growth", f"{latest_scheme['QoQ_Growth_AUM']:.2f}%")
        with col3:
            st.metric("6M Growth", f"{latest_scheme['6M_Growth_AUM']:.2f}%")
        with col4:
            st.metric("YoY Growth", f"{latest_scheme['YoY_Growth_AUM']:.2f}%")
        with col5:
            st.metric("CAGR (Approx)", f"{latest_scheme['CAGR_Approx']:.2f}%")
        with col6:
            st.metric("Folio Growth (YoY)", f"{latest_scheme['YoY_Growth_Folios']:.2f}%")
        
        # Section 3: Momentum & Trend
        st.markdown("#### 🚀 Momentum & Trend Indicators")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("3M Momentum", f"{latest_scheme['AUM_Momentum_3M']:.2f}%")
        with col2:
            st.metric("6M Momentum", f"{latest_scheme['AUM_Momentum_6M']:.2f}%")
        with col3:
            st.metric("Growth Acceleration", f"{latest_scheme['Growth_Acceleration']:.2f}%")
        with col4:
            trend_color = "🟢" if latest_scheme['Trend_Direction'] == 'Uptrend' else "🔴" if latest_scheme['Trend_Direction'] == 'Downtrend' else "🟡"
            st.metric("Trend Direction", f"{trend_color} {latest_scheme['Trend_Direction']}")
        with col5:
            st.metric("AUM/Folio Growth", f"{latest_scheme['AUM_Per_Folio_Growth']:.2f}%")
        
        # Section 4: Flow Metrics
        st.markdown("#### 💰 Flow & Activity Metrics")
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        with col1:
            st.metric("Net Inflow", f"₹{latest_scheme['Net_Inflow']:,.0f} Cr")
        with col2:
            st.metric("3M Net Inflow", f"₹{latest_scheme['Net_Inflow_3M_Sum']:,.0f} Cr")
        with col3:
            st.metric("6M Net Inflow", f"₹{latest_scheme['Net_Inflow_6M_Sum']:,.0f} Cr")
        with col4:
            st.metric("12M Net Inflow", f"₹{latest_scheme['Net_Inflow_12M_Sum']:,.0f} Cr")
        with col5:
            st.metric("Cumulative Inflow", f"₹{latest_scheme['Cumulative_Net_Inflow']:,.0f} Cr")
        with col6:
            st.metric("Inflow/Folio", f"₹{latest_scheme['NetInflow_Per_Folio']:.2f} Cr")
        
        # Section 5: Efficiency & Behavior Metrics
        st.markdown("#### ⚡ Efficiency & Behavioral Metrics")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Retention Rate", f"{latest_scheme['Retention_Rate']:.2f}%")
            st.caption("Higher is better")
        with col2:
            st.metric("Redemption Ratio", f"{latest_scheme['Redemption_Ratio']:.2f}%")
            st.caption("Lower is better")
        with col3:
            st.metric("Gross Sales Rate", f"{latest_scheme['Gross_Sales_Rate']:.2f}%")
        with col4:
            st.metric("Asset Gathering", f"{latest_scheme['Asset_Gathering_Ratio']:.2f}%")
            st.caption("Net/Gross sales")
        with col5:
            st.metric("Flow Consistency", f"{latest_scheme['Flow_Consistency_Score']:.1f}")
            st.caption("100 = Most consistent")
        
        # Section 6: Volatility Metrics
        st.markdown("#### 📊 Volatility & Stability Metrics")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("AUM Vol (3M)", f"₹{latest_scheme['AUM_Volatility_3M']:,.0f} Cr")
        with col2:
            st.metric("AUM Vol (6M)", f"₹{latest_scheme['AUM_Volatility_6M']:,.0f} Cr")
        with col3:
            st.metric("AUM Vol (12M)", f"₹{latest_scheme['AUM_Volatility_12M']:,.0f} Cr")
        with col4:
            st.metric("NetInflow Vol (3M)", f"₹{latest_scheme['NetInflow_Volatility_3M']:,.0f} Cr")
        with col5:
            st.metric("NetInflow Vol (6M)", f"₹{latest_scheme['NetInflow_Volatility_6M']:,.0f} Cr")
        
        st.markdown("---")
        
        # Visual Analytics
        st.markdown("### 📈 Visual Metrics Analysis")
        
        # Growth comparison chart
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Growth Rates Evolution")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=scheme_metrics['Date'], y=scheme_metrics['MoM_Growth_AUM'], 
                                    name='MoM', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=scheme_metrics['Date'], y=scheme_metrics['QoQ_Growth_AUM'], 
                                    name='QoQ', line=dict(color='green')))
            fig.add_trace(go.Scatter(x=scheme_metrics['Date'], y=scheme_metrics['YoY_Growth_AUM'], 
                                    name='YoY', line=dict(color='red')))
            fig.update_layout(template='plotly_white', height=400, 
                            yaxis_title='Growth (%)', hovermode='x unified')
            st.plotly_chart(fig, use_container_width=True, key='comp_growth_evolution')
        
        with col2:
            st.markdown("#### Momentum Indicators")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=scheme_metrics['Date'], y=scheme_metrics['AUM_Momentum_3M'], 
                                    name='3M Momentum', line=dict(color='orange')))
            fig.add_trace(go.Scatter(x=scheme_metrics['Date'], y=scheme_metrics['AUM_Momentum_6M'], 
                                    name='6M Momentum', line=dict(color='purple')))
            fig.add_trace(go.Scatter(x=scheme_metrics['Date'], y=scheme_metrics['Growth_Acceleration'], 
                                    name='Acceleration', line=dict(color='brown', dash='dash')))
            fig.update_layout(template='plotly_white', height=400, 
                            yaxis_title='Momentum/Acceleration (%)', hovermode='x unified')
            st.plotly_chart(fig, use_container_width=True, key='comp_momentum')
        
        # Efficiency metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Retention & Redemption")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=scheme_metrics['Date'], y=scheme_metrics['Retention_Rate'], 
                                    name='Retention Rate', line=dict(color='green'), fill='tozeroy'))
            fig.add_trace(go.Scatter(x=scheme_metrics['Date'], y=scheme_metrics['Redemption_Ratio'], 
                                    name='Redemption Ratio', line=dict(color='red')))
            fig.update_layout(template='plotly_white', height=400, 
                            yaxis_title='Rate (%)', hovermode='x unified')
            st.plotly_chart(fig, use_container_width=True, key='comp_retention')
        
        with col2:
            st.markdown("#### Market Share Evolution")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=scheme_metrics['Date'], y=scheme_metrics['Market_Share'], 
                                    name='Market Share', line=dict(color='blue'), fill='tozeroy'))
            fig.add_trace(go.Bar(x=scheme_metrics['Date'], y=scheme_metrics['Market_Share_Change'], 
                                name='MS Change', marker_color='lightblue'))
            fig.update_layout(template='plotly_white', height=400, 
                            yaxis_title='Market Share (%)', hovermode='x unified')
            st.plotly_chart(fig, use_container_width=True, key='comp_market_share')
        
        # Per folio metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### AUM Per Folio Trend")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=scheme_metrics['Date'], y=scheme_metrics['AUM_Per_Folio'], 
                                    name='AUM/Folio', line=dict(color='darkgreen'), fill='tozeroy'))
            fig.update_layout(template='plotly_white', height=400, 
                            yaxis_title='AUM per Folio (₹ Cr)', hovermode='x unified')
            st.plotly_chart(fig, use_container_width=True, key='comp_aum_per_folio')
        
        with col2:
            st.markdown("#### Flow Consistency Score")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=scheme_metrics['Date'], y=scheme_metrics['Flow_Consistency_Score'], 
                                    name='Consistency Score', line=dict(color='purple'), 
                                    fill='tozeroy', fillcolor='rgba(128,0,128,0.2)'))
            fig.update_layout(template='plotly_white', height=400, 
                            yaxis_title='Consistency Score', hovermode='x unified')
            st.plotly_chart(fig, use_container_width=True, key='comp_consistency')
        
        st.markdown("---")
        
        # Comparative Analysis
        st.markdown("### 🔍 Comparative Analysis with Peers")
        
        # Get peer schemes (same category)
        peer_schemes = latest_data[
            latest_data['Category_ID'] == latest_scheme['Category_ID']
        ].nlargest(10, 'AUM')
        
        comparison_metric = st.selectbox(
            "Select Metric for Peer Comparison",
            ['YoY_Growth_AUM', 'Retention_Rate', 'Market_Share', 'AUM_Per_Folio', 
             'Flow_Consistency_Score', 'Asset_Gathering_Ratio', 'CAGR_Approx'],
            key='peer_comp_metric'
        )
        
        fig = px.bar(
            peer_schemes.sort_values(comparison_metric, ascending=False),
            x='Scheme_Name',
            y=comparison_metric,
            color=comparison_metric,
            title=f'Top 10 Schemes in Category - {comparison_metric.replace("_", " ").title()}',
            color_continuous_scale='RdYlGn'
        )
        fig.update_layout(template='plotly_white', height=500, xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True, key='comp_peer_comparison')
        
        # Summary statistics table
        st.markdown("### 📋 Complete Metrics Summary Table")
        
        metrics_summary = pd.DataFrame({
            'Metric Category': [
                'Size', 'Size', 'Size', 'Size',
                'Growth', 'Growth', 'Growth', 'Growth', 'Growth',
                'Momentum', 'Momentum', 'Momentum', 'Momentum',
                'Flow', 'Flow', 'Flow', 'Flow',
                'Efficiency', 'Efficiency', 'Efficiency', 'Efficiency',
                'Volatility', 'Volatility', 'Volatility'
            ],
            'Metric Name': [
                'AUM', 'Market Share', 'Rank in Category', 'No of Folios',
                'MoM Growth', 'QoQ Growth', 'YoY Growth', 'CAGR', 'Folio Growth (YoY)',
                '3M Momentum', '6M Momentum', 'Growth Acceleration', 'Trend Direction',
                'Net Inflow', '6M Net Inflow', 'Cumulative Inflow', 'Inflow per Folio',
                'Retention Rate', 'Redemption Ratio', 'Asset Gathering', 'Flow Consistency',
                'AUM Vol (6M)', 'NetInflow Vol (6M)', 'AUM Per Folio'
            ],
            'Value': [
                f"₹{latest_scheme['AUM']:,.0f} Cr",
                f"{latest_scheme['Market_Share']:.3f}%",
                f"{int(latest_scheme['Rank_in_Category'])}",
                f"{latest_scheme['No_of_Folios']:,.0f}",
                f"{latest_scheme['MoM_Growth_AUM']:.2f}%",
                f"{latest_scheme['QoQ_Growth_AUM']:.2f}%",
                f"{latest_scheme['YoY_Growth_AUM']:.2f}%",
                f"{latest_scheme['CAGR_Approx']:.2f}%",
                f"{latest_scheme['YoY_Growth_Folios']:.2f}%",
                f"{latest_scheme['AUM_Momentum_3M']:.2f}%",
                f"{latest_scheme['AUM_Momentum_6M']:.2f}%",
                f"{latest_scheme['Growth_Acceleration']:.2f}%",
                latest_scheme['Trend_Direction'],
                f"₹{latest_scheme['Net_Inflow']:,.0f} Cr",
                f"₹{latest_scheme['Net_Inflow_6M_Sum']:,.0f} Cr",
                f"₹{latest_scheme['Cumulative_Net_Inflow']:,.0f} Cr",
                f"₹{latest_scheme['NetInflow_Per_Folio']:.2f} Cr",
                f"{latest_scheme['Retention_Rate']:.2f}%",
                f"{latest_scheme['Redemption_Ratio']:.2f}%",
                f"{latest_scheme['Asset_Gathering_Ratio']:.2f}%",
                f"{latest_scheme['Flow_Consistency_Score']:.1f}",
                f"₹{latest_scheme['AUM_Volatility_6M']:,.0f} Cr",
                f"₹{latest_scheme['NetInflow_Volatility_6M']:,.0f} Cr",
                f"₹{latest_scheme['AUM_Per_Folio']:.2f} Cr"
            ]
        })
        
        st.dataframe(metrics_summary, use_container_width=True, height=600)
        
        # Export metrics
        csv_metrics = metrics_summary.to_csv(index=False)
        st.download_button(
            label="📥 Download Metrics Summary (CSV)",
            data=csv_metrics,
            file_name=f"comprehensive_metrics_{selected_scheme_metrics.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    # Tab 10: Data Explorer
    with tab10:
        st.subheader("📋 Advanced Data Explorer")
        
        st.markdown("#### Multi-Dimensional Filters")
        
        # Filter controls
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            filter_categories = st.multiselect(
                "Categories",
                df['Category_Name'].unique(),
                key='explorer_cat'
            )
        
        with col2:
            filter_subcategories = st.multiselect(
                "Subcategories",
                df['Subcategory_Name'].unique(),
                key='explorer_subcat'
            )
        
        with col3:
            filter_schemes = st.multiselect(
                "Schemes",
                df['Scheme_Name'].unique(),
                key='explorer_schemes'
            )
        
        with col4:
            date_range = st.date_input(
                "Date Range",
                value=(df['Date'].min(), df['Date'].max()),
                min_value=df['Date'].min(),
                max_value=df['Date'].max(),
                key='explorer_date'
            )
        
        # Apply filters
        filtered_df = df.copy()
        
        if filter_categories:
            filtered_df = filtered_df[filtered_df['Category_Name'].isin(filter_categories)]
        if filter_subcategories:
            filtered_df = filtered_df[filtered_df['Subcategory_Name'].isin(filter_subcategories)]
        if filter_schemes:
            filtered_df = filtered_df[filtered_df['Scheme_Name'].isin(filter_schemes)]
        if len(date_range) == 2:
            filtered_df = filtered_df[
                (filtered_df['Date'] >= pd.to_datetime(date_range[0])) &
                (filtered_df['Date'] <= pd.to_datetime(date_range[1]))
            ]
        
        # Display filtered data summary
        st.markdown("#### Filtered Data Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", f"{len(filtered_df):,}")
        with col2:
            st.metric("Unique Schemes", f"{filtered_df['Scheme_Name'].nunique():,}")
        with col3:
            st.metric("Date Range", f"{filtered_df['Date'].nunique()} months")
        with col4:
            st.metric("Total AUM", f"₹{filtered_df['AUM'].sum():,.0f} Cr")
        
        # Column selector
        st.markdown("#### Select Columns to Display")
        all_columns = filtered_df.columns.tolist()
        default_cols = ['Date', 'Category_Name', 'Subcategory_Name', 'Scheme_Name', 
                       'AUM', 'Net_Inflow', 'No_of_Folios', 'No_of_Schemes']
        
        selected_columns = st.multiselect(
            "Choose columns",
            all_columns,
            default=[col for col in default_cols if col in all_columns],
            key='explorer_cols'
        )
        
        if selected_columns:
            # Display data
            st.dataframe(
                filtered_df[selected_columns].sort_values('Date', ascending=False),
                use_container_width=True,
                height=500
            )
            
            # Export options
            st.markdown("#### Export Options")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                csv = filtered_df[selected_columns].to_csv(index=False)
                st.download_button(
                    label="📥 Download as CSV",
                    data=csv,
                    file_name=f"amfi_filtered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            with col2:
                # Create Excel with multiple sheets
                from io import BytesIO
                output = BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    filtered_df[selected_columns].to_excel(writer, sheet_name='Filtered_Data', index=False)
                    
                    # Add summary sheet - flatten the multi-index columns
                    summary_df = filtered_df.groupby('Scheme_Name').agg({
                        'AUM': ['mean', 'max', 'min'],
                        'Net_Inflow': 'sum',
                        'No_of_Folios': 'mean'
                    })
                    
                    # Flatten multi-index columns
                    summary_df.columns = ['_'.join(col).strip() for col in summary_df.columns.values]
                    summary_df = summary_df.reset_index()
                    
                    summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                excel_data = output.getvalue()
                st.download_button(
                    label="📥 Download as Excel",
                    data=excel_data,
                    file_name=f"amfi_filtered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            
            with col3:
                # JSON export
                json_data = filtered_df[selected_columns].to_json(orient='records', date_format='iso')
                st.download_button(
                    label="📥 Download as JSON",
                    data=json_data,
                    file_name=f"amfi_filtered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            
            # Pivot table creator
            st.markdown("#### Custom Pivot Table")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                pivot_index = st.selectbox("Index (Rows)", selected_columns, key='pivot_idx')
            with col2:
                pivot_columns = st.selectbox("Columns", selected_columns, key='pivot_cols')
            with col3:
                pivot_values = st.selectbox("Values", numeric_cols, key='pivot_vals')
            with col4:
                pivot_aggfunc = st.selectbox("Aggregation", ['sum', 'mean', 'count', 'min', 'max'], key='pivot_agg')
            
            if st.button("Generate Pivot Table"):
                try:
                    pivot_table = filtered_df.pivot_table(
                        index=pivot_index,
                        columns=pivot_columns,
                        values=pivot_values,
                        aggfunc=pivot_aggfunc,
                        fill_value=0
                    )
                    
                    st.dataframe(pivot_table, use_container_width=True)
                    
                    # Download pivot
                    pivot_csv = pivot_table.to_csv()
                    st.download_button(
                        label="📥 Download Pivot Table",
                        data=pivot_csv,
                        file_name=f"pivot_table_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                except Exception as e:
                    st.error(f"Error creating pivot table: {e}")
        else:
            st.info("Please select at least one column to display")
        
        # Advanced search
        st.markdown("#### Advanced Search")
        search_col, search_val = st.columns([1, 3])
        
        with search_col:
            search_column = st.selectbox("Search in column", all_columns, key='search_col')
        
        with search_val:
            search_value = st.text_input("Search value", key='search_val')
        
        if search_value:
            search_results = filtered_df[
                filtered_df[search_column].astype(str).str.contains(search_value, case=False, na=False)
            ]
            st.markdown(f"**Found {len(search_results)} results**")
            st.dataframe(search_results[selected_columns if selected_columns else default_cols], 
                        use_container_width=True, height=300)

if __name__ == "__main__":
    main()
