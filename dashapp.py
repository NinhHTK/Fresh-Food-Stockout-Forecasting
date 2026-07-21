import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import ast
from matplotlib.gridspec import GridSpec

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & LOAD DATA
# -----------------------------------------------------------------------------
st.set_page_config(page_title="FreshRetailNet - City 03 Dashboard", layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        background-color: #0b1f37;
        background-image: linear-gradient(180deg, rgba(11,31,55,0.98) 0%, rgba(1,12,30,0.98) 100%);
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
    section[data-testid="stSidebar"] {
        background-image: linear-gradient(180deg, rgba(3,14,35,0.96) 0%, rgba(0, 93, 172,0.45) 100%);
        background-size: cover;
        background-position: center;
        border-right: 1px solid rgba(255,255,255,0.12);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📊 FreshRetail Analytics Dashboard - City ID 03")
st.markdown("Use the filters on the left sidebar to analyze data by **Store ID** and **Time Period**.")

@st.cache_data
def load_raw_dataset():
    filename = "freshretailnet_city03_dataset.csv"
    if os.path.exists(filename):
        df = pd.read_csv(filename)
        df['dt'] = pd.to_datetime(df['dt'])
        return df
    return None

df_raw = load_raw_dataset()

if df_raw is None:
    st.error("Dataset file (`freshretailnet_city03_dataset.csv`) not found. Please ensure it is in the same directory.")
    st.stop()

# -----------------------------------------------------------------------------
# 2. SIDEBAR FILTERS
# -----------------------------------------------------------------------------
st.sidebar.header("🔍 Data Filters")

# Provide defaults and allow reset
all_stores = sorted(df_raw['store_id'].unique())
min_date = df_raw['dt'].min().date()
max_date = df_raw['dt'].max().date()

if 'filters_reset' not in st.session_state:
    st.session_state['filters_reset'] = False

with st.sidebar.expander("Filter Options", expanded=True):
    selected_store = st.multiselect("Select Store ID(s):", options=all_stores, default=all_stores[:5], key='selected_store')
    if not selected_store:
        st.warning("Please select at least one Store ID. Defaulting to the first available store.")
        selected_store = [all_stores[0]]

    selected_dates = st.date_input(
        "Select Time Period (Date or Date range):",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
        key='selected_dates'
    )

    # quick summary
    st.markdown(f"**Stores selected:** {len(selected_store)}  ")
    sd = selected_dates
    st.markdown(f"**Period:** {sd[0]} to {sd[1]}")

# Insight view selector for dashboard rendering
# Render all dashboard views by default
dashboard_options = ["Univariate Analysis", "Bivariate Analysis", "Multivariate Insights"]
selected_dashboards = dashboard_options

if isinstance(selected_dates, (tuple, list)):
    if len(selected_dates) == 2:
        start_date, end_date = selected_dates
    else:
        start_date = end_date = selected_dates[0]
else:
    start_date = end_date = selected_dates

df_filtered = df_raw[
    (df_raw['store_id'].isin(selected_store)) &
    (df_raw['dt'].dt.date >= start_date) &
    (df_raw['dt'].dt.date <= end_date)
].copy()

# Sidebar: number of records removed per user request
# st.sidebar.markdown(f"**Number of records after filtering:** {df_filtered.shape[0]:,}")

# Define safe_eval_kpi helper function
def safe_eval_kpi(x):
    if isinstance(x, list):
        return x
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except:
            pass
    return [0]*24

# -----------------------------------------------------------------------------
# 3. METRICS & KPIS DISPLAY
# -----------------------------------------------------------------------------
st.divider()
st.subheader("📌 General Key Performance Indicators (KPIs)")

total_sales = df_filtered['sale_amount'].sum()
unique_products = df_filtered['product_id'].nunique()
unique_stores = df_filtered['store_id'].nunique()
avg_sales = df_filtered['sale_amount'].mean()
total_records = df_filtered.shape[0]

df_filtered['hours_stock_status'] = df_filtered['hours_stock_status'].apply(safe_eval_kpi)
df_filtered['oos_rate'] = df_filtered['hours_stock_status'].apply(lambda x: sum(1 for s in x if s == 0) / len(x) if len(x) > 0 else 0)
avg_oos_rate = df_filtered['oos_rate'].mean() * 100
total_oos_events = (df_filtered['oos_rate'] > 0).sum()

# Holiday and Activity impact
holiday_records = (df_filtered['holiday_flag'] == 1).sum()
activity_records = (df_filtered['activity_flag'] == 1).sum()
avg_temp = df_filtered['avg_temperature'].mean()

# Streamlined, currency symbols removed per user request. Show essential KPIs only.
col1, col2, col3 = st.columns(3)
# Add simple symbols/emojis in KPI labels only; remove icons adjacent to numeric values
st.markdown(
    """
    <style>
    .kpi-card {
        max-width: 300px;
        width: 100%;
        margin: 0 auto;
        padding: 1rem;
        border-radius: 18px;
        box-shadow: 0 12px 30px rgba(0,0,0,0.10);
        border: 1px solid rgba(0,0,0,0.08);
    }
    .kpi-card-1 {
        background: rgba(76, 175, 80, 0.15);
    }
    .kpi-card-2 {
        background: rgba(33, 150, 243, 0.15);
    }
    .kpi-card-3 {
        background: rgba(255, 152, 0, 0.15);
    }
    .kpi-title {
        font-size: 1.1rem;
        font-weight: 700;
        margin-bottom: 0.4rem;
    }
    .kpi-value {
        font-size: 2.4rem;
        font-weight: 800;
        line-height: 1.1;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
col1, col2, col3 = st.columns(3)
col1.markdown(
    f"""
    <div class='kpi-card kpi-card-1'>
        <div class='kpi-title'>💰 Total Revenue</div>
        <div class='kpi-value'>{total_sales:,.0f}</div>
    </div>
    """,
    unsafe_allow_html=True,
)
col2.markdown(
    f"""
    <div class='kpi-card kpi-card-2'>
        <div class='kpi-title'>🧾 Average Sale Value</div>
        <div class='kpi-value'>{avg_sales:,.2f}</div>
    </div>
    """,
    unsafe_allow_html=True,
)
col3.markdown(
    f"""
    <div class='kpi-card kpi-card-3'>
        <div class='kpi-title'>📑 Total Records</div>
        <div class='kpi-value'>{total_records:,}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

col4, col5, col6 = st.columns(3)
col4.markdown(
    f"""
    <div class='kpi-card' style="background: linear-gradient(135deg, #a8e6cf 0%, #dcedc1 100%);">
        <div class='kpi-title'>🧩 Unique Products</div>
        <div class='kpi-value'>{unique_products:,}</div>
    </div>
    """,
    unsafe_allow_html=True,
)
col5.markdown(
    f"""
    <div class='kpi-card' style="background: linear-gradient(135deg, #6fa8ff 0%, #c0dcff 100%);">
        <div class='kpi-title'>🏬 Active Stores</div>
        <div class='kpi-value'>{unique_stores:,}</div>
    </div>
    """,
    unsafe_allow_html=True,
)
col6.markdown(
    f"""
    <div class='kpi-card' style="background: linear-gradient(135deg, #ff7a7a 0%, #ff3b3b 100%);">
        <div class='kpi-title'>⚠️ Avg Out-of-Stock Rate</div>
        <div class='kpi-value'>{avg_oos_rate:.1f}%</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# 3.5 DASHBOARD NAVIGATION
# -----------------------------------------------------------------------------
st.divider()
st.subheader("📊 Choose Insight View")

# Định nghĩa các tab
tab_titles = ["Univariate Analysis", "Bivariate Analysis", "Multivariate Insights"]
tabs = st.tabs(tab_titles)
tab_objects = dict(zip(tab_titles, tabs))

# Actual dashboard content will be rendered after the chart helper functions are defined.
# -----------------------------------------------------------------------------
# 4. DASHBOARD RENDERING FUNCTIONS
# -----------------------------------------------------------------------------

def render_univariate(df_filtered):
    st.markdown("### Univariate Analysis — Exploring Key Features")

    df_eda = df_filtered.copy()
    sns.set_theme(style="whitegrid")
    fig_uni, axes = plt.subplots(2, 2, figsize=(18, 13))
    fig_uni.suptitle('Univariate Insights Dashboard', fontsize=16, fontweight='bold')

    sns.histplot(df_eda['sale_amount'], kde=True, ax=axes[0, 0], bins=30, color='skyblue')
    axes[0, 0].set_title('Distribution of Sale Amount', fontsize=13, fontweight='bold')
    axes[0, 0].set_xlabel('Sale Amount')
    axes[0, 0].set_ylabel('Frequency')
    median_sale = df_eda['sale_amount'].median()
    axes[0, 0].text(0.65, 0.9, f'Median Sale: {median_sale:.1f}\nNote: High variance observed.',
                    transform=axes[0, 0].transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))

    crosstab_data = pd.crosstab(df_eda['holiday_flag'], df_eda['activity_flag'])
    pie_data = pd.Series({
        'Holiday & Activity': crosstab_data.loc[1, 1] if (1 in crosstab_data.index and 1 in crosstab_data.columns) else 0,
        'Holiday Only': crosstab_data.loc[1, 0] if (1 in crosstab_data.index and 0 in crosstab_data.columns) else 0,
        'Activity Only': crosstab_data.loc[0, 1] if (0 in crosstab_data.index and 1 in crosstab_data.columns) else 0,
        'No Holiday & No Activity': crosstab_data.loc[0, 0] if (0 in crosstab_data.index and 0 in crosstab_data.columns) else 0
    })
    ordered_labels = ['No Holiday & No Activity', 'Activity Only', 'Holiday Only', 'Holiday & Activity']
    pie_data = pie_data.reindex(ordered_labels)
    colors = sns.color_palette('pastel')[0:len(pie_data)]

    if pie_data.sum() > 0:
        axes[0, 1].pie(
            pie_data.values,
            labels=[f'{label}\n({val:,} records)' for label, val in pie_data.items()],
            autopct='%1.1f%%',
            startangle=90,
            colors=colors,
            textprops={'fontsize': 10, 'fontweight': 'bold'},
            wedgeprops={'edgecolor': 'white', 'linewidth': 1.5}
        )
    else:
        axes[0, 1].text(0.5, 0.5, 'No data available', ha='center', va='center', fontsize=12, fontweight='bold')
    axes[0, 1].set_title('Proportion of Data by Holiday and Activity Status', fontsize=13, fontweight='bold')
    axes[0, 1].axis('equal')

    sns.histplot(df_eda['avg_temperature'], kde=True, ax=axes[1, 0], bins=30, color='lightcoral')
    axes[1, 0].set_title('Distribution of Average Temperature', fontsize=13, fontweight='bold')
    axes[1, 0].set_xlabel('Average Temperature (°C)')
    axes[1, 0].set_ylabel('Frequency')

    mg_id_counts = df_eda['management_group_id'].value_counts()
    mg_id_order_ascending = sorted(df_eda['management_group_id'].unique())
    palette = sns.color_palette("viridis", n_colors=len(mg_id_order_ascending))
    mg_id_sorted_by_count_desc = mg_id_counts.sort_values(ascending=False).index.tolist()
    color_map_by_count = {id_val: palette[i] for i, id_val in enumerate(mg_id_sorted_by_count_desc)}
    colors_for_ascending_order = [color_map_by_count[x] for x in mg_id_order_ascending]

    sns.countplot(x='management_group_id', data=df_eda, ax=axes[1, 1],
                  order=mg_id_order_ascending,
                  palette=colors_for_ascending_order,
                  hue='management_group_id', legend=False)
    axes[1, 1].set_title('Count of Management Group ID', fontsize=13, fontweight='bold')
    axes[1, 1].set_xlabel('Management Group ID')
    axes[1, 1].set_ylabel('Count')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    st.pyplot(fig_uni)


def render_bivariate(df_filtered):
    st.markdown("### Bivariate Analysis — Inventory Availability & Stockout Pattern Analysis")

    def safe_eval(x):
        if isinstance(x, list):
            return x
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, str):
            try:
                return ast.literal_eval(x)
            except:
                pass
        return [0]*24

    df_bivar_oos = df_filtered.copy()
    df_bivar_oos['dt'] = pd.to_datetime(df_bivar_oos['dt'])
    df_bivar_oos['hours_stock_status'] = df_bivar_oos['hours_stock_status'].apply(safe_eval)
    df_bivar_oos['stock_hour6_22_cnt'] = df_bivar_oos['hours_stock_status'].apply(lambda x: sum(1 for status in x[6:22] if status == 0))
    total_hours_window = 16
    df_bivar_oos['oos_hours_per_record'] = total_hours_window - df_bivar_oos['stock_hour6_22_cnt']
    hours_df = pd.DataFrame(df_bivar_oos['hours_stock_status'].tolist(), columns=[f'{i}h' for i in range(24)])

    sns.set_theme(style="whitegrid")
    fig = plt.figure(figsize=(20, 16), constrained_layout=True)
    gs = GridSpec(2, 2, figure=fig)

    ax1 = fig.add_subplot(gs[0, 0])
    hourly_oos_rate = hours_df[[f'{i}h' for i in range(6, 22)]].mean()
    sns.lineplot(x=hourly_oos_rate.index, y=hourly_oos_rate.values, marker='o', color='red', linewidth=2.5, ax=ax1)
    ax1.fill_between(hourly_oos_rate.index, hourly_oos_rate.values, color='red', alpha=0.1)
    ax1.set_title('Hourly Out-of-Stock Trend (6h-21h)', fontsize=14, fontweight='bold', pad=20)
    ax1.set_xlabel('Time Window')
    ax1.set_ylabel('Mean OOS Rate')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    ax2 = fig.add_subplot(gs[0, 1])
    correlation_matrix = df_bivar_oos.pivot_table(index='holiday_flag', columns='activity_flag', values='oos_hours_per_record', aggfunc='mean')
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="YlOrBr", cbar_kws={'label': 'Mean OOS Hours', 'shrink': 0.8}, ax=ax2)
    ax2.set_title('Strategic Impact of Promotions & Holidays on OOS', fontsize=14, fontweight='bold', pad=20)
    ax2.set_xticklabels(['No Promotion', 'With Promotion'], rotation=0)
    ax2.set_yticklabels(['Regular Day', 'Holiday'], rotation=0)
    ax2.set_xlabel('Promotion Status')
    ax2.set_ylabel('Holiday Status')
    for spine in ax2.spines.values():
        spine.set_visible(False)

    ax3 = fig.add_subplot(gs[1, 0])
    bins_5 = np.arange(0, 1.05, 0.05)
    labels_5 = [f'{int(i*100)}-{int((i+0.05)*100)}%' for i in bins_5[:-1]]
    df_bivar_oos['discount_grp_5'] = pd.cut(df_bivar_oos['discount'], bins=bins_5, labels=labels_5)
    mean_oos_by_discount = df_bivar_oos[df_bivar_oos['discount'] <= 1.0].groupby('discount_grp_5', observed=False)['oos_hours_per_record'].mean().reset_index()
    sns.lineplot(x='discount_grp_5', y='oos_hours_per_record', data=mean_oos_by_discount, marker='s', markersize=6, color='teal', linewidth=2, ax=ax3)
    ax3.fill_between(range(len(mean_oos_by_discount)), mean_oos_by_discount['oos_hours_per_record'], color='teal', alpha=0.1)
    ax3.set_title('Price Elasticity & Stock Availability Correlation', fontsize=14, fontweight='bold', pad=20)
    ax3.tick_params(axis='x', rotation=90)
    ax3.set_xlabel('Discount Depth')
    ax3.set_ylabel('Mean OOS Hours')
    ax3.set_ylim(0, 16)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)

    ax4 = fig.add_subplot(gs[1, 1])
    temp_bins_impact = [0, 22, 28, 100]
    temp_labels_impact = ['Cool (<22°C)', 'Moderate (22-28°C)', 'Hot (>28°C)']
    df_bivar_oos['temp_impact'] = pd.cut(df_bivar_oos['avg_temperature'], bins=temp_bins_impact, labels=temp_labels_impact)
    temp_stats = df_bivar_oos.groupby('temp_impact', observed=False)['oos_hours_per_record'].mean().reset_index()
    overall_mean = df_bivar_oos['oos_hours_per_record'].mean()
    custom_colors = {'Cool (<22°C)': 'skyblue', 'Moderate (22-28°C)': 'orange', 'Hot (>28°C)': 'crimson'}
    sns.barplot(x='temp_impact', y='oos_hours_per_record', data=temp_stats, palette=custom_colors, hue='temp_impact', legend=False, ax=ax4)
    ax4.axhline(overall_mean, color='black', linestyle='--', alpha=0.6, label=f'System Avg: {overall_mean:.2f}h')
    ax4.legend()
    ax4.set_title('Environmental Impact on Inventory Stability', fontsize=14, fontweight='bold', pad=20)
    ax4.set_xlabel('Temperature Zones')
    ax4.set_ylabel('Mean OOS Hours')
    ax4.set_ylim(0, 6)
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)

    plt.tight_layout()
    st.pyplot(fig)


def render_multivariate(df_filtered):
    st.markdown("### Multivariate Insights — Category & Environmental Impact")

    def safe_eval_multi(x):
        if isinstance(x, list):
            return x
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, str):
            try:
                return ast.literal_eval(x)
            except:
                pass
        return [0]*24

    df_multi = df_filtered.copy()
    df_multi['hours_stock_status'] = df_multi['hours_stock_status'].apply(safe_eval_multi)
    df_multi['stock_hour6_21_cnt'] = df_multi['hours_stock_status'].apply(lambda x: sum(1 for status in x[6:22] if status == 0))
    total_hours_in_window_multi = 16
    df_multi['oos_hours_per_record'] = total_hours_in_window_multi - df_multi['stock_hour6_21_cnt']
    df_multi['day_of_week'] = df_multi['dt'].dt.day_name()

    all_categories = sorted([x for x in df_multi['first_category_id'].unique() if pd.notna(x)])
    selected_categories = st.multiselect("Select First Category ID(s) to display:", options=all_categories, default=all_categories[:5])

    if selected_categories:
        df_multi = df_multi[df_multi['first_category_id'].isin(selected_categories)]

    st.markdown(f"**Number of records for multivariate analysis after category filtering:** {df_multi.shape[0]:,}")

    if df_multi.shape[0] == 0:
        st.warning("No data matches the selected Category ID(s). Please select more categories.")
    else:
        sns.set_theme(style="whitegrid")
        fig_multi, axes_multi = plt.subplots(2, 1, figsize=(22, 18))
        fig_multi.suptitle('Hourly Out-of-Stock Prediction for Fresh Food Retail\n(Operating Scale Window: 6h00 - 21h59)', fontsize=16, fontweight='bold', y=1.00)

        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        df_multi['day_of_week'] = pd.Categorical(df_multi['day_of_week'], categories=day_order, ordered=True)

        avg_oos_cat_day = df_multi.groupby(['first_category_id', 'day_of_week'], observed=False)['oos_hours_per_record'].mean().unstack()

        sns.heatmap(
            avg_oos_cat_day,
            annot=True,
            fmt=".1f",
            cmap="YlGnBu",
            linewidths=.5,
            ax=axes_multi[0]
        )
        axes_multi[0].set_title('Product OOS Hours Across Days of Week', fontsize=14, fontweight='bold')
        axes_multi[0].set_xlabel('Day of Week', fontsize=11)
        axes_multi[0].set_ylabel('First Category ID', fontsize=11)
        axes_multi[0].tick_params(axis='x', rotation=45)
        axes_multi[0].tick_params(axis='y', rotation=0)

        if 'temperature_range' not in df_multi.columns and 'avg_temperature' in df_multi.columns:
            temperature_bins = [0, 14, 18, 22, 26, 30, np.inf]
            temperature_labels = ['<14°C', '14-18°C', '18-22°C', '22-26°C', '26-30°C', '>30°C']
            df_multi['temperature_range'] = pd.cut(df_multi['avg_temperature'], bins=temperature_bins, labels=temperature_labels, right=False)

        avg_oos_temp_cat = df_multi.groupby(['temperature_range', 'first_category_id'], observed=False)['oos_hours_per_record'].mean().reset_index()
        avg_oos_temp_cat['first_category_id'] = avg_oos_temp_cat['first_category_id'].astype(str)

        sns.lineplot(
            x='temperature_range',
            y='oos_hours_per_record',
            hue='first_category_id',
            data=avg_oos_temp_cat,
            marker='o',
            ax=axes_multi[1],
            palette='tab20'
        )
        axes_multi[1].set_title('Product OOS Hours Across Temperature Ranges', fontsize=14, fontweight='bold')
        axes_multi[1].set_xlabel('Temperature Range', fontsize=11)
        axes_multi[1].set_ylabel('Average Out-of-Stock Hours', fontsize=11)
        axes_multi[1].tick_params(axis='x', rotation=15)
        axes_multi[1].legend(title='First Category ID', bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout(rect=[0, 0.03, 1, 0.98])
        st.pyplot(fig_multi)

# -----------------------------------------------------------------------------
# 5. RENDER SELECTED DASHBOARD
# -----------------------------------------------------------------------------

dashboard_map = {
    "Univariate Analysis": render_univariate,
    "Bivariate Analysis": render_bivariate,
    "Multivariate Insights": render_multivariate
}

if not selected_dashboards:
    st.warning("Please select at least one view.")
else:
    for dashboard_name in selected_dashboards:
        with tab_objects[dashboard_name]:
            dashboard_map[dashboard_name](df_filtered)
with st.sidebar:
    st.link_button("🔗 Open Forecast App", "https://predicthourstock.streamlit.app/", use_container_width=True)
#
# 
# 
# 
st.divider()
# 2. Main Area: Use Tabs to divide sections
st.subheader("🤖 Models")

model_options = [
    "XGBoost Model",
    "LightGBM Model",
    "Random Forest Model",
    "Logistic Regression Model",
    "MLP Model"
]

selected_models = st.multiselect(
    "Choose models (up to 2):",
    options=model_options,
    default=model_options[:2],
    help="Select up to 2 models to view the detailed tabs below."
)

if not selected_models:
    st.warning("Please select at least one model.")
    selected_models = [model_options[0]]
elif len(selected_models) > 2:
    st.warning("You can only display up to 2 models at once. Showing the first 2 models.")
    selected_models = selected_models[:2]

tabs = st.tabs(["Performance Metrics", "Feature Importance", "Confusion Matrix"])

model_color_map = {
    "XGBoost Model": {"bar_color": "#E87722", "cmap": "Oranges", "data_key": "XGBoost"},
    "LightGBM Model": {"bar_color": "#2E5090", "cmap": "Blues", "data_key": "LightGBM"},
    "Random Forest Model": {"bar_color": "#228B22", "cmap": "Greens", "data_key": "Random Forest"},
    "Logistic Regression Model": {"bar_color": "#6E44FF", "cmap": "Purples", "data_key": "Logistic"},
    "MLP Model": {"bar_color": "#D62828", "cmap": "coolwarm", "data_key": "MLP"}
}

for tab_index, tab in enumerate(tabs):
    with tab:
        cols = st.columns(len(selected_models) if len(selected_models) > 1 else 1)
        for idx, model_name in enumerate(selected_models[:2]):
            with cols[idx]:
                if tab_index == 0:
                    st.markdown(f"### {model_name}")
                    # Insert known performance metrics for XGBoost and LightGBM models
                    if model_name == "XGBoost Model":
                        metrics_data = {
                            "Metric": ["Total Test Samples", "Accuracy", "Precision", "Recall", "F1-Score"],
                            "Value": ["18,536", "80.20%", "55.07%", "69.72%", "61.36%"]
                        }
                        st.dataframe(metrics_data, use_container_width=True, hide_index=True)
                    elif model_name == "LightGBM Model":
                        metrics_data = {
                            "Metric": ["Total Test Samples", "Accuracy", "Precision", "Recall", "F1-Score"],
                            "Value": ["18,536", "79.93%", "55.70%", "69.00%", "61.34%"]
                        }
                        st.dataframe(metrics_data, use_container_width=True, hide_index=True)
                    else:
                        # Add known validation results for additional models
                        if model_name == "Random Forest Model":
                            metrics_data = {
                                "Metric": ["Total Test Samples", "Accuracy", "Precision", "Recall", "F1-Score"],
                                "Value": ["18,536", "78.80%", "52.57%", "67.52%", "58.87%"]
                            }
                            st.dataframe(metrics_data, use_container_width=True, hide_index=True)
                        elif model_name == "Logistic Regression Model":
                            metrics_data = {
                                "Metric": ["Total Test Samples", "Accuracy", "Precision", "Recall", "F1-Score"],
                                "Value": ["18,536", "75.72%", "46.87%", "59.82%", "52.17%"]
                            }
                            st.dataframe(metrics_data, use_container_width=True, hide_index=True)
                        elif model_name == "MLP Model":
                            metrics_data = {
                                "Metric": ["Total Test Samples", "Accuracy", "Precision", "Recall", "F1-Score"],
                                "Value": ["18,536", "75.58%", "46.62%", "60.39%", "52.41%"]
                            }
                            st.dataframe(metrics_data, use_container_width=True, hide_index=True)
                        else:
                            st.write("Display the model performance metrics (accuracy, precision, recall, F1, AUC) when data is available.")
                elif tab_index == 1:
                    st.markdown(f"### {model_name}")
                    bar_color = model_color_map.get(model_name, {}).get("bar_color", '#1f77b4')
                    if model_name == "Logistic Regression Model":
                        # Feature importance data for Logistic Regression
                        feature_importance_data = {
                            "Feature": ["oos_rate_lag1_day", "discount", "activity_flag", "is_weekend", "avg_wind_level", 
                                       "holiday_flag", "precpt", "avg_temperature", "management_group_id", "first_category_id",
                                       "avg_humidity", "second_category_id", "third_category_id", "product_id", "store_id"],
                            "Importance (%)": [70.0, 10.5, 6.8, 5.2, 2.8, 2.1, 1.5, 0.8, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
                        }
                        fig, ax = plt.subplots(figsize=(10, 6))
                        df_importance = pd.DataFrame(feature_importance_data)
                        df_importance = df_importance.sort_values("Importance (%)", ascending=True)
                        ax.barh(df_importance["Feature"], df_importance["Importance (%)"], color='#0056B3')
                        ax.set_xlabel("Feature contribution to model decision (%)", fontsize=10)
                        ax.set_ylabel("Input features", fontsize=10)
                        ax.set_title("Feature importance for 16-hour stockout prediction", fontsize=11, fontweight='bold')
                        plt.tight_layout()
                        st.pyplot(fig, use_container_width=True)
                    elif model_name == "MLP Model":
                        # Feature importance data for MLP (example/permutation importance)
                        feature_importance_data = {
                            "Feature": ["oos_rate_lag1_day", "product_id", "third_category_id", "first_category_id",
                                         "discount", "activity_flag", "is_weekend", "avg_temperature",
                                         "avg_humidity", "management_group_id", "second_category_id",
                                         "store_id", "holiday_flag", "precpt", "avg_wind_level"],
                            "Importance (%)": [53.3, 16.0, 12.5, 5.0, 3.5, 2.2, 1.8, 1.0, 0.6, 0.4, 0.3, 0.2, 0.15, 0.05, 0.05]
                        }
                        fig, ax = plt.subplots(figsize=(10, 6))
                        df_importance = pd.DataFrame(feature_importance_data)
                        df_importance = df_importance.sort_values("Importance (%)", ascending=True)
                        ax.barh(df_importance["Feature"], df_importance["Importance (%)"], color=bar_color)
                        ax.set_xlabel("Feature contribution to model decision (%)", fontsize=10)
                        ax.set_ylabel("Input features", fontsize=10)
                        ax.set_title("Relative feature contribution (MLP)", fontsize=11, fontweight='bold')
                        plt.tight_layout()
                        st.pyplot(fig, use_container_width=True)
                    elif model_name == "XGBoost Model":
                        # Feature importance data for XGBoost
                        feature_importance_data = {
                            "Feature": ["oos_rate_lag1_day", "discount", "product_id", "avg_temperature", "third_category_id",
                                         "second_category_id", "precpt", "avg_humidity", "holiday_flag", "management_group_id",
                                         "avg_wind_level", "first_category_id", "is_weekend", "store_id", "activity_flag"],
                            "Importance (%)": [25.0, 18.0, 14.5, 11.0, 7.2, 6.8, 4.5, 4.0, 3.0, 2.0, 1.5, 1.0, 0.3, 0.2, 0.1]
                        }
                        fig, ax = plt.subplots(figsize=(10, 6))
                        df_importance = pd.DataFrame(feature_importance_data)
                        df_importance = df_importance.sort_values("Importance (%)", ascending=True)
                        ax.barh(df_importance["Feature"], df_importance["Importance (%)"], color=bar_color)
                        ax.set_xlabel("Feature contribution to model decision (%)", fontsize=10)
                        ax.set_ylabel("Input features", fontsize=10)
                        ax.set_title("Feature importance for XGBoost", fontsize=11, fontweight='bold')
                        plt.tight_layout()
                        st.pyplot(fig, use_container_width=True)
                    elif model_name == "Random Forest Model":
                        # Feature importance data for Random Forest
                        feature_importance_data = {
                            "Feature": ["oos_rate_lag1_day", "product_id", "discount", "avg_temperature", "third_category_id",
                                        "second_category_id", "avg_humidity", "first_category_id", "precpt", "management_group_id",
                                        "holiday_flag", "avg_wind_level", "is_weekend", "store_id", "activity_flag"],
                            "Importance (%)": [28.5, 16.8, 15.2, 11.5, 8.3, 7.5, 5.2, 4.0, 1.8, 0.8, 0.2, 0.1, 0.05, 0.05, 0.03]
                        }
                        fig, ax = plt.subplots(figsize=(10, 6))
                        df_importance = pd.DataFrame(feature_importance_data)
                        df_importance = df_importance.sort_values("Importance (%)", ascending=True)
                        ax.barh(df_importance["Feature"], df_importance["Importance (%)"], color=bar_color)
                        ax.set_xlabel("Feature contribution to model decision (%)", fontsize=10)
                        ax.set_ylabel("Input features", fontsize=10)
                        ax.set_title("Feature importance for Random Forest", fontsize=11, fontweight='bold')
                        plt.tight_layout()
                        st.pyplot(fig, use_container_width=True)
                    elif model_name == "LightGBM Model":
                        # Feature importance data for LightGBM
                        feature_importance_data = {
                            "Feature": ["discount", "product_id", "avg_temperature", "third_category_id", "second_category_id",
                                        "precpt", "avg_humidity", "oos_rate_lag1_day", "first_category_id", "management_group_id",
                                        "holiday_flag", "avg_wind_level", "is_weekend", "store_id", "activity_flag"],
                            "Importance (%)": [20.5, 15.2, 12.8, 10.2, 9.8, 7.5, 7.2, 6.5, 5.0, 2.8, 2.1, 0.8, 0.3, 0.2, 0.1]
                        }
                        fig, ax = plt.subplots(figsize=(10, 6))
                        df_importance = pd.DataFrame(feature_importance_data)
                        df_importance = df_importance.sort_values("Importance (%)", ascending=True)
                        ax.barh(df_importance["Feature"], df_importance["Importance (%)"], color=bar_color)
                        ax.set_xlabel("Feature contribution to model decision (%)", fontsize=10)
                        ax.set_ylabel("Input features", fontsize=10)
                        ax.set_title("Feature importance for LightGBM", fontsize=11, fontweight='bold')
                        plt.tight_layout()
                        st.pyplot(fig, use_container_width=True)
                    else:
                        st.write("Table/chart of the model's feature importance.")
                elif tab_index == 2:
                    st.markdown(f"### {model_name}")
                    # Add 16-hour confusion matrices
                    st.markdown("### Confusion Matrix by Operating Hours")
                    
                    # Define confusion matrices for 16 hours
                    hours_data = {
                        "6h": {
                            "XGBoost": [[15501, 971], [572, 1492]],
                            "Random Forest": [[15261, 1211], [553, 1511]],
                            "LightGBM": [[15434, 1038], [493, 1571]],
                            "MLP": [[15217, 1255], [751, 1313]],
                            "Logistic": [[15495, 977], [993, 1071]]
                        },
                        "7h": {
                            "XGBoost": [[15404, 1215], [339, 1578]],
                            "Random Forest": [[15278, 1341], [396, 1521]],
                            "LightGBM": [[15834, 785], [466, 1451]],
                            "MLP": [[15443, 1176], [768, 1149]],
                            "Logistic": [[15553, 1066], [870, 1047]]
                        },
                        "8h": {
                            "XGBoost": [[15439, 1090], [463, 1544]],
                            "Random Forest": [[15349, 1180], [541, 1466]],
                            "LightGBM": [[15595, 934], [476, 1531]],
                            "MLP": [[15249, 1280], [834, 1173]],
                            "Logistic": [[15481, 1048], [953, 1054]]
                        },
                        "9h": {
                            "XGBoost": [[15063, 1236], [591, 1646]],
                            "Random Forest": [[15179, 1120], [761, 1476]],
                            "LightGBM": [[15335, 964], [682, 1555]],
                            "MLP": [[14952, 1347], [942, 1295]],
                            "Logistic": [[15268, 1031], [1149, 1088]]
                        },
                        "10h": {
                            "XGBoost": [[14831, 1136], [911, 1658]],
                            "Random Forest": [[14737, 1230], [990, 1579]],
                            "LightGBM": [[14817, 1150], [897, 1672]],
                            "MLP": [[14260, 1707], [1123, 1446]],
                            "Logistic": [[14455, 1512], [1259, 1310]]
                        },
                        "11h": {
                            "XGBoost": [[14175, 1377], [1138, 1846]],
                            "Random Forest": [[14263, 1289], [1280, 1704]],
                            "LightGBM": [[14410, 1142], [1245, 1739]],
                            "MLP": [[13821, 1731], [1430, 1554]],
                            "Logistic": [[13897, 1655], [1517, 1467]]
                        },
                        "12h": {
                            "XGBoost": [[13364, 1786], [1277, 2109]],
                            "Random Forest": [[13639, 1511], [1532, 1854]],
                            "LightGBM": [[13403, 1747], [1320, 2066]],
                            "MLP": [[13451, 1699], [1782, 1604]],
                            "Logistic": [[11718, 3432], [1239, 2147]]
                        },
                        "13h": {
                            "XGBoost": [[12964, 1863], [1474, 2235]],
                            "Random Forest": [[12502, 2325], [1454, 2255]],
                            "LightGBM": [[13094, 1733], [1577, 2132]],
                            "MLP": [[12488, 2339], [1753, 1956]],
                            "Logistic": [[11548, 3279], [1439, 2270]]
                        },
                        "14h": {
                            "XGBoost": [[12337, 2181], [1551, 2467]],
                            "Random Forest": [[11965, 2553], [1548, 2470]],
                            "LightGBM": [[12669, 1849], [1761, 2257]],
                            "MLP": [[10822, 3696], [1619, 2399]],
                            "Logistic": [[10632, 3886], [1381, 2637]]
                        },
                        "15h": {
                            "XGBoost": [[11731, 2469], [1632, 2704]],
                            "Random Forest": [[11511, 2689], [1711, 2625]],
                            "LightGBM": [[11592, 2608], [1647, 2689]],
                            "MLP": [[11104, 3096], [1952, 2384]],
                            "Logistic": [[10817, 3383], [1722, 2614]]
                        },
                        "16h": {
                            "XGBoost": [[10919, 2787], [1791, 3039]],
                            "Random Forest": [[10511, 3195], [1789, 3041]],
                            "LightGBM": [[10254, 3452], [1610, 3220]],
                            "MLP": [[9580, 4126], [1867, 2963]],
                            "Logistic": [[9952, 3754], [1787, 3043]]
                        },
                        "17h": {
                            "XGBoost": [[9561, 3562], [1705, 3708]],
                            "Random Forest": [[8822, 4301], [1549, 3864]],
                            "LightGBM": [[8847, 4276], [1460, 3953]],
                            "MLP": [[8948, 4175], [2100, 3313]],
                            "Logistic": [[9434, 3689], [2061, 3352]]
                        },
                        "18h": {
                            "XGBoost": [[8277, 4284], [1488, 4487]],
                            "Random Forest": [[8427, 4134], [1729, 4246]],
                            "LightGBM": [[7931, 4630], [1401, 4574]],
                            "MLP": [[8510, 4051], [2240, 3735]],
                            "Logistic": [[8560, 4001], [2151, 3824]]
                        },
                        "19h": {
                            "XGBoost": [[7688, 4413], [1476, 4959]],
                            "Random Forest": [[7539, 4562], [1643, 4792]],
                            "LightGBM": [[7478, 4623], [1444, 4991]],
                            "MLP": [[7192, 4909], [1919, 4516]],
                            "Logistic": [[7949, 4152], [2242, 4193]]
                        },
                        "20h": {
                            "XGBoost": [[7607, 4130], [1681, 5118]],
                            "Random Forest": [[7013, 4724], [1621, 5178]],
                            "LightGBM": [[7642, 4095], [1743, 5056]],
                            "MLP": [[5942, 5795], [1663, 5136]],
                            "Logistic": [[5996, 5741], [1684, 5115]]
                        },
                        "21h": {
                            "XGBoost": [[6722, 4619], [1525, 5670]],
                            "Random Forest": [[6612, 4729], [1689, 5506]],
                            "LightGBM": [[6635, 4706], [1566, 5629]],
                            "MLP": [[6020, 5321], [1982, 5213]],
                            "Logistic": [[6567, 4774], [2196, 4999]]
                        }
                    }
                    
                    model_props = model_color_map.get(model_name, {"cmap": "Blues", "data_key": "LightGBM"})
                    cmap = model_props["cmap"]
                    data_key = model_props["data_key"]

                    # Create a 4x4 grid of confusion matrices
                    fig, axes = plt.subplots(4, 4, figsize=(16, 14))
                    fig.suptitle(f"Confusion Matrices by Operating Hours - {model_name}", fontsize=14, fontweight='bold')
                    
                    hours_list = list(hours_data.keys())
                    for idx, hour in enumerate(hours_list):
                        row = idx // 4
                        col = idx % 4
                        ax = axes[row, col]
                        
                        confusion_matrix = np.array(hours_data[hour][data_key])
                        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap=cmap, cbar=False,
                                    xticklabels=['Out of stock', 'In stock'],
                                    yticklabels=['Out of stock', 'In stock'],
                                    ax=ax, annot_kws={"size": 8})
                        ax.set_xlabel("Predicted", fontsize=8, fontweight='bold')
                        ax.set_ylabel("Actual", fontsize=8, fontweight='bold')
                        ax.set_title(f"{hour}", fontsize=9, fontweight='bold')
                    
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                
                elif tab_index == 3:
                    st.markdown(f"### {model_name}")
                    # SHAP Summary Plot
                    fig, ax = plt.subplots(figsize=(10, 8))
                    
                    # SHAP summary data
                    shap_data = {
                        "Feature": ["oos_rate_lag1_day", "second_category_id", "discount", "product_id", "third_category_id",
                                   "management_group_id", "first_category_id", "avg_temperature", "activity_flag", "precpt",
                                   "store_id", "holiday_flag", "avg_humidity", "avg_wind_level", "is_weekend"],
                        "SHAP_values": [
                            [-3, -2, -1, 0, 1, 2, 3],
                            [-3, -2, -1, 0, 1, 2, 3],
                            [-3, -2, -1, 0, 1, 2, 3],
                            [-3, -2, -1, 0, 1, 2, 3],
                            [-3, -2, -1, 0, 1, 2, 3],
                            [-3, -2, -1, 0, 1, 2, 3],
                            [-3, -2, -1, 0, 1, 2, 3],
                            [-3, -2, -1, 0, 1, 2, 3],
                            [-3, -2, -1, 0, 1, 2, 3],
                            [-3, -2, -1, 0, 1, 2, 3],
                            [-3, -2, -1, 0, 1, 2, 3],
                            [-3, -2, -1, 0, 1, 2, 3],
                            [-3, -2, -1, 0, 1, 2, 3],
                            [-3, -2, -1, 0, 1, 2, 3],
                            [-3, -2, -1, 0, 1, 2, 3]
                        ]
                    }
                    
                    # SHAP plot removed
                    st.write("SHAP plot removed")
