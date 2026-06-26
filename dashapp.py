import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import os
import joblib
from datetime import datetime

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & LOAD DATA
# -----------------------------------------------------------------------------
st.set_page_config(page_title="FreshRetailNet - City 03 Dashboard", layout="wide")

st.title("📊 FreshRetail Analytics Dashboard - City ID 03")
st.markdown("Use the filters on the left sidebar to analyze data by **Store ID** and **Time Period**.")

@st.cache_data
def load_data():
    filename = "freshretailnet_city03_dataset.csv"
    df = pd.read_csv(filename)
    # Convert date column 'dt' to datetime format
    df['dt'] = pd.to_datetime(df['dt'])
    return df

try:
    df_raw = load_data()
except Exception as e:
    st.error(f"Dataset file (`freshretailnet_city03_dataset.csv`) not found. Please export it from your notebook. Details: {e}")
    st.stop()

# -----------------------------------------------------------------------------
# 2. SIDEBAR FILTERS
# -----------------------------------------------------------------------------
st.sidebar.header("🔍 Data Filters")

# Filter by Store ID
all_stores = sorted(df_raw['store_id'].unique())
selected_store = st.sidebar.multiselect("Select Store ID(s):", options=all_stores, default=all_stores[:5])

# Filter by Date (supports single day or date range selection)
min_date = df_raw['dt'].min().date()
max_date = df_raw['dt'].max().date()

selected_dates = st.sidebar.date_input(
    "Select Time Period (Date or Date range):",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

# Auto-resolve logic for single date or date range selection
if isinstance(selected_dates, tuple) or isinstance(selected_dates, list):
    if len(selected_dates) == 2:
        start_date, end_date = selected_dates
    else:
        start_date = end_date = selected_dates[0]
else:
    start_date = end_date = selected_dates

# Apply filters to DataFrame
df_filtered = df_raw[
    (df_raw['store_id'].isin(selected_store)) &
    (df_raw['dt'].dt.date >= start_date) &
    (df_raw['dt'].dt.date <= end_date)
]

st.sidebar.markdown(f"**Number of records after filtering:** {df_filtered.shape[0]:,}")

# -----------------------------------------------------------------------------
# 3. METRICS & KPIS DISPLAY
# -----------------------------------------------------------------------------
st.divider()
st.subheader("📌 General Key Performance Indicators (KPIs)")

total_sales = df_filtered['sale_amount'].sum()
unique_products = df_filtered['product_id'].nunique()
unique_stores = df_filtered['store_id'].nunique()

col1, col2, col3 = st.columns(3)
col1.metric("Total Revenue (Sale Amount)", f"{total_sales:,.2f}")
col2.metric("Unique Products Count", f"{unique_products:,}")
col3.metric("Active Stores Count", f"{unique_stores:,}")

# -----------------------------------------------------------------------------
# 4. UNIVARIATE ANALYSIS (MATPLOTLIB/SEABORN)
# -----------------------------------------------------------------------------
st.divider()
st.subheader("📈 1. Univariate Analysis")

df_eda = df_filtered.copy()
sns.set_theme(style="whitegrid")
fig_uni, axes = plt.subplots(2, 2, figsize=(18, 13))
fig_uni.suptitle('Univariate Analysis Dashboard', fontsize=16, fontweight='bold')

# Chart 1: Distribution of Sale Amount
sns.histplot(df_eda['sale_amount'], kde=True, ax=axes[0, 0], bins=30, color='skyblue')
axes[0, 0].set_title('1. Distribution of Sale Amount', fontsize=13, fontweight='bold')
axes[0, 0].set_xlabel('Sale Amount')
axes[0, 0].set_ylabel('Frequency')
median_sale = df_eda['sale_amount'].median()
axes[0, 0].text(0.65, 0.9, f'Median Sale: {median_sale:.1f}\nNote: High variance observed.',
                transform=axes[0, 0].transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))

# Chart 2: Pie Chart - Holiday and Activity Status Distribution
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

axes[0, 1].pie(
    pie_data.values,
    labels=[f'{label}\n({val:,} records)' for label, val in pie_data.items()],
    autopct='%1.1f%%',
    startangle=90,
    colors=colors,
    textprops={'fontsize': 10, 'fontweight': 'bold'},
    wedgeprops={'edgecolor': 'white', 'linewidth': 1.5}
)
axes[0, 1].set_title('2. Proportion of Data by Holiday and Activity Status', fontsize=13, fontweight='bold')
axes[0, 1].axis('equal')

# Chart 3: Distribution of Average Temperature
sns.histplot(df_eda['avg_temperature'], kde=True, ax=axes[1, 0], bins=30, color='lightcoral')
axes[1, 0].set_title('3. Distribution of Average Temperature', fontsize=13, fontweight='bold')
axes[1, 0].set_xlabel('Average Temperature (°C)')
axes[1, 0].set_ylabel('Frequency')

# Chart 4: Countplot of Management Group ID
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
axes[1, 1].set_title('4. Count of Management Group ID', fontsize=13, fontweight='bold')
axes[1, 1].set_xlabel('Management Group ID')
axes[1, 1].set_ylabel('Count')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
st.pyplot(fig_uni)

# -----------------------------------------------------------------------------
# 5. BIVARIATE ANALYSIS - OOS DIAGNOSTIC INSIGHTS (SEABORN/MATPLOTLIB)
# -----------------------------------------------------------------------------
st.divider()
st.subheader("📈 2. Bivariate Analysis: Out-of-Stock (OOS) Diagnostic Insights")

df_bivar_oos = df_filtered.copy()

# Date parsing cho OOS
df_bivar_oos['dt'] = pd.to_datetime(df_bivar_oos['dt'])
df_bivar_oos['day_of_week'] = df_bivar_oos['dt'].dt.day_name()
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
df_bivar_oos['day_of_week'] = pd.Categorical(df_bivar_oos['day_of_week'], categories=day_order, ordered=True)

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

df_bivar_oos['hours_stock_status'] = df_bivar_oos['hours_stock_status'].apply(safe_eval)

# In-stock calculations cho khung 6h - 21h59 (indices 6 to 21 inclusive, total 16 hours)
df_bivar_oos['stock_hour6_21_cnt'] = df_bivar_oos['hours_stock_status'].apply(lambda x: sum(1 for status in x[6:22] if status == 0))
total_hours_in_window_new = 16
df_bivar_oos['oos_hours_per_record'] = total_hours_in_window_new - df_bivar_oos['stock_hour6_21_cnt']

hours_df = pd.DataFrame(df_bivar_oos['hours_stock_status'].tolist(), columns=[f'{i}h' for i in range(24)])

sns.set_theme(style="whitegrid")
fig_bivar_oos_plots, axes_oos = plt.subplots(2, 2, figsize=(20, 15))
fig_bivar_oos_plots.suptitle('Bivariate Analysis: Factors Influencing Out-of-Stock Hours (6h - 21h59 window)', fontsize=16, fontweight='bold', y=1.02)

# Plot 1: Average Out-of-Stock Rate by Hour (6h - 21h59 window)
hourly_oos_rate = hours_df[[f'{i}h' for i in range(6, 22)]].mean()
sns.lineplot(x=hourly_oos_rate.index, y=hourly_oos_rate.values, marker='o', color='red', linewidth=2.5, ax=axes_oos[0, 0])
axes_oos[0, 0].set_title('1. Average Out-of-Stock Rate by Hour (6h - 21h59)', fontsize=14, fontweight='bold')
axes_oos[0, 0].set_xlabel('Hour of the Day (6h - 21h59)', fontsize=11)
axes_oos[0, 0].set_ylabel('Out-of-Stock Rate (0 - 1)')
axes_oos[0, 0].tick_params(axis='x', rotation=45)

# Plot 2: Average OOS Hours by Event Category
def get_event_category(row):
    if row['activity_flag'] == 1 and row['holiday_flag'] == 1:
        return 'Activity & Holiday'
    elif row['activity_flag'] == 1 and row['holiday_flag'] == 0:
        return 'Activity Only'
    elif row['activity_flag'] == 0 and row['holiday_flag'] == 1:
        return 'Holiday Only'
    else:
        return 'No Activity/Holiday'

df_bivar_oos['event_category'] = df_bivar_oos.apply(get_event_category, axis=1)
event_category_order = ['No Activity/Holiday', 'Activity Only', 'Holiday Only', 'Activity & Holiday']

avg_oos_by_event = df_bivar_oos.groupby('event_category', observed=False)['oos_hours_per_record'].mean().reindex(event_category_order)

sns.barplot(
    x=avg_oos_by_event.index,
    y=avg_oos_by_event.values,
    palette='tab10',
    hue=avg_oos_by_event.index,
    legend=False,
    edgecolor='black',
    ax=axes_oos[0, 1]
)

for i, val in enumerate(avg_oos_by_event.values):
    axes_oos[0, 1].text(i, val + (max(avg_oos_by_event.values)*0.01), f'{val:.2f}h',
                        ha='center', va='bottom', fontsize=10, fontweight='bold')

axes_oos[0, 1].set_title('2. Average OOS Hours by Event Category (6h - 21h59)', fontsize=13, fontweight='bold', pad=15)
axes_oos[0, 1].set_xlabel('Event Category', fontsize=11)
axes_oos[0, 1].set_ylabel('Average Out-of-Stock Hours (6h - 21h59)', fontsize=11)
axes_oos[0, 1].tick_params(axis='x', rotation=15)

# Plot 3: Top 5 Products by Average OOS Hours (Horizontal Bar Chart)
product_instock_hours = df_bivar_oos.groupby('product_id')['stock_hour6_21_cnt'].mean().reset_index()
product_oos = product_instock_hours.copy()
product_oos['avg_oos_hours'] = total_hours_in_window_new - product_oos['stock_hour6_21_cnt']

top_5_oos_products = product_oos.sort_values(by='avg_oos_hours', ascending=False).head(5)
top_5_oos_products['product_id'] = top_5_oos_products['product_id'].astype(str)

sns.barplot(
    x='avg_oos_hours',
    y='product_id',
    data=top_5_oos_products,
    palette='Set2',
    hue='product_id',
    legend=False,
    edgecolor='black',
    orient='h',
    ax=axes_oos[1, 0]
)

for i, v in enumerate(top_5_oos_products['avg_oos_hours']):
    axes_oos[1, 0].text(v + 0.1, i, f'{v:.2f}', va='center', fontsize=10, fontweight='bold')

axes_oos[1, 0].set_title('3. Top 5 Products by Average OOS Hours (6h - 21h59)', fontsize=13, fontweight='bold', pad=15)
axes_oos[1, 0].set_xlabel('Average Out-of-Stock Hours (6h - 21h59)', fontsize=11)
axes_oos[1, 0].set_ylabel('Product ID', fontsize=11)

# Plot 4: Average OOS Hours by Temperature Range
temperature_bins = [0, 14, 18, 22, 26, 30, np.inf] 
temperature_labels = ['<14°C', '14-18°C', '18-22°C', '22-26°C', '26-30°C', '>30°C']

df_bivar_oos['temperature_range'] = pd.cut(df_bivar_oos['avg_temperature'], bins=temperature_bins, labels=temperature_labels, right=False)
avg_oos_by_temp = df_bivar_oos.groupby('temperature_range', observed=False)['oos_hours_per_record'].mean().reset_index()

sns.barplot(
    x='temperature_range',
    y='oos_hours_per_record',
    data=avg_oos_by_temp,
    palette='viridis',
    hue='temperature_range',
    legend=False,
    edgecolor='black',
    ax=axes_oos[1, 1]
)

for i, val in enumerate(avg_oos_by_temp['oos_hours_per_record']):
    if pd.notna(val):
        axes_oos[1, 1].text(i, val + (avg_oos_by_temp['oos_hours_per_record'].max()*0.01), f'{val:.2f}h',
                            ha='center', va='bottom', fontsize=10, fontweight='bold')

axes_oos[1, 1].set_title('4. Average OOS Hours by Temperature Range (6h - 21h59)', fontsize=13, fontweight='bold')
axes_oos[1, 1].set_xlabel('Temperature Range', fontsize=11)
axes_oos[1, 1].set_ylabel('Average Out-of-Stock Hours (6h - 21h59)', fontsize=11)

plt.tight_layout()
st.pyplot(fig_bivar_oos_plots)

# -----------------------------------------------------------------------------
# 6. MULTIVARIATE ANALYSIS - SEABORN/MATPLOTLIB
# -----------------------------------------------------------------------------
st.divider()
st.subheader("📈 3. Multivariate Analysis")

df_multi = df_filtered.copy()

# Đồng bộ hóa xử lý chuỗi và tính toán OOS chính xác khung 6h-21h59 (16 tiếng)
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

df_multi['hours_stock_status'] = df_multi['hours_stock_status'].apply(safe_eval_multi)

df_multi['stock_hour6_21_cnt'] = df_multi['hours_stock_status'].apply(lambda x: sum(1 for status in x[6:22] if status == 0))
total_hours_in_window_multi = 16
df_multi['oos_hours_per_record'] = total_hours_in_window_multi - df_multi['stock_hour6_21_cnt']

df_multi['day_of_week'] = df_multi['dt'].dt.day_name()

# --- BỔ SUNG: LỌC THEO CATEGORY ID ---
st.markdown("##### 🔍 Multivariate Filters")
all_categories = sorted([x for x in df_multi['first_category_id'].unique() if pd.notna(x)])
selected_categories = st.multiselect("Select First Category ID(s) to display:", options=all_categories, default=all_categories[:10])

# Áp dụng lọc Category nếu người dùng có chọn
if selected_categories:
    df_multi = df_multi[df_multi['first_category_id'].isin(selected_categories)]

st.markdown(f"**Number of records for multivariate analysis after category filtering:** {df_multi.shape[0]:,}")

if df_multi.shape[0] == 0:
    st.warning("No data matches the selected Category ID(s). Please select more categories.")
else:
    sns.set_theme(style="whitegrid")

    fig_multi, axes_multi = plt.subplots(2, 1, figsize=(22, 18))
    fig_multi.suptitle('Hourly Out-of-Stock Prediction for Fresh Food Retail\n(Operating Scale Window: 6h00 - 21h59)', fontsize=16, fontweight='bold', y=1.00)

    # 1. Heatmap: Product OOS Hours Across Days of Week
    if 'day_of_week' in df_multi.columns:
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
    axes_multi[0].set_title('1. Product OOS Hours Across Days of Week', fontsize=14, fontweight='bold')
    axes_multi[0].set_xlabel('Day of Week', fontsize=11)
    axes_multi[0].set_ylabel('First Category ID', fontsize=11)
    axes_multi[0].tick_params(axis='x', rotation=45)
    axes_multi[0].tick_params(axis='y', rotation=0)

    # 2. Line Plot: Product OOS Hours Across Temperature Ranges
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
    axes_multi[1].set_title('2. Product OOS Hours Across Temperature Ranges', fontsize=14, fontweight='bold')
    axes_multi[1].set_xlabel('Temperature Range', fontsize=11)
    axes_multi[1].set_ylabel('Average Out-of-Stock Hours', fontsize=11)
    axes_multi[1].tick_params(axis='x', rotation=15)
    axes_multi[1].legend(title='First Category ID', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    st.pyplot(fig_multi)

import streamlit as st
import joblib
import os
import pandas as pd
import gdown

# -----------------------------------------------------------------------------
# 7. PHẦN AI DỰ ĐOÁN HẾT HÀNG (XGBoost) - GIAO DIỆN TỐI GIẢN (ĐỊNH DẠNG SỐ)
# -----------------------------------------------------------------------------
st.divider()
st.header("🤖 AI Dự đoán hết hàng (XGBoost)")
st.markdown("Chọn Store ID và Product ID, sau đó nhập các thông số kinh doanh để AI dự đoán.")

@st.cache_data
def load_raw_dataset():
    if os.path.exists('freshretailnet_city03_dataset.csv'):
        return pd.read_csv('freshretailnet_city03_dataset.csv')
    return None

# Hàm tự động tải mô hình từ Google Drive nếu chưa có
@st.cache_resource
def load_ai_model():
    model_path = 'xgb_model.pkl'
    if not os.path.exists(model_path):
        # ID Google Drive của file xgb_model.pkl bạn đã cung cấp
        file_id = '1_eFNhjIV0OYVuZ5yncRjC7yDAA0xNhNk'
        url = f'https://drive.google.com/uc?id={file_id}'
        st.info(f"Đang tải mô hình AI từ Google Drive, vui lòng đợi trong giây lát...")
        gdown.download(url, model_path, quiet=False)
    return joblib.load(model_path)

# 1. Load Dataset
df_data = load_raw_dataset()

# 2. Load Model an toàn qua Google Drive (thay vì kiểm tra cục bộ đơn thuần)
if df_data is not None:
    try:
        model = load_ai_model()
        
        # 1. Chọn Store ID
        all_stores = sorted(df_data['store_id'].dropna().unique().tolist())
        selected_store_id = st.selectbox("1. Chọn Store ID", options=all_stores)
        
        # 2. Lọc danh sách Product ID chỉ thuộc Store đó để sort/select
        df_filtered_store = df_data[df_data['store_id'] == selected_store_id]
        products_in_store = sorted(df_filtered_store['product_id'].dropna().unique().tolist())
        
        selected_product_id = st.selectbox("2. Chọn Product ID", options=products_in_store if products_in_store else [0])
        
        # Tự động trích xuất thông tin phân loại chuẩn từ dòng đầu tiên khớp
        product_info = df_filtered_store[df_filtered_store['product_id'] == selected_product_id].iloc[0]
        
        auto_mg_id = int(product_info['management_group_id'])
        auto_cat1 = int(product_info['first_category_id'])
        auto_cat2 = int(product_info['second_category_id'])
        auto_cat3 = int(product_info['third_category_id'])
        
        # Lấy thông số môi trường trung bình từ dataset để gán ngầm
        def_precpt = df_data['precpt'].dropna().mean()
        def_wind = df_data['avg_wind_level'].dropna().mean()

        # Khối thông tin chi tiết
        st.info(
            f"📍 **Thông tin sản phẩm & phân cấp:**\n\n"
            f"• **Management Group ID:** {auto_mg_id}  \n"
            f"• **Danh mục (Categories):** {auto_cat1} › {auto_cat2} › {auto_cat3}"
        )

        # === GIAO DIỆN CHÍNH ===
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**1. Bối cảnh & Sự kiện**")
            # Điều chỉnh bước nhảy discount = 0.1
            discount = st.number_input("Discount (%)", min_value=0.0, max_value=100.0, value=0.0, step=0.1, format="%.1f")
            holiday_flag = st.selectbox("Holiday Flag (Ngày lễ)", [0, 1], format_func=lambda x: "Có (1)" if x == 1 else "Không (0)")
            activity_flag = st.selectbox("Activity Flag (Khuyến mãi)", [0, 1], format_func=lambda x: "Có (1)" if x == 1 else "Không (0)")

        with col2:
            st.markdown("**2. Thời gian & Lịch sử**")
            is_weekend = st.selectbox("Is Weekend (Cuối tuần)", [0, 1], format_func=lambda x: "Cuối tuần (1)" if x == 1 else "Ngày thường (0)")
            # Tỷ lệ hết hàng: step 0.0001, hỗ trợ hiển thị 4 chữ số thập phân
            oos_rate_lag1_day = st.number_input(f"Tỷ lệ hết hàng hôm qua (0.0 - 1.0)", min_value=0.0, max_value=1.0, value=0.0, step=0.0001, format="%.4f")
            
            st.markdown("**3. Môi trường**")
            # Điều chỉnh bước nhảy nhiệt độ = 0.01
            avg_temperature = st.number_input("Nhiệt độ trung bình (°C)", value=25.0, step=0.01, format="%.2f")

        # Nút bấm dự đoán
# -----------------------------------------------------------------------------
# 7. PHẦN AI DỰ ĐOÁN HẾT HÀNG (XGBoost)
# -----------------------------------------------------------------------------
st.divider()
st.header("🤖 AI Dự đoán hết hàng (XGBoost)")

@st.cache_resource
def load_ai_model():
    model_path = 'xgb_model.pkl'
    if not os.path.exists(model_path):
        with st.spinner("Đang tải mô hình từ Google Drive..."):
            file_id = '1_eFNhjIV0OYVuZ5yncRjC7yDAA0xNhNk'
            url = f'https://drive.google.com/uc?id={file_id}'
            gdown.download(url, model_path, quiet=False)
    return joblib.load(model_path)

# Load model
try:
    model = load_ai_model()
    df_data = load_raw_dataset() # Giả định bạn đã có hàm này ở trên

    if df_data is not None:
        # Chọn Store/Product
        all_stores = sorted(df_data['store_id'].unique())
        selected_store_id = st.selectbox("1. Chọn Store ID", options=all_stores)
        
        df_filtered_store = df_data[df_data['store_id'] == selected_store_id]
        products_in_store = sorted(df_filtered_store['product_id'].unique())
        selected_product_id = st.selectbox("2. Chọn Product ID", options=products_in_store)
        
        product_info = df_filtered_store[df_filtered_store['product_id'] == selected_product_id].iloc[0]
        
        # Nhập thông số
        col1, col2 = st.columns(2)
        with col1:
            discount = st.number_input("Discount (%)", 0.0, 100.0, 0.0, 0.1)
            holiday = st.selectbox("Holiday Flag", [0, 1])
            activity = st.selectbox("Activity Flag", [0, 1])
        with col2:
            is_weekend = st.selectbox("Is Weekend", [0, 1])
            oos_lag = st.number_input("OOS Rate Lag (0-1)", 0.0, 1.0, 0.0, 0.0001)
            temp = st.number_input("Nhiệt độ trung bình", 0.0, 40.0, 25.0, 0.1)

        if st.button("🚀 Bấm Dự Đoán"):
            with st.spinner("AI đang phân tích..."):
                # CẤU TRÚC 15 CỘT - Đảm bảo thứ tự và tên cột khớp với lúc train
                features = pd.DataFrame([{
                    'store_id': int(selected_store_id),
                    'management_group_id': int(product_info['management_group_id']),
                    'first_category_id': int(product_info['first_category_id']),
                    'second_category_id': int(product_info['second_category_id']),
                    'third_category_id': int(product_info['third_category_id']),
                    'product_id': int(selected_product_id),
                    'discount': float(discount),
                    'holiday_flag': int(holiday),
                    'activity_flag': int(activity),
                    'precpt': float(df_data['precpt'].mean()),
                    'avg_temperature': float(temp),
                    'avg_humidity': 70.0,
                    'avg_wind_level': float(df_data['avg_wind_level'].mean()),
                    'oos_rate_lag1_day': float(oos_lag),
                    'is_weekend': int(is_weekend)
                }])
                
                # Dự đoán
                prediction = model.predict(features)
                
                # Hiển thị
                st.success("✅ Dự đoán hoàn tất!")
                khung_gio = [f"{h}:00 - {h}:59" for h in range(6, 22)]
                ket_qua = ["🔴 Hết hàng" if x == 1 else "🟢 Còn hàng" for x in prediction[0]]
                st.table(pd.DataFrame({"Khung giờ": khung_gio, "Dự báo": ket_qua}))
except Exception as e:
    st.error(f"Lỗi: {e}")
    st.write("Kiểm tra lại cấu trúc file model và input data.")