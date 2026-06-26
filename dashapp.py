import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import gdown
import matplotlib.pyplot as plt
import seaborn as sns
import ast

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & LOAD DATA
# -----------------------------------------------------------------------------
st.set_page_config(page_title="FreshRetailNet - City 03 Dashboard", layout="wide")

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

all_stores = sorted(df_raw['store_id'].unique())
selected_store = st.sidebar.multiselect("Select Store ID(s):", options=all_stores, default=all_stores[:5])

min_date = df_raw['dt'].min().date()
max_date = df_raw['dt'].max().date()

selected_dates = st.sidebar.date_input(
    "Select Time Period (Date or Date range):",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

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

sns.histplot(df_eda['sale_amount'], kde=True, ax=axes[0, 0], bins=30, color='skyblue')
axes[0, 0].set_title('1. Distribution of Sale Amount', fontsize=13, fontweight='bold')
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

sns.histplot(df_eda['avg_temperature'], kde=True, ax=axes[1, 0], bins=30, color='lightcoral')
axes[1, 0].set_title('3. Distribution of Average Temperature', fontsize=13, fontweight='bold')
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

df_bivar_oos['stock_hour6_21_cnt'] = df_bivar_oos['hours_stock_status'].apply(lambda x: sum(1 for status in x[6:22] if status == 0))
total_hours_in_window_new = 16
df_bivar_oos['oos_hours_per_record'] = total_hours_in_window_new - df_bivar_oos['stock_hour6_21_cnt']

hours_df = pd.DataFrame(df_bivar_oos['hours_stock_status'].tolist(), columns=[f'{i}h' for i in range(24)])

sns.set_theme(style="whitegrid")
fig_bivar_oos_plots, axes_oos = plt.subplots(2, 2, figsize=(20, 15))
fig_bivar_oos_plots.suptitle('Bivariate Analysis: Factors Influencing Out-of-Stock Hours (6h - 21h59 window)', fontsize=16, fontweight='bold', y=1.02)

hourly_oos_rate = hours_df[[f'{i}h' for i in range(6, 22)]].mean()
sns.lineplot(x=hourly_oos_rate.index, y=hourly_oos_rate.values, marker='o', color='red', linewidth=2.5, ax=axes_oos[0, 0])
axes_oos[0, 0].set_title('1. Average Out-of-Stock Rate by Hour (6h - 21h59)', fontsize=14, fontweight='bold')
axes_oos[0, 0].set_xlabel('Hour of the Day (6h - 21h59)', fontsize=11)
axes_oos[0, 0].set_ylabel('Out-of-Stock Rate (0 - 1)')
axes_oos[0, 0].tick_params(axis='x', rotation=45)

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

st.markdown("##### 🔍 Multivariate Filters")
all_categories = sorted([x for x in df_multi['first_category_id'].unique() if pd.notna(x)])
selected_categories = st.multiselect("Select First Category ID(s) to display:", options=all_categories, default=all_categories[:10])

if selected_categories:
    df_multi = df_multi[df_multi['first_category_id'].isin(selected_categories)]

st.markdown(f"**Number of records for multivariate analysis after category filtering:** {df_multi.shape[0]:,}")

if df_multi.shape[0] == 0:
    st.warning("No data matches the selected Category ID(s). Please select more categories.")
else:
    sns.set_theme(style="whitegrid")

    fig_multi, axes_multi = plt.subplots(2, 1, figsize=(22, 18))
    fig_multi.suptitle('Hourly Out-of-Stock Prediction for Fresh Food Retail\n(Operating Scale Window: 6h00 - 21h59)', fontsize=16, fontweight='bold', y=1.00)

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

# -----------------------------------------------------------------------------
# 7. PHẦN AI DỰ ĐOÁN HẾT HÀNG (XGBoost)
# -----------------------------------------------------------------------------
st.divider()
st.header("🤖 AI Dự đoán hết hàng (XGBoost)")
st.markdown("Chọn Store ID và Product ID, sau đó nhập các thông số kinh doanh để AI dự đoán trạng thái kho theo các khung giờ (6:00 - 21:59).")

@st.cache_resource
def load_ai_model():
    model_path = 'xgb_model.pkl'
    if not os.path.exists(model_path):
        with st.spinner("Đang tải mô hình AI từ Google Drive..."):
            file_id = '1_eFNhjIV0OYVuZ5yncRjC7yDAA0xNhNk'
            url = f'https://drive.google.com/uc?id={file_id}'
            gdown.download(url, model_path, quiet=False)
    return joblib.load(model_path)

try:
    model = load_ai_model()
    
    # 1. Chọn Store ID
    all_stores_ai = sorted(df_data['store_id'].dropna().unique().tolist()) if 'df_data' in locals() and df_data is not None else sorted(df_raw['store_id'].dropna().unique().tolist())
    selected_store_id = st.selectbox("1. Chọn Store ID", options=all_stores_ai)
    
    # 2. Lọc Product ID theo Store đã chọn
    df_dataset_source = df_data if 'df_data' in locals() and df_data is not None else df_raw
    df_filtered_store = df_dataset_source[df_dataset_source['store_id'] == selected_store_id]
    products_in_store = sorted(df_filtered_store['product_id'].dropna().unique().tolist())
    
    selected_product_id = st.selectbox("2. Chọn Product ID", options=products_in_store if products_in_store else [0])
    
    # Trích xuất phân cấp chuẩn từ dữ liệu sản phẩm tương ứng
    product_info = df_filtered_store[df_filtered_store['product_id'] == selected_product_id].iloc[0]
    
    auto_mg_id = int(product_info['management_group_id'])
    auto_cat1 = int(product_info['first_category_id'])
    auto_cat2 = int(product_info['second_category_id'])
    auto_cat3 = int(product_info['third_category_id'])
    
    def_precpt = df_dataset_source['precpt'].dropna().mean()
    def_wind = df_dataset_source['avg_wind_level'].dropna().mean()

    st.info(
        f"📍 **Thông tin sản phẩm & phân cấp:**\n\n"
        f"• **Management Group ID:** {auto_mg_id}  \n"
        f"• **Danh mục (Categories):** {auto_cat1} › {auto_cat2} › {auto_cat3}"
    )

    # Giao diện nhập thông số dự đoán
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**1. Bối cảnh & Sự kiện**")
        discount = st.number_input("Discount (%)", min_value=0.0, max_value=100.0, value=0.0, step=0.1, format="%.1f")
        holiday_flag = st.selectbox("Holiday Flag (Ngày lễ)", [0, 1], format_func=lambda x: "Có (1)" if x == 1 else "Không (0)")
        activity_flag = st.selectbox("Activity Flag (Khuyến mãi)", [0, 1], format_func=lambda x: "Có (1)" if x == 1 else "Không (0)")

    with col2:
        st.markdown("**2. Thời gian & Lịch sử**")
        is_weekend = st.selectbox("Is Weekend (Cuối tuần)", [0, 1], format_func=lambda x: "Cuối tuần (1)" if x == 1 else "Ngày thường (0)")
        oos_rate_lag1_day = st.number_input("Tỷ lệ hết hàng hôm qua (0.0 - 1.0)", min_value=0.0, max_value=1.0, value=0.0, step=0.0001, format="%.4f")
        
        st.markdown("**3. Môi trường**")
        avg_temperature = st.number_input("Nhiệt độ trung bình (°C)", value=25.0, step=0.01, format="%.2f")

    if st.button("🚀 Bấm Dự Đoán"):
        try:
            with st.spinner("Mô hình XGBoost đang dự đoán, xin vui lòng đợi..."):
                # Gom đủ 15 đặc trưng chính xác tuyệt đối
                input_data = pd.DataFrame([{
                    'store_id': int(selected_store_id),
                    'management_group_id': int(auto_mg_id),
                    'first_category_id': int(auto_cat1),
                    'second_category_id': int(auto_cat2),
                    'third_category_id': int(auto_cat3),
                    'product_id': int(selected_product_id),
                    'discount': float(discount),
                    'holiday_flag': int(holiday_flag),
                    'activity_flag': int(activity_flag),
                    'precpt': float(def_precpt),
                    'avg_temperature': float(avg_temperature),
                    'avg_humidity': 70.0,  # Gán ngầm mặc định tránh thiếu cột
                    'avg_wind_level': float(def_wind),
                    'oos_rate_lag1_day': float(oos_rate_lag1_day),
                    'is_weekend': int(is_weekend)
                }])
                
                prediction = model.predict(input_data)
                
                st.success("✅ Mô hình đã dự đoán xong! Trạng thái hết hàng từ 6:00 đến 21:59:")
                khung_gio = [f"{h}:00 - {h}:59" for h in range(6, 22)]
                
                pred_array = prediction[0] if len(prediction.shape) > 1 else prediction
                ket_qua = ["🔴 Cảnh báo Hết hàng" if x == 1 else "🟢 Còn hàng" for x in pred_array[:16]]
                
                df_ketqua = pd.DataFrame({"Khung giờ": khung_gio, "Dự báo từ AI": ket_qua})
                st.dataframe(df_ketqua, hide_index=True, use_container_width=True)
                
        except Exception as pred_err:
            st.error(f"Lỗi khi thực hiện dự đoán: {pred_err}")

except Exception as model_err:
    st.error(f"Không thể khởi tạo mô hình hoặc tải dữ liệu. Lỗi: {model_err}")