import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import gdown

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & LOAD DATASET REFERENCE
# -----------------------------------------------------------------------------
st.set_page_config(page_title="FreshRetailNet - AI Prediction", layout="centered")

st.header("🤖 AI Dự đoán hết hàng (XGBoost)")
st.markdown("Chọn Store ID và Product ID, sau đó nhập các thông số kinh doanh để AI dự đoán trạng thái kho theo các khung giờ (6:00 - 21:59).")

@st.cache_data
def load_dataset_source():
    filename = "freshretailnet_city03_dataset.csv"
    if os.path.exists(filename):
        return pd.read_csv(filename)
    return None

df_data = load_dataset_source()

if df_data is None:
    st.error("Dataset file (`freshretailnet_city03_dataset.csv`) not found. Please ensure it is in the same directory.")
    st.stop()

# -----------------------------------------------------------------------------
# 2. LOAD AI MODEL (XGBOOST)
# -----------------------------------------------------------------------------
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
    all_stores_ai = sorted(df_data['store_id'].dropna().unique().tolist())
    selected_store_id = st.selectbox("1. Chọn Store ID", options=all_stores_ai)
    
    # 2. Lọc Product ID theo Store đã chọn
    df_filtered_store = df_data[df_data['store_id'] == selected_store_id]
    products_in_store = sorted(df_filtered_store['product_id'].dropna().unique().tolist())
    
    selected_product_id = st.selectbox("2. Chọn Product ID", options=products_in_store if products_in_store else [0])
    
    # Trích xuất phân cấp chuẩn từ dữ liệu sản phẩm tương ứng
    product_info = df_filtered_store[df_filtered_store['product_id'] == selected_product_id].iloc[0]
    
    auto_mg_id = int(product_info['management_group_id'])
    auto_cat1 = int(product_info['first_category_id'])
    auto_cat2 = int(product_info['second_category_id'])
    auto_cat3 = int(product_info['third_category_id'])
    
    def_precpt = df_data['precpt'].dropna().mean()
    def_wind = df_data['avg_wind_level'].dropna().mean()

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