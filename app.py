import streamlit as st
import numpy as np
import joblib

# --- تحميل الموديل والـ Scaler المحفوظين ---
try:
    model = joblib.load("KMeans_model.pkl")
    scaler = joblib.load("scaler.pkl")   # لازم تكوني حفظتي الـ StandardScaler قبل التدريب
    st.success("✅ تم تحميل الموديل والـ Scaler بنجاح")
except:
    st.error("⚠️ لازم يكون عندك KMeans_model.pkl و scaler.pkl في نفس المجلد مع app.py")

# --- واجهة التطبيق ---
st.title("🔮 Wholesale Customers KMeans Prediction App")
st.write("ادخل بيانات العميل عشان نحدد هو ينتمي لأنهي Cluster")

# --- إدخال بيانات العميل ---
milk = st.number_input('Milk Spending', min_value=0, value=5000)
grocery = st.number_input('Grocery Spending', min_value=0, value=8000)
frozen = st.number_input('Frozen Spending', min_value=0, value=2000)
delicassen = st.number_input('Delicassen Spending', min_value=0, value=500)
detergents_paper = st.number_input('Detergents & Paper Spending', min_value=0, value=1000)

# --- زر التنبؤ ---
if st.button("🔍 Predict Cluster"):
    # تجهيز البيانات (نفس ترتيب التدريب)
    new_data = np.array([[milk, grocery, frozen, delicassen, detergents_paper]])

    # تطبيق الـ Scaler
    new_data_scaled = scaler.transform(new_data)

    # تنبؤ بالـ Cluster
    cluster_id = model.predict(new_data_scaled)[0]

    st.success(f"✅ العميل يتبع الكلستر رقم: {cluster_id}")
