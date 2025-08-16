import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.decomposition import PCA

# عنوان التطبيق
st.title("Wholesale Customers Clustering App")

# تحميل الموديل المدرب
try:
    model = joblib.load("KMeans_model.pkl")
    st.success("✅ تم تحميل الموديل بنجاح")
except:
    st.error("⚠️ ملف KMeans_model.pkl مش موجود في نفس المجلد مع app.py")

# رفع البيانات
uploaded_file = st.file_uploader("📂 ارفع ملف CSV (مثلاً Wholesale customers data.csv)", type=["csv"])

if uploaded_file is not None:
    # قراءة البيانات
    df = pd.read_csv(uploaded_file)
    st.write("### البيانات الأصلية:")
    st.dataframe(df.head())

    # إزالة الأعمدة الغير رقمية
    if "Channel" in df.columns and "Region" in df.columns:
        df = df.drop(['Channel', 'Region'], axis=1)

    # PowerTransformer
    pt = PowerTransformer(method='yeo-johnson')
    df_transformed = pd.DataFrame(pt.fit_transform(df), columns=df.columns)

    st.write("### البيانات بعد PowerTransformer:")
    st.dataframe(df_transformed.head())

    # Scaling
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(df_transformed.values)

    # التنبؤ بالكلستر باستخدام الموديل المحفوظ
    y_kmeans = model.predict(x_scaled)
    df['cluster'] = y_kmeans

    # PCA لعرض البيانات في 2D
    p
