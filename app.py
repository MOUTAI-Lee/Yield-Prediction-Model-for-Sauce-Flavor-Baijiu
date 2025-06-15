import streamlit as st

from predict import predict


def convert_to_float(values):
    """将输入值转换为浮点数列表"""
    return [float(x) for x in values]


# 标题
st.title("Yield Prediction Model of Sauce-Flavor Baijiu")

# 模型选择
st.subheader("Model Selection")
model_type = st.segmented_control(
    "Select Model",
    ["R1-2", "R3-5", "R6-7"],
    default="R1-2"
)

# 输入参数
st.subheader("Input Parameters")

col1, col2, col3 = st.columns(3)
with col1:
    p_moisture = st.number_input("P-moisture", value=0.0, min_value=0.0, format="%.2f")
    p_acid = st.number_input("P-acid", value=0.0, min_value=0.0, format="%.2f")
    p_sugar = st.number_input("P-sugar", value=0.0, min_value=0.0, format="%.2f")
    p_starch = st.number_input("P-starch", value=0.0, min_value=0.0, format="%.2f")
with col2:
    d_moisture = st.number_input("D-moisture", value=0.0, min_value=0.0, format="%.2f")
    d_acid = st.number_input("D-acid", value=0.0, min_value=0.0, format="%.2f")
    d_sugar = st.number_input("D-sugar", value=0.0, min_value=0.0, format="%.2f")
    d_starch = st.number_input("D-starch", value=0.0, min_value=0.0, format="%.2f")
with col3:
    p_lactic_acid = st.number_input("P-lactic_acid", value=0.0, min_value=0.0, format="%.2f")
    p_ethanol = st.number_input("P-ethanol", value=0.0, min_value=0.0, format="%.2f")
    d_lactic_acid = st.number_input("D-lactic_acid", value=0.0, min_value=0.0, format="%.2f")
    d_ethanol = st.number_input("D-ethanol", value=0.0, min_value=0.0, format="%.2f")

# 预测按钮
button = st.button("Predict")
# 预测
if button:
    st.write(
        predict(
            f'models/{model_type}',
            [p_moisture,
             p_acid,
             p_sugar,
             p_starch,
             d_moisture,
             d_acid,
             d_sugar,
             d_starch,
             p_lactic_acid,
             p_ethanol,
             d_lactic_acid,
             d_ethanol]
        )
    )
