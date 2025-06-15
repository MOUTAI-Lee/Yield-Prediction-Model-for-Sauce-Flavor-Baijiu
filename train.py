from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

def train_model(input_file, output_dir):

    # 1. 读取数据
    df = pd.read_csv(input_file)
    df = df.drop(columns=["sample"])  # 删除非特征列

    # 2. 特征与目标变量
    x = df.drop(columns=["yield"])
    y = df["yield"]

    # 3. 划分训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    # 3. 定义模型
    lr = LinearRegression()  # 线性回归
    xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)  # 梯度提升树
    svr = Pipeline([("scaler", StandardScaler()), ("svr", SVR())])  # 支持向量回归
    knn = Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsRegressor())])  # 最近邻回归
    rf = RandomForestRegressor(n_estimators=100, random_state=42)  # 随机森林

    # 4. 所有独立模型（包含单独 RandomForest）
    all_models = {
        "LinearRegression": lr,
        "XGBoost": xgb,
        "SVR": svr,
        "KNN": knn,
        "RandomForest": rf
    }

    # 5. 模型评估列表
    results = []

    for name, model in all_models.items():
        model.fit(x_train, y_train)  # 训练模型
        y_train_pred = model.predict(x_train)  # 训练集预测
        y_test_pred = model.predict(x_test)  # 测试集预测

        rmse_train = mean_squared_error(y_train, y_train_pred)  # 训练集均方误差
        r2_train = r2_score(y_train, y_train_pred)  # 训练集R2
        rmse_test = mean_squared_error(y_test, y_test_pred)  # 测试集均方误差
        r2_test = r2_score(y_test, y_test_pred)  # 测试集R2

        results.append({
            "Model": name,  # 模型名称
            "RMSE_Train": rmse_train,  # 训练集均方误差
            "R2_Train": r2_train,  # 训练集R2
            "RMSE_Test": rmse_test,  # 测试集均方误差
            "R2_Test": r2_test  # 测试集R2
        })

    # 6. 构建 stacking 模型（RF 作为元模型）
    stacking_model = StackingRegressor(
        estimators=[  # 定义基础模型
            ('lr', lr),  # 线性回归
            ('xgb', xgb),  # 梯度提升树
            ('svr', svr),  # 支持向量回归
            ('knn', knn)  # 最近邻回归
        ],
        final_estimator=RandomForestRegressor(n_estimators=100, random_state=42)  # 随机森林作为元模型
    )

    stacking_model.fit(x_train, y_train)  # 训练 stacking 模型

    # 7. 评估 stacking 模型
    y_train_stack = stacking_model.predict(x_train)  # 训练集预测
    y_test_stack = stacking_model.predict(x_test)  # 测试集预测

    rmse_train_stack = mean_squared_error(y_train, y_train_stack)  # 训练集均方误差
    r2_train_stack = r2_score(y_train, y_train_stack)  # 训练集R2
    rmse_test_stack = mean_squared_error(y_test, y_test_stack)  # 测试集均方误差
    r2_test_stack = r2_score(y_test, y_test_stack)  # 测试集R2

    results.append({
        "Model": "Stacking (RF as meta)",  # 模型名称
        "RMSE_Train": rmse_train_stack,  # 训练集均方误差
        "R2_Train": r2_train_stack,  # 训练集R2
        "RMSE_Test": rmse_test_stack,  # 测试集均方误差
        "R2_Test": r2_test_stack  # 测试集R2
    })

    results_df = pd.DataFrame(results).sort_values(by="RMSE_Test")  # 排序
    print(results_df)  # 打印结果

    # 8.模型持久化
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    joblib.dump(stacking_model, f'{output_dir}/stacking_model.joblib')  # 保存 stacking 模型

    # 保存所有基础模型
    for name, model in all_models.items():
        joblib.dump(model, f'{output_dir}/{name.lower()}.joblib')

    print("所有模型已保存到 models 目录")


if __name__ == '__main__':
    train_model("datasets/0502R1-2-yield-ml.csv", "models/R1-2")
    # train_model("datasets/0503R3-5-yield-ml.csv", "models/R3-5")
    # train_model("datasets/0503R6-7-yield-ml.csv", "models/R6-7")