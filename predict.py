#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from typing import Any, Dict, List

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline


def load_model(model_path: str) -> Any:
    """加载指定路径的模型

    :param model_path: 模型文件路径
    :return: 加载的模型对象
    """
    return joblib.load(model_path)


def predict(model_path: str, features_list: List[float]) -> Dict[str, float]:
    """使用模型预测产量

    :param model_path: 模型文件路径
    :param input_data: 输入数据 DataFrame
    :return: 包含各个模型预测结果的字典
    """

    if len(features_list) != 12:
        raise ValueError("输入特征数量必须为12个！")

    # 加载 stacking 模型
    stacking_model = load_model(f"{model_path}/stacking_model.joblib")

    # 构造 DataFrame
    input_df = pd.DataFrame([features_list], columns=[
        'P-mositure', 'P-acid', 'P-sugar', 'P-starch',
        'D-mositure', 'D-acid', 'D-sugar', 'D-starch',
        'P-lactic_acid', 'P-ethanol', 'D-lactic_acid', 'D-ethanol'
    ])

    # 调用模型预测
    predicted_yield = stacking_model.predict(input_df)[0]  # 预测产量
    return predicted_yield


if __name__ == "__main__":
    # 预设参数
    presets = [
        [39.6, 1.1, 1.0, 36.4, 41.3, 1.1, 1.1, 39.1, 1.6, 7.2, 1.6, 4.6],
        [41.7, 1.11, 0.83, 33.48, 43.1, 0.81, 1.68, 33.48, 2.6, 16.9, 2.6, 1.9],
        [44.90, 1.51, 0.48, 29.79, 43.90, 1.41, 1.27, 29.43, 2.3, 11.3, 4.1, 3.0],
        [44.10, 2.41, 0.45, 29.97, 44.70, 2.0, 1.89, 27.81, 2.0, 6.0, 2.7, 6.8],
        [45.80, 2.21, 0.31, 28.08, 45.40, 1.56, 1.05, 26.87, 2.5, 9.2, 3.9, 8.7],
        [42.4, 1.41, 0.61, 38.3, 41.0, 0.88, 1.73, 39.42, 3.4, 9.0, 3.1, 1.8],
        [40.6, 1.11, 1.24, 37.0, 40.3, 1.01, 1.46, 38.79, 2.8, 8.6, 3.0, 0.0],
        [44.3, 1.0, 0.66, 39.1, 40.9, 1.01, 1.55, 38.97, 4.6, 15.4, 2.8, 1.5]
    ]


    print("-" * 100)
    for preset in presets:
        predicted = predict("models/R1-2", preset)
        print(f"R1-2 外部样本预测产量：{predicted:.4f}")

    print("-" * 100)
    for preset in presets:
        predicted = predict("models/R3-5", preset)
        print(f"R3-5 外部样本预测产量：{predicted:.4f}")

    print("-" * 100)
    for preset in presets:
        predicted = predict("models/R6-7", preset)
        print(f"R6-7 外部样本预测产量：{predicted:.4f}")

    print("Done")
