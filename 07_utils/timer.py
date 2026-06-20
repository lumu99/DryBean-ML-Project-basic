# ============================================================
# DryBean-ML-Project-basic
# 第 4 步：工具函数模块 - 计时工具
#
# 对应期末作业要求：
# 1. 多算法实验分析需要比较算法推理速度
# 2. 后续训练模块和测试模块需要记录训练耗时与预测耗时
#
# 本文件功能：
# 1. 提供通用计时函数
# 2. 统计模型训练时间
# 3. 统计模型推理时间
# ============================================================

import time
from typing import Callable, Any


def time_function(func: Callable, *args, **kwargs) -> tuple[Any, float]:
    """
    统计任意函数的运行时间。

    参数：
        func: 需要计时的函数
        *args: 函数位置参数
        **kwargs: 函数关键字参数

    返回：
        result: 函数运行结果
        elapsed_time: 函数运行耗时，单位秒
    """
    start_time = time.perf_counter()
    result = func(*args, **kwargs)
    end_time = time.perf_counter()

    elapsed_time = end_time - start_time
    return result, elapsed_time


def time_model_fit(model: Any, X_train, y_train) -> tuple[Any, float]:
    """
    统计模型训练时间。

    参数：
        model: 待训练模型
        X_train: 训练特征
        y_train: 训练标签

    返回：
        model: 训练后的模型
        train_time: 训练耗时，单位秒
    """
    start_time = time.perf_counter()
    model.fit(X_train, y_train)
    end_time = time.perf_counter()

    train_time = end_time - start_time
    return model, train_time


def time_model_predict(model: Any, X_test) -> tuple[Any, float]:
    """
    统计模型预测时间。

    参数：
        model: 已训练模型
        X_test: 测试特征

    返回：
        y_pred: 预测结果
        predict_time: 推理耗时，单位秒
    """
    start_time = time.perf_counter()
    y_pred = model.predict(X_test)
    end_time = time.perf_counter()

    predict_time = end_time - start_time
    return y_pred, predict_time


if __name__ == "__main__":
    print("=" * 70)
    print("第 4 步检查：timer.py 计时工具加载成功")
    print("=" * 70)