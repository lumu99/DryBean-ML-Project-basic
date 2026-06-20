# ============================================================
# DryBean-ML-Project-basic
# 第 4 步：工具函数模块 - 模型评价工具
#
# 对应期末作业要求：
# 1. 多算法实验分析需要比较测试集精度
# 2. 后续还需要进行过拟合分析、分类报告、混淆矩阵分析
#
# 本文件功能：
# 1. 统一计算 accuracy
# 2. 统一生成 classification report
# 3. 统一生成 confusion matrix
# 4. 将模型评价结果整理成字典，便于保存为表格
# ============================================================

from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def calculate_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    计算分类准确率。

    参数：
        y_true: 真实标签
        y_pred: 预测标签

    返回：
        accuracy: 准确率
    """
    accuracy = accuracy_score(y_true, y_pred)
    return float(accuracy)


def get_classification_report_dict(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, Any]:
    """
    生成分类报告字典。

    参数：
        y_true: 真实标签
        y_pred: 预测标签

    返回：
        report: 包含 precision、recall、f1-score 等指标的字典
    """
    report = classification_report(
        y_true,
        y_pred,
        output_dict=True,
        zero_division=0,
    )
    return report


def get_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.ndarray:
    """
    生成混淆矩阵。

    参数：
        y_true: 真实标签
        y_pred: 预测标签

    返回：
        matrix: 混淆矩阵
    """
    matrix = confusion_matrix(y_true, y_pred)
    return matrix


def build_basic_result_row(
    model_name: str,
    train_accuracy: float,
    test_accuracy: float,
    train_time: float,
    predict_time: float,
) -> dict[str, Any]:
    """
    将一个模型的基础实验结果整理成一行字典。

    参数：
        model_name: 模型名称
        train_accuracy: 训练集准确率
        test_accuracy: 测试集准确率
        train_time: 训练耗时，单位秒
        predict_time: 测试集推理耗时，单位秒

    返回：
        result_row: 结果字典
    """
    overfit_gap = train_accuracy - test_accuracy

    result_row = {
        "model": model_name,
        "train_accuracy": round(train_accuracy, 6),
        "test_accuracy": round(test_accuracy, 6),
        "overfit_gap": round(overfit_gap, 6),
        "train_time_seconds": round(train_time, 6),
        "predict_time_seconds": round(predict_time, 6),
    }

    return result_row


if __name__ == "__main__":
    print("=" * 70)
    print("第 4 步检查：metrics.py 模型评价工具加载成功")
    print("=" * 70)