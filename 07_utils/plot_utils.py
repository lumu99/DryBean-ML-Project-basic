# ============================================================
# DryBean-ML-Project-basic
# 第 4 步：工具函数模块 - 图表工具（可视化增强版）
#
# 对应期末作业要求：
# 1. 多算法实验分析需要使用图表展示结果
# 2. GitHub 展示和论文中需要插入精度、速度、过拟合、鲁棒性等对比图
#
# 本文件功能：
# 1. 生成模型测试集精度对比图
# 2. 生成训练集与测试集精度对比图
# 3. 生成模型推理速度对比图
# 4. 生成模型过拟合差距对比图
# 5. 生成模型鲁棒性曲线图
#
# 注意：
# 本文件只负责保存图片，不弹出 UI 窗口。
# 算法运行阶段不要使用 UI 显示，符合期末作业要求。
# ============================================================

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def ensure_output_dir(output_path: Path) -> None:
    """
    确保图片输出目录存在。

    参数：
        output_path: 图片保存路径

    返回：
        None
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)


def plot_test_accuracy_comparison(
    result_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    绘制测试集准确率对比图。

    参数：
        result_df: 包含 model 和 accuracy 的结果表
        output_path: 图片保存路径
    """
    ensure_output_dir(output_path)

    plt.figure(figsize=(9, 5))
    plt.bar(result_df["model"], result_df["accuracy"])
    plt.xlabel("Model")
    plt.ylabel("Test Accuracy")
    plt.title("Test Accuracy Comparison")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_train_test_accuracy_comparison(
    result_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    绘制训练集与测试集准确率对比图。

    参数：
        result_df: 包含 model、train_accuracy、test_accuracy 的结果表
        output_path: 图片保存路径
    """
    ensure_output_dir(output_path)

    x_positions = range(len(result_df))
    bar_width = 0.35

    plt.figure(figsize=(10, 5))
    plt.bar(
        [x - bar_width / 2 for x in x_positions],
        result_df["train_accuracy"],
        width=bar_width,
        label="Train Accuracy",
    )
    plt.bar(
        [x + bar_width / 2 for x in x_positions],
        result_df["test_accuracy"],
        width=bar_width,
        label="Test Accuracy",
    )
    plt.xlabel("Model")
    plt.ylabel("Accuracy")
    plt.title("Train vs Test Accuracy")
    plt.xticks(list(x_positions), result_df["model"])
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_predict_time_comparison(
    result_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    绘制模型推理时间对比图。

    参数：
        result_df: 包含 model 和 predict_time_seconds 的结果表
        output_path: 图片保存路径
    """
    ensure_output_dir(output_path)

    plt.figure(figsize=(9, 5))
    plt.bar(result_df["model"], result_df["predict_time_seconds"])
    plt.xlabel("Model")
    plt.ylabel("Predict Time (seconds)")
    plt.title("Prediction Time Comparison")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_overfit_gap_comparison(
    result_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    绘制模型过拟合差距对比图。

    参数：
        result_df: 包含 model 和 overfit_gap 的结果表
        output_path: 图片保存路径
    """
    ensure_output_dir(output_path)

    plt.figure(figsize=(9, 5))
    plt.bar(result_df["model"], result_df["overfit_gap"])
    plt.xlabel("Model")
    plt.ylabel("Train Accuracy - Test Accuracy")
    plt.title("Overfitting Gap Comparison")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_robustness_curve(
    result_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    绘制鲁棒性曲线图。

    参数：
        result_df: 包含 model、noise_level、accuracy 的结果表
        output_path: 图片保存路径
    """
    ensure_output_dir(output_path)

    plt.figure(figsize=(10, 6))

    for model_name in result_df["model"].unique():
        model_df = result_df[result_df["model"] == model_name].sort_values("noise_level")
        plt.plot(
            model_df["noise_level"],
            model_df["accuracy"],
            marker="o",
            label=model_name,
        )

    plt.xlabel("Noise Level")
    plt.ylabel("Accuracy")
    plt.title("Robustness Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


if __name__ == "__main__":
    print("=" * 70)
    print("第 4 步检查：plot_utils.py 图表工具（可视化增强版）加载成功")
    print("=" * 70)