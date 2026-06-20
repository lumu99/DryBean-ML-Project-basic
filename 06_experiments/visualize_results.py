# ============================================================
# DryBean-ML-Project-basic
# 第 13 步：结果可视化与图表汇总
#
# 对应期末作业要求：
# 1. 将多算法实验分析结果可视化
# 2. 生成论文和 GitHub 可直接使用的图表
# 3. 补充过拟合分析结果表
#
# 本文件功能：
# 1. 读取训练结果、测试结果、速度结果、鲁棒性结果
# 2. 生成测试集精度对比图
# 3. 生成训练集与测试集精度对比图
# 4. 生成推理时间对比图
# 5. 生成过拟合差距对比图
# 6. 生成鲁棒性曲线图
# 7. 保存 overfit_table.csv
#
# 注意：
# 本文件不重新训练模型，只对现有结果文件进行可视化整理。
# ============================================================

from pathlib import Path
import importlib.util

import pandas as pd


# ============================================================
# 1. 项目路径配置
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = PROJECT_ROOT / "08_results"
FIGURES_DIR = RESULTS_DIR / "figures"
UTILS_DIR = PROJECT_ROOT / "07_utils"

TRAIN_SUMMARY_FILE = RESULTS_DIR / "train_summary.csv"
TEST_RESULTS_FILE = RESULTS_DIR / "test_results.csv"
SPEED_TABLE_FILE = RESULTS_DIR / "speed_table.csv"
ROBUSTNESS_TABLE_FILE = RESULTS_DIR / "robustness_table.csv"
OVERFIT_TABLE_FILE = RESULTS_DIR / "overfit_table.csv"

ACCURACY_FIGURE_FILE = FIGURES_DIR / "accuracy_comparison.png"
TRAIN_TEST_FIGURE_FILE = FIGURES_DIR / "train_test_accuracy_comparison.png"
SPEED_FIGURE_FILE = FIGURES_DIR / "speed_comparison.png"
OVERFIT_FIGURE_FILE = FIGURES_DIR / "overfit_gap_comparison.png"
ROBUSTNESS_FIGURE_FILE = FIGURES_DIR / "robustness_curve.png"

PLOT_UTILS_FILE = UTILS_DIR / "plot_utils.py"


# ============================================================
# 2. 通用模块加载函数
# ============================================================

def load_module_from_path(module_name: str, module_path: Path):
    """
    按文件路径加载 Python 模块。
    """
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)

    if spec.loader is None:
        raise ImportError(f"无法加载模块：{module_path}")

    spec.loader.exec_module(module)
    return module


# ============================================================
# 3. 数据读取函数
# ============================================================

def check_file_exists(file_path: Path) -> None:
    """
    检查文件是否存在。
    """
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在：{file_path}")


def load_result_tables():
    """
    读取所有结果表。
    """
    check_file_exists(TRAIN_SUMMARY_FILE)
    check_file_exists(TEST_RESULTS_FILE)
    check_file_exists(SPEED_TABLE_FILE)
    check_file_exists(ROBUSTNESS_TABLE_FILE)

    train_df = pd.read_csv(TRAIN_SUMMARY_FILE)
    test_df = pd.read_csv(TEST_RESULTS_FILE)
    speed_df = pd.read_csv(SPEED_TABLE_FILE)
    robustness_df = pd.read_csv(ROBUSTNESS_TABLE_FILE)

    return train_df, test_df, speed_df, robustness_df


# ============================================================
# 4. 结果整理函数
# ============================================================

def build_model_name_map():
    """
    模型键名到展示名称的映射。
    """
    return {
        "knn": "KNN",
        "svm": "SVM",
        "random_forest": "Random Forest",
        "ann": "ANN",
        "xgboost": "XGBoost",
    }


def build_accuracy_plot_df(test_df: pd.DataFrame) -> pd.DataFrame:
    """
    构造测试集精度对比图数据。
    """
    model_name_map = build_model_name_map()

    result_df = test_df.copy()
    result_df["model"] = result_df["model"].map(model_name_map).fillna(result_df["model"])

    return result_df[["model", "accuracy"]]


def build_speed_plot_df(speed_df: pd.DataFrame) -> pd.DataFrame:
    """
    构造推理时间对比图数据。
    """
    model_name_map = build_model_name_map()

    result_df = speed_df.copy()
    result_df["model"] = result_df["model"].map(model_name_map).fillna(result_df["model"])

    return result_df[["model", "predict_time_seconds", "avg_latency_ms"]]


def build_train_test_comparison_df(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    构造训练集与测试集精度对比数据，同时计算过拟合差距。
    """
    merged_df = pd.merge(
        train_df,
        test_df,
        left_on="model_key",
        right_on="model",
        how="inner",
    )

    comparison_df = pd.DataFrame({
        "model": merged_df["model_name"],
        "train_accuracy": merged_df["train_accuracy"],
        "test_accuracy": merged_df["accuracy"],
        "overfit_gap": merged_df["train_accuracy"] - merged_df["accuracy"],
        "train_time_seconds": merged_df["train_time_seconds"],
        "predict_time_seconds": merged_df["predict_time_seconds"],
        "avg_latency_ms": merged_df["avg_latency_ms"],
    })

    return comparison_df


def build_robustness_plot_df(robustness_df: pd.DataFrame) -> pd.DataFrame:
    """
    构造鲁棒性曲线图数据。
    """
    model_name_map = build_model_name_map()

    result_df = robustness_df.copy()
    result_df["model"] = result_df["model"].map(model_name_map).fillna(result_df["model"])

    return result_df[["model", "noise_level", "accuracy", "drop"]]

# ============================================================
# ANN Loss Curve 可视化（新增）
# ============================================================

def plot_ann_loss_curve(model_path, output_path):
    """
    绘制ANN训练loss曲线（MLPClassifier）
    """

    import joblib
    import matplotlib.pyplot as plt

    model = joblib.load(model_path)

    if not hasattr(model, "loss_curve_"):
        print("当前ANN模型没有loss_curve_，无法绘图")
        return

    plt.figure(figsize=(8, 5))
    plt.plot(model.loss_curve_, label="Training Loss")

    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("ANN Loss Curve")
    plt.legend()
    plt.tight_layout()

    plt.savefig(output_path, dpi=300)
    plt.close()

# ============================================================
# 5. 主流程
# ============================================================

def main() -> None:
    """
    结果可视化主流程。
    """
    plot_utils = load_module_from_path("plot_utils", PLOT_UTILS_FILE)

    train_df, test_df, speed_df, robustness_df = load_result_tables()

    accuracy_plot_df = build_accuracy_plot_df(test_df)
    speed_plot_df = build_speed_plot_df(speed_df)
    comparison_df = build_train_test_comparison_df(train_df, test_df)
    robustness_plot_df = build_robustness_plot_df(robustness_df)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    comparison_df.to_csv(OVERFIT_TABLE_FILE, index=False, encoding="utf-8-sig")

    plot_utils.plot_test_accuracy_comparison(
        accuracy_plot_df,
        ACCURACY_FIGURE_FILE,
    )

    plot_utils.plot_train_test_accuracy_comparison(
        comparison_df,
        TRAIN_TEST_FIGURE_FILE,
    )

    plot_utils.plot_predict_time_comparison(
        speed_plot_df,
        SPEED_FIGURE_FILE,
    )

    plot_utils.plot_overfit_gap_comparison(
        comparison_df,
        OVERFIT_FIGURE_FILE,
    )

    plot_utils.plot_robustness_curve(
        robustness_plot_df,
        ROBUSTNESS_FIGURE_FILE,
    )

    print("=" * 70)
    print("第 13 步：结果可视化与图表汇总")
    print("=" * 70)

    print("输入结果文件：")
    print(TRAIN_SUMMARY_FILE)
    print(TEST_RESULTS_FILE)
    print(SPEED_TABLE_FILE)
    print(ROBUSTNESS_TABLE_FILE)

    print("-" * 70)
    print("输出结果文件：")
    print(OVERFIT_TABLE_FILE)

    print("-" * 70)
    print("输出图表文件：")
    print(ACCURACY_FIGURE_FILE)
    print(TRAIN_TEST_FIGURE_FILE)
    print(SPEED_FIGURE_FILE)
    print(OVERFIT_FIGURE_FILE)
    print(ROBUSTNESS_FIGURE_FILE)

    print("=" * 70)
    print("第 13 步完成：图表与过拟合结果已生成")
    print("=" * 70)


if __name__ == "__main__":
    main()