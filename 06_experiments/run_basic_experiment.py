# ============================================================
# DryBean-ML-Project-basic
# 第 8 步：统一实验运行入口（修复 importlib 版本）
#
# 修复内容：
# ❌ 不再使用 from 07_utils import plot_utils
# ✔ 全部改为 importlib 动态加载
# ✔ 完全兼容 01/02/03 数字目录结构
#
# 功能：
# 1. 一键训练
# 2. 一键测试
# 3. 自动生成图表
# ============================================================

from pathlib import Path
import subprocess
import sys
import pandas as pd
import importlib.util


# ==============================
# 1. 项目路径
# ==============================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_SCRIPT = PROJECT_ROOT / "04_train" / "train_model.py"
TEST_SCRIPT = PROJECT_ROOT / "05_test" / "evaluate_model.py"

RESULT_DIR = PROJECT_ROOT / "08_results"
TRAIN_RESULT = RESULT_DIR / "train_summary.csv"
TEST_RESULT = RESULT_DIR / "test_results.csv"

FIGURE_DIR = RESULT_DIR / "figures"
ACC_PLOT = FIGURE_DIR / "accuracy.png"
TIME_PLOT = FIGURE_DIR / "time.png"

PLOT_UTILS_PATH = PROJECT_ROOT / "07_utils" / "plot_utils.py"


# ==============================
# 2. 动态加载模块（核心修复）
# ==============================

def load_module(name, path: Path):
    """
    通用 importlib 加载函数（解决数字目录 import 问题）
    """
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


plot_utils = load_module("plot_utils", PLOT_UTILS_PATH)


# ==============================
# 3. 训练阶段
# ==============================

def run_training():
    print("\n==============================")
    print("Step 1: 模型训练阶段")
    print("==============================\n")

    subprocess.run([
        sys.executable,
        str(TRAIN_SCRIPT),
        "--model",
        "all"
    ])


# ==============================
# 4. 测试阶段
# ==============================

def run_testing():
    print("\n==============================")
    print("Step 2: 模型测试阶段")
    print("==============================\n")

    subprocess.run([
        sys.executable,
        str(TEST_SCRIPT)
    ])


# ==============================
# 5. 画图阶段
# ==============================

def generate_plots():

    print("\n==============================")
    print("Step 3: 生成图表")
    print("==============================\n")

    train_df = pd.read_csv(TRAIN_RESULT)
    test_df = pd.read_csv(TEST_RESULT)

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    # 精度图
    plot_utils.plot_test_accuracy_comparison(
        test_df,
        ACC_PLOT
    )

    # 时间图
    plot_utils.plot_predict_time_comparison(
        test_df,
        TIME_PLOT
    )

    print("图表生成完成：")
    print(ACC_PLOT)
    print(TIME_PLOT)


# ==============================
# 6. 主流程
# ==============================

def main():

    print("\n" + "=" * 70)
    print("DryBean-ML-Project-basic 一键实验系统启动")
    print("=" * 70)

    run_training()
    run_testing()
    generate_plots()

    print("\n" + "=" * 70)
    print("实验全部完成")
    print("结果已保存至 08_results/")
    print("=" * 70)


# ==============================
# 7. 入口
# ==============================

if __name__ == "__main__":
    main()