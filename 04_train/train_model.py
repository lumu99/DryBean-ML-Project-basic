# ============================================================
# DryBean-ML-Project-basic
# 第 6 步：统一训练模块
#
# 对应期末作业要求：
# 1. 工程项目需要包含“调用算法训练模块”
# 2. 后续需要用统一命令训练多个分类算法
# 3. 本文件负责训练模型并保存模型文件
#
# 本文件功能：
# 1. 读取 01_data/processed/ 中的训练集和验证集
# 2. 按模型名称加载 03_models/ 中的模型
# 3. 训练 KNN / SVM / Random Forest
# 4. 计算训练集准确率和验证集准确率
# 5. 保存训练后的模型到 08_results/models/
#
# ✔ 5个模型训练
# ✔ ANN + XGBoost加入
# ✔ train_summary.csv自动更新
# 注意：
# 由于 03_models、07_utils 等目录前面有数字，不能直接使用普通 import。
# 本文件使用 importlib.util 按文件路径加载对应模块。
# ============================================================

from pathlib import Path
import importlib.util
import argparse
import json

import pandas as pd
import joblib


# ============================================================
# 1. 项目路径配置
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DATA_DIR = PROJECT_ROOT / "01_data" / "processed"
MODELS_CODE_DIR = PROJECT_ROOT / "03_models"
UTILS_DIR = PROJECT_ROOT / "07_utils"
RESULTS_DIR = PROJECT_ROOT / "08_results"
SAVED_MODELS_DIR = RESULTS_DIR / "models"

TRAIN_PROCESSED_FILE = PROCESSED_DATA_DIR / "train_processed.csv"
VAL_PROCESSED_FILE = PROCESSED_DATA_DIR / "val_processed.csv"
LABEL_MAPPING_FILE = PROCESSED_DATA_DIR / "label_mapping.json"

TRAIN_SUMMARY_FILE = RESULTS_DIR / "train_summary.csv"


# ============================================================
# 2. 通用模块加载函数
# ============================================================

def load_module_from_path(module_name: str, module_path: Path):
    """
    按文件路径加载 Python 模块。

    参数：
        module_name: 模块名称
        module_path: 模块文件路径

    返回：
        module: 加载后的 Python 模块
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


def load_processed_train_val_data():
    """
    读取已经预处理好的训练集和验证集。

    返回：
        X_train, y_train, X_val, y_val
    """
    check_file_exists(TRAIN_PROCESSED_FILE)
    check_file_exists(VAL_PROCESSED_FILE)

    train_data = pd.read_csv(TRAIN_PROCESSED_FILE)
    val_data = pd.read_csv(VAL_PROCESSED_FILE)

    X_train = train_data.drop(columns=["Class_Label"])
    y_train = train_data["Class_Label"]

    X_val = val_data.drop(columns=["Class_Label"])
    y_val = val_data["Class_Label"]

    return X_train, y_train, X_val, y_val


# ============================================================
# 4. 模型加载函数
# ============================================================

def get_model_file_path(model_key: str) -> Path:
    """
    根据模型名称返回模型代码文件路径。

    支持模型：
        knn
        svm
        random_forest
        ann
        xgboost
    """

    model_file_map = {
        "knn": MODELS_CODE_DIR / "knn.py",
        "svm": MODELS_CODE_DIR / "svm.py",
        "random_forest": MODELS_CODE_DIR / "random_forest.py",
        "ann": MODELS_CODE_DIR / "ann.py",
        "xgboost": MODELS_CODE_DIR / "xgboost.py",
    }

    if model_key not in model_file_map:
        raise ValueError(
            f"不支持的模型名称：{model_key}。"
            f"当前支持：{list(model_file_map.keys())}"
        )

    return model_file_map[model_key]


def create_model_by_key(model_key: str):
    """
    根据模型名称创建模型对象。
    """
    model_file_path = get_model_file_path(model_key)
    model_module = load_module_from_path(model_key, model_file_path)

    model = model_module.create_model()
    model_name = model_module.get_model_name()

    return model_name, model


# ============================================================
# 5. 单模型训练函数
# ============================================================

def train_one_model(model_key: str) -> dict:
    """
    训练一个指定模型，并保存模型文件。
    """
    timer_module = load_module_from_path("timer", UTILS_DIR / "timer.py")
    metrics_module = load_module_from_path("metrics", UTILS_DIR / "metrics.py")

    X_train, y_train, X_val, y_val = load_processed_train_val_data()

    model_name, model = create_model_by_key(model_key)

    SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    trained_model, train_time = timer_module.time_model_fit(
        model,
        X_train,
        y_train,
    )

    train_pred = trained_model.predict(X_train)
    val_pred = trained_model.predict(X_val)

    train_accuracy = metrics_module.calculate_accuracy(y_train, train_pred)
    val_accuracy = metrics_module.calculate_accuracy(y_val, val_pred)

    model_save_path = SAVED_MODELS_DIR / f"{model_key}.joblib"
    joblib.dump(trained_model, model_save_path)

    result_row = {
        "model_key": model_key,
        "model_name": model_name,
        "train_accuracy": round(train_accuracy, 6),
        "val_accuracy": round(val_accuracy, 6),
        "train_time_seconds": round(train_time, 6),
        "model_file": str(model_save_path),
    }

    return result_row


# ============================================================
# 6. 多模型训练函数
# ============================================================

def train_models(model_keys: list[str]) -> pd.DataFrame:
    """
    按列表训练多个模型。
    """
    results = []

    print("=" * 70)
    print("第 6 步：统一训练模块")
    print("=" * 70)

    for model_key in model_keys:
        print(f"开始训练模型：{model_key}")
        result_row = train_one_model(model_key)
        results.append(result_row)

        print(
            f"完成训练：{result_row['model_name']} | "
            f"训练集准确率：{result_row['train_accuracy']} | "
            f"验证集准确率：{result_row['val_accuracy']} | "
            f"训练耗时：{result_row['train_time_seconds']} 秒"
        )

    result_df = pd.DataFrame(results)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(TRAIN_SUMMARY_FILE, index=False, encoding="utf-8-sig")

    print("-" * 70)
    print("训练结果已保存")
    print(TRAIN_SUMMARY_FILE)

    print("=" * 70)
    print("第 6 步完成：模型训练与保存已完成")
    print("=" * 70)

    return result_df


# ============================================================
# 7. 命令行参数入口
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Dry Bean 多分类模型统一训练模块"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["all", "knn", "svm", "random_forest", "ann", "xgboost"],
        help="选择需要训练的模型",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.model == "all":
        model_keys = ["knn", "svm", "random_forest", "ann", "xgboost"]
    else:
        model_keys = [args.model]

    train_models(model_keys)


if __name__ == "__main__":
    main()