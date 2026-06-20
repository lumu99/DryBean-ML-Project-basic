# ============================================================
# DryBean-ML-Project-basic
# 第 2 步：数据预处理模块
#
# 对应期末作业要求：
# 1. 数据处理需要完成数据清洗和特征工程基础处理
# 2. 本文件先完成基础数据清洗，包括：
#    - 标签清洗
#    - 缺失值处理
#    - 重复行处理
#    - 特征标准化
#    - 标签编码
# 3. 处理后的数据保存到 01_data/processed/，供后续模型训练与测试使用
#
# 注意：
# 由于 01_data 目录以数字开头，不能直接使用普通 import。
# 本文件使用 importlib.util 按文件路径加载 01_data/loader.py。
# ============================================================

from pathlib import Path
import importlib.util
import json

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder


# ============================================================
# 1. 项目路径配置
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

LOADER_PATH = PROJECT_ROOT / "01_data" / "loader.py"
PROCESSED_DATA_DIR = PROJECT_ROOT / "01_data" / "processed"

TRAIN_PROCESSED_FILE = PROCESSED_DATA_DIR / "train_processed.csv"
VAL_PROCESSED_FILE = PROCESSED_DATA_DIR / "val_processed.csv"
TEST_PROCESSED_FILE = PROCESSED_DATA_DIR / "test_processed.csv"

LABEL_MAPPING_FILE = PROCESSED_DATA_DIR / "label_mapping.json"
PREPROCESS_SUMMARY_FILE = PROCESSED_DATA_DIR / "preprocess_summary.txt"


# ============================================================
# 2. 加载 01_data/loader.py
# ============================================================

def load_loader_module():
    """
    按文件路径加载 01_data/loader.py。

    原因：
        01_data 目录以数字开头，不能写成 from 01_data.loader import xxx。
        因此这里使用 importlib.util 进行路径加载。
    """
    spec = importlib.util.spec_from_file_location("loader", LOADER_PATH)
    loader = importlib.util.module_from_spec(spec)

    if spec.loader is None:
        raise ImportError(f"无法加载数据读取模块：{LOADER_PATH}")

    spec.loader.exec_module(loader)
    return loader


# ============================================================
# 3. 标签清洗
# ============================================================

def clean_label(label: object) -> str:
    """
    清洗 Dry Bean 标签中的脏数据。

    处理内容：
    1. 去除前后空格
    2. 转为大写
    3. 将常见数字污染还原：
       - 0 -> O
       - 3 -> E
    4. 将清洗后的标签统一为标准类别名称

    参数：
        label: 原始标签

    返回：
        清洗后的标准标签
    """
    if pd.isna(label):
        return "UNKNOWN"

    label_str = str(label).strip().upper()

    label_str = label_str.replace("0", "O")
    label_str = label_str.replace("3", "E")

    valid_labels = {
        "BARBUNYA",
        "BOMBAY",
        "CALI",
        "DERMASON",
        "HOROZ",
        "SEKER",
        "SIRA",
    }

    if label_str in valid_labels:
        return label_str

    return label_str


def clean_label_column(data: pd.DataFrame, label_column: str) -> pd.DataFrame:
    """
    对数据中的标签列进行清洗。

    参数：
        data: 输入数据
        label_column: 标签列名称

    返回：
        清洗后的数据
    """
    data = data.copy()
    data[label_column] = data[label_column].apply(clean_label)
    return data


# ============================================================
# 4. 特征清洗
# ============================================================

def clean_feature_columns(data: pd.DataFrame, label_column: str) -> pd.DataFrame:
    """
    清洗特征列。

    处理内容：
    1. 特征列转为数值类型
    2. 无法转换的值变为 NaN
    3. 将 inf 和 -inf 替换为 NaN

    参数：
        data: 输入数据
        label_column: 标签列名称

    返回：
        清洗后的数据
    """
    data = data.copy()

    feature_columns = [col for col in data.columns if col != label_column]

    for col in feature_columns:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data.replace([np.inf, -np.inf], np.nan, inplace=True)

    return data


def remove_duplicate_rows(data: pd.DataFrame) -> pd.DataFrame:
    """
    删除重复行。

    参数：
        data: 输入数据

    返回：
        删除重复行后的数据
    """
    data = data.copy()
    data = data.drop_duplicates()
    return data


# ============================================================
# 5. 缺失值处理与标准化
# ============================================================

def fill_missing_values(
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    test_data: pd.DataFrame,
    label_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """
    使用训练集每个特征的中位数填补缺失值。

    注意：
    验证集和测试集必须使用训练集统计量，避免数据泄漏。

    返回：
        填补后的 train / val / test
        median_values: 训练集各特征中位数
    """
    train_data = train_data.copy()
    val_data = val_data.copy()
    test_data = test_data.copy()

    feature_columns = [col for col in train_data.columns if col != label_column]

    median_values = train_data[feature_columns].median().to_dict()

    train_data[feature_columns] = train_data[feature_columns].fillna(median_values)
    val_data[feature_columns] = val_data[feature_columns].fillna(median_values)
    test_data[feature_columns] = test_data[feature_columns].fillna(median_values)

    return train_data, val_data, test_data, median_values


def standardize_features(
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    test_data: pd.DataFrame,
    label_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    使用 StandardScaler 对特征进行标准化。

    注意：
    scaler 只在训练集上 fit。
    验证集和测试集只 transform，避免数据泄漏。

    返回：
        标准化后的 train / val / test
    """
    train_data = train_data.copy()
    val_data = val_data.copy()
    test_data = test_data.copy()

    feature_columns = [col for col in train_data.columns if col != label_column]

    scaler = StandardScaler()
    scaler.fit(train_data[feature_columns])

    train_data[feature_columns] = scaler.transform(train_data[feature_columns])
    val_data[feature_columns] = scaler.transform(val_data[feature_columns])
    test_data[feature_columns] = scaler.transform(test_data[feature_columns])

    return train_data, val_data, test_data


# ============================================================
# 6. 标签编码
# ============================================================

def encode_labels(
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    test_data: pd.DataFrame,
    label_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """
    使用 LabelEncoder 将类别标签编码为数字。

    处理后：
    1. 删除原始 Class 字符串列
    2. 新增 Class_Label 数值标签列

    返回：
        编码后的 train / val / test
        label_mapping: 数字标签与类别名称的对应关系
    """
    train_data = train_data.copy()
    val_data = val_data.copy()
    test_data = test_data.copy()

    label_encoder = LabelEncoder()
    label_encoder.fit(train_data[label_column])

    train_labels = label_encoder.transform(train_data[label_column])
    val_labels = label_encoder.transform(val_data[label_column])
    test_labels = label_encoder.transform(test_data[label_column])

    label_mapping = {
        int(index): class_name
        for index, class_name in enumerate(label_encoder.classes_)
    }

    train_data = train_data.drop(columns=[label_column])
    val_data = val_data.drop(columns=[label_column])
    test_data = test_data.drop(columns=[label_column])

    train_data["Class_Label"] = train_labels
    val_data["Class_Label"] = val_labels
    test_data["Class_Label"] = test_labels

    return train_data, val_data, test_data, label_mapping


# ============================================================
# 7. 保存结果
# ============================================================

def save_processed_data(
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    test_data: pd.DataFrame,
    label_mapping: dict,
    summary_text: str,
) -> None:
    """
    保存处理后的数据、标签映射和预处理摘要。
    """
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    train_data.to_csv(TRAIN_PROCESSED_FILE, index=False, encoding="utf-8-sig")
    val_data.to_csv(VAL_PROCESSED_FILE, index=False, encoding="utf-8-sig")
    test_data.to_csv(TEST_PROCESSED_FILE, index=False, encoding="utf-8-sig")

    with open(LABEL_MAPPING_FILE, "w", encoding="utf-8") as f:
        json.dump(label_mapping, f, ensure_ascii=False, indent=4)

    with open(PREPROCESS_SUMMARY_FILE, "w", encoding="utf-8") as f:
        f.write(summary_text)


# ============================================================
# 8. 主预处理流程
# ============================================================

def run_preprocess() -> None:
    """
    执行完整基础预处理流程。
    """
    loader = load_loader_module()

    train_data, val_data, test_data = loader.load_raw_data()
    label_column = loader.find_label_column(train_data)

    original_train_shape = train_data.shape
    original_val_shape = val_data.shape
    original_test_shape = test_data.shape

    original_train_missing = int(train_data.isnull().sum().sum())
    original_val_missing = int(val_data.isnull().sum().sum())
    original_test_missing = int(test_data.isnull().sum().sum())

    original_train_duplicates = int(train_data.duplicated().sum())

    original_train_label_counts = train_data[label_column].value_counts().to_dict()

    train_data = clean_label_column(train_data, label_column)
    val_data = clean_label_column(val_data, label_column)
    test_data = clean_label_column(test_data, label_column)

    train_data = clean_feature_columns(train_data, label_column)
    val_data = clean_feature_columns(val_data, label_column)
    test_data = clean_feature_columns(test_data, label_column)

    train_data = remove_duplicate_rows(train_data)

    cleaned_train_shape = train_data.shape

    train_data, val_data, test_data, median_values = fill_missing_values(
        train_data,
        val_data,
        test_data,
        label_column,
    )

    train_data, val_data, test_data = standardize_features(
        train_data,
        val_data,
        test_data,
        label_column,
    )

    train_data, val_data, test_data, label_mapping = encode_labels(
        train_data,
        val_data,
        test_data,
        label_column,
    )

    final_train_missing = int(train_data.isnull().sum().sum())
    final_val_missing = int(val_data.isnull().sum().sum())
    final_test_missing = int(test_data.isnull().sum().sum())

    summary_text = f"""
DryBean-ML-Project-basic
第 2 步：数据预处理摘要

一、原始数据形状
训练集：{original_train_shape}
验证集：{original_val_shape}
测试集：{original_test_shape}

二、原始缺失值数量
训练集：{original_train_missing}
验证集：{original_val_missing}
测试集：{original_test_missing}

三、重复行处理
训练集原始重复行数量：{original_train_duplicates}
训练集删除重复行后形状：{cleaned_train_shape}

四、标签清洗说明
原始训练集标签分布：
{original_train_label_counts}

清洗后标签映射：
{label_mapping}

五、缺失值处理方法
使用训练集每个特征的中位数填补 train / val / test 中的缺失值。

六、标准化方法
使用 StandardScaler。
只在训练集上 fit，在验证集和测试集上 transform，避免数据泄漏。

七、最终缺失值数量
训练集：{final_train_missing}
验证集：{final_val_missing}
测试集：{final_test_missing}

八、输出文件
{TRAIN_PROCESSED_FILE}
{VAL_PROCESSED_FILE}
{TEST_PROCESSED_FILE}
{LABEL_MAPPING_FILE}
""".strip()

    save_processed_data(
        train_data,
        val_data,
        test_data,
        label_mapping,
        summary_text,
    )

    print("=" * 70)
    print("第 2 步：数据预处理模块")
    print("=" * 70)

    print("原始数据形状")
    print(f"训练集：{original_train_shape}")
    print(f"验证集：{original_val_shape}")
    print(f"测试集：{original_test_shape}")

    print("-" * 70)
    print("原始缺失值数量")
    print(f"训练集：{original_train_missing}")
    print(f"验证集：{original_val_missing}")
    print(f"测试集：{original_test_missing}")

    print("-" * 70)
    print("重复行处理")
    print(f"训练集原始重复行数量：{original_train_duplicates}")
    print(f"训练集删除重复行后形状：{cleaned_train_shape}")

    print("-" * 70)
    print("标签映射")
    print(label_mapping)

    print("-" * 70)
    print("最终缺失值数量")
    print(f"训练集：{final_train_missing}")
    print(f"验证集：{final_val_missing}")
    print(f"测试集：{final_test_missing}")

    print("-" * 70)
    print("处理后数据已保存")
    print(TRAIN_PROCESSED_FILE)
    print(VAL_PROCESSED_FILE)
    print(TEST_PROCESSED_FILE)
    print(LABEL_MAPPING_FILE)
    print(PREPROCESS_SUMMARY_FILE)

    print("=" * 70)
    print("第 2 步完成：数据清洗、标准化和标签编码已完成")
    print("=" * 70)


# ============================================================
# 9. 命令行运行入口
# ============================================================

if __name__ == "__main__":
    run_preprocess()