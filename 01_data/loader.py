# ============================================================
# DryBean-ML-Project-basic
# 第 1 步：数据读取模块
#
# 对应期末作业要求：
# 1. 工程项目中需要包含“数据加载模块”
# 2. 后续数据预处理、模型训练、模型测试都从这里统一读取原始数据
#
# 本文件功能：
# 1. 读取 01_data/raw/ 下的 train / val / test 三个原始数据文件
# 2. 自动识别标签列，优先识别 Class
# 3. 输出数据形状、字段名称、标签类别分布
# 4. 提供后续模块可复用的数据读取函数
#
# 注意：
# 本文件只做“读取数据”和“基础检查”
# 不做数据清洗、不做标准化、不做特征工程
# ============================================================

from pathlib import Path
import pandas as pd


# ============================================================
# 1. 项目路径配置
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_DIR = PROJECT_ROOT / "01_data" / "raw"

TRAIN_FILE = RAW_DATA_DIR / "Dry_Bean_Dataset_Dirty_train.csv"
VAL_FILE = RAW_DATA_DIR / "Dry_Bean_Dataset_Dirty_val.csv"
TEST_FILE = RAW_DATA_DIR / "Dry_Bean_Dataset_Dirty_test.csv"


# ============================================================
# 2. 文件检查与读取函数
# ============================================================

def check_file_exists(file_path: Path) -> None:
    """
    检查数据文件是否存在。

    参数：
        file_path: 数据文件路径

    返回：
        None

    如果文件不存在，抛出 FileNotFoundError。
    """
    if not file_path.exists():
        raise FileNotFoundError(f"数据文件不存在：{file_path}")


def read_csv_file(file_path: Path) -> pd.DataFrame:
    """
    读取 CSV 文件。

    参数：
        file_path: CSV 文件路径

    返回：
        data: pandas DataFrame
    """
    check_file_exists(file_path)
    data = pd.read_csv(file_path)
    return data


# ============================================================
# 3. 标签列识别与特征标签拆分
# ============================================================

def find_label_column(data: pd.DataFrame) -> str:
    """
    自动识别标签列。

    Dry Bean Dataset 通常使用 Class 作为标签列。
    如果不存在 Class，则默认使用最后一列作为标签列。

    参数：
        data: 输入数据表

    返回：
        label_column: 标签列名称
    """
    if "Class" in data.columns:
        return "Class"

    return data.columns[-1]


def split_features_and_label(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    将完整数据拆分为特征 X 和标签 y。

    参数：
        data: 包含特征和标签的数据表

    返回：
        X: 特征数据
        y: 标签数据
    """
    label_column = find_label_column(data)

    X = data.drop(columns=[label_column])
    y = data[label_column]

    return X, y


# ============================================================
# 4. 对外提供的统一数据读取接口
# ============================================================

def load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    读取原始训练集、验证集、测试集。

    返回：
        train_data: 原始训练集
        val_data: 原始验证集
        test_data: 原始测试集
    """
    train_data = read_csv_file(TRAIN_FILE)
    val_data = read_csv_file(VAL_FILE)
    test_data = read_csv_file(TEST_FILE)

    return train_data, val_data, test_data


def load_raw_train_val_test_xy() -> tuple[
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
    pd.Series,
]:
    """
    读取原始数据，并拆分为 X_train, y_train, X_val, y_val, X_test, y_test。

    返回：
        X_train, y_train
        X_val, y_val
        X_test, y_test
    """
    train_data, val_data, test_data = load_raw_data()

    X_train, y_train = split_features_and_label(train_data)
    X_val, y_val = split_features_and_label(val_data)
    X_test, y_test = split_features_and_label(test_data)

    return X_train, y_train, X_val, y_val, X_test, y_test


# ============================================================
# 5. 数据读取检查函数
# ============================================================

def print_data_overview() -> None:
    """
    打印原始数据的基本信息，用于验证数据读取是否成功。
    """
    train_data, val_data, test_data = load_raw_data()

    label_column = find_label_column(train_data)

    print("=" * 70)
    print("第 1 步：数据读取模块检查")
    print("=" * 70)

    print(f"项目根目录：{PROJECT_ROOT}")
    print(f"原始数据目录：{RAW_DATA_DIR}")

    print("-" * 70)
    print("数据文件读取成功")
    print(f"训练集形状：{train_data.shape}")
    print(f"验证集形状：{val_data.shape}")
    print(f"测试集形状：{test_data.shape}")

    print("-" * 70)
    print("字段信息")
    print(f"字段数量：{len(train_data.columns)}")
    print(f"字段名称：{list(train_data.columns)}")
    print(f"标签列：{label_column}")

    print("-" * 70)
    print("缺失值数量检查")
    print(f"训练集缺失值总数：{train_data.isnull().sum().sum()}")
    print(f"验证集缺失值总数：{val_data.isnull().sum().sum()}")
    print(f"测试集缺失值总数：{test_data.isnull().sum().sum()}")

    print("-" * 70)
    print("重复行数量检查")
    print(f"训练集重复行数量：{train_data.duplicated().sum()}")
    print(f"验证集重复行数量：{val_data.duplicated().sum()}")
    print(f"测试集重复行数量：{test_data.duplicated().sum()}")

    print("-" * 70)
    print("训练集标签类别分布")
    print(train_data[label_column].value_counts())

    print("=" * 70)
    print("第 1 步完成：原始数据读取正常")
    print("=" * 70)


# ============================================================
# 6. 命令行运行入口
# ============================================================

if __name__ == "__main__":
    print_data_overview()