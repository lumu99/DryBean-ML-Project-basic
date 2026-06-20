# ============================================================
# DryBean-ML-Project-basic
# 第 3 步：特征工程模块
#
# 对应期末作业要求：
# 1. 数据处理部分不仅要完成数据清洗，还要完成特征工程
# 2. 本文件基于 01_data/processed/ 中已经清洗好的数据进行特征工程分析
#
# 本文件功能：
# 1. 读取 train_processed / val_processed / test_processed
# 2. 统计特征基本信息
# 3. 计算训练集特征相关性矩阵
# 4. 识别高度相关特征对
# 5. 保存特征工程相关结果，作为后续论文素材
#
# 注意：
# 本阶段先不删除特征，避免影响后续基础模型训练。
# 这里的特征工程重点是“特征分析 + 冗余特征识别 + 论文素材保存”。
# ============================================================

from pathlib import Path

import pandas as pd


# ============================================================
# 1. 项目路径配置
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DATA_DIR = PROJECT_ROOT / "01_data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "08_results"
FIGURES_DIR = RESULTS_DIR / "figures"

TRAIN_PROCESSED_FILE = PROCESSED_DATA_DIR / "train_processed.csv"
VAL_PROCESSED_FILE = PROCESSED_DATA_DIR / "val_processed.csv"
TEST_PROCESSED_FILE = PROCESSED_DATA_DIR / "test_processed.csv"

FEATURE_STATISTICS_FILE = PROCESSED_DATA_DIR / "feature_statistics.csv"
FEATURE_CORRELATION_FILE = PROCESSED_DATA_DIR / "feature_correlation_matrix.csv"
HIGH_CORRELATION_FILE = PROCESSED_DATA_DIR / "high_correlation_features.csv"
FEATURE_ENGINEERING_SUMMARY_FILE = PROCESSED_DATA_DIR / "feature_engineering_summary.txt"


# ============================================================
# 2. 数据读取
# ============================================================

def check_file_exists(file_path: Path) -> None:
    """
    检查文件是否存在。

    参数：
        file_path: 文件路径
    """
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在：{file_path}")


def load_processed_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    读取已经完成清洗和标准化的 processed 数据。

    返回：
        train_data: 处理后的训练集
        val_data: 处理后的验证集
        test_data: 处理后的测试集
    """
    check_file_exists(TRAIN_PROCESSED_FILE)
    check_file_exists(VAL_PROCESSED_FILE)
    check_file_exists(TEST_PROCESSED_FILE)

    train_data = pd.read_csv(TRAIN_PROCESSED_FILE)
    val_data = pd.read_csv(VAL_PROCESSED_FILE)
    test_data = pd.read_csv(TEST_PROCESSED_FILE)

    return train_data, val_data, test_data


def get_feature_columns(data: pd.DataFrame) -> list[str]:
    """
    获取特征列名称。

    参数：
        data: 输入数据

    返回：
        feature_columns: 特征列列表
    """
    feature_columns = [col for col in data.columns if col != "Class_Label"]
    return feature_columns


# ============================================================
# 3. 特征统计分析
# ============================================================

def calculate_feature_statistics(
    train_data: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """
    计算训练集特征统计信息。

    包括：
    1. 均值
    2. 标准差
    3. 最小值
    4. 最大值
    5. 缺失值数量

    参数：
        train_data: 处理后的训练集
        feature_columns: 特征列列表

    返回：
        statistics_df: 特征统计信息表
    """
    statistics_df = pd.DataFrame({
        "feature": feature_columns,
        "mean": train_data[feature_columns].mean().values,
        "std": train_data[feature_columns].std().values,
        "min": train_data[feature_columns].min().values,
        "max": train_data[feature_columns].max().values,
        "missing_count": train_data[feature_columns].isnull().sum().values,
    })

    return statistics_df


# ============================================================
# 4. 特征相关性分析
# ============================================================

def calculate_correlation_matrix(
    train_data: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """
    计算训练集特征相关性矩阵。

    参数：
        train_data: 处理后的训练集
        feature_columns: 特征列列表

    返回：
        correlation_matrix: 特征相关性矩阵
    """
    correlation_matrix = train_data[feature_columns].corr()
    return correlation_matrix


def find_high_correlation_pairs(
    correlation_matrix: pd.DataFrame,
    threshold: float = 0.95,
) -> pd.DataFrame:
    """
    查找高度相关的特征对。

    参数：
        correlation_matrix: 特征相关性矩阵
        threshold: 相关系数绝对值阈值

    返回：
        high_corr_df: 高度相关特征对表格
    """
    high_corr_pairs = []

    columns = correlation_matrix.columns.tolist()

    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            feature_1 = columns[i]
            feature_2 = columns[j]
            corr_value = correlation_matrix.loc[feature_1, feature_2]

            if abs(corr_value) >= threshold:
                high_corr_pairs.append({
                    "feature_1": feature_1,
                    "feature_2": feature_2,
                    "correlation": round(float(corr_value), 6),
                    "abs_correlation": round(abs(float(corr_value)), 6),
                })

    high_corr_df = pd.DataFrame(high_corr_pairs)

    if not high_corr_df.empty:
        high_corr_df = high_corr_df.sort_values(
            by="abs_correlation",
            ascending=False,
        )

    return high_corr_df


# ============================================================
# 5. 保存特征工程结果
# ============================================================

def save_feature_engineering_results(
    statistics_df: pd.DataFrame,
    correlation_matrix: pd.DataFrame,
    high_corr_df: pd.DataFrame,
    summary_text: str,
) -> None:
    """
    保存特征工程分析结果。

    输出文件：
    1. feature_statistics.csv
    2. feature_correlation_matrix.csv
    3. high_correlation_features.csv
    4. feature_engineering_summary.txt
    """
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    statistics_df.to_csv(FEATURE_STATISTICS_FILE, index=False, encoding="utf-8-sig")
    correlation_matrix.to_csv(FEATURE_CORRELATION_FILE, encoding="utf-8-sig")
    high_corr_df.to_csv(HIGH_CORRELATION_FILE, index=False, encoding="utf-8-sig")

    with open(FEATURE_ENGINEERING_SUMMARY_FILE, "w", encoding="utf-8") as f:
        f.write(summary_text)


# ============================================================
# 6. 主特征工程流程
# ============================================================

def run_feature_engineering() -> None:
    """
    执行基础特征工程分析流程。
    """
    train_data, val_data, test_data = load_processed_data()

    feature_columns = get_feature_columns(train_data)

    statistics_df = calculate_feature_statistics(train_data, feature_columns)
    correlation_matrix = calculate_correlation_matrix(train_data, feature_columns)
    high_corr_df = find_high_correlation_pairs(correlation_matrix, threshold=0.95)

    summary_text = f"""
DryBean-ML-Project-basic
第 3 步：特征工程分析摘要

一、输入数据
训练集：{TRAIN_PROCESSED_FILE}
验证集：{VAL_PROCESSED_FILE}
测试集：{TEST_PROCESSED_FILE}

二、特征数量
当前特征数量：{len(feature_columns)}
特征名称：
{feature_columns}

三、特征工程处理内容
1. 基于训练集统计各特征的均值、标准差、最小值、最大值。
2. 基于训练集计算特征相关性矩阵。
3. 使用相关系数绝对值 >= 0.95 作为阈值，识别高度相关特征对。
4. 本基础阶段暂不删除特征，避免影响后续基础模型训练的完整性。
5. 高相关特征对会在论文中作为“特征冗余分析”的依据。

四、高度相关特征对数量
高度相关特征对数量：{len(high_corr_df)}

五、输出文件
{FEATURE_STATISTICS_FILE}
{FEATURE_CORRELATION_FILE}
{HIGH_CORRELATION_FILE}
{FEATURE_ENGINEERING_SUMMARY_FILE}
""".strip()

    save_feature_engineering_results(
        statistics_df,
        correlation_matrix,
        high_corr_df,
        summary_text,
    )

    print("=" * 70)
    print("第 3 步：特征工程模块")
    print("=" * 70)

    print("输入数据读取成功")
    print(f"训练集形状：{train_data.shape}")
    print(f"验证集形状：{val_data.shape}")
    print(f"测试集形状：{test_data.shape}")

    print("-" * 70)
    print("特征信息")
    print(f"特征数量：{len(feature_columns)}")
    print(f"特征名称：{feature_columns}")

    print("-" * 70)
    print("高度相关特征对")
    print(f"相关系数阈值：0.95")
    print(f"高度相关特征对数量：{len(high_corr_df)}")

    if not high_corr_df.empty:
        print(high_corr_df.head(10))
    else:
        print("未发现高度相关特征对")

    print("-" * 70)
    print("特征工程结果已保存")
    print(FEATURE_STATISTICS_FILE)
    print(FEATURE_CORRELATION_FILE)
    print(HIGH_CORRELATION_FILE)
    print(FEATURE_ENGINEERING_SUMMARY_FILE)

    print("=" * 70)
    print("第 3 步完成：特征统计与相关性分析已完成")
    print("=" * 70)


# ============================================================
# 7. 命令行运行入口
# ============================================================

if __name__ == "__main__":
    run_feature_engineering()