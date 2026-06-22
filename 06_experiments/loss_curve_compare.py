# -*- coding: utf-8 -*-
"""
机器学习与项目实践期末项目
Loss 曲线对比实验脚本

作用：
1. 按期末作业要求补充 loss 曲线分析；
2. 只对具有迭代训练损失记录的模型绘制 loss 曲线；
3. 不对 KNN、SVM、Random Forest 强行绘制 loss，避免不严谨；
4. 基于已经预处理完成的数据重新训练 ANN 和 XGBoost，用于记录 loss 变化；
5. 输出 loss 曲线图片和 loss 数值表，供论文使用。

输出：
08_results/figures/loss_curve_comparison.png
08_results/figures/ann_loss_curve.png
08_results/figures/xgboost_logloss_curve.png
08_results/loss_curve_ann.csv
08_results/loss_curve_xgboost.csv
08_results/loss_curve_summary.txt
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier


warnings.filterwarnings("ignore")


# ==============================
# 1. 路径配置
# ==============================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = PROJECT_ROOT / "01_data" / "processed" / "train_processed.csv"
VAL_PATH = PROJECT_ROOT / "01_data" / "processed" / "val_processed.csv"

RESULT_DIR = PROJECT_ROOT / "08_results"
FIGURE_DIR = RESULT_DIR / "figures"

RESULT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# ==============================
# 2. 绘图风格配置
# ==============================

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 300


ANN_COLOR = "#4C78A8"
XGB_COLOR = "#59A14F"
XGB_TRAIN_COLOR = "#F28E2B"


# ==============================
# 3. 数据读取函数
# ==============================

def load_processed_data():
    """
    读取已经预处理完成的训练集和验证集。

    要求：
    1. train_processed.csv 和 val_processed.csv 已经存在；
    2. 标签列优先识别为 Class；
    3. 如果不存在 Class，则默认最后一列为标签列。
    """
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(f"未找到训练数据文件：{TRAIN_PATH}")

    if not VAL_PATH.exists():
        raise FileNotFoundError(f"未找到验证数据文件：{VAL_PATH}")

    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)

    if "Class" in train_df.columns:
        label_col = "Class"
    else:
        label_col = train_df.columns[-1]

    X_train = train_df.drop(columns=[label_col])
    y_train = train_df[label_col].astype(int)

    X_val = val_df.drop(columns=[label_col])
    y_val = val_df[label_col].astype(int)

    print("数据读取完成")
    print(f"训练集特征：{X_train.shape}")
    print(f"验证集特征：{X_val.shape}")
    print(f"标签列：{label_col}")

    return X_train, y_train, X_val, y_val


# ==============================
# 4. ANN Loss 曲线
# ==============================

def train_ann_and_record_loss(X_train, y_train):
    """
    训练 ANN，并记录 sklearn MLPClassifier 的 loss_curve_。
    """
    print("\n开始训练 ANN，用于记录 loss_curve_ ...")

    ann_model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=0.0001,
        batch_size="auto",
        learning_rate="adaptive",
        learning_rate_init=0.001,
        max_iter=300,
        random_state=42,
        early_stopping=False,
        verbose=False,
    )

    ann_model.fit(X_train, y_train)

    ann_loss = pd.DataFrame({
        "iteration": np.arange(1, len(ann_model.loss_curve_) + 1),
        "ann_loss": ann_model.loss_curve_,
    })

    ann_loss_path = RESULT_DIR / "loss_curve_ann.csv"
    ann_loss.to_csv(ann_loss_path, index=False, encoding="utf-8-sig")

    print(f"ANN Loss 曲线数据已保存：{ann_loss_path}")
    print(f"ANN 迭代轮数：{len(ann_model.loss_curve_)}")
    print(f"ANN 初始 loss：{ann_model.loss_curve_[0]:.6f}")
    print(f"ANN 最终 loss：{ann_model.loss_curve_[-1]:.6f}")

    return ann_loss


def plot_ann_loss(ann_loss):
    """
    单独绘制 ANN Loss 曲线。
    """
    plt.figure(figsize=(8, 5))
    plt.plot(
        ann_loss["iteration"],
        ann_loss["ann_loss"],
        color=ANN_COLOR,
        linewidth=2,
        label="ANN Training Loss",
    )
    plt.xlabel("迭代轮数")
    plt.ylabel("Loss")
    plt.title("ANN 训练 Loss 曲线", fontsize=14, fontweight="bold")
    plt.grid(linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()

    output_path = FIGURE_DIR / "ann_loss_curve.png"
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

    print(f"ANN Loss 曲线图已保存：{output_path}")


# ==============================
# 5. XGBoost Logloss 曲线
# ==============================

def train_xgboost_and_record_loss(X_train, y_train, X_val, y_val):
    """
    训练 XGBoost，并通过 eval_set 记录训练集和验证集 mlogloss。
    """
    print("\n开始训练 XGBoost，用于记录 mlogloss ...")

    xgb_model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
    )

    xgb_model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=False,
    )

    evals_result = xgb_model.evals_result()

    train_loss = evals_result["validation_0"]["mlogloss"]
    val_loss = evals_result["validation_1"]["mlogloss"]

    xgb_loss = pd.DataFrame({
        "iteration": np.arange(1, len(train_loss) + 1),
        "xgboost_train_mlogloss": train_loss,
        "xgboost_val_mlogloss": val_loss,
    })

    xgb_loss_path = RESULT_DIR / "loss_curve_xgboost.csv"
    xgb_loss.to_csv(xgb_loss_path, index=False, encoding="utf-8-sig")

    print(f"XGBoost Loss 曲线数据已保存：{xgb_loss_path}")
    print(f"XGBoost 迭代轮数：{len(train_loss)}")
    print(f"XGBoost 初始验证 mlogloss：{val_loss[0]:.6f}")
    print(f"XGBoost 最终验证 mlogloss：{val_loss[-1]:.6f}")

    return xgb_loss


def plot_xgboost_loss(xgb_loss):
    """
    单独绘制 XGBoost 训练集和验证集 logloss 曲线。
    """
    plt.figure(figsize=(8, 5))
    plt.plot(
        xgb_loss["iteration"],
        xgb_loss["xgboost_train_mlogloss"],
        color=XGB_TRAIN_COLOR,
        linewidth=2,
        label="XGBoost Train mlogloss",
    )
    plt.plot(
        xgb_loss["iteration"],
        xgb_loss["xgboost_val_mlogloss"],
        color=XGB_COLOR,
        linewidth=2,
        label="XGBoost Val mlogloss",
    )
    plt.xlabel("迭代轮数")
    plt.ylabel("mlogloss")
    plt.title("XGBoost 训练与验证 Logloss 曲线", fontsize=14, fontweight="bold")
    plt.grid(linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()

    output_path = FIGURE_DIR / "xgboost_logloss_curve.png"
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

    print(f"XGBoost Logloss 曲线图已保存：{output_path}")


# ==============================
# 6. Loss 对比图
# ==============================

def plot_loss_comparison(ann_loss, xgb_loss):
    """
    绘制 ANN 与 XGBoost 的 Loss 对比图。

    注意：
    ANN 使用 sklearn MLPClassifier 的训练 loss；
    XGBoost 使用验证集 mlogloss。
    两者都反映迭代过程中损失变化趋势，但定义不完全相同。
    论文中应说明该点。
    """
    plt.figure(figsize=(8, 5))

    plt.plot(
        ann_loss["iteration"],
        ann_loss["ann_loss"],
        color=ANN_COLOR,
        linewidth=2,
        label="ANN Training Loss",
    )

    plt.plot(
        xgb_loss["iteration"],
        xgb_loss["xgboost_val_mlogloss"],
        color=XGB_COLOR,
        linewidth=2,
        label="XGBoost Val mlogloss",
    )

    plt.xlabel("迭代轮数")
    plt.ylabel("Loss / mlogloss")
    plt.title("ANN 与 XGBoost Loss 曲线对比", fontsize=14, fontweight="bold")
    plt.grid(linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()

    output_path = FIGURE_DIR / "loss_curve_comparison.png"
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

    print(f"Loss 曲线对比图已保存：{output_path}")


# ==============================
# 7. 输出说明文件
# ==============================

def save_summary(ann_loss, xgb_loss):
    """
    保存 Loss 曲线实验说明，方便论文引用。
    """
    summary_path = RESULT_DIR / "loss_curve_summary.txt"

    summary = f"""Loss 曲线补充实验说明

一、实验目的
本实验用于补充期末作业中“loss 曲线对比（非训练型不做）”的要求。

二、参与 loss 曲线分析的模型
1. ANN：
   sklearn MLPClassifier 提供 loss_curve_，可记录迭代训练损失。
2. XGBoost：
   通过 eval_set 和 eval_metric='mlogloss' 记录训练集与验证集的多分类 logloss。

三、未参与 loss 曲线分析的模型
1. KNN：
   KNN 为惰性学习方法，不存在逐轮训练 loss。
2. SVM：
   当前 sklearn SVC 实现不提供可直接绘制的 epoch loss 曲线。
3. Random Forest：
   随机森林为树集成模型，不提供连续迭代 loss 曲线。

四、实验结果概况
ANN 迭代轮数：{len(ann_loss)}
ANN 初始 loss：{ann_loss['ann_loss'].iloc[0]:.6f}
ANN 最终 loss：{ann_loss['ann_loss'].iloc[-1]:.6f}

XGBoost 迭代轮数：{len(xgb_loss)}
XGBoost 初始验证 mlogloss：{xgb_loss['xgboost_val_mlogloss'].iloc[0]:.6f}
XGBoost 最终验证 mlogloss：{xgb_loss['xgboost_val_mlogloss'].iloc[-1]:.6f}

五、输出文件
08_results/figures/ann_loss_curve.png
08_results/figures/xgboost_logloss_curve.png
08_results/figures/loss_curve_comparison.png
08_results/loss_curve_ann.csv
08_results/loss_curve_xgboost.csv
"""

    summary_path.write_text(summary, encoding="utf-8")
    print(f"Loss 曲线实验说明已保存：{summary_path}")


# ==============================
# 8. 主函数
# ==============================

def main():
    print("=" * 60)
    print("Loss 曲线对比实验")
    print("=" * 60)

    X_train, y_train, X_val, y_val = load_processed_data()

    ann_loss = train_ann_and_record_loss(X_train, y_train)
    xgb_loss = train_xgboost_and_record_loss(X_train, y_train, X_val, y_val)

    plot_ann_loss(ann_loss)
    plot_xgboost_loss(xgb_loss)
    plot_loss_comparison(ann_loss, xgb_loss)

    save_summary(ann_loss, xgb_loss)

    print("\n第 Loss 曲线补充实验完成。")
    print("请检查以下输出文件：")
    print("1. 08_results/figures/ann_loss_curve.png")
    print("2. 08_results/figures/xgboost_logloss_curve.png")
    print("3. 08_results/figures/loss_curve_comparison.png")
    print("4. 08_results/loss_curve_ann.csv")
    print("5. 08_results/loss_curve_xgboost.csv")
    print("6. 08_results/loss_curve_summary.txt")


if __name__ == "__main__":
    main()