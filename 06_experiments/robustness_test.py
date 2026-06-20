# ============================================================
# DryBean-ML-Project-basic
# 第 12 步：鲁棒性实验（加噪声测试）
#
# 功能：
# 1. 对测试集加入不同强度噪声
# 2. 测试 5 个模型性能下降情况
# 3. 输出 robustness_table.csv
# ============================================================

from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import accuracy_score


# ==============================
# 1. 路径
# ==============================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "01_data" / "processed"
MODEL_PATH = PROJECT_ROOT / "08_results" / "models"
RESULT_PATH = PROJECT_ROOT / "08_results"

TEST_FILE = DATA_PATH / "test_processed.csv"
ROBUST_FILE = RESULT_PATH / "robustness_table.csv"


# ==============================
# 2. 加载数据
# ==============================

def load_data():
    data = pd.read_csv(TEST_FILE)
    X = data.drop(columns=["Class_Label"]).values
    y = data["Class_Label"].values
    return X, y


# ==============================
# 3. 加噪声函数
# ==============================

def add_noise(X, noise_level):
    noise = np.random.normal(0, noise_level, X.shape)
    return X + noise


# ==============================
# 4. 主实验
# ==============================

def main():

    X, y = load_data()

    models = ["knn", "svm", "random_forest", "ann", "xgboost"]
    noise_levels = [0.0, 0.01, 0.05, 0.1]

    results = []

    print("=" * 70)
    print("第 12 步：鲁棒性实验")
    print("=" * 70)

    for model_name in models:

        model = joblib.load(MODEL_PATH / f"{model_name}.joblib")

        base_acc = None

        for noise in noise_levels:

            X_noisy = add_noise(X, noise)

            y_pred = model.predict(X_noisy)
            acc = accuracy_score(y, y_pred)

            if noise == 0.0:
                base_acc = acc

            results.append({
                "model": model_name,
                "noise_level": noise,
                "accuracy": acc,
                "drop": base_acc - acc
            })

            print(f"{model_name} | noise={noise} | acc={acc:.4f}")

    df = pd.DataFrame(results)

    RESULT_PATH.mkdir(parents=True, exist_ok=True)
    df.to_csv(ROBUST_FILE, index=False, encoding="utf-8-sig")

    print("-" * 70)
    print("鲁棒性结果已保存：")
    print(ROBUST_FILE)

    print("=" * 70)
    print("第 12 步完成")
    print("=" * 70)


# ==============================
# 5. 入口
# ==============================

if __name__ == "__main__":
    main()