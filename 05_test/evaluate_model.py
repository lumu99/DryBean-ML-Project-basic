# ============================================================
# DryBean-ML-Project-basic
# 第 7 步：统一测试模块（增强版：推理速度 + 实验闭环）
#
# 本次升级内容：
# ✔ 增加推理时间统计（predict latency）
# ✔ 生成 speed_table.csv
# ✔ 支持论文 30%实验分析要求
# ============================================================

from pathlib import Path
import joblib
import pandas as pd
import time

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ============================================================
# 1. 路径
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "01_data" / "processed"
MODEL_PATH = PROJECT_ROOT / "08_results" / "models"
RESULT_PATH = PROJECT_ROOT / "08_results"

TEST_FILE = DATA_PATH / "test_processed.csv"

RESULT_TABLE = RESULT_PATH / "test_results.csv"
SPEED_TABLE = RESULT_PATH / "speed_table.csv"


# ============================================================
# 2. 数据加载
# ============================================================

def load_test_data():
    data = pd.read_csv(TEST_FILE)
    X = data.drop(columns=["Class_Label"])
    y = data["Class_Label"]
    return X, y


# ============================================================
# 3. 模型评估（增强版）
# ============================================================

def evaluate_model(model_name, model, X_test, y_test):

    # ==============================
    # 推理时间统计（核心新增）
    # ==============================
    start_time = time.perf_counter()
    y_pred = model.predict(X_test)
    end_time = time.perf_counter()

    predict_time = end_time - start_time
    avg_latency = predict_time / len(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    return {
        "model": model_name,
        "accuracy": acc,
        "precision_macro": report["macro avg"]["precision"],
        "recall_macro": report["macro avg"]["recall"],
        "f1_macro": report["macro avg"]["f1-score"],

        # ✔ 推理时间（总 + 单样本）
        "predict_time_seconds": predict_time,
        "avg_latency_ms": avg_latency * 1000,

        "confusion_matrix": cm.tolist()
    }


# ============================================================
# 4. 主流程
# ============================================================

def main():

    X_test, y_test = load_test_data()

    models = ["knn", "svm", "random_forest", "ann", "xgboost"]

    results = []

    speed_results = []

    print("=" * 70)
    print("第 7 步：模型测试评估（增强版）")
    print("=" * 70)

    for name in models:

        model_file = MODEL_PATH / f"{name}.joblib"
        model = joblib.load(model_file)

        result = evaluate_model(name, model, X_test, y_test)
        results.append(result)

        speed_results.append({
            "model": name,
            "predict_time_seconds": result["predict_time_seconds"],
            "avg_latency_ms": result["avg_latency_ms"]
        })

        print(f"{name} 测试集准确率：{result['accuracy']:.4f}")
        print(f"{name} 推理时间：{result['predict_time_seconds']:.4f}s")

    RESULT_PATH.mkdir(parents=True, exist_ok=True)

    # ==============================
    # 保存结果表
    # ==============================
    pd.DataFrame(results).to_csv(RESULT_TABLE, index=False, encoding="utf-8-sig")
    pd.DataFrame(speed_results).to_csv(SPEED_TABLE, index=False, encoding="utf-8-sig")

    print("-" * 70)
    print("测试结果已保存：")
    print(RESULT_TABLE)
    print(SPEED_TABLE)

    print("=" * 70)
    print("第 7 步完成（增强版）")
    print("=" * 70)


if __name__ == "__main__":
    main()