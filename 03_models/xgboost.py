# ============================================================
# XGBoost 模型（强加分模型）
# ============================================================

from xgboost import XGBClassifier


def create_model():

    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        eval_metric="mlogloss",
        random_state=42
    )

    return model


def get_model_name():
    return "XGBoost"


if __name__ == "__main__":
    print("XGBoost模型加载成功")