# ============================================================
# DryBean-ML-Project-basic
# 第 5 步：模型模块 - Random Forest 分类模型
#
# 对应期末作业要求：
# 1. 实现多分类算法，用于 Dry Bean Dataset 分类任务
# 2. Random Forest 属于课外算法，可满足“至少一种课堂外算法”的要求
#
# 本文件功能：
# 1. 定义 Random Forest 模型创建函数
# 2. 不在本文件中训练模型
# 3. 后续由 04_train/train_model.py 统一调用
#
# 注意：
# 本文件只负责“创建模型”，不读取数据、不保存结果。
# ============================================================

from sklearn.ensemble import RandomForestClassifier


def create_model() -> RandomForestClassifier:
    """
    创建 Random Forest 多分类模型。

    参数设置说明：
        n_estimators=200:
            使用 200 棵决策树，保证模型稳定性。
        max_depth=None:
            不限制树深度，让模型充分学习特征关系。
        random_state=42:
            固定随机种子，保证实验可复现。
        n_jobs=-1:
            使用全部 CPU 核心，提高训练速度。

    返回：
        model: Random Forest 分类模型
    """
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )

    return model


def get_model_name() -> str:
    """
    返回模型名称。

    返回：
        Random Forest
    """
    return "Random Forest"


if __name__ == "__main__":
    model = create_model()

    print("=" * 70)
    print("第 5 步检查：Random Forest 模型创建成功")
    print("=" * 70)
    print(model)