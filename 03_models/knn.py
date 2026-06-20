# ============================================================
# DryBean-ML-Project-basic
# 第 5 步：模型模块 - KNN 分类模型
#
# 对应期末作业要求：
# 1. 实现多分类算法，用于 Dry Bean Dataset 分类任务
# 2. KNN 属于课内算法，可作为基础对照模型 baseline
#
# 本文件功能：
# 1. 定义 KNN 模型创建函数
# 2. 不在本文件中训练模型
# 3. 后续由 04_train/train_model.py 统一调用
#
# 注意：
# 本文件只负责“创建模型”，不读取数据、不保存结果。
# ============================================================

from sklearn.neighbors import KNeighborsClassifier


def create_model() -> KNeighborsClassifier:
    """
    创建 KNN 多分类模型。

    参数设置说明：
        n_neighbors=5:
            使用 5 个近邻进行投票分类。
        weights="distance":
            距离越近的样本权重越大。
        metric="minkowski":
            默认距离度量，p=2 时等价于欧氏距离。

    返回：
        model: KNN 分类模型
    """
    model = KNeighborsClassifier(
        n_neighbors=5,
        weights="distance",
        metric="minkowski",
        p=2,
    )

    return model


def get_model_name() -> str:
    """
    返回模型名称。

    返回：
        KNN
    """
    return "KNN"


if __name__ == "__main__":
    model = create_model()

    print("=" * 70)
    print("第 5 步检查：KNN 模型创建成功")
    print("=" * 70)
    print(model)