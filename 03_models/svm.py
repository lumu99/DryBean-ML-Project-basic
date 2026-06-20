# ============================================================
# DryBean-ML-Project-basic
# 第 5 步：模型模块 - SVM 分类模型
#
# 对应期末作业要求：
# 1. 实现多分类算法，用于 Dry Bean Dataset 分类任务
# 2. SVM 属于课内算法，是传统机器学习中的强分类模型
#
# 本文件功能：
# 1. 定义 SVM 模型创建函数
# 2. 不在本文件中训练模型
# 3. 后续由 04_train/train_model.py 统一调用
#
# 注意：
# 本文件只负责“创建模型”，不读取数据、不保存结果。
# ============================================================

from sklearn.svm import SVC


def create_model() -> SVC:
    """
    创建 SVM 多分类模型。

    参数设置说明：
        kernel="rbf":
            使用径向基核函数，适合非线性分类。
        C=10:
            惩罚参数，适当提高模型拟合能力。
        gamma="scale":
            使用 sklearn 默认自适应 gamma 设置。
        probability=False:
            基础分类任务只需要预测类别，不需要概率输出。

    返回：
        model: SVM 分类模型
    """
    model = SVC(
        kernel="rbf",
        C=10,
        gamma="scale",
        probability=False,
        random_state=42,
    )

    return model


def get_model_name() -> str:
    """
    返回模型名称。

    返回：
        SVM
    """
    return "SVM"


if __name__ == "__main__":
    model = create_model()

    print("=" * 70)
    print("第 5 步检查：SVM 模型创建成功")
    print("=" * 70)
    print(model)