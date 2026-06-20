# ============================================================
# ANN（MLP神经网络）模型
# ============================================================

from sklearn.neural_network import MLPClassifier


def create_model():

    model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        max_iter=300,
        random_state=42
    )

    return model


def get_model_name():
    return "ANN"


if __name__ == "__main__":
    print("ANN模型加载成功")