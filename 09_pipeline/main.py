# ============================================================
# DryBean-ML-Project-basic
# 第 14 步：一键运行主入口（最终交付版本）
#
# 功能：
# 1. 统一运行训练
# 2. 统一运行测试
# 3. 运行鲁棒性实验
# 4. 生成所有图表
# 5. 输出论文级结果
#
# 这是整个项目的“唯一入口”
# ============================================================

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ==============================
# 1. 执行命令工具
# ==============================

def run_script(script_path: Path):
    print("\n" + "=" * 70)
    print(f"执行：{script_path.name}")
    print("=" * 70)

    subprocess.run([sys.executable, str(script_path)])


# ==============================
# 2. 主流程
# ==============================

def main():

    print("\n" + "=" * 70)
    print("DryBean-ML-Project-basic 一键系统启动")
    print("=" * 70)

    # Step 1：训练
    run_script(PROJECT_ROOT / "04_train" / "train_model.py")

    # Step 2：测试
    run_script(PROJECT_ROOT / "05_test" / "evaluate_model.py")

    # Step 3：鲁棒性
    run_script(PROJECT_ROOT / "06_experiments" / "robustness_test.py")

    # Step 4：可视化
    run_script(PROJECT_ROOT / "06_experiments" / "visualize_results.py")

    print("\n" + "=" * 70)
    print("全部流程完成 ✔")
    print("结果已生成在 08_results/")
    print("=" * 70)


# ==============================
# 3. 入口
# ==============================

if __name__ == "__main__":
    main()