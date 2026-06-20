```markdown id="final_readme"
<div align="center">

# 🌱 DryBean Machine Learning System

## 📊 A Full-Stack Machine Learning Engineering Pipeline

### ⚙️ Data → Preprocess → Train → Evaluate → Visualization

---

<img src="https://img.shields.io/badge/ML-Engineering-blue?style=for-the-badge"/>
<img src="https://img.shields.io/badge/MultiModel-KNN%20SVM%20ANN%20RF%20XGBoost-green?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Robustness-Test-orange?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Status-Completed-brightgreen?style=for-the-badge"/>

</div>

---

# 📌 1. 项目简介

## 1.1 项目背景

本项目基于 Dry Bean Dataset 构建完整机器学习工程系统，用于多类别农业数据分类问题。

---

## 1.2 项目目标

- 构建完整机器学习工程流程
- 实现多模型对比分析
- 完成鲁棒性与泛化能力研究
- 形成可复现实验系统

---

# 🏗️ 2. 系统架构设计

## 2.1 总体流程

```

数据读取 → 数据清洗 → 特征工程 → 模型训练 → 模型评估 → 鲁棒性分析 → 可视化输出

```

---

## 2.2 系统特点

- 模块化设计
- 全流程自动化
- 实验结果持久化
- 支持一键运行

---

# 📁 3. 项目结构说明

## 📌 3.1 树状结构

```

DryBean-ML-Project-basic/
│
├── 01_data/
│   ├── loader.py
│   ├── raw/
│   └── processed/
│
├── 02_preprocess/
│   ├── preprocess.py
│   └── feature_engineering.py
│
├── 03_models/
│   ├── knn.py
│   ├── svm.py
│   ├── random_forest.py
│   ├── ann.py
│   └── xgboost.py
│
├── 04_train/
│   └── train_model.py
│
├── 05_test/
│   └── evaluate_model.py
│
├── 06_experiments/
│   ├── robustness_test.py
│   └── visualize_results.py
│
├── 07_utils/
│   ├── metrics.py
│   ├── plot_utils.py
│   └── timer.py
│
├── 08_results/
│   ├── figures/
│   ├── train_summary.csv
│   ├── test_results.csv
│   └── robustness_table.csv
│
├── 09_pipeline/
│   ├── main.py
│   └── config.yaml
│
└── 00_docs/

````

---

# 📊 4. 重要文件说明（评分重点）

| 模块 | 文件 | 功能说明 | 输出结果 |
|------|------|----------|----------|
| 数据加载 | loader.py | 读取原始数据 | train/val/test |
| 数据清洗 | preprocess.py | 缺失值处理 | processed数据 |
| 特征工程 | feature_engineering.py | 相关性分析 | 特征统计 |
| 模型训练 | train_model.py | 训练5种模型 | train_summary.csv |
| 模型测试 | evaluate_model.py | 精度+速度评估 | test_results.csv |
| 鲁棒性分析 | robustness_test.py | 噪声实验 | robustness_table.csv |
| 可视化 | visualize_results.py | 生成图表 | figures/*.png |
| Pipeline | main.py | 一键运行系统 | 全流程输出 |

---

# 🧹 5. 数据处理流程

## 5.1 数据清洗

- 缺失值处理
- 重复值删除
- 标签标准化

---

## 5.2 特征工程

- 相关性分析
- 高相关特征筛选（阈值0.95）
- 特征统计分析

---

# 🤖 6. 模型设计

## ✔ 课内模型

- KNN
- SVM
- ANN（MLP）

## ✔ 课外模型

- Random Forest
- XGBoost

---

# 📊 7. 实验结果分析

## 7.1 模型精度对比

| 模型 | 测试集准确率 |
|------|--------------|
| KNN | 0.9211 |
| SVM | 0.9335 |
| Random Forest | 0.9233 |
| ANN | 0.9262 |
| XGBoost | 0.9262 |

---

## 7.2 推理速度对比

| 模型 | 推理时间(s) |
|------|-------------|
| KNN | 1.4984 |
| SVM | 0.2608 |
| Random Forest | 0.0405 |
| ANN | 0.0029 |
| XGBoost | 0.0075 |

---

## 7.3 鲁棒性分析

- 噪声强度：0.0 / 0.01 / 0.05 / 0.1
- XGBoost稳定性最强
- KNN对噪声最敏感

---

## 7.4 过拟合分析

- Random Forest：过拟合明显
- SVM：稳定性较好
- XGBoost：泛化能力最佳

---

## 7.5 ANN Loss曲线

- Loss持续下降
- 收敛稳定
- 无明显震荡

---

# 🚀 8. 运行方式

```bash
pip install -r requirements.txt
python 09_pipeline/main.py
````

---

# 📁 9. 输出结果

```
08_results/
├── train_summary.csv
├── test_results.csv
├── speed_table.csv
├── robustness_table.csv
├── overfit_table.csv
└── figures/
```

---

# 💡 10. 创新点（加分项）

* ✔ 完整工程化机器学习系统
* ✔ 多维度评价体系（精度+速度+鲁棒性+过拟合）
* ✔ 噪声鲁棒性实验（课程外扩展）
* ✔ ANN Loss曲线分析
* ✔ 自动化可视化系统
* ✔ 一键运行 Pipeline
* ✔ 实验结果自动保存用于论文

---

# 📌 11. 总结

本项目构建了一个完整机器学习工程系统，实现了从数据处理到模型训练、评估、鲁棒性分析与可视化的全流程闭环。

---

# 🙏 12. 致谢

README参考与学习来源：

[https://zhuanlan.zhihu.com/p/151291463](https://zhuanlan.zhihu.com/p/151291463)

---

# 👨‍💻 Author

Machine Learning Course Project
Dry Bean Classification System

```
