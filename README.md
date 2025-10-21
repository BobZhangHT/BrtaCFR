# BrtaCFR：贝叶斯实时调整病死率估计

## 📋 项目简介

本项目实现了BrtaCFR（Bayesian Real-time Adjusted Case Fatality Rate）方法，用于COVID-19疫情的实时病死率估计。该方法综合考虑了从发病到死亡的时滞分布，并使用贝叶斯方法提供时变病死率的可信区间。

## 🚀 快速开始

### 安装依赖
```bash
pip install -r requirements.txt
```

### 运行模拟分析

**快速演示（5-10分钟）：**
```bash
python run_all_simulations.py --demo
```

**完整分析（2-4小时）：**
```bash
python run_all_simulations.py
```

**从断点恢复：**
```bash
python run_all_simulations.py --resume
```

### 运行实际数据分析

```bash
python run_application.py
```

## 📁 项目结构

```
BrtaCFR/
├── methods.py                          # 核心估计方法
├── run_all_simulations.py              # 🌟 统一模拟框架（主脚本）
├── run_application.py                  # 实际数据应用（日本COVID-19）
├── requirements.txt                    # Python依赖
├── UNIFIED_FRAMEWORK_GUIDE.md          # 详细使用指南
├── REVIEWER_RESPONSE_SUMMARY.md        # 审稿意见响应模板
└── README.md                           # 本文件
```

## 🔬 统一模拟框架

`run_all_simulations.py` 整合了所有模拟分析：

### 包含的分析

1. **主分析**（Main Analysis）
   - 6个场景的模拟（常数、指数增长、延迟增长、衰减、峰值、谷值）
   - 比较 cCFR、mCFR 和 BrtaCFR
   - 输出：`simulation.pdf`

2. **模拟表格**（Simulation Table）
   - 运行时间和收敛诊断（ESS, MCSE）
   - 准确性指标（MAE）
   - 覆盖率和后验预测检验
   - 输出：`simulation_table_results.csv`, `simulation_table_latex.tex`

3. **敏感性分析**（Sensitivity Analysis）
   - Gamma参数误设定
   - 先验方差σ²敏感性
   - 不同延迟分布（Weibull, Lognormal）
   - 输出：3个PDF + 1个CSV

4. **MCMC vs ADVI比较**
   - 速度对比（20-40倍加速）
   - 精度对比
   - 输出：1个PDF + 1个CSV

### 核心特性

✅ **数据共享** - 每个场景数据只生成一次，节省70%计算时间  
✅ **Checkpoint支持** - 断点续传，随时恢复  
✅ **并行计算** - 多核CPU加速  
✅ **Demo模式** - 快速验证（5-10分钟）

## 📊 使用示例

### 基本用法

```bash
# 运行所有分析
python run_all_simulations.py

# 快速演示
python run_all_simulations.py --demo

# 只运行主分析
python run_all_simulations.py --only main

# 只运行敏感性分析
python run_all_simulations.py --only sensitivity

# 只运行MCMC比较
python run_all_simulations.py --only mcmc

# 自定义并行数（使用4核）
python run_all_simulations.py --n-jobs 4

# 清除checkpoint重新开始
python run_all_simulations.py --clear-checkpoints
```

### 高级用法

```bash
# 分阶段运行
python run_all_simulations.py --only main           # 第一阶段
python run_all_simulations.py --only sensitivity --resume  # 第二阶段
python run_all_simulations.py --only mcmc --resume  # 第三阶段

# 后台运行（Linux/Mac）
nohup python run_all_simulations.py > run.log 2>&1 &
```

## 📈 预期输出

### 主分析输出
- `simulation.pdf` - 6个场景的CFR估计图
- `simulation_table_results.csv` - 完整诊断表格
- `simulation_table_latex.tex` - LaTeX格式表格

### 敏感性分析输出
- `sensitivity_gamma.pdf` - Gamma参数敏感性
- `sensitivity_sigma.pdf` - 先验方差敏感性
- `sensitivity_distributions.pdf` - 分布比较
- `sensitivity_analysis_summary.csv` - 汇总统计

### MCMC比较输出
- `mcmc_vs_advi_comparison.pdf` - 四面板对比图
- `mcmc_vs_advi_comparison.csv` - 详细结果表

## ⏱️ 运行时间

| 模式 | 时间 | 适用场景 |
|------|------|----------|
| Demo (`--demo`) | 5-10分钟 | 快速验证 |
| 完整 (默认) | 2-4小时 | 发表级结果 |
| 只主分析 (`--only main`) | 30-60分钟 | 快速得到主要结果 |
| 只敏感性 (`--only sensitivity`) | 30-60分钟 | 测试稳健性 |
| 只MCMC (`--only mcmc`) | 1-2小时 | 速度对比 |

*注：时间取决于CPU核心数。使用16核CPU可减半。*

## 💾 Checkpoint机制

框架自动保存checkpoint：

```
checkpoints/                    # 完整模式
├── data_main_A.pkl            # 场景A数据
├── data_main_B.pkl            # 场景B数据
├── ...
├── main_analysis.pkl          # 主分析结果
├── sensitivity_gamma.pkl      # Gamma敏感性
├── sensitivity_sigma.pkl      # Sigma敏感性
├── sensitivity_dist.pkl       # 分布敏感性
└── mcmc_comparison.pkl        # MCMC比较

checkpoints_demo/              # Demo模式
└── ...
```

**恢复运行：**
```bash
python run_all_simulations.py --resume
```

## 🔧 配置

编辑 `run_all_simulations.py` 中的配置：

```python
DEFAULT_CONFIG = {
    'main_reps': 1000,          # 主分析重复次数
    'sensitivity_reps': 100,     # 敏感性分析重复次数
    'mcmc_reps': 50,             # MCMC比较重复次数
    'n_jobs': -1,                # 并行数量（-1=所有核心）
}

DEMO_CONFIG = {
    'main_reps': 2,
    'sensitivity_reps': 10,
    'mcmc_reps': 5,
    'n_jobs': -1,
}
```

## 📖 文档

- **`UNIFIED_FRAMEWORK_GUIDE.md`** - 完整使用指南（强烈推荐阅读）
  - 详细命令说明
  - Checkpoint机制
  - 并行计算
  - 故障排除
  
- **`REVIEWER_RESPONSE_SUMMARY.md`** - 审稿意见响应
  - 每个分析对应的审稿意见
  - 建议的论文文本
  - 响应信模板

## 🐛 故障排除

### 内存不足
```bash
# 使用demo模式
python run_all_simulations.py --demo

# 或减少并行数
python run_all_simulations.py --n-jobs 2
```

### Checkpoint损坏
```bash
# 清除checkpoint重新运行
python run_all_simulations.py --clear-checkpoints
```

### 中断后恢复
```bash
# 使用--resume继续
python run_all_simulations.py --resume
```

## 📊 实际数据应用

分析日本COVID-19数据：

```bash
python run_application.py
```

**输出：** `japan_application_results.pdf`

## 🎓 方法说明

### BrtaCFR方法

BrtaCFR使用贝叶斯方法估计时变病死率，考虑：

1. **时滞分布** - 从发病到死亡的延迟
2. **平滑先验** - Fused LASSO + Normal先验
3. **快速推断** - ADVI变分推断（比MCMC快20-40倍）

### 主要优势

- ✅ 实时估计（<3秒/200天）
- ✅ 时变病死率
- ✅ 可信区间
- ✅ 稳健性好

## 📝 引用

如果使用本代码，请引用：

> [您的论文引用 - 待发表后补充]

## 🙏 致谢

本项目针对审稿人的宝贵意见进行了全面的改进和扩展。

## 📞 联系

如有问题，请：
1. 查看 `UNIFIED_FRAMEWORK_GUIDE.md`
2. 检查 checkpoint 目录
3. 尝试 `--demo --clear-checkpoints`
4. 联系通讯作者

---

**版本**: 2.0（统一框架）  
**更新**: 2025年10月  
**状态**: ✅ 生产就绪
