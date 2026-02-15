# 代码组织结构

## 整理说明

本次整理将实验过程中产生的大量测试脚本和中间文件归档到 `tmp/` 目录，保留核心代码和最新结果。

---

## 核心代码文件（根目录）

### 最终版脚本
| 文件 | 说明 |
|------|------|
| `run_quick_improvements.py` | **最终版**: SIREN + Ensemble，最佳结果 8.66m |
| `run_final_pipeline.py` | 完整流程：数据清洗 → ERA5 → 训练 → 验证 |

### 步骤脚本
| 文件 | 说明 |
|------|------|
| `step1_data_cleaning.py` | 数据清洗：移除移动传感器、异常值 |
| `step2_download_era5.py` | ERA5 数据下载 |
| `step3_neural_field_advanced.py` | Neural Field 训练（含多种PE方法） |

### 文档
| 文件 | 说明 |
|------|------|
| `IEEE_TIM_RESULTS_SUMMARY.md` | IEEE TIM 实验结果总结（主要文档） |
| `README.md` | 项目主文档 |
| `AGENTS.md` | 开发规范 |

---

## 归档目录（tmp/）

### tmp/scripts_archive/
包含实验过程中产生的各种测试脚本：
- `run_advanced_pinn.py`
- `run_comprehensive_validation.py`
- `run_loso_validation.py`
- `run_*.py` (共 20+ 个测试脚本)

### tmp/figures_archive/
生成的分析图表：
- `rf_analysis.png`
- `sensor_mobility_analysis.png`
- `spatial_bias_analysis.png`

### tmp/docs_archive/
旧的文档：
- `ERA5_INTEGRATION_REPORT.md`
- `EXPERIMENTS_REPORT.md`
- `IMPROVEMENT_ROADMAP.md`

### tmp/data_archive/
大型数据文件：
- `srtm_59_08.*` (SRTM DEM 数据)

### tmp/experiments_archive/
旧的实验结果和代码

---

## 快速开始

### 1. 数据清洗
```bash
python step1_data_cleaning.py
```

### 2. 运行最终版验证
```bash
python run_quick_improvements.py
```

### 3. 完整流程
```bash
python run_final_pipeline.py
```

---

## 关键结果

| 方法 | 最佳 MAE | 状态 |
|------|---------|------|
| NF + ERA5 | 11.19m | 基线 |
| **SIREN + Ensemble** | **8.66m** | ✅ **目标达成** |

目标: <10m ✅

---

## 数据文件

### 清洗后数据
- `data/processed/sensor_data_cleaned.csv` (115,417 样本, 7 传感器)
- `data/processed/sensor_data_with_real_era5.csv` (含 ERA5 特征)

### ERA5 数据
- `data/era5_shenzhen_complete.nc` (真实 ERA5 再分析数据)

---

## Git 忽略规则

见 `.gitignore`：
- 忽略 tmp/ 归档目录
- 忽略大型数据文件 (.nc, .tif, .zip)
- 忽略生成图表 (.png)
- 忽略模型检查点 (.pth, .ckpt)

---

## 恢复归档文件

如需恢复某个归档的脚本：
```bash
cp tmp/scripts_archive/run_xxx.py .
```

---

**整理时间**: 2026-02-13  
**整理前改动**: 58 个文件  
**整理后改动**: 12 个核心文件 + 归档
