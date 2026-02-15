# 关于 Fig5 (课程学习曲线) 的说明

## 当前状况

**Fig5 (课程学习曲线) 无法从真实数据生成**，原因如下：

1. **训练代码只打印了每个 epoch 的 loss，但没有保存到文件**
   - 代码位置: `run_advanced_improvements.py` 第 329 行
   - `print(f"    Epoch {epoch:3d}: Loss={loss.item():.4f}, Val MAE={mae:.2f}m")`
   
2. **没有 TensorBoard 日志或 CSV 文件记录训练历史**
   - 搜索了所有 `experiments/` 目录
   - 没有找到 `events.out.tfevents.*` 或 `*history*.csv` 文件

3. **只有最终结果**
   - `advanced_improvements_results.json` 只包含最终 MAE 值
   - 不包含训练过程中的 loss 曲线

## 解决方案选项

### 选项 1: 跳过 Fig5
- 不在论文中包含课程学习曲线图
- 用文字描述课程学习策略
- 最简单，确保数据完整性

### 选项 2: 重新运行训练并保存历史
- 修改 `run_advanced_improvements.py` 保存训练历史到 CSV/JSON
- 重新运行完整的 450 epoch 训练 (约 2-3 小时)
- 然后从保存的数据生成真实曲线

### 选项 3: 使用示意图 (明确标注)
- 创建概念图展示课程学习策略
- 明确标注为 "Illustrative Concept" 或 "Schematic"
- 使用箭头/阶段框表示 Easy→Medium→Hard 流程

## 建议

**推荐选项 1 或 3**，因为:
- 重新运行训练耗时较长
- 课程学习策略的效果已经通过最终 MAE 值体现 (3.79m)
- 可以用文字清晰描述三个阶段

如需重新运行训练保存历史，请运行:
```bash
source ~/miniconda3/bin/activate graphmamba
python run_advanced_improvements.py --save-history
```

(需要修改代码添加 `--save-history` 功能)
