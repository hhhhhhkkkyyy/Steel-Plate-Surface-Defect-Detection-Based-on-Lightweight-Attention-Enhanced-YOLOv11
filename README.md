# 基于轻量化注意力增强 YOLO11 的钢板表面缺陷检测系统

一个面向**钢材 / 钢板表面缺陷检测**的目标检测系统。项目在 **YOLO11** 检测框架的基础上，向骨干网络注入**轻量化 CBAM（L-CBAM）注意力模块**，并通过数据增强、余弦退火学习率、损失权重调优等手段，在 **NEU-DET** 钢材表面缺陷数据集上训练缺陷检测模型。系统同时提供了一个工业风格的 **Tkinter 桌面 GUI**，支持图片 / 摄像头实时检测、缺陷类别着色、置信度阈值调节，并内置**缺陷知识库**（缺陷名称、成因描述、解决方案与参考图片）。

最终模型在 4 类缺陷上达到 **mAP@0.5 ≈ 0.89**、**mAP@0.5:0.95 ≈ 0.56**，兼顾检测精度与工业落地所需的轻量化部署能力。

---

## 目录

- [项目简介](#项目简介)
- [功能特性](#功能特性)
- [数据集](#数据集)
- [模型结构](#模型结构)
- [环境依赖](#环境依赖)
- [目录结构](#目录结构)
- [快速开始](#快速开始)
- [数据预处理](#数据预处理)
- [模型训练](#模型训练)
- [测试与评估](#测试与评估)
- [GUI 使用说明](#gui-使用说明)
- [实验结果](#实验结果)
- [常见问题](#常见问题)
- [参考与致谢](#参考与致谢)

---

## 项目简介

钢板表面缺陷种类多、尺度差异大、部分缺陷纹理微弱，传统人工检测效率低且易漏检。本系统基于 YOLO11 目标检测模型，针对工业缺陷检测的特点进行了以下改进：

1. **轻量化注意力增强**：在骨干网络的大通道模块（C2f/C3k2）后嵌入轻量化 CBAM（通道注意力 + 空间注意力），强化关键缺陷特征表达，同时保持较低的额外计算开销。
2. **工业缺陷导向的数据增强**：使用 Mosaic、MixUp、Copy-Paste、HSV 扰动、翻转等策略，提升模型对小样本、多形态缺陷的鲁棒性。
3. **训练策略优化**：采用 SGD + 余弦退火 + 分层预热、加强边界框回归损失权重、TTA（测试时增强）验证等手段提升精度。
4. **可视化交互界面**：提供工业级 GUI，并内置缺陷知识库，检测后可查看缺陷成因与处理建议。

---

## 功能特性

- **轻量化 CBAM 注意力**：通道 + 空间双注意力，reduction=16 控制参数量。
- **YOLO11 多规格支持**：支持 `yolo11n / s / m`，项目默认使用 `yolo11m`。
- **数据预处理**：NEU-DET 原始数据清洗、类别筛选与 YOLO 格式转换。
- **完整训练流程**：训练、验证、TTA 评估、ONNX 导出。
- **工业级 GUI**：图片 / 摄像头检测、置信度阈值调节、缺陷着色框、检测结果保存。
- **缺陷知识库**：内置 6 类缺陷的中文名称、成因描述、解决方案与参考图片。
- **可扩展评估**：输出 mAP、Precision、Recall、混淆矩阵等指标。

---

## 数据集

项目使用公开的 **NEU-DET（NEU surface defect database）** 钢材表面缺陷数据集，原始共 6 类缺陷：

| 英文类别 | 中文含义 |
| --- | --- |
| crazing | 裂纹 |
| inclusion | 夹杂 |
| patches | 补丁 |
| pitted_surface | 点蚀表面 |
| rolled-in_scale | 轧入氧化皮 |
| scratches | 划痕 |

经过 `dataset.py` 过滤后，删除了 `crazing` 与 `rolled-in_scale` 两类，最终训练数据为 **4 类**：`inclusion`、`patches`、`pitted_surface`、`scratches`。

> 数据集需自行从 NEU-DET 官方渠道获取；仓库中仅保留处理后的目录结构与少量示例图，未包含完整原始数据。

---

## 模型结构

```text
输入图像 (320 × 320)
        │
        ▼
YOLO11 骨干网络（C2f / C3k2 等模块）
        │  在大通道模块后注入轻量化 CBAM
        ▼
轻量化 CBAM（通道注意力 + 空间注意力）
        │
        ▼
Neck + Head
        │
        ▼
输出：4 类缺陷的边界框 + 置信度
```

**轻量化 CBAM 结构：**

- **通道注意力分支**：全局平均池化 + 全局最大池化 → 共享 MLP（`reduction=16`）→ Sigmoid；
- **空间注意力分支**：沿通道维度取平均 / 最大 → 单层 3×3 卷积 → Sigmoid；
- 两分支依次对特征图进行重标定，增强对缺陷区域的关注。

注入位置为骨干网络中 `c > 64` 的大通道模块之后，保证在提升精度的同时只引入极少参数量。

---

## 环境依赖

推荐 Python 3.8+，使用 PyTorch 深度学习框架（无 GPU 时自动回退到 CPU）。

| 依赖 | 说明 |
| --- | --- |
| Python | 3.8+（开发环境为 3.10） |
| PyTorch | 2.x |
| ultralytics | 8.x（提供 YOLO11） |
| opencv-python | 图像读取与预处理 |
| numpy / pandas | 数值计算与数据处理 |
| matplotlib | 训练曲线与结果可视化 |
| PyYAML | 数据集配置读写 |
| psutil | 训练资源监控 |
| tqdm | 进度显示 |

安装示例：

```bash
pip install torch torchvision
pip install ultralytics opencv-python numpy pandas matplotlib pyyaml psutil tqdm
```

---

## 目录结构

```text
123/
├── gggggggui.py              # 工业版缺陷检测 GUI（主入口）
├── dataset.py                # NEU-DET 数据过滤（6类 → 4类）与格式转换
├── opop.py                   # 增强版训练（多尺度 + 类别权重 + CBAM + 混淆矩阵）
├── train.py                  # 优化训练（L-CBAM + 超参优化 + TTA + ONNX 导出）
├── trainoptimi.py            # 优化训练（同 train.py，独立输出目录）
├── train_base.py             # 基线训练（YOLO11m 无改进）
├── train_bb2.py              # L-CBAM + 余弦退火 + 分层预热训练
├── test.py                   # 检测 + 批量验证（输出 mAP 等指标）
├── 36.py                     # 带 TTA 的独立验证脚本
├── yolo11n.pt / yolo11s.pt / yolo11m.pt   # YOLO11 预训练权重
├── IMAGES/ ANNOTATIONS/      # NEU-DET 原始数据
├── processed_neu/            # 预处理后的 6 类数据集
├── neu_det/                  # 过滤后的 4 类数据集（含 data.yaml）
├── neu_train_results*/       # 各实验的训练输出（权重 / 曲线 / 混淆矩阵）
├── yolo11m_detection_results/  # 检测结果输出
└── runs/                     # ultralytics 运行记录
```

---

## 快速开始

1. 安装依赖（见 [环境依赖](#环境依赖)）。
2. 准备 NEU-DET 数据集，并运行 `dataset.py` 生成 4 类数据集 `neu_det/`。
3. 启动训练（以优化版为例）：

```bash
python train.py
```

4. 训练完成后，启动 GUI 进行检测演示：

```bash
python gggggggui.py
```

> GUI 默认加载 `neu_train_results/yolov11m_neu_4classes_opt/weights/best.pt`，请确保该权重已存在。

---

## 数据预处理

`dataset.py` 完成以下工作：

1. 读取 `processed_neu/`（6 类数据集）；
2. 删除 `crazing`、`rolled-in_scale` 两类；
3. 将保留的 4 类重新映射为 0~3 的类别 ID；
4. 复制图像并重写标签，生成 `neu_det/images` 与 `neu_det/labels`；
5. 自动生成 `neu_det/data.yaml` 配置文件；
6. 可视化若干过滤后的样本以人工核对。

---

## 模型训练

各训练脚本对应不同的消融实验：

| 脚本 | 实验内容 | 输出目录 |
| --- | --- | --- |
| `train_base.py` | 基线 YOLO11m（无注意力、无优化） | neu_train_results_baseline |
| `train_bb2.py` | L-CBAM + 余弦退火 + 分层预热 | neu_train_results_lcbam_lr |
| `trainoptimi.py` | L-CBAM + 完整超参优化 | neu_train_results_optimized |
| `train.py` | L-CBAM + 完整超参优化 + TTA + ONNX 导出 | neu_train_results |
| `opop.py` | 增强版（多尺度训练 + 类别权重 + 混淆矩阵） | neu_train_results_enhanced |

核心训练配置（以 `train.py` 为例）：

- 模型：`yolov11m`（预训练 `yolo11m.pt`）；
- 输入尺寸：320×320；
- 训练轮数：200 epoch；批大小：12；
- 优化器：SGD，初始学习率 `lr0=0.0015`，余弦退火，5 轮预热；
- 数据增强：Mosaic、MixUp、Copy-Paste、HSV 扰动、翻转、缩放、透视等；
- 损失权重：`box=8.5`、`cls=0.5`、`dfl=1.5`；
- 验证：TTA + 低置信度阈值评估；训练后尝试导出 ONNX。

---

## 测试与评估

`test.py` 提供两种能力：

- **检测**：加载最佳模型，对单张图 / 文件夹 / 视频执行检测，保存带框结果图、标签 txt 与裁剪目标；
- **批量验证**：对 `neu_det` 的测试集计算 mAP、Precision、Recall 等指标。

`36.py` 提供带 TTA 的独立验证，适合在训练结束后单独复测指标。

---

## GUI 使用说明

运行 `python gggggggui.py` 后，界面包含：

- **选择图片 / 打开摄像头**：加载待检测的图片或调用摄像头实时检测；
- **开始检测 / 停止检测**：控制检测流程；
- **置信度阈值**：0.05 ~ 0.90 可调，实时刷新；
- **检测结果列表**：显示每个缺陷的编号、类别、置信度与面积；
- **缺陷详情面板**：点击列表项，展示该缺陷的中文名称、成因描述、解决方案与参考图片；
- **保存检测结果**：导出带检测框的图片。

不同缺陷类别使用不同颜色的检测框，便于快速区分。

> 说明：GUI 内置的缺陷知识库覆盖 6 类缺陷；而训练得到的检测模型为 4 类（`crazing`、`rolled-in_scale` 在数据预处理阶段被移除）。如需检测全部 6 类，请相应调整 `dataset.py` 与重新训练模型。

---

## 实验结果

以下为各实验在 NEU-DET（4 类）验证 / 测试集上的指标（数据来源：各实验目录下的 `results.csv`）：

| 实验 | 说明 | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall |
| --- | --- | --- | --- | --- | --- |
| 基线 | YOLO11m 无改进 | 0.8851 | 0.5508 | 0.8359 | 0.8384 |
| L-CBAM + 学习率策略 | 余弦退火 + 预热 | 0.8867 | 0.5503 | 0.8414 | 0.7860 |
| 优化版（optimized） | L-CBAM + 超参优化 | 0.8878 | 0.5489 | 0.8470 | 0.8101 |
| 最终优化版（GUI 使用） | 进一步调优 | 0.8916 | 0.5629 | 0.8533 | 0.7955 |

最终模型（`neu_train_results/yolov11m_neu_4classes_opt/weights/best.pt`）取得 **mAP@0.5 = 0.8916**，mAP@0.5:0.95 = 0.5629，验证了轻量化注意力增强 YOLO11 在钢板表面缺陷检测任务上的有效性。

---

## 常见问题

**Q1：运行报数据集路径不存在？**

脚本中的数据集路径为开发机绝对路径，迁移后请统一修改 `DATA_YAML`、`MODEL_PATH` 等为实际路径。

**Q2：找不到最佳权重 `best.pt`？**

请先运行对应训练脚本，训练完成后权重会保存到 `neu_train_results*/<实验名>/weights/best.pt`，并在 GUI 中指向该文件。

**Q3：没有 GPU 可以训练 / 检测吗？**

可以，将脚本中的 `DEVICE` 改为 `"cpu"` 即可，但训练和推理速度会明显下降。

**Q4：为什么模型只检测 4 类，而不是 6 类？**

`dataset.py` 在预处理时删除了 `crazing` 与 `rolled-in_scale` 两类，如需 6 类检测，请修改该脚本并重新训练。

---

## 参考与致谢

- [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics)
- [CBAM: Convolutional Block Attention Module](https://arxiv.org/abs/1807.06521)
- [NEU-DET](http://faculty.neu.edu.cn/songkechen/zh_CN/zdylm/263270/list/)：钢材表面缺陷数据集
