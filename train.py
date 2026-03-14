import os
import warnings
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.torch_utils import torch, nn
import yaml

matplotlib.use('TkAgg')
plt.rcParams["font.family"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
warnings.filterwarnings('ignore')

# ================= 配置参数（YOLOv11m NEU-DET 4类优化版）=================
DATA_YAML = r"D:\666\PyCharm 2023.3.2\pythonProject\123\neu_det\data.yaml"
MODEL_TYPE = "yolov11m"
LOCAL_MODEL_PATH = r"D:\666\PyCharm 2023.3.2\pythonProject\123\yolo11m.pt"
EPOCHS = 200
BATCH_SIZE = 12            # 4060 Laptop 8GB 显存安全值（320分辨率下）
IMG_SIZE = 320
DEVICE = 0
SAVE_DIR = r"D:\666\PyCharm 2023.3.2\pythonProject\123\neu_train_results"

# ================= 轻量级CBAM注意力机制 =================
class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels)
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid()
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.shape
        avg_out = self.fc(self.avg_pool(x).view(b, c)).view(b, c, 1, 1)
        max_out = self.fc(self.max_pool(x).view(b, c)).view(b, c, 1, 1)
        channel_att = self.sigmoid(avg_out + max_out)
        x = x * channel_att

        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = self.spatial(torch.cat([avg_out, max_out], dim=1))
        x = x * spatial_att
        return x


# ================= 嵌入CBAM注意力 =================
def inject_attention(model):
    try:
        for idx, m in enumerate(model.model.model):
            # 在 C2f/C3k2 等大通道模块后插入
            if hasattr(m, 'c') and getattr(m, 'c', 0) > 64:
                cbam = CBAM(channels=m.c)
                original_forward = m.forward
                def new_forward(x, orig=original_forward, cb=cbam):
                    x = orig(x)
                    return cb(x)
                m.forward = new_forward
        print("✅ CBAM注意力模块已成功嵌入骨干网络")
    except Exception as e:
        print(f"⚠️ CBAM嵌入失败（将继续训练）：{e}")
    return model


# ================= 版本检查（可选）=================
def ensure_ultralytics_version():
    try:
        import ultralytics
        print(f"当前 ultralytics 版本: {ultralytics.__version__}")
    except:
        print("⚠️ ultralytics 未安装，正在尝试安装...")
        os.system("pip install ultralytics")
        exit()


# ================= 主训练函数 =================
def train_yolov11():
    if not os.path.exists(LOCAL_MODEL_PATH):
        raise FileNotFoundError(f"预训练权重不存在：{LOCAL_MODEL_PATH}")

    # 注意：YOLOv11 是 anchor-free 架构，重新计算锚框无效（仅提示）
    print("ℹ️ YOLOv11 为 anchor-free 模型，无需/无法自定义锚框，已跳过锚框计算")

    # 加载模型并注入注意力
    model = YOLO(LOCAL_MODEL_PATH)
    model = inject_attention(model)

    print(f"\n===== 开始YOLOv11m优化训练（NEU-DET 4类缺陷，目标 mAP@0.5 > 85%）=====")

    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        batch=BATCH_SIZE,           # 固定batch，显存友好
        imgsz=IMG_SIZE,
        device=DEVICE,
        project=SAVE_DIR,
        name=f"{MODEL_TYPE}_neu_4classes_opt",
        exist_ok=True,
        save=True,

        # ================= 关键超参数优化 =================
        optimizer='SGD',            # SGD 更适合小数据集收敛
        lr0=0.0015,                 # 较低初始学习率 + 余弦退火
        lrf=0.01,
        cos_lr=True,                # 余弦退火
        warmup_epochs=5,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,

        patience=80,
        close_mosaic=20,            # 最后20轮关闭mosaic，提升精度

        # ================= 数据增强（针对工业缺陷） =================
        mosaic=1.0,
        mixup=0.3,
        copy_paste=0.2,
        hsv_h=0.02, hsv_s=0.8, hsv_v=0.5,
        degrees=15.0,
        translate=0.15,
        scale=0.7,
        shear=3.0,
        perspective=0.001,
        flipud=0.5,
        fliplr=0.5,

        # ================= 损失权重 =================
        box=8.5,                    # 加强边界框回归（缺陷边缘重要）
        cls=0.5,
        dfl=1.5,

        # ================= 训练稳定性 =================
        cache='disk',
        amp=True,                   # 混合精度
        seed=42,
        plots=True,                 # 训练过程绘图
        save_period=20,             # 每20轮保存一次权重
        workers=4,                  # Windows下建议≤4，避免多进程问题
    )

    # ================= 终极验证（TTA + 低阈值榨指标）=================
    print("\n===== 开始终极验证（TTA + 低置信度）=====")
    val_results = model.val(
        data=DATA_YAML,
        imgsz=IMG_SIZE,
        batch=16,
        device=DEVICE,
        conf=0.001,
        iou=0.65,
        augment=True,               # TTA
        half=True,
        save_json=True,
        plots=True,
        verbose=True
    )

    m = val_results.results_dict
    print(f"\n===== 最终成绩 =====")
    print(f"mAP@0.5      : {m['metrics/mAP50(B)']:.4f}")
    print(f"mAP@0.5:0.95 : {m['metrics/mAP50-95(B)']:.4f}")
    print(f"Precision    : {m['metrics/precision(B)']:.4f}")
    print(f"Recall       : {m['metrics/recall(B)']:.4f}")

    # ================= 导出模型 =================
    best_pt = os.path.join(SAVE_DIR, f"{MODEL_TYPE}_neu_4classes_opt", "weights", "best.pt")
    print(f"\n✅ 最佳模型路径：{best_pt}")
    try:
        model.export(format='onnx', imgsz=IMG_SIZE, optimize=True)
        print("✅ ONNX 模型导出成功")
    except Exception as e:
        print(f"⚠️ ONNX 导出失败：{e}")

    return model


# ================= 主入口 =================
if __name__ == '__main__':
    # Windows 多进程安全保护
    try:
        from sklearn.cluster import KMeans  # 只是触发安装提示（实际不用）
    except ImportError:
        print("ℹ️ scikit-learn 未安装（本脚本无需锚框聚类，已跳过）")

    ensure_ultralytics_version()
    train_yolov11()
    print("=" * 70)
    print("🎉 训练全部完成！去查看 runs 目录下的结果吧！")
    print("=" * 70)
