# import os
# import warnings
# import matplotlib
# import matplotlib.pyplot as plt
# import torch
# import torch.nn as nn
# from ultralytics import YOLO
#
# # 基础配置
# matplotlib.use('TkAgg')
# plt.rcParams["font.family"] = ["SimHei", "DejaVu Sans"]
# plt.rcParams["axes.unicode_minus"] = False
# warnings.filterwarnings('ignore')
#
# # ================= 配置参数 =================
# DATA_YAML = r"D:\666\PyCharm 2023.3.2\pythonProject\123\neu_det\data.yaml"
# MODEL_TYPE = "yolov11m"
# LOCAL_MODEL_PATH = r"D:\666\PyCharm 2023.3.2\pythonProject\123\yolo11m.pt"
# EPOCHS = 200
# BATCH_SIZE = 12
# IMG_SIZE = 320
# DEVICE = 0
# SAVE_DIR = r"D:\666\PyCharm 2023.3.2\pythonProject\123\neu_train_results_lcbam"
#
#
# # ================= 轻量化CBAM注意力机制（L-CBAM）=================
# class LightweightCBAM(nn.Module):
#     def __init__(self, channels, reduction=16):  # 轻量化：缩减率提升至16
#         super().__init__()
#         # 通道注意力分支（轻量化设计）
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#         self.fc = nn.Sequential(
#             nn.Linear(channels, channels // reduction),  # 缩减率16，减少参数量
#             nn.ReLU(inplace=True),
#             nn.Linear(channels // reduction, channels)
#         )
#         # 空间注意力分支（简化设计）
#         self.spatial = nn.Sequential(
#             nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False),  # 单层3×3卷积
#             nn.Sigmoid()
#         )
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         b, c, h, w = x.shape
#         # 通道注意力
#         avg_out = self.fc(self.avg_pool(x).view(b, c)).view(b, c, 1, 1)
#         max_out = self.fc(self.max_pool(x).view(b, c)).view(b, c, 1, 1)
#         channel_att = self.sigmoid(avg_out + max_out)
#         x = x * channel_att
#         # 空间注意力
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
#         spatial_att = self.spatial(torch.cat([avg_out, max_out], dim=1))
#         x = x * spatial_att
#         return x
#
#
# # ================= 嵌入轻量化CBAM注意力 =================
# def inject_lcbam(model):
#     try:
#         for idx, m in enumerate(model.model.model):
#             # 在大通道模块后插入L-CBAM
#             if hasattr(m, 'c') and getattr(m, 'c', 0) > 64:
#                 lcbam = LightweightCBAM(channels=m.c)
#                 original_forward = m.forward
#
#                 def new_forward(x, orig=original_forward, lc=lcbam):
#                     x = orig(x)
#                     return lc(x)
#
#                 m.forward = new_forward
#         print("✅ 轻量化CBAM（L-CBAM）注意力模块已成功嵌入骨干网络")
#     except Exception as e:
#         print(f"⚠️ L-CBAM嵌入失败（将继续训练）：{e}")
#     return model
#
#
# # ================= L-CBAM版本训练函数 =================
# def train_yolov11_lcbam():
#     # 检查预训练权重
#     if not os.path.exists(LOCAL_MODEL_PATH):
#         raise FileNotFoundError(f"预训练权重不存在：{LOCAL_MODEL_PATH}")
#
#     print("ℹ️ YOLOv11 为 anchor-free 模型，无需自定义锚框")
#
#     # 加载模型并嵌入L-CBAM（仅L-CBAM，无其他优化）
#     model = YOLO(LOCAL_MODEL_PATH)
#     model = inject_lcbam(model)
#     print(f"\n===== 开始YOLOv11m + 轻量化CBAM（L-CBAM）训练（无其他优化）=====")
#
#     # 训练配置（仅加L-CBAM，使用默认超参数）
#     results = model.train(
#         data=DATA_YAML,
#         epochs=EPOCHS,
#         batch=BATCH_SIZE,
#         imgsz=IMG_SIZE,
#         device=DEVICE,
#         project=SAVE_DIR,
#         name=f"{MODEL_TYPE}_neu_4classes_lcbam",
#         exist_ok=True,
#         save=True,
#         # 仅保留基础参数，使用默认超参数（无优化）
#         patience=50,
#         cache='disk',
#         amp=True,
#         seed=42,
#         plots=True,
#         save_period=20,
#         workers=4,
#     )
#
#     # 验证（无TTA）
#     print("\n===== L-CBAM版本验证（无TTA）=====")
#     val_results = model.val(
#         data=DATA_YAML,
#         imgsz=IMG_SIZE,
#         batch=16,
#         device=DEVICE,
#         conf=0.25,
#         iou=0.5,
#         augment=False,
#         half=True,
#         save_json=True,
#         plots=True,
#         verbose=True
#     )
#
#     # 打印结果
#     m = val_results.results_dict
#     print(f"\n===== L-CBAM版本最终成绩 =====")
#     print(f"mAP@0.5 : {m['metrics/mAP50(B)']:.4f}")
#     print(f"mAP@0.5:0.95 : {m['metrics/mAP50-95(B)']:.4f}")
#     print(f"Precision : {m['metrics/precision(B)']:.4f}")
#     print(f"Recall : {m['metrics/recall(B)']:.4f}")
#
#     return model
#
#
# # ================= 主入口 =================
# if __name__ == '__main__':
#     # 版本检查
#     try:
#         import ultralytics
#         print(f"当前 ultralytics 版本: {ultralytics.__version__}")
#     except:
#         print("⚠️ ultralytics 未安装，正在尝试安装...")
#         os.system("pip install ultralytics")
#         exit()
#
#     # 开始L-CBAM训练
#     train_yolov11_lcbam()
#     print("=" * 70)
#     print("🎉 轻量化CBAM（L-CBAM）版本训练完成！去查看训练结果目录吧！")
#     print("=" * 70)
import os
import warnings
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from ultralytics import YOLO

# 基础配置
matplotlib.use('TkAgg')
plt.rcParams["font.family"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
warnings.filterwarnings('ignore')

# ================= 配置参数 =================
DATA_YAML = r"D:\666\PyCharm 2023.3.2\pythonProject\123\neu_det\data.yaml"
MODEL_TYPE = "yolov11m"
LOCAL_MODEL_PATH = r"D:\666\PyCharm 2023.3.2\pythonProject\123\yolo11m.pt"
EPOCHS = 200
BATCH_SIZE = 12
IMG_SIZE = 320
DEVICE = 0
# 修改保存目录，以区分实验
SAVE_DIR = r"D:\666\PyCharm 2023.3.2\pythonProject\123\neu_train_results_lcbam_lr"


# ================= 轻量化CBAM注意力机制（L-CBAM）=================
# （此部分代码与实验2完全相同，作为控制变量）
class LightweightCBAM(nn.Module):
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


# ================= 嵌入轻量化CBAM注意力 =================
# （此部分代码与实验2完全相同，作为控制变量）
def inject_lcbam(model):
    try:
        for idx, m in enumerate(model.model.model):
            if hasattr(m, 'c') and getattr(m, 'c', 0) > 64:
                lcbam = LightweightCBAM(channels=m.c)
                original_forward = m.forward

                def new_forward(x, orig=original_forward, lc=lcbam):
                    x = orig(x)
                    return lc(x)

                m.forward = new_forward
        print("✅ 轻量化CBAM（L-CBAM）注意力模块已成功嵌入骨干网络")
    except Exception as e:
        print(f"⚠️ L-CBAM嵌入失败（将继续训练）：{e}")
    return model


# ================= L-CBAM + 学习率策略版本训练函数 =================
def train_yolov11_lcbam_lr():
    # 检查预训练权重
    if not os.path.exists(LOCAL_MODEL_PATH):
        raise FileNotFoundError(f"预训练权重不存在：{LOCAL_MODEL_PATH}")

    print("ℹ️ YOLOv11 为 anchor-free 模型，无需自定义锚框")

    # 加载模型并嵌入L-CBAM
    model = YOLO(LOCAL_MODEL_PATH)
    model = inject_lcbam(model)
    print(f"\n===== 开始YOLOv11m + 轻量化CBAM + 余弦退火+分层预热训练 =====")

    # 训练配置（在L-CBAM基础上，添加学习率策略）
    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMG_SIZE,
        device=DEVICE,
        project=SAVE_DIR,
        name=f"{MODEL_TYPE}_neu_4classes_lcbam_lr", # 修改保存名称
        exist_ok=True,
        save=True,

        # ================= 新增：余弦退火+分层预热学习率策略 =================
        optimizer='SGD',            # SGD更适合小数据集收敛
        lr0=0.0015,                 # 较低初始学习率
        lrf=0.01,                   # 最终学习率
        cos_lr=True,                # 启用余弦退火
        warmup_epochs=5,            # 预热轮次
        warmup_momentum=0.8,        # 预热初始动量
        warmup_bias_lr=0.1,         # 偏置项更高的预热学习率

        # ================= 保留基础参数 =================
        patience=50,
        cache='disk',
        amp=True,
        seed=42,
        plots=True,
        save_period=20,
        workers=4,
    )

    # 验证（无TTA，与实验2保持一致）
    print("\n===== L-CBAM + 学习率策略版本验证（无TTA）=====")
    val_results = model.val(
        data=DATA_YAML,
        imgsz=IMG_SIZE,
        batch=16,
        device=DEVICE,
        conf=0.25,
        iou=0.5,
        augment=False,
        half=True,
        save_json=True,
        plots=True,
        verbose=True
    )

    # 打印结果
    m = val_results.results_dict
    print(f"\n===== L-CBAM + 学习率策略版本最终成绩 =====")
    print(f"mAP@0.5 : {m['metrics/mAP50(B)']:.4f}")
    print(f"mAP@0.5:0.95 : {m['metrics/mAP50-95(B)']:.4f}")
    print(f"Precision : {m['metrics/precision(B)']:.4f}")
    print(f"Recall : {m['metrics/recall(B)']:.4f}")

    return model


# ================= 主入口 =================
if __name__ == '__main__':
    # 版本检查
    try:
        import ultralytics
        print(f"当前 ultralytics 版本: {ultralytics.__version__}")
    except:
        print("⚠️ ultralytics 未安装，正在尝试安装...")
        os.system("pip install ultralytics")
        exit()

    # 开始L-CBAM + 学习率策略训练
    train_yolov11_lcbam_lr()
    print("=" * 70)
    print("🎉 轻量化CBAM + 余弦退火+分层预热版本训练完成！去查看训练结果目录吧！")
    print("=" * 70)
