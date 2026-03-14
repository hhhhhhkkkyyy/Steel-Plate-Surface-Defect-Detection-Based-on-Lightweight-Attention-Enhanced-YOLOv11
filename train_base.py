import os
import warnings
import matplotlib
import matplotlib.pyplot as plt
from ultralytics import YOLO

# 基础配置
matplotlib.use('TkAgg')
plt.rcParams["font.family"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
warnings.filterwarnings('ignore')

# ================= 基础配置参数 =================
DATA_YAML = r"D:\666\PyCharm 2023.3.2\pythonProject\123\neu_det\data.yaml"
MODEL_TYPE = "yolov11m"
LOCAL_MODEL_PATH = r"D:\666\PyCharm 2023.3.2\pythonProject\123\yolo11m.pt"
EPOCHS = 200
BATCH_SIZE = 12
IMG_SIZE = 320
DEVICE = 0
SAVE_DIR = r"D:\666\PyCharm 2023.3.2\pythonProject\123\neu_train_results_baseline"


# ================= 纯基础训练函数 =================
def train_yolov11_baseline():
    # 检查预训练权重
    if not os.path.exists(LOCAL_MODEL_PATH):
        raise FileNotFoundError(f"预训练权重不存在：{LOCAL_MODEL_PATH}")

    # 加载基础模型（无任何修改）
    model = YOLO(LOCAL_MODEL_PATH)
    print(f"\n===== 开始YOLOv11m基础训练（无优化/无注意力）=====")

    # 基础训练配置（仅保留核心参数，无任何超参数优化）
    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMG_SIZE,
        device=DEVICE,
        project=SAVE_DIR,
        name=f"{MODEL_TYPE}_neu_4classes_baseline",
        exist_ok=True,
        save=True,
        # 仅保留基础参数，使用默认超参数
        patience=50,
        cache='disk',
        amp=True,
        seed=42,
        plots=True,
        save_period=20,
        workers=4,
    )

    # 基础验证
    print("\n===== 基础验证（无TTA）=====")
    val_results = model.val(
        data=DATA_YAML,
        imgsz=IMG_SIZE,
        batch=16,
        device=DEVICE,
        conf=0.25,  # 默认置信度
        iou=0.5,  # 默认IOU
        augment=False,  # 无TTA
        half=True,
        save_json=True,
        plots=True,
        verbose=True
    )

    # 打印结果
    m = val_results.results_dict
    print(f"\n===== 基础训练最终成绩 =====")
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

    # 开始基础训练
    train_yolov11_baseline()
    print("=" * 70)
    print("🎉 基础训练完成！去查看训练结果目录吧！")
    print("=" * 70)