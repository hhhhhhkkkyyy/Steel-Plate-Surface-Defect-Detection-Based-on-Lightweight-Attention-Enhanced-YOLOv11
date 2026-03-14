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

# ================= 配置参数 =================
DATA_YAML = r"D:\666\PyCharm 2023.3.2\pythonProject\123\neu_det\data.yaml"
IMG_SIZE = 320
DEVICE = 0

# 请修改为您训练好的enhance模型路径
ENHANCED_MODEL_PATH = r"D:\666\PyCharm 2023.3.2\pythonProject\123\neu_train_results_optimized\yolov11m_neu_4classes_opt\weights\best.pt"


# ================= 带TTA的独立验证函数 =================
def validate_enhanced_model_tta():
    # 检查模型文件是否存在
    if not os.path.exists(ENHANCED_MODEL_PATH):
        raise FileNotFoundError(f"模型文件不存在：{ENHANCED_MODEL_PATH}\n请检查路径是否正确")

    print(f"正在加载增强模型：{ENHANCED_MODEL_PATH}")

    # 加载训练好的模型
    model = YOLO(ENHANCED_MODEL_PATH)
    print("✅ 模型加载成功")

    # 验证配置（启用TTA，与训练时保持一致）
    print("\n===== 开始验证增强模型（启用TTA）=====")
    val_results = model.val(
        data=DATA_YAML,
        imgsz=IMG_SIZE,
        batch=16,
        device=DEVICE,
        conf=0.001,  # 极低置信度阈值，保留更多候选框
        iou=0.65,  # 宽松的NMS阈值，减少有效框丢失
        augment=True,  # 启用测试时增强（TTA）
        half=True,  # 使用FP16推理
        save_json=True,  # 保存COCO格式的JSON结果
        plots=True,  # 绘制验证结果图
        verbose=True  # 详细输出
    )

    # 打印结果（格式与之前完全一致）
    m = val_results.results_dict
    print(f"\n===== 增强模型最终成绩（TTA版本）=====")
    print(f"mAP@0.5 : {m['metrics/mAP50(B)']:.4f}")
    print(f"mAP@0.5:0.95 : {m['metrics/mAP50-95(B)']:.4f}")
    print(f"Precision : {m['metrics/precision(B)']:.4f}")
    print(f"Recall : {m['metrics/recall(B)']:.4f}")

    # 额外信息
    print(f"\n===== TTA验证完成信息 =====")
    print(f"验证结果已保存至：{val_results.save_dir}")
    print(f"包含文件：混淆矩阵、PR曲线、检测结果可视化等")
    print("ℹ️ TTA已启用：通过水平翻转、多尺度缩放提升检测鲁棒性")

    return val_results


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

    # 开始TTA验证
    try:
        validate_enhanced_model_tta()
        print("=" * 70)
        print("🎉 增强模型TTA验证完成！去查看验证结果目录吧！")
        print("=" * 70)
    except Exception as e:
        print(f"❌ 验证过程中出现错误：{e}")
        print("请检查模型路径和数据配置是否正确")
