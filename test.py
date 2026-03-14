# # ================= 所有函数和类定义保持不变 =================
#
# if __name__ == '__main__':  # ← 关键：加上这一行保护！
#     from ultralytics import YOLO
#
#     # ensure_ultralytics_version()  # 如果有这个函数
#
#     # 加载最佳权重继续训练（resume）
#     model = YOLO(
#         r"D:\666\PyCharm 2023.3.2\pythonProject\123\neu_train_results_final8\yolov8n_neu_det_FINAL_INNOV\weights\best.pt")
#
#     # 继续训练（推荐关闭 plots 防止后期 OOM）
#     results = model.train(
#         resume=True,  # 自动从 best.pt 继续
#         plots=False,  # ← 训练时关闭绘图，防止显存炸
#         save_period=20,  # 每20轮保存一次
#         patience=100,
#         batch=8,           # 你当前能跑16就保持，炸了再降到8
#         workers=0,          # 可以保持，如果还报错临时改成 workers=0（慢但稳）
#     )
#
#     print("训练完成！现在开始生成可视化图...")
#
#     # ================= 训练完成后单独生成所有可视化图 =================
#     model.val(
#         data=r"D:\666\PyCharm 2023.3.2\pythonProject\123\processed_neu\data_updated.yaml",
#         imgsz=640,
#         batch=16,
#         device=0,
#         plots=True,  # ← 这里打开，一次性生成所有漂亮的图！
#         save=True,
#         save_json=True,
#         conf=0.001,
#         iou=0.6,
#         augment=True,  # TTA 榨指标
#         verbose=True,
#         # workers=0,
#     )
#
#     # 额外生成完整训练曲线
#     from ultralytics.utils.plotting import plot_results
#
#     plot_results(
#         r"D:\666\PyCharm 2023.3.2\pythonProject\123\neu_train_results_final8\yolov8n_neu_det_FINAL_INNOV\results.csv")

import os
import warnings
import matplotlib
import matplotlib.pyplot as plt
from ultralytics import YOLO
import cv2

# ===================== 基础配置（解决中文显示和负号问题）=====================
matplotlib.use('TkAgg')  # 后端配置，兼容大多数环境
plt.rcParams["font.family"] = ["SimHei", "DejaVu Sans"]  # 支持中文显示
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示为方框的问题
warnings.filterwarnings('ignore')  # 忽略无关警告

# ===================== 核心配置（需根据你的实际情况修改）=====================
# 训练好的最佳模型路径（对应训练代码中的最佳模型保存路径）
BEST_MODEL_PATH = r"D:\666\PyCharm 2023.3.2\pythonProject\123\neu_train_results\yolov11m_neu_4classes_opt\weights\best.pt"
# 输入待检测的目标（可选：单张图像路径/图像文件夹路径/视频路径）
INPUT_TARGET = r"D:\666\PyCharm 2023.3.2\pythonProject\123\neu_det\images\test"  # 可修改为单张图如 "test.jpg"
# 检测结果保存目录
OUTPUT_DIR = r"D:\666\PyCharm 2023.3.2\pythonProject\123\yolo11m_detection_results"
# 检测配置参数
CONF_THRESHOLD = 0.5  # 置信度阈值（过滤低置信度检测结果）
IOU_THRESHOLD = 0.45  # NMS IoU阈值
IMG_SIZE = 320  # 检测图像尺寸（与训练保持一致）
DEVICE = 0  # 检测设备（0=GPU，"cpu"=CPU）
AUGMENT = False  # 是否开启TTA测试增强（True=开启，检测速度变慢但精度更高）

def detect_with_yolov11():
    """
    加载YOLOv11最佳模型，执行检测任务，并保存检测结果
    """
    # 1. 检查模型文件是否存在
    if not os.path.exists(BEST_MODEL_PATH):
        raise FileNotFoundError(f"最佳模型文件不存在，请检查路径：{BEST_MODEL_PATH}")

    # 2. 创建结果保存目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 3. 加载YOLOv11模型
    print(f"正在加载模型：{BEST_MODEL_PATH}")
    model = YOLO(BEST_MODEL_PATH)
    print("模型加载成功！")

    # 4. 执行检测任务
    print(f"\n正在对目标：{INPUT_TARGET} 执行检测...")
    results = model(
        source=INPUT_TARGET,  # 待检测目标
        imgsz=IMG_SIZE,  # 图像尺寸
        conf=CONF_THRESHOLD,  # 置信度阈值
        iou=IOU_THRESHOLD,  # NMS IoU阈值
        device=DEVICE,  # 检测设备
        augment=AUGMENT,  # TTA增强
        save=True,  # 保存检测结果图像/视频
        save_txt=True,  # 保存检测结果标注文件（txt格式，每行对应一个检测框）
        save_conf=True,  # 保存置信度到txt文件
        save_crop=True,  # 保存裁剪后的检测目标
        project=OUTPUT_DIR,  # 结果保存根目录
        name="yolov11m_detect",  # 检测结果文件夹名称
        exist_ok=True,  # 覆盖已有同名文件夹
        show_labels=True,  # 显示检测标签
        show_conf=True,  # 显示置信度
        line_width=2  # 检测框线宽
    )

    # 5. 打印检测结果详情
    print(f"\n===== 检测完成！=====")
    for idx, result in enumerate(results):
        # 提取单条检测结果信息
        img_path = result.path
        det_boxes = result.boxes  # 检测框信息（坐标、置信度、类别）
        num_detections = len(det_boxes) if det_boxes is not None else 0
        class_names = result.names  # 类别名称映射

        # 打印单条结果详情
        print(f"\n【检测目标 {idx+1}】: {img_path}")
        print(f"检测到目标数量：{num_detections}")
        if num_detections > 0:
            for box in det_boxes:
                cls_id = int(box.cls[0])  # 类别ID
                cls_name = class_names[cls_id]  # 类别名称
                conf = box.conf[0].item()  # 置信度
                xyxy = box.xyxy[0].tolist()  # 检测框坐标（左上x, 左上y, 右下x, 右下y）
                print(f"  类别：{cls_name} | 置信度：{conf:.4f} | 坐标：{xyxy}")

    # 6. 提示结果保存路径
    detection_save_path = os.path.join(OUTPUT_DIR, "yolov11m_detect")
    print(f"\n所有检测结果保存至：{detection_save_path}")
    print(f"  - 检测图像/视频：{detection_save_path}（直接可视化检测框）")
    print(f"  - 标注文件（txt）：{detection_save_path}/labels")
    print(f"  - 裁剪目标：{detection_save_path}/crops")

    return results

def batch_val_test():
    """
    可选：批量验证/测试（与训练代码中的val逻辑一致，输出mAP等评估指标）
    适用于有标注的测试集/验证集
    """
    # 数据集配置文件路径（必须与训练一致，否则无法找到测试集）
    DATA_YAML = r"D:\666\PyCharm 2023.3.2\pythonProject\123\neu_det\data.yaml"
    # 核对上述路径是否存在，若不存在会报错，需手动修改为正确路径
    if not os.path.exists(DATA_YAML):
        raise FileNotFoundError(f"数据集配置文件不存在：{DATA_YAML}")
    model = YOLO(BEST_MODEL_PATH)

    print(f"\n===== 开始批量验证（使用test集）=====")
    val_results = model.val(
        data=DATA_YAML,
        split='test',  # 使用test集评估（也可改为'val'使用验证集）
        imgsz=IMG_SIZE,
        batch=16,
        device=DEVICE,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        augment=AUGMENT,
        save_json=False,
        plots=True,
        project=OUTPUT_DIR,
        name="yolov11m_val_results",
        exist_ok=True
    )

    # 提取并打印核心评估指标（这部分就是批量验证指标的输出逻辑）
    metrics = val_results.results_dict
    print(f"\n===== 批量验证指标 ======")
    print(f"mAP@0.5     : {metrics['metrics/mAP50(B)']:.4f}")
    print(f"mAP@0.5:0.95: {metrics['metrics/mAP50-95(B)']:.4f}")
    print(f"Precision   : {metrics['metrics/precision(B)']:.4f}")
    print(f"Recall      : {metrics['metrics/recall(B)']:.4f}")

def main():
    """
    主函数：程序入口
    """
    print("=" * 70)
    print("YOLOv11m 模型独立检测脚本")
    print("=" * 70)

    # 执行检测任务
    # detect_with_yolov11()

    # 可选：执行批量验证（如需评估mAP等指标，取消注释下方代码）
    batch_val_test()

    print("\n=" * 70)
    print("检测流程全部完成！")
    print("=" * 70)

if __name__ == "__main__":
    main()