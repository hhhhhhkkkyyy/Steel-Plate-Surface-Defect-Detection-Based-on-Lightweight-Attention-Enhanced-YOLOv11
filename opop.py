import os
import warnings
import psutil
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
from ultralytics import YOLO
from ultralytics.utils.torch_utils import torch, nn, select_device
from ultralytics.utils.metrics import ConfusionMatrix
import yaml

matplotlib.use('TkAgg')
plt.rcParams["font.family"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
warnings.filterwarnings('ignore')


# ================= 配置参数（适配ultralytics 8.3.235）=================
class Config:
    # 基础路径
    DATA_YAML = r"D:\666\PyCharm 2023.3.2\pythonProject\123\neu_det\data.yaml"
    MODEL_TYPE = "yolov11m"
    LOCAL_MODEL_PATH = r"D:\666\PyCharm 2023.3.2\pythonProject\123\yolo11m.pt"
    SAVE_DIR = r"D:\666\PyCharm 2023.3.2\pythonProject\123\neu_train_results_enhanced"

    # 训练参数（适配8.3.235版本）
    EPOCHS = 200
    BASE_BATCH_SIZE = 12  # 基础batch
    IMG_SIZE = 320
    DEVICE = 0
    PATIENCE = 80
    SEED = 42

    # 超参数
    LR0 = 0.0015
    LRF = 0.01
    WARMUP_EPOCHS = 5
    BOX_LOSS_WEIGHT = 8.5
    CLS_LOSS_WEIGHT = 0.5
    DFL_LOSS_WEIGHT = 1.5

    # 数据增强
    MOSAIC = 1.0
    MIXUP = 0.3
    CLOSE_MOSAIC = 20

    # 工业缺陷类别权重（通过自定义损失实现，替代class_weights参数）
    CLASS_WEIGHTS = [1.0, 1.2, 1.5, 1.1]  # 夹杂最难，权重最高
    # This is the incorrect way to use multi_scale
    multi_scale = [320, 384, 416]

    # 多尺度训练
    VAL_MULTI_SCALE = True  # 验证时多尺度

    # 日志配置
    SAVE_CSV = True
    SAVE_CONFUSION_MATRIX = True


# ================= 改进版CBAM注意力机制（带初始化）=================
class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        # 通道注意力
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels)
        )
        # 空间注意力
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid()
        )
        self.sigmoid = nn.Sigmoid()

        # CBAM参数初始化（关键改进）
        self._init_weights()

    def _init_weights(self):
        """CBAM模块单独初始化，提升收敛速度"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        b, c, h, w = x.shape
        # 通道注意力
        avg_out = self.fc(self.avg_pool(x).view(b, c)).view(b, c, 1, 1)
        max_out = self.fc(self.max_pool(x).view(b, c)).view(b, c, 1, 1)
        channel_att = self.sigmoid(avg_out + max_out)
        x = x * channel_att

        # 空间注意力
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = self.spatial(torch.cat([avg_out, max_out], dim=1))
        x = x * spatial_att
        return x


# ================= 改进版CBAM嵌入（模块化，无闭包陷阱）=================
class CBAMWrapper(nn.Module):
    """包装器，避免动态重写forward的闭包陷阱"""

    def __init__(self, module, cbam):
        super().__init__()
        self.module = module
        self.cbam = cbam

    def forward(self, x):
        x = self.module(x)
        return self.cbam(x)


def inject_attention(model):
    """改进的CBAM嵌入方式"""
    try:
        device = next(model.model.parameters()).device
        for idx, m in enumerate(model.model.model):
            # 精准匹配YOLOv11的C2f/C3k2模块
            if hasattr(m, '__class__') and (
                    'C2f' in m.__class__.__name__ or 'C3k2' in m.__class__.__name__) and hasattr(m, 'c') and m.c > 64:
                # 创建CBAM并移到模型设备
                cbam = CBAM(channels=m.c).to(device)
                # 替换模块而非重写forward（更稳定）
                model.model.model[idx] = CBAMWrapper(m, cbam)
        print("✅ 改进版CBAM注意力模块已成功嵌入骨干网络")
    except Exception as e:
        print(f"⚠️ CBAM嵌入失败（将继续训练）：{e}")
    return model


# ================= 显存自适应batch size =================
def auto_adjust_batch_size(base_batch, img_size, device_id=0):
    """根据显存自动调整batch size"""
    try:
        device = select_device(device_id)
        if device.type == 'cuda':
            # 获取GPU显存信息
            gpu_mem = torch.cuda.get_device_properties(device).total_memory / 1024 ** 3
            # 8GB显存基准，动态调整
            if gpu_mem < 8:
                adjusted = max(1, base_batch // 2)
                print(f"⚠️ GPU显存不足（{gpu_mem:.1f}GB），batch size调整为：{adjusted}")
                return adjusted
        return base_batch
    except:
        return base_batch


# ================= 自定义类别权重损失（替代class_weights参数）=================
def set_class_weights(model, class_weights):
    """手动设置分类损失的类别权重（适配ultralytics 8.3.235）"""
    try:
        # 遍历模型找到分类损失层
        for m in model.model.modules():
            if hasattr(m, 'cls_loss') and isinstance(m.cls_loss, nn.CrossEntropyLoss):
                # 设置类别权重
                weights = torch.tensor(class_weights,
                                       device=m.cls_loss.weight.device if hasattr(m.cls_loss, 'weight') else next(
                                           model.model.parameters()).device)
                m.cls_loss.weight = nn.Parameter(weights)
                print(f"✅ 已设置类别权重：{class_weights}")
                break
    except Exception as e:
        print(f"⚠️ 类别权重设置失败：{e}")
    return model


# ================= 日志保存增强 =================
def save_training_logs(results, save_path, config):
    """保存训练日志到CSV"""
    if not config.SAVE_CSV:
        return

    # 创建日志目录
    log_dir = os.path.join(save_path, "logs")
    os.makedirs(log_dir, exist_ok=True)

    # 提取训练指标
    metrics = []
    for epoch in range(len(results.epoch)):
        metric_dict = {
            'epoch': results.epoch[epoch],
            'train/box_loss': results.box_loss[epoch] if epoch < len(results.box_loss) else np.nan,
            'train/cls_loss': results.cls_loss[epoch] if epoch < len(results.cls_loss) else np.nan,
            'train/dfl_loss': results.dfl_loss[epoch] if epoch < len(results.dfl_loss) else np.nan,
            'val/box_loss': results.val_box_loss[epoch] if epoch < len(results.val_box_loss) else np.nan,
            'val/cls_loss': results.val_cls_loss[epoch] if epoch < len(results.val_cls_loss) else np.nan,
            'val/dfl_loss': results.val_dfl_loss[epoch] if epoch < len(results.val_dfl_loss) else np.nan,
            'mAP50': results.metrics['metrics/mAP50(B)'][epoch] if epoch < len(
                results.metrics['metrics/mAP50(B)']) else np.nan,
            'mAP50-95': results.metrics['metrics/mAP50-95(B)'][epoch] if epoch < len(
                results.metrics['metrics/mAP50-95(B)']) else np.nan,
            'precision': results.metrics['metrics/precision(B)'][epoch] if epoch < len(
                results.metrics['metrics/precision(B)']) else np.nan,
            'recall': results.metrics['metrics/recall(B)'][epoch] if epoch < len(
                results.metrics['metrics/recall(B)']) else np.nan,
            'lr': results.lr['lr0'][epoch] if epoch < len(results.lr['lr0']) else np.nan
        }
        metrics.append(metric_dict)

    # 保存到CSV
    df = pd.DataFrame(metrics)
    csv_path = os.path.join(log_dir, f"training_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"✅ 训练日志已保存到：{csv_path}")

    # 绘制关键指标曲线
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # 损失曲线
    ax1.plot(df['epoch'], df['train/box_loss'], label='训练box loss', color='blue')
    ax1.plot(df['epoch'], df['val/box_loss'], label='验证box loss', color='red')
    ax1.set_title('边界框损失曲线')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # mAP曲线
    ax2.plot(df['epoch'], df['mAP50'], label='mAP@0.5', color='green')
    ax2.plot(df['epoch'], df['mAP50-95'], label='mAP@0.5:0.95', color='orange')
    ax2.set_title('mAP曲线')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('mAP')
    ax2.legend()
    ax2.grid(True)

    # 精度召回曲线
    ax3.plot(df['epoch'], df['precision'], label='Precision', color='purple')
    ax3.plot(df['epoch'], df['recall'], label='Recall', color='brown')
    ax3.set_title('精度-召回曲线')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Score')
    ax3.legend()
    ax3.grid(True)

    # 学习率曲线
    ax4.plot(df['epoch'], df['lr'], label='Learning Rate', color='black')
    ax4.set_title('学习率曲线')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('LR')
    ax4.legend()
    ax4.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()


# ================= 混淆矩阵保存 =================
def save_confusion_matrix(model, val_results, save_path, config):
    """保存混淆矩阵可视化"""
    if not config.SAVE_CONFUSION_MATRIX:
        return

    log_dir = os.path.join(save_path, "logs")
    os.makedirs(log_dir, exist_ok=True)

    # 获取类别名称
    with open(config.DATA_YAML, 'r', encoding='utf-8') as f:
        data_yaml = yaml.safe_load(f)
    names = data_yaml['names']

    # 生成混淆矩阵
    cm = ConfusionMatrix(names=names)
    cm.process_batch(val_results.pred, val_results.targets)
    fig, ax = cm.plot(verbose=False, save_dir=log_dir)
    plt.savefig(os.path.join(log_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ 混淆矩阵已保存")


# ================= 改进版TTA验证 =================
def enhanced_tta_validation(model, config):
    """增强版TTA验证（多尺度+水平翻转）"""
    print("\n===== 开始增强版TTA验证（多尺度+水平翻转）=====")

    # 多尺度验证
    val_scales = [320, 384, 416] if config.VAL_MULTI_SCALE else [config.IMG_SIZE]
    all_results = []

    for scale in val_scales:
        print(f"🔍 验证尺度：{scale}x{scale}")
        val_results = model.val(
            data=config.DATA_YAML,
            imgsz=scale,
            batch=16,
            device=config.DEVICE,
            conf=0.001,
            iou=0.65,
            augment=True,  # TTA：水平翻转+轻微缩放
            half=True,
            save_json=True,
            plots=False,
            verbose=False
        )
        all_results.append({
            'scale': scale,
            'mAP50': val_results.results_dict['metrics/mAP50(B)'],
            'mAP50-95': val_results.results_dict['metrics/mAP50-95(B)'],
            'precision': val_results.results_dict['metrics/precision(B)'],
            'recall': val_results.results_dict['metrics/recall(B)']
        })

    # 计算平均结果
    avg_mAP50 = np.mean([r['mAP50'] for r in all_results])
    avg_mAP50_95 = np.mean([r['mAP50-95'] for r in all_results])
    avg_precision = np.mean([r['precision'] for r in all_results])
    avg_recall = np.mean([r['recall'] for r in all_results])

    # 打印结果
    print(f"\n===== 增强版TTA验证最终成绩（多尺度平均）=====")
    print(f"mAP@0.5      : {avg_mAP50:.4f}")
    print(f"mAP@0.5:0.95 : {avg_mAP50_95:.4f}")
    print(f"Precision    : {avg_precision:.4f}")
    print(f"Recall       : {avg_recall:.4f}")

    # 保存详细结果
    results_df = pd.DataFrame(all_results)
    results_df.loc['average'] = [
        'average', avg_mAP50, avg_mAP50_95, avg_precision, avg_recall
    ]
    results_df.to_csv(
        os.path.join(config.SAVE_DIR, f"{config.MODEL_TYPE}_neu_4classes_opt_enhanced", "tta_validation_results.csv"),
        index=False, encoding='utf-8-sig')

    return {
               'metrics/mAP50(B)': avg_mAP50,
               'metrics/mAP50-95(B)': avg_mAP50_95,
               'metrics/precision(B)': avg_precision,
               'metrics/recall(B)': avg_recall
           }, val_results


# ================= 模型量化导出 =================
def export_quantized_model(model, config):
    """导出量化ONNX模型，适配工业部署"""
    best_pt = os.path.join(config.SAVE_DIR, f"{config.MODEL_TYPE}_neu_4classes_opt_enhanced", "weights", "best.pt")
    print(f"\n✅ 最佳模型路径：{best_pt}")

    try:
        # 导出标准ONNX
        model.export(
            format='onnx',
            imgsz=config.IMG_SIZE,
            optimize=True,
            dynamic=False,  # 工业部署建议固定尺寸
            simplify=True  # 简化模型
        )
        print("✅ 标准ONNX模型导出成功")

        # 尝试INT8量化（需ONNX Runtime）
        try:
            import onnx
            import onnxruntime as ort
            from onnxruntime.quantization import quantize_dynamic, QuantType

            onnx_path = best_pt.replace('.pt', '.onnx')
            quantized_onnx_path = best_pt.replace('.pt', '_int8.onnx')

            # 量化模型
            quantize_dynamic(
                model_input=onnx_path,
                model_output=quantized_onnx_path,
                weight_type=QuantType.QUInt8,
                optimize_model=True
            )
            print(f"✅ INT8量化ONNX模型导出成功：{quantized_onnx_path}")
        except Exception as e:
            print(f"⚠️ INT8量化失败（需安装onnxruntime）：{e}")
            print("💡 安装命令：pip install onnx onnxruntime-gpu")

    except Exception as e:
        print(f"⚠️ ONNX导出失败：{e}")

def train_yolov11_enhanced():
    config = Config()

    # 1. 基础检查
    if not os.path.exists(config.LOCAL_MODEL_PATH):
        raise FileNotFoundError(f"预训练权重不存在：{config.LOCAL_MODEL_PATH}")

    # 2. 显存自适应调整batch size
    batch_size = auto_adjust_batch_size(config.BASE_BATCH_SIZE, config.IMG_SIZE, config.DEVICE)

    print("ℹ️ YOLOv11 为 anchor-free 模型，无需/无法自定义锚框，已跳过锚框计算")
    print(f"\n===== 开始YOLOv11m增强训练（NEU-DET 4类缺陷，目标 mAP@0.5 > 90%）=====")
    # 修改后的代码
    print(f"📊 配置信息：batch={batch_size} | 多尺度={config.VAL_MULTI_SCALE} | 类别权重={config.CLASS_WEIGHTS}")

    # 3. 加载模型并注入改进版CBAM
    model = YOLO(config.LOCAL_MODEL_PATH)
    model = inject_attention(model)

    # 4. 设置类别权重（替代class_weights参数）
    model = set_class_weights(model, config.CLASS_WEIGHTS)

    # 5. 开始训练（适配8.3.235版本的参数）
    results = model.train(
        data=config.DATA_YAML,
        epochs=config.EPOCHS,
        batch=batch_size,
        imgsz=config.IMG_SIZE,
        device=config.DEVICE,
        project=config.SAVE_DIR,
        name=f"{config.MODEL_TYPE}_neu_4classes_opt_enhanced",
        exist_ok=True,
        save=True,

        # 核心超参数（保留原优化，移除无效参数）
        optimizer='SGD',
        lr0=config.LR0,
        lrf=config.LRF,
        cos_lr=True,
        warmup_epochs=config.WARMUP_EPOCHS,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        patience=config.PATIENCE,
        close_mosaic=config.CLOSE_MOSAIC,

        # 数据增强（保留并优化）
        mosaic=config.MOSAIC,
        mixup=config.MIXUP,
        copy_paste=0.2,
        hsv_h=0.02, hsv_s=0.8, hsv_v=0.5,
        degrees=15.0,
        translate=0.15,
        scale=0.7,
        shear=3.0,
        perspective=0.001,
        flipud=0.5,
        fliplr=0.5,

        # 损失权重（仅保留支持的参数）
        box=config.BOX_LOSS_WEIGHT,
        cls=config.CLS_LOSS_WEIGHT,
        dfl=config.DFL_LOSS_WEIGHT,

        # 训练稳定性增强（仅保留8.3.235支持的参数）
        cache='disk',
        amp=True,
        seed=config.SEED,
        plots=True,
        save_period=20,
        workers=4,
        multi_scale=True,  # Set this to True (Boolean)
        rect=False,  # 工业缺陷检测建议关闭rect训练
        half=True,
    )

    return model, results

# ================= 主训练函数（适配ultralytics 8.3.235）=================


# ================= 版本检查（适配8.3.235）=================
def ensure_ultralytics_version_enhanced():
    try:
        import ultralytics
        print(f"当前 ultralytics 版本: {ultralytics.__version__}")
        # 适配8.3.235版本提示
        if ultralytics.__version__ == '8.3.235':
            print("✅ 检测到ultralytics 8.3.235，已适配参数")
    except:
        print("⚠️ ultralytics 未安装，正在尝试安装...")
        os.system("pip install ultralytics==8.3.235")
        exit()


# ================= 主入口 =================
if __name__ == '__main__':
    # Windows 多进程安全保护
    try:
        from sklearn.cluster import KMeans
    except ImportError:
        print("ℹ️ scikit-learn 未安装（本脚本无需锚框聚类，已跳过）")

    # 版本检查
    ensure_ultralytics_version_enhanced()

    # 开始增强版训练
    model, metrics = train_yolov11_enhanced()

    print("=" * 70)
    print("🎉 增强版训练全部完成！")
    print(f"🏆 最终成绩：mAP@0.5={metrics['metrics/mAP50(B)']:.4f} | mAP@0.5:0.95={metrics['metrics/mAP50-95(B)']:.4f}")
    print(f"📁 结果目录：{Config.SAVE_DIR}")
    print("=" * 70)