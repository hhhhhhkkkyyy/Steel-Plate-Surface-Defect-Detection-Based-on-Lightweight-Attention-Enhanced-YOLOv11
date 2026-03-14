import os
import shutil
import yaml
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm

matplotlib.use('TkAgg')
# 配置中文字体
plt.rcParams["font.family"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ================= 核心配置 =================
# 原始预处理数据集路径（已生成的processed_neu）
SRC_ROOT = r"D:\666\PyCharm 2023.3.2\pythonProject\123\processed_neu"
# 新输出路径（存放过滤后的4类数据集）
DST_ROOT = r"D:\666\PyCharm 2023.3.2\pythonProject\123\neu_det"

# 需要删除的类别
REMOVE_CLASSES = ["crazing", "rolled-in_scale"]
# 保留的类别（自动计算）
KEEP_CLASSES = [cls for cls in ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
                if cls not in REMOVE_CLASSES]
# 保留类别的新ID映射（0-3）
CLASS_ID_MAP = {cls: idx for idx, cls in enumerate(KEEP_CLASSES)}


# ================= 工具函数 =================
def create_new_dirs():
    """创建新的数据集目录结构"""
    dirs = [
        os.path.join(DST_ROOT, "images", "train"),
        os.path.join(DST_ROOT, "images", "val"),
        os.path.join(DST_ROOT, "images", "test"),
        os.path.join(DST_ROOT, "labels", "train"),
        os.path.join(DST_ROOT, "labels", "val"),
        os.path.join(DST_ROOT, "labels", "test")
    ]
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    print(f"✅ 新数据集目录创建完成：{DST_ROOT}")


def filter_label_file(src_label_path, dst_label_path):
    """
    过滤标签文件：删除指定类别的标注，并重映射保留类别的ID
    返回：是否保留该标签文件（有有效标注则保留）
    """
    keep_lines = []
    with open(src_label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            # 原类别ID转类别名
            orig_cls_id = int(parts[0])
            orig_cls_name = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"][
                orig_cls_id]

            # 跳过需要删除的类别
            if orig_cls_name in REMOVE_CLASSES:
                continue

            # 重映射类别ID
            new_cls_id = CLASS_ID_MAP[orig_cls_name]
            # 保留边界框信息
            new_line = f"{new_cls_id} {parts[1]} {parts[2]} {parts[3]} {parts[4]}\n"
            keep_lines.append(new_line)

    # 写入新标签文件（仅当有有效标注时）
    if keep_lines:
        with open(dst_label_path, "w", encoding="utf-8") as f:
            f.writelines(keep_lines)
        return True
    return False


def process_split(split_name):
    """处理单个数据集划分（train/val/test）"""
    # 源目录
    src_img_dir = os.path.join(SRC_ROOT, "images", split_name)
    src_label_dir = os.path.join(SRC_ROOT, "labels", split_name)

    # 目标目录
    dst_img_dir = os.path.join(DST_ROOT, "images", split_name)
    dst_label_dir = os.path.join(DST_ROOT, "labels", split_name)

    # 获取所有图像文件
    img_files = [f for f in os.listdir(src_img_dir) if f.endswith(".jpg")]
    if not img_files:
        print(f"⚠️ {split_name}集无图像文件，跳过")
        return

    # 遍历处理每个图像和标签
    keep_count = 0
    remove_count = 0
    for img_name in tqdm(img_files, desc=f"处理{split_name}集"):
        # 图像路径
        src_img_path = os.path.join(src_img_dir, img_name)
        dst_img_path = os.path.join(dst_img_dir, img_name)

        # 标签路径
        label_name = img_name.replace(".jpg", ".txt")
        src_label_path = os.path.join(src_label_dir, label_name)
        dst_label_path = os.path.join(dst_label_dir, label_name)

        # 情况1：原标签文件不存在 → 跳过（无标注）
        if not os.path.exists(src_label_path):
            remove_count += 1
            continue

        # 情况2：过滤标签文件
        has_valid_label = filter_label_file(src_label_path, dst_label_path)

        # 有有效标注才复制图像
        if has_valid_label:
            shutil.copy2(src_img_path, dst_img_path)
            keep_count += 1
        else:
            remove_count += 1

    print(f"✅ {split_name}集处理完成：保留{keep_count}张，移除{remove_count}张")


def generate_new_yaml():
    """生成新的data.yaml配置文件"""
    # 绝对路径
    train_path = os.path.abspath(os.path.join(DST_ROOT, "images", "train"))
    val_path = os.path.abspath(os.path.join(DST_ROOT, "images", "val"))
    test_path = os.path.abspath(os.path.join(DST_ROOT, "images", "test"))

    yaml_content = {
        "names": KEEP_CLASSES,
        "nc": len(KEEP_CLASSES),
        "train": train_path,
        "val": val_path,
        "test": test_path
    }

    yaml_path = os.path.join(DST_ROOT, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(yaml_content, f, sort_keys=False, default_flow_style=False, allow_unicode=True)

    print(f"✅ 新配置文件生成：{yaml_path}")
    print(f"📋 配置文件内容：")
    print(f"   类别数(nc): {len(KEEP_CLASSES)}")
    print(f"   类别(names): {KEEP_CLASSES}")
    print(f"   训练集: {train_path}")
    print(f"   验证集: {val_path}")
    print(f"   测试集: {test_path}")


def visualize_filtered_samples(split="train", num=3):
    """可视化过滤后的样本"""
    img_dir = os.path.join(DST_ROOT, "images", split)
    label_dir = os.path.join(DST_ROOT, "labels", split)

    # 获取有标签的图像
    img_files = []
    for f in os.listdir(img_dir):
        if f.endswith(".jpg"):
            label_path = os.path.join(label_dir, f.replace(".jpg", ".txt"))
            if os.path.exists(label_path) and os.path.getsize(label_path) > 0:
                img_files.append(f)

    img_files = img_files[:num]
    if not img_files:
        print(f"⚠️ {split}集无有效样本，跳过可视化")
        return

    # 绘制可视化图
    import cv2
    plt.figure(figsize=(12, 4 * num))
    for i, img_name in enumerate(img_files):
        # 读取图像
        img_path = os.path.join(img_dir, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        h, w = img.shape[:2]

        # 读取并绘制标签
        label_path = os.path.join(label_dir, img_name.replace(".jpg", ".txt"))
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls_id, xc, yc, bw, bh = map(float, parts)
                # 转换为像素坐标
                x1 = int((xc - bw / 2) * w)
                y1 = int((yc - bh / 2) * h)
                x2 = int((xc + bw / 2) * w)
                y2 = int((yc + bh / 2) * h)
                # 绘制框和类别名
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(img, KEEP_CLASSES[int(cls_id)], (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # 显示图像
        plt.subplot(num, 1, i + 1)
        plt.imshow(img, cmap="gray")
        plt.title(f"过滤后{split}样本 {i + 1}: {img_name}")
        plt.axis("off")

    # 保存可视化图
    save_path = os.path.join(DST_ROOT, f"{split}_filtered_samples.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✅ {split}集可视化图保存：{save_path}")


def main():
    print("=" * 70)
    print("📋 NEU-DET数据集过滤（删除指定类别）")
    print(f"🚫 待删除类别：{REMOVE_CLASSES}")
    print(f"✅ 保留类别：{KEEP_CLASSES}")
    print("=" * 70)

    # 1. 检查源目录是否存在
    if not os.path.exists(SRC_ROOT):
        print(f"❌ 源数据集目录不存在：{SRC_ROOT}")
        print("请先运行原始预处理代码生成processed_neu文件夹")
        return

    # 2. 创建新目录
    create_new_dirs()

    # 3. 处理各数据集划分
    process_split("train")
    process_split("val")
    process_split("test")

    # 4. 生成新的配置文件
    generate_new_yaml()

    # 5. 可视化验证
    visualize_filtered_samples("train")
    visualize_filtered_samples("val")

    print("=" * 70)
    print("🎉 数据集过滤完成！")
    print(f"📁 过滤后数据集路径：{DST_ROOT}")
    print(f"📄 新配置文件：{os.path.join(DST_ROOT, 'data.yaml')}")
    print(f"📊 保留类别：{KEEP_CLASSES}（共{len(KEEP_CLASSES)}类）")
    print("=" * 70)


if __name__ == "__main__":
    # 自动安装依赖
    try:
        from tqdm import tqdm
        import yaml
        import cv2
        import matplotlib.pyplot as plt
    except ImportError:
        print("⚠️ 缺少依赖包，正在自动安装...")
        os.system("pip install tqdm pyyaml opencv-python matplotlib")
        print("✅ 依赖安装完成，请重启程序")
        exit()

    main()