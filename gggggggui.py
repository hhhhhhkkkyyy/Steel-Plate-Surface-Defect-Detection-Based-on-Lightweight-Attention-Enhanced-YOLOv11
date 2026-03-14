# import cv2
# import tkinter as tk
# from tkinter import ttk, filedialog, messagebox
# from PIL import Image, ImageTk
# import numpy as np
# from ultralytics import YOLO
# import threading
# import time
#
# # ===================== 固定配置 =====================
# MODEL_PATH = r"D:\666\PyCharm 2023.3.2\pythonProject\123\neu_train_results\yolov11m_neu_4classes_opt\weights\best.pt"
# IMG_SIZE = 320
#
# # 不同缺陷对应颜色（BGR）
# COLOR_MAP = {
#     "inclusion": (255, 0, 0),
#     "crazing": (0, 255, 0),
#     "patches": (0, 0, 255),
#     "pitted_surface": (0, 255, 255),
#     "rolled-in_scale": (255, 255, 0),
#     "scratches": (255, 0, 255)
# }
#
#
# # ===================== 核心绘制函数 =====================
# def draw_boxes_with_id(image, results):
#     img = image.copy()
#     infos = []
#
#     if results.boxes is None:
#         return img, infos
#
#     names = results.names
#
#     for idx, box in enumerate(results.boxes):
#         cls_id = int(box.cls.item())
#         conf = float(box.conf.item())
#         x1, y1, x2, y2 = map(int, box.xyxy[0])
#         label = names[cls_id]
#
#         area = (x2 - x1) * (y2 - y1)
#         color = COLOR_MAP.get(label, (200, 200, 200))
#
#         # 画框
#         cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
#
#         # 左上角编号
#         cv2.putText(
#             img, str(idx + 1),
#             (x1 + 3, y1 + 18),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.6, color, 2, cv2.LINE_AA
#         )
#
#         infos.append(
#             f"{idx + 1} | {label:<14} | conf={conf:.2f} | area={area}px"
#         )
#
#     return img, infos
#
#
# # ===================== GUI 主类 =====================
# class YOLOIndustrialGUI:
#     def __init__(self, root):
#         self.root = root
#         self.root.title("钢材表面缺陷检测系统（工业版 YOLOv11）")
#         self.root.geometry("1500x850")
#
#         # 核心变量
#         self.model = YOLO(MODEL_PATH)
#         self.cap = None
#         self.running = False
#         self.last_image = None  # 原始图片
#         self.current_image = None  # 显示的图片（原始/检测后）
#
#         # 置信度变量
#         self.conf_var = tk.DoubleVar(value=0.25)
#
#         # 构建UI
#         self.build_ui()
#
#     # ===================== UI构建 =====================
#     def build_ui(self):
#         # 顶部标题栏
#         top = ttk.Frame(self.root)
#         top.pack(fill=tk.X, pady=10)
#
#         ttk.Label(
#             top, text="YOLOv11 钢材表面缺陷检测（工业可视化）",
#             font=("微软雅黑", 18, "bold")
#         ).pack(side=tk.LEFT, padx=20)
#
#         # 主内容区
#         main = ttk.Frame(self.root)
#         main.pack(fill=tk.BOTH, expand=True)
#
#         # ===== 左侧控制区 =====
#         left = ttk.Frame(main, width=260)
#         left.pack(side=tk.LEFT, fill=tk.Y, padx=10)
#         left.pack_propagate(False)
#
#         # 控制区标题
#         ttk.Label(left, text="检测控制", font=("微软雅黑", 14, "bold")).pack(pady=15)
#
#         # 功能按钮
#         ttk.Button(left, text="📁 选择图片", command=self.open_image).pack(fill=tk.X, pady=6)
#         ttk.Button(left, text="▶ 开始检测", command=self.start_detection).pack(fill=tk.X, pady=6)  # 新增开始检测按钮
#         ttk.Button(left, text="📷 打开摄像头", command=self.start_camera).pack(fill=tk.X, pady=6)
#         ttk.Button(left, text="⛔ 停止检测", command=self.stop_camera).pack(fill=tk.X, pady=6)
#
#         # 置信度阈值（带数值显示）
#         conf_frame = ttk.Frame(left)
#         conf_frame.pack(fill=tk.X, pady=10)
#
#         ttk.Label(conf_frame, text="置信度阈值").pack(side=tk.LEFT)
#
#         self.conf_value_label = ttk.Label(
#             conf_frame,
#             text=f"{self.conf_var.get():.2f}",
#             width=6,
#             anchor="center",
#             font=("Consolas", 11, "bold")
#         )
#         self.conf_value_label.pack(side=tk.RIGHT)
#
#         self.conf_scale = ttk.Scale(
#             left,
#             from_=0.05,
#             to=0.90,
#             variable=self.conf_var,
#             orient=tk.HORIZONTAL,
#             command=self.update_conf_label
#         )
#         self.conf_scale.pack(fill=tk.X)
#
#         # 保存按钮
#         ttk.Button(left, text="💾 保存检测结果", command=self.save_image).pack(fill=tk.X, pady=20)
#
#         # ===== 中间图像区 =====
#         center = ttk.Frame(main)
#         center.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
#
#         self.image_label = ttk.Label(
#             center, text="请选择图片或打开摄像头",
#             background="#dcdcdc", anchor="center"
#         )
#         self.image_label.pack(fill=tk.BOTH, expand=True)
#
#         # ===== 右侧信息面板 =====
#         right = ttk.Frame(main, width=380)
#         right.pack(side=tk.RIGHT, fill=tk.Y, padx=10)
#         right.pack_propagate(False)
#
#         ttk.Label(right, text="检测结果信息", font=("微软雅黑", 14, "bold")).pack(pady=10)
#
#         self.info_box = tk.Listbox(
#             right, font=("Consolas", 11),
#             height=30
#         )
#         self.info_box.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
#
#         # 底部状态栏
#         self.status_var = tk.StringVar(value="系统就绪")
#         ttk.Label(
#             self.root, textvariable=self.status_var,
#             relief=tk.SUNKEN, anchor=tk.W
#         ).pack(fill=tk.X, side=tk.BOTTOM)
#
#     # ===================== 置信度更新 =====================
#     def update_conf_label(self, val):
#         """更新置信度显示标签"""
#         self.conf_value_label.config(text=f"{float(val):.2f}")
#
#     # ===================== 图片选择 =====================
#     def open_image(self):
#         """选择图片并显示原始图片"""
#         path = filedialog.askopenfilename(
#             filetypes=[("Images", "*.jpg *.png *.bmp *.jpeg")]
#         )
#         if not path:
#             return
#
#         # 读取图片
#         self.last_image = cv2.imread(path)
#         if self.last_image is None:
#             messagebox.showerror("错误", "无法读取选择的图片")
#             return
#
#         # 显示原始图片
#         self.current_image = self.last_image.copy()
#         self.show_image(self.current_image)
#
#         # 清空之前的检测信息
#         self.info_box.delete(0, tk.END)
#
#         # 更新状态
#         self.status_var.set(f"已选择图片 | 等待检测（当前置信度：{self.conf_var.get():.2f}）")
#
#     # ===================== 开始检测（核心新增功能） =====================
#     def start_detection(self):
#         """使用当前置信度重新检测图片"""
#         # 检查是否已选择图片
#         if self.last_image is None:
#             messagebox.showwarning("提示", "请先选择图片！")
#             return
#
#         # 检查是否在摄像头模式
#         if self.running:
#             messagebox.showwarning("提示", "请先停止摄像头检测！")
#             return
#
#         # 执行检测
#         try:
#             result = self.model.predict(
#                 self.last_image,
#                 imgsz=IMG_SIZE,
#                 conf=self.conf_var.get(),
#                 verbose=False
#             )[0]
#
#             # 绘制检测框
#             img_drawn, infos = draw_boxes_with_id(self.last_image, result)
#
#             # 更新显示（覆盖原图）
#             self.current_image = img_drawn
#             self.show_image(img_drawn)
#
#             # 更新检测信息
#             self.update_info(infos)
#
#             # 更新状态
#             self.status_var.set(f"检测完成 | 置信度：{self.conf_var.get():.2f} | 共 {len(infos)} 个缺陷")
#
#         except Exception as e:
#             messagebox.showerror("检测错误", f"检测过程出错：{str(e)}")
#             self.status_var.set(f"检测失败：{str(e)}")
#
#     # ===================== 摄像头检测 =====================
#     def start_camera(self):
#         """打开摄像头实时检测"""
#         if self.running:
#             return
#
#         self.cap = cv2.VideoCapture(0)
#         if not self.cap.isOpened():
#             messagebox.showerror("错误", "无法打开摄像头")
#             return
#
#         self.running = True
#         # 启动摄像头检测线程
#         threading.Thread(target=self.camera_loop, daemon=True).start()
#         self.status_var.set("摄像头检测中 | 按停止按钮结束")
#
#     def camera_loop(self):
#         """摄像头检测循环"""
#         while self.running:
#             ret, frame = self.cap.read()
#             if not ret:
#                 break
#
#             # 实时检测（使用当前置信度）
#             result = self.model.predict(
#                 frame,
#                 imgsz=IMG_SIZE,
#                 conf=self.conf_var.get(),
#                 verbose=False
#             )[0]
#
#             # 绘制检测框
#             img_drawn, infos = draw_boxes_with_id(frame, result)
#
#             # 更新显示和信息
#             self.current_image = img_drawn
#             self.show_image(img_drawn)
#             self.update_info(infos)
#
#             time.sleep(0.03)
#
#         # 释放摄像头
#         self.cap.release()
#         self.status_var.set("摄像头检测已停止")
#
#     def stop_camera(self):
#         """停止摄像头检测"""
#         self.running = False
#         self.status_var.set("摄像头检测已停止")
#
#     # ===================== 显示与信息更新 =====================
#     def show_image(self, img_bgr):
#         """显示图片到界面（等比例缩放）"""
#         # 转换颜色空间
#         img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
#         h, w = img_rgb.shape[:2]
#
#         # 获取显示区域尺寸
#         lw = self.image_label.winfo_width()
#         lh = self.image_label.winfo_height()
#
#         # 避免初始尺寸为0
#         if lw <= 10 or lh <= 10:
#             lw = 800
#             lh = 600
#
#         # 等比例缩放
#         scale = min(lw / w, lh / h)
#         new_w = int(w * scale)
#         new_h = int(h * scale)
#
#         # 缩放图片
#         resized = cv2.resize(img_rgb, (new_w, new_h), cv2.INTER_AREA)
#         # 转换为TK显示格式
#         imgtk = ImageTk.PhotoImage(Image.fromarray(resized))
#
#         # 更新显示
#         self.image_label.config(image=imgtk, text="")
#         self.image_label.image = imgtk
#
#     def update_info(self, infos):
#         """更新检测信息列表"""
#         self.info_box.delete(0, tk.END)
#         if not infos:
#             self.info_box.insert(tk.END, "未检测到任何缺陷")
#             return
#         for line in infos:
#             self.info_box.insert(tk.END, line)
#
#     # ===================== 保存检测结果 =====================
#     def save_image(self):
#         """保存当前显示的图片（检测后）"""
#         if self.current_image is None:
#             messagebox.showwarning("提示", "没有可保存的图片！")
#             return
#
#         path = filedialog.asksaveasfilename(
#             defaultextension=".jpg",
#             filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")]
#         )
#         if path:
#             cv2.imwrite(path, self.current_image)
#             messagebox.showinfo("成功", "检测结果已保存")
#             self.status_var.set(f"检测结果已保存至：{path}")
#
#
# # ===================== 主入口 =====================
# if __name__ == "__main__":
#     root = tk.Tk()
#     app = YOLOIndustrialGUI(root)
#     root.mainloop()
#
#
import cv2
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
from ultralytics import YOLO
import threading
import time
import os

# ===================== 固定配置 =====================
MODEL_PATH = r"D:\666\PyCharm 2023.3.2\pythonProject\123\neu_train_results\yolov11m_neu_4classes_opt\weights\best.pt"
IMG_SIZE = 320

# 不同缺陷对应颜色（BGR）
COLOR_MAP = {
    "inclusion": (255, 0, 0),
    "crazing": (0, 255, 0),
    "patches": (0, 0, 255),
    "pitted_surface": (0, 255, 255),
    "rolled-in_scale": (255, 255, 0),
    "scratches": (255, 0, 255)
}

# ===================== 缺陷库配置 =====================
DEFECT_LIBRARY = {
    "crazing": {
        "cn_name": "裂纹（Crazing）",
        "desc": "钢材表面呈线状、条状裂纹，部分有分支，焊接处或受力部位易出现",
        "solution": "焊接前清理母材；控制焊接参数；渗透检测排查",
        "img_path": r"D:\666\PyCharm 2023.3.2\pythonProject\123\IMAGES\crazing_9.jpg"
    },
    "inclusion": {
        "cn_name": "夹杂（Inclusion）",
        "desc": "非金属或金属夹杂物，射线检测呈黑点/白点",
        "solution": "优化冶炼工艺；焊接后清理熔渣；射线检测筛选",
        "img_path": r"D:\666\PyCharm 2023.3.2\pythonProject\123\IMAGES\inclusion_87.jpg"
    },
    "patches": {
        "cn_name": "补丁（Patches）",
        "desc": "缺陷修复区域，色差明显，边缘有打磨痕迹",
        "solution": "修复时贴合母材；打磨平整并防腐",
        "img_path": r"D:\666\PyCharm 2023.3.2\pythonProject\123\IMAGES\patches_271.jpg"
    },
    "pitted_surface": {
        "cn_name": "点蚀表面（Pitted Surface）",
        "desc": "分散微小蚀坑，不锈钢在氯化物环境易发",
        "solution": "选用耐蚀材质；避免氯化物；定期防腐",
        "img_path": r"D:\666\PyCharm 2023.3.2\pythonProject\123\IMAGES\pitted_surface_166.jpg"
    },
    "rolled-in_scale": {
        "cn_name": "轧入氧化皮（Rolled-in Scale）",
        "desc": "热轧时氧化铁皮压入，灰黑色片状附着",
        "solution": "热轧前除鳞；优化轧制；酸洗处理",
        "img_path": r"D:\666\PyCharm 2023.3.2\pythonProject\123\IMAGES\rolled-in_scale_138.jpg"
    },
    "scratches": {
        "cn_name": "划痕（Scratch）",
        "desc": "线状沟痕，高温下呈灰黑色，常温保留金属光泽",
        "solution": "辊道清洁；避免异物；轻微打磨",
        "img_path": r"D:\666\PyCharm 2023.3.2\pythonProject\123\IMAGES\scratches_85.jpg"
    }
}


# ===================== 核心绘制函数 =====================
def draw_boxes_with_id(image, results):
    img = image.copy()
    infos = []
    defect_details = []  # 存储缺陷详细信息，用于后续关联缺陷库

    if results.boxes is None:
        return img, infos, defect_details

    names = results.names

    for idx, box in enumerate(results.boxes):
        cls_id = int(box.cls.item())
        conf = float(box.conf.item())
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = names[cls_id]

        area = (x2 - x1) * (y2 - y1)
        color = COLOR_MAP.get(label, (200, 200, 200))

        # 画框
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # 左上角编号
        cv2.putText(
            img, str(idx + 1),
            (x1 + 3, y1 + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6, color, 2, cv2.LINE_AA
        )

        infos.append(
            f"{idx + 1} | {label:<14} | conf={conf:.2f} | area={area}px"
        )
        defect_details.append({
            "id": idx + 1,
            "label": label,
            "conf": conf,
            "area": area
        })

    return img, infos, defect_details


# ===================== GUI 主类 =====================
class YOLOIndustrialGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("钢材表面缺陷检测系统（工业版 YOLOv11）")
        self.root.geometry("1600x900")

        # 核心变量
        self.model = YOLO(MODEL_PATH)
        self.cap = None
        self.running = False
        self.last_image = None  # 原始图片
        self.current_image = None  # 显示的图片（原始/检测后）
        self.current_defect_details = []  # 当前检测的缺陷详细信息

        # 置信度变量
        self.conf_var = tk.DoubleVar(value=0.25)

        # 构建UI
        self.build_ui()

    # ===================== UI构建 =====================
    def build_ui(self):
        # 顶部标题栏
        top = ttk.Frame(self.root)
        top.pack(fill=tk.X, pady=10)

        ttk.Label(
            top, text="YOLOv11 钢材表面缺陷检测系统（工业可视化）",
            font=("微软雅黑", 18, "bold")
        ).pack(side=tk.LEFT, padx=20)

        # 主内容区
        main = ttk.Frame(self.root)
        main.pack(fill=tk.BOTH, expand=True)

        # ===== 左侧控制区 =====
        left = ttk.Frame(main, width=260)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=10)
        left.pack_propagate(False)

        # 控制区标题
        ttk.Label(left, text="检测控制", font=("微软雅黑", 14, "bold")).pack(pady=15)

        # 功能按钮
        ttk.Button(left, text="📁 选择图片", command=self.open_image).pack(fill=tk.X, pady=6)
        ttk.Button(left, text="▶ 开始检测", command=self.start_detection).pack(fill=tk.X, pady=6)
        ttk.Button(left, text="📷 打开摄像头", command=self.start_camera).pack(fill=tk.X, pady=6)
        ttk.Button(left, text="⛔ 停止检测", command=self.stop_camera).pack(fill=tk.X, pady=6)

        # 置信度阈值（带数值显示）
        conf_frame = ttk.Frame(left)
        conf_frame.pack(fill=tk.X, pady=10)

        ttk.Label(conf_frame, text="置信度阈值").pack(side=tk.LEFT)

        self.conf_value_label = ttk.Label(
            conf_frame,
            text=f"{self.conf_var.get():.2f}",
            width=6,
            anchor="center",
            font=("Consolas", 11, "bold")
        )
        self.conf_value_label.pack(side=tk.RIGHT)

        self.conf_scale = ttk.Scale(
            left,
            from_=0.05,
            to=0.90,
            variable=self.conf_var,
            orient=tk.HORIZONTAL,
            command=self.update_conf_label
        )
        self.conf_scale.pack(fill=tk.X)

        # 保存按钮
        ttk.Button(left, text="💾 保存检测结果", command=self.save_image).pack(fill=tk.X, pady=20)

        # ===== 中间图像区 =====
        center = ttk.Frame(main)
        center.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        self.image_label = ttk.Label(
            center, text="请选择图片或打开摄像头",
            background="#dcdcdc", anchor="center"
        )
        self.image_label.pack(fill=tk.BOTH, expand=True)

        # ===== 右侧信息区（拆分：检测结果 + 缺陷详情）=====
        right = ttk.Frame(main, width=700)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=10)
        right.pack_propagate(False)

        # 检测结果列表 - 修复ttk.LabelFrame不支持font参数的问题
        result_frame = ttk.LabelFrame(right, text="检测结果列表")
        result_frame.pack(fill=tk.X, pady=5)
        # 单独添加标题标签来设置字体
        result_title = ttk.Label(result_frame, text="检测结果列表", font=("微软雅黑", 12, "bold"))
        result_title.pack(anchor="nw", padx=5, pady=2)

        self.info_box = tk.Listbox(
            result_frame, font=("Consolas", 11),
            height=10
        )
        self.info_box.pack(fill=tk.X, padx=5, pady=5)
        # 绑定列表点击事件
        self.info_box.bind('<<ListboxSelect>>', self.show_defect_detail)

        # 缺陷详情面板 - 修复ttk.LabelFrame不支持font参数的问题
        detail_frame = ttk.LabelFrame(right, text="缺陷详情（点击列表项查看）")
        detail_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        # 单独添加标题标签来设置字体
        detail_title = ttk.Label(detail_frame, text="缺陷详情（点击列表项查看）", font=("微软雅黑", 12, "bold"))
        detail_title.pack(anchor="nw", padx=5, pady=2)

        # 缺陷详情布局：左侧文字信息，右侧参考图片
        detail_inner = ttk.Frame(detail_frame)
        detail_inner.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 左侧文字信息
        text_frame = ttk.Frame(detail_inner, width=300)
        text_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5)
        text_frame.pack_propagate(False)

        self.defect_name_label = ttk.Label(text_frame, text="缺陷名称：--", font=("微软雅黑", 12, "bold"))
        self.defect_name_label.pack(anchor="w", pady=5)

        self.defect_desc_label = ttk.Label(text_frame, text="缺陷描述：--", wraplength=280, justify="left")
        self.defect_desc_label.pack(anchor="w", pady=5)

        self.defect_solution_label = ttk.Label(text_frame, text="解决方案：--", wraplength=280, justify="left")
        self.defect_solution_label.pack(anchor="w", pady=5)

        # 右侧参考图片
        img_frame = ttk.Frame(detail_inner)
        img_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)

        self.defect_img_label = ttk.Label(img_frame, text="参考图片", background="#dcdcdc")
        self.defect_img_label.pack(fill=tk.BOTH, expand=True)

        # 底部状态栏
        self.status_var = tk.StringVar(value="系统就绪")
        ttk.Label(
            self.root, textvariable=self.status_var,
            relief=tk.SUNKEN, anchor=tk.W
        ).pack(fill=tk.X, side=tk.BOTTOM)

    # ===================== 置信度更新 =====================
    def update_conf_label(self, val):
        """更新置信度显示标签"""
        self.conf_value_label.config(text=f"{float(val):.2f}")

    # ===================== 图片选择 =====================
    def open_image(self):
        """选择图片并显示原始图片"""
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.jpg *.png *.bmp *.jpeg")]
        )
        if not path:
            return

        # 读取图片
        self.last_image = cv2.imread(path)
        if self.last_image is None:
            messagebox.showerror("错误", "无法读取选择的图片")
            return

        # 显示原始图片
        self.current_image = self.last_image.copy()
        self.show_image(self.current_image)

        # 清空之前的检测信息
        self.info_box.delete(0, tk.END)
        self.clear_defect_detail()

        # 更新状态
        self.status_var.set(f"已选择图片 | 等待检测（当前置信度：{self.conf_var.get():.2f}）")

    # ===================== 开始检测 =====================
    def start_detection(self):
        """使用当前置信度重新检测图片"""
        # 检查是否已选择图片
        if self.last_image is None:
            messagebox.showwarning("提示", "请先选择图片！")
            return

        # 检查是否在摄像头模式
        if self.running:
            messagebox.showwarning("提示", "请先停止摄像头检测！")
            return

        # 执行检测
        try:
            result = self.model.predict(
                self.last_image,
                imgsz=IMG_SIZE,
                conf=self.conf_var.get(),
                verbose=False
            )[0]

            # 绘制检测框
            img_drawn, infos, defect_details = draw_boxes_with_id(self.last_image, result)

            # 更新显示（覆盖原图）
            self.current_image = img_drawn
            self.show_image(img_drawn)

            # 更新检测信息
            self.update_info(infos)
            self.current_defect_details = defect_details

            # 更新状态
            self.status_var.set(f"检测完成 | 置信度：{self.conf_var.get():.2f} | 共 {len(infos)} 个缺陷")

        except Exception as e:
            messagebox.showerror("检测错误", f"检测过程出错：{str(e)}")
            self.status_var.set(f"检测失败：{str(e)}")

    # ===================== 摄像头检测 =====================
    def start_camera(self):
        """打开摄像头实时检测"""
        if self.running:
            return

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("错误", "无法打开摄像头")
            return

        self.running = True
        # 启动摄像头检测线程
        threading.Thread(target=self.camera_loop, daemon=True).start()
        self.status_var.set("摄像头检测中 | 按停止按钮结束")

    def camera_loop(self):
        """摄像头检测循环"""
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break

            # 实时检测（使用当前置信度）
            result = self.model.predict(
                frame,
                imgsz=IMG_SIZE,
                conf=self.conf_var.get(),
                verbose=False
            )[0]

            # 绘制检测框
            img_drawn, infos, defect_details = draw_boxes_with_id(frame, result)

            # 更新显示和信息
            self.current_image = img_drawn
            self.show_image(img_drawn)
            self.update_info(infos)
            self.current_defect_details = defect_details

            time.sleep(0.03)

        # 释放摄像头
        self.cap.release()
        self.status_var.set("摄像头检测已停止")

    def stop_camera(self):
        """停止摄像头检测"""
        self.running = False
        self.status_var.set("摄像头检测已停止")

    # ===================== 显示与信息更新 =====================
    def show_image(self, img_bgr):
        """显示图片到界面（等比例缩放）"""
        # 转换颜色空间
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]

        # 获取显示区域尺寸
        lw = self.image_label.winfo_width()
        lh = self.image_label.winfo_height()

        # 避免初始尺寸为0
        if lw <= 10 or lh <= 10:
            lw = 800
            lh = 600

        # 等比例缩放
        scale = min(lw / w, lh / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        # 缩放图片
        resized = cv2.resize(img_rgb, (new_w, new_h), cv2.INTER_AREA)
        # 转换为TK显示格式
        imgtk = ImageTk.PhotoImage(Image.fromarray(resized))

        # 更新显示
        self.image_label.config(image=imgtk, text="")
        self.image_label.image = imgtk

    def update_info(self, infos):
        """更新检测信息列表"""
        self.info_box.delete(0, tk.END)
        if not infos:
            self.info_box.insert(tk.END, "未检测到任何缺陷")
            self.clear_defect_detail()
            return
        for line in infos:
            self.info_box.insert(tk.END, line)

    def clear_defect_detail(self):
        """清空缺陷详情面板"""
        self.defect_name_label.config(text="缺陷名称：--")
        self.defect_desc_label.config(text="缺陷描述：--")
        self.defect_solution_label.config(text="解决方案：--")
        self.defect_img_label.config(image="", text="参考图片")
        self.defect_img_label.image = None

    def show_defect_detail(self, event):
        """显示选中缺陷的详细信息"""
        # 获取选中的索引
        selected_indices = self.info_box.curselection()
        if not selected_indices:
            return
        idx = selected_indices[0]

        # 检查是否有缺陷详情数据
        if idx >= len(self.current_defect_details) or self.current_defect_details == []:
            return

        # 获取选中缺陷的信息
        defect_info = self.current_defect_details[idx]
        defect_label = defect_info["label"]

        # 从缺陷库获取详细信息
        if defect_label not in DEFECT_LIBRARY:
            self.defect_name_label.config(text=f"缺陷名称：{defect_label}（未知缺陷）")
            self.defect_desc_label.config(text="缺陷描述：无相关信息")
            self.defect_solution_label.config(text="解决方案：无相关信息")
            self.defect_img_label.config(image="", text="无参考图片")
            self.defect_img_label.image = None
            return

        # 更新文字信息
        library_info = DEFECT_LIBRARY[defect_label]
        self.defect_name_label.config(text=f"缺陷名称：{library_info['cn_name']}")
        self.defect_desc_label.config(text=f"缺陷描述：{library_info['desc']}")
        self.defect_solution_label.config(text=f"解决方案：{library_info['solution']}")

        # 显示参考图片
        img_path = library_info["img_path"]
        if os.path.exists(img_path):
            # 读取并缩放参考图片
            img = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # 获取显示区域尺寸
            lw = self.defect_img_label.winfo_width()
            lh = self.defect_img_label.winfo_height()
            if lw <= 10 or lh <= 10:
                lw = 300
                lh = 200

            # 等比例缩放
            h, w = img_rgb.shape[:2]
            scale = min(lw / w, lh / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            resized = cv2.resize(img_rgb, (new_w, new_h), cv2.INTER_AREA)

            # 转换为TK格式
            imgtk = ImageTk.PhotoImage(Image.fromarray(resized))
            self.defect_img_label.config(image=imgtk, text="")
            self.defect_img_label.image = imgtk
        else:
            self.defect_img_label.config(image="", text="参考图片不存在")
            self.defect_img_label.image = None

    # ===================== 保存检测结果 =====================
    def save_image(self):
        """保存当前显示的图片（检测后）"""
        if self.current_image is None:
            messagebox.showwarning("提示", "没有可保存的图片！")
            return

        path = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")]
        )
        if path:
            cv2.imwrite(path, self.current_image)
            messagebox.showinfo("成功", "检测结果已保存")
            self.status_var.set(f"检测结果已保存至：{path}")


# ===================== 主入口 =====================
if __name__ == "__main__":
    # 高DPI适配（可选）
    try:
        from ctypes import windll

        windll.shcore.SetProcessDpiAwareness(1)
    except:
        pass

    root = tk.Tk()
    app = YOLOIndustrialGUI(root)
    root.mainloop()