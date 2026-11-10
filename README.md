1![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/e9560ae7ccc94ee491b0f1ed59d3fa96.png)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/33890bbca6aa42d2b932ec7453a34a18.png)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/7ae5130e04354fdfb655b3b4b3dc2804.png)
| **项目** | **详情** |
| :--- | :--- |
| **数据集名称** | 夜间车辆检测数据集 |
| **图像总数** | 31,379 张 |
| **训练集** | 已划分（train） |
| **验证集** | 已划分（val） |
| **标注状态** | 已标注，标签与图片一一对应 |
| **标注工具** | LabelImg |
| **标注格式** | YOLO 格式（.txt） |
| **类别数量** | 1 类 |
| **类别名称** | car |
| **数据用途** | 夜间环境下车辆目标检测 |
| **适用模型** | YOLOv5, YOLOv7, YOLOv8 |
| **配套资料** | YOLOv8 模型（已训练 100 轮，可直接运行） |
| **使用条件** | 配置好 YOLO 环境后即可加载使用 |

以下是针对您提供的 **夜间车辆检测数据集** 的 **YOLOv8 详细训练与推理代码**，支持直接加载已划分的 `train/val` 数据集，使用 Ultralytics 官方框架，开箱即用。

---

## ✅ 一、环境安装

```bash
# 安装依赖
pip install ultralytics
# 或升级到最新版
pip install --upgrade ultralytics
```

> 官方文档：[https://docs.ultralytics.com](https://docs.ultralytics.com)

---

## ✅ 二、数据集目录结构

确保您的数据集结构如下：

```
night_vehicle_dataset/
├── images/
│   ├── train/      # 训练图像（约 28000+ 张）
│   └── val/        # 验证图像（约 3000+ 张）
├── labels/
│   ├── train/      # YOLO格式 .txt 标签文件
│   └── val/
└── data.yaml       # 数据集配置文件
```

---

## ✅ 三、创建 `data.yaml` 文件

在 `night_vehicle_dataset/` 目录下创建 `data.yaml`：

```yaml
# data.yaml

nc: 1                  # 类别数量
names: ['car']         # 类别名称

# 数据路径（相对路径）
train: ./night_vehicle_dataset/images/train
val: ./night_vehicle_dataset/images/val
```

---

## ✅ 四、完整训练代码（Python 脚本）

```python
from ultralytics import YOLO

# 1. 加载模型（可选择不同尺寸）
model = YOLO('yolov8s.pt')  # 推荐使用 yolov8s 或 yolov8m，平衡速度与精度

# 2. 开始训练
results = model.train(
    data='night_vehicle_dataset/data.yaml',  # 数据集配置文件
    epochs=100,                              # 训练轮数（您已有100轮模型）
    imgsz=640,                               # 输入图像大小
    batch=16,                                # 批次大小（根据GPU显存调整）
    device=0,                                # 使用 GPU 0，CPU 使用 None
    workers=4,                               # 数据加载线程数
    name='night_vehicle_yolov8s',            # 实验名称
    project='runs/train',                    # 结果保存路径
    exist_ok=False,                          # 禁止覆盖同名实验
    optimizer='AdamW',                       # 优化器
    lr0=0.01,                                # 初始学习率
    momentum=0.937,
    weight_decay=0.0005,
    patience=20,                             # 早停耐心值
    amp=True,                                # 混合精度训练（节省显存）
    
    # 数据增强（适用于夜间低光照场景）
    hsv_h=0.015,      # 色调扰动
    hsv_s=0.7,        # 饱和度增强（模拟灯光变化）
    hsv_v=0.5,        # 明度增强（增强暗区可见性）
    degrees=10.0,     # 随机旋转
    translate=0.2,
    scale=0.5,
    flipud=0.0,
    fliplr=0.5,       # 左右翻转
    mosaic=1.0,       # Mosaic 增强
    mixup=0.1,        # MixUp 增强
    copy_paste=0.3,   # Copy-Paste 增强（对小车有效）
)
```

---

## ✅ 五、验证与测试

### 1. 验证模型性能

```python
# 加载最佳模型
model = YOLO('runs/train/night_vehicle_yolov8s/weights/best.pt')

# 在验证集上评估
metrics = model.val()
print(f"mAP@0.5: {metrics.box.map50:.3f}")
print(f"mAP@0.5:0.95: {metrics.box.map:.3f}")
```

### 2. 测试单张图像

```python
# 推理单图
results = model('test_night.jpg', conf=0.5)
results[0].show()  # 显示结果
results[0].save(filename='result.jpg')
```

### 3. 批量推理或视频处理

```python
# 视频文件检测
model.predict(
    source='night_video.mp4',
    save=True,
    project='runs/detect',
    name='night_video_result',
    conf=0.5,
    imgsz=640
)

# 摄像头实时检测
model.predict(source=0, show=True, conf=0.5)
```

---

## ✅ 六、模型导出（用于部署）

```python
# 导出为 ONNX（通用推理）
model.export(format='onnx', dynamic=True, opset=13)

# 导出为 TensorRT（NVIDIA GPU 高性能）
model.export(format='engine', half=True)  # FP16 推理

# 导出为 TFLite（移动端）
model.export(format='tflite')

# 导出为 CoreML（iOS）
model.export(format='coreml')
```

---

## ✅ 七、训练结果输出

| 文件/目录 | 说明 |
|---------|------|
| `runs/train/night_vehicle_yolov8s/weights/best.pt` | 最佳模型权重 |
| `runs/train/night_vehicle_yolov8s/weights/last.pt` | 最终模型权重 |
| `results.png` | 训练指标曲线（loss, mAP, precision, recall） |
| `confusion_matrix.png` | 混淆矩阵 |
| `labels/*.jpg` | 带标注的可视化图 |


