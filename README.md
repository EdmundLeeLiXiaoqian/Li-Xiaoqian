# PaddleX实现目标检测baseline

在地铁口实现无人口罩检测功能

# 一、项目背景
因为我发现在地铁入口的检测人员十分辛苦，工作量大，有时会用疏漏，有时候还会遇到很多不配合之人。加上口罩检测本就让很拥挤的地铁入口人满为患，所以我就想到可以做一个无人口罩检测系统去在地铁口实现无人口罩检测功能
目标应用场景
    地铁口
实现的功能
    实现无人口罩检测
解决的问题
    减少接触、降低传播病毒的风险
创意的价值
    1. 创新价值：对原有产品体验做出改变，不仅仅是地铁口，人流密集的地方都需要进行口罩检测
    2. 业务价值：减少接触、降低传播病毒的风险的同时，降低了人工成本
    3. 公益价值：帮助人们建立佩戴口罩的防范意识

# 二、数据集简介

本项目使用的数据集是：[AI训练营]目标检测数据集合集，包含口罩识别 、交通标志识别、火焰检测、锥桶识别以及中秋元素识别。

该数据集已加载至本环境中，位于：data/data103743/objDataset.zip

## 1.数据加载和预处理


# 解压数据集（解压一次即可，请勿重复解压）
!unzip -oq /home/aistudio/data/data103743/objDataset.zip

from paddlex.det import transforms

# 定义训练和验证时的transforms
# API说明 https://paddlex.readthedocs.io/zh_CN/develop/apis/transforms/det_transforms.html
train_transforms = transforms.Compose([
    # 此处需要补充图像预处理代码
    transforms.Normalize(),
])

eval_transforms = transforms.Compose([
    # 此处需要补充图像预处理代码
    transforms.Normalize(),
])

# 读取PascalVOC格式的检测数据集，并对样本进行相应的处理。
import paddlex as pdx

# 定义训练和验证所用的数据集
# API说明：https://paddlex.readthedocs.io/zh_CN/develop/apis/datasets.html#paddlex-datasets-vocdetection
train_dataset = pdx.datasets.VOCDetection(
    data_dir='objDataset/facemask',
    file_list='objDataset/facemask/train_list.txt',
    label_list='objDataset/facemask/labels.txt',
    transforms=train_transforms,
    shuffle=True)

eval_dataset = pdx.datasets.VOCDetection(
    data_dir='objDataset/facemask',
    file_list='objDataset/facemask/val_list.txt',
    label_list='objDataset/facemask/labels.txt',
    transforms=eval_transforms)
/opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/matplotlib/__init__.py:107: DeprecationWarning: Using or importing the ABCs from 'collections' instead of from 'collections.abc' is deprecated, and in 3.8 it will stop working
  from collections import MutableMapping
/opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/matplotlib/rcsetup.py:20: DeprecationWarning: Using or importing the ABCs from 'collections' instead of from 'collections.abc' is deprecated, and in 3.8 it will stop working
  from collections import Iterable, Mapping
/opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/matplotlib/colors.py:53: DeprecationWarning: Using or importing the ABCs from 'collections' instead of from 'collections.abc' is deprecated, and in 3.8 it will stop working
  from collections import Sized
2021-08-15 21:07:51 [INFO]	Starting to read file list from dataset...
2021-08-15 21:07:53 [INFO]	598 samples in file objDataset/facemask/train_list.txt
creating index...
index created!
2021-08-15 21:07:53 [INFO]	Starting to read file list from dataset...
2021-08-15 21:07:53 [INFO]	170 samples in file objDataset/facemask/val_list.txt
creating index...
index created!


## 2.数据集查看


!tree objDataset -L 2
objDataset
├── barricade
│   ├── Annotations
│   ├── JPEGImages
│   ├── labels.txt
│   ├── test_list.txt
│   ├── train_list.txt
│   └── val_list.txt
├── facemask
│   ├── Annotations
│   ├── JPEGImages
│   ├── labels.txt
│   ├── test_list.txt
│   ├── train_list.txt
│   └── val_list.txt
├── fire
│   ├── Annotations
│   └── JPEGImages
├── MidAutumn
│   ├── Annotations
│   └── JPEGImages
└── roadsign_voc
    ├── Annotations
    └── JPEGImages

15 directories, 8 files


# 三、模型选择和开发

本基线系统以骨干网络为MobileNetV1的YOLOv3算法

## 初始化模型
# API说明: https://paddlex.readthedocs.io/zh_CN/develop/apis/models/detection.html#paddlex-det-yolov3

# 此处需要补充目标检测模型代码
model = pdx.det.YOLOv3(num_classes=len(train_dataset.labels), backbone='MobileNetV1')
In [8]
## 模型训练
# API说明: https://paddlex.readthedocs.io/zh_CN/develop/apis/models/detection.html#id1
# 各参数介绍与调整说明：https://paddlex.readthedocs.io/zh_CN/develop/appendix/parameters.html

# 此处需要补充模型训练参数
model.train(
    num_epochs=270,
    train_dataset=train_dataset,
    train_batch_size=8,
    eval_dataset=eval_dataset,
    learning_rate=0.000125,
    lr_decay_epochs=[210, 240],
    save_dir='output/yolov3_mobilenetv1')
    



# 五、总结与升华

遇到困难就是刚开始运行代码提示我内存不够，最后使用算例升级了一下版本就可以做出了了
我做的项目主要可以在地铁口无人识别口罩，实现人力资源的成分利用，提高生活工作质量

# 个人简介

我在AI Studio上获得青铜等级，点亮1个徽章，来互关呀~ https://aistudio.baidu.com/aistudio/personalcenter/thirdview/885368
