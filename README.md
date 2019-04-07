# pointnet2-keras

## 介绍

本项目是基于[pointnet2](https://github.com/charlesq34/pointnet2)的一个衍生项目，主要作用是基于keras构建pointnet2的神经网络。

## 点云分类

### 数据集

下载[ModelNet40](https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip)数据，解压到本项目根目录的`data`文件夹下

### cuda库

您需要手动编译`tf_ops`文件夹下所有cuda代码。

具体方法：执行`tf_ops/grouping/tf_grouping_compile.sh`和`tf_ops/sampling/tf_sampling_compile.sh`

但是能执行成功的前提是您有正确安装cuda和cudnn，而且文件里面的路径需要您手动根据实际进行调整。

原作者书写的编译教程：<https://github.com/charlesq34/pointnet2#compile-customized-tf-operators>

### 训练

运行 `python tf_cls.py`

### 可视化

运行`tensorboard --logdir summary`

## 性能

1. 训练集精确率和验证集精确率

![训练集精确率和验证集精确率](https://raw.githubusercontent.com/HarborZeng/pointnet2-keras/master/result_image/model_accuracy.png)

2. 训练集损失和验证集损失率

![训练集损失和验证集损失率](https://raw.githubusercontent.com/HarborZeng/pointnet2-keras/master/result_image/model_loss.png)

## TODO

- [ ] part segmentation
- [ ] semantic scene

## 参考资料

1. TianzhongSong/PointNet-Keras https://github.com/TianzhongSong/PointNet-Keras
2. charlesq34/pointnet2 https://github.com/charlesq34/pointnet2