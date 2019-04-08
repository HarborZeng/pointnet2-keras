# pointnet2-keras

> For English readme, please open an issue. I may consider to create one. If I am not, you can just use chrome to translate.

## 介绍

本项目是基于 [pointnet2](https://github.com/charlesq34/pointnet2) 的一个衍生项目，主要作用是基于 keras 构建 pointnet2 的神经网络。

由于 pointnet2 使用了 cuda 和 c++ 编写的代码，需要使用者自行编译动态链接库，在 python 代码里面使用`tf.load_op_library('some_opration.so'))`，而这些 op 是对 tf tensor 的一些操作，所以不可能使用keras的`Lambda`层来包裹这些op（其他非 so 文件的操作是可以包裹的，如：`Lambda(lambda x: K.concatenate(x, axis=-1))([grouped_points, grouped_xyz])`），使得最终使用 keras 构建的`logits`或叫`pred`或叫`prediction`不能使用`Model(input=some_input, output=logits)`来构建 model 实例，也就不能使用 keras 高阶 API ，如`model.fit(some_params)`或`model.fit_generator(another_params)`

综上所述，本项目使用keras仅仅使用其**创建层**的能力，也就是说，层与层之间，传递的还是 tensorflow 的 tensor，最终也只能使用 tensorflow 的 API 来进行训练。

本项目尽可能多的，使用了 keras 的 API:

```python
from keras import backend as K

K.set_session(sess)
K.max(x, axis=2)
K.permute_dimensions(x, [0, 2, 3, 1])
K.concatenate(x, axis=-1)
K.placeholder(dtype=np.float32, shape=(batch_size, num_point, 3))
```

等等。

## 点云分类

### 数据集

下载 [ModelNet40](https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip) 数据，解压到本项目根目录的`data`文件夹下

### cuda库

您需要手动编译`tf_ops`文件夹下所有cuda代码。

具体方法：执行`tf_ops/grouping/tf_grouping_compile.sh`和`tf_ops/sampling/tf_sampling_compile.sh`

如：

```shell
$ cd tf_ops/grouping
$ ./tf_grouping_compile.sh
```

> 请务必 cd 到相应 op 的目录下面在进行编译，脚本文件里面采用了**相对路径**，否则编译可能出现问题。

但是能执行成功的前提是您**有正确安装 cuda 和 cudnn**（相关安装方法，这里不再赘述，如果您还不会在 Linux 环境下安装 GPU 版本的 tensorflow 和其运行环境，建议您先不要着急使用本项目），而且文件里面的路径需要您手动根据实际进行调整。

原作者书写的编译教程：<https://github.com/charlesq34/pointnet2#compile-customized-tf-operators>

### 训练

运行:

```shell
$ python tf_cls.py
```

### 可视化

ckpt 文件保存在`summary`文件夹下。

运行:

```shell
$ tensorboard --logdir summary
```

## 性能

1. 训练集准确率和验证集准确率

![训练集准确率和验证集准确率](https://raw.githubusercontent.com/HarborZeng/pointnet2-keras/master/result_image/model_accuracy.png)

2. 训练集损失和验证集损失率

![训练集损失和验证集损失](https://raw.githubusercontent.com/HarborZeng/pointnet2-keras/master/result_image/model_loss.png)

3. 利用keras构建的神经网络

![利用keras构建的神经网络](https://raw.githubusercontent.com/HarborZeng/pointnet2-keras/master/result_image/nn_graph.gif)

## TODO

- [ ] part segmentation
- [ ] semantic scene

## 参考资料

1. TianzhongSong/PointNet-Keras https://github.com/TianzhongSong/PointNet-Keras
2. charlesq34/pointnet2 https://github.com/charlesq34/pointnet2