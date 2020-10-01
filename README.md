# 2020_gkd_ImageProcessingProgramming

# 2020年国科大秋季图像处理与分析编程课后作业一


**本作业主要由五个文件构成，每个文件分别对应作业中的每一问。**

## Q1：黑白图像灰度扫描
### 问题描述
实现一个函数 s = scanLine4e(f, I, loc), 其中 f 是一个灰度图像， I 是一个整数， loc 是一个字
符串。当 loc 为’row’时， I 代表行数。当 loc 为’column’时， I 代表列数。输出 s 是对应的相
关行或者列的像素灰度矢量。

调用该函数，提取cameraman.tif 和 einstein.tif 的中心行和中心列的像素灰度矢量并将扫描
得到的灰度序列绘制成图。

### 文件描述
输入为三个参数，f:图像路径，I：提取图像的第I行或第I列，loc：‘row’或‘col’，指定提取行或列。

执行文件后会产生图像第I行以及第I列像素灰度的直方图。

另外，如果图像路径不正确会产生提示信息“Image path is invalid, please input again!”如果图像索引超出范围
会提示信息“The pixel vector index is invalid, please input again!”如果loc参数不正确，会提示信息
“The loc parameter is invalid, please input again!”

## Q2：彩色图像转换为黑白图像
### 问题描述
图像处理中的一个常见问题是将彩色RGB图像转换为单色灰度图像，第一种常用的方法是去三个元素R，G，B
