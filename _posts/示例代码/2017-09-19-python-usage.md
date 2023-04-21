---
layout: post
title: python usage
category: 示例代码
tags: code
keywords: python代码
description: 
---

# python reshape和matlab reshape的区别

```
mat = [1:12]

>>mat =

     1     2     3     4     5     6     7     8     9    10    11    12

reshape(mat,[3,4])

>>ans =

     1     4     7    10
     2     5     8    11
     3     6     9    12
```

```
mat = np.arange(1,13)

mat

>>array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12])

r = np.reshape(mat,(3,4))

>>array([[ 1,  2,  3,  4],
       [ 5,  6,  7,  8],
       [ 9, 10, 11, 12]])

r.shape

>>(3, 4)
```

```
r = np.reshape(mat, (3,4), order="F")

r
array([[ 1,  4,  7, 10],
       [ 2,  5,  8, 11],
       [ 3,  6,  9, 12]])
```

需要在python程序中指明使用Fortran order, 如

```
np.reshape(matrix, (n,n), order="F")
```

Numpy默认是C order, Matlab是 Fortran order,就是python的reshape取行元素填充行元素，matlab取列元素填充列元素

# python中对文件、文件夹的操作需要涉及到os模块和shutil模块。

创建文件:

```
os.mknod("test.txt") 创建空文件
open("test.txt",w)   直接打开一个文件，如果文件不存在则创建文件
```

创建目录:

```
os.mkdir("file")     创建目录
```

复制文件:

```
shutil.copyfile("oldfile","newfile")       oldfile和newfile都只能是文件
shutil.copy("oldfile","newfile")           oldfile只能是文件夹，newfile可以是文件，也可以是目标目录
```

复制文件夹：

```
shutil.copytree("olddir","newdir")        olddir和newdir都只能是目录，且newdir必须不存在
```

重命名文件（目录）

```
os.rename("oldname","newname")       文件或目录都是使用这条命令
```

移动文件（目录）

```
shutil.move("oldpos","newpos")    
```

删除文件

```
os.remove("file")
```

删除目录

```
os.rmdir("dir")         只能删除空目录
shutil.rmtree("dir")    空目录、有内容的目录都可以删 
```

转换目录

```
os.chdir("path")    换路径
```

判断目标

```
os.path.exists("goal")    判断目标是否存在
os.path.isdir("goal")     判断目标是否目录
os.path.isfile("goal")    判断目标是否文件   
```