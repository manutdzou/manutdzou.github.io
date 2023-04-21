---
layout: post
title: python实现各种排序算法
category: 算法
tags: 算法基础
keywords: python
description: 排序算法 
---

# 用python实现各种排序算法

## 归并排序

归并排序也称合并排序，是分治法的典型应用。分治思想是将每个问题分解成一个个小问题，将每个小问题解决，然后合并。具体的归并排序就是，将一组无序数按n/2递归分解成只有一个元素的子项，一个元素就是已经排好序的了。然后将这些有序的子元素进行合并。合并的过程就是对两个已经排好序的子序列，先选取两个子序列中最小的元素进行比较，选取两个元素中最小的那个子序列并将其从子序列中去掉添加到最终的结果集中，直到两个子序列归并完成。

```python
#!/usr/bin/python  
import sys  
   
def merge(nums, first, middle, last):  
    ''''' merge '''  
    # 切片边界,左闭右开并且是了0为开始  
    lnums = nums[first:middle+1]   
    rnums = nums[middle+1:last+1]  
    lnums.append(sys.maxint)  
    rnums.append(sys.maxint)  
    l = 0  
    r = 0  
    for i in range(first, last+1):  
        if lnums[l] < rnums[r]:  
            nums[i] = lnums[l]  
            l+=1  
        else:  
            nums[i] = rnums[r]  
            r+=1  
def merge_sort(nums, first, last):  
    ''''' merge sort 
    merge_sort函数中传递的是下标，不是元素个数 
    '''  
    if first < last:  
        middle = (first + last)/2  
        merge_sort(nums, first, middle)  
        merge_sort(nums, middle+1, last)  
        merge(nums, first, middle,last)  
   
if __name__ == '__main__':  
    nums = [10,8,4,-1,2,6,7,3]  
    print 'nums is:', nums  
    merge_sort(nums, 0, 7)  
    print 'merge sort:', nums
```

稳定，时间复杂度 O(nlog n)

## 插入排序

```python
def insert_sort(num_list):
  """
  插入排序
  """
  for i in range(len(num_list)-1):
    for j in range(i+1, len(num_list)):
      if num_list[i]>num_list[j]:
        num_list[i],num_list[j] = num_list[j],num_list[i]
  return num_list
 
if __name__ == '__main__':  
    nums = [10,8,4,-1,2,6,7,3]  
    print 'nums is:', nums  
    insert_sort(nums)  
    print 'insert sort:', nums
```

稳定，时间复杂度 O(n^2)

## 选择排序

选择排序(Selection sort)是一种简单直观的排序算法。它的工作原理如下。首先在未排序序列中找到最小（大）元素，存放到排序序列的起始位置，然后，再从剩余未排序元素中继续寻找最小（大）元素，然后放到已排序序列的末尾。以此类推，直到所有元素均排序完毕。

```
import sys  
def select_sort(a):  
    ''''' 选择排序  
    每一趟从待排序的数据元素中选出最小（或最大）的一个元素， 
    顺序放在已排好序的数列的最后，直到全部待排序的数据元素排完。 
    选择排序是不稳定的排序方法。 
    '''  
    a_len=len(a)  
    for i in range(a_len):#在0-n-1上依次选择相应大小的元素   
        min_index = i#记录最小元素的下标   
        for j in range(i+1, a_len):#查找最小值  
            if(a[j]<a[min_index]):  
                min_index=j  
        if min_index != i:#找到最小元素进行交换  
            a[i],a[min_index] = a[min_index],a[i]  
   
if __name__ == '__main__':  
    A = [10, -3, 5, 7, 1, 3, 7]    
    print 'Before sort:',A    
    select_sort(A)    
    print 'After sort:',A
```

不稳定，时间复杂度 O(n^2)

## 希尔排序

希尔排序，也称递减增量排序算法,希尔排序是非稳定排序算法。该方法又称缩小增量排序，因DL．Shell于1959年提出而得名。先取一个小于n的整数d1作为第一个增量，把文件的全部记录分成d1个组。所有距离为d1的倍数的记录放在同一个组中。先在各组内进行排序；然后，取第二个增量d2


```
import sys  
def shell_sort(a):  
    ''''' shell排序  
    '''  
    a_len=len(a)  
    gap=a_len/2#增量  
    while gap>0:  
        for i in range(a_len):#对同一个组进行选择排序  
            m=i  
            j=i+1  
            while j<a_len:  
                if a[j]<a[m]:  
                    m=j  
                j+=gap#j增加gap  
            if m!=i:  
                a[m],a[i]=a[i],a[m]  
        gap/=2  
   
if __name__ == '__main__':  
    A = [10, -3, 5, 7, 1, 3, 7]    
    print 'Before sort:',A    
    shell_sort(A)    
    print 'After sort:',A
```

不稳定，时间复杂度 平均时间 O(nlogn) 最差时间O(n^s)1

## 堆排序 ( Heap Sort )

“堆”的定义：在起始索引为 0 的“堆”中：节点 i 的右子节点在位置 2 * i + 24) 节点 i 的父节点在位置 floor( (i – 1) / 2 )   : 注 floor 表示“取整”操作

堆的特性：每个节点的键值一定总是大于（或小于）它的父节点

“最大堆”：“堆”的根节点保存的是键值最大的节点。即“堆”中每个节点的键值都总是大于它的子节点。

上移，下移 ：当某节点的键值大于它的父节点时，这时我们就要进行“上移”操作，即我们把该节点移动到它的父节点的位置，而让它的父节点到它的位置上，然后我们继续判断该节点，直到该节点不再大于它的父节点为止才停止“上移”。现在我们再来了解一下“下移”操作。当我们把某节点的键值改小了之后，我们就要对其进行“下移”操作。

方法：我们首先建立一个最大堆(时间复杂度O(n))，然后每次我们只需要把根节点与最后一个位置的节点交换，然后把最后一个位置排除之外，然后把交换后根节点的堆进行调整(时间复杂度 O(lgn) )，即对根节点进行“下移”操作即可。 堆排序的总的时间复杂度为O(nlgn).

```
def build_max_heap(to_build_list):
  """建立一个堆"""
  # 自底向上建堆
  for i in range(len(to_build_list)/2 - 1, -1, -1):
    max_heap(to_build_list, len(to_build_list), i)
def max_heap(to_adjust_list, heap_size, index):
  """调整列表中的元素以保证以index为根的堆是一个最大堆"""
  # 将当前结点与其左右子节点比较，将较大的结点与当前结点交换，然后递归地调整子树
  left_child = 2 * index + 1
  right_child = left_child + 1
  if left_child < heap_size and to_adjust_list[left_child] > to_adjust_list[index]:
    largest = left_child
  else:
    largest = index
  if right_child < heap_size and to_adjust_list[right_child] > to_adjust_list[largest]:
    largest = right_child
  if largest != index:
    to_adjust_list[index], to_adjust_list[largest] = \
    to_adjust_list[largest], to_adjust_list[index]
    max_heap(to_adjust_list, heap_size, largest)
def heap_sort(to_sort_list):
  """堆排序"""
  # 先将列表调整为堆
  build_max_heap(to_sort_list)
  heap_size = len(to_sort_list)
  # 调整后列表的第一个元素就是这个列表中最大的元素，将其与最后一个元素交换，然后将剩余的列表再调整为最大堆
  for i in range(len(to_sort_list) - 1, 0, -1):
    to_sort_list[i], to_sort_list[0] = to_sort_list[0], to_sort_list[i]
    heap_size -= 1
    max_heap(to_sort_list, heap_size, 0)
if __name__ == '__main__':
  to_sort_list = [4, 1, 3, 2, 16, 9, 10, 14, 8, 7]
  print to_sort_list
  heap_sort(to_sort_list)
  print to_sort_list
```

不稳定，时间复杂度 O(nlog n)

## 快速排序

快速排序算法和合并排序算法一样，也是基于分治模式。对子数组A[p…r]快速排序的分治过程的三个步骤为：

分解：把数组A[p…r]分为A[p…q-1]与A[q+1…r]两部分，其中A[p…q-1]中的每个元素都小于等于A[q]而A[q+1…r]中的每个元素都大于等于A[q]；

解决：通过递归调用快速排序，对子数组A[p…q-1]和A[q+1…r]进行排序；

合并：因为两个子数组是就地排序的，所以不需要额外的操作。

对于划分partition 每一轮迭代的开始，x=A[r], 对于任何数组下标k，有：

1) 如果p≤k≤i，则A[k]≤x。

2) 如果i+1≤k≤j-1，则A[k]>x。

3) 如果k=r，则A[k]=x。

```
#!/usr/bin/python

import sys

def partion(array, p, r):
    x = array[r]
    i = p - 1
    for j in range(p, r):
        if (array[j] < x):
            i+=1
            array[j], array[i] = array[i], array[j]

    i+=1
    array[i], array[r] = array[r], array[i]
    return i

def quick_sort(array, p, r):
    if p < r:
        q = partion(array, p, r)
        quick_sort(array, p, q - 1)
        quick_sort(array, q + 1, r)

if __name__ == "__main__":
    array = [1, 3, 5, 23, 64, 7, 23, 6, 34, 98, 100, 9]
    quick_sort(array, 0, len(array) - 1)
    
    for a in array:
        sys.stdout.write("%d " % a)
```

不稳定，时间复杂度 最理想 O(nlogn)最差时间O(n^2)

