---
title: Convolutional Neural Networks
layout: default
has_children: false
nav_order: 3
mathjax: true
---

# Convolutional Neural Networks
{: .no_toc }

<details open markdown="block">
  <summary>
    Table of contents
  </summary>
  {: .text-delta }
1. TOC
{:toc}
</details>

## Introduction

Convolutional neural networks are highly utilized in architecture designs for processing of structured data. The most prominent example would be their use in image processing networks used for various task such as detection, classification or segmentation.

While fully connected layer and convolutional layer may be converted into the other, convolutions offer multiple non-negligible advantages for structured data.

## Convolution layer

{: .important }
>It is important not to confuse the mathematical operation of convolution to the 2D and 3D convolutions used in machine learning. While these operations share the same name they are calculated differently and present different things.

2D convolution is an operation on two matrices. One matrix consists of data we want to transform, the other is called the kernel. The result is a matrix of smaller size then the starting one. The convolution operation has multiple terms that affect the outcome and scope of the operation, these terms are listed and explained below.

### Kernel

{: .definition }
>The **kernel** of a convolution is a matrix (usually small in size, aprox. 3x3 or 5x5) with weights. It is these weights we want to learn in machine learning.

The elements of input matrix are multiplied by elements of the kernel matrix on corresponding positions and the result of the mutliplications are summed together to form a new element in the resulting matrix. This process is shown on the image below.

<div align="center">
  <img src="{{ site.baseurl }}/assets/images/convolution_layer.png" width="800">
</div>

The resulting matrix is smaller in size than the starting one. This size may be calculated with the following formula:

$$ s_{out} = s_{in} - \left\lceil\frac{k}{2}\right\rceil, $$

where $s_{in}$ is the size of input matrix, $s_{out}$ is the size of output matrix and $k$ is the kernel size. The division is rounded up.

### Stride

<div align="center">
  <img src="{{ site.baseurl }}/assets/images/placeholder.png" width="800">
</div>

### Dilation

<div align="center">
  <img src="{{ site.baseurl }}/assets/images/dilation_cnn.png" width="800">
</div>

### Padding

<div align="center">
  <img src="{{ site.baseurl }}/assets/images/placeholder.png" width="800">
</div>

## Pooling layer

Pooling layer is another type of layer higly utilized in architectures with convolutional layers. In convolutional layer the padding is usually used to preserve the spatial properties of input data, the pooling layer is used to lower the spatial dimension.

All the same rules regarding kernel, dilation and stride still apply. The main difference is how is the resulting element calculated.

There are two main types of pooling layer, maxpooling and minpooling. These layers take the kernel and from the input elements within the kernel take the maximal or minimal element and set it as a result. The image bellow shows a diagram of this process.

<div align="center">
  <img src="{{ site.baseurl }}/assets/images/pooling_layer.png" width="800">
</div>

## Gradient backpropagation

While [backpropagation]({{ site.baseurl }}{% link docs/backpropagation.md %}) is explained in the following article, the backpropagation through convolution and pooling layers is included here for completion and easier navigation.

### Convolution layer

### Pooling layer

## Expected knowledge
