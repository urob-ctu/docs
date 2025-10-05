---
title: Convolutional Networks
layout: default
nav_order: 4
parent: Models
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

## Core Concepts of CNNs

While a fully connected layer from an MLP *can* be used on an image (by flattening it into a vector), this approach is inefficient and ignores the image's structure. CNNs introduce two key principles to exploit this structure:

1. **Parameter Sharing:** In a fully connected layer, every input pixel would have a unique weight connecting it to a neuron. In a convolutional layer, the *same* small kernel (with its set of weights) is applied across the entire image. This means a feature detector (e.g., for a horizontal edge) learned in one part of the image is reused everywhere else. This dramatically reduces the number of parameters, making the network more efficient to train and less prone to overfitting.

2. **Spatial Locality:** CNNs assume that features that are close together in the input (e.g., pixels in a small patch) are more related than features that are far apart. By using small kernels, the network first learns simple, local patterns (edges, corners, colors), and subsequent layers combine these into more complex patterns (eyes, wheels, text). This builds a hierarchy of features.

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

$$ d_{out} = d_{in} - k + 1, $$

where $d_{in}$ is the size of input matrix, $d_{out}$ is the size of output matrix and $k$ is the kernel size. The division is rounded up.

### Stride

{: .definition }
>The **stride** is the number of steps the kernel takes when moving over the input matrix. The stride is usually set to 1, but may be set to any number. The stride is used to lower the spatial dimension of the resulting matrix.

The size of the resulting matrix may be calculated with the following formula:

$$ d_{out} = \left\lfloor\frac{d_{in} - k}{s}\right\rfloor + 1, $$

where $d_{in}$ is the size of input matrix, $d_{out}$ is the size of output matrix, $k$ is the kernel size and $s$ is the stride.

### Padding

{: .definition }
>The **padding** is the process of adding elements around the input matrix. The padding is used to preserve the spatial properties of the input matrix.

The value of the elements added may be set to different values, but the most common value is 0. This type of padding is called zero padding. Other types of padding are also used, but are less common.

<div align="center">
  <img src="{{ site.baseurl }}/assets/images/padding.png" width="800">
</div>

The size of the resulting matrix may be calculated with the following formula:

$$ d_{out} = \left\lfloor\frac{d_{in} + 2p - k}{s}\right\rfloor + 1, $$

where $d_{in}$ is the size of input matrix, $d_{out}$ is the size of output matrix, $k$ is the kernel size, $s$ is the stride and $p$ is the padding.

### Dilation

{: .definition }
>The **dilation** is the process of adding spaces between the elements of the kernel. The dilation is used to increase the receptive field of the kernel.

The dilation therefore stretches the kernel and increases the number of elements it interacts with. The dilation is usually set to 1, meaning that there are no spaces between the elements of the kernel.

<div align="center">
  <img src="{{ site.baseurl }}/assets/images/dilation_cnn.png" width="800">
</div>

The size of the resulting matrix may be calculated with the following formula:

$$ d_{out} = \left\lfloor\frac{d_{in} + 2p - k - (k - 1)\cdot (d - 1)}{s}\right\rfloor + 1, $$

where $d_{in}$ is the size of input matrix, $d_{out}$ is the size of output matrix, $k$ is the kernel size, $s$ is the stride, $p$ is the padding and $d$ is the dilation.

## Pooling layer

Pooling layer is another type of layer higly utilized in architectures with convolutional layers. In convolutional layer the padding is usually used to preserve the spatial properties of input data, the pooling layer is used to lower the spatial dimension.

All the same rules regarding kernel, dilation and stride still apply. The main difference is how is the resulting element calculated.

There are two main types of pooling layer, maxpooling and minpooling. These layers take the kernel and from the input elements within the kernel take the maximal or minimal element and set it as a result. The image bellow shows a diagram of this process.

<div align="center">
  <img src="{{ site.baseurl }}/assets/images/pooling_layer.png" width="800">
</div>

## Gradient backpropagation

While [backpropagation]({{ site.baseurl }}{% link docs/training/backpropagation.md %}) is explained in the training section, the backpropagation through convolution and pooling layers is included here for completion and easier navigation.

### Convolution layer

{: .definition }
>The backpropagation of convolutional layer with respect to the weights is defined as **convolution of input feature map with upstream gradient.**
>
>The backpropagation of convolutional layer with respect to the input feature map is defined as **convolution of padded upstream gradient with mirrored weights.**

We will show the backpropagation on an example where $ \boldsymbol{x} $ is the input feature map, $ \boldsymbol{w} $ is the kernel, $ \boldsymbol{y} $ is the output feature map and $ \frac{\partial\boldsymbol{p}}{\partial\boldsymbol{y}} $ is the upstream gradient.

**Backpropagation with respect to the weights:**

$$ \mathrm{vjp_{conv\_w}}\left(\frac{\partial\boldsymbol{p}}{\partial\boldsymbol{y}},\boldsymbol{x}\right) = \begin{bmatrix} \frac{\partial\boldsymbol{p}}{\partial w_{11}} & \frac{\partial\boldsymbol{p}}{\partial w_{12}} \\ \frac{\partial\boldsymbol{p}}{\partial w_{21}} & \frac{\partial\boldsymbol{p}}{\partial w_{22}}\end{bmatrix} = \mathrm{conv}\left(\begin{bmatrix} x_{11} & x_{12} & x_{13} \\ x_{21} & x_{22} & x_{23} \\ x_{31} & x_{32} & x_{33} \end{bmatrix}, \begin{bmatrix} \frac{\partial\boldsymbol{p}}{\partial y_{11}} & \frac{\partial\boldsymbol{p}}{\partial y_{12}} \\ \frac{\partial\boldsymbol{p}}{\partial y_{21}} & \frac{\partial\boldsymbol{p}}{\partial y_{22}}\end{bmatrix} \right) $$

**Backpropagation with respect to the input feature map:**

$$ \mathrm{vjp_{conv\_x}}\left(\frac{\partial\boldsymbol{p}}{\partial\boldsymbol{y}},\boldsymbol{w}\right) = \begin{bmatrix} \frac{\partial\boldsymbol{p}}{\partial x_{11}} & \frac{\partial\boldsymbol{p}}{\partial x_{12}} & \frac{\partial\boldsymbol{p}}{\partial x_{13}} \\ \frac{\partial\boldsymbol{p}}{\partial x_{21}} & \frac{\partial\boldsymbol{p}}{\partial x_{22}} & \frac{\partial\boldsymbol{p}}{\partial x_{23}} \\ \frac{\partial\boldsymbol{p}}{\partial x_{31}} & \frac{\partial\boldsymbol{p}}{\partial x_{32}} & \frac{\partial\boldsymbol{p}}{\partial x_{33}} \end{bmatrix} = \mathrm{conv}\left(\begin{bmatrix} 0 & 0 & 0 & 0 \\ 0 & \frac{\partial\boldsymbol{p}}{\partial y_{11}} & \frac{\partial\boldsymbol{p}}{\partial y_{12}} & 0 \\ 0 & \frac{\partial\boldsymbol{p}}{\partial y_{21}} & \frac{\partial\boldsymbol{p}}{\partial y_{22}} & 0 \\ 0 & 0 & 0 & 0\end{bmatrix}, \begin{bmatrix} w_{22} & w_{21} \\ w_{12} & w_{11} \end{bmatrix} \right) $$

{: .important }
>Very important property of the convolutional layer is:
>
>**The backpropagation is also a convolution operation!**

### Pooling layer

Backpropagation through pooling layer is a bit different than through convolutional layer. The pooling layer does not have any weights, so the backpropagation is calculated only with respect to the input feature map.

The backpropagation is calculated by taking the upstream gradient and placing it on the position of the maximal or minimal element in the kernel. The rest of the elements are set to 0.

## Convolutional neural network

The convolutional neural network is a concatenation of convolutional layers, activation function and optionally pooling layers.

The convolutional layers are used to extract features from the input data, while the pooling layers are used to lower the spatial dimension of the data.

## Expected Knowledge

Answer the following questions to test your understanding of convolutional networks.

1. **Core Advantages:** What are the two main advantages of using a convolutional layer over a fully connected (dense) layer for processing image data? Explain each concept briefly.

2. **Output Size Calculation:** You have an input feature map of size 64x64 pixels. You apply a single convolutional layer with a kernel size of 5x5, a stride of 2, and padding of 1. What will be the spatial dimension (height x width) of the output feature map? Show your calculation.

3. **Component Roles:** What is the primary purpose of a **pooling layer** (e.g., MaxPooling) in a CNN? How does its function of reducing spatial dimensions differ from using a large **stride** in a convolutional layer?

4. **Receptive Field:** How does **dilation** affect the receptive field of a neuron in a convolutional layer? Why might you use a 3x3 kernel with a dilation of 2 instead of a standard 5x5 kernel?
