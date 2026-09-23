---
title: Convolutional Networks
layout: default
nav_order: 4
parent: Models
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

Convolutional neural networks (CNNs) are the standard architecture for data with a spatial, grid-like structure: images above all, but also audio spectrograms, time series and 3D volumes. In computer vision they are used for classification, detection and segmentation, and they remain the usual feature-extracting *backbone* even in systems that add other components on top.

A convolutional layer is a special case of a fully connected layer: one whose weight matrix is sparse and whose entries are tied together. Conversely, a fully connected layer is a convolution whose kernel covers the whole input. The restrictions are the point. They encode assumptions about images that a fully connected layer would otherwise have to learn from data, and they cut the number of parameters by orders of magnitude.

## Core Concepts of CNNs

While a fully connected layer from an MLP *can* be used on an image (by flattening it into a vector), this approach is inefficient and ignores the image's structure. CNNs build on three principles that exploit this structure:

1. **Parameter Sharing:** In a fully connected layer, every input pixel would have a unique weight connecting it to a neuron. In a convolutional layer, the *same* small kernel (with its set of weights) is applied across the entire image. This means a feature detector (e.g., for a horizontal edge) learned in one part of the image is reused everywhere else. This dramatically reduces the number of parameters, making the network more efficient to train and less prone to overfitting.

2. **Spatial Locality:** CNNs assume that features that are close together in the input (e.g., pixels in a small patch) are more related than features that are far apart. By using small kernels, the network first learns simple, local patterns (edges, corners, colors), and subsequent layers combine these into more complex patterns (eyes, wheels, text). This builds a hierarchy of features.

3. **Translation Equivariance:** Because the same kernel is applied everywhere, shifting the input shifts the output feature map by the same amount. A detector for a wheel does not have to be learned separately for every position in the image. Pooling and the final classifier then add a degree of translation *invariance* on top, so that the prediction does not change when the object moves. These built-in assumptions (the *inductive bias* of the architecture) are what make CNNs data-efficient on images. [Transformers]({{ site.baseurl }}{% link docs/models/transformers.md %}) have no such bias and need either far more data or explicit positional information instead.

## Convolution Layer

{: .important }
>What deep learning calls *convolution* is, strictly speaking, **cross-correlation**: the kernel is slid over the input and multiplied element-wise *without* being flipped first. A mathematical convolution flips the kernel. Since the kernel is learned, a network can learn the flipped kernel just as easily, so the distinction has no practical consequence and frameworks, PyTorch's `nn.Conv2d` included, simply compute the cross-correlation. It matters only when comparing with signal-processing texts, and it reappears in the backward pass below, which *is* a true convolution.

A 2D convolution takes two matrices, the input (an image or a feature map) and a small kernel, and produces an output matrix. The operation has four hyperparameters, kernel size, stride, padding and dilation, which determine the size of the output and which input elements each output element sees. We introduce them one by one, and the output-size formula grows accordingly. Height and width are handled independently, so all formulas are written for one spatial dimension, with $$n_{in}$$ the input size and $$n_{out}$$ the output size along it.

### Kernel

{: .definition }
>The **kernel** (also called *filter*) of a convolution is a small matrix of weights, typically 3×3 or 5×5, occasionally 7×7 in the first layer of a network. These weights are the learnable parameters of the layer.

The kernel is placed on the input, the overlapping elements are multiplied pairwise and the products are summed to give one element of the output. The kernel is then moved on and the process is repeated for every position where it fits inside the input. The figure shows a 3×3 kernel on a 7×7 input, producing a 5×5 output.

<div align="center">
  <img src="{{ site.baseurl }}/assets/images/convolution_layer.png" width="800">
</div>

Without padding the output is smaller than the input, because the kernel has to fit inside it:

$$ n_{out} = n_{in} - k + 1, $$

where $$k$$ is the kernel size.

### Stride

{: .definition }
>The **stride** $$s$$ is the number of input elements the kernel moves between two consecutive positions. The default is $$1$$. A larger stride skips positions and therefore reduces the spatial size of the output.

The output size becomes

$$ n_{out} = \left\lfloor\frac{n_{in} - k}{s}\right\rfloor + 1. $$

The floor accounts for the last positions where the kernel no longer fits. A stride-2 convolution roughly halves the spatial size and is a common, learnable alternative to pooling.

### Padding

{: .definition }
>**Padding** adds $$p$$ extra elements on every side of the input before the convolution. It is used to control the output size and to let the kernel see the border elements as often as the interior ones.

Most often the added elements are zeros (*zero padding*). Reflecting or replicating the border values is also used, but is less common. Two settings are common enough to have names: *valid* padding, $$p = 0$$, and *same* padding, $$p = (k - 1)/2$$ for odd $$k$$ and stride $$1$$, which keeps the output the same size as the input. The figure shows same padding: a 7×7 input padded by one element, a 3×3 kernel and a 7×7 output.

<div align="center">
  <img src="{{ site.baseurl }}/assets/images/padding.png" width="800">
</div>

$$ n_{out} = \left\lfloor\frac{n_{in} + 2p - k}{s}\right\rfloor + 1. $$

### Dilation

{: .definition }
>**Dilation** $$d$$ spreads the kernel out so that consecutive kernel elements are $$d$$ input elements apart, leaving $$d - 1$$ gaps between them. The default $$d = 1$$ means no gaps. Dilation enlarges the region the kernel covers without adding weights.

A dilated kernel behaves like a larger kernel with an **effective size**

$$ k_{\text{eff}} = d\,(k - 1) + 1. $$

A 3×3 kernel with $$d = 2$$ covers the same 5×5 window as a 5×5 kernel, but has 9 weights instead of 25 and skips the elements in between. The figure shows exactly this case on a 7×7 input, giving a 3×3 output.

<div align="center">
  <img src="{{ site.baseurl }}/assets/images/dilation_cnn.png" width="800">
</div>

Replacing $$k$$ by $$k_{\text{eff}}$$ in the previous formula gives the general output size,

$$ n_{out} = \left\lfloor\frac{n_{in} + 2p - d\,(k - 1) - 1}{s}\right\rfloor + 1, $$

which is the formula you will find in the PyTorch documentation. For $$d = 1$$ it reduces to the padding formula above.

### Channels

So far the input was a single matrix. Real inputs have several **channels**: a colour image is a $$3 \times H \times W$$ tensor, and the output of a convolutional layer, called a **feature map**, is a $$C_{out} \times H_{out} \times W_{out}$$ tensor in which every channel is the response of one learned feature detector.

{: .definition }
>A convolutional layer with $$C_{in}$$ input channels and $$C_{out}$$ output channels has $$C_{out}$$ kernels, each of shape $$C_{in} \times k \times k$$, and one bias per output channel. Output channel $$o$$ is the sum of the 2D cross-correlations of every input channel with the corresponding slice of kernel $$o$$:
>
>$$ Y_o = b_o + \sum_{c=1}^{C_{in}} X_c \star W_{o,c}, \qquad o = 1, \dots, C_{out}, $$
>
>where $$\star$$ is the single-channel operation described above. The number of learnable parameters is therefore
>
>$$ C_{out}\,\bigl(C_{in}\,k^2 + 1\bigr). $$

The count does not depend on the image size, which is why parameter sharing is so strong of a tool. In numbers: the first layer of a typical network, 3 input channels to 64 output channels with 3×3 kernels, has $$64 \cdot (27 + 1) = 1792$$ parameters. A fully connected layer that maps a flattened $$224 \times 224 \times 3$$ image to just 64 numbers already needs $$150\,528 \cdot 64 \approx 9.6 \cdot 10^{6}$$ weights.

Two special cases are worth knowing. A **1×1 convolution** has $$k = 1$$: it mixes the channels at every position independently and is a cheap way to change the number of channels, it is used in bottleneck blocks of ResNet and Inception.

And the same operation exists in other dimensions: **1D convolutions** slide the kernel along one axis (audio, time series) and **3D convolutions** along three (video, volumetric scans). Everything on this page applies to them unchanged.

### Receptive Field

{: .definition }
>The **receptive field** of an output element is the region of the network's input that can influence it. For a single convolutional layer it is $$k \times k$$, or $$k_{\text{eff}} \times k_{\text{eff}}$$ if dilation $\neq1$.

Receptive fields grow with depth. With stride $$1$$, every additional layer adds $$k - 1$$ to the receptive field: two stacked 3×3 layers see a 5×5 window, three see 7×7. Strided convolutions and pooling layers multiply the growth of all later layers, and dilation enlarges it without new weights.

This is why modern networks stack small kernels instead of using large ones. Two 3×3 layers cover the same window as one 5×5 layer but need $$2 \cdot 9 = 18$$ weights per input-output channel pair instead of $$25$$, three 3×3 layers replace a 7×7 layer with $$27$$ weights instead of $$49$$, and every extra layer contributes an additional nonlinearity. It is also the mechanism behind the hierarchy of features from the introduction: early layers see a few pixels and detect edges, late layers see most of the image and detect objects.

## Pooling Layer

A **pooling layer** reduces the spatial size of a feature map by summarizing each $$k \times k$$ window with a single number. Kernel size, stride and padding mean the same as for a convolution and the output-size formula is the same. Unlike a convolution, pooling has no weights and is applied to each channel separately, so the number of channels does not change. The usual setting is a 2×2 window with stride 2, which halves the height and the width; the figure below uses a 3×3 window with stride 3.

<div align="center">
  <img src="{{ site.baseurl }}/assets/images/pooling_layer.png" width="800">
</div>

The two common types are **max pooling**, which keeps the largest value in the window, and **average pooling**, which takes the mean. Max pooling is the usual default inside a network. **Global average pooling** is average pooling with a window the size of the whole feature map: it turns a $$C \times H \times W$$ tensor into a vector of $$C$$ numbers. It is how modern architectures such as ResNet hand the last feature map to the classifier, and it makes the network accept images of any size.

Pooling and strided convolutions serve the same purpose. Pooling is fixed and parameter-free, whereas a strided convolution learns how to downsample. Recent architectures use the latter for most downsampling steps and keep pooling for the very beginning and the very end of the network.

## Gradient Backpropagation

While [backpropagation]({{ site.baseurl }}{% link docs/training/backpropagation.md %}) is explained in the training section, the backpropagation through convolution and pooling layers is included here for completeness. We use the notation of that page: $$\mathcal{L}$$ is the loss and $$\partial\mathcal{L}/\partial\boldsymbol{y}$$ the upstream gradient. Because a kernel weight is used at every output position, the gradient with respect to it is a *sum* over all these positions.

### Convolution layer

{: .definition }
>The gradient of a convolutional layer with respect to the **weights** is the **convolution of the input feature map with the upstream gradient**, where the upstream gradient is acting as the kernel.
>
>The gradient with respect to the **input feature map** is the **convolution of the zero-padded upstream gradient with the flipped kernel** (the kernel rotated by 180°).

We show both on an example with a $$3 \times 3$$ input feature map $$\boldsymbol{x}$$, a $$2 \times 2$$ kernel $$\boldsymbol{w}$$, stride $$1$$ and no padding, so the output feature map $$\boldsymbol{y}$$ is $$2 \times 2$$.

**Backpropagation with respect to the weights:**

$$ \mathrm{vjp_{conv\_w}}\left(\frac{\partial\mathcal{L}}{\partial\boldsymbol{y}},\boldsymbol{x}\right) = \begin{bmatrix} \frac{\partial\mathcal{L}}{\partial w_{11}} & \frac{\partial\mathcal{L}}{\partial w_{12}} \\ \frac{\partial\mathcal{L}}{\partial w_{21}} & \frac{\partial\mathcal{L}}{\partial w_{22}}\end{bmatrix} = \mathrm{conv}\left(\begin{bmatrix} x_{11} & x_{12} & x_{13} \\ x_{21} & x_{22} & x_{23} \\ x_{31} & x_{32} & x_{33} \end{bmatrix}, \begin{bmatrix} \frac{\partial\mathcal{L}}{\partial y_{11}} & \frac{\partial\mathcal{L}}{\partial y_{12}} \\ \frac{\partial\mathcal{L}}{\partial y_{21}} & \frac{\partial\mathcal{L}}{\partial y_{22}}\end{bmatrix} \right) $$

**Backpropagation with respect to the input feature map:**

$$ \mathrm{vjp_{conv\_x}}\left(\frac{\partial\mathcal{L}}{\partial\boldsymbol{y}},\boldsymbol{w}\right) = \begin{bmatrix} \frac{\partial\mathcal{L}}{\partial x_{11}} & \frac{\partial\mathcal{L}}{\partial x_{12}} & \frac{\partial\mathcal{L}}{\partial x_{13}} \\ \frac{\partial\mathcal{L}}{\partial x_{21}} & \frac{\partial\mathcal{L}}{\partial x_{22}} & \frac{\partial\mathcal{L}}{\partial x_{23}} \\ \frac{\partial\mathcal{L}}{\partial x_{31}} & \frac{\partial\mathcal{L}}{\partial x_{32}} & \frac{\partial\mathcal{L}}{\partial x_{33}} \end{bmatrix} = \mathrm{conv}\left(\begin{bmatrix} 0 & 0 & 0 & 0 \\ 0 & \frac{\partial\mathcal{L}}{\partial y_{11}} & \frac{\partial\mathcal{L}}{\partial y_{12}} & 0 \\ 0 & \frac{\partial\mathcal{L}}{\partial y_{21}} & \frac{\partial\mathcal{L}}{\partial y_{22}} & 0 \\ 0 & 0 & 0 & 0\end{bmatrix}, \begin{bmatrix} w_{22} & w_{21} \\ w_{12} & w_{11} \end{bmatrix} \right) $$

Where do these come from? Each output element is $$y_{ij} = \sum_{a,b} w_{ab}\, x_{i+a-1,\,j+b-1}$$. For a weight, the chain rule with several paths gives $$\partial\mathcal{L}/\partial w_{ab} = \sum_{i,j} \frac{\partial\mathcal{L}}{\partial y_{ij}}\, x_{i+a-1,\,j+b-1}$$, which is precisely the cross-correlation of $$\boldsymbol{x}$$ with $$\partial\mathcal{L}/\partial\boldsymbol{y}$$ as the kernel. For an input element, $$x_{mn}$$ contributes to every $$y_{ij}$$ through the weight $$w_{m-i+1,\,n-j+1}$$. Collecting these terms gives a cross-correlation with the index-reversed, i.e. flipped and padding the upstream gradient by $$k - 1$$ zeros on each side takes care of the border elements of $$\boldsymbol{x}$$, which are covered by fewer output positions. Reversing the kernel is exactly what turns a cross-correlation into a mathematical convolution.

{: .important }
>**The backward pass of a convolution is itself a convolution.** This is why convolutional layers are as cheap to train as to evaluate, and why frameworks implement both directions with the same optimized routines.
>
>Two caveats. The statement is exact for stride $$1$$. For stride $$s > 1$$ the gradient with respect to the input is a *transposed convolution*: first insert $$s - 1$$ zeros between the elements of the upstream gradient, then proceed as above. Transposed convolutions are also used the other way round, to *upsample* feature maps in segmentation decoders. And with channels, the gradient for kernel slice $$W_{o,c}$$ uses input channel $$c$$ and upstream-gradient channel $$o$$, the gradient for input channel $$c$$ sums over all output channels, and for a batch the weight gradient is summed over the samples.

If you implement this yourself, verify it with the gradient check described on the backpropagation page.

### Pooling layer

A pooling layer has no weights, so only the gradient with respect to the input is needed, and it is computed window by window:

- **Max pooling** routes the upstream gradient to the position of the maximum in the window. All other positions receive $$0$$. The forward pass therefore has to remember the index of the maximum, which is the memory a max-pooling layer needs.
- **Average pooling** spreads the upstream gradient evenly: every element of a $$k \times k$$ window receives $$1/k^2$$ of it.

With overlapping windows (stride smaller than the kernel) an input element belongs to several windows, and its contributions are summed, again the chain rule with several paths.

## Convolutional Neural Network

A convolutional neural network stacks the layers above. The basic building block is

<div align="center"><strong>convolution → batch normalization → ReLU</strong></div>

repeated a few times, followed by a downsampling step, either max pooling or a stride-2 convolution. Batch normalization is covered on the [Neural Network Training Fundamentals]({{ site.baseurl }}{% link docs/training/net_training_fundamentals.md %}#batch-normalization) page. Two rules of thumb govern the shape of the tensors as they flow through the network:

- **Spatial size shrinks, channel count grows.** Each downsampling step halves height and width, and the number of channels is typically doubled at the same time, so the computation per layer stays roughly constant. A $$224 \times 224 \times 3$$ image might become $$112 \times 112 \times 64$$, then $$56 \times 56 \times 128$$, and end as a $$7 \times 7 \times 512$$ feature map.
- **Early layers detect simple, local patterns; late layers combine them.** The receptive field grows with every layer, so the last feature map describes large parts of the image in terms of increasingly abstract features: edges, then textures, then object parts, then objects.

The last feature map is turned into a prediction by a *head*. Classically it was flattened and passed through a few fully connected layers (LeNet, AlexNet, VGG); modern networks reduce it by global average pooling and follow with a single linear layer (ResNet and its successors). The convolutional part without the head is called the **backbone**. It is what gets reused for other tasks, from detection and segmentation to [metric learning]({{ site.baseurl }}{% link docs/training/metric_learning.md %}), usually with weights pre-trained on a large classification dataset.

A few architectures worth knowing by name:

- **LeNet-5** (1998): the original CNN, for handwritten digits. Two rounds of convolution and pooling followed by fully connected layers.
- **AlexNet** (2012): LeNet scaled up and trained on GPUs, with ReLU and dropout. Its ImageNet win started the deep learning era.
- **VGG** (2014): nothing but stacked 3×3 convolutions. It established that depth with small kernels beats large kernels, the receptive-field argument above.
- **ResNet** (2015): adds *residual* (skip) connections, $$\boldsymbol{y} = F(\boldsymbol{x}) + \boldsymbol{x}$$, which give the gradient a direct path around each block and make networks with 50 to 150 layers trainable. ResNet-18 and ResNet-50 are still the default backbones in robotics and vision.

## Expected Knowledge

Answer the following questions to test your understanding of convolutional networks.

1. **Core Principles:** What are the main advantages of using a convolutional layer over a fully connected (dense) layer for processing image data? Name the three principles from this page and explain each briefly.

2. **Output Size Calculation:** You have an input feature map of size 64x64 pixels. You apply a single convolutional layer with a kernel size of 5x5, a stride of 2, and padding of 1. What will be the spatial dimension (height x width) of the output feature map? Show your calculation.

3. **Component Roles:** What is the primary purpose of a **pooling layer** (e.g., MaxPooling) in a CNN? How does its function of reducing spatial dimensions differ from using a large **stride** in a convolutional layer?

4. **Receptive Field:** How does **dilation** affect the receptive field of a neuron in a convolutional layer? Why might you use a 3x3 kernel with a dilation of 2 instead of a standard 5x5 kernel?

5. **Parameter Count:** A convolutional layer takes a 3-channel image of size 128x128 and produces 32 feature maps with 5x5 kernels. How many learnable parameters does it have, biases included? Does the answer change for a 256x256 image? How many weights would a fully connected layer need just to produce 32 numbers from the flattened 128x128x3 image?

6. **Stacking Small Kernels:** What is the receptive field of an output element after two stacked 3x3 convolutions with stride 1? After three? Compare the number of weights per input-output channel pair with a single 5x5 or 7x7 kernel, and name one further advantage of the stack.

7. **Backward Pass:** Why does the gradient with respect to the input use the flipped kernel, while the gradient with respect to the weights does not? What changes when the stride is 2?
