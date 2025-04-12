# Artificial-neural-network
神经网络搭建练习
- [LeNet](https://github.com/LEO690201/Artificial-neural-network/tree/LeNet-5)


  
LeNet, conceived by Yann LeCun et al. in 1998, represents an seminal convolutional neural network (CNN) with significance that includes:

- **Pioneering Architecture:** It was the first to integrate convolutional layers, pooling layers, and fully connected layers, establishing the blueprint for modern CNNs and accelerating the adoption of deep learning in computer vision.
  
- **Practical Application:** Its successful deployment in handwritten digit recognition, notably in check processing systems, demonstrated the practical efficacy of CNNs in real-world scenarios.
  
- **Foundational Techniques:** LeNet introduced key concepts such as local connections, weight sharing, and downsampling, which not only minimized the number of parameters but also set the design principles for subsequent influential models like AlexNet and ResNet.
  
- [AlexNet](https://github.com/LEO690201/Artificial-neural-network/tree/AlexNet)




   AlexNet, introduced by Alex Krizhevsky et al. in 2012, is a groundbreaking convolutional neural network that marked significant milestones, including:

- **Revival of Deep Learning:**
  It triumphed in the ImageNet competition with a Top-5 error rate of 15.3%, compared to 26.2% for traditional methods, igniting the deep learning revolution in computer vision.

- **Technological Innovations:**
  - **ReLU Activation:** First to employ ReLU (Rectified Linear Unit) to mitigate the vanishing gradient problem.
  - **Dropout Regularization:** Introduced to prevent overfitting, enhancing model generalization.
  - **GPU Acceleration:** Leveraged GPUs for parallel processing, overcoming computational constraints.
  - **Local Response Normalization (LRN):** Implemented to improve training by introducing a form of normalization across feature maps.

- **Structural Innovation:**
  Comprising 5 convolutional layers followed by 3 fully connected layers, it demonstrated the power of deeper architectures, influencing designs like VGG and ResNet.

- **Paradigm Shift:**
  It catalyzed a shift in computer vision from hand-crafted features to end-to-end learning, establishing the modern AI training paradigm based on large datasets, deep networks, and GPU-intensive computation.


- [VGG-16](https://github.com/LEO690201/Artificial-neural-network/tree/VGG-16)

- 

VGG-16 is a classic convolutional neural network (CNN) proposed by the Visual Geometry Group (VGG) at Oxford University in 2014. It achieved remarkable results in the ImageNet Large Scale Visual Recognition Challenge (ILSVRC) with a Top-5 error rate of 7.3%. Below are its key features and design principles:

---

##### **Core Design Features**

1. **Modular and Minimalist Architecture**  
   - **Exclusive Use of 3x3 Convolutions**: All convolutional layers use small 3x3 kernels (vs. AlexNet's 11x11/5x5), reducing parameters while increasing nonlinearity.  
   - **Stacking Strategy**: Cascading 2-3 convolutional layers achieves equivalent receptive fields to larger kernels (e.g., two 3x3 layers ≈ one 5x5 layer).  
   - **Max Pooling**: Spatial downsampling is done via five 2x2 max-pooling layers, halving feature map dimensions at each stage.

2. **Depth Breakthrough**  
   - **16-Layer Structure**: 13 convolutional layers (grouped into 5 stages) + 3 fully connected (FC) layers, doubling AlexNet's depth.  
   - **Parameter Scale**: ~138 million parameters (FC layers account for ~120 million, ~90% of the total).

3. **Standardized Components**  
   - ReLU activation for all layers.  
   - Dropout (rate=0.5) between FC layers to mitigate overfitting.

---

##### **Architecture Details (Standard Configuration)**

| Layer Type         | Configuration                                | Output Size (Input 224x224 RGB) |
|--------------------|---------------------------------------------|---------------------------------|
| **Convolutional Block 1** | 2×[3x3 conv, 64 channels] + max pool       | 112x112x64                     |
| **Convolutional Block 2** | 2×[3x3 conv, 128 channels] + max pool      | 56x56x128                      |
| **Convolutional Block 3** | 3×[3x3 conv, 256 channels] + max pool      | 28x28x256                      |
| **Convolutional Block 4** | 3×[3x3 conv, 512 channels] + max pool      | 14x14x512                      |
| **Convolutional Block 5** | 3×[3x3 conv, 512 channels] + max pool      | 7x7x512                        |
| **Fully Connected Layers** | 4096 → 4096 → 1000 (Softmax output)        | Class probabilities            |

---

##### **Innovations and Impact**

1. **Validation of Depth**: Demonstrated that **stacking small convolutions enables stable training of 16-layer networks**, inspiring deeper architectures.  
2. **Feature Transferability**: Pre-trained VGG convolutional blocks became a universal backbone for early CV tasks (e.g., object detection, segmentation).  
3. **Standardized Design**: Its modularity influenced later models like ResNet and DenseNet.

---

##### **Limitations and Improvements**

- **Parameter Redundancy**: Heavy FC layers (120M/138M params) were later replaced by global average pooling (e.g., Inception, ResNet).  
- **Computational Cost**: Deep convolutions require high inference resources, motivating lightweight designs (e.g., MobileNet's depthwise separable convolutions).  
- **Gradient Issues**: Pure stacking struggles with vanishing gradients, addressed by residual connections (ResNet).

---

##### **Practical Considerations**

- **Transfer Learning**: Use pre-trained convolutional blocks as feature extractors (PyTorch/TensorFlow provide official weights).  
- **Hardware Requirements**: Full VGG-16 needs ≥4GB GPU memory (batch_size=32); consider pruning/quantization for deployment.  
- **Efficient Alternatives**: For resource-constrained scenarios, use EfficientNet or MobileNetV3.  

---

VGG-16 remains a foundational model for teaching and benchmarking due to its simplicity and strong feature extraction capabilities.


# GoogleNet (Inception v1)

GoogleNet, also known as Inception v1, is a convolutional neural network architecture developed by Szegedy et al. in their paper "Going Deeper with Convolutions". It was the winner of the ILSVRC 2014 competition and set new records for classification and detection tasks.

## Key Features:
- **Inception Module**: The core component of GoogleNet is the inception module, which applies multiple convolutions and pooling operations in parallel within the same layer. This allows the network to capture features at different scales and improves its ability to generalize.
- **Depth and Width**: GoogleNet is both deep and wide, consisting of 22 layers and over 5 million parameters. This large number of parameters helps it learn complex patterns in the data.
- **Global Average Pooling**: Instead of fully connected layers, GoogleNet uses global average pooling before the final softmax layer, reducing the number of parameters and preventing overfitting.

## Architecture:
The architecture consists of several inception modules stacked on top of each other. Each inception module includes:
- 1x1 convolutions to reduce dimensionality.
- 3x3 and 5x5 convolutions to capture features at different scales.
- Max pooling to downsample the input.
- Concatenation of all these outputs to form the final output of the module.

## Applications:
GoogleNet has been used in various computer vision tasks such as image classification, object detection, and segmentation. Its effectiveness in capturing multi-scale features makes it particularly useful for complex recognition tasks.

## Advantages:
- High accuracy due to its ability to capture multi-scale features.
- Efficient use of computational resources through the inception module.
- Robustness to variations in input images.

## Disadvantages:
- Very high memory consumption due to the large number of parameters.
- Complexity in training and inference due to the inception module.

Overall, GoogleNet is a groundbreaking architecture that significantly advanced the state of the art in image recognition tasks by introducing the inception module and demonstrating the power of deep learning in visual understanding.


# ResNet18

ResNet18 is a convolutional neural network (CNN) architecture that is 18 layers deep. It was introduced by He et al. in their paper "Deep Residual Learning for Image Recognition" and won the ILSVRC 2015 competition with an error rate of 3.57%.

## Key Features:
- **Depth**: Consists of 18 layers, including convolutional layers, pooling layers, and fully connected layers.
- **Skip Connections**: Introduces skip connections or shortcuts to jump over some layers. This helps in mitigating the vanishing gradient problem and allows training deeper networks.
- **Bottleneck Design**: Uses bottleneck designs which reduce the number of parameters and computational complexity while maintaining performance.

## Architecture:
The architecture consists of several residual blocks, each containing three convolutional layers. The first layer has a stride of 2 to downsample the input, and the remaining layers have a stride of 1.

## Applications:
ResNet18 has been widely used in various computer vision tasks such as image classification, object detection, and segmentation due to its robustness and effectiveness.

## Advantages:
- Easier optimization compared to shallower networks.
- Strong performance on image recognition tasks.
- Can be easily modified for different tasks by changing the final layers.

## Disadvantages:
- High memory consumption due to the large number of parameters.
- Slower training time compared to shallower networks without residual connections.

Overall, ResNet18 is a powerful and versatile CNN architecture that has significantly advanced the state of the art in image recognition tasks.
