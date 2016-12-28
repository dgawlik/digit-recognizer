# digit-recognizer

Neural Networks for Kaggle MNIST Classification tutorial. Original [dataset](http://yann.lecun.com/exdb/mnist/) contains 70000 28x28 images, Kaggle splits them into 32000 training set and 28000 test set, what results in lower classification accuracy than those given on site.

### Networks

**NN**

| Layer | Dimensions |
|-------|------------|
|Input  | (100, 784) |
|Batch Normalization| |
| PRelu | 800 |
| Dropout 0.5| |
| Batch Normalization| |
| PRelu | 400 |
| Dropout 0.5 |
| Softmax| 10 |

**CNN**

| Layer | Dimensions |
|-------|------------|
|Input  | (100,28,28,1) |
|Batch Normalization| |
|Convolution| (5,5,1,20) |
|Batch Normalization|
|PRelu|
|Max Pool| (1,2,2,1)|
|Convolution| (5,5,20,40)|
|Batch Normalization |
|PRelu|
|Max Pool| (1,2,2,1)|
|Batch Normalization |
|Fully Connected PRelu| 1600|
|Dropout 0.2|
|Batch Normalization|
|Fully Connected PRelu| 400 |
|Dropout 0.2|
|Softmax| 10|

### Useful links

* [Batch Normalization](https://arxiv.org/pdf/1502.03167v3.pdf)
* [PRelu, He Initialization](https://arxiv.org/abs/1502.01852)
* [CNN](http://cs231n.github.io/)
* [Maxout, Dropout](http://www.jmlr.org/proceedings/papers/v28/goodfellow13.pdf)
* [Caffe Benchmarks](https://github.com/ducha-aiki/caffenet-benchmark)
