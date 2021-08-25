## Installation

<!-- TOC -->

- [Requirements](#requirements)
- [Install PASSL](#install-PASSL)


<!-- TOC -->

### Requirements

- Linux (Windows is not officially supported)
- Python 3.6+
- Paddle 2.1.0+
- CUDA 9.2+ (If you build PyTorch from source, CUDA 9.0 is also compatible)
- GCC 5+
- Numpy
- ftfy
- regex 
- boto3
- [visualdl](https://github.com/PaddlePaddle/VisualDL)


### Install PASSL

a. Clone the PASSL repository.

```
git clone https://github.com/PaddlePaddle/PASSL.git
```

b. Install Paddle following the [official instructions](https://www.paddlepaddle.org.cn/), e.g.,

```shell
python -m pip install paddlepaddle-gpu -i https://mirror.baidu.com/pypi/simple
```

Note: Make sure that your compilation CUDA version and runtime CUDA version match.
You can check the supported CUDA version for precompiled packages on the [Paddle website](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/conda/linux-conda.html).

`E.g.1` If you have CUDA 10.1 installed under `/usr/local/cuda` and would like to install Paddle 2.1.0,
you need to install the prebuilt PaddlePaddle with CUDA 10.1.

```shell
python -m pip install paddlepaddle-gpu==2.1.1.post101 -f https://paddlepaddle.org.cn/whl/mkl/stable.html
```


If you build PaddlePaddle from source instead of installing the prebuilt package, you can use more CUDA versions such as 9.0.
