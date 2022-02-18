# Large Scale Training with PASSL (Mix-Precision, LARS, and ZeRO)

## Using Mixed-Precision

Users can use **[Mixed-Precision](https://arxiv.org/abs/1710.03740)** to save GPU memory and increase throughput.
To use Mixed-Precision, one can set parameters in the configure file:
```
AMP:
   level: 'O1'
   save_dtype: 'float32'
   optimizers: None
   scale_loss: 32768.0
   auto_cast:
     enable: True
     custom_black_list: ["reduce_mean", "reduce_sum",
                         "c_softmax_with_cross_entropy", "elementwise_div"]
     level: 'O1'
```

## Using ZeRO


**[ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054)** is a technique proposed by **Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, Yuxiong He**.  To train a huge models with billions of parameters faces severe challenges in GPU memory comsumption. ZeRO can largely eliminate GPU memory redundancies by partitioning the optimizer states, gradients, and parameters across multiple devices.

Note: The latest Paddle develop version should be install when using sharding:
https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/develop/install/pip/linux-pip.html

Users can use ZeRO by setting the following parameters in configure file:
```
sharding:
   sharding_stage: 2
   offload: False
   accumulate_grad: False
```
**Note**: To train large models, users need to combine ZeRO with mixed-precision training.  
