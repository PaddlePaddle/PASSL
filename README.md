# PASSL
under develop...

## single gpu train
```
python tools/train.py -c configs/moco_v2_r50.yaml
```

## multiple gpus train

```
python tools/train.py -c configs/moco_v2_r50.yaml --num-gpus 8
```
or
```
CUDA_VISIBLE_DEVIVICE=4,5,6,7 python tools/train.py -c configs/moco_v2_r50.yaml --num-gpus 4
```
