python tools/passl2ppclas/convert.py \
    --checkpoint=./pretrain/epoch_100_pretrain.pdparams \
    --output=./pretrain/swav_ResNet50_100.pdparams --type="res50" \
    --ppclas=./tools/passl2ppclas/ppclas_res50_keys.txt \
    --passl=./tools/passl2ppclas/swav_ResNet50.txt
