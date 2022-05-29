#!/bin/bash
FILENAME=$1

# MODE be one of ['lite_train_lite_infer' 'lite_train_whole_infer' 'whole_train_whole_infer',  
#                 'whole_infer', 'klquant_whole_infer',
#                 'cpp_infer', 'serving_infer',  'lite_infer']

MODE=$2

dataline=$(cat ${FILENAME})
# parser params
IFS=$'\n'
lines=(${dataline})

function func_parser_key(){
    strs=$1
    IFS=":"
    array=(${strs})
    tmp=${array[0]}
    echo ${tmp}
}

function func_parser_value(){
    strs=$1
    IFS=":"
    array=(${strs})
    if [ ${#array[*]} = 2 ]; then
        echo ${array[1]}
    else
    	IFS="|"
    	tmp="${array[1]}:${array[2]}"
        echo ${tmp}
    fi
}

function func_get_url_file_name(){
    strs=$1
    IFS="/"
    array=(${strs})
    tmp=${array[${#array[@]}-1]}
    echo ${tmp}
}

model_name=$(func_parser_value "${lines[1]}")

if [ ${MODE} = "cpp_infer" ];then
   if [[ $FILENAME == *infer_cpp_linux_gpu_cpu.txt ]];then
	cpp_type=$(func_parser_value "${lines[2]}")
	cls_inference_model_dir=$(func_parser_value "${lines[3]}")
	det_inference_model_dir=$(func_parser_value "${lines[4]}")
	cls_inference_url=$(func_parser_value "${lines[5]}")
	det_inference_url=$(func_parser_value "${lines[6]}")

	if [[ $cpp_type == "cls" ]];then
	    eval "wget -nc $cls_inference_url"
	    tar xf "${model_name}_inference.tar"
	    eval "mv inference $cls_inference_model_dir"
	    mkdir data
	    cd data
    	    rm -rf ILSVRC2012
    	    wget -nc https://paddle-imagenet-models-name.bj.bcebos.com/data/whole_chain/whole_chain_infer.tar
    	    tar xf whole_chain_infer.tar
    	    ln -s whole_chain_infer ILSVRC2012
	    cd ..
	else
	    echo "Wrong cpp type in config file in line 3. only support cls"
	fi
	exit 0
   else
	echo "use wrong config file"
	exit 1
   fi
fi

model_name=$(func_parser_value "${lines[1]}")
model_url_value=$(func_parser_value "${lines[35]}")
model_url_key=$(func_parser_key "${lines[35]}")

if [[ $FILENAME == *use_dali* ]];then
    python_name=$(func_parser_value "${lines[2]}")
    ${python_name} -m pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/nightly --upgrade nvidia-dali-nightly-cuda102
fi

if [ ${MODE} = "lite_train_lite_infer" ] || [ ${MODE} = "lite_train_whole_infer" ];then
    if [[ $model_name == *lvvit* ]];then
        data_name="whole_chain_little_train_lvvit.tar"
    else
        data_name="whole_chain_little_train.tar"
    fi

    # pretrain lite train data
    mkdir -p data
    cd data
    rm -rf ILSVRC2012 whole_chain_little_train
    wget -nc https://passl.bj.bcebos.com/tipc/$data_name
    tar xf $data_name
    ln -s whole_chain_little_train ILSVRC2012
    cd ../../
fi
