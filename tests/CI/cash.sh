set -e

export passl_path=/paddle/PASSL/tests/CI
export log_path=/paddle/log_passl
passl_gpu_model_list=( \
	moco_v1_r50_pretrain \
	moco_v1_r50_linear \
	moco_v2_r50_pretrain \
	moco_v2_r50_linear \
	simclr_r50_IM_pretrain \
	simclr_r50_IM_linear \
	byol_r50_IM_pretrain \
	byol_r50_IM_linear \
	simsiam_r50_IM_pretrain \
	simsiam_r50_IM_linear \
	swav_r50_IM_pretrain \
	swav_r50_IM_linear \
)

function moco_v1_r50_pretrain(){
	cd ${passl_path}
	rm -rf log
	bash ./ssl/moco/moco_v1_r50_pretrain.sh
  loss=`tail log/workerlog.0 | grep "50/200" | cut -d " " -f17 `
  check_result 1.3840e+00 ${loss%?} $FUNCNAME}

function moco_v1_r50_linear(){
	cd ${passl_path}
	rm -rf log
	bash ./ssl/moco/moco_v1_r50_linear.sh
  loss=`tail log/workerlog.0 | grep "50/200" | cut -d " " -f17 `
  check_result 1.3840e+00 ${loss%?} $FUNCNAME}

function moco_v2_r50_pretrain(){
	cd ${passl_path}
	rm -rf log
	bash ./ssl/moco/moco_v2_r50_pretrain.sh
  loss=`tail log/workerlog.0 | grep "50/200" | cut -d " " -f17 `
  check_result 7.0774e+00 ${loss%?} $FUNCNAME}

function moco_v2_r50_linear(){
	cd ${passl_path}
	rm -rf log
	bash ./ssl/moco/moco_v2_r50_linear.sh
  loss=`tail log/workerlog.0 | grep "50/200" | cut -d " " -f17 `
  check_result 7.0774e+00 ${loss%?} $FUNCNAME}

function simclr_r50_IM_pretrain(){
	cd ${passl_path}
	rm -rf log
	bash ./ssl/simclr/simclr_r50_IM_pretrain.sh
  loss=`tail log/workerlog.0 | grep "50/200" | cut -d " " -f17 `
  check_result 1.3840e+00 ${loss%?} $FUNCNAME}

function simclr_r50_IM_linear(){
	cd ${passl_path}
	rm -rf log
	bash ./ssl/simclr/simclr_r50_IM_linear.sh
  loss=`tail log/workerlog.0 | grep "50/200" | cut -d " " -f17 `
  check_result 1.3840e+00 ${loss%?} $FUNCNAME}

function byol_r50_IM_pretrain(){
	cd ${passl_path}
	rm -rf log
	bash ./ssl/byol/byol_r50_IM_pretrain.sh
  loss=`tail log/workerlog.0 | grep "50/200" | cut -d " " -f17 `
  check_result 1.3840e+00 ${loss%?} $FUNCNAME}

function byol_r50_IM_linear(){
	cd ${passl_path}
	rm -rf log
	bash ./ssl/byol/byol_r50_IM_linear.sh
  loss=`tail log/workerlog.0 | grep "50/200" | cut -d " " -f17 `
  check_result 1.3840e+00 ${loss%?} $FUNCNAME}

function simsiam_r50_IM_pretrain(){
	cd ${passl_path}
	rm -rf log
	bash ./ssl/simsiam/simsiam_r50_IM_pretrain.sh
  loss=`tail log/workerlog.0 | grep "50/100" | cut -d " " -f17 `
  check_result 1.3840e+00 ${loss%?} $FUNCNAME}

function simsiam_r50_IM_linear(){
	cd ${passl_path}
	rm -rf log
	bash ./ssl/simsiam/simsiam_r50_IM_linear.sh
  loss=`tail log/workerlog.0 | grep "50/100" | cut -d " " -f17 `
  check_result 1.3840e+00 ${loss%?} $FUNCNAME}

function swav_r50_IM_pretrain(){
	cd ${passl_path}
	rm -rf log
	bash ./ssl/swav/swav_r50_IM_pretrain.sh
  loss=`tail log/workerlog.0 | grep "50/100" | cut -d " " -f17 `
  check_result 1.3840e+00 ${loss%?} $FUNCNAME}

function swav_r50_IM_linear(){
	cd ${passl_path}
	rm -rf log
	bash ./ssl/swav/swav_r50_IM_linear.sh
  loss=`tail log/workerlog.0 | grep "50/100" | cut -d " " -f17 `
  check_result 1.3840e+00 ${loss%?} $FUNCNAME}