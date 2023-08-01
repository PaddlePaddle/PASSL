# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/usr/bin/env bash
set -e

export passl_path=/paddle/PASSL/tests/CI
export data_path=/passl_data
export pretrained_path=/passl_pretrained
export log_path=/paddle/log_passl
mkdir -p ${log_path}

function before_hook() {
    echo "=============================paddle commit============================="
    python -c "import paddle;print(paddle.__git_commit__)"

    # install requirements
    cd /paddle/PASSL/
    echo ---------- install passl ----------
    export http_proxy=${proxy};
    export https_proxy=${proxy};
    python -m pip install -r requirements.txt
    python setup.py develop
    python -m pip install attr attrs --force-reinstall

    
    echo ---------- ln passl_data start ----------
    cd ${passl_path}
    rm -rf dataset
    ln -s ${data_path} ./dataset
    echo ---------- ln passl_data done ---------- 

    echo ---------- ln passl_pretrained start ----------
    cd ${passl_path}
    rm -rf pretrained
    ln -s ${pretrained_path} ./pretrained
    echo ---------- ln passl_pretrained done ----------
}

main() {
    before_hook
}

main$@
