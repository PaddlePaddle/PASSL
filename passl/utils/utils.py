# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import glob
import yaml
import random
import tempfile
import platform
import subprocess
import contextlib
from urllib.parse import urlparse, unquote

import numpy as np

import cv2
import paddle
import passl
from passl.utils import logger
from passl.utils.download import download_file_and_uncompress



class NoAliasDumper(yaml.SafeDumper):
    """
    Avoid yaml anchor
    """
    def ignore_aliases(self):
        return True


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def _get_user_home():
    return os.path.expanduser('~')


def _get_passl_home():
    if 'PASSL_HOME' in os.environ:
        home_path = os.environ['PASSL_HOME']
        if os.path.exists(home_path):
            if os.path.isdir(home_path):
                return home_path
            else:
                logger.warning('PASSL_HOME {} is a file!'.format(home_path))
        else:
            return home_path
    return os.path.join(_get_user_home(), '.paddleseg')


def _get_sub_home(directory):
    home = os.path.join(_get_passl_home(), directory)
    if not os.path.exists(home):
        os.makedirs(home, exist_ok=True)
    return home


USER_HOME = _get_user_home()
PASSL_HOME = _get_passl_home()
DATA_HOME = _get_sub_home('dataset')
TMP_HOME = _get_sub_home('tmp')
PRETRAINED_MODEL_HOME = _get_sub_home('pretrained_model')



def set_seed(seed=None):
    if seed is not None:
        paddle.seed(seed)
        np.random.seed(seed)
        random.seed(seed)


def show_env_info():
    env_info = get_sys_env()
    info = ['{}: {}'.format(k, v) for k, v in env_info.items()]
    info = '\n'.join(['', format('Environment Information', '-^48s')] + info +
                     ['-' * 48])
    logger.info(info)


def show_cfg_info(config):
    msg = '\n---------------Config Information---------------\n'
    ordered_module = ('batch_size', 'epochs', 'train_dataset', 'val_dataset',
                      'optimizer', 'lr_scheduler', 'loss', 'model')
    all_module = set(config.dic.keys())
    for module in ordered_module:
        if module in config.dic:
            module_dic = {module: config.dic[module]}
            msg += str(yaml.dump(module_dic, Dumper=NoAliasDumper))
            all_module.remove(module)
    for module in all_module:
        module_dic = {module: config.dic[module]}
        msg += str(yaml.dump(module_dic, Dumper=NoAliasDumper))
    msg += '------------------------------------------------\n'
    logger.info(msg)


def set_device(device):
    env_info = get_sys_env()
    if device == 'gpu' and env_info['Paddle compiled with cuda'] \
        and env_info['GPUs used']:
        place = 'gpu'
    elif device == 'xpu' and paddle.is_compiled_with_xpu():
        place = 'xpu'
    elif device == 'npu' and "npu" in paddle.device.get_all_custom_device_type(
    ):
        place = 'npu'
    elif device == 'mlu' and paddle.is_compiled_with_mlu():
        place = 'mlu'
    else:
        place = 'cpu'
    paddle.set_device(place)
    logger.info("Set device: {}".format(place))


def convert_sync_batchnorm(model, device):
    # Convert bn to sync_bn when use multi GPUs
    env_info = get_sys_env()
    if device == 'gpu' and env_info['Paddle compiled with cuda'] \
        and env_info['GPUs used'] and paddle.distributed.ParallelEnv().nranks > 1:
        model = paddle.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        logger.info("Convert bn to sync_bn")
    return model


def set_cv2_num_threads(num_workers):
    # Limit cv2 threads if too many subprocesses are spawned.
    # This should reduce resource allocation and thus boost performance.
    nranks = paddle.distributed.ParallelEnv().nranks
    if nranks >= 8 and num_workers >= 8:
        logger.warning("The number of threads used by OpenCV is " \
            "set to 1 to improve performance.")
        cv2.setNumThreads(1)


def worker_init_fn(worker_id):
    np.random.seed(random.randint(0, 100000))

@contextlib.contextmanager
def generate_tempdir(directory: str=None, **kwargs):
    '''Generate a temporary directory'''
    directory = TMP_HOME if not directory else directory
    with tempfile.TemporaryDirectory(dir=directory, **kwargs) as _dir:
        yield _dir


def load_entire_model(model, pretrained):
    if pretrained is not None:
        load_pretrained_model(model, pretrained)
    else:
        logger.warning('Weights are not loaded for {} model since the '
                       'path of weights is None'.format(
                           model.__class__.__name__))


def download_pretrained_model(pretrained_model):
    """
    Download pretrained model from url.
    Args:
        pretrained_model (str): the url of pretrained weight
    Returns:
        str: the path of pretrained weight
    """
    assert urlparse(pretrained_model).netloc, "The url is not valid."

    pretrained_model = unquote(pretrained_model)
    savename = pretrained_model.split('/')[-1]
    if not savename.endswith(('tgz', 'tar.gz', 'tar', 'zip')):
        savename = pretrained_model.split('/')[-2]
        filename = pretrained_model.split('/')[-1]
    else:
        savename = savename.split('.')[0]
        filename = 'model.pdparams'

    with generate_tempdir() as _dir:
        pretrained_model = download_file_and_uncompress(
            pretrained_model,
            savepath=_dir,
            cover=False,
            extrapath=PRETRAINED_MODEL_HOME,
            extraname=savename,
            filename=filename)
        pretrained_model = os.path.join(pretrained_model, filename)
    return pretrained_model


def load_pretrained_model(model, pretrained_model):
    if pretrained_model is not None:
        logger.info('Loading pretrained model from {}'.format(pretrained_model))

        if urlparse(pretrained_model).netloc:
            pretrained_model = download_pretrained_model(pretrained_model)

        if os.path.exists(pretrained_model):
            para_state_dict = paddle.load(pretrained_model)

            model_state_dict = model.state_dict()
            keys = model_state_dict.keys()
            num_params_loaded = 0
            for k in keys:
                if k not in para_state_dict:
                    logger.warning("{} is not in pretrained model".format(k))
                elif list(para_state_dict[k].shape) != list(model_state_dict[k]
                                                            .shape):
                    logger.warning(
                        "[SKIP] Shape of pretrained params {} doesn't match.(Pretrained: {}, Actual: {})"
                        .format(k, para_state_dict[k].shape, model_state_dict[k]
                                .shape))
                else:
                    model_state_dict[k] = para_state_dict[k]
                    num_params_loaded += 1
            model.set_dict(model_state_dict)
            logger.info("There are {}/{} variables loaded into {}.".format(
                num_params_loaded,
                len(model_state_dict), model.__class__.__name__))

        else:
            raise ValueError('The pretrained model directory is not Found: {}'.
                             format(pretrained_model))
    else:
        logger.info(
            'No pretrained model to load, {} will be trained from scratch.'.
            format(model.__class__.__name__))


def resume(model, optimizer, resume_model):
    if resume_model is not None:
        logger.info('Resume model from {}'.format(resume_model))
        if os.path.exists(resume_model):
            resume_model = os.path.normpath(resume_model)
            ckpt_path = os.path.join(resume_model, 'model.pdparams')
            para_state_dict = paddle.load(ckpt_path)
            ckpt_path = os.path.join(resume_model, 'model.pdopt')
            opti_state_dict = paddle.load(ckpt_path)
            model.set_state_dict(para_state_dict)
            optimizer.set_state_dict(opti_state_dict)

            epoch = resume_model.split('_')[-1]
            epoch = int(epoch)
            return epoch
        else:
            raise ValueError(
                'Directory of the model needed to resume is not Found: {}'.
                format(resume_model))
    else:
        logger.info('No model needed to resume.')


def worker_init_fn(worker_id):
    np.random.seed(random.randint(0, 100000))


def get_image_list(image_path):
    """Get image list"""
    valid_suffix = [
        '.JPEG', '.jpeg', '.JPG', '.jpg', '.BMP', '.bmp', '.PNG', '.png'
    ]
    image_list = []
    image_dir = None
    if os.path.isfile(image_path):
        if os.path.splitext(image_path)[-1] in valid_suffix:
            image_list.append(image_path)
        else:
            image_dir = os.path.dirname(image_path)
            with open(image_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if len(line.split()) > 1:
                        line = line.split()[0]
                    image_list.append(os.path.join(image_dir, line))
    elif os.path.isdir(image_path):
        image_dir = image_path
        for root, dirs, files in os.walk(image_path):
            for f in files:
                if '.ipynb_checkpoints' in root:
                    continue
                if f.startswith('.'):
                    continue
                if os.path.splitext(f)[-1] in valid_suffix:
                    image_list.append(os.path.join(root, f))
    else:
        raise FileNotFoundError(
            '`--image_path` is not found. it should be a path of image, or a file list containing image paths, or a directory including images.'
        )

    if len(image_list) == 0:
        raise RuntimeError(
            'There are not image file in `--image_path`={}'.format(image_path))

    return image_list, image_dir


class NoAliasDumper(yaml.SafeDumper):
    def ignore_aliases(self, data):
        return True


class CachedProperty(object):
    """
    A property that is only computed once per instance and then replaces itself with an ordinary attribute.

    The implementation refers to https://github.com/pydanny/cached-property/blob/master/cached_property.py .
        Note that this implementation does NOT work in multi-thread or coroutine senarios.
    """

    def __init__(self, func):
        super().__init__()
        self.func = func
        self.__doc__ = getattr(func, '__doc__', '')

    def __get__(self, obj, cls):
        if obj is None:
            return self
        val = self.func(obj)
        # Hack __dict__ of obj to inject the value
        # Note that this is only executed once
        obj.__dict__[self.func.__name__] = val
        return val


def get_in_channels(model_cfg):
    if 'backbone' in model_cfg:
        return model_cfg['backbone'].get('in_channels', None)
    else:
        return model_cfg.get('in_channels', None)


def set_in_channels(model_cfg, in_channels):
    model_cfg = model_cfg.copy()
    if 'backbone' in model_cfg:
        model_cfg['backbone']['in_channels'] = in_channels
    else:
        model_cfg['in_channels'] = in_channels
    return model_cfg



def _find_cuda_home():
    '''Finds the CUDA install path. It refers to the implementation of
    pytorch <https://github.com/pytorch/pytorch/blob/master/torch/utils/cpp_extension.py>.
    '''
    # Guess #1
    cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
    if cuda_home is None:
        # Guess #2
        try:
            which = 'where' if sys.platform == 'win32' else 'which'
            nvcc = subprocess.check_output([which,
                                            'nvcc']).decode().rstrip('\r\n')
            cuda_home = os.path.dirname(os.path.dirname(nvcc))
        except Exception:
            # Guess #3
            if sys.platform == 'win32':
                cuda_homes = glob.glob(
                    'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v*.*')
                if len(cuda_homes) == 0:
                    cuda_home = ''
                else:
                    cuda_home = cuda_homes[0]
            else:
                cuda_home = '/usr/local/cuda'
            if not os.path.exists(cuda_home):
                cuda_home = None
    return cuda_home


def _get_nvcc_info(cuda_home):
    if cuda_home is not None and os.path.isdir(cuda_home):
        try:
            nvcc = os.path.join(cuda_home, 'bin/nvcc')
            if not sys.platform == 'win32':
                nvcc = subprocess.check_output(
                    "{} -V".format(nvcc), shell=True).decode()
            else:
                nvcc = subprocess.check_output(
                    "\"{}\" -V".format(nvcc), shell=True).decode()
            nvcc = nvcc.strip().split('\n')[-1]
        except subprocess.SubprocessError:
            nvcc = "Not Available"
    else:
        nvcc = "Not Available"
    return nvcc


def _get_gpu_info():
    try:
        gpu_info = subprocess.check_output(['nvidia-smi',
                                            '-L']).decode().strip()
        gpu_info = gpu_info.split('\n')
        for i in range(len(gpu_info)):
            gpu_info[i] = ' '.join(gpu_info[i].split(' ')[:4])
    except:
        gpu_info = ' Can not get GPU information. Please make sure CUDA have been installed successfully.'
    return gpu_info


def get_sys_env():
    """collect environment information"""
    env_info = {}
    env_info['platform'] = platform.platform()

    env_info['Python'] = sys.version.replace('\n', '')

    # TODO is_compiled_with_cuda() has not been moved
    compiled_with_cuda = paddle.is_compiled_with_cuda()
    env_info['Paddle compiled with cuda'] = compiled_with_cuda

    if compiled_with_cuda:
        cuda_home = _find_cuda_home()
        env_info['NVCC'] = _get_nvcc_info(cuda_home)
        # refer to https://github.com/PaddlePaddle/Paddle/blob/release/2.0-rc/paddle/fluid/platform/device_context.cc#L327
        v = paddle.get_cudnn_version()
        v = str(v // 1000) + '.' + str(v % 1000 // 100)
        env_info['cudnn'] = v
        if 'gpu' in paddle.get_device():
            gpu_nums = paddle.distributed.ParallelEnv().nranks
        else:
            gpu_nums = 0
        env_info['GPUs used'] = gpu_nums

        env_info['CUDA_VISIBLE_DEVICES'] = os.environ.get(
            'CUDA_VISIBLE_DEVICES')
        if gpu_nums == 0:
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
        env_info['GPU'] = _get_gpu_info()

    try:
        gcc = subprocess.check_output(['gcc', '--version']).decode()
        gcc = gcc.strip().split('\n')[0]
        env_info['GCC'] = gcc
    except:
        pass

    env_info['Passl'] = passl.__version__
    env_info['PaddlePaddle'] = paddle.__version__
    env_info['OpenCV'] = cv2.__version__

    return env_info


