from tqdm import tqdm
from collections import OrderedDict
import paddle
import paddle.distributed as dist
from .hook import Hook
from .builder import HOOKS
from ..utils.logger import get_logger
from ..utils.misc import AverageMeter


@HOOKS.register()
class EvaluateHook(Hook):
    def __init__(self, eval_kargs=None):
        if eval_kargs is None:
            self.eval_kargs = {}
        else:
            self.eval_kargs = eval_kargs

    def train_epoch_end(self, trainer):
        if not hasattr(trainer, 'val_dataloader'):
            from ..datasets.builder import build_dataloader
            trainer.val_dataloader = build_dataloader(
                trainer.cfg.dataloader.val)
        logger = get_logger()

        logger.info(
            'start evaluate on epoch {} ..'.format(trainer.current_epoch + 1))
        dataloader = trainer.val_dataloader
        model = trainer.model
        total_samples = len(dataloader.dataset)
        accum_samples = 0
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        trainer.model.eval()
        outs = OrderedDict()
        for data in tqdm(dataloader):
            if isinstance(data, paddle.Tensor):
                batch_size = data.shape[0]
            elif isinstance(data, (list, tuple)):
                batch_size = data[0].shape[0]
            else:
                raise TypeError('unknown type of data')

            outputs = model(*data)

            current_samples = batch_size * world_size
            accum_samples += current_samples

            for k, v in outputs.items():
                if world_size > 1:
                    v_list = []
                    dist.all_gather(v_list, v)
                    v = paddle.concat(v_list, 0)
                    if accum_samples > total_samples:
                        v = v[:total_samples + current_samples -
                              accum_samples, ...]
                        current_samples = total_samples + current_samples - accum_samples

                res = dataloader.dataset.evaluate(v, *data, **self.eval_kargs)
                for k, v in res.items():
                    if k not in outs:
                        outs[k] = AverageMeter(k, ':6.3f')
                    outs[k].update(v, current_samples)

        log_str = f'Validate Epoch [{trainer.current_epoch}] '
        log_items = []
        for name, val in outs.items():
            if isinstance(val, AverageMeter):
                val = str(val)
            log_items.append(val)
        log_str += ', '.join(log_items)
        logger.infor(log_str)

        trainer.model.train()
