from .hook import Hook
from .builder import HOOKS


@HOOKS.register()
class LRSchedulerHook(Hook):
    def train_iter_end(self, trainer):
        trainer.lr_scheduler.step()
