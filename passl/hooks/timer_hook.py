import time

from .hook import Hook
from .builder import HOOKS
from ..utils import AverageMeter


@HOOKS.register()
class IterTimerHook(Hook):
    def epoch_begin(self, runner):
        self.t = time.time()

    def iter_begin(self, runner):
        if 'data_time' not in runner.logs:
            runner.logs['data_time'] = AverageMeter('data_time')
        runner.logs['data_time'].update(time.time() - self.t)

    def iter_end(self, runner):
        if 'time' not in runner.logs:
            runner.logs['time'] = AverageMeter('time')
        runner.logs['time'].update(time.time() - self.t)
        self.t = time.time()
