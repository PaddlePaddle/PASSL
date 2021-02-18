from .hook import Hook
from .builder import HOOKS


@HOOKS.register()
class OptimizerHook(Hook):
    def train_iter_end(self, trainer):
        trainer.optimizer.clear_grad()
        loss = 0
        for key, value in trainer.outputs.items():
            if 'loss' in key:
                loss += value
        loss.backward()

        trainer.optimizer.step()

        if 'loss' not in trainer.outputs:
            trainer.outputs['loss'] = loss
