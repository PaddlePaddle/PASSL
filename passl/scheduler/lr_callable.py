import math

class LRCallable(object):
    pass

class CosineWithFixLR(LRCallable):
    def __init__(self,
                 learning_rate,
                 step_each_epoch,
                 epochs,
                 decay_unit='epoch',
                 **kwargs):
        self.step_each_epoch = step_each_epoch
        self.epochs = epochs
        self.lr = learning_rate
        self.decay_unit = decay_unit

    def __call__(self, group, epoch):
        """Decay the learning rate based on schedule"""
        cur_lr = self.lr * 0.5 * (1. + math.cos(math.pi * epoch / self.epochs))
        if 'fix_lr' in group and group['fix_lr']:
            group['lr'] = self.lr
        else:
            group['lr'] = cur_lr