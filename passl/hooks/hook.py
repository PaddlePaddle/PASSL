class Hook:
    def run_begin(self, trainer):
        pass

    def run_end(self, trainer):
        pass

    def epoch_begin(self, trainer):
        pass

    def epoch_end(self, trainer):
        pass

    def iter_begin(self, trainer):
        pass

    def iter_end(self, trainer):
        pass

    def train_epoch_begin(self, trainer):
        self.epoch_begin(trainer)

    def val_epoch_begin(self, trainer):
        self.epoch_begin(trainer)

    def train_epoch_end(self, trainer):
        self.epoch_end(trainer)

    def val_epoch_end(self, trainer):
        self.epoch_end(trainer)

    def train_iter_begin(self, trainer):
        self.iter_begin(trainer)

    def val_iter_begin(self, trainer):
        self.iter_begin(trainer)

    def train_iter_end(self, trainer):
        self.iter_end(trainer)

    def val_iter_end(self, trainer):
        self.iter_end(trainer)

    def every_n_epochs(self, trainer, n):
        return (trainer.current_epoch + 1) % n == 0 if n > 0 else False

    def every_n_inner_iters(self, trainer, n):
        return (trainer.inner_iter + 1) % n == 0 if n > 0 else False

    def every_n_iters(self, trainer, n):
        return (trainer.iter + 1) % n == 0 if n > 0 else False

    def end_of_epoch(self, trainer):
        return trainer.inner_iter + 1 == trainer.iters_per_epoch
