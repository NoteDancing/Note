class LambdaCallback:
    def __init__(self,
                 on_train_begin=None,
                 on_train_end=None,
                 on_epoch_begin=None,
                 on_epoch_end=None,
                 on_episode_begin=None,
                 on_episode_end=None,
                 on_batch_begin=None,
                 on_batch_end=None,
                 on_test_begin=None,
                 on_test_end=None):
        self.on_train_begin = on_train_begin
        self.on_train_end = on_train_end
        self.on_epoch_begin = on_epoch_begin
        self.on_epoch_end = on_epoch_end
        self.on_episode_begin = on_episode_begin
        self.on_episode_end = on_episode_end
        self.on_batch_begin = on_batch_begin
        self.on_batch_end = on_batch_end
        self.on_test_begin = on_test_begin
        self.on_test_end = on_test_end
