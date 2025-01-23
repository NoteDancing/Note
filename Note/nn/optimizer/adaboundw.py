""" AdaBoundW
Copyright 2025 NoteDance
"""
import tensorflow as tf
from keras.src.optimizers import optimizer
import math


class AdaBoundW(optimizer.Optimizer):
    def __init__(
        self,
        learning_rate=1e-3,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8,
        weight_decay=0,
        final_lr=0.1,
        gamma=1e-3,
        amsbound=False,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        gradient_accumulation_steps=None,
        name="adaboundw",
        **kwargs,
    ):
        super().__init__(
            learning_rate=learning_rate,
            name=name,
            weight_decay=None,
            clipnorm=clipnorm,
            clipvalue=clipvalue,
            global_clipnorm=global_clipnorm,
            use_ema=use_ema,
            ema_momentum=ema_momentum,
            ema_overwrite_frequency=ema_overwrite_frequency,
            loss_scale_factor=loss_scale_factor,
            gradient_accumulation_steps=gradient_accumulation_steps,
            **kwargs,
        )
        self.weight_decay_ = weight_decay
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.final_lr = final_lr
        self.gamma = gamma
        self.amsbound = amsbound
    
    def __setstate__(self, state):
        self.__dict__.update(state)
        self.amsbound = False

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self.exp_avg = []
        self.exp_avg_sq = []
        if self.amsbound:
            self.max_exp_avg_sq = []
        self.base_lr = self._learning_rate
        self.step = []
        for var in var_list:
            self.exp_avg.append(
                self.add_variable_from_reference(
                    reference_variable=var, name="exp_avg"
                )
            )
            self.exp_avg_sq.append(
                self.add_variable_from_reference(
                    reference_variable=var, name="exp_avg_sq"
                )
            )
            if self.amsbound:
                self.max_exp_avg_sq.append(
                    self.add_variable_from_reference(
                        reference_variable=var, name="max_exp_avg_sq"
                    )
                )
            self.step.append(0)

    def update_step(self, gradient, variable, learning_rate):
        lr = tf.cast(learning_rate, variable.dtype)
        
        if tf.keras.backend.is_sparse(gradient):
            raise RuntimeError(
                'Adam does not support sparse gradients, please consider SparseAdam instead')
        
        exp_avg = self.exp_avg[self._get_variable_index(variable)]
        exp_avg_sq = self.exp_avg_sq[self._get_variable_index(variable)]
        if self.amsbound:
            max_exp_avg_sq = self.max_exp_avg_sq[self._get_variable_index(variable)]
        
        self.step[self._get_variable_index(variable)] += 1

        if self.weight_decay_ != 0:
            gradient = gradient + self.weight_decay_ * variable

        # Decay the first and second moment running average coefficient
        exp_avg.assign(self.beta1 * exp_avg + (1 - self.beta1) * gradient)
        exp_avg_sq.assign(self.beta2 * exp_avg_sq + (1 - self.beta2) * tf.square(gradient))
        if self.amsbound:
            # Maintains the maximum of all 2nd moment running avg. till now
            max_exp_avg_sq.assign(tf.maximum(max_exp_avg_sq, exp_avg_sq))
            # Use the max. for normalizing running avg. of gradient
            denom = tf.sqrt(max_exp_avg_sq) + self.epsilon
        else:
            denom = tf.sqrt(exp_avg_sq) + self.epsilon
        
        bias_correction1 = 1 - self.beta1 ** self.step[self._get_variable_index(variable)]
        bias_correction2 = 1 - self.beta2 ** self.step[self._get_variable_index(variable)]
        step_size = lr * math.sqrt(bias_correction2) / bias_correction1
        
        # Applies bounds on actual learning rate
        # lr_scheduler cannot affect final_lr, this is a workaround to apply lr decay
        final_lr = self.final_lr * lr / self.base_lr
        lower_bound = final_lr * (1 - 1 / (self.gamma * self.step[self._get_variable_index(variable)] + 1))
        upper_bound = final_lr * (1 + 1 / (self.gamma * self.step[self._get_variable_index(variable)]))
        step_size = tf.fill(denom.shape, step_size)
        step_size = step_size / denom
        step_size = tf.clip_by_value(step_size, clip_value_min=lower_bound, clip_value_max=upper_bound)
        step_size = step_size * exp_avg

        if self.weight_decay_ != 0:
            decayed_weights = variable * self.weight_decay_
            variable.assign_add(-step_size)
            variable.assign_sub(decayed_weights)
        else:
            variable.assign_add(-step_size)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "weight_decay": self.weight_decay_,
                "beta_1": self.beta_1,
                "beta_2": self.beta_2,
                "epsilon": self.epsilon,
                "final_lr": self.final_lr,
                "gamma": self.gamma,
                "amsbound": self.amsbound,
            }
        )
        return config