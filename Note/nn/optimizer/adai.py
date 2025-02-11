""" Adai
Implements Adaptive Inertia Estimation (Adai) algorithm.
It is proposed in the ICML2022 Oral paper  
`Adaptive Inertia: Disentangling the Effects of Adaptive Learning Rate and Momentum`.
https://arxiv.org/abs/2006.15815

Copyright 2025 NoteDance
"""
import tensorflow as tf
from keras.src.optimizers import optimizer


class Adai(optimizer.Optimizer):
    def __init__(
        self,
        learning_rate,
        beta0=0.1,
        beta2=0.99,
        epsilon=1e-03,
        weight_decay=0,
        decoupled=False,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        gradient_accumulation_steps=None,
        name="adai",
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
        self.beta0 = beta0
        self.beta2 = beta2
        self.epsilon = epsilon
        self.decoupled = decoupled
    
    def __setstate__(self, state):
        self.__dict__.update(state)
        self.decoupled = False

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self.exp_avg = []
        self.exp_avg_sq = []
        self.beta1_prod = []
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
            self.beta1_prod.append(
                self.add_variable_from_reference(
                    reference_variable=var, name="beta1_prod", initializer="ones"
                )
            )
            self.step.append(0)
    
    def _backend_update_step(self, grads, trainable_variables, learning_rate):
        """Collective update_step that can be overridden by the backend.
    
        It is overridden by torch for performance reasons, and
        by TF to support tf.distribute.
        """
        self.update_step(grads, trainable_variables, learning_rate)

    def update_step(self, grads, trainable_variables, learning_rate):
        lr = learning_rate
        
        param_size = 0
        exp_avg_sq_hat_sum = 0.
        
        for p, g in zip(trainable_variables, grads):
            param_size += tf.size(p)
            
            self.step[self._get_variable_index(p)] += 1
            
            exp_avg_sq = self.exp_avg_sq[self._get_variable_index(p)]
            
            bias_correction2 = 1 - self.beta2 ** self.step[self._get_variable_index(p)]

            if self.weight_decay_ != 0 and self.decoupled == False:
                g.assign_add(p * self.weight_decay_)
            elif self.weight_decay_ != 0 and self.decoupled == True:
                p.assign(p * (1 - lr * self.weight_decay_))
                
            exp_avg_sq.assign(self.beta2 * exp_avg_sq + (1 - self.beta2) * tf.square(g))
            
            exp_avg_sq_hat_sum += tf.reduce_sum(exp_avg_sq) / bias_correction2
        
        # Calculate the mean of all elements in exp_avg_sq_hat
        exp_avg_sq_hat_mean = exp_avg_sq_hat_sum / tf.get_static_value(param_size)
        
        for p, g in zip(trainable_variables, grads):
            exp_avg = self.exp_avg[self._get_variable_index(p)]
            exp_avg_sq = self.exp_avg_sq[self._get_variable_index(p)]
            beta1_prod = self.beta1_prod[self._get_variable_index(p)]
            
            bias_correction2 = 1 - self.beta2 ** self.step[self._get_variable_index(p)]

            exp_avg_sq_hat = exp_avg_sq / bias_correction2
            beta1 = tf.clip_by_value(1.0 - self.beta0 * (exp_avg_sq_hat / exp_avg_sq_hat_mean),
                           clip_value_min=0.,
                           clip_value_max=1.0 - self.epsilon)
            
            beta1_prod.assign(beta1_prod * beta1)
            bias_correction1 = 1 - beta1_prod
            
            exp_avg.assign(exp_avg * beta1 + (1 - beta1) * g)
            exp_avg_hat = exp_avg / bias_correction1
            
            p.assign_add(exp_avg_hat * -lr)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "weight_decay": self.weight_decay_,
                "beta0": self.beta0,
                "beta2": self.beta2,
                "epsilon": self.epsilon,
                "decoupled": self.decoupled,
            }
        )
        return config