""" PyTorch MADGRAD optimizer

MADGRAD: https://arxiv.org/abs/2101.11075

Code from: https://github.com/facebookresearch/madgrad
"""
# Copyright (c) NoteDance, Inc. and its affiliates.
#
# This source code is licensed under the Apache-2.0 license found in the
# LICENSE file in the root directory of this source tree.

import tensorflow as tf
from keras.src.optimizers import optimizer
import math


class MADGRAD(optimizer.Optimizer):
    def __init__(
        self,
        learning_rate=1e-2,
        epsilon=1e-6,
        momentum: float = 0.9,
        weight_decay=None,
        decoupled_decay: bool = False,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        gradient_accumulation_steps=None,
        name="madgrad",
        **kwargs,
    ):
        super().__init__(
            learning_rate=learning_rate,
            name=name,
            weight_decay=weight_decay,
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
        self.momentum = momentum
        self.epsilon = epsilon
        self.decoupled_decay = decoupled_decay
    
    @property
    def supports_memory_efficient_fp16(self) -> bool:
        return False

    @property
    def supports_flat_params(self) -> bool:
        return True

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self._grad_sum_sq = []
        self._s = []
        self._x0 = []
        self.step = []
        for var in var_list:
            self._grad_sum_sq.append(
                self.add_variable_from_reference(
                    reference_variable=var, name="grad_sum_s"
                )
            )
            self._s.append(
                self.add_variable_from_reference(
                    reference_variable=var, name="s"
                )
            )
            if self.momentum != 0:
                self._x0.append(tf.identity(var))
            self.step.append(0)

    def update_step(self, gradient, variable, learning_rate):
        lr = tf.cast(learning_rate, variable.dtype) + self.epsilon
        ck = 1 - self.momentum
        
        self.step[self._get_variable_index(variable)] += 1
        grad_sum_sq, s = self._grad_sum_sq[self._get_variable_index(variable)], self._s[self._get_variable_index(variable)]
        lamb = lr * math.sqrt(self.step[self._get_variable_index(variable)])

        if self.decoupled_decay:
            variable.assign(variable * 1.0 - lr)

        if isinstance(gradient, tf.SparseTensor):
            grad_val = gradient.values

            p_masked = tf.gather_nd(variable, gradient.indices)
            grad_sum_sq_masked = tf.gather_nd(grad_sum_sq, gradient.indices)
            s_masked = tf.gather_nd(s, gradient.indices)

            # Compute x_0 from other known quantities
            rms_masked_vals = tf.pow(grad_sum_sq_masked.values, 1 / 3) + self.epsilon
            x0_masked_vals = p_masked.values + 1 * (s_masked.values / rms_masked_vals)

            # Dense + sparse op
            grad_sq = gradient * gradient
            grad_sum_sq.assign_add(grad_sq * lamb)
            grad_sum_sq_masked.assign_add(grad_sq * lamb)

            rms_masked_vals = tf.pow(grad_sum_sq_masked.values, 1 / 3) + self.epsilon

            s.assign_add(gradient * lamb)
            s_masked.values.assign_add(grad_val * lamb)

            # update masked copy of p
            p_kp1_masked_vals = x0_masked_vals + -1 * (s_masked.values / rms_masked_vals)
            # Copy updated masked p to dense p using an add operation
            p_masked.values.assign_add(p_kp1_masked_vals * -1)
            variable.assign_add(p_masked * -1)
        else:
            if self.momentum == 0:
                # Compute x_0 from other known quantities
                rms = tf.pow(grad_sum_sq, 1 / 3) + self.epsilon
                x0 = variable + 1 * (s / rms)
            else:
                x0 = self.x0[self._get_variable_index(variable)]

            # Accumulate second moments
            grad_sum_sq.assign(grad_sum_sq + lamb * gradient * gradient)
            rms = tf.pow(grad_sum_sq, 1 / 3) + self.epsilon

            # Update s
            s.assign_add(gradient * lamb)

            # Step
            if self.momentum == 0:
                variable.assign(x0 + -1 * (s / rms))
            else:
                z = x0 + -1 * (s / rms)

                # p is a moving average of z
                variable.assign(variable * (1 - ck) + z * ck)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "momentum": self.momentum,
                "epsilon": self.epsilon,
                "decoupled_decay": self.decoupled_decay,
            }
        )
        return config