""" Adan Optimizer

Adan: Adaptive Nesterov Momentum Algorithm for Faster Optimizing Deep Models[J]. arXiv preprint arXiv:2208.06677, 2022.
    https://arxiv.org/abs/2208.06677

Implementation adapted from https://github.com/sail-sg/Adan
"""
# Copyright 2024 NoteDance
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf
from keras.src.optimizers import optimizer
import math


class MultiTensorApply(object):
    available = False
    warned = False

    def __init__(self, chunk_size):
        try:
            MultiTensorApply.available = True
            self.chunk_size = chunk_size
        except ImportError as err:
            MultiTensorApply.available = False
            MultiTensorApply.import_err = err

    def __call__(self, op, noop_flag_buffer, tensor_lists, *args):
        return op(self.chunk_size, noop_flag_buffer, tensor_lists, *args)
    

class Adan(optimizer.Optimizer):
    def __init__(
        self,
        learning_rate=1e-3,
        beta_1=0.98,
        beta_2=0.92,
        beta_3=0.99,
        epsilon=1e-8,
        weight_decay=0.0,
        no_prox=False,
        foreach=True,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        gradient_accumulation_steps=None,
        name="adan",
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
        self.weight_decay = weight_decay
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.beta_3 = beta_2
        self.epsilon = epsilon
        self.no_prox = no_prox
        self.foreach = foreach
    
    def __setstate__(self, state):
        self.__dict__.update(state)
        self.no_prox = False
    
    def restart_opt(self):
        for i,v in enumerate(self._trainable_variables):
            # State initialization
            self.step[i] = 0
            # Exponential moving average of gradient values
            self._exp_avg[i] = self.add_variable_from_reference(
                reference_variable=v, name="exp_avg"
            )

            # Exponential moving average of squared gradient values
            self._exp_avg_sq[i] = self.add_variable_from_reference(
                reference_variable=v, name="exp_avg_sq"
            )
            
            # Exponential moving average of gradient difference
            self._exp_avg_diff[i] = self.add_variable_from_reference(
                reference_variable=v, name="exp_avg_diff"
            )

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self._exp_avg = []
        self._exp_avg_sq = []
        self._exp_avg_diff = []
        self.params_with_grad = []
        self.grads = []
        self.exp_avgs = []
        self.exp_avg_sqs = []
        self.exp_avg_diffs = []
        self.neg_pre_grads = []
        self.neg_pre_grad = []
        self.step = []
        for var in var_list:
            self._exp_avg.append(
                self.add_variable_from_reference(
                    reference_variable=var, name="exp_avg"
                )
            )
            self._exp_avg_sq.append(
                self.add_variable_from_reference(
                    reference_variable=var, name="exp_avg_sq"
                )
            )
            self._exp_avg_diff.append(
                self.add_variable_from_reference(
                    reference_variable=var, name="exp_avg_diff"
                )
            )
            self.neg_pre_grad.append(0)
            self.step.append(0)

    def update_step(self, gradient, variable, learning_rate):
        lr = tf.cast(learning_rate, variable.dtype)

        self.step[self._get_variable_index(variable)] += 1
        
        bias_correction1 = 1 - self.beta1 ** self.step[self._get_variable_index(variable)]
        bias_correction2 = 1 - self.beta2 ** self.step[self._get_variable_index(variable)]
        bias_correction3 = 1 - self.beta3 ** self.step[self._get_variable_index(variable)]

        self.params_with_grad.append(variable)
        self.grads.append(gradient)
        
        if self.step[self._get_variable_index(variable)] == 1:
            self.neg_pre_grad[self._get_variable_index(variable)] = -tf.identity(gradient)
        
        self.exp_avgs.append(self._exp_avg[self._get_variable_index(variable)])
        self.exp_avg_sqs.append(self._exp_avg_sq[self._get_variable_index(variable)])
        self.exp_avg_diffs.append(self._exp_avg_diff[self._get_variable_index(variable)])
        self.neg_pre_grads.append(self._neg_pre_grad[self._get_variable_index(variable)])
        
        kwargs = dict(
            params=self.params_with_grad,
            grads=self.grads,
            exp_avgs=self.exp_avgs,
            exp_avg_sqs=self.exp_avg_sqs,
            exp_avg_diffs=self.exp_avg_diffs,
            neg_pre_grads=self.neg_pre_grads,
            beta1=self.beta1,
            beta2=self.beta2,
            beta3=self.beta3,
            bias_correction1=bias_correction1,
            bias_correction2=bias_correction2,
            bias_correction3_sqrt=math.sqrt(bias_correction3),
            lr=lr,
            weight_decay=self.weight_decay,
            eps=self.epsilon,
            no_prox=self.no_prox,
        )

        if self.foreach:
            _multi_tensor_adan(**kwargs)
        else:
            _single_tensor_adan(**kwargs)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "weight_decay": self.weight_decay,
                "beta_1": self.beta_1,
                "beta_2": self.beta_2,
                "beta_3": self.beta_3,
                "epsilon": self.epsilon,
                "no_prox": self.no_prox,
                "foreach": self.foreach,
            }
        )
        return config


def _single_tensor_adan(
    params, grads, exp_avgs, exp_avg_sqs, exp_avg_diffs, neg_pre_grads,
    beta1, beta2, beta3, bias_correction1, bias_correction2,
    bias_correction3_sqrt, lr, weight_decay, eps, no_prox
):
    for i, param in enumerate(params):
        grad = grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        exp_avg_diff = exp_avg_diffs[i]
        neg_grad_or_diff = neg_pre_grads[i]

        # for memory saving, we use `neg_grad_or_diff` to get some temp variable in an inplace way
        neg_grad_or_diff.assign_add(grad)

        exp_avg.assign(beta1 * exp_avg + (1 - beta1) * grad)  # m_t
        exp_avg_diff.assign(beta2 * exp_avg_diff + (1 - beta2) * neg_grad_or_diff)  # diff_t

        neg_grad_or_diff.assign(beta2 * neg_grad_or_diff + grad)
        exp_avg_sq.assign(beta3 * exp_avg_sq + (1 - beta3) * tf.square(neg_grad_or_diff))  # n_t

        denom = tf.sqrt(exp_avg_sq / bias_correction3_sqrt) + eps
        step_size_diff = lr * beta2 / bias_correction2
        step_size = lr / bias_correction1

        if no_prox:
            param.assign(param * (1 - lr * weight_decay))
            param.assign_add(-step_size * exp_avg / denom)
            param.assign_add(-step_size_diff * exp_avg_diff / denom)
        else:
            param.assign_add(-step_size * exp_avg / denom)
            param.assign_add(-step_size_diff * exp_avg_diff / denom)
            param.assign(param / (1 + lr * weight_decay))

        neg_grad_or_diff.assign(-1.0 * grad)


def _multi_tensor_adan(
    params, grads, exp_avgs, exp_avg_sqs, exp_avg_diffs, neg_pre_grads,
    beta1, beta2, beta3, bias_correction1, bias_correction2,
    bias_correction3_sqrt, lr, weight_decay, eps, no_prox
):
    if len(params) == 0:
        return

    neg_pre_grads = [neg_pre_grad.assign(neg_pre_grad + grad) for neg_pre_grad, grad in zip(neg_pre_grads, grads)]

    exp_avgs = [exp_avg.assign(beta1 * exp_avg + (1 - beta1) * grad) for exp_avg, grad in zip(exp_avgs, grads)]

    exp_avg_diffs = [
        exp_avg_diff.assign(beta2 * exp_avg_diff + (1 - beta2) * neg_pre_grad)
        for exp_avg_diff, neg_pre_grad in zip(exp_avg_diffs, neg_pre_grads)
    ]

    neg_pre_grads = [neg_pre_grad.assign(beta2 * neg_pre_grad + grad) for neg_pre_grad, grad in zip(neg_pre_grads, grads)]
    exp_avg_sqs = [
        exp_avg_sq.assign(beta3 * exp_avg_sq + (1 - beta3) * tf.square(neg_pre_grad))
        for exp_avg_sq, neg_pre_grad in zip(exp_avg_sqs, neg_pre_grads)
    ]

    denom = [tf.sqrt(exp_avg_sq / bias_correction3_sqrt) + eps for exp_avg_sq in exp_avg_sqs]
    step_size_diff = lr * beta2 / bias_correction2
    step_size = lr / bias_correction1

    if no_prox:
        params = [
            param.assign(param * (1 - lr * weight_decay) - step_size * exp_avg / d - step_size_diff * exp_avg_diff / d)
            for param, exp_avg, exp_avg_diff, d in zip(params, exp_avgs, exp_avg_diffs, denom)
        ]
    else:
        params = [
            param.assign(param - step_size * exp_avg / d - step_size_diff * exp_avg_diff / d)
            for param, exp_avg, exp_avg_diff, d in zip(params, exp_avgs, exp_avg_diffs, denom)
        ]
        params = [param.assign(param / (1 + lr * weight_decay)) for param in params]

    neg_pre_grads = [neg_pre_grad.assign(-1.0 * grad) for neg_pre_grad, grad in zip(neg_pre_grads, grads)]

    return params, neg_pre_grads