import tensorflow as tf
from keras.src.optimizers import optimizer


class LaProp(optimizer.Optimizer):
    def __init__(
        self,
        learning_rate=4e-4,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-15,
        amsgrad=False,
        centered=False,
        weight_decay=0,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        gradient_accumulation_steps=None,
        name="laprop",
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
        self.epsilon = epsilon
        self.amsgrad = amsgrad
        self.centered = centered
        self.steps_before_using_centered = 10

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self._exp_avg = []
        self._exp_avg_lr_1 = []
        self._exp_avg_lr_2 = []
        self._exp_avg_sq = []
        if self.centered:
            self._exp_mean_avg_beta2 = []
        if self.amsgrad:
            self._max_exp_avg_sq = []
        self.step = []
        for var in var_list:
            self._exp_avg.append(
                self.add_variable_from_reference(
                    reference_variable=var, name="exp_avg"
                )
            )
            self._exp_avg_lr_1.append(0.)
            self._exp_avg_lr_2.append(0.)
            self._exp_avg_sq.append(
                self.add_variable_from_reference(
                    reference_variable=var, name="exp_avg_sq"
                )
            )
            if self.centered:
                self._exp_mean_avg_beta2.append(
                    self.add_variable_from_reference(
                        reference_variable=var, name="exp_mean_avg_beta2"
                    )
                )
            if self.amsgrad:
                self._max_exp_avg_sq.append(
                    self.add_variable_from_reference(
                        reference_variable=var, name="max_exp_avg_sq"
                    )
                )
            self.step.append(0)

    def update_step(self, gradient, variable, learning_rate):
        lr = tf.cast(learning_rate, variable.dtype)
        exp_avg, exp_avg_sq = self._exp_avg[self._get_variable_index(variable)], self.exp_avg_sq[self._get_variable_index(variable)]
        if self.centered:
            exp_mean_avg_beta2 = self._exp_mean_avg_beta2[self._get_variable_index(variable)]
        if self.amsgrad:
            max_exp_avg_sq = self._max_exp_avg_sq[self._get_variable_index(variable)]
        beta1, beta2 = self.beta1, self.beta2

        self.step[self._get_variable_index(variable)] += 1

        # Decay the first and second moment running average coefficient
        exp_avg_sq.assign(exp_avg_sq * beta2 + (1 - beta2) * gradient * gradient)

        self._exp_avg_lr_1[self._get_variable_index(variable)] = self._exp_avg_lr_1[self._get_variable_index(variable)] * beta1 + (1 - beta1) * lr
        self._exp_avg_lr_2[self._get_variable_index(variable)] = self._exp_avg_lr_2[self._get_variable_index(variable)] * beta2 + (1 - beta2)

        bias_correction1 = self._exp_avg_lr_1[self._get_variable_index(variable)] / lr if lr.numpy()!=0. else tf.cast(1., variable.dtype) #1 - beta1 ** step
        step_size = 1 / bias_correction1

        bias_correction2 = self._exp_avg_lr_2[self._get_variable_index(variable)]
        
        denom = exp_avg_sq
        if self.centered:
            exp_mean_avg_beta2.assign(exp_mean_avg_beta2 * beta2 + (1 - beta2) * gradient)
            if self.step[self._get_variable_index(variable)] > self.steps_before_using_centered:
                mean = exp_mean_avg_beta2 ** 2
                denom = denom - mean

        if self.amsgrad:
            if not (self.centered and self.step[self._get_variable_index(variable)] <= self.steps_before_using_centered): 
                # Maintains the maximum of all (centered) 2nd moment running avg. till now
                max_exp_avg_sq.assign(tf.maximum(max_exp_avg_sq, denom))
                # Use the max. for normalizing running avg. of gradient
                denom = max_exp_avg_sq

        denom = tf.sqrt(denom / bias_correction2) + self.epsilon
        step_of_this_grad = gradient / denom
        exp_avg.assign(exp_avg * beta1 + (1 - beta1) * lr * step_of_this_grad)
        
        variable.assign_add(-step_size * exp_avg)
        
        if self.weight_decay != 0:
            variable.assign_add(variable * -lr * self.weight_decay)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "weight_decay": self.weight_decay,
                "beta_1": self.beta_1,
                "beta_2": self.beta_2,
                "epsilon": self.epsilon,
                "amsgrad": self.amsgrad,
                "centered": self.centered,
                "steps_before_using_centered": self.steps_before_using_centered,
            }
        )
        return config