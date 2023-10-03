import tensorflow as tf


def normalize_tuple(value, n, allow_zero=False):
    error_msg = (
        f"integers. Received: {value}"
    )

    if isinstance(value, int):
        value_tuple = (value,) * n
    else:
        try:
            value_tuple = tuple(value)
        except TypeError:
            raise ValueError(error_msg)
        if len(value_tuple) != n:
            raise ValueError(error_msg)
        for single_value in value_tuple:
            try:
                int(single_value)
            except (ValueError, TypeError):
                error_msg += (
                    f"including element {single_value} of "
                    f"type {type(single_value)}"
                )
                raise ValueError(error_msg)

    if allow_zero:
        unqualified_values = {v for v in value_tuple if v < 0}
        req_msg = ">= 0"
    else:
        unqualified_values = {v for v in value_tuple if v <= 0}
        req_msg = "> 0"

    if unqualified_values:
        error_msg += (
            f" including {unqualified_values}"
            f" that does not satisfy the requirement `{req_msg}`."
        )
        raise ValueError(error_msg)

    return value_tuple


class zeropadding2d:
    def __init__(self,input_size=None):
        self.input_size=input_size
        if input_size!=None:
            self.output_size=input_size
            
            
    def build(self):
        self.output_size=self.input_size
        
        
    def output(self, data, padding=(1, 1)):
        if isinstance(padding, int):
            padding = ((padding, padding), (padding, padding))
        else:
            if len(padding) != 2:
                raise ValueError(
                    f"`padding` should have two elements. Received: {padding}."
                )
            height_padding = normalize_tuple(
                padding[0], 2, allow_zero=True
            )
            width_padding = normalize_tuple(
                padding[1], 2, allow_zero=True
            )
            padding = (height_padding, width_padding)
        pattern = [[0, 0], list(padding[0]), list(padding[1]), [0, 0]]
        return tf.pad(data, pattern)
