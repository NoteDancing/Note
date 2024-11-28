import tensorflow as tf
from Note import nn


class avg_pool2d:
    def __init__(self, kernel_size, strides=None, padding=0, count_include_pad=True):
        """
        Args:
            kernel_size: int or tuple, the size of the pooling window.
            strides: int or tuple, stride of the pooling operation.
            padding: int, str, or tuple, the padding applied to the input.
            count_include_pad: bool, whether to include zero padding in the average calculation.
        """
        self.kernel_size = kernel_size
        self.strides = strides if strides is not None else kernel_size
        self.padding = padding
        self.count_include_pad = count_include_pad

        if not isinstance(padding, str):
            self.zeropadding2d = nn.zeropadding2d(padding=padding)

    def __call__(self, data):
        if not isinstance(self.padding, str):
            padded_data = self.zeropadding2d(data)
            padding = 'VALID'
        else:
            padded_data = data
            padding = self.padding

        # Apply avg_pool2d
        pooled = tf.nn.avg_pool2d(
            padded_data, ksize=self.kernel_size, strides=self.strides, padding=padding
        )

        if not self.count_include_pad and not isinstance(self.padding, str):
            # Calculate the effective kernel sizes for each window
            k_h, k_w = self.kernel_size if isinstance(self.kernel_size, (tuple, list)) else (self.kernel_size, self.kernel_size)

            # Compute the mask of valid elements (non-zero-padded)
            valid_mask = tf.ones_like(data, dtype=data.dtype)
            valid_mask = self.zeropadding2d(valid_mask)

            # Apply the same pooling operation to the mask
            valid_counts = tf.nn.avg_pool2d(
                valid_mask, ksize=self.kernel_size, strides=self.strides, padding='VALID'
            ) * (k_h * k_w)

            # Avoid division by zero
            valid_counts = tf.maximum(valid_counts, 1.0)

            # Adjust the pooled output to exclude zero-padded elements
            pooled = pooled * (k_h * k_w) / valid_counts

        return pooled
