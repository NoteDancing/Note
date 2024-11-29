import tensorflow as tf
from Note import nn


class avg_pool1d:
    def __init__(self, kernel_size, strides=None, padding=0, count_include_pad=True):
        """
        Args:
            kernel_size: int, the size of the pooling window.
            strides: int, stride of the pooling operation.
            padding: int, str, or tuple, the padding applied to the input.
            count_include_pad: bool, whether to include zero padding in the average calculation.
        """
        self.kernel_size = kernel_size
        self.strides = strides if strides is not None else kernel_size
        self.padding = padding
        self.count_include_pad = count_include_pad

        if not isinstance(padding, str):
            self.zeropadding1d = nn.zeropadding1d(padding=padding)

    def __call__(self, data):
        if not isinstance(self.padding, str):
            padded_data = self.zeropadding1d(data)
            padding = 'VALID'
        else:
            padded_data = data
            padding = self.padding

        # Apply avg_pool1d
        pooled = tf.nn.avg_pool1d(
            padded_data, ksize=self.kernel_size, strides=self.strides, padding=padding
        )

        if not self.count_include_pad and not isinstance(self.padding, str):
            # Calculate the effective kernel size for each window
            k = self.kernel_size if isinstance(self.kernel_size, int) else self.kernel_size

            # Compute the mask of valid elements (non-zero-padded)
            valid_mask = tf.ones_like(data, dtype=data.dtype)
            valid_mask = self.zeropadding1d(valid_mask)

            # Apply the same pooling operation to the mask
            valid_counts = tf.nn.avg_pool1d(
                valid_mask, ksize=self.kernel_size, strides=self.strides, padding='VALID'
            ) * k

            # Avoid division by zero
            valid_counts = tf.maximum(valid_counts, 1.0)

            # Adjust the pooled output to exclude zero-padded elements
            pooled = pooled * k / valid_counts

        return pooled
