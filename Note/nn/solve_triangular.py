import tensorflow as tf


def solve_triangular(A, B, *, upper, left=True, unitriangular=False):
    if unitriangular:
        diag_shape = tf.shape(tf.linalg.diag_part(A))
        ones = tf.ones(diag_shape, dtype=A.dtype)
        A = tf.linalg.set_diag(A, ones)
    
    if left:
        X = tf.linalg.triangular_solve(A, B, lower=not upper)
    else:
        X_T = tf.linalg.triangular_solve(tf.transpose(A), tf.transpose(B), lower=upper)
        X = tf.transpose(X_T)
    return X