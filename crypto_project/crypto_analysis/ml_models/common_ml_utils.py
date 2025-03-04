import cupy as cp


def to_cupy_array(df):
    with cp.cuda.Device(0):
        return cp.array(df.values, dtype=cp.float32)
