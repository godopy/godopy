def _check_quaternion_data(data):
    if not issubscriptable(data) or (hasattr(data, 'shape') and data.shape != (4,)) or len(data) != 4:
        raise ValueError("Quaternion data must be a 1-dimensional container of 4 items")
    map(_check_numeric_scalar, data)


def as_quaternion(data, dtype=None):
    if dtype is None:
        dtype = np.float32

    copy = False
    if isinstance(data, numpy.ndarray):
        if data.dtype != dtype:
            copy = True
    else:
        copy = True

    return Quaternion(data, dtype=dtype, copy=copy, can_cast=True)


class Quaternion(numpy.ndarray):
    pass
