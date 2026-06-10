import os
import ctypes
import numpy as np

def cartesian_product(arrays):
    la = len(arrays)
    dtype = arrays[0].dtype
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[..., i] = a
    return arr.reshape(-1, la)

# ---------------------------------------------------------------------------
# Shared library loading
# ---------------------------------------------------------------------------
# Resolve libcompress.so once and configure the ctypes signatures a single
# time, instead of reloading the library and re-declaring argtypes on every
# call. Resolution order: $ENCODE2_LIB -> next to this file under encode2/ ->
# the legacy absolute path.

_HERE = os.path.dirname(os.path.abspath(__file__))
_LIB_CANDIDATES = [
    os.environ.get('ENCODE2_LIB'),
    os.path.join(_HERE, 'encode2', 'libcompress.so'),
    '/encode2/libcompress.so',
]

c_float_p = ctypes.POINTER(ctypes.c_float)
c_int_p = ctypes.POINTER(ctypes.c_int)

_clib = None
_radius = None


def _load_library():
    global _clib
    if _clib is not None:
        return _clib

    lib_path = next((p for p in _LIB_CANDIDATES if p and os.path.exists(p)), None)
    if lib_path is None:
        raise FileNotFoundError(
            'Could not locate libcompress.so. Set the ENCODE2_LIB environment '
            f'variable or build it under {os.path.join(_HERE, "encode2")}.'
        )

    lib = ctypes.CDLL(lib_path, use_last_error=True)

    # void compress(const float* data, const float* preds, double eb,
    #               float data_range, size_t num_elements, float mean,
    #               float std, const char* filename, int* quant_inds)
    # quant_inds is an optional output buffer (NULL to skip) that receives the
    # raw quantization indices, before Huffman encoding and zstd compression.
    lib.compress.argtypes = [
        c_float_p,        # data
        c_float_p,        # preds
        ctypes.c_double,  # eb
        ctypes.c_float,   # data_range
        ctypes.c_size_t,  # num_elements
        ctypes.c_float,   # mean
        ctypes.c_float,   # std
        ctypes.c_char_p,  # filename
        c_int_p,          # quant_inds (out, may be NULL)
    ]
    lib.compress.restype = None

    # void quantize(const float* data, const float* preds, double eb,
    #               float data_range, size_t num_elements, int* quant_inds)
    lib.quantize.argtypes = [
        c_float_p,        # data
        c_float_p,        # preds
        ctypes.c_double,  # eb
        ctypes.c_float,   # data_range
        ctypes.c_size_t,  # num_elements
        c_int_p,          # quant_inds (out)
    ]
    lib.quantize.restype = None

    # void decompress(const char* cmp_path, const float* preds,
    #                 size_t num_elements, float* res)
    lib.decompress.argtypes = [
        ctypes.c_char_p,  # cmp_path
        c_float_p,        # preds
        ctypes.c_size_t,  # num_elements
        c_float_p,        # res (out)
    ]
    lib.decompress.restype = None

    # int get_radius()  -- center of the quant index range, used to convert raw
    # indices into signed residual codes. Optional: older .so builds may lack it.
    try:
        lib.get_radius.argtypes = []
        lib.get_radius.restype = ctypes.c_int
    except AttributeError:
        pass

    _clib = lib
    return _clib


def quant_radius():
    global _radius
    if _radius is None:
        lib = _load_library()
        try:
            _radius = int(lib.get_radius())
        except AttributeError:
            _radius = 32768  # SZ3 default
    return _radius


def my_compress(data_cpy, preds_cpy, eb, filename, data_range=None,
                verbose=False, return_quant=False, signed=True):
    assert data_range is not None
    assert len(data_cpy.shape) == 1, 'data should be flattened'
    assert len(preds_cpy.shape) == 1, 'preds should be flattened'
    if verbose:
        print(f'Provided datarange is {data_range}')
        print(f'ABS Error bound is {eb*data_range}')

    lib = _load_library()

    # ctypes [C-contiguous]
    # float32.
    data_cpy = np.ascontiguousarray(data_cpy, dtype=np.float32)
    preds_cpy = np.ascontiguousarray(preds_cpy, dtype=np.float32)
    data_size = data_cpy.size

    quant_hold = None
    quant_ptr = None  #skipped copy
    if return_quant:
        quant_hold = np.empty(data_size, dtype=np.int32)
        quant_ptr = quant_hold.ctypes.data_as(c_int_p)

    lib.compress(
        data_cpy.ctypes.data_as(c_float_p),   # data
        preds_cpy.ctypes.data_as(c_float_p),  # preds
        eb,                                    # error bound (relative)
        data_range,                            # value range
        data_size,                             # num elements
        0,                                     # data mean (unused by C)
        0,                                     # data std (unused by C)
        bytes(filename, encoding='utf8'),
        quant_ptr,                             # quant indices out
    )

    cmp_size = os.path.getsize(filename)
    if return_quant:
        if signed:
            # raw index
            quant_hold -= quant_radius()
        return cmp_size, quant_hold
    return cmp_size


def quantize(data_cpy, preds_cpy, eb, data_range=None, signed=True):
    assert data_range is not None
    assert len(data_cpy.shape) == 1, 'data should be flattened'
    assert len(preds_cpy.shape) == 1, 'preds should be flattened'

    lib = _load_library()

    data_cpy = np.ascontiguousarray(data_cpy, dtype=np.float32)
    preds_cpy = np.ascontiguousarray(preds_cpy, dtype=np.float32)
    data_size = data_cpy.size

    quant_hold = np.empty(data_size, dtype=np.int32)
    lib.quantize(
        data_cpy.ctypes.data_as(c_float_p),
        preds_cpy.ctypes.data_as(c_float_p),
        eb,
        data_range,
        data_size,
        quant_hold.ctypes.data_as(c_int_p),
    )
    if signed:
        # raw index
        quant_hold -= quant_radius()   
    return quant_hold


def decompress(filename, data, preds, eb):
    assert len(data.shape) == 1, 'data should be flattened'
    assert len(preds.shape) == 1, 'preds should be flattened'

    lib = _load_library()

    preds = np.ascontiguousarray(preds, dtype=np.float32)
    res_hold = np.zeros(data.size, dtype=np.float32)

    lib.decompress(
        bytes(filename, encoding='utf8'),
        preds.ctypes.data_as(c_float_p),
        data.size,
        res_hold.ctypes.data_as(c_float_p),
    )
    return res_hold
