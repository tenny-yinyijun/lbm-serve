"""NumPy array support for msgpack, wire-compatible with openpi's `openpi_client.msgpack_numpy`.

This is a deliberate reimplementation rather than a dependency. `openpi-client` pins
`numpy>=1.22.4,<2.0.0`, and this environment runs numpy 2.x, so installing it would either fail
to resolve or silently downgrade numpy under torch. What actually has to match between the two
repos is the *byte format* on the wire, which is small and stable:

    ndarray  -> {b"__ndarray__": True, b"data": <raw bytes>, b"dtype": <dtype.str>, b"shape": (...)}
    np scalar-> {b"__npgeneric__": True, b"data": <python scalar>, b"dtype": <dtype.str>}

Keep this file byte-identical in behaviour to
`external/openpi/packages/openpi-client/src/openpi_client/msgpack_numpy.py` in the open-world
checkout; a divergence here shows up as a decode error or, worse, a silently misinterpreted
observation. Both sides refuse void/object/complex dtypes rather than falling back to pickle.

Verified, not assumed: packing a representative observation and a `{"actions": [16,20]}`
response with openpi's codec in the open-world venv and with this one here produces identical
bytes in both directions. Re-run that check (see the recipe in
`vla_foundry/serving/smoke_test_lbm_policy_server.py`) if either side's msgpack major version
moves.
"""

import functools

import msgpack
import numpy as np


def pack_array(obj):
    if (isinstance(obj, (np.ndarray, np.generic))) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")

    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }

    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }

    return obj


def unpack_array(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])

    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])

    return obj


Packer = functools.partial(msgpack.Packer, default=pack_array)
packb = functools.partial(msgpack.packb, default=pack_array)

Unpacker = functools.partial(msgpack.Unpacker, object_hook=unpack_array)
unpackb = functools.partial(msgpack.unpackb, object_hook=unpack_array)
