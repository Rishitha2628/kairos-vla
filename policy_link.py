"""Tiny request/response protocol between the simulator (Python 3.10, NumPy 1) and the policy
server (Python 3.12, NumPy 2). Pickles are not portable across those NumPy versions, so messages are
a JSON header plus raw array bytes:

    [4-byte header length][JSON header][array 0 bytes][array 1 bytes]...

Arrays anywhere in the message (dict values, nested) are replaced in the header by
{"__nd__": i, "dtype": ..., "shape": ...}.
"""
import json
import socket
import struct

import numpy as np


def _pack(obj, blobs):
    if isinstance(obj, np.ndarray):
        arr = np.ascontiguousarray(obj)
        blobs.append(arr.tobytes())
        return {"__nd__": len(blobs) - 1, "dtype": arr.dtype.str, "shape": list(arr.shape)}
    if isinstance(obj, dict):
        return {k: _pack(v, blobs) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_pack(v, blobs) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def _unpack(obj, blobs):
    if isinstance(obj, dict):
        if "__nd__" in obj:
            return np.frombuffer(blobs[obj["__nd__"]], dtype=np.dtype(obj["dtype"])).reshape(obj["shape"]).copy()
        return {k: _unpack(v, blobs) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_unpack(v, blobs) for v in obj]
    return obj


def _recv_exact(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("peer closed the connection")
        buf.extend(chunk)
    return bytes(buf)


def send(sock, msg):
    blobs = []
    header = _pack(msg, blobs)
    header["__sizes__"] = [len(b) for b in blobs]
    raw = json.dumps(header).encode()
    sock.sendall(struct.pack("!I", len(raw)) + raw + b"".join(blobs))


def recv(sock):
    (n,) = struct.unpack("!I", _recv_exact(sock, 4))
    header = json.loads(_recv_exact(sock, n))
    blobs = [_recv_exact(sock, size) for size in header.pop("__sizes__")]
    return _unpack(header, blobs)


def connect(port, host="127.0.0.1"):
    return socket.create_connection((host, port))
