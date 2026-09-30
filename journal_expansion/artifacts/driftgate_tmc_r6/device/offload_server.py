"""§6.3 offload server (edge server). Receives features, runs server block + server exit, returns logits.

Protocol (TCP, big-endian), one request per message:
  client -> server : uint32 payload length L, then L bytes = B fp32 features of shape [B,128,8,8]
                     (L = B * 32768; B = 1 for one offloaded request)
  server -> client : uint32 reply length R, then float64 server_compute_ms, then B*10 fp32 logits
server_compute_ms is measured on the server (tensor creation -> logits copied back), so the client
can subtract it and keep the network-only time.

  python3 offload_server.py --weights models/driftgate_s1_s0_weights.pth --port 5555 --device cuda
"""
import argparse
import socket
import struct
import threading
import time

import numpy as np
import torch

from r6_device_models import FEATURE_BYTES, FEATURE_SHAPE, load


def recv_exact(conn, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("peer closed")
        buf.extend(chunk)
    return bytes(buf)


def serve(conn, addr, server, dev, lock):
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    try:
        while True:
            (n,) = struct.unpack(">I", recv_exact(conn, 4))
            payload = recv_exact(conn, n)
            b = n // FEATURE_BYTES
            with lock:
                t0 = time.perf_counter()
                f = torch.from_numpy(np.frombuffer(payload, dtype=np.float32).copy()).view(b, *FEATURE_SHAPE).to(dev)
                with torch.no_grad():
                    out = server(f).float().cpu().numpy().tobytes()
                ms = (time.perf_counter() - t0) * 1000.0
            reply = struct.pack(">d", ms) + out
            conn.sendall(struct.pack(">I", len(reply)) + reply)
    except ConnectionError:
        pass
    finally:
        conn.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    dev = torch.device(a.device)
    _, server = load(a.weights, dev)
    with torch.no_grad():                               # warm-up
        for _ in range(20):
            server(torch.zeros(1, *FEATURE_SHAPE, device=dev))
    lock = threading.Lock()
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((a.host, a.port))
    s.listen(16)
    print(f"offload server on {a.host}:{a.port} ({dev}{', ' + torch.cuda.get_device_name(0) if dev.type == 'cuda' else ''})", flush=True)
    while True:
        conn, addr = s.accept()
        threading.Thread(target=serve, args=(conn, addr, server, dev, lock), daemon=True).start()


if __name__ == "__main__":
    main()
