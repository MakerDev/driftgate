"""§6.3 offload client: Python standard library only (Jetson, Android Termux, laptop).

Sends one offloaded request (A = 32,768 bytes, one fp32 feature [128,8,8]) and waits for the logits,
`--n` times (default 500) after `--warmup` requests, and writes one CSV row per request:
  device, network, i, bytes_sent, bytes_recv, rtt_total_ms, server_compute_ms, network_ms, unix_time
network_ms = rtt_total_ms - server_compute_ms (send + transfer + reply, without the server's compute).
latency_calc.py --net-csv uses the network_ms column in place of 8A/B + RTT.

  python3 offload_client.py --host 192.168.0.10 --port 5555 --device-name "Galaxy S24" \
      --network-name "WiFi 5GHz" --out net_latency_s24_wifi.csv
"""
import argparse
import csv
import os
import random
import socket
import struct
import time

FEATURE_BYTES = 128 * 8 * 8 * 4


def recv_exact(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("server closed the connection")
        buf.extend(chunk)
    return bytes(buf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True)
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--batch", type=int, default=1, help="features per message (1 = one offloaded request)")
    ap.add_argument("--interval-ms", type=float, default=100.0, help="pause between requests")
    ap.add_argument("--device-name", required=True)
    ap.add_argument("--network-name", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rnd = random.Random(0)
    payload = bytes(rnd.getrandbits(8) for _ in range(FEATURE_BYTES * a.batch))
    msg = struct.pack(">I", len(payload)) + payload
    sock = socket.create_connection((a.host, a.port))
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    rows = []
    for i in range(a.warmup + a.n):
        t0 = time.perf_counter()
        sock.sendall(msg)
        (r,) = struct.unpack(">I", recv_exact(sock, 4))
        reply = recv_exact(sock, r)
        total = (time.perf_counter() - t0) * 1000.0
        (srv,) = struct.unpack(">d", reply[:8])
        if i >= a.warmup:
            rows.append([a.device_name, a.network_name, i - a.warmup, len(msg), 4 + r,
                         f"{total:.4f}", f"{srv:.4f}", f"{total - srv:.4f}", f"{time.time():.3f}"])
        if a.interval_ms > 0:
            time.sleep(a.interval_ms / 1000.0)
    sock.close()
    new = not os.path.exists(a.out)
    with open(a.out, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["device", "network", "i", "bytes_sent", "bytes_recv", "rtt_total_ms",
                        "server_compute_ms", "network_ms", "unix_time"])
        w.writerows(rows)
    net = sorted(float(x[7]) for x in rows)
    print(f"{len(rows)} requests: network median {net[len(net) // 2]:.2f} ms, "
          f"p95 {net[int(0.95 * len(net)) - 1]:.2f} ms -> {a.out}")


if __name__ == "__main__":
    main()
