"""Phase 6 — communication impairments for signals and model updates.

SignalChannel: per-ES scalar signal delivery with round-delay and packet loss.
On loss the receiver HOLDS the last delivered value (stale signal), which is
what a real controller would do; the staleness age is recorded so its effect
is measurable.

ParticipationSampler: per-round partial client participation.
"""
import numpy as np
from collections import deque


class SignalChannel:
    def __init__(self, delay=0, loss=0.0, seed=0):
        self.delay = int(delay)
        self.loss = float(loss)
        self.rng = np.random.default_rng(seed * 131 + 7)
        self.queues = {}      # es -> deque of values in flight
        self.last = {}        # es -> last delivered value
        self.age = {}         # es -> rounds since a fresh delivery

    def send_and_receive(self, signal_per_es):
        """Push this round's measurements; return what each ES's controller
        actually receives (delayed/stale). Strictly causal."""
        out, ages = {}, {}
        for es, v in signal_per_es.items():
            q = self.queues.setdefault(es, deque())
            q.append(v)
            delivered = None
            if len(q) > self.delay:
                candidate = q.popleft()
                if self.rng.random() >= self.loss:
                    delivered = candidate
            if delivered is not None:
                self.last[es] = delivered
                self.age[es] = 0
            else:
                self.age[es] = self.age.get(es, 0) + 1
            if es in self.last:
                out[es] = self.last[es]
                ages[es] = self.age[es]
        return out, ages


class ParticipationSampler:
    def __init__(self, fraction=1.0, seed=0):
        self.fraction = float(fraction)
        self.rng = np.random.default_rng(seed * 977 + 3)

    def sample(self, clients):
        if self.fraction >= 1.0:
            return list(clients)
        k = max(1, int(round(self.fraction * len(clients))))
        idx = self.rng.choice(len(clients), size=k, replace=False)
        return [clients[i] for i in sorted(idx)]
