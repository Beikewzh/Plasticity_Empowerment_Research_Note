"""Does the agent's plasticity tighten the certificate?  W1, memory = last reward, policy plays action 1 with prob p1 if the
last reward was 1, p0 otherwise.  Vary the responsiveness |p1 - p0|; report the policy's plasticity through the memory P(M),
the response gap sigma, the constant C = 2/(rho sigma^2), and D, dP at t = 2 (nats)."""
import sys, os, math
DIR = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(DIR, "..", "perfect_state"))
from plib import *
import numpy as np
from collections import defaultdict
T = 4; t = 2
def kl(p, q): return float(sum(pi * math.log(pi / qi) for pi, qi in zip(p, q) if pi > 0))
def run(p0, p1):
    mem = lambda o, a: o[-1][2]; pol = lambda t_, m: bern(p1 if m else p0)
    J = joint(W1("I"), agent_rule(mem, pol), T); obs_vals = sorted({O(t + 1)(k) for k in J})
    w_ha = defaultdict(float); n_hao = defaultdict(float)
    for k, p in J.items():
        h, a, o = hist(t)(k), A(t)(k), O(t + 1)(k); w_ha[(h, a)] += p; n_hao[(h, a, o)] += p
    classes = defaultdict(list)
    for (h, a) in w_ha: classes[(mem(*h), a)].append(h)
    D = dP = 0.0; Cmax = 0.0; sig_min = 9
    for (m, a), hs in classes.items():
        wm = sum(w_ha[(h, a)] for h in hs); P = np.array([[n_hao[(h, a, o)] / w_ha[(h, a)] for o in obs_vals] for h in hs])
        w = np.array([w_ha[(h, a)] / wm for h in hs]); pbar = w @ P; h0 = hs[0]
        K = np.array([[pol(t + 1, mem(h0[0] + (o,), h0[1] + (a,))).get(ap, 0.0) for o in obs_vals] for ap in (0, 1)])
        Dma = sum(wi * kl(pi, pbar) for wi, pi in zip(w, P)); dPma = sum(wi * kl(K @ pi, K @ pbar) for wi, pi in zip(w, P))
        V = P - pbar; U, S, Vt = np.linalg.svd(V, full_matrices=False); basis = Vt[S > 1e-10]
        sigma = float(np.linalg.svd(K @ basis.T, compute_uv=False).min()) if len(basis) else float("inf")
        rho = float(pbar[pbar > 1e-12].min()); C = 2 / (rho * sigma ** 2) if sigma > 1e-12 else float("inf")
        D += wm * Dma; dP += wm * dPma; Cmax = max(Cmax, C) if len(basis) else Cmax; sig_min = min(sig_min, sigma)
    PM = cmi(J, O(t + 1), A(t + 1), tup(Zt(mem, t), A(t)), nats=True)        # plasticity through the memory at t+1
    print(f"  p0={p0:.2f} p1={p1:.2f} |p1-p0|={abs(p1-p0):.2f}: P(M) {PM:.4f} | sigma {sig_min:.3f} | C {Cmax:7.1f} | D {D:.4f} dP {dP:.4f} | C*dP {Cmax*dP:.3f} >= D: {Cmax*dP >= D - 1e-9}")
print("W1, memory = last reward (insufficient): the agent's responsiveness vs the plasticity certificate (nats, t = 2)")
for p0, p1 in ((0.50, 0.50), (0.45, 0.55), (0.40, 0.60), (0.30, 0.70), (0.20, 0.85), (0.10, 0.90), (0.02, 0.98)): run(p0, p1)
