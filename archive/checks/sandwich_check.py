"""Plasticity sandwich under regime identifiability, exact at t = 2 (next observation O_3, next action A_3), nats.
Per reachable (m, a): p_h = law of O_3 given (h_2 = h, A_2 = a); pbar = law given (M_2 = m, A_2 = a); K(a' | o') = pi_3(a' | f(m, a, o')).
D_{m,a} = sum_h w_h KL(p_h || pbar);  dP_{m,a} = sum_h w_h KL(K p_h || K pbar);  sigma = smallest singular value of K on span{p_h - pbar};
rho = min pbar(o') on the support;  bound: D_{m,a} <= 2/(rho sigma^2) dP_{m,a}."""
import sys, os, math
DIR = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(DIR, "..", "perfect_state"))
from plib import *
import numpy as np
from collections import defaultdict
T = 4; t = 2
P0 = np.array([[0.35, 0.35, 0.30], [0.50, 0.20, 0.30], [0.20, 0.20, 0.60]]); P1 = np.array([[0.10, 0.10, 0.80], [0.10, 0.10, 0.80], [0.45, 0.45, 0.10]]); RW = np.array([[0.5, 0.9], [0.5, 0.1], [0.9, 0.5]])
def W3s():
    def init(): return [((s, 0), 1 / 3) for s in range(3)]
    def step(obs, acts, a):
        s = obs[-1][0]; return [((s2, r), float((P1 if a else P0)[s, s2] * (RW[s, a] if r else 1 - RW[s, a]))) for s2 in range(3) for r in (0, 1)]
    return init, step
def kl(p, q): return float(sum(pi * math.log(pi / qi) for pi, qi in zip(p, q) if pi > 0))
def sandwich(label, world, mem, pol, actions=(0, 1)):
    J = joint(world, agent_rule(mem, pol), T, actions)
    obs_vals = sorted({O(t + 1)(k) for k in J}); oi = {o: i for i, o in enumerate(obs_vals)}
    acts_vals = list(actions)
    w_ha = defaultdict(float); n_hao = defaultdict(float)
    for k, p in J.items():
        h, a, o = hist(t)(k), A(t)(k), O(t + 1)(k); w_ha[(h, a)] += p; n_hao[(h, a, o)] += p
    classes = defaultdict(list)
    for (h, a) in w_ha: classes[(mem(*h), a)].append(h)
    Dtot = dPtot = 0.0; worstC = 0.0; rows = []
    for (m, a), hs in classes.items():
        wm = sum(w_ha[(h, a)] for h in hs)
        P = np.array([[n_hao[(h, a, o)] / w_ha[(h, a)] for o in obs_vals] for h in hs])          # rows: p_h
        w = np.array([w_ha[(h, a)] / wm for h in hs]); pbar = w @ P
        # K: next action law for each next observation (memory update is a function of the current observation here)
        h0 = hs[0]
        K = np.array([[pol(t + 1, mem(h0[0] + (o,), h0[1] + (a,))).get(ap, 0.0) for o in obs_vals] for ap in acts_vals])   # |A| x |O|
        Dma = float(sum(wi * kl(pi, pbar) for wi, pi in zip(w, P)))
        dPma = float(sum(wi * kl(K @ pi, K @ pbar) for wi, pi in zip(w, P)))
        V = P - pbar; U, S, Vt = np.linalg.svd(V, full_matrices=False); basis = Vt[S > 1e-10]         # orthonormal basis of span{p_h - pbar}
        if len(basis) == 0: sigma = float("inf"); dim = 0
        else:
            KQ = K @ basis.T; sigma = float(np.linalg.svd(KQ, compute_uv=False).min()) if KQ.size else 0.0; dim = len(basis)
        supp = pbar > 1e-12; rho = float(pbar[supp].min())
        C = (2.0 / (rho * sigma ** 2)) if sigma > 1e-12 and np.isfinite(sigma) else float("inf")
        ok = (Dma <= C * dPma + 1e-9) if np.isfinite(C) else True
        assert ok, (label, m, a, Dma, dPma, C)
        Dtot += wm * Dma; dPtot += wm * dPma; worstC = max(worstC, C) if dim > 0 else worstC
        rows.append((m, a, dim, sigma, rho, C, Dma, dPma))
    print(f"{label}: t=2  D = {Dtot:.4f} nats, dP = {dPtot:.4f} nats, ratio D/dP = {(Dtot / dPtot) if dPtot > 1e-12 else float('inf'):.2f}, worst constant C = {worstC:.1f}")
    for m, a, dim, sigma, rho, C, Dma, dPma in rows:
        print(f"    class m={m} a={a}: regimes span dim {dim}, sigma {sigma:.3f}, rho {rho:.3f}, C = 2/(rho sigma^2) = {C:.1f}; D_ma {Dma:.4f} <= C dP_ma = {C * dPma if np.isfinite(C) else float('inf'):.4f}")
u2 = lambda t_, m: {0: .5, 1: .5}
sandwich("W1, policy reacts to last reward, memory = last reward", W1("I"), lambda o, a: o[-1][2], lambda t_, m: bern(0.85 if m else 0.20))
sandwich("W3s, policy reacts to the class, memory = class", W3s(), lambda o, a: (0 if o[-1][0] < 2 else 1), lambda t_, m: bern([0.2, 0.8][m]))
sandwich("W11, policy plays last reward, memory = last reward", W11(), lambda o, a: o[-1][1], lambda t_, m: {m: 1.0})
sandwich("W11, uniform policy, constant memory (counterexample)", W11(), lambda o, a: 0, u2)
