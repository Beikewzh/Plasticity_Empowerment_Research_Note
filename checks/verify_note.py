#!/usr/bin/env python3
"""Exact numerical checks of every numbered statement in main.tex
("The State Defect, Empowerment and Plasticity").

A small random world is enumerated exactly: observations O = (R, Obar) in {0,1}^2,
actions in {0,1}, a memory with two values updated by a random map
M_{t+1} = f(M_t, A_t, O_{t+1}) (M_1 = f1(O_1)), a random stochastic policy pi(a|m),
and an environment whose law of the next observation is drawn at random for every
(history, action) pair. All quantities are computed from the exact law of the
trajectory (O_1, A_1, ..., O_{T+1}, A_{T+1}). Each check is labelled with the number
of the statement it tests. Logarithms are natural. Pure Python.

Section 6 (values of later policies) uses the same random worlds, a world built so that the
conditions of Proposition 6.8 hold (InfoWorld), and the three memories of Example 6.9 (KeyWorld).
"""
import math
import random
import sys
from collections import defaultdict

OBS = [(r, b) for r in (0, 1) for b in (0, 1)]
ACTS = (0, 1)
MEMS = (0, 1)
TOL = 1e-9          # identities
FD_TOL = 1e-6       # finite-difference derivatives (relative)


def normalize(w):
    s = float(sum(w))
    return [x / s for x in w]


# ---------------------------------------------------------------- the world
class World:
    """Environment, recursive memory and policy. If rand_step = t, the memory
    at time t is a randomised encoding q(m | h_t) of the whole history; the
    updates after t are the recursive map f."""

    def __init__(self, seed, T, gamma, rand_step=None):
        self.seed, self.T, self.gamma, self.rand_step = seed, T, gamma, rand_step
        rng = random.Random(seed)
        self.p0 = dict(zip(OBS, normalize([rng.random() + 0.1 for _ in OBS])))
        self.f0 = {o: rng.choice(MEMS) for o in OBS}
        self.f = {(m, a, o): rng.choice(MEMS) for m in MEMS for a in ACTS for o in OBS}
        # make sure the memory is not constant
        self.f0[OBS[0]], self.f0[OBS[1]] = 0, 1
        self.pi = {m: dict(zip(ACTS, normalize([rng.random() + 0.2 for _ in ACTS]))) for m in MEMS}
        self._env = {}
        self._enc = {}

    def first_mem(self, o):
        return self.f0[o]

    def next_mem(self, m, a, o):
        return self.f[(m, a, o)]

    def beh(self, m):
        return self.pi[m]

    def clone(self):
        w = World(self.seed, self.T, self.gamma, self.rand_step)
        w.pi = {m: dict(d) for m, d in self.pi.items()}
        w._enc = {h: dict(d) for h, d in self._enc.items()}
        return w

    def env(self, h, a):
        key = (h, a)
        if key not in self._env:
            r = random.Random(f"{self.seed}|env|{h!r}|{a}")
            self._env[key] = dict(zip(OBS, normalize([r.random() + 0.05 for _ in OBS])))
        return self._env[key]

    def enc(self, h):
        if h not in self._enc:
            r = random.Random(f"{self.seed}|enc|{h!r}")
            self._enc[h] = dict(zip(MEMS, normalize([r.random() + 0.2 for _ in MEMS])))
        return self._enc[h]

    def trajectories(self):
        """List of (obs[0..T], acts[0..T], mems[0..T], prob), index i <-> time i+1."""
        T, out = self.T, []

        def rec(obs, acts, mems, prob):
            t = len(obs)
            h = (obs, acts)
            if self.rand_step == t:
                cands = self.enc(h)
            elif t == 1:
                cands = {self.f0[obs[0]]: 1.0}
            else:
                cands = {self.f[(mems[-1], acts[-1], obs[-1])]: 1.0}
            for m, pm in cands.items():
                for a, pa in self.pi[m].items():
                    if t == T + 1:
                        out.append((obs, acts + (a,), mems + (m,), prob * pm * pa))
                    else:
                        for o, po in self.env(h, a).items():
                            rec(obs + (o,), acts + (a,), mems + (m,), prob * pm * pa * po)

        for o, po in self.p0.items():
            rec((o,), (), (), po)
        return out


# ---------------------------------------------------------------- exact joint law
class Joint:
    """Exact law of finitely many outcomes; random variables are named 'atoms'."""

    def __init__(self, probs, atom_fn):
        self.P, self.N, self._atom_fn = probs, len(probs), atom_fn
        self._vals, self._H = {}, {}

    def vals(self, a):
        if a not in self._vals:
            self._vals[a] = self._atom_fn(a)
        return self._vals[a]

    def keys(self, atoms):
        atoms = list(atoms)
        if not atoms:
            return [()] * self.N
        return list(zip(*[self.vals(a) for a in atoms]))

    def H(self, atoms):
        k = tuple(sorted(set(atoms)))
        if k not in self._H:
            acc = defaultdict(float)
            for key, p in zip(self.keys(k), self.P):
                acc[key] += p
            self._H[k] = -sum(p * math.log(p) for p in acc.values() if p > 0)
        return self._H[k]

    def I(self, X, Y, Z=()):
        X, Y, Z = list(X), list(Y), list(Z)
        return self.H(X + Z) + self.H(Y + Z) - self.H(X + Y + Z) - self.H(Z)

    def CI(self, X, Y, W, Z=()):
        """Co-information I(X;Y;W | Z) = I(X;Y|Z) - I(X;Y|Z,W)."""
        return self.I(X, Y, Z) - self.I(X, Y, list(Z) + list(W))

    def Hc(self, X, Z=()):
        return self.H(list(X) + list(Z)) - self.H(list(Z))

    def E(self, values):
        return sum(p * v for p, v in zip(self.P, values))

    def cexp(self, values, atoms):
        num, den = defaultdict(float), defaultdict(float)
        ks = self.keys(atoms)
        for k, p, v in zip(ks, self.P, values):
            num[k] += p * v
            den[k] += p
        d = {k: num[k] / den[k] for k in den if den[k] > 0}
        return [d[k] for k in ks], d

    def cdist(self, target, atoms):
        acc, den = defaultdict(lambda: defaultdict(float)), defaultdict(float)
        for kc, kt, p in zip(self.keys(atoms), self.keys(target), self.P):
            acc[kc][kt] += p
            den[kc] += p
        return {kc: {kt: v / den[kc] for kt, v in d.items()} for kc, d in acc.items()}


def world_joint(w):
    tr = w.trajectories()
    T, g = w.T, w.gamma

    def atom(a):
        kind, t = a
        if kind == 'O':
            return [x[0][t - 1] for x in tr]
        if kind == 'A':
            return [x[1][t - 1] for x in tr]
        if kind == 'M':
            return [x[2][t - 1] for x in tr]
        if kind == 'R':
            return [x[0][t - 1][0] for x in tr]
        if kind == 'Ob':
            return [x[0][t - 1][1] for x in tr]
        if kind == 'h':
            return [(x[0][:t], x[1][:t - 1]) for x in tr]
        if kind == 'U':
            if t == 1:
                return [x[0][0] for x in tr]
            return [(x[2][t - 2], x[1][t - 2], x[0][t - 1]) for x in tr]
        if kind == 'G':
            return [sum(g ** k * x[0][t + k][0] for k in range(T - t + 1)) for x in tr]
        raise KeyError(a)

    return Joint([x[3] for x in tr], atom)


def Y(s, k):
    """Y_{s,k} = (M_s, A_s, O_{s+1}, ..., A_{s+k-1}, O_{s+k})."""
    out = [('M', s)]
    for i in range(k):
        out += [('A', s + i), ('O', s + i + 1)]
    return out


# ---------------------------------------------------------------- bookkeeping

class Rec:
    def __init__(self):
        self.err = defaultdict(float)
        self.size = defaultdict(float)
        self.n = defaultdict(int)
        self.kind = {}

    def eq(self, name, a, b):
        self.err[name] = max(self.err[name], abs(a - b))
        self.size[name] = max(self.size[name], abs(a), abs(b))
        self.n[name] += 1
        self.kind[name] = 'eq'

    def le(self, name, a, b):
        self.err[name] = max(self.err[name], max(0.0, a - b))
        self.size[name] = max(self.size[name], abs(a), abs(b))
        self.n[name] += 1
        self.kind[name] = 'le'

    def fd(self, name, numeric, formula):
        rel = abs(numeric - formula) / max(1e-12, abs(formula), abs(numeric))
        self.err[name] = max(self.err[name], rel)
        self.size[name] = max(self.size[name], abs(formula))
        self.n[name] += 1
        self.kind[name] = 'fd'

    def report(self):
        ok_all = True
        width = max(len(k) for k in self.err)
        print(f"{'check':<{width}}  {'n':>6}  {'max error':>10}  {'max |value|':>11}  result")
        for name in self.err:
            tol = FD_TOL if self.kind[name] == 'fd' else TOL
            ok = self.err[name] <= tol
            ok_all &= ok
            print(f"{name:<{width}}  {self.n[name]:>6}  {self.err[name]:>10.2e}  {self.size[name]:>11.4f}  {'PASS' if ok else 'FAIL'}")
        return ok_all



# ---------------------------------------------------------------- checks
def check_one_step(J, w, rec):
    T = w.T
    for t in range(1, T + 1):
        h, M, A, O1, A1, M1 = ('h', t), ('M', t), ('A', t), ('O', t + 1), ('A', t + 1), ('M', t + 1)
        R1, B1 = ('R', t + 1), ('Ob', t + 1)
        # Lemma 2.1, k = 0, 1
        rec.eq('Lem 2.1(a) H(A_t|h_t)=H(A_t|M_t)', J.Hc([A], [h]), J.Hc([A], [M]))
        rec.eq('Lem 2.1(b) H(A_t+1|h_t,A_t,O_t+1)=H(A_t+1|M_t,A_t,O_t+1)', J.Hc([A1], [h, A, O1]), J.Hc([A1], [M, A, O1]))
        rec.eq('Lem 2.1 I(h_t;A_t|M_t)=0', J.I([h], [A], [M]), 0.0)
        rec.eq('Lem 2.1 I(h_t;A_t+1|M_t,A_t,O_t+1)=0', J.I([h], [A1], [M, A, O1]), 0.0)
        E, EM = J.I([A], [O1], [h]), J.I([A], [O1], [M])
        P, PM = J.I([O1], [A1], [h, A]), J.I([O1], [A1], [M, A])
        D, D0, Dp = J.I([h], [O1], [M, A]), J.I([h], [O1], [M]), J.I([h], [O1], [M, A, A1])
        dE, dP = J.I([h], [A], [M, O1]), J.I([h], [A1], [M, A])
        rec.eq('Prop 3.2 E-E(M) = I(h;A|M,O\')', E - EM, dE)
        rec.eq('Prop 3.3 P(M)-P = I(h;A\'|M,A)', PM - P, dP)
        rec.eq('Prop 4.2 I(h,A;O\'|M) = D0+E', J.I([h, A], [O1], [M]), D0 + E)
        rec.eq('Prop 4.2 I(h,A;O\'|M) = E(M)+D', J.I([h, A], [O1], [M]), EM + D)
        rec.eq('Prop 4.2 D = D0 + dE', D, D0 + dE)
        rec.eq('Prop 4.3 I(h;O\',A\'|M,A) = D', J.I([h], [O1, A1], [M, A]), D)
        rec.eq('Prop 4.3 D = dP + D+', D, dP + Dp)
        rec.eq('Rem 4.4 I(h;A,O\',A\'|M) = D', J.I([h], [A, O1, A1], [M]), D)
        rec.le('Cor 4.5(b) dP <= D', dP, D)
        rec.le('Cor 4.5(c) dE <= H(A|M)', dE, J.Hc([A], [M]))
        for X, Yv in ((R1, B1), (B1, R1)):
            rec.eq('Prop 5.1 D split', D, J.I([h], [X], [M, A]) + J.I([h], [Yv], [M, A, X]))
            rec.eq('Prop 5.1 D0 split', D0, J.I([h], [X], [M]) + J.I([h], [Yv], [M, X]))
            rec.eq('Prop 5.1 D+ split', Dp, J.I([h], [X], [M, A, A1]) + J.I([h], [Yv], [M, A, A1, X]))
            xE = J.I([A], [X], [h]) - J.I([A], [X], [M])
            yE = J.I([A], [Yv], [h, X]) - J.I([A], [Yv], [M, X])
            rec.eq('Prop 5.2 dE = X part + Y part', dE, xE + yE)
            rec.eq('Prop 5.2 X part = I(h;A|M,X)', xE, J.I([h], [A], [M, X]))
            xP = J.I([X], [A1], [M, A]) - J.I([X], [A1], [h, A])
            yP = J.I([Yv], [A1], [M, A, X]) - J.I([Yv], [A1], [h, A, X])
            rec.eq('Prop 5.3 dP = X part + Y part', dP, xP + yP)
            rec.eq('Prop 5.3 Y part = I(h;A\'|M,A,X)', yP, J.I([h], [A1], [M, A, X]))
            Xl, Yl = J.I([h], [X], [M, A]), J.I([h], [Yv], [M, A, X])
            rec.eq('Prop 5.4 X part, first split', Xl, J.I([h], [X], [M]) + xE)
            rec.eq('Prop 5.4 X part, second split', Xl, xP + J.I([h], [X], [M, A, A1]))
            rec.eq('Prop 5.4 Y part, first split', Yl, J.I([h], [Yv], [M, X]) + yE)
            rec.eq('Prop 5.4 Y part, second split', Yl, yP + J.I([h], [Yv], [M, A, A1, X]))
        rec.eq('Prop 5.5 D', J.I([h], [R1], [M, A]) - J.I([h], [R1], [M, A, B1]), J.CI([h], [R1], [B1], [M, A]))
        rec.eq('Prop 5.5 D0', J.I([h], [R1], [M]) - J.I([h], [R1], [M, B1]), J.CI([h], [R1], [B1], [M]))
        rec.eq('Prop 5.5 D+', J.I([h], [R1], [M, A, A1]) - J.I([h], [R1], [M, A, A1, B1]), J.CI([h], [R1], [B1], [M, A, A1]))
        rec.eq('Prop 5.5 dE',
               (J.I([A], [R1], [h]) - J.I([A], [R1], [M])) - (J.I([A], [R1], [h, B1]) - J.I([A], [R1], [M, B1])),
               J.CI([A], [R1], [B1], [h]) - J.CI([A], [R1], [B1], [M]))
        rec.eq('Prop 5.5 dP',
               (J.I([R1], [A1], [M, A]) - J.I([R1], [A1], [h, A])) - (J.I([R1], [A1], [M, A, B1]) - J.I([R1], [A1], [h, A, B1])),
               J.CI([R1], [A1], [B1], [M, A]) - J.CI([R1], [A1], [B1], [h, A]))
        lo = max(0.0, J.I([h], [R1], [M, A]) - J.I([h], [B1], [M, A]))
        hi = min(J.I([h], [R1], [M, A]), J.I([h], [R1], [M, A, B1]))
        rec.le('Prop 5.7(c) bounds on D^{R only} consistent', lo, hi)


def check_steps(J, w, rec):
    """Lemma 2.1 for every k."""
    T = w.T
    for t in range(1, T + 1):
        h = ('h', t)
        for k in range(0, T + 2 - t):
            rec.eq('Lem 2.1 I(h_t;A_t+k|M_t,A_t:t+k-1,O_t+1:t+k)=0', J.I([h], [('A', t + k)], Y(t, k)), 0.0)


# ================================================================ Section 6
# Worlds used here provide: T, gamma, p0, env(h, a), first_mem(o), next_mem(m, a, o), beh(m).
# A history is h = (observations O_1..O_n, actions A_1..A_{n-1}); the reward is o[0].
# Trajectories run to (O_T, A_T); values use the final time T, so G_n = R_{n+1} + ... + gamma^{T-n-1} R_T.

LOG2 = math.log(2)


def mem6(w, h):
    cache = w.__dict__.setdefault('_mem6', {})
    if h not in cache:
        obs, acts = h
        m = w.first_mem(obs[0])
        for k in range(1, len(obs)):
            m = w.next_mem(m, acts[k - 1], obs[k])
        cache[h] = m
    return cache[h]


def joint6(w):
    T, tr = w.T, []

    def rec(obs, acts, mems, prob):
        t = len(obs)
        m = w.first_mem(obs[0]) if t == 1 else w.next_mem(mems[-1], acts[-1], obs[-1])
        for a, pa in w.beh(m).items():
            if pa <= 0:
                continue
            if t == T:
                tr.append((obs, acts + (a,), mems + (m,), prob * pa))
                continue
            for o, po in w.env((obs, acts), a).items():
                if po > 0:
                    rec(obs + (o,), acts + (a,), mems + (m,), prob * pa * po)

    for o, po in w.p0.items():
        if po > 0:
            rec((o,), (), (), po)

    def atom(a):
        kind, t = a
        if kind == 'O':
            return [x[0][t - 1] for x in tr]
        if kind == 'A':
            return [x[1][t - 1] for x in tr]
        if kind == 'M':
            return [x[2][t - 1] for x in tr]
        if kind == 'R':
            return [x[0][t - 1][0] for x in tr]
        if kind == 'h':
            return [(x[0][:t], x[1][:t - 1]) for x in tr]
        raise KeyError(a)

    J = Joint([x[3] for x in tr], atom)
    hists = {n: sorted(set(J.vals(('h', n))), key=repr) for n in range(1, T + 1)}
    return J, hists


def q_paths(w, pil, n, h, a):
    """E^{pi'}[G_n | h_n = h, A_n = a], by enumerating every continuation (no Bellman equation used)."""
    T, g = w.T, w.gamma
    total = 0.0

    def rec(hk, ak, prob, ret):
        nonlocal total
        obs, acts = hk
        k = len(obs)
        mk = mem6(w, hk)
        for o, po in w.env(hk, ak).items():
            if po <= 0:
                continue
            r = ret + g ** (k - n) * o[0]
            if k + 1 == T:
                total += prob * po * r
                continue
            h1 = (obs + (o,), acts + (ak,))
            m1 = w.next_mem(mk, ak, o)
            for a1, pa in pil[k + 1](m1).items():
                if pa > 0:
                    rec(h1, a1, prob * po * pa, r)

    rec(h, a, 1.0, 0.0)
    return total


def path_sum(w, pil, n, h, a, fn):
    """E^{pi'}[ sum_{k=n}^{T-1} gamma^{k-n} fn(k, h_k, A_k) | h_n = h, A_n = a ], by enumerating paths."""
    T, g = w.T, w.gamma
    total = 0.0

    def rec(hk, ak, k, prob):
        nonlocal total
        total += prob * g ** (k - n) * fn(k, hk, ak)
        if k + 1 == T:
            return
        obs, acts = hk
        mk = mem6(w, hk)
        for o, po in w.env(hk, ak).items():
            if po <= 0:
                continue
            h1 = (obs + (o,), acts + (ak,))
            m1 = w.next_mem(mk, ak, o)
            for a1, pa in pil[k + 1](m1).items():
                if pa > 0:
                    rec(h1, a1, k + 1, prob * po * pa)

    rec(h, a, n, 1.0)
    return total


def q_bellman_hist(w, pil, hists, t):
    """Proposition 6.2: Q^{pi'}_n(h, a) by the Bellman equation on the history."""
    T, g = w.T, w.gamma
    Q = {}

    def V(n1, h1):
        if n1 == T:
            return 0.0
        return sum(p * Q[(n1, h1, a1)] for a1, p in pil[n1](mem6(w, h1)).items() if p > 0)

    for n in range(T - 1, t - 1, -1):
        for h in hists[n]:
            m = mem6(w, h)
            for a in ACTS_OF(w, h):
                s = 0.0
                for o, po in w.env(h, a).items():
                    if po <= 0:
                        continue
                    h1 = (h[0] + (o,), h[1] + (a,))
                    s += po * (o[0] + g * V(n + 1, h1))
                Q[(n, h, a)] = s
    return Q


def ACTS_OF(w, h):
    """Every action is evaluated at every history that occurs (the current action is set)."""
    return list(w.beh(mem6(w, h)).keys())


def q_hat(w, J, pil, t, via='O'):
    """Definition 6.3 (via='O'), or the form of Lemma 6.4 (via='RM')."""
    T, g = w.T, w.gamma
    Qh, Vh = {}, {}
    for n in range(T - 1, t - 1, -1):
        if via == 'O':
            model = J.cdist([('O', n + 1)], [('M', n), ('A', n)])
            for (m, a), d in model.items():
                s = 0.0
                for (o,), p in d.items():
                    nxt = 0.0 if n + 1 == T else Vh[(n + 1, w.next_mem(m, a, o))]
                    s += p * (o[0] + g * nxt)
                Qh[(n, m, a)] = s
        else:
            _, er = J.cexp(J.vals(('R', n + 1)), [('M', n), ('A', n)])
            mm = J.cdist([('M', n + 1)], [('M', n), ('A', n)])
            for (m, a), d in mm.items():
                nxt = 0.0 if n + 1 == T else sum(p * Vh[(n + 1, m1)] for (m1,), p in d.items())
                Qh[(n, m, a)] = er[(m, a)] + g * nxt
        if n > t:
            for m in set(k[0] for k in J.cdist([('A', n)], [('M', n)])):
                Vh[(n, m)] = sum(p * Qh[(n, m, a)] for a, p in pil[n](m).items() if p > 0)
    return Qh, Vh


def hist_preds(w, h, a):
    """E[R_{n+1} | h_n, a] and P(M_{n+1} | h_n, a), from the environment and the memory map only."""
    m = mem6(w, h)
    er, pm = 0.0, defaultdict(float)
    for o, po in w.env(h, a).items():
        er += po * o[0]
        pm[w.next_mem(m, a, o)] += po
    return er, dict(pm)


def kl(p, q):
    return sum(pv * math.log(pv / q[k]) for k, pv in p.items() if pv > 0)


def check_sec6_terms(J, w, hists, rec, tag=''):
    """Lemma 6.1(b) for O and M', Proposition 6.6."""
    T = w.T
    out = {}
    for n in range(1, T):
        h, M, A, O1, A1, M1, R1 = ('h', n), ('M', n), ('A', n), ('O', n + 1), ('A', n + 1), ('M', n + 1), ('R', n + 1)
        I = J.I
        D, DM = I([h], [O1], [M, A]), I([h], [M1], [M, A])
        dP, Dp = I([h], [A1], [M, A]), I([h], [O1], [M, A, A1])
        rest_a = I([h], [O1], [M, A, M1])
        b1, b2 = I([h], [M1], [M]), I([h], [A], [M, M1])
        c2 = I([h], [M1], [M, A, A1])
        d2 = I([h], [O1], [M, A, A1, M1])
        rec.eq('Prop 6.6(a) D_n = D^M_n + I(h;O\'|M,A,M\')' + tag, D, DM + rest_a)
        rec.eq('Prop 6.6(b) D^M_n = I(h;M\'|M) + I(h;A|M,M\')' + tag, DM, b1 + b2)
        rec.eq('Prop 6.6(b) I(A;M\'|h)-I(A;M\'|M) = I(h;A|M,M\')' + tag, I([A], [M1], [h]) - I([A], [M1], [M]), b2)
        rec.eq('Prop 6.6(c) I(h;A\'|M,A,M\') = 0' + tag, I([h], [A1], [M, A, M1]), 0.0)
        rec.eq('Prop 6.6(c) D^M_n = dP_n + I(h;M\'|M,A,A\')' + tag, DM, dP + c2)
        rec.eq('Prop 6.6(d) D+_n = I(h;M\'|M,A,A\') + I(h;O\'|M,A,A\',M\')' + tag, Dp, c2 + d2)
        rec.le('Prop 6.6 dP_n <= D^M_n' + tag, dP, DM)
        rec.le('Prop 6.6 D^M_n <= D_n' + tag, DM, D)
        rec.eq('Prop 6.6 last term of (a) = last term of (d)' + tag, rest_a, d2)
        rec.eq('T6 I(h;M\'|M,A,O\') = 0' + tag, I([h], [M1], [M, A, O1]), 0.0)
        # Lemma 6.1(b): each information is the average divergence between the two predictions
        pO = J.cdist([O1], [M, A])
        pM = J.cdist([M1], [M, A])
        avgO = avgM = 0.0
        acc = defaultdict(float)
        for (hh, aa), p in zip(J.keys([h, A]), J.P):
            acc[(hh, aa)] += p
        for (hh, aa), p in acc.items():
            m = mem6(w, hh)
            env = {(o,): q for o, q in w.env(hh, aa).items() if q > 0}
            avgO += p * kl(env, pO[(m, aa)])
            _, hm = hist_preds(w, hh, aa)
            avgM += p * kl({(k,): v for k, v in hm.items() if v > 0}, pM[(m, aa)])
        rec.eq('Lem 6.1(b) D_n = E KL(P(O\'|h,A) || P(O\'|M,A))' + tag, D, avgO)
        rec.eq('Lem 6.1(b) D^M_n = E KL(P(M\'|h,A) || P(M\'|M,A))' + tag, DM, avgM)
        out[n] = dict(D=D, DM=DM, dP=dP, Dp=Dp, D0=I([h], [O1], [M]), dE=I([h], [A], [M, O1]),
                      rew=I([h], [R1], [M, A]), b1=b1, b2=b2, c2=c2)
    return out


def check_sec6_values(J, w, hists, pil, rec, t=1, tag=''):
    """Proposition 6.2, Lemma 6.4, Proposition 6.7 for the later policy pil."""
    T, g = w.T, w.gamma
    Qp = {}
    for n in range(t, T):
        for h in hists[n]:
            for a in ACTS_OF(w, h):
                Qp[(n, h, a)] = q_paths(w, pil, n, h, a)
    Qb = q_bellman_hist(w, pil, hists, t)
    for k, v in Qp.items():
        rec.eq('Prop 6.2 Bellman equation on the history' + tag, v, Qb[k])
    Qh, Vh = q_hat(w, J, pil, t, 'O')
    Qh2, _ = q_hat(w, J, pil, t, 'RM')
    for k, v in Qh.items():
        rec.eq('Lem 6.4 memory equation via E[R|M,A] and P(M\'|M,A)' + tag, v, Qh2[k])
    ERm = {}
    for n in range(t, T):
        _, ERm[n] = J.cexp(J.vals(('R', n + 1)), [('M', n), ('A', n)])
    PMm = {n: J.cdist([('M', n + 1)], [('M', n), ('A', n)]) for n in range(t, T)}

    def gap(n, h, a):
        return Qp[(n, h, a)] - Qh[(n, mem6(w, h), a)]

    def delta(n, h, a):
        er, _ = hist_preds(w, h, a)
        return er - ERm[n][(mem6(w, h), a)]

    def tau(n, h, a):
        if n + 1 == T:
            return 0.0
        _, pm = hist_preds(w, h, a)
        mm = PMm[n][(mem6(w, h), a)]
        keys = set(pm) | set(k[0] for k in mm)
        return sum((pm.get(k, 0.0) - mm.get((k,), 0.0)) * Vh[(n + 1, k)] for k in keys)

    for n in range(t, T):
        for h in hists[n]:
            m = mem6(w, h)
            for a in ACTS_OF(w, h):
                nxt = 0.0
                if n + 1 < T:
                    for o, po in w.env(h, a).items():
                        if po <= 0:
                            continue
                        h1 = (h[0] + (o,), h[1] + (a,))
                        m1 = w.next_mem(m, a, o)
                        nxt += po * sum(p * gap(n + 1, h1, a1) for a1, p in pil[n + 1](m1).items() if p > 0)
                rec.eq('Prop 6.7 one step: gap = delta + g tau + g E[next gap]' + tag,
                       gap(n, h, a), delta(n, h, a) + g * tau(n, h, a) + g * nxt)
                unrolled = path_sum(w, pil, n, h, a, lambda k, hk, ak: delta(k, hk, ak) + g * tau(k, hk, ak))
                rec.eq('Prop 6.7 unrolled: gap = E[sum g^k (delta + g tau)]' + tag, gap(n, h, a), unrolled)
    return Qp, Qh, Vh, gap, delta, tau


def random_later_policy(seed, T, t, mems, deterministic=False):
    rng = random.Random(f"later|{seed}")
    table = {}
    for k in range(t + 1, T):
        for m in mems:
            if deterministic:
                b = rng.choice(ACTS)
                table[(k, m)] = {a: (1.0 if a == b else 0.0) for a in ACTS}
            else:
                table[(k, m)] = dict(zip(ACTS, normalize([rng.random() + 0.05 for _ in ACTS])))
    return {k: (lambda m, k=k: table[(k, m)]) for k in range(t + 1, T)}


def behaviour_as_later(w, T, t):
    return {k: (lambda m: w.beh(m)) for k in range(t + 1, T)}


class InfoWorld:
    """Built so that the conditions of Proposition 6.8 hold: given (M_n, A_n), the law of the pair
    (R_{n+1}, M_{n+1}) is fixed, and how it is spread over the observations of each pair depends on
    the whole history. The behaviour is drawn separately (beh_seed); the environment is not."""
    OBS = [(r, b) for r in (0, 1) for b in (0, 1, 2)]

    def __init__(self, seed, T, gamma, beh_seed):
        rng = random.Random(f"info|{seed}")
        self.seed, self.T, self.gamma = seed, T, gamma
        self.p0 = dict(zip(self.OBS, normalize([rng.random() + 0.1 for _ in self.OBS])))
        self.f0 = {o: rng.choice(MEMS) for o in self.OBS}
        self.f0[self.OBS[0]], self.f0[self.OBS[1]] = 0, 1
        self.f = {(m, a, o): rng.choice(MEMS) for m in MEMS for a in ACTS for o in self.OBS}
        self.q = {}
        for m in MEMS:
            for a in ACTS:
                groups = sorted(set((o[0], self.f[(m, a, o)]) for o in self.OBS))
                self.q[(m, a)] = dict(zip(groups, normalize([rng.random() + 0.1 for _ in groups])))
        rb = random.Random(f"info|{seed}|beh|{beh_seed}")
        self.pi = {m: dict(zip(ACTS, normalize([rb.random() + 0.2 for _ in ACTS]))) for m in MEMS}
        self._env = {}

    def first_mem(self, o):
        return self.f0[o]

    def next_mem(self, m, a, o):
        return self.f[(m, a, o)]

    def beh(self, m):
        return self.pi[m]

    def env(self, h, a):
        key = (h, a)
        if key not in self._env:
            m = mem6(self, h)
            r = random.Random(f"info|{self.seed}|env|{h!r}|{a}")
            out = {}
            for (rew, m1), qg in self.q[(m, a)].items():
                members = [o for o in self.OBS if o[0] == rew and self.f[(m, a, o)] == m1]
                for o, p in zip(members, normalize([r.random() + 0.05 for _ in members])):
                    out[o] = qg * p
            self._env[key] = out
        return self._env[key]


class KeyWorld:
    """Example 6.9. memory = 'C' (M_1 = c), 'S' (M_1 = s) or 'C2' (M_1 = c, M_2 also stores s)."""

    def __init__(self, memory):
        self.T, self.gamma, self.memory = 3, 1.0, memory
        self.p0 = {(0, (c, s)): 0.25 for c in (0, 1) for s in (0, 1)}

    def env(self, h, a):
        obs, acts = h
        n = len(obs)
        if n == 1:
            c, s = obs[0][1]
            if a == c:
                return {(0, (0, s)): 0.5, (0, (1, s)): 0.5}
            return {(0, ('blank', s)): 1.0}
        if n == 2:
            x = obs[1][1][0]
            if x == 'blank':
                return {(0, 0): 0.5, (1, 0): 0.5}
            return {(1 if a == x else 0, 0): 1.0}
        return {(0, 0): 1.0}

    def first_mem(self, o):
        c, s = o[1]
        return s if self.memory == 'S' else c

    def next_mem(self, m, a, o):
        if isinstance(o[1], tuple):
            x, s = o[1]
            return (x, s) if self.memory == 'C2' else x
        return 'end'

    def beh(self, m):
        return {0: 0.5, 1: 0.5}


def key_later(m):
    x = m[0] if isinstance(m, tuple) else m
    if x in (0, 1):
        return {x: 1.0, 1 - x: 0.0}
    return {0: 0.5, 1: 0.5}


def check_example(rec):
    expect = {  # in units of log 2
        'C': dict(D=1, D0=1, dE=0, dP=0, Dp=1, DM=0),
        'S': dict(D=1, D0=0, dE=1, dP=0, Dp=1, DM=1, b1=0, b2=1, c2=1),
        'C2': dict(D=1, D0=1, dE=0, dP=0, Dp=1, DM=1),
    }
    for mem, ex in expect.items():
        w = KeyWorld(mem)
        J, hists = joint6(w)
        tag = f' [Ex 6.9 {mem}]'
        terms = check_sec6_terms(J, w, hists, rec, tag)
        for k, v in ex.items():
            rec.eq(f'Ex 6.9 memory {mem}: {k}_1 / log 2 = {v}', terms[1][k] / LOG2, v)
        rec.eq(f'Ex 6.9 memory {mem}: reward parts at n = 1, 2 are 0', terms[1]['rew'] + terms[2]['rew'], 0.0)
        beh = behaviour_as_later(w, 3, 1)
        Qp, Qh, Vh, gap, delta, tau = check_sec6_values(J, w, hists, beh, rec, 1, tag + ' pi')
        for h in hists[1]:
            for a in (0, 1):
                rec.eq(f'Ex 6.9 memory {mem}: Q^pi_1 = 1/2', Qp[(1, h, a)], 0.5)
                rec.eq(f'Ex 6.9 memory {mem}: error of the critic of pi = 0', gap(1, h, a), 0.0)
                rec.eq(f'Ex 6.9 memory {mem}: tau^pi_1 = 0', tau(1, h, a), 0.0)
        lat = {2: key_later}
        Qp, Qh, Vh, gap, delta, tau = check_sec6_values(J, w, hists, lat, rec, 1, tag + " pi'")
        for h in hists[1]:
            c = h[0][0][1][0]
            for a in (0, 1):
                rec.eq(f"Ex 6.9 memory {mem}: Q^pi'_1 = 1 if a = c else 1/2", Qp[(1, h, a)], 1.0 if a == c else 0.5)
                want = (0.25 if a == c else -0.25) if mem == 'S' else 0.0
                rec.eq(f"Ex 6.9 memory {mem}: error under pi' = {'+-1/4' if mem == 'S' else '0'}", gap(1, h, a), want)
                rec.eq(f"Ex 6.9 memory {mem}: tau^pi'_1 = error under pi'", tau(1, h, a), want)
        if mem == 'S':
            for (n, m, a), v in Qh.items():
                if n == 1:
                    rec.eq("Ex 6.9 memory S: hat Q^pi'_1 = 3/4", v, 0.75)


def check_info_world(rec):
    for seed, T in ((11, 4), (12, 3), (13, 4)):
        worlds = [InfoWorld(seed, T, 0.9, b) for b in (0, 1)]
        qhats = []
        for i, w in enumerate(worlds):
            J, hists = joint6(w)
            tag = ' [info world]'
            terms = check_sec6_terms(J, w, hists, rec, tag)
            for n in range(1, T):
                rec.eq('Prop 6.8 world: reward part = 0', terms[n]['rew'], 0.0)
                rec.eq('Prop 6.8 world: D^M_n = 0', terms[n]['DM'], 0.0)
                rec.le('Prop 6.8 world: D_n >= 0.001 (the world is not trivial)', 1e-3, terms[n]['D'])
            pols = [random_later_policy(f"{seed}|{j}", T, 1, MEMS, deterministic=(j == 2)) for j in range(3)]
            pols.append(behaviour_as_later(w, T, 1))
            qh_this = []
            for j, pil in enumerate(pols):
                Qp, Qh, Vh, gap, delta, tau = check_sec6_values(J, w, hists, pil, rec, 1, tag)
                for (n, h, a) in Qp:
                    rec.eq('Prop 6.8(a) Q^pi\'_n(h,a) = hat Q^pi\'_n(M_n,a) for every later policy', gap(n, h, a), 0.0)
                qh_this.append(Qh)
            qhats.append(qh_this)
            print(f"  info world seed {seed} behaviour {i} (T={T}, {J.N} trajectories) done", flush=True)
        for Qa, Qb in zip(qhats[0][:3], qhats[1][:3]):  # the three later policies shared by both behaviours
            for k in Qa:
                rec.eq('Prop 6.8(b) hat Q^pi\' is the same under a second behaviour', Qa[k], Qb[k])


def check_random_sec6(rec):
    for seed, T in ((1, 4), (2, 4), (3, 3), (4, 3)):
        w = World(seed, T, 0.9)
        J, hists = joint6(w)
        check_sec6_terms(J, w, hists, rec)
        pols = [random_later_policy(f"rand|{seed}|{j}", T, 1, MEMS, deterministic=(j == 1)) for j in range(2)]
        pols.append(behaviour_as_later(w, T, 1))
        for pil in pols:
            check_sec6_values(J, w, hists, pil, rec, 1)
        print(f"  section 6, random world seed {seed} (T={T}, {J.N} trajectories) done", flush=True)


def main():
    rec = Rec()
    for seed, T in ((1, 4), (2, 4), (3, 3), (4, 3), (5, 3)):
        w = World(seed, T, 0.9)
        J = world_joint(w)
        check_one_step(J, w, rec)
        check_steps(J, w, rec)
        print(f"seed {seed} (T={T}, {J.N} trajectories) done", flush=True)
    check_random_sec6(rec)
    check_info_world(rec)
    check_example(rec)
    ok = rec.report()
    print("ALL PASS" if ok else "SOME CHECKS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
