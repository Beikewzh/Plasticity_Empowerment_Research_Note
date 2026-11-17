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


def Z(s, k):
    """Z_{s,k} = (Y_{s,k}, A_{s+k})."""
    return Y(s, k) + [('A', s + k)]


# ---------------------------------------------------------------- bookkeeping
GAPS = defaultdict(lambda: float('-inf'))

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


def own_terms(J, n):
    """The agent's own five terms at time n (Definition 4.1)."""
    h, M, A, O1, A1 = ('h', n), ('M', n), ('A', n), ('O', n + 1), ('A', n + 1)
    return dict(D=J.I([h], [O1], [M, A]), D0=J.I([h], [O1], [M]), dE=J.I([h], [A], [M, O1]),
                dP=J.I([h], [A1], [M, A]), Dp=J.I([h], [O1], [M, A, A1]))


def inputs(t, k):
    """(M_t, A_{t:t+k-1}, O_{t+1:t+k}): what the updates from t+1 to t+k received."""
    return Y(t, k)


def check_steps(J, w, rec):
    T = w.T
    for t in range(1, T + 1):
        h = ('h', t)
        # Lemma 2.1 and Corollary 2.2
        for k in range(0, T + 2 - t):
            Ck = Y(t, k)
            rec.eq('Lem 2.1 I(h_t;A_t+k|M_t,A_t:t+k-1,O_t+1:t+k)=0', J.I([h], [('A', t + k)], Ck), 0.0)
            if t >= 2:
                rec.eq('Cor 2.2 I(X;A_t+k|...)=0 for X=(M_t-1,A_t-1,O_t)', J.I([('M', t - 1), ('A', t - 1), ('O', t)], [('A', t + k)], Ck), 0.0)
                rec.eq('Cor 2.2 I(X;A_t+k|...)=0 for X=O_t', J.I([('O', t)], [('A', t + k)], Ck), 0.0)
        # two steps
        if t + 1 <= T:
            M, A, O1, A1, O2, A2 = ('M', t), ('A', t), ('O', t + 1), ('A', t + 1), ('O', t + 2), ('A', t + 2)
            R1, B1, R2, B2 = ('R', t + 1), ('Ob', t + 1), ('R', t + 2), ('Ob', t + 2)
            M1, h1 = ('M', t + 1), ('h', t + 1)
            U = [M, A, O1]
            D2 = J.I([h], [O1, A1, O2], [M, A])
            a, b = own_terms(J, t), own_terms(J, t + 1)
            mid = J.I([h], [O2], [M, A, O1, A1])
            rec.eq('Prop 6.2(a) D_t:t+2 = D_t + I(h_t;O_t+2|M_t,A_t,O_t+1,A_t+1)', D2, a['D'] + mid)
            # (b) A_t+1 first; (c) A_t+1 last
            cond = J.I([h], [O1, O2], [M, A, A1])
            free = J.I([h], [O1, O2], [M, A])
            late = J.I([h], [A1], [M, A, O1, O2])
            rec.eq('Prop 6.2(b) D_t:t+2 = dP_t + I(h_t;O_t+1,O_t+2|M_t,A_t,A_t+1)', D2, a['dP'] + cond)
            rec.eq('Prop 6.2 text: I(h_t;O_t+1,O_t+2|M_t,A_t,A_t+1) = D+_t + last term of (a)', cond, a['Dp'] + mid)
            rec.eq('Prop 6.2 text: last term of (a) = I(h_t;O_t+2|M_t,A_t,O_t+1) + last term of (c)', mid, J.I([h], [O2], [M, A, O1]) + late)
            rec.eq('Prop 6.2(c) D_t:t+2 = I(h_t;O_t+1,O_t+2|M_t,A_t) + I(h_t;A_t+1|M_t,A_t,O_t+1,O_t+2)', D2, free + late)
            GAPS['dP_t (lost by A_t+1 in the condition)'] = max(GAPS['dP_t (lost by A_t+1 in the condition)'], a['dP'])
            GAPS['I(h_t;A_t+1|M_t,A_t,O_t+1,O_t+2) (lost by leaving A_t+1 out)'] = max(GAPS['I(h_t;A_t+1|M_t,A_t,O_t+1,O_t+2) (lost by leaving A_t+1 out)'], late)
            GAPS['D_t - I(h_t;O_t+1,O_t+2|M_t,A_t,A_t+1) (largest shortfall below D_t)'] = max(GAPS['D_t - I(h_t;O_t+1,O_t+2|M_t,A_t,A_t+1) (largest shortfall below D_t)'], a['D'] - cond)
            for X, V in (([O2], [A1]), ([R2], [A1]), ([B2], [A1, R2]), ([O2], []), ([A1], [O2]), ([A2], [A1]), ([O2], [A1, A2])):
                rec.eq('Lem 6.3 update from t to t+1', J.I([h1], X, [M1] + V), J.I([h], X, U + V) + J.I(U, X, [M1] + V))
            corr = J.I(U, [O2], [M1, A1])
            rec.eq('Prop 6.4 D_t:t+2 = D_t + D_t+1 - update term', D2, a['D'] + b['D'] - corr)
            br1 = J.I(U, [O2], [M1]) + J.I(U, [A1], [M1, O2])
            br2 = J.I(U, [A2], [M1, A1]) + J.I(U, [O2], [M1, A1, A2])
            rec.eq('Prop 6.5 first bracket = update term', br1, corr)
            rec.eq('Prop 6.5 second bracket = update term', br2, corr)
            rec.eq('Prop 6.5 first split', D2, a['D0'] + a['dE'] + b['D0'] + b['dE'] - br1)
            rec.eq('Prop 6.5 second split', D2, a['dP'] + a['Dp'] + b['dP'] + b['Dp'] - br2)
            rf = (J.I([h], [R1], [M, A]) + J.I([h1], [R2], [M1, A1]) - J.I(U, [R2], [M1, A1])
                  + J.I([h], [B1], [M, A, R1]) + J.I([h1], [B2], [M1, A1, R2]) - J.I(U, [B2], [M1, A1, R2]))
            rec.eq('Prop 6.6 reward first', D2, rf)
            sf = (J.I([h], [B1], [M, A]) + J.I([h1], [B2], [M1, A1]) - J.I(U, [B2], [M1, A1])
                  + J.I([h], [R1], [M, A, B1]) + J.I([h1], [R2], [M1, A1, B2]) - J.I(U, [R2], [M1, A1, B2]))
            rec.eq('Prop 6.6 rest first', D2, sf)
        # K steps
        for K in range(1, T + 2 - t):
            preds = []
            for k in range(1, K + 1):
                preds.append(('O', t + k))
                if k < K:
                    preds.append(('A', t + k))
            DK = J.I([h], preds, [('M', t), ('A', t)])
            terms = [J.I([h], [('O', t + k + 1)], Z(t, k)) for k in range(K)]
            rec.eq('Prop 7.2 D_t:t+K = sum of time-order terms', DK, sum(terms))
            own = [own_terms(J, t + k) for k in range(K)]
            corr = [J.I(inputs(t, k), [('O', t + k + 1)], [('M', t + k), ('A', t + k)]) for k in range(1, K)]
            rec.eq('Prop 7.4 D_t:t+K = sum D_t+k - update terms', DK, sum(x['D'] for x in own) - sum(corr))
            rec.eq('Prop 7.5 first split', DK, sum(x['D0'] + x['dE'] for x in own) - sum(corr))
            rec.eq('Prop 7.5 second split', DK, sum(x['dP'] + x['Dp'] for x in own) - sum(corr))
            rew = sum(J.I([('h', t + k)], [('R', t + k + 1)], [('M', t + k), ('A', t + k)]) for k in range(K))
            rew -= sum(J.I(inputs(t, k), [('R', t + k + 1)], [('M', t + k), ('A', t + k)]) for k in range(1, K))
            rst = sum(J.I([('h', t + k)], [('Ob', t + k + 1)], [('M', t + k), ('A', t + k), ('R', t + k + 1)]) for k in range(K))
            rst -= sum(J.I(inputs(t, k), [('Ob', t + k + 1)], [('M', t + k), ('A', t + k), ('R', t + k + 1)]) for k in range(1, K))
            rec.eq('Prop 7.5 reward first', DK, rew + rst)
            for k in range(1, K):
                n = t + k
                W = inputs(t, k)
                Mn, An, An1, On1 = ('M', n), ('A', n), ('A', n + 1), ('O', n + 1)
                c = J.I(W, [On1], [Mn, An])
                rec.eq('Prop 7.5 subtracted term, first split', c, J.I(W, [On1], [Mn]) + J.I(W, [An], [Mn, On1]))
                rec.eq('Prop 7.5 subtracted term, second split', c, J.I(W, [An1], [Mn, An]) + J.I(W, [On1], [Mn, An, An1]))
        # Lemma 7.3
        for k in range(1, T + 1 - t):
            n = t + k
            hn, Mn, W = ('h', n), ('M', n), inputs(t, k)
            An, An1, On1, Rn1, Bn1 = ('A', n), ('A', n + 1), ('O', n + 1), ('R', n + 1), ('Ob', n + 1)
            for X, V in (([On1], [An]), ([Rn1], [An]), ([Bn1], [An, Rn1]), ([On1], []), ([An], [On1]), ([An1], [An]), ([On1], [An, An1])):
                rec.eq('Lem 7.3 updates from t to t+k', J.I([hn], X, [Mn] + V), J.I([h], X, W + V) + J.I(W, X, [Mn] + V))


def bellman_pieces(J, w, t, K):
    """The memory's Bellman value over [t, t+K] (Definitions 6.7 and 7.6) and, for each n,
    the comparisons c_n, c^R_n, c^Obar_n over all trajectories (Propositions 6.8, 6.9, 7.7)."""
    g, N = w.gamma, J.N
    Qh, Vh = {}, {t + K: defaultdict(float)}
    c, cR, cB = {}, {}, {}
    for n in range(t + K - 1, t - 1, -1):
        hn, Mn, An, On1 = ('h', n), ('M', n), ('A', n), ('O', n + 1)
        Rn1, Bn1 = ('R', n + 1), ('Ob', n + 1)
        PM, Ph = J.cdist([On1], [Mn, An]), J.cdist([On1], [hn, An])
        V1 = Vh[n + 1]

        def wt(m, a, o):
            return o[0] + g * V1[w.f[(m, a, o)]]

        Q = {(m, a): sum(p * wt(m, a, o) for (o,), p in d.items()) for (m, a), d in PM.items()}
        V = {m: sum(w.pi[m][a] * Q[(m, a)] for a in ACTS) for m in MEMS if all((m, a) in Q for a in ACTS)}
        Qh[n], Vh[n] = Q, V
        kh, kM = J.keys([hn, An]), J.keys([Mn, An])
        PRh, PRM = J.cdist([Rn1], [hn, An]), J.cdist([Rn1], [Mn, An])
        PBh, PBM = J.cdist([Bn1], [hn, An, Rn1]), J.cdist([Bn1], [Mn, An, Rn1])
        cn, cRn, cBn = [], [], []
        for i in range(N):
            m, a = kM[i]
            cn.append(sum((Ph[kh[i]].get((o,), 0.0) - p) * wt(m, a, o) for (o,), p in PM[kM[i]].items()))
            cRn.append(sum((PRh[kh[i]].get((r,), 0.0) - pr)
                           * (r + g * sum(pb * V1[w.f[(m, a, (r, b))]] for (b,), pb in PBM[(m, a, r)].items()))
                           for (r,), pr in PRM[kM[i]].items()))
            tot = 0.0
            for (r,), phr in PRh[kh[i]].items():
                tot += phr * g * sum((PBh[kh[i] + (r,)].get((b,), 0.0) - pb) * V1[w.f[(m, a, (r, b))]]
                                     for (b,), pb in PBM[(m, a, r)].items())
            cBn.append(tot)
        c[n], cR[n], cB[n] = cn, cRn, cBn
    return Qh, Vh, c, cR, cB


def returns(J, w, t, K):
    g, N = w.gamma, J.N
    R = {n: J.vals(('R', n)) for n in range(t + 1, t + K + 1)}
    return {n: [sum(g ** j * R[n + j + 1][i] for j in range(t + K - n)) for i in range(N)] for n in range(t, t + K + 1)}


def check_return(J, w, rec):
    T, g, N = w.T, w.gamma, J.N
    for t in range(1, T + 1):
        h, M, A = ('h', t), ('M', t), ('A', t)
        kM = J.keys([M, A])
        for K in range(1, T + 2 - t):
            Qh, Vh, c, cR, cB = bellman_pieces(J, w, t, K)
            Gn = returns(J, w, t, K)
            for n in range(t, t + K):
                for i in range(N):
                    rec.eq('Prop 6.9(b) c_n = c^R_n + c^Obar_n', c[n][i], cR[n][i] + cB[n][i])
            # Proposition 7.7 at each n
            for n in range(t, t + K):
                hn, Mn, An = ('h', n), ('M', n), ('A', n)
                kMn = J.keys([Mn, An])
                gap_n = [x - Qh[n][kMn[i]] for i, x in enumerate(J.cexp(Gn[n], [hn, An])[0])]
                if n + 1 < t + K:
                    h1, M1, A1 = ('h', n + 1), ('M', n + 1), ('A', n + 1)
                    kM1 = J.keys([M1, A1])
                    gap_1 = [x - Qh[n + 1][kM1[i]] for i, x in enumerate(J.cexp(Gn[n + 1], [h1, A1])[0])]
                    inner = J.cexp(gap_1, [hn, An])[0]
                else:
                    inner = [0.0] * N
                for i in range(N):
                    rec.eq('Prop 7.7 one step of the return gap (Bellman value)', gap_n[i], c[n][i] + g * inner[i])
            # Propositions 6.8 (K=2) and 7.8
            lhs = [x - Qh[t][kM[i]] for i, x in enumerate(J.cexp(Gn[t], [h, A])[0])]
            rhs = J.cexp([sum(g ** (n - t) * c[n][i] for n in range(t, t + K)) for i in range(N)], [h, A])[0]
            name = 'Prop 6.8 two-step return gap' if K == 2 else 'Prop 7.8 K-step return gap'
            for i in range(N):
                rec.eq(name, lhs[i], rhs[i])
            # value fitted to the observed return (Remark 6.10, end of Section 7)
            mc = J.cexp(Gn[t], [M, A])[0]
            later = J.cexp([sum(g ** (n - t) * c[n][i] for n in range(t + 1, t + K)) for i in range(N)], [M, A])[0]
            for i in range(N):
                rec.eq('Sec 7 E[G|M_t,a] - hat q(M_t,a) = E[sum_k>=1 g^k c_t+k | M_t,a]', mc[i] - Qh[t][kM[i]], later[i])
            if K == 2:
                M1, A1, O1, O2 = ('M', t + 1), ('A', t + 1), ('O', t + 1), ('O', t + 2)
                R2 = J.vals(('R', t + 2))
                e = [x - y for x, y in zip(J.cexp(R2, [M, A, O1, A1])[0], J.cexp(R2, [M1, A1])[0])]
                ee = J.cexp(e, [M, A])[0]
                for i in range(N):
                    rec.eq('Rem 6.10 E[G|M_t,a] - hat q(M_t,a) = g E[update difference]', mc[i] - Qh[t][kM[i]], g * ee[i])
                GAPS['|E[G|M_t,a] - hat q(M_t,a)| (two steps)'] = max(GAPS['|E[G|M_t,a] - hat q(M_t,a)| (two steps)'],
                                                                      max(abs(mc[i] - Qh[t][kM[i]]) for i in range(N)))


def kl(P, Q):
    return sum(p * math.log(p / Q[x]) for x, p in P.items() if p > 0)


def check_info_value(J, w, rec):
    T, g, N = w.T, w.gamma, J.N
    dR = 1.0
    for t in range(1, T + 1):
        h, M, A, O1 = ('h', t), ('M', t), ('A', t), ('O', t + 1)
        kh, kM = J.keys([h, A]), J.keys([M, A])
        Ph, PM = J.cdist([O1], [h, A]), J.cdist([O1], [M, A])
        Dt = J.I([h], [O1], [M, A])
        KLs = [kl(Ph[kh[i]], PM[kM[i]]) for i in range(N)]
        rec.eq('Sec 8 D_t = E[KL(P(O|h,A)||P(O|M,A))]', Dt, J.E(KLs))
        for K in range(1, T + 2 - t):
            Qh, Vh, c, cR, cB = bellman_pieces(J, w, t, K)
            ell = {n: dR * sum(g ** j for j in range(t + K - n)) for n in range(t, t + K + 1)}
            # Lemma 8.1 and Corollary 8.2 on the pair behind c_t, with the weight of c_t
            V1 = Vh[t + 1]
            seen = set()
            for i in range(N):
                if kh[i] in seen or len(seen) >= 12:
                    continue
                seen.add(kh[i])
                m, a = kM[i]
                P, Q = Ph[kh[i]], PM[kM[i]]
                wf = {x: x[0][0] + g * V1[w.f[(m, a, x[0])]] for x in Q}
                diff = sum(P.get(x, 0.0) * wf[x] for x in Q) - sum(Q[x] * wf[x] for x in Q)
                rec.eq('Lem 8.1 / c_t is that difference of expectations', diff, c[t][i])
                EQ = sum(Q[x] * wf[x] for x in Q)
                K_ = kl(P, Q)
                for lam in (-2.0, -0.5, 0.7, 3.0):
                    Z = sum(Q[x] * math.exp(lam * wf[x]) for x in Q)
                    Ql = {x: Q[x] * math.exp(lam * wf[x]) / Z for x in Q}
                    rhs = K_ + math.log(sum(Q[x] * math.exp(lam * (wf[x] - EQ)) for x in Q)) - kl(P, Ql)
                    rec.eq('Lem 8.1 exact identity', lam * diff, rhs)
                rec.le('Cor 8.2 |E_P w - E_Q w| <= l sqrt(KL/2)', abs(diff), ell[t] * math.sqrt(K_ / 2))
            # Proposition 8.4(a): every piece
            bound = 0.0
            for n in range(t, t + K):
                hn, Mn, An, On1, Rn1, Bn1 = ('h', n), ('M', n), ('A', n), ('O', n + 1), ('R', n + 1), ('Ob', n + 1)
                Dn = J.I([hn], [On1], [Mn, An])
                rew, rst = J.I([hn], [Rn1], [Mn, An]), J.I([hn], [Bn1], [Mn, An, Rn1])
                rec.le('Prop 8.4(a) E|c_n| <= l_n sqrt(D_n/2)', J.E([abs(x) for x in c[n]]), ell[n] * math.sqrt(Dn / 2))
                rec.le('Prop 8.4(a) E|c^R_n| <= l_n sqrt(reward part/2)', J.E([abs(x) for x in cR[n]]), ell[n] * math.sqrt(rew / 2))
                rec.le('Prop 8.4(a) E|c^Obar_n| <= g l_n+1 sqrt(rest part/2)', J.E([abs(x) for x in cB[n]]), g * ell[n + 1] * math.sqrt(rst / 2))
                bound += g ** (n - t) * ell[n] * math.sqrt(Dn / 2)
            Gn = returns(J, w, t, K)
            gap = [x - Qh[t][kM[i]] for i, x in enumerate(J.cexp(Gn[t], [h, A])[0])]
            Egap = J.E([abs(x) for x in gap])
            rec.le('Prop 8.4(b) E|q(h,A) - hat q(M,A)| <= sum g^k l sqrt(D_t+k/2)', Egap, bound)
            if K == 2:
                h1, M1, A1, O2 = ('h', t + 1), ('M', t + 1), ('A', t + 1), ('O', t + 2)
                D1 = J.I([h1], [O2], [M1, A1])
                rp1 = J.I([h1], [('R', t + 2)], [M1, A1])
                rew0, rst0 = J.I([h], [('R', t + 1)], [M, A]), J.I([h], [('Ob', t + 1)], [M, A, ('R', t + 1)])
                rec.le('Prop 8.3 E|c_t| <= (1+g) dR sqrt(D_t/2)', J.E([abs(x) for x in c[t]]), (1 + g) * dR * math.sqrt(Dt / 2))
                rec.le('Prop 8.3 E|c^R_t| <= (1+g) dR sqrt(reward part/2)', J.E([abs(x) for x in cR[t]]), (1 + g) * dR * math.sqrt(rew0 / 2))
                rec.le('Prop 8.3 E|c^Obar_t| <= g dR sqrt(rest part/2)', J.E([abs(x) for x in cB[t]]), g * dR * math.sqrt(rst0 / 2))
                rec.le('Prop 8.3 E|c_t+1| <= dR sqrt(reward part of D_t+1 /2)', J.E([abs(x) for x in c[t + 1]]), dR * math.sqrt(rp1 / 2))
                rec.le('Prop 8.3 reward part of D_t+1 <= D_t+1', rp1, D1)
                rec.le('Prop 8.3 two-step gap bound', Egap, (1 + g) * dR * math.sqrt(Dt / 2) + g * dR * math.sqrt(D1 / 2))
            # Remark 8.5
            key = ('GK', t * 100 + K)
            J._vals[key] = Gn[t]
            preds = []
            for k in range(1, K + 1):
                preds.append(('O', t + k))
                if k < K:
                    preds.append(('A', t + k))
            DK = J.I([h], preds, [M, A])
            IG = J.I([h], [key], [M, A])
            rec.eq('Rem 8.5 I(h;G_t:t+K|M,A) = D_t:t+K - I(h;stream|M,A,G)', IG, DK - J.I([h], preds, [M, A, key]))
            mcgap = [x - y for x, y in zip(J.cexp(Gn[t], [h, A])[0], J.cexp(Gn[t], [M, A])[0])]
            rec.le('Rem 8.5 E|q(h,A) - E[G|M,A]| <= l_t sqrt(I(h;G|M,A)/2)', J.E([abs(x) for x in mcgap]), ell[t] * math.sqrt(IG / 2))
            rec.le('Rem 8.5 I(h;G|M,A) <= D_t:t+K', IG, DK)


def main():
    rec = Rec()
    for seed, T in ((1, 4), (2, 4), (3, 3), (4, 3), (5, 3)):
        w = World(seed, T, 0.9)
        J = world_joint(w)
        check_one_step(J, w, rec)
        check_steps(J, w, rec)
        check_return(J, w, rec)
        check_info_value(J, w, rec)
        print(f"seed {seed} (T={T}, {J.N} trajectories) done", flush=True)
    ok = rec.report()
    print("largest value over all worlds and times (shows the three orders differ):")
    for k, v in GAPS.items():
        print(f"  {k}: {v:.4f}")
    print("ALL PASS" if ok else "SOME CHECKS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
