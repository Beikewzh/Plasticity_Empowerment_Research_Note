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
            rec.eq('Prop 6.2 D_t:t+2 = D_t + I(h_t;O_t+2|M_t,A_t,O_t+1,A_t+1)', D2, a['D'] + mid)
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


def check_return(J, w, rec):
    T, g = w.T, w.gamma
    N = J.N
    R = {n: J.vals(('R', n)) for n in range(1, T + 2)}
    for t in range(1, T + 1):
        h, M, A = ('h', t), ('M', t), ('A', t)
        kh, kM = J.keys([h, A]), J.keys([M, A])
        # two steps (needs R_{t+2})
        if t + 1 <= T:
            h1, M1, A1, O1, O2 = ('h', t + 1), ('M', t + 1), ('A', t + 1), ('O', t + 1), ('O', t + 2)
            G2 = [R[t + 1][i] + g * R[t + 2][i] for i in range(N)]
            lhs = [x - y for x, y in zip(J.cexp(G2, [h, A])[0], J.cexp(G2, [M, A])[0])]
            Ph, PM = J.cdist([O1], [h, A]), J.cdist([O1], [M, A])
            _, ER2 = J.cexp(R[t + 2], [M, A, O1])
            ct, cR, cB = [], [], []
            PRh, PRM = J.cdist([('R', t + 1)], [h, A]), J.cdist([('R', t + 1)], [M, A])
            PBh, PBM = J.cdist([('Ob', t + 1)], [h, A, ('R', t + 1)]), J.cdist([('Ob', t + 1)], [M, A, ('R', t + 1)])
            _, ER2r = J.cexp(R[t + 2], [M, A, ('R', t + 1)])
            _, ER2rb = J.cexp(R[t + 2], [M, A, ('R', t + 1), ('Ob', t + 1)])
            for i in range(N):
                ph, pm = Ph[kh[i]], PM[kM[i]]
                ct.append(sum((ph.get((o,), 0.0) - pmo) * (o[0] + g * ER2[kM[i] + (o,)]) for (o,), pmo in pm.items()))
                cR.append(sum((PRh[kh[i]].get((r,), 0.0) - pmr) * (r + g * ER2r[kM[i] + (r,)]) for (r,), pmr in PRM[kM[i]].items()))
                tot = 0.0
                for (r,), phr in PRh[kh[i]].items():
                    pbm = PBM[kM[i] + (r,)]
                    pbh = PBh[kh[i] + (r,)]
                    tot += phr * g * sum((pbh.get((b,), 0.0) - pmb) * ER2rb[kM[i] + (r, b)] for (b,), pmb in pbm.items())
                cB.append(tot)
            for i in range(N):
                rec.eq('Prop 6.8(b) c_t = c^R_t + c^Obar_t', ct[i], cR[i] + cB[i])
            kh1, kM1 = J.keys([h1, A1]), J.keys([M1, A1])
            Ph1, PM1 = J.cdist([O2], [h1, A1]), J.cdist([O2], [M1, A1])
            c1 = [sum((Ph1[kh1[i]].get((o,), 0.0) - pmo) * o[0] for (o,), pmo in PM1[kM1[i]].items()) for i in range(N)]
            q1 = [x - y for x, y in zip(J.cexp(R[t + 2], [h1, A1])[0], J.cexp(R[t + 2], [M1, A1])[0])]
            for i in range(N):
                rec.eq('Prop 6.7 c_t+1 = one-step gap at t+1', c1[i], q1[i])
            u = [x - y for x, y in zip(J.cexp(R[t + 2], [M, A, O1])[0], J.cexp(R[t + 2], [M1])[0])]
            inner = J.cexp([a - b for a, b in zip(c1, u)], [h, A])[0]
            for i in range(N):
                rec.eq('Prop 6.7 two-step return gap', lhs[i], ct[i] + g * inner[i])
            rec.le('Prop 6.7 size of the update term (shown)', -max(abs(x) for x in u), 0.0)
        # K steps
        for K in range(1, T + 2 - t):
            Gn = {}
            for n in range(t, t + K + 1):
                Gn[n] = [sum(g ** j * R[n + j + 1][i] for j in range(t + K - n)) for i in range(N)]
            v = {n: J.cexp(Gn[n], [('h', n)])[0] for n in range(t, t + K + 1)}
            c, u = {}, {}
            for n in range(t, t + K):
                hn, Mn, An, On1 = ('h', n), ('M', n), ('A', n), ('O', n + 1)
                khn, kMn = J.keys([hn, An]), J.keys([Mn, An])
                Phn, PMn = J.cdist([On1], [hn, An]), J.cdist([On1], [Mn, An])
                _, Ev = J.cexp(v[n + 1], [Mn, An, On1])
                c[n] = [sum((Phn[khn[i]].get((o,), 0.0) - pmo) * (o[0] + g * Ev[kMn[i] + (o,)]) for (o,), pmo in PMn[kMn[i]].items())
                        for i in range(N)]
                if n > t:
                    Un = [('M', n - 1), ('A', n - 1), ('O', n)]
                    u[n] = [x - y for x, y in zip(J.cexp(v[n], Un)[0], J.cexp(v[n], [Mn])[0])]
            # one-step recursion (Prop 7.6) at each n
            for n in range(t, t + K):
                hn, Mn, An = ('h', n), ('M', n), ('A', n)
                gap_n = [x - y for x, y in zip(J.cexp(Gn[n], [hn, An])[0], J.cexp(Gn[n], [Mn, An])[0])]
                if n + 1 < t + K:
                    h1, M1, A1 = ('h', n + 1), ('M', n + 1), ('A', n + 1)
                    gap_1 = [x - y for x, y in zip(J.cexp(Gn[n + 1], [h1, A1])[0], J.cexp(Gn[n + 1], [M1, A1])[0])]
                    inner = J.cexp([a - b for a, b in zip(gap_1, u[n + 1])], [hn, An])[0]
                else:
                    inner = [0.0] * N
                for i in range(N):
                    rec.eq('Prop 7.6 one step of the return gap', gap_n[i], c[n][i] + g * inner[i])
            # full K-step formula (Prop 7.7)
            tot = [sum(g ** (n - t) * c[n][i] for n in range(t, t + K)) - sum(g ** (n - t) * u[n][i] for n in range(t + 1, t + K))
                   for i in range(N)]
            rhs = J.cexp(tot, [h, A])[0]
            lhs = [x - y for x, y in zip(J.cexp(Gn[t], [h, A])[0], J.cexp(Gn[t], [M, A])[0])]
            for i in range(N):
                rec.eq('Prop 7.7 K-step return gap', lhs[i], rhs[i])


def main():
    rec = Rec()
    for seed, T in ((1, 4), (2, 4), (3, 3), (4, 3), (5, 3)):
        w = World(seed, T, 0.9)
        J = world_joint(w)
        check_one_step(J, w, rec)
        check_steps(J, w, rec)
        check_return(J, w, rec)
        print(f"seed {seed} (T={T}, {J.N} trajectories) done", flush=True)
    ok = rec.report()
    print("ALL PASS" if ok else "SOME CHECKS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
