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


def bellman_pieces(J, w, t, K):
    """The memory's Bellman value with final time t+K (Definition 6.2) and, for each n,
    the comparisons c_n, c^R_n, c^Obar_n over all trajectories (Propositions 7.1, 7.3)."""
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
    """G_n for t <= n <= t+K with final time t+K."""
    g, N = w.gamma, J.N
    R = {n: J.vals(('R', n)) for n in range(t + 1, t + K + 1)}
    return {n: [sum(g ** j * R[n + j + 1][i] for j in range(t + K - n)) for i in range(N)] for n in range(t, t + K + 1)}


def kl(P, Q):
    return sum(p * math.log(p / Q[x]) for x, p in P.items() if p > 0)


def ratio(J, X, hside, mside):
    """E[log P(X|hside)/P(X|mside)] computed from the two conditional laws."""
    Ph, Pm = J.cdist(X, hside), J.cdist(X, mside)
    kh, km, kx = J.keys(hside), J.keys(mside), J.keys(X)
    return J.E([math.log(Ph[kh[i]][kx[i]] / Pm[km[i]][kx[i]]) for i in range(J.N)])


def check_section6(J, w, rec):
    T, g, N = w.T, w.gamma, J.N
    for n in range(1, T + 1):
        h, M, A, O1, A1, R1, B1 = ('h', n), ('M', n), ('A', n), ('O', n + 1), ('A', n + 1), ('R', n + 1), ('Ob', n + 1)
        # Lemma 6.1 and Corollary 6.2
        cases = [('D_n', [O1], [A]), ('D0_n', [O1], []), ('D+_n', [O1], [A, A1]), ('dE_n', [A], [O1]),
                 ('dP_n', [A1], [A]), ('reward part of D_n', [R1], [A]), ('rest part of D_n', [B1], [A, R1]),
                 ('reward part of dE_n', [A], [R1]), ('rest part of dP_n', [A1], [A, R1])]
        for name, X, Zs in cases:
            rec.eq('Lem 6.1 / Cor 6.2 %s = E[log ratio]' % name, J.I([h], X, [M] + Zs), ratio(J, X, [h] + Zs, [M] + Zs))
        rec.eq('Cor 6.2 dE_n = I(h;A|M,O) (Prop 3.2 form)', J.I([h], [A], [M, O1]),
               (J.I([A], [O1], [h])) - J.I([A], [O1], [M]))
        # memory prediction = average of history predictions (Lemma 6.1), X = O_{n+1}, Z = A_n
        PM, Ph, PhM = J.cdist([O1], [M, A]), J.cdist([O1], [h, A]), J.cdist([h], [M, A])
        for (m, a), d in PM.items():
            for (o,), p in d.items():
                avg = sum(ph * Ph[(hv, a)].get((o,), 0.0) for (hv,), ph in PhM[(m, a)].items())
                rec.eq('Lem 6.1 memory prediction = average of history predictions', p, avg)
        # Proposition 6.3
        kh, kM = J.keys([h, A]), J.keys([M, A])
        PRh, PRM = J.cdist([R1], [h, A]), J.cdist([R1], [M, A])
        Rv = J.vals(R1)
        dl = [x - y for x, y in zip(J.cexp(Rv, [h, A])[0], J.cexp(Rv, [M, A])[0])]
        for i in range(N):
            rec.eq('Prop 6.3 reward difference = sum rho [P(rho|h,a) - P(rho|M,a)]', dl[i],
                   sum(r * (PRh[kh[i]].get((r,), 0.0) - p) for (r,), p in PRM[kM[i]].items()))
    for t in range(1, T + 1):
        h, M, A = ('h', t), ('M', t), ('A', t)
        kh, kM = J.keys([h, A]), J.keys([M, A])
        for K in range(1, T + 2 - t):
            Gn = returns(J, w, t, K)
            R = {n: J.vals(('R', n)) for n in range(t + 1, t + K + 1)}
            q = J.cexp(Gn[t], [h, A])[0]
            # Proposition 6.4
            eh = {n: J.cexp(R[n + 1], [('h', n), ('A', n)])[0] for n in range(t, t + K)}
            em = {n: J.cexp(R[n + 1], [('M', n), ('A', n)])[0] for n in range(t, t + K)}
            first = J.cexp([sum(g ** (n - t) * eh[n][i] for n in range(t, t + K)) for i in range(N)], [h, A])[0]
            mem = J.cexp([sum(g ** (n - t) * em[n][i] for n in range(t, t + K)) for i in range(N)], [h, A])[0]
            dsum = J.cexp([sum(g ** (n - t) * (eh[n][i] - em[n][i]) for n in range(t, t + K)) for i in range(N)], [h, A])[0]
            for i in range(N):
                rec.eq('Prop 6.4 first line', q[i], first[i])
                rec.eq('Prop 6.4 second line (T = t+%d)' % K if K <= 3 else 'Prop 6.4 second line (T > t+3)', q[i] - mem[i], dsum[i])
            # Proposition 6.5
            if K > 1:
                Ph = J.cdist([('O', t + 1)], [h, A])
                _, v1 = J.cexp(Gn[t + 1], [('h', t + 1)])
                _, q1 = J.cexp(Gn[t + 1], [('h', t + 1), ('A', t + 1)])
            for i in range(N):
                hv, a = kh[i]
                m = kM[i][0]
                if K == 1:
                    rec.eq('Prop 6.5 Bellman equation for the history', q[i], eh[t][i])
                    continue
                rhs = 0.0
                for (o,), p in Ph[kh[i]].items():
                    h1 = (hv[0] + (o,), hv[1] + (a,))
                    m1 = w.f[(m, a, o)]
                    rhs += p * (o[0] + g * v1[(h1,)])
                    rec.eq('Prop 6.5 second line v = sum pi q', v1[(h1,)], sum(w.pi[m1][b] * q1[(h1, b)] for b in ACTS))
                rec.eq('Prop 6.5 Bellman equation for the history', q[i], rhs)
            # Proposition 6.7
            Qh, Vh, c, cR, cB = bellman_pieces(J, w, t, K)
            for n in range(t, t + K):
                for i in range(N):
                    rec.eq('Prop 6.7 step 2: first sum = c^R_n + c^Obar_n', c[n][i], cR[n][i] + cB[n][i])
            rec.eq('Prop 6.7 c^Obar = 0 at the last step', max(abs(x) for x in cB[t + K - 1]), 0.0)
            lhs = [q[i] - Qh[t][kM[i]] for i in range(N)]
            rhs = J.cexp([sum(g ** (n - t) * (cR[n][i] + cB[n][i]) for n in range(t, t + K)) for i in range(N)], [h, A])[0]
            for i in range(N):
                rec.eq('Prop 6.7 gap = E[sum g^k (c^R + c^Obar)] (T = t+%d)' % K if K <= 3 else 'Prop 6.7 gap (T > t+3)', lhs[i], rhs[i])


def main():
    rec = Rec()
    for seed, T in ((1, 4), (2, 4), (3, 3), (4, 3), (5, 3)):
        w = World(seed, T, 0.9)
        J = world_joint(w)
        check_one_step(J, w, rec)
        check_steps(J, w, rec)
        check_section6(J, w, rec)
        print(f"seed {seed} (T={T}, {J.N} trajectories) done", flush=True)
    ok = rec.report()
    print("ALL PASS" if ok else "SOME CHECKS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
