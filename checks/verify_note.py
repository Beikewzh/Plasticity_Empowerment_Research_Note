#!/usr/bin/env python3
"""Exact numerical checks of the identities in the note
"The State Defect, Plasticity, Empowerment and the Return".

A small random world is enumerated exactly:
  observations O = (R, Obar) in {0,1}^2 (R is the reward, Delta R = 1),
  actions A in {0,1}, a memory with two values updated by a random map
  M_{t+1} = f(M_t, A_t, O_{t+1}) (M_1 = f0(O_1)), a random stochastic policy
  pi(a|m), and an environment whose law of the next observation is drawn at
  random for every (history, action) pair, so it depends on the whole history.
Every quantity is computed from the exact law of the trajectory
(O_1, A_1, ..., O_{T+1}, A_{T+1}), so each identity must hold to floating-point
precision. Logarithms are natural. Pure Python, no dependencies.

Each check is labelled by the statement of the note it tests (e.g. "Prop 5.1").
Usage: python3 verify_note.py            (all checks)
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
def check_four_terms(J, w, rec):
    T = w.T
    for t in range(1, T + 1):
        ht, Mt, At, O1, A1, M1 = ('h', t), ('M', t), ('A', t), ('O', t + 1), ('A', t + 1), ('M', t + 1)
        D = J.I([ht], [O1], [Mt, At])
        D0 = J.I([ht], [O1], [Mt])
        dE = J.I([ht], [At], [Mt, O1])
        dP = J.I([ht], [A1], [Mt, At])
        Dp = J.I([ht], [O1], [Mt, At, A1])
        rec.eq('Lem 2.1: I(h_t;A_t|M_t)=0', J.I([ht], [At], [Mt]), 0.0)
        rec.eq('Lem 2.1: I(h_t;A_t+1|M_t,A_t,O_t+1)=0', J.I([ht], [A1], [Mt, At, O1]), 0.0)
        rec.eq('Prop 3.2: D=D0+dE', D, D0 + dE)
        rec.eq('Prop 3.2: D=dP+D+', D, dP + Dp)
        rec.eq('Prop 3.2: time order gives 0+D+0', J.I([ht], [At, O1, A1], [Mt]), D)
        E, EM = J.I([At], [O1], [ht]), J.I([At], [O1], [Mt])
        P, PM = J.I([O1], [A1], [ht, At]), J.I([O1], [A1], [Mt, At])
        rec.eq('Prop 3.4a: dE=E-E(M)', dE, E - EM)
        rec.eq('Prop 3.4a: dP=P(M)-P', dP, PM - P)
        rec.eq('Prop 3.4b: H(A|M)=E(M)+dE+H(A|h,O\')', J.Hc([At], [Mt]), EM + dE + J.Hc([At], [ht, O1]))
        rec.eq('Prop 3.4c: I(M\';A\')=I(M,A;A\')+P+dP', J.I([M1], [A1]), J.I([Mt, At], [A1]) + P + dP)
        rec.eq('Prop 3.4c: I(M,A;A\')+dP=I(h,A;A\')', J.I([Mt, At], [A1]) + dP, J.I([ht, At], [A1]))
        rec.eq('Lem 6.1a: I(M,A;O\')=I(h,A;O\')-D', J.I([Mt, At], [O1]), J.I([ht, At], [O1]) - D)
        rec.le('Prop 3.2: four terms >= 0', -min(D0, dE, dP, Dp), 0.0)

        # reward inside the terms
        R1, B1 = ('R', t + 1), ('Ob', t + 1)
        for X, Yv in ((R1, B1), (B1, R1)):
            rec.eq('Prop 7.1: dE formula', dE, (J.I([At], [X], [ht]) - J.I([At], [X], [Mt]))
                   + (J.I([At], [Yv], [ht, X]) - J.I([At], [Yv], [Mt, X])))
            rec.eq('Prop 7.1: dE first bracket', J.I([At], [X], [ht]) - J.I([At], [X], [Mt]), J.I([ht], [At], [Mt, X]))
            rec.eq('Prop 7.1: dP formula', dP, (J.I([X], [A1], [Mt, At]) - J.I([X], [A1], [ht, At]))
                   + (J.I([Yv], [A1], [Mt, At, X]) - J.I([Yv], [A1], [ht, At, X])))
            rec.eq('Prop 7.1: dP second bracket', J.I([Yv], [A1], [Mt, At, X]) - J.I([Yv], [A1], [ht, At, X]),
                   J.I([ht], [A1], [Mt, At, X]))
            Xl = J.I([ht], [X], [Mt, At])
            rec.eq('Prop 7.1: X line, first split', Xl, J.I([ht], [X], [Mt]) + J.I([ht], [At], [Mt, X]))
            rec.eq('Prop 7.1: X line, second split', Xl, (J.I([X], [A1], [Mt, At]) - J.I([X], [A1], [ht, At]))
                   + J.I([ht], [X], [Mt, At, A1]))
            Yl = J.I([ht], [Yv], [Mt, At, X])
            rec.eq('Prop 7.1: Y line, first split', Yl, J.I([ht], [Yv], [Mt, X])
                   + (J.I([At], [Yv], [ht, X]) - J.I([At], [Yv], [Mt, X])))
            rec.eq('Prop 7.1: Y line, second split', Yl, J.I([ht], [A1], [Mt, At, X]) + J.I([ht], [Yv], [Mt, At, A1, X]))
        rec.eq('Prop 7.2: D',
               J.I([ht], [R1], [Mt, At]) - J.I([ht], [R1], [Mt, At, B1]), J.CI([ht], [R1], [B1], [Mt, At]))
        rec.eq('Prop 7.2: D0',
               J.I([ht], [R1], [Mt]) - J.I([ht], [R1], [Mt, B1]), J.CI([ht], [R1], [B1], [Mt]))
        rec.eq('Prop 7.2: D+',
               J.I([ht], [R1], [Mt, At, A1]) - J.I([ht], [R1], [Mt, At, A1, B1]), J.CI([ht], [R1], [B1], [Mt, At, A1]))
        rE_rf = J.I([At], [R1], [ht]) - J.I([At], [R1], [Mt])
        rE_sf = J.I([At], [R1], [ht, B1]) - J.I([At], [R1], [Mt, B1])
        rec.eq('Prop 7.2: dE', rE_rf - rE_sf,
               J.CI([At], [R1], [B1], [ht]) - J.CI([At], [R1], [B1], [Mt]))
        rP_rf = J.I([R1], [A1], [Mt, At]) - J.I([R1], [A1], [ht, At])
        rP_sf = J.I([R1], [A1], [Mt, At, B1]) - J.I([R1], [A1], [ht, At, B1])
        rec.eq('Prop 7.2: dP', rP_rf - rP_sf,
               J.CI([R1], [A1], [B1], [Mt, At]) - J.CI([R1], [A1], [B1], [ht, At]))
        lo = max(0.0, J.I([ht], [R1], [Mt, At]) - J.I([ht], [B1], [Mt, At]))
        hi = min(J.I([ht], [R1], [Mt, At]), J.I([ht], [R1], [Mt, At, B1]))
        rec.le('Prop 7.5: bounds on U^R consistent', lo, hi)
        # reward error bound (Pinsker)
        lhs = J.E([abs(a - b) for a, b in zip(J.cexp(J.vals(R1), [ht, At])[0], J.cexp(J.vals(R1), [Mt, At])[0])])
        rec.le('extra: Pinsker bound for the next reward', lhs, math.sqrt(0.5 * J.I([ht], [R1], [Mt, At])))


def check_updates(J, w, rec):
    T, g = w.T, w.gamma
    for n in range(1, T + 1):
        hn, Mn, An, On1, An1 = ('h', n), ('M', n), ('A', n), ('O', n + 1), ('A', n + 1)
        combos = [([On1], []), ([On1], [An]), ([An1], [An]), ([On1], [An, An1]),
                  ([An], [On1]), (['G'], [An]), (['G'], [])]
        for X, V in combos:
            X = [('G', n)] if X == ['G'] else X
            lhs = J.I([hn], X, [Mn] + V)
            rhs = sum(J.I([('U', s)], X, Y(s, n - s) + V) for s in range(1, n + 1))
            rec.eq('Lem 2.2: sum over updates', lhs, rhs)
        # two-piece window lemma
        for t in range(1, n + 1):
            j = n - t
            lhs = J.I([hn], [On1], [Mn, An])
            rhs = J.I(Y(t, j), [On1], [Mn, An]) + J.I([('h', t)], [On1], Y(t, j) + [An])
            rec.eq('Lem 2.2: two groups of updates', lhs, rhs)
        # per-update charge and its two splits
        for s in range(1, n + 1):
            U, Zs, Ys = [('U', s)], Z(s, n - s), Y(s, n - s)
            T_sn = J.I(U, [On1], Zs)
            first = J.I(U, [On1], Ys) + J.I(U, [An], Ys + [On1])
            second = J.I(U, [An1], Zs) + J.I(U, [On1], Zs + [An1])
            rec.eq('Prop 4.2: first split', T_sn, first)
            rec.eq('Prop 4.2: second split', T_sn, second)
            rec.eq('Lem 2.1: I(U_s;A_n|Y_s,n-s)=0', J.I(U, [An], Ys), 0.0)
        # parts sum to the four terms
        parts = [sum(J.I([('U', s)], [On1], Y(s, n - s)) for s in range(1, n + 1)),
                 sum(J.I([('U', s)], [An], Y(s, n - s) + [On1]) for s in range(1, n + 1)),
                 sum(J.I([('U', s)], [An1], Z(s, n - s)) for s in range(1, n + 1)),
                 sum(J.I([('U', s)], [On1], Z(s, n - s) + [An1]) for s in range(1, n + 1))]
        terms = [J.I([hn], [On1], [Mn]), J.I([hn], [An], [Mn, On1]), J.I([hn], [An1], [Mn, An]),
                 J.I([hn], [On1], [Mn, An, An1])]
        for p_, q_ in zip(parts, terms):
            rec.eq('Prop 4.2: sums over s give D0,dE,dP,D+', p_, q_)
    # undiscounted charge = info discarded about the future stream
    for s in range(1, T + 1):
        for K in range(0, T - s + 1):
            lhs = sum(J.I([('U', s)], [('O', s + k + 1)], Z(s, k)) for k in range(K + 1))
            seq = []
            for k in range(K + 1):
                seq += [('A', s + k), ('O', s + k + 1)]
            rec.eq('Prop 4.1c: sum_k = I(U_s; future | M_s)', lhs, J.I([('U', s)], seq, [('M', s)]))
    # discounted sum of defects = discounted sum of charges
    lhs = sum(g ** (n - 1) * J.I([('h', n)], [('O', n + 1)], [('M', n), ('A', n)]) for n in range(1, T + 1))
    rhs = sum(g ** (s - 1) * sum(g ** k * J.I([('U', s)], [('O', s + k + 1)], Z(s, k)) for k in range(T - s + 1))
              for s in range(1, T + 1))
    rec.eq('Prop 4.1a,b: sum g^n D_n = sum g^s C_s', lhs, rhs)


def values(J, w):
    T = w.T
    v = {n: J.cexp(J.vals(('G', n)), [('h', n)])[0] for n in range(1, T + 1)}
    v[T + 1] = [0.0] * J.N
    return v


def check_value(J, w, rec):
    T, g = w.T, w.gamma
    v = values(J, w)
    for t in range(1, T + 1):
        ht, Mt, At = ('h', t), ('M', t), ('A', t)
        Gt = J.vals(('G', t))
        qh = J.cexp(Gt, [ht, At])[0]
        qm = J.cexp(Gt, [Mt, At])[0]
        S = [0.0] * J.N
        for k in range(0, T - t + 1):
            Zs, Ys, On, An, An1 = Z(t, k), Y(t, k), ('O', t + k + 1), ('A', t + k), ('A', t + k + 1)
            Ph, Pz = J.cdist([On], [ht] + Zs), J.cdist([On], Zs)
            _, vd = J.cexp(v[t + k + 1], Zs + [On])
            kz, khz = J.keys(Zs), J.keys([ht] + Zs)
            wfun = lambda zk, o: o[0] + g * vd.get(zk + (o,), 0.0)
            c = []
            DL = []
            for i in range(J.N):
                ph, pz = Ph[khz[i]], Pz[kz[i]]
                c.append(sum((ph.get((o,), 0.0) - pzo) * wfun(kz[i], o) for (o,), pzo in pz.items()))
                DL.append(sum(pho * math.log(pho / pz[(o,)]) for (o,), pho in ph.items() if pho > 0))
            Dtk = J.I([ht], [On], Zs)
            rec.eq('Prop 5.1: D_t,k = E[E_Z L log L]', Dtk, J.E(DL))
            rec.eq('Prop 3.7a: D_t,k = D0+dE', Dtk, J.I([ht], [On], Ys) + J.I([ht], [An], Ys + [On]))
            if t + k + 1 <= T + 1:
                rec.eq('Prop 3.7a: D_t,k = dP+D+', Dtk, J.I([ht], [An1], Zs) + J.I([ht], [On], Zs + [An1]))
            rec.le('Prop 3.7b: D_t,k <= D_t+k', Dtk,
                   J.I([('h', t + k)], [On], [('M', t + k), An]))
            # routes, second split: c = c^dP + c^D+
            PAh, PAz = J.cdist([An1], [ht] + Zs), J.cdist([An1], Zs)
            POha, POza = J.cdist([On], [ht] + Zs + [An1]), J.cdist([On], Zs + [An1])
            for i in range(J.N):
                zk, hzk = kz[i], khz[i]
                cdp = 0.0
                cdplus = 0.0
                for (a1,), pza in PAz[zk].items():
                    Wa = sum(po * wfun(zk, o) for (o,), po in POza[zk + (a1,)].items())
                    cdp += (PAh[hzk].get((a1,), 0.0) - pza) * Wa
                for (a1,), pha in PAh[hzk].items():
                    if pha <= 0:
                        continue
                    pz_o = POza[zk + (a1,)]
                    ph_o = POha[hzk + (a1,)]
                    cdplus += pha * sum((ph_o.get((o,), 0.0) - pzo) * wfun(zk, o) for (o,), pzo in pz_o.items())
                rec.eq('Prop 5.2a: c = c^dP + c^D+', c[i], cdp + cdplus)
            # routes, first split (cost averaged over A_{t+k}): c = c^D0 + c^dE, and dE route has no reward
            cbar = J.cexp(c, [ht] + Ys)[0]
            ky, khy = J.keys(Ys), J.keys([ht] + Ys)
            POhy, POy = J.cdist([On], [ht] + Ys), J.cdist([On], Ys)
            PAyo, PAhyo = J.cdist([An], Ys + [On]), J.cdist([An], [ht] + Ys + [On])
            for i in range(J.N):
                yk, hyk = ky[i], khy[i]
                cd0 = 0.0
                cde = 0.0
                for (o,), pyo in POy[yk].items():
                    wbar = sum(pa * (o[0] + g * vd.get(yk + (a, o), 0.0)) for (a,), pa in PAyo[yk + (o,)].items())
                    cd0 += (POhy[hyk].get((o,), 0.0) - pyo) * wbar
                for (o,), pho in POhy[hyk].items():
                    if pho <= 0:
                        continue
                    pa_h, pa_y = PAhyo[hyk + (o,)], PAyo[yk + (o,)]
                    cde += pho * g * sum((pa_h.get((a,), 0.0) - pay) * vd.get(yk + (a, o), 0.0)
                                         for (a,), pay in pa_y.items())
                rec.eq('Prop 5.2b: cbar = c^D0 + c^dE (no reward in dE)', cbar[i], cd0 + cde)
            S = [s_ + g ** k * ci for s_, ci in zip(S, c)]
        ES = J.cexp(S, [ht, At])[0]
        for i in range(J.N):
            rec.eq('Prop 5.1: q(h,a)-q(M,a) = E[sum g^k c]', qh[i] - qm[i], ES[i])
        vh = J.cexp(Gt, [ht])[0]
        vm = J.cexp(Gt, [Mt])[0]
        ESv = J.cexp(S, [ht])[0]
        for i in range(J.N):
            rec.eq('Prop 5.1: v(h)-v(M) = E[sum g^k c]', vh[i] - vm[i], ESv[i])
        # bounds
        gap_abs = J.E([abs(a - b) for a, b in zip(qh, qm)])
        rec.le('Cor 5.6: first bound', gap_abs,
               (1.0 / (1 - g)) * math.sqrt(0.5 * J.I([ht], [('G', t)], [Mt, At])))
        rec.le('Cor 5.6: second bound', gap_abs,
               (1.0 / (1 - g)) * sum(g ** k * math.sqrt(0.5 * J.I([ht], [('O', t + k + 1)], Z(t, k)))
                                     for k in range(T - t + 1)))
        # information routes to the return (relevant parts)
        G = [('G', t)]
        lhs = J.I([ht], G, [Mt, At])
        rhs = 0.0
        for j in range(0, T - t + 1):
            rhs += J.CI([ht], [('A', t + j + 1)], G, Z(t, j)) + J.CI([ht], [('O', t + j + 1)], G, Z(t, j) + [('A', t + j + 1)])
        rhs += J.I([ht], G, Z(t, T - t) + [('A', T + 1), ('O', T + 1)])
        rec.eq('Prop 5.5: first line', lhs, rhs)
        lhs = J.I([ht], G, [Mt])
        rhs = 0.0
        for j in range(0, T - t + 1):
            rhs += J.CI([ht], [('O', t + j + 1)], G, Y(t, j)) + J.CI([ht], [('A', t + j)], G, Y(t, j) + [('O', t + j + 1)])
        rhs += J.I([ht], G, Y(t, T - t + 1))
        rec.eq('Prop 5.5: second line', lhs, rhs)
        # TD bias
        vM = J.cexp(v[t + 1], [('M', t + 1)])[0]
        vU = J.cexp(v[t + 1], [('U', t + 1)])[0]
        R1 = J.vals(('R', t + 1))
        lhs = J.cexp(Gt, [Mt, At])[0]
        rhs1 = J.cexp([r + g * a for r, a in zip(R1, vM)], [Mt, At])[0]
        rhs2 = J.cexp([g * (a - b) for a, b in zip(vU, vM)], [Mt, At])[0]
        for i in range(J.N):
            rec.eq('Prop 5.4: Bellman on the memory', lhs[i], rhs1[i] + rhs2[i])
        zero = J.cexp([a - b for a, b in zip(vU, vM)], [('M', t + 1)])[0]
        for i in range(J.N):
            rec.eq('Prop 5.4: E[v(U\')-v(M\')|M\'] = 0', zero[i], 0.0)
        rec.le('Prop 5.4: size of the last term (shown)', -max(abs(x) for x in rhs2), 0.0)
    # per-update value pieces
    for n in range(1, T + 1):
        Gn = J.vals(('G', n))
        X0 = J.cexp(Gn, [('h', n), ('A', n)])[0]
        Xn = J.cexp(Gn, [('M', n), ('A', n)])[0]
        deltas = []
        prev = X0
        for s in range(1, n + 1):
            Xs = J.cexp(Gn, Z(s, n - s))[0]
            Xs_minus = J.cexp(Gn, [('U', s)] + Z(s, n - s))[0]
            for i in range(J.N):
                rec.eq('Prop 5.3: E[G|U_s,Z_s] = E[G|F_s-1]', Xs_minus[i], prev[i])
            deltas.append([a - b for a, b in zip(Xs_minus, Xs)])
            prev = Xs
        tot = [sum(d[i] for d in deltas) for i in range(J.N)]
        for i in range(J.N):
            rec.eq('Prop 5.3: pieces sum to the error', tot[i], X0[i] - Xn[i])
        for a in range(len(deltas)):
            for b in range(a + 1, len(deltas)):
                rec.eq('Prop 5.3: pieces uncorrelated', J.E([x * y for x, y in zip(deltas[a], deltas[b])]), 0.0)
        rec.eq('Prop 5.3: MSE = sum of mean squares', J.E([(x - y) ** 2 for x, y in zip(X0, Xn)]),
               sum(J.E([x * x for x in d]) for d in deltas))


def check_merge(J, w, rec, alpha, lam, p):
    """Gibbs identity and the exact loss of merging, at t = 1."""
    t = 1
    Gt = J.vals(('G', t))
    _, qh = J.cexp(Gt, [('h', t), ('A', t)])
    _, qm = J.cexp(Gt, [('M', t), ('A', t)])
    hs = set(J.vals(('h', t)))
    hm = dict(zip(J.vals(('h', t)), J.vals(('M', t))))
    rng = random.Random(7)
    for h in hs:
        m = hm[h]
        sh = {a: alpha * qh[(h, a)] for a in ACTS}
        sm = {a: alpha * qm[(m, a)] for a in ACTS}
        Zh = sum(p[a] * math.exp(sh[a] / lam) for a in ACTS)
        Zm = sum(p[a] * math.exp(sm[a] / lam) for a in ACTS)
        pih = {a: p[a] * math.exp(sh[a] / lam) / Zh for a in ACTS}
        pim = {a: p[a] * math.exp(sm[a] / lam) / Zm for a in ACTS}
        kl = lambda x, y: sum(x[a] * math.log(x[a] / y[a]) for a in ACTS)
        pr = normalize([rng.random() + 0.1 for _ in ACTS])
        pr = dict(zip(ACTS, pr))
        rec.eq('Prop 6.4: variational identity', sum(pr[a] * sh[a] for a in ACTS) - lam * kl(pr, p),
               lam * math.log(Zh) - lam * kl(pr, pih))
        gap = {a: qh[(h, a)] - qm[(m, a)] for a in ACTS}
        rec.eq('Prop 6.4: loss formula', lam * kl(pim, pih),
               lam * math.log(sum(pim[a] * math.exp(alpha * gap[a] / lam) for a in ACTS))
               - alpha * sum(pim[a] * gap[a] for a in ACTS))
        loss_direct = (sum(pih[a] * sh[a] for a in ACTS) - lam * kl(pih, p)) - (sum(pim[a] * sh[a] for a in ACTS) - lam * kl(pim, p))
        rec.eq('Prop 6.4: loss = lam KL(pi_m||pi_h)', lam * kl(pim, pih), loss_direct)


# ---------------------------------------------------------------- objective and gradients
def frozen_laws(J, w):
    T = w.T
    Pc = {n: J.cdist([('O', n + 1)], [('M', n), ('A', n)]) for n in range(1, T + 1)}
    Pm = {n: J.cdist([('O', n + 1)], [])[()] for n in range(1, T + 1)}
    return Pc, Pm


def payoffs(J, w, p, alpha, lam, laws):
    """Per-step payoff r~_n with the laws held fixed."""
    Pc, Pm = laws
    T, out = w.T, {}
    for n in range(1, T + 1):
        kma = J.keys([('M', n), ('A', n)])
        On, Ms, As, Rs = J.vals(('O', n + 1)), J.vals(('M', n)), J.vals(('A', n)), J.vals(('R', n + 1))
        out[n] = [alpha * Rs[i] + math.log(Pc[n][kma[i]][(On[i],)] / Pm[n][(On[i],)])
                  - lam * math.log(w.pi[Ms[i]][As[i]] / p[As[i]]) for i in range(J.N)]
    return out


def objective(w, p, alpha, lam, mu=0.0, t_mem=None):
    J = world_joint(w)
    T, g = w.T, w.gamma
    tot = 0.0
    for n in range(1, T + 1):
        ER = J.E(J.vals(('R', n + 1)))
        info = J.I([('M', n), ('A', n)], [('O', n + 1)])
        kl = J.E([sum(w.pi[m][a] * math.log(w.pi[m][a] / p[a]) for a in ACTS) for m in J.vals(('M', n))])
        tot += g ** (n - 1) * (alpha * ER + info - lam * kl)
    if t_mem is not None:
        tot -= mu * g ** (t_mem - 1) * J.I([('M', t_mem)], [('h', t_mem)])
    return tot, J


def check_policy_gradient(w, rec, alpha, lam, p, eps=1e-5):
    T, g = w.T, w.gamma
    J0, J = objective(w, p, alpha, lam)
    laws = frozen_laws(J, w)
    r = payoffs(J, w, p, alpha, lam, laws)
    # E[ ell_n | h_n, A_n ] = B_n(h,A) - d_n(h,M,A)  (Step 4)
    Pc, Pm = laws
    for n in range(1, T + 1):
        Ph = J.cdist([('O', n + 1)], [('h', n), ('A', n)])
        ell = [x - alpha * rr + lam * math.log(w.pi[m][a] / p[a]) for x, rr, m, a in
               zip(r[n], J.vals(('R', n + 1)), J.vals(('M', n)), J.vals(('A', n)))]
        lhs = J.cexp(ell, [('h', n), ('A', n)])[0]
        khA, kmA = J.keys([('h', n), ('A', n)]), J.keys([('M', n), ('A', n)])
        for i in range(J.N):
            ph = Ph[khA[i]]
            B = sum(po * math.log(po / Pm[n][o]) for o, po in ph.items())
            d = sum(po * math.log(po / Pc[n][kmA[i]][o]) for o, po in ph.items())
            rec.eq('Lem 6.1b: E[l|h,A] = B - d', lhs[i], B - d)
    # gradient formula G(m,a)
    Gf = {}
    for m in MEMS:
        for a in ACTS:
            tot = 0.0
            for j in range(1, T + 1):
                Sj = [sum(g ** k * r[j + k][i] for k in range(T - j + 1)) for i in range(J.N)]
                _, d = J.cexp(Sj, [('M', j), ('A', j)])
                pm_ = sum(pp for pp, mm in zip(J.P, J.vals(('M', j))) if mm == m)
                if pm_ > 0:
                    tot += g ** (j - 1) * pm_ * (d[(m, a)] - lam)
            Gf[(m, a)] = tot
    for m in MEMS:
        wp, wm = w.clone(), w.clone()
        wp.pi[m][0] += eps
        wp.pi[m][1] -= eps
        wm.pi[m][0] -= eps
        wm.pi[m][1] += eps
        num = (objective(wp, p, alpha, lam)[0] - objective(wm, p, alpha, lam)[0]) / (2 * eps)
        rec.fd('Prop 6.3: gradient (finite differences)', num, Gf[(m, 0)] - Gf[(m, 1)])


def check_memory_gradient(w, rec, alpha, lam, mu, p, eps=1e-5):
    t = w.rand_step
    T, g = w.T, w.gamma
    J0, J = objective(w, p, alpha, lam, mu, t)
    laws = frozen_laws(J, w)
    r = payoffs(J, w, p, alpha, lam, laws)
    Pc, Pm = laws
    # values under the objective V_n(h_n, M_n)
    Vlist = {}
    for n in range(1, T + 1):
        S = [sum(g ** i * r[n + i][k] for i in range(T - n + 1)) for k in range(J.N)]
        Vlist[n] = J.cexp(S, [('h', n), ('M', n)])[0]
    Vlist[T + 1] = [0.0] * J.N
    _, Vt = J.cexp(Vlist[t], [('h', t), ('M', t)])
    _, Vm = J.cexp(Vlist[t], [('M', t)])
    # (a) V_t(h,m) - V(m) = E[ sum_k g^k Cov_Z(L, W) | h, m ]
    ht = ('h', t)
    S = [0.0] * J.N
    for k in range(0, T - t + 1):
        Zs, On = Z(t, k), ('O', t + k + 1)
        Ph, Pz = J.cdist([On], [ht] + Zs), J.cdist([On], Zs)
        _, Vw = J.cexp(Vlist[t + k + 1], Zs + [On])
        kz, khz = J.keys(Zs), J.keys([ht] + Zs)
        Ms, As = J.vals(('M', t + k)), J.vals(('A', t + k))
        for i in range(J.N):
            n = t + k
            ma = (Ms[i], As[i])
            cov = 0.0
            for (o,), pzo in Pz[kz[i]].items():
                W = alpha * o[0] + math.log(Pc[n][ma][(o,)] / Pm[n][(o,)]) + g * Vw.get(kz[i] + (o,), 0.0)
                cov += (Ph[khz[i]].get((o,), 0.0) - pzo) * W
            S[i] += g ** k * cov
    _, ES = J.cexp(S, [ht, ('M', t)])
    for (h, m), val in ES.items():
        rec.eq('Prop 6.5b: V_t(h,m)-V(m) = E[sum g^k Cov(L,W)]', Vt[(h, m)] - Vm[(m,)], val)
    # (b) finite-difference derivative in q_t(m|h)
    hv = J.vals(ht)
    Ph_t = defaultdict(float)
    for pp, hh in zip(J.P, hv):
        Ph_t[hh] += pp
    qm_ = defaultdict(float)
    for pp, mm in zip(J.P, J.vals(('M', t))):
        qm_[mm] += pp
    hs = sorted(Ph_t, key=lambda x: -Ph_t[x])[:4]
    for hh in hs:
        q = w.enc(hh)
        formula = g ** (t - 1) * Ph_t[hh] * ((Vt[(hh, 0)] - mu * math.log(q[0] / qm_[0]))
                                             - (Vt[(hh, 1)] - mu * math.log(q[1] / qm_[1])))
        wp, wm = w.clone(), w.clone()
        wp._enc[hh] = {0: q[0] + eps, 1: q[1] - eps}
        wm._enc[hh] = {0: q[0] - eps, 1: q[1] + eps}
        num = (objective(wp, p, alpha, lam, mu, t)[0] - objective(wm, p, alpha, lam, mu, t)[0]) / (2 * eps)
        rec.fd('Prop 6.5a: gradient (finite differences)', num, formula)


# ---------------------------------------------------------------- the two examples
def check_examples(rec):
    bits = math.log(2)

    def joint(rows):
        P = [r[1] for r in rows]
        return Joint(P, lambda a: [r[0][a] for r in rows])

    # exclusive or: C uniform (= h), M constant, A uniform, R uniform, Obar = R xor C, A' = Obar
    rows = []
    for C in (0, 1):
        for A in (0, 1):
            for R in (0, 1):
                Ob = R ^ C
                rows.append(({('h', 1): C, ('M', 1): 0, ('A', 1): A, ('R', 2): R, ('Ob', 2): Ob,
                              ('O', 2): (R, Ob), ('A', 2): Ob}, 1 / 8))
    J = joint(rows)
    h, M, A, R, B, O, A1 = ('h', 1), ('M', 1), ('A', 1), ('R', 2), ('Ob', 2), ('O', 2), ('A', 2)
    rec.eq('Ex 7.3 (xor): D = 1 bit', J.I([h], [O], [M, A]) / bits, 1.0)
    rec.eq('Ex 7.3 (xor): reward-first D parts (0,1)', J.I([h], [R], [M, A]) / bits + 10 * J.I([h], [B], [M, A, R]) / bits, 10.0)
    rec.eq('Ex 7.3 (xor): rest-first D parts (1,0)', J.I([h], [R], [M, A, B]) / bits + 10 * J.I([h], [B], [M, A]) / bits, 1.0)
    rec.eq('Ex 7.3 (xor): dP = 0', J.I([h], [A1], [M, A]) / bits, 0.0)
    rec.eq('Ex 7.3 (xor): reward-first dP reward part -1',
           (J.I([R], [A1], [M, A]) - J.I([R], [A1], [h, A])) / bits, -1.0)
    rec.eq('Ex 7.3 (xor): reward-first dP rest part +1', J.I([h], [A1], [M, A, R]) / bits, 1.0)
    rec.eq('Ex 7.3 (xor): rest-first dP parts (0,0)',
           abs(J.I([h], [A1], [M, A, B])) / bits + abs(J.I([B], [A1], [M, A]) - J.I([B], [A1], [h, A])) / bits, 0.0)
    rec.eq('Ex 7.3 (xor): co-information -1 bit', J.CI([h], [R], [B], [M, A]) / bits, -1.0)
    # the observation shows the action: C uniform (= h), M constant, A uniform, R = 1{A=C}, Obar = A
    rows = []
    for C in (0, 1):
        for Aa in (0, 1):
            Rr = int(Aa == C)
            rows.append(({('h', 1): C, ('M', 1): 0, ('A', 1): Aa, ('R', 2): Rr, ('Ob', 2): Aa,
                          ('O', 2): (Rr, Aa)}, 1 / 4))
    J = joint(rows)
    rec.eq('Ex 7.4 (shows action): D = 1 bit', J.I([h], [O], [M, A]) / bits, 1.0)
    rec.eq('Ex 7.4 (shows action): dE = 0', J.I([h], [A], [M, O]) / bits, 0.0)
    rec.eq('Ex 7.4 (shows action): reward-first dE parts (1,-1)',
           J.I([h], [A], [M, R]) / bits - 10 * (J.I([A], [B], [h, R]) - J.I([A], [B], [M, R])) / bits, 11.0)
    rec.eq('Ex 7.4 (shows action): rest-first dE parts (0,0)',
           abs(J.I([h], [A], [M, B])) / bits + abs(J.I([A], [R], [h, B]) - J.I([A], [R], [M, B])) / bits, 0.0)
    rec.eq('Ex 7.4 (shows action): U^R = 1 bit (I(h;R|M,A)=1, I(h;Ob|M,A)=0)',
           J.I([h], [R], [M, A]) / bits - J.I([h], [B], [M, A]) / bits, 1.0)


# ---------------------------------------------------------------- main
def main():
    gamma, alpha, lam, mu = 0.9, 0.7, 0.5, 0.3
    rec = Rec()
    check_examples(rec)
    for seed, T in ((1, 4), (2, 4), (3, 3), (4, 3), (5, 3)):
        w = World(seed, T, gamma)
        J = world_joint(w)
        p = dict(zip(ACTS, normalize([random.Random(seed + 100).random() + 0.2 for _ in ACTS])))
        check_four_terms(J, w, rec)
        check_updates(J, w, rec)
        check_value(J, w, rec)
        check_merge(J, w, rec, alpha, lam, p)
        check_policy_gradient(w, rec, alpha, lam, p)
        print(f"seed {seed} (T={T}, {J.N} trajectories) done", flush=True)
    for seed in (11, 12, 13):
        w = World(seed, 3, gamma, rand_step=2)
        p = dict(zip(ACTS, normalize([random.Random(seed + 100).random() + 0.2 for _ in ACTS])))
        check_memory_gradient(w, rec, alpha, lam, mu, p)
        print(f"memory seed {seed} done", flush=True)
    ok = rec.report()
    print("ALL PASS" if ok else "SOME CHECKS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
