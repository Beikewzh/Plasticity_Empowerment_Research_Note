"""Exact checks for the research note 'Constructing sufficient agent states from plasticity and empowerment'.
T = 4, gamma irrelevant here (no values), bits unless stated.  Identities:
  dE = E - E(M) = I(h; A | M, O') >= 0;  dP = P_{t+1}(M) - P_{t+1} = I(h_t; A_{t+1} | M_t, A_t) >= 0;  D = D0 + dE;  dP <= D;
  finite window W = last k steps: LO + LE = I(W; O' | M, A) = D^(k);  0 <= LP <= D^(k) <= D;  D - D^(k) = I(h; O' | M, A, W).
Counterexamples: dP = 0 with D > 0 (policy ignores memory); D^(k) = 0 with D > 0 (dependence older than the window)."""
import sys, os
DIR = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(DIR, "..", "perfect_state"))
from plib import *
T = 4; EPS = 1e-7
def W6d(lag=2):                                  # reward follows the action taken `lag` steps before: R_{t+1} = 1 iff A_{t+1-lag} = 0 (lag=2: A_{t-1})
    def init(): return [((0,), 1.0)]
    def step(obs, acts, a):
        r = 1 if (len(acts) + 1 >= lag and acts[len(acts) - lag + 1] == 0) else 0
        return [((r,), 1.0)]
    return init, step
def window(k):                                   # W_t = (O_{t-k:t}, A_{t-k:t-1}) as a description of the history (clipped at the start)
    return lambda o, a: (tuple(o[-(k + 1):]), tuple(a[-k:]) if k > 0 else ())
const = lambda o, a: 0; coin = lambda o, a: o[-1][0]
out = []
def P_(*a):
    s = " ".join(str(x) for x in a); print(s); out.append(s)
def quantities(J, mem, t, k):
    H, At, An, Mt, On, Pa = hist(t), A(t), A(t + 1), Zt(mem, t), O(t + 1), past(t)
    Wt = Zt(window(k), t); MW = tup(Mt, Wt)
    E, EM = cmi(J, At, On, H), cmi(J, At, On, Mt); dE = E - EM; dE2 = cmi(J, H, At, tup(Mt, On))
    D, D0 = cmi(J, H, On, tup(Mt, At)), cmi(J, H, On, Mt)
    LO, LE = cmi(J, Wt, On, Mt), cmi(J, Wt, At, tup(Mt, On)); Dk = cmi(J, Wt, On, tup(Mt, At)); resid = cmi(J, H, On, tup(Mt, At, Wt))
    r = dict(E=E, EM=EM, dE=dE, D=D, D0=D0, LO=LO, LE=LE, Dk=Dk, resid=resid)
    assert abs(dE - dE2) < EPS and dE > -EPS, ("dE", dE, dE2)
    assert abs(D - D0 - dE) < EPS, ("D = D0 + dE", D, D0, dE)
    assert abs(LO + LE - Dk) < EPS, ("LO + LE = Dk", LO, LE, Dk)
    assert Dk <= D + EPS and abs(D - Dk - resid) < EPS, ("residual", D, Dk, resid)
    if t + 1 < T:
        Pn, PnM = cmi(J, On, An, tup(H, At)), cmi(J, On, An, tup(Mt, At)); dP = PnM - Pn; dP2 = cmi(J, H, An, tup(Mt, At)); LP = cmi(J, Wt, An, tup(Mt, At))
        assert abs(dP - dP2) < EPS and dP > -EPS and dP <= D + EPS, ("dP", dP, dP2, D)
        assert LP > -EPS and LP <= Dk + EPS, ("LP", LP, Dk)
        r.update(P=Pn, PM=PnM, dP=dP, LP=LP)
    return r
def row(label, world, mem, pol, k, t=2):
    J = joint(world, agent_rule(mem, pol), T); q = quantities(J, mem, t, k)
    P_(f"  {label:44s} k={k} t={t}: E {q['E']:.3f} E(M) {q['EM']:.3f} dE {q['dE']:.3f} | P {q.get('P', float('nan')):.3f} P(M) {q.get('PM', float('nan')):.3f} dP {q.get('dP', float('nan')):.3f} | D0 {q['D0']:.3f} D {q['D']:.3f} | window: LO {q['LO']:.3f} LE {q['LE']:.3f} LP {q.get('LP', float('nan')):.3f} D^(k) {q['Dk']:.3f} residual {q['resid']:.3f}")
    return q
u2 = lambda t, m: {0: .5, 1: .5}
P_("identities checked at every row (assertions); numbers in bits")
row("pennies, uniform, constant memory", W7p(), const, u2, 1)
row("pennies, uniform, memory = coin", W7p(), coin, u2, 1)
row("pennies, plays the coin, memory = coin", W7p(), coin, lambda t, m: {m: 1.0}, 1)
row("W1, uniform, constant memory", W1("I"), const, u2, 1)
row("W1, uniform, memory = state", W1("I"), DESCS["S_t"], u2, 1)
row("W1, agent 1, memory = last reward", W1("I"), lambda o, a: o[-1][2], lambda t, m: bern(0.85 if m else 0.20), 1)
row("W1, agent 1, memory = last reward", W1("I"), lambda o, a: o[-1][2], lambda t, m: bern(0.85 if m else 0.20), 0)
P_("counterexample 1: dP = 0 although D > 0 (the policy ignores its memory): pennies, uniform, constant memory above (dP 0, D 1)")
P_("counterexample 2: finite window misses an old dependence: reward follows the action taken 3 steps before; constant memory")
q = row("W6 lag 3, uniform, constant memory", W6d(3), const, u2, 1, t=3)
assert q['Dk'] < EPS and q['D'] > 0.5, q
q = row("W6 lag 3, uniform, constant memory", W6d(3), const, u2, 2, t=3)
assert q['Dk'] > 0.5 and q['resid'] < EPS, q
P_("all assertions passed"); open(os.path.join(DIR, "outputs.txt"), "w").write("\n".join(out) + "\n")
