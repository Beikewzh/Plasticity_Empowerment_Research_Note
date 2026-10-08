# Mathematical audit (outside the LaTeX document)

Date: 2026-10-08 (after an independent referee pass; its corrections are applied in `main.tex`). Companion to `main.tex` ("Constructing Sufficient Agent States from Plasticity and Empowerment").

## Rigorously established (finite alphabets, finite horizon, memory updated by a deterministic function,
## policy depending on the past only through the memory; all statements under the interaction distribution P)

- Prop. (reward--memory Markov): D_t = 0 => P(R_{t+1}, M_{t+1} | h_t, A_t) = P(. | M_t, A_t) on the support.
- Thm. (control): D_t = 0 for all t => backward induction on memory over the agent's own actions gives the optimum
  over all behaviours that stay within those actions (history-dependent included); with every action alive
  (full support) q*_t(phi_t(h), a) = Q*_t(h, a) at reachable h and greedy on q* is optimal. Existence only.
- Prop. (mismatches): E - E(M) = I(h; A | M, O') >= 0 and P_{t+1}(M) - P_{t+1} = I(h_t; A_{t+1} | M_t, A_t) >= 0,
  by entropy expansions; the cancellations use exactly the two structural facts above.
- Thm. (decomposition): D = D^0 + (E - E(M)); 0 <= dP <= D (data processing along h -> O' -> M' -> A' given (M, A)).
- Prop. (plasticity alone): dP = 0 does NOT imply D = 0 (any memory under a policy that ignores it; pennies: dP = 0,
  D = 1 bit). Sufficient condition: linearly independent action laws over the possible next observations after
  each (m, a) (needs |supp O'| <= |A|); unrealistic in general.
- Thm. (window): L_O* + L_E* = I(W; O' | M, A) = D^(k); 0 <= L_P* <= D^(k) <= D; D - D^(k) = I(h; O' | M, A, W).
- Prop. (direct objective): E[-log q(O' | M, A)] >= H(O' | h, A) + D_t, equality for the Bayes predictor; the first
  term is fixed by the interaction distribution. Student losses l_O + l_E have population value H(A | M) + H(O' | M, A).
- All identities and both counterexamples verified by exact enumeration (`checks/check.py`, `checks/outputs.txt`).

## Established only under an assumption
- Claim B: window losses zero + I(h; O' | M, A, W) = 0 (the window holds all relevant forgotten information,
  e.g. k-th order Markov observations) => D = 0. The assumption is exactly what finite data cannot certify.
- Theorem (control) part (b) needs full action support; part (a) is the general statement (optimum within the
  agent's own actions).
- Prop. (direct): "the first term does not depend on theta" needs a fixed interaction distribution (fixed behaviour
  or fresh on-policy data treated as fixed for the step). H(A | M) in the student pair equals H(A | h) on fresh
  on-policy data; on stale data it is an inertia term.

## Corrections made after the referee pass (were in the first draft of this note)
- "deterministic (or independently randomised) update": a randomised update breaks M_t = phi_t(h_t) and the
  identities (M_t a fresh coin, A_t = M_t, O_{t+1} = A_t: D = D^0 = 0 but Delta E = 1 bit); fixed to deterministic.
- "differ several steps later ... nothing in the stream reveals it": wrong; a later difference along followed paths
  shows as D_{t+j} > 0 (late, possibly outside any window). Fixed.
- "records the reward" in the plasticity sufficient condition: redundant; linear independence over next observations
  already forces the update to separate them. Fixed.
- Bias directions of the cross-entropy difference were reversed (poor window head => biased DOWNWARD). Fixed.
- "the action-conditioned loss and the pair have the same population minimisers": false (constant memory under an
  old policy reading an irrelevant history bit). Replaced by the exact statement: pair = D_t + I(h; A | M) on the
  data for lambda_E = 1; equal to D^0 + lambda_E Delta E only at the memory that generated the data.
- "kernel the agent can estimate from its own stream": needs repeated episodes or time-invariance. Fixed.
- Hutter attribution (his point is that the reduced process need not be Markov); Allen/Rakelly cited for the right
  results; Abel's framework is directed information over time, ours are single terms. Fixed.
- Symbol clashes (P for probability/distribution/plasticity; phi for memory map and policy parameters) and
  undefined gamma, return, V^lambda, Q*, sg, BPTT: fixed.

## Claims in the earlier PDFs that were incorrect or overstated, and are not reproduced
- "Perfect = greedy on a reward state + coverage" as an equivalence: the only-if direction is false (an optimal
  agent need not have a state). Corrected to: optimal iff optimal within own actions and coverage; the state is
  needed to read optimality off the memory, not to be optimal.
- "Both flows read exactly through the memory => state": false (W11: exogenous reward; delayed effects). The flows
  alone are blind where the action does not change the next observation and the policy ignores what it sees.
- Return-targeted empowerment inside information terms, and "testing cost" regret statements: dropped here per the
  brief (Target 1 is the predictive state; control is Theorem 1).
- "Write >= plasticity" as a recipe line: holds automatically for any memory the policy reads; not a constraint.
- "Any policy step gains at most (1/2) b^2 E^R(M)": the bound is for the tilt along the visible advantage only;
  not used in this note.
- Difference losses with a stopped-gradient teacher presented as a new memory objective: with sg on the teacher the
  gradient is that of the student loss alone; the differences are monitors/estimators, not objectives.
- "E[L_E] = lost empowerment": only against the whole history; against a window it is a lower bound, and only
  with Bayes-optimal heads; samples can be negative.

## Which objectives are actually sufficient
- Population, ideal heads, fixed interaction distribution: minimising E[-log q(O' | M, A)] over theta minimises
  D_t exactly (up to the fixed constant). Equivalently l_O + l_E up to H(A | M).
- Window versions (student vs look-back) reach only D^(k) <= D; zero certifies nothing beyond the window.
- L_P (next-action prediction) is a lower bound on D^(k); zero certifies nothing; degenerate under indifferent
  policies; never a loss on theta alone.

## Does the algorithm optimise the theoretical objective?
- Memory gradient = gradient of l_O + lambda_E l_E (student cross-entropies). In population, with heads that have learned
  the true laws, E[l_O + lambda_E l_E] = const + I(h; O' | M) + lambda_E I(h; A | M, O'); for lambda_E = 1 this is
  const + D_t + I(h; A | M) on the data (defect plus an inertia term toward the memory that generated the data).
  So yes, it descends the defect on the current data, up to that inertia term; it does not minimise sum_t D_t(theta)
  with theta-dependent P.
  The window heads and next-action heads contribute no gradient to theta; they estimate the gaps.
- Estimators exact only for Bayes-optimal heads; two time scales (heads > memory > policy); policy on detached memory;
  truncated BPTT limits the dependences the memory can learn to what crosses windows.
- No convergence claim (Claim C open).

## Open
- Convergence of gradient training to D_t = 0; finite-sample behaviour of cross-entropy differences; recovery time
  after an environment change; whether inverse / next-action heads improve gradients or sample efficiency relative
  to one action-conditioned predictor (to be settled on worlds where the two halves of the defect are known exactly).

## Strongest defensible contribution (one paragraph)
For an agent acting from an online memory in an arbitrary history-dependent environment, the predictive state
defect D_t = I(h_t; O_{t+1} | M_t, A_t) is zero exactly when the memory is a sufficient predictive state, and such
a state supports exact Bellman control on the support of the interaction (optimal over all behaviours once every
action is tried). Replacing the history by the memory can only lower Abel's empowerment and raise Abel's
plasticity, and the two mismatches are information terms of the agent's own stream: the empowerment mismatch is
exactly the half of the defect that action-free prediction cannot see (D = D^0 + Delta E), and the plasticity
mismatch is a lower bound on the defect readable from the agent's next action alone. Finite-window versions of
these quantities satisfy the same identities with the window in place of the history, bound D_t from below, and
differ from it by precisely the predictive information outside the window, which finite interaction cannot
certify. The resulting training objective coincides with action-conditioned prediction; what the mismatches add
is a characterisation and a pair of estimators, not a new minimiser, and whether they help learning beyond direct
prediction is open.
