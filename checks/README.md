# Exact checks for the note

`check.py` enumerates every trajectory of small history-dependent worlds (horizon 4) and asserts,
at every step, the identities of the note:

- empowerment mismatch  `E - E(M) = I(h; A | M, O') >= 0`
- plasticity mismatch   `P_{t+1}(M) - P_{t+1} = I(h_t; A_{t+1} | M_t, A_t) >= 0`
- decomposition         `D = D^0 + (E - E(M))`,  `0 <= dP <= D`
- finite window W (last k steps): `L_O* + L_E* = I(W; O' | M, A) = D^(k)`, `0 <= L_P* <= D^(k) <= D`,
  residual `D - D^(k) = I(h; O' | M, A, W)`
- counterexamples: `dP = 0` with `D = 1` bit (policy ignores its memory); `D^(1) = 0` with `D = 1` bit
  (reward follows the action taken three steps earlier).

The script imports helper libraries from the author's local `interaction_theory` folder
(`perfect_state/plib.py`, `rsi2/lib2.py`, `rsi/pairlib.py`); `outputs.txt` is the recorded output.
