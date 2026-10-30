# Sections, tables and figures cited by an earlier numbering

The comments, docstrings and READMEs of this code cite the book's sections, tables and figures by their final numbers, with an `S` for the online supplement (Table S13.13 is in the supplement). A few files cite an earlier numbering of the book, because they cannot change without changing a record:

- files whose SHA-256 is fixed by a registered protocol or a frozen record: `ch06/published/PROTOCOL-R2.md` fixes `cart_env.py`, `protocol.py`, `train_sac.py`, `train_tdmpc2.py`, `report.py` and `report_r2.py` of `ch06/published`, `ch06/cart_learning.py`, `ch05/cart_feedback.py` and `ch05/cart_methods.py`; `ch16/complete/FROZEN.json` fixes the files of `ch16/complete`, `ch10/allovi`, `ch10/cart/control.py`, `ch11/cart_mujoco`, `ch16/bench/hypotheses.py` and `ch16/bench/stats.py`;
- registered documents, which are not edited after their runs: `ch06/published/PROTOCOL.md` and `ch16/HYPOTHESES.md`;
- text that a script prints into its shipped results, which would then differ from the files in `results/`: the summaries of `ch11/mujoco_cart.py`, `ch12/layers.py`, `ch12/numerics.py` and `ch13/husky_variants.py`, with `ch10/cart/model.py`.

This table gives the place in the book of every item that they cite by the earlier numbering.

| Cited as | Now | Where it is cited so |
|---|---|---|
| Section 4.3.4 | Section 4.3.5 | `ch10/allovi/contract.py`, `ch10/allovi/governor.py`, `ch10/cart/control.py` |
| Table 4.4 | Table 4.1 | `ch10/allovi/contract.py`, `ch10/allovi/model.py`, `ch10/allovi/checks.py` |
| Table 5.1 | Table S5.1 | `ch06/cart_learning.py` |
| Table 5.3 | Table 5.2 | `ch06/cart_learning.py` |
| Table 5.5 | Table 5.3 | `ch05/cart_methods.py` |
| Table 6.2 | Table S6.1 | `ch06/cart_learning.py` |
| Table 6.4 | Table 6.3 | `ch06/published/PROTOCOL.md`, `ch06/published/report.py` |
| Table 9.1 | Table 9.6 | `ch16/HYPOTHESES.md` |
| Section 10.2.2 | Section S10.9.2 | `ch10/cart/model.py` |
| Section 10.2.4 | Section S10.9.4 | `ch10/allovi/loop.py` |
| Section 11.1.3 | Section S11.9.1 | `ch11/cart_mujoco/scene.py`, `sim.py` and `track.py`; `ch16/complete/model.py` and `PROTOCOL.md`; the results of `ch11` and `ch12` |
| Section 11.1.4 | Section S11.9.2 | `ch11/cart_mujoco/track.py`; the summaries of `ch11/mujoco_cart.py` and `ch12/numerics.py` |
| Section 11.2.1 | Section S11.10.1 | `ch12/numerics.py` and its results |
| Section 11.2.3 | Section S11.10.3 | `ch12/numerics.py` and its results |
| Section 11.2.5 | Section S11.6 | `ch12/numerics.py` and its results |
| Section 12.2.2 | Section S12.9.2 | `ch12/numerics.py` and its results |
| Table 12.4 | Table S12.1 | `ch12/layers.py`, `ch12/numerics.py` and their results |
| Table 13.6 | Table S13.13 | `ch13/husky_variants.py` and `ch13/results/variants-summary.txt` |

`ch16/HYPOTHESES.md` also records, under its deviations, a correction of one of its cross-references in the numbering of that time.
