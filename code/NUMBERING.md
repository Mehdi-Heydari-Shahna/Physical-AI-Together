# Sections, tables and figures cited by an earlier numbering

The comments, docstrings, READMEs and printed results of this code cite the book's sections, tables and figures as the book's version of 31 October 2026 numbers them, with an `S` for the online supplement (Table S13.11 is in the supplement); the results of release v1.0 are those of that version. The book's version of 4 November 2026 numbers some of these items differently, and a few files cite a still earlier numbering, because they cannot change without changing a record:

- files whose SHA-256 is fixed by a pre-specified protocol or a frozen record: `ch06/published/PROTOCOL-R2.md` fixes `cart_env.py`, `protocol.py`, `train_sac.py`, `train_tdmpc2.py`, `report.py` and `report_r2.py` of `ch06/published`, `ch06/cart_learning.py`, `ch05/cart_feedback.py` and `ch05/cart_methods.py`; `ch16/complete/FROZEN.json` fixes the files of `ch16/complete`, `ch10/allovi`, `ch10/cart/control.py`, `ch11/cart_mujoco`, `ch16/bench/hypotheses.py` and `ch16/bench/stats.py`;
- pre-specified documents, which are not edited after their runs: `ch06/published/PROTOCOL.md`, `ch16/HYPOTHESES.md` and `ch17/HYPOTHESES.md`;
- text that a script prints into its shipped results, which would then differ from the files in `results/`: the summaries of `ch09/cart_lead_time.py`, `ch11/mujoco_cart.py`, `ch12/layers.py`, `ch12/numerics.py` and `ch13/husky_variants.py`, with `ch10/cart/model.py`.

## Items cited by a still earlier numbering

This table gives the place in the book's version of 4 November 2026 of every item that these files cite by an earlier numbering.

| Cited as | Now | Where it is cited so |
|---|---|---|
| Section 4.3.4 | Section 4.3.5 | `ch10/allovi/contract.py`, `ch10/allovi/governor.py`, `ch10/cart/control.py` |
| Table 4.4 | Table 4.1 | `ch10/allovi/contract.py`, `ch10/allovi/model.py`, `ch10/allovi/checks.py` |
| Table 5.1 | Table S5.1 | `ch06/cart_learning.py` |
| Table 5.3 | Table 5.2 | `ch06/cart_learning.py` |
| Table 5.5 | Table 5.3 | `ch05/cart_methods.py` |
| Table 6.2 | Table S6.1 | `ch06/cart_learning.py` |
| Table 6.4 | Table 6.3 | `ch06/published/PROTOCOL.md`, `ch06/published/report.py` |
| Table 9.1 | Table 9.2 | `ch16/HYPOTHESES.md` |
| Theorem 9.12 | Theorem 9.15 | `ch09/README.md`, `ch09/cart_lead_time.py` and its summary, `ch10/templates/control_package.py`, `ch14/hdmm_governed.py`, `ch17/contrib/full_reverse.py` |
| Corollary 9.13 | Corollary 9.16 | `ch17/contrib/early_window.py`, `ch17/cards/reader.early-window.md`, `ch17/HYPOTHESES.md` |
| Proposition 9.16 | Proposition 9.19 | `ch09/README.md`, `ch09/cart_lead_time.py` |
| Example 9.17 | Example 9.20 | `ch09/README.md`, `ch09/cart_lead_time.py` and its summary (`[example 9.17]`) |
| Figure 9.4 | Figure 9.1 | `ch09/README.md`, `ch09/cart_lead_time.py` |
| Figure 9.5 | Figure 9.2 | `ch09/README.md` |
| Section 10.2.2 | Section S10.9.2 | `ch10/cart/model.py` |
| Section 10.2.4 | Section S10.9.4 | `ch10/allovi/loop.py` |
| Section 11.1.3 | Section S11.9.1 | `ch11/cart_mujoco/scene.py`, `sim.py` and `track.py`; `ch16/complete/model.py` and `PROTOCOL.md`; the results of `ch11` and `ch12` |
| Section 11.1.4 | Section S11.9.2 | `ch11/cart_mujoco/track.py`; the summaries of `ch11/mujoco_cart.py` and `ch12/numerics.py` |
| Section 11.2.1 | Section S11.10.1 | `ch12/numerics.py` and its results |
| Section 11.2.3 | Section S11.10.3 | `ch12/numerics.py` and its results |
| Section 11.2.5 | Section S11.6 | `ch12/numerics.py` and its results |
| Section 12.2.2 | Section S12.10.2 | `ch12/numerics.py` and its results |
| Table 12.4 | Table S12.2 | `ch12/layers.py`, `ch12/numerics.py` and their results |
| Table 13.6 | Table S13.11 | `ch13/husky_variants.py` and `ch13/results/variants-summary.txt` |

`ch16/HYPOTHESES.md` also records, under its deviations, a correction of one of its cross-references in the numbering of that time.

## Every other item whose number changed

Every other citation in this code uses the numbering of the book's version of 31 October 2026. This table gives the current number of every item of that version, in the book and in the supplement, whose number changed: chapters, sections, tables, figures, listings, algorithms and numbered results, compared by their labels in the book's sources; "merged or removed" marks an item whose content is now part of another or no longer printed.

| Version of 31 October 2026 | Version of 4 November 2026 |
|---|---|
| Section 1.3.4 | Section 1.3.3 |
| Section 1.3.5 | Section 1.3.4 (merged) |
| Section 1.3.6 | Section 1.3.4 (merged) |
| Table 1.2 | merged or removed |
| Table 1.3 | Table 1.2 |
| Figure 1.1 | merged or removed |
| Figure 1.2 | merged or removed |
| Figure 1.3 | Figure 1.1 |
| Figure 1.4 | Figure 1.2 |
| Figure 1.5 | Figure 1.3 |
| Box 1.2 | merged or removed |
| Box 1.3 | Section 1.2.1 |
| Table 3.1 | Table S3.2 |
| Table 3.2 | Table 3.1 |
| Table 3.3 | Table 3.2 |
| Table 3.4 | Table S3.3 |
| Table 3.5 | Table 3.3 |
| Figure 3.3 | Figure S3.1 |
| Figure 3.4 | Figure 3.3 |
| Figure 3.5 | Figure 3.4 |
| Section 4.3.6 | merged or removed |
| Figure 4.3 | merged or removed |
| Figure 4.4 | Figure 4.3 |
| Figure 4.5 | merged or removed |
| Figure 4.6 | Figure 4.4 |
| Figure 4.7 | merged or removed |
| Figure 5.3 | Figure S5.1 |
| Figure 5.4 | Figure S5.2 |
| Figure 5.5 | Figure S5.3 |
| Example 5.3 | Example 5.2 |
| Lemma 5.2 | Lemma S5.1 |
| Section 6.4.1 | Section 6.4 (merged) |
| Section 6.4.2 | Section 6.4 (merged) |
| Section 6.4.3 | Section 6.4 (merged) |
| Section 6.4.4 | Section 6.4 (merged) |
| Table 6.4 | Table S6.3 |
| Box 6.1 | merged or removed |
| Box 6.2 | Box S6.1 |
| Table 7.2 | Table S7.1 |
| Table 7.3 | Table 7.2 |
| Table 7.4 | Table 7.3 |
| Table 7.5 | Table S7.5 |
| Table 8.3 | Table S8.2 |
| Table 8.4 | Table S8.3 |
| Table 8.5 | Table S8.4 |
| Figure 8.4 | Figure S8.1 |
| Figure 8.5 | Figure S8.2 |
| Section 9.1.3 | Section 9.1.4 (merged) |
| Section 9.1.4 | Section 9.1.3 |
| Section 9.1.5 | Section 9.1.4 (merged) |
| Section 9.1.6 | Section 9.2 |
| Section 9.4.1 | Section 9.4.2 |
| Section 9.4.2 | Section 9.4.1 |
| Section 9.4.4 | Section 9.2.7 |
| Table 9.1 | merged or removed |
| Table 9.2 | Table 9.1 (merged) |
| Table 9.3 | Table 9.1 (merged) |
| Table 9.4 | Table 9.1 (merged) |
| Table 9.5 | merged or removed |
| Table 9.6 | Table 9.2 |
| Figure 9.1 | merged or removed |
| Figure 9.2 | Table 9.1 (merged) |
| Figure 9.3 | merged or removed |
| Figure 9.4 | Figure 9.1 |
| Figure 9.5 | Figure 9.2 |
| Box 9.1 | merged or removed |
| Example 9.11 | Example 9.12 |
| Example 9.17 | Example 9.18 |
| Example 9.19 | Example 9.20 |
| Lemma 9.12 | Lemma 9.14 |
| Lemma 9.13 | Section 9.3.1 |
| Proposition 9.6 | Proposition 9.7 |
| Proposition 9.8 | Proposition 9.9 |
| Proposition 9.9 | Proposition 9.10 |
| Proposition 9.10 | Proposition 9.11 |
| Proposition 9.16 | Proposition 9.17 |
| Proposition 9.18 | Proposition 9.19 |
| Proposition 9.20 | Proposition 9.13 |
| Theorem 9.14 | Theorem 9.15 |
| Corollary 9.7 | Corollary 9.8 |
| Corollary 9.15 | Corollary 9.16 |
| Table 10.2 | Table 10.3 |
| Table 11.1 | Table 11.3 |
| Table 12.1 | Table S12.1 |
| Table 12.2 | Table 12.1 |
| Figure 12.1 | Figure S12.2 |
| Section 13.3.2 | Section S13.8.1 |
| Section 13.3.3 | Section S13.8.2 |
| Section 13.3.4 | Section 13.3.3 |
| Section 13.3.5 | Section S13.8.4 |
| Section 13.3.6 | Section S13.8.5 |
| Section 13.3.7 | Section 13.3.4 |
| Figure 13.2 | Figure S13.1 |
| Figure 13.3 | Figure S13.2 |
| Figure 13.4 | Figure S13.3 |
| Figure 13.5 | Figure S13.4 |
| Figure 13.6 | Figure 13.2 |
| Figure 13.7 | Figure S13.5 |
| Figure 13.8 | Figure 13.3 |
| Figure 13.9 | Figure 13.4 |
| Figure 13.10 | Figure S13.6 |
| Table 14.1 | Table S14.1 |
| Table 14.2 | Table 14.1 |
| Table 14.3 | Table S14.3 |
| Table 14.4 | Table 14.2 |
| Table 14.5 | Table S14.11 |
| Table 14.6 | Table 14.3 |
| Figure 14.1 | Figure S14.1 |
| Figure 14.2 | Figure 14.1 |
| Figure 14.3 | Figure S14.5 |
| Figure 14.4 | Figure S14.2 |
| Figure 14.5 | Figure S14.3 |
| Figure 14.6 | Figure S14.4 |
| Figure 14.7 | Figure S14.6 |
| Figure 14.8 | Figure 14.2 |
| Figure 14.9 | Figure S14.7 |
| Figure 14.10 | Figure 14.3 |
| Figure 14.11 | Figure S14.8 |
| Figure 14.12 | Figure S14.9 |
| Section 15.2.5 | Section 15.2.4 |
| Section 15.2.6 | Section 15.2.5 |
| Section 15.3.3 | Section 15.3.2 |
| Table 15.1 | Table S15.2 |
| Table 15.2 | Table 15.1 |
| Table 15.3 | Table 15.2 |
| Figure 15.2 | Figure S15.1 |
| Figure 15.3 | Figure 15.2 |
| Figure 15.4 | Figure S15.2 |
| Figure 15.5 | Figure S15.3 |
| Table 16.3 | merged or removed |
| Table 16.4 | Table S16.5 |
| Table 16.5 | Table 16.3 |
| Table 16.6 | Table S16.2 |
| Table 16.7 | Table 16.4 |
| Table 16.8 | merged or removed |
| Table 16.9 | Table 16.5 |
| Table 16.10 | Table 16.6 |
| Figure 16.1 | Figure S16.3 |
| Figure 16.2 | Figure 16.1 |
| Figure 16.3 | Figure S16.2 |
| Figure 16.4 | Figure S16.4 |
| Box 16.1 | Box S16.1 |
| Figure 18.1 | merged or removed |
| Figure 18.2 | Figure 18.1 |
| Figure 18.3 | merged or removed |
| Section A.1.2 | Section A.1.1 |
| Section A.1.3 | Section A.1.2 |
| Section A.2.1 | Section A.2 (merged) |
| Section A.2.2 | Section A.2 (merged) |
| Section A.4 | Section A.3 (merged) |
| Section A.5 | Section A.3 (merged) |
| Section A.6 | Section A.3 (merged) |
| Section A.7 | Section A.4 (merged) |
| Section A.7.1 | Section A.4 (merged) |
| Section A.7.2 | Section A.4 (merged) |
| Section A.7.3 | Section SA.2 |
| Section A.7.4 | Section A.4 (merged) |
| Section A.8 | Section A.5 |
| Section A.8.1 | Section A.5.1 |
| Section A.8.2 | Section A.5.2 |
| Section A.8.3 | Section A.5.3 |
| Section A.9 | Section SA.6 |
| Section A.9.1 | Section A.3 (merged) |
| Section A.9.2 | Section A.3 (merged) |
| Section A.9.3 | Section A.3 (merged) |
| Section A.10 | merged or removed |
| Table A.1 | merged or removed |
| Table A.2 | Table A.1 |
| Section S3.2 | Section S3.5 |
| Section S5.4 | Section S5.6 |
| Section S6.2 | Section S6.3 |
| Section S6.3 | Section S6.4 |
| Table S6.3 | Table S6.4 |
| Section S6.4 | Section S6.6 |
| Table S6.4 | Table S6.5 |
| Section S6.5 | Section S6.7 |
| Table S6.5 | Table S6.6 |
| Section S6.6 | Section S6.8 |
| Table S6.6 | Table S6.7 |
| Section S6.7 | Section S6.9 |
| Table S6.7 | Table S6.8 |
| Table S6.8 | Table S6.9 |
| Table S6.9 | Table S6.10 |
| Table S6.10 | Table S6.11 |
| Table S6.11 | Table S6.12 |
| Table S6.12 | Table S6.13 |
| Table S6.13 | Table S6.14 |
| Table S6.14 | Table S6.15 |
| Table S7.1 | Table S7.2 |
| Table S7.2 | Table S7.3 |
| Table S7.3 | Table S7.4 |
| Table S7.4 | Table S7.6 |
| Section S8.3 | Section S8.5 |
| Table S10.3 | Table 10.2 |
| Table S12.1 | Table S12.2 |
| Figure S12.2 | Figure S12.3 |
| Section S12.2 | Section S12.3 |
| Table S12.2 | Table S12.3 |
| Figure S12.3 | Figure S12.4 |
| Section S12.3 | Section S12.4 |
| Table S12.3 | Table S12.4 |
| Figure S12.4 | Figure S12.5 |
| Section S12.4 | Section S12.5 |
| Table S12.4 | Table 12.2 |
| Section S12.5 | Section S12.6 |
| Section S12.6 | Section S12.7 |
| Section S12.7 | Section S12.8 |
| Section S12.8 | Section S12.9 |
| Section S12.9 | Section S12.10 |
| Section S12.9.1 | Section S12.10.1 |
| Section S12.9.2 | Section S12.10.2 |
| Section S12.10 | Section S12.11 |
| Section S12.10.1 | Section S12.11.1 |
| Section S12.10.2 | Section S12.11.2 |
| Section S12.10.3 | Section S12.11.3 |
| Section S12.10.4 | Section S12.11.4 |
| Section S12.10.5 | Section S12.11.5 |
| Section S13.3 | Section S13.4 |
| Table S13.3 | Table S13.4 |
| Section S13.4 | Section S13.5 |
| Table S13.4 | Table S13.5 |
| Section S13.5 | Section S13.6 |
| Table S13.5 | Table S13.6 |
| Section S13.6 | Section S13.7 |
| Table S13.6 | Table S13.7 |
| Section S13.7 | Section S13.8 |
| Table S13.7 | Table S13.8 |
| Section S13.8 | Section S13.9 |
| Table S13.8 | Table S13.9 |
| Section S13.9 | Section S13.1.1 |
| Table S13.9 | Table S13.10 |
| Section S13.10 | Section S13.2.1 |
| Table S13.10 | Table S13.12 |
| Section S13.11 | merged or removed |
| Table S13.11 | Table S13.15 |
| Table S13.12 | Table S13.3 |
| Table S13.13 | Table S13.11 |
| Table S13.14 | Table S13.13 |
| Table S13.15 | Table S13.14 |
| Table S14.1 | Table S14.2 |
| Table S14.2 | Table S14.4 |
| Table S14.3 | Table S14.5 |
| Table S14.4 | Table S14.6 |
| Table S14.5 | Table S14.7 |
| Table S14.6 | Table S14.8 |
| Table S14.7 | Table S14.9 |
| Table S14.8 | Table S14.10 |
| Figure S15.1 | Figure S15.4 |
| Figure S15.2 | Figure S15.5 |
| Table S15.2 | Table S15.3 |
| Section S15.3 | Section S15.4 |
| Table S15.3 | Table S15.4 |
| Section S15.4 | Section S15.5 |
| Table S15.4 | Table S15.5 |
| Section S15.5 | Section S15.6 |
| Table S15.5 | Table S15.6 |
| Section S15.6 | Section S15.7 |
| Section S15.7 | Section S15.8 |
| Section S15.8 | Section S15.10 |
| Table S16.2 | Table S16.3 |
| Section S16.3 | Section S16.4 |
| Table S16.3 | Table S16.4 |
| Section S16.4 | Section S16.6 |
| Section S16.5 | Section S16.7 |
| Section S16.6 | Section S16.8 |
| Section S16.7 | Section S16.9 |
| Chapter S19 | Chapter S0 |
| Section SA.2 | Section SA.4 |
| Section SA.3 | Section SA.5 |
| Section SA.4 | Section SA.7 |
