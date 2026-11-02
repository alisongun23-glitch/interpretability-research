# What the xor_circuits project found: gate-like circuits inside trained networks, and networks built from gate neurons

Scope: only the project's own files at `/Users/alisongun/Downloads/xor_circuits 2` (reports, summaries, plus code where a report was unclear). No training scripts were run. Every source link points to a local file. Evidence-type tags in brackets: **[acc]** end-to-end accuracy, **[probe]** correlational readout, **[abl]** ablation or knockout, **[patch]** activation patching or interchange (causal intervention), **[steer]** additive steering, **[transplant]** swapping parts between networks or between a network and a hand-built gate, **[sim]** representational similarity, **[struct]** structural count or size, **[stress]** robustness test of built machines.

## Q1. What are the project's main empirical findings, with their key numbers and evidence type?

### Takeaway
The project has two arcs. (1) **Analysis arc.** Trained networks (XOR MLPs, adder MLPs, tiny addition transformers) do contain causally used structure. It is gate-like only where the architecture or the data format forces it, and it is almost never the engineer's decomposition. The pre-registered transformer study found that the carry is a label-free direction, not a neuron. That direction moves outputs in 88–100% of interchanges but is far too noisy to act as a gate (margin 0.4–0.7 sd against 104 for a gate neuron). A later exploratory follow-up reframes it as a *count* (a_k + b_k + c_k) direction. (2) **Construction arc.** Single sigmoid neurons trained to margin 4 obey the static discipline. They compose with no rounding into a 37,521-gate Snake game and a 4,352-gate 8-bit CPU with zero wrong bits. That CPU, frozen and used as one attention head, lets a GPT reach 100% on held-out arithmetic, but only when REINFORCE also rewards the computer's own result.

### Cited Findings

#### A. Many small XOR networks (circuits, causal tests, transplants)
- Setup: 2→H→1 networks with a sigmoid output and BCE loss, trained by full-batch Adam (lr 0.05) from N(0, 1) weights. A run stops once all 4 inputs have |logit| ≥ 2.0 or fails after 20,000 steps. 30 seeds per architecture (tanh-4 baseline, relu-4, tanh-2, tanh-8, relu-8). Seeds 0–14 were used for discovery and the rest were held out — [results/REPORT.md](file:///Users/alisongun/Downloads/xor_circuits%202/results/REPORT.md)
- The report itself says that XOR needing at least two hidden units is "guaranteed by the architecture, not a discovery". The open questions are which parts, how consistent they are, and whether they can be swapped — [results/REPORT.md](file:///Users/alisongun/Downloads/xor_circuits%202/results/REPORT.md)
- Convergence [acc]: tanh-4 30/30, relu-4 23/30, tanh-2 9/30, tanh-8 30/30, relu-8 29/30. Median steps: 53, 65, 77, 36, 51 — [results/REPORT.md](file:///Users/alisongun/Downloads/xor_circuits%202/results/REPORT.md)
- Circuit types [abl-defined]. Every subset of hidden units was searched under mean ablation, and the primary circuit is the smallest minimal sufficient subset, typed by contribution labels at correlation ≥ 0.9:
  - tanh-4: `A AND NOT B + B AND NOT A` 70%, `NAND + OR` 17%, three minor types.
  - tanh-8: 77% / 13%. tanh-2: 67% / 33%.
  - relu-4 and relu-8 are fragmented: 12 types each, with the dominant type at only 30% and 38%.
  
  [results/REPORT.md](file:///Users/alisongun/Downloads/xor_circuits%202/results/REPORT.md)
- Redundancy [abl]:
  - Share of nets with more than one minimal circuit: tanh-4 67%, relu-4 17%, tanh-2 0%, tanh-8 100%, relu-8 69%.
  - Zero ablation gives the same sufficiency verdict as mean ablation in only 60% (tanh-4), 74%, 100%, 23% (tanh-8) and 66% of nets.
  
  [results/REPORT.md](file:///Users/alisongun/Downloads/xor_circuits%202/results/REPORT.md)
- Causal tests [abl, patch], tanh-4:
  - Ablating the primary circuit leaves 80% accuracy, against 77% for random same-size ablations. 50% means XOR is gone, because a constant output gets 2 of 4 right.
  - Patching the circuit makes the output follow the source 73% of the time, against 48% for a random set and 27% for all other units.
  - Other architectures: relu-4 94% / 45% / 6%; tanh-8 37% / 23% / 63%; relu-8 72% / 25% / 28%.
  
  [results/REPORT.md](file:///Users/alisongun/Downloads/xor_circuits%202/results/REPORT.md)
- Transplants without retraining [transplant]. The interface is the circuit's logit contribution. Results for tanh-4, 870 pairs:
  - Native 100%, ablation control 27%.
  - Direct 56%, aligned (moment-matched mean and spread) 99%.
  - Sham (the donor's same units at initialisation): direct 2%, aligned 10%.
  - Hand-built reference circuit: direct 27%, aligned 100%.
  - Direct transplant between nets with the same circuit type 96% (n = 442), between different types 14% (n = 428).
  - tanh-8 has an 80% ablation control, so its transplants are uninformative.
  - Across activation functions (tanh-4 ↔ relu-4, 1,380 pairs): direct 42%, aligned 95%, sham aligned 11%.
  
  [results/REPORT.md](file:///Users/alisongun/Downloads/xor_circuits%202/results/REPORT.md)
- Representational similarity [sim], tanh-4:
  - Linear CKA is 0.84 between trained nets and 0.75 between the same nets at initialisation.
  - Same circuit type 0.91, different type 0.77.
  - Primary-circuit logit contributions correlate at 0.91.
  - The report warns that with 4 inputs the representations have at most 3 dimensions, so CKA is coarse.
  
  [results/REPORT.md](file:///Users/alisongun/Downloads/xor_circuits%202/results/REPORT.md)
- Unit labels, trained vs untrained [struct]: NAND 7% vs 0%, A AND NOT B 12% vs 2%, B AND NOT A 18% vs 5%, NOT A 4% vs 15%, NOT B 2% vs 10% — [results/REPORT.md](file:///Users/alisongun/Downloads/xor_circuits%202/results/REPORT.md)
- Held-out confirmation:
  - The dominant type covers 53% of the 15 discovery nets and 87% of the 15 held-out nets.
  - On held-out nets, ablation leaves 78% (random 77%), direct transplant 71%, aligned 97%, sham aligned 8%.
  
  [results/REPORT.md](file:///Users/alisongun/Downloads/xor_circuits%202/results/REPORT.md)
- Hypothesis verdicts for tanh-4, with thresholds fixed in `xorlab/report.py`:
  - A, shared and interchangeable circuits: PARTLY SUPPORTED. 70% against a ≥80% rule; direct 56% against ≥80%.
  - B, different mechanisms with the same behaviour: SUPPORTED. 5 types, and same-type vs different-type transplant differs by +0.82.
  - C, shared features with different causal use: NOT SUPPORTED. CKA for different types is 0.77 against 0.75 untrained.
  - D, distributed or redundant computation: SUPPORTED. 67% have several minimal circuits, but 100% of nets still have at least one indispensable unit.
  
  [results/REPORT.md](file:///Users/alisongun/Downloads/xor_circuits%202/results/REPORT.md)

#### B. Tiling gates into a 4-bit adder vs an MLP trained on addition
- Tiling [acc]:
  - XOR, AND and OR gates (2→4→1 tanh) were trained with 20 seeds each at margins 2 and 6; all 120 converged.
  - 20 mixed-seed 4-bit ripple-carry adders were assembled.
  - With each gate's output snapped to 0/1: 100% of 256 sums at both margins.
  - Passing raw probabilities: margin 2 gives 99.5% on average (worst assembly 93.8%); margin 6 gives 100%.
  - At margin 2 the mean deviation grows from 0.06 at bit 0 to about 0.13 in the upper bits.
  
  [results_adder/ADDER_REPORT.md](file:///Users/alisongun/Downloads/xor_circuits%202/results_adder/ADDER_REPORT.md)
- A monolithic MLP (8→W→W→5 tanh, W = 16 and 64, 5 seeds each) was trained on 205 of the 256 problems [acc]. All reached 100% on training problems. On the 51 held-out problems, 9 of 10 nets scored 100% and one 96% — [results_adder/ADDER_REPORT.md](file:///Users/alisongun/Downloads/xor_circuits%202/results_adder/ADDER_REPORT.md)
- Probes [probe]:
  - Per-bit XOR probe: bit 0 is 100% trained vs 60–66% untrained (W = 16); bits 1–3 fall toward chance (bit 3: 53–65%).
  - The carry probe is 100%, but so is a probe on the raw inputs, so it "proves nothing".
  
  [results_adder/ADDER_REPORT.md](file:///Users/alisongun/Downloads/xor_circuits%202/results_adder/ADDER_REPORT.md)
- Knocking out the units most tied to each carry [abl] costs at most 9% accuracy on one bit, usually under 2% — [results_adder/ADDER_REPORT.md](file:///Users/alisongun/Downloads/xor_circuits%202/results_adder/ADDER_REPORT.md)
- Swapping in the opposite carry along the carry direction [patch]:
  - 0–5% of answers change as the tiled adder predicts; with the swap in layer 2, 100% stay unchanged.
  - Pushing 1.5×, 4× or 10× harder gives 0–6% [steer].
  - Verdict: "The carry direction exists but the network does not use it."
  
  [results_adder/ADDER_REPORT.md](file:///Users/alisongun/Downloads/xor_circuits%202/results_adder/ADDER_REPORT.md)
- Collapse to the total a + b [sim]:
  - η² at layer 2: 0.73 (W = 16) and 0.52 (W = 64), against 0.14 for raw inputs and 0.11–0.14 for untrained nets.
  - Same-total / random distance ratio at layer 2: 0.50 and 0.74, against about 1.0 untrained.
  - The collapse is "partial, not complete".
  
  [results_adder/ADDER_REPORT.md](file:///Users/alisongun/Downloads/xor_circuits%202/results_adder/ADDER_REPORT.md)
- Retracted claim: an earlier causal check replaced all of layer 2 with a same-total problem and got 99.8–100%. It was withdrawn as invalid because the network then just answers the other problem, whose answer is the same — [results_adder/ADDER_REPORT.md](file:///Users/alisongun/Downloads/xor_circuits%202/results_adder/ADDER_REPORT.md)

#### C. 16-bit addition MLPs
- Setup: 32 bits in, 17-bit sum out. Width 256, tanh, depth 2 and depth 4, 3 seeds each. 60,000 training problems, 40 epochs, tested on 100,000 fresh problems — [results_adder16/ADDER16_REPORT.md](file:///Users/alisongun/Downloads/xor_circuits%202/results_adder16/ADDER16_REPORT.md)
- Accuracy [acc, binned behaviour]:
  - 99.5–99.9% exact on random problems.
  - Exact accuracy by carry travel distance (the longest run of propagate positions one carry must cross):

    | travel distance | ≤ 7 | 8 | 10 | 12 | 13 | 15 |
    |---|---|---|---|---|---|---|
    | depth 2 | 97–100% | 91–94% | 69–76% | 42–49% | 27–37% | 15–28% |
    | depth 4 | 97–100% | 96–99% | 88–94% | 76–83% | 56–70% | 39–71% |

  - Confound: only 473 of 60,000 training problems (0.8%) have distance ≥ 8, and 15 have ≥ 12.
  
  [results_adder16/ADDER16_REPORT.md](file:///Users/alisongun/Downloads/xor_circuits%202/results_adder16/ADDER16_REPORT.md)
- Same-total distance ratio by layer [sim]: depth 4 goes 0.98 → 0.93–0.95 → 0.72–0.81 → **0.36–0.40**; depth 2 goes 0.97 → 0.85; untrained stays at 0.98–0.99 — [results_adder16/ADDER16_REPORT.md](file:///Users/alisongun/Downloads/xor_circuits%202/results_adder16/ADDER16_REPORT.md)
- Per-position XOR probe [probe]:
  - Raw inputs 47–52%, untrained nets about 70%.
  - Trained nets are near 100% at bits 0–1. In depth 4, layer 2: bit 6 97%, bits 9–10 86%. Most other positions are near or below the untrained baseline.
  
  [results_adder16/ADDER16_REPORT.md](file:///Users/alisongun/Downloads/xor_circuits%202/results_adder16/ADDER16_REPORT.md)
- Carry coordinate copy for k = 4, 8, 12 [patch]:
  - The answer moved by exactly 2^k in 0–4% of problems (random direction 0%), and 95–100% of answers were unchanged in most layers.
  - Exception: depth 4, layer 2, bits 8 and 12. Only 8–33% stayed unchanged there, but the carry-flipped answer did not appear.
  
  [results_adder16/ADDER16_REPORT.md](file:///Users/alisongun/Downloads/xor_circuits%202/results_adder16/ADDER16_REPORT.md)
- Correction: an earlier version binned problems by carry-chain length, which does not predict failure. Depth-2 nets score 95% on random problems with chain length 16 — [results_adder16/ADDER16_REPORT.md](file:///Users/alisongun/Downloads/xor_circuits%202/results_adder16/ADDER16_REPORT.md)

#### D. Small GPT on binary addition (first transformer study)
- Setup: 10-bit binary addition with a tiny pre-LN GPT-2 (d_model 64, 4 heads of 16, MLP 256). Three models: lsb-L1, lsb-L2, msb-L2. 2,500 steps of 128 problems, one seed each. The NumPy GPT's gradients were checked against finite differences, with worst relative error 6e-7 — [results_transformer/TRANSFORMER_REPORT.md](file:///Users/alisongun/Downloads/xor_circuits%202/results_transformer/TRANSFORMER_REPORT.md)
- Accuracy [acc]:
  - lsb-L1 and lsb-L2: 100% at every travel distance.
  - msb-L2: 96.7–97.2%. Its errors sit in the middle bits (bit 3 wrong on 3.0%) and are worst (92–93%) on problems with no propagate positions.
  
  [results_transformer/TRANSFORMER_REPORT.md](file:///Users/alisongun/Downloads/xor_circuits%202/results_transformer/TRANSFORMER_REPORT.md)
- Activation patching of resid_pre at one position [patch]:
  - lsb-L1, layer 0: s_{k−1} 0.35–0.38, a_{k−1} 0.31–0.38, b_{k−1} 0.14–0.33; every other position 0.00.
  - lsb-L2, layer 1: the query position flips 1.00.
  - msb-L2 reads the carry backwards from sum digits it has already emitted: s_{k+1} flips 0.25–0.61 at layer 1.
  
  [results_transformer/TRANSFORMER_REPORT.md](file:///Users/alisongun/Downloads/xor_circuits%202/results_transformer/TRANSFORMER_REPORT.md)
- Probes [probe]: trained and untrained models score the same (0.71–0.75 at layer 0, 0.95–1.00 at the final layer). The report's summary is "Presence is not use." A later report contradicts this (see section L) — [results_transformer/TRANSFORMER_REPORT.md](file:///Users/alisongun/Downloads/xor_circuits%202/results_transformer/TRANSFORMER_REPORT.md)
- Head ablations [abl]:
  - lsb-L1: L0H3 → 0%, L0H1 → 81%, L0H0 100%, L0H2 99%.
  - lsb-L2: L0H3 → 0%, L0H0 → 59%, all four layer-1 heads 100%.
  - msb-L2 needs all eight heads: L0H2 → 0.4%, while the mildest, L1H0, still drops to 55%.
  
  [results_transformer/TRANSFORMER_REPORT.md](file:///Users/alisongun/Downloads/xor_circuits%202/results_transformer/TRANSFORMER_REPORT.md)

#### E. Pre-registered study: is the transformer's carry a gate neuron?
- The pre-registration is timestamped "2026-10-14 21:38 EDT, before any model in this study was trained or analysed". It lists 14 predictions (P0–P8 with sub-items) and fixes every measure in advance — [results_transformer_gates/PREREG.md](file:///Users/alisongun/Downloads/xor_circuits%202/results_transformer_gates/PREREG.md)
- Models:
  - Family B: 1 layer, d_model 64, 4 heads, d_mlp 256, learned positions, seeds 0–4 (seed 1 = lsb-L1 from study D), format `a0..a9 + b0..b9 = s0..s10`.
  - Family C2: no position embedding, a learned per-head relative-offset bias, interleaved `a0 b0 s0 a1 b1 s1 …`, seeds 0–2.
  - Over the previous digit the carry is the single threshold function c_k = [a_{k−1} + b_{k−1} − s_{k−1} ≥ 1].
  
  [results_transformer_gates/REPORT.md](file:///Users/alisongun/Downloads/xor_circuits%202/results_transformer_gates/REPORT.md)
- P0 [acc]: every model is exact at 10 bits (99.95–100%). C2 fails on longer sums: 12 bits 76.7–81.1%, 16 bits 21.1–30.8%, 20 bits 3.5–7.2%. The cause is offsets above 31 that never get a gradient — [results_transformer_gates/REPORT.md](file:///Users/alisongun/Downloads/xor_circuits%202/results_transformer_gates/REPORT.md)
- Attention (P1 fail):
  - In B, one head reads a_k and b_k at 0.22–0.36 each, but a_{k−1} and b_{k−1} get only 0.12–0.21.
  - The models average a and b with equal weight, so attention delivers sums of bits, not bits. Individual previous bits are only 69–78% readable while the carry is 92–95% readable.
  
  [results_transformer_gates/REPORT.md](file:///Users/alisongun/Downloads/xor_circuits%202/results_transformer_gates/REPORT.md)
- Probes at resid_mid (P2, P3 fail) [probe]:
  - B: 91.8–95.2% trained vs 78–79% random-init. C2: 73–85% vs 59%.
  - Per k, B reaches 98–100% (median 100%).
  - Shuffled labels give 55.5–55.8% against a majority rate of 55.2%.
  - So readability *does* separate trained from untrained models, contrary to P3.
  
  [results_transformer_gates/REPORT.md](file:///Users/alisongun/Downloads/xor_circuits%202/results_transformer_gates/REPORT.md)
- No carry neuron (P4a pass) [probe / struct]:
  - The best MLP unit fits the carry table with R² 0.30–0.40.
  - A one-neuron carry gate trained the library way fits it with R² 1.000. Its weights are (8.5, 8.4, −8.1), bias −4.4, and its two levels are 104 within-level standard deviations apart.
  - The best units' level gap is only 1.4–2.1 sd. The library neurons' gaps are 122 / 233 / 123 / 238 / 158 (AND / OR / NAND / NOR / XOR).
  - No subset of up to 40 units gives a 99% linear readout of c_k (best 88–96%). The sum s_k needs only 3–4 units in four B models (20 in B:3) and 6–9 in C2.
  
  [results_transformer_gates/REPORT.md](file:///Users/alisongun/Downloads/xor_circuits%202/results_transformer_gates/REPORT.md)
- Count-threshold MLP (P4b fail, but a positive structural finding):
  - In B:0, B:1, B:2 and B:4, units with R² ≥ 0.8 on (a_k, b_k, c_k) number 96, 100, 98 and 78. 100% of them are count thresholds: OR3 25–40, majority 31–41, AND3 14–31.
  - This is the textbook XOR3 = [n ≥ 1] − [n ≥ 2] + [n ≥ 3].
  - B:3 has no clean units. C2 has 33, 40 and 7 such units, of which 64%, 3% and 0% are thresholds; the rest are mostly single-combination detectors.
  
  [results_transformer_gates/REPORT.md](file:///Users/alisongun/Downloads/xor_circuits%202/results_transformer_gates/REPORT.md)
- Interchange along a label-free direction (P5 fails only on its control clause) [patch]. PC1 of base−source differences, at resid_mid, at q_k:
  - Success: B 97.4–100%, C2 88.3–98.9%. PC1–4: 99.97–100%.
  - The label-trained probe direction gives 0.0% (C2:2 1.9%). Random directions average ≤ 1% (max 4.9%).
  - The random-init model's own PC1 gives 9.1–39.9% in five of eight models. That breaks P5's third clause, and the report reads it as "does PC1 matter", not "is PC1 the carry".
  - Probe vs PC1 cosine is 0.02 (B:1). Base and source differ by 0.01 along the probe and 0.5 along PC1.
  - After the patch, the whole generated sum matches the carry-flipped adder only 67–88% (B) and 47–49% (C2).
  
  [results_transformer_gates/REPORT.md](file:///Users/alisongun/Downloads/xor_circuits%202/results_transformer_gates/REPORT.md)
- Writing the exact gate carry in (P6 fail) [transplant / abl]:
  - Replacing PC1–4 with the gate adder's carry, written as a class mean, breaks every model (≤ 0.2% exact). Projecting the subspace out gives 0.0–0.2%.
  - Exploratory diagnosis: within-class spread along PC1 is 0.5–0.9× the class gap. Pinning to the true class mean leaves the bit right only 45–58% of the time. Shifting by the class gap flips 93–100%.
  - The report concludes "The model reads its carry *relative* to the rest of the residual."
  
  [results_transformer_gates/REPORT.md](file:///Users/alisongun/Downloads/xor_circuits%202/results_transformer_gates/REPORT.md)
- The transformer's carry as a gate (P7a, P7b fail; P7c sanity pass) [transplant]:
  - Gate margins are 0.41–0.70, against ≥ 4 required.
  - Snapped at 0.5, every table equals the carry table.
  - The trilinear gate's worst deviation at ε = 0.05 is 0.25–0.35, so it does not restore. PREREG had noted that even a perfect table cannot restore here.
  - Yet an analog 64-bit ripple adder that uses the transformer gate, with library XOR neurons restoring every sum bit, adds 100% of 100,000 sums for all five B models. C2 gets 94.9%, 97.7% and 69.1%.
  
  [results_transformer_gates/REPORT.md](file:///Users/alisongun/Downloads/xor_circuits%202/results_transformer_gates/REPORT.md); [results_transformer_gates/PREREG.md](file:///Users/alisongun/Downloads/xor_circuits%202/results_transformer_gates/PREREG.md)
- The MLP replaced by a hand-built library-XOR module (P8 fail) [transplant]:
  - B: 68.8 / 84.6 / 70.8 / 96.0 / 82.5%. C2: 5.9 / 1.6 / 6.2%.
  - Writing a random direction gives 0–11.6%. Wiring a_{k−1} in place of c_k gives 5.2–6.4% in B.
  - Given the *true* s_k, the module was exact. The failures come from the read side: a 10-bit sum needs about 30 snapped probe readings to all be right.
  
  [results_transformer_gates/REPORT.md](file:///Users/alisongun/Downloads/xor_circuits%202/results_transformer_gates/REPORT.md)

#### F. Exploratory follow-up: PC1 is part of a count, not a pure carry
- `followup_tests.py` describes itself as "Follow-up tests for 'The Carry Is Part of a Count, Not a Gate' (EXPLORATORY: run after the paper, not pre-registered)". It reuses the same models, directions and data seed — [followup_tests.py](file:///Users/alisongun/Downloads/xor_circuits%202/followup_tests.py)
- T1, what PC1 encodes. OLS of the PC1 coordinate, coefficients relative to c_k:
  - B: a_k +0.94 to +1.71, b_k +0.92 to +1.71, a_{k−1} and b_{k−1} +0.11 to +0.49, s_{k−1} −0.08 to −0.30.
  - R² is 0.70–0.97 for the full fit, but only 0.32–0.39 for c_k alone.
  - |cos(PC1, token-embedding difference)| is 0.81–0.87 in B and 0.05–0.39 in C2.
  - In C2:1 and C2:2 the b_k coefficient is negative (−0.85, −0.58).
  
  [results_followup/SUMMARY.md](file:///Users/alisongun/Downloads/xor_circuits%202/results_followup/SUMMARY.md)
- T2, context-conditioned transplant [transplant]. PC1 is set to the training mean for (k, a_k, b_k, carry) instead of (k, carry):
  - B:0, B:1, B:2 and B:4 reach 100% teacher-forced and 100% exact in full generation.
  - B:3: 88.7% teacher-forced, 21.7% in generation (PC1).
  - C2: 98.3–98.8% teacher-forced, 86.9–89.5% in generation (PC1).
  - Carry-only means give 54.8–57.7%, which is chance.
  - This undoes the P6 failure.
  
  [results_followup/SUMMARY.md](file:///Users/alisongun/Downloads/xor_circuits%202/results_followup/SUMMARY.md)
- T3, steering carry-0 rows by m × the class gap [steer]:
  - Along PC1, ×1 flips 92.2–100%, but ×2 flips only 31.7–47.5% and ×4 / ×8 about 43–57%. The response is non-monotone.
  - Along the probe direction: ×1 0–2.2%, ×8 0–18.3%.
  - Along a random direction: ×1 0.1–13.4%, ×8 44.5–57.6%.
  - Gap along the probe ÷ gap along PC1 is 0.02–0.15.
  
  [results_followup/SUMMARY.md](file:///Users/alisongun/Downloads/xor_circuits%202/results_followup/SUMMARY.md)
- T4, carry or answer? [patch] When sources also differ in a_k and b_k, the PC1 interchange output follows the *source's answer* 77.7–90.5% (B) and 68.7–85.7% (C2). It follows the carry only 9.5–31.3% — [results_followup/SUMMARY.md](file:///Users/alisongun/Downloads/xor_circuits%202/results_followup/SUMMARY.md)
- T5, token confound [patch]: on B pairs that also share s_{k−1}, success stays at 97.7–100%. The query token does not explain the interchange — [results_followup/SUMMARY.md](file:///Users/alisongun/Downloads/xor_circuits%202/results_followup/SUMMARY.md)
- T6, gate margin after removing each (a_k, b_k) group's mean: B rises from 0.53–0.70 to 2.79–3.54 (B:3 0.91), C2 to 0.86–1.45. Still below the gate criterion of 4 — [results_followup/SUMMARY.md](file:///Users/alisongun/Downloads/xor_circuits%202/results_followup/SUMMARY.md)
- T7 and T9, MLP units:
  - In B (except B:3), 82–102 units have η²(a_k, b_k, c_k) ≥ 0.8, and for 100% of them the count explains ≥ 95%. C2 count share is 0–56%.
  - Adding the bit position k raises the median unit η² to 0.95–0.97 (B:0, 1, 2, 4), 0.84 (B:3) and 0.69–0.84 (C2).
  
  [results_followup/SUMMARY.md](file:///Users/alisongun/Downloads/xor_circuits%202/results_followup/SUMMARY.md)
- T8, later bits after a teacher-forced flipped s_k:
  - In B, the next bit follows the local rule 63.0–85.5% of the time.
  - In C2 it follows the local rule only 14.4–22.9% and the *original* carry 77.1–85.6%, so C2 carries information from further back.
  
  [results_followup/SUMMARY.md](file:///Users/alisongun/Downloads/xor_circuits%202/results_followup/SUMMARY.md)

#### G. The smallest working neural gates (static discipline)
- Smallest architecture that learns each gate [acc]. 100 seeds each, stopping at |logit| ≥ 4:
  - AND, OR, NAND and NOR are one sigmoid neuron (3 weights, 100%). NOT is one neuron with 2 weights.
  - XOR and XNOR fail as one neuron and as a 2→1→1 chain (0%). They need 1 hidden neuron plus a skip connection to the output (7 weights), which trains 100% of the time, against 57% / 52% for 2→2→1 and 94% for 2→4→1.
  
  [results_minigates/MINIGATES_REPORT.md](file:///Users/alisongun/Downloads/xor_circuits%202/results_minigates/MINIGATES_REPORT.md)
- Static discipline at ε = 0.05: every input within ε of a rail must give an output within ε of the correct rail.

  | training margin | 2 | 3 | 3.5 | 4 | 6 |
  |---|---|---|---|---|---|
  | NOT, seeds passing | 0% | 0% | 100% | 100% | 100% |
  | every 2-input gate, seeds passing | 0% | 0% | 0% | 100% | 100% |
  | largest NAND weight | 6.3 | 9.3 | 10.8 | 12.3 | 18.3 |

  - The prespecified rule picks **margin 4**.
  - Working ε range at margin 4: 0.03–0.12 for 2-input gates (0.04–0.10 for OR and NOR), and 0.03–0.45 for NOT.
  
  [results_minigates/MINIGATES_REPORT.md](file:///Users/alisongun/Downloads/xor_circuits%202/results_minigates/MINIGATES_REPORT.md)
- Mechanism: a sigmoid's slope at threshold is |w|/4, and a gate restores only if this gain is above 1. At margin 2 the NOT weight is about 4 (gain 1), so its outputs reach only 0.12 / 0.88. At margin 4 the weight is about 8 (gain 2) — [results_minigates/MINIGATES_REPORT.md](file:///Users/alisongun/Downloads/xor_circuits%202/results_minigates/MINIGATES_REPORT.md)
- Fewest bits per weight that keep margin-4 gates passing: AND and NAND 3 bits, NOT 4, OR and NOR 4–5, XOR 5–7, XNOR 6–7 — [results_minigates/MINIGATES_REPORT.md](file:///Users/alisongun/Downloads/xor_circuits%202/results_minigates/MINIGATES_REPORT.md)
- Stress tests [stress]:
  - 400-layer random circuits (128 wires): margin 2 gives 21% wrong signals at layer 10 and 27% at layer 400. Margins 3 and 4 give 0%, with worst distance from the rail 0.14 and 0.027.
  - NAND latch held for 2,000 steps: margin 2 collapses to 0.56 / 0.44; margin 3 holds 0.93 / 0.07; margin 4 holds 0.98 / 0.02.
  
  [results_minigates/MINIGATES_REPORT.md](file:///Users/alisongun/Downloads/xor_circuits%202/results_minigates/MINIGATES_REPORT.md)

#### H. Snake game built from gate neurons
- Size, 16 × 16 board [struct]:
  - 37,521 gates, 18,801 of them memory (2,350 flip-flops of 8 NANDs each). 39,863 neurons and 119,597 weights.
  - Logic depth 48 gates or 51 neuron layers.
  - 79,734 multiply-adds per tick evaluating every gate once, 493,333 as simulated.
  - NAND-only version: 50,587 gates, depth 82, and only 24 distinct trained weights in the whole machine.
  - With the 2→4→1 adder gates the same netlist needs 178,269 neurons and 602,847 weights.
  
  [results_snake/SNAKE_REPORT.md](file:///Users/alisongun/Downloads/xor_circuits%202/results_snake/SNAKE_REPORT.md)
- Correctness [acc, stress]:
  - 0 wrong bits against a Python reference on every tick: 12,000 ticks on 8 × 8, 9,000 on 16 × 16, 400 on 32 × 32. Worst signal 0.031 (NAND-only 0.026).
  - A board-filling run lasted 18,300 ticks, ate 252 foods and grew the snake to 255 cells, still with 0 wrong bits.
  
  [results_snake/SNAKE_REPORT.md](file:///Users/alisongun/Downloads/xor_circuits%202/results_snake/SNAKE_REPORT.md)
- Where the gates go: cells (countdowns and occupancy) take 93% of gates, and half of all gates are flip-flops. Everything else, the game rules included, is 7% — [results_snake/SNAKE_REPORT.md](file:///Users/alisongun/Downloads/xor_circuits%202/results_snake/SNAKE_REPORT.md)
- Margin sweep: margins 1.5–3.25 fail on tick 1. Margin 3.5 survives 1,500 ticks (worst signal 0.205). Margin 4 survives with worst signal 0.031 — [results_snake/SNAKE_REPORT.md](file:///Users/alisongun/Downloads/xor_circuits%202/results_snake/SNAKE_REPORT.md)
- Noise added to every neuron output [stress]: no errors up to σ = 0.03; first errors after 27–147 ticks at σ = 0.04, and after 1–7 ticks at σ = 0.05 — [results_snake/SNAKE_REPORT.md](file:///Users/alisongun/Downloads/xor_circuits%202/results_snake/SNAKE_REPORT.md)
- Cost: 2.5 ms per tick in NumPy against 0.012 ms for plain Python (about 200×); about 1.3 ms in the browser — [results_snake/SNAKE_REPORT.md](file:///Users/alisongun/Downloads/xor_circuits%202/results_snake/SNAKE_REPORT.md)

#### I. 8-bit CPU built from gate neurons
- Size [struct]:
  - 4,352 gates, 2,433 of them memory (304 flip-flops). 4,389 neurons, 13,141 weights.
  - Depth 29 gates or 32 neuron layers.
  - Memory (32 bytes of RAM) takes 82%. Control unit plus ALU take about 500 gates (287 + 212).
  - SAP-1 style, with 8 instructions (LDA, ADD, SUB, NAND, STA, JMP, JZ, JC) and about 4 ticks per instruction.
  
  [results_cpu/CPU_REPORT.md](file:///Users/alisongun/Downloads/xor_circuits%202/results_cpu/CPU_REPORT.md)
- Correctness [acc]: 15,332 ticks of demos and random-byte programs with 0 wrong bits in every configuration. Fully neural worst signal is 0.030 (NAND-only 0.026) — [results_cpu/CPU_REPORT.md](file:///Users/alisongun/Downloads/xor_circuits%202/results_cpu/CPU_REPORT.md)
- Margin sweep:
  - Margins 2 and 2.5 fail on tick 1; margin 3 fails on tick 34.
  - Margin 3.25 works with worst signal 0.33; 3.5 gives 0.10; 4 gives 0.030.
  - The CPU tolerates smaller margins than the snake because its longest OR tree has 5 levels against the snake's 8.
  
  [results_cpu/CPU_REPORT.md](file:///Users/alisongun/Downloads/xor_circuits%202/results_cpu/CPU_REPORT.md)
- Noise [stress]: no errors up to σ = 0.03. At σ = 0.04, 1 of 3 runs failed (tick 257). At σ = 0.05 all runs failed within 53–225 ticks — [results_cpu/CPU_REPORT.md](file:///Users/alisongun/Downloads/xor_circuits%202/results_cpu/CPU_REPORT.md)
- Cost: 0.72 ms per tick against 0.003 ms for plain Python (about 250×) — [results_cpu/CPU_REPORT.md](file:///Users/alisongun/Downloads/xor_circuits%202/results_cpu/CPU_REPORT.md)

#### J. CPU-GPT: the frozen gate CPU as the fourth attention head
- Setup:
  - 2-layer PyTorch GPT: d_model 96, 4 heads of 24, MLP 384, 277 tokens (each number 0–255 is one token).
  - Baseline: 278,005 weights. CPU-head model: 283,646 weights, 14,929 of them for the head's pointers, gate and read-back. The frozen computer adds 4,389 neurons that are not counted.
  - Pointers (A, B, OP) and the call gate learn by REINFORCE with a leave-one-out baseline. Everything else learns by backprop.
  - Results come from a precomputed table of all 262,144 calls, run on the gate netlist (bit-sliced) in 11.5 s.
  
  [results_cpugpt/CPUGPT_REPORT.md](file:///Users/alisongun/Downloads/xor_circuits%202/results_cpugpt/CPUGPT_REPORT.md)
- Held-out operand pairs [acc]:
  - Baseline: add 2.3–3.4%, sub 1.9–10.4%, mul 18.0–18.4%, but 96–100% on training pairs (memorisation).
  - CPU head with answer-likelihood reward only: pointers right 0–71%, best add 67.5%.
  - CPU head + result check: seed 0 gets 100% on all three operations; seeds 1 and 2 get 100% / 100% / 86.5–86.8%.
  - Pricier calls (0.05): seed 0 is 100% everywhere and calls on 100% of math and 0% of non-math problems.
  
  [results_cpugpt/CPUGPT_REPORT.md](file:///Users/alisongun/Downloads/xor_circuits%202/results_cpugpt/CPUGPT_REPORT.md)
- Head ablation and operation swap [abl / patch]:
  - Switching the head off in the best models drops held-out accuracy to 0% (add), 1.6–1.9% (sub) and 8.1% (mul).
  - Swapping the operation the computer receives gives 0–3.6%.
  
  [results_cpugpt/CPUGPT_REPORT.md](file:///Users/alisongun/Downloads/xor_circuits%202/results_cpugpt/CPUGPT_REPORT.md)
- Analog check: 500 test problems were re-run on the fully analog neural CPU. 0 results differed, calls averaged 113 ticks, and the worst neuron output was 0.030 from a rail — [results_cpugpt/CPUGPT_REPORT.md](file:///Users/alisongun/Downloads/xor_circuits%202/results_cpugpt/CPUGPT_REPORT.md)
- Cost: a GPT forward pass is about 1.1 M multiply-adds. A call is 32 ticks (add or sub) or 54 + 32·y ticks (multiply), 113 on average, and takes about 0.08 s on the analog NumPy CPU — [results_cpugpt/CPUGPT_REPORT.md](file:///Users/alisongun/Downloads/xor_circuits%202/results_cpugpt/CPUGPT_REPORT.md)

#### K. Story GPT that routes math and word problems to the CPU head
- Setup: 4 layers, d_model 128, 4 heads, MLP 512, 64-token context, 7,000 steps of 32 windows on TinyStories, vocabulary 4,279. The training stream holds 5.16 M tokens, with 41,417 template problems and 31,433 word problems. The report calls it a "1.4 M-weight model" — [results_storygpt/STORYGPT_REPORT.md](file:///Users/alisongun/Downloads/xor_circuits%202/results_storygpt/STORYGPT_REPORT.md)
- Held-out word problems [acc]:
  - CPU head (+ embedding read-back), seeds 0 / 1: + 99.5 / 89.7%, − 90.5 / 99.6%, × 96.7 / 97.5%.
  - Copy read-back: 99–100% on + and −, 95.1 / 99.0% on ×.
  - Plain GPT: 0% / 2.0% / 11.5%. Version 1 models: 0% (they never call on these sentences).
  
  [results_storygpt/STORYGPT_REPORT.md](file:///Users/alisongun/Downloads/xor_circuits%202/results_storygpt/STORYGPT_REPORT.md)
- Pointers and calls [struct / acc]:
  - All three pointers land right in 90.3–100% of held-out word problems.
  - Across 207,168 held-out story words, which contain 629 operation words, there are 0 calls.
  - Switching the head off drops word problems to 0% / 1–2% / 7–9%. Swapping the operation drops them below 3%.
  
  [results_storygpt/STORYGPT_REPORT.md](file:///Users/alisongun/Downloads/xor_circuits%202/results_storygpt/STORYGPT_REPORT.md)
- Sentence shapes never seen in training mostly fail (0–38%). The gate rarely fires for − and ×, even when all three pointers are right (97–99% for − in two models) — [results_storygpt/STORYGPT_REPORT.md](file:///Users/alisongun/Downloads/xor_circuits%202/results_storygpt/STORYGPT_REPORT.md)
- Perplexity rose from 13.7–13.9 (version 1) to 14.4–14.5. A plain GPT trained to see as many story words as version 1 is back at 13.89. CPU-head models sit 0.12–0.23 above the plain GPT, which "may be seed noise" (one plain seed) — [results_storygpt/STORYGPT_REPORT.md](file:///Users/alisongun/Downloads/xor_circuits%202/results_storygpt/STORYGPT_REPORT.md)

#### L. Inconsistencies and corrections across reports (flagged for the writer)
- **Probes in transformers.** The first study found trained = untrained probe accuracy ("Probes are uninformative here") — [results_transformer/TRANSFORMER_REPORT.md](file:///Users/alisongun/Downloads/xor_circuits%202/results_transformer/TRANSFORMER_REPORT.md). The later study contradicts it: at resid_mid, random-init models read only 78–79% (B) and 59% (C2) against 92–95% trained. It explains that "the earlier report ... probed only resid_pre and the final layer with small least-squares probes" — [results_transformer_gates/REPORT.md](file:///Users/alisongun/Downloads/xor_circuits%202/results_transformer_gates/REPORT.md)
- **Is PC1 "the carry"?** The pre-registered report's headline is "The carry is a direction in the residual stream", and it explains the class-mean failure as relative readout — [results_transformer_gates/REPORT.md](file:///Users/alisongun/Downloads/xor_circuits%202/results_transformer_gates/REPORT.md). The follow-up refines this: PC1 loads about equally on a_k and b_k (T1), follows the source's *answer* rather than its carry on cross pairs (T4), and transplants work once means are conditioned on (a_k, b_k) (T2) — [results_followup/SUMMARY.md](file:///Users/alisongun/Downloads/xor_circuits%202/results_followup/SUMMARY.md)
- **How many pre-registered predictions failed.** The plain-English summary says "Six predictions fail: P1, P2, P3, P5, P6 and P8." The scorecard table in the same file also marks P4b, P7a and P7b as fail, which makes 9 failures, with 3 passes (P0, P4a, P7c sanity) and 2 vacuous passes — [results_transformer_gates/REPORT.md](file:///Users/alisongun/Downloads/xor_circuits%202/results_transformer_gates/REPORT.md)
- **TransformerLens replication.** The first transformer report says `run_transformer_tl.py` "has not been run by the author because PyTorch could not be installed" — [results_transformer/TRANSFORMER_REPORT.md](file:///Users/alisongun/Downloads/xor_circuits%202/results_transformer/TRANSFORMER_REPORT.md). Yet `results_transformer_tl/` holds .pt checkpoints and a summary.json dated 14 Oct, after the report (12 Oct). No report describes these results. They differ from the NumPy run: msb-L2 random_exact is 0.8953 (against 96.7–97.2%), and lsb-L1 head ablations give L0H0 66.3%, L0H1 11.2%, L0H2 0.8%, L0H3 5.2% (against 100% / 81% / 99% / 0%) — [results_transformer_tl/summary.json](file:///Users/alisongun/Downloads/xor_circuits%202/results_transformer_tl/summary.json)
- **Baseline range in CPU-GPT.** The opening says a same-size GPT without the computer "gets 2–34%", but the results table shows 1.9–18.4% on held-out math — [results_cpugpt/CPUGPT_REPORT.md](file:///Users/alisongun/Downloads/xor_circuits%202/results_cpugpt/CPUGPT_REPORT.md)
- **"Margin 2" means different things.** The minigates report says margin-2 gates are "the size used in the adder experiments" and cannot store a bit — [results_minigates/MINIGATES_REPORT.md](file:///Users/alisongun/Downloads/xor_circuits%202/results_minigates/MINIGATES_REPORT.md). But the adder gates were 2→4→1 networks with a *tanh* hidden layer (trained with xorlab's `train_batch`, per `run_adder.py`), while the latch test used single-sigmoid minigates. The shared quantity is the logit margin, not the architecture — [results_adder/ADDER_REPORT.md](file:///Users/alisongun/Downloads/xor_circuits%202/results_adder/ADDER_REPORT.md)
- **Runtimes.** README and reports disagree slightly: adder about 20 s vs 30 s, minigates about 1 min vs 45 s — [README.md](file:///Users/alisongun/Downloads/xor_circuits%202/README.md); [results_adder/ADDER_REPORT.md](file:///Users/alisongun/Downloads/xor_circuits%202/results_adder/ADDER_REPORT.md); [results_minigates/MINIGATES_REPORT.md](file:///Users/alisongun/Downloads/xor_circuits%202/results_minigates/MINIGATES_REPORT.md)
- **Self-corrections already in the reports.** The adder report withdrew its invalid "same-total layer swap" causal check, and the 16-bit report replaced its chain-length binning with travel distance — [results_adder/ADDER_REPORT.md](file:///Users/alisongun/Downloads/xor_circuits%202/results_adder/ADDER_REPORT.md); [results_adder16/ADDER16_REPORT.md](file:///Users/alisongun/Downloads/xor_circuits%202/results_adder16/ADDER16_REPORT.md)

### Inferences
- The recurring empirical pattern is that decodability does not predict use. This is near-total in some cases:
  - The 4-bit carry probe is 100% even on raw inputs, while the carry direction has 0–5% causal effect.
  - The transformer's logistic-probe direction has 0% interchange effect, while the label-free PC1 has 88–100%.
  
  The writer should treat correlational results in this project as weak, and the interchange and transplant numbers as the substantive evidence.
- The follow-up (T1, T2, T4) is the most recent word on the transformer carry. The pre-registered interchange pairs share every bit ≥ k, so a_k and b_k are fixed and carry and count cannot be told apart. On cross pairs, PC1 behaves like an answer or count coordinate. Any outside comparison should use the "count direction" framing, not "carry direction".
- The undocumented TransformerLens run suggests circuit details (which heads matter, msb accuracy) vary with initialisation and library. This matches the XOR finding of many circuits for one function. It is unreported, so it should be cited only as "unreported data".

### Gaps
- `followup_tests.py` refers to a paper titled "The Carry Is Part of a Count, Not a Gate", but no draft, .tex or paper file was found in the repository. Its claims and wording could not be checked.
- The TransformerLens results (`results_transformer_tl/summary.json`) have no accompanying report, so how they were produced and how the author interprets them are unknown.
- Error bars or CIs are missing for many headline numbers: XOR, adder and snake are deterministic or single runs; the transformer study D used one seed per model.

## Q2. Where does the project find gate-like or circuit-like structure in trained networks, and where does it explicitly fail to find it?

### Takeaway
Gate-like structure appears in three places:
- tiny XOR nets, where the architecture forces a two-unit decomposition;
- lowest-bit-first transformers, where the format makes the carry local and a two-head circuit is pinned to three positions;
- the B-family MLP, whose count-threshold units (OR3, MAJ, AND3) build parity the textbook way.

The project explicitly fails to find:
- the tiled adder's parts in an MLP trained on addition (4-bit: carry direction present but unused, with a collapse to the total instead; 16-bit: carry-like behaviour without a usable carry direction);
- any carry *neuron* in transformers (best R² 0.30–0.40);
- a carry *gate* (the PC1 margin is 0.4–0.7 sd, it does not restore, and the clean carry cannot be written in without context).

### Cited Findings
**Found (gate-like or circuit-like):**
- XOR nets: about 70–77% of tanh nets use `A AND NOT B + B AND NOT A` and 13–33% use `NAND + OR`. Trained nets have more gate-labelled units than at initialisation (e.g. NAND 7% vs 0%). The decomposition itself is "guaranteed by the architecture" — [results/REPORT.md](file:///Users/alisongun/Downloads/xor_circuits%202/results/REPORT.md)
- Same-type XOR circuits are interchangeable (96% direct transplant in tanh-4), and any circuit transplants after moment alignment (99%). Different-type circuits are not (14% direct) — [results/REPORT.md](file:///Users/alisongun/Downloads/xor_circuits%202/results/REPORT.md)
- The lowest-first transformer "implement[s] exactly the local rule": the only inputs to bit k are the previous digit's a, b and sum, each necessary and none sufficient alone. This shows as a crisp two-head circuit (L0H3 current digits, L0H1 previous sum digits) — [results_transformer/TRANSFORMER_REPORT.md](file:///Users/alisongun/Downloads/xor_circuits%202/results_transformer/TRANSFORMER_REPORT.md)
- In four of five B models the MLP's clean units are 100% count thresholds of a_k + b_k + c_k (OR3, majority, AND3), "exactly as a threshold-logic textbook would" — [results_transformer_gates/REPORT.md](file:///Users/alisongun/Downloads/xor_circuits%202/results_transformer_gates/REPORT.md); confirmed by T7 (count explains ≥ 95% for 100% of such units) — [results_followup/SUMMARY.md](file:///Users/alisongun/Downloads/xor_circuits%202/results_followup/SUMMARY.md)
- One label-free direction carries the carry's causal effect: 97–100% interchange in B, 88–99% in C2. Its snapped 8-entry table equals the carry truth table in every model, and B's direction works as a carry gate inside an analog 64-bit ripple adder (100% of 100k sums) — [results_transformer_gates/REPORT.md](file:///Users/alisongun/Downloads/xor_circuits%202/results_transformer_gates/REPORT.md)
- 16-bit MLPs show carry-like *behaviour*: accuracy depends on carry travel distance, deeper nets reach further, and the total builds up across layers. They also have XOR-like features at some positions (bit 6 97%, bits 9–10 86% in depth-4 layer 2) — [results_adder16/ADDER16_REPORT.md](file:///Users/alisongun/Downloads/xor_circuits%202/results_adder16/ADDER16_REPORT.md)
- The msb-first transformer learned a non-engineered trick: it reads the carry into bit k backwards from emitted digits s_{k+1} and s_{k+2} (patch flip rate up to 0.61) — [results_transformer/TRANSFORMER_REPORT.md](file:///Users/alisongun/Downloads/xor_circuits%202/results_transformer/TRANSFORMER_REPORT.md)

**Explicitly not found:**
- The 4-bit adder MLP "does not contain the tiled adder's parts":
  - per-bit XOR exists only at bit 0, which is itself an output bit;
  - carry probes are uninformative;
  - knocking out carry units costs ≤ 9%;
  - carry swaps work in 0–5%;
  - instead, by layer 2, η² of the total reaches 0.73 / 0.52.
  
  [results_adder/ADDER_REPORT.md](file:///Users/alisongun/Downloads/xor_circuits%202/results_adder/ADDER_REPORT.md)
- 16-bit: "no single carry wire that can be read and rewritten, unlike the tiled adder". The 2^k shift appears in 0–4% of patches, and the one layer where patching mattered disrupted answers without flipping the carry — [results_adder16/ADDER16_REPORT.md](file:///Users/alisongun/Downloads/xor_circuits%202/results_adder16/ADDER16_REPORT.md)
- "No model built a carry chain", and "the 'circuit' is clean only where the format allows a clean one" — [results_transformer/TRANSFORMER_REPORT.md](file:///Users/alisongun/Downloads/xor_circuits%202/results_transformer/TRANSFORMER_REPORT.md)
- Transformer carry neurons and gates:
  - The best unit has R² 0.30–0.40, and no subset of up to 40 units reads the carry at 99%.
  - The PC1 gate margin is 0.41–0.70 (criterion ≥ 4; trained carry neuron 104), and it does not restore at ε = 0.05.
  - Writing the gate carry in as a class mean gives ≤ 0.2%.
  - Replacing the MLP with a library-XOR module gives 69–96% (B) and 2–6% (C2).
  
  [results_transformer_gates/REPORT.md](file:///Users/alisongun/Downloads/xor_circuits%202/results_transformer_gates/REPORT.md)
- C2 MLP units are mostly single-combination detectors (for example ¬a ∧ b ∧ ¬c), not thresholds. B:3 has no clean units at all and needs 20 units for a 99% sum readout — [results_transformer_gates/REPORT.md](file:///Users/alisongun/Downloads/xor_circuits%202/results_transformer_gates/REPORT.md)
- Even after removing the current digits' group means, PC1's gate margin stays below 4 (B 2.79–3.54, B:3 0.91, C2 0.86–1.45). PC1 is better described as a count or answer coordinate than as a carry variable (T4: it follows the source's answer 68.7–90.5%) — [results_followup/SUMMARY.md](file:///Users/alisongun/Downloads/xor_circuits%202/results_followup/SUMMARY.md)
- The earlier literature review frames the transformer result against two prior papers:
  - Kruthoff (2024) maps a model onto an XOR/AND full adder with carry neurons, and claims 1-layer models do not carry.
  - Quirke & Barez (ICLR 2024) describe the carry as a head-level state.
  
  It says neither shows "a causally used linear carry direction". It also raises an unresolved framing issue: "MAJ(a, b, c) is the full-adder carry-out", so the MAJ count unit is not the ruled-out carry neuron — [reports/Transformer carry gates related work.md](file:///Users/alisongun/Downloads/xor_circuits%202/reports/Transformer%20carry%20gates%20related%20work.md)

### Inferences
- The project's consistent conclusion, stated in several reports, is that "a network finds whatever decomposition training makes easiest". It also says composition from primitives is "one valid description ... not the one the network uses" — [results_adder/ADDER_REPORT.md](file:///Users/alisongun/Downloads/xor_circuits%202/results_adder/ADDER_REPORT.md). Gate-like parts are found at the level of *functions computed by groups of units*: the count thresholds and the count direction. Single neurons that equal an engineer's gate are not found.
- Gate-likeness here is graded, not binary. The project already has a quantitative scale for it, the gate margin (|mean − 0.5| / within-combo sd): about 0.5 for the raw PC1, about 3 after context removal (B), and 104 for a trained gate neuron. This is a usable continuous measure for comparing with outside work.
- The MAJ units inside the B MLP (31–41 per model) compute the full-adder carry-out of the *current* position. A follow-up could test whether they are causally used as such, for example by ablating MAJ-type units and seeing whether the next position's carry breaks. The project never ran this test.

### Gaps
- No distributed alignment search (DAS) or multi-dimensional subspace search was run on the 4-bit or 16-bit MLPs. The adder16 report names DAS as "the stronger next test", so a nonlinear or multi-dimensional carry in the MLPs is not ruled out.
- The role of the MAJ units (current-position carry-out) was not causally tested. The literature review flags it as unresolved.
- No multi-layer transformer was given the gate-likeness analysis. Only 1-layer B and C2 were studied in the pre-registered work.

## Q3. Which design facts proved decisive?

### Takeaway
Five facts decided the results:
1. **The interface between parts.** Scale and offset decide whether parts compose, from XOR transplants (direct 56% vs aligned 99%) to the context-conditioned carry transplant (55% → 100%).
2. **Static discipline at margin 4**, which gives gain |w|/4 ≈ 2 > 1 at threshold.
3. **Deep OR trees of analog zeros.** These, not random circuits, expose weak margins.
4. **Rewarding the tool's own result** in REINFORCE. This, not the answer's likelihood, made the CPU head learnable.
5. **Data structure.** Memorisation traps, a hash-based split that withheld whole sums, and operand-order shortcuts shaped what the GPTs learned.

### Cited Findings
- **Interface in XOR.** Direct transplant gets 56%, but rescaling to the recipient's mean and spread gets 99%; sham aligned gets 10%. "The interface (scale and offset of the signal between parts) is what decides whether parts compose" — [results/REPORT.md](file:///Users/alisongun/Downloads/xor_circuits%202/results/REPORT.md); [results_adder/ADDER_REPORT.md](file:///Users/alisongun/Downloads/xor_circuits%202/results_adder/ADDER_REPORT.md)
- **Snapping or margin in tiled adders.** Snapped outputs give 100% at both margins. Raw outputs at margin 2 give 99.5% (worst assembly 93.8%) and at margin 6 give 100%. Error grows along the carry chain (0.06 → 0.13) — [results_adder/ADDER_REPORT.md](file:///Users/alisongun/Downloads/xor_circuits%202/results_adder/ADDER_REPORT.md)
- **Margin 4 and gain above 1.** The pass rate jumps from 0% to 100% between margin 3.5 and margin 4 for every 2-input gate. A gate restores only if its threshold slope |w|/4 exceeds 1: margin 2 gives NOT w ≈ 4 (gain 1), margin 4 gives w ≈ 8 (gain 2). Margin-2 latches collapse to 0.56 / 0.44 — [results_minigates/MINIGATES_REPORT.md](file:///Users/alisongun/Downloads/xor_circuits%202/results_minigates/MINIGATES_REPORT.md)
- A test in the suite checks this: the hand-built NAND sigmoid(18 − 12a − 12b) restores at ε = 0.05, and the same NAND with weights divided by 4 does not — [tests/check_all.py](file:///Users/alisongun/Downloads/xor_circuits%202/tests/check_all.py)
- **OR-tree depth failures.** Margin-3 gates pass 400-layer random circuits and hold a latch, yet the snake breaks them on tick 1. Its 256-input crash-detector OR tree (8 levels) amplifies analog zeros:
  - margin 3: 0.047 → 0.086 → 0.141 → 0.258 → 0.639 → 0.997 by level 5;
  - margin 3.5: reaches 0.382 at level 8 and 0.93 at level 9, so a 32 × 32 board with a 10-level tree would fail;
  - margin 4: settles at 0.030 at any depth.
  
  "Random circuits rarely put a gate in its worst case for many layers in a row; real machines do it all the time" — [results_snake/SNAKE_REPORT.md](file:///Users/alisongun/Downloads/xor_circuits%202/results_snake/SNAKE_REPORT.md); [results_minigates/MINIGATES_REPORT.md](file:///Users/alisongun/Downloads/xor_circuits%202/results_minigates/MINIGATES_REPORT.md)
- The CPU's longest OR tree is the 32-way RAM read (5 levels), so it tolerates margin 3.25 (worst signal 0.33) where the snake needed 3.5 — [results_cpu/CPU_REPORT.md](file:///Users/alisongun/Downloads/xor_circuits%202/results_cpu/CPU_REPORT.md)
- **Smallest XOR is the easiest to train.** 1 hidden neuron plus a skip connection converges 100/100, against 57/100 for 2→2→1. A chain of monotone units cannot be XOR — [results_minigates/MINIGATES_REPORT.md](file:///Users/alisongun/Downloads/xor_circuits%202/results_minigates/MINIGATES_REPORT.md)
- **REINFORCE with a result-check reward.** Rewarded only by the answer's likelihood, 0 of 3 seeds learned the head fully and 2 of 3 partly, after about 1,750 flat steps in which the network memorised training pairs. With +1 added when the computer's result equals the answer, pointers settle in 250–1,000 steps in every seed — [results_cpugpt/CPUGPT_REPORT.md](file:///Users/alisongun/Downloads/xor_circuits%202/results_cpugpt/CPUGPT_REPORT.md)
- Code settings:
  - Answer-only condition: `shaping=0.0, K=16, entropy=0.02, gate_bias=2.0, call_cost=0.0`.
  - "+ check": `shaping=1.0, K=8, entropy=0.01, call_cost=0.01`.
  - "pricier": `call_cost=0.05`.
  
  [run_cpugpt.py](file:///Users/alisongun/Downloads/xor_circuits%202/run_cpugpt.py); loss in `losses()` — [cpugpt/model.py](file:///Users/alisongun/Downloads/xor_circuits%202/cpugpt/model.py)
- In the story GPT the pointer reward is pure result-checking: P sampled pointer tries per position, rewarded when "the computer returned the next token". The call decision is rewarded by the answer's log-likelihood + check·right·call − call_cost·call, with defaults check = 1.0 and call_cost = 0.05. Training uses K = 2 and P = 64 for the first 1,500 steps, then 16 — [cpugpt/lm.py](file:///Users/alisongun/Downloads/xor_circuits%202/cpugpt/lm.py); [run_storygpt.py](file:///Users/alisongun/Downloads/xor_circuits%202/run_storygpt.py)
- **Call cost.** At a price of 0.01 every seed calls on every math problem, but also on every non-math one. At 0.05 the best seed calls on 100% of math and 0% of non-math. Seeds 1 and 2 instead stop calling for multiplication, of which there are only about 1,400 legal pairs: they memorise the training pairs and get about 20% on held-out ones. "Price it too high and it stops calling for multiplication pairs it has memorized" — [results_cpugpt/CPUGPT_REPORT.md](file:///Users/alisongun/Downloads/xor_circuits%202/results_cpugpt/CPUGPT_REPORT.md)
- **Memorisation traps.** The baseline scores 96–100% on training pairs and 1.9–18.4% on held-out ones. Single-token numbers make memorisation easy — [results_cpugpt/CPUGPT_REPORT.md](file:///Users/alisongun/Downloads/xor_circuits%202/results_cpugpt/CPUGPT_REPORT.md)
- **Operand swapping.**
  - Seeds 1 and 2 of "+ check" send the operands swapped in 97–99% of the three short wordings.
  - This is harmless for addition. For subtraction the network learned to fix b − a itself and still gets 100%.
  - For multiplication, a swapped call loops a times instead of b, and when a > 17 it exceeds the 600-tick limit and returns 0. That is "the whole 13% loss on multiplication".
  
  [results_cpugpt/CPUGPT_REPORT.md](file:///Users/alisongun/Downloads/xor_circuits%202/results_cpugpt/CPUGPT_REPORT.md); CAP = 600 in [gatecpu/calc.py](file:///Users/alisongun/Downloads/xor_circuits%202/gatecpu/calc.py)
- **Read-back path and a split artefact (story GPT).**
  - `is_test` hashes a·7919 + b·104729 + op·31337 mod 5. Since 7919 ≡ 104729 ≡ 4 (mod 5), the split depends only on a + b, and held-out + pairs are exactly the sums ≡ 2 (mod 5).
  - As a result, 227, 247 and 237 appear as training answers only 11, 13 and 29 times, against a median of 371 for other numbers in 192–255. They cause about 110 misreads per model.
  - A copy read-back (a trained bonus of about 3.0 on the result token's logit) cut misreads to 0–1. It also doubled learning speed: 71–72% vs 37–42% held-out template math at step 1,000.
  
  [results_storygpt/STORYGPT_REPORT.md](file:///Users/alisongun/Downloads/xor_circuits%202/results_storygpt/STORYGPT_REPORT.md); code in [cpugpt/data.py](file:///Users/alisongun/Downloads/xor_circuits%202/cpugpt/data.py)
- **Frozen bus tables.** Token → 8 bits and word → opcode are given, not learned (*more* → +, *ate*/*lost*/*gave* → −, *each* → ×). The gate decides whether to call. "Learning it would need gradients through the computer or far more exploration" — [results_cpugpt/CPUGPT_REPORT.md](file:///Users/alisongun/Downloads/xor_circuits%202/results_cpugpt/CPUGPT_REPORT.md); [results_storygpt/STORYGPT_REPORT.md](file:///Users/alisongun/Downloads/xor_circuits%202/results_storygpt/STORYGPT_REPORT.md)
- **Output order in transformers.** Lowest-first makes addition local (loss below 0.002 by step 400, 100% accuracy). Highest-first needed all 2,500 steps to reach 0.0065 and stays imperfect — [results_transformer/TRANSFORMER_REPORT.md](file:///Users/alisongun/Downloads/xor_circuits%202/results_transformer/TRANSFORMER_REPORT.md)
- **Attention averaging plus a relative readout.** The model averages a and b of a digit, so it never needs separate bits. Pinning PC1 to a class mean gives chance, while shifting by the class gap flips 93–100% — [results_transformer_gates/REPORT.md](file:///Users/alisongun/Downloads/xor_circuits%202/results_transformer_gates/REPORT.md). Conditioning the written value on (k, a_k, b_k) brings transplants to 100% in four B models — [results_followup/SUMMARY.md](file:///Users/alisongun/Downloads/xor_circuits%202/results_followup/SUMMARY.md)
- **Ablation choice.** Mean ablation is the default because zeroing a tanh unit also removes its offset. Zero and mean ablation agree on sufficiency in only 23–100% of nets, depending on architecture — [results/REPORT.md](file:///Users/alisongun/Downloads/xor_circuits%202/results/REPORT.md)
- **Restricted alignment maps.** Moment matching is used "not a fitted map, so it cannot simply learn the answer", citing Sutter et al. (NeurIPS 2025) on powerful maps making any circuit look transplantable — [results/REPORT.md](file:///Users/alisongun/Downloads/xor_circuits%202/results/REPORT.md)

### Inferences
- The same lesson appears at four scales: interface scale and offset (XOR), restoring margin (gates), context-conditioned value (transformer carry), and read-back format (story GPT). In every case, whether a part transplants depends on matching the receiver's expected signal statistics, not on the part being "the same function". This is a clean bridge to any outside paper about transplanting or hand-building circuits.
- Margin 4 gives an ε-interface of 0.05. The transformer's carry direction has an effective margin of about 0.5 (raw) to 3.5 (context-removed), which is below the threshold the gate study shows is needed for chained use. Yet B's gate still worked in a 64-bit ripple adder, because library XORs restored every sum bit. Restoration can therefore be supplied by neighbouring parts, which the PREREG pass criterion did not anticipate.

### Gaps
- The CPU-GPT report does not say whether the same `is_test` artefact (split by a + b mod 5) affected its held-out set. The code is shared, and by the same arithmetic held-out − and × pairs are also fixed residues of a + b (inference from [cpugpt/data.py](file:///Users/alisongun/Downloads/xor_circuits%202/cpugpt/data.py), not stated in that report).
- No ablation separates call cost from the result-check reward in the story GPT: both are on by default, and only one configuration was trained.
- k-input threshold gates (one neuron as an 8-input AND or OR) were deliberately excluded, so their effect on OR-tree failures and size is untested.

## Q4. What open questions and stated limitations do the reports list?

### Takeaway
The stated limitations cluster around four issues: small scale and few seeds (1–5 seeds, one width, 10–16 bits); single-direction linear tests (no DAS or nonlinear subspaces); confounds in the data distribution (rare long carries, held-out splits, single-token numbers); and the CPU-head interface being hand-specified (frozen bus tables, one call per answer, poor transfer to new sentence shapes). The explicit next steps named are: DAS on adders, training on more long-carry problems, transformers vs MLPs, k-input gates, more varied sentence shapes with held-out shapes, and learning the bus format.

### Cited Findings
- XOR:
  - Four inputs make any similarity metric low-resolution, and behaviour cannot distinguish mechanisms.
  - Boolean labels are "a lens", and ReLU units take graded values.
  - "Minimal circuit" depends on the ablation choice.
  - "Nothing here shows universality beyond XOR in 2→H→1 networks."
  
  [results/REPORT.md](file:///Users/alisongun/Downloads/xor_circuits%202/results/REPORT.md)
- 4-bit adder:
  - The task is small (256 problems) and the total fits in one number, so 16 bits is named as the natural next test (later done).
  - Carry tests use one straight-line direction per layer.
  - Only MLPs were tested.
  
  [results_adder/ADDER_REPORT.md](file:///Users/alisongun/Downloads/xor_circuits%202/results_adder/ADDER_REPORT.md)
- 16-bit adder:
  - One width (256), one optimiser, 3 seeds per depth.
  - Long travel distances are rare (0.8%), confounding the depth and data effects. Training on more of them "would separate the depth effect from the data effect".
  - "Distributed alignment search ... is the stronger next test."
  - Transformers are named as the comparison.
  
  [results_adder16/ADDER16_REPORT.md](file:///Users/alisongun/Downloads/xor_circuits%202/results_adder16/ADDER16_REPORT.md)
- First transformer study:
  - One seed per model, one width, 10 bits, and possibly polysemantic heads.
  - Single-position patching misses redundantly spread information.
  - The NumPy GPT does not match TransformerLens initialisation.
  
  [results_transformer/TRANSFORMER_REPORT.md](file:///Users/alisongun/Downloads/xor_circuits%202/results_transformer/TRANSFORMER_REPORT.md)
- Pre-registered transformer study:
  - One width, one layer, 10-bit training, 5 + 3 seeds. C2 is the authors' own design.
  - Linear probes only: "not linearly readable" does not mean absent.
  - The gate table uses only PC1, and "the four-dimensional subspace may hold a cleaner gate than PC1 alone".
  - A fitted sigmoid would restore more easily but "would no longer be the transformer's own response".
  - Deviations: B seeds 0, 2, 3, 4 use their seed for both initialisation and data stream. There were three analysis runs, with "same verdicts". The `pc1_diagnostic_exploratory` and unit-function breakdown were added after seeing results.
  
  [results_transformer_gates/REPORT.md](file:///Users/alisongun/Downloads/xor_circuits%202/results_transformer_gates/REPORT.md)
- PREREG pre-noted that P7b's restoring test must fail even for a perfect table: trilinear interpolation has gain 2 at corners like (0,1,1) — [results_transformer_gates/PREREG.md](file:///Users/alisongun/Downloads/xor_circuits%202/results_transformer_gates/PREREG.md)
- Minimal gates:
  - Sigmoid only; a non-monotone "bump" activation would make XOR a single neuron.
  - 2-input gates only.
  - The static discipline was "checked on a 9 × 9 grid of input points per corner, not proven".
  
  [results_minigates/MINIGATES_REPORT.md](file:///Users/alisongun/Downloads/xor_circuits%202/results_minigates/MINIGATES_REPORT.md)
- Snake and CPU:
  - 2-input gates only; storing the whole board is costly (a ring buffer would be cheaper).
  - Flip-flops are simulated with ideal clock edges and all eight gates updating together, so races and glitches are not modelled.
  - The LFSR always places the first food in the same spot.
  - CPU: 32 bytes, no stack or subroutines or immediates or input port, and LDA does not set Z. A 256-byte machine would be about 30,000 gates.
  
  [results_snake/SNAKE_REPORT.md](file:///Users/alisongun/Downloads/xor_circuits%202/results_snake/SNAKE_REPORT.md); [results_cpu/CPU_REPORT.md](file:///Users/alisongun/Downloads/xor_circuits%202/results_cpu/CPU_REPORT.md)
- CPU-GPT:
  - The bus format is given, not learned.
  - One call per answer, always at the last position, with a fixed calculator program: "The model does not write programs."
  - Single-token numbers favour memorisation.
  - 3 seeds, with seed-dependent results.
  
  [results_cpugpt/CPUGPT_REPORT.md](file:///Users/alisongun/Downloads/xor_circuits%202/results_cpugpt/CPUGPT_REPORT.md)
- Story GPT:
  - New sentence shapes mostly fail, and "the gate has learned the shapes rather than 'this is a word problem'". The next step is many more shapes with some held out.
  - The + split is "sums ≡ 2 (mod 5)".
  - Single-step problems only, results 0–255.
  - One plain seed, two seeds per CPU-head variant.
  - The model is small (1.4 M weights).
  
  [results_storygpt/STORYGPT_REPORT.md](file:///Users/alisongun/Downloads/xor_circuits%202/results_storygpt/STORYGPT_REPORT.md)
- The earlier literature review lists four additions to pre-empt reviewers:
  1. state whether interchange numbers are noising or denoising;
  2. report interchange accuracy on random-init models;
  3. show PC1 is active on-distribution (Makelov et al. illusion);
  4. explain why the MAJ count-threshold unit is not a carry gate.
  
  It also notes no source connecting the static discipline to sigmoid neural networks — [reports/Transformer carry gates related work.md](file:///Users/alisongun/Downloads/xor_circuits%202/reports/Transformer%20carry%20gates%20related%20work.md)

### Inferences
- Of the review's four suggested additions:
  - (2) is partly addressed by the pre-registered random-init PC1 control (9.1–39.9%), but not with DAS-style learned subspaces on random models.
  - (3) is partly addressed, because PC1 is by construction the principal direction of on-distribution counterfactual differences.
  - (1) and (4) are not addressed in any report found.
- The story GPT's failure on new shapes is a concrete, testable generalisation gap: pointers transfer partly, but the call gate does not. It is a natural target for a follow-up.

### Gaps
- No report explicitly states whether the interchange is noising or denoising. The code patch x ← x + P(x_src − x) copies the source into the base run, but the report does not label it in those terms.
- No seed-variance analysis exists for the 4-bit adder MLP, the snake margin sweep (3 runs each), or the msb transformer.

## Q5. What reusable tools exist in the repo for a follow-up?

### Takeaway
The repo has a complete, tested toolchain:
- XOR-net training and exhaustive circuit, ablation, patching and transplant analysis (`xorlab/`);
- a NumPy GPT with TransformerLens-style hooks and saved addition checkpoints (`tinygpt.py`, `results_transformer*`);
- the pre-registered analysis pipeline (probes, label-free flip directions, interchange, class-mean and context transplants, gate tables, restore tests; `transformer_gates.py`, `followup_tests.py`);
- a library of trained margin-4 gate neurons with a static-discipline checker (`gatesnake/gates.py`, `results_minigates/library.json`);
- an HDL plus logic and neural simulators (`gatesnake/hdl.py`, `gatesnake/sim.py`);
- a gate CPU with assembler, calculator program and bit-sliced batch runner (`gatecpu/`);
- PyTorch CPU-head models with REINFORCE losses (`cpugpt/`);
- browser UIs and PASS/FAIL test suites.

### Cited Findings
- `xorlab/`:
  - `model.py`: 2→H→1 nets, batched Adam, stopping rule.
  - `analysis.py`: `bool_label`, `forward(p, act, ablate, mode, patch)`, `minimal_circuits`, `causal_tests`, `moment_match`, `transplant`, `reference_contribution` (hand-built XOR), `linear_cka`.
  - Outputs include `results/checkpoints/*.npz` (initial and final weights) and `circuits.json`.
  
  [README.md](file:///Users/alisongun/Downloads/xor_circuits%202/README.md); [xorlab/analysis.py](file:///Users/alisongun/Downloads/xor_circuits%202/xorlab/analysis.py)
- Adder pipelines `run_adder.py` and `run_adder16.py`: tiled adders, carry and travel-distance helpers (`carries`, `chain_len`, `opposite_carry_source`, `built_distance`, `tiled_adder`, checked in tests). The saved 16-bit nets are in `results_adder16/nets.npz` — [README.md](file:///Users/alisongun/Downloads/xor_circuits%202/README.md); [tests/check_all.py](file:///Users/alisongun/Downloads/xor_circuits%202/tests/check_all.py)
- `tinygpt.py`: a NumPy pre-LN GPT with manual backprop and `forward(tokens, hooks=None, cache=False)`, where hooks replace named activations for patching or ablation, plus a finite-difference gradient check. `run_transformer_tl.py` is the TransformerLens version. Checkpoints are `results_transformer/{lsb-L1, lsb-L2, msb-L2}.npz` and `results_transformer_tl/*.pt` — [tinygpt.py](file:///Users/alisongun/Downloads/xor_circuits%202/tinygpt.py); [README.md](file:///Users/alisongun/Downloads/xor_circuits%202/README.md)
- `transformer_gates.py` (1,271 lines):
  - layouts `Flat` and `Inter`; `RelPosGPT` (relative-position model);
  - `generate`, `wilson` CIs, `lib_gate`, `gate_adder(A, B, carry_override, carry_gate)` (analog ripple adder from library neurons), `train_neuron(margin=4)`;
  - `Probe` (L2 logistic, IRLS), `greedy_readout`, `two_means`, `trilinear`, `restores3`;
  - `opposite_pairs`, `flip_dirs` (label-free PCs of counterfactual differences), `step_interchange`, `class_means`, `step_carry_transplant`, `pc1_diagnostic`, `gate_table`, `step_gate_from_transformer`, `step_module_transplant`;
  - `judge` (applies the pre-registered pass rules).
  
  Models are saved in `results_transformer_gates/models/` — [transformer_gates.py](file:///Users/alisongun/Downloads/xor_circuits%202/transformer_gates.py)
- `followup_tests.py`: T1–T9, including `ctx_means` (context-conditioned means), `t2_transplant`, `t3_steering`, `pairs(mode)` (same / cross / same-token pairs), `t8_continuation` and `t9_unit_dependence`. It runs on the same 8 models in about 10 minutes — [followup_tests.py](file:///Users/alisongun/Downloads/xor_circuits%202/followup_tests.py)
- `gatesnake/gates.py`:
  - `train(gate, arch, margin, seeds)`;
  - `restores(p, gate, eps, pts=9)`, the static-discipline test;
  - `restore_range`, `quantize(p, bits)`, `gate_cost`.
  
  `results_minigates/library.json` holds margin 4.0, ε-interface 0.05, 7 gate types with 8 trained nets each, and a baseline — [gatesnake/gates.py](file:///Users/alisongun/Downloads/xor_circuits%202/gatesnake/gates.py); [results_minigates/MINIGATES_REPORT.md](file:///Users/alisongun/Downloads/xor_circuits%202/results_minigates/MINIGATES_REPORT.md)
- `gatesnake/hdl.py`: a `Circuit` builder (AND/OR/NAND/NOR/XOR/XNOR, trees, mux, decoder, incdec, equal) with constant folding and deduplication; `Netlist` (counts, breakdown, stats, `eval_logic`); and `lower_to_nand`. `gatesnake/sim.py` provides `Machine(net, mode="logic"|"neural", library, dff="gates"|"ideal", snap, noise, latch_iters=6)` — [gatesnake/hdl.py](file:///Users/alisongun/Downloads/xor_circuits%202/gatesnake/hdl.py); [gatesnake/sim.py](file:///Users/alisongun/Downloads/xor_circuits%202/gatesnake/sim.py)
- `gatecpu/`:
  - `design.py` `build_cpu()`; `ref.py` (Python CPU); `asm.py` (assembler with OUT, HLT, labels, `.byte`);
  - `calc.py` (26-instruction calculator program; `run_calc(x, y, op, cap=600)`, `call_ticks`, `expected`);
  - `batch.py` (`BitMachine`, bit-sliced at 64 machines per uint64 word; `build_table`).
  
  The precomputed `results_cpugpt/cpu_table.npz` covers all 262,144 calls — [gatecpu/batch.py](file:///Users/alisongun/Downloads/xor_circuits%202/gatecpu/batch.py); [results_cpugpt/CPUGPT_REPORT.md](file:///Users/alisongun/Downloads/xor_circuits%202/results_cpugpt/CPUGPT_REPORT.md)
- `cpugpt/`:
  - `model.py`: `CPUHead` with `scores`, `inputs`, `run`, `bits`; `GPT(cpu=...)`; `losses(shaping, entropy, call_cost)` with leave-one-out REINFORCE.
  - `lm.py`: `StoryGPT` with bits / +embedding / +copy read-backs and `lm_losses`.
  - `data.py`: templates and `is_test`.
  - `text.py`: word-problem generator and `OPCODE_OF_WORD`.
  
  Trained models are in `results_cpugpt/models` and `results_storygpt/models` — [cpugpt/model.py](file:///Users/alisongun/Downloads/xor_circuits%202/cpugpt/model.py); [cpugpt/lm.py](file:///Users/alisongun/Downloads/xor_circuits%202/cpugpt/lm.py)
- Browser UIs (`snake_ui/`, `cpu_ui/`, `cpugpt_ui/`, `storygpt_ui/`) run the exported netlists fully analog in JavaScript, with live margin and noise controls and self-tests against the Python reference — [results_snake/SNAKE_REPORT.md](file:///Users/alisongun/Downloads/xor_circuits%202/results_snake/SNAKE_REPORT.md); [results_cpu/CPU_REPORT.md](file:///Users/alisongun/Downloads/xor_circuits%202/results_cpu/CPU_REPORT.md)
- Tests: `tests/check_all.py` (gradient checks, ablation and transplant identities, CKA invariances, adder helpers, gate restore checks, snake vs reference, bus-table copy) and `tests/test_circuits.py` (for `transformer_gates.py`; needs no trained model). Both print PASS/FAIL and change nothing — [tests/check_all.py](file:///Users/alisongun/Downloads/xor_circuits%202/tests/check_all.py); [tests/test_circuits.py](file:///Users/alisongun/Downloads/xor_circuits%202/tests/test_circuits.py)
- Performance note: the scripts set `OPENBLAS_NUM_THREADS=1`, because multithreaded BLAS on these small matrices slowed parallel runs "from minutes to over an hour" — [README.md](file:///Users/alisongun/Downloads/xor_circuits%202/README.md)

### Inferences
- Ready-made ingredients exist for a "hand-built vs learned circuit" follow-up:
  - library gate neurons as hand-specified reference parts;
  - `gate_adder` and `step_carry_transplant` for swapping a hand-built part into a trained model;
  - `ctx_means` and T2 for the context-conditioned version that succeeds;
  - `flip_dirs` and `step_interchange` for label-free causal directions;
  - the gate-margin metric for grading how gate-like a learned feature is.
  
  If the outside paper compares hand-written ("artificial") circuits with naturally learned ones, these map directly onto its components. The writer should check that paper's actual methods; I did not examine it.
- Cheap, testable follow-ups the existing code supports with small changes:
  - (a) Repeat the T2 context-conditioned transplant with the *trained single-neuron carry gate* output driving the written value, to see whether a gate neuron can feed the transformer once the count context is supplied.
  - (b) Run DAS (not implemented in the repo) against PC1 on the saved B/C2 models and the 16-bit MLPs in `nets.npz`.
  - (c) Train B-family models with a margin or static-discipline regulariser on the PC1 coordinate, to see whether a restoring carry gate can emerge.
  - (d) Hold out sentence shapes in the story GPT to measure gate transfer.

### Gaps
- No DAS, SAE or other learned-subspace tool exists in the repo, and `run_transformer_tl.py` has unreported outputs. Their reliability for follow-up use is unverified.
- I did not open `xorlab/model.py`, `gatesnake/design.py`, `gatecpu/design.py` or the UI code beyond their function lists and docstrings. Their APIs are listed from signatures, not from use.
