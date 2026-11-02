# Bridging circuits-style interpretability and logic-gate networks: background for linking Cammarata (Oct 2026) to the neural-logic-gate project

Scope note. The earlier review at `reports/Transformer carry gates related work.md` (the "carry review" below) already covers: addition transformers (Quirke & Barez; Kruthoff; Lee et al.; Nanda et al.; Zhong et al. Clock/Pizza; Stolfo; Nikankin; Zhou et al. Fourier; Hanna greater-than; length-generalisation papers), RASP/Tracr/MIPS, Liu et al. automata shortcuts, the full causal-abstraction and patching stack (Vig; Geiger 2021/2022; DAS; Wu et al. Alpaca; IOI; ACDC; Zhang & Nanda; Heimersheim & Nanda; Makelov et al. and the Wu et al. reply; Sutter et al. non-linear dilemma; Li & Janson optimal ablation; causal scrubbing; Hydra), probing pitfalls (Park et al. LRH; Marks & Tegmark; ITI; CausalGym; RAVEL; Kumar et al.; control tasks; MDL; amnesic/INLP/LEACE; random-init baselines Zhang & Bowman, Wieting & Kiela, Zhong & Andreas), Toy Models of Superposition (cited only via arXiv), threshold logic history (McCulloch-Pitts; Minsky-Papert; Minnick 1961; Muroga; Siu-Roychowdhury-Kailath; Parberry; Hajnal et al.), static discipline (MIT 6.111; Agarwal & Lang; von Neumann restoring organs), Hänni et al. computation in superposition, Barak et al. parity, and pre-registration. These are cited here only as "covered" and are not repeated. Everything below is new relative to that review.

Verification legend used throughout: **[opened]** = primary page fetched this session; **[snippet]** = confirmed only via search-result summaries; **[background, not opened]** = from my prior knowledge, URL/ID given for the writer to check, content not verified this session.

---

## 1. What is Nick Cammarata's most recent paper (as of 28 Oct 2026), in brief?

### Takeaway
Cammarata's latest work is an essay rather than a peer-reviewed paper: "Neural Networks Are Already Clean" (Goodfire, reported as 15 Oct 2026). It argues that individual neurons are mostly monosemantic "gates" with one main job. On this account, apparent polysemanticity comes mostly from weak tools, small or badly regularised models, and "recursive generalization" (neurons growing related offshoots), not from superposition. This puts it in direct tension with the project's finding (2) that carry is a direction rather than a clean gate neuron, which makes that finding a natural falsification test of the essay's claim.

### Cited Findings
- Title "Neural Networks Are Already Clean", author Nick Cammarata (Goodfire), with contributions from Curt Tigges (language neurons) and Tarun Gogineni (editing). It is labelled "A working hypothesis", about a 65-minute read **[opened]**. The page text I read did not show a date. — [already-clean.goodfire.com](https://already-clean.goodfire.com/)
- Secondary reports give the publication date as 15 Oct 2026. They also say it is not peer-reviewed and that a technical companion paper is forthcoming **[snippet]**. Neel Nanda reportedly called it interesting but said it does not substantially change the broader picture **[snippet; not verified on X]**. — [the-decoder](https://the-decoder.com/is-ai-interpretability-close-to-a-breakthrough-a-new-essay-has-researchers-divided/); [Singularity Hub, 26 Oct 2026](https://singularityhub.com/2026/10/26/is-ai-really-a-black-box-neural-networks-may-be-easier-to-understand-than-we-thought); [explainx.ai](https://explainx.ai/blog/goodfire-neural-networks-are-already-clean-cammarata-neurons-interpretable-90-percent-qwen3-32b-2026)
- Gist **[opened]**:
  - Audit method: examine each neuron's positive and negative "poles", top activations, upstream and downstream connections, and steering effects, then test the explanation on new examples.
  - Preliminary Qwen3-32B audit: about 90% of poles have "one main job", about 7% have two related jobs, and about 2–4% are not yet understood. Similar results are reported for Kimi K3.
  - InceptionV1 last layer: about 5% interpretable with early tools, about 50% with fixed tools, about 70% with modern regularisation, and about 99% in a same-size ConvNeXt (7.06M parameters).
  - The Zoom In cat/car neuron is reinterpreted as a V-shape or pose detector.
  - Superposition is called the "last hypothesis", not the null hypothesis.
  - Neurons are framed as "verbs" or gates that let some inputs through, with a "votes" model of weighted inputs summed before the nonlinearity.
  - Source: [already-clean.goodfire.com](https://already-clean.goodfire.com/)
- Squeezed-layer experiment **[opened]**: InceptionV1 layer 3b was cut from 480 to 10 neurons and still reached 58.35% top-1. An SAE on that layer explains 82% of variance but does not recover clean curve detectors, and splicing it in drops top-1 from 58.8% to 43.0%. — [already-clean.goodfire.com](https://already-clean.goodfire.com/)
- Falsifiable predictions the essay itself states **[opened]**:
  - Recursive generalization and superposition predict opposite developmental trajectories, so watching a neuron grow over training checkpoints can tell them apart.
  - Explanations must generalise to new inputs.
  - Steering a neuron must produce predicted outputs.
  - The essay does not analyse AND/OR/XOR directly, but it disputes reading the cat/car neuron as "OR(cat, car)". It also mentions a "double-negative regime" in SwiGLU neurons without giving a Boolean analysis.
  - Source: [already-clean.goodfire.com](https://already-clean.goodfire.com/)
- Universality claims in the essay **[opened]**: a "pose" neuron appears in DINOv2 and in ConvNeXt medium and large, with only a rough version in the smallest ConvNeXt. The author says he sees it in "every vision model" he examined. — [already-clean.goodfire.com](https://already-clean.goodfire.com/)
- An alphaXiv profile lists a June 2026 paper, "Structuring Sparsity: Block-Sparse Featurizers Capture Visual Concept Manifolds" **[snippet; Cammarata's authorship not confirmed]**. — [alphaXiv profile](https://www.alphaxiv.org/@nick-cammarata)

### Inferences
- The sharpest bridge is to project finding (2). The essay predicts that a well-trained, well-regularised network will put a task variable like carry in one neuron's pole ("one main job"). The project finds a best-unit R² of about 0.30–0.40 for carry against 1.0 for an engineered sigmoid gate, and a direction-level interchange success of 97–100% (carry review). That is a counter-example candidate. It only counts if the project's networks meet the essay's preconditions: elementwise nonlinearity, enough width, and regularisation. Otherwise the essay can say the network is "too small or badly trained".
- Project finding (1) speaks to the essay's gate framing. The essay treats neurons as gates. The project shows that a sigmoid neuron is a reliable gate only under a margin or static-discipline condition. This supplies a quantitative criterion ("gain > 1 at threshold, noise margins") that the essay's qualitative "one main job" lacks.

### Gaps
- I read only the first 100k of 171k characters of the essay, so the SAE section and appendices were truncated. The companion technical paper does not appear to be out yet. Exact date, venue and any quantitative audit protocol (thresholds, sample sizes) must come from the other researcher covering the paper.

---

## 2. Circuits and universality: do different networks learn the same features and circuits?

### Takeaway
The evidence supports weak universality: the same kinds of features recur, especially simple and early ones. It does not support strong universality, where the same circuit appears in every run. Gurnee et al. find that only 1–5% of GPT-2 neurons are universal across seeds. Chughtai et al. find that the same group-theoretic algorithm is learned, but which representations get learned varies by seed. SAE-based comparisons find similarity but call it weak. Cammarata's essay claims strong recurrence of specific neurons in vision models, which is the more aggressive end of this spectrum.

### Cited Findings
- **Zoom In (Olah, Cammarata, Schubert, Goh, Petrov, Carter; Distill 2020)** makes three "speculative claims": features, circuits, universality. Universality is "analogous features and circuits form across models and tasks". Its evidence came mainly from curve detectors and high-low frequency detectors in several vision models, which the authors present as preliminary **[background, not opened — fetch of distill.pub exceeded size limit; claim wording corroborated via search snippet]**. — [Distill Zoom In](https://distill.pub/2020/circuits/zoom-in/); [AN #111 summary](https://www.alignmentforum.org/posts/5CApLZiHGkt37nRQ2/an-111-the-circuits-hypotheses-for-deep-learning)
- **Curve Detectors (Cammarata, Goh, Carter, Schubert, Petrov, Olah; Distill 2020)** is part 1 of a three-article curve series that reports curve-detecting neurons across multiple vision models **[snippet]**. "Curve Circuits" (Cammarata et al., Distill 2021) hand-rewrote the curve circuit from its weights **[background, not opened]**. — [mlanthology entry](https://mlanthology.org/distill/2020/cammarata2020distill-curve); [Distill Curve Circuits](https://distill.pub/2020/circuits/curve-circuits/)
- **Chughtai, Chan & Nanda (ICML 2023), "A Toy Model of Universality"** **[snippet]**:
  - Networks trained on finite-group composition learn representation matrices in the embeddings and use ReLU MLPs to multiply them.
  - The paper splits strong universality (the same features and circuits in all similarly trained models) from weak universality (shared principles, different implementations).
  - It reports that the particular irreducible representations learned vary across seeds, which counts against strong universality **[background, not opened]**.
  - Stander et al. (2023, arXiv:2312.06581) later disputed parts of the mechanism with a coset-based account **[background, not opened]**.
  - Sources: [arXiv:2302.03025](https://arxiv.org/abs/2302.03025); [Stander et al.](https://arxiv.org/abs/2312.06581)
- **Gurnee et al. (TMLR 2024), "Universal Neurons in GPT2 Language Models"** **[snippet]**:
  - The authors computed pairwise activation correlations over 100M tokens across 5 GPT-2 seeds.
  - Only 1–5% of neurons pass the universality threshold against random baselines.
  - Universal neurons form families with simple roles, such as deactivating attention heads or changing next-token entropy.
  - Source: [arXiv:2401.12181](https://arxiv.org/abs/2401.12181); [TMLR via mlanthology](https://mlanthology.org/tmlr/2024/gurnee2024tmlr-universal)
- A 2025 follow-up (not by Gurnee) finds universal neurons stable across checkpoints (100k/200k/300k steps). It reports that ablating all non-universal neurons causes minimal change **[snippet; small 5M-token data]**. — [arXiv:2508.00903](https://arxiv.org/abs/2508.00903)
- **Lan et al. (2024/25), SAE feature universality** **[snippet]**: SAE features from Pythia and Gemma were paired by activation correlation and compared with SVCCA/RSA. Individual features correlate highly, and semantic subspaces such as emotions are more similar still. The authors nevertheless call the universality weak, and different SAEs do not consistently learn the same features. — [arXiv:2410.06981](https://arxiv.org/abs/2410.06981)
- **Wang et al. (2024), "Towards Universality"** compares Transformers and Mamba via SAEs and reports analogous induction circuits **[snippet only]**. — [arXiv:2410.06672](https://arxiv.org/abs/2410.06672)
- **Bricken et al. (2023), "Towards Monosemanticity"** reports SAE features that are largely shared across two 1-layer transformers with different seeds **[background, not opened]**. Cammarata's essay cites its median-neuron score of 0/14 as the polysemanticity baseline it disputes **[opened, via essay]**. — [transformer-circuits.pub 2023](https://transformer-circuits.pub/2023/monosemantic-features/index.html); [already-clean.goodfire.com](https://already-clean.goodfire.com/)
- Older representation-similarity lineage **[background, not opened]**: Li, Yosinski et al. (2015), "Convergent Learning", finds that some units match one-to-one across nets and others only as subspaces. Kornblith et al. (2019) introduced CKA. Huh et al. (2024) proposed the "Platonic Representation Hypothesis". — [arXiv:1511.07543](https://arxiv.org/abs/1511.07543); [arXiv:1905.00414](https://arxiv.org/abs/1905.00414); [arXiv:2405.07987](https://arxiv.org/abs/2405.07987)
- **Toy Models of Superposition** is covered by the carry review. The point it misses: TMS argues that an elementwise nonlinearity creates a privileged basis, so features are expected to align with neurons only when there is a privileged basis and no superposition pressure **[background, not opened]**. — [transformer-circuits.pub 2022](https://transformer-circuits.pub/2022/toy_model/index.html)

### Inferences
- Evidence for universality has meant one of three things: (a) qualitative recurrence of a feature type (curve detectors, pose neurons); (b) neuron-level max-correlation across seeds above a random-rotation baseline (Gurnee); or (c) subspace similarity (SVCCA, CKA, SAE matching). Evidence against has meant seed-dependence of which algorithm or representation is chosen (Chughtai; Clock vs Pizza, covered). The project's "no engineer's ripple-carry parts" result is a type-(c)-against claim relative to an engineered reference, which is a useful new comparison axis. Universality is usually tested between trained networks, not between a trained network and a human design.
- **Standard design to adopt:** train N ≥ 5 seeds, then compute for each unit the maximum correlation with (i) units in other seeds, (ii) the outputs of each engineered ripple-carry gate (XOR, AND, MAJ, carry), and (iii) the same after a random orthogonal rotation of the layer (null for a privileged basis). Carry readable as a direction but not a unit predicts that (ii) is low, a direction-level probe is high, and rotation does not hurt the probe.
- **Tension with the essay:** Cammarata reports strong neuron-level recurrence, while Gurnee reports 1–5%. The two measure different things: qualitative "same job" versus activation correlation. A bridge experiment should pre-register which criterion counts.

### Gaps
- I could not open Zoom In, Towards Monosemanticity, Chughtai's full text or Stander et al., so their exact numbers are unverified. I found no universality study that compares learned circuits against an engineered gate-level reference design, which suggests a gap the project can fill. I did not open "Many Circuits, One Mechanism" (arXiv:2606.06267, June 2026), which appeared in search and may bear on circuit-level universality ([arXiv:2606.06267](https://arxiv.org/abs/2606.06267)).

---

## 3. Neurons as logic: threshold logic, Boolean labels for neurons, logic-gate networks, XOR features

### Takeaway
Three bodies of work turn "neurons are gates" into something testable. Logic-gate networks (Petersen et al.) train relaxed gates and harden them with almost no accuracy loss. Compositional neuron explanations (Mu & Andreas) score Boolean formulas against neuron activations by IoU. XOR-feature results (Marks) show that probes can decode XORs even where no XOR gate is visible. Classical threshold logic (covered) supplies the single-unit limits.

### Cited Findings
- **Petersen, Borgelt, Kuehne & Deussen (NeurIPS 2022), "Deep Differentiable Logic Gate Networks"** **[opened, ar5iv]**:
  - Each neuron holds a softmax over all 16 two-input gates. The relaxations are probabilistic: AND = ab, OR = a + b − ab, XOR = a + b − 2ab.
  - Wiring is pseudo-random and fixed, and only the gate choice is learned.
  - After training, each neuron takes its argmax gate. "Most neurons converge to a single gate", and on MNIST "the gap is smaller than 0.1%".
  - Depth is limited to about 8–10 layers before "vanishing gradients". Identity-like gates (A, B, ¬A, ¬B) appear in later layers and act like a residual.
  - Results: MNIST 98.47%; CIFAR-10 62.14% (large×4); "beyond a million images of MNIST per second on a single CPU core".
  - Sources: [arXiv:2210.08277](https://arxiv.org/abs/2210.08277); [NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/0d3496dd0cec77a999c98d35003203ca-Abstract.html)
- Convolutional Differentiable Logic Gate Networks (Petersen et al., NeurIPS 2024) adds residual-style initialisation for deeper nets **[background, not opened]**. — [arXiv:2411.04732](https://arxiv.org/abs/2411.04732)
- **Mu & Andreas (NeurIPS 2020), "Compositional Explanations of Neurons"** **[snippet]**:
  - The method explains a neuron by beam-searching logical forms (AND, OR, NOT) over primitive concepts, scored by IoU between the thresholded neuron mask and the concept mask. Example: (water OR river) AND NOT blue.
  - In vision, many neurons are abstract-but-coherent and some are polysemantic. In NLI, neurons follow shallow lexical heuristics.
  - The explanations predict "copy-paste" adversarial examples.
  - Sources: [arXiv:2006.14032](https://arxiv.org/abs/2006.14032); [NeurIPS 2020](https://papers.nips.cc/paper/2020/hash/c74956ffb38ba48ed6ce977af6727275-Abstract.html)
- Network Dissection (Bau et al., CVPR 2017) is the single-concept IoU precursor of Mu & Andreas **[background, not opened]**. — [arXiv:1704.05796](https://arxiv.org/abs/1704.05796)
- **Marks (LessWrong, 3 Jan 2024), "What's up with LLMs representing XORs of arbitrary features?"** **[opened]**:
  - The "RAX" hypothesis: whenever a model linearly represents a and b, it also linearly represents a⊕b.
  - In LLaMA-2-13B (layer 14, residual stream), XOR probes were "perfect or near-perfect" in every case checked. In a weight-shuffled network, has_alice⊕has_not did not beat the ~0.75 OR baseline.
  - Implications: (i) probes should then fail under small distribution shift; (ii) a probe can latch onto b through a⊕b even when b is constant in training; (iii) a model with N features would represent on the order of 2^N.
  - Commenters could not always replicate. Fabien Roger's toy MLPs found incidental XORs mainly when features are highly redundant.
  - Source: [LessWrong post](https://www.lesswrong.com/posts/hjJXCn9GsskysDceS/what-s-up-with-llms-representing-xors-of-arbitrary-features)
- **Boolean-function analysis of threshold units** **[background, not opened]**:
  - O'Donnell's *Analysis of Boolean Functions* (2014) shows that Chow parameters (degree-0 and degree-1 Fourier coefficients) uniquely determine a linear threshold function.
  - Peres's theorem bounds the noise sensitivity of any LTF at O(√δ).
  - Both give per-neuron tests of "is this unit a threshold gate, and which one" that need no labelled concepts.
  - Source: [O'Donnell book (free online)](https://www.cs.cmu.edu/~odonnell/aobf/)

### Inferences
- **Project finding (1) maps onto Petersen.** DLGN avoids the margin problem by construction, because the relaxed gates are multilinear in probabilities and then hardened. The project's sigmoid gates must instead earn margin. A testable bridge is to compare the relaxed-to-hard gap and the maximum trainable depth for (a) DLGN-style gates and (b) sigmoid threshold gates trained with versus without a margin penalty. Petersen's 8–10-layer depth limit is a published reference point for the project's "deep OR-trees fail" result.
- **Mu & Andreas gives a ready-made, falsifiable labelling protocol** for "is unit k the engineer's XOR, AND or carry gate". Enumerate Boolean formulas over the input bits a_k, b_k, c_{k−1} up to depth 2–3 and report the best IoU per unit against a random-rotation null. Applied to the project's addition networks, the carry-as-direction result predicts that no unit exceeds a pre-set IoU for carry, while a probe direction does.
- **Marks's RAX is the key confound for carry-as-direction.** If XOR-like and MAJ-like combinations of the input bits are linearly decodable everywhere, including in random networks (project finding 2: carry readable even in random nets), then decodability is cheap. Causal use is what distinguishes a trained network (covered: interchange/DAS). Marks's reset-network control (weight shuffling) is a standard null worth copying alongside random initialisation.
- **On the essay's "neurons are gates" framing:** a falsifiable form is that each unit's Chow parameters and its sharpness match a single LTF. The project's gain and margin measurement is the analog-electronics version of the same test.

### Gaps
- I found no paper that applies Mu & Andreas-style Boolean-formula search to arithmetic or carry networks. I found no peer-reviewed follow-up to Marks's RAX post. I did not verify whether Petersen's later work reports relaxed-versus-hard gaps for deep (>10-layer) networks.

---

## 4. Noise margins and robustness in analog computation: is there a neural analogue of the static discipline?

### Takeaway
The carry review already notes that no source connects the static discipline to sigmoid nets. The closest neural-network formalisms are mean-field signal-propagation theory (order/chaos set by the gain χ) and noise-sensitivity bounds for threshold functions. Neither is framed as noise margins, so the project's "gain > 1 at threshold plus noise margins" criterion remains a novel bridge. Each component, however, has a citable analogue.

### Cited Findings
- Covered in the carry review: MIT 6.111 static discipline; Agarwal & Lang; von Neumann's restoring organs; Hajnal et al. on imprecise threshold gates. — see the carry review.
- **Signal propagation in deep random networks** **[background, not opened]**:
  - Poole et al. (NeurIPS 2016) and Schoenholz et al. (ICLR 2017) show that for tanh-like nets the quantity χ₁ = σ_w² E[φ′(h)²] separates an ordered phase (χ₁ < 1, perturbations shrink) from a chaotic phase (χ₁ > 1, perturbations grow). Trainable depth scales with the correlation length near χ₁ = 1.
  - This is the closest formal analogue to "gain at the operating point" controlling error accumulation with depth.
  - Sources: [arXiv:1606.05340](https://arxiv.org/abs/1606.05340); [arXiv:1611.01232](https://arxiv.org/abs/1611.01232)
- **Petersen et al. depth limit** **[opened]**: DLGNs train up to about 8–10 layers before vanishing gradients. "Too many operators can lead to vanishing probabilities". — [ar5iv 2210.08277](https://ar5iv.labs.arxiv.org/html/2210.08277)
- **Noise sensitivity of threshold functions** **[background, not opened]**: Peres's theorem gives NS_δ(LTF) = O(√δ), which bounds how much a single threshold gate amplifies input bit-flips. The Benjamini–Kalai–Schramm framework relates noise sensitivity to influences. — [O'Donnell, Ch. 5](https://www.cs.cmu.edu/~odonnell/aobf/)

### Inferences
- **The two regimes do different things.** In digital logic, a gate needs gain > 1 in the transition region and gain < 1 near the rails, so that it restores logic levels (regeneration). The mean-field χ is an average gain over the input distribution. A sigmoid gate trained with large margin has very low average φ′ on-distribution, because inputs sit in saturation (ordered, restoring), and high φ′ only at threshold. That matches the static discipline. A gate trained without margin sits near the linear regime, where perturbations neither shrink nor are cleaned up.
- **Falsifiable prediction.** For a depth-D OR-tree of trained sigmoid gates, output error under input noise σ should grow roughly geometrically, with ratio set by the measured on-distribution gain. It should stay bounded once each gate's noise margins NM_L = V_IL − V_OL and NM_H = V_OH − V_IH are positive and exceed σ. The standard figures would be measured transfer curves per gate, plus error versus depth for margin-trained and unregularised gates.
- **Analog latches.** A latch is a positive-feedback loop that needs loop gain > 1 at the metastable point to have two stable states. This is the same χ > 1 condition applied locally, which explains why latches fail without margin training.

### Gaps
- I found no paper that measures VIL/VIH/VOL/VOH-style transfer characteristics for trained neural units, or that frames signal propagation as noise margins. The mean-field and noise-sensitivity results were not opened this session and should be checked.

---

## 5. Arithmetic and algorithm circuits; causal-abstraction methodology

### Takeaway
This area is almost entirely covered by the carry review (Quirke & Barez, Kruthoff, Nanda, Clock/Pizza, Tracr/RASP, DAS, illusions, Sutter et al.). The useful additions are older learned-arithmetic architectures (Neural GPU, NALU), which are the natural "end-to-end networks that learn binary addition" baselines for project finding (2). Also useful is one extra over-powerful-probe warning (Marks's RAX, in §3).

### Cited Findings
- Covered: all addition and modular-arithmetic transformer mechanism papers; RASP and Tracr; DAS and causal abstraction; the Makelov illusion and its reply; the Sutter et al. "non-linear representation dilemma", which gives 100% IIA in random LMs; CausalGym's DAS-at-initialisation warning. — see the carry review.
- **Neural GPU (Kaiser & Sutskever, ICLR 2016)** learns binary addition and multiplication with length generalisation, using convolutional-GRU recurrence **[background, not opened]**. — [arXiv:1511.08228](https://arxiv.org/abs/1511.08228)
- **NALU (Trask et al., NeurIPS 2018)** builds arithmetic-biased units. Madsen & Johansen (ICLR 2020) show NALU's instability and propose improved units **[background, not opened]**. — [arXiv:1808.00508](https://arxiv.org/abs/1808.00508); [arXiv:2001.05016](https://arxiv.org/abs/2001.05016)
- **Learning Transformer Programs (Friedman, Wettig & Chen, NeurIPS 2023)** trains transformers constrained to be decompilable into RASP-like programs. It is the "trained but discrete" counterpart to Tracr's compiled weights **[background, not opened]**. — [arXiv:2306.01128](https://arxiv.org/abs/2306.01128)

### Inferences
- For finding (2), the strongest falsifiable design that the literature supports is already in the carry review: interchange or DAS with random-init controls on the interchange metric. A new addition is Mu & Andreas Boolean search plus Gurnee-style cross-seed correlation against the engineered ripple-carry gates (§2, §3), with Sutter et al.'s rule that the alignment map stays linear and low-dimensional.

### Gaps
- I did not search further for binary-addition mechanistic studies, since the carry review already reports that none exist in peer-reviewed venues.

---

## 6. Tool use and non-differentiable modules: why would a frozen gate-built CPU head need REINFORCE?

### Takeaway
The closest precedent for project finding (3) is Zaremba & Sutskever's RL-NTM. It trains discrete interface decisions with REINFORCE and the continuous parts with backprop, which is exactly the split the project found necessary. Toolformer, neural module networks (REINFORCE over layouts) and hard attention (REINFORCE over locations) are the other standard reference points. Adaptive Computation Time's ponder cost is the standard template for a per-call cost.

### Cited Findings
- **Zaremba & Sutskever (2015/16), "Reinforcement Learning Neural Turing Machines"** **[snippet, multiple]**:
  - The paper notes that discrete interfaces cannot be trained by backprop because they are not differentiable.
  - The RL-NTM "learns to make discrete decisions using the Reinforce algorithm, and learns to produce continuous outputs using backpropagation". It uses memory, input and output tapes, which make it Turing complete, and is evaluated on simple algorithmic tasks.
  - Sources: [arXiv:1505.00521](https://arxiv.org/abs/1505.00521); [Google Research](https://research.google/pubs/reinforcement-learning-neural-turing-machines/)
- **REINFORCE (Williams, 1992)** is the score-function estimator **[background, not opened]**. — [Machine Learning 8:229–256, DOI 10.1007/BF00992696](https://doi.org/10.1007/BF00992696)
- **Hard attention with REINFORCE** **[background, not opened]**: Mnih et al. (NeurIPS 2014), "Recurrent Models of Visual Attention", introduced the glimpse location policy trained with REINFORCE and a baseline. Xu et al. (ICML 2015), "Show, Attend and Tell", compares hard (stochastic, REINFORCE) and soft (deterministic) attention. — [arXiv:1406.6247](https://arxiv.org/abs/1406.6247); [arXiv:1502.03044](https://arxiv.org/abs/1502.03044)
- **Neural Module Networks** **[background, not opened]**: Andreas et al. (CVPR 2016) assemble modules from a parse. "Learning to Compose Neural Networks for QA" (Andreas et al., NAACL 2016) learns the layout policy with REINFORCE and the module weights with backprop. — [arXiv:1511.02799](https://arxiv.org/abs/1511.02799); [arXiv:1601.01705](https://arxiv.org/abs/1601.01705)
- **Toolformer (Schick et al., NeurIPS 2023)** **[background, not opened]**: self-supervised tool-call insertion. A sampled API call is kept only if adding its result reduces the LM loss on later tokens, so "the tool's result was useful" acts as a filter-style reward. — [arXiv:2302.04761](https://arxiv.org/abs/2302.04761)
- **Differentiable memories** **[background, not opened]**: NTM (Graves et al., 2014) and DNC (Graves et al., Nature 2016) are the soft-addressing alternative to discrete interfaces. — [arXiv:1410.5401](https://arxiv.org/abs/1410.5401); [Nature 538:471](https://www.nature.com/articles/nature20101)
- **Relaxations and estimators** **[background, not opened]**: Gumbel-softmax/Concrete (Jang et al.; Maddison et al., ICLR 2017) and the straight-through estimator (Bengio et al., 2013) are the standard alternatives to REINFORCE for discrete choices. — [arXiv:1611.01144](https://arxiv.org/abs/1611.01144); [arXiv:1308.3432](https://arxiv.org/abs/1308.3432)
- **Call cost** **[background, not opened]**: Adaptive Computation Time (Graves, 2016) adds a differentiable "ponder cost" that trades computation against accuracy. — [arXiv:1603.08983](https://arxiv.org/abs/1603.08983)
- **Computers inside networks** **[background, not opened]**: Looped Transformers as Programmable Computers (Giannou et al., ICML 2023) builds a computer from transformer weights. Together with Tracr (covered), it is the "compiled" opposite of learning to call a frozen module. — [arXiv:2301.13196](https://arxiv.org/abs/2301.13196)

### Inferences
- **Why REINFORCE and not soft attention.** A frozen gate-built CPU computes on discrete bits. A soft mixture of pointer addresses feeds it non-Boolean inputs, which by finding (1) fall inside its noise margins and give garbage or flat gradients. A reward on output correctness sidesteps the need for a gradient through the gates. This is the RL-NTM argument specialised to gates.
- **Falsifiable design.** Compare four pointer-training estimators: soft attention, STE, Gumbel-softmax with annealing, and REINFORCE with a baseline. Hold the frozen CPU and task fixed, report success rate over seeds, and test whether the soft-attention failure disappears when the CPU's gates are "softened" (low gain). If it does, the failure is caused by the static discipline and not by the estimator.
- **Call cost and memorisation.** An ACT-style per-call penalty λ should give a phase diagram in (λ, train-set size). Below a threshold the GPT calls the tool. Above it, the GPT memorises the training-set answers and fails on held-out operands. The held-out-operand split is the standard memorisation test, as in the length and OOD splits of the carry review's arithmetic papers.
- **Interpretability hook to Cammarata.** If the essay is right that neurons are "gates", a GPT that learns to call a gate-built CPU should, after training, contain units that read the CPU head's output cleanly (one main job). This can be checked with the essay's own audit protocol (poles, top activations, downstream weights, steering).

### Gaps
- I opened none of the tool-use or hard-attention papers this session except via snippets for RL-NTM. I found no paper that embeds a frozen, gate-level CPU as an attention head and compares estimators, which suggests this setting is new, though I did not search exhaustively (e.g. "neural CPU", "differentiable computer as attention head").

---

## 7. Cross-area summary: strongest result, open question and standard design per area

### Takeaway
In every area the established result is either "the representation is decodable" or "a discrete thing can be trained via relaxation or RL". What is open is whether learned networks implement engineer-recognisable gates at the neuron level. That question is where Cammarata's "already clean" thesis and the project's carry-as-direction result directly disagree.

### Cited Findings
- Universality: the strongest quantitative result is that 1–5% of GPT-2 neurons are universal across seeds ([arXiv:2401.12181](https://arxiv.org/abs/2401.12181)). The strongest qualitative result is cross-model curve and pose neurons ([Distill Zoom In](https://distill.pub/2020/circuits/zoom-in/); [already-clean.goodfire.com](https://already-clean.goodfire.com/)). The open question is strong versus weak universality ([arXiv:2302.03025](https://arxiv.org/abs/2302.03025)).
- Neurons as logic: the strongest result is that DLGNs harden with a gap under 0.1% on MNIST ([arXiv:2210.08277](https://arxiv.org/abs/2210.08277)). The standard labelling design is IoU over Boolean formulas ([arXiv:2006.14032](https://arxiv.org/abs/2006.14032)). The open question is whether linearly decodable XORs mean XOR computation ([LessWrong RAX](https://www.lesswrong.com/posts/hjJXCn9GsskysDceS/what-s-up-with-llms-representing-xors-of-arbitrary-features)).
- Discrete modules: the strongest precedent is the RL-NTM ([arXiv:1505.00521](https://arxiv.org/abs/1505.00521)).

### Inferences
- **Proposed core bridge experiment.** It has three pre-registered predictions, each with the null that falsifies it.
  - **(P1) Neuron cleanliness.** Apply Cammarata's audit and Mu & Andreas IoU search to the project's addition networks across ≥ 5 seeds, at widths and regularisation that meet the essay's "well-trained" bar. Cammarata predicts a carry unit with one main job. The project predicts a best-unit IoU or R² well below the gate reference and below a direction probe, with the probe surviving a random rotation.
  - **(P2) Development.** Track how the carry direction aligns with the neuron basis over checkpoints (kurtosis of its neuron-basis coefficients). Recursive generalization predicts alignment that sharpens and then grows offshoots. Superposition predicts persistent misalignment.
  - **(P3) Gate discipline.** For units that do look gate-like, measure transfer-curve gain and noise margins. Predict that only units satisfying the static discipline can be composed by downstream layers to depth > k. This uses Petersen's 8–10-layer limit and the mean-field χ as external reference points.
- Controls the literature demands: random-init and weight-shuffle nets (Marks; covered random-init work), a random-rotation baseline for neuron-basis claims (Gurnee-style), linear low-dimensional alignment maps only (Sutter et al., covered), and noising versus denoising stated explicitly (Heimersheim & Nanda, covered).

### Gaps
- The essay's exact audit thresholds and sample sizes, needed to pre-register P1 fairly, were not available in the portion I read and may only appear in the forthcoming technical paper.
