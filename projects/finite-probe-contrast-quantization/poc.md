# Research and proof-of-concept plan: finite-probe contrast quantization

**Research date:** 29 October 2026  
**Target:** a single MacBook Pro with M3 Max, with its actual memory and operating system discovered before configuration  
**Status:** research handoff, with targeted source audits and small validation experiments completed; the proposed quantizer and confirmatory experiments have not been executed  
**Primary decision:** test FPCQ with one disturbance amplitude per example shared across output channels, using an exact forest adversary, against strong reconstruction, expected-noise, and modern rounding controls.

## Reading and execution contract

This document is a standalone specification. It includes the mathematical mechanism, baseline recipes, data protocol, proposed interfaces, decision gates, and reporting rules. The original attachments help reproduce historical results but are not required to implement the new experiments. Reference identifiers such as [Q01] resolve in Section 12. Equations and algorithms marked **deduction** are derived here; they are not claimed as established historical novelties.

The coding agent may make routine, reversible implementation decisions. Changes to the scientific hypothesis, data population, protected splits, primary metric, quantization domain, baseline information, or inference format require an entry in the methodological amendment log before affected evaluation. Such a change is not permission to reuse an already examined test set as fresh confirmation.

Four evidence categories apply throughout:

- **Inspected:** directly read in an attachment, primary paper, official repository, or official documentation.
- **Measured here:** a computation actually performed during preparation, in the Linux CPU environment described below.
- **Deduction or estimate:** mathematics, a design recommendation, or a resource projection.
- **Proposed:** work for the coding agent; no outcome is promised.

All project command names beginning with **python -m fpcq** are proposed commands to implement. They do not exist in the uploaded project. Commands explicitly identified as system commands or original-archive commands already exist.

---

## 1. Executive assessment and recommended direction

### 1.1 Verdict

The attachments contain a legitimate mathematical and computational research prototype, but they do not establish a useful general quantization improvement. The scalar forest adversary is sound under its stated assumptions, and the supplied small-model results are reproducible. The main empirical uncertainty is whether the finite uncertainty model improves out-of-sample quantization after accounting for stronger baselines, optimizer effort, and calibration cost.

**Recommended primary direction:** retain measured upstream error directions, finite independent amplitudes, teacher-output contrasts, fixed affine integer grids, and an exact forest adversary. Replace the existing sum of separately maximized scalar-output losses with the exact vector-output loss in which each example has a single amplitude shared by every output channel. This removes a disclosed modeling relaxation without changing the model representation or adding inference parameters.

This is a **narrow methodological refinement and empirical research question**, not a new tree-inference theorem. Its most credible potential value is better code selection at low bit widths without the artificial conflict of channel-specific disturbances. It may offer no useful improvement: the measured relaxation gap in the existing final layers is only about 1.1–1.7%, and the coherent optimizer loses the original implementation's independent-row batching. A mathematical ranking reversal proves that the distinction can matter; it does not show that it matters enough in practice.

The initial meaningful workload is a new Fashion-MNIST dense classifier, with a fixed-prefix final-layer experiment first and a complete three-layer quantization experiment second. This is deliberately a quantization-component study on real data. A Pythia-70M transfer experiment is conditional on the measured scientific and computational gates. Neither a Fashion-MNIST result nor a Pythia-70M result establishes competitiveness for modern multibillion-parameter language models.

### 1.2 What the supplied evidence actually supports

The original architecture is 64 → 64 → 64 → 10 with ReLU hidden layers, 8,832 weights, three teacher seeds, and a shared 540-example digits test split. Calibration uses 128 examples from teacher training. The following means were independently recomputed from the saved models [U1].

| Method, 2 bits | Accuracy | Centered-logit MSE | Teacher-to-student KL, nats/example | Recorded calibration seconds |
|---|---:|---:|---:|---:|
| Extra nominal refinement | 97.4691% | 0.544899 | 0.010840 | 0.13867 |
| Contrast loss without robust radius | 97.4074% | 0.528409 | 0.010709 | 0.30152 |
| Original FPCQ | 97.3457% | 0.518636 | 0.009075 | 2.81201 |
| In-package GPTQ-style reference | 97.0988% | 0.769235 | 0.016431 | 0.00734 |

Original FPCQ reduces MSE by 4.82% and KL by 16.29% against extra nominal refinement, while accuracy falls by 0.1235 percentage points and recorded calibration time rises 20.28-fold. At 3 bits its KL is 0.002690 versus 0.002544 for contrast-only and 0.002172 for the GPTQ-style reference. At 4 bits its KL is 0.000443 versus 0.000390 for the reference. These are descriptive results on already inspected data, not confirmatory evidence or official GPTQ reproduction.

A separate issue affects the reported stress scores: each method is evaluated using its own upstream error direction. That changes the uncertainty set between methods. It is an informative descriptive diagnostic, but it cannot isolate which method handles the same disturbance better. The new mechanism comparison must hold the prefix and disturbance set fixed.

### 1.3 Work completed during this investigation

The preparation environment was Linux x86_64, Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0, and scikit-learn 1.8.0. No M3 Max was available, no new teacher was trained, and no language-model quantization was run.

| Check performed here | Result | What it does not establish |
|---|---|---|
| Rerun supplied scalar oracle tests | 750 exhaustive comparisons on 250 forests passed; maximum error 7.11 × 10⁻¹⁵; 5,000 interior probes bounded | Formal floating-point verification or unseen-input robustness |
| Reevaluate all saved quantized models | All 81 natural accuracy/MSE/KL evaluations matched exactly | New independent confirmation |
| Load exports and inspect histories | 243 layers loaded; 297 histories nonincreasing; largest stored recomputation error about 9.99 × 10⁻¹³ | Global or converged local optimality |
| Repeat seed-0, 2-bit original calibration from its saved teacher | Identical codes in all three layers; 3.399 seconds in this environment | Teacher-training reproduction or M3 timing |
| Independent vector forest oracle and cached row updates | 200 random problems, N ≤ 8 and m ≤ 5, agreed with exhaustive corners to 1.42 × 10⁻¹⁴ | A completed vector quantizer or task improvement |
| Fixed-grid counterexample | Scalar-row relaxation and vector objective rank two candidates differently | Typical-case prevalence |
| Independent strip-theorem checks | 11 finite-grid LPs matched the stated frontier; 325,000 sampled edges had no positive bound violation; the six-edge moment certificate gave (0, 0, 1/2) exactly | A continuum proof, novelty proof, or quantization guarantee |

The PDF's claimed larger verification suite was not supplied in this ZIP and was not rerun. The newly performed strip checks were separately implemented from the mathematical specification.

### 1.4 Selection among credible directions

| Direction | Why pursue or reject now | Decision |
|---|---|---|
| Scale original rowwise FPCQ immediately | Existing validity and fairness questions would be magnified; observed costs are already substantial | Reproduce as a baseline, not the primary escalation |
| Shared vector-output forest objective | Preserves central intent, fixes an identifiable relaxation, exact oracle remains cheap per candidate, clean falsification possible | **Primary candidate** |
| Expected-noise quadratic calibration | Same discrepancy directions and graph, much cheaper; may explain all gains | **Mandatory strong alternative and preferred simplification if equally good** |
| Joint no-edge finite robustness | Exact closed form, eliminates graph cost and graph semantics assumptions | Mandatory ablation and possible simpler endpoint |
| Denser bipartite graph with graph cuts | Valid scalar result does not automatically extend to vector couplings | Defer; do not use as a shortcut |
| New compact-domain straight-probe theorem | Scientifically distinct mathematical question, potentially valuable even if quantization fails | Conditional theory branch, not evidence of practical gain |
| Immediate large-LLM implementation | No local cost calibration or validated semantic-pair construction | Defer until explicit gates pass |

---

## 2. Problem statement, assumptions, and hypotheses

### 2.1 Beneficiaries and task

The intended beneficiaries are researchers and engineers choosing low-bit weight codes for a pretrained model while retaining an ordinary groupwise affine deployment format. The relevant workflow is: obtain a full-precision teacher; collect calibration activations; sequentially quantize layers; export codes/scales; evaluate the whole student.

The bottleneck under investigation is **accuracy of offline code selection when earlier quantized layers alter the inputs of later layers**. Accumulated mismatch and finite-calibration overfitting are established concerns in asymmetric PTQ [Q02–Q04]. Whether the particular ray uncertainty and contrast graph in FPCQ address an important remaining bottleneck is unproven.

There is no demonstrated new compression ratio: every compared fixed-format method can emit the same number of code bits. There is no inherent per-token speed mechanism: after export, inference is the same architecture and format. A useful outcome can still be improved fidelity at the same bytes, provided the offline cost is justified.

### 2.2 Inputs, outputs, and notation

For one frozen linear layer:

| Symbol | Definition and shape |
|---|---|
| W★ | Original weight matrix, m × d; the star is a label, not transpose |
| Hᵀ, Hᑫ | Teacher and quantized-prefix inputs on aligned examples, N × d |
| E = Hᑫ − Hᵀ | Measured input discrepancy, N × d |
| Y = Hᵀ(W★)ᵀ | Teacher pre-bias targets, N × m |
| Q | Integer weight codes, m × d |
| S, Z | Per-output/per-group scale and integer zero point |
| W(Q) | Dequantized candidate weights |
| G = (V, E_G) | A simple undirected forest on the N examples |
| ωᵢ, βᵢⱼ | Nonnegative node and edge weights |
| ρ | Nonnegative disturbance radius |
| αᵢ ∈ [1 − ρ, 1 + ρ] | One amplitude per example |
| A = HᑫW(Q)ᵀ − Y | Natural reconstruction residual, N × m |
| B = EW(Q)ᵀ | Output response to the measured discrepancy, N × m |

Biases are kept unchanged. They cancel in this local reconstruction error; they still participate in the actual network forward pass. Activations remain floating point. The selected experiment is weight-only post-training quantization.

Use the word **contrast** for an output difference between two selected examples. The graph loss preserves the teacher's difference; it does not force the outputs to agree. No language-semantic interpretation follows merely from using this name.

### 2.3 Original and selected objectives

For residual vector rᵢ(αᵢ) = Aᵢ + (αᵢ − 1)Bᵢ, define

\[
J_G(Q;\rho,\lambda)=\frac1m\max_{\alpha\in[1-\rho,1+\rho]^N}
\left[\sum_i\omega_i\|r_i(\alpha_i)\|_2^2+
\sum_{ij\in E_G}\beta_{ij}\|r_i(\alpha_i)-r_j(\alpha_j)\|_2^2\right].
\tag{1}
\]

Default normalization is ωᵢ = 1/N and βᵢⱼ = λ/|E_G|. If there are no edges, omit the edge term. This keeps λ interpretable as a ratio of averaged contrast error to averaged example error. It does not make different graphs or datasets statistically interchangeable.

The original implementation instead uses

\[
U_G(Q)=\frac1m\sum_{o=1}^{m}\max_{\alpha}
\left[\sum_i\omega_i r_{io}(\alpha_i)^2+
\sum_{ij}\beta_{ij}(r_{io}(\alpha_i)-r_{jo}(\alpha_j))^2\right].
\tag{2}
\]

For identical inputs and weights, J_G ≤ U_G. Each channel in (2) may select a different amplitude vector. The refinement replaces (2) by (1). All other changes must be separately identified.

Even (1) is not a completely physical adversary: independently choosing amplitudes for different examples need not correspond to any single realizable quantized prefix. The refinement ensures consistency across output channels, not realizability of every corner by a real model.

### 2.4 Facts, assumptions, and hypotheses

**Established or verified here**

- Fixed-grid code choice can materially affect model fidelity; nearest rounding is not a strong final comparator [Q01–Q06].
- On a forest, a squared affine vector loss over an independent amplitude box has an exact two-state endpoint adversary; Section 4 gives the proof.
- The supplied experiments and scalar oracle reproduce as described in Section 1.
- Output-aware rounding, asymmetric targets, contrast/distillation objectives, robust least squares, and coordinated rounding all predate this proposal [Q01–Q12, M01–M03].

**Mechanism assumptions**

- Some variation along observed discrepancy directions predicts harmful held-out mismatch or supplies useful regularization.
- Preserving the selected contrasts adds information beyond weighted ordinary reconstruction.
- The scalar-row relaxation changes useful code decisions often enough to justify coherent optimization.
- A local layer objective correlates sufficiently with whole-model teacher fidelity.
- Optimizer quality and calibration cost can be made competitive on a useful local scale.

| ID and hypothesis | Supporting evidence | Alternatives and discriminating test | Weakening/falsifying observation |
|---|---|---|---|
| H1: shared output amplitudes improve code selection over the rowwise relaxation | Exact ranking reversal; artificial channel disagreement is removable | Same prefix, graph, grid, warm start, move family, and budget; measure choices, J, U, and held-out KL | Almost no code changes; no held-out gain despite adequate sensitivity; excess cost dominates |
| H2: finite minimax adds value beyond nominal reconstruction | Original 2-bit fidelity improvement | Contrast-only, joint no-edge, and exact expected-noise quadratic controls; random-direction control | Gain disappears with these controls or only exists on the training adversary |
| H3: measured discrepancy directions are useful | They come from an actual quantized prefix | Cross-calibration-prefix projection test; norm-matched scrambled directions | Directions predict no more than scrambled/null controls; useful error variation is mostly orthogonal |
| H4: the contrast graph contributes | Original contrast-only improvement | Degree/edge-count controls; pair provenance; λ = 0 | Arbitrary graphs do as well, or graph gains vanish after stronger quadratic optimization |
| H5: lower local robust loss transfers to model fidelity | Plausible but not established | Fixed-prefix final layer, then all layers; KL primary with accuracy and cost guardrails | Only synthetic stress improves; clean KL/accuracy worsen |
| H6: local benefits survive practical costs | Export adds no model parameters | Total calibration time and quality/bytes/time frontiers; optional identical packed kernels | Equal-cost simple methods dominate; storage mapping changes codes; no amortization use case |

Positive results on H1 do not establish H2–H6. A failed H1 does not disprove the strip theorem or every use of structured robust calibration.

### 2.5 Strongest case for and against the idea

**For:** low-bit rounding is a discrete optimization problem with real propagation errors. A bounded uncertainty objective may select codes that tolerate plausible changes along measured discrepancy directions, and the shared vector formulation removes an artificial disagreement between output-channel adversaries. Structured contrasts may preserve distinctions neglected by elementwise MSE. Forest structure makes the inner maximization exact; the final representation remains ordinary and inexpensive to deploy.

For this selected affine-layer model, rᵢ(αᵢ) = Aᵢ + (αᵢ−1)Bᵢ exactly: its derivative Bᵢ determines every finite change. The proposal therefore has **no finite-versus-tangent approximation advantage at this layer**. Its hypothesized value is the bounded, coupled loss and resulting regularization. The PDF's nonlinear finite-probe phenomenon is a separate mathematical motivation, not a property demonstrated by this linear PoC.

**Against:** the natural prefix discrepancy is already known at α = 1, so a box around it may manufacture uncertainty. It can penalize helpful compensation, overemphasize unlikely corners, and overfit a hand-built graph. A graph loss at ρ = 0 is merely a quadratic reconstruction loss on augmented examples. Expected perturbations likewise reduce to a cheap quadratic. Joint row coupling can erase the computational advantage of the existing implementation. A better local proxy may still ignore downstream nonlinear sensitivity, which methods such as YAQA explicitly address [Q05].

**Headroom:** the original two-bit fidelity gains coexist with slightly worse accuracy and roughly twenty-fold calibration cost. Removing the rowwise relaxation produces only a small measured incumbent gap, which is not a uniform optimization bound. No inference-speed headroom should be attributed to code selection alone. Practical relevance must therefore be shown as a measured quality/size/calibration-cost tradeoff, not inferred from an exact inner solver.

---

## 3. Research findings and precise novelty assessment

### 3.1 Closest prior art

The matrix separates direct quantization alternatives from underlying optimization tools. Published performance belongs to each source's own conditions; none is a controlled comparison with this project.

| Work | Problem and mechanism | Assumptions/evidence and computational requirements | Reusable implementation and exact overlap |
|---|---|---|---|
| GPTQ [Q01] | Sequential second-order rounding with compensation | Layer reconstruction; large-model studies; dense Gram/factorization | Official code available with CUDA-specific calls; useful CPU dense port; compensation is established |
| AdaRound / BRECQ [Q06, Q07] | Learned rounding and block reconstruction | Calibration optimization, floor/ceil or block-local objectives; CNN experiments | Official implementations; asymmetric quantized-prefix targets already appear in AdaRound |
| GPTAQ [Q02] | Efficient asymmetric compensation | Accumulated prefix mismatch; neuron-decomposed update | Official code/paper; teacher targets with student-prefix inputs are established |
| ResComp [Q03] | Corrects compensation-dependent residuals | Original-versus-working target discrepancy; calibration sensitivity | Official repository; its neuron decomposition is not a global exact-target solve |
| CoreQ / K-CoreQ [Q04] | Mismatch correction and beam rounding | Current v2 equations; Appendix A.2 | Local port; official code unverified |
| YAQA [Q05] | Output-aware rounding | Current equations; Appendix A.3 | GPL-3.0 QTIP-based code; local scalar specialization |
| DiscQuant [Q08] | Discrepancy-inspired restricted rounding and model-output preservation | Full-model calibration, restricted feasible directions | Author implementation; rules out generic coordinated/output-preserving novelty |
| CL-Calib; relational distillation [Q09, Q10] | Contrastive calibration or relations between examples | Different losses and training settings | Demonstrates earlier use of information/relational preservation; no exact forest adversary identified |
| QDrop [Q11] | Randomly suppresses activation quantization during reconstruction | Low-bit activation/weight calibration; stochastic mismatch exposure | Official implementation; motivates cheaper amplitude-mixing controls, not a claimed identical method |
| TFIC [Q12] | Barrier-aware coordinated/cluster changes | October 2026 publication describes escape from single-code barriers | Publication inspected, benchmark not reproduced; pair/cluster moves cannot be claimed new |
| Robust least squares [M01, M02] | Bounded uncertain data and worst-case residuals | Classical robust optimization; structure governs tractability; quantized-data specialization | Reuse mathematical framework; bounded uncertainty and robust-versus-regularized tradeoffs are established |
| Tree max-sum and graph cuts [M03, M04] | Exact inference on trees; submodular binary minimization | Tree DP linear in graph size; cuts require sign/submodularity conditions | Reuse standard algorithms; no novelty claim for the solver primitive |

The source's in-package GPTQ-style reference is a diagnostic implementation. It is neither the strongest contemporary local comparison nor evidence against tuned official GPTQ.

### 3.2 What is, and is not, differentiated

The unlocated combination is: **measured upstream quantization directions + independent finite per-example amplitudes + preserved teacher output differences + fixed affine codes + an exact structured adversary**. The selected refinement adds **one amplitude shared across all outputs** and a controlled evaluation of the cost of the rowwise relaxation.

Historical novelty is unresolved. The shared vector formulation follows immediately from ordinary vector least squares and tree inference; calling it a new general theorem would overstate the contribution. Possible defensible outcomes are:

1. A useful quantization-specific objective with replicated gains over appropriate controls.
2. A new empirical finding about when robust, expected-noise, or asymmetric correction works.
3. An engineering improvement in exact calibration cost, if independently measured.
4. A reproducibility/negative-result contribution showing that the relaxation or finite probes do not matter in realistic local settings.

A prior paper, patent claim, or implementation using the same uncertainty factorization and graph loss could invalidate methodological novelty even if this implementation is new. A cheaper equivalent augmented least-squares objective that reproduces all gains would substantially narrow practical novelty.

### 3.3 Reuse decisions

Reuse validated grid/packing concepts, original scalar fixtures, ordinary tree DP, BLAS, standard linear algebra, public dataset readers, and authoritative baseline equations. Correct source edge cases before reuse. Keep adapters around baseline implementations instead of forcing their internals into the FPCQ optimizer.

Do not reuse the original test split as confirmation. Do not call the source ZIP open-source: no license notice was identified. Authorized local analysis can proceed; any redistribution of supplied code needs a documented basis. The standalone mathematics and proposed interfaces allow a clean implementation without depending on redistribution.

### 3.4 Role of the straight-probe PDF

The PDF studies a different objective: a readout F compared with itself under
T_t(x,y) = (x+t, y+tx), while requiring F(x,y+s)−F(x,y) ≥ μs on |x| ≤ L. It gives the strip optimum

\[
\mathcal E_\mu(L,r)=
\begin{cases}
\mu r^2/2,&0\le r\le2L,\\
2\mu L^2,&2L\le r\le3L,\\
\mu L(r-L),&r\ge3L,
\end{cases}
\tag{3}
\]
with zero value for L = 0 or r = 0 [U2].

The coordinate change u = x, v = y − x²/2 turns a probe into
(u,v) ↦ (u+t,v−t²/2). Two-edge and four-edge paths give lower bounds; an explicit continuous potential attains them. We inspected that proof and independently checked its implementation-level consequences. The domain is unbounded in y, and the lower slope is global on the source strip. A finite compact calibration set does not inherit this minimax floor.

The higher-dimensional logarithmic-coordinate construction reduces the global problem to a prescribed-leading-coefficient polynomial approximation problem. Its Chebyshev/optimal-design constants overlap established results by Dette, Melas, and Shpilev; closed-path approximation and cycle-mean methods are also classical [M05–M07]. Failure of a straight Euler step to preserve a nonlinear first integral is familiar in geometric integration [M08].

The precise strip frontier and its weaker semantic-slope formulation remain unlocated in the searched sources, not established historically novel. More importantly, **self-invariance and teacher-student matching are different optimization problems**. Neither the transitions at 2L and 3L nor the PDF's factor-four example prescribe a neural quantization radius or certify task accuracy. The PDF supplies motivation for finite perturbations and a separable mathematical research branch.

---

## 4. Selected mechanism and implementation specification

### 4.1 Exact forest oracle: proof and conventions

For a fixed candidate Q, the loss in (1) is a nonnegative sum of squared affine functions of α. It is convex in each coordinate. Holding the other coordinates fixed, a maximum on an interval occurs at an endpoint; applying this successively yields an attaining corner. Thus set αᵢ = 1 + ρsᵢ, sᵢ ∈ {−1,+1}.

Form vector values and scalar tables:

\[
v_i(s)=A_i+\rho s B_i,\quad
u_i(s)=\omega_i\|v_i(s)\|^2/m,\quad
c_{ij}(s,t)=\beta_{ij}\|v_i(s)-v_j(t)\|^2/m.
\tag{4}
\]

Root every component. For node i with parent p,

\[
M_{i\to p}(t)=\max_{s\in\{-1,+1\}}
\{u_i(s)+c_{ip}(s,t)+\sum_{k\in\operatorname{children}(i)}M_{k\to i}(s)\}.
\tag{5}
\]

Maximize at each root and sum over components. Child subproblems have disjoint variables, which proves the recursion. Store an argmax for each parent state and backtrack to obtain an attaining shared amplitude vector. Ties select state −1 deterministically.

Table construction costs O(m(N+|E_G|)); DP costs O(N+|E_G|). Edge tables must be transposed when traversal orientation reverses the stored edge. Forest roots and adjacency order must be deterministic.

~~~python
# Proposed reference interface; float64 CPU arrays.
def forest_oracle(node_cost, edge_cost, edges, parent, edge_to_parent,
                  postorder, roots, return_witness=True):
    # node_cost[N,2]; edges[M,2]; edge_cost[M,2,2] in edges' endpoint order.
    dp = node_cost.copy()
    choose = zeros((N, 2), dtype=int8)
    for child in postorder:
        p = parent[child]
        if p < 0:
            continue
        pair = oriented_cost(edge_cost, edges, edge_to_parent[child],
                             child=child, parent=p)  # [parent_state, child_state]
        trial = pair + dp[child][None, :]
        choose[child] = argmax_first(trial, axis=1)
        dp[p] += max(trial, axis=1)
    value = sum(max(dp[root]) for root in roots)
    states = traceback(choose, dp, parent, reversed(postorder))
    return value, states
~~~

All helper names in this pseudocode are specifications, not library APIs. For stored endpoints (a,b), oriented_cost returns edge_cost[e] when (a,b) = (parent,child), its transpose when (a,b) = (child,parent), and rejects any other incidence. Test reversed stored edges independently of traversal. Derive N from node_cost; traceback assigns each root its first maximizing state, then visits parents before children. When return_witness is false, return (value, None) and skip traceback.

### 4.2 Essential analytic controls

These identities are deductions from (1) and must become executable oracle tests.

**No disturbance:** ρ = 0 gives weighted quadratic reconstruction. If E = 0, changing ρ must have no effect.

**One output:** m = 1 gives J_G = U_G.

**No edges, shared vector amplitudes:**

\[
J_\varnothing=\frac1m\sum_i\omega_i
\left(\|A_i\|^2+\rho^2\|B_i\|^2+2\rho|A_i^\top B_i|\right).
\tag{6}
\]

This is a strong cheap robust baseline. Do not substitute the scalar formula
Σ_o(|A_io|+ρ|B_io|)², which evaluates the relaxation.

**Expected independent noise:** with δᵢ independent, zero mean, variance ν, and M = diag(ω)+L_β,

\[
\mathbb E[L]=\frac1m\left[
\operatorname{tr}(A^\top M A)
+\nu\sum_i M_{ii}\|B_i\|^2
\right].
\tag{7}
\]

Uniform δᵢ in [−ρ,ρ] gives ν = ρ²/3; independent endpoint signs give ν = ρ². This identity provides an exact, deterministic augmented least-squares baseline, avoiding Monte Carlo noise.

**Orthogonal-output diagnostic:** rotating A and B by the same orthogonal matrix preserves J, because it preserves vector norms. U generally changes. This is an oracle property, not a claim that arbitrary rotations preserve a nonlinear network or its quantization grid.

**Finite contrast certificate:** if J_G ≤ ε², every allowed amplitude and a connected path P whose edge weights are strictly positive satisfy
\[
\|r_u-r_v\|_2\le\sqrt{m}\,\epsilon\sqrt{\sum_{e\in P}\beta_e^{-1}}.
\tag{8}
\]
It is a finite-set linear-layer bound. A path containing a zero-weight edge has no finite certificate from this argument. Normalized small edge weights can make it loose; it does not establish downstream semantics.

### 4.3 Exact counterexamples to include in the harness

1. **Shared across examples is insufficient.** With scalar A₁ = A₂ = 0, B₁ = B₂ = 1, ωᵢ = 1/2, and one edge β = λ, forcing the same amplitude for both examples gives ρ²; independent amplitudes give ρ²(1+4λ). At ρ = 1/4, λ = 1 these are 0.0625 and 0.3125.

2. **Output relaxation changes code rankings.** Let N = 1, ρ = 1, no edges, Hᵀ = (0,1,0), Hᑫ = (1,1,0), and both teacher rows (4,0,−3). A 3-bit grid with scale 1 and zero 3 permits values −3 through 4. Candidate A has rows (1,0,0), (1,−2,0); candidate B has rows (1,−1,0), (2,−2,0). Before the common division by m = 2, (U(A),J(A)) = (8,4), whereas (U(B),J(B)) = (5,5). Verify the reversal exactly.

3. **Bipartite is not sufficient for the vector graph-cut claim.** On a four-cycle, drift vectors (1,0), (1,1), (0,1), (−1,1) have successive inner-product signs (+,+,+,−). Joint pair coefficients are −2ρ²βᵢⱼ BᵢᵀBⱼ; these signs cannot all be made attractive by vertex sign changes. The selected implementation therefore requires a forest. Do not import the scalar bipartite theorem.

These fixtures distinguish three different meanings of “shared”: one amplitude for the whole batch, one per example shared across outputs, and separate per-output adversaries.

### 4.4 Fixed grid and export invariants

For B_bits bits and group size g = 32, per output row and contiguous input group:

\[
l=\min(0,\min_j W^\star_{oj}),\quad
h=\max(0,\max_j W^\star_{oj}),\quad
s=\operatorname{FP32}\!\left(\max\{(h-l)/(2^{B_\mathrm{bits}}-1),10^{-12}\}\right),
\]
\[
z=\operatorname{clip}(\operatorname{round}_{\rm even}(-l/s),0,2^{B_\mathrm{bits}}-1),\quad
g_{o,g(j)}(q)=\operatorname{FP64}\!\left[
\operatorname{FP32}\!\left(s_{o,g(j)}\,\operatorname{FP32}(q-z_{o,g(j)})\right)\right],
\quad W(Q)_{oj}=g_{o,g(j)}(Q_{oj}).
\tag{9}
\]

The legal-value table g explicitly includes FP32 deployment dequantization and is then promoted exactly to float64 for calibration. This is the ordinary packed affine q/s/z representation with its arithmetic specified. All fixed-format methods optimize the same effective values, including rounding baselines and floor/ceil neighbors. Do not calibrate with an unrounded scale and export a different one. The positive scale floor reproduces the archive convention and must be included in the manifest. Use integer codes and zeros; reject nonintegral values before any cast. All-zero groups reconstruct exactly zero. Partial final groups use only real weights for scale selection. Keep the archive's ideal affine float64 product s(q−z) as a separately labeled historical-reproduction mode; the new deployment-consistent arithmetic is an explicit numerical refinement, not a silently changed reproduction.

**Precision contract:** execute the teacher and each actual prefix in the same frozen CPU FP32 forward implementation; MPS is an optional separately validated acceleration. Capture Hᵀ and Hᑫ at layer-input hooks, with identical item order, then promote copied values and original FP32 W★ to float64. Define the immutable local target Y by the float64 product in Section 2.2; retain actual FP32 teacher preactivation outputs separately for an arithmetic-discrepancy check. This idealized local linear target is common to every fixed-grid baseline. It is not a claim that FP32 accumulation equals a float64 product. Export/reload must recover exactly the FP32 levels represented by g. Evaluate natural KL on these deployed weights, using FP64 CPU log-softmax on the resulting FP32 logits. Record forward discrepancies and a paired FP64-forward sensitivity run on fixed codes. A gain comparable to arithmetic or backend variation is numerically unresolved. Never evaluate one method in FP64 and another in FP32. All corner/forest/quadratic identities remain valid for these fixed legal weight values.

The core comparison fixes grids across methods. Separately, the practical comparison allows each baseline its established clipping/scaling procedure, charges that work, and compares complete storage. These are two distinct experiments.

Serialize code bits in a documented little-endian bitstream with explicit shape, bit width, grouping, scales, zeros, bias, dtype, format version, and hashes. Padding bits must be zero and checked on load. Test all intended bit widths, odd element counts, empty optional bias, truncated files, invalid codes, and metadata mismatch. Count the actual container and backend bytes as well as ideal payload bytes.

### 4.5 Outer integer optimization

Use the same strong nominal initialization for the matched mechanism comparison. Evaluate RTN, validated target-preserving compensation, and converged nominal coordinate/pair candidates on calibration only; select the minimum of unweighted asymmetric reconstruction ||HᑫW(Q)ᵀ−Y||²/(Nm), with deterministic cost/code ties. Charge all initialization candidates equally to the compared refinements. This starting-code choice is made once, independently of each refinement's target objective. Baselines also retain their own authentic initialization in the practical comparison.

For joint descent, maintain A, B, u, c, and the current exact J. Changing Q[o,j] changes the dequantized weight by Δ = g(new_code) − g(old_code), hence

\[
v^{new}_{io}(s)=v^{old}_{io}(s)+\Delta(H^\mathsf Q_{ij}+\rho sE_{ij}).
\tag{10}
\]

Subtract that row's old squared contributions from each table and add the new ones. Other rows remain fixed. Run the exact oracle for each candidate. This gives O(N+|E_G|) work per candidate after cache formation.

~~~text
Initialize fixed grid and common starting codes.
Build A, B, unary/edge tables from original immutable inputs.
For a deterministic sweep over (output row, input coordinate):
    Score every allowed code from the same current state.
    Choose the lowest exact objective; ties keep the current code.
    If the decrease exceeds the numerical acceptance threshold:
        commit one code, its residual/drift row, and all affected tables atomically.
For deterministic within-row coordinate pairs:
    Try the 3 × 3 adjacent-code neighborhood, including the current pair.
    Commit only a coherent exact-objective decrease.
Repeat according to the frozen convergence or budget rule.
Rebuild all derived state from codes, compare objectives, and export.
~~~

Output rows are coupled. **Never simultaneously accept separately improving row proposals against a stale shared state.** Candidate codes for one coordinate may be batched; independently committed output rows may not. Pair changes may span two coordinates of one row initially; cross-row moves are a later optimizer ablation.

Construct the within-row feature-pair list once from H = (Hᑫ)ᵀHᑫ, independently of the objective being optimized. For coordinates with Hⱼⱼ > 0, rank other nondead coordinates ℓ by descending |Hⱼℓ|/√(HⱼⱼHℓℓ), breaking ties by smaller original coordinate index. For each j, retain its first min(k,d_live−1) neighbors; union the unordered (min(j,ℓ),max(j,ℓ)) pairs and sort lexicographically. Coordinates with exactly zero Hⱼⱼ have zero correlation and contribute no pairs; reject materially negative diagonals and investigate tiny negative roundoff. Use k = 2 as a new development starting choice, compare k = 1 for continuity with the original main experiment, include k = 0 as the no-pair ablation, and expand k only when the development quality/time frontier warrants it. Store k, the exact rule, and the pair-list hash. The matched objective comparison shares this list, traversal, and neighborhood. Return an empty list for k = 0 or fewer than two live coordinates; never include self pairs.

The optional centered-final-logit objective projects each residual with P = I−11ᵀ/m; it does not center codes or change grids. Changing original row o then changes the centered residual along eₒ−1/m, affecting every centered component. Equation (10)'s scalar-row subtraction is consequently insufficient. Use full table recomputation for this ablation first, or validate the general norm update ||r+t v||²−||r||² = 2t(rᵀv)+t²||v||² with v = eₒ−1/m. Apply the same projection to all compared objectives and targets. Every centered control, including centered U, uses coherent sequential acceptance; the source's independent output-row commits are no longer valid.

The loss is convex in continuous weights but the grid problem is discrete. Strict descent over a finite grid terminates if run without a budget, yet a capped run need not be coordinate-stationary. Certify only the neighborhood actually exhausted. A single no-change pass before later pair moves is not a final stationarity certificate; repeat single and pair passes until the entire declared cycle has no accepted move.

Use CPU float64 and no fast-math initially. For initial small-instance parity, use |a−b| ≤ 10⁻¹² + 10⁻¹⁰ max(|a|,|b|), and record absolute/relative errors. A failed check produces diagnostics; do not widen tolerance until it passes. Calibrate an acceptance bound on representative development problems before confirmation: at least twice the largest measured cache/full-rebuild discrepancy plus the scale-aware arithmetic floor. Freeze the formula and coefficients. When a candidate's predicted gain is comparable to that bound, rebuild its full tables and compare directly. Do not suppress cache errors by clipping losses. This is a numerical acceptance threshold, not a minimum scientific improvement.

### 4.6 Optional exact acceleration

First compile the scalar loops, retaining the Python/NumPy oracle. Numba nopython compilation with fastmath disabled is a researched ARM64 option [S05]. Compilation changes implementation, not the objective; parity is mandatory.

A second optional optimization is **certified candidate rejection**. Keep previously found sign witnesses. Their evaluated losses are lower bounds on the candidate's maximum. If one witness already gives candidate loss at least the incumbent J minus tolerance, reject without a fresh DP. Otherwise run the exact DP; only it can authorize acceptance. Never accept using an incomplete witness pool. Compare the complete accepted code trajectory against the unfiltered optimizer under identical deterministic order. Count witness checks, DP calls, and their costs.

This is standard constraint-generation logic applied as an engineering optimization; it is not a novelty claim. Do not build it unless profiling identifies oracle cost as a real bottleneck.

### 4.7 Graph construction

For the primary classifier experiment, preserve the original heuristic: Euclidean distance between normalized raw images, eligible edges only between different teacher argmax classes, deterministic Kruskal minimum spanning forest. Use teacher predictions from the frozen teacher; no quantizer accesses true evaluation labels.

Sort by (distance, smaller stable item ID, larger stable item ID). Build distances in tiles or from checked float64 dot products; never allocate all N²d pair differences. Verify small-instance equivalence with direct distances. Duplicate zero-distance items need deterministic handling; all-same-class cases can yield an empty forest.

Construct the graph from the calibration set once per teacher/calibration replicate and hold it fixed across methods. Its use at every layer is a design choice to preserve the source mechanism. Alternative teacher-feature graphs, degree controls, and disjoint pairs are exploratory ablations, not silent replacements.

---

## 5. Baselines, data, metrics, and confirmatory protocol

### 5.1 Three comparison tracks

**Track A: mechanism isolation.** Freeze a teacher, a nontrivial quantized prefix, the selected layer, calibration IDs, graph, targets, grid, initialization, proposal schedule, and amplitude radius. Change only the objective. Start with the final layer, whose output is directly evaluable and whose m = 10 makes coherent search inexpensive. Include a hidden layer afterward to expose downstream nonlinear effects. The primary causal controls are J versus U, ρ = 0, λ = 0, and exact expected noise.

**Track B: practical local comparison.** Quantize the complete classifier, letting each method use its own sequential prefix and authentic optimization workflow. All methods get the same calibration pool and development access. Compare teacher fidelity, task performance, complete bytes, and complete calibration cost. A method's own-prefix stress score is labeled with its uncertainty-set identity and is secondary.

**Track C: external context or transfer.** Published LLM numbers are context only. A locally run Pythia experiment is a new controlled comparison with its own frozen protocol. Do not combine published numbers, Mac numbers, and synthetic stress into a single ranking.

### 5.2 Required baseline roster

| ID | Method | Role and executable scope |
|---|---|---|
| B0 | Floating teacher and RTN | Numerical reference and harness sanity floor |
| B1 | Symmetric reconstruction, full-target asymmetric reconstruction, and graph-weighted reconstruction | Strong simple quadratic methods with objective-matched search and convergence checks |
| B2 | Exact expected independent-noise quadratic, Eq. (7) | Essential same-information alternative to minimax |
| B3 | Joint no-edge robust loss, Eq. (6) | Tests necessity of the graph |
| B4 | Original rowwise FPCQ, Eq. (2) | Closest prior implementation; both faithful source schedule and matched optimizer schedule |
| B5 | GPTQ, GPTAQ, and ResComp independent dense references | Contemporary compensation family; published-equation ports, not untouched production-package claims |
| B6 | Current CoreQ and K-CoreQ | Mismatch shrinkage and stronger bounded-search alternative |
| B7 | YAQA with scalar affine quantizer and exact-class Sketch B on the MLP | Contemporary output-aware rounding; documented local specialization |
| B8 | AdaRound asymmetric reconstruction and full-model KL projected rounding | A runnable gradient-based competitor; local adaptation details below |
| D1 | One amplitude for all examples | Diagnostic blind-spot control, not a strong competitor |
| D2 | Random directions and matched random graphs | Attribution controls, not evidence of competitiveness |

Appendix A supplies the compensation, CoreQ, YAQA, and gradient recipes. The simple quadratic family is fully specified below. Do not waive B2, B4, B6, B7, or all gradient-based comparisons merely to get a favorable result. If a material competitor cannot be implemented or validated locally, narrow the claim and report the gap. A head-to-head claim against that named method is blocked.

The original GPTQ package and the official YAQA/AWQ deployment stacks have CUDA assumptions [S06, Q05, Q13]. Their mathematics can often be run on CPU; this is an independent port whose parity must be tested. Large published calibration workloads are not required to run the same algorithm on an explicitly smaller local population, but the resulting claim is about that population.

### 5.3 Strong quadratic implementation

Work with input columns X = (Hᑫ)ᵀ, T = Yᵀ, and M = diag(ω)+L_β. With optional ridge tether κ ≥ 0 to W★,

\[
L(W)=\operatorname{tr}[(WX-T)M(WX-T)^\top]+\kappa\|W-W^\star\|_F^2,
\]
\[
H=XMX^\top+\kappa I,\quad C=TMX^\top+\kappa W^\star,\quad
W_c=CH^{-1}.
\tag{11}
\]

Solve systems; do not explicitly invert H in the production baseline. Completion of the square gives a standard quadratic around W_c. For a single row, gradient half-vector is g = wH − C_o. Changing w_j by Δ changes loss by 2g_jΔ + H_jjΔ². This permits exact best-code enumeration, fast cache updates, all-code pair moves on small layers, and the same adjacent-pair schedule as FPCQ.

The expected-noise comparator adds νEᵀdiag(M)E to H and no additional target cross term. An equivalent augmented-data implementation is a required independent check:

- Node rows: √ωᵢ Hᑫᵢ, target √ωᵢYᵢ.
- Contrast rows: √βᵢⱼ(Hᑫᵢ−Hᑫⱼ), target √βᵢⱼ(Yᵢ−Yⱼ).
- Noise rows: √(νMᵢᵢ)Eᵢ, target zero.

Ridge, if used, adds √κI with target √κW★ᵀ. Normalize all terms consistently by m when comparing reported objective values.

For singular H, use a checked least-squares/pseudoinverse reference and report rank. For fast factorized rounding, add explicitly reported trace-scaled damping. Do not silently make a singular direction “informative” by arbitrary inversion. Dead channels should retain their declared grid and be rounded by a deterministic convention; they must not trigger NaN or unbounded weights.

### 5.4 Baseline validation and tuning fairness

Before comparing methods, reproduce their equations on small matrices with positive-definite, singular, dead-channel, correlated-channel, clipped-grid, and permuted-group cases. Distinguish a mathematical algorithm port from a changed objective. A purported faithful port that fails parity does not enter the competitiveness table.

Use both:

1. **Matched search:** identical starts, candidate neighborhoods, traversal, and candidate-evaluation counts. This isolates the objective and optimizer components.
2. **Matched total elapsed cost and unrestricted development frontier:** let each baseline exploit its cheap algebra, established solver, and tuning. Show the best quality available at measured calibration budgets, including preprocessing and hyperparameter search.

Equal sweeps are not equal compute; equal compute alone can hide a meaningful but expensive component effect. Report both. Share generic caching, compiled loops, stable linear algebra, and packing improvements where applicable.

For development, initialize ρ with {0, 0.125, 0.25, 0.5}, including the source value; λ with {0, 0.125, 0.25, 0.5, 1}; and expected-noise ν with the corresponding uniform/endpoint values plus zero. These are an interpretable starting design, not a permanent search quota. Expand only when a boundary optimum or a measured amplitude distribution warrants it. Choose one complete configuration using development KL and explicit task/cost constraints; record every attempted configuration. Do not choose each layer using protected evaluation.

Classify configurations before selecting them: ρ = 0 is nominal/contrast quadratic reconstruction; ρ > 0 and λ = 0 is joint no-edge robustness; ρ > 0 and λ > 0 is the full joint forest. The primary candidate for the full contribution must have positive ρ and λ in the layers intended to test it. The first layer's E = 0 fast path is an explicit exception with no robustness claim. Zero settings remain mandatory competing controls. If a simpler family is preferred on development, record a pivot and change the primary claim before freeze; do not retain the full mechanism's name for a degenerate winner.

For damping, include the published/source default and trace-scaled neighboring values on a logarithmic scale; record rank and conditioning. For CoreQ beam width, increase 1 → 2 → 4 → 8 and further only while its development quality/cost frontier changes materially relative to measurement uncertainty. Give analogous search effort to other useful baselines. “Materially” here governs spending more development compute, not a minimum success effect.

Freeze two distinct comparator identities. **B★_fixed** is the lowest-development-KL validated noncandidate configuration using the exact primary grid, effective level values, calibration examples, teacher, bias/activation policy, and evaluation arithmetic. Ties at the declared numerical floor prefer lower complete calibration cost, then a fixed method-name order. **B★_practical** belongs to the separate authentic-format frontier, where clipping/scaling or other permitted format choices and actual bytes are disclosed. Never select a comparator in one regime and report it under the other.

The primary fidelity test uses **fixed complete configurations with realized costs reported**, not a claim of equal elapsed cost. Baseline schedules are tuned and extended on development as specified in Appendix A; any resource stopping constraint is frozen and reported. Matched candidate searches isolate mechanism, and the measured equal-total-cost frontier tests whether additional candidate expenditure explains a benefit. A primary KL win alone establishes neither equal-cost superiority nor practical dominance. Also retain all required baseline outputs for secondary comparisons. Claiming superiority over every required baseline needs supported comparisons with each; winning against B★_fixed alone does not justify that wording.

### 5.5 Data and teacher

**Primary dataset:** Fashion-MNIST, official 60,000 training and 10,000 test 28 × 28 grayscale images in ten classes, from Zalando Research [D01]. Preserve the official train/test boundary. Verify the four official MD5s, record SHA-256 and source revision, and retain the license notice. Appendix B lists filenames and checksums. Decode IDX directly with gzip/NumPy if this avoids an unnecessary torchvision binary dependency.

Normalize uint8 pixels by 255; flatten in documented row-major order. No image augmentation, resizing, PCA, or label filtering in the primary experiment. Check exact duplicates with byte hashes. If duplicates cross development/protected partitions, retain a primary official-split analysis and a separately preregistered duplicate-excluded sensitivity analysis; do not silently remove difficult examples.

Partition the 60,000 official training items once, stratified by class using a fixed recorded generator and stable original indices:

| Partition | Count | Allowed use |
|---|---:|---|
| T | 45,000 | Teacher training only |
| V | 5,000 | Teacher checkpoint/schedule selection |
| C | 5,000 | Quantization calibration pool and prefix construction |
| D | 2,500 | Quantizer/baseline configuration selection |
| P | 2,500 | Exploratory pilot, variance and resource calibration |

All methods use matched calibration draws from C. Initial N = 512 is suitable for the later 256-wide layers; include N = 128 for continuity and N = 1,024 for calibration-size/rank sensitivity. These choices expose rather than hide rank limitations. For the first 784-input layer, E = 0; apply the cheap quadratic equivalent instead of repeatedly running an unnecessary robust oracle. Increase N if development learning curves show meaningful calibration instability and measured capacity permits it. Freeze the chosen N before confirmation.

**Proposed teacher:** 784 → 256 → 256 → 10, ReLU hidden layers, affine biases, no dropout or batch normalization. It has 268,800 weights and 522 biases. This architecture is a controlled local vehicle, not a claim about the best Fashion-MNIST model.

Train in FP32 using AdamW, initial learning rate 0.001, β = (0.9, 0.999), ε = 10⁻⁸, weight decay 0.0001, and minibatch 256 as development defaults. Evaluate V cross-entropy after every epoch and retain the best checkpoint. Use the pilot to settle the learning-rate schedule and stopping rule: extend while the validation learning curve is still improving; compare a lower learning rate and a longer run before calling a plateau. Freeze that training protocol for all confirmation teachers. Do not treat an unfinished or numerically unstable teacher as evidence against quantization.

Record achieved validation accuracy/loss, training curves, checkpoint selection, and teacher sanity tests on V/P. Official protected teacher metrics are locked until the same protocol freeze as student metrics; they cannot guide training or architecture selection. A teacher that predicts one class, has nonfinite outputs, or fails a simple reproducible training sanity check blocks later work. There is no arbitrary required accuracy threshold: explain the quantization headroom of the teacher actually obtained.

### 5.6 Protected evaluation and replication

Do not inspect protected teacher or student metric results until the protocol, candidate, B★_fixed, any separately declared B★_practical, data IDs, analysis code, and sample size are frozen and hashed.

Choose the protected-data allocation using P-based precision calculations. With no application-derived precision target, the default is two stratified halves: preserving independent item confirmation is more informative than inventing a practical-effect cutoff.

- Use deterministic stratified A and B halves of 5,000 items each when no precision target is available, or when that allocation meets the declared target. A is the initial confirmation; B is a fresh item confirmation. Report the detectable-effect curve. Do not open B to repair an A result without recording an amended hypothesis.
- If a documented application-derived target cannot be met by halving but can be met with 10,000 items, use all 10,000 once for the primary test. Label subsequent fresh-teacher/calibration-seed repeats as stochastic-pipeline replication on the same examples, not independent item confirmation. Any method change then requires a genuinely new protected population or a clearly exploratory report. If neither allocation meets the target, state the attainable precision rather than pretending the larger allocation solves it.

This decision is made before seeing either allocation's results. Reusing a fixed test population across independent teachers is permitted, but its shared-example dependence must remain in the analysis.

An independent pipeline replicate uses a newly trained teacher seed and a matched calibration draw, with all methods applied to that same teacher and draw. The default has one calibration draw per teacher. If several draws reuse one teacher, average within teacher first and give teachers equal weight; the draws are nested, not independent teacher replicates. The original three digits seeds remain reproduction data. The primary interval conditions on the prospectively frozen teacher panel. Fresh-panel replication evaluates sensitivity to teacher randomness without pretending that a few teachers support precise population-wide inference.

### 5.7 Primary estimand and metrics

**Primary local claim:** at 2-bit weights, group size 32, the exact fixed grid and frozen complete configurations, the positive-ρ/positive-λ joint method reduces natural whole-model teacher KL relative to B★_fixed over the declared Fashion-MNIST image population **conditional on the frozen panel of teachers/calibration draws**. Costs are reported separately. Two bits is selected because the source showed its strongest distortion and possible benefit there; 3 and 4 bits are prespecified secondary boundary conditions, not interchangeable chances to win.

For example i and replicate r,

\[
\ell_{ri}=\sum_{c=1}^{10}p^\star_{ric}
(\log p^\star_{ric}-\log p^\mathsf Q_{ric}),\qquad
D_{ri}=\ell^{joint}_{ri}-\ell^{B^\star_{\rm fixed}}_{ri}.
\tag{12}
\]

Compute log-softmax stably, use natural logs, and aggregate equally over images and the frozen pipeline panel. Report Δ = mean(D), absolute KLs, the relative change when the baseline denominator is nonzero, and uncertainty. Negative Δ favors the candidate. The reported finite-benchmark mean is exact for those stored outputs; its bootstrap interval additionally assumes images are exchangeable within the declared class strata and conditions on the fixed class mix, model panel, and split policy. It does not cover arbitrary future architectures or collection shifts.

**Secondary metrics and guardrails**

| Metric | Definition / purpose |
|---|---|
| Accuracy and per-class accuracy | True-label correctness; report paired differences in percentage points |
| Teacher/student prediction disagreement | Fidelity of decisions even when teacher is wrong |
| Label cross-entropy | Task probabilistic quality; separate from teacher KL |
| Centered-logit MSE | Subtract each example's mean logit in both models, then mean squared difference over classes |
| Fixed-set J, U, J-no-edge and nominal loss | Exact deterministic mechanism diagnostics with common inputs/graph/radius |
| Worst-witness pattern and J/U gap | Attribution and conservatism; never count witness signs as observations |
| Code-change rates, clipping, local-stationarity gaps | Optimizer behavior and feasible-set use |
| Calibration time, tuning time, peak memory, exported bytes | End-to-end offline costs and actual format |
| Optional inference latency/throughput | Only after identical validated backend export |
| Actual-prefix-shift degradation | Tests externally constructed mismatch rather than only the training tube |

A statistically supported KL gain with worse accuracy is a fidelity tradeoff, not an overall accuracy improvement. Before confirmation, specify any application-derived allowable task regression in accuracy/cross-entropy. If no application tolerance is available, report the full tradeoff and avoid a categorical practical-win claim when important guardrail uncertainty remains.

### 5.8 Fixed-sample statistical procedure

The unit for natural loss is the image; the unit for pipeline randomness is the independent teacher/calibration replicate. Timing repetitions, class edges, tokens from the same document, and random probes are not extra independent accuracy samples.

Use P to estimate paired-loss variance, including replicate and example components. A useful planning model is

\[
D_{ri}=\Delta+u_r+v_i+\epsilon_{ri},\qquad
\operatorname{Var}(\bar D)\approx
\sigma_u^2/R+\sigma_v^2/n+\sigma_\epsilon^2/(Rn).
\tag{13}
\]

This is a planning model, not a Gaussian-loss assumption or a claim that teacher-population variance can be estimated precisely from a small pilot. Save the full paired matrix. Freeze candidate and comparator using D before the final P variance calculation. If earlier exploration has substantially reused P to choose the method, use unused P folds or fresh development pipelines for this final calculation and disclose the reuse.

**Executable pilot and panel-size rule.** Start with four independent pilot pipelines, the smallest starting batch that leaves three under a leave-one-teacher check; four is not a universal adequacy threshold or an experiment cap. For the pilot matrix with R_p rows and n_p columns, compute

~~~
row_class[r,c] = mean(D[r,i] for i in class c)
class_mean[c] = mean_r(row_class[r,c])
MS_teacher = n_p * sample_variance(overall_row_means, ddof=1)
MS_item = R_p * sum_i((column_mean[i]-class_mean[class_of[i]])**2) / (n_p-C_classes)
interaction[r,i] = D[r,i] - row_class[r,class_of[i]] - column_mean[i] + class_mean[class_of[i]]
MSE = sum(interaction**2) / ((R_p-1)*(n_p-C_classes))
sigma_u2 = max((MS_teacher-MSE)/n_p, 0)
sigma_v2 = max((MS_item-MSE)/R_p, 0)
sigma_e2 = MSE
R_balance(n) = max(2, ceil(max(0, n*sigma_u2-sigma_e2)/sigma_v2))
~~~

Here C_classes = 10. Removing class and teacher-by-class means aligns the planning variance with the primary fixed-stratum bootstrap. For balanced strata and approximately common within-class variances, σ_u² describes effective teacher-panel mean variability, including its average class interaction. The last formula, when σ_v² is positive above the numerical floor, balances estimated teacher Monte Carlo variance σ_u²/R against item and interaction variance σ_v²/n+σ_ε²/(Rn). This remains an approximate planning model; report heteroskedasticity and simulate the actual stratified estimator as a sensitivity check. Compute R_balance on the full pilot and after omitting each teacher. The minimum/maximum required counts R_lo/R_hi are a **planning sensitivity range**, not a confidence bound. Measure the marginal cost of another pilot and a complete fresh confirmation pipeline. Add one pilot only while its projected cost is smaller than (R_hi−R_lo) times the latter cost and local resource capacity permits it. Recompute after each addition. Stop when the counts coincide, another pilot cannot plausibly save the remaining planning spread, or measured capacity blocks expansion. Freeze R_hi and retain the range. This is a declared cost-sensitive planning heuristic; it provides no guaranteed population coverage.

If a zero item-variance estimate makes balance undefined, freeze a confirmation panel equal to the completed pilot count, at least four, and abandon the balance assertion. If R_hi is infeasible, choose the largest complete panel supported by an explicitly documented measured resource allocation and report that it does not satisfy the planning balance. Neither fallback broadens the conditional primary claim. All confirmation teachers and calibration draws are fresh. No sample-size decision uses a positive pilot-effect lower bound or a desired p-value.

**Image count and attainable sensitivity.** Apply Section 5.6. Report the projected confidence width and 90%-power detectable-effect curve at two-sided α = 0.05 across feasible R/n/cost, including the chosen design. With no application-specific useful effect, report attainable precision without inventing one. If a useful effect δ_design is supplied, simulate the complete paired estimator with shifted pilot differences and empirical tails to check its planned power. A missed precision target limits the conclusion; it does not justify lowering the success standard after evaluation.

**Frozen primary confidence interval.**

1. For each protected image, compute d_i = mean_r D_ri. Resample image IDs with replacement separately within each true class, preserving its protected count. Use exactly the same resampled IDs for every teacher and method. Do not resample teachers for the primary conditional-panel interval.
2. For each draw, compute the mean of the resampled d_i with the original class proportions. Use the 2.5th and 97.5th percentile of those means, with the linear quantile interpolation convention explicitly stored.
3. Determine the bootstrap draw count B_boot on P before freeze. Start with 4,096 draws and double in fixed-seed batches. For endpoint p ∈ {0.025,0.975}, set one-based sorted-estimate ranks k_lo = max(1, BinomQuantile(0.025;B_boot,p)) and k_hi = min(B_boot, BinomQuantile(0.975;B_boot,p)+1). Their values give a conservative Monte Carlo interval for that endpoint. Require its width to be at most 1% of the statistical CI width for both endpoints. This 1% controls numerical approximation, not an improvement target. Generate resamples in bounded batches; store only bootstrap means, not a B_boot × n index array. If the statistical width is at the measured arithmetic floor, record a degenerate numerical case; do not divide by zero. Store B_boot and the generator/seed; confirmation uses that fixed B_boot. If its endpoint precision later proves insufficient, mark the numerical decision unresolved or apply a separately preregistered continuation that depends only on Monte Carlo error, never on whether the interval excludes zero.
4. A two-sided 95% interval with upper endpoint below zero supports the stated conditional KL improvement. An interval spanning zero is inconclusive at its precision, not equivalence. Report every teacher's mean effect and leave-one-teacher-out conditional image intervals as sensitivity analyses; none authorizes deleting an inconvenient teacher.
5. Use the reserved B half and a prospectively fresh panel of the same R for replication when that allocation is available; freeze its separate named seed stream in WP06 along with the primary panel. If the same images must be reused, label only the pipeline as fresh. Broader teacher-population inference requires a separately designed study; a small crossed bootstrap is not a substitute.
6. For formal secondary comparisons, preregister one family and apply Holm correction to paired bootstrap tests. For each comparison, set d_i⁰ = d_i−mean(d within class c(i))+μ₀, with null μ₀ = 0. Resample d_i⁰ by the same class procedure and compute p = (1+count(|mean_b⁰−μ₀| ≥ |Δ−μ₀|))/(B_boot+1). This is a centered bootstrap approximation, not an exact randomization test. Retain unadjusted effect intervals for magnitude and label them accordingly. Otherwise mark secondary comparisons exploratory. A “wins at all bits” statement requires all relevant comparisons, not selection of a favorable bit.

Freeze R, n, pipeline seeds, item IDs, B_boot, missing-run handling, config identities, and analysis revision before opening protected metrics. Never add confirmation seeds because the result is almost significant.

**Incomplete pipelines.** The primary paired analysis requires all frozen method/pipeline pairs. An identified infrastructure interruption may resume the same state or restart the same seed/configuration after its cause is repaired; link attempts and retain cost. Method-specific instability, exhausted budget, or OOM under the declared configuration is an outcome, not permission to replace a seed. Unresolved missing pairs block the complete-pipeline primary claim; report failures and a clearly labeled complete-case sensitivity analysis. Do not invent a post hoc penalty or silently give only the candidate more resources.

To claim equivalence or absence of a practically relevant effect, predeclare an application-derived equivalence interval and show that the CI lies inside it. Otherwise a negative result should state its detectable effect and residual uncertainty.

### 5.9 Harness sensitivity and stress-score dependence

Require these before any model comparison:

- Identical method/configuration/model files produce zero paired differences.
- An intentionally degraded low-bit model produces detectable loss changes.
- Teacher compared with itself gives KL at the floating-point error floor.
- Reordering examples and relabeling graph vertices preserve the deterministic objective.
- J ≤ U, E = 0, ρ = 0, and one-output identities hold.
- Serialization reload preserves codes, weights, and outputs.

A graph stress score couples its vertices. Treat a fixed evaluation forest as one deterministic finite-population quantity. If an uncertainty interval across stress workloads is needed, construct independent disjoint item blocks with their forests fixed by the protocol and resample whole blocks. Never bootstrap individual edges as independent examples.

---

## 6. M3 Max feasibility and hardware-adaptive configuration

### 6.1 Supported core stack and version strategy

The scientific path uses native ARM64 Python, NumPy/SciPy CPU float64, and PyTorch FP32. Numba is an optional compiled-loop dependency that becomes useful for coherent coordinate search. MPS is optional for teacher training and forward passes. No remote GPU, paid API, extra computer, or specialized equipment is required.

Current official documentation inspected on the research date reports PyTorch 2.15, released 28 October 2026, with Python ≥ 3.11 and macOS ≥ 15 for supported MPS acceleration [S01]. Apple's overview still describes older requirements [S03]. Use release-specific documentation and actual smoke tests. Metal/MPS does not support float64 tensors [S02]; fallback settings do not make double allocation work. Keep the oracle, sensitive factorization, and acceptance tests on CPU.

Numba documentation lists native macOS ARM64 wheel support; the inspected 0.68.1 compatibility range includes Python 3.10–3.15 and NumPy below 2.6 [S05]. Select one mutually compatible native environment, verify wheels/imports, then lock exact versions and hashes. A sensible discovery candidate is Python 3.12, not a requirement to install the newest interpreter.

Do not copy the archive's Linux-only torch version suffix into a Mac lock. Do not upgrade macOS as an implicit prerequisite. If the installed OS cannot support the chosen MPS release, select a documented compatible CPU/software version and rerun tests.

### 6.2 Initial discovery

These are existing system commands, to run locally and record:

~~~bash
uname -m
sw_vers
sysctl -n hw.memsize
sysctl -n hw.physicalcpu
sysctl -n hw.logicalcpu
df -k .
vm_stat
pmset -g batt
python3 -c 'import platform; print(platform.platform()); print(platform.machine()); print(platform.python_version())'
~~~

Record chip/model information from system tools without publishing serial numbers. Inspect actual unified memory, available storage, free/available memory under normal load, native versus Rosetta architecture, power mode, and acceleration availability. The M3 Max product family has several memory/CPU/GPU configurations [S00]; this plan does not assume the maximum.

A proposed discovery module additionally records NumPy/SciPy BLAS configuration, thread settings, package versions, MPS is_built/is_available, and optional MLX information. Use a separate environment; installing project packages does not authorize changes to unrelated user environments.

### 6.3 Critical capability smoke tests

1. CPU float64 matrix multiplication, Cholesky, least-squares solve, and all exact-oracle fixtures.
2. Numba compilation with float64 and fastmath disabled; parity of objective, witnesses up to ties, and selected codes with the reference.
3. FP32 MLP forward, ReLU, log-softmax, backward, optimizer step, save/reload on CPU.
4. If using MPS, the same operations and a deterministic CPU/MPS comparison; explicitly synchronize. Turn silent fallback off during capability diagnosis.
5. Compare numerical discrepancies with the actual quantization effect sizes from development. If accelerator differences are not negligible for the intended claim, evaluate all methods on the CPU reference and retain fixed checkpoints.
6. For optional packed inference, validate exact code/scale mapping before benchmarking.

Failure of an accelerator test calls for the faithful CPU path. It does not invalidate the mechanism. A missing faithful CPU operation is a blocking prerequisite for claims depending on it.

### 6.4 Memory model

Let N be calibration vertices, d input width, m output width, K the number of code candidates batched for one coherent coordinate, and P the number of trainable teacher weights. A conservative baseline is:

| Contributor | Leading storage, bytes | Notes |
|---|---:|---|
| Hᵀ, Hᑫ, E | 24Nd | Three float64 arrays; one can be derived or memory-mapped |
| Y, A, B | 24Nm | Includes all output rows, which share the objective |
| Original/current weights and codes | Approximately 18md, plus grids | Two float64 copies and int16 codes; count actual extra copies |
| Global unary/edge tables | 8(2N + 4|E_G|) | Forest-size storage |
| Candidate DP/table workspace | Measured constant × 8K(N+|E_G|) | One coordinate's candidates, not all output rows |
| Gram matrices and factorization | c_H × 8d² | c_H depends on baseline copies/factors |
| YAQA factors | O(8(d²+m²)) | Stream per-example Jacobians; never allocate a full weight Hessian |
| Teacher training | About 16P plus activations/framework | FP32 weights, gradients, Adam first/second moments |
| Dataset/cache/framework/OS | Measured | Unified-memory allocations can overlap RSS/driver accounting |

For the proposed MLP, FP32 weights occupy about 1.03 MiB; weight/gradient/Adam storage starts near 4.1 MiB. One float64 784² Gram is about 4.69 MiB. All 70,000 images take about 52.3 MiB as uint8, or 209.4 MiB as FP32. These are array estimates, not measured process peaks.

At N = 512 and K = 16, compact joint-oracle working arrays can be small; actual NumPy temporaries and framework buffers must be measured. A full weight Hessian for 268,800 weights would require about 578 GB in FP64 and must never be formed. Ten-class Jacobians and Kronecker contractions avoid that allocation.

The original graph constructor creates an unordered-pair difference tensor with N(N−1)d/2 float64 elements: at N = 1,024 and d = 784, this tensor alone is about 3.06 GiB. A full ordered N²d implementation would take about 6.125 GiB. Tiled exact distances remove this avoidable allocation. Approximate neighbors change the graph and require an amendment.

### 6.5 Time model and calibration

For one joint scalar-coordinate sweep, the dominant count is

\[
O\!\left(md\,2^{B_\mathrm{bits}}(N+|E_G|)\right);
\]
a within-row adjacent-pair pass is O(9mP_pair(N+|E_G|)). Initial matrix products, graph construction, factorization, and export add costs. Joint row coupling may make a Python implementation much slower than the original vectorized rowwise method despite identical big-O notation.

A teacher epoch has a rough dense forward/backward work estimate near 6n_trainP operations. For 45,000 items and 268,800 weights, this is about 72.6 GFLOP per epoch. The coefficient is an estimate; actual device throughput determines time.

The first development run must measure:

- One teacher epoch after warm-up and its memory peak.
- Aligned activation collection and graph construction.
- Baseline Gram/sketch preparation and factorization.
- One full final-layer coordinate pass, one hidden-layer pass, and the selected pair pass.
- Reference versus compiled oracle throughput, compilation cost separately.
- Export, reload, and whole-model evaluation.
- Complete elapsed quantization including all preceding stages.

Measure more than one size before extrapolating. Report a runtime interval reflecting cache/thermal variability. Do not multiply the archive's seconds by a weight ratio and call that an M3 prediction.

Choose candidate-batch size, threads, and array blocking from measured memory/throughput. These can change without changing the experiment if numerical parity holds. Reducing calibration N, cutting graph edges, shrinking the model, changing precision, or dropping baselines changes the experiment and must be recorded before confirmation.

There is no arbitrary hour limit. A longer resumable local run is acceptable when power/sensitivity and information gain justify it. Conversely, continuing expensive runs without an actionable remaining uncertainty is not justified by an aspiration to a breakthrough.

### 6.6 Storage, interruption, and recovery

Estimate disk needs as immutable source/data + teacher checkpoints + per-method packed models + per-example outputs + temporary caches + atomic-write duplication. Keep image data in uint8 and stream logits. Retain unfavorable and failed-run records; large reproducible intermediates may be evicted only after retaining inputs/hashes and the regeneration command.

At a safe boundary, checkpoint codes, grid, frozen-input hashes, graph hash, objective version, row/coordinate/sweep cursor, RNG states, cumulative elapsed/evaluation cost, and the incumbent objective. Write to a temporary file, fsync where appropriate, and atomically rename. On resume, reconstruct A/B/tables from codes; verify hashes and the objective. A changed dependency/source/configuration cannot silently resume into the same run identity.

For joint optimization, checkpoint after a coherent commit or completed deterministic chunk. Never serialize an updated code array with stale residual tables. An interrupted candidate evaluation can be rerun from the previous committed state.

Teacher and gradient-rounding checkpoints additionally store optimizer moments, scheduler state, epoch/global-step/batch cursor, current sampler permutation, every RNG state, best-development checkpoint identity, loss/annealing schedule state, and stopping counters. A CPU interrupted/resumed fixture must reproduce the next updates and checkpoint-selection result. If exact stateful continuation is unsupported on a selected backend, restart the same frozen replicate from its initial seed and retain the interrupted attempt and cost; do not label a weights-only restart as resumed training. Measure and disclose MPS nondeterminism.

### 6.7 Optional MLX deployment

MLX's inspected affine API supports groups 32/64/128 and several bit widths including 2/3/4 [S04]. Its representation is scale × code + bias. Map existing FPCQ codes with bias = −scale × zero, following the pinned packing layout. Calling MLX quantize on reconstructed weights would select a new representation and is not a valid export of the optimized codes.

A 784-input group-32 layer needs padding to 800 for that backend. Pad inputs by zero and codes by the group zero point; preserve original scales, and count added bytes. Arithmetic forms can differ slightly in FP32, so test reconstructed weights and complete outputs.

Inference timing must use the same validated format/backend for every code-selection method. Warm the actual path, force lazy evaluation, synchronize around the measured region, interleave method order, and report cold-load time separately. A latency difference that vanishes with identical kernels is not a method advantage. Repeated timings quantify a workload's timing variability, not extra task samples.

---

## 7. Ordered work packages and decision gates

Every package writes an immutable configuration, machine-readable result, human decision note, and status in the run registry. Acceptance is about correctness or interpretable evidence; a positive scientific result is never required to complete a package.

### WP00 — Acquire, discover, and freeze provenance

**Objective/risk:** establish legal/technical access, actual hardware, and reproducible inputs.  
**Dependencies/entry:** this plan and a fresh local project; the source ZIP is optional.

**Actions:** create the repository and environment; run Section 6 discovery; obtain Fashion-MNIST from its official source; verify checksums; snapshot reference paper versions and any copied baseline licenses; record the attachment hashes if those files are present. Capture exact package versions and resolve source commits without inventing tags.

**Files/commands:** proposed modules cli, discover, provenance; proposed command:

~~~bash
python -m fpcq discover --out artifacts/environment.json
python -m fpcq assets verify --config configs/assets.toml
~~~

**Outputs/tests:** environment JSON with bytes and versions; assets manifest with origin, revision, license, size, SHA-256, permitted uses, and local path; dependency lock; verified native architecture. Resource use is only small checks and the dataset download.

**Gate:** continue when core assets and CPU dependencies are usable. If the supplied ZIP lacks a redistribution basis, use its equations/fixtures for local work and keep redistribution separate. If Fashion data is unavailable, stop that dataset claim; a substitute is an explicit amendment with new split/power design, not an invisible fallback.

### WP01 — Reconstruct and challenge the indispensable mathematics

**Objective/risk:** eliminate wrong endpoint, orientation, and cache semantics before benchmarking.  
**Dependencies:** WP00 CPU stack.

**Actions:** implement original scalar and selected vector reference oracles, exhaustive enumeration for small forests, Eq. (6) and (7), the three counterexamples, graph validation, and fixed-grid serialization. Check vector DP against exhaustive signs and cached updates against full rebuilding. Test disconnected forests, isolated vertices, ties, reversed edge storage, zero radius, zero drift, nonuniform weights, and invalid cycles. Reproduce the ranking reversal. Validate the centered-objective ablation with full table recomputation before any cache optimization.

If the original archive is present, run its tests without overwriting its results and reproduce the saved-model metrics. Correct the pair-selection zero-neighbor/single-channel bug in the new implementation. The optional strip verifier implements Section 10's LP and finite certificates, independently of quantization.

**Outputs/tests:** tests/test_oracle.py, test_updates.py, test_grid.py; oracle_validation.json with dimensions, tolerances, seeds, errors, and passed/failed cases. Choose exhaustive dimensions so all states really are enumerated; this is a correctness test, not a power calculation.

**Resources:** tiny CPU arrays; exhaustive growth is controlled by enumerated state count, not by pretending a large graph has been certified.

**Gate:** any discrepancy beyond the justified numerical tolerance is an implementation failure; block optimization. Exact agreement validates the implementation on those cases, not novelty or task value.

### WP02 — Build the leakage-safe evaluation harness and train pilot teachers

**Objective/risk:** ensure sensitivity, independent partitions, and functioning teachers.  
**Dependencies:** WP00–WP01.

**Actions:** generate T/V/C/D/P and a sealed inventory of official test IDs; reserve the final protected A/B allocation for WP06. Implement FP32 teacher training, NumPy/CPU reference inference, stable KL and paired metrics; train development teachers to an evidenced validation plateau. Implement identity/degraded-model harness controls. Collect no protected teacher or student metric results.

**Files/commands:** data.py, teacher.py, metrics.py, splits.json; proposed train and harness commands:

~~~bash
python -m fpcq teacher train --config configs/teacher.toml
python -m fpcq harness check --config configs/development.toml
~~~

**Outputs:** checkpoints with training/selection history and RNG; per-example development metrics; duplicate audit; harness_checks.json. Measure one epoch, inference batches, and peak memory.

**Gate:** continue if teacher training and the metric harness behave correctly. Broken training or insensitive numerical comparisons require repair. A low but well-understood teacher ceiling narrows the workload interpretation; it is not grounds to discard inconvenient quantization results.

### WP03 — Validate and tune the strong local baselines

**Objective/risk:** remove weak-baseline and optimizer-effort explanations.  
**Dependencies:** WP01–WP02.

**Actions:** implement Appendix A and B0–B8; validate tiny equations and special cases; run preliminary development calibration curves; record source/port/adaptation labels. Optimize quadratic baselines to their declared neighborhoods. Validate exact expected noise against explicit endpoint expectation on small N. Measure baseline costs here; finalize equal-total-cost budgets in WP06 after WP05 measures the complete joint optimizer.

**Outputs:** baseline_registry.json, parity reports, development frontier tables, baseline-specific locks/configurations, all failed attempts. Inputs are C/D/P only. Resources are measured per baseline; CPU factorization/gradient costs are included.

**Gate:** continue to a competitiveness claim only with valid strong baselines. A missing named baseline remains a comparison gap. A cheaper method already dominating all informative development settings is a legitimate reason to stop expensive candidate work or continue only a narrowly motivated diagnostic.

### WP04 — Measure the uncertainty model and available headroom

**Objective/risk:** determine whether the measured rays and rowwise gap are worth optimizing.  
**Dependencies:** correct oracle, trained pilot teachers, valid prefix baselines.

**Actions:** choose a reference quantized prefix from the tuned nominal family. Build alternative prefixes from independent calibration draws and relevant same-format code choices. On P, calculate error eᵏ for each alternative. For nonzero reference e⁰, compute

\[
\hat\alpha=\frac{\langle e^0,e^k\rangle}{\|e^0\|^2},\qquad
e^\perp=e^k-\hat\alpha e^0.
\]

Report amplitude distributions, signed alignment, orthogonal energy, and their output-weighted consequences. For near-zero e⁰, report a separate undefined-direction case using a threshold tied to numerical noise; never divide by an arbitrary small constant and call the coefficient meaningful. Compare with norm-matched scrambled directions. Measure J/U gaps and solve globally enumerable small-grid instances to separate optimizer failure from objective differences.

**Outputs:** drift_diagnostics.parquet or CSV, headroom.json, fixed-prefix activation caches, exact-small-grid results. Do not infer deployment probability from an adversarial box.

**Gate:** if the box is not predictive, continue only the explicitly labeled regularization hypothesis, or prefer B2. If rowwise and joint decisions are effectively identical at a well-measured local scale, avoid building a broad joint optimizer; record the negative/limited finding. If discrepancy is negligible or all models are already at the evaluation floor, select a previously defined harder bit/prefix condition using development data and explain the sensitivity problem.

### WP05 — Implement the minimum complete coherent quantizer

**Objective/risk:** test the selected contribution with exact updates.  
**Dependencies:** WP01 passes and WP04 identifies an informative condition.

**Actions:** implement sequential coordinate commits, shared candidate snapshots, exact table updates, full recomputation checks, pair moves, checkpoints, and complete cost boundaries. Begin with the final layer and common prefix. Add Numba only after reference correctness; compare trajectories. Add witness rejection only if profiling justifies it.

**Files/commands:** oracle.py, objective.py, joint_opt.py, checkpoint.py, kernels.py; proposed command:

~~~bash
python -m fpcq quantize --config configs/joint_fixed_prefix.toml
~~~

**Outputs/tests:** packed layers, move trace, objective history, witnesses, code-difference and cache-error reports; exact restart equivalence. Resource choice is from a complete-pass measurement, not isolated nanosecond timing.

**Gate:** accepted moves must decrease the declared objective beyond numerical uncertainty. If the implementation is too slow, preserve the objective while compiling/blocking. If only a reduced population fits, amend the study and restrict its claim. Do not silently turn independent amplitudes into a shared batch amplitude.

### WP06 — Run development comparisons and freeze confirmation

**Objective/risk:** determine variance, fair budgets, configuration, and test sensitivity without adaptive test selection.  
**Dependencies:** WP03–WP05.

**Actions:** run paired fixed-prefix and full-model pilots; tune only on D; use P to estimate variance and runtime. Complete matched-total-cost baseline frontiers now that WP05 has measured the joint optimizer. Evaluate the necessary ablations in Section 8. Pick the contribution-compatible candidate configuration, B★_fixed, and any separately claimed B★_practical. Apply Section 5.8 to choose protected allocation, fixed R/n and B_boot. Freeze primary/secondary claims, numerical tolerances, stopping rules, complete-pair failure handling, and cost boundaries. Review negative results before selection.

**Outputs:** preregistration.md, preregistration.json, method/configuration hashes, power_or_precision_report.json, baseline freeze, amendments to date. Freeze also the source revision of the analysis script.

**Gate:** proceed if the proposed confirmation can answer a material remaining question at measured local capacity. If it cannot, state the attainable effect/claim and either narrow or declare inconclusive. Do not escalate simply because a development p-value is promising.

### WP07 — Execute the controlled local confirmation

**Objective/risk:** obtain protected evidence for the frozen claim.  
**Dependencies:** WP06 signed-off artifact state; no protected results previously inspected.

**Actions:** execute all frozen methods on all fixed replicates/items. Interleave performance measurements. Resume only matching checkpoints. Apply Section 5.8's complete-pair and infrastructure-retry policy. Run the frozen analysis exactly once for the primary decision.

**Outputs:** raw predictions and paired per-example losses, complete timing/memory records, model/grid hashes, confidence intervals, cost frontiers, regressions and failure counts.

**Resources:** projected by WP06 with reserve for retries due solely to identified infrastructure failures. Scientific negative outcomes are not retry reasons.

**Gate:** improvement, tradeoff, negative evidence, and inconclusive results all advance to synthesis. A data/model mismatch invalidates affected runs and triggers repair plus a recorded analysis amendment; it is not averaged into a scientific conclusion.

### WP08 — Attribute, replicate, and stress-test the finding

**Objective/risk:** rule out extra compute, generic regularization, graph choice, and chance selection.  
**Dependencies:** correct primary run; retain the frozen method.

**Actions:** use reserved B if available for the independent item confirmation; otherwise run fresh pipeline seeds with the shared-test limitation disclosed. Execute prespecified robustness comparisons, confidence-interval sensitivity, bit-width boundaries, and rank/calibration-size effects. For post hoc changes, use a new protected population and label the amendment.

**Outputs:** replication report, attribution matrix, per-class and per-replicate results, failures/negative controls.

**Gate:** a replicated improvement over strong controls supports a scoped contribution. A gain erased by B2 or longer baseline search becomes a regularization/optimization finding. Inconsistent or tiny effects are reported with uncertainty and cost; they are not promoted by selecting favorable slices.

### WP09 — Conditional language transfer and backend validation

**Objective/risk:** test whether the mechanism transfers beyond the classifier.  
**Dependencies:** a useful held-out component effect or a specific scale-sensitive prediction, a resource forecast from the classifier implementation, and permitted public assets.

**Actions:** first run an asset/adapter probe: resolve revisions/licenses, load the model, validate document parsing, capture aligned teacher/prefix inputs, prove no-quantization forward parity, and measure a complete target-module pass. Admit transfer experiments only after that probe passes and its measured resource projection fits. Then execute Section 10.2 on a single late linear module under a common prefix. Only afterward consider all declared transformer block linears. Optionally validate MLX export and identical-kernel inference.

**Outputs:** adapter parity, immutable model/data revisions, document-clustered metrics, actual total bits, time/memory, and scope-specific conclusion.

**Gate:** a local time/memory limit does not refute the mechanism; it blocks that scale claim. Synthetic/classifier-only success does not justify language or semantic claims. A meaningful transfer failure is evidence against the tested generalization, retained in the report.

### WP10 — Synthesize and produce the reproducibility bundle

**Objective/risk:** make the result interpretable and independently resumable.  
**Dependencies:** all completed/blocked stages and their decision records.

**Actions:** generate tables/plots from immutable raw outputs; reconcile methodological amendments; run final integrity checks; state supported and unsupported claims separately. Produce positive, negative, and inconclusive conclusions with equal care.

**Outputs:** results_report.md, reproduction instructions, dependency locks, source revision, asset manifests, raw-result schemas, supported-claim table, limitations, and next-decision record. Never omit a failed method from the denominator of attempted runs.

**Gate:** complete when another agent can reproduce the reported analysis without chat memory. Remaining prerequisites are explicit blocks; remaining scientific uncertainty is connected to a discriminating next experiment.

---

## 8. Ablations, robustness, and failure analysis

The purpose is attribution, not an unlimited benchmark grid. Run low-cost indispensable controls first; escalate only when a specific explanation remains plausible.

| Test | Hold fixed | What it discriminates |
|---|---|---|
| Joint J versus rowwise U | Inputs, graph, grid, radius, moves, initialization | Output-sharing refinement |
| ρ = 0 | Graph and strong quadratic optimization | Contrast weighting versus finite robustness |
| λ = 0, joint closed form | Radius and error directions | Whether the graph is needed |
| Eq. (7) expected noise | Same directions, graph, tuning access | Minimax versus ordinary directional regularization |
| Common batch amplitude | Same radius and other settings | Loss of independent-example protection |
| Correct versus scrambled E | Norms, budget, labels/pairs | Whether measured direction geometry matters |
| Original versus degree/component-matched randomized forest | Node weights, λ, edge count | Pair semantics versus node-degree weighting |
| Disjoint meaningful pairs versus full forest | Same edge-weight normalization; report changed graph | Whether long tree paths or hubs drive the effect |
| No pairs versus adjacent pair search; common additional search | Candidate domain and time | Barrier escape versus objective contribution |
| Fixed-code grid versus tuned authentic formats | Full byte accounting | Fixed-grid result versus practical PTQ value |
| FP64 oracle versus proposed fast implementation | Complete inputs and tie policy | Numerical or backend artifacts |
| N/rank and bit-width sensitivity | Frozen analysis family | Overfitting, data insufficiency, high-precision loss floor |
| Centered final-logit objective shared by controls | All code/search inputs | Whether gains are mostly spent on a softmax-irrelevant common shift |
| Real alternative-prefix shifts versus synthetic rays | Declared reference prefix and evaluation population | Robustness-set realism |
| Layer-only versus complete sequential student | Teacher, data, metrics | Downstream/nonlinear transfer and accumulated error |

Randomizing a graph while preserving every degree and forest property is nontrivial. An exact simple control is to permute vertex identities on the existing topology, preserving component sizes and degrees as a multiset; record the assigned degree per example and compare stratified permutations if class/confidence degrees could confound. Do not call unconstrained random trees “degree matched.”

Domain-specific failure cases include: identical examples with inconsistent pairing; all teacher predictions in one class; highly imbalanced teacher confidence; near-zero E; dead ReLU channels; correlated or rank-deficient inputs; zero-range weight groups; clipping-dominated rows; final-logit common shifts; extreme hubs; large ρ whose worst corners are implausible; negative amplitude intervals when ρ > 1; and ties near quantization boundaries. Radius above one is allowed mathematically but reverses a measured direction for some probes; justify that separately.

A forest built only from nearest cross-class images can protect teacher errors as well as correct distinctions. Report teacher-correct and teacher-incorrect subsets descriptively without choosing the method on protected subsets. Improved teacher fidelity is not automatically improved label accuracy.

Classify failures explicitly:

- **Implementation/measurement failure:** oracle disagreement, stale cache, invalid target alignment, broken teacher, or omitted synchronization. Repair before interpretation.
- **Unsuitable proxy/sensitivity:** negligible drift, teacher loss floor, or a scope that cannot express the hypothesized mechanism. Choose an informative prespecified development condition or narrow the claim.
- **Resource limitation:** faithful procedure exceeds measured local memory/time capacity. Reduce implementation overhead first; any reduced experiment is labeled.
- **Evidence against the tested hypothesis:** correctly measured, adequately sensitive comparisons reject the predicted advantage.
- **Mechanism works without advantage:** exact robustness improves its own objective but is dominated by existing methods on task/cost.
- **Inconclusive:** uncertainty remains wider than the claim of interest; do not report a null as equivalence.

## 9. Reproducibility, interfaces, logging, and reporting

### 9.1 Minimal repository

Use one package and a small number of explicit interfaces; no experiment framework is required.

| Path | Responsibility |
|---|---|
| pyproject.toml; dependency lock | Package metadata, tested versions, reproducible environment |
| README.md | Setup, actual versus proposed commands, stage status, reproduction entry point |
| src/fpcq/__init__.py; __main__.py; cli.py | Package and module entry point; __main__ calls cli.main(); stage commands, validation, run identity |
| src/fpcq/data.py; teacher.py | Verified data partitions, teacher training/forward/checkpointing |
| src/fpcq/grid.py; export.py | Affine grids, strict packing, optional backend adapter |
| src/fpcq/graph.py | Exact deterministic forest construction and validation |
| src/fpcq/problem.py | Immutable layer problem and aligned activations |
| src/fpcq/oracle.py; objective.py | Reference scalar/vector DP, witnesses, analytic controls |
| src/fpcq/joint_opt.py; quadratic.py; kernels.py | Coherent descent, strong quadratic solvers, compiled exact loops |
| src/fpcq/baselines/ | Individually labeled compensation, CoreQ, YAQA, and gradient ports |
| src/fpcq/metrics.py; analysis.py | Per-example metrics, paired inference, immutable tables |
| src/fpcq/checkpoint.py; provenance.py | Atomic resume and asset/config/source identity |
| configs/ | TOML configurations, immutable frozen copies per run |
| tests/ | Exact fixtures, parity, invariance, harness, restart tests |
| data/manifests/ | Splits, checksums, source/license records; no hidden split generation |
| artifacts/runs/<run_id>/ | Config, environment, model, raw outputs, failures, costs |
| protocols/ | Preregistration, amendments, reference-inspection notes |
| decisions/ | Stage decisions and resumable project state |
| reports/ | Generated results, supported-claim table, limitations |

The project is programming work and should be versioned in its chosen repository. Do not rely on a chat attachment to preserve its state. Configure src-layout package discovery in pyproject.toml, install into the selected environment with python -m pip install -e ., then smoke-test python -m fpcq --help. These files and installation are prerequisites for every proposed module command.

### 9.2 Shared interfaces and invariants

~~~python
@dataclass(frozen=True)
class LayerProblem:
    teacher_weights: ndarray   # float64 [m,d], original FP32 values promoted
    teacher_inputs: ndarray    # actual FP32 hook inputs promoted to float64 [N,d]
    student_inputs: ndarray    # same capture convention, aligned and prefix-specific
    teacher_targets: ndarray   # float64 [N,m], immutable H_teacher @ W_teacher.T
    bias: ndarray              # original FP32 [m], unchanged
    item_ids: tuple
    prefix_hash: str
    input_hash: str
    graph: Forest
    node_weights: ndarray      # [N], nonnegative
    edge_weights: ndarray      # [M], nonnegative

@dataclass
class QuantizationResult:
    codes: ndarray
    grid: AffineGrid
    objective_name: str
    objective_value: float
    converged_neighborhood: str | None
    checkpoint_path: str
    cost_record: dict
    diagnostics: dict

def fit_layer(problem, grid, initial_codes, method_config, budget, resume=None):
    """Return QuantizationResult; never read protected labels/results."""

def fit_model(teacher, calibration_manifest, method_config, budget, resume=None):
    """For global gradient methods; return the same model-level result schema."""

def evaluate(model, evaluation_manifest, metric_config, protocol_hash):
    """Compute frozen metrics; require a matching protocol for protected splits."""
~~~

These are proposed interfaces. The eventual implementation must import the named types or replace the pseudocode annotations with actual definitions. It must not treat the snippets as already installed APIs.

Reject nonfinite arrays, misaligned IDs, invalid integer graph endpoints, repeated/self edges, cycles, negative weights, mismatched group grids, or inconsistent shapes. Validate before casts. Keep original W★, hook inputs, targets, and grid tables immutable: copy and mark NumPy buffers read-only, because a frozen dataclass alone does not freeze array contents. Compensation and optimizer work arrays are separate writable objects; verify input hashes.

The pair-selection fix is mandatory: return an empty list when requested neighbors = 0 or d < 2; exclude self indices explicitly; require nonnegative integer neighbor counts. The original slicing bug did not invalidate the reported main experiment, but it breaks minimal test cases.

### 9.3 Configuration example

This is an exploratory starting configuration, not a frozen confirmation claim.

~~~toml
[study]
phase = "exploratory"
dataset = "fashion_mnist"
root_seed = 20261029
primary_bits = 2
secondary_bits = [3, 4]
primary_metric = "teacher_kl_nats_per_example"

[quantization]
group_size = 32
scale_dtype = "float32"
oracle_dtype = "float64"
activation_dtype = "float32"
objective = "joint_forest"
rho = 0.25
lambda_contrast = 0.25
calibration_items = 512
code_domain = "all_levels"

[optimization]
commit_rule = "sequential_exact_decrease"
single_candidates = "all_levels"
pair_candidates = "adjacent_3x3"
pair_neighbors = 2
pair_policy = "abs_input_gram_correlation_topk_union"
stopping_rule = "development_calibrated_then_frozen"
fastmath = false

[evaluation]
protected_results_locked = true
common_prefix_mechanism_track = true
complete_cost_accounting = true
~~~

The parser must reject unresolved confirmation fields: R/n and panel/item IDs; candidate and B★_fixed identities plus exact grid/precision policy; any separate practical comparator and track; positive-ρ/positive-λ contribution label or recorded pivot; conditional-panel estimand; bootstrap B_boot/seed/strata/quantiles; complete-pair failure policy; numerical tolerance; budgets and stopping rule; protected teacher-metric lock; analysis revision. A confirmation config cannot inherit mutable defaults from a development file. Invalid family/parameter combinations and mixed comparator regimes are configuration errors, not report-time warnings.

### 9.4 Required experiment record

At minimum, the run JSON contains:

~~~json
{
  "schema_version": 1,
  "phase": "exploratory_or_confirmatory",
  "run_id": "content_hash_assigned_by_implementation",
  "status": "planned_running_completed_failed_blocked",
  "source_commit": null,
  "dirty_patch_sha256": null,
  "environment_manifest_sha256": null,
  "protocol_sha256": null,
  "config_sha256": null,
  "teacher_sha256": null,
  "dataset_revision": null,
  "split_manifest_sha256": null,
  "calibration_ids_sha256": null,
  "prefix_sha256": null,
  "graph_sha256": null,
  "grid_sha256": null,
  "method_name": null,
  "method_status": "published_equations_port_adaptation_or_diagnostic",
  "seed_manifest": {},
  "cost_seconds": {
    "asset_setup": null,
    "activation_collection": null,
    "graph_build": null,
    "statistics_or_sketch": null,
    "compile_once": null,
    "optimization": null,
    "export": null,
    "evaluation": null,
    "total_calibration": null,
    "total_method_selection": null
  },
  "peak_memory_bytes": null,
  "payload_bytes": null,
  "container_bytes": null,
  "accepted_moves": null,
  "candidate_evaluations": null,
  "oracle_calls": null,
  "termination_reason": null,
  "checkpoint_sha256": null,
  "failure": null
}
~~~

Null denotes an unfilled template field, not an observed zero. Completed records require all applicable values and explicit reasons for inapplicable fields.

Per-example natural-result rows contain run_id, replicate_id, teacher_id, calibration_draw_id, split, stable item_id, method, bit_width, true label, teacher/student prediction, teacher_KL_nats, label_NLL_nats, centered_logit_MSE, and evaluation status. Optional raw logits are stored once with IDs. A separate stress record contains the full set identity, graph/radius/weight hashes, J, U, witness, and normalization.

Every failed run records the exception, last valid checkpoint, elapsed cost, and scientific/infrastructure classification. A rerun gets a linked new attempt record. Do not silently replace a bad result file.

### 9.5 Randomness and timing

Use a deterministic root SeedSequence and named child streams for teacher initialization, minibatch order, calibration sampling, graph controls, optimizer restarts, and bootstrap. Save the concrete derived integers and generator algorithms; do not depend on an undocumented Python hash. Keep confirmation streams distinct from development.

Timing boundaries include activation collection, graph construction, statistics, optimization, export, and selection. Reusable teacher/model downloads and one-time compiler warm-up are disclosed separately; do not hide project-specific sketch collection as a free pretrained asset.

Warm up the measured code path until behavior stabilizes, synchronize around accelerated regions, use perf_counter_ns for elapsed time, and report process/thread/power conditions. Interleave methods within timing blocks. Record a distribution and robust central estimate rather than one favorable run. Report allocator/RSS/driver memory with platform units and avoid double-counting shared unified memory.

### 9.6 Reporting requirements

Every result table identifies workload, data population, grid/precision, prefix policy, baseline status, cost boundary, independent replicate count, uncertainty method, and whether the result was selected or protected. Show absolute metrics before relative changes.

Include all methods and predeclared conditions, including regressions, unstable seeds, failed selectors, and unavailable baselines. Plot quality against complete calibration time and actual bytes where informative. A statistically real effect can still be practically unhelpful; a component improvement can still be valuable without an end-to-end claim. State which of these occurred.

The project-state file records the last completed gate, current blocking issue, next exact command/configuration, active amendments, and immutable artifact paths. This makes recovery independent of chat history.

---

## 10. Outcome interpretation and evidence-triggered continuations

### 10.1 Decision table

| Observed outcome | Valid interpretation | Next action |
|---|---|---|
| J optimizer lowers J but not held-out KL | Better fit to the selected adversary; no fidelity gain | Inspect ray/graph realism; compare B2 and output-aware baselines; do not scale automatically |
| J and U choose almost identical codes with tight uncertainty on the tested scale | Original relaxation appears adequate locally | Prefer cheaper rowwise implementation; retain the mathematical counterexample |
| B2 matches or beats minimax at lower cost | Expected directional regularization explains the benefit | Select the simpler quadratic approach; revise contribution to an empirical/engineering finding |
| λ = 0 matches full forest | Graph cost has no demonstrated value | Prefer exact no-edge formula; drop semantic-contrast claims |
| Robust gain disappears after equal-cost baseline tuning | Original advantage was an optimizer/budget artifact | Report correction/negative result; do not retain superiority wording |
| KL improves but task loss or accuracy regresses | Fidelity tradeoff | Report magnitude/uncertainty and application relevance; no blanket practical-win claim |
| Protected KL improvement and acceptable guardrails replicate | Scoped evidence for better quantization on this workload | Validate cost Pareto frontier and language transfer |
| Significant effect is tiny relative to costs and residual uncertainty | Genuine but limited improvement | Quantify it; no “breakthrough” wording |
| Effect grows with dimension/correlation after controlled scaling | Plausible scale-dependent mechanism | Escalate only that tested dimension with measured memory/time and new confirmation |
| No gain with inadequate statistical sensitivity | Inconclusive | Report detectable-effect limit; obtain justified independent evidence or stop with that limit |
| Exact oracle fails or resumed state differs | Invalid implementation | Repair and revalidate before interpreting any metrics |
| Novelty-equivalent prior art appears | Replication or extension of existing work | Attribute it and revise the comparison; do not rename it to manufacture novelty |

A practical advantage should be expressed in a clear tradeoff: for example, improved teacher fidelity at the same exported bytes with a quantified one-time calibration cost. If deployment never reuses the quantized artifact enough to justify that cost, the benefit may be academically interesting but operationally weak. There is no universal percentage cutoff without an application.

### 10.2 Conditional small-language-model transfer

**Asset:** EleutherAI Pythia-70M, research model card and Apache-2.0 weights [D02]. Pin a resolved full model revision and tokenizer revision; prefer safetensors. The card describes six blocks, width 512, and about 70.4 million parameters, with roughly 18.9 million nonembedding parameters. Read the actual configuration and count unique parameter storage after loading.

**Data:** Salesforce WikiText, configuration wikitext-2-raw-v1 [D03]. Keep official train/validation/test boundaries and group by source article/document. Use train only for calibration, validation for development, and test only after freeze. The card's license metadata and prose refer to different Creative Commons versions; preserve the source notices and resolve redistribution terms before redistributing corpus bytes. Downloadability is not a complete license specification.

Treat dataset rows as lines, not documents. For the pinned raw files, strip outer whitespace only for boundary recognition: a first-level title has exactly one equals sign at each end, with a space after the opening sign and before the closing sign; a second/third-level heading remains inside the current article. For example, a pattern equivalent to ^=\s+[^=].*?\s+=$ with explicit rejection of a second leading/trailing equals sign identifies first-level markers. Retain the original text and join consecutive rows with one newline; include each article title in its own article. Nonempty preamble before the first title blocks admission until explicitly resolved; do not silently make it a document. Record original row ranges, title hashes, document IDs and parser version. A two-article fixture with nested headings must produce exactly two documents. Check the actual pinned format and reject unexpected boundaries rather than guessing.

**Smallest transfer experiment:** use a common quantized prefix and quantize one late MLP output projection. Capture actual module inputs with hooks on full teacher and student forward passes on identical token IDs, masks, and positions. An attention/residual model is not a simple sequential MLP. Assert relevant GPT-NeoX module names and shapes from the pinned implementation; respect parallel residual branches, normalization, QKV layout, and tied parameters. Teacher/student equality with no quantization is the mandatory adapter test.

The initial intervention is the final block's MLP output projection: resolve its module path from the pinned model, assert the expected width relationship, and store the exact selector. Build the common prefix with B1's validated asymmetric-target quadratic reconstruction applied to declared linear weights in earlier complete blocks; freeze its language-development configuration and module order separately. Do not inherit the Fashion-MNIST B★_fixed identity, which could be a global gradient method without a transferable prefix policy. Leave the final block's other modules and every later operation floating. Save the full module-by-module state manifest. This produces actual earlier-block mismatch without pretending that a parallel attention branch is a sequential predecessor. Any change to this intervention is a new recorded condition.

Select calibration token positions prospectively by document and position strata. All compared methods receive the same selected target positions and context. Keep layer-input calibration N distinct from the number of documents, sequences, and context tokens. Report all three.

For a first graph, use a declared forest on teacher activation-space distances between contexts with different teacher next-token argmax predictions, with norm convention fixed on development data. This is a **changed graph policy for text**, and it preserves activation/output contrasts; it is not a validated language-semantic oracle. Include no-edge, expected-noise, and randomized-pair controls. Minimal-pair language tests can be a later separately documented workload with explicit provenance and meaning.

Use fixed-length context windows without crossing document boundaries. Select and freeze context length from a development sensitivity curve and memory calibration; document how the first token and short documents are handled. For NLL, score each intended token once, with an explicit context/truncation policy; mask padding and avoid duplicate scoring of overlap. Compute perplexity as exp(total NLL / total scored tokens), not an average of sequence perplexities. Prefer streaming sequence batches while retaining each batch's full vocabulary logits. If vocabulary chunking is necessary, first compute a global logsumexp across all vocabulary chunks for each teacher/student token, then accumulate KL using those global normalizers. Never apply separate chunk softmaxes. Record whether logits are cached or recomputed and count that cost. Validate against full-vocabulary KL for multiple chunk sizes, including uneven final chunks, on the same scored positions.

The independent evaluation unit is the document. Bootstrap or analyze paired document totals with token counts, not every token as an independent observation. Pretraining contamination can affect absolute task performance; this is a matched teacher/student fidelity comparison, not a demonstration of newly learned capability.

**Baseline condition:** retain exact expected noise, original FPCQ, strong target reconstruction, and validated compensation/CoreQ references. Add a faithful output-aware or full-KL competitor before making broad language quantization superiority claims. The exact ten-class YAQA specialization does not silently become a transformer sketch; token mixing and sampled output gradients require the published transformer construction or an explicitly validated adaptation.

Only after the single-module result and complete cost estimate are informative should the agent quantize all declared transformer-block linear weights. Keep embeddings, normalizations, biases, and the LM head policy identical and count them. With FP32 embeddings/head retained, four-bit quantization of at most 18.9 million of 70.4 million parameters would reduce the ideal weight footprint only from about 282 MB to at least about 216 MB before metadata. This estimate explains why this small model is a mechanism test, not an eightfold whole-model compression demonstration.

Failure to transfer is a useful result. If the full-model effect is too small or expensive, report the layer claim without extrapolating to larger models. If a specific interaction absent from this model is hypothesized, state a locally testable prediction or identify it as unverified; do not excuse every negative result with hypothetical future scale.

### 10.3 Conditional mathematical branch

The mathematical branch remains independent of practical quantization. Pursue it if the project values an exact robustness theorem and targeted prior-art checking still leaves a defensible question.

**First reproduce the strip frontier without conflating it with calibration.** Under exact vertical covariance G(u,v) = μ(v+g(u)), a finite grid gives constraints

\[
-E\le\mu[g(w)-g(u)-(w-u)^2/2]\le E
\]
for |u| ≤ L and |w−u| ≤ r. Anchor one g value and minimize E with a sparse LP. Include the two-edge and four-edge configurations at the transitions; evaluate the explicit continuous optimizer. This is a numerical check, not a continuum proof for arbitrary readouts.

**Then define the compact-domain problem precisely before solving it.** Choose a finite semantic interval and state exactly which probe bases, endpoints, and vertical comparisons are allowed. Build a finite graph of rational source states and their probe endpoints. Variables are readout values and E; every probe gives two difference inequalities, and every allowed same-x vertical comparison gives a lower-slope inequality. Solve the LP, inspect dual path/cycle certificates, and check rational certificates on small cases. Refine the mesh to diagnose candidates.

A compact-domain theorem requires analytical lower/upper bounds or certified discretization error. A grid LP alone cannot prove it. Changing the vertical domain or admitting exterior states can change which loops are legal and hence the optimum. Compare against the classical approximation/cycle/optimal-design literature again for the exact new formulation [M05–M07].

**Branch acceptance:** a precise theorem, verified assumptions and certificates, and a bounded novelty statement. It supplies no model-accuracy, compression, or runtime claim unless a separate connection to an actual quantizer is proved and tested.

---

## 11. Audit record, unresolved risks, and limitations

### 11.1 Concrete weaknesses found and revisions made

The preparation used separate code/result, prior-art, and hardware/implementation investigations, followed by three distinct full-draft challenge audits and targeted rechecks after revision. These are agent-assisted reviews, not external peer review or formal verification.

| Defect or challenge | Evidence | Revision in this plan |
|---|---|---|
| Stress comparisons used different uncertainty sets | Source evaluator recomputes E from each method's prefix | Separate common-prefix mechanism study from natural whole-model evaluation |
| Scalar row adversaries were not one physical vector disturbance | Max/sum inequality and exact ranking reversal | Select vector objective; retain U as mandatory closest control |
| Joint updates were assumed to retain row batching | Coupled max changes acceptance semantics | Sequential coherent commits, candidate batching only, exact restart checks |
| Scalar bipartite proof could be overgeneralized | Frustrated vector four-cycle | Require forests for selected solver |
| Apparent robust gain could be ordinary regularization | Exact expectation identity | Make Eq. (7) a mandatory strongly optimized baseline |
| External baselines were outdated or weak | GPTAQ/ResComp/CoreQ/YAQA source research | Add mathematical reference recipes and parity gates |
| CoreQ pseudocode has a diagonal overwrite inconsistency | Printed algorithm and consistent triangular objective inspected | Keep the original Cholesky diagonal; validate branch sums |
| YAQA appendix factor notation can be misread | Main equation uses strict factors | Explicitly subtract identities and test two implementations |
| Tiny-oracle cases hit a source boundary bug | Executed zero-neighbor/single-channel pair tests | Correct selection and reject self pairs |
| Nominal and robust budgets were unequal | Source times and different pair domains | Matched search plus complete cost frontiers |
| Shared seeds/test examples could inflate certainty | Same source test-index hash | Conditional-panel image inference, class-adjusted variance planning, and fresh-panel replication |
| First-layer robustness could be misattributed | E = 0 exactly for weight-only input layer | Negative control and quadratic fast path |
| M3 compatibility was inferred from generic branding | Release-specific OS/dtype restrictions | Hardware discovery and faithful CPU oracle before dependent work |
| Existing graph construction could exhaust memory | O(N²d) temporary | Tiled exact graph construction |
| Packed storage could be mistaken for packed inference | Source dequantizes for matmul | Separate export/parity/backend gate and real byte accounting |
| Mathematical theorem could be mistaken for a quantization guarantee | Different objective and unbounded domain | Separate mathematical branch and practical hypothesis |

The three full-draft passes produced additional corrections:

| Audit pass | Weaknesses found | Changes and recheck outcome |
|---|---|---|
| Scientific validity and prior art | False finite-versus-derivative rationale for an affine residual; incomplete gradient recipes and pair selector; singular projection and zero-edge caveats | Removed the rationale; specified local gradient controls and pair policy; corrected pseudoinverse and path assumptions. The recheck found no new main-objective or baseline-formula error. Its minor pair-default provenance and zero-gradient fallback findings were then corrected. |
| Experimental validity | Mixed comparator regimes; degenerate candidates could inherit full-mechanism claims; discretionary sample-size/CI rules; survivor-only failure risk; early teacher-test ambiguity | Bound the primary to B★_fixed and positive ρ/λ; made inference conditional on a frozen panel; supplied class-adjusted planning, paired stratified bootstrap, explicit quantile/test calculations, missing-pair rules, and protected teacher locks. Recheck confirmed the six material findings resolved; final precision clarifications were incorporated. |
| Implementation and feasibility | Missing CLI entry point and edge orientation; ambiguous arithmetic; centered-cache error risk; circular stage dependencies; incomplete text parser/KL normalization and resume state | Added exact interfaces, effective FP32 legal levels, coherent centered acceptance, stage admission rules, article parsing, global vocabulary normalization, and complete optimizer checkpoints. Recheck confirmed the ten original findings resolved. Its remaining language-prefix selector issue was corrected to B1 with a separate language configuration. |

After the last corrections, all three scoped closure checks reported no additional substantive correction. These rechecks close the identified specification defects, not the scientific questions. Baseline software parity, M3 capability, runtime, protected results and any proof beyond the checked identities remain execution gates. The plan does not assign itself a quality score or imply that another review cannot find an error.

### 11.2 Remaining risks

- No current evidence establishes a task benefit for the selected joint quantizer; only its oracle and potential code-selection distinction have been checked.
- The measured original relaxation gap is small; useful headroom may be negligible.
- Independently varied examples may remain an unrealistic prefix uncertainty set even after fixing output sharing.
- The selected graph is a heuristic and may protect teacher mistakes or act only as weighting.
- Joint optimizer scaling may dominate; exactness per candidate does not make a whole sweep inexpensive.
- Local equation ports are not official implementation reproduction until parity is established.
- CoreQ official source was not verified; some GPTAQ/ResComp code-file access was blocked; CL-Calib and TFIC full-text access was incomplete.
- Statistical sensitivity depends on actual paired-loss variance and teacher variability; no significance or equivalence result is promised.
- The M3 Max workload has not been executed. Package support and array estimates are prerequisites, not measurements.
- Public novelty searches have blind spots. A new name or combination does not prove priority.
- The original archive's source-license status is unspecified.
- A computational PoC cannot establish real language semantics, deployed system usefulness, or a universal robustness guarantee.

### 11.3 Final acceptance checklist for execution

Before reporting a supported improvement, require all of the following:

- The exact uncertainty set and output-sharing rule are named.
- The target layer/model, data population, bit format, and baseline statuses are explicit.
- Oracle and update tests pass, including real edge cases and resume identity.
- The strong simple and contemporary local baselines are validated and fairly tuned, or the claim explicitly excludes missing comparisons.
- Calibration/tuning/measurement costs and actual storage are included.
- Confirmation was frozen before protected results and used the declared independent units.
- Required ablations discriminate minimax, contrasts, output sharing, and search effort.
- Important regressions and negative results remain visible.
- The claim follows the measured scope; unsupported scale, semantic, speed, or novelty claims are absent.

---

## 12. References and search-coverage record

All web sources below were investigated on 29 October 2026. “Inspected” describes actual access during preparation; it does not mean an implementation or benchmark was reproduced. Mutable repository URLs identify authoritative projects; the executing agent must record resolved source commits and dependency locks.

### 12.1 Supplied evidence

**[U1] Supplied FPCQ archive**, FPCQ(1).zip, internal notes dated 28 October 2026. SHA-256: 66dc3299a4786788e4280066b2b74c7e5b2e8e0f580f5bdc801fda384a18de75. Inspected FPCQ.md, FPCQ_research_note.md, README, all eight Python files, result JSON, saved teachers, and packed layers. Local source; no public URL or license was supplied. Its essential equations, findings, and conventions are reproduced in this plan.

**[U2] Supplied PDF**, *Robustness to straight probes while preserving meaningful change: An autonomous mathematical investigation inspired by quantization*, 28 October 2026, ten pages. SHA-256: 7a3e5b2ffb0a3395a553f3c1e2c23261ba9633588d2791b542b039093cf47c3f. Read extracted text and rendered formula pages; inspected strip/global proofs and bibliography. The PDF's stated verification-code bundle was not present in U1.

### 12.2 Quantization and related learning methods

**[Q01]** Elias Frantar, Saleh Ashkboos, Torsten Hoefler, Dan Alistarh. *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers*. ICLR 2023; arXiv:2210.17323, 2022. [Paper](https://arxiv.org/abs/2210.17323); [official repository](https://github.com/IST-DASLab/gptq). Inspected algorithm and matrix implementation/CUDA assumptions. Informs compensation baseline; published large-GPU results are contextual.

**[Q02]** Yuhang Li, Ruokai Yin, Donghyun Lee, Shiting Xiao, Priyadarshini Panda. *GPTAQ: Efficient Finetuning-Free Quantization for Asymmetric Calibration*. ICML 2025, PMLR 267:36690–36706; arXiv:2504.02692v3. [Proceedings](https://proceedings.mlr.press/v267/li25dn.html); [paper](https://arxiv.org/pdf/2504.02692); [official repository](https://github.com/Intelligent-Computing-Lab-Panda/GPTAQ). Inspected residual decomposition and Algorithm 1; individual code-file retrieval was incomplete. Rules out broad asymmetric-calibration novelty.

**[Q03]** Shuaiting Li, Juncan Deng, Kedong Xu, Rongtao Deng, Hong Gu, Minghan Jiang, Haibin Shen, Kejie Huang. *Rethinking Residual Errors in Compensation-based LLM Quantization*. ICLR 2026; arXiv:2604.07955v1, 9 April 2026. [Paper](https://arxiv.org/html/2604.07955v1); [official repository](https://github.com/list0830/ResComp). Inspected Eqs. 14–21, algorithm, and README sensitivity note. Core source access/license not fully verified; informs the compensation-aware comparator.

**[Q04]** Seohyeon Cha et al. *CoreQ: Learning-Free Mismatch Correction and Successive Rounding for Quantization*. arXiv:2602.05902v2, 8 May 2026. [Paper](https://arxiv.org/html/2602.05902v2); [versions](https://arxiv.org/abs/2602.05902). Inspected correction/rounding equations and pseudocode. February v1 used the title *Regularized Calibration with Successive Rounding for Post-Training Quantization*. Official code unverified.

**[Q05]** Albert Tseng, Zhaofeng Sun, Christopher De Sa. *Model-Preserving Adaptive Rounding*. ICML 2026, PMLR 306; arXiv:2505.22988v3, 3 June 2026. [Paper](https://arxiv.org/html/2505.22988v3); [proceedings](https://proceedings.mlr.press/v306/tseng26b.html); [code](https://github.com/Cornell-RelaxML/yaqa-quantization). Inspected rounding/sketch equations, appendix and repository license. No published benchmark reproduced.

**[Q06]** Markus Nagel, Rana Ali Amjad, Mart van Baalen, Christos Louizos, Tijmen Blankevoort. *Up or Down? Adaptive Rounding for Post-Training Quantization*. ICML 2020, PMLR 119:7197–7206. [Proceedings and paper](https://proceedings.mlr.press/v119/nagel20a.html). Inspected Eqs. 21–25 and floor/ceil formulation. Establishes learned and asymmetric rounding; the local MLP experiment is a declared adaptation.

**[Q06a]** Qualcomm AI Research. *FastForward AdaRound API reference*, inspected 29 October 2026. [Official implementation documentation](https://qualcomm-ai-research.github.io/fastforward/latest/reference/fastforward/algorithms/adaround/). Inspected defaults, loss reductions, initialization and activation omission. Mutable documentation: pin the corresponding source revision. Supplies development starting settings, not an identical original-paper experiment.

**[Q07]** Yuhang Li et al. *BRECQ: Pushing the Limit of Post-Training Quantization by Block Reconstruction*. ICLR 2021; arXiv:2102.05426. [Paper](https://arxiv.org/pdf/2102.05426); [official repository](https://github.com/yhhhli/BRECQ). Inspected block-reconstruction motivation and repository availability, not a full implementation audit. Different block/activation settings limit direct numerical comparison.

**[Q08]** Jerry Chee, Arturs Backurs, Rainie Heck, Li Zhang, Janardhan Kulkarni, Thomas Rothvoss, Sivakanth Gopi. *DiscQuant: A Quantization Method for Neural Networks Inspired by Discrepancy Theory*. COLT 2025, PMLR 291:924–951. [Proceedings](https://proceedings.mlr.press/v291/chee25a.html); [paper](https://arxiv.org/pdf/2501.06417); [repository](https://github.com/microsoft/DiscQuant). Inspected theoretical/practical distinction and practical optimizer. The proceedings also links the author's repository; resolve the authoritative revision during acquisition. Restrained rounding and KL optimization are prior art.

**[Q09]** Yuzhang Shang, Gaowen Liu, Ramana Rao Kompella, Yan Yan. *Enhancing Post-training Quantization Calibration through Contrastive Learning*. CVPR 2024:15921–15930. [Official proceedings](https://openaccess.thecvf.com/content/CVPR2024/html/Shang_Enhancing_Post-training_Quantization_Calibration_through_Contrastive_Learning_CVPR_2024_paper.html); DOI 10.1109/CVPR52733.2024.01507. Official indexed abstract/excerpts inspected; direct full text returned access errors. Supports broad overlap, not a verified detailed implementation comparison.

**[Q10]** Wonpyo Park, Dongju Kim, Yan Lu, Minsu Cho. *Relational Knowledge Distillation*. CVPR 2019:3967–3976. [Author project](https://cvlab.postech.ac.kr/research/RKD/); [paper](https://arxiv.org/pdf/1904.05068). Project and paper opened; relational distance/angle motivation inspected. Earlier teacher/student relation preservation resembles, but does not establish equivalence to, the selected forest loss.

**[Q11]** Xiuying Wei, Ruihao Gong, Yuhang Li, Xianglong Liu, Fengwei Yu. *QDrop: Randomly Dropping Quantization for Extremely Low-bit Post-Training Quantization*. ICLR 2022. [Paper](https://arxiv.org/pdf/2203.05740); [official repository](https://github.com/wimh966/QDrop). Inspected stochastic activation-quantization omission and procedure. Motivates mixing controls; Eq. (7) is this plan's direct derivation, not a claimed QDrop theorem.

**[Q12]** Anh Nguyen Duc, Linh Ngo Van, Sang Dinh Viet, Hoang Ta Duy, Tung Nguyen Son, Canh Dinh The, Tu Vu Van. *Escaping Rounding Barriers in Large Language Model Quantization via Barrier-Aware Ising Correction*. Journal of Science and Technology – Data Science and Artificial Intelligence 36(1), 15 October 2026. [Publisher](https://jst.vn/index.php/dsai/article/view/1545); [DOI](https://doi.org/10.51316/jst.194.dsai.2026.36.1.1). Abstract/reference metadata inspected; full text/code not obtained. Enough to invalidate generic cluster-move novelty, insufficient for a faithful local benchmark.

**[Q13]** Ji Lin et al. *AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration*. MLSys 2024. [Proceedings](https://proceedings.mlsys.org/paper_files/paper/2024/hash/42a452cbafa9dd64e9ba4aa95cc1ef21-Abstract-Conference.html); [official repository](https://github.com/mit-han-lab/llm-awq). Proceedings and installation/search/inference scope inspected. Scaling/format and CUDA deployment differ from fixed-grid scalar code selection.

**[Q14]** Miles Williams, Nikolaos Aletras. *On the Impact of Calibration Data in Post-training Quantization and Pruning*. ACL 2024; arXiv:2311.09755v2. [Paper](https://arxiv.org/html/2311.09755v2); DOI 10.18653/v1/2024.acl-long.544. Inspected study scope and calibration-sensitivity findings. Supports explicit calibration draws and separation of data/seed dependence; tables were not reanalyzed here.

### 12.3 Mathematical antecedents

**[M01]** Laurent El Ghaoui, Hervé Lebret. *Robust Solutions to Least-Squares Problems with Uncertain Data*. SIAM J. Matrix Analysis and Applications 18(4):1035–1064, 1997. [Author PDF](https://people.eecs.berkeley.edu/~elghaoui/Pubs/rob-ls.pdf); DOI 10.1137/S0895479896298130. Inspected structured uncertainty and Section 5. Broad robust fitting and hard box cases are established; exactness depends on uncertainty structure.

**[M02]** Stephen Becker, Richard Clancy. *Robust Least Squares for Quantized Data Matrices*. Signal Processing 176:107711, 2020. [Paper](https://arxiv.org/pdf/2003.12004); [DOI](https://doi.org/10.1016/j.sigpro.2020.107711). Main box-uncertainty counterpart inspected. Its matrix-entry uncertainty differs from measured per-example rays.

**[M03]** Frank R. Kschischang, Brendan J. Frey, Hans-Andrea Loeliger. *Factor Graphs and the Sum-Product Algorithm*. IEEE Transactions on Information Theory 47(2):498–519, 2001. [Author-hosted PDF](https://www.isiweb.ee.ethz.ch/papers/arch/aloe-2001-1.pdf); DOI 10.1109/18.910572. General message-passing/semiring background for exact tree inference; the specific recursion is proved directly in Section 4.

**[M04]** Vladimir Kolmogorov, Ramin Zabih. *What Energy Functions Can Be Minimized via Graph Cuts?* IEEE TPAMI 26(2):147–159, 2004. [Primary publication page](https://www.microsoft.com/en-us/research/publication/what-energy-functions-can-be-minimized-via-graph-cuts/); [DOI](https://doi.org/10.1109/TPAMI.2004.1262177). Binary submodularity/cut characterization inspected. It does not permit arbitrary signed vector couplings on bipartite graphs.

**[M05]** Holger Dette, Viatcheslav B. Melas, Petr Shpilev. *Optimal designs for estimating individual coefficients in polynomial regression with no intercept*. Discussion Paper 13/2019; Statistics & Probability Letters 158:108636, 2020. [Primary PDF](https://eldorado.tu-dortmund.de/bitstream/2003/38137/1/DP_1319_SFB823_Dette_Melas_Shpilev.pdf); DOI 10.1016/j.spl.2019.108636. Theorem 2.1 and Eq. (3.2) inspected; the PDF's global extremal polynomial family is already known.

**[M06]** S. P. Diliberto, E. G. Straus. *On the approximation of a function of several variables by the sum of functions of fewer variables*. Pacific Journal of Mathematics 1(2):195–210, 1951. [Publisher PDF](https://msp.org/pjm/1951/1-2/pjm-v1-n2-p04-s.pdf). Inspected closed-path characterization and problem definition. Independent univariate summands differ from one potential appearing twice, so this is related machinery, not an identified exact strip-formula equivalent.

**[M07]** Richard M. Karp. *A Characterization of the Minimum Cycle Mean in a Digraph*. Berkeley report UCB/ERL M77/47, 1977; Discrete Mathematics 23:309–311, 1978. [Primary report page](https://www2.eecs.berkeley.edu/Pubs/TechRpts/1977/29121.html). Bibliography/report access inspected; supports classical cycle-mean antecedents, not a novelty conclusion about the strip.

**[M08]** Ernst Hairer. *Geometric Numerical Integration*, lecture notes, 2010, based on Hairer, Lubich, Wanner (2006). [Author notes](https://www.unige.ch/~hairer/poly_geoint/week1.pdf). First-integral definitions and geometric-integration context inspected. Tangent invariance and finite straight-step invariance must not be conflated.

### 12.4 Hardware, software, and assets

**[S00]** Apple. *MacBook Pro (14-inch, M3 Pro or M3 Max, Nov 2023) — Technical Specifications*. [Official specifications](https://support.apple.com/en-us/117736). Memory/chip variants inspected; no inference about the user's installed configuration.

**[S01]** PyTorch Foundation. *PyTorch 2.15 Release Blog*, 28 October 2026. [Official release](https://pytorch.org/blog/pytorch-2-15-release-blog/). Compatibility section inspected; Python/macOS requirements inform discovery.

**[S02]** PyTorch. *MPS backend* and synchronization/memory API documentation, version 2.15. [Backend](https://docs.pytorch.org/docs/2.15/notes/mps.html); [synchronize](https://docs.pytorch.org/docs/2.15/generated/torch.mps.synchronize.html); [tensor memory](https://docs.pytorch.org/docs/2.15/generated/torch.mps.current_allocated_memory.html); [driver memory](https://docs.pytorch.org/docs/2.15/generated/torch.mps.driver_allocated_memory.html). Documentation/search-access inspection covered double restrictions and measurement APIs; some direct backend-page retries failed. Verify installed-version behavior locally.

**[S03]** Apple Developer. *Accelerated PyTorch training on Mac*. [Official overview](https://developer.apple.com/metal/pytorch/). Inspected older release/OS statements and reconciled them with S01 rather than assuming consistency.

**[S04]** MLX documentation, inspected version 0.32.4. [Installation](https://ml-explore.github.io/mlx/build/html/install.html); [quantize](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.quantize.html); [quantized_matmul](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.quantized_matmul.html). Affine groups/bits, scale/bias and packing constraints inspected. No kernel execution or speed measured.

**[S05]** Numba. [Installation/support documentation](https://numba.readthedocs.io/en/stable/user/installing.html); [0.68.1 release notes](https://numba.readthedocs.io/en/stable/release/0.68.1-notes.html); [published wheels](https://pypi.org/project/numba/0.68.1/). Native ARM64 support and version compatibility inspected. Lock a working combination; a supported wheel is not trajectory parity.

**[S06]** IST-DASLab. [GPTQ matrix implementation](https://github.com/IST-DASLab/gptq/blob/main/gptq.py). Inspected compensation/factorization and CUDA synchronization assumptions. Independent CPU ports need separate validation.

**[D01]** Han Xiao, Kashif Rasul, Roland Vollgraf / Zalando Research. *Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms*, 2017. [Official dataset repository](https://github.com/zalandoresearch/fashion-mnist); [license](https://github.com/zalandoresearch/fashion-mnist/blob/master/LICENSE); [paper](https://arxiv.org/abs/1708.07747). Dataset description, official filenames/checksums, and MIT notice inspected.

**[D02]** EleutherAI. *Pythia-70M model card*. [Model and documentation](https://huggingface.co/EleutherAI/pythia-70m). Architecture, parameter accounting, research scope, Apache-2.0 notice, and checkpoint conventions inspected. No weights downloaded or inference run in this investigation.

**[D03]** Salesforce. *WikiText dataset card*, wikitext-2-raw-v1. [Official dataset card](https://huggingface.co/datasets/Salesforce/wikitext). Split/configuration and license ambiguity inspected. Article extraction, version pinning, and local access remain WP09 prerequisites.

### 12.5 Coverage and stopping rationale

The search covered the proposal's own terminology and equivalent mechanisms: asymmetric/quantized-prefix calibration; residual and compensation correction; adaptive, coordinate, discrepancy, Ising and cluster rounding; robust/box/structured least squares; graph-Laplacian and relational losses; stochastic activation mixing; vector-output/shared-adversary tree inference; finite straight probes; Euler invariants; potential-difference approximation; closed paths and cycle means; prescribed-zero Chebyshev polynomials; polynomial design without an intercept; and relevant calibration patents.

Backward tracing from GPTAQ/CoreQ exposed AdaRound's earlier asymmetric formulation and classical robust least squares. Forward tracing added ResComp and current YAQA, changing the required baselines. The revised CoreQ title/formulation was checked rather than relying on its older title in the attachment. Additional searches for the shared-vector refinement mainly returned adjacent methods, not a different exact competing construction.

Adjacent current sources examined for scope included *Enhancing Post-Training Quantization via Future Activation Awareness* ([arXiv:2602.02538v1](https://arxiv.org/html/2602.02538v1)), *Gated Relational Alignment via Confidence-based Distillation for Efficient VLMs* ([ICML 2026](https://proceedings.mlr.press/v306/chen26am.html)), and *Finite Probes Suffice: Identifiability and Universality for Weight-Space Learning* ([Microsoft Research publication](https://www.microsoft.com/en-us/research/publication/finite-probes-suffice-identifiability-and-universality-for-weight-space-learning/)). Abstract/method-level inspection showed adjacent downstream-awareness, relational-QAT, or weight-space identification questions rather than the selected forest minimax.

A limited patent screen inspected Huawei's *Method for calibrating neural network quantization, apparatus, device, medium, and program product*, WO2024082932A1, published 25 April 2024 ([text](https://patents.google.com/patent/WO2024082932A1/en)). Its reviewed scale/sensitivity calibration did not identify the proposed mechanism. This is a search-coverage note, not a patentability or freedom-to-operate opinion.

Research stopped after repeated targeted synonym searches and citation tracing ceased to change the mechanism, required baseline families, or local implementation constraints. This is not proof that no equivalent exists. Blind spots include inaccessible full texts/code, other languages, unindexed submissions, monographs, unpublished work, proprietary systems, and a nonexhaustive patent landscape. Repeat the exact-mechanism search if a publishable empirical contribution survives the planned controls.

---

## Appendix A. Implementable references for contemporary baselines

These are independent mathematical references. Porting to a new model/grid is disclosed; no official software parity or published-score reproduction is implied. Use float64 CPU small-instance references first.

### A.1 Unblocked GPTQ, GPTAQ, and ResComp

Use X_f, X_q of shape d × N and immutable W₀ of shape m × d. Define

\[
H=X_qX_q^\top,\quad B=(X_f-X_q)X_q^\top,\quad
LL^\top=(H+\gamma I)^{-1},
\]
\[
P_1=\operatorname{triu}(BL,1)L^\top,\quad
P_2=\operatorname{triu}((H+B)L,1)L^\top.
\tag{A1}
\]

L is the lower Cholesky factor **of the inverse**, not the inverse of the lower Cholesky factor. At column j, snapshot u = U[:,j], set q = R_j(u), where R_j rounds to dequantized legal values, and update

\[
U_{:,j:}\leftarrow U_{:,j:}
-\frac{u-q}{L_{jj}}L_{j:,j}^\top
+u(P_1)_{j,j:}
+(W_{0,:,j}-u)(P_2)_{j,j:}.
\tag{A2}
\]

Products with u are outer products. Start U = W₀. GPTQ uses the first correction, GPTAQ the first two, ResComp all three [Q01–Q03]. Save codes before advancing. Use one snapshot for all terms. Any activation-order permutation must move each coordinate's original grid metadata and be inverted afterward.

For validation, P₁ and P₂ must vanish through column j in row j; the updated current column must equal q. GPTAQ reduces to GPTQ when X_f = X_q. Do not require the efficient ResComp decomposition to have that same limit after earlier compensation. Check each neuron-constrained update against an independently solved small KKT system; a whole-residual least-squares solve is a distinct comparator.

A preparation check used random m = 3, d = 5, N = 11 and γ = 0.03. Maximum deviations from independent constrained normal equations were 1.56 × 10⁻¹⁷, 8.33 × 10⁻¹⁷, and 5.55 × 10⁻¹⁷ respectively. This checks the equations in one small experiment, not a production port. The executing agent must repeat across conditioning/order/grid cases.

### A.2 CoreQ and bounded beam rounding

With the same column orientation, compute [Q04]

\[
D=W_0(X_f-X_q),\quad H=X_qX_q^\top+\gamma I,\quad
S=DX_q^\top H^{-1},\quad s=SX_q,\quad \eta=D-s,
\]
\[
\alpha_c=\frac{\|s\|_F^2}{\|s\|_F^2+\|\eta\|_F^2},\quad U=W_0+\alpha_cS.
\tag{A3}
\]

Set α_c = 0 for D = 0. Use positive damping; sort diag(H) ascending, permute H/U/grids, factor H = LLᵀ, and visit j in reverse:

\[
c_j=U_j+\sum_{k>j}(U_k-q_k)L_{kj}/L_{jj},\qquad
\Delta_j(q)=L_{jj}^{2}(q-c_j)^2.
\tag{A4}
\]

Greedy chooses the nearest legal q. Beam K expands all levels per retained suffix and keeps the K lowest accumulated costs, with deterministic ties. Preserve the Cholesky diagonal.

**Local safeguards and independently derived checks:** an identically zero undamped Gram returns nearest legal rounding and a zero-reachable-curvature flag; trace scaling cannot make zero trace positive. Verify that branch costs sum to \(\|(q-U)L\|^2\) and a full-width tiny beam matches enumeration. Finite K has no global guarantee.

Report α_c, damping, rank, and mismatch norms. Production uses γ > 0. For the zero-damping diagnostic use the Moore–Penrose solution S = D X_q⁺, not an ordinary inverse of a singular Gram. If N ≤ d and X_q has full column rank, s = D and α_c = 1 for nonzero D. This independently derived projection limit motivates the N > d sensitivity check; it is not the geometry of a finite-ridge fit.

### A.3 YAQA scalar-grid MLP specialization

Declare row-major vectorization. At the fully floating teacher, let x_i be layer input, p_i final class probabilities, and J_i the C × m Jacobian of final logits with respect to this layer's preactivation. Define

\[
K_i=J_i^\top(\operatorname{diag}p_i-p_ip_i^\top)J_i.
\]
For the exact-class expectation of Sketch B [Q05],
\[
H_I=\frac1{Nm}\sum_i\operatorname{tr}(K_i)x_ix_i^\top,\qquad
H_O=\frac1{Nd}\sum_i\|x_i\|^2K_i.
\tag{A5}
\]

At the final affine layer J_i = I. Ten classes make exact label expectation cheap; do not substitute observed true labels. Stream the Jacobians, avoiding a dense weight Hessian.

If either curvature factor is identically zero, use deterministic nearest legal rounding and report zero modeled curvature. Otherwise add trace-scaled diagonal damping, initially 10⁻⁴ times each factor's mean diagonal. Construct unpivoted LDL from lower Cholesky C by D = diag(C)² and L = C diag(1/diag(C)); check the reconstruction residual. Set A = L_O − I and B = L_I − I, and use

\[
\widehat W\leftarrow R\!\left(
W_0+A^\top\Delta B+A^\top\Delta+\Delta B
\right),\qquad \Delta=W_0-\widehat W.
\tag{A6}
\]

A direct scalar traversal proceeds by decreasing row-plus-column index. A second implementation starts at R(W₀) and iterates the fixed-point equation until the codes stabilize; test agreement on small well-conditioned cases with declared tie behavior. Never forget the identity subtraction.

For an independent curvature check, form a genuinely tiny explicit teacher-KL Hessian and compare with the sample Fisher \(K_i\otimes(x_ix_i^\top)\). Ordinary mean-input-covariance times mean-output-Fisher is a different KFAC diagnostic, not this sketch. Computing curvature at a quantized prefix is an additional adaptation. Fixed-grid clipping does not automatically satisfy every assumption in the published theory.

### A.4 Gradient-based rounding competitor

Implement both controls below. They are fully specified **local MLP adaptations**, not reproductions of published large-model scores. Use calibration C for gradients, D for development selection, immutable original biases/scales, and the same effective legal levels as every fixed-grid method.

**Common feasible domain.** For each original weight, find its bracketing legal lower/upper values, clipping both to the endpoint when outside the grid. Let y = clip((W₀−lower)/(upper−lower),0,1). Fix coincident endpoints and report original clipping. Interpolate unfixed weights as lower+x(upper−lower). Hard rounding chooses x < 1/2 down and x > 1/2 up; at exactly 1/2 choose the even original integer code. At endpoints, the hard forward must equal the exported/reloaded FP32 model. For local reconstruction, use CPU float64 fractions, inputs and targets. For global KL, form interpolated weights in float64 then cast to FP32 for the ordinary model forward; gradients pass through that cast, and log-softmax/KL is computed in float64. This follows the shared numerical contract and avoids an implicit MPS-double dependency.

**A4a — Asymmetric activation reconstruction.** Use the AdaRound stretched sigmoid h(v) = clip(1.2 sigmoid(v)−0.1,0,1), initialized with v = log((y+0.1)/(1.1−y)), so h(v) = y. Let act be ReLU for a hidden layer and identity for the output. Optimize

\[
L_{\rm rec}=\operatorname{mean}_{i,o}
\left[\operatorname{act}(H^\mathsf QW_{\rm soft}^\top+b)
-\operatorname{act}(Y+b)\right]^2,\quad
R_\beta=\operatorname{mean}_{j\ {\rm unfixed}}\left[1-|2h(v_j)-1|^\beta\right],
\quad L=L_{\rm rec}+\tau R_\beta .
\tag{A7}
\]

The mean is over output elements for reconstruction and unfixed weights for the regularizer. Start with Adam, learning rate 0.001, β₁/β₂ = (0.9,0.999), ε = 10⁻⁸, no weight decay, and batch min(256,N) sampled through seeded shuffled cycles. The current Qualcomm reference supplies useful starting settings: T = 10,000 steps, τ = 0.01, 20% regularizer-free warmup, and cosine exponent annealing 20 → 2 [Q06a]. In this plan, after warmup let u run from 0 to 1 and set β = 2+9(1+cos(πu)); before warmup use τ = 0. The retained ReLU is an explicit choice consistent with asymmetric activation reconstruction [Q06]; the inspected reference code omits that activation.

During development, compare learning rates {0.0003,0.001,0.003} and τ ∈ {0.001,0.01,0.1}, including the reference starting point. Evaluate hard checkpoints at the start, warmup end, and quarter intervals thereafter; charge these evaluations. Refine or expand a boundary optimum when its loss/time curve warrants it. Save planned horizon T, actual retained step, and full annealing history; truncating a schedule and restarting a shorter rescaled schedule are different configurations.

**A4b — Projected full-model KL rounding.** DiscQuant's practical reference uses uniform fraction initialization, AdamW/cosine optimization, and entrywise KL-gradient clipping; its reported coefficient 200 multiplies KL, and its 1,024-step/128-warmup recipe was for LLMs [Q08]. Those constants are not transferred silently to a per-parameter-mean MLP penalty.

For the local control, initialize unfixed x uniformly on [0,1] with a named seed; include a separately labeled x = y initialization. Use

\[
L_{\rm KL}=\operatorname{mean}_{i}{\rm KL}(p_i^\star\|p_i(x)),\quad
R_{\rm lin}=\operatorname{mean}_{j\ {\rm unfixed}}(1-2y_j)x_j .
\tag{A8}
\]

To choose a dimension-aware starting penalty from calibration only, compute the median absolute nonzero gradient of L_KL at the uniform initialization and divide by the median absolute nonzero gradient of R_lin. Call this τ₀ and store both medians; use that same reference τ₀ for the x = y initialization control. If either median is zero at the numerical floor, retain the explicit τ = 0 KL-only control and diagnose with an alternate uniform initialization. A zero gradient at one point does not prove that a weight is unreachable and does not authorize fixing it. Nearest rounding without optimization is permitted only for coordinates proved structurally disconnected or inactive over the entire feasible domain; report that proof/condition. A zero R_lin gradient simply removes that penalty's influence. Try τ/τ₀ ∈ {0,0.1,1,10} when defined; this is a starting search, not a cap.

Use AdamW with weight decay **zero**, β₁/β₂ = (0.9,0.999), ε = 10⁻⁸, initial rates {0.01,0.05,0.1}, batch min(256,N), and T = 1,024 as an initial development horizon. Warm the rate linearly for T/8 steps, then cosine-decay it to zero. At each step compute g_KL, clip it elementwise to [−c,c], add τ∇R_lin, perform the AdamW update, and project x to [0,1]; endpoints may reenter the interior. Start c = 1 and compare {0.5,1,∞} on development. The penalty coefficient multiplies the explicitly normalized R_lin; clipping occurs **before** adding that term. This manual gradient rule, normalization, disabled weight decay, and MLP batches define the named local adaptation. They are not a claim of exact DiscQuant implementation parity.

For both controls, plot hard-model D KL against full elapsed cost, include zero-step/nearest rounding, and extend T geometrically when the development curve is still improving. Compare an extended run and a smaller rate before declaring a plateau; uncertainty is estimated across development teachers, not repeated readings of one model. Freeze the selected initialization, optimizer, rate, batch sequence policy, penalty/schedule, horizon and retained step before confirmation. Record all failed/unstable attempts and complete state for resumption. An implementation that cannot complete a faithful local control is a baseline gap, not a favorable result.

These controls have a floor/ceil domain. Run a matched floor/ceil FPCQ condition and separately allow common all-code quadratic polishing, so a larger feasible set does not masquerade as robust-objective value. Do not attribute DiscQuant's theoretical rounding guarantees to this practical adaptation.

---

## Appendix B. Assets, numerical conventions, and reproducible checks

### B.1 Official Fashion-MNIST files

| File | Official MD5 |
|---|---|
| train-images-idx3-ubyte.gz | 8d4fb7e6c68d591d4c3dfef9ec88bf0d |
| train-labels-idx1-ubyte.gz | 25c81989df183df01b3e8a0aad5dffbe |
| t10k-images-idx3-ubyte.gz | bef4ecab320f06d8554ea6380940ec79 |
| t10k-labels-idx1-ubyte.gz | bb300cfdad3c16e7a12a480ee83cd310 |

Obtain URLs from the pinned official repository [D01], verify the compressed bytes, then record SHA-256. Validate IDX magic values, declared counts, dimensions, byte length, label range, and endianness. No unverified mirror is an automatic substitute.

### B.2 Global small-grid objective check

For tiny m,d,N where both \(2^N\) disturbance signs and \(2^{B_\mathrm{bits}md}\) code assignments can actually be enumerated, compute:

1. Every code assignment's J and U.
2. The globally best J code, globally best U code, and each code's cross-objective value.
3. The proposed local solver's gap to those exact minima.
4. The continuous nominal quadratic solution as a diagnostic, not automatically a certified lower bound on a different robust objective.

If a robust continuous lower bound is desired, choose fixed nonnegative weights summing to one over known sign scenarios. The weighted sum of their losses is below their maximum for every W. A **verified minimum** of that quadratic over all continuous W is therefore a discrete robust lower bound. A merely unfinished continuous optimizer's value is not a certified lower bound. Use direct normal-equation/pseudoinverse checks and residual-based numerical tolerance on these tiny instances.

### B.3 Mandatory numerical conventions

- Calibration/oracle math uses float64 CPU with the hook-input/local-target/deployed-FP32 contract in Section 4.4; teacher weights/scales retain their original FP32 values.
- Loss normalizations are explicit: mean over N and output count m, contrast mean over edges; no double normalization in augmented data.
- Nearest ties use round-to-even for grid construction; optimizer ties keep the incumbent and then deterministic lexicographic order.
- Negative node/edge weights, NaN/Inf, cycles, fractional codes, and invalid shapes are errors.
- A zero-drift first layer must match its ρ = 0 counterpart under the same optimizer.
- External teacher_targets always use immutable teacher weights. Baseline-prescribed internal correction terms may use working U: Appendix A's GPTAQ uses the current snapshot u, and ResComp also uses W₀−u. Do not replace those terms with external targets.
- Group membership follows the physical coordinate through permutations.
- Accumulated numerical cache error is measured and checked; it never becomes an unreported source of accepted gains.
- A packed export must preserve integer codes; backend requantization is a new experiment.
- All reported likelihoods use natural logs and stable log-softmax; minuscule negative computed KL is investigated against the numerical floor rather than arbitrarily clamped to manufacture a benefit.

### B.4 Reproduction when original attachments are available

After extracting the original package, these original commands exist:

~~~bash
python -m pip install -r requirements.txt
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python tests.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python forest_experiment.py --seeds 0 1 2 --bits 2 3 4 --out results/reproduction
python summarize.py
~~~

Run in a copy. Inspect checkpoint/output-path behavior first: the original scripts may expect saved teachers and fixed result locations, and their dependency file is not a lock. Importing the test function and writing its result to a separate audit directory avoids overwriting original verification records. A new output directory without teacher checkpoints can cause teacher retraining; that is different from reevaluating the supplied models. Use the source README for its fixed-path helper scripts, or adapt output paths explicitly.

The new plan remains executable without those attachments: Sections 2–5, Appendix A, and the fixtures specify the mechanism and baselines. In that case report “new implementation from the standalone specification,” not “reproduced the original archive.”

---

**End of handoff.** The next authorized action is WP00. The scientific outcome is deliberately open: a useful scoped improvement, a simpler sufficient method, a negative result, or an inconclusive local test are all valid results if the protocol and evidence support them.
