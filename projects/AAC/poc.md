# Adaptive Algebraic Completion: research assessment and executable local proof-of-concept plan

**Research cut-off:** 13 October 2026.  
**Target machine:** one MacBook Pro with an M3 Max; actual memory, storage, operating system, and available thermal/power envelope must be discovered.  
**Deliverable type:** research and implementation handoff, not a report of a trained AAC model.  
**Primary direction:** diagonal-preserving, input-adaptive source-block sparsity with output-wise error envelopes and measured execution costs, called **AAC-B** below solely as an experiment identifier.

**Navigation:** [Decision](#0-executive-decision) · [Prior art](#2-prior-art-and-novelty-assessment) · [Mechanism](#3-selected-mechanism-and-mathematical-specification) · [Evaluation](#6-data-workloads-metrics-and-confirmatory-design) · [Hardware](#7-m3-max-stack-memory-and-kernel-feasibility) · [Work packages](#8-executable-work-packages-and-decision-gates) · [References](#133-bibliography-and-inspected-artifacts) · [Executed checks](#appendix-a-checks-actually-executed-during-this-research)

## 0. Executive decision

Proceed with a gated local investigation, **not with a claim that a new multiplication algebra or a superior neural architecture has been discovered**. The central idea is useful enough to test: preserve inexpensive local interactions, selectively restore cross-group interactions, and adapt the model so that the omitted remainder is small. Its individual ingredients substantially overlap existing activation sparsity, sparsity-aware training, budget allocation, and structured computation. The potential contribution is a demonstrably better combination, not its name.

The selected experiment keeps the proposal's diagonal-plus-selected-source operator. It replaces its loose separable routing envelope with a tighter output-wise envelope, explicitly separates hard-budget and error-tolerance modes, and evaluates both the routing rule and a compressibility regularizer against matched controls. The initial insertion is the FFN down projection. Extension to the other FFN projections is conditional on measured quality and hardware headroom. Attention, normalization, tokenizer, and KV-cache semantics remain unchanged.

Three corrections materially affect the project:

1. The September 2026 algebra paper is genuine, but its trained four-group model uses the tensor-square rule with **9/16**, not the proposal's one-source **7/16**, row-level block arithmetic. Its reported throughput is not an M3 result or a controlled result for AAC. Reproduce both operator definitions locally, not the paper's large pretraining experiment. [R01]
2. Training representations to become sparse is already a central contribution of earlier work, particularly Q-Sparse and ProSparse. Weight-aware routing and nonuniform allocation are also established. The discriminating experiment must compare AAC-B with well-tuned weight-aware sparsity, CATS, and sparsity-aware adaptation, not merely with a static mask. [R02][R06][R07][R09][R10][R42]
3. Dense recovery is exact only for the **current student's linear layer at the same input**, in exact arithmetic. It does not restore the frozen teacher or repair previous sparse computations and cached states. A decreasing error envelope does not imply decreasing actual error or increasing language-model quality.

The minimum meaningful positive result is a reproducible local quality–latency advantage over the strongest eligible local comparator, with its cause isolated. There is no mandatory 10% speedup and no presumption that the result is positive. A reliable small effect can be scientifically useful without being commercially consequential. A statistically unresolved effect is inconclusive, not a win.

**Core local scale:** SmolLM2-135M for correctness and low-cost pilots; SmolLM2-360M for the primary realistic local comparison. These are affordable test assets, not claims about the best language models in 2026. The 1.7B member is a conditional scale check, not a prerequisite. Their official configurations support the necessary Llama-style gated FFN interface. [R34][R35][R36]

### What has already been done for this plan

Primary papers, relevant supplementary sections, selected public implementation files, MLX documentation, model configurations, and dataset documentation were inspected. Small algebraic and numerical checks were executed on a **Linux x86-64 CPU**, not on Apple Silicon. They passed; their code and results are included in Appendix A. No model was downloaded, adapted, or timed on an M3 Max. No MLX/Metal compatibility test was executed here. All quality, memory-peak, training-time, and latency outcomes below remain proposed measurements.

### What would justify continuing

Continue when the real model has an input-dependent approximation advantage, that advantage survives matched adaptation, and a realistic kernel has enough measured headroom to affect end-to-end execution. Stop or narrow the claim when any of those conditions fails with adequate sensitivity. Distinguish implementation failure, inadequate measurement, a resource limit, a bad small-scale proxy, a failed mechanism, and a mechanism that works but is already matched by simpler methods.

---

## 1. Problem statement, scope, and hypotheses

### 1.1 Intended use and bottleneck

The intended beneficiary is a developer running a small decoder language model locally, especially batch-one generation under memory-bandwidth and power constraints. The task is to reduce actual inference cost at controlled model quality without changing the model's external interface. The existing workflow uses optimized dense or quantized FFN projections, dense attention, and a KV cache.

The proposed bottleneck is repeated reading and multiplication of FFN weight blocks that contribute little for a particular input. Whether this dominates a particular M3 Max workload is **an empirical prerequisite**, not an established fact about this laptop. Attention, vocabulary projection, synchronization, sampling, or memory pressure may dominate instead. Prefill and decode are separate workload regimes; an intervention can help one and hurt the other.

The initial scientific population is held-out English text under fixed causal context rules, with code, mathematical-solution text, narrative completion, rare-token, and long-context guardrails. The local PoC cannot establish production assistant utility, broad reasoning competence, frontier-scale behavior, or a universal matrix-multiplication improvement.

### 1.2 Inputs, outputs, and accounting conventions

A projection takes a row vector or batch of row vectors, `x ∈ R^d` or `X ∈ R^(N×d)`, and a logical weight matrix `W ∈ R^(d×m)`. Model libraries may store its transpose; all equations in this document use row-vector convention. The output is a vector in `R^m`. Bias, when present, is added once and is never sparsified.

Partition the input into `q` groups of sizes `d_i` and the output into `q` groups of sizes `m_j`. Denote the resulting block by `W_ij ∈ R^(d_i×m_j)`. Equal groups are the first experiment; unequal groups and internal padding must use their true dimensions for cost accounting.

A **MAC** is one scalar multiply–accumulate. Report MACs separately from FLOPs; when giving the conventional two-FLOP-per-MAC estimate, label it. Report four different quantities: logical active MACs, padded/executed arithmetic where known, estimated weight bytes read, and measured wall-clock time. None is an interchangeable proxy for another.

A **source mask** is a `q`-bit integer. Bit `i` selects input group `i` for cross-group communication. It does not select a token, attention head, expert network, or output neuron. A **budget** is either a hard arithmetic/action constraint or an approximation tolerance; these are different contracts.

### 1.3 Evidence, assumptions, and falsifiable hypotheses

| Hypothesis | Existing support or reason to test | Strong alternative explanation | Observation that weakens it | Decisive local test |
|---|---|---|---|---|
| H1: Input-dependent source choice preserves more useful signal than a fixed schedule | Activation-sparsity methods establish that input-dependent importance can be useful; this does not establish coarse source-group concentration. [R02][R04] | A few globally important groups explain the benefit | Best fixed masks match adaptive masks, or an exact-error oracle has no advantage | Frozen-weight, exhaustive-mask reconstruction study; fixed versus adaptive and shuffled routing at identical block counts |
| H2: The diagonal-preserving family is a useful inductive bias | It preserves a local path for every group and includes a dense limit; these are algebraic properties, not quality evidence | Ordinary group dropping or a smaller dense FFN is sufficient | Those controls dominate after matched tuning and adaptation | Compare actual MAC/latency frontiers, including the diagonal-free family and physically narrowed FFNs |
| H3: The bound regularizer teaches useful compressibility beyond ordinary distillation | Plausible mechanism; closely related sparsity-inducing training already exists. [R09][R10][R24] | Gains come from extra training, teacher information, or suppressing activation scale | μ=0 or a generic sparsity penalty gives the same frontier; dense-student quality worsens | Paired adaptation with identical data, trainable parameters, teacher access, budget schedule, and optimizer |
| H4: The tighter envelope is a better deployable controller | The inequalities in §3 establish tighter exact-arithmetic bounds | Conservatism, routing cost, or norm-estimation cost erases the advantage | Similar actual errors but slower routing; tolerance mode almost always dense | U versus V versus B at equal selected actions and then equal total latency |
| H5: Skipped work leads to useful M3 Max end-to-end savings | MLX exposes masked/batched multiplication and custom Metal kernels; suitability is shape-specific. [R27][R28][R29] | Dense GEMM/GEMV or quantization is faster despite more arithmetic | Sparse kernels lose even with routing removed, or full-model overhead erases gains | Static optimistic kernel bound, then router-inclusive microbenchmark and full generation measurements |

**Established capabilities:** ordinary block multiplication, source masking, elementary norm inequalities, dense fallback at a single layer, and existing sparse/adaptive neural methods.

**Necessary assumptions:** learned activations and weights have exploitable block structure; training can improve that structure without unacceptable capacity loss; the selected source granularity maps to efficient physical loads; skipped weights are actually not read; and the target workload spends sufficient time in eligible projections.

**Speculative, nonessential extensions:** learned channel grouping, variable `q`, sequence-level context controllers, quantized AAC, global error propagation, richer algebra products, and full-model continued pretraining. Do not build these before diagnosing the primary mechanism.

### 1.4 Strongest positive and negative cases

The strongest positive case is a small decoder whose FFN work dominates batch-one generation, with a heavy-tailed distribution of useful source contributions. Diagonal computation preserves distributed local information while selected sources provide occasional long-range feature mixing. Distillation could organize hidden features into a structure that works well at several budgets. A small mask space could allow regular kernels without a learned router.

The strongest negative case is a well-mixed representation with large, similarly important groups. Group norms concentrate, the output-wise upper bound remains loose, and useful omitted contributions cancel. Recovering quality then requires nearly dense work. Meanwhile, optimized dense or quantized kernels use larger tiles and less control overhead, and a narrower FFN removes weight storage as well as arithmetic. Neither associativity nor access to full-bank weights protects AAC from this outcome.

### 1.5 Headroom calculations to test, not assume

For equal independent block contributions, a deliberately simplified isotropic model gives

`E[||error||²] / E[||dense output||²] ≈ ((q-k)(q-1))/q²`.

For `q=4`, the corresponding RMS ratios are approximately 0.750, 0.612, and 0.433 for `k=1,2,3`. This deduction explains why coarse sparsification of an unadapted, distributed representation may be destructive. It is not a prediction for a trained model. Test both isotropic and planted-compressible synthetic distributions to verify the diagnostic's sensitivity.

If eligible projections occupy fraction `f` of baseline runtime, their complete replacement costs ratio `r` including routing, and other added overhead is `η` of baseline runtime, an Amdahl-style estimate is

`speedup ≈ 1 / (1 - f + f*r + η)`.

Illustratively, substituting the arithmetic fraction 0.625 for `r` gives 1.29× when `f=0.6`, but only 1.081× when `f=0.2`. These are optimistic calculations, **not latency predictions**. Measure `f`, `r`, and overhead. A down-only experiment can establish a mechanism while having intrinsically modest full-model headroom.

---

## 2. Prior art and novelty assessment

### 2.1 Closest-work comparison

“Available” below describes inspected evidence, not a guarantee that an implementation runs locally. Published accelerator numbers are reference evidence, never controlled M3 comparisons.

| Work | Mechanism and important assumptions | Evidence / implementation inspected | Overlap and consequence for AAC |
|---|---|---|---|
| Associative algebra layers, 2026 [R01] | Fixed row-typed alternative product; both direct and tensor-square constructions | Full manuscript, appendices, PDF tables; approximately 110M models, 12.3B tokens each, one training run per model; NVIDIA execution | Direct origin of one-source row support. Trained rule differs from direct four-group rule. Local test is an operator adaptation, not pretraining replication |
| WINA, 2025 / ICLR 2026 [R02][R03] | Activation magnitude weighted by weight-row L2 norm; stronger theoretical optimality has additional assumptions | Paper and calibration/allocation source files; upstream workflow uses CUDA | Norm routing is not new. Port the scoring rule and disclose any omitted transforms or source-specific calibration behavior |
| TEAL, 2024 / ICLR 2025 [R04][R05] | Magnitude thresholding with nonuniform projection allocation | Paper metadata and official repository/README; NVIDIA-oriented sparse execution | Strong training-free control, including calibrated thresholds rather than only an intentionally weak uniform top-k rule |
| WiSparse, February 2026 [R06] | Learnable/calibrated weight-norm exponent and mixed allocation | Full method and appendix; source implementation link not located in inspected manuscript | Direct competitor to weight-aware routing and cost allocation; α=0 and α=1 connect important simpler controls |
| WAS, EMNLP 2025 [R07][R08] | Weight-row L1 importance, error-envelope reasoning, constrained Bayesian scheduling | Paper including equation/algorithm screenshots and repository | Both norm error control and cross-layer allocation precede AAC; use calibration-only scheduling locally |
| Q-Sparse, 2024 [R09] | Top-k activation sparsity, straight-through gradients, sparsity-aware training | Full paper, including continued-training and block-sparsity discussion | “Train the representation to make multiplication skippable” is not distinct by itself. Its block form is intra-block N:M sparsity, not AAC whole-source selection |
| CATS, 2024 [R42] | Threshold post-SiLU gate values; skip corresponding up-projection columns and down-projection rows | Full method and linked official implementation location | Particularly important practical comparator: it can remove work in two FFN projections while preserving SiLU outside the threshold |
| ProSparse, 2024 / COLING 2025 [R10] | Activation replacement and progressive sparsity regularization | Full manuscript | Generic sparsity regularization is established; activation changes and recovery expenditure must be disclosed |
| Sparsing Law / CETT context [R11] | Contribution-tail criteria and scale/training dependence | Inspected later paper's treatment; original CETT-source full text not successfully inspected | Tail-control intuition is not new. Do not equate a calibrated relative criterion with a cheap unconditional certificate |
| MoEfication, generalized MoEfication, DOT-MoE [R12][R13][R14] | Convert/group dense FFN features into selectively evaluated experts; newer work learns grouping | Primary papers/pages inspected | Learning channel groups is a prior-art-heavy pivot, not an automatic rescue of novelty |
| Universally slimmable networks [R15] | Train several computational widths and distill across them | Official repository and paper description | Multi-budget training and sandwich-style dense paths are reusable training techniques, not an AAC invention |
| Monarch [R16] | Products of structured matrix factors | Primary conference paper page | Strong structured alternative, but different storage/capacity and no automatic same-weight dense limit |
| CLADA, February 2025 [R17][R18] | Context-sensitive activation allocation | Primary manuscript and repository | Context-conditioned allocation already exists; its causal information requirements need care in local ports |
| BLASST, December 2025 [R19] | Dynamic blocked attention sparsity | Primary manuscript | Primarily attention-side, not an FFN routing comparator; potentially complementary |
| Spark, ActTail, Learning to Allocate [R20][R21][R22] | Activation sparsity, global allocation, and resource-adaptive execution on other axes | Primary manuscripts | Broad claims to novel dynamic resource allocation are untenable without a much narrower contribution |
| Approximate block matrix multiplication; sparsity-inducing inference [R23][R24] | Block approximation and training to create exploitable sparsity | Primary publication pages | Neither norm-controlled omitted work nor hardware-motivated sparsity training is a new general principle |

The supplied proposal dated WINA, CLADA, and BLASST as 2026 work. Publication records place their original submissions in 2025; conference/version dates must not be confused with first appearance. This correction does not weaken their relevance. [R02][R17][R19]

### 2.2 What the algebra pilot does and does not establish

The authors' reported generation improvement is 6.2–7.8%, with lower quality on the three reported downstream metrics. Training used two H100s per model. The inspected generation table also has different mean generated lengths across models. Consequently, the reported rate is useful feasibility evidence but not a fixed-output-length estimate transferable to another device. The local harness will remove that workload confound. Public kernel source was not located through the inspected manuscript; absence from that inspection is not proof that no code exists. [R01]

### 2.3 Precise novelty claim that remains worth testing

**Candidate methodological contribution:** an input-adaptive diagonal-plus-source operator whose small complete mask space permits output-wise error-envelope routing under executable cost constraints, combined with a controlled training objective that improves its quality–latency frontier beyond existing weight-aware sparsity and generic sparsity-aware adaptation.

This is **plausible differentiation, not demonstrated novelty**. The tighter inequalities below are elementary deductions. Their implementation is not a new algebraic-complexity theorem. Hardware-specific fusion, if successful, may be a valuable engineering result even when the mathematical method is not new.

The contribution is invalidated or substantially narrowed by any prior method with the same effective support family, controller, and training mechanism, or by matched baselines that explain the full gain. Preserve such findings rather than renaming the method. A newly discovered equivalent changes the project to replication or engineering evaluation unless a genuinely different discriminating hypothesis remains.

### 2.4 Reuse, rather than rebuild

Reuse the official MLX-LM model loader and Llama-style model implementation, native MLX dense/quantized paths, public model weights, and standard dataset/tokenizer infrastructure. Reuse baseline code only after inspecting its applicable license and documenting adaptations; publicly visible is not synonymous with permission to redistribute. Upstream CUDA/Triton kernels are methodological references, not Apple-compatible binaries. [R03][R05][R08][R25][R32][R33]

Implement one shared measurement harness, one common activation-gathering path, and shared eligible Metal sparse primitives where possible. Do not give AAC a fused kernel while timing baseline methods through Python loops. Conversely, retain stock optimized dense execution rather than forcing every method through a slow general-purpose abstraction.

---

## 3. Selected mechanism and mathematical specification

### 3.1 Forward operator and exact cost

For a mask `S ⊆ {0,…,q−1}`,

\[
\hat y_j=x_j W_{jj}+\sum_{i\in S,\,i\ne j}x_iW_{ij}.
\]

Define `M_ij(S)=1` when `i=j` or `i∈S`, and zero otherwise. Then `ŷ=x P_S(W)`, where `P_S` zeros the omitted blocks. This representation is for mathematical definition and correctness tests; **forming a dense masked weight matrix and calling dense GEMM does not save the claimed multiplication or memory traffic**.

The logical MAC count is

\[
C(S)=\sum_j d_jm_j+\sum_{i\in S}d_i\sum_{j\ne i}m_j.
\]

For equal group dimensions, the relative count is `[q+|S|(q−1)]/q²`. Bias adds the same output work to all masks. At `q=4`, nonempty masks have 7, 10, 13, or 16 block products. Empty `S` has four diagonal products and is a diagnostic boundary case; the primary family retains the original minimum of one selected source.

All-source selection recovers the current dense product. If padded internal groups are used, zero padding must not change group membership or the logical output; separately record the padded computation. Use exact original dimensions in quality comparisons.

### 3.2 Associativity: what can honestly be said

The two-by-two product in the proposal is associative; the executed integer checks in Appendix A exercise noncommuting blocks. That does not make arbitrary input-dependent routing associative.

A separate elementary observation applies to AAC's **fixed support**, for square matrices with compatible group partitions. Let

`A_S = {A : A_ij=0 for i≠j and i∉S}`.

This is a subalgebra under ordinary matrix multiplication for **any fixed S**, not just one selected source. For an unselected row `i`, only `A_ii` can be nonzero, and `(AB)_ij=A_ii B_ij=0` for `j≠i`. Thus multiplication is closed; ordinary associativity then applies. Reordering selected groups first gives the form `[[A,B],[0,D]]`, with block-diagonal `D`.

This is not a new multiplication law on the full parameter bank. In general, `P_S(AB) ≠ P_S(A)P_S(B)`. It must not be conflated with the original paper's parameterized graph-algebra product, and a rectangular neural projection alone does not have an associativity operation to prove. With `S=S(x)`, the overall map is generally nonlinear and no associative composition guarantee is claimed.

### 3.3 Three exact-arithmetic envelopes

For each block choose `σ_ij ≥ ||W_ij||₂`, where the norm is spectral. Define

\[
a_{ij}=\begin{cases}\|x_i\|_2\sigma_{ij},&i\ne j,\\0,&i=j.\end{cases}
\]

For the omitted source set `O=Sᶜ`, define

\[
B(S)=\left[\sum_j\left(\sum_{i\in O}a_{ij}\right)^2\right]^{1/2},
\quad V(S)=\sum_{i\in O}\left[\sum_j a_{ij}^2\right]^{1/2},
\quad U(S)=\sum_{i\in O,j}a_{ij}.
\]

Then

\[
\boxed{\|xW-\hat y\|_2\le B(S)\le V(S)\le U(S).}
\]

**Proof.** The omitted output in group `j` is `e_j=Σ_(i∈O,i≠j) x_iW_ij`. The triangle inequality and spectral norm inequality give `||e_j||₂≤Σ_(i∈O)a_ij`. Concatenated output groups have squared Euclidean norm equal to the sum of group squared norms, yielding the first inequality. Apply the triangle inequality to the nonnegative row vectors `(a_ij)_j` for the second, then `||v||₂≤||v||₁` for each row for the third.

The proposal's top-k score exactly minimizes U for a fixed k. The score `||x_i||₂ sqrt(Σ_(j≠i)σ_ij²)` similarly minimizes V. Neither statement claims to minimize actual error. For B, enumerate all masks at `q=4`; only 16 possibilities exist, or 15 after excluding the empty mask. These are reference optimizers for their respective envelopes, not novel combinatorial algorithms.

### 3.4 Mask selection, tolerance, and strict budgets

Use three explicit controller modes:

- **Fixed k:** enumerate the allowed masks with exactly k selected sources; select the one with minimum B. Resolve ties by estimated executable cost and then integer mask ID.
- **Tolerance mode:** among masks satisfying `B≤ε`, select the lowest estimated executable cost, then lower B, then mask ID. Full selection always has zero exact-arithmetic omission error. No subdense hard budget is guaranteed.
- **Hard MAC/action budget:** among masks whose actual scheduled cost meets the cap, select the lowest B, then estimated latency, then mask ID. Return whether `B≤ε` when a tolerance is also supplied. If no allowed mask meets the cap, return an explicit infeasible status; do not silently run dense.

A latency lookup is a prediction, not a hard real-time guarantee. Hardware load can vary. Distinguish logical MAC caps from padded execution caps. If `q=4` uses unequal groups, equal k need not mean equal cost.

It is generally impossible to guarantee both an arbitrary error tolerance and a strictly subdense budget on all inputs. The API must expose this conflict. Do not call a fallback that violates a hard budget a successful budget-constrained result.

**GPU reference pseudocode** — tensor operations, not a per-token Python loop:

```text
prepare(weight_revision, W, partition):
    sigma = valid_upper_bounds_of_off_diagonal_blocks(W)
    masks = integers 1 .. (2^q - 1)             # primary nonempty family
    keep_source[m, i] = bit_i(m)
    logical_cost[m] = sum of retained block MACs
    return immutable prepared state and its hashes

route(X, state, budget):
    r[n, i] = sqrt(sum(X[n, input_group_i]^2 in fp32))
    a[n, i, j] = r[n, i] * sigma[i, j]; a[:, i, i] = 0
    t[n, m, j] = sum_i((1 - keep_source[m, i]) * a[n, i, j])
    B[n, m] = sqrt(sum_j(t[n, m, j]^2))
    allowed = masks permitted by mode, k, and executed-cost cap
    chosen = deterministic lexicographic argmin using the rules above
    return chosen, selected_B, costs, tolerance_met, status

apply(X, W, chosen):
    # Reference implementation may calculate all terms; not a speed benchmark.
    # Deployment implementation must skip unselected weight loads and products.
    Y[n, output_group_j] = sum_i retained(n, i, j) *
                               (X[n, input_group_i] @ W[i, j])
    return Y + bias
```

Keep routing tensors on the device. Logging per-token masks on the host inside the timed path is prohibited. The small enumeration has `O(N·2^q·q²)` scalar work; it is not free, and q=8 is not automatically an improvement.

### 3.5 Valid norm bounds and cache invalidation

Use Frobenius block norms first: `||W||₂≤||W||F`. Another valid exact-arithmetic upper bound is `sqrt(||W||₁ ||W||∞)`. Their minimum is valid. Plain power iteration and a numerically obtained leading singular value are **not certified upper bounds**; do not place them in a field labeled as such.

If conservative bounds hide a promising actual-error oracle gap, optionally compute a tighter frozen-export upper bound. Let `G` be the smaller of `WᵀW` and `WWᵀ`. For integer p≥1,

`||W||₂ ≤ ||G^p||F^(1/(2p))`.

This follows because G is positive semidefinite and its maximum eigenvalue is bounded by its Frobenius-power norm. Try p=1,2,4, intersect with previous bounds, and use scaled repeated squaring to avoid overflow. This is an offline, potentially expensive calculation; measure export time and storage. It is not part of each inference call. Use SVD as a tightness diagnostic unless a conservative numerical enclosure is separately implemented.

During training, the regularizer's Frobenius norms must be differentiable functions of the **current** weights. A detached, stale norm table would remove the intended weight gradient and may invalidate the envelope. Stop gradients through mask selection, not through the selected envelope values. Refresh frozen inference tables after the last weight update.

A bound artifact must include the weight checksum, partition, precision/cast, normalization procedure, and bound algorithm. Any update, adapter merge, quantization, channel permutation, or precision conversion invalidates it. A lower-precision weight bank requires bounds for that actual bank.

For training, avoid undefined norm derivatives at zero. A permitted numerically stable upper surrogate is `sqrt(max(sum(v*v), epsilon_norm**2))`, using an explicitly logged `epsilon_norm` (initial fp32 value `1e−12`). Apply the same documented convention to block and group norms; use the exact zero envelope for a full mask. The floor only increases a nonzero-mask exact-arithmetic envelope, but changes its tightness and therefore must be logged. Do not use `sqrt(s+eps)-sqrt(eps)` as a claimed upper norm: it can underestimate. Inference may use exact zero handling with a tested safe norm implementation. Include zero-gradient and full-mask tests.

### 3.6 Floating-point and global-accuracy limits

The inequalities above concern exact arithmetic with exact upper bounds. A deployment that computes them in floating point is initially labeled `exact_arithmetic_envelope` with `numerically_audited=true/false`, not “formally certified runtime error.” Validate against fp64 CPU references over adversarial scales. Strict certification would additionally require conservative arithmetic for the bound and a rounding-error term for dense and sparse accumulation, with verified kernel reduction behavior. It is an optional separate project, not a prerequisite disguised as already solved.

NaN or infinity in activations, weights, or bounds is an invalid input/state diagnostic. Do not silently replace it by a low score. Tolerance mode may attempt a dense diagnostic, but must retain the invalid flag. Hard-budget mode must never invent a satisfied certificate.

The bound is local to a linear projection. Applying it independently to both branches of a gated FFN does not automatically bound their nonlinear product, downstream logits, or task error. Report empirical FFN and network errors separately.

### 3.7 Progressive recovery is not monotone actual accuracy

For q=3 scalar groups, take `x=(1,1,1)`, `W_10=1`, `W_20=−1`, and all other entries zero. The actual errors for masks `{0}`, `{0,1}`, and `{0,1,2}` are `0,1,0`: restoring a contribution can break cancellation. The envelope is nonincreasing along nested masks, but actual error need not be.

The initial implementation selects its mask **once before multiplication**. It does not perform several increasingly expensive products and count only the final one. A genuinely incremental variant must cache earlier contributions, account for every launch/read/addition, and distinguish nested masks from independently optimized nonnested masks. It is not required for the core PoC.

---

## 4. Training and model integration

### 4.1 Exact insertion point

For a gated FFN,

`g = SiLU(h W_gate)`, `u = h W_up`, `z = g ⊙ u`, `o = z W_down`.

Start by replacing only `z W_down` with AAC-B. Group the intermediate channels of z and output residual channels independently into q contiguous groups. The residual stream is not permuted. Gate, up, and down FFN parameters may all adapt so the hidden representation can reorganize; freeze attention, embeddings, output head, and normalization in the initial training scope. The official MLX-LM Llama implementation exposes these three projections and retains its existing cache/attention path. [R33]

Do not use down-only weight training as the sole test of representation reorganization; it cannot alter the upstream gate/up representation. It is a useful ablation. Do not compare full-FFN adaptation for AAC with tiny-adapter adaptation for all baselines without separately matching capacity and expenditure.

Before adaptation, all-source mode must agree numerically with the untouched model, including logits and cached generation. Maintain a separate untouched teacher. “Dense-student” means the current student with all its masks full, not the teacher.

### 4.2 Loss and information access

Use the following explicit local recipe, which is a proposed experiment rather than a claimed reproduction of a paper:

\[
L=L_{CE}+\lambda L_{KL}+\beta L_{FFN}+\mu L_{bound}.
\]

`L_CE` is next-token cross entropy. `L_KL` is teacher-to-student token-logit KL, temperature 1 initially, computed over the full vocabulary in chunks when necessary. If a top-logit approximation becomes necessary, label it as a changed loss and give it to all relevant methods; it cannot silently replace exact KL.

`L_FFN` is the mean squared difference between student and frozen-teacher FFN outputs at matched token positions, divided by a frozen per-layer calibration scale squared. Teacher and student may have different internal trajectories; this is a distillation target, not the local linear-bound comparison. Also log current-student dense versus sparse output at the **same student input** to measure the local mechanism.

Let `c_l = sqrt(E_calib[||teacher_FFN_output_l||₂²])`, with a documented small positive floor only for degenerate layers. For AAC-inserted projection outputs use the corresponding teacher projection-output scale. Define

`L_bound = mean_(tokens, eligible layers) B_l(S_l) / c_l`.

The scale is frozen before tuning μ. Do not let the student reduce its denominator to manipulate the penalty. The bound regularizer is a hypothesis about representational organization, not a task-quality guarantee.

At fixed k, the hard computation count is constant and a term `ν C` does not supply a useful selection gradient. Therefore prescribe budgets/curricula outside the differentiable loss. A later learned-budget controller would need an explicit gradient estimator or discrete optimizer and its own baseline; do not imply the original hard cost penalty learns routing automatically.

### 4.3 Concrete pilot training recipe

Use native MLX autograd and ordinary tensor operations for training. A masked dense reference may evaluate all terms and apply hard masks; report actual training cost and make **no sparse-training-speed claim** from it. Custom Metal kernels are initially inference-only.

Initialize teacher and student from the same pinned base checkpoint. Train all FFN weights with AdamW; begin with fp32 student parameters and optimizer states, and fp16 frozen teacher inference if the numerical smoke test passes. The exact optimizer epsilon, betas, weight decay, learning rate, clipping, and token batch become resolved configuration fields, not undocumented defaults. Starting values for a pilot are betas `(0.9,0.999)`, epsilon `1e−8`, weight decay `0.01`, and global gradient clipping at 1; they are not asserted optimal.

Bracket learning rate around `3e−5`, testing factors of three on development-only pilots and shrinking on nonfinite losses or unstable update norms. Give the same bracket and calibration information to matched adaptation baselines. Choose λ and β by short gradient-scale diagnostics followed by development NLL, retaining a no-distillation control. Test μ=0 and positive values corresponding approximately to 0.01, 0.1, and 1 times the task-gradient norm at the initial calibration point. This is a bounded initial search, not a rule to stop searching while a method is demonstrably undertuned. Expand a boundary optimum once, record why, and share comparable opportunities with competitors.

Start with full multiplication, then use

`k_min(t) = ceil(q - (q-k_target) * min(t/T_ramp,1))`.

The initial ramp occupies the first half of the selected pilot token budget. On one quarter of update batches, use the current student's dense path; on the others, use the scheduled sparse budget. The dense fraction and ramp are tunable training design choices, not scientific success thresholds. Use the same data order, dense-path access, teacher evaluations, and budget-sampling opportunities in the matched controls. Include a fixed-budget, no-ramp ablation if the schedule appears important.

Determine training token budgets from learning curves and local throughput, not a claim that an arbitrary short run is sufficient. Begin with one complete resumable pilot chunk that fits available storage and measures stable tokens/s. Double cumulative pilot tokens while development improvement remains larger than its measured noise and the experiment addresses an unresolved hypothesis. Select a **common fixed confirmatory adaptation budget** before protected evaluation. Do not extend a run because its test result is nearly favorable.

### 4.4 Gradient and scale controls

For AAC, hard routing is stop-gradient. Compare ordinary masked gradients against an explicitly labeled straight-through variant; neither can be assumed superior. For Q-Sparse-style controls, retain the original SiLU in pretrained models and use the appropriate post-activation mask. The paper's block scheme selects N entries within M consecutive channels; it is not whole-source-group activation. Its rescaling description does not fully specify the local normalization formula, so the handoff must label the implemented component variant rather than invent exact fidelity. [R09]

Test two fully specified variants when this ambiguity matters: no rescaling; and norm-preserving rescaling `z_masked * ||z||₂/max(||z_masked||₂,δ)`, with an all-zero special case returning zero. Match this opportunity across relevant controls. Neither variant may be called an exact reproduction solely on this basis.

Implement the unscaled input STE as `x + stop_gradient(mask*x - x)`: forward values are masked, while the surrogate input derivative is identity. The linear weight gradient still uses the masked forward activation. A rescaled variant substitutes its explicitly defined sparse value inside the stop-gradient difference. Within an N:M control, form consecutive groups of M=32 initially and select exactly N elements by magnitude with stable index ties; retain incomplete tails densely and account for them. Test N=8,16,24,32 as initial discrete candidates. AAC's primary gradient remains the ordinary hard-mask gradient; STE is a separately labeled ablation.

Measure scale gaming. An internal positive rescaling of an up-projection channel and inverse rescaling of the corresponding down-projection row preserves that FFN's function while potentially changing coarse group envelopes. Do not apply this identity to the nonlinear gate indiscriminately. Log activation norms, block norms, bound/output ratios, dense-student quality, and teacher error. Include a function-preserving rescaling stress test; a changing bound does not necessarily mean changing useful computation.

### 4.5 Conditional grouping extension

Only pursue regrouping if the exact-error oracle shows useful sparsity but contiguous grouping obscures it, or if a stable adaptive advantage is blocked by grouping. A safe hidden-channel permutation simultaneously permutes gate/up columns and down rows. Verify dense-logit equivalence. Arbitrarily permuting residual channels through attention and rotary embeddings is not an allowed shortcut.

Use calibration-only grouping, compare random and simple clustering permutations, and offer the same grouping to CATS and neuron/group-sparsity baselines. Reassess against MoEfication and DOT-MoE before claiming differentiation. Grouping is an explicit extension, not the original contribution quietly changed. [R12][R13][R14]

---

## 5. Baselines and comparison contracts

### 5.1 What counts as a fair local claim

All primary head-to-head results use the same model checkpoint family, tokenizer, evaluation inputs, numerical precision, cache policy, output length, and machine. Distinguish three tracks:

**Training-free track:** unchanged checkpoint, calibration allowed, no weight updates. Compare frozen AAC routing against frozen sparse alternatives and untouched dense inference.

**Matched-adaptation track:** compare methods with the same training population, teacher information, trainable parameter scope where structurally possible, and adaptation-token budget. Report actual teacher/student forward passes, optimizer steps, elapsed time, and energy when measured. Smaller structured models have different parameter counts; disclose that rather than pretending equality.

**Deployment track:** allow practical differences such as quantized weights and narrower networks, then compare the resulting quality–latency–memory frontier. Same precision is necessary for mechanism attribution, but excluding quantization would make a deployment-efficiency claim weak.

A method eliminated during development remains in the report with its reason, uncertainty, tuning record, and best measured configuration. Do not prune a competitor merely because the first implementation is slow or the first hyperparameter is poor. Never compare a tuned AAC configuration with an untuned baseline and call that an architectural gain.

### 5.2 Required local comparison set

| ID | Exact local definition / scope | Why required | Reproduction and fairness check |
|---|---|---|---|
| D0 | Unmodified optimized MLX-LM dense fp16 model; retain a separate fp32 reference for numerical diagnostics | Real performance and quality reference | Same checkpoint/tokenizer; test cached versus noncached logits and full-mask equivalence |
| D1 | Dense student adapted with the same data and distillation opportunity | Detect benefit from extra training rather than sparsity | Same token/teacher budget; report both D0 and D1 |
| DQ | Stock MLX-LM 8-bit and 4-bit supported quantized dense paths | Strong practical local alternative | Pin quantization settings and conversion provenance; assess quality separately, not as an equal-precision causal comparison [R32] |
| NARROW | Physically remove intermediate FFN channels, consistently slicing gate/up columns and down rows | Strong simple structured alternative with storage savings | Rank channels by calibration mean `\|z_r\|·\|\|W_down[r,:]\|\|₂`; tune aligned retained widths; adapt with the same teacher/data opportunity |
| STATIC | Best calibration-selected fixed source mask per eligible layer at each k; plus balanced position schedule | Isolate input dependence | Same AAC operator, blocks, weights, training recipe, and kernels; no routing cost hidden in the fixed method |
| ALG | Direct q=4 star row action and separate tensor-square q=4 action | Closest algebraic mechanism | Use support definition below; label converted-model experiments as local operator adaptations, not the published training run |
| TEAL-FFN | Per-projection magnitude thresholds estimated from calibration CDFs, with nonuniform FFN budget allocation | Strong unweighted activation-sparsity baseline | Include calibrated allocation, not only equal density; actual token density can vary [R04][R05] |
| WINA-FFN | `\|x_r\|·\|\|W[r,:]\|\|₂` scores; calibrated thresholds and nonuniform FFN allocation | Direct weight-aware routing comparator | Correct row convention; transformations only when unmasked equivalence is demonstrated [R02][R03] |
| WiSparse-FFN | `\|x_r\|·\|\|W[r,:]\|\|₂^α_l`, calibrated α and block/projection allocation | Contemporary stronger score/allocation family | Include α=0 and α=1; record search and scope differences [R06] |
| WAS-FFN | `\|x_r\|·\|\|W[r,:]\|\|₁` scores and calibration-only constrained scheduling | Error-envelope and scheduling competitor | Preserve L1 rather than silently substituting WINA's L2 score [R07][R08] |
| CATS-FFN | Full gate calculation, threshold post-SiLU gate, then skip associated up columns and down rows | Can save two FFN branches, unlike down-only AAC | Give it a joint FFN interface and a fused/shared sparse backend; report full gate cost [R42] |
| TOPK-TRAIN | SiLU-preserving top-k sparsity with straight-through training, plus an intra-block N:M variant | Matched sparsity-aware training comparator motivated by Q-Sparse | Label the exact normalization variant and local adaptation scope; do not claim complete Q-Sparse replication [R09] |
| GROUP-DROP | Input source-group dropping without mandatory diagonal contributions | Isolate value of local diagonal preservation | Compare at matched actual work/frontiers, not the same k, because cost differs |

These core alternatives are ordered implementation tasks, not all automatically multi-seed long runs. Frozen diagnostics and development pilots screen them; confirmed claims use the strongest credible survivors and document exclusions. If a necessary contemporary baseline cannot be implemented faithfully enough, narrow the claim to the named local adaptations rather than claiming superiority to the entire published method.

### 5.3 Algebra adapter specification

For the direct star row action, source `α` communicates to all output groups while every other group retains its diagonal contribution. Implement both the paper's position-based row-type schedule and a balanced cyclic schedule as separately named configurations. The published training schedule uses `α(p)=floor((p mod 768)/192)` with zero-based positions; keep absolute positions consistent through prefill and cached decode. [R01]

For a tensor-square adapter, represent `α`, output group `β`, and input group `κ` as two-bit indices. Retain a block exactly when each bit of κ belongs to the corresponding endpoint set `{α_bit, β_bit}`. Treat this as a set: do not duplicate a term when endpoint bits coincide. The per-row count is `1+2+2+4=9`; Appendix A independently checks the enumeration. Do not force a false equal-count comparison with AAC's 7 or 10 blocks. Compare frontiers or explicitly cost-matched mixtures with all scheduling costs included.

### 5.4 Shared calibration and baseline-specific adapters

Use a common calibration pool disjoint from adaptation, development, and protected evaluation. Stream activations through one layer at a time; do not cache every token at every layer. Store score CDF sketches or bounded deterministic samples, block-output targets, source revision, and calibration hashes.

**Threshold calibration.** For each projection and candidate retained fraction, take the corresponding quantile of the method's score distribution. Define the comparison as `score ≥ threshold`, including all ties; count actual retained entries. If a hard exact-count deployment is needed, top-k with deterministic index ties is a separate variant. Thresholding does not guarantee a per-token hard budget.

**Common allocation control.** Build a lookup table of development-free calibration distortion versus retained fraction for each eligible projection. Starting from a sparse feasible allocation, repeatedly add the increment with greatest calibration distortion reduction per actual weighted MAC. Distortion is squared FFN-output error at the same input. Weight increments by each projection's true dimensions. This shared allocator is a proposed strong control, not an assertion that every paper uses this exact algorithm.

Start retained-fraction lookup tables at increments of 0.05, include the exact AAC logical fractions and full density, and refine intervals around nondominated configurations by halving their spacing. Start narrow FFNs at retained fractions 1, 3/4, 1/2, and 1/4 rounded to a supported physical width, then refine the frontier. These are initial search discretizations, not success thresholds. A grid interval is sufficiently resolved when interpolated changes are smaller than measured development uncertainty or all distinct executable widths/masks in it have been tested. Threshold and allocator search uses calibration only; checkpoint/configuration choice uses development. Record the number of evaluations and stop after a predeclared batch without a resolvable frontier change, rather than after a favorable protected result.

**WINA-specific source fidelity.** Inspected upstream calibration files explicitly use CUDA and include model-specific weight tables; recompute costs from the local model instead of copying 7B/8B constants. Its greedy code evaluates a particular norm over an activation tensor and can propagate sparse activations. Preserve that source behavior in a separately named `wina_source_metric` mode where tensor semantics match; compare the common MSE allocator as another mode. Do not silently replace the metric and call the result an exact port. [R03]

**WiSparse-specific calibration.** The inspected appendix searches α on `{0,0.05,…,1.45}` and clamps norm powers below `1e−4`; include this source-derived candidate family. Implement block-density search followed by within-block allocation, and compare the shared allocator. A local reproducible search uses parent retention, recorded pseudorandom mutations of about 10% of blocks, projection-cost-weighted budget repair, and teacher KL on calibration data. Mutation step and evaluation count are selected by pilot saturation and frozen. Record this as `wisparse_ffn_local`, because scope and search engineering differ from the paper. [R06]

**WAS-specific scheduling.** For target mean sparsity r, implement a source-motivated schedule with `s_1∈[r−0.05,r]` and `s_l∈[s_(l−1),r+0.05]`, clipped to [0,1]. Reject schedules outside a logged mean-budget tolerance. Use a locally pinned TPE implementation if available and licensed; compare a fixed-seed random-search control over the same domain. Use calibration NLL, never an official test split, for its objective. Also test an unconstrained shared allocator so an unsuitable monotonicity assumption does not handicap the L1 score. Save every proposed schedule and its objective. [R07][R08]

**CATS-specific execution.** Compute `g=SiLU(h W_gate)`, use a frozen calibration threshold on `|g|`, then compute only selected columns of `h W_up` and selected rows of `W_down` before the output reduction. With uniform retained fraction ρ and square-equivalent FFN dimensions, the projection arithmetic is `dF + 2ρdF`, not `3ρdF`. Do not charge it as three sparse branches or implement only its final down-row omission. The training comparison uses the same local full-FFN adaptation allowance, explicitly distinct from the paper's adaptation recipe. [R42]

**Grouped versions.** Add weight-aware whole-source-group selection as a matched diagnostic, not as a replacement for stronger fine-grained methods. A fine-grained algorithm may pay more routing cost but preserve quality better; both effects belong in the frontier.

### 5.5 Reference-only and conditional alternatives

A Monarch-style local factorization may be tested after the mandatory simple structured baseline. Define it exactly as `W≈B1 P B2`, where B1 has q diagonal blocks of shape `(d/q)×(r/q)`, P is a fixed permutation of r latent coordinates, and B2 has q diagonal blocks of shape `(r/q)×(m/q)`. Its parameter/MAC count is `r(d+m)/q`, excluding permutation overhead. Fit with teacher reconstruction and then matched adaptation. This is a **Monarch-style local construction**, not an assertion of exact reproduction of the conference implementation. [R16]

CLADA supplies a context-adaptation reference. A local causal control may condition budget on already available prefix statistics or previous-token logits. Current-token surprisal computed from a future target or a forward pass not yet paid for is forbidden. Without a verified faithful causal port, label it `causal_context_control`, not a win over CLADA itself. [R17][R18]

ProSparse, MoEfication/DOT-MoE, Spark, ActTail, and Learning to Allocate inform extensions and broad-claim limits. Their larger training, altered architectures, or different sparsification scope make published rates noncomparable to the core local experiment. BLASST operates on attention and is not required to isolate FFN behavior. Excluding these from a direct table does not establish superiority to them. [R10][R12][R14][R19][R20][R21][R22]

---

## 6. Data, workloads, metrics, and confirmatory design

### 6.1 Asset and split policy

Do not download all possible corpora before the hardware/theory gates. Fetch model metadata and licenses first, then the smallest needed assets. Resolve repository revisions to immutable identifiers at execution time and save them; this document does not invent commit hashes.

| Asset | Purpose and exact selection | Provenance/access/license handling |
|---|---|---|
| `HuggingFaceTB/SmolLM2-135M` base | Smoke tests and low-cost pilot | Official card/config; preserve actual weight and tokenizer revisions and license files [R34] |
| `HuggingFaceTB/SmolLM2-360M` base | Main local model | Same provenance controls; no instruction-tuned substitution [R35] |
| `HuggingFaceTB/SmolLM2-1.7B` base | Conditional scale check | Only after peak-memory and scientific escalation gates [R36] |
| `Salesforce/wikitext`, `wikitext-103-raw-v1` | Core English adaptation, calibration, development, and protected held-out documents | Official dataset card. Its metadata/prose present inconsistent CC-BY-SA version information; preserve both and resolve applicable terms before redistribution. Do not assert a single verified license version [R37] |
| `EleutherAI/lambada_openai`, English test | Narrative/last-word guardrail | Official dataset documentation; inspect referenced license and upstream-text conditions, do not assume every underlying novel is freely redistributable [R39] |
| Official HumanEval data | Code conditional NLL on `prompt` and `canonical_solution` | Repository/data provenance and license preserved; generated code is not executed on the host [R40] |
| `openai/gsm8k`, `main`, official test | Mathematical-solution text NLL; optional answer accuracy if the dense model is above floor | Official card/schema/license; no fine-tuning on the test answers [R41] |
| SmolLM-Corpus | Optional broader adaptation only if narrow-corpus shift becomes the blocking explanation | Subset-specific conditions matter, particularly code provenance; do not treat umbrella metadata as a license for every underlying source [R38] |

If an asset is unavailable or its applicable terms cannot be accepted for the intended local use, stop the dependent work package. Choose a documented substitute with comparable domain and split integrity, amend the protocol, and restart affected calibration. An asset change after seeing test results requires a new confirmatory population. Do not scrape an unlicensed replacement silently.

### 6.2 Core article reconstruction and deterministic splits

Reconstruct WikiText articles from the official training rows. Identify top-level headings using a parser that distinguishes a single enclosing equals marker from subsection headings with multiple markers. Preserve content/newlines. Unit-test the parser on a manually reviewed sample, and save article IDs and text hashes. Do not randomly split rows of one article across train and evaluation.

Assign complete articles by SHA-256 of canonical text plus a fixed split salt: 80% adaptation, 5% calibration, 5% development, 10% sealed confirmation. These proportions are organizational starting choices, not a power calculation. Before sealing, use calibration/development article lengths and variance to determine how much of the protected pool will be needed. Keep unused protected articles reserved for a genuinely amended future confirmation; do not repeatedly sample new ones until significance appears.

Use the official validation split as an external development diagnostic only if declared in the manifest. Use the official test split as a secondary external check, not for threshold selection. If its number of independent articles is too small for a precise claim, report that limitation instead of treating individual tokens as independent replicates.

Deduplicate exact and near-duplicate project text across splits before training, using a declared normalization and document-level near-duplicate rule. Log removals and hash manifests. Public pretraining contamination cannot be ruled out from these assets; the resulting claims concern relative behavior of adapted versus reference models, not acquisition of previously unseen knowledge.

### 6.3 Tokenization and likelihood evaluation

Pin the exact tokenizer revision. Use the base model without a chat template. Honor its BOS/EOS configuration and avoid accidental double insertion. Store token arrays and document boundaries; int32 token storage is `4N` bytes for N tokens, excluding indices/metadata.

The general, secondary full-text likelihood context is 1,024 tokens. The primary deployment-matched likelihood protocol is defined in §6.4. Use document-local sliding windows with stride 512 and count every eligible target token **once**, masking labels belonging only to repeated context. Add BOS once per document according to the resolved tokenizer convention. Reset document state and do not permit future tokens in a window to affect earlier predictions. Publish exactly how first-token losses and short articles are handled.

For sparse position-dependent schedules, carry each target's absolute document position through overlap windows. The evaluator must not assign a different row type merely because the window starts elsewhere. For long documents beyond the model's configured positional limit, use a declared shared truncation/window reset policy, applied to all methods and recorded in metadata; do not silently extrapolate rotary positions.

In addition to sliding-window likelihood, validate cached sequential teacher-forced scoring against an equivalent unwindowed causal reference on short fixtures. These checks catch cache, mask, label-shift, and position bugs. Likelihood evaluation and performance workloads have different purposes and may use different context lengths, but each comparison must be internally matched.

### 6.4 Primary and secondary claims

**Primary local claim:** at the predeclared quality tolerance, AAC-B has lower end-to-end batch-one decode latency than the best eligible local comparison selected on development data.

**Primary quality population and execution:** documents with more than 512 tokenized tokens, evaluated as fixed-prefix continuation windows. For target starts t=512,640,768,…, prefill the immediately preceding 512 tokens through that method's **actual deployment prefill policy**, then teacher-force and score up to 128 continuation targets through its actual decode policy. Use the final shorter continuation when necessary. Count each target after the initial 512 exactly once and bootstrap entire documents. The first target uses the last prefill logit; subsequent targets use cached one-token forwards. Reset each window's cache; carry the declared logical row-type origin consistently. Use the same finite positional convention for every comparator. Report initial-prefix and short-document coverage separately through the full-text guardrail; they are not silently represented by the primary continuation metric.

A reference implementation may compute masked prefill outputs densely for correctness, but cannot silently replace the logical sparse operator by the dense operator. Conversely, a dense-prefill/sparse-decode deployment configuration must be scored with that mixed policy and its resulting cache, not by all-dense parallel likelihood. Thresholds and quality-based configuration selection must use this same deployment-matched protocol on calibration/development. Full-text NLL is reported separately and must disclose which logical policy it uses.

**Primary quality quantity:** paired difference in token-weighted held-out continuation negative log likelihood,

`ΔNLL = (Σ_d loss_AAC,d − Σ_d loss_baseline,d) / Σ_d target_tokens_d`,

in nats/token. For the quality-budget test, the reference in this expression is always D0, the original optimized dense checkpoint at the specified inference precision. Also compute and report the paired difference to the selected sparse/quantized/adapted competitor. Relative perplexity change is `exp(ΔNLL)−1`. Report both absolute continuation perplexities. Aggregate by the same target tokens; no method-specific truncation.

**Quality tolerance:** the supplied proposal suggested no more than 1% relative perplexity degradation. Use `δ_Q=ln(1.01)≈0.00995033` nats/token as an explicitly inherited, provisional application assumption, not as a universal standard or an invented minimum improvement. Report the zero-degradation comparison and the entire measured frontier as well. Freeze the tolerance before protected evaluation. Without an application owner, a quality-tolerated speedup must not be called “no quality loss.”

**Primary performance quantity:** paired log ratio of full-model decode milliseconds per generated token, excluding prefill but including actual routing, projection execution, cache update, vocabulary output, and the same token-selection work for every method. Report total prefill-plus-generation latency separately. A speed win requires a confidence interval supporting lower latency; no arbitrary minimum percentage is required.

**Comparator selection:** on development data, find each baseline's fastest configuration satisfying the common D0-anchored quality budget and declared guardrails. Select the fastest credible eligible baseline across families for the primary latency comparison. Retain slower higher-quality alternatives on the frontier. In confirmation, test AAC's quality against D0, not against an already degraded baseline; this prevents compounding two separate 1% allowances. Report direct AAC–competitor quality differences, and describe the result as “faster within a common quality budget,” not identical quality when they differ. A competing configuration whose protected quality misses the common budget does not automatically establish an AAC win; report the frozen comparison, include the already declared dense reference, and require fresh confirmation for any new selected competitor.

**Mechanistic co-analysis:** at matched logical/executed work and matched adaptation, compare AAC-B to STATIC and to adaptive μ=0. This asks whether adaptivity and the bound penalty explain a gain, not simply whether AAC can run.

**Secondary metrics and guardrails:** layer and FFN relative reconstruction error; B/U/V tightness; dense fallback rate; tolerance violations; source occupancy/entropy; p50/p95 document error; rare-token and domain NLL; prefill latency; total request latency; peak active/cached memory; packed-weight storage; compilation/export time; adaptation cost; output consistency across cache modes; and failures/nonfinite values. Do not summarize a severe domain regression away through an average.

Code/math solution NLL measures retention of probability assigned to reference text, not the ability to solve a problem. Optional generation task scores must include their own evaluation uncertainty and can be uninformative when the baseline is near chance/floor. Never use a near-zero benchmark score as evidence of no meaningful regression.

### 6.5 Performance workloads and timing boundaries

The provisional primary deployment workload is batch one, a 512-token prompt, and exactly 128 generated tokens. Standard generation obtains the first token from prefill logits, followed by 127 one-token forwards; record the actual forward-call count rather than calling these 128 identical model invocations. It represents a concrete local generation case, not an alleged universal workload distribution. Freeze it before looking at protected results. Secondary shape coverage includes prompt lengths 128 and 2,048, decode lengths 128 and 512, and batch four when feasible. Include a long-context case within the configured context limit. If an actual intended usage population becomes available before confirmation, replace the provisional workload through a recorded amendment, not post hoc favorable weighting.

Run two modes:

1. **Fixed-token cached replay:** all methods consume the same predetermined causal token stream, obtained independently of protected outputs. Compute normal logits and common token-selection work but force the next provided replay token. The router sees only currently available states, never future replay labels. This controls route/workload differences caused by differing generated text.
2. **Free generation of fixed length:** same prompts and generation policy, exactly the same number of generated tokens and corresponding model calls, EOS not used to shorten the timing run. Report this separately because token paths and quality can differ. Natural-EOS interactive latency is a secondary deployment metric, not the primary speed comparison.

Warm up every measured shape, dtype, and reachable compiled kernel path. Stop warming when compile events cease and pilot timings stabilize; record counts. Materialize outputs with `mx.eval`, synchronize the streams actually used before and after the timed region, and include cache updates. MLX is lazy; merely constructing a graph is not a timing result. Default synchronization need not cover a separately used stream. [R26][R31]

Use monotonic wall-clock timing around the complete boundary. Retain per-request and per-session measurements. Report aggregate tokens divided by aggregate seconds as throughput; do not average reciprocals inconsistently. Record cold compilation, weight packing, norm export, and asset loading separately, plus amortized costs for short and long sessions.

Use randomized paired method order or ABBA blocks within sessions, rotate prompt order, and keep the model workload identical. Record AC/battery status, Low Power Mode status when available, OS build, thermal/pressure indicators, other GPU use, and elapsed session time. Do not change security or system settings automatically. A session with swapping or severe unrelated contention is flagged and investigated under predeclared exclusion rules, not discarded solely for giving an unfavorable result.

`powermetrics` or another locally supported measurement may be used only after checking its capabilities and any permission requirement. Label package/device energy separately from whole-system energy. Battery-level changes or TDP estimates are not valid per-operation energy measurements. Energy is optional when measurement access is inadequate; latency is not replaced by an invented energy proxy.

### 6.6 Statistical units, replication, and sensitivity

Tokens from the same article, repeated timings from one hot loop, and checkpoints from one training run are not independent observations. The hierarchy is: adaptation seed, document/prompt, and independent timing session. Pair methods on the same document and seed where possible. Estimate variability due to model adaptation separately from variability due to test documents and timing sessions. General benchmark-variance concerns are supported by prior work; the procedure here is a proposed design. [R43]

Use development-only pilots to estimate these variance components. Begin with three adaptation seeds for variance diagnosis, not as an automatic sufficient final sample. Begin with three distinct timing sessions for a variance diagnostic, then choose the final count from the precision design rather than treating three as sufficient evidence. Increase independent sessions instead of only inner-loop repetitions. For confirmation, use a crossed bootstrap resampling seed IDs and document IDs independently while preserving method pairing; for timing, resample sessions and prompt clusters, retaining their internal repeated measurements. Report seed-specific effects in addition to pooled intervals. With very few seeds, declare the adaptation-generalization interval unstable.

Calibrate the fixed confirmatory sample size using the pilot variance and a predeclared scientifically relevant effect or desired confidence-interval width. A normal approximation `n≈[(z_(1−α)+z_(1−β))·s/δ]²` is only an initial diagnostic for an approximately independent paired quantity; the actual crossed design should use resampling-based power simulations from development data. For noninferiority, plan for the gap between the expected ΔNLL and δ_Q, not a generic effect size. Document the expected precision under the available resource budget.

Do not choose a minimum speedup merely to manufacture a pass/fail threshold. Report the detectable effect given resources and separately calculate practical break-even. If the resources cannot distinguish a small useful effect from zero, say “inconclusive at this precision.” A fixed-resource precision design is acceptable; optional stopping at the first favorable p-value is not.

Before protected evaluation, freeze: the primary AAC configuration; one strongest development-selected comparator; model size; adaptation budget; eligible layers; workload; quality margin; sample sizes; randomization; exclusions; and analysis script hash. The joint primary success rule is quality noninferiority to D0 **and** latency superiority to the frozen development-selected eligible comparator. Use an intersection-union interpretation for these co-requirements. If several primary comparators or budgets are declared, control that family explicitly; do not select the best protected result afterward. Mark broad ablations exploratory or apply a declared multiple-comparison procedure such as Holm to a small confirmatory family.

Protected outcomes are evaluated once under the fixed design. An amendment after inspection needs fresh independent confirmation; old outcomes remain exploratory. Do not rebrand a repeated test as a new experiment just by changing its random seed.

### 6.7 Practical significance and equal-cost interpretation

Report absolute milliseconds saved per request, relative latency, memory change, and adaptation/export cost. Estimate a break-even number of generated tokens as

`(additional adaptation + calibration + export seconds) / saved seconds per token`,

when the denominator is reliably positive. This is an amortization calculation, not a prediction of user adoption. Provide the underlying uncertainty and also cases with no break-even.

Show separate frontiers for unchanged weights, adapted models, and quantized deployment. A point can improve latency while increasing storage, training expense, or tail error; show those tradeoffs. A gain versus dense alone is insufficient to claim superiority to adaptive sparsity. A gain versus an unoptimized port is insufficient to claim a hardware contribution.

---

## 7. M3 Max stack, memory, and kernel feasibility

### 7.1 Hardware discovery before substantial downloads

Execute and save the following existing operating-system commands where present:

```bash
uname -m
sw_vers
sysctl -n hw.memsize
sysctl -n hw.ncpu
system_profiler SPHardwareDataType SPDisplaysDataType
vm_stat
df -h .
```

Also record process architecture from Python, available Python versions, Xcode/command-line-tool availability, free disk in bytes, and power/thermal information obtainable without privileged changes. Expect native `arm64`; a Rosetta Python is a configuration error to correct before measuring GPU behavior. Record the exact M3 Max variant rather than inferring it from the laptop's name.

The inspected MLX installation documentation requires Apple Silicon, Python at least 3.10, and macOS at least 14 for the relevant Apple path. Start with a native Python 3.11 environment if compatible packages resolve; install a currently compatible MLX/MLX-LM pair, then lock the actually installed versions and source revisions. The documentation displayed MLX 0.32.3 during this research, but that is not a claim that an untested dependency combination is validated on this laptop. [R25]

Do not promise support from a package name alone. The first smoke test must exercise model loading, the exact dtype, autograd through the chosen native operations, grouped shapes, cache updates, and the relevant multiplication APIs. Follow official installation instructions; do not fabricate a wheel or release number.

### 7.2 Software choices and fallback paths

**Primary:** Python, MLX, MLX-LM, NumPy for CPU references, a pinned tokenizer/dataset loader, and standard local JSON/CSV or Parquet tooling. Use established statistical tooling only after pinning and checking its local dependencies. The inspected MLX-LM and TEAL repositories identify MIT licenses, and the main SmolLM2-360M card identifies Apache 2.0 licensing; preserve the applicable notices and validate the actual downloaded revisions. [R05][R32][R35] MLX-LM already provides inference, adaptation infrastructure, and quantization capabilities, while its inspected Llama module gives the needed FFN integration points. [R32][R33]

**Native inference candidates:** optimized dense operations, `block_masked_mm`, and `gather_mm`. **Custom inference candidate:** `mx.fast.metal_kernel` after measured headroom justifies it. The official custom-kernel API exposes named inputs/outputs, source, launch geometry, shapes, and dtypes; this is not a promise of automatic gradients for the custom operation. [R27]

**Training fallback:** native MLX tensor operations and autograd, even if they calculate all blocks. This preserves the scientific model comparison but not a sparse-training-efficiency claim. CPU NumPy/fp64 is a correctness oracle, not the performance baseline. A CPU-only deployment cannot establish the desired GPU speedup. PyTorch MPS is not a required dependency; switching stacks must preserve precision and measurement fairness and be recorded as a feasibility amendment.

If the exact custom or masked operation fails locally, distinguish an API/version issue from an algorithmic failure. Keep correctness experiments executable through the native reference path. Do not proceed to a claimed hardware result until at least one faithful accelerated path exists.

### 7.3 Memory model and conditional model selection

For P total parameters and P_t trainable parameters, an initial fp32 student + fp16 frozen teacher + fp32 gradients and two fp32 Adam states has persistent parameter/state components approximately

`M_persistent = 6P + 12P_t bytes`.

For full-parameter adaptation this is `18P` bytes. This model excludes activations, temporary gradients, copied/packed weights, kernel workspaces, logits, graph compilation/cache, dataset buffers, allocator fragmentation, and macOS. It is a component estimate, **not peak memory**. Confirm the actual optimizer's state dtypes and whether any extra master copy exists.

The following calculations use the inspected official configurations and tied embedding assumption, which must be validated against the runtime parameter count. [R34][R35][R36]

| Base model | Parameters calculated from config | FFN trainable parameters | fp16 model weights, GiB | Student + teacher + FFN-only train states, GiB | Full-adaptation persistent components, GiB | B=1, T=4096 fp16 KV cache, GiB |
|---|---:|---:|---:|---:|---:|---:|
| 135M | 134,515,008 | 79,626,240 | 0.251 | 1.642 | 2.255 | 0.088 |
| 360M | 361,821,120 | 235,929,600 | 0.674 | 4.659 | 6.066 | 0.156 |
| 1.7B | 1,711,376,384 | 1,207,959,552 | 3.188 | 23.063 | 28.689 | 0.750 |

KV memory uses `2·L·B·T·n_kv_heads·head_dim·bytes_per_element`; it is an inference quantity, not a training-cache allocation. The configurations are respectively `(d,F,L,n_heads,n_kv)=(576,1536,30,9,3)`, `(960,2560,32,15,5)`, and `(2048,8192,24,32,32)`, with head dimension 64 and configured context 8192. [R34][R35][R36]

A full fp32 logit tensor alone costs `4·B·T·V` bytes: at B=1, T=1024, V=49152, about 0.188 GiB per such tensor. Multiple teacher/student probabilities, loss temporaries, and graph retention can multiply that. Chunk vocabulary KL and token batches; verify exact loss equivalence on small fixtures.

Start with 135M, microbatch one, training sequence length 256; measure a complete forward/backward/update and checkpoint. Increase to 512 and 1024 only after measuring peak usage. Use gradient accumulation to choose a reasonable effective token batch. For 360M repeat the capacity test rather than extrapolating parameter memory alone.

Reserve at least `max(6 GiB, 25% of physical memory)` for OS/other processes as an initial conservative policy, and lower the application cap further if current available headroom or memory-pressure telemetry requires it. This is a safety policy, not a discovered property of all M3 Max configurations. Never depend on swap to make a training run appear feasible. Adjust the cap from observed stable operation and documented user constraints.

Advance to 360M when the full pilot update, teacher path, and checkpoint fit the cap without persistent pressure. Advance to 1.7B only if its measured incremental activation/workspace needs fit in addition to the table's large persistent components. A machine that cannot fit 1.7B full-FFN training may still do its frozen inference/oracle check. Do not quietly substitute LoRA and call it the same representation-learning test.

### 7.4 Disk, time, and calibration model

Maintain a disk budget before each stage:

`models + tokenizer/data cache + token arrays + 2 resumable checkpoints + best checkpoint + optional packed exports + raw results + atomic-write headroom`.

A resumable FFN-only optimizer checkpoint includes approximately `4P_t` trainable weights plus `8P_t` moment states, additional frozen/base references or full weights according to the chosen serialization, and small RNG/configuration metadata. Do not serialize the teacher repeatedly. Retain the last good checkpoint while atomically writing the next; reserve space for both. Hash and reference a reusable base rather than duplicating it in every run when licensing and reconstruction permit.

Measure steady-state training tokens/s over enough updates to include optimizer and periodic evaluation/checkpoint effects. Estimate a run as `planned tokens / measured tokens_per_second + measured evaluation/checkpoint cost`; publish the estimate and its pilot range. Do not give an unsupported fixed-hours claim. Inference calibration similarly uses measured request costs per shape. Large activation dumps are not needed: stream fixed-size reservoirs/CDF sketches, and bound their memory explicitly.

Store measured peak active and cached allocator memory separately from OS pressure. MLX documents device/memory inspection APIs; probe the installed names/signatures and preserve the result instead of assuming legacy and current aliases are identical. [R30]

### 7.5 Masked multiplication and alignment constraints

The inspected `block_masked_mm` API accepts block size 32 or 64 and masks on its matrix tiles. It is not an arbitrary per-row dynamic source-mask primitive merely because its name contains “masked.” [R28]

For q=4, the 135M residual group width is 144 and the 360M width is 240, neither divisible by 32. Padding those **inside each group** to 160 and 256 introduces approximately 11.1% and 6.7% extra width respectively. Padding only the entire tensor tail does not align the internal group boundaries and can implement the wrong support. FFN hidden group widths 384 and 640 are aligned at 32. Count padded operations and memory, and test removal of every padded output coordinate.

A q=2 fallback on 360M has residual group width 480, which is aligned to 32, but its one-source arithmetic is 3/4 dense and offers a different tradeoff. A ragged custom q=4 kernel is another option; a padding penalty in one generic primitive does not by itself falsify q=4.

For independently masked tokens, a representation like `X[N,1,d]` with batched/broadcast weights and matching mask batch dimensions may be necessary; verify broadcasting semantics with small tests. A standard two-dimensional `X[N,d] @ W[d,m]` does not automatically accept an independent RHS block mask for every row. A per-token GEMV formulation may be poor for prefill; shape-dependent dense prefill is allowed, with its quality and cost measured as a distinct deployment policy.

### 7.6 Fixed-k gather path

`gather_mm` selects among batches of matrices using indices; it is not a general sparse channel gather by name alone. [R29] A proposed compatible fixed-k design is:

- Pack q² blocks into a bank `[q*q, d/q, m/q]` and input groups into `[N*q,1,d/q]`.
- Precompute the `C=q+k(q−1)` retained `(i,j)` action pairs for each same-k mask.
- For token n and each action, gather LHS batch index `n*q+i` and RHS bank index `i*q+j`.
- Multiply to get action outputs and reduce them by output owner j, with each diagonal counted once.

Confirm index/broadcast behavior locally against the reference. Intermediate action tensors and reductions are real costs. A variable-k implementation padded to all 16 actions can erase the supposed savings; do not claim variable-cost execution from it.

Packing a block bank may duplicate the original dense FFN bank. Count both if dense fallback requires the original layout. A reshape/transpose is not automatically a free contiguous layout for both kernels. Custom kernels reading canonical stored weights can avoid this duplication, but must demonstrate correct strides and efficient loads.

### 7.7 Dynamic dense fallback and custom kernel specification

A crucial implementation invariant is **do not evaluate both sparse and dense products to select one**. A device `where` expression around two already evaluated branches can do exactly that. A Python branch on a GPU mask introduces a device-to-host synchronization. Neither cost may be hidden.

There are three honest paths:

1. Fixed-k compiled graphs, with the budget selected before the device operation.
2. A fused dynamic Metal kernel that reads the device mask and loops only over retained source groups, including all groups for the full mask. Its full-mask path is mathematically dense but may not be as fast as stock dense GEMM/GEMV; measure it.
3. Explicit host readback followed by a native dense/sparse launch, charging synchronization and dispatch in the timed path.

Only shape-based decisions known before routing can freely choose native dense prefill without a new data-dependent synchronization. A cost table must model the path actually executable, not optimistically assume arbitrary per-token access to the best independent native kernel.

For a custom inference kernel, use output-owned tiles, fp32 accumulation initially, canonical weight indexing, and no cross-thread nondeterministic output atomics. For each output group j, always compute source j and loop over selected `i≠j`. Bounds checks must skip physical padded coordinates. Avoid loading an omitted weight before checking its mask. Router and projection may be separate GPU kernels first; fuse only if profiling shows a benefit. The compiler, launch configuration, source, and correctness fixture hashes are export metadata.

The router computes group norms in fp32 and enumerates B on the device. Test ties, near ties, zero groups, ragged boundaries, NaNs, infs, all masks, and long position/cache sequences. Do not publish a mathematical certificate based on undocumented reassociation or accumulator precision.

### 7.8 Kernel acceptance before expensive training

Profile the original dense model and every eligible projection shape. Implement a no-router fixed mask with the same sparse support and realistic data movement. This gives an **optimistic attainable reference**, not a proof of the best possible sparse kernel. If it already loses substantially to dense and the Amdahl estimate leaves no practical room, postpone training-dependent hardware claims while checking a second plausible layout/backend.

Escalate to custom Metal only when there is a quality/oracle reason to pursue the mask family and evidence that launch/layout overhead, rather than absent computational headroom, blocks native execution. Negative results in one library primitive are an implementation result; negative results across reasonable optimized paths with measured bandwidth/overhead can support a local hardware conclusion.

---

## 8. Executable work packages and decision gates

The commands in this section beginning with `python -m aac` are **proposed project commands that the coding agent must implement**. They are not claimed to exist in an upstream package. The OS commands in §7.1 and the self-contained NumPy check in Appendix A are existing executable interfaces. Every project command must accept `--config`, write a resolved configuration, and return a nonzero status for invalid/blocked runs.

### WP00 — Discover hardware, resolve licenses, and validate the stack

**Objective / risk:** establish actual local capacity and critical operation support without assuming a maximum-memory M3 Max.  
**Dependencies / entry:** fresh laptop, this document, public network access for permitted assets. No large model download required initially.

**Actions:** implement `hardware.py` and `assets.py`; run the OS inventory; create a native environment; inspect/accept licenses; pin packages and repository revisions; test MLX device availability and fp32/fp16 operations; test autograd, cache updates, `block_masked_mm`, and `gather_mm` on tiny matrices. Only then download 135M and run an unmodified forward and generation fixture. Verify tokenizer/configuration hashes. Check bfloat16 separately if proposed; do not assume it from the model card's saved dtype.

**Proposed command:** `python -m aac doctor --config configs/smoke.yaml`.

**Artifacts / tests:** `hardware.json`, `environment.json`, dependency lock, license inventory, asset manifest, `operation_support.json`; each operation records shape, dtype, success/error, and CPU-reference discrepancy. Confirm no Rosetta execution and no silent CPU fallback in a GPU benchmark.

**Resources:** tiny tensor tests first; then one 135M model and its tokenizer. Measure actual peak, free disk, and load cost.

**Gate / next action:** supported and licensed → WP01/WP02. Version or installation failure → repair and rerun, retaining errors. Missing custom support → native correctness path remains possible, hardware claim blocked. Insufficient space or unresolved asset terms → stop that dependency, not a scientific rejection. Do not silently upgrade macOS or accept terms on the user's behalf.

### WP01 — Prove definitions and test limiting cases

**Objective / risk:** eliminate wrong algebra, bound, dense limit, count, and numerical semantics before neural experiments.  
**Dependencies / entry:** CPU NumPy environment; independent of custom GPU kernels.

**Actions:** place Appendix A's script under tests; implement general rectangular/ragged reference operations and U/V/B; enumerate masks; add q=1, empty/full masks, zeros, nonfinite values, nonuniform groups, spectral/Frobenius comparisons, cancellation fixtures, and alias/stride tests. Verify the tensor-square adapter's distinct support. Test hard-budget infeasibility and deterministic ties. Compare finite differences away from routing ties with native gradients; do not expect a smooth derivative across a mask switch.

**Proposed command:** `python -m aac validate --suite math --config configs/smoke.yaml`.

**Artifacts / tests:** machine-readable assertion report, proof notes, forward/gradient error distributions, explicit counterexample fixtures. Exact integer associativity tests supplement but do not replace the proofs in §3.

**Resources:** small CPU matrices; GPU cross-check only after WP00. Scale stress tests logarithmically until dtype limits are approached and flag, rather than hide, overflow.

**Gate / next action:** all structural identities hold within declared numerical tolerances → WP03. Implementation error → fix and repeat. Exact-arithmetic violation of a claimed inequality → block dependent work and revise the mathematics. Floating-point disagreement → distinguish arithmetic error from a wrong bound; remove strict-certification wording until resolved.

**Initial numerical acceptance conventions:** use fp64 reference residual tolerance `1e−10·max(1, reference norm, bound)` for moderate-scale CPU fixtures. For ordinary-scale GPU fixtures, start with relative output L2 error ≤`5e−5` for fp32 and ≤`3e−3` for fp16 with fp32 accumulation, using a documented RMS-scale absolute tolerance for near-zero outputs. Also compare against a dense alternate-reduction control. These are engineering test tolerances, not an accuracy allowance for an approximate mask. Validate the full-mask model separately; its numerical NLL discrepancy must be small relative to the scientific quality margin. Never relax a tolerance solely to pass a failing kernel; investigate precision/reduction behavior and record any justified change before confirmation.

### WP02 — Build data manifests and an honest evaluation harness

**Objective / risk:** prevent leakage, label shifts, position errors, false speedups, and invalid independence assumptions.  
**Dependencies / entry:** WP00 licenses/tokenizer verified; may run alongside WP01.

**Actions:** implement article reconstruction, hashes, deterministic splits, tokenization, document-local loss accounting, causal replay, cache equivalence, and fixed-length generation. Build timing boundaries using materialization and synchronization. Implement per-document outputs and session-level timing records. Seal confirmation IDs in a manifest inaccessible to tuning commands by default.

**Proposed commands:** `python -m aac data prepare --config configs/pilot.yaml`; `python -m aac validate --suite harness --config configs/smoke.yaml`.

**Artifacts / tests:** `manifests/splits.json`, `tokens/`, parser review fixtures, `eval_documents.parquet`, causal/cache tests, A/A timing results. A/A compares the same model through two equivalent paths. Positive controls deliberately drop a known-important contribution for quality detection and add a known delay to a harness-only timing fixture; never include injected delays in scientific baselines.

**Resources:** streaming preprocessing, int32 token cache, bounded activation reservoir. Use the disk formula in §7.4.

**Gate / next action:** every target counted once, no future-information access, equivalent paths agree, and the harness detects known differences → WP03. Weak timing/document sensitivity → improve the design before interpreting differences. Split contamination or parser failure → rebuild manifests and invalidate affected downstream runs.

### WP03 — Measure the frozen model and the attainable approximation gap

**Objective / hypotheses:** H1/H2; determine whether the selected family and routing contain useful signal before adaptation or custom engineering.  
**Dependencies / entry:** WP01 and WP02 passed; 135M base loaded correctly.

**Actions:** profile dense end-to-end execution and eligible projections. Gather bounded calibration samples of down-projection inputs, weights, and outputs. For q=4 enumerate all masks and compute actual errors offline, with full computation charged to diagnostics. Compare B, V, U, magnitude, weight-aware group scores, best fixed masks, random/shuffled masks, and the exact-error oracle. Test q=2 as an alignment/granularity control. Repeat frozen diagnostics on 360M only after a capacity smoke test.

**Proposed command:** `python -m aac diagnose frozen --config configs/pilot.yaml`.

**Artifacts / outputs:** per-layer `mask_oracle.parquet` with model/layer/token IDs, mask, logical MACs, actual L2 error, B/V/U and norm method; source concentration/tail plots; measured FFN time fraction; router-free cost estimates. Oracle inputs remain calibration/development only.

**Resources:** stream one projection at a time; no all-layer activation dump. Calibrate sample counts until bootstrap intervals on adaptive-minus-fixed reconstruction are informative; additional independent documents matter more than repeated identical activations.

**Gate / next action:** oracle improves over fixed and B captures some of that gain → WP04/WP05. Oracle helps but bounds do not → test tighter frozen bounds or a clearly labeled empirical controller before abandoning the operator. No oracle gap in both model sizes → H1 is weakened; proceed only with a limited representation-learning diagnostic if a planted-structure test works and there is a specific reason adaptation could change the distribution. Do not escalate merely because larger models might someday help. No empirical headroom because FFNs are a small time fraction → narrow to a component claim or stop the deployment objective.

### WP04 — Implement and reproduce strong local baselines

**Objective / risk:** avoid mistaking weak controls for novelty.  
**Dependencies / entry:** validated data/harness and the common projection adapter; no custom AAC kernel required.

**Actions:** implement D0/D1, DQ, NARROW, STATIC, ALG, score/threshold families, CATS's joint FFN path, and TOPK-TRAIN. Save baseline-specific calibration behavior and scope differences. Reuse public licensed code when compatible; otherwise use the explicitly described local adapters. Inspect actual tensors to resolve row/column convention. Implement common allocation and source-specific modes where specified. Expose shape-based dense fallback equally.

**Proposed command:** `python -m aac baselines calibrate --config configs/pilot.yaml`.

**Artifacts / tests:** baseline fidelity cards; upstream source revisions/licenses; raw calibration curves and allocator trials; all-dense equivalence; retained-density/MAC counts; CATS up/down consistency; N:M exact counts; narrow-model channel consistency; row-type schedule/cache tests.

**Resources:** training-free calibration first. Share cached inputs/targets where valid. Tuning stops by development saturation and interval-based dominance, not by giving AAC more attempts. All calibration cost is logged.

**Gate / next action:** credible local implementations and at least one strong baseline per critical alternative explanation → WP06. Unsupported upstream kernels alone do not exclude a method; use the shared native backend. A missing forward definition, license, or important component → mark the method blocked and narrow future claims. A baseline dominates already → retain it as a comparator and reassess whether training can answer a distinct hypothesis.

### WP05 — Establish realistic hardware headroom before expensive adaptation

**Objective / hypothesis:** H5; test whether skipped block work could plausibly win on this machine.  
**Dependencies / entry:** dense profiling from WP03 and correct reference supports; may proceed alongside baseline calibration.

**Actions:** implement fixed-mask native masked/gather paths and a minimal output-owned static Metal prototype only if necessary. Test canonical versus packed layouts; count internal padding and duplicate storage. Benchmark all relevant projection shapes for batch-one decode and representative prefill. Measure a standalone device router. Construct an optimistic end-to-end estimate using measured f and actual sparse kernel ratios, not just MAC fractions.

**Proposed command:** `python -m aac bench headroom --config configs/pilot.yaml`.

**Artifacts / outputs:** shape/dtype/mask/backend latency table, memory/packing report, hardware profile, arithmetic/byte estimates, thermal/session logs, Amdahl calculation with uncertainty.

**Resources:** short calibrated kernel batches after warm-up; no full adaptation. Compile cost recorded separately. Do not spend extensive kernel engineering on a support family with no demonstrated approximation potential.

**Gate / next action:** favorable or unresolved but plausible static headroom → WP06; dynamic kernel deferred to WP07. Consistently slower static paths with little remaining hardware headroom → stop the local latency claim, but retain an explicitly narrower mathematical/quality experiment if useful. One API failure or one bad tile is not sufficient scientific rejection; test the alternative compatible path identified in §7 before concluding.

### WP06 — Run matched adaptation pilots and select the mechanism

**Objective / hypotheses:** H2/H3/H4; determine whether trainable compressibility adds anything beyond generic adaptation.  
**Dependencies / entry:** WP04 baseline correctness and WP05 plausible headroom, unless the claim is explicitly narrowed to quality only.

**Actions:** train 135M variants at equal adaptation data/teacher budgets: AAC-B μ=0; AAC-B μ>0; STATIC; best weight-aware adaptive method; CATS; TOPK-TRAIN; and NARROW survivors. Use paired seeds/data order where possible. For strict mechanism attribution, compare same projection scope and trainable tensors; for deployment, allow CATS its full joint FFN opportunity. Monitor dense-student quality and regularizer gaming. Use learning curves to choose budget/ramp/hyperparameters on development only, then transfer the selected experiment to 360M under measured capacity.

**Proposed command:** `python -m aac train pilot --config configs/pilot.yaml`.

**Artifacts / outputs:** resumable checkpoints, per-update losses/gradients/budgets/norm hashes, development NLL and reconstruction curves, actual training expenditure, mask distributions, failed runs, and a written selection decision.

**Resources:** measured tokens/s and peak-memory rule; gradient accumulation and recomputation if necessary. No wall-clock time estimate is asserted before calibration. Save before changing memory settings.

**Gate / next action:** μ>0 contributes beyond matched controls without unacceptable dense/tail degradation → keep it. μ=0 matches → remove novelty claim for the regularizer and retain simpler inference if competitive. STATIC matches → drop adaptive routing unless there is a separate hardware reason. Fine-grained/CATS/narrow baselines dominate → report the failure of AAC's proposed advantage; grouping or model-scale follow-up needs the diagnostic evidence in §10, not optimism. Undertrained or underpowered curves → explicitly inconclusive; continue only with a frozen, justified larger pilot budget.

### WP07 — Export and implement router-inclusive deployment

**Objective / risk:** turn a quality-promising operator into honest skipped execution.  
**Dependencies / entry:** WP06 selected a viable configuration; WP05 identified a plausible backend.

**Actions:** freeze student weights, recompute all norm tables, calibrate feasible execution costs, implement fixed-k and/or dynamic device-mask kernels, and test host-branch fallback as an explicitly charged alternative. Ensure full masks do not secretly evaluate both branches. Provide the same underlying sparse primitive and comparable optimization effort to compatible baselines. Save exact export layout, precision, compiler settings, bounds provenance, and reachable mask paths.

**Proposed commands:** `python -m aac export --config configs/pilot.yaml`; `python -m aac validate --suite deployment --config configs/pilot.yaml`.

**Artifacts / tests:** immutable deployment bundle; kernel-versus-reference numerical tests; all-mask and ragged-boundary tests; bounds audit; stale-table rejection; empty/full/invalid budget tests; cache-equivalence tests; weight-load/branch inspection; microbenchmark and full-model timing with routing included.

**Resources:** a student inference bank plus any explicitly counted packed copies; teacher not required for deployment. Export norm computation may be expensive and is timed. Variable mask occupancy and full-fallback frequency are included in the workload.

**Gate / next action:** numerical agreement, correct causal behavior, and plausible full-path advantage → WP08. Kernel bug → fix without changing the scientific configuration, retest. Sparse method correct but slower → local hardware failure or engineering limit, not a quality failure. Changing grouping, precision, projection scope, or loss to rescue speed requires a development amendment and potentially new adaptation, not an unrecorded export tweak.

### WP08 — Calibrate statistical sensitivity and freeze confirmation

**Objective / risk:** prevent adaptive selection, underpowered conclusions, and session artifacts.  
**Dependencies / entry:** a locked candidate and credible comparison frontier from development, with router-inclusive execution.

**Actions:** estimate document/seed/session variance; determine sample sizes or achievable precision; choose the strongest comparator using development only; freeze the protocol fields in §6.6; write analysis code before unsealing; verify A/A and known-difference controls again; record practical-quality assumptions and domain guardrails. Do not select the primary workload because it is AAC's fastest shape.

**Proposed command:** `python -m aac preregister --config configs/confirm.yaml`.

**Artifacts / outputs:** `prereg/protocol.md`, immutable `confirm.yaml`, comparator-selection report, power/precision simulation, exclusion rules, analysis source hash, protected dataset manifest hash, and amendment policy.

**Resources:** development-only extra evaluation and independent timing sessions. The resources needed for the desired interval width become an explicit go/no-go feasibility input.

**Gate / next action:** credible sensitivity within local resources → WP09. Inadequate sensitivity → choose a fixed precision-limited study and label it accordingly, or defer the broader claim; do not lower standards after seeing confirmation. Missing necessary baseline → narrow the claim before unsealing.

### WP09 — Execute the protected comparison

**Objective / claim:** test the frozen primary quality–latency claim.  
**Dependencies / entry:** WP08 complete; data/configuration/code/export hashes match.

**Actions:** run the preselected model/configuration/seed set, protected document evaluation, randomized paired timing sessions, and declared guardrails. Use exact fixed output lengths. Keep raw logs and unfavorable outcomes. Reject stale configurations automatically. Compute the predeclared intervals and joint primary decision without retuning.

**Proposed command:** `python -m aac evaluate confirm --config configs/confirm.yaml`.

**Artifacts / outputs:** per-document losses/counts, prompt/session timings, measured memory, raw masks or aggregated route summaries captured outside timing, primary/secondary effects with uncertainty, and a signed-by-hash decision report. No fabricated confidence interval when sample size is inadequate.

**Resources:** fixed design from WP08 and measured capacity. Checkpoints resume only under unchanged scientific state. Interrupted timing sessions restart their paired block and retain partial-session records rather than silently merging incompatible conditions.

**Gate / next action:** joint quality/latency evidence and no unacceptable guardrail failure → WP10 independent confirmation/attribution. Quality gain alone, speed alone, or a tradeoff → report that narrower outcome. Wide intervals → inconclusive. Credible domination by a baseline → negative result for the claimed local advantage. A harness/configuration failure → invalid experiment, repair and repeat on an appropriately unexposed or reclassified population.

### WP10 — Attribute, challenge, reproduce, and report

**Objective / risk:** determine whether any improvement survives stronger explanations and independent repetition.  
**Dependencies / entry:** WP09 results, including negative/inconclusive outcomes.

**Actions:** run the preregistered small attribution family, then clearly labeled exploratory stress tests in §9. Repeat the decisive result from a clean checkout/environment on the same laptop with an independent adaptation seed and reserved independent documents when the original claim changes. Regenerate the report entirely from raw artifacts. Verify an interrupted run resumes consistently. Audit baseline tuning and generic kernel sharing once more.

**Proposed commands:** `python -m aac ablate --config configs/ablations.yaml`; `python -m aac report --runs runs/`.

**Artifacts / outputs:** final frontier tables/plots with error bars; direct versus reference-only comparison labels; complete failure/omission inventory; cost amortization; reproduction instructions; audit log; claim-by-claim outcome matrix and evidence-triggered next decision.

**Resources:** allocate to the uncertainty still capable of changing the conclusion. Do not run a large grid of irrelevant variants. Independent confirmation is local and sequential, not a requirement for a second machine.

**Gate / next action:** reproducible differentiated gain → release a scoped local PoC with explicit limits. Equivalent existing method or generic optimization explains gain → credit it and narrow novelty. No advantage → publish/retain a useful diagnostic or negative result; do not claim success through an isolated favorable slice. If the result changes with model scale, explain the interaction and gather the scale-specific evidence in §10 before generalizing.

---

## 9. Attribution, robustness, and failure analysis

### 9.1 Minimum attribution family

Run a small, predeclared set of contrasts sufficient to explain the mechanism, rather than an uncontrolled Cartesian product:

| Question | Contrast | Control that must remain fixed | Interpretation |
|---|---|---|---|
| Does input information matter? | Adaptive B versus best fixed mask; shuffled masks preserving frequencies | Block count, weights, layer scope, teacher/training allowance | Shuffled/fixed parity weakens adaptive value |
| Does the bound penalty matter? | μ>0 versus μ=0 and generic activation sparsity penalty | Data order, trainable tensors, dense-path fraction, loss scales, budget | A gain only versus unadapted weights is not evidence for the penalty |
| Does diagonal preservation matter? | AAC versus ordinary source-group dropping and cost-matched arbitrary/static block controls | Actual work frontier, not identical k | A better simpler support family invalidates the claimed structural advantage |
| Does output-wise control matter? | U top-k, V top-k, B enumeration, actual-error oracle | Norm tables, masks, budget, input samples | Oracle-only success points to controller weakness; B gains with more time may not improve deployment |
| Does fusion explain the result? | Shared optimized kernels and dense shape dispatch for all compatible methods | Full timing boundary and precision | A generic engineering gain should be attributed to the kernel, not algebra |
| Does extra adaptation explain it? | Dense D1, CATS, top-k/weight-aware baselines with equal teacher access | Adaptation tokens and measured expenditure | A generic distillation gain is not AAC novelty |

For fixed-k comparisons, selecting different full weight-bank entries does not change stored parameter count but does change effective computation. Report both; do not infer equal capacity solely from parameter storage.

### 9.2 Boundary and adversarial tests

**Representation:** isotropic versus planted group-concentrated synthetic inputs; uniform source scores; one dominant source; orthogonal and highly correlated block outputs; sign cancellation; near-zero outputs; group rotations/permutations; function-preserving hidden rescalings. Use these to locate failure modes, not to substitute synthetic success for language modeling.

**Controller:** all masks, q=1/2/4, empty diagnostic mask, exact ties, ε=0, very large ε, cap below diagonal cost, conflicting tolerance/cap, norm-table staleness, stale costs, dense-fallback bursts, distribution drift, and near-threshold discontinuities. Check bounds on the actual cast weights used by the kernel.

**Model:** early/middle/late layers; down-only versus all-FFN scope if justified; the same masks with and without adaptation; dense-student versus frozen teacher; rare-token frequencies defined only from training; code, numeric text, long context, and documents with unusually high fallback. Check both average and tail errors. Overlapping-window position schedules must be invariant to the evaluator's tiling.

**Hardware:** batch one versus four; prefill versus decode; changing mask occupancy; mask-union effects across batches; internally padded versus ragged kernels; warm and cold compilation; packed versus canonical weights; fixed-k versus tolerance routing; full fallback through fused versus explicit host dispatch; long thermal sessions; low available RAM; and runtime cache growth. A sparse batch whose mask union touches all weights can lose memory-bandwidth savings despite low per-token MACs.

**Training:** dead/rarely selected blocks; nonfinite gradients; heavy penalty collapsing useful scale; held-out dense quality decreasing; same-seed and independent-seed restarts; checkpoint reload after optimizer update; changed data cursor; weight norm refresh; and OOM recovery. Test whether a resume repeats or skips training tokens.

**Security and assets:** HumanEval is a text likelihood task in the core plan. Do not run generated Python on the host. Do not execute downloaded repository scripts merely because they are cited. Inspect dependencies and licenses before installation; keep the dataset and model content as data, not instructions.

### 9.3 Harness failure classification

Every failed run receives one primary status plus notes:

- `invalid_implementation`: wrong mask, bad shape, stale norms, cache/label bug, incorrect gradient, or inequivalent baseline.
- `invalid_measurement`: lazy graph timing, missing synchronization, different output length, hidden CPU readback, unequal precision, or contaminated split.
- `resource_blocked`: verified RAM/disk/API/asset limitation with no faithful local path under current conditions.
- `insensitive_proxy`: too few independent units, model/task at floor, or small proxy cannot express the hypothesized structure.
- `hypothesis_weakened`: correct and adequately sensitive comparison contradicts a particular mechanism.
- `no_competitive_advantage`: mechanism works but a strong local alternative matches or dominates its frontier.
- `inconclusive`: remaining uncertainty spans materially different outcomes.
- `supported_local_claim`: frozen joint criteria pass, attribution supports the scoped claim, and independent confirmation does not overturn it.

These labels prevent both premature rejection and indefinite escalation. A faster microkernel with worse end-to-end latency is not a deployment win; a valid local error bound with no skipped work is not an efficiency win.

---

## 10. Continuation criteria and evidence-triggered alternatives

| Finding | Recommended decision | Evidence required before increasing complexity |
|---|---|---|
| Adaptive masks do not beat fixed masks, including actual-error oracle | Drop input routing; evaluate a static structured operator or conclude H1 fails locally | A different grouping or larger model must first show a measurable oracle gap, not just a theoretical possibility |
| Oracle helps but B/U are too conservative | Try tighter frozen norm upper bounds; separately evaluate an empirical controller with no certificate claim | Show bound slack, per-layer contribution correlations, and improved control at equal actual error/cost |
| μ=0 matches μ>0 | Remove the regularizer contribution claim | A new penalty must solve a demonstrated failure, and face generic sparsity-regularization controls |
| Generic top-k/CATS/weight-aware sparsity matches AAC | Attribute benefit to known sparsity; stop novelty claim or pursue a distinct engineering result | A demonstrated mask/layout property not already shared by that baseline |
| Native sparse paths fail but static hardware headroom exists | Implement one focused fused/ragged Metal path | Profiling must identify removable launch/layout/load overhead and a correctness-preserving design |
| All reasonable optimized sparse paths lose locally | End the local latency objective; retain a quality/theory result only if independently useful | Escalation requires a measured new shape/precision/workload regime, not a CUDA result from another machine |
| 135M fails while 360M shows stronger source concentration | Continue only at 360M and characterize scale dependence | Paired diagnostic improvement in structure and oracle gap, followed by actual adaptation/latency evidence |
| 360M is positive and 1.7B fits safely | Run a limited frozen and then matched-adaptation scale check | Same causal/evaluation definitions, memory measurements, and no new favorable-only dataset selection |
| Quality survives only with dense fallback on difficult inputs | Report a variable-cost tradeoff, including tail latency and fallback | A meaningful average saving with acceptable tail behavior in the intended workload |
| Quantized dense dominates in speed and memory | No practical local deployment advantage at that quality | Quantized AAC is a new explicitly scoped experiment with refreshed bounds, matched quantized baselines, and measured support |
| The same mechanism is found in prior art | Reclassify as replication/engineering comparison | A genuinely distinct hypothesis and new discriminating test; renaming is insufficient |

A local computational PoC cannot establish commercial value or broad real-world reliability. To support those claims later would require an actual application workload, accepted quality/latency guardrails, operational energy/cost evidence, and broader model/domain replication. Those are limits of the claim, not reasons to avoid the executable local investigation.

---

## 11. Repository, interfaces, configurations, and reproducibility

### 11.1 Minimal repository

```text
aac-poc/
  README.md
  pyproject.toml
  requirements.lock                 # actual resolved dependencies; not invented here
  configs/
    smoke.yaml
    pilot.yaml
    confirm.yaml
    ablations.yaml
  src/aac/
    __init__.py
    __main__.py
    cli.py
    hardware.py
    assets.py
    data.py
    model_adapter.py
    reference.py
    bounds.py
    policies.py
    baselines.py
    train.py
    export.py
    kernels.py
    bench.py
    evaluate.py
    statistics.py
    report.py
  kernels/                          # populated only after its gate
    aac_decode.metal
    score_masks.metal
  tests/
    test_algebra.py
    test_bounds.py
    test_limits.py
    test_gradients.py
    test_causality.py
    test_padding.py
    test_cache.py
    test_nll.py
    test_timing.py
    test_resume.py
  manifests/
    assets.json
    licenses.json
    splits.json
  prereg/
    protocol.md
    amendments.md
  artifacts/
    checkpoints/
    norm_tables/
    cost_tables/
    exports/
  runs/<run_id>/
    resolved_config.yaml
    environment.json
    state.json
    metrics.jsonl
    evaluation_documents.parquet
    latency.csv
    resources.jsonl
    failures.jsonl
    decision.md
  progress.md
  decisions.log.md
  search_log.md
```

Use a package entry point so `python -m aac` works after local installation. Tests may be run using a pinned test framework or the standard library. Include Appendix A in the repository, but do not confuse its CPU smoke script with the complete test suite.

### 11.2 Required typed contracts

The following are proposed interfaces to implement, not claims about existing MLX classes:

```python
@dataclass(frozen=True)
class LinearSpec:
    input_dim: int
    output_dim: int
    input_slices: tuple[tuple[int, int], ...]
    output_slices: tuple[tuple[int, int], ...]
    stored_weight_layout: str   # "out_in" or "in_out", explicitly validated
    dtype: str
    has_bias: bool
    weight_revision: str

@dataclass(frozen=True)
class BudgetSpec:
    mode: str                  # "fixed_k", "tolerance", "hard_cost"
    k: int | None
    tolerance: float | None
    cost_cap: int | None
    cost_unit: str              # "logical_macs", "scheduled_macs", "actions"
    cost_table_revision: str | None

@dataclass
class RouteResult:
    mask: Array                # uint32 [N], device-resident
    envelope: Array            # float32 [N], B for selected mask
    logical_macs: Array        # int64-compatible accounting [N]
    scheduled_cost: Array      # declared cost unit [N]
    tolerance_met: Array       # bool [N]
    status: Array              # integer enum; no hidden host readback
```

Each method implements `prepare(spec, weights, calibration, hardware) -> PreparedMethod`, `route(X, prepared, budget) -> RouteResult`, and `apply(X, prepared, route, backend) -> Y`. `PreparedMethod` is immutable after export and includes all hashes and required packed storage. If MLX has a limitation on a chosen integer dtype on a kernel path, use a supported representation with explicit overflow tests; do not silently overflow MAC counts.

Additionally define `FFNMethod.forward(hidden, positions, cache_context, budget, collect=False)`. CATS and joint FFN baselines use this interface so their opportunity to skip both up and down work is not destroyed by an overly narrow projection abstraction. `cache_context` exposes only causal state needed by the method, never target labels or future token information. FFN methods must not mutate attention caches except through the unchanged model path.

Separate scalar accounting/reference code from device execution. `collect=True` may add diagnostics, but timings use the declared deployment collection level and flush summaries after synchronization outside the measured boundary. Instrumentation overhead is either excluded consistently or reported explicitly.

### 11.3 Configuration rules

Every resolved configuration records at least:

```yaml
experiment:
  phase: exploratory                 # exploratory | confirmatory | replication
  hypothesis_ids: [H1, H3, H5]
  preregistration_hash: null          # required for confirmatory phase
assets:
  model_id: HuggingFaceTB/SmolLM2-360M
  model_revision: RESOLVE_AND_PIN
  tokenizer_revision: RESOLVE_AND_PIN
  dataset_manifest: manifests/splits.json
method:
  name: aac_b
  q: 4
  scope: ffn_down
  eligible_layers: FROM_CALIBRATION_DECISION
  norm_bound: frobenius
  controller: fixed_k
  k: 2
  minimum_sources: 1
  backend: SELECTED_BY_VALIDATED_LOCAL_PROFILE
training:
  trainable_scope: all_ffn_weights
  student_dtype: float32
  teacher_dtype: float16
  seed: 20261013
  sequence_length: 512
  microbatch: 1
  accumulation_steps: FROM_CAPACITY_AND_BATCH_DECISION
  token_budget: FROM_FROZEN_PILOT_DECISION
  optimizer: adamw
  dense_batch_fraction: 0.25
  lambda_kl: FROM_DEVELOPMENT
  beta_ffn: FROM_DEVELOPMENT
  mu_bound: FROM_DEVELOPMENT
measurement:
  inference_dtype: float16
  primary_batch: 1
  primary_prompt_tokens: 512
  primary_decode_tokens: 128
  quality_context: 1024              # secondary full-text guardrail
  quality_stride: 512               # secondary full-text guardrail
  primary_quality_prefix: 512
  primary_quality_horizon: 128
  quality_reference: dense_original_D0
  quality_margin_nats: 0.009950330853168092
  timing_sessions: FROM_PRECISION_DESIGN
```

Uppercase selection markers are deliberately unresolved **input-template values** tied to explicit gates; the command must reject them in a runnable resolved configuration. They are not permission to improvise during confirmation. Keep separate knobs for numerical inference precision, training precision, logical sparsity, executable padding, and statistics. Never store a rounded human-readable number as the only configuration precision.

### 11.4 Experiment-record schemas

`metrics.jsonl` contains one structured record per event:

```json
{"schema_version":1,"run_id":"...","phase":"exploratory","step":0,
 "seed":20261013,"model_revision":"...","method":"aac_b",
 "split_manifest_hash":"...","metric":"nll","value":null,
 "unit":"nats_per_token","aggregation":"sum_loss_div_sum_targets",
 "n_documents":0,"n_target_tokens":0,"status":"not_yet_measured"}
```

Null means missing/not measured, never zero. Successful evaluations include numerator and denominator, not only a rounded mean. Per-document records have document hash, target count, summed NLL, domain labels defined before evaluation, and status. Avoid storing source text in public results when licenses do not permit redistribution.

`latency.csv` includes run/config/export hashes, session ID, paired-block ID, prompt ID, method order, backend, precision, batch/context/output length, warm/cold status, wall seconds, generated tokens, routing mode, fallback count, compilation flags, active/cache memory, and exclusion flag/reason. Token-level timings within a session are nested observations, not new sample IDs.

`state.json` includes git commit and dirty diff hash, dependency lock hash, stage/gate, checkpoint path/hash, optimizer step, consumed-token cursor, RNG state references, budget-ramp state, selected mask-policy revision, norm/cost revisions, and pending next action. `progress.md` states the last successful gate and recovery command. `decisions.log.md` records choices with evidence, alternatives rejected, and interpretation impact.

### 11.5 Checkpointing and recovery

Save weights, optimizer moments, RNG states/counters for every framework used, data-shuffle order/cursor, gradient-accumulation position or only checkpoint at completed accumulation boundaries, budget schedule, precision, and complete configuration. Use atomic rename after a successful write/hash and retain the preceding good checkpoint. Teacher weights are immutable and referenced by hash.

On interruption, reload and verify all hashes before resuming. Recompute or verify norm tables against the restored weights. A test must compare an uninterrupted run with an interrupted/resumed run over the same updates, checking token order, losses, parameter differences, and policy state. Exact bitwise identity may not be supported by every GPU reduction; define and justify numerical tolerance, and report nondeterminism rather than assuming it away.

On OOM, restore the last complete checkpoint, halve microbatch if possible, increase accumulation to preserve effective token batch, and retain the same sample order. Retest numerical behavior. If sequence length, model, trainable scope, or precision changes, this is a methodological amendment, not transparent recovery. During inference, caches are reconstructed from the same prefix unless explicitly serialized and validated. An all-source step does not repair a previously sparse KV cache.

The agent may make routine non-destructive implementation choices—file organization, equivalent vectorization, logging improvements, and correctness-preserving kernel tiles—while recording them. Changes to hypotheses, population, metrics, quality tolerance, baseline access, trainable capacity, or interpretation require a dated amendment. Protected results can never become development data without being relabeled and replaced for confirmation.

---

## 12. Three audits and revisions incorporated into this plan

These are the author's structured challenge passes, not independent peer review. The finite CPU checks in Appendix A were executed; the remaining acceptance tests in this table are proposed local work.

### 12.1 Scientific and prior-art audit

| Defect or challenge found | Revision made | Residual risk / acceptance evidence |
|---|---|---|
| “Another algebra” could simply be masked ordinary multiplication | Explicit fixed-support subalgebra proof and distinction from graph-algebra parameterization; narrow novelty to measured combination | Prior equivalent combinations may still exist; retain prior-art discovery/amendment procedure |
| The trained algebra baseline was confused with the direct four-group rule | Specify separate star and tensor-square supports and counts | Local converted models are not a reproduction of large from-scratch training |
| Sparsity-aware training already teaches skippable representations | Require Q-Sparse-style training, generic penalties, and matched teacher/data controls | Methodological differentiation remains unproved until attribution results exist |
| Weight-aware norms, bounds, and mixed budgets already exist | Add WINA, WiSparse, WAS, and shared strong allocation controls | Local ports cannot establish superiority to inaccessible complete published deployments |
| A down-only baseline set would miss CATS's joint FFN advantage | Require its gate-first up/down execution and an FFN-level interface | Apple kernel quality must be comparable, not merely algorithmic correctness |
| Increasing completion does not imply better actual error | Include cancellation counterexample; restrict monotonicity statement to envelopes along nested masks | Task-level error can still worsen unpredictably and requires guardrails |
| A hard subdense budget conflicts with universal tolerance recovery | Separate controller contracts and return infeasibility explicitly | Real-time latency is still not a strict deterministic resource guarantee |

### 12.2 Experimental-validity audit

| Defect or challenge found | Revision made | Residual risk / acceptance evidence |
|---|---|---|
| Different generated lengths can manufacture throughput differences | Fixed replay and fixed-length free-generation protocols | Replay is a controlled workload, not the whole interactive-use population |
| Extra distillation, capacity, or fine-tuning may explain the gain | Dense-adapted and matched-adaptation baselines; report actual expenditure | Architecturally narrower models cannot be made identical in every resource dimension |
| Public benchmark tokens could leak through row-wise splits | Document-level reconstruction, hashing, protected manifests, development-only thresholding | Pretraining contamination remains unknown; scope claims to comparative retention |
| Repeated timings/tokens could be counted as independent samples | Crossed document/seed design and session-level timing clusters | Very small seed counts can still leave uncertainty; report it |
| A raw bound penalty can be reduced through scale/gauge effects | Frozen teacher scales, dense-student guardrail, function-preserving rescaling tests | Normalization does not prove the regularizer cannot be gamed |
| A power-iteration estimate or stale norm table could be called a certificate | Valid exact-arithmetic upper bounds, revision hashes, numerical-certification labels | Strict floating-point certification remains outside the core claim |
| Test-driven hyperparameter or workload selection could inflate results | Freeze candidate, comparator, workload, sample size, and analysis before unsealing | Amendments require fresh confirmation, which may be resource-limited |
| Dense-prefill evaluation can bypass the sparse decode mechanism entirely | Make primary quality cached continuation scoring under the actual deployed prefill/decode policy | More expensive quality evaluation; local throughput and precision design must account for it |
| Comparing quality only against an already degraded sparse baseline can compound tolerances | Anchor the common quality budget to D0 and report direct peer quality differences | A common quality budget is not a claim of identical quality |
| Average perplexity may hide code/math/rare-token regressions | Predeclared domain/tail guardrails and explicit floor warnings | These guardrails do not prove broad task competence |

### 12.3 Implementation and feasibility audit

| Defect or challenge found | Revision made | Residual risk / acceptance evidence |
|---|---|---|
| Nominal Apple support does not verify needed operations | Hardware/operation smoke gate before dependencies | No M3 execution was performed while writing this plan |
| q=4 groups do not align with 32-wide residual tiles in small assets | Internal-group padding arithmetic, ragged option, q=2 fallback | The best tile/layout is a local empirical question |
| GPU selection might compute both products or synchronize the host | Explicit execution alternatives; count full-mask fused path honestly | Dynamic batching/prefill may still erase benefits |
| Packed block storage might duplicate the dense weight bank | Count both banks and test canonical-layout kernels | A storage disadvantage may remain even if latency improves |
| Training a custom kernel may require an unavailable backward | Native autograd training; inference-only custom kernel initially | Training cost remains dense/reference-like and is not an efficiency result |
| Memory estimates omitted teacher, logits, activations, and optimizer state | Component table plus measured full-update peak gate | OS pressure, graph caches, and fragmentation cannot be predicted exactly |
| Interrupted training might change data or schedule state | Atomic full-state checkpointing and resume equivalence tests | GPU nondeterminism must be measured and bounded, not denied |
| License metadata is inconsistent or incomplete | Asset-specific blocking gate and no blanket redistribution claim | Some optional assets or code ports may need substitution or exclusion |
| An operation failure could be mislabeled scientific failure | Distinct failure statuses and faithful correctness fallback | A blocked hardware claim may remain unresolved on the available stack |

### 12.4 Remaining limitations and what was not done

No local M3 Max measurements, neural training runs, full model numerical equivalence tests, or end-to-end kernels have been executed for this report. The supplied proposal's claimed subjective scores are not evidence and are not retained as research ratings. The present CPU checks test definitions and finite examples only.

Some public repository pages were accessible while other raw paths or direct network downloads from the execution container failed. Selected WINA and MLX-LM source files were inspected through the web tool; no repository was cloned into a local execution environment for this research. The exact Apple installation, complete upstream baseline reproduction, model/dataset license acceptance, and any formal numerical certificate remain gated work.

The literature search is current through the stated cut-off, targeted, and citation-traced, not exhaustive. Patent searches did not produce an inspected, directly relevant patent claim that could support a novelty or clearance conclusion. This is not a patentability or freedom-to-operate opinion. Broader legal conclusions would require a different search and professional analysis.

---

## 13. Search coverage and evidence register

### 13.1 Search process and stopping rationale

The research began with the exact algebra-paper title and the proposal's named methods, then expanded beyond “adaptive algebra” to input-dependent block multiplication, norm-weighted activation pruning, contribution-tail error, sparsity-aware training, multi-budget/slimmable training, mixed sparsity allocation, FFN-to-expert conversion, and Apple-specific masked execution. Citation tracing connected WINA/TEAL to later weight-aware methods, connected trainable sparsity to Q-Sparse/ProSparse/CATS, and connected grouping refinements to MoEfication and DOT-MoE. Recent allocation and attention-sparsity work was checked to avoid overbroad novelty claims.

Consequential source inspection covered full primary methods/appendices for the closest algorithms, selected repository implementation files when behavior affected fidelity, official MLX API documentation, raw official model configurations, and dataset cards. PDF screenshots were inspected for the algebra experiment tables and WAS equations/algorithm. The final material addition was CATS's two-projection savings, which changed the required comparison set. Subsequent overlapping searches were not treated as evidence that no other method exists.

Blind spots include nonindexed or non-English work, unpublished implementations, patents/claims not located by the targeted queries, inaccessible code paths, unverified source licenses, new releases after the cut-off, and hardware behavior not executable in the research environment. The agent's job is not to redo all essential research; it is to resolve the explicitly listed local prerequisites and record any genuinely new competing evidence found during implementation.

### 13.2 How to read the references

Each reference identifies the inspected artifact and its role. Dates are first publication/submission unless a version or venue is specified. A paper result is evidence about its stated conditions, not a new measurement in this plan. Implementation proposals, mathematical deductions, arithmetic estimates, and actual CPU checks are labeled in the body. Stable URLs below make this document independent of the conversation and any chat-only citation system.


### 13.3 Bibliography and inspected artifacts

**R01 — Ilya Koziev and Ivan Oseledets. _Change the Product, Keep the Parameters: Associative Algebra Layers for Transformers._** 26 September 2026, arXiv:2609.32814v1. Full HTML and relevant PDF tables/appendices inspected. Informs the algebra baseline and limits of the published feasibility evidence. [Paper](https://arxiv.org/abs/2609.32814); [inspected version](https://arxiv.org/html/2609.32814v1).

**R02 — Sihan Chen, Dan Zhao, Jongwoo Ko, Colby Banbury, Huiping Zhuang, Luming Liang, and Tianyi Chen. _WINA: Weight Informed Neuron Activation for Accelerating Large Language Model Inference._** 26 May 2025; ICLR 2026. arXiv:2505.19427. Method/theory inspected; informs weight-aware scoring and assumptions, not an unconditional error-optimality claim. [Inspected version](https://arxiv.org/html/2505.19427v2).

**R03 — Microsoft / WINA contributors. _WINA implementation._** Repository state inspected 13 October 2026; exact local commit must be resolved on execution. Root and two implementation files inspected, not executed. Informs calibration, source-metric differences, CUDA dependence, and cost-table adaptation. Applicable source license not fully resolved during this inspection. [Repository](https://github.com/microsoft/wina); [`grab_acts.py`](https://raw.githubusercontent.com/microsoft/wina/main/wina/grab_acts.py); [`greedyopt.py`](https://raw.githubusercontent.com/microsoft/wina/main/wina/greedyopt.py).

**R04 — James Liu, Pragaash Ponnusamy, Tianle Cai, Han Guo, Yoon Kim, and Ben Athiwaratkun. _Training-Free Activation Sparsity in Large Language Models._** August 2024; ICLR 2025. arXiv:2408.14690. Publication record and official implementation description inspected. Informs TEAL's score/allocation baseline. [Paper](https://arxiv.org/abs/2408.14690).

**R05 — FasterDecoding / TEAL contributors. _TEAL official implementation._** Inspected 13 October 2026. README describes distinct supported inference and accuracy paths; repository identifies MIT licensing. No upstream CUDA/Triton benchmark was executed. [Repository and README](https://github.com/FasterDecoding/TEAL).

**R06 — Lei Chen, Yuan Meng, Xiaoyu Zhan, Zhi Wang, and Wenwu Zhu. _WiSparse: Boosting LLM Inference Efficiency with Weight-aware Mixed Activation Sparsity._** 16 February 2026, arXiv:2602.14452v1. Full method/appendix inspected. Informs exponent calibration and mixed allocation; a usable public implementation link was not located in the inspected manuscript. [Inspected paper](https://arxiv.org/html/2602.14452v1).

**R07 — Ming Wang, Miao Zhang, Xuebo Liu, and Liqiang Nie. _Weight-Aware Activation Sparsity with Constrained Bayesian Optimization Scheduling for Large Language Models._** EMNLP, November 2025. DOI:10.18653/v1/2025.emnlp-main.57. Primary text and PDF equation/algorithm pages inspected. Informs L1 scoring, envelope overlap, and scheduling. [Publication](https://aclanthology.org/2025.emnlp-main.57/); [PDF](https://aclanthology.org/2025.emnlp-main.57.pdf).

**R08 — HITSZ-Miao-Group. _WAS implementation._** Inspected 13 October 2026. Repository access confirmed; no full local execution or complete license audit performed. Informs availability and port provenance. [Repository](https://github.com/HITSZ-Miao-Group/WAS).

**R09 — Hongyu Wang, Shuming Ma, Ruiping Wang, and Furu Wei. _Q-Sparse: All Large Language Models can be Fully Sparsely-Activated._** July 2024, arXiv:2407.10969; v2 inspected. Full paper informs training-aware sparsity, STE, pretrained activation handling, and the distinction between intra-block N:M and AAC groups. Numerical normalization fidelity remains explicitly limited. [Inspected paper](https://arxiv.org/html/2407.10969v2).

**R10 — Chenyang Song et al. _ProSparse: Introducing and Enhancing Intrinsic Activation Sparsity within Large Language Models._** First submitted 2024; COLING 2025. arXiv:2402.13516. Full v3 manuscript inspected. Informs the prior-art challenge to sparsity-inducing regularization and the cost of changing activation functions. [Inspected version](https://arxiv.org/html/2402.13516v3).

**R11 — Yuqi Luo et al. _Sparsing Law: Towards Large Language Models with Greater Activation Sparsity._** First submitted 4 November 2024, arXiv:2411.02335. Method inspection used v2, 26 February 2025; current record also checked. Informs contribution-tail context and scale/training caution. The earlier original CETT paper was not fully inspected, so no exact implementation is attributed to it here. [Inspected method version](https://arxiv.org/html/2411.02335v2); [publication record](https://arxiv.org/abs/2411.02335).

**R12 — Zhengyan Zhang, Yankai Lin, Zhiyuan Liu, Peng Li, Maosong Sun, and Jie Zhou. _MoEfication: Transformer Feed-forward Layers are Mixtures of Experts._** Findings of ACL 2022. Primary publication page inspected. Informs the overlap of FFN grouping and selective expert execution. [Publication](https://aclanthology.org/2022.findings-acl.71/); [official repository](https://github.com/thunlp/moefication).

**R13 — Jaeseong Lee, Seung-won Hwang, Wonpyo Park, and Mingi Ji. _Breaking ReLU Barrier: Generalized MoEfication for Dense Pretrained Models._** EMNLP 2024. Primary publication page inspected. Informs the grouping/activation-extension landscape. [Publication](https://aclanthology.org/2024.emnlp-main.563/).

**R14 — Udbhav Bamba, Arnav Chavan, Aryamaan Thakur, Steve Teig, and Deepak Gupta. _DOT-MoE: Differentiable Optimal Transport for MoEfication._** 1 June 2026, arXiv:2606.01666v1. Primary manuscript inspected. Informs the novelty risk for learned channel grouping. [Paper](https://arxiv.org/html/2606.01666v1).

**R15 — Jiahui Yu and Thomas Huang. _Universally Slimmable Networks and Improved Training Techniques._** 2019, arXiv:1903.05134. Official repository/paper description inspected; not reproduced. Informs multi-width training and in-place distillation as existing techniques. [Paper](https://arxiv.org/abs/1903.05134); [repository](https://github.com/JiahuiYu/slimmable_networks).

**R16 — Tri Dao et al. _Monarch: Expressive Structured Matrices for Efficient and Accurate Training._** ICML 2022, PMLR 162. Primary conference publication page inspected. Informs structured-factor alternatives; the local rectangular construction in §5 is explicitly labeled rather than claimed as a faithful port. [Publication](https://proceedings.mlr.press/v162/dao22a.html).

**R17 — Yiheng Yang, Yujie Wang, Chi Ma, Lei Yu, Emmanuele Chersoni, and Chu-Ren Huang. _Sparse Brains are Also Adaptive Brains: Cognitive-Load-Aware Dynamic Activation for LLMs._** 26 February 2025, arXiv:2502.19078. Primary manuscript inspected. Informs context-conditioned allocation and causality checks; published results are not local measurements. [Inspected version](https://arxiv.org/html/2502.19078v1).

**R18 — Oldify / CLADA contributors. _CLADA implementation._** Repository inspected 13 October 2026. No local execution; license and exact causal port remain implementation prerequisites. [Repository](https://github.com/Oldify/CLADA).

**R19 — Jiayi Yuan et al. _BLASST: Dynamic BLocked Attention Sparsity via Softmax Thresholding._** 12 December 2025, arXiv:2512.12087v1. Primary manuscript inspected. Informs attention-side scope and exclusions, not a direct FFN benchmark. [Paper](https://arxiv.org/html/2512.12087v1).

**R20 — Chong You et al. _Spark Transformer: Reactivating Sparsity in FFN and Attention._** First submitted 7 June 2025; NeurIPS 2025, arXiv:2506.06644. Primary manuscript and publication record inspected. Informs competing architecture/sparsity mechanisms, not locally reproduced rates. [Paper](https://arxiv.org/abs/2506.06644).

**R21 — Wenwen Hou, Xinyuan Song, and Shiwei Liu. _ActTail: Global Activation Sparsity in Large Language Models._** 2026, arXiv:2603.12272v1. Primary manuscript and record inspected. Informs global allocation alternatives; not a required full local reproduction. [Paper](https://arxiv.org/html/2603.12272v1).

**R22 — Yuhang Chen et al. _End-to-End Dynamic Sparsity for Resource-Adaptive LLM Inference._** 26 June 2026, arXiv:2606.27743v1. Primary manuscript inspected. Introduces Learning to Allocate on broader resource axes; informs limits of a broad budget-aware novelty claim. [Paper](https://arxiv.org/html/2606.27743v1).

**R23 — Chuhan Yang and Christopher Musco. _Efficient Block Approximate Matrix Multiplication._** ESA 2023, published 30 August 2023. DOI:10.4230/LIPIcs.ESA.2023.103. Primary abstract/publication record and references inspected, not the full proof or implementation. Informs adjacent block-sampling prior art and its locality/importance-cost tradeoff. [Publication](https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.ESA.2023.103).

**R24 — Mark Kurtz et al. _Inducing and Exploiting Activation Sparsity for Fast Inference on Deep Neural Networks._** ICML 2020, PMLR 119. Primary publication page inspected. Informs the historical precedent for learning sparsity that is useful to an inference engine. [Publication](https://proceedings.mlr.press/v119/kurtz20a.html).

**R25 — Apple ML Research / MLX contributors. _MLX installation documentation._** Live documentation inspected 13 October 2026, displayed as version 0.32.3. Informs OS/architecture/Python prerequisites; not an executed compatibility test. [Documentation](https://ml-explore.github.io/mlx/build/html/install.html).

**R26 — MLX contributors. _Lazy Evaluation._** Documentation inspected 13 October 2026. Informs correct evaluation/timing boundaries. [Documentation](https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html).

**R27 — MLX contributors. _Custom Metal Kernels._** Documentation inspected 13 October 2026. Informs custom inference-kernel feasibility and API shape. [Documentation](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html).

**R28 — MLX contributors. _mlx.core.block_masked_mm._** API documentation inspected 13 October 2026. Informs tile-size/mask semantics and alignment constraints; local shapes still require tests. [API](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.block_masked_mm.html).

**R29 — MLX contributors. _mlx.core.gather_mm._** API documentation inspected 13 October 2026. Informs batched matrix-index gathering; the proposed action-bank layout is this plan's design. [API](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.gather_mm.html).

**R30 — MLX contributors. _Metal, device, and memory APIs._** Documentation inspected 13 October 2026. Informs feature-probed device/memory reporting. [Metal documentation](https://ml-explore.github.io/mlx/build/html/python/metal.html); [device information](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.device_info.html).

**R31 — MLX contributors. _mlx.core.synchronize._** API documentation inspected 13 October 2026. Informs stream-aware synchronization. [API](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.synchronize.html).

**R32 — MLX-LM contributors. _MLX LM._** Official repository/README inspected 13 October 2026; repository identifies an MIT license. Informs reusable generation, adaptation, and quantization paths. Exact revisions are to be locked on the laptop, not inferred from the mutable main branch. [Repository](https://github.com/ml-explore/mlx-lm).

**R33 — MLX-LM contributors. _Llama model implementation._** Raw source inspected 13 October 2026, including the MLP and cache/attention integration. Not locally executed. [Inspected source](https://raw.githubusercontent.com/ml-explore/mlx-lm/main/mlx_lm/models/llama.py).

**R34 — HuggingFaceTB. _SmolLM2-135M base model card and configuration._** Retrieved 13 October 2026. Official card and raw configuration inspected. Informs architecture, dimensions, context, and asset selection; memory quantities in §7 are independent calculations. [Model](https://huggingface.co/HuggingFaceTB/SmolLM2-135M); [config](https://huggingface.co/HuggingFaceTB/SmolLM2-135M/raw/main/config.json).

**R35 — HuggingFaceTB. _SmolLM2-360M base model card and configuration._** Retrieved 13 October 2026. Official card explicitly identifies Apache 2.0 licensing; raw config inspected. No local capability claim follows from its sample CUDA code. [Model](https://huggingface.co/HuggingFaceTB/SmolLM2-360M); [config](https://huggingface.co/HuggingFaceTB/SmolLM2-360M/raw/main/config.json).

**R36 — HuggingFaceTB. _SmolLM2-1.7B base model card and configuration._** Retrieved 13 October 2026. Official card and raw config inspected. Informs the optional scale and memory gate. [Model](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B); [config](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B/raw/main/config.json).

**R37 — Salesforce / WikiText dataset maintainers. _WikiText dataset card._** Retrieved 13 October 2026. Configurations and license metadata/prose inspected. License-version inconsistency is retained as a limitation. [Dataset](https://huggingface.co/datasets/Salesforce/wikitext).

**R38 — HuggingFaceTB. _SmolLM-Corpus dataset card._** Retrieved 13 October 2026. Subset and license/provenance notes inspected. Optional only; neither full download nor code-source rights were validated locally. [Dataset](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus).

**R39 — EleutherAI / OpenAI-preprocessed LAMBADA maintainers. _lambada_openai dataset card._** Retrieved 13 October 2026. English data and provenance/license links inspected. Informs narrative guardrail selection; underlying text redistribution is not blanket-authorized by this plan. [Dataset](https://huggingface.co/datasets/EleutherAI/lambada_openai).

**R40 — OpenAI. _HumanEval evaluation data and repository._** 2021 release; README inspected 13 October 2026. Informs code-text data and the prohibition on unsafe host execution in this local plan. [Repository](https://github.com/openai/human-eval).

**R41 — OpenAI / Karl Cobbe et al. _GSM8K; Training Verifiers to Solve Math Word Problems._** 2021; official dataset card inspected 13 October 2026 and identifies MIT licensing. Informs question/answer schema and mathematical-solution guardrails. [Dataset](https://huggingface.co/datasets/openai/gsm8k); [paper identifier](https://arxiv.org/abs/2110.14168).

**R42 — Donghyun Lee, Je-Yong Lee, Genghan Zhang, Mo Tiwari, and Azalia Mirhoseini. _CATS: Contextually-Aware Thresholding for Sparsity in Large Language Models._** First submitted 12 April 2024; v3 inspected. Full primary method inspected; repository link located, repository code not fully audited. Informs the mandatory joint-FFN comparator. [Paper](https://arxiv.org/html/2404.08763v3); [official implementation](https://github.com/ScalingIntelligence/CATS).

**R43 — Xavier Bouthillier et al. _Accounting for Variance in Machine Learning Benchmarks._** 1 March 2021, arXiv:2103.03098. Metadata and abstract inspected. Supports treating benchmark variation as more than a single-seed issue; the detailed statistical protocol here is a proposed design, not a reproduced method from this source. [Paper](https://arxiv.org/abs/2103.03098).

[R01]: https://arxiv.org/html/2609.32814v1
[R02]: https://arxiv.org/html/2505.19427v2
[R03]: https://github.com/microsoft/wina
[R04]: https://arxiv.org/abs/2408.14690
[R05]: https://github.com/FasterDecoding/TEAL
[R06]: https://arxiv.org/html/2602.14452v1
[R07]: https://aclanthology.org/2025.emnlp-main.57/
[R08]: https://github.com/HITSZ-Miao-Group/WAS
[R09]: https://arxiv.org/html/2407.10969v2
[R10]: https://arxiv.org/html/2402.13516v3
[R11]: https://arxiv.org/html/2411.02335v2
[R12]: https://aclanthology.org/2022.findings-acl.71/
[R13]: https://aclanthology.org/2024.emnlp-main.563/
[R14]: https://arxiv.org/html/2606.01666v1
[R15]: https://arxiv.org/abs/1903.05134
[R16]: https://proceedings.mlr.press/v162/dao22a.html
[R17]: https://arxiv.org/html/2502.19078v1
[R18]: https://github.com/Oldify/CLADA
[R19]: https://arxiv.org/html/2512.12087v1
[R20]: https://arxiv.org/abs/2506.06644
[R21]: https://arxiv.org/html/2603.12272v1
[R22]: https://arxiv.org/html/2606.27743v1
[R23]: https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.ESA.2023.103
[R24]: https://proceedings.mlr.press/v119/kurtz20a.html
[R25]: https://ml-explore.github.io/mlx/build/html/install.html
[R26]: https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html
[R27]: https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html
[R28]: https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.block_masked_mm.html
[R29]: https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.gather_mm.html
[R30]: https://ml-explore.github.io/mlx/build/html/python/metal.html
[R31]: https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.synchronize.html
[R32]: https://github.com/ml-explore/mlx-lm
[R33]: https://raw.githubusercontent.com/ml-explore/mlx-lm/main/mlx_lm/models/llama.py
[R34]: https://huggingface.co/HuggingFaceTB/SmolLM2-135M
[R35]: https://huggingface.co/HuggingFaceTB/SmolLM2-360M
[R36]: https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B
[R37]: https://huggingface.co/datasets/Salesforce/wikitext
[R38]: https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus
[R39]: https://huggingface.co/datasets/EleutherAI/lambada_openai
[R40]: https://github.com/openai/human-eval
[R41]: https://huggingface.co/datasets/openai/gsm8k
[R42]: https://arxiv.org/html/2404.08763v3
[R43]: https://arxiv.org/abs/2103.03098


---

## Appendix A. Checks actually executed during this research

The following standalone script was executed successfully in a Linux x86-64 CPU environment using Python 3.13.5 and NumPy 2.3.5. It requires only NumPy and the standard library. Save it as `validate_aac.py` and run `python validate_aac.py`; it writes `validation_results.json` beside itself. It does not import MLX, download a model, train a network, or benchmark Apple hardware.

It checks 2,800 finite mask/bound cases, 2,400 separable-score minimizations, 280 fixed-support closure/associativity cases, 100 two-block star associativity cases, the tensor-square support count, and a nonmonotone actual-error fixture. The maximum full-mask floating-point discrepancy is a reduction-order diagnostic, not proof of bitwise identity. The positive minimum observed bound slack does not establish universal numerical certification.

### A.1 Executed code

```python
"""CPU-only algebra and bound checks; not a neural or M3 Max benchmark."""
import itertools, json, platform, sys
from pathlib import Path
import numpy as np

SEED = 20261013
rng = np.random.default_rng(SEED)

def mask_matrix(q, bits):
    source = ((bits >> np.arange(q)) & 1).astype(bool)
    return np.eye(q, dtype=bool) | source[:, None]

def blocks(W, q):
    d, m = W.shape
    assert d % q == 0 and m % q == 0
    return W.reshape(q, d//q, q, m//q).transpose(0, 2, 1, 3)

def apply(x, W, q, bits):
    b = blocks(W, q)
    z = x.reshape(q, -1)
    return np.concatenate([sum((z[i] @ b[i,j] for i in range(q)
                               if i == j or ((bits >> i) & 1)),
                              np.zeros(b.shape[-1])) for j in range(q)])

def envelopes(x, sigma, bits):
    q = len(sigma)
    a = np.linalg.norm(x.reshape(q, -1), axis=1)[:, None] * sigma
    a = a.copy(); np.fill_diagonal(a, 0)
    omitted = (((bits >> np.arange(q)) & 1) == 0)
    B = np.linalg.norm(a[omitted].sum(axis=0))
    V = np.linalg.norm(a[omitted], axis=1).sum()
    U = a[omitted].sum()
    return float(B), float(V), float(U)

counts = {'bound_masks': 0, 'separable_topk': 0,
          'closure_associativity': 0, 'two_block_star_associativity': 0}
max_dense_discrepancy = 0.0
min_bound_slack = float('inf')
for q in (2, 3, 4):
    for t in range(100):
        W = rng.normal(size=(q*3, q*5))
        x = rng.normal(size=q*3) * np.repeat(np.exp(rng.normal(size=q)), 3)
        b = blocks(W,q)
        sigma = np.linalg.norm(b, axis=(-2,-1))
        vals = {}
        for bits in range(1 << q):
            out = apply(x,W,q,bits)
            err = np.linalg.norm(x @ W - out)
            B,V,U = envelopes(x,sigma,bits)
            tol = 1e-10 * max(1.0,U,np.linalg.norm(x@W))
            assert err <= B + tol and B <= V + tol and V <= U + tol
            if bits != (1<<q)-1:
                min_bound_slack = min(min_bound_slack, B-err)
            vals[bits] = (B,V,U)
            counts['bound_masks'] += 1
        max_dense_discrepancy = max(max_dense_discrepancy,
                                   np.linalg.norm(x@W - apply(x,W,q,(1<<q)-1)))
        norms = np.linalg.norm(x.reshape(q,-1),axis=1)
        off = sigma.copy(); np.fill_diagonal(off,0)
        for index, scores in [(1,norms*np.linalg.norm(off,axis=1)),
                              (2,norms*off.sum(axis=1))]:
            order = np.lexsort((np.arange(q),-scores))
            for k in range(q+1):
                bits = sum(1 << int(i) for i in order[:k])
                best = min(v[index] for s,v in vals.items() if s.bit_count()==k)
                assert abs(vals[bits][index]-best) <= 1e-10*max(1,best)
                counts['separable_topk'] += 1

# Exact integer arithmetic in a bounded range, including noncommuting 2x2 blocks.
for q in (2,3,4):
    for bits in range(1<<q):
        mask = np.kron(mask_matrix(q,bits),np.ones((2,2),dtype=bool))
        for _ in range(10):
            A,B,C = [rng.integers(-2,3,(2*q,2*q))*mask for _ in range(3)]
            assert np.all((A@B)[~mask]==0)
            assert np.array_equal((A@B)@C,A@(B@C))
            counts['closure_associativity'] += 1

def star(A,B):
    a,b,c,d = [A[:2,:2],A[:2,2:],A[2:,:2],A[2:,2:]]
    e,f,g,h = [B[:2,:2],B[:2,2:],B[2:,:2],B[2:,2:]]
    return np.block([[a@e,a@f+b@h],[c@e+d@g,d@h]])
for _ in range(100):
    A,B,C = [rng.integers(-2,3,(4,4)) for _ in range(3)]
    assert np.array_equal(star(star(A,B),C),star(A,star(B,C)))
    counts['two_block_star_associativity'] += 1

# Tensor-square row support: source k allowed if each bit comes from row type or output.
tensor_counts = []
for alpha in range(4):
    n = 0
    for beta in range(4):
        for kappa in range(4):
            n += all(((kappa>>r)&1) in {((alpha>>r)&1),((beta>>r)&1)}
                     for r in range(2))
    tensor_counts.append(n)
assert tensor_counts == [9,9,9,9]

# Actual error need not decrease when a single omitted interaction is restored.
x=np.ones(3); W=np.zeros((3,3)); W[1,0]=1; W[2,0]=-1
errors=[float(np.linalg.norm(x@W-apply(x,W,3,bits))) for bits in (1,3,7)]
assert errors == [0.0,1.0,0.0]

report = {
    'status': 'all_assertions_passed', 'seed': SEED,
    'environment': {'platform': platform.platform(), 'machine': platform.machine(),
                    'python': sys.version.split()[0], 'numpy': np.__version__,
                    'device': 'CPU; no MLX, Metal or M3 Max tests'},
    'counts': counts, 'max_dense_limit_l2_discrepancy': max_dense_discrepancy,
    'min_nonfull_Frobenius_envelope_slack': min_bound_slack,
    'q4_tensor_square_per_row_products': tensor_counts,
    'nonmonotone_actual_error_fixture': errors,
    'limitations': 'Finite tests support implementation checks, not proofs, training quality or speed.'
}
print(json.dumps(report,indent=2))
Path(__file__).with_name('validation_results.json').write_text(json.dumps(report,indent=2)+'\n')
```

### A.2 Recorded output

```json
{
  "status": "all_assertions_passed",
  "seed": 20261013,
  "environment": {
    "platform": "Linux-6.18.44-x86_64-with-glibc2.41",
    "machine": "x86_64",
    "python": "3.13.5",
    "numpy": "2.3.5",
    "device": "CPU; no MLX, Metal or M3 Max tests"
  },
  "counts": {
    "bound_masks": 2800,
    "separable_topk": 2400,
    "closure_associativity": 280,
    "two_block_star_associativity": 100
  },
  "max_dense_limit_l2_discrepancy": 2.043081869203268e-14,
  "min_nonfull_Frobenius_envelope_slack": 0.03721821719505544,
  "q4_tensor_square_per_row_products": [
    9,
    9,
    9,
    9
  ],
  "nonmonotone_actual_error_fixture": [
    0.0,
    1.0,
    0.0
  ],
  "limitations": "Finite tests support implementation checks, not proofs, training quality or speed."
}
```

The algebraic proofs and counterexample in §3 establish the mathematical claims independently of these finite tests. The script primarily protects the planned implementation from support/count/convention errors.

## Appendix B. Final claim checklist for the coding agent

Before declaring a positive result, the report must answer: What exact configuration improved which metric, for what population and workload? Which optimized local baseline was beaten? Were quality, tail behavior, memory, and adaptation/export costs disclosed? Did the μ=0 and fixed-mask controls isolate the contribution? Did actual device execution skip weights rather than merely mask results? Were all output lengths, precisions, caches, and timing boundaries matched? Did independent confirmation survive? Which claims remain unsupported?

A completed negative or inconclusive study is also a valid deliverable. Preserve its code, raw data, failed runs, and decision record. The project succeeds as research by resolving the specified uncertainties honestly; only measured evidence can establish that AAC succeeds as an efficiency method.
