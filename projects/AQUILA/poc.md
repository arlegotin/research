# Continuous hidden-neuron fields with adaptive numerical resolution
## Research assessment and executable M3 Max proof-of-concept plan

**Research date:** 13 October 2026.  
**Working experiment name:** Defect-Guided Continuous Feed-Forward Network (DG-CFF). This is an experiment label, not a claim of invention.  
**Target:** one MacBook Pro with an M3 Max, its actual memory and operating system discovered before execution. No remote accelerator, paid model API, or additional machine is required.  
**Deliverable status:** researched plan with small, executed CPU numerical checks; no trained-model improvement or M3 performance result has been demonstrated.

## 1. Executive assessment

The original proposal is mathematically meaningful, but its central architectural ingredients already exist. Integral Neural Networks explicitly parameterize continuously resizable neural weights; Integral Activation Transform work derives finite representations of integral layers; SplineNets learns continuous weight manifolds with conditional computation. MatFormer, Flextron, and related elastic networks already vary Transformer width, including input-conditioned choices. Accordingly, neither replacing a sum by an integral, using coordinate-generated weights, evaluating at unseen widths, nor routing tokens to different widths is a defensible standalone novelty claim. [R4–R10]

The recommended investigation is narrower:

> **Does a continuous intermediate-neuron coordinate, together with an input-dependent numerical-defect stopping rule, deliver a better measured language-model quality–latency tradeoff than well-trained discrete elastic FFNs and the fastest validated implementation of exactly the same continuous function?**

Retain ordinary finite-dimensional residual states, attention, embeddings, and KV caches. Replace only the intermediate index of the feed-forward network (FFN) by a one-dimensional function on a compact interval. Parameterize that function with piecewise-linear basis functions, evaluate it at nested quadrature nodes, and refine only tokens whose successive approximations differ sufficiently. This is an explicit restriction of the original function-valued Transformer idea, not a demonstration of that entire architecture.

The strongest threat is not another exotic model. It is **compilation**: the proposed finite-basis integral can be evaluated through coefficient-space matrix products, and its piecewise-affine structure permits efficient segment integration. A controller that saves nominal samples can still lose to these fixed computations after projections, routing, synchronization, and cache costs. Such controls are mandatory, not optional ablations.

The core program begins with inexpensive mathematical and hardware tests, then a frozen-layer test and a small natural-language model. It proceeds to training, independent confirmation, or a pretrained-model adaptation only when measured information justifies them. The primary credible local outcome is a result about a specified decoder architecture, dataset, budget, and M3 execution regime—not a claim that integrals generally improve LLMs.

**Current verdict:** a worthwhile, falsifiable empirical question remains; a new general neural-network paradigm has not been established. The strongest plausible contribution is a controlled finding about when numerical-resolution adaptation helps, or why compiler-equivalent and discrete alternatives erase its advantage. The literature search did not establish the absence of prior art for the exact controller combination.

### Evidence labels used throughout

| Label | Meaning |
|---|---|
| **Inspected** | Supported by the primary sources and inspection notes in the bibliography. |
| **Measured now** | Executed in the Linux CPU environment in §5, not on an M3 Max. |
| **Derived** | A mathematical consequence or resource estimate developed here; assumptions are explicit. |
| **Proposed** | An experiment, implementation choice, or acceptance procedure that has not been executed. |
| **Unknown / blocking** | Must be resolved before the dependent claim or workload is allowed. |

## 2. Problem, scope, and hypotheses

### 2.1 Intended benefit and actual bottleneck

The intended beneficiary is an engineer running compact autoregressive language models locally, particularly low-batch decoding. The existing workflow uses fixed-width FFNs at every token, even though some inputs might tolerate cheaper approximations. The desired output is a model with lower total request latency at an acceptable, explicitly measured predictive loss, or lower predictive loss within the same measured latency budget.

Whether FFNs dominate the user's M3 workload is **unknown**. Attention, vocabulary projection, memory traffic, KV-cache management, and kernel launch overhead may dominate instead. Profile before treating FFN cost as available headroom. Variable width is an established solution class, so the research question concerns the incremental value of continuous coordinates and a numerical controller, not the mere existence of wasted computation. [R8–R10]

Use these symbols consistently:

- `d`: residual dimension; `L`: number of decoder blocks; `V`: vocabulary size.
- `K`: number of trainable coefficient knots per FFN field. It controls parameters, not the number of quadrature samples.
- `N`: number of panels in a nested Clenshaw–Curtis rule; `M=N+1`: its actual node count.
- `B`, `T`: batch and token counts; `n=B*T` denotes flattened token states when appropriate.
- `h ∈ R^d`: the current normalized hidden state, using only information available causally at that layer.
- `F(h)`: the mathematical continuous FFN; `Q_N(h)`: a finite numerical approximation.
- `e_N(h)=RMS(Q_N(h)-F_ref(h))`: error against an independently converged reference, not a prediction-loss guarantee.
- `r_N(h)`: the inexpensive successive-approximation defect used by the controller.
- `C`: project-specific training and adaptation cost; `t`: measured inference time; `ℓ`: token-weighted negative log likelihood in nats/token.

### 2.2 Established facts, assumptions, and speculation

**Established:** numerical quadrature turns an integral layer into a weighted finite computation. Continuous kernels, coordinate-dependent weights, resizable layers, finite-basis integral activations, and elastic Transformer widths are prior art. Hardware must still evaluate a finite representation. [R1–R10]

**Assumptions requiring tests:** learned hidden-neuron fields have exploitable regularity; difficulty varies across tokens; a cheap defect identifies that variation; saved matrix work exceeds routing overhead; and a continuous parameterization provides an advantage unavailable to a discrete elastic model of comparable project cost.

**Not assumed:** larger numerical resolution always improves language loss; a zero defect means a small true error; the coordinate axis has a natural semantic ordering; quadrature samples correspond to independent learned neurons; or lower FLOPs imply lower M3 latency.

**Speculative extensions excluded from the core claim:** function-valued attention and residual streams, adaptive spatial coordinates inside every Transformer operation, full-scale foundation-model pretraining, learned nonuniform meshes, SIREN-generated kernels, certified model-level error bounds, and energy savings on unmeasured deployment hardware. An extension is admitted only through the evidence-triggered branches in §12.

### 2.3 Hypothesis register

| ID | Testable hypothesis and predicted observation | Support and alternative explanation | What weakens or falsifies it locally |
|---|---|---|---|
| H1: usable continuum | For trained weights, off-grid evaluations converge toward the same FFN function, and the model remains useful under independently accurate integration. | Integral formulations and resizable layers support representability, not task advantage. Apparent gains could be fixed-grid overfitting. [R4–R7] | Training-grid loss improves while independent integration changes outputs substantially or worsens language loss; no affordable resolution approximates the stated integral. |
| H2: useful heterogeneity | At similar error or task quality, different tokens require substantially different numerical work. | Conditional computation motivates heterogeneity, but does not establish it for this integrand. [R7, R9] | An offline oracle cannot improve the cost–error frontier beyond timing uncertainty; almost every token needs the same or maximum level. |
| H3: useful defect | Successive quadrature differences predict when extra work changes useful outputs, better than a cheap discrete or learned controller. | Embedded quadrature is established numerical methodology. It is not a certificate here. [R22] | False-safe decisions dominate, defect has little predictive value, or one-shot routing performs as well with less overhead. |
| H4: continuum-specific value | Continuous fields beat discrete elastic FFNs when both receive the same controller opportunities, tuning, and project budget. | Plausible sharing between nearby coordinates; no direct evidence found for this exact LM claim. | The advantage disappears with a discrete defect controller, stronger MatFormer training, matched capacity, or generic distillation. |
| H5: end-to-end value | Quality–latency gains survive the fastest same-function implementation, attention, vocabulary projection, routing, and cache costs. | Entirely unmeasured on the target. Compiler equivalence is a strong counterargument. | Candidate lies on or behind the controlled baseline frontier, or a mean gain conceals unacceptable latency-tail or memory regressions. |
| H6: transfer across scale | A mechanism-level signal persists at a second locally meaningful scale or in a faithful pretrained FFN adaptation. | Possible but unestablished; small models can be launch-bound. | A signal only exists in hand-designed integrands or one model/configuration and disappears at a scale where its claimed bottleneck is actually present. |

These are separate hypotheses. H1 can hold while H5 fails. Failure of a poorly trained proxy does not settle H6, but a speculative scale effect is not permission to repeat negative experiments indefinitely.

### 2.4 Strongest case for and against

**For:** a compact field could share information between neighboring neuron coordinates, support smooth changes in compute, and expose numerical difficulty that a fixed-width FFN cannot exploit directly. If a small subset of tokens needs fine resolution, and the M3 spends enough time in large FFN projections, nested reuse could lower request cost. A controller based on actual approximation change may transfer across budgets better than a router trained for one discrete menu.

**Against:** the coordinate ordering is invented; interpolation can reduce useful independence of neurons; trained activations may have sharp or oscillatory features; fixed quadrature can be exploited during optimization; and numerical accuracy need not align with language accuracy. Finite-basis factorization may remove the expensive dependence on sample count. Discrete elastic networks already supply variable compute, and their static or one-shot execution can be friendlier to GPU kernels. These are substantive mechanisms for failure, not reasons to avoid testing.

### 2.5 Headroom calculation

**Derived.** If the measured baseline spends fraction `f` of its request time in the FFN work the method can actually remove, and the remaining FFN fraction is `r`, then an optimistic speedup is

`S = 1 / [(1-f) + f*r + ρ]`,

where `ρ` is added routing, numerical bookkeeping, and cache overhead divided by baseline request time. This is only an accounting model: memory traffic and kernel fusion can change both terms.

For illustration, not a hardware measurement: with `f=0.50`, `r=0.50`, and `ρ=0.05`, the speedup is `1.25×`, not `2×`. With `f=0.20` and `ρ=0.15`, the same nominal FFN reduction makes the request slower. Measure `f`, the cheapest attainable `r`, and `ρ` before large training. A masked implementation that executes every level has no such savings.

## 3. Research landscape and novelty assessment

### 3.1 Search scope and stopping point

The investigation searched original and adjacent terms: neural/integral operators; continuously resizable and integral neural networks; integral activation transforms; spline weight manifolds; implicit weight representations; elastic, slimmable, nested, and adaptive-width Transformers; conditional/tree FFNs; adaptive and learned quadrature; analytical activation integration; mobile reparameterization; Apple MPS operation support; dataset/model licenses; and preliminary patent terminology.

Backward tracing reached operator approximation, spline conditional computation, slimmable networks, and classical adaptive quadrature. Forward tracing included AWN's February 2026 revision, 2026 inference-aware FFF work, mobile integral reparameterization published online in 2025, and a May 2026 neural-PDE quadrature preprint. The final targeted refinement searches predominantly returned the already identified architectural equivalents plus adjacent numerical-loss integration work. That is sufficient to select the experiment; it is not proof that all prior art has been found.

Public primary repositories, official documentation, paper HTML, abstracts, and official conference slides were inspected. Important limitations are recorded per reference. In particular, the full INN CVPR paper was not retrieved successfully; its official slides and implementation were inspected instead. No broad legal patentability or freedom-to-operate analysis was performed.

### 3.2 Closest alternatives

| Prior art | Problem, mechanism, and assumptions | Evidence / compute / availability | Exact overlap and implication |
|---|---|---|---|
| Neural operators, FNO, DeepONet [R1–R3] | Learn mappings between functions, often sampled physical fields; structured kernels or branch/trunk representations. | Primarily operator/PDE settings. Function-space attention also exists [R29]. Their task semantics and resolution axes differ from a latent FFN axis. | Establish the integral viewpoint. A PDE resolution demonstration would not establish LM hidden-width value. |
| Integral Neural Networks / TorchIntegral [R4, R5] | Interpolate trainable weights into a continuous representation and discretize at different widths; conversion from discrete models is supported. | Official slides include both successful converted initialization and weaker from-scratch examples. Public Apache-2.0 implementation; inspected interpolation and quadrature code. | Directly anticipates continuous width. Mandatory closest architectural control. Neither arbitrary width nor coordinate-generated weights is new. |
| Integral Activation Transform [R6] | Finite-basis continuous kernels reduce to finite-dimensional transforms with integral activations; selected cases allow analytical integration. | Full preprint and publication metadata inspected. Not a local autoregressive throughput result. | Anticipates the finite-basis reduction. Prevents claiming an infinite capacity advantage or a new analytical-integral principle. |
| SplineNets [R7] | Continuous spline manifolds of weights with input/path-conditioned selection. | CNN experiments; paper discusses costly per-sample conditional computation and a linearity-based implementation workaround. | Continuous coordinates plus conditional computation already coexist. The new experiment must distinguish numerical-defect control and practical execution. |
| MatFormer v2 [R8] | Nested FFN prefixes, sampled training granularity, and layer-wise MixMatch configurations. | Transformer evidence at much larger scales; paper-based local implementation is straightforward. v2 samples one granularity per optimization step. | Strong discrete elastic baseline. Do not weaken it by using a mismatched older training description. |
| Flextron [R9] | Elastic attention/MLP widths with input- and budget-conditioned routers, importance ordering, and surrogate-assisted training. | Full method inspected; published large-model setting differs from the core scratch model. FFN-only local adaptation is feasible but is not a full reproduction. | Token-adaptive width is already known. Learned one-shot routing is a compulsory competing explanation. |
| AWN, latest inspected v5 [R10, R11] | Learns effective width during training using importance decay and variational objectives; permits deployment truncation. | ICLR 2026; official dynamic Transformer code is encoder–decoder, not this decoder-only backbone. Code license restricts use and redistribution. | Relevant learned-width competitor, not the same per-token defect mechanism. License and faithful adaptation must be handled explicitly. |
| Universally Slimmable Networks / DynaBERT [R12, R13] | Width/depth flexibility, ordering, sandwich training, and distillation. | Original vision/encoder settings. Reusable training ideas, not interchangeable LM benchmark numbers. | Generic improvements must be offered to both continuous and discrete methods. |
| Fast Feedforward Networks / INSTANT [R14, R15] | Tree-based conditional FFN computation; INSTANT aligns training with hardened inference. | FFF official code is MIT; INSTANT repository and abstract inspected, full paper/license access incomplete. Embedded/vision evidence does not establish LM wins. | Strong conditional-computation screen, especially when routing overhead is the claimed advantage. |
| Reparameterizable integral mobile networks [R16] | Integral representations coupled to inference-oriented reparameterization. | Abstract inspected; mobile architecture, full implementation unverified. | Additional reason not to claim compilation or mobile integral acceleration is itself novel. |
| Adaptive / learned quadrature [R22–R24] | Error indicators, adaptive rules, and learned/geometric integration for numerical functionals. | Different integrals and tasks; no direct M3 LM evidence. | A successive-rule controller is established methodology; quadrature overfitting is a known concern in adjacent neural optimization. |
| Gemma 3n [R18] | A deployed model family incorporating elastic architectural ideas. | Official developer documentation, not a locally reproducible training comparison. | Elasticity is deployed engineering, not an untouched research space. Published numbers are context only. |

### 3.3 What may be reused

Use PyTorch's dense layers, autograd, attention, optimizers, and MPS instrumentation after operation-specific validation. Reuse established tokenizer/data libraries, official assets, and permissible FFF components. TorchIntegral's public code is valuable for a closest-method port and for understanding conversion; do not inherit an entire old vision training stack merely to interpolate two matrices. Preserve attribution and license notices when copying permitted code. [R5, R14, R19–R21]

Reuse algorithms, not unsupported claims. A local FFN-only Flextron adaptation must be named as such. A clean project implementation of a paper is not evidence that the original large-scale result has been reproduced. Restricted AWN code must not be silently incorporated into a permissively licensed project. [R9–R11]

### 3.4 Precise candidate contribution and invalidation test

The potential methodological difference is **using a nested numerical defect of a continuous FFN to allocate per-token inference work, with reusable evaluations and a validated same-function compiler baseline**. Neither nested integration nor token routing is new. Its exact combination may already exist in unlocated work, and an untested combination is not automatically a contribution.

A credible contribution would require all of the following: a useful learned continuous function; predictive token-dependent numerical difficulty; a controller that exploits it; a result that survives a discrete version of the same controller; and an end-to-end benefit over the best fixed implementation of the same integral. The experiment in §8 is designed to separate those steps.

The novelty claim is invalidated if a prior method already has the same function parameterization, defect-controlled per-input resolution, and computational reuse, even under another name. Practical novelty is weakened if the only difference is implementation language, M3 porting, a dataset change, or a new hyperparameter. Such outcomes may still support an engineering contribution or useful empirical replication; label them correctly.

## 4. Selected mathematical mechanism and implementation

### 4.1 Shared backbone

Implement a decoder-only, pre-normalized Transformer with tied token embeddings and output weights. Use causal self-attention, residual additions, and LayerNorm with epsilon `1e-5`. Core activation is exact Gaussian GELU (`approximate="none"`). Use no dropout initially, identically across methods; include common dropout tuning if overfitting is observed. Initial context length is 256 tokens.

Use RoPE with head dimension 64 and base 10,000 in the scratch backbone. For pair index `j`, rotate Q and K by position times `10000^(-2j/d_head)`. KV entries contain already-rotated keys. All alternatives share positions, attention, embeddings, normalization, tokenizer, loss masking, and cache implementation. Dense SwiGLU is also a strong static baseline, using an intermediate size near `8d/3` to match the two-matrix GELU FFN's parameter count; report the actual rounded size and count. [R17]

Initial model configurations are in §9. They are calibration starting points, not ceilings. A change in backbone is a new experimental stratum, not a quiet correction to a failed comparison.

### 4.2 Continuous FFN

For each layer define

\[
F_\theta(h)=g\int_0^1 v_\theta(s)\,G(a_\theta(s)^\top h+b_\theta(s))\,ds+c,
\qquad G(z)=z\Phi(z).
\]

Here `a(s), v(s) ∈ R^d`, `b(s) ∈ R`, `c ∈ R^d`, and `g=K` is fixed for a given model. There is no activation after the integral; the usual residual addition follows it.

Let `t_k=k/(K-1)` for `k=0,...,K-1`. Use continuous piecewise-linear hat functions `ψ_k`, nonnegative and summing to one on `[0,1]`:

\[
a(s)=\sum_k\psi_k(s)C_{A,k},\quad
v(s)=\sum_k\psi_k(s)C_{V,k},\quad
b(s)=\sum_k\psi_k(s)C_{b,k}.
\]

Trainable arrays are `C_A[K,d]`, `C_V[K,d]`, `C_b[K]`, `c[d]`. Initially set `K=4d`; compare other values only as recorded development configurations. This parameter count is close to the ordinary GELU FFN's, rather than concealing extra capacity in a coordinate-generating network.

For coordinates `s`, compute `u=s*(K-1)`, `left=min(floor(u),K-2)`, `fraction=u-left`. The field value is `(1-fraction)*C[left] + fraction*C[left+1]`. At `s=1`, this returns the final knot exactly. Coordinates outside `[0,1]` are errors, not silently clamped. Internal arithmetic uses FP32 on MPS and FP64 for the independent CPU reference.

Initialization: `C_A` uses zero-mean variance `1/d`, `C_V` variance `1/K`, and biases start at zero. Apply the same residual-output scaling `1/sqrt(2L)` to candidate and ordinary FFN output weights. Because interpolation changes covariance, measure initial output RMS on a saved synthetic normal-input batch and apply a **single, resolution-independent output rescaling** to match the baseline RMS. Save that factor. Do not tune initialization on test language data and never make `g` depend on quadrature resolution.

### 4.3 Nested quadrature

Use even power-of-two panel counts `N ≥ 2` with nodes

\[
s_j=\frac{1-\cos(\pi j/N)}{2},\quad j=0,\ldots,N.
\]

These include both endpoints and satisfy `nodes(2N)[::2] == nodes(N)` within floating-point precision. The positive Clenshaw–Curtis weights integrate on `[0,1]`, sum to one, and have endpoint weight `1/[2(N²-1)]`. Appendix A provides the exact reference implementation used in the numerical checks.

For `d=256, K=1024`, the initial panel ladder is `128,256,512,1024`, hence node counts `129,257,513,1025`. For another `K`, begin at the largest power of two not exceeding `K/8`, and double through the first power of two at least `K`. Actual convergence against §4.5 determines whether this ladder must be extended or rejected. These are starting quadrature levels, not an assertion that `M≈K` is accurate.

At a refinement, reuse old field projections and activations, evaluate only new odd-indexed nodes, and apply the **new weights to all nodes**. Old Clenshaw–Curtis weights do not simply halve. Avoid recomputing coordinate interpolation for unchanged weights during inference: cache each required sampled matrix or its factorized representation, and charge that cache in memory and preparation cost.

Also test a uniform composite nested rule and the segment-aware implementation. Clenshaw–Curtis is a transparent default, not assumed optimal for piecewise-linear fields. Select the fixed quadrature family using development cost–accuracy curves before confirmatory training. A family change must preserve the stated integral and be recorded.

### 4.4 Two algebraically equivalent implementations

For flattened states `H[n,d]`, let `Ψ[M,K]` contain basis values and `w[M]` the quadrature weights. The straightforward sampled implementation is

\[
Q_N(H)=g\{G[H(\Psi C_A)^\top+\Psi C_b]\odot w\}(\Psi C_V)+c.
\]

The factorized implementation is

\[
Z=HC_A^\top+C_b,\qquad
Q_N(H)=g\{[G(Z\Psi^\top)\odot w]\Psi\}C_V+c.
\]

**Derived and numerically checked:** these are the same function, including gradients in exact arithmetic. The number of trainable coefficients is unchanged. Do not claim the second representation as a new discovery relative to finite-basis integral-transform prior art. [R6]

The production factorized path must exploit the two nonzero hat coefficients per node. Compute interpolated `Z` by gathers, and accumulate back to knot coefficients with an efficient validated reduction. A dense `Ψ` multiplication costing `O(nMK)` is a correctness oracle, not an acceptable strongest compiler baseline. Test MPS scatter accumulation against deterministic segment reductions; choose by correctness and measured time. The segment implementation below avoids scatter altogether.

The sampled path has approximately `2ndM` multiply-accumulates. The local-basis path has approximately `2ndK + O(nM)` plus activation work. For `M<K`, the sampled path can still be faster; for large `M`, coefficient-space evaluation may dominate. Benchmark both for every relevant shape and precision. Freeze a development-selected dispatch table keyed by layer shape, batch regime, and resolution; apply such dispatch optimization to all relevant competitors.

### 4.5 Independent segment integration and analytical control

Within interval `[t_k,t_{k+1}]`, write `s=t_k+Δu`, `Δ=1/(K-1)`, and

`p(u)=p0+(p1-p0)u`, where `p_k=h·C_A[k]+C_b[k]`.

The contribution is determined by two scalar moments:

\[
J_0=\int_0^1G(p(u))du,\qquad J_1=\int_0^1uG(p(u))du.
\]

Accumulate `Δ*(J0-J1)` into knot `k` and `Δ*J1` into knot `k+1`, then multiply the resulting knot coefficients by `g*C_V` and add `c`. Implement adjacent contributions by shifted additions; no node-sized vector of dimension `d` and no atomic scatter is necessary.

**Reference and production competitor:** evaluate each segment with Gauss–Legendre nodes, initially orders 8 and 16, doubling until independent higher-order agreement is adequate for the output/gradient tolerance. Use CPU FP64 and independent analytical checks for the truth reference. The fastest stable lower-order MPS segment rule can be a production baseline even when the CPU truth uses a richer rule. Streaming segment blocks controls activation memory without changing the function.

**Analytical GELU control, derived here but not claimed novel:** with `φ` and `Φ` the standard normal density and distribution,

\[
A_0(z)=\tfrac12[(z^2-1)\Phi(z)+z\phi(z)],\qquad
A_1(z)=\tfrac13[z^3\Phi(z)+(z^2+2)\phi(z)],
\]

satisfy `A0'=G` and `A1'=zG`. For `δ=p1-p0 ≠ 0`,

\[
J_0=\frac{A_0(p_1)-A_0(p_0)}{\delta},\quad
J_1=\frac{A_1(p_1)-A_1(p_0)-p_0[A_0(p_1)-A_0(p_0)]}{\delta^2}.
\]

At `δ=0`, `J0=G(p0)` and `J1=G(p0)/2`. Near-zero slopes and extreme tails can cause cancellation in FP32. Do not deploy the raw quotients merely because the symbolic formula is exact. Use a validated Taylor/erfc formulation or a segment-Gauss fallback; until such a path is validated, the numerical segment rule is the production control. The executed check used 16-point Gauss fallback for `|δ|<0.1`; that threshold is **not validated for M3 FP32 deployment**.

An optional ReLU variant permits exact root-split polynomial integration, which is a useful independent control and has prior-art overlap with analytical integral activations. It is not a license to change the core GELU task after a negative result. Any activation change creates a separately identified experiment. [R6]

### 4.6 Defect controller

For layer `l` and current normalized state `h`, use

\[
r_{l,N}(h)=\frac{\operatorname{RMS}[Q_{2N}(h)-Q_N(h)]}{10^{-6}+\operatorname{RMS}(h)}.
\]

Calibrate `s_l` as the median positive defect at the first decision level on a fixed development-calibration sample, with a floor of `1e-6`; record it. The threshold is `τ_l=τ*s_l`, with one global `τ` selected for the frozen latency budget. This scaling is fixed at inference and adds no runtime future information.

At every decision return the **finer** result when `r≤τ_l`. A token cannot stop after only the initial coarse rule. If it reaches the maximum level without satisfying the threshold, return the finest result and set `exhausted=true`. Any NaN/Inf is an invalid run, not an automatic acceptance. Token-level decisions must depend only on current-layer states and computed approximations, never continuation targets or future-layer states.

```python
# Specification pseudocode, not a pre-existing package API.
def adaptive_ffn(H, layer, ladder, tau_l):
    # H: [n_tokens, d]; output order must be preserved.
    active = all_token_indices(H)
    cache = layer.evaluate_nodes(H, ladder[0])
    coarse = layer.reduce(cache, ladder[0])
    output = empty_like(H)
    trace = new_trace()
    for N in ladder[1:]:
        # Only active rows and new nodes are evaluated.
        cache = layer.refine_nodes(H[active], cache, N)
        fine = layer.reduce(cache, N)  # includes reweighted old nodes
        defect = rms(fine - coarse, axis=-1) / (1e-6 + rms(H[active], axis=-1))
        accept = defect <= tau_l
        if N == ladder[-1]:
            trace.mark_exhausted(active[~accept])
            accept = all_true_like(accept)
        output[active[accept]] = fine[accept]
        trace.record_actual_work(cache, active, accept)
        active, cache, coarse = select_unaccepted(active, cache, fine, accept)
        if active_is_empty_with_accounted_sync(active):
            break
    return output, trace
```

This pseudocode intentionally exposes an implementation cost: determining active rows and exiting a loop can synchronize a GPU with its host. A Python implementation must measure those synchronizations. Do not hide them behind an idealized FLOP estimate. An evaluate-all-and-mask implementation is a correctness control only.

### 4.7 Why the defect is not an error certificate

A triangle of height one supported on `[0.59,0.61]`, with its peak at `0.60`, integrates to `0.01`. Both the `N=4` and `N=8` rules sample it as zero. Their defect is therefore zero although both are wrong. The function is representable using 101 uniform piecewise-linear knots; a constant positive GELU activation and a scaled output field embed it in the selected FFN class.

This counterexample was measured now and is also directly verifiable analytically. It rules out a universal guarantee based solely on two nested estimates. The plan instead measures false-safe decisions against an independent reference, tests adversarial spikes and cancellations, and limits any claim to the evaluated distribution. A rigorously certified variant would require additional smoothness bounds and a separate cost analysis.

### 4.8 Shared method interface and invariants

All FFNs implement the following **proposed project contract**:

```python
@dataclass
class FFNResult:
    y: Tensor                  # same leading dimensions as h, final dimension d
    trace: dict                # counters; no target labels

class FFNMethod(nn.Module):
    def forward(self, h, *, mode: str, budget: float | None) -> FFNResult: ...
    def prepare_inference(self, checkpoint_sha256: str, device, dtype) -> dict: ...
    def invalidate_cache(self) -> None: ...
    def parameter_manifest(self) -> dict: ...
```

Modes are `train_sampled`, `static`, `defect`, `one_shot`, and `reference`; unsupported modes raise an error. The training mode chooses one valid resolution/configuration according to the frozen recipe. `budget` is a normalized conditioning value or a fixed policy identifier, **not an uncharged inference oracle**. Cache validity includes parameter revision, knots, nodes, dtype, device, and activation.

Required trace counters include new-node evaluations, node terms reduced, coefficient projections, refinement count, active-token counts per level, groups launched, host synchronizations, and cap exhaustion. FLOPs and counters are diagnostics; stopwatch measurements establish performance.

Invariants: quadrature mass one; resolution-independent parameters and output gain; exact endpoint behavior; causal routing; no evaluation of accepted-token refinements; cache invalidation after parameter updates; identical padding and loss masks; and equivalent full-sequence versus incremental logits within validated numerical tolerance.

## 5. Validation performed during this investigation

These checks resolve algebraic and numerical ambiguities only. They are **not** M3 benchmarks, training experiments, or evidence of improved language modeling.

**Measured environment:** Python 3.13.5; Linux 6.18.44, x86-64; NumPy 2.3.5; SciPy 1.17.0; PyTorch 2.10.0+cpu; MPS unavailable; random seed 1013; one PyTorch CPU thread. Calculations used FP64. The original executable test is reproduced in Appendix B so this handoff does not depend on intermediate research files.

| Check actually run | Conditions | Observed result | What it establishes / does not establish |
|---|---|---|---|
| Quadrature normalization, polynomial moments, and nesting | Panels `2,4,8,16,32,64,128`; monomials of degrees `0..min(N,10)` | Positive weights; maximum mass error `2.22e-16`; maximum tested moment error `1.11e-16`; nesting error zero | Correct reference construction for tested rules, not convergence for arbitrary learned integrands. |
| Sampled versus factorized FFN | Batch 5, `d=8`, `K=17`, 65 nodes, exact GELU | Maximum output discrepancy `4.44e-16`; maximum gradient discrepancy `6.11e-16`; autograd gradcheck passed | Algebraic equivalence and a small numerical implementation check; no hardware speed comparison. |
| Analytical GELU segment moments | 1,000 endpoint pairs in `[-5,5]`, including near-equal slopes; 16-point Gauss fallback for `|δ|<0.1`; 128-point reference | Maximum `J0` error `1.78e-14`; maximum `J1` error `5.31e-13` | Formula is consistent on this bounded CPU test set; FP32 stability, tails, and its production gradient remain unverified. |
| ReLU root-split moments | 1,000 intervals, integration over positive segment portions | Maximum `J0` error `1.33e-15`; maximum `J1` error `5.33e-15` | A second independently checkable limiting case, not evidence that ReLU is a better LM activation. |
| Defect failure construction | Triangle described in §4.7, `N=4` and `8` | Both quadratures and defect equal zero; true integral `0.01` | A counterexample to universal defect certification, within the proposed field class. |

No model checkpoints, tokenized corpus, M3 traces, timing distributions, energy readings, or language benchmark results were produced during this planning assignment. Any such result mentioned below is a requested future artifact, not a completed experiment.

## 6. Baselines and training design

### 6.1 Comparison tiers

The primary claim is restricted to the shared decoder backbone and **FFN-only compute adaptation**. Within that scope, the compulsory direct comparisons are B0–B5 below. B6 is a relevance screen for broader conditional computation; B7 is an additional learned-width screen with a licensing/fidelity gate. Large published Flextron, MatFormer, and Gemma results are context, never pasted into a local head-to-head table.

| ID | Locally executable specification | Required tuning and fidelity checks | Role |
|---|---|---|---|
| B0: independent dense models | Ordinary GELU FFNs at widths near `K/8,K/4,K/2,K`, plus a parameter-matched SwiGLU variant. Train independent narrow models, not only prefixes of a large model. | Same backbone, tokenizer, data access, optimizer search, loss evaluation, precision, and project-cost accounting. Width rounding and real parameters recorded. | Strong simple quality–latency frontier. |
| B1: discrete elastic / MatFormer v2 | Nested prefixes of ordinary FFN input rows, biases, and output columns. Sample one global granularity per optimizer step; evaluate homogeneous widths and per-layer MixMatch. | Implement the inspected v2 recipe rather than averaging four full passes and calling it v2. Tune uniform sampling versus a finest-width emphasis. Include sandwich/distillation only as separately costed alternatives. | Strong discrete elasticity without continuous interpolation. [R8] |
| B2: discrete defect control | Use B1 weights and the same successive-output defect formula, layer scaling, refinement reuse, and stopping convention as DG-CFF. Prefix contributions can be reused exactly. | The ordinary prefix sum has no artificial `1/width` normalization. A width-dependent rescaling must be a separately trained baseline, not a silent change. Optimize cumulative partial sums. | Isolates the contribution of continuous coordinates from the contribution of a numerical-looking stopping rule. |
| B3: learned one-shot routing | FFN-only Flextron-style adaptive-width model on the common backbone, using an input/budget router and importance-aware preparation. Also apply the same one-shot router to the continuous model. | Follow the explicit local protocol below; label it an FFN-only adaptation, not full Flextron reproduction. Keep router-label generation and repair cost in the budget. | Stronger controller alternative and test of whether sequential defects are worth their overhead. [R9] |
| B4: static integral / INN-like | Same continuous coefficient fields, no token controller; fixed quadrature levels. Also use a coordinate-aware interpolation port of the closest TorchIntegral weight-resizing mechanism. Screen both scratch and converted initialization. | Match coordinate convention, gain, initialization, and interpolation. Check conversion error and count repair training. Never pass a nonuniform grid to a size-only interpolation path. | Closest architectural prior art and continuum-without-adaptation control. [R4, R5] |
| B5: same-function compiler | Sampled dense, sparse-basis factorized, segment-Gauss, and any numerically validated analytical GELU implementation of the **same frozen continuous weights**. | Output and gradient parity; independent integral accuracy; best dispatch by actual shape. Preparation and additional cached weights are counted. | Strongest explanation that the method merely computes an existing function inefficiently. |
| B6: tree-conditional FFN | Screen official MIT FFF on the common residual interface. Select tree depth and leaf width to span the same parameter/latency region; include hardened inference and a training–inference alignment treatment motivated by INSTANT. | Standard FFF and the added alignment treatment are separately named. Do not claim full INSTANT reproduction without its full protocol and license. Training all branches and inference one path must be timed honestly. | Broader conditional-computation competitor. [R14, R15] |
| B7: learned architectural width | AWN using its inspected latest paper and permitted official implementation, or a separately documented paper-based adaptation when permissible. | Resolve restricted license first. Encoder–decoder code is not a drop-in decoder LM. Preserve the importance-decay/variational mechanism and validate the adaptation before treating it as AWN evidence. | Contemporary width-learning screen; not required to prove a same-function compiler result, but relevant to broader width-efficiency claims. [R10, R11] |

A method cannot be excluded because its first pilot is inconveniently strong. Select all compulsory closest controls and all development-nondominated competitors for confirmatory replication. Screening a clearly dominated method may end after a valid, sufficiently sensitive pilot; retain that result and its tuning record. If B6/B7 cannot be used faithfully, restrict the final language to the completed comparison set rather than claiming superiority over all contemporary adaptive networks.

### 6.2 Exact local one-shot routing protocol

Use this executable protocol for B3 and for its continuous counterpart. It is a **project-defined controlled adaptation informed by Flextron**, not a claim to reproduce all of its surrogate training and elastic-attention machinery. [R9]

1. Prepare elastic backbones using the same eligible training data and width/resolution choices. For a converted discrete checkpoint, optionally order neurons by a saved development importance estimate, e.g. mean absolute activation times output-column norm; then repair all affected alternatives using identical access to data. Report that heuristic as the project's ordering choice.
2. Freeze each backbone. On development-training articles, capture current-layer states and evaluate every allowed local width/resolution against the layer's richest validated output. Define local target error `E_j(h)=RMS(y_j-y_ref)^2/[1e-6+RMS(y_ref)^2]`. Profile the actual static cost table `c_lj` for the relevant batch regime.
3. For a log-spaced set of positive cost multipliers `λ`, form labels `argmin_j[E_j(h)+λ*c_lj]`. Save labels, but do not expose reference outputs or future-layer states to the inference router. This supervised oracle is a training device, not an executable competitor.
4. Fit `R_l(h,log λ)=W2*GELU(W_h*h+w_λ*log λ+b1)+b2`, with hidden size initially 128, to these labels using cross-entropy. Compare to a smaller linear router and a larger router only if held-out calibration loss and total runtime justify it. The same alternatives are available to both parameterizations.
5. Evaluate routed **end-to-end** validation NLL. Local output error may not predict language loss. If needed, refine labels using single-layer counterfactual continuation loss on a saved development subset, or use a validated loss surrogate. This costs additional full-model evaluations and is charged to the method. Runtime inputs remain unchanged.
6. Select `λ` using the same latency-budget rule as the defect threshold. A one-shot router executes exactly one chosen configuration per token; it does not compute every branch before selecting one. Optional joint repair training is available to both one-shot models and is separately accounted for.

This protocol supplies a strong causal, executable routing alternative without hiding an unspecified surrogate inside “implement Flextron.” A broader claim specifically against full Flextron requires reproducing its fuller training treatment and attention elasticity in a separate stratum; the core result does not make that claim.

### 6.3 Baseline reproduction and fairness checks

Before architectural comparisons, establish that the dense model learns above unigram and simple bigram controls, that increased dense width can make a detectable difference in the tested regime, and that the elastic model's maximum width is not obviously broken relative to independently trained dense models at comparable exposure. No fixed percentage reproduction tolerance is asserted; inspect paired learning curves and uncertainty at matched cost and matched tokens.

For TorchIntegral-inspired conversion, sampled weight interpolation must use actual coordinate values. The inspected repository contains both a coordinate-aware grid-sampling path and a size-based interpolation path whose function evaluation depends on the number of points rather than their locations. Its quadrature helpers also require compatible nodes and weights. The project uses its own tested hat interpolation for nonuniform Clenshaw–Curtis nodes; it does not blindly combine a helper named “Legendre” with a different grid. [R5]

For scratch versus conversion, run a paired exploratory pilot if scratch continuous training trails its discrete counterpart. Initialize coefficient rows from a shared dense checkpoint, measure the resulting function mismatch, and repair on training data. Conversion is not identity: interpolation and integration change the FFN. If conversion helps, give discrete elastic alternatives the same pretrained asset and count teacher pretraining once in a shared-asset ledger, then count method-specific conversion and repair separately. A converted-only success must be reported as such.

### 6.4 Training recipes, search, and resource accounting

**Core starting recipe:** AdamW, `betas=(0.9,0.95)`, `eps=1e-8`, gradient norm clip 1.0; start learning-rate search at `1e-4,3e-4,1e-3` and weight decay at `0,0.01,0.1`. These are starting search points, not claims of optimality. Use linear warmup over the first 2% of the scheduled tokens, then cosine decay to 10% of the initial rate. Keep scheduler semantics fixed when comparing equal-token runs; record the schedule under equal-cost runs.

Train the continuous model by sampling one global quadrature level per optimizer step, initially uniformly over the validated ladder. Train B1 analogously over its prefix ladder. Sampling across layers independently is a separate, shared eligible recipe. Do not backpropagate through the hard defect stopping decision in the minimum implementation. This avoids adding a second unvalidated learning algorithm.

A sandwich recipe that trains smallest, largest, and sampled intermediate configurations, and in-place or external-teacher distillation, are known improvements. Screen them for both families when the initial elastic backbone is weak; charge every forward/backward/teacher pass. A winner on a search boundary triggers expansion of that boundary if expected information gain and local cost justify it. Generic kernel, normalization, or training fixes must be shared, followed by reruns of affected comparisons. [R8, R12, R13]

Use two distinct accounting views:

- **Primary practical view:** fixed project-specific active-compute budget, including backbone training, conversion, router labels, surrogate fitting, and repair. Charge cold preparation separately and include it in amortized deployment scenarios. A baseline may use a saved cost advantage for additional training under the same budget.
- **Attribution view:** matched training tokens, architecture scale, and optimizer exposure, with actual extra compute reported. This can isolate inductive bias but is not automatically the best practical comparison.

Do not equate training tokens with compute when one method evaluates several widths per update. Shared reusable checkpoints are listed separately from project-specific expenditures. Hyperparameter search gets a recorded, comparable exploration budget; useful published recipes do not count as new training expenditure, but executing them does.

Determine the main training budget from development learning curves and local throughput, not an arbitrary token quota. Use geometrically increasing token blocks, record learning slope, dense-width separation, routing stability, and confidence in cost estimates. Freeze the first scientifically informative budget that the laptop can execute and that supports the selected uncertainty target. If the mechanism only emerges later, the documented curves—not optimism—must justify escalation. The last checkpoint within the frozen budget is eligible; selecting the luckiest checkpoint after test inspection is not.

## 7. Data and workload specification

### 7.1 Core natural-language data

Use the public Salesforce WikiText dataset, configuration **`wikitext-103-raw-v1`**, at a pinned repository revision. The inspected card reports a roughly 315 MB download and approximately 549 MB of represented split data. Its 1,801,350 training rows are **rows, not independent articles**. Validation and test have 3,760 and 4,358 rows respectively. Actual file sizes, rows, article counts, bytes, and token counts must be measured after retrieval. [R25]

**License gate:** the inspected README metadata names CC BY-SA 3.0 and GFDL, while its prose names CC BY-SA 4.0. Preserve this discrepancy in `manifests/licenses.json`; do not silently resolve it. Record source revision, upstream attribution, applicable notices, and the intended local use. Do not redistribute raw text or a repackaged corpus as part of the project by default. If the intended use cannot satisfy the documented terms, block this dataset pending clarification or a separately documented replacement; this plan is not legal advice. [R25]

Preprocessing is deliberately conservative:

1. Read each official split in its original row order. Preserve raw text and newlines; do not lowercase, normalize Unicode, remove headings, or clean punctuation.
2. Reconstruct articles using top-level headings matching the corpus convention (`^ = [^=].* =\s*$` after removing only a terminal newline), explicitly excluding `==` and deeper headings. Inspect a deterministic sample of boundaries and count unmatched leading fragments. Store the exact parser and tests; amend the rule if the source revision differs.
3. Preserve article text byte-for-byte after reconstruction, using the row text exactly as supplied rather than inserting guessed spaces. Empty rows are preserved. Represent each document as BOS, its encoded text, then EOS, with one of each boundary token per document; never attend across documents. Document-content hashes are computed before adding these tokens.
4. Hash each article's exact UTF-8 content. Group exact duplicates before creating any project subdivision. Report train/validation/test exact duplicates and a train-only near-duplicate audit. Do not remove difficult test documents because their scores are inconvenient. Any predeclared duplicate-excluded sensitivity analysis is separate from the official-split primary result.
5. Partition **official training articles** by a salted stable content hash into 90% fit, 5% development-calibration, and 5% sealed independent-confirmation pools. The salt is recorded before modeling. These allocations are operational starting splits, not guarantees of sufficient precision; inspect article counts and lengths before sealing. Official validation is available for hyperparameter and frontier selection. Official test remains protected until §8's freeze.
6. Fit a byte-level BPE tokenizer only on the fit partition, with vocabulary 4,096 for the initial scales, no text normalization, and saved BOS/EOS/PAD IDs. A larger-vocabulary stratum may use 8,192, trained on the same fit data. Save the tokenizer JSON, library version, and hash. All models within a comparison use the identical tokenizer.
7. Store token arrays as `uint16` when all IDs fit, otherwise `uint32`, plus document offsets and byte counts. Training examples are 256-token causal windows wholly within a document; mask padding and do not learn padding targets. Document boundaries reset attention and positions.

If official test contains too few independent articles to estimate the desired effect, do not reinterpret thousands of rows or tokens as independent samples. Report the conditional official-test result and use the sealed training-source confirmation pool as a distinct, larger held-out population. It must never have been used for fitting the tokenizer, training, calibration, or selection.

### 7.2 Optional datasets and asset restrictions

TinyStories is an optional inexpensive synthetic-text diagnostic, not the natural-language confirmation corpus. Its card describes generated stories and a CDLA-Sharing-1.0 license. A TinyStories-only improvement cannot establish benefits on naturally occurring text or general LLM ability. Do not download it unless the pilot needs a cheaper learned-language diagnostic. [R26]

The optional pretrained branch uses `HuggingFaceTB/SmolLM2-135M`, not an invented checkpoint name. Its card and actual config were inspected; pin the revision and verify the Apache-2.0 notices before download. Original pretraining contamination is not fully resolved by this investigation, so paired adaptation comparisons share the checkpoint and disclose that uncertainty. [R27]

### 7.3 Primary request population and causal scoring

Primary requests are article-contained, nonoverlapping 256-token windows with the first 128 tokens as prompt and the next 128 as targets. Include every eligible window in the protected population selected before model evaluation; record excluded short articles and leftover tokens. A secondary all-length analysis covers shorter articles using length bins. Claims from the primary population are explicitly limited to its length condition.

For each request, prefill the 128-token prompt. Its last logit predicts the first target. Then feed the first 127 target tokens one at a time through a KV cache to score the remaining 127 targets. This gives exactly 128 next-token scores under the same teacher-forced inputs for every model. It is not free-running generation, but it supports fair quality and decoding-cost comparisons without divergent token paths.

At each cached single-token step all already stored keys, including the current position's key when applicable, are causally allowed. PyTorch's non-square `is_causal=True` masking uses an upper-left alignment; naively using it with one query and a long KV cache can therefore implement the wrong mask. Use `is_causal=False` for a single last-position query with no future keys, or an explicit absolute-position mask for multi-token cache updates. Set attention dropout to zero during evaluation. Test full-sequence and cached logits before timing. [R20]

Performance regimes: primary batch size 1, prompt 128, continuation 128; secondary batched requests and longer contexts chosen from actual memory headroom. Do not average these regimes into an unlabelled universal speedup. Record prefill, decode, and total request boundaries separately.

## 8. Confirmatory evaluation and statistical protocol

### 8.1 Primary claim, budget anchor, and model selection

The primary estimand is a **paired change in token-weighted NLL at a prespecified measured request-latency budget**, for the protected eligible WikiText request population, after matched project-specific training cost. Lower NLL is better.

Anchor the budget `B*` to the development-measured mean end-to-end request time of the B1 MatFormer model at half of its maximum FFN width, using the common backbone and batch-1 workload. This is an interpretable, baseline-defined budget—not a candidate-selected favorable point. Freeze its numeric value and timing uncertainty before protected evaluation. If this anchor is outside the range the continuous method can express, record infeasibility rather than moving it toward a favorable result.

For each eligible method, select the lowest validation NLL configuration whose measured latency distribution supports feasibility at `B*`. Tune static widths, MixMatch allocations, `τ`, or router `λ` on development data only. The strongest eligible discrete competitor at that budget is chosen on validation, not on test. B4/B5 remain mandatory even when they do not win the discrete screen. Configurations near the timing boundary need more measurement or conservative selection; do not treat one lucky timing as budget compliance.

The primary performance table reports NLL and actual latency jointly. An improvement at the common budget is not automatically a speedup. A **strict Pareto improvement** is a stronger claim requiring evidence that NLL and latency are both lower than the named comparator. Tradeoffs, conditional improvements, and unexplained regressions receive those names instead.

### 8.2 Metrics and aggregation

| Metric | Definition, unit, and aggregation | Use |
|---|---|---|
| Primary quality | `sum(-log p(target))/number_of_unmasked_targets`, nats/token, token-weighted over the frozen request population | Primary task outcome. Do not average article perplexities. |
| Perplexity | `exp(global NLL)` | Human-readable secondary quantity; report absolute NLL differences as well. |
| Primary cost | Wall-clock seconds per complete request, then divide by 128 for ms/scored token | Includes prefill, cached decoding, attention, FFN, routing, cache updates, output logits, and identical log-probability extraction boundaries. |
| Latency tails | p50/p95 request and inter-token time, with session-aware uncertainty | Guardrails against sporadic refinement or synchronization stalls. |
| Memory | Peak process RSS, MPS tensor/driver allocation, prepared-cache bytes, and observed swap change | Prevent a speed claim from concealing memory/offloading costs. |
| Numerical fidelity | Output/gradient errors versus the independent integral; defect calibration and false-safe rate | Mechanism validity, not a substitute for NLL. |
| Routing/work | Counts from §4.8, maximum-level fraction, accepted-level histogram, measured host synchronizations | Attribution and failure diagnosis. |
| Total expenditure | Training/adaptation/search active time, processed tokens, cold preparation time, model/checkpoint/cache bytes | Practical cost accounting. |
| Optional different-tokenizer quality | Bits per UTF-8 byte over a precisely shared text population | Only for cross-tokenizer context, not substituted for the primary paired metric. |

Report absolute differences and ratios where meaningful. Give numeric regressions as prominently as gains. There is no invented minimum percentage improvement. Practical importance is assessed from measured milliseconds, throughput, deployment frequency, tail behavior, memory, and uncertainty—not from a threshold chosen to favor the candidate.

For deployment amortization, compute `T_total(R)=T_prepare+R*T_request` for an explicit request count `R`, and report the break-even request count when preparation differs. Training cost is a separate project investment and is not disguised as reusable pretrained cost.

### 8.3 Exploratory versus protected phases

Before unlocking official test, write and hash `protocol/preregistration.md` containing: code and dependency revisions, data/tokenizer/split hashes, model scales, eligible methods, chosen checkpoints, exact training budgets, primary population, `B*`, tuning outputs, primary contrasts, numerical tolerances, timing protocol, sample size, seeds, analysis scripts, multiplicity handling, and failure rules.

The protected evaluator writes per-request records to a sealed result directory and a one-time summary. It must not feed results back into architecture selection. If a protected result motivates a material method change, label the first result exploratory for that revised claim and use the untouched confirmation pool or a new independently frozen dataset. Record every amendment with its reason and the data already seen.

### 8.4 Independent units and sample-size calibration

Documents/articles are the data-clustering unit. Multiple windows and tokens from the same article are dependent observations. Training seeds represent model-training replicates, not extra language documents. Repeated timings represent measurement repetitions, not extra models or documents.

Use development pilots to estimate three variance components: paired article-level quality variation, paired training-seed variation, and timing-session variation. Start with more than one training replicate when affordable; if seed variance cannot be estimated reliably, call the result conditional on checkpoints rather than pretending it establishes a training-procedure effect. Determine the final fixed number of seeds, articles, and timing sessions **before protected evaluation**.

Choose a target precision in nats/token and milliseconds based on the measured frontier separation, run-to-run noise, and a plausible deployment consequence. Document why that precision is informative. A rough paired normal approximation,

`n ≈ [(z_(1-α/2)+z_power)*σ_pair/δ]^2`,

is only a planning aid for genuinely independent replicate differences. It is not applied to raw tokens, and it is not a substitute for simulation using the observed cluster structure. Vary plausible variance estimates upward to assess sensitivity to a noisy pilot. If the needed precision is unaffordable, report the detectable effect and an inconclusive result rather than repeatedly inspecting protected significance.

### 8.5 Analysis procedure

For each frozen seed pair and article, retain summed negative log likelihood and target count. Estimate the pooled token-weighted difference with resampling that preserves article clusters. For a training-procedure claim, independently resample paired seed IDs and article IDs (a crossed bootstrap), reconstructing the weighted estimator on each replicate. Report the conditional-on-checkpoints article interval separately from the interval including training variation. When seeds are few, include a conservative seed-level paired analysis and explicitly state its instability.

Use paired timing comparisons in randomized blocks across whole measurement sessions. Resample sessions/blocks, not individual GPU kernel calls as if independent. Use enough bootstrap replicates that Monte Carlo uncertainty in the interval endpoints is negligible relative to the claimed effect; check by increasing the replicate count, rather than declaring a magic fixed count sufficient.

Specify one primary quality contrast against the development-selected best discrete competitor. Mandatory same-function and continuous-static contrasts establish attribution. Use Holm-adjusted tests or simultaneous bootstrap intervals for multiple confirmatory superiority claims; secondary frontiers and slice discoveries are exploratory unless preregistered. The claim of joint Pareto dominance needs simultaneous evidence for both quality and latency, not two opportunistically selected unadjusted intervals.

A confidence interval spanning useful benefit and harm is inconclusive. A narrow interval near zero supports no meaningful local advantage at the achieved precision. A precise disadvantage is evidence against the selected implementation and regime. It does not prove that every continuous architecture is impossible.

### 8.6 Timing protocol and harness validation

Execute native FP32 paths first, with the laptop connected to power and in a recorded power mode. Record other substantial processes, battery/power status, operating system, available memory, and optional thermal information. Do not require privileged instrumentation to execute the core experiment.

Warm model weights, prepared interpolation caches, attention caches, and allocator paths using unscored development requests. Continue warmup until consecutive timing blocks are stable relative to pilot noise; record the rule and excluded warmup count. Reset KV caches between requests. Include per-request cache allocation or reuse exactly as specified for every method. Measure cold start separately.

Call `torch.mps.synchronize()` immediately before starting and after ending a timed request. Use a monotonic high-resolution clock. Internal synchronizations caused by routing remain inside the boundary. Do not synchronize every ordinary kernel in the primary benchmark, which would artificially punish all methods; use separate instrumented runs for attribution. MLX, if used, requires explicit realization of lazy results before timing completion. [R19, R21]

Randomize method order within matched request/session blocks and use reverse-order blocks to detect drift. Record all raw timings. Reject a block only for preregistered measurement faults such as an interrupted process or detected background update, not because it is slow. Thermal degradation under sustained use is a result, not automatically an outlier.

Validate the harness with: identical-method A/A comparison; sampled/factored output equivalence; a deliberate extra matrix multiplication that should increase cost; a deliberately narrowed or corrupted model that should change quality; accepted-token counters proving real skipping; and cached/full causal parity. Parallel teacher-forced quality evaluation may replace sequential scoring only after per-token routes and logits agree within the validated tolerance. Timing always uses the specified request path.

## 9. M3 Max feasibility and hardware-adaptive execution

### 9.1 Discovery before assumptions

The machine's memory configuration is unknown. Do not infer it from the maximum advertised M3 Max configuration. Run these existing macOS commands and save their outputs:

```bash
uname -m
sw_vers
sysctl -n hw.memsize
system_profiler SPHardwareDataType SPDisplaysDataType
df -h
vm_stat
sysctl vm.swapusage
```

Confirm native `arm64` execution, available storage, actual unified memory, macOS version, power status, and whether the intended Python process is running through Rosetta. Record any unavailable optional command rather than inventing its output. The Apple hardware specification is a reference, not a substitute for discovery. [R28]

### 9.2 Software stack and compatibility gate

**Primary stack:** native-arm64 Python, PyTorch with MPS for training/inference, NumPy/SciPy for independent CPU FP64 mathematics, pytest, PyYAML, Hugging Face Hub/datasets/tokenizers, and PyArrow for artifacts. Use standard-library timing, hashing, JSON, and subprocess facilities where possible. Avoid a CUDA-only training framework, distributed runtime, or a custom Metal extension in the core path.

Select an actually available stable PyTorch wheel compatible with the discovered Python/macOS version. The retrieved documentation included a `2.14` path; that does not establish the availability or suitability of a particular installation wheel on the user's machine. Record the installed version, platform tag, package sources, and wheel hashes, and freeze only after the operation probe succeeds. Do not fabricate a package version to make the plan look more reproducible. [R19]

Existing bootstrap commands, after selecting a compatible native Python, are:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch numpy scipy pytest pyyaml huggingface_hub datasets tokenizers pyarrow
python -m pip freeze > requirements.lock
```

These commands create an initial environment, not the final hashed dependency manifest. Capture resolved wheel URLs/hashes through the install report or a wheelhouse manifest; then reproduce the environment from that lock on a fresh virtual environment. Add `transformers` only for the optional pretrained branch. Preserve third-party notices; never assume downloadable model weights are automatically unrestricted open-source assets.

Set the fallback environment variable **before importing PyTorch**:

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=0
```

Then test `torch.backends.mps.is_built()` and `is_available()`. The inspected MPS documentation states a macOS 14-or-later requirement for its documented path; actual package/runtime compatibility remains a gate. A successful dense tensor operation is not sufficient. [R19]

Probe forward, backward, optimizer updates, shape changes, and inference for: matrix multiplication; exact GELU/erf; LayerNorm; RoPE arithmetic; scaled dot-product attention with the actual masks; embedding lookup; gather/index selection; scatter or segment reduction; cumsum where used; scalar comparisons and active-token grouping; cross-entropy; and KV-cache updates. Test the smallest and largest intended shapes. Compare against CPU FP64 or a CPU FP32 reference as appropriate.

The inspected PyTorch fallback source has explicitly registered fallback cases in addition to its environment-controlled general behavior. Therefore the variable alone is not proof of wholly accelerated execution. Profile representative paths, record host/device transitions, and reject hidden large CPU computations from an “MPS accelerated” claim. Do not use embedding renormalization or other unnecessary operations that trigger an avoidable special fallback. [R19]

Do not move FP64 quadrature construction or SciPy functions onto MPS. Generate nodes, weights, and small reference tables on CPU, then transfer FP32 constants. No FFT or complex tensor support is required by the selected core mechanism.

### 9.3 Faithful fallback rules

If a reduction primitive lacks a stable MPS path, try a mathematically equivalent gather/segment implementation and re-run numerical and timing tests. If exact GELU's production path is inadequate, use a validated equivalent real-valued formula; do not silently replace it with tanh-GELU or ReLU. Compare changed kernels across all affected methods.

A CPU-only path may establish correctness or a CPU performance result, but does not establish M3 GPU competitiveness. If PyTorch overhead is the obstacle while a measured optimistic bound shows meaningful hardware headroom, a full paired MLX implementation may be justified. MLX's inspected installation documentation requires native Apple Silicon Python and supported macOS, and its lazy evaluation must be materialized at benchmark boundaries. Port candidate **and baselines**, repeat the full parity tests, and label the framework stratum. [R21]

A custom Metal kernel is an evidence-triggered optimization project, not a prerequisite hidden inside the promised laptop PoC. If it is needed to realize a predicted gain but is not implemented, report that limit. Do not declare the mathematical hypothesis false solely because a Python routing loop is slow.

Reduced precision or quantization is not required. FP32 is the primary comparison. FP16/BF16 may be investigated only after kernel support, numerical stability, training behavior, and routing-decision parity are tested, and only with equivalent options for competitors. Different roundoff can change threshold decisions; that is a scientific sensitivity, not just a compiler detail.

### 9.4 Parameter and memory model

For the shared scratch architecture with a `4d` GELU FFN and tied embeddings, an approximate parameter count is

`P ≈ 12*L*d² + V*d`,

ignoring small bias and normalization terms. Count actual tensors in every implementation. Initial configurations are:

| Configuration | Layers, residual/head dimensions, vocabulary | Approximate parameters | FP32 weights + gradients + two Adam moments (`16P`) |
|---|---|---:|---:|
| Plumbing / first learned diagnostic | `L=6,d=256,heads=4,V=4096` | 5,767,168 | 88 MiB |
| Main-scale candidate | `L=8,d=384,heads=6,V=4096` | 15,728,640 | 240 MiB |
| Scale-sensitivity candidate | `L=12,d=512,heads=8,V=8192` | 41,943,040 | 640 MiB |

These values are arithmetic estimates, not peak process memory. They do not establish that a full workload fits or runs at a useful rate.

Model peak memory as

\[
M_{peak}\approx16P+M_{optimizer\ workspace}+M_{activations}+M_{attention}
+M_{logits}+M_{KV}+M_{quadrature/cache}+M_{data}+M_{runtime}.
\]

For byte width `s=4`, useful conservative allocation models are:

- Saved activations: start from `s*L*B*T*(12d+3M_max)` and measure the implementation's actual multiplicative overhead. This is a planning model, not a tight bound.
- Unfused attention intermediates: one or more arrays of size `s*L*B*heads*T²`. Whether they are materialized depends on the actual kernel.
- Logits and loss workspace: at least `s*B*T*V`, potentially more during loss computation. Chunking the loss is a shared optimization.
- Standard multi-head KV cache: `2*s*L*B*T*d`; grouped-query attention uses the appropriate KV-head ratio.
- Sampled FFN cache: approximately `2*s*L*M_max*d` for a single richest level, plus biases and index tables. Separately materialized nested levels approximately double this amount for a geometric ladder, but measure the exact sum including endpoints.
- Coefficient-space segment activations: proportional to `B*T*K*q` for Gauss order `q` if materialized all at once; stream intervals or Gauss nodes when needed.
- Training interpolation remains differentiable. Do not reuse detached inference caches inside training; that would sever gradients to coefficient fields.

For context, the first model with `B=2,T=256,M_max=1025` has about 72 MiB under the saved-activation term above, before attention, logits, autograd overhead, and allocator behavior. This illustrative arithmetic is not an observed peak. Use tensor and driver measurements plus process RSS, because unified memory, cached allocations, and operating-system usage are not identical quantities.

### 9.5 Calibration and resource selection

Measure idle memory and ordinary system variability first. Reserve headroom for the observed operating-system working set, allocator spikes, and checkpoint serialization. Use `torch.mps.recommended_max_memory()` when the installed version provides it, but do not treat that value as permission to consume all unified memory. Record `current_allocated_memory()` and `driver_allocated_memory()` when supported. Never disable memory safeguards or intentionally rely on sustained swap to force a configuration to fit. [R19]

For each eligible shape, run representative training updates, prefills, and cached decoding under the richest and typical adaptive resolutions. Increase microbatch until throughput ceases to improve or the measured memory margin is threatened. Prefer reducing microbatch and increasing gradient accumulation to preserve the effective batch before reducing context length, width, or layer count. Accumulate losses by the total number of nonpadding targets, not by an unweighted average of differently sized microbatches.

Record sustained training throughput `q_train` in tokens/second and request throughput in requests/second, both after warmup and during a sustained block. Estimate a proposed training stage with `D/q_train` plus measured validation, checkpoint, and router costs. The rough dense-transformer arithmetic `6*P*D` FLOPs is a cross-check only; it excludes several material terms and cannot predict M3 elapsed time reliably.

Choose the initial learned scale that (a) fits without adverse memory pressure, (b) can expose a dense-width quality difference, and (c) has a measured FFN workload large enough to test the proposed savings. The 6M model can serve as plumbing even when it cannot satisfy (c). Move to the next scale when profiling or learning curves identify a specific small-scale limitation and the estimated information gain justifies the cost. Do not enlarge merely because the first result is negative.

### 9.6 Stage-specific storage and runtime estimates

| Stage | Storage / memory estimate | Runtime estimation procedure |
|---|---|---|
| CPU mathematics and operation probe | Small arrays and logs; no dataset or model asset download | Time the actual test suite; no sustained training required. |
| Data preparation | Roughly 0.32 GB downloaded plus 0.55 GB represented WikiText data before caches; token arrays need `2D` or `4D` bytes, where `D` is measured token count | Measure parsing/tokenization on a fixed article batch, extrapolate, then compare to actual completion. |
| Scratch training | Memory model above; one resumable FP32 checkpoint roughly `12P` bytes for weights and Adam moments, plus metadata and serialization overhead | `D/q_train` from local sustained updates, plus all ancillary passes. |
| Frozen-layer calibration | Streaming `n*d` states for one layer at a time; errors/cost labels are small scalars per choice | Measure states/second and Gauss reference cost on a development shard. Do not retain all vocabulary logits. |
| Confirmation | Per-request scalar metrics and route summaries, not complete hidden states; raw timing/session logs | Measured requests/second times frozen request count, methods, seeds, and repetitions. |
| Optional 135M adaptation | About 2.16 GB decimal for `16P` alone at 135M parameters, before activations and runtime; asset size must be retrieved from its actual manifest | Pilot a faithful FFN adaptation and use measured adaptation throughput, not a published GPU throughput. |

Check free disk before allocating dataset copies, wheel caches, and checkpoints. Do not keep every intermediate full optimizer state by default: preserve resumable latest, selected checkpoints, provenance, and scalar histories, with a documented retention policy that never deletes unfavorable results or their identifying configuration. Checkpoint cadence follows measured save overhead and acceptable recomputation, not an arbitrary wall-clock promise.

### 9.7 Interruption and recovery

Save atomically at optimizer-step boundaries: model, optimizer, scheduler, gradient-scaler state if used, CPU and available device RNG states, data sampler permutation and offset, consumed-token count, accumulated project cost, selected width-sampling state, and configuration/data hashes. Initial core width sampling occurs on CPU to make its RNG state explicit. Write to a temporary file, flush, checksum, then rename.

On resume, verify every hash and parameter shape, reconstruct inference caches from the current weights, and run a saved canary batch. A continuous versus resumed CPU run should match exactly when deterministic operations permit it; an MPS run must meet its prevalidated tolerance and preserve route/work accounting. A failed canary blocks resumed confirmatory results. Changing microbatch, precision, node rules, or optimizer state requires an explicit amendment if it affects the experimental treatment.

On out-of-memory, preserve failure logs and the last valid checkpoint, release caches, and select the next memory-safe microbatch. Never silently shorten requests or lower resolution in the evaluator. A failed allocation is not a favorable low-cost data point.

## 10. Risk-ordered work packages and decision gates

All `python -m cff_poc ...` commands below are **proposed project commands that the coding agent must implement**. They are not existing package interfaces. Their names define the intended CLI. Existing third-party commands are identified separately in §9.

### WP00 — Environment, provenance, and permissions

**Objective/risk:** establish that the proposed local experiment is executable and assets may be used. **Dependencies:** none. **Inputs:** this plan, target laptop, permitted network access.

**Actions:** create the repository in §11; implement `doctor`; run hardware discovery; create the native environment; pin the installed package set; save asset URLs and source revisions; record WikiText's licensing discrepancy and AWN's restrictions. Probe MPS availability without downloading model weights. Capture full repository commit SHAs when code is reused; do not rely on mutable `main` URLs for execution.

**Command/files:** `python -m cff_poc doctor --output manifests/hardware.json`; `manifests/environment.json`, `assets.json`, `licenses.json`, dependency lock and wheel manifest.

**Tests/output:** structured pass/fail for architecture, disk, memory, runtime import, MPS availability, and each license decision. Missing permissions are explicit `blocked` entries. **Resources:** negligible model allocation; record idle memory and storage.

**Gate/next:** pass → WP01–WP03. Unsupported environment → repair or test a faithful documented fallback before dependent work. Unresolved dataset permission → mathematical work may continue, natural-language asset download cannot. AWN restriction → exclude its code, preserve a baseline-coverage limitation or obtain an authorized path; never quietly vendor it.

### WP01 — Numerical kernel and operator correctness

**Objective/risk:** H1's mathematical implementation; invalid quadrature or fake continuous representation. **Entry:** native environment or independent CPU environment passes WP00. **Inputs:** Appendix B, deterministic test seeds.

**Actions:** implement `quadrature.py` and continuous FFN reference paths; reproduce §5; add endpoint, constant/affine field, mass, off-grid coordinate, normalization, sampled/factored, segment-reference, and gradient tests. Exercise GELU tails, near-zero slopes, large hidden norms, and the known defect counterexample.

**Commands/files:** `python -m cff_poc test-math --device cpu`; `tests/test_quadrature.py`, `test_integral_ffn.py`, `artifacts/math/results.json`.

**Acceptance:** CPU FP64 ordinary test output errors below `1e-10` with scale-aware relative checks; no unexplained gradient mismatch; triangle test must demonstrate failure rather than “pass” a false certificate. These are numerical test tolerances, not improvement targets. Tighten when the estimated effect is smaller than implementation error.

**Resources:** CPU small matrices; no language training. **Next:** discrepancy → implementation failure, repair before performance work. Agreement → WP02 and WP05. An intrinsically unstable analytic formula may be replaced by the validated segment rule without changing the mathematical target.

### WP02 — MPS primitives, dispatch, and no-data headroom

**Objective/risk:** H5's hardware path, silent CPU fallback, launch overhead, and compiler dominance. **Entry:** WP00–WP01. **Inputs:** initial model shapes, representative smooth and irregular random fields.

**Actions:** implement `probe-ops` and `bench-kernels`; test forward/backward and an optimizer step for every critical operation; compare sampled, factorized, and segment implementations; benchmark a zero-work grouping/controller loop and realistic nested reuse. Include batch-1 decoding shapes and batched training shapes. Measure dense FFN, attention, and vocabulary-projection components in an untrained complete backbone.

**Commands/files:** `python -m cff_poc probe-ops --device mps`; `python -m cff_poc bench-kernels --config configs/pilot.yaml`; `kernel_dispatch.json`, raw timings, memory traces.

**Acceptance:** output discrepancies typically no larger than `5e-5*(1+RMS(reference))` in FP32 test cases, and gradients within predeclared scale-aware tolerances; investigate worst cases and tighten if needed. Actual skipping counters and host synchronization counts must match the algorithm. No unsupported path may be labelled accelerated.

**Resources:** allocate one model/shape at a time; measured memory model. **Next:** unsupported op → faithful replacement and retest. Same-function compiler wins → keep it as the primary opponent; do not abandon learned-quality tests solely from random fields. Even optimistic overhead bound cannot fit the budget at any locally feasible informative shape → stop adaptive implementation growth and report a local execution limitation, while retaining the cheap static-continuum diagnostic. Credible headroom → WP03–WP06.

### WP03 — Corpus, tokenizer, and evaluation harness

**Objective/risk:** leakage, dependent observations, causal-cache bugs, biased timing. **Entry:** WP00 permission and storage gates. **Inputs:** pinned WikiText revision.

**Actions:** implement §7 preprocessing and split sealing; train tokenizer on fit only; create document-contained windows; implement full and incremental inference, loss accounting, and paired session timing. Add A/A, intentional-corruption, extra-work, and padding tests. Ensure target text is never passed into routing except as ordinary preceding teacher-forced input.

**Commands/files:** `python -m cff_poc prepare-data --config configs/base.yaml`; `python -m cff_poc test-harness`; dataset/tokenizer/split manifests and `tests/test_causality.py`, `test_metrics.py`.

**Acceptance:** exact byte/token counts and document IDs; no overlap between fit/calibration/confirmation hashes; correct 128-target scoring; full/cached logits agree; identical models produce no systematic quality difference; timing detects added work. **Resources:** measured disk/tokenization estimate, no full training.

**Next:** parser ambiguity or leakage → fix and regenerate all downstream manifests before sealing. Too few test articles → plan wider uncertainty or a distinct sealed confirmation population. Valid harness → WP04.

### WP04 — Reproduce the relevant discrete and static baselines

**Objective/risk:** weak or unfair opponents. **Entry:** WP02–WP03 correct. **Inputs:** common backbone, fit/validation data, initial recipe.

**Actions:** implement B0–B4 interfaces; train geometrically increasing pilot blocks; plot validation learning curves and actual training cost. Implement MatFormer v2 sampling, a static MixMatch search, and discrete cumulative-prefix reuse. Screen FFF and the permitted AWN path for relevance, fidelity, and reachable frontier. Save every attempt.

**Commands/files:** `python -m cff_poc train --config configs/pilot.yaml --method dense`; repeat method identifiers `matformer`, `inn_static`, and eligible screens; `baseline_fidelity.md` and cost ledger.

**Acceptance:** language learning above trivial controls; measurable width effect or a diagnosed inadequate scale; no obvious maximum-width elastic implementation bug. Tuning boundaries and divergence are investigated, not interpreted as candidate superiority. **Resources:** WP02 throughput determines token blocks and microbatch; no sealed data access.

**Next:** broken baseline → repair before any win claim. Underpowered scale → adjust using learning/profiling evidence. Usable baseline frontier → WP05–WP07. A stronger screen must enter later comparisons, not be dropped.

### WP05 — Minimum complete continuous model

**Objective/risk:** H1, initialization, and quadrature-dependent learning. **Entry:** WP01 and usable WP04 backbone. **Inputs:** same data, parameter scale, static quadrature dispatch.

**Actions:** implement differentiable coefficient fields and random-resolution training; train a static continuous model through the same pilot blocks. Compare each level to independent segment integration on saved validation states. If scratch underperforms, execute the paired converted-initialization diagnostic from §6.3. Test untrained intermediate node counts and an independently chosen quadrature family.

**Commands/files:** `python -m cff_poc train --method cff_static --config configs/pilot.yaml`; `python -m cff_poc calibrate --kind integral-fidelity`; per-layer convergence and language-loss tables.

**Acceptance:** a useful learned function whose independent integration behavior is understood. A positive result is not required. Distinguish interpolation approximation, training instability, and a fixed-grid model that does not approximate its asserted integral. **Resources:** same calibrated training scale; stream one layer's reference states.

**Next:** no usable language model after justified initialization/tuning → selected parameterization evidence is negative; diagnose capacity/geometry before scale. Fixed-grid gaming → revise training or narrow to discretized architecture, not claim continuum validity. Usable continuum → WP06.

### WP06 — Frozen-layer oracle, controller, and real overhead

**Objective/risk:** H2–H3 and whether adaptive work is worth training further. **Entry:** frozen usable continuous and discrete elastic pilots; correct MPS dispatch. **Inputs:** development-calibration states, independent reference, cost table.

**Actions:** for each layer/state evaluate all allowed levels offline, record true numerical errors and defects, and construct a best-cost-per-error oracle. Train/calibrate defect and one-shot controllers, including the same controllers on discrete prefixes. Execute the actual grouped inference path. Measure the gap between an uncharged oracle bound, a cost-charged policy, and real end-to-end timing.

**Commands/files:** `python -m cff_poc calibrate --kind routing`; `controller_calibration.parquet`, false-safe curves, oracle/real frontier, synchronized timing logs.

**Acceptance:** no oracle labels or rich reference states at inference; measured active work agrees with predictions; false-safe cases and cap rates visible. **Resources:** bounded streamed states, with sample count determined by calibration precision rather than storing all activations.

**Next:** oracle shows no useful heterogeneity → evidence against H2 for this learned model; avoid expensive router optimization. Oracle helps but actual controller loses → controller/hardware diagnosis; one-shot or per-request branch may be justified. Candidate helps only relative to an unoptimized implementation → fix comparator. Credible real signal or unresolved informative uncertainty → WP07.

### WP07 — Joint pilot, scale choice, and precision planning

**Objective/risk:** H4–H6, training-cost confounds, and underpowered confirmation. **Entry:** valid baselines, continuum, routing, and measurement. **Inputs:** development learning curves and cost/error results.

**Actions:** compare static/defect/one-shot crossed with continuous/discrete parameterizations; include fastest same-function compiler. Tune common recipes and baseline allocations. Estimate seed/article/session variance, project cost, budget anchor, feasible model scale, and precision. Evaluate a second scale only when the first is launch-bound, lacks a meaningful width effect, or shows a repeatable mechanism requiring confirmation.

**Commands/files:** `python -m cff_poc run-pilot --config configs/pilot.yaml`; `pilot_analysis.md`, `power_design.json`, proposed frozen configurations.

**Acceptance:** a fixed confirmatory design whose costs are measured locally and whose interval resolution is sufficient for the intended claim, or an explicit inability to reach it. **Resources:** extrapolate from actual sustained throughput; include all tuning and router costs.

**Next:** frontier is consistently dominated with informative precision → report negative/no advantage; no automatic larger run. Genuine but small unresolved signal → choose a justified fixed replication design. Insufficient affordable precision → report inconclusive. An interpretable design → WP08.

### WP08 — Freeze and execute the controlled comparison

**Objective/risk:** unbiased evidence for the primary claim. **Entry:** WP07 design and all blocking gates resolved. **Inputs:** immutable configurations, protected data, preselected seed list.

**Actions:** hash preregistration; prohibit protected feedback into tuning; execute fixed training/selection rules and request evaluation; run paired timing blocks and the frozen statistics script. Preserve failed runs and apply only preregistered replacement rules for genuine infrastructure failures.

**Commands/files:** `python -m cff_poc freeze --config configs/confirm.yaml`; `python -m cff_poc confirm --manifest protocol/frozen_manifest.json`; raw request records and adjusted intervals.

**Acceptance:** complete planned observations or an honest incomplete/blocked label; all comparisons traceable to costs and hashes; no silent post-test threshold changes. **Resources:** WP07's measured schedule and checkpoint strategy.

**Next:** positive → WP09 attribution/independent confirmation before a strong claim. Negative → WP09 targeted explanation without relabelling the primary result. Inconclusive → report achieved precision; a new study requires a new design and untouched evidence.

### WP09 — Attribution, robustness, and independent confirmation

**Objective/risk:** alternative explanations for apparent gains. **Entry:** valid primary result, positive or negative. **Inputs:** frozen outputs, sealed confirmation pool where still untouched.

**Actions:** execute the relevant ablations in §11.3; test the most threatening explanation first. Re-evaluate after shared baseline optimization if a fairness defect is found, recording the original result as invalid or exploratory as appropriate. Confirm a material positive claim on untouched articles and an independent training replicate set selected before seeing those outcomes.

**Commands/files:** `python -m cff_poc ablate --manifest protocol/ablation_manifest.json`; `attribution.md`, confirmation results, amendments.

**Acceptance:** the stated contribution survives its necessary controls, or the claim is narrowed to the surviving explanation. **Resources:** prioritize discriminating tests; avoid a blind full factorial of irrelevant settings.

**Next:** continuous-specific value survives → consider WP10 if a pretrained claim is worthwhile. Only generic routing/regularization helps → report that finding. Regression or instability → do not average it away. All outcomes → WP11.

### WP10 — Conditional pretrained-scale extension

**Objective/risk:** whether the local learned mechanism transfers beyond scratch-model evidence. **Entry:** an informative prior result, a specific scale hypothesis, measured affordable memory/throughput, and asset permission. This is not required to complete a valid negative core study.

**Actions:** adapt the verified SmolLM2-135M base model with the exact configuration described in §12. Preserve SwiGLU, tokenizer, RoPE, normalization, and KV-head structure. Compare shared-checkpoint continuous and discrete FFN adaptations, same repair data and project budget. Run operation parity and unmodified-checkpoint regression before training.

**Commands/files:** `python -m cff_poc prepare-pretrained --config configs/pretrained.yaml`; `pretrained_conversion.md`, original-versus-loaded parity, adaptation ledger.

**Acceptance:** no architectural substitution concealed as a conversion; unmodified model parity; adaptation costs and contamination uncertainty disclosed. **Resources:** measured 135M pilot, microbatch adaptation, no remote GPU.

**Next:** memory/operation gate fails → retain core claim and record limitation. Conversion cannot preserve useful behavior despite controlled repair → report negative adaptation evidence. Positive result → independent confirmation required; still no claim about all LLMs.

### WP11 — Reproducibility and final synthesis

**Objective/risk:** an irreproducible, selectively reported, or nonresumable result. **Entry:** any terminal outcome, including early blocking. **Inputs:** all manifests, histories, raw records, and decisions.

**Actions:** run resume canaries; reconstruct a selected numerical/evaluation result from a fresh environment; build the report automatically from raw data; include every attempted configuration and failure category. Separate confirmed results, exploratory discoveries, compiler equivalences, estimates, and unexecuted extensions.

**Commands/files:** `python -m cff_poc resume-check --run RUN_ID`; `python -m cff_poc report --manifest protocol/frozen_manifest.json`; `REPORT.md`, `RESULTS.json`, reproducibility checklist, and decision log.

**Acceptance:** a competent new agent can trace every headline number to a checkpoint/configuration, data population, timing boundary, and raw record. Unknowns remain labelled. **Resources:** one saved canary and selected rerun; scale to measured reproducibility uncertainty if needed.

**Terminal decisions:** continue with a specific evidence-backed extension; narrow the contribution; report no local advantage; report an implementation/resource limit; or report inconclusive sensitivity. There is no “success” state created merely by completing code.

## 11. Repository, records, and attribution program

### 11.1 Minimal repository and configuration conventions

```text
cff-poc/
  README.md
  pyproject.toml
  requirements.lock
  manifests/
    hardware.json  environment.json  wheels.json
    assets.json  licenses.json  splits.json  tokenizer.json
  configs/
    base.yaml  pilot.yaml  confirm.yaml  pretrained.yaml
  src/cff_poc/
    __init__.py  __main__.py  cli.py
    hardware.py  data.py  model.py  ffn.py  quadrature.py  routing.py
    train.py  evaluate.py  benchmark.py  statistics.py  records.py
  tests/
    test_quadrature.py  test_integral_ffn.py  test_gradients.py
    test_causality.py  test_metrics.py  test_cache.py  test_resume.py
  protocol/
    preregistration.md  frozen_manifest.json
    amendments.md  decision_log.md  ablation_manifest.json
  artifacts/<run_id>/
    config.resolved.yaml  provenance.json  metrics.jsonl
    checkpoints/  raw_requests.parquet  raw_timings.parquet
    numerical_checks.json  decision.md  failure.json
  REPORT.md
  RESULTS.json
```

Use one common backbone class, separate FFN implementations, and one evaluator. Avoid copying and diverging entire training scripts per method. A method implementation cannot override test population, cost measurement, tokenizer, or loss aggregation.

Example **proposed configuration**, resolved before execution:

```yaml
schema_version: 1
phase: pilot
method: cff_defect
model:
  layers: 6
  d_model: 256
  heads: 4
  vocabulary: 4096
  context: 256
  activation: gelu_exact
  norm_epsilon: 0.00001
ffn:
  knots: 1024
  gain: 1024
  basis: linear_hat
  quadrature: clenshaw_curtis
  panels: [128, 256, 512, 1024]
  controller: successive_defect
  threshold_scale_source: development_calibration
  global_threshold: null  # resolved by calibration; forbidden null in confirmation
training:
  precision: float32
  width_sampling: one_global_choice_per_step
  effective_batch_targets: null  # measured/tuned in pilot, then common and frozen
  microbatch: null
  project_compute_budget_seconds: null
  data_revision: null
  tokenizer_sha256: null
evaluation:
  prompt_tokens: 128
  target_tokens: 128
  batch_size: 1
  primary_budget_seconds: null
  population_manifest_sha256: null
reproducibility:
  seed: 1013  # diagnostic seed, not the final replication design
  source_revision: null
  dependency_lock_sha256: null
```

The resolver validates `gain==knots`, at least two valid nested controller levels, no unknown fields, compatible head dimensions, and immutable non-null provenance in confirmation. Training and inference caches must be separate objects. File names and content hashes, not conversational memory, carry the project state.

### 11.2 Record schemas and reproducibility

Every run gets a unique ID linked to its parent configuration and methodological amendment, if any. The immutable `provenance.json` contains: UTC start, local timezone, hardware and environment hashes, source revision and dirty diff, dependency lock, data and tokenizer hashes, method and baseline role, seed, phase, precision, initialization asset, checkpoint ancestry, and inference dispatch hash.

Each `metrics.jsonl` entry records at least:

```json
{
  "schema_version": 1,
  "run_id": "assigned-at-run-creation",
  "phase": "pilot",
  "step": 0,
  "fit_targets_seen": 0,
  "active_compute_seconds": 0.0,
  "validation_nll_nats_per_token": null,
  "rss_peak_bytes": null,
  "mps_tensor_bytes": null,
  "mps_driver_bytes": null,
  "swap_delta_bytes": null,
  "status": "started",
  "failure_category": null
}
```

Null means not measured, not zero. Raw request records contain `run_id, seed, article_hash, request_id, prompt_length, target_count, sum_nll, prefill_seconds, decode_seconds, total_seconds, batch_size, route_summary, exhausted_count, session_id, order_in_block`. Timings and quality may be separate linked tables when parallel quality evaluation has passed the causal parity check. Do not save complete logits by default; that can dominate storage.

Store all failed runs, unfavorable hyperparameters, parser exclusions, rejected timing blocks with reasons, and unselected checkpoints' scalar histories. A baseline fix invalidates affected comparisons; it does not retroactively make an old win valid. Record exactly which outputs are exploratory versus protected.

The coding agent may choose ordinary non-destructive engineering details, such as parser implementation, temporary-file layout, or mathematically equivalent vectorization. Changes to hypothesis, activation, basis family, evaluation population, principal cost boundary, baseline eligibility, training resource policy, or primary statistical contrast require a methodological amendment. Append a decision at the end of every work package: evidence, interpretation, action, blocked dependencies, and next command.

### 11.3 Required ablations and adversarial tests

Run the tests required by the actual claim. Do not create an expensive full factorial before a main signal exists.

| Test | Exact intervention / comparison | Interpretation |
|---|---|---|
| Parameterization × controller | Discrete prefix versus continuous hats, each static, defect-controlled, and one-shot-routed | Separates continuum-specific value from generic adaptive computation. |
| Same-function compilation | Freeze continuous weights; compare sampled, factorized, segment, and validated analytic paths | A speedup that vanishes here is an implementation artifact, not an integral advantage. |
| Static layer allocation | Give discrete and continuous static models equivalent layer-wise budget search | Prevent a global-width straw baseline. |
| Generic training improvements | Share stronger initialization, width sampling, distillation, dropout, and optimizer tuning | Detects training-recipe rather than architectural advantage. |
| Capacity and storage | Actual coefficient/router parameters, sampled-cache bytes, dense width, and low-rank/factored controls | Prevents extra parameters or cached matrices from being presented as free continuous capacity. |
| Quadrature family and off-grid resolution | Uniform composite, nested Clenshaw–Curtis, and segment reference; node counts not used in training | Tests a genuine shared integral instead of dependence on one discretization. |
| Quadrature overfitting | Compare training-rule loss/output to independent richer integration during and after training | Lower discrete loss can coexist with a poorly represented continuous function; adjacent neural numerical work motivates this concern. [R23, R24] |
| Coordinate regularity | Compare hat basis to a piecewise-constant/box-basis control; measure variation across neighboring trained knots | Distinguishes interpolation inductive bias from simple neuron lookup. A post-training knot permutation changes the function and is not an invariance test. |
| Difficulty attribution | Correlate defects and selected levels with true local error and single-layer counterfactual NLL changes | Numerical accuracy and task accuracy may diverge. Correlation alone does not establish runtime value. |
| Router information firewall | Remove all reference states/labels at inference; assert router inputs and causal prefixes | Prevents oracle leakage and future-layer dependence. |
| Real skipping | Compare trace counts and profiler events with an evaluate-all-and-mask control | Prevents claims of saved compute when all branches execute. |
| Pathological fields | Narrow spikes, alternating knot values, endpoint spikes, near-cancellation, large slopes, and zero-slope GELU tails | Exposes false-safe defects and numerical instability within the field class. |
| Maximum-level behavior | Requests engineered or selected in development to force all refinement levels | Reveals worst-case slowdown and latency tails; maximum budget is not free. |
| Batch/context/precision | Batch 1 versus larger feasible batches; longer contexts; validated lower precision | Establishes the regime of benefit and precision-sensitive routing changes. |
| Data slices | Article length, punctuation/numbers, rare-token frequency, headings, and development-defined difficulty bins | Tests favorable-slice selection; protected post-hoc slices remain exploratory. |
| Independent initialization and conversion | Scratch versus shared-checkpoint conversion and repair | Separates a trainability/ordering issue from representational failure. |
| Fresh confirmation and resume | Untouched article pool, independently chosen training replicates, resumed versus continuous run | Tests repeatability and state/provenance integrity. |

**Static layer-allocation search:** start from both a homogeneous low-width configuration and a homogeneous high-width configuration. Evaluate one-layer, one-level neighboring changes on development NLL and actual request cost; build a nondominated set and expand its improving neighbors until no untested affordable neighbor improves the frontier or the shared development search budget is exhausted. Use both forward growth and backward pruning to reduce dependence on one greedy starting point. Include homogeneous configurations regardless. Apply identical search machinery and a comparable search-cost allowance to B1 and B4. This is a controlled local MixMatch search, not an assertion that the global allocation optimum was found.

## 12. Outcome interpretation and evidence-triggered alternatives

### 12.1 Allowed conclusions

| Observed outcome | Defensible conclusion | Next action |
|---|---|---|
| Continuous model approximates a stable useful integral, and defect routing beats strong discrete and same-function controls with confirmed end-to-end benefit | Evidence for the specified continuous-resolution/controller combination in the tested local regime | Confirm at a second meaningful scale or independent natural-text population; sharpen, do not broaden, the mechanism claim. |
| Continuous model helps quality, but static integration is as good or faster than adaptation | Continuous parameterization or integral activation may be useful; sequential adaptation is unnecessary locally | Investigate the inductive bias with capacity/training controls. Do not claim token-adaptive superiority. |
| Discrete defect or learned routing matches the gain | Benefit is generic adaptive computation rather than continuity | Report the simpler discrete solution and the attribution finding. |
| Factorized or segment evaluation erases apparent speedup | Original implementation incurred avoidable numerical cost | Use the compiler path; classify any contribution as implementation analysis, not a novel continuum speed principle. |
| Only a fixed training grid works well | Evidence for a discretized architecture, not robust function-space behavior | Diagnose quadrature overfitting; a redesigned training rule is a new experiment. |
| Oracle helps, but Python/MPS routing loses | Heterogeneity exists; selected runtime cannot exploit it efficiently | Consider one-shot routing, coarser per-request selection, or a paired framework/kernel optimization only if the measured headroom pays for it. |
| Synthetic integrands or TinyStories succeed, natural text does not | Mechanism demonstration without the intended application benefit | Retain the negative natural-text finding; do not generalize synthetic success. |
| Broad confidence intervals or few independent articles/seeds | Inconclusive at achieved precision | Report sensitivity and a separately designed follow-on only when its information gain is worth measured cost. |
| Strong, stable local disadvantage after fair tuning | Evidence against the selected design in the stated regime | Stop this branch or select a specifically motivated refinement; no “it must work at scale” escape clause. |
| Hardware/asset gate blocks the claim | Prerequisite or resource limitation | Preserve all completed evidence and explicitly state the untested claim. |

### 12.2 Prioritized refinements, not an undifferentiated menu

**First alternative: one-shot control.** Trigger only when the offline resolution oracle has useful headroom and sequential defects fail mainly through repeated projections/synchronizations. Reuse B3's common router so that the extra architectural degree of freedom is isolated. This changes the selected contribution from numerical stopping to continuous representations under a learned router; record that pivot.

**Second alternative: static integral activation.** Trigger when the continuous model improves quality but the compiled reference is faster than adaptive sampling. Study the integral-induced nonlinearity against matched dense, box-basis, and low-rank controls. This preserves continuous parameter sharing but abandons per-token numerical refinement; it has substantial IAT prior-art overlap. [R6]

**Third alternative: richer local basis or learned mesh.** Trigger only when quantitative residuals show that linear hats are the limiting representation and fixed higher-order integration has affordable cost. Search the exact refinement again against spline and integral-network literature before attributing novelty. A denser kernel generator or SIREN is not a default rescue: it may destroy the cheap compiler control while merely adding parameters.

**Do not silently pivot to PDEs.** Physical-domain operator learning may be an excellent separate project, but a positive PDE surrogate benchmark answers a different question from continuously addressable latent neurons in a language model. The original intent must remain visible in the final report.

### 12.3 Optional pretrained branch: exact architectural preservation

The inspected `SmolLM2-135M` configuration has 30 layers, residual width 576, 9 attention heads, 3 KV heads, FFN intermediate size 1536, vocabulary 49,152, tied embeddings, SiLU activation, context limit 8192, RoPE base 100,000, and RMSNorm epsilon `1e-5`. Its stored dtype is BF16; that is not proof that BF16 training is the right M3 path. Load and verify an unmodified FP32 comparison first if memory permits. [R27]

Its gated FFN cannot be replaced by the two-field GELU expression without changing the architecture. Use three coefficient fields:

\[
F(h)=g\int_0^1v(s)\,\operatorname{SiLU}(a(s)^\top h)\,[r(s)^\top h]ds,
\]

with the checkpoint's actual bias conventions. Preserve the checkpoint's residual and normalization semantics. Coefficient-space factorization and segment Gauss integration remain available, while the specific GELU primitive in §4.5 does not apply. Validate any claimed analytical shortcut separately.

Initialize fields from the shared checkpoint under a documented ordering, measure conversion loss before adaptation, and give discrete elastic competitors the same checkpoint, calibration information, and repair budget. All preprocessing and tokenization stay faithful to the original asset. A reduction in adaptation quality caused by changing the tokenizer or gating operation would be an invalid comparison.

This branch can test a small pretrained model locally; it cannot establish billion-parameter scaling, general assistant quality, or superiority to current frontier LLMs. Use the same causal scoring and uncertainty principles, and disclose potential pretraining overlap with the evaluation corpus.

## 13. Audit record and remaining risks

These are **self-audits of the researched plan**, not independent peer review and not executed M3 validation. Three distinct challenge passes led to the revisions below.

### 13.1 Scientific and prior-art audit

| Defect identified | Change made to the plan | Residual risk / acceptance evidence |
|---|---|---|
| The initial intuition could be mistaken for a new architecture class despite INN, IAT, and neural operators. | Narrowed the contribution to the controlled continuum-versus-discrete question and explicitly labelled the FFN-only restriction. | Exact controller combination may still have prior art; no universal novelty or patentability claim. |
| Continuous coordinates plus adaptive computation already appear in SplineNets and flexible Transformers. | Added crossed parameterization/controller controls and a one-shot baseline. | Local Flextron adaptation is not full large-model reproduction; claims are scoped accordingly. |
| A finite basis permits cheaper equivalent computation. | Derived and tested factorization, added segment integration and analytic GELU as mandatory opponents. | MPS stability and fastest dispatch remain unmeasured gates. |
| A small defect is not a certificate, even for linear-hat fields. | Added the explicit zero-defect/positive-integral counterexample and false-safe analysis. | Distributional calibration can fail under shift; no certified claim is allowed. |
| Numerical convergence can conflict with task optimization. | Added independent integration, off-grid rules, and quadrature-overfitting diagnostics. | A useful grid-dependent model may not support the intended continuum interpretation. |
| A favorable microscopic result would not establish LLM utility. | Required natural text, end-to-end requests, compiler controls, and an optional explicitly separate pretrained branch. | Larger-scale transfer remains open after a small local study. |

### 13.2 Experimental-validity audit

| Defect identified | Change made | Residual risk / acceptance evidence |
|---|---|---|
| Weak old elastic training could manufacture a gain. | Corrected MatFormer v2 to one sampled granularity per step; shared stronger training options and static layer allocation. | Search cannot guarantee a global optimum; record boundaries and tuning spend. |
| Router or oracle labels could leak unavailable information. | Defined the inference input contract and confined references/future-layer information to charged offline training. | Code-level assertions and causal parity tests must verify the firewall. |
| Same token count can conceal more optimization compute. | Separated primary project-cost comparison from matched-token attribution. | Equal wall time depends on hardware/system conditions; retain actual measured expenditures. |
| Token rows and repeated timings could be counted as independent samples. | Article-cluster and crossed seed analysis; session-block timing uncertainty. | Few test articles or seeds may still leave the result inconclusive. |
| An adaptive search could select a favorable budget or protected checkpoint. | Baseline-defined `B*`, frozen selection, one primary contrast, protected data, and amendment/fresh-confirmation rules. | Development overfitting remains possible; independent confirmation is required for a strong result. |
| A mean speed gain could conceal routing stalls, memory, or cold preparation. | Added tail metrics, cache memory, cold cost, and amortization. | Practical usefulness depends on the actual deployment request mix. |
| Synthetic success might be presented as realistic success. | WikiText is the natural-text core; TinyStories is labelled synthetic and optional. | WikiText is still a narrow domain, not a general assistant benchmark. |

### 13.3 Implementation and feasibility audit

| Defect identified | Change made | Residual risk / acceptance evidence |
|---|---|---|
| Nominal Apple Silicon support does not establish required kernels. | Added forward/backward/optimizer probes, fallback tracing, and faithful whole-comparison fallback rules. | Actual M3 paths remain untested until WP02. |
| GPU timing could measure submission rather than completion. | Explicit boundary synchronization, warmup, lazy-evaluation handling, and randomized sustained sessions. | Thermal and background-process variation must be measured, not assumed absent. |
| Non-square causal masking can break cached decoding. | Specified absolute-position masks or the valid one-query no-future case and parity tests. | Library changes require rerunning the probe. |
| INN interpolation helpers may ignore nonuniform node coordinates. | Use explicit hat interpolation and node/weight tests rather than a size-only resize path. | A port still needs gradient, boundary, and conversion checks. |
| AWN code and WikiText metadata were not straightforward permissive assets. | Added explicit licensing gates and nonredistribution defaults; no silent resolution of conflicting metadata. | Unclear terms can block the relevant comparison; no legal conclusion is offered. |
| Unknown RAM or large activations could defeat a nominally small parameter count. | Full memory model, measured allocator/RSS/swap, streaming references, microbatch-first adaptation. | A very constrained machine may support only a conditional small-scale claim. |
| Resume could reuse stale sampled weights or an altered training stream. | Atomic checkpoints, complete sampler/RNG state, cache reconstruction, and canary checks. | Bitwise MPS determinism is not promised. |
| Work packages could require a result from a later expensive stage before running an early gate. | Separated untrained runtime bounds, static learned models, frozen-layer diagnosis, and joint training. | Early proxy negatives must be classified by what the proxy can actually express. |

A final internal consistency pass also separated panel counts from node counts, fixed gain from resolution-dependent normalization, differentiable training paths from detached inference caches, and a numerical reference from a task-optimal finite discretization. These are concrete implementation requirements, not quality scores.

### 13.4 Explicit limitations at handoff

The plan has not established M3 throughput, a successful training recipe, a positive benchmark result, a valid universal error bound, or exhaustive novelty. Some important papers were available only as abstracts or official slides, some repository URLs are mutable until the agent pins commits, and AWN/INSTANT integration may be constrained by licensing or fidelity. The patent reconnaissance was limited and is not a freedom-to-operate assessment.

The CPU checks establish only the listed mathematical implementation facts and counterexample. The planned study can support a local empirical claim or a useful negative finding. It cannot by itself establish practical utility for an unspecified customer, broad reasoning improvements, foundation-model scaling laws, or legal clearance.

## 14. References and inspection ledger

References use ordinary public URLs and identifiers so this document stands alone. Access date for web resources is **13 October 2026**. A mutable repository URL identifies the inspected location; execution must save a full commit SHA and asset hash. “Inspected” does not mean all experiments in a source were independently reproduced.

**[R1] Neural operators.** Nikola Kovachki and colleagues, *Neural Operator: Learning Maps Between Function Spaces With Applications to PDEs*, Journal of Machine Learning Research 24(89), 2023. [JMLR article](https://www.jmlr.org/papers/v24/21-1524.html). Inspected publication page and abstract. Establishes function-space operator framing and its PDE context; does not establish latent-width language-model gains.

**[R2] Fourier Neural Operator.** Zongyi Li and colleagues, *Fourier Neural Operator for Parametric Partial Differential Equations*, arXiv:2010.08895, October 2020; ICLR 2021. [Record](https://arxiv.org/abs/2010.08895). Inspected abstract and metadata. Structured Fourier operator is relevant historical context, not a required FFT baseline for the selected FFN axis.

**[R3] DeepONet.** Lu Lu, Pengzhan Jin, and George Em Karniadakis, *DeepONet: Learning nonlinear operators for identifying differential equations based on the universal approximation theorem of operators*, arXiv:1910.03193, December 2019, inspected revised record. [Preprint](https://arxiv.org/abs/1910.03193). Related expanded publication: Lu Lu and colleagues, *Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators*, Nature Machine Intelligence, 2021, [DOI:10.1038/s42256-021-00302-5](https://doi.org/10.1038/s42256-021-00302-5). Abstract/record inspected; Nature full text not successfully retrieved. Branch/trunk operator evaluation resembles arbitrary-coordinate outputs, not the exact proposed kernel.

**[R4] Integral Neural Networks.** Kirill Solodskikh, Azim Kurbanov, Ruslan Aydarkhanov, Irina Zhelavskaya, Yury Parfenov, Dehua Song, and Stamatios Lefkimmiatis, *Integral Neural Networks*, CVPR 2023, pp. 16113–16122. [Publication landing page](https://openaccess.thecvf.com/content/CVPR2023/html/Solodskikh_Integral_Neural_Networks_CVPR_2023_paper.html); [official conference slides](https://cvpr.thecvf.com/media/cvpr-2023/Slides/21686.pdf). The full paper retrieval failed; official slide text and screenshots, including architecture and initialization comparisons, were inspected. Direct overlap: continuously resizable weight functions and quadrature. Slide evidence distinguishes from-scratch and converted initialization; it is not a new local replication.

**[R5] TorchIntegral implementation.** TheStageAI, *TorchIntegral*, public repository, inspected 2026. [Repository](https://github.com/TheStageAI/TorchIntegral); [README](https://raw.githubusercontent.com/TheStageAI/TorchIntegral/main/README.md); [quadrature implementation](https://raw.githubusercontent.com/TheStageAI/TorchIntegral/main/torch_integral/quadrature.py); [interpolation implementation](https://raw.githubusercontent.com/TheStageAI/TorchIntegral/main/torch_integral/parametrizations/interpolation_weights.py). README, license declaration, wrapper example, quadrature, and interpolation paths inspected. Apache-2.0. Material implementation issue: coordinate-aware grid sampling and size-based interpolation are not interchangeable on nonuniform nodes. The repository's general conversion/tracing support is not assumed to cover this decoder unchanged.

**[R6] Integral Activation Transform.** Zezhong Zhang, Feng Bao, and Guannan Zhang, *Improving the Expressive Power of Deep Neural Networks Through Integral Activation Transform*, arXiv:2312.12578, December 2023; International Journal of Numerical Analysis and Modeling 21(5), 739–763, 2024. [Full preprint HTML](https://arxiv.org/html/2312.12578v1); [publication record](https://www.global-sci.com/ijnam/article/view/10588); DOI:10.4208/ijnam2024-1030. Full HTML formulation and publication metadata inspected. Relevant finite-basis reduction, box-basis relation, and analytical activation integration; does not establish the proposed M3 routing result.

**[R7] SplineNets.** Cem Keskin and Shahram Izadi, *SplineNets: Continuous Neural Decision Graphs*, arXiv:1810.13118, October 2018. [Full HTML](https://arxiv.org/html/1810.13118). Method and implementation discussion inspected. Continuous spline weight manifolds and conditional selections directly overlap the broad idea; CNN experiments and conditional-computation costs differ from this LM setting.

**[R8] MatFormer.** Devvrit and colleagues, *MatFormer: Nested Transformer for Elastic Inference*, arXiv:2310.07707, initially October 2023; inspected **v2, 15 December 2024**, NeurIPS 2024. [Full v2 HTML](https://arxiv.org/html/2310.07707v2); [project page](https://devvrit.github.io/matformer/). Training and elastic/MixMatch mechanisms inspected. Crucial version detail: one sampled granularity per optimization step in v2. Large-model published training exposure must not be treated as equal-cost local evidence.

**[R9] Flextron.** Ruisi Cai, Saurav Muralidharan, Greg Heinrich, Hongxu Yin, Zhangyang Wang, Jan Kautz, and Pavlo Molchanov, *Flextron: Many-in-One Flexible Large Language Model*, arXiv:2406.10260, June 2024. [Full HTML](https://arxiv.org/html/2406.10260v1). Elastic preparation, routing, surrogate, and training sections inspected. Input/budget routing is prior art. The proposed controlled FFN-only router protocol is not represented as a reproduction of full attention-and-MLP Flextron.

**[R10] Adaptive Width Neural Networks.** Federico Errica, Henrik Christiansen, Viktor Zaverkin, Mathias Niepert, and Francesco Alesiani, *Adaptive Width Neural Networks*, arXiv:2501.15889, January 2025; inspected **v5, 16 February 2026**, ICLR 2026. [Record](https://arxiv.org/abs/2501.15889); [v5 HTML](https://arxiv.org/html/2501.15889v5). Method and updated record inspected. Width learning via decaying importance and variational training is relevant but not identical to inference-time numerical defects.

**[R11] AWN code and license.** NEC Research, *Adaptive-Width-Neural-Networks*. [Repository](https://github.com/nec-research/Adaptive-Width-Neural-Networks); [README](https://raw.githubusercontent.com/nec-research/Adaptive-Width-Neural-Networks/main/README.md); [license text](https://raw.githubusercontent.com/nec-research/Adaptive-Width-Neural-Networks/main/LICENSE.txt). README, license, and dynamic Transformer implementation inspected. The code is under restrictive research-use terms, not an assumed permissive open-source license; redistribution and derivative-use conditions must be checked. Its Transformer is not a ready-made decoder-only replacement.

**[R12] Universally Slimmable Networks.** Jiahui Yu and Thomas Huang, *Universally Slimmable Networks and Improved Training Techniques*, ICCV 2019; arXiv:1903.05134. [Record](https://arxiv.org/abs/1903.05134). Abstract and metadata inspected. Arbitrary-width training, sandwich techniques, and in-place distillation provide relevant shared training controls; primary setting is vision.

**[R13] DynaBERT.** Lu Hou and colleagues, *DynaBERT: Dynamic BERT with Adaptive Width and Depth*, NeurIPS 2020; arXiv:2004.04037. [Record](https://arxiv.org/abs/2004.04037). Abstract and record inspected. Width/depth adaptation and distillation are established alternatives; encoder experiments are not comparable decoder results.

**[R14] Fast Feedforward Networks.** Peter Belcak and Roger Wattenhofer, *Fast Feedforward Networks*, arXiv:2308.14711, August 2023. [Paper HTML](https://arxiv.org/html/2308.14711v2); [official code](https://github.com/pbelcak/fastfeedforward). Paper and repository/API/license inspected. MIT code supports a practical conditional-computation screen; published theoretical or isolated speed factors are not M3 LM measurements.

**[R15] INSTANT.** Renan Beran Kilic, Kasim Sinan Yildirim, and Giovanni Iacca, *INSTANT: Inference-Aware Fast Feedforward Networks*, ACM Transactions on Embedded Computing Systems 25(4), article 62, 2026. [DOI:10.1145/3815117](https://doi.org/10.1145/3815117); [official repository](https://github.com/DIOL-UniTN/INSTANT). Repository, abstract/publication metadata, and training entry-point documentation inspected; full publisher text was not retrieved. Inference-aware training is a relevant recent improvement. A reusable root license was not established during inspection, so copying code requires a gate.

**[R16] Mobile integral reparameterization.** Jinhua Lin, Xin Li, Bowen Ren, Xiangdong Hao, Quan Wang, and Yu Yao, *Lightweight Reparameterizable Integral Neural Networks for Mobile Applications*, IEEE Transactions on Neural Networks and Learning Systems 37(2), 809–821, 2026; online publication 27 October 2025. [DOI:10.1109/TNNLS.2025.3612265](https://doi.org/10.1109/TNNLS.2025.3612265); [PubMed record](https://pubmed.ncbi.nlm.nih.gov/41144416/). Abstract and metadata inspected; full paper/code not verified. Resembles inference-oriented integral compilation and limits broad novelty claims; no controlled local comparison is inferred.

**[R17] Gated FFNs.** Noam Shazeer, *GLU Variants Improve Transformer*, arXiv:2002.05202, February 2020. [Record](https://arxiv.org/abs/2002.05202). Abstract/record inspected. Supports including a gated dense FFN as a strong simple architectural alternative rather than comparing only to one activation family.

**[R18] Deployed elastic context.** Google, *Gemma 3n* official documentation and developer guide, 2025. [Documentation](https://ai.google.dev/gemma/docs/gemma-3n); [developer guide](https://developers.googleblog.com/en/introducing-gemma-3n-developer-guide/). Official documentation inspected. Deployment context for MatFormer-related elasticity; not a controlled head-to-head benchmark or a blanket license assertion.

**[R19] PyTorch MPS and numerical APIs.** PyTorch project, documentation and implementation inspected 2026. [MPS backend](https://docs.pytorch.org/docs/2.14/notes/mps.html); [synchronize](https://docs.pytorch.org/docs/2.14/generated/torch.mps.synchronize.html); [recommended memory](https://docs.pytorch.org/docs/2.14/generated/torch.mps.recommended_max_memory.html); [GELU](https://docs.pytorch.org/docs/2.14/generated/torch.nn.GELU.html); [MPS fallback source](https://raw.githubusercontent.com/pytorch/pytorch/main/aten/src/ATen/mps/MPSFallback.mm). Documentation and fallback implementation inspected. Documentation version paths and source `main` are not a claim that any particular wheel has been installed or tested on the target.

**[R20] Attention masking.** PyTorch project, `scaled_dot_product_attention` documentation, inspected 2026. [API](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html). Mask semantics and evaluation-dropout behavior inspected. Motivates explicit non-square cached-decoding parity tests; does not establish MPS kernel availability.

**[R21] MLX.** Apple ML research, MLX installation and lazy-evaluation documentation, inspected 2026. [Installation](https://ml-explore.github.io/mlx/build/html/install.html); [lazy evaluation](https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html). Official requirements and execution semantics inspected; retrieved documentation identified release 0.32.3. No MLX implementation or target performance was tested here.

**[R22] Adaptive quadrature.** Walter Gander and Walter Gautschi, *Adaptive Quadrature—Revisited*, BIT Numerical Mathematics, 2000. [DOI:10.1023/A:1022318402393](https://doi.org/10.1023/A:1022318402393). Publication abstract/method description inspected. Establishes the historical numerical context; it does not supply a universal certificate for learned hidden-neuron fields.

**[R23] Quadrature and neural optimization error.** Santiago Badia and Kishore Nori, *Adaptive anisotropic composite quadratures for residual minimisation in neural PDE approximations*, arXiv:2605.00308v1, 1 May 2026. [Full HTML](https://arxiv.org/html/2605.00308). Abstract, introduction, error framework, and adaptive-method sections inspected. The source concerns PDE residual losses. Its warning about optimizing an inaccurate numerical functional motivates, by analogy, the independent-integration audit here; its PDE conclusions are not asserted to prove LM behavior.

**[R24] Learned quadrature.** Costas Smaragdakis, *Learning Geometric-Aware Quadrature Rules for Functional Minimization*, arXiv:2508.05445v1, 7 August 2025. [Full HTML](https://arxiv.org/html/2508.05445v1). Abstract and QuadrANN formulation inspected. Geometry-conditioned quadrature weights on point clouds are adjacent prior art, not the same latent-width/controller mechanism.

**[R25] WikiText.** Salesforce, *WikiText* dataset card and README, inspected 2026. [Dataset](https://huggingface.co/datasets/Salesforce/wikitext); [README with metadata](https://huggingface.co/datasets/Salesforce/wikitext/blob/main/README.md). Configurations, split row/byte metadata, and conflicting license statements inspected. The original external blog was not successfully retrieved. Execution must pin revision and preserve provenance rather than relying on current card text alone.

**[R26] TinyStories.** Ronen Eldan and Yuanzhi Li, *TinyStories* dataset card. [Dataset](https://huggingface.co/datasets/roneneldan/TinyStories). Card and CDLA-Sharing-1.0 declaration inspected. Synthetic-language diagnostic only; not required for the core natural-text study.

**[R27] SmolLM2-135M.** Hugging Face / HuggingFaceTB, *SmolLM2-135M* model card and configuration. [Model](https://huggingface.co/HuggingFaceTB/SmolLM2-135M); [actual config](https://huggingface.co/HuggingFaceTB/SmolLM2-135M/raw/main/config.json). Card/license declaration and full architecture configuration inspected. Weights were not downloaded or executed in this investigation.

**[R28] Target hardware reference.** Apple, *MacBook Pro (16-inch, Nov 2023) — Technical Specifications*. [Official specification](https://support.apple.com/en-us/117737). Specifications inspected as configuration context. The user's actual model, GPU core count, RAM, available disk, and operating system remain discovery tasks.

**[R29] Continuum attention.** Edoardo Calvello, Nikola B. Kovachki, Matthew E. Levine, and Andrew M. Stuart, *Continuum Attention for Neural Operators*, JMLR 26(300), 1–52, 2025. [Publication](https://www.jmlr.org/papers/v26/24-0879.html). Abstract/publication page inspected. Function-space attention is additional established context, not evidence for continuous latent-width LM acceleration.

### Search-coverage summary and blind spots

The searches covered architectural equivalents, implementation/compiler equivalents, conditional-width competitors, numerical error control, training–inference mismatch, negative initialization evidence, licenses, and target-runtime constraints. An additional targeted pass found recent neural-PDE adaptive/learned quadrature work but no inspected source that established the proposed M3 FFN comparison. This absence is not a novelty proof.

Blind spots include full text/code for the mobile reparameterizable network, full INSTANT methodology/license, full-paper INN details beyond slides and code, unpublished/internal systems, unindexed recent preprints, and a comprehensive patent search. The broad mathematical, continuous-width, and token-routing novelty claims are already contradicted by inspected prior art; the remaining narrow empirical question is deliberately tested rather than presumed original.

## Appendix A. Numerical conventions and local test acceptance

The implementation of `clencurt01` in Appendix B is the complete reference for the initial quadrature rule. For even power-of-two `N`, interior weights can equivalently be written

\[
w_j=\frac1N\left[1-\sum_{k=1}^{N/2-1}\frac{2\cos(2k\theta_j)}{4k^2-1}
-\frac{\cos(N\theta_j)}{N^2-1}\right],\quad \theta_j=\pi j/N,
\]

with the endpoint weights specified in §4.3. Do not substitute the weights for a rule on `[-1,1]` without the factor-of-two domain transformation.

`hat_basis` is a dense correctness implementation. Production kernels use its two-entry sparse structure, not a dense basis tensor unless that is actually the fastest measured small-shape path. The reference uses exact Gaussian GELU, and all arrays in the CPU comparison are FP64. Gain and output bias are omitted in the small algebraic test because the same fixed gain and bias preserve the equivalence.

Start production parity tests with output tolerance `atol=1e-5, rtol=5e-5` for representative FP32 outputs, and gradient tolerance `atol=1e-5, rtol=1e-3`; supplement elementwise checks with scale-aware RMS and worst-case errors. These are initial implementation-validation tolerances, not statistical success criteria and not guaranteed adequate for every shape. Investigate failures rather than relaxing tolerances until tests pass. When the estimated benchmark effect approaches the numerical discrepancy, tighten the implementation or classify the effect as unresolved.

For CPU reference convergence, compare successive segment-Gauss orders using `atol=1e-12, rtol=1e-10` on small reference problems; verify the same result through the analytical moments where stable. For trained models, additionally compare induced token losses so that a tiny layer error is not mistaken for a guaranteed tiny task effect. The highest training quadrature level must be checked against this reference; it is not declared correct simply because it is the largest available level.

The original script below writes `math_checks.json` next to itself. Save it as `checks_math.py` and run the existing command `python checks_math.py` in the CPU reference environment. Its output will report the **new execution environment**, which may differ from the environment in §5. Passing it on the target remains a CPU numerical check unless separate MPS tests are run.

## Appendix B. Executed CPU reference check, reproduced in full

```python
"""Small CPU/FP64 checks for the proposed research plan; not an M3 benchmark."""
from __future__ import annotations
import json, math, platform, sys
from pathlib import Path
import numpy as np
import scipy
from scipy.special import ndtr
from numpy.polynomial.legendre import leggauss
import torch


def clencurt01(n: int) -> tuple[np.ndarray, np.ndarray]:
    if n < 2 or n & (n - 1):
        raise ValueError('n must be a power of two >= 2')
    theta = np.pi * np.arange(n + 1) / n
    x = (1 - np.cos(theta)) / 2
    w = np.zeros(n + 1, dtype=np.float64)
    w[0] = w[-1] = 1 / (2 * (n*n - 1))
    v = np.ones(n - 1)
    for k in range(1, n//2):
        v -= 2 * np.cos(2*k*theta[1:-1]) / (4*k*k - 1)
    v -= np.cos(n*theta[1:-1]) / (n*n - 1)
    w[1:-1] = v / n
    return x, w


def hat_basis(x: np.ndarray, k: int) -> np.ndarray:
    if k < 2 or np.any((x < 0) | (x > 1)):
        raise ValueError('invalid knot count or coordinate')
    u = x * (k - 1)
    left = np.minimum(np.floor(u).astype(int), k-2)
    frac = u - left
    b = np.zeros((len(x), k), dtype=np.float64)
    b[np.arange(len(x)), left] = 1-frac
    b[np.arange(len(x)), left+1] = frac
    return b


def gelu(z):
    return z * ndtr(z)


def gelu_primitives(z):
    phi = np.exp(-0.5*z*z) / np.sqrt(2*np.pi)
    cdf = ndtr(z)
    return .5*((z*z-1)*cdf + z*phi), (z**3*cdf+(z*z+2)*phi)/3


def gelu_segment_moments(p0, p1, fallback=True):
    """J0=integral G(p0+du)du; J1=integral u G(p0+du)du on [0,1]."""
    p0, p1 = np.broadcast_arrays(np.asarray(p0, float), np.asarray(p1, float))
    d = p1-p0
    small = np.abs(d) < 0.1
    safe_d = np.where(small, 1., d) if fallback else d
    f00, f10 = gelu_primitives(p0)
    f01, f11 = gelu_primitives(p1)
    j0 = (f01-f00)/safe_d
    j1 = ((f11-f10)-p0*(f01-f00))/(safe_d*safe_d)
    if fallback and np.any(small):
        t,w=leggauss(16); u=(t+1)/2; w=w/2
        vals=gelu(p0[small,None]+d[small,None]*u)
        j0[small]=vals@w
        j1[small]=vals@(w*u)
    return j0,j1


def main():
    torch.set_num_threads(1)
    torch.manual_seed(1013)
    rng=np.random.default_rng(1013)
    out={'environment': {'python':sys.version.split()[0], 'platform':platform.platform(),
         'machine':platform.machine(), 'numpy':np.__version__, 'scipy':scipy.__version__,
         'torch':torch.__version__, 'mps_available':torch.backends.mps.is_available()}, 'seed':1013}
    moments=[]
    for n in [2,4,8,16,32,64,128]:
        x,w=clencurt01(n)
        moment_err=max(abs(np.dot(w,x**p)-1/(p+1)) for p in range(min(n,10)+1))
        nest_err=np.max(abs(clencurt01(2*n)[0][::2]-x))
        moments.append({'panels':n,'mass_error':float(abs(w.sum()-1)),
                        'max_tested_monomial_error':float(moment_err), 'nesting_error':float(nest_err),
                        'min_weight':float(w.min())})
        assert moment_err < 2e-14 and nest_err < 2e-15 and np.all(w>0)
    out['clenshaw_curtis']=moments
    x,w=clencurt01(64)
    psi=torch.tensor(hat_basis(x,17)); wt=torch.tensor(w)
    hh=torch.randn(5,8,dtype=torch.float64,requires_grad=True)
    aa=torch.randn(17,8,dtype=torch.float64,requires_grad=True)
    vv=torch.randn(17,8,dtype=torch.float64,requires_grad=True)
    bb=torch.randn(17,dtype=torch.float64,requires_grad=True)
    def sampled(h,a,v,b):
        z=h@(psi@a).T + psi@b
        return (torch.nn.functional.gelu(z,approximate='none')*wt)@(psi@v)
    def factored(h,a,v,b):
        z=(h@a.T+b)@psi.T
        return ((torch.nn.functional.gelu(z,approximate='none')*wt)@psi)@v
    ys=sampled(hh,aa,vv,bb); yf=factored(hh,aa,vv,bb)
    gs=torch.autograd.grad(ys.square().sum(),(hh,aa,vv,bb),retain_graph=True)
    gf=torch.autograd.grad(yf.square().sum(),(hh,aa,vv,bb))
    yerr=float((ys-yf).abs().max().detach())
    gerr=max(float((s-f).abs().max()) for s,f in zip(gs,gf))
    out['basis_factorization']={'batch':5,'d':8,'knots':17,'nodes':65,
             'max_abs_output_error':yerr,'max_abs_gradient_error':gerr,
             'gradcheck':bool(torch.autograd.gradcheck(factored,(hh,aa,vv,bb),atol=1e-5,rtol=1e-4))}
    assert yerr<1e-12 and gerr<1e-11
    # All intervals integrate an affine vector times an activated affine scalar.
    p0=rng.uniform(-5,5,1000); p1=rng.uniform(-5,5,1000)
    p1[:5]=p0[:5]+np.array([0.,1e-8,1e-6,1e-4,1e-2])
    u,qw=leggauss(128); u=(u+1)/2; qw=qw/2
    vals=gelu(p0[:,None]+(p1-p0)[:,None]*u)
    j0r=vals@qw; j1r=vals@(qw*u)
    j0,j1=gelu_segment_moments(p0,p1)
    out['gelu_segment_moments']={'intervals':1000,'endpoint_range':[-5,5],
              'small_slope_fallback':'16-point Gauss-Legendre when |p1-p0| < 0.1',
              'reference':'128-point Gauss-Legendre',
              'max_abs_J0_error':float(abs(j0-j0r).max()),
              'max_abs_J1_error':float(abs(j1-j1r).max())}
    assert np.max(abs(j0-j0r))<1e-10 and np.max(abs(j1-j1r))<1e-10
    # ReLU primitive, splitting its single possible root on each segment.
    dp=p1-p0
    alpha=np.zeros_like(p0); beta=np.ones_like(p0)
    neg=(p0<=0)&(p1<=0); beta[neg]=0
    inc=(p0<0)&(p1>0); alpha[inc]=-p0[inc]/dp[inc]
    dec=(p0>0)&(p1<0); beta[dec]=-p0[dec]/dp[dec]
    rj0=p0*(beta-alpha)+.5*dp*(beta**2-alpha**2)
    rj1=.5*p0*(beta**2-alpha**2)+dp*(beta**3-alpha**3)/3
    ur=alpha[:,None]+(beta-alpha)[:,None]*u
    vr=np.maximum(p0[:,None]+dp[:,None]*ur,0)
    rr0=(beta-alpha)*(vr@qw)
    rr1=(beta-alpha)*np.sum(vr*ur*qw,axis=1)
    out['relu_exact_segments']={'intervals':1000, 'max_abs_J0_error':float(abs(rj0-rr0).max()),
                                'max_abs_J1_error':float(abs(rj1-rr1).max())}
    # A triangle supported entirely between nodes, representable by 101 uniform knots.
    def spike(s):return np.maximum(1-np.abs(s-.60)/.01,0)
    qc=[]
    for n in [4,8]:
        xn,wn=clencurt01(n); qc.append(float(wn@spike(xn)))
    out['defect_counterexample']={'panels':[4,8],'quadrature_values':qc,
                  'embedded_defect':abs(qc[1]-qc[0]),'true_integral':.01,
                  'construction':'triangle with vertices (0.59,0), (0.60,1), (0.61,0); 101 uniform knots suffice'}
    assert qc==[0.,0.]
    Path(__file__).with_name('math_checks.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))

if __name__=='__main__': main()
```

## Appendix C. Completion checklist for the coding agent

Before declaring any terminal result, verify that the report names the exact claim tested; every headline comparison has a valid baseline and cost boundary; all protected-data amendments are visible; all required controls for the surviving explanation were run; and no failed or unfavorable configuration was silently discarded.

Before declaring a positive contribution, additionally verify that the candidate beats the relevant continuous-static and same-function compiler controls, or explicitly narrow the claim to what does survive; that discrete elasticity and routing received fair treatment; that the measured effect is distinguishable from article, seed, timing, and numerical uncertainty; and that an independent confirmation supports the stated scope.

Before declaring a negative result, verify that the implementation is correct, the proxy expresses the relevant bottleneck, the baseline and candidate were trained competently, and the achieved sensitivity can distinguish no advantage from insufficient evidence. Report hardware limitations, unsuitable proxies, invalid measurements, and genuine negative findings as different outcomes.

**The project is complete when the evidence supports an interpretable decision—not when a positive result has been obtained.**
