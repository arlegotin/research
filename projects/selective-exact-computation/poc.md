# Selective Exact Computation: research and local proof-of-concept plan

**Research date:** 30 October 2026.  
**Target:** one MacBook Pro with an M3 Max; actual memory, free storage, operating system, and GPU configuration must be discovered.  
**Deliverable status:** researched, audited implementation plan; no M3 performance experiment has been executed.  
**Primary direction:** certified stochastic output-head screening with a packed low-bit sketch, selective reads of original weight rows, and numerical refinement only when a decision remains ambiguous.

**Navigation:** [Mechanism](#4-mathematical-mechanism-and-implementation-specification) · [Evaluation](#5-fair-evaluation-and-protected-confirmation) · [M3 feasibility](#6-m3-max-feasibility-and-adaptive-configuration) · [Work packages](#7-ordered-work-packages-and-decision-gates) · [Interfaces and records](#9-repository-interfaces-experiment-records-and-resumability) · [Audits](#11-audit-record-unresolved-risks-and-limits-of-the-handoff) · [Sources](#12-sources-and-search-coverage-record).

## 1. Executive assessment and selected direction

Proceed with a bounded research PoC. The supplied work establishes a credible mathematical mechanism and a useful small diagnostic, but it does **not** establish faster inference, smaller models, or an original general sampling principle. The strongest defensible project is to determine whether certified selective evaluation can reduce **actual output-head latency** against optimized dense and fused sampling on an M3 Max, while preserving a precisely defined full-softmax conditional distribution. A negative performance result would be scientifically useful if the implementation is correct and the cost attribution is complete.

The broad ingredients already exist: lower-envelope squeezing, bounded Gumbel search, lazy log-linear sampling, exact decisions from approximate probabilities, and adaptive precision. More recent antecedents include fused dense head sampling and a public implementation describing a certified low-bit greedy head with selective full-row fallback. Interval refinement for Gumbel noisy argmax and quantile lookup hardware also predate this project. Do not claim that combining an approximate screen with exact correction, using Gumbels, or using a lookup table is itself a discovery. [R01], [R02], [R03], [R04], [R05], [R08], [R10], [R12]

### 1.1 The decision

Implement **a certified Gumbel shortlist over an unchanged pretrained output head**, starting with native MLX six-bit affine packs and group size 64. Treat six bits as an evidence-based starting configuration, not a predetermined winner. Recompute residual certificates for the actual emitted packed representation. Use a simple auditable FP32 Metal kernel if the native quantized matrix multiplication cannot be bounded rigorously. Compare against the fastest qualified dense implementation and a dense fused Gumbel implementation that shares the same numerical improvements.

Use **SmolLM2-135M** to connect the experiment to the supplied evidence and **Qwen3-0.6B-Base** as the primary larger-head local workload. Their BF16 heads occupy 54 MiB and 296.75 MiB respectively; the latter tests substantially more bandwidth-sensitive work without requiring a large model. The fixed transformer body, input embeddings, hidden states, and KV semantics remain part of the reference. This is an output computation experiment, not a new model architecture or training project. [R25], [R26], [R27]

The supplied residual exponential race remains a required mathematical and algorithmic control. Its near-minimax distinct-query result is valuable, but Gumbel screening is the primary systems path because selection can stay in logit space, avoiding exponentiation and subtraction of nearly equal positive masses. This is an explicit refinement of the supplied engineering idea, which already included both constructions. It preserves the central intent: use inexpensive information to identify the expensive computations necessary for one exact random output.

### 1.2 What could be contributed

| Candidate contribution | Present status | Discriminating evidence required |
|---|---|---|
| A useful certified M3 output-head implementation | Plausible engineering contribution; unbuilt and unmeasured | Correct enclosures and same-tape decisions; complete latency reduction against qualified dense/fused baselines |
| A certificate, traffic, and latency frontier across real head states | Plausible new empirical result | Protected workloads; full metadata and fallback accounting; negative configurations retained |
| The supplied finite-box additive-one minimax theorem | Internally sound under its stated model; historical originality unresolved | Explicit oracle model, independent proof check, and further priority review before a novelty publication |
| A block-cost extension of that theorem | Elementary deduction specified in Section 10; not a major novelty claim | Proof plus comparison with ordinary coalesced row reads, which have the same block-touch probabilities |
| A new general exact-sampling or quantization principle | Not supported | Existing prior art already covers the broad principle |
| A smaller complete model, a superior general four-bit compressor, or a universal inference speedup | Not supported and outside the primary claim | Separate matched-quality, full-system evidence that this project does not presently possess |

The exact question is falsifiable. If valid bounds are too loose, the screen is too expensive, the dense baseline is already faster, or numerical refinement dominates, report that result. If only a component improves, claim only that component improvement. If the strict GPU path cannot be justified, a certified CPU diagnostic is still executable locally, but cannot establish fast exact GPU inference.

### 1.3 Evidence labels used throughout

- **Inspected evidence:** source documents, code, metadata, or preserved supplied outputs examined during this research.
- **Measured now:** small checks actually executed in a Linux x86_64 environment, identified in Section 2. They are not M3 measurements.
- **Deduction/estimate:** calculations and proposed derivations, with assumptions stated.
- **Proposed:** work for the coding agent; no outcome is presumed.
- **Blocking unknown:** must be resolved before a dependent claim or experiment proceeds.

All project commands beginning `python -m sec` below are **proposed commands that the coding agent must implement**. They do not exist in a published package. Existing operating-system commands and verified library interfaces are identified separately. This document contains the essential mathematics, algorithms, tests, interfaces, and decision rules; access to the original conversation or intermediate research notes is unnecessary.

## 2. Problem, supplied evidence, scope, and hypotheses

### 2.1 Task and beneficiaries

The immediate beneficiaries are developers of local autoregressive inference engines, especially when a large vocabulary makes the output projection material at batch size one. A standard decoding step computes a final hidden vector, reads every output-weight row to produce all logits, and samples one token. The proposed method still stores all original rows, but scans a smaller sketch and reads original rows selectively.

The bottleneck is **hypothesized for the target laptop**, not established by the source experiment. Measure the output head plus sampling fraction of full decoding before making end-to-end claims. The same mechanism could apply to other large categorical linear heads; no evidence here establishes benefits for arbitrary neurons, hidden layers, attention, training, or agent orchestration. The separate attention-representation theorems in the supplied report are not part of this PoC.

### 2.2 Precise target

Let:

- `V` be vocabulary size; `d` the hidden dimension; `B` the request batch size, with primary `B=1`.
- `W ∈ R^(V×d)` be the **exact dyadic values represented by the original checkpoint's BF16 head**; `h ∈ R^d` be the exact finite values emitted by the frozen body and promoted losslessly to FP32 where applicable.
- `z_i = Σ_j W_ij h_j + bias_i`, interpreted as a real sum of those finite dyadic inputs. Both selected models have no output bias; support a finite bias in the interface and account for it if another model is admitted.
- `τ>0` be the exact dyadic value of a declared positive finite FP32 temperature; primary bits `0x3f800000`, meaning `τ=1`.
- `P(i|h)=exp(z_i/τ)/Σ_j exp(z_j/τ)` over the declared vocabulary support.

The strict target is **one token drawn from this conditional distribution in the ideal random-bit model**. It is not the rounded probability vector produced by an unspecified framework softmax. Numerical intervals and adaptive refinement make this definition implementable without computing every logit to infinite precision.

The canonical policy field is `temperature_f32_bits`, not an ambiguous decimal. Convert a requested decimal to nearest-even FP32 once during configuration, record its original text and the resulting bits/exact dyadic value, and use those bits in every matched method. Reject zero, negative and nonfinite values after conversion. A positive subnormal temperature is a valid declared target but uses the CPU-certified path; do not flush it to zero on GPU. A different temperature representation requires an amendment and certified conversion/arithmetic, rather than a silent float64-to-float32 cast.

The body remains the same deterministic finite-precision computation in every matched comparison. Exactness of this head does not make the body an exact-real transformer, does not repair an approximate KV cache, and does not promise byte-identical tokens to a stock sampler with different random-number consumption. During autoregressive validation, identical histories and the same per-step random tapes must give identical tokens between the matched certified dense and selective heads.

Primary policy is untruncated full softmax: no top-k, top-p, min-p, repetition penalty, or implicit generation-config override. All special tokens remain in the declared support unless a fixed mask is explicitly preregistered and applied to every method. Fixed masks and known additive biases can be handled by deleting forbidden indices or transforming intervals. Greedy decoding, nucleus membership, probability reporting, and log-normalizers require different outputs/certificates and are deferred extensions.

### 2.3 Exactness must have a name

| Label | Meaning | What establishes it |
|---|---|---|
| `ideal_theorem` | Algorithm samples `P` with exact real arithmetic and independent unbiased bits | Mathematical proof under stated assumptions |
| `certified_real_head` | Actual implementation encloses real dot products and random perturbations, returning only a proved winner or continuing refinement | Audited numerical operations, certified metadata, same-tape refinement, and correct fallback; tests supplement the argument |
| `finite_reference_replay` | Identical token to a specified finite-arithmetic, finite-tape dense implementation | Exact coupling tests and a precisely specified kernel/random transformation; this alone is not the ideal-law claim |
| `empirical_only` | No mismatches observed, or small KL/TV in tests | Finite observations only; cannot be called certified or exact |

A practical counter-based PRNG is a reproducible pseudorandom realization of the bit-tape model, not a mathematical source of infinitely many independent fair bits. State this assumption for both dense and selective methods. Never use a successful chi-square test or a million matching samples as a proof of exactness. [R04], [R05], [R32]

### 2.4 Audit of the supplied assets

Source S1 is the 30 October 2026 report, *Beyond fixed precision: Exact attention representations and selective categorical sampling*. Source S2 is its accompanying reproducible ZIP. Their identifiers are in Section 12. All **38 manifest entries** in the ZIP matched their declared sizes and SHA-256 hashes. The separately attached PDF and bundled PDF differ in bytes but have identical extracted text. This checks internal file consistency, not independent truth of the historical execution claims.

Inspected files included the report, README, sampling notes, all three sampler implementations and saved results, `research_real_head/evaluate_head.py`, `validate_reference.py`, downloader, model config/provenance/checksums, and saved hidden states. The checkpoint itself was absent. The hidden-state array has shape `(29,576)`, float64, and roundtrips exactly through float32, consistent with promoted reference states.

The supplied real-head result used 29 correlated prefix states from eight disclosed prompts. Its values are **analytic expectations evaluated on measured bounds**, not measured selective row traffic or wall-clock timings:

| Sketch and group size | Mean bound width parameter `ε_i` | Mean predicted original-row queries | Logical traffic / dense BF16 |
|---|---:|---:|---:|
| 4 bits, group 64 | 20.69697 | 49,021.9187 / 49,152 | 1.2786035 |
| 6 bits, group 64 | 4.67424 | 167.6365 / 49,152 | 0.4096606 |
| 8 bits, group 64 | 1.14104 | 2.7430 / 49,152 | 0.5313058 |
| 6 bits, one group of 576 | 7.33071 | 3,341.8113 / 49,152 | 0.4464615 |
| 8 bits, one group of 576 | 1.78917 | 6.7324 / 49,152 | 0.5036092 |

The final two rows already show why minimizing query count is insufficient: whole-row eight-bit metadata gives lower logical traffic than group-64 eight-bit metadata despite more queried rows. The report's six-bit format assumes symmetric packing plus one FP16 scale and residual norm per group. The code actually materializes a dequantized FP64 sketch; no optimized six-bit sparse GPU execution is present. The original head stays stored.

Preserved files report 13,996,000 sampler draws and zero same-random-input mismatches. Those totals were checked, but those millions of draws were not rerun during this assignment. The current report's error padding, `ε*(1+1e-12)+1e-10` and a further assertion tolerance, is explicitly empirical rather than a floating-point enclosure proof. Upward rounding of a previously underestimated computed norm does not repair that gap.

### 2.5 Small validations actually performed for this plan

Environment: Linux x86_64, Python 3.12.14, NumPy 2.3.5. These checks used independent small implementations and did not modify the supplied archive.

| Check | Result measured now | Interpretation |
|---|---|---|
| Heterogeneous residual race, `V=9`, 100,000 draws | Zero coupled mismatches; mean queries 0.61164, theoretical 0.6102924, Monte Carlo SE 0.0027010 | Supports implementation of the real-arithmetic construction, not hardware certification |
| Gumbel screening over all corners of 128 five-dimensional boxes | 4,096 checks; zero omitted true maximizers, including zero-width coordinates | Validates small combinatorial cases |
| Known-coordinate Gumbel query fixture | Measured 0.16653 reads, SE 0.0011781; correct expectation `1/6` | Exposes the need to exclude known coordinates |
| Four-block residual race, 100,000 draws | Zero full/pruned winning-event mismatches; mean 0.64427 blocks, theory 0.6465781, SE 0.0027145 | Supports the block construction in Section 10 |
| Proposed dyadic Gumbel lookup geometry, 60 selected bins with Decimal precision 70 | Maximum checked noise interval width about `2.829e-4` at mantissa width 12; table arithmetic gives exactly 4 MiB | Sanity check, not a certified MPFR table or timing result |
| Exact-Fraction FP32 arithmetic emulation, 2,400 rows | Zero violations of the Section 4.9 scalar envelope under RNE/RTZ and explicit output FTZ; largest error/bound ratio 0.8436992 | Tests the arithmetic model and derivation; no Metal compiler or M3 device was exercised |

Concrete numerical failures reproduced:

1. For `l=[1,1]`, `u=[1,2]`, summing Gumbel query costs over all coordinates gives `1/3`; querying only unknown coordinates gives the correct `1/6`.
2. For `l=[a,a]`, `u=[1,2a]` with `a=1e-20`, floating subtraction `U-d_i` produces zero and an invalid negative-infinity cost estimate; the correct Gumbel expectation is `4/3-2a/(1+a)`, approximately `4/3`.
3. With log lower bounds `[-1000,-1000]` and log upper bounds `[0,0]`, max-shifted exponentiation still underflows all lower masses, giving `L=0` and an undefined baseline categorical distribution.

These are required regression fixtures. None establishes M3 compatibility or a performance improvement.

### 2.6 Hypotheses, alternatives, and falsifiers

| ID | Testable hypothesis and evidence | Plausible alternative explanation | Observation that weakens or rejects it |
|---|---|---|---|
| H1 | Correct deterministic certificates permit large reductions in original-row reads on real hidden states. Supplied six-bit diagnostic supports possibility. | Eight prompts are unusually favorable; heuristic padding is too small; a different packed quantizer widens residuals. | Valid bounds on protected states select most rows, or certified fallback is frequent. This rejects that certificate/format regime, not all possible SEC. |
| H2 | Saved reads exceed all screen, RNG, metadata, reduction, dispatch, and refinement costs. | Savings are only logical bytes; an unfused baseline or cache reuse creates the apparent gain. | Matched, synchronized complete head path is slower than qualified dense/fused implementations. |
| H3 | Certified implementation preserves the specified conditional law. | Finite tests miss rare numerical mistakes; PRNG keys are reused; fallback resamples difficult cases. | Any invalid enclosure, omitted contender, incorrect tape continuation, or demonstrated bias invalidates the strict implementation. |
| H4 | A larger head amortizes fixed overhead and permits meaningful end-to-end improvement. | Body dominates; larger vocabulary also widens bounds or increases numerical work. | Qwen head gain fails to transfer to body-interleaved decoding, or measured Amdahl headroom is negligible. |
| H5 | A particular refinement, if activated, improves the complete tradeoff. | Extra bits or a generic dense optimization gives the same improvement. | Advantage disappears against matched extra-bit, metadata, preprocessing, or fused controls. |

No hypothesis assumes a minimum percentage gain. Statistical detectability, absolute usefulness, and the cost of extra resident memory are separate questions. Scale escalation requires a specific mechanism supported by measured trends, not a general promise that larger models will eventually work.

## 3. Research landscape and novelty boundary

### 3.1 Closest prior art

| Work | Problem, mechanism, and assumptions | Evidence and computational requirements | Availability and overlap with this project |
|---|---|---|---|
| Sheldon 2013 [R01] | Discrete rejection sampling with a cheap lower squeeze; its adaptive envelope construction uses log-concavity | Density-oracle theory and algorithms; sequential proposals | Report describes C/MATLAB code, not inspected. Once bounds exist, the squeeze principle is directly reusable; neural vocabulary ordering need not be log-concave. |
| A* Sampling, Maddison et al. 2014 [R02] | Bounded optimization of Gumbel-perturbed objectives | Exact mathematical sampling with valid bounds and partitions | Establishes bounded random-perturbed search. SEC does not invent this mechanism. |
| Mussmann et al. 2017 [R03] | True top-k log-linear scores bound a tail; instantiate only relevant tail Gumbels; MIPS accelerates retrieval | Exact algorithm needs a valid tail bound. Approximate-MIPS experiments have nonzero error bounds and exclude preprocessing from headline timings. | Direct output-space ancestor. Reuse lazy sampling ideas; uncertified ANN cannot be called an exact baseline. |
| Brassard, Devroye, Gravel 2019 [R04] | Sample exactly from progressively refined approximate probabilities | Rejection decisions and finite-bit precision costs explicitly analyzed | Direct prior art for selective exactification; no identified match to the supplied distinct-coordinate box formula. |
| Haney et al. 2022 [R05] | Inverse-CDF intervals, including proposed interval Gumbel noisy argmax | Maintain noisy-value intervals until one is strictly greater; finite arithmetic matters | Direct conceptual predecessor for the numerical refinement path. Its demonstrated implementation concerns privacy noise, not this head kernel. |
| Chewi et al. 2022 [R06] | Minimax construction of rejection envelopes for shape-constrained densities | Counts preprocessing density queries for bounded rejection ratios | Related query theory, different family/objective from one sample over a known independent box. |
| AdaptiveSoftmax, Baharav et al. 2024 [R07] | Approximate top-k softmax values through adaptive entry sampling | Probabilistic accuracy; paper discusses aggressive tuning and difficulty beating BLAS with adaptivity | Approximate control/context, not an exact categorical sampler. Sample-efficiency figures are not M3 latency figures. |
| FlashSampling, Ruiz et al. 2026, v3 [R08] | Fuse dense head tiles with Gumbel maxima, avoid full logits materialization | NVIDIA experiments, FP32 score arithmetic; exact tile reduction once scores exist | Apache-2.0 CUDA/Triton implementation; algorithm is a necessary dense control. Published hardware results are not local head-to-head evidence. |
| Mausberg 2026 preprint [R09] | Exact neural sampling with certified intervals, rare precision refinement, lower-mass/residual constructions, and structural assumptions | Full cost accounting; large-rank usefulness explicitly unproven | Inspected selected lemmas and sections, not all proofs. Broad exact neural sampling is already discussed. |
| knapcio GLM runtime, September 2026 [R10] | Eight-bit head screen and certified BF16 row fallback for greedy argmax | Public README/runtime description; multi-DGX implementation | Very close engineering antecedent. Certificate source retrieval failed; correctness and performance were not independently verified. Stochastic sampling differs, but broad coarse-head fallback novelty is unavailable. |
| Streaming and lookup sampling patents [R11], [R12] | Streaming Gumbel selection; PRNG-indexed quantile lookup and comparator argmax | Hardware descriptions; the lookup embodiment uses a finite approximation | Technical prior-art signals. No legal claim interpretation, status conclusion, or freedom-to-operate opinion. |
| Softmax reparameterization 2026 [R13] | Shift every output row by a common vector before quantization | Primary-host abstract inspected through search; full primary text inaccessible | Overlaps common-mode/row-centering refinements. Do not claim centering as new or rely on uninspected experimental details. |
| SoftWater 2026 [R14] | Output-head rate allocation informed by softmax curvature and calibration | Approximate quantization; primary abstract inspected | Relevant to broad quality/compression claims, not a strict-law comparator. No local reproduction performed. |
| LQER 2024 [R15] | Low-rank approximation of quantization residual, including activation-derived scaling | Approximate output recovery; official repository includes software-emulated quantization | Low-rank residual correction is established. A certified remaining residual and real measured cost would be required for an SEC extension. |
| ARCHead 2026 [R40] | Activation-calibrated low-bit output head with low-rank residual correction | Official README inspected; optional CUDA Triton path, no local reproduction | Head-specific approximate compression already combines these ingredients. It does not establish a strict selective-read sampler. |

For FlashSampling, the inspected September 2026 v3 reports up to 10% TPOT reduction on its NVIDIA/vLLM experiments; older project descriptions give a different maximum. This plan uses neither number as a local target. Its finite-FP32 implementation should not be described as a proof of exact-real transcendental sampling. [R08]

### 3.2 Explicit novelty answers

1. **What exists?** The general selective evaluation pattern, the sampling transformations, adaptive precision, cheap head screening for greedy inference, dense sampling fusion, and low-rank correction.
2. **What can be reused?** Original model assets and body implementations; MLX packing/native dense operations; the supplied real-arithmetic sampler constructions; established counter-based randomness; MPFR-directed arithmetic; existing provenance and data loaders.
3. **What is substantively different?** Potentially the particular certified packed-head realization and its measured performance/cost frontier on Apple Silicon. The precise independent-box additive-one result remains a separate candidate formulation with uncertain historical priority.
4. **Why might it matter?** Exact stochastic selection can avoid reading most original head weights while retaining the original conditional law, if certificates are sufficiently informative and the extra work is cheaper than dense reading.
5. **What isolates the difference?** Compare selective versus full-row execution with identical numeric/RNG/fusion machinery; separately compare the fastest native dense path. Compare certificate variants with unchanged codes and equal preprocessing access.
6. **What invalidates novelty?** A prior implementation/theorem with the same operative claim, or a result explained entirely by known fusion, easier numerical semantics, extra metadata, or a weak baseline. Such a finding changes the label to replication/engineering; it does not make an honest local measurement useless.

### 3.3 Why this direction, rather than the alternatives

| Direction | Decision and technical reason |
|---|---|
| Certified Gumbel head screen | **Primary.** It preserves the proposed purpose, admits coupled winner checks, avoids probability subtraction on the hot path, and fits modest local models. |
| One-batch residual race | **Mandatory control.** Strong oracle result; tests whether minimax query efficiency translates into latency. Numerical exponential/residual arithmetic and scan overhead must be paid. |
| Ordinary squeeze rejection | **Mandatory simple control.** May use fewer distinct queries on a favorable realized target; can lose to GPU synchronization. |
| Dense fused head sampling | **Mandatory strongest simple alternative.** Could capture most available improvement without a sketch or extra model storage. |
| Greedy-only certificate | Diagnostic boundary condition; does not answer the stochastic full-softmax question. |
| ANN/MIPS with no exclusion certificate | Approximate context only. Building a large index is not the cheapest initial test and changes the guarantee. |
| Four-bit whole-model compression | A different quality/storage objective; include a tradeoff reference only if broad compression claims are pursued. |
| Attention-cache representation changes, whole-network exactification, training a new model | Outside the selected mechanism. They would create a substantially different research project. |

Research covered historical and current exact sampling, output-head quantization, numerical predicates/refinement, GPU fusion, Apple kernels, data/model provenance, and relevant public engineering/patent descriptions. The search record and access limits are in Section 12. Absence of a found identical theorem is not proof of priority.

## 4. Mathematical mechanism and implementation specification

### 4.1 Deterministic residual certificates

Partition hidden coordinates into contiguous groups of size `g`, with any final padding explicitly zero. Let `q_ij` be the emitted integer code, `s_ig` its scale, and `b_ig` its affine bias. Interpret emitted finite metadata as exact dyadic numbers:

`W_hat_ij = s_ig*q_ij + b_ig`.

Offline, compute an upward enclosure

`rho_ig >= ||W_ig - W_hat_ig||_2`.

At runtime compute upward `H_g >= ||h_g||_2`. Write the unbiased real original dot as `z_i^0=W_i·h`. Then the real sketch dot `a_i*=W_hat_i·h` satisfies

`|z_i^0-a_i*| <= eps_i = Σ_g rho_ig H_g`.

If a kernel returns `a_i` with certified arithmetic error `delta_i`, use

`z_i^0 ∈ [a_i - eps_i - delta_i, a_i + eps_i + delta_i]`.

Add the original output bias afterward with outward arithmetic to obtain the interval for `z_i=z_i^0+bias_i`; apply the same convention to fine-row evaluation. The affine quantizer bias `b_ig` is a different object. Both selected models have zero output bias, but an admitted biased head must preserve it explicitly. The Cauchy–Schwarz inequality establishes the residual part. Certificate correctness depends on the **actual decoded sketch**, not the quantizer name, nominal bit width, empirical residual quantiles, or calibration-set maximum. Calibration can choose a format; it cannot justify discarding an unbounded error tail.

Build `rho` with exact dyadic arithmetic or MPFR intervals. A convenient exact test for a candidate stored bound `r>=0` is `r*r >= Σ_j residual_j^2`, evaluated exactly with integer ratios. Compute a high-precision approximate square root only to propose `r`; increase the representable bound until the exact inequality passes. This can be chunked and vectorized with interval screening, reserving big-integer checks for uncertain groups. Preserve all metadata hashes.

### 4.2 The supplied residual-race theorem and its limits

For known `0<l_i<=f_i<=u_i`, let

`L=Σ_i l_i`, `d_i=u_i-l_i`, `p_i=l_i/L`, `Q=Σ_i d_i/(L+d_i)`.

An expensive oracle query reveals one `f_i`; all possible targets in the independent Cartesian box are allowed. Draw independent `T~Exp(L)`, `J~Cat(l/L)`, and `E_i~Exp(1)`. Select `C={i:E_i<d_i*T}` and query those coordinates in one batch. Return the earliest event among `(T,J)` and `(E_i/(f_i-l_i),i)` for `i∈C`, with zero-rate times set to infinity. If `C` is empty or `C⊆{J}`, return `J` without a query.

The baseline clock contributes mass `l_i`; residual clocks contribute `f_i-l_i`. Excluded residual clocks cannot beat `T`. Consequently the result is `Cat(f/Σf)` and the unshortened expected number of distinct queries is exactly `Q`. Coupling the lower corner with a change in one coordinate gives

`Σ_i [d_i/(L+d_i)](1-p_i) <= inf_A sup_f E_f N_A <= Q`.

The gap between the displayed endpoints is `Σ_i p_i*d_i/(L+d_i)<1`. This is a distinct-coordinate oracle result, not an arithmetic, bit, memory, or latency optimum. It permits randomized adaptive competitors but does not prohibit a real model implementation from exploiting known correlations outside the independent-box abstraction. Source S1 provides the original statement used here.

For `|z_i-a_i|<=epsilon` and `τ=1`, `l_i=exp(a_i-epsilon)` and `u_i=exp(a_i+epsilon)` give `Q<=exp(2epsilon)-1`; temperature replaces `epsilon` by `epsilon/τ`. This upper bound can be very pessimistic with heterogeneous intervals and peaked probabilities. The source's six-bit mean residual bound is several logit units, so a very small observed query fraction must be confirmed on real distributions rather than inferred from a small-error assumption.

**Stable diagnostic computation:** retain log bounds. Compute `log d_i` with validated log-difference arithmetic; compute `log L` by log-sum-exp. Evaluate ratios from log quantities. Treat zero widths exactly. Avoid `U-d_i` subtraction: compute each exclusion sum by prefix/suffix log-sum-exp or high precision. If a range cannot be represented faithfully, use MPFR, not clipping. These diagnostics are not permitted to leak fully evaluated logits into the deployed selection algorithm.

### 4.3 Gumbel screening

With exact independent standard Gumbels `G_i`,

`argmax_i (z_i + τ G_i) ~ P`.

Multiplying the usual score `z_i/τ+G_i` by positive `τ` avoids dividing logits by a tiny temperature. If `z_i∈[lo_i,hi_i]`, define

`M=max_j(lo_j+τG_j)`, `S={i:hi_i+τG_i>=M}`.

Every true maximizer lies in `S`. A singleton can be returned without reading its original weight row. Otherwise evaluate original rows for surviving unknown logits and select the true perturbed maximizer. Equality must retain a candidate.

In the ideal scalar-box model, with `l_i=exp(lo_i/τ)`, `u_i=exp(hi_i/τ)`, and `U=Σu_i`, the zero-query probability is

`alpha=Σ_i l_i/(l_i+Σ_{j≠i}u_j)`.

The expected queries, **summing only coordinates with `u_i>l_i`**, are

`Q_G=Σ_{i: u_i>l_i} [u_i/(L+u_i-l_i) - l_i/(U-u_i+l_i)]`.

A no-query algorithm cannot assign label `i` more probability than its minimum across the box; singleton Gumbel certification attains the sum of those minima. This optimizes zero-query probability, not necessarily expected queries. Source S1 supplies the formulas. These expressions are diagnostic ideals: random-number interval widths and finite arithmetic introduce extra contenders, so they do not equal the implemented shortlist cost automatically.

### 4.4 Numerical design: exact decisions without GPU logarithms

The strict implementation maintains intervals for `z_i` and `G_i`. It returns `k` only when

`lower_score[k] > max_{j≠k} upper_score[j]`.

It prunes `i` only when `upper_score[i] < max_j lower_score[j]`. Random-bit refinement, arithmetic refinement, row ordering, and dense fallback never replace the original random draw. All refinements intersect old and new valid intervals; an empty intersection is an implementation error.

Adaptive Gumbel interval decisions are prior art. The following lookup layout is an engineering choice to avoid relying on GPU log accuracy, not a novelty claim. [R05], [R12]

#### A. Reproducible lazy random tapes

Use a checked counter-based generator, such as the Philox4x32-10 reference from Random123, with its license retained. Port only the required integer operations and compare all CPU/Metal words to the upstream known-answer tests. [R41]

Assign every sampled state an explicit unique 64-bit `sample_id` in the manifest. For each vocabulary row, use a stream keyed by `(run_seed, method-independent purpose="gumbel")`, with counter words `(sample_id_low32, sample_id_high32, row_id, word_block_index)`. Use a recorded key derivation for other purposes; do not reuse a Gumbel stream for a baseline clock or label draw. Define word order and consume bits most-significant-first. Detect counter exhaustion; do not wrap. The selective and matched dense methods must use identical Gumbel streams regardless of tiling or candidate order.

#### B. Primary dyadic lookup, avoiding a frequent infinite tail bin

A uniform 16-bit prefix table is a correct diagnostic but a poor assumption for a rare slow path. At `V=151936`, its top bin contains about 2.318 rows on average, and the probability of at least two is about 67.3%; their infinite upper bounds require refinement. Do not clamp the bounds.

Instead construct the initial uniform variate as follows, using disjoint consecutive bits of the row's tape:

1. Read one fair side bit `S`.
2. Let `e` be the number of zero bits before the next one; `P(e=k)=2^(-k-1)`.
3. Read `m=12` mantissa bits as integer `r∈{0,...,4095}`. Let the unused continuation bits represent `R∈[0,1)`.
4. Set `v=(1+(r+R)/2^m)*2^(-e-2)` and `U=v` if `S=0`, otherwise `U=1-v`.

Each `(S,e,r)` bin has width and probability exactly `2^(-e-m-2)`; the bins partition `(0,1)` up to measure-zero endpoints. Thus `U` is uniform. Its interval is known from the prefix, and `G(U)=-log(-log U)` is monotone. For `S=1`, reverse the endpoints when converting the `v` interval to a `U` interval.

Precompute directed FP32 lower/upper Gumbel bounds for `S∈{0,1}`, `e=0,...,63`, and every `r`, using MPFR and exact dyadic endpoints. This is `2*64*4096*2*4 = 4,194,304` bytes. Never form `1-v` in ordinary double for deep-tail bins: those endpoints may require 77 bits. Use interval composition with directed rounding and validate every saved endpoint at higher precision. Increase MPFR precision until its enclosure can be safely rounded outward; 128 bits is a starting precision, not an unconditional accuracy guarantee.

If `e>=64`, use the same tape in the CPU high-precision path. The ideal probability per row is `2^-64`, so a union bound for Qwen is about `8.24e-15` per sample; this says nothing about ordinary near-score ties, whose frequency must be measured. There is no cap on `e` or on refinement that returns an uncertified token. The geometric loop terminates almost surely.

#### C. Refinement and slow path

After coarse screening, compute tight intervals from original rows for surviving candidates. If a strict winner exists, return it. Otherwise refine the **same** candidates' unused random bits and score intervals, beginning with MPFR precision 128 and doubling as needed. Use exact dyadic dot products where the remaining dot uncertainty matters. Preserve already proved exclusions; any discarded upper bound was below a valid surviving lower bound.

A simple exact dot oracle is `sum(Fraction.from_float(float(w))*Fraction.from_float(float(h)))`, with BF16 values first converted losslessly to FP32. It is intentionally slow and suitable for adversarial tests and rare unresolved rows. A C++/GMP integer accumulator may replace it after matching the reference. MPFR-directed conversion of that exact rational and interval logarithms resolve comparisons. The entire vocabulary need not be recomputed at high precision when prior exclusions remain certified.

For diagnostic finite-reference mode only, declare its finite Gumbel map, FP32 score arithmetic, and lowest-token-ID tie rule. Strict ideal-real mode continues refinement on overlapping intervals; it does not apply a finite tie rule to an unresolved real comparison. Independent continuous perturbations tie with probability zero. An interruption produces an unfinished sample checkpoint, not an arbitrary answer.

### 4.5 Auditable packed arithmetic

Current MLX affine quantization supports six bits and stores a scale and bias per group. Its actual quantizer includes edge-dependent adjustments; reuse its emitted bytes instead of reimplementing the simplified formula from documentation. [R17], [R19]

For `mx.quantize(W.astype(mx.float32), group_size=64, bits=6)`, freeze the returned packed array and FP32 metadata. In the inspected pinned implementation, four consecutive six-bit codes occupy three bytes `a,b,c`:

```text
q0 = a & 63
q1 = (a >> 6) | ((b & 15) << 2)
q2 = (b >> 4) | ((c & 3) << 4)
q3 = c >> 2
```

For `j=4t+r`, the byte offset is `row*(3*d_padded/4)+3*t`, and metadata group is `floor(j/g)`. Scales may be negative. Validate integer codes separately from floating dequantization. All values `0...63`, byte/uint32/group/row boundaries, constant groups, and zero-padded tails must be covered. Other bit widths require their own verified decoder; do not infer layout from the six-bit case.

The strict custom kernel uses explicit FP32 `fma(s,float(q),b)` and a recorded fixed FMA/reduction tree. For the inspected `mx.fast.metal_kernel` interface, set `compile_options={"math_mode": "safe"}` and use explicit `metal::precise::sqrt`; validate these options in the pinned installation. A standalone Metal build can use `-fno-fast-math` for the safe/precise contract. Preserve the compiler options and generated source. Native MLX `quantized_matmul` is a valuable performance baseline, but nominal FP32 dtype does not establish the necessary error bound. Its dispatch must be audited before it replaces the strict kernel. [R16], [R18], [R20], [R21]

The inspected Metal specification permits toward-zero rounding and subnormal flushing. Therefore use a conservative basic-operation model with `u=2^-23` and `eta=2^-126`, not a silent round-to-nearest assumption. Reject NaN/Inf inputs. Route nonzero subnormal original inputs/metadata to a CPU path that preserves them. Canonicalize every center-kernel intermediate with magnitude below `eta` to zero, and check every decode, accumulator, and tree output for `abs_bits>=0x7f7fffff`; discard its finite-center certificate and fall back if flagged. This catches RTZ overflow saturation even if later cancellation would hide it. Endpoint arithmetic may widen to infinity instead. [R16]

The GPU path accepts only hidden values exactly representable as FP32; BF16 promotion is lossless. An activation with additional precision must use a preserving CPU path or a newly certified conversion contract. The original output bias follows the same finite-value/domain rule. This check prevents the certificate from silently proving a different rounded input problem.

The initial correctness implementation may use operation-wise intervals:

```text
enclose_basic_result(r):
    if r is NaN or overflowed: return unresolved
    if abs(r) < smallest_normal_f32:
        return [-smallest_normal_f32, +smallest_normal_f32]
    return [previous_f32(r), next_f32(r)]

interval_fma(x_interval, y_point, acc_interval):
    choose product extrema using sign(y_point)
    evaluate the two endpoint FMAs
    enclose each result outward; take lower/upper endpoints
```

Use integer bit manipulation for neighboring FP32 endpoints so a GPU flush does not erase the outward step. For known nonnegative expressions, clamp the lower endpoint to zero; preserve exact zero only when established algebraically. Keep interval endpoints normal-or-zero by widening outward. Operands that are themselves intervals require endpoint extrema. With correct basic operations, safe compilation, no overflow, and the explicit underflow widening, this is a concrete conservative path even if its performance is poor.

For throughput, replace per-weight intervals with a **proved scalar error radius tied to the actual instruction graph**, as detailed in the numerical appendix in Section 4.9. Do not replace interval arithmetic with a fitted epsilon. Tightening a bound is an optimization with a proof obligation and exact-oracle regression checks.

### 4.6 Runtime pipeline and fallback policy

The minimum complete GPU path is:

1. Compute activation block norm upper bounds and any global magnitude bound once per hidden state.
2. Scan the packed sketch; compute logit intervals and initial dyadic-table random intervals. Produce score bounds and block maxima of lower scores.
3. Reduce to the global lower threshold. Mark every row whose upper score is at least that threshold.
4. If a winner is already strictly certified, return it. Otherwise read original rows selected by the mark, compute tighter score intervals, and reduce again.
5. Refine unresolved comparisons using the same random tapes; invoke CPU high precision only where necessary. Count its synchronization and transfers in latency.

Begin with a fixed full-vocabulary launch in step 4 whose inactive rows return before any original-weight load. This avoids assuming dynamic compaction is free and avoids an `O(Kd)` gather tensor. It still performs `O(V)` scheduling/flag work, which must be measured. Add GPU prefix-scan compaction or block-coalesced reads only if profiling identifies that work as material. Sorting candidates is permitted because row identity determines randomness.

Choose a dense crossover threshold from **development measurements** of complete costs. If many rows survive, use the certified dense path with the **same** random tapes and existing intervals. A pre-screen dispatcher based only on immutable shape/temperature and pilot cost models can avoid an unprofitable screen. Report pure SEC and dispatched SEC separately. Neither fallback may reset the seed, discard tail candidates, change the policy, or omit its own cost.

### 4.7 Reference pseudocode

```python
# Proposed interfaces, not an existing package. All interval endpoints are outward.
def sample_sec(h, asset, tape, temperature):
    validate_finite_positive_temperature(temperature)
    if not gpu_numeric_domain_supported(h, asset, temperature):
        return sample_dense_certified_cpu(h, asset, tape, temperature)

    z_interval = certified_sketch_intervals(h, asset)
    g_interval, rng_state = dyadic_gumbel_intervals(tape, V=asset.V)
    score = add_intervals(z_interval, mul_positive(temperature, g_interval))
    alive = safe_contenders(score)  # upper >= max lower; retain equality
    token = strict_winner(score, alive)
    if token is not None:
        return certified_result(token, counters())

    if dense_crossover_predicted(alive, h.shape, frozen_dispatch_config):
        return sample_dense_certified(h, asset, tape, temperature,
                                      prior_intervals=score)

    unknown = rows_needing_original_read(alive, z_interval, asset)
    z_interval[unknown] = intersect(z_interval[unknown],
                                   original_row_dot_intervals(h, asset, unknown))
    while True:
        score[alive] = add_intervals(z_interval[alive],
                                    mul_positive(temperature, g_interval[alive]))
        alive = retain_only_certified_contenders(score, alive)
        token = strict_winner(score, alive)
        if token is not None:
            return certified_result(token, counters())
        # Read continuation bits, never fresh replacement uniforms.
        # Refine both sources of uncertainty fairly; ensure each unresolved
        # competitor receives unbounded precision if ambiguity persists.
        g_interval[alive], rng_state = refine_gumbels_same_tape(alive, rng_state)
        z_interval[alive] = refine_dyadic_dots_if_needed(h, asset, alive,
                                                       z_interval[alive])
```

`safe_contenders` and refinement retain the previous certified global exclusions. When refining only a subset, retain the maximum lower bound and the largest upper bound among all still-live competitors; never forget an unevaluated competitor. Intersecting disjoint intervals, an empty candidate set, an out-of-range row, or counter reuse is a fatal correctness error with a reproducible failure bundle.

`rows_needing_original_read` excludes already known/cached logits and rows whose exact values can be recovered from a proved zero residual without accessing original weights. Noise-only ambiguity triggers noise refinement, not a redundant original-row read. A zero residual does not automatically make a rounded dot exact: refine its arithmetic from the known packed coefficients if necessary. Count those operations and packed rereads separately. The temperature domain check verifies its canonical bits and routes positive subnormals to CPU.

### 4.8 Essential limiting cases and harness tests

Test `V=1`; identical rows; all-zero hidden vector; zero residual; one uncertain coordinate; heterogeneous widths; all lower mass underflows in ordinary floating arithmetic; very large common logit shifts; positive/negative scales; padded dimensions; non-contiguous inputs; exact finite-score ties; near real-score ties; `τ→0+` and very large finite `τ`; signed zero; subnormal inputs; accumulator cancellation; overflow attempts; invalid temperatures; candidate sets of sizes zero, one, and `V`; forced dense fallback; CPU fallback; and interrupted/resumed random tapes.

For real-score intervals, provide exact dyadic small matrices and enumerate all interval corners when feasible. For the scalar samplers, compare pruned and full constructions with identical random variables. Force a known-wrong bound and a deliberately dropped candidate to verify that the test harness can detect failure. Benchmark two aliases of the same method to check that the timing analysis does not manufacture an advantage. These tests are necessary for the scientific mechanism, not a substitute for proof.

### 4.9 Numerical error-radius implementation contract

This subsection supplies a conservative implementable route; it is a deduction to be checked against generated code before `certified_real_head` is enabled. A source or compiler change invalidates the associated certificate until re-audited.

The strict FMA implementation has one rounded dequantization per term and a finite dot/reduction graph. The audited construction must be recorded in `numerics_contract.md`. Optional stored magnitude envelopes are counted as metadata; the primary scalar construction below computes a group magnitude bound from existing scales and biases instead.

The universally available initial path is the operation-wise interval code in Section 4.5. The optimized path may use the explicit conservative recurrence below, evaluated with upward arithmetic, so its safety does not rest on an unexplained gamma constant:

```text
For each term j:
    wj = fma(sg, qj, bg), with explicit intermediate FTZ convention
    Dj = up(u * abs_exact_coefficient_bound_j + eta)
    # |wj - exact(sg*qj+bg)| <= Dj

For each per-thread accumulator, start center a=0 and error E=0:
    r = fma(wj, hj, a)
    M = up(abs(wj)*abs(hj) + abs(a))
    E = up(E + Dj*abs(hj) + u*M + eta)
    a = r

For every fixed-tree addition of partials (a,Ea) and (b,Eb):
    r = a+b
    E = up(Ea + Eb + u*(abs(a)+abs(b)) + eta)
    a = r
```

Here an exact-coefficient magnitude bound can be `up(|s_g|*63+|b_g|)` for Q6, or a tighter stored value. The recurrence bounds dequantization error, accumulation roundoff, and FTZ absolutely. It is valid for cancellation and toward-zero rounding. Error-update arithmetic itself uses the nonnegative upward primitives. Its extra per-term operations may be costly; that cost is real. For original-row dots, `Dj=0` after lossless BF16 conversion, and the same recurrence applies.

A faster explicit scalar radius follows. Let `T` be a power-of-two number of nonempty lanes. Each lane executes at most `ceil(d_p/T)` dot FMAs, followed by a fixed `log2(T)` addition tree; masked padding must not add uncounted operations. Define

```text
Kdot = ceil(d_p/T) + log2(T)
K = Kdot + 1                         # includes one affine-decode rounding
Ndot = d_p + (T-1)                   # all dot FMAs and reduction additions
H1_g >= sum_{j in g} abs(h_j)
H1 >= sum_g H1_g
M_ig >= max_{j in g} abs(s_ig*q_ij+b_ig)
A_i = upward_sum_g(M_ig*H1_g)
delta_i = [K*u*A_i + eta*(H1+Ndot)] / (1-K*u)
```

Require `K*u<1`. A no-extra-metadata `M` is the upward bound on `max(|b_g|,|63*s_g+b_g|)`: absolute value of an affine expression is maximized at an endpoint over `q∈[0,63]`. Obtain the second endpoint by outward FMA, accounting for either scale sign. Compute `H1` and every positive operation upward. Precompute upward constants `alpha>=K*u/(1-K*u)` and `beta>=1/(1-K*u)` with exact rational/MPFR arithmetic, then evaluate `delta=up(alpha*A_i + beta*eta*(H1+Ndot))`. This avoids an unqualified GPU division.

**Why the bound holds:** expand the fixed graph. Each exact contribution `(s*q+b)*h` experiences at most `K` relative errors with magnitude at most `u`, giving aggregate deviation at most `K*u/(1-K*u)` times the sum of absolute contributions. Additive decode FTZ errors contribute at most `eta*H1`; dot/reduction FTZ errors contribute at most `eta*Ndot`. Their later amplification is at most `1/(1-K*u)`. This proves the displayed bound despite cancellation, conditional on the declared primitives and graph. Overflow paths are excluded by the per-operation guard. For `d=576,T=32`, `K=24` and `Ndot=607`; compute constants from these integers rather than copying a rounded decimal.

Audit actual source counts after any loop, lane, padding, compiler, or reduction change. This scalar bound and the per-term recurrence need not dominate one another numerically; each must independently enclose the exact error. The independent arithmetic-model check in Section 2 exercised `(d,T)=(1,1),(8,2),(17,4),(32,8)`, both rounding modes, and exponents extending to the normal/subnormal boundary. Finite success supplements the graph argument. If a faster path cannot be justified, retain the operation-wise path, report its cost, and narrow the performance conclusion.

For block norms, accumulate `h_j*h_j` with upward nonnegative FMA, then use precise square root and an outward upper step. The result upper-bounds `||h_g||_2`; this costs `O(d)` per state. Accumulate `Σ rho_ig H_g` upward. Score addition and multiplication by `τ` also use outward primitives. Offline table bounds, residual bounds, arithmetic margins, and online comparison intervals all belong to one certificate chain; none may be omitted.

## 5. Fair evaluation and protected confirmation

### 5.1 Claims and comparison hierarchy

**Primary systems claim:** on the preregistered Qwen3-0.6B-Base, batch-one, temperature-one, held-out natural-text state population, under **varied-state replay**, the complete certified selective head has lower mean wall-clock microseconds per sampled token than the strongest qualified local certified dense alternative, without a correctness failure. The complete boundary ends when a scalar token and status are available to the host consumer and all used streams have completed. Confirm practical competitiveness separately against the fastest conventional native dense path. Do not collapse these comparators: the latter may implement a slightly different finite-arithmetic distribution. Hot repeated-state and body-interleaved measurements are separate secondary regimes, never alternatives from which to choose a favorable primary after inspection.

**Primary mechanism claim:** at fixed head values, hidden states, random tapes, numerical guarantees, and generic optimizations, the sketch reduces original-row work enough to reduce the complete head time. Savings in logical reads alone support a narrower result.

**Secondary system claim:** the same improvement reduces full serial decode latency with the same body and KV computation. Establish it only through integrated measurements. No perplexity improvement is expected from a correctly exact head.

The Qwen workload is chosen because of its verified large head and small complete model, not after selecting a favorable test result. SmolLM2 is a preregistered replication/scale condition. If Qwen assets or hardware prerequisites fail, a Smol-only study is a scope amendment before protected outcomes are accessed, not a silent replacement of the primary population.

### 5.2 Baselines and their qualification

| ID | Method | Role and exact comparison requirements |
|---|---|---|
| D0 | Stock MLX-LM body, dense tied head, compiled native categorical | Practical local baseline. Disable all generation-policy extras, include sampling, evaluate the returned token. Preserve its finite arithmetic and document difference from the strict target. |
| D1 | Dense original-head projection with the same interval/random-tape machinery as SEC | Required same-target baseline. Read all rows; use efficient bounded FP32 dot products and identical rare high-precision refinement. Do not evaluate every row in MPFR. |
| D2 | Fused tiled full-row Gumbel interval sampler | Strong dense alternative inspired by FlashSampling; keeps local score contenders rather than materializing full logits. Same certificates, lookup, refinement opportunities, and tuning effort as SEC. |
| S0 | Classical sequential squeeze rejection with per-sample row cache | Closest simple exact algorithmic baseline using the same positive mass intervals. Count proposal setup, repeated proposals, distinct row reads, and CPU/GPU synchronization. |
| S1 | Supplied single-batch residual race | Mandatory oracle/mechanism baseline. Same bounds, row evaluator, original-row cache, and numerical contract. CPU interval reference always required; optimized GPU race only if measured cost can plausibly compete. |
| S2 | One-batch Gumbel shortlist | Direct predecessor and starting implementation. Distinguish ideal exact-Gumbel diagnostic from the implemented certified noise intervals. |
| S3 | Upper-score-first adaptive Gumbel refinement | Control for batching versus fewer queries. Always refine the largest remaining upper-score contender; stop only on a strict certificate. Count synchronization. |
| A4/A6/A8 | Same-body native MLX four/six/eight-bit heads without correction | Approximate latency/quality/memory frontier. Same codes and metadata where formats match. Never label their conditional law unchanged. |
| Qfull | A strong complete quantized local model, optional | Needed only for a broad claim about superiority to whole-model quantization. Use a calibrated supported method and/or a faithfully converted llama.cpp Q4_K_M reference, with matched data and quality analysis. Different body states prevent an exact head comparison. |

For S0, draw `I~Cat(u/U)` and a height `Y~Uniform(0,u_I)`. If `Y<=l_I`, accept without querying; otherwise fetch/cache `f_I` and accept if `Y<=f_I`, else repeat. The proposal and decisions must use certified arithmetic in strict mode. Proposal-table construction is paid once **per new hidden state**, not amortized over unrelated changing logits. Ordinary squeeze is not required to adopt the report's log-concavity assumptions because valid envelopes are supplied externally.

S1/S0 with transcendental interval operations can be first implemented on CPU for correctness and query-count comparison. A slow interpreted reference is not a credible systems baseline. Before declaring a latency advantage over these algorithms, either implement their fastest plausible local path using the shared backend or explicitly restrict the measured superiority claim to the qualified dense methods. Use measured setup/dispatch costs to document why an optimized serial alternative could or could not plausibly win; do not substitute an unmeasured asymptotic argument.

D1 and D2 each receive packed-independent generic optimizations, efficient reductions, the dyadic random table, and high-precision work only for unresolved contenders. D2 need not reproduce CUDA instructions or published GPU numbers. If a local fused port remains slower than D0/D1, the faster implementation still sets competitiveness. Record baseline source, dtypes, layouts, compile flags, candidate kernels, tuning data, and the selected configuration before confirmation. A method with invalid numerical bounds cannot enter the strict baseline pool.

D0's practical configuration must remain visible and independently tuned. Measure supported native precision settings on development data; a certification-process setting such as `MLX_ENABLE_TF32=0` must not silently slow or change D0. Use separate processes/configurations when needed, preserving paired input states and explicit timing boundaries. Record whether a setting actually changes dispatch or speed on this device; do not presume it does.

For D2, tile `t` computes `L_t=max_{i in t} lower_score_i` and retains **every** row with `upper_score_i>=L_t`, including equality. A center-score argmax or one arbitrarily chosen row per tile is insufficient. After global `L=max_t L_t`, discard only retained rows with upper score strictly below `L`, then refine all remaining comparisons as required. Either allocate a counted `O(V)` worst-case contender buffer or provide a deterministic replay/overflow route that recovers all contenders with the same tapes. Never truncate a tile to a fixed candidate count. Fusion may save ordinary logits materialization even though worst-case certificate state is still `O(V)`.

The primary same-target comparator is the baseline chosen as fastest on development data from D1/D2 and any other fully qualified strict implementation. For the stronger wording “beats both dense alternatives,” require simultaneous favorable comparisons to D1 and D2 on confirmation, with the multiple-comparison rule below. Report D0 regardless of outcome. Choosing the fastest confirmatory baseline after looking at test times is descriptive only, unless the family-wise analysis already accounts for it.

Mussmann's exact lazy-tail algorithm is a reference when a true top-k/tail certificate is available cheaply. Obtaining top-k through an already completed expensive dense projection is only a diagnostic of random-number generation, not an acceleration. Approximate MIPS, AdaptiveSoftmax, and nonlocal published FlashSampling numbers are contextual. Do not claim a controlled head-to-head win against them. [R03], [R07], [R08]

### 5.3 Assets, provenance, and state populations

**Models.** Use the pinned originals in Section 6. Download permitted public safetensors and tokenizer/config files. Do not require `trust_remote_code=True`; these architectures are supported locally. Hash every file, keep licenses/notices, and verify shape, source dtype, tied embedding identity, vocabulary mapping, and output bias. Quantization applies only to the auxiliary head sketch for strict comparisons.

**Natural text, primary:** Salesforce `wikitext`, configuration `wikitext-2-raw-v1`. Its card identifies CC-BY-SA-3.0/GFDL and downloadable data of about 7.75 MB. Resolve and record the dataset repository commit before download; keep raw file hashes. Use original train articles for format/calibration exploration, validation articles for tuning and pilot noise estimation, and test articles as protected confirmation. The published rows are text lines, **not independent documents**. Reconstruct articles using top-level headings: a stripped line matching exactly one opening `= ` and one closing ` =`, excluding nested `==` headings, begins a new article. Preserve line order and blank lines; inspect ambiguous parses on training data. Record resulting counts rather than assuming a particular number. [R34]

**Independent domain checks:** HumanEval prompt text for code, and GSM8K question text for mathematical prose. These repositories are public with MIT licenses; pin a commit and hash data/license files at acquisition. Use only the text fields needed here, with no evaluation of generated code and no reference solutions as prompts. HumanEval has no train split; reserve task IDs deterministically by `SHA256("sec-v1|"+task_id) mod 5`: bucket 0 development, remaining buckets protected. Use GSM8K train for development and its original test set for protected shift evaluation. These are workload probes, not new claims about solving code/math. Report their narrower provenance and possible training contamination. [R35], [R36]

If a dataset becomes unavailable or licensing cannot be verified, use another openly licensed real text collection only through a recorded amendment made before inspecting its performance outcomes. Synthetic vectors are not a replacement for a real-workload claim. The complete small-matrix and numerical study remains possible without any text dataset.

**State construction.** Tokenize plain text with each model's exact pinned tokenizer, without a chat template or an implicit special-token policy. Record `add_special_tokens` and any BOS insertion explicitly; use the same recorded token IDs for every method. Keep per-document context separate. For long documents, use contiguous chunks of at most 2,048 tokens initially; reset KV at chunk boundaries, record chunk offsets, and treat all chunks as one document for uncertainty. Later context conditions may increase to the model/hardware limits without mixing contexts across documents.

Generate candidate hidden states by running the frozen model body in evaluation mode. Select positions by a deterministic seeded uniform reservoir within each document, independent of logits, margins, entropy, or candidate count. Start the pilot with at least one state per document; choose the final number of positions and random-tape replicates from variance decomposition. A good allocation spends more effort on new documents while between-document variation dominates, and on additional states/tapes only while within-document Monte Carlo noise materially contributes. Freeze the selection code, counts, and seed before reading protected head outcomes. All sampled positions, including unhelpful ones, remain in the result table.

Head-only replay may use cached states. For end-to-end checks, run the body immediately before the head, and separately generate autoregressive trajectories. Never assume cached-state timings reproduce normal cache behavior. A small source-equivalence check must show that the adapter's extracted normalized hidden states followed by the stock tied projection reproduces the stock MLX model call under the same kernel/dtype; this tests the extraction, not equality to the strict-real dot.

**Contamination and leakage.** Model pretraining may include these public texts. That limits capability/generalization claims but does not invalidate paired computational comparisons on fixed states. Do not claim unseen-language modeling ability. Calibration and confirmation documents must be disjoint by normalized-content hash, and all positions from one document stay in one partition. Hidden-state caches include model, tokenizer, body, precision, position, and source hashes. Full test logits may be computed only by the evaluation harness after the protocol is locked; they are inaccessible to method selection and dispatch training.

### 5.4 Metrics and units

For each method, state, and random tape, report:

- Complete head elapsed time in microseconds, including sketch, bounds, RNG, selection, full-row reads, fallback, all required synchronization, and returning the token to the agreed consumer boundary.
- Original rows read, distinct rows read, physical/coalesced blocks touched if instrumentable, packed bytes, metadata bytes, and **modeled** bytes including copies. Label modeled traffic separately from hardware counters.
- Initial contenders; additional contenders attributable to arithmetic margins and random intervals; full-dot refinements; CPU refinement incidence and time; dense fallback incidence and time; precision/bits consumed. Separate CPU refinement of a few survivors from a full-vocabulary CPU fallback, recording reason, rows, precision growth, transfers and unfinished status.
- Peak active and cached device bytes, process RSS, source/derived payload bytes, preprocessing seconds, compile seconds, and amortization break-even number of tokens.
- Mean, median, p95, and maximum observed latency. The mean is primary because rare correct fallbacks are part of expected work; median-only reporting can conceal that cost.

For approximate controls, compute head-level forward KL and total variation against high-precision dense softmax on evaluation states, plus held-out token NLL/perplexity if a model-quality tradeoff is reported. Use stable log-normalization and enough precision to distinguish error from reference numerical uncertainty. State the probability mass or numerical error bounds used; do not floor probabilities silently. Correct strict methods target zero distribution change by construction; their quality agreement tests detect bugs but do not prove exactness.

For integrated inference, report time to first token separately from decode time per output token, tokens/second, total request time, context/generated lengths, EOS behavior, peak memory, and any fallback tails. Use natural-EOS runs for workload realism and a separate fixed-work replay for paired timing; do not silently force generation beyond EOS and call it ordinary application latency. Throughput and serial latency are separate endpoints. Measure energy only if a consistent local counter is available without altering the workload; it is optional, and proxy estimates must not be described as measured power savings.

### 5.5 Aggregation, uncertainty, and fixed confirmation

The independent workload unit is a **document** (or a HumanEval/GSM8K task), not a token, seed, or timing repeat. Define for each document the mean head time across its preregistered positions and random tapes. The primary effect is the unweighted mean across documents of paired differences

`Delta_j = mean_tapes,positions(time_baseline - time_SEC)` in microseconds/token.

Positive values favor SEC. Also report the ratio of aggregate means and percent change, with paired confidence intervals. Token-weighted results are useful secondary deployment summaries but answer a different population question. Report each corpus/model independently; any combined result needs fixed weights specified before testing.

Run paired randomized method orders in measurement blocks, balanced across thermal sessions. Repeated measurements of a document estimate timing noise; they do not create more independent documents. The primary inference is **conditional on the fixed measured sessions**: resample whole document vectors jointly, preserving all methods, positions, random tapes, blocks and session observations for a selected document. Average sessions with frozen equal weights and report session-specific effects/contrasts. Documents and sessions are crossed; do not treat one as nested in the other or bootstrap only two sessions as a reliable estimate of arbitrary future-session variability. If inference over sessions is essential, preregister a crossed-cluster/random-effects analysis and collect enough independently initialized sessions to estimate that variation. If the corpus is treated as a complete finite target population, distinguish measurement/randomness uncertainty from extrapolation to other documents.

All sessions contributing to that primary estimate are preregistered and completed within WP08 before primary unblinding. WP09's fresh-day session is a separate secondary replication; its observations are not retrospectively added to the frozen primary estimate or CI.

**Pilot procedure:** use only development documents to estimate (a) between-document variance, (b) within-document state/randomness variance, and (c) within-block timing variance. Choose enough repetitions that timing noise no longer materially drives the CI; then spend additional budget on independent documents. Pilot warm-up continues until successive windows no longer show a trend large enough to affect the planned precision; freeze the resulting warm-up rule and record its cost separately.

**Rare-fallback calibration:** deliberately force every CPU/dense/refinement path and measure its cost over difficult finite inputs, separately from natural-incidence estimation. Count natural events across fresh independent random tapes conditional on each frozen state; do not count repeated timings of the same tape as new incidence trials. With zero events in `n` independent tape trials, the exact one-sided 95% binomial upper bound is `1-0.05^(1/n)` for that fixed-state event probability. Pool states only under a declared mixture/independence model; otherwise use stratified bounds and report workload coverage. A stress-case observed cost is not a proved latency maximum. Use a stated range of plausible tail costs to test whether unseen fallback could change the planned mean-effect resolution; if it could, increase pilot incidence trials and the fixed confirmation schedule or declare the expected-latency conclusion insufficiently resolved. Section 10's ideal gap bound can supplement this analysis only under its uniform-width assumptions. No pilot with zero rare events may justify assuming zero tail variance, and no hard runtime cap may turn an unresolved exact sample into an approximate answer.

Before confirmation, record a smallest useful absolute effect `delta_use` in microseconds/token, based on the measured baseline head time, body fraction, intended request length, extra memory/preprocessing, and a stated local-use scenario. This is a **decision threshold**, not a fabricated required percentage win. Also report detection below it if feasible. Use the pilot to select a target CI half-width `h` and document count; a planning approximation is `n ≈ (1.96*s_D/h)^2`, with simulation/bootstrap calibration for clustering and heavy tails. Set `h` small enough to distinguish zero from effects near `delta_use`, explaining the choice. If no application-specific useful threshold can be justified, publish several absolute-effect resolutions and classify practical value as unresolved.

Use one fixed confirmatory sample and a fixed repetition schedule chosen from that pilot. Do not keep running until significance. If the required independent sample exceeds the available protected population or feasible resource budget, run the declared feasible census/sample and report the attained CI and detectable effect; classify insufficient precision as inconclusive. A later extension requires a new preregistration and fresh documents or genuinely independent confirmation, not reanalysis until a favorable p-value appears.

Use **primary-first gatekeeping**. The single frozen primary model/policy/competitor contrast passes only if its two-sided 95% CI lies wholly above zero. Formal secondary superiority claims are enabled only after that primary passes, and use Holm testing at family level 0.05 across all preregistered secondary baseline/domain/temperature contrasts; otherwise secondary intervals remain descriptive/exploratory. Claiming superiority to both D1 and D2 requires both relevant contrasts to pass their assigned primary/secondary gate. A practical superiority claim against D0 belongs in the secondary family. Freeze the hypotheses, sidedness and method for obtaining bootstrap tests/compatible intervals in `analysis.md`, and verify them on synthetic null/injected-effect data. This is one sequential gatekeeping design, not separate uncorrected primary and secondary families. Report all tested configurations, and never elevate the best test-set configuration to primary after inspection. Bootstrap resample counts are increased until Monte Carlo error in interval endpoints is negligible relative to the chosen reporting precision, then fixed; resamples are not new data.

Unfinished exact samples are not ordinary exclusions. Record elapsed work and unresolved status; finish them from their original tapes when possible. If any remain unresolved, report censoring and the limitation on the expected-latency claim, rather than reporting a mean conditioned silently on easy completed cases. A lower bound from elapsed time or explicit resource failure is a different outcome from a completed mean.

### 5.6 Timing boundaries and environmental controls

Use native ARM64 processes. Record power source, macOS power mode, display/external-device load, ambient conditions if available, and concurrent heavy processes. Use the same power configuration throughout. Compile, materialize assets, and warm all selected shapes before the timed phase; report cold-start and preprocessing separately.

MLX is lazy. Evaluate inputs before starting, launch the complete method, force evaluation of the returned token and status, and synchronize **every used stream** before stopping the monotonic high-resolution clock. Synchronizing only the default stream is insufficient if work uses another stream. A device-only interval can be reported alongside the complete serial boundary, but cannot replace it. [R30]

Interleave methods in randomized balanced blocks. Log every timing, including slow correct fallbacks. Exclude a run only under a preregistered external-failure rule, such as sleep, a system update, or a detected unrelated process event; retain the excluded run and reason. Thermal throttling caused by the method itself is an outcome, not grounds to delete the sample. If memory pressure/swap invalidates an in-memory comparison, rerun a complete balanced block after reducing an agreed workload, recording the scope change.

Measure three cache regimes separately: repeated-state hot microbenchmark, **primary varied-state replay**, and normal body-interleaved decode. Freeze replay state order, balanced block structure, each method's residency/warm-up policy and host consumer boundary before confirmation. Artificial cache flushing is a labeled sensitivity test; it must not be applied only to the baseline. Sequential repeated head calls can exaggerate or reverse the benefit because the sketch and original head occupy different cache capacities.

### 5.7 Confirmation lock

Before protected state/head evaluation, write and hash `protocol/confirmatory.yaml` and `protocol/analysis.md`, commit source and dependency lockfiles, and record:

1. Primary claim, reference law, dataset revisions/IDs, model, canonical temperature bits, context rule, state positions, random tapes, cache regime, replay order and consumer boundary.
2. Qualified baseline pool, development selection, SEC parameters, fallback thresholds, kernel source hashes, and numerical contract.
3. Metrics, aggregation, uncertainty procedure, fixed session weights, sample/repeat sizes, fallback-incidence/tail sensitivity design, multiple-comparison family, exclusion policy, and practical-decision rationale.
4. Resource forecast, stopping/failure rules, and the exact commands for evaluation and reporting.

Keep protected reports inaccessible to tuning code by directory/API separation and a runner that checks the protocol hash. The coding agent will technically control the files, so this is a procedural guard, not a security claim. A bug discovered during confirmation invalidates affected results; record it, repair using development fixtures, and decide whether fresh confirmation is required. Never quietly replace a protected run.

## 6. M3 Max feasibility and adaptive configuration

### 6.1 Stack and compatibility

Use native ARM64 Python 3.12 as the initial environment, MLX/MLX-LM for the unchanged body and native baselines, NumPy for CPU diagnostics, safetensors/Hugging Face tooling for assets, and MPFR through a verified gmpy2 binding or a small C++ helper for directed arithmetic. No training, remote accelerator, paid model API, or extra machine is required. Prefer binary wheels; lock exact resolved packages after the smoke tests.

Inspected release candidates are MLX `v0.32.4`, commit `36969c094354086e15d2a327a9c50feb20b9cf11`, and MLX-LM `v0.32.2`, commit `93e6d4a6447b3080861985947f0820e76fcb957d`. Their coexistence on the user's laptop is **unverified**. MLX's inspected installation documentation requires Apple-silicon Python >=3.10 and macOS >=14.0; custom/source builds have additional Xcode/C++20/CMake requirements. Do not automatically upgrade the user's OS. [R22], [R23], [R24]

MLX and MLX-LM are MIT-licensed. MPFR has LGPL obligations if redistributed; gmpy2 and any bundled dependency retain their own licenses. Retain upstream notices, avoid relabeling supplied unlicensed code, and keep model/data licenses in the asset manifest. The supplied ZIP has no explicit code license in its file listing: use its mathematical descriptions and user-authorized local study, but do not publish its source under an invented license.

Existing verified APIs include `mx.quantize`, `mx.quantized_matmul`, `mx.fast.metal_kernel`, `mx.eval`, `mx.synchronize`, and the memory counters cited below. `gather_qmm` gathers batch matrices, not arbitrary vocabulary rows of a two-dimensional head; do not assume it provides the desired sparse GEMV. Apple Metal does not provide native FP64 arithmetic for this route, so the high-precision fallback is CPU-side. Set `MLX_ENABLE_TF32=0` in certification processes and audit actual dispatch; it is not itself a proof of FP32 semantics. [R18], [R20], [R21]

The model adapter should use the pinned model's inner body and tied `embed_tokens.as_linear` boundary where applicable. Do not invent an `output_hidden_states=True` MLX-LM API. Inspect the pinned class and implement a tested adapter. Native MLX-LM already compiles categorical sampling, so an unfused CPU softmax is not an acceptable practical baseline. [R28], [R29]

### 6.2 Hardware discovery: existing commands

The following are existing macOS commands, not proposed project modules:

```bash
uname -m
sw_vers
sysctl -n hw.memsize
sysctl -n machdep.cpu.brand_string
df -k .
vm_stat
sysctl vm.swapusage
xcrun --show-sdk-version
```

Capture MLX device information, Metal availability, actual package versions, Python architecture, compiler versions, recommended working-set information if exposed, and library configuration. Check `mx.metal.is_available()` and then query the documented `mx.device_info(mx.gpu)`; preserve the mapping instead of assuming fixed keys. Explicit device selection prevents accidentally reporting a changed default CPU device. If collecting `system_profiler` output, redact serial numbers and hardware UUIDs from shared artifacts. [R31]

Determine available RAM from system conditions rather than total installed RAM alone. Establish a working budget after reserving observed OS/background consumption and an explicitly documented safety allowance. Calibrate real peak memory and paging before large work. `set_memory_limit` is not a hard process cap; monitor active memory, allocator cache, peak memory, process RSS, and swap. [R31]

### 6.3 Pinned models and local scale

| Property | SmolLM2-135M | Qwen3-0.6B-Base |
|---|---|---|
| Repository | `HuggingFaceTB/SmolLM2-135M` | `Qwen/Qwen3-0.6B-Base` |
| Revision | `93efa2f097d58c2a74874c7e644dbc9b0cee75a2` | `da87bfb608c14b7cf20ba1ce41287e8de496c0cd` |
| Vocabulary × hidden | `49152 × 576` | `151936 × 1024` |
| Layers; KV heads; head dimension | `30; 3; 64` | `28; 8; 128` |
| Source weights / tied embeddings | BF16 / yes | BF16 / yes |
| Context ceiling from config | 8,192 | 32,768 |
| License | Apache-2.0 | Apache-2.0; ungated at inspection |

The supplied Smol checkpoint size is 269,060,552 bytes, with SHA-256 `80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1`; verify on download. Qwen's inspected repository metadata reports 596,049,920 BF16 parameters, or 1,192,099,840 parameter bytes before file overhead. Resolve exact file size/hash during acquisition. In particular, Qwen specifies head dimension **128**; `hidden_size/num_attention_heads` would give the wrong KV-memory estimate. [R25], [R26], [R27]

The original tokenizer and generation-config files are retained for provenance, but the experiment explicitly overrides truncation and sampling defaults. Never merge an instruction model's recommended top-p policy with the selected full-softmax experiment.

### 6.4 Memory and storage model

Let `d_p` be padded hidden width, `g` group size, `q` sketch bits, `m_meta` metadata bytes per row/group, `n_state` cached states, and `K` surviving rows. The principal terms are:

```text
M_sketch = V*d_p*q/8 + V*(d_p/g)*m_meta
M_KV = 2*layers*B*context*n_kv_heads*head_dim*bytes_per_KV_element
M_states = n_state*d*bytes_per_hidden_element
M_peak = model_resident + M_sketch + M_KV + activations
         + sampler_vectors + RNG_table + CPU/GPU_transfer_buffers
         + preprocessing_chunk + allocator_cache + conversion_temporaries
```

Original tied head weights are already part of `model_resident`; count them once. Sketches add storage. Initial Q6 affine metadata uses FP32 scale, bias, and residual norm, so `m_meta=12`. Any extra magnitude envelopes, second norm, low-rank factors, duplicate host arrays, or alternative layouts must be added explicitly. Compute group coefficient maxima from existing scale/bias where possible to avoid an uncounted metadata array.

| Calculated quantity, MiB = 2^20 bytes | SmolLM2 | Qwen3 |
|---|---:|---:|
| Original BF16 head | 54.000 | 296.750 |
| FP32 full-head copy, if created | 108.000 | 593.500 |
| FP64 full-head copy, if created | 216.000 | 1,187.000 |
| Q6 codes only | 20.250 | 111.281 |
| Q6 plus FP32 scale/bias/residual bound, g=64 | 25.313 | 139.102 |
| BF16 KV per 1,024 tokens, B=1 | 22.500 | 112.000 |
| FP32 hidden cache for 8,192 states | 18.000 | 32.000 |
| Shared dyadic random-interval table | 4.000 | 4.000 |

These are arithmetic estimates, not measured peaks. The strict Q6 sketch scan alone is `0.46875` times dense BF16 head bytes before any full-row reads. If every selected BF16 row is read once without a copy, add `E[K]/V`; real cache-line overfetch, activation/norm reads, flags, table lookups, gathers, and fallbacks add more. The original report's `0.40625` base ratio applies to a different symmetric FP16-metadata format.

Do not construct full `W`, `W_hat`, residuals, and squared residuals simultaneously in FP64. A preprocessing tile of `r` rows needs approximately the sum of its live arrays, e.g. four FP64 matrices alone cost `32*r*d_p` bytes, plus original/codes/metadata. Choose `r` from the measured available budget and verify peak usage; 1,024 rows is a harmless initial calibration tile, not a mandated workload cap. CPU exact checks may be slower than array operations, so measure their per-group rate before processing a full head.

Save original head rows in a lossless row-addressable CPU format or map the checkpoint without materializing a full FP32 host duplicate. If extraction creates another BF16 file, count its disk bytes. Exact fallback reads only unresolved rows where possible. A worst-case candidate buffer sized for `V` costs `O(V)`, while a naive gathered `W[K,:]` costs `O(Kd)` and may become a complete duplicate; the initial masked-row kernel avoids that duplicate.

Disk planning includes original checkpoint/download temporary, optional conversion copy, one active sketch, cached states, raw results, compiler cache, and atomic-write temporary space. Qwen's weights plus one extracted head/sketch and modest state shards are a few GiB, not a measured guarantee. Check actual sizes before committing downloads. Generate alternative sketches sequentially and retain their immutable manifests/results; regenerate rejected variants if storage is tight. Do not delete the source checkpoint or unique raw evidence merely to continue a run.

### 6.5 Cost calibration and end-to-end headroom

Use the measured model

`t_SEC = t_scan_and_bounds + t_rng_and_selection + t_original_rows(K,layout) + t_refinement + t_dispatch`.

Measure joint pipeline times as well as isolated stages; overlapping costs cannot simply be summed twice. Fit row-read time over candidate sizes and spatial patterns drawn from development traces, including `K=0,1,V`. A simple fit with launch intercept, bytes, and active row blocks is sufficient if residual errors are reported. Do not extrapolate from a sparse kernel's throughput at large `K` to a one-row case.

If the baseline head/sampler occupies fraction `f` of measured full-decode time and the complete head speedup is `s`, ideal whole-decode speedup without new overhead is

`S_total <= 1 / ((1-f)+f/s)`.

For illustration only, `f=0.2` and `s=2` imply about `1.111×`; if `f=0.05`, the same head result implies about `1.026×`. Actual `f`, `s`, and added dispatch/CPU costs must be measured. Likewise, dividing head bytes by an assumed bandwidth gives a lower-bound illustration, not a laptop speed prediction. Use observed effective bandwidth and cache regime for forecasts.

Preprocessing amortization is `N_break_even = additional_preprocessing_time / positive_per_token_time_saving`, with consistent units. If the saving is nonpositive or too uncertain, report break-even as nonexistent or unresolved. Disclose reusable pretrained assets separately from project-specific packing, certificate generation, tuning, compilation, and evaluation.

### 6.6 Hardware-dependent rules and faithful fallbacks

- If Qwen fits the measured budget with safe workspace and no sustained paging, use it as planned. Load only one model at a time. Smaller available memory first reduces preprocessing tile/state-shard size; it does not justify silently changing reference weights.
- If long contexts fail but the batch-one core fits, reduce context under a recorded scope amendment and retain the component claim. The primary mechanism does not need maximum context.
- If MLX or custom Metal cannot be qualified on the installed OS, follow the explicit CPU/native-only branches below. Mark GPU performance blocked. Do not substitute remote hardware or pretend that CPU diagnostics complete the GPU claim.
- If native six-bit dispatch fails or cannot be enclosed, use the audited custom decoder/kernel. An eight-bit fallback is a different tested configuration, not evidence that six-bit works.
- If a GPU primitive cannot support the declared arithmetic contract, use CPU intervals or a verified primitive implementation. Empirical padding is not a faithful fallback.
- If the sketch alone plus unavoidable selection exceeds an already qualified comparator with convincing measurement precision, stop the corresponding claim/optimization path before a broad kernel project. A pre-WP05 comparison to D0 rules out practical native competitiveness, not an unmeasured same-target D1/D2 gain. Investigate a specific cheaper certificate/fusion only if profiling identifies headroom; otherwise report the scoped local negative result.
- If the smallest model is launch-bound but measured cost models predict headroom at Qwen's larger head, proceeding to Qwen is justified. A failed tiny proxy alone does not reject the scale-dependent mechanism.

Longer runs and larger local models are admissible only when measured information gain, resource fit, and unresolved hypotheses justify them. There is no predetermined GPU-hour, model-size, or percentage-success quota.

**Executable fallback branches:** if Python/NumPy and MPFR work but MLX/native body support does not, run `WP00 → WP01 → WP11`: exact small-matrix samplers, numerical emulation and the failure report require no Metal kernels or model download. This is a completed prerequisite diagnosis, not real-model GPU evidence. If native MLX body/packing works but the strict custom kernel fails qualification, label the backend `native_body_cpu_certified_head`; run WP03 for development states/D0, WP04 using CPU exact certificate checks, and CPU portions of WP05 for oracle/query diagnostics, then WP11. Do not require D2 or WP06–WP09, and do not claim strict GPU speed. Re-enter WP02 only if a specific controllable Metal implementation is later justified. These paths preserve the target law in the computations they actually perform.

## 7. Ordered work packages and decision gates

The coding agent may make routine, reversible implementation decisions within this specification: file organization refinements, chunk sizes, compatible dependency pins, ordinary bug fixes, and kernel tuning on development inputs. Changes to the target law, dataset population, primary metric, guarantee, baseline fairness, or confirmation analysis require a dated methodological amendment and a decision about fresh confirmation. Progress must be stored in the repository, not only in chat.

The dependency path is:

`WP00 → WP01 → WP02 → WP03 → WP04 → WP05 → WP06 → WP07 → WP08 → WP09 → WP11`.

WP10 is activated only by an explicit diagnostic trigger; any changed scientific claim re-enters WP07 and uses fresh confirmation. WP00–WP02 should resolve inexpensive fatal issues before downloading large assets or investing in optimized kernels.

The line above is the **qualified GPU** dependency path. Section 6.6 defines the two shorter CPU/native-only branches. Every package may also terminate into WP11 with a recorded blocking or negative outcome; absent later artifacts must be labeled “not reached,” not fabricated.

### WP00 — Environment, scope, provenance, and license gate

**Objective/risk:** establish the actual laptop, available resources, asset access, and exactness contract. **Inputs/dependencies:** this document; no model download required. **Entry:** permission to work locally and public network access for permitted assets.

**Actions and files:** create the repository layout in Section 9; implement `python -m sec doctor --out artifacts/environment.json`. Run the existing discovery commands in Section 6, inspect dependency metadata, resolve compatible ARM64 packages, and write `requirements.lock` or an equivalent exact lock with hashes. Capture `licenses/`, source revisions, network access results, and the target definition in `protocol/scope.md`. Resolve model and data metadata without fetching large weights. Initialize `DECISIONS.md`, `PROGRESS.md`, and `protocol/amendments.jsonl`.

**Outputs/tests:** environment JSON with units in bytes, architecture, OS, package/compiler versions, device info, available disk, memory-pressure observations, and proposed resource budget; download/license manifest; import and tiny array operation checks. Confirm Python is ARM64, not translated x86. Inspect that MPFR directed-rounding contexts work on a nonrepresentable rational and that saved metadata preserves the rounding mode used. Do not assume a latest development gmpy2 page implies a stable package version; lock the actually installed compatible release. [R22], [R32], [R33]

**Resources:** negligible relative to model work; measure install/compiler footprint. **Acceptance/branches:** if the environment qualifies, continue. If MLX is unavailable, declare a CPU-only branch and retain mathematical/certificate experiments; GPU claim blocked. If essential asset access/license fails, resolve an explicit alternative before dependent work. Do not bypass access controls or change the project to a paid service.

### WP01 — Mathematical oracle and exact arithmetic harness

**Objective/risk:** validate H3 and the indispensable selection mechanism before hardware optimization. **Dependencies:** WP00. **Entry:** CPU integer/Fraction and MPFR functionality verified.

**Actions and files:** implement `sec/oracles.py`, `sec/samplers_cpu.py`, `sec/intervals_cpu.py`, `sec/random_tape.py`, and `tests/test_oracles.py`. Implement dense Gumbel, residual race with singleton shortcut, sequential squeeze with per-sample cache, Gumbel shortlist, and adaptive upper-score-first selection. Use exact rational fixtures for dot products and intervals. Implement stable `Q`, lower bound, `Q_G`, and `alpha` diagnostics with zero-width exclusions and MPFR fallback. Build `sec/build_gumbel_table.py` for both the primary dyadic table and uniform-prefix diagnostic; verify every table entry.

**Command to implement:** `python -m sec verify-math --config configs/math.yaml --out artifacts/math/`.

**Outputs/tests:** theorem-oracle comparisons, all corner checks at enumerably small dimensions, all limiting cases in Section 4.8, the three reproduced numerical fixtures, and random-tape known-answer/continuation tests. Report counts and exact failures, not “passed” alone. For small rational target probabilities, check deterministic probability identities where possible; use distribution simulations only as supplementary diagnostics. Verify the primary table's exact bin probabilities sum to one including the explicitly handled tail.

**Resources/calibration:** tiny arrays, a 4 MiB table, CPU high precision. Time table construction and exact group checks to forecast later preprocessing. **Acceptance/branches:** any counterexample to pruning or the declared theorem blocks hardware work until fixed or the mechanism is rejected. A flawed test harness is an implementation failure. Slow CPU execution is not evidence against GPU latency, but determines which oracle checks can be run broadly. Continue only with a concrete exact reference and all known edge cases handled.

### WP02 — Packed format and Metal numerical qualification

**Objective/risk:** test the critical local operations and H3's numerical assumptions. **Dependencies:** WP01. **Entry:** native MLX/Metal availability verified by WP00; the strict GPU arithmetic itself is what this package qualifies. With no native backend, take the WP01-to-WP11 branch instead.

**Actions and files:** implement `sec/metal/packed_q6.metal`, `interval_ops.metal`, `dot_interval.metal`, `dot_center.metal`, `random_tape.metal`, and `sec/numerics_contract.py`. Use tiny synthetic BF16-representable weights and FP32 activations. Validate native six-bit pack/unpack against an independent integer decoder. Qualify signed scales, all code values, row/group boundaries, padding, and strides. Map every arithmetic path to the numerical contract. Preserve generated kernel source, math mode, precision choices, reduction topology, and operation counts in `artifacts/qualification/`.

**Command to implement:** `python -m sec qualify-backend --config configs/backend.yaml --out artifacts/qualification/`.

**Outputs/tests:** exact CPU dyadic dots enclosed by every tested GPU interval with **zero added tolerance**; CPU/Metal PRNG words equal; strict comparison/candidate sets equal to the oracle on adversarial inputs. Test RTZ/FTZ arithmetic in a CPU emulator as well as actual device behavior. Exercise all underflow, max-finite, overflow, NaN, and fallback paths, including an overflowing intermediate followed by cancellation. Qualification includes a code/spec audit; passing examples alone does not establish the primitive contract.

**Resources:** small arrays first, then shape-specific synthetic memory probes only after budget checks. No pretrained model required. **Acceptance/branches:** a valid but slow interval kernel is sufficient to continue correctness study. A faster center kernel requires the proved envelope in Section 4.9. If effective compilation cannot be established, use a controllable Metal extension or retain CPU-only scope; do not mark empirical behavior certified. A six-bit layout failure is repaired here before data collection. Freeze kernel qualification IDs; later changes rerun affected qualification.

### WP03 — Asset acquisition, body adapter, and native baseline

**Objective/risk:** verify real-model semantics and measure whether the head is a relevant bottleneck. **Dependencies:** WP00–WP02, including a recorded `native_body_cpu_certified_head` outcome when applicable. **Entry:** native body/packing smoke tests pass and public asset licenses/access are verified; strict GPU qualification is required only for the GPU branch.

**Actions and files:** implement `sec/assets.py`, `sec/data.py`, `sec/model_adapter.py`, and `sec/collect_states.py`. Download to temporary files; verify sizes and SHA-256; atomically rename. Acquire Smol first, validate, then Qwen if resource fit holds. Implement final normalized hidden-state extraction using the pinned model class and preserve tied input embeddings. Parse and partition development data. Build immutable per-document hidden-state shards and token manifests. Establish D0 native head plus compiled categorical and a full decode timing boundary.

**Commands to implement:** `python -m sec acquire --config configs/assets.yaml`; `python -m sec collect-states --split development --config configs/workloads.yaml`; `python -m sec bench-native --config configs/pilot.yaml`.

**Outputs/tests:** file manifest with licenses, model/tokenizer revisions, dtype/shape/tied-weight identity, adapter-equivalence checks, development states, and a measured head/body timing breakdown. Source equivalence compares extracted-body-plus-native-head with the stock call on identical token IDs; any mismatch must be explained before SEC is connected. Do not overwrite reference states during a validation run.

**Resources:** weights as in Section 6, sequential model loading, chunked prefills, KV and state-cache budgets. Measure tokens/sec, peak memory, and disk before expanding states. **Acceptance/branches:** if the adapter is wrong, repair it; do not reinterpret the difference as SEC error. If the head is a small fraction, retain component research but bound end-to-end prospects. If Qwen cannot fit, reduce chunk/workspace first; otherwise record a narrower model scope before confirmation. No full test-set outcomes are accessed here.

### WP04 — Real certificates and inexpensive headroom screen

**Objective/risk:** test H1 and whether expensive kernel work is justified. **Dependencies:** WP03. **Entry:** verified original head, development states, and exact preprocessing oracle.

**Actions and files:** implement `sec/pack.py`, `sec/certificates.py`, and `sec/diagnose_bounds.py`. Recreate the supplied symmetric group-64 diagnostic only if the optional original script/assets are available; without them, label the result an independent native-affine diagnostic, not exact historical reproduction. Build the primary native affine Q6 pack, residual norms, and magnitude metadata from actual decoded dyadics. Process rows in checkpointed tiles. Explore supported bit/group choices using development data; begin with 4/6/8 bits and groups 32/64, then admit 3/5 bits or padded group 128 if the observed storage/certificate frontier leaves them plausible. This is a screening order, not a claim that omitted choices cannot win.

**Commands to implement:** `python -m sec build-sketch --config configs/sketch.yaml`; `python -m sec diagnose-bounds --split development --config configs/pilot.yaml`.

**Outputs/tests:** exact verification of every stored residual norm; format/hash manifest; per-state ideal `Q`, `Q_G`, lower bound, actual contenders with noise intervals, width/margin distributions, zero-query predictions, and complete modeled bytes. Independent exact-dyadic spot/edge checks supplement the all-group squared-norm check. Compare the new affine results with the supplied symmetric result without pretending the formats are identical.

**Resources/calibration:** `O(Vd)` preprocessing and each diagnostic scan; measure seconds per tile and bytes per live array. Build only one full candidate artifact at a time. **Acceptance/branches:** invalid bounds are a correctness defect. Valid but uniformly wide bounds reject that configuration. At this stage only D0 is timed: if the unavoidable cost forecast exceeds it, label practical-competitiveness headroom negative, but do not infer a same-target D1/D2 result before WP05 qualifies those controls. Continue inexpensive strict diagnostics if they answer a useful separate question; stop a costly practical branch when its own cost case is ruled out. CPU/native-only mode goes through CPU WP05 and then WP11. Otherwise continue when a real-state configuration shows plausible cost headroom, or produce the justified negative diagnostic.

### WP05 — Qualified dense and classical sampler baselines

**Objective/risk:** prevent a favorable result caused by weak competitors or differing semantics. **Dependencies:** WP04. **Entry:** valid metadata and real development inputs.

**Actions and files:** on the qualified GPU branch, implement D1 and D2 in `sec/methods/dense_certified.py` and `sec/metal/dense_fused.metal`; adapt S0–S3 to the shared `HeadOracle` interface. D1/D2 use the same random table, arithmetic qualification, strict winner test, and rare CPU refinement as SEC. Tune native/strict dense row tiles and reductions on the same development inputs and a comparable measurement budget. Benchmark simple proposal setup and synchronization to determine whether GPU implementations of S0/S1 are justified. On the native-body/CPU-certified-head branch, implement only the certified CPU dense and S0–S3 references needed for correctness/query evidence, then report; a failed custom shader is not a dependency of that branch. Keep CPU-only algorithms explicitly labeled.

**Command to implement:** `python -m sec qualify-baselines --config configs/baselines.yaml --out artifacts/baselines/`.

**Outputs/tests:** baseline manifest with implemented guarantee and timing boundary; appropriate coupled agreement; kernel/parameter tuning records; complete setup+sample costs. D1/D2/S2/S3 use identical Gumbel tapes and must agree. S0 is coupled to its full proposal/height sequence with all needed masses available; S1 is coupled to the unpruned complete residual race using the same clocks. Do not demand identical tokens across different constructions merely because their distributions coincide. D2 computes all original rows and retains every interval contender under Section 5.2's tile/global rules; it must not use an unnecessarily expensive MPFR full projection. Verify D0 is not degraded by avoidable dtype casts or host transfers.

**Resources:** model plus one method's workspace; no training. **Acceptance/branches:** a baseline with an incorrect policy or target is repaired before comparison. A slow custom dense kernel does not establish competitiveness; improve it or qualify the stronger available baseline and narrow the claim. If dense fusion captures the entire benefit, record a useful negative result for selective reads. Proceed with a written baseline-fairness checklist tied to actual code, not just names.

### WP06 — Minimum complete selective GPU path

**Objective/risk:** test H2 with all necessary overhead present. **Dependencies:** WP05. **Entry:** a valid strict arithmetic path, candidate cost headroom, and qualified dense baselines.

**Actions and files:** implement `sec/methods/selective_gumbel.py`, sketch+bound+noise screening, global threshold, masked full-row refinement, final strict winner reduction, and CPU fallback. Integrate the exact dyadic table and counter mapping. Use fixed-vocabulary flags first; compact only after profiling. Maintain per-sample counters and an emergency same-tape dense fallback. Expose pure SEC separately from the development-trained dispatch policy.

**Command to implement:** `python -m sec verify-method --method selective_gumbel --split development --config configs/sec.yaml`.

**Outputs/tests:** no mismatches with the same-target dense oracle in small exhaustive/adversarial cases and development replay; all tested exact dots enclosed; all fallback branches forced successfully. Return a proof-status record for every token. Check row/metadata hashes and candidate IDs. Fail closed on an empty contender set, invalid interval intersection, nonfinite input, or inconsistent random-tape cursor.

**Resources:** `O(V)` scratch, the shared table, packed sketch, original model, and bounded transfer buffers; no `V×d` gathered temporary. Measure peak memory at worst-case `K=V`. **Acceptance/branches:** correctness failures block latency claims. A correct slow implementation may proceed to a targeted profile if an identified overhead is reducible; otherwise publish the negative component finding. It does not justify a broad rewrite without a new cost hypothesis.

### WP07 — Pilot, parameter selection, resource forecast, and freeze

**Objective/risk:** resolve variance, overhead, baseline tuning, and configuration choices before confirmation. **Dependencies:** WP06. **Entry:** methods correct on development data.

**Actions and files:** implement `sec/benchmark.py`, `sec/pilot.py`, and `sec/preregister.py`. Measure complete costs across candidate bit/group configurations, tile sizes, layout/compaction choices, cache regimes, and temperatures. Prune configurations only for measured correctness failure, domination under declared metrics, or a proved hardware incompatibility. Select one primary configuration and baseline on development data; the primary cache regime remains varied-state replay. Fit the dense crossover from timings. Estimate document/state/tape/timing variance, force slow paths and estimate natural fallback incidence as specified in Section 5.5. Set a fixed confirmation schedule, conditional-session inference design, precision goal, multiplicity gates and absolute practical-use rationale.

**Commands to implement:** `python -m sec pilot --config configs/pilot.yaml`; `python -m sec freeze-protocol --config configs/confirmation_draft.yaml`.

**Outputs/tests:** development frontier; cost-model residuals; memory/throughput forecast with uncertainty; sample-size/repeat calculation; configuration provenance; immutable protocol/source/dependency hashes. Check analysis on equivalent-method labels and known injected differences. Estimate total duration as preprocessing plus state generation plus all fixed timed calls and expected refinement; include forecast error from pilot rate variability.

**Resources:** calibrated from actual per-stage rates. **Acceptance/branches:** freeze only after all decision-critical choices are explicit. If useful effects are below feasible sensitivity, classify the planned result as potentially inconclusive and state the detectable range. If no competitive candidate survives, proceed to WP11 with a negative result or activate a specifically justified WP10 refinement; do not access the test set to search for a winner.

### WP08 — Protected component confirmation

**Objective/risk:** confirm the primary claim without adaptive selection. **Dependencies:** WP07. **Entry:** protocol/source/config hashes match and protected workload IDs are sealed.

**Actions and files:** implement `sec/confirm.py` to generate/retrieve protected states, run complete head methods in balanced blocks, and retain raw observations. A separate evaluator may compute full exact/reference information for correctness diagnostics, but it is not exposed to method selection. Complete every preregistered primary session before unblinding the primary result. Run the fixed schedule; retain failures, tails, and unfavorable conditions. Analyze only with the locked script.

**Command to implement:** `python -m sec confirm --protocol protocol/confirmatory.yaml --out runs/confirmatory/`.

**Outputs/tests:** raw JSONL/Parquet, per-document aggregates, confidence intervals, primary/secondary contrasts, correctness/fallback counts, actual resource use, and all exclusions. Every observed mismatch is investigated with a self-contained counterexample; a single genuine certificate violation invalidates the strict claim. A nonzero distribution-test p-value cannot override a mismatch.

**Resources:** fixed sample and repetition schedule from WP07, checkpointed by balanced timing block. **Acceptance/branches:** CI above zero with valid implementation supports the component effect; comparison against D0 determines practical competitiveness. CI containing zero is inconclusive at the attained precision. CI below zero is evidence against latency advantage in the declared regime. A severe tail/memory regression remains an explicit tradeoff even with a favorable mean. Do not retune and reuse the same protected data for confirmation.

### WP09 — Integrated decode, shift, and independent confirmation

**Objective/risk:** test H4, cache/state confounds, and robustness. **Dependencies:** WP08 or a preregistered component-negative diagnostic. **Entry:** integrated adapter passes same-history state checks.

**Actions and files:** implement `sec/decode.py` and workload profiles for body-interleaved replay and genuine autoregressive generation. Compare certified dense and SEC under identical per-step tapes and policies. Preserve KV hashes or selected exact state checkpoints and token histories to localize any divergence. Run protected code/math shift workloads using frozen configurations; run an additional balanced session on a different day/thermal initialization as a separate secondary replication without changing the method. Do not append it to WP08's already frozen primary estimate. Include selected context/batch/temperature boundaries from Section 8.

**Command to implement:** `python -m sec confirm-decode --protocol protocol/decode_confirmatory.yaml`.

**Outputs/tests:** token-by-token matched certified trajectories, full request/decode timings, head fraction, EOS/length behavior, cache-regime contrast, cross-domain and session estimates. A head-only cached-state result is never substituted for this output. For D0, use separately defined practical workload comparisons because its finite sampler does not necessarily share trajectories.

**Resources:** KV scaling formula and measured body throughput; sequential batches by available memory. **Acceptance/branches:** a consistent end-to-end gain supports only the tested policies/hardware/models. Loss of speed after body interleaving narrows the finding to the component or cache regime. Synthetic success without real transfer rejects generality. If the component was clearly slower, a small integrated diagnostic may explain why but does not demand a long generation campaign.

### WP10 — Evidence-triggered refinements or mathematical extension

**Objective/risk:** actively seek a defensible improvement when a specific failure mechanism is identified. **Dependencies:** relevant diagnostics from WP04–WP09. **Entry:** a written trigger, baseline, cost estimate, and falsifier for exactly one proposed refinement.

**Actions and files:** follow Section 10. Examples are tighter norm combinations when certificate slack dominates; a common residual shift when measured common structure exists; a low-rank residual correction when its extra bytes are plausibly cheaper than another bit; block reads when RNG/compaction/gather locality dominates; or pre-screen dense dispatch when a predictable regime is unfavorable. Write `protocol/amendments.jsonl` before changing the primary method. Implement in a separately identified method/config, preserving old results.

**Outputs/tests:** matched ablation against the simplest equivalent improvement, proof of any modified certificate, complete metadata/preprocessing/latency accounting, and a decision about fresh independent confirmation. For a theorem contribution, write a self-contained statement/proof with its oracle restrictions and distinguish the elementary block corollary from an unproved general optimality claim.

**Resources:** forecast from the relevant measured bottleneck; do not implement all extensions by default. **Acceptance/branches:** proceed only if the refinement beats its matched simpler control on development data and has a credible mechanism. If it is known prior art, label it accordingly. If it adds complexity without value, retain the negative result and stop that branch. No possible future refinement excuses a negative result for the tested implementation.

### WP11 — Reproduction, audit, and result synthesis

**Objective/risk:** produce a reviewable research result whether positive, negative, or inconclusive. **Dependencies:** completed or explicitly blocked/terminated prior packages. **Entry:** raw artifacts and decisions are preserved.

**Actions and files:** implement `sec/report.py` and `sec/reproduce.py`. For each reached stage, reproduce a representative artifact or check: rebuild an asset shard if one exists, replay the numerical counterexamples if WP01 was reached, and repeat a balanced timing block only if timing qualification was completed. Compare artifact hashes, decoded weights, target tokens and numeric qualification IDs as applicable; timing need not be bit-identical. An early blocked report reproduces its environment/failure evidence without requiring models, GPU shaders or a confirmation protocol that never existed. Conduct the three audits in Section 11 and update conclusions without deleting history.

**Commands to implement:** `python -m sec reproduce --manifest artifacts/reproduce_manifest.json`; `python -m sec report --run-index artifacts/run_index.json --include-failed-runs`. Add `--protocol protocol/confirmatory.yaml` only when a confirmation freeze exists. The run index enumerates reached/blocked stages and valid artifact manifests; the report generator must accept an early negative or blocked study.

**Outputs/tests:** `RESULTS.md`, machine-readable summary, plots/tables derived from raw data, environment/dependency/source/asset manifests, license notices, limitations, and a restart guide. Report what was actually executed, all relevant baselines, uncertainty, absolute effects, tail/memory tradeoffs, and the negative configurations. Use separate language for mathematical, component, and end-to-end findings.

**Resources:** a reproducible subset of expensive work, chosen to cover critical paths rather than blindly rerunning everything. **Acceptance/branches:** missing provenance or inconsistent resumed state blocks a reproducibility claim. A clean partial negative study is a valid completion. Publication-level novelty still needs broader scholarly scrutiny; this plan and its internal review do not establish universal priority or external peer review.

### 7.1 Failure classification used at every gate

| Code | Failure class | Required interpretation and next action |
|---|---|---|
| `IMPLEMENTATION_INVALID` | Wrong decoder, interval, state extraction, random tape, or timing boundary | Repair on development fixtures; invalidate affected results. No scientific rejection yet. |
| `PROXY_UNSUITABLE` | Tiny shape cannot express bandwidth savings, or synthetic states omit real structure | Move to a justified real/local scale after a cost model; do not infer broad rejection. |
| `RESOURCE_BLOCKED` | Memory, OS, compiler, asset access, or storage prevents the intended path | Use a faithful smaller/chunked/CPU branch and narrow the claim; mark untested target explicitly. |
| `UNDERPOWERED` | CI is too broad for the relevant effect | Report attained precision; preregister fresh extension if worthwhile. No significance chasing. |
| `MECHANISM_NEGATIVE` | Valid bounds fail to prune on representative states | Reject/narrow that certificate regime; investigate a specific supported refinement if justified. |
| `NO_COMPETITIVE_ADVANTAGE` | Mechanism works, but complete cost is worse than strong baselines | Report the result and useful boundary; do not call query savings a speedup. |
| `COMPONENT_ONLY` | Head improves but complete decode does not | Retain component result with Amdahl/cache explanation; no end-to-end claim. |
| `CONFIRMED_TRADEOFF` | Mean improves while memory, tails, or preprocessing worsen | Report the full frontier and application-dependent usefulness. |

## 8. Attribution, robustness, and failure analysis

### 8.1 Required attribution controls

| Question | Control/ablation | Interpretation |
|---|---|---|
| Are selective original-row reads responsible? | SEC versus D1/D2 with identical certified arithmetic, random tapes, table, and reduction improvements | If only an older unfused baseline loses, the demonstrated contribution is generic fusion. |
| Is the certificate information useful? | Replace valid bounds by deliberately wide valid bounds while keeping the same sketch; use zero-width synthetic bounds as the opposite limit | The first should approach dense correction cost; the second should need no original-row reads, aside from separately identified numerical ambiguity. |
| Is querying fewer rows sufficient? | Compare S0, S1, S2, S3 with identical intervals and row oracle | Separate distinct queries, full-row bytes, sequential synchronization, and total time. |
| Are bits or metadata driving the gain? | Fixed codes with different valid certificate representations; fixed sampler across 4/6/8-bit sketches and supported group sizes | Choose the measured frontier. Do not compare a richer metadata method against a stripped baseline without accounting. |
| Is the LUT merely slowing an unnecessary baseline? | Give both dense and selective methods the same primary LUT; compare conventional finite Gumbels only in a labeled weaker-guarantee mode | Strict and empirical numerical contracts cannot be mixed. |
| Do random-prefix tails dominate? | Uniform-prefix versus dyadic lookup, same ideal target and same continuation semantics | Isolates a numerical implementation choice already anticipated by prior art. |
| Is memory locality the real cause? | Actual candidates, equal-count random candidates, contiguous candidates, and row permutations with consistent token mapping | Separates candidate information from row clustering/cache behavior. Random candidates are timing diagnostics, not samplers. |
| Is a dispatcher hiding losses? | Report pure SEC, dispatched SEC, and always-dense on every workload | The dispatcher is a separate engineering feature, not evidence that every regime benefits. |
| Is the apparent result an artifact of replay? | Cached head replay versus varied states versus body-interleaved decode | Narrows claims to the regime actually improved. |

Do not run a full Cartesian product of all dimensions without purpose. Use development diagnostics to identify plausible interactions, then preregister a limited confirmatory set that addresses those interactions. Report omitted boundaries explicitly.

### 8.2 Boundary conditions that must be represented

- **Distribution shape:** diffuse, sharply peaked, nearly tied leading logits, heavy-tailed probabilities, heterogeneous interval widths, and real low-probability winners. A cheap method must not lose rare tail tokens.
- **Temperature:** primary 1; development sensitivities 0.2, 0.5, and 2, plus extreme positive values in correctness tests. These are different workloads, not an opportunity to choose a favorable policy after testing.
- **Batch and context:** primary batch one; test small batches only after calibration because dense GEMM may gain more reuse than SEC. Include short and longer context profiles within measured memory limits, with the exact same KV dtype and body implementation.
- **Numerical stress:** large positive/negative common shifts, zero-width intervals, extreme dynamic range, subnormal values, signed zero, cancellation, exact finite ties, near real ties, invalid metadata, and maximal finite intermediates.
- **Storage/layout:** non-contiguous source views, explicit row padding, token-ID permutations, tied embeddings, negative scales, corrupt/truncated packs, and incompatible cache versions.
- **Operational stress:** interrupted downloads, interrupted preprocessing, partial timing blocks, paused exact refinements, GPU allocation failure, session restart, and a resumed run with changed dependencies.

A corrupted sketch must be rejected by provenance checks. A deliberately weaker but valid certificate should cost more work while retaining the same answer. A tighter invalid certificate should be detected by the exact enclosure tests; it must never be a tuning option.

### 8.3 Interpreting a gain or failure

For every apparent gain, ask in this order: Is the target and guarantee unchanged? Are bounds valid? Is the same random experiment being compared? Does the gain survive the fastest dense implementation and shared optimizations? Are all setup/metadata/fallback costs included? Does it persist on protected documents and a fresh timing session? Does it improve the whole application or only the head? What regressions accompany it?

For a negative result, localize whether failure comes from certificate slack, arithmetic proof margins, random interval width, packed GEMV throughput, row-load inefficiency, host synchronization, memory residency, or body dominance. Measure each cause instead of adding algorithmic complexity speculatively. A conservative certificate failure can motivate a specific valid bound refinement; it does not justify empirical clipping or replacing rare uncertain samples with approximate decisions.

## 9. Repository, interfaces, experiment records, and resumability

### 9.1 Minimal repository layout

The paths below are proposed local project paths. Implement only modules needed by admitted work packages; keep command dispatch thin.

```text
selective-exact/
  README.md
  pyproject.toml
  requirements.lock
  PROGRESS.md
  DECISIONS.md
  RESULTS.md
  numerics_contract.md
  licenses/
  protocol/
    scope.md
    confirmatory.yaml
    decode_confirmatory.yaml
    analysis.md
    amendments.jsonl
  configs/
    assets.yaml
    math.yaml
    backend.yaml
    workloads.yaml
    sketch.yaml
    baselines.yaml
    sec.yaml
    pilot.yaml
    confirmation_draft.yaml
  sec/
    __main__.py
    assets.py
    data.py
    model_adapter.py
    collect_states.py
    pack.py
    certificates.py
    intervals_cpu.py
    numerics_contract.py
    oracles.py
    random_tape.py
    build_gumbel_table.py
    samplers_cpu.py
    diagnose_bounds.py
    benchmark.py
    pilot.py
    preregister.py
    confirm.py
    decode.py
    report.py
    reproduce.py
    methods/
      native_dense.py
      dense_certified.py
      selective_gumbel.py
      residual_race.py
      squeeze.py
      adaptive_gumbel.py
    metal/
      packed_q6.metal
      interval_ops.metal
      dot_interval.metal
      dot_center.metal
      random_tape.metal
      dense_fused.metal
      selective.metal
  tests/
    test_oracles.py
    test_packing.py
    test_enclosures.py
    test_random_tape.py
    test_model_adapter.py
    test_resume.py
  data/
    manifests/
    raw/
    states/development/
    states/protected/
  assets/
    original/
    sketches/
    gumbel_tables/
  artifacts/
    environment.json
    qualification/
    math/
    baselines/
    run_index.json
    reproduce_manifest.json
  runs/
    exploratory/<run_id>/
    confirmatory/<run_id>/
    failed/<run_id>/
```

Do not commit model/data payloads into ordinary Git history. Commit code, schemas, small manifests, locks, protocols, decisions, and report-generation code. Checkpoints and raw numerical outputs remain content-addressed artifacts with a restart manifest. The project may use a local repository without any external service.

### 9.2 Shared interfaces

These are proposed signatures; dataclasses may use NumPy/MLX-backed arrays with the stated logical types.

```python
class HeadAsset:
    model_revision: str
    weight_sha256: str
    vocabulary_sha256: str
    V: int
    d: int
    d_padded: int
    source_dtype: str              # "bfloat16" for the selected models
    packed_codes: object          # uint32/native row-major bytes
    scales: object                # float32 [V, d_padded/group_size]
    affine_biases: object         # same shape/dtype
    residual_norm_up: object      # nonnegative outward float32, same shape
    group_size: int
    bits: int
    original_rows: object         # lossless, row-addressable values
    output_bias: object | None    # original finite [V] values; not affine_biases
    certificate_id: str

class HeadOracle:
    def sketch_intervals(self, h, asset) -> "LogitIntervals": ...
    def original_intervals(self, h, row_ids, asset) -> "LogitIntervals": ...
    def exact_dyadic_rows(self, h_bits, row_ids, asset) -> object: ...

class HeadMethod:
    method_id: str
    guarantee: str
    def prepare(self, asset, config) -> "PreparedMethod": ...
    def sample(self, h, sample_id, rng_seed, temperature) -> "SampleResult": ...
    def synchronize(self) -> None: ...

class SampleResult:
    token_id: int | None
    status: str                   # certified / native_finite / unresolved / invalid
    guarantee: str
    certificate_id: str | None
    telemetry: dict
    resume_state: dict | None
```

`LogitIntervals` contains row IDs, outward lower/upper endpoints, arithmetic contract ID, and validity flags. Row IDs are unique, in range, and mapped to original token IDs. Shape/dtype/stride assumptions are checked at method preparation, outside timed per-token work where safe; data-dependent numeric domain checks remain inside the timed boundary. No method interface exposes evaluation-only dense logits, target labels, test-derived temperature choices, or a reference winner.

Logit intervals include the original output bias exactly once. The full declared vocabulary support includes every head row, including reserved IDs without an ordinary text rendering. Integrated decoding feeds the sampled valid row ID into the same model embedding; it must not resample or silently mask a token because its display string is empty or unavailable. Logs may display the numeric ID. Any intentional vocabulary mask is a separately frozen policy shared by all comparators.

`prepare` may create static assets once, but anything depending on the current hidden state belongs to the measured sample operation. A row cache is per state/sample unless a mathematically legitimate repeated-state workload is explicitly being measured. Distinguish original row bytes resident in memory from an already-computed logit; persistent weight residency is allowed for all methods, whereas secretly reusing evaluated logits across samples changes the work.

### 9.3 Configuration conventions

Use explicit units and reject unknown config keys. YAML is acceptable; save its canonical JSON form and SHA-256 alongside each run. Initial method settings are:

```yaml
schema_version: 1
experiment_phase: exploratory
reference:
  law: real_dot_full_softmax
  weight_dtype: bfloat16
  hidden_storage_dtype: float32
  body_quantization: none
  temperature_requested_text: "1.0"
  temperature_f32_bits: "0x3f800000"   # normative, exact dyadic target
  top_k: null
  top_p: null
  min_p: null
  fixed_mask: null
method:
  name: selective_gumbel
  quantization: mlx_affine
  bits: 6
  group_size: 64
  metadata_dtype: float32
  arithmetic: qualified_center_or_interval
  gumbel_table: dyadic_side_exponent_mantissa
  mantissa_bits: 12
  table_exponents: 64
  overflow_behavior: same_tape_certified_fallback
  dense_dispatch: disabled_until_pilot
workload:
  batch_size: 1
  initial_context_tokens: 2048
  state_selection: uniform_within_document
  primary_cache_regime: varied_state_replay
  consumer_boundary: host_scalar_token_and_status_all_streams_complete
analysis:
  primary_unit: document
  primary_metric: mean_paired_head_latency_us
  inference_over_sessions: conditional_on_frozen_sessions
  session_weights: equal
  confidence_level: 0.95
  confirmation_sample_size_source: frozen_pilot_design
```

The exploratory string `qualified_center_or_interval` must be replaced by a concrete kernel qualification ID before confirmation. Missing sample sizes, timing repeats, dispatch thresholds, or selected baseline IDs are legitimate **pilot outputs**, but the freeze command must refuse to produce a confirmatory manifest until they are concrete. This avoids inventing sample sizes while preventing an ambiguous execution handoff.

### 9.4 Run and observation schemas

Each run's `manifest.json` must contain:

```text
schema_version, run_id, phase, status, start_utc, end_utc,
source_commit, source_dirty_patch_sha256, dependency_lock_sha256,
protocol_sha256, canonical_config_sha256, environment_sha256,
model_repository, model_revision, model_file_hashes,
tokenizer_hashes, dataset_revision, document_manifest_sha256,
hidden_state_manifest_sha256, weight_layout_sha256,
sketch_sha256, certificate_sha256, gumbel_table_sha256,
kernel_source_sha256, compiled_options, qualification_id,
rng_algorithm, rng_mapping_version, seed, sample_id_range,
temperature_f32_bits, temperature_requested_text,
timing_boundary, cache_regime, replay_order_hash,
warmup_rule, block_order_seed, session_weights,
resource_budget_bytes, actual_peak_memory_bytes,
output_shard_hashes, failure_or_exclusion_reason, parent_run_id
```

Fields belonging to stages not reached are `null`, with an explicit `not_reached` or `not_applicable` reason in a parallel `missing_fields` mapping. Only applicable existing artifacts are required and verified on resume. Empty strings, dummy IDs and invented hashes are forbidden. `run_index.json` lists stage status, run manifests and applicable protocols, so a WP00/WP01-only result is a valid report input.

Each raw timing observation has typed fields:

| Field group | Required values |
|---|---|
| Identity | run/method/model/document/chunk/position/sample/timing-block/session IDs; exploratory or confirmatory flag |
| Workload | `V,d,d_p,B`, context tokens, canonical temperature bits, support-mask ID, hidden-state hash |
| Timing | elapsed nanoseconds as integer; stage nanoseconds where measured; complete boundary flag; synchronization/stream IDs |
| Result | token ID or null; guarantee/status; coupled reference token if evaluator supplies it; mismatch classification |
| Work | ideal diagnostic `Q/Q_G` when available; actual initial/final contenders; original distinct rows; row-read count; block count; modeled bytes; fallback kind/count/time |
| Numerics | maximum bound width; arithmetic/noise contribution; MPFR precisions/rounds; random words consumed; invalid/overflow flags |
| Resources | device active/cache/peak bytes, process RSS, swap/pressure observations, compile/warm status |
| Integrity | config/protocol/kernel/certificate hashes; failure/exclusion reason; raw record checksum |

Record unavailable hardware counters as `null` with a reason, never as zero. Stage timers may perturb performance; collect detailed profiles in a separate labeled run and use minimally instrumented complete timings for the primary comparison. Counters needed for correctness and fallback accounting remain consistent across methods.

Store long arrays and state data in versioned binary shards; JSONL/Parquet is suitable for records. Use explicit endianness and BF16 raw-bit conventions. Failure records include the smallest reconstructible input, original/packed rows and metadata, hidden bits, canonical temperature bits, random tape prefixes/cursors, intervals, source/qualification IDs, and the failure explanation. Redact machine identifiers and avoid copying unnecessary source text into reports.

### 9.5 Checkpointing and resume rules

Downloads use `temporary → verified size/hash → atomic rename`; mere file existence is not completion. Preprocessing checkpoints each completed immutable row tile and its checksums. A state shard becomes visible only after its document IDs, hashes, model/body config, and token positions are complete. Do not overwrite custom and framework-reference caches with each other.

Timing blocks are transactional. If interrupted, mark the entire incomplete balanced block invalid, retain its partial records, and rerun the whole block under the original schedule. Do not splice a warm half-block into a cold resumed block. A resumed exact sample saves the candidate set, all interval endpoints, canonical temperature bits, consumed random prefixes/cursors, precision levels, and relevant asset hashes; otherwise restart that sample from its original sample ID and seed, not a newly sampled seed. A method-dependent difficult sample is still counted as unfinished until completed; interruption bookkeeping must not erase its cost or convert it into an ordinary external-failure exclusion.

On resume, verify all applicable source/dependency/config/model/tokenizer/data/kernel/certificate/table hashes before accepting any cached artifact. Null fields for unreached stages are not missing existing dependencies; a missing hash for an artifact that was actually used is an integrity failure. A changed dependency or layout creates a new run lineage and requalification gate. Record OOM, allocation failure, disk exhaustion, crashes, and numerical failures without deleting the failed run. Reducing batch/context or changing a fallback is an explicit config change; timing comparability is re-established before mixing results.

`PROGRESS.md` records the current work package, completed gates, pending decisions, exact restart command, and artifact locations. `DECISIONS.md` explains why a configuration, baseline, or branch was admitted or rejected. `protocol/amendments.jsonl` includes old/new protocol hashes, date, rationale, data already observed, affected claims, and fresh-confirmation requirements.

### 9.6 Required result presentation

The final research report must include the full cost frontier, primary paired effect in microseconds and percent, uncertainty, all relevant negative configurations, measured memory/preprocessing/tail regressions, and the strongest baseline used. Include candidate/latency scatter or distributions, candidate causes, head fraction/Amdahl projection versus observed end-to-end effect, and a resource/timing ledger. Use plots only from preserved raw results; label estimates separately.

Possible headline vocabulary is deliberately constrained:

- “Certified conditional head sampler” requires the numerical contract and valid fallback.
- “Faster head” requires a complete timed head comparison.
- “Faster generation” requires integrated decode evidence.
- “Less traffic” requires a declared logical model or actual counter, named accordingly.
- “Smaller model” is unavailable when the original head plus sketch is larger.
- “Novel theorem” requires a separate priority argument; “candidate formulation” is the present status.

## 10. Outcome decisions and evidence-triggered refinements

### 10.1 Decision table

| Result | Conclusion | Next action |
|---|---|---|
| Certificates valid; protected head effect passes its gate; native practical baseline also beaten; end-to-end gain survives and serves the stated use case | Useful local exact-head engineering result, with quantified limits | Reproduce, release allowed code/artifacts, then test a larger local head only for a specific scaling prediction |
| Same-target strict baseline beaten, but native dense remains faster | Selective computation reduces the cost of certification; deployment speed advantage unproven | Report both numbers; identify whether exact-law guarantees justify extra cost for a real use case |
| Early unavoidable-cost forecast already exceeds D0; D1/D2 not yet qualified | Practical native headroom is negative under that forecast; strict comparative result is unmeasured | Stop the costly practicality branch or complete useful inexpensive strict diagnostics; report what was not reached |
| Query/byte savings, no complete head gain | Mathematical mechanism works but current hardware realization is not competitive | Profile and pursue one cost-specific refinement, or stop with a boundary result |
| Head improves; whole decode does not | Component result only | Explain measured body fraction, cache effects, and added synchronization; no system-speed claim |
| Valid Q6 bounds fail to prune on real states | Supplied favorable diagnostic does not generalize to this format/population | Compare format/group/extra-bit controls; investigate valid tighter certificates only if there is residual cost headroom |
| Synthetic success, realistic failure | Proxy omitted an essential distribution or systems property | Retain synthetic theorem check; reject practical generalization without contrary real evidence |
| Precision insufficient or confidence intervals broad | Inconclusive | Report achievable resolution and a justified fresh-study cost; no claim of equivalence or no effect |
| Any numerical certificate failure | Current strict implementation invalid | Preserve counterexample; repair and requalify; invalidate affected protected outcomes |
| Close prior art resolves priority | Replication or engineering adaptation | Keep the useful measurement; remove historical novelty language |

### 10.2 Tighter certificates without losing the guarantee

If `eps_i`, rather than arithmetic or random intervals, dominates surviving candidates, compare

`min(||r_ig||_2||h_g||_2, ||r_ig||_1||h_g||_infinity, ||r_ig||_infinity||h_g||_1)`

per group, with each term independently enclosed. This is established Hölder duality. Extra norms cost metadata and arithmetic; compare against spending those bytes on a smaller group or another sketch bit. An ordinary full-row L2 bound cannot mathematically tighten exact block-L2 bounds, since `Σ_g||r_g||_2||h_g||_2 <= ||r||_2||h||_2`; it may be a cheaper metadata alternative, not a stronger bound.

If rounding residuals contain a measurable common component, choose a fixed `c` offline and write `W_i-W_hat_i=c+r_i`. Then certify shifted logits

`z_i-c·h = W_hat_i·h + bias_i + r_i·h`.

Softmax is unchanged by the shared shift. Selected original-row refinement must evaluate that same shifted score, or explicitly include `c·h` with a certified interval. Mixing shifted bounds with unshifted fine logits is a correctness failure. Compare zero shift, simple mean shift, and any development-selected shift with identical bits and budgets. Common-row shifts have current prior art; this is a possible certificate improvement, not a new symmetry. [R13]

If the residual has concentrated low-rank structure, fit `R≈A B^T` on development/static weights, add `A(B^T h)` to the sketch, and certify the remaining residual. Costs include `V*r` coefficients, `d*r` factors, their arithmetic errors, activation movement, and launches. Compare with extra sketch bits, no-correction approximate head quality, and known low-rank error reconstruction. Use rank only after measuring singular-value/activation structure and full cost; do not train a new model to rescue the premise. Low-rank correction itself, including an output-head application, is established. [R15], [R40]

### 10.3 Disjoint block oracle extension

If measured gather/RNG/compaction costs dominate, partition rows into disjoint physical blocks `B`. In the independent positive-weight box, define `D_B=Σ_{i∈B}(u_i-l_i)` and `p_B=Σ_{i∈B}l_i/L`.

Draw the usual baseline clock/label and one independent `E_B~Exp(1)` per block. Query block `B` when `E_B<D_B*T`. After reading it, compute `R_B=Σ_{i∈B}(f_i-l_i)` and event time `E_B/R_B`. If the block event wins, sample inside it proportionally to `f_i-l_i` with independent randomness. The superposed rates give the original categorical law.

The expected number of queried blocks is

`Q_block=Σ_B D_B/(L+D_B)`.

A coupling lower bound for a block query is at least `[D_B/(L+D_B)](1-p_B)`, because raising that whole block changes its aggregate probability by that amount. The full TV distance may be larger; equality is not asserted. Thus the minimax sandwich has gap less than one block. With fixed nonnegative query costs `c_B`, the analogous weighted gap is below `max_B c_B` for nontrivial positive costs. This is a direct extension of the supplied argument, not a general hardware optimality theorem.

**Critical matched control:** rowwise residual race followed by coalescing touches block `B` with the same probability, because conditional on `T`, no row in the block is selected with probability `exp(-D_B*T)`. A block implementation therefore cannot claim a new block-traffic reduction merely from aggregation. It might save RNG, scheduling, or repeated loads; it may compute extra unneeded row values. Measure those differences. The theorem does not apply directly to overlapping cache lines, a Gumbel shortlist, or arbitrary matrix-engine costs.

### 10.4 Scale, probability-gap headroom, and stopping

Increasing `V` can amortize fixed per-call/launch overhead against a larger weight scan, but fixed per-row overhead and weight traffic both scale with `V` at fixed `d`; that ratio need not improve. Candidate tails and cache behavior also change. Increasing `d` increases saved full-row work and may widen residual/numerical bounds. Larger batches favor dense matrix reuse. These effects have competing signs; test measured scaling slopes, not vocabulary size alone.

For ideal independent Gumbels and scores `z/τ+G`, the largest-two-score gap obeys the elementary bound `P(gap<=x)<=1-exp(-x)<=x`. It follows by representing scores as exponential races: conditioned on the first event, the remaining arrival rate is no greater than the original total rate. For the unscaled scores `z+τG`, replace `x` by `x/τ`. If all fine-score intervals have a proved width at most `w`, ambiguity implies a true gap no larger than `2w`, giving a conservative bound `1-exp(-2w/τ)`. This deduction can motivate additional numerical precision; it does **not** bound fallback latency or justify replacing a rare event with an approximate answer. A varying unbounded interval width requires its own conditioning argument, so use the uniform bound only when its assumptions are actually satisfied.

Stop an implementation branch when its valid certificates and measured unavoidable costs leave no plausible complete advantage, or when the relevant protected comparison is clearly negative and no specific new mechanism is supported. Continue when a measured bottleneck has a concrete removable cause with a favorable revised cost budget. Narrow the claim when local resources or tested scale do not cover it. Document any remaining scale hypothesis and its discriminating prediction; neither reject it solely from an unsuitable toy nor preserve it indefinitely without evidence.

## 11. Audit record, unresolved risks, and limits of the handoff

### 11.1 Review procedure and concrete corrections

After drafting, the research used three separate assistant review passes: scientific/prior-art review, experimental-validity/numerical review, and an implementation walkthrough. They are distinct internal challenges, not external peer review, independent laboratory replication, or verification on the target laptop. Corrections were propagated into the mathematical specification, pseudocode, evaluation protocol, work-package dependencies, interfaces and reporting requirements.

| Draft audit | Material defects found and revised | Targeted follow-up |
|---|---|---|
| Scientific/prior-art | Missing general output bias; ambiguous temperature; incorrect per-row scaling sentence; approximate expectation presented as exact; unnecessary reads of known logits | Formulas, canonical bits, query mask, shifted target, exact fixture expression and novelty/source limits rechecked; no further substantive correction identified |
| Experimental validity/numerics | Unlocked cache regime; crossed sessions treated ambiguously; unobserved rare-fallback costs; cross-construction token coupling; unclear multiplicity gate | Varied replay, fixed-session document vectors, tail sensitivity, construction-specific oracles and gatekeeping rechecked; a follow-up clarified that WP09 replication stays outside WP08's primary CI; final pass found no further substantive correction |
| Implementation walkthrough | Unsound possible one-winner-per-tile interpretation; GPU dependencies in CPU fallback; early D0 gate conflated with strict baseline; report required nonexistent confirmation files; overstated symmetric-format reproduction | Contender retention, branches, scoped gates, reporting and D0 settings rechecked; a follow-up defined null/reason fields for unreached stages; final pass found no further substantive correction |

All three final targeted challenge passes completed on the revised handoff. This means no further substantive plan correction was identified in those passes, not that the future implementation, its novelty, or its performance has been proved. Final artifact checks verified UTF-8 readability, the twelve requested sections, resolved source identifiers, balanced code fences and parseable Python/YAML examples; these are document-integrity checks, not execution of the proposed package.

The following record also includes consequential weaknesses resolved during evidence inspection and small validation, before the complete-draft audits.

| Weakness identified during research and review | Correction carried into the plan | Acceptance condition or residual risk |
|---|---|---|
| A broad novelty claim would overlap squeeze rejection, bounded Gumbel search, interval refinement, and current head screening | Sections 1 and 3 restrict differentiation to a particular certified implementation, its empirical frontier, and a separately qualified narrow theorem | Historical priority remains unresolved; publication-level claims require the missing source/code checks |
| Supplied query expectations could be mistaken for measured speed or GPU traffic | Sections 2 and 5 separate saved outputs, newly executed checks, logical bytes, hardware measurements, and complete latency | No M3 speed result exists in this plan |
| Heuristic residual padding is not a numerical proof | Sections 4.1, 4.5 and 4.9 specify exact residual inequalities, outward operations, a graph-specific error radius, and CPU refinement | Device/compiler qualification is blocking; tests alone do not discharge it |
| A finite uniform-prefix Gumbel table would produce frequent infinite upper-tail intervals at realistic vocabulary sizes | Section 4.4 specifies a probability-preserving dyadic-tail table and continuation of the original bits | Ordinary near-tie frequency and the complete RNG/table cost are unmeasured |
| GPU assumptions about rounding, FTZ and overflow could silently break a certificate | The numerical contract covers RNE/RTZ, subnormal input routing, explicit intermediate FTZ and per-operation overflow guards | The generated instruction graph must match this contract |
| Formula cancellation, underflow and known coordinates caused concrete diagnostic errors | Section 2.5 records three reproduced failures; Section 4 uses exclusion sums/log arithmetic and excludes known coordinates from charged reads | These become mandatory regression cases before interpreting real-head diagnostics |
| A naive dense baseline could give SEC credit for generic fusion or different numerical semantics | D0/D1/D2 separate practical native speed from same-target certified comparison; reusable optimizations must be shared | A weak local fused port cannot justify a claim against the published NVIDIA implementation |
| Tokens, RNG seeds and timing repetitions could inflate apparent sample size | Section 5 defines document-level populations, fixed-session document-vector resampling, paired blocks and fixed confirmation | Small document populations, rare tails or large session contrasts can leave an inconclusive result |
| Q6 metadata, tied weights and actual KV head dimension could invalidate memory/traffic estimates | Section 6 counts emitted FP32 scale/bias/radius metadata, one shared tied head, padding and Qwen's configured 128-dimensional KV heads | Actual allocations, scratch and paging still require local measurement |
| Block aggregation or ordinary full-row L2 might be mislabeled as a stronger new mechanism | Section 10 proves the limited block statement, requires coalesced-row control, and identifies the redundant L2 bound | Cache behavior and overlapping physical fetches are outside the oracle theorem |
| Centered bounds could be mixed with unshifted exact logits | Section 10 requires a consistent shifted target or explicit common-term restoration | Every activated refinement needs a new numerical audit and matched ablation |

### 11.2 Blocking unknowns and scenario-specific responses

| Scenario | Required response |
|---|---|
| Same mechanism already exists under another name | Relabel the contribution as replication/engineering; retain only demonstrably different implementation or empirical claims |
| Strongest published CUDA baseline cannot run on M3 | Use the strongest tuned local native and faithful algorithmic controls; report the hardware restriction, with no cross-device speedup claim |
| Less memory or disk is available than assumed | Apply Section 6's measured budget: sequential models, tiled preprocessing, streamed documents, smaller extraction chunks, and reduced concurrent buffers; preserve the same head and target |
| Safe Metal operations or required compiler controls are unavailable | Use a supported standalone Metal route if qualification succeeds; otherwise retain CPU-certified diagnostics and stop the strict GPU performance claim |
| Models/data are inaccessible or license terms prevent intended use | Preserve the failure record, use an explicitly reviewed compatible alternative only through a methodological amendment, and avoid pretending it is the original protected population |
| Synthetic success disappears on real states | Diagnose whether valid widths, state distribution, quantizer or traffic changed; do not treat synthetic draw agreement as a performance result |
| Small-model failure may not express a scale effect | Use the measured cost model and shape sweeps to state a discriminating larger-shape prediction; do not infer untested large-model success |
| Tiny gain disappears after tuning or between sessions | Report absolute effects and uncertainty; conclude no demonstrated gain or inconclusive evidence as appropriate |
| Latency improves while memory or tail latency worsens | Report the full frontier and amortization; a primary-metric win alone does not establish a superior overall tradeoff |
| A run resumes with changed code, kernel, table, data, or seed mapping | Refuse silent resume; start a new run/amendment with separate evidence, retaining the interrupted result |
| Selected rows are few but actual original-weight traffic is large | Investigate cache lines, gather materialization, inactive-load behavior and shader occupancy; do not substitute analytic query count for observed traffic |
| A rare hard sample triggers a deadline or counter exhaustion | Save an unfinished sample or report explicit failure; never resample, clamp, or silently emit an approximate token |
| The head is faster but end-to-end decoding is unchanged | Report the component result and measured headroom; no whole-system acceleration claim |

### 11.3 What this plan cannot establish

The local PoC can establish conditional-law correctness under a stated arithmetic/random-bit model, an implementation's conformance to that model, and measured behavior on specified local workloads. It cannot by itself establish universal hardware superiority, broad language-model quality improvements, useful performance at all scales, mathematical independence of a finite PRNG, or the absence of all historical prior art. It does not establish freedom to operate or patentability.

The highest implementation risk is the complete numerical chain, especially shader semantics and rare fallback. The highest performance risk is that a valid low-bit scan plus metadata, randomness, reductions and synchronization costs as much as a tuned dense head. The highest interpretation risk is giving an oracle-query theorem credit for a latency gain caused by generic fusion. These risks are explicit early gates and matched comparisons, not reasons to assume either success or failure.

## 12. Sources and search-coverage record

### 12.1 Supplied sources and provenance

**S1 — Supplied report.** *Beyond fixed precision: Exact attention representations and selective categorical sampling*, dated 30 October 2026; authorship not independently established. Attached filename `beyond_quantization_research(1).pdf`, 500,223 bytes, SHA-256 `b9b848ccc0a1e5a24b53274a2b0f7f30abe251df0a268fea2a2fa71e47573730`. Inspected the full extracted text and relevant rendered sampling pages. Used for the finite-box theorem, sampler formulas, stated exactness limits and preliminary head experiment. A report supplied with an idea is evidence to audit, not independent confirmation.

**S2 — Supplied reproducibility archive.** `beyond_quantization_reproducible(1).zip`, 796,470 bytes, SHA-256 `0f56badf29ad418aff9eef27403a58b635ecd64bb030ada572a13ba5f79692f9`. All 38 listed manifest objects verified. Inspected sampler implementations, head evaluation/provenance code, saved JSON/NPZ outputs and the report source. No pretrained checkpoint was included. Original and bundled PDFs have identical extracted text. Source S2 supports the implementation/evidence audit, not new target-hardware measurements.

If these original attachments are unavailable to the implementing agent, proceed using the mathematical specifications, fixtures, model pins and result summaries in this handoff. Mark historical bundle reproduction unavailable rather than reconstructing fictional original files. The new experiment must acquire its own approved public assets and build its own executable package.

### 12.2 Research and closest-alternative references

All web records were inspected or queried on **30 October 2026**. “Inspected” specifies the actual scope below; a linked repository does not imply execution or complete code review. References with abstract-only access support only the limited claims explicitly attributed to them.

**R01 — Daniel R. Sheldon (30 September 2013), _Discrete Adaptive Rejection Sampling_.** [Technical report][R01]. Inspected Sections 2–3 and Algorithm 2. Establishes the cheap lower squeeze and supplies a simple control. Its adaptive envelope uses log-concavity; that ordering assumption is not attributed to vocabulary rows. Mentioned C/MATLAB implementation was not inspected.

**R02 — Chris J. Maddison, Daniel Tarlow, Tom Minka (2014), _A* Sampling_.** [NeurIPS paper][R02]. Inspected Gumbel-process definitions, bounded-search construction and exactness argument. Establishes bounded perturbed optimization as prior art. Its tasks/partitions and numerical implementation are not a verified M3 head baseline.

**R03 — Stephen Mussmann, Daniel Levy, Stefano Ermon (2017), _Fast Amortized Inference and Learning in Log-linear Models with Randomly Perturbed Nearest Neighbor Search_.** [UAI/PMLR record][R03]; [inspected original PDF](https://auai.org/uai2017/proceedings/papers/75.pdf). Inspected algorithms, theorems and experimental/preprocessing qualifications. Exact lazy tail sampling requires true top-k/tail information; approximate retrieval experiments provide different guarantees. It informs the optional certified-tail control and prohibits treating ANN as exact without a certificate.

**R04 — Gilles Brassard, Luc Devroye, Claude Gravel (2019; preprint 2018), _Remote Sampling with Applications to General Entanglement Simulation_.** _Entropy_ 21(1):92, [DOI][R04]; [inspected primary preprint](https://arxiv.org/pdf/1807.06649). Inspected approximate-probability rejection/refinement and finite-bit sections. Direct antecedent for exactification. Publisher access returned HTTP 429; the accessible author preprint supplies the technical evidence.

**R05 — Samuel Haney, Damien Desfontaines, Luke Hartman, Ruchit Shrestha, Michael Hay (July 2022), _Precision-based attacks and interval refining: how to break, then fix, differential privacy on finite computers_.** [Primary workshop paper][R05]. Inspected interval inverse-CDF construction and explicit noisy-argmax/Gumbel discussion. It directly anticipates interval refinement; it does not demonstrate the proposed Metal head pipeline.

**R06 — Sinho Chewi, Patrik R. Gerber, Chen Lu, Thibaut Le Gouic, Philippe Rigollet (2022), _Rejection sampling from shape-constrained distributions in sublinear time_.** AISTATS, PMLR 151:2249–2265, [record and paper][R06]. Inspected query-model definition and rejection-envelope results. Related minimax research with a different family and preprocessing objective. Linked software was not executed.

**R07 — Tavor Z. Baharav, Ryan Kang, Colin Sullivan, Mo Tiwari, Eric Luxenberg, David Tse, Mert Pilanci (2024), _Adaptive Sampling for Efficient Softmax Approximation_.** [NeurIPS paper][R07]. Inspected task, experiments, tuning discussion and wall-clock limitations. Supports an approximate context/control and the warning that fewer sampled entries need not beat optimized dense computation. No laptop reproduction was performed.

**R08 — Tomas Ruiz, Zhen Qin, Yifan Zhang, Xuyang Shen, Yiran Zhong, Mengdi Wang (2026), _FlashSampling: Fast and Memory-Efficient Exact Sampling_, v3, 25 September 2026.** [Versioned paper][R08]; [official code](https://github.com/FlashSampling/FlashSampling). Inspected tiled algorithm, numerical appendices and repository/license. Establishes the dense fusion control; its CUDA/Triton NVIDIA measurements are contextual, not M3 evidence. Prefer v3 over conflicting older headline maxima.

**R09 — Samuel Mausberg (8 October 2026), _Exact Sampling from Neural Networks: Depth, Attention, and Decoding_.** [Author landing page][R09]; [PDF](https://sammausberg.com/research/papers/exact_sampling_networks.pdf); [DOI](https://doi.org/10.5281/zenodo.23249881). Inspected selected Sections 2 and 9–11, Lemmas B.14–B.16, and Theorem B.27. These cover interval exactification and lower-mass/residual ideas. The 187-page document and linked formalization were not fully independently verified. Structural assumptions and unproved practical large-rank usefulness limit transfer.

**R10 — knapcio (runtime documented September 2026), _GLM-5.3-Flash-4x-DGX-Spark-TP4_.** [Repository][R10]; [runtime description](https://github.com/knapcio/GLM-5.3-Flash-4x-DGX-Spark-TP4/blob/main/docs/runtime.md). Inspected README/runtime claims about the eight-bit certified greedy head. Fetching `overlay/glm_cert_head.py` and `overlay/cert_math.py` failed. This is evidence of a close public description, not verified implementation correctness or a transferable speed result. Recover and inspect those files before a first-implementation publication claim.

**R11 — Jeffrey G. Bernstein, David Wingate, John Redford (published 19 January 2017), _Sampling variables from probabilistic models_, US20170017892A1.** [Patent publication][R11]. Inspected bibliographic and streaming-Gumbel passages. Limits broad streaming/fusion originality. No patent-family search, legal interpretation or status opinion is supplied.

**R12 — Samuel George Davis / Myrtle Software Ltd (published 16 November 2023; listed priority 25 September 2020), _Efficient circuit for sampling_, US20230368774A1.** [Patent publication][R12]. Inspected quantile lookup, random index and argmax description. Finite approximate lookup is an engineering antecedent; it is not the same as certified lazy interval continuation. No legal conclusions follow.

**R13 — Asim Kadav, Christian Flores, Chirag Arora, Varun Kotte, Hongbo Zheng, Lan Yan, Priya Shanmugasundaram, Tracy Holloway King (25 September 2026), _Softmax Reparameterization for Output-Head Quantization_.** [Primary arXiv identifier][R13]. Only the primary-host search-indexed abstract was accessible; full abstract page/PDF/HTML/source retrieval failed. The limited evidence supports the common-row-shift overlap. No detailed method or numerical performance claim in this plan depends on uninspected content. Revisit before proposing centering novelty.

**R14 — Joao V. Cavalcanti, Ashia C. Wilson (12 August 2026), _SoftWater: Class-Aware Rate Allocation for Softmax Quantization_.** [Primary abstract][R14]. Inspected abstract only. It identifies output-softmax-aware approximate rate allocation and calibration dependence, informing the broader compression distinction. Detailed experimental and implementation claims were not independently examined or used as local evidence.

**R15 — Cheng Zhang, Jianyi Cheng, George Anthony Constantinides, Yiren Zhao (2024), _LQER: Low-Rank Quantization Error Reconstruction for LLMs_.** ICML, PMLR 235:58763–58779, [record][R15]; [paper](https://arxiv.org/html/2402.02446v2); [official code](https://github.com/ChengZhang-98/lqer). Inspected low-rank residual/activation-scaling construction and repository description. The code describes software-emulated quantization and is not a proven M3 speed baseline. It establishes the non-novel low-rank correction ingredient.

### 12.3 Platform, numerical, asset, and implementation references

**R16 — Apple (4 June 2026 revision), _Metal Shading Language Specification_.** [Official specification][R16]. Downloaded and inspected Sections 1.6.3 and 8.1–8.6, including safe/fast modes, rounding, FTZ and basic-operation accuracy. This defines assumptions to qualify locally; it does not prove what an uninspected generated shader executes. The scalar radius in Section 4.9 is this plan's conditional deduction, not a theorem claimed by Apple.

**R17 — MLX contributors, _mlx.core.quantize_ documentation (accessed 30 October 2026).** [Official API][R17]. Inspected affine groups, supported bit widths and return metadata. Used to select actual native packs; certificate construction is independent of calibration accuracy. Pin code because moving documentation can change.

**R18 — MLX contributors, _mlx.core.quantized_matmul_ and related quantized operations (accessed 30 October 2026).** [Official API][R18]; [gather variant](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.gather_qmm.html). Inspected supported arguments and gather semantics. Supports native performance baselines, not a universal numerical certificate or arbitrary-row sparse GEMV capability.

**R19 — MLX contributors, Metal quantization kernels at commit `36969c094354086e15d2a327a9c50feb20b9cf11`.** [Pinned kernel source][R19]; [dispatch source](https://github.com/ml-explore/mlx/blob/36969c094354086e15d2a327a9c50feb20b9cf11/mlx/backend/metal/quantized.cpp). Downloaded/inspected Q6 packing and affine quantizer implementation; selected dispatch files also inspected. This grounds byte decoding and warns against replacing emitted metadata with a simplified quantizer. All possible dispatch paths were not formally verified.

**R20 — MLX contributors, _Custom Metal Kernels_ and _mlx.core.fast.metal_kernel_ (accessed 30 October 2026).** [Guide][R20]; [API signature and compile options](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.metal_kernel.html). Inspected custom-source inputs/outputs, contiguity behavior and safe math option. The API is an implementation route; on-device compile/support remains WP02's gate.

**R21 — MLX contributors, _Precision_ (accessed 30 October 2026).** [Official documentation][R21]. Inspected float32 matrix multiplication precision controls and limitations. Motivates explicit `MLX_ENABLE_TF32=0` plus source qualification; a dtype label or environment variable alone does not prove the custom target.

**R22 — MLX contributors, _Installation_ (accessed 30 October 2026).** [Official instructions][R22]. Inspected Apple-silicon Python/macOS minimums and source-build dependencies. Used for environment discovery and compatibility gates, not as evidence that the user's machine has those prerequisites.

**R23 — MLX contributors (2026), _MLX v0.32.4 release_.** [Release][R23]. Inspected release metadata and pinned commit `36969c094354086e15d2a327a9c50feb20b9cf11`. Candidate version for reproducibility; compatibility with the target OS and paired MLX-LM version is untested here.

**R24 — MLX-LM contributors (2026), _MLX-LM v0.32.2 release_.** [Release][R24]. Inspected release metadata and pinned commit `93e6d4a6447b3080861985947f0820e76fcb957d`. Use only after local import/model smoke checks; resolve supported dependencies into a lock rather than assuming an arbitrary version combination works.

**R25 — Qwen team, _Qwen3-0.6B-Base configuration_.** [Pinned config][R25]. Inspected dimensions, layers, vocabulary, tied embeddings and KV head dimension. Supports the memory calculations and adapter contract. Those dimensions are model-specific, not guesses from the model name.

**R26 — Qwen team, _Qwen3-0.6B-Base model card, repository metadata and license_.** [Model card][R26]; [repository metadata](https://huggingface.co/api/models/Qwen/Qwen3-0.6B-Base); [license](https://huggingface.co/Qwen/Qwen3-0.6B-Base/raw/da87bfb608c14b7cf20ba1ce41287e8de496c0cd/LICENSE). Inspected card, BF16 parameter metadata, public access and Apache-2.0 terms. Weights were not downloaded or run in this research. Preserve the frozen revision/file hashes during WP03.

**R27 — HuggingFaceTB, _SmolLM2-135M model card and configuration_.** [Official card][R27]; [configuration](https://huggingface.co/HuggingFaceTB/SmolLM2-135M/raw/main/config.json). Inspected card/config and supplied provenance. Supports the continuity workload, tied-head dimensions and Apache-2.0 availability. Local target-device reproduction remains proposed.

**R28 — MLX-LM contributors, _Qwen3 model implementation_ at commit `93e6d4a6447b3080861985947f0820e76fcb957d`.** [Source][R28]. Inspected body/head boundary and tied embedding projection. The coding agent must verify adapter output against the stock model at this pin, including cache and final normalization.

**R29 — MLX-LM contributors, _Sampling utilities_ at the same pinned commit.** [Source][R29]. Inspected compiled native categorical sampling path. Prevents a weak unfused CPU-only practical baseline. It does not supply the exact-real dyadic-head semantics of D1/D2.

**R30 — MLX contributors, _mlx.core.synchronize_ (accessed 30 October 2026).** [API][R30]. Inspected stream-specific synchronization. Together with explicit evaluation this supports the timing boundary; every stream actually used must be completed.

**R31 — MLX contributors, memory/device APIs (accessed 30 October 2026).** [Peak memory][R31]; [active memory](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.get_active_memory.html); [cache memory](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.get_cache_memory.html); [memory limit](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.set_memory_limit.html); [explicit device information](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.device_info.html). Inspected counter/limit semantics and device information. Combine with process/OS observations; one allocator counter is not total unified-memory use or a hard cap.

**R32 — MPFR team, _GNU MPFR manual_, inspected version 4.2.2.** [Official manual][R32]. Inspected directed rounding, conversions and elementary functions. Supplies high-precision interval primitives; LGPL requirements apply to redistribution. MPFR point evaluation at a large precision is insufficient without direction and interval composition.

**R33 — gmpy2 contributors, _Contexts_ documentation (accessed 30 October 2026).** [Official documentation][R33]. Inspected MPFR context/rounding controls. The rendered documentation includes development-version material; select and lock an actually available compatible release locally, with operation-level rounding tests.

**R34 — Salesforce / WikiText authors, _WikiText dataset card_ (accessed 30 October 2026).** [Card][R34]. Inspected `wikitext-2-raw-v1` splits, format and CC-BY-SA-3.0/GFDL designation. Supports primary text provenance. Article reconstruction and duplicate checks are explicitly part of this project rather than assumed from line counts.

**R35 — OpenAI (2021; repository accessed 30 October 2026), _HumanEval_.** [Official repository][R35]; [license](https://github.com/openai/human-eval/blob/master/LICENSE). Inspected task format and MIT license. Use prompt text only for a code-domain head workload. No benchmark score or execution of generated code is needed.

**R36 — OpenAI (2021; repository accessed 30 October 2026), _Grade School Math / GSM8K_.** [Official repository][R36]; [MIT license](https://github.com/openai/grade-school-math/blob/master/LICENSE). Inspected split/text format and license; repository is archived. Use question text only and pin data. It is a shift probe, not a new mathematical-reasoning evaluation.

**R37 — llama.cpp contributors, _llama-bench_ documentation (accessed 30 October 2026).** [Official README][R37]. Inspected benchmark scope: generation timings omit sampling/tokenization. Therefore it can contextualize engine performance but cannot replace the required complete head-plus-sampler boundary.

**R38 — Jonathan Richard Shewchuk (1997), _Adaptive Precision Floating-Point Arithmetic and Fast Robust Geometric Predicates_.** [Author page and paper][R38]. Inspected adaptive-filtering rationale and floating-point caveats. General numerical background for cheap filters followed by exact work; it is not evidence for a new sampling method or for Metal-specific arithmetic.

**R39 — Szymon Kobus, Deniz Gündüz (2025), _Speculative Sampling via Exponential Races_.** [Findings of ACL record][R39]. Inspected abstract/framework. Establishes adjacent exponential-race decoding context. Draft/target pass reduction is a different computation boundary from one fixed output head; no cross-paper speed comparison is made.

**R40 — Şuayp Talha Kocabay, Talha Rüzgar Akkuş, Kamer Ali Yuksel (2026), _ARCHead: Activation-Metric Residual Correction for Large Language Model Output Heads_.** [Official repository][R40]; [paper identifier](https://arxiv.org/abs/2608.02703). Inspected README, citation and MIT-license declaration; full paper retrieval failed. Supports the existence of head-specific low-bit/low-rank correction and its optional CUDA path. No reported benchmark maximum is used as independently verified evidence.

**R41 — D. E. Shaw Research / John K. Salmon, Mark A. Moraes, Ron O. Dror, David E. Shaw (2011; maintained repository), _Random123 / Parallel Random Numbers: As Easy as 1, 2, 3_.** [Official repository][R41]. Inspected counter/key model and Philox implementation/test availability. Reuse integer operations and known-answer tests with license notices. The practical generator does not prove infinite mathematical independence.

### 12.4 Search coverage, reconciliation, and stopping point

Research used two search systems, primary-paper/repository/documentation retrieval, backward reference tracing from the supplied report, and forward searches for 2025–2026 implementations. Search terms deliberately extended beyond “Selective Exact Computation.” The following families capture the material searches rather than claiming an exhaustive catalog of every query:

| Search family | Representative queries and tracing | Material result |
|---|---|---|
| Classical exact sampling and oracle complexity | `discrete adaptive rejection sampling squeeze`, `exact categorical sampling lower upper bounds`, `minimax query complexity rejection sampling`, `sampling approximate probabilities exact refinement` | Established squeezing/precision ancestors; clarified the distinct finite-box oracle objective |
| Random perturbation and lazy search | `A star sampling`, `lazy Gumbel exact log linear top k MIPS`, `Gumbel interval refining`, `Gumbel random bits exact sampling` | A*/Mussmann/Haney overlap; true tail certificates and finite-bit assumptions matter |
| Current heads and dense competitors | `certified low bit LM head BF16 fallback`, `FlashSampling exact fused head sampling`, `output head quantization 2026`, `selective exact logits sampling` | Close greedy-screen implementation; a stronger fused dense control; revised novelty boundary |
| Selected numerical refinement | `Gumbel lookup table sampling`, `inverse CDF interval random variate`, `Gumbel arbitrary precision sampling`, Metal rounding/FTZ/precise function documentation | Lookup antecedent; explicit interval design; GPU logs cannot simply be assumed accurate |
| Certificate refinements | `Softmax Reparameterization for Output-Head Quantization`, `2609.31291 code`, `LQER low rank quantization error reconstruction`, `ARCHead`, `SVDQuant low rank branch` | Centering/low-rank ingredients already known; only inspected sources promoted to detailed claims |
| Local feasibility | MLX Q6 packing, affine metadata, `gather_qmm`, custom Metal compile options, FP32 precision, synchronization, memory, pinned model configs and cards | Concrete decoder/API route; no FP64 GPU assumption; corrected KV and metadata costs |
| Data and evaluation | Original WikiText/HumanEval/GSM8K cards, source and licenses; llama-bench boundaries | Public local workloads, document-level independence, complete sampler timing |
| Patents and deployed descriptions | Streaming Gumbel and quantile-LUT patent publications; public certified-head runtime description | Broad mechanism overlap without accepting legal or unverified deployment claims |

Apparent conflicts were resolved by comparing assumptions and measurement boundaries. Exact top-k/tail sampling and approximate MIPS are different guarantees. AdaptiveSoftmax's entry savings are not GPU wall time. FlashSampling's earlier and v3 maxima refer to different reported versions. Exact-real random choices differ from exact reduction over finite scores. Supplied six-bit results use a symmetric logical format and heuristic FP64 numerics, while the proposed native affine format includes more metadata and a certified numerical contract. None of these pairs should be mixed in a single unqualified performance claim.

After the focused dyadic-Gumbel, centering, low-rank and block-cost refinement searches, additional inspected results supplied overlaps or implementation qualifications without changing the selected core question. Research therefore stopped at a stable implementable design and explicit testable unknowns, not at a predetermined paper count. Relevant blind spots remain: full access to R13 and R40's paper, certificate source for R10, unindexed/private implementations, some historical probability literature, a complete patent-family search, and actual local compiler/device behavior. Recovering inaccessible sources is necessary before strong priority claims; none is needed to assert that broad selective exactification is already known.

The reference register, source hashes, explicit equations and procedural gates make this a standalone handoff. Every reported local performance result remains to be produced by the proposed experiment.

[R01]: https://people.cs.umass.edu/~sheldon/papers/discrete_ars.pdf
[R02]: https://papers.nips.cc/paper_files/paper/2014/file/937debc749f041eb5700df7211ac795c-Paper.pdf
[R03]: https://proceedings.mlr.press/r15/mussmann17a.html
[R04]: https://doi.org/10.3390/e21010092
[R05]: https://tpdp.journalprivacyconfidentiality.org/2022/papers/HaneyDHSH22.pdf
[R06]: https://proceedings.mlr.press/v151/chewi22a.html
[R07]: https://proceedings.neurips.cc/paper_files/paper/2024/file/d52dbd66219dc4e432e0bd4f9c25c4c3-Paper-Conference.pdf
[R08]: https://arxiv.org/html/2603.15854v3
[R09]: https://sammausberg.com/research/papers/exact-sampling-networks
[R10]: https://github.com/knapcio/GLM-5.3-Flash-4x-DGX-Spark-TP4
[R11]: https://patents.google.com/patent/US20170017892A1/en
[R12]: https://patents.google.com/patent/US20230368774A1/en
[R13]: https://arxiv.org/abs/2609.31291
[R14]: https://arxiv.org/abs/2608.12026
[R15]: https://proceedings.mlr.press/v235/zhang24j.html
[R16]: https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf
[R17]: https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.quantize.html
[R18]: https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.quantized_matmul.html
[R19]: https://raw.githubusercontent.com/ml-explore/mlx/36969c094354086e15d2a327a9c50feb20b9cf11/mlx/backend/metal/kernels/quantized.h
[R20]: https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html
[R21]: https://ml-explore.github.io/mlx/build/html/usage/precision.html
[R22]: https://ml-explore.github.io/mlx/build/html/install.html
[R23]: https://github.com/ml-explore/mlx/releases/tag/v0.32.4
[R24]: https://github.com/ml-explore/mlx-lm/releases/tag/v0.32.2
[R25]: https://huggingface.co/Qwen/Qwen3-0.6B-Base/raw/da87bfb608c14b7cf20ba1ce41287e8de496c0cd/config.json
[R26]: https://huggingface.co/Qwen/Qwen3-0.6B-Base/raw/da87bfb608c14b7cf20ba1ce41287e8de496c0cd/README.md
[R27]: https://huggingface.co/HuggingFaceTB/SmolLM2-135M/raw/main/README.md
[R28]: https://github.com/ml-explore/mlx-lm/blob/93e6d4a6447b3080861985947f0820e76fcb957d/mlx_lm/models/qwen3.py
[R29]: https://github.com/ml-explore/mlx-lm/blob/93e6d4a6447b3080861985947f0820e76fcb957d/mlx_lm/sample_utils.py
[R30]: https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.synchronize.html
[R31]: https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.get_peak_memory.html
[R32]: https://www.mpfr.org/mpfr-current/mpfr.html
[R33]: https://gmpy2.readthedocs.io/en/latest/contexts.html
[R34]: https://huggingface.co/datasets/Salesforce/wikitext/blob/main/README.md
[R35]: https://github.com/openai/human-eval
[R36]: https://github.com/openai/grade-school-math
[R37]: https://github.com/ggml-org/llama.cpp/blob/master/tools/llama-bench/README.md
[R38]: https://www.cs.cmu.edu/~quake/robust.html
[R39]: https://aclanthology.org/2025.findings-acl.936/
[R40]: https://github.com/suayptalha/archead
[R41]: https://github.com/DEShawResearch/random123
