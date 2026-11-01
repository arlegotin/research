# Counterexample-Anchored Variational Repair
## Research assessment and executable M3 Max proof-of-concept plan

**Research date:** 13 October 2026.  
**Status:** Research and experimental design, not a demonstrated improvement.  
**Target:** One MacBook Pro with an M3 Max; actual memory, operating system, and storage must be discovered. No remote GPU, paid inference API, or second machine is required by the core program.  
**Primary direction:** Counterexample-conditioned minimum-intervention program repair, with a decisive test of whether a variational proposal path improves on well-tuned endpoint optimization.  
**Evidence labels:** **Verified** means inspected primary-source evidence; **measured here** means the CPU checks described below; **deduction** means an argument or calculation in this document; **proposed** means work not yet executed; **unknown** means an unresolved prerequisite or scientific question.

### Navigation

[Assessment](#1-executive-assessment-and-recommended-direction) · [Problem and hypotheses](#2-problem-statement-scope-assumptions-and-hypotheses) · [Prior art](#3-research-findings-prior-art-and-novelty-assessment) · [Implementation](#4-mechanism-and-implementation-specification)

[Evaluation](#5-baselines-data-metrics-and-confirmatory-protocol) · [M3 feasibility](#6-m3-max-feasibility-and-hardware-adaptive-configuration) · [Work packages](#7-risk-ordered-work-packages-and-decision-gates) · [Attribution](#8-attribution-robustness-and-failure-analysis)

[Reproducibility](#9-reproducibility-experiment-records-and-reporting) · [Outcome decisions](#10-outcome-interpretation-and-evidence-triggered-continuation) · [Audits](#11-audit-record-unresolved-risks-and-limitations) · [References](#12-references-and-search-coverage-record) · [Executed checks](#appendix-a-mathematical-checks-actually-executed)

## 1. Executive assessment and recommended direction

### 1.1 Verdict

Pursue a small, complete repair experiment, not a new foundation-model architecture. The practical opportunity is to learn which bounded activation interventions correct a particular, externally observed failure, and to spend fewer complete decoding attempts finding a good repair. The variational path is the component under investigation, not an assumed source of intelligence.

The strongest technically credible objection is more general than the proposal's quadratic limiting case. **When only the last intervention is deployed, every path objective induces an endpoint objective by minimizing over its interior states.** This remains true with nonlinear potentials and state-dependent metrics. Such a path can change finite-budget optimization behavior or encode a useful regularizer; it does not, by itself, add a new family of model outputs or constitute an executed internal reasoning trajectory. Section 4 gives the precise qualification and the required controls.

Select **CAVR-P**, where P means *proposal path*. The model is frozen. A learned counterexample-conditioned potential proposes a short path in a bounded, low-dimensional intervention space. Only the endpoint is applied during repair generation. An executable verifier, not the learned potential, controls acceptance. Implement the practical, locally approximated behavior-cost variant **CAVR-P-S** first, and the exact anchor-KL diagnostic **CAVR-P-E** on a calibrated subset. Do not describe results from S as validation of an unapproximated full-output KL objective.

The principal scientific claim to test is:

> On previously unseen program tasks, under the same information, frozen model, intervention basis, counterfactual training data, and measured online budget, does path-based proposal search yield higher independently audited repair success than the strongest nonvariational endpoint controller, without hiding regressions or offline costs?

A secondary claim asks whether **any** learned latent controller improves the practical accuracy–latency tradeoff over strong textual repair and verifier-selected sampling. These claims can have different answers. A useful endpoint controller with a useless path is a worthwhile outcome, but not confirmation of stationary-action neural computation.

### 1.2 Material corrections and explicit refinements

The supplied proposal is the starting hypothesis, not evidence of a successful system. This handoff makes the following changes explicitly.

| Issue in the starting formulation | Resolution in this plan |
|---|---|
| TTC's headline “27.8%” could be read as a relative gain or a frozen-backbone result. | The inspected MATH-500 table reports 25.0% to 52.8%: **27.8 percentage points**, or 111.2% relative to that starting accuracy. Its full-fine-tuning row is 46.8%; the difference from that row is 6.0 points. The experiment trains added layers jointly with the backbone. It is not a laptop frozen-adapter comparison. [R01] |
| An optimized path is described as the model correcting itself through a trajectory. | In CAVR-P, interior states are proposals evaluated by a surrogate, not executed model states. The endpoint-elimination result restricts the architectural interpretation. |
| A failed test is called a boundary condition. | It is conditioning information for the potential unless an explicit endpoint constraint is imposed. Here only the initial intervention and trust-region constraints are boundary/feasibility conditions. |
| Changing the number of path nodes also changes the strength of the unnormalized KL sum. | Fix total horizon to one and multiply both running costs by the time step. Retain the original scaling only as a labeled reproduction diagnostic. |
| “Regenerate the affected part” leaves error localization and cache validity unspecified. | The primary experiment regenerates the **whole function** from a common repair prompt. Suffix-only regeneration is a later optimization that all methods must receive. |
| A trainable projection can change after counterfactual labels were collected. | Freeze the intervention basis before collecting labels for a training round. A new basis requires new counterfactual labels or an exactly verified coordinate transformation. |
| Exact behavior KL is assumed cheap. | Distinguish exact finite-anchor KL from its learned local quadratic approximation. Charge actual model calls. Neither is a certificate of semantic preservation. |
| A target of +3–5 points and a 1.5× latency ceiling could become an arbitrary success rule. | Use paired effect estimates, uncertainty, and an accuracy–cost frontier. Select sensitivity and practical thresholds before confirmation from pilot information and an explicit utility scenario. |

The plan preserves the central intent: **learn from real counterfactual repairs, minimize intervention, and use independent verification**. It does not silently substitute an ordinary prompting benchmark for a latent intervention experiment.

### 1.3 What is already established, and what is not

Adaptive latent verification and steering, feedback control of activations, gradient-based latent/text refinement, energy objectives, and trajectory optimization all have close prior art. Their individual use cannot support a broad novelty claim. The narrow possible contribution is an empirically justified combination of counterexample-conditioned intervention learning, behavior-aware repair cost, and a demonstrably advantageous finite-compute proposal policy. Sections 3 and 8 identify the comparisons that could invalidate that claim.

**Measured here:** float64 CPU tests reproduce quadratic endpoint elimination to approximately machine precision; validate the state-dependent discrete gradient against automatic differentiation; and demonstrate why a constrained optimum need not have zero unconstrained gradient. These are mathematical implementation checks, not LLM results. The complete reproducible script is included in Appendix A.

**Not measured here:** M3 Max throughput, Metal operator parity, activation controllability, surrogate generalization, repair success, or superiority to any baseline. The core workload is designed to be locally feasible, but those feasibility and scientific claims have explicit gates.

## 2. Problem statement, scope, assumptions, and hypotheses

### 2.1 Beneficiary and workflow

The intended initial beneficiary is a developer running a local coding assistant that generates short Python functions and has executable examples or unit tests. The existing workflow is: generate code, run tests, provide the failure to the model, regenerate, and retain a candidate that passes available checks. Best-of-N generation and iterative textual debugging are strong alternatives, not straw men. Self-debugging is established prior art; a counterexample/fault-localization repair system also already exists. [R08, R09]

The suspected bottleneck is **expensive and inconsistent correction after a recoverable error**, not a universal shortage of model parameters. Whether this bottleneck exists on the selected tasks and model is unknown. Measure the fraction of tasks initially failing, recoverable by ordinary retries, and recoverable by bounded latent perturbations before investing in a learned action.

The primary task is single-function Python repair with finite visible tests and a separate protected test suite. This is not repository-level maintenance, production software assurance, or proof of correctness on all Python inputs. A secondary finite-domain task provides a complete verifier, but cannot establish usefulness on real code.

### 2.2 Formal objects and information boundaries

For task `i`, define:

- `x_i`: problem specification, function signature, and permitted initial examples.
- `y_i^0`: initial code produced without latent intervention.
- `T_i^vis`: visible tests available to every repair method.
- `c_i`: structured visible-test feedback, including failure category, failing input, expected and observed values when available, and the previously passing-test mask.
- `q_i = format(x_i, y_i^0, T_i^vis, c_i)`: the common repair prompt.
- `T_i^audit`: protected tests unavailable to the potential, generator, acceptance rule, tuning loop, and stopping policy.
- `p_theta(y | q_i, z)`: the frozen model with the precisely specified intervention of Section 4.
- `z in R^d`: an intervention coordinate, not a token embedding sequence or the complete model state.
- `V_phi(z; e_i)`: learned prediction of visible verification failure, conditioned on a frozen feature vector `e_i` extracted from `q_i`.
- `G_psi(z; e_i)`: positive-definite intervention metric, optionally fitted to local behavior change.
- `D_i(z)`: precisely defined finite-anchor divergence; `Dhat_i(z)` is its approximation.

The generator can see visible tests. It cannot see audit tests, audit outcomes, reference implementations, task IDs as semantic features, or the text of any protected solution. The controller may use only information a deployed system would have at that point. Offline training outcomes on training tasks are permitted, but their cost and additional information must be disclosed.

The independent unit is a **task family**, after duplicate and template grouping. Multiple seeds, candidate interventions, tests, or timings on the same task are dependent observations.

### 2.3 Scope and non-goals

The core model stays frozen throughout collection and evaluation. Only small CPU-side predictors are trained. The default research instrument is `Qwen/Qwen2.5-Coder-1.5B-Instruct`, selected for an inspectable dense architecture and manageable repeated counterfactual inference, not because it is a contemporary performance leader. Its native configuration has 28 layers, hidden width 1536, 12 query heads, two KV heads, and tied embeddings. [R20]

Run a current compact-model textual-repair challenger as well: first `Qwen/Qwen3.5-4B`; use `Nanbeige/Nanbeige4.1-3B` as the predeclared alternative when the first lacks a validated local inference path or is dominated on development data. Both are real downloadable candidates with inspected model cards; their manufacturer benchmarks do not establish which is strongest on this workload. Select by the actual development frontier, not by reputation. [R21, R22]

Do not require full-model retraining, a large process reward model, a neural ODE framework, CUDA-specific kernels, or an open-ended software-agent benchmark for the core claim. Do not claim whole-model reasoning quality from activation movement or small stationarity residuals.

### 2.4 Central hypotheses and their discriminating observations

| Hypothesis | Why it is plausible; evidence status | Competing explanation | Evidence that weakens or falsifies it locally |
|---|---|---|---|
| **H1: controllable repair headroom.** Bounded interventions can improve unseen failed drafts. | Steering is established; repair-specific controllability is unknown. | Perturbations merely diversify sampling; a new random seed does as well. | Paired intervention sweeps show no recovery advantage over equal-cost unsteered samples, despite demonstrably recoverable tasks and sufficient precision. |
| **H2: learnable counterfactual response.** A small predictor ranks effects of `z`, not just task difficulty. | ATLAS motivates lightweight latent assessment, but uses a different supervisory setup. [R02] | Predictor recognizes easy tasks, feedback length, or leaked task identities. | Within-task ranking on held-out tasks is no better than a `z`-blind model; shuffling `z` within tasks does not hurt; optimizer-selected proposals exploit errors. |
| **H3: geometry helps preservation.** Learned local behavior cost improves the repair–damage tradeoff. | KL-aware controlled generation and feedback control are precedents. [R03, R05] | Radius restriction or diagonal preconditioning alone explains the result. | Identity/constant metrics or metric-matched direct optimization give the same frontier; learned metric fails directional KL prediction. |
| **H4: proposal-path value.** A short path improves finite-budget search. | Possible optimization bias; no claim of additional endpoint expressivity. | More surrogate evaluations, multistart, a different objective, or smoothing explains the result. | Direct, straight-path, multistart, natural-gradient, and curvature-preconditioned controls remove the gain. |
| **H5: useful end-to-end gain.** Improvements survive all online and amortized offline costs. | Avoiding one full unsuccessful decoding could repay many small CPU evaluations; this is an estimate. | Hidden prefill, KL, testing, collection, or model-selection cost creates an apparent win. | Strong textual repair dominates at every meaningful budget, or collection cost cannot amortize for a plausible workload. |
| **H6: external validity.** Improvements survive protected tests and new failure families. | Separate tests can reveal overfitting to visible examples; complete correctness remains unproven. | Test-specific patching, benchmark contamination, or simple synthetic mutations explains success. | Visible-test improvements vanish or reverse on audit tests or family-held-out evaluation. |

### 2.5 Strongest case for and against

**For:** The model may already contain the needed program, but a particular generation context fails to select it. Real failing inputs are informative. A small model of local intervention outcomes may concentrate search on recoverable regions at much lower online cost than additional full generations. A behavior-aware trust region may reduce unnecessary perturbations. These are plausible mechanisms to test, not observations.

**Against:** The repair prompt already exposes the same failure in text. The best useful intervention may vary sharply by task or decoding seed, defeating a small predictor. A path over a surrogate can find confidently wrong regions. Small hidden-state norm does not imply small semantic change. Exact KL can cost more than another repair. The available tasks may be too few to learn and then detect a small genuine effect. Finally, even perfect path optimization has no endpoint-capacity advantage in this deployment formulation.

Do not explain away every negative result as insufficient scale. Escalate only when a measured bottleneck predicts a specific benefit from more examples, another intervention basis, a stronger locally runnable model, or an actually executed control schedule.

### 2.6 Headroom and practical value calculations

Let `f` be the probability an initial output fails visible tests and `r_m` the conditional probability that method `m` returns code passing both base and audit tests on that triggered population. For the specified common-initial-draft policy, visibly passing outputs are bypassed identically, so the population identity is

`Delta_success = f * (r_CAVR - r_baseline)`.

Use consistent task weighting when estimating these quantities. A different policy that also edits nontriggered outputs requires an additional term for changes on that population; do not apply this identity to it unchanged.

A ten-point conditional repair gain affects overall success by only two points when the relevant failure fraction is 0.2. Measure both denominators; never advertise conditional improvement as overall improvement. In the specified bypass policy, outputs already correct on all base and audit checks are unchanged, so whole-output regression on that group is structurally zero. This does not eliminate damage to previously passing parts of a triggered, partially failing program.

For a common initial draft, let `t_base` be initial generation plus testing, `t_m` the conditional online repair cost, and `C_m` the one-time project-specific collection/training cost. The average per-task cost over `N` tasks is

`E[t_total,m] = t_base + f * E[t_m | triggered] + C_m / N`.

When CAVR saves `s > 0` seconds per future triggered repair, its time-only break-even volume is `(C_CAVR - C_baseline) / s`. There is no finite break-even if `s <= 0`, unless a separately stated value of additional correctness compensates for the extra cost. Report scenarios, not an invented commercial valuation.

## 3. Research findings, prior art, and novelty assessment

### 3.1 What already exists

The closest landscape is not confined to papers using the word *action*. It includes inference-time activation optimization, learned steering policies, feedback controllers, counterexample-guided program repair, representation interventions, and classical trajectory optimization. The following matrix distinguishes overlap from a demonstrated equivalent system. “Code available” means a primary source links an implementation, not that it was executed here.

| Prior art | Problem and mechanism | Relevant evidence and conditions | Compute / implementation status | Exact overlap and remaining difference |
|---|---|---|---|---|
| **PPLM** [R05] | Controlled generation by optimizing a frozen LM's internal state against an attribute objective with a KL term. | Topic/sentiment control rather than externally verified program repair. | Public PyTorch implementation; inspected perturbation/regularization code. Repeated model differentiation is materially different from a small surrogate-only solve. | Frozen-state optimization plus a learned objective and behavior regularization is established. Counterexample labels and path-policy evidence would have to add value. |
| **ATLAS** [R02] | Small latent verifier selects adaptive activation-steering interventions. | Reasoning setting; latent-verifier targets are derived from an external process-reward setup, not the proposed unit-test counterfactual labels. | Inspected paper and architecture description; a usable official implementation was not established in this search. Cheap verifier inference does not make supervisory data free. | Adaptive verifier-guided steering is not novel. A repair adaptation must outperform a discrete learned steering policy using the same data. |
| **A-LQR / local linear control** [R03] | Layer-wise local linearization and feedback gains steer hidden activations toward semantic targets. | Semantic-control tasks and model-dependent Jacobian construction, not this protected-test repair protocol. | Author-linked code exists. Offline linearization and online feedback costs must both be counted; native M3 execution is unverified. | Closed-loop/Jacobian-based correction is established. A learned repair geometry is not novel merely because it is called variational. |
| **Feedback-controller activation steering** [R04] | PID-style feedback regulates steering during model computation. | Different steering tasks and targets. | Primary paper inspected; no local reproduction performed. | Feedback, accumulated error, and momentum-like correction are not distinctive components. |
| **TTC** [R01] | Context-dependent finite-horizon control with structured state-space and symplectic computation inside an LLM. | The inspected MATH-500 comparison includes joint backbone training; Section 1 corrects the gain's units and baseline. | Public repository; specialized implementation and substantial adaptation/training, not the frozen laptop protocol. | Optimal-control layers and physics-inspired latent planning already exist. Its published accuracy is not a matched baseline for CAVR. |
| **ReFT** [R06] | Train low-rank interventions in model representations. | Parameter-efficient adaptation rather than online counterexample-conditioned path search. | Public research/software ecosystem; native MLX integration still requires explicit validation. | Low-rank representation intervention is reusable prior art, not a new architecture here. |
| **∇-Reasoner and COLD** [R07, R28] | Differentiate objectives over relaxed textual representations; iterative or stochastic energy-based refinement. | Output/token-space formulations with their own likelihood, reward, or constraint objectives. | Model passes and gradients can dominate cost; published results are not comparable without matching task and resources. | Gradient-based reasoning refinement and energy-constrained generation are established. Hidden-coordinate repair is a different parameterization, not automatically a superior method. |
| **Self-debugging; counterexample-guided repair** [R08, R09] | Use execution explanations, failing examples, and in the latter case fault localization to revise programs. | Real program-repair settings; different languages/models/protocols must not be conflated. | Reuse prompting and verification concepts. A laptop model substitution is an adaptation, not reproduction of the original scores. | Counterexamples and external acceptance already exist. The new component must beat their strong local instantiations. |
| **CHOMP** [R10] | Optimize trajectories using smoothness and obstacle costs with geometry-aware gradient updates. | Motion planning, not language generation. | Classical algorithmic ideas can be reimplemented on a tiny CPU optimization problem. | Path smoothing, covariant updates, and learned trajectory costs are historical precedents. An action label is not methodological novelty. |
| **SLAP** [R11] | Variational semantic trajectories with discrete Euler–Lagrange structure. | Video-language temporal interpolation, not the same repair workflow. | Paper inspected; full local reproduction is not required for the code-repair claim. | Semantic stationary-action paths already exist, but setting and deployed trajectory differ. |
| **Energy Transformer, least-action training, YuriiFormer, Momentum Streams** [R12–R15] | Energy/transport interpretations or optimizer-inspired Transformer computation and training. | Architectural or training studies, rather than this frozen online repair policy. | Their full training experiments are outside the core laptop program. | Energy, minimal representation motion, and optimizer-inspired residual dynamics cannot support broad novelty claims. |
| **Recurrent depth; rewarded latent trajectories** [R16, R17] | Repeated actual latent computation and training signals across loops. | Recurrence is executed by the model; the rewarded-loop work uses a different training setup. | Reuse conceptual distinctions, not published numbers as local controls. | Internal repeated computation and path supervision exist. CAVR-P's virtual proposal nodes must not be described as the same kind of executed reasoning. |

Two negative-results papers matter to interpretation. *Exact Feedback Is Not Control* studies deterministic textual constraints and finds that precise feedback does not automatically yield successful correction. *Outcome Rewards Do Not Guarantee Verifiable or Causally Important Reasoning* separates answer outcomes from properties of reasoning. Neither proves that CAVR will fail; together they rule out treating feedback availability or improved pass rate as sufficient evidence of a sound internal mechanism. [R18, R19]

### 3.2 Precise novelty position

**Established, not claimable as new:** low-rank activation edits; learned latent quality predictors; inference-time optimization; KL regularization; adaptive steering; feedback control; external-test-gated repair; trajectory smoothness; discrete variational derivatives; optimization over entire paths.

**Plausible differentiation, not established novelty:** learning a counterexample-specific response surface from both successful and failed interventions, estimating intervention damage separately, and showing that a particular finite-budget proposal policy improves verified repair relative to all equivalent simpler controls.

**Potential empirical contribution:** a reproducible positive or negative result about when path search helps over endpoint search; a validated low-cost response model for repair; or evidence that learned local behavior geometry predicts unintended changes. Each could be useful without introducing new physical mathematics.

**Potential engineering contribution:** a correct, cache-safe, low-memory implementation and an accuracy–cost improvement on Apple Silicon. That is not automatically a new learning principle.

**Not established:** absence of an equivalent publication or patent; general semantic correctness; superiority on frontier models; commercial value; or a capability advantage unavailable to arbitrary endpoint optimization.

An earlier system would defeat a broad combination claim if it already learns failure-conditioned intervention outcomes, penalizes unrelated behavior changes, and compares a deployed proposal trajectory against equivalently informed direct control. A later discovery of such a system changes the study into replication, application transfer, or a controlled extension. Keep the experiment; amend the novelty claim rather than renaming the same mechanism.

### 3.3 Reuse decisions and chosen alternative

Reuse the frozen model and tokenizer, MLX model components, standard CPU automatic differentiation, author-maintained benchmark data and judgment semantics, conventional uncertainty estimation, and established constrained optimization. Build only the intervention wrapper, counterfactual-data pipeline, small predictors, shared proposal interface, budget controller, and analysis glue specific to this question.

Do **not** begin by porting TTC's specialized training stack or by pretraining a recurrent model. Neither is a prerequisite for the local question. Do not begin by implementing every controller paper. First reproduce the mathematical objective family of the strongest direct alternative with shared assets; then add a paper-faithful controller only where its model, task, data, and local operations can genuinely be matched.

The selected direction wins over immediate architectural retraining because it exposes the decisive uncertainty cheaply and remains falsifiable. It wins over a prompting-only pivot because that pivot would never test latent repair. A causal, actually executed multi-checkpoint controller remains a separate, evidence-triggered extension, not a fallback used to immunize the original path hypothesis from negative results.

## 4. Mechanism and implementation specification

### 4.1 Frozen intervention map

Use one explicit intervention location initially. For a repair prompt `q`, let `h_ref(q)` be the **unsteered block-output residual at the last prompt token** at selected layer `ell`. Let

`a(q) = max(||h_ref(q)||_2, epsilon_scale)`. Compute the reference norm in float32 even when model activations are stored in a lower precision.

Choose a frozen matrix `B in R^(H x d)` with `B^T B = I`. At the output of block `ell`, apply

`h_t' = h_t + a(q) * B z`

to the last prompt token and every subsequently generated token. All earlier prompt positions are unchanged. `a(q)` is detached, fixed across proposals for that repair context, and is not recomputed from steered activations. Consequently, `||z||` measures the perturbation relative to the reference residual norm. It does **not** guarantee that every later token has the same relative perturbation; log those ratios.

Use a seeded QR-orthonormalized random basis for the first controllability diagnostic. A train-only contrastive/SVD basis is an explicit second basis experiment if random directions lack useful headroom. In that experiment, form contrasts from matched training-task failed and verified repair contexts, compute training-only centering and SVD, and retain an orthonormal subspace. Such contrasts are a hypothesis about useful directions, not causal evidence. Freeze the selected basis before final counterfactual collection. Never pool stale coordinate labels from another basis or layer without an exact checked mapping.

Provisional engineering starting points are `d=32`, zero-based layer `ell=18` in the 28-layer instrument, and path lengths `K=3` and `5`. They are not resource or scientific ceilings. First test late-middle layer fractions near one half, two thirds, and five sixths, and dimensions 8/16/32 if data are scarce. Increase rank only when held-out intervention response or control headroom improves enough to justify its extra label and optimization demand. Select using training/development data only.

The MLX Qwen2 implementation exposes block outputs and mutating KV caches; its block returns the attention and MLP residual result. Wrap or minimally vendor that forward path. Do not assume PyTorch forward hooks work on MLX modules. Record the source revision and patch. [R26]

**Required invariants:**

1. `z=0` reproduces the unmodified model within a dtype-justified logit tolerance and produces the same deterministic continuation, except documented near-tie cases.
2. Nonzero `z` changes exactly the specified layer and absolute token positions. Padding does not shift the mask.
3. Cached and uncached generation agree for each tested `z`; proposals never share already-mutated post-intervention caches.
4. All weights, normalization parameters, tokenizer settings, basis, feature transforms, and precision are identical across matched methods.
5. NaN/Inf, invalid cache shape, or model-hash mismatch is an infrastructure failure, not a failed semantic repair.

The first implementation re-prefills the complete repair prompt for every proposal. A later optimization may cache the common prefix ending **before the last prompt token**, because no earlier position was intervened on. Each proposal must own an independent branch of the cache, and parity tests must cover divergent proposals. Count feature extraction and cache-copy costs. Never preserve a cache merely because a prompt is “mostly the same.”

### 4.2 Counterexample features and training labels

Construct one canonical prompt format containing the problem, function signature, current failed code, visible-test descriptions, and deterministic failure feedback. Apply the same format and information limits to textual and latent repair. Escape code and feedback as data, not higher-priority instructions. Bound oversized traces by a deterministic rule that preserves failure type and actual failing inputs; record truncation.

Extract `h_ref(q)` without steering. Fit feature standardization and, when useful, PCA on **training tasks only**. Start with a small context representation, such as 64 PCA coordinates plus an explicit failure-category encoding, fraction of visible tests failing, and code/prompt lengths. Store a feature schema and transform hash. Never include a task ID, audit count, reference-code embedding, or protected outcome. A contextual linear/quadratic predictor is a required control before a larger MLP.

For each training repair episode, collect rows

`(task_family, q_hash, feature_hash, model_hash, basis_hash, z, decode_seed, candidate_code, visible_outcomes, costs)`.

Include unsteered candidates, failed interventions, successful interventions, antithetic directions, and several radii. Use paired decoding seeds across perturbations when estimating their effects; additional independent seeds estimate whether the effect survives generation randomness. Duplicate candidates still consume compute and remain in the record. Cache identical test executions only under the same code and test manifest, and give every method the same cache policy.

Initial radius probes may span approximately 0.005, 0.01, 0.03, and 0.1 in the normalized coordinates. These are probes, not assumed safe radii. Select a region where interventions measurably alter behavior, produce recoveries beyond extra unsteered sampling, and avoid pervasive syntax collapse. Derive the operational trust radius from this training/development response curve. If useful changes require larger radii, test that hypothesis explicitly rather than silently clipping away the mechanism.

Define `v_i(z)` as the fraction of visible tests failing, treating timeout, syntax error, and invalid return protocol as failure under the benchmark's rules. Define `s_i(z)` as the indicator that all visible tests pass. Train a small predictor with a failure-fraction head and an auxiliary all-pass head. A suitable starting loss is

`L = BCE(vhat, v) + w_s * BCE(shat, s) + w_r * pairwise_rank_loss`,

where ranking pairs come from the same episode and have different measured outcomes. Soft-target BCE for a failure fraction is a prediction loss, not a claim that individual tests are independent Bernoulli samples. Weight tasks equally; otherwise tasks with more counterfactuals dominate the learner. Tune auxiliary weights on grouped validation, not protected outcomes.

Start with a two-hidden-layer width-64 MLP using smooth SiLU or tanh activations and sigmoid output heads, and compare with the simpler contextual predictor. For a ranking pair with measured `v_i < v_j`, use a smooth logistic loss `softplus(-(vhat_j-vhat_i)/tau_rank)` with positive, development-selected `tau_rank`; exclude tied-outcome pairs. Use a task-bootstrap ensemble to obtain a dispersion diagnostic. The potential is

`V_phi(z; e) = mean_j vhat_j(z; e) + kappa * std_j vhat_j(z; e)`.

Compute ensemble dispersion with a stable positive floor inside the square root, and test its gradient when all members agree. The standard deviation is not a calibrated confidence interval. Calibrate selection errors on held-out tasks and test whether it actually reduces optimizer exploitation. Treat `kappa=0` as an essential control. Three ensemble members are an inexpensive starting configuration, distinct from independent repetitions of the whole learning experiment; increase only if validation uncertainty or stability warrants it.

**The learning gate is intervention-sensitive generalization:** report within-task pairwise accuracy or rank correlation, selection regret relative to collected candidates, probability calibration, and performance of optimizer-selected proposals subsequently executed. Compare with a `z`-blind predictor and within-task shuffled-`z` labels. A good pooled AUROC caused solely by recognizing easy tasks does not pass this gate.

Later repair rounds create different contexts. The initial PoC uses proposals conditioned on one original failed draft; it may restart search or sampling within that context. Iteratively replacing the draft and feedback is a separately trained policy extension. Before claiming iterative performance, collect representative later-round training contexts, without using protected tasks for adaptation.

### 4.3 Behavior cost and learnable geometry

An exact KL between distributions over all completed programs is generally not available cheaply. Define a finite, reproducible alternative. Select a fixed set `A_i` of teacher-forced positions in the original failed code after repair prompt `q_i`: the first continuation position and evenly spaced positions in a bounded prefix. Deduplicate positions and keep only valid prefix indices; empty or invalid original code still has the first continuation anchor. Sixteen anchors within the first 256 code tokens are a starting diagnostic; choose adequate coverage by development stability, not by hidden-test outcomes.

For each anchor `t`, evaluate the next-token distribution under the **same** prompt and original code prefix, with the intervention mask from Section 4.1. Define

\[
D_i(z)=\frac{1}{|A_i|}\sum_{t\in A_i}
D_{KL}\!\left(p_\theta(\cdot\mid q_i,y^0_{i,<t},z)
\;\Vert\;p_\theta(\cdot\mid q_i,y^0_{i,<t},0)\right).
\]

This is an exact KL for the finite experiment consisting of a uniform anchor index and its next token. It is **not** the full autoregressive sequence KL. It measures neither functional equivalence nor correctness on untested inputs. Compute softmax/log-softmax in float32; stream or project only anchor logits to avoid retaining a full sequence-by-vocabulary tensor.

CAVR-P-S uses the local approximation

\[
\widehat D_i(z)=\tfrac12 z^\top F_\psi(e_i,0)z,
\qquad F_\psi(e,z)=\operatorname{diag}(\operatorname{softplus}(f_\psi(e,z))).
\]

Fit this behavior model separately from repair outcomes. For a local direction `u`, compare distributions at `z` and `z+epsilon*u` using symmetric anchor KL

`S = (D_KL(p_z || p_z+epsilon*u) + D_KL(p_z+epsilon*u || p_z))/2`.

For sufficiently small displacement, `2*S/epsilon^2` estimates `u^T F(e,z) u`. Use multiple directions and step sizes; establish a stable quadratic range empirically. Fit nonnegative directional costs and evaluate them on held-out tasks and directions. A diagonal approximation can fail to capture important cross-coordinate terms; diagnose that before attributing a negative result to all behavior geometry. A constant diagonal estimate is the first inexpensive alternative, and a diagonal-plus-low-rank PSD model is an evidence-triggered refinement.

Let `fbar` be the positive training-set mean of `trace(F)/d`, frozen after fitting. Use

`G_psi(e,z) = I + eta * F_psi(e,z) / max(fbar, epsilon_metric)`.

This fixes the overall reference scale and ensures positive definiteness. `eta=0` gives the identity metric. Fit the behavior model from observed divergences, not by minimizing the final action with respect to its own scale. Otherwise the metric can collapse and manufacture “low action.” Log eigenvalue bounds and calibration error. Keep raw measured KL and normalized metric units distinct.

The **primary S policy** uses the surrogate behavior cost and a latent trust region; exact anchor KL is an after-decision diagnostic on a preselected sample and is not an uncharged online guard. Report diagnostic computation separately. An exact endpoint-KL rejection guard is a predeclared optional policy, shared by every latent comparator and charged online. Select that policy, or its absence, on development data before confirmation. Neither version may imply a semantic preservation guarantee.

CAVR-P-E replaces the approximate running divergence by exact finite-anchor KL. It requires validated model derivatives with respect to `z`, or fully charged finite differences. Run it only after the backend gradient and memory gate; retain the same anchor definition and output policy. If it is slower than another complete repair, that is a result rather than a reason to hide its cost. A successful S policy alone does not validate E or the unspecified full-output distribution in the original proposal.

### 4.4 Normalized action and constraints

Fix horizon `T=1`, `Delta=T/K`, `z_0=0`, and `||z_j||_2 <= r` for every node. Require `r>0`, `lambda>0`, `mu>=0`, `eta>=0`, `kappa>=0`, nonnegative auxiliary loss weights, and positive numerical floors. Reject nonfinite costs or scales rather than silently clipping an invalid run into apparent success. For S, let `U_i(z)=lambda*V_phi(z;e_i)+mu*Dhat_i(z)`. Optimize

\[
\mathcal A_i(z_{1:K})=
\sum_{k=0}^{K-1}\frac{(z_{k+1}-z_k)^\top G_\psi(e_i,z_k)(z_{k+1}-z_k)}{2\Delta}
+\Delta\sum_{k=1}^{K}U_i(z_k).
\]

For E, replace `Dhat` by `D`. All terms are optimization costs, not physical energies. Units and normalization of the learned costs must be recorded. A low action can reflect staying near zero while failing the task, so acceptance is independent of action value.

There is no separate terminal target or hard predicted-success constraint in this primary formulation. That preserves the structure of the starting proposal. An added terminal failure penalty is a distinct ablation, not a silent improvement. If it helps, compare direct endpoints using the same terminal objective and state that the objective changed.

With `d_j=z_{j+1}-z_j`, the unconstrained interior derivative is

\[
 r_j=\frac{G(e,z_{j-1})(z_j-z_{j-1})}{\Delta}
 -\frac{G(e,z_j)d_j}{\Delta}
 +\frac{1}{2\Delta}\,\nabla_{z_j}\left[d_j^\top G(e,z_j)d_j\right]_{d_j\;\mathrm{held\;fixed}}
 +\Delta\nabla U_i(z_j),\quad 1\le j<K.
\]

The free-endpoint derivative is

\[
 r_K=\frac{G(e,z_{K-1})(z_K-z_{K-1})}{\Delta}+\Delta\nabla U_i(z_K).
\]

Omitting the metric-derivative term changes the optimization problem. The derivative has been checked against automatic differentiation in Appendix A. At a trust-region boundary, use a projected-gradient residual

`R_eta(Z) = ||Z - Project_C(Z - eta_step * grad A(Z))|| / eta_step`,

with an explicitly logged step size, or a proper KKT residual. Report raw gradients too, but do not require them to vanish at constrained optima. A small residual indicates approximate stationarity of the fitted objective, not correctness or a globally optimal repair.

### 4.5 The endpoint-equivalence result and necessary controls

Let `Z(v)` be the feasible paths ending at `v`, and define

\[
F_{\mathrm{eff}}(v)=\inf_{Z\in Z(v)}\mathcal A_i(Z).
\]

Then, simply by partitioning the feasible paths by their endpoint,

\[
\inf_Z\mathcal A_i(Z)=\inf_v F_{\mathrm{eff}}(v).
\]

If the path space is compact and the objective continuous, minimizers exist; otherwise the equality of infima still expresses the relevant reduction. Since the decoder observes only `v`, eliminating the interior states does not remove any deployed intervention. A nonlinear metric does not evade this argument. However, evaluating or optimizing `F_eff` may itself be difficult: the identity does **not** prove that one gradient step on `V(v)` is equally effective.

For constant `G` and no running cost, the minimum kinetic action with fixed endpoint `v` is

`v^T G v / (2T)`,

attained at `z_k=(k/K)*v`. Thus merely increasing the number of nodes cannot improve that objective. For a quadratic action `A(w)=0.5*w^T Q*w - b^T*w + constant` with positive-definite interior Hessian and **inactive interior constraints**, partition `w=(u,v)` into interior and endpoint coordinates. The closed-form reduction below applies to that unconstrained/interior case; active ball constraints require constrained elimination instead. The reduced Hessian and linear term are

`Q_eff = Q_vv - Q_vu Q_uu^(-1) Q_uv`,

`b_eff = b_v - Q_vu Q_uu^(-1) b_u`.

Solve systems rather than forming explicit inverses. The float64 experiment in Appendix A agrees with full-path optimization to machine precision.

Required empirical controls are therefore stronger than a single weak terminal optimizer:

- **Endpoint-only optimization:** `K=1`, with independently development-tuned regularization, the same predictor, metric, trust radius, and training data.
- **Straight-path endpoint optimization:** optimize only `v`, but evaluate the exact restricted objective `A(v/K,2v/K,...,v)`. This uses the same running costs and number of nodes, isolating path bending from a different effective penalty.
- **Competitive search and conditioning:** multistart direct optimization, a metric-preconditioned/natural-gradient update, and an equally budgeted bank of independent endpoint proposals. Compare position and increment parameterizations of the path.
- **Limiting-case checks:** constant/quadratic action with Schur elimination, identity versus constant versus state-dependent metric, and original versus normalized KL scaling.

Equal node count is not equal compute. Charge all objective evaluations, reverse-mode passes, line-search evaluations, and wall time. If a path beats naive descent but not a properly preconditioned endpoint method, attribute the gain to optimization conditioning, not a new reasoning principle.

### 4.6 Solver and deployment loop

Use CPU float32 for learned-predictor inference/optimization, and float64 for analytical tests. Start with projected gradient plus backtracking and projected Adam; use an unconstrained or bounded parameterization with L-BFGS only after verifying its treatment of the ball. Select the best valid direct optimizer on development data with comparable tuning expenditure. Do not differentiate through a projection and assume the resulting stationary point has the same KKT interpretation without checking it.

Initialize a path at zero and with a short straight path in a projected negative-potential-gradient direction. Use controlled additional restarts only within the same budget afforded to direct search. Record the complete path, initialization, objective components, evaluated-node count, projected residual, clipping fraction, and elapsed time. Stop on the predeclared numerical tolerance or available budget, not when a protected outcome is favorable. Do not spend the entire repair deadline on a surrogate solve: reserve a measured upper-quantile generation-and-test allowance, and return the best previously feasible proposal if the remaining allowance is exhausted.

```text
repair(task_public, original_code, visible_result, policy, budget, rng):
    if visible_result.all_pass:
        return original_code, status="unchanged_visible_pass"
    q = canonical_repair_prompt(task_public, original_code, visible_result)
    context = frozen_features_and_scale(q)       # charged to this policy
    incumbent = original_code
    incumbent_result = visible_result
    while budget.can_afford_proposal_generation_and_test():
        proposal = policy.propose(context, budget, rng)
        assert proposal.model_hash == context.model_hash
        assert proposal.basis_hash == context.basis_hash
        assert proposal.endpoint is finite and within trust region
        # Optional exact KL guard only when frozen as part of this policy.
        candidate = generator.generate(q, proposal.endpoint, budget, rng)
        result = visible_verifier.run(candidate, task_public.visible_manifest, budget)
        log_attempt(proposal, candidate, result, all_costs)
        if result.is_infrastructure_error:
            mark_run_invalid_or_retry_under_predeclared_rule()
        elif preserves_previously_passing(result, incumbent_result) and \
             result.failure_count < incumbent_result.failure_count:
            incumbent, incumbent_result = candidate, result
        if incumbent_result.all_pass:
            break
        # Core policy keeps q fixed. Iterative-context policy is a separate configuration.
    return incumbent, incumbent_result
```

For a fixed-context policy, subsequent proposals still condition on the original feedback, even when a partial incumbent is retained. They must nevertheless preserve the incumbent's already passing tests to be accepted. A failure to find further progress is not grounds to relax acceptance after seeing audit results.

No previously passing visible test may become failing in an accepted update. Strictly fewer remaining failures is required for partial acceptance; all-pass has highest priority. The fallback is the best previously validated incumbent. A tie is not accepted merely because the potential, action, or code length is lower. When benchmark tests are nondeterministic or inconsistent, use the predeclared quarantine rule; never silently count an unstable pass as verification.

Bypassing already visibly correct programs creates structural zero interventions on that group. It does not demonstrate that the learned metric preserves correctness. Measure preservation on **triggered** repairs, passing-test retention, protected behavior, and controlled perturbation diagnostics.

### 4.7 Shared interfaces and artifacts

The following are proposed project interfaces, not existing package APIs. Implement them in a small package named `cavr`.

```python
@dataclass(frozen=True)
class TaskPublic:
    task_key: str                 # opaque key; never a learned feature
    family_key: str
    specification: str
    entry_point: str
    visible_manifest: str        # content-addressed local artifact
    provenance_hash: str

@dataclass(frozen=True)
class RepairContext:
    prompt: str
    prompt_hash: str
    features: Array
    reference_scale: float
    model_hash: str
    basis_hash: str
    feature_transform_hash: str
    original_feedback: Feedback

@dataclass
class Proposal:
    endpoint: Array
    path: Array | None
    model_hash: str
    basis_hash: str
    diagnostics: dict

class RepairPolicy(Protocol):
    def propose(self, context: RepairContext,
                budget: Budget, rng: RNG) -> Proposal: ...

class FrozenGenerator(Protocol):
    def generate(self, prompt: str, z: Array,
                 budget: Budget, rng: RNG) -> Generation: ...
    def anchor_logits(self, prompt: str, original_code: str,
                      z: Array, anchors: Sequence[int]) -> Array: ...

class VisibleVerifier(Protocol):
    def run(self, code: str, visible_manifest: str,
            budget: Budget) -> VerificationResult: ...
```

`Budget` owns a monotonic deadline plus counters for prefill tokens, decoded tokens, model forward/backward calls, surrogate evaluations, and verifier invocations. The audit interface is deliberately absent from `RepairPolicy`, `RepairContext`, and `VisibleVerifier`. Store audit access in a separate evaluation process invoked only after outputs and policy manifests are frozen.

`VerificationResult` must distinguish `pass`, `wrong_answer`, `syntax_error`, `exception`, `timeout`, `resource_limit`, `invalid_output`, `harness_error`, and `cancelled_by_budget`. It includes per-test pass masks, normalized failure details, elapsed seconds, and test-manifest hash. Security-sensitive raw traces remain in restricted artifacts; user-facing summaries and model prompts receive sanitized data.

## 5. Baselines, data, metrics, and confirmatory protocol

### 5.1 Baseline hierarchy and fairness

Freeze the comparison scope before protected evaluation. “Strongest” below means the strongest validated configuration on development data within the declared local budget and information regime, not an unverified claim about every method available worldwide.

| ID / role | Configuration and implementation | What it establishes; reproduction requirement |
|---|---|---|
| **B0: original output** | Common frozen model, no repair. | Initial capability and trigger rate. Shared initial drafts for same-model attribution. |
| **B1: textual repair** | Full visible failure feedback; whole-function regeneration. Compare a direct correction prompt, explanation/self-debug prompt, and training-only demonstrations. | Main simple practical baseline. Tune temperature, output allowance, and feedback format; do not force an unnecessarily long explanation. |
| **B2: budgeted sampling** | Independent unsteered samples from the same repair context; verifier selection with the same acceptance rule. Include iterative feedback as a separately declared policy. | Whether diversity and extra attempts suffice. Allow useful prefix caching and early stopping under the common deadline. |
| **B3: random latent search** | Same basis, layer, radius distribution, and generated-candidate budget, without a learned potential. | Perturbation diversity versus learned control. Also compare an equal-cost change of decoding seed at `z=0`. |
| **B4: learned finite-bank policy** | Rank a training-derived bank of allowed interventions with the same potential; include a small directly trained one-step endpoint predictor. | Whether cheap selection/amortization explains the benefit. This is an **ATLAS-style adaptation**, not a reproduction of ATLAS's PRM supervision or published scores. |
| **B5: competitive direct endpoint** | Projected Adam/gradient, multistart, and metric-preconditioned variants with the same `V`, `F`, `G`, radius, data, and online budget. | **Indispensable scientific comparator.** Choose the best valid development configuration. The primary selected comparator is the strongest policy across B3–B6, so a better finite-bank or amortized policy is not excluded. |
| **B6: straight-path endpoint** | Optimize the endpoint of the restricted full path objective, including every running-cost evaluation. | Distinguishes path bending from a different objective or stronger regularization. |
| **B7: variational controls** | Identity/constant/state-dependent metrics, `K=1`, multiple `K`, curvature preconditioning, and equal independent endpoint hypotheses. | Attribution, not a separate claim of practical competitiveness. |
| **B8: current compact-model text repair** | Native validated inference for the contemporary model selected in Section 2.3, with B1/B2-quality tuning. | A stronger system-level challenge. Its own initial draft and complete workflow cost must be included; it is not a shared-model causal comparison. |
| **B9: adaptation-cost challenger, conditional** | Retrieval of verified training repairs and, when justified by deployment claims, a small parameter-efficient code-repair adaptation on the same public training material. | Required before a broad claim that the project-specific training investment is the best local use of that investment. It is not needed to establish the narrow path-versus-endpoint result. |

For B9, retrieval never indexes protected tasks or audit solutions. A learned adapter needs a separate frozen model identity and freshly collected latent labels if later used as a CAVR backbone. Do not equate a larger adapter's training budget with a small potential simply by counting trainable parameters; report actual data and compute.

**Closest-paper comparisons:** PPLM-style objective controls are represented by B5, but changing its model and objective is not paper-faithful reproduction. A faithful ATLAS or A-LQR run is eligible only if the source implementation, labels, model, and task mapping are available and critical local operations pass validation. Record the adaptation delta when translating semantic targets into verifier repair objectives. TTC and large recurrent-training results remain nonlocal references unless a genuine compatible configuration is reproduced; their published numbers never enter the paired comparison table. A failure of a CUDA-only package to import on the Mac does not demonstrate inferiority of its algorithm.

Give all same-model methods the same counterexample, public tests, model precision, parser, cache optimizations, test cache policy, output-token ceilings, and deadline semantics. Share counterfactual training rows, features, and learned metric where the mechanism claim requires equal information. Disclose which simple methods do not naturally consume these assets, and include B4/B9 to assess their opportunity cost. Maintain a tuning ledger: configurations tried, development costs, rejection reasons, and best-so-far rule. Equalize credible tuning effort, not just the final inference budget.

### 5.2 Dataset selection and a consequential split correction

Use **MBPP+ v0.2.0** as the primary protected code-repair benchmark. The author-maintained release contains 378 tasks after exclusions from the preceding release. Use original MBPP material outside the reserved tasks for training and development. The original full MBPP allocation is 374 training, 90 validation, 500 test, and 10 prompt tasks. [R23, R24a]

**Do not assume MBPP+ is simply the original MBPP test split.** Inspected task metadata and serializer cases show IDs spanning original allocations, including demonstration and training IDs. Reserve **every MBPP+ task ID and its near-duplicate family before** constructing training data, PCA, bases, demonstrations, retrieval indexes, or tuning examples. Otherwise a superficially normal “MBPP train / MBPP+ test” experiment leaks. [R24a, R24b]

Implement this outcome-independent split procedure:

```text
protected_ids = all IDs in the pinned MBPP+ v0.2.0 artifact
original_train_ids = 601..974
original_validation_ids = 511..600
original_test_ids = 11..510
original_prompt_ids = 1..10

train_candidates = original_train_ids - protected_ids
dev_candidates = original_validation_ids - protected_ids

group all candidates/protected tasks using normalized statement hashes,
entry-point/signature metadata, and source/template similarity.
remove from train/dev every group intersecting protected_ids.
resolve any remaining train/dev group collision deterministically,
placing the whole group in dev or removing it from train.
never use an original prompt/demo task that is protected.
write the final counts, exclusions, and immutable manifests.
```

The remaining training/development counts were **not downloaded and enumerated here**; they are an explicit early metadata gate, not assumed sufficient. The original ID ranges are documented upstream; the actual downloaded artifact and hash are the source of truth if metadata disagree. Do not quietly move protected examples into training to restore a convenient count. Report this as a custom leakage-controlled training allocation, not the unmodified official split.

The protected workload uses the pinned MBPP+ specification, entry point, base inputs, and extended inputs. Use base inputs and their permitted expected outputs as visible tests. Extended inputs are reserved for the audit. Generate reference outputs in the trusted benchmark service using the pinned reference implementation and author judgment rules; never send reference code or extended inputs to the repair model. The source evaluator distinguishes `base_input`, `plus_input`, `canonical_solution`, `entry_point`, and tolerances. [R24c]

Original MBPP training tasks have their original test descriptions. Preserve their documented semantics. Training feedback and labels are based only on those permitted tests; absence of additional hidden training tests is a limitation, not a reason to borrow protected plus inputs.

Reserve **HumanEval+** as an independent secondary code-task confirmation, with its exact available release and hash frozen before access to results. It has a different task distribution and may be too small to resolve a small effect; do not pool it post hoc merely to obtain significance. An optional later LiveCodeBench slice can test a different interface or publication period, but requires its own license, task-format, and contamination analysis. A recent task timestamp alone does not prove an unknown model training cutoff excludes it. [R24a, R29]

### 5.3 Oracle quality, licensing, and completeness

The ordinary MBPP/MBPP+ claim is **passing the specified test and oracle protocol**, not universal correctness. Source inspection found important special handling: set-equivalent outputs for some tasks, a non-`None` test for certain outputs, additional accepted outputs in some cases, and an unconditional acceptance branch for `are_equivalent`. Therefore naive scalar equality is not an author-faithful replacement, and not every official check is equally discriminating. [R24d, R24e]

Keep the full pinned official score for comparability, but predeclare an **oracle-sensitive guardrail cohort** excluding active tasks whose checks are unconditional or only weakly informative: `are_equivalent`, `sum_div`'s broad zero-acceptance case, and entries in `MBPP_OUTPUT_NOT_NONE_TASKS`. Intersect the names with the actual release; do not assume all remain present. Report its exact size and the contribution of flagged tasks to every gain. Valid set/float comparison rules are not automatically “weak” and should not be removed just because they are special cases. Do not claim reliable repair if the apparent advantage is confined to permissive oracles.

The core protocol does not repair benchmark ambiguities after looking at method outcomes. A revised oracle is a separately named evaluation and requires reference validation and a methodological amendment. Retain original scores alongside it. This distinction prevents quietly changing the benchmark to favor either method.

For a complete-verifier check, add a small, explicitly finite-domain expression task: `E ::= x | y | c | abs(E) | (E+E) | (E-E) | (E*E)`, with integer constants `c` in `{-3,...,3}` and maximum expression depth three. The variables range over `{-3,...,3}`. Use a restricted AST interpreter, not unrestricted host `eval`. Construct a valid target expression and a failed draft by one operator substitution; compare outputs on all 49 input pairs. For this plumbing check, all 49 input/output constraints may be visible, making its declared acceptance rule complete on the finite domain. A repaired expression can be declared equivalent **on this domain**; no claim extends outside it. Include enumeration of the finite one-operator repair neighborhood as a strong baseline; broader expression synthesis would require its own explicit grammar/search budget. Synthetic results validate control and acceptance plumbing, not realistic code usefulness.

MBPP's dataset documentation identifies CC BY 4.0; the model cards for the selected Qwen and Nanbeige candidates identify Apache 2.0. EvalPlus's repository-level license and dataset documentation must be preserved together with underlying-data attribution and file-level notices. Some evaluator code retains MIT notices; do not strip those while vendoring. The inspected MLX and MLX-LM licenses are MIT; Podman is Apache 2.0. Preserve their notices and inspect the separate base-container image and dependency licenses as well. [R25d, R26, R27] Record weight, code, and dataset licenses separately, including any conversion artifacts. “Downloadable” is not a substitute for license review. [R20–R24, R24d]

### 5.4 Safe verifier architecture and fidelity tests

Generated Python is untrusted code. A host subprocess with a timeout, or the benchmark's `reliability_guard`, is not a security sandbox. Use a rootless container in a dedicated local Linux VM; Podman on macOS provides such a VM-based route. Do not expose the host home directory, credentials, container socket, network, or arbitrary writable paths. [R27]

Use two logically separate services:

- **Reference/judgment service:** owns test manifests, expected outputs, official special-oracle code, tolerances, and reference timings. It never imports generated code. Reference implementations execute in an isolated reference container, not in the generator process.
- **Candidate service:** receives only generated code and the input values needed for the current visible test, or for a sealed final audit. It executes the function in a restricted container and returns bounded, typed results through a data-only protocol. Expected audit outputs and reference code are never mounted there.

Implement JSON with explicit tags for tuples, sets, numeric arrays, and supported scalar types; do not deserialize untrusted pickle or execute a returned expression. For the documented nonserializable-output checks, return only the exact observable the official oracle uses and flag the task as weak. Validate maximum message size and nesting; derive routine output limits from reference behavior with documented margin. Preserve the pinned author's set, tolerance, and special-oracle semantics in the trusted judge. Candidate code cannot supply a pass flag accepted by the parent.

This extraction of comparison logic is a **security adaptation** of the author evaluator, not an assumption of automatic equivalence. Before model experiments, compare its results against the pinned original evaluator running wholly inside a disposable sandbox on reference solutions, ordinary wrong programs, exceptions, edge-case floats/sets, and mutation-generated failures across the eligible workload. Preserve evaluation order and statefulness where the original semantics require them. Differences must be explained and fixed or reported as exclusions before policy outcomes are available. Time limits need the same local reference calibration; IPC and container overhead remain part of online cost.

Do not let this adapter become a new unvalidated benchmark. If parity cannot be established, use the original evaluator inside a tightly isolated container for a limited benchmark-only diagnostic and explicitly downgrade the adversarial-verification claim. Such a run is not sufficient for the primary hardened workflow until candidate-forged status and test-file access can no longer manufacture a pass.

Harness adversarial fixtures must include infinite loops, memory exhaustion, filesystem/network attempts, fake success output, attempted result-file overwrite, introspection of expected answers, invalid serialization, and mutation of built-ins. Isolation and deterministic judgment are engineering requirements, not a claim of formally proved container security. Audit containers see only their current test inputs; audit results are released to analysis only after the experiment is sealed.

### 5.5 Primary estimand and secondary outcomes

Let `Y_i,m,s` indicate that the final selected output for task `i`, method `m`, and training/decoding replicate `s` passes **both base and extended tests** under the pinned oracle. The primary estimand is the paired difference in task-averaged final success between CAVR-P-S and the frozen strongest nonvariational endpoint policy selected from B3–B6 at a common online budget. Average replicates within task before calculating the main mean; group dependent tasks into families for uncertainty.

Report the full task population, not just successful repairs or selected hard examples. Also report conditional repair success among the common initially failing cohort, initial failure fraction, the nondegenerate-oracle guardrail, and results by prespecified failure family. Present absolute percentage-point effects; relative changes may be supplemental with their denominators.

Secondary outcomes are visible all-pass rate, protected failure rate among accepted repairs, loss of previously passing checks, fraction of cases left unchanged, syntax/timeout rates, code changes, latent norm, exact anchor divergence on the fixed diagnostic sample, candidate counts, generated/prefill tokens, and model/solver/verifier time. Syntax-edit distance and KL are proxies, not semantic correctness. For multiple-passing-test tasks, report how often a proposed repair would break a prior pass **before** the acceptance gate rejects it; otherwise the gate conceals proposer damage.

Report end-to-end latency in seconds with median and tail quantiles, deadline completion, peak memory in bytes/GiB, and disk consumption. Count all required work. FLOP estimates may be reported from a declared operation model; do not call them hardware-measured FLOPs unless a validated counter actually exists. Token counts, model passes, surrogate work, and wall time are the primary portable accounting units.

### 5.6 Budget definition and timing protocol

Calibrate a meaningful online budget from the development distribution of a well-tuned single feedback repair, including prompt processing, generation, serialization, visible tests, and acceptance. Use its warm-run 90th-percentile latency as the initial central budget `b`; then preregister that measured value. This is a workload-derived operating point, not a predetermined number of seconds or success target. Report predeclared frontier points near `b/2` and `2b` as secondary tradeoffs, and state when token limits rather than time are binding.

All policies may spend the budget intelligently: textual sampling may use multiple shorter candidates; latent methods must reserve time for generating and checking their proposed candidate. The timing boundary begins immediately after the common initial failure is available and ends when a valid incumbent is returned or the deadline expires. Add initial generation/test cost for whole-workflow reporting. Model loading and VM startup are separately reported cold-start costs; do not mix them into some methods and omit them from others.

Use a monotonic clock and explicit MLX materialization/synchronization at measured boundaries. MLX offers a synchronization API; asynchronous work otherwise makes host timing misleading. [R25b] Poll cancellation during generation at a declared granularity shared by methods, kill timed-out candidate processes, and do not count an unverified over-deadline candidate as success. Record unavoidable polling overshoot; a method systematically exceeding its limit must be rerun with a corrected controller.

Warm the model, representative prompt lengths, solver, and container before timed blocks. Keep the laptop on a consistent power profile and avoid unrelated heavy workloads. Randomize or counterbalance method order within task/replicate blocks. Measure null-method timing jitter and thermal drift before choosing repeat counts. Repeat timings sufficiently to estimate uncertainty; do not turn timing repeats into new accuracy samples. Run the same zero-intervention pipeline under two method labels to check that the harness does not manufacture a difference.

For mechanism attribution, additionally compare at equal decoded-token/candidate counts and equal total surrogate evaluation budgets, explicitly including line search. These do not replace realistic wall-clock frontiers. Offline collection, feature/basis construction, model conversion, training, tuning, and storage are reported separately and incorporated into break-even scenarios.

### 5.7 Statistical design and sensitivity

Use a fixed-sample confirmation on the frozen eligible protected corpus, not repeated significance checks. If a sequential design is later needed, preregister a valid alpha-spending or confidence-sequence procedure **before** seeing its first outcomes. Do not continue sampling seeds until a desired p-value appears.

For a single paired binary result per independent task, let `q` be the discordant-pair probability and `delta` the mean paired difference. Then

`Var(Y_C - Y_B) = q - delta^2`,

so a planning approximation for two-sided 5% significance and 80% power is

`n approximately (1.96 + 0.842)^2 * q / delta^2`.

These constants describe one conventional planning scenario, not the only admissible design. With hypothetical `q=0.2`, 378 independent tasks give an approximate detectable difference of **6.44 percentage points**. About **1,745** tasks would be needed for three points and **3,925** for two points under that approximation. The calculations were run here; they are not measured discordance, exact power, or a promise that the benchmark has 378 independent families.

Estimate discordance and within-task/seed correlation from development pilots. Simulate the planned clustered, repeated-training analysis at the actual eligible task counts. Show detectable-effect curves instead of claiming that a three-to-five-point target is automatically testable. Additional training seeds reduce algorithmic uncertainty but do not create new independent tasks. If the available corpus is insensitive to the desired practical difference, report that limit and preregister a fresh eligible workload before making a small-effect claim.

Use paired task/family bootstrap intervals for mean differences and latency tradeoffs, with an additional treatment of training-seed variation. A crossed bootstrap or hierarchical model may be used after simulation checks; with only a few training runs, show the individual run effects and avoid pretending seed variance is precisely known. Start planning with three independent whole-training runs, then set the actual count from pilot variance and resource projections before confirmation. One trained model with three decoding seeds is not three training replications.

For single-seed binary paired outcomes, report the discordant counts and an exact McNemar-style test as a transparent supplement. Do not apply that test to seed-averaged fractional outcomes as though they were binary. Bootstrap task families when duplicates remain grouped; report both the benchmark's conventional task-weighted score and the prespecified family-weighted sensitivity.

There are two principal superiority claims: path versus the selected nonvariational latent controller, and the selected latent policy versus the selected textual baseline. Control familywise error with a predeclared Holm adjustment for these two comparisons, or a clearly specified hierarchical test. The policy and comparator selections occur on development data. The frontier, geometry, subgroup, and current-model comparisons are secondary unless explicitly included in the multiplicity plan. An exploratory winner needs fresh confirmation.

### 5.8 Freeze, protection, and interpretation rules

Before protected outputs are generated, seal the hypotheses, task/family manifests, eligibility and oracle flags, model/basis/predictor hashes, prompts, decoding settings, stopping policy, budgets, baselines, tuning record, primary metric, uncertainty code, and planned comparisons. Commit and hash this protocol. Data-preparation code may inspect protected metadata needed for reservation; it must not use protected outcome information to select methods or filters.

The audit evaluator runs only on frozen final outputs. Do not feed an audit failure back into repair, choose the best checkpoint after the audit, or publish the best training seed. Return all selected outputs and keep all failed runs. A methodological amendment affecting the claim requires a fresh independent confirmation or an explicit exploratory label.

A non-significant result is not equivalence. Evidence for practical equivalence requires a predeclared smallest effect worth distinguishing and an interval contained within that region. Since the intended commercial use is unspecified, do not invent such a threshold as a universal truth; report several transparent utility scenarios and the measured frontier. A statistically positive but fragile gain, a gain isolated to weak oracles, or a gain erased by fair tuning is not a breakthrough.

## 6. M3 Max feasibility and hardware-adaptive configuration

### 6.1 Discover the actual target before selecting a workload

The following are real macOS/Python commands, unlike the proposed project commands in Section 7:

```bash
uname -m
sw_vers
sysctl -n hw.memsize
sysctl -n hw.logicalcpu
df -h .
vm_stat
python3 --version
```

Record available memory, disk space, architecture, power mode, free-space path, and OS version. Hardware inventory tools may reveal serial numbers or other identifiers; redact them from public manifests. Confirm native `arm64` Python rather than a translated x86 environment. Do not assume the machine has the maximum M3 Max memory configuration, or silently convert “36 GB” into an observed value.

A missing supported macOS/Python combination, unavailable permitted model, or insufficient safe memory is a blocking prerequisite with an explicit alternative below. Routine environment discovery is not deferred essential research: the relevant operations and selection rules are specified here, but their result depends on the user's physical laptop.

### 6.2 Software stack and dependency strategy

Use **MLX + MLX-LM for frozen inference**, **PyTorch on CPU for small predictors and the variational solver**, NumPy/SciPy for diagnostics/statistics, and a local Podman VM/container for untrusted execution. The inspected MLX installation documentation requires Apple Silicon, a native supported Python, and macOS 14 or newer; it lists Python 3.10 or newer at the research date. Verify the selected wheel's own requirements, which can change. [R25a]

Create a clean native virtual environment. Install mutually compatible stable wheels for `mlx`, `mlx-lm`, `torch`, `numpy`, `scipy`, `pyyaml`, `pytest`, and the model/dataset loading packages actually used. Avoid CUDA/vLLM extras and source builds unless a justified local requirement appears. Run `pip check`, capture exact versions, and create a lock with hashes or an equivalent reproducible environment record after the smoke test. Do not fabricate a version pin before verifying that its wheel and APIs work on the actual OS.

The candidate-evaluation environment is separate from the model environment. Install only the CPU benchmark dependencies and pinned evaluator source inside the ARM64 Linux image. Preserve file-level licenses. Do not copy a GPU-oriented installation command from a repository README into the Mac plan.

The ordinary dense Qwen2 model forward path is the primary instrument because its layer and cache operations were inspected. The current Qwen3.5 implementation has a different hybrid state mechanism; use it initially as an inference-only challenger and do not transplant the Qwen2 intervention wrapper by name matching. [R26] Nominal library support is not proof of numerical parity, gradient support, or performance.

The core S solver does not need Metal autodifferentiation through the LLM. Explicitly place the PyTorch predictors, tensors, and optimizer states on `device="cpu"`. Features and counterfactual outputs cross a small explicit boundary to CPU. Do not enable a silent GPU-to-CPU operation fallback inside an allegedly accelerated exact-KL loop. For E, test the exact forward/backward path with respect to `z` separately; if unsupported, report E as locally blocked or run a clearly labeled, fully charged finite-difference diagnostic. This does not block S's narrower claim.

### 6.3 Memory model and actual calculations

For a dense model with `P` parameters and `b_w` bytes per stored parameter, weights occupy approximately `P*b_w`, excluding temporary conversion copies, metadata, and allocator behavior. A standard grouped-query-attention cache at sequence length `S`, batch `B_batch`, `L` layers, `n_KV` KV heads, head dimension `d_h`, and `b_KV` bytes per element requires approximately

`M_KV = 2 * L * B_batch * S * n_KV * d_h * b_KV`.

Using the inspected 1.5B instrument configuration and tied embeddings yields the following **calculated**, not measured, contributors. [R20]

| Contributor | Calculation / estimate |
|---|---:|
| Architecture-derived parameter count | 1,543,714,304 parameters |
| FP16 weight storage | 2.875 GiB |
| Batch-one FP16 KV at 2,048 tokens | 56 MiB |
| Batch-one FP16 KV at 4,096 tokens | 112 MiB |
| All float32 logits for 2,048 positions and vocabulary 151,936 | 1.159 GiB |
| Float32 logits for 16 anchors only | 9.273 MiB |
| A `1536 x 32` float32 intervention basis | 0.188 MiB |
| Tiny CPU predictor and path optimizer | Typically small relative to weights; measure parameters, gradients, and optimizer states rather than assuming zero cost. |

The cache is not the dominant uncertainty for this instrument. Prefill attention, intermediate activations, conversion copies, full-vocabulary logits, backend workspaces, runtime allocations, file caching, Python objects, and the verifier VM can dominate the gap between weight size and peak resident memory. Exact model-gradient experiments retain more activations. A practical planning envelope of several additional GiB is only a provisional allowance; replace it with measured peaks before collection.

Measure MLX peak allocated memory in bytes using the available documented API and record process/OS memory pressure separately. Unified-memory counters overlap: do not simply add MLX allocation and process RSS as if they were disjoint physical pools. Use fresh-process measurements where resetting a peak counter is version-dependent. [R25c]

### 6.4 Calibration and adaptation rules

Run a calibration ladder on **training examples**, starting with batch one, the 1.5B model, a short prompt, and a short continuation. Increase to the observed development distribution's prompt and code lengths. Measure prefill tokens/second, decode tokens/second, time to first token, anchor-evaluation time, CPU solver evaluations/second, visible-test latency, total wall time, peak memory, and swap/pressure behavior. Include the longest eligible development case, not only a convenient median example.

Require stable completion without resource-induced output changes or severe swapping. Reserve headroom for the OS, CPU predictor, and VM based on measured simultaneous use; determine a safe allocation ceiling before increasing batch size or sequence length. Do not use all nominal unified memory simply because the model fits once.

Apply fallback rules in this order:

1. Remove unnecessary retained tensors and project only needed logits; serialize independent candidates; reduce batching and avoid simultaneous model copies. These changes should preserve mathematical outputs and must pass parity tests.
2. Shorten redundant prompt material using a fixed, shared formatting rule only when task semantics and required feedback remain intact. Report any changed evaluation population due to context limits.
3. Use a validated lower-storage precision or quantization **as a different frozen instrument**, shared by all methods. Recollect response labels and retrain predictors; compare a development slice against the original precision to determine whether controllability changes.
4. Select a smaller eligible model only when it retains demonstrable ordinary-repair capability and latent control headroom. A model incapable of the task cannot meaningfully falsify the repair mechanism. Conversely, a larger local model is justified when pilot evidence isolates a capability limitation and measurements show that it fits.
5. If no faithful local configuration can express and test the mechanism, stop the dependent claim and report a resource-limited result. Do not switch to a remote service or call a surrogate-only toy a complete CAVR experiment.

The model's published weight dtype and the chosen inference dtype can differ. A conversion to FP16 or quantized weights is not assumed semantically neutral. Save conversion settings, compare unsteered and steered outputs on development data, and never train the potential against one precision while deploying another without fresh validation.

### 6.5 Collection, runtime, and storage estimates

Let `N_f` be training repair contexts, `M_z` interventions per context, `R` decoding replicates, `L_in/L_out` average prompt/output tokens, `t_vis` average visible-test cost, and measured throughputs `v_prefill/v_decode`. A first forecast is

`T_collect approximately N_f*M_z*R*(L_in/v_prefill + L_out/v_decode + t_vis + t_overhead)`.

This is an accounting model, not a prediction of M3 speed. Cache reuse, batching, code duplication, and prompt-length variation need measured corrections. One million decoded tokens takes `1,000,000/v_decode` seconds before prefill and verification; use the laptop's measured `v_decode`, not a borrowed benchmark rate. Estimate behavior-metric label cost separately, including both sides of each local KL comparison.

Grow collection in batches that improve the learning-curve or uncertainty estimate. An initial small pilot might use dozens of failing training contexts and several antithetic perturbations each; this is a starting allocation, not a hard data quota. Choose later batch size from uncertainty in within-task ranking and optimized-proposal success, while respecting disjoint task validation. Stop an unproductive data expansion when additional labels no longer resolve an identified uncertainty at credible cost—not when a fixed arbitrary hour count elapses.

Predictor training is small compared with repeated LLM decoding, but verify this with measured epochs, early-stopping behavior, and CPU utilization. The main risk is insufficient independent tasks rather than insufficient minibatch throughput. Increase network capacity only if a held-out learning curve supports it.

Storage planning includes source and converted weights, temporary conversion copies, the VM image, pinned datasets, generated code, token/output logs, feature arrays, model/predictor checkpoints, and raw outcomes. Save only the frozen projected features needed for learning rather than full per-token hidden states by default. For `N_rows` rows with `p` float32 features, feature storage alone is `4*N_rows*p` bytes; code and provenance add to it. Forecast against actual free space with room for atomic checkpoint copies. Never overwrite the original downloaded model to save space without preserving a reproducible source hash.

### 6.6 Local isolation, interruption, and recovery

Create a minimal ARM64 candidate image. A starting VM allocation of roughly two CPU cores and a few GiB is an engineering probe, not a required hardware configuration. Test resource enforcement and coexistence with inference, then set allocations from observed reference-task needs. A Podman run should use no network, a read-only root filesystem, no capabilities, no-new-privileges, bounded processes/memory/CPU, a small temporary filesystem, and only per-attempt mounted inputs. Confirm the installed CLI's options with its help before constructing the wrapper. [R27]

Use outer wall-clock enforcement as well as per-function reference-calibrated limits. Keep model generation and untrusted execution sequential initially to avoid memory pressure and timing interference. Overlapping them is a later system optimization, shared where applicable and separately measured.

Write each completed attempt atomically with a content hash. Save RNG states, data cursor, model/basis/predictor hashes, and optimizer state at training checkpoints. A resumed training job must reject incompatible manifests. A half-written candidate is not a result; preserve it as an interrupted artifact. Restart a partially completed timing block rather than stitching pre-interruption and post-interruption latency into a fictional uninterrupted measurement. Account for interrupted work in total project costs while excluding invalid latency observations under a declared rule.

## 7. Risk-ordered work packages and decision gates

### 7.1 Repository and command conventions

Use a single small repository. This is a proposed layout; none of its project commands are claimed to exist already.

```text
cavr-poc/
  README.md
  pyproject.toml
  requirements.lock
  cavr/
    cli.py
    config.py                 # schema validation; immutable run configuration
    provenance.py             # hashes, manifests, atomic writes, amendments
    budget.py                 # shared deadline and work accounting
    data/{mbpp.py,splits.py,feedback.py,schemas.py}
    model/{mlx_qwen2.py,features.py,cache.py,anchors.py}
    learning/{potential.py,metric.py,collect.py,train.py}
    policies/{text.py,random.py,bank.py,endpoint.py,path.py}
    verify/{protocol.py,runner.py,judge.py,sandbox.py}
    analysis/{paired.py,power.py,frontier.py,report.py}
  tests/
    test_math.py
    test_budget.py
    test_zero_parity.py
    test_cache_branching.py
    test_split_leakage.py
    test_judge_parity.py
    test_sandbox_adversaries.py
    test_resume.py
    test_analysis_nulls.py
  configs/{instrument.yaml,baselines.yaml,pilot.yaml,confirm.yaml}
  containers/{candidate.Containerfile,reference.Containerfile}
  manifests/{sources.json,hardware.json,licenses.json,splits.json}
  protocol/{exploration.md,confirmation.md,amendments.jsonl,decisions.jsonl}
  data/{raw,public,train,dev,protected}
  artifacts/{features,bases,predictors,runs,reports}
```

Protect `data/protected` and reference assets from the generator and candidate services. A pathname convention alone is not access control; implement the service boundary and restricted mounts. Keep immutable raw artifacts separate from processed copies.

The agent must implement the following project command surface, or an equivalently documented one: `python -m cavr.cli doctor`, `math-checks`, `prepare-data`, `verify-smoke`, `backend-smoke`, `baseline-pilot`, `collect`, `fit`, `pilot`, `freeze`, `evaluate`, and `report`. The arguments and outputs belong in `--help` and the README. These names are **proposed project commands**, not fabricated third-party package functions.

### 7.2 Minimal configuration contract

The following illustrates the **project's proposed schema**, not a ready-made library configuration. `null` fields are blocking choices to be filled from named gates; the confirmation command must reject unresolved values. Starting dimensions are exploratory defaults, not validated optima.

```yaml
schema_version: 1
stage: exploratory
model:
  repository: Qwen/Qwen2.5-Coder-1.5B-Instruct
  revision: null                 # resolve/hash in WP00; never silently use latest
  dtype: float16                # qualify this instrument in WP03
  backend: mlx
intervention:
  layer_index: 18
  rank: 32
  basis_hash: null              # freeze after WP05
  mask: last_prompt_and_generated
  scale: unsteered_reference_l2
  radius: null                 # response-curve selection, WP05
potential:
  device: cpu
  checkpoint_hash: null        # WP06
metric:
  kind: identity               # learned alternative requires WP06 calibration
  checkpoint_hash: null
policy:
  kind: cavr_path_surrogate
  path_nodes: 3
  horizon: 1.0
  exact_endpoint_guard: false
  lambda: null
  mu: null
  solver: projected_gradient
  solver_config_hash: null
budget:
  seconds: null                # measured development operating point
  generation_token_limit: null
  reserve_policy_hash: null
verification:
  visible_manifest_hash: null
  require_no_lost_passes: true
  acceptance: strict_failure_reduction
  sandbox_manifest_hash: null
protocol_hash: null
```

Every baseline resolves the same applicable fields. Irrelevant fields are explicitly marked unused rather than assigned misleading values. A metric checkpoint may remain absent only when `kind: identity`; a predictor checkpoint is absent only for a policy that genuinely does not use a predictor. Schema validation must enforce these conditional requirements. Exploration may run with values supplied on the command line, but the fully resolved configuration is saved before each run.

### WP00 — Establish prerequisites and provenance

**Objective / risk:** ensure the program can be run legally, locally, and reproducibly without assuming the hardware configuration. **Entry:** this handoff and a fresh target laptop. **Dependencies:** none.

**Actions:** create the repository and environment inventory; run Section 6.1 commands; record available storage and native architecture; inspect installed container tooling; resolve permitted source/model/dataset licenses; create content-addressed source records. Check availability of small metadata/configuration files before downloading model weights. Create an initial decision log and preserve the handoff in the repository.

**Files / command:** `doctor`, `manifests/hardware.json`, `sources.json`, `licenses.json`, `protocol/decisions.jsonl`. Record a source URL and revision/hash independently of a display name. Do not store access tokens or serial numbers.

**Tests / output:** machine-readable hardware values with units, redacted environment report, license/access status for every core asset, and a disk/memory forecast. **Resources:** metadata and package inspection only; no training.

**Gate / next action:** proceed when a supported native stack and permitted core assets are plausible. Resolve a routine local dependency issue before dependent work. If a model is unavailable, choose the declared compatible alternative and record the changed instrument; if benchmark licensing or access fails, block its claim and establish a separately named workload. Do not silently substitute undocumented data.

### WP01 — Validate the mathematical problem before infrastructure growth

**Objective / risk:** catch endpoint redundancy, sign/scaling errors, invalid stationarity claims, and a misleading toy demonstration. **Entry:** CPU scientific packages available. **Dependencies:** WP00.

**Actions:** put Appendix A into `tests/test_math.py` or a runnable diagnostic; reproduce constant-metric straight paths, quadratic Schur elimination, state-dependent metric derivatives, and constrained projected-gradient behavior. Add a test that an unnormalized constant KL cost grows with `K`, while the normalized running cost does not. Test finite outputs for extreme but allowed predictor logits.

**Files / command:** `math-checks`, `tests/test_math.py`; output JSON with environment, tolerances, seeds, residuals, and pass/fail fields. **Resources:** tiny float64 CPU arrays, not a model download.

**Gate / next action:** any failed identity or gradient check blocks the solver. Fix the implementation rather than interpreting an error as a novel effect. Successful checks permit the empirical finite-budget hypothesis; they do not establish a path advantage. Record endpoint equivalence in the project's claim ledger so it cannot disappear from later reporting.

### WP02 — Reserve data and build a trustworthy evaluation harness

**Objective / risk:** prevent split leakage, permissive-oracle overclaims, unsafe code execution, and false passes. **Entry:** permitted metadata/data access and a local container route. **Dependencies:** WP00–01.

**Actions:** pin MBPP+ v0.2.0; enumerate IDs; execute Section 5.2 reservation and grouping; calculate actual train/dev/protected counts and oracle flags; create public and sealed views. Build the candidate/reference services and trusted judge. Validate canonical reference outputs and exact oracle semantics. Implement known-correct, wrong, timeout, fake-status, and hostile-code fixtures; compare the adapted judge with the original evaluator inside a disposable container.

**Files / command:** `prepare-data`, `verify-smoke`, `data/splits.py`, `verify/*`, container files, split and oracle manifests, parity report. Keep raw references out of the model process.

**Tests / output:** zero protected-family overlap with training, PCA/basis fitting, demonstrations, and retrieval; reference solutions behave according to the pinned benchmark; mutations produce expected failures; equivalent methods produce identical outcomes; adversarial status output cannot create a trusted pass. Record any non-discriminating oracle rather than pretending it is complete. **Resources:** CPU/VM; derive limits from reference workloads.

**Gate / next action:** fix unexplained judge differences and isolation failures before model collection. If the surviving train/dev tasks are too few, assess grouped sensitivity and use additional disjoint, permitted training material—not protected-task leakage. A benchmark-only unhardened diagnostic is explicitly downgraded and cannot pass the primary verifier gate.

### WP03 — Prove backend and cache correctness, then measure capacity

**Objective / risk:** verify that the intended latent operation is actually executed on the local stack. **Entry:** math and verifier tests pass. **Dependencies:** WP02.

**Actions:** instantiate a tiny random dense model using the inspected native model class and a locally validated small configuration; test block/output masking and cache branching before downloading large weights. Then load the 1.5B instrument, fix dtype, and run zero-intervention, nonzero-intervention, cached/uncached, and deterministic decoding tests. Calibrate prompt-length and output-length memory/time curves. Implement a prefix cache only after its branch-isolation tests pass.

**Files / command:** `backend-smoke`, `model/mlx_qwen2.py`, `cache.py`, `anchors.py`, `features.py`, `tests/test_zero_parity.py`, `test_cache_branching.py`; output a backend qualification report and memory trace.

**Tests / output:** masks alter only intended positions; model weights hash unchanged; `B^T B` close to identity; zero-edit logits agree within a measured dtype tolerance; cache reuse reproduces a fresh forward under each proposal; nonzero edits have a measurable effect; no silent fallback or corrupted state. **Resources:** first tiny arrays, then one model at a time; Section 6 calibration ladder.

**Gate / next action:** numerical/operator failure is an implementation blocker. If memory fails, apply the faithful resource ladder and requalify the new instrument. If the exact-KL derivative path fails, block E, not S. If no meaningful intervention occurs, inspect scale and layer placement before claiming no controllability.

### WP04 — Establish strong repair baselines and available headroom

**Objective / risk:** determine whether ordinary repair is already sufficient and whether the model can solve the tasks. **Entry:** valid generator and verifier. **Dependencies:** WP03.

**Actions:** generate common initial drafts on training/development tasks; reproduce B0/B1/B2 and parser behavior; tune a small, justified set of prompt/temperature/output strategies. Measure recovery from additional unsteered seeds and feedback. Evaluate the contemporary compact-model challenger after its own inference smoke test and capacity check. Apply generic safe cache and parser optimizations consistently.

**Files / command:** `baseline-pilot`, `policies/text.py`, baseline configuration ledger, per-task raw generations and tests, throughput/frontier report. **Resources:** measured generation and test costs; begin with a diverse training/development subset and grow until baseline variance and headroom are identifiable.

**Tests / output:** full information parity, correct trigger/fallback behavior, duplicate accounting, deadline correctness, code-extraction consistency, and measured text-repair headroom. Include the fraction of truncated generations and failures caused by missing model capability rather than a repair decision.

**Gate / next action:** if the baseline is broken or weakly configured, fix it before evaluating CAVR. If almost no failures remain, the task has inadequate headroom; move to a predeclared harder but locally expressible workload. If ordinary retries never recover known-solvable tasks and the stronger local model does, qualify that model as a new instrument before collecting latent labels. Do not label model incapacity a failed path mechanism.

### WP05 — Test causal intervention response before learning a large dataset

**Objective / risk:** distinguish useful latent controllability from random-sampling diversity. **Entry:** recoverable errors and qualified model. **Dependencies:** WP04.

**Actions:** on training tasks, sweep normalized radii and selected late-middle layers using paired seeds and antithetic random directions. Include zero edits and extra unsteered samples at the same cost. Inspect syntax collapse, repair rate, passing-test damage, and diversity. Choose the basis/layer/radius using these data; if needed, test the declared training-only contrastive basis. Freeze the chosen instrument and basis before final label collection.

**Files / command:** `collect --phase controllability` as a proposed CLI option; intervention manifest, response curves, and a task-grouped oracle-selection diagnostic. The oracle maximum over a candidate pool is an upper-bound diagnostic, not a deployable method.

**Tests / output:** positive-control perturbations change logits; paired seed effects are logged; no protected data enter basis selection; radius units are consistent; reported gains survive equal-budget extra sampling. **Resources:** forecast from WP04 rates and stop/add batches based on uncertainty about response, not a desired result.

**Gate / next action:** useful response permits WP06. No response with a broken intervention is an implementation problem. No response with demonstrably recoverable tasks, multiple justified subspaces, and adequate sensitivity is evidence against local low-rank repair—not a universal theorem. A noisy result permits an information-gain-justified batch; an absent task capability calls for a qualified stronger local instrument, not unlimited perturbations.

### WP06 — Learn and validate the counterfactual potential, then geometry

**Objective / risk:** ensure the learner predicts intervention effects and does not exploit leakage or its own uncertainty. **Entry:** frozen basis and credible control headroom. **Dependencies:** WP05.

**Actions:** collect both successful and failed interventions on disjoint training families; create task-weighted datasets; fit contextual linear/quadratic and small MLP predictors; validate within-task ranking, selection regret, calibration, and optimized-proposal execution. Add task-bootstrap uncertainty only when it improves held-out selection. Then collect local directional KL labels and fit constant/diagonal/state-dependent behavior models. Keep these labels, units, and losses separate from repair rewards.

**Files / command:** `collect`, `fit --target potential`, `fit --target metric` (proposed options); model cards for predictors, training logs, held-out selection and KL calibration reports. **Resources:** counterfactual collection dominates; choose additional data using grouped learning curves. CPU training must record actual parameter/optimizer memory and elapsed time.

**Tests / output:** task-balanced losses; zero audit exposure; stale-basis rejection; deterministic feature transforms; `z`-blind and shuffled-label controls; SPD metric and fixed normalization; finite gradients; no improvement claims from pooled task-difficulty prediction.

**Gate / next action:** a useful potential permits the complete repair pilot. If it fails despite oracle headroom, investigate feature capacity, label noise, and sample sensitivity in that order. Do not increase architecture complexity without a measured learning bottleneck. If geometry fails but the potential succeeds, carry identity/constant geometry forward and withhold the learned-geometry claim. If the optimizer consistently finds surrogate errors, reduce unsupported extrapolation or change collection; external rejection alone does not make an inefficient policy useful.

### WP07 — Run the minimum complete mechanism and decisive controls

**Objective / risk:** determine whether the path adds anything beyond an informed endpoint policy at realistic cost. **Entry:** validated response predictor; identity metric is sufficient to start. **Dependencies:** WP06.

**Actions:** implement S and B3–B7 through shared interfaces. Run paired development repairs with full acceptance/fallback, measured online budgets, equal-surrogate-work diagnostics, and the straight-path objective. Add learned geometry only after its validation. Run E on a calibration-selected subset when exact derivatives and memory pass; count every model call. Evaluate action/residual diagnostics against actual outcomes without using them as acceptance certificates.

**Files / command:** `pilot`, `policies/path.py`, `endpoint.py`, `bank.py`, `budget.py`; output per-attempt paths/costs and a comparison frontier. **Resources:** reserve generation/testing allowance before solving; measure solver overhead rather than assuming it negligible.

**Tests / output:** limiting-case equivalence, clipping/KKT behavior, fair restarts, equal information, same cache/parser optimizations, no deadline cheating, and unchanged fallback. Test numerical gradient errors before interpreting path behavior.

**Gate / next action:** if path gains survive the strongest endpoint controls and costs, prepare a confirmatory path test. If latent repair works but path does not, retain a direct-policy study and preregister a narrow falsification/precision test for the path; do not market stationary action. If all latent methods are dominated, diagnose model/predictor/compute causes and stop escalation unless a specific measured fix exists. An underpowered development comparison is not proof of equivalence.

### WP08 — Freeze a sensitive and affordable confirmation

**Objective / risk:** prevent adaptive benchmark selection and underpowered overclaims. **Entry:** complete working policies and trustworthy development measurements. **Dependencies:** WP07.

**Actions:** select and freeze the best nonvariational latent and textual comparators; finalize exact-guard policy, anchors, trust radius, path length, model/dtype, training replicates, and budget points. Simulate paired/clustered power and intervals using development noise. Forecast collection, training, evaluation, and disk costs on the actual laptop. Seal the primary metric, multiplicity treatment, oracle-sensitive cohort, exclusion rules, task families, and analysis scripts.

**Files / command:** `freeze`, `configs/confirm.yaml`, `protocol/confirmation.md`, signed/hash manifest, preconfirmation analysis-null tests. **Resources:** calculations plus any justified final development calibration; no protected method outcomes.

**Tests / output:** all referenced assets exist and match hashes; no active tuning process reads audit outputs; sample sensitivity is stated in units of independent tasks/families; no primary choice depends on protected results. The run plan explicitly names which scientific claim remains detectable locally.

**Gate / next action:** proceed when the workload is locally supportable and the claim matches sensitivity. If a small desired effect is not detectable, keep the fixed benchmark as an estimation study and define fresh independent confirmation before claiming small-effect superiority. Do not “solve” low power by counting every seed as another task or by weakening the baseline.

### WP09 — Execute the protected comparison without outcome-driven changes

**Objective / risk:** obtain unbiased paired outcomes and honest costs. **Entry:** sealed protocol and all core gates passed. **Dependencies:** WP08.

**Actions:** train the preregistered independent predictor replicas using fixed training manifests; generate common initial drafts for matched-model comparisons; run all selected policies with randomized timing blocks and frozen budgets. Persist final outputs before releasing audit results. Execute the full and oracle-sensitive audits with the trusted judge; retain timeout, rejection, and failed-run records. Run stronger-model workflows with their own initial cost and denominators.

**Files / command:** `evaluate --protocol <sealed-manifest>` as a proposed interface; immutable run shards, timing/memory traces, final-code manifests, sealed audit files. **Resources:** use the measured forecast and safe concurrency; no remote fallback.

**Tests / output:** complete task/replicate coverage or explicitly missing entries; hashes and seed identities consistent; no audit feedback reaches the generator; deadlines and cancellation reasons valid; resume events recorded. Re-execute invalid timing blocks under the preregistered rule, not because their accuracy was unfavorable.

**Gate / next action:** infrastructure corruption blocks inference and requires repair plus documented rerun scope. A legitimate negative result is retained. An interruption resumes from validated atomic state; incompatible checkpoints are rejected. No tuning or sample extension follows a favorable or unfavorable protected glimpse within the same confirmation.

### WP10 — Attribute effects and confirm transfer without post hoc relabeling

**Objective / risk:** determine why a gain occurred and whether it survives a new workload. **Entry:** completed protected comparison or a clearly labeled exploratory result. **Dependencies:** WP09 for confirmatory attribution; predeclared development diagnostics may run earlier.

**Actions:** execute the frozen ablation/robustness program in Section 8. Evaluate the independent HumanEval+ confirmation with the already selected policies and a predeclared budget mapping. Reassess subgroup and oracle sensitivity, learned-metric calibration, surrogate exploitation, and precision changes. New hypotheses generated by these results require a separate protocol and fresh data.

**Files / command:** `evaluate` with secondary protocol manifests; `analysis/paired.py`, `frontier.py`; transfer, ablation, and failure-taxonomy reports. **Resources:** forecast from actual earlier rates; do not run every speculative variant automatically.

**Tests / output:** primary effects survive generic optimization sharing, stronger baseline tuning performed without protected reuse, and the relevant mechanism controls. Transfer comparisons preserve information boundaries and do not claim contamination-free data without evidence.

**Gate / next action:** success isolated to one family narrows the claim; disappearance under a valid control changes attribution; synthetic-only success does not pass real-code validation. If intervals remain wide, call the result inconclusive and justify any fresh study by expected information, not by a need to rescue the story.

### WP11 — Synthesize, audit, and publish the actual disposition

**Objective / risk:** prevent selective reporting and an implementation that cannot be resumed or independently checked. **Entry:** valid completed outputs or a documented blocked/negative outcome. **Dependencies:** applicable preceding packages.

**Actions:** produce the report specified in Section 9, apply the three audits in Section 11 to the executed evidence, test a clean rerun of representative artifacts, and choose an outcome from Section 10. Recalculate offline amortization with measured costs. Publish all configurations and eligible aggregate results, including unsuccessful variants and limitations, subject to licenses and safe handling of generated code.

**Files / command:** `report`, `artifacts/reports/final.md`, machine-readable results table, `protocol/decisions.jsonl`, reproducibility checklist. **Resources:** CPU analysis and a measured minimal rerun.

**Gate / next action:** a reproducible narrow result can be reported without a broad breakthrough claim. Missing provenance, unexplained judge disagreements, or unreported confounding blocks a positive causal conclusion. The agent may make routine non-destructive implementation fixes, but hypothesis, population, budget, baseline, and metric changes require an amendment and an appropriate new confirmation.

### WP12 — Conditional executed-control extension, not part of the core claim

**Trigger:** evidence that a valid response model and bounded controls help repair, plus a concrete observation that intervention order or placement across actual model checkpoints matters beyond an equivalent endpoint proposal. A negative virtual-path result alone is not this evidence.

**Actions:** define a new causal controller applying bounded controls at multiple selected blocks or generation checkpoints, with actual state transitions and explicitly charged execution. Compare with direct optimization of the **entire same control schedule**, a learned one-step schedule, and appropriate local feedback control. Keep overall control degrees of freedom, intervention strength, feedback access, and compute comparable. The action may regularize the schedule, but direct schedule optimization can represent it too.

**Gate:** first show an order/placement effect on training/development examples using interventions that hold endpoints or total perturbation comparable. Then test whether the proposed objective predicts a useful schedule rather than merely adding more intervention sites. Update the prior-art assessment before claiming a new controller. This extension needs its own protocol, labels, cache rules, and fresh confirmation; it cannot retroactively validate CAVR-P.

## 8. Attribution, robustness, and failure analysis

### 8.1 Required discriminating tests

| Test | Competing explanation addressed | Required interpretation |
|---|---|---|
| Zero intervention under two method labels | Harness or timing artifact | Outcomes must agree; timing variation defines a null noise floor. |
| Correct references and known mutants | Insensitive or broken verifier | Judge must detect the intended differences before method comparisons. |
| `z`-blind and within-task shuffled-`z` predictors | Task-difficulty recognition or leakage | Genuine response prediction must deteriorate under these controls. |
| Counterexample removed or mismatched across same-type tasks | Model uses only generic repair priors | A failure-conditioned claim needs a corresponding loss of useful selection, not necessarily every individual case. |
| Equal-cost extra unsteered seeds versus random edits | Diversity alone | Separate latent controllability from additional sampling. |
| Direct multistart, finite bank, amortized endpoint, and natural-gradient policies | Extra search, capacity, or preconditioning | The path claim fails if an equally informed simpler policy removes the advantage. |
| Straight-path restricted objective | Changed objective masquerading as trajectory benefit | An advantage over K=1 alone is insufficient. |
| Identity, constant, and learned state-dependent metrics | Radius or scaling rather than learned geometry | Require held-out metric calibration and a preservation frontier benefit. |
| K scaling with fixed horizon and node-cost normalization | More nodes imply more KL/reward weight | Do not compare unnormalized objectives as though only path resolution changed. |
| Same generic cache/parser optimizations for all methods | Unequal engineering effort | Report both correctness-first and optimized shared infrastructure when relevant. |
| Accepted proposals versus all attempted proposals | Gate conceals proposer damage | Quantify rejected regressions and their cost, not only final accepted behavior. |
| Full benchmark versus oracle-sensitive cohort | Permissive checks create apparent correctness | A reliability claim must survive the stricter meaningful cohort. |
| Precision/basis/model changes with fresh labels | Instrument mismatch or accidental augmentation | Treat each frozen instrument as distinct; do not reuse invalid coordinates. |
| Independent task/failure families | Memorized errors or favorable slicing | Report transfer and uncertainty without selecting families after outcomes. |

### 8.2 Action and stationarity diagnostics

Log kinetic, predicted-failure, and behavior costs separately. Analyze their association with **measured** repair and passing-test damage within task, fixed `K`, and fixed objective scaling. Raw action values across tasks, metrics, or horizons are not automatically comparable. Normalize residual summaries for dimension where useful, while preserving raw values and the projected-gradient step size.

Test whether decreasing the fitted action during optimization improves the eventual true outcome, and whether optimizer-selected points are more miscalibrated than random held-out points. A smooth, low-residual path leading to failing code is direct evidence that the surrogate or objective is not a correctness certificate. Report stationary failures, boundary solutions, oscillation, clipping saturation, and solver restarts.

Use actual verifier outcomes only for offline diagnostic analysis and the declared online acceptance rule. Do not quietly perform additional hidden verifier queries at every interior node: that would transform the algorithm and its information/compute budget.

### 8.3 Boundary regimes and failure taxonomy

Predefine failure categories from information available before repair: syntax/import, wrong arithmetic or boundary condition, type/shape mismatch, exception, timeout, and multi-test disagreement. Keep a separate label for benchmark ambiguity and infrastructure errors. Error-family holdout must group related tasks and mutations rather than randomly splitting rows from the same program.

Stress long feedback, multiple simultaneous failures, a single failing case amid many passing ones, already-correct visible outputs, empty/invalid model output, ambiguous specifications, and adversarially formatted error strings. Test duplicate candidate collapse and attempts that change formatting but not behavior. Examine whether large interventions disproportionately harm later tokens because reference normalization is fixed.

A semantic failure, a budget cancellation, an unsupported operator, a malformed judge response, and a model with no repair headroom are different findings. Store them separately. Exclusions based on model-specific failure erase part of the effect; the primary workload retains legitimate syntax/time/resource failures as failures unless a predeclared outcome-independent eligibility rule says otherwise.

## 9. Reproducibility, experiment records, and reporting

### 9.1 Configuration and immutable identity

Use schema-validated YAML for human-readable configuration and canonical JSON for hashing. Resolve defaults into an explicit immutable run manifest before execution. Include model repository and revision, local weight/tokenizer hashes, conversion recipe, backend/source patch, dependency lock, feature/basis/predictor hashes, task and test manifests, prompt templates, dtype, anchors, layer/rank, radius, objective weights, solver/restart settings, decoding policy, seed hierarchy, and every budget limit.

The seed hierarchy separates dataset shuffling, basis construction, collection directions, counterfactual decoding, predictor initialization/bootstrap, solver initialization, final decoding, and timing-order randomization. Log actual seeds rather than only a top-level integer. Deterministic generation is useful for parity, but model/backend determinism must be tested, not assumed.

A minimal per-attempt JSONL record should contain:

```json
{
  "schema_version": 1,
  "run_id": "content-addressed-run",
  "protocol_hash": "...",
  "stage": "exploratory_or_confirmatory",
  "task_key": "opaque",
  "family_key": "opaque",
  "method_id": "...",
  "replicate_id": 0,
  "attempt_id": 0,
  "model_hash": "...",
  "basis_hash": "...",
  "predictor_hash": "...",
  "prompt_hash": "...",
  "seed_map": {},
  "proposal_artifact": "relative/path-and-hash",
  "code_artifact": "relative/path-and-hash",
  "visible_test_hash": "...",
  "status": "wrong_answer",
  "visible_pass_mask": [],
  "accepted": false,
  "costs": {
    "prefill_tokens": 0, "decode_tokens": 0,
    "model_forward_calls": 0, "model_backward_calls": 0,
    "surrogate_node_evaluations": 0, "verifier_calls": 0,
    "feature_seconds": 0.0, "solve_seconds": 0.0,
    "generate_seconds": 0.0, "verify_seconds": 0.0,
    "total_online_seconds": 0.0, "deadline_overshoot_seconds": 0.0
  },
  "memory": {"mlx_peak_bytes": null, "process_rss_bytes": null},
  "failure_reason": null,
  "resume_parent": null
}
```

The zero values above illustrate a schema, not measured output. Store paths and raw numerical diagnostics in referenced artifacts with hashes. Do not put audit fields in the online attempt record. A separate sealed audit table joins by final-output hash after generation is complete. Different methods may return identical code; record that fact rather than deduplicating it away from method-level denominators.

### 9.2 Required reports

Produce a complete task/replicate coverage table, flow of exclusions and failures, split-overlap audit, source/license manifest, baseline tuning ledger, learning curves, controllability sweep, metric calibration, and per-method quality–cost results. Include confidence intervals, discordant pairs where meaningful, seed-wise effects, and all prespecified guardrails. Report null and negative controls even when they undermine the main story.

Separate **measured results**, **estimates**, **exploratory observations**, and **confirmed effects** in prose and tables. Include initial capability, conditional repair, whole-workflow accuracy, audit rejection, online cost, total project-specific cost, and break-even scenarios. Show the larger/current-model comparison without mixing different initial-failure populations into a paired repair effect.

Record every protocol amendment with timestamp, reason, affected claims, whether any protected results were already accessible, and the new confirmation requirement. Preserve progress and next-decision records after every work package. A restarted agent must reconstruct the state from repository artifacts, not chat memory.

A clean reproduction must be able to verify hashes, recreate the environment, rerun mathematical/harness/backend tests, regenerate a declared sample of outputs, and recalculate the report from raw records. Do not redistribute model weights, protected source data, or third-party code contrary to their licenses. Publish safe instructions and hashes when redistribution is restricted.

## 10. Outcome interpretation and evidence-triggered continuation

| Observed outcome | Defensible conclusion | Next action |
|---|---|---|
| Valid path gain over the strongest equal-information endpoint policy, surviving costs and guardrails | A finite-budget proposal-path advantage in the tested regime; not extra endpoint expressivity or proof of general reasoning improvement. | Fresh workload/seed confirmation, mechanism attribution, and measured deployment scenarios. |
| Latent repair beats text, but the path ties or loses to direct control | Learned counterexample-conditioned intervention may be useful; stationary-action contribution is unconfirmed or unnecessary. | Keep the simplest winning endpoint policy. Report the path result, including its uncertainty. |
| Path gain disappears under a straight-path or preconditioned control | Changed regularization or optimization conditioning explains the effect. | Attribute and reuse that simpler ingredient; do not claim a unique variational trajectory mechanism. |
| Learned geometry helps but path does not | Behavior-aware control may improve preservation, independent of a multi-node proposal. | Refine metric calibration and direct geometry-aware optimization. |
| Oracle perturbations show headroom, predictor fails | Controllability exists in collected candidates, but amortized response learning is insufficient. | Test a specific feature/data/noise hypothesis; expand only where learning curves predict information gain. |
| Predictor succeeds, but optimized proposals fail | Surrogate exploitation, extrapolation, or objective mismatch. | Collect on-policy training counterfactuals, calibrate uncertainty, or change the objective; revalidate without protected adaptation. |
| All latent policies are slower or less accurate than tuned text repair | No local practical advantage under the measured operating points. | Stop practical claims; retain useful diagnostics. A future scale hypothesis needs positive evidence, not conjecture alone. |
| Synthetic complete-verifier success but no realistic-code transfer | Correct plumbing or a domain-specific toy result, not a validated coding improvement. | Analyze domain mismatch; do not count synthetic examples as realistic confirmation. |
| Confidence intervals include worthwhile gains and losses | Inconclusive at available sensitivity. | Report bounds and the independent-task requirement. Seek fresh data only with a preregistered design. |
| Failure due to memory, unsupported derivatives, or unavailable licensed assets | Local feasibility limitation of the affected variant. | Use the faithful fallback ladder; narrow the claim rather than changing the experiment invisibly. |
| Gain only on permissive oracles, leaked families, or selected seeds | Invalid or misleading positive result. | Correct the evaluation, disclose the defect, and obtain fresh confirmation. |
| Equivalent prior art is found | Novelty claim must shrink to replication, transfer, engineering, or a verified incremental difference. | Update the comparison and reuse the prior implementation where appropriate. |

A more ambitious internal-error-prediction extension, without external verification at every repair, is not part of this program. It would remove the most reliable grounding signal and require an independent study of error detection, confidence, and false intervention. Do not treat it as a natural consequence of passing finite coding tests.

## 11. Audit record, unresolved risks, and limitations

### 11.1 Scientific and prior-art audit performed during this research

This was an internal challenge-and-revision process, not independent peer review.

| Defect or challenge found | Revision made | Residual uncertainty / acceptance criterion |
|---|---|---|
| A nonlinear metric was implicitly expected to evade endpoint reduction. | Added general endpoint elimination, the restricted straight-path comparator, and explicit finite-budget rather than expressivity claims. | Only controlled empirical search results can establish a practical path benefit. |
| TTC's units and training conditions could overstate comparable evidence. | Inspected the paper table and repository; corrected percentage points and joint training; excluded incomparable published scores from head-to-head results. | A faithful TTC replication would still require exact checkpoint/data/hardware clarification. |
| CAVR resembled ordinary controlled generation more than the initial physics-only framing suggested. | Added PPLM, representation interventions, feedback steering, COLD, counterexample repair, and CHOMP to the novelty constraints. | Literature coverage is not proof of absence of an equivalent system. |
| Virtual proposal nodes were described as internal reasoning. | Named CAVR-P explicitly and separated the executed-control extension. | No claim about causal reasoning quality follows from outcome improvement alone. |
| “Minimum intervention” implied semantic preservation. | Separated latent norm, finite-anchor KL, passing-test retention, and protected behavior. | No finite incomplete test suite proves unrestricted program equivalence. |
| Counterexample conditioning was called a boundary condition. | Specified true feasibility constraints and removed the physical-boundary implication. | A future hard endpoint constraint is a changed method. |

### 11.2 Experimental-validity audit performed during drafting

| Defect or challenge found | Revision made | Residual uncertainty / acceptance criterion |
|---|---|---|
| MBPP+ crosses the original MBPP splits. | Required global task/family reservation before training, basis/PCA fitting, or demonstrations. | Actual surviving counts are an early metadata gate; contamination in foundation-model pretraining remains unknown. |
| Finite tests and special oracles could reward trivial or nonsemantic behavior. | Inspected evaluator logic; added official-score fidelity, oracle-sensitive reporting, and a separate bounded complete-verifier task. | Official benchmark success still has limited semantic scope. |
| Generated Python could spoof a pass or inspect expected answers inside an ordinary harness. | Added isolated candidate execution, external trusted comparison, data-only result serialization, and adversarial harness fixtures. | Container security is not formally proved; judgment parity is a blocking local test. |
| Increasing `K` changed KL strength in the starting formula. | Fixed horizon and running-cost normalization; retained original scaling only as a labeled diagnostic. | Other hyperparameter interactions require development sensitivity tests. |
| Predictor accuracy could come from task difficulty rather than counterfactual effects. | Added within-task ranking, grouped splits, `z`-blind/shuffled controls, and real execution of optimized proposals. | Useful generalization may require more independent task families than are available. |
| Three seeds or hundreds of tests could be counted as independent evidence. | Defined task/family units, repeated-training analysis, and an explicit sensitivity calculation. | A few training replicas cannot precisely characterize all seed variation. |
| A small benchmark could be assumed able to resolve a three-point gain. | Calculated illustrative paired power; required pilot-based simulation and honest inconclusive outcomes. | Small effects may remain unresolved locally without a larger permitted workload. |
| Acceptance bypass could create “zero regressions” by construction. | Required triggered-case proposer damage and audit analysis, not merely untouched correct outputs. | The trusted gate guarantees only its declared visible-test rule. |
| Additional labels, KL passes, caching, or tuning could be hidden. | Shared assets and generic optimizations; separated offline/online/evaluation-only costs; added amortization and tuning ledgers. | Practical value depends on future reuse volume and latency requirements not supplied by the user. |

### 11.3 Implementation and feasibility audit performed during drafting

| Execution challenge | Revision made | Blocking test or next branch |
|---|---|---|
| A fresh Mac might not support the assumed runtime/precision. | Added native hardware discovery, isolated environment locking, and a tiny-model backend test before weight-scale work. | WP03 parity and memory qualification. |
| Nominal MPS support could hide unsupported full-model gradients. | Kept S's solver on CPU and the frozen LM on MLX; separated E's derivative gate. | No silent fallback or claim of measured M3 compatibility. |
| A trainable basis made historical labels stale. | Froze the basis per collection round and added hash rejection. | Any changed instrument requires fresh labels or an exact checked map. |
| Cache reuse could corrupt counterfactual comparisons. | Specified absolute intervention positions and independent pre-intervention cache branches. | Cached/uncached and cross-proposal tests must pass. |
| Full logits or simultaneous copies could defeat an apparently small model budget. | Added explicit logit/KV/weight calculations, anchor-only projection, and measured allocation ladders. | Stable representative-length runs without severe memory pressure. |
| Partial regeneration required an unspecified localization mechanism. | Used whole-function repair in the core and made suffix repair a later shared optimization. | No unearned cost advantage from fictional affected-region knowledge. |
| Hard deadlines could omit slow unsuccessful work. | Centralized budget ownership, reserve estimates, overshoot logging, and cancellation outcomes. | Null timing and deadline tests before confirmation. |
| An interrupted run could resume with changed weights or different timing conditions. | Added atomic content-addressed checkpoints and restart rules for invalid timing blocks. | `test_resume.py` must reject mismatches and preserve failed work. |
| A permissive scorer adaptation could become a different benchmark. | Required source-faithful oracle extraction and parity against the original inside isolation. | Unexplained judgment differences block the primary comparison. |

A subsequent challenge pass found three additional concrete ambiguities and corrected them: the Schur-complement formula now explicitly excludes active interior constraints; the primary endpoint comparison can select a finite-bank or amortized policy rather than only iterative optimizers; and numerical sign/scale constraints, empty-code anchors, explicit CPU placement, and configuration validation are specified. The portable Appendix A script was extracted from this document and rerun successfully after assembly. Reference identifiers, code-fence balance, UTF-8 decoding, and absence of chat-only citation markers were also checked programmatically. These checks validate the handoff's internal artifacts, not the future experiment.

A further consistency pass checked the main dependencies: the data split precedes feature/basis training; mathematical checks precede solver claims; controllability precedes expensive label growth; predictor validation precedes path comparison; and protocol freezing precedes protected method outcomes. The remaining unresolved items are physical-laptop measurements, exact post-reservation dataset counts, validated local judgment/backend parity, and the scientific hypotheses themselves. They are explicit gates, not completed tests.

### 11.4 What this handoff cannot establish by itself

No LLM repair experiment, M3 Max benchmark, or independent reproduction was executed during this research. The CPU mathematical checks are real and reproducible, but they do not prove Metal compatibility or a positive repair effect. Public benchmark contamination cannot be ruled out from model cards alone. Protected test results are still finite and sometimes weakly informative. No legal patentability or freedom-to-operate conclusion is offered. A successful local result would justify its precise tested claim—not wholesale replacement of Transformers, reliable self-correction without external grounding, or an advantage at every scale.

## 12. References and search-coverage record

### 12.1 How to use the evidence register

The register gives stable paper identifiers and source URLs, the inspected material, and the decision it informs. All web/code documentation was accessed for this research on **13 October 2026**. An arXiv version URL is version-specific; repository `main`/`master` URLs and model cards are rolling sources. Their local execution revisions must be resolved and hashed in WP00 rather than invented here. “Inspected” does not mean reproduced or exhaustively audited. Source availability is not a finding of numerical compatibility.

Conference labels in the supplied proposal were not all independently established. TTC's ICML proceedings entry was inspected; where this register cites an arXiv paper without a verified proceedings entry, its existence and inspected content—not every claimed venue—are the verified facts.

**[R01] Wang, Peihao, et al. (2026). _Beyond Test-Time Memory: State-Space Optimal Control for LLM Reasoning_ / TTC.** arXiv:2603.09221, inspected v2. [Paper](https://arxiv.org/html/2603.09221v2); [PDF](https://arxiv.org/pdf/2603.09221v2); [ICML proceedings](https://proceedings.mlr.press/v306/wang26jt.html); [author repository](https://github.com/VITA-Group/TTC-Net). Inspected methods/training discussion, repository README, and a screenshot of Table 2 on PDF page 11. Informs the close-control precedent and baseline/percentage-point correction. Published training conditions are not a frozen-M3 comparison; exact checkpoint naming should be reconciled with code before any reproduction.

**[R02] Nguyen, Tuc, and Thai Le (2026). _ATLAS: Verifier-Guided Adaptive Latent Activation Steering for Efficient LLM Reasoning_.** arXiv:2601.03093. [Inspected v4](https://arxiv.org/html/2601.03093v4). Inspected method and latent-verifier architecture/supervision descriptions. Informs the learned steering-bank baseline and separation of online verifier cost from supervisory-data cost. No usable official implementation was established in this search; local code-task adaptation is not paper-faithful reproduction.

**[R03] Skifstad, Julian; Xinyue Annie Yang; and Glen Chou (2026). _Local Linearity of LLMs Enables Activation Steering via Model-Based Linear Optimal Control_.** arXiv:2604.19018. [Paper](https://arxiv.org/html/2604.19018v1); [author-linked code](https://github.com/trustworthyrobotics/lqr-activation-steering). Inspected linearization, feedback, computation discussion, and implementation link. Informs the local-linear/natural-gradient alternatives and offline Jacobian accounting. Semantic steering is not the protected-test repair protocol; code was not run on M3.

**[R04] Nguyen, Dung V., et al. (2025; subsequent 2026 venue context). _Activation Steering with a Feedback Controller_.** arXiv:2510.04309. [Paper](https://arxiv.org/html/2510.04309v1); [record](https://arxiv.org/abs/2510.04309). Inspected the feedback-controller formulation. Informs the conclusion that adaptive feedback and PID-like activation correction are established, not novel here. No local replication was performed.

**[R05] Dathathri, Sumanth, et al. (2019/2020). _Plug and Play Language Models: A Simple Approach to Controlled Text Generation_.** arXiv:1912.02164. [Paper record](https://arxiv.org/abs/1912.02164); [repository](https://github.com/uber-research/PPLM); [inspected implementation](https://raw.githubusercontent.com/uber-research/PPLM/master/run_pplm.py). Inspected abstract and inference-time perturbation/KL code. Informs the strongest historical endpoint-optimization equivalence. Original attribute generation and legacy model/software differ from the present repair experiment.

**[R06] Wu, Zhengxuan, et al. (2024). _ReFT: Representation Finetuning for Language Models_.** arXiv:2404.03592. [Record](https://arxiv.org/abs/2404.03592). Inspected the primary abstract and representation-intervention description. Informs reuse of low-rank activation intervention rather than an architectural novelty claim. No detailed MLX port or reproduction was established from this source.

**[R07] Wang, Peihao, et al. (2026). _∇-Reasoner: LLM Reasoning via Test-Time Gradient Descent in Latent Space_.** arXiv:2603.04948. [Paper](https://arxiv.org/html/2603.04948v1). Inspected representation and optimization formulation. Informs the distinction between relaxed textual-representation refinement and a low-dimensional hidden intervention. Model-differentiation costs and the different output parameterization limit direct numerical comparison.

**[R08] Chen, Xinyun; Maxwell Lin; Nathanael Schärli; and Denny Zhou (2023). _Teaching Large Language Models to Self-Debug_.** arXiv:2304.05128. [Record](https://arxiv.org/abs/2304.05128). Inspected primary description of self-debugging and execution feedback. Informs the strong textual baseline. This plan does not quote its published gains as a local controlled result.

**[R09] Orvalho, Pedro; Mikoláš Janota; and Vasco Manquinho (2025). _Counterexample Guided Program Repair Using Zero-Shot Learning and MaxSAT-based Fault Localization_.** arXiv:2502.07786. [Paper](https://arxiv.org/html/2502.07786v1). Inspected the counterexample/repair/fault-localization workflow. Informs prior art for the external repair loop and an alternative source of value from localization. Its task and model setup differ. This identifier is **not** the recurrent-depth paper.

**[R10] Ratliff, Nathan; Matthew Zucker; J. Andrew Bagnell; and Siddhartha Srinivasa (2009). _CHOMP: Covariant Hamiltonian Optimization for Motion Planning_.** [Author research page](https://www.nathanratliff.com/thesis-research/chomp). Inspected the author explanation of trajectory costs and covariant optimization. Informs classical equivalents and the preconditioned-path control. It is an adjacent-field precedent, not evidence that the language-model repair adaptation succeeds.

**[R11] Fang, Xiang, and Wanlong Fang (2026). _SLAP: The Semantic Least Action Principle for Variational Video-Language Modeling_.** arXiv:2605.30750. [Paper](https://arxiv.org/html/2605.30750v1). Inspected semantic-trajectory and discrete variational formulation. Informs the scope correction from generic “semantic action” to a video-language setting. A code-repair advantage and the supplied conference label were not independently established by this inspection.

**[R12] Zimin, Aleksandr; Yury Polyanskiy; and Philippe Rigollet (2026). _YuriiFormer: A Suite of Nesterov-Accelerated Transformers_.** arXiv:2601.23236. [Paper](https://arxiv.org/html/2601.23236v1). Inspected optimizer-derived architecture discussion. Informs the novelty limit on acceleration-inspired Transformer computation. Its trained architecture is not an online frozen-repair comparator.

**[R13] Gai, Jingchu; Nai-Chieh Huang; and Jiayun Wu (2026). _Momentum Streams for Optimizer-Inspired Transformers_.** arXiv:2605.24425. [Paper](https://arxiv.org/html/2605.24425v1); [author-linked repository](https://github.com/gaijingchu/Momentum-Streams-for-Optimizer-Inspired-Transformers). Inspected the momentum-stream mechanism and comparison framing. Informs why second-order or optimizer-inspired residual dynamics alone are insufficient novelty. Repository existence is not local replication.

**[R14] Hoover, Benjamin, et al. (2023). _Energy Transformer_.** arXiv:2302.07253. [Record](https://arxiv.org/abs/2302.07253). Inspected primary description of energy-driven attention computation. Informs historical scope, not a detailed repair baseline or a claimed local reproduction.

**[R15] Karkar, Skander; Ibrahim Ayed; Emmanuel de Bézenac; and Patrick Gallinari (2020). _A Principle of Least Action for the Training of Neural Networks_.** arXiv:2009.08372. [Record](https://arxiv.org/abs/2009.08372). Inspected transport/minimal-motion training motivation. Informs the conclusion that minimizing representation motion is established. Different training and task setting; no inference-time repair evidence is inferred.

**[R16] Geiping, Jonas, et al. (2025). _Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach_.** arXiv:2502.05171. [Paper](https://arxiv.org/html/2502.05171v1); [record](https://arxiv.org/abs/2502.05171); [author code](https://github.com/seal-rg/recurrent-pretraining). Inspected the executed recurrent-computation approach and source project. Informs the distinction between actual latent recurrence and virtual proposal nodes. Its training expenditure is outside the core laptop program.

**[R17] Williams, Jonathan, and Esin Tureci (2026). _Prioritize the Process, Not Just the Outcome: Rewarding Latent Thought Trajectories Improves Reasoning in Looped Language Models_.** arXiv:2602.10520. [Inspected formulation](https://arxiv.org/html/2602.10520v1); [current record](https://arxiv.org/abs/2602.10520). Inspected the loop/trajectory reward formulation and checked the current v3 record, which identifies ICML 2026 and links author code. Informs existing path-level learning signals. Do not interpret its process-supervision design as a general certificate of verifier-free correctness or as the same intervention-search problem.

**[R18] Jiang, Haitong, et al. (2026). _Exact Feedback Is Not Control: Evaluating Text-based Closed-Loop Revision in LLMs_.** arXiv:2609.28150. [Paper](https://arxiv.org/html/2609.28150v1). Inspected deterministic constraint/revision evaluation. Informs conservative feedback and control claims. Different constraints and models mean its results neither prove nor disprove CAVR's coding hypothesis.

**[R19] Yu, Qinan, et al. (2026). _Outcome Rewards Do Not Guarantee Verifiable or Causally Important Reasoning_.** arXiv:2604.22074. [Paper](https://arxiv.org/html/2604.22074v1). Inspected distinctions between outcome performance and reasoning properties. Informs attribution and limits on mechanistic interpretation. It is not evidence that a particular proposed repair succeeds or fails.

**[R20] Qwen team (2024; rolling model repository). _Qwen2.5-Coder-1.5B-Instruct_.** [Model card](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct); [inspected configuration](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct/raw/main/config.json). Inspected architecture, weight dtype, and license metadata. Informs the primary instrument and derived memory calculations. Card performance claims are not treated as current superiority; exact local weight/tokenizer revisions remain to be frozen.

**[R21] Qwen team (2026; rolling model repository). _Qwen3.5-4B_.** [Model card](https://huggingface.co/Qwen/Qwen3.5-4B). Inspected model identity, architecture family, and license. Informs a contemporary compact challenger. Manufacturer benchmarks and nominal support do not establish its local repair frontier or intervention compatibility.

**[R22] Nanbeige LLM Lab (2026; rolling model repository). _Nanbeige4.1-3B_.** [Model card](https://huggingface.co/Nanbeige/Nanbeige4.1-3B). Inspected model family and license metadata. Informs the predeclared alternative compact challenger. No local speed or quality was measured here.

**[R23] Google Research / Austin et al. (2021; maintained dataset documentation). _Mostly Basic Python Problems (MBPP)_.** [Original project README](https://raw.githubusercontent.com/google-research/google-research/master/mbpp/README.md); [dataset card](https://huggingface.co/datasets/google-research-datasets/mbpp); [card source](https://huggingface.co/datasets/google-research-datasets/mbpp/raw/main/README.md). Inspected split ID ranges, counts, and dataset licensing. Informs allocation and attribution. The exact post-reservation local counts have not been enumerated here.

**[R24] EvalPlus maintainers / Liu et al. (2023 onward). _EvalPlus / HumanEval+ / MBPP+_.** [Repository](https://github.com/evalplus/evalplus). The following inspected components support distinct decisions; retain both repository-level and file-level licenses.

- **[R24a] Release and dataset documentation.** [README](https://raw.githubusercontent.com/evalplus/evalplus/master/README.md); [MBPP+ v0.2.0 release](https://github.com/evalplus/mbppplus_release/releases/tag/v0.2.0); [dataset viewer/card](https://huggingface.co/datasets/evalplus/mbppplus). Informs the 378-task release, benchmark distinction, and visible task-ID range. Release metadata and viewer inspection are not a downloaded full-data audit.
- **[R24b] MBPP loader and serialization.** [Loader](https://raw.githubusercontent.com/evalplus/evalplus/master/evalplus/data/mbpp.py); [data utilities](https://raw.githubusercontent.com/evalplus/evalplus/master/evalplus/data/utils.py). Inspected version selection, serialization cases, and asset construction. Informs the cross-original-split reservation requirement and structured-input handling.
- **[R24c] Evaluation orchestration.** [Source](https://raw.githubusercontent.com/evalplus/evalplus/master/evalplus/evaluate.py). Inspected reference-output construction, base/plus separation, and final scoring logic. Informs the public/audit interface and prohibition on passing full problem objects to the repair policy.
- **[R24d] Execution and comparison logic.** [Source](https://raw.githubusercontent.com/evalplus/evalplus/master/evalplus/eval/__init__.py). Inspected execution, timeout, tolerance, special-oracle, and process-result paths. Informs judgment fidelity, weak-oracle warnings, and the security adaptation. The code's own guards are not treated as a proven sandbox.
- **[R24e] Special oracles.** [Source](https://raw.githubusercontent.com/evalplus/evalplus/master/evalplus/eval/_special_oracle.py). Inspected set-equivalence and nonserializable-output rules. Informs typed serialization and the prespecified oracle-sensitive cohort.
- **[R24f] Repository license.** [License](https://raw.githubusercontent.com/evalplus/evalplus/master/LICENSE). Inspected the repository license text. Underlying dataset and individual-file notices still require preservation.

**[R25] Apple MLX maintainers (documentation inspected 2026-10-13).**

- **[R25a] Installation requirements.** [Documentation](https://ml-explore.github.io/mlx/build/html/install.html). Informs native Apple Silicon/macOS/Python prerequisites; exact selected wheels still need validation.
- **[R25b] `mlx.core.synchronize`.** [API documentation](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.synchronize.html). Informs timing synchronization. No actual M3 timings were collected.
- **[R25c] `mlx.core.get_peak_memory`.** [API documentation](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.get_peak_memory.html). Informs byte-valued allocation measurement and the need to distinguish allocator counters from total physical-memory use.
- **[R25d] MLX license.** [Inspected license](https://raw.githubusercontent.com/ml-explore/mlx/main/LICENSE). MIT license; informs permitted reuse and notice preservation, without replacing review of bundled dependencies.

**[R26] MLX-LM maintainers (rolling implementation inspected 2026-10-13).** [Repository](https://github.com/ml-explore/mlx-lm); [Qwen2 implementation](https://raw.githubusercontent.com/ml-explore/mlx-lm/main/mlx_lm/models/qwen2.py); [Qwen3.5 implementation](https://raw.githubusercontent.com/ml-explore/mlx-lm/main/mlx_lm/models/qwen3_5.py); [MIT license](https://raw.githubusercontent.com/ml-explore/mlx-lm/main/LICENSE). Inspected the license, block forward structure, cache mutation/offset behavior, and the distinct hybrid model implementation. Informs the explicit intervention wrapper and model-specific compatibility gates. Record a commit and patch before execution; source inspection does not establish gradient or performance parity.

**[R27] Podman maintainers (current documentation inspected 2026-10-13).** [Podman machine](https://docs.podman.io/en/latest/markdown/podman-machine.1.html); [Podman run](https://docs.podman.io/en/latest/markdown/podman-run.1.html); [Apache 2.0 license](https://raw.githubusercontent.com/containers/podman/main/LICENSE). Inspected the license, macOS VM route and isolation/resource options. Informs the local candidate sandbox. Actual installed-version flags, ARM64 image behavior, and enforcement must pass WP02; no security proof is claimed.

**[R28] Qin, Lianhui; Sean Welleck; Daniel Khashabi; and Yejin Choi (2022). _COLD Decoding: Energy-based Constrained Text Generation with Langevin Dynamics_.** arXiv:2202.11705. [Record](https://arxiv.org/abs/2202.11705). Inspected primary description of differentiable energy-constrained generation. Informs adjacent terminology and a historical alternative to action-based framing. Detailed local reproduction was not performed.

**[R29] LiveCodeBench authors and maintainers (2024 onward). _LiveCodeBench_.** [Repository](https://github.com/LiveCodeBench/LiveCodeBench). Inspected the maintained benchmark source/project description as a possible later independent workload. Exact task release, eligible license, adapter, and model-contamination assumptions are unresolved for that optional extension; it is not a hidden dependency of the core PoC.

### 12.2 Search coverage and remaining blind spots

Research covered the supplied named papers, exact-title/version verification, primary implementations, and adjacent terminology. Representative query families included “test-time control” and symplectic LLMs; latent verifiers and adaptive activation steering; feedback/PID/LQR/Jacobian steering; frozen-state optimization and KL; counterexample-guided and latent program repair; energy-based/relaxed-text decoding; learned trajectory costs and covariant motion planning; recurrent-depth and latent-trajectory rewards; exact feedback and reasoning-causality negative results; Apple Silicon model/gradient/cache support; and benchmark splits, loaders, special oracles, and licenses.

Backward tracing exposed classical trajectory optimization and controlled generation as stronger novelty constraints than a physics-only search. Adjacent-method searches added feedback controllers and counterexample/fault-localization repair. Forward/current checks included 2026 steering/reasoning studies and current compact model/library sources. Implementation inspection materially changed the plan through cache semantics, exact-KL costs, cross-split MBPP+ membership, and permissive evaluator branches.

The final mechanism-focused searches ceased to change the main competing explanations: learned steering, regularized endpoint optimization, feedback control, and trajectory-preconditioned search cover the decisive local comparisons. This is a coverage judgment, not an exhaustive catalog or proof of originality. Software and data inspection continued to produce protocol corrections and therefore was not stopped at a paper quota.

Patent-oriented searches and exact phrase searches were noisy and did not establish a validated relevant patent family or claim interpretation. **Patent coverage is inconclusive**; no patentability or freedom-to-operate opinion follows. Other blind spots include inaccessible/unlocated implementations, unverified venue labels for some preprints, future work after the research date, undisclosed deployed systems, foundation-model training data, and exact reproducibility of published large-scale results. A newly identified equivalent method requires an amended novelty assessment, not suppression of that source.

## Appendix A. Mathematical checks actually executed

### A.1 Environment and results

Executed in this research environment, **not on a Mac**: Linux x86_64, Python 3.13.5, NumPy 2.3.5, SciPy 1.17.0, PyTorch 2.10.0+cpu. Seed: 20261013. Float64 was used for action/derivative checks.

| Check | Measured result |
|---|---:|
| Full quadratic path vs eliminated endpoint, maximum endpoint error | `1.1102230246251565e-16` |
| Reconstructed full path error | `1.1102230246251565e-16` |
| Quadratic objective difference | `8.881784197001252e-16` |
| Full quadratic gradient infinity norm | `2.1094237467877974e-15` |
| Straight-path kinetic action, K = 1, 2, 4, 8 | Approximately `3.578720086962768` for every K |
| Manual state-dependent DEL vs automatic differentiation | Maximum error `4.440892098500626e-16` |
| Automatic gradient vs finite-difference gradient check | Passed |
| Boundary optimum for minimizing `-z` on `[-1,1]` | Raw gradient `-1`; projected gradient `0` |

The script also computes the explicitly hypothetical power figures and architecture-based memory contributors in Sections 5–6. Those are calculations, not measured task performance or M3 memory usage. No repair-success benchmark was executed.

### A.2 Reproducible script

Save the following as `math_checks.py` in the project and run `python math_checks.py` in an environment with NumPy, SciPy, and PyTorch. The output path below is made relative for portability; the numerical calculations are the ones executed during this research. A package-version-dependent variation at roundoff scale is expected; the assertions express the intended tolerances.

```python
"""CPU mathematical checks only; no LLM or Apple-hardware benchmark."""
from __future__ import annotations
import json, platform, sys
from pathlib import Path
import numpy as np
import scipy
from scipy.stats import norm
import torch

def main() -> dict:
    rng = np.random.default_rng(20261013)
    out = {"environment": {"platform": platform.platform(), "python": sys.version.split()[0],
           "numpy": np.__version__, "scipy": scipy.__version__, "torch": torch.__version__}, "seed": 20261013}
    d, K, T = 3, 4, 1.0
    dt = T/K
    M = rng.normal(size=(d,d)); G = M.T@M+np.eye(d)
    N = rng.normal(size=(d,d)); H = N.T@N+.2*np.eye(d)
    F = np.diag([.4,.8,1.3]); a = rng.normal(size=d)
    lam, mu = .7,.2
    D = np.eye(K*d)
    for k in range(1,K): D[k*d:(k+1)*d,(k-1)*d:k*d] = -np.eye(d)
    Q = D.T@np.kron(np.eye(K),G/dt)@D + np.kron(np.eye(K),dt*(lam*H+mu*F))
    b = np.tile(dt*lam*H@a,K)
    x = np.linalg.solve(Q,b)
    Qii,Qie,Qee = Q[:-d,:-d],Q[:-d,-d:],Q[-d:,-d:]
    bi,be = b[:-d],b[-d:]
    S = Qee-Qie.T@np.linalg.solve(Qii,Qie)
    beff = be-Qie.T@np.linalg.solve(Qii,bi)
    e = np.linalg.solve(S,beff)
    xs = np.r_[np.linalg.solve(Qii,bi-Qie@e),e]
    obj = lambda xx: float(.5*xx@Q@xx-b@xx)
    out["quadratic_elimination"] = {"max_endpoint_error":float(np.max(abs(x[-d:]-e))),
          "max_path_error":float(np.max(abs(x-xs))),"objective_error":abs(obj(x)-obj(xs)),
          "gradient_linf":float(np.max(abs(Q@x-b)))}
    assert np.allclose(x,xs,atol=1e-11,rtol=1e-11)
    v = rng.normal(size=d)
    straight=[]
    for steps in (1,2,4,8):
        z = np.arange(steps+1)[:,None]*v/steps
        dz = np.diff(z,axis=0)
        A = .5/(T/steps)*np.einsum('ki,ij,kj->',dz,G,dz)
        exact = .5/T*v@G@v
        straight.append({"K":steps,"action":float(A),"error":float(abs(A-exact))})
        assert abs(A-exact)<1e-12
    out["straight_path"] = straight
    torch.set_default_dtype(torch.float64)
    z = torch.tensor(rng.normal(scale=.2,size=(K,d)),requires_grad=True)
    C = torch.tensor(rng.normal(scale=.3,size=(d,d)))
    at,Ft = torch.tensor(a),torch.tensor(F)
    def action(zz):
        allz = torch.cat((torch.zeros(1,d),zz),dim=0)
        A = sum(.5/dt*((allz[k+1]-allz[k])**2*(1+torch.nn.functional.softplus(C@allz[k]))).sum() for k in range(K))
        return A + sum(dt*(lam*.5*((allz[j]-at)**2).sum()+mu*.5*allz[j]@Ft@allz[j]) for j in range(1,K+1))
    grad = torch.autograd.grad(action(z),z)[0].detach().numpy()
    allz=np.vstack((np.zeros(d),z.detach().numpy())); Cn=C.numpy(); manual=[]
    for j in range(1,K+1):
        r=(1+np.logaddexp(0,Cn@allz[j-1]))*(allz[j]-allz[j-1])/dt
        if j<K:
            dz=allz[j+1]-allz[j]; logits=Cn@allz[j]
            r=r-(1+np.logaddexp(0,logits))*dz/dt + Cn.T@((1/(1+np.exp(-logits)))*dz**2)/(2*dt)
        r=r+dt*(lam*(allz[j]-a)+mu*F@allz[j]); manual.append(r)
    err=float(np.max(abs(grad-np.array(manual))))
    out["state_dependent_DEL"]={"manual_autograd_error":err,"finite_difference_gradcheck":bool(torch.autograd.gradcheck(action,(z,),eps=1e-6,atol=1e-6,rtol=1e-5))}
    assert err<1e-11
    point,derivative,eta = 1.,-1.,.1
    residual=(point-np.clip(point-eta*derivative,-1.,1.))/eta
    out["constrained_optimum"]={"gradient":derivative,"projected_gradient":float(residual)}
    assert residual==0.
    zsum=norm.ppf(.975)+norm.ppf(.8)
    out["power_approximation"]={"zsum_squared":float(zsum**2),"mde_pp_n378_q02":float(100*zsum*np.sqrt(.2/378)),
          "n_for_3pp_q02":int(np.ceil(zsum**2*.2/.03**2)),"n_for_2pp_q02":int(np.ceil(zsum**2*.2/.02**2))}
    L,nh,nkv,h,ff,voc = 28,12,2,1536,8960,151936
    hd=h//nh
    params=voc*h+L*(h*h+h+2*(h*nkv*hd+nkv*hd)+h*h+3*h*ff+2*h)+h
    out["memory_estimates"]={"parameters":params,"fp16_weights_gib":params*2/2**30,
          "kv_2048_mib":2*L*nkv*hd*2048*2/2**20,"kv_4096_mib":2*L*nkv*hd*4096*2/2**20,
          "all_logits_2048_fp32_gib":2048*voc*4/2**30,"logits_16_fp32_mib":16*voc*4/2**20}
    return out

if __name__ == '__main__':
    result=main(); print(json.dumps(result,indent=2))
    Path('math_checks_results.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
```
