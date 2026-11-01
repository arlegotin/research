# Idea #1 — Research and executable proof-of-concept specification

**Research date:** 3 October 2026  
**Status:** Reviewed research specification; mathematical reference checks executed; neural comparison and Mac measurements not yet executed.  
**Target:** One local Apple Silicon MacBook with an M3 Max chip; memory configuration discovered at startup.  
**Primary project:** Fixed, input-conditioned compatibility caches for adapting a shared producer while preserving a frozen consumer.  
**Version:** 1.0

## Navigation

[1. Executive decision](#executive) · [2. Source reconstruction](#source) · [3. Technical landscape and prior art](#landscape) · [4. Formalization and deductions](#formal) · [5. Hypotheses](#hypotheses) · [6. Algorithms and interfaces](#algorithms) · [7. Data and models](#data) · [8. Baselines and fairness](#baselines) · [9. Statistical protocol](#statistics) · [10. Mac implementation and resources](#hardware) · [11. Repository and reproducibility](#repository) · [12. Experiment ladder](#ladder) · [13. Tests and executed checks](#checks) · [14. Conditional routing branch](#routing) · [15. Traceability](#traceability) · [16. Adversarial reviews](#reviews) · [17. References and search record](#references) · [Appendix A. Executed reference code](#appendix-a) · [Appendix B. Configuration and reporting contracts](#appendix-b).

<a id="executive"></a>
## 1. Executive decision and claim boundaries

### 1.1 Recommendation

**Proceed with a bounded-by-evidence investigation of a refined evaluation of the original compatibility proposal, not with a claim of a new general neural architecture.** Preserve its essential mechanism: associate each retained training input with the frozen downstream consumer's local feature sensitivity, then use those fixed sensitivities during producer adaptation. Change the comparison protocol substantially. In particular, add cached linearized-logit distillation, cheaper conditional metrics, and a private producer as serious alternatives.

The defensible novelty claim is presently **an application/implementation and controlled evaluation of established functional-regularization machinery under an explicit interface-access constraint**. A useful input-association effect is a hypothesis, not an established new algorithmic principle. Elastic Feature Consolidation (EFC) already constructs local feature-space sensitivity before averaging it; functional regularization, Jacobian compression, backward-compatible training, and derivative-informed distillation cover much of the surrounding mechanism. [R01][R04][R05][R06][R10]

The strongest reason the proposal might work is that a nonlinear consumer can use different feature directions at different inputs. A single global metric then penalizes some changes that are locally harmless, while isotropic feature matching penalizes all changes. A fixed local cache may permit more useful producer adaptation without repeatedly evaluating an expensive consumer.

The strongest reason it might fail is that **a local quadratic is a lossy, stale approximation to a directly available distillation objective**, and its additional memory may buy less than more replay examples, a simple adapter, a cached linearized consumer, or a second producer. At the source proposal's small MNIST architecture, the rank-eight factors alone occupy 1,048,576 bytes, whereas another FP32 producer occupies 935,424 bytes. This is exact parameter/storage arithmetic, not a Mac measurement. A domain-known system permitted to keep two producers can preserve the old producer exactly. A genuine single-shared-producer requirement must therefore be stated rather than imposed merely to make the proposal competitive.

**First decisive experiment:** after correctness checks, run a paired, development-selected retention–adaptation comparison of full local geometry, rank-eight local geometry, global geometry, feature L2, cached linearized-logit distillation, exact output distillation, and frozen/private producers. Include a retrained spectrum-preserving orientation control. Use the original clean-to-90°-rotated image task for continuity, but first verify that the task actually requires producer change and has input-dependent consumer geometry. A constructed two-anchor test establishes harness sensitivity; it does not establish practical value.

### 1.2 What success would mean

Four questions must remain separate:

| Question | Required evidence | What it would not establish |
|---|---|---|
| Does the mechanism behave as derived? | Numerical identities, null controls, and a positive controlled construction | Better neural learning or biological realism |
| Does local geometry improve the shared-producer tradeoff? | Protected paired comparisons at a predeclared retention operating point and the accompanying frontier | Superiority to unconstrained multi-producer systems |
| Is the improvement due to the proposed association? | Adequately tuned orientation, scalar-weighting, coarse-conditional, and memory controls | A special social or cognitive relationship representation |
| Does caching have practical value? | Full quality–storage–access–time accounting against the strongest admissible alternatives | An automatic speedup from fewer consumer calls |

Do not reuse the supplied proposal's proposed two-percentage-point gain, ten-percent speedup, or fivefold access reduction as universal success requirements. They were investment choices, not evidence of useful effect sizes. This plan reports application-dependent break-even conditions and uses development variability to determine confirmation precision. The inherited one-percentage-point retention limit remains a **benchmark operating point**, not a claim about acceptable degradation in a deployment.

### 1.3 Decision on the second interpretation

Keep adaptive internal fusion as a conditional, lower-priority branch. Tracking the realized gain of one fixed pair is already equivalent to tracking the utility of a composite “A+B” action. Current adaptive-deferral and complementary-routing work further weakens a broad novelty claim. The shared-residual structure in the current L2D-SLDS formulation must not be omitted from its description. [R13] FlexRouter concerns complementary model sets, while MoCA makes complementarity depend on a final summarizer. [R14][R15] Section 14 specifies the narrow causal assay worth retaining; it is not a parallel mandate to build a large learned router.

### 1.4 Material blockers and evidence limits

There is **no identified mathematical or model-size blocker** to the essential local experiments. Actual MPS correctness, available memory, training throughput, and statistical precision are unmeasured and are explicit execution gates. Current software documentation verifies an execution route, not this machine's behavior.

The uploaded source describes earlier neural pilots, simulations, and downloadable code. Only its text was supplied here. The historical archive and checkpoints were not available and were not independently reproduced. This document does not promote those numbers into new evidence. The independently executed checks in Section 13 and Appendix A are small CPU mathematical tests, not adaptation training or M3 benchmarks.

A primary-source license grant for the original MNIST data was not established in this investigation. TorchVision's software license does not grant dataset rights. The essential experiment path therefore has an explicit MIT-licensed Fashion-MNIST alternative, and the representative sensor experiment uses UCI HAR's stated CC BY 4.0 terms. Dataset substitution is recorded and narrows continuity with the original proposal; it is not concealed. [R27][R28][R29]

<a id="source"></a>
## 2. Faithful reconstruction of the supplied idea

### 2.1 Source material and evidence boundaries

**S0** is the supplied file `Pasted text(20261003-123932).txt`, containing both the proposed idea and the assignment. The proposal occupies its lines 5–772; the current assignment begins at line 774. The old sandbox links inside S0 are historical text, not available artifacts. No access to its originating discussion is assumed.

Use the following evidence labels in reports: **Established/source-verified** for an inspected publication or official implementation; **Derived** for a mathematical consequence proved here; **Executed** for a check actually run in this investigation; **Hypothesis** for a predicted learning effect; **Estimate** for analytical resource accounting; **Proposed** for work the coding agent must implement; and **Unresolved** for an empirical or access uncertainty. Source-reported experiments must retain that label unless reproduced.

### 2.2 What S0 actually proposes

The motivating language is that one component should represent something about another component that its own state does not adequately capture. S0 operationalizes this in two asymmetric computational dependencies rather than two complete chatting models.

In the first, an upstream representation changes while a downstream consumer must remain usable. At a fixed boundary, retain training anchors, reference producer features, and low-rank factors of the consumer's input-specific softmax sensitivity. During adaptation, update the producer and a new head, freeze the old consumer, and penalize producer movement in those cached directions. The reference does not move. Anchor inputs are stored, so the method is not exemplar-free. The intended benefit is better stability–plasticity, or fewer recurring consumer evaluations, or both; these are different claims.

In the second, a cheap branch may query an expensive branch. The gate sees current cheap-branch information and historical delayed joint outcomes, but not the current optional branch before querying it. It learns or tracks actual fuser benefit rather than agreement. Random audits of unqueried examples are charged. The source itself recognizes that a scalar pair-gain state can be exactly a conventional composite-option utility tracker.

S0's source-reported digit pilot was mixed and did not demonstrate superiority to feature L2 or a global metric. Its reported streaming simulation used generated errors rather than trained neural branches. Those caveats remain intact.

### 2.3 Changes introduced in this specification

| Change | Why it is introduced | Relation to the original idea |
|---|---|---|
| Add cached linearized-logit distillation | Separates a benefit of cached derivatives from the particular Fisher quadratic approximation | Same information boundary, conventional comparator; not claimed as original validation |
| Add private producer and domain-specific adapter | Domain-known separate heads permit simpler ways of preserving the old model | Tests unrestricted practical value; single-producer analysis remains separately meaningful |
| Add trace-weighted global and teacher-class-bucket metrics | Distinguishes full input association from cheap conditional weighting | Stronger explanations of a possible gain, not replacement mechanisms |
| Explicit boundary-only versus recurring consumer access | A cache has no access advantage unless access is actually restricted or costly | Makes S0's intended constraint operational |
| License-aware image fallback and subject-disjoint HAR stress test | Keeps a locally executable path with documented data rights and a non-image transfer setting | HAR is a representative refinement, not evidence of natural drift |
| Precision-based replication and independent confirmation | Removes arbitrary improvement targets and favorable-seed selection | Strengthens rather than alters the central question |
| CPU reference factorization plus tested MPS training | Avoids assuming backend support from API names alone | Engineering route; not a scientific mechanism change |

### 2.4 Brain analogy and interpretation limits

No neuroscience premise is necessary for either selected mechanism. This investigation does not claim that biological neurons maintain these matrices, that backpropagation describes human learning, or that a compatibility cache creates selfhood, valuation, consciousness, or social understanding. S0 already treats the brain language as motivation rather than evidence. Biological plausibility is therefore **not tested**, while computational usefulness is tested directly.

The important distinction is not “ordinary networks have no relationships.” Neural composition already couples components, and backpropagation propagates local downstream objective sensitivity. What changes here is the **persistence and information budget** of a particular downstream constraint across a later adaptation interval. A restriction can be useful even if the whole system is representable as one larger network. Conversely, renaming a gradient or a utility statistic does not create a new mechanism.

<a id="landscape"></a>
## 3. Relevant technical landscape and prior art

### 3.1 Which limitations are actually implicated?

| Candidate limitation | Level at which it occurs | Existing response | Interpretable test here |
|---|---|---|---|
| An encoder update invalidates a fixed consumer | Training/update protocol and interface contract, not every neural architecture | Backward-compatible training, functional distillation, replay, frozen features | Old-consumer degradation versus new-head performance, with held-out anchors excluded from training |
| Isotropic or globally averaged preservation restricts useful changes | Choice of regularizer | Feature Fisher metrics, subspace methods, functional priors | Local versus global/conditional constraints at matched resources |
| Repeated consumer differentiation is costly or unavailable | Deployment/training-system access | Cached outputs, local surrogates, separate modules | End-to-end time and derivative access, including construction and diagnostics |
| Separate branch accuracies miss redundancy | Gate information and joint error structure | Adaptive deferral, composite-action tracking, complementary selection | Executable fusion utility under fixed marginals and causal feedback |
| LLMs are thought to lack “other-model awareness” | Often a vague interpretation rather than an established limitation | Attention, expert routing, external tools, memory and ensembles already create interactions | No general LLM capability claim is tested in this plan |

A contemporary documented LLM example, Qwen3, uses transformer attention, feed-forward computation, normalization, residual pathways, and both dense and mixture-of-experts variants; its training separates pretraining from post-training. [R16] These mechanisms do not automatically preserve a downstream interface after an arbitrary component replacement. Conversely, the mere existence of multiple experts does not make a compatibility cache novel. Low-rank parameter adaptation supplies another inexpensive update alternative, but parameter efficiency by itself is not an old-consumer retention guarantee. [R17] Serving-time attention-key/value caching is a different resource problem from retaining a training-time sensitivity cache. [R18]

No LLM is required for the first proof of concept. Large vocabularies would greatly increase output-Jacobian construction cost, sequence interfaces introduce masks and many positions, and a shared backbone may have many consumers. These are reasons to avoid extrapolating from ten-class models—not reasons to assume large-scale failure. A successful small experiment would justify a separate, explicitly budgeted extension, not a frontier-model claim.

### 3.2 Closest prior-art comparison

Reported evidence below describes what the cited authors evaluated; none of their benchmark results was independently reproduced here. Different task definitions, access, architectures, and memories make their accuracy numbers inappropriate as numerical baselines for this experiment.

| Source and date | Mechanism and overlap | Meaningful difference from this project | Reported evidence and limitations | Consequence |
|---|---|---|---|---|
| **EFC**, Magistri et al., ICLR 2024 [R01] | Local feature sensitivity is derived and then averaged into an empirical feature matrix; feature drift is penalized | This project keeps anchor-specific matrices for a frozen consumer and stores exemplars | Exemplar-free class-incremental experiments; prototype machinery and task protocol differ | Closest geometric predecessor. Call the global comparison an EFC-style regularizer, not a reproduction of full EFC |
| **EFC++**, preprint 2025; IJCV publication 28 September 2026 [R02][R31] | Extends the same feature-consolidation family with prototype rebalancing and a decoupled training treatment | Its central extra machinery is not per-anchor consumer caching | Incremental-classification evaluation, not this restricted-access interface setting | Updating the literature date matters, but does not establish a new local-cache principle |
| **Provable Continual Learning via Sketched Jacobian Approximations**, Heckel, AISTATS 2022 [R04] | Compresses old Jacobian information and constrains later updates | Sketch is in parameter space across observations; this cache acts on a named activation interface | Theory under specified model assumptions and empirical continual-learning tests; not a proof for this nonlinear adaptation objective | Jacobian sketching and quadratic behavior preservation are established |
| **Towards Backward-Compatible Representation Learning**, Shen et al., 2020 [R05] | Uses an old classifier to guide a new representation toward compatibility | Repeatedly available old consumer and retrieval-oriented compatibility differ from fixed local derivatives | Representation/retrieval experiments; not a byte-matched cached-metric study | Exact old-consumer supervision is a mandatory strong comparison |
| **Functional Regularisation for Continual Learning**, Titsias et al., 2019/2020 [R06] | Retains inducing inputs and a functional posterior | Bayesian last-layer/function-space construction, rather than per-anchor feature quadratic | Continual-learning experiments; inducing selection and posterior approximation add costs | Functional memory at selected inputs is not new |
| **FROMP**, Pan et al., NeurIPS 2020 [R07] | Memorable examples, network-Jacobian-based GP functional prior, covariance-weighted preservation | Couples remembered function values through a kernel; this proposal uses independent per-anchor interface metrics | MNIST and Split CIFAR experiments. The authors report that the kernel did not significantly improve over their L2 variant on Split CIFAR, unlike MNIST | Strong warning that elaborate geometry can lose to memory selection or simpler regularization; broader continual-learning claims require this family too |
| **Dark Experience Replay**, Buzzega et al., NeurIPS 2020 [R08] | Replays inputs with stored logits; DER++ adds label replay | Logits are sampled through training history rather than only at a fixed boundary | Continual-learning benchmarks; different update protocol | Include label and logit replay, but do not call boundary-only KD a faithful DER reproduction |
| **Gradient Projection Memory**, Saha et al., ICLR 2021 [R09] | SVD-based activation subspaces constrain parameter gradients | Parameter-update projection rather than consumer-specific local feature drift | Continual-learning benchmarks with its own subspace and memory rules | Subspace protection is established; it becomes mandatory for a broader algorithm-family claim |
| **Sobolev Training**, Czarnecki et al., 2017 [R10]; **Knowledge Transfer with Jacobian Matching**, Srinivas and Fleuret, ICML 2018 [R11] | Train using derivatives as well as outputs; connect local perturbations and distillation | Usually match student/teacher derivatives, not merely cache a frozen consumer's metric | Function approximation and transfer experiments under their settings | Derivative-informed transfer and local linearization are established vocabulary and mechanisms |
| **2ndMatch**, Zheng and Shlizerman, 2025 preprint / v2 March 2026 [R32] | Matches teacher/student `J^T J` directional quadratic forms via random projections | It trains student sensitivity against an accessible teacher, not a fixed consumer-metric penalty on feature displacement | Diffusion pruning/finetuning experiments; reported gains were not reproduced here | Further narrows broad sensitivity-matching novelty; “second-order” here denotes a Gram metric, not a Hessian |
| **Decoupled Neural Interfaces using Synthetic Gradients**, Jaderberg et al., ICML 2017 [R12] | Predict downstream training gradients to decouple updates | Learns a gradient predictor rather than retaining a fixed compatibility geometry | Decoupled training experiments | “Model what another module needs” is not by itself a new contribution |
| **L2D-SLDS**, Montreuil et al., current v3 in 2026 [R13] | Switching states, partial feedback, expert/internal residual modeling, shared residual structure | Proposed secondary branch is a small internal fuser with controlled marginal invariance | Nonstationary deferral evaluation under its own protocol | Do not reduce it to independent marginal trackers; an adapted composite option is a strong competitor |
| **FlexRouter**, 29 September 2026 [R14]; **MoCA**, 21 May 2026 [R15] | Complementary selection; respectively set coverage and summarizer-conditioned joint value | Coverage is not the same as realized executable-fuser loss | LLM ensemble/routing experiments; not single-laptop internal-branch proof | General complementarity-aware routing novelty is already occupied |

### 3.3 What was verified in the original EFC implementation

The official repository was inspected, including `continual_learning/utils/empirical_feature_matrix.py` and `continual_learning/ElasticFeatureConsolidation.py`. [R03] The former calculates the linear classifier's log-softmax feature derivatives, forms probability-weighted outer products, sums them over examples, and divides by the processed sample count. The latter constructs a quadratic feature penalty using `lambda * previous_efm + damping * I`, and contains prototype and replay-related machinery outside that penalty.

Observed source blob identifiers were `c47d4822e23e37bf8094076105fee4afffd66b07` for the matrix utility and `32e33e40aee243cf6d6019049a03a63a1aa80cf2` for the training class. These identify inspected file contents, **not a claimed repository commit**. The README advertises an MIT license; preserve its notices when reusing code. The first README fetch failed because the filename was `Readme.md`, not `README.md`; subsequent source access succeeded. No complete official training run or MPS port was executed.

Our implementation must not copy incidental behaviors such as dropping a remainder batch when computing the empirical average. It must also not count an unused higher-order Jacobian helper as a requirement of the closed-form matrix code. Reimplement the small, explicitly specified regularizer and independently test it; use a full upstream checkout only for a later full-method reproduction.

### 3.4 Narrowest defensible contribution

The complete combination—fixed per-anchor activation-space factors, a nonlinear fixed consumer, explicit derivative access, and byte- and time-accounted adaptation—was not found as an exact experimental match in the inspected sources. That observation **does not prove novelty**. The mathematical ingredients are conventional, and an earlier derivative-cache, local functional prior, or task-aware distillation implementation could eliminate even the combination claim.

The most defensible research output is therefore a reproducible answer to: **when does retaining the association between an input and a consumer's sensitivity improve the useful adaptation frontier, and when is that association a poor use of resources?** A negative answer, a cached-linearized alternative winning, or a private producer dominating would all be valuable results. None should be disguised as confirmation of a general relational architecture.

<a id="formal"></a>
## 4. Formalization and deductions

### 4.1 Problem and information contract

At an adaptation boundary, the source system is a producer `f_theta0` and a consumer `g_phi`. A new consumer `h_psi` must learn a target domain. The primary scientific comparison requires **one shared adapted producer** and keeps `g_phi` fixed. Old-domain inference uses `g_phi(f_theta(x))`; new-domain inference uses `h_psi(f_theta(x))`. Domain identity is supplied. This is not class-incremental recognition without task identity.

Let `N` be the number of retained anchors, `d` the feature dimension, `C` the old consumer's number of classes, and `b` the anchor minibatch size. Define

\[
z_i^0=f_{\theta_0}(x_i),\qquad a_i^0=g_\phi(z_i^0),\qquad
p_i=\operatorname{softmax}(a_i^0/T),\quad T>0.
\]

Two access regimes are reported separately:

**A-U, unrestricted consumer:** its frozen parameters and input gradients remain available during adaptation. Exact KL, label replay, cached methods, and inexpensive alternatives compete. Count all retained state and consumer evaluations.

**A-B, boundary derivatives for optimization, plus declared post-run validation:** at the boundary a provider permits consumer forward evaluations and first derivatives on the declared anchor set. Thereafter the adapting process cannot query the consumer during optimization. The provider retains it for a fixed, separately budgeted post-run development/final evaluation schedule. This is not zero consumer access for model selection; the allowance is declared equally for all methods. A checkpoint digest identifies the consumer version. This is not a generic black-box prediction API: output-only access does not supply Jacobians for free. Finite-difference or surrogate training would be different methods with additional queries and must not be silently substituted.

A-B restricts consumer access, not automatically all old labels or producer copies. State the old-data retention policy and deployment architecture independently. Exact KL is an A-U comparison or an A-B quality reference with greater access, not an admissible A-B competitor. Private producers remain admissible unless a real shared-producer constraint rules them out.

### 4.2 The local objective

For each anchor, compute

\[
J_i=\left.\partial g_\phi(z)/\partial z\right|_{z_i^0}\in\mathbb R^{C\times d},
\quad F_i=\operatorname{diag}(p_i)-p_ip_i^\top,
\quad G_i=J_i^\top F_iJ_i.
\]

For a sufficiently smooth consumer and a local displacement `delta`, differentiation of log-sum-exp gives

\[
T^2\operatorname{KL}\left(p_i\;\middle\|\;\operatorname{softmax}(g_\phi(z_i^0+\delta)/T)\right)
=\tfrac12\delta^\top G_i\delta+O(\|\delta\|^3). \tag{1}
\]

The first-order term vanishes at the reference because the two distributions agree. The softmax Hessian contributes `F/T^2`, which cancels the exterior `T^2`. Consumer second derivatives multiply a zero first derivative at the reference; they still matter for higher-order behavior. This is a local expansion, not a guarantee for the full adaptation trajectory. For a piecewise-linear consumer, apply the expansion within a fixed activation region and inspect boundary crossings separately.

Use the numerically convenient factor

\[
A_i[c,:]=\sqrt{p_{ic}}\left(J_i[c,:]-\sum_k p_{ik}J_i[k,:]\right),
\qquad A_i^\top A_i=G_i.
\]

If `A_i = U_i diag(sigma_i) V_i^T`, retain

\[
K_i=\operatorname{diag}(\sigma_{i,1:r})V_{i,1:r}^\top\in\mathbb R^{r\times d}.
\]

The exact softmax rank is at most `min(d,C-1)`; “full local” means all these rows, not a stored dense `d by d` matrix. Zero singular rows may remain as padded zeros. Use one **common** scale

\[
s=N^{-1}\sum_i\operatorname{tr}(G_i).
\]

The main adaptation loss is

\[
L=\frac1{b_n}\sum_j\operatorname{CE}(h_\psi(f_\theta(x_j^{new})),y_j^{new})
 +\frac{\lambda}{2bs}\sum_{i\in B}\|K_i\delta_i\|_2^2
 +\frac{\eta}{2bd}\sum_{i\in B}\|\delta_i\|_2^2,\qquad
\delta_i=f_\theta(x_i)-z_i^0. \tag{2}
\]

This specification stores **raw** factors and `s`; normalization happens exactly once in the loss. It differs from S0's illustrative helper, whose factors were already normalized. Do not combine the conventions. Keep `eta` independent of `lambda`. The fixed initial value `eta=0.01` is a development starting point, not evidence of optimality. The main geometric comparison first holds it common; a subsequent equally budgeted damping sensitivity comparison permits `eta=0` and other values.

### 4.3 Affine-consumer boundary case

**Derived.** For `g(z)=Wz+b` and finite softmax logits, every probability is positive and

\[
\ker G_i=\{\delta:W\delta\in\operatorname{span}(\mathbf1)\}.
\]

Thus input-dependent probabilities can change the strength and relative weighting of the metric, but not its exact nullspace. A rotating-nullspace explanation cannot be established with an affine consumer alone. A nonlinear consumer can change `J_i` and its sensitive subspace. This motivates the nonlinear primary consumer and an affine negative/boundary control. It does **not** prove that local probability weighting is useless for affine consumers.

### 4.4 A strong cached comparator: linearized-logit distillation

The same boundary derivatives permit a less aggressive approximation:

\[
q_i^{lin}(\delta)=\operatorname{softmax}\left(\log p_i+J_i\delta/T\right),\qquad
L_{lin}=\frac{T^2}{bs}\sum_{i\in B}\operatorname{KL}(p_i\|q_i^{lin}(\delta_i)). \tag{3}
\]

This preserves the softmax nonlinearity while linearizing only the consumer logits. It is **exact for an affine consumer at arbitrary displacement**, because adding a shared logit constant does not change softmax. For a nonlinear consumer it remains local and can still fail, but separates softmax-curvature error from consumer-linearization error.

Store class contrasts `D_i[c,:]=J_i[c,:]-J_i[C-1,:]` for `c=0,...,C-2`. At runtime append a zero last row to `D_i delta/T` and add `logp_i`. This eliminates one redundant Jacobian row. For a known affine consumer, deduplicate the identical `D` across anchors. Do not force this comparator to waste memory to manufacture parity. For the main nonlinear consumer, its full contrast cache has nine rows per anchor versus the proposed rank-eight factor's eight; compare both actual bytes and full-rank methods. Equation (3) is a derived conventional comparator, not a claim that this investigation invented linearized distillation.

### 4.5 Truncation, locality, and coordinate dependence

For fixed `delta`, the error from truncating an exact factor obeys

\[
0\le\tfrac1{2s}\delta^\top(G_i-K_i^\top K_i)\delta
\le \frac{\sigma_{i,r+1}^2}{2s}\|\delta\|^2. \tag{4}
\]

A large fraction of retained average spectral energy does not ensure that the actual update direction avoids discarded sensitivity. Log the observed displacement-weighted truncation loss as well as spectral energy. Rank is therefore selected on development performance and bytes, not on an energy percentage alone.

For a consumer with logits `[z^2,0]` at `z=0`, the Jacobian and quadratic penalty are zero, but moving to `z=1` changes its probability distribution. The exact KL is approximately `0.1201145` at `T=1`; this failure was independently checked. Damping can discourage movement but supplies no Taylor-remainder bound.

Full quadratic penalties are invariant to consistent interface permutations. General invertible coordinate changes also preserve the exact quadratic when features and metrics are transformed covariantly. However, Euclidean low-rank truncation and random-orientation controls are not generally invariant to arbitrary rescaling. An apparent orientation effect could partly reflect representation conditioning. The primary control therefore receives its own fair coefficient search, and a positive result triggers the optional whitened-coordinate control in Section 6.6.

Do not refresh the reference toward the current producer during the minimum experiment. Repeatedly small local drift around moving references can accumulate into large drift from the original system. A later refresh method must retain the original target distribution, pay its new consumer access, and be labeled a separate method.

### 4.6 Why the small positive construction is not enough

For two anchors with `G1=diag(1,0)` and `G2=diag(0,1)`, a global average constrains both coordinates at both anchors. Suppose each desired displacement is `(1,1)` and new loss is half squared distance to that displacement, averaged over anchors. Local displacements `(0.1,1)` and `(1,0.1)` yield new loss `0.405`; globally regularized displacements `(0.1,0.1)` at both anchors yield `0.81`. Both have old quadratic drift `0.005`. These are analytic solutions along their respective regularization paths, and the arithmetic was checked.

This establishes a possible distinction in a quadratic problem with independently adjustable anchor features. A shared neural producer couples examples, an actual consumer can be nonlinear, and held-out examples are not independently controlled. The neural tests must establish that these distinctions survive. Report this construction only as a known-positive evaluation check.

<a id="hypotheses"></a>
## 5. Falsifiable hypotheses and competing explanations

| ID | Hypothesis and conditions | Predicted observation | Null/alternative explanation | What would weaken it |
|---|---|---|---|---|
| H1 | The cache faithfully approximates a smooth frozen consumer locally | Full-factor quadratic agrees with exact KL as perturbations shrink; gradients reach only the current producer | Implementation error, normalization error, numerical cancellation | Persistent nonconvergence away from zero-KL degeneracies |
| H2 | Input association permits useful shared-producer adaptation when consumer sensitivities vary and producer change is needed | Better target accuracy at the declared old-retention operating point than L2 and global geometry | Coefficient tuning, scalar sensitivity, easy new-head adaptation, cache coverage | Coarse conditional/scalar metrics, L2, or frozen features match it within adequately resolved uncertainty |
| H3 | Correct sensitive directions explain the improvement | Properly tuned retrained orientation controls lose useful frontier quality, without gross optimization failure | Generic preconditioning or feature-scale effects | Random or whitened-random directions perform equally well |
| H4 | The local cache is a good use of storage/information | Advantage remains on a quality–bytes frontier including more-anchor L2/replay and private producers where admissible | More stored information, old labels, or extra boundary work explains it | Byte-matched simpler methods dominate or private producers meet the deployment budget more cheaply |
| H5 | Caching pays for its construction under a specified access/workload regime | Matched-quality total runtime or permitted-access feasibility improves, including construction and evaluation | Fewer counted calls but more dispatch, transfers, or factor work | Break-even horizon exceeds the intended update workload; consumer is too cheap |
| H6 | Useful behavior transfers beyond an invertible image transformation | Subject-disjoint HAR stress-test improvement with the same causal controls | Rotation-specific adaptation or label/data reconstruction shortcut | No effect on the representative task despite reproducing sensitivity and plasticity conditions |
| H7 | A learned relation state is needed for the optional fusion branch | It improves over same-information composite EMA and generic history models | Merely recognizing joint utility, already captured by scalar tracking | Scalar composite utility matches it; pair and composite EMA are identical as expected |

The mechanism may depend on interface width, consumer nonlinearity, task conflict, cache coverage, update length, and actual need to change the producer. A tiny task that eliminates all of these conditions is an invalid universal rejection test. However, adding complexity merely until a favorable outcome appears is also invalid. Before each expansion, name the missing condition, measure it on development data, and test that condition directly. Do not select confirmation tasks or checkpoints after seeing their held-out outcomes.

<a id="algorithms"></a>
## 6. Reference algorithms, tensor contracts, and state transitions

### 6.1 Cache object

Implement an immutable logical `CompatibilityCache` with the following tensor and metadata contract. Serialization may use a tensor dictionary plus JSON; do not unpickle untrusted third-party objects.

| Field | Shape/type | Meaning |
|---|---|---|
| `anchor_ids` | `[N]`, stable integer IDs | Original data identity, not current loader row positions |
| `anchor_inputs` | Image uint8 `[N,28,28]` or HAR float32 `[N,9,128]` | Original-domain inputs before model preprocessing; choose and hash a canonical representation |
| `z0` | `[N,d]`, float32 at training time | Fixed producer reference computed in its frozen evaluation state |
| `K` | `[N,r,d]`, float32 | Raw sensitivity factors; absent for nonlocal methods |
| `logp0` | `[N,C]`, float32 | Stable reference log probabilities; store when the method needs them, not automatically for all byte comparisons |
| `D` | `[N,C-1,d]`, float32, or one shared affine matrix | Class-contrast derivative for the linearized comparator; not stored by local-only methods |
| `s` | scalar, float64 metadata and device-cast scalar | Full, untruncated mean trace, never recomputed after adaptation |
| `trace_i`, `tail_eigenvalue_i` | optional `[N]` | Diagnostics or scalar control; count them if retained by the executing method |
| Metadata | JSON | Schema version, source and consumer hashes, named interface, preprocessing/split hashes, `T`, rank, dtype, code revision, constructor seed, construction time/access counts |

The evaluation archive can contain more diagnostics than the adapting method is allowed to retain. Separate `audit/` from `method_state/`; count both for total research disk use, but do not grant the algorithm information from the audit directory. A byte-budget claim must state which of algorithmic retained bytes, peak construction memory, total disk, and deployment weights it matches.

### 6.2 Build procedure

1. Load one source checkpoint on CPU and set producer and consumer to evaluation mode. Freeze their parameters. Cache construction is outside `torch.inference_mode()` because input derivatives are needed. Reference features may be obtained in `no_grad` and detached.
2. Validate anchor identity, shape, label policy, preprocessing, and no split overlap. Compute the producer reference in its canonical FP32 source representation, then convert those reference features to CPU float64 for consumer differentiation. Record this convention. Use a float64 copy of the consumer for the reference Jacobian calculation.
3. For each anchor, enable gradients only on a leaf feature vector `[d]`. Compute logits and `log_softmax(logits/T)`, then a first-order Jacobian `[C,d]`. A loop over class logits with `autograd.grad` or `autograd.functional.jacobian(create_graph=False)` is the trusted initial implementation. Vectorization is optional after parity tests.
4. Form `A`, compute its SVD on CPU float64, and keep up to `min(r,C-1,d)` rows. Compute the full trace from `A.square().sum()`, not from truncated factors. Retain a nonnegative discarded-eigenvalue diagnostic in the audit archive.
5. Check all values are finite and factors reconstruct the expected full matrix at full rank. If `s` is zero or numerically degenerate, mark the consumer/cache pair as invalid for trace-normalized comparisons. Do not silently divide by an arbitrary tiny constant and train. Investigate constant logits, saturation, preprocessing, or a truly zero derivative. An unnormalized comparison is a separately named repair with reselected coefficients.
6. Detach every cache tensor; cast training payloads to FP32; write atomically with checksums. Verify FP32-cast penalty accuracy against float64 on held-out random displacements. Store original reference and consumer hashes. No optimizer state belongs in this cache. Backend roundoff can make the CPU-canonical reference differ slightly from an MPS evaluation at the same checkpoint; measure and report this initial discrepancy in E01 rather than assuming bitwise equality across devices.

A consumer with dropout, batch-dependent normalization, or stateful behavior must be frozen in a well-defined evaluation state. The default consumers use neither dropout nor batch normalization. A changed checkpoint or preprocessing digest invalidates the cache even if dimensions still match.

### 6.3 Runtime penalties

For `z:[b,d]`, IDs indexing the cache, and `delta=z-z0[ids]`, use:

```python
# Illustrative core; the project wrapper must perform validation and accounting.
u = torch.einsum('brd,bd->br', cache.K[ids], delta)
local = 0.5 * u.square().sum(dim=1).mean() / cache.s
isotropic = 0.5 * delta.square().mean()  # division by b*d
loss = new_cross_entropy + lambda_reg * local + eta * isotropic
```

The global matrix is `Gbar = mean_i G_i`, calculated on exactly the same anchors and normalized by the same `s`. Its loss is `mean_i delta_i^T Gbar delta_i/(2s)`. For L2, use `mean(delta^2)/2` with its independently tuned coefficient. Do not equate numerical lambda values across methods with equal regularization strength.

For exact KL, run the current anchor features through the frozen consumer **without detaching those features**. The standalone baseline computes raw `T*T * mean(sum(p0*(logp0-logq), classes))` with its own coefficient. Divide by the shared `s` only in the explicitly normalized numerical approximation comparison; do not require the production exact baseline to construct a Jacobian to obtain it. Consumer parameters remain frozen but input gradients must reach the producer. For Equation (3), replace current consumer logits by the cached contrast-linearized construction; no consumer call occurs during adaptation.

For source label replay, use `CE(g_phi(f_theta(anchor)), source_label)` with a separately tuned replay weight. The old consumer still has frozen parameters, so this is A-U unless the provider supplies an admissible executable consumer. Cached label data alone does not eliminate consumer computation. A target-head-only label replay is a different baseline and does not directly supervise the old interface.

### 6.4 Adaptation and state transitions

The allowed state sequence is `source_train -> source_frozen -> cache_built -> adapting -> selected -> confirmed_or_rejected`. A checkpoint, dataset, or preprocessing change creates a new state lineage rather than modifying an existing completed run.

Initialize the new consumer as a deep copy of the old consumer for the primary same-label tasks. Reset its optimizer state. Pair the initial producer, new-consumer initialization, new-example batches, and anchor draws across methods within each seed. Use independent random generators for source training, adaptation data, anchor selection, and orientation construction so an ablation does not accidentally perturb another stream.

In each adaptation step, compute target CE on `b_n` examples, then the appropriate anchor objective on `b` examples, and update only the producer and new consumer. Under A-U the old consumer is in evaluation mode with `requires_grad=False`; assert its parameter gradients are absent and its parameters/buffers retain their hashes. Under A-B the optimizer-facing code receives only permitted cache state and has no consumer callback; the provider evaluates only completed, predeclared checkpoints. Any per-step exact-drift diagnostics belong to A-U or a separately charged audit run, not hidden A-B training feedback. Use `zero_grad(set_to_none=True)`. Do not retain graphs across steps. Keep the producer in the same deterministic architectural mode across methods.

When both compatibility coefficients are zero, skip anchor computation altogether and recover the unconstrained parent update, including random-generator state. A mere zero multiplied by a computed loss is insufficient for a clean systems baseline. When only `lambda=0` but `eta>0`, the method is damping-only, not unconstrained.

At inference, the original method has no additional router, cache lookup, or regularization computation. It uses the adapted producer and the chosen domain's head. The training cache may be discarded after the final checkpoint unless future updates require it. Report this separately from private-producer deployment storage.

### 6.5 Cheap conditional controls

Implement these before contemplating a learned metric predictor:

- **Trace-weighted global:** `Ghat_i=(trace(G_i)/s)*Gbar`. It has the anchor's total sensitivity but global orientation. Store only the scalar and global matrix.
- **Teacher-class-bucket global:** let `k_i=argmax p_i` at the boundary; average full matrices within each teacher-predicted class and use that bucket at the anchor. Use the overall `s`. Empty buckets fall back to `Gbar`. This requires no extra source labels. Store bucket IDs and matrices and count them.
- **Diagonal global:** preserve only `diag(Gbar)` with common normalization. This tests whether generic coordinate scaling explains the benefit.

The class-bucket method is an ordinary coarse conditional comparator, not claimed as a new contribution. Its cost can exceed the global matrix but remain lower than per-anchor factors. Do not force every possible baseline into every early debugging run; all of these become part of attribution before any final input-specific-geometry claim.

### 6.6 Spectrum-preserving orientation control

For each anchor draw a seeded independent Haar-distributed orthogonal `Q_i` using Gaussian QR with diagonal-sign correction, then set `K_i_control=K_i Q_i`. Preserve the factor's singular spectrum, rank, trace, anchor data, reference features, damping, and optimizer schedule. **Retrain** with this cache; a test-time corruption of a trained method is not the relevant control. Tune its coefficient on the same development budget. Report gradient norms, nonfinite rates, and convergence to distinguish misorientation from a broken optimizer.

A positive attribution result triggers an additional conditioning test when feature covariance is markedly anisotropic. Estimate covariance on source-training reference features only, regularize its eigenvalues by a recorded development-selected floor, and whiten the coordinates. Apply the same random-orientation construction to the metric in those coordinates, then transform it back. This preserves the spectrum in whitened coordinates, not necessarily in original coordinates; label it accordingly. Freeze the covariance/floor before confirmation. A conclusion about “correct directions” should acknowledge any sensitivity to coordinate choice.

<a id="data"></a>
## 7. Data, preprocessing, and model configurations

### 7.1 Dataset provenance and access

No pretrained or open-weight model is required. All networks are trained from scratch, so pretrained-model contamination and remote model licensing are not hidden dependencies. Download public data during setup; all training and evaluation then run locally without paid APIs or cloud accelerators.

| Dataset | Verified source and rights/access status | Role and limits |
|---|---|---|
| MNIST | TorchVision's documented MNIST loader and official source/mirror definitions [R27]. Four compressed IDX resources have published checksums in that implementation. No account is required by that loader. A primary-source dataset license grant was not established here | Preferred continuity test only after the executing researcher confirms permissible use; do not infer rights from the loader's license |
| Fashion-MNIST | Original Zalando Research repository advertises an MIT license and standard training/test resources [R29] | Fully specified image fallback using the same 28×28, ten-class protocol; results are called Fashion-MNIST, not MNIST replication |
| UCI HAR, dataset 240 | Official UCI download and DOI `10.24432/C54S4K`; stated CC BY 4.0 [R28] | Representative local sensor stress test; give attribution, record the exact archive hash, and distinguish supplied windows from raw continuous signals |

Pin the downloaded bytes with SHA-256, recording their original URL, download time, upstream version/tag where available, and a copy of the terms/attribution. A version number is not invented when the dataset has none. Archive extraction must reject path traversal and verify expected dimensions and finite values. Stop a dataset branch if rights or integrity cannot be established; use the declared alternative rather than guessing.

The MNIST compressed-resource MD5 values verified in the TorchVision source are:

| File | Upstream MD5 |
|---|---|
| `train-images-idx3-ubyte.gz` | `f68b3c2dcbeaaa9fbdd348bbdeb94873` |
| `train-labels-idx1-ubyte.gz` | `d53e105ee54ea40749a09fcbcd1e9432` |
| `t10k-images-idx3-ubyte.gz` | `9fb629c4189551a2d022fa330f9573f3` |
| `t10k-labels-idx1-ubyte.gz` | `ec29112dd5afa0611ce80d1b7f02629c` |

Use these only for MNIST; the fallback has different files/checksums. MD5 here is upstream integrity metadata, not the project's security or provenance hash. The project additionally records SHA-256.

### 7.2 Image split and transformation protocol

Start with the official training partition. Create nominal 50,000 training, 5,000 development, and 5,000 diagnostic examples **before any transformation**. Use stable raw IDs and a split generator seeded with `20261003`. Perform exact-content deduplication by hashing the original uint8 image. Group duplicate training images before allocation so no identical root image crosses train/development/diagnostic boundaries; keep conflicting labels as an explicit data-quality issue rather than silently relabeling. Match the target proportions within each class as closely as duplicate groups permit and record actual counts.

The official test partition remains protected. A mechanical hash-only audit may identify exact train/test overlaps before model fitting; remove any overlapping training group, leave the official test partition intact, and record the removals. This audit may not calculate model scores, choose features, or select transformations from test images. The split manifest specifies this rule in advance. Also report performance on test groups not duplicated within the test set as a sensitivity analysis; do not silently replace the official test metric.

Old inputs are canonical clean images divided by 255 and flattened to `[784]`. New inputs are `torch.rot90(image, k=1, dims=(-2,-1))`, then divided by 255 and flattened. The label stays the original class. No interpolation, learned alignment, cropping, or augmentation is used in the first comparison. Parent IDs remain identical across clean and transformed versions, and both versions always remain in the same split. It is legitimate that the two training domains share parent training IDs; it is not legitimate for one version to be in training and another in validation.

Select anchors by a deterministic seeded permutation of **training** parent IDs, not by observed test loss. Use the same prefix ordering for every anchor-budget condition. Main anchor selection is uniform without replacement, avoiding an undisclosed label-based selection benefit. If class-balanced or uncertainty-based selection is later explored, make it an explicit common selection factor, account for its scans, and confirm it separately.

### 7.3 The inverse-transformation baseline is mandatory

**Derived fairness issue:** the known 90° rotation is invertible. A learner holding the rotated labeled training set can recover clean labeled training examples by rotating back. The task therefore cannot establish a realistic scarcity of old-domain examples by pretending only the retained anchors exist.

Include two strong alternatives:

**Inverse-label replay:** reconstruct clean examples from each target training batch and apply old-consumer label supervision under A-U. No extra replay-image buffer is needed; count the inverse transform and consumer computation.

**Inverse-feature teacher:** retain the original producer, reconstruct clean target examples, and match the current producer's clean features to the original producer's features. This does not require recurring old-consumer access and is admissible in A-B when retaining the producer is allowed. Count the teacher producer's weights and forwards. This provides much broader feature-preservation coverage than a small anchor cache.

The primary equal-anchor comparison is still a valid **controlled regularizer experiment**, but cannot by itself support a restricted-old-data practical claim. Do not suppress these alternatives because they make the toy task less favorable. Their potential dominance is a useful result and a motivation for the non-invertible sensor stress test.

### 7.4 Image architecture and initial training configuration

Use the same model family for MNIST or Fashion-MNIST:

| Component | Definition | Parameter count, including biases |
|---|---|---:|
| Producer | `Linear(784,256), ReLU(), Linear(256,128)` | 233,856 |
| Old consumer | `Linear(128,512), Tanh(), Linear(512,512), Tanh(), Linear(512,10)` | 333,834 |
| New consumer | Independent deep copy of old consumer | 333,834 |
| All participating modules | Producer plus both consumers | 901,524 |
| Adaptation trainable | Producer plus new consumer | 567,690 |

Starting configuration: Adam with learning rate `1e-3`, default betas `(0.9,0.999)`, epsilon `1e-8`, no weight decay; source batch 64; adaptation target batch 64 plus anchor batch 32; 256 anchors; `T=2`; local rank eight; full rank at most nine; `eta=0.01`. Source 2,000 steps and adaptation 1,500 steps are **initial calibration horizons inherited from S0**, not required convergence limits. Log at fixed 100-step intervals during development. Extend source or adaptation duration when development learning curves show material unresolved improvement, using the common convergence rule in Section 9.

The affine-consumer boundary control replaces the old/new consumers with `Linear(128,10)` and retrains the source. It is a different architecture experiment, not a resource-matched nonlinear competitor. Its purpose is to test the prediction about common nullspaces and exact linearized logits.

### 7.5 Representative sensor task: UCI HAR

UCI HAR supplies windows of smartphone inertial measurements for six activities and a subject-based training/test partition. Its windows overlap by 50%, so random window splitting is inappropriate for claims about unseen people. [R28]

Use the nine supplied inertial channels in this fixed order: `body_acc_x,y,z`, `body_gyro_x,y,z`, `total_acc_x,y,z`, each with 128 readings. Load the corresponding `train/Inertial Signals/*_train.txt` or `test/Inertial Signals/*_test.txt`, `subject_train.txt`/`subject_test.txt`, and `y_train.txt`/`y_test.txt` from the original archive. Verify these names against the extracted README and manifest before proceeding; reject an updated but structurally different dataset rather than guessing its ordering. Labels are mapped from 1–6 to 0–5.

Within the official training subjects, hash each distinct subject ID with the fixed split seed, sort by hash, and allocate the first 13 subjects to training, next four to development, and remaining four to diagnostics. The expected original counts are 21 training subjects and nine official test subjects; verify them rather than silently accepting another archive. Keep all windows from a subject together. Subject IDs are grouping metadata, not model inputs.

Compute channel means and standard deviations using only training-subject windows. Apply the same fixed normalization to every split and domain. A zero-variance channel uses a documented floor and is flagged; never estimate normalization on validation/test subjects. The supplied signals have already undergone dataset preprocessing; do not describe them as newly collected raw sensor streams.

Define the target condition explicitly as a **simulated gyroscope fault**, not natural drift: after normalization, set gyro-z to zero and add zero-mean Gaussian noise with standard deviation 0.2 in normalized units to gyro-x and gyro-y. Generate a fixed corruption realization for each `(split, parent_window_id, corruption_seed)` using a cryptographic-hash-derived seed; reuse it across methods. This is non-invertible with respect to the missing clean gyro-z. The chosen severity is an initial stress condition, not a measured device failure distribution. Only development data may justify a revised severity; freeze it before confirmation. Include no-fault and missing-gyro-z-only diagnostic conditions.

The evaluator retains original clean windows for old-domain scoring. After the source boundary, each algorithm's training interface exposes only the target training windows and its declared retained source state. The original archive remains in the audit store, not an undeclared training input. This is an experimental information constraint, not a claim that cryptographic access control is essential to the algorithm. Full clean replay is reported as a higher-data-access comparison, with its storage/availability labeled.

HAR producer: `Conv1d(9,32,5,padding=2), ReLU, MaxPool1d(2), Conv1d(32,64,5,padding=2), ReLU, AdaptiveAvgPool1d(8), Flatten, Linear(512,128)`. Each consumer is `Linear(128,128), Tanh, Linear(128,128), Tanh, Linear(128,6)`. Expected counts are 77,440 producer parameters and 33,798 per consumer, 145,036 total; assert counts in code. Start with 256 anchors, rank four/full rank five, target/anchor batches 64/32, `T=2`, and the same optimizer conventions. Calibrate duration from development curves rather than importing image step counts as convergence evidence.

Primary HAR accuracy is **subject-macro accuracy**: calculate each subject's fraction correct, then average subjects equally. Also report window-weighted accuracy and cross-entropy, but do not choose the better-looking metric after testing. The subject-macro primary metric prevents large subjects from dominating. The initial nine held-out people bound external generalization precision; additional optimization seeds do not create additional people.

A positive HAR stress-test result establishes behavior on a controlled sensor corruption with unseen subjects. It does not establish real hardware-replacement validity, natural correlation-only drift, or safety-critical activity recognition performance.

<a id="baselines"></a>
## 8. Baselines, resource regimes, and fair tuning

### 8.1 Mandatory comparison matrix

“Best baseline” means the strongest **reproducible local alternative within the stated information and architecture regime**, selected by the same development rule—not a published score on another benchmark.

| ID | Method | Key resources/information | Required role |
|---|---|---|---|
| B00 | Unconstrained adaptation | New data; shared producer and new head | Establish the parent and whether forgetting occurs |
| B01 | Frozen producer, new head only | Original producer; no cache | Tests whether producer adaptation is needed |
| B02 | Cached feature L2 | Anchor inputs and reference features | Strong simple feature-preservation baseline |
| B03 | Global full feature metric | Same anchors/references plus one full global matrix | Closest EFC-style geometric comparison |
| B04 | Trace-weighted global | B03 plus scalar per anchor | Tests scalar conditionality rather than orientation |
| B05 | Teacher-class-bucket global | References, bucket IDs and bucket matrices | Tests cheap coarse conditionality |
| B06 | Global diagonal metric | References and one diagonal vector | Tests coordinate scaling/preconditioning |
| M01 | Full local factors | Same anchors plus full `C-1` factors | Isolates local association without truncation |
| M02 | Rank-eight local image / rank-four HAR | Compressed local factors | Main compact-cache candidate |
| C01 | Retrained orientation control | Same factor spectra, independent rotated directions | Attribution, not merely a weak predictive baseline |
| B07 | Cached linearized-logit KL | References, log probabilities, contrast Jacobians | Strong same-boundary-information comparator |
| B08 | Exact output KL | Stored target probabilities; recurring frozen-consumer input gradients | Strong direct-objective A-U comparator and A-B quality reference |
| B09 | Source label replay | Retained source labels/inputs and recurring old consumer | Tests ordinary rehearsal under its label/access budget |
| B10 | Exact boundary-logit plus label replay | Retained input, logits and labels; separately tuned two terms | Strong rehearsal combination; not mislabeled as full DER |
| B11 | Private source producer plus adapted producer | Another source producer; domain-known dispatch | Unrestricted engineering envelope with exact source retention |
| B12 | Frozen producer plus target-only residual adapter | Small target-specific trainable adapter and new head | Parameter/storage-efficient alternative where domain-specific modules are allowed |
| B13 | Inverse-label replay on image task | Reconstructed clean target examples; no image replay cache; recurring old consumer | Exposes invertible-transformation shortcut |
| B14 | Inverse-feature teacher on image task | Original producer plus reconstructed clean target examples | Strong A-B alternative with full training-distribution coverage |

Implement B12 as `z_target=z+V ReLU(Uz)`, with `U:128->k`, `V:k->128`, `k=16` initially, zero-initialize the final projection so the adapter begins as identity. Apply it only on the target path. Old inference bypasses it. Tune its width within the same available storage envelope, and count both parameters and optimizer state. It is deliberately not an exact parameter match to M02; it is a quality–cost alternative.

B11 retains the original source producer and trains a copy with the same target consumer and target optimization schedule as B00. Source inference uses the untouched producer and old consumer, so retention should be bitwise unchanged within one deterministic backend. Do not charge it two producer forwards per example when the domain-known dispatcher uses only one. Conversely, an application needing both domains' outputs for each input must measure both executions for every method.

### 8.2 Do not burden exact distillation with unnecessary derivatives

Equation (1) comparisons use a common `s` to test numerical approximation. A standalone exact-KL baseline does **not** need to construct a Jacobian or know `s`. Implement its production objective as raw `T^2 * mean(KL)` with an independently tuned coefficient. An equivalent normalized coefficient can be calculated for analysis, but do not charge exact KD for building the proposal's cache. This distinction is essential in the systems comparison. The same principle applies to any simple baseline that does not need teacher sensitivities.

### 8.3 Three comparison regimes, not one fictitious universal match

**R-equal-inputs:** same anchors, source checkpoint, batches, model shape, update horizon, and permitted data. This isolates the regularizer. It does not equalize stored information or cost.

**R-equal-bytes:** set a method-state byte budget equal to M02's actual serialized required payload at the selected architecture and anchor count. Give simpler methods as many anchors or adapter parameters as fit. Use one nested anchor ordering and a deterministic largest-fitting-prefix rule. Include inputs, labels/probabilities when needed, reference features, factors/matrices, IDs, and necessary metadata. Report both uncompressed tensor payload and serialized bytes; compression ratios are not silently mixed. A full dataset already legitimately present for target training is common data storage, so its inverse transformation is not charged as a new replay cache.

**R-deployment-envelope:** compare quality, peak adaptation memory, retained training state, deployment weights, and measured runtime without excluding different architectures. This includes B11/B12 and applicable inverse-data alternatives. A method that only wins under R-equal-inputs has not established a storage or engineering advantage.

For illustration only, a 256-anchor image cache with uint8 inputs, FP32 reference features, and rank-eight FP32 factors has `256*(784+512+4096)=1,380,352` tensor bytes before IDs and metadata. That simplified budget fits 1,065 L2 anchors or 1,742 input-plus-int64-label replay examples. These are arithmetic illustrations; the executed largest-prefix calculation uses the full actual schema. It also counts the frozen consumer required for label replay where the comparison makes that storage noncommon.

### 8.4 Scope of the prior-art reproduction claim

B03 reproduces the **specified global penalty on a common anchor interface**, not EFC's prototype classification system. B10 is a fixed-boundary rehearsal combination, not DER's trajectory-based buffer policy. The first program is not an exhaustive continual-learning benchmark and cannot claim to outperform all of EFC++, FROMP, GPM, or modern continual-learning algorithms.

If the claim is expanded from interface adaptation to continual-learning algorithm superiority, first port the complete closest methods using their official task assumptions, optimizer/state rules, and memory accounting; validate those ports on their published miniature examples. FROMP's cross-example functional prior and GPM's parameter-gradient constraints cannot be replaced by weak independent approximations while retaining their names. This additional work is a prerequisite for the expanded claim, not a hidden prerequisite for the narrow comparison already specified.

### 8.5 Tuning and convergence fairness

Source training is shared per seed. Tune it using source development data without looking at adaptation test outcomes. For adaptation, begin with `lambda in {0.1,1,10,100}` for normalized metrics/L2 and the equivalent broad raw-coefficient exploration for exact KD/replay. Expand a method's grid geometrically whenever its best retention-relevant operating point is at an endpoint or the retention threshold is not bracketed. Apply the same expansion rule to every tunable method. Do not abandon a baseline because its first coefficient scale was wrong.

Calibrate learning rate around `1e-3` using the same multiplicative neighborhood for every method; first test `1e-3/sqrt(10)`, `1e-3`, and `1e-3*sqrt(10)` where stable. A method with more hyperparameters receives an explicitly recorded search budget and reports it; compare both an equal-candidate-evaluation search and an equal-development-time view if search cost could explain a practical advantage. Do not spend unlimited optimization on the proposal while leaving baselines at defaults.

Starting development uses three paired seeds to detect gross failures and choose which intervals need refinement. This is not a sufficient confirmation sample. A convergence extension is triggered when the difference between successive equal-length development training blocks is larger than its estimated paired development uncertainty and remains favorable, or when the best valid checkpoint is still at the final block boundary. Extend the affected method and the relevant leading competitors; re-evaluate duration selection under the same rule. Stop extending only when the development decision is stable at the required precision or the measured local time/resource budget makes the result unresolved. Learning rate changes, clipping, or longer training are method configurations, not invisible rescues.

Main training uses no gradient clipping unless the source or adaptation calibration shows nonfinite or unstable updates. Any adopted clipping threshold is then common or equally tuned, recorded, and frozen. A constant gradient-scale mismatch is solved by fair coefficient tuning, not by concluding that one geometry is inherently worse.

<a id="statistics"></a>
## 9. Evaluation, statistical analysis, and decision rules

### 9.1 Outcomes and estimands

For seed `j`, let `A0_j` be the original source checkpoint's old-domain accuracy and `Aold_mj`, `Anew_mj` the adapted method's accuracies. Define old loss `Dold_mj=100*(A0_j-Aold_mj)` and target accuracy `Y_mj=100*Anew_mj`, both in **percentage points**. Accuracy fractions must not be confused with percentage points in code or plots. For HAR, calculate the specified subject-macro accuracy before these differences.

The primary operating point is the configuration selected on development data to maximize target accuracy while respecting the inherited retention threshold `tau=1.0` percentage point. Freeze this configuration, including the selection algorithm, before the protected test evaluation. The confirmatory scientific estimand is the **mean paired target-accuracy difference across independent training seeds at this selected operating point, together with mean old-domain loss**. A mean retention result does not guarantee retention for every seed; report each seed and the fraction exceeding `tau`. This explicit mean-estimand refinement avoids selecting or discarding seeds using test outcomes.

A confirmatory superiority claim requires both a positive appropriately adjusted interval for target improvement and an upper interval for the candidate's mean old loss no greater than `tau`. The competing configuration must also meet the stated retention condition to call the comparison a same-retention win. If either side misses its protected retention condition, report the resulting two-dimensional operating points, not a fabricated constrained improvement. Do not choose another lambda after looking at test retention.

Secondary outcomes are old/new cross-entropy, exact held-out consumer KL, median and upper-quantile feature displacement, displacement-weighted truncation error, numerical failure rate, retained bytes, boundary/recurring consumer work, and measured time/memory. They diagnose the mechanism and costs; an improvement in one does not override a worse primary outcome. No model-based judge is used.

Show the full development stability–plasticity frontier. On protected data evaluate the frozen selected points and, only if declared in advance, a fixed small coefficient grid as a **descriptive** frontier with no post-test winner selection. A new favorable coefficient discovered there requires a new confirmation procedure.

### 9.2 Development, selection, and confirmation separation

Use three logically distinct sets of runs:

**Exploration:** the initial paired development seeds and diagnostic constructions may change the implementation or task. All changes are logged. Diagnostic examples are never training anchors; opening diagnostics makes them exploratory, not protected final evidence.

**Precision calibration:** after fixing a candidate and a conventional champion on development data, use fresh development-only training seeds to estimate paired variance and stability of the selection policy. Begin with enough independent pairs to estimate variance (at least three); add pairs while variance uncertainty materially changes the planned feasible confirmation size. This is precision planning, not repeated significance testing. Report the initial count and every expansion.

**Confirmation:** freeze code, configuration, split manifests, coefficient selection, candidate/champion, contrast family, sample size, and analysis. Use fresh seeds generated from the public protocol hash, with a disjoint seed namespace from development. Test data may be scored only in the confirmation command. Persist an immutable `confirmation_lock.json` before launching it. No method, stopping rule, or task revision uses those test outcomes.

A later hardware failure that prevents scoring does not make the data scientifically inspected, but any completed test outputs are recorded. A changed method after test inspection starts a new claim version and needs new independent confirmation. Reusing the same public test set must be disclosed; for a substantively new claim, obtain an independent split/task or treat the result as exploratory rather than calling it fresh protection.

### 9.3 Configuration selection

For each method, rank its development configurations by target accuracy among configurations whose old-loss uncertainty is compatible with the retention goal. Prefer a one-sided development upper bound below `tau` where attainable; this builds a margin without pretending to guarantee final retention. Break ties by lower measured resource cost, then a deterministic configuration hash. If none is feasible, mark that method as infeasible at the threshold and keep its frontier for reporting; never silently drop it from the comparison table.

The conventional champion is selected **before testing** from the eligible nonlocal/local-surrogate baselines in the relevant regime. M01/M02 are candidates, not members of their own conventional champion pool. B11/B12 are included in the unrestricted engineering pool but not confused with the single-shared-producer scientific pool. Include applicable inverse-data alternatives in the image task pool. Keep L2 and global geometry as named mechanistic contrasts even when neither is champion.

The attribution control receives independent tuning under the same rule. Do not compare a tuned candidate to the candidate's numerical lambda copied into a randomly oriented metric and call the resulting optimization mismatch causal evidence.

### 9.4 Replication count and precision

Let `s_D` be the development calibration standard deviation of a paired accuracy difference and let `h` be the desired half-width in percentage points. Under the usual approximate seed-level normal model, choose the smallest integer `n>=3` satisfying

\[
t_{1-\alpha/(2m),n-1}\,s_{D,upper}/\sqrt n\le h, \tag{5}
\]

where `m` is the number of predeclared confirmatory intervals and `s_D,upper` is a conservative upper uncertainty estimate of the seed standard deviation, not merely its most favorable small-pilot estimate. A chi-square upper variance bound is acceptable when the seed-normal approximation is reasonable; otherwise use a development bootstrap sensitivity analysis and expose its small-sample weakness. Estimate retention-loss variability separately and choose `n` sufficient for all decision-critical intervals.

Precision is chosen from the decision, not a universal gain target. Start with a reporting resolution no coarser than half the benchmark retention allowance. If a plausible development candidate advantage is smaller, target a half-width no greater than half that advantage. This is a **resolution target**, not proof that the estimated advantage will reproduce or be practically useful. A pilot with zero observed variance requires further calibration or a conservative variance floor justified by independent data; it does not justify an almost unreplicated claim. An exact invariant, such as unchanged B11 source predictions, can instead establish that particular zero difference algebraically, without pretending it estimates variability of target-learning gains.

Where application owners supply an actual acceptable loss or cost margin, use it instead and preregister its rationale. Without such a utility model, report detected effects, errors avoided per 10,000 decisions, bytes, and amortized time; leave economic usefulness conditional. Do not invent a minimum useful accuracy gain solely to ensure an affirmative conclusion.

If Equation (5) implies more local work than the measured machine/time budget can support, either refine the task/variance model on development data or report the effect as unresolved at the desired precision. Do not silently revert to five seeds and interpret a broad interval as failure. More seeds reduce training-run uncertainty but cannot repair a small number of independent test subjects or eliminate dataset shift.

### 9.5 Uncertainty, dependence, and multiplicity

Report raw paired seed differences and simultaneous two-sided t intervals for the predeclared accuracy contrasts, and a simultaneous one-sided upper bound for old loss. A conservative Bonferroni construction with explicit `m` is sufficient and straightforward; a predeclared Holm testing procedure is also acceptable, but do not call ordinary unadjusted intervals simultaneous. Treat candidate-versus-champion, candidate-versus-L2, candidate-versus-global, candidate-versus-orientation, and the retention gate as one planned family when they all support a single success narrative. Add another dataset or another selected candidate to that family or explicitly make it a separate preregistered confirmation.

Seed intervals are conditional on the fixed test set. For population sensitivity, use a **crossed paired bootstrap**: draw seed indices with replacement, draw one common set of test-example groups (or HAR subjects) with replacement, and apply the same two draws to every method in a replicate. Do not independently resample examples within each selected seed and then treat the resulting seed-by-example entries as independent observations. For HAR, retain all windows of each drawn subject; report the small number of subjects prominently. Use at least enough bootstrap replicates that Monte Carlo error of interval endpoints is negligible at the stated reporting precision, and verify endpoint stability by increasing the replicate count. A starting 10,000 replicates is a computation setting, not a new unit of replication.

When source split variability matters, repeat a development split study using independent grouped splits. A result confirmed on one final split is scoped to that split/task; it does not become a cross-dataset result merely by changing random initialization. Keep optimization seeds, split seeds, corruption seeds, anchor seeds, and repeated timing blocks distinct in manifests and analysis.

No optional significance stopping is permitted in confirmation. Run the frozen `n` even if early seeds look decisive. Development can stop or redirect on feasibility/precision grounds because it does not provide confirmatory p-values. Failed/nonfinite runs are not omitted: report intention-to-run counts, failure reasons, and a predeclared failure policy. The default policy treats a method-caused nonfinite/unusable run as a failure of reliable execution: it precludes an unconditional positive confirmation claim for that candidate under this protocol. Completed-pair accuracy summaries remain descriptive and explicitly conditional; do not impute an invented score or delete failed seeds. A source-training failure similarly invalidates the claimed reliable source pipeline until a new development/confirmation policy is established. Only clearly external interruptions may be resumed or identically restarted; method instability remains an outcome.

### 9.6 Interpretation of outcomes

| Observed outcome | Supported conclusion | Unsupported conclusion |
|---|---|---|
| Correct local expansion, no neural frontier gain | The implementation expresses the intended local geometry; useful adaptation gain not shown here | Geometry can never help at another interface or scale |
| Local beats global/L2 but not bucket/scalar control | Conditional weighting may matter; fine input-specific orientation not established | A special relationship representation is needed |
| Local wins only at larger retained bytes | A quality–storage tradeoff exists | Resource-matched superiority |
| Linearized logits win | Cached local consumer surrogates are promising in this setting | The original quadratic is the successful contribution |
| Private producer/adapter or inverse-data baseline dominates | The unrestricted practical case for this cache is weak | Shared-producer scientific comparison is mathematically meaningless |
| Toy win, HAR null with wide interval | Constructed effect established; transfer unresolved | Robust real-world improvement or general refutation |
| Higher retention but worse target accuracy beyond the stated operating point | Different frontier/regularization strength | Unqualified capability improvement |
| Fewer consumer calls but slower total time | Possible access advantage, no measured speed advantage | Efficient acceleration |
| Replicated protected improvement with attribution and cost controls | Narrow evidence supporting local compatibility caching under the tested conditions | Consciousness, biological realism, or frontier-LLM gains |

<a id="hardware"></a>
## 10. Real MacBook M3 Max implementation and resource accounting

### 10.1 Hardware discovery comes first

The plan does not assume a particular M3 Max RAM or GPU-core configuration. The initial discovery script records native architecture, machine model/chip description, physical and available unified memory, GPU information, macOS version, free disk, swap status, Python executable/architecture, installed package versions, power mode, and relevant environment variables. It must run before choosing training scale.

The following are ordinary macOS/Python setup commands, **not commands already executed on a Mac in this investigation**:

```bash
uname -m
sw_vers
sysctl -n hw.memsize
sysctl vm.swapusage
vm_stat
system_profiler -json SPHardwareDataType SPDisplaysDataType SPSoftwareDataType
df -h .
python3 --version
```

Store their stdout/stderr and exit status, including unavailable optional fields, in `artifacts/hardware/`. Use `psutil.virtual_memory()` and disk usage for programmatic availability checks. `arm64` is the intended native route; do not accidentally benchmark an x86 Python under translation and label it representative M3 performance.

Let physical memory be `M`, currently available memory be `A`, and the backend's recommended working set be `R` when MPS is available. Set initial OS/application headroom `H=max(8 GiB,0.25*M)` and an experiment process budget `B=max(0,min(0.6*A,M-H))`. These are conservative operational policies, not claims about a minimum required RAM configuration. Run serially. Launch longer work only when its calibrated upper working-set estimate with a safety factor fits `B`, free disk covers the artifact forecast, and sustained swap growth is absent. Recheck before each stage; a machine with substantial nominal RAM can still be unavailable because of other workloads.

### 10.2 Verified software route

As researched on 3 October 2026, the official releases identify **PyTorch 2.14.1** and compatible **TorchVision 0.29.1**, released on 30 September 2026. [R19][R20] The current MPS guide describes macOS 14+ availability checks. [R21] These are source-verified release/documentation facts, not an installed-and-tested target environment.

Use native CPython 3.13 where available, a project-local virtual environment, and pinned `torch==2.14.1`, `torchvision==0.29.1`. NumPy, SciPy, pytest, and psutil supply arrays/statistics/tests/resource observation. Resolve their compatible native wheels once during setup, save the full lock and wheel hashes, and do not upgrade mid-study. JSON is the primary configuration format, avoiding an unnecessary configuration framework.

```bash
# Check the interpreter before creating the environment.
python3 -c 'import platform,sys; print(platform.machine(),sys.version); assert platform.machine()=="arm64"; assert (3,11)<=sys.version_info[:2]<=(3,14)'
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --only-binary=:all: torch==2.14.1 torchvision==0.29.1 numpy scipy pytest psutil
python -m pip check
python -m pip freeze > requirements.lock.txt
python -c 'import torch,torchvision; print(torch.__version__,torchvision.__version__); print(torch.backends.mps.is_built(),torch.backends.mps.is_available())'
```

Installation syntax follows the official pip-based Mac route; the explicit compatible pair is verified by the release notes. [R20][R30] A failed wheel resolution is a setup blocker to diagnose, not permission to build an unrecorded source distribution or use cloud CUDA. A separately recorded fallback is the documented PyTorch 2.13.0/TorchVision 0.28.0 pair, with all parity tests rerun. CPU-only execution is a valid scientific fallback when a compatible local installation works; it cannot support an MPS speed claim.

### 10.3 Backend choice and operation boundaries

Use **PyTorch CPU float64** for the trusted small Jacobian, SVD, QR, eigendecomposition, finite-difference, and statistical reference operations. Use **PyTorch MPS FP32** for ordinary training/inference only after operation and gradient parity tests. No CUDA kernels, distributed training, paid services, FlashAttention, custom Metal code, or compilation are required.

The latest PyTorch patch specifically fixes MPS SVD/lstsq correctness and regression issues, including rank-deficient/ill-conditioned cases. [R19] Therefore, do not describe SVD as universally unsupported on MPS; the reason for CPU factorization is a simpler audited numerical path and independence from device/version-specific linear-algebra behavior. CPU construction and transfers are intentional and counted, not silent fallback.

Before Python starts, leave `PYTORCH_ENABLE_MPS_FALLBACK` unset or explicitly set it to `0`; log its value. The documentation states that `1` enables CPU fallback for unsupported operations. Do not set the allocator high-watermark ratio to zero, which disables the safety limit. [R22] Leave fast-math overrides off. Where the documented memory-fraction function is available, choose a conservative fraction no larger than `min(0.8,0.8*B/R)` and still monitor overall process/system pressure; an MPS allocation cap is not a cap on every CPU allocation. [R23]

MLX is a legitimate Apple Silicon alternative with documented installation, lazy evaluation, and shared-memory behavior. [R24][R25][R26] It is not chosen initially because this experiment benefits more from one trusted PyTorch reference and directly comparable autograd implementations than from maintaining two ports. No MLX speed claim is assumed. A later MLX port must preserve gradients/objectives and explicitly force evaluation when timing; it is an implementation experiment, not evidence that the underlying method improved.

### 10.4 Analytical memory estimates

For `P_train` trainable and `P_frozen` frozen parameters, FP32 weights, gradients and two Adam moment tensors give the lower-bound persistent model accounting

\[
M_{model}=16P_{train}+4P_{frozen}\ \text{bytes}. \tag{6}
\]

This excludes optimizer step metadata, activations, temporary kernels, CPU copies, framework runtime and allocation caches. It is not a peak-memory estimate. Inference weights for all modules use `4*(P_train+P_frozen)` bytes.

| Item | Main image configuration | Interpretation |
|---|---:|---|
| All FP32 model weights | 3,606,096 bytes, about 3.44 MiB | Includes producer and both consumers |
| Persistent adaptation model/grad/Adam lower bound | 10,418,376 bytes, about 9.94 MiB | Excludes activations and runtime |
| Rank-eight factors | 1,048,576 bytes | Training cache only |
| Reference features | 131,072 bytes | 256×128 FP32 |
| Uint8 anchor images | 200,704 bytes | Without labels, IDs, or metadata |
| Global full matrix | 65,536 bytes | 128×128 FP32 |
| Full contrast derivatives for B07 | 1,179,648 bytes | 256×9×128 FP32, plus references/probabilities |
| One additional private producer | 935,424 bytes | Persistent deployment weights when kept |
| Whole 60,000-image training partition | About 44.9 MiB uint8, or 179.4 MiB FP32 | Keep uint8 and transform batches rather than maintaining unnecessary duplicates |

A fully materialized `N×C×d` float64 Jacobian would use about 2.5 MiB here; construction should still stream anchors and avoid retaining unnecessary graphs. Small CPU model copies, current batches, saved activations, serialization buffers, and dataset caches must be included in measured working sets. There is no attention KV cache in these models.

For HAR, all three modules have 145,036 parameters; the same formula applies using 111,238 trainable and 33,798 frozen parameters. Raw FP32 nine-channel windows use `4*9*128=4,608` bytes each, so anchor inputs can cost more than rank-four factors. Byte matching must therefore be recalculated, not copied from the image example.

A **planning envelope**, not a measurement, is 1–3 GiB process/backend working set for these small experiments. Use at least 1.5 times the measured preflight maximum for launch planning; invalidate this envelope if the actual backend or loader uses more. Heavy swapping is not a viable memory strategy. Do not add process RSS and driver allocations as though they were guaranteed disjoint physical memory on a unified-memory system; report both and observe system pressure.

### 10.5 Compute, break-even, and full-suite forecasting

A dense-layer estimate for the main image adaptation is roughly 131 million multiply–accumulates per step using a three-forward-equivalent approximation for trainable forward/backward work. Exact old-consumer distillation adds roughly 21 million multiply–accumulates for a 32-anchor consumer forward and input backward. These derived estimates omit nonlinearities, optimizer operations, dispatch, transfers, and runtime allocation. They are not laptop timing predictions.

Let measured construction and per-step times be `C_m` and `t_m`. For `S` adaptation steps, compare

\[
T_m(S)=C_m+S\,t_m+T_{evaluation,m}+T_{loading,m},\quad
S_{break}=\frac{C_{cache}-C_{exact}}{t_{exact}-t_{cache}} \tag{7}
\]

when the denominator is positive. If it is nonpositive, caching has no time break-even against that exact comparator under the measured workload. Include source-feature extraction, CPU float64 differentiation/factorization, transfer, serialization if required, and all method-specific monitoring. Report one-time setup/source pretraining separately and also report the complete program cost, including tuning and failed runs.

Consumer access has its own ledger: sample forwards, class-direction VJP equivalents, and actual elapsed construction/backpropagation time. A batched call is not one unit regardless of its batch or number of output directions. Ordinary per-anchor full-logit construction costs up to `N*C` reverse directions; an optimized contrast implementation can reduce redundancy but must pass exact parity and report actual counts. Exact KD typically needs one loss-directed VJP per anchor per update, not `C` separate directions. Do not compare these as equal calls.

Forecast the suite only after preflight:

`total forecast = shared source trainings + sum over planned configurations/seeds of (cache construction + adaptation steps × measured step time + evaluation) + analysis/checkpoint overhead`.

Counts are determined by the calibration, baseline matrix, confirmation precision, and conditional gates, not by an arbitrary fixed suite size. Save a machine-generated `budget_forecast.json` before each costly stage, including optimistic/central/conservative timing ranges and available disk. A run that does not fit the user's actual local resources remains unlaunched with an explicit reason.

### 10.6 Timing and memory microbenchmark protocol

Use AC power and a recorded power mode; avoid other heavy jobs. Record macOS, Python, package versions, process architecture, ambient operating conditions available to the user, and whether external workloads or thermal throttling were observed. Do not invent GPU power or energy readings when instrumentation is unavailable.

Measure CPU and MPS versions of representative complete training steps, not just the factor multiplication. Begin with 20 warm-up steps and blocks of 100 measured steps; extend warm-up if block performance has not stabilized. Synchronize MPS immediately before and after a timed block. Separately measure batch-one inference latency and the declared batch throughput. Keep preprocessing, batch transfer, and actual selected branches in end-to-end measurements; also report kernel-only diagnostics when useful, clearly labeled.

Randomize or alternate method order across repeated blocks. Use block-level paired intervals, not thousands of correlated kernel timings as independent samples. Increase repetitions until the cost ordering is resolved at useful precision or identify it as noise-limited. A speed difference smaller than thermal/order variation is inconclusive. Include cold loading/construction separately from warm repeated execution and use the intended deployment horizon for amortization.

Sample `torch.mps.current_allocated_memory()` and `driver_allocated_memory()`, process RSS, available memory, and swap status. The current-allocation APIs are documented; a sampled maximum is **not a guaranteed allocator peak**. [R23] Record sampling cadence and its limitation, use an optional platform profiler for suspicious transient peaks, and retain a safety margin. Catch OOM/unsupported-operation exceptions without silently enabling fallback or dropping examples.

### 10.7 Scale changes that preserve the question

If preflight is memory- or time-limited, use producer `784->128 ReLU->64`, consumers `64->128 Tanh->128 Tanh->10`, 64 anchors, rank four, and target/anchor batches 32/16. Recompute all baselines and byte matches. This preserves nonlinear local-versus-global geometry, but may alter task capacity and whether producer adaptation is necessary; rerun the viability gate.

Microbatching with gradient accumulation may preserve an optimizer update when both target and anchor losses are weighted by their true example counts and the model has no batch-dependent state. Verify it against the full batch before relying on it. Shortening training until the effect disappears does not preserve the convergence hypothesis.

Increase scale only to test a named missing condition, such as a more expensive real consumer or measurable input-dependent sensitivity absent at the smaller scale. Synthetic extra consumer depth can map a break-even curve but must be labeled a **synthetic cost-scaling experiment**, not a demonstrated deployment speedup. Large-vocabulary/sequence-model extensions require a new derivative-sampling and masking specification and are not silently part of this executable core.

<a id="repository"></a>
## 11. Coding-agent repository, manifests, and recovery

### 11.1 Implementation status and repository structure

**Only the mathematical reference program reproduced in Appendix A was implemented and executed for this deliverable.** The project modules and `python -m idea1.cli ...` commands below are a specification for the coding agent to implement. They are not presented as existing installed software. No absent archive or previous chat is a dependency.

Build only the modules needed by the next gate:

```text
idea1/
  README.md                         # setup, current gate, claim boundaries
  pyproject.toml                    # package/CLI and dependency metadata
  requirements.lock.txt             # resolved target-Mac environment
  src/idea1/
    cli.py                          # explicit subcommands and stage guards
    config.py                       # JSON validation and canonical config hashes
    data.py                         # image/HAR loading, grouping, transforms
    models.py                       # declared producers, consumers and adapter
    cache.py                        # CPU factor/contrast construction and metadata
    objectives.py                   # all baseline losses, no hidden normalization
    train.py                        # paired batches, frozen-state checks, checkpoints
    evaluate.py                     # split guards and per-example predictions
    analysis.py                     # paired statistics, frontiers, byte matches
    benchmark.py                    # synchronized timing and memory observation
    manifests.py                    # hashes, provenance, statuses, atomic writes
  scripts/
    sanity_checks.py                # copy Appendix A verbatim first
  tests/
    test_math.py
    test_gradients.py
    test_data_integrity.py
    test_controls.py
    test_backend_parity.py
    test_resume.py
    test_end_to_end.py
  configs/
    image_initial.json
    har_initial.json
    confirmation_lock.json          # generated only after development decisions
  data/
    raw/                            # checksummed source downloads, audit access
    prepared/                       # immutable arrays and split manifests
  artifacts/
    hardware/
    research/                       # inspected sources and original-source audit
    E00/ ... E11/                    # stage status, budget forecast and gate decision
    source/<source_id>/
    runs/<run_id>/
    reports/
```

A single Python package with a small CLI is sufficient. Do not build a distributed scheduler, web dashboard, remote experiment service, or generalized architecture-search platform. Use local JSON/JSONL/CSV and tensor files. A human-readable report is generated from those raw artifacts rather than edited independently to match expectations.

### 11.2 Required CLI behavior

After the agent implements the package, install it with `python -m pip install -e . --no-deps`. Each proposed command supports `--dry-run`, reports its resolved configuration, refuses to overwrite completed artifacts, and checks prerequisite gate records. CLI errors use a nonzero exit code and preserve a diagnostic JSON record.

`doctor` discovers hardware and package capabilities. `prepare-data` verifies licenses/integrity and writes splits. `self-test` runs trusted numerical checks. `pretrain` produces immutable source checkpoints. `build-cache` creates only permitted method state. `run-stage` expands an explicit stage plan into serial jobs. `benchmark` executes synchronized measurements. `plan-confirmation` calculates precision and writes a lock. `confirm` refuses unlocked or altered configurations. `report` rebuilds tables and conclusions from raw data.

No subcommand may infer that a dataset named `test` is safe for tuning. `evaluate.py` takes a split role enum (`train`, `development`, `diagnostic`, `protected_test`) and an evaluation-purpose enum. It rejects protected-test access without a valid confirmation lock. A separate integrity-only path can compute raw hashes under the predeclared duplicate-audit rule and cannot import models or compute scores.

### 11.3 Run identity and artifact schema

Set `run_id=SHA256(canonical_config + source_hash + split_hash + code_snapshot_hash + seed_bundle + backend)[:20]`. Store the full hash too. Include data versions and the corruption manifest in the canonical configuration. Do not encode a favorable result into a run ID or silently reuse a source checkpoint trained on different splits.

Each run writes:

```text
manifest.json                # immutable input/provenance declaration
status.json                  # planned/running/complete/failed/interrupted, reason
metrics.jsonl                # step-level losses, data counts, elapsed times
checkpoint_latest.pt         # model/optimizer/torch RNG tensors and scalar state
checkpoint_latest.json       # step, sampler position, Python/NumPy RNG state
checkpoint_selected.pt       # selected only by the frozen development rule
predictions_<split>.npz       # root ID, subject/group, truth, probabilities, seed
costs.json                   # construction/recurring access, bytes, timing, memory
stdout.log / stderr.log
gate_summary.json            # eligibility and next action, not retrospective prose
```

The manifest contains code commit or full source snapshot hash plus dirty diff; exact Python and package versions; hardware/backend/precision; environment variables; source and consumer checkpoint hashes; dataset URL/archive hash/license record; split/preprocessing/corruption hashes; all hyperparameters; all seeds; actual batch/update counts; anchor order/IDs; source and target head initializations; access/architecture regime; and planned outcomes/analysis version. Record model tensor shapes and parameter counts so a future reader can detect a silent architecture change.

Costs distinguish source pretraining, boundary construction, adaptation, evaluation, tuning and serialization; algorithm state versus audit storage; CPU versus MPS work; and cold versus warm timing. Preserve failed, negative, and inconclusive runs with the same schemas. Public report copies should omit hardware serial numbers and unnecessary local usernames from raw system-profiler output.

### 11.4 Checkpointing and interruption recovery

Checkpoint at the fixed logging boundary and before planned termination. Write temporary files, flush/fsync, and atomically rename on the same filesystem. Save model parameters/buffers, Adam state, update count, data sampler permutation and cursor, independent RNG states, cache/source/config digests, and already-evaluated split roles. Python/NumPy RNG states belong in JSON-compatible serialization; torch/MPS RNG tensors can be saved alongside other trusted tensors. Load only locally generated trusted checkpoints with the appropriate safe tensor/state-dictionary loading path.

On resume, verify every immutable digest and restore sampler/RNG state before taking the next batch. Completed runs are skipped only when their status and all expected output hashes validate. A stale “complete” marker without predictions/cost records is not enough. An uninterrupted-versus-resumed CPU test must match updates within numerical tolerance; on MPS, document any reproducibility limitation and retain equivalent data ordering. Never claim cross-backend bitwise reproducibility.

On OOM or unsupported MPS operations, stop the run, preserve the last valid checkpoint and error, and release resources. Do not quietly change batch size, enable CPU fallback, or restart until a favorable result appears. During development, a repaired configuration receives a new ID; gradient accumulation or CPU fallback must pass parity and cost recalibration. During confirmation, use the predeclared external-failure policy and, if a backend/configuration change is necessary, revise and relock the confirmation program before making a new claim. Nonfinite loss caused by the method is an outcome, not an external interruption.

<a id="ladder"></a>
## 12. Numbered implementation and experiment ladder

Every stage writes its input hashes, actual resource use, outputs, and a gate decision (`proceed`, `repair`, `revise`, `stop_path`, or `inconclusive`). “Stop path” means stop investing in this implementation/settings; it is not a universal scientific refutation. All project commands in this section are **proposed interfaces to implement**, except standard Python/pytest/pip commands. Start from the repository structure and Appendix A; no prior project package is assumed.

### E00 — Source audit, hardware, environment, and data availability

| Field | Specification |
|---|---|
| Purpose / hypotheses / prerequisites | Establish executable premises for all hypotheses. Requires this document, the target Mac, local disk, and initial internet access for open-source packages/data |
| Implementation / inputs / command | Implement `config`, `manifests`, `doctor`, then data integrity/download paths. Run `python -m idea1.cli doctor --out artifacts/hardware` and `python -m idea1.cli prepare-data --config configs/image_initial.json`. Confirm MNIST rights or activate the named Fashion-MNIST fallback; verify HAR before its stage |
| Checks / baselines / metrics | Native architecture, package compatibility, MPS availability, known model tensor operations, dataset checksums/dimensions, group splits, no protected score access. No model-improvement comparison yet |
| Resources / launch guard | Downloads, imports, tiny tensors only; record memory and free disk. Refuse longer launches if the measured process envelope cannot fit Section 10's budget |
| Artifacts | `artifacts/hardware/*`, `artifacts/E00/status.json`, dependency lock, dataset source/rights records and immutable split manifests |
| Branches | Proceed when a local CPU route and valid dataset exist. Repair packages/paths if needed. Revise to the licensed fallback or CPU. Stop only the unavailable backend/data path. Unresolved rights or incompatible local software are explicit blockers, not negative learning evidence |

### E01 — Trusted mathematics, gradient flow, and backend parity

| Field | Specification |
|---|---|
| Purpose / hypotheses / prerequisites | H1 and correctness of the competing explanations; E00 environment passes |
| Implementation / inputs / command | Copy Appendix A to `scripts/sanity_checks.py`; run `python scripts/sanity_checks.py > artifacts/E01/reference.json`. Implement `cache`/`objectives` and the unit tests. Run `python -m pytest tests/test_math.py tests/test_gradients.py tests/test_controls.py`; then `python -m idea1.cli self-test --backend mps` where available |
| Checks / baselines / metrics | Factor identity, Taylor convergence, affine linearized exactness, fixed-reference hashes, orientation spectra, lambda/eta-zero parent recovery, full-versus-factor loss, no cache/consumer gradients. Compare CPU float64, CPU FP32 and MPS FP32 loss/gradient directions |
| Resources / launch guard | Small matrices and at most one tiny model; no dataset training. Log actual peak and dtype. Backend parity must pass before MPS training |
| Artifacts | `artifacts/E01/` numerical JSON, pytest output, backend capability matrix, known failure demonstrations |
| Branches | Proceed if the derivation and executable losses agree. Repair normalization, graph, shape or numerical bugs before any pilot. CPU-only continuation is permitted after MPS failure. A correct zero-Jacobian finite-drift failure supports the boundary analysis, not a broken implementation |

### E02 — Source training and an evaluation harness that does not leak

| Field | Specification |
|---|---|
| Purpose / hypotheses / prerequisites | Establish source quality and trustworthy measurement for H2–H6; E00/E01 pass |
| Implementation / inputs / command | Implement `models`, paired samplers, `pretrain`, split-guarded `evaluate`, checkpoints. Run `python -m idea1.cli pretrain --config configs/image_initial.json --seed-group development` with the initial Section 7 schedule, extending by the common development convergence rule |
| Checks / baselines / metrics | Source validation learning curves; chance-level or permuted-label control; same-checkpoint rerun predictions; train/validation/test ID barriers; counts; frozen-copy source retention. Verify an input-disjoint held-out set is never used as an anchor. Use B00/B01 as short diagnostic adaptations |
| Resources / launch guard | One serial source model at a time. Forecast from a measured 100-step block before completing source training. Save one shared source checkpoint per seed, not a separately optimized source for each method |
| Artifacts | `artifacts/source/<source_id>/`, `artifacts/E02/` source calibration, split-audit results and source selection policy |
| Branches | Proceed when source training is meaningful and its remaining convergence uncertainty is understood. Repair chance-level training or data-order bugs. If frozen features already solve the target as well as adaptation, mark this task uninformative for H2 and investigate a named harder condition on development data rather than claiming failure of all local geometry |

### E03 — Minimal mechanism positive and null assay

| Field | Specification |
|---|---|
| Purpose / hypotheses / prerequisites | H1/H2 harness sensitivity; E01 pass. Independent of expensive neural source convergence |
| Implementation / inputs / command | Implement the two-anchor quadratic construction in Section 4.6 with exact solutions across a lambda grid and identical-metric controls. Run `python -m idea1.cli run-stage E03 --config configs/image_initial.json` |
| Checks / baselines / metrics | Recover old drift 0.005 with target losses 0.405/local and 0.81/global in the worked example. When all `G_i` are identical, local and global losses/gradients agree exactly. Consistent permutations preserve results. Report exact analytical objectives, not classification accuracy |
| Resources / launch guard | Negligible small-array CPU work; no training data downloads needed |
| Artifacts | `artifacts/E03/solutions.json`, frontier data, invariant test output |
| Branches | Proceed if positive and null cases behave correctly. Repair the harness if it manufactures a null-case win. An analytic positive is permission to use the measurement code, not permission to claim neural or practical success |

### E04 — Small neural viability comparison

| Field | Specification |
|---|---|
| Purpose / hypotheses / prerequisites | Test whether H2's conditions exist; E02 source/harness and E03 pass |
| Implementation / inputs / command | Implement B00/B01/B02/B03/B07/B08/M01/M02 and orientation construction. On three development seeds run paired initial grids through `python -m idea1.cli run-stage E04 --config configs/image_initial.json`. Include B11 and image inverse-data alternatives early enough to expose an obviously dominated practical path |
| Checks / baselines / metrics | Old and new development accuracies, exact diagnostic KL, source/consumer immutability, spectral/subspace variation, feature displacement, and whether a trained producer improves target performance over converged frozen features. Do not select outcomes on test data |
| Resources / launch guard | Budget every source/cache/step block from E02 timing; serial execution; terminate unstable configurations with recorded failures. This screen may be short but cannot conclude final superiority |
| Artifacts | `artifacts/runs/`, `artifacts/E04/viability.json`, preliminary frontiers, cache audit plots/data |
| Branches | Proceed to fair tuning if a retention/plasticity tradeoff and input-specific geometry are present. Repair optimization when all methods fail. Revise task/head/scale only to instantiate a measured missing condition. Stop the practical cache path when a permitted simple architecture dominates with an already adequate quality/cost margin; preserve the narrow scientific question if still informative. Wide uncertainty is inconclusive |

### E05 — Fair development frontiers and conventional champion

| Field | Specification |
|---|---|
| Purpose / hypotheses / prerequisites | H2 and a fair test of the main alternative; E04 identifies an informative setting |
| Implementation / inputs / command | Complete B04–B06/B09/B10/B12 and task-applicable B13/B14. Apply Section 8's coefficient/LR/convergence rules, matched source checkpoints and batch streams. Run `python -m idea1.cli run-stage E05 --config configs/image_initial.json` |
| Checks / baselines / metrics | Retention threshold bracketed or infeasibility explicitly reported for every method; comparable tuning effort; no selected endpoint still improving without investigation. Select the compact/full candidate and conventional champion by the frozen development rule, not by narrative preference |
| Resources / launch guard | Generate an expanded `budget_forecast.json` before launching. Reuse valid caches across coefficients; count their construction once per actual cache and account for all exploratory work |
| Artifacts | `artifacts/E05/frontiers.csv`, search ledger including rejected/failed configurations, selected preliminary policy and convergence evidence |
| Branches | Proceed to attribution/cost if a potentially useful frontier remains. Repair unfair tuning or unresolved convergence. If local wins vanish, stop that local-setting path or report the conventional winner. A small unresolved difference triggers precision calibration, not a success or failure label |

### E06 — Attribution and full storage/data accounting

| Field | Specification |
|---|---|
| Purpose / hypotheses / prerequisites | H3/H4; candidate/champion from E05 |
| Implementation / inputs / command | Retrain C01 with its own tuning; implement largest-fitting-prefix byte matching, rank variants, scalar/coarse controls, and optional whitened orientation if conditioning matters. Include B11/B12 and inverse-data alternatives in their permitted envelope. Run `python -m idea1.cli run-stage E06 --config configs/image_initial.json` |
| Checks / baselines / metrics | Correct singular spectra and trace; equal anchor-prefix policy; actual serialized bytes; no hidden teacher state; gradient/convergence diagnostics; displacement-weighted spectral tail; independent rank versus anchor-count effects |
| Resources / launch guard | Reforecast using actual larger-anchor payloads and training data loader costs; no concurrency. Only run extra conditioning/rank tests motivated by the observed development ambiguity |
| Artifacts | `artifacts/E06/attribution.csv`, byte ledger, rank/coverage frontiers, explanation of the admissible architecture/data regimes |
| Branches | Proceed with a geometry claim only if correct orientation matters beyond tuned alternatives. Revise the claim to scalar conditioning, storage, or cached surrogate benefits if those explain it. Stop unnecessary fine-grained caching when dominated. Inconclusive conditioning effects remain explicit |

### E07 — Measured systems break-even

| Field | Specification |
|---|---|
| Purpose / hypotheses / prerequisites | H5; correct implementations and candidate quality region from E05/E06 |
| Implementation / inputs / command | Implement synchronized complete-step benchmarks, CPU construction timing, memory observation and access hooks. Run `python -m idea1.cli benchmark --stage E07 --config configs/image_initial.json` on actual CPU/MPS, including B07/B08 and the strongest applicable cheap baseline |
| Checks / baselines / metrics | Equal quality or explicit quality gap; actual preprocessing/transfer; no unnecessary Jacobian charged to exact KD; cold/warm separation; paired block order; measured break-even; source-only deployment memory versus adaptation cache |
| Resources / launch guard | Warm-up and block sizes from Section 10; memory cap and swap checks. Do not run artificial giant consumers merely to claim a win. Any depth scaling is a separately labeled workload characterization |
| Artifacts | `artifacts/E07/timing_blocks.csv`, `memory_samples.jsonl`, access ledger, `break_even.json`, environment snapshot |
| Branches | Proceed with a speed claim only when complete matched-quality cost improves at the intended horizon. Retain an access-only claim if useful but slower. Repair synchronization/fallback errors. Noise-limited or non-amortizing cases are inconclusive/negative for systems value, not mathematical falsifications |

### E08 — Representative HAR transfer and boundary tests

| Field | Specification |
|---|---|
| Purpose / hypotheses / prerequisites | H6 and whether a constructed-image result generalizes. E00/E01 pass for HAR; E04–E06 identify a credible mechanism or a specifically diagnosed missing image condition |
| Implementation / inputs / command | Prepare subject-grouped HAR, the declared non-invertible fault and no-fault control, and the Section 7 model. Run `python -m idea1.cli prepare-data --config configs/har_initial.json`, then `python -m idea1.cli run-stage E08 --config configs/har_initial.json`. Repeat the relevant E02–E06 calibration with HAR data; do not reuse image test-tuned coefficients |
| Checks / baselines / metrics | Subject disjointness, frozen train-only normalization, identical fault realizations across methods, no old raw window access beyond declared state; subject-macro primary metric; fresh nonlinear geometry/plasticity diagnostics; private producer and label/feature replay accounting |
| Resources / launch guard | Measure actual convolutional step time and window storage first. Forecast grouped evaluation and retained inputs, whose cost differs from uint8 images. Launch only when the named transfer question remains informative |
| Artifacts | `artifacts/E08/` subject/corruption manifests, selected policies, per-subject results, transfer and boundary-case analysis |
| Branches | Proceed to protected confirmation for a specific surviving claim. If only the constructed task works, restrict scope. Revise a demonstrably noninformative severity on development data before locking. A wide nine-subject uncertainty is not repaired by reporting individual windows as independent |

### E09 — Precision calibration, protocol freeze, and launch authorization

| Field | Specification |
|---|---|
| Purpose / hypotheses / prerequisites | Make H2–H6 confirmable; relevant development, attribution, and cost gates complete |
| Implementation / inputs / command | Use fresh development calibration seeds, compute Equation (5), finalize the comparison family and admissible regimes. Run `python -m idea1.cli plan-confirmation --from artifacts/E05 --attribution artifacts/E06 --out configs/confirmation_lock.json`; include HAR/E07 records when their claims are included |
| Checks / baselines / metrics | Every decision-critical outcome/contrast, coefficient, rank, duration, task, seed count, failure rule and analysis version is explicit; no protected prediction file has been read. Recompute full time/disk forecast from selected run counts |
| Resources / launch guard | Only additional calibration justified by variance uncertainty, then CPU planning. Confirmatory work launches only if local memory/disk and actual available execution budget support the planned precision |
| Artifacts | Immutable `confirmation_lock.json`, exact run list, expected uncertainty, budget forecast and source/code/dataset digests |
| Branches | Proceed if a testable frozen program fits. Repair missing decisions before unlocking any test. Revise the intended precision with explicit consequences if constrained; do not lower it to manufacture significance. Otherwise stop investment or record an unresolved empirical question |

### E10 — Independent protected confirmation

| Field | Specification |
|---|---|
| Purpose / hypotheses / prerequisites | Confirm only the locked hypotheses; E09 lock and all applicable guards pass |
| Implementation / inputs / command | Run `python -m idea1.cli confirm --lock configs/confirmation_lock.json`. Train the selected candidate, preselected conventional champion, named mechanistic baselines/control on all frozen fresh seeds. Source checkpoints remain paired within each new seed. Evaluate protected splits under the locked policy |
| Checks / baselines / metrics | No adaptive coefficient/seed selection; exact planned paired comparisons; source baseline scored on the same test cases; retained consumer immutability; run/failure counts; simultaneous uncertainty; crossed/subject-cluster sensitivity analysis |
| Resources / launch guard | Serial runs, stage budget, checkpoint/resume, stable machine conditions. Do not stop early for significance. External interruption handling follows Section 11 |
| Artifacts | Full run manifests/predictions, locked-analysis outputs, per-seed/per-subject differences, cost records, all failures |
| Branches | Proceed to reporting for positive, negative, mixed, or inconclusive results alike. Repair only verified evaluation-code defects with a complete audit and affected reruns. Method changes after test inspection require a new claim/confirmation program |

### E11 — Reproducible report and investment decision

| Field | Specification |
|---|---|
| Purpose / hypotheses / prerequisites | Convert evidence into scoped conclusions; development-only or E10 results available and labeled |
| Implementation / inputs / command | Run `python -m idea1.cli report --artifacts artifacts --out artifacts/reports/final.md`. Rebuild tables from raw records; include methods that failed or were infeasible; produce quality–retention and quality–cost frontiers without cherry-picked axes |
| Checks / baselines / metrics | Every claim links to run IDs and its contrast; no diagnostic win overrides the primary outcome; no full-EFC/FROMP claim from a partial penalty; no CPU time presented as M3 time; hypothetical estimates distinguished from measurements |
| Resources / launch guard | CPU analysis and ordinary disk output; no new training or test-driven hyperparameter changes |
| Artifacts | Final report, machine-readable claims ledger, compact reproducibility bundle description, explicit residual uncertainties and next gate if warranted |
| Branches | Extend only a supported narrow mechanism or named unresolved condition. Prefer the simpler winner when matched. Report a rigorous negative or inconclusive result without converting it into a claim that all relational computation fails |

<a id="checks"></a>
## 13. Test specification and work actually executed

### 13.1 Required implementation tests

| Test ID | Input and expected invariant | Tolerance / failure policy |
|---|---|---|
| T01 factor reconstruction | CPU float64 random smooth consumers; `K^T K` equals `J^T F J` at full rank | Small well-conditioned examples: max absolute error below `1e-10`; scale-aware relative test for larger matrices |
| T02 local KL expansion | Unit directions with displacement `1e-1,1e-2,1e-3`; approximation error decreases before cancellation dominates | Inspect absolute and relative errors; exclude near-zero denominators from ratio summaries, but retain them in absolute-error reporting |
| T03 affine exactness | Linearized logits versus exact affine logits at finite displacements | CPU float64 absolute KL difference below `1e-10` for the reference inputs |
| T04 autograd | Penalties influence current producer/new head as intended; cache and frozen consumer have no parameter gradients | Assert graph and parameter hash properties; exact KD must propagate through its input |
| T05 disabling the contribution | Same initialization/batches/RNG, `lambda=eta=0`, no anchor work | Recover CPU parent updates; do not call damping-only unconstrained |
| T06 metric null/positive controls | Identical `G_i` recovers global; two-anchor construction reproduces Section 4.6 | Exact small-array tolerances; no statistical success test needed |
| T07 orientation | Independent orthogonal rotations preserve each factor's singular values/rank/trace | CPU float64 errors below `1e-10`; verify the control actually changes orientations |
| T08 truncation | Full-minus-truncated quadratic is nonnegative and bounded by Equation (4) | Permit only floating-point roundoff; verify reported full trace does not use truncated energy |
| T09 degenerate inputs | Empty anchors, invalid IDs/rank/T, constant logits, nonfinite inputs, wrong interface shape, changed checkpoint/preprocess | Empty construction/invalid metadata raises a clear error; zero-rank auxiliary tests produce zero quadratic; production normalization never silently divides by zero |
| T10 cache reference/state | Loading, checkpointing, training, and repeated evaluation do not mutate `z0`, factors or consumer | Hash before/after; changed consumer invalidates cache |
| T11 data barriers | Root-image duplicates/transform pairs and HAR subjects cannot cross train/held-out groups | Assert exact set disjointness; protected scoring without a lock raises an error |
| T12 loss reductions | Unequal last minibatches, accumulation, and per-class KL sums use intended means | Compare against explicit Python sums; do not drop remainders or divide by classes twice |
| T13 baseline invariants | Private source producer and target-only adapter preserve old predictions; affine B07 can deduplicate its derivative | Exact CPU equality where operation path is identical; cost ledger reflects actual model executions |
| T14 CPU/MPS parity | FP32 small fixed batches: losses, gradients, parameter updates and cache penalties | Start with `atol=1e-5, rtol=1e-4` for losses and `rtol=1e-3` for non-negligible gradient norms/directions; investigate failures rather than widening tolerances until they pass. Float64 construction remains CPU |
| T15 resume | Interrupt at a non-epoch boundary; restore sampler, optimizer and all RNG states | Same CPU updates/predictions as uninterrupted run; document any backend-specific nondeterminism |
| T16 full pipeline | Tiny training subset, all essential baselines, build cache, train, validate, save/load and report | Finite outputs, correct counts/hashes, no hidden state, and no protected split touched |

For sequence/window models in this plan the length is fixed at 128, not an arbitrary language context length. Reject incorrect HAR lengths rather than padding without a specified mask. Variable-length LLM testing is out of scope. The optional router's feedback queue/reset edge cases are specified in Section 14.

### 13.2 Independent checks performed for this investigation

The Appendix A program was executed in **Linux, Python 3.13.5, PyTorch 2.10.0+cpu**, with MPS unavailable. It uses CPU float64, one CPU thread, seed `20261003`, 12 anchors, feature dimension six, and four output classes for the random nonlinear-consumer tests. It does not load a benchmark dataset or train an adapting neural producer.

| Executed check | Observed result |
|---|---:|
| Full-factor reconstruction, maximum absolute error | `6.94e-17` |
| Median exact-KL/quadratic ratio at displacement 0.1 | `0.9998279353` |
| Median ratio at displacement 0.01 | `1.0000341890` |
| Median ratio at displacement 0.001 | `1.0000039438` |
| Maximum relative error at displacement 0.001 | `0.0003482091` |
| Affine linearized-KL maximum absolute discrepancy | `8.88e-16` |
| Consistent-permutation penalty discrepancy | `0.0` |
| Orientation-control singular-spectrum maximum discrepancy | `1.11e-16` |
| Positive violation of the truncation upper bound | `0.0` |
| Detached gradient-flow assertions | Passed |
| Affine nullspace-projector maximum discrepancy | `3.72e-15` |
| Zero-Jacobian finite-displacement exact KL | `0.1201145070` |
| Two-anchor positive construction | Local target loss `0.405`, global `0.81`, both old quadratic drift `0.005` |

The same program checked the image parameter counts and storage arithmetic reported earlier. These numerical identities support the mathematical implementation. The positive construction is deliberately chosen to contain the effect and is not an unbiased learning benchmark. The finite-drift counterexample shows a real limitation rather than a coding error.

**Not executed here:** the proposed image/HAR training comparisons, official EFC/FROMP benchmark reproductions, the new complete CLI/repository, MPS parity/training/timing, the final statistical program, or a learned streaming router. S0's historical runs remain source-reported. The coding agent must not merge those historical numbers with new runs as independent replications.

<a id="routing"></a>
## 14. Conditional branch: adaptive fusion, not an automatic second project

### 14.1 Reduction and scope

S0's second formulation is technically coherent but weak as a new-mechanism claim. If `A` and `B` have residual variances one and correlation `rho`, the equal-weight fuser has MSE `(1+rho)/2`. At `rho=-0.8` it is 0.1 and at `rho=0.8` it is 0.9, while each marginal MSE remains one. This is a direct variance calculation. It shows why joint information can matter, not why a learned relational architecture is needed.

For a fixed executable fuser, define `Delta_t=loss(A_t)-loss(F(A_t,B_t))`. An EMA of `Delta_t` is exactly a utility estimate for the composite action “run A+B.” Renaming the same scalar as a relationship state cannot constitute an improvement. The current L2D-SLDS shared-residual model, a same-information generic recurrent model, and a composite utility tracker are strong comparators for any learned extension. [R13] Coverage-based model-set selection and a final fuser's realized task loss are not interchangeable outcomes. [R14][R15]

Do not activate this branch merely because compatibility caching is unpromising. Activate it only for an independent application with delayed labels, persistent joint error structure, and a meaningful optional-computation cost. Its novelty remains an evaluation/application claim unless new evidence identifies a distinct mechanism.

### 14.2 ER01 — Minimal causal routing assay, proposed but not executed here

| Field | Specification |
|---|---|
| Purpose / hypothesis | Validate the pair/composite equivalence and causal information budget for H7 before building neural branches |
| Prerequisites | E00/E01 local setup; an explicit reason to pursue the secondary application |
| Inputs / configuration | Generate streams initially 20,000 steps long. Draw independent standard normals `u,v`, set residuals `eA=u`, `eB=rho*u+sqrt(1-rho^2)*v`. Use independent nonconstant targets, e.g. `y~N(0,4)`, and predictions `A=y+eA`, `B=y+eB`. The environment hides `y` until feedback. Alternate `rho` between −0.8 and +0.8 with independently randomized regime durations uniform over integers 200–800. No regime label is supplied to executable gates |
| Work / command | Implement a small optional `routing.py` and `python -m idea1.cli run-stage ER01 --config configs/routing_assay.json`; this file is to be created from this section, not assumed to exist. No trained neural branch is claimed |
| Baselines / controls | Never B, always B, marginal-only tracker, pair-gain EMA, composite-gain EMA sharing identical update code, and a labeled regime oracle as an information-advantaged reference. Add constant-correlation, identical-branch and useless-B boundaries |
| Timing / feedback | Decide before current B is computed. Label delay is 20 steps initially; zero and 100 are development robustness conditions. Randomly audit 10% of unqueried cases using independent stored uniforms. Audits execute B and count in cost, but never change the already emitted prediction. Update histories only when the corresponding delayed label and paired output are available |
| Metrics / analysis | Report prediction MSE and executed-B fraction separately, plus `MSE+kappa*executed_fraction`; `kappa=0.3` is an illustrative source setting, not a measured hardware price. Select any deployment coefficient from a real cost/quality tradeoff before confirmation. Use independent streams or temporal blocks, not timesteps as independent replications |
| Resources / artifacts | Small CPU arrays and a delayed-feedback queue. Measure actual runtime; write `artifacts/ER01/` event ledgers, predictions, costs, regimes hidden from the policy, and equivalence assertions |
| Decision | Exact pair/composite equality is expected and narrows novelty. A scalar tracker winning over marginals only establishes the need for joint information in this construction. A neural/learned-state project needs an application and stronger-comparator advantage beyond this assay; otherwise stop at the scalar solution |

For each event, store generation time, output-emission time, query/audit decision, execution propensity, prediction(s), and scheduled feedback time. A fixed-delay FIFO may suffice initially, but the test suite must also handle empty queues, the last unobserved tail, zero delay, out-of-order arrivals in a robustness test, and reset between independent streams. A label released at time `t` cannot affect a decision already emitted at `t` unless the chosen event ordering explicitly processes feedback first; choose and freeze that order. Use feedback-first ordering for the initial assay, with zero-delay labels released after the current prediction so they are never contemporaneously leaked.

Initial EMA updates use `(1-beta)*R+beta*Delta` on valid paired feedback only, with count, observation age and missingness stored separately. Tune `beta` on development streams; start with `{0.01,0.03,0.1}` and extend if the response timescale is outside the grid. Pair and composite implementations must make identical decisions for the same histories and random draws. Do not interpret a query-selected sample as an unconditional estimate without checking the sampling mechanism. A contextual gate that selects on current information may require propensity correction or explicit conditional estimation; random audits alone do not make every naive estimator unbiased.

If a learned internal-branch extension is justified, freeze encoders and the actual latent fuser before gate training, use separate/cross-fitted training trajectories, give the generic-history baseline exactly the same observations and memory capacity, and include a current shared-residual approach with the composite action. Count branch training, router training, delayed-feedback buffers, audits, and live inference. A weight update to B or the fuser invalidates the interpretation of an unchanged historical utility state unless versioning/reset is specified. This extension requires its own E09-style protocol freeze; it is not authorized by a synthetic EMA success.

<a id="traceability"></a>
## 15. Claim-to-experiment traceability

| Claim / question | Evidence or rationale | Implementation | Experiments and baselines | Metric / gate |
|---|---|---|---|---|
| C1: local Fisher is a faithful local KL approximation | Equation (1), executed numerical check; feature geometry related to EFC [R01] | `cache.py`, `objectives.py`, T01–T04 | E01, E03; full quadratic vs exact KL | Local error/convergence; repair before training if incorrect |
| C2: input association matters for learning | H2; two-anchor distinction is only a possibility proof | Local/global/scalar/bucket losses | E04–E06, E10; B02–B07, M01/M02 | Protected target difference plus retention condition |
| C3: the orientation rather than generic regularization causes a gain | H3 and coordinate-conditioning caveat | Orientation and optional whitening controls | E06/E10; C01 independently tuned | Paired attribution contrast and stable optimization |
| C4: caching is worthwhile under restricted consumer optimization access | H4/H5; explicit A-B contract | Provider access guard, byte/call ledgers | E06/E07/E10; B07, B14 where applicable | Admissibility, bytes and actual cost—not just call counts |
| C5: unrestricted practical value survives simpler architecture | Exact private-copy invariance; adapter and inverse-data deductions | B11–B14 | E04–E07, E10 as applicable | Quality–deployment/training-cost frontier |
| C6: effect transfers to representative signals | H6; UCI data provenance [R28] | HAR grouped loader/fault generator | E08/E10 | Subject-macro effects and clustered uncertainty |
| C7: claimed effect is not test-selection or seed noise | Locked paired statistical procedure | `analysis.py`, `evaluate.py`, manifests | E09/E10 | Simultaneous intervals, failure counts, no optional stopping |
| C8: local execution is real | Official software route [R19][R20][R21][R23][R24]; no target measurements yet | `doctor`, `benchmark`, checkpointing | E00/E01/E07 | Backend parity, memory headroom, measured full cost |
| C9: learned relational routing adds more than composite utility | Exact reduction and current routing prior art [R13][R14][R15] | Optional queue/tracker/gate | ER01, then a separate locked neural study if justified | Actual fuser loss plus charged compute against strong adaptive alternatives |
| C10: contribution is narrow, not a new social architecture | Prior-art comparison and S0's own caveats | Claim ledger/report rules | All reporting at E11 | Every conclusion matched to its actual evidence boundary |

<a id="reviews"></a>
## 16. Three adversarial self-reviews and revisions

These were distinct review passes over the drafted specification, not independent external peer reviews. The changes below are incorporated in the relevant algorithm, baseline, statistical, and execution sections. Repeated review does not prove correctness; remaining empirical risks have gates or explicit scope limits.

### 16.1 Scientific and novelty review

| Material weakness found | Revision made to the actual plan | Residual risk |
|---|---|---|
| Calling the local Fisher cache a broadly new relational architecture would ignore EFC and functional/Jacobian regularization | Narrowed contribution to interface/access evaluation; added EFC implementation inspection, FROMP, Jacobian/Sobolev and sketching comparison; full-method claims explicitly excluded | An exact earlier implementation may still reduce the combination to rediscovery |
| A linear consumer cannot support the proposed varying-nullspace explanation | Added the affine kernel deduction, an affine boundary control, and a nonlinear main consumer | Nonlinear models may nevertheless have nearly constant useful directions |
| Fisher quadratic may lose mainly because it also linearizes softmax curvature | Added cached linearized-logit KL, its exact affine property, contrast storage and deduplication | It remains local for nonlinear consumers and may be inferior to exact KD |
| An independent-feature toy grants degrees of freedom a shared producer may not have | Labeled the positive construction a harness test and required neural/held-out transfer | A small network may still lack the relevant adaptation conflict |
| Biological language risks suggesting untested cognition | Removed biological claims from the mechanism and success criteria | No neuroscience conclusion is supported |
| A changing reference could conceal cumulative drift | Fixed reference/checkpoint hashes and made refresh a separate paid-access method | Fixed caches can become stale during useful large adaptation |

### 16.2 Experimental and statistical review

| Material weakness found | Revision made to the actual plan | Residual risk |
|---|---|---|
| Domain-known heads make a second producer an obvious competitor; its weights can be cheaper than the cache | Added B11/B12 and separate single-producer versus unrestricted envelopes with actual byte accounting | A real application has not yet supplied its architecture/storage constraint |
| The known image rotation reconstructs clean labeled training data | Added inverse-label replay and inverse-feature teaching; restricted rotation to a diagnostic role for old-data scarcity claims | The sensor task is still a controlled stress test, not observed deployment drift |
| Extra factor bytes, anchor selection, and coarse weighting can masquerade as geometry | Added byte-matched anchor prefixes, scalar/bucket metrics, audit-state separation and orientation controls with fair tuning | Matching bytes does not equalize information content, convergence or data utility |
| Copying one coefficient across methods is unfair | Added per-method bracketing, LR/convergence calibration, raw exact-KL coefficients and full search ledgers | Finite search can still miss a strong configuration; scope claims accordingly |
| Test-time feasibility filtering and favorable-seed deletion bias constrained comparisons | Freeze selected configurations before testing; report joint retention/target outcomes and all seed failures | Mean retention does not guarantee every run or deployment satisfies the limit |
| Arbitrary improvement targets and five-seed defaults can overstate or miss effects | Replaced with decision-resolution/variance-based replication, simultaneous paired intervals and explicit infeasible-precision outcomes | Small development variance estimates and nine test subjects limit precision |
| Window/timestep pseudoreplication and optional stopping could inflate evidence | Subject grouping, crossed bootstrap, fixed confirmation sample and delayed-feedback event ledger | Population inference beyond the available subjects/tasks remains limited |
| Improved retention or a weighted routing score could hide degraded target quality | Primary outcomes and compute/error components remain separate; descriptive secondary metrics cannot rescue a failed primary claim | Application-specific utility remains unspecified |

### 16.3 Implementation and hardware review

| Material weakness found | Revision made to the actual plan | Residual risk |
|---|---|---|
| Historical package links might be treated as available working code | Source archive explicitly unavailable; only Appendix A is implemented; every project command marked proposed | The coding agent must still build/test the small repository |
| MPS API existence is not numerical validation | CPU float64 cache reference, current patch-release inspection, device parity gate and no silent fallback | Target machine performance and unsupported-operation details remain unmeasured |
| Exact KD could be unfairly charged for the proposal's Jacobian normalization | Standalone exact baseline uses raw KL and its own coefficient | A badly instrumented implementation could still count shared work incorrectly |
| Nominal RAM and parameter bytes are not total working-set feasibility | Discovery, OS headroom, measured preflight, full lower-bound accounting and swap guards | Backend transient peaks may exceed sampled maxima |
| Normalized factors versus normalized loss can introduce a factor-of-s error | Chose raw factors plus one runtime normalization; explicit reduction and lambda-zero tests | Floating-point cancellation at nearly zero KL needs scale-aware diagnostics |
| Resume and data-loader randomness were underspecified | Full sampler/optimizer/RNG checkpoint contract, immutable digests and interruption parity test | Exact cross-backend replay is not guaranteed |
| “Boundary-only” access could hide development evaluation queries | Separate optimizer access from paid, fixed post-run provider validation; all such evaluations counted | A stricter deployment with no validation feedback would need a different selection policy |
| Dataset software licenses were being confused with data rights | Explicit license records, MNIST uncertainty, licensed image fallback and UCI attribution | Terms/archive discrepancies must be resolved at ingestion rather than assumed away |

### 16.4 Stress-test outcome matrix

If the idea is a rediscovery, retain the reproducible evaluation and drop algorithmic novelty. If the brain analogy is wrong but the method works, make an engineering claim only. If the method works only in the constructed case, publish that boundary without practical generalization. If the effect is smaller than noise, report the interval and precision needed; do not claim equivalence without a justified equivalence margin. If stronger tuning or equalized resources remove the gain, prefer the conventional explanation. If only a secondary metric improves, report the tradeoff. If one seed or split drives the result, inspect it as exploratory and require independent confirmation. If hardware support fails, continue on a validated local CPU path or stop the backend path. If available memory is low, reduce scale with the declared viability recheck. If the pilot lacks the necessary varying sensitivities or adaptation conflict, test a named condition rather than either forcing a positive result or issuing a universal rejection.

**Residual blockers to positive empirical claims:** no independent adaptation study, no protected confirmation, no target-Mac measurements, no application-specific utility/access contract, and no evidence of large-scale transfer. These are explicit future experiment requirements, not claims that the research plan itself is unfinished.

<a id="references"></a>
## 17. References, inspection record, and search coverage

### 17.1 Citation convention

`[Rxx]` citations resolve to ordinary source URLs through the link definitions at the end of this section. Entries identify publication/version dates and inspection depth. No numeric benchmark gain from another paper is used as a measured result of this project. S0 is the supplied text identified in Section 2; its historical experiments and inaccessible package are not independent external validations.

### 17.2 Bibliography and inspection depth

| ID | Bibliographic reference / verified source | Inspection and use |
|---|---|---|
| [R01] | Magistri et al. **Elastic Feature Consolidation for Cold Start Exemplar-free Incremental Learning.** ICLR 2024; arXiv:2402.03917, inspected v3 | Full HTML method, local/averaged feature matrix and regularizer; decisive prior art |
| [R02] | Magistri et al. **EFC++: Elastic Feature Consolidation with Prototype Re-balancing for Cold Start Exemplar-free Incremental Learning.** IJCV, published 28 September 2026; DOI `10.1007/s11263-026-03018-4` | Publication metadata and related full preprint inspected; recent development distinct from 2025 preprint |
| [R03] | Magistri et al. Official **elastic_feature_consolidation** repository | README/license indication and the two source files identified in Section 3.3; no full training reproduction |
| [R04] | Heckel. **Provable Continual Learning via Sketched Jacobian Approximations.** AISTATS 2022; arXiv:2112.05095 | Full HTML methods/theoretical scope; parameter-space versus activation-space distinction |
| [R05] | Shen et al. **Towards Backward-Compatible Representation Learning.** 2020; arXiv:2003.11942v2 | Full HTML method; frozen old-classifier compatibility comparison |
| [R06] | Titsias et al. **Functional Regularisation for Continual Learning.** arXiv:1901.11356v4, ICLR 2020 | Full HTML; inducing/function-posterior predecessor |
| [R07] | Pan et al. **Continual Deep Learning by Functional Regularisation of Memorable Past.** NeurIPS 2020 | Proceedings PDF main methods and ablations inspected, including page images for the functional prior and Split CIFAR discussion; full implementation not reproduced |
| [R08] | Buzzega et al. **Dark Experience for General Continual Learning: a Strong, Simple Baseline.** NeurIPS 2020; arXiv:2004.07211v2 | Full HTML; logit/label replay baseline lineage |
| [R09] | Saha, Garg and Roy. **Gradient Projection Memory for Continual Learning.** ICLR 2021; arXiv:2103.09762v1 | Full HTML; parameter-gradient subspace distinction |
| [R10] | Czarnecki et al. **Sobolev Training for Neural Networks.** 2017; arXiv:1706.04859v3 | Full HTML; derivative-informed training and projected derivatives |
| [R11] | Srinivas and Fleuret. **Knowledge Transfer with Jacobian Matching.** ICML 2018, PMLR 80 | Proceedings PDF methods and page image inspected; local linearization/derivative transfer distinction |
| [R12] | Jaderberg et al. **Decoupled Neural Interfaces using Synthetic Gradients.** ICML 2017, PMLR 70 | Proceedings abstract/method description at overview level; not a decisive exact-equivalence claim |
| [R13] | Montreuil et al. **Learning-to-Defer in Non-Stationary Time Series via Switching State-Space Models.** arXiv:2601.22538v3, 2026 | Full HTML current version; shared residual structure and partial-feedback adaptive comparator |
| [R14] | **FlexRouter: Learning Complementary Model Sets for Flexible LLM Routing.** arXiv:2609.38585v1, 29 September 2026 | Full HTML; complementary set selection and coverage/fusion distinction. Cited as this dated preprint, not a future conference event |
| [R15] | Zhang et al. **Mixture of Complementary Agents for Robust LLM Ensemble.** arXiv:2605.24048v1, 21 May 2026 | Full HTML; final-summarizer-conditioned complementarity |
| [R16] | Qwen Team. **Qwen3 Technical Report.** arXiv:2505.09388, 2025 | Architecture and training sections; illustrative documented LLM, not a claim to be the latest or strongest model |
| [R17] | Hu et al. **LoRA: Low-Rank Adaptation of Large Language Models.** arXiv:2106.09685 | Abstract/overview for parameter-efficient adaptation context; not ported or benchmarked here |
| [R18] | Kwon et al. **Efficient Memory Management for Large Language Model Serving with PagedAttention.** arXiv:2309.06180 | Abstract/overview; separates serving KV memory from this training cache |
| [R19] | PyTorch. **2.14.1 release notes**, 30 September 2026 | Official release page and API record; version and MPS linear-algebra fixes verified |
| [R20] | TorchVision. **0.29.1 release notes**, 30 September 2026 | Official release page/API record explicitly states compatibility with PyTorch 2.14.1 |
| [R21] | PyTorch 2.14 documentation. **MPS backend** | Device/availability route and current macOS requirement; not a target machine test |
| [R22] | PyTorch 2.14 documentation. **MPS environment variables** | Fallback, allocator-watermark and fast-math behavior |
| [R23] | PyTorch 2.14 documentation. **torch.mps** | Synchronization, RNG and memory APIs; sampled-memory limitations are this plan's measurement reasoning |
| [R24] | MLX documentation. **Installation** | Native package/OS route; alternative backend, not installed here |
| [R25] | MLX documentation. **Lazy evaluation** | Timing caveat for a future port |
| [R26] | MLX documentation. **Unified memory** | Backend data-management context |
| [R27] | TorchVision 0.28 source documentation. **torchvision.datasets.mnist** | Dataset loader, mirrors and checksums verified; no dataset rights inferred |
| [R28] | UCI Machine Learning Repository. **Human Activity Recognition Using Smartphones**, dataset 240; DOI `10.24432/C54S4K` | Official data description, subject partition, window overlap and CC BY 4.0 notice; the proposed corruption is not source-derived |
| [R29] | Zalando Research. **Fashion-MNIST** original repository | Dataset structure and MIT license indication; declared image fallback |
| [R30] | PyTorch. **Start Locally / Previous Versions** installation documentation | Mac pip route and the documented 2.13.0/0.28.0 fallback pair; latest-version conclusion instead taken from release records |
| [R31] | Magistri et al. EFC++ preprint, **arXiv:2503.10439v3**, 2025 | Full preprint method inspected; distinguishes preprint history from 2026 journal publication |
| [R32] | Zheng and Shlizerman. **2ndMatch: Finetuning Pruned Diffusion Models via Second-Order Jacobian Matching.** arXiv:2506.05398v2, 28 March 2026 | Full HTML main method, particularly directional Gram matching; adjacent application, not an exact cache-equivalence finding |

### 17.3 Compact search record

Research was conducted through **3 October 2026**. Searches began with the supplied proposal's exact references, then expanded by mechanism rather than social terminology. The investigation followed EFC to its successor and implementation, feature regularization toward functional memory/Jacobian methods, and the routing branch toward current shared-residual and complementary-selection work.

| Search direction / important terms | Sources examined / consequence | Coverage limit |
|---|---|---|
| Feature Fisher, empirical feature matrix, elastic feature consolidation, local versus averaged sensitivity | EFC/EFC++, official matrix and training code; broad local-metric novelty removed | Did not audit every downstream implementation or reproduce published benchmarks |
| Jacobian sketch, Gauss–Newton/functional preservation, backward-compatible representation | Heckel and BCT; separated parameter-space sketches, consumer supervision and activation caching | No claim that a search proves absence of all activation-cache variants |
| Functional regularization, inducing inputs, memorable past, logit replay | FRCL, FROMP, DER; added memory/selection and strong rehearsal explanations | FROMP's arXiv HTML endpoint failed; its proceedings PDF supplied the decisive main-text evidence instead |
| Sobolev training, Jacobian matching, synthetic gradients, subspace memory | Derivative-training and GPM papers; rejected renaming derivatives as a new relationship mechanism | Some more distant architectural neighbors were inspected only at overview level |
| Joint errors, composite action utility, adaptive deferral, switching state space, complementary routing | Current L2D-SLDS v3, FlexRouter, MoCA | Did not independently reproduce their LLM/streaming results; internal-branch port still prospective |
| Apple Silicon training, MPS fallback/SVD correctness, MLX evaluation/memory | Current official releases and backend documentation; CPU-factor/MPS-training route selected | Target Mac unavailable; kernel behavior and full runtime remain execution gates |
| Dataset source, overlap/group leakage, licenses, invertible domain transformations | TorchVision source, original Fashion-MNIST repository, UCI HAR | MNIST primary-source rights unresolved; licensed fallback prevents blocking the essential path |

Several broad search queries returned irrelevant matches; exact titles, primary publication pages, citation following and repository source inspection were used instead of relying on those snippets. The diffusion-model 2ndMatch lead was followed to its main method during the final review: it matches teacher/student directional Gram sensitivities rather than retaining this fixed displacement penalty. [R32] This reinforces the narrow novelty label. Uninspected implementations and later citations remain coverage limits, not evidence that no equivalent exists.

Further searching is most likely to change **the novelty label**, not the necessity of the specified strong baselines or the mathematical local/global distinction. Before submitting a positive result as a new algorithm, repeat forward-citation and exact-mechanism searches against the frozen final implementation and log any new equivalent. This is a research-integrity requirement for the future publication claim, not a deferral of the primary-source research completed for this plan.

[R01]: https://arxiv.org/html/2402.03917v3
[R02]: https://link.springer.com/article/10.1007/s11263-026-03018-4
[R03]: https://github.com/simomagi/elastic_feature_consolidation
[R04]: https://arxiv.org/html/2112.05095
[R05]: https://arxiv.org/html/2003.11942v2
[R06]: https://arxiv.org/html/1901.11356v4
[R07]: https://proceedings.neurips.cc/paper/2020/file/2f3bbb9730639e9ea48f309d9a79ff01-Paper.pdf
[R08]: https://arxiv.org/html/2004.07211v2
[R09]: https://arxiv.org/html/2103.09762v1
[R10]: https://arxiv.org/html/1706.04859v3
[R11]: https://proceedings.mlr.press/v80/srinivas18a/srinivas18a.pdf
[R12]: https://proceedings.mlr.press/v70/jaderberg17a.html
[R13]: https://arxiv.org/html/2601.22538v3
[R14]: https://arxiv.org/html/2609.38585v1
[R15]: https://arxiv.org/html/2605.24048v1
[R16]: https://arxiv.org/html/2505.09388
[R17]: https://arxiv.org/abs/2106.09685
[R18]: https://arxiv.org/abs/2309.06180
[R19]: https://github.com/pytorch/pytorch/releases/tag/v2.14.1
[R20]: https://github.com/pytorch/vision/releases/tag/v0.29.1
[R21]: https://docs.pytorch.org/docs/2.14/notes/mps.html
[R22]: https://docs.pytorch.org/docs/2.14/mps_environment_variables.html
[R23]: https://docs.pytorch.org/docs/2.14/mps.html
[R24]: https://ml-explore.github.io/mlx/build/html/install.html
[R25]: https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html
[R26]: https://ml-explore.github.io/mlx/build/html/usage/unified_memory.html
[R27]: https://docs.pytorch.org/vision/0.28/_modules/torchvision/datasets/mnist.html
[R28]: https://archive.ics.uci.edu/dataset/240/human%2Bactivity%2Brecognition%2Busing%2Bsmartphones
[R29]: https://github.com/zalandoresearch/fashion-mnist
[R30]: https://pytorch.org/get-started/previous-versions/
[R31]: https://arxiv.org/html/2503.10439v3
[R32]: https://arxiv.org/html/2506.05398v2


<a id="appendix-a"></a>
## Appendix A. Complete independently executed CPU reference program

Save this block as `scripts/sanity_checks.py` and run it with Python and PyTorch. It is the actual small program used for Section 13's observations. The executed source file SHA-256 was `cd2458fb54b92b7d77624b46ed85b6e8a121feaf34bbda71a790fbd159641e1c`. It is deliberately a reference/check program, not the complete future training package. Production cache validation, rank selection, dataset loading and access accounting must follow Sections 6–12. No external data or previous artifact is needed for this check. Numerical values can change slightly with library versions; the invariants and tolerances, not exact decimal strings, are the test targets.

```python
"""Small, CPU-only mathematical checks; no dataset training or M3 timing."""
import copy
import json
import math
import platform
import sys
import torch
from torch import nn


def make_factors(consumer, z0, temperature=2.0):
    """Return raw full-rank Fisher factors. CPU float64 reference only."""
    if z0.ndim != 2 or len(z0) == 0 or temperature <= 0:
        raise ValueError("Need nonempty [N,d] features and T > 0")
    model = copy.deepcopy(consumer).cpu().double().eval().requires_grad_(False)
    z = z0.detach().cpu().double()
    factors, jacobians, probs, logps, grams = [], [], [], [], []
    for zi in z:
        zi = zi.clone().requires_grad_(True)
        logits = model(zi)
        if logits.ndim != 1 or logits.numel() < 2:
            raise ValueError("Consumer must return C >= 2 logits for one input")
        J = torch.autograd.functional.jacobian(model, zi, create_graph=False)
        lp = (logits / temperature).log_softmax(-1).detach()
        p = lp.exp()
        centered = J - (p[:, None] * J).sum(0, keepdim=True)
        A = p.sqrt()[:, None] * centered
        _, sv, vh = torch.linalg.svd(A, full_matrices=False)
        r = min(z.shape[1], logits.numel()-1)
        K = sv[:r, None] * vh[:r]
        factors.append(K.detach()); jacobians.append(J.detach())
        probs.append(p); logps.append(lp); grams.append((A.T @ A).detach())
    return {"K": torch.stack(factors), "J": torch.stack(jacobians),
            "p": torch.stack(probs), "logp": torch.stack(logps),
            "G": torch.stack(grams), "z0": z}


def kl_from_logits(logp0, logits, T):
    return T*T*(logp0.exp()*(logp0-(logits/T).log_softmax(-1))).sum(-1)


def run():
    torch.set_num_threads(1)
    torch.manual_seed(20261003)
    T, n, d, C = 2.0, 12, 6, 4
    g = nn.Sequential(nn.Linear(d, 9), nn.Tanh(), nn.Linear(9, C)).double()
    g.eval().requires_grad_(False)
    z0 = torch.randn(n, d, dtype=torch.float64)
    c = make_factors(g, z0, T)
    reconstructed = c["K"].transpose(1,2) @ c["K"]
    reconstruction = float((reconstructed-c["G"]).abs().max())
    assert reconstruction < 1e-12
    direction = torch.randn_like(z0)
    direction /= direction.norm(dim=1, keepdim=True)
    ratios = {}
    for eps in [1e-1, 1e-2, 1e-3]:
        delta = eps*direction
        quad = 0.5*torch.einsum('ni,nij,nj->n', delta, c['G'], delta)
        exact = kl_from_logits(c['logp'], g(z0+delta), T)
        ratios[str(eps)] = {"median_exact_over_quadratic":float((exact/quad).median()),
                           "max_relative_error":float(((exact-quad).abs()/quad).max())}
    assert ratios['0.001']['max_relative_error'] < 0.01
    # Linearized logits are exactly equivalent for an affine consumer, even at finite displacement.
    affine = nn.Linear(d, C).double().eval().requires_grad_(False)
    ac = make_factors(affine, z0, T)
    delta = 3*torch.randn_like(z0)
    linearized = T*ac['logp'] + torch.einsum('ncd,nd->nc', ac['J'], delta)
    affine_error = float((kl_from_logits(ac['logp'], affine(z0+delta), T)-
                          kl_from_logits(ac['logp'], linearized, T)).abs().max())
    assert affine_error < 1e-12
    # Consistent interface permutation.
    perm = torch.randperm(d)
    delta = torch.randn_like(z0)
    before = torch.einsum('nrd,nd->nr',c['K'],delta).square().sum()
    after = torch.einsum('nrd,nd->nr',c['K'][:,:,perm],delta[:,perm]).square().sum()
    permutation_error = float((before-after).abs())
    assert permutation_error < 1e-12
    # Spectrum-preserving per-anchor rotation, including discarded-energy bound.
    max_spectrum_error = 0.0
    max_bound_violation = 0.0
    for i in range(n):
        Q, R = torch.linalg.qr(torch.randn(d,d,dtype=torch.float64))
        Q = Q*torch.where(R.diag() < 0, -1., 1.)[None, :]
        kr = c['K'][i] @ Q
        max_spectrum_error = max(max_spectrum_error,float((torch.linalg.svdvals(kr)-
                                    torch.linalg.svdvals(c['K'][i])).abs().max()))
        k1 = c['K'][i,:1]
        loss_full = (c['K'][i]@delta[i]).square().sum()/2
        loss_trunc = (k1@delta[i]).square().sum()/2
        upper = torch.linalg.svdvals(c['K'][i])[1].square()*delta[i].square().sum()/2
        max_bound_violation = max(max_bound_violation,float(loss_full-loss_trunc-upper))
    assert max_spectrum_error < 1e-12 and max_bound_violation < 1e-12
    # Gradients reach current features but never cache or frozen consumer weights.
    z = z0.clone().requires_grad_(True)
    loss = torch.einsum('nrd,nd->nr',c['K'],z-z0+0.01).square().mean()
    loss.backward()
    assert z.grad is not None and z.grad.abs().sum() > 0
    assert all(v.grad_fn is None and not v.requires_grad for v in c.values())
    assert all(p.grad is None for p in g.parameters())
    # Affine consumers have the same exact nullspace for all finite softmax inputs.
    projectors = []
    for ki in ac['K']:
        _, _, vh = torch.linalg.svd(ki, full_matrices=True)
        B = vh[C-1:]
        projectors.append(B.T @ B)
    common_nullspace_error = max(float((P-projectors[0]).abs().max()) for P in projectors)
    assert common_nullspace_error < 1e-10
    # A truly nonlinear zero-Jacobian failure: logits [z^2, 0], z0=0, z1=1.
    logp = torch.tensor([-math.log(2),-math.log(2)],dtype=torch.float64)
    zero_jacobian_exact_kl = float(kl_from_logits(logp,torch.tensor([1.,0.],dtype=torch.float64),1.))
    assert zero_jacobian_exact_kl > 0.1
    # Exact convex assay: same reference-output drift, different target costs.
    v = torch.ones(2,2,dtype=torch.float64)
    local_delta = torch.tensor([[0.1,1.0],[1.0,0.1]],dtype=torch.float64)
    global_delta = torch.full((2,2),0.1,dtype=torch.float64)
    gs = torch.stack([torch.diag(torch.tensor([1.,0.])),torch.diag(torch.tensor([0.,1.]))]).double()
    def costs(delta):
        return {"new_loss":float(0.5*(delta-v).square().sum(1).mean()),
                "old_quadratic_drift":float(0.5*torch.einsum('ni,nij,nj->n',delta,gs,delta).mean())}
    # Parameter and storage arithmetic; biases included.
    producer = nn.Sequential(nn.Linear(784,256),nn.ReLU(),nn.Linear(256,128))
    head = nn.Sequential(nn.Linear(128,512),nn.Tanh(),nn.Linear(512,512),nn.Tanh(),nn.Linear(512,10))
    pp = sum(p.numel() for p in producer.parameters()); ph=sum(p.numel() for p in head.parameters())
    memory = {"producer_parameters":pp,"consumer_parameters":ph,"total_parameters":pp+2*ph,
              "adaptation_trainable_parameters":pp+ph,"persistent_adam_tensor_bytes":16*(pp+ph)+4*ph,
              "rank8_factor_bytes":4*256*8*128,"private_producer_weight_bytes":4*pp,
              "float32_global_matrix_bytes":4*128*128,
              "byte_matched_uint8_l2_anchors":(256*(784+4*128+4*8*128))//(784+4*128),
              "byte_matched_uint8_replay_anchors":(256*(784+4*128+4*8*128))//(784+8)}
    out = {"environment":{"python":sys.version.split()[0],"torch":torch.__version__,
            "platform":platform.system(),"mps_available":torch.backends.mps.is_available()},
            "seed":20261003,"n_anchors":n,"feature_dim":d,"classes":C,
            "factor_reconstruction_max_abs":reconstruction,"taylor_checks":ratios,
            "affine_linearized_kl_max_abs_error":affine_error,
            "permutation_penalty_abs_error":permutation_error,
            "orientation_spectrum_max_abs_error":max_spectrum_error,
            "rank_tail_bound_max_positive_violation":max_bound_violation,
            "detached_gradient_check":True,"affine_nullspace_projector_max_abs_error":common_nullspace_error,
            "zero_jacobian_exact_kl":zero_jacobian_exact_kl,
            "convex_assay":{"local":costs(local_delta),"global":costs(global_delta)},
            "memory_arithmetic":memory}
    print(json.dumps(out,indent=2))

if __name__ == '__main__':
    run()

```

<a id="appendix-b"></a>
## Appendix B. Configuration, launch, and reporting contracts

### B.1 Initial image configuration

The coding agent creates `configs/image_initial.json` from the following valid JSON. `image_choice` requires E00 to record the actual dataset and its rights status; it is an explicit selection policy, not a hidden runtime guess. The default runnable branch is Fashion-MNIST. Set it to MNIST only after documenting permitted use. All stage configurations embed the resolved choice and dataset hash.

```json
{
  "schema_version": 1,
  "project": "idea1_compatibility",
  "stage": "development",
  "image_choice": "fashion_mnist",
  "selection_reason": "licensed_image_fallback_unless_MNIST_rights_verified",
  "split_seed": 20261003,
  "split_proportions": {"train": 0.8333333333333334, "development": 0.08333333333333333, "diagnostic": 0.08333333333333333},
  "group_duplicates_before_split": true,
  "remove_train_groups_matching_test_hashes": true,
  "protected_test_scoring": false,
  "input_scale": "uint8_divide_255",
  "target_transform": {"kind": "rot90", "k": 1},
  "producer": {"input_dim": 784, "hidden_dim": 256, "feature_dim": 128, "activation": "relu"},
  "consumer": {"hidden_dims": [512, 512], "classes": 10, "activation": "tanh"},
  "new_consumer_init": "deepcopy_source_consumer_reset_optimizer",
  "optimizer": {"name": "adam", "lr": 0.001, "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.0},
  "initial_source_steps": 2000,
  "initial_adaptation_steps": 1500,
  "source_batch": 64,
  "target_batch": 64,
  "anchor_batch": 32,
  "anchor_count": 256,
  "anchor_selection": "seeded_uniform_without_replacement_nested_prefix",
  "temperature": 2.0,
  "local_rank": 8,
  "factor_storage": "raw",
  "trace_normalization": "mean_full_trace_once_in_loss",
  "initial_eta": 0.01,
  "lambda_initial_grid": [0.1, 1.0, 10.0, 100.0],
  "lambda_extension_factor": 10.0,
  "learning_rate_multipliers": [0.31622776601683794, 1.0, 3.1622776601683795],
  "checkpoint_log_interval_steps": 100,
  "initial_development_pairs": 3,
  "retention_allowance_pp": 1.0,
  "factorization_backend": "cpu_float64",
  "training_backend": "mps_if_validated_else_cpu",
  "training_dtype": "float32",
  "mps_silent_fallback": false,
  "access_regimes": ["A-U", "A-B"],
  "comparison_regimes": ["equal_inputs", "equal_required_bytes", "deployment_envelope"],
  "confirmation_allowed": false
}
```

Configuration validation rejects unknown keys, inconsistent class counts, rank above `min(d,C-1)`, invalid batch sizes, unsupported transforms and a confirmation request without an immutable lock. User-specified additional training duration or a smaller measured hardware budget is recorded as a configuration change. The initial horizons do not impose an upper experimental limit.

For B10, the initial two-dimensional rehearsal grid is `label_weight, raw_KL_weight in {0.1,1,10}`; the single-term boundaries are already represented by B08/B09. Expand the axes using the same endpoint/bracketing rule rather than declaring the combined baseline weak from nine uninformative points. B12 starts at adapter width 16 and explores the largest widths permitted by the declared matched-byte budget when needed. For M01/M02, rank sweeps are a development factor with explicit extra tuning cost: image ranks `{1,2,4,8,9}` and HAR ranks `{1,2,4,5}`. Do not present five candidate ranks as a single untuned method against one fixed baseline configuration.

B13/B14 reconstruct **32** clean examples from a deterministic, seed-controlled subset of each 64-example target batch in the initial configuration, matching the nominal anchor batch count. A larger inverse-data batch is a separately counted configuration. All algorithms use the same number of optimizer updates unless the equally applied development convergence rule selects another duration; quality-at-equal-updates and quality-at-selected-cost are then reported separately.

### B.2 HAR overrides

Create `configs/har_initial.json` by copying common optimizer/runtime/statistical fields and replacing all image-specific fields with the following values. Do not leave a dormant image transform in the resolved HAR configuration.

```json
{
  "schema_version": 1,
  "project": "idea1_compatibility",
  "dataset": "uci_har_240_original",
  "split_seed": 20261003,
  "official_training_subject_partition": [13, 4, 4],
  "subject_allocation": "sort_SHA256_of_seed_and_subject_id",
  "protected_official_test_subjects": true,
  "channels": ["body_acc_x", "body_acc_y", "body_acc_z", "body_gyro_x", "body_gyro_y", "body_gyro_z", "total_acc_x", "total_acc_y", "total_acc_z"],
  "window_length": 128,
  "normalization": "source_training_subject_channel_mean_std",
  "fault": {"zero_channels": ["body_gyro_z"], "noise_channels": ["body_gyro_x", "body_gyro_y"], "noise_std_after_normalization": 0.2, "realization": "fixed_hash_seed_per_split_root_window"},
  "producer": {"kind": "har_conv", "feature_dim": 128},
  "consumer": {"hidden_dims": [128, 128], "classes": 6, "activation": "tanh"},
  "anchor_count": 256,
  "target_batch": 64,
  "anchor_batch": 32,
  "temperature": 2.0,
  "local_rank": 4,
  "initial_source_steps": 2000,
  "initial_adaptation_steps": 1500,
  "primary_accuracy": "subject_macro",
  "retention_allowance_pp": 1.0,
  "confirmation_allowed": false
}
```

The inherited 2,000/1,500 HAR horizons are only first timing/learning-curve blocks. E08 calibrates them independently. The manifest stores the actual subject IDs and channel means/stds, not just the algorithm used to obtain them.

### B.3 Confirmation lock generation

`plan-confirmation` must populate, validate and hash all of the following before `confirm` runs: chosen dataset/condition and admissible regimes; candidate and conventional-champion method/config IDs; named baseline/control IDs; code/environment/source/data/split/preprocess/corruption digests; source training and development selection policies; resolved step counts and coefficients; numerical tolerances; seed-generation namespace and exact seed list; planned sample size and its precision calculation; primary/secondary metrics; retention estimand and allowance; simultaneous contrast family and alpha; bootstrap grouping; missing/failure policy; timing protocol and permitted provider-validation budget; and the forecast/launch authorization.

Do not supply an example lock containing made-up final winners, confidence intervals, sample sizes, or future hashes. These quantities are genuinely produced by development and hardware discovery. The **algorithm to resolve them is specified**, so their absence before execution is not an invitation to invent values.

For A-B, the lock states a fixed **post-run validation** allowance: the same predeclared development checkpoint/endpoint schedule and dataset size for each evaluated configuration. No optimizer step may call the consumer. Provider validation and final diagnostics are counted separately from boundary derivative construction. If a stricter application forbids even post-run consumer validation, this plan's retention-aware selection cannot be assumed available; selection must be calibrated on another permitted source or evaluated as an explicitly different access regime.

### B.4 Reporting workflow and claims ledger

Generate the final research report in this order: input/environment provenance; flow of all planned/attempted/completed/failed runs; development selection and cost; protected paired primary effects and retention feasibility; attribution controls; resource/data/architecture frontiers; transfer and boundary cases; and scoped decision. Each table includes units, replication unit, denominator, interval construction, and run IDs. Raw per-example predictions support recomputation but are not independent training replicates.

A machine-readable claim entry contains `claim_id`, `statement`, `evidence_status`, `regime`, `dataset_condition`, `run_ids`, `baseline_ids`, `metric`, `estimate`, `interval_method`, `interval`, `retention_status`, `cost_scope`, `supported_scope`, `unsupported_extrapolations`, and `decision`. Null estimates or unavailable intervals are represented as explicit JSON null with a reason, never replaced with zero. Reports may not infer success from missing failure rows.

The final conclusion should be one of a narrow supported improvement, a simpler-method win, a useful but non-novel implementation tradeoff, a constructed-only effect, or an unresolved comparison with quantified uncertainty. The purpose of the program is to discriminate among these outcomes, not to defend Idea #1 in advance.
