# Forward-only neural adaptation from structured output probes

## Research and proof-of-concept execution plan

**Research cutoff:** 29 October 2026.  
**Target:** one MacBook Pro with an M3 Max; actual memory, macOS, storage, and GPU configuration must be discovered.  
**Deliverable status:** researched experimental design, mathematical checks, and implementation specification. No proposed neural-network benchmark or M3 Max timing has been executed.  
**Primary decision:** determine whether fresh-batch qualification of reusable perturbation directions adds useful efficiency to an established output-model, subspace trust-region method.

This document is the complete handoff. A coding agent should create the repository described below, implement the specified interfaces and experiments, preserve the decision records, and follow the gates. Commands beginning `python -m fopoc` are **proposed project commands to implement**, not existing software supplied with this document. Standard Python, operating-system, and explicitly identified package commands are distinguished from them.

## Contents

1. [Executive assessment](#1-executive-assessment)
2. [Problem, constraints, and hypotheses](#2-problem-constraints-and-hypotheses)
3. [Research findings and novelty assessment](#3-research-findings-and-novelty-assessment)
4. [Mathematical limits and checks already performed](#4-mathematical-limits-and-checks-already-performed)
5. [Selected mechanism and numerical specification](#5-selected-mechanism-and-numerical-specification)
6. [Workloads and assets](#6-workloads-and-assets)
7. [Baselines and implementation fidelity](#7-baselines-and-implementation-fidelity)
8. [Evaluation, tuning, and confirmation](#8-evaluation-tuning-and-confirmation)
9. [M3 Max feasibility and calibration](#9-m3-max-feasibility-and-calibration)
10. [Repository, interfaces, and records](#10-repository-interfaces-and-records)
11. [Ordered work packages and gates](#11-ordered-work-packages-and-gates)
12. [Attribution, robustness, and failure analysis](#12-attribution-robustness-and-failure-analysis)
13. [Interpretation and evidence-triggered alternatives](#13-interpretation-and-evidence-triggered-alternatives)
14. [Audit record and residual limitations](#14-audit-record-and-residual-limitations)
15. [References and search coverage](#15-references-and-search-coverage)
16. [Appendix: executed small-check code](#16-appendix-executed-small-check-code)

## 1. Executive assessment

### 1.1 Verdict

**The broad idea is legitimate but substantially established.** Random weight search, paired directional measurements, structured perturbations, adaptive search distributions, and fitting parameter updates from vector output changes all have close precedents. In particular, residual subspace evolution strategies (RSES) implement almost exactly the proposed residual-change recombination; ensemble Kalman inversion (EKI), derivative-free Gauss–Newton methods, and composite trust-region optimization explain its mathematical structure. PseuZO already uses model-output perturbations for neural-network optimization. Contemporary adaptive low-rank methods make a comparison only with hill climbing or original MeZO inadequate. [R1–R16]

**Pursue one narrower primary experiment:** an output-logit subspace trust-region optimizer whose small dictionary retains directions according to their marginal contribution on a separate, freshly sampled **training** minibatch. Recompute every retained direction's output response at the current weights. Keep a substantial fresh-random component. Compare against an efficient no-retention optimizer and a separate matched-information ablation, strong scalar and vector-feedback alternatives, and conventional gradient training.

The candidate is called **`logit_tr_qualified` only as a repository identifier**. This name is not a novelty claim. Its potential contribution is the specific retention rule and its measured behavior under fair budgets. Whether that combination constitutes publishable methodological novelty remains unresolved. A useful negative result, an equivalence result, or a reproducible empirical improvement in a precisely delimited setting would each be a legitimate outcome.

The main local workload is **Oxford-IIIT Pet classification with a pretrained ResNet-18, a shared fitted classifier head, and a 65,536-parameter additive adapter before a frozen nonlinear suffix**. This preserves the central difficulty of changing network behavior without differentiating through the network. It avoids mistaking an inexpensive linear classifier on cached features for a new training method. The proposal's original suggestion of a small pretrained classifier is retained; a billion-parameter language model is unnecessary to test the selected mechanism. [R26–R29]

### 1.2 What this project could establish

The strongest intended local claim is:

> On the specified fixed pretrained model, adapter, real-image task, data split, and actual M3 Max configuration, the qualified output-model method achieves lower protected-test negative log likelihood at a prespecified optimizer time budget than the strongest eligible comparator selected on development data, and controlled ablations attribute the difference to output-model curvature and/or fresh-batch direction qualification. Query cost, memory, accuracy, calibration, setup cost, and failures are reported alongside it.

A separate matched-query analysis tests information efficiency. If a query benefit disappears after CPU solves, transfers, qualification, and rejected steps are charged, report a query-efficiency result without claiming a practical speed improvement. If it beats forward-only methods but loses to well-tuned gradient training, describe it as a forward-only improvement. If it cannot outperform the shared head or a temperature adjustment, the practical value is correspondingly narrow.

This cannot establish a general replacement for backpropagation, large-model scalability, biological plausibility, energy savings, or superior learning from scratch. The real bottleneck motivating large-model forward-only methods is established in published settings; the existence of that bottleneck on this small laptop workload is **not** established and must be measured. [R2, R3, R9]

### 1.3 Evidence categories used throughout

| Label | Meaning in this handoff |
| --- | --- |
| Verified source finding | Inspected primary paper, source code, official documentation, or asset page; cited in §15. |
| Measured here | The small Linux/NumPy checks in §4.5 only. |
| Deduction | Mathematical consequence or design inference stated with its assumptions. |
| Estimate | Tensor-size calculation or forecast; not an observed M3 result. |
| Proposed experiment | Work the coding agent must execute. |
| Unknown / gate | Information requiring target-machine or experimental discovery before dependent work. |

### 1.4 Execution priorities

First establish algebra, oracle accounting, local numerical support, and adapter headroom. Then reproduce the closest baseline mechanisms. Only then invest in tuning, confirmation, and additional workloads. A learned neural proposal generator, large-scale meta-training, a new network architecture, and a custom Metal implementation are outside the initial path. They become relevant only if an observed limitation specifically warrants them.

## 2. Problem, constraints, and hypotheses

### 2.1 Precise problem

Let a fixed pretrained network and an inserted trainable component define logits

$$
z_W(x)=f_{\theta_0,W}(x)\in\mathbb R^C,
\qquad W\in\mathbb R^{m\times n},\quad q=mn.
$$

For a batch $B$, optimize

$$
J_B(W)=\frac1{|B|}\sum_{(x,y)\in B}
\left[\log\sum_{c=1}^C e^{z_{W,c}(x)}-z_{W,y}(x)\right]
+\frac{\gamma}{2}\|W\|_F^2.
$$

The optional common weight penalty $\gamma\ge0$ is part of the training objective. It is distinct from local-model damping $\lambda$, which controls a proposed step and is not a test-loss term. Mechanism-isolation runs use a common $\gamma$, initially zero. Competitive runs may tune $\gamma$ for every method on the same development protocol. Test negative log likelihood excludes both penalties.

The experimenter can set $W$, run the model on labeled training examples, and observe its logits. They may inspect matrix shapes and reuse frozen forward activations. They cannot differentiate through the primary network during the candidate's adaptation. Computing derivatives of the small known cross-entropy function, an explicit regularizer, or a coefficient-space surrogate is allowed.

This is a **vector-output oracle**, richer than a scalar-loss oracle. A forward pass may expose these outputs at little additional inference cost, but storing, transferring, and processing them is not free. Gains must not be presented as violations of lower bounds for a less informative scalar oracle.

### 2.2 Clarification of the original intent

The original accept/reject mutation algorithm remains a diagnostic baseline. The principal refinement combines measured directions rather than retaining only one winner. The further experimental contribution is a small, explicitly specified direction-selection rule; no neural noise generator is required.

The original broad constraint was no backward differentiation through the primary network. The principal comparison uses the stricter, practically useful boundary of **ordinary forward evaluations only**. Forward-mode automatic differentiation, analytical Jacobian-vector products, and gradient-informed proposals are admitted as clearly labeled diagnostics or separate references, not silently included in this oracle class. Forward-gradient training already exists and is a different no-backpropagation route. [R24]

Pretrained ResNet weights and the common fitted head have a conventional training history. The claim concerns adaptation after these assets are fixed, not the absence of gradients anywhere in the asset supply chain.

### 2.3 Beneficiaries and unverified value assumptions

Potential beneficiaries are researchers and developers adapting a model when its forward implementation is accessible but its backward path is unavailable, undesirable, or disproportionately expensive. This includes some inference runtimes and expensive vector-valued simulators. The laptop PoC directly represents only the model-adaptation case. It does not demonstrate a physical-hardware or deployment bottleneck.

The candidate needs useful low-dimensional structure, sufficiently stable response measurements, and directions whose usefulness persists across examples and nearby parameter states. It also needs savings large enough to repay collecting multiple probes and fitting a model. These are hypotheses, not promises.

### 2.4 Hypotheses and alternatives

| ID | Testable hypothesis and supporting rationale | Strong alternative explanation | Discriminating observation |
| --- | --- | --- | --- |
| H1: local output model | A small direction set predicts enough of a combined update to enable useful steps. Composite DFO establishes this principle under assumptions; the selected ReLU network is locally piecewise affine. [R5–R7] | Steps work only because they are tiny, or benefit is entirely first-order descent. | Actual/predicted reduction and logit-prediction error versus radius; compare scalar slopes, output first-order contraction, and full outer-loss fitting on identical probes. |
| H2: qualified reuse | Marginal usefulness on a separate training batch predicts next-batch usefulness better than same-batch score or unqualified history. | Generic momentum, extra examples, or selection noise explains the gain. | Matched-cost qualified/same-batch/random-history/previous-update controls; next-batch score correlation and loss reduction. |
| H3: competitiveness | The complete method improves protected-test NLL at a frozen time budget over the development-selected strongest relevant forward-only comparator. | More tuning, extra queries, a weak baseline, or a favorable seed explains it. | Locked configurations, independent training replications, full timing, current baselines, and uncertainty intervals. |
| H4: practical tradeoff | An improvement survives setup cost and does not conceal important accuracy, memory, or reliability regressions. | NLL gains are only temperature recalibration; memory saved is already eliminated by a simple gradient implementation. | Head-only and temperature controls, tuned SGD/Adam references, sampled memory maxima and cold-start costs. |
| H5: structure | Selected/reused directions capture more useful response or gradient energy than a fresh isotropic family at the same cost. | Merely drawing rank-one noise is assumed to reduce dimension. | Projection and response-spectrum diagnostics; shared activation-informed proposals; dense and rank-one energy-matched controls. |

A synthetic H1 success is necessary evidence of implementation consistency, not sufficient evidence for H2–H4. A failure on a proxy with no transfer-stable directions cannot by itself reject H2 in every problem. Conversely, if actual useful-direction persistence is measured and qualification still adds no benefit after fair tuning, that is meaningful evidence against the selected rule.

## 3. Research findings and novelty assessment

### 3.1 Prior-art comparison

Costs below describe algorithmic structure. They are not comparable published speed measurements. Let $q$ be trainable parameters, $M$ output/residual coordinates, $k$ probe directions, $r$ a matrix subspace rank, and $K$ a population size. A call must also be charged for its examples and evaluated network segment.

| Work / family | Problem, mechanism, and assumptions | Evidence and material cost | Availability and exact overlap |
| --- | --- | --- | --- |
| SPSA / antithetic ES [R1, R2] | Scalar noisy objective; simultaneous perturbations yield directional gradient estimates. | Two evaluations can form an estimate; variance and sample requirements still depend on geometry/dimension. ES results concern particular RL/distributed settings. | Simple independent implementation. Establishes magnitude-based perturbation learning; the original hill-climber is weaker. |
| MeZO [R3] | Memory-efficient neural fine-tuning, perturb/update with regenerable random seeds. | Published pretrained-language-model results; no general from-scratch advantage follows. Two forward evaluations per basic update. | MIT repository; reuse bookkeeping concepts and validate estimator. Baseline, not the strongest contemporary endpoint. |
| EKI [R4] | Parameter/output ensemble cross-covariances; derivative-free inverse problems and neural learning, including generalized outer losses. | Ensemble span can limit reach; reduced ensemble-space solves avoid an unnecessary output-sized inverse. | Paper equations can be independently implemented. Output feedback and least-squares recombination are established. |
| DFO-GN / DFBGN [R5, R6] | Residual interpolation, changing random subspaces, trust-region acceptance, geometry and point reuse. | Nonlinear least-squares tests and convergence analyses under their stated assumptions. Typical storage $O((M+q)k)$; algebra includes $Mk^2+qk^2+k^3$. | DFBGN pure-Python GPL-3.0 code. Closest established optimizer with subspace safeguards/recycling. |
| Composite DFO / TRFD [R7] | Known convex outer function of an expensive vector map; finite-difference inner model, constrained local solve. | Theory assumes smooth inner Jacobian and suitable model accuracy. A stochastic ReLU adaptation does not inherit that theorem automatically. | GPL-3.0 MATLAB reference; CE Python variant is an adaptation. Full CE on a logit surrogate is not a new optimization principle. |
| RSES [R8] | Stack residual changes from $k$ perturbations, ridge-fit a combination, apply it in parameter space. | Includes small MLP regression. Actual algebra includes $Mk^2+qk+k^3$, beyond the abstract's cubic-solve shorthand. | Paper links Julia code/Zenodo; license not verified here. Nearly exact correspondence to the original Idea C for least squares. |
| PseuZO [R9] | Estimate output Jacobians, contract with outer-loss gradients, maintain EMA/window history. | Neural fine-tuning evidence; storage reduced through seeds and final representations. Published implementation's differentiation and sample-row conventions matter. | Public CUDA-oriented code; no repository-wide license established by the inspected tree. Directly defeats broad novelty of “use predictions rather than one loss.” |
| SubZero / SubZero+ [R10, R11] | Matrix updates in periodically refreshed random low-rank subspaces; newer variant averages one-sided probes and applies Adam in coefficient coordinates. | LLM full/PEFT studies. SubZero+ needs $K+1$ forwards, QR work, and $2r^2$ moment values per matrix. | Original SubZero GPL-3.0 code; newer paper algorithm inspected, distinct released code not established. Strong local mechanism baseline. |
| LOZO / LOZO+ [R12, R13] | Low-rank perturbations; newer method retains subspace columns using measured loss improvement. | Greedy reuse directly overlaps “learn smarter noise.” Selection queries and batch pairing materially affect cost and interpretation. | LOZO+ paper and source inspected; independent implementation required if redistribution permission remains absent. Include a paired-batch correction as an explicitly modified strong comparator. |
| AGZO; AIM-ZO; AdaSub-ZO [R14–R16] | Activation-guided, maintained, or finite-difference-learned projection subspaces. | Current 2026 work makes learned meaningful directions a crowded contribution. Activation extraction, basis maintenance, extra probes, and changing-coordinate moments must be charged. | Primary papers inspected; see §7 for locally relevant forms and fidelity requirements. Activation access must be offered symmetrically. |
| FZOO [R17] | Multi-query one-sided perturbations with population loss normalization; systems parallelism. | Estimator and parallel implementation both matter. A CUDA throughput result is not an M3 measurement. | MIT author code; independent serial estimator plus shared local batching optimizations. Strong simple population comparator. |
| ReLIZO; SGES; ASEBO / Guided ES [R18–R21] | Reuse historical scalar queries or gradient-estimate directions; adapt exploratory/exploitative subspaces. | History, guided distributions, and subspace mixtures are established; usefulness depends on alignment and staleness. | Availability/licenses differ. At least a shared-basis scalar/history control is mandatory; recycling alone is not novel. |
| LOREN; CurvZO [R45, R46] | Scalar-feedback adaptive covariance/curvature and adaptive sparse inclusion probabilities. | LLM experiments under their own protocols; both have paper/code distinctions that matter to a local port. | Mandatory development pilots with explicit corrected/paper variants (§7.9–§7.10). Neither full outputs nor qualification may receive credit merely for beating isotropic noise. |
| BSZO; ZO-LDSD [R47, R48] | Bayesian accumulation of subspace directional observations; forward-only learned Gaussian search mean. | Distinct scalar-history/proposal mechanisms. Cached repeated observations are not independent new data; theoretical alignment assumptions do not automatically describe the practical learned policy. | Inspected broader references, not mandatory for the narrow deletion-score claim. Required reconsideration if the claim shifts to posterior-guided queries or learned search distributions. |
| MeZO-SVRG [R43] | Anchor estimates and scalar minibatch control variates. | Prompting, batch size, and estimator variance change published MeZO behavior; anchors require real data/calls. | Conditional development comparator if the gain is attributed primarily to data reuse or variance reduction. Anchor cost cannot be free. |
| ZO-Finetuner [R22] | Compact learned per-block perturbation policy reused across downstream tasks. | Downstream adaptation is forward-only; optimizer meta-training uses first-order trajectories and differentiates post-update loss through the model. | Author implementation exists. Meta-training expenditure and oracle access cannot be omitted. Not required to train an LLM meta-optimizer for this one-adapter experiment. |
| Forward-gradient and local/closed-form learning [R24, R25] | Change differentiation mode or the learning rule, rather than estimate updates from ordinary inference calls alone. | Relevant to the broader ambition of avoiding backpropagation, with different information and architectural assumptions. | Separate references/boundaries. No claim against all no-backpropagation algorithms follows from this project. |

### 3.2 What can be reused

Reuse PyTorch/torchvision model and dataset implementations, NumPy/SciPy numerical routines, the native DFBGN implementation in a license-compatible environment, and established optimizer equations. Reuse public baseline code only under its actual license. A public GitHub page does not establish an open-source license for all modifications.

Source-specific implementation traps already identified are incorporated in §7: coordinate-state resets for SubZero+, batch pairing and omitted evaluations for LOZO+, and output-history row semantics for PseuZO. This is essential baseline research completed for this plan, not a request that the coding agent discover the landscape anew.

### 3.3 What might be substantively different

The proposed differentiating element is the following **specific, unvalidated rule**: fit a joint logit update on batch A; estimate each raw direction's conditional contribution to that joint update on disjoint training batch C; retain positive contributors only after an actual successful update; remeasure those directions on the next fresh A/C pair. This combines response curvature with an explicitly cross-batch criterion for retaining experiments.

The hypothesis is that this rejects directions that explain incidental errors in one minibatch, while preserving complementary directions that only become useful in combination. A simpler same-batch score, previous-update momentum, activation basis, or LOZO+ selection may do equally well or better. The additional data partition can also waste training information and make the method worse.

The exact rule was not located in the inspected methods. That absence is **not proof of novelty**. Historical stochastic trust regions, validation-based step selection, ensemble methods, adaptive experimental design, and optimization histories make broad novelty implausible. A positive result must be described first as a controlled empirical finding. Any methodological novelty claim needs an additional, focused source comparison against the final implemented rule.

### 3.4 Six explicit novelty questions

1. **What exists?** Every broad ingredient and the basic residual recombination mechanism.
2. **What is reusable?** Existing forward models, low-rank proposals, residual/composite solvers, trust-region safeguards, statistical procedures, and baseline implementations where licensed.
3. **What differs?** At most the specified cross-batch marginal-contribution retention rule and its integration under charged resource budgets.
4. **Why might it matter?** Useful directions may transfer across examples while one-batch improvements are noisy; joint response models can recognize complementarity.
5. **What isolates it?** Pair the initial dictionary, fresh-proposal stream, batches, model, solver, and trust rule; allow the retained slots to differ as required by the policy. Use identical complete dictionaries for one-step information/solver controls. Compare both matched-information attribution runs and independently efficient implementations under equal budgets. Separately vary scalar, per-example-loss, and full-logit feedback.
6. **What invalidates originality?** An earlier method with the same operational rule; an algebraic reduction to an existing rule; or an apparent gain entirely explained by a shared generic optimization. Such a discovery changes attribution and the research claim, not the obligation to report the experiment honestly.

### 3.5 Reconciling apparently conflicting evidence

Distributed RL success, memory-saving pretrained-language-model adaptation, and slow supervised training from scratch address different objectives, hardware, sample costs, and initializations. They do not contradict one another. Likewise, lower iteration counts with a large population may lose at a fixed forward budget. Low-rank restriction can reduce variance within a useful subspace while introducing approximation bias outside it. Reusing history can improve stability while becoming invalid when examples or local geometry change. These tradeoffs motivate separate measurements rather than a single “better than backpropagation” assertion. [R2–R4, R10–R16]

A limited patent-literature screen found on-device perturbation training and ensemble-Kalman weight updates in published applications. This reinforces the crowding of broad mechanisms. It is not a patentability, validity, infringement, or freedom-to-operate assessment. [R41, R42]

## 4. Mathematical limits and checks already performed

### 4.1 The dimensionality obstacle is real but conditional

For $L(w)=\frac12\|w\|^2$ and a uniform unit direction $u$, the optimal line step removes $(u^Tw)^2/2$. Since $E[uu^T]=I/q$, expected fractional reduction is $1/q$. An exact projection onto a uniform $k$-dimensional subspace removes $k/q$ in expectation. This is a deduction for the spherical quadratic, not a universal neural-network lower bound.

Collecting $2k$ central probes to get a $k$-direction step does not automatically improve progress per call. An advantage requires favorable curvature, alignment, transferable history, non-scalar information that is actually used, or a systems benefit. If the accessible gradient energy remains spread across all $q$ coordinates, a small probe budget remains a severe constraint.

### 4.2 Rank-one noise is not automatically low-dimensional optimization

For independent standard Gaussian vectors $a\in\mathbb R^m,b\in\mathbb R^n$, $u=\operatorname{vec}(ab^T)$ has covariance $I_{mn}$. The distribution uses a compact factorization but its span is the full matrix space. For a linear objective with matrix gradient $G$,

$$
E\left[\|\langle G,ab^T\rangle ab^T\|_F^2\right]
=(m+2)(n+2)\|G\|_F^2.
$$

The corresponding dense-Gaussian expression is $(mn+2)\|G\|_F^2$. Thus unnormalized random rank-one noise can have worse estimator variance. With unit-norm factors, $E[uu^T]=I/q$; correct any scalar gradient-estimator normalization accordingly. These moment identities are deductions, not measured evidence of a neural-network advantage.

Distinguish **a rank-one probe**, **a persistent low-dimensional parameterization**, and **a learned well-aligned subspace**. The selected method uses the first and tests the third; its accumulated adapter remains a dense matrix.

### 4.3 Where full outputs can add information

Let $S_i\approx J_z u_i$, and let $e=\nabla_z\ell(z,y)$. In the small-probe limit,

$$
\frac{\ell(z(W+hu_i)) - \ell(z(W-hu_i))}{2h}
\longrightarrow e^T S_i.
$$

An output estimator contracted once with the same outer gradient has the **same first-order directional information** as scalar differences. The additional usable object is, for example,

$$
H_{ij}^{\rm out}=\frac1{|A|}\sum_{b\in A}
S_{b,:,i}^T[\operatorname{diag}(p_b)-p_bp_b^T]S_{b,:,j},
$$

or the full function $\ell(z+S\alpha)$. This captures cross-direction curvature of the outer loss. In general it omits inner-network second derivatives. **A favorable boundary of W1 matters:** one additive W followed by frozen affine/BatchNorm/ReLU operations gives piecewise-affine logits in W. Within a fixed activation region, exact output responses therefore recover the true projected CE Hessian; finite probes or candidate steps crossing activation boundaries need not. Measure activation-pattern changes and test the smooth nonlinear W0/W2 cases before extrapolating beyond this architecture. Prediction vectors cannot identify unprobed parameter directions without additional assumptions.

### 4.4 Equivalence and adversarial limiting cases

For least squares, response fitting gives

$$
\delta w=-Q(S^TS+\lambda I)^{-1}S^Tr
=-QS^T(SS^T+\lambda I)^{-1}r.
$$

The identity as written assumes $\lambda>0$; for zero damping use a stated minimum-norm pseudoinverse limit, not an undefined inverse. The second form exposes the ridge/ensemble-covariance relationship. Use the small $k\times k$ form when appropriate; do not make an EKI baseline expensive by unnecessarily inverting an $M\times M$ matrix.

For nonorthogonal directions, physical step length is $\alpha^TQ^TQ\alpha$, not $\|\alpha\|^2$. A coefficient-only ridge changes under rescaling or duplication of basis vectors. The candidate therefore whitens the parameter-space Gram matrix.

Central probes cannot discover a purely even local response. For $f(w_1,w_2)=w_1w_2$ at zero, separate-axis first-order responses vanish, and simultaneous signed perturbations have identical outputs. The same problem occurs when both factors of a bilinear adapter start at zero. This is a mechanism boundary, not a reason to fabricate a gradient. Use an additive adapter here; retain the even-response example as a negative control.

The retention score also has an instructive limit. For a quadratic coefficient model $q_C(\alpha)=c+b_C^T\alpha+\tfrac12\alpha^TH_C\alpha$, deletion gives $v_i=-\alpha_i[\nabla q_C(\alpha)]_i+\tfrac12\alpha_i^2H_{C,ii}$. At an unconstrained, undamped same-batch minimizer, the first term vanishes and all nonzero components with positive diagonal curvature receive positive scores automatically. Thus positive same-batch saliency is weak evidence of reusable direction quality. The meaningful experiment is prediction of benefit on later fresh batches relative to simpler retention policies.

There is an even sharper exhaustion case. Exact undamped minimization of a spherical quadratic in an orthonormal span Q gives $g_{after}=(I-QQ^T)g_{before}$; every just-used direction then has zero next slope, despite a positive old-point deletion score $v_i=\alpha_i^2/2$. A separate batch of the same deterministic objective does not rescue it. The rule needs **post-update persistence**, for example because the constrained/damped step leaves useful work in that span or geometry changes favorably. Remeasuring next iteration avoids stale arithmetic but does not recover the wasted queries. This is a required W0 boundary test and a serious argument against the rule, not an implementation defect.

### 4.5 Checks actually run while preparing this plan

Environment: Linux x86-64, Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0, float64, NumPy generator seed `20261029`. No PyTorch or Apple GPU was used. These results validate algebra and a small numerical solve only.

| Check | Result actually observed | Interpretation |
| --- | --- | --- |
| 100,000 exact random-line steps on a 64-dimensional spherical quadratic | Mean removed fraction 0.015556742; predicted 0.015625; Monte Carlo SE 0.000067995 | Consistent with $1/q$, not evidence of a new optimizer. |
| Reduced ridge versus equivalent ensemble-space expression, $q=60,M=45,k=7$ | Relative step difference $7.79\times10^{-16}$ | Algebraic equivalence verified numerically. |
| Physical-Gram regularization under an invertible basis change | Relative step difference $7.37\times10^{-16}$ | Parameter-step invariance verified. |
| Coefficient-identity ridge under the same basis change | Relative step change 0.25005 | A concrete counterexample to ignoring basis geometry. |
| Scalar finite difference versus contracted output response in a small tanh network | Difference fell from $2.75\times10^{-6}$ at $h=10^{-1}$ to $2.47\times10^{-12}$ at $h=10^{-4}$ | First-order equivalence, not extra first-order information. |
| Six-coefficient CE ball-constrained solve | SLSQP success in 9 iterations; step norm 0.40000000000047 for radius 0.4; KKT residual $5.61\times10^{-8}$ | The proposed small CPU solve is implementable; target workload still needs testing. |
| Square-root CE residual identity | Absolute objective difference $2.22\times10^{-16}$ | Established residual solvers can optimize the same CE objective. |
| LOREN row-wise covariance score, m=7,n=5,rho=0.4,epsilon=0.03 | Analytic score versus a central finite difference of log density at step 10^-5: relative error $5.22\times10^{-11}$; covariance square-root identity error $4.43\times10^{-16}$ | Checks the sign/scaling of the explicitly corrected baseline, not its training quality. |

To reproduce the checks independently: use the dimensions and formulas above, a seeded Gaussian matrix generator, `scipy.special.logsumexp/softmax`, and `scipy.optimize.minimize(method="SLSQP", jac=True)` with an analytic coefficient gradient and the constraint $0.4^2-\|\beta\|^2\ge0$. Test exact algebra with relative error below $10^{-10}$ in float64; check the quadratic mean within five Monte Carlo standard errors. Do not treat matching these particular random realizations as the only valid correctness test.

## 5. Selected mechanism and numerical specification

### 5.1 State and input conventions

The primary trainable tensor is $W\in\mathbb R^{256\times256}$, initially zero. The optimizer state contains `W`, trust radius `Delta`, probe radius `h`, damping policy, a list of retained rank-one direction factors/identifiers, RNG state, sampler state, and budget counters. Every step draws disjoint training batches A and C. Initial logical sizes are 64 and 64 for capability calibration; the final sizes are selected by §8–§9 before confirmation.

A is used to fit the update. C is used for the actual-update safeguard and direction qualification. Both are training data. Neither is the development set or protected test set. They may be revisited across iterations as part of stochastic optimization, but their indices and realized preprocessing must be logged.

Within each iteration sample A∪C without replacement, requiring $|A|+|C|\le N_{train}$; split its ordered IDs into the two disjoint portions. Across iterations use a keyed permutation sampler with an explicit epoch/cursor and boundary rule: start a new permutation when a full union will not fit, log the unused tail, and do not merge overlapping epoch fragments. Dataset IDs and tensor row positions are different fields.

All frozen BatchNorm statistics remain fixed; dropout is disabled; preprocessing is deterministic in the core experiment. The same examples, order, and any stochastic state are used for every plus/minus/base/candidate comparison within an iteration.

### 5.2 Directions and probe normalization

A fresh rank-one direction is $D_i=a_i b_i^T$, where independent Gaussian vectors are normalized to unit Euclidean norm. Consequently $\|D_i\|_F=1$. Store the factors and their seed; use a deterministic sign convention for identity comparisons. Dense Gaussian directions normalized to the same Frobenius norm are a required control.

Retained directions occupy at most a fraction $\rho$ of the $k$ slots, initially $\rho=1/2$. At least one fresh direction is always present. This mixture is a hypothesis-bearing policy, not a proven optimal exploration rate. Tune the permitted fraction on development data, including zero; a winner at zero is evidence against qualification.

Before model queries, form $G_{ij}=\langle D_i,D_j\rangle_F$, computable for rank-one factors as $(a_i^Ta_j)(b_i^Tb_j)$. Reject exact duplicates, including sign duplicates. Use float64 eigendecomposition; require $g_{min}/g_{max}>\tau_G$, initially $\tau_G=\sqrt{\epsilon_{64}}$, and inspect sensitivity to this numerical threshold on W0. Remove the newest dependent retained direction first, then redraw fresh proposals. Allow at most 10k redraws before reducing effective dimension; log retries and actual dimension. This cap prevents an implementation loop, not a scientific exclusion. Do not silently count unusable directions as effective probes.

For **fresh isotropic** directions with covariance $I/q$, the unbiased linear-limit estimate is $(q/k)\sum_i s_iD_i$. This statement does not apply to an adaptive retained dictionary. For the same-dictionary scalar control use its physical projected gradient $Q(Q^TQ)^{-1}s=\widetilde Q R^Ts$, with a tuned step or a constrained linear model; do not call it an unbiased full-space gradient. Native Gaussian, Rademacher, and low-dimensional baseline conventions retain their own scaling. The coefficient-fitting method needs no gradient-estimator multiplier because its coefficients are solved directly.

### 5.3 Measure the local response

Compute base logits on $B=A\cup C$, then

$$
S_{b,:,i}=\frac{z_{W+hD_i}(x_b)-z_{W-hD_i}(x_b)}{2h}.
$$

Center base logits and each response across the class axis. CE is invariant to a common logit offset; centering prevents a class-independent response from masquerading as useful output energy. Store the response tensor with shape `[B,C,k]`, sample IDs, class order, parameter-state hash, and `h`.

Always construct each perturbed tensor from an exact saved CPU float32 base: calculate `W_base + sign*h*D_i` in CPU float64, then round once to float32 and transfer. Check the realized plus/minus displacements using the same relative rounding guardrail as candidates. Restore `W_base` after every evaluation, including exceptions. Do not rely on a long floating-point `+h,-2h,+h` cycle. The authoritative committed state must be bitwise unchanged after a rejected trial within the same environment.

Probe with float32 on MPS, explicitly transfer logits to CPU while preserving float32, then cast on CPU to float64 before subtraction. Use `logits.detach().cpu().double()`, with separate operations, and the transfer smoke test in §9. This reduces additional arithmetic error but cannot undo rounding already present in float32 predictions. Calibrate `h` against both repeat-evaluation noise and nonlinear model error (§9).

### 5.4 Physical-norm coefficient model

Flatten directions into columns of $Q\in\mathbb R^{q\times k}$. With $G=Q^TQ=V\operatorname{diag}(g)V^T$, let

$$
R=V\operatorname{diag}(g^{-1/2}),\qquad
\widetilde Q=QR,\qquad \widetilde S=SR.
$$

Then $\widetilde Q^T\widetilde Q=I$, and a step is

$$
\delta w=\widetilde Q\beta=Q\alpha,\quad \alpha=R\beta,
\quad \|\delta w\|=\|\beta\|.
$$

Solve the small convex problem

$$
\min_{\|\beta\|\le\Delta}
m_A(\beta)=\operatorname{CE}_A(z_A+\widetilde S_A\beta)
+\frac\gamma2\|w+\widetilde Q\beta\|^2
+\frac\lambda2\|\beta\|^2.
$$

Use CPU float64 SLSQP with an explicitly supplied analytic gradient, initialized at zero. This is a conventional small constrained optimization, not differentiation through the network. The gradient is

$$
\nabla m_A=\frac1{|A|}\sum_b\widetilde S_b^T(p_b-y_b)
+\gamma\widetilde Q^T(w+\widetilde Q\beta)+\lambda\beta.
$$

The Hessian for diagnostics, curvature controls, and optional Newton acceleration is

$$
\nabla^2m_A=\frac1{|A|}\sum_b\widetilde S_b^T
(\operatorname{diag}(p_b)-p_bp_b^T)\widetilde S_b
+(\gamma+\lambda)I.
$$

Compute the softmax covariance action without materializing a `[C,C]` tensor per example: weighted response Gram minus the outer product of the weighted response mean. Never form a full $q\times q$ Hessian.

Solve in the unit-ball variable $v=\beta/\Delta$, with analytic chain-rule gradients, `ftol=1e-10`, and a 200-iteration safeguard initially. These are numerical tolerances, not scientific stopping rules. Require finite values, $\|v\|\le1+10^{-7}$, and model decrease. Project only that tiny numerical overshoot back to the unit ball and recompute all modeled quantities. A radius-dependent absolute tolerance must not admit an oversized step when Delta is tiny.

For stationarity, let $f(v)=m_A(\Delta v)$, $L_v=\max(\|\nabla^2 f(0)\|_2,\sqrt{\epsilon_{64}}\max(1,|f(0)|))$, and $r_{PG}=\|v-\operatorname{Proj}_{\|\cdot\|\le1}(v-\nabla f(v)/L_v)\|$. Initially require $r_{PG}\le10^{-6}$; validate this tolerance against independent W0 solutions and loss accuracy, then freeze it. This is a scaled projected-gradient residual, not a claimed bound on objective error.

If SLSQP fails, retry once with a 1,000-iteration safeguard and the same objective. If it still fails, use a logged feasible Cauchy fallback: in v coordinates, take $d=-g/\|g\|$ at zero, begin with $t=1$, and halve t until $f(td)\le f(0)+10^{-4}t g^Td$. Stop after 60 halvings, floating-point underflow, or an unchanged realized float32 state, whichever occurs first. A zero gradient or inability to find an observable feasible decrease is a stall. Charge solver retries; evaluate the fallback through the same actual-forward acceptance rule. Do not silently replace a failed solve or discard that run.

After constructing the float32 proposed W, compute its **realized** delta from the saved float32 base in CPU float64. Check the physical norm and modeled predictions using the realized step's dictionary projection; log any component outside the intended span caused by rounding. If the norm exceeds the radius beyond the measured rounding envelope, rescale once and recompute, or reject as numerically invalid. If most components round away, invoke the precision-stall gate rather than pretending the double-precision step was committed.

Define damping as $\lambda=\lambda_{rel}\max(\operatorname{tr}(H_0)/k,h_{floor})$, where $H_0$ is output curvature at zero. `h_floor` is a numerical floor chosen from the calibrated response scale, recorded in the configuration. Include zero and positive damping in development tuning. The physical weight penalty $\gamma$ is handled separately.

### 5.5 Candidate acceptance and radius changes

Let $\widehat J_A(\beta)$ be the modeled training objective without the damping term. Define

$$
\mathrm{pred}_A=J_A(W)-\widehat J_A(\beta),\quad
\mathrm{ared}_A=J_A(W)-J_A(W+\delta W),
\quad r_A=\mathrm{ared}_A/\mathrm{pred}_A.
$$

The numerator and denominator use **the same A examples**. Do not divide an improvement measured on C by a prediction fitted on A and call that trust-region agreement.

If `pred_A` is below the calibrated numerical comparison tolerance, do not apply the candidate. Otherwise evaluate the actual candidate once on A∪C. Accept only if `ared_A >= 0.1*pred_A` and `ared_C >= -tol_C`, with `tol_C` derived from repeated identical forward evaluations, not from a statistical significance test. This is a training heuristic; repeated acceptance decisions do not produce confidence guarantees.

On rejection, retain the committed weights, halve `Delta`, and clear the qualification dictionary. On acceptance, if `r_A >= 0.75` and the step uses at least 80% of the radius, double `Delta` up to the development-calibrated cap; otherwise retain it. Keep `h` independent of radius within the calibrated reliable range. If no reliable radius remains or all differences become numerically invisible, record a stalled/precision-limited state and invoke the diagnostic gate rather than shrinking indefinitely.

The numerical constants are conventional starting policies. Freeze them after development; report radius/damping sensitivity. No convergence theorem for the cited deterministic methods is asserted for this stochastic, qualified variant.

### 5.6 The proposed direction-retention rule

After an accepted step, use the **old base point's** measured C responses and the realized step's projected raw coefficients. Explicitly, $\delta_{real}=\operatorname{vec}(\operatorname{float64}(W_{trial})-\operatorname{float64}(W_0))$, $\alpha_{real}=G^{-1}Q^T\delta_{real}$, and $e_\perp=\delta_{real}-Q\alpha_{real}$. The rounding gate screens e_perp; the score attributes only the measured in-dictionary component. In the formulas below write $\alpha=\alpha_{real}$ and $w=\operatorname{vec}(W_0)$. Define

$$
\widehat J_C(\alpha)=\operatorname{CE}_C(z_C+S_C\alpha)
+\frac\gamma2\|w+Q\alpha\|^2,
$$

$$
v_i=\widehat J_C(\alpha-\alpha_i e_i)-\widehat J_C(\alpha).
$$

This asks whether component $i$ helps the joint fitted update on separate examples, holding other coefficients fixed. It is a **marginal deletion score**, not a Shapley value and not a causal importance guarantee. Do not re-optimize the other coefficients separately for each deletion in the primary version.

Retain at most `floor(rho*k)` raw directions with `v_i > tol_score`, sorted by decreasing score, ties broken by direction ID. `tol_score` is a numerical tolerance from loss arithmetic, not a selectively chosen positive-effect threshold. Store the original factors, not transformed response columns. Zero-coefficient directions have zero score and are not promoted. On the next step, draw new A/C batches and remeasure every retained direction at the new W. **Never reuse an old output row as though it belonged to a new example.**

The score is invariant to flipping a raw direction's sign with the corresponding coefficient. It is deliberately tied to the sampled raw dictionary and need not be invariant to arbitrary rotations of that dictionary. Gram whitening makes the *physical step* geometrically sound; it does not turn this selection heuristic into a basis-independent theorem.

Deletion can increase the physical norm when raw directions partly cancel. Record every deleted-step norm and whether it exceeds Delta; the primary score remains the explicitly defined surrogate evaluation, not a comparison of two feasible trust-region optima. Test a radius-clipped deletion variant only as a separately named development ablation. Use realized projected coefficients after rounding; the actual-candidate acceptance objective uses the exact known penalty at the realized W, while deletion scores use their declared surrogate states.

### 5.7 Complete primary iteration

```python
# Pseudocode. All oracle.logits calls pass through the cost ledger.
# Indexing by A/C below means their recorded row slices, not arbitrary IDs.
W0 = committed_W.clone()
A, C = sampler.next_disjoint_training_batches()
D = fill_with_retained_and_fresh_directions(state, k)
D = remove_dependence_and_replace_before_querying(D)
z0 = center_classes(oracle.logits(W0, A + C, purpose="base").logits)
for i, direction in enumerate(D):
    Wp, Wm = rounded_validated_probe_states(W0, direction, h)
    zp = oracle.logits(Wp, A + C, purpose="probe_plus").logits
    zm = oracle.logits(Wm, A + C, purpose="probe_minus").logits
    S[:, :, i] = center_classes((zp - zm) / (2*h))  # already CPU float64
restore_exactly(W0)
Q, R = physical_gram_whitening(D)
beta, solve_status = solve_small_ce_model(z0[A], S[A], R, W0,
                                        gamma, damping, Delta)
valid = False
pred_A = -float("inf")
if solve_status.has_finite_feasible_candidate:
    alpha = R @ beta
    Wtrial, alpha_real, valid = round_and_validate_realized_step(W0, D, alpha)
    if valid:
        pred_A = modeled_reduction_without_damping(
            A, alpha_real, exact_regularizer_at=Wtrial)
if not valid or pred_A <= numerical_tolerance:
    reject_and_shrink_and_clear_history()
else:
    z_trial = oracle.logits(Wtrial, A + C, purpose="candidate").logits
    if sufficient_actual_A_decrease(z_trial) and C_safeguard(z_trial):
        commit(Wtrial)
        state.retained = positive_marginal_deletion_scores_on_C(
            W0, z0[C], S[C], D, alpha_real)  # old base even after commit
        update_radius_from_A_agreement()
    else:
        restore_exactly(W0)
        reject_and_shrink_and_clear_history()
ledger.commit_iteration(state, all_attempted_calls, solve_status)
```

An ordinary completed iteration uses $(2k+2)(|A|+|C|)$ suffix-example evaluations: base, paired probes, and actual candidate. A no-candidate iteration uses $(2k+1)(|A|+|C|)$. Additional retries, diagnostics, or line searches must be counted explicitly. A new batch invalidates any previously cached base logits.

### 5.8 Cheapest complete distinguishing experiment

Use a deterministic nonlinear residual network with a known low-dimensional active subspace and two independently drawn example batches. Compare the exact same direction dictionary for scalar first-order steps, residual ridge recombination, and full-logit fitting. Compare qualification trajectories with paired initial/fresh proposals while allowing retained slots to differ. Include an isotropic quadratic where there should be no special advantage and a rapidly rotating active subspace where history should fail. This checks whether the proposed effect can exist and whether the harness can identify its cause.

Then use the real internal-adapter workload. Synthetic success alone cannot validate fresh-batch transfer on real images. The cheapest *decisive* real-data test is the primary candidate versus its no-retention counterpart and the strongest development-selected comparator, after the opportunity and fidelity gates have passed.

## 6. Workloads and assets

### 6.1 W0: analytical and small nonlinear diagnostics

Implement these before downloading the vision assets:

| Fixture | Construction | What it can establish |
| --- | --- | --- |
| Spherical quadratic | $L(w)=\|w\|^2/2$; vary $q$; exact optimal line/projection solutions. | Correct direction scaling, query counts, and the absence of a universal dimension-free gain. |
| Known residual map | $r(w)=Aw-b$, with Gaussian orthogonal singular vectors and controlled singular values. Include isotropic and ill-conditioned cases. | Ridge/EKI/RSES equivalence; physical-Gram invariance; useful versus irrelevant directions. |
| Active-subspace residual | $r_x(w)=g_x(V^Tw)-y_x$, with fixed orthonormal V and sample-specific maps, e.g. $g_x(t)=A_x\tanh(B_xt+c_x)$, drawn from a recorded distribution; targets generated at a shared teacher parameter. | Whether a mechanism can exploit known low-dimensional structure across genuinely different example responses. Report active and ambient dimensions separately. |
| Drifting active subspace | Rotate $V$ between batches by a known angle; include zero and rapid drift. | A failure regime for stale history and an interpretable qualification challenge. |
| Small tanh classifier | Gaussian inputs, fixed seeded weights/targets, a small trainable matrix, exact CPU derivative oracle available only in tests. | Response finite-difference accuracy, coefficient gradient/Hessian, and interaction with nonlinear logits. |
| Degenerate responses | Duplicate directions, class-offset-only logits, zero response, bilinear-at-zero, and artificially independent plus/minus noise. | Whether the implementation rejects false signals or breaks under known pathologies. |

Use dimensions sufficient to express each mechanism: e.g. ambient $q\in\{64,256,1024\}$, active rank 4 or 16, and several probe counts below and above that rank. These are inexpensive diagnostic scales, not a claim about production networks. Generate independent fixtures from recorded seeds and include zero-noise cases before adding controlled noise. Compare algorithms at the same actual oracle-evaluation budget, including candidate checks.

### 6.2 W1: primary real-image adaptation

**Dataset.** Oxford-IIIT Pet, 37 breed categories. Download the original images and annotations through the official source or the verified torchvision loader. The official dataset page states CC BY-SA 4.0; preserve attribution and upstream notices. The inspected loader supplies archive MD5 checks `5c4f3ee8e5d25df40f4fd59a7f44e54c` for images and `95a8c909bbe2e81eed6a22bccdf3f68f` for annotations. Also compute SHA-256 of the actual files. Expected official manifest counts are 3,680 `trainval` and 3,669 `test`; verify rather than invent local counts. [R26, R28]

Use category labels mapped to 0–36 and decode images as RGB. Hash the decoded width, height, channel order, and pixel bytes together. Preserve official IDs and class names. Before any outcome analysis, exclude unreadable images with a logged decoder error and remove training images whose decoded-pixel hash exactly duplicates a protected-test image. Preserve the official valid test population, including its multiplicities, and group identical test images for any image-level resampling. Record all exclusions and their effect on counts. Near-duplicate and ImageNet overlap remain limitations unless separately resolved; do not claim they have been ruled out.

Within the remaining official `trainval`, form exact decoded-pixel duplicate groups **before** splitting. Exclude conflicting-label groups from trainval with a recorded reason. Within each class, sort groups by `SHA256("fopoc-v1|" + pixel_group_hash)` and assign whole groups to development until reaching at least ceil(0.2×class_image_count); put remaining groups in training. Record actual proportions and require both partitions to contain every class. This prevents duplicates from leaking across training/development while preserving a deterministic approximately 20% split. Store manifests; do not resplit for favorable outcomes. All head fitting, A/C sampling, activation statistics, and proposal learning use training only. Development selects configurations. Official test metrics are computed after preregistration and confirmation training.

**Pretrained model.** Use the explicit `torchvision.models.ResNet18_Weights.IMAGENET1K_V1` enum, not a drifting `DEFAULT`. Its inspected source points to `resnet18-f37072fd.pth`; record the resolved download URL and full local hash. Use its official deterministic transforms: resize shorter edge to 256, bilinear interpolation, 224 center crop, rescale to [0,1], ImageNet channel normalization. Record the installed transform representation, including antialias behavior. [R27, R29]

**Architecture.** Run the pretrained prefix through `layer3` to obtain `[B,256,14,14]`. Insert a biasless 1×1 residual map

$$
h'_{b,:,s}=h_{b,:,s}+Wh_{b,:,s},\qquad W_0=0.
$$

Then run the unchanged `layer4`, global average pool, and a common 37-class linear head. Only W is trainable during primary adaptation. Set the entire backbone and head to evaluation mode, including frozen BatchNorm. The adapter contains 65,536 free parameters. Rank-one probes do not impose a permanent low-rank capacity restriction on accumulated W.

**Common head.** First compute the unadapted pooled 512-dimensional features from training/development images. Fit a multinomial logistic head on training features using CPU float64 L-BFGS with an explicit CE gradient and an L2 penalty on head weights, not its bias. Select its penalty on development data with the bracket-and-refine rule in §8. Verify solver stationarity and compare against a separately implemented standard logistic solver if convergence is suspect. Freeze the chosen head. Record the fitted parameters, preprocessing, fit iterations/time, and development choice.

This strong fitted head is both the shared initializer and a practical no-adapter baseline. Its preparation costs are shared but visible. Gradient-based head fitting is allowed as preparation; no candidate run can claim a completely gradient-free asset history. Include a fitted temperature-only calibration of this head, chosen on development data, as another cheap practical reference.

**Opportunity gate.** Tune conventional SGD/momentum and Adam on exactly W, allowing gradients through frozen `layer4`. Check whether they improve development performance beyond the head/temperature controls. If only training loss improves, the adapter may overfit. If neither training nor development improves, inspect gradients, adapter placement, regularization, and solver correctness. If adequate sensitivity still reveals no adaptation opportunity, declare W1 unsuitable for a useful-adaptation claim. A predeclared next location is after `layer2` with a 128×128 additive adapter and the corresponding longer frozen suffix; this is a new workload configuration requiring a recorded amendment and fresh development calibration before test access.

**Caching.** Cache the frozen layer3 features in float32 memory-mapped arrays for every method. Never cache layer4 outputs as substitutes for forwards after W changes. Store caches as ordinary CPU arrays/tensors: inference-mode tensors can break the gradient baseline when autograd needs to save an input. Use `no_grad` for cache extraction or reload NumPy arrays into ordinary tensors outside inference mode. [R32]

The deterministic crop is a deliberate controlled workload. It is not a claim to the strongest possible vision training recipe. A later shared augmentation test must recompute or separately cache all specified transforms and charge that cost.

### 6.3 W2: independent boundary test from scratch

Use the original MIT-distributed Fashion-MNIST dataset: 60,000 training and 10,000 test 28×28 grayscale images, ten classes. Verify the original archive checksums and source revision; apply the same exact-duplicate isolation rules as W1 and assign whole training duplicate groups to an approximately 10% per-class development split using independent fixed salt `fopoc-fashion-v1`. Preserve the valid official test population and log exclusions. Do not mix it with W1 test decisions. [R30]

The specified boundary model is `Flatten → Linear(784,64) → tanh → Linear(64,10)`, with biases, 50,890 trainable parameters. Inputs are float32 pixels divided by 255. Use one recorded Xavier initialization per run pair, no dropout or normalization, and no data augmentation in the core boundary test. Train all parameters. For structured probes, draw rank-one directions per weight matrix and normalized independent directions for biases, giving each of the four blocks equal expected squared probe norm; normalize the concatenated direction. All methods share the mask and parameterization.

The equal-energy four-block probe has within-block covariance $I/(4q_j)$, where q_j is that block's size, not global covariance I/q. Fresh scalar estimators using this proposal require block correction $4q_j$; alternatively use a separately named globally isotropic proposal. Test the linear-limit expectation explicitly.

Before W2 fidelity checks, every matrix-only baseline must declare its bias handling. The default local extension is a Gaussian direction for each bias, sharing the method's scalar probe slope and updating the bias with that slope times its direction under the stated normalization; charge bias optimizer/history state. A population method accumulates those contributions with its same population coefficients. Label this vector extension in the method card and validate it independently. Do not freeze biases only for LOZO/SubZero variants. A common bias-free mask is a possible amended workload, applied to every method before tuning.

This is a boundary test, not the primary claim. A favorable result motivates larger and deeper from-scratch tests; an unfavorable result documents failure at this scale. Neither establishes what happens in billion-parameter pretraining. If W1's model/data terms cannot be satisfied, W2 is an explicit narrower fallback; the report must state that pretrained-model adaptation was not demonstrated.

### 6.4 Asset and licensing policy

Torchvision's BSD-3-Clause code license is not a blanket license for all pretrained weights or their training data. Its own README distinguishes these. Record the checkpoint's notices and the research-use assumptions; downloading ImageNet itself is unnecessary. Do not redistribute model weights or datasets merely because they were downloadable. If relevant terms are unclear or incompatible, stop the dependent asset stage and use the clearly stated fallback. [R29]

Save code, configurations, hashes, manifests, scalar results, and permitted artifacts. Verify licenses for actual reused code, including GPL components and per-file headers. Do not copy an unlicensed repository into the deliverable and label it MIT. Mathematical reimplementations must still cite their source methods.

## 7. Baselines and implementation fidelity

### 7.1 Comparison tiers

**Mandatory development portfolio for a contemporary forward-only competitiveness claim:** tuned paired Gaussian/Rademacher ZO with SGD/momentum/Adam and multiquery options; SubZero+; paired LOZO+; AGZO; AIM-ZO; AdaSub-ZO; FZOO; PseuZO classification adaptation; LOREN's source and corrected row-wise variants; CurvZO's paper specification; DFBGN with the exact CE residualization below; and efficient no-retention full-logit trust-region variants. Each requires fidelity checks and a development pilot spanning its relevant refresh/history policy. Full tuning continues for competitive or still-improving challengers under §8; not every inferior pilot needs a full confirmation run. A documented unavailable/ambiguous implementation narrows the claim rather than becoming a defeated comparator.

**Always-run practical references:** shared head, head plus temperature, conventional SGD/momentum, and Adam/AdamW on the same W. Give the gradient references the same cache and precision, and do not accidentally disable autograd through the frozen suffix. If memory becomes material, include a properly checkpointed suffix gradient variant with its recomputation cost.

The optional checkpointed suffix uses the installed PyTorch checkpoint API with explicit `use_reentrant=False`; verify loss/gradient parity and RNG behavior on the actual path before measuring its memory/time tradeoff. It is a baseline opportunity, not an assumed free memory reduction. [R39]

**Diagnostic controls:** original hill climbing, scalar steps with the candidate's exact direction dictionary, RSES/EKI equivalence fixtures, per-example CE-residual response fitting, and the attribution variants in §12. These establish causes; they do not replace the contemporary portfolio.

Published billion-parameter results remain contextual references. A common small-vision reimplementation is not a reproduction of those language-model experiments. Full native configurations can exceed a particular local time/memory envelope; report a measured exclusion and a scoped claim rather than silently substituting a weaker method.

### 7.2 Strong simple scalar methods

For Gaussian or Rademacher $u$ with covariance I, implement

$$
\widehat g=\frac1K\sum_i
\frac{\operatorname{CE}_B(W+\epsilon u_i)-\operatorname{CE}_B(W-\epsilon u_i)}{2\epsilon}u_i+\gamma W.
$$

Tune SGD, momentum, and Adam on this estimator, charging states and all $2K$ evaluations. The known quadratic regularizer is an exact analytic control variate for every method; forcing scalar baselines to estimate it with noisy directions while the candidate uses its exact derivative would be unfair. Native reproduction first uses gamma=0; a native decay convention and the common once-per-step regularizer are distinct recorded configurations. For subspace methods, project the known regularizer into their native coordinates or use a documented decoupled decay option. Include `K=1` as well as a query-count ladder. Normalize physical perturbation energy only in explicitly named matched-probe controls; do not silently change native algorithms. [R1, R3]

For the original hill-climber, compare current and candidate on the same batch. A baseline loss can be reused only while weights, batch, transformations, and stochastic state are identical. On a new batch, recompute it. Tune mutation radius and include an adaptive-radius version as a strong simple diagnostic, clearly distinguished from the original fixed-radius algorithm.

### 7.3 SubZero+ v2: required exact conventions

Use the inspected **28 September 2026 v2**, not the stale v1 title. For each matrix, draw sign-corrected orthonormal Gaussian QR bases $U,V$, hold them for F updates, and draw $Z_j\sim N(0,I)$ in $r\times r$. Compute

$$
C=\frac1K\sum_j\frac{J(W+\epsilon UZ_jV^T)-J(W)}\epsilon Z_j.
$$

Apply Adam to C, then map its coefficient update through U,V. Reset **both moments and the local bias-correction counter** whenever the bases change. This costs $K+1$ forward batches. Native candidates include K=99, ranks 16 and 128, beta=(0.9,0.95), Adam epsilon $10^{-8}$, and refresh interval 50 or 100. Include these alongside smaller query counts under equal total budgets. Use natural 256×256 W; no LoRA reshaping is needed. [R11]

Our local fidelity tests: QR orthogonality and sign conventions; native Gaussian scaling; identical-batch baseline/probes; exact state resets; and coefficient-to-physical-update agreement. Do not assume a lower K is equivalent, or compare 400 population steps to 400 two-query steps. Original SubZero is a separate central-probe/SGD control, not obtained merely by switching the newer optimizer's Adam flag off. [R10]

### 7.4 LOZO+ and the specified paired-batch correction

Native ordinary updates keep a left orthonormal U fixed for $\nu$ iterations, redraw Gaussian V, measure a central slope s along $UV^T$, and apply $W\leftarrow W-(\eta/r)sUV^T$. At refresh, loss feedback selects U columns. The inspected source revision is `9cdf1c27ebe364b35b49467f7072c9640d3978d7`, repository `fangjj2000/LOZO-`, file `LOZO+/GELOZOtrainer.py`. Its interval reference loss and column-candidate losses use different minibatches. The source also spends an interval baseline omitted by one paper cost table. [R13]

Use **`lozoplus_paired`**, an explicitly disclosed correction, in competitive comparisons:

1. Save exact `W_start`; run the native $\nu$ updates with fixed U.
2. Form $C=U^T(W_{end}-W_{start})$.
3. On one fresh training selection batch, evaluate `W_start` and each candidate $W_{start}+U[:,i]C[i,:]$.
4. Retain columns with paired improvement above the configured threshold, up to the retention cap; fill the rest with a random orthogonal complement.
5. Keep `W_end` as the actual model. Selection changes the next basis only.

The true cost is $2\nu+r+1$ evaluations per full interval. Moving the baseline to interval end corrects pairing without increasing that true count. For $T=20,000,r=4,\nu=100$, the accounting identity is 41,000, excluding monitoring. These are counts derived from the source, not runs performed here.

Include native ranks 2/4/8 and intervals 50/100/200 in development. Tune thresholds in the local loss units, preserving the native decreasing 0.1 schedule as a candidate. Native cross-batch selection remains a diagnostic, not the sole version to beat. With one biasless matrix the source's freezing of one-dimensional parameters causes no mask discrepancy. Test that a selection batch ID mismatch is rejected by the corrected implementation.

### 7.5 Activation-informed competitors

**AGZO.** For W(out×in), collect its baseline input matrix H(in×samples/spatial positions), initialize a randomized range approximation, and use three power iterations `Q=qr(Y); Y=H@(H.T@Q)` before final QR. Draw Gaussian R(out×r), set $D=RA^T$, and use one shared baseline and one perturbed loss. Native rank 1 and power count 3 are mandatory candidates; ranks 4/16 are additional options. Count activation processing and CPU QR/transfers. [R14]

**AIM-ZO.** With the same H(in×n) convention, maintain an activation basis by $Q^+=\operatorname{qf}(Q+\eta_q H(H^TQ)/n)$. Equivalently, use $X=H^T$ in the paper's $X^TXQ$ convention. Test dimensions explicitly and never form an n×n sample covariance. Each population member uses the first stored basis columns plus a sampled tail, forms $ab^TQ_{act}^T$, and normalizes its Frobenius norm to $\sqrt{q}$. QR does not automatically order columns by eigenvalue. With N≥2 population losses $l_i$, the scalar estimator is $\sum_i(l_i-\bar l)D_i/((N-1)\epsilon)$. There is one center plus N perturbed forwards. Its leave-one-out population control variate is not the candidate's cross-batch deletion score. [R15]

**Important local simplification, our deduction:** the adapter input h is frozen, so its activation covariance does not change with W. Include a static train-only PCA basis, with its covariance construction charged, as a strong inexpensive competitor and a generic proposal option offered to the other methods. Reshape the adapter input to a 256×(batch×196) matrix; do not materialize an unnecessary convolutional patch tensor. Online Oja maintenance may have little advantage on this particular architecture. Label this boundary rather than implying that the workload fully represents multilayer LLM activation drift.

**AdaSub-ZO.** Its scalar basis-training stage uses separate training batches and finite differences in factors A,B before QR; ordinary steps use central differences along $UZV^T$. Every refresh with M basis-learning steps costs an additional 2M forwards. For a local implementation preserve the printed basis-perturbation experiment $W+(A\pm\mu E_A)Z(B\pm\mu E_B)^T$; adding an unmentioned outer scale changes it. [R16]

The inspected paper's printed momentum-transport expression has a dimensional inconsistency. The mathematically dimension-correct local adaptation is

$$
M_{new}=\kappa(U_{new}^TU_{old})M_{old}(V_{old}^TV_{new}).
$$

Record this as a correction, test against projecting the physical old momentum, and do not claim exact source reproduction. If this ambiguity cannot be resolved to a defensible implementation, report it and limit claims relative to this recent preprint; do not count a broken implementation as a beaten baseline.

### 7.6 FZOO and PseuZO

**FZOO.** Use N Rademacher probes and one baseline on the same batch. The inspected code uses population SD and coefficients $(L_i-L_0)/(N\operatorname{sd}_{pop}(L_1,\ldots,L_N))$, with epsilon absorbed into its learning-rate convention. The paper writes an explicit epsilon divisor; record the chosen mapping and tune the corresponding learning rate. Include native N=8 and a population-size ladder. Skip and log numerically identical-loss populations instead of dividing by zero. Its local serial estimator is valid; CUDA parallel speedups are not imported as evidence. [R17]

**PseuZO classification adaptation.** Store history entries `(seed_j, epsilon_j, delta_logits_j, source_batch_ids, source_state_hash)`, with the native **one-sided** difference $\Delta z_j=z(W_j+\epsilon_j u_j;B_j)-z(W_j;B_j)$. This requires two forward batches per new entry. With $D_t=(\operatorname{softmax}(z_t)-Y_t)/B$, the practical positional-history update is

$$
\widehat g_t=\sum_{j=\max(0,t-L+1)}^{t}\beta^{t-j}u_j
\langle\Delta z_j,D_t\rangle/\epsilon_j.
$$

Its source contracts old output rows with current outer derivatives by tensor position, without example-ID matching. Preserve this in a named native-semantics adaptation, use fixed batch size, and add a fixed-batch/aligned diagnostic. Do not quietly remeasure all history and call it PseuZO. The code omits the paper EMA's `(1-beta)` factor; map learning rates explicitly. Use gamma=0 in the native fidelity test because source weight decay occurs inside its history loop. A once-per-step common regularizer is a separately recorded correction. [R9]

The published language implementation differentiates a detached final representation through the output head/loss; this classification adaptation differentiates CE on logits only. It is a faithful study of the estimator boundary relevant here, not an exact LLM reproduction. A trainer file carries an Apache-2.0 header, while a repository-wide license was not established; inspect the actual files before reuse.

### 7.7 Strong output/residual and curvature controls

**Exact objective bridge.** Define

$$
r_b(W)=\sqrt{2\operatorname{CE}_b(W)/N},
\qquad r_{reg}(W)=\sqrt\gamma\,\operatorname{vec}(W).
$$

Then $\frac12\|r(W)\|^2=J(W)$. The native DFBGN API minimizes $\sum r_b^2$, so this bridge returns native objective **2J**. Report native f/2 and scale absolute stopping tolerances accordingly; alternatively pass r/√2 and document that scaling consistently. Run native DFBGN on this fixed, full-training residual vector, with a small subspace dimension. Chunk neural forwards without changing residual order. Set the library's function-evaluation budget from actual example cost; wrap every callback in the ledger. Obtain argument names from the installed documented API rather than inventing an invocation. Confirm native convergence on its own small examples before adapting the callback. [R6]

Use stable CE; exact zero contributes residual zero. Do not add a hidden positive loss floor. The square-root residual's conditioning can be unfavorable near zero, which should be diagnosed rather than hidden. DFBGN is locally feasible in principle at this q with a small subspace; a large full-batch call is a cost tradeoff, not automatic grounds to exclude it.

For mechanism isolation, implement fresh-probe RSES-style ridge fitting of these per-example residuals on the same A/C examples as the candidate, with the same acceptance rule. Call this a stochastic residual-model adaptation. Also run native RSES/EKI equations on W0 least squares, where equivalence can be exact. [R4, R8]

The competitive memoryless method, **`logit_tr_fresh_efficient`**, draws fresh directions and probes **A only**. It needs C logits only at the base and actual candidate for the safeguard, not a response tensor S_C. Its cost is $(2k+2)|A|+2|C|$ suffix examples per proposed step, compared with the candidate's $(2k+2)(|A|+|C|)$. Give it this saving. Also tune a named fit-on-all-training-union/no-qualification variant; if its acceptance policy changes, disclose that change. Select the strongest no-retention variant on development for the required primary retention contrast.

A separate **`logit_tr_fresh_matched`** attribution control performs the same A∪C probes as §5 and disables retention. It isolates retention with matched information but must not be the only practical comparator. Include scalar first-order contraction on the same basis and a scalar quadratic model in a small fixed k-subspace. The latter estimates diagonal and cross curvature with its necessary additional evaluations; it must not receive free Hessian information. If a positive result is specifically attributed to curvature, add the contemporary scalar diagonal-curvature HiZOO comparator or explicitly narrow attribution to the tested scalar curvature control. [R23]

### 7.8 Baseline reproduction gate

Each baseline needs `method_card.json`: paper/version, source URL/revision, license/implementation route, exact oracle boundary, equations, native defaults considered, local changes, parameter mask, query formula, and fidelity-test results. Test one update against an independently expressed tiny reference and the paper's limiting cases. Compare code counts with ledger counts. Resolve discrepancies before an expensive comparison. Do not claim reproduced paper accuracy when only update equations have been checked.

### 7.9 LOREN: source behavior and a specified row-wise correction

LOREN is directly relevant scalar prior art for adaptive curvature/noise. The inspected MIT source is commit `b80a32bc394332d1a1f115f41dfb5dcf7b326d37`, `model/glue_transformer_loren.py`. It differs from the paper's repeated row-block covariance: it tiles one vector a across W and uses a global inner product, and it updates a and momentum inside the population loop. A local source port and a corrected row-wise variant must be named separately. [R45]

For **`loren_source`**, initialize a∈R^n with Gaussian entries, full W-shaped momentum M=0, and fixed rho>0. Let v=vec(1_m aᵀ), s=vᵀv, t=√(rho+s), c=(√rho+t)/(st). A standard Gaussian vector u produces D=reshape(u−c(vᵀu)v). Collect K≥2 losses at W+epsilon D using a frozen base and the same batch; no center call is needed. Then, in population order, regenerate each u and D using the **current** a, set w_j=L_j−mean(L), and perform

$$
M\leftarrow\beta M+\frac{w_jD_j}{\epsilon(K-1)},\qquad
W\leftarrow W-\eta M,
$$

$$
a\leftarrow a-\nu\frac{w_j\epsilon^2}{(K-1)t}
\operatorname{rowsum}\left[(v^Tu)u-v-c\{(v^Tu)^2-s\}v\right].
$$

This intentionally preserves the source's sequential state semantics for fidelity. Use gamma/weight decay zero initially; native decay is applied repeatedly inside that loop and is not interchangeable with once-per-population regularization. Replace CUDA-only device literals for the local port, preserve seeds/dtypes, and use exact-base restoration. Zero s is a numerical failure to log, not an undocumented division fix. Cost is K forward batches. The source distribution has global covariance I−vvᵀ/(rho+||v||²); it does not implement the claimed repeated row-block covariance. This distinction is structural, not just a learning-rate factor.

For **`loren_rowwise_frozen`**, the explicit paper-structure adaptation used here, define P=rho I_n+aaᵀ, C=P^-1, t=√(rho+||a||²), and xi=1/[t(t+√rho)]. Draw Gaussian matrices Z_j and

$$
D_j=\frac{Z_j-\xi(Z_ja)a^T}{\sqrt\rho},\quad
g_W=\frac{\sum_j w_jD_j}{\epsilon(K-1)},\quad
g_a=\frac{\sum_jw_j\left[ma/(\rho+\|a\|^2)-D_j^TD_ja\right]}{K-1}.
$$

Freeze W,a during all probes **and accumulation**, then update M once, W once, and a once. This is ordinary Gaussian score-gradient learning of a with a preconditioned mean update; it is not an invented natural-gradient update for a. Fixed positive rho preserves positive definiteness. For CE-only stochastic estimates plus the common quadratic penalty, the analytic preconditioned mean contribution is $\gamma WC$. The covariance-score contribution for the same smoothed objective is $-\gamma\epsilon^2ma/(\rho+\|a\|^2)^2$, obtained by differentiating $\gamma\epsilon^2m\operatorname{tr}(C)/2$. Validate both identities. Start gamma=0 for native fidelity, then allow this explicit regularized corrected variant and a named source-decay variant in development.

The once-per-population recurrence is $M\leftarrow\beta M+g_W$, $W\leftarrow W-\eta M$, $a\leftarrow a-\nu g_a$, with no implicit 1−beta factor. In the source expression, `rowsum` means reshape the flattened vector to (m,n) and sum along the m rows.

Development includes paper K=6 and source K=10, rho=0.1/1 anchors, momentum 0.9, and logarithmic tuning of eta, nu, and epsilon. These are sourced candidate settings, not universal optima. Choose the stronger defensible variant on development. Validate covariance and the sign of the a score by differentiating the Gaussian log density while holding the sampled displacement fixed; an intermediate paper expression has a sign inconsistency, while its final score expression supports the formula above. The inspected source's own score has an absorbable fixed scale; do not describe the entire implementation as mathematically meaningless.

### 7.10 CurvZO: scalar adaptive sparse probing

The paper version maintains one importance S_i per parameter/block. Use its **coordinate** form on W as the primary local adaptation, G=q=65,536, and a separately named 256-row-block version as an efficiency control. The official implementation's one-block-per-parameter-tensor convention would give G=1 here and cannot express adaptive selection; it is only a degenerate diagnostic. [R46]

Initialize S_i=1. Set a_i=√S_i, p_i=a_i/Σa, d_eff=1/Σp_i², and H=−Σp_i log(p_i)/log(G). Every K_B steps, including step zero, set expected perturbed-block count

$$
B=B_{min}+(B_{max}-B_{min})\{\alpha d_{eff}/G+(1-\alpha)H\}.
$$

Solve for c>0 such that π_i=min(1,c√S_i) and Σπ_i=B, using monotone bisection with a repeatedly doubled upper bracket or exact water-filling. Draw independent Bernoulli masks m_i with these inclusion probabilities and Gaussian blocks z_i; v_i=m_i z_i. On one paired batch,

$$
s=\frac{L(W+\epsilon v)-L(W-\epsilon v)}{2\epsilon},\quad
g_i=s v_i/\pi_i,\quad
S_i\leftarrow(1-\beta)S_i+\beta s^2\|v_i\|^2/\|v\|^2.
$$

Use sampling-time π for the update W_i←W_i−eta g_i. Update scores for **all** blocks, including decay of unselected ones. Cost is two forwards; B is a sparsity budget, not a query count. An all-zero mask produces a zero step with no redundant forwards and zero instantaneous score. Never force a selected block while pretending the original Bernoulli inclusion probabilities remain valid. Apply a tiny positive numerical score floor, log its activation, and test zero-signal/all-selected limits. The squared directional slope is an importance proxy, not an observed Hessian.

The paper does not give every numerical controller setting. Code-derived development anchors are beta=0.1, refresh=20, B_min/G=0.1, B_max/G=0.8, and alpha=0.7; label them code-derived and tune, not exact paper defaults. The inspected source at `e32b332ba0f623c14392e1a28c6e9b3b6d6b0dfe` uses materially different score/probability/budget recurrences and an invalid nonempty-mask correction. Preserve this discrepancy in its method card; do not call a weakened native port the paper algorithm. No repository-wide license was established; a trainer's inherited Apache header does not settle all files. Mathematical reimplementation of the equations is the preferred route here.

## 8. Evaluation, tuning, and confirmation

### 8.1 Primary claim and endpoints

**Primary population:** the fixed valid official W1 test-image set, with the exact frozen training/development split, pretrained checkpoint, head, adapter location, and environment. The primary uncertainty statement is conditional on this dataset and these assets; independent optimizer seeds do not create independent datasets.

**Primary metric:** arithmetic mean test negative log likelihood in **nats per image** at a preregistered optimizer wall-time budget $T_*$. Use the last fully committed state whose completion time is within the budget. The primary comparison is `logit_tr_qualified` minus the strongest eligible forward-only configuration selected on development data. Lower is better. No test-based checkpoint selection or best-of-seed reporting.

**Required second comparison:** candidate versus the strongest development-selected **efficient** no-retention full-logit variant at the same budget. This protects against mistaking an established output-model optimizer for a successful new retention rule. Use two-sided Holm-adjusted p-values across these two primary contrasts and simultaneous Bonferroni t intervals as specified in §8.5; if the selected strongest comparator is the same no-retention method, there is only one distinct contrast.

The headline can be positive only within this declared scope. “Beats every implemented method” additionally requires intervals/comparisons against each eligible method, with familywise adjustment. Choosing a strongest comparator on independent development data avoids test selection bias but does not guarantee that it remains the strongest on test.

**Required secondary metrics and guardrails:** mean accuracy, macro breed accuracy, NLL after development-fitted temperature calibration, per-breed errors, failure/stall rate, forward-example counts by segment and purpose, elapsed seconds, CPU solve/transfer/proposal fractions, cache/setup time, storage bytes, sampled memory maxima, and final inference latency. Report absolute differences and relative time/query ratios where denominators are meaningful. Do not describe a small statistically supported NLL gain as a breakthrough without practical effect sizes and tradeoffs.

**Mechanistic efficiency:** a separately frozen suffix-example budget $Q_*$, with NLL-versus-Q curves and the same endpoint rule. Count every base, plus/minus probe, candidate, rejected candidate, selection batch, and resampled precision diagnostic. This endpoint tests information efficiency independent of solver speed; it cannot establish end-to-end efficiency alone.

### 8.2 Resource ledger and measurement boundaries

Use actual time, not an assumed “backward = two forwards” conversion. Calibrate full-image forward, prefix, suffix, and gradient-step times separately. Report raw forward examples, physical calls, and backward examples; any forward-equivalent conversion must state the locally measured ratio and architecture.

The optimizer clock includes data/cache reading, CPU direction generation, transfers, all model calls, linear algebra, restoration, qualification, accepted updates, routine logging, and required checkpoints. A separate monitoring clock records development/test evaluation. Use a common prespecified monitoring schedule; competitive endpoint selection cannot spend uncharged extra evaluations. Report total experiment cost including monitoring and tuning, even when the primary optimizer clock excludes those held-out evaluation intervals.

At a time deadline, an incomplete iteration is restored/discarded as an update, but its spent time and calls remain charged. The scored checkpoint is the last committed one before the deadline. Do not allow candidate-specific overshoot to deliver a stronger state. A query-budget runner may decline an iteration whose conservative maximum call count exceeds the remaining budget; report unused budget. Never credit the same rejected computation twice as both a completed step and a free retry.

Separate common pretrained assets from project expenses. Show warm optimizer time, cold single-task adaptation including cache/head construction, and genuine reuse scenarios. The amortized break-even expression for A adaptations is

$$
T_{setup}+A T_{adapt,new}<T_{setup,ref}+A T_{adapt,ref}.
$$

Use it only when the same setup is actually reusable. Tuning expenditure is reported separately and included in a time-to-first-use comparison.

### 8.3 Development tuning without weakening baselines

1. Finish numerical/operator/fidelity gates before performance tuning.
2. For each method, bracket perturbation scale by actual signal/locality measurements (§9). Tune learning rates in logarithmic steps until both a too-small/no-progress region and an unstable/degrading region have been observed. Refine the stable bracket. Expand a boundary optimum; do not freeze a truncated favorable range.
3. Use the same train/development splits and the same aggregate tuning resource allocation for competing families. Count all native/default reproduction checks and failed trials. If a family requires a materially larger search, show results versus cumulative tuning cost and a sensitivity analysis giving its credible extra tuning effort. Equal numbers of hyperparameter combinations alone are not equal opportunity.
4. Use native candidates specifically listed in §7, then a sequential grid/racing schedule over remaining meaningful parameters. Early elimination uses development curves at matched resource rungs. Preserve a slower-but-still-improving challenger; do not eliminate it solely for losing early to a fast starter.
5. Use multiple paired development seeds for contenders. Choose the final configuration by mean development NLL at the frozen horizon; tie within development uncertainty goes to the lower-cost/simpler configuration. Do not select a different configuration on each test seed.
6. Give the candidate the same treatment. Its direction count is not fixed at 16/32 by convention: begin with k=1,4,8,16,32 and extend only if measured response rank and progress per cost support extension. Retention fractions include 0, 0.25, 0.5, and 0.75. These are search starting points, not an arbitrary upper limit on the science.

A stronger configuration discovered during development remains eligible even if it changes the expected winner. Baselines may use their native batch sizes under time/query budgets. For one-step solver/information ablations, enforce identical logical batches and full probe dictionaries. For retention-policy trajectories, pair the initial dictionary, batches, and fresh-proposal stream; retained slots necessarily diverge. Distinguish both attribution designs from the independently tuned practical comparison.

### 8.4 Choosing horizons from learning curves

Do not pick $T_*$ from the candidate's best-looking test point. On development data, run a doubling ladder of total optimizer time, starting from enough measured time for every eligible method to complete several meaningful updates, including one refresh cycle for history/subspace methods. Continue while a competitive baseline still shows development improvement distinguishable from run noise.

Choose the primary horizon from the best *baseline* development learning curve: the first ladder point after which the improvement from another doubling is smaller than its paired development uncertainty for two consecutive increments. The rule is a plateau diagnostic, not a claim of global convergence. Retain the earlier and later ladder points as prespecified secondary endpoints, and include a longer horizon if a competitor is demonstrably still improving. If no plateau is observed within the measured available resources, freeze the largest feasible horizon as a **resource-limited comparison**, state that limitation, and retain the slow-learning method rather than declaring it defeated asymptotically.

Set $Q_*$ using the analogous baseline-only development analysis in suffix-example units. Neither horizon may be revised after protected test results are accessed. Store all horizon-selection evidence in the preregistration. No runtime estimate in this document substitutes for this local calibration.

### 8.5 Pilot-calibrated fixed-sample confirmation

Use independent run pairs as the primary replication unit. A pair shares model/head, split, initialization where applicable, and keyed training-sampler randomness. Proposal RNG streams are method-specific; identical numeric seeds do not imply identical experiments once methods consume data differently. Repeat timings within a run are not additional replications. The W1 head is fixed in the primary analysis; varying head fit or train/development split belongs to a separately reported robustness analysis. This conditional scope is necessary given known benchmark variance sources. [R40]

Use a **fixed eight development-only pilot pairs per primary contrast**, providing seven variance degrees of freedom. This is a modest variance-planning experiment, not a claim that eight confirmation runs are sufficient. Do not extend this pilot based on favorable effects. Keep all pilot data outside confirmation. A separate replacement pilot after a documented hardware/measurement defect requires a dated amendment and rerunning every affected method.

To choose precision without inventing a minimum percent improvement, use an explicit **planning heuristic** from development: average the paired loss difference across pilot seeds for each development image; estimate its dispersion using duplicate-group-aware resampling, and scale its standard error to the known test-manifest size without opening test predictions. Let $h_{image,j}$ be the resulting approximate 95% half-width for contrast j. This is a convenient scientific resolution target, **not an irreducible uncertainty floor for the fixed-test-set estimand**.

Under a normal paired-difference planning model, set $\alpha=0.05$, planning-bound error $\zeta=0.05$, and

$$
s_{upper,j}=s_{pilot,j}\sqrt{7/\chi^2_{\zeta/m,7}}.
$$

This is a simultaneous upper-SD planning bound on the development distribution under that model; its transfer to test-run variance is unverified. Publish n-versus-precision forecasts using the point SD and several confidence levels so the conservatism is visible. Choose the smallest fixed n_j of new run pairs satisfying

$$
t_{1-\alpha/(2m),n_j-1}\,s_{upper,j}/\sqrt {n_j}\le h_{image,j},
$$

where m is the number of distinct primary contrasts, and set **n=max_j n_j**. Iterate the t-quantile calculation rather than using a normal approximation blindly. This attempts to balance optimizer Monte Carlo precision with the chosen development-derived resolution; it does not require an effect of any particular size or guarantee the target will be met. Record n, predicted elapsed/storage cost, and the detectable-effect curve.

If image-level dispersion or run SD is zero/numerically unresolved, or catastrophic failures prevent finite endpoints, stop this variance-based planning gate. Diagnose resolution/failure and preregister an amended feasibility design or a justified absolute precision; do not treat an apparent n≈2 as adequate evidence. If required n is infeasible, preserve the precision target, run only a separately declared feasibility-limited design, and call it inconclusive at that resolution. More compute may be scheduled before confirmation based on the forecast; no arbitrary “one weekend” limit is imposed here.

After n is frozen, run all pairs without inspecting test contrasts. For each j, compute the actual confirmation mean difference $\bar d_j$, SD $s_j$, two-sided paired t-test with Holm-adjusted p-value, and simultaneous Bonferroni interval

$$
\bar d_j\ \pm\ t_{1-\alpha/(2m),n-1}s_j/\sqrt n.
$$

Headline improvement requires the upper endpoints of **both distinct primary intervals** to be below zero. These intervals are exact for independent normal run differences and approximate otherwise; Holm-adjusted p-values alone do not define adjusted intervals. Retain the empirical difference distribution, failures, and a preregistered paired-bootstrap sensitivity interval; do not choose whichever analysis is significant. If the achieved interval is wider than planned, report the precision miss without adding favorable-looking seeds. Severe outliers or invalid approximations require a disclosed limitation and fresh amended confirmation, not a silent analysis replacement.

This interval describes optimizer randomness conditional on the fixed images. Also provide a two-axis uncertainty analysis that resamples run pairs and image duplicate groups independently; label its sampling assumptions and the limited information about repeated animal identities. Per-breed stratification is useful for reporting, not a license to treat 37 breeds as independent training experiments. Do not extrapolate a one-dataset result to a population of tasks.

### 8.6 Preregistration, test protection, and amendments

Before protected evaluation, write an immutable `preregistration.json` and human-readable companion containing: claim, oracle boundary, assets/hashes, data manifests/exclusions, methods/configurations, comparator-selection rule, T/Q budgets, endpoints, seeds, sample size and calculation, monitoring schedule, statistical analysis, failure policy, and planned robustness checks. Commit it and record its hash.

The training/tuning commands must refuse test-label metric evaluation. A separate `confirm-evaluate` command requires the preregistration hash and a completed-run manifest. It writes all results at once. Access is a reproducibility boundary, not a claim that a local user cannot inspect files.

Exploratory observations can change the plan before confirmation, with a dated amendment. After test inspection, a changed mechanism, baseline set, endpoint, population, or interpretation is exploratory on that test. New seeds alone do not provide a fresh image population if the method was redesigned using its test errors. Seek a genuinely untouched second test set/task or state that independent confirmation remains outstanding. Do not repeatedly reuse the same test set under the label “fresh confirmation.”

### 8.7 Failure handling and honest effect reporting

A numerical failure returns the last finite committed checkpoint for endpoint scoring and is flagged; it is not discarded. An external power interruption resumes from the checkpoint and preserves spent-work counters. An OOM or unsupported operation invokes §9's documented fallback, requiring a common recalibrated configuration if the mathematical workload changes. Report failed-run proportions and reasons by method.

For small effects, report the absolute NLL change, accuracy difference in percentage points, queries/seconds saved, and interval width. A confidently nonzero but operationally negligible effect is a small result. An interval spanning meaningful improvement and harm is inconclusive. A precise non-improvement against a strong baseline is evidence against the local value proposition. Improving NLL while harming accuracy or memory is a tradeoff, not an unqualified win.

## 9. M3 Max feasibility and calibration

### 9.1 What is feasible in principle, and what remains unverified

The proposed model and adapter are small enough that tensor storage alone does not suggest a need for a remote accelerator. The uncertain resources are forward/backward workspaces, operating-system pressure, the number of probes required for progress, and the total tuning/confirmation workload. M3 Max machines have different memory/GPU configurations; the largest advertised configuration is not an assumption about the user's laptop. [R35]

No M3 test was run for this document. The first local gate must establish numerical correctness and throughput on the actual machine before estimating hours or claiming an advantage. A CPU fallback on that same laptop satisfies locality but may change the practical runtime conclusion.

### 9.2 Discovery and dependency selection

These are real macOS discovery commands, to run individually and save their outputs in `environment/host.txt`:

```bash
uname -m
sw_vers
sysctl -n hw.memsize
sysctl -n machdep.cpu.brand_string
df -k .
xcode-select -p
pmset -g batt
```

Also record Python architecture/version, available memory and swap, GPU description without device serial numbers, power mode, and background workload. Use a native arm64 Python and an isolated environment. Discover a **released matching PyTorch/torchvision wheel pair** from their official compatibility information for that Python and macOS version. Install NumPy, SciPy, Pillow, and the small logging/test dependencies in the same environment. Run the operation suite before locking exact versions and wheel hashes. A lock generated on Linux does not certify macOS compatibility. [R29, R31]

Do not copy an unverified “latest” version from this document. Official pages inspected at the cutoff exposed different release/documentation tracks and minimum-OS statements. The executable rule is: select a supported published pair, verify its actual MPS path, then freeze the tested environment. Retain the discovery URLs, package metadata, resolved versions, and source revisions in the run manifest. A later dependency upgrade requires repeating numerical and baseline-fidelity gates; do not mix versions within confirmation.

The primary stack is PyTorch/torchvision for the network, NumPy/SciPy CPU float64 for outer optimization, and memory-mapped NumPy feature caches. Keep model/activation/probe tensors float32 on MPS. Metal's MPS path does not provide the needed float64 arithmetic; allocating a double tensor there is not repaired by enabling fallback. Run QR, eigendecomposition, the small CE solve, and statistics explicitly on CPU. [R31, R36]

Set `PYTORCH_ENABLE_MPS_FALLBACK=0` before Python import for primary calibration so unsupported operations surface rather than silently changing timing. Do not disable MPS memory watermarks to force a larger batch. Record all MPS-related environment settings. No quantization, mixed precision, `torch.compile`, custom Metal kernels, or GPU QR is needed initially. [R38]

### 9.3 Critical operation and transfer gate

Test a synthetic miniature of the **actual execution path**, before downloading model/data assets: 1×1 adapter convolution, frozen convolution/BatchNorm-in-eval/ReLU blocks, pooling, linear logits, CPU CE, parameter replacement/restoration, and MPS-to-CPU transfers. Test the first-order comparator's backward path through frozen suffix weights to trainable W. `requires_grad=False` on frozen weights must not suppress that backward path.

Include this real-API transfer check:

```python
import torch
assert torch.backends.mps.is_built()
assert torch.backends.mps.is_available()
x_cpu = torch.arange(1, 9, dtype=torch.float32) / 3
x_mps = x_cpu.to("mps")
torch.mps.synchronize()
y32 = x_mps.detach().cpu()     # transfer without changing dtype
y64 = y32.to(torch.float64)    # CPU conversion only
assert torch.equal(y32, x_cpu)
assert torch.equal(y64, x_cpu.to(torch.float64))
```

An official PyTorch issue reports silent incorrect combined MPS→CPU/float64 copies in some 2.14/nightly builds. It was **not reproduced here** and does not establish that every current wheel is affected. It motivates the explicit two-stage transfer and local assertion; never accept all-zero responses without checking the transfer path. Record the installed build's result. [R44]

The inspected upstream fix restores an exception for unsupported fused destination types; it does not make fused float64 conversion supported, and a merged fix does not identify the installed release wheel. Extend the staged-copy smoke test to signed binary fractions, transposed tensors, and strided views, preserving float32 before casting. [R44]

Compare CPU and MPS logits/CE on fixed inputs and the same saved parameters, including small plus/minus probes. Use relative/absolute tolerances derived from repeatability, float32 scale, and the CPU reference; publish observed errors instead of assuming bitwise cross-device equality. Test gradients against CPU finite differences on the miniature only. Preserve exact base restoration and finite outputs on the full model. Fail the gate on unsupported operations, silent CPU fallback, broken gradient flow, or unexplained transfer/response errors.

### 9.4 Tensor-size estimates and peak-memory model

The following are arithmetic estimates, not measured peaks. MiB/GiB use powers of 1024. W1 counts assume all official files pass validation.

| Item | Calculation / approximate storage |
| --- | --- |
| Original ResNet-18 checkpoint parameters | 11,689,512 parameters; about 44.6 MiB in float32 before replacing the head. [R27] |
| W and one exact snapshot | 256×256×4 bytes each = 0.25 MiB each. |
| Adapter gradient plus two Adam moments | Three W-sized float32 tensors = 0.75 MiB, excluding activations. |
| One cached layer3 image | 256×14×14×4 = 200,704 bytes. |
| Official trainval cache | 3,680 images ≈0.688 GiB; exclusions can reduce it. |
| All 7,349 image caches | ≈1.374 GiB, plus small labels/IDs/manifests. |
| Logical B=128 input features | 24.5 MiB if all resident at once; microbatching reduces live input storage. |
| Response tensor B=128,C=37,k=32 | 1,212,416 bytes ≈1.156 MiB in CPU float64. |
| 32 explicit dense W directions | 8 MiB in float32 or 16 MiB in float64. Rank-one factors need far less. |
| k×k coefficient matrix at k=32 | 8 KiB in float64. |
| All unadapted pooled 512-dimensional features | About 14.35 MiB in float32. |

Budget peak memory as

$$
M_{peak}=M_{frozen}+M_W+M_{snapshots}+M_{live\ activations/workspaces}
+M_{responses/solver}+M_{resident\ cache}+M_{runtime/allocator}.
$$

The first-order path additionally keeps required backward activations; frozen parameters alone do not eliminate them. For this adapter, optimizer state is tiny, so a large memory advantage is **not assumed**. Measure both paths. Do not extrapolate the original MeZO memory ratio to this architecture.

Use `torch.mps.current_allocated_memory()` and `driver_allocated_memory()` when available, plus process RSS, system available memory, and swap pressure. The first excludes allocator cache; the second includes broader driver allocations. Neither is a documented historical peak counter. Sample at explicit phase boundaries and a recorded polling cadence; call the result a **sampled maximum**. RSS and MPS figures overlap on unified memory and must not be summed as independent pools. Use fresh processes for comparable peak measurements. [R33]

Storage planning includes downloaded images/annotations and extracted files, the cache above, environment/wheels, per-run checkpoints, per-image endpoint losses, and logs. Inspect actual download lengths and extracted size; do not substitute compressed archive size for required free storage. Forecast `setup_bytes + n_runs*(checkpoint_bytes + endpoint_bytes + expected_log_bytes)` with room for atomic checkpoint replacement. Keep scalar probe diagnostics/seeds/hashes rather than every full trial tensor; retain selected full fixtures and failed-run evidence.

### 9.5 Numerical and throughput calibration

**Probe scale.** At zero W, a rule proportional only to $\|W\|$ gives no perturbation. Start a doubling/halving scan around $h_0=\epsilon_{32}^{1/3}\max(1,\|W\|_F)$ for unit-norm directions, and record physical activation/logit changes. This is a search anchor, not a universal optimum. Use training examples and both random and retained-like directions. Estimate a response-noise scale from repeated identical logits and an explicit float32-roundoff proxy $\epsilon_{32}\max(1,\|z\|_\infty)$; deterministic repeats alone cannot expose quantization.

Find an interval where paired logit changes exceed that scale and central responses at h/2,h,2h are consistent. Initial diagnostic tolerances are signal at least 100 times the scale and ≤5% relative response disagreement, measured only on directions with observable signal. These numerical criteria are tunable calibration assumptions, not an improvement threshold. Confirm any failure with CPU float64 miniature calculations and CPU float32 full-network probes. Inspect class-centered response spectra and actual candidate errors. If no interval has both signal and locality, this oracle/precision combination is unsuitable; do not lower the criteria secretly or report zero derivatives as evidence against learning.

Report the fraction of proposed directions with observable centered-logit signal, not just successful examples. Cap each doubling/halving scan at 60 changes and stop earlier on overflow, underflow, or unchanged realized float32 states; a failed bracket is a recorded precision/locality limitation. Run this numerical probe/restore **preflight before the first W1 baseline/candidate pilot**; WP06 later performs the full throughput and tuning calibration.

For trust radius, begin around an observable h and expand/shrink on training-only pilots until both reliable local predictions and a clear loss of locality are observed. Choose initial/cap radii from that bracket; tune them on development independently from h. Log ReLU activation changes in separately charged diagnostic passes. Define the damping curvature floor as $h_{floor}=\epsilon_{64}\max(1,\operatorname{median}(\operatorname{tr}(H_0)/k))$ over the calibration fixtures, record its units/scale, and test sensitivity. An exactly flat output model must yield a stall, not an invented curvature signal.

**Realized-step precision.** Construct the proposed W from the saved CPU float32 base and a CPU float64 summed update, then round once to float32 before transfer. Measure the rounding error. Require actual physical norm ≤Delta×(1+10^-6); a small overshoot may be rescaled once to Delta×(1−10^-5). Use the realized dictionary projection for predicted reduction and qualification. If the out-of-span rounding residual exceeds $\sqrt{\epsilon_{32}}$ times the realized step norm, or rescaling still violates feasibility, reject as precision-limited and diagnose; do not count a different step as the intended one. These are numerical guardrails whose sensitivity is checked before confirmation.

**Throughput.** Warm exact tensor shapes until timing no longer has an obvious compilation/allocation trend; save the warm-up trace. Time blocks with `time.perf_counter()` and `torch.mps.synchronize()` immediately before/after. Separately measure prefix extraction, suffix forward, first-order forward/backward, transfer, direction construction, small solve, selection, checkpointing, and monitoring. Record median and dispersion over independent timing blocks; these are measurement replications, not training seeds. [R34]

Try physical batches 8,16,32,… until the measured throughput plateau or safe memory envelope is reached. Choose the smallest batch near the plateau that leaves measured OS/runtime headroom under normal background load. Do not derive safety from total RAM alone. All methods may use fair efficient microbatching with the same logical batch semantics: aggregate CE as a sum divided by total examples, accumulate first-order gradients accordingly, and preserve fixed BatchNorm/dropout behavior. Confirm full-batch/microbatch equivalence within numerical tolerance.

Counterbalance method order across run pairs, keep power state consistent, and record interruptions/thermal or system-load changes. Runtime is not an energy measurement. Randomize or block run order rather than always benchmarking the candidate after caches and system state have warmed.

### 9.6 Forecasts, fallbacks, and recovery

For each method use the actual query formula and measured microbatch throughput:

$$
T_{run}\approx\sum_{segments,purposes}N_{examples}\tau_{segment/example}
+T_{solve}+T_{proposal}+T_{transfer}+T_{I/O}+T_{monitor}.
$$

Use end-to-end pilot timing as the authority when this additive estimate misses overlap or allocator effects. Forecast the full tuning portfolio, fixed pilots, confirmation n, secondary budget runs, and planned ablations; report uncertainty from timing dispersion and observed step counts. Do not announce a fixed hour count before measuring. The decision to scale should reference this forecast and the unresolved scientific uncertainty.

Fallback order is: reduce **physical** microbatch while preserving logical batches; use memory-mapped/streamed caches; keep the small solver on CPU; use the same float32 model on CPU if the MPS path fails. Recalibrate all competitors. A smaller image resolution/model, changed logical batch, lower k, reduced precision, or quantization changes scientific conditions and requires a recorded amendment and common comparison. MLX is a possible later independently verified port, with forced evaluation for its lazy execution and explicit RNG handling; it is not assumed equivalent or needed for the initial PoC. [R37]

Checkpoint at committed update boundaries, atomically replacing the checkpoint only after its payload/hash is complete. Include W, every optimizer/basis/history state, trust/probe/damping settings, all used RNG states, sampler cursors, objective/config/source/asset hashes, accumulated time/query counters, pending/wasted work, and the last completed record ID. Store the previous valid checkpoint until the next one verifies. On interruption restore that state; spent work since it was saved is charged and replayed deterministically where possible. Assert stepwise CPU replay equality and documented MPS tolerance; cross-version/device bitwise replay is not promised.

Freeze a common wall-time checkpoint cadence, saving at the next committed boundary and at final termination. Do not require a full disk snapshot after every optimizer step: that would penalize inexpensive methods with more updates. Lightweight call/commit ledger events still occur at the required boundaries, and both logging and checkpoint costs are charged.

Use append-only write-ahead call reservations and completion records. Abrupt power loss may leave an in-flight tail whose exact work is unknowable; retain its conservative reserved-cost bound and mark uncertainty rather than reconstructing an imaginary exact count. Runs with an unresolved budget-ordering ambiguity cannot silently enter a clean fixed-budget confirmation contrast.

An OOM must restore the exact committed model and produce a failure record. Retry in a fresh process with smaller physical microbatches only after the equivalence test. If the mathematical workload changes, stop and amend before comparing. Native solvers with an internal loop require an accepted-incumbent/budget adapter (§10); restarting them from initialization on timeout is not an acceptable competitive implementation.

## 10. Repository, interfaces, and records

### 10.1 Minimal project layout and authority

Create one ordinary version-controlled Python project. The following paths are **proposed**, not files already supplied:

```text
pyproject.toml
README.md
environment/                  # host report, dependency lock, wheel/source hashes
configs/                      # workload/method/calibration/confirmation JSON
src/fopoc/
  __main__.py                 # command dispatch and configuration validation
  assets.py                   # downloads, notices, checksums, manifests
  workloads.py                # W0/W1/W2 and exact parameter masks
  oracle.py                   # forward/gradient boundaries and restoration
  sampling.py                 # keyed batches and independent RNG streams
  directions.py               # dense/rank-one/activation bases, Gram geometry
  coefficient_model.py        # stable CE, gradient, solve, deletion scores
  methods/                    # one adapter per named method/version
  budget.py                   # actual calls, examples, time, reservations
  runner.py                   # atomic commit, checkpoints, interruption
  calibrate.py                # numeric/hardware/throughput discovery
  analysis.py                 # frozen endpoints, paired intervals, attribution
  report.py                   # tables/plots and claim-status generation
tests/                        # mathematical, oracle, method-fidelity fixtures
data/manifests/               # permitted IDs/hashes; raw data excluded from git
cache/                        # regenerable mapped features
runs/<run_id>/                # immutable config, events, checkpoints, endpoints
decisions/                    # append-only progress, gates, amendments
reports/                      # development and confirmation kept distinct
```

The coding agent may make routine, reversible implementation choices, repair defects, select compatible packages, and optimize shared data movement without asking about each one. It must record a methodological amendment for changes to the mechanism, oracle, population, primary endpoint, baseline opportunity, numerical meaning, or interpretation. It must not silently change the hypothesis after a negative result. Save decisions and the next executable step in project artifacts so the work can resume without chat memory.

### 10.2 Required shared interfaces

These signatures describe the project's interface, not an external package API:

```python
class Oracle:
    def logits(self, state, batch, purpose: str) -> "Response": ...
    def loss(self, state, batch, purpose: str) -> "LossResponse": ...
    def first_order(self, state, batch, purpose: str) -> "GradientResponse": ...
    def trial(self, base_state, delta) -> "RestoringContext": ...

class StepMethod:
    def initialize(self, context) -> "MethodState": ...
    def reserve(self, context, state) -> "CostReservation": ...
    def step(self, context, state) -> "StepResult": ...
    def serialize(self, state) -> dict: ...
    def restore(self, payload, context) -> "MethodState": ...

class NativeLoopMethod:
    def run(self, context, budget, incumbent_sink) -> "RunResult": ...
```

`Response` contains CPU float64 logits, original float32 dtype/device, ordered sample IDs, class order, parameter hash, data-transform hash, and the ledger event ID. `LossResponse` may avoid transferring all logits where a scalar method can efficiently reduce on-device; compare its numerical accuracy and charge actual transfers. Exposing the same oracle capability does not require every method to pay for unused tensors. `first_order` is accessible only to named gradient methods/diagnostics; forward-only methods must pass an autograd-use audit. The prefix cache is shared, but each logical oracle evaluation executes the complete trainable adapter and frozen suffix.

For numerically matched attribution, every scalar method must also support stable CPU float64 CE/reduction and central subtraction from the same float32 logits used by the candidate; the optimizer receives only the scalar. Include this as a competitive development option and charge its transfers. A faster native float32 reduction remains eligible if calibrated. Otherwise an apparent vector-information benefit could be ordinary scalar cancellation error.

Canonicalize every proposed parameter vector once to the authoritative float32 tensor **before** evaluating logits, the exact weight penalty, hashes, and committed endpoints. This includes DFBGN's CPU float64 interpolation coordinates: its callback's regularization residual must correspond to the realized W32, and its returned x must be canonicalized before scoring. The native solver may retain its interpolation coordinates, but the actual oracle is the declared rounded map.

`StepResult` contains proposal/realized delta hashes, accepted flag/reason, completed/wasted cost, trust diagnostics, numerical validity, and the proposed next method state. Only `runner` atomically commits. Synchronize MPS before recording completion time, so an update enqueued before T but completed after T is not credited. A trial context restores base parameters even on exceptions. Immutable hashes detect model, batch, and response mixing.

`Budget` supports time and segment-example budgets separately, has an explicit warm-up/initialization policy, and reserves the worst-case required calls for an iteration. A time deadline is enforced on completed operations; incomplete updates are rolled back but charged. For query budgets, do not start a multiquery step that cannot fit, unless the method has a predeclared mathematically valid anytime estimator. Report unspent budget and stopping granularity. Native method-specific basis/PCA initialization is charged inside the optimizer boundary unless explicitly reported as method-specific cold setup; it is never silently shared for free.

### 10.3 Native DFBGN integration without false API assumptions

The inspected official source revision is `82da43b59acdd0788291cf5addd5144766954743`. Its public interface includes `solve(objfun, x0, args=(), fixed_block=None, bounds=None, rhobeg=None, rhoend=1e-8, maxfun=None, user_params=None, objfun_has_noise=False, scaling_within_bounds=False, true_objfun=None)`. It returns `x`, `resid`, `f`, `nf`, `flag`, `msg`, and optional diagnostics. `maxfun` is a genuine native limit; a callback, wall-time limit, and resumable-state API are **not** exposed in this inspected version. [R6]

For fixed-query runs, pass an explicit `maxfun` computed from remaining suffix examples divided by N_train, use a small explicit `fixed_block`, and score the returned solution; native budget termination preserves an incumbent. Do not leave the default budget or enable `true_objfun`, which can trigger full-Jacobian diagnostic work. Do not enable expensive diagnostic matrices merely to obtain lightweight timestamps. Store/restore the NumPy global RNG used by the native solver.

For time-budget competitiveness, provide a minimal, disclosed GPL-compatible instrumentation patch that records valid incumbents with synchronized completion times and exits through the solver's normal saved-incumbent mechanism. Validate patch parity against unmodified runs on W0, including budget expiration during initial interpolation and a geometry step. A minimally instrumented uninterrupted run can instead record the incumbent as of T while completing the in-flight operation afterward; charge/report overshoot and score only the timestamped incumbent. The project must not score initialization merely because an arbitrary callback exception lost native state.

Faithful restart requires interpolation points/residuals, radii, histories, counters, and RNG, not only `soln.x`. Either implement and test that state snapshot or label this method uninterrupted-only and schedule complete runs. If no defensible wall-time wrapper is achieved, it remains eligible for the query comparison, and the time-based claim explicitly excludes it. That is a scoped result, not evidence that DFBGN was defeated.

Profile its interpolation implementation when adding q regularization residuals. The inspected version performs per-residual Python-loop triangular solves. A vectorized multiple-right-hand-side solve is an allowed disclosed optimization, with W0 parity tests and native/optimized timing disclosure if material. Do not attribute avoidable language-loop overhead to inferior information efficiency, or remove regularization residuals without preserving their exact contribution.

### 10.4 Configuration conventions

Use JSON with a schema version and reject unknown keys. Store the fully resolved configuration with every run, not only a filename. Required groups are `workload`, `assets`, `split`, `oracle`, `method`, `numerics`, `sampler`, `budget`, `monitoring`, `environment`, and `analysis`. Quantities include units in their key or schema (`time_seconds`, `memory_bytes`, `nll_nats_per_example`). Seeds use separate named streams for split, initialization, fresh directions, batch order, and bootstrap.

A development calibration configuration may contain unresolved values with explicit source gates, such as `time_seconds: null` and `resolve_at: WP06`. A confirmation configuration must contain no unresolved scientific values and must reference the immutable preregistration. Starting calibration values from §5/§9 are not automatically final hyperparameters. Parameter masks are serialized as ordered names/shapes and checked against actual trainable counts.

Use a common logarithmic wall-time/query monitoring grid frozen after calibration, rather than monitoring every k optimizer steps. Record the boundary lag if an atomic operation crosses a scheduled monitor point. Monitoring uses development data and is excluded from optimizer time only under the common rule in §8; its cost remains visible. No training path can select checkpoints using test labels or metrics.

### 10.5 Experiment-record schemas

| Artifact | Required fields and units |
| --- | --- |
| `manifest.json` | run ID, phase (fixture/development/confirmation), source and dependency hashes, resolved config, method-card hash, host ID without serial, asset/cache/split hashes, named seeds, start/end UTC, status/failure reason, parent/resume links. |
| `events.jsonl` | monotonically increasing event ID; phase/purpose; model/batch/direction hashes; batch size; physical microbatches; segment; forward/backward call and example counts; synchronized seconds; transfer/solve/proposal/I/O seconds; sampled memory bytes; finite/checksum flags. |
| `steps.jsonl` | iteration; A/C IDs or immutable manifest reference; h/Delta/lambda/gamma; effective k; Gram spectrum; response spectrum/rank threshold; requested/realized norm; projected residual; predicted/actual A and C reductions; acceptance ratio; score/retention summaries; committed state ID. |
| `endpoint.npz` + JSON metadata | ordered sample IDs, per-image unpenalized NLL in nats, predicted/true classes, optional centered logits, method/seed/checkpoint/budget hashes; no aggregate-only replacement. |
| `resources.json` | setup/cold/warm boundaries, timing traces, all work counts, sampled maxima/cadence, cache/storage size, power/background notes, exclusions/overshoot. |
| `decision.jsonl` | date, work package, evidence paths/hashes, pass/revise/stop/inconclusive decision, rationale, amendment ID, next action. |
| `method_card.json` | native source, version/license, formulas, changes, oracle, cost model, defaults considered, fidelity checks, known ambiguities. |

Never store only successful runs. Preserve failures, rejected configurations, tuning expenditures, unsupported kernels, and protocol deviations. Retain sufficient small fixtures to recreate probe arithmetic; save seeds and factors for large runs. Hash arrays in a declared order/dtype. Pin downloaded source files and assets with SHA-256 even when upstream filenames are stable.

For protected evaluation, predeclare a numerical failure policy. Use the **same inference wrapper for every method**: if an example's logits are nonfinite, return W1's common frozen-head prediction, or uniform ten-class probabilities for W2 (NLL log 10), and record a catastrophic numerical-fallback event without consulting the label. Include fallback computation in inference cost and publish results with this defined wrapper and the raw failure rate. A failure-free superiority claim is prohibited if such fallbacks occur. This prevents undefined NLL from becoming a dropped seed or test-guided search through old checkpoints.

The report generator must separate verified identities, development discoveries, protected confirmation, and untested extrapolations. Include loss-versus-time and loss-versus-examples, absolute endpoint effects with intervals, cost/memory breakdowns, failures, and the full eligible-comparator table. A claim-status field must identify whether H1–H5 are supported, contradicted locally, or unresolved.

## 11. Ordered work packages and gates

The critical dependency path is WP00 → WP01 → WP02 → WP03 → {WP04, WP05} → WP06 → WP07 → WP08 → WP10. WP09 provides independently scoped robustness; WP11 is conditional escalation. Baseline and candidate implementation can proceed independently after the common oracle passes. No expensive confirmation depends on an unresolved fidelity or asset gate.

All commands below beginning `python -m fopoc` are interfaces the coding agent must create. They are not claimed to exist now. Each command writes a machine-readable result and an append-only decision record, exits nonzero on an invalid gate, and prints the next permitted action. Existing commands such as `python -m pytest` become usable only after the environment and proposed test suite exist.

### WP00 — Verify access, environment, and executable scope

**Objective/risk:** establish that the local stack, necessary assets, and reuse terms support the intended experiment; prevent expensive work built on nominal compatibility.

**Inputs/entry:** this plan, the actual laptop, public source URLs; no training assets required. **Actions/files:** create the minimal project and `decisions/`; run §9 discovery; inspect matching release support; install an isolated stack; implement `python -m fopoc doctor --output environment/doctor.json`; run the miniature MPS forward/backward/transfer tests with fallback disabled. Create `assets.lock.json` with URL, version, license/notices, expected upstream checksum where supplied, and subsequently computed SHA-256. Record baseline code licensing and clean-room/reuse route.

**Tests/outputs:** `doctor.json` with actual RAM/free disk bytes, OS/architecture, versions, supported operations, and observed numeric errors; an asset-access/legal-terms checklist limited to actual downloads/reuse. **Resources:** no model/data download yet; package installation and small tensors only. Measure installed environment size.

**Gate/next:** pass to WP01 if the miniature and terms are acceptable. Fix installation/kernel/transfer errors and retest before continuing. Use the documented CPU path if valid and locally practical; label acceleration unresolved. If necessary asset terms cannot be satisfied, prepare W2's narrower path and amend the claim. A blocked download is an access limitation, not a learning failure.

### WP01 — Establish mathematical and measurement invariants

**Objective/hypotheses:** H1 consistency, physical geometry, and an evaluation harness that cannot manufacture a benefit.

**Inputs/entry:** WP00 working CPU numerical stack; no pretrained checkpoint. **Actions/files:** implement W0, `directions.py`, `coefficient_model.py`, and fixtures; implement `python -m fopoc validate-math --config configs/w0.json`. Reproduce §4 identities, first-order contraction, Gram whitening, score sign/scale invariance, the quadratic deletion-score identity, CE-residual scaling, and the unit-ball constrained solve. Include duplicate/zero directions, zero logits, saturated logits, extremely small radii, nonfinite inputs, and the even-response bilinear case.

**Tests/outputs:** well-conditioned nonzero algebra fixtures have relative error <10^-10 in float64; zero/saturated/ill-conditioned stress cases use explicitly condition-aware absolute/relative tolerances. Monte Carlo identities must fall within their precomputed uncertainty. Compare small solves to exhaustive one/two-dimensional searches and an independent convex reference; save `math_checks.json` with dimensions, seeds, errors, units, expected pass/fail. Equivalent methods must produce equivalent trajectories to numerical tolerance. An injected objective improvement must be detected; shuffled method labels and identical-state comparisons must yield zero systematic effect.

**Resources:** seconds/minutes-scale expectation for tiny CPU fixtures, to be measured; no basis for an M3 throughput claim. **Gate/next:** repair any identity or harness failure before WP02. An analytically expected failure of central probes on an even-response fixture is a passing negative control. If the proposed rule algebraically collapses to an existing method, retain the experiment as replication/empirical comparison and revise novelty immediately.

### WP02 — Acquire data, build manifests, and prepare shared assets

**Objective/risk:** reproducible population, leakage control, and identical reusable preparation.

**Inputs/entry:** WP00 permissions/terms and WP01 invariants passed. **Actions/files:** implement `python -m fopoc prepare-assets --workload pets_adapter_v1`; download official images/annotations and the explicitly named checkpoint; verify hashes and decoding; create the deterministic split, duplicate groups, exclusions, transforms, and class-order manifests. Build float32 layer3 caches and unadapted pooled features. Fit/validate the common head with the exact objective in §6, select its regularization and temperature on development, and freeze it. Do not evaluate test metrics.

**Tests/outputs:** train/development/test ID disjointness, expected/actual counts, no exact-pixel duplicates crossing training/development or trainval/protected test, cache-versus-live logits, head stationarity, 65,536 adapter count, full preprocessing/provenance report. Test cache tensors in the gradient path. **Resources:** measured downloads/extraction plus ≈1.374 GiB full feature cache, CPU head fitting, and one prefix/suffix pass per preparation need; record cold setup seconds and bytes.

**Gate/next:** pass to WP03 if manifests and numeric equivalence hold. Quarantine corrupted files under a recorded rule and regenerate manifests before training. If class mapping or cache/head integrity fails, fix preparation; do not interpret resulting optimizer performance. Any change to exclusions after protected evaluation requires an amendment and cannot be presented as pristine confirmation.

### WP03 — Build the oracle ledger and establish adaptation headroom

**Objective/hypotheses:** validate the measured workflow and show W1 has a meaningful adaptation opportunity before expensive candidate tuning.

**Inputs/entry:** frozen WP02 assets. **Actions/files:** implement `oracle.py`, `budget.py`, `runner.py`, sampler/checkpoint interfaces, and gradient references. Run `python -m fopoc validate-oracle --workload pets_adapter_v1` and `python -m fopoc headroom --config configs/w1_headroom.json`. Audit no-grad boundaries, same-batch comparisons, exact restoration, synchronized timestamps, row IDs, canonical float32 states, and segment-example counts. Tune SGD/momentum/Adam on W with the bracket rule; compare development NLL and accuracy against head/temperature controls.

**Tests/outputs:** scripted known-call sequences match the ledger exactly; a rejected or interrupted trial leaves committed W unchanged; CPU/MPS and microbatch paths agree within calibrated error; resumed tiny runs match uninterrupted runs; first-order gradients are nonzero where expected. `headroom.json` contains development curves, training loss, gradient norms, and tuning boundary evidence.

**Resources:** first full-model calibration and short learning-curve ladder, initially with small physical batches. **Gate/next:** if gradients/measurements are broken, repair. If training can improve but development cannot, diagnose overfitting and report limited generalization headroom. If adequate tuning reveals no adapter opportunity, activate the predeclared layer2 amendment or use W0/W2 for a narrower mechanism study. Only genuine headroom proceeds to a useful-adaptation competitiveness claim.

### WP04 — Reproduce eligible baseline mechanisms

**Objective/risk:** prevent an apparent advance caused by obsolete, broken, or unnecessarily expensive comparators.

**Inputs/entry:** WP03 oracle passed; §7 source specifications and native references. **Actions/files:** implement simple ZO/hill climbing, efficient fresh logit TR, residual-model controls, and the mandatory contemporary portfolio in `methods/`; complete each method card. Use `python -m fopoc validate-method --method <registered_id> --suite fidelity`. Check each one-step update against an independently expressed tiny reference, native state refresh, scaling, query formula, and special cases. Implement/validate DFBGN's native budget/valid-incumbent bridge; do not bypass its internal state semantics.

**Tests/outputs:** per-method fidelity JSON, equation/source discrepancy log, license/revision record, and measured cost prototype. A local classification port passes by matching the stated adaptation, not by matching a published LLM accuracy. Learned-basis/activation competitors receive at least fidelity tests and a development pilot spanning their refresh policy; they are not excluded just because the candidate initially looks good.

**Resources:** tiny fixtures first; then bounded measured pilots. Forecast the entire portfolio before broad tuning. **Gate/next:** repair mistaken implementations. For a genuine unresolved paper/code ambiguity, include explicitly named source and corrected variants where defensible; otherwise mark the method unverified and narrow the claim. Failed compatibility alone does not make a baseline scientifically inferior. Eligible, valid methods continue to WP06; persist their negative pilots as well.

### WP05 — Implement the minimum complete candidate

**Objective/hypotheses:** test H1/H2 with the exact selected mechanism, without optional proposal networks or GPU solvers.

**Inputs/entry:** WP03 interfaces and WP01 numerics. **Actions/files:** implement `logit_tr_qualified`, physical Gram handling, central responses, the unit-ball solve, realized-step checks, A/A trust agreement, C safeguard, deletion score, and remeasurement. Run `python -m fopoc validate-method --method logit_tr_qualified --suite mechanism`; then `python -m fopoc run --config configs/w0_candidate.json`.

**Tests/outputs:** passing restoration, sign/basis tests, fresh-row/state-hash checks, zero-coefficient score, finite/flat-response behavior, conditional deletion scores, exact calls, and no-network-autograd instrumentation. Produce score-versus-next-batch usefulness, model prediction error versus radius, and accepted/rejected step traces. Compare efficient and matched no-retention controls separately.

**Resources:** W0 first, then a small W1 pilot using locally measured batches; response memory follows B×C×k. **Gate/next:** if local modeling fails on known linear fixtures, repair. If it works on fixtures but fails from measurable W1 nonlinearity/precision, resolve h/radius or label proxy/oracle limitations. If retention is already neutral/worse, keep rho=0 as an eligible winner; do not add a generator to rescue the result without new evidence.

### WP06 — Calibrate resources, tune fairly, and test causal alternatives

**Objective/hypotheses:** determine whether any apparent H2/H3 benefit survives generic optimizations, strong tuning, and real cost.

**Inputs/entry:** WP04/WP05 valid implementations and WP03 headroom. **Actions/files:** implement `python -m fopoc calibrate --config configs/calibration.json`, `python -m fopoc tune --portfolio configs/portfolio.json`, and the development ablation runner. Follow §9 numerical/microbatch calibration, §8 tuning and baseline-only horizon ladder, and §12's required controls. Give all relevant methods static activation proposals, caching, analytic regularizer treatment, and efficient transfers when mathematically compatible. Test k/rank/history/radius sensitivity and retained-direction persistence.

**Tests/outputs:** resource forecast with component seconds/bytes, tuning traces and boundary expansions, strongest eligible comparator, strongest efficient no-retention variant, chosen time/query horizons, a complete development evidence table, and reasons for every exclusion. Display performance versus cumulative tuning expenditure as well as training cost.

**Resources:** now determined by measured pilots and the portfolio, not an assumed fixed duration. **Gate/next:** if the benefit vanishes with a simple optimization, attribute it accordingly and prefer the simpler implementation. If development time loses but queries win, a dated **pre-test amendment** may change the primary resource axis to Q; then run WP07's pilots/sample-size calculation at Q and freeze its two contrasts. Otherwise T remains primary and Q remains prespecified secondary evidence; they are not interchangeable after seeing test outcomes. If effects are noisy but plausible, WP07 calibrates precision; do not declare success. If candidate and efficient no-retention show no demonstrated development difference at the measured precision, stop adding complexity and report the rule's lack of demonstrated value; do not claim formal statistical equivalence without a declared equivalence margin.

### WP07 — Freeze confirmation and demonstrate adequate planned sensitivity

**Objective/risk:** prevent adaptive test selection and unsupported small-sample confidence.

**Inputs/entry:** settled mechanism, eligible comparator set, resolved calibration, and no protected test metrics inspected. **Actions/files:** run the fixed eight development pilot pairs per distinct primary contrast; implement §8.5's sample-size/precision sensitivity calculation; finalize configurations, seeds, budgets, common monitoring, fallback ladder, and statistical procedures. Run `python -m fopoc preregister --config configs/confirmation_draft.json`. Validate and hash the preregistration and all dependencies.

**Tests/outputs:** zero unresolved config values, unique held-out confirmation seed streams, immutable preregistration hash, sample-size worksheet in nats/image and seconds, approximate/model-dependent inference caveats, explicit handling of failures and unspent budgets, and the command list for every run.

**Resources:** measured pilot n0 plus the resulting forecast for n; protect enough storage for all logs/checkpoints. **Gate/next:** pass to WP08 only if the design is executable and its inferential scope is clear. Infeasible n gives a declared precision-limited design, not an invented success threshold. Degenerate SD, unresolved baseline validity, or a test-access breach requires diagnosis/amendment before confirmation. No result-dependent pilot extension or optional-significance stopping.

### WP08 — Execute protected controlled comparison

**Objective/hypotheses:** test the frozen H3/H2 endpoint claims.

**Inputs/entry:** WP07 immutable protocol. **Actions/files:** implement `python -m fopoc confirm-train --preregistration <path>` and `python -m fopoc confirm-evaluate --preregistration <path> --require-hash <hash>`. Run all n pairs in blocked/randomized order. At minimum include candidate, selected strongest forward-only comparator, selected efficient no-retention comparator if distinct, and the selected gradient practical reference. Preserve wall-time and query-budget runs as distinct experiments. Do not choose best checkpoints/seeds or revise a deadline from observed test curves.

**Tests/outputs:** every registered run accounted for, synchronized budget-end checkpoints, all endpoint arrays, failures/fallbacks, query/time/memory tables, paired primary intervals and adjusted p-values, and the separate image-resampling sensitivity analysis. Broader frozen-comparator test results may be secondary; superiority to the whole portfolio needs its own multiplicity-controlled evidence.

**Resources:** n times measured per-method budgets plus charged preparation/evaluation; interruptions use the frozen recovery rules. **Gate/next:** both distinct primary interval upper bounds below zero support the narrowly scoped headline, subject to validity/guardrails. A query-only win is reported as such. Wide intervals are inconclusive; precise neutral/harmful results argue against the selected local value proposition. A protocol or implementation failure invalidates the affected inference and is reported, not silently rerun until favorable.

### WP09 — Robustness and independent scope extension

**Objective/risk:** determine whether an established local effect depends on favorable piecewise-linear geometry, one head/split, or one operating condition.

**Inputs/entry:** WP06 mechanistic evidence; protected W1 conclusions only after WP08. **Actions/files:** execute the preregistered subset of §12 sensitivity checks; implement W2 with its independent data split and correctly normalized block probes; repeat the strongest relevant controls and a strong first-order reference. If W1 inspired a new algorithm or endpoint, register that revision before touching W2 test metrics. Use `python -m fopoc robustness --config configs/robustness.json`.

**Tests/outputs:** variant-specific claims and uncertainty, smooth-network/model-error curves, head/split dependence, preprocessing and inference-calibration controls, and W2 train-from-scratch boundaries. **Resources:** recalibrate W2's actual full-network passes; do not compare its call counts numerically with W1 suffix passes.

**Gate/next:** consistent effects justify a broader **two-workload** empirical statement. Failure on W2 narrows scope; diagnose whether transfer structure existed before rejecting H2 generally. New mechanisms discovered from W1 require untouched confirmation, not new seeds on the same inspected errors. Proceed to WP10 regardless of sign.

### WP10 — Reproduce, report, and hand off the evidence

**Objective/risk:** make favorable and unfavorable findings independently auditable.

**Inputs/entry:** completed or explicitly terminated work packages. **Actions/files:** reproduce critical fixtures in a fresh environment and one representative valid checkpoint/endpoint; check all manifests and budgets; generate `reports/final.md`, tables, standard exported plots, and an archive manifest from raw records. Run `python -m fopoc report --runs <manifest> --output reports/`. Include all exclusions, failed runs, tuning cost, amendments, source discrepancies, and limitations.

**Tests/outputs:** every headline traces to frozen raw data and a script; no test-guided selection disguised as confirmation; no unmatched units or missing charges; environment reconstruction documented; GPL/code/asset redistribution terms respected. **Resources:** mainly CPU analysis and a short replay, measured separately from training.

**Gate/next:** deliver supported/narrowed/negative/inconclusive findings with exact commands and hashes. Fix reporting defects without retuning on protected outcomes. WP11 is allowed only on its evidence trigger; otherwise stop with the completed evidence package.

### WP11 — Conditional scale or mechanism escalation

**Objective/risk:** investigate a plausible scale-dependent effect without excusing repeated local failure.

**Entry trigger:** measured direction persistence, useful response spectrum, stable surrogate agreement, and benefit over the strongest simpler control, or a specific measured bottleneck with a credible remedy. A positive toy loss curve alone is insufficient.

**Actions/outputs:** before increasing q, suffix depth, class count, or adapting multiple layers, state the prediction and scaling model: response storage O(B C k), raw factor storage O(k(m+n)), Gram O(k²(m+n)) via factors, typical model assembly O(B C k²), and dense solve overhead roughly O(k³) per Newton-like step. Measure whether k must grow with q, whether retained subspaces track drift, and how many actual forwards compensate for extra bookkeeping. Preserve the same oracle/cost discipline and amend scope before new protected results.

**Gate/next:** scale locally only when the forecast fits the measured laptop resources and the next experiment can distinguish the prediction. If the effect requires a larger model than the laptop can run, publish the local component evidence and the untested scaling prediction; do not claim it validated. A learned proposal generator is justified only if the measured failure is proposal alignment and the simpler history/activation alternatives have been exhausted, with its training cost and additional oracle access explicitly charged.

## 12. Attribution, robustness, and failure analysis

### 12.1 Required development controls

Run these before fixing a mechanistic explanation. They are separate experiments with their own charged budgets; an extra diagnostic forward in a primary training run is not free. Freeze the subset needed for a confirmatory attribution claim before protected evaluation. If the mechanism is changed in response, record an amendment.

| Question | Matched control / experiment | What would weaken the proposed explanation? |
| --- | --- | --- |
| Do logits add useful information beyond scalar slopes? | On identical states, dictionaries, A/C examples, physical radius, and CPU float64 loss arithmetic, compare scalar projected-gradient steps, output first-order contraction, per-example CE-residual fitting, and full-logit CE fitting. Give scalar quadratic fitting all needed extra calls and charge them. | Any full-model gain disappears once scalar cross curvature or numerically accurate loss differences are afforded equal resources. |
| Does qualification earn its extra C probes? | Compare candidate with efficient fresh TR at equal time and query budgets, and with fresh-matched TR at identical probes. Also fit the fresh method on the full union without qualification as a separately tuned practical variant. | Candidate beats only the matched-information version that computes unused C responses. |
| Is fresh training qualification the useful part? | Same-batch deletion score; C-qualified score; random retention with the same slot count; largest coefficient magnitude; previous-update direction/momentum; LOZO+ paired selection. Pair fresh streams and batches, allowing retained slots to diverge. | Same-batch, random, or cheap momentum retention explains the gain; C's partition reduces useful fitting data. |
| Are retained directions reusable? | At the accepted post-update parameters, use a small charged diagnostic dictionary containing selected and rejected directions and fit a joint step on fresh A′, then evaluate actual full-step and fixed-other-coefficient deletion losses on independent D′. Signed line probes are a cheaper secondary proxy. | Conditional usefulness does not transfer better than simple scores. A complementary direction may fail the standalone line test while helping jointly; that alone does not reject H2. |
| Is low rank doing the work? | Dense versus rank-one unit-energy proposals with matched realized perturbation scales; static train-only PCA proposals available to all compatible methods. | Advantage vanishes after sharing a static activation basis, or proposals merely have different effective radii. |
| Does curvature explain a gain? | Full outer-loss fit versus its first-order model and diagonal coefficient approximation; scalar methods including LOREN/CurvZO and, if needed, HiZOO; exact projected Hessian on W0 only. | Extra query population or step-size adaptation explains it; no advantage survives close curvature controls. |
| Is the architecture unusually favorable? | Track ReLU pattern changes and response prediction error; repeat in smooth tanh W0/W2 networks and the declared alternative adapter location. | Benefit occurs only inside fixed activation regions and vanishes with inner-network curvature/drift. This narrows scope. |
| Is NLL improvement just calibration? | Frozen head, development-fitted head temperature, and per-method post-training temperature fitted on development; report accuracy and classwise loss. | Cheap calibration removes the NLL gain without a useful accuracy/resource advantage. |
| Are generic optimizations responsible? | Share prefix cache, microbatching, analytic regularizer, CPU vectorization, accurate/efficient scalar transfers, and basis construction. Show cold/warm costs. | Improvement is an implementation optimization available to all competitors. |
| Does score extrapolation matter? | Record deletion-state norms; compare the declared score with a named radius-clipped deletion ablation. | Retention depends on unreliable extrapolated model values rather than next-batch usefulness. |

Direction-level correlations are descriptive measurements nested within a run. Thousands of probe scores are not thousands of independent training trials. For uncertainty about retention-policy effects, aggregate within independent paired runs or use an explicitly hierarchical model; do not inflate significance by treating every direction/batch as independent.

### 12.2 Harness validation and negative controls

The harness must pass both **equivalence** and **known-difference** fixtures. Run identical algorithms under different labels with identical random streams; mathematical endpoints should agree. On a controlled quadratic, inject an exact useful direction and show the pipeline detects its known improvement. Use independent data in the positive-transfer synthetic fixture, and a rapidly drifting subspace where stale retention should lose. Repeat with no label signal to expose leakage; failure to generalize is expected, although an optimizer can still lower training loss.

Test class-logit offsets, class-order permutations, direction sign flips, duplicate directions, zero responses, nonorthogonal cancellation, bilinear zero initialization, and identical versus unpaired stochastic plus/minus evaluations. With zero noise, identical base reevaluation should not create spurious acceptance. With controlled stochastic noise, record common-random-number pairing and demonstrate what breaks when it is removed. A deliberate negative control may fail learning while passing correctness.

Validate statistical reporting using fabricated paired arrays with analytically checkable mean/SD/intervals, duplicate contrasts collapsing to m=1, and missing/failed runs. Identical algorithms/streams must agree. On independently random labels, there should be no reproducible above-chance predictive information about those labels; legitimate training-loss or relative NLL differences can still arise from regularization/calibration. Use fabricated or identical-method nulls for type-I-error simulation. One null-control p-value does not certify the analysis.

### 12.3 Sensitivity and boundary regimes

Vary one important dimension at a time before a factorial sweep: h and Delta independently; k and numerical response rank; damping and true weight penalty; A/C sizes and split ratio; retention fraction/age; class count/output saturation on W0; ambient versus active dimension; response noise and subspace drift. Use the development-selected configuration as the center and expand boundary optima. Aging may be a diagnostic but must not be inserted silently into the primary rule.

Evaluate another head initialization and an independently hashed training/development split as separately scoped robustness, because the primary intervals condition on the original head/split. A shared deterministic augmentation set is a later option: charge preparation/transformation and offer it to all methods. Do not cherry-pick breeds, exclude difficult examples post hoc, or infer robustness from one favorable slice. Report per-breed failures with multiplicity caveats.

Test cold- and warm-cache conditions, physical microbatch changes preserving logical batches, recorded background-load blocks, and fresh-process memory. A storage/page-cache advantage can be real but must be identified; it is not automatically a better learning rule.

### 12.4 Failure classification and mandatory next action

| Observed condition | Classification | Required next step |
| --- | --- | --- |
| Wrong updates, missing calls, stale rows, broken restoration, incompatible tensors | Implementation/measurement failure | Repair and repeat affected gates; invalidate affected comparisons. |
| FO cannot exploit the adapter after adequate tuning | Unsuitable opportunity/proxy for useful-adaptation comparison | Diagnose and use the predeclared amended workload; preserve the original finding. |
| W1 has adaptation headroom but no useful directional persistence | Negative evidence for H2's premise on this workload | Report the absence; an amended regime may be investigated, but cannot erase this negative result as merely a bad proxy. |
| Stable algebra but no observable float32 probe interval | Numerical oracle limitation | Check staged transfers and CPU references; amend the common execution path or declare local feasibility unresolved. |
| OOM, insufficient storage, or unacceptable measured runtime | Local resource limitation | Apply faithful fallback; forecast again; report reduced scope without asserting scientific impossibility. |
| Intervals include plausible benefit and harm | Insufficient precision | Report inconclusive at the planned resolution; no significance-driven additions. |
| Candidate learns, but efficient no-retention or a contemporary comparator matches/beats it | Mechanism works without demonstrated added value | Prefer the simpler/stronger method; report a negative refinement result. |
| H1 works, persistence exists, and qualification reliably harms progress | Evidence against this selection rule | Reject the rule in the tested regime; a new rule is a new hypothesis. |
| Query gain disappears in wall time | Isolated information-efficiency benefit | Quantify overhead and report the scoped tradeoff. |
| Gain disappears after calibration, extra tuning, or baseline correction | Confounded interpretation | Withdraw the stronger claim and preserve the correction. |

## 13. Interpretation and evidence-triggered alternatives

### 13.1 What each outcome means

**Best plausible outcome:** the locked method improves the protected primary endpoint against both required comparators, survives strong baseline tuning and shared optimizations, and attribution supports a benefit from cross-batch reusable directions. Report absolute effects, runtime/query savings, and reliability/memory tradeoffs. The contribution remains local empirical evidence for the specified rule; a general optimization principle requires stronger theory and broader tests.

**Useful narrower outcome:** output-model fitting is effective, but qualification does not help; choose the efficient established no-retention method. Alternatively, qualification helps queries but CPU/selection overhead removes its time advantage. Both findings can inform research without an end-to-end breakthrough claim.

**Negative outcome with adequate sensitivity:** the candidate/refinement fails against strong alternatives despite measured adaptation headroom and stable probes. This argues against the selected rule for this application. It does not prove that all forward-only learning or full-output information is useless.

**Inconclusive outcome:** the oracle is too noisy, the proxy lacks adaptation headroom, or required precision exceeds measured resources. Measured absence of directional persistence on an otherwise useful task remains negative evidence for H2, as specified in §12.4. State the missing observation that would resolve it. A failed installation, inaccessible asset, underpowered design, or unsuitable toy is not an architectural conclusion.

### 13.2 Evidence-triggered pivots in priority order

1. **Qualification adds no value but the output solver does:** drop qualification and study the established composite/DFBGN/RSES relationship. A sound implementation or benchmark is an engineering/empirical contribution, not automatically a new method.
2. **Static activation geometry explains the gain:** adopt a shared PCA/AGZO-style basis and test whether an output-model advantage remains. An expensive adaptive history needs measured value.
3. **Scalar curvature or a population estimator wins:** use that method as the practical result and analyze why full-output handling was unnecessary.
4. **Response fitting is accurate but CPU work dominates:** profile coefficient solves, vectorization, transfer, and DFBGN residual loops. Apply shared parity-tested optimizations before changing the scientific claim.
5. **History fails only under fast drift:** investigate a measured age/novelty criterion or alternative adapter placement as a new development hypothesis. Hindsight-selected retention is not the original confirmed algorithm.
6. **Local evidence supports a scale-dependent effect:** follow WP11, measuring required k and overhead as scale grows. Remote-GPU experiments are future work, not a dependency of the core PoC.

A learned perturbation generator is not the default escape route. ZO-Finetuner explores meta-learned proposal policies with additional training information; ZO-LDSD directly learns a Gaussian search mean from forward-only feedback. These make the broad generator idea established territory. A later generator must justify a specific feedback advantage and charge its learning/history costs. [R22, R48]

The strongest case **for** this experiment is that vector responses and known outer curvature can exploit complementary directions while fresh training examples filter accidental relevance. The strongest case **against** it is that existing composite/subspace solvers already exploit that information, C doubles much of the probe cost, a static feature basis may suffice, and deletion scores may select noise or extrapolation. The gates allow either case to prevail.

## 14. Audit record and residual limitations

### 14.1 Three distinct audits and the revisions they caused

The draft was challenged independently for scientific/prior-art validity, experimental validity, and executable M3 feasibility. The following are substantive corrections incorporated into this handoff, not claims about experiments that have already succeeded.

| Audit | Challenge or defect found | Revision and verification path |
| --- | --- | --- |
| Scientific/prior-art | The broad output-recombination idea overlaps residual/composite DFO, EKI, RSES, and PseuZO. | Restricted potential novelty to the explicit qualification rule; added exact equivalence/limiting cases and mandatory simpler controls. |
| Scientific/prior-art | LOREN/CurvZO and late-2026 learned-subspace competitors could explain an apparent advance. | Added current baseline pilots, inspected equations/source, and documented native versus corrected variants. |
| Scientific/prior-art | Full-output curvature is unusually accurate in the chosen piecewise-affine-in-W architecture. | Qualified the Hessian statement; required activation-boundary diagnostics and smooth nonlinear boundary workloads. |
| Scientific/prior-art | Positive deletion scores can be tautological, outside the trust region, or identify already-exhausted directions. | Added exact quadratic identities, deletion-norm logging, post-update conditional-usefulness diagnostics, and a negative exhaustion fixture. |
| Experimental validity | No-retention was paying for unused C responses; scalar competitors were denied exact regularizer information. | Added efficient no-retention variants, separate matched-probe ablations, and analytic regularizer contributions with native conventions disclosed. |
| Experimental validity | Different float32/float64 loss arithmetic could mimic a vector-information advantage. | Required numerically matched scalar loss options and explicit transfer cost. |
| Experimental validity | Pilot extension and confidence language were insufficiently defined. | Fixed pilot size, explicit normal-model variance planning, max-n across contrasts, simultaneous intervals, and separate conditional/image uncertainty. |
| Experimental validity | Query and time wins could become interchangeable after test inspection. | Default time-primary protocol; query-primary amendment must precede pilots/preregistration or remain secondary. |
| Experimental validity | Random-label relative NLL differences and missing persistence could be dismissed incorrectly. | Corrected null controls; absence of persistence on a useful real workload remains negative evidence for H2's premise. |
| Implementation/feasibility | MPS float64 transfer, inference tensors, physical-radius tolerance, and async timestamps could break correctness. | Staged CPU conversion, ordinary caches, unit-ball solve, realized-step validation, exact restoration, and synchronized commits. |
| Implementation/feasibility | Native DFBGN lacked the assumed time/callback/resume API. | Pinned actual API, native maxfun, explicit instrumentation/uninterrupted alternatives, objective scaling, and incumbent parity tests. |
| Implementation/feasibility | AIM activation dimensions, W2 block normalization/biases, and abrupt-loss accounting were ambiguous. | Correct dimensions and covariance scaling; explicit vector extensions; write-ahead ledger and uncertain-tail policy. |
| Implementation/feasibility | Fixed runtime/memory assumptions could hide an infeasible confirmation program. | Discover actual machine; use tensor estimates only as estimates; measure throughput and forecast tuning/pilot/n before escalation. |

The audits also checked that forward-only methods never receive hidden primary-network derivatives, the real task is more than cached linear-head fitting, all methods share preprocessing/cache opportunities, source discrepancies are not silently “fixed,” and unfavorable runs have a complete reporting route. Corrected specifications were re-read; unresolved items below are explicit experimental/access gates rather than asserted results.

### 14.2 Stress scenarios and planned response

**Lower-memory laptop or older macOS:** discovery and operation gates select a supported stack and faithful microbatch/CPU path. No maximum-memory assumption or silent precision/model reduction is allowed.

**Tiny but real improvement:** pilot-calibrated precision can detect small effects if feasible, with no arbitrary percentage requirement. Report a statistically supported small effect as small; inspect calibration/accuracy/resource tradeoffs.

**Noisy or unfavorable result:** retain failed/negative runs, use the fixed analysis, and distinguish a narrow interval against the claim from inadequate precision. Do not retune on protected outcomes or reclassify every real-workload negative as an unsuitable proxy.

**A new stronger baseline or exact prior-art match appears:** add it during development and repeat the affected gate. After protected evaluation, describe the earlier result's scope and obtain new confirmation for a changed claim. An earlier exact rule removes the novelty claim without invalidating a useful replication.

**Solver crash, zero responses, or budget cutoff:** restore canonical state, inspect transfer/precision/inner-solver diagnostics, preserve native incumbents, and charge all knowable/reserved work. No fabricated success or free restart.

**Need for larger scale:** use measured alignment/rank/persistence and cost scaling to justify the next local test. If the laptop cannot reach the relevant regime, report the strongest demonstrated component claim and the precise untested scaling prediction.

### 14.3 Remaining uncertainties

No local training quality, M3 throughput, peak memory, actual asset download, package lock, or confirmatory sample size has been measured here. Those values are intentionally resolved by early work packages. The qualification rule has no established convergence guarantee in this stochastic setting and may systematically retain exhausted directions. A precise prior-art search cannot establish universal novelty or legal freedom to operate.

The one-adapter real task favors shared caching and stable activation geometry; it does not reproduce multilayer LLM drift or full-model memory pressure. ImageNet pretraining overlap cannot be completely reconstructed from the checkpoint alone. Repeated animal identities and a single task limit population generalization. Run-level t inference relies on distributional approximation, especially with small n or failure mixtures. Some recent preprints lack complete code or internally consistent source behavior; the plan identifies specific adaptations instead of claiming exact reproduction.

The coding agent's first deliverable is therefore the WP00/WP01 evidence and decision record, followed by measured headroom and fidelity—not a promised optimizer victory.

## 15. References and search coverage

### 15.1 How to read the evidence ledger

References are ordinary document identifiers. URLs/DOIs below are stable entry points; source commits and inspected preprint versions are included where material. Unless separately stated, access/inspection occurred during this research through **29 October 2026**. Paper findings are author-reported under their own experiments, not locally reproduced. Algorithm equations and adaptations essential to execution are supplied in the body; the coding agent is not expected to reconstruct the research direction from these links.

### 15.2 Foundations and closest output-based methods

**R1 — James C. Spall (1992), “Multivariate Stochastic Approximation Using a Simultaneous Perturbation Gradient Approximation.”** IEEE Transactions on Automatic Control. [DOI](https://doi.org/10.1109/9.119632); [author-hosted paper](https://www.jhuapl.edu/spsa/pdf-spsa/spall_tac92.pdf). Inspected foundational estimator description. Establishes paired stochastic perturbation prior art; does not demonstrate this neural adaptation or qualification mechanism.

**R2 — Tim Salimans et al. (2017), “Evolution Strategies as a Scalable Alternative to Reinforcement Learning.”** [Paper](https://arxiv.org/abs/1703.03864); [OpenAI research account](https://openai.com/index/evolution-strategies/). Inspected primary research description and reported conditions. Distributed Atari/MuJoCo results support forward-only policy learning; the authors also describe poor supervised-MNIST scaling. Neither result is an equal-budget M3 supervised comparison.

**R3 — Sadhika Malladi et al. (NeurIPS 2023), “Fine-Tuning Language Models with Just Forward Passes.”** [Proceedings](https://proceedings.neurips.cc/paper_files/paper/2023/hash/a627810151be4d13f907ac898ff7e948-Abstract-Conference.html); [MeZO code](https://github.com/princeton-nlp/MeZO). Inspected paper/official repository estimator and memory-efficient seed regeneration. Establishes the strong basic ZO baseline and workload-specific pretrained-LM memory evidence. Its reported ratios are not predictions for W1.

**R4 — Nikola Kovachki and Andrew M. Stuart (2019), “Ensemble Kalman Inversion: A Derivative-Free Technique for Machine Learning Tasks.”** [DOI](https://doi.org/10.1088/1361-6420/ab1c3a); [preprint](https://arxiv.org/abs/1808.03620). Inspected full method and learning examples, including generalized loss formulations. Provides vector-response/ensemble subspace prior art and the equivalence challenge. Ensemble-span restrictions and problem assumptions matter.

**R5 — Coralia Cartis and Lindon Roberts (2019), “A derivative-free Gauss–Newton method.”** Mathematical Programming Computation 11, 631–674. [DOI](https://doi.org/10.1007/s12532-019-00161-7); [author institution record](https://www.maths.ox.ac.uk/node/32599); [DFO-GN code](https://github.com/numericalalgorithmsgroup/dfogn). Inspected residual interpolation and trust-region formulation. Establishes that vector residuals provide quadratic loss models from linear response models; deterministic smooth assumptions do not transfer automatically to the candidate.

**R6 — Coralia Cartis and Lindon Roberts (online 2022; volume 2023), “Scalable subspace methods for derivative-free nonlinear least-squares optimization.”** Mathematical Programming 199, 461–524. [DOI](https://doi.org/10.1007/s10107-022-01836-1); [official DFBGN repository](https://github.com/numericalalgorithmsgroup/dfbgn); [user guide](https://numericalalgorithmsgroup.github.io/dfbgn/build/html/userguide.html); [pinned solver source](https://github.com/numericalalgorithmsgroup/dfbgn/blob/82da43b59acdd0788291cf5addd5144766954743/dfbgn/solver.py). Inspected subspace method, API, solver/model termination, and GPL-3.0-or-later source. Closest reusable structured-residual baseline; no native wall-time/callback/resume API at that revision.

**R7 — D. Davar and G. N. Grapiglia (2025), “TRFD: A Derivative-Free Trust-Region Method Based on Finite Differences for Composite Nonsmooth Optimization.”** SIAM Journal on Optimization 35, 1792–1821. [DOI](https://doi.org/10.1137/24M1701587); [full preprint](https://arxiv.org/html/2410.09165v1); [author code](https://github.com/danadavar/TRFD). Inspected known-convex-outer-function model and assumptions. Directly undermines novelty of optimizing CE on a linearized vector-output map. Its deterministic regularity/convergence setting is different from stochastic ReLU qualification.

**R8 — Francesco Alemanno, “Residual subspace evolution strategies for nonlinear inverse problems.”** [arXiv:2512.10325v2](https://arxiv.org/html/2512.10325v2); [Julia artifact DOI](https://doi.org/10.5281/zenodo.17872272). First arXiv submission is December 2025; the inspected rendered manuscript is dated 24 August 2026, a metadata distinction preserved here. Inspected ridge recombination, evaluation counts, experiments, and stopping-policy details. This is extremely close to original Idea C. Code was located; its redistribution license was not established. Reported regression/calibration successes are not proof of superiority under this plan's protected protocol.

**R9 — Pengyun Yue, Xuanlin Yang, Mingqing Xiao, and Zhouchen Lin (NeurIPS 2025), “PseuZO: Pseudo-Zeroth-Order Algorithm for Training Deep Neural Networks.”** [Proceedings](https://papers.nips.cc/paper_files/paper/2025/hash/9a9afa70eead1805f00e3a0df2a41157-Abstract-Conference.html); [official code](https://github.com/YangBigMn/PseuZO). Inspected paper and trainer/history code. Direct output-Jacobian/outer-gradient/EMA prior art. The classification adaptation, positional row history, EMA scaling, and source decay differences are explicit in §7.6; published language-task results are contextual.

### 15.3 Structured, adaptive, and adjacent learning methods

**R10 — Ziming Yu et al. (ICCV 2025), “Zeroth-Order Fine-Tuning of LLMs in Random Subspaces.”** [Proceedings](https://openaccess.thecvf.com/content/ICCV2025/html/Yu_Zeroth-Order_Fine-Tuning_of_LLMs_in_Random_Subspaces_ICCV_2025_paper.html); [inspected preprint](https://arxiv.org/html/2410.08989v3); [SubZero code](https://github.com/zimingyy/SubZero). Inspected low-dimensional perturbations, refresh policy, experimental context, and GPL-3.0 repository metadata. Original central-probe SubZero is distinct from the newer coefficient-Adam method.

**R11 — Ziming Yu, Shuyao Xiao, Xingyu Zhao, Sike Wang, Pan Zhou, Peiyu Zang, Xiangda Yan, Yongjie Yang, and Jia Li (28 September 2026, v2), “SubZero+: Memory-Efficient Adaptive Zeroth-Order LLM Fine-Tuning in Random Subspaces.”** [Pinned PDF](https://arxiv.org/pdf/2608.15665v2). Downloaded and inspected algorithm, defaults, moment resets, and evaluation-count matching. A stale v1 title appears in search results; this plan uses v2. A distinct released implementation was not verified. Preprint results do not establish local classification performance.

**R12 — Yiming Chen, Yuan Zhang, Liyuan Cao, Kun Yuan, and Zaiwen Wen (ICLR 2025; 2024 preprint), “Enhancing Zeroth-order Fine-tuning for Language Models with Low-rank Structures.”** [Inspected preprint](https://arxiv.org/html/2410.07698v1); [LOZO code](https://github.com/optsuite/LOZO). Inspected low-rank perturbations, reused factor, 1/r convention, and low-rank momentum. Establishes that compact structured/history-based updates are not new; its LLM scale is contextual.

**R13 — Jinjie Fang, Chengxun Jin, Tianxing Man, Yi Chang, and Bin Gu (ICML 2026), “LOZO+: Provably Efficient Zeroth-Order Fine-Tuning via Greedy Low-Rank Subspace Selection.”** [Proceedings/PDF](https://proceedings.mlr.press/v306/fang26k.html); [source revision](https://github.com/fangjj2000/LOZO-/tree/9cdf1c27ebe364b35b49467f7072c9640d3978d7). Inspected official PDF, `LOZO+/GELOZOtrainer.py`, scripts, and metadata. Direct loss-feedback retention competitor. The batch-selection/count discrepancies and proposed correction are detailed in §7.4. No repository-wide reuse license was established.

**R14 — Wei Lin, Yining Jiang, Qingyu Song, Qiao Xiang, and Hong Xu (ICML 2026), “AGZO: Activation-Guided Zeroth-Order Optimization for LLM Fine-Tuning.”** [Proceedings](https://proceedings.mlr.press/v306/lin26n.html); [preprint](https://arxiv.org/html/2601.17261v4); [author implementation](https://github.com/Yining-Jiang/ActivationGuidedZO-NPU). Inspected activation-subspace construction, power iterations, scaling, and available code context. Supports a strong activation-informed competitor; an NPU/CUDA-oriented implementation does not certify MPS support.

**R15 — Yue Xie et al. (28 September 2026, v1), “AIM-ZO: Activation-Informed Subspace Maintenance for Zeroth-Order LLM Fine-Tuning.”** [Pinned preprint](https://arxiv.org/pdf/2609.35257v1); [author code](https://github.com/EkkoXy/AIM-ZO). Downloaded/inspected maintained-versus-active subspace, Oja update, shared/sampled directions, and population estimator. Listed as a submitted preprint, not an accepted ICLR 2027 paper. Especially relevant to claims about persistent meaningful directions; static W1 input geometry is a limitation of the local adaptation.

**R16 — Chuang Tong and Ling Chen (10 October 2026, v1), “Zeroth-Order Fine-Tuning of LLMs in Learnable Low-Rank Subspaces.”** [Pinned preprint](https://arxiv.org/pdf/2610.15225v1). Downloaded/inspected AdaSub-ZO basis learning, extra-query cost, and momentum transport, including visual checking of the printed matrix expression. No runnable official implementation was verified. The dimension-correct transport in §7.5 is disclosed as a correction, not an exact reproduction.

**R17 — Sizhe Dang et al. (2025 preprint; ICLR 2026 code release), “Fast Zeroth-Order Optimizer for Fine-Tuning Large Language Models towards Adam-Scale Speed.”** [Inspected v2](https://arxiv.org/html/2506.09034v2); [FZOO repository](https://github.com/DKmiyan/FZOO). Inspected estimator, paper/code epsilon and population-SD conventions, serial path, and MIT license. Separates algorithmic population normalization from reported CUDA parallel speedups; neither establishes M3 speed.

**R18 — Xiaoxing Wang, Xiaohan Qin, Xiaokang Yang, and Junchi Yan (NeurIPS 2024), “ReLIZO: Sample Reusable Linear Interpolation-based Zeroth-order Optimization.”** [Proceedings](https://papers.nips.cc/paper_files/paper/2024/hash/1b3750390ca8b931fb9ca988647940cb-Abstract-Conference.html); [ReLIZO code](https://github.com/Thinklab-SJTU/ReLIZO). Inspected historical-query/interpolation mechanism and source availability. Supports history-reuse prior art; experiments include synthetic functions, attacks, architecture search, and PEFT. Full native experimental reproduction was not performed.

**R19 — Fei-Yu Liu, Zi-Niu Li, and Chao Qian (IJCAI 2020), “Self-Guided Evolution Strategies with Historical Estimated Gradients.”** [Proceedings](https://ijcai.org/proceedings/2020/205); [DOI](https://doi.org/10.24963/ijcai.2020/205). Inspected primary description and historical-gradient subspace construction. Establishes adaptive exploration/exploitation from prior estimates; RL/black-box benchmarks are not direct W1 comparisons.

**R20 — Krzysztof M. Choromanski, Aldo Pacchiano, Jack Parker-Holder, Yunhao Tang, and Vikas Sindhwani (NeurIPS 2019), “From Complexity to Simplicity: Adaptive ES-Active Subspaces for Blackbox Optimization.”** [Proceedings](https://papers.nips.cc/paper_files/paper/2019/hash/88bade49e98db8790df275fcebb37a13-Abstract.html). Inspected ASEBO's primary method description and linked work. Dynamic sensing subspaces and adaptive exploration predate the proposal; no broad “learn directions” novelty follows here.

**R21 — Niru Maheswaranathan, Luke Metz, George Tucker, Dami Choi, and Jascha Sohl-Dickstein (ICML 2019), “Guided evolutionary strategies: augmenting random search with surrogate gradients.”** [Proceedings](https://proceedings.mlr.press/v97/maheswaranathan19a.html). Inspected surrogate-subspace search distribution and assumptions. Important adjacent prior art, but externally supplied surrogate gradients may exceed this project's ordinary-forward oracle.

**R22 — Kairun Zhang, Haoyu Li, Yanjun Zhao, Yifan Sun, and Huan Zhang (ICML 2026), “Learning a Zeroth-Order Optimizer for Fine-Tuning LLMs.”** [Official proceedings entry](https://proceedings.mlr.press/v306/zhang26da.html); [inspected full preprint](https://arxiv.org/html/2510.00419v2); [ZO-Finetuner code](https://github.com/ASTRAL-Group/ZO_Fine_tuner). Inspected optimizer inputs, per-block learned scales, and meta-training process. Direct learned-noise prior art; downstream forward-only execution does not erase its gradient-based training expenditure.

**R23 — Yanjun Zhao, Sizhe Dang, Haishan Ye, Guang Dai, Yi Qian, and Ivor Tsang (ICLR 2025), “Second-Order Fine-Tuning without Pain for LLMs: A Hessian Informed Zeroth-Order Optimizer.”** [HiZOO preprint](https://arxiv.org/html/2402.15173v4); [code](https://github.com/Yanjun-Zhao/HiZOO). Inspected three-point diagonal-curvature method and code availability. A scalar curvature reference; extra calls/diagonal state must count. No blanket repository reuse license was established.

**R24 — Atılım Güneş Baydin, Barak A. Pearlmutter, Don Syme, Frank Wood, and Philip Torr (2022), “Gradients without Backpropagation.”** [Preprint](https://arxiv.org/abs/2202.08587); [primary author-organization record](https://www.microsoft.com/en-us/research/publication/gradients-without-backpropagation/). Inspected forward-mode directional-derivative formulation. Avoiding reverse-mode backpropagation is broader than ordinary inference-only optimization; this route uses a different oracle.

**R25 — Robert O'Shea and Bipin Rajendran (2026), “Closed-form feedback-free learning with forward projection.”** Nature Communications. [DOI/primary article](https://doi.org/10.1038/s41467-026-69161-1). Primary abstract/metadata inspected; full article access was blocked by the publisher identity redirect in this environment. Included only as an adjacent architectural/learning-rule boundary, not as a reconstructed baseline or evidence against the selected mechanism.

### 15.4 Workload, software, and measurement sources

**R26 — Omkar M. Parkhi, Andrea Vedaldi, Andrew Zisserman, and C. V. Jawahar / Oxford VGG (2012), Oxford-IIIT Pet / “Cats and Dogs.”** [Official dataset and paper links](https://www.robots.ox.ac.uk/~vgg/data/pets/). Inspected categories, files, annotation structure, and the current page's CC BY-SA 4.0 terms. Grounds real-data provenance, attribution, and the 37-class workload; actual local manifest validity remains a gate.

**R27 — PyTorch/torchvision contributors, ResNet-18 model documentation (living documentation).** [Official model/weight entry](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html). Inspected explicit weight enum, parameter count, transform description, and asset metadata. Grounds checkpoint/preprocessing selection and tensor estimates. The tested installed release must be locked; a web page alone is not an execution guarantee.

**R28 — PyTorch/torchvision contributors, Oxford-IIIT Pet loader source (living source).** [Official loader](https://docs.pytorch.org/vision/main/_modules/torchvision/datasets/oxford_iiit_pet.html). Inspected official archive URLs/MD5, category mapping, and split handling. The plan retains upstream checks and adds local SHA-256/manifests; no files were downloaded as a training benchmark here.

**R29 — PyTorch/torchvision contributors, repository, compatibility and model source.** [Repository/README/license](https://github.com/pytorch/vision); [ResNet source](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py). Inspected release compatibility, BSD-3-Clause code licensing, pretrained-asset caveats, and weight identifier. Code licensing is distinct from pretrained weights/training-data terms.

**R30 — Han Xiao, Kashif Rasul, Roland Vollgraf / Zalando Research (2017), Fashion-MNIST.** [Official repository/data documentation](https://github.com/zalandoresearch/fashion-mnist). Inspected original counts, format, downloadable assets, and MIT distribution. Provides the independent small from-scratch boundary workload, not a substitute for W1's pretrained-adaptation claim.

**R31 — Apple and PyTorch contributors, PyTorch on Metal/MPS support (living documentation).** [Apple setup/support](https://developer.apple.com/metal/pytorch/); [MPS notes](https://docs.pytorch.org/docs/stable/notes/mps.html); [official documentation source](https://github.com/pytorch/docs/blob/site/main/notes/mps.md). Inspected installation requirements, capability checks, and execution guidance. Documentation tracks differ; the plan uses local operation tests and a tested release lock. Float64-transfer limitations are additionally grounded in R44's source/fix.

**R32 — PyTorch contributors, `torch.autograd.grad_mode.inference_mode` (living documentation).** [Official API](https://docs.pytorch.org/docs/stable/generated/torch.autograd.grad_mode.inference_mode.html). Inspected tensor/autograd restrictions. Grounds ordinary-tensor cache construction so the frozen-suffix gradient reference remains valid. A valid forward-only path alone does not test first-order compatibility.

**R33 — PyTorch contributors, MPS memory APIs (living documentation).** [Current tensor memory](https://docs.pytorch.org/docs/stable/generated/torch.mps.current_allocated_memory.html); [driver memory](https://docs.pytorch.org/docs/stable/generated/torch.mps.driver_allocated_memory.html); [recommended maximum](https://docs.pytorch.org/docs/stable/generated/torch.mps.recommended_max_memory.html). Inspected reported quantities and API availability. Grounds sampled measurements; these values are not automatically historical peaks or disjoint unified-memory pools.

**R34 — PyTorch contributors, reproducibility and MPS synchronization guidance.** [Reproducibility notes](https://docs.pytorch.org/docs/stable/notes/randomness.html); [MPS API](https://docs.pytorch.org/docs/stable/mps.html). Inspected RNG/determinism caveats and synchronization operation. Grounds timing boundaries and the refusal to promise cross-device/version bitwise replay.

**R35 — Apple, MacBook Pro with M3-family technical specifications.** [Official specification page](https://support.apple.com/en-us/117737). Inspected configuration variations. Supports hardware discovery rather than assuming the maximum unified-memory option; no target-laptop facts were inferred from the product page.

**R36 — SciPy contributors, `minimize(method='SLSQP')` API (living documentation).** [Official reference](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html). Inspected constrained-solver, gradient, tolerance, and termination interface. The actual small solve in §4 ran with SciPy 1.17.0 on Linux; future M3 execution uses a locally tested lock.

**R37 — Apple MLX contributors, execution and array documentation.** [Documentation](https://ml-explore.github.io/mlx/build/html/index.html); [lazy evaluation](https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html); [random API](https://ml-explore.github.io/mlx/build/html/python/random.html). Inspected as an alternative Apple-Silicon stack. Lazy evaluation/RNG semantics require their own timing and equivalence validation; no MLX implementation is claimed or required for the initial plan.

**R38 — PyTorch contributors, MPS environment variables.** [Official reference](https://docs.pytorch.org/docs/stable/mps_environment_variables.html). Inspected fallback and memory-watermark behavior. Grounds explicit fallback settings and the decision not to bypass protective allocator limits; numerical support must be tested on the actual build.

**R39 — PyTorch contributors, activation checkpointing.** [Official API](https://docs.pytorch.org/docs/stable/checkpoint.html). Inspected recomputation and reentrant/RNG considerations. Grounds an optional stronger gradient-memory baseline; it requires local parity and timing tests rather than assumed benefit.

**R40 — Xavier Bouthillier et al. (MLSys 2021), “Accounting for Variance in Machine Learning Benchmarks.”** [Proceedings](https://proceedings.mlsys.org/paper_files/paper/2021/hash/0184b0cd3cfb185989f858a1d9f5c1eb-Abstract.html). Inspected sources of benchmark variability and experimental-design implications. Supports distinguishing optimizer seeds, data splits, initialization, and tuning; it does not license treating repeated timings as independent ML outcomes.

### 15.5 Patents, source corrections, and final landscape additions

**R41 — Qualcomm, “On-device neural network training for edge devices,” published US patent application US20250384337A1 (18 December 2025).** [Patent publication text](https://patents.google.com/patent/US20250384337A1/en). Inspected abstract/claim context concerning perturbation-based neural training. Broad-overlap screen only; no legal conclusion or current enforceability assertion is made.

**R42 — TDK, “Machine learning device, machine learning program, and machine learning method,” published application WO2020261509A1 (30 December 2020).** [Patent publication text](https://patents.google.com/patent/WO2020261509A1/en). Inspected ensemble-Kalman/neural-weight-update context. Reinforces prior use of derivative-free ensemble weight adjustment. Patent families, claim construction, national prosecution, and freedom to operate were not exhaustively investigated.

**R43 — Tanmay Gautam, Youngsuk Park, Hao Zhou, Parameswaran Raman, and Wooseok Ha (ICML 2024), “Variance-reduced Zeroth-Order Methods for Fine-Tuning Language Models.”** [Proceedings](https://proceedings.mlr.press/v235/gautam24a.html); [MeZO-SVRG source](https://github.com/amazon-science/mezo_svrg). Inspected anchor/control-variate mechanism, experimental conditions, and Apache-2.0 repository. Context for prompting/batch/variance sensitivity and a conditional comparator when reuse/variance reduction explains a result.

**R44 — PyTorch upstream issue #197715 and fix #197722 (September 2026).** [Issue](https://github.com/pytorch/pytorch/issues/197715); [fix commit](https://github.com/pytorch/pytorch/commit/ee2ad0c945e7c3a16163bf118b8a42c1df50a10a). Inspected reproducer, reported build behavior, and fix semantics. This is an upstream bug report/source finding, not our reproduction on M3. It directly motivates separate float32 transfer and CPU float64 conversion.

**R45 — Hyunseok Seung, Jaewoo Lee, and Hyunsuk Ko (AAAI 2026), “Low-Rank Curvature for Zeroth-Order Optimization in LLM Fine-tuning.”** [DOI](https://doi.org/10.1609/aaai.v40i30.39715); [full paper](https://ojs.aaai.org/index.php/AAAI/article/download/39715/43676); [LOREN source revision](https://github.com/hseung88/loren/tree/b80a32bc394332d1a1f115f41dfb5dcf7b326d37). Inspected covariance/score derivations, trainer/update/default files, and MIT license. Source global-versus-rowwise covariance and sequential-population differences are explicit in §7.9. A small score derivative check was run; LLM performance was not reproduced.

**R46 — Shuo Wang, Ziyu Chen, and Ming Tang (ICML 2026), “CurvZO: Adaptive Curvature-Guided Sparse Zeroth-Order Optimization for Efficient LLM Fine-Tuning.”** [Proceedings/PDF](https://proceedings.mlr.press/v306/wang26z.html); [source revision](https://github.com/Eurekabb/CurvZO/tree/e32b332ba0f623c14392e1a28c6e9b3b6d6b0dfe). Inspected full algorithm/appendix, training/default files, probability/budget recurrences, and license metadata. §7.10 uses explicit paper-equation local variants; source discrepancies and empirical-proxy limitations are not concealed.

**R47 — Jian Feng and Zhihong Huang (January 2026, v4), “Robust and Efficient Zeroth-Order LLM Fine-Tuning via Adaptive Bayesian Subspace Optimizer.”** [Pinned BSZO preprint](https://arxiv.org/abs/2601.01452v4); [author code](https://github.com/AeonianQuill/BSZO). Inspected posterior/subspace algorithm and cached-observation reuse. Relevant Bayesian/history prior art; repeated assimilation of the same cached scalar is not new independent evidence. This plan makes no inherited posterior-confidence guarantee.

**R48 — Valery Parfenov, Grigoriy Evseev, Andrey Veprikov, Nikolay Bushkov, Stanislav Moiseev, and Aleksandr Beznosikov (14 February 2026, v1), “Zero-Order Optimization for LLM Fine-Tuning via Learnable Direction Sampling.”** [Full preprint](https://arxiv.org/html/2602.13659v1); [ZO-LDSD source](https://github.com/brain-lab-research/zo_ldsd). Inspected practical forward-only learned-mean population method and distinction from theoretical alignment dynamics. Direct objection to novelty of training a noise distribution; no dimension-free practical guarantee is imported.

### 15.6 Search scope, stopping rationale, and blind spots

The search began with the user-provided ES, MeZO, SubZero, ZO-Finetuner, and EKI sources, followed references backward to SPSA, residual/Gauss–Newton/composite optimization, ensemble methods, and active-subspace ES, and traced later work through the research cutoff. Equivalent descriptions included vector-valued derivative-free optimization, residual interpolation/recombination, known outer losses, Jacobian estimation from outputs, finite-difference subspaces, ensemble covariance updates, adaptive perturbation distributions, query recycling, and activation-guided search.

Dedicated follow-ups inspected 2025–2026 proceedings/preprints, implementations and recent versions, scalar curvature/variance controls, source-paper discrepancies, MPS numeric/memory APIs, dataset/model terms, and a limited patent screen. The refinement search used combinations such as `derivative-free cross-batch directions retention`, `zeroth-order marginal retention output`, `trust region direction deletion residual subspace reuse`, and `zeroth-order validation batch subspace selection`. These produced overlapping families and irrelevant terminology collisions, but no inspected exact match to the specified C-qualified deletion rule. This is an observation about the search, not proof of uniqueness.

The audits materially expanded the baseline set with LOREN/CurvZO and clarified recent AIM-ZO/AdaSub-ZO. Subsequent focused checks changed implementation safeguards and interpretation rather than revealing another indispensable mechanism for the **narrow** retention claim. The stopping rule was coverage of the mechanism, closest alternatives, material numerical/asset constraints, and resolving identified source inconsistencies—not a paper count. If the implemented claim broadens, the research must reopen around that specific change.

Remaining blind spots include inaccessible/unindexed work, non-English terminology, unpublished or proprietary systems, unreleased recent code, full patent-family/legal analysis, and effects at scales the laptop cannot reproduce. The publisher-blocked R25 was kept adjacent and limited to inspected abstract/metadata; source files/PDFs were used where browser HTML failed for consequential methods. Preprint date/title inconsistencies are identified rather than silently normalized. No search result, author benchmark, or software-support page substitutes for the preregistered local experiment.

## 16. Appendix: executed small-check code

### 16.1 Algebra and coefficient-solve checks

This is the exact small CPU script used for the first seven rows of §4.5. Save this code as `research_checks.py` and run `python research_checks.py` in an environment with NumPy and SciPy. It is supplied reference code, unlike the proposed `fopoc` commands. The observed environment is recorded in §4.5; numeric roundoff can differ across libraries/platforms. The early six-coefficient demo solves directly in beta at fixed radius 0.4; the final production specification adds unit-ball scaling and realized-float32 safeguards. This script is not the candidate trainer or an M3 compatibility test.

```python
"""Small mathematical checks for the research handoff, not an M3 benchmark."""
import json
import platform
import sys
import numpy as np
from scipy.special import logsumexp, softmax
from scipy.optimize import minimize

rng = np.random.default_rng(20261029)
out = {"environment": {"python": sys.version, "platform": platform.platform(),
                       "numpy": np.__version__, "seed": 20261029}}

# Exact one-direction minimization of the spherical quadratic.
q, reps = 64, 100000
v = rng.normal(size=q); v /= np.linalg.norm(v)
u = rng.normal(size=(reps, q)); u /= np.linalg.norm(u, axis=1)[:, None]
fractions = (u @ v) ** 2
out["quadratic"] = {"dimension": q, "replications": reps,
                    "observed_fraction": float(fractions.mean()),
                    "predicted_fraction": 1/q,
                    "monte_carlo_se": float(fractions.std(ddof=1)/np.sqrt(reps))}
del u

# Ridge/Gauss-Newton and the ensemble-space covariance expression agree.
q, m, k = 60, 45, 7
Q = rng.normal(size=(q, k)) / np.sqrt(q)
J = rng.normal(size=(m, q)) / np.sqrt(q)
S = J @ Q; r = rng.normal(size=m); lam = 0.17
a = -np.linalg.solve(S.T @ S + lam*np.eye(k), S.T @ r)
step_reduced = Q @ a
step_covariance = -Q @ S.T @ np.linalg.solve(S @ S.T + lam*np.eye(m), r)
out["ridge_eki"] = {"relative_step_error": float(np.linalg.norm(step_reduced-step_covariance)/np.linalg.norm(step_reduced))}

# A physical-norm penalty is invariant to invertible basis changes.
R = rng.normal(size=(k, k)) + 3*np.eye(k)
G = Q.T @ Q
a = -np.linalg.solve(S.T @ S + lam*G, S.T @ r)
b = -np.linalg.solve((S@R).T @ (S@R) + lam*(Q@R).T @ (Q@R), (S@R).T @ r)
out["physical_gram"] = {"relative_step_error": float(np.linalg.norm(Q@a-Q@R@b)/np.linalg.norm(Q@a))}
bad_b = -np.linalg.solve((S@R).T @ (S@R) + lam*np.eye(k), (S@R).T @ r)
out["coefficient_ridge_counterexample"] = {"relative_change": float(np.linalg.norm(step_reduced-Q@R@bad_b)/np.linalg.norm(step_reduced))}

# A small nonlinear network; analytic JVP here is a reference check, not training.
B, ni, nh, nc = 17, 12, 8, 4
X = rng.normal(size=(B, ni)); W = rng.normal(size=(ni, nh))/np.sqrt(ni)
V = rng.normal(size=(nh, nc))/np.sqrt(nh)
U = rng.normal(size=W.shape); U /= np.linalg.norm(U)
y = rng.integers(nc, size=B)
def f(w): return np.tanh(X@w) @ V
def ce(z): return float(np.mean(logsumexp(z, axis=1)-z[np.arange(B), y]))
z = f(W); e = softmax(z, axis=1); e[np.arange(B), y] -= 1; e /= B
exact_s = ((1-np.tanh(X@W)**2) * (X@U)) @ V
exact_slope = float(np.sum(e*exact_s))
rows = []
for h in [1e-1, 1e-2, 1e-3, 1e-4]:
    zp, zm = f(W+h*U), f(W-h*U)
    scalar = (ce(zp)-ce(zm))/(2*h)
    contracted = float(np.sum(e*(zp-zm)/(2*h)))
    rows.append({"h": h, "scalar": scalar, "contracted_output": contracted,
                 "difference": abs(scalar-contracted), "analytic": exact_slope})
out["first_order_equivalence"] = rows

# Convex coefficient solve: use analytic derivatives only of the small outer loss.
B, nc, k = 23, 5, 6
z = rng.normal(size=(B, nc)); S = rng.normal(size=(B, nc, k))
S -= S.mean(axis=1, keepdims=True)
y = rng.integers(nc, size=B); lam=0.03; radius=0.4
def obj(beta):
    zz = z+np.einsum('bck,k->bc', S, beta)
    p = softmax(zz, axis=1); p[np.arange(B), y] -= 1
    val = np.mean(logsumexp(zz, axis=1)-zz[np.arange(B), y])+lam*np.dot(beta,beta)/2
    grad = np.einsum('bck,bc->k',S,p)/B+lam*beta
    return float(val), grad
res = minimize(obj, np.zeros(k), jac=True, method='SLSQP',
               constraints=[{"type":"ineq", "fun":lambda b:radius**2-np.dot(b,b),
                             "jac":lambda b:-2*b}],
               options={"ftol":1e-12, "maxiter":200})
beta=res.x; grad=obj(beta)[1]
mult = max(0, -float(np.dot(beta,grad))/max(float(np.dot(beta,beta)),1e-30))
out["coefficient_solve"] = {"success": bool(res.success), "iterations":int(res.nit),
     "norm":float(np.linalg.norm(beta)), "radius":radius,
     "regularized_model_decrease":obj(np.zeros(k))[0]-obj(beta)[0],
     "kkt_residual_norm":float(np.linalg.norm(grad+mult*beta))}

# Exact objective equivalence of sqrt-CE residualization.
ce_each=logsumexp(z,axis=1)-z[np.arange(B),y]
rr=np.sqrt(2*ce_each/B)
out["sqrt_ce"]={"absolute_objective_error":abs(float(rr@rr/2-np.mean(ce_each)))}

assert abs(out['quadratic']['observed_fraction']-1/64) < 5*out['quadratic']['monte_carlo_se']
assert out['ridge_eki']['relative_step_error'] < 1e-10
assert out['physical_gram']['relative_step_error'] < 1e-10
assert rows[-1]['difference'] < 1e-8
assert res.success and np.linalg.norm(beta) <= radius+1e-8
assert out['coefficient_solve']['kkt_residual_norm'] < 1e-5
assert out['sqrt_ce']['absolute_objective_error'] < 1e-12
print(json.dumps(out, indent=2))
```

### 16.2 Supplied rerun of the LOREN score check

The following self-contained reconstruction of the tiny Gaussian-density check was saved and rerun during the final audit. It gives the displayed $5.22\times10^{-11}$ relative score error and $4.43\times10^{-16}$ covariance-square-root error with NumPy 2.3.5. The earlier inline expression produced a negligibly different last digit; this supplied script and its rerun are the authoritative reproducible record. The sampled displacement is held fixed when differentiating the density. It verifies algebra only.

```python
"""Tiny CPU check of the corrected LOREN row-wise Gaussian score.

This checks a density derivative, not optimizer performance or MPS support.
The sampled displacement is held fixed during finite differencing.
"""
import json
import numpy as np

rng = np.random.default_rng(20261029)
m, n = 7, 5
rho, epsilon, fd_step = 0.4, 0.03, 1e-5
a = rng.normal(size=n)
Z = rng.normal(size=(m, n))

P = rho * np.eye(n) + np.outer(a, a)
C = np.linalg.inv(P)
t = np.sqrt(rho + a @ a)
xi = 1.0 / (t * (t + np.sqrt(rho)))
T = (np.eye(n) - xi * np.outer(a, a)) / np.sqrt(rho)
D = (Z - xi * np.outer(Z @ a, a)) / np.sqrt(rho)
displacement = epsilon * D

def log_density(a_trial):
    P_trial = rho * np.eye(n) + np.outer(a_trial, a_trial)
    sign, logdet = np.linalg.slogdet(P_trial)
    assert sign > 0
    quadratic = np.sum((displacement @ P_trial) * displacement)
    return 0.5 * m * logdet - quadratic / (2.0 * epsilon**2)

score = m * a / (rho + a @ a) - D.T @ (D @ a)
finite_difference = np.empty(n)
for j in range(n):
    e = np.zeros(n)
    e[j] = fd_step
    finite_difference[j] = (log_density(a + e) - log_density(a - e)) / (2.0 * fd_step)

reference_norm = np.linalg.norm(finite_difference)
assert reference_norm > 0
out = {
    "numpy": np.__version__,
    "seed": 20261029,
    "m": m, "n": n, "rho": rho, "epsilon": epsilon,
    "finite_difference_step": fd_step,
    "score_relative_error": float(np.linalg.norm(score - finite_difference) / reference_norm),
    "flipped_score_relative_error": float(np.linalg.norm(-score - finite_difference) / reference_norm),
    "covariance_square_root_relative_error": float(np.linalg.norm(T @ T.T - C) / np.linalg.norm(C)),
}
assert out["score_relative_error"] < 1e-7
assert out["covariance_square_root_relative_error"] < 1e-12
print(json.dumps(out, indent=2))
```
