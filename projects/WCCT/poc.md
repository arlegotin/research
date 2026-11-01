# Witness-Refined Capability-Change Tokens
## A research and local proof-of-concept plan for a behavior-to-possibility compiler

**Research date:** 14 October 2026.  
**Target:** a MacBook Pro with an M3 Max; actual memory, storage, operating system, and usable acceleration must be discovered.  
**Status:** researched proposal with analytical checks and a small exploratory simulation, not a demonstrated learning result.  
**Primary deliverable of execution:** an independently checked comparison showing whether the proposed representation earns an advantage over strong continuous, physical-model, tree-based, and search baselines—or a reproducible explanation of why it does not.

**Navigation:** [Assessment](#1-executive-assessment) · [Problem and hypotheses](#2-problem-scope-assumptions-and-claims) · [Prior art](#3-research-landscape-and-novelty-assessment) · [Executed checks](#4-checks-actually-performed-during-preparation) · [Benchmark](#5-synthetic-benchmark-and-physical-semantics) · [Mechanism](#6-proposed-mechanism-and-implementation-specification) · [Baselines](#7-baselines-and-fair-comparison-rules) · [Evaluation](#8-confirmatory-evaluation-and-statistical-protocol) · [Hardware](#9-m3-max-feasibility-and-hardware-adaptive-execution) · [Work packages](#10-ordered-work-packages-and-decision-gates) · [Ablations](#11-attribution-adversarial-tests-and-failure-analysis) · [Repository](#12-executable-repository-interfaces-and-numerical-conventions) · [Decisions](#13-interpretation-continuation-and-evidence-triggered-pivots) · [Audits](#14-audit-record-remaining-risks-and-final-challenge-tests) · [References](#15-references-and-evidence-register).

---

## 1. Executive assessment

### 1.1 Recommendation

Investigate **witness-refined capability-change tokens (WCCT)** for **external attachment design on partially observed spring-and-latch mechanisms**. A token represents a recurring change in an operating capability, such as widening the range of forces that opens a mechanism without excessive displacement or velocity. It is not an object label, a raw trajectory code, a design parameter bin, or a claim about every physically possible future.

The proposed pipeline is:

> Observe a small, explicitly costed set of response traces → infer a distribution over capability changes produced by candidate attachments → select a desired change → find an attachment predicted to realize it → evaluate it → retain the design only after separate numerical verification.

A code comprises a discrete effect motif, a bounded continuous residual, an explicit interface and validity context, and a record of training counterexamples that required distinctions in the vocabulary. The testable contribution is whether **refining a vocabulary around failures of capability transfer**, rather than curve reconstruction, improves design generalization and measurement efficiency.

This is a narrow, potentially useful research hypothesis—not an established new paradigm. Behavioral representations, action-grounded symbols, intervention-preserving abstractions, functional inverse design, and counterexample-guided refinement already exist. DeepSym is especially close to the original behavioral-token idea; symbolic-geometric abstraction repair is especially close to the refinement idea. Neither a new name nor their combination establishes novelty. [R01–R10]

### 1.2 The most important correction to the brainstorm

**Two systems can have identical observable behavior before an edit and respond differently to the same internal edit.** Therefore, “encode behavior, then infer arbitrary geometry modifications” is not identifiable without additional assumptions or measurements.

The core experiment instead adds known spring/damper components at a standardized observable port. This preserves the original intent—designing capability from behavior—while giving the intervention a well-defined physical meaning. Arbitrary internal geometry changes remain an explicit negative control and possible later extension. Restricting the interface is not itself novel; it is required for a defensible experiment. [R02, R08–R10]

A second correction is equally consequential: **non-destructive, subthreshold probes need not reveal a hidden latch threshold**. The primary experiment consequently allows one standardized release-capable characterization trial on a resettable simulated copy. A safe-probe-only arm tests uncertainty and abstention, not magical recovery of information absent from the observations.

### 1.3 What was established during preparation

An exact spring counterexample confirms the internal-edit ambiguity. A small vectorized latch simulation confirms that the proposed constraints can create a nontrivial opening-force window. It also exposes a weak-baseline trap: across 12 exploratory systems, mean grid coverage was 0.05894 without an attachment, 0.35569 using one retrospectively selected fixed attachment, and 0.39634 using a separate best grid attachment for each system. Thus only **0.04065 coverage units** remained above that in-sample fixed-design reference in this toy check. These are not trained-model results, confidence intervals, general optima, or evidence that tokens help. Section 4 gives the complete conditions and limitations.

### 1.4 What would count as a meaningful result

A useful local result would be a reproducible improvement in independently evaluated opening-force coverage on held-out mechanism templates at a fixed number of simulator trajectories, accompanied by a favorable or clearly disclosed compute/storage tradeoff. Attribution requires an incremental gain over a matched continuous representation and over the same counterexample/label-selection machinery without discrete codes.

A stronger follow-on result would show that a **capability-first proposal procedure**, using the same trained model and information, finds physical realizations more efficiently than directly ranking attachment parameters. That result is separate from predictive compression. Neither establishes arbitrary mechanism invention, real-world robustness, a universal token algebra, or superiority at foundation-model scale.

### 1.5 Immediate priorities

The execution order is numerical correctness → observability → headroom over strong simple search → common baseline harness → minimum token model → controlled pilot → protected comparison → attribution and independent confirmation. Do not begin with a transformer, a large codebook, a robot simulator, a broad multimodal dataset, or an expensive training run.

---

## 2. Problem, scope, assumptions, and claims

### 2.1 Intended user and current workflow

The hypothetical beneficiary is an engineer repeatedly adapting unfamiliar small mechanisms to a desired force-response envelope. A conventional workflow characterizes the system, fits a physical or learned model, optimizes design parameters, and tests candidates. The proposed representation would replace neither physics nor verification; it would supply a reusable prior over **capability changes**.

There is no supplied industrial dataset, customer study, or measured engineering bottleneck. The claim that repeated characterization and inverse design consume enough resources to justify learning is therefore an **application assumption**. The local program must measure both the available search headroom and the number of future tasks needed to amortize training. A fast low-dimensional optimizer may eliminate the practical opportunity entirely.

### 2.2 Objects and notation

Let:

- \(s\) denote a mechanism, including hidden graph, physical parameters, and a specified initial state/history;
- \(h_s\) denote the actually observed characterization traces, including input waveforms, timestamps, masks, censoring indicators, and known experimental conditions;
- \(d=(k_a,c_a,b_a)\) denote an external attachment at a named port, with force \(b_a-k_a x_p-c_a v_p\);
- \(q=(F,\gamma,r)\) denote an operating trial: commanded force amplitude, actuation gain, and ramp duration;
- \(V_s(d,q)\) denote the normalized outcome vector defined in Section 5;
- \(\mathcal F_s(d)\) denote the subset of the declared operating-query domain passing all declared constraints;
- \(\Delta_s(d,q)=V_s(d,q)-V_s(0,q)\) denote the attachment-induced capability-change field;
- \(z\in\{1,\ldots,K\}\) denote a discrete motif and \(r_z\in\mathbb R^p\) its continuous residual coordinates. The residual symbol is distinguished from the ramp duration by its subscript.

The vocabulary depends on the interface, query domain, time horizon, observable outputs, and task constraints. Changing those can change which systems or edits may be grouped. A code ID without that context is not a physical semantic unit.

### 2.3 Primary task

From the same characterization data and declared design domain available to each method, choose an attachment that maximizes **audited robust force coverage**, with a bounded number of online trial trajectories. Every proposed attachment must respect the parameter domain and must not open the mechanism during installation or the specified shock guard.

The primary population is two explicitly held-out synthetic graph templates, equally weighted, with independently drawn parameter sets. An IID parameter holdout is secondary. This tests transfer to those templates, not a statistical population of all possible topologies.

The principal unit is a **base mechanism**, not an individual amplitude query, trajectory, neural initialization, or timing repetition. All outcomes from one mechanism are dependent.

### 2.4 Established, assumed, hypothesized, and speculative

| Category | Content | Consequence |
|---|---|---|
| Established prior art | Predictive states, behavioral interconnection, action/effect symbols, abstraction repair, functional inverse design, constrained Bayesian optimization | Reuse concepts and strong baselines; do not claim these as inventions. [R01–R13] |
| Mathematical deductions here | Internal-edit ambiguity; approximate similarity need not be transitive; finite margin error controls decisions only away from boundaries | Implement exact negative controls and boundary-sensitive evaluation. |
| Modeling assumptions | Resettable simulated copies; a known external port; passive attachment laws; declared parameter ranges; observable opening event | Explicitly narrow the claim; test violations separately. |
| Empirical hypotheses | Recurrent capability changes exist; they can be inferred from limited observations; a discrete motif improves the tradeoff beyond continuous alternatives | These are the primary research uncertainties. |
| Speculative extensions | Arbitrary geometry editing, scar operators, higher-order interaction alphabets, cross-domain transfer, complete reachable-set topology, physical deployment | Not prerequisites, not conclusions of the core PoC. |

### 2.5 Central hypotheses and discriminating observations

| ID | Hypothesis and why it is plausible | Main alternative explanation | Evidence that weakens or falsifies it locally |
|---|---|---|---|
| H0: observability | Characterization plus allowed online tests contain useful information about which attachment works. Predictive-state/system-identification traditions motivate this, but do not establish it for this generator. | Information is absent; the apparent skill comes from the average training mechanism. | A best fixed attachment matches a mechanism-specific oracle within numerical/statistical resolution; indistinguishable observations require different designs; posterior uncertainty remains irreducible. |
| H1: recurring effects | Attachment-induced capability changes share reusable structure across mechanisms. | Effects vary smoothly without meaningful reusable partitions; clusters reflect only attachment parameters. | No rate–distortion or transfer benefit versus PCA, continuous regressors, or attachment-only models; code usage collapses or tracks topology IDs rather than effects. |
| H2: witness refinement | Splits based on mistaken capability transfer use a fixed vocabulary budget better than reconstruction-only clustering. | The gain is merely boundary weighting, extra labels, or more clusters. | Equal-label, equal-size weighted continuous/tree and ordinary clustering controls remove the gain. |
| H3: discrete contribution | A hard motif plus a small residual offers useful generalization, sample efficiency, or computational reuse. | Continuous residuals, generic regularization, or the ensemble do all the work. | Soft-mixture, no-code, shuffled-code, or continuous end-to-end models match or improve the full tradeoff. |
| H4: compiler value | Generating a desired capability change before choosing an attachment improves realization efficiency. | A standard forward surrogate and direct optimization suffice. | With the same model and labels, direct parameter acquisition is as good or better; many generated motifs have no admissible realizations. |
| H5: local utility | Online savings survive training, fitting, decoding, and verification overhead. | Simulation is already too cheap and the design space too small. | Negative time savings or an implausibly large amortization horizon. |

Supporting literature is evidence that these mechanisms are worth comparing, not evidence that these hypotheses are true. The experiment may support H2 while rejecting H3, or H3 while finding no practical H5 advantage.

### 2.6 Strongest case for and against

The strongest favorable case is a family of mechanisms with qualitatively recurring operating-window changes, sparse characterization data, costly evaluation, and enough distribution shift that sharing effect motifs helps more than interpolating raw parameters. A finite vocabulary could pool data, expose invalid transfer cases, and make capability-level candidate generation efficient.

The strongest opposing case is more prosaic: the attachment domain is only three-dimensional, a physical fit or random forest already discovers the important structure, and hard quantization loses exactly the small differences that determine whether a force is feasible. Boundary discontinuities amplify small errors. A continuous model can represent every finite-code mapping, so there is no general expressivity theorem in favor of tokens. Any advantage must arise from inductive bias, regularization, amortization, interpretability useful to a decision, or a measured computational structure—not from greater theoretical representational capacity.

---

## 3. Research landscape and novelty assessment

### 3.1 What already exists

The relevant lineage is broader than tokenization. Predictive state representations describe systems through action-conditioned future observations. Behavioral systems theory organizes systems by admissible trajectories and their interconnections. Bisimulation aggregates behaviorally similar states with task-related error bounds under specified assumptions. None makes “what a system permits” an untouched mathematical substrate. [R01–R03]

DeepSym learns action-grounded discrete object, relation, and effect categories for planning. Skills-to-symbols work connects abstraction to action initiation and effects. Active automata learning already builds distinctions from experiments and counterexamples; symbolic-geometric repair already revises abstractions through counterexamples. Calling that process “witness refinement” does not make it new. [R04–R07]

Causal abstraction research explicitly addresses correspondence between interventions at different levels. Recent formulations include transformations of mechanisms, not only setting variables. This is the closest conceptual check on the proposed edit-aware semantics. [R08–R10]

Functional inverse design also exists without a discrete token model. RIGID uses random-forest structure to identify parameter regions satisfying functional-response targets. It is a particularly important simple competitor to any capability-first proposal. [R11]

Predictive compression also has a substantial causal-state/rate–distortion lineage. Quantization optimized for identification error and compression optimized for control objectives predate this proposal. Thus “preserve task-relevant distinctions rather than raw reconstruction” is not itself novel. [R21, R22, R31]

A 2026 causal-abstraction validation study separately challenges observational and functional agreement using interventional failure cases. It strengthens the need for adversarial twins and explicit intervention scope, but its access to low-level interventions and supplied abstraction maps differs from this experiment. A broad patent screen also found existing machine-learning-guided materials-design disclosures; it provides no basis for claiming the overall inverse-design loop is new. [R32, R33]

### 3.2 Closest-prior-art comparison matrix

The “evidence” column refers to what the cited work establishes in its setting, not an independently reproduced result here.

| Prior art | Problem and mechanism | Assumptions/evidence | Cost and implementation | Exact overlap and remaining difference |
|---|---|---|---|---|
| Predictive representations of state [R01] | Encode states using predictions of action-conditioned tests. | Controlled stochastic systems; representation-size result relative to a minimal POMDP. | Classical mathematical construction, not a ready-made attachment designer. | Anticipates Probeprints. Does not establish discrete edit-effect motifs or few-probe inverse-design gains. |
| Behavioral systems/interconnection [R02] | Model admissible trajectories and compose systems through shared variables. | Explicit behavior and interconnection semantics. | Analytical/control framework. | Anticipates possibility-based semantics and the external-port restriction. This project contributes no new interconnection theorem. |
| Bisimulation metrics [R03] | Task-relevant state aggregation and error metrics. | Finite MDPs; value-related guarantees. | Finite-state computations can grow with state count. | Anticipates task-relative equivalence; no automatic guarantee for unknown hybrid physical edits. |
| DeepSym [R04] | Binary bottleneck predicts action effects; tree rules support planning. | Simulated manipulation and visual puzzles; supplied action repertoire. | Public MIT code; small bottleneck is local, original ROS/CoppeliaSim pipeline is not required here. | Strong overlap with behavioral categories and symbolic planning. Main comparison must be an adapted action/effect bottleneck, not just a reconstruction autoencoder. |
| Skills to Symbols [R05] | Derive symbols suitable for planning from skill behavior. | Existing skills and their relevant initiation/effect structure. | Symbol-learning/planning framework; not the same experimental interface. | Anticipates capability-dependent abstractions; physical attachment synthesis is a different application, not automatically a methodological invention. |
| L* and AALpy [R06, R27] | Learn finite automata using queries and counterexamples. | Finite alphabet and specified oracle assumptions; modern library includes several learning algorithms. | Portable Python library; exact-equivalence oracle unavailable for an unknown continuous mechanism. | Anticipates distinguishing experiments and partition repair. Use as a finite-state diagnostic, not pretend the latch data meet the theorem. |
| Symbolic-geometric repair [R07] | Repair inadequate action abstractions from counterexamples. | Geometric sets and action predicates; symbolic formula repair and active counterexample sampling. | Published algorithm; use a local query-partition analogue with explicit adaptation. | Anticipates refinement and repair. Formula edits are not edits to a physical mechanism. |
| Causal abstractions [R08–R10] | Preserve interventions or mechanism transformations across abstraction levels. | Explicit low/high-level intervention maps and error notions. | Primarily formal frameworks in the sources inspected. | Anticipates edit-consistent semantics; novelty cannot be claimed for a commuting diagram alone. |
| RIGID [R11] | Invert functional targets through random-forest leaf regions. | Finite training examples; task-specific response functions. | MIT implementation; synthetic validation code available, some physical simulators absent from repository. | Anticipates capability-first inverse design; required tree/leaf-region baseline. |
| Task-optimal exploration [R12] | Gather information relevant to the downstream task. | Linear dynamical-system results, including LQR. | Experiment-design methods; different assumptions from hybrid latches. | Active probing is not token novelty; gains from query selection must be isolated. |
| Constrained BO [R13, R26] | Model objective/constraints and choose informative feasible candidates. | Surrogate uncertainty and acquisition assumptions. | CPU double-precision GP is practical for this small design domain. | Strong contemporary workflow baseline; may make token machinery unnecessary. |
| Compositional abstraction refinement [R14] | Refine overapproximations for control synthesis. | Known dynamics/reachability and compatibility assumptions. | State-space cost can be substantial; small examples are local. | Anticipates compositional possibility abstractions, not observed-data attachment design. |
| S³, August 2026 [R15] | Optimize discrete state partitions with a smooth simulation surrogate. | Known deterministic twice-differentiable dynamics and sound reachability construction. | Public MIT CPU/JAX implementation with Spiral, Mountain Car, Unicycle. | Directly relevant current abstraction baseline on smooth diagnostics. Its soundness does not carry over to an empirical token predictor or a discontinuous latch model. |
| Task/predictive compression [R21, R22, R31] | Allocate finite representation capacity to predictive, identification or control loss. | Causal-state, SISO FIR, or linear-control assumptions depending on source. | Analytical/optimization frameworks; no need to run an inapplicable control benchmark. | Anticipates task-aware code distortion; does not establish this conditional attachment-effect vocabulary. |
| Causal-abstraction validation [R32] | Test proposed abstractions with intervention-based error and controlled invalid examples. | Supplied models/maps and interventions into the simulated system. | Public benchmark described; not reproduced here. | Anticipates testing beyond observational similarity. Core WCCT must not claim a causally valid explanation from response compression alone. |

### 3.3 Evaluation of all five brainstorming directions

**Probeprints:** keep as the observation layer, but do not assume discretizing observations is useful. Predictive states, system identification, and task-conditioned effect learning already cover much of the concept. The core encoder may remain continuous while the proposed effect representation is discrete. This is an explicit architectural refinement. [R01, R04]

**Possibility boundaries:** retain the finite-horizon, task-defined feasible operating envelope. Do not claim that a force-window experiment learns global configuration-space topology. A later geometric feasibility study would need a separate planner, clearance metric, contact model, and benchmark; it is not necessary to test the current representation.

**Scars:** defer. History-aware constitutive models already infer path-dependent response from observable histories. Their treatment of initialization and discretization is relevant to any later plasticity/damage extension. The current irreversibly releasing latch is a controlled hidden mode, not a general material-memory model. [R16]

**Interaction excess:** retain the factorial difference as a diagnostic for combinations of attachments, not the primary vocabulary. Nonlinear input–output expansions are established; Volterra results have assumptions such as fading memory that do not automatically cover persistent latch changes. [R17]

**Holonomy:** defer to a separate domain. Contemporary geometric gait optimization already handles cycles, transitions, and kinematic/dynamic effects, including simulation and hardware examples. Its natural group/geometric structure is attractive, but changing to locomotion now would abandon the strongest local identifiability and design tests. [R18]

Do not combine these into one universal alphabet in the PoC.

### 3.4 Apparently conflicting evidence about discretization

The 2026 *Compression Gap* preprint reports limits of fixed action tokenization in particular VLA configurations. Its comparisons vary with the encoder and codebook setting; they are not a theorem against discrete representations. The September 2026 M²Tok preprint reports benefits from multi-head/multi-codebook action tokenization on different benchmarks and configurations. These findings concern different systems, capacity allocations, and comparisons. They motivate capacity/residual sweeps and matched controls here, not a vote for or against all tokens. Published VLA success rates must not be copied into this project as comparable baseline numbers. [R19, R20]

### 3.5 Precise contribution ledger

| Proposed item | Classification now | Discriminating test | Evidence that would invalidate a novelty claim |
|---|---|---|---|
| External-port, intervention-consistent formulation | Necessary modeling discipline; established conceptual foundation | Exact internal-vs-external edit counterexample | Existing behavioral and causal abstraction theory already covers the principle. |
| Vocabulary of attachment-induced capability-change fields | Plausible application/representation refinement | Compare effect tokens with raw-response tokens and continuous effect models at equal information | A prior method already learns the same conditional functional-change vocabulary and uses it for physical inverse design. |
| Witness-driven partition loss based on failed capability transfer | Plausibly differentiated algorithmic combination, not established novelty | Equal-codebook, equal-label MSE clustering, weighted trees, and continuous witness-trained controls | Existing abstraction-repair/task-oriented quantization method is equivalent after renaming variables. |
| Capability-first attachment realizer | Engineering integration unless it yields a new search mechanism/effect | Same trained predictor, token-first versus direct acquisition | RIGID-style inversion or ordinary surrogate optimization achieves the same procedure/tradeoff. |
| Reproducible cross-template comparison | Potential empirical contribution regardless of algorithmic novelty | Protected evaluation and independent replication | Dataset construction leaks template identity or manufactures a discretization advantage. |

**Answer to the novelty question:** much of the original proposal is known. The precise combination above remains an unresolved novelty question, and no bounded search proves absence of prior art. A useful negative result or an empirical characterization of when discretization helps would still be a legitimate contribution. No patentability or freedom-to-operate opinion is offered.

### 3.6 Reuse rather than rebuild

Reuse NumPy/SciPy numerical primitives, PyTorch dense/convolutional operations, scikit-learn forests and decompositions, and a current compatible BoTorch/GPyTorch stack. Reuse published DeepSym bottleneck mechanics in an adapted observation/effect model. Reuse RIGID leaf-region logic after compatibility tests. Keep S³ and AALpy in optional isolated environments for their faithful diagnostic tasks. Do not import ROS, a large VLA, or a GPU-only simulator merely because a prior paper used one. [R24–R30]

---

## 4. Checks actually performed during preparation

### 4.1 Execution environment and status

The calculations below were run in a Linux x86-64 environment, **not on an M3 Max**:

```text
OS/runtime: Linux-6.18.44-x86_64-with-glibc2.41
Python: 3.13.5
NumPy: 2.3.5
SciPy: 1.17.0
Random generator for latch pilot: numpy.random.default_rng(1401026)
```

No neural model, forest, Bayesian optimization loop, tokenizer, or Apple acceleration path was trained or benchmarked. Source inspection is not a successful reproduction of the cited methods.

### 4.2 Exact internal-edit counterexample

Mechanism A is one linear spring with stiffness 1. Mechanism B is two series springs with stiffnesses 2 and 2. Both have equivalent boundary stiffness 1, so every static boundary force–displacement observation is identical.

Now interpret the internal edit as “increase the designated spring’s stiffness by 1.” A has new stiffness 2. B has new equivalent stiffness \(3\times2/(3+2)=1.2\). At force 1.5 their displacements are respectively 0.75 and 1.25. Thus the binary capability “reach displacement at least 1” differs after the edit despite identical original boundary curves.

The distinction is exact algebra, not floating-point evidence. A representation of the original boundary response alone cannot select the correct answer for both systems. With equal prior probabilities and no distinguishing observation, binary classification accuracy cannot exceed one half on this deliberately ambiguous pair.

Adding the same external spring, stiffness 0.7 and rest position 0.4, instead gives both systems

\[
x=(F+0.7\cdot0.4)/(1+0.7).
\]

The numerical comparison over 101 forces from 0 to 3 had maximum difference zero. This does not establish equivalence for arbitrary dynamic systems; it illustrates why a specified external interface avoids this particular ambiguity.

### 4.3 Approximate similarity is not an equivalence relation

For responses 0, 0.09, and 0.18 with tolerance 0.1, the first is close to the second and the second to the third, but the first is not close to the third. Therefore the implementation must use an actual partition, report within-cell distortions, and not apply transitive closure to a nearest-neighbor similarity graph while calling its components guaranteed behavioral-equivalence classes.

### 4.4 Exploratory latch simulation

The pilot used

\[
\dot x=v,\qquad m\dot v=u(t)+k_a\ell_a-(k+k_a+k_L L)x-(c+c_a)v,
\]

where \(L\) starts at 1 and switches permanently to 0 at the first upward crossing of \(x=g\). Position and velocity remain continuous; deleting the latch spring removes its stored energy. This is an explicitly **dissipative idealized release model**, not a validated model of a named physical latch.

For 12 systems, independent uniform draws were made in this exact order, drawing the full vector of 12 values for each parameter before moving to the next parameter:

```text
m: [0.8, 1.2], k: [0.6, 1.4], c: [0.1, 0.45],
k_L: [3, 7], g: [0.18, 0.32]
```

The design grid was \(k_a,c_a\in\{0,0.5,1,2\}\), \(\ell_a\in\{0,0.5,1\}\). Its 48 parameter triples contain only 40 physically distinct attachments because rest position is irrelevant at zero spring stiffness. The final benchmark must canonicalize designs by \((k_a,c_a,b_a=k_a\ell_a)\).

Each operating trial started at rest, installed the attachment under zero external force for 2 time units, then ramped force to amplitude \(A\) in 0.4 units and held it until total time 6. Amplitudes were 41 equally spaced values in [0.2,3]. A trial passed if installation did not release the latch, final displacement was at least 1, maximum displacement was at most 1.5, and maximum absolute velocity was at most 1.5. A separate installation-plus-shock trial used a triangular force pulse of height 0.25 and width 0.2 after installation and required the latch to remain closed.

| Quantity | Measured exploratory value |
|---|---:|
| Operating trajectories | 23,616 |
| Additional shock trajectories | 576 |
| Mean unedited grid coverage | 0.0589431 |
| Mean best-per-system grid coverage | 0.3963415 |
| Best fixed attachment, chosen on these same 12 systems | \((k_a,c_a,\ell_a)=(2,2,0.5)\) |
| Mean coverage using that fixed attachment | 0.3556911 |
| Mean best-per-system excess over fixed attachment | 0.0406504 |

The first vectorized RK4 run at step 0.002 took about **2.97 seconds for the operating batch only**. The saved rerun, which also calculated the fixed-design diagnostic, recorded **3.34 seconds** for that batch. No timing repetition study was performed. The reported timing excludes shock runs, reference solves, training, optimization, and I/O; it is not an end-to-end measurement or a Mac prediction.

All per-system best designs used the largest allowed added damping. This is evidence to require a **best fixed attachment and a simple high-damping heuristic**, not evidence of intelligent inverse design. The fixed-design selection was retrospective and in-sample; proper validation selection remains to be done.

### 4.5 Numerical cross-check and limitations

Eighteen randomly selected operating trajectories were checked with independently implemented event-based SciPy DOP853, relative tolerance \(10^{-10}\), absolute tolerance \(10^{-12}\), maximum step 0.01, and dense-output samples spaced 0.0005 for extrema. The historical check used unnormalized margins `(final_x−1, 1.5−max_x, 1.5−max_abs_v)`; the final benchmark later normalizes the velocity margin as specified in Section 5. The largest absolute error over these three historical outcome margins was approximately 0.001656 at RK4 step 0.002 and 0.000307 at step 0.0005. No pass/fail disagreements occurred in those 18 cases.

This is a small numerical smoke test, **not a certification of extrema, event accuracy, or near-boundary labels**. The prototype switched the latch after a full RK4 step, producing delayed events. The final solver must localize events and explicitly test near-threshold trajectories. SciPy documents that event detection depends on sign changes within steps; multiple crossings can be missed without suitable step control. [R25]

### 4.6 Reproduction recipe for these checks

Reproduce the spring algebra directly. For the pilot, use the equations, seed, ordered parameter draws, Cartesian design grid, amplitudes, and trial definitions above. Batch arrays in system-major, design-major, amplitude-major order. Implement ordinary classical RK4 with the latch state fixed within each step and update it after each step for this historical smoke-test reproduction only. Track running extrema and installation release. Resample 18 row indices without replacement using the same generator after parameter generation. Compare those rows with the event-restarted reference integration. The values in Section 4.4 should match within rounding except wall time; exact reproduction of the weak event method is not approval to use it for confirmatory labels.

---

## 5. Synthetic benchmark and physical semantics

### 5.1 Scope and dimension conventions

The benchmark uses dimensionless quantities. An engineering realization would require explicit choices of length, time, and mass scales; force scales as \(M_0L_0/T_0^2\), stiffness as \(M_0/T_0^2\), and damping as \(M_0/T_0\). Do not present the dimensionless force interval as newtons or the simulated material law as a validated physical specimen.

Each trial is a fresh reset copy. Installation and subsequent operation are one trajectory with shared state; the operating phase must **not** be reset after installation. A separate shock trial gets its own reset copy. Copy/reset availability is a material experimental assumption and would have costs or limitations in a physical experiment.

### 5.2 Mechanism generator

Use a connected grounded spring/damper network with one to three moving nodes. Node 0 is fixed ground. For an oriented edge from \(i\) to \(j\), extension is \(e=x_i-x_j\), taking ground displacement as zero. Apply equal and opposite forces from \(k_e e+c_e\dot e\). Orientation must not affect physical results.

For moving-node incidence matrix \(B\), positive diagonal mass matrix \(M\), and port vector \(e_p\), the equations are

\[
\dot x=v,
\]
\[
M\dot v=-B\{k\odot B^Tx+c\odot B^Tv\}
+e_p\{u(t)+b_a-k_a x_p-c_a v_p-k_L Lx_p\}.
\]

At the first upward crossing \(x_p=g\), set \(L=0\) permanently, with no impulse or velocity reset. The removed energy is \(k_Lg^2/2\) at an exact crossing. This defines the model's energy ledger. It must not be interpreted as a conservative latch.

| Template | Edges (ground is 0) | Port | Use |
|---|---|---:|---|
| G0 | (0,1) | 1 | Training and analytical/headroom diagnostics |
| G1 | (0,1), (1,2) | 2 | Training |
| G2 | (0,1), (0,2), (1,2) | 2 | Training |
| G3 | (0,1), (1,2), (2,3) | 3 | Development transfer population only |
| G4 | (0,1), (0,2), (1,3), (2,3) | 3 | Protected transfer population |
| G5 | (0,1), (1,2), (2,3), (1,3) | 3 | Protected transfer population |

Draw every moving mass uniformly in [0.8,1.2], every edge stiffness in [0.6,1.4], every edge damping in [0.1,0.45], \(k_L\) in [3,7], and \(g\) in [0.18,0.32]. Use independent seeded draws and save exact float64 values. Within a template all those parameters are continuous; the generator does not draw a discrete material class for the tokenizer to recover.

Primary mechanisms are linear between latch events. A gated stress suite adds passive cubic edge forces \(\alpha_e e^3\), \(\alpha_e\in[0,0.8]\), with energy \(\alpha_e e^4/4\). It is a declared model-law shift, not a replacement benchmark selected after unfavorable results. Coulomb friction, impacts, plastic damage, and multiple latches are separate extensions requiring new correctness tests; they are not silently approximated by this generator.

### 5.3 External design domain and composition

Use \(k_a,c_a\in[0,2]\), rest position \(\ell_a\in[0,1]\), and \(b_a=k_a\ell_a\). Equivalently, require \(0\le b_a\le k_a\). Store \((k_a,c_a,b_a)\) canonically. When \(k_a=0\), \(b_a=0\); set the display rest position to zero. Attachment IDs are hashes of canonical numerical values and units, not row numbers.

Initialize a shared finite candidate set with a nested Sobol design in the box of \((k_a,c_a,\ell_a)\), plus the 40 distinct diagnostic-grid attachments and the no-edit design. A starting Sobol count of 128 is an engineering starting point. On development systems, double it until reference best-candidate coverage changes by less than the measured evaluation resolution, or until the measured cost requires a documented finite-menu scope. Use the **same frozen candidate set for every primary method**. Continuous-domain optimization is a separate secondary track; do not compare its optimum with a finite-menu method without identifying the domain difference.

Two simultaneous attachments on the same port combine exactly as

\[
(k_1,c_1,b_1)\oplus(k_2,c_2,b_2)=(k_1+k_2,c_1+c_2,b_1+b_2),
\]

when the sum remains in the declared domain. This is a physical force-law identity. It does **not** imply \(\Delta(d_1\oplus d_2)=\Delta(d_1)+\Delta(d_2)\). Installation order can also matter if a latch releases between installations. The composition stress test uses simultaneous installation from the same reset state; sequential installation is a different history-dependent task.

Report the dimensionless edit-cost proxy

\[
C(d)=k_a/2+c_a/2+\tfrac12 k_a\ell_a^2.
\]

This is a disclosed proxy for added stiffness, damping, and stored preload energy—not money, manufacturing effort, or a validated engineering cost. The primary design domain is the bounded attachment set; cost is a secondary Pareto measure and deterministic tie-breaker, not an arbitrary weighted addition to coverage.

### 5.4 Characterization observations

Each mechanism supplies four standardized trials before online design search:

1. Three low-amplitude trials of duration 2: \(u_j(t)=0.05\sin^2(\pi t/2)\sin(2\pi f_jt)\), with \(f_j\in\{0.5,1,2\}\), no attachment, initial rest.
2. One release-capable trial of duration 6: \(u(t)=3\operatorname{clip}((t-2)/0.4,0,1)\), no attachment, initial rest.

These are **numerical characterization programs**, not a proven safe physical protocol. Record any release and constraint exceedance. The three small probes form the safe-probe-only diagnostic arm; do not filter mechanisms using hidden parameters to guarantee those observations are informative. The release-capable trial explicitly relaxes the original few-safe-probes ambition. A later physical study would need a validated fixture, abort policy, and treatment of censored traces before using such a program.

Observe applied input, port displacement, port velocity, and time. Save full traces for audit; encode 128 evenly spaced samples per trial, with exact duration, waveform ID, and missing-data mask. Test 64/256 samples on development data for aliasing. Velocity is a directly supplied simulated sensor in the core experiment; a derivative-estimation/noisy-position-only arm is a separate, disclosed sensor change.

The noiseless primary characterization isolates the representation. Add Gaussian sensor noise at predeclared standard deviations relative to training-channel scales for robustness; keep the same realization for paired methods on a mechanism. Do not claim the noiseless result establishes noisy real-world observability.

### 5.5 Operating programs and outcomes

Every operating trial has a 2-unit zero-input installation prefix and a 4-unit operation phase. After installation,

\[
u(t)=\gamma F\operatorname{clip}((t-2)/r,0,1),\qquad t\in[2,6].
\]

Use \(F\in[0.2,3]\), gains \(\gamma\in\{0.95,1,1.05\}\), and ramp durations \(r\in\{0.36,0.4,0.44\}\). These nine gain/ramp combinations are a **finite robustness audit**, not all disturbances in a continuous uncertainty set.

The continuous outcome components are

\[
V_1=x_p(6)-1,\quad V_2=1.5-\max_{0\le t\le6}x_p(t),\quad
V_3=(1.5-\max_{0\le t\le6}|v_p(t)|)/1.5.
\]

The first two use length scale 1; the third uses velocity scale 1.5. Additional components encode installation and shock pass/fail as +1 or −1. These last two are **signed indicators, not physical distance-to-failure margins**. An installation failure is any latch release during the prefix, including exactly at its endpoint.

The shock guard uses a separate 4-unit trajectory with the same installation prefix, followed by

\[
u_{shock}(t)=0.25\gamma\max(0,1-|t-2.1|/0.1)
\]

and requires no latch release throughout. Check all three gains. The shock condition is independent of the operating force and ramp duration and must be computed/stored once per mechanism/design/gain, not treated as hundreds of independent labels. Installation likewise needs only one independent computation per mechanism/design.

Operational feasibility at a query requires all three continuous constraints and both guard conditions. A numerical gray zone around a continuous margin is handled as unresolved until refined; it is not assigned favorable feasibility by rounding.

### 5.6 Labels, query grids, and the primary objective

Start teacher training with nine equally spaced force amplitudes and all nine gain/ramp combinations: 81 operation queries per mechanism/design, plus one installation and three shock trials. This costs **85 trajectories per complete signature** under the simple non-sharing accounting convention. Operation trajectories already contain installation; the separately charged installation trial is retained for a clean guard record. It can later be eliminated for all methods if the harness proves equivalent accounting and the protocol is amended before confirmation.

A standard online design-evaluation bundle uses 17 force amplitudes, all three gains, and nominal ramp 0.4: 51 operating trajectories plus the four guard trials, hence **55 trajectories**. It returns the complete observed margin/guard record, not just a favorable summary. Cache only exact repeated requests; report attempted and unique calls separately. All methods use the same cache rule and receive the same type of information.

For final blinded evaluation, begin with 65 amplitudes and all nine scenarios, plus guards: **589 trajectories per returned design**. Increase force-grid resolution and refine boundary intervals on development data until discretization uncertainty is below the planned effect-estimation resolution. Freeze the resulting audit rule before protected evaluation. The amplitude grid is a quadrature convention, not a proof of a continuous interval.

For the final chosen amplitude set \(A\), define

\[
J_s(d)=\frac1{|A|}\sum_{F\in A}
\mathbf1\left\{\text{all nine operating scenarios pass at }F,
\text{installation passes, and all shock gains pass}\right\}.
\]

**Primary metric:** mean \(J_s(d_{method})\), equal-weighted over G4 and G5, after a frozen online trajectory budget. Higher is better. Return zero if a guard fails or the method explicitly abstains from proposing a design. A crashed method is reported separately and handled by the predeclared operational failure convention in Section 8—not deleted from the population.

Also report covered force width \(2.8J\), the longest connected run of passing amplitudes, actual boundary locations with refinement uncertainty, and nominal versus nine-scenario coverage. Do not assume the successful amplitudes form one interval or perform monotone bisection without checking that assumption.

### 5.7 Dataset sizes and leakage prevention

The first learning pilot starts with 128 training mechanisms distributed nearly equally over G0–G2, 12 nonzero training attachments per mechanism sampled from the shared pool, and the no-edit signature. That is approximately \(128\times13\times85=141{,}440\) labeled trajectories plus 512 characterization trials. This is a resource-calibration starting point, not a scientifically justified final sample size.

Keep all variants, traces, attachments, and noise replicates of a base mechanism in one partition. Use a separate development population including G3. Keep protected G4/G5 parameter seeds and outcomes inaccessible to training and tuning. Allocate the confirmatory mechanism count from pilot variance and measured hardware throughput as specified in Section 8. Increase training systems, attachments per system, or query resolution only when the pilot identifies which dimension limits performance.

No topology ID, hidden stiffness, latch threshold, graph adjacency, full capability signature, teacher code, or final audit response may enter the inference input. Training signatures are privileged **targets shared with all supervised competitors**, not privileged test inputs. Model-informed system identification may search the complete publicly declared graph family; it must not receive the identity or parameter values of the test mechanism.

---
## 6. Proposed mechanism and implementation specification

### 6.1 What the abstraction must preserve

For an exact boundary behavior set \(\mathcal B_s\), interconnecting the same external attachment means imposing the same port relation and projecting onto the measured variables. If \(\mathcal B_{s_1}=\mathcal B_{s_2}\), those set operations produce identical observed interconnected behavior, provided initial conditions and interface semantics match. Unique response prediction additionally requires well-posedness. This is a direct application of behavioral interconnection reasoning, not a new theorem. [R02]

A finite collection of probes does not establish equality of these sets. In the latch generator, systems differing only in \(g\) are indistinguishable under every input that stays below both thresholds. That remains true even with perfect sensors. The implementation must separate **irreducible observational ambiguity** from model error and compression error.

The learned codebook is an empirical partition of observed training effect signatures. It is not a sound abstraction of an infinite state space, and its stored counterexamples are **witness records, not correctness certificates**.

### 6.2 A useful elementary error bound

Let \(m_s(d,F)\) be the minimum normalized outcome over all constraints and audited scenarios. If, on a finite evaluation menu, every component prediction has error at most \(\epsilon\), then minima also have error at most \(\epsilon\). Feasibility therefore agrees whenever \(|m_s(d,F)|>\epsilon\). Consequently,

\[
|\widehat J_s(d)-J_s(d)|\le
\frac{1}{|A|}\#\{F\in A:|m_s(d,F)|\le\epsilon\}.
\]

If the maximum coverage error over all candidate designs is bounded by \(\eta\), exact maximization of predicted coverage has finite-menu regret at most \(2\eta\). With acquisition/search approximation error \(\zeta\), the corresponding bound becomes \(2\eta+\zeta\).

These are elementary deductions, not a claim of methodological novelty. The difficulty is obtaining a uniform bound from limited observations; a validation residual quantile does not establish it under topology shift. Use the inequalities to motivate measuring boundary mass and error, not to advertise certified safety. A small mean trajectory error can coexist with poor design decisions when many queries lie near a boundary.

### 6.3 Signature representation

Flatten a complete training signature into 247 numbers:

```text
operation margins: [9 force amplitudes, 3 gains, 3 ramps, 3 margins] = 243
installation signed indicator: 1
shock signed indicators: [3 gains] = 3
```

Store the axis values and flattening order explicitly. The no-edit signature is \(B_i\in\mathbb R^{247}\); the edited signature is \(Y_i\); the effect vector is \(E_i=Y_i-B_i\), with row \(i\) identifying a mechanism/design pair. Installation and shock labels occur once in the loss, not once per broadcast operating query.

For intermediate amplitudes, interpolate prototype tables piecewise linearly along the force axis. For intermediate gain/ramp values in a stress test, use multilinear interpolation within the declared grid. Never extrapolate outside its hull without returning an out-of-domain flag. This interpolation is an engineering approximation; it must be compared with denser teacher grids and a continuous-query baseline, particularly for narrow feasibility islands.

### 6.4 Shared characterization encoder and base predictor

Implement a small trace encoder, not a language model. A practical starting architecture is two shared 1D convolution layers, channels 32 and 64, kernel width 5, stride 2, SiLU activation, followed by masked temporal pooling. Concatenate each probe's pooled features with waveform ID, duration, and censoring flags; concatenate the four fixed probe embeddings and map through two width-128 dense layers. The resulting \(h\)-embedding is continuous.

A base head \(b(h)\) predicts the 247-number no-edit signature. Train it only from no-edit targets. It must not see the proposed attachment. The encoder/base predictor can be shared across comparisons to isolate the effect representation; also allow each strong baseline to tune its encoder end to end in the practical comparison. Record which assets are shared and their training cost.

Normalize observation channels using training mechanisms only. Do not standardize each trace to unit variance: that would erase stiffness and damping information. Preserve force/displacement magnitude, elapsed time, masks, and probe identity. Normalize losses by physically defined margin scales; any additional robust scaling is fitted on training data and shared across methods.

### 6.5 Teacher vocabulary: witness-refined divisive quantization

The initial teacher groups all nonzero edit-effect signatures in one leaf. Reserve an exact identity code for the no-edit design; its prototype and residual are zero. Repeatedly split leaves using actual effect coordinates that expose incorrect capability transfer.

For leaf prototype \(c\) and training row \(i\), reconstruct a counterfactual edited signature \(\widetilde Y_i=B_i+c\). Define

\[
D_i(c)=L_{margin}(Y_i,\widetilde Y_i)
+\lambda_{FP}L_{false\ feasible}(Y_i,\widetilde Y_i)
+\lambda_{FN}L_{missed\ feasible}(Y_i,\widetilde Y_i).
\]

Here \(L_{margin}\) is mean Huber loss over operation margins plus separately normalized guard loss. The two decision terms are fractions of force amplitudes for which the nine-scenario capability label is falsely positive or falsely negative. They use true no-edit signatures **only during teacher construction**. Numerical gray-zone labels are excluded from sign-loss fitting until refined, with the exclusion count recorded.

For every eligible leaf, consider coordinate-threshold splits of \(E_i\). For each coordinate, use midpoints between sorted values when the leaf is small; for larger leaves use a saved deterministic quantile threshold set. Compute each child's arithmetic-mean prototype and the reduction in summed \(D_i\). Choose the split with greatest reduction, breaking ties by leaf ID, coordinate ID, then threshold. Each child must contain data from at least two distinct base mechanisms; validate stricter support requirements if leaves memorize individual systems.

Save, for each split, the two witness rows with the largest corrected capability disagreement, the query/constraint causing it, the threshold, and before/after distortions. An ordinary effect difference without a capability error may still justify an approximation split, but mark it as such rather than inventing a counterexample.

The starting capacity sweep is total code counts 8, 16, 32, and 64 including identity; extend only if validation distortion and student predictability justify it. The final limit follows data support and measured cost. Choose the smallest capacity within one validation standard error of the best downstream score, unless a predeclared Pareto choice favors another point. This is a complexity-selection rule, not a universal vocabulary size.

Use development data to choose \(\lambda_{FP},\lambda_{FN}\), including zero as an essential reconstruction-only control. The same boundary/decision weighting must be available to the continuous and tree baselines. Do not obtain a token advantage by giving only this method the task's actual loss.

### 6.6 Bounded continuous residual

After a partition is fixed, fit a global rank-\(p\) PCA basis \(U\in\mathbb R^{247\times p}\) to training residuals \(E_i-C_{z_i}\) in the physically normalized signature coordinates, without additional whitening. Handle any roundoff-level residual mean as specified in Section 12.3. Use \(p\in\{0,2,4,8\}\) initially. Teacher residual coordinates are

\[
r_i=U^T(E_i-C_{z_i}).
\]

Store each code's maximum observed training residual norm \(\rho_z\) and its support count. At inference, clip predicted residuals to this ball, return an out-of-support flag if clipping was needed, and report sensitivity to the bound. Do not interpret the training maximum as a probabilistic coverage guarantee. A tiny numerical floor handles the exact-zero case without division by zero.

The decoded effect is

\[
\widehat\Delta(h,d)=C_{\widehat z(h,d)}+U\widehat r(h,d),
\qquad
\widehat Y(h,d)=b(h)+\widehat\Delta(h,d).
\]

The core teacher/prototype decoder has **no raw-observation or attachment bypass** around \((z,r_z)\). The base head cannot see \(d\). Enforce the identity \(\widehat\Delta(h,0)=0\) exactly. These constraints make the edit representation testable; they do not constrain the strong unrestricted continuous comparator.

### 6.7 Observable student

Train an MLP from the characterization embedding and canonical attachment parameters to code logits and residual coordinates. Begin with two width-128 layers. Use cross-entropy for teacher code assignment, Huber residual loss, and the reconstructed edited-signature loss above. Normalize objective terms by their training scale and select their weights on development data, exposing the same downstream loss to matched competitors.

Teacher labels are derived from simulator outcomes on training mechanisms. At test time, the student sees only the four characterization traces and a candidate attachment. It never receives the target edited signature or the oracle-assigned code.

The primary representation uses the maximum-probability hard code. The categorical probability vector is retained for uncertainty diagnostics and out-of-support handling, not silently treated as a free high-capacity continuous representation. An ensemble of independently trained students supplies variation in predictions; use development calibration to add empirical residual bands. The first calibration pilot can use three members, but the final ensemble/replication count follows measured variability and cost.

An ambiguous safe-probe twin should produce uncertainty or abstention; the complete pipeline, including the base-signature head, must be checked because code uncertainty alone may miss a shared base-prediction error. It must not be “fixed” by leaking the latent threshold, adding a hidden feature, or selecting only correctly guessed twins. If categorical confidence is unreliable under shift, label it unreliable and use conservative validation bands or an explicit unknown state; no calibration method here warrants a topology-shift safety guarantee.

### 6.8 Token data structure and invariants

A token record has this schema:

```text
TokenRecord
  code_id: integer, stable within codebook version
  codebook_sha256: string
  interface_id: string             # named port, coordinate/sign and units
  domain_sha256: string            # query grid, horizon, constraints, generator law
  prototype_delta: float32[247]
  residual_basis_sha256: string
  residual_radius: float
  support_mechanism_ids: list[str]  # training only; not inference features
  split_parent: integer | null
  witnesses: list[WitnessRecord]
  empirical_distortion: dict       # margin error, FP/FN, boundary slices

WitnessRecord
  source_rows: tuple[str, str]
  query_and_constraint: dict
  observed_values: tuple[float, float]
  split_coordinate: integer
  split_threshold: float
  reason: "false_feasible" | "missed_feasible" | "approximation"
```

Invariants: codebook IDs are versioned; codes are never reinterpreted in place; query-axis order is immutable; residual coordinates cannot be used with another basis; training IDs cannot become model inputs; no identity edit yields a nonzero effect; incompatible interfaces or horizons produce an error/unknown result, not a guessed composition.

### 6.9 Minimal training pseudocode

The functions below describe **project code to implement**, not an existing package API.

```python
def fit_wcct(train, dev, capacity, residual_rank):
    assert disjoint_base_mechanisms(train, dev)
    scaler = fit_observation_scaler(train.probes)
    encoder, base = fit_no_edit_predictor(train, dev, scaler)

    base_targets = train.no_edit_signatures_for_each_edit_row()
    edited_targets = train.edited_signatures
    delta = edited_targets - base_targets
    tree, codes, prototypes, witnesses = divisive_partition(
        delta=delta,
        base_targets=base_targets,
        edited_targets=edited_targets,
        mechanism_ids=train.mechanism_ids,
        capacity=capacity,
        task_distortion=capability_transfer_loss,
        identity_code=True,
    )
    basis = fit_training_residual_pca(delta - prototypes[codes], residual_rank)
    residual_targets = (delta - prototypes[codes]) @ basis
    radii = per_code_max_norm(residual_targets, codes)
    student = train_observable_student(
        probes=train.probes, designs=train.canonical_designs,
        code_targets=codes, residual_targets=residual_targets,
        prototypes=prototypes, basis=basis, encoder=encoder, base=base,
        validation=dev,
    )
    return freeze_versioned_model(student, prototypes, basis, radii, witnesses)
```

Store teacher distortion and student inference error separately. A useful teacher vocabulary that cannot be inferred from allowed observations is not a successful system.

### 6.10 Common online adaptation and primary search

The representation-only design comparison uses a common acquisition/search adapter. Each predictor supplies a prior estimate of the online bundle score from its predicted signatures. Fit a per-mechanism GP to residuals between observed bundle scores and that prior, over the three canonical attachment coordinates. Use a CPU float64 Matérn-5/2 kernel with separate length scales and a fitted or validated nugget. Enumerate acquisition values over the shared finite candidate pool rather than requiring a mixed-domain optimizer.

Start the online history with the no-edit design and the globally fixed attachment selected on development data. These evaluations cost two bundles, unless they are canonically identical. Then use expected-improvement or a validated log-EI implementation to select unused candidates. The zero-prior GP is the no-pretraining BO baseline. Evaluate any richer constrained-GP variant as a strong workflow comparator; the common adapter is for attribution, not a restriction imposed on every baseline. [R13, R26]

The pretrained codebook is frozen during protected search. Online observations update the common adapter, not global token definitions. Full training-time vocabulary refinement on protected instances would confound transfer with transductive learning and require a separate protocol.

At budget exhaustion, a method may select a tested or untested candidate, but this rule must be identical for all methods. The default is the best predicted admissible candidate, with its tested/untested status reported. The final independent audit is not fed back into search. Returning an untested candidate is not claimed to be a physically verified recommendation until that audit completes.

### 6.11 Capability-first generator/realizer: a separate test

To preserve the generative intent, implement this second proposal mode after the shared-acquisition comparison works:

1. For each supported motif, combine its prototype with \(b(h)\) and rank the resulting desired envelope by robust predicted coverage; retain uncertainty/support flags.
2. Form each motif's candidate preimage \(D_z=\{d:\arg\max P(z\mid h,d)=z\}\). Empty sets mean the desired motif currently has no proposed physical realization; count them.
3. Within a nonempty preimage, select the candidate maximizing the same calibrated acquisition used in direct search, with the same residual decoder and deterministic cost tie-breaker.
4. Test it, update the common online adapter, and recompute rankings. Keep an exploration option selected on development data so a mistaken motif ranking cannot permanently exclude all alternatives.

Compare with direct acquisition over the entire pool using the **identical predictor**. Also compare with a continuous capability-target realizer and the RIGID-style leaf-region method. This finite-menu generator proposes behaviors and attachment realizations; it does not create arbitrary new geometry. A failure to improve over direct search narrows the result to a representation finding and rejects the additional compiler-search claim locally.

### 6.12 Information and bit-budget accounting

Do not call a system “a six-bit physical model” because its motif has 64 values. Its information includes characterization traces/embedding, the base prediction, residual coordinates, the codebook, confidence vector, ensemble, domain context, and any online history.

For a strict edit-code compression experiment, serialize only \(z\) and quantized residuals and reconstruct with a fixed decoder and the same shared base information. Count \(\lceil\log_2K\rceil+p b_r\) transmitted bits per edit, plus amortized codebook/basis storage and any variable-length framing. Give the continuous comparator the same actual serialized budget. Evaluate several residual bit depths and report saturation. This is a separate rate–distortion experiment; the operational float32 pipeline is judged by measured memory, time, and performance, not an artificial bit claim.

---

## 7. Baselines and fair comparison rules

### 7.1 Required locally executable baselines

| ID | Method and implementation | Main purpose | Required reproduction/correctness checks |
|---|---|---|---|
| B0 | No edit; development-selected best fixed attachment; fixed high-damping heuristic with development-selected spring/preload | Expose trivial headroom and prior-only performance | Fixed design selected without protected outcomes; canonical duplicates removed; direct solver verifies it. |
| B1 | Scrambled Sobol candidate ordering, random search, and complete finite-pool enumeration on small diagnostics | Strong simple search and finite-menu ceiling | Enumeration equals stored max; seeded ordering deterministic; every evaluated trajectory charged. |
| B2 | Physical system identification plus candidate search | Strong domain-informed competitor | Fit all declared graph templates, never the hidden true template ID; recover noiseless G0 identifiable parameters; report non-identifiability and fit residuals. |
| B3 | CPU float64 GP BO, including contemporary log-EI/noisy constrained variants where appropriate | Strong small-dimensional black-box optimization | Reproduce an official constrained toy example; verify constraint sign; compare acquisition enumeration with a hand-computed small case. [R13, R26] |
| B4 | Continuous end-to-end signature predictor and continuous-query predictor | Main alternative to discretization | Same observations, targets, loss opportunities and optimizer; no restriction to a weak autoencoder or linear latent. |
| B5 | Adapted DeepSym action/effect binary bottleneck | Closest behavioral-symbol mechanism | Shared probe encoder replaces images; edit parameters replace supplied actions; effect-prediction objective and binary bottleneck semantics tested. It is an adaptation, not reproduction of published robot scores. [R04, R28] |
| B6 | Random-forest forward predictor and RIGID-style compatible-leaf inverse search | Strong interpretable functional inverse-design method | Hand-test leaf intersections and candidate membership; compare forward forest predictions before/after any compatibility patch. [R11, R29] |
| B7 | Witness-weighted continuous model and adaptive decision-tree partition, without the proposed discrete effect bottleneck | Isolate counterexamples/task weighting from token identity | Same witness rows, label budget, task weights, and depth/capacity tuning opportunities. |
| B8 | Uniform/K-means effect vocabulary plus the same residual and student; reconstruction-only divisive vocabulary | Isolate witness-driven vocabulary construction | Same K, residual rank, base encoder, data and downstream acquisition. |

B0–B4 and B6 can defeat the application without establishing anything about token-specific attribution. B5, B7, B8 and the no-code/soft-code ablations are indispensable before claiming a new discrete mechanism.

### 7.2 Physical system-identification baseline in detail

For each declared graph template, fit positive masses, edge stiffnesses/dampers, latch stiffness and threshold to the characterization traces with bounded nonlinear least squares. Use float64 and event-aware simulation. Start from multiple Sobol initializations, include a coarse threshold/profile search to avoid a zero-gradient latch threshold in subthreshold traces, and compare templates using validation prediction error plus an explicitly selected complexity penalty. Search all declared templates, including held-out ones as a known hypothesis family; do not give the fitter their hidden identity.

Fit the small-probe-only and release-capable conditions separately. In the former, preserve a set/profile of plausible thresholds rather than claiming a single identified threshold. In the latter, use observed release timing/port displacement consistently with the available trace sampling, not a hidden exact event value unavailable to neural methods. Predict design outcomes by simulating the fitted model, or an ensemble of plausible fits, and optimize over the same candidate pool.

Parameter fits can use millions of surrogate RHS evaluations; these are **compute costs**, not additional experimental ground-truth labels. Count CPU time, fitting restarts, and model evaluations separately from actual benchmark-oracle calls. An oracle that supplies true parameters is diagnostic only and never a competitive baseline.

### 7.3 Continuous and discrete controls in detail

The unrestricted continuous signature predictor maps \((h,d)\) to the full 247-number effect vector, with an exact zero-edit constraint. Use a small MLP and a larger validated version; allow residual connections and task-weighted losses. The continuous-query model maps \((h,d,q)\) directly to continuous operation margins and shared guard heads, using the same training queries. This control tests whether prototype-grid interpolation is the limiting factor rather than discretization.

The matched soft-code control replaces the hard prototype lookup by a continuous convex combination of the same prototypes and keeps the same residual branch. Report its full probability-vector capacity. A separate continuous latent/PCA basis predictor tests low-rank smooth structure without clustered prototypes. The soft-code control does not alone prove equivalence to every continuous architecture; it is a targeted mechanism test.

For the DeepSym adaptation, implement binary Gumbel-sigmoid with a verified straight-through hard forward pass and soft backward surrogate, plus a deterministic thresholded inference path. Begin with a sweep of bit counts comparable to the candidate vocabulary's capacity, but disclose that \(m\) binary units can support up to \(2^m\) patterns while a K-category code has K motifs. Include active pattern count, not only nominal capacity. Do not copy an old training loop's logging or checkpoint bugs into the baseline. [R04, R28]

### 7.4 RIGID-style baseline in detail

Fit forests to functional outcomes from the same probe-derived features and canonical design parameters. Offer both training-only engineered trace summaries and the shared learned embedding; select between them on development data. Use both a continuous-margin forest for the shared search adapter and a binary functional classifier for inverse target satisfaction.

The later inspected RIGID version uses tree-derived target-satisfaction scores averaged over the forest, with likelihood-based sampling. Its published leaf-intersection approximation can discard design-space intersections associated with different leaves. Preserve that version as a named reference adaptation, not an unspecified single-tree inverse lookup. [R11, R29]

For this experiment, let the auxiliary response coordinate be force amplitude; its binary label is the nine-scenario robust pass condition with the guards included. Tree inputs are `(probe_features, canonical_design, F)`. For each target force set, fix probe features, prune branches incompatible with that target, intersect relevant leaf IDs as in the reference method, recover their design boxes, assign their learned positive-class scores, and average the resulting per-tree score functions. Test leaf bounds and multi-interval targets against exhaustive traversal of a small hand-built tree.

On the finite pool, compute every candidate's score exactly and either rank or sample without replacement from the normalized scores; select rank versus sampling and the exploration setting on development data. This is an explicit finite-menu replacement for continuous MCMC, not an inaccurate claim to have reproduced published metamaterial scores. To maximize coverage rather than satisfy one prespecified interval, score targets from the same predicted force sets considered by the capability-first method; also retain direct forward-forest acquisition so a poor inverse-target heuristic cannot handicap the baseline. A continuous-domain extension may use Metropolis–Hastings with documented proposal, burn-in, duplicate rejection, and convergence checks, but is not required for the finite-menu comparison.

A zero-everywhere target score is an unachievable/unknown proposal, not a case removed from the evaluation denominator. Include the paper-style 1,000-tree, leaf-size-one setting in addition to the regularized pilot sweep. Offer training-only class balancing to all relevant classifiers; any synthetic oversampling is labeled as such, never counted as a new physical observation, and is forbidden from crossing data partitions. Patch obsolete feature-count access only if required by the resolved scikit-learn version, preserving predictions and bounds semantics. [R29]

### 7.5 Contemporary context and diagnostic-only methods

S³ is locally plausible on its supplied smooth Spiral example: its repository documents CPU JAX and a macOS/Linux setup. Run that example in an isolated environment only when a claim about optimizing discrete state abstractions is made. Its twice-differentiable known-dynamics assumptions are not the unknown hybrid-latch problem. Excluding it from the main design comparison is a **problem mismatch**, not a claim that it cannot run on the laptop. [R15, R30]

AALpy is appropriate for a deliberately finite-state, finite-alphabet diagnostic with known reset semantics. Use the resolved release's supported L* and newer alternatives where relevant; do not give only WCCT adaptive counterexamples. An approximate randomized equivalence test on continuous traces is not the exact oracle assumed by automata-learning theory. [R06, R27]

Full VLA systems, HANO's original material workloads, and geometric locomotion policies are contextual references, not numerical head-to-head baselines for this task. Their datasets, embodiment, information, and objectives differ. No incomparable published score should appear in the primary leaderboard. [R16, R18–R20]

### 7.6 Tuning and accounting

Perform tuning on training/development mechanisms only. Start each model family with an inexpensive smoke configuration, then use the same measured training-time or evaluation-credit allocation per family, while allowing family-appropriate hyperparameters. For neural models vary width/depth, learning rate, regularization, residual rank, and capacity; for forests vary tree count, depth, leaf support and feature representation; for GPs vary kernel/noise/prior choices; for system identification vary template penalty and restart allocation.

Use validation predictive loss to reject broken fits, but choose competitive configurations by the actual development design metric and compute Pareto frontier. Extend baseline tuning before confirmation if its learning curve or optimization diagnostics show obvious underfitting or premature stopping. Every such extension is available to the proposed method under the same rule and recorded in the tuning ledger.

Maintain two comparisons:

**Matched-information/attribution:** same characterization, training labels, witness rows, model-capacity neighborhood, downstream acquisition and online budget. Generally useful optimizations are shared.

**Practical end to end:** each method uses its best validated workflow, including no-training search, physical priors, caching, and any useful preprocessing. Charge all project-specific preparation. A method that uses fewer resources need not waste them to match a neural model; report the actual tradeoff.

---

## 8. Confirmatory evaluation and statistical protocol

### 8.1 Freeze before opening protected outcomes

Write and hash `protocol/confirmatory.yaml` after pilot calibration. It must identify the primary population, method versions, selected configurations, seeds, candidate domain, online budget, metric, numerical gray-zone rule, audit grid, failure handling, paired analysis, replication count, and multiplicity procedure. Commit the hash before generating/releasing protected outcomes.

Use independent seed namespaces for train, development, protected transfer, protected IID, and later confirmation. The tuning process can know the mathematical template definitions, but must not inspect protected response traces, labels, summary scores, or oracle optima. Once protected results influence a decision, that population is exploratory for the revised claim and fresh confirmation is required.

### 8.2 Primary comparison

The main estimate is

\[
\delta=\tfrac12\{\overline{J_{WCCT}-J_{B^*}}\}_{G4}
+\tfrac12\{\overline{J_{WCCT}-J_{B^*}}\}_{G5},
\]

where \(B^*\) is the strongest credible workflow baseline selected on development data before confirmation. Always also report the continuous, physical-ID, GP, fixed-design, and RIGID comparisons, even when they were not selected as \(B^*\). Selection on development data does not entitle the report to suppress a stronger observed baseline.

Report absolute coverage-point difference, covered-force-width difference, confidence interval, per-template results, paired score distribution, and the percentage of systems helped/harmed. Relative improvement is secondary and undefined or misleading when the denominator is zero; never report only a favorable ratio.

### 8.3 Budget selection without manufacturing a win

Let \(b\) be the number of online design bundles, giving a primary online trajectory budget of \(55b\), plus the four characterization trajectories. Use a common fixed budget curve in development, beginning at the two shared initial designs and extending toward complete-pool enumeration. Choose a confirmation budget that is informative for the proposed use case and lies before all strong methods saturate, using the **baseline/headroom curves and measured evaluation cost**, not the location of the largest WCCT advantage. Save the entire development curve and the selection rationale.

If the whole candidate pool can be faithfully searched at negligible cost compared with fitting/training, stop the practical-efficiency claim rather than inventing an unrealistic evaluation penalty. A representation-only scientific study may continue if its claim and metrics are narrowed explicitly.

The final blind audit is a separate, identical evaluation expense, not adaptive information available to the optimizer. Include that expense in deployment cost. A final failed audit is a failed returned design; do not permit only the proposed method to try again for free.

### 8.4 Pilot-calibrated sample size and precision

Use an initial development pilot with independent base mechanisms from the declared development population, not repeated amplitudes treated as independent. Estimate paired score variance and between-training-seed variability. Select a fixed confirmatory sample size before protected access.

For equal stratum sizes \(n\) and paired within-stratum standard deviations \(s_4,s_5\), the approximate standard error of the equal-weight mean is

\[
SE\approx\tfrac12\sqrt{s_4^2/n+s_5^2/n}.
\]

For a desired half-width \(h\), a planning approximation is

\[
n\approx\frac{1.96^2(s_4^2+s_5^2)}{4h^2}.
\]

Because G4/G5 are protected, estimate planning variance conservatively from G3 and held-out training-template development mechanisms; include a variance-inflation sensitivity calculation. Use the pilot to simulate the final paired analysis and check achieved precision under skew, zeros, and failures.

No application-supplied economically meaningful coverage difference is available. Therefore do not invent a required improvement percentage. Report achievable half-width and minimum detectable effects under measured resource budgets. Choose precision small enough to resolve a scientifically useful fraction of the independently estimated **headroom above the strongest baseline**, and record that fraction as a measurement-design choice rather than a success threshold. If realistic precision cannot distinguish the effect from zero, label the result inconclusive.

For illustration only, with paired SD 0.10, a simple unstratified two-sided 95% interval of approximate half-width 0.02 requires about 97 independent systems; half-width 0.01 requires about 385. These are algebraic planning examples, not measured variances or prescribed sample sizes.

### 8.5 Replication and uncertainty

Use multiple complete training seeds; repeated acquisitions/noise seeds on the same mechanism do not create new independent physical systems. The deployment policy may be a fixed ensemble, but independent ensemble-training repeats remain a source of algorithmic variation.

Pilot a crossed design of mechanisms and training repeats. Increase repeats until the estimated Monte Carlo contribution from training randomness is small relative to the planned mechanism-sampling uncertainty, or report it as a material unresolved component. Do not average away unfavorable training seeds or select the best seed. Repeated optimization seeds conditional on one fixed training corpus do not estimate uncertainty over alternative training corpora; state that conditioning. A claim about training-data variability requires separately regenerated training corpora and a new cost/replication calculation.

The primary interval is a stratified paired bootstrap over independent mechanisms, preserving all per-mechanism paired method results. When multiple independent training repeats are included, use a crossed bootstrap that independently resamples training-repeat IDs and mechanism IDs, not a bootstrap that treats every mechanism/seed cell as independent. State whether inference targets the frozen trained artifact or the training procedure. With too few repeats to estimate procedure-level uncertainty credibly, limit the inference to frozen artifacts and present repeat sensitivity separately.

### 8.6 Multiplicity and repeated inspection

Predeclare one primary comparison, one primary budget, and one primary metric. Confirmatory mechanism claims against the continuous no-code control, reconstruction-only vocabulary, and witness-without-token control form a secondary family; use Holm adjustment or report simultaneous intervals. Exploratory capacity sweeps and subgroup analyses are labeled exploratory, not retroactively promoted to primary endpoints.

Use a fixed-sample confirmation. Do not repeatedly inspect significance and keep adding systems. A future sequential design is acceptable only after specifying an actual valid sequential rule and fresh protocol; it is not the default here.

### 8.7 Numerical audit and gray zones

Use a separate event-based float64 verifier with independently written force assembly and event logic, not the same function called with a smaller step. Share only declarative mechanism/attachment specifications. Cross-check against analytic linear solutions between events on small systems.

The fast solver must localize release within a step, integrate to the event, apply the mode change exactly once, then continue. Validate event time, continuity, energy removal, and response extrema. For the reference solver, segment at waveform breakpoints and latch events. Use dense output to bracket extrema and scalar root finding where valid; test progressively tighter tolerances/max steps. When reliable extrema bounds are unavailable, report numerical convergence intervals rather than formal certification.

Define a componentwise numerical uncertainty estimate from the refinement study. A continuous margin whose sign is unresolved must trigger additional refinement. If it remains unresolved within the measured feasible resource envelope, score it conservatively as failure for the operational primary metric and report an optimistic bound as sensitivity. Apply this symmetrically. If these unresolved cases materially change the method ranking, the result is numerically inconclusive, not a win.

Independent implementations reduce shared numerical bugs; they do not validate the physical model itself.

### 8.8 Guardrails and failure handling

Always report false-feasible predictions, installation releases, shock failures, maximum displacement/velocity violations, abstention/unknown rate, latency, peak memory, preprocessing/training cost, and design-cost proxy. A coverage gain that depends on an unreported guard regression is not accepted.

For a method exception or resource crash, retain the failure record. Fix demonstrable implementation bugs and rerun the entire affected paired block with a versioned amendment. For a genuine frozen-workflow failure, operational score is zero and failure frequency is reported; also give the conditional-on-completion score so the cause is visible. Never discard a difficult mechanism solely because one method fails.

### 8.9 Harness validation

Before comparing learning methods, establish that:

- Two wrappers around the same policy yield exactly paired choices/scores under the same seeds.
- The harness detects the known no-edit versus finite-grid oracle difference from Section 4.
- Shuffled training targets remove useful supervised prediction rather than producing a mysterious gain.
- Budget exhaustion, duplicate requests, cache hits, invalid attachments, and final-audit isolation are correctly accounted for.
- Boundary perturbations change labels in the expected direction in analytically tractable cases.
- Repeated timings are summarized as timing variability, not added as independent systems to a significance test.

---

## 9. M3 Max feasibility and hardware-adaptive execution

### 9.1 What is known and what is not

Apple lists several M3 Max memory configurations, including configurations well below the maximum. The user's exact machine is unknown. PyTorch's inspected MPS documentation exposes availability checks and currently describes macOS 14.0 or newer for its example; this is not proof that an arbitrary installed version or every operator works. [R23, R24]

The core workload requires small dense neural networks, CPU numerical integration, forests, PCA, and small Gaussian processes. There is no essential CUDA kernel, distributed training, paid API, downloaded model checkpoint, or physical device. Feasibility is structurally plausible, but **local compatibility and throughput remain unmeasured**.

### 9.2 Initial discovery: existing commands

These are actual shell/Python facilities, not project commands:

```bash
uname -m
sw_vers
sysctl -n hw.memsize
sysctl -n hw.ncpu
system_profiler SPHardwareDataType SPDisplaysDataType
df -h .
vm_stat
python3 --version
```

Record arm64 versus Rosetta, total unified memory, currently available storage, power state, low-power mode, and macOS version. Do not install a large toolchain or modify system settings destructively. Select an available supported ARM64 Python version after checking package wheels; Python 3.11 or 3.12 is a reasonable first compatibility candidate, not a required or preverified choice.

### 9.3 Software stack and dependency policy

Core packages: NumPy, SciPy, scikit-learn, PyTorch, matplotlib, pytest, a YAML reader, and optionally psutil for process telemetry. BO adds compatible BoTorch and GPyTorch. Standard-library JSON, CSV, SQLite, pathlib, hashlib, and dataclasses suffice for records and indexing. Use `.npy`/`.npz` plus JSON manifests initially; no database server or large data framework is necessary.

Resolve compatible stable releases with native ARM64 wheels at execution time, run the smoke tests, then save exact package versions, wheel/source hashes, dependency lock, Python version, OS and source revision. Documentation observed during research included PyTorch 2.14, SciPy 1.18, and BoTorch 0.18.1 pages; **these are documentation observations, not a tested installation prescription**. The small Linux check used different versions. Do not fabricate a working lockfile before local resolution. [R24–R26]

Use separate environments for optional S³/AALpy reproduction to avoid forcing old or specialized dependencies into the core environment. The S³ repository recommends Python 3.11 and CPU JAX. Do not require Docker on the Mac; the repository documents a native setup. [R27, R30]

### 9.4 Critical operator and precision checks

Run CPU float64 simulation and CPU float64 GP fitting by default. Neural training uses float32; teacher/reference labels retain float64, and any float32 label conversion must be shown negligible compared with solver uncertainty.

Test on the resolved MPS installation: convolution, pooling/masking, dense layers, SiLU, softmax/cross-entropy, argmax/gather, residual clipping, Adam updates, checkpoint save/load, and gradient flow through the adapted binary bottleneck. Compare small seeded CPU/MPS losses and gradients within a tolerance justified by accumulated float32 roundoff. Train a tiny overfit example on both backends and compare achieved predictions, not merely successful import.

Do not assume float64, all sparse operations, custom kernels, or every optimizer path are supported on MPS. None is necessary to the primary neural model. If a needed neural operation fails or silently falls back, use the faithful CPU float32 implementation for every matched neural comparison and record the performance effect. A CPU fallback is acceptable; changing the learning mechanism to fit an accelerator is not.

Example availability probe using documented facilities:

```python
import platform
import torch

print(platform.platform(), platform.machine(), torch.__version__)
print("mps_built", torch.backends.mps.is_built())
print("mps_available", torch.backends.mps.is_available())
if torch.backends.mps.is_available():
    x = torch.randn(8, 16, device="mps", requires_grad=True)
    loss = (x @ x.T).square().mean()
    loss.backward()
    torch.mps.synchronize()
    print("smoke_loss", float(loss.detach().cpu()))
```

This is only an availability/small-operation check; it does not replace the full kernel/gradient test matrix.

### 9.5 Memory model

For float32 Adam training with \(P\) parameters, allow approximately \(16P\) bytes for parameters, gradients, and two moment arrays, before activations, temporary kernels, allocator caching, or extra parameter copies. Some optimizer implementations allocate additional buffers; measure them. For a million-parameter model the nominal parameter/gradient/moment component is only about 16 MB; this does not mean total training memory is 16 MB.

Use

\[
M_{peak}\approx M_{runtime}+M_{dataset\ resident}+M_{model/optimizer}
+M_{activations}+M_{solver\ batch}+M_{temporary/cache}.
\]

CPU and MPS use unified physical memory. Do not add reported CPU and device allocations blindly if they refer to shared or overlapping allocations. Record process resident memory, available system memory, swap/memory pressure, and available MPS telemetry independently. Feature-test telemetry functions on the installed version rather than assuming an API exists.

A signature needs \(247\times4=988\) bytes in float32 before metadata. For 512 mechanisms and 25 designs this is about 12.6 MB for one signature tensor; keep no-edit signatures once per mechanism. Four 128-sample, four-channel probe arrays for 512 mechanisms require about 4.2 MB. By contrast, retaining every internal integration step for millions of trajectories can dominate storage by orders of magnitude. Store summary outcomes for all runs and detailed traces only for characterization, a reproducible audit sample, disagreements, and selected designs.

An ensemble is trained serially and its members are evaluated in chunks if needed. Codebook storage is \(4K\times247\) bytes plus basis and metadata. For the small per-mechanism GP with \(b\) observations, kernel storage scales as \(O(b^2)\) and dense factorization as \(O(b^3)\). A GP over every global training trajectory is not proposed.

### 9.6 Calibration and sizing procedure

1. Measure fast and reference solver throughput separately for G0, G3, G4-like and G5-like **nonprotected synthetic calibration parameters**, including near-event cases. Do not use protected outcome seeds for this benchmark.
2. Measure complete training steps, validation prediction, candidate scoring, GP updates, and independent audit throughput at several batch sizes.
3. Synchronize MPS before and after timed regions, warm up kernels, distinguish cold start from steady state, and record wall time including CPU-to-device movement where used. Interleave method timing order and repeat across sessions to expose thermal/system variation.
4. Select a conservative working-memory budget from actual free memory and observed peak pressure, retaining operating-system headroom. Begin with small batches; increase until throughput stops improving or memory pressure rises. Do not target the advertised memory maximum.
5. Estimate each stage using measured rates and its exact workload count. Apply an observed variability margin, not a claimed hardware speedup based on nominal GPU cores.

For \(N\) training systems and \(D\) nonzero training designs per system,

\[
N_{train\ trajectories}=85N(D+1)+4N.
\]

For \(N_t\) test mechanisms, \(M\) compared workflows, \(R\) complete procedure repeats, \(b\) online bundles, and final audit count \(a\), a conservative count is

\[
N_{online+audit}\approx N_tMR(55b+a)+4N_t,
\]

before cache savings and before any secondary endpoint audits. Cache savings may reduce compute, but logical per-method budget use is still recorded. At the initial audit resolution \(a=589\). A one-million-trajectory stage takes about 16.7 minutes at 1,000 trajectories/second, 2.8 hours at 100/second, or 27.8 hours at 10/second. These are conditional arithmetic examples, **not M3 throughput estimates**. The reference solver may be substantially slower than the vectorized fast solver.

Training time is measured steps/epoch × epochs × configurations × independent repeats, plus teacher construction and validation. Teacher split search must be profiled: coordinate scans are cheap at this scale, but naive evaluation of every candidate against every full signature can be the dominant preprocessing cost. Cache prefix statistics where possible and share equivalent optimizations with baseline trees/clustering.

### 9.7 Resource-dependent branches

If memory is tight, stream signatures and train ensemble members serially; reduce physical batch size with gradient accumulation that preserves the effective batch and schedule. If CPU simulation is the bottleneck, vectorize by template and use a measured number of worker processes without oversubscribing BLAS threads. If reference verification dominates, first improve boundary bracketing and audit only predeclared selected designs rather than every search candidate—but never replace the final verifier by the surrogate.

Reducing the number of independent test mechanisms changes statistical precision; reducing query resolution changes the scientific task; reducing network size changes capacity. Those are methodological amendments, not invisible memory optimizations. Report the resulting limits. A smaller artifact that faithfully tests a component is preferable to an unfaithful imitation of a larger model, but the conclusion must be narrowed accordingly.

### 9.8 Checkpointing, interruption, and recovery

Write datasets in independently validated shards with hashes and atomic completion markers. Save model, optimizer, scaler, codebook, residual basis, split tree, calibration state, all random-generator states, epoch/minibatch position, sampler state, and dependency/source hashes. Include the online acquisition state, observations, pending request IDs, used budget, cache keys and final selection state.

Use a write-ahead request ledger. A simulator request has a unique ID and transitions through pending, completed, or failed. Resume must not obtain the same measurement for free or charge it twice. Checkpoints are atomic temporary-file writes followed by rename. Keep the last known-good checkpoint, not only the newest one.

Perform an interruption test during data generation, neural training, and online search. CPU deterministic paths should reproduce the uninterrupted run exactly where supported; GPU paths may require numerical rather than bitwise parity. If exact GPU continuation is not supported, retain that fact and use the CPU reproducibility path for the decisive recheck.

---
## 10. Ordered work packages and decision gates

All `python -m wcct ...` commands below are **proposed project commands that the coding agent must implement**, not existing installed software. The scientific definitions in Sections 5–8 take precedence over convenience defaults. Every work package writes a decision record with `continue`, `revise`, `stop`, or `inconclusive`, its evidence, and its next dependency. A stopped application claim does not automatically stop a justified component study.

### WP00 — Establish a reproducible local execution envelope

**Objective/risk:** unsupported dependencies, wrong architecture, insufficient disk, and misleading acceleration assumptions. **Entry:** this plan and a fresh target laptop; no simulation or training required.

**Actions and files:** create the repository in Section 12; run the discovery commands in Section 9 into `artifacts/environment/hardware.json`; create an isolated native Python environment; install the smallest numerical/test stack first, then PyTorch and optional BO dependencies. Record wheel versions, source revisions, dependency hashes and license notices in `environment.lock` and `THIRD_PARTY.md`. Implement `python -m wcct doctor --out artifacts/environment` and a CPU/MPS operation-and-gradient smoke test using the actual proposed encoder, hard lookup, loss, and optimizer. Do not download checkpoints or the original robot simulation stack.

**Tests/outputs/resources:** finite CPU float64 integration and linear solve; finite forward/backward gradients; tiny-dataset overfit; CPU-versus-MPS float32 loss/gradient comparisons within a recorded tolerance justified by CPU repeat variability. Profile increasing mini-batches and one small simulation batch; save peak memory, throughput and transfer-inclusive latency. This stage uses tiny arrays and no full dataset.

**Acceptance/branches:** use MPS only after those tests pass. On unsupported MPS operations, use the faithful CPU neural path for every affected matched comparison. On package conflict, isolate optional reference implementations rather than changing the primary mathematics. If even the CPU numerical/test stack cannot be installed, record a blocking compatibility issue; do not proceed with invented performance estimates. If storage is low, stream shards and remove only regenerable project-owned intermediates with a recorded policy.

### WP01 — Test information and composition before learning

**Objective/risk:** non-identifiability, false equivalence, and an ill-defined token algebra. **Dependencies:** WP00 numerical stack. **Inputs:** Section 4 counterexamples and the exact laws in Section 5.

**Actions and files:** implement `tests/test_identifiability.py`, `tests/test_attachment_algebra.py`, and `python -m wcct diagnostics analytic`. Reproduce the series-spring counterexample algebraically. Generate two latches differing only in threshold, with both thresholds above every safe-probe displacement; require identical recorded safe traces and a later query with different release outcomes. Verify simultaneous attachment composition by adding `(k,c,b)` and comparing against explicitly summed forces. Test non-transitivity of epsilon proximity and reject an implementation that obtains semantic classes by transitive closure alone.

**Outputs/resources:** `analytic_checks.json`, explicit paired traces, event and margin differences, and a table listing which information would distinguish each pair. Only a handful of short trajectories is necessary; calibrate their runtime through WP00.

**Acceptance/branches:** exact identities must pass to floating-point precision and numerical identities must converge under refinement. If safe traces differ because one twin releases, fix the constructed control rather than claiming observability. If internal-edit twins remain observationally identical but need different edits, retain them as a mandatory negative control and continue with external attachments. If a proposed extension requires hidden information, add a priced measurement or narrow its claim before implementation.

### WP02 — Implement and challenge the physical oracle

**Objective/risk:** wrong dynamics, event errors, free preload energy, and shared verification bugs. **Dependencies:** WP01. **Inputs:** graph definitions, trial protocols and margin conventions in Section 5.

**Actions and files:** implement `physics/spec.py`, `physics/fast.py`, `physics/reference.py`, and `physics/analytic.py`. The fast path batches equal-topology systems and performs event-localized integration. The reference path separately assembles edge forces, segments input breakpoints, uses float64 event integration, and searches extrema on dense output. Analytical linear state-transition checks use a matrix exponential or an independently solved linear system between events. Include the initial attachment energy in the energy ledger; a preloaded spring is not a source of free work.

Implement `python -m wcct validate-physics --config configs/physics.yaml`. Test zero forcing, zero attachment, damping dissipation, node relabeling, graph incidence signs, no-release/release trajectories, event exactly at installation end, and near-grazing release. Compare direct energy change with input work, damping loss, and the declared latch-energy removal. Reject negative masses/stiffnesses, out-of-domain attachments, NaNs, and invalid graph connectivity.

**Outputs/resources:** a convergence table for margins, event times and extrema on ordinary and boundary-targeted cases; solver disagreement traces; calibrated trajectories/second and memory by template. Begin with G0/G1 and add G2/G3; reserve parameter-sampled protected G4/G5 outcomes for confirmation, but structural unit tests may exercise their force assembly with non-evaluation fixtures.

**Acceptance/branches:** continue only when disagreement is below the numerical resolution required by the observed capability boundaries, or cases are explicitly classified unresolved and handled symmetrically. If a particular fast scheme cannot localize events reliably, use the reference solver for labels and profile the cost; do not weaken the audit. If different correct solvers disagree about an idealized event convention, resolve the model definition and regenerate affected data before learning. A numerical bug is not evidence against tokens.

### WP03 — Establish headroom and an informative workload

**Objective/risk:** a fixed attachment or cheap search solves the task, or all designs fail. **Dependencies:** WP02. **Inputs:** G0–G3 development fixtures only; canonical candidate pools.

**Actions and files:** implement `design/candidates.py`, `evaluation/headroom.py`, and `python -m wcct headroom --config configs/pilot.yaml`. Begin with the 12-system diagnostic specification in Section 4, then use independent development systems on G0–G3. Evaluate the canonical 40-design grid, nested Sobol pool, no edit, best fixed attachment, high-damping heuristic, and inexpensive GP/random search. Audit selected contenders independently; on the smallest manageable development subset, verify the whole candidate menu to assess the reliability of the finite-menu ceiling. Do not label an unverified fast-solver maximum a rigorous oracle bound.

Plot per-system coverage, design-cost tradeoffs, budget curves, boundary mass, and the frequency of different optimal attachments. Measure how much performance remains above the strongest simple workflow, not merely above doing nothing. Increase development mechanisms until uncertainty about headroom—not WCCT's unimplemented advantage—is resolved at the available numerical precision.

**Outputs/resources:** `headroom.json`, fixed-design choice, canonical pool hash, baseline budget curves, and cost projections. Scale only from measured rates. If full-pool enumeration is already cheap, retain it as the strongest practical comparator.

**Acceptance/branches:** proceed to an application comparison only when there is resolvable improvement available over fixed/search baselines and the task actually produces distinct attachment choices. If G0 lacks headroom, test the already declared multi-node templates before rejecting the mechanism. If G0–G3 remain trivial, do not tune constraints until a token method wins. Record a task amendment motivated by an engineering need, such as damping cost or a second measured load condition, and rerun headroom before building models. If no defensible nontrivial task survives, deliver a negative application assessment; a rate–distortion study may continue under an explicitly narrower claim.

### WP04 — Freeze a common data and evaluation harness

**Objective/risk:** leakage, unequal queries, inconsistent margins, or hidden simulation costs. **Dependencies:** WP02–WP03. **Inputs:** accepted generator version and development-selected candidate/query settings.

**Actions and files:** implement `data/generate.py`, `data/splits.py`, `evaluation/oracle.py`, `evaluation/ledger.py`, and the interfaces in Section 12. Generate the initial 128-system training pilot and disjoint development mechanisms. Partition mechanisms before generating their attachment/query rows. Save probe channels, 247-number signatures, separate guard labels, numerical status, generator version and hashes. Construct protected seed manifests without running their scored outcomes. Implement oracle access through a budgeted interface that does not expose hidden mechanisms to methods.

Implement `python -m wcct make-data --config configs/pilot.yaml`, `python -m wcct test-harness`, and `python -m wcct estimate-cost`. Test identical-policy wrappers, shuffled labels, exact finite-menu maxima, invalid/duplicate requests, interrupted requests, and unavailable audit feedback.

**Outputs/resources:** immutable pilot shards, split manifests, ledger test reports, and estimated full-stage costs. The pilot count is 141,440 signature trajectories plus 512 characterization trajectories under the initial 128-by-13 specification. Store only necessary signatures and selected traces.

**Acceptance/branches:** no train/development/protected mechanism hash overlaps; no method receives a hidden ID/parameter or an uncharged label; paired wrappers agree. A failure blocks all learned comparisons. If generation cost exceeds the viable envelope, reduce storage and improve vectorization first; changing scientific sample size or query resolution requires a recorded amendment and revised precision claims.

### WP05 — Reproduce strong local baselines before adding tokens

**Objective/risk:** an apparent gain caused by weak alternatives. **Dependencies:** WP04. **Inputs:** identical training/development shards, oracle interface, and configuration schema.

**Actions and files:** implement B0–B4 and B6 first in `methods/`; validate physical-ID recovery on analytic G0 cases, constrained-GP acquisition signs on a small known problem, and forest leaf inversion on a hand-built tree. Train the unrestricted continuous signature/query predictors using the same task losses and full target access. Compare best-fixed, GP, physical-ID, continuous, and RIGID-like workflows across the development budget curve. Use both engineered probe features and a learned probe embedding for the forest where feasible; do not force a tree to ingest only a poorly scaled raw trace.

Run `python -m wcct baseline-checks` and `python -m wcct tune --phase exploratory --method <name>`. Record the installed versions, resolved API choices, restart/tuning costs, convergence, prediction residuals, and exceptions. RIGID/DeepSym code is an implementation reference; transplanting an old repository without tests is not faithful reproduction.

**Outputs/resources:** baseline correctness reports and development performance/cost frontiers. Neural and tree settings are in Section 10.1 below; physical-ID restart counts are doubled while held-out prediction or design score improves beyond numerical/seed variability, then frozen.

**Acceptance/branches:** if a baseline fails an identifiable tiny case, fix it before ranking. If a stronger baseline eliminates headroom, stop or narrow the application claim. If two baselines are statistically indistinguishable in development, choose the simpler/lower-cost one as the primary comparator while retaining both in confirmation. If a relevant implementation is incompatible, adapt its documented mathematics with parity tests and label the adaptation; do not silently exclude a feasible strong baseline.

### WP06 — Build the minimum complete WCCT and attribution controls

**Objective/risk:** no recurring motifs, teacher-only success, hidden continuous bypass, or task weighting mistaken for token value. **Dependencies:** WP05; nontrivial remaining headroom or an explicitly narrowed representation study.

**Actions and files:** implement `representation/signature.py`, `representation/partition.py`, `representation/residual.py`, `models/probe_encoder.py`, and `methods/wcct.py`. Build deterministic teacher partitions, save actual witnesses, fit the bounded PCA residual, and train the observable student. Implement identity, K-means/reconstruction-only, no-residual, continuous low-rank, soft-code, witness-trained continuous/tree, and adapted DeepSym controls. Add tests that perturb raw observations or attachment inputs downstream of the bottleneck and verify no forbidden decoder path exists.

Use `python -m wcct build-vocabulary --config ...`, `python -m wcct train --method ...`, and `python -m wcct evaluate --split development`. Measure teacher distortion, student assignment/prediction error, active code/support counts, residual contribution, false-feasible errors and design score. Hold out entire systems when checking code transfer.

**Outputs/resources:** versioned vocabularies with witness provenance, trained artifacts, capacity/residual curves, and shared-acquisition comparisons. Profile teacher construction separately from training. Extend data or capacity only when the measured error decomposition identifies a solvable limitation and strong baselines get the same opportunity.

**Acceptance/branches:** teacher success without student observability is insufficient; revise probes only through a costed protocol amendment. If a code is effectively an attachment-ID lookup, compare the attachment-only token control before proceeding. If residuals carry all useful variation, classify the result as a continuous representation and reject the discrete claim. If a task-weighted continuous model matches WCCT, retain the task-objective insight without crediting tokenization.

### WP07 — Calibrate and lock the confirmatory protocol

**Objective/risk:** underpowered confirmation, budget cherry-picking, seed selection, or unbounded tuning. **Dependencies:** WP05–WP06 plus passed harness and restart tests.

**Actions and files:** use development-only paired runs to estimate score variance, training/acquisition variation, throughput, memory, numerical ambiguity and attainable precision. Select primary baseline, budget, model configuration, inference mode, ensemble/repeat count, sample size, confidence procedure and multiplicity family according to Section 8. Calibrate the audit force grid by doubling it and refining boundaries on development examples until changes are below the declared reporting resolution; store the final common grid. Retain a denser-grid sensitivity schedule.

Implement `python -m wcct calibrate` and `python -m wcct freeze-protocol`. Write `protocol/confirmatory.yaml`, a human-readable rationale, and SHA-256 hashes of every material dependency. Unit-test the analysis on simulated paired scores with known null and positive effects. The sample-size calculator must not read protected outcomes.

**Outputs/resources:** signed-off machine-readable protocol, preregistered analysis script, capacity/cost ledger and failure policy. “Signed-off” means a local immutable project record, not external peer review.

**Acceptance/branches:** proceed only when precision can answer the declared question within measured local resources. Otherwise choose a narrower claim/population before protected access, or mark the planned comparison resource-limited/inconclusive. Do not reduce the required baseline set to create enough budget for more favorable proposed-method runs.

### WP08 — Execute the protected comparison

**Objective/risk:** establish the primary result without adaptive contamination. **Dependencies:** WP07 protocol hash matches all active artifacts.

**Actions and files:** `python -m wcct confirm --protocol protocol/confirmatory.yaml` generates independent G4/G5 parameter draws, runs every frozen workflow with paired trial conditions, and saves blinded method outputs. Methods receive only their allowed observations. Run final independent audits after selections are committed. `python -m wcct analyze --protocol ...` computes the frozen primary interval and guardrails, including all failures. Store complete per-mechanism results before producing aggregate plots.

**Outputs/resources:** protected raw ledger, chosen designs, audit traces/status, paired score table, primary analysis, and all unfavorable outcomes. Estimated cost follows Section 9.6 with the actual audit grid and number of methods/repeats.

**Acceptance/branches:** a favorable primary interval is preliminary evidence for this frozen comparison, not yet token-specific causation. A negative well-resolved effect is evidence against the local competitive claim. An interval spanning meaningful benefit and harm is inconclusive. Implementation/audit failure invalidates the affected block; fix and rerun all affected paired methods with an amendment, never only WCCT. Do not reopen tuning against protected outcomes.

### WP09 — Test attribution, transfer boundaries, and the compiler claim

**Objective/risk:** wrong explanation for a gain, fragile boundary behavior, and confusion between prediction and invention. **Dependencies:** WP08 for confirmatory analyses; predeclared ablations may run alongside WP08 without influencing it.

**Actions and files:** execute the matrix in Section 11, starting with matched continuous/task-weighted controls and numerical refinement. Implement capability-first motif proposals in `design/compiler.py` and compare with identical-predictor direct acquisition. Conduct attachment-composition and safe-only twin tests, then cubic-force/noise/missing-sensor shifts. Recheck selected results with a separately written numerical path. For any new claim suggested by exploration, freeze a fresh protocol and fresh parameter draws before confirmation.

**Outputs/resources:** effect-attribution table, failure atlas, generative realization success/empty-preimage rates, direct-versus-compiler paired results, and fresh-confirmation records. Allocate work in descending relevance to the observed gain; do not run every speculative extension merely because it exists.

**Acceptance/branches:** gains explained by task weighting, residual capacity, or extra information do not support the discrete claim. Gains surviving attribution support a scoped empirical result. Compiler failure rejects the additional search mechanism but need not erase a valid representation result. Noise/nonlinear failure limits generalization; it does not justify claiming simulator-to-real transfer.

### WP10 — Package a reproducible decision, including a negative one

**Objective/risk:** selective reporting and a result that cannot be resumed or checked. **Dependencies:** completed or explicitly stopped earlier packages.

**Actions and files:** run `python -m wcct report`; assemble `REPORT.md`, `DECISIONS.md`, `RESULTS.json`, frozen configs, locks, seeds, source hashes, raw metrics, failure records and a compact fresh-environment reproduction command. Include results relative to every required strong baseline, attribution outcomes, costs, numerical limitations, and break-even analysis. Run the interruption/resume and identical-policy checks again with final artifacts. Complete the three audits in Section 14 against what actually happened, not the hoped-for outcome.

**Outputs/resources:** a reproducible result bundle with a minimal smoke test and a full-run manifest; no requirement to redistribute third-party datasets or checkpoints not licensed for that purpose.

**Acceptance/branches:** classify the outcome using Section 13. A well-supported negative finding is a completed research outcome. An unexecuted stage is explicitly unexecuted. Do not label a retained hypothesis a demonstrated improvement.

### 10.1 Concrete exploratory training and tuning defaults

These are **starting configurations**, not measured optima, hardware guarantees, or hard research limits. They make implementation determinate while leaving adaptation governed by evidence.

For neural predictors, start with AdamW, learning rate `1e-3`, weight decay `1e-4`, full float32, and effective batch size 128 mechanism/design rows. Shuffle by a saved sampler and balance mechanisms so a mechanism with more cached designs does not silently receive more weight. Begin with the specified width-128 architecture. Validate once per epoch; for initial debugging use up to 200 epochs and retain the best validation checkpoint with patience 20. Before declaring convergence, check that doubling the training allowance or lowering the learning rate does not improve development performance beyond seed/numerical variability. An epoch cap reached with improving curves is a censored run, not a converged baseline.

Start learning-rate tuning at `{3e-4, 1e-3, 3e-3}`, width at `{64,128,256}`, and weight decay at `{0,1e-4}`. Use staged exploration, not the full Cartesian product by default: first ensure overfitting of a tiny sample, then tune learning rate at width 128, then capacity/regularization. Apply equal tuning opportunities to comparable neural models. Expand a boundary of the search range when the best configuration lies there and measurable improvement remains. Log training steps, examples, wall time and all trials, including failures; do not equate an equal number of differently priced trials with equal expenditure.

For the teacher, begin with `lambda_FP=lambda_FN=1` and include `{0,0}`, `{1,1}`, `{2,1}`, and `{1,2}` as development candidates after loss normalization. For the student, initialize CE, normalized residual-Huber and reconstructed-signature weights at one; tune one decade up/down only when gradient magnitudes and development errors indicate imbalance. Share reconstructed task-loss choices with continuous models. Use the differentiable definition in Section 12.3, not a hard feasibility indicator with zero gradient.

For forests, begin with 256 trees; tune minimum leaf sizes `{2,5,10}` base-mechanism-equivalent samples and feature subsampling `{0.5,1.0}`, with unlimited depth only when leaf support is enforced. Compare doubling the tree count against prediction/selection variability before freezing. Data rows from one mechanism are not independent support. For the DeepSym adaptation, start binary lengths `{3,4,5,6}` and a temperature schedule from 1.0 to 0.3; validate deterministic hard inference and provide a higher-capacity option when these settings underfit. Document the straight-through adaptation rather than claiming every detail matches the original code.

For GP acquisition, use exact CPU-double posterior calculations on each small online history, normalize the three canonical coordinates by their declared bounds, and standardize scalar scores using training/history statistics without future outcomes. Check multiple kernel hyperparameter starts. If fitting is unstable, use a validated fixed-hyperparameter configuration selected on development data, not arbitrary per-test rescue. Every method receives the same rescue policy in the common-adapter comparison.

---

## 11. Attribution, adversarial tests, and failure analysis

Run the tests needed to explain the observed result before extending the scope. Report effect sizes and paired uncertainty, not only whether an ablation is “significant.”

| Test | Controlled comparison | What a positive result would mean; what failure means |
|---|---|---|
| Discrete identity | Hard WCCT vs continuous full signature, low-rank continuous and soft-code controls | Benefit attributable to discreteness requires more than beating one weak continuous bottleneck. Equality favors the simpler implementation. |
| Witness refinement | WCCT vs reconstruction-only partition, K-means, witness-weighted continuous and adaptive tree | A shared witness benefit without a token-specific increment is an objective/data-selection contribution. |
| Continuous bypass | Residual rank zero, small ranks, larger ranks; frozen base vs practical end-to-end encoder | If the benefit appears only with a high-capacity residual, do not credit a compact discrete alphabet. |
| Trivial code semantics | Attachment-only code prediction; prototype lookup by attachment bin; no-probe input | Comparable scores show that codes may encode the design menu, not transferable system-dependent changes. |
| Semantic bottleneck | Teacher oracle assignment vs observable student; true base signature vs predicted base as diagnostic only | Identifies where performance is lost. Oracle-assisted scores are not deployment results. |
| Query weighting | Equal margin/decision losses, shared witness rows, equal teacher queries | Removes the possibility that WCCT wins because it alone sees boundary-relevant labels. |
| Budget and compute | Equal logical observations; equal practical total wall-time frontier | Distinguishes measurement efficiency from overall efficiency. Cache hits or expensive teacher construction cannot disappear from cost accounting. |
| Composition | Train pure spring and pure damper effects; evaluate allowed combined attachments on fresh systems | Tests transfer to held-out combinations; force-law addition is already known and not a learned discovery. |
| Interaction excess | Compute `Y(d1⊕d2)-Y(d1)-Y(d2)+Y(0)` using reset, matched trials | Measures when adding effect tokens is invalid; all four outcomes are charged. A nonlinear residual is not automatically a reusable interaction alphabet. |
| Safe-only twins | Identical subthreshold traces with different release thresholds | Calibrated uncertainty/abstention is correct. Confident systematic discrimination suggests leakage or an invalid twin construction. |
| Internal-edit twins | Same pre-edit behavior, different consequences of a component edit | Deliberately violates the core external-interface assumption. Failure here does not falsify a correctly scoped external model. |
| Numerical boundaries | Halve fast step, tighten reference tolerances, densify force grid and extrema search | A ranking erased by convergence is a numerical artifact or unresolved result. |
| Topology and parameters | G4/G5 separately; IID holdout; later fresh parameter tails | Shows whether an average is driven by one template or an easy slice. Two templates do not establish universal topology transfer. |
| Sensor noise | Fixed primary noiseless model evaluated with 0.5%, 2%, and 5% of training-channel SD added to displacement/velocity; separately retrain noise-aware comparators | Tests mismatch and trainable robustness separately. Use shared per-mechanism/probe noise draws, not different favorable noise for WCCT. |
| Sensor omission | Remove velocity, provide displacement/input/time only to all methods | Tests the idealized direct-velocity sensor assumption. Finite-difference velocity is a shared, costed preprocessing alternative. |
| Model mismatch | Add the declared passive cubic edge terms; vary latch threshold/parameter tails; initially evaluate without retraining | A useful stress test, not real-world validation. Distinguish failure to detect out-of-domain inputs from failure to adapt with new training data. |
| History mismatch | Attach after a nonzero prior load or a released latch, explicitly providing or withholding that history | Exposes missing-state dependence. The core reset model makes no claim about these histories. |
| Irrelevant appearance | Add random graphical metadata or node labels independent of dynamics | A behavioral method should ignore them; numerical node permutation must preserve the physical prediction. No visual model is necessary. |
| Null controls | Same-policy wrappers, shuffled targets, random code labels with matched support, zero-edited effect | Detects evaluation leakage, arbitrary label artifacts and overfitting. |
| Compiler attribution | Capability-first vs direct search with identical predictor, adapter, candidate pool and budget | Isolates the generative proposal procedure. Prediction gains alone do not support this claim. |

Adversarial cases include narrow disconnected success bands, release exactly at the installation boundary, a design that opens without commanded force because of preload, a high-damping attachment that violates a new cost constraint, missing hidden-node damping, and an attachment outside the trained interface. A model must not label “unknown” as “safe.” Preserve unknown frequency and the resulting utility loss.

For strict bitrate studies, use actual fixed-point residual serialization at 4, 8, and 16 bits per coordinate, training-only ranges, saturation flags and deterministic round-to-nearest conventions. Include codebook/basis bytes and context/base costs. Re-evaluate accuracy after decoding bytes; do not estimate bitrate from a float tensor's conceptual labels.

A useful failure atlas contains one representative and the worst verified example of each failure mode, but summary statistics include every case. Save the original query, observations, predicted margins, actual margins, chosen design, code/support/residual flags and solver uncertainty. Do not select only visually convincing examples.

---
## 12. Executable repository, interfaces, and numerical conventions

### 12.1 Minimal repository

```text
wcct-poc/
  README.md                 # scope, bootstrap, smoke test, full-run instructions
  research_poc_plan.md       # this standalone specification
  pyproject.toml
  environment.lock          # resolved versions/hashes, not speculative pins
  THIRD_PARTY.md
  DECISIONS.md               # append-only stage outcomes and amendments
  protocol/
    exploratory.yaml
    confirmatory.yaml
    amendments/
  configs/
    physics.yaml
    pilot.yaml
    methods/
  src/wcct/
    __main__.py
    types.py
    doctor.py
    physics/{spec,fast,reference,analytic}.py
    data/{generate,splits,storage}.py
    representation/{signature,partition,residual}.py
    models/{probe_encoder,losses,calibration}.py
    methods/{base,fixed,sobol,physical_id,gp,continuous,deepsym,rigid,wcct}.py
    design/{candidates,adapter,compiler}.py
    evaluation/{oracle,ledger,headroom,metrics,statistics,report}.py
  tests/
    test_identifiability.py
    test_attachment_algebra.py
    test_physics.py
    test_signature.py
    test_partition.py
    test_no_leakage.py
    test_budget.py
    test_resume.py
    test_statistics.py
  data/{train,development,protected}/
  artifacts/{environment,vocabularies,models,runs,failures,reports}/
```

Use versioned YAML with explicit units and generator/schema versions. Refuse unknown configuration keys rather than silently ignoring misspellings. Keep code, dataset, protocol, vocabulary and model hashes distinct; a single generic “version” cannot identify which component changed.

### 12.2 Shared method and oracle contracts

The following Python is an **interface specification**, not a claim that a package implementing it already exists. Use dataclasses or equivalent validated records for the referenced types.

```python
class Method:
    def fit(self, training_data, development_data, config):
        """Return a frozen artifact and complete training/cost metadata."""
        raise NotImplementedError

    def begin(self, public_case, observations, candidate_pool, seed):
        """public_case contains protocol/interface data, never hidden physics."""
        raise NotImplementedError

    def propose(self, history):
        """Return one canonical, unused attachment, or explicit abstention."""
        raise NotImplementedError

    def observe(self, result):
        """Accept only a budget-ledger-issued online BundleResult."""
        raise NotImplementedError

    def finalize(self, history):
        """Commit exactly one attachment/abstention before audit access."""
        raise NotImplementedError

    def state_dict(self):
        """Include RNG, adaptation, acquisition and pending-request state."""
        raise NotImplementedError
```

`ObservationSet` contains arrays `time[probe,sample]`, `input`, `displacement`, `velocity`, `mask`, duration, waveform IDs, censoring flags, interface ID and protocol hash. It contains no mechanism seed, true graph ID, physical parameters, exact hidden event time, or target signature. Methods may receive an opaque case handle for accounting, but predictors may not use it as a feature.

`BundleResult` contains canonical attachment, actual public query coordinates, continuous operation margins, installation/shock indicators, numerical uncertainty/status, logical trajectory charge, request ID, and timing metadata. It does not contain the final denser audit or latent simulator state. `Selection` contains attachment, tested status, model version, predicted score, support flags and abstention reason. Ground-truth files are owned by the evaluator, not directly imported into method modules.

Use a strict mode that executes methods in a worker process receiving only serialized public records. This is an engineering leakage barrier, not a security boundary against malicious code. Unit tests monkeypatch hidden-parameter access to raise an error.

### 12.3 Remove implementation ambiguity in the teacher and student

**Canonical coordinates.** Store `(k,c,b)` in float64 in the candidate manifest, using deterministic generation and explicit decimal/byte encoding for hashes. Snap only documented construction duplicates, not arbitrary nearby continuous designs. For an imported `(k,c,l)`, compute `b=k*l`; require `b=0` when `k=0`. Clamp nothing silently. Each finite-pool design has one immutable ID and one physical triple.

**Event and pass conventions.** The latch releases on the first upward threshold hit, including equality. A release at or before time 2 is an installation failure. Continuous limits use nonnegative signed margins as passing, but unresolved numerical signs receive the gray-zone treatment from Section 8.7. Terminal displacement is taken at exactly time 6; extrema include initial and terminal points and both sides of the continuous event. Release is not a position/velocity reset. Signed guards are categorical indicators, never interpreted as distances from a physical boundary.

**Energy convention.** At installation, the mechanism begins at rest with the external preloaded spring already connected. Its initial stored energy is `0.5*k*l**2` when `k>0`; the chosen idealized installation supplies that energy. Audit subsequent input work, damping loss and latch-energy removal against the complete initial energy. The PoC does not model a robot physically installing the attachment or the installation work path.

**Deterministic partition.** For a leaf coordinate with at most 256 distinct effect values, consider all midpoints between consecutive distinct values. Otherwise use quantile levels `j/64`, `j=1,...,63`, with linear interpolation: for sorted values of length n and probability p, interpolate between indices floor((n−1)p) and ceil((n−1)p) using the fractional part of (n−1)p; deduplicate thresholds and discard cuts with an empty child or fewer than two base mechanisms per child. Evaluate all 247 coordinates, with stable coordinate and leaf ordering. Compute prototypes in float64. If no positive loss reduction remains, stop and report the attained K rather than producing empty or duplicate codes. A greedy partition may have local minima; compare multiple reconstruction initializations as an exploratory control, not as hidden WCCT-only compute.

**Huber and PCA conventions.** Use Huber transition 1 in the normalized margin units, averaging 243 operation entries and the four guard entries separately before summing. The residual PCA operates in these already defined signature coordinates, with no further whitening in the core path. Because leaf prototypes are arithmetic means, global residual mean is zero up to roundoff. Explicitly subtract and store any numerical mean; use an orthonormal basis and add the stored mean back during decoding. If additional feature scaling is used, include its inverse in both the equations and serialization and apply it to matched controls.

**Differentiable task training.** Hard capability labels are appropriate for teacher split evaluation and final metrics but have no useful gradient. For the student and continuous controls, at each force amplitude collect the 27 operation margins and four shared guards into a vector `v`. Define

\[
 m_\tau(v)=-\tau\log\sum_j\exp(-v_j/\tau),\qquad
 p_\tau(v)=\operatorname{sigmoid}(m_\tau(v)/\tau).
\]

Use numerically stable log-sum-exp and binary cross-entropy against the true robust capability label, with the same FP/FN weights as an available choice for every neural comparator. Begin with `tau=0.05` in normalized units and test `{0.02,0.05,0.1}` on development data. This is a training surrogate, not the definition of J, a calibrated probability, or a safety bound. As temperature approaches zero, the soft minimum approaches the actual minimum; its finite-temperature bias is why it is validated rather than assumed harmless.

For an auxiliary task gradient through hard categorical choice, use a straight-through one-hot vector: `w = one_hot(argmax(p)) + p - stop_gradient(p)`. The forward decoded prototype is hard; the backward surrogate is soft. Cross-entropy with the teacher remains the direct assignment loss. Save and test both train-mode and deterministic inference semantics. Use the same distinction for the adapted binary DeepSym control. No hidden high-capacity continuous mixture is permitted at hard-code inference.

**Calibration and unknowns.** Reserve a disjoint subset of development mechanisms for confidence calibration after hyperparameter choice, or use explicitly nested mechanism-level cross-validation. Record ensemble dispersion and empirical prediction-error bands. A useful conservative diagnostic computes the maximum normalized margin error across all calibrated queries/designs for each calibration mechanism, then takes its empirical 95th percentile; label it empirical, with sample size and coverage. It is not distribution-free under topology shift. Unknown/support violations cause a flag and, where the frozen policy specifies, abstention; report any benefit/coverage lost to abstention. Do not tune the abstention threshold using protected outcomes.

### 12.4 Dataset and run schemas

A mechanism manifest contains generator version, template ID, parameter draw, initial-state convention, seed and SHA-256. Its hidden fields are evaluator-only. A signature row contains mechanism hash, canonical design ID, query/domain hash, base/edited array references, masks, solver version, tolerance settings and numerical status. Store a readable schema with each dataset shard and validate shape, units, finiteness and hashes on load.

Each run JSON record contains:

```text
run_id, phase, timestamp_utc, source_commit, dirty_patch_hash,
protocol_hash, configuration_hash, environment_hash,
data_manifest_hash, split_hash, method, model_hash, codebook_hash,
training_seed, acquisition_seed, sensor_noise_seed,
logical_characterization_trajectories, logical_online_trajectories,
physical_solver_calls, RHS_evaluations, cache_hits, failed_requests,
preprocessing_seconds, training_seconds, adaptation_seconds,
inference_seconds, online_solver_seconds, audit_seconds,
peak_process_RSS_bytes, device_memory_observations, measurement_boundary,
selected_design_id, tested_status, prediction, audit_result,
failure_type, restart_parent, amendment_id
```

Use separate data, model, acquisition and noise RNG streams derived from a master seed with a stable derivation algorithm. Store the resulting seeds explicitly. A protected mechanism's generator seed must never become a model input or a correlated batch index.

### 12.5 Logical cost, shared caches, and resume semantics

Count a new online design bundle as 55 logical trajectories under the frozen initial protocol, regardless of which method requested it or whether another method has already populated a shared physical cache. A shared cache saves actual computation but cannot give a method free information. Characterization costs four trajectories per task per deployed workflow even when the paired benchmark physically reuses the same observations.

A request for a previously observed canonical attachment returns no new information and cannot consume an unbounded sequence of free attempts. The interface rejects repeated proposals and logs them; a reproducibly looping frozen policy is an operational failure. Separate programmer error from a genuine algorithmic failure using the versioned repair policy. Invalid attachments are never silently projected into the legal domain.

A write-ahead ledger uses `(run_id, case_id, request_sequence)` as the logical request key and `(physics_hash, design_id, query_hash, noise_seed, numerical_settings)` as the physical cache key. Reserve logical cost before execution. On interruption, resume the same request: an incomplete physical run can be recomputed, but the logical observation is charged exactly once when its result is available. Record wasted actual compute separately. Never reuse a low-accuracy cached result as a higher-accuracy audit because the keys ignored numerical settings.

### 12.6 Licenses, provenance, and agent authority

The core dataset is generated from the equations in this plan; no downloaded weights, scraped observations, or restricted benchmark assets are required. Retain seed/generator provenance and a project-owner-approved license before public release. The public DeepSym, RIGID, AALpy, S³, and BoTorch sources inspected identify permissive MIT licensing; retain their actual pinned license files and notices when reusing code. A repository license does not automatically license external simulator assets, data or weights. [R26–R30]

The agent may make routine non-destructive choices such as chunk size, local directory naming, logging format, equivalent vectorization, and tested compatibility patches. It may not silently change the hypothesis, target population, information access, candidate domain, oracle budget, loss eligibility, primary endpoint, baseline set, statistical test, or failure policy. Such changes require an amendment with rationale, affected artifacts, contamination consequences, and the conditions for fresh confirmation.

Keep all unfavorable runs. Store enough state and written decisions for another coding agent to resume without chat history. Public release and installation of optional large external toolchains are not automatic consequences of this plan.

---

## 13. Interpretation, continuation, and evidence-triggered pivots

### 13.1 Outcome taxonomy

| Observed outcome | Supported conclusion | Required next action |
|---|---|---|
| Physics/harness checks fail | Implementation or measurement is invalid. | Repair, regenerate affected artifacts and rerun paired blocks; make no representation claim. |
| Safe-only twins are ambiguous | The observations lack identifying information. | Keep uncertainty/abstention or price an additional probe; do not enlarge the network as the first remedy. |
| Fixed attachment or exact cheap search matches the ceiling | Little practical headroom exists in this workload. | Stop the efficiency claim; retain the negative finding or justify a genuinely different task before expansion. |
| Teacher motifs compress effects but observable student fails | The code may be compressible but not inferable from allowed data. | Diagnose probe sufficiency and code predictability; no design-system success claim. |
| WCCT wins only against reconstruction baselines | Task weighting or a weak comparator may explain the gain. | Require end-to-end continuous, RIGID, physical-ID and witness-without-token comparisons. |
| WCCT beats strong workflows but not matched continuous controls | An engineering/workflow gain may exist, but discreteness is not isolated. | Prefer the simpler winning mechanism; report token-specific hypothesis as unsupported. |
| WCCT beats matched controls at fixed labels but costs much more overall | A measurement/compute tradeoff exists. | Quantify amortization and the scenarios where it matters; do not call it an unconditional improvement. |
| WCCT improves held-out-template utility with stable attribution and guardrails | Evidence for a scoped synthetic representation advantage. | Fresh independent confirmation, then more realistic simulation and externally justified workloads. |
| Capability-first proposals add no gain over identical-predictor direct search | The additional compiler-search hypothesis fails locally. | Retain any valid representation result; remove claims about better generative search. |
| Interval includes important benefit and harm | Inconclusive at current precision. | Report achievable precision; a fresh, justified larger study is possible, not repeated peeking. |
| Small stable gain has no plausible amortization | A genuine but practically negligible effect. | Report it without a breakthrough claim; stop application escalation unless a credible use case changes the economics. |
| Gains vanish under noise/cubic/history shifts | Transfer is narrower than the primary simulator. | Diagnose and label the limitation; additional training/measurements are new experiments, not retroactive validation. |

### 13.2 End-to-end value and break-even

Let `T_setup` include task-specific data generation, teacher construction, tuning, training and necessary verification development that is incremental relative to the baseline. Let `T_b` and `T_w` be measured per-task costs for the baseline and WCCT at a **comparable audited quality/guardrail level**, including characterization, adaptation, search and final audit. If `T_b > T_w`, a simple compute break-even estimate is

\[
N_{break}=T_{setup}/(T_b-T_w).
\]

If the denominator is nonpositive, there is no compute break-even under those conditions. Report separate experimental-measurement and compute accounting; do not assign invented laboratory dollar costs to cheap simulated trajectories. Retraining after distribution/interface changes reduces amortization. Baselines with reusable training assets also receive amortization credit. This calculation is an estimate from measured workloads, not a business case without actual users.

A practical improvement requires either better quality for a justifiable added cost or lower total cost at comparable quality. A tiny mean coverage gain cannot conceal more installation/shock failures because the primary score includes those guards and the report discloses their rates.

### 13.3 When to increase scale

Increase independent systems when uncertainty, not bias or a broken solver, limits the conclusion. Increase model/code capacity when teacher/student error decompositions show underfitting and development scaling curves improve relative to equally strengthened baselines. Increase design dimension only when the current finite menu is genuinely too easy and the added parameter has an engineering interpretation. Increase query density when numerical/interpolation error changes outcomes.

Do not use a possible foundation-model-scale benefit to explain away every local failure. Equally, a one-degree-of-freedom pilot without recurrent compositional effects cannot disprove all physical token representations. The locally testable claim is the one in Sections 2 and 8. Evidence of code reuse and improving held-out-template performance is the condition for exploring broader scales, not enthusiasm alone.

### 13.4 Evidence-triggered alternatives

If continuous effect prediction wins, pivot to a **continuous behavior-to-capability surrogate** with the same independent verifier. That preserves the engineering objective while abandoning an unsupported discrete requirement.

If the information barrier dominates, investigate **task-directed characterization** using the same probe budget and continuous/physical baselines. This is adjacent to established task-optimal exploration, not a newly invented active-learning principle. [R12]

If fixed-menu realization is easy but history is the real bottleneck, a later scar study must add explicit irreversible internal variables and compare with history-aware constitutive models. If a configuration-space boundary is essential to the actual application, create a separate geometric benchmark with verified clearance and force constraints. Neither pivot is authorized solely because it makes a token method look better.

A real-world claim ultimately requires measured physical specimens or an independently validated external dataset with compatible interface information. That is beyond the equipment-free core PoC. Until such evidence exists, label results **synthetic simulation results**, even when two numerical solvers agree.

---

## 14. Audit record, remaining risks, and final challenge tests

These are audits conducted while constructing the plan, not independent peer reviews and not claims that the coding agent's future implementation already passes them.

### 14.1 Scientific and prior-art audit

| Defect identified | Revision made | Residual risk / acceptance test |
|---|---|---|
| Behavioral tokens could simply rename predictive states or DeepSym. | Added PSR, DeepSym, skills-to-symbols and predictive-compression lineage; required adapted DeepSym and strong continuous baselines. | The precise combination may also exist. Revoke novelty if an equivalent implementation is found; retain a replication/application study. |
| Counterexample repair could be advertised as a new invention. | Included active automata and symbolic-geometric abstraction repair; narrowed the claim to conditional empirical value of the specific effect representation. | Algorithmic novelty remains unresolved. Equal-witness continuous/tree controls are mandatory. |
| Intervention consistency might overlook mechanism-transforming causal abstractions. | Added existing causal frameworks, including mechanism transformations, and restricted the physical interface explicitly. | No new general causal-abstraction theorem is asserted. |
| Task-aware quantization might be misrepresented as new. | Added system-identification quantization, goal-oriented control compression and causal rate–distortion sources. [R21, R22, R31] | Task loss alone cannot support novelty; downstream incremental benefit is required. |
| Arbitrary internal design edits are unidentifiable from boundary behavior. | Constructed an exact counterexample and restricted the core to known external attachments. | Finite probes still provide only approximate identification; test ambiguity and abstention. |
| “Three safe probes” could be credited with hidden threshold knowledge. | Added a costed release-capable trial and safe-only twin controls. | Real characterization may be destructive or unavailable; no physical safety claim. |
| A finite force envelope could be described as all possible futures. | Defined an exact query domain, finite scenarios and horizon. | Global reachability/topology remains untested. |

### 14.2 Experimental-validity audit

| Defect identified | Revision made | Residual risk / acceptance test |
|---|---|---|
| No-edit baseline makes the toy improvement look much larger than it is. | Actually evaluated a fixed-attachment reference; added a headroom gate before learning. | Multi-template headroom remains unmeasured. Do not train at scale until WP03 resolves it. |
| Grid contains duplicate physically identical designs. | Canonicalized attachments by `(k,c,b)` and required duplicate-aware budgets. | Candidate generation/hash tests must pass. |
| A high-capacity residual or soft mixture could carry the whole answer. | Bounded residual, no decoder bypass, rank/soft-code/serialized-bit controls. | End-to-end practical models may have different capacities; disclose both attribution and workflow results. |
| Witness labels could give only WCCT extra information. | Shared teacher queries, task losses and witness controls. | Tuning and preprocessing costs can still differ; full expenditure ledger required. |
| Teacher reconstruction could be confused with deployable prediction. | Separated oracle-teacher, base-prediction and observable-student errors. | Sparse probes may not predict useful codes on new templates. |
| Samples and random seeds could be counted as independent mechanisms. | Mechanism-level paired and crossed-replication analysis; fixed-sample confirmation. | Very few training repeats limit procedure-level claims. |
| A shared cache or final verifier could give free extra observations. | Separate logical cost, physical cache and blind audit; request ledger and process-isolation tests. | Evaluator bugs remain possible; same-policy and resume tests are blocking. |
| Near-boundary numerical error could change rankings. | Independent event-aware reference, extrema search, gray-zone refinement and denser-grid sensitivity. | Empirical convergence is not a formal interval proof or physical validation. |

### 14.3 Implementation and feasibility audit

| Defect identified | Revision made | Residual risk / acceptance test |
|---|---|---|
| Hard feasibility/argmax losses have no ordinary training gradient. | Specified differentiable soft-min/BCE and explicit straight-through semantics; hard labels retained for actual scoring. | Gradient estimator bias and temperature sensitivity need matched controls. |
| PCA normalization could make encoding/decoding equations inconsistent. | Fixed core coordinates, orthonormal basis and explicit residual-mean handling. | Round-trip and no-edit tests are mandatory. |
| Preload could appear to create free energy. | Included initial attachment energy and declared idealized installation work. | Actual installation is not modeled. |
| Current package documentation could be mistaken for M3 execution evidence. | CPU-first path, per-operation MPS checks, measured throughput and lock-before-run strategy. | Actual machine configuration/compatibility remains unknown. |
| A first-retrieved RIGID version could omit the later likelihood-sampling baseline. | Checked v6, added forest target-likelihood semantics, paper-style capacity, finite-menu sampling adaptation and zero-likelihood failure handling. | Paper/repository differences require pinned parity checks; no published-score reproduction is asserted. |
| A recent abstraction baseline might be excluded merely for using JAX. | S³ retained as an optional native CPU smooth-system diagnostic with isolated environment. | Its assumptions do not match the hybrid latch; no invalid direct score comparison. |
| Long reference runs or low memory could derail completion. | Profile by stage, stream signatures, serial ensembles, resume tests, and explicit precision/resource branches. | A full procedure-level confirmation may require substantial local runtime. Estimates are not measurements. |
| Resume could double-charge observations or change stochastic state. | Write-ahead logical/physical ledgers, atomic checkpoints and injected-interruption tests. | GPU bitwise determinism is not guaranteed; CPU parity path retained. |
| A coding agent could silently change the research question. | Defined authority boundaries, protocol hashes and amendments requiring fresh confirmation where necessary. | Human judgment remains necessary for genuinely new workload choices. |

### 14.4 Final unresolved risks

The selected vocabulary may have no advantage over a continuous surrogate; greedy partitions may be hard to predict; the best attachment may remain nearly universal; prototype interpolation may erase narrow useful bands; finite characterization may not identify hidden dynamics; uncertainty may be unreliable under topology shift; and a learned search policy may cost more than exhaustive low-dimensional simulation. All are live hypotheses, not implementation tasks assumed to succeed.

The literature search is bounded and does not establish absence of prior art. Some sources were inspected only through abstracts/metadata; a potentially related primary project webpage did not expose readable technical content. Recent preprints are not independent replications. The physical model is deliberately idealized, the numerical pilot is small, and no M3 Max training or controlled baseline comparison was performed during preparation.

The audit is considered complete for this handoff only in the limited sense that every material issue found above has a specified correction or blocking test. It is not a declaration of perfection. Execution must rerun these challenges against the actual code and results.

---
## 15. References and evidence register

All links below were consulted during preparation unless explicitly labeled an unresolved lead. “Inspected” states the actual level of access; it does not mean the experiment or proof was independently reproduced. Dates distinguish original publication/submission from later inspected versions. Repository snapshots must be pinned during execution; a moving branch URL is not a reproducibility lock.

**R01.** Michael L. Littman, Richard S. Sutton and Satinder Singh. *Predictive Representations of State*. NeurIPS 2001. [Paper](https://papers.nips.cc/paper_files/paper/2001/file/1e4d36177d71bbb3558e43af9577d70e-Paper.pdf). Inspected PDF text and first-page rendering. Establishes the closest predictive-state lineage; controlled-system assumptions do not establish few-probe attachment design. Supports the representation framing but limits novelty.

**R02.** Jan C. Willems. *The Behavioral Approach to Open and Interconnected Systems*. IEEE Control Systems 27(6), 46–99, 2007. DOI: [10.1109/MCS.2007.906923](https://doi.org/10.1109/MCS.2007.906923). [Institutional record](https://research.rug.nl/en/publications/the-behavioral-approach-to-open-and-interconnected-systems/). Inspected metadata/abstract, not a full proof audit. Informs behavior/interconnection semantics; no new interconnection theorem is claimed here.

**R03.** Norm Ferns, Prakash Panangaden and Doina Precup. *Metrics for Finite Markov Decision Processes*. UAI 2004; later arXiv deposit. [arXiv:1207.4114](https://arxiv.org/abs/1207.4114). Inspected abstract/metadata. Finite-MDP abstraction/error framework; resembles task-relative grouping, but its guarantees are not transferred to the hybrid simulator.

**R04.** Alper Ahmetoglu, M. Yunus Seker, Justus Piater, Erhan Oztop and Emre Ugur. *DeepSym: Deep Symbol Generation and Rule Learning for Planning from Unsupervised Robot Interaction*. JAIR 75, 709–745, 2022. DOI: [10.1613/jair.1.13754](https://doi.org/10.1613/jair.1.13754). [Inspected full text](https://arxiv.org/html/2012.02532v3). Methods, comparisons and appendices inspected. Action-grounded discrete effect learning is very close prior art; original comparisons do not settle superiority over every end-to-end continuous model.

**R05.** George Konidaris, Leslie Pack Kaelbling and Tomás Lozano-Pérez. *From Skills to Symbols: Learning Symbolic Representations for Abstract High-Level Planning*. JAIR 61, 215–289, 2018. DOI: [10.1613/jair.5575](https://doi.org/10.1613/jair.5575). [Journal page](https://www.jair.org/index.php/jair/article/view/11175). Inspected landing page/abstract. Relevant skill-dependent abstraction lineage, not a directly reproduced physical inverse-design baseline.

**R06.** Dana Angluin. *Learning Regular Sets from Queries and Counterexamples*. Information and Computation 75, 87–106, 1987. DOI: [10.1016/0890-5401(87)90052-6](https://doi.org/10.1016/0890-5401(87)90052-6). [Accessible paper](https://people.eecs.berkeley.edu/~dawnsong/teaching/s10/papers/angluin87.pdf). Inspected observation-table and algorithm text and rendered pages. Distinguishing experiments and counterexamples are established; membership/equivalence-oracle assumptions differ materially from noisy continuous mechanics.

**R07.** Wil Thomason and Hadas Kress-Gazit. *Counterexample-Guided Repair for Symbolic-Geometric Action Abstractions*. 2021 preprint; IEEE Transactions on Robotics 39(5), 4152–4165, 2023. DOI: [10.1109/TRO.2023.3294918](https://doi.org/10.1109/TRO.2023.3294918). [Inspected full text](https://arxiv.org/html/2105.06537v1). Algorithm and experimental sections inspected. Close counterexample/refinement prior art; repairing symbolic formulas is distinct from physically editing a mechanism.

**R08.** Paul K. Rubenstein, Sebastian Weichwald, Stephan Bongers, Joris M. Mooij, Dominik Janzing, Moritz Grosse-Wentrup and Bernhard Schölkopf. *Causal Consistency of Structural Equation Models*. UAI 2017. [arXiv:1707.00819](https://arxiv.org/html/1707.00819v1). Inspected formal definitions and explanatory examples. Informs intervention consistency; does not eliminate the need to specify physically meaningful low/high-level interventions.

**R09.** Eigil F. Rischel and Sebastian Weichwald. *Compositional Abstraction Error and a Category of Causal Models*. UAI/PMLR 161, 2021. [Proceedings](https://proceedings.mlr.press/v161/rischel21a.html); [inspected text](https://arxiv.org/html/2103.15758v2). Definitions and composition discussion inspected. Particularly relevant to ambiguous intervention implementations and accumulation of abstraction error; metric assumptions matter.

**R10.** Atticus Geiger et al. *Causal Abstraction: A Theoretical Foundation for Mechanistic Interpretability*. JMLR 26(83), 1–64, 2025. [Journal page](https://jmlr.org/papers/v26/23-0058.html). Inspected abstract/metadata, not the entire formal development. Explicitly includes mechanism transformations; limits claims that edit-aware abstraction semantics are unprecedented.

**R11.** Wei “Wayne” Chen, Rachel Sun, Doksoo Lee, Carlos M. Portela and Wei Chen. *Generative Inverse Design of Metamaterials with Functional Responses by Interpretable Learning*. First submitted 8 December 2023; inspected v2 and the later v6 of 30 October 2024. [Current inspected text](https://arxiv.org/html/2401.00003v6). Related journal DOI: [10.1002/aisy.202400611](https://doi.org/10.1002/aisy.202400611). Methods/results inspected, including forest target-likelihood sampling and its treatment of unachievable targets. Required functional-design competitor; published physical simulators are not all included in the repository.

**R12.** Andrew Wagenmaker, Max Simchowitz and Kevin Jamieson. *Task-Optimal Exploration in Linear Dynamical Systems*. ICML/PMLR 139, 10641–10652, 2021. [Proceedings](https://proceedings.mlr.press/v139/wagenmaker21a.html). Inspected abstract/metadata. Supports task-directed experiment design as established prior art; linear/LQR results are not guarantees for latches.

**R13.** Jacob R. Gardner et al. *Bayesian Optimization with Inequality Constraints*. ICML/PMLR 32(2), 937–945, 2014. [Proceedings](https://proceedings.mlr.press/v32/gardner14.html). Inspected abstract and bibliographic record. Establishes the constrained-BO comparator lineage; current executable details come from R26, not an assumed legacy API.

**R14.** Pierre-Jean Meyer and Dimos V. Dimarogonas. *Compositional Abstraction Refinement for Control Synthesis*. 2017 preprint. [arXiv:1712.06013](https://arxiv.org/html/1712.06013v1). Inspected assumptions, algorithm description and example results. Relevant to compositional feasibility abstractions under known-dynamics assumptions; no black-box inverse-design result is inferred.

**R15.** Jordan Peper et al. *S³: A Smooth Simulation Surrogate for Optimizing Discrete Abstractions of Dynamical Systems*. 16 August 2026 preprint. [arXiv:2608.15920v1](https://arxiv.org/html/2608.15920v1). Inspected formulation, assumptions, optimization and case-study sections. Contemporary discrete-abstraction optimization under known smooth dynamics; reported soundness is assumption-dependent and is not inherited by empirical WCCT. Repository inspected separately in R30.

**R16.** Binyao Guo, Zihan Lin and QiZhi He. *History-Aware Neural Operator: Robust Data-Driven Constitutive Modeling of Path-Dependent Materials*. 12 June 2025; revised 1 September 2025. [arXiv:2506.10352v2](https://arxiv.org/html/2506.10352v2). Inspected v1 methods/results and v2 metadata/full-text update. History-dependent elastoplasticity/damage modeling is relevant to deferred scar hypotheses, not a directly comparable attachment benchmark.

**R17.** Stephen Boyd and Leon O. Chua. *Fading Memory and the Problem of Approximating Nonlinear Operators with Volterra Series*. IEEE Transactions on Circuits and Systems CAS-32(11), 1150–1171, November 1985. [Author's paper page](https://web.stanford.edu/~boyd/papers/fading_volterra.html). Inspected author abstract and earlier technical-report metadata. Nonlinear-response expansion is established; fading-memory assumptions are consequential for persistent latch/history effects.

**R18.** Yanhao Yang and Ross L. Hatton. *Geometric Gait Optimization for Kinodynamic Systems Using a Lie Group Integrator*. Robotics: Science and Systems, 2025. [Paper](https://www.roboticsproceedings.org/rss21/p119.pdf). Inspected rendered first and fourth pages, including the abstract and geometric formulation. Close context for the holonomy idea; no gait optimization was executed here.

**R19.** Takuya Shiba. *The Compression Gap: Why Discrete Tokenization Limits Vision-Language-Action Model Scaling*. 3 April 2026 preprint. [arXiv:2604.03191v1](https://arxiv.org/html/2604.03191v1). Inspected methods/results and capacity comparisons, not only the abstract. Negative evidence about particular fixed action bottlenecks; not a universal impossibility theorem for discrete physical representations.

**R20.** Chunpu Xu et al. *M²Tok: Multi-head Multi-codebook Discrete Action Tokenization for Vision-Language-Action Models*. 16 September 2026 preprint. [arXiv:2609.18259v1](https://arxiv.org/html/2609.18259v1). Inspected architecture and experimental sections. Contrasting positive tokenization evidence in different tasks/capacity settings; published numbers are not head-to-head baselines for this PoC.

**R21.** Sarah Marzen and James P. Crutchfield. *Circumventing the Curse of Dimensionality in Prediction: Causal Rate-Distortion for Infinite-Order Markov Processes*. 9 December 2014 preprint. [arXiv:1412.2859v1](https://arxiv.org/html/1412.2859v1). Inspected causal-state/rate–distortion definitions and examples. Predictively sufficient lossy state grouping is established. Use arXiv submission/version metadata rather than the inconsistent regenerated date shown inside the HTML body.

**R22.** Koji Tsumura. *Optimal Quantization of Signals for System Identification*. 12 May 2009 preprint. [arXiv:0905.1862v1](https://arxiv.org/html/0905.1862v1). Inspected problem formulation and high/low-resolution discussion. Quantization optimized for parameter-estimation error, rather than reconstruction alone, predates this proposal; the analyzed SISO FIR/least-squares setting differs from hybrid capability design.

**R23.** Apple. *MacBook Pro (16-inch, Nov 2023)—Technical Specifications*. [Official specifications](https://support.apple.com/en-us/117736), consulted 14 October 2026. Inspected chip/memory configurations. Establishes that memory varies; does not identify the user's actual machine or measure workload performance.

**R24.** PyTorch contributors. *MPS backend* and *torch.mps* documentation. [Backend notes](https://docs.pytorch.org/docs/2.14/notes/mps.html); [MPS API](https://docs.pytorch.org/docs/2.14/mps.html), consulted 14 October 2026. Inspected availability/synchronization/memory APIs. Documentation compatibility is not execution evidence; resolved local version and each required training operation must be tested.

**R25.** SciPy contributors. *scipy.integrate.solve_ivp*. [Official reference](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html), consulted 14 October 2026. Inspected event, dense-output and tolerance descriptions. Event detection can miss multiple within-step zero crossings; step refinement and independent checks remain necessary.

**R26.** BoTorch contributors. *Constraints* and *Closed-loop batch, constrained BO in BoTorch* documentation. [Constraints](https://botorch.org/docs/constraints); [tutorial](https://botorch.org/docs/tutorials/closed_loop_botorch_only); [source/license](https://github.com/meta-pytorch/botorch), consulted 14 October 2026. Inspected current constrained/log-EI examples and MIT license text. Tutorial not executed here; use an actual resolved local version, not guessed APIs.

**R27.** DES Lab and AALpy contributors. *AALpy*. [Repository](https://github.com/DES-Lab/AALpy); [documentation](https://des-lab.github.io/AALpy/), consulted 14 October 2026. Inspected README, supported algorithm/model descriptions and MIT designation. Reusable finite-state learning implementation; suitable oracle assumptions must be supplied, not presumed for continuous physics.

**R28.** Alper Ahmetoglu and contributors. *DeepSym implementation*. [Repository](https://github.com/alper111/DeepSym); [inspected model source](https://raw.githubusercontent.com/alper111/DeepSym/master/models.py), consulted 14 October 2026. README, model/training code and block definitions inspected; MIT designation checked. Original robot dependencies are unnecessary for the adapted bottleneck. No published scores were reproduced; test adapted inference/training semantics explicitly.

**R29.** DIGITLab23. *RIGID implementation*. [Repository](https://github.com/DIGITLab23/RIGID); [inspected inverse-design source](https://raw.githubusercontent.com/DIGITLab23/RIGID/main/inverse.py), consulted 14 October 2026. README, MIT license designation and leaf/region inverse-design code inspected. Synthetic validation scripts are available; acoustic/optical response simulators are not included. Pin and test any modern scikit-learn compatibility changes.

**R30.** Trustworthy Engineered Autonomy Lab. *S³ abstraction optimization implementation*. [Repository](https://github.com/Trustworthy-Engineered-Autonomy-Lab/s3-abstraction-optimization), consulted 14 October 2026. README, native CPU/macOS environment instructions, examples and MIT designation inspected. Python/JAX dependencies need an isolated local check; the example was not executed on an M3 Max during this research.

**R31.** Li Wang, Chao Zhang, Samson Lasaulce, Lina Bariah and Merouane Debbah. *Goal-Oriented State Information Compression for Linear Dynamical System Control*. 14 July 2024 preprint. [arXiv:2407.10224v1](https://arxiv.org/html/2407.10224v1). Inspected formulation and resource-allocation discussion. Supports the established principle of task-dependent information allocation under linear-control assumptions; does not establish a physical edit alphabet.

**R32.** Maxime Méloux, Tiago Pimentel, François Portet and Maxime Peyrard. *Validating Causal Abstraction Metrics on Simulated Complex Systems*. 30 June 2026 preprint, arXiv July identifier. [arXiv:2607.00267v1](https://arxiv.org/html/2607.00267v1). Inspected formulation, controlled failure cases, results and limitations. Relevant warning that observational agreement need not establish abstraction validity. Its intervention access and supplied abstraction maps differ from this black-box design task; no general low-sample guarantee is adopted.

**R33.** Baldur Andrew Steingrimsson et al.; original assignee Imagars LLC. *Machine Learning to Accelerate Alloy Design*. US patent application publication **US20200257933A1**, 13 August 2020. [Publication text](https://patents.google.com/patent/US20200257933A1/en). Inspected bibliographic record and portions of technical disclosure as a broad design-surrogate prior-art screen. No claim chart, patentability conclusion, legal-status determination or freedom-to-operate opinion was performed.

### 15.1 Search scope, chronology, and stopping rationale

Research was conducted through 14 October 2026. Searches covered the original terms and their equivalents: predictive states; behavioral systems and interconnection; bisimulation; action-grounded symbols and affordances; skills-to-symbols; active automata and distinguishing queries; counterexample-guided abstraction repair; intervention-preserving causal abstraction and mechanism transformations; task-optimal experiment design; task-oriented quantization and predictive rate–distortion; functional metamaterial inverse design; constrained BO; history-aware constitutive operators; nonlinear Volterra response; and geometric gait/holonomy optimization.

Backward tracing connected recent papers to Angluin, predictive-state theory, behavioral systems, causal consistency and geometric mechanics. Forward tracing checked 2025–2026 work on discrete abstraction optimization, causal-abstraction metrics and action-token bottlenecks. Later RIGID and HANO versions were checked rather than assuming the first retrieved version was current. Repository inspection focused on actual model/leaf-inversion implementations, public examples, missing simulators, licenses and native CPU support. Hardware research used official Apple, PyTorch, SciPy and BoTorch documentation. Patent searching was a limited broad screen, not a comprehensive legal search.

The final targeted refinement searches used combinations such as “counterexample abstraction inverse design,” “task-oriented quantization system identification,” “behavioral equivalence design interventions,” and “capability tokens physical design.” These surfaced additional established quantization and causal-validation lines, which were integrated. They did not establish an executed prior method exactly matching this complete benchmark and architecture, but that observation is **not evidence that none exists**.

One potentially related primary project page, [“Behavioral evidence after system change,” Fueled Analytics](https://fueledanalytics.com/research/behavioral-evidence-after-system-change), was surfaced but returned no readable technical text through the available browser. Its content, date and exact overlap remain unresolved; no positive novelty claim relies on dismissing it. Searches also surfaced *Intervention Design for Effective Sim2Real Transfer* ([Mozifian et al., 2020](https://arxiv.org/abs/2012.02055)); its inspected abstract concerns invariance-inducing environment interventions, not physical attachment invention, so it was not promoted to a core comparator.

Additional targeted searches were producing the same main competing mechanisms rather than a new executable baseline family. Research therefore stopped after incorporating the last material distinctions, not after an arbitrary paper quota. Remaining blind spots include inaccessible/unindexed work, industrial systems, non-English literature, relevant unpublished implementations, detailed patent claims, and independent reproductions of recent preprints. A newly discovered close equivalent should change the novelty classification; it need not erase a useful empirical comparison.

### 15.2 Evidence-status summary

**Verified from inspected sources:** the prior-art relationships, source-specific assumptions, public repository contents/licensing indications, and documented software APIs described above. **Actually executed here:** the algebraic checks and small Linux/x86-64 latch pilot in Section 4. **Deductions/estimates:** identifiability counterexamples, finite-menu error reasoning, workload/memory formulas, and conditional cost projections. **Proposed but unexecuted:** the multi-template dataset, all trained methods, controlled baseline comparisons, confirmatory statistics, M3 Max compatibility/performance tests, and physical validation.

No pretrained assets or public benchmark numbers are required to reproduce the core experiment. No result in this plan demonstrates that WCCT is novel, superior, safe on hardware, or a breakthrough. The deliverable is an implementable way to determine which of those narrower claims, if any, earns evidence.

---

## 16. Handoff checklist

Start with WP00–WP03, not model training. Preserve the explicit restriction to external attachments and the cost of release-capable characterization. Keep protected outcomes inaccessible until the protocol is frozen. Run the strongest feasible baselines, including fixed design and cheap exhaustive search where applicable. Separate hard-code attribution from practical workflow performance and from capability-first proposal performance. Count actual information, residual capacity, preprocessing, training and verification costs. Retain failures and ambiguity. Report the narrowest conclusion supported by the completed checks.

The coding agent's first substantive deliverable should be a hardware/physics/observability/headroom report with concrete measurements and a continue/revise/stop decision. The final deliverable should be a reproducible comparison and an honest decision, not a predetermined positive story.
