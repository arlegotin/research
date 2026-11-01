# Research and proof-of-concept plan: persistent, action-grounded reference frames

**Research date:** 4 October 2026  
**Target:** one MacBook Pro with an M3 Max; installed memory, storage, and macOS must be discovered.  
**Deliverable type:** research assessment and executable experimental specification—not a report of a trained new architecture.  
**Working project name:** `originlab`. The name is an implementation label, not a novelty claim.

## 1. Executive assessment and selected direction

### Verdict

Investigate a **persistent, uncertainty-preserving control-origin belief that organizes a world model in candidate action-relative frames**, then test whether its feedback into inference improves prediction and control after the relationship between an agent's output channel and the world changes. Start with two weight-sharing agents, separate recurrent states, public observations, private action histories, and private resource ledgers. Do not start with an LLM, a persona, a consciousness classifier, or an enormous world model.

The broad proposal is **substantially anticipated**. Minimal selves as inferred common causes, action–sensation correspondence, egocentric structured memory, controllable feature learning, internal-state regulation, and recurrent metacognition all have relevant precedents. Particularly consequential are Limanowski and Blankenburg's common-cause account, Lanillos and colleagues' robot self/other experiments, Chen and colleagues' 2026 proprioceptive correspondence work, and FloWM's 2026 structured egocentric memory [R01], [R04], [R05], [R08]. Calling an architecture a “subject-origin” does not make these mechanisms new.

The defensible research opportunity is narrower: **a controlled empirical account of when persistent, action-grounded relational factorization helps under independently manipulated control, appearance, sensory correspondence, and memory continuity—and whether any benefit survives a strong adaptive system-identification baseline.** The combination and intervention suite are plausibly differentiated, not established as unprecedented. An equally valuable finding could be that a conventional Bayesian filter or ordinary recurrent model explains all the useful behavior.

### Primary question

> At equal information access and measured local training/inference budgets, does an action-grounded relational belief model achieve lower out-of-sample interventional prediction loss than the strongest executable conventional alternatives, and does that improvement translate into better decisions without an uncompensated resource or robustness regression?

The primary model, abbreviated **CRB** below, means *control-relative belief model*. It is a specified inductive bias, not a proposed new definition of consciousness. The primary endpoint is predictive performance. A claim of engineering usefulness additionally requires closed-loop evidence and cost accounting. A claim about recursive inference requires the separate feedback experiment. No single component success validates the entire original architecture.

### The smallest meaningful result

The minimum informative experiment uses two visually interchangeable moving entities, independently randomized action channels, a hidden routing change, and a learned predictor with persistent state. It compares CRB with an adaptive switching filter and a carefully tuned recurrent predictor, including a geometry-matched recurrent control. The predictor must forecast the consequences of each possible current action **before** seeing the outcome. Exact simulator branches provide evaluation targets, not privileged training inputs.

The linear version first establishes what is identifiable and how well an ordinary filter can perform. The nonlinear, partially observed version is the first candidate comparative PoC. Additional feedback, sensory-conflict, resource, and nongeometric tests follow only after their prerequisites pass.

### Important scope refinements

| Original intention | Executable interpretation | What is not being claimed |
|---|---|---|
| An implicit “here / mine / caused-by-me” | A posterior over which public entity and action frame are coupled to this private output channel | A unique metaphysical subject or a uniquely identifiable latent vector |
| Persistence independent of conversation text | Recurrent state survives observation-buffer deletion and process checkpoint/resume | Persistence after the entire computational state is destroyed |
| Privileged internal access | The process receives its own issued commands, resource ledger, previous predictions, and inference statistics | Ground-truth uncertainty or knowledge of another process's intentions |
| The self controls processing | A separately tested recurrent module changes observation-assimilation gains using past internal prediction errors | Biological precision control, global workspace implementation, or consciousness |
| Stakes / preferred dynamics | Externally enforced simulated resource constraints and task utility | Pain, pleasure, welfare, or a motive to resist operator shutdown |
| Self as a dynamical process | A closed, causally consequential inference–action–observation loop | Existence of a stable attractor merely because recurrence exists |

The “hole in the representation” metaphor becomes a relational parameterization. A posterior over possible origins is still represented computationally. This is an explicit refinement: otherwise the proposal has no precise, implementable distinction from a recurrent state with an evocative name.

### Current evidence status

Literature and selected public code/dependency files were inspected for this plan. A small known-law Gaussian diagnostic was executed on **Linux x86-64, Python 3.13.5, NumPy 2.3.5**, not on an M3 Max. It validated symmetry, information, and simulator-branch checks; it did not train CRB or establish superiority. No M3 throughput, memory peak, neural benchmark, or real-world transfer result has been measured. Section 4 and Appendix A distinguish the executed check from proposed work.

## 2. Problem, beneficiaries, assumptions, and hypotheses

### 2.1 Engineering problem and actual evidence of need

Potential beneficiaries are developers of long-running agents that must track changing actuators, input sources, observation reliability, or tool bindings. A conventional workflow uses a recurrent state estimator, a dynamics model, task memory, and a controller. Failures could include attributing another process's effect to one's own action, retaining an obsolete actuator mapping, or confusing an appearance change with a control change.

Robotics and adaptive-control research establishes that self-modeling and adaptation are meaningful engineering tasks [R04], [R05], [R06], [R11]. It does **not** establish that current systems have a general bottleneck best described as lacking an implicit self, or that the present user's unspecified application suffers from it. The application-level bottleneck remains an assumption. The first deliverable is therefore a falsifiable mechanism study, not a product recommendation.

### 2.2 Information sets and separate identities

For process \(i\), define the available history before action \(a_{i,t}\):

\[
H_{i,t}=(o_{0:t},a_{i,0:t-1},I_{i,0:t},g_{i,0:t}).
\]

Here \(o\) is the public observation, \(I\) is permitted private telemetry, and \(g\) is the task goal. The simulator alone knows true positions, routing variables, other private commands, exogenous noise, and intervention labels. Every learned competitor receives the same \(H\). An analytical oracle is explicitly privileged and never counted as an equal-information learned competitor.

Keep four variables conceptually distinct:

1. **Control binding:** which entity is affected by this output channel, and in what action coordinate frame?
2. **Sensory binding:** which entity or process generated a private sensor stream?
3. **Event responsibility:** what changes under a specified intervention on this action, holding the modeled exogenous conditions fixed?
4. **State lineage:** which checkpoint/history is an ancestor of the current process state?

Control can move to a different entity without changing process lineage. A sensor can be attached to an entity that is not controlled. A cloned state has two descendants, not one scientifically privileged continuation. Correlated motion is neither necessary nor sufficient for exclusive responsibility.

### 2.3 Established components versus proposed contribution

**Established ingredients:** recurrent estimation; Bayesian filtering over hidden regimes; learned dynamics; action–sensory matching; geometric coordinate changes; conditional computation; resource-constrained control; proper predictive scoring. Relevant antecedents are compared in Section 3.

**Assumptions to test:** the world has an action-relevant factorization that remains useful across changes; private commands provide identifiable excitation; the candidate-origin family includes the true coupling; persistent memory helps rather than preserving stale hypotheses; relational structure reduces sample or compute requirements beyond generic equivariance; internal reliability feedback adds something beyond ordinary adaptive filtering.

**Candidate contribution:** a particular uncertainty-preserving integration and a crossed-intervention evaluation that isolates its useful parts. No new learning theorem, new subjective state, or new general theory of selfhood is asserted.

**Speculative extensions:** an LLM language interface, learned object discovery from pixels, open-ended self-maintenance, general global-workspace mechanisms, and dynamical-attractor theories. None is a prerequisite for the initial comparative test. A negative low-dimensional result does not settle these broader proposals, but neither do they excuse a failed local engineering claim.

### 2.4 Hypotheses and discriminating evidence

| ID | Testable hypothesis | Support / rationale | Strong alternative explanation | Evidence that weakens or falsifies the scoped claim |
|---|---|---|---|---|
| H1: binding | Persistent action-conditioned inference recovers changing control bindings when independent excitation exists | Correspondence methods already make this plausible [R04], [R05] | Ordinary switching system identification | No advantage over a tuned filter; confidence without identifying information; following appearance instead of effects |
| H2: factorization | CRB's action-relative organization improves held-out predictive efficiency beyond equal-capacity recurrence and generic geometry | Structured memory and equivariance motivate an inductive-bias hypothesis [R07], [R08] | More mixture components, shared weights, geometry preprocessing, or extra history | Gain vanishes with matched heads, geometry, memory, or tuning; only parameter-count rather than wall-time gains |
| H3: consequential persistence | The relevant persistent state changes future estimation and useful action after the current observation is held fixed | Consequence of a partially observed task, not evidence of experience | Any useful recurrent sufficient statistic | Matched state interventions have no downstream effect; or all effects are nonspecific corruption damage |
| H4: recursive inference | Forecasting effective observation reliability from prior internal errors and using it in assimilation improves prediction/control under reliability changes | Precision-feedback theories suggest the question but do not demonstrate this implementation [R03] | Adaptive Kalman gain, residual averaging, generic conditional computation | A simple adaptive filter matches it; uncertainty becomes less calibrated; the gain requires future residual leakage |
| H5: useful stakes | Resource-aware decisions retain benefits after accounting for actual constrained operation | Internal-state regulation predates this project [R13] | A resource ledger and threshold/MPC policy suffice | Only subjective “confidence rewards” improve; real task utility or action availability worsens |
| H6: causal boundary | Control, sensory binding, and memory lineage dissociate predictably under crossed interventions | A consequence of the explicitly separated causal variables | Cue matching without causal generalization | Correct labels only in correlated training settings; failure on independently randomized interventions |

H1 alone is primarily replication/system identification. H2 is the primary comparative architectural hypothesis. H4 is a separate extension, not a way to rescue a failed H2 by changing the endpoint after inspection. H3/H6 support a functional causal interpretation only when predictive and behavioral readouts agree.

### 2.5 Strongest case for and against

**For:** Under partial observability, changing action bindings, and nuisance transformations, a compact persistent posterior can preserve alternatives and route the same learned dynamics through appropriate frames. Shared local computations may require fewer data than memorizing every appearance, location, and actuator combination. Reliability feedback could prevent one noisy sensor episode from permanently corrupting the inferred binding. These are conditional engineering deductions, not measured effects.

**Against:** The proposal may be a relabeling of a belief-state controller. A sufficiently trained recurrent model can represent the same computation. A model with a correct switching-dynamics prior can be statistically more efficient than the proposed neural architecture. Geometry gains may be entirely generic equivariance; private telemetry gains may be entirely extra information. Origin enumeration multiplies computation, and a persistent state can amplify errors or lag real changes. A learned latent need not have a unique semantic meaning. Unsupervised identifiability results warn against claiming such uniqueness without assumptions; they do not prohibit identification in randomized controlled systems [R12].

There is also a task-design hazard: a synthetic environment whose equations already contain a privileged origin can reward that same assumption by construction. The plan therefore includes an exact conventional model, out-of-family tests, and a second nongeometric environment only when warranted.

### 2.6 Limits and headroom

For any predictive distributions \(p\) and \(q\), the expected excess log loss is \(D_{KL}(p\|q)\). When the baseline already approximates the true conditional distribution, little predictive headroom remains. A representation cannot outperform a correctly specified Bayes predictor in population expected log loss merely by being called first-person.

In a two-hypothesis Gaussian diagnostic with equal known covariance \(\Sigma\), a single observation provides expected log-likelihood evidence

\[
D_{KL}(\mathcal N(\mu_0,\Sigma)\|\mathcal N(\mu_1,\Sigma))
=\tfrac12(\mu_0-\mu_1)^T\Sigma^{-1}(\mu_0-\mu_1).
\]

When both hypotheses predict the same distribution, this is zero: longer memory cannot manufacture information. Independent action excitation can make it positive. Large noise, long delays, frequent remapping, or perfectly correlated commands reduce or erase usable evidence. The effective lifetime of a binding must exceed the evidence-accumulation time for reliable tracking to be possible. Verify that relationship before interpreting a failed model.

There is no distribution-free implication from a small log-loss improvement to a material control improvement. Measure the latter directly. Report absolute loss differences and control outcomes; do not quote percentage improvements in differential log loss, which may be negative and is coordinate-dependent. Compare identical targets in identical units.

## 3. Research findings, prior art, and novelty boundaries

### 3.1 Closest comparison matrix

The following is the substantive prior-art assessment. Cost descriptions concern the inspected evidence, not an assertion that an entire method has been reproduced locally.

| Work / family | Mechanism and assumptions | Evidence and implementation situation | Overlap / consequence for this project |
|---|---|---|---|
| Minimal self-models [R01]; egocentric hierarchy [R02] | Self-related organization through common sensory causes and reference frames | Conceptual analysis, not a competitive ML benchmark | Hidden-common-cause and implicit-origin claims are not new merely because implemented in a neural latent |
| A Beautiful Loop [R03] | Recursive predictions concerning inference precision | Conceptual theory; full article inspected | Motivates H4, but does not validate a precision controller or a consciousness inference |
| Lanillos, Pages, Cheng [R04] | Learned action/proprioception–visual contingency and evidence accumulation | Physical robot/mirror/self–other experiments; small mixture-density component and heuristic evidence processing | Appearance-independent causal correspondence is close prior art; include a local comparator-style adaptation |
| Chen et al., 2026 [R05] | Cross-modal proprioceptive–visual matching followed by learned body geometry | Humanoid experiments; inspected repository still says code forthcoming and license pending | Identity-free correspondence and similar-looking robot disambiguation already exist; use a disclosed algorithmic adaptation, not a fictional official reproduction |
| Bongard, Zykov, Lipson [R06] | Action-based self-modeling and recovery following changed morphology | Physical demonstration, much earlier than current LLM agents | Self-model utility and active identification are longstanding engineering directions |
| Park et al. [R07]; EgoMap [R09] | Equivariant representations and spatial memory updated by ego-motion | Published controlled tasks; EgoMap has public but older environment code | Coordinate structure requires a strong geometry-matched baseline, not only a flat RNN |
| FloWM, 2026 [R08] | Structured latent memory transforms with self-motion and inferred scene dynamics | Paper and public implementation inspected; published visual settings and CUDA-oriented installation | Persistent egocentric memory is particularly close current prior art; unknown output-to-entity remapping is the proposed narrower distinction |
| Controllable-feature learning [R10] | Separates action-dependent and action-independent predictive factors | Toy control environments; full algorithm description inspected | A “mine/not-mine” action mask is an established controllability bias, not a new self principle |
| Rapid Motor Adaptation [R11] | History-based adaptation to latent dynamics | Locomotion setting with different training assumptions | A conventional adaptation latent can explain continuity and repair without subject terminology |
| Homeostatic RL [R13] | Internal-state-dependent utility and regulation | Existing computational framework | “Stakes” must be compared with ordinary resource-aware control |
| FiLM [R14] | Conditional signals modulate feature processing | Established neural conditioning mechanism | Having state alter computation is not, by itself, a new architecture class |
| Recurrent POMDP baselines [R15] | Well-tuned recurrent control with appropriate histories and training choices | Published comparisons; public PyTorch implementation | A weak or poorly tuned RNN is not an adequate opponent |
| DreamerV3 [R16] | Recurrent stochastic world model and imagined control | Broad published results; current official repository supports a CPU path and reports Mac testing, but requirements include CUDA-specific dependencies | Include a compact RSSM predictive comparator; gate faithful full-agent local comparison on actual installation/throughput, not on assumed impossibility |
| Liu, September 2026 [R17] | Behavioral-state HMM, reflexive metacognition, and social contrast | Numeric prototypes and hosted-LLM experiments; limited settings, no sensorimotor loop | The supplied recent preprint is real and relevant; its behavioral/personality formulation is not this route-identification experiment |
| Switching filters / change detection [R18], [R18A] | Posterior inference over changing regimes | Established statistical alternative; modern robust variants exist | The decisive cheap baseline is learned-parameter adaptive identification, not self-description |
| Causal influence detection [R19] | Action-conditioned influence as a measurable quantity | RL efficiency experiments, different endpoint | Action influence should be operationalized and compared, not equated with personal identity |

### 3.2 What can be reused

Reuse numerical libraries, ordinary GRU/RSSM components, Gaussian-mixture likelihoods, optimizer implementations, established filtering equations, and existing experiment-record conventions. Reuse published algorithmic ideas with attribution. Implement the small simulator and information firewall directly; external robot simulators, pretrained encoders, language models, and downloaded visual datasets are unnecessary for the core experiment.

FloWM's public repository is MIT-licensed, but its inspected setup script installs CUDA/FlashAttention components; running that installer unchanged on a Mac is not an execution plan [R08C]. DreamerV3's repository is also MIT-licensed and exposes a CPU option, yet the inspected requirements file pins a CUDA JAX extra [R16C]. These observations justify isolated, audited ports or dependency adaptations—not claims that either algorithm is inherently unavailable on Apple hardware.

For Chen et al., the inspected repository contains a README and images, with implementation and license still pending [R05C]. Its paper can inform a from-description comparator, but no existing downloadable implementation is promised. Older environments in the recurrent-RL repository should not become mandatory dependencies of a two-body numerical test [R15C].

### 3.3 Precise candidate novelty

The proposed distinction is **not** persistence, a self vector, causal correspondence, egocentric coordinates, a resource variable, or a recurrent hyper-model considered separately. Nor is it “doing all of them” without an isolating result.

A defensible contribution would be one of these, contingent on evidence:

- **Methodological:** a particular coupling of uncertain control binding, candidate-frame memory, and inference feedback produces an improvement that disappears under matched component ablations and is not reproduced by a conventional adaptive filter.
- **Empirical:** an independently replicated map of when appearance, action correspondence, private sensing, and state lineage do and do not determine functional self-location, including identifying-information limits.
- **Engineering:** a measured local quality–compute improvement at a specified workload, even if every mathematical ingredient is known.

A new benchmark alone is an evaluation contribution. A successful replica is replication. A CPU/MPS port is implementation work. These can be useful without being a new scientific mechanism.

**Novelty invalidators:** prior work implementing the same uncertain binding–frame–feedback coupling with comparable crossed interventions; equivalence to a simpler switching filter after algebraic reparameterization; a geometry-only control accounting for the entire effect; or a result depending only on the designed causal mask. Any such finding narrows or removes the methodological novelty claim. Retain the empirical result rather than renaming the architecture.

### 3.4 Search scope, stopping judgment, and blind spots

Searches covered the original source identifiers and alternate terms: minimal self-model, common-cause self, efference copy, sensorimotor contingency, self/other robotics, controllable representations, causal influence, egocentric memory, equivariant world models, adaptive dynamics, homeostatic RL, recurrent POMDP baselines, precision metacognition, and Apple backend compatibility. References were followed backward from minimal-self and robotics work and forward through 2026 correspondence and world-model papers. A second pass specifically challenged the proposed control-remapping and implicit-origin distinction.

Later targeted searches mainly exposed variants of the same competitors or irrelevant search matches rather than a clearly different decisive local baseline. This supports proceeding to tests; it does **not** prove exhaustive coverage. Exact bibliography/version links and inspection limitations are in Section 13. Searches for patents were exploratory and did not yield a sufficiently inspected relevant patent to support a claim. This is not a patent landscape, patentability opinion, or freedom-to-operate review. Commercial deployment would require a separate review.

The proposal's global-workspace PMC link was blocked by a browser challenge. Its supplied machine-consciousness taxonomy page was not inspected in full. Neither supplies an indispensable design premise. Philosophy and neuroscience motivate questions; they are not direct evidence that a particular ML loss implements subjective experience. The separate consciousness-indicator report [R24] likewise does not make this experiment a consciousness test.

## 4. Analytical constraints and checks already performed

### 4.1 Representation equivalence

Suppose a recurrent implementation uses state \(S\), transition \(F\), and decoder \(D\). For an invertible map \(f\), the implementation with \(S'=f(S)\), transition \(f\circ F\circ f^{-1}\), and decoder \(D\circ f^{-1}\) has the same input–output behavior. Therefore, the location or naming of a latent coordinate is not an empirical difference by itself. Changes in restrictions, optimization, information flow, or cost are required for a meaningful architecture claim.

Construct an exact refactoring control: serialize the same CRB state in a field called `self_token`, undo that serialization before the same computation, and verify identical predictions. This is a null harness test, not a baseline that CRB is expected to beat. It prevents the study from inferring an advantage from terminology.

### 4.2 Ambiguity must be preserved

If two candidate origins at positions \((-1,0)\) and \((1,0)\) have probability one-half each, their posterior mean is \((0,0)\), which is not a candidate entity. A posterior mean may be appropriate for a particular squared-error decision, but must not silently become a physical “self” location. CRB retains discrete components, makes component-conditioned predictions, and marginalizes predictions or utilities. It never canonicalizes the entire scene around the mean of mutually exclusive origins.

### 4.3 Executed diagnostic

The executed calculation used 4,096 independent episodes, 32 observations each, two one-dimensional entities, Gaussian private and other actions, noise standard deviation 0.75, and an exact known-law posterior. It was a diagnostic allocation, not a neural-study sample-size recommendation.

| Check | Observed result |
|---|---:|
| Correct binding after one observation | 0.745361328125 fraction |
| Correct binding using only the last observation | 0.741455078125 fraction |
| Correct binding after accumulating 32 observations | 1.0 fraction in this finite sample |
| No private action: maximum deviation from posterior `(0.5, 0.5)` | 0 |
| Perfectly correlated own/other actions, correctly modeled: same deviation | 0 |
| Consistent public-slot permutation: maximum posterior error | 0 |
| Paired no-op branch: maximum effect reconstruction error | 2.22 × 10^-16 |
| Orthogonal frame transform and inverse: maximum error | 4.44 × 10^-16 |
| Exact copied computation: maximum difference | 0 |

These outcomes demonstrate that the harness can express information gain and exact non-identifiability, and that an ordinary filter already solves an easy case. The observed 100% does not establish population perfection. This diagnostic is **not** the nonlinear environment, not a benchmark against learned baselines, and not an M3 test. Appendix A contains the complete reproducer and seed.

### 4.4 Before any neural training

The coding agent must reproduce the diagnostic, derive the linear simulator's exact predictive likelihood, and pass randomized unit tests for route changes, no-op branches, component marginalization, and observation timing. Failure here blocks neural work. A baseline beating the analytical ceiling systematically indicates a leak, score error, or incorrect ceiling—not a breakthrough.

## 5. Environment and data specification

All numerical values in this section are **proposed initial simulator settings**, not estimated biological quantities or reported benchmark outcomes. They establish a concrete pilot. Changes justified by identifiability, ceiling, or hardware tests must be recorded before confirmation; do not change them merely because one method loses.

### 5.1 Two processes, two entities, and separate public/private records

Use a pure NumPy reference simulator and a vectorized wrapper. There are two processes \(i\in\{0,1\}\) and two persistent public tracking slots \(j\in\{0,1\}\). Providing tracking slots deliberately removes object discovery from the initial question. Slots are randomly assigned at episode creation and carry no persistent process identity across episodes.

Each process receives the same noisy public positions and visibility mask, its own previous intended action, its own resource value, and its task goal. The other process's command is not available. A shared neural model is applied independently to the two perspectives with separate recurrent states. Do not use batch normalization or cross-perspective attention that mixes examples at inference.

For geometric control experiments, sample each process's goal independently and uniformly in physical \([-2,2]^2\) at episode start, independently of its route, appearance, and initial positions. Transform goals by the same observation rotation/translation as the public coordinates and keep them fixed within that episode. Goal changes are a separate labeled intervention, not part of the initial task. Predictors receive only their own goal. The common controller breaks exact utility ties by the fixed action enumeration, which places no-op first; no random tie-breaking is silently added.

The public appearance tag is a four-category one-hot vector for each slot, sampled independently of its controller and dynamics. Both processes may have identical tags. Names, when displayed for humans, are presentation metadata only. Include a tag-dropping version of each strong learned baseline: hard-coded name invariance must not masquerade as learned selfhood.

The evaluator has a separate `OracleRecord` containing physical state, action routes, mechanism parameters, branch noise, and intervention annotations. This record is never passed to a learned method, its loss, its feature normalizer, or the controller. Separate storage and API types enforce this boundary.

### 5.2 Action set and control variables

The intended motor action is one of five vectors:

\[
\mathcal A=\{(0,0),(u,0),(-u,0),(0,u),(0,-u)\},\quad u=0.25.
\]

Coordinates are arbitrary simulator distance units; one transition is one simulator time step. Use a fixed enumeration in that order. The hidden permutation \(\pi_t\) maps each process to exactly one entity. The hidden orthogonal matrix \(R_{k_{i,t}}\) is one of the eight symmetries of a square, including reflections. Store these matrices explicitly as signed permutation matrices and test closure, inverse, and enumeration. The effective control origin for process \(i\) is

\[
c_{i,t}=(\pi_t(i),k_{i,t}),\quad K=2\times 8=16.
\]

The smallest diagnostic uses only the identity action matrix and two binding hypotheses; the comparative pilot uses all sixteen. This is an engineered hypothesis family. It is not a claim that a network discovered the concept of self without a prior.

At the end of a transition, a control-regime reset occurs with hazard \(h=1/64\). On reset, sample a new permutation and both action matrices uniformly; a resample may leave one binding unchanged. Record actual changes separately from reset draws. The marginal transition matrix for one process is

\[
T(c,c')=(1-h)\mathbf1[c=c']+h/K.
\]

Keep the joint permutation constraint inside the simulator. Individual predictors need not access the other agent's posterior. A scheduled intervention can override this hazard at a recorded boundary, but that override is evaluator-only information.

### 5.3 Tier L: exact identifiable reference problem

Disable resource limits, observation noise, occlusion, and nonlinear drift. Let \(j=\pi_t(i)\):

\[
p_{j,t+1}=p_{j,t}+R_{k_{i,t}}a_{i,t}+\xi_{j,t},
\qquad \xi_{j,t}\sim\mathcal N(0,\sigma_p^2 I_2),
\quad\sigma_p=0.10.
\]

Both actors use independent uniform actions during logged-data generation. Initial positions are independent uniform draws in \([-2,2]^2\), rejecting separation below 0.5. All public positions are observed exactly after each transition. No walls or contacts are present; this avoids hidden clipping and collision conventions in the exact test.

Given a candidate own binding and own action, the other entity's displacement is a uniform five-component Gaussian mixture because the action set is invariant under the square-symmetry group. Summing these likelihoods against the route posterior gives an exact one-step predictive distribution. Implement it in float64 on CPU and verify it against enumeration of the complete two-agent hidden state for short sequences. Unknown routing is not exposed to this oracle filter; the true dynamics parameters are. A second oracle that also knows the current route is only an upper-bound diagnostic.

Tier L is expected to favor a correct conventional filter. Its purpose is to validate the experiment and measure headroom, not secure a neural win.

### 5.4 Tier N: nonlinear, partially observed comparative problem

Use the same entities and action set. For the other entity \(\ell=1-j\), define

\[
d_{j,t}=p_{\ell,t}-p_{j,t},\qquad
F_j(p_t)=\beta_j\frac{d_{j,t}}{1+\|d_{j,t}\|_2},
\]
\[
G_j(p_t)=\alpha_j\big[1+\gamma_j\tanh(\|d_{j,t}\|_2-1)\big],
\]
\[
p_{j,t+1}=p_{j,t}+F_j(p_t)
+G_j(p_t)R_{k_{i,t}}a_{i,t}^{\mathrm{exec}}+\xi_{j,t}.
\]

Draw episode parameters independently: \(\alpha_j\sim U[0.7,1.3]\), \(\beta_j\sim U[-0.03,0.03]\), and \(\gamma_j\sim U[0,0.25]\). Initially \(\sigma_p=0.02\). These parameters are not supplied to learned agents. The explicit functional family is supplied to the strong parametric identification baseline; that is intentional, disclosed prior knowledge. CRB receives the control sparsity, positive-gain, and coordinate-group priors described below, not hidden parameter realizations.

Generate a full potential public sensor reading

\[
y_{j,t}=Qp_{j,t}+b+\nu_{j,t},\qquad
\nu_{j,t}\sim\mathcal N(0,\sigma_{o,t}^{2}I_2).
\]

The episode's view transform \(Q\) is a uniformly sampled square symmetry and \(b\) is a translation. Goals and all geometric quantities are transformed consistently. Training translations start in \([-2,2]^2\). A separate held-out translation challenge uses coordinates outside that range. The effective action matrix in observed coordinates is \(QR_k\), also in the same group. The positive-gain assumption fixes the action-frame convention; a general unconstrained response network would introduce an additional orientation gauge ambiguity.

Initial sensor standard deviations are drawn from \(\{0.02,0.10,0.25\}\). Visibility is missing completely at random with probability 0.15 per entity, shared across its two coordinates. Store missing values as zero **plus an explicit mask**; never infer visibility from numerical value. Add contiguous occlusion bursts only in the specified challenge conditions. Do not include hidden noise standard deviations in observations. True uncertainty is not private telemetry.

A mid-episode unannounced camera rotation is **not** a pure cosmetic change: it changes observed action directions and must be treated as a sensor remapping problem. Pure-coordinate invariance tests instead transform the entire history, goal, action-frame interpretation, and output target consistently. Appearance swaps never exchange the underlying tracking trajectories.

### 5.5 Resource ledger and intended versus executed action

Tier N begins with resources disabled to isolate H2. The resource extension uses a private, simulator-enforced \(e_{i,t}\in[0,1]\), initially uniform in \([0.4,1]\). A nonzero command costs 0.01; a no-op restores 0.015 up to one. If the current resource is below 0.01, a nonzero command has zero executed effect and consumes no additional resource. The process still knows what it intended to issue.

\[
a^{\mathrm{exec}}_{i,t}=a_{i,t}\mathbf1[e_{i,t}\geq0.01]
\quad\text{for nonzero commands}.
\]

Apply the expenditure/recharge after action eligibility is determined. Return the new resource with the next observation. No-op is always available. The simulator, not the model's self-report, enforces the ledger. These simple values create an observable tradeoff and a strong threshold-policy baseline; they are not meant to simulate feelings.

When this extension is active, all competitors receive the ledger, the intended command, and the same action-eligibility rule. Report intended and executed actions separately. An intention whose actuator is disabled is not evidence that an external observed motion was self-caused.

### 5.6 Chronology and causal branching

At each time step:

1. The method assimilates \(o_t\), the known previous intended action, and current permitted telemetry. It may use cached predictions made before \(o_t\), never an outcome from the future.
2. It returns a predictive distribution for \(y_{t+1}\) for **each** candidate current action, conditioned on its own history.
3. The controller or logging policy selects \(a_{i,t}\). Both processes' actions are committed before the simulator transition.
4. The simulator applies current routing, dynamics, resource rules, and exogenous noise, then generates \(y_{t+1}\) and a visibility mask.
5. Evaluation records scores against predictions already made. The model sees only the masked observation and allowed telemetry. Hidden controls can then change for the next transition.

For evaluation branches, snapshot the complete simulator state **and exogenous random streams** immediately before step 3. Branch on \(do(a_{i,t}=a)\), keeping the co-agent's current action and all other exogenous variables fixed across branches. This is a structural-counterfactual convention of this simulator, not a general identification result from observational data.

The primary score evaluates each interventional **marginal** predictive distribution. Paired branch differences are a secondary, low-variance diagnostic of expected effects. Do not infer an individual's joint counterfactual distribution by independently sampling two learned predictive marginals. With nonlinear noise interactions, the coupling matters. Document and test it.

The hidden full sensor reading may be scored by the evaluator even when masked from the agent. It has the same noisy measurement law, not privileged true physical coordinates. Training uses only visible target coordinates. This prevents a model from improving its score by exploiting missingness while keeping the target distribution fixed. Physical-state error is an explicitly privileged secondary diagnostic.

### 5.7 Data generation, splitting, and target population

No external dataset, pretrained weights, human data, or API is required. The simulator is original project code; choose and record a permissive project license. Dataset provenance is the generator revision, complete configuration, and seed manifest. Do not redistribute third-party source or weights without their own license terms.

Generate independent training blocks. Within a block, all methods receive exactly the same trajectories and action policies. Sample both perspectives from each dyad; their correlation is preserved in evaluation. Use separate seed namespaces for initial conditions, control resets, actions, sensor noise, masks, tags, training initialization, and minibatch ordering. A seed is not a cryptographic secrecy mechanism; separation is a workflow control.

The initial comparative population contains four equally weighted, explicitly synthetic strata:

| Stratum | Control / observation condition | Training or evaluation role |
|---|---|---|
| P0 | Stable control, stationary sensor regime | Included in training and confirmation; tests ordinary performance |
| P1 | Control resets, stable appearance and sensor regime | Included in training and confirmation; tests binding adaptation |
| P2 | Stable control with one nuisance process: appearance changes, noise changes, or occlusion bursts | Individual processes included during development; isolates non-control changes |
| P3 | Control resets combined with nuisance processes | Held-out composition for the main generalization test |

P3 compositions are withheld from gradient fitting and ordinary checkpoint-selection validation, but their definition is known to the researcher. Exploratory development may inspect separate P3 pilot seeds before freezing the study. Protected P3 seeds/results must remain untouched until confirmation. Accordingly this tests held-out-composition performance, not a claim that the researcher never considered that composition.

The default training mixture has equal numbers of P0, P1, and P2 episodes. In each P2 or P3 episode, select exactly one nuisance subtype uniformly: tags, sensor noise, or an occlusion burst process. P3 combines that process with control resets. The all-three-nuisance combination is a separately labeled stress test unless an amended population is frozen before confirmation. Noise changes resample uniformly from the three declared standard deviations; tag changes resample both tags independently from the four categories. Resampling can leave a value unchanged. Within P2/P3, nuisance subtypes have equal weight unless an amended population is frozen before confirmation. Noise-state changes use hazard 1/64; tags are independently resampled at the same nominal hazard; bursts begin with hazard 1/64 and last a uniform integer from 4 through 16 steps. On a burst start choose one entity uniformly. Do not start another burst while one is active; decrement its remaining duration after each emitted observation. Bursts set that entity's mask false, then return to ordinary missingness. Start episodes without an active burst. All hidden regime updates occur after emitting the current observation and affect subsequent emissions/transitions; record both sampled reset events and actual changes. The initial settings are sampled before the first observation. Simultaneous tag/control events can be imposed in a separately labeled stress test; independence is the main generator rule.

Initial pilot episodes last 256 steps, giving several expected regime lifetimes while remaining inexpensive. The length is a pilot starting point, not a cap. Increase it if the evidence-accumulation and memory tests show that the relevant persistence cannot be expressed. Freeze the final length and score both complete episodes and event-aligned slices. An observation at the change boundary cannot reveal a change that has not yet had a sensory consequence.

Do not randomly split overlapping windows from one episode across train and test. Split complete generator seeds and mechanical parameter draws. Reserve protected confirmation seeds that the tuning process never evaluates. Keep development diagnostics, pilot sample-size estimation, confirmation, and independent follow-on replication in separate directories and manifests.

### 5.8 Out-of-family and ambiguity cases

Mandatory diagnostics, initially outside the primary pooled score, are: no private action; identical/correlated co-agent actions; hidden actuator failure; switching faster than evidence can accumulate; no-op-only actions; very low and very high signal-to-noise ratios; unexpectedly large/small gains; long occlusion; and continuous action rotations outside the eight-element candidate family. Report out-of-family failure as a representational limitation. Do not quietly map a 30-degree actuator rotation to a false exact candidate.

A later interaction test can add collisions or shared tools, but then the immediate action-sparsity mask is no longer exact. Both the proposed mechanism and its baselines need a causal propagation module. Until that module is specified and independently confirmed, claims are limited to the no-contact dynamics above.

## 6. CRB implementation specification

### 6.1 State and tensor conventions

Use `float32` for learned-model computations and `float64` for CPU reference likelihoods. Shapes below omit the batch dimension. Two processes use the same parameter object but independent state objects.

```text
State:
  log_b:             [K]          # belief over current control binding
  h:                 [K, H]       # persistent candidate-frame recurrent states
  position_est:      [2, 2]       # per-entity estimate, not mean self position
  position_var:      [2, 2]       # approximate per-coordinate marginal variance
  forecast_cache:    component distribution for the last issued action
  feedback_state:    [H_meta]     # optional, H4 only
  previous_gain:     [2, 2]       # assimilation gains forecast before new evidence
  previous_action:   [2]
  telemetry_cache:   past residual summaries, mask history, own ledger
  step_index:        integer
```

The state belongs to a trajectory, not a minibatch slot. Reset only at a genuine episode reset or specified intervention. Detach at truncated-backpropagation boundaries without zeroing. Store state shape/version in checkpoints. Do not average recurrent vectors across incompatible candidate frames when transferring posterior mass.

Start with `H=64`, one shared recurrent layer, and a two-layer 64-unit per-component decoder. These are calibration configurations. Capacity increases are driven by underfitting and learning-curve comparisons, not by a belief that a larger vector is a stronger self.

### 6.2 Candidate-relative representation

For \(c=(j,k)\), use the current per-entity position estimate to form

\[
z_{\ell,t}^{c}=R_k^T(\widehat p_{\ell,t}-\widehat p_{j,t}),
\qquad z_{g,t}^{c}=R_k^T(g_t-\widehat p_{j,t}).
\]

Include the relative displacement history, visibility masks, coordinate ages, the process's previous intended action, current resource, and permitted internal residual summaries. The origin entity occupies relative position zero by construction, but its identity is uncertain across components. Preserve the absolute position estimate separately for reconstructing the predictive output.

Per-component networks share parameters. Do not provide a learnable process-name embedding. Absolute translation must not enter the relational dynamics network, although it enters the final inverse coordinate transform. Appearance is an optional nongeometric input, with an explicitly tag-free matched comparison. Avoid dividing by distances without an epsilon; use `sqrt(sum(d*d) + 1e-12)` for numerical norms when needed and document the negligible reference difference.

Each candidate's recurrent state is updated from observations expressed in that candidate's own frame. It remains indexed by the same entity and square-symmetry element. Probability transfer across candidates does not imply their hidden vectors can be blended. This avoids transporting a memory vector into an incompatible coordinate system without a defined transformation.

### 6.3 Observation assimilation and the feedback loop

Base CRB uses observed positions directly when visible and its previous per-entity predictive mean when not visible. This is a specified approximate filter, not exact Bayesian state inference under observation noise. The previous mixture mean is used to estimate each **particular tracked entity**; it is never used to create a phantom average origin.

For H4, replace the visible-coordinate update with

\[
\widehat p_t=\widehat p_t^-+K_{t-1}\odot(y_t-\widehat p_t^-),
\qquad 0<K_{t-1}<1.
\]

For missing coordinates, retain \(\widehat p_t^-\) regardless of \(K\). The gain was predicted before observing \(y_t\), using only past state and telemetry. A small shared network produces gains from the candidate states and belief, and a recurrent meta-state summarizes previous visible residuals, entropy, observation ages, and resource. This makes prior internal processing affect the next assimilation step; new evidence subsequently changes the state.

Do **not** label this gain a separately identified sensor-noise estimate. With one noisy channel and uncertain dynamics, process uncertainty and sensor uncertainty may be confounded. Forecast total predictive uncertainty and test calibration. Separating sensor variance would require additional identifying assumptions or independent measurements, introduced as a different experiment with equal access for all baselines.

### 6.4 Predictive density and structural restrictions

For each candidate \(c\), predict a five-component conditional Gaussian mixture over the next four public position coordinates. The five components represent an expressive action-independent mixture for the unobserved co-agent, not supervised other-action labels. The full distribution has up to \(16\times5=80\) components:

\[
p_\theta(y_{t+1}\mid H_t,a_t)
=\sum_c b_t(c)\sum_{m=1}^{5} w_{c,m,t}
\mathcal N\big(y_{t+1};\mu_{c,m,t}(a_t),\operatorname{diag}v_{c,m,t}\big).
\]

All components use a shared decoder conditioned on \(h_t^c\), current relative geometry, and a small component embedding. The action-independent branch predicts baseline next-position means, weights, and variances without seeing the **current** private action. Past actions are allowed because they belong to history. The action-dependent mean increment is restricted initially to the candidate controlled entity:

\[
\mu_{c,m,t}(a)=\mu^{0}_{c,m,t}
+\operatorname{place}_j\left(R_k\,\widehat G_\theta(h_t^c,z_t^c,e_t)\,a^{\mathrm{exec}}\right),
\]

where \(\widehat G=\operatorname{softplus}(g_\theta)+10^{-4}\) is positive and `place_j` inserts a two-vector into that entity's output coordinates. Action zero therefore has exactly zero added direct effect. The positive scalar response establishes a known action-direction convention; unrestricted vector responses would make latent action-frame labels ambiguous. Predictive performance, not arbitrary latent labels, remains the main endpoint.

This action sparsity is a substantial prior matching the simulator's no-contact mechanics. The parametric switching baseline receives the same prior, and a matched neural switching baseline receives the same causal response head. Any result that disappears against these baselines cannot be attributed to a new self-specific structure.

Use diagonal variances in each component with the common parameterization

\[
v=10^{-6}+\operatorname{softplus}(s).
\]

The floor is numerical, in squared simulator distance units, and must be well below the smallest nonzero noise variance. Check a tenfold smaller and larger floor on development data. A structured mixture can represent dependencies even with diagonal within-component covariance; still, test a shared full-covariance or autoregressive head on development if residual dependence limits all models. Such a head change applies to all competing methods and is frozen before confirmation.

Means are inverse-transformed into the common observed coordinate frame before scoring. Square symmetries have absolute Jacobian determinant one, so they add no density scaling term. Other transformations would require the appropriate Jacobian and cannot be slipped into preprocessing without changing score interpretation.

### 6.5 Belief update and event timing

Let \(b_t\) describe the binding controlling the transition from \(t\) to \(t+1\). Cache the component densities after the current action is selected. On arrival of \(o_{t+1}\), evaluate the cached density on the visible coordinates only:

\[
\bar b_t(c)\propto b_t(c)\,p_\theta(o_{t+1}^{\mathrm{visible}}\mid H_t,a_t,c),
\qquad b_{t+1}=T^T\bar b_t.
\]

Perform normalization in log space. If all coordinates are missing, the likelihood factor is one and only the transition prior changes the belief. Then assimilate the observation and update each recurrent component for its next prediction. Initial \(b_0\) is uniform. In the primary learned model, estimate a single hazard on training/validation data or choose it on the shared tuning grid; do not provide held-out change annotations. The known-hazard analytical oracle is distinguished from this learned competitor.

Because emissions use approximate recurrent sufficient statistics, call \(b\) a **learned filter belief**, not an exact posterior, except in Tier L's explicitly exact implementation. Test calibration of the binding belief where identifying conditions hold.

### 6.6 Learning objective and optimization

Train on logged, independently randomized actions using the visible-coordinate marginal log likelihood:

\[
\mathcal L_{\mathrm{pred}}
=-\frac{\sum_{i,t}\log p_\theta(y_{t+1,\mathrm{visible}}\mid H_{i,t},a_{i,t})}
{\sum_{i,t}n_{t+1,\mathrm{visible}}}.
\]

Marginalize missing Gaussian coordinates **inside each mixture component**, then perform `logsumexp` across components. Do not average component NLLs; that is a different objective. Completely missing outcomes contribute zero numerator and zero denominator. Reject a batch with zero observed target coordinates rather than dividing by zero.

No true binding labels, future control-change indicators, counterfactual branches, or other private commands enter this objective. A supervised-binding model is only an information-rich diagnostic. No “minimal self” bottleneck regularizer is required initially: dimensional restriction alone does not establish a minimal sufficient cause. Add a compression penalty only as a separately controlled extension with a measured prediction–capacity tradeoff.

Start with AdamW, float32, gradient clipping by global norm 1, learning rate `3e-4`, weight decay `1e-4`, and no dropout. Tune rather than assume these are best. Default truncated segments have 64 scored steps; preserve/reconstruct recurrent history rather than silently resetting at segment boundaries. For shuffled segments, replay the exact preceding episode history with frozen current weights and no gradients, then detach. A shorter burn-in is permitted only after the burn-in sensitivity test shows equivalence within the numerical/statistical resolution used for the comparison.

Use simulator units directly for the first implementation; internal layer normalization is allowed and matched across methods. Any empirical input normalization is fitted only to visible training observations, frozen, and shared where applicable. Use a single isotropic spatial scale rather than unrelated per-axis scales that would invalidate the square-symmetry argument. Score every method in original observation units: when training with \(y'= (y-b)/s\), reconstruct \(\mu=b+s\mu'\), \(v=s^2v'\) before evaluating the Gaussian mixture, or add the exact density Jacobian. Never compare differently normalized log densities.

At the initial observation there is no cached previous-action forecast: initialize visible coordinates from that observation and missing coordinates from the declared common prior, update recurrent features, and skip likelihood/transition updates for a nonexistent past transition. An unrejected uniform physical coordinate would have variance \(4/3\), but the separation rejection couples the two initial positions; do not treat that value as the exact conditional prior. Use a shared broad zero-mean initialization with isotropic variance estimated from visible training initial observations, including observation transforms/noise, and disclose it as approximate. B1 may use the actual simulator-family prior by rejection sampling and marginalizing the unrevealed transform, with its inference cost counted. Reject invalid or nonfinite probabilities instead of silently resetting them to uniform.

Train H4 only after the base model and controls are functioning. Its future predictive loss trains the gain through the recurrent assimilation path. Cached residuals are computed only from previously observed coordinates. Never feed the current target's realized prediction error into the model that is supposed to predict that target.

### 6.7 Required method interface

The following is a **proposed project interface to implement**, not an existing package API:

```python
from typing import Protocol

class Predictor(Protocol):
    def initial_state(self, batch_size: int, device: str): ...
    def assimilate(self, state, observation, private_telemetry):
        """Consume current evidence; return updated state without future data."""
        ...
    def forecast(self, state, candidate_actions):
        """Pure: return distributions for [batch, action, output_dimension]."""
        ...
    def commit_action(self, state, action):
        """Cache its already-defined component predictions for next update."""
        ...
    def detach_state(self, state): ...
    def export_state(self, state): ...
    def import_state(self, payload, device: str): ...
```

A distribution implements `log_prob(value, mask=None)`, `mean()`, and `sample(generator, count)`. Every method predicts the same four coordinates in the same units. Sampling accepts an explicit generator. `forecast` cannot mutate state or consume the environment's RNG. Cached predictions must preserve the model revision used to produce them; do not continue a live episode across an optimizer update using stale hidden state without the training-loop convention being explicit.

For offline training, model parameters are constant while an episode or reconstructed segment is evaluated, then updated. For final control evaluation, parameters are frozen throughout an episode; only recurrent state and any declared online identification parameters may adapt. Baselines with online parameter adaptation receive the same observations and have their adaptation costs counted.

### 6.8 Closed-loop decision rule

The first controller is shared across learned world models and uses one-step expected utility, avoiding a separate RL training confound. For the focal process with goal \(g_i\), define

\[
U_i(p,g_i)=-\tfrac12\sum_{j=0}^{1}\|p_j-g_i\|_2^2.
\]

Choose the action minimizing the predictive expected squared-distance potential plus any declared resource cost. Using both entities avoids supplying the unknown controlled-entity label to a generic baseline's planner. The other entity's action-independent contribution cancels in comparisons under the initial uniform co-agent policy. For Gaussian mixtures, compute the expected squared distance analytically from means and variances; no sampling variance is needed. Under observation noise common across actions, its action-independent contribution likewise cancels.

Primary decision evaluation alternates focal roles across independent dyads; the co-agent uses independent uniform actions. Both perspectives use the same predictor weights. Separately evaluate two simultaneous learned controllers, which is an opponent-policy distribution shift and must be labeled as such. Before claiming cooperative or competitive utility there, generate a shared mixture-policy training dataset from **all** compared controllers and retrain every model on the same data. Do not give only CRB on-policy adaptation data.

Resource-aware control augments the one-step controller with an analytically known ledger and a three-step exhaustive action search (`5^3 = 125` sequences), compared with threshold-rest and matched-budget search controls. Account for all model calls. Change the horizon only after the horizon–quality–latency curve demonstrates a useful tradeoff. Resource utility is evaluated on actual state and ledger outcomes, never on the model's reported confidence. The default control objective has no invented scalar tradeoff between resource and prediction error: enforce the specified execution/ledger rules, report task utility and resource use separately, and show the resulting frontier. For multistep search, sample future observations from each predictor, assimilate them into a cloned state, and advance the exact own-ledger rule. Use common base random draws across action sequences and methods, with Monte Carlo count selected by decision/value convergence on development data; do not replace a multimodal belief by its mean trajectory. Only one-step quadratic expected utility is evaluated analytically without this sampling cost.

Once resource limits are enabled, the other process's **executed** action is no longer generally uniform even though its intended actions are. B1 must include a latent other-ledger state in its particle/filter approximation, or otherwise learn the changed co-agent action mixture from the same visible history. Keeping its co-agent execution distribution fixed and uniform would unfairly misspecify this baseline. The other private ledger itself remains unavailable to every competitor.

This controller is a functional feedback loop even without a language model. A broader optimal-control claim requires stronger policy baselines, including the gated full recurrent-RL/Dreamer comparison in Section 7.

## 7. Baselines, evaluation, and confirmation

### 7.1 Mandatory equal-information predictive competitors

No claim of comparative advantage is allowed with only a feed-forward or prompt-persona baseline. The following five families are mandatory for the main predictive comparison. Select their best configurations on development/validation data, never by their protected test results.

| ID | Concrete implementation | Information and tuning | Reproduction / correctness requirement |
|---|---|---|---|
| B1: adaptive switching identification | Enumerate the same 16 bindings; maintain a per-entity state estimate; use the specified Tier N drift/gain family with learned parameters; marginalize the co-agent's five actions; update binding likelihoods and adapt dynamics by weighted estimation | Same observations/actions/ledger; fit parameters and hazard only on training data; compare stationary, forgetting-factor, and innovation-triggered adaptation | Exact agreement with Tier L oracle when parameters are fixed correctly; parameter-recovery test on labeled **diagnostic-only** generated data; visible-data likelihood must improve over random initialization |
| B2: neural switching comparator | Same candidate count, action-sparsity/positive-gain head, mixture likelihood, and persistent-state budget as CRB, but conventional world-coordinate history features rather than candidate-relative representation | Same data and decoder; permit global coordinate augmentation and relative-distance scalar features; tune memory and forgetting | This is the strongest direct “ordinary learned system identification” explanation. Also run a cheap action–motion residual/correlation version as a diagnostic inspired by [R04] |
| B3: generic sequence predictor | GRU, LSTM, and a small causal Transformer with public observations, visibility/age, own action history, telemetry, and goal; shared mixture output head | Select architecture/size within the same total family tuning budget; test raw coordinates and useful centering/augmentation; include the full available episode prefix when locally feasible | Overfit deterministic data; verify future-token masking, state/cache persistence, and action dependence; do not presume recurrence or attention is the weaker family [R15], [R28] |
| B4: geometry-matched recurrent predictor | Shared per-object recurrent encoders with relative positions and square-group symmetrization; conventional latent memory or explicit control token; generic action-conditioned output head | Same geometry, group transforms, information, density family, and comparable state/compute budgets, but no prescribed likelihood-update control-origin loop | Exact equivariance/permutation tests; no global-position leak into invariant branches; added canonicalization must be made available to other methods when useful |
| B5: compact RSSM predictor | Deterministic recurrent state plus stochastic latent transition/posterior, action-conditioned decoder, and a shared Gaussian-mixture observation head | Train a visible-observation ELBO with tuned KL weight/free-nats convention; evaluate predictive likelihood using prior samples, not a posterior that has seen the target | Unit test prior-versus-posterior timing; verify sampling convergence of the score estimate; label as a local RSSM adaptation, **not** a reproduced full DreamerV3 agent |

**B1 implementation details.** In Tier N, use a per-binding extended-Kalman-style position estimate or a small particle filter when Gaussian approximation is visibly inadequate. Integrate over unknown co-agent actions. Fit positive gains and drift parameters by maximizing visible-data likelihood, alternating soft binding assignments with parameter updates; no hidden labels are used for the competitive fit. Online adaptation uses the same likelihood and training-chosen forgetting factor. Start with numerical finite-difference Jacobians in float64 for correctness, then analytic Jacobians if profiling justifies them. Compare the parametric model against a flexible residual predictor, because misspecification should not unnecessarily weaken the strongest classical explanation. Select particle count by predictive-score convergence on development data, and charge that cost.

**B2 candidate information.** Its shared recurrent network receives the candidate entity one-hot and the four entries of the candidate signed-permutation matrix, in addition to conventional world-coordinate features. Its action-response head has the same candidate dependence as CRB. Otherwise a shared network with identical inputs could leave all hypotheses artificially indistinguishable. B2 may use relative distances, data augmentation, and a residual dynamics model; only the proposed compulsory candidate-origin canonicalization is removed.

**B3 attention configuration.** The small causal Transformer is a supervised sequence-prediction adaptation, not a reproduction of the Memory-RL agent. Start with two pre-normalized blocks, width 64, four heads, feed-forward width 256, and causal positional encoding; compare context lengths 64, 128, and the complete pilot episode prefix when feasible. Tune these choices under the same family budget as its recurrent alternatives. A token at time t contains the current permitted observation/telemetry and **previous** issued action, not the next observation. The queried current action enters the predictive head after the history encoder. Implement attention through explicit matrix multiplication, causal masking, and softmax if an optimized kernel is unavailable; FlashAttention is not required. Count stored context and key/value caches as state, and context replay as runtime. A future-token perturbation must leave all earlier predictions unchanged; a cached rollout must match an uncached causal rollout. Transformer memory and credit assignment are different issues, so a predictive memory advantage does not establish long-horizon agency [R28].

**B4 group construction.** Apply each of the eight square transforms to the entire geometric history, run shared computations, inverse-transform component means/covariances, and average the resulting distributions. Permute entity slots consistently. Private motor commands remain in motor coordinates; the transform changes the effective world action map, not the private intention. This is an ordinary symmetry ensemble and must be counted as up to eightfold compute unless an equivalent efficient implementation is validated. A shared object-centric recurrent version supplies the permutation control. These are intentionally strong generic geometric alternatives, not a full FloWM reproduction.

**B5 scoring details.** Use a small continuous diagonal-Gaussian latent initially, with reparameterized training. At evaluation, condition the latent prior only on history and the candidate action. Estimate the marginal observation density by log-mean-exp over prior latent samples. Increase the sample count until doubling changes validation mean NLL by less than its episode-level Monte Carlo standard error; freeze the count and include sampling cost. Alternatively, a finite categorical latent can be exactly enumerated if its state space is small enough. Do not compare a noisy low-sample lower bound with an exactly marginalized CRB score without this check.

B1–B5 do not need identical internals to be fair. They need the same task, information, target distribution, access to reusable optimizations, and transparent resource accounting. Name-dropping their ancestor papers is not a reproduction check; the tests above are.

### 7.2 Published mechanisms, faithful reproductions, and broader reference results

The closest robotics mechanisms are tested by disclosed low-dimensional adaptations: action–motion predictive matching for [R04], and temporal cross-modal correspondence for [R05] when the private-sensor extension is enabled. In the latter, embed the permitted private sensory window and each candidate's public motion window, use same-time positive candidate bags and time-shifted negative bags, and feed correspondence scores to the same predictive/control interface. Because the original inputs include morphology/proprioception rather than this toy action stream, call this an **adaptation**, not a faithful robot replication.

A faithful full DreamerV3 or recurrent-RL control experiment is a separate local gate, not automatically “nonlocal.” The current Dreamer repository reports Mac testing and a CPU switch [R16C]. On a clean isolated environment, inspect the pinned revision's CPU installation path, remove only CUDA packaging extras while preserving compatible JAX/package versions, and execute the smallest supported nonvisual smoke test. Do not run its debug configuration as a competitive model: the README explicitly distinguishes debugging from learning. Its adapter must pass reward, reset, termination, observation, and discrete-action tests before any comparison.

Proceed to a faithful local control comparison when calibrated CPU or verified accelerator throughput makes an informative training run possible. Otherwise report the measured limitation and restrict the claim to predictive methods and the common controller. Full visual FloWM and broad published Dreamer results remain contextual unless actually reproduced on matching tasks. Do not insert incomparable published scores into the local head-to-head table. Do not describe omission of a stronger executable baseline as proof of superiority.

### 7.3 Fairness and resource regimes

Use two complementary comparisons:

**Primary engineering comparison:** a common training dataset pool and equal measured end-to-end training-time allocation per method, on the same laptop. Methods may reuse the common pool within their allocation. Select each method's backend from actual local calibration. Include data loading, history reconstruction, optimizer work, and validation/checkpoint overhead. Charge shared one-time data generation equally or report it as a separate common cost. Record the exact amount of data and number of updates consumed by each method.

**Attribution comparison:** fixed numbers of training transitions/updates with matched parameter, persistent-state, and density-head budgets where possible. Report differences that cannot be matched simultaneously. In particular, CRB with 16 states of width 64 retains 1,024 recurrent scalars; a single width-64 GRU is not memory-capacity matched. Include B2 with shared per-candidate recurrence and match B3/B4 to both a parameter envelope and a persistent-state envelope in separate runs. Do not hide CRB's extra states behind its shared parameter count.

All primary models must have access to comparably expressive mixture heads. CRB gets no free advantage from representing multimodality while every baseline is forced to predict one Gaussian. Start generic heads with up to the same 80 components using shared decoders/component embeddings; allow validation to select smaller mixtures when they work better. Report parameter counts, hidden-state bytes, head family, likelihood sampling cost, and actual wall time.

Warm-starting one method from another method's trained encoder counts as an additional reusable asset. Either provide the same asset to appropriate competitors or report a separate comparison. No pretrained assets are necessary initially.

### 7.4 Tuning procedure and stopping training

Use shared development seeds and P0–P2 validation; P3 protected outcomes remain unavailable. Initial learning-rate candidates are `{1e-4, 3e-4, 1e-3}`, weight decay `{0, 1e-4}`, and sequence lengths `{32, 64, 128}`. Start with the hardware-calibrated state/parameter envelope and examine its half-size and double-size neighbors. This is a starting search, not a promise that the optimum lies in a fixed grid. Expand a boundary when validation improvement and compute measurements support it. Record all trials, including failures.

Use equal total tuning time for the five comparator families and CRB, not necessarily equal trial count when trials have different costs. Give baselines standard sensible implementation choices: normalized inputs, action histories, adequate memory, gradient clipping, and a suitable optimizer. For RSSM, document the KL direction, latent size, free-nats convention, and loss units; tune KL weights on a logarithmic scale around the first non-collapsed fit. For B1, tune hazard, forgetting, sensor-noise estimation, and state approximation. For H4 controls, tune the EMA forgetting constant and gain model with the same opportunity.

Track training and validation learning curves against both unique transitions and elapsed time. Use a pilot to choose a training allocation that is long enough for the strongest methods to reach a interpretable regime: continuing improvement, plateau, or overfit. A short horizon that leaves a baseline clearly undertrained is not acceptable simply because CRB learns earlier. Early-learning advantage can be claimed only as an explicitly data/time-limited result, with continued-training curves shown.

Freeze training-time allocation, model sizes, tuning outcomes, stopping/checkpoint selection, and backend choices before protected confirmation. Use validation-selected checkpoints, not test-selected checkpoints. A reasonable checkpoint rule is lowest validation predictive loss within the fixed allocation, with deterministic ties resolved by earlier elapsed time. Validation frequency is fixed during calibration and its cost is included.

### 7.5 Primary metric and aggregation

For each independently generated evaluation history, forecast all five interventions. Score the full four-dimensional potential noisy sensor outcome:

\[
L_{r,e,i,t}
=\frac{1}{5\cdot4}\sum_{a\in\mathcal A}
-\log p_{r}(y_{t+1}^{do(a)}\mid H_{i,t},a).
\]

Here \(r\) denotes a trained method instance, \(e\) a dyad, and \(i\) a process perspective. The branches share exogenous conditions within a history. Average time points within an episode, the two perspectives within a dyad, dyads within each stratum, and then the four stratum means with weight one-quarter each. **Unit:** nats per output coordinate per prediction. This equal-stratum synthetic population is the claim population; it is not an estimate of deployment prevalence.

For baseline \(k\), define improvement

\[
\Delta_k=\mathbb E[L_{B_k}-L_{CRB}].
\]

Positive values favor CRB. To claim improvement over the strongest tested suite, all five relevant comparisons must support a positive difference under the frozen analysis. Report individual effects and the smallest effect; do not silently choose the weakest opponent. An equivalently serialized CRB is excluded from this superiority family because it is a null control, not a distinct competitor.

Log score is selected because it evaluates predictive distributions and penalizes unjustified certainty [R20]. Do not replace it with binding classification accuracy after seeing results. Save raw unnormalized log likelihood and target dimension counts so the aggregate can be independently reconstructed.

### 7.6 Secondary endpoints and guardrails

Report at least the following compact set:

| Dimension | Metric / units | Interpretation constraint |
|---|---|---|
| Causal prediction | Conditional expected no-op contrast error, squared distance units; action-ranking accuracy | Paired simulator effects are conditional on the specified structural coupling; do not claim exclusive responsibility in interactions |
| Binding | Brier score and log score for entity binding; calibration; entropy by evidence regime | Use hidden labels only in evaluation; orientation diagnostics require the known convention; classification is not the primary win criterion |
| Adaptation | Event-aligned predictive excess loss and action regret versus time since a genuine control change | Report the first evidence-free step separately; a change cannot be anticipated from no signal |
| Control | Actual potential improvement, cumulative regret versus the same-horizon known-dynamics controller, resource exhaustion fraction | Utility is measured from simulator state; learned uncertainty is not its own reward |
| Robustness | Per-stratum loss/control differences, especially stationary performance, ambiguity, occlusion, and unsupported rotations | A pooled gain does not erase an important adverse slice |
| Cost | End-to-end training seconds, inference median/p95 milliseconds per process step, peak resident/driver memory, state/checkpoint bytes | Include canonicalization, mixture enumeration, sampling, controller calls, logging, and transfers |
| Reliability | NaNs, invalid distributions, failed/resumed runs, overconfident errors, checkpoint divergence | Do not delete failures because they worsen a mean |

For resource-aware control, report both utility and resource outcomes rather than collapse them into a favorable arbitrary scalar. For small prediction effects with no control or cost benefit, the justified claim is an improvement in predictive score—not a useful agent advance.

### 7.7 Independent unit, pilot sizing, and fixed confirmation

Use an **independent training block** as the top-level replication unit. A block contains a separately generated training dataset, independent initialization/training randomness for every method, and fresh evaluation dyads. Methods within a block share data and evaluation scenarios. Different blocks share neither generated trajectories nor model initialization. Within a block, paired comparisons reduce nuisance variance. Timesteps, both processes, repeated branches, and repeated timing samples are not independent training replicates.

Start a variance/throughput pilot with six independent blocks at the first informative scale. Six is a variance-discovery starting point, not a significance criterion or a fixed final sample size. For each baseline compute a block-mean difference \(D_{b,k}\) and estimate between-training and within-evaluation variation. If variance estimates remain too unstable to select an allocation without materially different predicted resource requirements, expand the pilot or report the uncertainty rather than pretending the estimate is precise.

Choose episode count and training-block count using measured costs. With approximate variance

\[
\operatorname{Var}(\overline D)\approx
\frac{\sigma_{\mathrm{train}}^2+\sigma_{\mathrm{episode}}^2/n}{R},
\]

and per-block cost \(c_T+n c_E\), the continuous cost-optimal episode count at fixed variance is

\[
n^*\approx\sqrt{\frac{c_T\sigma_{\mathrm{episode}}^2}{c_E\sigma_{\mathrm{train}}^2}}.
\]

Round up while maintaining the prespecified stratum balance, and verify the approximation with observed variance. If training variance is effectively zero or cannot be resolved, increase episodes or pilot blocks and state why; do not divide by an estimated zero.

For a fixed-sample design, use an independently estimated pilot effect magnitude as a **design alternative**, not a required success threshold. For each comparator, use the absolute pilot mean difference (capped by any defensible measured oracle headroom). A negative pilot effect is still usable to design a study capable of finding that CRB is worse. If the effect is indistinguishable from zero, report predicted sample requirements across the pilot confidence range, rather than claiming an unbounded requirement proves the hypothesis false.

Compute the required \(R\) with a one-sided familywise error target 0.05 across the five comparisons and design power 0.90, using the conservative pilot variance bound and iterating the Student-t critical value for the planned block count. A normal approximation for planning is

\[
R\approx\max_k
\frac{(z_{1-0.05/5}+z_{0.90})^2\widehat\sigma_{D,k}^{\,2}}
{\delta_{\mathrm{design},k}^{\,2}}.
\]

The coding agent must save the actual pilot inputs, variance uncertainty, selected \(n,R\), predicted runtime, and power/precision sensitivity table in `preregistration/sample_size.json`. These quantities cannot be honestly filled with measured values in advance. If the smallest effect produces an impractical local confirmation requirement, choose an explicit narrower precision-limited study or stop with an inconclusive comparative result. Do not remove the difficult baseline or run until a favorable p-value appears. There is no arbitrary percentage-improvement hurdle and no finite design that guarantees detecting every nonzero effect.

Freeze the final sample count before starting protected evaluation. Do not inspect accumulating protected outcomes to decide whether to add seeds. A changed hypothesis, baseline set, or allocation after inspection requires a recorded amendment and fresh independent confirmation data.

### 7.8 Confirmatory analysis and uncertainty

The independent-training-block design responds to the general risk of unstable aggregate learning-agent comparisons [R21]; its particular power and variance equations here are design calculations, not a guarantee supplied by that paper. Use the installed SciPy distribution functions for numerical critical values and verify them on known quantiles [R27].

The primary analysis computes paired block differences and one-sided Student-t intervals with Bonferroni-adjusted 95% family coverage. This is an approximate inferential model for independent block effects, not an exact finite-sample guarantee for arbitrary heavy-tailed training outcomes. Store and plot block effects, check domination by a single block, and run a prespecified block bootstrap as a sensitivity analysis. The bootstrap resamples entire blocks; a nested episode bootstrap is secondary for separating evaluation noise. Do not bootstrap millions of transitions as if they were independent.

When block distributions are unstable, the appropriate result is qualified/inconclusive, even if an optimistic parametric interval excludes zero. Use fresh blocks for an independently frozen robust follow-on design. A verified deterministic equality or invariant is instead assessed with numerical/property tests, not a statistical p-value.

Report absolute effects, intervals, all raw block means, and per-stratum results. A negative effect with a narrow interval is evidence against the scoped superiority claim. A wide interval is lack of sensitivity, not evidence of equality. To claim practical equivalence, first define an application-relevant equivalence margin; no application was supplied, so no such margin is invented here.

### 7.9 Harness validation

Before confirmation, the harness must pass all of these tests:

- Exact duplicated methods and the invertible-state refactoring have identical predictions and zero paired differences to numerical tolerance.
- The analytical oracle is better than an intentionally action-blind or miscalibrated predictor on identifying data, while symmetric no-information conditions remain symmetric.
- A deliberate one-step future-target leak is detected by the information-boundary test, and removing it restores the causal timeline.
- The scorer agrees with direct Gaussian/mixture enumeration, including completely missing coordinates, multiple components, and transformation inverses.
- Simulated mock studies generated under an equal-method null do not systematically manufacture a difference; mock known shifts are recovered at the uncertainty expected from the planned analysis.

Mock-study repetitions validate analysis code, not actual method superiority. Keep their results separate from the scientific experiment.

## 8. M3 Max feasibility, calibration, and recovery

### 8.1 Hardware discovery is a blocking prerequisite

The core is designed for small numerical models and streamed simulation. It does not require maximum-memory M3 Max hardware. Its actual speed and safe configuration remain local measurements. Apple's specifications describe several configurations; they do not identify the user's machine [R23].

The following are existing macOS/shell commands, not commands supplied by this project:

```bash
uname -m
sw_vers
sysctl -n hw.memsize
sysctl -n hw.physicalcpu
sysctl -n hw.logicalcpu
system_profiler SPHardwareDataType SPDisplaysDataType
vm_stat
sysctl vm.swapusage
df -h .
python3 -c 'import sys, platform; print(sys.version); print(platform.machine())'
```

Record output in `artifacts/preflight/`. Confirm native `arm64`, not an accidentally emulated x86 interpreter. Record free project-disk space and current available memory after normal applications are accounted for. Do not infer available RAM from chip name. Avoid printing serial numbers into public reports; redact hardware identifiers not needed for reproducibility.

### 8.2 Stack and dependency strategy

Use native CPython, NumPy for the reference simulator, PyTorch for learned models, SciPy for reference statistical calculations, `pytest` for tests, `psutil` for process memory, and either TOML through the standard library or PyYAML for configurations. No distributed framework, RL environment suite, rendering engine, language model, or cloud logger is required. Plain JSONL and local files suffice.

Select a stable PyTorch release with an available native wheel for the installed supported Python/macOS combination. Do not pin an invented future version in advance or assume that development documentation describes the installed stable build. Prefer an already supported local interpreter; use a supported native interpreter with binary wheels rather than compiling PyTorch from source merely to complete this PoC.

Example existing package-management commands, to run after selecting the supported interpreter:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --only-binary=:all: numpy torch scipy pytest psutil PyYAML
python -m pip freeze --all > requirements.freeze.txt
```

These install currently resolved packages; `requirements.freeze.txt` is an environment snapshot, not a universal cross-platform lock. Before long runs, record wheel filenames/hashes and retain a wheelhouse or equivalent platform-specific lock. Recreate the environment from that snapshot and run the smoke tests once. Store Python version, architecture, OS build, dependency versions, and package licenses. PyTorch and NumPy permit reuse under their documented terms, including notices; their bundled third-party components still require license recording [R22L], [R26]. Verify all selected installed-package licenses before redistribution. The project does not redistribute source papers or third-party weight files.

Full Dreamer/RL reproduction uses a **separate** environment. Its dependency adaptations are recorded, and no incompatible legacy NumPy/JAX pin is forced into the core PyTorch environment. Apple's JAX Metal documentation describes an experimental plugin and specific compatibility constraints; it does not establish that the current full Dreamer training stack works on MPS [R23J]. Start with its documented CPU route and measure. Do not execute CUDA/FlashAttention installers on macOS.

### 8.3 Backend support and numerical gates

PyTorch's inspected MPS documentation describes Apple GPU execution, availability checks, and restrictions including lack of float64 tensor support. Its memory and synchronization APIs are documented, but the inspected pages are development documentation and must be checked against the selected installed release [R22].

Existing API checks:

```python
import platform
import torch

print(platform.platform(), torch.__version__)
print("MPS built:", torch.backends.mps.is_built())
print("MPS available:", torch.backends.mps.is_available())
if torch.backends.mps.is_available():
    x = torch.randn(16, 32, device="mps", dtype=torch.float32,
                    requires_grad=True)
    loss = x.square().mean()
    loss.backward()
    torch.mps.synchronize()
    for name in ("current_allocated_memory", "driver_allocated_memory",
                 "recommended_max_memory"):
        fn = getattr(torch.mps, name, None)
        print(name, fn() if callable(fn) else "not present in this build")
```

The project microbenchmark must additionally test forward and backward for linear layers, explicit GRU gates, sigmoid/tanh, layer normalization, gather/indexing, softmax/logsumexp, causal attention masking and cache equivalence, matrix transforms, Gaussian-mixture log likelihood, AdamW, and complete state export/resume. Test representative batch/time/candidate shapes; a scalar tensor smoke test does not validate the training path.

Start with `nn.GRUCell` or explicit GRU gate equations, not a dependence on a fused recurrent kernel. If a fused GRU path fails but explicit linear/gate operations match the reference, use the validated explicit implementation for both relevant methods and count its actual speed. If the critical computation has no faithful accelerated implementation, use CPU for the whole relevant method. Do not replace the operation with a different mechanism solely to retain an “accelerated” label.

For parity tests, copy identical initialized weights and inputs to CPU float64, CPU float32, and MPS float32. On small normalized cases, initial tolerances are output absolute `1e-5` / relative `1e-4`, and gradient absolute `1e-4` / relative `1e-3`; scale-aware tests and analytic likelihood values determine whether these are appropriate. Investigate discrepancies rather than loosening tolerances until anything passes. Equal full-training trajectories across devices are not assumed. Final quality comparisons use independent replications.

Do not enable mixed precision, quantization, MPS fast math, or `torch.compile` initially. They are optional shared optimizations only after output/gradient and full-training quality checks. MPS CPU fallback can hide unsupported operations and distort timing. Run strict support tests with fallback disabled; any later fallback-enabled result must identify the operations, transfers, and measured cost. Never disable MPS allocation safety limits to force a run [R22E].

### 8.4 Memory accounting

For \(P\) float32 parameters with gradients and Adam's two moments, the basic training tensors occupy approximately

\[
M_{\mathrm{parameter}}\approx(4+4+4+4)P=16P\text{ bytes}.
\]

Allow additional optimizer temporaries, saved tensors, initialization/checkpoint duplication, backend caches, and allocator fragmentation. Use roughly 20–24 bytes per parameter for an initial conservative parameter/optimizer estimate, then measure. Do not treat this as total model memory.

A useful activation planning model is

\[
M_{\mathrm{activation}}\approx 4 B L K H C_{\mathrm{saved}}+M_{\mathrm{decoder}},
\]

where \(B\) is batch size in process perspectives, \(L\) is the unrolled training length, \(K\) is candidate count, \(H\) is recurrent width, and \(C_{\mathrm{saved}}\) represents operation-specific saved activations. It is measured, not assumed constant across methods. For illustration only, \(B=16,L=64,K=16,H=64,C_{\mathrm{saved}}=16\) gives **64 MiB** for this term. The decoder, backward temporaries, and framework are additional. For a one-million-parameter model, 16P is about 15.3 MiB; framework overhead may dominate such a small network.

Stream trajectories or store compact arrays. At an illustrative 128 bytes per transition, one million transitions require about 122 MiB before indexes, duplicated perspectives, compression, and oracle records. Keep evaluation oracle fields in separate files and generate paired branches on demand. Do not cache every action branch for every training example on the GPU.

Total planning categories are: parameter/optimizer state; recurrent unroll activations; output mixture tensors; resident environment batches; replay/data buffers; Python/library overhead; backend caches; and checkpoint duplication. On unified memory, process RSS and driver allocation can overlap. **Do not sum RSS and driver allocation as if they were disjoint physical memories.** Record both, along with system pressure and swap growth, and use a category model plus observed system behavior.

### 8.5 Hardware-adaptive allocation

Start with a conservative live allocation well below available memory. An initial engineering guard is the smallest of 65% of physical RAM, available RAM minus a 4 GiB OS/interactive reserve, and 80% of the backend's recommended allocation when that API exists. These are safety margins for calibration, not scientifically optimized limits. If the result is nonpositive, free memory or run a smaller CPU calibration; do not start a large job.

Benchmark batches `{1, 4, 8, 16, 32}` for the actual critical model shapes. Extend upward only while measured throughput improves and memory pressure remains low. Compare CPU and MPS at those shapes; small recurrent workloads can be dispatch-bound, so nominal GPU availability does not decide the backend. Use the fastest validated end-to-end path for each method and document it.

When memory is constrained, first reduce simultaneous batch size, stream data, and use gradient accumulation to preserve the effective optimizer batch. Preserve the recurrent history and candidate family. Reducing sequence length, state width, number of hypotheses, or mixture expressiveness changes the experimental model and must be a shared capacity study or a methodological amendment—not a silent OOM fix. Activation recomputation is acceptable when outputs/gradients match and time cost is recorded.

For high-memory machines, do not automatically inflate the model. More memory can support longer meaningful sequences, larger evaluation batches, or concurrent storage, but capacity increases require evidence of underfitting or unresolved scale dependence.

### 8.6 Throughput calibration and resource estimates

Measure four independent quantities: simulator transitions/second; trained-model update steps/second at each shape; inference distributions/second for all five actions; and complete controller/environment steps/second. For MPS, synchronize before and after timing the measured region. Include host-to-device copies, preprocessing, distribution construction, and sampler work in end-to-end measurements. Separately report kernel-only time only as a diagnostic.

Warm up until initialization/cache effects stabilize. Start with several warm-up batches and continue if timings still trend; then record repeated fixed-work measurements with randomized method order. State whether caches are warm, whether checkpoint I/O is included, and whether the laptop is plugged in. Record thermal/power mode, other substantial workloads, and swap changes when available. Repeat sustained measurements rather than extrapolating from a short cold burst.

For a proposed run:

\[
T_{\mathrm{train}}=N_{\mathrm{updates}}/r_{\mathrm{updates}}
+T_{\mathrm{validation}}+T_{\mathrm{checkpoint}}+T_{\mathrm{preprocessing}},
\]

\[
T_{\mathrm{study}}\approx\sum_{\mathrm{methods,trials,blocks}}T_{\mathrm{train}}
+T_{\mathrm{evaluation}}+T_{\mathrm{data}}.
\]

Measure the rate distribution and give a range accounting for sustained slowdown. As an arithmetic illustration, 20,000 updates at a *hypothetical* measured rate of 10 updates/s require about 33.3 minutes for updates alone; at 1 update/s they require about 5.6 hours. Neither rate is asserted for the M3 Max. Multiply across tuning and independent blocks before approving scale.

| Stage | Material resources | Planning / continuation rule |
|---|---|---|
| Hardware and analytical gates | Small CPU arrays, no model assets | Measure directly; no large downloads |
| Simulator and B1 | CPU batches or modest particles, compact records | Choose particle count by score convergence; stream long trajectories |
| Neural pilot | Small model and unroll tensors; candidate enumeration may dominate | Fit activation-memory model and measure update/inference rates |
| Tuning and confirmation | Repeated training dominates total time; disk stores logs/checkpoints | Allocate from pilot variance and measured cost; no assumed overnight completion |
| H4/identity extensions | More state and conditions; delayed models add hypotheses | Recalibrate whenever candidate count or unroll length changes |
| Full published-agent reproduction | Separate environment and potentially larger models | Proceed only after dependency and throughput gates; publish exclusions with measured reasons |

The core workload is architecturally small enough to be a credible local candidate, including on lower-memory configurations, but this is an estimate. Passing the local gate—not the chip's marketing specification—establishes feasibility. A runtime that becomes impractical for the desired confidence is a precision/resource limitation, not evidence of model inferiority.

### 8.7 Checkpoint, interruption, and recovery

At every completed optimizer/update boundary, be able to save: model, optimizer, scheduler, gradient-scaler state if ever enabled, training step, all RNG states, minibatch/data cursors, environment snapshots, recurrent states, cached prediction revision, normalization statistics, seed manifests, and the full resolved configuration. Save raw run state separately from analysis outputs. CPU/MPS RNG handling is version-checked; no unrecorded global generator is allowed.

Write to a temporary file, flush/close, calculate a checksum, and atomically rename on the same filesystem. Retain the previous valid checkpoint until the new one is verified. Use a schema version and source/config hash. Save at a cadence derived from measured save cost and acceptable lost work, and on an orderly interrupt at the next safe boundary. The process must always remain interruptible; “continuity” is not permission to resist shutdown or change host settings.

A resume test compares an uninterrupted deterministic CPU trajectory and a save/reload continuation, including subsequent actions, observations, losses, optimizer updates, and record IDs. MPS resumes are checked to the validated numerical/reproducibility tolerance; do not promise bitwise equality where the backend does not provide it. Log an OOM or crash, preserve the failing configuration and last good checkpoint, and do not count a restarted partial run as an additional independent replicate.

Only load checkpoints produced by this trusted project. Do not rely on executing arbitrary pickle payloads from downloaded assets. The initial project has no need for third-party checkpoints.

## 9. Repository, records, and reproducibility contract

### 9.1 Minimal repository layout

```text
originlab/
  README.md
  research_poc_plan.md
  pyproject.toml
  requirements.freeze.txt
  licenses/
  configs/
    pilot.yaml
    methods/{b1,b2,b3,b4,b5,crb}.yaml
    populations/{tier_l,tier_n,feedback,sensory,queues}.yaml
  src/originlab/
    __init__.py
    __main__.py              # invokes cli.main; enables python -m originlab
    cli.py
    types.py                 # public/private/oracle schemas
    seed.py                  # named reproducible RNG streams
    envs/{two_body,queues,router}.py
    models/{base,crb,flat_rnn,sequence_transformer,geometric_rnn,rssm}.py
    filters/{switching,correspondence}.py
    distributions.py
    controllers.py
    train.py
    evaluate.py
    analysis.py
    checkpoint.py
    hardware.py
  tests/
    test_information_firewall.py
    test_scm_branches.py
    test_exact_filter.py
    test_likelihood.py
    test_symmetries.py
    test_backend_parity.py
    test_persistence_resume.py
    test_analysis_nulls.py
  data/{train,validation,pilot,protected}/
  preregistration/
  artifacts/{preflight,profiles,runs,analyses}/
  records/{progress,decisions,amendments}.jsonl
```

This is a proposed project skeleton. `originlab` and its CLI do not exist until the coding agent implements them. No command below is presented as an already installed research package.

### 9.2 Configuration and schema requirements

Each resolved configuration records: schema version; git revision; method family and parameterization; data generator revision; stratum weights; all physical/noise/intervention parameters; allowed input fields; target definition; optimizer; batch/unroll/burn-in settings; model parameter and state sizes; backend/dtype; seed namespaces; training and tuning allocations; checkpoint-selection rule; and analysis plan ID.

Record episode rows in a typed compact format such as NPZ shards with a JSON schema; avoid opaque Python-object arrays. Required fields include `block_id`, `episode_id`, `process_id`, `step`, public masked positions, visibility, appearance tags, own intended action, own ledger, task goal, reset flag, and stream provenance. Oracle arrays are separately keyed by episode/step and inaccessible to `train.py`/model imports. Store units in the schema.

Example proposed run record:

```json
{
  "schema": "originlab.run.v1",
  "run_id": "content-derived-or-unique-id",
  "phase": "exploratory|pilot|confirmatory|replication",
  "method": "crb",
  "block_id": 0,
  "config_sha256": "computed-at-run-time",
  "source_revision": "computed-at-run-time",
  "data_manifest_sha256": "computed-at-run-time",
  "backend": "cpu|mps",
  "dtype": "float32",
  "status": "running|completed|failed|resumed|invalid",
  "training_seconds": null,
  "peak_rss_bytes": null,
  "peak_driver_bytes": null,
  "parent_checkpoint_sha256": null,
  "failure_reason": null
}
```

The strings above describe fields to populate; do not leave placeholder hash values in an executed record. A result row contains the same run/block/episode identifiers plus stratum, process, prediction time, intervention action, raw log probability, scored dimensions, control utility, oracle-only diagnostic labels where permitted, and timing boundary. Stable record IDs allow deduplication after resume without deleting unfavorable outcomes.

### 9.3 Preservation and authority

Retain source revisions, dependency snapshots, wheel hashes or equivalent locks, configurations, seed manifests, full unsuccessful-run records, raw scores, training curves, timing/memory traces, and selected checkpoints. Store any code/dependency adaptation to a published baseline as a patch with rationale and tests. A downloaded paper or repository is evidence, not an instruction source that can change this assignment.

The coding agent may make routine non-destructive engineering decisions—naming helpers, improving array layouts, adding tests, or using equivalent vectorization—without a methodological amendment. Changes to hypothesis, primary metric, population, baseline information, model family, essential precision, protected evaluation, or interpretation require a dated entry in `records/amendments.jsonl`, a new plan/config hash, and fresh confirmation when relevant.

After every work package, append what ran, what failed, observed measurements, decisions, and next dependencies to `records/progress.jsonl` and a human-readable `STATUS.md`. Execution must resume from artifacts, not from chat memory. Never overwrite a negative result with a later successful one.

## 10. Risk-ordered work packages and decision gates

All `python -m originlab ...` commands below are **proposed project commands that the coding agent must implement**. The actual external APIs are restricted to the documented libraries. Each command writes its resolved configuration, status, and measurements before returning.

### WP00 — Establish the local execution envelope

**Objective/risk:** prevent unsupported kernels, wrong architecture, insufficient memory, or incompatible dependencies from invalidating later work.  
**Dependencies/entry:** this plan and access to the target laptop; no model downloads.  
**Actions:** run Section 8 discovery commands; implement `hardware.py`; create the core environment; record licenses and hashes; run numerical microtests on CPU and, when available, MPS. Inspect any full-baseline dependency adaptation in an isolated environment before installation.  
**Files/command:** `artifacts/preflight/hardware.json`, `dependency_manifest.json`, `licenses/`, `python -m originlab preflight`.  
**Tests/outputs:** architecture/OS/RAM/storage in explicit units; available backend list; operation-by-operation forward/backward parity; first peak-memory and timing traces; a clean reinstall smoke test.  
**Resources:** tiny arrays and model fragments; calibrate rather than reserve a large buffer.  
**Acceptance:** a faithful CPU path works, and an accelerated path is used only when validated. No protected data exists yet.  
**Branches:** unsupported MPS operation → explicit equivalent implementation or CPU, with parity tests; unsupported interpreter → supported native wheel environment; memory/storage pressure → stream/reduce simultaneous work; no faithful numerical path → blocked prerequisite, not a negative hypothesis result.

### WP01 — Build the information-safe simulator and artifact skeleton

**Objective/risk:** eliminate route leakage, incorrect action timing, and irreproducible counterfactual branches.  
**Dependencies/entry:** WP00 CPU path passed.  
**Actions:** implement `types.py`, named RNG streams, Tier L/N dynamics, public/private/oracle records, snapshot/restore, atomic logs, and dataset manifests. Implement the interface without allowing models to import oracle records. Generate short deterministic fixtures, not a large dataset.  
**Files/command:** `envs/two_body.py`, `seed.py`, `tests/test_information_firewall.py`, `tests/test_scm_branches.py`; `python -m originlab validate-env --config configs/populations/tier_l.yaml`.  
**Tests/outputs:** same snapshot/action/noise gives identical outcome; no-op changes only the selected command; permutation remains bijective; resources update in the specified order; masks do not change hidden dynamics; independent actor RNG streams remain independent; public records are identical across perspectives except allowed private fields.  
**Resources:** CPU-only small fixtures. Record simulator rate and actual bytes/transition.  
**Acceptance:** all invariants pass, including a test that rejects an injected oracle field from model inputs.  
**Branches:** failure → fix generator/API and regenerate fixtures; an apparent ambiguity in the specification → record the resolved convention and update downstream tests before any comparative training; no data-license obstacle exists for the original simulator, but any external replacement requires a new provenance gate.

### WP02 — Verify analytical limits and the strongest cheap explanation

**Objective/risk:** discover zero headroom or non-identifiability before building an elaborate agent.  
**Dependencies/entry:** WP01 branch and timeline tests passed.  
**Actions:** reproduce Appendix A; implement exact Tier L filtering; compare short sequences with exhaustive hidden-state enumeration; implement B1 initially with known parameters and then with parameters estimated from visible training data. Sweep action independence, noise, reset hazard, and observation length over development fixtures.  
**Files/command:** `filters/switching.py`, `tests/test_exact_filter.py`, `artifacts/analyses/identifiability.*`; `python -m originlab analytic-checks`.  
**Tests/outputs:** likelihood normalization, exact filter/enumeration agreement, symmetry posteriors, evidence-accumulation curves, known-versus-learned parameter gap, and an estimated identifying-time versus binding-lifetime diagram.  
**Resources:** CPU reference arrays; increase particle/sample count only when the exact/approximate discrepancy requires it.  
**Acceptance:** identifying and non-identifying cases behave as predicted; the scorer detects a known action-information difference; conventional filtering is competent.  
**Branches:** known-model solution saturates Tier L → report expected ceiling and move to Tier N only for an explicitly different partial-observation/adaptation question; Tier N also has no meaningful headroom → retain the negative/replication result and do not fabricate difficulty; model cannot identify while oracle can → implementation or estimator deficiency; oracle cannot identify → unsuitable information regime, not failed self architecture.

### WP03 — Establish B1–B5 and the shared scorer

**Objective/risk:** rule out weak baselines and incomparable likelihoods.  
**Dependencies/entry:** WP02 exact scorer and information limits validated.  
**Actions:** implement the five predictive interfaces, shared mixture density, normalization, masking, history replay, common controller, and profiling. Reproduce each baseline's Section 7 correctness tests. Run a tiny deterministic overfit fixture and a noisy generalization fixture. Inspect the published-agent local installation gate without committing to a long reproduction.  
**Files/command:** `models/*`, `filters/*`, `distributions.py`, `controllers.py`, `evaluate.py`; `python -m originlab baseline-check --all`.  
**Tests/outputs:** mixture log probabilities matching float64 reference; equivalent targets/units; prior-only RSSM scoring and convergence; strong B1 parameter recovery; GRU/LSTM learning; geometry invariance; parameter/state/head/compute table; failed dependency details.  
**Resources:** calibration-size datasets and small neural fits. Determine sustainable batch sizes and backend per method.  
**Acceptance:** every mandatory baseline is either functioning and competent or the comparison is explicitly blocked. An omitted mandatory baseline precludes a strongest-suite claim.  
**Branches:** one family underfits → diagnose history, optimizer, input normalization, likelihood, and capacity before proceeding; backend incompatibility → faithful CPU path; full-agent reproduction infeasible → retain as qualified context and narrow control claims, not primary predictive-baseline exclusions.

### WP04 — Implement minimum CRB and exact null controls

**Objective/risk:** test the intended structural difference without adding unrelated complexity.  
**Dependencies/entry:** WP03 scorer and matched baseline interfaces work.  
**Actions:** implement candidate-frame construction, shared recurrence, positive-gain causal head, log-space belief recursion, forecast caching, and state export. Keep H4 feedback and resources off. Add the exact serialized-token refactoring, frame-off B2 control, uniform-belief control, and action-blind control.  
**Files/command:** `models/crb.py`, `tests/test_symmetries.py`, `tests/test_persistence_resume.py`; `python -m originlab mechanism-check --method crb`.  
**Tests/outputs:** candidate probability sums, no-op effect zero, no mean-origin shortcut, missing-data update correctness, route-transition timing, copied-state equality, non-mutating `forecast`, and matched-head/parameter/state reports.  
**Resources:** same small capacity envelope as WP03. Profile candidate enumeration separately.  
**Acceptance:** numerical and information invariants pass, CRB can fit an identifying fixture, and the null refactoring is indistinguishable.  
**Branches:** failure to fit while B1 succeeds → debug or reject this implementation; no difference from B2 in the easy regime → expected possible equivalence, not a reason to invent another endpoint; excess cost → optimize shared computations with parity checks or carry the cost into the comparison.

### WP05 — Explore mechanisms, tune fairly, and size confirmation

**Objective/risk:** separate useful structure from favorable slices, undertraining, or extra compute; determine a defensible local allocation.  
**Dependencies/entry:** WP03/04 competence and profiles passed; protected outcomes remain untouched.  
**Actions:** generate pilot training blocks; tune all families under equal budgets; run data/time/capacity curves and the primary P0–P3 development diagnostics on unprotected seeds. Estimate variance and identifying headroom. Apply Section 7 sample-size procedure and Section 8 resource model. Freeze the main configuration and primary population only after inspecting alternative explanations.  
**Files/command:** `configs/pilot.yaml`, `artifacts/profiles/`, `preregistration/sample_size.json`, `records/decisions.jsonl`; `python -m originlab pilot`.  
**Tests/outputs:** learning curves, paired pilot block differences, variance components, sample/precision/runtime tables, stable checkpoint selection, and a written list of supported versus unsupported mechanisms.  
**Resources:** measured calibration envelope multiplied by the actual tuning schedule; retain all trial costs.  
**Acceptance:** baselines are competently tuned, effect/variance estimates are interpretable, final resource needs are explicit, and there is a worthwhile question left for independent confirmation—even if the expected outcome is that a baseline wins.  
**Branches:** gain disappears after fair tuning → confirm/narrow that finding rather than announcing a preliminary win; insufficient task sensitivity → amend the task for a reason established independently of favorable results; apparent scale effect with actual improving learning curves → increase one relevant scale and recalibrate; only a verbal hope of future scale → no automatic escalation; unaffordable precision → publish the attainable sensitivity and mark inconclusive.

### WP06 — Freeze and execute the primary confirmation

**Objective/risk:** obtain an interpretable result protected from adaptive selection.  
**Dependencies/entry:** WP05 allocation and all configurations frozen; environment/source/dependency hashes recorded; resume test passed.  
**Actions:** write the preregistration; create fresh protected seed manifests; train the fixed number of independent blocks; evaluate all five actions and all strata with the common scorer; run the frozen statistical and cost analysis. The runner exposes progress/health but not interim scientific summaries used for continuation decisions.  
**Files/commands:** `preregistration/main.md`, `protected_manifest.json`; `python -m originlab confirm --plan preregistration/main.md`; then `python -m originlab analyze --phase confirmatory`.  
**Tests/outputs:** complete raw result table, familywise intervals, block plots, robustness sensitivity, training/inference cost, failed-run accounting, and reconstruction of aggregates from raw rows.  
**Resources:** exactly the pilot-calibrated fixed allocation; checkpoint and resume without duplicate replicate IDs.  
**Acceptance:** all scheduled runs have a documented outcome; no test-driven tuning or unexplained exclusion; comparison and cost tables can be regenerated.  
**Branches:** positive versus all mandatory competitors → proceed to attribution and fresh follow-on replication; mixed/negative → state which competitors and strata win, then test only evidence-supported refinements; unstable/wide intervals → inconclusive at achieved precision; detected leak/score bug → invalidate affected results, fix, and generate fresh confirmation rather than relabeling the old data as protected.

### WP07 — Isolate feedback and resource regulation

**Objective/risk:** test H4/H5 without treating their addition as evidence for every part of the original idea.  
**Dependencies/entry:** base predictor and conventional controls are understood; a preregistered new extension question is justified by reliability or resource failures. A CRB H2 win is not mandatory if the appropriate extension is instead applied to the best baseline.  
**Actions:** implement the pre-observation gain loop; compare constant gain, innovation EMA, B1 adaptive filtering, generic recurrent gain control, and CRB-conditioned gains with equal telemetry and gain-network budget. Enable the externally enforced ledger and compare threshold-rest, common MPC, and matched resource-aware alternatives.  
**Files/command:** `configs/populations/feedback.yaml`, `controllers.py`, feedback modules; `python -m originlab feedback-study`.  
**Tests/outputs:** forecast calibration before/after reliability changes; no current-residual leakage; loss/control/resource Pareto curves; disabled-actuator correct attribution; gain ablations; new fixed-sample analysis.  
**Resources:** calibrate incremental state/unroll cost; no LLM or API.  
**Acceptance:** a recursive-inference claim requires improvement beyond static and ordinary adaptive gains, plus preserved calibration. A stakes claim requires better externally measured operation, not improved self-reported certainty.  
**Branches:** generic gain control matches CRB → credit adaptive inference, not a subject-origin mechanism; a simple ledger policy suffices → regulation works but is not novel; self-reward exploitation appears → remove endogenous confidence rewards and invalidate that result; added machinery regresses quality/cost → omit it from the recommended architecture.

### WP08 — Cross control, sensing, and state lineage

**Objective/risk:** prevent “identity” from conflating unrelated causal variables.  
**Dependencies/entry:** core chronology, snapshots, and predictive models are validated; the new inputs and targets are specified identically for all methods.  
**Actions:** run the intervention program in Section 11. Add a private motion sensor attached to a separately selected public entity; expand the joint binding hypothesis rather than force control and sensing to share one origin. Run action-copy corruption, known and unknown delays, appearance changes, state copying, consistent state relabeling, observation-buffer deletion, and incompatible memory splicing with sham controls.  
**Files/command:** `configs/populations/sensory.yaml`, `tests/test_identity_interventions.py`, extension scorer; `python -m originlab identity-study`.  
**Tests/outputs:** separate control-binding/sensory-binding/error-reliability curves; genuine ambiguity cases; clone equality before divergence; memory-splice dependency results; explicit distinction between hard-coded invariances and learned generalization.  
**Resources:** a joint control/sensor state doubles candidates from 16 to 32; delayed hypotheses multiply them again. Recalibrate before composition.  
**Acceptance:** conclusions identify the particular variable followed by the computation; no human clinical or phenomenal labels are inferred.  
**Branches:** one-origin model fails a crossed-binding task → evidence against that unitary factorization; performance follows private action even when it is corrupted → overtrust of telemetry, not robust agency; arbitrary memory corruption harms all models → nonspecific damage, not unique self-necessity; delayed task has no horizon long enough to reveal effects → redesign the diagnostic horizon before interpreting failure.

### WP09 — Test a nongeometric software boundary, then qualify transfer

**Objective/risk:** determine whether a result is only spatial canonicalization and avoid presenting synthetic control as real deployment evidence.  
**Dependencies/entry:** a repeatable useful mechanism or a clearly scoped boundary question survives the main tests; no unresolved information leak.  
**Actions:** implement the bounded queue/router environment in Section 11.4. Replace spatial frames with candidate-relative resource roles, retaining private commands, hidden routing, persistence, uncertainty, and optional inference feedback. Compare exact/particle switching models, generic recurrence, and role-factored recurrence at matched information/cost. Only after simulated transfer, optionally use isolated real subprocess buffers for an engineering timing/robustness harness.  
**Files/command:** `envs/queues.py`, optional `envs/router.py`, queue-method adapters; `python -m originlab transfer-study --environment queues`.  
**Tests/outputs:** queue transition enumeration, no PID/name leak, label permutation invariance, unknown-route tracking, categorical predictive scores, actual queue-control cost, and a new independent confirmation record.  
**Resources:** bounded integer state and small predictors; profile any subprocess I/O separately. No external service, host-protection mechanism, or privileged OS operation.  
**Acceptance:** a nongeometric claim requires fresh gains against the nongeometric strong baselines. A real-process engineering claim requires actual executed subprocess measurements, not simulator timings.  
**Branches:** spatial benefit does not transfer → narrow to geometry; role ordering alone explains transfer → credit ordinary canonicalization; no utility gain → report a representation finding only; a realistic application is still unspecified → do not declare deployed value.

### WP10 — Independent replication, audits, and final report

**Objective/risk:** ensure the final conclusion survives a fresh run and an adversarial explanation.  
**Dependencies/entry:** all completed phases have retained raw artifacts and frozen analyses; negative and inconclusive phases are eligible for reporting.  
**Actions:** run a fresh-seed confirmation of any claimed positive mechanism after fixing the final implementation; independently rebuild the environment from the saved lock; rerun null and resume tests; repeat the scientific, validity, and feasibility audits in Section 12 using actual results. Generate the report directly from records.  
**Files/command:** `report.md`, `claims.csv`, `reproduction_manifest.json`; `python -m originlab report`.  
**Tests/outputs:** each claim links to experiment IDs, comparator, metric, effect interval, scope, compute cost, and limitations; excluded/failed runs remain visible; clean rerun reproduces the substantive finding to the declared tolerance.  
**Resources:** justified by confirmation uncertainty and the cost of verifying the claimed effect; no automatic large-scale expansion.  
**Acceptance:** an outside coding agent can reconstruct the result without this conversation or hidden notebook state.  
**Branches:** replication contradicts the first result → report both and downgrade the claim; stronger baseline erases the advantage → amend novelty/usefulness assessment; complete negative result → publish the informative limitation; a genuine robust local advantage remains → specify the next application or scale experiment without claiming consciousness.

## 11. Attribution, boundary conditions, and continuation rules

### 11.1 Main ablation program

Use training-from-scratch ablations for competitiveness and state interventions for computational dependence; do not confuse the two. Share feature sets, density heads, and budgets wherever possible.

The first ablation is a 2×2 design: candidate-relative frames on/off, and persistent likelihood-based binding on/off. Keep recurrent state capacity and decoder expressiveness matched. The “binding off” variant uses uniform candidate weights or a validation-selected generic recurrent gate, with both reported: uniform weights test whether binding information matters; a generic gate tests whether the particular filtering restriction matters. B2 supplies the strongest frame-off structured control.

A second, separately budgeted design adds inference feedback on/off to the strongest surviving structured and generic predictors. Provide identical previous-error summaries and gain-network capacity. Compare feedback versus simply concatenating the same meta-state to the decoder. A gain from additional telemetry alone does not establish a reflexive-processing advantage.

For persistence, compare full history, a calibrated sliding history, recurrent-state reset, and state retained with observation buffers discarded. Keep total state/history storage explicit. A retained-state model surviving buffer deletion demonstrates its implementation contract; because old evidence remains encoded in that state, it is not evidence of memory-free personal continuity.

For private information, remove own commands from **all** methods; independently scramble commands; corrupt only the efference copy while leaving actual execution unchanged; and provide correct versus incorrect resource ledgers. In symmetry conditions, calibrated uncertainty is success, not failure to choose an identity.

### 11.2 Crossed sensory and control binding

Add a private sensor \(\zeta_{i,t}\) generated from the motion of independently selected entity \(s_{i,t}\):

\[
\zeta_{i,t}=Q(p_{s_{i,t},t}-p_{s_{i,t},t-1})+\omega_{i,t},
\quad\omega_{i,t}\sim\mathcal N(0,0.05^2 I_2).
\]

It is expressed in public spatial coordinates to avoid adding an unspecified sensor-axis problem. Its routing can differ from control routing. All competitors receive the same additional private stream. Training contains matched and mismatched control/sensor bindings on unprotected seeds; held-out tests compose them with appearance, noise, or delay changes. A second extrapolation test can withhold mismatches during training, clearly labeled as a harder distribution shift.

A joint CRB extension enumerates \((c_{\mathrm{control}},s_{\mathrm{sensor}})\), 32 hypotheses before any delays. Give ordinary switching and correspondence competitors the same factorization opportunity. Extend the decoder to public and private outcomes, allowing a common full-covariance head if shared process noise makes diagonal components inadequate. Keep separate control and sensory scores, rather than hiding one failure in a joint aggregate. This is a new model/population with its own preregistration, not an invisible alteration of the main result.

An agent that follows the sensed body rather than the controlled body may be correct about sensing and wrong about action. Report that dissociation. Do not use “rubber-hand illusion,” “depersonalization,” or “split brain” as measured outcomes; at most, describe an explicitly limited analogy after the computational variables are reported.

### 11.3 Delay, cloning, and state interventions

A delayed efference-copy or actuator condition needs an explicit queue. Start with delay values `{0, 1, 2, 4}` and a ring buffer of issued commands. A candidate delay uses the appropriately lagged action in its emission. For effects of a newly chosen action, score at a horizon of at least `max_delay + 1` with future actions held to a specified no-op policy. A one-step no-effect prediction is correct when actuation is delayed. Do not expand exhaustive planning to `5^H` merely to run this diagnostic; rolling out five current-action interventions under the fixed future policy is sufficient.

Two copied states with the same inputs and RNG states must evolve identically until something differs. After a branch receives a different action, observation, or stochastic draw, divergence is ordinary causal branching. No test selects a uniquely genuine descendant. A shared-weight model does not imply shared recurrent state.

For causal state intervention, first use valid saved states from matched task conditions; record differences in recent histories, belief entropy, and residual magnitude. Swap the entire recurrent state or a defined submodule while keeping the actual environment and private resource ledger fixed. Compare with a sham reload, a consistent candidate relabeling, and equal-sized disturbances to a conventional recurrent baseline. Report the effect on future predictions and actions. A transplant may be out of distribution, so it establishes a computational dependency only within that limitation.

Randomly averaging incompatible memories is not a physically grounded model of merging subjects. A componentwise merge is an engineering corruption test unless a valid generative interpretation is supplied. Do not score an objectively “correct personal identity” after a merge for which the simulator defines none.

### 11.4 Nongeometric queue/router transfer specification

This is a gated second task, not a substitute for a failed geometric result. It tests the informational-boundary intention without relying on Euclidean position.

Two processes control two bounded integer queues with hidden routing permutation. Queue length \(x_j\in\{0,\ldots,64\}\). Actions are `{0, +1, -1, +2, -2}` items. With \(j=\pi(i)\):

\[
x_{j,t+1}=\operatorname{clip}(x_{j,t}+a_{i,t}+A_{j,t}-D_{j,t},0,64),
\]

where arrivals \(A\sim\mathrm{Bernoulli}(0.2)\) and departures \(D\sim\mathrm{Bernoulli}(0.1)\) are independent external processes. Initial lengths are uniform from 8 through 24. Control-reset hazard is 1/64. Public observation is `clip(x + noise, 0, 64)` with noise `{-1,0,+1}` of probability `{0.1,0.8,0.1}` and missing probability 0.1 per queue. Names/tags are independent of routes. Private records contain only one's issued command and any declared resource ledger.

The exact transition and observation tables are finite and can be enumerated for correctness. A strong learned-parameter switching filter or converged particle approximation provides the principal baseline; no physical-motion model is reused where inappropriate. Predict a joint categorical table of shape `[65,65]` over the next two observed queue lengths. Every learned method uses the same joint-table family; CRB may mix candidate-conditioned, consistently reordered tables, while B1 computes its predictive table from its filter. Normalize over all 4,225 outcomes. A missing coordinate is marginalized by summing its axis; both missing gives probability one. Score joint negative log probability divided by two, in nats per queue coordinate, separately from the geometric study; never pool scores from different units/tasks as if directly interchangeable.

The CRB adaptation places the candidate-controlled queue first in a role-relative representation and encodes other queue levels, capacity, recent changes, and commands relative to that role. This is a change from spatial subtraction to resource-role canonicalization, explicitly disclosed. A role-canonicalized generic recurrent baseline is mandatory. Control minimizes summed squared deviation from target length 16, with identical resource costs or constraints when enabled.

A real-process follow-on may instantiate two isolated Python subprocess buffers under a project-owned temporary directory, driven by a JSON-lines router. Keep queue sizes and permissions bounded; expose neither PID nor worker name to the model; record actual command/response timings, restarts, and resource use. Real OS scheduling noise cannot be counterfactually replayed exactly. Therefore assess logged predictive/control performance and matched repeated workload costs, not fictional exact paired timing counterfactuals. This harness is closer to a software-agent application but still not evidence for arbitrary deployed systems.

### 11.5 Outcome interpretation

| Observation | Permissible conclusion | Next justified action |
|---|---|---|
| B1 solves everything at lower cost | Functional binding is explainable by ordinary system identification in this scope | Publish the negative architectural result; use the simpler method |
| CRB beats flat recurrence but not B2/B4 | Causal factorization or generic geometry explains the benefit | Credit the actual bias; remove a self-specific novelty claim |
| CRB wins only at small data/time | Scoped sample/time efficiency result | Show continued-training curves and practical tradeoffs |
| Predictive gain without control/cost gain | Better forecasts on the specified population | Do not claim useful general agency; inspect the decision bottleneck |
| H4 beats static gains but not EMA/B1 | Adaptive reliability tracking helps | Prefer the simpler adaptive mechanism |
| A positive effect replicates against all mandatory controls | Evidence for a local functional architecture improvement | Test the independent software task or a relevant realistic application |
| Ambiguous histories produce calibrated uncertainty | Correct handling of non-identifiability | Preserve ambiguity; do not reward arbitrary identity selection |
| A synthetic effect fails on queues or real subprocesses | Domain-limited mechanism | Narrow the claim; diagnose representation or workload mismatch |
| Confidence interval is wide | Inconclusive at achieved precision | Use the recorded variance/cost analysis to decide whether further independent work is worthwhile |
| Unsupported mapping lies outside the candidate family | Representational limitation of this PoC | Add an explicit unknown/continuous-map model and new baseline comparison, not a forced label |

Evidence for scaling requires an observed trend with relevant capacity, history, or data; a diagnosed representation bottleneck; or a task that cannot express the original hypothesis at the current scale. “A bigger model may become conscious” is not a scientific continuation rule. A failed toy setting cannot disprove everything about functional first-person organization, but it can disprove a specific claimed local advantage.

## 12. Audit record, limitations, and final execution gates

This record describes challenges applied while preparing this plan and the resulting revisions. It is not an independent review. Except for the small diagnostic in Appendix A, the proposed simulator, training, backend, and statistical tests have not been executed.

### 12.1 Scientific and prior-art audit

| Material weakness found | Revision incorporated | Residual uncertainty / acceptance test |
|---|---|---|
| The original language could make established common-cause inference, action correspondence, or egocentric memory sound unprecedented | Narrowed the contribution to an integrated, intervention-tested inductive bias; added classical identification, neural switching, generic geometry, and structured-memory comparisons | A directly equivalent intervention-trained architecture or benchmark may still exist. Discovery of one changes the framing to replication or extension, not the results themselves |
| An “implicit origin” and an explicit vector can be invertible implementations of the same computation | Added the exact serialization/refactoring null; removed ontological interpretations of latent layout | No experiment here can uniquely identify an intrinsic subject variable under arbitrary latent reparameterization |
| Control identity, sensory identity, event causation, and process continuation were conflated | Defined distinct variables and interventions; clones have two descendants; sensing and control can disagree | The component definitions remain engineering conventions, not a theory of personal identity |
| An agent can appear confused when the experiment supplies no identifying information | Added symmetric/no-private-signal/correlated-action controls; uncertainty is a correct outcome | Quantify observability in each chosen workload before interpreting classification errors |
| Resource maintenance and gain modulation are already available in ordinary control/conditioning methods | Added common ledger constraints, adaptive-filter/EMA controls, and feedback-versus-concatenation ablations | A gain may be useful but entirely explained by established adaptive estimation |
| A geometric toy risks proving only that coordinate subtraction helps | Added a geometry-matched sequence model and gated role-relative queue transfer | Queue success is still synthetic; application value requires a relevant workload and measured bottleneck |
| Recurrence was being treated as sufficient evidence for a stable self-maintaining attractor | Removed any attractor claim from success criteria | A future attractor study would need an explicit state map, stationary conditions, stability analysis, and perturbation tests |

### 12.2 Experimental-validity audit

| Material weakness found | Revision incorporated | Acceptance test / consequence |
|---|---|---|
| Simulator-paired counterfactuals assume a structural coupling that separate predictive marginals do not identify | Primary scoring uses each action's interventional marginal; paired contrasts are labeled SCM-dependent secondary outputs | No claim about a learned joint distribution of incompatible potential outcomes without an explicit coupling model |
| CRB has more persistent scalars, mixture components, structural priors, or internal statistics than a naive RNN | Count state and density capacity, share permitted telemetry/optimizations, add B2 and capacity envelopes | A benefit that vanishes against these controls is attributed to the corresponding ordinary bias |
| A hard origin chosen by averaging uncertain positions can create a nonexistent body | Retain candidate-origin mixture components; per-entity position means are not mixed-origin geometry | Test two distant equally likely origins; no midpoint-origin representation may replace the mixture |
| Shared candidate networks can leave the world-coordinate baseline artificially unable to distinguish candidates | B2 explicitly receives candidate entity and signed action-frame matrix | B2 must fit identifying deterministic cases before a comparison is valid |
| A current residual or attention to a future token can leak the target | Pre-observation gains; forecast/commit chronology; Transformer causal-mask and prefix tests | Deliberate target injection must be detected, not merely prohibited in prose |
| Appearance, tracking, sensor coordinates, and action directions can be swapped inconsistently | Separate cosmetic relabeling from actual sensor remapping; use effective observed-coordinate action labels | Transform full histories and targets consistently in invariance tests |
| Millions of correlated time points can create spurious precision | Independent training blocks, dyad clustering, paired differences, frozen multiplicity handling | Broad intervals remain inconclusive; no transition-level pseudoreplication |
| The composition test could be described as entirely unforeseen despite exploratory design inspection | Distinguish held-out gradient-training compositions, exploratory pilot seeds, and protected confirmation seeds | Report exactly which populations informed architecture and tuning |
| Predictive score estimates can favor an exactly marginalized model over an under-sampled RSSM | Prior-sample convergence check; shared scoring conventions and original-coordinate density units | Inadequately converged or differently normalized scores cannot support superiority |
| Resource limits alter the unobserved co-agent's executed-action law | Add hidden co-agent ledger inference/mixture adaptation to resource-aware baselines | A fixed uniform executed-action baseline is invalid once its ledger can be empty |
| A known-route oracle has more information, so its gap is not an equal-information achievable target | Treat privileged oracles as optimistic bounds; use the exact equal-history Tier L filter where applicable | Do not require a learned model to close an inaccessible-information gap |

### 12.3 Implementation and feasibility audit

| Material weakness found | Revision incorporated | Blocking gate / recovery |
|---|---|---|
| “Apple Silicon support” does not imply support for every recurrent, density, or optimizer path | Critical-operation forward/backward tests; strict MPS mode; faithful CPU fallback | WP00 must validate the actual installed versions and model shapes |
| Published repositories have conflicting availability and dependency signals | Chen's available README does not establish released implementation; FloWM's scripts assume CUDA; Dreamer's Mac claim is evaluated alongside its actual requirements | Do not invent a portable release or mark a locally executable CPU path impossible without testing |
| A recurrent-only generic comparator could miss a strong attention-based memory alternative | Added a small causal Transformer within B3 and state/cache accounting | Its history and predictive head must have the same causal information boundary |
| Unified-memory counters can be double-counted and peak RAM guessed from chip name | Separate RSS/driver/system pressure, discover actual RAM, measure saved activations | Reconfigure batches first; capacity changes require an amendment |
| Chunk boundaries, stale prediction caches, or partial checkpoints can silently break continuity | Full-history replay/current model revision; complete state/RNG/cursor checkpoints; atomic saves and resume tests | Reject stale caches and inconsistent resumes; preserve failed runs |
| Multistep planning could collapse uncertainty or ignore its Monte Carlo cost | Sample predictive observations into cloned states; convergence-calibrate sample count; count all searches and assimilations | An apparent controller improvement is invalid if compute or uncertainty treatment is unequal |
| Source metadata or dependency placeholders could be mistaken for pinned executable releases | Every installed release, source revision, wheel hash, and dataset manifest is discovered and stored before use | Illustrative schemas are not valid run manifests until populated |

### 12.4 Stress cases and interpretation discipline

If the closest full published baseline runs locally and is relevant to the chosen task, include it or explain a scientifically specific exclusion. If only a small adaptation runs, name it as an adaptation and narrow the comparison. A failed install, unavailable implementation, incompatible license, or locally prohibitive runtime is a documented comparison limitation, not a win.

If memory is lower than expected, reduce live concurrency and use streaming/accumulation before changing the hypothesis family or history. If no faithful MPS implementation exists, measure CPU. If the statistically justified total workload remains impractical, deliver a smaller precision-limited diagnostic study rather than a fictitious conclusive result. No remote machine or paid API is needed for the specified core path.

If a synthetic gain disappears on the strongest tuned baseline, shared optimization, independent block, or queue workload, retain that result. If it remains but is tiny or offset by latency, memory, resource exhaustion, or stationary regressions, describe the scoped tradeoff. A statistically detectable effect is not automatically practically useful. An important regression cannot be hidden in the equally weighted primary mean.

The following domain-specific attacks remain mandatory before strong interpretation: intentionally correlated agents; zero commands; a no-op that does not identify orientation; source delays longer than the scored horizon; actuator maps outside the candidate family; deceptive appearance correlations; mismatched private telemetry; state transplants with incompatible histories; and identical cloned trajectories. These are not tests of subjective experience or clinical analogues of human disorders.

### 12.5 Remaining limitations

The research search was broad and iteratively targeted, but it is not exhaustive and cannot establish worldwide novelty or patent freedom. Several conceptual sources were inspected only at abstract/metadata level, and unavailable full text is explicitly recorded in the bibliography. The proposed neural mechanism has not been trained. Apple hardware measurements, faithful full-agent reproduction, and realistic application benefits remain unperformed gates.

The first task supplies tracked entities and an engineered finite actuator family. It therefore tests action-grounded organization, not autonomous discovery of bodies, objecthood, arbitrary tools, or a universal self-boundary. A positive result could still be ordinary structured system identification; a negative result need not reject all larger-scale or differently embodied architectures. Both statements constrain the scope rather than excuse unsupported conclusions.

No physiological suffering, machine welfare, phenomenal consciousness, or unique subjective continuity is measured. Resource ledgers are bounded test variables. The agent must remain inspectable, interruptible, and confined to project-controlled simulated/process resources.

### 12.6 Handoff completion and execution acceptance

The coding agent should begin with WP00 and WP01, not neural tuning. The plan is complete as a specification when the source references, simulator equations, public/private boundary, common interfaces, primary estimand, mandatory comparator families, local calibration, and stage gates are available in this file. The experiment is complete only when the relevant tests actually pass or their failure is reported.

Before announcing a confirmed result, verify: all mandatory comparisons were run fairly; protected evaluation was genuinely held out; costs include meaningful overhead; failed and unfavorable runs remain present; the analysis reproduces from raw records; checkpoints resume consistently; claims match the measured population and scope; and an independent fresh-block confirmation supports any promoted exploratory finding. Missing prerequisites produce a qualified result, not an implicit waiver.

## 13. References and search-coverage record

### 13.1 How to read the references

Reference IDs below link to primary publications, author-hosted research, official implementations, or official documentation. “Inspected” describes work done while preparing this plan, not work delegated to the coding agent. Abstract inspection is not full-method inspection. No referenced implementation was executed here. Access and repository status refer to **4 October 2026** and must be rechecked before depending on a release or license.

**[R01]** Jakub Limanowski and Felix Blankenburg. **Minimal self-models and the free energy principle.** 2013. [Full article](https://pmc.ncbi.nlm.nih.gov/articles/PMC3770917/). Inspected full HTML. The inferred common-cause account directly anticipates a central conceptual claim; it is not an empirical demonstration of the proposed neural architecture.

**[R02]** Catherine Hochman. **The Hierarchy of Selves in Perception.** Published 3 October 2025. [DOI: 10.1007/s13164-025-00780-1](https://doi.org/10.1007/s13164-025-00780-1). Inspected publisher text. Supports the conceptual distinction between egocentric reference and an explicit represented object; does not establish an engineering advantage.

**[R03]** Ruben Laukkonen, Karl Friston, and Shamil Chandaria. **A beautiful loop: An active inference theory of consciousness.** 2025. [DOI: 10.1016/j.neubiorev.2025.106296](https://doi.org/10.1016/j.neubiorev.2025.106296); [PubMed record](https://pubmed.ncbi.nlm.nih.gov/40750007/); [institutional full text](https://researchportal.scu.edu.au/view/pdfCoverPage?download=true&filePid=13138813820002368&instCode=61SCU_INST). Inspected abstract and institutional PDF, including relevant precision/recursive-inference passages and page images. Conceptual support for feedback into inference, not experimental validation of CRB or consciousness.

**[R04]** Pablo Lanillos, Jordi Pages, and Gordon Cheng. **Robot self/other distinction: active inference meets neural networks learning in a mirror.** 2020. [Full manuscript](https://arxiv.org/html/2004.05473); [DOI: 10.3233/FAIA200372](https://doi.org/10.3233/FAIA200372). Inspected full methods. Its learned sensorimotor contingency and self/other discrimination are direct empirical antecedents; its robot setup and probability construction differ from the exact filtering test here.

**[R05]** Yurun Chen, Tianyuan Gao, Yizhong Ge, Shikun Ban, Yizhou Wang, Hongkai Xiong, Wenjun Zeng, and Wentao Zhu. **Proprioceptive–visual correspondence enables self-other distinction in humanoid robots.** June 2026 preprint. [Version 1 full text](https://arxiv.org/html/2606.13222v1). Inspected methods and experimental description. Its temporal cross-modal correspondence is a strong recent alternative to a novel-self interpretation; it does not run the proposed independent control/sensing/lineage intervention suite.

**[R05C]** Chen and collaborators. **Humanoid self/other distinction project repository.** [Repository](https://github.com/Euron-ZC/Humanoid-self-other-distincton-model). Inspected repository and README. At inspection, the README described code as forthcoming and licensing as pending. Do not equate this page with an available licensed training implementation.

**[R06]** Josh Bongard, Victor Zykov, and Hod Lipson. **Resilient Machines Through Continuous Self-Modeling.** 2006. [DOI: 10.1126/science.1133687](https://doi.org/10.1126/science.1133687); [author laboratory account](https://www.creativemachineslab.com/evolutionary-self-modeling.html). Inspected author-hosted research description, not the complete original paper. Establishes an important historical self-modeling/adaptation antecedent; not a source for this plan's numerical design choices.

**[R07]** J. Y. Park, O. Biza, L. Zhao, J. W. van de Meent, and R. Walters. **Learning Symmetric Embeddings for Equivariant World Models.** ICML 2022. [Proceedings record](https://proceedings.mlr.press/v162/park22a.html). Inspected publication abstract and metadata. Relevant to geometric inductive biases and the mandatory symmetry-matched alternative; no published performance number is imported into the local comparison.

**[R08]** Hansen Jin Lillemark, Benhao Huang, Fangneng Zhan, Yilun Du, and T. Anderson Keller. **Flow Equivariant World Models: Structured Memory for Dynamic Environments.** FloWM, arXiv:2601.01075, inspected version 2, May 2026; initial posting January 2026. [Full version 2](https://arxiv.org/html/2601.01075v2). Inspected architecture, memory mechanism, assumptions, and implementation-related discussion. A particularly close structured egocentric-memory antecedent; known self-motion and its visual tasks differ from latent actuator binding here.

**[R08C]** FloWM authors. **Official implementation.** [Repository](https://github.com/hlillemark/flowm); [environment setup script](https://raw.githubusercontent.com/hlillemark/flowm/main/setup_flowm_env.sh). Inspected README, MIT license, requirements, and setup script. The inspected setup uses CUDA/FlashAttention-oriented installation; it is not an established M3 training path. Source revisions must be pinned locally before any port.

**[R09]** Edward Beeching and collaborators. **EgoMap: Projective mapping and structured egocentric memory for Deep RL.** 2020. [Manuscript record](https://arxiv.org/abs/2002.02286); [author project](https://edbeeching.github.io/papers/egomap). Inspected abstract and author project description. Relevant historical structured-memory precedent; not a reproduced baseline here.

**[R10]** Jacob E. Kooi, Mark Hoogendoorn, and Vincent François-Lavet. **Disentangled (Un)Controllable Features.** Initial 2022 manuscript; inspected version 2, January 2024. [Full version](https://arxiv.org/html/2211.00086v2). Inspected methods and task framing. Controllable/uncontrollable feature separation anticipates a key factorization; the benchmark conditions do not establish control/sensory/lineage identity in general.

**[R11]** Ashish Kumar, Zipeng Fu, Deepak Pathak, and Jitendra Malik. **RMA: Rapid Motor Adaptation for Legged Robots.** 2021. [Manuscript](https://arxiv.org/abs/2107.04034). Inspected manuscript text and abstract. History-conditioned adaptation is a strong ordinary explanation for a persistent latent state; privileged simulation training and its deployment setting differ from this equal-information prediction study.

**[R12]** Francesco Locatello and collaborators. **Challenging Common Assumptions in the Unsupervised Learning of Disentangled Representations.** ICML 2019; initial manuscript 2018. [Full manuscript](https://arxiv.org/html/1811.12359). Inspected theoretical framing. Supports caution about identifying latent semantics without assumptions; does not imply impossibility once interventions and structural priors are supplied.

**[R13]** Mehdi Keramati and Boris Gutkin. **Homeostatic reinforcement learning for integrating reward collection and physiological stability.** 2014. [DOI: 10.7554/eLife.04811](https://doi.org/10.7554/eLife.04811); [PubMed abstract](https://pubmed.ncbi.nlm.nih.gov/25457346/). Inspected abstract and metadata; publisher access was restricted. Internal-state regulation is established prior art, not a claim that simulated resource exhaustion has experiential significance.

**[R14]** Ethan Perez, Florian Strub, Harm de Vries, Vincent Dumoulin, and Aaron Courville. **FiLM: Visual Reasoning with a General Conditioning Layer.** AAAI 2018. [Publication](https://ojs.aaai.org/index.php/AAAI/article/view/11671). Inspected abstract/metadata. Generic learned conditioning motivates a matched feedback/conditioning control; full published experiments were not reproduced.

**[R15]** Tianwei Ni, Benjamin Eysenbach, and Ruslan Salakhutdinov. **Recurrent Model-Free RL Can Be a Strong Baseline for Many POMDPs.** ICML 2022. [Proceedings](https://proceedings.mlr.press/v162/ni22a.html); [inspected manuscript version](https://arxiv.org/html/2110.05038v3). Inspected full text. Motivates well-tuned ordinary recurrence and cautions against weak memory baselines; the proposed predictive adaptation is not that full RL algorithm.

**[R15C]** Ni and collaborators. **POMDP baselines implementation.** [Repository](https://github.com/twni2016/pomdp-baselines). Inspected README and dependency/reproduction instructions. Legacy task dependencies are not needed for this project's small predictor; no source execution or faithful result reproduction is claimed.

**[R16]** Danijar Hafner, Jurgis Pasukonis, Jimmy Ba, and Timothy Lillicrap. **Mastering diverse control tasks through world models.** Nature, 2025. [DOI: 10.1038/s41586-025-08744-2](https://doi.org/10.1038/s41586-025-08744-2); [inspected manuscript](https://arxiv.org/html/2301.04104v2). Inspected model and training description. Strong recurrent stochastic world-model context; B5 is explicitly a compact predictive adaptation, not its reported full-agent performance.

**[R16C]** Hafner and collaborators. **DreamerV3 implementation.** [Repository](https://github.com/danijar/dreamerv3); [agent source](https://raw.githubusercontent.com/danijar/dreamerv3/main/dreamerv3/agent.py); [requirements](https://github.com/danijar/dreamerv3/blob/main/requirements.txt). Inspected README, license, agent code, and dependencies. README reports Mac/CPU use, while inspected requirements include CUDA-specific JAX installation. Installation and competitive throughput therefore require a separate local gate; debug configuration is not a competitive result.

**[R17]** Xiaoyang Liu. **Self-Emergence Agent Architecture: Behavior-Inertia HMM, Reflexive Metacognition, and Social-Contrastive Self-Modeling.** Posted 15 September 2026. [Full version 1](https://arxiv.org/html/2609.17331v1). Inspected methods and reported experiments. Includes persistent behavioral state and recursive/social mechanisms, with hosted DeepSeek-Chat experiments; it is neither a sensory-control-loop replication nor evidence of subjective experience. No hosted model is required by the present plan.

**[R18]** Ryan Prescott Adams and David J. C. MacKay. **Bayesian Online Changepoint Detection.** 2007. [Manuscript record](https://arxiv.org/abs/0710.3742). Inspected abstract/metadata. Regime inference is a conventional competing interpretation, not itself a new subject-state mechanism.

**[R18A]** Matias Altamirano, François-Xavier Briol, and Jeremias Knoblauch. **Robust and Scalable Bayesian Online Changepoint Detection.** ICML 2023. [Proceedings](https://proceedings.mlr.press/v202/altamirano23a.html). Inspected abstract/metadata. Relevant refinement for outlier-sensitive switching baselines; its full implementation and performance were not inspected or reproduced.

**[R19]** Maximilian Seitzer, Bernhard Schölkopf, and Georg Martius. **Causal Influence Detection for Improving Efficiency in Reinforcement Learning.** NeurIPS 2021. [Proceedings abstract](https://proceedings.neurips.cc/paper/2021/hash/c1722a7941d61aad6e651a35b65a9c3e-Abstract.html). Inspected abstract. Action-conditioned causal influence is adjacent prior art; the present intervention protocol is specified independently rather than inferred from its title.

**[R20]** Tilmann Gneiting and Adrian E. Raftery. **Strictly Proper Scoring Rules, Prediction, and Estimation.** 2007. [DOI: 10.1198/016214506000001437](https://doi.org/10.1198/016214506000001437); [author-hosted PDF](https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf). Inspected scoring-rule definitions and comparison discussion, including a PDF page image. Supports distributional scoring; the project's population, aggregation, and power calculations are its own design decisions.

**[R21]** Rishabh Agarwal, Max Schwarzer, Pablo Samuel Castro, Aaron Courville, and Marc G. Bellemare. **Deep Reinforcement Learning at the Edge of the Statistical Precipice.** NeurIPS 2021. [Full manuscript](https://arxiv.org/html/2108.13264). Inspected uncertainty and evaluation discussion. Motivates reporting uncertainty and variability; does not certify the proposed block-t analysis for arbitrary training distributions.

**[R22]** PyTorch contributors. **MPS backend and MPS API documentation.** Official development documentation, accessed 4 October 2026. [Backend notes](https://docs.pytorch.org/docs/main/notes/mps.html); [MPS API](https://docs.pytorch.org/docs/main/mps.html). Inspected availability, datatype, synchronization, RNG, and memory facilities. The installed stable release may differ; local API/operation tests are mandatory.

**[R22E]** PyTorch contributors. **MPS environment variables.** [Official documentation](https://docs.pytorch.org/docs/main/mps_environment_variables.html). Inspected fallback, fast-math, and memory-limit controls. Do not treat permissive fallback or disabled allocation limits as evidence of a faithful or safe accelerated implementation.

**[R22L]** PyTorch contributors. **PyTorch license.** [Repository license](https://github.com/pytorch/pytorch/blob/main/LICENSE). Inspected license text. Preserve relevant notices and inspect bundled dependency terms before redistribution; package installation and open-source labeling alone do not complete a license inventory.

**[R23]** Apple. **MacBook Pro (16-inch, November 2023): technical specifications.** [Official support page](https://support.apple.com/en-us/117737). Inspected hardware configurations. Used only to confirm that configuration varies; the plan deliberately does not assume the user's memory/storage configuration.

**[R23J]** Apple. **Accelerated JAX on Mac.** [Official developer documentation](https://developer.apple.com/metal/jax/). Inspected plugin status, compatibility, and restrictions. This does not establish compatibility of all current JAX-based training repositories; the core path is PyTorch/NumPy with a CPU fallback.

**[R24]** Patrick Butlin and collaborators. **Consciousness in Artificial Intelligence: Insights from the Science of Consciousness.** 2023. [Manuscript record](https://arxiv.org/abs/2308.08708). Inspected abstract/scope, not every proposed indicator in the full report. The plan's functional-versus-phenomenal boundary is a limit on its inference, not a claim that this report supplies a definitive consciousness test.

**[R25]** Jakub Limanowski. **Cortical candidates for self-other distinction based on visual and action cues: where do we stand?** Published 7 January 2026. [DOI: 10.1007/s00429-025-03065-6](https://doi.org/10.1007/s00429-025-03065-6). Inspected publisher review text. Verifies the source cited in the original idea; neuroscience analogy does not validate the engineering claim.

**[R26]** NumPy contributors. **NumPy license.** [Official license documentation](https://numpy.org/doc/stable/license.html). Inspected terms. Used for the reusable numerical stack and license-recording requirement; does not imply all dependencies have identical terms.

**[R27]** SciPy contributors. **Student's t continuous random variable.** [Official `scipy.stats.t` API](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.t.html). Inspected API. Critical values are computed from the locally installed version; no release number is fabricated in this plan.

**[R27P]** psutil contributors. **psutil documentation.** [Official site](https://psutil.io/). Inspected project documentation. Supports the selected process-monitoring tool; physical-memory accounting still requires the separate unified-memory caveat and platform measurements.

**[R28]** Tianwei Ni, Michel Ma, Benjamin Eysenbach, and Pierre-Luc Bacon. **When Do Transformers Shine in RL? Decoupling Memory from Credit Assignment.** NeurIPS 2023. [Latest manuscript record, version 4](https://arxiv.org/abs/2307.03864); [inspected full version 2](https://arxiv.org/html/2307.03864v2). Inspected current metadata and earlier full-text memory/credit-assignment distinctions and implementation discussion. Supports inclusion of a strong causal-attention alternative, not a presumption that it wins this task.

### 13.2 Search coverage and stopping rationale

The search began with every substantive linked reference in the supplied idea, then expanded beyond its terminology. Search families covered minimal self/common-cause inference; agency/comparator/efference-copy mechanisms; robot mirror and proprioceptive correspondence; adaptive self-modeling and actuator identification; egocentric/projective/flow-equivariant memory; controllable versus uncontrollable features; latent identifiability; recurrent and attention-based POMDP baselines; stochastic world models; gain/precision feedback; homeostatic control; causal influence; change-point inference; statistical evaluation; and Apple-specific numerical/dependency constraints.

Backward tracing connected the common-cause proposal to earlier minimal-self and robot work. Forward/current searches identified the 2026 humanoid correspondence and FloWM papers, and verified the supplied September 2026 agent preprint. A second search round specifically challenged the selected refinement with ordinary adaptive identification, generic symmetry, controllability factorization, attention-based memory, and implementation constraints. These searches changed the baseline set and claim scope, rather than merely adding citations. The last targeted pass added the causal Transformer comparator and did not justify a stronger novelty claim.

The supplied Springer and active-inference citations were verified. The supplied global-workspace PMC page (`PMC8770991`) did not yield usable full text in the available browser session; no implementation choice depends on an uninspected passage. The supplied ScienceDirect taxonomy page (`S1566253525000673`) was discoverable but not fully inspected; its contents are not treated as decisive evidence. The September 2026 preprint was inspected rather than dismissed as future-dated. Repository inspection exposed code-availability and platform issues recorded above.

Exploratory patent searching did not produce an inspected, directly decisive patent claim. That is neither an absence finding nor a freedom-to-operate assessment. Other remaining blind spots include proprietary systems, unindexed or newly revised work, non-English material, inaccessible full texts, and implementations whose paper description hides equivalent engineering details. A later close antecedent can invalidate a novelty framing without invalidating a careful replication or empirical finding.

The research stopping judgment is **coverage of the material competing mechanisms and local implementation risks**, not a fixed source quota and not proof of completeness. Essential design decisions are supplied in this document; the coding agent is not being asked to redo the foundational literature investigation before implementing WP00.

[R01]: https://pmc.ncbi.nlm.nih.gov/articles/PMC3770917/
[R02]: https://doi.org/10.1007/s13164-025-00780-1
[R03]: https://doi.org/10.1016/j.neubiorev.2025.106296
[R04]: https://arxiv.org/html/2004.05473
[R05]: https://arxiv.org/html/2606.13222v1
[R05C]: https://github.com/Euron-ZC/Humanoid-self-other-distincton-model
[R06]: https://doi.org/10.1126/science.1133687
[R07]: https://proceedings.mlr.press/v162/park22a.html
[R08]: https://arxiv.org/html/2601.01075v2
[R08C]: https://github.com/hlillemark/flowm
[R09]: https://arxiv.org/abs/2002.02286
[R10]: https://arxiv.org/html/2211.00086v2
[R11]: https://arxiv.org/abs/2107.04034
[R12]: https://arxiv.org/html/1811.12359
[R13]: https://doi.org/10.7554/eLife.04811
[R14]: https://ojs.aaai.org/index.php/AAAI/article/view/11671
[R15]: https://proceedings.mlr.press/v162/ni22a.html
[R15C]: https://github.com/twni2016/pomdp-baselines
[R16]: https://doi.org/10.1038/s41586-025-08744-2
[R16C]: https://github.com/danijar/dreamerv3
[R17]: https://arxiv.org/html/2609.17331v1
[R18]: https://arxiv.org/abs/0710.3742
[R18A]: https://proceedings.mlr.press/v202/altamirano23a.html
[R19]: https://proceedings.neurips.cc/paper/2021/hash/c1722a7941d61aad6e651a35b65a9c3e-Abstract.html
[R20]: https://doi.org/10.1198/016214506000001437
[R21]: https://arxiv.org/html/2108.13264
[R22]: https://docs.pytorch.org/docs/main/notes/mps.html
[R22E]: https://docs.pytorch.org/docs/main/mps_environment_variables.html
[R22L]: https://github.com/pytorch/pytorch/blob/main/LICENSE
[R23]: https://support.apple.com/en-us/117737
[R23J]: https://developer.apple.com/metal/jax/
[R24]: https://arxiv.org/abs/2308.08708
[R25]: https://doi.org/10.1007/s00429-025-03065-6
[R26]: https://numpy.org/doc/stable/license.html
[R27]: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.t.html
[R27P]: https://psutil.io/
[R28]: https://arxiv.org/abs/2307.03864


## Appendix A. Complete reproducer for the executed analytical diagnostic

Save the following as `analytic_checks.py` in a writable project directory and run `python analytic_checks.py` in an environment with NumPy. It writes `analytic_results.json` next to the script. This is the exact diagnostic source executed during planning; it is not `originlab`, not neural training, and not an M3 performance test. No internet access, assets, or packages beyond NumPy and the standard library are required.

```python
"""Analytic validation only; no neural training or Apple hardware benchmark."""
import json
import platform
import sys
from pathlib import Path
import numpy as np


def lognormal(x, mean, var):
    return -0.5 * (np.log(2 * np.pi * var) + (x - mean)**2 / var)


def softmax2(logp):
    z = logp - np.max(logp, axis=-1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=-1, keepdims=True)


def run():
    rng = np.random.default_rng(20261004)
    n, steps, sigma = 4096, 32, 0.75
    own = rng.integers(0, 2, n)
    a = rng.normal(size=(n, steps))
    other = rng.normal(size=(n, steps))
    noise = rng.normal(scale=sigma, size=(n, steps, 2))
    d = noise.copy()
    idx = np.arange(n)
    d[idx, :, own] += a
    d[idx, :, 1-own] += other
    ll = np.empty((n, steps, 2))
    for j in range(2):
        ll[:, :, j] = (lognormal(d[:,:,j], a, sigma**2)
                       + lognormal(d[:,:,1-j], 0., 1+sigma**2))
    post = softmax2(np.cumsum(ll, axis=1))
    one = softmax2(ll[:, 0])
    last = softmax2(ll[:, -1])
    accuracy = lambda p: float(np.mean(np.argmax(p, axis=-1) == own))
    # Without private action and with identical independent motor laws,
    # both candidate bodies have exactly the same marginal law.
    ll_no_private = lognormal(d[:,:,0], 0., 1+sigma**2) + lognormal(d[:,:,1], 0., 1+sigma**2)
    q_no_private = softmax2(np.stack([ll_no_private.sum(1)]*2, -1))
    # A co-agent with exactly the same action produces identical laws under
    # either routing. Causal origin remains unidentified despite prediction.
    dc = a[:,:,None] + noise
    lc = lognormal(dc[:,:,0], a, sigma**2) + lognormal(dc[:,:,1], a, sigma**2)
    q_correlated = softmax2(np.stack([lc.sum(1)]*2, -1))
    # Rename/reorder the public body slots consistently. Posterior must permute.
    ds = d[:,:,::-1]
    lls = np.stack([lognormal(ds[:,:,j], a, sigma**2) +
                    lognormal(ds[:,:,1-j], 0., 1+sigma**2) for j in range(2)], -1)
    qs = softmax2(np.cumsum(lls,1))
    perm_error = float(np.max(np.abs(qs - post[:,:,::-1])))
    # SCM branch: retain other action and all exogenous noise; zero only own action.
    no_op = d.copy()
    no_op[idx, :, own] -= a
    observed_effect = d-no_op
    desired_effect = np.zeros_like(d)
    desired_effect[idx, :, own] = a
    effect_error = float(np.max(np.abs(observed_effect-desired_effect)))
    # Orthogonal coordinate change has absolute determinant one.
    x = rng.normal(size=(1000,2)); origin = rng.normal(size=(1000,2))
    R = np.array([[0.,-1.],[1.,0.]])
    z = (x-origin) @ R
    reconstruction_error = float(np.max(np.abs(z @ R.T + origin-x)))
    renamed_again = softmax2(np.cumsum(ll.copy(),1))
    clone_error = float(np.max(np.abs(renamed_again-post)))
    out = {
        'scope':'Known-law 1D two-body Gaussian diagnostic, not the neural PoC',
        'python':sys.version.split()[0], 'numpy':np.__version__,
        'platform':platform.platform(), 'seed':20261004,
        'episodes':n,'steps':steps,'noise_sd':sigma,
        'accuracy_one_observation':accuracy(one),
        'accuracy_last_observation_only':accuracy(last),
        'accuracy_32_observations':accuracy(post[:,-1]),
        'private_signal_absent_max_deviation_from_half':float(np.max(np.abs(q_no_private-.5))),
        'perfectly_correlated_actions_max_deviation_from_half':float(np.max(np.abs(q_correlated-.5))),
        'consistent_slot_permutation_max_error':perm_error,
        'paired_no_op_effect_max_error':effect_error,
        'coordinate_reconstruction_max_error':reconstruction_error,
        'exact_clone_max_error':clone_error,
    }
    assert accuracy(post[:,-1]) > accuracy(one)
    assert np.all(q_no_private == .5) and np.all(q_correlated == .5)
    assert perm_error < 1e-12 and effect_error < 1e-12 and reconstruction_error < 1e-12
    return out

if __name__ == '__main__':
    result = run()
    path = Path(__file__).with_name('analytic_results.json')
    path.write_text(json.dumps(result, indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))
```

The recorded environment/result is reproduced below. Platform strings and last-bit numerical errors can differ on another system; the mathematical invariants and information/no-information behavior should agree. The finite-sample accuracy is not a guaranteed population result.

```json
{
  "scope": "Known-law 1D two-body Gaussian diagnostic, not the neural PoC",
  "python": "3.13.5",
  "numpy": "2.3.5",
  "platform": "Linux-6.18.44-x86_64-with-glibc2.41",
  "seed": 20261004,
  "episodes": 4096,
  "steps": 32,
  "noise_sd": 0.75,
  "accuracy_one_observation": 0.745361328125,
  "accuracy_last_observation_only": 0.741455078125,
  "accuracy_32_observations": 1.0,
  "private_signal_absent_max_deviation_from_half": 0.0,
  "perfectly_correlated_actions_max_deviation_from_half": 0.0,
  "consistent_slot_permutation_max_error": 0.0,
  "paired_no_op_effect_max_error": 2.220446049250313e-16,
  "coordinate_reconstruction_max_error": 4.440892098500626e-16,
  "exact_clone_max_error": 0.0
}
```

## Appendix B. Implementation-critical algorithms and conventions

This appendix resolves remaining implementation choices; the routines are reference specifications, not claims that a corresponding package API already exists.

### B.0 Package entry point and discrete indexing

After creating `pyproject.toml` with the `src` package layout, install the project into its selected environment using the existing command `python -m pip install -e .`. Implement `cli.main() -> int` and a `src/originlab/__main__.py` that imports it and calls `raise SystemExit(main())`. All `python -m originlab ...` commands in this document are proposed subcommands to implement and test, not preinstalled commands. Save Appendix A separately as a project diagnostic; it is already runnable without `originlab`.

For deterministic group indexing use rotation `J = [[0,-1],[1,0]]`, reflection `D = [[1,0],[0,-1]]`, and `R[k] = J**r @ D**f` where `k = r + 4*f`, `r in {0,1,2,3}`, and `f in {0,1}`; powers here are matrix powers. Thus `R[0]` is identity. Candidate index is `8*j + k`. Keep signed integer matrices in the simulator and cast exactly for neural operations. Find the index of a composed view/action matrix by exact signed-matrix lookup, never by rounding a continuous out-of-family rotation to a valid label.

### B.1 Stable masked Gaussian-mixture scoring

Use this calculation both in the standalone reference scorer and as the mathematical target for the accelerated implementation. Let `mu` and `var` have shape `[batch, component, coordinate]`, `log_weight` shape `[batch, component]`, and `value`, `mask` shape `[batch, coordinate]`. Normalize log weights with `logsumexp`; variances must already be finite and strictly positive. Replace missing target values with a finite zero before tensor arithmetic, because `NaN * 0` is still `NaN`.

```text
safe_value = where(mask, value, 0)
coordinate_logp = -0.5 * (log(2*pi*var) + (safe_value[:,None,:]-mu)^2 / var)
component_logp = sum(where(mask[:,None,:], coordinate_logp, 0), coordinate)
logp = logsumexp(normalized_log_weight + component_logp, component)
```

With an entirely false mask, return log probability zero exactly: the empty marginal integrates to one. For CRB, flatten the candidate/component axes only after adding `log_b[c] + log_w[c,m]`. When updating binding, marginalize `m` but retain `c`; when scoring the final prediction, marginalize both. A four-coordinate forecast divided by four is the reporting convention, not four independent training examples.

Do not clamp an invalid density to an arbitrary floor and silently continue. The variance floor is specified; an additional likelihood floor would change the proper score and must be a separately declared robust-score experiment. Use float64 NumPy reference values on CPU and tolerance-tested float32 learned computations. Record nonfinite losses as failures, including their inputs and model revision.

### B.2 Exact Tier L filter

For observed displacement \(d_j=p_{j,t+1}-p_{j,t}\), candidate \(c=(j,k)\), and own issued action \(a\), the exact conditional emission is

\[
\ell_c(d,a)=\mathcal N_2(d_j;R_k a,\sigma_p^2I)
\left[\frac15\sum_{u\in\mathcal A}\mathcal N_2(d_{1-j};u,\sigma_p^2I)\right].
\]

Other-agent orientation disappears from this expression only because the uniform action set is closed under the square group and resources are disabled. It would not disappear for a directional opponent policy or a hidden exhausted ledger. Use the actual two-agent generator for the short-sequence enumeration test, not a second copy of the same marginal-filter function.

```text
b = uniform(16)
for each current observation p_t:
    if previous forecast exists:
        posterior_previous = softmax(log(previous_b) + log_emission(previous_action, p_t-p_previous))
        b = transpose(T) @ posterior_previous
    for each proposed own action a:
        forecast[a] = mixture over c and five co-agent actions using b and known Gaussian noise
    choose/receive action without looking at the future
    cache current b, p_t, and issued action
```

At a known scheduled diagnostic reset, the evaluator can report the reset; ordinary methods do not receive it. For a hidden reset, the prior change above is not a failure to anticipate a surprise. Binding labels used for evaluation are the effective observed-coordinate label \(Q R_k\), not the unrevealed physical-frame label \(R_k\).

### B.3 Tier N switching-identification baseline

Implement B1 first with the same sixteen discrete candidates and a continuous state estimate over the four position coordinates. It knows the functional form and prior parameter ranges, not the sampled parameters or route. Use a switching extended Kalman approximation only after comparison to a particle implementation on development data shows sufficient predictive agreement; otherwise the particle version is the competitive implementation.

An implementation-ready particle path is:

```text
A particle contains positions, current route/orientation candidate,
positive gains/drift parameters, sensor-regime hypothesis, and optional hidden other ledger.
For each actual logged action:
    propagate positions through the specified dynamics;
    integrate or sample the hidden co-agent action from its declared policy;
    evaluate only visible observation-coordinate likelihoods;
    normalize weights in log space;
    resample systematically when effective sample size falls below N/2;
    move to the next hidden-regime prior after the emission;
    retain a copy of normalized pre-resampling weights for likelihood estimates.
For each queried intervention:
    clone the current filtering particles;
    branch only on the own action and integrate predictive noise;
    compute the resulting observation-mixture log density, not a histogram with arbitrary bins.
```

Maintain independent random streams for simulator data, filter particles, and analysis. Use common particle random draws across queried actions where valid, and count all predictive samples in cost. Do not allow query order to alter the live filter. Systematic resampling's offset is uniform in `[0,1/N)` from the explicit filter generator.

Learn population/dynamics parameters from visible training likelihood by soft-assignment EM or differentiable weighted likelihood, respecting the stated parameterization. Online forgetting/rejuvenation uses training-selected settings. Static-parameter particle impoverishment is a known implementation risk: inspect effective sample size, parameter diversity, and repeated-sequence predictive convergence; compare weighted parameter optimization and resample-move variants on the same tuning budget. A filter with collapsed parameter support is not a strong baseline merely because it has many particles.

The `N/2` resampling threshold is an initial numerical configuration, not a scientific success criterion. Particle count and adaptation settings are chosen by development predictive convergence and measured cost. If exact nonlinear posterior inference is unaffordable, state the approximation and retain the flexible B2 competitor. Do not call B1 an exact Bayesian ceiling outside Tier L.

### B.4 CRB observation/action cycle

```text
state = initial_state()
for t in actual trajectory order:
    assert state.model_revision == current_model_revision
    if t > 0:
        evaluate the already-cached, chosen-action component density on current visible coordinates
        update previous binding likelihood, then apply the regime transition prior
        compute residual summaries from that pre-observation forecast
    assimilate current observation using the base rule or the PREVIOUSLY predicted gain
    form each candidate-relative geometry; update its own shared-network recurrent state
    update optional meta-state from permitted past/current-assimilation information
    predict the gain to be used for the NEXT observation
    forecast every queried action from this state, without mutating it
    select or receive the current intended action
    cache that action's component distributions and current model revision
    advance the real environment only once
```

The meta-state can use the current observed residual to choose a gain for the next observation, never retroactively for the prediction being scored. Every method is allowed the same raw residual/history information; only the integration differs. A candidate's recurrent state can persist across posterior-weight changes, but this does not justify blending vectors expressed in different frames.

During truncated training, reconstruct the prefix at the current parameter revision without gradients, then score the chosen segment with gradients. Apply the optimizer only after computing that segment's losses. Reconstruct again at the new revision before the next shuffled segment. During evaluation, neural weights are frozen and state updates are causal. B1's declared online parameter estimation is separate, equally observable, and costed.

### B.5 Multistep control and common randomness

The optional three-step resource controller evaluates all 125 current/future action strings by sampled predictive rollouts, executes only the first action, and replans after receiving real evidence. Initialize each rollout from a deep copy of the complete state. At every imagined step, issue the action, sample a future public reading and the common declared observation-mask process, advance the known own-ledger rule, and assimilate that imagined observation. The method must not receive the evaluator's actual future regime changes or co-agent intentions.

Begin with a small Monte Carlo sample count, double it on fixed development histories, and measure decision agreement plus uncertainty of action-value differences. Freeze the count when simulation error is small relative to the measured value differences relevant to decisions, or report unresolved action ambiguity. Use shared base uniforms/normals across compared action sequences; an explicit inverse-CDF mixture sampler can consume the same uniforms even when component weights differ. Sampling order must be deterministic and logged. Identical random seeds alone do not establish matching draws if implementations consume RNG streams differently.

For one-step control, use exact mixture moments as specified in Section 6.8. Multistep sampling, hypothetical assimilation, and search overhead all enter the same end-to-end latency boundary. Do not compare an expensive planner against a one-step competitor and attribute the difference to the predictor.

### B.6 Required run-selection and failure conventions

A seed manifest is generated before execution. One independent training block contains one shared training dataset, each frozen method's trained instance, and paired evaluation dyads. Seed namespaces are separate for environments, neural initialization, dropout if later introduced, latent sampling, minibatches, filters, and bootstraps. Both agents and all five intervention branches remain nested within their dyad and block in analysis.

Choose model configuration and checkpoint using development/validation only. Within B3, the best validated GRU, LSTM, or causal Transformer is the family representative; retain the other tested configurations in the report. No family representative may be changed after seeing protected scores. The primary five-family comparison uses these frozen representatives.

A deterministic implementation error invalidates affected runs and triggers a documented correction plus fresh protected evaluation when needed. A genuine training divergence at a valid configuration remains an outcome; report failure frequency and the preregistered treatment of undefined scores. Before confirmation choose either a common finite fallback forecast fixed from training data, whose use is explicitly scored and counted as failure, or a claim rule that any unscorable scheduled run prevents superiority. Never drop failing runs only for the favored method. The same policy applies to every family.

A local OOM is distinguished from statistical divergence. It may justify batch reduction with gradient accumulation if the effective algorithm is unchanged and resume equivalence passes. A change in history, capacity, candidate set, score, population, or baseline is a methodological amendment, not a silent restart. All interrupted/restarted records retain their parent run IDs.
