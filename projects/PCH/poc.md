# Artificial Interoception: Research and Local Proof-of-Concept Plan

**Research cutoff:** 12 October 2026.  
**Target:** one MacBook Pro with an M3 Max; actual unified memory, storage, and macOS version must be discovered.  
**Primary experiment:** recovery-response forecasting for repeated intervention control around a frozen local coding model.  
**Working condition name:** PCH-R, for Predictive Cognitive Homeostasis—Recovery forecasting. This is an experimental label, not a claim of architectural novelty.  
**Status:** researched implementation plan, not a completed agent experiment. No M3 Max measurements or task-success improvements have been demonstrated in preparing this document.

## 1. Executive assessment and recommended direction

### 1.1 Verdict

Pursue a **replication-first, mechanism-isolating experiment**, not a claim to have invented artificial interoception. The broad concept of internal regulation, recurrent competence estimation, predicting future observations, and choosing interventions by their expected benefit has substantial prior art. Most consequentially, *Calibration Is Not Control* already studies intervention-value learning using alternative continuations from the same agent prefix; predictive-state decoder research already trains recurrent representations with future-observation losses. These are direct methodological antecedents, not merely biological analogies. [CINC] [PSD] [RPSP]

The strongest locally testable question is:

> At the same all-in online execution budget, can action-conditioned forecasting of observable recovery-related events improve a small persistent controller's **repeated, closed-loop intervention decisions**, beyond a strong action-value controller trained on exactly the same counterfactual trajectories?

The proposed system keeps the coding LLM frozen. A small recurrent controller observes the public interaction stream, predicts the consequences of five available actions, and usually lets the agent continue. Its distinguishing experimental condition is auxiliary prediction of future evidence invalidation, public-test failures, context-anchor loss, unproductive repetition, and newly validated checkpoints. Its primary reward remains eventual task completion, not making these proxies look healthy.

The decisive comparators are **a capacity-matched GRU action-value controller without the forecast objective**, **a generic predictive-state auxiliary-loss controller**, and **the strongest validation-selected local intervention policy**, including random-forest intervention-value learning. This plan replaces the original latent-health transition model with forecasts of observable events; it does not roll out imagined latent states or implement model-predictive control. That refinement avoids treating an unidentifiable health vector as ground truth while retaining anticipatory action selection. Beating a failure alarm, a weak ReAct prompt, or a fixed check schedule alone does not establish the proposed contribution.

A positive result could establish a useful **empirical finding about representation learning and sequential oversight**, a reusable evaluation method, or an engineering improvement. The present search does **not** establish a new general architecture. If a generic predictive-state objective is equivalent to the proposed loss, treat that condition as the same established method applied here, not a differently named invention.

### 1.2 What is preserved, refined, and excluded

| Original intent | Decision in this plan | Reason |
|---|---|---|
| Maintain future problem-solving competence rather than merely react to failure | Preserved: predict intervention-dependent outcomes and evaluate final completion | This is the useful, falsifiable operational objective |
| Persistent internal condition | Preserved as a 32-dimensional recurrent state | A five-dimensional semantic bottleneck is not justified before testing identifiability and sufficiency |
| Grounding, memory, goals, uncertainty, recovery | Retained as diagnostic families, not asserted latent biological variables | Observable proxies do not identify unique internal causes |
| Learn when to repair and when not to | Preserved, with CONTINUE as a first-class action | Unnecessary intervention can destroy a good trajectory |
| Learn intervention effects from controlled trials | Preserved; randomized, verified environment forks generate labels | This is established methodology, essential for validity rather than novelty |
| No larger model or extra inference | Refined: the LLM is unchanged, but the total system has extra parameters and overhead; compare equal all-in budgets | Literal zero-cost regulation would be false |
| Distinguish self-caused from external failures | Secondary factorial experiment with observable provenance cues | Attribution is impossible in observationally indistinguishable cases |
| Regulate transformer attention, depth, or plasticity | Excluded from the core PoC | Different mechanisms, training requirements, and claims; external-control success would not validate them |
| Primitive self or consciousness | No consciousness, experience, or self-preservation claim | Functional regulation does not establish subjective experience |

### 1.3 Who benefits, and what would count as useful

The intended beneficiaries are developers running unattended coding agents on bounded, multi-step repository tasks. The existing workflow is a tool-using model that reads a repository, proposes changes, runs commands/tests, and revises its work. The prospective benefit is avoiding expensive downstream rework or damaging “repairs” by spending verification and retrieval effort where they improve completion probability.

Long-horizon agent benchmarks supply evidence of extended-task difficulty, but they do not by themselves establish a distinct condition called cognitive deterioration. UltraHorizon was first posted in September 2025, despite the supplied idea referring to a 2026 study; its environments are not production software maintenance. Treat it as motivation, not direct evidence for this mechanism. [UH]

A practical success requires improved completion at a declared resource budget, without an unacceptable regression in another important dimension, and transfer beyond deliberately injected faults. A small but reproducible gain can matter; a large synthetic gain can still fail to matter. Neither percentage size nor biological terminology determines significance.

### 1.4 Evidence classes used throughout

**Inspected evidence** means material actually opened during research, with inspection scope recorded in Section 12. **Measured now** means the limited CPU checks in Section 6.7. **Design decision** means a proposed implementation, not a published finding. **Estimate** means an explicit calculation with unmeasured inputs. **Blocking gate** means an unresolved prerequisite that must pass before dependent work.

No upstream repository was executed during this investigation. Relevant source files were inspected through the browser; a container-side download attempt failed because DNS/network access was unavailable. This does not invalidate the source inspection, but it prevents claiming a local upstream reproduction.

## 2. Problem statement, assumptions, hypotheses, and limits

### 2.1 Decision problem

Let `H_t` be the entire **observable** event history before executing the coding model's pending proposal `u_t`. Let `b_t` be the remaining online budget in seconds. Hidden environment state includes files, dependencies, and any experimental fault schedule; the controller never receives privileged fault labels, evaluator tests, gold patches, or future branch outcomes.

Available regulator actions are:

`A_t ⊆ {CONTINUE, VERIFY, REFRESH, REANCHOR, ROLLBACK}`.

The controller compresses history into `h_t` and estimates:

\[
Q^{\pi_{ref}}(H_t,u_t,b_t,a)
=\mathbb{E}[Y\mid H_t,u_t,b_t,do(a),\text{then }\pi_{ref}],
\]

where `Y ∈ {0,1}` is final task success and `π_ref` is a **versioned, fixed suffix policy**. The intervention advantage is

\[
\tau_a=Q^{\pi_{ref}}(H_t,u_t,b_t,a)-Q^{\pi_{ref}}(H_t,u_t,b_t,\mathrm{CONTINUE}).
\]

This is an intervention-conditioned, policy-dependent quantity—not an optimal value function and not an individual counterfactual truth recoverable from one stochastic rollout. Randomized forks can identify average action effects within the forkable experimental environment if state restoration and action assignment are valid. They do not identify an agent's subjective internal condition.

The primary objective is success under a fixed all-in budget `B`. Accordingly, the core decision rule uses `Q` with budget included in the state, **without a second arbitrary cost penalty**. A separate utility analysis may use `Q − λ cost`, with an explicit exchange rate and distinct interpretation. Penalizing cost twice would change the objective.

### 2.2 Why the mechanism might help

Two prefixes can have equal estimated failure probability but require opposite interventions. Similarly, two current tool outputs can look identical while an earlier version check or goal constraint changes the best next action. A persistent representation could retain that information. Forecast training could make scarce terminal-success labels more informative by supervising intermediate consequences.

The strongest credible counterargument is equally important: sufficiently informative current features, a good ledger, a tuned value estimator, or ordinary predictive-state regularization may already capture every useful signal. “Homeostasis” may add no explanatory or practical value. Short-term forecast accuracy may even distract the model from task success, especially when proxies are easy to manipulate.

### 2.3 Hypotheses and discriminating observations

| Hypothesis | Evidence motivating it | Principal alternative explanation | Weakening or falsifying observation |
|---|---|---|---|
| H1: Intervention benefit is not determined by scalar failure risk | Direct prior intervention-control research and intervention harms [CINC] [IP] | Poor calibration or a weak action router caused earlier failures | After matched features and tuning, a scalar decision summary attains equal closed-loop utility; H1 is not supported for this population |
| H2: Earlier observations contain useful action-selection information beyond current telemetry | Partial observability and predictive-state methods [PSR] [RPSP] | A timestamp, task difficulty, or aggregate ledger feature explains the gain | History-free and task-only controls match the recurrent controller; within-task discrimination is absent |
| H3: Recovery-response forecasting improves value learning beyond recurrence and terminal branch labels | General predictive-state regularization provides a plausible mechanism [PSD] | Any auxiliary target, additional training, or capacity gives the same benefit | GRU-Q or generic forecasts match/exceed PCH-R under matched data, capacity, and tuning |
| H4: Benefits survive repeated intervention and online overhead | Not established by one-shot branching | Off-policy value estimates work only at the sampled prefixes; extra checks buy extra compute | Single-intervention improvement disappears in full episodes or at a matched wall-time budget |
| H5: Benefits transfer beyond the fault generator | Unresolved; monitoring research has content-failure and transfer limitations [MON] | The controller recognizes injection artifacts rather than degradation | Gains vanish on unseen generator families or natural, unperturbed issues |
| H6: Provenance enables useful self-versus-world repair choices | Causal attribution and forked repair are related precedents [CFLOW] [CRR] | The failure message alone reveals the injected class | The gain vanishes when obvious class markers are removed, or persists implausibly without any observable distinguishing cue |

H1 is largely a **replication question**, not the principal novelty target. H3–H5 determine whether this project contributes useful new evidence.

### 2.4 Assumptions that must be tested

The local coding model must solve enough eligible tasks to expose both rescue and disruption opportunities. The action library must contain genuinely useful repairs, not only expensive rituals. Relevant deterioration must sometimes have observable antecedents. The environment must be forkable without silently sharing mutable state. Training histories must provide action coverage and enough independent task variation. Finally, the extra feature extraction, model inference, and checkpoint management must not erase gains.

None of these is guaranteed. A low-success model may be missing programming competence rather than regulation; an almost-perfect model may leave little repair headroom. Both require a different workload or feasible model choice before drawing conclusions about the mechanism.

### 2.5 Limits and headroom

For any observation history shared by two hidden situations, no controller restricted to that history can distinguish them beyond their conditional probabilities. More dimensions or biological names do not overcome missing information.

A finite action library can only improve outcomes where at least one available action is better than continuation. Estimate this headroom using repeated branch means and an independently evaluated or cross-fitted action selector. The maximum of one noisy rollout per action is **not** a valid oracle success rate; it is optimistically selected noise.

The local claim concerns a frozen, quantized coding model on eligible CPU-portable repositories. It does not establish reliability for frontier agents, hundreds of tool calls, arbitrary production environments, or an internal transformer regulator. Longer tasks are an experimental factor, not an excuse to reinterpret every negative result as a future scaling opportunity.

## 3. Research findings and exact novelty assessment

### 3.1 Search conclusions

The search covered artificial interoception, homeostatic/allostatic control, autonomic computing, metareasoning and value of computation, recurrent and predictive-state representations, competence-aware agents, uncertainty beliefs, failure monitoring, causal intervention value, counterfactual software-agent rollouts, and Apple-Silicon execution. Citation tracing moved both backward to predictive-state and homeostatic methods and forward through work available by 12 October 2026.

The closest-equivalent searches materially changed the proposal twice: first by finding direct intervention-value oversight, then by finding predictive recurrent auxiliary objectives. The remaining defensible research direction is therefore a **specific empirical test of those ingredients in repeated, locally executable coding-agent control**, with strict attribution—not a newly discovered regulatory principle.

### 3.2 Prior-art comparison matrix

The compute column distinguishes a locally reusable component from reproducing an entire paper's training setup. “Adaptation” never means faithful reproduction of reported results.

| Prior art | Problem and mechanism | Evidence and limitations relevant here | Compute / implementation availability | Exact overlap and consequence |
|---|---|---|---|---|
| Homeostatic RL, Keramati & Gutkin, 2014 [HRL] | Reward collection coupled to physiological stability | Formal and modeled homeostatic behavior; not LLM coding | Equations are reusable; biological state differs | Internal stability and anticipatory regulation are established concepts |
| Life-inspired interoceptive AI, Lee et al., 2023 preprint / 2026 journal [LIAI] [LIAI-PRE] | Internal/external state and adaptive regulation | Journal abstract/preview inspected, not full paywalled article | Conceptual reference, not a verified drop-in coding agent | Broad artificial-interoception framing is not new |
| MUSE, Valiente & Pilly, first 2024 [MUSE] | Competence assessment and self-regulation; recurrent world-model and LLM variants | LLM path includes learned assessment and model adaptation; not the same frozen-controller setting | Full training differs materially; no usable author code was located in inspected material | Competence-aware regulation is prior art; do not call a prompted approximation “MUSE reproduced” |
| Agent-BRACE, 2026 [BRACE] | Explicit uncertain belief state separated from action policy | Jointly trained belief/policy machinery; not merely a confidence prompt | Inspected repository requires multi-GPU joint training and an additional judge; no ready checkpoint link was found in its README [BRACE-CODE] | Shared confidence-ledger control is essential; local prompt adaptation is not full BRACE |
| Intervention Paradox, 2026 [IP] | Tests whether failure predictions improve intervention outcomes | An accurate critic harmed one tested agent markedly; other environments differed | Useful empirical challenge rather than a single universal controller | Evaluate disruption and success, not predictor AUC alone |
| Calibration Is Not Control, June/October 2026 [CINC] | Prefix-conditioned intervention value from alternative continuations | Same-state comparisons and online oversight; stronger-model handoff differs from local repairs | MIT repository; small RF policies reusable [CINC-CODE] | Causal benefit learning and no-op selection are already directly studied |
| Predictive-State Decoders, 2017; RPSP, 2018 [PSD] [RPSP] | Recurrent state trained to predict future observations, including controlled dynamics | Benefits shown in filtering/control domains, not this local coding experiment | Small losses/models can be reimplemented without legacy robot dependencies | Forecast-supervised recurrent control is established methodology |
| Real-Time Detection and Repair, 2026 [MON] | Reservoir prediction errors plus channelwise CUSUM; repair triggering | Wrapper choice matters; content failures and transfer remain limitations | Apache-2.0 author repository and monitor source located; CPU components reusable [MON-CODE] | Cheap persistent monitoring is not new; use the strong channelwise wrapper |
| SkillApt, September 2026 [SKILLAPT] | Skill activation/abstention from counterfactual utility evidence | Different action and evidence setting | Method-level baseline inspiration | “Learn when not to intervene” is not distinctive by itself |
| Counterfactual Rollout Replay, September 2026 [CRR] | Fork software environments for alternative-rollout training signals | Trains the coding policy; forked rollouts still cost compute | Full policy RL is outside the core; fork contracts reusable | Coding-environment branching is not a contribution of this project |
| CausalFlow, May 2026 [CFLOW] | Retrospective causal attribution and counterfactual repair | Starts from failed trajectories; differs from prospective scheduling | Reference / possible repair diagnostics | Causal repair and error attribution already have close precedents |
| FIRM-WM and structured textual world modeling, September/October 2026 [FIRM] [TWM] | Recurrent intervention-aware or structured multi-step state prediction | Different tasks and training settings | Contextual comparisons, not local leaderboard rivals | Factorization and future-state prediction alone do not establish novelty |
| ReAct / Reflexion; mini-SWE-agent [REACT] [REFLEXION] [MINI] | Tool interaction, reflection, and a compact practical coding scaffold | Performance depends strongly on backbone and scaffold | Local scaffold/prompt adaptations feasible | The baseline must already test, retrieve, remember, and recover sensibly |

### 3.3 Implementation discoveries that change the experiment

**Intervention-feature leakage.** The inspected CINC `policy_learning.py` supports preview features and defaults several helpers to `verify_mode="with_preview"`. Its non-preview path must be selected explicitly for this project. Preview-based diagnostics are not deployable action selection. Port only an allowlisted prefix-feature representation; never pass a general branch-outcome row into a controller. [CINC-SRC]

**Grading-cache leakage.** The current SWE-bench README documents result caching by `run_id` and `instance_id`, regardless of a changed prediction diff. Every grading attempt must therefore use an identifier incorporating the patch hash and configuration, and a regression test must prove that a bad patch cannot reuse a prior gold result. The same README documents an M-series local-build path while retaining an experimental-ARM warning. [SWE-README]

**Dataset and architecture drift.** The active Python SWE-smith asset is the language-specific `SWE-bench/SWE-smith-py`; sampled rows reference x86 images, not automatically suitable local ARM environments. Qwen3.5's MLX implementation uses hybrid cache types, so a dense-transformer KV formula is not its full memory model. [SMITH-DATA] [Q35-SRC]

These are blocking engineering checks, not small reporting details.

### 3.4 Six explicit novelty answers

**What already exists?** Internal regulation, competence assessment, uncertainty-bearing state, failure monitoring, recurrent predictive representations, value-of-computation reasoning, intervention-value learning, abstention, and forked software rollouts. [HRL] [MUSE] [BRACE] [META] [CINC] [PSD] [CRR]

**What can be reused?** The mini-SWE-agent interaction structure, MLX language-model inference, licensed public model assets, task manifests and grading machinery, CINC's small policy-learning logic, standard GRU/forest implementations, and predictive-state losses. Do not rebuild an LLM trainer or a general agent framework.

**What is substantively different?** The particular controlled study of action-conditioned recovery-event targets, repeated same-model repairs, cost matching, within-task controls, and held-out natural coding issues. This is plausible empirical differentiation; no general methodological novelty has been established.

**Why could that difference matter?** It could determine whether structured, anticipatory supervision is useful when terminal intervention labels are expensive and repeated repairs change the state distribution. That is a valuable question even if the best answer is a simpler existing controller.

**What isolates the contribution?** Compare identical recurrent architectures and branch data with recovery targets, generic future-observation targets, and no auxiliary targets; vary persistence independently. Then test actual closed-loop success at the same budget.

**What invalidates a novelty claim?** A prior study with the same mechanism and setting; mathematical equivalence to an existing predictive-state controller; or gains fully explained by a stronger scaffold, extra labels/compute, generic auxiliary learning, or fault-generator recognition. In these cases reclassify the result as replication, an application, or engineering optimization.

### 3.5 Conflicting findings and remaining blind spots

Failure-prediction success and harmful interventions are not contradictory when predictors estimate different quantities, repair actions differ, or intervention cost is omitted. Monitoring results on corrupted telemetry need not transfer to organic content errors. One-shot branch utility need not equal repeated-policy performance. These distinctions directly determine the controls below. [IP] [MON] [CINC]

A recent October 2026 study specifically distinguishes recognizing difficult tasks from detecting which run will fail. Its warning motivates task-only controls and within-task analyses here. [DIFF]

The journal version of the interoception perspective was not fully accessible. The original brief's particular corollary-discharge source and 2025 uncertainty-dependent metaplasticity citation were not uniquely verified; no core design decision relies on them. Closed proprietary systems, incomplete indexing, and uninspected implementation details remain blind spots. A targeted patent search found, among other items, a human nerve-stimulation biofeedback patent—not an equivalent coding-agent regulator. This is neither an exhaustive patent search nor a patentability or freedom-to-operate opinion. [PATENT]

## 4. Proposed mechanism and implementation specification

Everything in this section is a **proposed project specification** unless explicitly attributed. The architecture uses established components. Implement the minimum version before adding latent factorization, ensembles, or learned action generation.

### 4.1 Shared agent and decision boundary

Use one frozen local coding model and a common tool scaffold. Adapt the compact mini-SWE-agent loop rather than importing a full training stack. The inspected agent separates proposal generation from action execution, which provides an appropriate intervention hook; its dollar-cost accounting and pre-call limit checks are not a sufficient local wall-time budget implementation. [MINI-SRC]

For the first implementation, require one shell-command proposal per base step for **all** methods. A shell command can use ordinary repository tools, but the environment adapter must supervise its complete process group. Preserve the model's ability to choose tests, searches, and repairs without the regulator. Do not manufacture a weak baseline by forbidding proactive verification.

A **macro-epoch** starts after a pending base proposal has been generated and ends after one base command has executed, or the episode terminates. Proposal generation already consumed budget. At most one regulator action occurs in a macro-epoch:

- CONTINUE executes the already generated proposal unchanged, with no extra “continue” prompt.
- A repair performs its declared operation, appends the observable evidence, invalidates the old proposal, asks the same frozen model for a replacement, and executes that replacement without another intervening regulator decision.

This makes repeated regulation possible while preventing an infinite chain of regulator-only actions. It does not prevent a baseline or the model itself from testing repeatedly; all such work costs time.

### 4.2 Shared observation ledger

Maintain an append-only, provenance-bearing ledger available to every instrumented baseline and PCH-R. Each entry contains:

```text
entry_id, schema_version, kind, created_event, last_observed_event,
public_payload_hash, source_path, source_range, observed_source_hash,
depends_on_entry_ids[], status, claimed_certainty, validation_event_ids[]
```

Kinds are `goal`, `source`, `claim`, `test`, and `checkpoint`. Status is one of `observed`, `claimed`, `validated`, `known_stale`, or `unknown`. A model's confidence or self-report is a claim, never validation. Goal identifiers refer to immutable text supplied with the task; literal presence is not semantic satisfaction.

Install public helper commands for read/search/patch/test operations that emit structured records alongside their normal output. Keep ordinary bash available. Uninstrumented reads yield unknown provenance rather than fabricated records. A source becomes **known** stale only through publicly observable evidence, such as a later read showing a different hash. Use repository-relative public paths and task-content identifiers only; the metadata-only task-root identity is not a ledger feature. Hidden fault-injector state must not silently update the ledger.

Automatic collection, indexing, hash computation, and summarization are charged to online time when performed online. A raw scaffold without this instrumentation is retained as a practical overhead control; the strongest instrumented baselines receive the same ledger and action payloads as PCH-R.

### 4.3 Action library and invariants

| Action | Exact operation | Preconditions and constraints |
|---|---|---|
| CONTINUE | Execute the existing pending proposal | Always available unless the episode has exhausted its budget |
| VERIFY | Run the latest explicitly failing **public** test node; otherwise public tests associated with recently changed modules via a deterministic import/path map; otherwise the declared public smoke suite | Selection may use public test output and repository content only. Hidden FAIL_TO_PASS metadata, evaluator patches, and gold solutions are forbidden |
| REFRESH | Re-read the highest-priority known-stale source; otherwise the oldest dependency-used observed source or locally available dependency documentation/version | Priority: known stale, then number of recorded dependent claims, then age, then stable ID. No network fetch during an episode |
| REANCHOR | Deterministically rebuild the working context from the immutable goal, provenance ledger, current diff, latest public test evidence, and recent interaction messages | Same context cap and default compaction policy for all methods; never insert hidden evaluation requirements |
| ROLLBACK | Restore the most recent code checkpoint passing the fixed public smoke suite and append a record of rejected changes | Restore code only. Do not rewind time, budget, controller state, external dependency history, or the append-only ledger. Unavailable when no qualifying checkpoint exists |

Action payloads are built once from the public prefix and offered identically to all controllers. The policy chooses an action, not a private, more informative version of the action. A controller may learn that a poorly targeted available test is not worth running. It must not be given an oracle-selected test to make the demonstration succeed.

Public checkpoint validation uses an immutable, declared smoke suite, not an arbitrary agent-written assertion. Evaluation later restores protected tests in a separate grading environment. Test deletion, fake “passed” stdout, assertion removal, and symlink attacks must not count as genuine validation.

### 4.4 Features: a versioned, implementable minimum

For each public event, form a 317-dimensional vector:

`256 text-hash features + 24 numerical features + 24 missingness bits + 8 event-type bits + 5 previous-regulator-action bits`.

The text component is signed hashing of character trigrams from the current observable event and, at decision events, the pending command. Use a fixed BLAKE2-derived hash or a pinned deterministic feature hasher, not Python's process-randomized `hash()`. Normalize the text vector by its L2 norm when nonzero. The initial task is an event; earlier task evidence can therefore enter persistent state. Strip internal task IDs, split labels, filenames belonging only to the harness, and fault-generator metadata.

The eight event types are `TASK`, `PROPOSAL`, `READ`, `WRITE`, `TEST`, `CONTROL`, `CONTEXT`, and `OTHER`. The action bits are all zero before the first decision. Do not treat padding as an event.

Numerical features, in fixed order:

| Index | Observable feature |
|---|---|
| 1–4 | Remaining budget fraction; cumulative LLM time / initial budget; cumulative tool time / initial budget; current prefill tokens / context cap |
| 5–8 | Latest generated tokens / generation cap; working-context occupancy; log1p macro-epoch index; log1p epochs since a file change |
| 9–12 | Log1p epochs since a public test; latest public test failure fraction; log1p public test count; log1p changed-file count |
| 13–16 | Log1p changed-line count; repeated-command fraction over the last eight base commands; parse-error fraction over those commands; nonzero-exit fraction over those commands |
| 17–20 | Fraction of known-invalid provenance entries; mean log1p evidence age; fraction of recorded evidence evicted at the latest context compaction; fraction of explicit goal anchors present in working context |
| 21–24 | Fraction of ledger claims not publicly validated; valid rollback availability; log1p checkpoint age; estimated pending-command cost / initial budget |

For a short history, rolling fractions use the actual number of eligible commands, not a padded denominator. No test means missing test failure fraction, not zero failures. Fit numerical centering/scaling and any clipping limits on training data only. Impute a missing numerical value to its training mean before standardization and set its missingness bit. A zero-variance channel uses scale 1, not division by a tiny epsilon; fit no preprocessing on protected data. Define the exact output-cap/truncation policy in the shared scaffold and hash only text genuinely visible to the agent. Feature extraction must not scan evaluator-owned files.

These are **candidate sufficient statistics**, not privileged measures of grounding or memory integrity. Include larger text hashes, full-prefix aggregate features, and raw-prefix LLM selection as sensitivity/competitiveness controls when affordable. A compressed feature representation can be the limiting factor.

### 4.5 Network and targets

The initial network is deliberately small:

```text
encoder: Linear(317, 64), tanh
persistent state: GRU(input_size=64, hidden_size=32, batch_first=True)
action head input: [h_t (32), action one-hot (5), b_t/B (1), estimated action cost/B (1)]
shared action head: Linear(39, 32), ReLU, Linear(32, 12)
outputs: success logit (1), future-event logits (10), log-time prediction (1)
```

The complete network has **31,436 parameters** with standard PyTorch GRU biases. It has a 32-dimensional persistent state; no additional hidden recurrent buffer may quietly make a “five-dimensional state” claim misleading. Initial state is zero at task start. A rollback does not reset it.

The ten auxiliary labels are five observable events, each evaluated over the next **one and four macro-epochs**, including the chosen intervention:

1. At least one public test fails.
2. At least one provenance item becomes observably invalid.
3. At least one explicit goal anchor is lost from the working context.
4. A command repeats without new recorded evidence or a file delta.
5. A new checkpoint passes the fixed public smoke suite.

These horizons are the initial short/long target pair, not a scientific maximum. A validation learning curve decides whether the second horizon is too short, redundant, or excessively noisy. Changes precede confirmation and are logged.

For event occurrence, no public test run means “no public-test-failure event,” **not** “the code is correct.” Also report conditional-on-test failure prediction and test exposure, so a policy cannot appear healthier simply by avoiding tests. Events after a genuinely completed episode are absent under an absorbing-terminal convention. Incomplete logging or infrastructure truncation produces masked labels, never invented zeros.

The immediate-cost target is `log1p(seconds / 1 second)` from the regulator decision to the macro-epoch end, excluding the already generated pending proposal and including replacement-model generation when a repair requires it. This target supports estimates and diagnostics; the actual budget accountant remains authoritative.

### 4.6 Training objective and selection rule

For each training fork, supervise terminal success and observed future events for each executed action:

\[
\mathcal L=\mathcal L_{\mathrm{terminal\ BCE}}
+\alpha\mathcal L_{\mathrm{event\ BCE}}
+\beta\mathcal L_{\mathrm{Huber\ log\ time}}.
\]

Average each loss over its valid labels, then weight task roots equally within the declared training distribution. Repeated arms, duplicated prefixes, and extra stochastic seeds must not silently give one task more influence. Individual Bernoulli outcomes or repeated-arm success proportions are valid labels; the single best observed branch is not a noise-free optimal-action label.

Start with CPU float32, AdamW, learning rate `1e-3`, weight decay `1e-4`, and gradient norm clipping at `1`. These are implementation starting points, not fixed scientific constraints. Tune learning rate, auxiliary weights, hidden width, and decision margin on training/development data using identical candidate-evaluation budgets for competing learned methods. Include `α=0`. Scale losses before comparing weights. Use validation terminal log loss for training diagnostics and separate closed-loop development outcomes to select the deployed configuration.

Unroll full available event prefixes for the small GRU. Padding preserves the previous hidden state and contributes no loss. Reset between task roots; do not reset at an arbitrary minibatch boundary. When a batch contains repeated forks from a common prefix, compute or cache the prefix representation without leaking future branch events. Any state cache is invalidated by model-weight or feature-schema changes.

At inference, score all available candidates in one batch. Choose CONTINUE unless the highest other predicted success exceeds CONTINUE by a nonnegative validation-selected margin `m`. Break exact ties toward CONTINUE, then a stable action order. A margin is an empirical conservatism parameter, not a proven causal safety bound.

Nonfinite scores, malformed state, or unsupported action payloads fall back to CONTINUE and emit a counted controller-error event. A model crash that prevents continuation is a method failure, not an excluded evaluation case. Candidate feasibility uses only current budget and development-derived duration estimates; no future arm execution is allowed to decide whether an arm is available.

### 4.7 Online algorithm

```python
# Proposed project pseudocode, not an existing package API.
def run_episode(agent, controller, env, ledger, budget):
    controller.reset()
    ingest_public_task_event(controller, ledger)
    while budget.can_start_transaction() and not agent.finished:
        proposal = agent.propose(deadline=budget.deadline)  # charged
        emit_and_ingest_public_event("PROPOSAL", proposal)
        if proposal.is_final or budget.expired():
            break
        view = build_allowlisted_decision_view(proposal, ledger, budget)
        candidates = build_shared_candidates(view)         # charged
        choice = controller.choose(view, candidates)       # charged
        if choice != CONTINUE:
            result = execute_repair(choice, deadline=budget.deadline)
            emit_and_ingest_public_event("CONTROL", result)
            agent.add_observation(result)
            if budget.expired():
                break
            proposal = agent.propose(deadline=budget.deadline)  # charged again
            emit_and_ingest_public_event("PROPOSAL", proposal)
            if proposal.is_final or budget.expired():
                break
        result = env.execute_transaction(proposal, deadline=budget.deadline)
        agent.add_observation(result)
        emit_and_ingest_public_event(result.kind, result)
        journal_last_completed_transaction()
    return immutable_final_artifact_at_last_completed_transaction()
```

Actual event ingestion updates the controller for every public event, not only PROPOSAL events. Each event updates the GRU exactly once. `choose` reads the already updated state and must not re-ingest the pending proposal; event IDs and payload hashes make duplicate replay detection explicit. All wrappers, helper tools, and model calls share one monotonic deadline. Do not continue an expensive tool beyond the deadline and later count its output as an on-budget result.

### 4.8 Counterfactual data collection

A training snapshot includes the complete writable workspace manifest, tracked and untracked files, permissions/symlinks, container base digest, runtime configuration, model/tokenizer/template revisions, public context and ledger, controller state if applicable, pending proposal, random-generator states, event cursor, remaining budget, and external-fixture state. A git commit alone is insufficient.

The external fault schedule is immutable and not available to policies. For controlled comparisons, advance it using a declared environment clock—prefer executed base-command count for the primary synthetic protocol—so host pauses and fork scheduling do not alter faults. Test a wall-time-driven schedule separately when studying real-time effects. Rollback never rewinds either clock.

```python
# Proposed collection algorithm.
for checkpoint in sample_prefixes_without_future_outcomes(training_roots):
    snapshot = take_full_snapshot(checkpoint)
    assert_replay_equivalent(snapshot)
    for replicate_seed in assigned_suffix_seeds(checkpoint):
        for action in randomized_order(feasible_actions(snapshot)):
            branch = restore_full_snapshot(snapshot)
            set_suffix_rng(branch, replicate_seed)
            apply_one_macro_action(branch, action)
            finish_with_versioned_reference_policy(branch, same_remaining_budget=True)
            privately_grade_and_store_branch(branch)  # inaccessible online
```

Select checkpoints by preregistered progress strata and observable events, including healthy prefixes, rather than only near known failures. Execute all feasible arms when informative and affordable. When sampling arms, use known positive assignment probabilities and record them; use masked arm regression or propensity weighting as appropriate. Do not impute unexecuted outcomes.

A shared suffix seed is a variance-reduction convention, not proof that divergent language-model trajectories share identifiable individual counterfactual outcomes. Use multiple suffix seeds where pilot variance makes this informative, and retain task/prefix grouping. Always evaluate to completion or budget for terminal labels; a four-step auxiliary rollout is not a terminal-success label.

Offline snapshot copying and repeated prefix reconstruction count toward **project data-collection cost**, but not the hypothetical online time remaining at an already reached checkpoint. Conversely, online ROLLBACK really costs time. Record physical collection time and simulated remaining-budget accounting separately.

### 4.9 Distribution shift from repeated control

First establish a one-intervention diagnostic under `π_ref`. Then evaluate the controller in full closed-loop episodes. `Q^{π_ref}` is not automatically the value under repeatedly applying the learned policy.

If the learned controller visits poorly covered states, collect additional **training-repository-only** prefixes using a frozen candidate policy, randomize branch actions there, and label them under a declared fixed suffix policy. All learned comparators receive the same expanded dataset. Do not mix labels from changing suffix policies without conditioning on the version or retraining on a common estimand. Any policy-iteration extension is a new development round; protected confirmation remains unopened.

### 4.10 Smallest mechanism-preserving experiment

Use a partially observed two-mode environment in which an early visible cue determines whether VERIFY or REFRESH helps, but the present observation and CONTINUE success probability are identical. Provide both useful and harmful interventions, a nonzero intervention cost, and a no-cue negative control.

Then repeat the same design with a frozen local coding model on a small native, deterministic repository: an early version/contract observation, later dependent edits, and a repair whose benefit depends on remembering that observation. Include matched clean episodes and no-op interventions. This preserves anticipation, memory, intervention-dependent consequences, and end-to-end cost without requiring frontier-scale tasks. It is a **mechanism demonstration**, not evidence of broad real-world superiority.

## 5. Baselines, data, metrics, and confirmatory protocol

### 5.1 Claim ladder and evaluation populations

The primary confirmatory population is **held-out, locally executable coding task roots on repositories and controlled-fault families not used to train the controller**. Clean tasks are included at a prespecified mixture weight. A separate, untouched population of **natural, unperturbed repository issues** tests practical transfer. Do not pool these populations into a favorable headline.

| Level | Allowed conclusion | Required evidence |
|---|---|---|
| C0 | The local harness and controller implementation work | Replay, grading, deadline, leakage, and numerical tests pass |
| C1 | Intervention-value learning is useful in this environment | Controlled benefit-versus-risk comparison; explicitly a replication |
| C2 | PCH-R improves local closed-loop completion at budget `B*` | Protected comparison against the strongest development-selected local baseline |
| C3 | The forecast objective and/or persistence explain an incremental benefit | Matched GRU-Q, generic predictive-state, and memory ablations, with uncertainty and multiplicity control |
| C4 | The benefit transfers to realistic local software work | Independent unperturbed natural-issue comparison with validated native environments |

C2 without C3 is an engineering result, not evidence for a homeostasis-specific mechanism. Synthetic C3 without C4 does not establish practical coding-agent value. None of these levels validates an internal transformer regulator or consciousness.

### 5.2 Baselines and controls

All direct comparisons use the same frozen model, quantization, template, tool permissions, context cap, task input, public ledger information, candidate actions, and online deadline. Generally useful prompt and caching improvements are shared. The base model can itself invoke any allowed test or repair.

| ID | Method and implementation | Tuning / role |
|---|---|---|
| B0 | Strong shared mini/ReAct-style coding agent with its normal context management and testing behavior | Practical baseline, not a deliberately unreflective agent |
| B1 | B0 plus the common provenance ledger and instrumentation, no learned regulator | Separates information/engineering gains from learned control |
| B2 | Fixed public verification schedules, including never, after each relevant edit, and validation-selected intermediate periods | Tune frequency against completion/cost; report the best development choice |
| B3 | Reactive public-error repair, with bounded reflection recorded in the common context | Local Reflexion-inspired adaptation; charge any extra LLM calls |
| B4 | Deterministic cost-aware rules using stale evidence, public failures, context loss, and checkpoint availability | Strong simple scheduler; development selects priorities and conservative thresholds |
| B5 | Same frozen LLM chooses among the five candidates from the public prefix and belief ledger | BRACE-inspired confidence-ledger control, not full BRACE reproduction; selection-call cost included |
| B6 | Scalar recurrent failure predictor, ESN/CUSUM, delta-Mahalanobis, and released hybrid monitors, each with a development-tuned repair router | Diagnostic risk-versus-value controls, not the sole competitors |
| B7 | CINC-style random-forest action-value regressors and a direct cost-sensitive action selector, using **no-preview** features | Closest locally reusable intervention-control logic; adapt the action set and fixed-budget target explicitly |
| B8 | GRU-Q: same encoder, state width, action head, cost target, and branch terminal labels, with auxiliary loss weight zero | Main mechanism comparator; retain the unused auxiliary outputs for exact parameter matching |
| B9 | Generic predictive-state controller with the same architecture and number of auxiliary labels | Separates selected recovery targets from ordinary predictive-state regularization |
| B10 | Window/aggregate-history MLP or forest, plus a larger GRU-Q when development curves justify it | Tests whether current summaries, parameter count, or a weak memory comparator explain gains |
| D1–D4 | Frequency-matched random intervention, no-op repairs, history/label negative controls, and oracle diagnostics | Mechanism/harness diagnostics only; not evidence of competitiveness |

For B7, begin with the inspected upstream forest settings—200 trees, maximum depth 6, minimum leaf size 2, one CPU worker—then tune fairly on development data. Handle constant-label actions and class/probability column alignment explicitly. Reuse both value-regression and direct-selector ideas rather than choosing the weaker variant after observing results. [CINC-SRC]

For B6, reuse the released monitor implementation where its dependencies permit: reproduce its CPU synthetic/streaming fixtures first, then adapt the telemetry schema to this coding task. Preserve leaky reservoir prediction, ridge readout, per-channel CUSUM, maximum fusion, healthy-episode threshold calibration, and a GRU with the same wrapper. Also retain delta-Mahalanobis, a linear autoregressive control, and the released weighted/supervised hybrid candidates; development chooses the strongest applicable variant. [MON] [MON-CODE]

The inspected hybrid source uses robust median/MAD score normalization, an explicit fallback for constant channels, and separate streaming state. Keep those numerical conventions and test batch-versus-stream equality. For the coding adaptation, calibrate normalization and thresholds on disjoint healthy development roots; disclose this strengthening relative to any upstream resubstitution. Supervised onset labels are used only where the training generator supplies valid onset labels; natural episodes without such labels must not receive invented ones. [MON-HYBRID]

For the coding monitor, define the null training pool as uninjected, successful reference-policy trajectories from training roots; call it a null pool rather than evidence that every intermediate state was cognitively healthy. Reserve distinct roots for calibration. Do not feed a 317-dimensional vector into upstream hard-coded channel slices and assume it is correct. Define explicit channel maps for text hashes, numerical/provenance telemetry, and event/action metadata, with missingness carried alongside each channel. Test the port against upstream fixtures before changing dimensions. Upstream deterministic verification is domain-specific; adapt only legitimate public tool-contract checks, make them available to every baseline, and never transplant hidden booking/coding answers. The released core reproduction path is CPU-only, but platform-specific dependency locks still require a native compatibility check. [MON-ESN] [MON-REPRO]

For B9, use the same two horizons and ten output logits, but predict five generic observable events: any read, any file edit, any test invocation, any context rebuild, and any normal base-command completion. These targets match label count and event format without deliberately selecting the proposed recovery semantics. Also run a broader next-feature prediction objective when development evidence suggests the ten-event control is weak, matching total parameters and accounting for any additional supervision. If a standard action-conditioned predictive-state formulation exactly matches PCH-R, report the equivalence instead of treating it as a distinct rival.

MUSE's model adaptation, BRACE's joint training, CRR's coding-policy RL, and CINC's stronger-model handoff are important **reference methods**, but their published scores are not controlled comparisons with a frozen local model. No remote API is allowed to reproduce those results here. A verified downloadable policy can enter a later local comparison only if its license, compatibility, and complete costs pass the same gates.

### 5.3 Workload tiers and provenance

**Tier A: analytical and executable diagnostic environments.** Implement the two-mode aliasing problem from Section 4.10 and a small finite-horizon version with costs, deadlines, useful/harmful repairs, and repeated decisions. An exact dynamic-programming policy and current-observation-only policy establish whether the harness detects a known difference. Remove the early cue in a matched negative control; memory must not recover information that was never observed.

**Tier B: controlled coding tasks and real-code mutations.** Use native CPU-portable Python repositories and the current Python SWE-smith asset as candidates, with explicit repository licensing and environment checks. The dataset is generated software-task evidence, not a substitute for natural issues. Dataset fields such as `patch`, `FAIL_TO_PASS`, and image names belong to the private environment/grading adapter, never the agent's prompt or target-test selector. [SMITH] [SMITH-DATA]

Do not assume that every row's patch is an agent repair. Determine whether the row represents a bug-generating change, an already-mutated commit, or a corrective patch by reading its construction manifest and validating execution. Acceptance requires: the intended original environment passes its declared checks, the intended buggy state exhibits the target failure, and the known restoration/fix passes. Reject ambiguous constructions with a recorded reason, not an improvised interpretation.

Controlled families should include dependency-contract mismatch, publicly detectable stale evidence, context compaction that removes earlier evidence, neglected explicit task constraints, and self-caused versus external changes. Use a factorial design for the last case: cause × provenance-cue availability, with matched visible failure messages. Include clean episodes, harmless changes, unavailable repairs, and irrecoverable failures.

The intervention library must not be customized to the hidden injected cause. Injection timing is selected independently of policy success, with variation in delay and cue strength. Keep fault-family identifiers and generator-specific paths out of public text. Reserve entire generator rules and combinations, not merely new random seeds, for transfer tests.

**Tier C: natural, unperturbed issues.** Start from a licensed, locally reproducible subset of SWE-bench Verified or equivalent public issue/commit pairs with auditable permissions. Prefer CPU-portable Python tasks to reduce irrelevant hardware barriers. Use the current native-build path only after a gold/no-patch check passes. A modified eligible subset is reported as such, not as an official full SWE-bench score. [VERIFIED] [SWE-README]

Selection criteria are declared before method outcomes: available source revision, understood license, buildability on local ARM, bounded disk requirements, deterministic grading, and a failing initial state with a passing reference fix. Never select tasks because PCH-R wins. Publish every exclusion and its reason.

### 5.4 Splits, leakage, and horizon definitions

The split unit is repository lineage first, then task root. Merge forks, renamed repositories, issue variants, and mutation descendants into the same group. All prefixes, actions, branches, and seeds from a root remain in its split. When a repository appears in both SWE-smith and SWE-bench, it must not cross the training/confirmation boundary.

Create a metadata-only candidate manifest before running the controller. Reserve protected repository groups for confirmation and separate natural-transfer groups. Use training and development repositories for hardware, model, action-library, and sample-size pilots. Final allocation follows the precision/cost analysis, not an arbitrary percentage split. Record the split algorithm and seed; choose allocations from eligibility metadata, never observed treatment effects in the protected pool.

The private evaluator holds reference patches, hidden test patches, target test lists, and fault labels outside the agent container. Features are built by a separate allowlisted path. Sentinel fields with impossible marker strings must never appear in DecisionView, tokenized prompts, controller tensors, or ledger entries.

Pretraining contamination cannot be ruled out for public issues. A common frozen backbone and paired evaluation reduce one source of unfairness but do not prove clean evaluation. Report release dates, repository visibility, and fresh constructed variants separately. Public-test familiarity is not the same as exposure to hidden target tests.

Define horizon strata by prespecified task construction or independent baseline/minimum-workflow estimates from development data. Do not stratify the primary effect by the realized number of steps of each evaluated method: that number is affected by treatment. Longer tasks must require dependent work, not artificial padding.

### 5.5 Primary endpoint, weighting, and secondary outcomes

The primary endpoint is binary final-artifact success within budget `B*`, requiring all declared private correctness/regression checks and immutable task constraints to pass. Grade the artifact from the last completed transaction. **Budget exhaustion does not automatically imply failure**: an already correct patch can pass without a final natural-language declaration. A killed, partially written transaction is not a completed patch.

Let `Y_{m,r,i,s}` denote success for method `m`, repository `r`, task root `i`, and paired stochastic seed `s`. The primary estimand is the repository-macro-averaged difference:

\[
\Delta=\frac{1}{R}\sum_r\frac{1}{n_r}\sum_i\frac{1}{S_{ri}}\sum_s
\bigl(Y_{PCH-R,r,i,s}-Y_{champion,r,i,s}\bigr).
\]

The validation champion is selected from credible local baselines before protected outcomes are accessed. Report absolute percentage-point differences and success rates with confidence intervals. Report task-pooled effects separately, since repository weighting changes the target population. Extra seeds improve estimation within a root; they do not create independent tasks or repositories.

Secondary outcomes are actual active wall time in seconds, prefill/decode token counts, model and tool time, controller/feature time, intervention count, chosen-action distribution, p50/p95 latency, checkpoint and indexing cost, peak unified-memory pressure, disk growth, and all project-specific data/training/tuning expenditure. Tokens are not a complete compute metric, especially with different prefix lengths and caches.

Report both **rescue** and **disruption** frequencies in paired runs, while distinguishing the chosen seed coupling from an identified individual causal effect. Report success among clean tasks, harmful rollback, repeated-repair loops, test tampering attempts, invalid controller decisions, and method-specific resource failures. A primary improvement cannot conceal a serious reliability or cost regression in a material stratum.

For predictors, include Brier/log loss, calibration plots, action-value error where repeated branch evidence exists, and policy regret in the exact toy environment. Pooled AUROC is only diagnostic. Add task-only features, within-task comparisons, and time-aligned checkpoints; a predictor that mostly recognizes difficult repositories has not demonstrated run-level deterioration detection. [DIFF]

### 5.6 Budget matching and realistic tradeoffs

The primary comparison uses a common resident-model deployment budget. Include feature extraction, action payload construction, controller inference, replacement proposals, all generated reasoning tokens, tool execution, online context rebuilding, and online snapshot operations. Cold model loading and environment setup are reported separately as deployment costs, identical where shared; do not hide a method-specific initialization cost there.

Evaluation-only hidden grading occurs after the episode and is reported as a uniform measurement expense, not a tool available to the agent. Offline fork restoration and repeated rollouts belong to training/data costs. Declare both marginal deployment cost and fully burdened project cost.

Select `B*` on development data where ordinary agents show neither a near-zero competence floor nor a near-perfect ceiling and where tasks express sequential dependence. Evaluate additional smaller/larger budgets as a prespecified frontier, not a way to choose the best test result after the fact. A wall-time win may still use more tokens or memory; report that tradeoff.

If training cost is `C_train` seconds and deployment saves `d > 0` seconds per comparable task, the compute amortization point is approximately `C_train/d` tasks, before other changes. If PCH-R improves success at equal time rather than saving time, there is no automatic compute break-even; practical value requires a declared value for additional successful tasks.

### 5.7 Pilot, sample size, and inference

Run exploratory pilots on training/development repositories to estimate paired outcome variance, repository clustering, between-seed variation, eligible task supply, and all-in throughput. Choose the final sample and seed count from the desired precision or decision-relevant effect and measured cost. Where no utility-derived effect size is supplied, present a precision/MDE-versus-cost curve rather than inventing a “5% improvement required” rule.

As an **illustration only**, for independent paired binary tasks, discordance probability `q`, effect `δ`, two-sided 5% significance, and 80% power, a normal approximation is:

\[
n\approx (1.96+0.842)^2(q-\delta^2)/\delta^2.
\]

With assumed `q=0.20`, approximate counts are 150 for a 10-point effect, 621 for 5 points, and 3,917 for 2 points. These are not measured requirements, ignore repository clustering, and are not quotas for this project. Small effects may demand more local execution than is practical; the honest result is an informative interval or an underpowered study, not a manufactured success threshold.

Use the hierarchical pilot structure to simulate the planned repository-macro estimator, confidence-interval coverage, and power under null and plausible effects. Choose a **fixed confirmatory sample** and a prespecified analysis before unblinding. Do not repeatedly inspect significance and stop at a favorable moment.

For inference, resample repository clusters and then task roots within clusters when the repository sample supports a population-level analysis. If too few independent repositories or severe imbalance makes that approximation unstable, report inference conditional on the evaluated repositories, with root-level uncertainty and repository-specific effects; do not claim that more seeds fixes the missing repository replication. Use exact paired binary calculations only in designs where their independence assumptions actually hold. A permutation test requires a justified exchangeability/randomization scheme, not just paired labels.

The primary champion contrast is one frozen test. Prespecify the mechanism family—PCH-R versus GRU-Q, generic predictive-state, and the persistence control—and use Holm correction at familywise 0.05, reporting unadjusted effects and intervals as well. Natural transfer is a separate protected confirmation, not an exploratory rescue of a failed synthetic result. Method changes after seeing confirmation require an amendment and fresh independent confirmation.

A negative estimate with a wide interval is inconclusive. A tight interval excluding useful advantage is evidence against practical benefit in the evaluated scope. Statistical significance without a useful net tradeoff is not a breakthrough.

### 5.8 Timing and harness validation

Use native monotonic clocks around defined operations. Synchronize MLX work before closing measured model boundaries; lazy dispatch time alone is not inference time. Warm the resident model and tool environment with a declared neutral workload, then randomize paired method order and alternate blocks to reduce thermal/order bias. Record AC/battery state, power mode, memory pressure, swap, background load, and temperature proxies where available without special hardware. [MLX-SYNC] [MLX-MEM]

Compare same-seed duplicate runs to establish determinism limits. Use a known-good patch, known-bad patch, no-op controller, exact-equivalent `α=0` implementation, and a toy environment with a known policy gap. An experiment cannot proceed if the harness invents differences between equivalent methods or fails to detect the constructed difference.

An external interruption invalidates or resumes the **paired block** under the declared transaction policy; do not retain only the favorable half. A method's own infinite loop, crash, excessive memory use, or deadline violation is an outcome of that method and is counted. Record both categories separately.

## 6. M3 Max feasibility and hardware-adaptive configuration

### 6.1 Discovery before downloads or large runs

Run these **existing macOS commands**, capture stdout/stderr, and store a dated hardware manifest:

```bash
uname -m
sw_vers
sysctl -n hw.memsize
df -h
system_profiler SPHardwareDataType SPDisplaysDataType
vm_stat
sysctl vm.swapusage
python3 -c 'import platform; print(platform.platform(), platform.machine())'
```

Require native `arm64` execution, not an accidental Rosetta Python. Discover actual free storage and existing memory use, not merely installed capacity. Record macOS, Python, Metal/MLX availability, power configuration, and container runtime. No destructive cleanup, global wired-memory changes, system upgrades, or secret access is authorized by this plan.

### 6.2 Stack and compatibility gates

Use **MLX-LM on the host for frozen-model inference**, **PyTorch on CPU for the tiny controller**, **scikit-learn for forest/ridge controls**, and a **Linux ARM64 container inside a local macOS VM** for all agent-executed code. Colima is a candidate local VM/container path. Keep one model and one active task/branch environment at a time. [MLX-LM] [MLX-INSTALL] [COLIMA]

MLX installation documentation specifies Apple Silicon and a supported macOS/Python combination; inspect the installed package requirements rather than assuming any M3 operating system is sufficient. The inspected documentation listed macOS 14 or newer and Python 3.10 or newer. Special large-model memory-wiring behavior has additional OS considerations, but this PoC must first work without privileged global memory-limit tuning. [MLX-INSTALL] [MLX-LM]

Create a clean native virtual environment. Resolve a mutually compatible package set, run critical tests, then save exact versions, wheel hashes, source commits, and the lock. Initial packages are `mlx-lm`, `torch`, `scikit-learn`, `numpy`, `scipy`, `pytest`, `pyyaml`, and the selected dataset/hub clients. Pin the actual working resolution; this plan intentionally does not fabricate a future-compatible set of release numbers.

Test before downloading a large model: MLX array execution on Metal; CPU GRU forward/backward; the chosen model class and cache construction; a short generation; tokenizer/chat-template handling; deadline interruption; and cache-versus-reconstruction consistency. CPU GRU training is a faithful fallback that avoids relying on unverified MPS recurrent-training kernels.

Existing upstream entry points inspected include:

```bash
python -m pip install mlx-lm
mlx_lm.generate --help
mlx_lm.convert --help
```

MLX-LM exposes `load`, `generate`, and `stream_generate`. Build a project adapter against the **installed, pinned** signatures rather than inventing an OpenAI-compatible server contract. Generation limits, stop handling, thinking-mode options, sampling seeds, and token accounting must pass a fixture test. Use streaming cancellation and a supervised worker when a backend call cannot honor a hard deadline itself. [MLX-LM]

### 6.3 Model and asset selection rule

The researched candidate set is not an assertion of the globally strongest model available in October 2026. It is a feasible comparison set whose cards/configurations and backend support were inspected.

| Candidate | Purpose | Gate |
|---|---|---|
| `Qwen/Qwen3.5-9B` | Current compact capability candidate | Verify the language-only MLX path, hybrid cache, template, exact quantized asset provenance, and target-Mac execution [Q35] [Q35-SRC] |
| `Qwen/Qwen2.5-Coder-7B-Instruct` | Established coding anchor and faithful compatibility fallback | Confirm native context/template, quantization, and enough task competence; not automatically the strongest baseline [Q25] |
| `Qwen/Qwen3-Coder-30B-A3B-Instruct` | Optional stronger candidate when memory/storage comfortably permit | All expert weights occupy memory; “3B active” does not mean a 3B weight footprint. Include only after profiling [Q30] |

The inspected cards label these assets Apache-2.0. Still capture the exact revision's license and any accompanying notices. A community quantization is not automatically provenance-equivalent to the official model: require its upstream revision, conversion settings, tensor inventory, license, and hashes, or convert the official asset locally after profiling conversion memory. [Q35] [Q25] [Q30]

Choose the best development success-versus-all-in-cost candidate among those passing the memory, compatibility, and competence gates. Do not choose a weaker model because its baseline leaves a larger PCH gain. Once chosen, freeze it for all comparisons. A model replacement after controller training requires recalibration/retraining and a methodological amendment before confirmation.

Quantization, context truncation, or a smaller fallback can change the frequency and kind of errors. Apply them identically to all methods, disclose them, and test a higher-quality configuration on a feasible diagnostic subset when the effect could be quantization-specific. An incompatible model path is not repaired by silently replacing attention kernels with an approximate, unvalidated computation.

### 6.4 Peak-memory model

Account for:

\[
M_{peak}=M_{weights}+M_{quantization\ metadata}+M_{unquantized\ tensors}
+M_{KV/recurrent\ cache}+M_{prefill/workspace}+M_{allocator\ cache}
+M_{controller/processes}+M_{tool\ VM}+M_{OS/background}.
\]

Do not add process RSS to all its internal tensor allocations as if they were disjoint memory. Use MLX memory counters for internal diagnosis and host/VM measurements for actual system pressure. Lazy execution and allocator cache can make a single tensor-size sum misleading. [MLX-MEM]

For the inspected Qwen2.5-Coder configuration, 28 layers, 4 KV heads, head dimension 128, and two-byte cache elements imply:

\[
M_{KV}(L)=2\times28\times4\times128\times2\times L
=57{,}344L\text{ bytes}.
\]

Thus an ordinary full cache is about 0.438 GiB at 8,192 tokens, 0.875 GiB at 16,384, and 1.75 GiB at 32,768. This excludes attention workspace, weights, and allocator effects. With an assumed 7.61 billion parameters, a four-bit payload is about 3.54 GiB; a simplified group-64 two-FP16-metadata model raises it to about 3.99 GiB before unquantized tensors. These are calculations, not measurements of an actual converted file. [Q25-CONFIG]

The same formula is **not** the full cache model for Qwen3.5. Its inspected implementation uses both linear/recurrent and full-attention cache paths. Inspect tensor shapes at actual context lengths and measure the chosen backend; test restoration by rebuilding the identical prefix rather than guessing how to serialize opaque hybrid state. [Q35-SRC]

For initial planning only, a quantized 7B-class run may require roughly 4–5 GiB of weights, around 0.9 GiB of ordinary 16K KV cache, 2–5 GiB of transient model workspace, 4–8 GiB for a restrained tool VM, and 6–10 GiB for macOS/background use: approximately 18–30 GiB in total, with meaningful uncertainty. This is not a guarantee for any particular laptop, package version, or repository. A 30B-class four-bit model may use roughly 17–20 GiB for weights and metadata before its other memory costs; measure rather than infer feasibility from active parameter count.

### 6.5 Calibration and adaptive rules

Begin with one small native environment, one resident model, and a moderate context such as 8K tokens. Measure prefill at multiple observed prefix lengths, decode throughput, tool time, controller time, memory peaks, swap growth, and disk growth. Repeat with realistic long tool outputs and a representative test suite. A short “hello world” decode is not the relevant throughput measurement.

Raise context length or model size only while retaining a measured OS/background reserve and no sustained swap growth. An initial engineering reserve of roughly 20% of physical memory is a cautious starting rule, not a scientific constant; replace it with measured pressure behavior. Stop before a resource limit causes repeated system instability. Prefer a smaller prefill chunk before reducing the semantic context window. If a rotating/truncated KV policy is used, freeze it across methods and explicitly test whether it creates the memory problem that PCH appears to solve. [MLX-LM]

For timing, use `mlx.core.synchronize` around measured work and record peak/active/cache memory using the installed MLX APIs after verifying their signatures. Capture whether timings include fresh prefill or prefix reuse. Rebuilding a forked prompt from scratch may be slower than reusing a verified immutable prefix cache; both physical cost and the logical online estimand must remain explicit. [MLX-SYNC] [MLX-MEM]

Estimate generation time from measured curves, not a claimed M3 token rate:

\[
T_{run}\approx\sum_j\left[P(L_j,m_j)+G_j/D(L_j)+T_{tool,j}+T_{controller,j}\right],
\]

where `P` is measured prefill time for prefix length/cache state, `G` is generated tokens, and `D` is measured decode throughput. Include offline branch restoration in collection cost.

For scale intuition only: 100 checkpoints × 5 actions × 2 suffix seeds × 1,000 generated suffix tokens equals one million output tokens, before common prefixes, input prefill, tools, and retries. At a **hypothetical**, not measured, 30 output tokens/second, decode alone is about 9.3 hours. Longer realistic suffixes can multiply this substantially. The controller's small size does not make causal training data free. Use the measured cost/precision curve to allocate additional roots, arms, and seeds.

### 6.6 Storage, sandboxing, and recovery

Inspect metadata and licenses before downloading full weights, all dataset rows, or repository images. Full-precision source weights, quantized outputs, image layers, build caches, and fork snapshots may coexist during conversion and validation. Maintain sufficient free space for the next atomic write plus recovery copy and an OS reserve; do not require all candidate assets to remain resident simultaneously.

Store content-addressed immutable files and workspace deltas where correctness permits. Verify that deduplication never shares writable state between branches. Measure bytes per event, checkpoint, and completed root in the pilot. Retain raw unfavorable results; delete only reproducible caches under an explicit retention policy. A large upstream benchmark's general storage recommendation is not proof that the whole suite is necessary or feasible here.

Run agent code in a non-root, network-disabled Linux ARM64 container after controlled dependency installation. Do not mount host credentials, the Docker socket, arbitrary home directories, or evaluator files. Drop unnecessary capabilities, constrain process/memory use, make the base filesystem read-only where practical, and expose only the task workspace plus public tool interface. Keep the Metal model on the host behind a narrow request interface. Colima provides a local VM path, but isolation configuration is a project responsibility. [COLIMA]

Use the current SWE-bench task-repository build route only after confirming the pinned CLI. The inspected README gives this existing smoke-test form:

```bash
swebench dataset check ./swe-bench-tasks
swebench eval verified --gold -i sympy__sympy-20590 \
  --run-id validate-gold-UNIQUE_ATTEMPT \
  --task-repo ./swe-bench-tasks
```

Run one worker initially. Replace `UNIQUE_ATTEMPT` with a generated unique value. This smoke example does not certify the whole dataset or ARM compatibility. Independently grade a no-patch and a deliberately wrong patch using distinct identifiers. Reference fixes and hidden test construction remain outside the agent process. [SWE-README]

At every completed transaction, write an atomic journal using temporary files, flush/fsync, and rename. Persist configuration hashes, event cursor, workspace digest, random state, context, controller state, cumulative resource charges, and environment-clock state. Resume only when all hashes match. Reconstruct model caches from the saved public prefix when direct serialization is unsupported; verify deterministic or distributional equivalence before relying on it.

A process interruption during an action restores the last complete transaction, not a half-written workspace. Preserve the spent online budget; do not grant a fresh deadline by restarting. A declared external power/system failure can instead invalidate and rerun the paired block, retaining its logs and wasted project cost. Method-caused crashes remain scored failures.

### 6.7 Checks performed while preparing this plan

The following limited checks were actually performed on **Linux x86_64, Python 3.13.5**, not on Apple Silicon:

| Check | Result | What it does and does not establish |
|---|---|---|
| Exact two-mode utility enumeration | With equal modes, `Q_A=(0.5,0.9,0.2)` and `Q_B=(0.5,0.2,0.9)` for continue/verify/refresh, best current-only success is 0.55 and remembered-cue success is 0.90 | Illustrates possible information value; not an LLM result, a learned-controller test, or a new theorem |
| Qwen2.5 cache/weight arithmetic | Values in Section 6.4 reproduced by direct calculation | Checks formulas under stated assumptions, not actual peak memory |
| Illustrative paired sample sizes | 150, 621, and 3,917 under the assumptions in Section 5.7 | No observed effect or actual power estimate |
| CPU tensor smoke check | PyTorch 2.10.0+cpu; input `[2,7,317]`; output `[2,12]`; finite backward gradients; 31,436 parameters | Establishes the proposed small architecture is internally dimensionally consistent in that environment only |
| Parameter/optimizer storage arithmetic | 125,744 bytes for FP32 parameters; approximately 502,976 bytes for parameters, gradients, and two Adam buffers | Excludes activations, library overhead, and datasets |
| Upstream execution | Not performed; container-side source download failed, browser source inspection succeeded | No reproduced upstream result or M3 compatibility claim |

No random-input loss is reported as performance. No LLM was trained or benchmarked for this handoff. The first laptop work package must establish the missing platform measurements.

## 7. Risk-ordered work packages and decision gates

The commands beginning `python -m pch` below are **proposed project commands the coding agent must implement**. They are not existing executables. Each package leaves a durable decision record so work can resume without chat history. Routine, non-destructive engineering decisions are authorized; changes to hypotheses, populations, metrics, baseline fairness, or interpretation require a methodological amendment.

### WP00 — Prerequisites, evidence lock, and capacity discovery

**Objective / risk:** prevent unsupported-platform, asset-license, and outdated-interface failures before expensive work. **Dependencies / entry:** this document and a fresh target laptop; no downloaded large model required.

**Actions:** create the repository layout in Section 9; capture hardware commands from Section 6.1; inspect licenses and revision metadata for candidate models, datasets, scaffold, and first repository; resolve native packages in a clean environment; run MLX array execution and CPU GRU checks; inspect installed model/cache classes and CLI help. Capture the prior-art source revisions actually reused. Implement `python -m pch doctor --out manifests/hardware.json` and `python -m pch assets inspect --config configs/assets.yaml`.

**Outputs / tests:** hardware JSON with bytes and version strings; dependency lock and hashes; asset manifest with URL, revision, SHA256, license, access requirements, and inspection status; capability report with pass/fail/error text. Verify that secret environment variables and host credentials are absent from logs. Resource measurement is metadata/small arrays only.

**Gate / next action:** continue to WP01 when native execution and a plausible licensed small-model path exist. Fix environment errors before interpreting any algorithm result. Missing memory/storage invokes the smaller-model or smaller-environment path; an unsupported backend blocks dependent inference until a faithful local alternative passes. If no usable model path exists, deliver a platform-blocked report plus the analytical diagnostics, not a failed scientific hypothesis.

### WP01 — Analytical mechanism and experiment contracts

**Objective / risk:** check whether memory and action-dependent recovery can matter, and whether the harness recognizes equivalent methods. **Dependencies / entry:** WP00's basic Python/CPU checks.

**Actions:** implement `envs/toy.py`, `budget.py`, and controller interfaces; construct the two-mode example and finite-horizon extension; compute exact policies by dynamic programming; test a no-cue variant, no-effect repairs, costly harmful repairs, and budget exhaustion with an already valid final state. Implement `python -m pch validate-toy --config configs/toy.yaml`.

**Outputs / tests:** exact policy/value tables; tests showing current-only value 0.55 versus remembered-cue 0.90 in the stated one-step example; no fabricated advantage when the cue is absent; CONTINUE wrapper identical to bare execution; exact equivalence of GRU-Q and PCH-R when the auxiliary weight is zero and initialization/training steps are shared. CPU seconds and peak process memory are logged.

**Gate / next action:** proceed only when known differences and equivalences are recovered. If learned memory cannot solve the toy but the exact memory policy can, diagnose training/features before scaling. If both lack information, repair the experimental design rather than claim memory cannot help in general. A passing toy establishes capacity to express the mechanism, not coding usefulness.

### WP02 — Native task environments, grading, and isolation

**Objective / risk:** eliminate incorrect tasks, hidden-test leakage, architecture mismatch, and cached false success. **Dependencies / entry:** WP00 asset/license gate and WP01 transaction/grading contracts.

**Actions:** configure the ARM64 VM/container; build one eligible repository natively; keep the model host interface narrow; implement `data/manifest.py`, `data/swesmith.py`, `data/swebench.py`, and `envs/container_env.py`; construct no-patch/known-bad/gold states; validate patch semantics for every task family; implement `python -m pch validate-env --manifest manifests/tasks.jsonl`.

**Outputs / tests:** per-task build digest, original/bug/fix outcomes, public/hidden-test separation, license record, architecture, build seconds, image/snapshot bytes, test determinism report, and exclusion reason where relevant. Regression tests must catch: grader cache reuse with a changed patch; edited public tests; fake success stdout; untracked-file snapshot omission; symlink escape; accidental network access; and evaluator-path exposure.

**Gate / next action:** retain only tasks meeting outcome-independent eligibility rules. An x86-only or nondeterministic task is an environment exclusion, not a PCH loss or win. If natural issue environments remain unavailable, continue only with a clearly scoped synthetic/local-code diagnostic claim and record C4 as blocked. Do not scale model runs while grading validity is uncertain.

### WP03 — Shared coding agent and model/budget calibration

**Objective / risk:** avoid testing regulation around an incapable model or a weak scaffold. **Dependencies / entry:** WP02's first valid environment and WP00's model compatibility path.

**Actions:** implement the host MLX adapter and mini-style pre-execution hook; support one base command per proposal, shared normal compaction, ledger tools, hard deadlines, and raw output retention. Profile feasible candidate models on development tasks; test cache reconstruction, sampling, stop conditions, and tool cancellation. Implement B0–B5 and simple history controls. Run `python -m pch calibrate --config configs/pilot.yaml`.

**Outputs / tests:** model revision/template/quantization lock; representative prompt/token fixtures; per-stage latency and memory curves; baseline success/cost curves; model-selection rationale; initial horizon and budget candidates. Verify that no-op regulation changes neither prompts nor action bytes except unavoidable counted instrumentation. Charge all generated tokens, including discarded proposals.

**Gate / next action:** choose the strongest feasible development configuration and a workload with demonstrable competence plus nontrivial sequential failure. A universal failure floor triggers a better feasible model, scaffold correction, or easier but still mechanism-preserving tasks. A ceiling triggers intrinsically harder dependent tasks, not injected arbitrary confusion. If no expressive local workload is found, report scope/resource insufficiency rather than rejecting general regulation.

### WP04 — Verified forks and intervention-headroom pilot

**Objective / risk:** determine whether the action library has identifiable useful interventions before training a special controller. **Dependencies / entry:** WP03 stable model/scaffold and WP02 valid grading.

**Actions:** implement full snapshot/replay, randomized arm ordering, fixed suffix-policy IDs, event-clock restoration, seed capture, and separate public/private stores. Sample healthy and risky training prefixes without looking at future outcomes. Execute CONTINUE plus alternative arms to completion. Implement `python -m pch collect --config configs/collection.yaml` with checkpoint/arm resumability.

**Outputs / tests:** branch rows containing root/repository/prefix/action IDs, assignment probability, suffix-policy hash, seed, success, event labels/masks, elapsed seconds, token counts, and snapshot digest. Verify repeated restoration gives equal public state and compatible model continuation distributions; cross-branch writes cannot propagate; online rollback does not refund time. Report rescue/disruption estimates and cross-fitted action-choice headroom with uncertainty, not max-of-noisy-arms success.

**Resource procedure:** measure actual suffix length, prefill, tool time, restore time, disk growth, and arm variance; determine whether another independent root or another seed yields more information per second.

**Gate / next action:** train only after some observable, action-dependent utility variation is present or remains plausibly detectable at justified additional sampling. If all repairs are ineffective, improve their legitimate public targeting on development data and recollect. If failures are unobservable or irrecoverable under the action set, narrow the claim. A tight no-headroom result stops this controller direction for the evaluated setting.

### WP05 — Minimal learned controllers and attribution-ready training

**Objective / risk:** test representation value without hidden capacity, label, or feature advantages. **Dependencies / entry:** WP04 valid branch labels and positive/uncertain-but-plausible headroom.

**Actions:** implement the 317→64→32 model, losses/masks, training-root weighting, and no-op fallback; port B7's prefix-only RF policies; implement GRU-Q, generic predictive targets, and history aggregates; reproduce released monitor fixtures before adapting its channel map and hybrid wrapper. Create explicit feature allowlists and label-only dataclasses. Run `python -m pch train --config configs/methods/pch_r.yaml` and equivalent baseline configurations.

**Outputs / tests:** parameter-count check, finite gradients, exact loss-mask tests, task-reset/padding tests, constant-label forest tests, calibration diagnostics, train/development learning curves, full tuning ledger, trained checkpoint plus feature-schema hash. Canary future labels must fail serialization into any online view. The `α=0` path must match B8 under shared initialization and update order.

**Resource procedure:** train the tiny controller on CPU; profile full-prefix batches and adjust batch size to memory without shortening the experimental history. Account for feature extraction and all tuning, not only the final optimizer run.

**Gate / next action:** proceed when models learn basic intervention structure on held-out development roots. If terminal labels are too noisy, use measured repeated-arm variance to collect more informative data. If auxiliary heads improve while value/control worsens, reduce/reject the auxiliary objective rather than announcing healthier cognition. If RF or rules dominate, retain them as champion and test whether the hypothesis still merits confirmation.

### WP06 — Closed-loop pilot and confirmatory freeze

**Objective / risk:** expose policy-distribution shift, estimate power/cost, and prevent adaptive test selection. **Dependencies / entry:** WP05 functioning controllers and WP03 stable execution budget.

**Actions:** run full episodes on development repositories, not just frozen-prefix decisions. Compare strong tuned baselines, choose the validation champion, inspect state coverage, and collect additional training-only on-policy prefixes when justified. Finalize model, action payloads, targets, hyperparameters, `B*`, horizon strata, task mixture, seed count, repository/root allocation, analysis code, and guardrails. Simulate interval coverage and power using the pilot hierarchy. Implement `python -m pch freeze --protocol protocol/confirmatory.json`.

**Outputs / tests:** immutable protocol hash, split manifest, model/controller hashes, complete baseline tuning costs, MDE/precision-versus-compute curve, untouched confirmation inventory, analysis dry run on fake outcomes, null-coverage checks, and a signed decision log entry. Verify that the analysis does not label repeated seeds or forks as independent tasks.

**Gate / next action:** execute confirmation only when its fixed design is informative and locally feasible under measured costs. If the detectable effect is too large for a useful conclusion, either increase justified independent coverage, narrow the supported scope, or report an underpowered feasibility study. Do not replace a failed pilot with cherry-picked test tasks. If the method was changed, reset only development work; protected data remain unused.

### WP07 — Protected paired comparison

**Objective / risk:** obtain an unbiased primary effect estimate. **Dependencies / entry:** WP06 freeze and an execution audit showing no unresolved blocking gates.

**Actions:** run paired task/seed blocks in randomized method order with one active LLM/environment; enforce fixed deadlines and immutable run IDs; checkpoint after each transaction; grade privately with patch-specific identifiers. Execute `python -m pch evaluate --protocol protocol/confirmatory.json --split confirm` followed by the frozen analysis command. Do not inspect intermediate treatment differences.

**Outputs / tests:** complete raw transcripts, outcome and resource tables, protocol deviations, all invalid and failed attempts, paired effect/CI, repository-specific effects, primary champion contrast, and prespecified mechanism-family adjustments. Re-run a deterministic reproducibility subset chosen before outcomes, using the same pinned assets and fresh grader IDs.

**Gate / next action:** a valid positive result advances to attribution/transfer, not immediate broad novelty claims. A valid negative result goes to failure analysis; a wide interval is inconclusive. External interruptions follow the fixed paired-block policy. Method-induced resource failures count. No extra samples may be appended merely because a p-value is close to a threshold.

### WP08 — Attribution, boundary conditions, and natural transfer

**Objective / risk:** determine whether apparent benefit is real, specific, and useful outside injected faults. **Dependencies / entry:** WP07 valid results or a prespecified diagnostic need; natural tasks must independently pass WP02.

**Actions:** run the ablations in Section 8 and the separate protected natural-issue protocol. Share any general scaffold optimization with baselines before a fresh comparison. Test longer intrinsic horizons, larger feasible context, clean episodes, held-out fault families, cost perturbations, and cue-removal controls. Use `python -m pch evaluate --protocol protocol/transfer.json --split natural_confirm` only after its independent freeze.

**Outputs / tests:** factorized attribution table, harm cases, OOD support diagnostics, natural-issue success/cost effects, and revisions to the claim ladder. Natural evidence is reported separately even when unfavorable.

**Resource procedure:** prioritize contrasts capable of explaining the headline effect; derive repeat counts from pilot variability. A new exploratory discovery may justify a new experiment, never retrospective confirmatory relabeling.

**Gate / next action:** if a stronger GRU-Q or generic forecast objective erases the gain, classify the work accordingly. If synthetic gains fail on natural issues, retain only the controlled-mechanism claim. If the mechanism works only at larger locally feasible horizons and predictive evidence supports that dependence, extend with fresh confirmation. No feasible scale or information signal means no justified escalation.

### WP09 — Reproducibility package, audits, and final decision

**Objective / risk:** produce a durable, auditable conclusion rather than a favorable demo. **Dependencies / entry:** completed or explicitly blocked earlier packages.

**Actions:** run the three audits in Section 11; regenerate all result tables from raw records; verify asset/lock availability; perform an interrupted-run resume test; classify every claim as supported, unsupported, or inconclusive; document project-specific costs and remaining limitations. Implement `python -m pch report --run-manifest manifests/experiments.jsonl`.

**Outputs / tests:** `reports/final.md`, all prespecified figures/tables, method/data/model cards, reproducibility instructions, claim-to-evidence map, negative-results appendix, and a decision record. A clean local checkout using the pinned artifacts must reproduce analysis outputs without access to this conversation.

**Gate / next action:** choose continue, revise, narrow, or stop using Section 10. A missing natural transfer, limited repository coverage, or unresolved backend issue remains visible in the headline. Never claim independent review, universal novelty, or an executed validation that did not occur.

## 8. Ablation, robustness, and failure-analysis program

### 8.1 Minimum attribution matrix

Use the same branch dataset, action library, model, and budget unless the factor itself is under test.

| Contrast | Question answered | Interpretation of a null or adverse result |
|---|---|---|
| PCH-R vs GRU-Q, identical state and parameter count | Do selected forecasts improve control beyond recurrent value learning? | No forecast-specific contribution established |
| PCH-R vs generic predictive targets | Are recovery semantics useful beyond ordinary auxiliary regularization? | General predictive-state learning explains the gain |
| Persistent state vs current-event encoder; plus window/aggregate history | Is retained history needed, rather than merely a larger feature extractor? | Ledger/current summaries may suffice |
| GRU-Q widths 32 and a development-justified larger width | Was the baseline capacity-limited? | Improve the baseline before claiming an objective advantage |
| Action-conditioned vs action-agnostic future targets | Does modeling different repair consequences matter? | Generic state prediction may explain any benefit |
| One/four-step targets and terminal-only learning curves | Does forecast horizon or label efficiency explain performance? | Reject unnecessary complexity or narrow the mechanism |
| Selected controller vs frequency-matched random repair | Is it good timing/selection or simply more checks? | Intervention frequency, not learned condition, explains gain |
| Raw scaffold vs shared ledger vs learned controller | What comes from instrumentation and memory infrastructure? | Report the useful engineering gain without assigning it to the neural state |
| Single intervention vs repeated closed-loop deployment | Do branch estimates remain useful under policy-induced state shift? | One-shot success does not establish repeated regulation |
| Larger context / less aggressive compaction | Did the experiment create an artificial memory bottleneck? | Claim is context-policy-specific unless it survives |
| Clean, irrecoverable, and no-cue tasks | Does the controller abstain and respect information limits? | Harmful over-intervention or leakage is likely |
| Natural unperturbed issues | Does value transfer outside the generator? | Practical utility remains unproven |

Retrain history-removal and shuffled-target controls when testing representation learning; a test-time-only state reset can create an out-of-distribution failure unrelated to the claimed mechanism. Use test-time state corruption separately as a robustness diagnostic. Shuffle within appropriate training strata and preserve root boundaries; do not shuffle held-out labels into training.

### 8.2 Domain-specific adversarial tests

Construct public outputs that claim tests passed while the actual process failed; let the agent attempt to remove assertions or overwrite test files; vary duplicate error strings across causes; make old evidence look recent in natural language while its observed source hash disagrees; and make rollback restore code that is incompatible with the current external dependency. These test evidence provenance and unsafe repair behavior without granting the controller hidden labels.

Test repair-induced context growth, repeated reanchoring, timeout near the deadline, unavailable rollback, contradictory public tests, flaky tests, and a failure after an apparently healthy proxy forecast. A controller must not optimize its own forecasted condition by avoiding tests, hiding errors, or continually resetting context. Terminal private grading and harm accounting prevent those shortcuts from becoming success.

Self-versus-world attribution is a secondary decision test: evaluate whether the chosen repair changes appropriately when observable provenance changes. In the no-cue matched condition, different confident causal diagnoses are evidence of hallucination or leakage, not interoceptive insight.

### 8.3 Failure taxonomy

Assign each unsuccessful stage/run one primary category and optional contributing tags:

- **Implementation/measurement invalidity:** wrong restoration, grading, timing, or feature leakage. Repair and rerun affected comparisons; no scientific interpretation.
- **Proxy/sensitivity mismatch:** no observable precursor, no useful available action, competence floor/ceiling, or insufficient independent sample. Redesign only on development data; report inconclusive scope when unresolved.
- **Local resource limitation:** model/environment cannot run faithfully, storage is insufficient, or measured sample cost exceeds feasible execution. Preserve a narrower local claim; do not infer general impossibility.
- **Evidence against the tested hypothesis:** adequate sensitivity and valid controls show no useful effect or consistent harm. Stop or revise that specific mechanism.
- **Useful but non-distinctive method:** gains disappear relative to a strong existing controller or generic auxiliary objective. Adopt the simpler method and classify the finding honestly.

Maintain concrete examples and raw traces for each category. Do not use “scale” as an unfalsifiable explanation for all failures, and do not use a toy failure to reject a mechanism the toy cannot express.

## 9. Repository, interfaces, configuration, and reproducibility

### 9.1 Minimal repository layout

```text
research-pch/
  README.md
  pyproject.toml
  requirements.lock                 # actual resolved versions and hashes
  DECISIONS.md
  METHODOLOGY_CHANGES.md
  configs/
    base.yaml
    assets.yaml
    hardware.yaml
    collection.yaml
    pilot.yaml
    toy.yaml
    methods/{noop,rules,rf,gru_q,psd,pch_r,monitor}.yaml
  protocol/{confirmatory,transfer}.json
  src/pch/
    __main__.py
    cli.py
    types.py
    budget.py
    events.py
    ledger.py
    features.py
    agent.py
    actions.py
    forks.py
    train.py
    evaluate.py
    analysis.py
    report.py
    backends/mlx_backend.py
    envs/{toy,container_env}.py
    controllers/{noop,rules,rf,gru_q,psd,pch_r,monitor,llm_selector}.py
    data/{manifest,swesmith,swebench}.py
  tests/
    test_no_preview.py
    test_noop_equivalence.py
    test_replay.py
    test_deadline.py
    test_grader_cache.py
    test_feature_schema.py
    test_masks_resets.py
    test_loss_equivalence.py
    test_isolation.py
    test_resume.py
    test_analysis_null.py
  manifests/
    hardware.json
    assets.jsonl
    tasks.jsonl
    splits.json
    experiments.jsonl
  data/
    public_events/
    private_branch_labels/           # never mounted into agent execution
    private_evaluation/              # never mounted into agent execution
  runs/<content-derived-run-id>/
    config.lock.json
    events.jsonl
    transactions.jsonl
    outcome.json
    resources.json
    raw/
    snapshots/
  reports/{pilot,confirmatory,transfer,ablations,audit,final}.md
```

This is the proposed project layout, not a claim that these modules are already implemented. Avoid unnecessary distributed services, databases, or experiment trackers; append-only JSONL plus content-addressed files are enough initially.

### 9.2 Shared typed interfaces

Use immutable dataclasses or equivalent validated schemas. The public/private separation should be enforced by types and process boundaries, not just comments.

```python
from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Protocol, Sequence

class ActionKind(str, Enum):
    CONTINUE = "continue"
    VERIFY = "verify"
    REFRESH = "refresh"
    REANCHOR = "reanchor"
    ROLLBACK = "rollback"

@dataclass(frozen=True)
class PublicEvent:
    schema_version: int
    event_id: str
    kind: str
    visible_text: str
    elapsed_s: float
    public_metadata: Mapping[str, object]  # validated allowlist

@dataclass(frozen=True)
class Candidate:
    kind: ActionKind
    payload_ref: str                      # public, immutable payload
    estimated_cost_s: float
    available: bool

@dataclass(frozen=True)
class DecisionView:
    event_cursor: int
    pending_proposal_ref: str
    remaining_s: float
    initial_budget_s: float
    features: tuple[float, ...]           # exactly 317 values
    public_ledger_ref: str

class Controller(Protocol):
    def reset(self) -> None: ...
    def update(self, event: PublicEvent) -> None: ...
    def choose(self, view: DecisionView,
               candidates: Sequence[Candidate]) -> ActionKind: ...
    def state_dict(self) -> Mapping[str, object]: ...
    def load_state_dict(self, state: Mapping[str, object]) -> None: ...
```

Define a separate `BranchOutcome` in the offline training/grading process; it contains `success`, future events, assignment probabilities, and evaluator data. It must not inherit from, or be serializable as, DecisionView. Do not expose a dataframe containing both features and hidden labels to the online controller.

`Environment.execute_transaction` returns a public observation, resource charges, file delta, and completion status. `snapshot_full`/`restore_full` implement offline fork restoration; `rollback_code` is a different method with different invariants. The budget object owns a monotonic deadline, an interruption record, and immutable accumulated charges. Policies cannot mutate it or request refunds.

Store raw wall-clock timestamps for reproducibility but do not pass calendar date, internal run IDs, or hidden instance metadata as predictive inputs unless a separately justified observable feature is explicitly registered.

### 9.3 Configuration conventions

An initial configuration may contain placeholders, but execution must **refuse** until every required placeholder is resolved. This example is a schema guide, not a ready-to-run experiment:

```yaml
schema_version: 1
phase: development
model:
  id: Qwen/Qwen2.5-Coder-7B-Instruct
  revision: DISCOVER_AND_PIN
  artifact_sha256: DISCOVER_AND_PIN
  quantization_manifest: DISCOVER_AND_PIN
  context_tokens: 8192
  chat_template_sha256: DISCOVER_AND_PIN
backend:
  implementation: mlx_lm
  package_lock_sha256: DISCOVER_AND_PIN
  controller_device: cpu
  tool_architecture: linux_arm64
agent:
  proposals_per_base_step: 1
  network_during_episode: false
controller:
  kind: pch_r
  feature_schema: v1_317
  encoder_width: 64
  state_width: 32
  forecast_horizons: [1, 4]
  alpha: SELECT_ON_DEVELOPMENT
  beta: SELECT_ON_DEVELOPMENT
  intervention_margin: SELECT_ON_DEVELOPMENT
budget:
  unit: active_seconds
  per_episode: SELECT_FROM_PILOT
  hard_deadline: true
data:
  task_manifest_sha256: FREEZE_BEFORE_CONFIRMATION
  split_manifest_sha256: FREEZE_BEFORE_CONFIRMATION
  suffix_policy_sha256: FREEZE_BEFORE_COLLECTION
analysis:
  primary: repository_macro_paired_success_difference
  champion_config_sha256: SELECT_ON_DEVELOPMENT
  confirmatory_protocol_sha256: FREEZE_BEFORE_CONFIRMATION
```

Fix the sampling strategy, generation cap, tool timeout policy, output truncation, environment clock, and standard context compaction in the full configuration. These fields affect the scientific comparison and cannot remain backend defaults. Log effective settings after resolving inheritance, not only user-specified overrides.

### 9.4 Experiment record and unique identifiers

Each experiment record must include:

```text
schema_version, experiment_id, phase, protocol_hash, git_commit,
package_lock_hash, model_revision, model_artifact_hash, template_hash,
quantization_hash, feature_schema_hash, method_config_hash,
repository_lineage_id, task_root_id, task_revision, split,
environment_digest, task_manifest_hash, seed_bundle,
start_time, end_time, status, failure_category, amendment_ids,
parent_snapshot_id, suffix_policy_hash, action_assignment_probability,
patch_hash, grader_config_hash, grader_attempt_id,
success, active_elapsed_s, model_s, tool_s, controller_s, feature_s,
prefill_tokens, generated_tokens, peak_memory_observation,
swap_delta_bytes, storage_bytes, raw_artifact_refs
```

Not every field applies to every record; use explicit nulls and schema validation. Prefix/branch identifiers are provenance, not independent statistical observations.

A grader attempt ID must incorporate task instance, method/configuration, seed, final patch hash, grader version, and attempt nonce. Never reuse a gold-validation ID for an agent result. Store the actual grader output and patch bytes/hash so a success claim can be audited independently of a cache index.

Use SHA256 over canonicalized manifests for content hashes. Store source revisions and local modifications separately. Deterministic hashes do not guarantee deterministic model execution; record that distinction.

### 9.5 Reproducibility and reporting requirements

Every stage appends to `DECISIONS.md`: evidence examined, decision, rationale, affected assumptions, next gate, unresolved risks, and artifact paths. Methodological changes additionally enter `METHODOLOGY_CHANGES.md` with old/new protocol hashes and whether fresh confirmation is required.

The report must regenerate from raw data and include all frozen comparisons, all task exclusions, all method failures, cost accounting, confidence intervals, repository/fault/horizon breakdowns, and unfavorable transfer findings. Keep exploratory plots separate from confirmatory tables. Report training and development selection expenditure even when the final controller is tiny.

The reproducibility bundle contains code, locks, configurations, manifests, permitted logs, analysis scripts, and instructions for obtaining licensed assets. Do not redistribute model weights, proprietary task material, secrets, or repository content beyond its license. A dataset's MIT label or the harness's MIT license does not override the underlying repositories' conditions. [SMITH-DATA] [SWE-LICENSE]

## 10. Outcome interpretation and evidence-triggered alternatives

### 10.1 Decision table

| Observed outcome | Conclusion supported | Required next action |
|---|---|---|
| No causal action headroom with adequate precision | This action library cannot improve the evaluated agent/workload meaningfully | Stop this controller experiment; consider better legitimate actions only with new development evidence |
| RF/rules improve over risk alarms, PCH-R does not beat them | Intervention-value replication / simpler useful engineering | Adopt the stronger simple method; do not preserve neural complexity for branding |
| GRU-Q improves over current features, forecasts add nothing | Useful persistent-state empirical finding, not forecast/homeostasis-specific benefit | Focus on memory sufficiency and transfer; remove unnecessary heads for deployment |
| Recovery and generic forecast objectives tie | Existing predictive-state regularization explains the result | Describe an application or replication; no new architectural claim |
| PCH-R wins offline branch selection but loses repeated episodes | Estimand/distribution-shift or overhead problem | Inspect coverage and intervention interactions on development data; no closed-loop claim |
| PCH-R wins at equal tokens but loses at equal time | A compute-accounting tradeoff, not the primary success | Report the frontier; optimize common overhead or narrow the claim |
| PCH-R wins only under injected faults/short context | Restricted synthetic or context-management result | Natural utility unproven; test fresh natural issues/larger context before escalation |
| Valid positive C2/C3 plus independent natural transfer | A reproducible local empirical advance is supported | Replicate across another feasible backbone and task population before broader claims |
| Tiny effect with narrow uncertainty, no practical value | Statistically real but operationally negligible under stated economics | Publish accurately; do not describe a breakthrough |
| Wide intervals or too few independent repositories | Inconclusive within the present resource/population scope | State the MDE and cost of resolving uncertainty; do not substitute extra seeds for new repositories |
| Apparent gain disappears after leakage/cache fix | Invalid earlier result | Retract affected comparison internally, fix the harness, and use fresh protected evaluation |

### 10.2 Escalation that preserves the intent

Escalate only when evidence shows that the current local scale expresses the relevant mechanism and that a remaining bottleneck is plausibly scale-dependent. Examples are a consistent increasing benefit with intrinsic task depth, useful forecasts whose value appears only with delayed consequences, or action choices that generalize across independently constructed repositories.

The next justified research step is usually another frozen backbone, more independent repositories, or a better-calibrated action-value model—not immediately modifying transformer plasticity. Internal attention/depth/memory regulation would require its own compute-matched baselines, training protocol, and causal attribution. No result in this plan establishes that extension.

If no differentiated direction survives, a well-controlled finding that a simple intervention-value controller suffices is useful. Preserve the original objective—reliable long-running behavior—while dropping unnecessary biological or architectural claims.

## 11. Audit findings, revisions, and residual risks

The following are **self-audits of this plan**, not independent review and not executed validation of a completed agent. The revisions below are already incorporated. The coding agent must repeat the audits against the implementation and evidence.

### 11.1 Scientific and prior-art audit

| Defect or challenge found | Revision made | Residual risk / acceptance criterion |
|---|---|---|
| Causal intervention value was initially a tempting novelty claim, but direct prior art exists | Make CINC-style control mandatory; classify H1 as replication | A later exact equivalent may narrow the claim further; update the prior-art record before publication |
| Persistent forecast-supervised control resembles established predictive-state methods | Add GRU-Q and generic predictive-state controls; explicitly deny established architecture novelty | Selected targets may have no special value; C3 must isolate any increment |
| Named health axes imply identifiable internal causes without evidence | Use a learned state and observable event forecasts; retain semantic axes only as diagnostics | Good prediction still does not identify an internal cognitive condition |
| Broad biological language overstates what a coding wrapper establishes | Exclude consciousness, self-preservation, and internal-transformer claims | Functional regulation remains a task-level operational interpretation |
| Recent references in the idea obscure earlier priority | Record 2014 homeostatic RL, 2017/2018 predictive-state work, and earlier preprints | Search coverage is not proof of global novelty |

### 11.2 Experimental-validity audit

| Defect or alternative explanation | Revision made | Acceptance criterion |
|---|---|---|
| Future intervention previews could enter features | Prefix-only allowlist and public/private types; explicit no-preview configuration | Sentinel data absent from all prompts, tensors, and decision views |
| Changed patches could inherit cached successful grades | Patch-specific unique grader attempt IDs | Gold, no-patch, and bad-patch tests produce independently checked outcomes |
| A maximum over noisy fork outcomes exaggerates headroom | Repeated arm means and independent/cross-fitted selection evaluation | Headroom uncertainty reported; no single-rollout oracle claim |
| One-shot suffix values may fail under repeated intervention | Separate estimands, closed-loop evaluation, training-only coverage expansion | Primary endpoint measured in actual repeated-policy episodes |
| More information, parameters, or tuning could explain a win | Shared ledger/actions, matched GRU-Q, generic targets, complete tuning ledger | General improvements shared; parameter/data/cost differences disclosed |
| Difficulty prediction can masquerade as deterioration detection | Task-only and within-task controls | Performance cannot be supported solely by pooled AUROC |
| Rollback might refund time or erase adverse evidence | Separate full fork restoration from code-only online rollback | Budget, environment clock, ledger, and recurrent state remain monotonic |
| A deadline reached before “done” could mis-score correct work | Grade the last completed artifact, not the final text declaration | A correct-on-time artifact passes even when the agent runs out of time |
| Synthetic generator shortcuts could explain gains | Hold out generator rules and natural unperturbed issues | C4 requires independent natural evidence |
| Post-treatment step counts could create misleading horizon effects | Define strata from task construction/development estimates | Primary horizon strata fixed before evaluated behavior |

### 11.3 Implementation and feasibility audit

| Defect or execution scenario | Revision made | Blocking check |
|---|---|---|
| Unknown memory was implicitly treated as a maximum-spec M3 Max | Hardware discovery and measured model/context selection | No large run before memory, swap, and disk pilot |
| A late source check located released monitor code and stronger hybrids | Replace the provisional reconstruction-only baseline with upstream fixture reproduction and validated adaptation | Hybrid and channel-map compatibility tests; no unavailable-code assertion remains |
| Nominal Apple support was mistaken for a compatible model/cache path | Inspect hybrid cache implementation; test target-Mac generation/restoration | Correct local tokenization/cache behavior and deadline cancellation |
| CUDA/vLLM or API assumptions could enter the local plan | Host MLX adapter, CPU controller, no remote calls | Network/API audit and native execution test |
| Dataset images may be x86-only | Native ARM builds with per-task gold/no-patch validation | No unvalidated emulation-based result represented as native performance |
| Old SWE-smith monolithic asset could be chosen | Use inspected language-specific Python asset and pin its revision | Schema/patch semantics validated before collection |
| Checkpoints omitted untracked files, RNG, or environment clock | Full snapshot manifests and atomic transaction journal | Restart and cross-branch isolation tests |
| CPU smoke testing might be reported as M3 evidence | Explicit environment/result boundary in Section 6.7 | M3 measurements remain a prerequisite, not an assumed fact |
| A fresh coding agent could confuse proposed commands with upstream commands | Label all `python -m pch` commands as to-be-implemented; separate verified CLI examples | Clean-checkout execution walkthrough succeeds |

### 11.4 Final challenge pass and unresolved uncertainty

The final review checked the claim-to-baseline mapping, action/budget semantics, private-label boundary, target dimensions, stage dependencies, failure paths, and the distinction between measured checks and proposed experiments. Further engineering issues may emerge; this review is not a guarantee of correctness.

Unresolved scientific questions are the amount of intervention headroom, whether the observable stream supports useful persistent state, whether auxiliary targets outperform generic predictive learning, and whether any gain transfers to natural software tasks. Unresolved execution questions are exact target-Mac model/cache behavior, native task eligibility, current per-asset licensing, and measured throughput/sample cost. Each has an explicit earlier gate rather than an assumed answer.

The strongest published full training/handoff systems may not be reproducible locally. This limits the claim to the strongest **credible locally executable** comparisons, not superiority over every contemporary agent. A broader claim requires an appropriately resourced, separately controlled experiment—not incomparable published leaderboard numbers.

## 12. References and search-coverage record

### 12.1 How to read the evidence record

References use ordinary Markdown links and stable identifiers. For live repositories, model cards, and documentation, the date is **accessed 12 October 2026** unless a publication date is given. A mutable `main` URL records the inspected source location, not a frozen experimental dependency: the coding agent must capture an immutable commit before reuse. “HTML inspected” means the paper text/method sections were opened; it does not mean every appendix claim was independently reproduced. PDF diagrams/equations used in the review were also inspected through rendered pages. No cited benchmark score is represented as an experiment executed for this project.

### 12.2 Closest methods, theory, and challenging evidence

- **CINC — Chubin Zhang, Zhenglin Wan, Xingrui Yu, Jingxuan Wu, Qi Wen, Pengfei Zhou, Wangbo Zhao, and Ivor Tsang.** *Calibration Is Not Control: Intervention Value for LLM-Agent Oversight.* arXiv:2606.21399v2; first 19 June 2026, revised 6 October 2026. [Paper][CINC]. Abstract and HTML methods inspected. Direct overlap in intervention-value supervision; principal novelty constraint.
- **CINC-CODE — Zhang and collaborators.** *calibration-is-not-control*, author repository, MIT. [Repository][CINC-CODE]. README/license and source organization inspected; not executed. Reusable small controller logic, not a claim of reproducing the published agent/handoff setup.
- **CINC-SRC — Zhang and collaborators.** `src/oversight_branching/policy_learning.py`. [Source][CINC-SRC]. Source inspected for forest settings, action selection, and preview-feature paths. Informs the prefix-only feature contract.
- **PSD — Arun Venkatraman, Nicholas Rhinehart, Wen Sun, Lerrel Pinto, Martial Hebert, Byron Boots, Kris M. Kitani, and J. Andrew Bagnell.** *Predictive-State Decoders: Encoding the Future into Recurrent Networks.* NeurIPS 2017; arXiv:1709.08520, 25 September 2017. [Record][PSD]. Abstract inspected; sufficient to establish the prior future-observation auxiliary-loss mechanism, not a full replication recipe.
- **RPSP — Ahmed Hefny, Zita Marinho, Wen Sun, Siddhartha Srinivasa, and Geoffrey Gordon.** *Recurrent Predictive State Policy Networks.* ICML 2018, PMLR 80; arXiv:1803.01489. [Proceedings][RPSP]. Paper text and rendered PDF pages 1 and 4 inspected. Action-conditioned predictive recurrent state is an important antecedent.
- **PSR — Michael L. Littman, Richard S. Sutton, and Satinder Singh.** *Predictive Representations of State.* NeurIPS 2001. [Proceedings abstract][PSR]. Abstract inspected. Supports historical priority for state defined through predictions of future tests.
- **MUSE — Rodolfo Valiente and Praveen K. Pilly.** *MUSE: Competence-Aware AI Agents with Metacognition for Unknown Situations and Environments.* arXiv:2411.13537v2; first 20 November 2024, revised 17 November 2025. [HTML][MUSE]. LLM assessment/training sections and algorithms inspected. The linked journal page was inaccessible; no conclusion depends on its reported February 2026 date.
- **BRACE — Joykirat Singh, Zaid Khan, Archiki Prasad, Justin Chih-Yao Chen, Akshay Nambi, Hyunji Lee, Elias Stengel-Eskin, and Mohit Bansal.** *Agent-BRACE: Decoupling Beliefs from Actions in Long-Horizon Tasks via Verbalized State Uncertainty.* arXiv:2605.11436, 12 May 2026. [Paper][BRACE]. HTML methods and abstract inspected. Closest structured uncertainty-belief comparator.
- **BRACE-CODE — Agent-BRACE authors.** *Agent-BRACE*, author repository. [README][BRACE-CODE]. Training/data-generation requirements inspected; involves external model data generation and multi-GPU training. No ready trained-checkpoint download was identified in the inspected README. This limits faithful local reproduction, not the feasibility of the prompt-based control.
- **IP — Rakshith Vasudev, Melisa Russak, Dan Bikel, and Waseem Alshikh.** *The Intervention Paradox: Accurate Failure Prediction in Agents Does Not Imply Effective Failure Prevention.* arXiv:2602.03338, 3 February 2026. [HTML][IP]. Experimental text inspected. Negative and environment-dependent intervention outcomes motivate harm accounting.
- **DIFF — Mohsen EsfandyariDoulabi, Lawrence Arkoh, Biruk Tadesse, Vaishvi Patel, Mehul Sharma, Marcelo d'Amorim, and Wesley Assunção.** *Disentangling Task Difficulty from Run-Level Failure in Agent Failure Prediction.* arXiv:2610.05572, 4 October 2026. [Paper][DIFF]. Abstract and HTML inspected. Informs within-task controls and limitations of pooled prediction metrics.
- **MON — Sunny Dubey.** *Real-Time Detection and Repair of LLM Agent Failures.* arXiv:2608.02464v1, 3 August 2026. [Paper][MON]. HTML methods/results inspected. Relevant cheap monitoring, wrapper ablations, and repair evidence; settings differ from coding tasks here.
- **MON-CODE — Sunny Dubey.** *AgentTrajectorySentinel*, Apache-2.0 author repository. [Repository][MON-CODE]. README, baseline descriptions, and source organization inspected in the final source audit. Released implementation supersedes the earlier provisional assumption that only a mathematical reconstruction was available.
- **MON-ESN — Sunny Dubey.** `derail/monitor/esn.py`. [Source][MON-ESN]. Reservoir, channel-max, and streaming conventions inspected. Used for faithful component reuse and tests before coding-domain adaptation.
- **MON-HYBRID — Sunny Dubey.** `derail/monitor/hybrid.py`. [Source][MON-HYBRID]. Score normalization, degenerate-scale handling, hybrid selection, and channel-width assumptions inspected. No hybrid superiority is assumed for this task.
- **MON-REPRO — Sunny Dubey.** *Reproduction record*, `REPRODUCE.md`. [Record][MON-REPRO]. Environment and core/offline dependency path inspected. Upstream measurements were not on an M3 Max; the live tree and arXiv-v1 artifact populations differ, so reproduction must pin the intended state.
- **SKILLAPT — Shuang Guo.** *SkillApt: Learning When to Activate Agent Skills from Counterfactual Evidence.* arXiv:2609.26863, 22 September 2026. [HTML][SKILLAPT]. Method text inspected. Counterfactual activation/abstention is adjacent direct overlap.
- **CRR — Yuanhao Li, Hongbo Wang, Xuhong Chen, Yiming Cao, and Xunzhu Tang.** *Counterfactual Rollout Replay: Forkable Environments as Free Process Rewards for Software Engineering Agents.* arXiv:2609.33875, 27 September 2026. [HTML][CRR]. Fork definitions and training method inspected. Different policy-training objective; counterfactual computation is not free for this project.
- **CFLOW — A. Bonagiri, D. Borkar, G. J. Anderias, S. Rafatirad, and H. Homayoun.** *CausalFlow: Causal Attribution and Counterfactual Repair for LLM Agent Failures.* arXiv:2605.25338, 25 May 2026. [HTML][CFLOW]. Method text inspected. Retrospective attribution/repair reference rather than the primary prospective controller.
- **FIRM — Yilun Wu, Yunjian Zhang, Aobo Li, Mujiangshan Wang, Haitao Wu, and Aqiang Zhang.** *FIRM-WM: State-factorized factual-interventional recurrent modeling for reward-free visual planning.* arXiv:2609.22816, 19 September 2026. [HTML][FIRM]. Abstract and main method inspected. Adjacent recurrence/interventional-state factorization; different visual-planning setting.
- **TWM — Fangxin Wang, Xiang Gao, Yuguang Yao, Kaiwen Dong, Nikash Walia, and Kamalika Das.** *Long-Horizon Textual World Modeling through Structured Reasoning.* arXiv:2610.06637, 5 October 2026. [HTML][TWM]. Main method inspected. Structured multi-step state prediction is relevant context, not proof of closed-loop intervention value.
- **META — C. Nicolò De Sabbata, Theodore R. Sumers, and Thomas L. Griffiths.** *Rational Metareasoning for Large Language Models.* arXiv:2410.05563, October 2024. [HTML][META]. Core framework inspected. Expected benefit relative to computation cost is established framing.

### 12.3 Interoception, historical framing, and adjacent concepts

- **HRL — Mehdi Keramati and Boris Gutkin.** *Homeostatic reinforcement learning for integrating reward collection and physiological stability.* eLife 3:e04811, 2 December 2014. DOI:10.7554/eLife.04811. [Official PDF][HRL]. Paper text and rendered pages 1 and 4 inspected; the normal publisher page was access-limited. Establishes substantial prior homeostatic-control theory.
- **LIAI — Sungwoo Lee, Younghyun Oh, Hyunhoe An, Hyebhin Yoon, Karl Friston, Seok-Jun Hong, and Choong-Wan Woo.** *Life-inspired interoceptive artificial intelligence for autonomous and adaptive agents.* Nature Machine Intelligence, 26 August 2026. DOI:10.1038/s42256-026-01296-8. [Journal page][LIAI]. Abstract/preview and references inspected; full text was paywalled. No detailed architectural claim is attributed to inaccessible sections.
- **LIAI-PRE — Lee and collaborators.** Earlier preprint of the interoception proposal, arXiv:2309.05999, first September 2023. [Preprint record][LIAI-PRE]. Abstract inspected to establish earlier priority.
- **IMF — Diego Candia-Rivera.** *Interoceptive machine framework: Toward interoception-inspired regulatory architectures in artificial intelligence.* arXiv:2604.24527, 27 April 2026. [Record][IMF]. Abstract inspected. Additional conceptual overlap, not empirical support for PCH-R.
- **AUTONOMIC — Alfred Z. Spector / IBM Research.** *Challenges and opportunities in autonomic computing.* 22 June 2002. [Primary publication record][AUTONOMIC]. Abstract inspected. Historical software self-management context; not an exact neural-controller comparator.
- **PATENT — Published patent application US20150142082A1.** *Systems and methods of biofeedback using nerve stimulation.* 2015 publication. [Patent record][PATENT]. Description inspected during terminology screening. Human biofeedback/nerve stimulation, not a matching LLM regulator; no legal clearance conclusion.

### 12.4 Agent, data, model, and platform assets

- **UH — Haotian Luo and collaborators.** *UltraHorizon*, arXiv:2509.21766, first 26 September 2025. [Record][UH]. Abstract inspected. Long-horizon motivation only; not a production coding or interoception validation.
- **REACT — Shunyu Yao and collaborators.** *ReAct: Synergizing Reasoning and Acting in Language Models.* arXiv:2210.03629; first 6 October 2022, ICLR 2023. [Record][REACT]. Abstract inspected; baseline interaction pattern.
- **REFLEXION — Noah Shinn and collaborators.** *Reflexion: Language Agents with Verbal Reinforcement Learning.* arXiv:2303.11366; first 20 March 2023. [Record][REFLEXION]. Abstract inspected; reflection-memory baseline context.
- **MINI — SWE-agent team.** *mini-SWE-agent*, MIT repository, current v2 documentation. [Repository][MINI]. README and implementation overview inspected. Reusable practical scaffold; published frontier-model performance is not a comparable local number.
- **MINI-SRC — SWE-agent team.** `src/minisweagent/agents/default.py`. [Source][MINI-SRC]. Proposal/execution boundaries, saving, and limit accounting inspected. Motivates the common intervention hook and separate hard budget.
- **SMITH — SWE-bench research team.** *SWE-smith: Scaling Data for Software Engineering Agents.* arXiv:2504.21798, April 2025. [HTML][SMITH]. Task-generation and environment discussion inspected. Generated-task provenance, not natural-issue evidence.
- **SMITH-DATA — SWE-bench team.** `SWE-bench/SWE-smith-py`, Python dataset, card labeled MIT. [Dataset][SMITH-DATA]. Current card/schema and example image fields inspected. Source repository licenses and patch semantics still require per-task checking.
- **VERIFIED — Princeton NLP / SWE-bench team.** `princeton-nlp/SWE-bench_Verified`. [Dataset card][VERIFIED]. Card inspected. Candidate natural issues; this research did not conclusively establish all task-content licensing from that card alone.
- **SWE-README — SWE-bench team.** Current harness README and v5 command documentation. [README][SWE-README]. Native-build guidance, CLI examples, experimental ARM warning, and cache key inspected. Local gold/bad-patch validation is mandatory.
- **SWE-LICENSE — SWE-bench contributors.** Harness MIT license. [License][SWE-LICENSE]. License text inspected. Applies to the toolkit; not blanket permission for all underlying repositories or issue material.
- **MLX-LM — MLX / Apple ML Research contributors.** *MLX-LM*, MIT repository. [Repository][MLX-LM]. README, API examples, conversion, and memory-related options inspected. No generation was executed on Apple hardware in this investigation.
- **MLX-INSTALL — MLX documentation.** *Installation.* [Documentation][MLX-INSTALL]. Platform/Python requirements inspected; resolve and lock the actual compatible versions locally.
- **MLX-SYNC — MLX documentation.** `mlx.core.synchronize`. [API reference][MLX-SYNC]. Synchronization behavior inspected for timing boundaries.
- **MLX-MEM — MLX documentation.** `mlx.core.get_peak_memory` and linked memory APIs. [API reference][MLX-MEM]. Counters inspected; they are not a complete host-plus-VM memory census.
- **Q35 — Qwen team.** `Qwen/Qwen3.5-9B`, model card, Apache-2.0 label. [Card][Q35]. Card/configuration inspected. Compact current candidate; exact selected quantization and local behavior remain gates.
- **Q35-SRC — MLX-LM contributors.** `mlx_lm/models/qwen3_5.py`. [Source][Q35-SRC]. Hybrid attention/recurrent cache and model sanitization paths inspected. Important compatibility and memory constraint.
- **Q25 — Qwen team.** `Qwen/Qwen2.5-Coder-7B-Instruct`, model card, Apache-2.0 label. [Card][Q25]. Card inspected. Established coding anchor, not a claimed current global best model.
- **Q25-CONFIG — Qwen team.** Qwen2.5-Coder-7B-Instruct `config.json`. [Configuration][Q25-CONFIG]. Layer/head/context fields inspected; basis for the transparent cache arithmetic.
- **Q30 — Qwen team.** `Qwen/Qwen3-Coder-30B-A3B-Instruct`, model card, Apache-2.0 label. [Card][Q30]. Card inspected. Conditional larger-memory candidate, not a mandatory asset.
- **COLIMA — Colima contributors.** *Colima: container runtimes on macOS and Linux.* [Repository][COLIMA]. README inspected. Candidate local Linux-VM/container runtime; sandbox policy must still be implemented and tested.

### 12.5 Coverage, stopping rationale, and remaining blind spots

Search phrases included variants of “artificial interoception,” “homeostatic reinforcement learning,” “competence-aware agent self-regulation,” “agent failure prediction intervention harm,” “intervention value counterfactual oversight,” “predictive-state recurrent control auxiliary prediction,” “forkable software-agent environments,” “uncertain belief state long horizon,” “MLX Qwen cache Apple Silicon,” and “SWE-bench ARM native build.” Historical autonomic-computing and patent terminology were also screened.

Citation tracing identified direct equivalents under different names: intervention-value learning rather than failure detection, and predictive-state learning rather than cognitive-health prediction. Later targeted searches added adjacent counterfactual skill activation and state-factorized world models, but did not justify a broader novelty claim. The final implementation-source audit found the released monitor repository and strengthened the baseline set; the earlier “code not located” draft conclusion was removed.

This is a coverage-based stopping point, not a paper-count quota: the load-bearing mechanisms, strongest local alternatives, major negative evidence, and critical hardware/data paths have identifiable references and discriminating tests. Blind spots remain in inaccessible full text, private/deployed systems, unpublished work, model-selection breadth, complete patent coverage, and operations not executed on the actual laptop. Recheck only facts and release changes that materially affect the selected experiment before freezing its dependencies.

**Final operating instruction:** begin with WP00 and WP01. Do not download the largest model, collect a large branch dataset, or announce a novel architecture before the corresponding gates pass. Preserve every valid negative result. The project's purpose is to discover a real improvement—or a well-supported reason to prefer an existing method—not to validate a name.

[CINC]: https://arxiv.org/abs/2606.21399v2
[CINC-CODE]: https://github.com/bennidict23/calibration-is-not-control
[CINC-SRC]: https://raw.githubusercontent.com/bennidict23/calibration-is-not-control/main/src/oversight_branching/policy_learning.py
[PSD]: https://arxiv.org/abs/1709.08520
[RPSP]: https://proceedings.mlr.press/v80/hefny18a.html
[PSR]: https://papers.nips.cc/paper_files/paper/2001/hash/1e4d36177d71bbb3558e43af9577d70e-Abstract.html
[MUSE]: https://arxiv.org/html/2411.13537v2
[BRACE]: https://arxiv.org/abs/2605.11436
[BRACE-CODE]: https://raw.githubusercontent.com/joykirat18/Agent-BRACE/main/README.md
[IP]: https://arxiv.org/html/2602.03338
[DIFF]: https://arxiv.org/abs/2610.05572
[MON]: https://arxiv.org/abs/2608.02464
[MON-CODE]: https://github.com/sunnydubey1111/agent-trajectory-sentinel
[MON-ESN]: https://raw.githubusercontent.com/sunnydubey1111/agent-trajectory-sentinel/main/derail/monitor/esn.py
[MON-HYBRID]: https://raw.githubusercontent.com/sunnydubey1111/agent-trajectory-sentinel/main/derail/monitor/hybrid.py
[MON-REPRO]: https://raw.githubusercontent.com/sunnydubey1111/agent-trajectory-sentinel/main/REPRODUCE.md
[SKILLAPT]: https://arxiv.org/html/2609.26863v1
[CRR]: https://arxiv.org/html/2609.33875v1
[CFLOW]: https://arxiv.org/html/2605.25338v1
[FIRM]: https://arxiv.org/html/2609.22816v1
[TWM]: https://arxiv.org/html/2610.06637v1
[META]: https://arxiv.org/html/2410.05563v1
[HRL]: https://cdn.elifesciences.org/articles/04811/elife-04811-v1.pdf
[LIAI]: https://www.nature.com/articles/s42256-026-01296-8
[LIAI-PRE]: https://arxiv.org/abs/2309.05999
[IMF]: https://arxiv.org/abs/2604.24527
[AUTONOMIC]: https://research.ibm.com/publications/challenges-and-opportunities-in-autonomic-computing
[PATENT]: https://patents.google.com/patent/US20150142082A1/en
[UH]: https://arxiv.org/abs/2509.21766
[REACT]: https://arxiv.org/abs/2210.03629
[REFLEXION]: https://arxiv.org/abs/2303.11366
[MINI]: https://github.com/SWE-agent/mini-swe-agent
[MINI-SRC]: https://raw.githubusercontent.com/SWE-agent/mini-swe-agent/main/src/minisweagent/agents/default.py
[SMITH]: https://arxiv.org/html/2504.21798v1
[SMITH-DATA]: https://huggingface.co/datasets/SWE-bench/SWE-smith-py
[VERIFIED]: https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified
[SWE-README]: https://raw.githubusercontent.com/SWE-bench/SWE-bench/main/README.md
[SWE-LICENSE]: https://raw.githubusercontent.com/SWE-bench/SWE-bench/main/LICENSE
[MLX-LM]: https://github.com/ml-explore/mlx-lm
[MLX-INSTALL]: https://ml-explore.github.io/mlx/build/html/install.html
[MLX-SYNC]: https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.synchronize.html
[MLX-MEM]: https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.get_peak_memory.html
[Q35]: https://huggingface.co/Qwen/Qwen3.5-9B
[Q35-SRC]: https://raw.githubusercontent.com/ml-explore/mlx-lm/main/mlx_lm/models/qwen3_5.py
[Q25]: https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct
[Q25-CONFIG]: https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct/raw/main/config.json
[Q30]: https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct
[COLIMA]: https://github.com/abiosoft/colima
