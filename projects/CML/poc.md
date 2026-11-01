# Counterfactual Memory Learning: Research and Local Proof-of-Concept Plan

**Research date:** 13 October 2026  
**Execution target:** one MacBook Pro with an M3 Max; actual unified memory, storage, operating system, and available acceleration must be discovered.  
**Deliverable type:** standalone research handoff, including executable specifications and small checks already performed.  
**Status:** the algebra has been independently checked on a non-Apple CPU. No memory gate has been trained for this report, no learning advantage has been measured, and no M3 Max benchmark has been run.

## 1. Executive assessment and selected direction

**Proceed with a restricted, exact finite-intervention experiment—not a claim of a new three-agent architecture.** The useful research question is whether compact, exact accept-versus-reject effects improve the quality or cost of learning binary memory admission, compared with strong differentiable and discrete-policy alternatives.

The broad idea is substantially preceded. Learned memory retention, delta-rule fast weights, eligibility traces, and counterfactual memory objectives already exist. Particularly close September 2026 work includes MGPO and **LatentHarness**, which was absent from the supplied proposal. The latter's write-credit equation contains the same transported key-product structure, but uses a gradient-based credit signal. This substantially narrows the possible contribution; Section 3 distinguishes the overlaps precisely. [R1] [R2] [R3]

The selected implementation is **Trace-CF**, with two explicitly different training variants:

- **Trace-PG, primary:** use finite paired losses in a standard local-expectation gradient for a binary write policy. The gradient identity is established statistical machinery; the research contribution, if any, is its efficient realization and measured usefulness in evolving matrix memory.
- **Trace-Reg, companion:** retain the proposal's utility-prediction-and-thresholding approach, with an unbounded utility head and clearly specified target. This is necessary to determine whether regression itself helps or becomes the bottleneck.

Both use the same exact rank-one trace. Neither requires an auditor at inference. The primary audit covers the remaining episode; the proposed 64-event horizon is a secondary, deliberately truncated condition. Full-horizon sparse sampling removes one avoidable ambiguity about what training objective is being optimized.

**Most important caveat:** a fixed-control intervention is not the total effect of a write when that write changes subsequent gate actions, representations, or generated inputs. A concrete sign-reversal example appears in Section 4.10. The primary architecture must enforce memory- and action-independent future control features, rather than merely describe them that way.

A second limitation follows from the proposed architecture itself: with independent binary gates and logits affine in the memory read, replacing sampled actions by their probabilities exactly reproduces the expected memory and cannot increase expected cross-entropy. Section 4.11 proves and checks this. A positive result must therefore be an optimization-at-cost or measured write-cost result, not a claim that stochastic binary memory is intrinsically a better predictor than its fractional counterpart.

### 1.1 What would constitute a contribution?

A defensible positive result would combine: numerically faithful finite intervention estimates; a favorable accuracy–training-cost tradeoff against the best validated local alternatives; attribution to finite delayed credit rather than extra information or weak baselines; and survival of a predeclared distribution shift. A faster exact auditor with no learning advantage is a narrower engineering result. A careful negative result identifying horizon bias, lack of predictable headroom, or action-feedback failure is also useful, but is not evidence of superior memory learning.

No fixed percentage gain is a success threshold. The original proposal's provisional 3% hurdle is replaced by uncertainty estimates and measured Pareto tradeoffs. There is no evidence yet for improvements in general reasoning, large-model pretraining, or autonomous exploration.

### 1.2 Explicit changes from the original proposal

| Change | Reason | Preserved intent |
|---|---|---|
| Add Trace-PG alongside utility regression | Avoid making a learned utility regressor the only route from exact effects to better decisions | Learn memory admission from delayed consequences |
| Use a uniform sample of anchors with remaining-episode audits as primary | Avoid unaccounted horizon truncation and trace-cap selection bias | Compact audits, not a dense branch per candidate |
| Keep 64-event audits as a sensitivity experiment | Test whether the original inexpensive setting is sufficient | Investigate the original finite-horizon proposal faithfully |
| Include derivative credit, DisARM, continuous BPTT, modern memory recurrences, and exact dictionaries | A win against all-write or an untuned score-function estimator would not establish value | Seek genuine improvement over strong alternatives |
| Start with a fixed representation, then require a learned-representation stage for broader claims | Separate credit-estimator behavior from representation nonstationarity | Preserve slow learning plus fast memory plus gate learning |
| Make a real-text stage conditional | Synthetic validity is cheaper to establish; text introduces additional prerequisites | Scale only after evidence justifies it |

## 2. Problem, scope, and hypotheses

### 2.1 Intended task and beneficiaries

The target is an online predictor receiving a fixed stream of observations and later queries. Its matrix memory is limited; writing one association can interfere with another. The intended beneficiaries are designers of recurrent or externally augmented predictors that need rapid adaptation without retraining a large backbone after each observation.

The existing workflow is to learn representations and continuous memory gates through prediction loss, or to manage storage using recency, frequency, novelty, confidence, or replacement rules. The hypothesized bottleneck is **poor or expensive credit assignment for discrete writes**, not an established inability of current architectures to learn useful memory. Whether this bottleneck matters at the tested scale is itself an experiment.

The core claim is local: **on specified online associative-prediction workloads, exact finite write effects can be collected and used economically enough to improve memory-policy learning.** It is not a claim about all memory systems, a new universal learning principle, or the necessity of three components.

### 2.2 Inputs, outputs, and objective

An episode contains an exogenous event sequence of length `T`. Each event has a type, an observable payload, an optional query, and a target with an explicit feedback-release time. The learner sees only fields allowed at that time. Candidate write actions are binary. Query predictions occur before the event's memory update.

For a fixed episode with `Q > 0` scored queries, define loss in **nats per query**:

\[
L(a_{1:T};\theta)=\frac{1}{Q}\sum_{s\in\mathcal Q}\operatorname{CE}(z_s,y_s).
\]

The training objective is `J(theta, psi) = E[L]`, over exogenous episodes and policy randomness. Zero-query episodes are disallowed in the main generator; unit tests separately support a zero-query episode and define its training contribution as zero. Query count and target availability cannot depend on gate actions.

The primary scientific evaluation measures expected query loss under the learned stochastic policy, with auditing disabled. Deterministic thresholded evaluation is a required deployment assessment, reported separately. A stochastic-policy improvement does not establish a deterministic-policy improvement. Continuous-gate competitors are evaluated in their native continuous mode as well as, where relevant, a separately tuned binary deployment mode.

### 2.3 Established facts, assumptions, and hypotheses

The existence of delta memories, learned gates, discrete gradient estimators, and memory-management policies is prior art. Exact rank preservation under the particular affine recurrence follows from the derivation below and was checked numerically. Neither implies that historical write utility is predictable, that binary admission is the right action space, or that auditing is cost-effective.

| Hypothesis | Supporting rationale, not a demonstrated gain | Credible alternative explanation | Observation that weakens or falsifies it |
|---|---|---|---|
| H1: writes have decision-relevant delayed effects not captured by cheap features alone | Interference and changing facts can make immediate novelty misleading | A tuned reliability/recency rule or stronger memory update solves the task | Restricted, cross-fitted policies and strong heuristics exhaust attainable headroom |
| H2: pre-write observable features predict useful decisions | Recurrence cues and noisy source-quality cues can carry information | A constructed cue simply labels the answer | Gain vanishes with fair cue access, held-out codebooks, or sensible shifts; impossible-information control behaves incorrectly |
| H3: finite effects improve learning over cheaper gradient signals | A secant can differ in sign from a tangent through nonlinear loss | Smooth gates can exploit partial writes and avoid the binary problem | Full BPTT, DisARM, or derivative credit matches or beats Trace-CF at matched cost |
| H4: compact tracing saves material resources | A rank-one state difference needs two vectors | Reader scoring, framework launch overhead, or slow-model replay dominates | No favorable end-to-end frontier despite smaller trace storage |
| H5: utility remains useful when the slow model learns | Fresh within-snapshot targets avoid obvious staleness | Moving representations erase the relationship learned by the gate | Fixed-model gains disappear under controlled joint learning |
| H6: independence restrictions retain enough useful behavior | Output-only memory can affect predictions while leaving future controls fixed | The best gate requires access to the evolving memory | Memory-conditioned baselines dominate and the restricted architecture has little remaining headroom |

### 2.4 Information limits and headroom

A future query that is independent of every observable feature cannot be predicted by a better credit estimator. Likewise, observationally indistinguishable corruptions and genuine changes cannot reliably be separated at write time. The environment must contain some imperfect predictive structure, but must not reveal a hidden `useful`, `corrupt`, `will_be_queried`, or `true_change` flag to the gate.

Three headroom diagnostics must be distinguished:

1. **Clairvoyant intervention oracle:** uses realized future effects; an optimistic diagnostic, not an achievable policy or an optimal joint sequence of actions.
2. **Cross-fitted restricted policy:** learns from dense effects on training episodes using only permitted `h_t`, then is evaluated on new episodes. This estimates whether useful information is learnable, not merely present retrospectively.
3. **Best ordinary predictor/system:** continuous gates, stronger recurrences, or dictionaries may eliminate the bottleneck without counterfactual learning.

Do not infer learnable headroom from an oracle that sees future labels. Do not add all marginal benefits and call the result an achievable combined improvement: write effects interact.

## 3. Research findings and novelty assessment

### 3.1 Closest-prior-art comparison

The table separates a comparable local component implementation from a reproduction of an entire published system. Published results at different model sizes and tasks are context, not controlled scores against this PoC.

| Prior work | Problem and mechanism | Exact overlap / important difference | Evidence, implementation, and local treatment |
|---|---|---|---|
| MGPO, 2026 [R1] | Textual memory writer trained from downstream counterfactual utility | Long-horizon memory credit already exists. It compares adjacent textual memory states and uses return construction, rather than transporting a single delta write through later shared updates | Inspected formulation and cost discussion. Reported large writer/reader configuration is not the core laptop comparator; any small adaptation must be called MGPO-inspired |
| LatentHarness, 2026 [R2] | Counterfactual latent action training and gain-credited delta-memory writing | Equation 10 factors write credit into residual content, later loss gradient, and a product transporting key relevance. This is very close to the proposal's transport geometry. Finite binary loss differences remain a distinct testable choice | Inspected write-credit equations and experimental sections. Reproduce the derivative-credit component locally, not the entire latent reasoning model |
| HiMPO, 2026 [R3] | Hindsight relevance filtering for memory-policy credit | Counterfactual pre/post-memory likelihood and memory policy optimization are known; no identified exact rank-one evolving-state audit | Inspected method equations and scope. Contextual reference, not an equal-task published baseline |
| DeltaNet, 2024 [R4] | Delta-rule fast weights; products of low-rank update factors and parallel execution | The memory recurrence and its affine algebra are established, not inventions here | Use direct recurrence and reference parity; optimized chunking is a fairness consideration |
| Gated DeltaNet, 2024 [R5] | Decay plus delta correction | Contains the basic scalar-gated memory family under study | Mandatory tuned recurrence/continuous-gate baseline |
| Kimi Linear / KDA, 2025 [R6] | Channel-wise decay and delta attention | Stronger control of forgetting can compete with selective binary writes | Implement the specified single-block recurrence locally; larger published models are reference-only |
| Gated DeltaNet-2, 2026 [R7] | Separate channel-wise erasure and writing | More expressive memory updates may solve apparent admission problems | Mandatory small recurrence adaptation before claiming a system-level advantage; not a full language-model reproduction |
| Preconditioned DeltaNet, 2026 [R8] | Curvature-aware recurrence for correlated keys | Addresses interference through update geometry, an alternative to rejecting writes | Inspected method structure and synthetic scope. Include online ridge/RLS locally; add a faithful preconditioned implementation before making a broad claim against this family |
| Local expectation gradients; Rao–Blackwellized gradients [R9] [R10] | Integrate selected discrete variables conditional on others | The exact binary expectation identity used in Trace-PG is established | Reuse the identity; contribution must be domain-specific computation or evidence |
| ARM; DisARM [R11] [R12] | Coupled binary trajectories with unbiased gradient estimates | Strong, inexpensive alternatives to one-action counterfactual enumeration | Mandatory two-trajectory baselines, verified against exact small enumeration |
| UORO [R13] | Unbiased rank-one approximation to a recurrent state–parameter Jacobian | Rank-one recurrent credit is known, but its object is a randomized Jacobian estimate, not an exact finite state difference | Reference-only unless making a general online-gradient claim; full BPTT is locally tractable |
| e-prop [R14] | Eligibility traces combined with learning signals | Delayed trace-based credit is established; finite intervention loss is different | Conceptual reference. Do not force a spiking-network benchmark into an associative task |
| Differentiable plasticity [R15] | Slow learning controls fast adaptation | Multiple learning timescales and learned plasticity are established | Motivates an ordinary end-to-end comparator; component counting provides no novelty |
| Episodic Memory Reader, 2019 [R16] | Learned retention for streaming inputs and later questions | Learning what to keep for unseen future queries is already a task and policy formulation | Inspected paper; original repository is linked by authors but not executed here. Local bounded-buffer policies are system alternatives |
| Cache replacement / Hawkeye [R17] | Learn retention behavior from access history and later-known reuse | Cache retention based on future reuse is an adjacent historical comparison | Author publication record inspected, not a full implementation audit. Use cache baselines; do not claim exact algorithmic equivalence |

### 3.2 What is different—and what remains unresolved

**Not novel:** three adaptive timescales; memory admission; the delta rule; rank-one linear algebra; future-utility objectives; conditioning a binary gradient on both possible actions; or using a later learning signal with a transported eligibility-like quantity.

**Plausibly differentiating combination:** sparse audits of an actual binary intervention, propagated exactly through continuing common-control memory updates, followed by finite downstream loss evaluation and cost-accounted policy learning. Its nearest discriminator is finite secant credit versus derivative credit using the same anchors, head, data, and memory recurrence.

**Potential engineering contribution:** caching a linear head's projection of each trace, so exact counterfactual cross-entropy can be computed without rerunning an expensive reader. This is an implementation optimization, not by itself a new learning principle.

**Potential empirical contribution:** characterize when finite effects outperform, match, or lose to differentiable and antithetic estimators, including the importance of horizon, interference, predictable cues, and feedback restrictions.

**Unresolved methodological extension:** a divergence-aware low-rank branch representation, described in Section 11, may extend exact auditing when future binary gates depend on memory. Its algebra has been checked, but its novelty, practical rank growth, and learning value have not been established.

Novelty would be invalidated by prior work that already combines the same finite evolving-state intervention, sparse exact transport, and comparable training use. A renamed equation, new implementation, or synthetic win is not sufficient to distinguish the method. Before publication, repeat forward citation and implementation searches for the September 2026 papers and record any newly released code. The current search is not proof of absence and is not a patentability or freedom-to-operate opinion.

### 3.3 Reuse rather than rebuild

Reuse PyTorch autograd and optimizers, NumPy float64 references, standard discrete estimators, and the equations in Section 6. Use FLA's naive recurrence as a cross-check rather than making its accelerator stack a dependency. Its inspected reference stores the transposed state, applies query scaling by default, casts arithmetic to float32, and emits a **post-write** read. Our query convention is pre-write. A direct unmodified call would therefore not implement the experiment. Set or remove scaling explicitly and compare aligned states/timelines. [R18]

Use synthetic data generated by this project for the mandatory path. No pretrained model, paid API, private dataset, or remote accelerator is required. The optional WikiText asset is separately gated by provenance and license reconciliation.

## 4. Mathematical and implementation specification

### 4.1 State, event order, and frozen controls

Use row-major mathematical state `M: [dv, dk]`; batched tensors are `[B, dv, dk]`. Keys and queries are `[B, dk]`, values `[B, dv]`. Normalize nonzero keys; reject malformed zero-norm keys in the generator. Initially use unit-norm queries with **no implicit attention scaling**.

At event `s`:

1. Reveal only the observable fields available before prediction.
2. Compute the pre-write query `r_s = M_{s-1} q_s`, produce logits, and score when the target is released.
3. Score old audit traces for this query using their **pre-update** trace vectors.
4. Compute the candidate write, obtain the factual action, and update the factual memory.
5. Propagate existing traces through this update.
6. Insert a newly selected trace for the current write. Never apply this event's transport to its own new trace twice.
7. Update raw-observation feature statistics for later events.

No-write query events use `a=0`, `beta=0`; decay still follows the declared event clock. The main synthetic condition applies the same scalar `lambda` on every event, including queries, and uses `lambda=1` in a clock-semantics sensitivity test. Reset memory only at declared episode/document boundaries.

Freeze `theta`, `psi`, dropout masks or dropout-disabled behavior, normalization statistics, and all other parameters during an audited episode. Evaluation mode alone is not a substitute for freezing adaptive buffers. All branches share exogenous input and control sequences. No trace may cross a model update or reset. The first implementation uses no dropout and no running-statistics normalization. The core generator releases each query label immediately after prediction. A delayed-feedback extension must either keep the relevant parameters unchanged until targets complete, or derive and validate an explicit off-policy/staleness treatment. Merely saving old logits does not make a later update at a changed policy unbiased.

### 4.2 Restricted feature contract

Primary gate features are a concatenation of: a 64-dimensional raw key representation; the current reported class one-hot vector of length 32 on observation events; two noisy observable cues; normalized log observation count; time since the key's last raw observation; and normalized episode position. This gives `F=101` at the anchor configuration. The count and last-seen timestamp are measured **before** incorporating the current observation.

Feature history is a function of the exogenous observations, not of accepted writes. For example, an observation rejected by the gate still increments the raw observation count. Count and timestamp arrays contain no label history and occupy `256 × 2 × 4 = 2,048` bytes as int32 arrays. Count those bytes as online state and provide the same statistics to competing methods.

Forbidden in the restricted controller: memory reads or residuals; factual prediction loss; accepted-write counts; a hidden state updated differently after accepted/rejected actions; branch-dependent query selection; or features generated from memory-conditioned model outputs. Future actions must remain identical when the audited write is flipped with the same random uniforms. Test this invariant programmatically.

A richer memory-conditioned baseline is allowed—and desirable—but belongs to a separate information/architecture group. An exact restricted-audit claim cannot quietly transfer to it.

### 4.3 Memory transition and exact trace

For fixed controls:

\[
P_s=\lambda_sM_{s-1},\quad u_s=\beta_s(v_s-P_sk_s),\quad
M_s=P_s+a_su_sk_s^\top.
\]

Equivalently,

\[
M_s=\lambda_sM_{s-1}(I-a_s\beta_s k_sk_s^\top)+a_s\beta_sv_sk_s^\top.
\]

Let `M^(1)` and `M^(0)` differ only by accepting/rejecting candidate `t`. Compute `u_t` from the shared **pre-decision** state even when the factual action is zero. Then

\[
M_t^{(1)}-M_t^{(0)}=u_tk_t^\top.
\]

Subtracting later affine transitions gives

\[
\Delta M_s=\lambda_s\Delta M_{s-1}(I-a_s\beta_sk_sk_s^\top).
\]

Thus `Delta M_s = u_t w_s^T`, with

\[
w_t=k_t,\qquad
w_s=\lambda_s\{w_{s-1}-a_s\beta_sk_s(k_s^\top w_{s-1})\}.
\]

At a later pre-write query:

\[
\Delta r_s=r_s^{(1)}-r_s^{(0)}=u_t(w_{s-1}^\top q_s).
\]

This is exact real-arithmetic closure for the specified recurrence, not a first-order approximation. The later delta residuals depend on their branch memories; subtracting the affine form already accounts for that dependence. It does **not** account for different later controls.

The primary audit engine supports this scalar-decay delta recurrence only. It must reject KDA/GDN2/RLS configurations unless a separately derived transport is implemented and tested; their transition factors are not interchangeable.

### 4.4 Signed reconstruction and targets

Store the factual anchor action `a_t` with each trace. For a factual query read `r_f`,

\[
r_{alt}=r_f+(1-2a_t)\Delta r_s.
\]

The beneficial-write increment is

\[
d_{t,s}=(2a_t-1)\{\ell(r_{alt},y_s)-\ell(r_f,y_s)\}.
\]

This sign convention audits both accepted and rejected writes. It does not condition the training set on accepting a write.

Primary full-episode effect:

\[
D_t=\frac1Q\sum_{s>t,\ s\in\mathcal Q}d_{t,s}
=L(a_t=0,a_{-t})-L(a_t=1,a_{-t}).
\]

The original finite-horizon target is retained separately:

\[
Y_t^{(H)}=\frac1H\sum_{t<s\leq\min(t+H,T),\ s\in\mathcal Q}d_{t,s}.
\]

Its denominator remains `H` near the end of the episode, and an empty window contributes zero. For a truncated approximation to `D_t`, divide the same truncated sum by `Q`, **not** by `H`. Store `target_kind`, denominator, requested horizon, realized number of scored queries, and whether the episode ended. Never pool these differently scaled targets without an explicit conversion.

Short windows generally omit parts of the full-policy derivative; exact transport within a window does not make the truncated training objective unbiased for `J`.

### 4.5 Audit selection and local-expectation policy gradient

Conditional on the exogenous stream, let `p_t = sigmoid(f_psi(h_t))` and sample all actions using independent uniforms. Because `h_t` is independent of actions, the action distribution factors across candidates. Shared parameters do not invalidate this factorization.

For fixed `theta`:

\[
\nabla_\psi J=-\mathbb E\left[\sum_tD_t\nabla_\psi p_t\right]
=-\mathbb E\left[\sum_tD_tp_t(1-p_t)\nabla_\psi f_\psi(h_t)\right].
\]

This is a local expectation/Rao–Blackwell construction, not a new gradient identity. [R9] [R10] It is conditional on the other sampled actions; it does not enumerate all jointly optimal action sequences.

Primary sampling: pre-generate the candidate event indices for an episode, choose `K=min(K_requested,Nw)` uniformly without replacement using an independent audit RNG, and set each inclusion probability to `pi=K/Nw`. Only the auditor sees the selected indices. Start with `K=8` as a calibration point; because traces last to episode end, active count never exceeds eight. No silently dropped trace, age-based overflow eviction, or “audit only positive/surprising writes” rule is permitted.

For a batch of `B` episodes, recompute current policy probabilities from stored features under the unchanged snapshot and differentiate

\[
\mathcal L_{gate}=-\frac1B\sum_b\sum_{t\in A_b}
\frac{\operatorname{stopgrad}(D_t)}{\pi_t}p_\psi(h_t).
\]

Take **one optimizer step per collected batch**. Repeatedly reusing those targets after changing the policy is not the same unbiased on-policy estimator. Trace-Reg may study replay separately, with snapshot age recorded.

Do not multiply by the factual action or divide by its propensity: both action outcomes have been evaluated. The inclusion correction is for selecting anchors, not actions. Do not divide by `K` again after applying `1/pi`; that changes gradient scaling.

The primary policy has no hard probability clamp. Initialize logits near zero. If a pilot shows premature saturation, test a declared exploration floor `p = epsilon/2 + (1-epsilon)sigmoid(z)` or an entropy schedule. Differentiate the actual probability, record the schedule, and evaluate both the training distribution and any floor-free deployment policy separately. An exploration alteration is shared with binary baselines where applicable.

Conditioning reduces variance of an individual score-function contribution relative to its unconditioned counterpart. It does not establish lower variance of the whole shared-parameter gradient, superiority to an optimized baseline, or better progress per second. Those require measurements.

### 4.6 Trace-Reg and derivative-credit comparator

Trace-Reg predicts a real-valued utility `g_psi(h)`. **Its output must not be constrained by a sigmoid:** beneficial and harmful targets have both signs. Fit squared error as the conditional-mean reference and Huber loss as the proposed robust alternative. Use known anchor-inclusion weights when candidate counts or sampling probabilities differ, and compute training normalization statistics with the same declared population weights. Standardize using training-only statistics, undo the transformation before thresholding, and record clipping as a changed objective rather than a harmless preprocessing step.

With no explicit cost, deterministic admission is `1[g(h)>0]`; validation may select a threshold, with the same tuning allowance for competitors. A logistic exploration policy can use `sigmoid(g/temperature)`. Regression accuracy, sign calibration, and decision regret are diagnostic; downstream query loss decides usefulness.

Matched derivative credit uses the same anchor set and horizon:

\[
\widetilde D_t=-\frac1Q\sum_{s>t}\nabla_{r_s}\ell(r_s^{f},y_s)^\top
u_t(w_{s-1}^\top q_s).
\]

Here `u_t` is the stored pre-decision correction vector. The derivative is evaluated at the factual path. Also compare the all-reject endpoint and midpoint derivative in small diagnostics, charging any additional evaluations. This component comparison isolates finite loss evaluation from transported derivative credit, not a full reproduction of LatentHarness.

Ordinary full BPTT through a continuous gate is exact differentiation of its smooth objective, not an inferior local heuristic by definition.

### 4.7 Reader optimization and storage

Primary logits are affine in the memory read:

\[
z_s=b_\theta(x_s)+C_\theta r_s.
\]

Freeze `C` within the episode and cache `c_t = C u_t` when an anchor is created. Then

\[
z_{alt}=z_f+(1-2a_t)c_t(w^\top q_s).
\]

Compute both losses with stable log-sum-exp; the result is the exact finite cross-entropy difference. A nonlinear reader instead requires evaluating its alternative read; it cannot use this shortcut. Run that case as an overhead stress test. Share cached features, batching, and affine-head optimizations with dense-branch and derivative baselines wherever they apply.

For `K` active anchors, vector-state storage is `O(K(dv+dk))`, or `O(K(dk+C))` when only an affine-head-projected content vector is retained. Transport is `O(K dk)` per event. Query scoring adds at least `O(K C)` for an affine head; generic readers may dominate. Per-anchor metadata, stored features, optimizer state, and pending feedback must also be counted.

### 4.8 Reference pseudocode

All functions below are project interfaces to implement, not an existing package API.

```python
# One frozen-parameter episode. Core generator releases labels immediately
# after the pre-write prediction. Delayed labels use saved logits, not rescoring.
M = zeros(dv, dk)
traces = []
selected, inclusion_probability = sample_candidate_indices(episode, audit_rng, K)

for s, event in enumerate(episode):
    h, k, v, q, lam, beta = memory_free_features(event, raw_history)
    if event.is_query:
        rf = M @ q
        zf = reader(event.public_query, rf)
        y = feedback.release_after_prediction(event)
        lf = cross_entropy(zf, y)
        for tr in traces:                  # vectorize this dimension in practice
            if tr.in_window(s):
                dr = tr.u * dot(tr.w, q)
                za = reader(event.public_query, rf + (1 - 2*tr.action)*dr)
                delta = (2*tr.action - 1) * (cross_entropy(za, y) - lf)
                tr.raw_loss_difference += delta
    P = lam * M
    u = beta * (v - P @ k) if event.can_write else zeros(dv)
    p = gate_probability(h) if event.can_write else 0.0
    a = int(action_uniform[s] < p) if event.can_write else 0
    M = P + a * outer(u, k)
    for tr in traces:
        tr.w = lam * (tr.w - a * beta * k * dot(k, tr.w))
    if s in selected:
        traces.append(Trace(u=u.copy(), w=k.copy(), action=a,
                            features=h.copy(), pi=inclusion_probability,
                            anchor=s, raw_loss_difference=0.0,
                            snapshot_id=snapshot_id))
    raw_history.observe(event)             # all raw observations, not just writes
    retire_expired_traces_without_crossing_snapshot(traces, s)

# Divide completed sums by their declared Q or H; detach all audit targets.
# Retain unfinished fixed-H traces at episode end as right-censored finite-episode
# windows with the original denominator. Never continue them across a reset.
```

Use masks for padded batch events. A padded event must neither decay memory nor advance a trace's real-event horizon. The same applies to dummy query slots added for tensor shape convenience.

### 4.9 Stability, finite versus differential effects, and horizon diagnostics

With unit keys, `0 <= beta <= 1`, and `0 <= lambda <= 1`, each shared-control transport is nonexpansive. Hence `||w_s|| <= product(lambda) ||k_t||`. This does not bound loss differences without additional bounds on the reader and loss gradients.

At `lambda=.995`, the upper envelope after 64 events is approximately `0.72557`; after 512 events it is `0.07681`; the decay half-life is about `138.28` events. Thus a 64-event horizon cannot be justified by assuming the trace has necessarily vanished. Key collisions may attenuate it faster; measure the actual tail contribution by age.

A scalar example isolates the finite/tangent distinction: loss `0.5(r-.75)^2`, reject read `0`, accept read `2`. Exact beneficial-write utility is `-.5`, but a tangent at reject predicts `+1.5`. This demonstrates disagreement, not learning superiority. A continuous write of strength `.375` has zero loss here, illustrating why the continuous baseline may be better than either binary action.

A fixed matrix has the same allocated storage regardless of how many writes are accepted. Do not invent a per-write storage charge. Measure real compute savings or impose an explicit, separately motivated write/energy constraint. Interference is an actual consequence, already included in future query loss.

### 4.10 A concrete failure of fixed-control causal interpretation

Let scalar memory initially be zero. The first candidate writes value `1` with `beta=1`. The second observation has value `-2`, `beta=.5`, and its gate writes iff current memory exceeds `.5`. Decay is one. Score final memory with `0.5(M-1)^2`.

Accepting the first write yields final `M=-.5`. Rejecting it while **holding the later action at write** yields `M=-1`; fixed-control utility is `+.875`. Recomputing the later gate after rejecting instead produces no second write and final `M=0`; adaptive utility is `-.625`.

The sign reverses. Common random numbers alone do not repair this issue: controls must actually remain the same, or the alternative branch must recompute them. This example was explicitly simulated in the supplemental checks.

### 4.11 A stronger limitation: expected memory and the fractional-gate control

**Deduction from this plan's assumptions, not a claim of a novel theorem.** Condition on the complete exogenous episode and fixed model parameters. Since `a_t` is independent of earlier actions and hence of `M_{t-1}` under that conditioning,

\[
\mathbb E[M_t]=\lambda_t\mathbb E[M_{t-1}](I-p_t\beta_tk_tk_t^\top)
+p_t\beta_tv_tk_t^\top.
\]

This is exactly the deterministic memory recurrence obtained by setting `a_t=p_t`. Induction from the same initial state establishes `M_t^fractional = E[M_t^binary]`. With an affine-in-memory read head, the fractional logits are also the expected binary logits. Cross-entropy is convex in logits, so Jensen's inequality gives, query by query,

\[
\ell(z_s^{fractional},y_s)\leq
\mathbb E_a[\ell(z_s^{binary},y_s)].
\]

The inequality also holds after summing queries and averaging episodes. It concerns a fixed learned gate and reader, not which optimizer finds better parameters within a finite budget. It does not compare the fractional policy with a particular deterministic threshold policy, nor does it hold for arbitrary nonlinear memory-conditioned readers or feedback-dependent gates.

**Mandatory diagnostic:** after training every applicable binary gate, evaluate the same checkpoint with `a=p`, without retraining. In particular record `fractionalized_trace`, which still uses a counterfactually trained gate and is therefore not an independent non-CF learning baseline. Its expected loss should be no greater than the stochastic checkpoint up to evaluation/numerical uncertainty; a violation flags insufficient policy sampling or a violated architectural assumption. This conversion incurs inference cost but no extra model fitting.

This sharply limits the strongest case for stochastic admission in the primary affine setup. Its possible value is a better finite-budget learning procedure, real savings from skipping expensive writes, or near-deterministic decisions with little Jensen gap—not a representational advantage. Dense masked tensor arithmetic may not skip any work when `a=0`; measure actual execution before claiming a write-cost benefit. A fixed matrix also saves no storage bytes by rejecting writes.

Do not remove the affine case or weaken the continuous baseline to evade this result. The already planned nonlinear-head test is a separate boundary condition, not a guaranteed favorable replacement. No binary-gate method is expected to defeat this fixed-checkpoint bound. A counterfactually trained gate deployed fractionally could still outperform an ordinarily trained continuous gate at equal training cost; that would be a learning-signal result with explicitly changed deployment semantics.

The check in Appendix B3 enumerated 64 action sequences in each of 20 small problems. Maximum expected-versus-fractional memory discrepancy was `6.661e-16`; expected binary minus fractional cross-entropy ranged from `0.00753` to `0.09909` nats/query. These are random unit-test cases, not learned-policy benchmark gains. A deliberately nonlinear scalar reader also verifies why the affine qualification matters.

## 5. Workloads, representations, and information controls

### 5.1 Mandatory synthetic generator

Implement `AssociativeStreamV1` from the specification below. Generated data are project-created synthetic evidence, not a model of all real information streams. License newly written generator code explicitly; do not copy an external dataset's license onto generated data.

**Anchor dimensions:** `T=512`, `dk=dv=64`, `C=32`, `Nkeys=256`, active key subset size `A=128`. These are starting calibration points, not optimized settings or mandatory upper limits. The first 32 events are observations. Each following block of four events contains three observations and one query, with the query slot uniformly randomized within the block. This yields exactly `Q=120` queries and `Nw=392` candidates. All scheduling is independent of learner actions.

**Episode construction:**

1. Draw a new active subset and a new independent ground-truth class `truth[i]` uniformly from 32 classes for every active key. Re-randomize these associations every episode, including validation and test. A slow network must not solve the task by memorizing key-to-label mappings across episodes.
2. Build 16 cluster centers and key vectors from independent standard normals. For key `i` in cluster `g`, set `raw_key[i] = normalize(sqrt(rho)*center[g] + sqrt(1-rho)*noise[i])`, with centers and noise separately normalized. Main `rho=.35`; cluster membership is not a hidden semantic answer. Reuse the same generated vectors across methods in a paired replicate.
3. Draw an unobserved `hot[i] ~ Bernoulli(.25)`. Draw independent observation weights `exp(Normal(0,1))` for active keys and normalize them. These weights determine observation frequency; they do not directly equal query propensity.
4. At each observation, sample a key from those weights. With probability `.02`, replace its truth with a different uniformly sampled class before generating the report. Draw an unobserved source-quality state: high with probability `.7`, low otherwise. The report equals current truth with probability `.98` for high quality and `.55` for low quality; otherwise report a uniformly chosen different class.
5. Expose a noisy source cue equal to the quality-state bit with probability `.75`; otherwise flip it. Independently expose a noisy relevance cue equal to `hot[i]` with probability `.75`; otherwise flip it. Neither source state, true-change indicator, corruption indicator, nor actual future schedule is exposed. Cues are deliberately imperfect and available to every relevant baseline.
6. After this observation, create a private query ticket for that key with probability `.6` if hot and `.05` otherwise. Draw its delay: with probability `.7`, draw an integer log-uniformly from `[1,64]`; otherwise uniformly from `[65,256]`. Define integer log-uniform as `floor(exp(U(log(1), log(65))))`, clipped to `[1,64]`. Tickets are scheduling data, not learner features.
7. At a query slot, pop the earliest-due ticket with due time no later than the current event; break ties by an independent random tie-breaker. When there is no due ticket, sample among previously observed keys with weight `4` for hot and `1` otherwise. Query the **current truth**, not necessarily the class reported at the ticket's creation. Remove only the popped ticket; multiple tickets for one key are allowed. Tickets beyond episode end are not scored.
8. Output a query's key before prediction and its target only after prediction. A query does not write its answer into memory in the core task. Its feedback may train parameters after the episode, but is not an extra observation. Candidate updates occur only on observation events.

Every random decision belongs to a named independent stream: codebook, truth/change, observation schedule, corruption/cues, tickets/queries, model initialization, action uniforms, audit selection, and evaluation policy draws. Derive their seeds from a recorded master seed and stable namespace hash, not Python's process-randomized built-in `hash`.

Retain private event annotations for evaluation slices, never in the learner batch. Realized “never queried” observations need not coincide with a predefined hidden type. Measure their frequency instead of claiming the generator guarantees a particular mix.

### 5.2 Development, protected tests, and shifts

Use three disjoint seed namespaces: exploratory training/development, validation/tuning, and protected confirmation. Codebook families used for confirmation must also be independently generated. Save the generator revision and parameter hash; changing either after viewing protected outcomes is a methodological amendment requiring fresh confirmation data.

| Condition | Role | Parameters changed from anchor |
|---|---|---|
| Orthogonal, clean, short recall | Correctness positive control | At most 16 distinct orthogonal keys; no corruption/change; `lambda=beta=1`; queries follow an observed fact |
| Reliability-dominated | Strong-simple-method control | Low overlap; quality cue remains informative; most delays short. A tuned cue-aware heuristic is expected to be competitive |
| Delayed interference, anchor | Main IID workload | The complete generator above |
| Long-delay shift | Predeclared main shift | Long-delay mixture weight `.7`, long interval `[129,384]`; other parameters unchanged |
| Interference shift | Robustness | `rho=.65`, with the same active-key count |
| Load shift | Robustness | Active set `A=256`, with the same memory size |
| Cue degradation and reversal | Robustness / shortcut diagnosis | Cue accuracy `.5`, then `.25`; report both, without selecting the favorable one |
| No predictable benefit | Negative-information control | Remove relevance/quality cues and make future target relevance independent of all gate features; use an exchangeable one-key/no-history construction for the strict impossibility test |

For the strict negative-information test, use independent one-shot trials with a single decision, identical public features, and a future target chosen independently so that the two actions have equal expected loss. Concretely, use one scalar write followed by one query, `r=a` for `a` in `{0,1}`, logits `[2r-1, 1-2r]`, and an independent uniform binary target. Both binary actions have equal expected cross-entropy; continuous `a=.5` is a different action space and can do better, so that does not indicate leakage. A constant optimal policy may still exist in other cue-erased workloads; cue erasure alone is not a proof of impossibility.

Use separate, small parameter sweeps over overlap, delay, change rate, and cue accuracy during development. Freeze the actual confirmatory population before unsealing tests. Negative or favorable shifts are not removed retroactively. A “shift-robust” claim requires the declared shift result, not any one selected slice.

### 5.3 Representation stages

**Stage A—controlled credit experiment.** Use `k=q=raw_key` and class values as the first 32 standard basis vectors in 64 dimensions. Use affine logits `C r`, initially `C = temperature × [I_32, 0]`. Calibrate the temperature on development/validation data, then freeze it for the matched-estimator comparison. No trainable key-ID lookup is necessary. Gate hidden width is 32, one hidden layer with tanh or SiLU selected on development and fixed across learned-gate methods.

This deliberately simple stage preserves memory interference, nonlinear prediction loss, delayed credit, and binary decisions. It does not establish the value of jointly learning representations. A scalar-temperature or gate effect cannot be advertised as a new representation-learning result.

**Stage B—slow-model learning.** Add learnable linear key and query maps initialized near identity, a class-value embedding initialized from Stage A, and an affine read head. Keep raw-observation gate features independent of memory and of these learned representations in the primary implementation. Train slow parameters by ordinary task-loss BPTT through the factual action sequence, with the sampled actions detached.

For each batch: collect under snapshot `(theta_0, psi_0)`; compute the gate gradient; replay raw episodes with the recorded actions to compute the slow-model gradient at `theta_0`; then apply both optimizer updates. Do not apply either update until both gradients based on the snapshot have been computed. All traces are retired before the next snapshot. Replay cost is charged. Competitors that can retain their graph and avoid replay are allowed to do so.

Use an all-write warm start only as a shared development choice. The suggested 100 warmup episodes and 2,000 subsequent episodes are pilot anchors, not evidence-based convergence budgets. Fit learning curves and set budgets as described in Section 7. Methods may use different validated warm-start schedules when competing on the end-to-end frontier; account for all their costs.

**Nonlinear-head stress.** Replace the affine read contribution with `W2 tanh(W1 r)` using a small hidden width, then rerun the exactness tests and profile alternative scoring. This tests whether an engineering win depends on the affine head. It is not necessary to involve a transformer to test this limitation.

### 5.4 Leakage and auxiliary-memory tests

The policy-facing `PublicEvent` must not contain target, truth, hidden event type, future tickets, or future query count by key. Store those in `PrivateAnnotations` passed only to evaluators. Perturb future targets and private annotations while holding public inputs fixed: actions before their release must be unchanged. Perturb an early action and replay with shared uniforms: all later restricted gate features and action probabilities must match exactly.

Count raw-history arrays, recency metadata, codebook storage, neural weights, lookup tables, and caches. Report shared static model assets separately from per-stream mutable memory, but hide neither. A dictionary baseline may store labels because that is its memory; the proposed gate may not get an uncharged per-key label cache disguised as a feature preprocessor.

## 6. Baselines and discriminating controls

### 6.1 Mandatory local comparison set

Use the same generated episodes, observation cues, query timing, and loss. Compare estimators within the same architecture first; compare entire systems separately. An information-richer or continuous-action system can legitimately be stronger, but the table must disclose that difference.

| ID | Implementation / tuning | Scientific role |
|---|---|---|
| `all_write` | Always admit; jointly tune scalar decay, correction, and permitted read calibration | Establish whether admission is necessary |
| `never_write`, `random_write` | No writes; independent Bernoulli writes, acceptance rate matched on validation | Harness and sparsity controls, not competitiveness claims |
| `heuristic` | Thresholds or a small grid-rule using raw count, age, source/relevance cues; frequency-only and recency-only ablations | Strong simple explanation |
| `change_surprise` | Bounded latest-report cache or memory residual; tune change persistence and source-cue thresholds; charge its state | Richer-history alternative; label its information access |
| `continuous_bptt` | `a=p` in the recurrence; full-episode BPTT; native continuous evaluation | Strong ordinary learned gate, with a larger action space |
| `binary_st` | Forward binary action, surrogate `a_hard + p - stopgrad(p)`; full BPTT | Common biased binary estimator; do not call it unbiased |
| `reinforce` | Reward-to-go, trained action-independent baseline, entropy/exploration tuning; also two-sample leave-one-out baseline | Strong score-function comparator rather than naive REINFORCE |
| `arm`, `disarm` | Two coupled full action trajectories per episode; equations below | Essential competing unbiased binary estimators |
| `derivative_trace` | Same anchors, controls, horizon, and gradient-learning wrapper as Trace-PG; derivative signal in Section 4.6 | Isolate finite versus differential credit |
| `dense_cf` | Explicit independent alternative memory per selected anchor, same action tape and exact losses | Correctness oracle and auditor cost comparison; labels should agree |
| `trace_pg`, `trace_reg` | Exact trace; direct gradient versus utility regression | Selected method and faithful original variant |
| `fractionalized_trace` | Same trained Trace-PG checkpoint, replace sampled `a` by `p`; no refitting | Mandatory Jensen/mean-memory diagnostic; still uses CF training, not an independent non-CF baseline |
| `kda_block`, `gdn2_block` | Small recurrence adaptations defined below, trained by ordinary BPTT | Stronger forgetting/erasure mechanisms |
| `rls` | Online regularized least-squares memory, optional decay and cue weighting | Interference handled by a simpler alternative geometry |
| `dictionary` | Byte-budgeted exact-key map, FIFO/LRU/LFU eviction and tuned cue-aware admission; smoothed categorical predictions | Strong system alternative on identifiable symbolic keys |
| `larger_memory` | Best validated ordinary method at larger dimensions under time/memory envelopes | Opportunity cost of spending resources on auditing |

Do not complete every expensive configuration before the cheap headroom gate. However, a system-level positive claim cannot be finalized without the strong applicable alternatives. An omitted competitor narrows the claim explicitly.

A nearest-neighbor buffer is a required additional system alternative when observations use continuous aliases without stable key IDs. Its index, stored keys/values, search, and rebuild costs count. The core exact-ID generator intentionally permits the exact dictionary to win; hiding that comparator would make the task artificially favorable to a neural matrix.

### 6.2 Implementable discrete-estimator formulas

All loss values in these formulas are detached; gate networks receive gradients through probabilities or logits only. Use the actual loss sign, not a reward sign copied from another implementation.

For independent uniforms `u`, let `p=sigmoid(z)`, `a_plus = 1[u > 1-p]`, `a_minus = 1[u < p]`, and `DeltaL = L(a_plus)-L(a_minus)`. Both complete trajectories use identical exogenous events. ARM's logit-gradient sample is

\[
g_i^{ARM}=\Delta L\,(u_i-1/2).
\]

DisARM's sample is

\[
g_i^{DisARM}=\tfrac12\Delta L\,(-1)^{a_i^-}
\mathbf1[a_i^+\ne a_i^-]\sigma(|z_i|).
\]

Differentiate the surrogate `sum_i stopgrad(g_i)*z_i`, averaging episodes, not action coordinates. For a transformed exploration probability use its actual logit and the correct chain rule. These are established estimators, not project inventions. [R11] [R12]

A REINFORCE term is `(future_loss_t - baseline_t) * log_prob(a_t)`. Future loss includes all later queries divided by episode `Q`. Train the baseline with a separate squared-error objective and detach it in the policy loss. It may depend on state before `a_t`, not on `a_t` or its consequences. Also evaluate two independent action trajectories with a leave-one-out loss baseline. Tune baseline learning and loss scaling; poor baseline fitting is not evidence for Trace-CF.

Validate all estimator means by enumerating at most six to ten decisions in tiny fixed problems. Test nontrivial shared-parameter gates as well as one free logit per action. Record bias, covariance, and gradient error per elapsed second. Do not compare only unbaselined score-function variance.

### 6.3 Modern recurrence adapters

These equations define **small component adaptations**, not complete reproductions of the published networks.

For KDA in our orientation, with diagonal `D=diag(alpha)` and channel decay `alpha` in `[0,1]^dk`:

\[
P=M D,\qquad M'=P+\beta(v-Pk)k^\top.
\]

Generate `alpha` and `beta` from the same permitted current features using a small controller, then train through the recurrence. Match parameter budgets in the architectural comparison, while also reporting native parameter counts. [R6]

For the GDN2 adapter, use `e=b⊙k`, `z=w⊙v`, `b∈[0,1]^dk`, `w∈[0,1]^dv`, and

\[
P=M D,\qquad M'=P+(z-Pe)k^\top.
\]

Verify this orientation against the source equations and an independent scalar-loop implementation. Tie channels to recover simpler recurrences as a unit test. Read before the update in this project. [R7]

For RLS, initialize inverse covariance `P_inv = I/ridge`, model memory `M=0`, forgetting factor `0 < rho_rls <= 1`, and apply

\[
g=\frac{P_{inv}k}{\rho_{rls}+k^\top P_{inv}k},\quad
M'=M+(v-Mk)g^\top,
\]
\[
P_{inv}'=\frac{P_{inv}-(P_{inv}k)(k^\top P_{inv})/(\rho_{rls}+k^\top P_{inv}k)}{\rho_{rls}}.
\]

Use a positive ridge and check symmetry/positive definiteness against a float64 batch least-squares reference on small cases. Cue weighting can be implemented by scaling both `k` and `v` by `sqrt(weight)`; a positive weight is learned/tuned only from permitted data. Count the extra `dk × dk` covariance state. Symmetrization, stabilization, and CPU float64 fallback are measured implementation choices, not free operations. This is an online-ridge alternative, **not** a claimed reproduction of Preconditioned DeltaNet.

### 6.4 Dictionary and rule specifications

Implement two bounded dictionary variants, not just a straw-man last-value cache. The first stores `(key_id, last_accepted_class, age, access_count)` and admits reports according to a validation-tuned source-cue threshold. Its predicted class receives probability `1-(C-1)*epsilon`, others `epsilon`, with `0 < epsilon < 1/C` calibrated on validation. A missing key predicts the training class prior, uniform in the balanced synthetic task. The second stores a float32 categorical count vector per key, updates it as `counts = gamma*counts + weight*one_hot(report)`, and predicts `(counts+alpha)/(sum(counts)+C*alpha)`. Use the observed source cue for a tuned positive weight, never hidden correctness. Store all timestamps, IDs, counts, and hash/index capacity in the byte budget. Tune smoothing and decay jointly with admission on development data; expand boundary optima.

FIFO evicts the earliest inserted entry; LRU the least recently observed/queried stored key; LFU the smallest stored access count. Break ties by age then numeric key ID. Measure actual capacity under the budget, not a nominal number of entries that ignores index overhead. The categorical-count dictionary may retain fewer keys than the last-class dictionary; this is part of the tradeoff.

The change-detection rule stores a current accepted report, a candidate replacement report, its consecutive-observation count, and bounded metadata. A matching report refreshes the current value; a different report becomes the candidate; a repeated matching candidate replaces the current report after a tuned persistence count. Compare immediate acceptance for a strong source cue versus persistence filtering for a weak cue. A residual-surprise baseline ranks `norm(v-lambda*M@k)` and tunes an acceptance interval or threshold. Both use only current/past observable reports; both are explicitly richer-state or memory-conditioned alternatives, not valid features for the restricted primary gate.

### 6.5 Tuning and reproduction checks

Use log-spaced learning rates initially around `1e-4, 3e-4, 1e-3, 3e-3`; these are search seeds, not prescribed optima. Scalar delta baselines start with `beta` in `{.1,.5,1}` and decay half-lives `{32,128,512,infinite}` converted to per-event `lambda`. Expand a boundary optimum before confirmation. Tune gate width, thresholds, temperature, entropy, truncation, and regression loss where applicable. Full BPTT is mandatory when it fits; TBPTT at `32,64,128` is a separate speed–bias frontier, not a replacement for full BPTT.

Give every method a documented search envelope and comparable validation access. Equal trial counts alone are not fair when trial costs differ; report both total search time and model-fitting time. Share generic optimizations and allow architecture-appropriate implementations. Validate that baselines learn the clean recall task, improve over random initialization, and recover known limiting cases before comparing difficult tasks.

For public reference code, resolve and save an immutable commit before use. The inspected FLA `main` URL is a discovery reference, not a version pin. Save the MIT notice for reused code. No upstream repository was executed for this report. [R18]

## 7. Evaluation, resource matching, and statistical protocol

### 7.1 Primary metrics and evaluation population

Primary metric: episode-mean negative log likelihood, **nats/query**, averaged equally across independently generated episodes within the declared workload. Report accuracy as percent correct, not as a replacement for a loss regression. Also report deterministic-policy loss, accuracy by observation age, load/interference slices, acceptance rate, and loss after genuine changes versus corrupt reports. Define observation age as event distance since the latest earlier report for that key matching the query's current true class, with a separate never-observed bin. Also report time since the latest accepted write for the key. These evaluator-only quantities must not enter the restricted gate.

For a stochastic policy, estimate each episode's expected loss using common evaluation uniforms across compatible methods. Increase evaluation draws during the pilot until policy Monte Carlo error is small relative to between-training-replicate uncertainty; freeze the draw count for confirmation. Draws and individual queries are not additional training replicates. If stochastic evaluation is costly, more independent episodes can be used instead, with the analysis retaining the correct nesting.

The primary learning comparison is Trace-PG against the **validation-selected best equal-cost non-finite-CF local comparator**, chosen before protected test access. Eligible methods are the ordinary, heuristic, derivative-credit, discrete-estimator, recurrence, dictionary, and larger-memory alternatives in Section 6, including the null controls when competitive. Trace-PG/Trace-Reg, dense finite-CF supervision, and fractionalized Trace-CF are not eligible to serve as their own independent learning comparator; they have separate family/efficiency contrasts. Freeze the selected comparator. Claiming to beat every baseline additionally requires simultaneous comparisons against all of them, not merely a favorable comparison with the selected one.

Secondary metrics: total project-specific training and tuning seconds; deployment seconds per stream/event and query; peak process memory; MPS allocated/driver bytes; mutable online state bytes; checkpoint bytes; extra reader evaluations; query latency; gradient bias/variance diagnostics; and utility calibration. Define “false admission” by the sign of independently audited realized utility, not by the generator's corruption flag. A true change can be harmful to store in a particular window, and a corrupted report can occasionally have positive realized marginal effect.

### 7.2 Three comparisons that must not be conflated

**Mechanism-matched:** same dimensions, representations, gate features, episodes, action space, and labels; compare learning signal or auditor only. The dense and trace auditors should produce matching gradients with the same anchors. This establishes attribution but may have unequal runtime.

**Training-cost matched:** use the same hardware and measured end-to-end budget, including collecting audits, forward/replay passes, fitting the gate and slow model, validation, and method-specific preprocessing. Plot loss versus total fitting time and separately versus total tuning-plus-fitting time. Report data consumed at each point. More episodes within an equal-time budget are allowed.

**System frontier:** permit different memory sizes, action spaces, state layouts, and training schedules. Report nondominated combinations of query loss, training time, deployment time, and online bytes. A lower-loss method that costs substantially more is a tradeoff, not an overall win.

Use pilot convergence curves to choose budget landmarks, for example fractions and multiples of the strongest baseline's measured time to its stable validation region. Landmarks are reporting points, not a hard termination rule. If no stable region is reached, state that fact, compare at the measured budgets, and avoid claims about asymptotic quality. A baseline still improving at the boundary needs an expanded development budget or a bounded-budget claim.

### 7.3 Independent units and paired replication

An outer replicate consists of independent model initialization/training randomness, an independent codebook family, and an independent held-out episode bank. Pair all methods inside that replicate on the same exogenous training and evaluation streams where the comparison allows it. Different training methods will produce different action histories; forcing their factual actions to match during the learning comparison would change the problem.

Compute one mean loss per method, condition, and outer replicate. Let `d_j = loss_baseline,j - loss_trace,j`. Report mean `d`, relative reduction using the paired baseline denominator, individual replicate differences, and uncertainty across outer replicates. Never calculate the primary confidence interval as if all queries were independent.

Primary interval: a paired mean interval across outer replicates using a predeclared Student-t procedure, accompanied by a paired bootstrap sensitivity analysis that resamples whole replicates. Inspect pilot skew/outliers before choosing this procedure; when normal approximation is implausible, preregister a robust or sufficiently replicated bootstrap analysis instead. A tiny number of seeds is descriptive, not a reliable license to report narrow intervals. Separately report within-replicate episode uncertainty and codebook sensitivity.

### 7.4 Pilot-to-confirmation sample-size rule

Do not adopt the proposal's five seeds as an immutable scientific sample size. Run independent exploratory pilot replicates and estimate the standard deviation of paired differences, validation learning-curve variability, and per-replicate cost. An initial batch of six is a planning default for obtaining these estimates, not a confirmation test. Additional pilots are justified by unstable variance estimates or unresolved baseline learning, not by waiting for a positive mean.

For a candidate design effect `delta_design`, estimate

\[
n\approx\left(\frac{(z_{1-\alpha/2}+z_{.8})s_{upper}}{\delta_{design}}\right)^2,
\]

then refine using the actual t-based procedure and the multiplicity plan. Here `s_upper` is a conservative upper uncertainty bound for the paired standard deviation. For the default positive-pilot path, use half the observed positive pilot gap as the planning effect, documenting that this is a conservative planning assumption, **not** a minimum acceptable improvement. Evaluate sensitivity to a smaller effect. This selection uses development data only.

When the pilot gap is zero or negative, do not invent a positive target to justify confirmation. First resolve headroom, signal predictability, and baseline convergence. If an application-relevant effect remains plausible but the pilot is uninformative, report the effect sizes detectable at affordable local sample sizes and choose a fixed precision target from the measured headroom/cost analysis before proceeding. With no application specified, there is no universal monetary or percentage definition of practical importance.

Write `power_plan.json` with pilot statistics, variance assumptions, proposed `n`, detectable effects, measured run cost, and feasible alternatives. If the justified sample size exceeds available local resources, label the result underpowered/inconclusive or narrow the claim. Do not substitute many correlated queries for training replicates. After selecting `n`, freeze it; no continuation until significance. An interruption may resume the same predeclared sample, not change its stopping rule.

### 7.5 Confirmatory hypotheses and multiplicity

Before protected evaluation, freeze a manifest containing hypotheses, generator hashes, methods, chosen comparator, dimensions, exact thresholds/policy mode, training/tuning budgets, replicate seeds, evaluation counts, primary metric, exclusions, and analysis code revision.

The default confirmatory family tests: (i) IID loss improvement over the selected comparator; (ii) improvement on the long-delay shift; and (iii) degradation when genuine targets are replaced by the predeclared shuffled-target control. Use Holm correction across these tests. A finite-versus-derivative superiority claim adds its direct contrast to the family before testing. The generic “useful exact auditor” claim does not require winning every learning comparison; its narrower evidence is numerical parity plus the measured auditor frontier.

Shuffling is a falsification aid, not a logical requirement that every true target-dependent method must collapse completely. If only a constant mean effect is useful, shuffling may preserve a gain. In that case the proposed claim about individualized delayed credit is not established; compare a constant-utility policy explicitly.

Protected data remain sealed until the configuration and analysis manifest are hashed. A test result that inspires a new model is exploratory for that result; confirmation needs fresh independent data and a recorded amendment. Retain failed runs and all declared workload results.

### 7.6 Timing and system variability

Use synchronized wall-clock boundaries. On MPS, synchronize immediately before and after timed sections; on CPU, fix and record thread count. Benchmark both CPU and MPS for the actual small sequential workload rather than presuming the GPU is faster. Distinguish cold startup/compilation, warmed steady-state execution, end-to-end fitting, and inference with audits disabled. Include data generation/transfers when material; show a compute-only breakdown as a secondary diagnostic.

Warm up until repeated block timings have stabilized according to a predeclared coefficient-of-variation criterion chosen in calibration. Then use randomized blocked method order and repeated blocks sufficient to estimate timing uncertainty. Report median and spread, battery/power state, concurrent heavy processes, and any observed throttling. Timing blocks are observations about runtime, not independent evidence of prediction improvement. Do not flush caches for one method but not another, or compare an optimized trace implementation with avoidably slow Python dense branches.

Energy is optional: report it only when a calibrated supported measurement is available. Do not infer energy from wall time alone or require privileged system instrumentation for the core PoC.

## 8. M3 Max feasibility and hardware-adaptive execution

### 8.1 Discovery before installation or scale decisions

Run these existing system commands and save their outputs; do not infer memory capacity from the chip name:

```bash
uname -m
sw_vers
sysctl -n hw.memsize
system_profiler SPHardwareDataType SPDisplaysDataType
xcode-select -p
df -h .
python3 --version
```

Record power state, free disk, available memory, and whether the Python interpreter is native arm64 rather than translated. Inspect the selected PyTorch wheel's requirements before installing. Do not automatically upgrade macOS, install privileged tooling, or replace the user's system Python.

Use a project virtual environment, stable compatible PyTorch, NumPy, pytest, PyYAML, and SciPy for analysis. Use standard-library JSON/CSV and compressed NumPy arrays for artifacts; no database service is needed. Resolve the newest **released stable compatible** wheel from official package/release metadata, test it, then freeze exact dependencies. This is a selection rule, not a claim that an untested version is compatible. During research, Apple's setup page named 2.11.0 while PyTorch documentation resolved to 2.14; do not infer the installed or compatible version from either label alone. [R19] [R20]

Do not require FLA's specialized kernels for local execution. Its advertised accelerator support does not establish that a particular kernel works on MPS. The pure tensor recurrence is the reference and fallback. [R18]

### 8.2 Operation-level gate

The first project command must test CPU float64, CPU float32, and MPS float32 when available. Test batched matrix multiplication, outer products/einsum, normalization, indexing/masks, linear layers, tanh/SiLU, log-sum-exp/cross-entropy, autograd through the full recurrence, optimizer updates, checkpoint round-trips, and seeded random inputs.

Generate fixed action tapes on CPU for parity tests so numerical differences in probabilities do not accidentally change the experiment being compared. Compare full reads, scalar losses, and gradient norms/directions, including late-horizon cases. Use tolerances relative to a scale such as `max(1, norm(reference))`, not elementwise relative errors around zero. Start with float64 trace tolerances around `1e-10` and float32 normalized tolerances around `1e-4`; tighten or diagnose against observed error and loss scale. These are engineering test bounds, not prediction-improvement thresholds.

MPS support requires an actual available backend and successful critical operations. Disable implicit CPU fallback during correctness and performance probes; inspect warnings and device placement. If an operation fails or disagrees materially, use an entirely CPU implementation for that comparison, or implement a faithful alternative and test every affected method. Never let hidden device transfers make a timing comparison uninterpretable. The documented MPS interfaces include synchronization, allocator/driver memory inspection, recommended working-set size, and RNG-state save/restore. [R20] [R22]

Use float32 for the core learning experiment and CPU float64 for references. Do not introduce quantization, half precision, custom Metal code, or a different memory update merely to obtain speed before the mechanism has been validated. A fast custom path must pass forward, gradient, and trajectory parity and be shared wherever applicable.

### 8.3 Memory model and estimates

These are arithmetic estimates, **not measured peak usage**. Let `b=4` bytes for float32, batch size `B`, active traces `K`, gate feature width `F`, parameter count `P`, and stored query count `Q`.

| Contributor | Approximate bytes / qualification |
|---|---|
| Factual memory | `b B dv dk` |
| Trace vectors | `b B K (dv+dk)`; projected affine-head form can replace `dv` by `C` |
| Trace features | `b B K F`, plus timestamps/actions/masks/targets |
| Dense alternative memories | `b B K dv dk`, excluding their duplicated features/readers |
| Slow-model parameters and Adam state | Roughly `16 P` bytes for float32 parameters, gradients, and two moment arrays; implementation/workspaces can add more |
| Uncheckpointed recurrence-state history | At least `b B T dv dk` for a naive stored-state BPTT path; this is a component estimate, not its entire graph |
| Input/features and read activations | Depends on `B T`, feature width, reader, and what is recomputed; measure explicitly |
| Raw feature statistics | Count/timestamp table per stream plus any baseline-specific metadata |
| Delayed feedback | Saved factual/alternative logits or loss-relevant sufficient data until labels release; optional variant only |
| Allocator, framework, Python, OS | Not represented by tensor formulas; measured separately |

At `dv=dk=64`, one memory is **16 KiB**. Eight rank-one traces require **4 KiB** of vector storage, versus **128 KiB** for eight dense shadows: a **32× saving in that component only**. At `B=16`, factual matrices use 256 KiB and trace vectors 64 KiB. Naively storing `B=16, T=512` full memory states contributes 128 MiB before other graph data. The `F=101`, width-32 two-layer gate has approximately 3,297 parameters. These estimates make the small mechanism test plausible locally; they do not establish its throughput or an end-to-end 32× saving.

For the optional small transformer, four width-128 layers with standard attention and a width-512 feed-forward sublayer have roughly 0.8–1 million parameters, depending on embeddings and tying. Parameter/optimizer state is modest; sequence activations, repeated windows, and training duration may dominate. Measure rather than extrapolate from parameter count alone.

Process resident memory and MPS driver allocations may overlap in unified memory. Report both and explain the accounting; do not blindly add them as disjoint physical usage. Sample allocator state during work and capture the process high-water mark. Device APIs that report current rather than peak bytes require repeated sampling or profiling.

### 8.4 Throughput calibration and configuration rules

Profile representative batches for `B` in a geometric sequence starting at one, `T` at 128 and 512, dimensions 32 and 64, and `K` at 1 and 8. Expand only where there is a scientific reason, such as a changed interference regime or an unresolved cost crossover. Measure collection, trace transport, alternative scoring, policy backward, slow-model replay/backward, optimizer steps, evaluation, and checkpointing separately.

For each method estimate

\[
\widehat{time}=N_{batches}\,t_{batch}+N_{eval}\,t_{eval}+t_{tuning}+t_{setup}+t_{logging},
\]

using measured distributions, not a single best timing. Save predictions and later compare them with actual elapsed time. There is no honest fixed run-hour estimate before calibration on the user's machine.

Prefer the faster validated CPU or MPS path for each method, then also show a common-backend comparison when backend differences materially affect interpretation. Select a microbatch that leaves ample OS headroom and avoids sustained memory pressure or swap. A provisional 60% of physical RAM working-set guard is an operational safety starting point, not a scientific limit; refine it from observed pressure and the device's recommended working set.

When memory is tight, reduce microbatch size and accumulate gradients to preserve the intended effective batch and episode set. Use activation checkpointing or recomputation for all applicable BPTT methods, charging time. Reducing memory dimensions, horizon, episode length, or the strongest baseline is a methodological change, not a transparent hardware fallback. Record it and rerun matched comparisons at the changed scale.

Estimate disk requirements from measured bytes per episode, checkpoint size, and the planned experiment count. Reserve room for atomic temporary writes and multiple recovery checkpoints. Keep episode seeds, generator versions, aggregate and per-episode metrics, audit targets, configurations, failures, and selected diagnostic trajectories. Large redundant tensors can be regenerated when their provenance is sufficient; do not delete unfavorable outcomes to save space.

### 8.5 Checkpointing and recovery

Checkpoint at completed batch boundaries using write-to-temporary plus atomic rename. Save model/gate/optimizer/scheduler states, snapshot ID, next episode and batch IDs, Python/NumPy/PyTorch RNG states, named generator states, MPS RNG state when used, data/code/config hashes, accumulated budget, and progress records. Write metrics transactionally with a last-committed record index.

For interruption mid-batch, roll back to the last completed batch and replay from its saved random states. Do not resume a half-updated optimizer or combine old traces with new parameters. Save partial traces only in a separately tested exact mid-episode recovery implementation; the default is rollback/replay. Perform a forced-interruption test and compare the resumed CPU run with an uninterrupted reference. MPS may not be bitwise reproducible; require documented numerical/decision agreement within the accepted backend contract.

After resource exhaustion, save the failure and profiling data, select a smaller microbatch with unchanged effective batch, and rerun the interrupted batch. A scientifically material change requires an amendment and new affected comparisons, not an unrecorded retry.

## 9. Repository, interfaces, and work packages

### 9.1 Minimal repository layout

```text
counterfactual-memory/
  README.md
  research_poc_plan.md
  pyproject.toml
  requirements.in
  requirements.lock.txt
  configs/
    anchor.yaml
    methods/*.yaml
    workloads/*.yaml
    confirmation.yaml
  cml/
    __main__.py
    environment.py       # discovery, versions, device probes
    data.py              # generator, public/private event split
    features.py          # action-independent history contract
    memory.py            # delta, KDA, GDN2, RLS references
    readers.py           # affine and nonlinear heads
    traces.py            # dense and compact auditors
    policies.py          # gate, heuristics, bounded dictionaries
    estimators.py        # PG, regression, ST, ARM, DisARM
    train.py             # snapshots, collection/replay, checkpoints
    evaluate.py          # paired streams, protected-test checks
    profile.py           # timing and memory boundaries
    analysis.py          # replicate aggregation and intervals
    records.py           # hashes, manifests, atomic I/O
  tests/
    test_trace_dense.py
    test_gradients.py
    test_timeline_masks.py
    test_no_leakage.py
    test_baseline_limits.py
    test_backend_parity.py
    test_resume.py
  artifacts/
    environment/
    pilot/
    preregistration/
    runs/<run_id>/
    reports/
  decisions/DECISIONS.md
  decisions/AMENDMENTS.md
```

Keep the small CPU checks in an optional `checks/` directory. The reference checks in Appendix B are sufficient to recreate them without this chat or another attachment.

### 9.2 Shared interface contract

```python
@dataclass
class PublicEvent:
    index: int
    kind: str                 # "observation" or "query"
    key_features: Tensor      # no hidden key truth or future metadata
    reported_class: int | None
    source_cue: float | None
    relevance_cue: float | None
    key_id: int | None        # allowed in exact-ID workload only

@dataclass
class TraceRecord:
    anchor: int
    action: int
    u: Tensor
    w: Tensor
    features: Tensor
    inclusion_probability: float
    snapshot_id: str
    raw_loss_difference: float
    target_kind: str          # "episode_Q" or "window_H"
    denominator: int
    expiry_event: int
    query_count_scored: int

class MemoryMethod(Protocol):
    def reset(self, batch_size: int, device: str) -> None: ...
    def read(self, q: Tensor) -> Tensor: ...
    def update(self, k: Tensor, v: Tensor, controls: dict) -> None: ...
    def mutable_state_bytes(self) -> int: ...
    def state_dict(self) -> dict: ...
```

The training harness owns feedback release, feature history, action randomness, timing, and evaluation—not each competing method independently. Baseline adapters must implement the same semantics or expose a documented incompatibility. `TraceRecord.raw_loss_difference` uses float64 accumulation on CPU or an equivalently tested accumulation strategy; the transported state may remain float32.

### 9.3 Configuration and experiment records

Illustrative **proposed** configuration; the agent must implement and validate its schema:

```yaml
schema_version: 1
stage: fixed_representation
workload: associative_stream_v1
seed_namespace: development
model:
  dk: 64
  dv: 64
  classes: 32
  reader: affine
  gate_hidden: 32
memory:
  beta: 0.5
  lambda_per_event: 0.995
  read_timing: pre_write
policy:
  method: trace_pg
  mode: bernoulli
  exploration_floor: 0.0
audit:
  selection: uniform_without_replacement
  anchors_per_episode: 8
  horizon: remaining_episode
  target_denominator: episode_query_count
  overflow: error
training:
  effective_batch: 16
  microbatch: auto_profiled
  optimizer: adam
  learning_rate: 0.001
  snapshot_scope: batch
  updates_per_collected_batch: 1
evaluation:
  audits_enabled: false
  primary_policy_mode: stochastic
  deterministic_threshold: validation_selected
```

Budget, replicate count, evaluation draws, and final hyperparameters are required fields in the frozen confirmation file, filled from the pilot—not left as the illustrative defaults.

Each `run.json` must include: schema version; run ID; exploratory/confirmatory status; source revision and dirty-tree hash; environment/dependency/data hashes; method/config; all seed namespaces; snapshot/checkpoint IDs; start/end timestamps; successful/failed/interrupted status; training/tuning/evaluation seconds; consumed events and queries; peak-memory measurements and their definitions; outputs; and linked methodological amendments.

`episodes.jsonl`: `run_id, replicate_id, condition, episode_id, policy_draw_id, loss_sum_nats, query_count, correct_count, accepted_writes, candidates, age_slice_sums, elapsed_seconds`.

`audits.npz` or equivalent typed records: anchor features, signed raw effects, denominator, full/truncated horizon, factual action, sampling probability, snapshot, scored-query count, and trace/dense discrepancy when checked. `decisions/DECISIONS.md` stores progress, current blockers, next action, and why a branch was taken. Never require chat memory to resume.

### 9.4 Risk-ordered work packages

All `python -m cml ...` commands below are **proposed project commands to implement**, not tools already available. Existing system commands are listed in Section 8.1.

#### WP00 — Establish a faithful local execution path

**Risk:** unsupported critical operations, unsuitable environment, or unaccounted resources. **Entry:** a fresh project directory and permission to use non-destructive local resources. **Dependencies:** none.

Implement `environment.py` and `profile.py`; run system discovery, dependency resolution, CPU reference, MPS operation probes, and a checkpoint round-trip. Command: `python -m cml probe --output artifacts/environment`. Outputs: machine report, package lock, operation results, backend recommendation, available-memory/disk estimate, and license manifest for dependencies/reused code.

**Resources:** tiny arrays, then one representative batch; no large download. **Accept:** at least one faithful CPU path works; any selected MPS path passes critical parity. **On failure:** fix environment or use CPU; if neither works, stop execution as an infrastructure blocker, not evidence against the mechanism. The optional text asset does not block synthetic work.

#### WP01 — Verify algebra, signs, clocks, and gradients

**Risk:** the indispensable estimator is wrong. **Entry:** WP00 CPU path. Implement `memory.py`, `traces.py`, and the numerical tests; reproduce Appendix A/B checks, extend to multiple simultaneous anchors, varying scalar controls, rejected anchors, zero residuals, no queries, reset/decay zero, repeated and orthogonal keys, padded batches, and nonlinear readers.

Command: `python -m cml verify --suite mathematical`. Outputs: maximum absolute/relative errors, dtype/device metadata, exact-enumeration gradients, and deliberately failing memory-feedback counterexample test. Check ARM/DisARM means, the sparse-audit inclusion correction, and the exact mean-memory/Jensen identity. Verify fractionalized inference for all applicable binary checkpoints. Add shared-parameter gradient enumeration; the checks performed for this report used free logits.

**Resources:** CPU seconds-to-small-batch scale, calibrated locally rather than promised here. **Accept:** discrepancies fit the numerical tolerance and all sign/timing invariants pass. **On failure:** block learning comparisons until corrected. If exact common-control parity passes but adaptive branches disagree, that is an expected scope limitation, not a trace bug.

#### WP02 — Build a valid task and test attainable headroom

**Risk:** hidden labels, no learnable signal, or an unsuitable proxy. **Entry:** WP01. Implement `data.py`, `features.py`, public/private separation, and the clean recall/control suites. Use dense branches on small episodes to measure full effects, positive/negative/zero prevalence, late-tail mass, and the fractional-gate gap. This gate must distinguish learning headroom from the absence of any stochastic-binary representational advantage in the affine setting. Train a cross-fitted restricted utility policy and compare cue-aware heuristics, an exact dictionary, and ordinary gates.

Command: `python -m cml headroom --config configs/anchor.yaml`. Outputs: generator hashes, leakage-test results, query-age/event histograms, oracle versus restricted-policy gaps, and baseline pilot losses.

**Resources:** small fixed-representation streams; dense all-anchor audits only at tiny `T`, sparse audits at anchor scale. **Accept:** valid harness plus a workload expressing the intended tradeoff. **If no oracle gap:** this task lacks relevant marginal headroom; diagnose load/geometry, do not assert global falsification. **If oracle gap but no cross-fitted gap:** investigate information limits or policy class before scaling. **If a cheap baseline solves the workload:** report that finding; do not artificially weaken it. A substantively different workload requires a new scoped claim.

#### WP03 — Establish strong baselines and reproducible timing

**Risk:** apparent novelty produced by weak or mismatched comparisons. **Entry:** WP02 valid tasks; sufficient potential headroom for a learning comparison. Implement the applicable methods in Section 6, tune on development/validation, and establish full-BPTT feasibility. Verify modern recurrence limits and any reused reference code against pinned source.

Command: `python -m cml baseline-pilot --config configs/anchor.yaml`. Outputs: per-method learning curves, search ledger, implementation versions, convergence/limiting-case checks, CPU/MPS profiles, and provisional Pareto plots.

**Resources:** measured per-method pilots, not the entire confirmatory matrix. **Accept:** strong methods learn and receive fair information/tuning; full counterfactual and trace paths use aligned reads. **On implementation failure:** repair or document an exclusion that narrows the claim. **On a clearly superior ordinary method:** test whether Trace-CF has a plausible cost niche before committing to broad training.

#### WP04 — Implement the complete minimum Trace-CF learner

**Risk:** exact targets fail to produce a correct or useful training update. **Entry:** WP01–03. Implement uniform anchor sampling, signed finite targets, affine-head projection, single-snapshot Trace-PG, Trace-Reg, and matched derivative credit. Keep audit computation out of autograd except the intended policy surrogate. Compare dense and compact training gradients on the same batches.

Command: `python -m cml train --method trace_pg --config configs/anchor.yaml --mode pilot`. Outputs: correct gradient checks, learning curves, utility/sign diagnostics, target-age records, audit overhead breakdown, and inference-without-audits parity.

**Resources:** start with fixed representations and calibrated microbatch. **Accept:** a minimum end-to-end learner runs, costs are measurable, and dense/trace training agrees numerically. **If targets are accurate but learning does not improve:** compare derivative, DisARM, saturation, regression calibration, and interactions; do not call lower utility MSE a success. **If trace cost loses to dense at this scale:** report the crossover and decide whether a scientifically relevant larger shape justifies testing.

#### WP05 — Calibrate budgets, sample size, and preregistration

**Risk:** underpowered or adaptively selected evidence. **Entry:** valid implementations and pilot curves. Perform independent pilot replications; estimate paired variance, evaluation Monte Carlo error, state/latency scaling, and tuning costs. Select the validation-best comparator, primary budgets, sample-size plan, and fixed workload parameters. Hash the protected-test manifest without reading outcomes.

Commands: `python -m cml design --pilot artifacts/pilot` and `python -m cml preregister --config configs/confirmation.yaml`. Outputs: `power_plan.json`, cost forecast with uncertainty, frozen configurations, exclusion rules, analysis tests, and confirmation hash.

**Resources:** determined by pilot variance and throughput. **Accept:** a fixed, feasible design can answer a stated question. **If infeasible:** choose a narrower but scientifically meaningful precision/claim or report resource-limited inconclusiveness. Do not drop a strong baseline simply to make a win affordable.

#### WP06 — Execute the protected fixed-representation comparison

**Risk:** the main learning/cost hypothesis fails. **Entry:** WP05 signed/frozen manifest. Execute all declared outer replicates and workloads; keep shared episode banks, randomized method timing order, checkpoint recovery, and raw unsuccessful outcomes. Auditing is disabled in task evaluation; a separate diagnostic audit job can label decisions without affecting them.

Commands: `python -m cml confirm --manifest artifacts/preregistration/manifest.json`; `python -m cml analyze --scope confirmatory`. Outputs: per-replicate effects, adjusted intervals/tests, cost frontiers, all declared regressions, and failure logs.

**Resources:** the calibrated fixed design. **Accept:** completion and valid evidence, not necessarily a positive result. **Continue:** only to questions supported by the result. **Negative:** distinguish no gain, dominated cost, or insufficient precision. Any new method suggested by test inspection starts a new exploratory branch and needs fresh confirmation.

#### WP07 — Test slow learning, attribution, and independent replication

**Risk:** fixed-model gains are artificial, stale, or not caused by finite credit. **Entry:** WP06 indicates a plausible benefit or an unresolved mechanism question worth the measured cost. Implement Stage B snapshot/replay learning; repeat strongest comparisons and target controls using fresh seeds. Run horizon, nonlinear-head, gate-sharing, cue, and information-access ablations in Section 10, prioritizing the alternative explanations actually supported by the pilot.

Command: `python -m cml robustness --plan artifacts/preregistration/robustness.json`. Outputs: attribution contrasts, representation-drift diagnostics, actual learned-system frontiers, and a clean-environment replication record.

**Resources:** measured Stage B batches plus selected factorial tests, not every possible combination. **Accept for a broader claim:** learning and attribution survive independent confirmation. **If only Stage A works:** narrow the result to fixed representations and analyze target drift. **If shuffling or constant labels match:** do not claim individualized delayed credit caused the gain.

#### WP08 — Conditional extension, not automatic escalation

**Risk:** synthetic-only results or restrictive controls limit usefulness. **Entry:** prior evidence justifies one specific extension. Choose either the small real-text experiment in Section 11.2 or the adaptive-gate branch diagnostic in Section 11.1 according to the observed bottleneck. Do not launch both just to accumulate experiments.

Commands: `python -m cml extend --plan <frozen_extension_config>`. Outputs: separate prerequisite checks, amended scoped hypothesis, resource calibration, strongest local baselines, and fresh test results.

**Resources:** local-only, projected from measured batches; no required large pretrained weights. **Accept:** a validated new scoped claim. **If blocked by license, data, or memory:** retain the completed synthetic result and state the extension remains untested. **If transfer fails:** report it; do not redefine the synthetic task after seeing failure.

#### WP09 — Synthesize, audit, and release the reproducible record

**Risk:** selective reporting or a handoff that cannot be reproduced. **Entry:** completed or explicitly blocked earlier work. Generate a report directly from recorded outputs; include unfavorable methods, uncertainty, hardware, total costs, amendments, limits, and the difference between measured and proposed results. Run a fresh-environment smoke reproduction and recovery test.

Command: `python -m cml report --include-failures`. Outputs: final Markdown report, compact machine-readable results, source/dependency/data manifests, decision log, and instructions for regenerating tables. **Resources:** analysis/reproduction cost logged separately. **Accept:** every reported number traces to an immutable run and every scientific claim has the required comparison. **On discrepancy:** correct the report and rerun affected analysis; do not manually edit tables to hide inconsistencies.

## 10. Attribution, robustness, and failure analysis

The following tests are ordered by the alternative explanation they resolve. A complete factorial cross of every setting is unnecessary; select development tests by information gain, then freeze the confirmatory subset.

| Test | Exact matched control | Interpretation and next action |
|---|---|---|
| Trace versus dense | Same anchors, action tape, reader, snapshot, and losses | Disagreement is a correctness defect. Agreement plus lower measured cost supports only an auditor contribution |
| Finite versus derivative | Same anchor/horizon/reader budget; record extra scoring and backward time | A persistent finite-credit advantage isolates the proposed signal; equality weakens that differentiation |
| Genuine versus shuffled labels | Permute effects across anchors from different episodes within snapshot, position, and scored-query-count strata; preserve target marginal scale | Loss of benefit supports individualized signal use. Surviving benefit requires constant-target and cue-only explanations |
| Constant utility | Replace effects by training mean within the same declared strata | Tests whether simply changing global write rate explains the result |
| Gate frozen / removed | Shared warm start, fixed admission rate, and all-write alternatives | Distinguishes adaptation from a good initialization or static sparsity |
| Horizon | `H=16,64,256` and full remaining episode; report actual late-tail effects | Determines truncation bias and whether a larger horizon is worth its cost |
| Audit coverage | Known-probability uniform sampling at multiple `K`; action-stratified sampling only as a separately weighted variant | Tests sample efficiency and whether overflow/selection created a result |
| Reader | Affine versus nonlinear read head, with identical exactness checks | Exposes reader-cost dependence rather than confusing algebraic storage savings with full-system speed |
| Representation drift | Fixed theta; snapshot-fresh joint learning; deliberately stale target replay | Distinguishes learned representation benefits from stale utility targets |
| Gate feature ablation | Remove relevance cue, source cue, count, age, or geometry separately; share changes with baselines | Detects generator shortcuts and identifies the actual information used |
| Memory-aware controller | Add a small declared set of memory probes; recompute complete branches in diagnostics | Measures the cost of the independence restriction; does not validate the old trace for changed controls |
| Capacity and geometry | Larger ordinary memory, RLS, different overlap/load, exact dictionary | Determines whether extra capacity or better geometry is the simpler explanation |
| Combined controller | Share an encoder between prediction and gate while matching parameter count and preserving allowed information | Tests a role-separation engineering choice, not an irreducible three-player principle |
| Fractionalized checkpoint | Replace binary draws by the same probabilities without retraining; keep the affine reader fixed | Tests the exact mean-memory/Jensen bound and reveals whether stochasticity itself carries a prediction penalty |
| Cost accounting | Include warmup, tuning, replay, target scoring, caching and evaluation; share optimizations | A gain erased by these costs is not a practical win |
| Evaluation harness | Identical-method A/A comparison; known beneficial/harmful write; sign-inverted target; no-memory and clean recall | Detects false gains, inverted signs, and insensitive evaluation |

When an ablation shares learned encoder parameters between the gate and the slow predictor, recompute memory-free gate features with gradients and add the policy-surrogate contribution to the direct task gradient for those shared parameters. Targets remain detached. Silently detaching the shared gate encoder would optimize a different objective; label that as a separate stop-gradient ablation rather than full joint learning.

For shuffling, only exchange records with the same model/policy snapshot; otherwise target nonstationarity becomes a confound. If a batch lacks enough records in a stratum, merge adjacent **predeclared** position/count bins or use multiple episodes in that snapshot, not future snapshots. Store the actual permutation. Shuffled and constant-target controls must have comparable compute: performing useless audits is acceptable for attribution, but their no-audit efficient versions belong in the system frontier too.

Do not threshold tiny estimated utilities into definitive “good/bad write” labels without a numerical tolerance and uncertainty discussion. For audited anchors report realized marginal decision regret `max(D_t,0) - a_t*D_t`, using full-episode `D_t` and inclusion weighting; this is nonnegative conditional one-action regret, not regret against an optimal joint policy. Compare this quantity and downstream loss as well as sign accuracy. Report whether failures concentrate on near-zero effects, long-delay facts, distribution shifts, or true changes after misleading observations.

## 11. Continuation criteria and evidence-triggered extensions

### 11.1 Primary selection versus alternatives

The restricted exact method is selected because its object is well-defined, its indispensable algebra is testable, and a complete local comparison is small. Utility regression alone is simpler to explain but introduces approximation and thresholding; it remains a companion. Ordinary continuous gates may be the best practical solution and are not a research failure if they win. General autoregressive counterfactual rollouts are not the primary direction because they remove the exact inexpensive closure and introduce a different problem.

**Conditional pivot: exact branches with sparse action divergence.** Pursue this only when a memory-conditioned gate demonstrably improves prediction and its paired branches usually diverge at few future actions. Assume the input, keys, values, decay and correction coefficients still remain exogenous/shared. Allow only the gate action to differ, with a small number of specified memory probes.

Let factual action be `a`, alternative action `b`, and `D = M_alt - M_f`. With `u_f = beta(v-lambda M_f k)`, subtract transitions:

\[
D'=\lambda D(I-b\beta kk^\top)+(b-a)u_fk^\top.
\]

Represent `D=UW^T`. Transport existing columns by

\[
W'=\lambda[W-b\beta k(k^\top W)],
\]

then append content column `(b-a)u_f` and key column `k` when the actions differ. Initialize the first intervention with its correct alternative-minus-factual sign. Rank is at most one plus the number of subsequent divergent actions, before dependencies among columns are exploited.

A probe `M_alt q` can be obtained as `M_f q + U(W^T q)`; use it to recompute the alternative gate with shared random uniforms. Reconstructing arbitrary full-memory features can erase the advantage. Switch to an exact dense alternative when factor storage or measured update cost exceeds a dense branch. Any rank truncation changes the estimator and requires a separate approximation study; it is not the default.

The identity was checked with forced divergent actions, not a trained adaptive policy. Derive and enumerate-test the sequential policy-gradient estimator before using these adaptive effects for learning; the product-Bernoulli proof in Section 4.5 cannot simply be reused for memory-dependent future features. Search for equivalent low-rank branching and finite-state sensitivity methods before claiming novelty. If rank quickly becomes large, this pivot is a diagnostic result, not an excuse to keep increasing complexity.

### 11.2 Conditional real-text experiment

A local teacher-forced language-model test is appropriate only after a credible synthetic benefit or a precise hypothesis about transfer. Begin with a small byte-level transformer trained locally: four layers, width 128, four attention heads, feed-forward width 512, context window 128–256, and a 64-dimensional memory block. These are calibration anchors, not assertions of adequate language-model quality.

Start by training one shared no-memory backbone locally and then freezing it for the memory-policy comparison; charge its project-specific training cost and give all methods the same warm-start asset. Learn the small memory maps/read head under the snapshot protocol. A later jointly trained backbone must include the shared-parameter policy-gradient term described in Section 10, or explicitly freeze/stop-gradient the gate feature path and narrow its claim.

At token time `t`, compute a memory-free backbone state from the fixed observed token prefix/window. Predict token `t+1` with logits `base_logits(h_t) + C(M_{t-1}q_t)`. Only after scoring, write keys/values derived from `h_t`. Never feed the memory-modified output back into backbone hidden states, the key/value cache, future gate features, or token inputs. The token tape is teacher-forced. This tests output-only memory augmentation, not unrestricted generation.

Candidate public asset: `Salesforce/wikitext`, configuration `wikitext-2-raw-v1`. The inspected card describes about 13.5 MB of decoded split data, so downloading the data is not the dominant expected resource. However, its metadata lists CC BY-SA 3.0/GFDL while prose names CC BY-SA 4.0. Treat that inconsistency as an explicit provenance/redistribution gate. Resolve original asset terms and attribution before redistributing text or derivatives; do not assert a clean license from the card alone. Core synthetic work does not depend on this resolution. [R21]

Pin the dataset revision and archive checksums, document loader version and preprocessing, retain official train/validation/test splits, reconstruct document boundaries, and do not count individual dataset rows as independent documents. A byte tokenizer avoids a separate pretrained tokenizer or vocabulary license. Train only on the training split; use validation for all choices. Reset at document boundaries and do not propagate a trace across a parameter update. For training chunks, freeze parameters for the complete scored chunk and retire traces at its end; persistent test-time document memory is a declared distribution difference to test, not silently equated with chunk training.

Compare no memory, all-write delta memory, full-BPTT gate, DisARM, derivative credit, Trace-CF, and a larger local-context baseline at matched end-to-end cost. Include a simple cache predictor/interpolation baseline when applicable. Report nats or bits per byte, with document-level aggregation and independent training replication. A gain here supports a small-model teacher-forced text claim only. It does not establish reasoning improvement, foundation-model efficiency, or autonomous memory management.

If the exact-ID synthetic task is dominated by dictionaries, a continuous-alias workload is an alternative **only with a new application rationale**. Keep true entity IDs private, remove exact-ID count features or replace them with charged observable matching, and add nearest-neighbor memory baselines. Do not remove IDs merely to make the proposed matrix win and retain the old claim.

### 11.3 Outcome classification

| Observed outcome | Permitted conclusion | Next action |
|---|---|---|
| Dense/trace mismatch under shared controls | Broken implementation or violated assumptions | Repair before any learning claims |
| Exact trace and cheaper audits, but no task gain | Useful auditor implementation at tested shapes, not superior policy learning | Publish/retain the narrower result; investigate cost crossover or stop |
| Oracle effects large, restricted policy cannot exploit them | Information/policy-class limitation | Diagnose observability; do not scale blindly |
| Trace-Reg predicts utilities but decisions do not improve | Regression/policy mismatch or interacting decisions | Compare direct PG, calibration, and constant-policy controls |
| Trace-PG beats naive methods but loses to tuned DisARM/BPTT | No demonstrated competitive learning advantage | Adopt the stronger method or narrow to estimator diagnostics |
| Benefit vanishes against dictionaries or stronger recurrence | The original bottleneck was not the best use of resources | Report the simpler practical winner |
| Fixed-model benefit survives cost and shift tests | Supported restricted local result | Attempt learned representations with fresh confirmation |
| Joint-learning benefit and attribution survive replication | Evidence for selective online memory learning in this scope | Consider a justified text or adaptive-gate extension |
| Positive average conceals a severe declared regression | A tradeoff, not unqualified improvement | Report slices and decide under an explicit deployment objective |
| Wide intervals, insufficient runs, unresolved baseline convergence | Inconclusive | State detectable effects and costs of resolving uncertainty |
| Synthetic benefit but no text transfer | Task-specific benefit | Identify representation/feedback differences; do not claim general transfer |

## 12. Checks performed for this report

These checks were executed during research, before writing the plan. They used **Linux x86-64 CPU, Python 3.13.5, NumPy 2.3.5**. The research environment also had CPU-only PyTorch; it was not an M3 Max and provides no direct evidence of MPS compatibility or runtime. The scripts are reproduced in Appendix B.

| Check actually run | Result | What it establishes / does not establish |
|---|---|---|
| 120 float64 trace trials; 80-step streams; dimensions `(3,5)`, `(16,16)`, `(64,64)`; both factual action choices; nonlinear reader | 16,680 paired-read checks; maximum state error `1.846e-15`; read error `1.057e-15`; nonlinear CE-utility error `1.332e-15` | Supports the implementation of shared-control trace algebra in these cases; not universal numerical stability |
| Same suite in float32, with explicit dtype assertions | Maximum state error `7.208e-7`; read error `5.960e-7`; nonlinear CE-utility error `4.648e-7` | Small errors in tested CPU trajectories, not MPS or very long-horizon validation |
| Exact enumeration of six Bernoulli actions, 64 trajectories | Local-expectation versus score-function mean difference `4.857e-17`; versus central finite difference `2.367e-11` | Checks the expected gradient identity for free logits in a tiny memory problem, not a learned shared gate |
| Exact integration of antithetic estimators over 729 uniform-space regions for a six-bit interacting loss table | ARM error `5.551e-17`; DisARM error `6.245e-17` | Checks signs and coefficients of the comparator formulas |
| Divergence-aware factor update, 48 steps and 23 forced action differences | Maximum dense/factor discrepancy `8.882e-16` | Checks algebra with diverging actions, not sparse rank growth or learned-policy value |
| Explicit scalar feedback counterexample | Fixed-control utility `+.875`; recomputed-policy utility `-.625` | Demonstrates a possible reversal outside the exact restricted interpretation |
| 100 affine-logit reconstruction trials | Maximum logit error `1.776e-14` | Checks the projection-caching identity |
| 20 expected-memory/Jensen trials, each enumerating 64 binary trajectories | Maximum state error `6.661e-16`; fractional loss never exceeded expected binary loss | Checks the architecture-specific prediction limitation, not optimizer performance |

The original specification referred to earlier experiments, including a curriculum failure and a small trace validation. Their code and outputs were not supplied, so they are treated as reported context, not independently reproduced evidence. The new checks above do not reproduce a gate-training advantage. There are no measured learning curves, Mac timings, energy estimates, or benchmark wins in this report.

## 13. Audit record and remaining risks

This is a self-audit conducted while constructing the plan, not an independent peer review. The final pass checked dependencies and interpretation after the following revisions.

### 13.1 Scientific and prior-art audit

| Material weakness found | Revision made | Residual risk / acceptance condition |
|---|---|---|
| Original novelty framing omitted a near-equivalent transported write-credit formulation | Added LatentHarness and made finite-versus-derivative evidence central | A prior finite-intervention implementation may still exist; no priority claim before refreshed citation/code search |
| Direct paired policy gradient could be advertised as new | Identified local expectation/Rao–Blackwell prior art and separated identity from implementation | Practical variance/cost advantage remains unmeasured |
| Three processes could be mistaken for a three-player principle | Removed that claim and made shared-controller separation only an ablation | No conclusion about component necessity without a matched test |
| Ordinary gates were implicitly characterized as weak local heuristics | Added exact smooth BPTT and strong binary estimators | Baseline tuning and implementation quality still require execution |
| An affine reader plus independent gates creates a Jensen disadvantage for stochastic binary inference | Derived expected-memory equivalence, executed an enumeration check, and mandated fractionalized-checkpoint evaluation | Only optimization-at-cost or measured write-cost gains remain plausible in that setup; nonlinear heads need separate evidence |
| Finite horizon was conflated with full-policy credit | Primary remaining-episode target; explicit truncated target and denominator | Full episodes are still finite; longer-term deployment needs a new objective |

### 13.2 Experimental-validity audit

| Material weakness found | Revision made | Residual risk / acceptance condition |
|---|---|---|
| Auditing only accepted or unexpired writes biases collected examples | Uniform candidate sampling with known inclusion probability; rejected-write signs tested | Any adaptive sampling extension must derive its weighting |
| An action-dependent count or prediction feature silently invalidates future-control equality | Public feature contract and intervention invariance test | Richer features require full/adaptive branches |
| Synthetic noise labels or ticket schedules could leak utility | Private annotations separated; noisy cues shared with baselines; feature perturbation tests | Generator cues may still be too convenient; shifts and text assess that limitation |
| Auxiliary frequency statistics can hide a second memory | Explicit byte accounting and no label history in primary features | Other methods may need richer state; compare frontiers honestly |
| A dictionary could solve exact-key recall more efficiently | Mandatory byte-budgeted dictionary/system comparison | Matrix-memory value may require a different, independently justified application |
| A positive utility-prediction score could replace task improvement | Downstream query loss is primary; regression diagnostic only | Thresholded and stochastic policies may disagree |
| Shuffling may retain useful average signal | Added constant-target and acceptance-rate controls | Shuffling is not a complete causal attribution proof |
| Five seeds or many query-level samples could imply false precision | Pilot-calibrated independent replication and fixed confirmatory design | Local compute may limit precision; inconclusive is allowed |
| Stronger tuning or extra replay could explain the win | Full cost ledger, shared optimizations, validation-selected comparator, and system frontier | Implementation optimization remains a substantial engineering variable |

### 13.3 Implementation and feasibility audit

| Material weakness found | Revision made | Residual risk / acceptance condition |
|---|---|---|
| Reference delta kernel used a different read clock and orientation | Pre-write contract, transposed-state parity, explicit scaling/dtype tests | Any optimized kernel needs the same checks |
| Negative utility was inconsistent with a sigmoid regression head | Separate unbounded utility output from policy probability | Robust regression may not estimate the conditional mean |
| Model updates could invalidate unfinished traces | Batch snapshots, fresh targets, no trace crossing; rollback/replay recovery | Delayed real feedback can require waiting or a separately derived off-policy method |
| Scalar constants accidentally promoted a float32 reference test | Added dtype-preserving constants and explicit dtype assertions before reporting final float32 results | New operations may reintroduce dtype or device conversions |
| Maximum M3 Max RAM and nominal backend support were assumed | Actual hardware discovery, operation probes, CPU fallback, microbatch adaptation | Local throughput remains unknown until WP00/WP05 |
| Documentation version labels and dataset license labels conflicted | Stable-wheel selection gate; optional dataset provenance gate | These prerequisites must be resolved locally rather than guessed |
| A mid-batch interruption could mix optimizer/snapshot states | Transactional batch-boundary checkpointing and forced interruption tests | MPS nondeterminism can prevent bitwise replay; disclose the achieved contract |
| Sparse divergence might be mistaken for guaranteed low rank | Dense fallback and explicit action-divergence/rank measurements | Adaptive auditing may be no cheaper than full branching |

### 13.4 Adversarial cases still required during execution

Stress nearly collinear keys, a gate saturated near zero/one, large but nearly canceling loss differences, zero correction vectors, decay resets, no future queries, heavily right-censored writes, two harmful writes that cancel each other, and a helpful write that becomes harmful after a genuine fact change. Verify that prediction of a query's own target cannot precede its release and that a “query-only” event cannot accidentally update memory with that label.

Also test heterogeneous episode lengths with padding, horizon expiration on query events, two traces created on adjacent events, target delay beyond a snapshot boundary, and numerical/backend differences that cross a decision threshold. Treat these as distinct failures rather than aggregating them into a single average error.

Remaining uncertainties are scientific rather than cosmetic: whether write utility is predictable beyond constructed cues, whether a hard admission decision is preferable to a soft update, whether the restricted architecture retains sufficient expressivity, whether finite credit has a favorable cost frontier, and whether any local benefit transfers. No result in this report resolves those questions.

## 14. References and search-coverage record

References below are primary papers, official documentation, author records, or inspected implementation/data sources. Dates identify the inspected version where important. A code link is not evidence that the code was executed. Technical summaries in Section 3 are deliberately narrower than a complete review of each paper.

**R1 — Jiaming Tang, Mingyan Liu, Armin Sarabi.** *Learning What to Remember: Long-horizon Counterfactual Memory Optimization.* arXiv:2609.37930v1, 29 September 2026. [Paper](https://arxiv.org/html/2609.37930v1). Inspected formulation, return/utility construction, and cost/limitations discussion. Supports overlap in long-horizon memory objectives; not a reproduction of its large-model experiments. No runnable official release was verified in this investigation.

**R2 — Xiaoqiang Wang, Suyuchen Wang, Bang Liu.** *LatentHarness: Learning Latent Actions for Memory and Reasoning via Counterfactual Policy Distillation.* arXiv:2609.39740v1, 30 September 2026. [Paper](https://arxiv.org/html/2609.39740v1). Inspected delta writes, counterfactual action branches, Equation 10, its explicit fixed-hidden-state partial-derivative caveat, and evaluation scope. Most consequential close prior art; no full-system code execution.

**R3 — Jiangze Yan et al.** *HiMPO: Hindsight-Informed Memory Policy Optimization for Less-Entangled Credit in Long-Horizon Agents.* arXiv:2606.16285v2, 1 September 2026. [Paper](https://arxiv.org/html/2606.16285v2). Inspected local counterfactual utility and hindsight filtering equations. Evidence of related memory-credit objectives, not identical state transport.

**R4 — Songlin Yang, Bailin Wang, Yu Zhang, Yikang Shen, Yoon Kim.** *Parallelizing Linear Transformers with the Delta Rule over Sequence Length.* arXiv:2406.06484v1, 10 June 2024. [Paper](https://arxiv.org/html/2406.06484v1). Inspected recurrence and parallelization structure. Establishes delta-rule and product-factor prior art; its accelerated execution should not be confused with a naive loop baseline.

**R5 — Songlin Yang, Jan Kautz, Ali Hatamizadeh.** *Gated Delta Networks: Improving Mamba2 with Delta Rule.* arXiv:2412.06464v1, 9 December 2024. [Paper](https://arxiv.org/html/2412.06464v1). Inspected scalar gating/decay formulation. Defines a key ordinary-learning baseline family.

**R6 — Kimi Team.** *Kimi Linear: An Expressive, Efficient Attention Architecture.* arXiv:2510.26692v1, 30 October 2025. [Paper](https://arxiv.org/html/2510.26692v1). Inspected KDA recurrence and channel-wise decay. Full large-model results are contextual; the specified local block is an adaptation.

**R7 — Ali Hatamizadeh, Yejin Choi, Jan Kautz.** *Gated DeltaNet-2.* arXiv:2605.22791v1, 21 May 2026. [Paper](https://arxiv.org/html/2605.22791v1). Inspected channel-wise erase/write equations. Provides a stronger update family; no claim to reproduce its complete training setup locally.

**R8 — Neehal Tumma, Noel Loo, Daniela Rus.** *Preconditioned DeltaNet: Curvature-aware Sequence Modeling for Linear Recurrences.* arXiv:2604.21100v1, 22 April 2026. [Paper](https://arxiv.org/html/2604.21100v1); [author-linked implementation](https://github.com/ntumm120/preconditioned-deltanet). Inspected preconditioning rationale, recurrence sections, and synthetic-evaluation scope; linked repository not executed. Motivates a geometry-based alternative and limits broad competitiveness claims when not faithfully included.

**R9 — Michalis K. Titsias.** *Local Expectation Gradients for Doubly Stochastic Variational Inference.* arXiv:1503.01494v1, 4 March 2015. [Paper](https://arxiv.org/html/1503.01494v1). Inspected conditional integration of a discrete variable. Establishes the statistical principle behind the proposed gradient wrapper.

**R10 — Runjing Liu, Jeffrey Regier, Nilesh Tripuraneni, Michael I. Jordan, Jon McAuliffe.** *Rao-Blackwellized Stochastic Gradients for Discrete Distributions.* arXiv:1810.04777, inspected v2; first posted October 2018. [Paper](https://arxiv.org/html/1810.04777v2). Inspected variance-reduction formulation; relevant statistical prior art, not a matrix-memory implementation.

**R11 — Mingzhang Yin, Mingyuan Zhou.** *ARM: Augment-REINFORCE-Merge Gradient for Stochastic Binary Networks.* arXiv:1807.11143, first posted 30 July 2018; ICLR 2019. [Abstract/version record](https://arxiv.org/abs/1807.11143); [OpenReview](https://openreview.net/forum?id=S1lg0jAcYm). Inspected metadata; formula cross-checked in the DisARM paper and against exact integration here. Related policy application: Yunhao Tang, Mingzhang Yin, Mingyuan Zhou, *Augment-Reinforce-Merge Policy Gradient for Binary Stochastic Policy*, arXiv:1903.05284v1, 13 March 2019, [paper](https://arxiv.org/html/1903.05284v1).

**R12 — Zhe Dong, Andriy Mnih, George Tucker.** *DisARM: An Antithetic Gradient Estimator for Binary Latent Variables.* arXiv:2006.10680v2, 3 December 2020; NeurIPS 2020. [Paper](https://arxiv.org/html/2006.10680v2). Inspected ARM review, Equations 6–8, assumptions, and comparison discussion. The paper also identifies concurrent U2G work; terminology search should include that alias.

**R13 — Corentin Tallec, Yann Ollivier.** *Unbiased Online Recurrent Optimization.* arXiv:1702.05043v2, 27 March 2017. [Paper](https://arxiv.org/html/1702.05043v2). Inspected rank-one recurrent gradient approximation and unbiasedness scope. Distinguishes a Jacobian estimator from the finite-state object here.

**R14 — Guillaume Bellec, Franz Scherr, Elias Hajek, Darjan Salaj, Robert Legenstein, Wolfgang Maass.** *Biologically inspired alternatives to backpropagation through time for learning in recurrent neural nets.* arXiv:1901.09049v2, 21 February 2019. [Paper](https://arxiv.org/html/1901.09049v2). Inspected eligibility/learning-signal decomposition. Historical trace-credit prior art, not evidence for finite-intervention superiority.

**R15 — Thomas Miconi, Jeff Clune, Kenneth O. Stanley.** *Differentiable plasticity: training plastic neural networks with backpropagation.* arXiv:1804.02464v3, 31 July 2018. [Paper](https://arxiv.org/html/1804.02464v3). Inspected slow/fast state formulation and implementation discussion. Supports the ordinary plasticity/BPTT alternative.

**R16 — Moonsu Han, Minki Kang, Hyunwoo Jung, Sung Ju Hwang.** *Episodic Memory Reader: Learning What to Remember for Question Answering from Streaming Data.* ACL 2019. DOI: 10.18653/v1/P19-1434. [ACL record](https://aclanthology.org/P19-1434/); [full text](https://arxiv.org/html/1903.06164v3); [author-linked code](https://github.com/h19920918/emr). Inspected streaming memory-management formulation. Code was identified, not run.

**R17 — Akanksha Jain, Calvin Lin.** *Back to the Future: Leveraging Belady's Algorithm for Improved Cache Replacement.* ISCA, June 2016. [Author publication record and paper link](https://www.cs.utexas.edu/~akanksha/). Metadata inspected as adjacent cache-retention history; the full implementation was not audited. This reference is not used to establish an exact equivalence or a performance claim.

**R18 — Flash Linear Attention contributors.** *flash-linear-attention* repository and naive gated-delta reference, inspected 13 October 2026. [Repository](https://github.com/fla-org/flash-linear-attention); [inspected reference source](https://raw.githubusercontent.com/fla-org/flash-linear-attention/main/fla/ops/gated_delta_rule/naive.py). Inspected README/platform claims, source license notice, state orientation, query scaling, dtype conversion, and output order. Source is MIT-licensed as indicated in the inspected files; pin a commit before reuse. No upstream execution here.

**R19 — Apple Developer.** *Accelerated PyTorch training on Mac.* Inspected 13 October 2026. [Official page](https://developer.apple.com/metal/pytorch/). Inspected setup requirements and backend availability check. Nominal support does not validate this project's operations or throughput.

**R20 — PyTorch contributors.** *MPS backend notes* and *torch.mps API documentation*, documentation resolved to 2.14 during research. [Backend notes](https://docs.pytorch.org/docs/2.14/notes/mps.html); [MPS API](https://docs.pytorch.org/docs/2.14/mps.html). Inspected device checks, synchronization, memory accounting, and RNG APIs. Installed-version compatibility remains a local prerequisite.

**R21 — Salesforce dataset repository.** *WikiText dataset card*, inspected 13 October 2026. [Card](https://huggingface.co/datasets/Salesforce/wikitext/blob/main/README.md). Inspected configuration/split metadata, approximate decoded byte totals, and inconsistent license labels. No text model was trained; legal/provenance reconciliation remains an optional-stage gate.

**R22 — PyTorch contributors.** *MPS environment variables.* [Official documentation entry](https://docs.pytorch.org/docs/stable/mps_environment_variables.html). Entry redirects by documentation version. Use the installed-version page and an actual operation/device-placement probe to verify fallback behavior; do not assume the redirect establishes execution compatibility.

### 14.1 Search scope and access limitations

Searches covered counterfactual memory utility, finite write interventions, rank-one delta transport, fast weights, eligibility traces, recurrent online gradients, local expectation/Rao–Blackwell estimators, ARM/DisARM/U2G, streaming memory policies, cache replacement, modern gated-delta variants, current public implementation constraints, Apple/PyTorch operations, and small public text assets. Backward references supplied historical equivalents; targeted searches for newer memory-credit work identified the most consequential additional competitor, LatentHarness.

Representative query families included “delta memory counterfactual write utility,” “finite rank one memory intervention,” “counterfactual future memory policy optimization,” “binary policy local expectation gradient,” “learned cache replacement future reuse,” and current Gated DeltaNet/FLA/MPS terms. Later targeted searches mainly added baseline strength, aliases, and implementation constraints rather than a broader defensible novelty claim. This was coverage-driven, not a claim that a quota of papers proves novelty.

Primary HTML papers, official documentation, and reference source code were opened for consequential technical claims. Public repositories were inspected only to the extent stated above; no upstream benchmark or training code was executed. Secondary search results were used for discovery, not as performance evidence. An adjacent September 2026 work, *Compact but Moving: Intervention-Relevant Geometry in Recurrent World Models* (arXiv:2609.21787), was found at the abstract level, but its full HTML was not accessible in this investigation; no novelty conclusion rests on it.

Blind spots include unpublished work, non-indexed implementations, newly revised September/October papers, full replication of large systems, and a meaningful patent landscape. A targeted patent search did not produce decisive technical evidence and is not a patent review. No claim is made that all related work has been found. These limits narrow a priority claim, not the validity of running the proposed controlled local experiment.

### 14.2 Final handoff contract

The coding agent may make routine non-destructive choices about file organization, vectorization, logging, and compatible package resolution. Changes to the hypothesis, observable information, event timing, objective, evaluation population, baselines, precision, or cost accounting require a recorded methodological amendment. Every expensive run must identify which uncertainty it resolves and which earlier gates passed.

A completed investigation may recommend the proposed method, a simpler alternative, a narrower auditor contribution, a justified extension, or stopping. Its success as research is an interpretable and reproducible answer—not a predetermined positive result.

## Appendix A. Reproducing the targeted checks

Save the three code blocks in Appendix B as `checks/validate_mechanism.py`, `checks/additional_checks.py`, and `checks/mean_memory_check.py` in the same directory. Install a compatible NumPy in an isolated environment, record its version, then run:

```bash
python checks/validate_mechanism.py
python checks/additional_checks.py
python checks/mean_memory_check.py
```

These are actual scripts executed for this report, unlike the proposed `cml` commands. They write JSON next to themselves. Numeric results may vary slightly with NumPy/platform, but the stated assertions should hold for the small CPU cases. The first script includes literal scalar counterexample summaries; the second explicitly simulates the feedback branches and validates those numbers. None of the scripts trains a policy, validates MPS, measures general throughput, or substitutes for WP01's additional batch/shared-parameter tests.

## Appendix B. Executed reference scripts

### B1. `checks/validate_mechanism.py`

```python
"""Small independent checks; not a learned-system benchmark. NumPy CPU only."""
from __future__ import annotations
import itertools, json, platform, sys
from pathlib import Path
import numpy as np


def step(M, k, v, a, lam, beta):
    P = M.dtype.type(lam) * M
    return P + M.dtype.type(a) * M.dtype.type(beta) * np.outer(v - P @ k, k)


def ce(z, y):
    z = np.asarray(z, dtype=np.float64)
    m = z.max()
    return float(m + np.log(np.exp(z-m).sum()) - z[y])


def trace_checks(dtype, ntrials=120):
    rng = np.random.default_rng(238517)
    max_state = max_read = max_loss = 0.0
    comparisons = 0
    for trial in range(ntrials):
        dv, dk = [(3, 5), (16, 16), (64, 64)][trial % 3]
        T = 80
        ks = rng.normal(size=(T, dk)); ks /= np.linalg.norm(ks, axis=1, keepdims=True)
        qs = rng.normal(size=(T, dk)); qs /= np.linalg.norm(qs, axis=1, keepdims=True)
        if trial % 4 == 0: ks[1::2] = ks[0]
        ks, qs = ks.astype(dtype), qs.astype(dtype)
        vs = rng.normal(size=(T, dv)).astype(dtype)
        actions = rng.integers(0, 2, size=T)
        lams = rng.uniform(.9, 1., size=T).astype(dtype)
        betas = rng.uniform(0., 1., size=T).astype(dtype)
        if trial % 5 == 0: betas[:] = 1
        if trial % 7 == 0: lams[:] = 1
        if trial % 11 == 0: lams[40] = 0
        t = trial % 20
        M = (rng.normal(size=(dv, dk)) * .1).astype(dtype)
        for s in range(t): M = step(M, ks[s], vs[s], actions[s], lams[s], betas[s])
        u = betas[t] * (vs[t] - lams[t]*M @ ks[t]); w = ks[t].copy()
        M0 = step(M, ks[t], vs[t], 0, lams[t], betas[t])
        M1 = step(M, ks[t], vs[t], 1, lams[t], betas[t])
        A = (rng.normal(size=(11,dv))/np.sqrt(dv)).astype(dtype)
        B = (rng.normal(size=(7,11))/np.sqrt(11)).astype(dtype)
        for s in range(t+1, T):
            D = np.outer(u,w)
            max_state = max(max_state, float(np.max(np.abs((M1-M0)-D))))
            r0, r1 = M0@qs[s], M1@qs[s]
            dr = u * (w @ qs[s])
            max_read = max(max_read, float(np.max(np.abs((r1-r0)-dr))))
            y = int(rng.integers(7))
            dense_gain = ce(B@np.tanh(A@r0), y)-ce(B@np.tanh(A@r1), y)
            for actual in (0,1):
                rf = r1 if actual else r0
                ra = rf + (1-2*actual)*dr
                lf, la = ce(B@np.tanh(A@rf), y), ce(B@np.tanh(A@ra), y)
                recovered = (2*actual-1)*(la-lf)
                max_loss = max(max_loss, abs(recovered-dense_gain))
                comparisons += 1
            assert M0.dtype == dtype and M1.dtype == dtype and w.dtype == dtype
            w = lams[s]*(w-dtype(actions[s])*betas[s]*ks[s]*(ks[s]@w))
            M0 = step(M0,ks[s],vs[s],actions[s],lams[s],betas[s])
            M1 = step(M1,ks[s],vs[s],actions[s],lams[s],betas[s])
    return dict(trials=ntrials, paired_read_checks=comparisons,
                max_state_abs_error=max_state, max_read_abs_error=max_read,
                max_nonlinear_ce_gain_abs_error=max_loss)


def exact_gradient_check():
    rng = np.random.default_rng(1925); T=6; dv=3; dk=4
    k=rng.normal(size=(T,dk)); k/=np.linalg.norm(k,axis=1,keepdims=True)
    q=rng.normal(size=(T,dk)); q/=np.linalg.norm(q,axis=1,keepdims=True)
    v=rng.normal(size=(T,dv)); C=rng.normal(size=(3,dv)); ys=rng.integers(3,size=T)
    z=rng.normal(size=T); p=1/(1+np.exp(-z))
    acts=list(itertools.product((0,1),repeat=T))
    losses={}
    for aa in acts:
        M=np.zeros((dv,dk)); loss=0.
        for s in range(T):
            loss+=ce(C@(M@q[s]),int(ys[s]))/T
            M=step(M,k[s],v[s],aa[s],.98,.7)
        losses[aa]=loss
    def expected(zz):
        pp=1/(1+np.exp(-zz))
        return sum(np.prod(np.where(a,pp,1-pp))*losses[a] for a in acts)
    sf=np.zeros(T); rb=np.zeros(T); sf_second=np.zeros(T); rb_second=np.zeros(T)
    for a in acts:
        prob=np.prod(np.where(a,p,1-p)); gsf=(np.array(a)-p)*losses[a]
        grb=np.zeros(T)
        for t in range(T):
            a0=list(a); a1=list(a); a0[t]=0; a1[t]=1
            grb[t]=p[t]*(1-p[t])*(losses[tuple(a1)]-losses[tuple(a0)])
        sf+=prob*gsf; rb+=prob*grb
        sf_second+=prob*gsf**2; rb_second+=prob*grb**2
    eps=1e-5; fd=np.zeros(T)
    for t in range(T):
        e=np.zeros(T); e[t]=eps
        fd[t]=(expected(z+e)-expected(z-e))/(2*eps)
    return dict(action_sequences=2**T, max_rb_vs_score_abs_error=float(np.max(abs(rb-sf))),
                max_rb_vs_finite_difference_abs_error=float(np.max(abs(rb-fd))),
                rb_gradient=rb.tolist(),
                per_coordinate_variance_ratio_vs_unbaselined_score=((rb_second-rb**2)/(sf_second-sf**2)).tolist())


def adaptive_rank_check():
    rng=np.random.default_rng(9713); dv,dk,T=5,7,48
    M0=rng.normal(size=(dv,dk))*.1; M1=M0.copy()
    U=np.zeros((dv,0)); W=np.zeros((dk,0)); maxerr=0.; divergences=0
    for s in range(T):
        k=rng.normal(size=dk); k/=np.linalg.norm(k); v=rng.normal(size=dv)
        lam=.99; beta=.6
        # Force independent branch control differences to test general identity.
        a0=int(rng.integers(2)); a1=int(rng.integers(2))
        residual=beta*(v-lam*M0@k)
        W=lam*(W-a1*beta*np.outer(k,k@W))
        if a1!=a0:
            U=np.column_stack((U,(a1-a0)*residual)); W=np.column_stack((W,k)); divergences+=1
        M1=step(M1,k,v,a1,lam,beta); M0=step(M0,k,v,a0,lam,beta)
        maxerr=max(maxerr,float(np.max(abs((M1-M0)-U@W.T))))
    return dict(steps=T, action_divergences=divergences, max_abs_error=maxerr)


def main():
    out=dict(environment=dict(platform=platform.platform(),python=sys.version.split()[0],numpy=np.__version__,backend='CPU; no Apple hardware'),
             trace_float64=trace_checks(np.float64), trace_float32=trace_checks(np.float32),
             gradient=exact_gradient_check(), adaptive_rank=adaptive_rank_check(),
             gate_feedback_counterexample=dict(accept_final=-.5, reject_fixed_final=-1., reject_adaptive_final=0.,
                                                fixed_control_utility=.875, adaptive_utility=-.625),
             finite_vs_tangent_counterexample=dict(r0=0.,r1=2.,target=.75,exact_utility=-.5,linearized_at_reject_utility=1.5))
    assert out['trace_float64']['max_nonlinear_ce_gain_abs_error']<1e-11
    assert out['trace_float32']['max_nonlinear_ce_gain_abs_error']<1e-4
    assert out['gradient']['max_rb_vs_finite_difference_abs_error']<1e-8
    assert out['adaptive_rank']['max_abs_error']<1e-11
    path=Path(__file__).with_name('validation_results.json'); path.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))

if __name__=='__main__': main()
```

### B2. `checks/additional_checks.py`

```python
"""Targeted CPU checks for the research plan; not a learning benchmark."""
from __future__ import annotations
import itertools, json, math
from pathlib import Path
import numpy as np
from validate_mechanism import step, ce


def main():
    rng = np.random.default_rng(81423)
    d = 6
    logits = rng.normal(size=d)
    p = 1/(1+np.exp(-logits))
    actions = list(itertools.product((0,1), repeat=d))
    # Deliberately interacting, nonsymmetric arbitrary binary loss table.
    losses = {a: float(rng.normal()) for a in actions}
    truth = sum(np.prod(np.where(a,p,1-p))*(np.array(a)-p)*losses[a] for a in actions)
    intervals = []
    for pi in p:
        ends = sorted(set([0., float(pi), float(1-pi), 1.]))
        intervals.append([(lo,hi) for lo,hi in zip(ends[:-1],ends[1:])])
    arm = np.zeros(d); disarm = np.zeros(d); nregions = 0
    for region in itertools.product(*intervals):
        u = np.array([(a+b)/2 for a,b in region]); mass = np.prod([b-a for a,b in region])
        plus = (u > 1-p).astype(int); minus = (u < p).astype(int)
        diff = losses[tuple(plus)] - losses[tuple(minus)]
        arm += mass * diff * (u-.5)
        disarm += mass * .5 * diff * ((-1.)**minus) * (plus != minus) / (1+np.exp(-np.abs(logits)))
        nregions += 1
    # Actual gate-feedback branching, with shared second input but recomputed action.
    M = np.zeros((1,1)); k = np.ones(1)
    accepted = step(M,k,np.ones(1),1,1.,1.)
    rejected = step(M,k,np.ones(1),0,1.,1.)
    rule = lambda state: int(state[0,0] > .5)
    af = rule(accepted)
    end1 = step(accepted,k,np.array([-2.]),af,1.,.5)
    end0fixed = step(rejected,k,np.array([-2.]),af,1.,.5)
    end0adaptive = step(rejected,k,np.array([-2.]),rule(rejected),1.,.5)
    loss = lambda state: .5*float(state[0,0]-1)**2
    fixed = loss(end0fixed)-loss(end1)
    adaptive = loss(end0adaptive)-loss(end1)
    # Affine-head counterfactual logits agree without rerunning a full reader.
    affine_err = 0.
    for _ in range(100):
        C=rng.normal(size=(32,64)); u=rng.normal(size=64); rf=rng.normal(size=64)
        b=rng.normal(size=32); alpha=float(rng.normal()); a=int(rng.integers(2))
        via_read=C@(rf+(1-2*a)*u*alpha)+b
        via_logits=(C@rf+b)+(1-2*a)*(C@u)*alpha
        affine_err=max(affine_err,float(np.max(np.abs(via_read-via_logits))))
    out = {
        'antithetic_regions':nregions,
        'arm_exact_integral_max_gradient_abs_error':float(np.max(np.abs(arm-truth))),
        'disarm_exact_integral_max_gradient_abs_error':float(np.max(np.abs(disarm-truth))),
        'simulated_feedback':{'accepted_final':float(end1[0,0]),'rejected_fixed_final':float(end0fixed[0,0]),'rejected_adaptive_final':float(end0adaptive[0,0]),'fixed_utility':fixed,'adaptive_utility':adaptive},
        'affine_logits_trials':100,'affine_logits_max_abs_error':affine_err,
        'decay':{'lambda':.995,'lambda_power64':.995**64,'lambda_power512':.995**512,'half_life_events':math.log(.5)/math.log(.995)}
    }
    assert np.max(np.abs(arm-truth)) < 1e-12
    assert np.max(np.abs(disarm-truth)) < 1e-12
    assert abs(fixed-.875)<1e-12 and abs(adaptive+.625)<1e-12
    assert affine_err<1e-12
    path=Path(__file__).with_name('additional_results.json')
    path.write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(out,indent=2))

if __name__=='__main__': main()
```

### B3. `checks/mean_memory_check.py`

```python
"""Exact small enumeration of the mean-memory/Jensen limitation; CPU NumPy."""
from __future__ import annotations
import itertools, json
from pathlib import Path
import numpy as np
from validate_mechanism import step, ce


def main():
    rng = np.random.default_rng(80712)
    T, dv, dk, classes = 6, 3, 4, 5
    max_state_error = 0.0
    gaps = []
    for _ in range(20):
        k = rng.normal(size=(T, dk)); k /= np.linalg.norm(k, axis=1, keepdims=True)
        q = rng.normal(size=(T, dk)); q /= np.linalg.norm(q, axis=1, keepdims=True)
        v = rng.normal(size=(T, dv))
        C = rng.normal(size=(classes, dv))
        y = rng.integers(classes, size=T)
        p = rng.uniform(.1, .9, size=T)
        soft = np.zeros((dv, dk)); soft_history = []; soft_loss = 0.
        for s in range(T):
            soft_loss += ce(C @ (soft @ q[s]), int(y[s])) / T
            soft = step(soft, k[s], v[s], p[s], .98, .7)
            soft_history.append(soft.copy())
        means = np.zeros((T, dv, dk)); expected_loss = 0.
        for actions in itertools.product((0, 1), repeat=T):
            prob = np.prod(np.where(actions, p, 1-p))
            M = np.zeros((dv, dk)); loss = 0.
            for s in range(T):
                loss += ce(C @ (M @ q[s]), int(y[s])) / T
                M = step(M, k[s], v[s], actions[s], .98, .7)
                means[s] += prob * M
            expected_loss += prob * loss
        max_state_error = max(max_state_error, float(np.max(np.abs(means-np.array(soft_history)))))
        gaps.append(float(expected_loss-soft_loss))
    # A nonlinear reader is outside the affine-logit Jensen conclusion.
    nonlinear_logits = lambda r: np.array([6*(r-.5)**2, 0.])
    binary_loss = .5*ce(nonlinear_logits(0.),0)+.5*ce(nonlinear_logits(1.),0)
    fractional_loss = ce(nonlinear_logits(.5),0)
    out = dict(trials=20, action_sequences_per_trial=2**T,
               max_expected_vs_fractional_memory_abs_error=max_state_error,
               min_binary_minus_fractional_ce_gap=min(gaps),
               max_binary_minus_fractional_ce_gap=max(gaps),
               nonlinear_reader_counterexample=dict(binary_expected_ce=binary_loss,
                                                    fractional_ce=fractional_loss))
    assert max_state_error < 1e-12
    assert min(gaps) > -1e-12
    assert fractional_loss > binary_loss
    Path(__file__).with_name('mean_memory_results.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(out,indent=2))

if __name__=='__main__': main()
```

[R1]: https://arxiv.org/html/2609.37930v1
[R2]: https://arxiv.org/html/2609.39740v1
[R3]: https://arxiv.org/html/2606.16285v2
[R4]: https://arxiv.org/html/2406.06484v1
[R5]: https://arxiv.org/html/2412.06464v1
[R6]: https://arxiv.org/html/2510.26692v1
[R7]: https://arxiv.org/html/2605.22791v1
[R8]: https://arxiv.org/html/2604.21100v1
[R9]: https://arxiv.org/html/1503.01494v1
[R10]: https://arxiv.org/html/1810.04777v2
[R11]: https://arxiv.org/abs/1807.11143
[R12]: https://arxiv.org/html/2006.10680v2
[R13]: https://arxiv.org/html/1702.05043v2
[R14]: https://arxiv.org/html/1901.09049v2
[R15]: https://arxiv.org/html/1804.02464v3
[R16]: https://aclanthology.org/P19-1434/
[R17]: https://www.cs.utexas.edu/~akanksha/
[R18]: https://github.com/fla-org/flash-linear-attention
[R19]: https://developer.apple.com/metal/pytorch/
[R20]: https://docs.pytorch.org/docs/2.14/mps.html
[R21]: https://huggingface.co/datasets/Salesforce/wikitext/blob/main/README.md
[R22]: https://docs.pytorch.org/docs/stable/mps_environment_variables.html
