# Research and local proof-of-concept plan: query-conditioned exponential-moment memory

**Working project name:** FAM-Jet (Functional Adaptive Memory with Temporal Jets)  
**Research cutoff and plan date:** 13 October 2026  
**Execution target:** one MacBook Pro with an M3 Max; installed memory, free storage, macOS and toolchain must be discovered. No remote GPU, paid model API or additional machine is required by the core program.  
**Deliverable type:** research assessment and executable experimental specification, not a report of trained-model results.  
**Primary decision:** whether exact temporal first moments improve the retrieval–resource tradeoff over strong, equally informed, equally budgeted alternatives.

## 1. Executive assessment

### 1.1 Verdict

**Investigate the mechanism, but withdraw the broad architectural-novelty claim.** A memory that represents a function of forgetting rate, exposes its derivatives, and permits later temporal queries is not a new category of neural representation. The proposed first-derivative recurrence is a repeated-pole linear system, closely related to classical gamma memory. Laplace-based temporal representations and the Time-Local Transformer substantially overlap with its interpretation. Most consequentially, **Multiscale HiPPO already represents a memory state as a function of inverse timescale and queries it after encoding**. Its authors also explicitly discuss its equivalence, under a change of coordinates, to a finite bank of HiPPO systems. [R01], [R02], [R03], [R04], [R05], [R06]

There is nevertheless a concrete empirical question worth answering:

> Under a fixed, fully accounted persistent-state budget, does a query-conditioned associative memory with exact, dimensionless first temporal moments provide more useful retrospective temporal information per byte and per unit of end-to-end cost than tuned distinct-pole memories, polynomial memories, temporal summaries and modern delta-rule memories?

A positive answer would support an **inductive-bias, numerical-conditioning or systems contribution**, depending on its cause. It would not demonstrate that function-valued neurons possess greater information capacity than vectors. A negative answer can identify when additional decay anchors, overwrite mechanisms or ordinary temporal summaries are preferable.

### 1.2 Selected direction and explicit changes to the original proposal

Keep the defining intent: **encode the history before learning the eventual query, then reuse the retained state for different temporal interpretations without replay**. Make four explicit refinements.

1. Store the value and the **dimensionless moment** `lambda * (-dS/dlambda)`, with normalization in an augmented value column. This changes coordinates, not the information retained.
2. Make the initial reader a **single-pass, query-conditioned decoder with access to all retained anchors**. Multi-step probing is an attribution experiment, not a prerequisite or an assumed source of additional information.
3. Exclude continuous interpolation from the indispensable mechanism. Test it separately, because unconstrained Hermite interpolation can turn a positive normalization function negative.
4. Establish the temporal-memory claim on controlled streams and a real sensor workload **before** attaching a language model. This is a staged component test, not a replacement of agent-memory validation by an easier benchmark.

The minimum complete core is a streaming writer, a fixed-size memory, a post-encoding reader, exact event-level targets, matched baselines, and measured ingestion/query costs. It does not need an LLM, custom GPU kernel, vector database, or numerical ODE solver.

### 1.3 What has and has not been established now

| Evidence class | Status as of this plan |
|---|---|
| Inspected external evidence | Original papers/technical articles, current documentation, selected official model/dataset cards, and implementation files listed in Section 16. |
| Analytical deductions | Exact moment recurrence; normalized derivative; gamma/Laguerre coordinate equivalence; finite-state and read-time-information limitations; resource formulas below. |
| Executed checks | Small NumPy CPU checks on Linux x86_64: direct-sum agreement, derivative agreement, equivalent-coordinate agreement, finite-difference cancellation, and one interpolation counterexample. Section 5 gives results and Appendix A makes the check reproducible. |
| Not executed | M3 Max compatibility tests, local latency/throughput measurements, dataset ingestion, neural training, baseline reproduction, confirmatory evaluation, or language integration. |
| Unresolved | Whether any accuracy advantage exists, whether it persists after baseline tuning, its practical value, and whether a publishable methodological distinction survives broader review. |

**No benchmark gain, Apple-hardware speed, training duration, or first-invention claim is asserted.**

### 1.4 Operational bottleneck and available headroom

The intended beneficiaries are systems that repeatedly ask different temporal questions about a stream whose full history is expensive or undesirable to retain: monitoring, bounded-memory event analysis, and eventually persistent agents. The workflow is `observe → compress once → answer multiple later queries`. The suspected bottleneck is not demonstrated demand from a particular customer; it is the loss of query-relevant temporal distinctions during compression.

Measure headroom before training elaborate readers. On exactly the same event histories, compute an exact full-history arithmetic answer and the best simple bounded-memory answer. For loss `L`, the local observable headroom is `L_simple - L_oracle`. This is not an attainable improvement forecast. On language tasks, repeat the decomposition with oracle evidence given to the same frozen decoder: poor oracle-evidence performance means that the decoder, not only memory, limits the experiment.

Let `C_m(T)` be ingestion cost and `R_m` query cost for method `m`. For `Q` queries after a history, total cost is

`E_m(T,Q) = preprocessing_m + C_m(T) + Q * R_m + output_generation_m(Q)`.

Report this quantity for the actual workload, not only the recurrent update. If jets cost more to ingest but less to query, their break-even query count is `(C_jet-C_base)/(R_base-R_jet)` only when the denominator is positive and the accuracy constraint is satisfied. Otherwise there is no such break-even. Compression does not avoid computing keys, values, token embeddings or language-model outputs.

## 2. Problem definition, assumptions and falsifiable hypotheses

### 2.1 Stream and query contract

A history is an ordered sequence of events

`e_i = (history_id, event_id, t_i, key_i, value_i, observed_mask_i, auxiliary_i)`.

Timestamps are monotone nondecreasing. A query is

`q = (history_id, query_id, T, key, task, temporal_arguments)`.

Only events with `t_i <= T` are available. The eventual query and its temporal arguments must not influence historical writes. A shared history state may serve many queries; reads must not change it. A query issued after the last event may cause **temporary** decay to its query time, without changing the stored snapshot. All methods receive the same raw events, public timestamps and query fields. A method may choose its own causal encoder in the learned-writer comparison.

The output is a numeric estimate and/or a task-specific categorical distribution, an explicit support/abstention output where applicable, and diagnostic costs. An uncertainty score is not automatically supplied by a derivative.

The principal tasks are **hard-window observed-event means** and **contrasts between adjacent historical windows**. These targets are intentionally not exponentially weighted means. They require different uses of the same retained history and do not make the proposed exponential basis the answer by construction. Latest-value, contradiction, absent-key and sharp-boundary recall are important secondary tests.

### 2.2 Distinctions that must remain intact

- **Elapsed time versus event count:** vary them independently. A long empty gap and many interfering writes are different challenges.
- **Post-encoding selection versus adaptive writing:** the former is the core claim. Every comparator may also use the query at read time.
- **Representation versus reader:** retain separate comparisons with fixed/shared writers and with independently tuned learned writers.
- **Numerical state versus total system memory:** fixed recurrent bytes do not include all weights, training activations, external indices or raw history unless explicitly counted elsewhere.
- **Approximation versus evidence:** a smoothly varying or internally consistent answer may still be wrong.
- **Simulation versus deployment:** sensor-window questions establish a realistic streaming calculation, not successful autonomous software maintenance.

### 2.3 Hypothesis register

| ID | Hypothesis and predicted observation | Support or motivation | Alternative explanation / weakening observation |
|---|---|---|---|
| H0-correct | The specified recurrence equals the corresponding direct sums, and its normalized derivative is correct. | Algebra; executed CPU checks. | Failure under ordinary finite ranges is a bug or numerical-design defect, not evidence of useful novelty. |
| H1-basis | Exact moments improve held-out hard-window/contrast loss relative to a tuned equal-byte distinct-pole bank. | Repeated poles supply age-weighted responses without subtracting close FP32 samples. | A better anchor grid, wider baseline reader, more tuning, or an invertible coordinate change eliminates the gain. |
| H2-reader | The structured derivative/age features aid sample efficiency or generalization beyond a generic decoder of the same state. | They expose a physically interpretable local scale response. | A generic full-state reader matches them; then credit the temporal basis, not derivative-aware reasoning. |
| H3-load | Any H1 gain survives a nontrivial range of key interference and history lengths. | This is necessary for associative, rather than single-signal, usefulness. | It appears only with collision-free keys or at negligible load; narrow the claim accordingly. |
| H4-cost | The advantage survives preprocessing, ingestion, repeated queries and decoder overhead on the target laptop. | Exact updates require ordinary elementwise operations and contractions. | Extra traffic, launch overhead, precision or reader computation removes the advantage. |
| H5-transfer | The selected configuration or tuning procedure improves the same task on unseen real sensor histories. | Real signals possess temporal structure, but this is an assumption about the chosen task. | Gains disappear under chronological shift, missingness, simple summaries or realistic costs. |
| H6-probe | Two or three adaptive probes beat a compute-matched one-shot reader given the same state. | Possible useful optimization/conditional-computation bias. | No gain, or gain disappears with equal depth/FLOPs. Multiple probes reveal no new stored information. |
| H7-language | A memory advantage improves answers through the same frozen language decoder. | A conditional extension only. | Oracle evidence fails, latent-to-text decoding dominates, or time-aware retrieval wins. |

H1 and H4 are the core research decision. H5 is required for a useful real-stream claim. H6 and H7 are not necessary to finish an informative core PoC. Failure of H6 does not negate H1. A failure of a low-sensitivity toy setting must not be generalized to regimes the setting did not represent.

## 3. Prior art, competing explanations and novelty boundary

### 3.1 Closest-alternative matrix

The costs below describe mechanisms, not incomparable published end-to-end timings. `K` is the number of temporal components, `d_k,d_v` associative dimensions, and `n` polynomial order.

| Family / inspected source | Problem and mechanism | Assumptions and evidence limits | Local implementation / material cost | Exact overlap and implication |
|---|---|---|---|---|
| Gamma memory / repeated-pole filters [R01] | Cascaded leaky states create polynomial-times-exponential impulse responses. | Classical temporal filtering, not a modern agent benchmark. | Elementary recurrence; linear state in cascade order. | The first jet is a gamma-state coordinate. No novelty in this recurrence alone. |
| Laplace temporal representations [R02], [R03] | Leaky integrators represent history in Laplace coordinates; derivatives/inversion expose temporal structure and associations. | Cognitive-model objectives differ from practical LLM retrieval. | Small reference systems are local; exact arbitrary inverse transforms are not finite-state. | Function-of-rate memory and temporal differentiation are established. |
| Time-Local Transformer [R04], [R22] | Attends over a compressed, approximately reconstructed timeline. | Biological plausibility is an explicit objective; published results are not a claim of modern LM leadership. | Official PyTorch Gaver cell; precision and input-discretization details matter. | Retrospective timeline access is already implemented. |
| HiPPO and multiscale/associative HiPPO Zoo [R05], [R06], [R07] | Polynomial history coefficients, continuum scale querying, and explicit associative updates. | Small demonstrations and reconstruction experiments do not establish dominance over all tuned baselines. | CPU reference plus native PyTorch port; temporal and scale orders consume state. | Closest direct challenge to the original gap. Compare, not merely cite. |
| RetNet / exponential retention [R08] | Multiple retained temporal scales with recurrent execution. | Full published architecture is more than an exponential bank. | Equal-byte `2J`-anchor bank is the decisive controlled comparator; label it retention-style, not full RetNet. | Tests whether more distinct decays already contain the useful information. |
| Gated DeltaNet, KDA and Gated DeltaNet-2 [R09], [R10], [R11] | Content-dependent decay and targeted overwrite; GDN-2 separates erase/write gates. | Published large-LM results use different training resources and tasks. | Small equation-faithful token mixers are local; official fused paths use Triton/FLA. Include convolution caches. | Strong contemporary alternative for interference, not merely temporal smoothing. |
| Mamba-3 [R12] | Improved state-space discretization, rotations and multi-input/multi-output recurrence. | Full architecture/training must not be replaced by a scalar EMA and called Mamba-3. | Reference context initially; native full-mixer comparison is an escalation if it changes the claim. | Modern recurrent competitors go beyond old scalar retention. |
| Mixture-of-Memories [R13] | Routed multiple memories. | Total stored banks, including inactive ones, matter. | Small routed delta-memory variant can be evaluated after load is identified as limiting. | Selection among memories and capacity allocation are not new in themselves. |
| Adaptive log-linear attention [R14] | Adaptive forgetting across a hierarchy. | Hierarchical storage regime differs from a strictly fixed-size state. | Report separately on an explicit memory–history curve. | Adaptive decay itself is not a novelty claim. |
| Full history + exact arithmetic | Stores all events; evaluates the requested function directly. | Growing storage; attainable exactness only for the specified numeric task. | CPU vectorized oracle, and measured practical full-history reference. | Separates information loss from prediction failure. |
| Bounded buckets, per-key summaries, ring/reservoir | Retain aggregates or selected exact events. | Can lose within-bucket detail or old facts. | Cheap CPU implementation; count timestamps, keys, counts and indices. | Strong simple methods can invalidate practical value even when neural comparisons look favorable. |

Neural-operator work already extends architectures to function spaces; the 2026 journal article cited in the motivating brief has an inspectable 2025 preprint. It is useful context but not a temporal-memory baseline. [R15] The strongest objections here arise from gamma/Laplace/HiPPO mechanisms, not from broadly related KAN, hypernetwork or continuous-dynamics terminology.

### 3.2 What can be reused

Reuse PyTorch autograd/tensor operations, NumPy/SciPy CPU numerical routines, the UCI raw dataset, and permitted reference implementation ideas. HiPPO Zoo supplies inspectable polynomial, integration, associative-memory and multiscale scripts under its repository license. Pin the inspected revision, retain notices, and validate any port against a small CPU reference. Do not install its entire experimental stack merely to run one recurrence. [R07]

GDN-2's inspected source license is **NVIDIA Source Code License-NC**, with research/evaluation use limitations. It is not a permissively licensed dependency. Keep any licensed reference checkout separate with its notices; document intended use before running or redistributing it. An independently written implementation of the paper equations still requires normal attribution and is not a legal clearance for commercial deployment. [R11]

The inspected Time-Local Transformer tree did not surface a license file. Treat source reuse/redistribution as blocked until permission or applicable licensing is established. The paper and a separately written equation-based reference can still inform a research comparison; do not silently vendor the code. [R22]

### 3.3 Precise candidate contribution

At present, the defensible candidate is:

> A controlled characterization, and possibly an effective implementation, of normalized associative exponential-moment states for deferred temporal queries, identifying when exact moments outperform distinct-pole, polynomial, overwrite and summary alternatives at measured resource budgets.

Possible outcomes are classified rather than relabeled:

| Finding | Appropriate contribution type |
|---|---|
| Exact same function as gamma memory in new notation | Reinterpretation or replication. |
| Same accuracy but faster/stabler native implementation | Engineering/numerical contribution. |
| Better generalization from a structured reader at identical information/cost | Inductive-bias or learning-method contribution, contingent on prior-art review. |
| Basis advantage across fair tasks and budgets | Empirical representation result; not greater capacity than all vector states. |
| Real application gain with known machinery | Application/systems result. |
| No advantage once controls are fair | Useful negative result and design guidance. |

The novelty claim would be invalidated by an earlier method with the same state equations and normalized read features, by an established basis implementation that achieves the same effect, or by a finding that only added state/compute explains the gain. **No exact-match search result is necessary to reject an overbroad claim:** the analytical equivalences already reject it.

### 3.4 Research coverage and remaining uncertainty

Searches covered function-space architectures; Laplace memory, temporal derivatives and inverse transforms; gamma/Erlang/Laguerre/repeated-pole filters; HiPPO/LMU/multiscale polynomial memories; retention, delta rules, KDA and recent 2026 recurrent models; read-time routing; interference studies; benchmark successors; MPS execution; and limited patent triage. Backward tracing led to gamma memory and the 2012 Laplace model; forward tracing led to HiPPO Zoo, GDN-2, Mamba-3, adaptive log-linear attention and LongMemEval-V2.

Later targeted searches reinforced the same competing mechanisms rather than establishing a clearly unclaimed architectural category. This is **coverage-based saturation for choosing this experiment**, not proof that all prior art has been found. Some publisher pages and late preprints were intermittently inaccessible. Preprint claims were not independently reproduced. Patent triage found the LMU patent family, including EP3935568A1; only technical disclosure/metadata were inspected, not a legal claim chart, validity analysis or freedom-to-operate assessment. [R23]

## 4. Mechanism and implementation specification

### 4.1 State, units and exact update

Let age be `a_i = T - t_i >= 0`, and let `lambda_j > 0` have units inverse time. Writes are independent of the rate parameter:

\[
X_i = g_i\phi(k_i)[v_i;1]^\top,\qquad g_i\geq0,\quad\phi(k_i)\geq0.
\]

The last value column is a normalization channel. At each anchor retain

\[
Z_j(T)=\sum_i e^{-\lambda_j a_i}X_i,\qquad
H_j(T)=\sum_i(\lambda_j a_i)e^{-\lambda_j a_i}X_i
       =-\partial_{\log\lambda_j}Z_j(T).
\]

`Z,H` have shape `[batch, heads, anchors, key_dim, value_dim+1]`. Each history also stores its last timestamp and validity status. Rates are model parameters/buffers, not a separate copy per history, unless they are genuinely history-specific. Count such copies if introduced.

For elapsed time `Delta`, define `d_j=lambda_j*Delta`, `a_j=exp(-d_j)`:

\[
H'_j=a_jH_j+(a_jd_j)Z_j,\qquad Z'_j=a_jZ_j+X_t.
\]

Use the **old** `Z` in the moment update. A new impulse has age zero and contributes nothing immediately to `H`. Multiple simultaneous events are additive; their order is immaterial for this additive writer, but not necessarily for overwrite baselines.

Proof: every old age becomes `a_i+Delta`, so its exponential gains `exp(-lambda*Delta)` and its dimensionless age gains `lambda*Delta`. This is exact over a gap, not Euler integration. Input-independent, nonuniform event timing requires no ODE solver.

The derivative is with respect to a continuous family whose historical writes are held fixed. If `g`, `k` or `v` depends on the probed rate, product-rule terms are required. A content-conditioned writer is permitted if it is independent of that rate and causal. The terminal contraction in Section 4.5 additionally requires a writer that can be computed without its own recurrent memory.

### 4.2 Read and support features

For a nonnegative query feature `p=phi(q)`, contract the key axis. Write

- `u = p^T Z[..., :d_v]`, `z = p^T Z[..., d_v]`;
- `h = p^T H[..., :d_v]`, `b = p^T H[..., d_v]`.

With a fixed rate-independent `eps > 0`,

\[
r=\frac{u}{z+\epsilon},\qquad
\partial_{\log\lambda}r=-\frac{h}{z+\epsilon}
                         +\frac{r b}{z+\epsilon}.
\]

For genuinely supported reads, the dimensionless mean age is `b/z`; mean age in original units is `b/(lambda*z)`. Compute these only above a preregistered numerical support floor. Return a support flag, and zero-fill undefined features. The smoothed ratio `b/(z+eps)` is not an exact mean age.

When `eps=0`, the derivative is `-lambda * Cov_w(age,value)` under query/key/decay weights. It describes a temporal contrast, **not epistemic uncertainty**. Values may be signed even though normalization weights are nonnegative.

Initial reader inputs, per anchor, are `[r, dlog_rate_r, log1p(z), b/(z+eps), log(lambda*t_ref), support]`, concatenated with query identity, task and normalized temporal arguments. Preserve raw contracted `u,h,z,b` as an alternative reader input: ratios may hide useful amplitude or saturate. Use a fixed training-derived reference duration `t_ref` (initially the median positive training gap); express query endpoints as age/`t_ref` and rate features as `log(lambda*t_ref)`. Under a change of physical time units, transform `t_ref` with the timestamps. Fit all feature standardization constants on training only.

`eps` and numerical support floors must be recorded in the configuration. Start with FP32 `eps=1e-8` for order-one normalized writes; sweep `1e-6,1e-8,1e-10` on development data and include rescaling tests. These are numerical starting values, not evidential support thresholds. A learned abstention threshold is chosen on a separate development calibration set and frozen.

### 4.3 Ordinary-vector and gamma controls

Between impulses:

\[
\dot Z=-\lambda Z,\qquad \dot H=-\lambda H+\lambda Z.
\]

This is a two-state repeated-pole/gamma system. Calling `H` a jet does not add information beyond this state. [R01]

Use the invertible coordinate control

\[
L_0=Z,\quad L_1=Z-2H,\quad H=(L_0-L_1)/2.
\]

Its exact impulse update is

\[
L'_0=aL_0+X,\qquad L'_1=aL_1-2adL_0+X.
\]

The associated kernels are `exp(-lambda*t)` and `(1-2*lambda*t)*exp(-lambda*t)`, a scaled two-function Laguerre span. This control must recover identical predictions when transformed back before the same reader. A claim that the jet representation intrinsically beats this equivalent state is an implementation or optimization claim, not an expressivity result.

A distinct-pole bank approximates a log-rate derivative by

\[
-\partial_{\log\lambda}Z(\lambda)
\approx-\frac{Z(\lambda e^\delta)-Z(\lambda e^{-\delta})}{2\delta}.
\]

Small `delta` reduces truncation error but can magnify FP32 cancellation. Permit the baseline to tune spacing and to use all raw anchors directly. Do not force it to use a numerically poor difference formula.

### 4.4 Reader and ablation architecture

Use a two-hidden-layer SiLU MLP as the default numeric reader; width 128 is only a smoke/pilot starting point. Output the requested value vector and an independent support logit where required. Use task identifiers and continuous interval endpoints, not a separate head for every training horizon.

The following comparisons are distinct:

| Reader | Inputs | Purpose |
|---|---|---|
| Structured | Contracted anchor values, slopes, support and age features plus query | Candidate convenient inductive bias. |
| Generic contracted | All raw contractions `u,h,z,b` plus query | Same information without hand-selected ratios. |
| Generic full state | Flattened `Z,H` plus query, with factorized first layer when necessary | Tests whether restricting access favored the proposal or comparator. |
| Fixed scale | Development-selected anchor(s), no query-conditioned scale routing | Diagnostic lower control, not the main competitor. |
| Value-only bank reader | All `2J` raw/normalized anchor reads and query; optional learned differences | Main H1 comparator. |

Compare both parameter-matched and compute-matched reader frontiers. Equal hidden width is not parameter matching when input widths differ. Do not add unused parameters as padding. Fit widths/ranks under explicit budgets and report any residual mismatch.

For the optional probe version, a controller produces a distribution over stored anchors, reads a mixture, updates its hidden vector, and repeats for two or three steps. The comparator gets the entire available state and equal total reader depth/FLOPs. Report all anchor contractions, not just the final chosen anchor, in probe cost. If every anchor was already read to choose the probe, claim no selective-bandwidth saving.

### 4.5 Exact terminal training without storing every recurrent state

For terminal queries and state-independent writers, compute the final state by chunked contractions:

\[
Z_j=\sum_i w_{ji} g_i\phi(k_i)[v_i;1]^\top,\quad
H_j=\sum_i (\lambda_j a_i)w_{ji}g_i\phi(k_i)[v_i;1]^\top,
\quad w_{ji}=e^{-\lambda_j a_i}.
\]

These expressions give exact gradients through the writer and anchors while avoiding a tensor of shape `[B,T,J,d_k,d_v+1]`. Form weights times keys as `[B,J,chunk,d_k]`, then matrix-multiply against values, accumulating across chunks. Recompute history states after every parameter update; cached states made with old rates or writers are invalid.

Apply the same optimization to value-only memories and other eligible linear baselines. For nonlinear/recurrent encoders and overwrite baselines, use full backpropagation with checkpoint/recompute, or a verified associative scan. Do not give the candidate full history gradients while silently truncating a baseline's gradients. Temporal samples may be generated offline for training, but **online ingestion timing must use the actual incremental implementation**.

### 4.6 Reference update interface and pseudocode

The following is a proposed project implementation, not an existing installed package API. The vectorized core deliberately contains no host synchronization. The public wrapper validates inputs before calling it.

```python
from dataclasses import dataclass
import torch

@dataclass
class JetState:
    z: torch.Tensor  # [B, H, J, K, V+1]
    h: torch.Tensor  # same layout; dimensionless first moment
    # The owning stream object keeps last_time as CPU float64/int64.


def jet_step(state: JetState, key: torch.Tensor, value: torch.Tensor,
             gate: torch.Tensor, delta: torch.Tensor, rates: torch.Tensor,
             event_present: torch.Tensor) -> JetState:
    """Inputs: key[B,H,K], value[B,H,V], gate[B,H], delta[B],
    rates[H,J], event_present[B]. All floating inputs/core states FP32.
    Validated upstream: finite values, key/gate >= 0, rates > 0,
    delta >= 0, and masked padding has delta=0 and event_present=False.
    A real no-write gap has delta>0 and event_present=False.
    """
    d = delta[:, None, None] * rates[None, :, :]  # [B,H,J]
    a = torch.exp(-d)
    # Avoid inf*0 on extreme finite gaps; normal d=0 is handled exactly.
    safe_d = torch.where(torch.isfinite(d), d, torch.zeros_like(d))
    ad = torch.where(torch.isfinite(d), a * safe_d, torch.zeros_like(d))
    a = a[..., None, None]
    ad = ad[..., None, None]
    augmented = torch.cat((value, torch.ones_like(value[..., :1])), dim=-1)
    write_gate = gate * event_present[:, None].to(gate.dtype)
    write = (write_gate[..., None, None] * key[..., :, None]
             * augmented[..., None, :])
    old_z = state.z
    new_h = a * state.h + ad * old_z
    new_z = a * old_z + write[:, :, None, :, :]
    return JetState(new_z, new_h)
```

This core handles an overflowing positive `d` as complete decay; the wrapper must reject NaNs, negative deltas and nonfinite input values rather than letting them masquerade as long gaps. For very small `d`, `a≈1` is expected. Validate gradients away from the overflow branch. Empty streams return zero state and unsupported reads.

Common method interface to implement:

```text
init_state(batch_shape, device, dtype) -> State
encode(events, state=None, mode="stream" | "terminal") -> State
advance_copy(state, query_time) -> State
read(state, query_batch) -> Prediction, Diagnostics
persistent_nbytes(state) -> integer
parameter_nbytes() -> integer
save_state/load_state(..., schema_version, parameter_hash) -> State
```

`encode` cannot access query fields. `read` may allocate temporary workspace but cannot read disk history in a bounded-state run. Use a sandboxed evaluation wrapper that passes only the snapshot and query to enforce this. Static weights and public schema knowledge are allowed; per-history caches hidden in an encoder object are not.

### 4.7 Numerical and semantic invariants

Compute timestamp differences in CPU float64 or integer ticks **before** conversion to FP32. Never subtract Unix-epoch timestamps after storing them in FP32. Fix one time unit for each dataset and record it; rescaling all times and `t_ref` by `c` and all rates by `1/c` should preserve `Z,H`, dimensionless reader inputs and numeric answers. For impulse-driven polynomial states carrying inverse-time units, scale their reader features by `t_ref` (or use the corresponding integrated reconstruction) before comparing unit conversions.

Reject or explicitly quarantine out-of-order events; do not sort a deployed stream using future information. For an offline dataset, stable sorting by `(timestamp,event_id)` is allowed and recorded. Additive simultaneous writes must be permutation-invariant. An overwrite baseline may use this stable order because order is part of its input semantics.

Check `z>=0` within numerical tolerance, finite states and gradients, old-state use, padding invariance, chunk-boundary equivalence, query-order invariance, checkpoint continuity and rate-parameter invalidation. A nonnegative support estimate is not proof that the retrieved association is correct.

For missing sensor values, use one observed scalar event per `(station,variable)` or independent normalization for each value channel. A single shared normalization column is invalid when different value coordinates have different observation masks. The initial real-data experiment uses one pollutant, avoiding this ambiguity.

### 4.8 Interpolation is a gated extension

To evaluate intermediate rates, interpolate **unnormalized** `Z` and its normalization column in `x=log(lambda)`, using slope `-H`. Do not interpolate normalized reads and describe the result as an exact associative read. Cubic Hermite interpolation has no automatic positivity guarantee, even for a positive decreasing normalization function.

Only enable interpolation after measuring error against exact direct sums over a dense query-rate grid on development histories. Log negative denominators, relative/absolute numerator errors, cancellation, boundary errors and fallback frequency. Do not extrapolate beyond the stored range. A denominator floor or fallback to an anchor changes the function; report the resulting bias and added cost. A monotonicity-preserving scalar normalizer does not guarantee accurate signed numerator interpolation. Freeze trust thresholds before confirmation. No interpolation certificate is claimed by this plan.

## 5. Checks actually performed during this investigation

### 5.1 Environment and fixture

The executed program used Python **3.13.5**, NumPy **2.3.5**, CPU on **Linux x86_64**. It is not a test of PyTorch, Metal or an M3 Max. Seed: `20261013`. The fixture contained 512 events with lognormal time gaps, positive random key features of dimension 5, three signed value coordinates, a normalization column, random nonnegative gates, and four rates geometrically spaced from `0.001` to `1`.

The reference explicitly summed all event contributions at the final time. Relative error means Frobenius norm of the difference divided by the reference norm, with a tiny zero-denominator guard. Appendix A contains the complete independent checker.

| Executed check | Observed relative error |
|---|---:|
| FP64 recurrent `Z` versus FP64 direct sum | `1.0731e-15` |
| FP64 recurrent `H` versus FP64 direct sum | `1.8252e-15` |
| Recovered FP64 gamma/Laguerre-coordinate `H` versus jet recurrence | `6.8016e-16` |
| FP32 recurrent `Z` versus FP64 direct sum | `6.9465e-7` |
| FP32 recurrent `H` versus FP64 direct sum | `7.8878e-7` |
| Recovered FP32 equivalent-coordinate `H` versus FP32 jet recurrence | `7.8502e-7` |
| Analytic normalized log-rate derivative versus FP64 central difference | `2.4007e-11` |

These checks establish consistency on this fixture, not universal stability. They do not measure learnability, information retention on a task, or superiority to a tuned exponential bank.

### 5.2 Finite-difference diagnostic

For direct-sum states, approximate the moment using log-rate offsets `±delta`:

| `delta` | FP64 relative moment error | FP32 relative moment error |
|---:|---:|---:|
| `1e-1` | `7.3239e-4` | `7.3218e-4` |
| `1e-2` | `7.3349e-6` | `4.9763e-5` |
| `1e-3` | `7.3350e-8` | `7.3394e-4` |
| `1e-4` | `7.3254e-10` | `1.0852e-3` |
| `1e-5` | `9.4103e-11` | `2.8422e-2` |
| `1e-6` | `1.0106e-9` | `1.3109e-1` |

This supports testing numerical conditioning. It does **not** show that a value-only bank should approximate derivatives or use close anchors at all. A direct reader of separated anchors may be better than either numerical differences or jets.

### 5.3 Interpolation counterexample

For one positive impulse of age 1, the normalization function is `N(lambda)=exp(-lambda)`. Cubic Hermite interpolation in log-rate between anchors `1` and `1000`, using exact endpoint slopes, attained approximately **−0.135026** at rate **25.5623**; the true value there is approximately **7.9147e-12**.

The interval is deliberately broad. This is a counterexample to guaranteed positivity, not a claim that carefully spaced interpolation always fails. It is why continuous interpolation is an independently validated extension rather than a condition for running the core experiment.

## 6. Datasets, tasks and protected evaluation populations

### 6.1 Synthetic stream generator: precise starting specification

Implement `data/synthetic.py` as a pure deterministic generator: `(generator_version, split_seed, episode_id, config) -> events, queries, oracle_labels`. Different seed namespaces generate training, development, pilot, confirmation and independent replication. No target is computed from a compressed representation. Generate the entire history before sampling its queries.

**Primary temporal-basis setting.** Start with `K=16` public key identities, one-hot key features, and `d_v=8` independent numeric value coordinates. The fixed writer uses the raw values, `g=1`, and no trainable projection. This intentionally isolates temporal representation from key learning. Its limits must be stated in the result title. Run the dense-key/interference setting below before making an associative-memory claim.

For each history, sample one of four equally weighted signal regimes. For each key and value coordinate, independently initialize a level from `Normal(0,1)`. The regimes are:

- **Piecewise stationary:** at a key's observations, with probability `1/32`, replace its level with a fresh draw; observation noise has standard deviation `0.1`.
- **Slow trend plus resets:** level increments by a per-key slope drawn from `Normal(0,0.01)` in reference-time units; reset level and slope with probability `1/128` per key observation. Add noise `0.1`.
- **Periodic plus impulses:** a sinusoid with random phase, amplitude uniform `[0.5,1.5]`, period log-uniform `[16,256]` time units, plus independent sparse impulses with probability `1/64` and amplitude `Normal(0,2)`; noise `0.1`.
- **Irregular revisions:** a piecewise level as above, plus paired excursions and later reversals separated by a random 1–16 observations of that key. Do not reveal the future reversal in an input flag.

These numbers define reproducible *initial* development distributions, not universal benchmarks. Before confirmation, freeze any pilot-driven changes and retain the original results. Include separate out-of-distribution tests with broader periods, faster resets, heavy-tailed noise and reversal gaps not used in training.

Sample keys uniformly for the primary set; use Zipf-like key frequencies only as a separate frozen robustness set. Select gaps from `{0,1,2,4,16}` with probabilities `{0.05,0.55,0.25,0.10,0.05}`. Independent no-write-gap fixtures isolate elapsed time. This finite gap alphabet permits exact, cached polynomial transitions; continuous lognormal gaps are an additional robustness test, not an excuse to weaken a polynomial comparator.

Initial pilot lengths are `128,512,2048` events. Choose two or more confirmatory lengths using development-only headroom curves: one where temporal compression is nontrivial, and a longer one that materially increases observations per key or elapsed span. Keep the full selected length distribution identical across methods. A likely next stress length is `4096`, but it is not a hardware promise or an upper limit.

**Dense-key/interference setting.** Provide public 64-bit key IDs plus a deterministic, stateless positive feature map derived from each ID and a fixed feature-map seed. Use normalized squared Gaussian features in dimension `d_k=16` initially; document the pseudorandom hash-to-Gaussian transformation. Do not keep a per-history key dictionary off-budget. Use loads `K=8,16,64`, drawing disjoint IDs across training and test. All methods receive the raw IDs/features and may use their allowed causal encoders. Ring/bucket baselines may store exact IDs, counting those bytes. Report interference loss separately from temporal-basis loss. Repeat with learned key/query maps on raw input features; that is a separate learned-writer arm.

Fixed positive features deliberately impose overlap. A loss there is not evidence that arbitrary learned associative addressing is impossible. Conversely, a win only with one-hot keys supports a narrower temporal-summary claim.

### 6.2 Query and label generation

Let `A=T-t_first` be the available age span. Draw hard-window widths log-uniformly from `[max(1,A/128), A/2]`, then draw a lower age endpoint uniformly so the window lies inside the history. Use half-open age intervals `[a,b)`; `[0,b)` includes observations at the current time. For latest-value ties, use the stable `(timestamp,event_id)` order. Record the rule exactly in data metadata.

For requested key `k`, the mean target is

\[
y_{\mathrm{mean}}=\frac{\sum_i\mathbf1\{k_i=k,\ a\leq T-t_i<b\}v_i}
 {\sum_i\mathbf1\{k_i=k,\ a\leq T-t_i<b\}}.
\]

Contrast targets subtract the mean in `[a+w,a+2w)` from the mean in `[a,a+w)`. Sample `w` and `a` so both windows lie within the available span. Each task has equal primary weight; each history contributes equal weight regardless of its number of queries.

For primary numeric loss, labels require at least one observed event in every needed interval. Generate invalid/empty cases independently for the abstention suite rather than manufacturing arbitrary numeric targets. Draw query keys independently of values. Do not sample only around change points. Include a separate adversarial boundary set with events just inside/outside an interval.

Generate an initial eight queries per history, balanced across the two numeric tasks, to estimate within-history variability. The final number is chosen from pilot variance/cost: more queries can stabilize a history's mean but **do not increase the independent history count**. Create fresh queries from a separate seed after selecting a history, and verify that permuting queries leaves state hashes unchanged.

Secondary tasks have explicit labels: latest observed value for a key, latest corrected categorical fact, presence/absence, and change sign. Add exact key/value binding with random categorical values as a capacity stress test. State clearly that exact sparse recall and smooth aggregates are different objectives.

**Anti-tailoring controls:** include an exponential weighted-average task only as a labeled diagnostic where the model's kernel matches the target. It cannot be the headline primary result. Include all-equal values, constant signals, single impulse, random unstructured values, and a direct full-history oracle. A method should not gain credit for solving a task whose answer was embedded in its input metadata.

### 6.3 Real workload: Beijing multi-site sensor histories

Use the UCI **Beijing Multi-Site Air Quality** dataset, DOI `10.24432/C5RK5G`, archive `PRSA2017_Data_20130301-20170228.zip`. The catalog documents 12 sites, 420,768 hourly records, observations from March 2013 through February 2017, missing values, and a CC BY 4.0 license. These are source metadata, not counts independently verified during this investigation. [R16]

The initial task uses **observed PM2.5 values across all 12 stations**, one scalar event per observed `(station,hour)`. This avoids hidden per-channel missingness errors. Preserve missingness by omitting the missing measurement, not by treating it as zero. Known elapsed time still advances memory. Do not impute target values. Units and station identity come from the raw schema; preserve the catalog attribution and verify actual column names on ingestion.

Use 14-day histories and retrospective windows of 1–168 hours. Labels are observed-event means or adjacent-window contrasts, not estimates of the true time-integrated pollution field. Provide an independent observed-count output for diagnosing sparse intervals. A practical user who needs regulatory exposure measures would require a different missingness/weighting model; no such claim is made here.

**Chronological split:** training from March 2013 through August 2014; development/calibration from September–December 2014; protected test from January 2015 through February 2017. Drop any episode whose history crosses a split boundary. Fit scaling only on training observations. Preserve extreme values; do not clip test outliers to improve a metric. Save raw and normalized targets.

Training episodes may overlap within the training partition. Protected histories should be generated from fixed, nonoverlapping 14-day blocks, with all stations kept together. Group adjacent histories into 28-day temporal clusters for primary resampling; repeat uncertainty estimates with 56-day clusters as a dependence sensitivity analysis. The number of real temporal clusters is finite and modest: extra queries, stations or overlapping windows do not manufacture independent replication. Wide intervals must remain an inconclusive result.

Run two transfer modes: (a) frozen-weight, **shape-matched** synthetic transfer; (b) identical method-specific training/tuning on the real training/development partitions. For mode (a), train a separate synthetic model with 12 one-hot keys and one scalar value using the same selected method/tuning rule; do not load an incompatible 16-key, eight-value checkpoint. Only target-domain training-derived unit normalization is then applied. Call this calibrated frozen-weight transfer, not completely calibration-free zero-shot transfer. Mode (b) tests task-adapted real utility. Fix the mode-specific primary metric and report both without selecting whichever looks better after testing.

Because the real target is scalar, set `d_v=1` for the fixed-writer temporal experiment and use one-hot station keys of dimension 12. For `J=4,8,16`, jet payloads are 768, 1,536 and 3,072 FP32 bytes before timestamp/status metadata. Do not inflate an unused 64-dimensional value vector merely to claim a large state budget. Learned value expansions are allowed only in the independently tuned arm and count fully.

If the archive is temporarily unavailable, complete the synthetic program and label real transfer untested. A replacement dataset requires a recorded amendment, separately verified license and independently chosen protected split. Do not silently substitute another benchmark.

### 6.4 Language/agent benchmarks: conditional, not part of the inexpensive core

LongMemEval supplies temporal reasoning, updates and abstention questions; LongMemEval-V2 is a newer experience-oriented agent benchmark, with histories much larger and broader than this PoC. The latter's context-gathering formulation and file/RAG baselines make clear that neural-state comparisons alone do not establish practical agent value. [R17], [R18]

BABILong stresses distributed-fact reasoning; LongBench v2 spans long-document tasks rather than specifically scale-conditioned history queries. Their use is an escalation after a suitable language bridge exists, not a substitute for the mechanism experiment. [R19], [R20]

Before use, pin dataset revisions and inspect the actual data license separately from repository code licenses and source-corpus licenses. Never train on official test labels, oracle evidence selections or benchmark judge rationales. A local judge is not interchangeable with an official paid-model judge without validation; report exact-answer measures and a blinded human-audited subset, and label any score using a different judge as non-comparable. Synthetic event QA can validate an integration path, but cannot validate general LongMemEval or autonomous-agent performance.

## 7. Baseline implementation and fair comparison rules

### 7.1 Baseline tiers and required comparison scope

**Core mandatory controls:** equal-byte distinct-pole memory, equivalent gamma/Laguerre coordinates, generic readers, exact arithmetic oracle, temporal buckets and exact-event bounded storage.

**Required before a claim of competitiveness beyond the selected pair:** HiPPO/Multiscale HiPPO task adaptation, and a modern delta-rule mixer selected using development data from GDN-2 and its tied KDA/GDN subfamilies. Implement the closest published mechanisms rather than only weak recurrent baselines.

**Supplementary local comparisons:** Time-Local/Gaver timeline adaptation, official associative HiPPO task adaptation, compact GRU, and optional routed memories. Their inclusion and any licensing/precision exclusions are recorded before confirmation. An excluded direct rival limits the claim; it must not be represented as a defeated baseline.

**Nonlocal reference results:** published large-model RetNet, Mamba-3, GDN-2/KDA or full agent systems. [R08], [R10], [R12], [R18] Do not place published accuracies or CUDA speeds beside newly trained laptop numbers as a controlled comparison. This program can compare local mechanisms, not reproduce every large pretrained system.

### 7.2 Distinct-pole exponential memory: the decisive comparator

Use `K=2J` value-only augmented states, with the same key/value shapes and FP32 dtype as `J` jets:

`Z_l' = exp(-lambda_l*Delta) * Z_l + X`.

The state payload is identical. Give its reader all anchors, query fields and support features available from its state. Evaluate fixed log-spaced rates, rates selected by development optimization, and learned bounded log-rates. The winning development configuration represents this family; do not compare only to the initial grid.

Define the training horizon range from training data only. A starting rate range is `[1/H_max, 1/H_min]`; also tune an extension beyond either endpoint because the best basis is not necessarily centered exactly on the requested windows. For learned rates use `lambda=exp(ell_min+(ell_max-ell_min)*sigmoid(theta))`. Independent anchors need not be sorted internally; sort only for optional finite differences/interpolation. Permit coincident rates, but record collapse. The bank may include a no-decay state as a tuned variant; a corresponding mixed jet/value configuration must count all states and is a separate candidate.

Allow paired finite differences with development-selected spacing and direct raw-anchor MLP reads. Test lower/higher state budgets, not only a single equality point. The central test is a state-basis allocation question: two coordinates at one repeated pole versus two distinct poles.

### 7.3 Polynomial-memory baselines

**Single-scale/multiscale bank.** For a polynomial order `n`, form the translated-Legendre generator with zero-based indices:

\[
A_{ij}=-\sqrt{2i+1}\sqrt{2j+1}
\begin{cases}(-1)^{i-j},&i<j\\1,&i\ge j,\end{cases}
\qquad b_i=\sqrt{2i+1}.
\]

For the event-measure task, use impulse updates

`P' = exp(A*Delta/tau) @ P + (b/tau) * X`,

independently for each key/value feature. This is an explicitly labeled event-input adaptation. For reproduction of a sampled continuous-signal experiment, use the published zero-order-hold input term instead: solve the augmented matrix exponential for `B_d`, without confusing the two input conventions. Let the reader access all coefficients and query endpoints; include a reconstruction/integration read as an optional structured decoder. Do not force signed polynomial normalizers to behave like nonnegative exponential denominators.

**HiPPO Zoo log-timescale model.** Allocate `n*m` coefficients per input feature, with `n*m` matched to `2J` or compared on the nearest measured state frontier. For `L=log(1/epsilon)` and the Legendre Jacobi matrix `Jx` on `[-1,1]` (zero diagonal; off-diagonal `r/sqrt((2r-1)(2r+1))`), the inspected script builds

`G = sqrt(epsilon) * expm((L/2)*Jx)`.

Its rank-one injection is `B=b*beta^T`, with

`beta_r = sqrt(epsilon)*sqrt((2r+1)*L)*i_r(L/2)`,

where `i_r` is the modified spherical Bessel function. Alternatively calculate this coefficient by converged CPU quadrature of the normalized Legendre basis; compare it with the script's expression. Dynamics are `dP/dt=(A P G)/tau0`; event impulses add `B*X/tau0`. For the published continuous-input reproduction use its ZOH construction. Query log-horizon by evaluating the scale basis and contracting the scale axis. [R06], [R07]

For efficient local updates, diagonalize the symmetric `G=U diag(gamma) U^T` once, store `P U`, and propagate each column with `expm(A*gamma_j*Delta/tau0)`. Verify this coordinate implementation against the dense Kronecker-system exponential. Precompute only a bounded cache for the declared gap alphabet; arbitrary-gap preparation costs must be counted. Use CPU float64 for matrix construction, then validate an FP32 run. Do not assume MPS supports every matrix-exponential path.

**Code-inspection correction:** the generic `polynomials.py` multiscale helper implements the basic rate-coordinate construction; the log-rate construction is in `scripts/multiscale_hippo.py`. A generic-helper-only baseline would omit the relevant stronger variant. The latter includes `build_G_log_g` and Bessel-based input coefficients. Do not treat the finite truncated-Jacobi matrix exponential as a proof of an exact infinite-dimensional multiplication operator; reproduce the finite published system and separately measure approximation error. [R07]

**Associative HiPPO control.** The inspected `AssocMemHiPPO` implementation uses learned scalar key/query locations in `[0,1]`, evaluated in a shifted Legendre basis, with per-bank minimum-norm delta updates. Its actual code, rather than a contradictory vector-map docstring, determines the reference. Port that update and its causal HiPPO encoder; adapt the read coordinate to the external query, and label the adaptation. Count both polynomial history and associative state. Reproduce the supplied short associative task before interpreting a failure on this project's task. [R07]

### 7.4 GDN-2, KDA and gated-delta comparison

Write an independent native tensor recurrence with state `S[K,V]`:

\[
\bar S=\operatorname{Diag}(\alpha)S,\qquad
S'=\bar S+k[(w\odot v)-\bar S^\top(b\odot k)]^\top.
\]

The external query reads `S^T q`. Preserve normalized key/query vectors, decay before erasure, and separate erase/write gates. Setting `b=w=beta` recovers KDA; additionally tying `alpha` across key channels recovers gated delta. These are exact algebraic regression tests, not approximate substitutes. [R10]

The local full-mixer adaptation includes causal depthwise short convolution (initial kernel size 4), SiLU, learned projections and gates, FP32 recurrent state, output normalization/gating, and a task reader. Count previous convolution inputs in persistent bytes. Start gate logits from the paper's parameterization; permit the published erase-range variant and tied subfamilies in development selection. Pass `Delta` and task-independent event timing to the encoder; distinguish the published per-event recurrence from an explicitly named continuous-gap-decay extension. Both may be tuned, but never claim the extension is a verbatim reproduction.

Do not import a Triton kernel and assume MPS execution. The inspected wrapper dispatches fused/chunked Triton/FLA paths. The local tensor implementation is an accuracy/mechanism comparator with its own measured performance. Optimize equivalent ordinary operations for all methods; if a faithful native path is too slow, run CPU at a shared informative scale or limit the competitiveness claim. Omitting this baseline solely because it is stronger is not permitted. [R11]

### 7.5 Time-Local/Gaver timeline comparison

For fixed reconstruction lags `tau_a` and even `N` function evaluations, retain leaky states at `s_(a,k)=k*log(2)/tau_a`, `k=1..N`. All such states count; deduplicate identical rates only with an exact equivalence test. The Gaver read is a fixed linear combination of these states, followed by a query-conditioned reader. The inspected implementation uses a ZOH-style `exprel` write term and float64-related buffers. These are not the same as an unscaled event impulse. [R04], [R22]

For a separately labeled impulse adaptation, use `F'=exp(-s*Delta)F+X` and the standard Stehfest weights:

\[
\eta_k=\log2\,(-1)^{k+N/2}
\sum_{j=\lceil k/2\rceil}^{\min(k,N/2)}
\frac{j^{N/2+1}(2j)!}{(N/2-j)!(j!)^2(k-j)!(2j-k)!},
\quad\hat f(\tau)=\tau^{-1}\sum_k\eta_kF(k\log2/\tau).
\]

Compute weights in CPU float64 using log-factorials. Test impulse responses and compare FP32/FP64 errors. If accurate FP64 is needed, compare byte budgets accordingly and report the cost; do not force an unstable FP32 baseline while allowing the candidate its preferred precision. A Gaver-derived timeline plus an MLP is an adaptation, not the complete published Transformer.

### 7.6 Strong simple methods and exact references

Implement all storage in packed arrays with known capacity, not Python dictionaries whose overhead is ignored.

**Recent ring:** retain as many exact `(timestamp,key,value)` records as fit. Answer directly from retained records; expose absence rather than inventing old values. Tune retention policy only on development data.

**Uniform reservoir:** retain a fixed sample using a seeded streaming reservoir algorithm; retain the total count and RNG state or counter. Use key/window-filtered sample means; report zero-sample behavior and its uncertainty. Include a recent-ring plus reservoir split selected on development data.

**Mergeable temporal buckets:** each bucket stores `(key, oldest_time, newest_time, count, sum[value])`. Create a singleton for every write. When full, merge the chronologically adjacent same-key pair with minimum combined time span, with deterministic tie-breaking. When no merge is possible, apply the frozen whole-key eviction policy. At read time, allocate a bucket fraction by temporal overlap, treating point buckets exactly, and compute a count-weighted mean. Compare this decoder to a learned reader of the same retained buckets. Count keys, bounds, counts, occupancy flags and merge/eviction metadata. This is a specified engineering baseline, not a claim of a formal streaming approximation guarantee.

**Per-key latest table:** retain the newest exact value/timestamp for each retained key; use bounded LRU eviction if necessary. This is a particularly strong correction/latest-value baseline, not a hard-window oracle.

**Exact history:** store all records, filter by time/key and compute the answer. Report storage and query cost. Its exact arithmetic answer is the label oracle; its measured implementation is also a realistic alternative when raw retention is affordable. A trained full-attention model is an additional learned reference, not a mathematical upper bound.

### 7.7 Budgets, tuning and reproduction safeguards

Persistent state means **all information that depends on a history and survives between events/queries**, including key tables, time metadata, normalization, caches, conv buffers, RNG counters and optional exact stores. Report logical tensor/packed-array bytes and physical resident/driver memory separately. Static model parameters/buffers have a separate byte ledger and enter total end-to-end memory. No external corpus, encoder cache or per-history Python object may be hidden outside the declared budget.

Use three complementary comparisons:

1. **Exact payload match:** `J` jets versus `2J` augmented exponential states, identical writers and dtypes, with timestamp/status metadata also matched.
2. **Matched resource envelope:** each family is tuned under the same maximum persistent bytes and active trainable parameter budget; report exact used bytes instead of padding. Compare latency as an outcome, not an assumed match.
3. **Pareto/frontier evaluation:** allow each family reasonable state/parameter/precision configurations and report quality versus actual total cost. A strict parameter match alone is not a deployment conclusion.

Initial optimizer for learned components: AdamW, FP32, learning-rate candidates `1e-4,3e-4,1e-3,3e-3`, weight decay `0` or `1e-2`; width candidates `64,128,256`. These initialize a shared search process rather than permanently cap tuning. Include a stable learning-rate schedule, gradient-norm logging and development-selected clipping. Search rates/basis order/state allocation as method-specific parameters. Continue tuning a family while development learning curves show unresolved optimization or worthwhile expected information; record all trials, compute and validation queries. Compare both equal-tuning-cost results and the best locally credible validated configuration, so an inefficient search does not manufacture a winner.

For fixed writers, train the numeric reader on many independent generated episodes; use fresh training histories to avoid memorization. For learned writers, recompute states per update. Use identical training-history draws and evaluation partitions across methods. Different methods may need different learning rates or convergence lengths. A common fixed epoch count is not evidence that all baselines converged.

A baseline is accepted only after algebra/unit tests, a task-relevant positive control, and an adequate development learning curve. Tiny-set overfitting is useful for testing the training path but is not a claim about generalization. Preserve failed runs and distinguish optimizer failure from representational failure.

## 8. Metrics and confirmatory statistical protocol

### 8.1 Claims, estimands and guardrails

**Primary H1 estimand:** expected reduction in normalized squared error for a fresh history from the frozen synthetic temporal-basis population, averaged equally over mean and contrast queries and the four regimes. The primary comparison is the selected jet configuration versus the strongest development-selected equal-byte distinct-pole configuration. The collision-free population supports a temporal-basis claim; the dense-key suite separately supports or rejects its associative extension.

For task `c`, estimate a target scale `s_c` from training targets only and freeze it. Use

`loss(h,q,c) = mean_over_output_coordinates((prediction-target)^2) / s_c^2`.

If a coordinate is constant on training, give it a separate absolute-error report rather than divide by an arbitrarily tiny variance. Aggregate queries within task/history, then tasks within history, then histories/regimes. Report unnormalized RMSE/MAE too; real PM2.5 results retain their native units. The primary contrast is `Delta_L = L_baseline - L_jet`, so positive is favorable.

Secondary outcomes are latest/categorical accuracy, absent-key false-positive rate, abstention coverage–risk curves, calibration error for any trained confidence head, loss by horizon/load/regime, training sample efficiency, and catastrophic/nonfinite output frequency. A support failure must not silently remove a hard primary example. Valid primary labels always receive a prediction/loss; record abstention separately. Before confirmation, define a common failure fallback: an unsupported or nonfinite prediction returns the training-only unconditional task mean, with a failure flag and its resulting full loss. If state becomes nonfinite, quarantine that episode state, use the same fallback for remaining queries and preserve the triggering inputs. Never drop failed examples or silently rerun only unfavorable seeds. Report the numerical-failure rate separately; a claimed stable method must pass its declared stability guard, not hide failures behind average loss.

**H4 metrics:** persistent bytes/history, static model bytes, peak physical memory observations, ingestion events/second, ingestion total time, query p50/p95 latency, query throughput under specified batching, serialized snapshot size, preprocessing/indexing time, training time and operation/FLOP estimates. Use events rather than tokens until a tokenizer exists. Report measured wall time separately from analytic FLOPs; hardware vendor FLOPs do not substitute for timing.

A statistically positive but negligible improvement is not a breakthrough. No universal practical threshold is imposed. Present the cost–quality frontier and the uncertainty on both. Where an application error tolerance or cost constraint is genuinely specified before testing, evaluate it directly. In this project it is currently unspecified, so a claim of broad economic or user value would be premature.

### 8.2 Protected confirmation and selection

Create separate development, pilot and protected confirmation manifests before extensive tuning. Pilot histories are never reused as protected histories. The agent may inspect all exploratory results; a confirmation runner should not expose per-model protected scores until every predeclared run in a comparison is complete or a documented failure is resolved.

Before opening protected outputs, freeze:

- generator/data revisions and evaluation population; exclusions; task weights;
- the primary method pair, selected state budget and configuration; secondary budgets/families;
- training data and stopping schedules, tuning selection rule, train/test seed lists;
- metric code, uncertainty method, sample/replicate sizes and multiple-testing procedure;
- numerical tolerances, support behavior, timing boundaries, and nonfinite/failure handling.

The principal pair receives one primary test at one development-selected budget. Additional budgets, dense-key claims and all-family superiority contrasts form a secondary family controlled with Holm's procedure at familywise 0.05. Alternatively, a single predeclared aggregate across budgets can be primary, but that choice must precede protected results. Do not choose the favorable budget after testing and call it confirmation.

For competitiveness, compare against **each** required family and show its uncertainty, not only against an average of weak baselines. A win against the distinct-pole bank does not imply a win against HiPPO, GDN-2 or simple storage. Test-selection over the protected set invalidates the claim and requires fresh confirmation.

### 8.3 Independent units and uncertainty

Synthetic histories are independent sampling units. Queries within a history are dependent. Training seeds represent algorithmic variation, not additional independent histories. Pair methods using common history draws, data-order seeds and evaluation queries; model initializations still differ according to their architectures.

Store the paired history-level difference `d[h,s]` for history `h` and training replicate `s`. The target averages over both new histories and the specified training procedure. Use crossed resampling: independently resample histories and paired training replicates, calculate the mean difference on the sampled Cartesian product, and form an interval. Stratify history sampling by the frozen regime/task population weights. Increase bootstrap draws until the reported interval endpoints' Monte Carlo uncertainty is negligible relative to the interval width; save the resampling seed. Also report each training replicate's effect and the pooled raw distribution.

Use the real-data temporal clusters from Section 6.3 instead of individual questions or stations. Report 28-day and 56-day clustering results and the finite number of available clusters. With too few training replicates or temporal clusters, bootstrap tails can be unreliable: report this limitation, show raw paired effects and avoid strong generalization claims. Do not use query-count binomial intervals for clustered accuracy.

### 8.4 Pilot-based size and resource allocation

Start pilot measurement with enough independent histories to estimate the paired-loss distribution and multiple independently trained runs to expose seed variability. A practical initialization is 128 pilot histories and four training replicates on the cheap setting; these are starting measurements, not a mandatory final sample size or a guarantee of adequate variance estimates. Expand only the pilot when nuisance-variance estimates remain unstable. Do not use a favorable effect estimate as the sole basis for stopping.

A useful planning model is

`d[h,s] = mu + history_effect[h] + seed_effect[s] + residual[h,s]`,

with approximate mean variance

`V(H,S) = sigma_history^2/H + sigma_seed^2/S + sigma_residual^2/(H*S)`.

Fit or estimate these components from pilot data, retaining conservative uncertainty bounds. Use them to allocate additional histories versus training replicates. For a simple planning approximation, the two-sided 0.05 / 80%-power detectable effect is approximately `2.80*sqrt(V)`; this is a sensitivity calculation, not the confirmatory test or a mandatory success threshold. Verify planned coverage/power using simulated resampling with the observed dependence and skewness.

Publish a curve of detectable effects and interval widths versus measured local compute. Select a fixed `(H,S)` and report what it can resolve. If real-data sampling cannot support small-effect detection, say so; generating more overlapping windows does not fix it. If a scientifically useful precision is locally unattainable, finish the correctly scoped result as inconclusive rather than promise a later win. Any sequential alternative must specify an actual valid confidence-sequence/alpha-spending design before use; the default here is fixed-sample confirmation.

### 8.5 Harness validation and independent replication

Run an identical-method A/A control through the entire pipeline. Numerical equivalence should give essentially zero paired differences; independent retrainings of the same family should not systematically favor one label. Test a known-better fixture, such as exact history versus a ring that provably lacks a requested event, to demonstrate sensitivity. Inject reversed temporal labels, an intentionally wrong moment sign and a query-leaking writer as red-team controls; the harness must detect the former failures and flag the latter forbidden information path.

After a positive protected result, replicate the **frozen** primary comparison on new synthetic histories and fresh training replicates; do not retune it. For real data, a second task slice or held-out station analysis is robustness, not a magically independent new city/time period. A later new dataset can supply stronger replication with a new preregistration.

## 9. M3 Max feasibility, calibration and performance measurement

### 9.1 Discover the actual machine first

The following are existing shell commands, not project commands. Run read-only discovery and save outputs; missing optional utilities are not failures of the scientific mechanism.

```bash
uname -m
sw_vers
sysctl -n hw.memsize
sysctl -n hw.physicalcpu
system_profiler SPHardwareDataType SPDisplaysDataType
vm_stat
sysctl vm.swapusage
df -h .
python3 --version
```

Record actual chip, architecture, unified-memory bytes, macOS build, power mode, available disk and free-memory conditions. Do not infer the installed memory from the M3 Max's maximum supported configuration. Avoid concurrently running heavy applications, keep power conditions consistent, and record thermal/power changes rather than silently discarding slow runs.

### 9.2 Software and dependency strategy

Primary stack: a supported native-arm64 CPython in a project virtual environment; NumPy/SciPy for CPU references and precomputed small matrices; PyTorch for CPU/MPS tensors and autograd; pytest; a YAML/JSON parser; optional pandas for data preparation and matplotlib for reports. The core has no CUDA, Triton, JAX-Metal or MLX dependency. This choice minimizes differing autograd/precision paths while allowing reference comparisons.

The inspected PyTorch 2.14 MPS documentation specifies macOS 14 or later and documents availability checks and MPS memory/timing interfaces. This is an inspected documentation version, **not a claim that an identical wheel has been installed or tested here**. Resolve an available compatible stable wheel on the actual machine, consult that installed version's docs, run the compatibility matrix below, then lock exact package versions and wheel hashes. Do not upgrade mid-confirmation. If the OS cannot support the selected MPS path, use a supported CPU environment or obtain permission for an OS change; never perform an unapproved system upgrade. [R24]

Typical existing bootstrap commands, after selecting a compatible local Python, are `python3 -m venv .venv`, activation, and `python -m pip install ...`. The coding agent must produce a tested `requirements.lock`/equivalent with hashes, rather than copying unverified version numbers into a claimed working environment. Record `pip freeze`, Python build and `torch.__config__.show()`.

For a native-MPS performance run, set `PYTORCH_ENABLE_MPS_FALLBACK=0` before importing torch. Keep fast-math off in the numerical reference configuration. Official documentation warns that disabling the allocator high-watermark limit can cause system failure; **do not set it to zero to force a run through memory pressure**. Record environment settings. A separate CPU reference is preferable to silent per-operation fallback. [R25]

### 9.3 Required local compatibility matrix

Before training, run CPU float64 and CPU/MPS float32 tests on actual planned shapes for: `exp`, multiplication/addition, reductions, `matmul`/`einsum`, `softplus`/`sigmoid`/SiLU, normalization, gather/indexing, masking, MLP forward/backward, differentiable rates, gradient checkpointing, serialization and resumed RNG. Test short convolution for delta baselines. Precompute polynomial matrices in CPU float64; separately test any training path through their parameters.

Use `torch.backends.mps.is_built()` and `is_available()`. Check `torch.mps.synchronize`, `current_allocated_memory`, `driver_allocated_memory` and `recommended_max_memory` on the installed version. [R24] Each operation/gradient needs a reference tolerance justified from its scale and conditioning. Start with relative error `1e-5` plus absolute error `1e-6` for ordinary bounded FP32 fixtures, and inspect violations rather than widening tolerances automatically. Long ill-conditioned cases need an error-vs-length envelope, not a single permissive threshold.

CPU float64 is the oracle. FP32 is the initial recurrent-storage and training precision. FP16/BF16 or quantized states are optional separate experiments only after error and task-equivalence checks. Model-weight quantization does not justify quantizing small accumulated memory states. A successful NumPy/Linux check is not evidence of MPS support.

### 9.4 Memory model and estimates

For `H` heads and `J` anchors, FP32 jet-state payload per history is

\[
M_{\rm state}=4\cdot 2HJd_k(d_v+1)\ \text{bytes},
\]

plus time/status and any other history-dependent metadata. The equal-byte exponential bank has `2J` anchors and the same payload.

| Example, estimates only | Payload |
|---|---:|
| `H=1,J=4,d_k=16,d_v=8` | 4,608 bytes = 4.5 KiB |
| `H=1,J=8,d_k=32,d_v=64` | 133,120 bytes = 130 KiB |
| Same second shape in FP16, not recommended initially | 66,560 bytes = 65 KiB |
| Batch 16, four heads, second shape in FP32 | 8.125 MiB persistent state |
| Naively saving those states at 2,048 steps | 16.25 GiB **for states alone** |

The original 64 KiB FP16 estimate omitted normalization; the complete corresponding tensor payload is 65 KiB. Runtime activations are a much greater risk than a single inference state.

For FP32 Adam-style training with `P` trainable parameters, budget approximately `16P` bytes for weights, gradients and two optimizer moments, plus implementation temporaries. Add encoder/reader activations, checkpoint buffers, polynomial transition workspaces, data batches, validation snapshots, allocator overhead and host arrays. Frozen weights need no optimizer moments but still occupy memory; frozen decoders with trainable input prefixes still require backward activations through the decoder.

A terminal-contraction example with `B=16,J=8,T=4096,d_k=32` needs 64 MiB for an FP32 `[B,J,T,d_k]` weight–key buffer; another such moment buffer would double that portion. Chunking reduces this buffer without changing the sum, subject to summation-order numerical checks. Do not form all outer-product states across time.

The core source data and small model checkpoints should be modest compared with LLM weights, but exact storage is discovered from asset manifests and generated artifacts. Before every substantial download/run, require free disk for the asset, extraction, temporary copy, checkpoints and logs. Stream synthetic examples instead of writing an unbounded corpus.

### 9.5 Hardware-adaptive configuration policy

Begin at batch 1, one head, the 4.5 KiB state shape, length 128, FP32. This is a smoke fixture, not the scientific experiment. Measure actual peak allocations and process memory after warm-up. Increase batch/chunk size geometrically until utilization is useful or the safe memory envelope is approached; then select the largest stable setting below it.

Define the working envelope from the **minimum** of available system memory after an observed OS reserve and the driver's recommended working set. Initially leave a conservative reserve (for example one quarter of that envelope); adjust only after repeated observed peaks and swap-pressure checks, not in pursuit of a favorable result. Do not sum resident memory and Metal driver memory as if they were independent physical pools on unified memory; record both and use OS pressure as the system-level check.

If a configuration exceeds memory: first reduce batch size with accumulation preserving the effective batch; then reduce exact contraction chunks; then use verified recomputation/checkpointing. Only after those changes consider a smaller scientific shape or history length, which is a methodological amendment if it changes the selected regime. Never shorten a history for one comparator only. A slow-but-correct CPU path remains a faithful fallback for small mechanisms; its timing is reported as CPU, not MPS.

All runtime forecasts derive from measured pilot throughput. For `U` optimizer steps, estimate `U * measured_seconds_per_step`, adding validation/checkpoint cost; fit cost versus length rather than assuming linearity from one tiny shape. Log forecast intervals and actuals. No wall-clock duration for this machine is claimed in this plan.

### 9.6 Timing boundaries and reliability

Measure separately: preprocessing; writer/encoder ingestion; state-only query; reader; optional language decode; serialization; and total end-to-end work. Report cold-start/first-call separately from steady state. Include both single-history/single-query latency and explicitly declared batched throughput. A batch-size speedup is not a single-query latency improvement.

Use evaluation mode, disable dropout and gradients for inference timing, and keep training timing separate. Warm each shape until compilation/caching effects settle, and record the warm-up policy. Synchronize MPS immediately before and after timed device work using `torch.mps.synchronize()`. Use a monotonic high-resolution host timer. For sub-millisecond kernels, time a declared block of operations between synchronizations, then measure actual interactive latency separately. Do not call `.item()` or transfer tensors to the CPU inside an allegedly asynchronous hot loop unless that cost belongs to the implementation.

Randomize method order within paired timing blocks. Record at least multiple separated blocks; use pilot variation to choose a fixed repetition count and report block-level uncertainty, not millions of kernel repetitions as independent machines. Do not silently remove thermal throttling; report sustained behavior and a standardized rested comparison when both are relevant. Report p50/p95 over real query mixtures, not only a favorable shape.

Current allocated tensor memory, driver allocation, resident memory and swap observations measure different things. Sample/reset between runs, save peak observations and explain measurement limitations. Avoid uncontrolled overlapping runs. Checkpoint atomically at safe optimizer/episode boundaries and keep a known-good checkpoint before each larger configuration.

## 10. Optional frozen-language integration, with a bounded claim

### 10.1 Entry condition and assets

Enter only after the numerical/harness gates pass and either (a) H1/H4 are supported in a relevant regime, or (b) oracle-evidence diagnostics specifically indicate a missing language component that could change interpretation. Do not download or train a large decoder before this gate.

Use the official **Qwen/Qwen3-0.6B** as the initial frozen decoder; **Qwen/Qwen3-1.7B** is a second-size escalation when measured memory and the oracle-evidence test justify it. The inspected model cards label both Apache 2.0. Record repository revisions, tokenizer/config/weight hashes and license files; use safetensors and avoid unnecessary remote-code execution. These are chosen reusable assets, not a claim they are the strongest models available. [R21]

Advertised parameter-count weight estimates alone are roughly 2.4/6.8 GB in FP32 or 1.2/3.4 GB in FP16 for the two sizes. Actual loaded parameter count, buffers, activations and caches determine feasibility. These are estimates, not measured footprints. Do not rely on a CUDA-only quantizer or automatic silent CPU/off-disk offload. Test a native eager-attention path on the actual installed Transformers/PyTorch versions before any training.

### 10.2 Concrete adapter path

First create templated event-language episodes with independently generated keys, revisions, temporal windows and canonical finite answers. Use held-out paraphrase templates and held-out event IDs. Keep oracle numeric/categorical labels from the event generator; no paid teacher is required. This is an integration fixture, not a claim of natural-dialogue memory.

Encode each event independently or with a small common causal context using the frozen decoder's hidden features. The eventual query is absent. Under `no_grad`, frozen history features can be cached for training with their model/tokenizer hashes; the cache is training expenditure and cannot remain an uncounted inference-history store. Train the event projections, memory reader and prefix projector. Recompute states when these projections change.

Map the query-conditioned memory read to `P` virtual prefix embeddings of decoder hidden width using a small MLP (`P=4` as a smoke starting point, tuned within the shared budget). Concatenate prefix embeddings with embeddings of the query and teacher-forced answer. Mask prefix/query positions from the token loss. Freeze decoder weights with `requires_grad=False` but **do not use `no_grad` on the answer-decoding forward pass**, because the prefix needs gradients through the frozen decoder. During testing, supply no answer tokens; generate or score the canonical answer candidates.

Implement three reference modes: no-memory query-only prefix with equal trainable budget; exact/oracle evidence through the same prefix/decoder bridge; and the same bridge driven by each competing compressed memory. Also include ordinary text evidence from time-filtered exact storage/BM25 on the same event text. Count corpus, index, retained text and decoder tokens. A semantic vector index is an optional development-selected retrieval baseline, not a hidden external memory available only to one method.

### 10.3 Acceptance and guardrails

Before training memory comparisons, verify that an oracle-evidence bridge can answer the fixture and that prefix gradients are nonzero while decoder weights remain unchanged. Check first-answer logits and label masking manually on a tiny example. A broken bridge makes the language experiment inconclusive, not a failure of the temporal state.

Turning the adapter off must **bypass** prefix injection entirely; multiplying virtual tokens by zero does not restore the original decoder because extra tokens still alter attention/positions. Verify identical adapter-off logits to the frozen backbone. Measure active-adapter regressions on held-out query-only examples and a license-cleared ordinary text set; do not claim “no language-model regression” solely because weights were frozen.

Only after this bridge works should a protected, suitably licensed real language benchmark subset be registered. Preserve original categories, timestamps and official question semantics. Report local exact-answer/human-audited metrics and language-generation cost. Success on a subset, a smaller backbone or an altered judge must be labeled accordingly. Failure to transfer from numeric streams may require narrowing the contribution, not automatically increasing model size.

## 11. Risk-ordered work packages and decision gates

All commands under `python -m famjet...` below are **proposed project commands that the coding agent must implement**. They do not currently exist in this handoff. Existing external commands were identified separately in Section 9. Do not mark a work package complete because an output file was created; its acceptance tests must pass.

### WP00 — Prerequisites, provenance and capacity discovery

**Objective/risk:** discover actual local feasibility and licensing before dependent work.  
**Dependencies/entry:** this plan and a target laptop; no pretrained-model download.  
**Actions:** run Section 9.1 discovery; create a native virtual environment; resolve and freeze dependencies; execute the critical-op matrix; verify UCI access/license; record reference revisions and any licensing restrictions. Run timing probes for native CPU and MPS ordinary operations.  
**Files/command to implement:** `famjet/doctor.py`, `configs/smoke.yaml`; `python -m famjet.doctor --out artifacts/environment`.  
**Outputs:** `hardware.json`, `dependency_lock`, `op_support.json`, `assets.json`, license notices and smoke timing data, all with timestamps/hashes.  
**Resources:** batch 1 and tiny tensors; no LLM weights.  
**Accept/branch:** proceed on a correct CPU path even if MPS fails, with a CPU scope/cost record. Block affected accelerated claims if an op is unsupported. Reduce framework complexity, not mathematical fidelity. If licensing blocks a reference checkout, use an appropriately attributed independent equation implementation where permissible or explicitly narrow the claim; do not silently reuse restricted code.

### WP01 — Algebra and numerical invariants

**Objective/risk:** H0, gamma equivalence, normalization, unit handling, interpolation failure.  
**Dependencies/entry:** working CPU numerical environment from WP00.  
**Actions:** implement direct sums, streaming jets, equivalent coordinates and log-rate read derivative; run Appendix A and add edge cases from Section 4.7. Add autograd checks for rates/writes, direct-versus-recurrent gradients and terminal-versus-streaming states. Repeat bounded fixtures in FP32 on each intended device. Add a finite-bits/linear-kernel indistinguishability diagnostic, described in Section 12.  
**Files/command:** `memory/jet.py`, `memory/oracle.py`, `tests/test_algebra.py`, `tests/test_time.py`, `tests/test_gradients.py`; `python -m pytest tests/test_algebra.py tests/test_time.py tests/test_gradients.py`.  
**Outputs:** numerical error envelopes, golden small fixtures, failing inputs and precision/device comparisons.  
**Resources:** small CPU float64 reference; no model training.  
**Accept/branch:** ordinary finite-range tests must meet predeclared scale-aware tolerances. Fix bugs before proceeding. A repeatable precision failure requires a numerical-design change and a shared baseline precision comparison. Interpolation failure disables interpolation, not the anchor-based core.

### WP02 — Data and evaluation harness

**Objective/risk:** exact labels, query independence, leakage, missingness and unit correctness.  
**Dependencies/entry:** WP01 arithmetic oracle.  
**Actions:** implement deterministic generators, raw UCI preparation, chronological split manifests, exact means/contrasts and absent support. Verify a small hand-computed suite with zero gaps, boundaries, missing data and revisions. Enforce read-only state queries and inaccessible raw history for bounded-state runs. Add A/A, intentionally broken-update and known-better oracle controls.  
**Files/command:** `data/synthetic.py`, `data/beijing.py`, `data/schema.py`, `eval/oracle.py`, `eval/splits.py`, `tests/test_harness.py`; `python -m famjet.prepare --config configs/data.yaml`.  
**Outputs:** checksummed manifests, schema validation, target distributions, exclusion counts and immutable split roots.  
**Resources:** stream synthetic data; download/extract only the small real dataset after disk/license checks.  
**Accept/branch:** every label and time-boundary test passes; paired A/A comparisons produce no systematic difference. Correct a broken harness before training. If the dataset is unavailable, preserve a synthetic-only branch explicitly; real transfer remains untested.

### WP03 — Simple and equal-information baselines

**Objective/risk:** establish whether useful headroom exists and whether extra states explain the proposal.  
**Dependencies/entry:** WP02.  
**Actions:** implement `2J` exponential states, finite-difference/read variants, equivalent-coordinate reader, packed ring/reservoir/buckets/latest table, and exact-history reference. Implement full byte accounting and query-only model controls. Train a simple reader on fixed-writer states; select informative lengths from development headroom, not from protected results.  
**Files/command:** `memory/expbank.py`, `memory/buckets.py`, `memory/exact_store.py`, `readers/mlp.py`, `eval/budgets.py`; `python -m famjet.run --config configs/development_simple.yaml`.  
**Outputs:** method correctness records, headroom curves, serialized-state audits and initial convergence traces.  
**Resources:** state and terminal-contraction memory from Section 9; locally calibrated batches.  
**Accept/branch:** all baseline positive controls pass. If exact bounded storage already answers the target at the selected budget, choose a harder *development* load justified by the use case or conclude there is little headroom there; do not hide the simple winner. If a generic reader cannot learn an easy sufficient-statistic fixture, repair the training setup.

### WP04 — Minimum complete FAM-Jet

**Objective/risk:** H1/H2 with the smallest full architecture.  
**Dependencies/entry:** WP01–WP03.  
**Actions:** connect state-independent writer, dimensionless jet state, structured and generic readers; implement exact terminal training, streaming inference and state serialization. Keep interpolation/probing off. Verify gradients and outputs across streaming/terminal paths. Train the same task using the main bank comparator and common data draws.  
**Files/command:** `models/task_model.py`, `readers/structured.py`, `train.py`, `state_io.py`; `python -m famjet.run --config configs/development_jet.yaml`.  
**Outputs:** complete candidate checkpoint, predictions, state manifests, training curves, stream/terminal equivalence report.  
**Resources:** small trainable model; no frozen decoder.  
**Accept/branch:** a gain against a weak reader is not a pass; generic and tuned-bank controls must be functional. A working mechanism without advantage remains a valid result. Escalate state size only when headroom and kernel/load diagnostics explain why it could change the answer.

### WP05 — Closest and contemporary mechanisms

**Objective/risk:** prevent an obsolete-baseline or known-equivalent result from being called a contribution.  
**Dependencies/entry:** WP03 harness; WP00 licensing/op checks.  
**Actions:** implement polynomial/Multiscale HiPPO and native delta-rule mixers from Section 7; verify published-system reductions and independent CPU reference agreement. Reproduce an impulse/reconstruction fixture and the associative-HiPPO short-task behavior where that comparator is used. Inspect and record differences from original input/query semantics. Add the Time-Local/Gaver comparison subject to its licensing/precision gate.  
**Files/command:** `memory/hippo.py`, `memory/delta.py`, `memory/gaver.py`, `tests/test_baselines.py`, `docs/baseline_adaptations.md`; `python -m famjet.reproduce --config configs/baseline_reproduction.yaml`.  
**Outputs:** verified equations, reference revision hashes, coordinate/gate-reduction tests, adaptation ledger, development training traces and per-family resource frontier.  
**Resources:** small CPU matrix construction; bounded native tensor training; no official billion-parameter pretraining.  
**Accept/branch:** if a comparator cannot be faithfully represented at the chosen local scale, either choose a common informative scale before freezing or exclude it with a narrower claim. Do not treat an unsupported fused kernel as an accuracy defeat. Do not proceed to a broad superiority claim while required baseline learning curves remain visibly underfit.

### WP06 — Pilot, attribution and sensitivity calibration

**Objective/risk:** H1–H4 identifiability, optimization, statistical power and actual resource use.  
**Dependencies/entry:** WP04/WP05 accepted implementations.  
**Actions:** run pilot histories and training replicates; tune all required families; vary state size, length and key load independently; estimate variance components, compute/sample tradeoffs and sustained timings. Compare structured/generic readers, equivalent bases and shared versus learned writers. Diagnose whether failures are information, optimization, interpolation or precision failures.  
**Files/command:** `eval/power.py`, `eval/timing.py`, `analysis/pilot.py`, `configs/pilot.yaml`; `python -m famjet.pilot --config configs/pilot.yaml`.  
**Outputs:** pilot report, state/parameter/latency frontier, convergence evidence, MDE/precision curves, recommended frozen sample sizes and cost forecasts.  
**Resources:** selected by measured throughput/peaks, not by an assumed M3 Max performance number.  
**Accept/branch:** proceed to confirmation when comparison fairness and sensitivity are adequate, regardless of whether the exploratory sign is favorable. If uncertainty cannot be resolved locally, freeze a narrower claim and report its limits. If only a single favorable slice works, make it an explicitly narrower hypothesis and use fresh confirmation, not a hidden filter.

### WP07 — Preregister and execute protected comparisons

**Objective/risk:** distinguish a reproducible effect from selection and measurement artifacts.  
**Dependencies/entry:** accepted WP06 design and feasible local schedule.  
**Actions:** write/harden the confirmation manifest; lock code, dependencies, configurations, data/seeds, stopping rules and metrics. Execute all paired models. Keep failed runs. Complete the fixed sample/replicate plan; unblind aggregated protected results once the declared comparison is complete. Apply Section 8 analysis and correction rules.  
**Files/command:** `preregistration.md`, `manifests/confirm.json`, `eval/confirm.py`; `python -m famjet.confirm --manifest manifests/confirm.json`.  
**Outputs:** immutable run ledger, raw query predictions, history/seed contrasts, intervals, corrected secondary analyses, timing blocks and resource measurements.  
**Resources:** forecast from WP06, with recovery checkpoints.  
**Accept/branch:** positive, null, negative and inconclusive are all valid completions. A methodological amendment requires a fresh protected set for the changed claim. Infrastructure failures can be repaired/restarted with the same predeclared inputs only when the repair does not select favorable results.

### WP08 — Real transfer, boundary conditions and practical tradeoffs

**Objective/risk:** H3/H5 and useful rather than contrived temporal-memory benefit.  
**Dependencies/entry:** core implementations and frozen real-data protocol; protected real test unused.  
**Actions:** run the two transfer modes, simple time-aware alternatives and all required locally credible competitors; test missingness, unequal key frequencies, continuous gaps, abrupt regime shifts and memory saturation. Report native-unit error and ingestion/query/storage tradeoffs. Evaluate optional shared recent-window hybrids only as separate, byte-accounted arms.  
**Files/command:** `configs/real_transfer.yaml`, `analysis/robustness.py`; `python -m famjet.run --config configs/real_transfer.yaml`.  
**Outputs:** real-data paired predictions, block-level uncertainty, per-regime failures, kernel/reader diagnostics and total-cost curves.  
**Resources:** moderate event histories; use terminal training where eligible.  
**Accept/branch:** synthetic-only success narrows the result. A simpler summary matching performance at lower cost blocks a claim of practical advantage. A low-powered real comparison remains inconclusive rather than confirming or refuting transfer. Preserve failures under rare events and missingness.

### WP09 — Conditional probes, interpolation and language bridge

**Objective/risk:** H6/H7, only when earlier evidence justifies the added mechanism.  
**Dependencies/entry:** the relevant gates in Sections 4.8 and 10.1.  
**Actions:** for probing, match a full-information one-shot reader; for interpolation, run exact-grid error/positivity/fallback diagnostics; for language, pass oracle-evidence, gradient and adapter-off tests before memory comparisons. Register each changed primary claim on fresh data.  
**Files/command:** `readers/probe.py`, `readers/interpolate.py`, `language/prefix_adapter.py`, `configs/extension.yaml`; `python -m famjet.extension --config configs/extension.yaml`.  
**Outputs:** extension-specific controls, costs and transfer report, or an explicit “not entered” decision with evidence.  
**Resources:** measured locally; only this package may require public frozen model weights.  
**Accept/branch:** drop an extension whose advantage is explained by extra information, compute or an unstable numerical shortcut. A language bridge failure is not retroactive evidence against correct numeric memory. No requirement to perform this package to claim a completed core negative result.

### WP10 — Independent replication and final synthesis

**Objective/risk:** reproducibility, selective reporting and overclaiming.  
**Dependencies/entry:** at least one completed protected comparison or a documented blocking limitation.  
**Actions:** replicate supported claims on fresh histories/training replicates; test interruption/resume equivalence; rerun report generation from raw artifacts in a clean locked environment. Perform scientific, validity and feasibility audits. Produce a decision record covering unfavorable as well as favorable configurations.  
**Files/command:** `analysis/report.py`, `reports/final.md`, `DECISIONS.md`, `AUDIT.md`; `python -m famjet.report --runs runs --out reports/final.md`.  
**Outputs:** complete reproducibility manifest, effect/cost tables, plots, supported claim wording, limitations, continuation or stop decision.  
**Resources:** no blind scaling; a replication allocation comes from the selected claim's uncertainty and measured cost.  
**Accept/branch:** another agent can reconstruct the result without chat history. Missing data, unsupported hardware claims, unlogged tuning or unresolvable bias block a strong conclusion, not the honest release of a limited diagnostic report.

## 12. Attribution, adversarial cases and escalation logic

### 12.1 Required attribution matrix

| Change/control | What stays matched | What it can establish |
|---|---|---|
| `J` jets versus `2J` ordinary anchors | Writer, FP32 state payload, query availability; tuned reader budgets | Repeated-pole versus distinct-pole allocation. |
| Jet coordinates versus equivalent gamma/Laguerre coordinates | Exact stored information and transformed reader | Numerical/optimization differences only. |
| Structured versus generic contracted/full-state reader | Same state, query and reader resource envelope | Contribution of explicit derivative/age features. |
| Fixed versus learned anchors for both families | Data and tuning process | Whether a poor grid explained the result. |
| Shared fixed versus independent learned writers | Same raw information and total budgets | Whether a temporal-basis result survives representation learning. |
| One-shot versus two/three probes | Same available state, total depth/FLOPs; measured contractions | Conditional-computation or optimization value, not new information. |
| Interpolation off/on | Same anchors; cost/error/fallback accounting | Continuous-query engineering tradeoff. |
| Exact keyed reads versus overlapping features | Controlled load and temporal distributions | Whether interference erases the advantage. |
| Recent window added to every method | Same additional bytes and source events | Value of a hybrid, not a jets-only benefit. |
| FP64 CPU, FP32 CPU/MPS, optional lower precision | Byte frontier and reference tolerances | Numerical artifact versus a stable task effect. |

### 12.2 A useful no-uniform-dominance deduction

For fixed keys, gates, ages and anchors, both memories are linear maps from event values to finite vectors. Let `F_J` contain rows `exp(-lambda_j*age)` and `lambda_j*age*exp(-lambda_j*age)`, and `F_E` contain the distinct exponential kernels. On sufficiently many generic distinct ages these equal-dimensional row spaces are not generally nested.

Therefore one can choose a nonzero value vector `v` in `null(F_J)` but not in `null(F_E)`, and another with the reverse property. Histories using `+v` and `-v`, with identical timestamps/keys/gates, then have identical one memory but distinguishable other memory; normalization columns are identical because values do not affect them. Rescale values to a bounded range if desired. This is a direct linear-algebra construction, not a universal theorem about trained nonlinear writers.

Implement an SVD-based fixture with residual/rank diagnostics to show both kinds of indistinguishability. Select ages/rates using development-independent deterministic fixtures and avoid ill-conditioned numerical rank claims. This establishes why an improvement must depend on a task distribution or resource/numerical preference: neither finite kernel bank uniformly retains all possible historical value information. Finite-precision state also cannot losslessly encode arbitrary unbounded discrete histories.

### 12.3 Domain-specific adversarial tests

Include long empty gaps with no new information; many same-time writes; a rare key queried long after its last observation; unequal observation rates; repeated contradictory updates; stale high-magnitude outliers; keys with identical/near-identical features; global timestamp shifts; changes of time units; heavy-tailed gaps; windows narrower than anchor resolution; far-old sharp intervals; long stationary stretches followed by a sudden reversal; and very small normalization mass with large signed values.

Test monotonically growing history length at fixed number of relevant facts, and growing fact load at fixed elapsed horizon. Recent controlled memory-interference work makes that separation particularly important, but its findings concern tested recurrent architectures and regimes rather than all possible compressors. [R26]

Test mass scaling: duplicate every observation or multiply all write gates by a constant and inspect the expected normalized-read invariance and epsilon bias. Test opposite-signed values whose numerators cancel while support remains large. Test stream resumption after a query at a later time so the query cannot accidentally age the persisted state. Test a learned anchor update followed by a stale cached state; the system must reject that mismatch.

### 12.4 Evidence-triggered alternatives, not an unranked menu

The primary direction remains exact first moments. Only branch when evidence identifies a bottleneck:

| Evidence | Next justified action | Claim to avoid |
|---|---|---|
| More ordinary anchors match/beat jets across the fair frontier | Use the simpler bank; publish allocation/conditioning results if informative. | Calling the same bank a new function-valued architecture. |
| Gamma-coordinate control matches exactly | Retain the numerically better coordinate implementation. | Claiming jets add representational power. |
| HiPPO is clearly stronger for hard windows | Prefer polynomial memory for that task; investigate complementary hybrid only if residual-error profiles differ. | Combining two names and assuming novelty. |
| Interference, not temporal resolution, dominates | Test a delta-rule writer or a routed/key-partitioned hybrid, with all banks counted and a new hypothesis. | Assuming additional derivative order repairs arbitrary binding. |
| Only synthetic exponential queries show gains | Stop the broad utility claim; expand genuinely different tasks before investing in a decoder. | Calling target-kernel matching a breakthrough. |
| A large-state gain appears with clear under-resolution at small state | Escalate at a common larger budget when calibration permits and validate a predicted transition. | “It will work at scale” without a falsifiable prediction. |
| Small stable gain but higher total cost | Report a tradeoff and application-dependent preference. | Overall superiority. |
| Numeric gain but oracle language bridge fails | Repair or replace the bridge on development data. | Rejecting the memory mechanism or claiming agent success. |
| Simple exact/time-aware retrieval dominates within realistic storage | Recommend retrieval for the application; retain compact-memory work only for genuine storage constraints. | Hiding corpus/index costs for one side or ignoring the simple winner. |

## 13. Repository, artifact schemas and resumability

### 13.1 Minimal repository to create

```text
famjet-poc/
  README.md                     # how to start from this plan
  research_poc_plan.md           # this document, unchanged original plus amendments
  pyproject.toml
  requirements.lock             # generated after local compatibility checks
  configs/
    smoke.yaml
    data.yaml
    development_simple.yaml
    development_jet.yaml
    baseline_reproduction.yaml
    pilot.yaml
    confirm.yaml
    real_transfer.yaml
  famjet/
    doctor.py  prepare.py  run.py  train.py  state_io.py
    reproduce.py  pilot.py  confirm.py  extension.py  report.py  # CLI dispatchers
    data/       schema.py  synthetic.py  beijing.py
    memory/     jet.py  expbank.py  oracle.py  hippo.py  delta.py
                buckets.py  exact_store.py  gaver.py
    readers/    mlp.py  structured.py  probe.py  interpolate.py
    models/     task_model.py
    eval/       oracle.py  splits.py  budgets.py  timing.py  power.py  confirm.py
    language/   prefix_adapter.py
    analysis/   pilot.py  robustness.py  report.py
  tests/
    test_algebra.py  test_time.py  test_gradients.py  test_harness.py
    test_baselines.py  test_resume.py  test_no_leakage.py
  manifests/                     # assets, split roots, confirmations and hashes
  artifacts/environment/
  data/raw/                      # provenance/license manifest; not silently vendored
  data/derived/                  # deterministic caches with source/config hashes
  runs/<run_id>/                 # raw records, checkpoints, stdout and failures
  reports/
  preregistration.md
  DECISIONS.md  AUDIT.md  PROGRESS.md
```

Make the module command layout consistent: for example the CLI `python -m famjet.confirm` may be a thin dispatcher to `famjet.eval.confirm`; the same rule applies to `pilot`, `extension`, `reproduce` and `report`. Implement the listed dispatchers and test every documented `--help` entry point. Do not leave references to nonexistent entry points in a purportedly executable repository.

### 13.2 Configuration contract

Example **smoke** configuration; nullable confirmatory fields are deliberate blocking placeholders, not implicit defaults:

```yaml
schema_version: 1
phase: smoke
claim: numerical_and_pipeline_correctness
seed: 20261013
data:
  kind: synthetic
  generator_version: initial_spec_v1
  key_mode: one_hot
  keys: 16
  value_dim: 8
  length: 128
  queries_per_history: 8
  split_manifest: null
model:
  method: jet
  heads: 1
  anchors: 4
  key_dim: 16
  state_dtype: float32
  rates: {mode: fixed_log_grid, min: 0.001, max: 1.0}
  reader: {kind: structured_mlp, width: 128, hidden_layers: 2}
  interpolation: false
  probes: 1
  eps: 1.0e-8
training:
  writer: fixed
  encode_mode: terminal
  microbatch: 1
  effective_batch: 1
  optimizer: adamw
  learning_rate: 0.001
  weight_decay: 0.0
  steps: null
resources:
  device: discover
  cpu_fallback_in_mps_run: false
  memory_envelope_bytes: null
confirmation:
  hypotheses_hash: null
  histories: null
  training_replicates: null
  test_seed_manifest: null
  analysis_hash: null
```

A smoke run may use a small explicit number of optimizer steps solely to test the training path. A confirmation run must reject `null` scientific fields, an unversioned split, changed code/weights, or a development seed namespace. Its configuration must name the selected baseline family/configurations and actual byte/parameter envelopes. Save the fully resolved configuration, not only user-supplied overrides.

### 13.3 Run and prediction schemas

`run.json` contains `run_id`, phase, UTC timestamp, git commit/diff hash, plan/amendment hash, dependency lock hash, machine/environment hash, method/config hash, dataset/split hash, model/tokenizer revision if any, training/evaluation seeds, expected and actual budgets, status, failure category, start/end timestamps and checkpoint paths.

`predictions.jsonl` or Parquet contains one row per query:

```text
run_id, train_replicate, history_id, temporal_cluster_id, query_id,
split, regime, task, key_load, event_count, history_span, time_unit,
query_window, target, prediction, support_label, support_score,
normalized_loss, native_unit_error, nonfinite_flag, abstained,
state_payload_bytes, query_diagnostics
```

Do not store unavailable results as zeros. Use explicit missing/status fields. Maintain the mapping from every prediction to its exact raw event/query manifest. Protect any benchmark raw data according to its license.

`timing.jsonl` records method/run, shape, device, precision, timing boundary, cold/warm status, randomized block ID, repetitions, elapsed time, synchronization policy, event/query counts, observed memory, swap/power observations and failure status. Summary statistics are generated from these raw records.

`assets.json` records URL/DOI, revision, hash, downloaded bytes, observed schema, license, permitted project use, extraction/preprocessing commands and exclusions. Separate pretrained external assets from project-specific training expenditures. Record exploratory tuning cost by family as well as final-run cost.

### 13.4 Checkpoint and interruption policy

Checkpoint model, optimizer, scheduler, gradient-scaler state if used, all Python/NumPy/torch CPU/MPS RNG states supported by the installed stack, data-sampler position, gradient-accumulation position, configuration hash and source/data revisions. Prefer checkpoints at complete optimizer steps; if accumulated gradients are not serialized, restart the unfinished optimizer step rather than resume halfway with missing gradients.

For a live stream, serialize `Z,H` or baseline state, last timestamp, occupancy/conv caches, event offset and writer/rate parameter hashes. State made with different parameters cannot be resumed as if consistent. Save to a temporary file and atomically rename; verify checksums on load. Preserve the previous valid checkpoint and a failure journal.

Test an interrupted/resumed run against an uninterrupted run. Require exact CPU deterministic agreement where supported, otherwise a predefined numerical/trajectory tolerance plus identical consumed examples and optimizer-step counts. Do not claim bitwise MPS determinism without an actual test. A resume with different data ordering or stale state is a new run, not a continuation.

Do not use compressed checkpoint file size as the memory-capacity budget: logical allocated state and dtype determine the retained representation, even when zeros compress well. Physical peak memory and serialization size are additional metrics.

### 13.5 Agent authority and durable decisions

The coding agent may choose routine non-destructive implementation details, improve tensor batching, fix clearly demonstrated bugs, or reduce microbatch/chunk size while preserving the declared mathematical computation. It must log those changes with tests. Changes to hypotheses, raw information access, task populations, splits, primary metrics, state precision/capacity, baseline architecture, confirmation stopping rules or interpretation require a dated methodological amendment and, where needed, fresh protected data.

Update `PROGRESS.md` with the last completed gate, current artifact paths, blocking questions, next action and resource calibration. Update `DECISIONS.md` with alternatives considered, evidence used and the decision—not only favorable summaries. The project must resume from these artifacts without conversation memory.

## 14. Interpretation, continuation and completion criteria

A successful research execution ends with a defensible decision, not necessarily a positive model result.

| Outcome | Required evidence | Conclusion and next action |
|---|---|---|
| **Implementation invalid** | Broken direct-sum, gradient, budget, label or resume tests. | No scientific comparison yet. Repair and rerun the affected dependency chain; retain the invalid outputs with a clear label. |
| **Proxy insensitive** | Oracle/simple baselines are indistinguishable, no temporal changes occur at the tested scale, or reader controls cannot use known sufficient information. | Redesign development fixtures to express the mechanism; no rejection of the unexpressed hypothesis. Freeze fresh confirmation afterward. |
| **Local resource limitation** | Correct common-scale path is too costly or unsupported after faithful fallbacks. | Report exactly which claim cannot be tested locally and what component evidence exists. Do not substitute a CUDA result or call it a failure of the memory idea. |
| **Evidence against the selected mechanism** | Adequately precise fair comparisons show jets worse or dominated in the predeclared population/frontier. | Stop that configuration/claim; prefer the supported alternative. Escalation requires a specific predicted missing regime, not hope. |
| **Works, but no distinct advantage** | Correctness and task performance are good, but equivalent/stronger simple methods match them. | Classify as reproduction/engineering, not new architecture. Ship the simpler method when appropriate. |
| **Inconclusive** | Intervals remain wide, seeds conflict, real clusters are too few, or fairness remains unresolved. | State the detectable effect and missing evidence. Choose the next experiment by expected information per measured cost, not significance chasing. |
| **Narrow empirical improvement** | Fresh protected and replication evidence supports H1, with valid controls and disclosed tradeoffs. | Claim only the tested basis/population/budget advantage. Require H3/H5/H4 for broader associative/real/practical claims. |
| **Useful systems result** | Required baselines are beaten or a clear resource frontier is improved in a realistic task, with full costs and regressions disclosed. | Continue to a suitable language/application study; novelty remains separately assessed. |

The following must be present in the final project report: the full evaluated method/configuration inventory; all primary and important secondary results; paired uncertainty; accuracy–memory–latency frontiers; training/tuning expenditure; precision and numerical failures; source/asset licenses and revisions; failed runs; deviations/amendments; and a clear distinction between synthetic, real numeric and language/agent evidence.

Use claim wording such as: “On the preregistered observed-event temporal-query distribution, under X bytes of persistent state, method A changed normalized error by Y with interval Z relative to the tuned comparator, at the reported ingestion/query costs.” Fill those fields only from executed results. Do not use “groundbreaking,” “arbitrarily long exact memory,” “new function-valued neural-network category,” or “state of the art” without the corresponding much stronger evidence.

## 15. Audit record and unresolved risks

These are audits performed while constructing this plan, not independent external review and not an execution audit of a finished repository. The second challenge pass below identifies additional corrections beyond the initial literature and numerical assessment.

### 15.1 Scientific and prior-art audit

| Defect found | Revision made | Residual uncertainty / acceptance check |
|---|---|---|
| The apparent invention was described as a new function-valued memory. | Added the gamma/repeated-pole equivalence and mandatory coordinate control; narrowed the claim to an empirical/numerical tradeoff. | A full combination-specific novelty search may uncover further equivalents; no first-invention claim is authorized. |
| The original gap understated a continuum-queryable competing memory. | Made HiPPO Zoo's log-timescale model a required relevant comparator, not just background. | Reproduce its finite system and task adaptation before comparison. |
| Read-time probing was treated as extra retained information. | Require an all-information, compute-matched one-shot reader. | Conditional-computation benefits remain experimentally open. |
| Derivative stability could be mistaken for answer confidence. | Separated temporal sensitivity from calibrated abstention/reliability. | Calibration must hold under shift; derivative-only confidence is not accepted. |
| Smooth-memory targets could guarantee a favorable result. | Put hard windows/contrasts first and exponential targets in diagnostics. | Synthetic distributions can still favor an inductive bias; real transfer and simple summaries are necessary. |
| Fixed-size state was treated as universally more informative. | Added the nonnested-kernel/nullspace argument and interference/bits limitations. | Nonlinear learned writers require their own analysis; the linear argument is not overgeneralized. |

### 15.2 Experimental-validity audit

| Defect found | Revision made | Residual uncertainty / acceptance check |
|---|---|---|
| Jets used twice the temporal payload of a single bank. | Exact `J` versus `2J` augmented-state comparison, plus a broader resource frontier. | Account for all metadata, caches, conv state and active parameters. |
| Weak grids, undertrained baselines or larger readers could explain a gain. | Added learned/tuned anchors, convergence checks, generic readers, equal-tuning and best-credible views. | Any unfinished baseline optimization limits the conclusion. |
| Temporal distance and interference were conflated. | Independent length/load/gap axes and rare-key tests. | One-hot-key wins alone remain narrow. |
| Many queries/stations/seeds could inflate sample size. | History-level and crossed-seed analysis; temporal real-data clusters; pilot-based precision. | Small finite real-data cluster count may prevent small-effect claims. |
| Nonfinite/unsupported outputs could be excluded favorably. | Defined a common training-prior fallback and mandatory failure accounting. | Stability guard must be preregistered; average loss does not excuse hidden failures. |
| Exact full context was called an upper bound for learned models. | Separated exact arithmetic label oracle, practical raw-history method and learned full-attention reference. | Real language decoders introduce an additional bottleneck. |
| Tiny gains could be promoted using an arbitrary percentage threshold. | Replaced it with effect/uncertainty, power curves, explicit costs and application-conditional usefulness. | Practical user value remains unspecified and cannot be inferred from significance. |

### 15.3 Implementation and feasibility audit

| Defect found | Revision made | Residual uncertainty / acceptance check |
|---|---|---|
| An M3 Max was assumed to have maximum memory and suitable kernels. | Added actual discovery, op/gradient tests, CPU reference fallback and measured resource envelopes. | Nothing in this plan substitutes for a local compatibility run. |
| Small inference state concealed large training activation cost. | Added the 16.25 GiB BPTT-state example and exact terminal contractions/checkpointing. | Nonlinear baselines need their own measured backward-memory profile. |
| Continuous interpolation could produce invalid normalizers. | Executed a counterexample; removed interpolation from the indispensable core and required error/fallback tests. | Useful interpolation ranges remain an experimental question. |
| A generic HiPPO helper would select the weaker scale-coordinate variant. | Inspected the actual multiscale script and specified the log-rate construction and source pins. | Finite projection/discretization error and Bessel/normalization conventions need reproduction tests. |
| Source descriptions hid a scalar-location associative implementation and nonpermissive licensing. | Recorded code/doc distinction, GDN-2 NC restrictions, and missing Time-Local license gate. | Source permission and commercial-use questions are not resolved by this research plan. |
| Missing value channels could share an invalid denominator. | Scalar observed PM2.5 core and explicit per-variable normalization rule. | Any multivariate extension must repeat missingness tests. |
| **Second-pass defect:** raw log-rates/query times would break reader unit equivariance. | Added a transformed reference duration and dimensionless reader inputs; noted polynomial-state units. | Test the complete predictor, not only recurrence state, under unit conversion. |
| **Second-pass defect:** a 16-key/eight-value synthetic checkpoint cannot directly serve a 12-key/scalar real task. | Specified separately trained shape-matched synthetic models for frozen-weight transfer. | This is calibrated, shape-matched transfer, not a claim that one unchanged model handles arbitrary schemas. |
| Frozen language weights were confused with no backward activations or unchanged behavior. | Explicit gradient-through-prefix path, oracle bridge test, true adapter-off bypass and active-mode guardrails. | Actual decoder memory/support and benchmark transfer are untested. |
| Resume and project entry points were underspecified. | Added parameter/state hashes, atomic checkpoints, consumed-data/RNG records and explicit CLI dispatchers. | Run interruption equivalence and every documented command before declaring handoff execution complete. |

A final consistency pass checked dependency order, the distinction between source facts and proposed work, core operation without a language model, confirmation before protected inspection, and negative/inconclusive reporting. No additional blocking conceptual defect was identified in that pass; this is not a guarantee of correctness or completeness. Remaining prerequisites are explicit gates, and remaining scientific uncertainty is attached to falsifiable hypotheses.

## 16. References, inspected evidence and search record

The identifiers below are ordinary Markdown references. URLs point to public sources; versions, source files and inspection limits are specified so the coding agent need not reconstruct them from conversation citations. Access date for web/code/documentation inspection: **13 October 2026**. A paper's reported performance is not independently validated merely because the paper was read.

### 16.1 Evidence ledger

**R01 — Classical gamma memory.** Bert de Vries and José C. Príncipe. *A Theory for Neural Networks with Time Delays.* NeurIPS 1990, proceedings published 1991. [Original paper][R01]. Inspected the mathematical text and rendered pages containing the gamma cascade. This is direct historical overlap with the repeated-pole mechanism, not evidence of modern language-task performance. The coordinate equivalence in Section 4 is also derived explicitly in this plan.

**R02 — Laplace representations of history.** Karthik H. Shankar and Marc W. Howard. *A Scale-Invariant Internal Representation of Time.* Neural Computation 24(1), 134–193, 2012; online publication 2011. DOI: `10.1162/NECO_a_00212`. [Publisher identifier][R02]; [PubMed record](https://pubmed.ncbi.nlm.nih.gov/21919782/). Inspected metadata and abstract and traced the mechanism through later primary sources R03–R04. This establishes historical context, not a directly reproduced associative-memory baseline.

**R03 — Associative temporal Laplace memory.** Marc W. Howard, Zahra G. Esfahani, Bao Le and Per B. Sederberg. *Learning temporal relationships between symbols with Laplace Neural Manifolds.* arXiv:2302.10163, first submitted 20 February 2023. [Inspected full-text version v2, 26 August 2024][R03]; [v4 metadata, 22 September 2024](https://arxiv.org/abs/2302.10163v4). Relevant sections describe symbol-conditioned temporal relationships. Full-text inspection here was of v2, not a claim that every later revision was compared. The cognitive-symbol setting differs from modern long-context language evaluation.

**R04 — Time-Local Transformer.** Dickson, Mochizuki-Freeman, Kabir and Tiganj. *Time-Local Transformer.* Computational Brain & Behavior; online 2025, journal issue 2026. DOI: `10.1007/s42113-025-00253-9`. [Article][R04]. Inspected relevant architecture, inversion and experimental sections. The implemented timeline uses Gaver–Stehfest inversion, so treating it simply as a direct high-order derivative implementation would be inaccurate. Its motivation and experiments do not establish state-of-the-art results for the present task; R22 records the implementation and permission limitations.

**R05 — HiPPO foundations.** Albert Gu, Tri Dao, Stefano Ermon, Atri Rudra and Christopher Ré. *HiPPO: Recurrent Memory with Optimal Polynomial Projections.* NeurIPS 2020; arXiv:2008.07669. [Paper record][R05]. Inspected metadata/abstract; the numerical generators relevant to this handoff were additionally inspected in R07. Establishes polynomial history projection as prior art. No original-paper benchmark reproduction was performed during this research.

**R06 — Closest explicit multiscale competitor.** Jack Goffinet, Casey Hanks and David E. Carlson. *HiPPO Zoo: Explicit Memory Mechanisms for Interpretable State Space Models.* arXiv:2602.21340v2, 29 May 2026; first submitted 24 February 2026. [Full text][R06]; [version record](https://arxiv.org/abs/2602.21340v2). Inspected multiscale and associative mechanisms and supporting appendices, including the coordinate-change discussion. Its continuum-of-timescales construction directly narrows the proposed novelty. Reported synthetic reconstruction and short associative experiments have different tasks and budgets from this plan. The arXiv record's ICML statement was observed; proceedings publication was not separately verified.

**R07 — HiPPO Zoo implementation.** Goffinet and contributors. [Official repository][R07]. Verified default-branch commit: `04e38b3cf246f4c50cb6c6a7e10597e2e37eebc7`. Inspected [`src/mpm/models.py`](https://github.com/jackgoffinet/hippo-zoo/blob/04e38b3cf246f4c50cb6c6a7e10597e2e37eebc7/src/mpm/models.py), [`src/mpm/polynomials.py`](https://github.com/jackgoffinet/hippo-zoo/blob/04e38b3cf246f4c50cb6c6a7e10597e2e37eebc7/src/mpm/polynomials.py) and [`scripts/multiscale_hippo.py`](https://github.com/jackgoffinet/hippo-zoo/blob/04e38b3cf246f4c50cb6c6a7e10597e2e37eebc7/scripts/multiscale_hippo.py). Repository MIT labeling was observed. The inspected core uses JAX/Flax; Torch experiment scripts also exist. The scalar-location associative implementation and log-rate script, rather than docstring shorthand or the generic helper alone, inform the required port. Code was inspected, not executed.

**R08 — Multiscale retention.** Yutao Sun and colleagues. *Retentive Network: A Successor to Transformer for Large Language Models.* arXiv:2307.08621, 2023. [Paper record][R08]. Inspected abstract/metadata. Informs the distinct-decay-bank comparison and the recurrent/parallel/chunkwise context. This plan explicitly calls its simplified bank “retention-style,” not a reproduction of the entire pretrained RetNet architecture.

**R09 — Gated delta-rule memory.** Songlin Yang, Jan Kautz and Ali Hatamizadeh. *Gated Delta Networks: Improving Mamba2 with Delta Rule.* arXiv:2412.06464v3, 6 March 2025; first submitted December 2024; ICLR 2025. [Paper record][R09]. Inspected abstract/metadata; the recurrence reductions are additionally supported by the detailed method/code inspection in R10–R11. Provides a stronger overwrite-aware alternative to additive associative memory, without implying its published training scale is reproduced locally.

**R10 — Contemporary erase/write baseline.** Ali Hatamizadeh, Yejin Choi and Jan Kautz. *Gated DeltaNet-2: Decoupling Erase and Write in Linear Attention.* arXiv:2605.22791v1, 21 May 2026. [Full text][R10]. Inspected recurrence, gate construction and implementation-related appendices. Informs the native tensor baseline and exact KDA/gated-delta reductions. Reported large-model training is context, not a controlled laptop comparison or an independently verified performance claim.

**R11 — GDN-2 implementation and license.** NVIDIA. [Official repository][R11]; [`lit_gpt/gdn2.py`](https://github.com/NVlabs/GatedDeltaNet-2/blob/main/lit_gpt/gdn2.py); [`LICENSE`](https://github.com/NVlabs/GatedDeltaNet-2/blob/main/LICENSE). Inspected the wrapper, gate/conv conventions, Triton/FLA dispatch and license. Recorded file blob hashes: wrapper `4a0199c69b2097d2637bb8714d707bd53e67e91d`; license `d50677fec39f78e04515c615d30ad16741f8a29c`. These are file hashes, not invented commit pins. The license limits use to research/evaluation; it is not a permissive open-source license. Resolve and record the corresponding full commit before any reuse; permission for a later use remains a gate.

**R12 — Newer state-space reference.** Lahoti, Li, Chen, Wang, Bick, Kolter, Dao and Gu. *Mamba-3: Improved Sequence Modeling using State Space Principles.* arXiv:2603.15569v1, 16 March 2026. [Full text][R12]. Inspected the principal state-space changes, including richer discretization and state interactions. Included to avoid treating Mamba-2 as the newest relevant family member. A native, faithful implementation and its large-scale reported outcomes were not reproduced here; it is a broader reference rather than an unperformed controlled comparison.

**R13 — Routing between memories.** Jusen Du, Weigao Sun, Disen Lan, Jiaxi Hu and Yu Cheng. *MoM: Linear Sequence Modeling with Mixture-of-Memories.* arXiv:2502.13685v1, 19 February 2025. [Full text][R13]; [repository](https://github.com/OpenSparseLLMs/MoM). Inspected the routing mechanism and memory-accounting implications. Query/content-dependent selection between states is not a new concept. Different routing, state size and information retention mean its published outcomes cannot directly establish the present fixed-budget tradeoff.

**R14 — Adaptive hierarchical decay.** Yaxita Amin, Helen Zichen Li, Mengfan Zhang and Samet Ayhan. *Adaptive Memory Decay for Log-Linear Attention.* arXiv:2605.06946v1, 7 May 2026. [Full text][R14]. Inspected the content-conditioned decay and hierarchical-state mechanism. Supports the assessment that adaptive forgetting itself is not novel. A hierarchy with state growing with history differs from a strictly fixed-state budget and must not be silently treated as the same constraint.

**R15 — Function-space architecture context.** Julius Berner, Miguel Liu-Schiaffini, Jean Kossaifi, Valentin Duruisseaux, Boris Bonev, Kamyar Azizzadenesheli and Anima Anandkumar. *Principled Approaches for Extending Neural Architectures to Function Spaces for Operator Learning.* arXiv:2506.10973v1, 12 June 2025. [Preprint record][R15]; [2026 journal DOI](https://doi.org/10.1038/s42256-026-01267-z). Inspected preprint metadata and abstract. Access to the journal article was limited; full journal-text verification is not asserted. This supports the broad-context distinction, not the decisive recurrence, novelty or numerical claims.

**R16 — Core real-world dataset.** Song Chen / UCI Machine Learning Repository. *Beijing Multi-Site Air Quality.* Dataset dated 2017; repository donation record 2019. DOI: `10.24432/C5RK5G`. [Dataset catalog][R16]; [DOI](https://doi.org/10.24432/C5RK5G). Inspected catalog schema, temporal coverage, missingness information, archive listing and CC BY 4.0 license. The data were not downloaded or independently audited during this investigation. WP00/WP02 must verify the actual archive, chronology, station identities and counts before use; no measured real-data result is reported.

**R17 — Long-term conversational memory evaluation.** Di Wu and colleagues. *LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory.* arXiv:2410.10813v2; ICLR 2025. [Full text][R17]; [official repository](https://github.com/xiaowu0162/longmemeval). Inspected task categories, evaluation design and repository overview. Informs eventual temporal/update/abstention assessment. An unavailable paid or external judge cannot become a hidden core dependency; a local scoring modification must be labeled nonofficial. Data permission is a separate check from code licensing.

**R18 — Newer agent-memory benchmark.** Di Wu, Zixiang Ji, Asmi Kawatkar, Bryan Kwan, Jia-Chen Gu, Nanyun Peng and Kai-Wei Chang. *LongMemEval-V2: Evaluating Long-Term Agent Memory Toward Experienced Colleagues.* arXiv:2605.12493v1, 12 May 2026, work in progress. [Paper record][R18]; [official repository](https://github.com/xiaowu0162/LongMemEval-V2). Inspected abstract/metadata and repository-level information. Its richer experience/context-gathering setting informs the need for strong file/search/RAG alternatives. Large histories and distinct evaluations are not part of the minimum local claim; no benchmark run or full data/license audit was performed here.

**R19 — Distributed-fact reasoning.** Yuri Kuratov, Aydar Bulatov, Petr Anokhin, Ivan Rodkin, Dmitry Sorokin, Artyom Sorokin and Mikhail Burtsev. *BABILong: Testing the Limits of LLMs with Long Context Reasoning-in-a-Haystack.* arXiv:2406.10149v2, 6 November 2024; NeurIPS 2024 Datasets and Benchmarks. [Paper record][R19]; [official repository](https://github.com/booydar/babilong). Inspected paper metadata/abstract and repository task documentation. Relevant only after the language bridge works. Constituent-corpus permissions and generation provenance remain asset gates; no dataset or model performance was reproduced.

**R20 — Broader long-context evaluation.** Yushi Bai and colleagues. *LongBench v2: Towards Deeper Understanding and Reasoning on Realistic Long-context Multitasks.* ACL 2025. [Proceedings record][R20]; [dataset card](https://huggingface.co/datasets/zai-org/LongBench-v2). Inspected the proceedings entry and dataset-level task information. Provides broader evaluation context, not a direct temporal-kernel diagnostic. Version, permission and scoring checks precede use, and results on a restricted local subset cannot be presented as full benchmark scores.

**R21 — Optional frozen decoder assets.** Qwen Team. *Qwen3-0.6B* and *Qwen3-1.7B*, official model cards, 2025. [0.6B card][R21]; [1.7B card](https://huggingface.co/Qwen/Qwen3-1.7B). Inspected architecture/use guidance and Apache 2.0 labeling. Weights were not loaded; revision pins and actual dtype/gradient/MPS compatibility must be verified locally. Their choice is a manageable shared decoder for a controlled adapter experiment, not a claim that they are the latest or strongest available models.

**R22 — Timeline implementation details.** cogneuroai / Time-Local Transformer contributors. [Repository][R22]; [`gaver_cell.py`](https://github.com/cogneuroai/time-local-transformer/blob/main/gaver_cell.py). Inspected the repository tree and complete cell. Recorded cell blob: `a380990b7add55b4f0109ad03a8d2d40ae5b3b31`. The code's input integration, inversion coefficients and float64-related buffers inform Section 7. The inspected tree had no license file; this does not establish permission to vendor code. Preserve the permission gate and independently specify any equation-based implementation. The tree response was not treated as a verified commit pin.

**R23 — Patent-disclosure triage, not legal clearance.** Aaron R. Voelker and Christopher David Eliasmith / Applied Brain Research. *Legendre memory units in recurrent neural networks.* EP3935568A1, published 12 January 2022; claimed priority 6 March 2019. [Patent disclosure][R23]. Inspected the technical disclosure and bibliographic record as adjacent prior art. No conclusion about claim validity, enforceability, current legal status, patentability or freedom to operate follows. A legal opinion would require separate qualified review and jurisdiction-specific investigation.

**R24 — Local acceleration documentation.** PyTorch contributors. *MPS backend* and *torch.mps*, PyTorch 2.14 documentation, accessed 13 October 2026. [Backend notes][R24]; [MPS API documentation](https://docs.pytorch.org/docs/2.14/mps.html). Inspected availability/platform guidance and synchronization/memory APIs. Documentation support is not a successful local operation or gradient test. Lock the version actually installed after WP00; the core does not require pretending an untested wheel or kernel works.

**R25 — MPS execution settings.** PyTorch contributors. *MPS Environment Variables*, PyTorch 2.14 documentation. [Official page][R25]. Inspected CPU fallback, fast-math and allocator-watermark guidance. Informs native-backend measurement boundaries and the prohibition on disabling protective limits simply to force a run through memory pressure. The selected conservative execution settings are this plan's protocol, not benchmark measurements or a promise of deterministic MPS execution.

**R26 — Interference and load diagnostics.** Kichang Lee, JaeYeon Park, Songkuk Kim and JeongGil Ko. *How Linear Attention Remembers.* arXiv:2609.33093v1, 27 September 2026. [Full text][R26]. Inspected the controlled memory/intervention analysis and scope. Informs separate fact-load, temporal-distance and selective-interference tests. Its findings concern the recurrent architectures and scales investigated; they are not a theorem covering all fixed-state compressors. No independent replication was executed here.

### 16.2 Search coverage and stopping rationale

Research was conducted through **13 October 2026** using primary-paper discovery, backward/forward reference tracing, official repository inspection, current framework documentation, dataset/model cards and a targeted patent-disclosure check. The search was organized by mechanisms that could invalidate the proposal, not by a fixed paper quota.

| Search family and representative terms | Consequence for this plan |
|---|---|
| “function-valued neural networks,” “neural operator architecture principles,” “function spaces,” hypernetworks and conditional features | Broad functional representation is established; operator learning is separated from bounded retrospective memory. |
| “gamma memory,” “gamma neural network,” “repeated poles,” “Laguerre memory,” “Laplace temporal memory,” “Laplace Neural Manifolds,” derivatives of forgetting rate | Exposed the strongest historical mechanism overlap and motivated exact coordinate-equivalence controls. |
| “HiPPO Zoo,” “multiscale HiPPO,” “continuous timescale,” “associative polynomial memory,” “Legendre memory units” | Found the direct continuum-queryable competitor, traced its log-rate construction and inspected the actual scalar-location associative code. |
| “Time-Local Transformer,” “Gaver Stehfest,” inverse Laplace history reconstruction | Distinguished reconstruction from moment features and identified different continuous-input semantics, precision costs and a source-permission gate. |
| “Gated DeltaNet 2,” “Kimi Delta Attention,” “Mamba-3,” “mixture of memories,” adaptive hierarchical forgetting | Strengthened contemporary overwrite/routing comparisons, accounted for extra state and avoided assuming CUDA kernels execute on Apple hardware. |
| “How Linear Attention Remembers,” memory load, recall interference, temporal distance | Added independent load/distance axes rather than interpreting every long-horizon failure as decay. |
| “LongMemEval V2,” temporal memory benchmarks, BABILong, LongBench v2, time-aware retrieval | Preserved the eventual agent-memory question while preventing unsupported transfer from arithmetic/sensor tasks and weak retrieval comparisons. |
| PyTorch MPS operation/gradient support, memory APIs, model cards, dataset licenses, source licenses | Put actual hardware discovery, native-operation smoke tests, asset permission and measured resource gates ahead of dependent experiments. |
| “Legendre memory units” patents and adjacent temporal-memory disclosures | Identified relevant patent disclosure without claiming legal clearance or legal novelty. |

Later targeted tracing primarily changed **baseline fidelity, input semantics, numerical conditioning, licensing and claim scope**, rather than establishing a defensible new class of memory. The resulting coverage is sufficient to choose and falsify the proposed local experiment, not to prove that no equivalent unpublished, differently named or inaccessible method exists. A future publication claim requires another targeted search against its actual successful mechanism and final results.

Remaining blind spots include unpublished implementations; incomplete journal access for the broad function-space article; revisions not fully compared where explicitly noted; unexecuted external code; unverified local package/kernel behavior; language-dataset constituent licenses; and the lack of a specified product workload or utility function. The research does not claim exhaustive patent searching, independent source reproduction, or direct knowledge of the target laptop's resources.

## Appendix A. Complete numerical validation script executed during this investigation

This script is included so the handoff has no dependency on another chat attachment. Save the code as `validation/numerical_checks.py` in the proposed repository after creating that directory. With NumPy installed in the active environment, `python validation/numerical_checks.py > artifacts/numerical_checks.json` runs the check; create `artifacts/` first. These are ordinary Python/shell commands, not an existing project CLI.

The output is diagnostic JSON, not an automatic pass/fail verdict. Section 5 records the observed environment and results; WP01 must add assertions appropriate to the tested condition and extend the edge cases. Last digits may vary with numerical libraries. This script does not train a model, test PyTorch gradients, benchmark a GPU, or prove Apple compatibility.

```python
"""Independent NumPy checks; not model training or Apple hardware benchmarks."""
import json, platform, sys
import numpy as np
SEED=20261013
rng=np.random.default_rng(SEED)
n,kdim,vdim=512,5,3
t=np.cumsum(rng.lognormal(0,1,n)); t-=t[0]
rates=np.geomspace(1e-3,1.,4)
k=rng.uniform(.05,1,(n,kdim)); v=rng.normal(size=(n,vdim)); g=rng.uniform(0,1,n)
x=g[:,None,None]*k[:,:,None]*np.concatenate([v,np.ones((n,1))],axis=1)[:,None,:]
age=t[-1]-t

def direct(lam,dtype=np.float64):
    la=np.array(lam,dtype=dtype); aa=age.astype(dtype)
    w=np.exp(-la[:,None]*aa[None,:])
    z=np.einsum('jn,nkv->jkv',w,x.astype(dtype))
    j=np.einsum('jn,nkv->jkv',w*la[:,None]*aa[None,:],x.astype(dtype))
    return z,j

def recurrence(dtype):
    z=np.zeros((len(rates),kdim,vdim+1),dtype=dtype); j=np.zeros_like(z)
    l0=z.copy();l1=z.copy();last=t[0]
    lam=rates.astype(dtype)
    for ti,xi in zip(t,x):
        d=(lam*dtype(ti-last))[:,None,None]; a=np.exp(-d)
        j=a*j+(a*d)*z; z=a*z+xi.astype(dtype)
        l1=a*l1-2*(a*d)*l0+xi.astype(dtype); l0=a*l0+xi.astype(dtype)
        last=ti
    return z,j,l0,l1

def rel(a,b):return float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-30))
z,j=direct(rates); out={'environment':{'os':platform.platform(),'python':sys.version.split()[0],'numpy':np.__version__,'device':'CPU, x86_64 Linux; not M3 Max'},'seed':SEED,'shape':[n,kdim,vdim,len(rates)]}
for dt in [np.float64,np.float32]:
    zz,jj,l0,l1=recurrence(dt)
    out[str(dt)]={'state_relative_error':rel(zz,z),'jet_relative_error':rel(jj,j),'laguerre_basis_jet_relative_error':rel((l0-l1)/2,jj)}
q=rng.uniform(.1,1,kdim)
num=np.einsum('k,jkv->jv',q,z[...,:-1]);den=np.einsum('k,jk->j',q,z[...,-1]);jn=np.einsum('k,jkv->jv',q,j[...,:-1]);jd=np.einsum('k,jk->j',q,j[...,-1]);r=num/den[:,None]
slope=-jn/den[:,None]+r*jd[:,None]/den[:,None]
def read(lam):
    zz,_=direct(lam); return np.einsum('k,jkv->jv',q,zz[...,:-1])/np.einsum('k,jk->j',q,zz[...,-1])[:,None]
h=1e-5; fd=(read(rates*np.exp(h))-read(rates*np.exp(-h)))/(2*h)
out['normalized_log_rate_derivative_relative_error']=rel(slope,fd)
out['finite_difference_jet_relative_errors']={}
for dt in [np.float64,np.float32]:
    out['finite_difference_jet_relative_errors'][str(dt)]={}
    for h in [1e-1,1e-2,1e-3,1e-4,1e-5,1e-6]:
        zp,_=direct(rates*np.exp(h),dt);zm,_=direct(rates*np.exp(-h),dt)
        out['finite_difference_jet_relative_errors'][str(dt)][str(h)]=rel(-(zp-zm)/(2*h),j)
# Single old nonnegative impulse, broad interpolation interval in log-rate.
x0,x1=0.,np.log(1000.);u=np.linspace(0,1,10001);hh=x1-x0
f0,f1=np.exp(-np.exp(x0)),np.exp(-np.exp(x1));m0=-np.exp(x0)*f0;m1=-np.exp(x1)*f1
herm=(2*u**3-3*u**2+1)*f0+(u**3-2*u**2+u)*hh*m0+(-2*u**3+3*u**2)*f1+(u**3-u**2)*hh*m1
imin=int(herm.argmin()); la=float(np.exp(x0+u[imin]*hh))
out['hermite_positive_denominator_counterexample']={'age':1.,'endpoint_rates':[1.,1000.],'interpolated_min':float(herm[imin]),'rate_at_min':la,'exact_at_rate':float(np.exp(-la))}
print(json.dumps(out,indent=2))
```

[R01]: https://proceedings.neurips.cc/paper/1990/file/6c524f9d5d7027454a783c841250ba71-Paper.pdf
[R02]: https://doi.org/10.1162/NECO_a_00212
[R03]: https://arxiv.org/html/2302.10163v2
[R04]: https://link.springer.com/article/10.1007/s42113-025-00253-9
[R05]: https://arxiv.org/abs/2008.07669
[R06]: https://arxiv.org/html/2602.21340v2
[R07]: https://github.com/jackgoffinet/hippo-zoo
[R08]: https://arxiv.org/abs/2307.08621
[R09]: https://arxiv.org/abs/2412.06464v3
[R10]: https://arxiv.org/html/2605.22791v1
[R11]: https://github.com/NVlabs/GatedDeltaNet-2
[R12]: https://arxiv.org/html/2603.15569v1
[R13]: https://arxiv.org/html/2502.13685v1
[R14]: https://arxiv.org/html/2605.06946v1
[R15]: https://arxiv.org/abs/2506.10973v1
[R16]: https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data
[R17]: https://arxiv.org/html/2410.10813v2
[R18]: https://arxiv.org/abs/2605.12493v1
[R19]: https://arxiv.org/abs/2406.10149v2
[R20]: https://aclanthology.org/2025.acl-long.183/
[R21]: https://huggingface.co/Qwen/Qwen3-0.6B
[R22]: https://github.com/cogneuroai/time-local-transformer
[R23]: https://patents.google.com/patent/EP3935568A1/en
[R24]: https://docs.pytorch.org/docs/2.14/notes/mps.html
[R25]: https://docs.pytorch.org/docs/2.14/mps_environment_variables.html
[R26]: https://arxiv.org/html/2609.33093v1
