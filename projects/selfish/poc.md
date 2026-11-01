# Research and proof-of-concept plan: self-interested neural units

**Research date:** 12 October 2026  
**Target execution environment:** one MacBook Pro with an M3 Max; actual memory, macOS, storage and usable acceleration must be discovered.  
**Deliverable status:** researched design and executable handoff, not a trained system or a claim of improved performance.  
**Primary working method name:** `economy_rb`. This is a project identifier, not a claim that a new algorithm has been established.

## 1. Executive assessment and decision

### Verdict

The broad proposal is substantially anticipated. Neural economies, neuronal competition, local reinforcement learning, cooperation among neuron populations, and replacement of unsuccessful components are not a new paradigm. Particularly close precedents include the Neural Bucket Brigade, coagent networks, and **Learning by Competition of Self-Interested Reinforcement Learning Agents**. The last of these explicitly gives hidden units a private objective related to their outgoing weights. Modern continual-learning methods already measure usefulness and replace or perturb low-utility neurons or weights. [R01–R08, R11–R17, R28]

**No methodological novelty is established by this investigation.** The defensible research opportunity is a tightly controlled empirical question:

> Can locally updated stochastic neurons, paid according to counterfactual task contribution and subject to resource charges, bankruptcy, rebirth and local connection changes, improve online adaptation under distribution shift after accounting for all credit-computation and management costs—beyond strong coagent, reset, regularization and sparse-training alternatives?

Choose this direction rather than an ungrounded attempt to produce general intelligence. Start by reproducing the closest self-interested-neuron implementation and testing mathematical invariants. Then test the incremental effects of attribution, turnover and topology. A useful outcome may be a negative result, an equivalence result, or a limited improvement in an identified operating regime.

The primary implementation uses a **single stochastic hidden layer and a locally supervised linear readout**. This permits exact two-action counterfactual evaluation at modest cost. A mandatory small two-hidden-layer experiment tests whether the credit mechanism survives genuine dependencies between hidden neurons. Scaling a deep, recurrent economic society is not part of the initial competitive claim.

### What is preserved, and what is changed

| Original intent | Executable interpretation | Limitation or explicit refinement |
|---|---|---|
| Individual neurons have interests | Each unit owns its incoming weights and connections and improves its private task-contribution-minus-cost payoff | These are myopic optimization rules, not deliberative agents planning their lifetime |
| Energy, influence and survival | A reserve accumulates discounted net utility; nonpositive reserves can trigger replacement | Credits are computational accounting units, not physical energy; raw popularity is not rewarded |
| Units change connections | Units propose incoming-edge additions, deletions or swaps, subject to a global capacity constraint | A trusted executor evaluates proposals; the initial graph is feedforward |
| Unsuccessful units disappear | A bankrupt identity is replaced by a newly initialized identity in the same capacity slot | Fixed-capacity population turnover, not unrestricted population growth; dormancy is deferred |
| No conventional end-to-end backprop initially | Hidden units use local score-function or conditional policy-gradient updates; the readout uses its own supervised update | A global loss evaluator provides scalar credit. This is **not fully decentralized learning** |
| Cooperation and specialization may emerge | Measure intervention effects and functional specialization; test a two-layer dependency task | Correlated activations are not proof of cooperation; the shallow experiment cannot establish hierarchy or autonomous memory |
| About 1,000 neurons | Validate small cases, then include 1,000 hidden units when calibrated throughput supports informative repetitions | One thousand is an intended local scale, not a magical threshold for emergence |

### Decision priorities

The first scientific threat is **equivalence to existing methods**, not insufficient model size. A discounted reserve is a utility trace with a threshold. Implement its mathematically equivalent, non-economic formulation and require identical results. The second threat is obtaining a gain merely through better credit assignment or extra label-based computation. The third is comparing against inadequately tuned backpropagation or outdated replacement heuristics.

A successful result must survive those explanations. Even then, describe it as an improvement in the tested online-learning setting, not a demonstration that intelligence generally emerges from selfishness.

## 2. Problem, scope, beneficiaries and hypotheses

### 2.1 Operational problem

An online predictor receives blocks of observations, predicts before seeing their labels, receives those labels, and adapts. The relationship between inputs and labels may change. Model capacity, elapsed adaptation time and storage are limited. Intended beneficiaries are researchers and, conditionally, developers of locally adapting predictors. No production customer, application-specific service level, or established commercial bottleneck has been supplied.

The hypothesized limitation is inefficient renewal of representations: some learned components become obsolete, while indiscriminate replacement destroys useful ones. Loss of plasticity has been demonstrated in specified continual-learning experiments, but that does not establish that it is the bottleneck on every proposed workload. Regularization and perturbation already help in several settings; more recent work shows that replacing individual weights can outperform replacing entire units in small or normalized networks. [R13, R28]

The existing workflow to beat is therefore **well-tuned online learning with regularization, selective renewal and appropriate non-neural models**, not an intentionally fragile network trained once and frozen.

### 2.2 Inputs, outputs and access contract

An observation block is `X: float32[B,d]`; labels are integer class IDs `y: int64[B]`. The predictor outputs a normalized probability vector per observation, with all predictions committed before any label from that block is available. Task boundaries and synthetic teacher parameters are not supplied to methods. Boundary-aware methods are diagnostics only.

The core task has immediate feedback after each block. Blocks introduce a feedback delay of up to `B` observations; this is disclosed and matched for every method, including sequential tree models. A `B=1` sensitivity condition tests genuinely per-observation adaptation. Delayed labels, recurrent environments and actions that affect future observations are separate extensions, not silently included in the theory.

No pretrained weights, paid APIs, remote GPUs or hidden external data are required. The candidate has no access to files, network services, evaluation labels, experiment selection or the reward implementation. Its “agency” is limited to numerical state inside the model.

### 2.3 Hypotheses and discriminating observations

| ID | Hypothesis and supporting basis | Strong alternative explanation | Weakening/falsifying observation and scope |
|---|---|---|---|
| H1: credit | Counterfactual scalar credit makes local stochastic learning competitive enough to study turnover | Standard variance reduction, not an economy; coagent/difference-reward precedents already explain it [R06, R09] | Exact-gradient checks fail: implementation invalid. Correct method consistently loses to conditional local-gradient control: reject the chosen estimator, not all local learning |
| H2: turnover | Utility-triggered replacement improves adaptation beyond fixed-rate or activity-based renewal | Any reset, a better utility selector, or extra exploration explains the result [R13–R17, R28] | No advantage against matched reset timing/rate, exact-utility ranking, CBP, SWR and regularization: no demonstrated value from the reserve policy |
| H3: topology | Private edge changes help allocate a limited connection budget after support drift | Generic sparse optimization or additional replay computation [R18, R19] | Gain disappears with the same proposal evaluator applied to a non-economic learner or against tuned RigL: do not attribute it to selfish agency |
| H4: incentives | Task-valued credit resists activity, duplication and weight-scale gaming better than raw influence payment | The trusted global evaluator, not decentralized organization, provides the protection | Activity or payment rises without out-of-sample task improvement; useful complements are eliminated; parasites persist: incentive design fails in that regime |
| H5: practical value | Any learning gain survives complete end-to-end costs and important regressions | More computation, label re-use, capacity, different inference or latency | The candidate is dominated on predictive risk, elapsed time and memory, or hides a large stationary/recurrent-task regression: no practical advantage |
| H6: dependencies | Correct interventions transmit useful credit through hidden-to-hidden dependencies | Only the cheap final-layer special case works | Correct small two-layer learning fails despite adequate sensitivity and successful controls: restrict any result to shallow networks |

These are separable claims. Confirming H1 does not confirm H2–H6. No hypothesis requires consciousness, felt self-preservation, or an evolutionary transition to general intelligence.

### 2.4 Strongest case for and against

**Case for:** sparse, changing tasks may reward a division of labor. Cheap exact intervention scores in a shallow network can direct local adaptation and selective renewal without hidden-layer backpropagation. Resource charges and targeted renewal could protect useful components while exploring new input supports. The mechanism is observable enough to diagnose causal failure rather than merely inspect a final score.

**Case against:** the global evaluator may be doing the difficult work. With fixed topology, the local update is a familiar gradient estimator. Reserves are utility traces, bankruptcy is replacement, and edge mutations are local search. Private marginal credit is not a balanced market price; redundant components and complementary coalitions create difficult attribution problems. Mutation and survival thresholds add tuning burden, instability and overhead. Continuous backpropagation, UPGD, SWR or an adaptive forest may win outright.

**Headroom:** for a synthetic teacher with known conditional probability `q(y|x)`, estimate the Bayes risk `E[-log q(y|x)]` on independently generated evaluation observations. Report the baseline’s excess risk over this floor. If the baseline is already near the floor relative to the achievable confidence interval, that workload cannot support a meaningful improvement claim. For real data the Bayes floor is unknown; do not substitute a retrospectively tuned model as a claimed optimum.

### 2.5 Established, assumed, proposed and speculative

Established prior capabilities are local policy-gradient training, difference-reward credit, selective component replacement and dynamic sparsity. Assumptions specific to this plan are available label feedback, a computable external loss, useful drift at laptop-scale, and enough signal to learn a newborn’s output connection before elimination. The proposed research contribution is an experimentally tested allocation mechanism and its boundary conditions, not those existing ingredients. Fully local pricing, negotiated coalitions, recurrence, open-ended birth, delayed survival objectives and autonomous predictive hierarchies remain speculative extensions.

## 3. Research findings and novelty assessment

### 3.1 Closest prior-art matrix

“Inspected” below describes this investigation, not a reproduction run. Bibliography entries identify stable sources and access limitations.

| Prior work | Problem and mechanism | Evidence, conditions and cost | Availability / inspected material | Exact overlap and consequence |
|---|---|---|---|---|
| Neural Bucket Brigade, 1989 [R01] | Local credit allocation through weight-substance payments in fixed-topology feedforward/recurrent networks | Simple demonstrations; not modern large-scale evidence | Author-hosted article, algorithm and experiments; old HTML omits figures | Neural economics and local reward propagation substantially predate this project |
| Hayek / economy of agents, 1999 [R02] | Market organization of computational agents | Blocks World demonstrations with task-specific structure; modern competitiveness unknown | Publisher abstract and metadata; full article not inspected | Economy/survival framing is not a contribution by itself |
| SANE, 1996 [R03] | Evolve populations of neurons assembled into networks | Historical control-task comparisons; many network evaluations | Author publication page and abstract; not source-code reproduced | Population-level cooperation, specialization and neuron selection already exist |
| Coagents, 2011–2023 [R04–R06] | Units execute local stochastic policies and learn from reinforcement | Theory under stated assumptions; later scaling work identifies variance as a major obstacle | Primary abstracts; 2023 PDF including theorem discussion and rendered pages | “Each neuron is an RL agent” and no hidden backprop are established |
| Weight Maximization, 2020/AAAI 2022 [R07, R08] | Hidden agents improve an outgoing-weight-based private objective | Multiplexer and three control tasks; not an optimal task solver; binary and continuous architectures differ | Full paper, experimental appendix, official NumPy source | Closest conceptual and executable baseline; self-interest and approximate goal alignment already studied |
| Difference Rewards Policy Gradients [R09] | Counterfactual credit for decentralized policies | Known-reward and learned-reward settings; partial-observability caveats are explicit | Full revised article, baseline argument and limitations | Difference credit is not new; do not extend one-step guarantees to arbitrary recurrent histories |
| Neuron Shapley, 2020 [R10] | Attribute network behavior to neurons across coalitions | Attribution, not a resource-economy training result | Primary abstract | Interventional neuronal importance is established; leave-one-out is not Shapley attribution |
| UPGD / UPS, 2023–2024 [R11, R12] | Protect useful parameters and perturb less useful ones | Streaming nonstationary supervised tasks and an RL extension | Full methods and official repository optimizer listing | Utility-driven local search and selective plasticity are strong competing explanations |
| Continual Backpropagation, 2024 [R13, R14] | Renew low-contribution mature units during ordinary learning | Long continual-learning experiments; results depend on task and regularization | Article and official `GnT` source | Birth/death with usefulness traces strongly overlaps standard generate-and-test |
| ReDo, 2023 [R15] | Recycle dormant units | Deep-RL evidence; use on classification must be labeled a transfer | Primary publication page | Activity-based recycling is a necessary simpler comparator |
| Attribution-based utility / GXD, May 2026 [R16] | Rank reset candidates using activation displacement and target-logit sensitivity | Preprint experiments in continual networks; independent verification not established here | Full method and reset discussion; executable official code not verified | “Counterfactual-aware utility for neuron renewal” is already an active line |
| Shapley Neuron Values, May 2026 [R17] | Importance-based neuron protection in continual learning | Different stability-focused setting; large published experiments are not local head-to-head results | HTML paper; no implementation reproduction | Further narrows neuronal-credit novelty, but not a direct substitute for online drift adaptation |
| RigL / Deep Rewiring [R18, R19] | Learn sparse connectivity with constrained capacity | Sparse supervised training; sparse arithmetic speedups depend on implementation | Primary publication pages | Changing topology is established; dense masking alone is not compute saving |
| Dynamic Weighted Majority [R20] | Add, weight and remove experts under drift | Established ensemble alternative; experts are larger than individual neurons | Primary JMLR abstract | Competitive adaptive populations do not require a neuronal economy |
| Market-based Architectures, 2025 [R21] | Proposes broader market organization for learning | Blue Sky/conceptual work, not a validated stronger empirical baseline | Full HTML | Current economic framing is not empty prior-art territory |
| Selective Weight Reinitialization, 2025 [R28] | Reset low-utility individual weights | Especially relevant to small networks and layer normalization | Full algorithm and experiments | Unit-level death can be a worse granularity than edge/weight renewal |

### 3.2 Reconciliation of apparently conflicting evidence

The literature does not show either that ordinary backpropagation universally fails online or that neural competition universally matches it. Weight Maximization compares hidden training rules in a specified actor architecture, and its paper acknowledges better task-solving choices. Coagent scaling work does not claim state-of-the-art performance. Continual-learning studies use much longer exposure histories than a short toy experiment; L2 regularization and perturbation succeed in some of those settings. SWR shows that architecture and replacement granularity matter. Therefore, comparisons must match the task, duration, information, inference convention and resource budget, and must include strong simple optimization choices. [R06, R07, R13, R28]

The 2026 attribution work is relevant to baseline selection but remains a preprint. Its reset-cost proxy is not a measurement of future plasticity benefit. This plan measures both immediate damage and subsequent adaptation instead of assuming that either predicts the other perfectly. [R16]

### 3.3 Explicit answers on novelty and reuse

1. **Already exists:** neuronal private objectives, local stochastic updates, market-like credit, evolutionary neuron populations, utility traces, selective renewal and adaptive topology.
2. **Reuse:** the official Weight Maximization implementation for the closest replication; official CBP utility/reset logic; UPGD’s optimizer; library online baselines; standard numerical and experimental infrastructure. Use clean-room equations where a license or implementation is unavailable.
3. **Potential difference:** the specific combination of exact conditional contribution, an event-triggered resource account and private constrained topology proposals, studied with complete cost accounting. This is only plausible differentiation; it is not demonstrated methodological novelty.
4. **Possible value:** identify when event-triggered, contribution-informed resource allocation improves adaptation and when it fails, including a clear negative result about the economic interpretation.
5. **Isolation:** hold credit, learning, reset operator, average turnover, topology opportunities and information access fixed in successive factorial controls. Require equality against the reserve/utility-trace alias.
6. **Invalidating evidence:** an existing method with the same state and transition equations; the alias control explaining the entire effect; utility-plus-reset baselines matching the method; or gains vanishing under compute/information matching. Update the novelty ledger rather than rename the mechanism.

A new implementation, acronym or benchmark result is not automatically a new method. If the only result is a faithful replication or an engineering speedup, label it that way.

### 3.4 Selected direction versus alternatives

| Direction | Value and risk | Decision |
|---|---|---|
| Raw firing, degree or weight-magnitude payment | Closest to unqualified influence-seeking, but easy to game or confound with scale | Keep as an adversarial diagnostic, not the main architecture |
| Entirely local market with conserved payments | Preserves stronger decentralization; close to historical bucket-brigade work; much harder credit problem | Defer until a clearly stated local task and budget-consistent mechanism exist |
| Counterfactual private utility with controlled turnover and topology | Computable, inspectable and falsifiable; global evaluator is a major concession | **Primary research direction** |
| Backpropagation with exactly the same controller | Simpler path to practical value | Required comparator and evidence-triggered pivot |
| Deep recurrent economic society | Potentially expresses memory and hierarchical dependencies; expensive counterfactuals and unclear credit | Small dependency check only; no broad performance claim |

## 4. Precise mechanism and implementation specification

All equations in this section specify the proposed project, except where explicitly attributed. They are not claims that the system has been trained successfully.

### 4.1 Network and ownership

For one observation `x ∈ R^d`, with `N` capacity slots and `C` classes:

\[
 a_i=(m_i\odot w_i)^\top x+b_i,\quad
 p_i=\sigma(a_i),\quad h_i=\mathbf 1[u_i<p_i],\quad u_i\sim U[0,1),
\]
\[
 z=c+\sum_{i=1}^{N}v_i h_i,\quad q=\operatorname{softmax}(z),\quad
 L(z,y)=\operatorname{logsumexp}(z)-z_y.
\]

Store `W[N,d]`, `M[N,d]` Boolean, `b[N]`, `V[C,N]`, `c[C]`. Unit `i` controls `W[i]`, `b[i]` and proposes changes to `M[i]`. Output consumer units control `V` and `c`; a hidden unit may not directly inflate its outgoing weights. No shared hidden parameters. The mask is enforced after every update. Biases are not charged as incoming edges in the primary contract; their storage is still counted.

Use numerically stable sigmoid, log-softmax and log-sum-exp. Do not clamp activation probabilities while retaining an unclamped score derivative. The core uses an ordinary sigmoid; record saturation frequencies. Any exploration-floor mixture requires its **correct** score derivative, a separate configuration and an amendment before confirmation.

Initialize allowed incoming weights as independent zero-mean Gaussian draws with standard deviation `1/sqrt(k_i)`, hidden biases to zero, and readout weights as independent zero-mean Gaussian draws with standard deviation `1/sqrt(N)`. Masked coordinates are exactly zero. These are project initialization conventions, not a claimed optimum. The matched stochastic controls share initialization and random uniforms. ReLU baselines use activation-appropriate initialization and are not forced to use the candidate’s initialization.

### 4.2 Exact two-action contribution

Freeze the entire pre-update network. For every unit, hold the other hidden activations fixed and calculate

\[
 z_i^0=z-v_i h_i,\quad z_i^1=z_i^0+v_i,
\]
\[
 \delta_i=L(z_i^0,y)-L(z_i^1,y).
\]

A positive `δ_i` means that firing improves the task reward, conditional on the other sampled hidden units. The realized zero-ablation contribution is `d_i=h_i δ_i`. The conditional expected contribution is `p_i δ_i`.

For a block, compute counterfactual logits in neuron chunks. Complexity is `O(BNC)` after the forward pass; do **not** rerun the whole network `N` times for this one-layer case. Calculate both branches even when the sampled unit is silent. Accumulate block means and gradients without storing a full trajectory of counterfactual logits.

The primary scalar credit supplied to the unit is `δ_i`; the unit combines it with its own activation probability and local inputs. This is local **parameter updating**, not local **credit computation**. The evaluator uses the label, readout and all other unit outputs. Label queries are free only in the information sense: their computation and reuse must still be counted.

### 4.3 Private payoff, action learning and readout learning

Let `λ_f ≥ 0` be the charge per firing, `λ_k ≥ 0` the full-input-degree charge, and `λ_0 ≥ 0` the per-slot maintenance charge. Units are **nats of task utility per observation**; these are artificial prices.

Conditional expected net operating payoff is

\[
 \pi_i=p_i(\delta_i-\lambda_f)-\lambda_0-\lambda_k k_i/d,
 \qquad k_i=\sum_jm_{ij}.
\]

The primary hidden update (`rb`: conditional averaging over a unit’s two actions) is

\[
 g_i=\frac1B\sum_{r=1}^{B}p_{ri}(1-p_{ri})(\delta_{ri}-\lambda_f)x_r,
\]
\[
 w_i\leftarrow w_i+\eta_h(m_i\odot g_i)-\eta_h\lambda_w w_i,
\quad
 b_i\leftarrow b_i+\eta_h\frac1B\sum_r p_{ri}(1-p_{ri})(\delta_{ri}-\lambda_f).
\]

`λ_w` is an optional conventional weight-decay hyperparameter and must be available to controls. The fixed slot and degree charges do not appear in the firing gradient because they do not depend on the sampled action. They affect reserves and structural choices.

The diagnostic sampled difference-reward update (`dr`) replaces the scalar coefficient with

\[
 (h_{ri}\delta_{ri}-\lambda_f h_{ri})(h_{ri}-p_{ri}).
\]

Both are known policy-gradient constructions, not the proposed novelty. Conditional averaging removes this unit’s sampling noise, not that of all other units or the data. A difference baseline does not universally minimize variance; measure it rather than assume it.

The output layer updates locally from its input vector `h` and its own supervised error:

\[
 V\leftarrow V-\eta_o\frac1B\sum_r(q_r-e_{y_r})h_r^\top,
\quad c\leftarrow c-\eta_o\frac1B\sum_r(q_r-e_{y_r}).
\]

All right-hand sides use the same pre-update snapshot. Output updates do not propagate derivatives into the hidden layer. SGD is the reference implementation. A tuned per-coordinate Adam implementation may be used consistently for manual gradients, with resettable moments and coordinate ages; do not silently turn the hidden update into backpropagation.

### 4.4 What the fixed-network mathematics does and does not guarantee

For a fixed one-layer architecture, readout and sample, let `R=-L-λ_f Σ_i h_i` and let `P_{θ_i}(h_i|x)` denote unit `i`’s Bernoulli policy with incoming parameters `θ_i=(w_i,b_i)`; this policy is distinct from the net payoff `π_i`. The counterfactual baseline `R(h_i=0,h_-i)` is independent of the unit’s sampled action. Consequently,

\[
 E[(R-R_i^0)\nabla\log P_{\theta_i}(h_i\mid x)]
 =E[R\nabla\log P_{\theta_i}(h_i\mid x)].
\]

Averaging over `h_i` gives `p_i(1-p_i)[R_i^1-R_i^0]x`. Including the unit’s firing charge gives the coefficient above. Other units’ firing charges are constant with respect to this unit in the one-layer architecture. Thus the fixed-topology hidden update follows the gradient of expected reward minus firing charges in expectation. This is the standard baseline/conditional-expectation reasoning, not a convergence theorem for the complete project. [R04, R09]

Changing the readout, resetting identities, mutating edges, reusing a finite audit buffer and making survival decisions invalidate any inference of monotonic long-run improvement from this identity alone. A small learning rate does not establish global convergence. With deeper networks, a unit can change other units’ firing costs; the one-layer cost argument must be revised as specified in Section 4.10.

### 4.5 Reserves, accounting and bankruptcy

Each unit has reserve `E_i` in accumulated utility credits, age in observations, generation ID and running activity statistics. For block size `B_b`:

\[
 \rho_b=\exp(-B_b/T_E),\qquad
 E_i^{b+1}=\rho_b E_i^b+B_b\bar\pi_i^b-c_{\rm proposal}n_i^b.
\]

`T_E` is a decay horizon measured in observations. `n_i^b` counts structural proposals actually evaluated for that unit; successful and unsuccessful proposals are charged. The primary settles expected contribution `p_i δ_i`, not a fictitious sum of physically received energy. A realized-payment variant using `h_i δ_i` is an attribution/noise ablation.

A unit is bankrupt when its reserve is nonpositive **and** its age is at least `T_mature`. Replacement is subject to a predeclared service quota: accumulate `N*B_b/T_turnover_min` replacement tokens; spend whole tokens on eligible units, sorted by reserve then stable unit ID. Tokens cannot authorize removal of a solvent unit. Cap stored tokens at `N` to avoid an unbounded delayed purge. This quota is a population-management constraint, not an emergent market.

On replacement, retire the old generation ID, release its edges, and initialize a new uniformly sampled distinct input mask with **the same degree as the retired slot** and fresh Gaussian weights. This preserves the total active-edge count during rebirth even after private topology changes; resetting every newborn to the original degree could exceed the global capacity. Zero its outgoing weights and hidden bias, reset its optimizer state, utility traces and age, and give endowment `E_birth`. No other slot is resized to accommodate the newborn. Record endowment issuance. A newborn with zero outgoing weights initially receives no task credit; it must remain protected long enough for the readout to learn to use it. Derive `T_mature` and `E_birth` from development measurements of that bootstrap time and maintenance costs. Do not hide continual newborn failure by increasing subsidies after seeing confirmation results.

**Primary reset operator:** zero outgoing weights without output-bias compensation. This matches the zero-ablation credit definition. A compensated reset is a separate shared control: add `v_i r_i` to the output bias using the pre-reset running mean `r_i`, then zero `v_i`. For that control, measure reset utility against `z-v_i(h_i-r_i)`, not against simple zero firing. Do not mix a compensated reset with an uncompensated utility interpretation.

The default immediately refills a slot; the live capacity remains `N`. Record “retired identities,” not “a shrinking neural population.” Persistent dormancy and population expansion require a new mechanism and are out of scope.

### 4.6 Essential equivalence and budget warnings

For constant block size, write `u_b=B*barπ_b-c_proposal*n_b`. Then, between resets,

\[
 E_t=\rho^tE_0+\sum_{j=0}^{t-1}\rho^{t-1-j}u_j.
\]

Rescaling this expression gives an exponential utility trace plus an initialization term. Implement an `equivalent_trace` controller from this closed-form/state-space identity. It must make the same choices, including resets, as the account-based implementation. For variable final-block sizes, use the product of the actual `ρ_b` values, not a constant-decay shortcut.

**Therefore an advantage over the identical utility-trace formulation is impossible except through an implementation or measurement difference.** The scientific question is whether this particular allocation policy is useful compared with other policies—not whether calling a number “energy” adds capability.

Marginal contributions are not budget-balanced. With two active redundant OR components, team benefit can be one while both zero-ablation contributions are zero. With complementary AND components, team benefit can be one while the sum is two. Do not normalize contributions to force a budget: a normalization depending on the sampled action can change the learning objective and estimator.

Maintain a trusted **external credit issuer** and report total issuance, charges, reserve decay and endowments. This is an open accounting mechanism, not a closed conserved economy. There is no theorem of truthful pricing, coalition-proofness or strategic incentive compatibility.

### 4.7 Private incoming-connection changes

This module is required before claiming that the full primary mechanism, rather than fixed-topology turnover, has been tested.

Maintain `1 ≤ k_i ≤ d` and `sum_i k_i ≤ K_max`. Start with `k_0` uniformly selected distinct input coordinates per unit and `K_max ≥ N*k_0`. The main matched sparse experiment fixes `K_max` before confirmation; dense and matched-capacity controls receive the same allowable budget. Dormant masked weights are exactly zero and have zero optimizer moments.

Maintain a FIFO **training** audit buffer containing only observations whose pre-update predictions were already committed and whose labels were revealed. All methods may use the same buffer. It is not the protected reporting set.

At a structural opportunity, select one unit uniformly from solvent, mature units that can pay `c_proposal`. Select uniformly among legal move types: add an inactive input, drop an active input if `k_i>1`, or swap an active/inactive pair. Choose affected coordinates uniformly. New connection weights are independent draws from the original incoming initialization distribution at the proposed degree. Unchanged weights are not silently renormalized. A proposal exceeding `K_max` is illegal and is not evaluated or charged.

Freeze the current post-learning, post-replacement model for this proposal. Sample the other units’ activations on an audit minibatch using a separate random stream; evaluate `δ_i` for both current and proposed incoming policies. The readout and other units are fixed, so `δ_i` itself is unchanged. Compute

\[
 \Delta G_i=\frac1A\sum_{r=1}^{A}(p'_{ri}-p_{ri})(\delta_{ri}-\lambda_f)
             -\lambda_k(k'_i-k_i)/d.
\]

Accept only if `ΔG_i > 0` in float64 accumulation. A near-zero difference within the numerical comparison tolerance is a rejection. The proposal fee is charged regardless of acceptance and therefore is not subtracted a second time from the acceptance difference. Newly added/removed coordinate optimizer state is reset; existing coordinate state is preserved. Record proposed move, estimated gain, cost and decision.

This is a **myopic private best-response search on a training sample**, not a guarantee of improvement on future data. Its fixed-snapshot payoff improvement coincides with the corresponding conditional task-plus-private-cost improvement in the one-layer case. Sequentially apply accepted proposals and recompute the next proposal’s snapshot. Never accept a set of interacting proposals using stale simultaneous scores.

Use a proposal token accumulator with rate in proposals per observation. The initial diagnostic rate is one proposal per update block, not one per neuron; expand only when a pilot shows proposal benefit exceeds its measured cost. Freeze the chosen rate, audit-buffer size and audit-minibatch size before confirmation. Compare against no rewiring, random accepted rewiring with equal rates, and the **same evaluator attached to non-economic learning**.

### 4.8 Update-order pseudocode

```text
for block in stream:
    X, event_ids = stream.observations_only()
    h, p, logits, snapshot_id = model.predict_with_cache(X, rng_activation)
    q_report = (1-epsilon)*softmax(logits) + epsilon/classes
    ledger.commit_predictions(event_ids, q_report, snapshot_id)
    y = stream.reveal_labels_after_commit(event_ids)

    # All quantities below refer to the same pre-update model.
    delta = credit.two_action_differences(X, h, logits, y, frozen_model)
    hidden_gradient = local_gradient(X, p, h, delta, rule="rb")
    output_gradient = local_output_gradient(h, logits, y)
    operating_payoff = mean(p*(delta-lambda_f), over_examples)
                       - lambda_0 - lambda_k*degree/d
    cache_preupdate_reset_statistics()

    apply_hidden_and_output_updates_simultaneously()
    account.decay_and_settle(operating_payoff, block_size)
    account.advance_ages(block_size)
    retire_and_replace_eligible_identities_within_token_quota()
    audit_buffer.append_only_revealed_training_examples(X, y)
    evaluate_sequential_private_proposals_with_fresh_snapshots()
    assert_masks_capacity_finiteness_and_ledger_invariants()
    commit_event_transaction_and_optional_checkpoint()
```

The ledger must distinguish counterfactual credit evaluation, ordinary prediction, output learning, hidden learning, accounting, replacement and structural audit computation. The evaluator never changes parameters or random-stream state used by ordinary prediction except through its declared separate stream.

### 4.9 Minimum complete and cheapest informative experiment

The smallest **complete one-layer** experiment uses 32 stochastic hidden units, a small sparse changing teacher, exact two-action credit, nonzero maintenance charges, surviving and replaced identities, and legal edge proposals. This size is for finding numerical and causal errors, not proving competitive scaling. First verify all four essential behaviors occur: useful learning, solvent persistence, justified turnover, and accepted topology changes.

The smallest **discriminating comparison** adds the identical trace alias, conditional coagents without turnover, counterfactual ranking with matched replacement rate, and a backpropagation model with the same controller. If only the name “economy” changes, outcomes must be identical. If every apparent improvement is explained by one of these simpler mechanisms, record that before expanding the benchmark.

### 4.10 Mandatory dependency test: two hidden layers

Implement a DAG with first-layer Bernoulli units, second-layer Bernoulli units and a linear readout. Begin with `2+2` units for enumeration; then use `16+16` or `64+32` for a learnable dependency task, selected by calibration. The latter also supports the closest Weight Maximization replication.

Intervening on a first-layer unit must **recompute every affected descendant** using common, independently generated uniform random numbers. Keeping the second layer frozen computes a different intervention and can produce zero credit for an essential parent. Compute reward for `do(h_i=0)` and `do(h_i=1)` by rerunning the descendant graph. Cost is now up to `O(number_of_audited_units × forward_cost)`; the shallow shortcut is invalid.

For the cost-sensitive gradient, include the change in **all descendant firing charges** in the branch reward, plus the parent’s charge. Holding other agents’ costs constant is no longer valid. Direct degree and maintenance costs remain fixed during a firing intervention. For reserve settlement, record separately the task component, downstream-cost externality and own costs so that there is no double counting.

The one-step DAG baseline is independent of the parent’s action when evaluated with independent exogenous randomness and a fixed intervention. Test that identity by exhaustive enumeration. Do not claim the same argument for counterfactuals that condition on action-dependent future histories. The difference-reward literature explicitly warns about such partial-observability problems. [R09]

The competitive one-layer claim may proceed after a correct but slower dependency implementation. If the multi-layer implementation cannot learn a task that its controls learn, restrict the result to shallow networks. Recurrence and emergent memory remain untested.

## 5. Baselines, implementation fidelity and fair tuning

### 5.1 Comparison hierarchy

There is no verified universally strongest learner for the as-yet ungenerated evaluation population. “Strongest relevant” means the best development-selected, faithfully implemented members of the families below, not a claim that one paper is universal state of the art. All are potentially locally executable at the specified small-model scale; their **full published benchmark suites** are not prerequisites to a local comparison. A faithful algorithm transferred to another dataset is not a reproduction of its published result.

| Method ID | Implementation and configuration contract | Role and reproduction requirement |
|---|---|---|
| `linear_online` | scikit-learn `SGDClassifier(loss="log_loss")`, causal standardization, full known class inventory at initialization; tune learning rate/schedule and regularization | Strong simple baseline. Compare its first updates with a direct linear implementation; save installed version and actual estimator parameters [R26] |
| `arf_online` | River `forest.ARFClassifier`; tune ensemble size, feature sampling and leaf configuration; retain adaptive drift handling | Strong nonlinear tabular comparator. Predict the whole block before calling `learn_one`; missing initial probabilities become uniform. Account for trees, background trees and per-tree memory limits [R27] |
| `bp_dense` | One-layer ReLU and sigmoid networks with supervised backpropagation; development-select SGD with momentum or AdamW, weight decay, and normalization where appropriate | Required practical neural baseline. Include dense and parameter-matched widths. Do not deny it useful continuous activations or LayerNorm merely because those differ from the candidate |
| `bp_stochastic_matched` | Same Bernoulli forward network, masks, capacity, random uniforms and readout as the candidate; explicitly labeled straight-through gradient surrogate | Architectural diagnostic, not the strongest practical baseline or an unbiased gradient estimator |
| `coagent_shared` | Bernoulli hidden units use shared `R=-L-λ_f Σh`; local coefficient `(R-b_previous)(h_i-p_i)`; output learner as in §4 | Local-learning baseline. `b_previous` is a scalar EMA from **previous** blocks; alternatively a development-trained action-independent input critic. Never subtract an action-dependent same-sample reward mean without correcting the resulting estimator [R04–R06] |
| `coagent_rb_fixed` | Exactly §4 conditional gradient, fixed masks and immortal identities; allow the same ordinary regularization and replay budget | Indispensable control. A win against noisy REINFORCE alone is not evidence for economics |
| `weightmax_original` / `weightmax_port` | Original NumPy two-layer multiplexer implementation first; separately labeled adapter for common supervised tasks | Closest self-interested-neuron precedent. Preserve the source update order in the reproduction and disclose every transfer change [R07, R08] |
| `cbp` | Continual Backpropagation: utility EMA, maturity gate, low-utility selection, fractional replacement accumulator, correct state reset; development-select documented contribution variants | Required modern unit-renewal baseline. Check the official implementation and the reset-mean ordering issue below [R13, R14] |
| `redo` | Score a unit by mean absolute activation divided by the layer mean; reset sufficiently dormant mature units at a tuned interval | Activity-based renewal control. This is a supervised-task transfer, not a reproduction of Atari performance [R15] |
| `upgd_first_order` | First-order utility estimate from `-parameter × loss_gradient`, temporal averaging, utility-dependent noisy gradient attenuation and weight decay | Required gradient-based utility alternative. Use the inspected updated official implementation as the versioned definition; do not silently substitute an earlier paper variant [R11, R12] |
| `swr` | Selective weight reinitialization, using low absolute-weight or absolute weight-times-gradient utility; reset selected coordinates from their initialization distribution | Required recent comparator, especially at small widths and with LayerNorm. Start with the paper’s SGD setting; an Adam transfer must reset coordinate state [R28] |
| `gxd_reset` | Attribution-based neuron ranking using the mean-reference target-logit score below, with a matched unit-replacement controller | Relevant 2026 preprint comparator, not independently verified state of the art. Clean-room local adaptation; no verified official code is assumed [R16] |
| `rigl_local` | Fixed connection budget; remove small-magnitude active weights and add inactive coordinates with large potential loss gradients | Required for an edge-allocation advantage claim. Original finite-training scheduling and a disclosed continual adaptation are separate configurations [R18] |

Run `linear_online`, `arf_online`, `bp_dense`, `coagent_rb_fixed`, `weightmax_port`, `cbp`, `upgd_first_order`, `swr`, and `gxd_reset` before a broad competitive claim on tabular tasks. `redo` and `rigl_local` become mandatory when attributing gains specifically to dormancy selection or topology. Shared-reward coagents, stochastic backpropagation, random features and algebraic aliases are diagnostic controls, not substitutes for this competitive set.

If a required baseline cannot be validated, the comparison is incomplete **for the claim that requires it**. Continue useful mechanism tests, identify the missing comparator, and narrow the claim. Do not quietly omit the baseline or compare against a deliberately degraded port.

### 5.2 Closest-prior-art replication: Weight Maximization

The inspected official multiplexer configuration uses 20 input bits, hidden widths `64,32`, block size one, Bernoulli temperatures one, actor trace parameter zero, L2 weight-maximization utility, no weight regularization, and layer learning rates `0.01,0.001,0.0001`. The reference configuration runs 800,000 episodes per run and ten runs; these are **the source’s reproduction conditions**, not an arbitrary workload cap for this project. The critic learning rate is zero. [R08]

Generate the task directly: draw four address bits and sixteen data bits independently and uniformly; interpret the first four bits as the address `0…15` in a fixed documented bit order; the label is the addressed data bit. A sampled correct action earns `+1`, an incorrect action `-1`. Retain the original softmax output policy and reinforcement update for the reference replication. This requires no obsolete control-environment dependency.

Inspect and pin `weightmax.py`, `main.py`, `task.py` or the actual environment module located in the pinned tree, and `config/config_mp.ini`. The official README command is `python main.py -c config_mp.ini`; confirm its configuration-path resolution before use. Replace deprecated NumPy scalar aliases only through a recorded compatibility patch with fixed-input update parity. Do not invent an unobserved release tag.

A significant fidelity issue: the inspected source passes a first-order outgoing-weight expression using **updated outgoing weights and a stored raw update quantity** upstream. That is not automatically identical to the literal change in squared outgoing weights or an optimizer’s realized parameter displacement. Maintain `source_faithful` and, if useful, `literal_norm_change` variants separately. Before transfer, save a two-layer, two-update numerical fixture showing the exact private rewards and parameter changes. [R07, R08]

First reproduce correct task generation and positive learning on a pilot horizon. Calibrate the full reference workload locally. Reproduce the published experimental conditions when feasible; compare learning-curve shape and uncertainty, not a demand for one rounded published score. If the full reference schedule is unaffordable, report a partial replication and do not label the common-task adapter a validated reproduction. In the adapter, changing the output to supervised cross-entropy, omitting traces or changing layer count must be named in its manifest.

### 5.3 Reset, utility and optimizer details that can alter the result

**CBP.** The local adapter must reproduce its selected utility, bias correction, maturity, reset fraction and outgoing-weight handling. The currently inspected `gnt.py` appears to clear an activation-mean statistic before a later compensation step reads it. This is a source-ordering concern, not a claim about every historical experiment. Test a nonzero-mean feature explicitly. Keep an upstream-faithful fixture and a corrected implementation that saves the pre-reset mean, with the patch disclosed. Include both if the change materially changes results. [R14]

**GXD.** For the one-layer readout, use

\[
 s_i=\operatorname{EMA}\left[\operatorname{mean}_{r}|(h_{ri}-\bar h_i)V_{y_r,i}|\right].
\]

Here the derivative is of the target-class **logit**, not cross-entropy. The reference mean and replacement operation must agree: preserve the mean contribution through bias compensation when removing a unit. This baseline helps distinguish exact loss attribution from a cheaper attribution surrogate. [R16]

**UPGD.** Port the first-order update from the pinned updated source, including its factor of two, bias correction, utility normalization and noise placement; record the mathematical update in `baseline_specs.md`. Verify one-step parity with explicit gradients and pre-generated noise. A zero utility normalizer and a CPU scalar accidentally combined with MPS tensors are blocking numerical/device cases. A device-safe scalar is a compatibility change; a new zero-denominator fallback is an explicit algorithmic edge-case convention, tested against ordinary SGD and reported. Higher-order utility estimation is not silently represented by this first-order baseline. [R12]

**SWR.** Use the official implementation and a scalar fixture for threshold selection, coordinate resampling and update order. For a fractional requested reset count, randomized rounding uses the fractional part as a Bernoulli probability, never a number greater than one. The paper’s prose and executable behavior must resolve any pseudocode ambiguity. Record whether utility uses pre-update weights and gradients. [R28]

**RigL.** Dormant-edge growth requires the *potential* dense gradient, such as an outer product of backpropagated preactivation error and inputs, before applying the mask. Reading the gradient through a hard-zero mask gives zero on dormant edges and is not RigL. Charge dense gradient calculation even if inference is sparse. A continual, nondecaying replacement schedule is a disclosed adaptation, not a faithful reproduction of a finite-horizon schedule. [R18]

### 5.4 Shared controllers and compulsory matched controls

Factor the system into `learner`, `credit`, `account`, `selector`, `resetter` and `topology` components. This is scientific instrumentation, not a public framework project. The minimum attribution set is:

| Contrast | What must remain equal | What it can establish |
|---|---|---|
| `economy_rb` versus `trace_alias` | Every update, initialization, fee, age, threshold and random draw | Identical outcomes are required; reserves do not add magic beyond their recurrence |
| Full method versus `coagent_rb_fixed` | Prediction, gradient estimator, initial graph, optimizer, data | Total effect of turnover/topology, still confounded between those components |
| Full method versus no-turnover, no-topology, and neither | All other settings; report actual edge and reset histories | Necessity and interaction of the two mechanisms |
| Bankruptcy versus exact-utility ranking at matched reset count | Same credit, age eligibility, reset operation and per-block count | Value of the survival gate rather than merely useful replacement |
| Bankruptcy versus random selection at matched reset count | Same opportunity times, eligible population, count and newborn treatment | Value of the utility ranking; rates must not explain the result |
| Candidate controller on local gradients versus on backpropagation | Same labels, audit buffer, prices, intervention definition and capacity | Whether local self-interested learning is necessary for the gain |
| Accepted private proposals versus random proposals and shared proposal evaluator | Same proposal budget, audit data, graph capacity and charged computation | Contribution of private selection versus generic structure search |
| Expected versus realized contribution settlement | Same hidden estimator, price scale and accounting time constants | Value of lower-noise survival decisions, not a new policy gradient |
| Zero-ablation versus mean-preserving removal utility | Match the reset reference in each arm | Sensitivity to the definition of an absent neuron |

A reset-count-matched control may use the candidate’s realized counts only as a **diagnostic paired intervention**, not as a deployable baseline with privileged future information. At block `b`, pass only the already determined count for `b`, never future counts or selected identities. Include a deployable fixed-rate comparator tuned on development data.

For continuous backpropagation units, exact reset-utility evaluation compares original logits with logits after the specified removal reference; there is no Bernoulli two-action identity to assume. Charge this evaluator identically and label the controller transfer.

### 5.5 Tuning and resource matching

Separate two questions:

1. **Mechanism comparison:** same architecture, stream, initial masks, observation count, label availability, optimizer options and management opportunities. Count every counterfactual and audit replay operation even when nominal parameter counts match.
2. **Practical comparison:** allow each method its best validated local device, architecture and regularizer. Compare predictive risk against measured end-to-end adaptation time, inference latency and peak memory. A dense baseline may use more parameters if it still fits the same measured resource envelope. More capacity is not free merely because both models fit in RAM.

Begin development sweeps at learning rates `{10^-4,10^-3,10^-2}` per independently updated layer, weight decay `{0,10^-5,10^-3}`, and activation-appropriate initializations. These are **exploratory starting points**, not a fixed search boundary or a required winning range. Expand a boundary winner geometrically; refine between promising stable values. Use source defaults as additional candidates. Examine SGD/momentum and AdamW for backpropagation; manual-gradient SGD and resettable Adam for local learners. A first-order method must not receive second-order information through its optimizer.

Initially use `N=32` for correctness, `N=256` for throughput/plasticity pilots, and include `N=1000` for the intended-scale comparison when calibrated repetitions are feasible. Add intermediate or larger widths only to resolve an identified capacity or scaling uncertainty. Match initial active connection counts for mechanism contrasts; also retain the best dense baseline. For forests, start from the installed library defaults and vary tree count and tree size according to the same measured memory/time envelope; do not confuse a per-tree memory allowance with a forest-wide bound.

Tune reset rates, utility decay and maturity in **observations**, not unspecified optimizer steps. Source defaults expressed in updates must be converted using the actual block size and preserved in a faithful-source arm. Each baseline receives comparable development opportunity in measured compute, plus a source-default reproduction check. If one family is much cheaper, allow it more inexpensive settings; also report results under equal trial counts to expose tuning sensitivity. Record tuning cost separately and amortized over realistic reuse horizons, without inventing an application lifetime.

Early stopping is allowed during development only when an independently scored shorter run reliably predicts the longer-run ranking. Plasticity methods can look identical early, so a short-run elimination rule must first be validated on complete development trajectories. Do not discard baselines just because they initially learn slowly. Freeze the surviving configurations, all expansion rules and the baseline champion before protected evaluation. Any incomplete tuning leaves a corresponding limitation on competitiveness.

## 6. Data, task generation and confirmatory evaluation

### 6.1 Controlled synthetic population: primary causal evidence

Implement a versioned generator, not an informal description of “concept drift.” Use these initial defaults in development; freeze the realized generator specification before confirmation.

- `d=32`, binary classes, `J=8` teacher features, each with four distinct uniformly selected input coordinates. For feature `j`, draw normal coefficients on its support and normalize their Euclidean norm to one. Draw eight normal output coefficients and normalize their joint norm to one. Define `f_A(x)=2 Σ_j a_j tanh(w_j^T x)` with `x ~ N(0,I_d)` and `P(y=1|x)=sigmoid(f_A(x))`. Draw Bernoulli labels using a separate data RNG. Save teacher parameters only in evaluator-owned artifacts.
- Independently generate teachers A, B and C for each world. Primary families are stationary A; abrupt alternating A/B; gradual A/B; and recurrent A/B/C/A/B/C. For gradual drift, linearly interpolate **teacher logits**, `f_t=(1-s_t)f_A+s_t f_B`, with a triangular interpolation weight spanning each regime duration. Abrupt and recurrent families change at regime boundaries. Methods receive neither boundaries nor teacher identity.
- Evaluate six equal-duration regimes initially: stationary repeats A; abrupt uses A/B/A/B/A/B; gradual traverses A→B→A→B→A→B→A; recurrence uses A/B/C/A/B/C. Six regimes are the minimum design here to observe repeated transitions and a second visit to all three recurring teachers, not a claim of long-term plasticity. Add complete cycles if the development slope analysis requires longer exposure.
- A **world** consists of independently drawn teachers plus the input/label randomness for all four families. Use separate child RNG streams per family. Pair every method on the same realized examples. Reuse neither development teachers nor protected world seeds.

This task is intentionally modest and fully auditable. It can establish an advantage over a specified stochastic task population, not real-world intelligence. It is not a mechanism-matched oracle for the candidate: the teacher uses continuous features and noisy labels, whereas the candidate samples binary features. Also run a linear teacher, parity/complementarity cases, dense teachers and irrelevant-input expansion as boundary tests.

The evaluator knows the Bayes probability. Record expected Bayes log loss `-q*log(q)-(1-q)*log(1-q)` as a headroom reference. Prefer paired expected cross-entropy to the known probability as a **secondary low-noise diagnostic**, while the primary learner and primary prequential score use only sampled labels. Never let oracle probabilities, supports or Bayes errors influence model updates, confirmation selection or drift alarms.

### 6.2 Horizon, prices, reserves and maturity are calibrated—not guessed

During development, first train immortal fixed-topology versions with all economic charges zero. On independent pilot worlds, estimate `T_learn`, the observation count required for a strong local or backpropagation learner to reach its stable learning region after a change. Fit learning curves and compare successive doubled horizons using world-level uncertainty. A stable region means the additional reduction from another equally long segment is no longer resolved at the pilot’s precision; it is not proof of an asymptotic optimum.

Set candidate regime durations around `{0.5,2,8} × T_learn` for fast, intermediate and slow drift, rounded to whole blocks. Choose the intermediate condition for the primary claim unless development shows it cannot express any measurable change; report all three as predeclared boundary conditions if affordable. If no plateau is observed, use the longest validated horizon, report unresolved learning times, and do not make an indefinite-plasticity claim. Horizon selection must consider **all strong baselines**, not deliberately choose the first horizon where ordinary SGD deteriorates.

Calibrate a newborn bootstrap time `T_boot` by inserting a zero-outgoing-weight newborn into an otherwise learning network and measuring the observation delay until its output weight and credit become informative. Use held-out pilot events and report the distribution. A grace period selected from its upper quantile is a protection against immediate starvation, not evidence that the newborn is useful. Development-select maturity from `{T_boot,2T_boot,4T_boot}`; expand if most newborns remain unidentifiable. Exempting a newborn from death does not exempt it from accounting charges.

Define `u_ref` as the median, across informative pilot units, of the absolute block-averaged expected task contribution `p_i δ_i`. If this scale is zero or indistinguishable from numerical error, **do not normalize by an arbitrary tiny denominator**: stop economic tuning and resolve failure of learning/credit first. Begin positive price candidates with `λ_f, λ_0, λ_k` equal to `{0,0.1,1} × u_ref`, alongside source-independent zero-cost controls. Scale choices are tuning coordinates, not required effect sizes. Include a nonzero maintenance and edge charge in the complete mechanism; allow development to show that every such setting is dominated.

Choose reserve decay times from `{T_boot,T_learn,4T_learn}` and turnover-cap times from `{T_learn,4T_learn}` initially. Set a newborn endowment sufficient to cover its selected grace period under the measured adverse operating-payoff quantile. For constant block size and an estimated per-observation deficit `c>0`, a conservative endowment for `m` grace blocks is `E_birth ≥ B*c*Σ_{r=1}^m ρ^(-r)`; calculate it stably and verify against simulated zero-credit accounts. Report sensitivity to smaller and larger endowments. This is explicit external subsidy, not self-financed reproduction.

Start the proposal audit buffer at one measured learning window, bounded by the declared replay budget. Compare smaller windows and no reuse on development data. Set proposal fees from measured relative audit cost times a frozen credit-price conversion; also include zero-fee and equal-fee controls. That conversion is a design parameter, not an observed exchange rate between nats and joules. Fix all prices, window sizes, grace periods, proposal rates and endowments before confirmation. No automatic target-bankruptcy controller may quietly tune itself using protected scores.

### 6.3 Real sensor drift: supporting evidence with a licensing gate

UCI’s **Gas Sensor Array Drift Dataset** contains 128-dimensional measurements, six gas labels, 13,910 observations and ten batches. The source page records a corrected tenth batch. Use the official current archive, save its SHA-256, parse indexed LIBSVM features into 128 ordered columns, and validate labels and counts before training. The inspected page is internally inconsistent about licensing: legacy descriptive text restricts commercial use, whereas the current footer states CC BY 4.0. Do not silently choose the more permissive reading. [R22]

**Gate:** before using this arm, record whether the intended use satisfies the restrictive research-only wording or obtain authoritative clarification. Do not redistribute the dataset or claim commercial permission on this plan’s authority. If the conflict cannot be resolved for the intended use, block this arm and use the image-based fallback below, with an explicitly narrower transfer claim. This is an operational restriction, not a legal opinion.

Development uses batches 1–3 only, with rolling validation `1→2` and `1–2→3`. Freeze preprocessing statistics using batch 1; use `std=max(std,10^-6)` and document constant columns. In a fresh final run, learn on batches 1–3 under the selected protocol, then score batches 4–10 prequentially. Protected labels may enter training **only after their predictions have been committed**. They may not drive global hyperparameter selection, price changes or early stopping.

The ten batches provide coarse chronological structure; true timestamp order within a file has not been verified. Primary order is archived file order. A predeclared within-batch shuffle sensitivity tests order dependence and is labeled a synthetic rearrangement, not another independent history. Keep sensor measurements at their supplied feature level; do not infer raw signal access. Do not use per-batch oracle hyperparameters tuned on that batch’s future labels.

Report per-batch NLL, accuracy, classwise error and overall observation-weighted loss, with an equal-batch summary to expose dominance by large batches. This is **one observed history**. Multiple model seeds estimate conditional algorithmic randomness, not a population of independent real-world deployments.

### 6.4 Image-based transfer / licensed fallback

Fashion-MNIST’s official repository supplies 60,000 training and 10,000 test grayscale 28×28 images across ten classes and identifies an MIT license. Use its canonical files, validate byte counts and checksums, retain the license and provenance, and never train on official test labels. [R23]

Stratify the training set into 50,000 development-training images and 10,000 development-validation images using a fixed seed recorded before experiments. Develop on those partitions only. For final runs, use the full official training split after freezing the method; keep official test images and labels protected. Scale pixels by 255 without fitting future statistics.

Construct sequences of fixed pixel permutations and, in a separate condition, label permutations. Each regime uses a fresh training-order permutation, and previously seen transformations recur. Prediction on protected test probes occurs at fixed observation checkpoints with updates disabled. Probe labels never enter audit buffers, normalizers, selectors or the learner. Choose regime exposure using the same development calibration principle, not by examining official-test curves.

This is real image content under **artificial shift**, not evidence of natural sensor drift. Repeated images or different permutations are not independent new image populations. Report uncertainty conditional on the fixed image set and separately across transformation sequences. Use a strong dense network for arbitrary pixel permutations; for an unpermuted image claim, include a suitable small convolutional network rather than declare superiority over image learning generally.

### 6.5 Metrics and prediction semantics

**Primary synthetic metric:** per-observation prequential negative log likelihood, in **nats per observation**. Average observations within each family; weight the four primary families equally within each world; average optimization-seed replicates within a world; then average worlds. Preserve familywise results so an aggregate improvement cannot conceal a stationary or recurrent-task collapse.

For stochastic networks, the primary deployed predictor uses **one sampled hidden state per observation**. Its score is `-log q_y` for that realized prediction. This evaluates expected realized log loss, not `-log E[q_y]`; these differ. A Monte Carlo probability-mixture predictor is a separate deployment configuration with its sampling cost counted. Never score the mixture while charging one draw, or train on one distribution and describe results as another without disclosure.

For all methods, apply the same disclosed probability safety mixture at the scoring interface: `q_report=(1-ε)q+ε/C`, initially `ε=10^-6`. This prevents infinite log loss from a forest’s zero probability and bounds scores. Check that distributions sum to one before mixing. Candidate training and its counterfactuals use the ordinary stable, unmixed cross-entropy defined in §4; record that small training/reporting distinction. A sensitivity analysis on development data checks that clipping/mixture choices do not determine the winner; change the primary convention only before freezing.

Let `D_w=L_baseline,w-L_candidate,w`, so positive differences favor the candidate. Report absolute differences and confidence intervals in nats/observation. Relative reduction is supplementary and must name its denominator. Do not turn a percentage of a nearly zero excess-risk estimate into a headline.

Secondary measures: classification error; classwise error; Brier score and reliability diagnostics; loss by time since shift; cumulative excess loss above the Bayes reference on synthetic data; recurrence recovery; sampled/expected firing rate; active degree distribution; identity lifetime; replacement and proposal acceptance rates; duplicate/complementary feature behavior; and intervention-based functional specialization. Prediction latency, adaptation throughput, complete wall time, peak process memory, device allocation and artifact storage are mandatory cost guardrails. Correlation, degree, activation entropy and reserve inequality are descriptive—not intelligence metrics.

No claim of physical energy efficiency is allowed without an actual validated energy measurement. Artificial firing charges, elapsed time and processor utilization are not joules.

### 6.6 Independent units, uncertainty and sample-size calibration

The primary independent unit is a **synthetic world**, not a training step, batch, timing repeat or model seed. Use paired worlds across methods. Model seeds are nested replicates whose losses are averaged within a world. Start pilot variance estimation with several independent worlds and at least two model seeds; add worlds or seeds according to measured between-world and within-world variance, rather than multiplying seeds indiscriminately.

Use the variance decomposition `Var(world mean) ≈ σ_world² + σ_seed²/K_seed` and measured cost to allocate replication. Usually an additional world is more informative than another optimization seed when `σ_world²` dominates. Fit this decomposition on pilot data; do not assume it in advance.

Before protected evaluation, predeclare a primary comparison against the development-selected **strongest baseline champion** under each named resource contract. A claim of beating *every* locally validated family additionally requires simultaneous comparisons with all frozen family champions. Let `M` be the number of primary contrasts, including resource envelopes and any separate confirmatory transfer claim. Use familywise 95% intervals, initially paired world-level t intervals with Bonferroni correction `α/M`. State the approximation when world-level differences are not approximately normal; inspect normality and tail behavior on pilots, not by choosing the friendliest test after confirmation.

Choose the fixed number of worlds from pilot variance and a **precision objective**, not a desired positive result. For candidate half-width `e`, solve iteratively for `n` in

\[
 t_{1-\alpha/(2M),n-1}\,s_D/\sqrt n\le e.
\]

Use a conservative upper uncertainty estimate for `s_D`, and validate prospective interval coverage by simulation from the generator and by resampling pilot worlds. A defensible precision scale is half the smallest stable risk difference between adjacent **development baseline Pareto-frontier** configurations at the candidate’s measured resource envelope: an effect below that resolution is unlikely to change the local configuration choice. This is a measurement-resolution criterion, not a required percentage win. If the frontier provides no nonzero stable gap, report achievable half-width versus total measured runtime and choose a declared precision/resource point; do not invent application utility.

If pilots show severe heavy tails, preregister world-level bootstrap intervals with an explicitly stated asymptotic limitation, or use the bounded-loss alternative below. Do not switch inferential procedures after seeing which produces significance. Bootstrap resamples are computations, not additional experiments.

A distribution-free sensitivity bound is available because mixed-probability loss is bounded. With `L_max=log(C/ε)` and `D∈[-L_max,L_max]`, simultaneous Hoeffding half-width is

\[
 e_H=2L_{max}\sqrt{\log(2M/\alpha)/(2n)}.
\]

This can be extremely conservative. Report it as a rigorous robustness check when feasible; do not pretend that small-sample bootstrap intervals have the same finite-sample guarantee.

The sample-size manifest records `n`, seed nesting, `M`, metric aggregation, desired and achievable half-width, pilot source IDs, statistical assumptions and projected full cost. Freeze it before unlocking any confirmatory aggregate. A smaller local budget leads to **inconclusive at the achieved precision**, not “no effect.” No repeated inspection until significance. Operational monitors may inspect crashes, NaNs, resource use and protocol violations but not unblinded comparative protected scores.

For gas and Fashion-MNIST, report their conditional uncertainty separately. Do not pool their seeds with synthetic worlds or treat ten batches as thousands of independent tests. A batch/block bootstrap is at best exploratory with so few heterogeneous batches; retain per-batch estimates rather than oversell its precision.

### 6.7 What counts as success

A positive primary confidence interval establishes an average risk improvement under its frozen comparison contract. It does **not** by itself establish economic novelty or practical benefit. To claim the selected mechanism matters, the relevant matched-control contrasts must also survive their declared uncertainty and the algebraic alias must match exactly. To claim practical advantage, the candidate must offer a non-dominated, reproducible risk/time/memory tradeoff and disclose latency, recurrence and subgroup regressions.

No arbitrary percentage improvement is required. A small genuine effect is reportable as small. Without an application value function, report the Pareto tradeoff instead of inventing a conversion from milliseconds to accuracy. A stationary regression, loss of an entire gas class or catastrophic loss on recurring tasks blocks an unqualified “better overall” statement even if average NLL improves.

After exploration identifies a promising regime, perform a fresh, locked confirmation using new synthetic worlds and new transformation sequences. Changing the primary hypothesis, architecture, prices, metric, horizon, data access or baseline champion after protected inspection requires a recorded amendment and fresh independent confirmation. Previously viewed results remain exploratory forever.

## 7. M3 Max feasibility, stack and hardware-adaptive execution

### 7.1 What has and has not been verified

The tensor algebra and storage counts support a credible local PoC; actual M3 Max throughput, installed-wheel compatibility, memory pressure and long-run stability **have not been measured here**. The numerical checks in Appendix A ran on Linux x86_64 with NumPy, not on Apple hardware. They establish limited mathematical/implementation identities, not learning performance or MPS support.

Apple’s inspected PyTorch page describes an MPS path and lists macOS, Python and developer-tool prerequisites. Its displayed stable release differed from the version reached by the current PyTorch documentation. Therefore select a compatible released ARM64 wheel locally and record the exact successful combination; do not treat a marketing page or a documentation redirect as proof of installed-version support. [R24, R25]

### 7.2 Discovery and dependency strategy

The following are **existing macOS shell commands**, not proposed project commands:

```bash
uname -m
sw_vers
sysctl -n hw.memsize
sysctl -n machdep.cpu.brand_string
df -h .
system_profiler SPHardwareDataType SPDisplaysDataType
xcode-select -p
python3 --version
```

Capture output in `artifacts/hardware/discovery.txt`, with wall-clock date, power source, free storage, memory pressure and relevant environment variables. Redact machine serial numbers from public reports. Verify `arm64`; do not accidentally benchmark an x86_64 interpreter under translation. Discover memory rather than assuming the largest M3 Max configuration. Installing missing developer tools or changing system software may require the owner’s action; do not upgrade the operating system or alter global Python packages without authorization.

Use an isolated environment and a Python minor version for which compatible ARM64 wheels exist for **all** selected packages. Prefer released packages over nightlies. The intended stack is NumPy for the float64 reference; PyTorch for CPU/MPS dense float32 training; SciPy for statistical routines; scikit-learn and River for baselines; pytest for tests; and standard-library TOML/JSON/CSV, hashing, logging and compression where possible. This is a selection rule, not an unverified fixed lockfile. After successful smoke tests, save exact versions, wheel hashes, interpreter path, architecture and a reproducible dependency lock. Do not fabricate a release number before local resolution.

The candidate requires no graph framework, multi-agent simulator, CUDA extension, sparse custom kernel, model download or paid service. Use dense tensors with Boolean masks: this implements sparse connectivity faithfully but does **not** earn sparse-kernel speed claims. MLX is an optional later engineering port, not a dependency or an assumed faster implementation. A switch of backend requires parity tests and fresh performance measurement.

Retain package licenses and dependency notices. The inspected CBP and UPGD repositories identify MIT licenses; the SWR repository also identifies MIT and reports a macOS test environment. River’s source license is BSD 3-Clause. These observations do not verify every transitive dependency or grant rights to an unrelated dataset. Weight Maximization’s README identifies Apache 2.0; preserve its notice and inspect the pinned source tree before redistribution. The project can use independently written algorithm adapters where code reuse is unavailable, with attribution and no false claim of byte-for-byte reproduction. [R08, R12, R14, R27, R28]

### 7.3 Critical-operation gate

Run a deterministic small fixture on CPU float64, CPU float32 and MPS float32 where available. The fixture must exercise: matrix multiplication, sigmoid, log-sum-exp, log-softmax, indexed target-logit access, masks, random uniforms, reductions, local manual updates, backpropagation baselines, optimizer state reset, ranking, graph edits, and serialization/restoration of RNG state. Check both common and edge branches: zero outgoing weights, all-silent units, saturated sigmoid, absent classes, empty eligible sets, zero utility and final partial blocks.

Use CPU float64 as the oracle. Float64 accumulation for proposal decisions may run on the CPU after a small, explicitly timed transfer of sufficient statistics; do not assume the MPS device supports that precision. Use the same decision arithmetic for matched controls. Initial parity tolerances are `atol=10^-6, rtol=10^-5` for scaled, well-conditioned one-step float32 outputs and gradients; measure actual error and use analytical scale bounds where a tolerance fails. Do not repeatedly loosen thresholds to make failures disappear. Exact combinatorial identities and mask/count invariants have no approximate tolerance. Cross-device long trajectories need not be bitwise equal; compare one-step numerical fidelity and distributional behavior separately.

PyTorch documents explicit MPS synchronization, RNG state access and allocation-reporting functions. Verify each API in the installed version before calling it. Disable undeclared automatic CPU fallback during device qualification: an unsupported operation should produce a visible error. A faithful whole-method CPU execution is an acceptable fallback and must be timed as such. Mixed-device execution is permitted only as an explicitly implemented, tested and fully charged variant. [R25]

Do not use float16, bfloat16, quantization, surrogate sparse kernels or changed gradient estimators as emergency fixes. These can alter probabilities, bankruptcy ordering and learning. Introduce any such variant only after float32 correctness and as a separately evaluated approximation.

### 7.4 Memory model and arithmetic estimates

For one hidden layer,

\[
 P=N(d+C+1)+C.
\]

Float32 weights use `4P` bytes. Weights plus gradients and two Adam moment arrays use approximately `16P` bytes, before step counters and framework overhead. The Boolean mask uses about `Nd` bytes when stored unpacked. Activations require multiples of `4B(d+N+C)` bytes. Two counterfactual branches in a neuron chunk of size `q` require approximately `8BqC` bytes for logits, plus temporary reductions. Audit data require `4Ad+8A` bytes for `A` examples before metadata. Eligibility, ages, reserves, utility means, per-coordinate optimizer ages, RNG state and transaction copies must be counted separately.

| Illustrative configuration | Parameters, float32 | Weights + gradients + two moments | One full `B×N×C` logit array |
|---|---:|---:|---:|
| `N=1000,d=128,C=6,B=64` | 540,024 bytes | 2,160,096 bytes | 1,536,000 bytes |
| `N=1000,d=784,C=10,B=64` | 3,180,040 bytes | 12,720,160 bytes | 2,560,000 bytes |

These are **derived array sizes**, not resident-memory measurements. Python, frameworks, the device driver, optimizer counters, temporary tensors and retained computation graphs can dominate them. A second hidden layer adds its own weight and activation arrays; naive batched counterfactual graph copies can multiply peak usage and must instead be processed sequentially/chunked.

The gas feature matrix is approximately 7.1 MB in float32 by dimensional arithmetic. A complete 70,000-image Fashion-MNIST array is approximately 55 MB as uint8 or 220 MB as float32, excluding labels and copies. Keep canonical image storage uint8 and convert batches on demand; this is lossless storage, not quantization of learned parameters. Download sizes and archive hashes still require local verification.

### 7.5 Calibration and configuration rules

Measure full update blocks for `N=32,256,1000` at `B=1,32,64`, with neuron chunks that fit. Those are starting probe points, not performance promises. For each method, measure prediction, credit, learning, topology, logging and checkpoint costs separately and end-to-end. Use a reference stream long enough that launch overhead and initialization do not dominate the estimate. Continue timing repetitions until the median and its uncertainty support the scheduling decision; do not count timing repeats as predictive-evaluation replicates.

Begin with a working allocation below half of measured currently available memory and below half of the device’s recommended budget, reserving headroom for the OS and temporary copies. These are conservative safety rules, not required utilization targets. Discover their actual values locally; stop and resolve existing memory pressure rather than assume physical RAM is free. Process RSS and MPS driver allocations overlap in unified memory and must not simply be added as independent physical allocations.

If memory is tight, first reduce counterfactual chunk size, stream data, release stale graphs and reduce checkpoint duplication. These changes preserve the model and feedback contract. Only then consider a smaller block size; because that changes feedback delay and learning dynamics, rerun all affected methods and refreeze the protocol. Reduce model size only through a declared scale branch, not by silently replacing the 1,000-unit test. If even the framework cannot run safely, use the NumPy/CPU correctness path and report a hardware-limited competitive evaluation.

Estimate each stage from measurements, for example

\[
 T_{confirm}=\sum_{config,world,seed}
 (H/B)\,t_{block}(config,B)+T_{probes}+T_{checkpoints}+T_{analysis}.
\]

Use a separate term for the two-layer intervention path, where credit can require one descendant rerun per audited unit. Forecast both median and conservative tail runtime, total writes and peak storage. Account for failed work and repeated source reproduction. No wall-time estimate in this plan is a measurement of the target machine.

| Stage | Dominant cost | Local calibration / escalation rule |
|---|---|---|
| Enumeration and fixture tests | Tiny state spaces and CPU arrays | Run first; any failure blocks dependent learning |
| Baseline source reproduction | Published update count × measured small-network update time | Pilot first; retain partial-versus-full reproduction labels |
| Mechanism pilots | Worlds × seeds × drift horizons × methods | Expand only to resolve variance, learning time or a specified failure |
| Tuning | Complete development trajectories and management overhead | Log cumulative budget by family; validate any early-elimination rule |
| Confirmation | Fixed preregistered sample size | Begin only when the whole workload is forecast feasible or declare achieved precision in advance |
| Dependency experiment | Descendant intervention reruns | Keep small and exact first; sampling is a new estimator requiring validation |
| Reproducibility audit | Selected full reruns plus crash/restart tests | Required even when average accuracy appears favorable |

Choose CPU or MPS per method using development timings and parity; do not force trees or tiny models onto an unsuitable backend. Do not compare an optimized candidate against a baseline left on an unnecessarily slower faithful backend without also reporting its best local configuration.

### 7.6 Timing boundaries and system variability

Use AC power and record the power mode, temperature/thermal-pressure indicators that are available without privileged tools, thread settings and background load. Randomize or interleave method timing order across repeat sessions. Do not claim exact physical energy from these indicators.

Warm up framework initialization and the relevant kernels on disposable model state. Restore initial parameters and RNG states before the measured run. Synchronize MPS immediately before and after timed GPU regions; use a monotonic high-resolution clock. Avoid per-operation synchronization in the production path unless it is required: microbenchmark instrumentation can change the total cost. Measure a second end-to-end run with only boundary synchronization.

Report separately cold startup, reusable dataset loading/preprocessing, per-observation prediction, adaptation including credit/audit replay, periodic probes, checkpointing, and total elapsed time. State whether model-specific preprocessing is inside the primary comparison. The principal adaptation comparison includes **all candidate-specific management work**. Repeated evaluation forward passes are not free inference. Use identical logging detail in timed comparisons or measure logging overhead explicitly.

### 7.7 Checkpoint and interruption contract

Checkpoint only at a complete event-block transaction boundary. State includes parameters, masks, optimizer moments and coordinate ages, unit generations, reserves, utility/activation statistics, grace ages, turnover/proposal tokens, fee debits, the audit buffer and cursor, preprocessing, stream position, event IDs, all random-stream states, counters, configuration, source revision and data hashes. Save device RNG through the verified installed API; pre-generated counterfactual random uniforms must be reproducible on replay.

Write to a temporary file, flush, then atomically rename; update a small manifest last. A crash mid-block rolls back to the last committed state and replays the entire uncommitted block. Event IDs prevent duplicate scoring or a second label update. Keep failed attempts and their elapsed cost in the ledger rather than erasing them.

Test uninterrupted versus resumed CPU runs for exact equality in the same locked environment. For MPS, test available determinism explicitly. If bitwise replay is unavailable, save the relevant generated randomness and enforce one-step tolerances plus a documented statistical equivalence test; do not advertise bitwise reproducibility. An unexplained divergence in reserves, masks or event order is a blocking correctness failure, not acceptable device noise.

## 8. Repository, interfaces, commands and experiment records

### 8.1 Minimal repository layout

The coding agent must create this repository; it does not yet exist as a delivered trained implementation.

```text
ego-poc/
  README.md                         # setup, run/resume instructions, current decision
  research_poc_plan.md               # this handoff, with revision history
  pyproject.toml
  dependency-lock.*                  # actual successful platform-specific lock
  configs/
    pilot.toml
    frozen/                         # immutable confirmation configs and manifests
  src/ego_poc/
    __main__.py                     # CLI dispatcher
    hardware.py                     # discovery, compatibility, calibration
    streams.py                      # observations/labels transaction interface
    generators.py                   # synthetic worlds, multiplexer, transforms
    data.py                         # licensed downloads, parsers, causal preprocessing
    methods/base.py
    methods/local.py
    methods/backprop.py
    methods/baselines.py
    credit.py                       # shallow exact and DAG intervention evaluators
    population.py                   # reserves, aliases, eligibility, reset identities
    topology.py                     # sequential proposals and capacity enforcement
    rng.py
    runner.py
    checkpoint.py
    metrics.py
    inference.py                    # world-level uncertainty and paired analysis
    report.py
  tests/
    test_enumeration.py
    test_credit.py
    test_deep_interventions.py
    test_baseline_fixtures.py
    test_population.py
    test_topology.py
    test_stream_leakage.py
    test_resume.py
    test_device_parity.py
    test_statistics.py
  provenance/
    sources.jsonl
    licenses/
    baseline_specs.md
    patches/
  data/raw/                         # excluded from public repository when required
  data/manifests/
  artifacts/
    hardware/
    pilot/
    confirmation/
    checkpoints/
    failures/
    reports/
  decisions.md
  amendments.jsonl
```

Use flat modules when sufficient; do not spend effort constructing a general multi-agent framework. Separate scientific controls in configuration rather than copy-pasting subtly different training loops.

### 8.2 Shared method and stream interfaces

Proposed Python protocol, with concrete semantics the implementation must preserve:

```python
class OnlineMethod:
    def predict(self, x, *, event_ids, rng):
        """Return probabilities [B,C], immutable update cache, cost counters.
        No labels; no parameter update; validates one prediction per event.
        """
    def learn(self, x, y, *, cache, event_ids):
        """Exactly one committed-label update, including declared management.
        Return scalar diagnostics and component timings. Reject stale cache.
        """
    def state_dict(self):
        """Complete serializable state, including method-owned RNG/counters."""
    def load_state_dict(self, state):
        """Validate schema/config/data hashes before restoring."""

class Stream:
    def observations(self):
        """Return block X and unique event IDs; withhold labels and task ID."""
    def commit_predictions(self, event_ids, probabilities):
        """Durably record outputs before enabling reveal."""
    def reveal(self, event_ids):
        """Return labels only for a correctly committed observation block."""
```

The method returns raw normalized probabilities; the runner applies the one common reporting mixture exactly once, commits those reported probabilities, and scores them. Training uses the original cached logits/probabilities. The runner owns scoring, device synchronization boundaries, protected metadata and transaction ordering. A model may inspect labels only inside `learn`, not by holding a reference to a complete stream object. Use immutable snapshot IDs to prevent accidentally computing credit from updated weights. Do not call a method’s scoring interface again after it learns and replace its original prediction.

Keep RNG streams distinct for world construction, observation/label generation, initialization, activation, optimizer noise, reset selection, newborn weights, topology proposals, audit sampling and timing-order randomization. Derive seeds from a recorded master seed plus stable named indices using a specified hash/seed-sequence function, not Python’s process-randomized `hash`. Pair corresponding streams across matched controls where their semantics agree; do not force false common randomness between unlike algorithms.

### 8.3 Configuration conventions

Use explicit units in keys, for example `reserve_decay_observations`, `maturity_observations`, `audit_buffer_examples`, `checkpoint_every_observations` and `max_active_edges`. Store integers for counts, float64-compatible numeric values for prices, and enumerated strings for methods/reference interventions. Avoid hidden defaults in a frozen run. A final short block uses its actual size in gradients, reserve decay, age advancement and quota accrual.

A **pilot-only example**, not a ready confirmation configuration:

```toml
schema_version = 1
phase = "pilot"
method = "economy_rb"
device = "discover"
dtype = "float32"
world_family = "sparse_teacher_v1"
input_dim = 32
classes = 2
hidden_units = 32
block_size = 32
initial_degree = 4
max_active_edges = 128
credit_rule = "two_action_exact"
hidden_rule = "rb"
reset_reference = "zero"
settlement = "conditional_expected"
optimizer = "sgd"
hidden_lr = 0.001
output_lr = 0.01
weight_decay = 0.0
score_uniform_mixture = 0.000001
calibrate_prices = true
calibrate_horizon = true
calibrate_replicates = true
```

A confirmation config must have `phase="confirmation"`, concrete hardware/backend, all prices and timescales, full initial/reset distributions, proposal rates/fees, degree bounds, all seeds, metric weights, test plan, horizon, sample size and checksum references. Any unresolved `discover`, calibration flag or missing scientific field blocks confirmation. There is no default guessed memory configuration.

### 8.4 Commands to implement

The following are **proposed project CLI commands**, not existing third-party APIs:

```bash
python -m ego_poc discover --out artifacts/hardware
python -m ego_poc validate --suite mathematical
python -m ego_poc validate --suite baseline-fixtures
python -m ego_poc validate --suite device --devices cpu,mps
python -m ego_poc data-prepare --manifest data/manifests/approved.toml
python -m ego_poc reproduce --baseline weightmax --mode source-faithful
python -m ego_poc calibrate --config configs/pilot.toml
python -m ego_poc run --config configs/pilot.toml
python -m ego_poc freeze --manifest configs/frozen/confirmation.toml
python -m ego_poc run --config configs/frozen/confirmation.toml
python -m ego_poc resume --run-id RUN_ID
python -m ego_poc analyze --manifest configs/frozen/confirmation.toml
python -m ego_poc audit --run-id RUN_ID
python -m ego_poc report --manifest configs/frozen/confirmation.toml
```

Implement useful nonzero exit codes: invalid configuration, protocol violation, numerical failure, resource exhaustion and incomplete run. `freeze` validates that prerequisites, baseline fixtures, licensing, sample-size planning and config locks exist. It must not falsely guarantee immutable secrecy: access control is procedural unless separate permissions are configured. Never print protected comparative scores through a progress bar before the planned analysis.

### 8.5 Required record schemas

`run.json` contains: schema version, run ID, phase, UTC timestamps, code commit and dirty-tree diff hash, config hash, dependency lock hash, machine/OS/backend, dataset/generator hashes, world/family/seed IDs, source-baseline version, information contract, resource contract, intended observation count, checkpoint policy and status.

`events.jsonl` or compressed columnar equivalent contains: run ID, unique event-block ID, observation count, observation offset, committed prediction hash, label-reveal sequence number, snapshot ID, raw summed NLL and squared-error sums, correct count, per-class totals, component timing in seconds, hardware counters with units, active edges, firing sums, deaths/births/proposals/acceptances, net issued credits, decay loss, fees and account totals. Keep raw per-event probabilities and labels where storage allows; for deterministic regeneration, retain input/event hashes and generator state. Do not store only smoothed plots.

`unit_events` records unit slot, generation, parent if any (normally none), birth/death observation, reserve immediately before/after settlement, eligibility, reason, incoming mask change, intervention reference and reset-state hash. `decisions.md` records each gate, evidence path, result, next action and unresolved risk. `amendments.jsonl` records old/new scientific contracts, reason, affected results and fresh-confirmation requirement.

Preserve unsuccessful settings, NaNs, crashes, excluded runs and exclusion reasons. Valid bad performance is never a reason for exclusion. Distinguish reusable setup expense, development expense and per-deployment expense. The agent may make routine non-destructive implementation choices, but changes to hypotheses, data access, primary metrics, model ownership, rewards, comparators or confirmation rules require a methodological amendment.

## 9. Risk-ordered work packages and decision gates

Each package produces artifacts that allow another agent to resume without this conversation. Resource forecasts are computed from §7 rather than assigned arbitrary time limits. A failed gate records one of: implementation failure, invalid measurement, unsuitable task/sensitivity, resource limitation, evidence against a hypothesis, or a correct mechanism with no competitive advantage.

### WP00 — Establish an executable, lawful environment

**Objective/risk:** unknown hardware, dependency support, data rights and version drift. **Dependencies/entry:** this plan and access to the target laptop; no dataset or model download assumed.

**Actions/files:** create the minimal repository; run discovery; resolve an isolated ARM64 environment; implement `hardware.py`; archive primary source metadata, exact code revisions and licenses in `provenance/`; resolve or block the gas license condition; implement synthetic generation without external data. Capture the actual installed signatures and versions of third-party APIs rather than relying on current online docs forever.

**Tests/outputs:** `discovery.txt`, machine-readable hardware manifest, dependency lock, license decisions, source revision ledger and a successful tiny CPU calculation. Verify available disk for forecast artifacts, not merely dataset downloads. Hardware identifiers and private paths are redacted from shareable reports.

**Resources:** discovery and tiny arrays; estimate framework baseline RSS before allocating models. **Acceptance:** native ARM64 interpreter, functioning CPU reference, known legal status for each enabled asset and saved locks. **Branches:** CPU works but MPS does not → continue correctness on CPU and test faithful acceleration later; package conflict → isolate/port the small algorithm with fixture parity; unresolved asset rights → disable that arm; neither CPU nor a safe environment works → blocked prerequisite, not evidence against the idea.

### WP01 — Validate the mathematics before training

**Objective/risk:** biased credit, false decentralization claims and economic bookkeeping errors. **Dependencies:** WP00 CPU reference. **Actions/files:** turn Appendix A into `test_enumeration.py`; add exhaustive two-layer enumeration, variable-block reserve recurrence, fee settlement, newborn grace/endowment, alias equivalence and capacity tests. Test both task-only and firing-cost-sensitive gradients. Construct OR duplicates, AND complements and a necessary downstream relay.

**Tests/outputs:** a signed mathematical-fixture report with inputs, expected outputs and float64 errors. Use central finite differences and exact enumeration independently. Check that permutation of hidden-unit IDs permutes, rather than changes, results. Zero readout weights must produce zero task credit; adding a constant to all class logits must not change cross-entropy or credit. Illegal edges and duplicate indices must be rejected.

**Resources:** tiny finite state spaces, no large dataset. **Acceptance:** gradient agreement within stated numerical error, exact mask/count invariants and exact reserve/trace alias agreement under shared arithmetic. **Branches:** identity failure → fix the implementation/specification before training; budget conservation failure in OR/AND → expected, preserve minted-credit interpretation; dependence only works after descendant reruns → correct the evaluator and update cost forecast; fixed theory passes but moving-network convergence remains unknown → proceed, without a convergence claim.

### WP02 — Build and attack the evaluation harness

**Objective/risk:** label leakage, seed pseudoreplication, scoring ambiguity and false positives. **Dependencies:** WP01. **Actions/files:** implement `streams.py`, `metrics.py`, `runner.py`, named RNG streams and event transactions. Generate independent worlds; implement uniform, Bayes-oracle evaluator-only, deliberately corrupted and linear predictors. Implement paired-world analysis and sample-size calculations.

**Tests/outputs:** copying a learner under two names with identical RNG must produce identical scores and zero paired differences; a Bayes oracle must outperform an intentionally corrupted predictor in expectation on sufficiently many generated worlds; shuffled labels independent of inputs must not yield reproducible out-of-sample learning gains. A method trying to access labels before commitment or score twice after updating must raise an error. Simulated null studies assess the proposed interval procedure’s coverage and false-positive behavior; their simulation repetitions are not training-data replicates.

**Resources:** generated data and simple predictors; derive simulation effort from uncertainty in coverage estimates. **Acceptance:** all access and equivalence tests pass, metric algebra matches hand calculations, oracle headroom is measurable, and analysis assumptions are recorded. **Branches:** known differences undetected → calibrate sensitivity before rejecting a learner; cloned methods differ → RNG/order/harness bug; oracle advantage absent because the task is saturated or noisy → revise the development generator and record the change; invalid inference → fix it before any protected evaluation.

### WP03 — Reproduce the closest precedent and strong simple learners

**Objective/risk:** novelty based on ignorance or comparison against broken baselines. **Dependencies:** WP02 and licensed source records. **Actions/files:** build the multiplexer, run Weight Maximization source fixtures and calibrated reproduction; implement linear, dense backpropagation and ARF adapters; compare basic updates with trusted library behavior. Record source-faithful versus transferred variants in `baseline_specs.md`.

**Tests/outputs:** multiplexer truth table for all addresses on constructed bit vectors; published-condition manifest; learning curves and uncertainty; baseline convergence on stationary linear/nonlinear tasks; fixed-input update traces. For an API compatibility patch, retain the original and patched outputs on the same fixture. An original RL reward cannot be silently replaced with supervised cross-entropy in the reproduction report.

**Resources:** measured update time times the source schedule, plus small benchmark pilots. **Acceptance:** verified task, valid reference updates and working strong simple learners. Full reproduction status is explicit. **Branches:** no baseline learns → investigate task/preprocessing/harness; only the port fails → fix or exclude the port with a blocking limitation; Weight Maximization is correct and competitive → treat its overlap as substantive and narrow novelty; original schedule exceeds local capacity → report partial replication, retain exact fixtures, and restrict comparative claims accordingly.

### WP04 — Implement the minimum complete economic mechanism

**Objective/risk:** the core idea may reduce to ordinary optimization or fail to sustain useful units. **Dependencies:** WP01–WP03. **Actions/files:** implement `local.py`, `credit.py`, `population.py` and `topology.py` exactly as §4; begin with 32 units; implement the trace alias and controller-disabled configurations first. Use plain SGD until numerical fixtures pass. Add manual Adam only with coordinate-age and reset tests.

**Tests/outputs:** end-to-end runs exhibit learning, nonzero costs, solvent useful identities, actual eligible replacement and accepted/rejected legal topology proposals. Trace and reserve implementations must match. Fees are deducted when a proposal is evaluated; a fee-induced nonpositive reserve becomes eligible at the **next block’s** retirement gate, preserving the stated ordering. Proposed structural logits must agree with brute-force forward recomputation; sequential proposals must see preceding accepted edits. Resetting a unit must reset only its incident optimizer coordinates and designated state.

**Resources:** small arrays; measure credit/audit overhead separately. **Acceptance:** all invariants hold, complete behaviors occur without forcing resets by hand, and oracle task headroom allows learning to be evaluated. **Branches:** no positive credit → fix learning/initialization before tuning economics; perpetual extinction → inspect prices, decay, bootstrap subsidy and information delay; no accepted proposals → verify gain computation and available useful edges before concluding topology is useless; alias alone explains behavior → expected equivalence, not a novel economy; correct behavior but poor learning → retain negative result and continue only targeted diagnostics.

### WP05 — Validate contemporary alternatives and attribution controls

**Objective/risk:** generic plasticity preservation or hidden information advantages explain a win. **Dependencies:** WP03–WP04. **Actions/files:** implement/pin CBP, UPGD, SWR and GXD adapters; add ReDo and RigL when their claims are active; reproduce source scalar fixtures; implement shared-controller and reset-count-matched controls. Test uncorrected/corrected CBP mean handling where material.

**Tests/outputs:** every baseline has an update-equation specification, version, source/default config, data contract, tuning space, reset-state fixture and at least one functioning development trajectory. Verify that potential dormant-edge gradients are not accidentally zero. Expose audit buffer access to non-economic methods through the same budgeted interface.

**Resources:** calibrate each method and rank implementation uncertainty before broad sweeps. **Acceptance:** required comparison families are faithful enough for the stated task transfer and their important hyperparameters have been exercised. **Branches:** a cheap conventional method matches the candidate → investigate whether anything remains beyond engineering; all utility methods help similarly → report shared mechanism, not selfishness; newest comparator cannot be faithfully specified → block “best contemporary” claims but retain narrower comparisons; a generally useful optimization helps the candidate → apply it to eligible competitors before confirmation.

### WP06 — Calibrate horizons, hyperparameters, hardware and precision

**Objective/risk:** unfair tuning, insufficient training exposure, hidden runtime and underpowered conclusions. **Dependencies:** WP04–WP05 with working harness. **Actions/files:** run the device gate; measure end-to-end costs; estimate learning/bootstrap times and variance components; tune only on development worlds and allowed data prefixes; construct baseline Pareto frontiers; set prices and reset/structure schedules; calculate fixed confirmatory sample size and storage forecast.

**Tests/outputs:** `pilot_report.md`, timing raw data, tuning ledger including failures, best configurations, per-family results, sensitivity findings, selected precision, full projected workload and a proposed frozen manifest. Verify that method rankings are not artifacts of initialization, block size, hidden-state inference sampling, logging or temperature drift in timings.

**Resources:** derived from observed cost and expected information gain. Allocate an additional experiment only to a recorded unresolved decision; do not continue searching merely for a favorable result. **Acceptance:** working complete method, sufficiently strong baselines, informative task/horizon, stable measurement, and a fixed confirmation plan feasible at disclosed precision. **Branches:** local resources insufficient for desired precision → choose a smaller explicitly scoped claim or declare intended test locally inconclusive; brief tasks miss renewal effects → extend complete cycles before judging renewal; advantages vanish under tuning → no competitive improvement, proceed to report or a justified diagnostic; plausible regime-specific effect → preregister that narrower regime and confirm on fresh worlds, not the pilot cases.

### WP07 — Run the frozen comparison and transfer evidence

**Objective/risk:** adaptive selection and unsupported generalization. **Dependencies:** all required WP06 gates, signed config/analysis manifest, enabled asset licenses. **Actions/files:** validate `freeze`; execute independent worlds and nested seeds; run the protected gas history or licensed fallback under its separate contract; checkpoint transactionally; retain all valid negative outcomes. The runner suppresses comparative protected scores until the planned analysis point.

**Tests/outputs:** complete raw predictions/statistics, costs, valid/failed run ledger, all expected sample IDs and signed protocol compliance report. Check counts and hashes without inspecting comparative outcomes. A configuration change ends the old confirmation attempt and starts an amendment process.

**Resources:** measured fixed workload; no stopping when results become positive. **Acceptance:** all predeclared valid runs completed or an explicitly labeled incomplete study with achieved precision. **Branches:** implementation/hardware failure → repair only after determining whether the repair changes the scientific contract; protocol breach → invalidate affected confirmation and use fresh worlds; positive, negative or overlapping intervals → all proceed to attribution/interpretation rather than selective reporting.

### WP08 — Dependencies, adversarial tests and reproducibility

**Objective/risk:** shallow-only credit, exploitable incentives, fragile gains and inconsistent resumes. **Dependencies:** WP04 for mathematical dependency tests; WP06 for learning calibration; WP07 results are not needed to choose the predeclared mandatory tests. **Actions/files:** implement the small two-layer evaluator and compare with shared coagents, Weight Maximization and backpropagation on the multiplexer/dependency task; run the adversarial matrix in §10; perform uninterrupted/resumed runs and clean-environment reruns of the important result and a negative control.

**Tests/outputs:** descendant-rerun gradient parity, dependency learning curves, cost scaling, clone/relay/rare-feature stress reports, checksum or numerical-parity reports and full replay logs. Recurring-teacher retention is measured after useful units have become temporarily inactive. A discovery made after confirmation is exploratory unless independently reconfirmed.

**Resources:** exact small networks first; scale only when correct small networks express the relevant dependency. **Acceptance:** no unexplained causal-credit or state-replay error; claims narrowed to tested robustness. **Branches:** deep task correct but too expensive → shallow performance claim only; deep task lacks representational capacity → expand within measured capacity before drawing a mechanistic conclusion; reward exploitation succeeds → document failure and test an explicit revised mechanism on fresh data; tiny or unstable effects → no breakthrough claim; independent rerun fails → unresolved reproducibility, not success.

### WP09 — Synthesize, audit and decide

**Objective/risk:** overclaiming a positive metric or burying failures. **Dependencies:** prior executed packages and their status records; incomplete packages are allowed but must remain visible. **Actions/files:** generate tables and individual-world plots; compare all frozen baselines and ablations; perform the three audits in §11; update novelty and limitation ledgers; produce `artifacts/reports/final_report.md` plus machine-readable result tables.

**Tests/outputs:** traceability from each headline to configuration, data, baseline, metric, interval and cost boundary; report every material regression and missing prerequisite; verify that a fresh agent can regenerate tables from raw artifacts. No report code may choose the best protected seed or discard poor worlds. **Resources:** stored results and selected verification reruns, calibrated locally. **Acceptance:** one of the evidence classes below is justified with its scope and next action. **Branches:** preserve or revise the mechanism only when a specific unresolved hypothesis warrants it; otherwise conclude replication/equivalence, practical domination, negative evidence or inconclusiveness without manufacturing novelty.

## 10. Robustness, adversarial scenarios and interpretation

### 10.1 Required stress matrix

| Stress / failure mode | Implementation | Interpretation and corrective branch |
|---|---|---|
| Raw activity farming | Add independent, high-variance nuisance inputs; compare activity/degree payment to task credit | More payment without predictive benefit falsifies that influence proxy; do not redefine activity as success |
| Duplicate free riding | Duplicate a useful feature and split or rescale its readout contribution | Measure whether marginal credit incorrectly starves useful redundant capacity; OR example predicts non-additive credit |
| Complementary coalition | Use an AND/parity feature pair and the dependency relay | Test whether individually weak/complementary units survive discovery; failure may require coalition credit, which is a new expensive mechanism |
| Scale gaming | Apply function-preserving rescaling where the architecture permits it; compare magnitude-based payment | Invariance failures expose proxy dependence. Do not apply ReLU rescaling identities to sigmoid units where they are not valid |
| Credit-oracle manipulation | Restrict unit access to its numerical state; test forbidden changes to the evaluator, labels or payment coefficients | Any access violation is an implementation/security failure, not emergent intelligence |
| Extinction spiral | Increase costs, delay informative output learning and test zero-credit newborns | Identify subsidy and grace dependence; a population sustained only by external births is not economically self-sufficient |
| Monopoly / useful rare features | Skew class frequencies and delay a rare task’s return | Report classwise and recurrence damage, not only mean loss; need for insurance/protected capacity becomes an explicit pivot |
| Audit-buffer overfitting | Compare proposal gains on audited training samples with independent evaluator-only probes | Proposal acceptance without fresh predictive gain indicates selection overfit; reduce selection complexity or improve uncertainty control before reconfirmation |
| Moving readout / stale proposals | Interleave sequential proposals and readout updates in a deliberately broken control | Correct implementation rejects stale snapshot IDs; synchronous frozen scoring is mandatory |
| Degree collapse / useless expansion | Use linear, dense and support-shifting teachers; inspect degree and cost trajectories | A penalty may merely prune everything; a capacity constraint may block growth. Compare matched degree controls and RigL |
| Feedback delay and missing labels | Introduce a declared delayed-label queue and randomly missing feedback in an exploratory branch | Do not reuse immediate-feedback theory unchanged; include queue state and charge delayed credit bookkeeping |
| Hardware/version sensitivity | Repeat core fixtures on CPU/MPS and a clean locked environment | Numerical account crossings can cause trajectory divergence; report stability and backend-specific outcomes |

Sensitivity tests vary one scientific dimension at a time or use a predeclared small factorial design: credit estimator, survival gate, topology, price scale, maturity, reset reference, neuron count, sparsity and drift speed. The entire Cartesian product is unnecessary unless an identified interaction requires it. Choose contrasts for identification, not an arbitrary experiment quota.

### 10.2 Compute and information counterfactuals

If the candidate reuses `A` labeled audit examples for proposals, give eligible neural baselines access to the same past-example buffer. Under the equal-information experiment, match unique labels and reveal timing. Under the equal-management-work experiment, compare its audit expense with additional ordinary updates or conventional reset evaluation on that same past buffer. Under a fixed-latency deployment contract, slower methods cannot peek at future labels while faster methods perform extra work on already available data.

Do not equate a fixed observation count with equal compute. Conversely, do not let a fast learner consume more future stream observations and call the result a matched-information online comparison. Report separate data-efficiency and compute-efficiency curves.

### 10.3 Evidence classes and next actions

| Outcome | Permitted conclusion | Next action |
|---|---|---|
| Existing equations or trace alias fully explain behavior | Replication/equivalence; no new economic learning principle | Publish a clear implementation or negative novelty finding; pursue a refinement only with a discriminating hypothesis |
| Local counterfactual learning works, economy adds nothing | Evidence for a known credit construction, not survival-based advantage | Retain the cheapest fixed-controller method; investigate variance or engineering only if useful |
| Utility renewal helps equally with backpropagation | Generic renewal/controller effect | Pivot to a simpler continual-learning controller, explicitly abandoning a necessity claim for self-interested local learning |
| Topology helps equally under conventional optimization | Generic structure search or sparsity benefit | Reuse established sparse training; report engineering differences rather than rename it |
| Candidate improves a locked regime and survives controls/costs | Narrow, reproducible empirical improvement; methodological novelty still requires equation-level prior-art review | Independent confirmation and a mechanism-specific extension; expand scope one justified dimension at a time |
| Synthetic gain fails on sensor/image evidence | Controlled synthetic effect only | Diagnose mismatch and revise the application claim; do not generalize to practical intelligence |
| Correct candidate consistently loses with precise intervals | Evidence against this implementation/setting; possible practical domination | Stop that direction unless a specific missing condition has independent support |
| Intervals are broad or full workload is locally infeasible | Inconclusive at stated precision/resource scale | Report achievable sensitivity; redesign a narrower informative local test without claiming absence of all future effects |
| Essential relay fails only under an incorrect intervention | Implementation/theory-application error | Repair descendant credit and rerun; not evidence against self-interested learning |
| Proper dependency task learns but large graphs are too costly | Local causal-credit feasibility, no demonstrated deep efficiency | Study an explicitly biased/approximate credit estimator only as a new hypothesis with its own validation |

A future scale argument warrants escalation only when there is evidence that the smaller test cannot express the hypothesized dependency or when measured learning/cost trends predict an informative next scale. “Perhaps emergence happens later” alone is not such evidence. Equally, a 32-unit experiment that cannot represent the teacher cannot falsify a claim about representations it lacks.

## 11. Audits performed on this plan and remaining risks

These are **author self-audits**, not independent peer review. Research/source inspection and the limited tests in Appendix A were performed; the proposed model training, baseline reproduction, data downloads and Mac compatibility checks were not. The design below incorporates the corrections found during drafting.

### 11.1 Scientific and prior-art audit

| Defect found | Revision incorporated | Acceptance criterion / residual risk |
|---|---|---|
| Broad “new paradigm” claim contradicted by close historical and modern work | Explicitly classify the original mechanism as substantially anticipated; require Weight Maximization replication and continual-learning comparators | No novelty headline without an equation-level difference and a discriminating experiment; unindexed prior art remains possible |
| An energy account can merely rename a discounted utility statistic | Require exact trace alias and acknowledge equivalence | Alias must match; a nonmatching result is a bug, not a discovery |
| Individual marginal rewards do not conserve team reward | Added OR/AND arithmetic and external credit/endowment ledger | No closed-market, budget-balance or welfare theorem claim |
| Cheap exact scalar credit is not fully local information | Expose global label/readout evaluator and charge it | Any practical/locality claim must state the oracle dependency |
| Newer utility and weight-reinitialization work weakens differentiation | Added GXD, UPGD and SWR controls, including small/normalized networks | A baseline unavailable or inadequately tuned limits the corresponding competitive claim |
| A shallow supervised model cannot establish autonomous hierarchy or memory | Added a mandatory small dependency task and explicit scope limits | Even successful dependencies do not establish recurrent memory, general intelligence or experience |
| Positive fixed-network gradient identity does not imply convergence with births/fees/rewiring | Separate algebra from dynamical hypotheses | Stability and value remain empirical; no general convergence guarantee asserted |

### 11.2 Experimental-validity audit

| Defect found | Revision incorporated | Acceptance criterion / residual risk |
|---|---|---|
| Holding descendants fixed can erase valid parent credit | Interventions rerun descendants with common exogenous randomness | Enumeration catches the error; deep intervention cost may still be prohibitive |
| Updating the readout before credit changes the reward being estimated | One frozen pre-update snapshot for gradients and settlement | Snapshot IDs and hand fixtures must agree |
| Newborn zero-output weights initially produce no task signal | Explicit measured bootstrap grace and visible subsidy | No immediate starvation by construction; dependence on subsidy still reported |
| Zero-ablation credit mismatched mean-preserving resets | Primary zero-reference uncompensated reset; separate matched mean-reference variant | Removal utility and actual reset reference must correspond |
| Weak/noisy coagents could manufacture an apparent win | Added conditional-gradient immortal control and strong practical backpropagation | Credit variance improvement alone is not an economic contribution |
| Extra replay, capacity or inference draws could explain improvements | Separate information, compute, parameter and deployment contracts | Count every audit/forward pass; baselines receive comparable useful optimizations |
| Repeated seeds or shuffled batches were liable to masquerade as independent evidence | Independent synthetic worlds; conditional single-history/image-set uncertainty | No seed-level claim of generalization across real deployments |
| Pilot selection might choose a favorable drift horizon or stop at significance | Horizon calibration across baselines and fixed protected sample sizes | Methodological changes require fresh confirmation; achieved precision disclosed |
| Synthetic success might be oversold as practical transfer | Natural sensor arm with a license gate and labeled artificial-image fallback | Failure to transfer narrows the conclusion rather than disappearing from reporting |
| Mixture log loss and realized stochastic log loss could be confused | Explicit one-draw primary predictor and separately charged mixture variant | Score the actual deployed prediction distribution |

### 11.3 Implementation and feasibility audit

| Defect found | Revision incorporated | Acceptance criterion / residual risk |
|---|---|---|
| Unknown unified memory and version mismatch on documentation pages | Discovery, compatible-wheel selection, lockfile and critical-operation gate | No claim of M3 validation until locally measured |
| Sparse architecture could be confused with sparse-kernel acceleration | Dense masked core and full time accounting | No unsupported kernel or sparse speedup claim |
| Source code may differ from paper shorthand | Weight Maximization source-order fixtures; separate literal-objective port | Published equations and reused executable behavior are not silently reconciled |
| CBP activation-mean reset order may nullify compensation | Saved-mean unit test and disclosed corrected variant | Do not assign the current-source concern to every historical experiment |
| Utility normalization can divide by zero or mix devices | Explicit finite/device-safe scalar handling and zero-utility test | For the first-order UPGD transfer, when the normalizer is numerically zero, set the gate to 1/2, making its doubled gradient step an ordinary noisy gradient step; label this edge-case convention |
| A newborn reset could exceed the global edge budget after rewiring | Rebirth preserves the retired slot’s current degree | Total active edges must be unchanged by replacement itself |
| Adam state and age may survive a supposed new identity | Per-coordinate moment/age resets and scalar optimizer fixtures | A stock global-step reset that disturbs unaffected units is not accepted |
| RigL dormant-edge gradients may be inadvertently masked out | Compute potential dense gradients before mask application | Growth must select real nonzero candidates on a constructed fixture |
| Fees, proposal tokens or event IDs could be lost on resume | Complete transactional state and crash/replay tests | No silent double label use, duplicate births or forgiven fees |
| Dataset page contains conflicting usage statements | Block uncertain use rather than assume permissive rights | The gas arm remains conditional; this plan is not a rights determination |

### 11.4 Residual uncertainties and final challenge pass

The strongest remaining scientific risk is that the trusted utility evaluator plus ordinary renewal explains everything. That would be an informative result, not a failed research process. Exact counterfactual learning can have a low-variance estimator while still being slower or statistically weaker than backpropagation. Deep credit costs can eliminate the shallow computational advantage. Fixed-capacity rebirth may destroy rare useful representations, and private proposal selection may overfit its audit buffer. Prices and subsidies may require task-specific tuning that erases deployment value.

The implementation walk-through leaves explicit blocking gates rather than invented answers: actual Mac memory/OS/wheels; baseline source revisions; gas usage permission; measured bootstrap/learning time; acceptable full-run cost and achievable precision. CPU fallback, licensed synthetic/image workloads, source fixtures, checkpoint state and narrower claim branches are specified so these unknowns do not create circular dependencies.

A second challenge pass checked these corrections against the work-package dependencies: mathematical tests precede learning; baseline validation precedes freezing; prices/horizon/sample size are frozen before protected inspection; dependency tests do not require a prior positive result; and an incomplete baseline or asset gate limits only the claims that depend on it. This is a consistency check, not a guarantee that implementation will reveal no new defect. The coding agent must repeat the three audits after code and results exist.

## 12. References and search-coverage record

### 12.1 Evidence status and coverage

Research was conducted on **12 October 2026**. Search families included selfish/egoistic neurons, neural economies, bucket-brigade credit, Hayek machines, symbiotic neuron evolution, coagent/policy-gradient neurons, Weight Maximization, difference rewards and counterfactual credit, neuron Shapley values, utility-based continual learning, selective unit/weight reset, plasticity loss, dynamic rewiring, online drift ensembles, and Apple Silicon/MPS operation support. Historical references were traced backward; searches for the selected contribution/turnover refinement brought in 2025 and 2026 utility/attribution work. Official implementation inspection resolved material update-order, optimizer-state and reset-reference questions.

The final focused passes clarified baseline implementation and licensing rather than establishing a new economic mechanism. This is a researched coverage judgment, **not an exhaustive census or proof of absence of prior art**. Search indexing was uneven, some exact-term queries returned irrelevant results, and primary direct links were more useful than snippets for several important sources. Some paywalled works were inspected only through author/publisher abstracts; those limitations are identified below. No independent reproduction of the entire selected combination was located and verified. Publicly accessible code was inspected but not trained here.

Patents and deployment history received preliminary screening only; no decisive patent claim analysis or verified deployed system established the proposed combination’s novelty. This plan is not a patentability, freedom-to-operate or legal-licensing opinion. Non-English work, poorly indexed reports, unpublished systems and code changes after the access date remain blind spots. There is no claim that the closest accessible paper is necessarily the closest work anywhere.

Source records use stable IDs below. In the implementation repository, add access timestamps, retrieved-content hashes, exact code commits and compatibility patches to `provenance/sources.jsonl`. A suggested record is `{id,title,authors,year,url,doi,inspected_parts,claim_informed,conditions,limitations,relation_to_proposal,accessed_at,content_hash,commit}`. Leave an unverified field explicitly null rather than inventing it.

### 12.2 Bibliography with inspected evidence and limitations

**[R01] Jürgen Schmidhuber. “A Local Learning Algorithm for Dynamic Feedforward and Recurrent Networks.” Connection Science 1(4), 403–412, 1989.**  
Author-hosted text: <https://people.idsia.ch/~juergen/bucketbrigade/bucketbrigade.html>; algorithm: <https://people.idsia.ch/~juergen/bucketbrigade/node3.html>; experiments: <https://people.idsia.ch/~juergen/bucketbrigade/node6.html>. Inspected historical HTML text and algorithm/experiment sections. Neural Bucket Brigade resource/payment mechanisms materially anticipate the broad economy/local-learning framing. Historical tasks and exposition do not establish modern competitive performance; the hosted version omits figures.

**[R02] Eric B. Baum. “Toward a Model of Intelligence as an Economy of Agents.” Machine Learning 35, 155–185, 1999.**  
DOI: <https://doi.org/10.1023/A:1007593124513>; publisher: <https://link.springer.com/article/10.1023/A:1007593124513>. Inspected publisher metadata and abstract, not the paywalled full article. Hayek-style agent economies are a substantive conceptual precedent. Their structured experimental setting does not validate neuron-level competitiveness on this plan’s tasks; no detailed algorithm is claimed from inaccessible sections.

**[R03] David E. Moriarty and Risto Miikkulainen. “Efficient Reinforcement Learning Through Symbiotic Evolution.” Machine Learning 22, 11–32, 1996.**  
Author record: <https://nn.cs.utexas.edu/?moriarty:mlj96=>; DOI: <https://doi.org/10.1007/BF00114722>. Inspected author/publisher summaries. SANE evolves neuron populations whose combinations form networks. It anticipates selection among cooperating neuronal components, but evolutionary fitness evaluation is not this plan’s online private-credit update. Detailed replication would require the full method, which is not assumed inspected.

**[R04] Philip S. Thomas. “Policy Gradient Coagent Networks.” NeurIPS, 2011.**  
Proceedings: <https://papers.nips.cc/paper_files/paper/2011/hash/1e6e0a04d20f50967c64dac2d639a577-Abstract.html>. Inspected proceedings abstract and metadata. Establishes a direct local-agent policy-gradient lineage; the plan does not infer all theorem assumptions from the abstract.

**[R05] James Kostas, Chris Nota and Philip Thomas. “Asynchronous Coagent Networks.” ICML, PMLR 119, 5426–5435, 2020.**  
Proceedings: <https://proceedings.mlr.press/v119/kostas20a.html>. Inspected primary abstract/record. Relevant to asynchronous interacting local learners; not empirical support for the reserve/rewiring combination or a guarantee under arbitrary asynchronous mutations.

**[R06] James E. Kostas, Scott M. Jordan, Yash Chandak, Georgios Theocharous, Dhawal Gupta, Martha White, Bruno Castro da Silva and Philip S. Thomas. “Coagent Networks: Generalized and Scaled.” 2023.**  
Record: <https://arxiv.org/abs/2305.09838>; PDF: <https://arxiv.org/pdf/2305.09838>. Inspected PDF text and rendered pages, including formulation and experimental context. Supports scalable coagent formulations in specified settings and motivates variance-aware local-learning controls. Published larger control experiments are context, not matched local benchmark numbers.

**[R07] Stephen Chung. “Learning by Competition of Self-Interested Reinforcement Learning Agents.” arXiv submission 2020, revised 2021; AAAI 2022.**  
Record: <https://arxiv.org/abs/2010.09770>; full text: <https://arxiv.org/html/2010.09770>. Inspected main method, experimental setup and appendix discussion. Weight Maximization directly anticipates selfish hidden-unit objectives. Its scope includes stochastic networks and specific control/multiplexer comparisons, not a demonstrated universally strongest learner. The paper’s approximation and executable source behavior require separate fidelity checks.

**[R08] Stephen Chung. Official Weight Maximization implementation, `weight_max`. Accessed 2026.**  
Repository: <https://github.com/stephen-chung-mh/weight_max>; inspected core: <https://raw.githubusercontent.com/stephen-chung-mh/weight_max/main/weightmax.py>; inspected configuration: <https://raw.githubusercontent.com/stephen-chung-mh/weight_max/main/config/config_mp.ini>. Inspected README, core NumPy updates and multiplexer configuration. Informs the closest-prior-art reproduction, source-order distinction, compatibility patches and license record. Source branch was inspected; no immutable commit or successful target-machine execution is asserted here.

**[R09] Jacopo Castellini, Sam Devlin, Frans A. Oliehoek and Rahul Savani. “Difference Rewards Policy Gradients.” arXiv 2020, revised 2023; journal DOI registered with the cited 2022 article record.**  
Record: <https://arxiv.org/abs/2012.11258>; full text: <https://arxiv.org/html/2012.11258v2>; DOI: <https://doi.org/10.1007/s00521-022-07960-5>. Inspected formulation and discussion of counterfactual baselines, including limitations involving action-dependent histories. Supports the credit lineage, not newness. The finite one-step identities here are independently derived/tested rather than imported as a general convergence theorem.

**[R10] Amirata Ghorbani and James Zou. “Neuron Shapley: Discovering the Responsible Neurons.” 2020.**  
Record: <https://arxiv.org/abs/2002.09815>. Inspected primary abstract/metadata. Counterfactual neuron responsibility predates the proposal. Attribution analysis is not itself a paid, self-modifying online neural population; complete Shapley computation is not assumed cheap.

**[R11] Mohamed Elsayed and A. Rupam Mahmood. Utility-based continual-learning papers, 2023–2024.**  
“Utility-based Perturbed Gradient Descent: An Optimizer for Continual Learning”: <https://arxiv.org/html/2302.03281v2>. “Addressing Loss of Plasticity and Catastrophic Forgetting in Continual Learning”: <https://arxiv.org/abs/2404.00781>, full text <https://arxiv.org/html/2404.00781v1>. Inspected utility definitions, update descriptions and experimental scope. Directly relevant to contribution-based protection/perturbation. Do not merge different paper/source variants into an unnamed baseline.

**[R12] Mohamed Elsayed and collaborators. Official `upgd` implementation. Accessed 2026.**  
Repository: <https://github.com/mohmdelsayed/upgd>; inspected update listing: <https://raw.githubusercontent.com/mohmdelsayed/upgd/main/README.md>. Inspected first-order optimizer code, installation notes and MIT notice. Provides the concrete updated normalization/noise rule. The full historical environment includes older dependencies; a small faithful modern adapter still needs fixture and device validation.

**[R13] Shibhansh Dohare, J. Fernando Hernandez-Garcia, Qingfeng Lan, Parash Rahman, A. Rupam Mahmood and Richard S. Sutton. “Loss of plasticity in deep continual learning.” Nature 632, 768–774, 2024.**  
Article: <https://www.nature.com/articles/s41586-024-07711-7>; DOI: <https://doi.org/10.1038/s41586-024-07711-7>. Inspected primary article and experimental framing. Documents loss of plasticity and continual backpropagation in specified long sequences; regularization/perturbation comparisons also matter. Neither universal failure of standard learning nor indefinite success on every future task follows.

**[R14] Shibhansh Dohare and collaborators. Official `loss-of-plasticity` repository. Accessed 2026.**  
Repository: <https://github.com/shibhansh/loss-of-plasticity>; inspected core: <https://raw.githubusercontent.com/shibhansh/loss-of-plasticity/main/lop/algos/gnt.py>. Inspected utility calculation, maturity/replacement, reset and optimizer-state handling, plus repository license. Current source raises the saved-activation-mean ordering question identified in the audit. A pinned historical experiment may differ; this plan does not attribute the concern to all published results.

**[R15] Ghada Sokar, Rishabh Agarwal, Pablo Samuel Castro and Utku Evci. “The Dormant Neuron Phenomenon in Deep Reinforcement Learning.” ICML, PMLR 202, 2023.**  
Proceedings: <https://proceedings.mlr.press/v202/sokar23a.html>; record: <https://arxiv.org/abs/2302.12902>. Inspected primary proceedings summary. ReDo provides a relevant activity-based renewal comparator. Its original reinforcement-learning benchmark is not directly reproduced by a supervised tabular transfer; source-specific details must be pinned before claiming such a reproduction.

**[R16] Patrick Elisii, Lucas Beauchemin and Dawer Jamshed. “Attribution-Based Neuron Utility for Plasticity Restoration in Deep Networks.” 7 May 2026, preprint.**  
Full text: <https://arxiv.org/html/2605.06834v1>; record: <https://arxiv.org/abs/2605.06834>. Inspected attribution/reference definition, replacement relation and experiments. Directly challenges novelty in contribution-based neuron renewal. Its target-logit, mean-reference utility differs from exact loss interventions; official code and independent replication were not verified.

**[R17] Mohammad Ali Vahedifar, Abhisek Ray and Qi Zhang. “Shapley Neuron Values for Continual Learning: Which Neurons Matter Most?” 15 May 2026, preprint.**  
Record: <https://arxiv.org/abs/2605.15877>; full text: <https://arxiv.org/html/2605.15877v1>. Inspected record and broad method framing in full text. Importance-based protection/freezing further narrows novelty, but its continual-learning objective is not identical to online renewal under drift. Detailed source reproduction and independent verification remain unperformed.

**[R18] Utku Evci, Trevor Gale, Jacob Menick, Pablo Samuel Castro and Erich Elsen. “Rigging the Lottery: Making All Tickets Winners.” ICML, PMLR 119, 2943–2952, 2020.**  
Proceedings: <https://proceedings.mlr.press/v119/evci20a.html>. Inspected primary record and algorithm description. RigL is the relevant gradient-based sparse-topology alternative. Its original finite-training schedule and hardware results are not automatically transferable to a continuous stream or dense masked MPS implementation.

**[R19] Guillaume Bellec, David Kappel, Wolfgang Maass and Robert Legenstein. “Deep Rewiring: Training very sparse deep networks.” arXiv 2017; ICLR 2018.**  
Record: <https://arxiv.org/abs/1711.05136>. Inspected primary abstract. Demonstrates that tightly constrained changing connectivity is established prior art; it does not supply the selected reserve mechanism or a matched current benchmark.

**[R20] J. Zico Kolter and Marcus A. Maloof. “Dynamic Weighted Majority: An Ensemble Method for Drifting Concepts.” JMLR 8, 2007.**  
Article record: <https://jmlr.org/papers/v8/kolter07a.html>. Inspected primary summary. Useful expert addition/deletion and reweighting provide a simpler conceptual alternative. The stronger local tabular comparison uses a currently maintained adaptive-forest implementation rather than treating this historical ensemble as the sole competitor.

**[R21] Abhimanyu Pallavi Sudhir and Long Tran-Thanh. “Market-based Architectures in RL and Beyond.” AAMAS Blue Sky / preprint, 2025.**  
Full text: <https://arxiv.org/html/2503.05828v1>. Inspected conceptual architecture discussion and historical framing. Relevant to search coverage and decentralized-market alternatives; it is not evidence of a proven competitive neuron economy.

**[R22] Alexander Vergara. “Gas Sensor Array Drift Dataset.” UCI Machine Learning Repository, 2012, with recorded later correction.**  
Dataset page: <https://archive.ics.uci.edu/dataset/224/gas+sensor+array+drift+dataset>; DOI: <https://doi.org/10.24432/C5RP6W>. Inspected metadata, feature/batch description, correction notice and conflicting use/license statements. Raw files were not downloaded or verified here. Enables real sensor-drift evidence only after the license/provenance gate; within-batch chronological order remains unverified.

**[R23] Han Xiao, Kashif Rasul and Roland Vollgraf / Zalando Research. “Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms.” 2017.**  
Official data/code repository: <https://github.com/zalandoresearch/fashion-mnist>. Inspected dataset specification, canonical files and MIT license identification. Supports a locally sized image-content transfer task; artificial permutations do not establish natural-world drift. Downloads and checksums remain local prerequisites.

**[R24] Apple Developer. “Accelerated PyTorch training on Mac.” Accessed 12 October 2026.**  
Documentation: <https://developer.apple.com/metal/pytorch/>. Inspected requirements and MPS installation guidance. Supports a plausible Apple Silicon stack, not actual support for every operation in this project. The release displayed there differed from the version reached by PyTorch documentation; local wheel validation is authoritative.

**[R25] PyTorch contributors. MPS documentation. Accessed 12 October 2026.**  
Stable entry: <https://docs.pytorch.org/docs/stable/mps.html>; inspected version reached during browsing: <https://docs.pytorch.org/docs/2.14/mps.html>; backend notes: <https://docs.pytorch.org/docs/stable/notes/mps.html>. Inspected synchronization, RNG and allocation-reporting API documentation. Installed availability and numerical fidelity must be verified; documentation is not an M3 measurement.

**[R26] scikit-learn contributors. `SGDClassifier` API documentation. Accessed 12 October 2026.**  
Documentation: <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html>. Inspected online classification API and parameter documentation. Defines a reusable simple baseline; exact installed version, class initialization, schedules and probability behavior must be recorded.

**[R27] River contributors. `forest.ARFClassifier` documentation and source license. Accessed 12 October 2026.**  
Documentation: <https://riverml.xyz/latest/api/forest/ARFClassifier/>; license: <https://github.com/online-ml/river/blob/main/LICENSE>. Inspected API, ensemble/memory parameters and BSD 3-Clause license. A maintained nonlinear online drift comparator; API defaults and background-tree memory are versioned rather than assumed constant.

**[R28] J. Fernando Hernandez-Garcia, Shibhansh Dohare, Jun Luo and Richard S. Sutton. “Reinitializing weights vs units for maintaining plasticity in neural networks.” CoLLAs, 2025.**  
Record: <https://arxiv.org/abs/2508.00212>; full text: <https://arxiv.org/html/2508.00212v1>; official code: <https://github.com/JFernando4/collas_2025_swr_paper>. Inspected method, utility/reset choices, small-network/normalization findings, and repository README/license. SWR is a mandatory modern comparator. Repository macOS compatibility claims do not establish this project’s local compatibility; the specific source functions still require pinned fixture validation. The source function file was not successfully retrieved during this investigation, so no core-source inspection is claimed for this repository.

## Appendix A. Small validation checks actually executed

### A.1 Scope, environment and results

Executed during this investigation in a Linux x86_64 environment with Python 3.13.5 and NumPy 2.3.5. No training benchmark, dataset experiment, source-baseline reproduction or Mac performance test was executed. The following checks are deliberately small enough to audit independently.

Exhaustive enumeration of a three-Bernoulli-unit, two-class model compared the expected shared-reward, zero-ablation difference-reward and conditional-gradient estimators with central finite differences of expected reward, including a firing cost. Maximum absolute discrepancies were approximately `1.27×10^-10`. A constructed two-layer relay gave the correct derivative `0.19556664935259668` with descendant interventions, whereas incorrectly freezing the descendant gave zero. A discounted-reserve closed-form identity agreed to approximately `2.22×10^-16`. OR/AND games showed why marginal payouts are not a conserved budget.

The estimator variance values below apply to **this chosen finite-state example only**. They are not a comparative-learning result, an empirical variance bound, or evidence of a novel estimator. Array-size outputs are arithmetic estimates, not process-memory measurements.

```json
{
  "environment": {
    "os": "Linux",
    "machine": "x86_64",
    "python": "3.13.5",
    "numpy": "2.3.5"
  },
  "max_abs_gradient_error_vs_finite_difference": {
    "shared": 1.2701638352208278e-10,
    "diff": 1.270164251554462e-10,
    "rb": 1.270164251554462e-10
  },
  "gradient_variance_trace_this_example_only": {
    "shared": 1.0525330442873566,
    "mean_baseline": 0.11449488579022986,
    "diff": 0.0750005201328318,
    "rb": 0.0012444629583379468
  },
  "two_layer_correct_gradient": 0.19556664935259668,
  "two_layer_intervention_gradient": 0.19556664935259668,
  "two_layer_wrong_frozen_descendant_gradient": 0.0,
  "reserve_closed_form_abs_error": 2.220446049250313e-16,
  "nonconserving_payouts": {
    "OR": {
      "team_reward": 1,
      "credit_sum": 0
    },
    "AND": {
      "team_reward": 1,
      "credit_sum": 2
    }
  },
  "memory_estimate_bytes": {
    "parameters_float32": 540024,
    "adam_weights_grads_two_moments": 2160096,
    "counterfactual_logits_BNC": 1536000
  }
}
```

### A.2 Complete reproducible check script

Save the following as `check_mechanism.py` and run `python check_mechanism.py` in an environment with NumPy. The output includes the executing environment, so those fields will differ on the target laptop. Convert these assertions to repository tests rather than treating this standalone script as the complete model.

```python
"""Finite-state checks for the proposed plan; not a training benchmark."""
import itertools, json, platform, sys
import numpy as np

sigmoid = lambda x: 1.0 / (1.0 + np.exp(-x))
a = np.array([-0.7, 0.2, 1.1], dtype=np.float64)
v = np.array([[0.8,-0.3],[-0.2,0.6],[0.4,-0.1]],dtype=np.float64)
b = np.array([0.1,-0.2]); firing_cost = 0.03
states = np.array(list(itertools.product([0., 1.], repeat=3)))
def reward(h):
    z = b + h @ v
    return float(z[1] - np.logaddexp.reduce(z) - firing_cost*h.sum())
def exact(a):
    p = sigmoid(a)
    weights = np.prod(np.where(states == 1., p, 1.-p),axis=1)
    rs = np.array([reward(h) for h in states])
    scores = states-p
    shared = rs[:,None]*scores
    diff = np.empty_like(shared); rb = np.empty_like(shared)
    for t,h in enumerate(states):
        for i in range(3):
            zero=h.copy(); zero[i]=0
            one=h.copy(); one[i]=1
            diff[t,i]=(rs[t]-reward(zero))*scores[t,i]
            rb[t,i]=p[i]*(1-p[i])*(reward(one)-reward(zero))
    mean_reward=float(weights@rs)
    constant=(rs-mean_reward)[:,None]*scores
    means = {k:weights@g for k,g in [('shared',shared),('diff',diff),('rb',rb)]}
    var = {k:float(weights @ np.sum((g-weights@g)**2,axis=1))
           for k,g in [('shared',shared),('mean_baseline',constant),('diff',diff),('rb',rb)]}
    return mean_reward,means,var
f,means,variance=exact(a)
eps=1e-6
fd=np.array([(exact(a+eps*np.eye(3)[i])[0]-exact(a-eps*np.eye(3)[i])[0])/(2*eps) for i in range(3)])
errors={k:float(np.max(abs(g-fd))) for k,g in means.items()}
assert max(errors.values())<1e-8
# Two-layer chain: h~Bernoulli(p), b~Bernoulli(.1+.8h), R=b.
p=float(sigmoid(np.array(.3))); chain_exact=.8*p*(1-p)
chain_diff=0.
for h,ph in [(0.,1-p),(1.,p)]:
    for u,pu in [(0.05,.1),(.5,.8),(.95,.1)]:
        actual=float(u < .1+.8*h)
        cf=float(u<.1)
        chain_diff+=ph*pu*(actual-cf)*(h-p)
assert abs(chain_exact-chain_diff)<1e-12
# Removing a parent but holding b fixed produces no credit, incorrectly.
chain_naive=0.
# Reserve recurrence equals discounted utility, until a reset changes initial state.
rng=np.random.default_rng(42); u=rng.normal(size=23); rho=.93; E0=1.4
E=E0
for value in u: E=rho*E+value
closed=rho**len(u)*E0+sum(rho**(len(u)-1-t)*value for t,value in enumerate(u))
assert abs(E-closed)<1e-12
# At (1,1): zero-ablation credits sum to zero for OR, two for AND.
credits={}
for name,fun in [('OR',lambda x,y:max(x,y)),('AND',lambda x,y:x*y)]:
    actual=fun(1,1)
    credits[name]={'team_reward':actual,'credit_sum':(actual-fun(0,1))+(actual-fun(1,0))}
# Derived memory arithmetic, not measured resident memory.
N,d,C,B=1000,128,6,64
P=N*d+N+N*C+C
result={
 'environment':{'os':platform.system(),'machine':platform.machine(),'python':sys.version.split()[0],'numpy':np.__version__},
 'max_abs_gradient_error_vs_finite_difference':errors,
 'gradient_variance_trace_this_example_only':variance,
 'two_layer_correct_gradient':chain_exact,'two_layer_intervention_gradient':chain_diff,'two_layer_wrong_frozen_descendant_gradient':chain_naive,
 'reserve_closed_form_abs_error':abs(E-closed),'nonconserving_payouts':credits,
 'memory_estimate_bytes':{'parameters_float32':4*P,'adam_weights_grads_two_moments':16*P,'counterfactual_logits_BNC':4*B*N*C}}
print(json.dumps(result,indent=2))
```

## Appendix B. Additional implementation conventions

These conventions complete the local adapters. Source-faithful reproduction always retains its pinned original semantics; a changed convention is a named project transfer, not an unnoticed correction to a paper.

### B.1 Resettable Adam and moving averages

For an ascent direction `g`, coordinate moments update as `m=β1*m+(1-β1)g`, `v=β2*v+(1-β2)g²`; increment that coordinate’s update counter `t`, bias-correct with `1-β1^t` and `1-β2^t`, and use `+η*m_hat/(sqrt(v_hat)+ε_opt)`. Negate a loss gradient for descent. Initial reference values are `β1=0.9`, `β2=0.999`, `ε_opt=10^-8`, with development tuning disclosed. The optional hidden-weight decay is decoupled as `(1-ηλ_w)W` in the Adam transfer; SGD follows §4 directly. Do not apply undeclared decay to biases or the readout. A conventional separately tuned readout decay is allowed only as an explicitly configured shared regularizer.

Coordinate counters measure **optimizer updates**, not observations. Unit maturity and account decay measure **observations**. Reset incident coordinates’ moments and counters to zero while leaving unrelated coordinates unchanged. Masked parameters and their moments remain zero; adding an edge starts a fresh coordinate state. Numerically compare the no-reset case with an ordinary Adam implementation and compare a reset coordinate with a newly constructed scalar optimizer.

For an observation-timescale EMA, update `m_b=exp(-B_b/T)*m_previous+(1-exp(-B_b/T))*batch_mean`; its initialization bias factor is `1-exp(-age_observations/T)` when every observation since birth has been included. Do not replace it with an optimizer-step age accidentally. An empty stream leaves initialization unchanged; no division by a zero-age bias factor is permitted.

### B.2 Local utility-baseline formulas

For a unit with hidden activation `h_i`, incoming row `w_i` and outgoing column `v_i`, the selected CBP-style contribution uses `mean(abs(v_i))*mean(abs(h_i))`; the adaptable contribution uses `mean(abs(v_i))*mean(abs(h_i-r_i))/mean(abs(w_i))`, followed by its configured EMA/bias correction. Use the saved running mean `r_i` for compensated resets. A zero incoming-magnitude denominator must produce a logged, explicit bounded-denominator convention in the local adapter, not an infinity. Include the simpler contribution version to expose dependence on that convention. [R14]

The updated first-order UPGD reference can be expressed for each parameter coordinate `θ_j` with loss gradient `g_j` as

\[
 u_j\leftarrow\beta_u u_j+(1-\beta_u)(-\theta_jg_j),\quad
 U=\max_j u_j,\quad
 s_j=\sigma\left(\frac{u_j/(1-\beta_u^{t_j})}{U}\right),
\]
\[
 \theta_j\leftarrow(1-\eta\lambda)\theta_j
 -2\eta(g_j+\xi_j)(1-s_j),\quad \xi_j\sim N(0,\sigma_\xi^2).
\]

The maximum in this inspected source is over uncorrected utilities. Preserve that detail and use source/default settings as tuning candidates. Compute all utilities before parameter changes. If `|U|` is below machine-epsilon times the declared utility scale, the local safe-edge convention is `s_j=1/2`; log its frequency and test the all-zero case. Negative nonzero maxima retain the source formula rather than being silently changed to absolute maxima. This baseline uses global loss gradients and is not advertised as backpropagation-free. [R12]

For SWR’s local threshold variant, evaluate `u_j=|θ_j|` or `u_j=|θ_j g_j|` on the declared pre-update snapshot, compute a mean over the eligible parameter tensor, and reset coordinates below `k*mean(u)` at every configured interval. A fixed-fraction variant sorts coordinates by utility and resets a declared fraction. Tensor scope, inclusion of biases, interval and replacement distribution are configuration fields; use the pinned reference for a source-faithful run. Local adapters must not reset normalizer parameters by accident. When all utilities are exactly zero, a strict threshold selects none; a separately configured fixed-fraction rule retains its defined count. [R28]

For GXD, use §5.3 exactly. For ReDo, an all-zero activation denominator gives zero scores and is treated as a dormant layer only after the baseline’s declared maturity/interval protection; this is an edge-case convention requiring a startup fixture. For all rank-based selectors, stable-ID tie breaking is deterministic; an explicitly randomized tie policy requires its own RNG and must be shared by matched controls.

### B.3 Baseline hyperparameters and fairness checklist

Record a full schedule, not just a learning rate: update interval, utility timescale, maturity, replacement count rule, initialization family, clipping, gradient normalization, replay, bias treatment, optimizer-state reset and source version. Run the official/default configuration and a fair local development-selected configuration. A baseline’s required fields cannot be left at guessed modern defaults during confirmation.

For adaptive forests, serialize the entire estimator, including drift detectors, background learners and random state, and validate recovery on an identical suffix. For online linear models, pass the complete predefined class inventory at the first update; do not learn class identity from protected labels before it appears unless the same metadata is available to all methods. No model receives synthetic task boundaries.

For constant-budget practical comparisons, let a cheaper baseline spend its saved adaptation allowance on extra updates over **already revealed** buffer samples, larger capacity or additional prediction draws, with those choices fixed on development data. Also retain its simplest one-pass configuration. Show both data-matched and complete-cost-matched results; neither contract alone answers every claim.

### B.4 Final report contract

The final experimental report must start with one of: confirmed narrow improvement; replicated known mechanism; useful engineering optimization; equivalence/no additional mechanism; precise negative result in the tested regime; or inconclusive at achieved precision. It must then state the tested population, primary effect and uncertainty, best baseline/configuration, full time/memory tradeoff, important regressions, locality/oracle assumptions, confirmation status and missing gates.

Attach machine-readable tables for all valid settings and a failure ledger. Do not label a result “groundbreaking” based on an acronym, an interesting visualization, one favorable seed, a toy task, or the mere fact that individual units have a numerical survival account. A surprising positive result deserves stronger attempts to explain it with the simpler controls already specified here.

**End of standalone handoff.**
