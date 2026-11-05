# POC_EXECUTION_BRIEF

## 1. Project decision and authority

**To the autonomous coding agent: execute this brief using `gpt-6.1-sol` with `xhigh` reasoning.** Implement and run the authorized stages on one MacBook with an M3 Max, follow the evidence gates, and produce an honest research result. This document is the execution plan, not a report of an already completed neural-network PoC.

**Selected project:** determine whether an inexpensive, online predictor of parameter gradients, using a short history of oriented exact gradients and a learned query policy, reduces actual elapsed training time to a fixed validation quality on a small causal Transformer.

**Decision to inform:** is this particular approach worth a larger neural-training investigation, or do unavailable gradient information, prediction error, correction variance, or local execution overhead already make it unattractive?

The smallest defensible contribution is a **controlled local feasibility or limitation result** for causal gradient prediction under a precisely specified information and cost model. The candidate contribution is the measured relationship among history-span coverage, prediction error, adaptive query allocation, and time to quality. Learning gradients, recombining gradient history, maintaining coordinates, and using control variates are established ideas. Do not claim that this combination is globally novel or that it is a new theorem.

### 1.1 Primary question and success criterion

> On the specified WikiText-2 byte-language-modeling workload, does the four-gradient neural predictor with residual-driven randomized querying achieve at least a 10% reduction in synchronized end-to-end time to a frozen validation-loss target, compared with AdamW and the strongest development-selected simple/linear predictor, while using at least 25% fewer exact backward passes to that target and preserving held-out test quality?

The primary method is named **NH4-A**, a descriptive experimental identifier:

- **N:** a small neural coefficient predictor.
- **H4:** four previously queried exact gradients per parameter tensor.
- **A:** adaptive probability of querying the current exact gradient.

Use the residual-corrected estimator specified below. Do not switch to a different estimator after seeing disappointing results.

A supported result requires the complete criteria in Section 10. A point estimate, a smaller backward count, or good gradient cosine alone is insufficient. A valid negative or infeasibility result completes the project. A noisy pilot or a broken implementation does not refute the hypothesis.

### 1.2 Revisions to the original proposal

| Original direction | Selected scope and reason | Excluded conclusion |
|---|---|---|
| Large LLM | Four-layer, approximately 3.26-million-parameter causal Transformer; one resource-only reduction to approximately 1.85 million parameters is allowed before comparisons | No billion-parameter scaling, reasoning-capability, or production-cost claim |
| Treat the attached theorem as support for predicting gradients | Use it only to identify information and representation assumptions; it assumes the current signal is already supplied | No inherited prediction, convergence, or runtime guarantee |
| Explicitly transport an evolving orthonormal gradient basis | Store four oriented gradients directly in fixed parameter coordinates; their span evolves implicitly | No claim to invent frame transport or to validate a higher-rank extension of the theorem |
| Alternate cheap updates and exact-gradient updates | Randomly query exact gradients and form a residual-corrected estimator; queried steps are generally not exact-gradient updates | No description of the corrected branch as ordinary exact backpropagation training |
| Learned layerwise decisions and recursive learners | One whole-model query decision; one small online learner; ordinary ambient AdamW | No layerwise-backward, recursive self-improvement, or meta-training study |
| Broad practical advance over low-rank optimizers | Include a specified GaLore reference, but make only a local claim against methods actually evaluated | No superiority claim over InGaLore, GoLore, GaLore 2, ADA-GP, or PGD without faithful comparisons |

This is a substantive narrowing, not an attempt to obtain a favorable toy result. The real-text, fresh-minibatch, closed-loop training comparison is mandatory for an empirical training-efficiency claim. Synthetic examples are correctness controls only.

### 1.3 Fixed requirements versus engineering discretion

Fixed: estimator and information order; data source and byte encoding; whole-model query; model family and minimum size; comparator roles; tuning allowance; quality criterion; fresh confirmation seeds; timing boundaries; failure handling; cumulative caps; scope of claims.

You may choose ordinary engineering details such as function names inside the prescribed modules, efficient tensor layouts, buffered logging, and supported attention implementation. Record material choices. You may repair ordinary defects within the repair allowance. Do not change the scientific hypothesis, evaluation distribution, effect threshold, comparator selection rule, or estimator to rescue results.

No remote experimental compute, CUDA-only dependency, paid computation, proprietary dataset, distributed service, dashboard, or experiment-management server is needed or authorized by this brief. Hosted coding/research agents and normal dependency/data downloads are separate from experimental compute.

---

## 2. Evidence and mathematical audit

### 2.1 Input provenance and scope of examination

The supplied archive is `neural_feedback_research(1).zip`, SHA-256:

`60bfb6cec3d92d98a3c6d26476434bd45fd29c73e09b6d48095d5e5a1ae2b985`.

The planning instruction was `Pasted text(20261104-115844).txt`, SHA-256:

`171a2c8a2b161bb51b978751f0fae93c985cdf677357aae8ed1e2ab735663581`.

All supplied research files were examined: `README.txt`, `feedback_research_note.tex`, its 12-page PDF, `spectral_proxy_appendix.tex`, both Python experiment scripts, both JSON result files, and `feedback_results/feedback_tradeoff.png`. The PDF's critical theorem, memory, and appendix pages and the figure were inspected visually. Source code and reported numerical results were checked against the prose.

If the archive accompanies this brief, retain it unchanged under `inputs/` and extract into `inputs/source_note/`. The algorithm below does not require the paper as an executable dependency: the necessary definitions and decisions are included here. If the archive is unavailable to you, record that you could not independently rerun its original programs; do not invent a replacement paper. Proceed with the independently specified PoC and tiny checks.

The source author field is “Mathematical investigation prepared in this conversation,” dated 4 November 2026. Treat it as an unpublished supplied research note, not an independently published or formally verified theorem.

### 2.2 What the main theorem says

Let integers satisfy \(1\le k<d\), let \(x\in S^{d-1}\subset\mathbb R^d\), and let \(P_x=xx^\top\). Define \(\mathcal P_d=\{xx^\top:x\in S^{d-1}\}\), the set of rank-one orthogonal projectors. The admissible maps are continuous:

\[
E:\mathcal P_d\times\mathbb R^d\to\mathbb R^k,\qquad
D:\mathcal P_d\times\mathbb R^k\to\mathbb R^d,
\quad
u(x)=D(P_x,E(P_x,x)).
\]

**The encoder already knows the oriented current signal \(x\); the decoder already knows its exact line \(P_x\).** The composite update has geodesic Lipschitz constant at most \(K\ge1\):

\[
\|u(x)-u(y)\|\le K\arccos(x^\top y).
\]

For uniform surface measure, the two risks are

\[
R_{\rm vec}=\mathbb E\|u(X)-X\|^2,\qquad
R_{\rm loss}=\mathbb E(1-X^\top u(X))^2.
\]

The latter is twice the expected remaining loss after one update from zero on

\[
f_x(\theta)=\tfrac12(x^\top\theta-1)^2,\qquad
\nabla f_x(0)=-x,\quad \nabla^2 f_x=P_x.
\]

The source's Theorem 2 bounds the infimum of either risk between

\[
L_k(K)=\frac{2}{B(k/2,1/2)}
 \int_0^{1/K}(1-K\rho)^2\sin^{k-1}\rho\,d\rho
\]

and

\[
U_{d,k}(K)=\frac{2}{B(k/2,(d-k)/2)}
 \int_0^{1/K}(1-Kr)^2r^{k-1}(1-r^2)^{(d-k)/2-1}\,dr.
\]

For fixed \(d,k\), this gives \(\Theta_{d,k}(K^{-k})\). **Only when \(d=k+1\)** is the sharp leading coefficient established:

\[
C_k=\frac{4}{B(k/2,1/2)k(k+1)(k+2)},\qquad
\mathcal R_{k+1,k}(K)=C_kK^{-k}+O_k(K^{-k-2}).
\]

An explicit construction uses \(Q\in\mathbb R^{d\times k}\), \(Q^\top Q=I\), message \(m=Q^\top x\), and yields \(u_K(x)=\min(1,K\|Q^\top x\|)x\). The globally continuous decoder extension is given in source equation (13). The proof uses Borsuk-Ulam, odd/even decomposition, averaging over great spheres, and beta-distributed projection length. No essential defect was found in these arguments under the stated assumptions. This is an audit finding, not a formal proof-assistant certificate. [S1, Proposition 1 and Theorem 2, §§2–4, pp.2–5, equations (1)–(18)]

Theorem 3 separately assumes an \(L_E\)-Lipschitz message and an \(L_D\)-Lipschitz decoder **in the message at fixed \(P\)**. It proves

\[
R\ge\left(1-L_EL_D\sqrt{\frac{\min(k,d-1)}{d-1}}\right)_+^2.
\]

It does not justify substituting \(K=L_EL_D\) in the composite theorem without controlling the decoder's dependence on \(P\). [S1, §5, pp.5–6, equations (19)–(20)]

### 2.3 What memory and the appendix establish

| Source result | Exact applicability | Consequence for this PoC |
|---|---|---|
| Discrete chart, equation (22) | Given current rank-one \(P\), choose \(j=\arg\max P_{jj}\), \(q=Pe_j/\sqrt{P_{jj}}\), transmit \(q^\top g\) for a current \(g\in\operatorname{im}P\) | Exact reconstruction uses current information and discontinuous routing; it does not predict \(g\) |
| Retained orientation, equation (23) | Exact rank-one projectors with \(\|P_{t+1}-P_t\|_{\rm op}\le\delta<1\); \(q_{t+1}=P_{t+1}q_t/\|P_{t+1}q_t\|\) | Denominator at least \(\sqrt{1-\delta^2}\); requires current \(P\), \(d\)-coordinate state, and maintenance |
| Randomization, equation (21) | Each seed independently specifies an admissible continuous scheme with the same composite stability bound | Does not bound arbitrary stateful/adaptive learners |
| Exact quadratic proxy, Theorem 4 | Fixed symmetric quadratic, zero shifted start, \(T\) queried gradients, scalar-span controllers, Euclidean Gram/loss interface, basis independent of controller parameters | \(\min(T,r)\)-dimensional Krylov proxy is exact for that interface; construction costs up to \(T\) Hessian-vector products |
| Analytic obstruction, Theorem 5 | One fixed differentiable proxy must reproduce every three-query controller's exact loss/gradient-Gram interface | An exact universal lower bound, not a robust approximate bound or a bound for one finite trajectory |

Locations: routing §6.2 p.6; memory §6.3 p.7; appendix Theorem 4 p.10 and Theorem 5 pp.11–12. The quadratic result is a classical Krylov/Lanczos/Gauss-quadrature consequence [S1, S10]. Coordinatewise Adam operations, a changing neural loss, arbitrary long horizons, and learning the basis jointly with controller parameters fall outside it.

The nonlinear appendix example is globally well-conditioned in exact arithmetic, but its perturbation decreases as \(O(n^{-3})\). Its supplied numerical rank check uses discrete-sine points extending outside the monotone-controller interval. A planning probe using 12 equally spaced points in \([-.99,-.01]\) had smallest singular value about \(3.95\times10^{-16}\) and default float64 numerical rank 11. This does not contradict its exact analytic rank proof; it makes an approximate or finite-precision interpretation particularly inappropriate.

### 2.4 The bridge that does not exist

The original theorem studies **communication after a signal is known**. Computing \(P_x\) or \(m=Q^\top x\) from a fresh neural gradient would already spend the derivative work this project hopes to avoid. The free projector has intrinsically \(d-1\) degrees of freedom; it is substantial side information. The theorem does not price acquiring it, applying it, or storing its frame.

Even a constant, perfectly known line does not reveal an unpredictable scalar coefficient in that line. Conversely, a small network producing a full-dimensional update is not necessarily a narrow communication channel. No theorem here bounds its parameter count or establishes its predictive quality.

**Do not compute current gradients, Jacobians, Hessian-vector products, fresh gradient-derived projectors, or per-example gradients on an unqueried training step.** Forward loss/activation summaries are permitted and charged. Current exact gradients are permitted only after the query decision.

The surviving guarantees for the implementation are elementary algebraic identities described next. Predictability, runtime benefit, neural stability, and useful confidence calibration are empirical hypotheses.

---

## 3. The mathematical mechanism to implement

### 3.1 State and prediction

Let \(\theta_t\in\mathbb R^P\) contain all trainable parameters. The current independent minibatch is \(B_t\), its mean next-byte cross-entropy is \(\ell_t(\theta_t;B_t)\), and its exact gradient, when queried, is

\[
g_t=\nabla_\theta\ell_t(\theta_t;B_t).
\]

Group parameters by unique trainable tensor, including biases and normalization parameters. Do not duplicate tied embedding weights. Write group size \(P_b\), with \(\sum_bP_b=P\).

For each group retain the most recent \(r=4\) **exact, raw, unclipped** queried gradient vectors and their query-step indices. Represent them as unit columns \(v_{b,j}\), norms \(n_{b,j}\), and valid masks, ordered newest first. Zero-norm columns are zero vectors with norm zero. Let

\[
V_b=[v_{b,1},\ldots,v_{b,r}]\in\mathbb R^{P_b\times r},
\quad s_b=\max(n_{b,1},10^{-8}).
\]

The controller predicts coefficients \(a_b\in[-2,2]^r\). Define

\[
d_b=V_ba_b,\qquad
\kappa_b=\min\left(1,\frac{2}{\max(\|d_b\|,10^{-8})}\right),
\quad h_b=s_b\kappa_bd_b .
\]

Missing slots have coefficient zero. Thus \(\|h_b\|\le2s_b\). This prediction cap is part of the fixed algorithm and is shared by learned, affine, and extrapolation predictors where applicable.

The vector \(h_t\) is the concatenation of \(h_b\). History retains orientation in fixed parameter coordinates. No \(P\times P\) projector, parameter-space SVD, learned full-vector readout matrix, or large controller autograd graph is needed.

### 3.2 Randomized exact-gradient query and correction

After computing and detaching current forward features, fix \(h_t\) and probability \(p_t\in[.25,1]\). Draw a fresh independent \(U_t\sim{\rm Uniform}[0,1)\), set \(A_t=1[U_t<p_t]\), and form

\[
z_t=h_t+\frac{A_t}{p_t}(g_t-h_t).
\tag{M1}
\]

For \(A_t=0\), do not compute \(g_t\); use \(z_t=h_t\). For \(A_t=1\), compute the complete gradient once and use the cached pre-query \(h_t,p_t\). Implement the \(p_t=1\) branch directly as \(z_t=g_t\), avoiding float32 cancellation in \(h_t+(g_t-h_t)\).

Conditional on the current weights, batch, forward randomness, and pre-query predictor state, \(g_t,h_t,p_t\) are fixed for the fresh coin. Therefore

\[
\mathbb E_A[z_t]=g_t,\qquad
\mathbb E_A\|z_t-g_t\|^2
=\frac{1-p_t}{p_t}\|g_t-h_t\|^2.
\tag{M2}
\]

These are direct derivations, not consequences of the source theorem. Current-loss-dependent querying is allowed. Updating the prediction after seeing \(g_t\), then substituting it into (M1), is not allowed.

At \(p=.25\), the conditional estimator MSE is three times the predictor residual MSE. At \(p=1\), \(z=g\). When \(A=1,p<1\), the update input is generally **not** the exact gradient. Call this a “queried step,” not an “exact update.”

Use ordinary full-dimensional AdamW on the globally clipped \(z_t\). **Only raw \(z_t\) has the stated unbiasedness.** Clipping, Adam's moments/preconditioning, and weight decay do not produce an unbiased population-gradient update. No Transformer convergence guarantee follows.

This is a one-batch Bernoulli residual estimator. It is not PGD's difference of predicted means on two independently sampled subbatches. A batch-independent stale predictor would cancel in that different construction; it does not cancel in (M1).

### 3.3 A small, fully specified controller

Use one CPU float32 controller shared across parameter groups: a `Linear(30,32)` trunk followed by `tanh`, a `Linear(32,4)` coefficient head, and a separate `Linear(32,1)` risk head. The coefficient head receives the hidden representation normally; the risk head receives **`hidden.detach()`**. Both heads use the same hidden values, but risk supervision cannot backpropagate into the trunk.

The four coefficient outputs give \(a_j=2\tanh(o_j)\); the risk output gives \(\widehat e_b=\operatorname{softplus}(o_5)+10^{-4}\). This is 1,157 trainable scalars. The risk head supplies a positive normalized residual-risk score. Its log-Huber training below targets a robust center of log residual energy, not a guaranteed estimate of the arithmetic conditional mean \(\mathbb E[\|g_b-h_b\|^2/s_b^2\mid\text{features}]\).

Initialize trunk weights with seeded Xavier-uniform initialization (gain 1) and zero trunk bias. Set both head weights to zero, coefficient biases to \((\operatorname{atanh}(.5),0,0,0)\), and risk bias to \(\log(\exp(1)-1)\). Initial prediction is the newest cached gradient, up to the stated \(10^{-8}\) normalization floor and cap.

The 30 detached features, in this exact order, are:

| Features | Count | Definition |
|---|---:|---|
| Slot ages | 4 | \(\min((t-q_j)/32,1)\); zero for invalid slots |
| Relative gradient norms | 4 | \(\operatorname{clip}(\log((n_j+10^{-8})/(s_b+10^{-8})),-4,4)/4\) |
| Valid masks | 4 | 0 or 1 |
| History Gram entries | 10 | Upper triangle including diagonal of \(C_b=V_b^\top V_b\), row-major order |
| Reference norm | 1 | \(\operatorname{clip}(\log(s_b),-20,10)/10\) |
| Current loss | 1 | Current batch mean loss divided by \(\log256\) |
| Loss deviation | 1 | \(\operatorname{clip}((\ell_t-\bar\ell_{t-1})/(\bar\ell_{t-1}+10^{-8}),-1,1)\) |
| Activation deviation | 1 | \(\operatorname{clip}(R_t/(\bar R_{t-1}+10^{-8})-1,-1,1)\) |
| Learning-rate ratio | 1 | Current model learning rate divided by selected base rate |
| Progress | 1 | \(t/N_{\max}\), clipped at 1 |
| Previous observed error | 1 | Last queried normalized residual, clipped to \([0,4]\), divided by 4; initialized to 1/4 |
| Time since last query | 1 | \(\min((t-q_1)/8,1)\) |

Here \(R_t\) is the RMS of the last residual-stream hidden activation **before** the final LayerNorm over batch, position, and channel. It is a single global scalar reused for all groups. Use exponential averages with decay .95, updated **after** the prediction/decision; initialize them from the first forward. Using one global activation summary avoids a hook framework and repeated per-layer synchronization.

Keep \(C_b\) cached between exact queries. Compute norms and Gram entries with torch float32 on MPS, batch small feature tensors into **one CPU transfer per forward**, run the tiny controller on CPU with one torch CPU thread, and transfer coefficients back in one batch. This deliberate device boundary is allowed and must be timed. Avoid `.item()` inside parameter loops.

### 3.4 Query policy

Aggregate group risk with its scale:

\[
\widehat R_t=
\frac{\sum_b s_b^2\widehat e_b}
     {\sum_b s_b^2+10^{-16}},
\qquad
p_t=\operatorname{clip}\left(\sqrt{\widehat R_t/\lambda},.25,1\right).
\tag{M3}
\]

Force \(p_t=1\) for the first 64 optimizer updates, while no history exists, or after seven consecutive unqueried updates. Any nonfinite feature/prediction before querying forces \(p=1,h=0\) for that step and records a numerical warning. Mark its cached supervision invalid and skip that controller update; otherwise invalid features would corrupt the learner. Still insert a finite exact gradient into history. Two such events within 32 updates end the attempt. Repeated nonfinite controller output is a run failure, not an indefinitely tolerated fallback.

The square-root rule minimizes a local surrogate

\[
(1/p-1)R+\lambda p
\]

when \(R\) is the true conditional mean normalized residual energy and the local costs are fixed. The log-Huber-trained score is not that guaranteed mean-energy estimate; substituting it into this square-root rule is heuristic. Calibrating \(\lambda\) matches query probability, not statistical risk calibration. No global policy optimality or certified error bound follows. The normalization makes \(\lambda\) dimensionless; backward cost is absorbed into this calibrated cost penalty.

**Calibrate \(\lambda\) once on development, then freeze it across confirmation runs.** Use the development NH4-A run with \(\lambda=4\); after its first 256 steps, take its last 128 pre-query \(\widehat R_t\) values (all must be finite; otherwise mark that trial's calibration incomplete) and solve by bisection for a new \(\lambda\) making the average unforced probability equal to \(p_0=0.4979829448218345\). Restrict \(\lambda\) to \([10^{-3},10^3]\), use at most 40 bisection iterations, and stop when the mean probability differs from \(p_0\) by at most \(10^{-4}\). If the interval does not bracket the desired mean, select the endpoint with the nearest attainable mean probability and record the mismatch. Apply this update once for the rest of that development run. Persist the resulting \(\lambda\) from the selected learning-rate trial. Each of the two F trials gets this same single calibration; select only among trials that reached update 256 and completed it within their 120-second development allowance. Do not extend a trial or silently accept the initial \(\lambda=4\) as calibrated. If neither trial calibrates, the adaptive method is ineligible for confirmation and the primary study stops as practically infeasible or inconclusive, according to the measured cause. This is the complete gate-tuning allowance. Do not adapt \(\lambda\) in confirmation.

Constant-probability controls use \(p_0\), the same 64-update warmup, and the same seven-skip cap. For a constant \(p\), the stationary queried fraction with this cap is

\[
q=\frac{p}{1-(1-p)^8}.
\]

Thus \(p_0\) yields \(q=.5\), while the floor \(p=.25\) yields \(q\approx.2778126325\), before warmup. Do not advertise 75% backward savings from the probability floor. Report realized fractions, including all forced queries.

### 3.5 Online learning, with no gradient-label leakage

After a queried step has cached \(g_t,h_t,p_t,V_b,C_b,s_b\), construct small targets

\[
y_b=V_b^\top g_b/s_b,\qquad
r_b^{\rm old}=\|g_b-h_b\|^2/s_b^2.
\]

Recompute the small controller's coefficient output from its cached features for learning, and include the same coefficient/norm cap. Compute the safe differentiable norm as \(\sqrt{\max(a_b^\top C_ba_b,10^{-16})}\) from the recomputed coefficients, then use \(\kappa_b=\min(1,2/\text{norm}_{\rm safe})\). Flooring **before** the square root implements the same \(10^{-8}\) norm floor and avoids the singular derivative of \(\sqrt{0}\); do not place an unfloored square root in this graph or reuse a detached cached cap as a constant. A materially negative Gram quadratic, beyond float32 roundoff, is a numerical defect to investigate. If \(\widetilde a_b=\kappa_ba_b\), the differentiable direction surrogate is

\[
L_{\rm dir}=\frac1{n_{\rm groups}}\sum_b
\left(\widetilde a_b^\top C_b\widetilde a_b-2\widetilde a_b^\top y_b\right).
\]

The omitted \(\|g_b\|^2/s_b^2\) is constant in controller parameters. Define \(L_{\rm risk}\) as the **mean over parameter groups** of the Huber loss with threshold 1 between \(\log(\widehat e_b)\) and \(\log(r_b^{\rm old}+10^{-4})\). The total supervised loss is

\[
L_{\rm controller}
=\frac1{p_t}\left(L_{\rm dir}+0.1L_{\rm risk}\right).
\]

Detach features, propensities, histories, scales, Gram matrices, and labels. Do not differentiate through \(p_t\), the main network, or the training trajectory. With labels revealed by the independent coin, inverse-propensity weighting corrects the local loss-gradient selection effect **before** controller gradient clipping and Adam; its factor is bounded by four. It does not make the clipped adaptive optimizer update unbiased. This is supervised online regression, not meta-learning or reinforcement learning.

Use one CPU Adam update per eligible queried step, controller learning rate \(10^{-3}\), betas \((.9,.999)\), epsilon \(10^{-8}\), and no weight decay. Backpropagate the combined loss, then clip **trunk plus coefficient-head gradients together at norm 5**, and clip **risk-head gradients separately at norm 5**, before the optimizer step. A single global clip would let risk gradients alter coefficient learning despite the detached hidden representation; it is prohibited. One Adam optimizer with these disjoint parameter blocks is sufficient because its states are parameterwise. No replay buffer, offline pretraining, or controller transfer between model runs. When no valid history exists, take the forced exact step, insert the raw gradient, initialize statistics, and skip **both** controller losses. Subsequent warmup steps with a valid history train normally.

Compute logged residual norms directly from \(g-h\), not by subtracting nearly equal terms in the expanded quadratic. Preserve the raw exact gradient before writing update values into `parameter.grad`. Assign gradients into separate owned update buffers and clip those buffers only; in particular the direct `z = true_g` branch must not cause in-place clipping to mutate `true_g`. Likewise preserve the cached pre-query prediction until all residual labels/logs are formed. History buffers must own their storage.

### 3.6 Main-model optimizer

All non-GaLore methods use the same native torch AdamW implementation, full ambient moments, betas \((.9,.999)\), epsilon \(10^{-8}\), and global gradient-norm clipping at 1.0. Apply weight decay .01 to tensors with at least two dimensions and zero to biases/LayerNorm vectors. Advance moments, bias correction, weight decay, and optimizer-step count on **every** update, queried or not.

Use 64-update linear learning-rate warmup, then a constant learning rate. Two allowed base rates are \(3\times10^{-4}\) and \(10^{-3}\). Keeping the schedule independent of query count avoids accidentally giving methods different optimizer clocks.

Do not transport Adam's diagonal second moments between changing bases. Arbitrary rotations generally introduce off-diagonal covariance; diagonal state is not exactly covariant. Ambient moments avoid this issue.

### 3.7 Required information order and pseudocode

Every NH4-A/NH4-F/LH4/NH1 step uses one **autograd-enabled** main forward. On an unqueried step, release that graph without backpropagating. This saves backward work, but not graph-building overhead or the forward activation peak. A no-grad forward followed by a second tracked forward is outside the core implementation.

```text
initialize model, ambient AdamW, fresh controller, empty exact-gradient history
initialize independent RNG streams for data, controller, and query uniforms
for t in 0 .. N_max-1:
    check cumulative and run budgets before launching the next step
    clear every model .grad
    fetch the predetermined fresh minibatch B_t

    loss, activation_rms = tracked_forward(model, B_t)
    features = detach(pre-query history statistics and current observables)
    h, risk = predict_from_old_history(features)
    supervision_valid = finite(features, h, risk) and old_history_exists
    if controller inputs/outputs are nonfinite:
        h = 0; p = 1; supervision_valid = false; record bounded warning
    else:
        p = fixed_or_adaptive_probability(risk, warmup, consecutive_skips)
    cache effective h, p, features, old Gram matrices, scales, timestamps, validity
    A = (fresh_uniform < p)

    if A:
        loss.backward()                         # one full main backward
        true_g = independent_detached_copy(all model gradients)
        update on-device finite-status flags; fail upon detection at a declared boundary
        z = true_g if cached_p == 1 else cached_h + (true_g - cached_h) / cached_p
        if cached supervision is valid:
            labels = targets_from(true_g, cached_h, OLD history)
    else:
        z = cached_h
        release loss graph and all retained activation references

    record raw-estimator norms and clipping information
    copy z into owned model .grad update buffers; globally clip only those buffers
    main_optimizer.step()                       # all tensors, one clock
    update on-device finite-status flags for the main update

    if A:
        if cached supervision is valid:
            train small controller once using OLD features/targets and 1/p
        insert true_g, not z, into owned history buffers
        refresh cached Gram matrices and previous observed residuals
        reset consecutive_skips
    else:
        increment consecutive_skips
    update loss/activation EMAs after the decision
    buffer logs
    if scheduled:
        synchronize; inspect pending finite flags and stop on failure
        evaluate fixed validation panel; save crossing data
    if two consecutive evaluations meet target:
        synchronize; save required target checkpoint; stop successfully
save failure/censoring status if a limit or numerical failure ended the run
```

Use the numerical-monitoring policy in Section 7.1 rather than adding per-step host checks to asynchronous baselines. Do not retain main-model graphs across iterations. On queried steps, regular backward frees the graph; also release Python references. The test harness must count main-model backward invocations separately from controller backward invocations.

---

## 4. Data, model, and environment

### 4.1 Dataset and acquisition

Use `Salesforce/wikitext`, configuration `wikitext-2-raw-v1`, repository revision

`b08601e04326c79dfdd32d625aee71d232d685c3`.

The commit's LFS metadata provides these checksums [S11]:

| Split | Download bytes | SHA-256 |
|---|---:|---|
| train | 6,357,543 | `e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7` |
| validation | 657,209 | `204929b7ff9d6184953f867dedb860e40aa69c078fc1e54b3baaa8fb28511c4c` |
| test | 732,610 | `5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91` |

Download from the following pattern, replacing `SPLIT` with one of the three literal split names:

`https://huggingface.co/datasets/Salesforce/wikitext/resolve/b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-2-raw-v1/SPLIT-00000-of-00001.parquet`

Verify size and SHA-256 before parsing. These are source-metadata checksums; the planning session did not download the dataset. Use PyArrow to read the ordered `text` column. Define each byte stream exactly as `"\n".join(rows).encode("utf-8")`, with no normalization, lowercasing, vocabulary fitting, or special tokens. Preserve empty rows. Save processed stream hashes and lengths.

Train only on the official training split. Use the official validation split for development selection and the frozen stopping metric. Its reuse is deliberate; it is not an untouched confirmatory dataset. Fresh confirmation training seeds and the untouched official test split provide the reserved evidence. Never feed validation/test losses or bytes to the controller.

Create the validation panel from 128 nonoverlapping 128-target chunks selected at evenly spaced indices across all valid nonoverlapping chunks of the validation stream. Each input is 128 bytes and targets are the next 128 bytes; stride is 128, with the necessary final extra byte. This gives exactly 16,384 target bytes. Check uniqueness of chosen chunk indices. Retain the manifest.

After all locked training runs finish, evaluate their target or terminal checkpoints on an analogous 2,048-chunk test panel, exactly 262,144 target bytes. If the stream were unexpectedly too short, use all complete chunks and record the count; do not change datasets. Cross-entropy is in natural-log units per byte. Do not compare its numerical perplexity to published word- or subword-tokenized WikiText perplexity.

For each model run, sample independent start positions uniformly with replacement from all valid training starts. Use fresh sampled positions each optimizer step; never repeat a batch deliberately over a skip cycle. Paired methods share the same seed-indexed sequence of batch positions, though a cheap control may not forward all allocated batches.

**Acquisition fallback:** retry a failed transfer at most twice, then use a locally supplied copy of the same three pinned files with matching checksums. This is the same data, not a scientific fallback to an easier corpus. If unavailable, complete CPU correctness and analytic diagnostics, preserve the reproducible downloader, and report the real-text comparison blocked. Do not substitute a synthetic corpus and claim the primary result. Preserve dataset attribution and supplied license notices; the dataset metadata and descriptive prose use different historical license labels, so do not invent a definitive reinterpretation.

### 4.2 Model

Implement a plain pre-LayerNorm decoder-only Transformer:

| Item | Fixed default |
|---|---|
| Layers | 4 |
| Width | 256 |
| Attention heads | 4 |
| Context length | 128 bytes |
| Batch size | 16 sequences, 2,048 target bytes/update |
| Vocabulary | 256 bytes |
| MLP | Linear to \(4d\), GELU, Linear to \(d\) |
| Position representation | Learned absolute embeddings |
| Attention | Causal; ordinary dense QKV and output projections |
| Normalization | LayerNorm, epsilon \(10^{-5}\), learned scale and bias |
| Weight tying | Input token embedding tied to output projection; no output bias |
| Other biases | Enabled for QKV, attention output, and both MLP linears |
| Dropout | 0 |
| Initialization | Normal(0,.02) weights/embeddings; zero linear biases; LayerNorm scale 1, bias 0 |
| Precision | float32 on MPS; no autocast, quantization, or mixed precision |
| Loss | Mean next-byte cross-entropy over every batch position |
| Compilation | Eager execution; no compile/Metal-kernel project |

The expected parameter count is \(4(12d^2+13d)+(256+128)d+2d=3,257,856\). Assert the implemented count. No BatchNorm or mutable forward statistics are allowed.

Use supported `scaled_dot_product_attention` if its tiny correctness and backward probes pass. Otherwise use explicit masked scaled dot products, softmax, and matmul for **all** methods; select before timing comparisons and record it. Do not silently enable CPU fallback for a missing MPS operator.

### 4.3 Actual-Mac inspection

Before allocating a study, record:

- Machine/chip, physical RAM, available disk, macOS version, Python architecture/version, torch/NumPy/PyArrow versions.
- `torch.backends.mps.is_built()` and `is_available()`, actual tensor devices, dtype, attention path, optimizer flags, CPU thread count, and relevant MPS environment variables.
- Power connection, low-power mode if observable, other active heavy GPU workloads, and thermal/OS interruptions. Do not require privileges or install monitoring services.
- Import and execute a tiny forward/backward/AdamW, matrix multiplication, reductions, cross-entropy, LayerNorm, GELU, and device-transfer probe.

Suggested read-only inspection uses `sw_vers`, `sysctl -n hw.memsize`, `platform`, `shutil.disk_usage`, and the torch capability checks. Avoid recording hardware serial numbers. Do not infer RAM, free storage, supported precision, or macOS compatibility from “M3 Max.”

Disable automatic fallback with `PYTORCH_ENABLE_MPS_FALLBACK=0` before importing torch. Do not turn off the MPS allocation safety watermark. Official stable documentation redirected to version 2.15 during research; this does not establish the version installed or installable on the target laptop. Use the compatible installed stable release, verify required operations, and freeze it. [S12]

If the target Mac is unavailable, complete source-independent CPU correctness and the code/configuration/report scaffold. Stop before hardware claims, clearly identifying which commands remain unexecuted. Linux CPU timings in the planning environment do not establish M3 performance.

### 4.4 Resource ceilings and scale selection

Hard project disk cap: **8 GiB**, including environment, downloads, checkpoints, logs, and outputs. Require at least 12 GiB free before setup. Retain only one resumable full-state checkpoint for the active run and final model-only checkpoints for completed runs, with one narrow exception: retain the two temporary full-state A development checkpoints until their rate is selected, then discard the unselected full state; never delete raw scalar results or failures to meet the cap.

Process memory cap:

\[
M_{\rm cap}=\min(8\ {\rm GiB},\,0.25\times\text{physical RAM}).
\]

Apply this ceiling independently to observed process RSS and observed MPS driver allocation; do not add overlapping unified-memory counters. Stop launching work on sustained system memory pressure or a measured cap breach. A necessary temporary allocation exceeding the cap is a resource failure. Do not induce swap to finish a benchmark.

For the default model, parameters, gradients, and two float32 Adam states account for approximately 49.71 MiB; four history vectors add another 49.71 MiB. These are arithmetic lower-level accounts, not peak-memory forecasts. Prediction copies, true-gradient copies, activations, attention intermediates, backend workspaces, and caches must be measured. Never allocate an object scaling as \(P^2\).

Calibrate the default model first. If it fails memory or the conservative full-matrix runtime forecast, the **only architecture reduction** is width 192 with six heads, retaining four layers and context 128; expected parameters 1,853,568. If memory still fails, batch size may fall once from 16 to 8 for every method. Make these choices before comparative pilot results, based only on resource measurements. Do not reduce below these floors for the primary study. Record the narrowed scope. If the floors fail, report practical infeasibility.

Use provisional validation cadence 16 throughout resource calibration and development, including the cost forecast; it is the most expensive allowed cadence. Select reference horizon \(N_0\) from \(\{1024,2048,4096\}\), largest first, using measured costs: its conservative AdamW duration including validation must be at most four minutes, and the projected remaining complete study plus 25% margin must fit the cumulative cap. Set \(N_{\max}=\lceil1.5N_0\rceil\) for every locked method, with the additional nine-minute per-run limit. If 1,024 updates cannot meet the resource forecast at the minimum model, stop the main study rather than shortening to a nearly all-warmup experiment.

---

## 5. Comparators, controls, and tuning

### 5.1 Mandatory locked matrix

There are seven method rows and five paired fresh seeds: **35 locked model runs**.

| ID | Method | Purpose |
|---|---|---|
| A | Ordinary native AdamW, current full gradient every step | Practical full-backpropagation reference |
| B | Development-selected best of cached-plain, cached-CV, extrapolated-CV | Strong cheap temporal reference; selection rule below |
| C / LH4 | Same four-gradient dictionary and 30 features, affine coefficient scores, fixed query probability | Tests whether nonlinear prediction adds value |
| D / NH1-F | Same neural controller, only newest history slot valid, fixed query probability | Tests value of additional historical directions |
| E / NH4-F | Four-gradient neural predictor, fixed query probability | Isolates prediction/memory from adaptive querying |
| F / NH4-A | Four-gradient neural predictor and learned residual-driven probability | Primary candidate |
| G | Specified float32 GaLore-AdamW reference | Contextual memory/subspace comparator with full backward passes |

Methods C–F use (M1), the same prediction cap, online supervision, warmup, and maximum skip gap. Fixed-probability variants may omit risk-head training/computation/state because they do not use it. Detaching the risk-head input and independently clipping the two parameter blocks makes this omission leave the coefficient learner unchanged. Preserve identical initialization/RNG draws for the retained trunk and coefficient parameters. All learned variants still compute the observed residual needed by the shared previous-observed-error feature. Do not charge a competitor for unnecessary work merely to equalize an implementation.

C replaces the hidden-layer MLP coefficient scores by `Linear(30,4)` with the same bounded output and norm cap, newest-gradient initialization, and online Gram loss. It is an affine-score baseline over the same features, not a faithful reproduction of PGD.

D masks slots 2–4 and their features to zero and uses the same shared MLP. It still has Adam momentum; “one-history” does not mean a completely memoryless optimizer.

### 5.2 Cheap baseline selection

Develop these three B candidates:

1. **cached-plain:** \(h=g_{\rm last}\); use \(g\) when queried and \(h\) otherwise. This familiar biased reuse baseline does not claim (M2).
2. **cached-CV:** \(h=g_{\rm last}\), with (M1).
3. **extrapolated-CV:** with two queried gradients at \(q_1>q_2\),
   \[
   h_b^{\rm pre}=g_{b,q_1}
     +\frac{t-q_1}{\max(q_1-q_2,1)}
       (g_{b,q_1}-g_{b,q_2});
   \]
   apply the common \(2s_b\) prediction-norm cap, then (M1). With only one gradient, use cached-CV.

All three use fixed \(p_0\), warmup, and the seven-skip cap. Their decisions depend only on history/RNG. On unqueried steps they may **omit the main forward entirely**; current activations/loss are not needed. Advance the common batch-position stream and optimizer clock, but record that those allocated bytes were not forwarded. This is a real efficiency advantage of a simpler predictor, and must not be artificially removed.

Training loss is missing on those steps, not zero. Validation remains identical. Do not inject extra train forwards just to make a complete loss log.

B is the valid variant/rate with best development validation loss under the common 120-second development allocation. Apply the deterministic near-tie rules in Section 5.4. Preserve all candidates, actual elapsed costs, and selection measurements.

### 5.3 GaLore reference

Use the authors' ordinary float32 GaLore-AdamW logic, not an 8-bit/CUDA package. Vendor the minimal optimizer/projector code under its license, pin the retrieved commit/hash, and test against the unmodified CPU source on fixed gradient sequences. The source was inspected during planning, but not executed on MPS. [S4]

Apply projection only to the four weight matrices in each Transformer block, not embeddings, biases, or LayerNorm parameters. Rank is one quarter of the smaller matrix dimension, projection gap 200 optimizer updates, `proj_type="std"`, scale .25. Globally clip the current full gradient to norm 1 **before** projection, then use betas \((.9,.999)\), epsilon \(10^{-8}\), bias correction, and the declared decay groups. Preserve the source's update/epsilon/decay conventions in the projected optimizer and report differences from native torch AdamW; do not assume they are bitwise interchangeable.

A required small MPS SVD probe determines whether the projector can stay on device. If unsupported, explicitly transfer only the matrix being decomposed to CPU, perform its SVD, and return the basis. Charge transfers and decomposition. This is a named, visible CPU operation, not automatic fallback.

For this comparator, allow two base rates \(10^{-3}\) and \(4\times10^{-3}\) to account for the .25 projection scale. Those rates are a fixed two-trial allowance, not a post-result favor. Its nonprojected parameters use the same selected base rate; report this convention.

Choose G's projector **before either scored learning-rate trial**, using bounded resource microbenchmarks. Estimate exact-SVD cost including device transfers, the first refresh, and every 200-step refresh over the proposed horizon. If this accounts for more than 20% of G's projected runtime, permit one predeclared engineering alternative: randomized range finding with oversampling 8, two power iterations, reduced QR, then SVD of the small matrix, with all work charged. Select the faster valid projector from these measured forecasts and freeze it for both G rate trials and all confirmation runs. No mid-trial substitution or extra rate sweep is allowed. If the alternative later fails development, mark G invalid/blocked rather than silently switching algorithms.

The exact-SVD vendor path must match the unmodified authors' CPU implementation. The randomized path instead needs deterministic shape, orthogonality, reconstruction, and optimizer-update checks **given the same supplied projectors**; it need not reproduce the source's exact SVD basis or trajectory. Label it a GaLore implementation with a randomized projector, informed by GaLore 2, not a reproduction of GaLore 2's distributed/quantized system. [S6]

If the source cannot be retrieved or this implementation cannot pass its CPU reference check within the repair allowance, mark G blocked and remove only its comparisons. Do not fabricate it from a name. The primary prediction study may proceed with A–F, but no claim involving GaLore is then allowed.

### 5.4 Tuning and freeze

Use development seed 20261104. Each of the nine candidates—A; the three B variants; C; D; E; F; G—gets exactly two rate trials: **18 development trials**, each with the same **120-second training allocation**, inclusive of initialization, online work, and cadence-16 validation. There is no 512-update scientific cap: a faster method may complete more updates. Stop before launching a step predicted to overrun the allowance; finish the in-flight step and one terminal validation, recording any measured overrun separately. Compare terminal development losses at this common allowance, and include actual complete elapsed cost in the ledger. A safety ceiling of 32,768 updates only protects against a broken clock/loop; reaching it invalidates that trial for equal-time ranking and requires a bounded measurement repair, not a favorable score.

Preserve \(L_{64}\) and the scheduled losses at \(N_0\) for **both** A-rate trials if reached during development. After selecting A's rate by its 120-second result, use its existing \(N_0\) anchor; if it has not yet reached \(N_0\), continue that same trial to \(N_0\) within the stage budget. Run the two A rates early, retain their temporary full states until selection, and discard the unselected full state afterward. If continuation requires a fresh process, restore weights, optimizer, RNG, batch index, and counters exactly. Accumulate elapsed active segments including save/load and initialization costs, excluding time spent executing other trials. Its anchor extension cannot change the already selected rate. Controller architecture, learning rate, rank, clipping, warmup, and feature definitions are fixed. Development runs may exceed \(N_{\max}\); the progress feature retains its specified clamp at 1. Locked runs obey their update cap.

Both F trials calibrate \(\lambda\) once as in Section 3.4. No separate grid search is permitted. Select each method's rate by best terminal development validation loss, restricted to valid, fully calibrated trials when calibration applies. For losses within .01 nats/byte, prefer the lower rate; when choosing B's variant, first prefer the simpler cached-plain, then cached-CV, then extrapolated-CV. Select the primary simple/linear reference **R from B and C only** with the same loss ranking, favoring B for a tie. A remains a separate mandatory comparator. Freeze R before confirmation.

A has no controller; it should use its fastest validated native AdamW flags supported by the chosen backend. Use those same flags for ambient AdamW in B–F. Do not keep A on an intentionally slower Python optimizer while optimizing the proposed path.

Write `protocol.lock.json` containing source revision, dataset/processed hashes, model, batch, dtype, attention path, \(N_0,N_{\max}\), validation manifest/cadence, target, all selected rates, B variant, R identity, \(\lambda\), seeds, order, tolerances, caps, and analysis rules. Hash it and the code. Only then start confirmation.

---

## 6. Correctness and mechanism diagnostics

### 6.1 Tiny checks that must precede timing conclusions

Use CPU float64 for analytic/reference tests; deterministic tests do not need multiple random training seeds. Start with absolute tolerance \(10^{-10}\) and relative tolerance \(10^{-8}\) for small well-scaled CPU identities. A failed identity is a bug to investigate, not a tuning outcome.

| Check | Required evidence and failure it detects |
|---|---|
| Two-outcome correction | Enumerate \(A=0,1\), verify (M2), including data-dependent \(p\); no Monte Carlo required |
| Exact-query limit | With \(p=1\), model updates/moments match ordinary AdamW for at least 20 tiny steps; controller changes cannot alter the main update |
| Perfect-predictor control | For a constant known gradient \(h=g\), (M1) is \(g\) for every query outcome; confirms that perfect prediction can preserve updates |
| Anti-leak control | In a test copy set \(h_{\rm query}=g\), leave \(h_{\rm no\ query}=h\ne g\), and use \(0<p<1\); the expectation check must fail |
| History ownership | Mutating/clearing `.grad` cannot change stored exact history; corrected \(z\) is never inserted as \(g\) |
| Gradient count | Unqueried model steps invoke no main `backward`, `autograd.grad`, JVP, VJP, or HVP; controller backward is counted separately |
| Controller surrogate | Gram-loss coefficient derivatives agree with direct tiny-vector loss and finite differences; propensity is detached; zero history, zero coefficients, and canceling columns have finite gradients |
| Risk-path isolation | With identical trunk/head initialization, fixed features, labels, propensities, and query sequence, enabling versus disabling risk training leaves coefficient/trunk parameters and Adam states equal for several updates; catches hidden gradient or clipping coupling |
| Representation | Direct \(Va\) agrees with QR representation \(Q(Ra)\); orthogonal changes of the QR basis with consistent coefficient changes preserve the decoded vector |
| Transformer oracle | Uniform logits give \(\log256\); changing future inputs cannot change earlier logits; float64 directional derivatives agree with autograd |
| Restart | Save/reload a tiny run reproduces the next batch indices, query coin, history, and next update |
| Measurement | A known injected CPU delay increases measured wall time; missing synchronization is detected by the calibration harness |
| GaLore | When G is available, exact-SVD vendor matches author CPU implementation at fixed projector refreshes; randomized projector uses its separately specified checks; a documented G block waives only these G-specific checks |

An arbitrary coefficient MLP is not invariant to arbitrarily mixing its chronological history slots. The representation test changes an **internal QR representation**, not the semantic identities of the four past gradients. Intentionally sign-corrupting a history without transforming its coefficients is a bug control, not a scientific competitor.

Compare CPU float64, CPU float32, and MPS float32 on the same tiny network. Initial MPS acceptance targets are loss absolute error \(\le10^{-4}\), relative gradient L2 error \(\le2\times10^{-3}\), and a 20-step \(p=1\) trajectory relative parameter error \(\le3\times10^{-3}\). These are engineering tolerances, not mathematical constants. If they fail, identify the operator and compare both float32 backends to the float64 oracle. One justified common relaxation up to fivefold is allowed before locking, with a documented scale/conditioning explanation and errors far smaller than the .02-nat quality margin. Otherwise use a correct alternative shared operator or stop MPS claims. Never loosen tolerances after seeing comparative gains.

### 6.2 Cheap falsification before extensive controller development

First measure default exact forward, backward, optimizer, prediction decode, feature transfer, and memory-refresh costs. Then perform a 128-step teacher-forced diagnostic on an ordinary AdamW training trajectory after 64 warmup steps. This is one explicitly counted calibration/diagnostic attempt, separate from the scored A development runs; its extra probes cannot slow the A timing used for reference selection. Do not save full gradient traces: stream scalar statistics and, at most, four exact gradient histories. Budget and count every diagnostic backward.

At each diagnostic step, compute candidate predictions **before** the current exact gradient is revealed. For the diagnostic predictor, expose labels/history according to a simulated fixed query mask with the same warmup/probability/cap; never supply all teacher-forced gradients to its memory. The reference model itself still uses every exact gradient, so this diagnoses forecastability on the reference path only.

Measure:

- Fresh-batch gradient/prediction residual energy, cosine, norm ratio, and per-group contributions.
- Residual energy for \(r=1,2,4\) history spans using the same available old gradients.
- A repeated-minibatch control on 16 paired probes, clearly marked artificial, to distinguish input sampling noise from weight drift.
- A scalar-only covariance/noise estimate from 16 pairs of independent minibatches at one frozen checkpoint; these are extra charged backward queries.
- The known best in-span projection error, used only offline as an oracle diagnostic.

For an orthonormal basis \(Q\) of the history span and \(h\) in that span,

\[
\|g-h\|^2=\|(I-QQ^\top)g\|^2+\|Q^\top g-Q^\top h\|^2.
\tag{M4}
\]

Use small Gram matrices and a rank-revealing CPU float64 eigensolve, or tiny QR references, to calculate this decomposition. Record the rank cutoff, accounting for float32 precision in the original Gram construction; converting a cached float32 matrix to float64 does not recover lost information. Check borderline ranks with direct tiny-vector references. Do not misclassify numerical rank loss as a theorem about neural rank.

Large out-of-span error limits this dictionary; large within-span error implicates prediction. Low alignment with an individual stochastic gradient alone is not a proof that mean descent is useless. The frozen-checkpoint batch-pair estimate and the closed-loop pilot are needed before a negative training conclusion.

**No numerical threshold on cosine is a theorem-level rejection criterion.** The opportunity gate combines measured cost headroom, meaningful post-warmup learning, and valid closed-loop behavior.

### 6.3 Pilot rules

Within the first 25 minutes of Stage 3, obtain at least one closed-loop run of A, cached-plain, E, and F using their full 120-second development allocations. These are the first four of the 18 development trials, not additional pilot attempts. Inspect the first 512 updates when available and the full time-limited trajectories. Diagnose before completing the remaining trials.

Continue if the implementation is valid, reference learning is meaningful, the timing model leaves a possible 10% gain, and E/F make sustained progress after warmup without pervasive numerical failures.

One bounded repair is allowed for a concrete defect: aliasing, wrong tensor shape, incorrect bias correction, data slicing, a demonstrably unsupported backend operation, or mischarged timing. A poor predictor is not an implementation bug by definition.

If the pilot is noisy or E/F have a small disadvantage, do not declare a refutation. Complete the preallocated development trials or one remaining-budget diagnostic that resolves a named ambiguity. If the question cannot be distinguished at the planned scale, classify it inconclusive and stop; do not add seeds until a favorable result appears.

A poor but stable result on the single development seed remains exploratory and cannot establish the five-seed primary negative. Continue the preallocated comparison when a useful decision remains feasible. Early termination is justified by a measured unavoidable-cost/resource obstruction, an independently verified analytic diagnostic resolving a restricted claim, or an invalid/blocked experiment. If budget or scale prevents confirmation, report inconclusive evidence with the pilot finding preserved. No large benchmark is required after a fatal gate.

---

## 7. Timing, computation, and memory accounting

### 7.1 Primary timing boundary

Use one fresh process per model run. Load/import dependencies and identify the already-prepared dataset before the per-run timer; then start the primary CPU monotonic timer **before the first MPS API call or device allocation**, immediately before model/controller/optimizer allocation and initialization. Perform device-availability inspection in the separate setup process; do not warm the run process beforehand. It includes:

- Device initialization and first-use backend work occurring inside the run.
- All model and controller forwards/backwards, warmup, prediction, Gram updates, copies, CPU transfers, probability decisions, optimizer work, data batching, required logging, validation, and the required target/terminal checkpoint.
- Any repair/restart within a reported run attempt. A new attempt has its own status and its costs also enter the project ledger.

The fresh process has no outstanding MPS queue, so do **not** synchronize before starting its timer: that call could itself initialize the device outside the measurement. After the timer starts, synchronize immediately before each reported elapsed measurement or stopping boundary. Buffered timestamps may be unsynchronized for debugging but cannot support a speed claim.

For numerical monitoring, accumulate finite-status flags and the earliest flagged update on device. Inspect them at the common scheduled synchronization boundaries, before accepting validation attainment, and at the terminal boundary. Stop upon detection and mark the entire attempt numerically failed; no state at or after the earliest flagged update can support attainment. F may detect failure earlier through its already-required batched feature/label transfers. Do not use per-step `.item()` decisions or `error_if_nonfinite=True` clipping to introduce otherwise unnecessary host barriers in A or other asynchronous paths. This is the declared detection latency, not permission to continue after observing a failure.

The reported time to quality is the elapsed time **after the second consecutive below-target validation evaluation and the required checkpoint write**. This includes the cost of verifying attainment. Keep the crossing bracket for the **first evaluation in the qualifying pair of consecutive below-target evaluations** separately, using the last above-target evaluation before that pair. An earlier transient crossing that did not qualify is not this bracket. Imports/common dataset download/preprocessing are reported once as setup cost, and all development/tuning/failed-attempt costs are reported as total R&D cost.

Controller training is online and included in every run. There is no pretrained controller to amortize. If later investigating reuse, it is a new study with its training cost and a stated number of deployments.

Do not synchronize A every training step merely to imitate F's necessary host decision. F's feature transfer and branching barrier are real overhead. Separate synchronized microbenchmarks may identify costs, but primary full runs preserve each algorithm's natural execution.

### 7.2 Supporting metrics

Record per run:

- Quality-attainment status, confirmed time, first-crossing time interval, final validation loss, final test loss.
- Main tracked forwards, main untracked forwards, exact main backwards, controller backwards, optimizer updates, allocated data bytes, actually forwarded bytes, and all diagnostic queries.
- Post-warmup and total query fractions, forced-query reasons, consecutive-skip distribution.
- Raw correction norms, global clipping fraction and scale, prediction caps, observed residuals, risk predictions, propensities, and weighted calibration summaries.
- Observed RSS peak, observed MPS tensor/driver allocation maxima, and any memory-pressure interruption.
- Complete setup/development/confirmation/repair/report time ledger and disk use.

MPS `Event(enable_timing=True)` can provide elapsed stream intervals if the installed version passes a probe. It is **not** a count of executed FLOPs or automatically a pure GPU-active-time sum. Optional signpost/Metal traces may explain bottlenecks if available without privileged setup. If cumulative GPU-active time is unavailable, say so; report operation counts and measured stream intervals with their exact definitions. Do not invent “GPU compute saved” from backward counts alone. [S12]

Sample MPS tensor and driver memory at common phase boundaries and with a lightweight periodic monitor; record sampling interval. Tensor allocation excludes allocator caches; driver allocation includes caches and MPS framework allocations. These are observed maxima, not guaranteed within-kernel high-water marks. OS RSS high-water and sampled RSS must be labeled separately. Do not sum overlapping unified-memory counters.

### 7.3 Cost model and stopping logic

Let \(F,B,U\) denote measured forward, backward, and ordinary optimizer costs, and \(H\) all extra predictor/query/history overhead per update. For this tracked-forward candidate,

\[
c_A\approx F+B+U,\qquad c_F\approx F+qB+U+H.
\]

For update-count ratio \(\rho=N_F/N_A\), a necessary approximate 10% timing opportunity is

\[
\rho(F+qB+U+H)\le .9(F+B+U),
\]

before unequal evaluation/setup costs. Use complete measured time for the final verdict.

An illustrative arithmetic calculation with \(F=1,B=2,U=.1,H=.15,q=.5\) gives \(c_A=3.1,c_F=2.25\), and permits \(\rho\le1.24\) for a 10% time reduction. These are chosen units, not M3 measurements. Extra iterations can erase the benefit rapidly.

The cheap B control can cost roughly \(q(F+B)+U\), because it need not forward unqueried batches. Include this stronger comparison in opportunity estimates.

If even zero-error predictions at the attainable query rate cannot offer 10% gain at equal update count once measured unavoidable overhead is included, stop the claimed backward-efficiency study. A future claim about reducing the number of updates would be a different mechanism requiring its own evidence. If observed overhead has a specific obvious implementation defect, one bounded correction is allowed before this decision.

Use median and upper-quartile timings from at least five short repeated blocks after first-use warmup to forecast costs. Repeated microbench blocks are timing samples, not independent training outcomes. Add a 25% forecast margin. Never launch a run when its forecast plus the reserved reporting allowance exceeds the remaining cap.

---

## 8. Evidence gates and cumulative budgets

The hard cumulative experimental/local-analysis compute ceiling is **480 minutes (eight hours)**. These are allocation limits, not measured runtime predictions. The first informative pilot is targeted within approximately 60 minutes of cumulative local experimental computation once dependencies, data, and the required implementation are ready; the full study may finish well before its ceiling.

Setup/download and implementation effort are separately estimated at 4–8 agent-hours, with up to 45 minutes of ordinary installation/download troubleshooting. This is an unmeasured planning estimate. The full end-to-end commitment may be roughly one long working day or two work sessions, even though experimental computation is capped at eight hours. Do not present software-development time as GPU training time.

All experiments, exploratory trials, failed runs, retried jobs, calibration, diagnostic probes, and fallback branches accumulate against these limits. Unused stage allowance may be reassigned before confirmation only with a ledger entry and unchanged total cap; it is not a fresh budget. The final 30-minute analysis/preservation allocation is protected.

| Stage | Question, minimum work, and prerequisites | Required evidence | Cap | Decision and next action |
|---|---|---|---:|---|
| 0. Constructive feasibility | Inspect actual Mac; acquire/check data; verify no current-gradient oracle in the design; read available baseline source | Environment, data manifest, dependency status, information-flow checklist | 15 min compute | Continue if data/backend are available. Same-data transfer retries only. If blocked, complete unaffected CPU work and report the precise block |
| 1. Tiny correctness | Implement minimal model, (M1), history, controller Gram loss, references; run Section 6.1 | Test results, counterexample control, CPU/MPS differences | 10 min compute | Continue only after valid checks. Concrete bugs use repair reserve. Unresolved core defect means invalid experiment, not negative science |
| 2. Resource calibration | Measure representative F/B/U/H, first-use costs, memory; choose allowed model/batch/horizon and measure validation cost at provisional cadence 16 before comparative outcomes | Calibration JSON, scaling forecast with margin, resource choice | 10 min | Continue if minimum meaningful configuration fits. Prescribed width reduction, then batch reduction only if memory still fails; no lower floors. Otherwise practical infeasibility |
| 3. Mechanism pilot and development | Initial A/B/E/F runs within the 18 two-rate trials; separately counted teacher-forced diagnostic; one selected A anchor; one lambda calibration per F trial | All trials, failures, chosen settings, target, lock file | 55 min | Continue if valid and scientifically informative. Ambiguity gets only a named bounded diagnostic. Fatal cost/information/instability findings may stop; noisy pilot may be inconclusive |
| 4. Locked comparison | Frozen protocol; five fresh seeds; seven rows, at most 35 runs × 9 min | Raw logs, timestamps, checkpoints, success/censoring counts | 315 min | Execute predeclared matrix without retuning. Stop individual failed runs. If global cap threatens completeness, report incomplete matrix/inconclusive comparisons |
| 5. Targeted diagnosis | Only a specific unresolved mechanism question from completed runs; at most two short additional model runs | Marked exploratory diagnosis, no new primary test | 20 min | No fishing. Skip if conclusion already clear. No change to the locked outcome |
| 6. Analysis and evidence preservation | Analyze all completed/failed runs, untouched test panel, regenerate tables/plots/report | Reproducible report and archive | 30 min protected | Always complete for reached stages; no forced late training |
| Repair reserve | At most two materially distinct code/backend repairs or reruns across the whole study | Original failures, fix, impacted results, rerun cost | 25 min | If exhausted, stop affected claims; do not relabel engineering failure as hypothesis failure |
| **Total** | **All stages and branches combined** | **One cumulative ledger** | **480 min** | **No automatic extension** |

Maximum model-training attempt count is **62**: up to five smoke/calibration/teacher-forced attempts combined, 18 development trials, 35 locked runs, two diagnosis runs, and two repair attempts. The five early slots cover the 16-step smoke run, default and at most two resource-reduced calibrations, and the separate 192-update teacher-forced trajectory (64 warmup plus 128 diagnostic updates). Charge that last trajectory to Stage 3 time and the other early work to Stages 1–2; this is one attempt pool, not extra stage allowances. The initial A/B/E/F pilot runs are the first four development trials. Continuing the selected A development trial to its anchor is the same attempt; its additional computation is charged. Any extra attempted run consumes a remaining appropriate slot rather than silently increasing the cap. Tiny algebra tests and frozen-checkpoint derivative probes have separate counters but consume the same wall-time budget.

At each stage transition, write a short gate decision with measured evidence, remaining wall time, remaining run slots, next authorized action, and limitations. Do not build later-stage orchestration before the earlier gates pass.

### 8.1 Stopping and failure specifics

- Nonfinite model loss/gradient/update: stop upon detection under the declared monitoring policy, record the earliest flagged update and last valid state, and mark the attempt failed. Do not resume after observing a NaN.
- A single nonfinite controller prediction may force \(p=1,h=0\) and skip that controller supervision update; two such events within 32 updates end the attempt as numerical failure.
- Query rate drifting upward is a result, not a reason to lower \(\lambda\) during confirmation.
- OOM or memory-cap breach: failed attempt. Only pre-confirmation resource reduction is allowed.
- Per-run time/update limit without target: right-censored nonattainment, not “converged at the timeout.”
- External interruption: preserve the attempt. One predeclared repair rerun is allowed if the cause is documented independently of performance; do not selectively rerun slow outcomes.
- Material post-lock code fixes invalidate affected comparisons. Label the repaired analysis exploratory unless a complete fresh paired comparison fits the remaining cap. Never mix old defective baselines with new corrected candidate runs.

---

## 9. Locked experimental protocol

### 9.1 Quality target

After selecting A's development rate, use its recorded \(N_0\) anchor if already reached; otherwise continue the same development trial to \(N_0\). Let \(L_{64}\) be its validation loss at the end of the 64-update exact warmup, and \(L_{\rm end}\) the mean of its two scheduled validation losses ending at \(N_0\), not losses after \(N_0\). Set

\[
\tau=L_{\rm end}+0.2(L_{64}-L_{\rm end}).
\]

This target uses post-warmup improvement, not the easy initial fall from random logits. Require \(L_{64}-L_{\rm end}\ge.25\) nats/byte and that A's first stable crossing on development occurs after update 256. The .25 requirement prevents benchmarking a virtually flat/no-learning run; the 256 requirement leaves substantial operation after warmup.

If the target is crossed too early, use the **single predeclared stricter target** \(L_{\rm end}+0.05(L_{64}-L_{\rm end})\), still requiring crossing after 256 and before \(N_0\). If no such meaningful stable target exists, the planned horizon is uninformative and the primary study stops as inconclusive. Calibration already selected the largest resource-feasible horizon; no unbudgeted extension or smaller favorable target is allowed. Do not choose a target from candidate gains.

Evaluate initially, at update 64, then at the common locked cadence. A hit requires **two consecutive** scheduled validation losses \(\le\tau\); the second checkpoint is the target checkpoint. No interpolation is used to invent a faster attained time.

Use cadence 16 for calibration/development. After the A anchor is available, choose the common locked cadence from \(\{16,32,64\}\): the largest interval whose projected wall-time bracket is at most 2.5% of A's anchor target time, while validation costs no more than 15% of the complete A run. If both cannot be met, retain the smaller interval needed for resolution and count its overhead; if even cadence 16 cannot resolve a 10% effect, flag the primary speed conclusion as potentially inconclusive. Preserve crossing brackets so this can be checked directly. Recheck the complete-study cost forecast at locking without weakening the 25% resource margin.

### 9.2 Independent seeds and run order

Confirmation seeds are **20261105, 20261106, 20261107, 20261108, 20261109**. Development uses 20261104 only. Derive separate reproducible RNG streams for initialization, data sampling, query uniforms, and controller initialization. All methods within a seed receive identical starting main weights and batch-position sequence.

Controller weights/history reset each run. Use the same underlying query-uniform stream across eligible methods; different probabilities may produce different decisions. Do not transfer learned controller weights or hidden states from development.

Generate the run order before locking by shuffling method order within each seed with RNG seed 104729, then rotate seed-block order with seed 130363. Save the exact list. Run one experiment at a time. Do not run all baseline jobs cold and all candidates warm; a fresh process and balanced order reduce this bias. Record interruptions/thermal changes rather than dropping unfavorable blocks.

### 9.3 What comparisons can establish

- F versus A and frozen R answers the primary question.
- E versus C tests nonlinear coefficient learning over matched features/history/query probabilities.
- E versus D tests more historical directions with matched predictor/query structure.
- F versus E tests the total value of the adaptive risk policy. A policy-specific allocation claim additionally requires realized post-warmup query fractions within five percentage points, or a clearly labeled cost-quality frontier interpretation.
- G supplies a measured low-rank optimizer reference; it does not turn the study into a state-of-the-art optimizer competition.
- B's ability to skip unnecessary forwards is part of the practical reference, not a fairness violation.
- Diagnostics of exact gradients on F's queried steps must be inverse-propensity weighted to correct preferential observation. Report \(\sum A_te_t/p_t\) divided by the number of eligible steps, plus effective sample size and weight range, with warmup separate. At the random time-to-target stopping point this is an **inverse-propensity-weighted descriptive diagnostic**, not a guaranteed unbiased mean conditional on the attained adaptive trajectory. The per-step conditional identity survives. Use fixed-length, predeclared windows if claiming exact mean-unbiasedness in expectation over the adaptive experiment.

The gradient audit already used by F is algorithmic cost. Extra shadow gradients for diagnosis must run separately on a frozen checkpoint/replay and be charged as diagnosis, never silently used to improve the live policy.

---

## 10. Analysis and honest conclusions

### 10.1 Primary analysis

The independent unit is a complete seeded training run. Tokens, parameter groups, and steps are not independent replicates of algorithm performance.

For every method/seed report target attainment, censored/failure status, elapsed time, backward count, validation loss, test loss, and memory. Present individual observations; do not hide them behind one average.

For a comparator J with all five paired target attainments, compute

\[
d_i=\log(T_{F,i}/T_{J,i}),\qquad
\bar d\pm t_{.975,4}\,s_d/\sqrt5,
\]

and the one-sided 95% upper endpoint

\[
U_J=\bar d+t_{.95,4}s_d/\sqrt5.
\]

Report the geometric mean ratio \(\exp(\bar d)\), the two-sided interval, all five ratios, and \(U_J\). State that the t interval assumes approximately symmetric/normal seed-level log ratios; with five seeds it is a coarse uncertainty summary, not a distribution-free guarantee. Also give the paired range and leave-one-seed-out estimates; do not use those exploratory summaries to change the decision rule.

The primary speed criterion is **both** \(U_A\le\log(.9)\) and \(U_R\le\log(.9)\), where R is the frozen winner of B/C. Both required comparisons must pass; do not select the favorable comparator after confirmation. Report every other method comparison descriptively unless explicitly marked secondary.

Check resolution conservatively: replace each comparator's full attained time in the speed-ratio denominator by its **elapsed time at the last above-target evaluation before the qualifying pair**. The resulting stricter comparison must not overturn the geometric-mean point estimate of a 10% reduction. This sensitivity analysis cannot create a success; it can identify an unresolved timing-resolution limit.

For the derivative-saving condition, compute paired exact-backward-count ratios to A **at target**, including warmup and all algorithmic queries. Their geometric mean must be at most .75, with no seed exceeding 1.0. This avoids claiming savings from a low query fraction when extra iterations require more total gradients.

For the independent quality guard, after all runs freeze, compare target-checkpoint test losses \(L^{\rm test}_{F,i}-L^{\rm test}_{J,i}\) separately for **both J=A and J=R**. Each paired difference's one-sided 95% upper t endpoint must be at most **.02 nats/byte**, approximately a 2% multiplicative byte-perplexity difference. This is a predeclared practical noninferiority margin; exact equality is not statistically demonstrable. If the margin is unacceptable for a future application, that application requires a new study.

Thus “same quality” means the same frozen validation target plus this explicit held-out tolerance. State both.

### 10.2 Censoring, failures, and insufficient precision

If either method lacks a target attainment in any paired seed, do not compute an ordinary complete-case speed interval as the primary result. Report attainment fractions, the full capped trajectories, and censored times. Optional restricted-time summaries must keep status indicators and must not treat a timeout as an attained time.

All five F runs must reach the target for the supported primary contribution. If the baseline attains and a valid F repeatedly fails within the same cap, that supports a scoped practical negative result, not universal nonconvergence. If both fail, the target/horizon may be infeasible and the comparison is inconclusive.

A five-seed interval crossing the practical threshold is inconclusive about a 10% gain, even if its point estimate looks attractive. For valid fully attained pairs, a one-sided 95% **lower** endpoint \(\bar d-t_{.95,4}s_d/\sqrt5>\log(.9)\) against either required comparator supports rejection of the claimed 10% advantage within this regime; merely failing the upper-endpoint success test does not. Do not add fresh seeds after looking unless they were already reserved in the locked protocol; this brief reserves none.

Do not discard runs for being unusually slow, having low gradient cosine, or hurting a desired conclusion. Report numeric failures and their rates separately from ordinary censored nonattainment.

### 10.3 Mechanism interpretation

Use the decomposition (M4), timing breakdown, query rate, and clipping frequency to explain the result:

| Observation | Supported interpretation | What it does not prove |
|---|---|---|
| History projection leaves large residual even with oracle coefficients | This four-gradient dictionary lacks much of the current sampled-gradient signal | All low-dimensional or input-conditioned predictors are impossible |
| In-span residual small but learned prediction poor | The chosen online feature/learner/training rule is inadequate | The representation itself is inadequate |
| Raw prediction adequate, correction/clipping frequent and progress poor | Residual-correction variance or nonlinear optimizer interaction is a plausible bottleneck | Raw unbiasedness guarantees useful optimizer behavior |
| Good learning per update but no elapsed gain | Controller synchronization, memory traffic, validation, or backend overhead consumed savings | Large GPUs or larger models must also fail |
| Repeated-batch diagnostic works, fresh-batch training does not | Sampling variability is a likely obstacle in this regime | Training on artificially repeated batches validates general language training |
| E beats D, but F does not beat E | History helps; learned query allocation is not established | The full original controller architecture is supported |
| C or B removes F's advantage | Added nonlinear learning/adaptive machinery is not justified here | All gradient prediction is useless |
| F beats A/R but test guard fails | Faster attainment of the selected validation metric with a quality/generalization tradeoff | Same-quality training acceleration |
| F improves with no backward-count reduction | Another optimization/implementation effect may explain it | Success at replacing expensive gradient calculations |

For actual update directions \(\Delta\theta_t\), the relevant first-order quantity is \(-g_t^\top\Delta\theta_t\), not only cosine between \(h_t\) and \(g_t\). Measure it on queried steps in a bounded diagnosis, including weight decay if reporting the actual displacement. It is local sampled-batch evidence, not a certificate of global loss decrease.

### 10.4 Required outcome labels

Use one primary label, with a narrower secondary finding if justified:

1. **Supported contribution:** the specified local criterion passes valid comparisons, counting all costs. State model/data/backend/target and missing external baselines. A mechanism-specific claim requires its ablation.
2. **Informative negative result:** valid implementation and adequate evidence oppose the specified hypothesis under tested conditions, or a verified obstruction resolves an explicitly stated restricted claim.
3. **Practical infeasibility:** measured costs or a supported scaling forecast prevent the intended experiment/application within caps.
4. **Inconclusive evidence:** uncertainty, missing access, inadequate horizon, incomplete matrix, or limited budget prevents the decision.
5. **Invalid experiment:** implementation, reference, data, or measurement defects invalidate the attempted comparison.

These labels are not interchangeable. A timeout, bug, or non-significant difference is not automatically a scientific refutation. Preserve unfavorable results. Recommend at most one next step, tied to the dominant unresolved obstacle, rather than a broad roadmap.

---

## 11. Minimal implementation handoff

### 11.1 Stack and repository

Use Python 3.11 or newer supported by the target's stable torch wheel, PyTorch, NumPy, SciPy, PyArrow, psutil, Matplotlib, and pytest. Do not install transformers/datasets/bitsandbytes merely to use one optimizer. Vendor only the inspected ordinary GaLore files and required license notice; remove packaging-only imports with a documented patch and CPU equivalence test.

The following is a structure to **create**, not an existing repository:

```text
poc/
  README.md
  POC_EXECUTION_BRIEF.md
  config.json
  requirements.lock.txt
  protocol.lock.json
  src/
    model.py
    data.py
    history.py
    controller.py
    estimator.py
    train.py
    measure.py
    study.py
    analyze.py
    cli.py
  vendor/galore/
  tests/
    test_math.py
    test_model.py
    test_no_leakage.py
    test_reference.py
    test_checkpoint.py
  data/manifest.json
  results/
    budget_ledger.jsonl
    development/
    confirmation/
    diagnostics/
  report/
    REPORT.md
    tables.csv
    figures/
```

Use JSON configuration and CSV/JSONL results. No database or service.

| Module | Responsibility and minimum interface |
|---|---|
| `data.py` | Download/verify pinned files; preprocess bytes; generate split/evaluation manifests and deterministic batch-position streams |
| `model.py` | Model and loss; optional detached activation-RMS return; parameter-count assertion |
| `history.py` | Owned exact-gradient storage, norms/ages/Gram cache, prediction decoding and scalar labels |
| `controller.py` | `predict(features, memory_summary) -> coefficients, normalized_risk_score`; small supervised update |
| `estimator.py` | `choose_probability(state, risk)`; `form_estimate(cached_prediction, true_gradient_or_none, p, A)`; no model differentiation |
| `train.py` | One method/run, strict ordering, optimizer stepping, checkpoint/RNG state |
| `measure.py` | MPS synchronization, timers, counters, memory sampling, cumulative budget enforcement |
| `study.py` | Gates, bounded trials, locking and deterministic run order; resume without resetting costs |
| `analyze.py` | Status-aware tables, paired statistics, diagnostics, plots and report |
| `cli.py` | Thin command interface; no alternate implementation of scientific logic |

An unqueried path must make `true_gradient_or_none=None` explicit. Do not let controller code access model gradient buffers. Keep reference and candidate update code independently checkable rather than one branch-heavy function that can reproduce the same bug in both.

### 11.2 Commands the agent must implement and run

Before creating the environment, select an installed **ARM64 Python 3.11+ interpreter with compatible torch/PyArrow binary wheels**, record its absolute path and version, and use it to create the venv. The `python3` command below must resolve to that verified interpreter. If none is available within the setup allowance, report that precise environment block; do not accidentally start a long source build. Create the prescribed `poc/` files and run the following commands from that project root.

These command interfaces are prescribed, **not commands already verified on the Mac**:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --only-binary=:all: torch numpy scipy pyarrow psutil matplotlib pytest
python -m pip freeze > requirements.lock.txt

PYTORCH_ENABLE_MPS_FALLBACK=0 python -m src.cli inspect --out results/environment.json
python -m src.cli data --config config.json
python -m pytest -q
PYTORCH_ENABLE_MPS_FALLBACK=0 python -m src.cli smoke --config config.json --steps 16
PYTORCH_ENABLE_MPS_FALLBACK=0 python -m src.cli calibrate --config config.json
PYTORCH_ENABLE_MPS_FALLBACK=0 python -m src.cli develop --config config.json
python -m src.cli lock --config config.json --out protocol.lock.json
PYTORCH_ENABLE_MPS_FALLBACK=0 python -m src.cli confirm --protocol protocol.lock.json
python -m src.cli analyze --protocol protocol.lock.json
python -m src.cli report --protocol protocol.lock.json
```

The `lock` command must refuse to freeze unresolved required decisions or failed core correctness gates. A documented G block waives only G-specific checks and removes its five confirmation runs/comparisons; it does not waive any A–F requirement. If a required A–F row has no valid selected configuration, preserve the evidence and stop the primary comparison rather than inventing a setting. The `confirm` command must reject a changed code/config hash, enforce remaining time/attempt slots, and append every attempted run to the ledger before execution. It must not restart the study on resume.

If stopped early, `analyze` and `report` must accept a stage-result manifest instead of demanding a nonexistent full comparison. Implement this ordinary case; do not fabricate empty successful rows.

### 11.3 Configuration and raw-result contents

The initial config must contain the fixed defaults in this brief and explicit fields for calibration-produced choices. Those choices are resolved by the stated rules before locking, not by free-form agent invention.

Every raw run record must include method/version, seed/substream seeds, code/protocol hashes, actual device/dtype, model size, selected rates, query-rule parameters, data/evaluation hashes, start/end times, status, reason, initial/final loss, target, crossing bracket, counts, resource maxima, clipping summaries, and checkpoint identity.

Per-step logs may be buffered and limited to scalar sufficient statistics. Full gradients are not required research artifacts. Preserve chronological query probabilities and outcomes, pre-query risks/residuals when revealed, updates, and validation events. Limit log volume without discarding failures.

### 11.4 Final artifacts for the stages actually reached

Always deliver:

- `REPORT.md` with decision, scope, gates reached, claims/evidence, resource use, limitations, deviations, and one justified next step.
- Reproducible runnable source and concise setup/use instructions for implemented stages.
- Frozen environment/source/data identifiers and the cumulative ledger.
- Tiny-check results or a verifiable analytic obstruction if those resolved the project.

If neural experiments occurred, also deliver raw scalar results including failures, target/terminal checkpoints as feasible within disk cap, the locked protocol, tables, and essential figures:

1. Validation loss versus synchronized elapsed time, with target line and censored runs visible.
2. Per-seed time-to-quality and exact-backward-count comparisons.
3. Prediction residual decomposition and query/clipping behavior if used in interpretation.
4. Cost and observed-memory comparison, with measurement definitions.

Use static scientific plots. A report generator must regenerate essential tables/plots from raw logs without retraining. Package the final local repository/report into one reproducible archive if the delivery environment supports it; do not include the downloaded corpus or environment binaries unnecessarily.

For an analytic/data/backend early stop, require only artifacts applicable to that stop. No empty benchmark framework is needed to make the outcome look complete.

---

## 12. Research context, claim map, and limits

### 12.1 Closest work and what it changes

The search date was **4 November 2026**. Searches covered synthetic gradients, direct weight-gradient prediction, gradient reuse/extrapolation, learned history recombination, control-variate prediction, GaLore-family subspaces, MPS execution, and data provenance. Relevant paper sections and official implementations were read where accessible. This was a decision-focused search, not an exhaustive novelty certification.

| Prior work | Decision-relevant distinction |
|---|---|
| Synthetic gradients / DNI [S2] | Predicts activation-error signals and learns from true feedback; local parameter differentiation can remain. Establishes learned feedback, but is not identical to omitting all target-model backward work |
| ADA-GP [S3] | Already uses a small weight-gradient predictor, exact-gradient warmup, and alternating predicted/backpropagation phases. Its adaptivity includes a staged GP:BP schedule; hardware/vision results do not establish native Mac language-model timing |
| PGD v1 [S7] | Fits a linear/structured predictor and uses true/predicted gradients on control subbatches; differs in predictor inputs and within-batch backward allocation. A same-information affine history predictor is a controlled adaptation, not a reproduction |
| PGD v2 [S8] | Materially revised: reduced-precision hand-written reverse mode and a discounted simulated inference-fleet ledger. It does not supply evidence of elapsed-time savings on one Mac |
| GaLore [S4] | Computes current gradients and compresses optimizer work/state. It does not remove current backpropagation |
| GoLore [S5] | Shows stochastic convergence pitfalls and explicitly re-expresses momentum when bases change; its stated MSGD assumptions do not cover this AdamW predictor |
| GaLore 2 [S6] | Addresses decomposition overhead and sign indeterminacy; basis consistency alone is established prior art |
| InGaLore [S9] | Incremental SVD and basis-update/loss-spike analysis overlap the proposed motivation. Full method was inaccessible, so no invented implementation is permitted |
| Recent learned history recombination [S13] | The indexed primary abstract already describes scalar coefficients over gradient-history averages. Full text was inaccessible; it further prevents claiming that scalar-history recombination is new |
| Topological frame/canonicalization and adaptive measurements [S14–S15] | Show why continuity and adaptivity models matter. They do not confer neural-training speed guarantees |

**Excluded baseline claims:** a complete ADA-GP or PGD v1 reproduction would require substantially different predictor/information interfaces and, for the published demonstrations, different workloads. InGaLore's full algorithm was not verified. GoLore's convergence result is not an AdamW guarantee. This brief therefore authorizes an empirical local feasibility result against its explicit baselines, not a practical state-of-the-art advance across these methods.

If the local result is positive, a faithful closest-method comparison is a necessary future step before a broad publication or deployment claim. If it is negative, it can still explain whether the cheap causal-history regime is worth further investigation.

### 12.2 Central claim-to-evidence map

| Claim | Status/source | Remaining uncertainty | Resolving evidence |
|---|---|---|---|
| The source's conditional risk law holds under its stated model | Source analytic result; audited, not formally verified [S1] | Historical priority; general-dimensional sharp coefficient | Outside core PoC; report exact scope |
| Source memory can communicate a known rank-one signal | Source result with exact projector/current scalar [S1] | Whether useful neural gradients are predictable | Fresh-batch real-text pilot |
| Four stored gradients supply useful current directions | New hypothesis | Staleness, sample noise, drift | Oracle span residual plus closed-loop comparison |
| Neural coefficients improve over cheap alternatives | New empirical hypothesis | Need for nonlinear features | E vs C, B; equal information and charged training |
| Learned residual risk improves query allocation | New empirical hypothesis | Miscalibration, shifting residual distribution | F vs E, realized counts, IPW risk diagnostics |
| Raw estimator (M1) is conditionally unbiased | Direct derivation and enumerated checks | Finite-precision/implementation ordering | Tiny identity and anti-leak tests |
| Applied AdamW steps converge or are unbiased | **Not claimed** | Nonlinear clipping/moment interactions | Empirical quality/stability only |
| Avoiding backwards reduces total time | New hardware-dependent hypothesis | Extra steps, CPU/GPU barriers, memory traffic | Synchronized time to frozen quality |
| A positive local result is a new global algorithmic advance | **Not established** | Strong prior art and missing faithful comparisons | Narrow report; future study only if warranted |

### 12.3 Work actually performed during planning

The planning environment was Linux x86-64 with NumPy/SciPy, not the user's Mac. No neural model was trained, no M3 timing was measured, and no target dependencies were installed. Independent agents performed separable mathematical, literature, and experimental reviews; their findings were reconciled into this design.

Executed CPU formula/diagnostic checks included:

| Check | Observed planning result |
|---|---|
| Source continuous decoder identity | Maximum error approximately \(4.04\times10^{-16}\) |
| Source transported rank-one loop | Reconstruction error approximately \(4.74\times10^{-16}\); final frame dot initial \(-1\) |
| Six-query quadratic proxy | Loss/Gram/meta-gradient errors approximately \(2.22\times10^{-16}\), \(3.61\times10^{-16}\), \(4.51\times10^{-17}\) |
| Extra unqueried terminal-gradient caveat | Six-dimensional proxy squared-norm discrepancy approximately \(1.43\times10^{-5}\) |
| Nonlinear obstruction precision probe | Monotone-interval grid smallest singular value approximately \(3.95\times10^{-16}\); exact theorem is not a robust numerical rank assertion |
| Residual estimator enumeration | For \(g=(1,2,-1), h=(.25,-.5,1),p=.25\), query branch \((3.25,9.5,-7)\), exact weighted mean \(g\), MSE \(32.4375=3\|g-h\|^2\) |
| Deliberate branch-dependent prediction leak | Bias norm approximately 2.466; test detects the defect |
| IPW regression and projection decomposition | Arithmetic identities matched to displayed floating-point precision |
| Parameter/memory counts | Default 3,257,856 parameters; Adam-array and H4 accounts each 49.7109375 MiB |
| Query-cap arithmetic | Minimum stationary fraction .2778126325; fixed coin .4979829448 for half queries |
| Illustrative cost calculation | 24% extra updates exhaust a 10% gain for the explicitly chosen \(F,B,U,H,q\) example |

The zero-norm autograd precaution was checked against official PyTorch documentation [S12]; its executable Torch test remains assigned to the coding agent because Torch was unavailable in the planning environment. These calculations support the specification and expose pitfalls. They are not experimental evidence that the proposed neural controller works. The supplied 250,000-sample Monte Carlo figures were read as reported results; the full Monte Carlo suite was not rerun as a neural or hardware benchmark.

---

## 13. Adversarial reviews and stress-test disposition

Three distinct post-draft reviews of this brief—mathematics/novelty, experimental design, and execution from an empty project—were completed, and the affected sections were rechecked after repair. Apply the same questions to the implemented study before its final report. The disposition below records material decisions, not a guarantee of perfection.

### 13.1 Mathematical and novelty review

Check the exact information order, the conditional expectation derivation, predictor scale/cap, inverse-propensity supervision, zero-norm/rank-deficient histories, and attribution. Verify that no source theorem is promoted into a convergence or speedup claim. Challenge claimed novelty against ADA-GP, both PGD versions, and recent history recombination.

Material issues found and repaired in planning:

- The attachment supplies current gradients/subspaces: replaced the unsupported inference with explicit empirical hypotheses.
- Frame transport and history mixing already have close precedents: removed algorithmic-priority claims.
- Arbitrary basis rotations do not preserve diagonal Adam covariance: retained ambient optimizer state.
- Updating the predictor after a query can invalidate residual correction: fixed pre-query caching and storage ownership.
- The queried branch is not an exact-gradient update: changed terminology and made the estimator explicit.
- Log-Huber risk training does not estimate arithmetic mean energy: qualified the gate as a score-based heuristic and separated propensity calibration from risk calibration.
- Zero-norm square-root derivatives and nonfinite fallbacks can corrupt learning: floor before the square root, skip invalid supervision, and cache only effective fallback values.
- Risk supervision could change the prediction ablation through shared hidden gradients or clipping: detached its hidden input, used disjoint gradient caps, and added an update-equivalence check.
- Unbiased local diagnostics do not automatically remain unbiased after random target stopping: restricted that claim to predeclared fixed-length windows.

### 13.2 Experimental review

Try to explain a positive result by weak baselines, extra labels, unequal tuning, skipped setup, target choice, stale-data repetition, censoring, or instrumentation barriers.

Material issues found and repaired:

- A cached predictor need not execute a forward on cheap steps: its fast path is allowed.
- A confidence-selected sample misses confident errors: randomized queries and inverse-propensity losses/diagnostics are mandatory.
- The skip cap changes query frequency: target actual fractions and correct the fixed-probability comparator.
- Initial learning can finish before memory operates: target uses post-warmup progress and a minimum crossing time.
- Low gradient cosine may reflect stochastic noise: add paired-minibatch and span diagnostics, and require closed-loop evidence.
- A timeout is not an attained runtime: preserve censoring and prohibit complete-case primary speed claims.
- Validation target reuse can overstate generalization: reserve an untouched test panel with explicit quality tolerance against both A and frozen R.
- A 512-update development cap would give fast methods less training time: replaced it with equal 120-second allocations and explicit overrun accounting.
- Allowing R to be A would remove the promised strong predictor comparison: select R only from B/C and require both speed comparisons.
- A weak single-seed pilot cannot establish the confirmatory negative: added explicit negative/inconclusive interval rules and restricted early-stop claims.

### 13.3 Execution review

Walk through the commands from an empty project on the actual Mac. Verify data checksums, supported operations, no automatic fallback, complete ledger accounting, bounded repairs, sufficient reporting reserve, and ability to report an early stop.

Material issues found and repaired:

- Post-forward decisions create real synchronization costs: batch transfers and count them; do not slow the baseline artificially.
- A no-grad gate followed by recomputation has a different cost equation: core implementation uses one tracked forward.
- MPS memory APIs expose current allocations, not a universal peak counter: label observed maxima and avoid double counting.
- A parameter-count claim does not predict peak memory: enforce measured caps and minimum scientifically meaningful size.
- Missing InGaLore/recent full text cannot be filled by guesswork: mark coverage limits and narrow comparison claims.
- Development continuation required discarded optimizer state: allow temporary A-selection checkpoints and accumulate resumed active costs.
- Horizon extension and cadence selection were circular: use provisional cadence 16, preserve the largest feasible horizon, and stop if its target is uninformative.
- First synchronization and numerical checks could distort baseline timing: start the cold timer before any MPS call and inspect device-side finite flags at declared boundaries.
- Projector fallback and pilot diagnostics could hide extra trials: choose G's projector before scoring, count smoke/calibration/teacher-forced attempts explicitly, and retain the cumulative 480-minute cap.

### 13.4 Required stress cases

| Scenario | Predeclared response |
|---|---|
| Theorem is nonconstructive or supplies an expensive oracle | The PoC does not call the theorem's encoder/projector; audit current-gradient access; stop if a hidden oracle appears |
| Neural gradients violate low-dimensional temporal structure | Measure oracle history-span residual; do not blame only the learner |
| Mechanism is equivalent to known methods | Attribute it and report the new empirical regime/limitation; do not rename a reproduction as novelty |
| Only favorable synthetic/repeated-batch cases succeed | No primary training-efficiency contribution |
| Tuned simple baseline removes the gain | Report that extra learned machinery is unjustified in this regime |
| Setup/history/preprocessing dominates | Include cost; no unpriced amortization |
| Float32 changes behavior | Compare to CPU reference; fix common operator or narrow/stop the backend claim |
| Memory/runtime exceeds estimates | Apply the prescribed resource-only reductions before lock, or report infeasibility |
| Pilot is inconclusive | Use only the bounded allocated diagnostic/development work; no indefinite rescue |
| Essential dataset is inaccessible | Same-file checksum-verified copy or blocked real-text study |
| InGaLore/recent paper remains inaccessible | Preserve the documented research limit; no fabricated baseline |
| Main hypothesis is false | Produce a valid negative result with failure attribution and reproducible evidence |

Do not assert that the study or brief is flawless. Unresolved risks include generalization from a short byte-level training run, small seed count, MPS implementation sensitivity, estimated rather than certified risk, and incomplete comparison to external predictor architectures. Those risks constrain the conclusion; they do not justify concealing results.

---

## References

Ordinary references below are sufficient without chat-specific citation tokens. Access/search date for external sources: **4 November 2026**. Where a work has materially different versions, use the versioned URL.

**[S1] Supplied unpublished research package.** *A sharp stability law for continuous feedback compression: One-dimensional task structure, narrow learning signals, and the role of memory.* Author field: “Mathematical investigation prepared in this conversation.” 4 November 2026. Archive `neural_feedback_research(1).zip`; main `feedback_research_note.pdf/.tex`; companion `spectral_proxy_appendix.tex`. Critical locations: Proposition 1 pp.2–3; Theorem 2 pp.3–5; Theorem 3 pp.5–6; equations (22)–(23) pp.6–7; Appendix Theorem 4 p.10 and Theorem 5 pp.11–12. SHA-256 and complete file inventory are in Section 2.

**[S2] Max Jaderberg, Wojciech Marian Czarnecki, Simon Osindero, Oriol Vinyals, Alex Graves, David Silver, Koray Kavukcuoglu.** *Decoupled Neural Interfaces using Synthetic Gradients.* ICML 2017, PMLR 70:1627–1635. Relevant §§2.2 and 2.4. [Proceedings](https://proceedings.mlr.press/v70/jaderberg17a.html), [full paper](https://proceedings.mlr.press/v70/jaderberg17a/jaderberg17a.pdf). Relevant full-text sections read.

**[S3] Vahid Janfaza, Shantanu Mandal, Farabi Mahmud, Abdullah Muzahid.** *ADA-GP: Accelerating DNN Training By Adaptive Gradient Prediction.* MICRO 2023, 1092–1105. Relevant §§3.1–3.6 and hardware/evaluation §§4–5. [DOI](https://doi.org/10.1145/3613424.3623779), [author-hosted full paper](https://people.tamu.edu/~abdullah.muzahid/files/adagp.pdf). Full mechanism and adaptive schedule read; no official code repository was verified.

**[S4] Jiawei Zhao, Zhenyu Zhang, Beidi Chen, Zhangyang Wang, Anima Anandkumar, Yuandong Tian.** *GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection.* ICML 2024, PMLR 235:61121–61143. Relevant §§4.1–4.2, Algorithms 1–2. [Proceedings](https://proceedings.mlr.press/v235/zhao24s.html), [paper](https://arxiv.org/abs/2403.03507), [official implementation](https://github.com/jiaweizzhao/GaLore). Optimizer/projector source inspected at `galore_torch/adamw.py` and `galore_torch/galore_projector.py`; target execution unverified.

**[S5] Yutong He, Pengrui Li, Yipeng Hu, Chuyan Chen, Kun Yuan.** *Subspace Optimization for Large Language Models with Convergence Guarantees.* ICML 2025, PMLR 267:22468–22522; arXiv:2410.11289v2, 4 June 2025. Relevant §3/Theorem 4, §4 equation (2), §5/Algorithm 1/Theorem 8. [Proceedings](https://proceedings.mlr.press/v267/he25i.html), [full PDF](https://arxiv.org/pdf/2410.11289), [official code](https://github.com/pkumelon/Golore). Relevant PDF sections read.

**[S6] DiJia Su, Andrew Gu, Jane Xu, Yuandong Tian, Jiawei Zhao.** *GaLore 2: Large-Scale LLM Pre-Training by Gradient Low-Rank Projection.* arXiv:2504.20437, 29 April 2025. Relevant §§4.1.2–4.1.3 on randomized SVD and sign indeterminacy. [Versioned full text](https://arxiv.org/html/2504.20437v1). Relevant sections read; no distributed implementation reproduced.

**[S7] Kamil Ciosek, Nicolò Felicioni, Juan Elenter Litwin.** *Linear Gradient Prediction with Control Variates.* arXiv:2511.05187v1, 7 November 2025. Relevant Algorithm 1 and §§3–5.3, §7. [Versioned full text](https://arxiv.org/html/2511.05187v1). This version's linear/structured predictor and experiment must not be conflated with v2.

**[S8] Kamil Ciosek, Nicolò Felicioni, Juan Elenter, Ehsan Imani.** *Gradient Prediction with Control Variates in the Cheap-Forward Regime.* arXiv:2511.05187v2, 2 September 2026. Relevant §§1–3.2 and Appendix A.1. [Versioned full text](https://arxiv.org/html/2511.05187v2), [DOI](https://doi.org/10.48550/arXiv.2511.05187). Relevant resource model and estimator read. Discounted ledger results are not measurements of this PoC's elapsed-time objective.

**[S9] Xinmin Meng, Muhammad A. A. Abdelgawad, Peng Jing, Ray C. C. Cheung, Hong Yan.** *Memory-efficient neural network training via gradient compression through continuous basis tracking.* Neurocomputing 680 (2026), article 133278; online 7 March, issue 1 June 2026. [DOI](https://doi.org/10.1016/j.neucom.2026.133278), [publisher](https://www.sciencedirect.com/science/article/pii/S0925231226006752), [university record](https://scholars.cityu.edu.hk/en/publications/memory-efficient-neural-network-training-via-gradient-compression/). University abstract and publisher-indexed abstract/introduction/section excerpts inspected; direct publisher full text returned 403. Full algorithm and independent timing validation unavailable.

**[S10] Gene H. Golub, Gérard Meurant.** *Matrices, moments and quadrature.* Author-hosted manuscript `mmq_1993.pdf`, 1993 manuscript identifier. Relevant §§2–4 on spectral measures, quadrature, and Lanczos. [Full manuscript](https://gerard-meurant.fr/mmq_1993.pdf). Supports the classical origin of the appendix's quadratic construction; not a neural speedup source.

**[S11] Stephen Merity, Caiming Xiong, James Bradbury, Richard Socher; Salesforce dataset maintainers.** *Pointer Sentinel Mixture Models* (2016) and WikiText dataset release. [Paper](https://arxiv.org/abs/1609.07843), [dataset](https://huggingface.co/datasets/Salesforce/wikitext), [pinned Parquet conversion commit, 4 January 2024](https://huggingface.co/datasets/Salesforce/wikitext/commit/b08601e04326c79dfdd32d625aee71d232d685c3). The commit's `wikitext-2-raw-v1` LFS entries supply sizes/checksums. Metadata read; data bytes not downloaded during planning.

**[S12] PyTorch contributors.** *PyTorch MPS backend, MPS environment variables, MPS APIs, and AdamW documentation.* Version 2.15 pages reached from stable documentation, accessed 4 November 2026. [Backend](https://docs.pytorch.org/docs/2.15/notes/mps.html), [MPS API](https://docs.pytorch.org/docs/2.15/mps.html), [environment](https://docs.pytorch.org/docs/2.15/mps_environment_variables.html), [tensor memory](https://docs.pytorch.org/docs/2.15/generated/torch.mps.current_allocated_memory.html), [driver memory](https://docs.pytorch.org/docs/2.15/generated/torch.mps.driver_allocated_memory.html), [events](https://docs.pytorch.org/docs/2.15/generated/torch.mps.event.Event.html), [AdamW](https://docs.pytorch.org/docs/2.15/generated/torch.optim.AdamW.html), [autograd mechanics](https://docs.pytorch.org/docs/2.15/notes/autograd.html) (non-differentiable functions and invalid-operation masking). Documentation read; installed target version/capabilities require inspection.

**[S13] Minyoung Choi, Dalta Imam Maulana, Wanyeong Jung.** *Lightweight and Versatile Learned Optimization by Recombination of Gradient History.* arXiv:2610.09604, 7 October 2026. [Primary record](https://arxiv.org/abs/2610.09604). Indexed primary abstract inspected; direct full-text retrieval failed. Cited as unresolved recent overlap, not as a verified skipped-gradient method or reproduced baseline.

**[S14] Nadav Dym, Hannah Lawrence, Jonathan W. Siegel.** *Equivariant Frames and the Impossibility of Continuous Canonicalization.* ICML 2024, PMLR 235:12228–12267. Relevant §2 and continuous-frame/canonicalization model. [Proceedings](https://proceedings.mlr.press/v235/dym24a.html), [full paper](https://arxiv.org/pdf/2402.16077). Related mathematical context; not a claim that the supplied coefficient has established priority.

**[S15] Mario Ullrich.** *A Borsuk-Ulam theorem for adaptive continuous measurements.* arXiv:2610.33781v1, 24 October 2026. Relevant §1, Theorem 1, and distinction between continuous functionals and discontinuous adaptive selection. [Versioned full text](https://arxiv.org/html/2610.33781v1). Relevant result read; its adaptive-query model differs from both the supplied channel theorem and neural gradient prediction.

## Final instruction to the executing agent

Execute the authorized stages autonomously until a gate resolves the study or a genuine indispensable resource is blocked. Preserve the evidence needed to distinguish success, a useful limitation, infeasibility, inconclusiveness, and invalid experimentation. Report only work actually performed. A justified early stop with reproducible evidence is a completed R&D outcome.

