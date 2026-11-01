# EIAL-G: usefulness-first proof of concept

**Version:** 2.0 — reviewer-driven replacement plan  
**Date:** 13 October 2026  
**Target:** one MacBook Pro with an M3 Max; discover its actual configuration.  
**Workload:** UCI SMS Spam Collection, frozen `sentence-transformers/all-MiniLM-L6-v2`, binary affine classification.  
**Methods:** logistic regression plus threshold tuning; exact greedy event search; EIAL-G.  
**Status:** executable research specification, not an implemented benchmark or a positive experimental result.

This document is a complete handoff to a coding agent. It contains the problem, algorithms, data rules, evaluation protocol, resource gates, implementation interfaces, and decision rules. No earlier document or conversation is required. Project commands below are interfaces to implement, not commands claimed to exist already.

## 1. Decision and analysis of the review

### 1.1 Verdict

**Accept the review and change the order and scope of the PoC. Keep the mathematical mechanism; move practical usefulness ahead of broad infrastructure and attribution work.**

The question for this stage is:

> On the same cached MiniLM features, can EIAL-G produce a worthwhile held-out quality–cost advantage over both a well-tuned logistic classifier and ordinary exact greedy search?

The earlier plan already recognized this question, proposed SMS plus MiniLM as its first cache, and explicitly admitted that neither real-data superiority nor M3 performance had been measured. Its sequencing nevertheless required numerous additional optimizers, sampling controls, constraint machinery, and ultimately a multi-dataset panel before the central comparison was resolved. The review identifies a sequencing defect, not a refutation of the mathematics.

Mathematical checks answer whether event enumeration and the conditional integral are implemented correctly. They do not answer whether the resulting classifier is useful. Likewise, avoiding backpropagation is not itself an advantage when all three methods train small heads on the same cached features.

### 1.2 What changes

| Earlier scope or ambiguity | Replacement in this PoC | Reason |
|---|---|---|
| Broad benchmark and many mandatory baseline families | Exactly three method families on one dataset and one encoder | Resolve the cheapest decisive comparison first. |
| Initially reduced-dimensional heads, followed by capacity sweeps | Use all nonconstant MiniLM embedding coordinates for all three methods | Avoid a PCA bottleneck or a deliberately restricted logistic baseline. This is an explicit capacity change, not a hidden optimization. |
| Four-way fit/selection/certification/test allocation | Approximately 60% fit, 20% selection, 20% protected test, grouped before splitting | No operational risk certificate is claimed at this stage. Preserve data for the quality comparison. |
| FPR constraints and groupwise certification | Unconstrained positive-class F1; still report false positives, precision, and recall | Isolate metric learning without a second hypothesis about constraints. |
| Many sampler and optimizer controls before practical evidence | Shared greedy engine now; further regularization/sampling controls only after a signal | Three methods can screen usefulness, but cannot fully attribute causality. |
| Broad continuation after correctness and cache gates | No automatic expansion after implementation success | Continuation depends on protected quality/cost evidence, not a working sampler. |
| Potentially ambiguous warm-start cost | One explicitly charged, fixed default logistic initializer | EIAL-G must not receive an expensive fitted classifier for free. |

The integral component is **not** replaced with a deterministic plateau-width heuristic. EIAL-G retains an exact one-dimensional conditional Gibbs update. The global empirical target remains Gaussian-referenced and F1-weighted; this stage simply omits its feasibility indicator.

### 1.3 Qualifications to the review

A single SMS result is an economical screening result, not proof about every task where integral-based learning might help. A null result with a wide interval is inconclusive, not equivalence. A result against these two baselines does not establish competitiveness with every spam classifier. A positive result also does not prove that analytic integration, rather than Gaussian regularization or stochastic exploration, caused the gain.

“Worthwhile” must not be invented after seeing results. No user-specific false-alarm cost, retraining frequency, latency requirement, or minimum useful gain has been supplied. Section 8 therefore separates statistical evidence, a limited research-continuation decision, and business usefulness. It does not substitute an arbitrary “1 F1 point” or “2× speedup” target for missing requirements.

### 1.4 Immediate execution boundary

Implement and evaluate the three-method comparison. Do **not** initially implement SVMs, CMA-ES, MeZO, SPSA, smooth-F1 training, elliptical slice sampling, additional encoders, additional datasets, multiclass heads, LoRA, production certificates, or a generic optimizer framework. Do not commission a new comprehensive novelty survey before this screen. Those are conditional follow-on work, not hidden prerequisites.

## 2. Research question, evidence, and permissible claims

### 2.1 Hypotheses

**H-quality:** At a predeclared project-specific fitting/tuning budget, EIAL-G improves protected-test positive-class F1 over both comparator pipelines.

**H-cost:** At a predeclared acceptable quality level, EIAL-G reaches an acceptable classifier with lower fully charged cost than the available simpler alternative. Failure to detect an F1 difference does not establish acceptable quality or noninferiority.

**H-engine:** The event engine obtains the correct empirical decision counts along a line, and EIAL-G uses the corresponding analytically integrated conditional law. This is a prerequisite, not the success criterion.

Plausible negative explanations are that logistic regression plus threshold tuning already captures the available signal; greedy search obtains the same gain without Gaussian integration; the Gaussian search mostly reproduces its initializer; or exploration costs more than any gain is worth. All are legitimate outcomes.

### 2.2 Evidence status

The supplied earlier plan reported small mathematical/numerical checks, including 400 small event problems. This revision does not treat those as real-data results, does not certify their production implementation, and does not rerun or newly claim those measurements. The earlier plan itself explicitly stated that real-data competitiveness and M3 performance were unmeasured.

For this revision, the earlier plan and reviewer text were inspected, and the official dataset/model cards and relevant library documentation were checked. No MiniLM encoding, three-method training run, protected-test comparison, or M3 benchmark was performed. A direct dataset retrieval attempt in the authoring runtime failed; actual archive hashes and parsed row counts remain execution-time checks. No dataset count below is presented as an observed download count.

### 2.3 Prior art and novelty boundary

Exact discontinuous line search is not the proposed novelty. Minimum error rate training is an important predecessor; the original assessment identified the exact line-search construction in Och's 2003 work [R1]. Logistic fitting followed by decision-threshold tuning is an established separation of estimation from decisions [R4]. Gaussian/Gibbs weighting is also an existing construction; this plan makes no invention claim for it.

The possible contribution tested here is an **empirical improvement from a particular specialized search procedure on cached neural heads**. Even a positive result supports only a statement naming this dataset, encoder, split, method definitions, and cost regime. An exact line integral is not exact multidimensional optimization, and neither constitutes a general replacement for neural-network backpropagation.

The review changes priorities, not the unresolved novelty assessment. A fresh method-level prior-art audit becomes necessary before a novelty claim or publication submission, not before this usefulness screen.

## 3. Data, representations, and access boundaries

### 3.1 Dataset and immutable provenance

Use the **UCI SMS Spam Collection**, dataset 228, DOI `10.24432/C5CC84`, with spam as label 1 and ham as label 0 [R2]. The official card lists 5,574 instances and a CC BY 4.0 license. The card describes raw labeled messages assembled from several sources and says the messages are not chronologically sorted. This is not a prospective deployment sample.

Use the download linked by the official card. Save the original archive, the extracted `SMSSpamCollection` bytes, the readme, retrieval date, source URL, SHA-256 hashes, and attribution. Parse each nonempty physical record at its first tab; preserve the remaining message text, including internal tabs. Use strict UTF-8 decoding first; on failure inspect the raw bytes and record an explicit decoding decision rather than silently replacing characters. Validate every label and nonempty message.

Report actual raw, parsed, invalid, and retained counts by class. A discrepancy with the card is an audit item, not permission to add or remove rows to force 5,574. Invalid records are quarantined with an explicit reason before splitting. Do not drop difficult valid examples.

### 3.2 Duplicate/template grouping before splitting

Keep raw message text for encoding. Separately construct a deterministic grouping key:

1. Unicode NFKC normalization, case folding, and whitespace collapse.
2. Replace URL-like spans matching `https?://\S+|www\.\S+` with `<URL>`.
3. Replace each Unicode digit run `\d+` with `<NUM>` and collapse whitespace again.

Identical resulting keys receive the same group ID. This also groups exact normalized duplicates. Keep all valid rows, including mixed-label groups; report such conflicts rather than resolving them using model predictions. Compute group IDs from key hashes and stable row IDs from archive identity plus record position. Group construction uses no model scores.

This rule catches only the specified exact/template equivalences. It is not a guarantee that all near-duplicates, shared authors, or shared spam campaigns have been identified. Save group sizes, class counts, largest groups, and mixed-label counts. Unknown dependence remains a limitation. Do not inspect protected error examples to invent a more favorable grouping rule.

### 3.3 Primary split

Sort retained rows by stable ID. Use a pinned implementation of `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=20261013)` with the group IDs. Assign fold 0 to **test**, fold 1 to **selection**, and folds 2–4 to **fit**. Approximate proportions are 20/20/60; unequal groups can prevent exact proportions. Store the resulting row-to-split mapping, not merely the seed.

Both classes must occur in each role, and no group may cross roles. If allocation is infeasible, stop before training and record a split-protocol amendment. Do not search seeds until a favorable classifier result appears. Report class and group counts; stratification is not an extra source of training labels.

The test role must be logically inaccessible to training, tuning, and candidate selection. Its labels may be used by the split-builder for stratification, but not by optimizer or pilot code. Data already exist publicly, so this is a reproducible process boundary, not a claim of cryptographic secrecy.

### 3.4 Pilot data are inside fit

Before final fitting, make a group-stratified four-fold division of **fit only**, with seed `20261014`. Use one fold as `pilot_eval` and the other three as `pilot_train`. Use this solely for implementation debugging, range checks, runtime calibration, uncertainty calibration, and freezing the final experiment policy.

Pilot scalers and initializers use only `pilot_train`. Final scalers and initializers use all of `fit`. Neither selection nor test outcomes may influence the pilot protocol. The pilot is not an additional dataset, and its performance is not confirmation.

### 3.5 Frozen encoder and features

Use `sentence-transformers/all-MiniLM-L6-v2`, pinning an immutable model repository commit and tokenizer/config files. The model card specifies 384-dimensional embeddings, documents attention-mask-aware mean pooling and normalization, and notes default truncation beyond 256 word pieces; the repository carries an Apache-2.0 label [R3]. Record the actual locked configuration and license. Do not silently change pooling, maximum length, encoder, or model precision.

Encode raw text once, in evaluation/inference mode, with float32 model computation and L2-normalized float32 output. Explicitly set and record a 256-token maximum sequence length; count truncation without consulting labels. Use one backend and one cache for all methods. Encoding protected text with a fixed, label-independent encoder is allowed; fitting a corpus transform on protected text is not.

For the final fit, compute coordinate means and population standard deviations (`ddof=0`) from fit embeddings, in float64. Drop exactly zero-variance coordinates for every method; reject nonfinite data. Standardize retained coordinates with these fit statistics. Do not fit PCA, select coordinates using labels, or tune a dimension. Preserve the entire retained feature space, normally 384 coordinates. Append an intercept only in the event-method matrix:

\[
 h_i=[x_i;1]\in\mathbb R^{d+1}.
\]

Pass `x_i` without the appended constant to logistic regression with `fit_intercept=True`. All three methods therefore deploy the same affine hypothesis class. If every coordinate is constant, the task is threshold-only; record that as a representation failure for this PoC rather than proceeding to broad claims.

### 3.6 Label permissions

| Operation | Fit labels | Selection labels | Test labels |
|---|---:|---:|---:|
| Fit logistic coefficients, initializer, or event transitions | Yes | No | No |
| Fit training objective / keep training incumbent | Yes | No | No |
| Freeze pilot ranges, budgets, repeat count | Fit-only pilot | No | No |
| Final threshold and candidate/configuration selection | No additional fitting | Yes, under frozen policy | No |
| Final comparison and error analysis | No model changes | No model changes | Once artifacts are frozen |

After selection, do not refit on fit plus selection. Such a refit would change the score scale and classifier for which the threshold was selected.

## 4. Three method specifications

### 4.1 Shared objective and output

For scores `s`, predictions are `s >= 0`, with spam positive. For ordinary thresholding, use `score >= threshold` consistently.

\[
 F_1=\frac{2TP}{2TP+FP+FN}.
\]

Use int64 confusion counts and set F1 to zero when its denominator is zero. Precision or recall with a zero denominator is reported as undefined, with its denominator, not silently called perfect. Training uses unweighted positive-class F1 for both event methods. Logistic class weights affect its fit objective, not the definition of evaluation F1.

Every final candidate is one affine classifier: frozen encoder identity, feature mask/means/scales, coefficients, intercept, final threshold, IDs and hashes. No coefficient averaging, ensemble voting, posterior mixture, or additional inference-time sampling is permitted in this stage.

### 4.2 L — logistic regression with exact threshold tuning

Use scikit-learn's float64 L-BFGS L2 logistic solver. Start with `C` in `{1e-3, 1e-2, 1e-1, 1, 10, 100, 1000}` and `class_weight` in `{None, "balanced"}`: 14 candidate configurations. These are one baseline family, not extra benchmark methods. The API documents inverse-regularization `C`, balancing, and supported solvers [R5].

Set `tol=1e-8`, initially `max_iter=5000`. Treat a convergence warning or nonfinite solution as a failed fit requiring investigation; it cannot be scored as a weak baseline. On pilot data, confirm that tighter tolerance or additional iterations do not materially change coefficients' objective value or predictions. A boundary-winning `C` extends the pilot grid by one decade on that side; continue until the range is bracketed or additional extension gives identical selected predictions. Freeze the resulting grid before final selection. No grid extension after test access.

Use the L2 API appropriate to the locked version. Current documentation deprecates the `penalty` argument in favor of `l1_ratio=0`; older releases use `penalty="l2"` [R5]. Verify semantics with the actual installed version rather than copying incompatible arguments.

Fit coefficients on fit only. Score selection, optimize its F1 threshold by enumerating every distinct score transition, and select the configuration by the common rule in Section 6. Keep the default unweighted `C=1` candidate in the record even if not selected. Logistic threshold tuning must not be a coarse grid or a fixed 0.5-probability cutoff [R4].

### 4.3 Shared initializer for G and E

Fit **one default, unweighted `C=1` logistic classifier on fit**, using the same features and convergence standard as L. Do not obtain this initializer by running L's entire selection-tuned search. Reuse the identical fit artifact within an experimental run, but attribute its cost to both event pipelines as a dependency.

Let its raw scores be `f_i`. For training initialization only, optimize a fit-set threshold `t_fit` exactly. This is part of fitting, not an independently validated calibration. Define

\[
 q_s=\operatorname{IQR}_{fit}(f),\qquad
 s_i^0=(f_i-t_{fit})/q_s.
\]

Use `q_s=1` when the IQR is zero. The scale is positive, so it does not alter the initializer's decisions. Transform its coefficients and intercept consistently. The event classifier is

\[
 s_i(z)=s_i^0+h_i^\top z,\qquad z_0=0.
\]

The fitted default head plus a residual is still a single affine head. It is not extra model capacity. Include the zero-correction candidate at every budget. Final threshold calibration is again done on selection for **all** methods, including these event classifiers.

**Cost implication:** an event method cannot be faster than its own initializer when that initializer already reaches the required quality. Do not claim a faster optimizer by excluding this fit, or by charging L for a full hyperparameter sweep while comparing it with an already-tuned EIAL run. Section 7 requires both first-use and fixed-recipe cost views.

### 4.4 Shared directions and state

Use random-scan coordinate directions only: independently draw a coordinate uniformly from `{0,...,d}` at each transition. The intercept participates exactly like another coordinate. Generate and record a direction stream from its own RNG, separate from the Gibbs sampling RNG. Pair direction seeds between G and E. Neither selection results nor the current reward choose the next direction.

Both methods use the same feature arrays, event engine, score recomputation, training objective, start, and checkpoint policy. No dense directions or extra feature access are granted to E. A later direction change must be applied symmetrically and is a methodological amendment.

### 4.5 G — exact greedy event search

At state `z` and direction `v`, enumerate the full real line and every prediction-change event as specified in Section 5. Choose a step obtaining the largest fit F1 among open intervals, representable singleton events, and the current point.

Use this deterministic tie policy. If `a=0` achieves the maximum, stay at `a=0`. Otherwise form one verified finite representative for each maximum-reward open interval and include every maximum-reward representable singleton. For a bounded interval use an overflow-safe midpoint; for a half-infinite interval choose a finite point one `max(1, abs(finite_endpoint))` step into the interval, repairing overflow/rounding with checked alternatives. Choose the representative with the smallest

\[
 D(z)=d\|z_{1:d}\|_2^2+z_{d+1}^2,
\]

then smallest `abs(a)`, then smallest `a`. Record the representative rule; it is a tie-breaking convention, not Gaussian mass optimization. Verify all chosen predictions directly before accepting.

This is exact empirical line maximization wherever the advertised real prediction regions have a representable candidate under the numerical contract. It has no Gaussian CDF calculation and no integral reward weighting. Its direction choice is stochastic; its line choice is greedy.

Track coordinates checked with no state change since the last change. When every coordinate has been checked without a change, stop and report coordinatewise stationarity for this search rule. Do not keep burning the allocated budget to obscure a faster completion. This is not a global optimum certificate.

### 4.6 E — EIAL-G

For fixed positive `T` and `sigma`, define the search target

\[
 \pi(z)\propto \exp\{F_{1,fit}(z)/T\}\,\mathcal N(z;0,C),
 \qquad C=\operatorname{diag}(\sigma^2/d,\ldots,\sigma^2/d,\sigma^2).
\]

This is a proper empirical search distribution, not a calibrated Bayesian posterior. Standardized fit coordinates make the scale interpretable; they do not make the prior uniquely correct.

Along `z + a v`, let

\[
 b_i=h_i^\top v,\qquad a_i=-s_i(z)/b_i\quad(b_i\ne0).
\]

Completing the Gaussian square gives

\[
 V=(v^\top C^{-1}v)^{-1},\qquad m=-Vv^\top C^{-1}z.
\]

For each open event interval `I_j=(l_j,u_j)`, its reward `R_j` is constant. Compute

\[
 \log W_j=R_j/T+\log\left[\Phi((u_j-m)/\sqrt V)-\Phi((l_j-m)/\sqrt V)\right].
\]

Sample an interval with probabilities proportional to `W_j`, then sample `a` from `N(m,V)` truncated to that interval. Set `z <- z + a v` after direct prediction verification.

The interval probabilities use analytic Gaussian integrals; drawing the interval and step is stochastic. Random-scan conditional resampling preserves the specified target in exact arithmetic. This does not imply rapid mixing or global optimization. Do not reset the chain to its best state, reject every lower-reward state, anneal `T`, clip to a moving trust region, or select state-dependent directions while claiming this is the same sampler.

Use the pilot grid

- `sigma` in `{0.3, 1, 3}`;
- `T = c / max(1, n_positive_fit)` with `c` in `{1, 10, 100}`.

This makes nine starting configurations. Pilot boundary winners extend the relevant range by a factor of ten on that side; stop extension when bracketed or when the range becomes prior-dominated/indistinguishable with no improvement. Record the actual final grid and all pilot outcomes. Recompute the formula's positive count for each actual fit; do not carry a pilot's numeric temperature into a differently sized fit by accident.

Maintain the **chain state** and a separate **fit-only incumbent**. The incumbent is the greatest visited training F1, then smallest `D(z)`, then earliest occurrence. Save incumbents at frozen checkpoints, including the initializer. It is the incumbent candidates that undergo final selection thresholding. Do not average chain coefficients.

A win by E over G is a win for this full search procedure. It is not yet proof that analytic integration alone, rather than stochastic exploration or prior geometry, is the cause.

## 5. Minimal shared event engine and numerical contract

### 5.1 Event enumeration

Compute direct float64 scores from the current coefficients. For a coordinate direction, slopes are the corresponding feature column; do not perform a dense matrix product merely to obtain a column.

At `a=-infinity`, a negative slope predicts positive, a positive slope predicts negative, and a zero slope retains `score >= 0`. Sort roots `-score/slope` for nonzero slopes and process truly equal roots together. Before the first root, after the last root, and between successive roots, confusion counts are constant.

At a shared root `r`, with the convention `score >= 0`, there are three states:

- Left open interval: counts before the event.
- Singleton at `r`: positive-slope examples have flipped to positive; negative-slope examples remain positive at zero.
- Right open interval: negative-slope examples have now flipped to negative.

Update counts by labels, using integers. For example, a positive-labeled negative-to-positive flip increments TP and decrements FN; a negative-labeled flip increments FP and decrements TN. Do not process tied examples in an arbitrary sequential order and score those nonexistent intermediate states.

With labels `[1,1]` and scores `[a,-a]`, F1 is `2/3` on either side and `1` at zero. G must inspect that singleton when representable. E correctly assigns it zero continuous Gaussian probability. An ignored singleton must not be marketed as an exact greedy argmax.

The line sweep costs `O(n log n)` for sorting and `O(n)` for counts, with linear scratch storage. Direct score recomputation costs `O(n d)` and is included in timing. Both methods recompute after each transition in the reference implementation. A validated incremental implementation may replace this only symmetrically, with periodic direct checks and recorded timing boundaries.

### 5.2 Floating-point rules

Use float64 CPU arithmetic for head scores, roots, sorting, and Gaussian calculations. This is analytic integration for the stored-feature problem, evaluated numerically; do not claim arbitrary-precision exactness for a floating implementation.

Reject nonfinite inputs. A zero slope is an exactly zero stored slope, not one below an arbitrary threshold. Do not merge close roots with `isclose`. Use stable sorting; ambiguous or equal rounded roots must be ordered/grouped by comparing their underlying finite-float ratios using `fractions.Fraction` or another verified high-precision fallback. Overflowed roots and intervals with no representable interior require diagnostics, not silent clipping.

Compute normal interval masses in log space using `log_ndtr`, log-survival differences for positive tails, and stable `log1mexp` algebra [R6]. Ordinary subtraction of two nearly equal CDFs is insufficient. Use an 80-digit `mpmath` fallback for cancellation or suspect narrow intervals, increasing precision until results stabilize. Count and time fallbacks. A zero result from underflow is not evidence that an interval has zero mathematical mass.

SciPy's `truncnorm` expects standardized bounds `(l-m)/sqrt(V)` and `(u-m)/sqrt(V)`, not raw coefficient bounds [R7]. Validate draws with infinite endpoints and tails. If a draw rounds onto a boundary, use a high-precision inverse-CDF calculation using the same random variate, then verify the resulting float64 coefficients. Do not silently reject/resample repeatedly until the empirical reward improves.

For both methods, recompute the actual candidate predictions and compare them with the engine's promised counts. A mismatch blocks that run. If no representable coefficient vector can realize an advertised region needed by the method, record `numerical_resolution_failure`; do not silently remove positive Gibbs mass or claim exactness. Distinguish this from a scientifically poor but valid model.

Do not build a general computational-geometry library before the first real experiment. Implement the required branches, independent small fixtures, and direct checks; add further numerical machinery only for an observed failing case.

### 5.3 Implementation pseudocode

```text
prepare_line(z, fit):
    scores = base_scores + H @ z
    k = next_state_independent_coordinate()
    slopes = H[:, k]
    roots = checked_sorted_ratios(-scores, slopes)
    intervals, singleton_points = sweep_integer_counts(roots, y)
    return k, intervals, singleton_points

greedy_step(z):
    k, intervals, points = prepare_line(z, fit)
    a = best_verified_F1_step(intervals, points, current_step=0,
                             tie_policy=specified_greedy_policy)
    z_next = z; z_next[k] += a
    verify_direct_predictions(z_next)
    return z_next

gibbs_step(z, sigma, T, rng):
    k, intervals, points = prepare_line(z, fit)
    V = C[k, k]                 # coordinate case
    m = -z[k]                   # zero-centered diagonal reference
    log_weights = [R_j/T + log_normal_mass(l_j, u_j, m, sqrt(V))]
    j = categorical_from_log_weights(log_weights, rng)
    a = checked_truncated_normal(m, sqrt(V), l_j, u_j, rng)
    z_next = z; z_next[k] += a
    verify_direct_predictions(z_next)
    return z_next               # do not replace with the incumbent
```

The `z_next` assignments in executable code must copy or intentionally mutate state; do not accidentally alias a saved candidate. Save numerical arrays without pickle and persist RNG state separately.

### 5.4 Required tests, not a new proof project

Before real fitting, tests must cover: random small integer fixtures against an independent direct-prediction oracle; zero slopes; identical and opposite-slope tied roots; all-negative/all-positive labels; intercept-only moves; both infinite tails; narrow Gaussian intervals; constant-reward normal partition summing to one; interval-subdivision invariance; sample frequencies on one known 1D mixture; and interrupted-versus-uninterrupted RNG replay.

Use branch coverage and retained failing fixtures, not an impressive count of random cases, as the completion criterion. Integer count comparisons must agree exactly. For ordinary numerical fixtures require mass-sum error below `1e-10`; validate narrow/tail log masses against high precision to `1e-8` absolute log error or better. Investigate systematic errors rather than loosening tolerances to get a pass. Sampling discrepancies are assessed against their Monte Carlo standard errors, not a deterministic bit-equality expectation.

The evaluation harness must also pass an A/A identical-prediction comparison, a hand-computable known-improvement fixture, and a canary test proving that changing test labels cannot change any fitted coefficient, threshold, selected configuration, or checkpoint. Mathematical success alone never opens the expansion gate.

## 6. Pilot, tuning, selection, and protected evaluation

### 6.1 Two freezes

**Protocol freeze:** after the fit-only pilot, commit the asset/split hashes, method definitions, final hyperparameter grids, candidate schedule, resource ceilings, seeds/repeat count, primary budget, bootstrap settings, and decision margins. This occurs before using selection labels.

**Candidate freeze:** after the frozen fitting/selection procedure, hash every final classifier and predicted test score array before loading test labels for evaluation. Test outcomes never choose a threshold, seed, configuration, duration, or primary endpoint.

A protocol amendment after either boundary is visible in `DECISIONS.md`. After test access, the revised procedure is exploratory and needs fresh confirmation; the original test is not reset by rerunning a script.

### 6.2 Pilot calibration without exhausting the research budget

Run the minimal tests, a converged logistic baseline, and short G/E trajectories on `pilot_train` and `pilot_eval`. First establish realistic event/solver throughput and whether any nonzero correction changes held-out decisions. Do not start with a large grid on the protected split.

Use the pilot grids in Section 4. Examine training versus pilot-evaluation F1, false positives, actual prediction changes, zero-correction selection, prior scale, and how many coordinates have been visited. A chain with no useful movement is a diagnostic, not evidence of general superiority or general impossibility. A valid trajectory with worse held-out behavior is retained.

Choose a per-trial transition budget by beginning with enough work to visit the feature coordinates meaningfully and doubling while pilot curves still show resolvable improvement or a competitor is demonstrably unfinished. Budget expansion must have a written reason. If costs exceed available local resources before the method operates meaningfully, classify the result as resource-limited. Do not subsample or reduce only the event methods' feature space to force a favorable runtime.

Calibrate repeated fitting from five exploratory paired pipeline runs. Let `s_seed` be the standard deviation of paired pilot F1 effects, and `h_sample` a paired cluster-bootstrap 95% half-width on pilot evaluation. Choose a target optimizer Monte Carlo half-width `h = h_sample/4` and plan approximately

\[
 R\ge \max\{5,\lceil(1.96s_{seed}/h)^2\rceil\}.
\]

Use the largest required value across the two comparator effects and check the finite-sample t-quantile when finalizing. If the pilot bootstrap is degenerate, use hand-constructed paired-error sensitivity simulations and report that calibration instead; do not divide by zero. If the required repetition is locally impractical, report unresolved optimizer variability, rather than inflating the number of independent test samples. Freeze R before final selection/test. No test-driven extra seeds.

### 6.3 Practical tuning schedule and matched G/E opportunity

Let `K` be the number of frozen EIAL `(sigma,T)` configurations. Use **K independent direction-stream restarts of the same greedy algorithm**, not K different greedy methods. Each greedy restart begins at the same default initializer. Pair its direction stream with the EIAL trial at the corresponding index. This gives the inexpensive comparator a credible multistart opportunity and matches G/E trial and checkpoint exposure.

Use four nonzero checkpoints per trial, at approximately `1/8, 1/4, 1/2, 1` of the calibrated maximum trial work, plus the zero-correction candidate. A checkpoint contains a fit-only incumbent. G may finish earlier at its specified stationarity condition; do not charge fictitious extra work or withhold the finished classifier until the deadline. E uses its full frozen budget unless a numerical/resource failure occurs.

L runs the frozen regularization/weight grid. Generic solver warm starts or array optimizations are allowed when validated and charged; do not deliberately make L start from scratch for every fit to inflate its cost. Document candidate order.

Give each family the same **maximum project-specific fitting/tuning wall-time ceiling**, calibrated to accommodate L's entire converged grid and a meaningful G/E schedule, including initializer and threshold-selection work. Equal ceilings do not imply equal actual expenditure; unused time is not a penalty. If the full predefined schedule does not fit, revise/calibrate before the protocol freeze, or report the affected run as resource-limited. Do not silently truncate L and compare it against a complete EIAL search.

Retain candidate-ready times and intermediate curves. For the fixed primary budget, only candidates whose coefficients and required selection work were actually available within that budget are eligible. Selection computation is not free work performed after the clock stops. A single atomic-step overshoot is logged with its duration and is not credited as a within-budget completion.

Report the number of configurations, restarts, checkpoints, distinct selected thresholds, and selection-label evaluations. The three methods receive the same data and deployment calibration. G/E exposure is explicitly matched; L's natural grid may have a different candidate count. This stage screens practical pipelines and does not pretend every source of selection complexity is identical.

### 6.4 Exact threshold function

For each candidate, sort selection scores, group identical scores, and sweep all achievable threshold prediction sets, including all-negative and all-positive predictions. Compute F1 from integer counts. Compare rational F1 values by cross multiplication to avoid approximate ties. Choose the highest F1, then fewer FP, then the higher threshold. Use the same implementation for L, G, and E.

For finite scores, distinct observed score values with `score >= threshold`, plus an all-negative sentinel, enumerate all required sets. Store an all-negative or all-positive decision as an explicit mode when an infinite threshold is used; do not write nonstandard JSON infinities or fake finite thresholds.

This selection threshold can undo a poor intercept but cannot rotate the learned decision boundary. Therefore a gain surviving this step is more informative than a gain over untuned logistic output. Never report only the unrecalibrated comparator to make E appear stronger.

### 6.5 Final selection and repeated pipelines

For each independent pipeline repetition, run its complete frozen schedule and select **one** candidate per method using selection F1, then fewer selection FP, then earlier fully charged candidate-ready time, then stable candidate ID. This policy includes all hyperparameter/restart/checkpoint selection. Report its entire cost.

Do not select the best repetition by test results. Each repetition yields its own selected classifier; report the mean of the R test F1 values and their seed variability. L may be deterministic and yield identical predictions each time; reusing that result does not create extra evidence. This is the performance of a repeated fitting/selection pipeline, not an inference-time ensemble.

For a single demonstrator classifier, use the first predeclared repetition, not the most favorable one. Save it in addition to all R evaluation artifacts. Give its individual confusion counts so that an average does not masquerade as one deployable model.

### 6.6 Outcomes and analysis unit

The primary quality outcome is positive-class test F1, reported in `[0,1]`, with paired differences also in **percentage points**. For comparator `B` in `{L,G}`:

\[
 \Delta_B=\frac1R\sum_{r=1}^R
 \left[F_1(y,\hat y_{E,r})-F_1(y,\hat y_{B,r})\right].
\]

Report TP, FP, FN, TN, precision, recall, FPR, disagreement counts, added false alarms, recovered spam messages, training/selection/test gaps, and frequency of selecting the zero correction. Include actual counts and per-1,000-message rates; F1 alone is not an operational utility function.

The independent observational unit is the held-out **message group** under the observable grouping rule, not an optimizer seed, a checkpoint, or a repeated timing. All claims are conditional on this benchmark population and training split. Unknown group dependence and possible encoder pretraining exposure to public data remain limitations.

### 6.7 Paired uncertainty and small-sample sensitivity

Bootstrap held-out groups with replacement, retaining all rows in each selected group and using exactly the same resample for all methods. Recompute each repetition's F1 from the resampled confusion counts, then average effects. Do not average per-message “F1 contributions”; F1 is nondecomposable.

Report a cluster-only interval conditional on the fitted artifacts. Also compute a crossed group/seed bootstrap by independently resampling paired repetition indices and group indices within each bootstrap draw. It reflects observed optimizer variability, not retraining on new training populations. It does not remove bias from a particular training split or estimate variability over all possible tuning experiments. Use the more conservative lower bound for decisions when the two differ; report both.

Freeze the bootstrap replication count by checking endpoint stability on pilot data; start at 20,000 draws and increase only for numerical endpoint stability, not significance. Compute group-level confusion-count arrays and batch resampling to avoid a large `bootstrap × example × seed` allocation. A/A identical predictions must yield exactly zero paired quality effects.

To allow either a quality or runtime route without choosing a favorable story after the test, predeclare four directional comparisons: two quality effects and two fully charged runtime savings. Use family alpha 0.05 with Bonferroni one-sided alpha `0.0125` per comparison. The quality lower bound is the 0.0125 bootstrap quantile. Timing bounds use the interleaved block bootstrap described in Section 7. These are approximate bootstrap uncertainty statements, not exact finite-sample guarantees.

Show ordinary descriptive 95% intervals as well. A wide interval spanning both loss and useful gain is **inconclusive**. Nonsignificance is not equivalence; additional optimizer seeds cannot create more independent SMS examples.

Before unblinding, use the pilot data to simulate realistic paired prediction improvements/losses at the planned test size, preserving group and label structure. Report the rough detectable F1 effect and how many changed spam/ham decisions it corresponds to. If the available positives/groups cannot resolve the relevant effect, say so before interpreting the outcome. Do not access test labels to choose a different test fraction.

## 7. Cost accounting and M3 Max execution

### 7.1 Cost boundaries

Record both physical research expenditure and attributed per-method pipeline costs. Reusing an initializer or embedding cache once in the harness does not make it free for one method and expensive for another. Conversely, do not sum attributed shared costs and present the sum as physical elapsed time.

The primary runtime outcome is **project-specific pipeline time**, starting with raw embeddings available locally and ending with the selected, serialized classifier. Include feature-transform fitting/application, the default initializer where needed, all candidate fits/searches, all threshold/candidate selection, checkpoint work, and failed attempts in the frozen schedule. A winning candidate discovered early still incurs the later search work used to decide that it wins.

Also report:

| View | Included work | Interpretation |
|---|---|---|
| Cold task execution | Tokenization, encoding, cache I/O, plus project-specific pipeline | Same encoder cost for all methods; no invented encoder/backprop memory saving. |
| Frozen-recipe retraining | One already-chosen fitting recipe, required initializer, fixed training duration, threshold calibration, serialization | Separates optimizer cost from hyperparameter search. State that recipe discovery was paid previously. |
| Research/discovery total | Data-specific pilot runs, range extensions, numerical debugging runs, failed trials, and final repetitions | Total project expenditure, reported separately and broken down by family/common work. |
| Setup/download | Environment installation, asset downloads, model/cache bytes | Separately measured; network speed is not optimizer speed. |

The formal runtime route in Section 8 uses the primary pipeline-time view, named before test access. Other views are mandatory context, not alternative unregistered significance tests. A pipeline-time saving caused by a cheaper tuning schedule is reported as such; it is not called a faster Gibbs transition or a faster optimizer.

For frozen-recipe costs, actually rerun the selected recipes using fixed steps/seeds and calibration policy; do not estimate them by subtracting inconvenient parts of a measured total. Check agreement with the corresponding classifier. Always show the default `C=1` threshold-tuned candidate inside the L family as a cost floor. If it already delivers the required quality, adding a dependent event search cannot make that same workflow faster.

Save sorting, count sweep, Gaussian mass, truncated sampling, score-recomputation, thresholding, and initialization time separately. Report completed transitions, examples processed, unique events, and numerical fallback counts. “One exact line query” is not comparable to “one scored example.”

### 7.2 Timing protocol

Use monotonic wall time, one experimental process at a time, fixed/reportable BLAS thread counts, AC power, recorded power mode, and no competing benchmark jobs. Warm kernels once for steady-state comparisons; keep cold startup separate. Synchronize MPS before and after any accelerator timing. The float64 head methods run on CPU.

Use randomized/interleaved method order in repeated fresh-process timing blocks. First measure identical-workload A/A timing variation and timer resolution. Calibrate the block count on the pilot, then freeze it before final timing. Use paired block resampling for runtime savings `time_B - time_E`, preserving each block's method pairing and configuration. Do not pick the fastest run; report median, spread, raw blocks, and the adjusted lower bound.

Define a measurement-resolution margin `epsilon_time` from the pilot A/A absolute-difference distribution and timer resolution. A runtime saving must exceed this margin. It is a measurement-noise threshold, not a business-value threshold. Repeated timing blocks are not independent quality observations.

### 7.3 Hardware discovery and software

Before installing large assets, record:

```bash
uname -m
sw_vers
sysctl -n hw.memsize
df -h .
vm_stat
```

Confirm a native ARM64 Python environment. Resolve compatible stable versions of NumPy, SciPy, scikit-learn, PyTorch, sentence-transformers, mpmath, pytest, threadpoolctl, psutil, and a configuration reader. Lock actual versions and package hashes after the smoke tests. Do not invent version pins or use a mutable model `main` branch as provenance. Record NumPy/BLAS configuration and Python/platform versions.

Use CPU NumPy/SciPy for float64 event work. PyTorch's current MPS documentation explicitly excludes float64 tensors; use MPS only for optional float32 encoding [R8]. The inspected MPS page is development documentation, not proof that a particular stable wheel works on the user's macOS. Check the installed backend and actual model operations locally. No remote GPU, paid API, or CUDA-only component is required.

For a small, fixed text fixture, compare CPU and MPS embeddings under the same model/configuration. Require finite, normalized, correctly shaped outputs and investigate discrepancies above `1e-4` absolute coordinate error. This is a predeclared smoke tolerance, not a theorem about every example. If MPS is unavailable or inconsistent, generate the **entire shared cache on CPU**. Do not mix backend caches between competing methods.

### 7.4 Memory and calibration

For the source-card scale `n=5574, d=384`, calculated—not measured—storage is approximately:

- Float32 embeddings: `n*d*4 = 8,561,664` bytes, about 8.6 decimal MB.
- One float64 dense feature matrix: `n*d*8 = 17,123,328` bytes, about 17.1 MB.
- Float64 affine parameters: `(d+1)*8 = 3,080` bytes per vector.

These figures exclude model weights, activations, tokenizer, library state, arrays copied by solvers, bootstrap blocks, and event scratch space. They show why head-state memory is not the claimed bottleneck. The official model card's roughly 22.7 million parameters suggest about 91 MB for float32 parameter values alone [R3]; this is not peak encoder memory.

Measure peak resident memory, memory pressure, available disk, and encoder batch throughput on the actual laptop. Start encoding with a modest batch (for example 16), increase only while measured memory remains comfortably below the available budget, and halve on allocation failure. Batch size is a hardware knob, not a different model or a different dataset. Resume using verified cache chunks. Unload the encoder before fitting heads for every method.

Run optimizer trials sequentially. Do not allocate all event records for every trajectory at once. Batch bootstrap operations and retain only required candidate vectors, predictions, counters, and RNG states. If observed head resources contradict the small-matrix estimate, investigate array duplication or an erroneous quadratic implementation before reducing scientific scope.

The relevant resource forecast is measured encoding time plus measured full-grid/trial cost times planned pipeline repetitions and timing blocks. Record that forecast and available resource headroom in the pilot report. No runtime or M3 compatibility estimate in this document is a measured benchmark.

## 8. Decision gates: what earns continuation?

### 8.1 Predeclare the value inputs

Save these fields before protected evaluation:

```yaml
minimum_worthwhile_f1_pp: null       # supplied operational/research value, not fitted to results
maximum_acceptable_f1_loss_pp: 0.0  # conservative default for a runtime claim
maximum_extra_false_positives: null
maximum_pipeline_seconds: null
minimum_worthwhile_seconds_saved: null
primary_cost_view: project_specific_pipeline
family_alpha: 0.05
```

`null` means genuinely unspecified, not zero and not permission to choose a convenient number after the test. Complete the experiment even when value inputs remain unspecified; the decision label then distinguishes a credible local research signal from demonstrated practical usefulness.

When a real application supplies error costs, retraining frequency, or delay costs, freeze them before evaluating its value. Show additional FP and recovered spam explicitly. A simple value accounting can combine avoided false-negative cost, added false-positive cost, adaptation cost, and integration/maintenance burden. Do not convert F1 to money without those inputs.

### 8.2 Validity gate

Any leakage, broken baseline convergence, unmatched feature space, omitted initializer, invalid sampler, inconsistent threshold rule, or material numerical mismatch makes the affected comparison **invalid**. Repair it before a protected comparison. If found after unblinding, retain the invalid result and mark subsequent corrected analyses appropriately; do not quietly overwrite it.

A functioning implementation, correct integral, high training F1, or successful model download is never a usefulness pass.

### 8.3 Quality route

A **credible local quality signal** requires adjusted lower bounds for E minus L and E minus G above zero, measured at the frozen primary budget, with all costs and error tradeoffs reported. If a minimum worthwhile F1 gain was supplied, both bounds must also exceed that margin after converting percentage points to raw F1 units. Any supplied false-positive or cost limits must pass.

Inspect a prespecified leave-one-test-group-out influence table after unblinding. If removing one group reverses the mean gain, label the result fragile; do not delete that group. Report seed variability and zero-correction frequency. An effect that is mostly one ambiguous template is weaker evidence than a consistent change across unrelated messages.

A positive F1 effect accompanied by substantial additional false alarms is a tradeoff, not automatically better spam filtering. With no supplied error-cost preference, state that operational acceptability is unresolved.

### 8.4 Runtime route

A credible runtime signal requires adjusted lower bounds on E's pipeline-time savings over **both** named baseline pipelines above `epsilon_time`, and quality lower bounds at least `-delta_NI` against both, where `delta_NI` is the predeclared maximum acceptable F1 loss. With the conservative default `delta_NI=0`, identical paired predictions can demonstrate equal observed quality, but a nonsignificant quality difference does not qualify as noninferiority.

Apply any supplied minimum seconds-saved or maximum-cost requirements. Check the default-logistic cost floor and frozen-recipe view. If those reveal that an already-adequate cheap classifier is faster, report only the narrower first-use tuning-pipeline result; it does not establish a generally useful speed advantage.

A faster head sweep that disappears under total pipeline accounting is not a runtime pass. A speedup only against G, while an equally good L is cheaper, does not justify continuing E as the preferred practical method.

### 8.5 Outcome table and authorized next action

| Observed result | Interpretation | Authorized next action |
|---|---|---|
| E beats L and G with credible quality/cost evidence and passes predeclared practical limits | Local usefulness evidence | One focused attribution/replication step; broader benchmarking still requires that check. |
| Credible, nonfragile E advantage, but value inputs are unspecified | Promising local research signal; practical usefulness unresolved | At most one targeted independent replication/value-assessment step, not the old full roadmap. Record its information value and added cost. |
| E beats L but matches or loses to G | Exact search may help; probability integration has not earned its complexity | Retain G/L results; stop expansion of EIAL-G on this evidence. |
| E beats G but not tuned L | Complex search has not beaten the cheap practical alternative | Prefer L for this task; no broad EIAL expansion. |
| E improves training or selection but not protected results | Generalization/selection failure | Stop local superiority claims and document it. Do not add datasets until one yields a win. |
| Quality effectively matches and E is slower | No local advantage | Complete the negative report and stop this PoC. |
| Bounds contain both loss and a plausible worthwhile gain | Inconclusive | Quantify the sample/variability limitation. More seeds are allowed only in a new predeclared study when optimizer noise, not sample size, is the problem. |
| Correctness or resource prerequisite fails | Invalid or resource-limited, not a scientific defeat | Repair the specific blocker within the same scope or terminate with evidence. |

A negative SMS result does not prove that all constrained or higher-dimensional integral methods fail. It does remove this experiment as justification for automatic expansion. A different application needs a concrete, independently motivated hypothesis; a post-hoc claim that “it must work somewhere” is not enough.

### 8.6 First follow-on after a positive screen

Before attributing a gain specifically to analytic integration, add a **Gaussian-penalized exact greedy line-MAP control** and a **sampled approximation to the same conditional target**. These test ordinary regularization and analytic-versus-sampled integration. They are outside the initial three-method PoC and become a separate recorded study only after its gate.

Develop those controls on development data. The already-inspected SMS test can support transparent post-hoc diagnostics, but cannot independently confirm a revised algorithm or a new winner. Use fresh protected data for confirmation. Before making a claim about practical spam classification beyond this frozen representation, also compare a strong task-specific text baseline, such as tuned TF–IDF plus a linear classifier. Do not equate a MiniLM-head win with best-in-class spam filtering.

Only after a reproducible, interpretable advantage should a second encoder, constraints, or a broader dataset panel be considered. There is no unconditional instruction to implement the earlier roadmap.

## 9. Four ordered work packages

Every work package writes its status, evidence, unresolved issues, and next action to `DECISIONS.md`. A failure is retained, not deleted.

### WP1 — Freeze the workload and establish local prerequisites

**Inputs:** this document, the target laptop, public model/dataset access.  
**Actions:** hardware/software discovery; lawful asset retrieval and hashing; strict parsing; grouping; primary and fit-only pilot split construction; backend smoke test; shared embedding cache; fit-only feature transformations.  
**Modules/commands:** `data.py`, `features.py`, `doctor.py`; proposed `python -m eial_sms doctor` and `python -m eial_sms prepare`.  
**Outputs:** hardware/lock/provenance manifests, group and split maps, cache hashes, feature schema, actual class counts, truncation and memory report.  
**Acceptance:** both labels in required roles; no group overlap; finite reproducible features; no protected labels in fitting paths.  
**Failure branches:** CPU fallback for MPS; corrected parser for explained records; stop for unresolved asset rights, bad splits, or unavailable faithful model.  
**Resources:** one small encoder/data cache, measured batch calibration; no other datasets/models.

### WP2 — Implement only the three methods and indispensable tests

**Dependency:** WP1 numerical stack; synthetic fixtures can run before encoding finishes.  
**Actions:** one shared metric/threshold function; converged L; default initializer; event/count engine; G; E; checkpoint and replay; tests in Section 5.  
**Modules/commands:** `linear.py`, `events.py`, `greedy.py`, `gibbs.py`, `select.py`; proposed `python -m eial_sms check`.  
**Outputs:** passing count/numerical/calibration/replay tests, method-specific failure fixtures, a default logistic reference artifact, cost instrumentation.  
**Acceptance:** event counts agree with an independent oracle; Gaussian law checks pass; L converges; identical inputs/predictions give identical evaluation; no selection/test labels enter transitions.  
**Failure branches:** repair the exact failing component; do not add new optimizers or broad geometry infrastructure.  
**Resources:** small CPU fixtures and cached matrices; no broad training campaign.

### WP3 — Run the fit-only pilot and freeze the experiment

**Dependency:** valid minimal implementation.  
**Actions:** short pilot curves; range checks; measured timing/memory; repeat/sensitivity calibration; candidate and budget policy; practical-value fields; final configuration freeze.  
**Modules/commands:** `run.py`, `analyze.py`; proposed `python -m eial_sms pilot` and `python -m eial_sms freeze`.  
**Outputs:** pilot report, resource forecast, complete frozen YAML, deterministic seed registry, `PROTOCOL.md` and its hash.  
**Acceptance:** credible complete baselines, meaningful search operation, budget accounting, uncertainty method checked, no use of selection/test outcomes for protocol choices.  
**Failure branches:** bounded correction of a diagnosed problem; record resource limitation or low expected information value instead of starting a larger panel.  
**Resources:** calibrated from observed throughput; no assumed M3 timings.

### WP4 — Execute the frozen comparison and issue a decision

**Dependency:** immutable protocol and sufficient measured resources.  
**Actions:** final fit/selection for every predeclared repetition; freeze candidates; protected evaluation; interleaved timing; paired uncertainty; influence table; reproduce one complete repetition from saved state; generate report directly from artifacts.  
**Modules/commands:** proposed `python -m eial_sms run`, `python -m eial_sms evaluate`, and `python -m eial_sms report`.  
**Outputs:** all per-example predictions, all selected models, full configuration/restart/checkpoint histories, costs, uncertainty tables, failure registry, final gate decision.  
**Acceptance:** every table entry traces to immutable data/config/model hashes; unfavorable outcomes retained; decision follows Section 8; no broader claim than the evidence supports.  
**Failure branches:** invalid, resource-limited, inconclusive, negative, or positive—each is a valid documented termination state.  
**Resources:** frozen forecast and checkpoints; no post-test search for favorable seeds or workloads.

## 10. Repository, interfaces, and reporting contract

### 10.1 Minimal repository

```text
eial-sms/
  README.md
  PROTOCOL.md
  DECISIONS.md
  pyproject.toml
  requirements.lock
  configs/pilot.yaml
  configs/frozen.yaml
  src/eial_sms/
    __main__.py
    doctor.py
    data.py
    features.py
    metrics.py
    events.py
    linear.py
    greedy.py
    gibbs.py
    select.py
    run.py
    analyze.py
  tests/
    test_events.py
    test_normal_mass.py
    test_thresholds.py
    test_access_boundaries.py
    test_replay.py
    fixtures/
  artifacts/
    provenance/
    splits/
    features/
    pilot/
    frozen/
    runs/
    reports/
```

A single well-structured file can initially contain several of these functions. The layout identifies responsibilities; it is not permission to spend the PoC building a framework. The coding agent may make routine packaging and non-destructive implementation decisions without asking for approval. Changes to the method, data population, grouping, metric, budget fairness, or test protocol require a recorded amendment.

### 10.2 Shared interfaces to implement

```text
FitData:
    ids, group_ids, y, X_float64, H_float64, feature_manifest_hash

BaseHead:
    coefficients, intercept, fit_threshold, score_scale,
    initialization_seconds, dependency_hashes

Candidate:
    method, pipeline_seed, trial_id, config_id, checkpoint_id,
    coefficients, intercept, fit_confusion_counts,
    ready_time_seconds, source_state_hash

fit_linear_candidates(FitData, grid, budget) -> Candidate sequence
fit_event_candidates(FitData, BaseHead, method, configs, budget, rngs)
    -> Candidate sequence and transition diagnostics
choose_threshold(scores, labels) -> threshold_or_mode, confusion_counts
select_candidate(candidates, SelectionData) -> immutable SelectedClassifier
evaluate_frozen(classifier_manifest, ProtectedTestData) -> prediction records
```

`fit_event_candidates` must not accept a SelectionData/TestData object or a callback that returns their rewards. Test evaluation must refuse a missing or mismatched candidate/protocol hash. Protect logical data roles even if the small raw arrays could physically fit in one process.

A `SelectedClassifier` stores the final threshold separately from the base intercept, with an explicit constant-decision mode where needed. Export the mathematically combined affine coefficients; verify export/reload predictions. Do not inadvertently apply the initializer threshold twice or confuse standardized-feature coefficients with raw-embedding coefficients.

### 10.3 Configuration and run records

The frozen configuration must contain immutable dataset/model hashes, split/group definitions, label convention, full-feature setting, numerical policies, solver settings, grids, K restarts/configurations, actual R, checkpoint work, primary wall budget, per-trial work budget, RNG seeds, thread counts, bootstrap count, timing-block count, and Section 8's value fields.

Unresolved execution essentials such as model commit, final grids, repeat count, or time budget **block protocol freeze**. Unspecified business-value fields do not block the experiment; they constrain the continuation wording.

Use independent deterministic RNG streams for data allocation, pipeline/trial directions, Gibbs draws, bootstrap, and timing order. Store generator type/state and seed derivation. Save current chain, incumbent, visited-coordinate stopping state for G, next checkpoint, and cumulative spent time atomically. Resume must not reset a budget or reuse a different cache/feature transform.

For every run record:

```text
schema_version, run_id, method, phase, pipeline_seed, trial_id,
config_hash, protocol_hash, code_commit, environment_hash,
data_hash, split_hash, model_revision, feature_hash,
status, failure_reason,
n_fit, n_select, n_test, n_groups, n_positive, feature_dimension,
initializer_seconds, fit_seconds, select_seconds, checkpoint_seconds,
pipeline_seconds, peak_rss_bytes, cache_bytes,
transitions, unique_events, direct_score_checks, precision_fallbacks,
selected_candidate_id, selected_threshold_or_mode,
TP, FP, FN, TN, F1, precision, recall, FPR
```

Keep role-specific counts and metrics separate; the schema lists names, not permission to populate test metrics before candidate freeze. Valid statuses include `completed`, `interrupted`, `invalid`, `numerical_resolution_failure`, `resource_limited`, and `failed_solver`. Record failures and their elapsed cost.

Save per-example scores and predictions for every final repetition with row/group IDs. Do not print message contents in public logs by default. Use JSON/CSV plus non-pickled numeric arrays, file hashes, and atomic writes. Reload only self-produced, hash-verified artifacts.

### 10.4 Required final report

Start with the decision and the evidence that triggered it. Include a three-row principal table:

| Method | Test F1 and uncertainty | FP / FN | Project pipeline seconds | Fixed-recipe seconds | Peak memory | Selected zero correction? |
|---|---|---|---|---|---|---|
| L: logistic + threshold | Measured only | Measured only | Measured only | Measured only | Measured only | Not applicable |
| G: exact greedy | Measured only | Measured only | Measured only | Measured only | Measured only | Measured only |
| E: EIAL-G | Measured only | Measured only | Measured only | Measured only | Measured only | Measured only |

Show mean pipeline quality and seed spread separately; the demonstrator repetition gets its own confusion table. Add the two paired F1 effects with adjusted bounds, two timing savings with adjusted bounds, candidate exposure, complete cost-quality curves, pilot-versus-test distinction, group influence, truncation/grouping limitations, and all failures.

Every graph uses the same cached representations and clearly states whether its horizontal axis is transitions, head-only time, complete pipeline time, or cold execution time. Do not label one as another. Exploratory curves remain labeled exploratory. Do not put hypothetical favorable numbers in a results table.

End with one of: invalid, resource-limited, inconclusive, no local advantage, promising local signal with unresolved value, or local usefulness evidence under explicit assumptions. State exactly which next action Section 8 authorizes, including “none.” A clean negative result completes the PoC.

## 11. Audit of this revision and residual risks

This is a design audit, not an independently executed review or a benchmark validation.

| Potential defect | Revision made | Remaining limit |
|---|---|---|
| Large infrastructure spend before usefulness | Four work packages; three methods; one encoder/dataset; explicit expansion gate | Correct minimal implementations still require real engineering. |
| Logistic handicapped by PCA or an untuned threshold | Full shared embedding space; converged regularization/weight grid; exact shared calibration | A frozen MiniLM affine head is not the strongest conceivable spam pipeline. |
| Event method receives a trained head for free | Fixed default initializer, separate dependency/cost ledger, default-head cost floor | A warm-started EIAL win is adaptation, not fully derivative-free training from scratch. |
| E wins from more restarts or validation exposure than G | Matched G/E trial count, directions, checkpoint count, and selection rules | Prior weighting and stochastic line selection remain coupled until follow-on controls. |
| Continued training silently changes the Gibbs law | Separate chain/incumbent, fixed target, state-independent directions | Numerical approximation and slow mixing still require diagnosis. |
| Greedy loses legitimate boundary solutions | Singleton-event states included and verified | Some exact real regions may not be representable in float64. |
| Certification consumes scarce data without testing the main question | Remove constraints/certification from this stage | No deployment FPR guarantee is established. |
| One positive seed or one spam template drives the conclusion | Full repeated pipelines, paired group uncertainty, influence report | A small public corpus can still be underpowered or unrepresentative. |
| “Quality or runtime” becomes post-hoc outcome shopping | Four predeclared directional bounds; one primary cost view | Approximate bootstrap coverage is not exact finite-sample error control. |
| “Worthwhile” is defined after seeing the result | Value fields frozen; research signal separated from business usefulness | Missing operational preferences cannot be inferred from F1 alone. |
| Negative first dataset becomes an excuse to search until success | No automatic panel expansion; explicit local negative/inconclusive outcomes | A separately motivated application can still deserve a new study. |

**Most important unresolved question:** whether the probability-weighted search improves an actual protected quality–cost tradeoff. This document deliberately leaves that answer open; implementing it is the way to obtain the evidence.

## 12. References and inspection scope

References below support specific background and implementation facts. Algorithm choices, budgets, statistical procedures, and decision rules in this document are proposed design decisions unless explicitly attributed otherwise. All essential formulas and execution rules are given above; reading these links is not a substitute for a missing specification.

**[R1]** Franz Josef Och. *Minimum Error Rate Training in Statistical Machine Translation*. ACL, 2003, pp. 160–167. DOI: `10.3115/1075096.1075117`. [ACL record](https://aclanthology.org/P03-1021/). The earlier supplied plan identified and inspected this as exact discontinuous line-search prior art. This revision checked the bibliographic record; it did not repeat the earlier full-paper analysis. Used to retain the novelty limitation, not to assert that the SMS algorithm is identical to MERT.

**[R2]** Tiago Almeida and José Hidalgo / UCI. *SMS Spam Collection*. Dataset citation 2011; repository donation date 2012. DOI: `10.24432/C5CC84`. [Official dataset card and download](https://archive.ics.uci.edu/dataset/228/sms+spam+collection). Card, expected count, format, source description, and CC BY 4.0 notice inspected on 13 October 2026. Archive contents and actual parsed counts were not verified in this revision.

**[R3]** Sentence Transformers. *all-MiniLM-L6-v2*. [Official model repository/card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). Inspected on 13 October 2026 for 384-dimensional output, pooling/normalization, 256-word-piece default truncation, parameter-count display, and Apache-2.0 label. No immutable revision was downloaded or executed here; the coding agent must pin one.

**[R4]** scikit-learn developers. *Tuning the decision threshold for class prediction*. [Official user guide](https://scikit-learn.org/stable/modules/classification_threshold.html). Inspected on 13 October 2026 for the distinction between estimation and decisions and the warning against using fitting data as independent threshold-validation data. The project's exact sweep and explicit split policy are specified above rather than delegated to a default threshold grid.

**[R5]** scikit-learn developers. *LogisticRegression*. [Official API](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html). Inspected on 13 October 2026 for L-BFGS/L2, float64 arrays, class weights, regularization, convergence arguments, warm starts, and the documented penalty-parameter deprecation. The displayed documentation version does not establish the installed target-laptop version; inspect and lock the latter.

**[R6]** SciPy developers. *scipy.special.log_ndtr*. [Official API](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.log_ndtr.html). Inspected on 13 October 2026. Used for log-normal-CDF evaluation. Stable interval differencing and high-precision fallback are explicit additional requirements, not guarantees attributed to this one function.

**[R7]** SciPy developers. *scipy.stats.truncnorm*. [Official API](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.truncnorm.html). Inspected on 13 October 2026 for standardized truncation bounds and sampling interface. Local tail/rounding validation is still required.

**[R8]** PyTorch contributors. *MPS backend*. [Official development documentation](https://docs.pytorch.org/docs/main/notes/mps.html). Inspected on 13 October 2026 for backend discovery and the documented absence of float64 support. The stable URL redirected without useful content in one retrieval; development documentation was read instead. No claim is made that a specific stable wheel or complete MiniLM workload has been validated on an M3 Max.

**Source history, not an external dependency.** This revision responds to a reviewer who praised baseline rigor, numerical handling, separated data, and full costs, but requested a first comparison of MiniLM/SMS using logistic-plus-threshold, exact greedy, and EIAL-G before expansion. The supplied earlier document was titled *Research and proof-of-concept plan: event-integrated neural decision adaptation*, dated 13 October 2026. Its relevant algorithm, admissions, and scope changes are restated here. The coding agent does not need that document.

**Coverage and limitations.** This was a reviewer-driven redesign with targeted official-source checks, not a new systematic literature search, patent search, or proof of novelty. General web searches were unhelpful in several cases; direct official pages supplied the relevant information. No recent derivative-free paper was added merely to enlarge the bibliography. The key new evidence must come from the controlled experiment, not from further citations or more synthetic correctness counts.

---

## Completion contract

The coding agent is done when it has either documented a blocking invalid/resource condition or delivered the frozen three-method comparison, full costs, uncertainty, reproducible artifacts, and the corresponding decision. It is **not** required to make EIAL-G win. It is **not** authorized by this plan to expand until a favorable result appears.
