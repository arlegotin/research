# POC_EXECUTION_BRIEF

## 1. Project decision and mandate

**Project:** finite-catalog compression and certification of signed cosine recommendation profiles in 1,128-dimensional frozen item embeddings.

**Intended executor:** an autonomous coding agent configured as **gpt-6.1-sol with xhigh reasoning**. Implement and run the authorized stages on one MacBook with an M3 Max. This document is your complete research instruction. Execute the gates, repair ordinary implementation defects within the cumulative allowance, inspect the results, and finish with an evidence-backed report. A justified early stop is a completed outcome. Do not substitute a plan for the authorized execution.

**Research decision:** determine whether a catalog-aware geometric compression method adds enough ranking fidelity over simpler compression methods to justify its construction cost, while retaining at most one quarter of a user's positive and negative exemplars. Establish separately whether the underlying signed scorer has useful discrimination on later explicit ratings.

The supplied circle theorem is a valid-looking, constructive result under narrow assumptions. It does **not** establish the same margin formula, prototype-count bound, confidence interpretation, or recommendation improvement in many dimensions. The practical high-dimensional score is already a nearest-prototype classifier. Representative-set and multiple-interest recommendation methods are established prior art. Accordingly, this project tests a specific empirical opportunity; it does not announce a new general theorem or a new family of recommenders.

### 1.1 Smallest defensible contribution

Produce a reproducible characterization of **when an existing finite-direction support-preservation mechanism can compress signed, high-dimensional preference histories**, its effect on rankings and later-rating discrimination, and its construction/storage/serving tradeoff against strong feasible controls.

The proposed method is a **budgeted greedy cover of catalog directions**, applied separately to positive and negative histories. Its certificates are elementary consequences of support-function approximation. The candidate contribution is the measured regime and its limitations, including whether catalog-aware coverage offers an advantage over both a practical clustering baseline and a same-information regret baseline. Global priority for that empirical combination has not been established.

A positive result is valuable only if its scope is clear. A valid finding that Ward medoids or worst-regret selection suffice, that the signed reference scorer is weak, or that preprocessing/caching defeats the operational case is also valuable.

### 1.2 Primary falsifiable question

> At fourfold or greater reduction in stored exemplar IDs, does catalog-aware greedy coverage improve user-mean top-20 overlap with the full signed scorer by at least **0.02 absolute** over the strongest compression comparator selected on development users, while reaching at least **0.95 mean overlap**, on the locked high-dimensional MovieLens study?

Use a separate **0.01 absolute macro-AUC noninferiority guardrail** for compression versus the full signed scorer on later explicitly rated items. Also compare the full scorer with a tuned simple preference baseline; imitation alone does not validate a recommender.

These are fixed R&D acceptance requirements, not estimates of achievable results:

- Fourfold reduction is a material reduction of the **hot preference-ID profile**, not a promise of fourfold total system memory or latency reduction.
- Top-20 overlap of 0.95 allows one changed item per list on average.
- An advantage of 0.02 means 0.4 additional reference items retained per 20-item list on average. A smaller gain does not justify claiming a material benefit from this extra construction machinery.
- A 0.01 AUC allowance limits degradation in observed positive-versus-negative discrimination. It is not an online business-impact equivalence.

The expected mechanism is that covering the candidate directions actually used for ranking preserves important maxima better than representing average geometric clusters. It may fail when histories are dispersed, rare directions require distinct exemplars, top-20 score gaps are tiny, opposite-class errors cancel in ways the classwise objective ignores, or the reference score itself is a poor preference model. A same-catalog worst-regret comparator may remove the apparent advantage.

### 1.3 Scope, exclusions, and the revision to the original problem

Preserve the practical purpose: use multiple explicit likes and dislikes in many-dimensional embeddings without training a new neural network. Make these changes explicit in the report:

| Original mathematical setting | Selected executable setting |
|---|---|
| Entire unit circle; all boundaries known | A finite, known catalog of high-dimensional item vectors |
| Freely synthesized unit prototypes | Subsets of observed liked/disliked item vectors |
| Exact prescribed class regions | Fidelity to a specified empirical signed scorer, plus separate rating evaluation |
| Globally optimal continuum score margin | Measured finite-catalog score errors and sufficient ranking certificates |
| Sharp 3r and 3/2 count results | Fixed empirical fourfold compression target with no inherited count guarantee |

Do not claim to solve arbitrary high-dimensional boundary synthesis, infer a full preference region from sparse observations, prove universal compactness, improve calibrated confidence, or improve an online recommender. Exclude neural fine-tuning, learned item embeddings, approximate-nearest-neighbor infrastructure, distributed compute, CUDA, external paid compute, new-user cold start, newly arriving catalog items, implicit-negative sampling, and publication-priority claims.

The serving scenario is **varying candidate subsets of a fixed known catalog with a fixed preference snapshot**. An exact cached score vector supports those subsets too. Therefore, the operational result is a **storage–construction–latency comparison**, not an automatic serving speedup.

All claims about performance on the M3 Max must come from that machine. Hosted-agent reasoning and research tools are outside the local experimental-compute restriction.

## 2. Evidence and mathematical foundation

### 2.1 Inputs inspected during preparation

The user supplied:

1. **Circular_Cosine_Preference_Theorem.zip**, containing:
   - Circular_Cosine_Preference_Theorem/Circular_Cosine_Preference_Theorem.md: explanatory summary.
   - Circular_Cosine_Preference_Theorem/Circular_Cosine_Preference_Theorem.ipynb: the paper, proofs, verification programs, recorded results, and figure.
   - Mac metadata files, which contain no additional mathematical document.
2. **Pasted text(20261029-091043).txt**: instructions to prepare this execution brief.

The mathematical reference below is **[A1]**, the notebook. Its author is not identified; do not invent one. The notebook dates itself 29 October 2026. The explanatory Markdown's claims about “confidence” must be read subject to the notebook's narrower cosine-score-gap definition.

All proof sections, code cells, recorded outputs, and the plotted example were inspected. The audit found no essential defect in the two main theorems under their stated assumptions. This is a substantive audit, not a claim of formal verification or specialist publication review.

Attachment SHA-256 identifiers:

~~~text
Circular_Cosine_Preference_Theorem.zip
0fe2846fc4a4407dffdc402d9354aa7bc53a66c803b4d5486ecba04a96abb721

Circular_Cosine_Preference_Theorem.ipynb
75e6c6a1160cf905fd461f22d6bcdaf23cd63d49dc272e304dc83889bd75502b

Circular_Cosine_Preference_Theorem.md
f163037e7a8bfc098831e9f78af94dd8c11ff6625a501d136e06d1303bea967b
~~~

Receive the ZIP if available and preserve it as source provenance. The definitions and necessary diagnostics are reproduced here, so its absence does not block the finite-catalog implementation. If unavailable, explicitly mark the literal notebook rerun as unavailable; do not claim to have executed it.

### 2.2 What the source establishes

Let n = 2r cyclically ordered boundaries partition the unit circle into alternating positive and negative open arcs, with lengths \(\ell_i>0\) summing to \(2\pi\). Let

\[
u(\theta)=(\cos\theta,\sin\theta),\qquad
S(\theta)=\max_{p\in P}\cos(\theta-p)-\max_{q\in N}\cos(\theta-q).
\]

Both prototype sets are finite and nonempty and all their vectors lie on that same circle. Their weights are one, the threshold is zero, and score scaling is fixed. An exact realization has the prescribed strict sign on every open arc. Put

\[
a=\tfrac12\min_i\ell_i,\qquad
M_\eta=\min_{d_{S^1}(\theta,D)\ge\eta}|S(\theta)|,\quad 0<\eta\le a.
\]

[A1, §1, Theorems 1–2; §§2–5] establishes:

- The largest possible margin is \(2\sin a\sin\eta\).
- A realization is optimal at one admissible buffer if and only if it is optimal at every such buffer.
- Boundary clearances \(t_i\) satisfy \(t_i>0\) and \(t_i+t_{i+1}\le\ell_i\). The common optimizers are exactly those with every \(t_i\ge a\).
- Starting at \(t_i=a\), one cyclic sweep of
  \[
  t_i\leftarrow\min(\ell_{i-1}-t_{i-1},\,\ell_i-t_{i+1})
  \]
  preserves feasibility and produces at least r tight arc constraints. Put sites at \(b_i+t_i\) and \(b_{i+1}-t_{i+1}\), merging them when equal.
- At most 3r prototypes suffice; this worst-case bound is sharp even for classification alone.
- If C is the minimum exact-realization count and \(C_*\) is the minimum count attaining optimal margins, then \(\sup C_*/C=3/2\). This is a supremum over partitions, not a universal requirement of 50% extra sites.

For an arbitrary exact realization with \(c=\min_i t_i\), [A1, §3, Eq. (7)] gives \(M_\eta=2\sin c\sin\eta\) for \(0<\eta\le c\). The small-width optimum at \(\eta=a\) is \(2\sin^2a\), asymptotic to \(\ell_{\min}^2/2\).

**Constructive bridge audit.** The sweep is linear only after exact ordered boundaries are supplied. Learning those boundaries from noisy ratings is an absent oracle, not part of the algorithm. The sweep is not an exact optimizer of the instance-specific minimum prototype count; the notebook explicitly distinguishes those problems. Its rational count verifier enumerates equality patterns and uses Fourier–Motzkin elimination only on tiny instances. Do not import that enumeration into the recommendation pipeline.

The source's perturbation guarantee controls the score under small prototype or implementation perturbations. All exact realizations have the same class boundaries; a larger score gap does not increase the actual on-circle distance to those fixed boundaries. The margin is neither a probability nor evidence of statistical generalization. The multiclass observation in §7 is outside this binary PoC.

### 2.3 Work actually performed for this brief

These are **planning checks in a hosted environment**, not the requested PoC and not M3 Max benchmarks:

| Check performed | Result and interpretation |
|---|---|
| Reran notebook cell 10 | 240 greedy and 240 arbitrary configurations; 1,200 optimal-margin and 1,200 upper-bound checks passed; maximum absolute discrepancy \(1.2455314557513475\times10^{-15}\) |
| Reran notebook cell 12 | Eight named count cases, 80 random rational partitions, and the k = 1,…,20 overhead family passed |
| Inspected notebook figure and source | Five-versus-six-prototype example agrees with its formulas; no new empirical recommendation claim |
| Evaluated the off-plane counterexample below | At \(a=\pi/12,\eta=a/2\), margin 0.13052619222005157 versus circular formula 0.06756532886252364 |
| Downloaded and inspected two official dataset archives | Dates, formats, dimensions, sizes, and hashes verified; see §4 |
| Streamed ratings for eligibility only | Established feasible cohort sizes; no recommendation scores, compression outcomes, or model-quality measurements were inspected |

Independent agents contributed mathematical, literature, and data/experimental reviews; their findings were reconciled in this brief. Rerunning supplied code corroborates its reported experiments but is not an independent implementation of those checks.

### 2.4 What survives in many dimensions

For normalized item vectors \(x,p,q\in S^{d-1}\),

\[
h_P(x)=\max_{p\in P}x^\top p,\qquad S(x)=h_P(x)-h_N(x)
=\tfrac12\left(\min_{q\in N}\|x-q\|_2^2-\min_{p\in P}\|x-p\|_2^2\right).
\]

Thus the sign is ordinary nearest-prototype classification, and each maximum is the support function of the corresponding convex hull. These are established identities, consistent with nearest-prototype and representative-set literature [R1–R4]. Evaluating the score requires no optimization oracle: use dot products and maxima.

Two limitations are decisive:

1. **The circular formula does not even survive arbitrary off-plane prototypes on a circular query slice.** For \(0<a<\pi/6\), take
   \[
   x(\theta)=(\cos\theta,\sin\theta,0),\quad P=\{(0,0,1)\},
   \]
   \[
   N=\{(-\sin a,\cos a,0),(-\sin a,-\cos a,0)\}.
   \]
   Then
   \[
   S(\theta)=\sin a\cos\theta-\cos a|\sin\theta|.
   \]
   Its positive arc is \((-a,a)\), and its eroded margin is \(M_\eta=\sin\eta\) for \(0<\eta\le a\). On \([0,\pi]\), the score is \(\sin(a-\theta)\); symmetry and the interval endpoints establish that margin. It exceeds \(2\sin a\sin\eta\). Off-plane unit vectors project to cosines of amplitude below one, violating the source's unit-circle-prototype assumption. This is a counterexample to transferring the theorem, not to the theorem itself.

2. **Exact global class-support compression cannot remove distinct unit prototypes.** At query \(x=p\), one has \(p^\top p=1>p^\top q\) for every distinct unit q. Every prototype is exposed. This obstruction holds even in two dimensions. It does not prohibit approximate finite-catalog compression or cancellation between the two signed errors.

More generally, finite same-scale prototype scores have active-pair boundaries in hyperplanes \(x^\top(p-q)=0\); arbitrary high-dimensional prescribed preference regions need not admit an exact finite representation. The finite-catalog reformulation avoids claiming such an inverse solution.

### 2.5 The certificates you must implement

For retained **subsets** \(P'\subseteq P,\ N'\subseteq N\), define at catalog item j:

\[
e_j^+=h_P(x_j)-h_{P'}(x_j)\ge0,\qquad
e_j^-=h_N(x_j)-h_{N'}(x_j)\ge0.
\]

Then, exactly,

\[
S'_j-S_j=e_j^- - e_j^+.
\]

If the two losses are bounded by \(E_j^+,E_j^-\), then

\[
S_j-E_j^+\le S'_j\le S_j+E_j^-.
\]

With uniform bounds \(\epsilon_+,\epsilon_-\),

\[
|S'_j-S_j|\le\max(\epsilon_+,\epsilon_-).
\]

Do not use an unnecessarily loose sum as the subset score-error bound. The sum appears in **pairwise order**: an original gap larger than \(\epsilon_++\epsilon_-\) is sufficient to preserve the ordering. In particular, strict top-20 membership is certified when

\[
S_{(20)}-S_{(21)}>\epsilon_++\epsilon_-.
\]

For original top-20 set T, a sharper query-specific sufficient condition is

\[
\min_{i\in T}(S_i-E_i^+)>
\max_{j\notin T}(S_j+E_j^-).
\]

A positive original sign is protected when \(S_j>E_j^+\); a negative sign when \(S_j<-E_j^-\). Add numerical guards as specified in §3.5.

These sufficient certificates can fail even when observed rankings are identical. Record **uncertified**, not **incorrect**, in that situation. For full fixed-catalog guarantees use actual verified maximum support losses, not the construction tolerance unless every row passed that tolerance. Certificates over the entire user-eligible catalog automatically apply to any subset of that catalog. They do not cover new item vectors, a changed history, a changed embedding transform, or synthesized centroids.

Do not turn classwise failure into a universal signed-compression impossibility. For example, \(P=N=\{e_1,e_2\}\) and \(P'=N'=\{e_1\}\) give identical zero signed scores while each class loses support at \(e_2\).

### 2.6 Central claim-to-evidence map

| Claim | Status/source | Remaining uncertainty and resolving check |
|---|---|---|
| Circular optimum and sparse synthesis | Source result; [A1, §§1–5]; proofs audited, supplied checks rerun | Reproduce tiny independent examples; no high-dimensional transfer |
| High-dimensional score equals nearest-prototype classification | Established identity; derivation above; [R1] | Verify Euclidean-distance and dot-product implementations agree |
| Arbitrary ambient prototypes invalidate the same circular envelope | Explicit derivation above; numeric endpoint check performed | Independent scalar check in Stage 1 |
| Finite support loss implies signed score/rank bounds | Elementary derivation in this brief | Exhaustive tiny subset checks and high-precision numeric comparisons |
| Greedy covering is new | **Rejected**; representative-set/set-cover prior art [R2–R4] | No novelty claim to rescue |
| Fourfold compression preserves useful recommendation rankings | Primary untested hypothesis | Locked user comparison, paired CIs, controls |
| The signed scorer predicts explicit preferences usefully | Untested, separate question | Later-rating AUC against simple tuned scorers |
| Compression saves serving time or system memory | Conditional, untested | Build-inclusive timings, profile-ID bytes, exact cache controls |
| Selected workload fits M3 Max | Planning estimate | Inspect actual hardware, calibrate, apply caps |


## 3. Computational mechanism and numerical contract

### 3.1 Inputs, outputs, and invariants

For each user, let P and N be the full **retained historical** positive and negative profiles after the preprocessing in §4. They contain \(H_+\) and \(H_-\) distinct within-class embedding vectors, with \(8\le H_c\le64\). “Full” always means these declared capped profiles, not every rating the user ever supplied.

Let \(X_u\in\mathbb R^{M_u\times1128}\) contain the eligible catalog, excluding **every** pre-cutoff item rated by that user, including neutral ratings and ratings beyond the retained history cap. Keep item IDs in ascending order. Use CPU float64 normalized vectors.

Fix budgets independently by class:

\[
B_c=\max(1,\lfloor H_c/4\rfloor).
\]

All compression methods receive the same P, N, X, and class budgets. They return exactly \(B_+\) positive and \(B_-\) negative **source item IDs**, preserving class membership. Thus profile-ID count is at most one quarter of the declared full count. Count an exact duplicate-removal step separately; do not manufacture compression by counting identical vectors repeatedly.

Required outputs per user/method: selected IDs; class sizes and budgets; fit status; construction parameters; support losses; requested-tolerance coverage; numerical validation status; ranking/utility metrics; timing identifiers; and failure details. Empty profiles, invalid vectors, foreign IDs, or a claimed certificate contradicted by independent checking are implementation/data failures, not favorable exclusions.

### 3.2 Proposed method: budgeted greedy finite-direction cover

For one class with H prototype vectors \(v_i\), compute

\[
A_{ji}=x_j^\top v_i,\quad h_j=\max_i A_{ji},\quad
C_{ji}=\mathbf 1\{h_j-A_{ji}\le\epsilon-2\tau\}.
\]

Here \(\tau=10^{-10}\) is the default float64 guard, subject to verification below. Each column covers catalog directions where that prototype is within the target tolerance of the full class support.

Use one global epsilon for both signs. On development users only, evaluate
\(\{0.0025,0.005,0.01,0.02\}\).
Select the epsilon with highest development user-mean top-20 overlap averaged over the four masks. If means differ by at most \(10^{-12}\), choose the smaller epsilon. Freeze it before confirmation. This is a small declared tuning grid, not a search for a mathematical optimum.

~~~text
budgeted_cover(A, prototype_ids, query_ids, B, epsilon, tau):
    require finite float64 A with at least one row
    require epsilon > 2*tau >= 0, 1 <= B <= number_of_columns
    require IDs unique and both ID arrays ascending
    h = row_max(A)
    C = (h[:, None] - A <= epsilon - 2*tau)
    selected = []
    uncovered = all rows
    g = vector filled with -infinity

    while len(selected) < B:
        if any uncovered:
            gain[i] = count of uncovered rows covered by unused column i
            choose unused i with maximum gain, breaking exact ties by item ID
            require gain[i] > 0
        else:
            # Spend the same budget as the comparators; improve remaining support.
            deficit = h - g
            choose query j with maximum deficit, tie by query item ID
            if deficit[j] > tau:
                choose unused maximizer of A[j, i], tie by prototype item ID
            else:
                choose lowest-ID unused prototype

        append i
        g = elementwise_max(g, A[:, i])
        remove rows with C[:, i] true from uncovered

    recompute g from the selected columns
    return selected IDs, h, g, uncovered count
~~~

If the worst-deficit branch cannot find an unused maximizer despite a deficit greater than tau, recompute that row with the scalar reference. Either select the correctly identified unused witness or classify the discrepancy as a numerical/implementation defect; do not silently insert an unrelated point.

**Termination and guarantees.** The budgeted algorithm always ends after B selections; it need not cover all rows. If every row is covered, the requested per-class tolerance is certified after independent numerical validation. Otherwise report the achieved maximum support loss and uncovered fraction. Do not label epsilon as an achieved bound.

Without the budget cap, ordinary greedy set cover terminates within H selections because every row has an original maximizer. The usual harmonic-factor bound applies relative to the optimum of the **guarded Boolean coverage matrix C**, not the unguarded ideal epsilon cover or the joint signed-score optimum. Removing numerically borderline edges can increase the covering optimum. The standard greedy charging argument bounds each selected set's charges to newly covered rows by reciprocal remaining coverage in an optimal set, giving the harmonic factor. This is background, not a new result or a guarantee for the capped version.

Do not bisect epsilon on the assumption that a particular greedy implementation's selected count is monotone. The optimal count is monotone; the heuristic path need not be.

### 3.3 Required comparators and ablation

| Method | Exact implementation and purpose |
|---|---|
| Full signed reference | \(h_P-h_N\), weight one, threshold zero. Correctness oracle for fidelity; no compression |
| Signed Ward medoids | Separately cluster each class into exactly \(B_c\) clusters using Euclidean Ward linkage on normalized vectors; select each cluster's source point minimizing the sum of squared distances, ties by item ID. This is PinnerSage-inspired [R5], not a reproduction of its entire production model |
| Catalog-aware worst-regret greedy | Initialize with the within-class squared-distance medoid. Repeatedly find the catalog row with largest support deficit and add that row's full-support maximizing unused prototype; ties by IDs. Fill with lowest unused IDs only when all deficits are within tau. Stop at \(B_c\). Same catalog information and budgets as the proposal |
| Random subsets | Uniform without replacement within each class, exactly \(B_c\). Use numpy.random.default_rng(numpy.random.SeedSequence([base_seed, userId, class_code])) with base seeds 11, 29, 47 and class_code 1 for positive, 2 for negative. Average the three outcomes within user; seeds are not independent experimental units |
| Restricted-catalog cover ablation | On the predeclared robustness users, construct coverage from the first 512 eligible IDs in the §4.5 hash-mask order; assess on their disjoint complement. Also construct a full-catalog cover and assess on that same complement. Labels and budgets are identical. This isolates access to evaluation directions, not all catalog access; it does not claim unseen-item certification |

Use SciPy linkage with method “ward” and Euclidean input, then cut the tree to exactly B clusters. Do not feed a cosine-distance matrix into Ward as though it were Euclidean. On unit vectors squared Euclidean distance equals \(2-2\cos\), but the library's Ward interface still requires its documented geometry [R12]. Check singleton and duplicate-vector cases and freeze the SciPy version.

Choose the **primary compression comparator** between Ward medoids and worst-regret greedy using development user-mean overlap. Break a difference within \(10^{-12}\) in favor of worst-regret, the same-information comparator. Freeze that choice; still report both on confirmation. Random is a reference/control, not an adequate sole comparator.

### 3.4 Preference-quality controls

All use the same capped positive/negative history and frozen item vectors:

- **Positive-only full maximum:** \(h_P(x)\). This tests whether adding the specified negative maximum helps.
- **Mean-contrast/Rocchio-style scorer:** \(w=\bar p-\lambda\bar n\), score \(x^\top w\); choose one global \(\lambda\in\{0,0.5,1,2\}\) on development later-rating AUC. Do not normalize a zero w; return zero scores and record the degenerate case.
- **Class-balanced linear ridge scorer:** minimize
  \[
  \frac1{2H_+}\sum_{p\in P}(p^\top w-1)^2+
  \frac1{2H_-}\sum_{n\in N}(n^\top w+1)^2+\alpha\|w\|^2.
  \]
  Use \(\alpha\in\{0.001,0.01,0.1,1\}\), selected globally on development later-rating AUC. If F stacks history rows, y is ±1 and W has the class weights above, set \(Z=W^{1/2}F,\ b=W^{1/2}y\), solve
  \[
  (ZZ^\top+\alpha I)c=b,\qquad w=Z^\top c.
  \]
  This is a small positive-definite solve with at most 128 rows. Check the primal gradient residual and an independent tiny direct solution.

Break equal development AUCs by the larger ridge penalty or smaller mean-contrast lambda, respectively. Choose the best of these two tuned scorers and positive-only maximum as the frozen **simple utility comparator**. Report all three on confirmation.

The signed reference's lambda stays **one**. Do not alter it to rescue poor confirmation performance. The linear controls need no neural training and are affordable strong alternatives for assessing whether the full nonlinear preference score is useful. No claim of superiority over modern collaborative recommenders follows from this restricted comparison.

If final preprocessing leaves no utility-evaluable development user, use fixed defaults lambda = 1 and alpha = 0.1, designate mean-contrast as an untuned reference, and mark the tuned-comparator usefulness claim unavailable. Do not select another cohort. Confirmation AUCs may still be reported under that explicit limitation, but cannot satisfy the stated tuned-comparator guardrail. With only the expected 17 development utility users, parameter/comparator selection can be noisy; retain every planned confirmation control.

### 3.5 Numerical conventions and independent checking

The mathematical input is the stored normalized float64 matrix. Preserve original relevance values too. Reject nonfinite entries, zero row norms, or norm errors exceeding \(10^{-12}\) after normalization; investigate rather than silently altering data. Do not clip dot products to [-1,1] in the reference path: tiny excursions should be diagnosed, and clipping can hide arithmetic mistakes.

For verified ordinary binary64 dot products, let \(u=2^{-53}\) and

\[
\beta=\gamma_{2d}\max_j\|x_j\|\,\max_i\|v_i\|,\qquad
\gamma_m=\frac{mu}{1-mu}.
\]

The usual finite-sum rounding argument and Cauchy–Schwarz bound give a conservative dot-product error beta. A support-deficit error bound is approximately \(2\beta+2u\). At d = 1128 and near-unit norms this is about \(5.1\times10^{-13}\). Use tau = \(10^{-10}\) only after confirming it exceeds ten times the calculated deficit bound and the observed independent-reference discrepancies.

Compute independently checked upper losses as
\(E_j^c=\max(0,\widehat e_j^c)+\tau\).
If a measured deficit is below \(-\tau\), fail the check. Use another conservative \(4\tau\) guard in strict sign/pairwise/top-set comparisons to cover score arithmetic. Requested-tolerance coverage requires the recomputed deficit plus tau to be at most epsilon.

Check a deterministic sample of 100 dot products, all worst-loss witnesses, and all rows within \(10\tau\) of a claimed threshold against Python math.fsum, with a few Decimal evaluations on tiny inputs. The independent verifier must recompute **both full and retained support maxima from their actual vectors**, using a separate blocking/path; do not reuse the selector's full-support array or coverage bitset. Include that full recomputation in calibration and promised-certificate construction costs. Verify all catalog losses in float64, scalar-check the witnesses, and store discrepancies. Constructing at epsilon minus two tau leaves slack for this independent validation; any still-borderline row is uncertified until its higher-precision witness check resolves it.

If the rounding model is not supported by the installed backend, disable that backend or fall back to a scalar/reference CPU calculation for validation. Do not keep the certificate while switching to float32, GPU reduced precision, or approximate maxima.

Rank by decreasing computed score with ascending movie ID for exact ties. Mark a top-20 boundary within \(4\tau\) as numerically ambiguous; independently recompute all potentially boundary-crossing rows in a \(10\tau\) band. If ambiguity remains, keep deterministic rankings, include those cases in primary metrics, report their frequency, and do not issue a strict certificate. If more than 1% of evaluation masks remain ambiguous, add a predeclared tie-aware overlap sensitivity analysis and qualify the primary interpretation. Do not delete difficult ties.

These are **numerical certificates under the checked CPU float64 model**, not machine-checked formal proofs. Distinguish actual observed ranking agreement, sufficient certification, and preference correctness.

## 4. Data, provenance, and development separation

### 4.1 Fixed primary data

Use the **standalone Tag Genome 2014** vectors and **MovieLens 20M** ratings, downloaded from the official GroupLens endpoints:

| Input | URL | Bytes and SHA-256 observed during this brief's preparation |
|---|---|---|
| Tag Genome 2014 | https://files.grouplens.org/datasets/tag-genome/tag-genome.zip | 43,510,670; 389e895fcf58144f89cae1f45d6f44430ed2ce6b6db988d0037e6643b23bc786 |
| MovieLens 20M | https://files.grouplens.org/datasets/movielens/ml-20m.zip | 198,702,078; 96f243c338a8665f6bcc89c53edf6ee39162a846940de6b7c8c48aeada765ff3 |

Read and retain the respective READMEs [R8, R9]. Downloads were successfully performed during planning. Accessibility on the target Mac is still to be checked. Download at most twice per archive after a transient failure; checksum failure permits one re-download, not repeated silent substitution.

The standalone archive's tag-relevance file is dated 23 March 2014; its README 30 March; its directory 2 April. An official announcement dated 2 April 2014 corroborates the release. This is corroborated archival provenance, not cryptographic proof of an immutable historical snapshot. The current server modification time alone is not the feature-creation time.

**Do not use the genome bundled in the current ML-20M ZIP.** Its genome files are dated October 2016, after the rating period. The features were themselves learned from tags, ratings, and reviews [R8–R10]. Merely splitting those ratings by time would not fix the feature leakage. Use the separate 2014 artifact and a later global cutoff. Global temporal separation matters in recommendation evaluation [R11].

The study is research under the published dataset terms. Acknowledge GroupLens and the prescribed papers. Package acquisition scripts and hashes, not redistributed dataset bytes. Do not send correspondence or request deployment rights as part of this task.

### 4.2 Parsing and feature construction

The standalone files are headerless tab-separated text:

- tag_relevance.dat: MovieID, TagID, Relevance.
- movies.dat: MovieID, Title, MoviePopularity.
- tags.dat: TagID, Tag, TagPopularity.

ML-20M ratings.csv is UTF-8 CSV with userId, movieId, rating, timestamp. Join by MovieLens movie ID, never by row position. Standalone tag IDs belong to that export.

Validate 9,734 movie IDs, 1,128 tag IDs, complete rectangular movie–tag coverage, uniqueness of pairs, finite relevance in [0,1], and nonzero row norms. Build rows/columns in ascending numeric ID order. Chunk parsing rather than materializing several large pandas copies. Do not round small relevance values to zero.

**Primary embedding:** use all 1,128 original relevance features, then normalize each movie row to unit Euclidean norm in float64. No centering, whitening, feature filtering, PCA, learned projection, or selecting an encoder based on performance. This is a high-dimensional community-feature embedding, not a modern language-model embedding; that limits generalization.

Keep the unnormalized input and transformation metadata. Do not train any new embedding model. Dimensionality controls below are diagnostics, not replacements for the native primary test.

### 4.3 Histories and evaluation labels

Use cutoff **2014-05-01 00:00:00 UTC**, Unix time 1398902400.

- Positive rating: at least 4.0.
- Negative rating: at most 2.0.
- Neutral: 2.5, 3.0, 3.5. Never treat these or unrated items as explicit negatives.
- History: matched ratings strictly before cutoff.
- Evaluation: matched ratings at or after cutoff through the dataset's end, before 2015-04-01 UTC.
- Active-history requirement: at least one matched pre-cutoff rating of any value on or after **2014-03-01 UTC**, time 1393632000.
- Retain the 64 most recent distinct embedding vectors per sign, or all if fewer. Order by descending timestamp then ascending movie ID. Remove exactly identical normalized vectors within each sign; retain the most recent representative ID. Record this deduplication separately.
- Require at least eight retained vectors of each sign.
- Retain cross-class embedding coincidences and flag them as contradictory evidence; do not silently remove inconvenient users.
- Exclude all pre-cutoff seen movie IDs from every query/evaluation catalog, including neutral and capped-away history items.

Check duplicate user–movie records. If duplicates exist, use the latest pre-cutoff value for history; for an unseen item's repeated post-cutoff values use its first later rating for utility. Log counts and verify this rule does not invalidate the planning cohort counts. Do not let a post-cutoff correction change the history.

A user's full eligible catalog is all usable 2014 movie vectors minus those pre-cutoff seen IDs. Require at least 1,000 eligible items for the declared mask protocol; report exclusions. Its item embeddings may be used during construction: that is explicitly **transductive catalog access**, not access to their future rating labels.

### 4.4 Cohort and the eligibility probe

The preparation scan, before vector deduplication and final exclusion checks, found:

| Cohort condition | Users | Users also having at least four later positives and four later negatives |
|---|---:|---:|
| At least eight historical ratings per sign | 56,112 | 566 |
| Same, active since 2014-03-01 | 1,443 | 323 |

The scan matched 19,743,826 rating rows to the 2014 catalog. These counts and archive hashes were measured; recommendation outcomes were not.

Use the active-history cohort. Sort users by the hexadecimal SHA-256 of the UTF-8 string **circle-probe:v1:{userId}**, with numeric ID as a final tie-break. First 64 users are development. The next **512** are the default confirmation cohort; the deterministic size rule in §7.3 may choose 256 or 1024 before confirmation outcomes are opened.

Under the preparation scan, the first 64 development users included 17 utility-evaluable users; the next 256/512/1024 included 63/114/230. Final exact counts can change after the declared validation/deduplication rules. Report the difference and reason.

Select primary users from **history information only**. Do not require future engagement to enter the compression cohort. Utility analyses use the conditional subset with at least four later positives and four later negatives. Report its size and the missing-utility count. Do not fill that subset with a different favorable cohort or treat its repeated rating pairs as independent users.

Store development and confirmation manifests separately. During development, code may compute confirmation eligibility counts and resource estimates, but must not expose confirmation model scores, fidelity metrics, or AUCs. Unit tests and synthetic data are separate from both.

Utility hyperparameters use later labels from **disjoint development users**, over the same calendar window as confirmation. This is retrospective held-out-user validation with historically earlier frozen features and histories. It is not an as-of-May-2014 forward-calendar deployment backtest: the chosen hyperparameters would not yet have been selected at that cutoff. No confirmation label may enter tuning or feature construction. Preserve this distinction when interpreting “later-rating” discrimination.

### 4.5 Four fixed candidate masks per user

Create these without future labels, all as subsets of the full eligible catalog:

1. All eligible items.
2. Exactly 1,000 eligible IDs selected by ascending SHA-256 of **ccpref-mask:v1:{userId}:{movieId}**.
3. The 1,000 eligible items with the greatest number of pre-cutoff ratings across ML-20M users.
4. The 1,000 eligible items with the fewest such pre-cutoff ratings.

Break popularity ties by movie ID. Count neutral ratings for popularity; do not use the genome's popularity column or post-cutoff popularity. Construct the compressed profile once from the full eligible catalog, then reuse it on all masks.

Equal weighting of these four masks is a fixed evaluation convention, not an estimate of production traffic. Report each mask separately. The randomly filtered catalog is used to assess **fidelity to the full scorer**, never as a sample of true negative preferences.

### 4.6 Data-access fallback

If the official 2014 archive is unavailable, corrupted after the bounded retry, or its provenance cannot be reconciled, complete the independent mathematical and synthetic checks and report the real-data step blocked. If a verified local copy supplied by the user matches the stated hash, use it.

Do not silently replace it with the 2016 bundled genome, a later dataset, generated “realistic” ratings, or a different encoder. A synthetic-only conclusion cannot support real recommendation feasibility. If ML-20M ratings are unavailable but the 2014 vectors are accessible, a clearly labeled feature-geometry diagnostic may proceed within the existing budget; it does not fulfill the primary real-user comparison or utility guardrail.

## 5. Minimal implementation contract

### 5.1 Stack and repository

Use ordinary local Python scripts with NumPy, SciPy, pandas for bounded chunk parsing, matplotlib for static figures, psutil, threadpoolctl, and pytest. Use Python 3.11 or 3.12 if supported by the actual installed wheels. Do not force a new interpreter installation if an existing supported native arm64 Python is suitable.

CPU float64 is the primary backend. The project does not require PyTorch, MPS, Metal, CUDA, a vector database, notebook execution infrastructure, a web service, or an optimizer package.

Intended repository:

~~~text
ccpref-poc/
  README.md
  pyproject.toml
  requirements-lock.txt
  config/study.json
  config/locked.json
  src/ccpref/
    __init__.py
    __main__.py
    cli.py
    environment.py
    data.py
    profiles.py
    scores.py
    compression.py
    certificates.py
    evaluation.py
    timing.py
    analysis.py
    ledger.py
  tests/
    test_geometry.py
    test_compression.py
    test_certificates.py
    test_data_protocol.py
    test_metrics.py
  source/                    # supplied notebook/summary if available
  data/raw/                  # ignored by git; official archives
  data/processed/            # ignored; reproducible local arrays/manifests
  results/
    environment.json
    provenance.json
    ledger.jsonl
    protocol_lock.json
    runs.jsonl
    user_metrics.csv
    failures.jsonl
    selected_profiles.jsonl
    tables/
    figures/
    report.md
~~~

Use one small JSON configuration. No experiment manager beyond a ledger and stage dispatcher is needed. Create a local git repository if none exists; do not push or publish it. If already working in an externally synced repository, follow its instructions and preserve unrelated changes.

### 5.2 Interfaces you must create

~~~text
load_catalog(config) -> normalized float64 matrix, movie IDs, provenance
build_profiles(ratings, catalog, config) -> profile records, seen IDs, cohort manifests
score_signed(X, P, N, block_rows) -> float64 vector
select_cover(A, IDs, query_IDs, B, epsilon, tau) -> selection and diagnostics
select_worst_regret(A, history_vectors, IDs, query_IDs, B, tau) -> selection
select_ward_medoids(history_vectors, IDs, B) -> selection
verify_support_losses(X, full_history, selected_history) -> losses and numeric evidence
evaluate_user(profile, methods, masks, later_labels) -> per-mask and per-user records
budget_guard(stage, projected_seconds, projected_bytes) -> continue/stop with ledger entry
analyze(results, locked_protocol) -> tables, figures, decision record
~~~

Separate selection from scoring, certification from ranking, and observed-rating utility from score fidelity. Functions must make array shape, dtype, normalization, class, and ID assumptions explicit. Use bounded row blocks; start at 2,048 rows and calibrate.

Every attempted method–user fit gets a unique run ID and a terminal status. Write append-only raw records before aggregate analysis. Do not overwrite a failure with a successful retry; link the retry to its parent.

### 5.3 Intended commands

The commands below are **interfaces for you to implement and verify**, not commands already tested during brief preparation:

~~~bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m ccpref inspect --config config/study.json
python -m ccpref fetch --config config/study.json
python -m ccpref prepare --config config/study.json
python -m pytest -q
python -m ccpref smoke --config config/study.json
python -m ccpref calibrate --config config/study.json
python -m ccpref pilot --config config/study.json
python -m ccpref freeze --config config/study.json
python -m ccpref confirm --config config/locked.json
python -m ccpref robustness --config config/locked.json
python -m ccpref report --config config/locked.json
~~~

Implement a budget-aware “run-authorized” command that calls only stages whose prerequisites passed and always attempts final reporting. A resume command must consult the ledger, avoid repeating completed valid work, and keep the same cumulative budget.

Inspect installed versions first, install compatible native wheels, run the small correctness suite, and then freeze the exact resolved versions. No target dependency version or binary compatibility was verified in planning. Record Python, NumPy, SciPy, BLAS, architecture, thread settings, OS, and package-lock hash.

A minimal configuration must include the fixed dates, thresholds, seeds/hash strings, history caps, budgets, epsilon grid, mask definitions, user-count rule, numerical guard, all resource caps, timeout policy, and output locations. The frozen file additionally records the chosen epsilon, compression comparator, utility comparator/hyperparameters, confirmed cohort, exact data hashes, thread count, selected timing users, and source-code commit.


## 6. Experiments, correctness, and inference

### 6.1 Tiny correctness must precede comparative results

Build independent scalar/reference checks, not tests that merely restate the vectorized implementation.

| Check | Minimum construction and expected result |
|---|---|
| Circular theorem examples | Arc weights (1,1), (3,1), and (2,3,2,3). Construct rational clearances; compare five/six-site examples and the \(2\sin a\sin\eta\) envelope at eta/a = 0.1, 0.5, 1. Use analytic piece endpoints and stationary points or an independent high-precision derivation, not a grid as proof |
| Same-plane high-dimensional lift | Embed all circle queries/prototypes through one orthonormal map into d = 128 and 1128. Dot products and scores must agree within the validated arithmetic tolerance. Require order/top-k agreement only where original gaps exceed the guard; apply the tie/ambiguity convention otherwise |
| Off-plane counterexample | Reproduce §2.4 at \(a=\pi/12,\eta=\pi/24\); it must exceed the circle formula and agree with \(\sin\eta\) |
| Distance identity | On small normalized inputs, compare dot-score and explicit nearest squared-distance score |
| Exhaustive tiny subsets | For 20 deterministic cases with at most four vectors per class and 12 queries, enumerate all nonempty subsets; verify actual support losses, signed-error identity, and every issued sign/top-k certificate |
| Cover oracle | On those small cases, enumerate covers and budgeted selections. Check coverage equivalence, selected membership/count, termination, and that a completed greedy cover is feasible. Do not assert it always has optimal count |
| Positive compression control | The tightly clustered construction below must meet its analytically protected top-20 result and support tolerance |
| Noncompressible control | The orthogonal construction below must fail the requested small support tolerance at quarter budgets; the implementation must not issue a false certificate |
| Cancellation and ties | Use P = N and matched retained subsets; signed errors can vanish despite large support losses. Check constant scores, exact ties, and zero mean-contrast vector |
| Deliberate defect controls | Remove a necessary selected witness, corrupt a claimed loss bound, and mislabel a selected ID. Independent validation must reject these. Test that changing post-cutoff ratings cannot change histories or selected profiles |
| Metrics | Tiny manually calculable AUC with ties; top-k overlap against exhaustive sorting; aggregation once per user |
| Ridge and clustering | Compare tiny ridge primal/dual solutions and gradient residuals; Ward returns exactly B nonempty clusters and actual member medoids |

Use deterministic seeds 20261029 for synthetic vectors and 314159265 for tiny cases. These checks do not require repeated seeds to establish their identities.

For the standalone five/six-site test, use boundaries 0°, 72°, 180°, 252°. Five sites: positives 18°, 234°; negatives 126°, 270°, 342°. Six sites: positives 36°, 216°; negatives 108°, 144°, 288°, 324°. Convert degrees to radians before computing cosine. The six-site optimum has a = pi/5; the five-site configuration has minimum clearance pi/10. The circle formulas in §2.2 give the expected margin where their buffer conditions apply.

**Positive control, specified:** create 32 distinct positive prototypes within Euclidean distance \(5\times10^{-4}\) of \(e_1\), and 32 negative prototypes within the same distance of \(-e_1\), by normalizing the center plus an orthogonal perturbation. Generate 600 unit queries: 20 with first coordinates \(0.75+0.005j,\ j=0,\ldots,19\), and the remainder with first coordinates in [-0.75,0.25]; fill the orthogonal part with deterministic unit directions. Any nonempty subset of each class loses at most 0.001 of class support, and the reference gap between these 20 items and the remainder is bounded far above the resulting score uncertainty. Every valid compressed subset should preserve this top-20 membership. Run at d = 128 and 1128. This is a sensitivity/correctness control, not evidence about real preferences.

**Hard control, specified:** take \(P=\{e_1,\ldots,e_{32}\}\), \(N=\{e_{33},\ldots,e_{64}\}\), and query \(x_j=\cos(0.1)e_j+\sin(0.1)e_{65}\) for j = 1,…,64, at d = 128 and 1128. Query IDs differ from prototype IDs, so seen-item exclusion does not remove them. Each own-class query needs its corresponding prototype to keep support loss below \(\cos(0.1)\). Quarter budgets cannot satisfy any epsilon in the declared grid on the entire catalog. Report this expected failure; do not “fix” it by weakening epsilon.

If the supplied notebook is available, rerun its two verifier cells as an additional provenance check. Avoid the plot dependency if unnecessary. These reruns do not replace the independent checks above.

### 6.2 Smallest experiment matrix

| Experiment | Scope | Claim resolved |
|---|---|---|
| Tiny suite and two synthetic controls | As above; at most 48 synthetic method/settings in addition to enumeration | Construction and certificate correctness; ability to observe success and failure |
| Development/pilot | First 64 development users; native d = 1128; four cover epsilons, both strong compression comparators, three random seeds, full and utility controls | Tune declared parameters, assess measurement sensitivity, forecast costs |
| Locked primary | Default 512 confirmation users, or prelocked 256/1024; one cover epsilon; all comparators; four masks; native d = 1128 | Primary overlap contrast, achievable fidelity, failure/tail behavior |
| Utility guardrail | Later-rating eligible subset of exactly those users | Compression versus full AUC; full scorer versus fixed simple control |
| Compression-budget sensitivity | First 64 confirmation IDs; cover/Ward/worst-regret at half and one-eighth class histories, in addition to primary quarter result | Whether failure is specific to aggressive budget; no rescue of the primary result |
| Catalog-direction ablation | Same 64 users; 512-direction cover versus full-direction cover, evaluated on the disjoint complement | How much success depends on catalog awareness |
| Precision diagnostic | First 16 confirmation IDs; float32 scoring of the frozen profiles versus CPU float64 | Score/ranking changes and why float64 certification cannot be silently transferred |
| Serving and storage comparison | Eight preselected development users spanning small/median/large H; fixed masks; cache controls | True build, storage, latency, amortization, and preprocessing dominance |
| Optional shared item-similarity cache | One full catalog, only if the resource gate permits; at most 15 minutes from the robustness allocation | Whether a feasible exact shared cache changes the operational conclusion |

Freeze the robustness users and timing users before confirmation. Budget sensitivities use the original frozen epsilon; do not retune after viewing their results. All robustness results are secondary/descriptive, not another opportunity to pass a failed primary test.

A d = 128 projection of real vectors is **not required**. Native d = 1128 remains the primary regime. Same-plane lifts test ambient-coordinate invariance; they do not show robustness to genuinely high intrinsic dimension. Report effective rank and dispersion from the native profiles rather than treating a low-dimensional toy as a substitute.

### 6.3 Metrics

For user u and mask m, let T be the full signed top-20 set and \(\widehat T\) a compressed set:

\[
R_{u,m}=\frac{|T\cap\widehat T|}{20},\qquad
R_u=\frac14\sum_m R_{u,m}.
\]

The primary paired difference is
\(\Delta_u=R_u^{cover}-R_u^{frozen\ comparator}\).
Report means and uncertainty, every mask separately, and distributions, not only a pooled item-level average.

Required fidelity/certification measurements:

- Mean user overlap; exact top-20 set-match rate.
- Fractions of users with \(R_u<0.90\) and \(R_u\ge0.95\); median and 10th percentile.
- Maximum and 95th-percentile absolute signed score error per user.
- Per-class maximum support loss, requested-epsilon coverage fraction, and complete-coverage rate.
- Fraction of masks with a valid strict top-20 certificate; fraction correct but uncertified; **false-certificate count, which must be zero**.
- Reference top-20/21 score-gap distribution and numerical ambiguity rate.
- Class sizes, imbalance \(H_+/H_-\), number of unique full-support witnesses, and history effective rank \((\operatorname{tr}G)^2/\operatorname{tr}(G^2)\) for the history Gram matrix. These are descriptive explanatory covariates, not post-hoc filters.

For later-rating utility, calculate within each eligible user

\[
AUC_u=\frac1{n_u^+ n_u^-}
\sum_{i:y_i=+1}\sum_{j:y_j=-1}
\left[\mathbf1(s_i>s_j)+\tfrac12\mathbf1(s_i=s_j)\right].
\]

Use all later eligible explicit positive/negative items; no sampled unknowns. A rank-based implementation is preferable for large histories, checked against the pairwise formula on tiny inputs. Macro-average users, not rating pairs. Report numbers of users, positive/negative ratings, neutral exclusions, and missing-utility users.

AUC here is discrimination **conditional on items users chose to rate and the declared activity cohort**. It is not randomized exposure, counterfactual recommendation value, full-catalog relevance, calibrated satisfaction, or online engagement. Do not present NDCG computed by labeling all unrated catalog items irrelevant.

### 6.4 Uncertainty, locking, and decision thresholds

Use a paired user bootstrap with 5,000 resamples and seed 20261030 for means and paired differences. Resample users with all their masks together. Average random-subset seeds within user before inference. Use two-sided percentile 95% intervals; retain raw user-level data so the intervals can be regenerated. For utility, bootstrap the conditional eligible-user set separately.

The two primary requirements are a conjunction:

1. Lower 95% bound for mean cover overlap is at least 0.95.
2. Lower 95% bound for mean paired overlap advantage over the frozen comparator is at least 0.02.

These support the stated scoped mechanism advantage only when correctness and failure criteria pass. The other comparator remains visible: if it performs better on confirmation, report that and do not describe the proposal as the best method overall.

For a **useful recommendation-profile compression** interpretation also require:

- Lower 95% bound for compressed-minus-full macro-AUC is at least -0.01.
- Full-scorer AUC lower bound exceeds 0.5, and full-minus-frozen-simple-comparator lower bound is at least -0.01.
- No materially disqualifying operational result is concealed. State the actual build and cache tradeoff even if the quality thresholds pass.

If another reported simple control materially outperforms the full signed scorer on confirmation, state that result and do not claim competitive recommendation quality solely because the development-selected comparator passed. Do not retune or switch the formal comparator after observing confirmation.

A claim that negatives improve discrimination requires a separate reported positive full-minus-positive-only interval; do not infer it merely because both signed models match each other. Treat such secondary comparisons as descriptive unless an effect threshold and inferential family were separately frozen.

Classification of the primary findings:

- If both primary lower bounds pass: support the declared overlap claim.
- If the upper bound of a required quantity is below its threshold: evidence against that required effect in the tested regime.
- If an interval straddles its threshold: inconclusive about that threshold.
- If compression fidelity passes but comparative gain does not: compact profiles may be feasible; the proposed covering machinery has not established a material advantage.
- If fidelity passes but utility is weak or uncertain: report a compression result only, with recommendation usefulness negative or unresolved.
- If a simpler method matches or wins: prefer it for the scoped engineering recommendation; preserve the unsuccessful proposal's measurements.

Do not infer “no effect” merely from lack of statistical significance. Do not continue increasing the sample after examining confirmation outcomes. Do not treat masks, ratings, bootstrap replicates, or seeds as independent users.

Freeze the entire protocol in config/locked.json and record its SHA-256 plus code commit before any confirmation scores are produced. The lock includes chosen epsilon and comparators, user IDs, budgets, masks, numerical conventions, timeouts, backend/thread settings, sample-size decision, analysis, and outcome rules. Later changes to the hypothesis, input distribution, primary metric, or algorithms are exploratory. A corrected bug invalidates affected runs; rerun the same locked design within the repair reserve, preserving both versions.

### 6.5 Failures and missing results

Record every timeout, parse failure, numerical failure, dependency failure, and aborted run with elapsed time, memory, traceback/reason, and retry linkage.

An unresolved reference or baseline correctness defect invalidates the affected comparative claim. Do not obtain a positive result by allowing a broken baseline to score poorly. Report complete paired results descriptively and conservative bounds for all locked users: missing proposed overlap may lie in [0,1], as may missing comparator overlap. For a conservative advantage lower bound, impute missing proposed overlap as zero and missing comparator overlap as one; reverse those choices for an upper bound. Do not silently use complete-case CIs as the full-cohort answer.

More than 5% unresolved missing/invalid primary method–user pairs prevents a supported primary claim. At or below 5%, construct pessimistically completed **full locked-cohort user vectors** and apply the same 5,000-resample user bootstrap. Support requires their lower 95% bounds to pass both primary thresholds, as well as the validated-pair analysis; an imputed point estimate alone is insufficient. Bootstrap optimistic completed vectors analogously for upper bounds and negative/inconclusive interpretation. Label the operational failures. An invalid full reference prevents a fidelity result for that user.

Apply the same pessimistic/optimistic completion and bootstrap to computationally missing AUCs **among users who have enough future labels**, using the [0,1] AUC bounds and the correct direction for each comparison. An unresolved utility-baseline defect invalidates its guardrail; a failed method cannot disappear from the evaluation. Distinguish these computational failures from users who genuinely lack the required future labels.

Missing future labels are different: those users remain in compression evaluation and are outside the explicitly conditional utility estimand. With fewer than 50 utility-evaluable confirmation users, report descriptive utility results and mark the recommendation guardrail inconclusive. Fifty is a coverage floor, not a guarantee of adequate precision; the actual interval still decides.

## 7. M3 Max feasibility, timing, and cumulative budgets

### 7.1 Inspect the actual target first

Do not infer RAM, disk, macOS, acceleration support, precision, or thermal conditions from “M3 Max.” Record:

- macOS version, hardware model/chip, native architecture, physical/logical cores.
- Total RAM, currently available memory, swap condition, free disk in the project volume.
- Python executable/architecture; installed dependencies; NumPy/SciPy and BLAS configuration.
- Detected thread pools; whether running under translation; power source and relevant background workload if observable.

Use read-only system inspection such as sw_vers, uname, sysctl, Python platform/sysconfig, psutil, shutil.disk_usage, numpy.show_config, and threadpoolctl. Do not change system power settings or terminate unrelated applications.

Before creating the virtual environment or installing dependencies, record a start timestamp and use standard-library/system commands for the initial architecture, memory, and disk check. Save setup command start/end times in a simple text/JSON log, then import them into the ledger once it exists. Installation and bootstrap work do not escape the six-hour accounting because the ledger was not yet implemented.

Use one process for scientific runs. Calibrate one versus four BLAS threads on development data and freeze the faster stable setting; no unconstrained per-method thread tuning. Some Apple Accelerate configurations may require environment variables before NumPy import rather than runtime thread control. Verify behavior and record what was actually controlled.

The CPU reference avoids dependence on GPU float64 support. Optional GPU acceleration is outside this PoC: it adds a separate numerical/backend question without being necessary for the planned scale.

If the target Mac is unavailable, complete source review, independent mathematics, and portable correctness work that can run locally, but mark all target performance claims blocked. Do not call hosted timings M3 results.

### 7.2 Hard resource ceilings and scaling

**Total local workload cap: six hours, 360 minutes, cumulative across all stages, failed attempts, tuning, repeated timings, and repairs.** Include setup/data commands in their allocated stage. Agent reasoning and code-writing latency are not experimental computation; record their elapsed effort separately. Never reset the workload ledger when switching stages or fallbacks.

| Allocation | Maximum command/runtime minutes |
|---|---:|
| Environment, dependency/data access, provenance, preparation | 25 |
| Tiny correctness and analytic diagnostics | 25 |
| Resource calibration | 20 |
| Development mechanism pilot and declared tuning | 45 |
| Locked comparison and utility | 135 |
| Predeclared robustness and operational cache controls | 35 |
| Bounded repairs/retries across the entire project | 40 |
| Preserve evidence, analyze, regenerate plots/tables, report | 35 |
| **Total** | **360** |

Reserve the final 35 minutes: stop starting experiments once 325 minutes have been consumed, even if a stage budget remains. Stage caps and the total both apply. Unused time may improve final analysis but does not authorize new experiments or extra tuning. The optional shared-cache test receives at most 15 of the 35 robustness minutes.

Additional caps:

- Peak process-tree RSS: the minimum of **8 GiB**, **25% of physical RAM**, and **50% of available memory measured at start**. Require at least a 1 GiB usable cap and at least 2 GiB available before data processing; otherwise stop for local resource infeasibility or wait for user-directed availability, without changing other applications.
- Working data, environments, and results: **4 GiB disk**. Require at least 8 GiB free initially. Keep archives plus necessary extracted files; do not extract unused bundled genome data.
- At most **15,000 method–user runs**. One unit is one user × method × fitted parameter/random-seed configuration, including all four candidate masks and associated metrics. Each timing rebuild or retry counts again; masks and repeated score calls within one timing batch do not individually add units. Include development, robustness, timing rebuilds, retries, and failed attempts. Primary maximum is 1,024 confirmation users. Repeated scoring is additionally bounded by time and at most 2,000 timing batches.
- Per user/method construction timeout selected at calibration as \(\min(30\text{ s},\max(10\text{ s},5t_{95}))\), where \(t_{95}\) is the measured development construction cost for that method. Freeze it. Stages still stop at their global limits.
- At most two ordinary repair cycles for any one defect and 40 repair minutes in total. No unbounded redesign campaign.

Approximate storage, calculated from dimensions rather than measured Mac peaks:

| Array/state | Size before overhead |
|---|---:|
| 9,734 × 1,128 float64 vectors | 87,839,616 bytes, about 83.8 MiB |
| One 9,734 × 128 float64 similarity matrix | 9,967,616 bytes, about 9.5 MiB |
| One class cover matrix at 64 prototypes | About 0.6 MB as Boolean bytes |
| Full 9,734² float64 item Gram matrix | 758,006,048 bytes, about 0.71 GiB |
| Full 128-ID profile / compressed 32-ID profile | 512 / 128 bytes at uint32, before metadata |
| One full per-user float64 score cache | Up to 77,872 bytes, before metadata |
| One full sorted movie-ID cache | Up to 38,936 bytes at uint32 |

The last two caches are much larger than a small ID profile but can serve exact rankings cheaply. A fixed full-catalog top-20 ID list is only 80 bytes and is a required simple control for identical repeated requests; it cannot handle arbitrary filtered masks.

Core costs for one class are \(O(MH_cd)\) to compute full similarities and approximately \(O(B_cMH_c)\) for straightforward Boolean greedy selection. Ward needs roughly quadratic history-distance work and is tiny at H ≤ 64. Score a request of L items from K selected prototypes in \(O(LKd)\). A score cache needs lookups plus top-k extraction; a ranked-ID cache scans the saved order against membership.

Process users sequentially. Never accumulate all user × catalog × history matrices. Keep per-user matrix workspaces transient. CSV parsing, NumPy copies, Boolean matrices, bootstrap arrays, subprocesses, and memory-mapped resident pages count in peak memory.

**Planning estimate:** native arrays and histories fit comfortably below the stated caps if processed sequentially; the study should take hours rather than days of local computation. Actual throughput is unmeasured until calibration. Expect approximately 0.5–1 hour of setup, 4–8 hours of implementation/debugging effort, up to 6 hours of cumulative local workload, and 1–2 hours of analysis/write-up effort, with overlap possible. A realistic end-to-end planning range is roughly 8–16 hours; it is not a benchmark or completion guarantee.

### 7.3 Calibration and sample-size rule

Calibrate on eight development users spanning small, median, and large history counts. Use 1,000 and full-catalog query sizes and representative history sizes, including 64 per sign where available. Measure dot products, each selector, independent certification, full ranking, utility calculation, and data preparation. Measure actual process-tree peak memory.

Forecast each remaining stage using observed sizes and operation counts, multiply the extrapolated runtime by **two**, and include all methods, four masks, random repeats, retries, and analysis reserve. Do not extrapolate only the fast inner kernel.

Initial calibration selects costs/block/thread settings only. **After the development pilot selects the comparator, and before protocol freeze**, choose the final confirmation N. Default N is 512. From development paired differences against that comparator, estimate standard deviation s and compute

\[
N_{\mathrm{precision}}=\left\lceil(1.96s/0.01)^2\right\rceil.
\]

This targets an approximate 0.01 mean-difference interval half-width, enough to resolve a 0.02 practical threshold in favorable circumstances; it is not a formal power guarantee. Choose the smallest allowed N in {512,1024} at least as large as this estimate, provided the doubled cost forecast fits all caps.

If 512 does not fit, 256 is permitted **only before confirmation**, only if the precision estimate is at most 256, and only if the full 1,128-dimensional/catalog/history/method protocol still fits. Record the narrower user coverage. If required precision exceeds 1024 or the smallest scientifically meaningful design cannot fit, report practical infeasibility or an explicitly inconclusive pilot. Do not replace the primary test with fewer dimensions, a few hundred catalog items, weaker baselines, reduced precision, or favorable synthetic cases.

The preparation counts suggest 114 utility-evaluable users at N = 512 and 230 at N = 1024. Do not choose N after inspecting confirmation AUC differences. If utility precision remains poor, preserve that inconclusive result.

Resize row blocks to meet memory ceilings before shrinking scientific scope. Recalibrate once after an engineering repair, charged to the repair allowance. Do not repeatedly recalibrate until an optimistic estimate appears.

### 7.4 Timing boundaries and cache comparisons

Separate evaluation-oracle work from production construction, and count each honestly.

For cover and worst-regret construction, include gathering history vectors, full catalog-by-history similarities, selection, validation needed for promised bounds, and metadata serialization. If untimed quality runs reuse the full matrix across methods, that is allowed for efficiency, but do not report the reused matrix as free production preprocessing.

For Ward/random construction, include their actual history-only selection cost; do not charge them a full catalog scan that is needed only by the experimental evaluator. Their query scoring still uses the selected vectors. For full-profile serving, count source-ID storage, actual gather/scoring, and top-k extraction.

Mandatory operational controls:

1. Full uncompressed profile scored on demand.
2. Compressed profile scored on demand.
3. Exact per-user catalog score cache: construct with the full scorer; queries gather eligible scores and select top-20.
4. Exact full ranked-ID cache: sort once; for a mask, scan the order for its first 20 members.
5. Exact fixed top-20 ID cache for repeated **unchanged full-catalog** requests only.

Profile IDs share one global item matrix. Do not compare a compressed ID list with unnecessarily duplicated full-history vectors. Report persistent **hot serving state**, retained raw-history state, shared item storage, and transient buffers separately. If raw history is retained for updates, hot-profile reduction does not reduce that raw store.

A shared exact item Gram matrix is feasible enough at this catalog size to merit a bounded check if the actual RSS cap is at least 2 GiB and projected construction is below 15 minutes. Build it in row blocks, verify sample entries independently, and measure its one-time cost and full-profile query/build costs through lookups. Count the roughly 0.71 GiB shared state. This is an architectural control, not required infrastructure for every experiment. If unavailable, do not claim comprehensive optimality for frequent-history-refresh workloads.

Measure:

- First process request after loading arrays, explicitly called **process-cold**, not cold operating-system cache.
- Warm construction and warm query latency with all methods using the same resident catalog and thread setting.
- True end-to-end costs including data/model loading separately.
- Query scoring plus filtering plus top-20 selection; no exclusion of expensive gathers or sorts.
- Median and interquartile range across five timing blocks, each at least 0.2 seconds where feasible, with three fresh-process construction runs on the eight timing users. Batch cheap calls to avoid timer quantization. Bound everything by the stage cap.
- Peak RSS from fresh processes via psutil process-tree sampling and the platform's process high-water mark; record byte/unit conversions. tracemalloc alone misses numerical arrays and is insufficient.

Use perf_counter_ns. CPU operations here are synchronous, so no GPU synchronization is needed. Warm methods equally, rotate timing order deterministically, keep inputs identical, avoid concurrent benchmarks, and record interruptions. Do not infer algorithmic improvement from a Python-loop baseline versus a vectorized candidate.

For R repeated requests report

\[
T_{\mathrm{total},m}(R)=T_{\mathrm{build},m}+R\,T_{\mathrm{request},m},
\]

at R = 1, 10, 100, 1000, and report any break-even solution. If the denominator for a proposed speedup is nonpositive, say there is no break-even. Include the shared-cache cost separately and its stated amortization over users. Do not select a favorable request count after seeing timings. These reuse scenarios are assumptions, not measured production frequency.


## 8. Evidence gates and autonomous actions

Aim to obtain an informative **eight-user smoke/pilot within roughly one hour of local command runtime** when dependencies and downloads cooperate. The stage ceilings are conservative maxima, not time to spend automatically. Do not construct later-stage infrastructure before its prerequisite evidence exists.

| Stage and question | Prerequisites and minimum work | Evidence/artifacts | Budget | Decision and next action |
|---|---|---|---:|---|
| 0. Constructive feasibility: are the inputs and target usable? | Read this brief; inspect target using standard library/system commands; verify data access, archive hashes/dates, schemas, cohort, memory/disk; establish ledger | Environment/provenance records; cohort counts; validated source inputs | 25 min | Continue if native high-dimensional data, viable cohort, and resources exist. One bounded dependency/download repair. If indispensable data or target unavailable, complete unaffected mathematics and report exactly what is blocked |
| 1. Tiny correctness: is the mechanism actually implemented? | Stage 0 mathematical inputs; scalar score, selectors, certificates, metrics; §6.1 suite | Tests, tiny raw values, counterexample derivation, certificate witness records | 25 min | Continue only if invariants, known successes/failures, and independent references pass. False certificate or wrong full score requires repair. If unresolved after allowance, classify invalid experiment and stop empirical comparison |
| 2. Resource calibration: can the full scientific design fit? | Correct tiny implementation; eight development profiles; measured construction/scoring/verification/memory | Cost table, scaling forecast, proposed thread/block settings | 20 min | Continue if doubled forecast fits. Change blocks/threads once as justified. Do not shrink scientific dimensions/catalog. Final N is selected after Stage 3, because its variance/comparator are not known yet |
| 3. Mechanism pilot: can the effect be measured and is the protocol ready? | Stage 2 feasible; all 64 development users; declared epsilon and utility grids; controls | Dev metrics, chosen parameters/comparators, precision estimate, final N and forecast, complete lock | 45 min | Proceed to confirmation if correctness holds, chosen N/precision fit, and inputs are valid. A noisy or unfavorable pilot alone is not a scientific refutation. No additional hyperparameter grid. If precision/resource requirements cannot fit, report an inconclusive or infeasible pilot |
| 4. Locked comparison: does the predeclared effect hold? | Frozen protocol hash and code; confirmation outcomes unopened | Append-only records for every locked user/method; failures; fixed-cohort metrics | 135 min | Execute once in a predeclared order. Ordinary defects may use repair reserve, invalidating/rerunning affected results. Stop on hard caps. Never change thresholds or add users after results |
| 5. Diagnosis and robustness: what explains the observed outcome? | Valid primary output, or a scoped failure needing the listed controls | Budget curve, restricted-direction result, float32 discrepancy, cache/storage/timing table | 35 min | Run only predeclared controls that fit. Omit optional cache if its gate fails and narrow the operational claim. Do not convert a robustness success into a new confirmation claim |
| 6. Report: what can actually be concluded? | All preserved evidence, even for an early stop | Reproducible tables/figures, report, commands, limitations, outcome category | 35 min reserved | Finish after any prior gate outcome. State work not reached. Recommend at most one evidence-justified next step |

The 40-minute repair reserve is shared by all stages and does not enlarge their scientific scope. A repair may fix code, parsing, numerical mistakes, or compatibility; it may not change the primary hypothesis, give a method more data, drop a strong comparator, or relax the target.

If the primary hypothesis appears false in development, still execute the locked comparison when scientifically valid and affordable: this provides the useful negative result the project is meant to support. If an analytic obstruction invalidates a proposed universal claim, document it immediately, but do not confuse the already-demonstrated failure of the unqualified circle extension with failure of this finite-catalog project.

Save a ledger checkpoint before each potentially expensive subprocess, including a predicted maximum. The monitor must stop new work before the reserved analysis budget is consumed. On stop, flush raw outputs and generate a partial report; do not leave only a traceback.

## 9. Outcome interpretation and required final artifacts

### 9.1 Use the correct outcome category

| Category | Evidence required here | Interpretation that is allowed |
|---|---|---|
| Supported contribution | Correct implementation, adequate locked comparison, specified thresholds/CIs and scoped operational assessment | A reproducible high-dimensional compression result in this catalog/cohort; algorithmic priority remains unclaimed |
| Informative negative result | Valid comparison with an interval ruling out the required effect, or a verified analytic counterexample to a specific extension | The specified cover mechanism/operating target is unsupported under tested conditions; a simpler baseline may be preferable |
| Practical infeasibility | Measured costs or conservative measured scaling violate the declared target/caps while preserving scientific meaning | The intended local study/workload does not fit; not a mathematical refutation |
| Inconclusive evidence | Threshold-straddling intervals, insufficient utility coverage, blocked target/data, or insufficient valid runs | The claim remains unresolved; identify the exact missing evidence |
| Invalid experiment | Unrepaired scoring, data, baseline, leakage, certificate, or measurement defect | No comparative scientific conclusion from affected runs; preserve the defect and reproduction |

Assign separate statuses to: mathematical transfer, primary ranking fidelity/advantage, observed-rating usefulness, and operational practicality. A study can support one and be negative or inconclusive on another.

A failure at quarter budgets is not a proof that no compressed signed representation exists. It can reflect the cover heuristic, the independently allocated class budgets, fixed lambda, source-embedding geometry, catalog size, or stringent ranking gaps. The matched regret baseline and the half/eighth sensitivity help distinguish these explanations, within the limited tested family. Arbitrary learned or freely synthesized prototypes remain outside scope.

The mathematical obstruction to exact global support pruning and the off-plane counterexample are valid independent findings. Neither predicts the real-data result by itself.

### 9.2 Final research report

Write a standalone report with:

1. The selected question and the actual conclusion in the first paragraph.
2. Source theorem scope, what was changed, the high-dimensional identities, and the counterexample.
3. The exact data snapshot, transformations, history definition, cohort/utility coverage, and limitations of observed-rating evaluation.
4. Frozen protocol, chosen parameters, deviations, stages reached, and cumulative resource use.
5. Primary effect estimates and intervals, all strong controls, user tails, certificates, errors, and failed runs.
6. Separate preference-quality and fidelity results, including negative-feedback and simple-scorer comparisons.
7. Real measured M3 environment, build-inclusive costs, exact-cache controls, persistent/temporary byte counts, and explicit reuse assumptions.
8. An outcome classification for each claim; why negative, infeasible, inconclusive, and invalid outcomes differ.
9. Remaining uncertainty and **one** next step worth doing, or a reason not to pursue the approach further.
10. References and reproducibility commands.

Do not claim a novel mathematical theorem for the finite-catalog bounds, a new max-sim recommender, success on arbitrary semantic embeddings, deployment rights, statistical generalization to all users, or an online improvement.

### 9.3 Artifacts required only for stages actually reached

- Runnable source and concise setup/resume/use instructions if implementation occurred.
- Exact dependency lock, environment/provenance records, configuration and lock hashes, source commit, deterministic cohort/mask definitions, and seeds.
- Raw per-user/per-mask measurements, selected IDs, failures/retries, numerical witness checks, and the cumulative resource ledger when experiments occurred.
- Regenerable summary CSV tables and a small set of static figures: overlap distribution/paired comparison; support error versus ranking-gap/certification behavior; budget sensitivity; and storage/build/query tradeoff. Avoid a dashboard.
- A research report. For analytic or access stops, include the verifiable derivation/counterexample, probe evidence, and blocked commands instead of empty benchmark tables.
- A compact code/results archive if needed for handoff, excluding licensed raw data, virtual environments, temporary matrices, and caches reproducible from public downloads.

The report-generation command must work from raw records without retuning or rerunning confirmation. Include failures in that workflow. Preserve all unfavorable results. Do not invent a plot, timing, statistical interval, or hardware validation for a stage that was not run.

## 10. Research coverage, novelty boundary, and review dispositions

### 10.1 Closest prior art and project selection

Research and source inspection were conducted on **29 October 2026** using two search engines, primary papers, official dataset documentation, and relevant backward/forward references. Search terms included circular/inverse Voronoi prototypes, maximum-margin nearest prototypes, multiple-interest maximum-cosine recommendation, positive/negative preference prototypes, support-function coresets, regret-minimizing representatives, finite-direction hitting sets, and Tag Genome temporal provenance.

This was targeted, decision-relevant research, not an exhaustive global novelty search.

| Alternative or prior mechanism | Finding | Project consequence |
|---|---|---|
| Directly generalize the circular optimum/3r/50% overhead | Assumptions fail; explicit off-plane counterexample | Reject that claim; retain only applicable score identities |
| Train general high-dimensional maximum-margin prototypes | Existing nearest-prototype margin literature, including 2018 and a 2025 follow-up [R1, R13] | A broad margin-training project offers weak novelty and adds nonconvex optimization; not selected |
| Multiple user interests scored by a maximum | Established in recommendation, including MIND and PinnerSage [R5, R6] | Multiple vectors alone are not the contribution |
| Positive and negative preference prototypes | Already explicit in PPA++ [R7], with a different trained architecture | Signed prototypes alone are not new |
| Preserve maxima with a compact representative subset | Regret/minimum-representative and hitting-set literature [R2–R4] is the closest mathematical prior art | Reuse and attribute known machinery; test its high-dimensional signed-history regime |
| Project all embeddings to one circle | Loses information and lacks a guarantee of preserving relevant high-dimensional rankings | Not an acceptable shortcut to a favorable result |
| Finite-catalog signed compression with measured limits | Constructive, simple, independently testable, and locally bounded | Selected as the smallest defensible R&D study |

[R3] uses relative support loss over all directions and requires positive support in every direction, equivalently the origin inside the convex hull. Small histories in 1,128 dimensions cannot generally satisfy that interior condition. This study instead uses **additive loss over observed catalog directions**. Do not transplant its hardness or approximation theorems without a reduction. [R4]'s “k” denotes a query-reference rank, not this study's retained-profile budget.

The promising gap is empirical and operational: strong prior systems and geometric methods do not themselves answer this precise signed-history, finite-catalog, high-dimensional, cache-aware test. That is a reason to run the diagnostic, not proof of publication novelty.

### 10.2 What was and was not accessed

The supplied documents were read in full; the relevant sections of full-text sources and official documentation [R1–R12] were inspected at the locations listed below. The short official 2014 announcement was available through search results; direct page requests sometimes timed out. Both dataset archives were actually downloaded and inspected.

For [R13], only the publisher abstract/metadata was verified; full PDF access failed during review. The full 2013 MaxMF paper and the older foundational epsilon-kernel proof were not needed for the selected algorithm and were not verified. Do not imply they were exhaustively checked. PPA++ cites an earlier PPA paper; that predecessor was not separately audited.

No recommendation PoC, real-model comparison, or M3 Max benchmark was performed while preparing this brief. The source reruns, counterexample arithmetic, and data eligibility scan are the actual computations listed in §2.3.

### 10.3 Three review passes and material repairs

The preparation used distinct mathematical/novelty, experimental, and execution reviews. The actionable core was challenged against positive-result confounds and empty-project execution. The resulting dispositions include:

| Review finding or stress scenario | Disposition in this brief |
|---|---|
| The theorem could hide an inaccessible construction oracle | Identified prescribed boundaries as essential input; excluded boundary learning and higher-dimensional theorem claims |
| Off-plane prototypes defeat the transferred circle formula | Included an explicit analytic/numeric counterexample; source theorem retained within scope |
| Mechanism is equivalent to known methods | Attributed nearest prototypes, max-interest scoring, and regret/set-cover machinery; narrowed contribution to a measured regime |
| Classwise errors can cancel | Used the sharper subset score bound and prohibited universal negative conclusions from support failure |
| Budgeted cover might be mistaken for an epsilon guarantee | Required actual losses, uncovered rows, and verified certificates; no claim of complete cover when the cap binds |
| Roundoff can change tied rankings | Required gap-aware checks, guarded certificates, independent support recomputation, and ambiguity reporting |
| A positive result could be only a favorable synthetic construction | Native 1,128-dimensional real-user confirmation is mandatory; synthetic success is a control only |
| A strong same-information baseline could remove the gain | Included worst-regret selection and practical Ward medoids, with matched budgets and declared tuning |
| Fidelity could preserve a poor recommender | Separate later-rating AUC and tuned simple controls |
| Temporal rating split could hide feature leakage | Use the verified standalone 2014 genome; reject the later bundled feature file |
| Too few later ratings could make utility impossible | Actual eligibility scan; active-history cohort and honest conditional utility sample/CI |
| Static or filtered serving could be solved by caching | Exact score, full-rank, and fixed-top-20 cache controls; profile-ID accounting; optional shared Gram control |
| Preprocessing could dominate any gain | Charge full support computation, validation, model construction, and stated reuse counts |
| Memory/runtime or numerical backend could fail | CPU float64 reference, measured calibration, fixed caps, bounded repairs, explicit infeasible/invalid outcomes |
| Pilot could be noisy or main hypothesis false | No confirmation peeking or rescue tuning; permit negative and inconclusive completion |
| Sources/data/hardware could be unavailable | Bounded retries and unaffected analytic work; no silent substitute or unperformed-validation claim |

Residual risks are explicit: the single older movie catalog may not represent current semantic embeddings; fixed histories can become stale over the evaluation year; future-rating coverage is conditional; bootstrap intervals describe user variation for this fixed catalog, not uncertainty over datasets; exact caches may dominate practical serving; numerical certificates rely on the checked arithmetic model. These are limitations to report, not reasons to alter the frozen experiment.

## 11. References and precise source locations

**[A1] Supplied manuscript, author not identified.** *Sharp margins and prototype complexity for circular cosine preferences.* Notebook dated 29 October 2026, in Circular_Cosine_Preference_Theorem.zip, path Circular_Cosine_Preference_Theorem/Circular_Cosine_Preference_Theorem.ipynb. Critical locations: §1 Theorems 1–2; §2 Eqs. (1)–(3); §3 Eqs. (4)–(7); §4 Eqs. (8)–(9); §5 sharp ratio; §6 explicit five/six-site example; §7 scope/perturbations; §§9–11 executable checks and figure. The accompanying .md is an explanatory summary, not an additional independent proof.

**[R1] Yoshifumi Kusunoki, Chiharu Wakou, Keiji Tatsumi (2018).** *Maximum-Margin Model for Nearest Prototype Classifiers.* Journal of Advanced Computational Intelligence and Intelligent Informatics 22(4), 565–577. [DOI: 10.20965/jaciii.2018.p0565](https://doi.org/10.20965/jaciii.2018.p0565). [Primary PDF](https://www.jstage.jst.go.jp/article/jaciii/22/4/22_565/_pdf). Critical locations: §3.1, Eqs. (11)–(13), classwise nearest-prototype discriminants; §3.2, Eqs. (14)–(16), margin formulation. This is prior art for high-dimensional margin-based prototype classification, not the circle theorem.

**[R2] Danupon Nanongkai, Atish Das Sarma, Ashwin Lall, Richard J. Lipton, Jun Xu (2010).** *Regret-Minimizing Representative Databases.* Proceedings of the VLDB Endowment 3, 1114–1124. [Primary PDF](https://vldb.org/pvldb/vol3/R99.pdf). Critical locations: §3.1, Definitions 2–4, especially absolute regret as full-minus-subset gain on p.1116; §5 greedy representative selection. Establishes the representative-maxima vocabulary and mechanism.

**[R3] Yanhao Wang, Michael Mathioudakis, Yuchen Li, Kian-Lee Tan (2020).** *GRMR: Generalized Regret-Minimizing Representatives.* Technical report dated 21 July 2020, arXiv:2007.09634. [Stable record](https://arxiv.org/abs/2007.09634); [primary PDF](https://arxiv.org/pdf/2007.09634). Critical locations: §2.1, Definitions 1–2 and Condition 1; §2.2 geometric interpretation; §4.2 Algorithm 3; §5 experimental regimes. The report's all-direction relative-error assumptions differ from this brief.

**[R4] Pankaj K. Agarwal, Nirman Kumar, Stavros Sintos, Subhash Suri (2017).** *Efficient Algorithms for k-Regret Minimizing Sets.* SEA 2017, LIPIcs 75, 7:1–7:23. [DOI: 10.4230/LIPIcs.SEA.2017.7](https://doi.org/10.4230/LIPIcs.SEA.2017.7). [Primary PDF](https://drops.dagstuhl.de/storage/00lipics/lipics-vol075-sea2017/LIPIcs.SEA.2017.7/LIPIcs.SEA.2017.7.pdf). Critical locations: §3 representative-size/coreset discussion; §4.1 Lemma 7 and Algorithm 1, hitting-set reduction. Its relative and positive-orthant formulations are not copied as guarantees here.

**[R5] Aditya Pal, Chantat Eksombatchai, Yitong Zhou, Bo Zhao, Charles Rosenberg, Jure Leskovec (2020).** *PinnerSage: Multi-Modal User Embedding Framework for Recommendations at Pinterest.* KDD 2020. [DOI: 10.1145/3394486.3403280](https://doi.org/10.1145/3394486.3403280). [Primary PDF](https://arxiv.org/pdf/2007.03634). Critical locations: §§2–3.1, Ward clusters and fixed item embeddings; Eq. (2), source-item medoid; §5.3.2, maximum-cosine candidate ranking. The proposed signed Ward control adapts these components; it does not reproduce their complete system.

**[R6] Chao Li, Zhiyuan Liu, Mengmeng Wu, Yuchi Xu, Pipei Huang, Huan Zhao, Guoliang Kang, Qiwei Chen, Wei Li, Dik Lun Lee (2019).** *Multi-Interest Network with Dynamic Routing for Recommendation at Tmall.* CIKM 2019. [DOI: 10.1145/3357384.3357814](https://doi.org/10.1145/3357384.3357814). [Primary PDF](https://arxiv.org/pdf/1904.08030). Critical location: §3.1 Eq. (3), maximum item–interest inner product; later sections learn the interests. Included for attribution, not as a directly comparable fixed-embedding implementation.

**[R7] Yuxi Zhang, Ji Zhang, Feiyang Xu, Lvying Chen, Bohan Li, Ning Wang, Huawei Tu, Lei Guo, Hongzhi Yin (2026).** *PPA++: Preference Prototype-Aware Learning with Large Language Model for Universal Cross-Domain Recommendation.* Data Science and Engineering 11, 213–229, published 26 January 2026. [DOI and primary full text: 10.1007/s41019-025-00322-w](https://link.springer.com/article/10.1007/s41019-025-00322-w). Critical location: §4.2, Eqs. (5)–(8), positive and negative preference prototypes. Its training/attention and inference function differ from this project.

**[R8] GroupLens (2014).** *Tag Genome Data Set README* and standalone Tag Genome 2014 release. [README](https://files.grouplens.org/datasets/tag-genome/README.html); [official directory](https://files.grouplens.org/datasets/tag-genome/); [release page](https://grouplens.org/datasets/movielens/tag-genome/). Critical sections: Dataset; Usage License; Tag Relevance Data File Structure; Movie Data File Structure. Release corroboration: Jesse Vig, [*Tag Genome Dataset Released*, 2 April 2014](https://grouplens.org/blog/tag-genome-dataset-released/). Archive contents and hashes were inspected directly during preparation.

**[R9] GroupLens (2016 export).** *MovieLens 20M Dataset README.* [Official README](https://files.grouplens.org/datasets/movielens/ml-20m-README.html); [dataset page](https://grouplens.org/datasets/movielens/20m/). Critical sections: Summary; Usage License; Movie IDs; Ratings Data File Structure; Tag Genome. It specifies the rating period through March 2015 and the learned nature of bundled genome data. Dataset attribution requested by GroupLens: F. Maxwell Harper and Joseph A. Konstan (2015), *The MovieLens Datasets: History and Context*, ACM TiiS 5(4), Article 19, [DOI 10.1145/2827872](https://doi.org/10.1145/2827872). The latter is included as dataset attribution; its full text was not needed for a methodological claim here.

**[R10] Jesse Vig, Shilad Sen, John Riedl (2012).** *The Tag Genome: Encoding Community Knowledge to Support Novel Interaction.* ACM Transactions on Interactive Intelligent Systems 2(3), Article 13, 13:1–13:44. [DOI: 10.1145/2362394.2362395](https://doi.org/10.1145/2362394.2362395). [Primary PDF](https://files.grouplens.org/papers/tag_genome.pdf). Relevant locations: §§1.3–1.4 and §3, feature construction/provenance. Supports treating these as learned community features rather than raw independent item content.

**[R11] Yitong Ji, Aixin Sun, Jie Zhang, Chenliang Li (2022 version).** *A Critical Study on Data Leakage in Recommender System Offline Evaluation.* arXiv:2010.11060v4, 30 October 2022; ACM TOIS [DOI: 10.1145/3569930](https://doi.org/10.1145/3569930). [Primary full text](https://arxiv.org/html/2010.11060v4). Critical locations: §2 and Table 2 on global/local timelines; §3 evaluation design. Its experiments concern implicit recommendation; it informs leakage precautions rather than validating this explicit-rating protocol.

**[R12] SciPy contributors (documentation accessed 29 October 2026).** *scipy.cluster.hierarchy.linkage.* [Official documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html). Critical location: Notes, requirement that Ward use Euclidean geometry. The page returned version 1.18.0 documentation during research; this does not assert that version is installed or required on the target Mac.

**[R13] Yoshifumi Kusunoki, Tomoharu Nakashima (2025).** *Revised Margin-Maximization Method for Nearest Prototype Classifier Learning.* JACIII 29(6), 1565–1576, published 20 November 2025. [DOI: 10.20965/jaciii.2025.p1565](https://doi.org/10.20965/jaciii.2025.p1565). [Publisher page](https://www.fujipress.jp/jaciii/jc/jacii002900061565/). **Access limitation:** publisher abstract/metadata only; full PDF not verified. Used only to flag continuing prior work, not to import a theorem or benchmark.

---

**Execution completion rule:** finish with the strongest honest conclusion supported by the stages reached. Do not manufacture success, silently weaken the task, or leave the project waiting for confirmation about ordinary reversible implementation choices. If an indispensable resource is blocked, identify it precisely, finish unaffected work, preserve evidence, and state what was not validated.
