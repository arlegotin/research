# Execution brief: finite-data certificates of stochastic postprocessing incompatibility

**Recipient:** an autonomous coding agent configured as `gpt-6.1-sol` with `xhigh` reasoning.  
**Research and design date:** 29 October 2026.  
**Target experimental machine:** one MacBook with an M3 Max; inspect its actual configuration before allocating work.  
**Deliverable you must produce:** an executed, reproducible research result following the evidence gates below. This document is the execution instruction, not a report of a completed PoC.

## 1. Project decision and the contribution to test

Implement and evaluate **a finite-sample lower-confidence certificate for the directed stochastic simulation deficiency between two unknown three-state channels**. Learn a complete decision-game witness on training counts, freeze it, and evaluate it on independent validation counts. Compare it with an optimized biased binary witness using identical data and calibration, and with an all-data confidence-region relaxation of the usual stochastic-factorization problem.

The primary research question is:

> On the precisely specified mixture of structured and general three-state alternatives in Section 7, at 262,144 observations per channel/input cell and a per-instance false-positive probability of at most 0.05, does the complete learned witness increase detection probability by at least **10 percentage points over each of the two strong references**, while retaining valid uncertainty control?

There are six channel/input cells, so this is **1,572,864 simulated observations per comparison**, including training. The effect threshold means at least 100 additional incompatibilities detected per 1,000 comparisons at the same observation cost. This is a project decision threshold, not a claim about a particular laboratory's acceptable shot budget. It was chosen to demand a material improvement; it must not be changed to fit the results. Additional shot budgets describe the operating range but do not replace this primary endpoint.

### 1.1 Why this is the selected project

The supplied paper's main mathematical construction survives the planning audit. However, neither channel postprocessing, its linear-programming test, nor decision-game witnesses are new mechanisms. In particular, optimized biased binary distinguishability already detects every nondivisible pair with an invertible square classical source channel. A comparison only against eigenvalues, entropy, or equal-prior distinguishability would exaggerate the opportunity. Established comparison theory and the relevant divisibility results are discussed in [R1–R5].

The smallest defensible contribution is therefore **a carefully delimited finite-data comparative characterization**, supported by a transparent composite-null certificate, exact small oracles, and explicit numerical accounting. A positive result would show that this particular complete-witness procedure offers useful statistical power beyond two credible references in the tested observation model. An informative negative would show that the additional decision-game freedom does not deliver the specified gain, or that the supplied weak example requires too many observations for the intended use. Historical priority of the specific statistical construction remains provisional; do not describe it as a newly discovered principle.

The mechanism under test is the additional freedom in a three-action reward table relative to a biased two-action discrimination game. It can increase the population witness margin, but may also increase its sampling range, make witness learning unstable, or consume too much training data. Those competing effects make the question falsifiable. The advantage should disappear when a binary game is already optimal, when all methods have saturated power, when training selects a poor witness, or when the signal is below the observation budget. These are expected possibilities, not reasons to remove instances.

### 1.2 Deliberate revision of the original ambition

The original practical purpose—detecting an impossible system-only intermediate transformation despite innocuous scalar diagnostics—is preserved. The PoC will **certify postprocessing incompatibility from two endpoint-channel datasets**. It will not infer the direction of time from unordered data, measure elapsed time, derive thermodynamics, identify a microscopic environmental mechanism, quantify the memory needed for reversal, or implement a general quantum/process-tensor experiment.

Two endpoint channels provide insufficient information to establish operational Markovianity. A compatible factorization means a memoryless realization is possible for those marginals; it does not establish that the actual process has no memory. An incompatible factorization excludes that explanation under the stipulated preparation, stationarity, and sampling assumptions. Drift, preparation dependence, leakage, or different experimental controls can also violate those assumptions. The operational distinction is established in [R6–R7]. This narrowing is necessary for valid interpretation and fair comparisons, rather than a claim that a classical toy solves the quantum hardware problem.

Do not add a quantum simulation, hardware campaign, record-storage optimization, or broad algorithm search. If the selected study yields a valid negative or an analytic limitation, finish that result.

### 1.3 Fixed decisions and permitted engineering discretion

**Fixed:** the estimand, sampling model, four reported methods, 20/80 training/validation split, primary observation count, significance level, primary effect threshold, generators and acceptance rules, independent development/confirmation streams, outcome rules, and cumulative resource ceilings.

**You may choose:** straightforward array layouts, script organization within the interfaces below, serialization details, and ordinary repairs that preserve the mathematics. You may use fewer secondary runs according to the resource rules. A change to a fixed scientific choice is a protocol deviation: preserve the old results, label the changed work exploratory, and do not reuse confirmation data as fresh evidence.

Proceed autonomously through authorized gates. Ask for help only for an indispensable missing resource that cannot be handled by the stated fallback. Do not stop after writing code or proposing commands: execute the applicable stages, inspect the evidence, and produce the appropriate completed outcome.

## 2. Evidence already established, and what remains untested

This planning work used three parallel reviewers for source mathematics, quantum/operational literature, and finite-sample design. The lead reconciled their findings and checked decision-relevant primary sources. This was not external peer review. Small exact and numerical probes were performed in a hosted **Linux x86-64 environment**, with Python 3.12.14, NumPy 2.3.5, and SciPy 1.17.0 available. No M3 Max timing, Monte Carlo power study, quantum hardware experiment, or completed PoC was performed.

| Central claim | Evidence status and location | Remaining uncertainty / resolving action |
|---|---|---|
| The three-state family contains every standard example of the stated form | Source proof: [S1], §§1–3, Theorem “Fixed three-state obstruction,” equations labeled `eq:AB`, `eq:keyfactor`, `eq:order-test`; independent small exact checks passed | Reproduce the tiny audit, not the entire source workload |
| No finite universal list of separate weak monotones is complete | Source theorem with an explicit finite obstruction | Does not imply infinitely many measured parameters or an algorithmic advantage |
| Directed deficiency is computable by a small LP | Established comparison quantity [R1–R3]; explicit derivation in §4 | Verify independent primal/dual implementations and normalization |
| The advertised signed quotient has entries −1/180 | Exact rational planning check; §3.3 | Reproduce exactly |
| Its actual deficiency is 1/810 | New planning derivation, with exact primal and dual certificates in §3.3 | Reproduce independently; this is not an empirical discovery |
| Its best biased binary witness has gap 1/900 | Eight-LP planning probe found this optimum; an exact attaining witness is given below | Verify the upper optimum using the binary LP; the attaining value alone is a rigorous lower bound |
| A learned witness admits the stated one-sided held-out certificate without a 27-way correction | Derivation in §4.3, independently reviewed; concentration principles are established [R8–R9] | Validate implementation, independence, and numerical evaluation |
| Complete witnesses deliver a material finite-data advantage | **Hypothesis, untested** | Locked comparison in §§7–9 |
| The endpoint data identify hidden environmental memory or a time arrow | **Unsupported as stated** | Excluded; reproduce the identifiability counterexample in §3.5 |
| The full study fits the laptop budgets | Planning estimate only | Inspect the Mac and run the resource calibration gate |

### 2.1 Supplied material and audit provenance

The original inputs are `Pasted text(20261029-162743).txt`, `research_note(2).html`, and `research_reproducibility(2).zip`. The archive expands under `research_reproducibility/`. Retain immutable copies in `sources/` if supplied. The externally attached HTML and the archive's `research_note.html` are byte-identical, with SHA-256:

`96227e3573cac07474ed95b3dcc529824c7cc13be1fd4c33774d6bbeb2b3986c`

The HTML has readable MathML with TeX annotations. Its mathematical source is `research_note.tex`; no ambiguous figure extraction was needed. All supplied mathematical documents, scripts, and saved result files were examined during planning, including the following.

| Archive member | Relevance and disposition |
|---|---|
| `README.txt` | Execution/provenance description; distinguishes proof from finite checks |
| `research_note.html`, `research_note.tex` | Consolidated mathematical source; §§1–7 and 10–12 bear on the proposed application |
| `quantum_pauli_order_notes.txt`, `quantum_pauli_order_verify.py` | Classical and Pauli constructions, novelty qualifications, continuous-time caveats; quantum branch excluded from the benchmark |
| `stochastic_audit_order.py`, `stochastic_audit_order_results.json` | Independent generic rational-inverse audit and source-reported results |
| `records_research_note.txt`, `records_verify.py` | Exact retained-record cardinality results, including extensions beyond the consolidated paper; separate problem, excluded from implementation |
| `stochastic_proof_note.txt`, `stochastic_recovery.py`, `stochastic_recovery_results.json` | Square-alphabet recovery rigidity and numerical checks; separate problem, excluded from the primary experiment |

The planning audit actually reran 616 ordered-pair checks for each of the classical and Pauli families, at `n = 2,3,4,5,10`; 216 classical comparisons using generic rational inversion, at `n = 2,3,4,5`; the record verifier's 9,408 flow comparisons, 13,955 upper-bound checks, and 68 integer minima; and 12 numerical recovery-boundary cases. All passed. The source's larger totals—52,216 family comparisons, 3,116 independent inverse comparisons, 1,750 random recovery channels, 350 histories, and 1,127 interpolation checks—remain **source-reported**, rather than rerun totals. None is evidence about detection power on the proposed finite-data problem.

Two later generator-only feasibility probes, without observation counts or power testing, examined 100 and 200 raw proposals per family. Family S accepted 89/100 and 163/200; Family G accepted 23/100 and 37/200. These hosted-environment exploratory checks support proceeding to target calibration; they are not locked evaluation data, certified final truth, or Mac performance measurements.

The full definitions and algorithms needed for execution are contained here. If the archive is unavailable on the Mac, implement the stated exact oracles and continue; mark source-script reproduction unavailable. Do not claim to have rerun unavailable files. Missing optional source copies do not block the statistical study.

## 3. Necessary mathematics and scope

### 3.1 Channels, order, distance, and deficiency

All arrays use **columns as inputs**. A channel (A\in\mathbb R^{3\times3}) satisfies (A_{ji}\ge0) and \(\sum_j A_{ji}=1\). The observation \(j\) is conditional on prepared input \(i\). Define

\[
A\preceq B\quad\Longleftrightarrow\quad
B=QA\text{ for some }Q\in\mathrm{St}_3,
\qquad
\mathrm{St}_3=\{Q\ge0:\mathbf1^TQ=\mathbf1^T\}.
\]

This order points toward further processing, opposite the frequent “more informative is larger” Blackwell convention. It is a preorder on all labeled channels; output permutations may give equivalent channels.

For equal-sized channels,

\[
D(A,B)=\max_i\frac12\sum_j|A_{ji}-B_{ji}|,
\qquad
\Delta(A,B)=\min_{Q\in\mathrm{St}_3}D(B,QA).
\tag{1}
\]

Here \(D\) is maximum-column total variation, **not** the entrywise sum, Frobenius norm, or spectral norm. The optimization is directed. Compactness gives a minimum; \(\Delta=0\) exactly when the factorization exists. The TV normalization is half the \(\ell_1\) norm throughout.

For any channels \(A,A',B,B'\), contraction of TV under stochastic processing gives

\[
|\Delta(A,B)-\Delta(A',B')|
\le D(A,A')+D(B,B').
\tag{2}
\]

This provides a stable uncertainty bound without inversion. In contrast, \(BA^{-1}\) is unique only for invertible \(A\), and can amplify estimation error near singularity. A pseudoinverse is not a valid substitute for optimization over all admissible \(Q\).

### 3.2 What the supplied theorem establishes

The source family is

\[
P(u,v)=\begin{pmatrix}
(1+2u)/3&(1-u)/3&(1-u)/3\\
(1-u)/3&(2+u+3v)/6&(2+u-3v)/6\\
(1-u)/3&(2+u-3v)/6&(2+u+3v)/6
\end{pmatrix},
\tag{3}
\]

with \(0<u<1\), \(0<v<(2+u)/3\). These channels are positive, symmetric, doubly stochastic, and invertible. Their common eigenvectors are \((1,1,1),(-2,1,1),(0,1,-1)\), with eigenvalues \(1,u,v\). Composition multiplies the two parameters. Therefore

\[
P(u,v)\preceq P(U,V)
\iff U/u\le1\ \text{and}\ 3V/v\le2+U/u.
\tag{4}
\]

The test covers every stochastic postprocessor because invertibility forces the quotient, rather than because postprocessors were restricted to this family.

For \(n\ge2\), let \(d=1/(20n),q=1-d,c=1-d^2\) and

\[
A_i=P(q^{4i}/4,q^i/4),\qquad
B_i=P(cq^{4i}/6,2q^i/9),\quad 0\le i<n.
\tag{5}
\]

Their only nonidentity comparisons are \(A_i\preceq B_j\) for \(i\ne j\). With \(z=q^{j-i}\), the critical quotient inequality is

\[
3+cz^4-4z=(z-1)^2(z^2+2z+3)-d^2z^4.
\]

One monotone cannot reject two failed diagonal pairs without contradicting the allowed cross pairs. Thus no fixed finite family of separate scalar scores, compared componentwise, is complete. Countably many continuous scores of the form \(f_T(A)=\min_Q D(T,QA)\), for a dense countable target set, suffice. This existence proof does not give an efficient finite score list, an optimal truncation, or a data-efficient witness search.

The source also proves a conservative false-positive deficiency lower bound \(1/[25200(k+1)^2]\) for any \(k\) monotones. The separation shrinks quadratically in this construction. With ordinary square-root sampling error, resolving such a shrinking margin can demand quartic growth in sample count; this is a scaling observation, not a minimax sample-complexity theorem.

The scope qualifications are material:

- The family itself has only two parameters, and pairwise comparison uses (4). The theorem is not an impossibility of finite measurements, pair-dependent optimization, or finite approximation at fixed tolerance.
- For the restricted common-eigenbasis continuous-time generator relation, the source's §7 gives two complete scores: \(-\log u\) and \(-3\log v+\log u\). Do not transfer the discrete postprocessing obstruction to that relation.
- The fixed-qubit Pauli construction in §10 uses Bloch multipliers \((u,v,v)\), with CP conditions \(u\le1,2v\le1+u\). It is a separate quantum extension; it does not turn the classical LP or its confidence proof into a quantum theorem.
- The retained-record and square-alphabet recovery theorems in §§8–9 concern different constraints. Their proofs do not identify a hidden environment from endpoint data or supply the proposed statistical advantage.

### 3.3 Exact anchor: quotient negativity and operational distance differ

For the requested example,

\[
A=P(1/4,1/4),\quad B=P(1/6,9/40),\quad
BA^{-1}=\begin{pmatrix}
7/9&1/9&1/9\\
1/9&161/180&-1/180\\
1/9&-1/180&161/180
\end{pmatrix}.
\tag{6}
\]

Both nontrivial eigenvalues decrease, and the quotient is inadmissible. More strongly, its induced contraction on zero-sum vectors is

\[
\frac12\max_{j,k}\|Qe_j-Qe_k\|_1=9/10.
\]

Consequently all equal-prior two-state TV distinguishabilities decrease from \(A\) to \(B\), despite failed stochastic divisibility. This does not defeat generalized biased distinguishability.

The operational deficiency is **exactly**

\[
\Delta(A,B)=1/810.
\tag{7}
\]

An attaining stochastic matrix and its residual are

\[
Q_* =\begin{pmatrix}
211/270&89/810&89/810\\
59/540&721/810&0\\
59/540&0&721/810
\end{pmatrix},\qquad
B-Q_*A=\begin{pmatrix}
-1/810&0&0\\
1/1620&1/810&-1/810\\
1/1620&-1/810&1/810
\end{pmatrix}.
\]

A matching feasible dual reward is

\[
H_* =\begin{pmatrix}
0&1/3&1/3\\
1/9&4/9&0\\
1/9&0&4/9
\end{pmatrix},\qquad w=(1/9,4/9,4/9).
\tag{8}
\]

Its dual objective in (10) is exactly \(1/810\). Verify these equalities with `fractions.Fraction`, independently of the optimizer.

There is also a direct lower-bound check. Let \(C=B-QA\), \(t=D(B,QA)\), and \(J\) be the all-ones matrix. Since \(A^{-1}=4I-J\), the nonnegativity of \(Q_{23},Q_{32}\) implies

\[
\operatorname{rowSum}(C_2)-4C_{23}\ge1/180,
\qquad
\operatorname{rowSum}(C_3)-4C_{32}\ge1/180.
\]

Column sums of \(C\) vanish. Adding gives
\(T=-(C_{11}+C_{12}+C_{13})-4(C_{23}+C_{32})\ge1/90\).
The coefficient ranges in the three columns are \(1,4,4\). A weighted sum of a zero-sum vector is at most its TV norm times the coefficient range, so \(T\le9t\), hence \(t\ge1/810\).

For an elementary biased binary witness, use prior \(3/5\) on input \(e_3\) and prior \(2/5\) on input \((e_1+e_2)/2\). Then

\[
x=(-1,-1,3)^T/5,
\quad Ax=(0,0,1/5)^T,
\quad Bx=(1/45,-1/900,161/900)^T.
\]

The optimal guessing success rises from \(3/5\) to \(541/900\), a gain of \(1/900\). The optimized binary LP also returned this optimum in the planning probe. Thus even the exact anchor already has a strong established detector [R5].

### 3.4 A necessary observation-cost bound already derived

Use the valid null \(A_0=A\), \(B_0=P(1/6,2/9)\), with stochastic quotient \(P(2/3,8/9)\). Define, with natural logarithms,

\[
d_0=\frac{341}{720}\log\frac{341/720}{17/36}
+\frac{179}{720}\log\frac{179/720}{1/4}
\approx5.90567325847\times10^{-6}.
\]

For \(n\) independent shots in every one of the six cells, the KL divergence between the full alternative and null data laws is \(2nd_0\). Data processing to the binary rejection event implies that any test with null rejection probability at most 0.05 and alternative power at least 0.8 must satisfy

\[
2nd_0\ge \operatorname{kl}(0.8\|0.05)
=1.90644205418.
\tag{9}
\]

Therefore \(n\ge161,408\), or **at least 968,448 balanced total shots**, is necessary. This is not an achievable guarantee, the closest possible null, an optimal bound, or a statement about adaptive measurement allocation. It is a direct change-of-measure calculation; [R10] gives related general inequalities. Reproduce it at high precision.

Further planning calculations found that the simple global confidence penalty falls below \(1/810\) only near 8.63 million shots per cell. For fixed oracle witnesses and an 80% validation fraction, the corresponding mean-penalty crossover is approximately 1.001 million for (8) and 1.335 million for the binary witness. These ignore witness learning and random fluctuations. Their ratio is \(4/3\), not evidence of a twofold shot saving. Do not report these crossovers as measured power or sufficient sample requirements.

### 3.5 Exact endpoint-identifiability control

Enumerate the following finite processes once; no repeated seeds are needed. Let \(X_0,X_1\) be independent uniform ternary variables and \(X_2=(X_0+X_1)\bmod3\). Every pairwise marginal is uniform independent, so both initial-to-later channels equal \(J/3\) and have zero deficiency. Nevertheless, \(X_2\) depends on \(X_0\) conditional on \(X_1\). An independent triple has the same pairwise channels and is Markovian.

This demonstrates an information limitation of the inputs, not a defect of a particular estimator. Also note that any two valid \(A,B\) can be realized by the joint law
\(p(X_1=j,X_2=k\mid X_0=i)=A_{ji}B_{ki}\), using an auxiliary record of the initial label. This gives the synthetic endpoint pairs a consistent classical realization, but neither identifies an actual environmental mechanism nor minimizes its memory.

## 4. Proposed certificate: exact specification

### 4.1 Observation model and input/output contract

For each prepared input \(i\in\{0,1,2\}\), obtain independent categorical samples from \(A_{\cdot i}\) and \(B_{\cdot i}\), using separate repeated trajectories for the two endpoints. Uses within each cell are iid; cells and training/validation blocks are independent. The preparation and readout labels are known and stable. An environment may retain memory within a trajectory but is reset to the same distribution between repetitions.

The primary method receives only integer count arrays, cell sample sizes, and α. It receives no true probabilities, generator label, inverse, exact eigenbasis, deficiency, or seed capable of reconstructing the truth.

For each cell with total \(n\), use \(n_{\rm tr}=\lfloor n/5\rfloor\) and \(n_{\rm val}=n-n_{\rm tr}\). In simulation draw the two count blocks independently and sum them for full-data methods. Do not reconstruct a split by rounding empirical frequencies. Counts use `int64`; probability calculations use float64.

Return at least: raw lower bound, clipped lower bound, detection boolean, status, chosen witness, validation ranges, numerical enclosure/check information, training objective, solver status/residuals, elapsed seconds, and failure reason. A method failure returns no certificate, never infinity or a positive default.

### 4.2 Population primal and dual

Build the general primal LP with 19 variables \(Q_{jk},s_{ji},t\): minimize \(t\), subject to column-stochastic \(Q\),
\(-s_{ji}\le B_{ji}-\sum_kQ_{jk}A_{ki}\le s_{ji}\), and \(\sum_j s_{ji}\le2t\) for each \(i\). All slacks and \(t\) are nonnegative. This is the independent reference formulation.

For three-output zero-sum residuals only, TV equals maximum absolute entry. Consequently a second, ten-variable primal uses
\(-t\le B_{ji}-\sum_kQ_{jk}A_{ki}\le t\).
Verify equality of these two formulations. Do not generalize the maximum-entry shortcut to four or more outputs.

The complete dual is the following 15-variable LP:

\[
\begin{aligned}
\max_{H,w,z}\quad &\sum_{ji}H_{ji}B_{ji}-\sum_k z_k,\\
&w_i\ge0,\quad \sum_iw_i=1,\quad 0\le H_{ji}\le w_i,\\
&z_k\ge\sum_iH_{ji}A_{ki}\quad\text{for every }j,k.
\end{aligned}
\tag{10}
\]

Eliminating \(z\), write

\[
W_H(A,B)=\langle H,B\rangle-\sum_k\max_j(HA^T)_{jk}.
\tag{11}
\]

Every feasible \(H\) satisfies \(W_H\le\Delta\), and maximizing gives equality. To derive this, express each TV residual as a maximum over rewards in [0,1], mix the input-column maximum with weights \(w\), exchange compact convex min/max, and maximize over each stochastic column of \(Q\). This is the finite decision-game dual of established deficiency theory, not a claim of a new order criterion.

Train by solving (10) on training frequencies. Solver choice is HiGHS dual simplex (`method="highs-ds"`) unless the target build requires a documented equivalent HiGHS choice. Use deterministic ordering and fixed tie handling. Do not use validation counts to choose a reward table or split fraction.

**Witness sanitization:** interpret the finite floating reward entries as exact rational numbers, subtract each column's minimum, and let \(R\) be the sum of the resulting column maxima. If \(R=0\), return the zero witness. Otherwise divide by \(R\); its column maxima then sum to one and supply admissible \(w\). This changes only a column-shift-invariant objective and its scale. Recompute the actual objective after sanitization. For both learned methods, freeze the same training cutoff τ_train = τ_num and return the zero witness if the sanitized training gap is at most that cutoff. Treat a raw column-range sum at most τ_train as zero before normalization, for both methods. The training optimizer affects power; validity requires only a feasible frozen witness, not an exactly optimal one.

### 4.3 Independent validation without unnecessary multiplicity

Enumerate the 27 deterministic postprocessors \(f:\{0,1,2\}\to\{0,1,2\}\) in lexicographic order. For fixed \(H\), define

\[
g_f(A,B)=\sum_{ji}H_{ji}B_{ji}
-\sum_{ki}H_{f(k),i}A_{ki},\qquad W_H=\min_f g_f.
\]

For validation frequencies define \(\widehat g_f\) by the same formula, and

\[
r_{B,i}=\max_j H_{ji}-\min_j H_{ji},\qquad
r_{A,i}(f)=\max_kH_{f(k),i}-\min_kH_{f(k),i},
\]

\[
b_f=\sqrt{\frac{\log(1/\alpha)}2
\sum_i\left(\frac{r_{B,i}^2}{n_{B,i}^{\rm val}}
+\frac{r_{A,i}(f)^2}{n_{A,i}^{\rm val}}\right)}.
\tag{12}
\]

Use α = 0.05 and

\[
L_{\rm full}=\max\{0,\min_f(\widehat g_f-b_f)\}.
\tag{13}
\]

The square is of the oscillation, not the oscillation of squared entries. If a range is zero its contribution is zero. If all ranges vanish, return zero. Empty validation cells are invalid inputs.

**Coverage argument:** conditional on training, choose a deterministic true minimizer \(f_*\) of \(g_f(A,B)\). The empirical minimum in (13), before clipping, is at most \(\widehat g_{f_*}-b_{f_*}\). The independent bounded-summand Hoeffding bound gives probability at most α that this exceeds \(g_{f_*}=W_H\le\Delta\). Clipping at zero preserves a lower bound on the nonnegative deficiency. Thus \(\Pr(L_{\rm full}>\Delta)\le\alpha\). The range formula follows by adding the squared ranges of all independent weighted observations. The usual concentration argument is discussed in [R8].

There is **no α/27 correction**: the lower-bound proof uses the one fixed true minimizer. It does not assert simultaneous coverage of all 27 individual contrasts. Conversely, choosing the maximum of several learned witnesses after seeing validation results, choosing a favorable shot budget after repeated inspection, or reusing validation for tuning is not licensed by this proof. Report each predeclared method/budget separately. This is not a sequential stopping protocol.

### 4.4 Pseudocode

```text
certify_full(train_A, train_B, val_A, val_B, alpha):
    validate shapes, nonnegative integer counts, positive cell sizes
    Atrain, Btrain = column frequencies from training counts
    solve the 15-variable dual LP on Atrain, Btrain
    if solver failed: return NO_CERTIFICATE with recorded failure
    H = exact-rational column-shift-and-normalize(solver.H)
    if H is zero or sanitized_training_gap <= tau_train: return lower_bound=0, detected=false
    best = +infinity
    for f in lexicographic product({0,1,2}, repeat=3):
        g = exact rational contrast of H and validation counts
        V = exact rational sum of squared ranges / cell sizes
        b_upper = conservative numerical upper evaluation of sqrt(log(1/alpha)*V/2)
        best = min(best, conservative_lower(g - b_upper))
    lower_bound = max(0, best)
    detected = lower_bound > frozen_numerical_guard
    return certificate, witness, ranges, diagnostics, timing
```

Initialization, optimization, and the 27-map loop terminate after these finite operations. Do not introduce iterative witness refinement after validation. Any fallback to a zero witness is recorded as a nondetection.

## 5. Required references and ablation

Implement the following four methods. The two strong references determine the primary research verdict; the simple reference supplies context and a correctness cross-check.

| Method ID | Inputs and role | What its result can establish |
|---|---|---|
| `full_split` | Complete dual on 20% training; (13) on 80% validation | Proposed finite-data certificate |
| `binary_split` | Optimized biased binary game; identical split and (13) | Strong known diagnostic; also the targeted restriction of the reward class |
| `region_lp` | All observations; simultaneous exact binomial intervals and 55-variable lifted LP | Strong inverse-free tomography/reference certificate |
| `global_tv` | All observations; deficiency plus global TV uncertainty correction | Simple credible conservative reference |

Spectral contraction and equal-prior TV are exact illustrative controls on the anchor, not competitive statistical baselines. Do not use an uncalibrated inverse-negative-entry rule as the principal reference.

### 5.1 Optimized biased binary witness

For each of the eight sign vectors \(s\in\{-1,+1\}^3\), solve on training data

\[
\max_{x,u,y}\frac12\left(s^T\widehat Bx-\sum_k y_k\right),
\quad u\ge x,\ u\ge-x,\ \sum_i u_i\le1,
\quad y\ge\widehat Ax,\ y\ge-\widehat Ax,
\tag{14}
\]

with \(x\) free and \(u,y\ge0\). Constant sign vectors cannot have positive optimum; they may be handled analytically as zero, while the reference implementation still checks them once. Discard a raw candidate if \(\|x_{\rm raw}\|_1\le\tau_{\rm train}\), before normalization; for nonconstant s this is exactly its raw reward-range sum. For each remaining candidate x, normalize to unit l1 norm and construct its reward table below; then apply the shared sanitization and training-gap cutoff. Find the largest recomputed retained gap, max_gap. Among candidates with gap at least max_gap − τ_train, select the first in lexicographic sign order. If none exceeds the shared cutoff, return zero. The reward construction is

\[
H_{ji}=(|x_i|+s_jx_i)/2,\qquad w_i=|x_i|.
\tag{15}
\]

For nonconstant \(s\), (11) is exactly \(\tfrac12(s^TBx-\|Ax\|_1)\). Sanitize and validate this \(H\) with precisely the same code used by `full_split`. Training sign selection incurs no validation multiplicity penalty. The population optimum is
\(G=\max_{\|x\|_1\le1}(\|Bx\|_1-\|Ax\|_1)/2\le\Delta\).

For invertible \(A\), this criterion is complete for the sign of nondivisibility: if a column of \(BA^{-1}\) has a negative entry, choose \(x=A^{-1}e_k/\|A^{-1}e_k\|_1\). Its output norm increases. The bound \(G\le\Delta\) remains valid when \(A\) is singular, but the completeness argument does not. This distinction prevents claiming a fundamental detection advantage over biased binary tests in the invertible main benchmark.

The comparison isolates reward-class freedom because confidence formula, split, samples, optimizer family, and stopping rules are shared. Neither learned method gets variance-based tuning, extra training restarts, or truth-assisted witness selection in the confirmatory protocol. Maximizing raw training gap need not maximize power; a negative result concerns this fixed procedure.

### 5.2 All-data confidence-region LP

For each of the 18 entries of \(A,B\), make a two-sided Clopper–Pearson interval with noncoverage α/18, equivalently tail probability α/36. Each count is marginally binomial; the union bound remains valid despite within-column multinomial dependence [R11]. Let endpoints be \(\ell^A,u^A,\ell^B,u^B\).

For count \(k\) out of \(n\), the lower endpoint is zero if \(k=0\), otherwise the α/36 quantile of \(\mathrm{Beta}(k,n-k+1)\). The upper endpoint is one if \(k=n\), otherwise the \(1-\alpha/36\) quantile of \(\mathrm{Beta}(k+1,n-k)\). Handle these endpoints explicitly.

First implement the ten-variable interval-propagation LP:

\[
\min t:\quad Q\in\mathrm{St}_3,\quad 0\le t\le1,
\quad\sum_kQ_{jk}\ell^A_{ki}-t\le u^B_{ji},
\quad\sum_kQ_{jk}u^A_{ki}+t\ge\ell^B_{ji}.
\tag{16}
\]

It is a lower bound, not exact joint confidence-set feasibility. Then use the following stronger **55-variable LP** as the reported `region_lp`: variables \(Q\) (9), \(A'\) (9), \(B'\) (9), \(Z\) (27), and \(t\) (1), all within [0,1]. Require stochastic \(Q,A',B'\), the confidence boxes for \(A',B'\), and for every \(j,k,i\), with \(\ell=\ell^A_{ki},u=u^A_{ki}\),

\[
\begin{aligned}
Z_{jki}&\ge\ell Q_{jk}, &Z_{jki}&\ge A'_{ki}+uQ_{jk}-u,\\
Z_{jki}&\le uQ_{jk}, &Z_{jki}&\le A'_{ki}+\ell Q_{jk}-\ell.
\end{aligned}
\]

Also require \(\sum_jZ_{jki}=A'_{ki}\), and
\(-t\le B'_{ji}-\sum_k Z_{jki}\le t\); minimize \(t\). These are the elementary bilinear box-envelope inequalities often called the McCormick relaxation [R12]. The actual products \(Z_{jki}=Q_{jk}A_{ki}\) are feasible whenever the confidence boxes cover. Hence the optimum is at most the true deficiency. At zero-width boxes the products are exact and this reduces to the three-output primal.

Joint optimization over unknown \(A,B,Q\) is bilinear. Do not describe either relaxation as a globally exact solution to that problem. The lifted LP dominates (16); verify this in calibration. It is the prespecified strengthening, so there is no result-dependent decision to include it. A small deterministic expected-count planning probe found both relaxations equal on the anchor; this was not a sampled power comparison.

**Critical numerical rule:** a minimization LP's primal objective is not itself a safe lower certificate. Use a dual lower bound with residual correction. For
\(\min c^Tx\) subject to \(Mx\le b,Ex=e,\ell\le x\le u\), choose any \(\lambda\ge0\) and any \(\nu\), and put \(r=c+M^T\lambda+E^T\nu\). Then

\[
L_{\rm dual}=-\lambda^Tb-\nu^Te+
\sum_a\min(r_a\ell_a,r_au_a)
\tag{17}
\]

is a lower bound regardless of imperfect dual feasibility. With SciPy's usual sensitivity convention, start with \(\lambda=\max(0,-\texttt{ineqlin.marginals})\), \(\nu=-\texttt{eqlin.marginals}\); verify the signs on exact fixtures. Evaluate (17) using exact rational conversion of finite coefficients/multipliers, or a checked outward enclosure. Record both primal and lower bounds. Return \(\max(0,L_{\rm dual})\) after numerical safeguards.

### 5.3 Simple all-data TV reference

For a ternary empirical column from \(n_i\) samples, a standard bound is
\(\Pr(\mathrm{TV}(\widehat p,p)>r)\le6e^{-2n_ir^2}\) [R13]. Allocate failure α/6 per channel/input column:

\[
r_i=\min\left\{1,\sqrt{\frac{\log(36/\alpha)}{2n_i}}\right\}.
\]

Use a certified lower bound \(\Delta^-\) for the empirical deficiency and report
\(L_{\rm global}=\max(0,\Delta^- -\max_i r_{A,i}-\max_i r_{B,i})\).
Obtain \(\Delta^-\) by evaluating a sanitized empirical dual witness, or by (17), not by trusting a primal objective. This reference is deliberately conservative but fully specified and uses all observations.

### 5.4 Important methods outside the comparison

Process-tensor tomography accesses interventions and multitime information absent from these inputs. General quantum confidence-region methods also solve a different model [R6–R7,R14]. Do not compare their published shot counts or runtimes to this benchmark. Exact globally optimized composite-likelihood inference, sequential designs, optimal preparation allocation, and variance-adaptive witness learning are excluded. Accordingly the claim is about the specified finite-sample procedures, not the best possible diagnostic or optimal sample complexity.

## 6. Numerical validity and independent correctness

### 6.1 Precision policy

Use CPU float64 for LPs and array operations. No GPU, Metal, CUDA, PyTorch, quantum simulation package, or acceleration layer is needed. The largest optimization has 55 variables. Exact rational arithmetic is for tiny oracles, sanitized witness coefficients, and final linear-certificate evaluation, rather than for solving all LPs.

Use `Fraction` to regard each returned binary floating value as an exact finite coefficient when checking feasibility and linear certificates. For (12), evaluate logs/square roots with at least 50 decimal digits, round the radius upward, and the resulting lower bound downward. Python's `decimal` plus explicit neighboring representable values is sufficient for the fixed α=1/20 and rational range sum; verify the enclosure against 80-digit evaluation. If using another small high-precision library, record the version and its rounding limitations.

Confidence-interval endpoints are a separate numerical issue. Start with the official SciPy exact interval or beta-quantile routines [R11]. Move endpoints outward by the larger of \(10^{-12}\) and 32 float64 ulps, clip to [0,1], and verify monotonicity. A fixed padding and agreement on selected fixtures are not proofs of production coverage. Before a positive region certificate is returned, **every interval actually used** must pass a conservative tail check: for an interior lower endpoint ell, an upper enclosure of \(\Pr_{\ell}\{\mathrm{Bin}(n,\ell)\ge k\}\) must be at most the allocated tail probability; for an interior upper endpoint u, an upper enclosure of \(\Pr_u\{\mathrm{Bin}(n,u)\le k\}\) must be at most that probability. For the region method the tail probability is the exact rational α/36 = 1/720. Use the corresponding allocation when validating the final analysis intervals in (18), rather than reusing α/36 there.

A self-contained bounded implementation of that check is:

1. Treat each proposed finite endpoint as an exact number. Use 60-digit Decimal intervals, with ROUND_FLOOR/ROUND_CEILING for ordinary arithmetic. Trap invalid operations, NaNs, and underflow that could turn a positive upper enclosure into zero. Decimal ln and exp use correctly rounded HALF_EVEN semantics: apply them to interval endpoints and then take next_minus/next_plus; merely changing the context rounding mode is insufficient [R21]. Enclose a constant π once using Machin's identity \(\pi=16\arctan(1/5)-4\arctan(1/239)\), alternating power series, and the first omitted term.
2. Enclose \(t_k=\binom nk p^k(1-p)^{n-k}\) through its log. Use exact integer factorials for arguments at most 64. For larger integer m, use
   \[
   S(m)=(m+1/2)\log m-m+\tfrac12\log(2\pi)+1/(12m)-1/(360m^3),
   \quad S(m)<\log(m!)<S(m)+1/(1260m^5).
   \]
   This positive-real remainder bound follows from Stirling's series [R20]; handle 0! and p=0,1 directly. Use interval subtraction for the log binomial coefficient so cancellation cannot lose the enclosure.
3. For the upper confidence endpoint, p is above k/n. Sum the lower tail starting at k and moving downward using successive-mass ratio \(r_j=j(1-p)/[(n-j+1)p]\). The ratios decrease as j decreases. For the lower endpoint, sum the upper tail from k upward with \(r_j=(n-j)p/[(j+1)(1-p)]\), which decreases as j increases. Verify a conservative ratio bound below one before using the remainder bound. If the current mass is t and the next-ratio upper bound is r<1, the remaining tail is at most \(t r/(1-r)\). Partial sums give a lower bound; partial sums plus that remainder give an upper bound.
4. Stop when the tail upper bound proves validity, or the lower bound proves that the endpoint must move outward. If the enclosure straddles the target, extend the sum or double the outward endpoint displacement. Cache verified (k,n,tail-probability) results. Bound work by one second per endpoint, 20,000 recurrence steps per check, and absolute added endpoint displacement \(10^{-8}\). The time, recurrence-step, and displacement limits are cumulative across that endpoint's retries; displacement is measured from its initial proposed value. These are numerical safeguards far below the benchmark effect scale, not altered statistical confidence levels. An unresolved endpoint returns a numerical failure, never a guessed certificate.

Validate the checker against exact rational binomial tails for small n and at least 20 difficult large-n fixtures, including k=0,1,floor(n/3),n−1,n at several n. Its implementation and speed are **not already verified**; this is an explicit tiny-correctness/calibration task. Do not evaluate a naive full n-term tail or an unbounded hypergeometric series at the largest n.

It is safe to skip endpoint certification only when the method returns an exact zero lower bound regardless of the preliminary LP; label such intervals unverified. If verification widens any endpoint, rebuild the LP or reevaluate (17) with the old multipliers and **all final widened coefficients and bounds**. The lower bound for the narrower unverified boxes cannot be retained. Re-solving once is permitted and counts against the LP/time caps. Validate all intervals used for the final primary inference, even when their associated effect is unfavorable.

If valid outward brackets cannot be produced within the existing repair/resource allowance, mark the region comparator blocked and the required comparative conclusion inconclusive. Do not quietly replace it by a weaker interval method and retain a superiority claim. The formal coverage statements concern the mathematical algorithms; implementation validity is supported by these explicit enclosures and checks, not by a solver-success flag.

LP coefficients formed by arithmetic, such as confidence endpoints and their products in the relaxation, must correspond to the coefficients being certified. Use exact finite endpoints to form the definitive linear constraints for (17), even if a rounded float copy is supplied to the solver. A dual bound from those definitive coefficients remains valid for any returned multipliers. Stochastic rational empirical frequencies can be supplied independently to the final certificate evaluation.

Calibrate a **numerical decision guard** before the pilot. Let \(e_{\max}\) be the largest discrepancy among exact/independent solutions and the certificate's numerical enclosures on the prescribed tiny tests. Set
\(\tau_{\rm num}=\max(10^{-10},20e_{\max})\).
Require \(\tau_{\rm num}\le10^{-7}\), three orders of magnitude below the smallest main-benchmark deficiency \(10^{-3}\). Otherwise repair or stop. This factor is a conservative implementation guard, not a substitute for the confidence proof. Freeze it in the protocol. Detect only when the conservative lower bound exceeds this guard; borderline cases are nondetections.

**Population upper bounds and canonical channels:** a raw solver primal objective is not a certified upper bound if its postprocessor is infeasible. Clip a returned Q's negative entries to zero, normalize each column exactly, substitute a uniform column if its sum is zero, and conservatively recompute \(D(B,QA)\). This feasible value is the population upper bound; use a sanitized dual witness for its lower bound. Retain the solver objective only as a diagnostic. All generation, truth evaluation, and sampling must use the same stored canonical probability columns: normalize once, store their exact finite/rational representation and the float64 sampler copy, and bound any conversion discrepancy below \(\tau_{\rm num}/100\). Use that discrepancy in truth brackets and reject invalid negative or zero-sum columns. Do not normalize truth and sampled data independently in different modules. The simulation is ordinary numerically audited Monte Carlo, not a formal proof of a pseudorandom generator's distribution.

For routine solver residual targets, start with primal/dual feasibility tolerances \(10^{-9}\) and require a verified population primal–dual gap at most \(10^{-8}\) on main-instance truth. Do not demand unsupported HiGHS tolerances. Tighten only to resolve a demonstrated error, then rerun the affected checks. Record solver warnings, presolve, actual tolerances, versions, and failure statuses [R15].

### 6.2 Tiny checks required before any power interpretation

| Check | Independent expected result / error it catches |
|---|---|
| Anchor fractions (6)–(8) and direct lower bound | Exact −1/180 quotient, 1/810 deficiency; catches norm and multiplication-order errors |
| Binary anchor witness and eight-sign optimum | Exact attaining gain 1/900; catches factor-of-two and biased-prior mistakes |
| \(A=I\), arbitrary stochastic \(B\) | \(Q=B,\Delta=0\); \(B=A\) likewise |
| Construct \(B=QA\) from independent stochastic \(Q,A\) | Deficiency zero, including boundary \(Q\); catches false rejection of valid compositions |
| \(A=J/3,B=I\) | Δ=2/3 by optimal constant guessing; catches singular-source failure |
| Independent 19-variable and 10-variable primal versus 15-variable dual | Agreement on at least 20 tiny random pairs; no shared constraint builder for all three |
| At least two positive pairs checked by a second HiGHS algorithm | Independent numerical route; exact fixtures remain the real oracle |
| Output/input permutations | Common input relabeling and independent output relabelings preserve deficiency and witness values after transformation |
| 27 deterministic maps versus direct support function | Exact equality of (11) and the minimum contrast; catches transpose/index errors |
| Zero-width confidence boxes | Both region LPs equal population deficiency; lifted bound never weaker than basic bound beyond tolerance |
| Safe numerical lower bound | (17) stays below an independently verified optimum even after deliberately perturbing dual multipliers |
| Holdout discipline | Changing validation counts leaves the trained witness byte-for-byte unchanged; no truth object reaches a method |
| High-precision interval endpoints | Outward rounding, tails, endpoint cases, no NaNs/empty stochastic boxes |
| Exact multitime counterexample | Both endpoint channels \(J/3\), yet different Markov status; catches interpretation overreach |

Also perform an exhaustive conditional-coverage unit check: for fixed rational witnesses (include \(H=I/3\)) and null \(A=B=J/3\), enumerate the \(3^6=729\) validation outcome patterns with one observation per cell. Each pattern has exact probability \(1/729\). Evaluate the certificate at production α=0.05 and, solely for this unit check, α=0.5. At α=0.05 this tiny case is vacuous: the certificate cannot become positive. The α=0.5 check must include positive certificates (for example, empirical A with identical output columns and empirical B equal to I), while their total exact null probability remains at most 0.5. This catches formula/branch errors without an LP per outcome; it does not establish universal coverage empirically. The displayed positive outcome must equal \(2/3-\sqrt{\log(2)/3}>0\), which also prevents an always-zero implementation from passing. The production protocol always uses α=0.05. Enumerating unused training outcomes for an unlearned witness adds no information and is not required.

## 7. Data generation and locked experiment

### 7.1 Data provenance and unit of evaluation

No private dataset or hardware access is required. All data are deterministically reproducible synthetic categorical counts from known finite channels. The generators isolate statistical detection of a precise mechanism; they are not fitted device-noise models and do not justify generalization to quantum processors.

One independent evaluation unit is **one channel pair drawn from the stipulated distribution plus its independently sampled count blocks**. The same pair and counts are supplied to all methods. At different observation budgets, use separately seeded independent count blocks; a pair is reused only to describe its response to budget. Do not treat those budgets as independent channel instances in pooled inference.

Use `numpy.random.SeedSequence` and store spawned seed identifiers. Suggested root integers are `2026102901` for development, `2026102902` for pilot, `2026102903` for confirmation, and `2026102904` for stress tests. Derive child streams from stage, pair ID, budget, channel, input, and training/validation block; avoid Python's process-randomized `hash`. These are fixed reproducibility seeds, not evidence of independence by themselves—the stream construction and sampling model supply that.

### 7.2 Alternative distribution: a fixed 50/50 mixture

Draw a Bernoulli(1/2) family label independently for each main pair. Generate from that family's rejection-conditioned distribution below. Truth is used to define and label the benchmark only, never to train or calibrate a method.

**Family S: structured contractions.** Draw \(u\sim U[0.15,0.60]\), \(v\sim U[0.15,0.60]\), \(r\sim U[0.25,0.85]\), and γ log-uniform on [0.004,0.08]. Put \(s=(2+r)/3+\gamma\), \(A=P(u,v)\), \(B=P(ru,sv)\). Accept only if \(s<1\), both matrices are strictly stochastic, and \(10^{-3}\le\Delta(A,B)\le0.03\). Apply a uniformly selected simultaneous permutation similarity \(A\leftarrow SAS^T,B\leftarrow SBS^T\). These cases retain decreasing spectral parameters while violating (4); their accepted distribution is explicitly conditional, not representative of all noise processes.

**Family G: general positive channels.** Independently draw the columns of \(A\) and \(Q_0\) from Dirichlet(2,2,2), columns of \(C\) from Dirichlet(1,1,1), and λ uniformly on [0,1]. Set \(B=(1-\lambda)Q_0A+\lambda C\). Accept if all entries of \(A,B\) are at least \(10^{-4}\), \(\kappa_2(A)\le100\), and \(10^{-3}\le\Delta(A,B)\le0.03\). The construction is not constrained to be symmetric, commuting, or simultaneously diagonalizable. Retain commutator norm and condition number as descriptive metadata. Do not reject a pair based on a method's power, witness, or observed gain.

The deficiency band covers small but nonzero observable incompatibilities, avoids numerical-boundary labels, and includes the anchor's scale. It is a designed stress distribution, not a prevalence estimate. Its width is fixed before development. Report the actual deficiency, conditioning, entry-minimum, and family distributions so that any concentration of results is visible.

Use certified population primal/dual brackets to decide membership. If a bracket crosses a band boundary, reject that generator draw as a boundary-ambiguous proposal, log it, and count its compute. Do not repeatedly refine only draws that favor a method. For a requested pair, allow at most 500 proposals; cap population truth solves at 20,000 across all stages. If acceptance is below 2% in calibration, or the cap prevents the prescribed main sample, stop with a generator/feasibility finding rather than changing the distribution after looking at performance. Every proposal and rejection reason must be counted; store summary counts and the accepted matrices.

### 7.3 Null and scope controls

Draw null pairs from an equal-weight mixture of four categories, separately from the alternatives:

1. Interior: columns of \(A,Q\) are independent Dirichlet(2,2,2), \(B=QA\).
2. Stochastic boundary: \(A=P(u,v)\) with \(u,v\) uniform on [0.15,0.60]; \(Q=P(r,(2+r)/3)\), \(r\) uniform on [0.25,0.85]; \(B=QA\).
3. Exact relabeling: draw \(A\) as in category 1, and take \(Q\) uniformly among the six permutation matrices.
4. Singular: with equal probability use \(A=J/3\), or draw two independent Dirichlet columns and duplicate the second as the third; draw \(Q\) as in category 1 and set \(B=QA\).

Form products in float64, re-establish column sums within documented tolerance, and retain the generating \(Q\) as a truth certificate. Require the conservatively evaluated generating residual \(D(B,Q_{\rm gen}A)\le\tau_{\rm num}/10\). This bounds actual deficiency below the rejection guard despite roundoff. Repair or reject a construction that fails this condition; do not label a noticeably perturbed pair as an exact null. Exact rational null fixtures supply the stricter correctness gate.

### 7.4 Minimal experiment matrix

| Experiment | Independent units and budgets | Purpose |
|---|---|---|
| Exact controls | Once per deterministic fixture | Mathematical and indexing correctness; not statistical repetitions |
| Development | Up to 100 alternative and 100 null pairs; at most three budgets | Debugging and calibration only; never merged into confirmation |
| Mechanism pilot | 100 alternatives and 100 nulls at the primary budget; 100 anchor repetitions at \(n=2^{20}\) | Verify detectable signals, fair references, and anticipated precision; not the final claim |
| **Primary locked comparison** | **1,000 iid alternative-mixture pairs, one count realization each, \(n=2^{18}\) per cell** | Paired power improvement against both strong references |
| Null calibration | 1,000 iid null-mixture pairs at \(n=2^{10},2^{18},2^{22}\) | False-positive implementation audit across budget and boundary/singularity |
| Budget response, secondary | Same 1,000 alternative pairs at \(n=2^{10},2^{14},2^{20},2^{22}\), fresh counts | Describe operating range; no interpolated sample-efficiency claim from this sparse grid |
| Anchor response, secondary | 500 repetitions per budget at \(n=2^{10},2^{14},2^{18},2^{20},2^{22}\) | Relate exact tiny margin and information lower bound to measured power |
| Conditioning stress, secondary | 100 fresh pairs, two budgets \(2^{18},2^{22}\), as specified below | Separate numerical stability and information loss from special-family behavior |

For the conditioning stress, take 50 accepted Family G pairs and replace \(A,B\) by \(AT_\eta,BT_\eta\), with \(T_\eta=\eta I+(1-\eta)J/3\), using η=0.01 for 25 and η=0.0001 for 25. Use the other 50 pairs with η=0 exactly. Recompute truth; do not assume the previous deficiency is preserved. This changes available information, so plot both the new deficiency and conditioning. At η=0 both channels have constant columns and a valid postprocessor exists. These are secondary controls, not a way to obtain the primary gain by making the binary test incomplete.

All four methods run on every retained dataset unless a predeclared resource reduction removes an entire secondary experiment. Use efficient multinomial counts; never allocate an array with one element per simulated shot. Millions of virtual observations are not a measurement of laboratory throughput.

### 7.5 Freeze and permitted workload reductions

Before generating confirmation counts, save `protocol.lock.json` with generator rules, root seeds, accepted software versions, witness selection, split, α, primary budget/threshold, numerical guard, run counts, and analysis rules. Hash it and commit or otherwise version the implementation. The main accepted channel matrices can then be generated without exposing them to method code; lock their manifest before drawing counts.

Resource reductions are allowed only before confirmation and in this order: remove conditioning stress, reduce anchor repetitions from 500 to 200, remove the two smallest secondary budgets, then remove remaining secondary budget-response experiments. Retain the 1,000 primary alternatives and the 1,000 null pairs at the primary budget. If those core runs cannot fit after calibration, report practical infeasibility or an analytic/pilot result; do not silently lower primary precision or move its observation budget. Null runs at the other two budgets may be dropped only with explicit narrowing of the coverage-audit claim.

## 8. Metrics, uncertainty, and decision rules

### 8.1 Primary inference

The primary metric is the paired difference in detection probability between `full_split` and each strong reference, averaged over the specified random 50/50 alternative mixture at \(n=2^{18}\). Detection means a conservative lower bound greater than the frozen numerical guard.

For each reference \(b\), record \(n_+\) cases where full detects and \(b\) does not, and \(n_-\) cases with the reverse pattern. The difference estimate is \((n_+-n_-)/1000\). To obtain simultaneous conservative 95% intervals for both comparisons, construct two-sided Clopper–Pearson intervals for each of the four marginal probabilities \(p_{+,b},p_{-,b}\), each with noncoverage 0.05/4. Then use

\[
[L_{+,b}-U_{-,b},\ U_{+,b}-L_{-,b}].
\tag{18}
\]

This remains valid despite dependence between the discordance indicators; the union bound supplies joint coverage. The iid mixture generator makes each pair-level indicator identically distributed. Do not pool repeated observations of the fixed anchor or repeated budgets as additional independent channel pairs.

**Supported primary improvement:** both lower endpoints in (18) are at least 0.10; validity gates pass; and the Family G descriptive results do not show a material reversal. For the latter check, compute the same paired difference on Family G and require its point estimate to be nonnegative. This additional prespecified restriction limits the claim; it is not a second opportunity to declare success. Give family-specific uncertainty and do not claim a population-wide guarantee from that point estimate.

**Informative negative against the specified target:** at least one upper endpoint in (18) is below 0.10, with valid implementations, adequate completed primary data, and no material numerical failure. If the full method beats the region LP but the binary reference removes the gain, explicitly state that the primary criterion failed. A smaller, well-estimated gain may be reported as secondary evidence, not renamed success.

**Inconclusive:** neither rule applies, a required reference remains blocked, or precision/validity is insufficient. Do not add seeds until a preferred conclusion appears. The predeclared 1,000 observations of the random comparison experiment are the confirmation sample; further work would require a new protocol.

The 10-percentage-point criterion is intentionally easier to estimate honestly than a continuous “shots to 80% power” crossing on a sparse grid. Do not fit a favorable curve and assert a twofold saving. You may state observed powers and brackets on the tested grid, with censored thresholds if 80% is not reached.

### 8.2 Secondary metrics and audit checks

Report, by method, family, and budget: detection power; false-positive fraction on nulls; nominal lower-bound coverage against population truth; median and distribution of lower-bound magnitude; learning failures; solver/numerical failures; training versus validation time; total method time; peak process memory; and preprocessing cost. Plot results against true deficiency and source condition number. A larger mean lower bound without valid coverage is not an improvement.

Use binomial intervals for fixed-budget power summaries. Family-level summaries are conditional on the sampled family labels; the mixture primary analysis is prespecified. Secondary plots are descriptive unless multiplicity-adjusted claims are explicitly identified. Report all tested budgets and conditions.

Analytic coverage supplies the statistical guarantee; finite simulations are an implementation audit, not proof that the true false-positive rate is exactly 5%. Flag a null-calibration cell for investigation if its one-sided binomial tail under \(p=0.05\) is below \(0.001/M\), where \(M\) is the number of reported method × budget × null-category/pooled audit cells fixed in the lock file. For alternative truth brackets \([\Delta^-,\Delta^+]\), a definite coverage error is \(L>\Delta^+\); \(L\le\Delta^-\) is definitely covered; overlap \(\Delta^-<L\le\Delta^+\) is numerically unresolved. Refine only within the common budget or report unresolved counts and a possible-error upper rate. Apply the same audit to definite coverage errors; do not silently score overlaps as successes or failures. A flag triggers bounded diagnosis, not automatic scientific rejection of the hypothesis.

### 8.3 Failures, missing runs, and timing fairness

Every scheduled dataset gets a row for every method. A timeout, solver failure, invalid interval, or numerical-guard failure is a nondetection for practical power and has its own failure code. Retain the original row when retrying. Only one documented repair/retry per failed configuration is permitted within the shared repair allowance.

For the primary superiority analysis, perform a conservative missingness check: count a failed candidate as nondetection and a failed reference as detection. Require the conclusion to survive it. If any strong method has more than 1% failed primary runs, do not call the comparison a supported positive or informative negative; diagnose or classify it invalid/inconclusive. A failed reference is not evidence that the proposed mechanism is better. For an informative negative, also apply the opposite, candidate-favorable imputation: failed full runs are detections and failed reference runs are nondetections. Require at least one primary upper confidence endpoint to remain below 0.10 under that assignment. Otherwise classify the missingness-contaminated result as inconclusive. A reliability failure may support an engineering limitation, but not a refutation of the mathematical mechanism.

Time boundaries include frequency formation, witness training, interval construction, certificate evaluation, and output-record construction. Record count generation and common truth/preprocessing separately and also report end-to-end totals. Do not charge truth generation to only one method. The binary method pays for all its sign-pattern solves; the full-data reference pays for intervals and lifting. No amortization is claimed.

Report one cold-process smoke timing and warmed timings after five untimed calls per method. Randomize method execution order per dataset using a separate deterministic order stream. Do not change data or fit tolerances to the order. Use `time.perf_counter_ns`; a CPU-only implementation has no device synchronization step. For peak memory use an isolated representative subprocess and macOS `/usr/bin/time -l`, or a documented equivalent with units verified; `tracemalloc` alone misses native allocations. Record temperature/power mode and interruptions when known, rather than interpreting laptop throttling as mathematical complexity.

## 9. Budgets and evidence gates

### 9.1 Inspect the actual Mac first

Record chip/model, operating system, architecture of Python, physical RAM, free memory, free disk, Python/dependency versions, BLAS/HiGHS backend, and thread settings. Suggested read-only commands are `sw_vers`, `uname -m`, `sysctl -n hw.memsize`, `system_profiler SPHardwareDataType`, and `df -h .`; inspect rather than assume their availability or output. Check that Python is native arm64. A Rosetta environment must be labeled and fixed if feasible before timing.

Use one worker initially, with intended numerical-library threading explicitly limited and recorded. No remote compute, CUDA-only dependency, paid experimental computation, or proprietary dataset is authorized. Ordinary package/source downloads are allowed. No speed or memory claim from the hosted planning environment transfers to the laptop.

Set the experiment memory cap to the smaller of **4 GiB and 25% of detected physical RAM**. Allow at most **2 GiB of result/intermediate data**, plus **3 GiB for the environment and downloaded sources**. Require at least 8 GiB free disk before installation; if less, use an existing suitable environment or reduce optional downloads without deleting unrelated user files. Keep counts/results in chunked files and avoid retaining solver histories for every successful LP.

### 9.2 Cumulative ceilings

The **experimental command wall-time cap is six hours** across the entire project, including development probes, generator attempts, calibration, tuning, failures, retries, secondary branches, and analysis commands. Sum active command elapsed times in a persistent ledger; also report actual end-to-end elapsed time. Changing stages does not reset the ledger. Single-worker execution makes this accounting simple.

Hard count ceilings are **20,000 generated count datasets**, **200,000 LP solve attempts** including truth and binary subproblems, and **20,000 logical population-truth bracket evaluations**. A bracket evaluation may call more than one LP; every underlying solver invocation also counts separately toward the 200,000 LP total. These are ceilings, not workloads to fill. Log attempts before starting them so failures count. Stop launching experiments when only the reporting reserve remains.

| Stage | Experimental wall-time allowance | Main question, minimum evidence, decision |
|---|---:|---|
| 0. Constructive feasibility | 10 min | Inspect actual hardware; verify supplied formulas, data model, source availability, and tiny stack; stop only the affected work if access is missing |
| 1. Tiny correctness | 15 min | Implement primal/dual, shared certificate, binary and region references; exact anchor and required invariants pass; any essential mismatch blocks power work |
| 2. Resource calibration | 10 min | Time 100 representative datasets including largest-n intervals and lifted LP; measure memory and generation acceptance; freeze tolerances and feasible secondary scope |
| 3. Mechanism pilot | 30 min | Run the specified pilot; check positive/null controls, reference strength, and likely primary precision; decide continue, bounded repair, or scoped early result |
| 4. Locked primary and null comparison | 180 min | Freeze protocol; complete core confirmation; retain failures and original runs; no method/scientific tuning |
| 5. Justified secondary work | 50 min | Run the permitted budget/anchor/conditioning matrix only if core remains valid and budget forecast fits |
| Shared repair allowance | 35 min | At most two distinct implementation repairs across stages; no new scientific rescue branch |
| 6. Diagnosis and report reserve | 30 min | Recompute essential summaries from raw files, interpret outcome, preserve artifacts; never spend this reserve on more experiments |
| **Total** | **360 min** | All stage allowances and unused portions stay within this single cap |

Allow a solver time limit of five seconds for these tiny LPs and a dataset-method timeout of 30 seconds initially. Calibration may lower these limits; do not raise them above 30 seconds without identifying a concrete abnormal case and spending the repair allowance. A repeated timeout for a 55-variable LP is a diagnostic issue, not an expected scaling regime.

All performance estimates here are unmeasured planning targets. The dense matrices are tiny; dominant costs are repeated solver setup, binomial intervals, conservative arithmetic, and Python orchestration, not storing simulated shots. At fixed dimension the total work scales approximately with dataset count × per-dataset method calls; histogram sampling avoids linear storage in \(n\). Increasing virtual shots can affect RNG and special-function cost, so include the largest \(n\) in calibration.

Project total time from an empty environment is estimated at **45–90 minutes setup, 4–8 hours implementation/debugging, 1–2 hours interpretation, and up to six hours experimental command time**, with some overlap possible. This is an estimate of an autonomous implementation effort, not a promise of one-hour delivery. The informative pilot should require roughly an hour of experimental computation after core implementation; the decisive computation should remain in hours, not days.

### 9.3 Concrete transitions and stop rules

**Gate 0 — constructive feasibility.** Reproduce (6)–(9) and the source-family identities using the smallest implementation. Verify that the observation data needed by the proposed certificate exist in the intended interface. If only one uncontrolled output trajectory is available, the prescribed channel/count model is not met; do not apply the test. For this synthetic study, data access is fully constructive. A missing Mac blocks target-device measurement but not mathematical correctness and code preparation; complete those unaffected tasks and label remaining measurements unperformed.

**Gate 1 — correctness.** Require exact fixtures and all independent formulations to agree within the frozen policy. Bounds must have the correct direction, and coverage code must use independent validation. For a material discrepancy, spend at most one repair cycle, rerun affected checks, then stop as invalid/inconclusive if unresolved. Do not build the full experiment runner while the estimand or confidence direction is wrong.

**Gate 2 — calibration.** Measure per-method median and 95th-percentile wall times, peak RSS, truth-generation attempts, and special-function costs. Forecast remaining work using twice the measured 95th-percentile dataset cost, including all binary LPs and validation arithmetic. Require the core primary/null run forecast to use at most 80% of its remaining allocated time; retain reserve for variation. Remove secondary work in §7.5 order if necessary. If the core cannot fit, do not change precision or statistical thresholds to make it fit. Record practical infeasibility, or stop after a diagnostic whose scope is explicit.

**Gate 3 — mechanism pilot.** Positive exact margins must be recoverable as validation counts become precise, and nulls must not generate systematic positive lower bounds. The fixed anchor is deliberately difficult; failure to detect it at \(2^{20}\) alone is not a fatal error. If all methods have either near-zero or near-one power at the primary budget, run one bounded diagnostic at the already specified adjacent secondary budget to distinguish lack of signal from a coding problem. The primary budget remains fixed. If this reveals a fatal input/algorithm error, repair within the shared allowance. If it only suggests a smaller gain, proceed with the inexpensive core confirmation unless an analytic argument has already settled the scoped claim. A noisy pilot is not a valid negative verdict.

An analytic early stop is allowed if a verified identity shows the candidate and strong reference have exactly the same certificates throughout the fixed evaluation distribution, or another mathematical obstruction makes the proposed gain impossible. Preserve the derivation and minimal checking code. Merely seeing similar values on 100 pilot cases is insufficient.

**Gate 4 — locked comparison.** Run the core before optional plots/robustness. Recompute the paired inference from saved counts/results using an independent analysis script. If a data-leakage or material numerical defect is discovered, retain invalid runs; repair once if allowance remains and use a fresh confirmation stream containing newly generated accepted channel pairs AND fresh count blocks. Resampling only counts on the previously inspected confirmation pairs is insufficient. All old and new computations count toward the caps. If a fresh adequate confirmation cannot fit, report invalid experiment or inconclusive evidence; do not mix old invalid and new data.

**Gate 5 — secondary diagnosis.** Spend remaining optional time only on the stated operating-range and conditioning questions. A favorable subgroup does not replace a failed mixture endpoint. If the binary baseline removes the improvement, diagnose margin versus range versus training error using existing saved witnesses; do not launch a new optimizer family to rescue the claim.

**Gate 6 — report.** Finish even after an early stop. The reason, evidence, limitations, and reproducible commands are the research product. A budget exhaustion, implementation bug, or inaccessible reference is not a scientific refutation.

## 10. Minimal implementation and commands

Use ordinary local scripts. Start with Python 3.12 if a native interpreter is available, NumPy, SciPy, Matplotlib, and a small test runner such as pytest. Use standard-library `fractions`, `decimal`, `json`, `csv`, `hashlib`, `time`, and `resource` where sufficient. Add an arbitrary-precision library only for the specified numerical cross-check if needed. Do not add services, a dashboard, a database, an ML framework, or a distributed scheduler.

The planning probes used NumPy 2.3.5 and SciPy 1.17.0 on Linux; this does not verify wheels or behavior on the target Mac. Install available native compatible releases, execute the smoke checks, then pin exact installed versions in `requirements.lock.txt`. Download/install time is setup effort, while any trial computations are charged to the experimental ledger. Current official SciPy docs describe the required HiGHS and exact binomial interfaces [R11,R15]; record any installed-version difference.

```text
channel_poc/
  README.md
  POC_EXECUTION_BRIEF.md
  requirements.lock.txt
  config.json
  protocol.lock.json
  sources/                      immutable supplied files if available
  src/
    channels.py                 P(u,v), generators, invariants, exact fixtures
    population.py               independent primal and dual LP builders
    certificates.py             witness validation, safe arithmetic, bound records
    baselines.py                binary training, confidence boxes, lifted LP, global TV
    sampling.py                 counts and isolated RNG streams; no method truth access
    experiment.py               bounded runner, ledger, failure/retry accounting
    analysis.py                 inference and figure/table regeneration
  scripts/
    inspect_environment.py
    check_exact.py
    calibrate.py
    run.py
    freeze_protocol.py
    report.py
  tests/                        only substantive mathematical/statistical checks
  results/
    environment.json
    budget_ledger.jsonl
    instances.npz
    counts/                     integer sufficient statistics, chunked
    runs.jsonl                  every method attempt, including failures
    tables/
    figures/
    REPORT.md
```

Suggested interfaces to create:

```text
population.solve(A, B) -> lower_bound, upper_bound, Q, H, diagnostics
certificates.validate(H, val_A_counts, val_B_counts, alpha) -> Certificate
baselines.binary_train(train_A_counts, train_B_counts) -> H, diagnostics
baselines.region(all_A_counts, all_B_counts, alpha) -> Certificate
sampling.draw(pair, n, seed_key) -> train_counts, validation_counts
experiment.run(stage, locked_config) -> append-only results and ledger
analysis.summarize(results, protocol) -> tables, plots, outcome evidence
```

Separate truth/generator objects from method inputs by interface design. A `Certificate` should include `lower_raw`, `lower_clipped`, `detected`, `status`, `elapsed_ns`, `solver_attempts`, `numeric_guard`, `witness_id`, and `failure_reason`. Store raw train/validation counts so every certificate can be regenerated. Store complete witnesses for primary, anchor, and failed runs; successful secondary witnesses may be regenerated from counts if disk is tight.

The commands below are **interfaces you must implement**, not commands already verified on the user's machine. Run from a newly created project directory. Adapt interpreter invocation to the inspected native environment, preserving scientific settings.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install numpy scipy matplotlib pytest
.venv/bin/python -m pip freeze > requirements.lock.txt
.venv/bin/python scripts/inspect_environment.py --out results/environment.json
.venv/bin/python scripts/check_exact.py
.venv/bin/python -m pytest -q
.venv/bin/python scripts/run.py --stage smoke --config config.json
.venv/bin/python scripts/calibrate.py --config config.json
.venv/bin/python scripts/run.py --stage pilot --config config.json
.venv/bin/python scripts/freeze_protocol.py --config config.json --out protocol.lock.json
.venv/bin/python scripts/run.py --stage confirm --protocol protocol.lock.json
.venv/bin/python scripts/run.py --stage secondary --protocol protocol.lock.json
.venv/bin/python scripts/report.py --protocol protocol.lock.json --results results
```

`smoke` must take under two minutes and use at most ten datasets. `confirm` runs only locked primary/null work; `secondary` obeys the locked resource reductions and remaining ledger. `report` must regenerate the essential outputs from saved files without running new experiments. Do not accidentally rerun the experiment during report generation. No command may silently resume with a changed protocol hash.

## 11. Outcome interpretation and required artifacts

### 11.1 Allowed conclusions

| Outcome | Required evidence and precise interpretation |
|---|---|
| **Supported contribution** | All validity gates pass and the primary improvement rule holds against both strong references. State the finite-data procedure, mixture, budget, confidence level, and observed limits. Historical originality and transfer to hardware remain separate questions. |
| **Informative negative result** | Valid, adequate locked evidence places at least one primary improvement upper bound below 0.10, or a verified analytic obstruction rules out the scoped hypothesis. Explain whether binary equivalence, sampling ranges, training cost, or weak signal caused the limitation. |
| **Practical infeasibility** | Actual measured costs or a justified forecast prevents the core study within the Mac ceilings. Identify the bottleneck and completed evidence; do not report hypothetical larger-run results. |
| **Inconclusive evidence** | Confidence intervals straddle the decision threshold, an essential comparison is missing, or the completed evidence cannot resolve the question. Name the specific uncertainty. |
| **Invalid experiment** | A substantive implementation, leakage, baseline, sampling, or numerical defect invalidates the attempted comparison. Preserve it and distinguish any repaired fresh confirmation. |

An underpowered run is not an informative negative. A timeout is not proof of mathematical failure. Passing the anchor is not novelty. Improvement over `global_tv` alone does not meet the primary criterion. A positive result on Family S alone cannot justify a claim about general noisy dynamics. Neither a finite positive nor a finite negative establishes universal superiority or inferiority.

The final report must start with the decision supported by the evidence. Include a compact claim/evidence table, the exact anchor and KL calculation, primary paired differences with intervals, null/coverage audit, resource accounting, deviations and failed runs, and a scoped interpretation. Distinguish analytically proved properties, reproduced source results, observed measurements, and untested hypotheses.

### 11.2 Final artifacts, proportional to the reached stage

- Runnable source and pinned environment, with concise setup and usage instructions, if implementation occurred.
- The original brief, locked configuration/hash, seed manifest, source identifiers/checksums, and environment/backend/precision record.
- Raw integer count blocks, accepted channel matrices and truth bounds, per-attempt method results, generator rejection summaries, failures, retries, and cumulative resource ledger for experiments actually run.
- Essential tables: exact checks; primary power differences; method power/coverage/failure rates; timing/memory; budget consumption. Essential figures: power versus tested observation budget; primary paired differences; deficiency/conditioning diagnosis when those data exist. Avoid decorative or redundant plots.
- `results/REPORT.md`, including an outcome from §11.1, limitations, novelty qualification, and exactly one next-step recommendation if warranted.
- A one-command path to regenerate the report tables/figures without resampling.

For an analytic early stop, provide the verifiable derivation or counterexample and only the minimal checking code required. Do not manufacture empty benchmark files to fill a template. For unavailable target hardware, provide completed code/exact checks and a clear list of target-Mac commands still unexecuted; label target performance unmeasured.

The most plausible next step after a valid but small gain is a separately preregistered variance-sensitive or allocation study applied equally to both witness classes. After a valid failure caused by endpoint identifiability, a different intervention/data model would be needed. Do not start either within this budget.

## 12. Research coverage, novelty boundary, and review disposition

### 12.1 Literature search and what it changed

Searches were conducted on **29 October 2026** using two web search engines and accessible primary full texts. Terms included Blackwell/postprocessing order dimension, fixed-output channel order, matrix majorization, Le Cam deficiency computation and confidence estimation, stochastic degradation tests, finite-sample divisibility, learned witnesses, biased distinguishability, process-tensor Markovianity, and reliable process tomography. Backward links from the source's Kuramochi/Hack references and established divisibility papers were checked; a current search identified a relevant September 2026 quantum-witness preprint.

The main decisions changed because (a) deficiency and LP comparison already have direct prior art; (b) biased binary tests are much stronger than the proposed spectral comparison; (c) operational memory requires different information; and (d) statistical witness learning is itself established. The paper's potentially original fixed-alphabet theorem remains of mathematical interest, but it does not supply the claimed practical advantage automatically.

The exact closest statistical application match was not located in this targeted search. That absence is limited evidence, not a novelty certificate. Older matrix-majorization material, particularly Dahl (1999), was available only at abstract level; the two 2026 SSRN leads cited by the supplied notes were not verified in full during this planning work. A recent quantum witness preprint [R16] was verified only through its indexed primary abstract after full-text retrieval attempts failed. No theorem, finite-sample guarantee, or implementation from that preprint is assumed here. These limitations block broad priority claims, not the self-contained classical diagnostic study.

### 12.2 Three review passes and material findings

**Mathematical and novelty review.** The main source proofs and small examples survived an independent audit. Review corrected the operational signal from signed-entry magnitude \(1/180\) to deficiency \(1/810\); separated finite monotone dimension from finite measurement/algorithm complexity; and identified complete biased-binary comparison as essential prior art. The numerical LP lower-bound direction and the no-27-factor holdout proof were separately challenged and checked. Residual uncertainty: historical originality of the fixed-alphabet and specific statistical specialization is not established.

**Experimental review.** The initial temptation to compare mainly against spectra was rejected. The protocol now uses a matched optimized binary restriction and a full-data lifted confidence-region LP; counts, training costs, and significance are explicit. It includes general noncommuting pairs, exact null boundaries, singular controls, independent confirmation, conservative failure handling, and a fixed power endpoint. A sparse shot curve cannot support a precise twofold efficiency claim, so the primary endpoint is a directly estimated 10-percentage-point gain at a fixed budget. That endpoint was fixed as a planning decision before any power experiment; the mean-penalty probes are not confirmation evidence.

**Execution review.** The mathematical task reduces to at most 55-variable CPU LPs and sufficient-statistic count simulation. The instructions specify native-Mac inspection, exact oracles, quantile/numerical checks, commands to create, cumulative caps, generator-attempt limits, resource forecasts, and early-stop artifacts. No inaccessible dataset or nonconstructive theorem is an algorithmic dependency. The final affected-section reviews also required production endpoint validation, a nonvacuous small coverage check, matched zero-gap/tie rules, feasible population upper bounds, and failure imputations protecting negative as well as positive conclusions. Those repairs are incorporated. Residual implementation speed and special-function checker risk remain explicit gates, not hidden behind a solver-success flag.

### 12.3 Stress cases and their disposition

| Adversarial scenario | Required response already incorporated |
|---|---|
| The theorem supplies only countably many targets, not a finite search algorithm | Use the explicit finite LP dual; do not enumerate dense targets |
| Realistic data violate iid preparation/reset assumptions | State the model failure; do not call the bound a memory certificate for those data |
| The mechanism is equivalent to a known method | Credit Blackwell/Le Cam and biased distinguishability; test only the stated finite-data gain |
| Only the special commuting family works | Show Family G results; primary scope condition prevents a broad positive claim |
| A properly optimized baseline removes the gain | Classify the scoped target as negative if intervals resolve it; preserve the result |
| Preprocessing or interval evaluation dominates | Charge it to method/end-to-end time; include it in resource calibration |
| Finite precision changes a verdict | Use exact fixtures, conservative evaluation, dual lower certificates, and the guard; unresolved cases are nondetections/blockers |
| RAM/runtime exceeds the estimate | Drop prespecified secondary work first; preserve core precision or report infeasibility |
| The pilot is inconclusive | Continue only the fixed core if feasible; no endless tuning or repeated favorable-seed search |
| An essential source is inaccessible | The mathematics here is executable; record provenance limits; missing comparator correctness blocks comparative claims |
| The main hypothesis is false | Report the valid negative/limitation; no mandatory larger benchmark or new project |

## 13. References and source locations

Citation keys are local to this standalone file. “Read” means decision-relevant source content was inspected during planning, not that every cited publication was fully reproduced or every historical reference was exhaustively searched.

**[S1] Supplied working paper.** *Finite channels with no finite system of monotone clocks*. Author field: “Mathematical investigation developed in this conversation.” 29 October 2026. `research_note(2).html`; identical archive copy `research_reproducibility/research_note.html`, with source `research_note.tex`. Critical locations: §§1–4, main theorem and `eq:matrix`, `eq:product`, `eq:order-test`, `eq:AB`, `eq:keyfactor`, `eq:deficiency-clock`; §5 binary proposition; §6 quantitative proposition; §7 continuous-time qualification; §§8–9 independent record/recovery results; §10 Pauli extension; §§11–12 computation and novelty limitations. All supplementary archive files are inventoried in §2.1. This is an unpublished supplied working note; mathematical audit is separate from publication-level originality.

**[R1] David Blackwell (1953).** *Equivalent Comparisons of Experiments*. Annals of Mathematical Statistics 24(2), 265–272. [DOI: 10.1214/aoms/1177729032](https://doi.org/10.1214/aoms/1177729032). Foundational attribution; direct original publisher full text was not relied on here. The finite randomization statement was checked through [R2–R4] and derived explicitly for the present LP.

**[R2] Maxim Raginsky (2011).** *Shannon Meets Blackwell and Le Cam: Channels, Codes, and Statistical Experiments*. IEEE ISIT. [Author PDF](https://maxim.ece.illinois.edu/pubs/raginsky_ISIT11.pdf); [DOI: 10.1109/ISIT.2011.6033729](https://doi.org/10.1109/ISIT.2011.6033729). Read §§II–III for TV, channel distance, and Le Cam versus Shannon comparison. Our postprocessing-only deficiency is the Le Cam quantity, not the broader input/output coding relation.

**[R3] `dit` project, channel-order implementation (documentation accessed 29 October 2026).** [Official `dit.channelorder.deficiency` source](https://dit.readthedocs.io/en/stable/_modules/dit/channelorder/deficiency.html), function `le_cam_deficiency`. Read the explicit LP, absolute-residual variables, and TV normalization. This is direct implementation prior art. Its channel arrays use input rows; transpose when comparing with this brief. Do not copy its failure-to-infinity behavior into a statistical certificate. `dit` is not a required dependency.

**[R4] Francesco Buscemi and Nilanjana Datta (2016).** *Equivalence between divisibility and monotonic decrease of information in classical and quantum stochastic processes*. Physical Review A 93, 012101. [Full preprint, arXiv:1408.7062](https://arxiv.org/pdf/1408.7062); [DOI: 10.1103/PhysRevA.93.012101](https://doi.org/10.1103/PhysRevA.93.012101). Read Lemma 1, Proposition 1/Lemma 2 for the semiclassical comparison, and Proposition 2/Corollary 2 for the complete quantum information condition. Its preparation/readout information order must be distinguished from intervention-based process Markovianity.

**[R5] Steffen Wißmann, Heinz-Peter Breuer, and Bassano Vacchini (2015).** *Generalized trace distance measure connecting quantum and classical non-Markovianity*. Physical Review A 92, 042108. [Full preprint, arXiv:1507.08867](https://arxiv.org/pdf/1507.08867); [DOI: 10.1103/PhysRevA.92.042108](https://doi.org/10.1103/PhysRevA.92.042108). Read §II, especially Eqs. (6)–(7), Lemma 1, and Theorem 2. Biased Helstrom matrices and invertibility are central. The present binary LP and its quantitative deficiency bound are derived in §§5.1 and 3.3.

**[R6] Felix A. Pollock, César Rodríguez-Rosario, Thomas Frauenheim, Mauro Paternostro, and Kavan Modi (2018).** *Operational Markov Condition for Quantum Processes*. Physical Review Letters 120, 040405. [Full preprint, arXiv:1801.09811](https://arxiv.org/pdf/1801.09811); [DOI: 10.1103/PhysRevLett.120.040405](https://doi.org/10.1103/PhysRevLett.120.040405). Read the operational condition, Eq. (2), the divisibility lemma, and the unnumbered product factorization immediately before Eq. (3), and the distance measure in Eq. (3). Establishes why causal-break interventions contain information absent from endpoint channels.

**[R7] Simon Milz, M. S. Kim, Felix A. Pollock, and Kavan Modi (2019).** *Completely Positive Divisibility Does Not Mean Markovianity*. Physical Review Letters 123, 040401. [Full preprint, arXiv:1901.05223](https://arxiv.org/pdf/1901.05223); [DOI: 10.1103/PhysRevLett.123.040401](https://doi.org/10.1103/PhysRevLett.123.040401). Read Definitions 1–3 and Eqs. (2)–(5). Distinguishes divisibility by inversion, operational divisibility, and multitime memory.

**[R8] Andreas Maurer and Massimiliano Pontil (2009).** *Empirical Bernstein Bounds and Sample Variance Penalization*. COLT 2009. [Full preprint, arXiv:0907.3740](https://arxiv.org/pdf/0907.3740). Read Theorem 1/Corollary 2 and the variance-sensitive alternatives. The primary protocol uses the elementary Hoeffding form; the stratified independent-summand range calculation and intersection-union argument are explicitly derived here. Variance-sensitive methods are not silently substituted after evaluation.

**[R9] Jonas M. Kübler, Wittawat Jitkrittum, Bernhard Schölkopf, and Krikamol Muandet (2022).** *A Witness Two-Sample Test*. AISTATS, PMLR 151, 1403–1419. [Primary paper](https://proceedings.mlr.press/v151/kubler22a/kubler22a.pdf). Read §§2–3, especially the training/test separation in §3. Learned held-out witnesses are prior art; their equality-null/asymptotic setting differs from the composite postprocessing-null finite bound specified here.

**[R10] Emilie Kaufmann, Olivier Cappé, and Aurélien Garivier (2016).** *On the Complexity of Best-Arm Identification in Multi-Armed Bandit Models*. JMLR 17(1), 1–42. [Primary paper](https://jmlr.org/papers/volume17/kaufman16a/kaufman16a.pdf). Read Lemma 1 and Appendix A.1 for the change-of-measure/event-divergence inequality. Equation (9) is the elementary fixed-balanced-design specialization derived for this anchor; no bandit algorithm is required.

**[R11] C. J. Clopper and E. S. Pearson (1934).** *The Use of Confidence or Fiducial Limits Illustrated in the Case of the Binomial*. Biometrika 26(4), 404–413. [DOI: 10.1093/biomet/26.4.404](https://doi.org/10.1093/biomet/26.4.404); [accessible original-paper copy](https://www.barestatistics.nl/uploads/1/1/7/9/11797954/clopper__pearson_1934.pdf). Confidence-belt construction on pp. 406–407 was inspected by the statistical reviewer. The lead also checked the [official SciPy `proportion_ci` documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats._result_classes.BinomTestResult.proportion_ci.html), which specifies `method="exact"`. Software numerical evaluation is distinct from the mathematical coverage theorem.

**[R12] Garth P. McCormick (1976).** *Computability of Global Solutions to Factorable Nonconvex Programs: Part I—Convex Underestimating Problems*. Mathematical Programming 10, 147–175. [DOI: 10.1007/BF01580665](https://doi.org/10.1007/BF01580665). Metadata/abstract checked; full original text was not accessible. The four product-envelope inequalities required by this brief are explicitly written and follow directly from bounded \(Q,A'\); no inaccessible theorem is an execution dependency.

**[R13] Tsachy Weissman, Erik Ordentlich, Gadiel Seroussi, Sergio Verdú, and Marcelo J. Weinberger (2003).** *Inequalities for the L1 Deviation of the Empirical Distribution*. HP Laboratories HPL-2003-97(R.1). [Primary technical-report mirror](https://shiftleft.com/mirrors/www.hpl.hp.com/techreports/2003/HPL-2003-97R1.pdf). Read Theorem 2.1, Eqs. (7)–(8), and the fact that its φ is at least 2. The report's \(L_1\) distance is twice the TV convention here; the ternary constant is \(2^3-2=6\).

**[R14] Le Phuc Thinh, Philippe Faist, Jonas Helsen, David Elkouss, and Stephanie Wehner (2019).** *Practical and Reliable Error Bars for Quantum Process Tomography*. Physical Review A 99, 052311. [Full preprint, arXiv:1808.00358](https://arxiv.org/pdf/1808.00358); [DOI: 10.1103/PhysRevA.99.052311](https://doi.org/10.1103/PhysRevA.99.052311). Read §§II–III, Theorems 1–2, and Appendix C. Relevant prior art for confidence regions and channel figures of merit, not a directly matched classical baseline.

**[R15] SciPy project (documentation accessed 29 October 2026).** [Official `linprog(method="highs")` documentation](https://docs.scipy.org/doc/scipy/reference/optimize.linprog-highs.html). Read solver statuses, feasibility options, marginals, and sign conventions. The installed target version/backend must still be checked and recorded.

**[R16] Pritam Roy, Saheli Mukherjee, and Bivas Mallick (2026).** *Witness-based quantification of non-Markovianity without complete Choi state reconstruction*. arXiv:2609.36080, submitted 28 September 2026. [Primary abstract](https://arxiv.org/abs/2609.36080). **Indexed abstract only; full text unverified.** Its claimed witness-based quantum lower bounds and limited-correlator measurements are a current overlap warning for any future broad quantum novelty claim. No mathematical result from it is used in this PoC.

**[R17] Yui Kuramochi (2022).** *Infinite dimensionality of the post-processing order of measurements on a general state space*. Journal of Physics A 55, 435301. [Full preprint, arXiv:2111.14129v2](https://arxiv.org/pdf/2111.14129); [DOI: 10.1088/1751-8121/ac99b0](https://doi.org/10.1088/1751-8121/ac99b0). Read Theorems 1–2 and Lemma 7, PDF pp. 16–18. Its standard-example construction uses three outcomes on one side and \(2n-1\) on the other. The supplied fixed-output strengthening is a narrower candidate contribution; the broad infinite-dimension claim is already established.

**[R18] Pedro Hack, Daniel A. Braun, and Sebastian Gottwald (2024).** *Majorization Requires Infinitely Many Second Laws*. Proceedings of the Royal Society A 480, 20230464. [Full preprint, arXiv:2207.11059v3](https://arxiv.org/pdf/2207.11059); [DOI: 10.1098/rspa.2023.0464](https://doi.org/10.1098/rspa.2023.0464). Read §§4.2–5, Theorem 1 in the arXiv version (journal numbering differs). Its strict-monotone condition for state majorization is not the supplied weak-monotone channel theorem; ordinary weak majorization admits finite scores.

**[R19] Geir Dahl (1999).** *Matrix Majorization*. Linear Algebra and its Applications 288, 53–73. [DOI: 10.1016/S0024-3795(98)10175-1](https://doi.org/10.1016/S0024-3795(98)10175-1). Primary abstract only; full-text access failed. Retain this limitation when discussing the historical originality of fixed-alphabet postprocessing geometry.

**[R20] NIST Digital Library of Mathematical Functions (accessed 29 October 2026).** [§5.11, Gamma-function asymptotic expansions](https://dlmf.nist.gov/5.11), Eq. 5.11.1 and §5.11(ii), first paragraph. Read the positive-real first-neglected-term sign and magnitude bound. Applying the gamma recurrence gives the factorial-log enclosure in §6.1. This is a finite numerical bound, not use of an uncontrolled asymptotic approximation.

**[R21] Python Software Foundation (documentation accessed 29 October 2026).** [Official Decimal documentation](https://docs.python.org/3/library/decimal.html), entries for ln, exp, sqrt, next_minus, next_plus, and rounding contexts. Read the distinction between correctly rounded HALF_EVEN elementary functions and directed ordinary arithmetic. Verify the installed Python version supports the used interfaces; no Python 3.14/3.15-only feature is required.

**Final instruction:** complete the evidence-supported branch, preserve unfavorable results, and stop when the scoped question is answered or a gate legitimately prevents its resolution. The value of this PoC is the trustworthiness and usefulness of the resulting research decision.
