# Nilpotent residual arithmetic: research assessment and executable PoC plan

**Research date:** 30 October 2026.  
**Target machine:** one MacBook Pro with an Apple M3 Max; actual memory, storage, macOS version, and numerical-library support must be discovered.  
**Deliverable type:** standalone research and experimental handoff, with small checks already executed and a risk-ordered implementation plan.  
**Scope:** the nilpotent residual-stability result in section 5 of the supplied `nilpotent_and_matrix.md`. The matrix-softmax theorem is outside this project. The notebook mentioned by the seed was unavailable and was not read; no claim here relies on its asserted proof checks, code, or results.

## 1. Executive assessment and selected direction

### 1.1 Recommendation

Pursue **finite-depth optimal metric design for jointly nilpotent random residual computations**, with two connected but separately evaluated outputs:

1. A mathematical and computational account of the least coordinate distortion needed to bound expected forward and adjoint amplification, including exact small-case answers and reductions justified by symmetry.
2. Only if a concrete numerical bottleneck is found, a fixed-semantics arithmetic experiment asking whether the resulting representation improves error, dynamic range, or total cost over strong existing numerical methods.

The first output is the core laptop PoC. It does not require training a neural network, downloading a dataset, a remote GPU, or a paid API. The second is an evidence-triggered extension, not a condition retroactively imposed on a valid component-level result.

The seed's sharp asymptotic theorem survives reconstruction under its stated assumptions. However, dual-number differentiation, nilpotent CP-map filtrations, stable-coordinate optimization by linear matrix inequalities (LMIs), diagonal scaling by geometric programming, and symmetry reduction are established. The proposed implementation must reuse that machinery and describe its contribution accurately. [BM14] [BEFB94] [LP11] [NR99] [GP04] [Ishii21]

**No neural accuracy, training-speed, inference-speed, or floating-point accuracy advantage has been demonstrated.** The strongest plausible differentiation is a precise finite-depth specialization, useful exact constants or certificates, and reduced computational cost for this particular model. Historical novelty of the sharp metric-cost theorem remains unresolved. There is no basis for calling this a new number system or a breakthrough neural architecture.

### 1.2 What was established during this investigation

The work performed for this plan included primary-source research, reconstruction and independent mathematical challenge of the nilpotent proof, two independent floating-point negative controls, exact moment calculations, a small convex-optimization comparison, and an 80-decimal-digit check of a solver failure. All computation was on **Linux x86-64**, not on an M3 Max.

| Finding | Evidence and implication |
|---|---|
| The claimed asymptotic costs are mathematically consistent | Section 2 gives a complete proof, including constants and the determinant argument omitted from the seed's abbreviated presentation. |
| Both finite-depth gain constraints are convex constraints in the same metric | A Schur-complement LMI handles the apparent inverse-metric obstacle. A generic small SDP is therefore a necessary reference, not an optional weak competitor. |
| For the unit-chain test, tuning the same `L^-j` grade rule removes most of the apparent gap | For a unit chain of nilpotent depth 3, total depth 100, and squared-gain cap 8, the best scalar grade scale has condition number about 80,142.9518; the exact unrestricted optimum is about 79,932.3970. The incremental improvement is approximately 0.263%. The untuned value 1,000,000 is for this chain comparison; the seed's exterior example is a different algebra. |
| Smaller internal magnitudes can leave numerical accuracy unchanged | In a float32 chain with depth 4,096 and nilpotent depth 12, binary grade scaling reduced the final maximum coefficient magnitude from about 9.91e16 to 352.106, while decoded results were bit-for-bit identical. |
| Solver residuals can create a false improvement | After an orthogonal change of basis, an SDP returned an apparently smaller condition number, but its adjoint gain was 8.0000701444, exceeding the cap 8. High-precision reevaluation confirmed the violation. This candidate must be rejected. |

These are exploratory checks and implementation diagnostics, not protected confirmatory results or M3 performance measurements. Section 10 records the numerical details; Appendix A contains code sufficient to reproduce the central checks independently.

### 1.3 Decision boundaries

The project may succeed as mathematics, as numerical engineering, as a negative result, or in more than one category. Keep the categories separate:

- **Mathematical result:** prove a finite-depth characterization, establish a sharp constant, or identify an exact predecessor to the seed theorem.
- **Engineering result:** compute the same validated optimum with lower total time or memory than the strongest faithful locally executable formulation.
- **Arithmetic result:** improve decoded numerical error or range at a specified cost, or lower cost at a specified error, with conversions and metric construction included.
- **Replication or negative result:** existing methods already solve the problem adequately, the extra metric freedom has negligible value, or internal stabilization does not improve physical outputs.

Do not count a better objective than a correctly solved equivalent full SDP as a success. It indicates a numerical or modeling discrepancy until resolved. Do not convert a small mathematical improvement into a practical claim without measuring its consequence.

## 2. Problem statement, assumptions, and mathematical audit

### 2.1 Intended beneficiaries and the actual bottleneck

Potential users are researchers implementing derivative-carrying arithmetic, stochastic residual primitives, and structured linear computations. Their existing workflow is to choose an algebraic representation, propagate its coordinates, and use a norm or numerical precision policy to keep computation manageable.

The demonstrated bottleneck is specific: **at fixed per-layer noise scale, a single coordinate metric cannot keep both expected gains bounded through arbitrarily large depth without becoming increasingly distorted.** This is a mathematical limitation of the stated stochastic model. It is not evidence that a deployed application currently fails, that an entire neural network is unstable, or that changing coordinates improves its input-output conditioning.

The computational task is: given fixed generators, a horizon, and a permitted gain, return a feasible metric, its distortion and volume cost, and enough evidence to assess proximity to the optimum. That is a useful component-level task even before an application has been identified.

### 2.2 Exact model and conventions

Work over real matrices initially. Let `d` be the state dimension, `s` the number of generators, and

\[
N_1,\ldots,N_s\in\mathbb R^{d\times d},\qquad
W_\ell=I+\sum_{i=1}^s\xi_{\ell i}N_i,\qquad
P_L=W_L\cdots W_1.
\]

Layers are independent, coefficients have zero mean, and their within-layer covariance is the identity. Gaussianity and independence between components of the same layer are unnecessary. Finite second moments suffice for the identities below.

The generators are **associatively jointly nilpotent of depth `r`**: every word of length `r+1` vanishes, and at least one word of length `r` is nonzero. This is stronger than each generator being nilpotent, and is different from nilpotence of a Lie algebra of commutators.

For a symmetric positive-definite metric `Q`, choose `T` with `T.T @ T = Q` and put `B = T @ P_L @ inv(T)`. The symmetric square root is one choice. If a library returns a lower Cholesky factor `Q = R @ R.T`, use **`T = R.T`**.

The metric is deterministic conditional on the fixed generator tuple, horizon, and specified coefficient moments; it is not fitted to the same realized random path inside the expectation. Adaptive path-dependent exponent-management baselines can be evaluated empirically, but their guarantees do not follow by substituting a random metric into the identities below.

Define squared gains

\[
g_f(Q)=\lambda_{\max}\left(Q^{-1/2}\mathbb E[P_L^TQP_L]Q^{-1/2}\right),
\]

\[
g_b(Q)=\lambda_{\max}\left(Q^{1/2}\mathbb E[P_LQ^{-1}P_L^T]Q^{1/2}\right).
\]

The constraint is `max(g_f, g_b) <= C`, with fixed `C > 1`. These quantities have no physical units after the metric is fixed. RMS amplification is their square root. Multiplying `Q` by a positive scalar leaves both gains and its condition number unchanged.

Validate `d>=1`, finite real generator entries, integer `L>=0`, finite `C>1`, and a finite symmetric positive-semidefinite covariance before constructing a solver. For nonconstant amplitudes, validate a finite real schedule of length `L`. **Handle the identity case first:** if `L=0`, the total variance `V=0`, or all effective generators vanish, return `Q=I`, both gains 1, condition number 1, and determinant 1 without forming factors or dividing by `L` or `V`. The formulas with those denominators below apply to the remaining positive-variance cases. An empty nonidentity Kraus list must take this branch, with dimension supplied by the instance rather than inferred from its first factor.

The primary mathematical objective is

\[
\kappa_*(L,C)=\inf_{Q\succ0,\;g_f,g_b\le C}
\kappa_2(Q),\quad
\kappa_2(Q)=\lambda_{\max}(Q)/\lambda_{\min}(Q).
\]

Normalize `lambda_max(Q)=1` for a separate volume objective:

\[
D_*(L,C)=\sup_{Q\succ0,\;Q\preceq I,\;g_f,g_b\le C}\det Q.
\]

At the determinant optimum the largest eigenvalue is 1, since scaling upward otherwise improves the objective. Minimizing condition number and maximizing determinant are different optimization problems; one solution need not solve both.

Report `log2(kappa(Q))` and `-log2(det(Q))` to avoid enormous numbers, together with the original values when representable. The coordinate map has `kappa(T)=sqrt(kappa(Q))` and volume multiplier `det(T)=sqrt(det(Q))`. These are geometric quantities, **not automatically lost mantissa bits or lost information**. The metric-unit ellipsoid has volume proportional to `det(Q)^(-1/2)`, the opposite direction from coordinate-volume contraction.

### 2.3 What the gain bound does and does not control

For every deterministic input `x`, the forward condition bounds `E ||T P_L x||^2` by `C ||T x||^2`. The adjoint statement concerns a deterministic terminal covector. A loss gradient that depends on the same random product is not automatically covered. Nor do these conditions bound the inverse product, the smallest singular value, every realization, or every nonlinear trained network.

A generic consequence is `E ||B||_op^2 <= E ||B||_F^2 <= d C`; this dimension-dependent bound must not be reported as the stronger `C` bound on samplewise operator norm. Gaussian and heavy-tailed coefficient laws can have the same exact expected gains but different overflow frequencies and tail behavior.

With encoding and decoding,

\[
x_0\mapsto Tx_0\mapsto TP_LT^{-1}(Tx_0)\mapsto T^{-1}TP_Lx_0=P_Lx_0.
\]

Thus a pure coordinate change preserves the exact physical map and its physical Jacobian. If inputs, outputs, or residual amplitudes are changed instead, record that as a different computation.

### 2.4 Exact moments and filtration

Define the positive linear maps

\[
\Psi(X)=\sum_iN_i^TXN_i,\qquad
\Psi^*(X)=\sum_iN_iXN_i^T.
\]

Independence and centering give

\[
\mathcal T_L(X):=\mathbb E[P_L^TXP_L]
= (\operatorname{Id}+\Psi)^L(X)
=\sum_{k=0}^{\min(r,L)}\binom Lk\Psi^k(X).
\tag{1}
\]

The adjoint moment map replaces `Psi` by `Psi*`. These are exact population moments, so Monte Carlo is unnecessary for the primary metric objective. Multiplicative-noise moment recursions are established control-theory tools. [BEFB94] [ZDZ18]

Let `F_0 = R^d`, `F_{r+1} = {0}`, and

\[
F_j=\operatorname{span}\{\operatorname{im}(N_{i_j}\cdots N_{i_1})\},
\quad f_j=\dim F_j,\quad
E_j=F_j\cap F_{j+1}^{\perp},\quad e_j=f_j-f_{j+1}.
\]

Then `N_i E_j` lies in the sum of levels strictly above `j`. The filtration and its level dimensions are closely related to the existing CP nilpotent type; they are not new invariants. [BM14]

Compute the filtration without enumerating words: start with a basis of `F_0`, then form `F_(j+1)=span_i(N_i F_j)`, compressing the concatenated image bases at each step. Use exact rational rank arithmetic for small rational inputs, or the declared construction certificate for generated families. A floating QR/SVD basis is a numerical representation of that structure: record rank decisions, orthogonality residuals, and residuals of `N_i F_j` outside the represented `F_(j+1)`. A threshold alone does not certify exact nilpotence. This polynomial sequence of subspace computations is also the default route when structure discovery must be timed.

### 2.5 Complete proof of the sharp asymptotic costs

The following proof is reconstructed here from the stated model. It is self-contained and does not assume the missing notebook's conclusions. It concerns `L>=1` and a nonzero effective tuple; the identity branch was specified above.

**Necessary conditioning.** From the first positive term in (1), every forward-feasible metric satisfies

\[
L\Psi(Q)\preceq(C-1)Q.
\]

Positivity permits iteration, giving

\[
\Psi^j(Q)\preceq((C-1)/L)^jQ.
\]

Since `Psi^r(Q) >= lambda_min(Q) Psi^r(I)` and `Psi^r(I)` is nonzero,

\[
\boxed{\kappa(Q)\ge
\left(\frac L{C-1}\right)^r\|\Psi^r(I)\|_{\rm op}.}
\tag{2}
\]

The seed's alternative bound

\[
\kappa(Q)\ge \frac{\binom Lr}{C-1}\|\Psi^r(I)\|_{\rm op}
\tag{3}
\]

also follows from the `r`th term when `L >= r`. Use the larger bound for finite-depth diagnostics; neither dominates for all parameters. These bounds require only the forward constraint, so they remain necessary when the adjoint constraint is added.

**Necessary eigenvalue and determinant costs.** Normalize `lambda_max(Q)=1`. Put

\[
R_j=(\Psi^*)^j(I),\qquad \rho_j=\lambda_{\min}^{+}(R_j)>0.
\]

The range of `R_j` is `F_j`, so `R_j >= rho_j Pi_Fj`. Trace duality gives

\[
\operatorname{Tr}(QR_j)=\operatorname{Tr}(\Psi^j(Q))
\le d((C-1)/L)^j.
\]

The positive compression `Pi_Fj Q Pi_Fj` therefore has operator norm at most

\[
\frac d{\rho_j}((C-1)/L)^j.
\tag{4}
\]

The min-max principle implies that at least `f_j` eigenvalues of `Q` obey this bound. Group the ordered eigenvalues by the nested counts `f_j` to obtain

\[
\det Q=O\left(L^{-\sum_{j=1}^r j(f_j-f_{j+1})}\right)
=O(L^{-\nu}),\qquad \nu=\sum_{j=1}^rf_j.
\tag{5}
\]

**Matching upper construction.** Let `A = sum_i ||N_i||_F^2`. For `0<t<=1`, define

\[
T_t=\sum_{j=0}^r t^j\Pi_{E_j},\qquad Q_t=T_t^TT_t.
\]

Every nonzero generator block from level `j` to level `k>j` acquires factor `t^(k-j)`, hence

\[
\sum_i\|T_tN_iT_t^{-1}\|_F^2\le t^2A.
\]

Both one-step expected Gram matrices are bounded by `(1+t^2 A) I`. Conditional expectation through independent products bounds both length-`L` gains by `(1+t^2 A)^L <= exp(L t^2 A)`. For `A>0`, choose

\[
t^2=\min\{1,\log(C)/(LA)\}.
\tag{6}
\]

Then `Q_t` is feasible,

\[
\kappa(Q_t)=t^{-2r}=\Theta(L^r),\qquad
\det Q_t=t^{2\sum_jje_j}=\Theta(L^{-\nu}).
\]

If every generator is zero, use `r=0`, `Q=I`, gains 1, condition number 1, and determinant 1. All asymptotic constants above depend on fixed generators, dimension, and `C`. The results are not uniform as algebra size, weights, or nilpotency depth also grow.

This proves the seed's conditioning, eigenvalue-envelope, and determinant orders. It does not establish historical priority. Filtration-dependent spectral asymptotics also appear in controllability-Gramian analysis, under different dynamics and objectives. [SJP15]

### 2.6 A useful extension: total variance, not depth alone

For independent layers `W_l = I + a_l sum_i xi_li N_i`, write `V = sum_l a_l^2`. The exact moment map is

\[
\prod_{\ell=1}^L(\operatorname{Id}+a_\ell^2\Psi)
=\sum_{k=0}^r\beta_k\Psi^k,\qquad
\beta_k=e_k(a_1^2,\ldots,a_L^2),
\tag{7}
\]

where `e_k` is the elementary symmetric polynomial. The first coefficient is `V`; the previous proof yields `inf kappa(Q)=Theta(V^r)` and `sup det(Q)=Theta(V^-nu)` as `V -> infinity`. The upper bound is `exp(A t^2 V)`. These bounds hold uniformly over amplitude schedules, including concentrated variance. They need no assumption that variance is spread over many layers.

This is a deduction from the same mechanism, not an independently established novel contribution. Finite-depth constants still depend on all `beta_k`, not just `V`. In particular, `a_l=L^(-1/2)` makes `V=1`, removing the growing conditioning requirement by changing the physical residual computation. Depth-dependent residual scaling has substantial existing literature. [Marion25]

For covariance `Sigma = R R.T`, replace generators with `M_a = sum_i R_ia N_i`. Positive-definite covariance preserves their span and filtration; singular covariance can change both. Do not apply the scalar-variance extension when covariance shape changes between layers without recomputing the full moment operators.

### 2.7 Central hypotheses and falsifiers

| ID | Hypothesis and supporting evidence | Main alternative explanation | Observation that weakens or falsifies the claim |
|---|---|---|---|
| H1 | The stated exponents and finite-depth identities describe the specified model. Supported by the proof and executed exact checks. | Incorrect algebra construction, rank detection, or distribution assumptions. | An exact counterexample satisfying every assumption; numerical disagreement alone first triggers implementation diagnosis. |
| H2 | Additional metric freedom can reduce finite-depth distortion beyond the best scalar grade scale. The exact `r=3` example proves a small instance of this. | Comparing against an untuned scale; violating the gain cap; changing the objective or noise. | The validated optimum equals the tuned simple baseline for the tested family, or the difference is below certificate uncertainty. |
| H3 | An exact structural reduction can reduce time or memory at equal solution quality. Symmetry and diagonal preservation give a plausible mechanism. | Giving the candidate sparsity, cached moments, or better solver settings withheld from the baseline. | Gains disappear against the best generic reduced formulation or after all reusable optimizations are shared. |
| H4 | In a separately identified workload, metric design improves the decoded error-cost or range-cost tradeoff. No positive evidence yet; binary scaling controls are negative. | Exponent-range changes, more precision, changed outputs, different operation ordering, or unfair baseline algorithms. | Same decoded error at greater total cost; a generic binary rescaling, float64, or stronger coefficient algorithm matches or dominates it. |

H1 is a mathematical statement. H2 is a finite-instance optimization claim. H3 is a performance claim. H4 is an application claim. None substitutes for another.

## 3. Prior art, reuse, and novelty assessment

### 3.1 Closest alternatives

| Approach | Problem and mechanism | Assumptions/evidence | Requirements and reusable implementation | Exact overlap and difference |
|---|---|---|---|---|
| Nilpotent CP maps, Bhat–Mallick [BM14] | Kraus-word nilpotence, kernel/image filtrations and dimension types. | Structural mathematical results; no neural performance claim. | Small matrix algebra; reuse definitions and proof tools. | Supplies central algebraic ingredients, but the inspected paper does not solve the stated two-sided metric-cost optimization. |
| LMI stability and coordinate conditioning [BEFB94] | Quadratic metrics, minimum-distortion stable coordinates, multiplicative-noise moments. | Established deterministic/stochastic control settings. | Generic SDP; reuse CVXPY and current conic solvers. | Closest historical methodology. An LMI or minimum-condition-number formulation is not new. |
| Convex condition-number optimization [LP11] | Converts PSD-representable metric problems to convex programs. | General convex-set analysis and numerical methods. | Generic convex optimization. | Covers the optimization machinery, not an inspected exact predecessor to the nilpotent exponents. |
| Matrix scaling by GP [NR99] | Log-coordinate formulations for nonnegative matrix scaling/balancing. | Different prescribed-sum or balancing objectives. | Exponential-cone or geometric programming. | Strong threat to broad claims for the diagonal solver; the finite-horizon coefficients and objective are the specialization. |
| SDP symmetry reduction [GP04] | Average over a group to restrict to invariant matrices. | Requires actual invariance of constraints and objective. | Standard convex modeling plus a proved representation. | Justifies diagonal/grade reductions; symmetry reduction itself is established. |
| Finite-time stochastic stability [ZDZ18] | Exact transitions and matrix inequalities for finite-horizon mean-square bounds. | More general time-varying systems and metrics. | CPU matrix operations and LMIs. | Shares the stochastic question; allowing a different metric at every layer changes this project's common-metric problem. |
| Gramian asymptotics [SJP15] | Filtration dimensions control eigenvalue and determinant orders. | Linear affine control and small-time controllability. | Analytic estimates and matrix algebra. | Strong structural resemblance; different map and optimization variable. |
| Noncommutative optimization [BFGOW21] | Simultaneous conjugation, moment maps, null cones, quantitative scaling. | Broad group actions; some guarantees depend on positive capacity. | General algorithms are substantial infrastructure. | Adjacent quantitative conditioning theory; nilpotent tuples lie in a degenerating regime. No exact equivalent theorem located in inspected sections. |
| Classical matrix balancing [JLL14] [SciPyBal] | Reduce row/column norm imbalance using a diagonal similarity. | Eigenvalue computation; documented accuracy regressions exist. | `scipy.linalg.matrix_balance`, wrapping LAPACK. | Useful diagnostic, but it does not solve the prescribed two-sided finite-horizon problem. |
| Dual/Weil/Taylor arithmetic [Ishii21] [JAXJet] [TaylorSeries] | Propagate truncated polynomial or derivative coefficients. | Established differentiation mechanisms and implementations. | JAX, TaylorSeries.jl, existing Weil-algebra code. | Rules out novelty for derivative-carrying nilpotents or factorial normalization alone. |
| Modern Taylor-mode optimization [Dangel25] | Collapse derivative sums for linear differential operators. | Task-specific operator computations; published GPU settings differ from this laptop. | `torch-jet`, subject to primitive coverage. | Required comparator for a PDE-operator pivot; not a replacement for all coefficients of arbitrary residual products. |
| Residual-depth scaling [Marion25] | Scale residual amplitudes according to depth and weight regularity. | Neural-network assumptions differ from this linear model. | Simple scalar changes; existing research code. | A strong architecture alternative if changing the physical map is allowed; a coordinate change with decoding does not obtain the same external effect. |

### 3.2 What is known, reusable, different, and potentially valuable

**Already exists:** the number-system constructions, exact second-moment propagation, the structural filtration, stable-metric optimization, GP scaling, group averaging, and practical Taylor arithmetic. Broad dual-number neural-training concepts also appear in an inspected patent publication; that establishes historical activity, not performance or a legal conclusion. [Patent19]

**Reuse:** NumPy/SciPy for CPU linear algebra and sparse operations; CVXPY with Clarabel and SCS for convex references; existing normalized-coefficient AD interfaces if an application extension is justified. Implement the small model-specific moment operators, validation, and benchmark harness. Do not build an autodiff engine, a generic conic solver, or a neural training stack for the core PoC.

**Potentially substantive difference:** the exact sharp common-metric distortion statement, its finite-depth constants, and a documented relation between algebraic interaction structure and practical metric-selection costs. The `r=3` formula in section 4.6 is derived here, but has not been established as historically new. The total-variance extension is mathematically useful and modest.

**Why it might matter:** an explicit frontier tells an implementer what distortion is unavoidable; an exact reduction can make that calculation cheap; a valid negative result can prevent time spent trying to obtain external robustness from a coordinate relabeling. A mathematical component result need not wait for an end-to-end neural benchmark.

**Discriminating tests:** compare tuned one-parameter, diagonal, grade-only, block, and unrestricted metrics under identical generators and gain caps; compare equivalent formulations at matched validated accuracy; decode every numerical experiment into the same external units.

**Novelty invalidation:** a prior theorem covering the same common metric, two moments, normalization, and exponents invalidates the corresponding originality claim. An existing GP or symmetry-reduced SDP that produces the same formulation invalidates an algorithmic-invention claim, even if a new implementation is useful. Record such findings positively as resolved attribution, then retain only a replication or engineering claim.

### 3.3 Selection among plausible directions

| Direction | Decision and technical reason |
|---|---|
| New nilpotent neural architecture | Not selected. The seed supplies a linear random-product theorem, not a learned computation, representational advantage, or task definition. Nonlinearities and trained dense Jacobians may destroy the filtration. |
| New higher-order AD number system | Not selected. Closest existing arithmetic and normalized Taylor interfaces already cover the basic mechanism. |
| Finite-depth optimal metric geometry | **Selected.** Preserves the seed's question, has exact observables, meaningful small oracles, strong local baselines, and a direct path to falsification. |
| Faster structural metric computation | Included as an engineering hypothesis once equivalence and accuracy are established. Generic reduced convex formulations remain competitors. |
| Finite-precision coefficient computation | Conditional extension after a real failure mode and amortization case are established. Current negative controls argue against assuming a benefit. |
| Time-varying metrics, deterministic drift, learned nonlinear blocks | Deferred. They change assumptions and require a new theorem or explicit empirical claim. The inaccessible notebook's drift extension is not treated as evidence. |

The strongest case for proceeding is a precise theorem plus a small, testable optimization problem. The strongest case against a practical payoff is that known convex methods already solve it, the simple tuned scale can be nearly optimal, and coordinate scaling often changes neither decoded accuracy nor external conditioning. Both cases remain visible throughout the experiment program.

## 4. Selected mechanism and implementation specification

### 4.1 Exact finite-depth metric programs

Let the nonidentity CP map have a Kraus representation

\[
\mathcal F_L(Q)=\mathcal T_L(Q)-Q=\sum_{a=1}^K A_a^T Q A_a.
\]

An explicit representation uses each nonzero length-`k` generator word with multiplier `sqrt(binomial(L,k))`. For a general amplitude schedule, replace the binomial coefficient with `beta_k` from (7).

The forward constraint is the LMI

\[
F(Q):=(C-1)Q-\sum_a A_a^T Q A_a\succeq0.
\tag{8}
\]

The adjoint constraint is equivalent, for `Q>0`, to

\[
G(Q):=
\begin{pmatrix}
(C-1)Q&QA_1&\cdots&QA_K\\
A_1^TQ&Q&&0\\
\vdots&&\ddots&\\
A_K^TQ&0&&Q
\end{pmatrix}\succeq0.
\tag{9}
\]

Indeed, the Schur complement gives `(C-1)Q >= sum_a Q A_a Q^-1 A_a.T Q`; congruence by `Q^-1` recovers the original backward condition. This is an application of established LMI machinery, not a new general optimization principle. [BEFB94]

Solve

\[
\max_{Q,u}\ u\quad\text{subject to}\quad
uI\preceq Q\preceq I,\quad F(Q)\succeq0,\quad G(Q)\succeq0,\quad u\ge0.
\tag{10}
\]

The optimum is `u_* = 1/kappa_*`. The construction in (6) proves existence of a positive feasible value for every finite nontrivial instance. An optimizer returning `u` indistinguishable from zero or reporting infeasibility is not evidence against that theorem.

For the separate determinant frontier, maximize `log_det(Q)` subject to (8), (9), and `Q<=I`. This is convex log-determinant optimization, not a linear-objective SDP. A third optional program maximizes determinant subject to a fixed condition-number budget; record it as a separate Pareto slice rather than conflating the two optima.

### 4.2 Gain evaluation and moment construction

Implement `moment(X, generators, coefficients, adjoint=False)` as

```python
term = X.copy()
result = coefficients[0] * term  # beta_0 = 1
for k in range(1, r + 1):
    term = sum(N @ term @ N.T for N in generators) if adjoint else \
           sum(N.T @ term @ N for N in generators)
    result = result + coefficients[k] * term
return result
```

Use `binomial(L,k)=0` for `k>L`. Do not update `term` sequentially once per generator: the sum is one application of the map. For raw layer simulation, all generator contributions must use the **previous** state; an in-place cascade silently inserts extra words.

Evaluate gains by symmetric eigendecomposition or generalized symmetric eigenvalue problems, with symmetric roundoff cleanup `(X+X.T)/2`. Prefer linear solves to explicit inverses. Independently reevaluate difficult candidates using higher precision and the original uncompressed moment operator. Store the unnormalized and normalized matrices and the gain-check precision.

For uniform layers, build coefficients with integer binomials and a log-domain representation when necessary. For schedules, update elementary symmetric coefficients in descending order: `beta[k] += a_l^2 * beta[k-1]`. Carry at most `r+1` terms. The exact moment identity is polynomial in `L` of degree at most `r`; long horizons do not require simulating every layer.

**Kraus construction without exponential word enumeration:**

- One generator: use `A_k = sqrt(beta_k) N^k`, omitting exact zero powers.
- Exterior creation operators: one canonical ordered product per nonempty subset `S`, multiplied by `sqrt(beta_|S| * |S|!)`. Word permutations differ by signs, which disappear in the CP sum.
- Generic small instances: use Choi recursion. With column-major vectorization, `J_0=vec(I)vec(I).T`, `J_(k+1)=sum_i (I kron N_i) J_k (I kron N_i).T`, and `J=sum_(k>=1) beta_k J_k`. Factor `J` and reshape eigenvectors in the same column-major convention. Check the resulting map on matrix units and independent symmetric matrices. Never assume a reshuffling convention without this test.

The Choi dimension is `d^2`; its storage is `O(d^4)`. A rank-revealing factorization gives at most `d^2` Kraus factors, but can be costly. Analytic factorization is preferable when available. Both the baseline and candidate receive it.

Do not silently discard small Choi eigenmodes. If the discarded factors have total squared Frobenius norm `epsilon`, either gain can change by up to `kappa(Q)*epsilon`. A numerically small omitted mode can matter greatly for an ill-conditioned metric. Preserve exact structural zeros, bound reconstruction error, and reject an unresolved compression error larger than the gain slack.

### 4.3 Structured diagonal program

First verify that both CP maps preserve diagonal matrices. A sufficient condition is that every generator is a weighted partial permutation: at most one nonzero per row and per column. Let

\[
M_{ab}=\sum_i N_i[a,b]^2,\qquad
H=\sum_{k=0}^r\beta_k M^k.
\]

For `Q=diag(q)`, `q_a=exp(z_a)`, define `A(z)=diag(q) H diag(q)^-1`. Then the exact forward gain is the maximum **column sum** and the exact adjoint gain the maximum **row sum** of `A(z)`.

Minimize `tau` subject to

\[
-\tau\le z_a\le0,
\quad\sum_a H_{ab}e^{z_a-z_b}\le C\ \forall b,
\quad\sum_b H_{ab}e^{z_a-z_b}\le C\ \forall a.
\tag{11}
\]

This is a convex exponential-cone program, with optimum `tau=log(kappa)` within the allowed diagonal class. For numerical stability, remove the diagonal identity and use `logsumexp(log(H_ab)+z_a-z_b) <= log(C-1)` on positive off-diagonal terms. An empty row or column needs no nonidentity constraint. Store `z` as the authoritative output; do not force `exp(z)` into an underflowed array.

Maximize `sum(z)` under `z<=0` and the same gain constraints for the diagonal determinant problem. This is an established style of geometric programming. [NR99]

### 4.4 When the diagonal restriction is exact

Diagonal preservation alone is insufficient. A separate symmetry certificate must show that an unrestricted optimum can be averaged into the diagonal class without worsening the objective. The feasible set in (10) is convex and invariant under an appropriate unitary conjugation; `logdet` is concave and invariant. This is the standard symmetry principle. [GP04]

Approved certificates for the initial suite are:

- **Weighted single chain:** `U_theta=diag(exp(i*j*theta))` sends its generator to `exp(i*theta)N`. Average the Hermitian extension over more than `r` equally spaced phases. Every off-diagonal entry vanishes, and the average is real diagonal. Averaging only an alternating real sign matrix would be insufficient.
- **Exterior creation:** for every sign vector, `D_s e_A=(product_(i in A) s_i)e_A`. Each generator changes only by a sign, and averaging kills all off-diagonal entries. Independent unequal generator variances preserve this sign symmetry.
- **Independent edge generators:** each nonzero edge is its own generator `w_ab E_ab`. Arbitrary coordinate phases preserve the CP map, so a diagonal optimum exists.
- **Isotropic exterior creation:** signed permutation symmetry additionally makes one metric value per grade sufficient. Unequal variances generally invalidate this stronger grade reduction.

For multiple identical characters, only block diagonal reduction follows. For a generic tuple, label (11) a **restricted diagonal baseline** until an explicit symmetry proof is supplied. Orthogonal conjugation transports the allowed metric class and preserves the unrestricted optimum; retaining only matrices diagonal in the new coordinates changes the optimization problem.

### 4.5 Exterior grade reduction and scalar baseline

For the isotropic exterior algebra with `m` generators and `d=2^m`, write `q_j` for the grade-`j` metric, `j=0,...,m`. The exact gains in each grade are

\[
g_f(j)=\sum_{k=0}^{m-j}\binom{m-j}{k}\,k!\,\beta_k\,q_{j+k}/q_j,
\]

\[
g_b(j)=\sum_{k=0}^{j}\binom jk\,k!\,\beta_k\,q_j/q_{j-k}.
\tag{12}
\]

The conditioning program now has only `m+1` log variables. The log-determinant objective is `sum_j binomial(m,j) log(q_j)`. This can analyze an implicit large algebra without materializing its `2^m` coordinates; it does **not** establish that arithmetic on that entire algebra is locally affordable.

For the strongest simple baseline, optimize the scalar family `q_j=t^(2j)` using the exact gains, rather than fixing `t^2=1/L` or using only the conservative Frobenius bound (6). On canonical chain/exterior/edge families, gains are nondecreasing in `t`; bracket on `(0,1]` and bisect to the largest feasible value. On a general filtration, the same feasibility monotonicity follows by averaging phase conjugations with the Poisson kernel: the smaller-scale triangular product is an average of unitarily conjugated larger-scale products, and quadratic Jensen inequalities preserve both gain bounds. Include that argument in the proof record if extending the scalar solver beyond the canonical families.

For homogeneous degree-one generators, `T_t N_i T_t^-1 = t N_i`. This identity must be an attribution control. Omitting decoding changes the residual amplitudes; keeping decoding changes only coordinates.

### 4.6 Exact small oracles

**One square-zero edge:** for `N=sigma E_21`,

\[
Q_*=\operatorname{diag}(1,\min\{1,(C-1)/(L\sigma^2)\}),
\quad \kappa_* =\max\{1,L\sigma^2/(C-1)\}.
\tag{13}
\]

The determinant optimum is `1/kappa_*`. This tests scale normalization, both gain directions, the inactive-constraint case, and the SDP objective.

**Unit chain with `r=3`:** assume `L>=3`, `L>C-1`, and define

\[
K=C-1,\quad \gamma=\binom L2/L^2,\quad\delta=\binom L3/L^3,
\quad a=\frac{\gamma K}{\sqrt{\gamma^2+\delta\gamma K}+\gamma},\quad b=1/\gamma.
\]

If `a+b<=K`, then

\[
Q_*=\operatorname{diag}(1,a/L,ab/L^2,a^2b/L^3),
\quad\kappa_*=L^3/(a^2b).
\tag{14}
\]

To verify global optimality within the diagonal class, write successive scaled ratios `a=Lq_1/q_0`, `b=Lq_2/q_1`, `c=Lq_3/q_2`. Reflection and convex averaging in log coordinates allow `a=c`. The outer constraint is `a+gamma*a*b+delta*a^2*b<=K`; the middle constraint is `b+gamma*a*b<=K`. The product `a^2 b` is bounded by `a(K-a)/(gamma+delta*a)`. Its maximizing stationary point gives (14), and `a+b<=K` checks the middle constraint. Section 4.4 promotes this to the unrestricted optimum. Outside these conditions, use the convex program rather than extrapolating the formula.

This formula is a useful derived oracle. Its historical originality has not been established.

### 4.7 Validating optimization results and bounding uncertainty

Every solver output must be evaluated in the original mathematical problem. Retain solver status, primal/dual residuals, gain violations, conditioning, normalization error, and the objective bound. `optimal` is not a proof; `optimal_inaccurate` is not automatically unusable, but must pass the same independent checks.

Maintain three evidence labels:

1. **Analytically certified:** a formula/proof establishes the answer, with numerical evaluation error separately controlled.
2. **Numerically validated with bounds:** feasible candidate and a conservative objective interval have independently checked residual/error bounds.
3. **Exploratory candidate:** a solver result without a reliable objective enclosure. It cannot establish a small improvement or global optimality.

For diagonal programs, a practical lower bound follows from convexity. Let a known feasible scalar metric give `tau_0`; all improving solutions can be represented with `z in [-tau_0,0]^d` and `tau in [0,tau_0]`. For any nonnegative multipliers `lambda` of convex inequalities `g_i(x)<=0`, form `ell(x)=tau+sum lambda_i*g_i(x)`. Its tangent at an approximate solution is a global affine lower bound. Minimize that affine function over this box explicitly. Evaluate with outward error bounds to obtain a conservative lower bound on `tau_*`; a strictly feasible candidate supplies the upper bound. Near-stationary dual multipliers make this bound useful, but their mere existence does not guarantee tightness.

For the full SDP, the same principle can bound a dual residual. For PSD multipliers `Y1,Y2,Y3,Y4` of `Q-uI`, `I-Q`, `F(Q)`, `G(Q)`, put

\[
R=Y_1-Y_2+F^*(Y_3)+G^*(Y_4).
\]

An upper bound on `u_*` is

\[
\operatorname{Tr}(Y_2)+\max(0,1-\operatorname{Tr}(Y_1))+
\operatorname{Tr}(R_+),
\tag{15}
\]

using the redundant box `0<=Q<=I`, `0<=u<=1`. A feasible normalized metric supplies a lower bound on `u_*`; invert the interval to bound `kappa_*`. Reconstruct the linear adjoints from the actual model and test them by trace duality. Explicitly, `F*(Y)=(C-1)Y-sum_a A_a Y A_a.T`. If a symmetric backward-block multiplier `Z` is partitioned with index 0 for the leading block and indices `a` for factors, then `G*(Z)=(C-1)Z_00+sum_a Z_aa+sum_a(A_a Z_a0+Z_0a A_a.T)`. Do not treat approximate PSD multipliers as exact: account for their negative eigenvalues and arithmetic uncertainty, or retain the exploratory label.

An exact certificate sentinel is `N=E21, L=100, C=8`: choose `Y1=diag(0,1)`, `Y2=diag(7/100,0)`, `Y3=diag(1/100,0)`, `Y4=0`. These give `R=0` and upper bound `u_*=7/100`, attained by `Q=diag(1,7/100)`. Rational arithmetic can test the certificate path without trusting a nearly singular numerical dual solution.

High precision alone is not a formal interval certificate. For a claimed rigorous numerical bound, use outward-rounded interval arithmetic or exact rational bounds where applicable, and retain the enclosure calculations. If that path is unavailable, report numerical validation honestly and restrict hard optimality claims to proved oracles.

To repair slight infeasibility, mix the candidate with a strictly feasible metric from (6), using a stricter cap such as `sqrt(C)` there. Both LMIs are affine, so the mixture has a controllable feasibility margin. Renormalize afterward and report its actual objective. Scaling `Q` by a scalar cannot repair a gain violation.

The uncompressed moment check remains authoritative. If numerical uncertainty is comparable to the proposed effect, increase accuracy or label the comparison inconclusive; never count a violation as a gain.

For a deterministic minimization comparison, a numerically certified improvement requires **`upper_bound(candidate) < lower_bound(baseline)`** under the same cap and objective. The conservative absolute improvement is `lower_bound(baseline)-upper_bound(candidate)`. Favorable point estimates or individually narrow but overlapping intervals do not establish an improvement. Distinguish an enclosure of a returned metric's achieved objective from an enclosure of the optimum: the latter also needs a valid lower certificate. Report a feasible objective and its certified suboptimality gap separately. An exact analytic proof can establish a difference that numerical enclosures have not resolved.

## 5. Baselines, workloads, metrics, and confirmatory protocol

### 5.1 Claim hierarchy and competitive baseline hierarchy

The primary scientific claim is a **validated finite-depth distortion frontier** for the specified generator families. The primary engineering hypothesis is lower total cost for computing that same frontier. Numerical arithmetic utility is a separately preregistered extension.

| ID | Implementation and configuration | Role and reproduction check |
|---|---|---|
| B0 | `Q=I`, exact moment evaluation | Sanity control. Report infeasibility when the gain exceeds `C`; do not use an infeasible raw metric as the main competitive baseline. |
| B1 | Filtration metric from the conservative bound (6) | Reproduce the theorem's attainable order. Check both gains; label the bound conservative. |
| B2 | Best scalar filtration scale, exact gain bisection | **Strong simple baseline.** Share the moment kernel; use the largest validated feasible scale. Reproduce (13) and the tuned `r=3` value. |
| B3 | Arbitrary diagonal log-convex program (11) | Strong structured baseline. Global only with a symmetry certificate. Use one precise formulation and modern conic solvers, not a poorly tuned gradient heuristic. |
| B4 | Metric constant on each grade, or block diagonal on repeated-character subspaces | Attribution control and potentially strong competitor. Enforce only justified restrictions; anisotropy can break grade symmetry. |
| B5 | Full metric SDP (10), analytic or compressed Kraus factors, sparse block construction | Unrestricted small-instance reference. Apply common preconditioning, sparsity, and solver tuning. Validate against (13), (14), and orthogonal invariance. |
| B6 | Best generic formulation after **all available symmetry and algebraic reductions** | Strongest engineering comparator. In some families this is algebraically identical to the proposed specialized program. Equality is a valid outcome and rules out claiming a new optimizer advantage. |
| D1 | `scipy.linalg.matrix_balance` on a specified aggregate, with permutation/scaling settings logged | Diagnostic only. This eigenvalue-oriented routine can isolate triangular blocks and leave them unscaled; beating it is insufficient. [SciPyBal] |
| D2 | Residual amplitudes scaled by `L^(-1/2)` in Euclidean coordinates | An alternative physical computation, not a same-function comparator. Shows the consequence of relaxing the fixed-amplitude assumption. |
| N1 | Float32 and float64 direct sparse coefficient recurrences | Required only for the arithmetic extension. Same coefficient data, order conventions, and output interface. |
| N2 | Binary grade scaling, adaptive exponent management, and compensated evaluation where applicable | Strong range/roundoff alternatives. Count exponent metadata and all conversions. |
| N3 | Independent normalized Taylor implementation, and final-output algorithms appropriate to the task | Use TaylorSeries.jl or supported JAX `jet` primitives if needed; compare product-tree or power-sum/Newton-identity algorithms when only final polynomial coefficients are requested. [JAXJet] [TaylorSeries] |

The proposed implementation is primarily a model-specific dispatcher and exact moment/reduction layer over B2–B6. Its novelty is not assumed. A speed comparison only against an unreduced dense SDP establishes the benefit of exploiting structure relative to that formulation; it does not establish superiority over known symmetry-aware optimization.

B2-versus-B3/B5 comparisons test **H2** and report both achieved distortion and cost. They do not establish **H3** unless both methods satisfy the same independently specified optimum-accuracy contract. The primary H3 contrast is the specialized implementation versus B6 with the same admissible metric class, information, and certificate requirements. A cheap scalar answer and an accurate unrestricted optimum are different outputs when their objective gap exceeds that contract.

For B5/B6, pilot both Clarabel and SCS with the same externally validated accuracy requirement. Select a solver/configuration by a frozen rule, such as lowest median total time among configurations meeting the requirement on all pilot correctness cases. If a small baseline cannot be solved faithfully, retain it as an unresolved reference and use analytic oracles and a validated reduced formulation for the narrower claim. Do not replace it with an intentionally weak solver.

### 5.2 Workload construction and provenance

No external dataset or pretrained asset is necessary for the core mathematical claim. Use generated exact structures with a recorded schema, generator coefficients, source revision, and license chosen for the new code/data. Synthetic evidence establishes properties of these workloads only.

Start with the following **structural coverage cases**, not a leaderboard selected after observing gains:

| Family | Construction and coverage purpose | Ground truth / symmetry |
|---|---|---|
| Zero and one-edge cases | `N=0`; `N=sigma E_21`, including active and inactive gain constraints | Exact (13), handling `r=0`, `L=0`, and zero coefficients. |
| Single chains | One weighted subdiagonal shift of size `r+1`; include unit, unequal positive, and mixed-sign nonzero weights | Exact depth `r`, `f_j=r+1-j`, `nu=r(r+1)/2`; diagonal optimum certificate. |
| Exterior creation | Bitmask basis, signed wedge multiplication; include isotropic and unequal generator strengths | `d=2^m`, `r=m`, `f_j=sum_(k=j)^m binomial(m,k)`, `nu=m*2^(m-1)`; sign symmetry, plus grade symmetry only when isotropic. |
| Independent-edge DAGs | A chosen acyclic graph; one weighted matrix unit for each edge | Depth is longest directed path; image levels and diagonal symmetry known from construction. Branching probes costs not characterized by depth alone. |
| General filtration-respecting tuples | Block matrices with blocks only from lower to higher levels, including some level-skipping blocks | Joint nilpotence structurally guaranteed; exact `r` requires a surviving word. Diagonal optimality is not presumed. |
| Orthogonal conjugates | Apply the same recorded orthogonal transform to all generators | Same unrestricted objective and gain geometry; nuisance replicas, not independent problem instances. Tests basis handling. |
| Nonorthogonal conjugates | Apply a known invertible rational transform | Same algebraic depth, potentially different Euclidean distortion constants. Label the changed physical coordinate convention. |

The first implementation cases are `r=1`, `r=3`, and exterior `m=3`: they supply an exact frontier, a nontrivial finite-depth oracle, and the seed's noncommutative example. Add `r=2` as a moment-order check, then increase size to resolve a specific scaling or memory question. These are starting cases, not arbitrary caps on investigation.

The default randomized confirmatory population is fully specified below. It is a chosen synthetic population, not a measured real-world distribution. Amend it before confirmation if the scientific claim demands a different population; retain the reason and do not choose it by looking for favorable protected outcomes.

| Field | Default construction / weighting |
|---|---|
| Families | `chain`, `edge_dag`, and `block_tuple`, each with population weight `1/3`. |
| Weight anisotropy | `a=0` and `a=2`, each weight `1/2` within family. Every nonzero weight is an independent equiprobable sign times `2^u`, with integer `u` uniform on `[-a,a]`. The `a=2` stratum spans a 16-fold magnitude ratio. |
| Starting size tiers | Nilpotent depth `r` from the ordered list `[2,3,4,6]`. Chains have one vertex per grade; DAGs and block tuples have two. Vertex order is `(grade,within_grade)` in lexicographic order, grades `0,...,r`. |
| Resource-based tier selection | Before inspecting comparative objective or speed effects, retain the largest common contiguous prefix of those tiers supported by the baseline feasibility/resource calibration in section 6, across all six family/anisotropy combinations and primary horizon cells. Every retained tier has equal population weight. The resource policy and retained prefix are frozen. A baseline accuracy/resource failure must be reported; if no common tier works, the mixed-population claim is blocked. A narrower family-specific claim requires an explicit amended population. |
| Chain tuple | One generator with the `r` nonzero adjacent-grade subdiagonal weights drawn as above; `d=r+1`, `s=1`. |
| Edge-DAG tuple | Include the backbone edges `(j,0)->(j+1,0)` for all `j<r`. For every other ordered vertex pair whose target grade exceeds its source grade, independently include its edge with probability `1/2`. Each included edge is a separate weighted matrix-unit generator; number of generators equals number of edges. Enumerate edges by source vertex then target vertex. The backbone proves depth exactly `r`. |
| Block-filtration tuple | `d=2(r+1)`, `s=3`. Generator 0 contains only the weighted backbone edges `(j,0)->(j+1,0)`. Generators 1 and 2 independently include each permitted strictly grade-raising entry with probability `1/2`, with independent weights when included. Retain an all-zero extra generator if drawn. Generator 0 raised to power `r` is nonzero, so the exact depth is known without a numerical rank test. |
| Noise and amplitudes | Identity covariance; constant amplitude 1. No Monte Carlo coefficient law is needed for exact moments. |
| Primary cells per tuple | The four horizons `[max(1,floor(r/2)), r, 10*r, 100*r]`, all at `C=2`, in canonical physical coordinates. Each horizon receives weight `1/4`. Use the arithmetic mean of paired log-time ratios within each cell, then the equal-weight mean across its four cells, as specified in section 5.6. |
| Sensitivity cells | `C=8` at the same horizons; exact rational orthogonal conjugates; optional additional caps or larger horizons with a named question. These are secondary unless explicitly included in a new primary protocol. They do not add independent tuples or change primary weights. |
| Stratum weights | For `T` retained tiers, each `(family,a,r)` stratum has weight `1/(6*T)`. Use equal planned independent-tuple counts per stratum by default; pilot precision calibration may choose a different allocation only if recorded before confirmation. Population weights remain unchanged. |

Freeze tier membership from structural coverage and measured baseline resource/accuracy feasibility, not candidate-versus-baseline advantage. A size or anisotropy range deliberately selected because a pilot suggests a crossover is a separate targeted population and needs fresh confirmation. The listed tiers are starting coverage, not a prohibition on investigating larger sizes when they resolve a specific question. Stop an exploratory horizon expansion when its question is resolved or faithful validation is unavailable; never stop a frozen primary sequence because its results are unfavorable.

**Deterministic generation contract.** Store a 32-byte master seed as 64 lowercase hexadecimal digits. For a tuple, derive its seed as `SHA256(master_seed_bytes + ASCII("family=F;a=A;r=R;index=I"))`, substituting the table's family name and nonnegative decimal integers without leading zeros. For an integer draw in `[0,n)`, hash `tuple_seed_bytes + b"\0" + label.encode("ascii") + b"\0" + counter.to_bytes(8,"big")`, starting the counter at zero; interpret the digest as an unsigned big-endian 256-bit integer, reject values at least `floor(2^256/n)*n`, and otherwise return its remainder modulo `n`. Increment the counter only on rejection. Use independent labels `gG.srcS.dstD.include`, `.sign`, and `.exponent`, where `G` is the generator index (0 for the single chain or the DAG edge-construction stream), and `S,D` are flattened vertex indices. Draw inclusion with `n=2` and include when the result is 1; sign with `n=2` mapping 0 to -1 and 1 to +1; exponent with `n=2*a+1` then subtract `a`. Backbone inclusion is deterministic and needs no draw. For DAGs, assign final generator indices after sorting the included edges. Exact weights are stored as sign/exponent pairs, not decimal approximations. Serialize the descriptor as UTF-8 JSON with sorted keys, compact separators, integer fields, and a final newline; hash those bytes. Add golden descriptors/hashes so two independent implementations can reproduce identical instances without relying on an unspecified RNG library default.

Generate canonical structures from integer or rational coefficients and retain the structural certificate. For small unknown rational tuples, use exact subspace-rank recursion from section 2.4 and inspect a surviving longest word; avoid exhaustive word enumeration. **A basis transform must have authoritative exact semantics.** For invariance sentinels, use rational orthogonal Householder matrices `U=I-2*v*v.T/(v.T*v)` with recorded nonzero integer `v`, or products thereof, and reconstruct the exact descriptor at validation precision. The simplest fixed sentinel is `v=(1,2,...,d)`. Evaluate transported canonical moments and bound errors in floating realizations. Exact similarity, not a rounded SVD threshold, supplies nilpotence. For nonorthogonal rational transforms use their exact inverse, not their transpose.

If raw rounded transformed generator arrays are instead declared to define the stochastic model, treat them as potentially **near-nilpotent**: use the full untruncated finite-horizon recurrence and do not assert exact truncation at `r` or exact orthogonal invariance. A float64 QR factor followed by `U@N@U.T` is not an exact orthogonal similarity over the represented numbers. Independently rotated rounded Kraus factors also define a finite CP map that need not equal powers of a rounded transformed generator. Appendix A3 is explicitly the latter numerical perturbation diagnostic, not a certification of exact nilpotence for rounded arrays.

### 5.3 Primary and secondary metrics

For every instance and cap, report:

- `g_f`, `g_b`, gain violation, and gain-evaluation uncertainty; an invalid candidate has no competitive score.
- `kappa(Q)`, `log2(kappa(Q))`, and an objective interval if available.
- `logdet(Q)` at the normalized determinant optimum, separately from conditioning minimization.
- Lower bounds (2), (3), upper constructive bound (6), and the gap to B2/B5/B6.
- Total wall time in seconds, with construction, canonicalization, solve, and independent validation broken out.
- Peak process memory in bytes/MiB, including canonicalized cone data and solver workspace where measurable.
- Failure/inaccuracy rate and censored resource failures; never omit these from aggregate tables.

For a baseline `b` and candidate `p`, distinguish absolute conditioning difference, relative change `1-kappa_p/kappa_b`, and log change `log2(kappa_b)-log2(kappa_p)`. The coordinate-map log-condition change is half the last quantity. Tiny effects are legitimate mathematical findings when reliably resolved; their practical value is a separate question.

For engineering comparisons, the primary outcome is the per-instance **log ratio of total validated completion time**, `log(time_baseline/time_candidate)`, at a frozen gain/objective-accuracy requirement. Report absolute seconds as well. Include the cost of recovering a filtration or symmetry when it is not supplied by the input specification. A method that fails validation is not awarded a fast time.

The default primary timing analysis is deliberately conservative: a confirmatory aggregate speed claim requires **every scheduled primary tuple/horizon cell for both methods to complete validly** under the frozen contract and resource policy. If any such unit fails, the full-population log-speedup claim is unresolved. Report all completion rates and causes, and label any joint-success timing summary descriptive. Do not drop unsuccessful pairs, renormalize weights, assign a fast invalid completion, or redefine the population after outcomes are visible. A narrower follow-up population must be defined by structural/input characteristics, frozen, and tested on fresh instances. A failure-aware capped-time endpoint is an alternative only if justified and preregistered before confirmation; it is not the default here.

For a numerical extension, the primary metric must be chosen from an application contract before confirmation: for example, total time to return all requested coefficients within a componentwise tolerance. Report absolute error, error normalized by a declared per-degree scale, overflow/underflow count, and total memory. Do not use relative error alone for coefficients near zero, or call an underflowed omitted coefficient an accurate zero.

Every accuracy-qualified arithmetic score needs a trustworthy reference or independently valid error enclosure for **every requested coefficient of that scored instance**. If reference cost permits only a subset, freeze that subset before viewing outcomes and define the accuracy-qualified population as that subset. Other cases may support throughput or range diagnostics, but cannot count as accurate completions. Reference construction is common evaluation infrastructure and excluded consistently from method runtime; method-specific certification needed to deliver the promised answer belongs inside the completion-time contract. Do not charge common offline checking to only one competitor.

### 5.4 Evaluation before implementation choices

Freeze the following semantics at the start of the harness:

1. Same generator tuple, coefficient moments, horizon, gain cap, physical basis, and metric objective for every direct comparison.
2. Same exact moment coefficients; same covariance whitening; same analytic sparsity and Kraus simplifications when applicable.
3. Same numerical acceptance test. Tune solver tolerances to achieve that test; identical tolerance numbers across different formulations are not automatically fair.
4. Same permitted knowledge. A supplied grading/symmetry is available to all methods. An inferred grading/symmetry incurs measured discovery cost for any method using it.
5. Same endpoint or prefix-output contract in arithmetic tasks. Do not demand online output solely to exclude a stronger final-output algorithm.

Do not match FLOPs artificially when the value proposition is reduced computation. Instead compare equal mathematical output quality and report total resources. For an arithmetic precision comparison, include both equal-precision and equal-error views; float64 is a legitimate practical competitor even if it uses more bytes per entry.

### 5.5 Exploratory versus confirmatory work

All measurements in this document are development evidence. Keep them in `runs/exploratory/`; do not reuse their cases as independent confirmation of a selected effect.

Before confirmation, commit `protocol/confirmatory.yaml`, a method registry, dependency lock, instance-distribution specification, sample-size calculation, accuracy requirements, primary contrast, secondary analyses, resource rules, and exclusion rules. Then generate fresh confirmatory seeds and instance files. Record their hashes. Do not inspect protected outcome metrics until the implementation and analysis are frozen.

If a new failure changes the hypothesis, population, primary metric, baseline, or interpretation, write an amendment with the reason and timing. Use a fresh independent confirmation set for the changed claim. A correctness bug may be fixed, but invalidate affected prior results and record the new code hash rather than silently replacing them.

### 5.6 Appropriate inference and sample-size calibration

**Mathematical identities and exact finite cases:** use algebraic proof, exact enumeration, and numerical error bounds. A regression slope near `r` is a useful implementation check, not a proof of asymptotic optimality. Do not attach statistical significance to an exact deterministic difference.

**Random-instance generalization:** the independent unit is a newly generated generator tuple, with all its horizons, caps, and basis transforms grouped together. Multiple depths, repeated timings, and coordinate rotations of the same tuple are dependent observations. Use section 5.2's four equally weighted primary cells within each tuple, then average tuple effects within each stratum and combine using the frozen population weights. Bootstrap tuples within strata and recombine with those same weights; neither realized counts nor failures change them. The all-scheduled-units completion rule in section 5.3 applies before a positive aggregate speed claim.

The exact default timing estimator is

\[
d_{ih}=\frac1{B_{ih}}\sum_{b=1}^{B_{ih}}
\log\frac{t_{\mathrm{baseline},ihb}}{t_{\mathrm{candidate},ihb}},
\qquad d_i=\frac14\sum_{h=1}^{4}d_{ih},
\qquad \widehat\mu=\sum_jw_j\frac1{n_j}\sum_{i\in j}d_i.
\]

Here `i` is an independent tuple, `h` its horizon, `b` a paired timing block, and `j` its stratum. Report `exp(mu_hat)` as a **geometric speedup**, with transformed interval endpoints. It is not the ratio of expected total-work times. Absolute times and aggregate catalog costs are separate descriptive metrics. Taking the logarithm of arithmetic mean times is a different estimand and is not the default.

**Timing noise:** first measure pilot paired log-time differences with randomized or alternating method order, using enough blocks to estimate both within-instance timing variability and between-instance variability. For a fixed-sample primary mean log-speedup, select the number of independent tuples from the pilot standard deviation `s` and a prespecified desired confidence half-width `h`, approximately `n=(1.96*s/h)^2`, then round upward and verify by simulation or a suitable small-sample interval calculation. Estimate paired repeats from the within-instance variance so they contribute a minority of the desired uncertainty. Do not substitute more timing repetitions for more independent tuples.

For the stratified default, use the actual weighted variance rather than treating all cells as independent: with stratum weights `w_j`, tuple-level pilot standard deviations `s_j`, and planned counts `n_j`, the approximate variance is `sum_j(w_j^2*s_j^2/n_j)`. With equal counts, start from `n_per_stratum=ceil((1.96/h)^2*sum_j(w_j^2*s_j^2))`, then increase as required by the selected small-sample interval/simulation and pilot uncertainty. Record the final integer allocation and bootstrap resampling scheme before generating confirmation. A zero observed pilot variance is a reason to check resolution and variability, not to plan zero replication.

Choose `h` from a real decision: e.g., the precision needed to distinguish whether preprocessing amortizes over the anticipated number of uses, or to resolve the measured pilot effect. Document its rationale. It is an uncertainty target, not an invented minimum percentage improvement. If no plausible use frequency exists, report the frontier with confidence intervals and refrain from a practical-benefit claim.

Use one fixed confirmatory sample size; no repeated significance peeking. A paired tuple-level bootstrap or paired-mean interval can quantify workload-population uncertainty, with its method fixed in advance. For a fixed finite catalog, report the complete catalog and timing-block uncertainty without pretending the catalog is a random sample of all algebras. One primary contrast needs no multiplicity correction; multiple confirmatory claims require a declared familywise procedure such as Holm correction or an explicit hierarchical testing order. Exploratory subgroup findings remain exploratory.

A confidence interval spanning no effect is **inconclusive** unless narrow enough to exclude effects that would change the stated engineering decision. A validated deterministic lack of improvement on a particular instance is evidence about that instance, not a universal negative theorem.

### 5.7 Timing and measurement boundaries

The core is CPU-only and synchronous. Warm imports, solver initialization, and the relevant code path before steady-state measurements. Measure cold end-to-end runs separately. Do not mix cached preprocessing on one method with cold preprocessing on another.

Measure from input-instance loading or already-loaded arrays according to a frozen boundary. Include structure discovery, moment construction, cone canonicalization, solving, output normalization, and validation for the main engineering metric. Publish component times to explain the result. Record BLAS implementation, thread count, solver linear-system backend, power source, low-power mode, and memory pressure. Tune single-thread versus native threaded execution in the pilot and apply the selected policy fairly.

If the frozen delivery procedure retries, raises precision, or repairs a metric before returning a valid answer, include all of those costs from the first attempt through valid completion. An unsuccessful attempt does not disappear when a later attempt succeeds. Never average only successful scheduled timing repeats: a repeat that remains unsuccessful under the frozen procedure makes its cell incomplete under the primary completion rule. Record the retry policy and deadline in advance and apply the same completion contract to all methods.

For very short operations, measure a batch whose duration dominates timer resolution, but retain the batch as one timing observation. Randomize method order within blocks; investigate drift rather than selectively deleting slow trials. Record all raw timings and a predeclared exclusion reason for system interruptions. If an optional GPU path is later added, explicit synchronization is required and compile/transfer costs must be separated; no GPU path is needed to validate this plan.

### 5.8 Mandatory attribution and harness controls

- Duplicate a method under two labels; it must not produce a systematic quality or timing advantage after measurement uncertainty.
- Check exact Rademacher enumeration on a tiny tuple against the CP recurrence.
- Check the rank-one oracle, the finite-depth chain oracle, and the exterior values in section 10.
- Check transformed versus original coordinates, including encoding and decoding.
- Compare full SDP and exact diagonal/grade reduction on certified families; reject a discrepancy beyond numerical bounds.
- Break grade symmetry with anisotropic generators; grade-only equality must not be assumed.
- Test forward-only versus both constraints, conditioning versus determinant objectives, and exact finite-depth versus conservative bounds.
- Share sparse kernels, log-domain coefficients, caching, and analytic Kraus compression across competitors.
- Reproduce the binary-scaling negative control before any accuracy claim.
- Add a known deliberately gain-violating metric; the validator must reject it, including in a rotated basis.
- Make a candidate return immediately with invalid outputs on the slowest cases; the report must refuse a full-population speed-win label.
- Supply favorable point estimates with overlapping objective intervals; the report must refuse a certified objective-win label.
- In the arithmetic branch, corrupt only the highest requested coefficient; the instance must fail its accuracy-qualified completion test.

## 6. M3 Max feasibility and hardware-adaptive execution

### 6.1 Hardware discovery is the first executable gate

Do not assume the maximum memory configuration. Apple's M3 Max specifications include materially different configurations; the actual machine controls admissible workloads. [AppleM3]

The following are existing macOS commands, not project scripts:

```bash
uname -m
sw_vers
sysctl -n hw.memsize
sysctl -n hw.physicalcpu
sysctl -n hw.logicalcpu
sysctl -n machdep.cpu.brand_string
sysctl vm.swapusage
vm_stat
df -h .
pmset -g batt
```

Record only relevant hardware fields, not machine serial numbers or unrelated personal paths. Confirm native `arm64` Python and detect Rosetta translation if applicable. Record free storage, available memory, current swap, power mode, and numerical-library BLAS configuration. If a command is unavailable, capture that fact and use Python/platform or Activity Monitor observations rather than inventing a value.

### 6.2 Software stack and verified support

Use a native Python 3.12 virtual environment initially, NumPy/SciPy CPU float64, CVXPY, Clarabel, SCS, and mpmath for difficult small checks. JSON/YAML configuration, pytest for essential correctness tests, and a lightweight process-memory monitor complete the core. No CUDA, Metal training path, neural model, or specialized physical equipment is required.

Versions actually used in the small Linux checks were Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0, CVXPY 1.9.3, Clarabel 0.11.1, SCS 3.3.1, and mpmath 1.4.1. Keep those versions in a separate reproduction environment when reproducing the existing numbers. This is not a claim that the combination was installed on an M3 Max. For new competitive measurements, start with the newer verified NumPy 2.5.4 and SciPy 1.18.1 releases; their Python 3.12 arm64 macOS wheels were listed on the inspected package records. Rerun the sentinels before using the new stack, and give every comparator the same validated environment. [NumPyRecord] [SciPyRecord] [NumPyCurrent] [SciPyCurrent]

Public records also list a Python 3.12 universal2 CVXPY 1.9.3 wheel and an arm64 Clarabel 0.11.1 wheel. Official solver documentation lists SDP and exponential-cone support. Documentary wheel availability is not an executed target-platform test. [CVXPYInstall] [CVXPYSolvers] [CVXPYPyPI] [ClarabelPyPI]

SCS's current package documentation distinguishes platform backends: Linux wheels can use MKL while macOS defaults differ and an Accelerate option is described. Do not extrapolate Linux solver timings to the Mac, or silently compare different backends. Verify the actual installed API before using backend-specific settings. [SCSPyPI]

If native `python3.12` is absent, install a maintained native CPython through the machine's existing trusted package manager, or select a currently distributed universal2 installer from Python's official macOS downloads. If using a different supported minor version, substitute its executable below only after checking wheel compatibility, then record the change. Do not require an obsolete installer merely to match the exploratory interpreter. [PythonMac]

Existing setup commands, executed in a new project directory:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy==2.5.4 scipy==1.18.1 cvxpy==1.9.3 clarabel==0.11.1 scs==3.3.1 mpmath==1.4.1 pytest pyyaml psutil threadpoolctl
python -m pip check
mkdir -p environment
python -m pip freeze > environment/requirements.lock.txt
```

Allow the package resolver to install compatible dependencies; do not copy the investigation's temporary manual installation procedure. If a pin lacks a native compatible wheel, first select a supported Python version with wheels; otherwise select the nearest supported release without changing the mathematical interface, record the reason, and rerun all oracles. Freeze the resulting versions and wheel/source hashes before confirmation. Do not install an Intel-only environment and report it as native Apple Silicon. Once the proposed `pyproject.toml` and package entry files in section 9 exist, run `python -m pip install -e .` and refresh the environment lock before invoking `python -m nilp_poc`.

Test the exact required operations: symmetric eigensolvers, Cholesky and solves in float64, sparse matrix multiplication, an exponential-cone problem, a small PSD-cone problem, and `log_det` if the determinant objective is enabled. A successful import or advertised Apple Silicon support is insufficient. Clarabel and SCS must each pass the relevant tiny formulation before being used as a reference. General solver availability does not guarantee resolution of very ill-conditioned metrics.

CPU float64 is deliberate: the indispensable objects are small, sometimes severely conditioned matrices and convex optimization problems. Reduced precision would change the evidence quality. The M3 GPU is not assumed to support the needed solver or precision path; using the CPU is a faithful implementation, not a weakened scientific proxy.

### 6.3 Dependencies and licenses

The core uses openly licensed software: NumPy/SciPy (BSD-family core licenses, with additional bundled notices), CVXPY and Clarabel (Apache-2.0), SCS (MIT), and mpmath (BSD-family). Verify the exact installed distributions' license files and retain notices if redistributing code or binaries. Distribution license metadata were inspected; the NumPy wheel's combined license expression also identifies bundled components. [NumPyCurrent] [SciPyCurrent] [CVXPYPyPI] [ClarabelPyPI] [SCSPyPI] [MpmathRecord]

No model-weight license, dataset account, or proprietary solver is required. Do not make MOSEK, MATLAB, or a paid cloud service a mandatory reference. TaylorSeries.jl's MIT license and the `torch-jet` MIT code are relevant only if their conditional tasks are activated. A downloadable paper is not automatically permission to redistribute it; store bibliographic links and permitted metadata rather than bundling publishers' PDFs indiscriminately.

### 6.4 Memory model

Use actual dtype widths and sparse nonzero counts. Before constructing a solver model, estimate:

\[
M_{\rm peak}\approx M_{\rm Python/libs}+M_{\rm generators}+M_{\rm moments}
+M_{\rm factors}+M_{\rm canonicalized}+M_{\rm solver}+M_{\rm validation}.
\]

Material contributors are:

| Component | Storage estimate / concern |
|---|---|
| Dense generators | `8*s*d^2` bytes in float64; sparse edge/shift representations can be much smaller. |
| Dense CP moment recurrence | A few `d*d` buffers plus generators; do not retain every level unless needed. |
| Choi construction | At least `8*d^4` bytes per dense matrix; eigendecomposition and copies multiply this. |
| Kraus factors | `8*K*d^2` bytes if dense, with `K<=d^2`; canonical examples have much smaller `K`. |
| Backward LMI | Cone order `p=d*(K+1)`. A dense copy alone costs `8*p^2` bytes; solver factorizations and modeling copies can dominate. Sparse arrow structure and chordal decomposition may reduce actual use. |
| Diagonal log program | `d+1` variables and terms proportional to `nnz(H-I)`; `H` can become denser than `M`. |
| Exterior grade program | `m+1` variables and `O(m^2)` terms without materializing `2^m` states. |
| Sample-path simulation | Approximately two state buffers `2*B*d*itemsize`, plus generator/coefficient chunks and recorded outputs. Stream trajectories; do not store all `B*L*d` states. |
| High precision | Object overhead and precision-dependent storage; measure actual small-case RSS rather than treating an 80-digit number as an 8-byte float. |

Illustrative **storage calculations, not measured peak memory**:

- Exterior `m=3`: `d=8`, analytic `K=7`, so the backward block is `64x64`, about 32 KiB as one dense float64 array.
- Exterior `m=4`: `d=16`, analytic `K=15`, block order 256, about 0.5 MiB per dense array.
- Generic `d=16` using the conservative `K=d^2` bound: block order 4,112, about 129 MiB per dense array before solver workspace.
- Generic `d=32` at that bound: block order 32,800, about 8.0 GiB per dense array. Several copies and factorization can exceed an ordinary laptop's available memory.

The core oracles and small canonical SDP comparisons are consequently plausible on even the smaller M3 Max memory configurations. This is an estimate based on the formulation sizes, not a measured Mac guarantee. Large generic full SDPs are not the core workload.

### 6.5 Throughput calibration and sizing rules

Run the tiny correct cases first and measure wall time, peak RSS, cone dimensions, nonzeros, iterations, and backend. Increase one structural dimension at a time. For dense moment evaluation, an initial operation model is `O(r*s*d^3)`; sparse multiplications are closer to `O(r*sum(nnz(N_i))*d)`. Solver cost must be learned from actual cone data and measurements rather than an asserted universal cubic law.

Fit a conservative local scaling envelope from adjacent valid sizes. Predict the next job's memory and time with uncertainty, then measure it. Choose the largest next case that resolves an outstanding claim while leaving memory for the OS, validation, and observed variability. An initial operational policy may reserve at least 25% of currently available memory and avoid concurrent solver jobs; revise that reserve from measured memory pressure. This is protection against swapping and invalid timing, not a scientific upper limit on experiment count or depth.

For each stage, budget total time as number of instances times measured construction/solve/validation cost, with observed variance and planned replication. Report the forecast before a long batch. Longer runs are acceptable when they resolve a material uncertainty. Do not launch a broad Cartesian sweep before the calibration and baseline checks pass.

Use analytic/grade reductions to explore large implicit horizons or grades when the claim is about the metric frontier. If the claim is about actual arithmetic, size the explicit state representation and simulation separately. An implicit `m=40` exterior metric calculation does not demonstrate feasible computation with `2^40` explicit coefficients.

### 6.6 Faithful fallbacks, interruption, and recovery

- If a conic package fails to install, resolve supported native wheels or use the other validated open-source solver. Exact formulas and the proof checks can still run. A SciPy local optimizer may produce exploratory diagonal candidates, but does not silently replace a validated global reference.
- If a full SDP becomes numerically unreliable, first use a known basis transformation, better scaling, analytic factors, a second solver, and higher-precision checking. When changing variables `Q=T0.T X T0`, transform the bounds on **Q** too; imposing `uI<=X<=I` changes the original objective.
- If memory is insufficient, reduce full-oracle dimension, use a proved exact structural reduction, or restrict the claim. Do not delete high-grade modes, silently quantize, or substitute a different algebra.
- If exact structure is unavailable, label rank-sensitive cases uncertain. Near-nilpotence is a robustness experiment, not permission to truncate the model.
- Checkpoint at instance and timing-block boundaries. Save immutable input/configuration hashes, solver logs, the last validated metric, raw results, and completed-unit identifiers. An interrupted solver can be restarted; do not claim exact solver-state resumption unless its API actually supports it.
- Resume only when code, dependency lock, instance hash, precision, and measurement boundary match. Otherwise start a new run linked to the previous run. Write results atomically and retain failed and interrupted records.

No local resource failure is evidence against the asymptotic theorem. It can, however, limit the engineering usefulness of the generic optimization approach and must be reported as such.

## 7. Ordered work packages and decision gates

The coding agent may make routine, reversible implementation choices without seeking further permission. Changes to the hypothesis, physical semantics, evaluation population, primary metric, baseline fairness, or confirmation procedure require a recorded methodological amendment. All progress and decisions belong in project files, so execution can resume without this conversation.

The order is **WP0 → WP1 → WP2 → WP3 → WP4 → WP5 → WP6 → WP7 → WP9**. WP8 is a separately gated branch after WP4 or WP7; it receives its own freeze and confirmation before any application claim. Do not postpone an invalidating theory or harness check until after a large implementation.

### WP0 — Discover the machine, dependencies, and admissible claims

- **Objective/risk:** resolve actual local capacity, numerical support, and licensing before dependent work.
- **Entry/dependencies:** this document and a fresh target laptop; no unavailable notebook or dataset is required.
- **Actions:** create the repository layout and importable package entry files in section 9; run hardware discovery; install a native environment and `python -m pip install -e .`; run `pip check`; capture BLAS and solver backends; execute tiny eigensolver, Cholesky, sparse, exponential-cone, PSD-cone, and optional log-determinant cases. Write the selected claim categories and scope in `protocol/scope.yaml`.
- **Files/commands:** `environment/hardware.json`, `environment/capabilities.json`, dependency lock, `LICENSES.md`; proposed `python -m nilp_poc inspect` and `python -m nilp_poc verify --suite capabilities`.
- **Tests/outputs:** finite float64 outputs, solver solutions matching elementary analytic values, native architecture recorded, no missing packages or silent unsupported precision path.
- **Resources:** small matrix operations and dependency downloads; measure base RSS and free storage. No large assets.
- **Acceptance/next action:** pass → WP1. Installation or operation failure → use the faithful fallback in section 6 and rerun. If no validated conic path is available, continue the proof/oracle/negative-control work but label the optimization comparison blocked; do not replace it with unsupported claims.

### WP1 — Reconstruct the mechanism and its boundary conditions

- **Objective/risk:** verify H1 and catch confusion about joint nilpotence, transposes, metrics, and physical readout.
- **Entry/dependencies:** WP0; section 2 equations and Appendix A.
- **Actions:** implement chain and exterior generators; implement exact moment recurrence and independent short Rademacher enumeration; record `r,f_j,nu`; reproduce the proof in `theory/metric_costs.md`; implement the rank-one and `r=3` analytic oracles; reproduce binary scaling equivariance. Add the non-jointly-nilpotent counterexample in section 8.
- **Files:** `algebra.py`, `moments.py`, `oracles.py`, `tests/test_algebra.py`, `tests/test_moments.py`, `tests/test_oracles.py`, `theory/assumptions.md`.
- **Tests/outputs:** exact structural zero words; a nonzero longest word; exterior signs and anticommutation; agreement of enumeration and moments within an explicitly computed roundoff budget; Appendix A values; identical decoded bits in the matching normal-range control.
- **Resources:** tiny CPU cases; enumeration only for a small number of sign variables, with the count computed before execution. Exact polynomial moments handle large `L` cheaply.
- **Acceptance/next action:** agreement and complete proof → WP2. Disagreement → distinguish code error, floating error, and assumption failure using exact small arithmetic. A genuine counterexample to the stated theorem stops the current claim and produces a documented mathematical correction, not a larger experiment.

### WP2 — Build the evaluation harness and reproduce strong baselines

- **Objective/risk:** ensure a positive result cannot be created by a weak baseline or invalid gain check.
- **Entry/dependencies:** WP1; shared interfaces and schemas from section 9.
- **Actions:** implement B0/B1/B2/B3/B5, analytic Kraus factors, independent gain validation, objective-bound bookkeeping, and timing/RSS capture. Implement CVXPY Schur blocks exactly as (9). Test sparse versus dense construction on small cases. Pilot both installed conic solvers. Add equality, rotated-basis, deliberately infeasible, and zero-case controls.
- **Files/commands:** `metrics.py`, `solvers/full_sdp.py`, `solvers/diagonal.py`, `solvers/scalar_grade.py`, `validation.py`, `bench.py`; proposed `verify --suite baselines` and `solve --config configs/oracles.yaml`.
- **Tests/outputs:** B2/B3/B5 reproduce (13); B3/B5 agree with (14) within validated error; orthogonal transforms preserve the true optimum; the rotated infeasible example is rejected. Store raw residuals and dual data, not only a pass flag.
- **Resources:** use `d=2,4,8` structural oracles before any generic large cone. Calibrate actual memory/model-building overhead.
- **Acceptance/next action:** faithful baselines and validator → WP3. If a baseline is numerically unresolved, repair scaling/precision or narrow the comparison; do not promote another method using that failure as an unqualified win.

### WP3 — Implement the minimum structural specialization

- **Objective/risk:** test H2/H3 without inventing a new optimizer or applying an unjustified restriction.
- **Entry/dependencies:** WP2; explicit diagonal/grade/block symmetry certificates.
- **Actions:** implement diagonal-preservation tests, the `M,H` construction, exterior grade formulas (12), and a dispatcher that selects only proved exact reductions. Use the same moment data and solver engine for candidate and baselines. Implement independent-edge and block-filtration generators to test beyond a single chain. Add the generic reduced baseline B6, including any simplifications learned here.
- **Files:** `structure.py`, `solvers/grade.py`, `solvers/dispatcher.py`, `theory/symmetry_certificates.md`, `tests/test_reductions.py`.
- **Tests/outputs:** transformed moment maps reconstruct the original maps; grade/diagonal/full objectives agree where proved; anisotropy disables only the unjustified grade reduction; non-symmetric tuples remain unrestricted or explicitly restricted.
- **Resources:** moment and cone sizes predicted before construction. Avoid word enumeration and `2^m` state materialization where an exact implicit formula exists.
- **Acceptance/next action:** correct specialization → WP4. If B6 is identical and equally efficient, record “known formulation suffices”; continue only the finite-depth geometry or diagnostic question that remains. This is not permission to relabel the same method as novel.

### WP4 — Run pilots, calibrate resources, and resolve the value proposition

- **Objective/risk:** estimate headroom, numeric sensitivity, timing variability, and practical amortization before confirmation.
- **Entry/dependencies:** validated WP2/WP3 implementations.
- **Actions:** first freeze the resource-feasible tier prefix using section 5.2's structural schema and baseline calibration without inspecting comparative effects. Then run the structural pilot families, `C=2` and reproduction cap 8, moderate anisotropy, and adjacent problem sizes. Tune solvers against common external accuracy; measure construction/solve/validation cost separately. Assess how much of the B1-to-optimum gain is already captured by B2. Keep H2 objective comparisons separate from H3 equal-output timing. Evaluate whether an arithmetic bottleneck exists at all. Calculate stratified confirmation sample size and memory/time forecasts.
- **Files:** `protocol/pilot.yaml`, `runs/exploratory/`, `analysis/pilot_summary.json`, `protocol/resource_policy.yaml`, `decisions.md`.
- **Tests/outputs:** credible objective intervals; predicted versus measured memory; pilot variance decomposition; method-selection rule; explicit reason for each larger scale. A size increase must resolve a scaling, crossover, or boundary-condition uncertainty.
- **Resources:** scheduled from measured throughput. No blind broad sweep; no fixed requirement to train a model or run for a specified number of days.
- **Acceptance/next action:** a meaningful unresolved claim with adequate measurement sensitivity → WP5. No metric/engineering advantage → retain a negative/replication objective and proceed to a proportionate confirmation or WP9. A potentially tiny effect → refine numerical accuracy before deciding; do not reject it merely for missing an arbitrary percentage threshold. A precision bottleneck satisfying section 8.2 → authorize the scoped WP8 branch.

### WP5 — Freeze the confirmatory design

- **Objective/risk:** prevent adaptive case selection, peeking, and shifting success definitions.
- **Entry/dependencies:** WP4 with a scientifically interpretable claim and resource forecast.
- **Actions:** freeze hypotheses, baseline portfolio/dispatcher, configurations, precision/error budgets, primary metric, stratum weights/counts, exact topology and seed schema, within-tuple cell weights, sample-size rationale, analysis method, exclusions, and resource rules. Include the all-scheduled-units completion rule, objective-interval separation test, and golden generator hashes. Commit the source and protocol, then generate fresh instance seeds. Preserve their hashes before outcome access.
- **Files/commands:** `protocol/confirmatory.yaml`, `protocol/freeze.json`, `data/confirmatory/manifest.json`; proposed `python -m nilp_poc freeze --protocol protocol/confirmatory.yaml`.
- **Tests/outputs:** a reproducible hash over code, lockfile, configuration, and population definition; separation from exploratory cases; no reliance on inaccessible chat information.
- **Resources:** negligible relative to experiments.
- **Acceptance/next action:** complete freeze → WP6. Unresolved semantics or baseline configuration → return to WP4. Do not ask the end user to supply routine scientific decisions already determined by this plan.

### WP6 — Execute the controlled local comparison

- **Objective/risk:** test the frozen H2/H3 claim on the declared population with honest uncertainty and resource accounting.
- **Entry/dependencies:** WP5; protected inputs untouched by exploratory tuning.
- **Actions:** run paired methods using the recorded timing schedule; validate every metric independently; preserve raw measurements, failed runs, solver outputs, and all completed jobs. Generate the predeclared objective, accuracy, time, memory, and failure summaries. Use high-precision/analytic validation on a prespecified sentinel subset and every disputed apparent gain.
- **Files/commands:** `runs/confirmatory/<run_id>/`, `results.jsonl`, `metrics/`, `logs/`, `analysis/confirmatory.json`; proposed `run --protocol ...` and `report --run ...`.
- **Tests/outputs:** all scheduled units accounted for; no invalid candidate included as competitive; deterministic objective wins satisfy `upper(candidate)<lower(baseline)` or an explicit analytic proof; uncertainty computed by stratified tuple-level inference. A failed primary unit leaves the full-population speed claim unresolved; joint-success timing stays descriptive.
- **Resources:** WP4 forecast, updated only for recorded resource events. Checkpoint after each instance/block.
- **Acceptance/next action:** complete interpretable evidence → WP7. Implementation defect → invalidate affected records, repair, amend, and use fresh confirmation if the interpretation changed. Insufficient precision, power, or resources → label inconclusive and state the specific additional evidence needed. Do not keep adding samples until significance appears.

### WP7 — Attribute effects and independently confirm their scope

- **Objective/risk:** determine whether improvements come from the claimed structure rather than hidden precision, information, or baseline differences.
- **Entry/dependencies:** WP6, including negative or inconclusive findings.
- **Actions:** execute the preregistered ablations in section 8; audit all basis changes and readouts; compare grade, diagonal, block, and full metrics; share generic optimizations; rerun selected validated cases in a fresh process/environment from the lockfile. If a new causal explanation emerges, treat it as exploratory and freeze fresh confirmation for that revised claim.
- **Files:** `analysis/attribution.md`, `runs/replication/`, `protocol/amendments/`, `audit/experiment_review.md`.
- **Tests/outputs:** equivalent-method control remains null; gains survive the strongest eligible baseline; independent reconstruction reproduces essential values and decisions; failures have named causes.
- **Resources:** allocate to the specific remaining confound. Do not repeat the entire sweep without a concrete unresolved risk.
- **Acceptance/next action:** attributable, reproducible result → WP9. Gain explained by generic optimization or invalid restriction → narrow/retract the contribution. Genuine numerical bottleneck → WP8 only if its entry contract is complete.

### WP8 — Conditional fixed-semantics arithmetic experiment

- **Objective/risk:** test H4; prevent internal norm reduction from being mistaken for useful computation.
- **Entry/dependencies:** validated metric methods, a documented baseline failure or cost bottleneck, a concrete output/error contract, and a reason ordinary binary rescaling or float64 does not already resolve it adequately.
- **Actions:** use the smallest computation preserving the mechanism: coefficients through degree `r` of `prod_l(1+xi_l*z)` in the truncated polynomial algebra. Decide from the beneficiary whether only final coefficients or every prefix is needed. Implement encoding, scaled recurrence, decoding, and an independent high-precision reference using the exact same represented inputs. Compare N1–N3, including normalized coefficient conventions and final-output algorithms when allowed. Count metric construction, conversions, exponent metadata, and amortized reuse. Freeze a separate confirmatory protocol after the pilot.
- **Files:** `applications/polynomial_stream.py`, `applications/reference.py`, `applications/baselines.py`, `protocol/arithmetic.yaml`, separate exploratory/confirmatory directories.
- **Tests/outputs:** binary-scaling normal-range equivalence; input-quantization versus arithmetic-error separation; overflow/underflow and cancellation diagnostics; matched error-cost tables; unchanged decoded semantics. A mantissa–exponent output representation is available to every competitor if used.
- **Resources:** `O(B*r)` streamed state. Every accuracy-scored instance needs trustworthy reference/error bounds for every requested coefficient, with precision increased until the bounds resolve the accuracy contract. If this is affordable only on a subset, preregister that subset as the accuracy-qualified population; do not score unreferenced instances as accurate. Reference generation is common evaluation infrastructure; include method-specific certification costs in the corresponding method's runtime. Calibrate actual `L*r` throughput before selecting long streams.
- **Acceptance/next action:** reproducible Pareto improvement over the strongest eligible baseline → report narrowly and consider a real application. No advantage → stop the application branch and retain the mathematical result. Synthetic success alone does not establish an AD, ODE, PDE, or neural-training improvement; each such transfer would need a new application contract and contemporary baselines.

### WP9 — Synthesize, reproduce, and decide

- **Objective/risk:** deliver a complete result, including unfavorable evidence and unresolved novelty.
- **Entry/dependencies:** all authorized relevant branches completed, stopped with reasons, or explicitly blocked.
- **Actions:** build the report from immutable run artifacts; distinguish proved, measured, estimated, and untested claims; record the final prior-art attribution; execute the three audits in section 11 against the finished repository; reproduce a minimal clean run; package source, lockfiles, configs, raw results, and decision records.
- **Files:** `report.md`, `claim_ledger.csv`, `reproduction.md`, `decisions.md`, `audit/final.md`, `manifest.sha256`.
- **Tests/outputs:** every reported number traces to a raw record and code/config hash; every requested work package has a disposition; reproduction commands work from a fresh environment; no secret notebook or conversation context is needed.
- **Resources:** report generation and a minimal repeat of the scientific sentinels, not unnecessary full retesting.
- **Acceptance/next action:** choose continue, narrow, revise, or stop using section 10. A publication-level priority claim needs specialist historical scrutiny; do not claim peer review or send external messages without separate authorization.

## 8. Attribution, robustness, and failure analysis

### 8.1 Domain-specific adversarial cases

| Scenario | Exact check or expected behavior | Interpretation |
|---|---|---|
| Individually square-zero, not jointly nilpotent | Take `N1=E12`, `N2=E21`. Both square to zero, but `Psi(I)=I` and Euclidean gains are `2^L`. In fact `Tr(T_L(Q))=2^L Tr(Q)`, so no forward-feasible metric exists when `C<2^L`. | Reject the joint-nilpotence assumption; do not apply a finite polynomial truncation. |
| Every sampled exterior degree-one element squares to zero | Verify `(sum xi_i N_i)^2=0`, while products of distinct generators survive to degree `m`. | Confirms that per-layer square-zero behavior does not determine deep cost. |
| Shared noise across all layers | For `r=1`, use one common zero-mean unit-variance scalar: `P_L=I+L*xi*N`; the noise term scales as `L^2`, not `L`. | Independence/conditional-moment assumptions matter. Do not fit the iid exponent to this population. |
| Nonzero mean | For `r=1`, constant coefficient 1 gives deterministic `I+L*N`; compare against the centered model. | Drift is a different mechanism. Any drift theorem needs its own proof. |
| Depth-scaled amplitudes | Set `a_l=1/sqrt(L)`, so `V=1`. | Bounded conditioning is expected; this is not a counterexample to the fixed-amplitude theorem. |
| Concentrated versus diffuse variance | Match `V` while varying `beta_k`; compare exact finite-depth frontiers. | Same asymptotic order need not mean identical constants. |
| Singular covariance | Remove or perfectly correlate generator directions, then recompute their effective span. | The joint depth can fall; using the old `r` is invalid. |
| Near-nilpotent leakage | Add a small reverse edge to a shift and use the full, untruncated moment recurrence. | Arbitrarily small leakage can create long-depth behavior outside the theorem. No numerical-zero threshold establishes exact nilpotence. |
| Rounded basis changes | Compare an exact rational transform descriptor with raw rounded transformed arrays, retaining the declared model semantics. | Exact descriptors permit transported truncated moments; rounded arrays require an untruncated near-nilpotent check. Floating QR orthogonality is not an exact algebraic certificate. |
| Unjustified diagonal/grade restriction | Use a generic filtration tuple, anisotropic exterior generators, and rotated coordinates. | A restricted solver may be useful but is not the unrestricted optimum without symmetry. |
| Dense metric overhead | Measure basis conversion, transformed generator density, per-layer cost, and decoding. | A mathematically better metric may destroy sparse arithmetic and lose end-to-end. |
| Same moments, different tails | Compare Rademacher, Gaussian, and a standardized heavy-tailed distribution. | Exact expected gains can agree while peak range differs. Finite variance alone does not give finite variance of a squared-norm estimator; avoid unjustified normal confidence intervals for heavy-tail means. |
| Nonlinear activation inserted between layers | Check whether its Jacobian preserves the same filtration for every relevant state. | If not, the seed theorem no longer directly describes the network. |
| Loss-dependent terminal gradient | Compare a fixed covector with one selected from the realized product. | The fixed-direction expected adjoint guarantee does not imply the same bound for dependent gradients. |
| `C` close to 1 and large `r log L` | Evaluate in log coordinates/high precision and inspect gain slack after decoding metrics. | Numeric failure is not mathematical infeasibility. |
| Opposite volume/conditioning choices | Solve both objectives and compare their eigenvalue distributions. | Improving one does not automatically improve the other or arithmetic accuracy. |

For tails, use exact moment formulas for the mean-gain claim and appropriate independent-path quantile or exceedance estimates for range risk. A sample with no extreme events does not prove absence of rare failures.

### 8.2 Numerical benefit and amortization gate

Before WP8, write a one-page application contract containing the physical output, input representation, tolerance, streaming requirement, reuse frequency, and failure of the strongest existing implementation. The entry test is evidence-based: a diagnosed overflow, underflow, cancellation, or cost problem that the metric could plausibly change. Mere large internal norms are insufficient.

For the polynomial workload, `x_j` is the ordinary coefficient of `z^j`, not the derivative `j!*x_j`. JAX already exposes the distinction through `factorial_scaled`; do not create an overflow advantage by comparing normalized coefficients with an unnecessarily factorial-scaled baseline. [JAXJet]

Construct high-precision truth from the exact float32/float64 inputs used by all algorithms, for example via their exact integer ratios. Separately report error relative to pre-rounding input data if that is an application concern. This prevents input quantization from being attributed to the recurrence.

For every scored instance, validate all requested coefficients, including the highest degree and near-zero coefficients, against trustworthy references or error enclosures. Precision-doubling agreement is a diagnostic; if the claim requires a certified error, use an actual enclosure or exact arithmetic. A limited reference budget reduces the preregistered accuracy-qualified population rather than relaxing its truth requirement. Keep common reference construction outside algorithm timing, and include certification work needed by the algorithm's promised output contract.

When only final coefficients are required, compare direct recurrence with balanced products and the power-sum/Newton relation:

\[
s_k=\sum_\ell \xi_\ell^k,\qquad p_0=1,\qquad
p_n=\frac1n\sum_{k=1}^n(-1)^{k-1}s_k p_{n-k}.
\]

These have different accuracy and cost tradeoffs, including cancellation and power overflow. They must be validated, not assumed superior or excluded because they reorganize the computation. If every prefix is required for a real consumer, keep that constraint explicit for all algorithms.

Let `T_setup` include metric design and reusable conversion work, and let `Delta T` be the validated per-use saving versus the strongest baseline at the same error. If `Delta T>0`, the break-even reuse count is `ceil(T_setup/Delta T)` after including any baseline setup difference. If `Delta T<=0`, there is no runtime amortization. Compare that count to a stated realistic use frequency; do not assume unlimited reuse. Report both cold and amortized cost.

### 8.3 Failure taxonomy and response

| Failure class | Evidence | Required action |
|---|---|---|
| Broken implementation or measurement | Oracle disagreement, mismatched transpose/order, changed semantics, invalid gain, corrupted checkpoint | Fix the defect; invalidate affected runs; do not interpret it as a scientific negative. |
| Unsuitable proxy or insufficient sensitivity | Test lacks the relevant word length, noise regime, precision range, or detectable timing resolution | Redesign the proxy or improve sensitivity, with an amendment and fresh confirmation where needed. |
| Local resource limit | Native support absent, memory forecast exceeded, trustworthy solver accuracy unavailable | Use a faithful reduction/fallback or narrow the claim; retain the failure record. |
| Evidence against a hypothesis | Validated equality/dominance of strong baselines, or an exact counterexample satisfying the claim's assumptions | Reject or narrow that hypothesis; preserve other valid conclusions. |
| Mechanism works but offers no advantage | Internal gain decreases, but best scalar/convex baseline or decoded numerical performance matches it | Report a replication/negative utility result; do not rename the same mechanism. |
| Inconclusive | Objective interval, performance interval, or unresolved reference overlaps the claimed effect | State what precision, sample size, or missing comparison would resolve it, and whether that work is locally justified. |

## 9. Repository, interfaces, configuration, and reproducibility

### 9.1 Minimal repository layout

These are **proposed project files** for the executing coding agent. They are not claimed to exist merely because this plan names them.

| Path | Responsibility |
|---|---|
| `pyproject.toml` | Package metadata, dependencies, CLI entry point. |
| `src/nilp_poc/__init__.py`, `__main__.py` | Importable package and `python -m nilp_poc` entry; `__main__.py` calls `cli.main()`. Include `__init__.py` in subpackages as needed. |
| `src/nilp_poc/algebra.py` | Exact-structure descriptors and chain/exterior/DAG/block generators. |
| `src/nilp_poc/moments.py` | CP maps, finite-horizon coefficients, Choi construction, original gain evaluator. |
| `src/nilp_poc/structure.py` | Filtration metadata, symmetry certificates, allowed reductions. |
| `src/nilp_poc/oracles.py` | Closed forms, independent enumeration, exact diagnostic cases. |
| `src/nilp_poc/metrics.py` | Metric representations, normalization, encoding/decoding, objectives. |
| `src/nilp_poc/solvers/` | Scalar, diagonal, grade, block, full-SDP, and dispatcher implementations. |
| `src/nilp_poc/validation.py` | Independent gain checks, objective bounds, precision escalation, repair. |
| `src/nilp_poc/bench.py` and `cli.py` | Paired timings, memory capture, resumable run orchestration. |
| `src/nilp_poc/applications/` | Created only if WP8 is activated. |
| `tests/` | Scientific sentinels and invariance/negative controls. |
| `configs/`, `protocol/`, `theory/` | Parameters, frozen methods, amendments, and self-contained proofs. |
| `environment/`, `data/` | Machine/lock metadata and immutable instance manifests. |
| `runs/`, `analysis/`, `audit/` | Raw outputs, derived summaries, challenge records. |
| `decisions.md`, `reproduction.md`, `report.md` | Resumable status, clean-run instructions, final findings. |

### 9.2 Shared interfaces

Use typed dataclasses or equivalent validated records. Required conceptual signatures are:

```text
make_instance(spec: InstanceSpec) -> Instance
moment(instance: Instance, X, adjoint: bool = False, precision="float64") -> Matrix
solve_metric(instance: Instance, method: MethodSpec) -> MetricSolution
validate_metric(instance: Instance, solution: MetricSolution,
                validation: ValidationSpec) -> ValidationResult
simulate(instance: Instance, coefficients, initial_state,
         representation: MetricSolution | None, output_contract) -> SimulationResult
benchmark(instance: Instance, method: MethodSpec, schedule) -> BenchmarkRecord
```

`Instance` contains `d,s,r`, generators or a lossless structured descriptor, coefficients `beta`, the horizon/schedule, covariance, exact-structure provenance, level dimensions, optional basis transform, symmetry certificate ID, and content hash. A basis transform is part of the instance semantics, not an unlogged optimization.

`MetricSolution` contains a dense `Q`, log-diagonal values, or log-grade values plus the representation-to-physical-basis map; objective type; normalization; solver configuration/version; raw status; preprocessing/solve time; dual data if used; and evidence label. A log-grade metric must include grade multiplicities and the rule expanding it to physical coordinates.

`ValidationResult` contains both gains and their uncertainty, minimum metric eigenvalue or log counterpart, condition/volume objective, original-LMI residuals, map-reconstruction error, objective interval, precision used, and `accepted` with a machine-readable reason. Keep acceptance separate from solver status.

### 9.3 Configuration example

The following is an implementable starting schema, not a claim that its CLI already exists. The numerical error budgets are initial pilot settings; tighten them when a smaller effect must be resolved, and freeze them before confirmation. They are not minimum improvement targets.

```yaml
schema_version: 1
phase: exploratory
instance:
  family: weighted_chain
  nilpotent_depth: 3
  weights: [1, 1, 1]
  horizon: 100
  amplitude_schedule: {kind: constant, value: 1.0}
  covariance: identity
gain_cap_squared: 2.0
objective: condition
methods:
  - scalar_grade_exact
  - diagonal_log_cone
  - full_sdp
solver:
  candidates: [CLARABEL, SCS]
  dtype: float64
  use_analytic_kraus: true
validation:
  target_gain_abs_uncertainty: 1.0e-9
  target_log_condition_interval_width: 1.0e-7
  accept_gain_upper_bound_only: true
  high_precision_start_decimal_digits: 80
  preserve_inaccurate_results: true
benchmark:
  boundary: construct_canonicalize_solve_validate
  warmup_rule: pilot_until_initialization_effect_is_resolved
  repetition_rule: frozen_from_pilot_variance
  method_order: paired_randomized
resume:
  require_matching_hashes: true
```

If an interval upper bound at a cap-active analytic optimum is slightly above `C` because of evaluation uncertainty, use an explicitly interior feasible point for numerical comparison or retain the exact proof separately. Do not quietly relax the physical cap for one method.

### 9.4 Proposed project commands

Implement these commands before using them. They are not existing third-party package interfaces:

First write `pyproject.toml` with a standard setuptools build backend and package discovery under `src`, create `__init__.py`, `cli.py` with `main()`, and `__main__.py` calling that function, then run `python -m pip install -e .` in the active environment. Confirm `python -m nilp_poc --help` succeeds from the project root without manually modifying `PYTHONPATH`. Record the installed source revision and refresh the lockfile after this setup. These package files are part of WP0, not an assumed preexisting application.

```bash
python -m nilp_poc inspect --out environment/hardware.json
python -m nilp_poc verify --suite core
python -m nilp_poc solve --config configs/oracles.yaml
python -m nilp_poc pilot --protocol protocol/pilot.yaml
python -m nilp_poc freeze --protocol protocol/confirmatory.yaml
python -m nilp_poc run --protocol protocol/confirmatory.yaml --resume
python -m nilp_poc report --run runs/confirmatory/RUN_ID
```

Use `pytest` as an existing test runner only after writing the specified tests. A clean reproduction should regenerate the canonical instances from their descriptors, verify hashes, run the sentinels, then reproduce a selected result and report. Keep large sweeps optional rather than making a minimal reproduction depend on every prior measurement.

### 9.5 Experiment record and data integrity

Use append-only JSONL for run events and result records, with arrays stored in NPZ or a documented sparse format. Every final record must include:

```text
schema_version, run_id, parent_run_id, phase, timestamp_utc
code_commit, dirty_tree_diff_hash, dependency_lock_hash, hardware_id
instance_id, instance_hash, family, d, s, r, f_j, nu, horizon, beta_hash
stratum_id, population_weight, tuple_cell_weight, scheduled_primary_unit_id
seed, rng_algorithm, covariance_hash, basis_transform_hash
method_id, method_version, solver_version, solver_options, backend, threads
objective_type, gain_cap_squared, metric_representation, metric_file_hash
solver_status, validation_status, failure_reason, evidence_label
primary_population_complete, pair_completion_status, timing_summary_scope
forward_gain, adjoint_gain, gain_error_bounds
log_condition, logdet, objective_lower_bound, objective_upper_bound
map_reconstruction_error, primal_residuals, dual_residuals, precision
construction_s, canonicalization_s, solve_s, validation_s, total_s
timing_block_id, method_order, warm_or_cold, peak_rss_bytes
paired_block_effect, attempt_count, repair_count, completion_elapsed_s
memory_pressure, swap_observation, power_mode, interrupted
```

Use `null` for unavailable quantities with a reason; never fill them with invented zeros. Store raw solver logs, accepted and rejected metrics, exact generator descriptions, full configurations, seed states, and exclusion decisions. Protect exploratory/confirmatory separation in the directory structure and run manifest.

Resume writes atomically to a temporary record followed by rename. Mark an instance complete only after outputs and checksums are committed. If code or numerical settings change, create a new run ID. Do not mix partial runs from different states into a single result without explicit labeling.

The reporting script must show all scheduled instances, including failures and censored jobs, and must trace every plotted number to its record. Useful plots are `log(kappa)` versus `log(L)` with analytic envelopes, the gain–distortion frontier, valid completion time versus problem size, and error–cost Pareto curves only for an activated application branch. Use standard plotting tools and export data with figures; figures are not substitutes for the raw tables.

## 10. Executed evidence, outcome interpretation, and continuation

### 10.1 Small checks actually performed

These checks were executed during preparation, independently challenged, and rerun by the lead investigation. They are **exploratory Linux CPU checks**, not Mac measurements, not a protected benchmark, and not external peer review. The complete essential reproduction code is embedded in Appendix A; no separate intermediate file is required to understand or repeat them.

**Exact-moment evaluations.** For the scalar grade choice `q_j=100^(-j)`:

| Instance | Raw forward gain | Raw adjoint gain | Scaled forward gain | Scaled adjoint gain |
|---|---:|---:|---:|---:|
| Unit chain `r=1, L=100` | 101 | 101 | 2 | 2 |
| Unit chain `r=2, L=100` | 5,051 | 5,051 | 2.495 | 2.495 |
| Unit chain `r=3, L=100` | 166,751 | 166,751 | 2.6567 | 2.6567 |
| Exterior `m=3, L=100` | 1,000,201 | 1,000,201 | 7.9402 | 7.9402 |

For the exterior case, the condition number is `10^6`, the determinant is `10^-24`, and `nu=12`. Direct enumeration of 64 short Rademacher paths for a separate two-generator, three-dimensional tuple agreed with the moment recursion to a maximum absolute difference of about `3.6e-15` in float64. These are deterministic evaluations, not sampling estimates of an expectation.

**Finite-depth frontier.** An 80-decimal-digit evaluation of the analytic `r=3, L=100, C=8` oracle gives

```text
best scalar grade condition: 80142.951804145696057147117355539...
unrestricted optimum:       79932.396991777622204239699648102...
relative reduction:         0.2627240544903190253... percent
```

The corresponding optimal diagonal is approximately

```text
[1,
 0.02488520261586591,
 0.0005027313659770891,
 0.000012510571903690899]
```

The reference scalar scale was itself optimized. **For this unit chain**, the untuned `q_j=L^-j` has `kappa=1,000,000`; most of that gap is removed by scalar tuning. This is a different algebra from the seed's exterior example, whose same grade rule gives gain 7.9402 close to the cap 8. The chain comparison does not imply a similarly large tuning gap for the exterior algebra.

**Conic formulations.** With CVXPY 1.9.3 and Clarabel 0.11.1, using the small instance above:

| Formulation | Reported status | Recomputed condition number | Recomputed maximum gain | Interpretation |
|---|---|---:|---:|---|
| Diagonal log-cone | `optimal` | 79,932.3969926 | 7.99999999999 | Agrees with the analytic optimum to about `1e-11` relative condition error; numerical agreement, not a general solver proof. |
| Full SDP, canonical basis | `optimal_inaccurate` | 79,932.4031044 | 7.99999986500 | Near the analytic solution and gain-feasible in the numerical check; exact accuracy must be assessed independently of status. |
| Full SDP, orthogonally rotated basis | `optimal_inaccurate` | 79,931.7790664 | 8.00007014439 | **Invalid.** Its apparently better objective is explained by a violated adjoint constraint. |

Reevaluating the rotated result at 80 decimal digits gave adjoint gain `8.00007014441007209...`, confirming that the failure was not a display-rounding artifact. Its backward-block minimum eigenvalue was about `-8.25e-10`; this small raw LMI residual was amplified by the ill-conditioned metric. This is why the original gain check is compulsory.

This probe rotates a finite list of Kraus factors in float64 and evaluates that represented finite CP map. Ideal orthogonal invariance supplies the reference, while the rounded transform/factors are numerical perturbations. It does not certify exact nilpotence of a raw rounded transformed generator. The actual rotation and factors are stored, and the high-precision check reads those same stored inputs rather than recomputing QR from a seed. The gain violation is much larger than the observed transformation-roundoff scale.

Only tiny formulation probes were run. No reliable speedup, scaling curve, generic large-instance optimum, or M3 compatibility claim follows from them. SCS was installed for the investigation but was not used to produce the comparison table above; a local SCS comparison remains proposed work.

**Floating-point negative controls.** In the primary control, seed 56312 generated 4,096 Gaussian coefficients rounded to float32. For `r=12`, `x_0=e_0`, and binary grade scale `t=1/16`, final maximum absolute coefficients were approximately `9.9108938745643e16` unscaled and `352.1056823730469` scaled. Decoding by powers of two gave identical float32 bit patterns and maximum absolute difference zero. A second independent test at `r=5, L=1000, t=1/32`, with a different seed and nontrivial initial vector, also produced identical decoded bits.

These observations support an elementary explanation: in a normal-range binary computation with matching operation order, exact power-of-two rescaling can commute with rounding. They do not show that all coordinate transformations are harmless, that all scaling is useless, or that range avoidance cannot matter.

### 10.2 Outcome rules

| Outcome | Permitted conclusion | Next action |
|---|---|---|
| Exact theorem predecessor found | The seed theorem is a replication/application of known work. | Correct attribution; pursue only a remaining finite-depth or engineering question with value. |
| New finite-depth formula/proof survives specialist comparison | A narrow mathematical contribution may exist. | Prepare a proof-focused result with explicit assumptions and prior-art boundary; keep priority provisional until reviewed. |
| B3/B5 improve B2 by a validated amount | Extra metric freedom helps on the specified instance/population. | Report the full effect and computation cost. Do not claim superiority over equivalent convex formulations. |
| Specialized implementation beats B6 at matched accuracy and total cost | A reproducible engineering advantage exists for the declared family/hardware. | Confirm on fresh tuples, retain failure regimes, and state whether the gain comes from algorithmic structure or implementation. |
| Reduction only beats an unreduced SDP | Exploiting known structure helps relative to that formulation. | Report that limited comparison; do not claim a novel optimization method. |
| Internal gains shrink but decoded errors/cost do not improve | Coordinate stabilization has no demonstrated application benefit in the tested regime. | Publish/report the negative result or design guidance; stop the application branch. |
| Arithmetic Pareto improvement survives N1–N3 and independent confirmation | A component numerical benefit is demonstrated. | Seek a real consumer whose contract matches the successful regime; do not jump directly to LLM or general neural claims. |
| Validated effects are tiny or expensive to obtain | Mathematical headroom exists, practical utility may be negligible. | State the break-even condition; avoid escalating scale without evidence of greater information or utility. |
| Results conflict across bases or solvers | Numeric formulation/conditioning is unresolved. | Reevaluate original gains and exact oracles; no positive claim until resolved. |
| Local proxy cannot express proposed scale-dependent behavior | The larger claim remains untested. | Identify the specific missing interaction/scale and evidence needed; neither reject it universally nor use it to excuse every negative local result. |

### 10.3 Evidence-triggered pivots

Only three pivots are justified by this investigation:

1. **Theory/diagnostic result:** if practical improvements disappear, finish a rigorous stability-cost characterization and a reusable validator. This preserves the seed's core question.
2. **Numerical representation result:** if range failures are common and fixed-semantics exponent management helps, compare against existing binary scaling and wider precision; describe the benefit as numerical engineering.
3. **Different architecture assumptions:** if an actual application requires drift, learned nonlinearities, or time-varying metrics, formulate a new model and claim ledger before experimenting. Ordinary residual scaling is a primary competitor. This is an explicit extension, not evidence already supplied by the nilpotent theorem.

No pivot is authorized merely to find a favorable benchmark. If known methods fully resolve the useful task, stop with that conclusion.

## 11. Audit record, unresolved risks, and limitations

### 11.1 Defects found and revisions made during preparation

The investigation used separate research agents for mathematical challenge, prior-art search, and practical-baseline analysis, followed by lead reconstruction and executed checks. This is independent task-level review within the preparation process, not human peer review.

| Audit area | Material weakness found | Change in this handoff |
|---|---|---|
| Scientific/prior art | Broad novelty claims overlap with decades of LMI, GP, and algebraic work. | Narrowed originality to the exact theorem/finite-depth questions; made known convex methods explicit baselines. |
| Scientific interpretation | Coordinate gain and determinant could be mistaken for physical conditioning or information preservation. | Added encoding/decoding invariance, square-root condition/volume factors, and separate numerical-utility gate. |
| Mathematical completeness | The supplied determinant proof was abbreviated and depended on an unavailable notebook for details. | Reconstructed the trace/min-max proof and explicit upper construction in section 2. |
| Formulation | The inverse in the backward constraint could lead to an unnecessary nonconvex optimizer. | Derived the exact Schur-complement LMI and made the full SDP a reference. |
| Baseline fairness | Comparing only against `t^2=1/L` exaggerates the gain. | Added exact scalar tuning; the finite-depth example reveals only about 0.263% incremental improvement. |
| Attribution | Diagonal preservation does not prove diagonal optimality. | Required explicit sign/phase/permutation symmetry certificates and labeled generic diagonal results restricted. |
| Numerical validity | A tiny raw LMI residual produced a false better optimum after rotation. | Added original gain reevaluation, independent precision checks, objective bounds, and rejection/repair rules. |
| Floating-point validity | Smaller internal values were assumed to improve accuracy. | Executed two binary-scaling negative controls and made them mandatory harness tests. |
| Feasibility | A generic Kraus/Choi formulation can exceed laptop memory despite small generator storage. | Added analytic factorization, cone-size memory models, measured sizing, and exact structural fallbacks. |
| Inference | Seeds, depths, and repeated timings can be counted as independent problems. | Defined tuple-level units, separate deterministic evidence, pilot-calibrated fixed-sample inference, and protected confirmation. |
| Reproducibility | Interrupted runs could mix code or precision states. | Added immutable hashes, atomic records, run lineage, and resume compatibility checks. |
| Scientific model semantics | Rounded orthogonal conjugation can destroy exact nilpotence, and a path-dependent metric would invalidate the fixed-metric expectation identity. | Made exact rational transform descriptors authoritative, required untruncated checks for raw rounded models, and made metric determinism explicit. |
| Scientific interpretation | The scalar-tuning comparison could be read as a claim about the seed's exterior example. | Identified the unit chain explicitly in sections 1 and 10 and kept the exterior calculation separate. |
| Experimental validity | Dropping failed pairs could manufacture a speed advantage. | Required all scheduled primary cells to complete for a full-population speed claim, retained descriptive conditional timing, and counted retries/repairs. |
| Experimental reproducibility | Random populations lacked exact topology, family weights, and cell weights. | Added an exact generator/seed contract, resource-based tier selection, golden hashes, fixed stratum weights, and stratified tuple inference. |
| Numerical evidence | Narrow but overlapping objective intervals or partially referenced arithmetic outputs could be misclassified as improvements. | Required separated certified objective bounds and references/enclosures for every coefficient of every accuracy-scored instance; separated H2 and H3 output contracts. |
| Fresh-environment execution | Identity cases could divide by zero; `src` modules were not installed; failure scripts could crash or regenerate different QR factors. | Added early identity/input checks, polynomial filtration recursion, explicit package entry/install steps, and robust scripts that preserve actual factors and failure records. |

After the complete draft existed, three separate full-document reviews covered scientific/prior-art validity, experimental validity, and implementation/feasibility. Their material findings are included above. Corrections were integrated and followed by targeted rereads; the final rereads found no further substantive correction within the reviewed scope. The timing-estimator ambiguity required one additional narrow pass, which verified the explicit paired-log estimator and failure handling now in sections 5.6–5.7. This is a review of the finished plan and its mathematical/specification content, not a claim that the proposed production repository or confirmatory experiments already exist.

The corrected A3/A4 scripts were rerun after adding stored-factor provenance and failure handling; all reported numerical values remained unchanged. Four targeted mocked cases—missing solution values and a solver exception for each of the full and diagonal formulations—preserved failure records without crashing. These checks address those concrete script defects; they do not establish generic solver reliability or target-Mac compatibility.

### 11.2 Required final execution audits

The coding agent must repeat three distinct reviews against the implemented repository and resulting evidence:

- **Scientific/prior-art audit:** try to find an equivalent theorem or method; test whether a coordinate relabeling explains the claim; verify that all assumptions and contribution labels match the result.
- **Experimental-validity audit:** try to explain every positive effect using stronger baseline tuning, extra precision/information, shared optimizations, gain violations, protected-set leakage, or adaptive sampling.
- **Implementation/feasibility audit:** reproduce from only this document and the repository on a fresh native environment; check every command, required file, dependency, license, numerical path, memory forecast, and interruption case.

For each defect, record the affected claim, evidence, correction, and downstream rechecks. Repeat targeted review until the next pass finds no substantive correction to the tested claim. That is a scoped completion criterion, not a guarantee of absolute correctness or novelty.

### 11.3 Remaining uncertainty

Historical priority of the exact nilpotent metric-cost theorem and derived finite-depth formulas is unresolved. Older operator-similarity, quantitative nilpotent scaling, and control literature are plausible remaining sources. The literature search is not proof of absence of prior art and is not a patentability or freedom-to-operate opinion.

The actual target machine has not been inspected. Native package support has documentary evidence, but the required operations and solver accuracy must pass WP0 on that machine. No target runtime, peak-memory measurement, battery behavior, or GPU claim is supplied by the Linux checks.

The core workloads are synthetic mathematical objects. No evidence yet shows that a real numerical application needs the optimized metric, that its setup can amortize, or that a trained network retains the model's assumptions. The application branch deliberately remains contingent on a diagnosed problem.

High-precision reevaluation is not the same as a formal interval proof. General numerical optimum certificates and large-instance solver robustness remain implementation tasks. Exact small oracles and analytical equivalences provide the initial trustworthy reference.

## 12. References and search-coverage record

### 12.1 Primary references

The entries below identify the material actually inspected and its role. Bibliographic links are ordinary stable URLs/DOIs and do not depend on chat citation markers. Dynamic documentation was accessed on 30 October 2026.

- **BM14 — B. V. Rajarama Bhat and Nirupama Mallick.** *Nilpotent Completely Positive Maps*. Positivity 18, 567–577 (2014); preprint 2013. DOI: <https://doi.org/10.1007/s11117-013-0262-4>. Full text: <https://arxiv.org/pdf/1310.0591>. Inspected Proposition 2.2, the CP nilpotent type, and filtration results. Structural antecedent, not an inspected proof of the complete optimization theorem here.
- **BEFB94 — Stephen Boyd, Laurent El Ghaoui, Eric Feron, Venkataramanan Balakrishnan.** *Linear Matrix Inequalities in System and Control Theory*. SIAM, 1994. <https://web.stanford.edu/~boyd/lmibook/lmibook.pdf>. Inspected condition-number/scaling sections, minimum-condition stable coordinates, and multiplicative-noise chapter. Principal historical source for the optimization methodology; different exact model/claim.
- **LP11 — Zhaosong Lu and Ting Kei Pong.** *Minimizing Condition Number via Convex Programming*. SIAM Journal on Matrix Analysis and Applications 32(4), 1193–1211 (2011). DOI: <https://doi.org/10.1137/100795097>. Author text: <https://zhaosong-lu.github.io/ResearchPapers/cond_opt_rev.pdf>. Inspected convex/SDP formulations and preconditioning discussion. Direct novelty constraint on generic condition-number optimization.
- **NR99 — Arkadi Nemirovski and Uriel Rothblum.** *On Complexity of Matrix Scaling*. Linear Algebra and its Applications 302–303, 435–460 (1999). DOI: <https://doi.org/10.1016/S0024-3795(99)00212-8>. Author text: <https://www2.isye.gatech.edu/~nemirovs/arkadi.pdf>. Inspected matrix-scaling/GP formulations. Establishes the broad scaling framework, not the specific gain-cost frontier.
- **GP04 — Karin Gatermann and Pablo A. Parrilo.** *Symmetry Groups, Semidefinite Programs, and Sums of Squares*. Journal of Pure and Applied Algebra 192, 95–128 (2004); preprint 2002. <https://arxiv.org/pdf/math.AC/0211450>. Inspected Theorem 3.3 and invariant convex reduction. Supports the averaging technique; each generator family still needs its own invariance check.
- **ZDZ18 — Tianliang Zhang, Feiqi Deng, Weihai Zhang.** *Finite-time Stability and Stabilization of Linear Discrete Time-varying Stochastic Systems*. 2018 preprint. <https://arxiv.org/pdf/1806.08470>; DOI: <https://doi.org/10.1016/j.jfranklin.2018.10.026>. Inspected definitions, moment transitions, and matrix inequalities. Related finite-horizon stochastic analysis, including time-varying metrics.
- **SJP15 — Edward Schmerling, Lucas Janson, Marco Pavone.** *Optimal Sampling-Based Motion Planning under Differential Constraints: the Drift Case with Linear Affine Dynamics*. 2014 preprint, 2015 revision. <https://arxiv.org/abs/1405.7421>; <https://arxiv.org/pdf/1405.7421>. Inspected Lemmas III.1–III.4. Related filtration-dependent Gramian spectrum/determinant orders under a different control problem.
- **BFGOW21 — Peter Bürgisser, Cole Franks, Ankit Garg, Rafael Oliveira, Michael Walter, Avi Wigderson.** *Towards a Theory of Non-commutative Optimization: Geodesic First and Second Order Methods for Moment Maps and Polytopes*. 2019, expanded 2021. <https://arxiv.org/pdf/1910.12375>. Inspected simultaneous conjugation, null-cone/norm optimization, and quantitative conditioning sections. Important adjacent theory; no exact theorem match established.
- **JLL14 — Rodney James, Julien Langou, Bradley R. Lowery.** *On Matrix Balancing and Eigenvector Computation*. 2014 preprint. <https://arxiv.org/pdf/1401.5766>. Inspected power-of-two balancing and accuracy counterexamples. Motivates measuring the actual downstream error; its task is eigenvector computation.
- **SciPyBal — SciPy developers.** `scipy.linalg.matrix_balance` documentation. <https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.matrix_balance.html>. Inspected signature, triangular-block permutation behavior, and power-of-two scales. Diagnostic baseline specification; not an optimizer for the present constraints.
- **Ishii21 — Hiromi Ishii.** *Automatic Differentiation With Higher Infinitesimals, or Computational Smooth Infinitesimal Analysis in Weil Algebra*. CASC 2021, LNCS 12865, 174–191. DOI: <https://doi.org/10.1007/978-3-030-85165-1_11>. <https://arxiv.org/pdf/2106.14153>; code <https://github.com/konn/smooth>. Inspected algebraic AD mechanism and repository. Establishes arbitrary Weil-algebra differentiation as prior art.
- **JAXJet — The JAX authors.** `jax.experimental.jet` documentation and implementation. <https://docs.jax.dev/en/latest/jax.experimental.jet.html>; <https://github.com/jax-ml/jax/blob/main/jax/experimental/jet.py>. Inspected the normalized-coefficient/factorial option and source path. Reusable AD baseline; exact primitive coverage and version must be checked for any activated application.
- **TaylorSeries — Luis Benet, David P. Sanders, and contributors.** TaylorSeries.jl repository and documentation. <https://github.com/JuliaDiff/TaylorSeries.jl>; <https://juliadiff.org/TaylorSeries.jl/stable/userguide/>. Inspected coefficient conventions, `Taylor1`, and MIT license. Independent CPU Taylor implementation, not a reproduced benchmark in this investigation.
- **Dangel25 — Felix Dangel, Tim Siebert, Marius Zeinhofer, Andrea Walther.** *Collapsing Taylor Mode Automatic Differentiation*. NeurIPS 2025. <https://arxiv.org/abs/2505.13644>; <https://proceedings.neurips.cc/paper_files/paper/2025/file/d835e25415cfe4d6d48a0be3b7b763bb-Paper-Conference.pdf>; code <https://github.com/f-dangel/torch-jet>. Inspected mechanism, experimental task, and implementation limitations. Relevant to a differential-operator pivot; published GPU numbers are not controlled M3 comparisons.
- **Marion25 — Pierre Marion, Adeline Fermanian, Gérard Biau, Jean-Philippe Vert.** *Scaling ResNets in the Large-depth Regime*. JMLR 26(56), 1–48 (2025). <https://www.jmlr.org/papers/volume26/22-0664/22-0664.pdf>. Inspected scaling/regularity assumptions and limiting regimes. Supports treating ordinary residual scaling as established adjacent work, with different nonlinear-network assumptions.
- **Patent19 — Timothy Anderson, Monica Martinez-Canales, Vinod Sharma.** *Neural Network Training*, publication US20190050727A1, 14 February 2019. <https://patents.google.com/patent/US20190050727A1/en>. Inspected publication metadata and dual/hyper-dual training description. Historical mechanism overlap only; patent performance statements are not treated as experimental evidence, and legal status/scope was not assessed.
- **CVXPYInstall — CVXPY developers.** Installation documentation. <https://www.cvxpy.org/install/>. Inspected supported platforms, Python/dependency requirements, and installation routes. Documentary feasibility evidence only.
- **CVXPYSolvers — CVXPY developers.** Solver features documentation. <https://www.cvxpy.org/tutorial/solvers/>. Inspected supported cone types and solver options. Determines the required capability smoke tests.
- **CVXPYPyPI — CVXPY developers.** CVXPY 1.9.3 distribution record, released 19 September 2026. <https://pypi.org/project/cvxpy/1.9.3/>. Inspected release, Python 3.12 universal2 wheel, and Apache-2.0 metadata. Installation on the target was not performed.
- **ClarabelPyPI — Paul Goulart and Clarabel contributors.** Clarabel 0.11.1 distribution record, released 11 June 2025. <https://pypi.org/project/clarabel/0.11.1/>; documentation <https://clarabel.org/stable/python/getting_started_py/>. Inspected arm64 wheel, cone capabilities, API, and Apache-2.0 metadata. Used in Linux probes.
- **SCSPyPI — SCS contributors.** SCS 3.3.1 package record and backend documentation. <https://pypi.org/project/scs/3.3.1/>. Inspected platform-specific linear-system backends. Installed but not used for the reported conic comparison; target capability/performance checks remain necessary.
- **AppleM3 — Apple.** *MacBook Pro (14-inch, M3 Pro or M3 Max, Nov 2023): Technical Specifications*. <https://support.apple.com/en-us/117736>. Inspected official model/CPU/configuration information. Subsequent repeated page requests were rate-limited; actual target specifications remain a local discovery task.
- **NumPyRecord — NumPy developers.** NumPy 2.3.5 distribution record. <https://pypi.org/project/numpy/2.3.5/>. Inspected release and BSD license metadata; this was the local validation version, not a claim of latest release.
- **SciPyRecord — SciPy developers.** SciPy 1.17.0 distribution record. <https://pypi.org/project/scipy/1.17.0/>. Inspected release and BSD license metadata; this was the local validation version.
- **NumPyCurrent — NumPy developers.** NumPy 2.5.4 distribution record, released 10 October 2026. <https://pypi.org/project/numpy/2.5.4/>. Inspected Python requirement, Python 3.12 arm64 macOS wheels, and combined license metadata. Proposed for new target measurements, not used for the existing Linux results.
- **SciPyCurrent — SciPy developers.** SciPy 1.18.1 distribution record, released 21 August 2026. <https://pypi.org/project/scipy/1.18.1/>. Inspected Python requirement and Python 3.12 arm64 macOS wheels. Proposed for new target measurements, not used for the existing Linux results.
- **MpmathRecord — Fredrik Johansson and mpmath contributors.** mpmath 1.4.1 distribution record. <https://pypi.org/project/mpmath/1.4.1/>. Inspected arbitrary-precision support and BSD-family license; used for 80-digit checks, not claimed as a formal interval certification by itself.
- **PythonMac — Python Software Foundation.** *Python Releases for macOS*. <https://www.python.org/downloads/macos/>. Inspected official installer availability and universal2 options; target interpreter selection and operation tests remain WP0.

### 12.2 Additional leads, access limits, and search scope

The search extended beyond the seed terminology through CP nilpotence, Kraus words, Jordan/transient growth, common quadratic Lyapunov metrics, multiplicative-noise systems, finite-horizon stability, minimum-condition transformations, diagonal balancing, geometric programming, symmetry reduction, controllability-Gramian asymptotics, simultaneous conjugation/null cones, Weil/Taylor AD, residual-depth scaling, and binary floating-point rescaling. Both available search engines were used. Relevant references were traced backward into older control/scaling work and forward into recent AD implementations and package support.

Representative query families included `nilpotent completely positive maps type`, `minimum condition number stable coordinate transformation`, `finite time stochastic Lyapunov metric`, `diagonal similarity geometric programming row column sums`, `nilpotent simultaneous conjugation condition number`, `Taylor coefficients factorial_scaled jet`, and targeted combinations of `nilpotent`, `common metric`, `determinant`, and `residual`. A targeted patent search located the dual-number training publication above.

Additional primary material inspected during adjacent-field tracing included Mullhaupt–Riedel's *Exponential Condition Number of Solutions of the Discrete Lyapunov Equation* (<https://arxiv.org/pdf/1803.04046>), a switched-system Lie-algebraic stability paper (<https://web.ece.ucsb.edu/~hespanha/published/lie.pdf>), and Trevor Jones's *Canonical Filtrations of Finite-Dimensional Algebras* (<https://arxiv.org/pdf/2406.11765>). These reinforced the distinction between associative nilpotence, established scaling theory, and the exact common-metric objective; they were not used as evidence of an exact predecessor.

Spirito–Astolfi's *An Explicit Solution to the Generalized Lyapunov Matrix Inequality* (2024), DOI <https://doi.org/10.1016/j.sysconle.2024.105905>, was inspected at abstract/introduction level; full text was blocked. A search lead titled *Backpropagation as a Nilpotent Linear System* could not be verified from accessible primary content and is not used to support a claim. Newer differential-operator work, including the official STDE++ publication page (<https://jmlr.org/papers/v27/25-1474.html>), was inspected only for contextual scope; no reproduction or compatibility assertion is made.

The later targeted searches primarily returned the same LMI, matrix-scaling, invariant-theory, and AD families, or unrelated uses of “nilpotent residual.” They did not reveal a new decisive implementation constraint or an inspected exact theorem equivalent. That is the stopping rationale for this research pass, rather than a paper-count quota. Remaining blind spots include older unindexed operator-similarity results, inaccessible full texts, and specialist historical knowledge.

The essential research has been performed for this handoff. The executing agent need not repeat the entire literature review. It should check a newly encountered close predecessor and refresh software compatibility if execution is substantially later, while preserving provisional attribution until priority is resolved.

[BM14]: https://arxiv.org/pdf/1310.0591
[BEFB94]: https://web.stanford.edu/~boyd/lmibook/lmibook.pdf
[LP11]: https://zhaosong-lu.github.io/ResearchPapers/cond_opt_rev.pdf
[NR99]: https://www2.isye.gatech.edu/~nemirovs/arkadi.pdf
[GP04]: https://arxiv.org/pdf/math.AC/0211450
[ZDZ18]: https://arxiv.org/pdf/1806.08470
[SJP15]: https://arxiv.org/pdf/1405.7421
[BFGOW21]: https://arxiv.org/pdf/1910.12375
[JLL14]: https://arxiv.org/pdf/1401.5766
[SciPyBal]: https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.matrix_balance.html
[Ishii21]: https://arxiv.org/pdf/2106.14153
[JAXJet]: https://docs.jax.dev/en/latest/jax.experimental.jet.html
[TaylorSeries]: https://github.com/JuliaDiff/TaylorSeries.jl
[Dangel25]: https://arxiv.org/abs/2505.13644
[Marion25]: https://www.jmlr.org/papers/volume26/22-0664/22-0664.pdf
[Patent19]: https://patents.google.com/patent/US20190050727A1/en
[CVXPYInstall]: https://www.cvxpy.org/install/
[CVXPYSolvers]: https://www.cvxpy.org/tutorial/solvers/
[CVXPYPyPI]: https://pypi.org/project/cvxpy/1.9.3/
[ClarabelPyPI]: https://pypi.org/project/clarabel/0.11.1/
[SCSPyPI]: https://pypi.org/project/scs/3.3.1/
[AppleM3]: https://support.apple.com/en-us/117736
[NumPyRecord]: https://pypi.org/project/numpy/2.3.5/
[SciPyRecord]: https://pypi.org/project/scipy/1.17.0/
[NumPyCurrent]: https://pypi.org/project/numpy/2.5.4/
[SciPyCurrent]: https://pypi.org/project/scipy/1.18.1/
[MpmathRecord]: https://pypi.org/project/mpmath/1.4.1/
[PythonMac]: https://www.python.org/downloads/macos/


## Appendix A. Standalone reproduction of the executed small checks

These are complete small-check scripts, not the proposed production CLI. Copy each block to the indicated project-relative filename. They require only the dependencies imported in the block. Their algebraic values are reproducible; exact solver iterates/status and timing can vary across platforms. The scripts do not implement all validation and resource guards required by the work packages.

### A1. Moment values, analytic rank-one cases, a numerical grade candidate, and binary equivariance

Save as `checks/core_checks.py`; run `python checks/core_checks.py`. The SLSQP output is explicitly an exploratory candidate, not a certified generic optimum.

```python
"""Small independent numerical checks; Linux CPU, not M3 compatibility evidence."""
import json
import math
import platform
import sys

import numpy as np
from scipy.optimize import brentq, minimize


def psi(q, ns, adjoint=False):
    return sum((n @ q @ n.T if adjoint else n.T @ q @ n) for n in ns)


def cp_sum(q, ns, depth, r, adjoint=False):
    out = q.copy()
    term = q.copy()
    for k in range(1, r + 1):
        term = psi(term, ns, adjoint)
        out += math.comb(depth, k) * term
    return out


def gain(q, ns, depth, r):
    ev, vec = np.linalg.eigh(q)
    root = (vec * np.sqrt(ev)) @ vec.T
    invroot = (vec / np.sqrt(ev)) @ vec.T
    f = invroot @ cp_sum(q, ns, depth, r) @ invroot
    b = root @ cp_sum(np.linalg.inv(q), ns, depth, r, True) @ root
    return [float(np.linalg.eigvalsh(f)[-1]), float(np.linalg.eigvalsh(b)[-1])]


def exterior(m):
    ns = []
    for i in range(m):
        n = np.zeros((2**m, 2**m))
        for mask in range(2**m):
            if mask & (1 << i):
                continue
            sign = -1 if (mask & ((1 << i) - 1)).bit_count() % 2 else 1
            n[mask | (1 << i), mask] = sign
        ns.append(n)
    return ns


def chain(r, weights=None):
    n = np.zeros((r + 1, r + 1))
    for i in range(r):
        n[i + 1, i] = 1.0 if weights is None else weights[i]
    return [n]


def symmetric_grade_opt(r=3, depth=100, c=8.0):
    # For L > C-1 on a unit chain each feasible q_{j+1}/q_j < 1.
    # Therefore q_0=1 and min -log(q_r) is the conditioning problem.
    h = np.zeros((r + 1, r + 1))
    for i in range(r + 1):
        for j in range(i + 1):
            h[i, j] = math.comb(depth, i - j)
    a = brentq(lambda a: sum(math.comb(depth, k) * a**k for k in range(r+1)) - c, 1e-12, 1.0)
    z0 = np.arange(1, r + 1) * math.log(a)
    def sums(z):
        zz = np.r_[0.0, z]
        scaled = h * np.exp(zz[:, None] - zz[None, :])
        return np.r_[scaled.sum(axis=0), scaled.sum(axis=1)]
    opt = minimize(lambda z: -z[-1], z0, method='SLSQP',
                   constraints=[{'type':'ineq', 'fun':lambda z: c - sums(z)}],
                   options={'ftol':1e-12,'maxiter':300})
    q = np.exp(np.r_[0.0, opt.x])
    return {'r':r, 'L':depth, 'C':c, 'solver_success':bool(opt.success),
            'solver_message':str(opt.message), 'uniform_q':[float(a**k) for k in range(r+1)],
            'uniform_condition':float(a**(-r)), 'candidate_q':q.tolist(),
            'candidate_condition':float(1/q[-1]), 'candidate_gains':sums(opt.x).tolist(),
            'warning':'Numerical feasible candidate only; no dual certificate and no M3 measurement.'}


def power_two_invariance():
    rng = np.random.default_rng(42031)
    r, depth, t = 5, 1000, np.float32(1/32)
    x = rng.normal(size=r+1).astype(np.float32)
    scale = (t ** np.arange(r+1)).astype(np.float32)
    z = x * scale
    max_seen = 0.0
    for xi in rng.normal(size=depth).astype(np.float32):
        a = x.copy()
        b = z.copy()
        x[1:] = a[1:] + xi*a[:-1]
        z[1:] = b[1:] + (xi*t)*b[:-1]
        max_seen = max(max_seen, float(np.max(np.abs(x))))
    decoded = z / scale
    return {'r':r,'L':depth,'t':float(t),'dtype':'float32',
            'bitwise_equal_after_decode':bool(np.array_equal(x.view(np.uint32),decoded.view(np.uint32))),
            'max_absolute_unscaled_state':max_seen,'unscaled_output':x.tolist(),
            'decoded_output':decoded.tolist()}


results = {'environment': {'python':sys.version,'platform':platform.platform(),'numpy':np.__version__}}
results['exterior_m3_L100'] = {'raw_gain':gain(np.eye(8), exterior(3), 100, 3),
    'scaled_gain':gain(np.diag([100.0**(-mask.bit_count()) for mask in range(8)]), exterior(3),100,3),
    'condition':100.0**3,'determinant':100.0**(-12)}
results['chains'] = [{'r':r,'L':100, 'raw_gain':gain(np.eye(r+1),chain(r),100,r),
    'scaled_gain':gain(np.diag([100.0**(-j) for j in range(r+1)]),chain(r),100,r)} for r in (1,2,3)]
results['r1_oracle'] = []
for depth,sigma,c in [(1,0.5,2.0),(100,2.0,8.0),(1000,0.1,1.5)]:
    ratio=min(1.0,(c-1)/(depth*sigma**2))
    results['r1_oracle'].append({'L':depth,'sigma':sigma,'C':c,'q':[1,ratio],
         'gain':gain(np.diag([1,ratio]),chain(1,[sigma]),depth,1),'optimal_condition':1/ratio})
results['finite_grade_candidate']=symmetric_grade_opt()
results['power_two_invariance']=power_two_invariance()
print(json.dumps(results,indent=2))

```

### A2. Independent float32 power-of-two negative control

Save as `checks/power_two_control.py`; run `python checks/power_two_control.py`. This reproduces the depth-4,096 example without a conic solver.

```python
"""Negative control: degree scaling can change norms without changing FP error.

Run: python power_two_equivariance.py
No result here is an M3 Max performance or compatibility measurement.
"""
import json
import platform

import numpy as np


def main():
    seed = 56312
    rng = np.random.default_rng(seed)
    r, depth, binary_scale = 12, 4096, 4
    xis = rng.standard_normal(depth).astype(np.float32)
    x = np.zeros(r + 1, dtype=np.float32)
    x[0] = 1
    y = x.copy()
    t = np.float32(2.0 ** -binary_scale)
    for a in xis:
        # NumPy evaluates RHS before assignment, so these are simultaneous
        # updates using the previous layer's state, not an in-place cascade.
        x[1:] = x[1:] + a * x[:-1]
        y[1:] = y[1:] + (a * t) * y[:-1]
    decoded = np.ldexp(y, binary_scale * np.arange(r + 1)).astype(np.float32)
    result = {
        "experiment": "power_two_level_scaling_negative_control",
        "seed": seed,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "depth": depth,
        "nilpotent_depth": r,
        "dtype": "float32",
        "level_scale": float(t),
        "same_decoded_bits": bool(np.array_equal(x.view(np.uint32), decoded.view(np.uint32))),
        "max_abs_difference": float(np.max(np.abs(x - decoded))),
        "max_abs_unscaled": float(np.max(np.abs(x))),
        "max_abs_scaled": float(np.max(np.abs(y))),
        "min_abs_nonzero_scaled": float(np.min(np.abs(y[y != 0]))),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

```

### A3. Full SDP, reduced convex program, rotated failure diagnostic, and independent enumeration

Save as `checks/conic_probe.py`; run `python checks/conic_probe.py > checks/conic_probe_results.json`. This exercises the actual CVXPY interface used in the investigation. It deliberately reports raw solver output; apply the acceptance rules in section 4.7 before treating any candidate as valid. A warning or slight gain violation should remain in the record.

```python
"""Targeted convex-formulation checks; not a benchmark or M3 measurement."""
import json, math, platform, sys, time
from pathlib import Path
import numpy as np
import scipy
import cvxpy as cp
import clarabel, scs


def evaluate(q, factors):
    if q is None:
        return {"positive_definite": None, "reason": "no_metric_returned"}
    q = np.asarray(q)
    if not np.all(np.isfinite(q)):
        return {"positive_definite": None, "reason": "nonfinite_metric"}
    try:
        ev, u = np.linalg.eigh(q)
    except np.linalg.LinAlgError as exc:
        return {"positive_definite": None, "reason": "metric_eigensolver_error",
                "error": str(exc), "q": q.tolist()}
    if ev[0] <= 0:
        return {"positive_definite": False, "min_eigenvalue": float(ev[0]),
                "q": q.tolist()}
    root = (u*np.sqrt(ev))@u.T
    invroot = (u/np.sqrt(ev))@u.T
    mats = [root@a@invroot for a in factors]
    f = np.eye(len(q)) + sum(a.T@a for a in mats)
    b = np.eye(len(q)) + sum(a@a.T for a in mats)
    return {"positive_definite": True, "condition": float(ev[-1]/ev[0]),
            "forward_gain": float(np.linalg.eigvalsh(f)[-1]),
            "adjoint_gain": float(np.linalg.eigvalsh(b)[-1]),
            "q": q.tolist()}


def min_eig_or_none(matrix):
    if matrix is None or not np.all(np.isfinite(matrix)):
        return None
    try:
        return float(np.linalg.eigvalsh(matrix)[0])
    except np.linalg.LinAlgError:
        return None


def full_sdp(factors, c, solver="CLARABEL"):
    if not factors:
        return {"solver": solver, "status": "input_error",
                "error": "Empty factors: dispatch the identity case before this probe."}
    d = factors[0].shape[0]
    q = cp.Variable((d,d), symmetric=True)
    t = cp.Variable()
    fw = (c-1)*q - sum(a.T@q@a for a in factors)
    k = len(factors)
    block = [[(c-1)*q]+[q@a for a in factors]]
    for i,a in enumerate(factors):
        block.append([a.T@q]+[q if i==j else np.zeros((d,d)) for j in range(k)])
    constraints = [q-t*np.eye(d) >> 0, np.eye(d)-q >> 0, fw >> 0, cp.bmat(block) >> 0, t>=0]
    p = cp.Problem(cp.Maximize(t), constraints)
    start = time.perf_counter()
    error = None
    try:
        if solver == "CLARABEL":
            p.solve(solver=solver, tol_gap_abs=1e-11, tol_gap_rel=1e-11, tol_feas=1e-11, max_iter=200)
        else:
            p.solve(solver=solver, eps_abs=1e-8, eps_rel=1e-8, max_iters=200000)
    except (cp.error.SolverError, ValueError, ArithmeticError) as exc:
        error = {"type": type(exc).__name__, "message": str(exc)}
    elapsed = time.perf_counter()-start
    reported_t = None if t.value is None else float(t.value)
    if reported_t is not None and not math.isfinite(reported_t):
        reported_t = None
    out = {"solver": solver,
           "status": p.status or ("solver_error" if error else "no_solver_status"),
           "solver_error": error, "reported_t": reported_t,
           "wall_s": elapsed, "gain_check": evaluate(q.value, factors)}
    out["candidate_available"] = q.value is not None and reported_t is not None
    out["min_forward_lmi_eig"] = min_eig_or_none(fw.value)
    out["min_backward_lmi_eig"] = min_eig_or_none(cp.bmat(block).value)
    bound = None if q.value is None or reported_t is None else q.value-reported_t*np.eye(d)
    out["min_bound_lmi_eig"] = min_eig_or_none(bound)
    return out


def diagonal_log(h, c):
    d = len(h)
    z, tau = cp.Variable(d), cp.Variable(nonneg=True)
    constraints = [z<=0,z>=-tau]
    for axis in (0,1):
        for j in range(d):
            ij = [(a,b) for a in range(d) for b in range(d)
                  if a!=b and h[a,b]>0 and (b if axis==0 else a)==j]
            if ij:
                terms = cp.hstack([math.log(h[a,b])+z[a]-z[b] for a,b in ij])
                constraints.append(cp.log_sum_exp(terms)<=math.log(c-1))
    p=cp.Problem(cp.Minimize(tau),constraints)
    try:
        p.solve(solver="CLARABEL",tol_gap_abs=1e-11,tol_gap_rel=1e-11,tol_feas=1e-11,max_iter=200)
    except (cp.error.SolverError, ValueError, ArithmeticError) as exc:
        return {"status": p.status or "solver_error", "candidate_available": False,
                "solver_error": {"type": type(exc).__name__, "message": str(exc)}}
    if z.value is None or tau.value is None or not np.all(np.isfinite(z.value)):
        return {"status": p.status or "no_solver_status", "candidate_available": False,
                "reason": "no_finite_log_metric_returned"}
    q=np.exp(z.value-np.max(z.value))
    scaled=h*q[:,None]/q[None,:]
    return {"status":p.status,"candidate_available":True,"condition":float(np.exp(np.ptp(z.value))),
            "q":q.tolist(),"forward_gain":float(scaled.sum(axis=0).max()),
            "adjoint_gain":float(scaled.sum(axis=1).max())}


def main():
    depth,r,c=100,3,8.0
    n=np.diag(np.ones(r),k=-1)
    factors=[math.sqrt(math.comb(depth,k))*np.linalg.matrix_power(n,k) for k in range(1,r+1)]
    h=sum(math.comb(depth,k)*np.linalg.matrix_power(n,k) for k in range(r+1))
    gamma=math.comb(depth,2)/depth**2
    delta=math.comb(depth,3)/depth**3
    a=gamma*(c-1)/(math.sqrt(gamma**2+delta*gamma*(c-1))+gamma)
    b=1/gamma
    analytic_q=np.array([1,a/depth,a*b/depth**2,a*a*b/depth**3])
    results={"environment":{"python":platform.python_version(),"platform":platform.platform(),
        "numpy":np.__version__,"scipy":scipy.__version__,"cvxpy":cp.__version__,
        "clarabel":clarabel.__version__,"scs":scs.__version__},
        "case":{"L":depth,"r":r,"C":c},
        "kraus_factors": [factor.tolist() for factor in factors],
        "analytic":evaluate(np.diag(analytic_q),factors),
        "full_sdp":full_sdp(factors,c),"diagonal_log":diagonal_log(h,c)}
    # Exact orthogonal conjugation preserves the unrestricted objective.
    # This float64 QR/factor construction is a numerical perturbation of it;
    # retain the actual factors so validation does not silently regenerate them.
    rng=np.random.default_rng(837142)
    u,_=np.linalg.qr(rng.normal(size=(r+1,r+1)))
    rotated=[u@a@u.T for a in factors]
    results["rotation"] = u.tolist()
    results["rotation_orthogonality_error_fro"] = float(np.linalg.norm(u.T@u-np.eye(r+1), ord="fro"))
    results["rotated_kraus_factors"] = [factor.tolist() for factor in rotated]
    results["rotated_input_semantics"] = "Stored float64 Kraus factors, a numerical perturbation of exact conjugation."
    results["rotated_full_sdp"]=full_sdp(rotated,c)
    # Exact CP gain identity for short Rademacher products, independent path enumeration.
    from itertools import product
    nn=[np.diag(np.array([1.,2.]),k=-1),np.array([[0,0,0],[0,0,0],[1,0,0.]])]
    q=np.array([[2.,.2,.1],[.2,1.5,.1],[.1,.1,1.]])
    exact=np.zeros((3,3))
    for signs in product((-1.,1.),repeat=6):
        p=np.eye(3)
        for ell in range(3):
            w=np.eye(3)+signs[2*ell]*nn[0]+signs[2*ell+1]*nn[1]
            p=w@p
        exact+=p.T@q@p/64
    recur=q.copy()
    for ell in range(3):
        recur=recur+sum(a.T@recur@a for a in nn)
    results["enumerated_moment_identity_error"]=float(np.max(np.abs(exact-recur)))
    print(json.dumps(results,indent=2))


if __name__=="__main__": main()
```

### A4. High-precision analytic oracle and verification of the rotated candidate

Save the following as `checks/high_precision.py` and run it after A3. It reconstructs the same represented rotated factors and checks the stored candidate at 80 decimal digits. Higher precision here checks arithmetic consistency; it is not an outward-rounded interval certificate.

```python
import json
from pathlib import Path
import mpmath as mp

mp.mp.dps = 80
L, C = 100, mp.mpf(8)
gamma = mp.binomial(L, 2) / L**2
delta = mp.binomial(L, 3) / L**3
a = gamma*(C-1)/(mp.sqrt(gamma**2+delta*gamma*(C-1))+gamma)
b = 1/gamma
kstar = L**3/(a*a*b)
u = mp.findroot(lambda x: sum(mp.binomial(L,k)*x**k for k in range(4))-C,
                (mp.mpf('.02'),mp.mpf('.03')))
out = {"decimal_digits": 80,
       "analytic_optimal_condition": str(kstar),
       "best_scalar_condition": str(u**-3),
       "relative_reduction_percent": str(100*(1-kstar*u**3))}

record_path = Path('checks/conic_probe_results.json')
if not record_path.exists():
    out['rotated_check_status'] = 'skipped_missing_A3_record'
else:
    record = json.loads(record_path.read_text())
    candidate = record.get('rotated_full_sdp', {}).get('gain_check', {})
    factors = record.get('rotated_kraus_factors')
    if not candidate.get('positive_definite') or candidate.get('q') is None:
        out['rotated_check_status'] = 'skipped_no_positive_definite_candidate'
    elif factors is None:
        out['rotated_check_status'] = 'skipped_missing_stored_factors_rerun_A3'
    else:
        # JSON float round-trips preserve the represented float64 inputs.
        # Validate those exact stored factors; never recompute QR from a seed.
        q = mp.matrix(candidate['q'])
        e, v = mp.eigsy(q)
        if min(e) <= 0:
            out['rotated_check_status'] = 'rejected_not_positive_definite_at_high_precision'
        else:
            d = len(e)
            root = v*mp.diag([mp.sqrt(x) for x in e])*v.T
            invroot = v*mp.diag([1/mp.sqrt(x) for x in e])*v.T
            forward, backward = mp.eye(d), mp.eye(d)
            for stored_factor in factors:
                transformed = root*mp.matrix(stored_factor)*invroot
                forward += transformed.T*transformed
                backward += transformed*transformed.T
            out['rotated_check_status'] = 'evaluated_stored_float64_factors'
            out['rotated_forward_gain'] = str(mp.eigsy(forward, eigvals_only=True)[d-1])
            out['rotated_adjoint_gain'] = str(mp.eigsy(backward, eigvals_only=True)[d-1])
            out['rotated_condition'] = str(e[d-1]/e[0])
print(json.dumps(out, indent=2))
```

### A5. First execution decision

Begin with WP0 and A1/A2. Proceed to A3 only after native solver capability checks pass. A4 and the analytic formulas adjudicate small disputed results. Then implement the minimum shared harness before increasing scale. A negative application result or an exact prior-art match is a completed scientific outcome when it is documented and reproducible.
