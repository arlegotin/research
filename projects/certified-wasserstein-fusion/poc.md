# Execution brief: does a theorem-derived certificate add value to Wasserstein-median fusion?

**Recipient:** autonomous coding agent configured as **gpt-6.1-sol, xhigh reasoning**.  
**Prepared:** 23 October 2026.  
**Target:** one locally available MacBook with an M3 Max; CPU computation and validated arithmetic.  
**Assignment:** implement and execute the evidence-gated study below, inspect its results, and deliver a reproducible research conclusion. This document is the research plan, not a report of a completed PoC.

## 1. Project decision and contribution

### 1.1 The one question to answer

**Does adding the supplied theorem's data-adaptive tent-function certificate to a strong generic certificate pipeline reduce total fusion latency by at least 10%, while every returned shortcut carries a valid 5% bound relative to the continuous, free-support \(W_1\)-median optimum?**

Test this on the specified synthetic distribution of three two-cloud particle predictions in \(\mathbb R^2\), with equal aggregation weights and supplied hypotheses/assignments. The primary comparison includes feature extraction, candidate construction, compression, mathematical-certificate verification, rejection overhead, and the same raw-particle fallback. It is conditional on that workload and metadata.

The **5% objective tolerance** comes from the supplied application proposal. The **10% latency reduction** is a predeclared R&D investment hurdle: an extra certificate component should produce a material improvement before integration is considered. It is a planning decision, not a measured robotics requirement. Do not lower it after seeing results.

The proposed contribution is deliberately narrow: a constructive, verified certificate derived from the particular disjoint tent map in the supplied proof, together with evidence about its incremental value or redundancy. General Wasserstein medians, medoids, centroid bounds, compression, stability arguments, moment relaxations, and primal-dual certification are established ideas. Historical priority for the supplied theorem or this particular extension is unresolved.

### 1.2 Why the original proposal has been revised

The supplied complementary-measure theorem appears sound, but the original shortcut has two stronger comparators than the headline experiment makes apparent:

1. An ordinary **centroid lower bound already certifies the exact 3–4–5 example within 5%**. The value is \(L_{\rm mean}=1.127738761253718\), against candidate cost \(7/6\), giving a 3.4518549% certificate. This follows from the established centroid transport bound [R3, R4] and the calculation in §2.
2. **Mass-preserving two-cluster compression** leads to an eight-variable continuous multimarginal LP. It can be solved by enumerating 58 precomputed bases, without paying general-purpose optimizer startup on every call. Its generic stability interval is at least as informative as the original forced-50/50 interval when using the same anchors.

Consequently, reproducing a large speed ratio against a particle LP would not establish a new useful mechanism. Study the incremental effect of the tent certificate **after the generic pipeline has exhausted its inexpensive certificates**. Both policies receive identical candidates, input information, arithmetic quality, and fallback. This late insertion deliberately isolates additional certificate strength; it does not test the separate possibility of saving the already-small compressed solve by moving the tent stage earlier.

The original \(2E\) rule remains an ablation. A valid outcome may be that generic compression absorbs the benefit or that rigorous verification costs more than the solver calls avoided.

### 1.3 Scope and exclusions

The primary mathematical problem is

\[
F_p(q)=\frac13\sum_{i=1}^3 W_1(q,p_i),\qquad
F_p^\star=\inf_{q\in\mathcal P_1(\mathbb R^2)}F_p(q),
\]

with ordinary Euclidean ground distance and arbitrary output support. Inputs are finite, exactly normalized probability measures; output is a finite probability measure plus a verifiable objective bound, or an explicit unresolved/failure status.

Do not claim improved localization accuracy, uncertainty calibration, safety, energy use, real-system prevalence, universal speedups, or a new general optimal-transport algorithm. No actual \(W_1\)-median robotics workload has been supplied. The rotation-synchronization literature establishes a motivation for measure-valued uncertain poses, but it uses a different optimization pipeline [R5].

Exclude \(W_2\), squared transport objectives, entropic surrogates, higher-dimensional pose manifolds, learned clustering, more than three input distributions, tree extensions, median-space classifications, and the supplied paper's algebra appendix from implementation. Unequal aggregation weights receive mathematical/unit checks only. Implementing their weighted geometric-median tuple oracle is outside the principal experiment.

Do not train a perception model or procure a proprietary dataset. The deterministic synthetic study below is the authorized accessible study, with an explicit conditional interpretation. If suitable real predictions later become available, they require a separately frozen follow-on protocol.

### 1.4 Fixed choices versus engineering discretion

**Fixed:** objective, continuous support, equal primary weights, tolerance, main generator, shared-candidate comparisons, validated arithmetic in primary timing, independent correctness checks, development/evaluation split, resource ceilings, and outcome rules.

**Autonomous engineering choices:** ordinary file organization within the specified modules, sparse storage details, vectorization, safe caching within one call, and fixes within the repair allowance. Select the permitted baseline variant and numerical precision using development/calibration rules only. Do not introduce a new optimizer, learned router, favorable dataset, GPU backend, relaxed accuracy target, or additional tuning campaign.

## 2. Evidence and mathematics you must preserve

### 2.1 Supplied material and its status

The following files were read during planning:

| Material | Original location in the planning workspace | Status and use |
| --- | --- | --- |
| **wasserstein_geometry_research(1).pdf** | /workspace/scratch/80466e2bb97e/upload/wasserstein_geometry_research(1).pdf | 17-page working note, dated 23 October 2026, attributed to “Autonomous mathematical investigation”; not peer reviewed. Theorem 2.1, pp. 3–4, is essential. |
| **Pasted text(20261023-165551).txt** | /workspace/scratch/80466e2bb97e/upload/Pasted text(20261023-165551).txt | Application proposal and task instructions. Reported experiments/timings are claims, not reproduced evidence. |

SHA-256 identifiers:

~~~text
wasserstein_geometry_research(1).pdf
a4ba0963cd573c87b9891a33a017e79ca0cc0f2a0fab62fd8e9247e821e17df0

Pasted text(20261023-165551).txt
2b7c30483aec8b4c6d47587eb63935711f97dd5a20d77c3fb8d72c74726e798e
~~~

Those paths identify what was inspected; do not assume they exist on the Mac. Put copies in an optional **inputs/sources/** directory if received. This brief contains all mathematics and data-generation decisions needed to execute without them. The referenced **fusion_certificate_experiment.zip**, the note's source archive, and their raw measurements were **not attached or recovered**. Their absence does not block this independent study; do not cite their timings as your results.

The entire PDF, including its classifications, tree theorem, experimental claims, references, and unrelated appendix, was read. Essential theorem pages were also visually inspected after extraction. The core proof was reviewed independently. No formal theorem-prover verification occurred; no broad claim about correctness of every unrelated result is needed here.

### 2.2 Exact complementary theorem

For three distinct points \(x_1,x_2,x_3\) in any metric space, define

\[
\mu_i=\tfrac12\sum_{j\ne i}\delta_{x_j},\quad
S_j=\tfrac12\sum_i\lambda_i d(x_i,x_j),\quad
C=\min_j S_j .
\]

The omitted-point index is the input index. For nonnegative \(\lambda_i\), the supplied Theorem 2.1 proves

\[
\inf_q\sum_i\lambda_i W_1(q,\mu_i)=C,
\qquad
q_0=\mu_{j_0},\quad j_0\in\arg\min_j S_j,
\]

even with arbitrary additional ambient support. Also
\(W_1(\mu_i,\mu_j)=d(x_i,x_j)/2\).
Normalization \(\sum_i\lambda_i=1\) is imposed in this study, though the theorem permits unnormalized weights.

Strict triangle inequalities and strictly positive weights are required only for the note's *complete minimizing-face characterization*. They are not requirements for the optimal-value formula. Do not transfer uniqueness or support characterization to degenerate cases.

There is no hidden optimizer in computing \(q_0\): three distances and three scores suffice. On the exact family this is precisely the optimized input-medoid calculation. Do not claim an output or speed improvement over that calculation.

For equal weights, let \(P=d_{12}+d_{13}+d_{23}\), \(M=\max d_{ij}\), and \(\tau=P-2M\). Then

\[
C=(P-M)/6,\qquad L_{\rm pair}=P/12,\qquad
C-L_{\rm pair}=\tau/12.
\]

For the 3–4–5 triangle: \(C=7/6\), \(L_{\rm pair}=1\), and the probability mixture has objective \(4/3\). This is a consensus-objective comparison; it says nothing about the true location.

### 2.3 Perturbation and compression: established principles, specialized derivation

If explicit couplings show \(W_1(p_i,\mu_i)\le\varepsilon_i\), set \(E=\sum_i\lambda_i\varepsilon_i\). Triangle inequalities give

\[
|F_p(q)-F_\mu(q)|\le E\quad\text{for every }q,\qquad
C-E\le F_p^\star,\qquad F_p(q_0)\le C+E.
\]

Thus the original real-arithmetic bounds are \(F_p(q_0)-F_p^\star\le2E\), and, when \(C>E\), relative gap at most \(2E/(C-E)\).

For assigned particles at the two non-omitted anchors \(x_a,x_b\), let

\[
r_i=\sum_k w_{ik}\|y_{ik}-x_{\ell_{ik}}\|,\quad
\alpha_i=\sum_{\ell_{ik}=a}w_{ik},\quad
b_i=|\alpha_i-\tfrac12|\|x_a-x_b\|.
\]

Then \(\varepsilon_i=r_i+b_i\) is an explicit upper bound. Charge every particle, including outliers and incorrect assignments. Set \(R=\sum_i\lambda_i r_i\), \(B=\sum_i\lambda_i b_i\), so \(E=R+B\).

Alternatively preserve \(\alpha_i\) in the compressed measure \(\widehat p_i\), solve its continuous median, and denote its optimum by \(\widehat C\). Then

\[
\widehat C-R\le F_p^\star,\qquad
F_p(\widehat q)\le \widehat C+R.
\]

For anchor compression, \(W_1(\widehat p_i,\mu_i)=b_i\), hence
\[
|\widehat C-C|\le B,\qquad
[\widehat C-R,\widehat C+R]\subseteq[C-E,C+E].
\]

**Consequence:** the exact compressed solution and its stability interval dominate the original forced-half-mass interval as an acceptance mechanism. The original rule's possible benefit against it is reduced computation, not extra certified coverage. This interval-nesting statement is a derivation for this brief, not a novelty claim.

When candidate selection is numerically ambiguous, use the chosen score \(S_{j_0}\) in an upper bound and \(\min_jS_j\) in a lower bound. Do not assert a literal \(2E\) bound for an incorrectly selected near-tied index. The explicit coupling upper bound below avoids this issue entirely.

### 2.4 Proposed extension: direct tent moments

For \(\{j,k,\ell\}=\{1,2,3\}\), define

\[
a_j=\frac{d(x_j,x_k)+d(x_j,x_\ell)-d(x_k,x_\ell)}2,\qquad
\phi_j(z)=\max(0,a_j-d(x_j,z)).
\]

The source proof shows that \(\Phi=(\phi_1,\phi_2,\phi_3)\) is 1-Lipschitz from the ambient metric space to \(\ell_1^3\). Its positive coordinates have disjoint supports. For arbitrary actual inputs, compute

\[
h_{ij}=\int\phi_j\,dp_i
      =\sum_k w_{ik}\max(0,a_j-\|x_j-y_{ik}\|).
\]

For any output \(q\), its feature expectation has coordinates \(a_jt_j\), with \(t\ge0\) and \(\sum_jt_j\le1\); take \(t_j=0\) if \(a_j=0\). Consequently

\[
\boxed{\quad
L_\Phi=\min_{t\ge0,\;\sum t_j\le1}
\sum_{i,j}\lambda_i|a_jt_j-h_{ij}|
\ \le\ F_p^\star .
\quad}
\]

This is a **new derivation for the proposed study from the supplied proof**. It is not a result stated in the attachment, and historical novelty is not established. Its general ingredients are standard contractive-feature and moment lower bounds.

At exact complementary inputs, \(L_\Phi=C\). Moreover
\[
\|h_i(p_i)-h_i(\mu_i)\|_1\le W_1(p_i,\mu_i)
\quad\Longrightarrow\quad L_\Phi\ge C-E.
\]
The implication follows by the reverse triangle inequality, uniformly in \(t\), followed by minimization.

The direct feature bound does **not** require actual complementarity, two modes, accurate labels, or a small perturbation. Those properties influence usefulness, not validity. Known distinct anchors and a valid ground metric suffice. In particular, a large mismatch must weaken certification or trigger fallback rather than invalidate a purported bound.

It is not guaranteed to dominate the centroid or compressed-optimum bound. Combining valid lower bounds by a maximum is legitimate; adding them is generally not.

### 2.5 Solver-free construction and a separately verifiable witness

For each \(a_j>0\), write \(b_{ij}=h_{ij}/a_j\). The axis cost is convex and piecewise linear, with knots \(0,b_{1j},b_{2j},b_{3j},1\) and slope
\[
g_j(t)=a_j\sum_i\lambda_i\operatorname{sign}(t-b_{ij})
\]
between knots. Starting from \(t=0\), consume negative-slope intervals in increasing slope order until total coordinate allocation reaches one or no negative interval remains. There are at most nine data knots and twelve intervals across all axes. Increasing slopes along each axis preserve the necessary prefix order.

Do **not** trust a floating-point greedy optimum as a lower bound. Instead construct coefficients \(s_{ij}\) satisfying \(|s_{ij}|\le\lambda_i\). For any such coefficients,

\[
\boxed{\quad
L(s)=\min\!\left(0,\min_j a_j\sum_i s_{ij}\right)
      -\sum_{i,j}s_{ij}h_{ij}
\ \le\ F_p^\star .
\quad}
\]

Proof: \(\lambda_i|a_jt_j-h_{ij}|\ge s_{ij}(a_jt_j-h_{ij})\); minimize the resulting affine function over the simplex with vertices \(0,e_1,e_2,e_3\). Thus verification needs only coefficient feasibility and an outward enclosure of this scalar expression. It need not establish greedy optimality, sort ordering, or divide by a small arm.

Recover good coefficients from the greedy result as follows. Let \(\beta\le0\) be the last consumed slope if capacity is exhausted; otherwise \(\beta=0\). Set \(s_{ij}=+\lambda_i\) above a knot, and \(-\lambda_i\) below. At tied knots use a common fraction
\[
\alpha=\operatorname{clip}_{[-1,1]}
\frac{\beta/a_j-\sum_{\text{non-ties }i}s_{ij}}
     {\sum_{\text{ties }i}\lambda_i},
\qquad s_{ij}=\lambda_i\alpha .
\]

Use stored breakpoint identities for exact ties in the floating construction. Represent each clipped \(\alpha\) as the exact rational value of its binary64 representation, then multiply by the exact \(\lambda_i\). In the floating construction, declare an arm unstable if it is nonfinite or \(a_j\le64\,\epsilon_{\rm binary64}D\), and set that coordinate's coefficients to zero. This threshold is only a conservative search rule; verification still uses the true enclosed arm. Any reconstruction error can weaken the witness but cannot invalidate it if its coefficient constraints are enforced exactly.

A finite witness may be weaker than the optimized \(L_\Phi\). If you report implemented nonregression against \(C-E\), explicitly take the maximum with a separately verified \(C-E\). Do not confuse theoretical dominance with a numerically recovered witness.

### 2.6 Generic lower bounds that must be present

**Centroids.** For \(m_i=\sum_k w_{ik}y_{ik}\),
\[
L_{\rm mean}=\min_z\frac13\sum_i\|z-m_i\|\le F_p^\star .
\]
The norm-of-centroid-difference transport inequality is established [R3, Appendix B; R4, Remark 2.2]. The displayed median bound follows by applying it to every candidate \(q\), then minimizing over its possible centroid.

For complementary 3–4–5 inputs, the centroids form a half-scale copy of the triangle:
\[
L_{\rm mean}=\frac16\sqrt{\frac{3^2+4^2+5^2+4\sqrt3(6)}2}
=1.127738761253718 .
\]
For an equilateral triangle, this certificate leaves a relative gap \(2/\sqrt3-1\approx15.47\%\). Neither geometry alone is a sufficient experiment.

**Pairwise metric bound.** If certified \(D_{ij}^{-}\le W_1(p_i,p_j)\) are available,
\[
L_{\rm pair}=(D_{12}^{-}+D_{13}^{-}+D_{23}^{-})/6\le F_p^\star.
\]
One inexpensive construction uses compressed pair distances:
\[
D_{ij}^{-}=\max(0,\widehat D_{ij}^{-}-r_i^{+}-r_j^{+}).
\]
Use exact two-by-two transport endpoint minimization for compressed pairs. Do not solve raw pairwise OT solely to make a “cheap” bound without charging its cost.

**Majority weight.** If an input has weight at least \(1/2\), returning that input is globally optimal in any metric [R2, Example 2.6]. Check this general control; do not count it as evidence for complementary geometry.

## 3. Algorithms, certificates, and numerical conventions

### 3.1 Define the mathematical inputs exactly

Save every coordinate as binary64 and treat its stored binary value as an exact real input. Preserve it in NPZ plus hexadecimal representations in diagnostic records when needed. Do not round-trip through short decimal printouts.

Generated particle masses are nonnegative integer counts with common denominator \(M_w=2^{20}\), summing exactly to \(M_w\) for each distribution. Aggregation weights are exactly \(1/3\), and the acceptance tolerance is exactly \(1/20\). Within a cloud, distribute integer counts as evenly as possible, giving deterministic remainder units to ascending particle indices. Remove zero-mass atoms before constructing LPs and record the resulting sizes.

For external inputs used in an optional future run, normalize exact rational interpretations explicitly and report the normalization; do not silently clip substantial negative masses or discard particles.

The ground metric is computed from these coordinates. No float32, GPU, MPS, CUDA, entropic regularization, or unreported objective rescaling is permitted. Use unit-diameter generators and additional scale/translation tests. Certificates refer to the stored empirical measures, not to an unknown physical distribution behind them.

### 3.2 Rigorous arithmetic is part of the method

Use **python-flint/Arb** for outward ball arithmetic and exact rational helpers. Official documentation describes rigorous error tracking, endpoint operations, and Apple Silicon binary support [R8]. This dependency was researched but **not installed or validated on the target Mac during planning**.

Start at 128-bit Arb precision. Allow one increase to 256 bits for an ambiguous sign/branch/acceptance decision. If still unresolved, return “not certified” and continue to the shared fallback. A float-only exploratory path is permitted for debugging but cannot supply the primary certificate claim or primary timings.

Convert binary64 values using their integer ratios into exact rationals before creating balls. Compute norms, arm lengths, positive parts, moments, and certificate expressions with enclosing operations. Use Arb's enclosing minimum/maximum or explicit lower/upper endpoint formulas; an ordinary comparison between overlapping balls is not a proof. Serialize each exact lower/upper endpoint using its integer mantissa and binary exponent, obtainable from an exact endpoint's man_exp operation; decimal midpoint strings are not the certificate representation.

The decision predicate is
\[
L^{-}>0,\qquad 20\,U^{+}\le21\,L^{-}.
\]
Evaluate it rigorously without dividing by a nearly zero lower bound. It implies
\[
F_p(q)-F_p^\star\le U^{+}-L^{-},\qquad
\frac{F_p(q)-F_p^\star}{F_p^\star}\le\frac{U^{+}}{L^{-}}-1\le0.05.
\]
If a verified feasible coupling has exact upper endpoint zero, return it with an exact-zero certificate: nonnegativity proves \(F_p(q)=F_p^\star=0\). Define its reported relative gap as zero by convention and verify the equivalent inequality \(F_p(q)\le1.05F_p^\star\). Otherwise, if \(L^{-}=0\), report an absolute bound only and continue to fallback. Do not relabel an absolute tolerance as the required relative guarantee.

Cache verified distances within one policy call. Both policies use the same caching rules; the baseline must not compute tent-only distances in advance. Conversion to exact inputs, feature evaluation, interval verification, and precision retries belong inside the primary timer.

### 3.3 Two-target upper bound for the common candidate

Choose \(j_0\) from the three ideal scores, with deterministic index tie-breaking, and let \(q_0=\frac12(\delta_A+\delta_B)\) be the corresponding complementary measure. Both policies use this same heuristic candidate.

For each \(p_i=\sum_kw_k\delta_{y_k}\):

~~~text
calculate float64 dA[k], dB[k]
order particles by (dA[k] - dB[k], original_index)
remaining_A = M_w / 2                # exact integer counts
for k in this order:
    mass_A[k] = min(count[k], remaining_A)
    mass_B[k] = count[k] - mass_A[k]
    remaining_A -= mass_A[k]
require remaining_A == 0
enclose sum_k (mass_A[k]*||y_k-A|| + mass_B[k]*||y_k-B||)/M_w
~~~

Aggregate the three upper endpoints with exact weights. This is an \(O(n\log n)\) transport calculation with an explicit feasible coupling. Wrong float ordering cannot make its verified upper bound invalid.

For diagnostic objective intervals, also use the two-target transport dual: for any exact rational scalar \(v\),
\[
W_1(p_i,q_0)\ge
\tfrac12v+\sum_k w_k\left[d_B(k)+\min(0,d_A(k)-d_B(k)-v)\right].
\]
Use a float boundary cost difference as a rational witness and enclose the expression. This checks the achieved candidate cost without trusting the sort.

### 3.4 Compressed solver: require the strong implementation

For each input preserve its two assigned cloud masses. Use two compression variants:

* **Anchor compression:** each cloud collapses to its supplied anchor.
* **Centroid compression:** each cloud collapses to its own weighted centroid, rounded once to binary64 and thereafter treated exactly. If a cloud has zero mass, retain a harmless placeholder center and zero mass. Charge its construction and the explicit particle-to-center cost.

This second variant prevents a weak baseline caused by forcing generic compression to use imperfect common anchors. Its six centers need not coincide. Each variant still has two atoms per input and eight multimarginal variables. A centroid is an inexpensive chosen representative; the geometric median would minimize a cloud's \(W_1\) collapse cost. Do not generalize this comparison to every compression scheme, and do not add another cluster optimizer to the present study.

For tuples \(b\in\{0,1\}^3\), calculate
\[
c_b=\min_y\tfrac13\sum_{i=1}^3\|z_{i,b_i}-y\|.
\]
Let \(A_b=(1,b_1,b_2,b_3)^T\), and let the right-hand side be \(r=(1,\alpha_{11},\alpha_{21},\alpha_{31})^T\), where \(\alpha_{i1}\) is the mass at atom 1. Solve
\[
\min_{\gamma\ge0}c^T\gamma,\qquad A\gamma=r.
\]
This is the discrete specialization of the established continuous multimarginal formulation [R2, §5.1, Eq. (39)–(41), Theorem 5.1].

At startup enumerate all 70 four-column subsets; retain the 58 nonsingular matrices and compute their inverses with exact rational arithmetic. This count was checked during planning; verify it independently. Fifty-six determinants have magnitude one and two have magnitude two. Do not hard-code a floating inverse without validation.

At each call:

1. Evaluate all basic solutions with exact rational marginal arithmetic; feasibility is exact, including zero components.
2. Every feasible basis gives a feasible coupling. Bound its exact cost using the eight tuple-cost intervals.
3. The minimum of the lower cost bounds over **all feasible bases** is a lower bound \(\widehat C^{-}\) on the compressed optimum. The minimum upper cost bound is an optimum upper bound.
4. Select a feasible basis by a deterministic upper-cost rule. For each used tuple construct a geometric-median point, store its rounded binary64 coordinates, and push the basis masses to those points.
5. Re-evaluate explicit tuple-to-output transport distances for those rounded points. Their weighted sum \(\widehat U\) bounds the compressed candidate's actual objective.
6. If \(R^{+}\) bounds the explicit compression cost, return \(L_c=\max(0,\widehat C^{-}-R^{+})\), candidate \(q_c\), and \(U_c=\widehat U+R^{+}\).

This formulation returns its own candidate; never restrict it to \(q_0\). Its output has at most four tuple points before merging duplicates. In a strict real-arithmetic optimum \(\widehat U=\widehat C\); the implementation explicitly accounts for rounding and candidate reconstruction.

Validate the enumerator against HiGHS and tiny exact rational cases before timing it. Include the HiGHS eight-variable version only as an overhead-control result, not as the strongest baseline. Report one-time enumeration/import cost separately and per-call costs inside timing.

### 3.5 Three-point Euclidean median primitive

Use one tested primitive for compressed costs, centroids, and raw tuple costs. For points \(z_1,z_2,z_3\), let \(s_i\) be the side opposite \(z_i\), and let \(\Delta\) be triangle area.

* Coincident/repeated points: resolve directly.
* If a vertex has angle at least \(120^\circ\), that vertex is a median. With \(u=z_j-z_i\), \(v=z_k-z_i\), \(t=u\cdot v\), \(A=u\cdot u\), \(B=v\cdot v\), use the exact rational test \(t\le0\) and \(4t^2\ge AB\), after handling \(AB=0\). This is equivalent to \(t\le-\sqrt{AB}/2\) and avoids a floating branch decision.
  The minimum unweighted sum is the sum of its two incident side lengths.
* Otherwise the minimum unweighted sum is
  \[
  G=\sqrt{\frac{s_1^2+s_2^2+s_3^2+4\sqrt3\,\Delta}{2}}.
  \]
  Divide by three for this study's cost.
* For a constructive interior point, use barycentric weights proportional to
  \[
  \omega_i=\frac1{4\Delta+\sqrt3(s_j^2+s_k^2-s_i^2)},
  \qquad y=\frac{\sum_i\omega_i z_i}{\sum_i\omega_i}.
  \]
  Near a vanishing denominator, use the certified vertex/branch logic and the allowed precision increase. Any returned point is evaluated as an explicit feasible candidate; its approximate optimality is not trusted.

For stored input coordinates, rounded compressed centers, and exactly accumulated rational centroids, the rational test decides the branch exactly. Use outward intervals for the value within that branch. If an internal routine instead receives interval-valued coordinates and cannot decide its branch after the precision allowance, use a safe lower bound equal to the minimum of the lower endpoints of all three vertex sums and the interior expression, and a safe upper bound equal to the minimum of the three vertex upper sums. The true branch is among these possibilities. Do not make an uncertified branch choice.

The interior cost formula follows from three \(120^\circ\) rays at an interior median: if their lengths are \(r_i\), then
\(\sum s_i^2=2\sum r_i^2+\sum_{i<j}r_ir_j\) and
\(\Delta=\sqrt3\sum_{i<j}r_ir_j/4\); substituting gives \(G=\sum r_i\).
Check the barycentric construction against stationarity and independent convex optimization on tiny cases. This primitive must handle skinny and nearly \(120^\circ\) triangles correctly.

### 3.6 Tent routine and primary policy

~~~text
tent_witness(inputs, anchors):
    calculate float arms and actual tent moments
    build ordered knots and negative-slope intervals for each stable arm
    t = (0,0,0); capacity = 1; beta = 0
    consume intervals by (slope, axis, lower_endpoint)
    set fully consumed endpoints to their stored knot values
    when capacity is exhausted set beta to the last consumed slope
    reconstruct alpha in [-1,1] using the subgradient rule
    convert alpha to exact rational values; s[i,j] = alpha[i,j]/3
    verify coefficient boxes exactly
    evaluate L(s) with verified actual distances and moments
    return max(0, lower_endpoint(L(s))) and the witness

generic_prefix(instance, include_raw_medoid):
    validate input; construct common q0 and verified U0
    L = verified centroid lower bound
    if certify(U0, L): return ACCEPT(q0, U0, L)
    for compression in [anchor, per_cloud_centroid]:
        obtain candidate qc, Uc, Lc, and inexpensive compressed pair lower bounds
        retain candidate with lowest verified U; L = max(L, Lc, Lpair)
        if certify(best_U, L): return ACCEPT(best_q, best_U, L)
    if include_raw_medoid:
        compute verified raw pairwise distances and an input-medoid candidate
        add its U and its pairwise lower bound to the same pool
        if certify(best_U, best_L): return ACCEPT(best_q, best_U, best_L)
    return UNRESOLVED(best_q, best_U, best_L, reusable_state)

baseline(instance):
    result = generic_prefix(instance, frozen_medoid_setting)
    if result is ACCEPT: return result
    return shared_raw_fallback(instance)

proposed(instance):
    result = generic_prefix(instance, frozen_medoid_setting)
    if result is ACCEPT: return result
    Lt = tent_witness(instance)
    L = max(result.L, Lt)
    if certify(result.U, L): return ACCEPT(result.q, result.U, L)
    return shared_raw_fallback(instance)
~~~

The primary proposed policy does not reorder the generic stages or introduce new candidates. Tent acceptance can only reduce the fallback set, although its cost can increase total runtime. Do not move it earlier after seeing confirmation results.

The **raw-medoid setting** is the single permitted baseline-strengthening choice. During the pilot run three randomized-order paired repetitions of the baseline with this stage on versus off, including its three pairwise LPs, verification, and saved fallback work. For each repetition sum time across all pilot cases. Select “on” only if it has a smaller aggregate in all three paired repetitions; otherwise select “off,” including exact ties. Both policies inherit the same frozen setting. Report all variant totals. This deterministic, conservative selection rule avoids an undefined “timing noise” threshold and is bounded baseline selection, not a new research project.

### 3.7 Continuous raw-particle reference and fallback

For \(p_i=\sum_{k=1}^{n_i}w_{ik}\delta_{y_{ik}}\), build
\[
\min_{\Gamma\ge0}\sum_{abc}c_{abc}\Gamma_{abc},
\quad
\sum_{bc}\Gamma_{abc}=w_{1a},\
\sum_{ac}\Gamma_{abc}=w_{2b},\
\sum_{ab}\Gamma_{abc}=w_{3c},
\]
where \(c_{abc}=\min_z\frac13(\|z-y_{1a}\|+\|z-y_{2b}\|+\|z-y_{3c}\|)\).

This represents the unrestricted continuous optimum [R2]. A finite LP limited to input sites or an arbitrary grid generally does not. Such an LP supplies a feasible upper comparison, not a continuous lower oracle.

Use SciPy HiGHS on CPU with sparse constraints and an explicit method, initially **highs-ds**. Numerical solver success is not certification [R7]. Read the actual installed documentation, record statuses, residuals, marginals, and supported options. No fake warm start through SciPy's x0 argument.

Construct verified bounds even from imperfect numeric results:

**Dual lower bound.** Interpret finite solver dual potentials \(u_i\) as exact binary rationals; if potentials are absent or nonfinite, use the all-zero feasible dual for a lower bound of zero and record the defect. With rigorously lower-bounded tuple costs \(c_{abc}^{-}\), compute
\[
\delta=\max\left(0,\max_{abc}(u_{1a}+u_{2b}+u_{3c}-c_{abc}^{-})\right)
\]
using an upward bound. Then
\[
L_{\rm raw}=\sum_{i,k}w_{ik}u_{ik}-\delta
\]
is valid: subtracting \(\delta\) from every potential in the first marginal repairs all constraints, and that marginal has mass one. If using a reduced constraint matrix, restore missing potentials as zero and verify all full tuple constraints.

**Primal repair.** Use an integer transport grid \(K=2^{48}\), divisible by the input mass denominator. Reject NaN/Inf solver entries as an unusable proposed plan; a residual-only greedy plan is still a valid diagnostic upper candidate. For every finite entry set its proposed count to \(\max(0,\lfloor K\,\Gamma_{abc}^{\rm float}\rfloor)\), taking the floor of its exact binary-rational interpretation. Process positive proposed counts in descending order, with lexicographic ties. Allocate the minimum of that count and the three remaining marginal counts. All allocations are nonnegative integers. Then greedily match the first nonempty residual bin in each marginal, allocating their minimum and advancing emptied bins. This fills all residuals exactly in at most \(n_1+n_2+n_3-2\) additional tuples. It is always feasible; do not assume the repair is small without measuring it.

Push repaired tuple mass to stored candidate geometric-median points, and sum verified tuple-to-point distances for a feasible objective upper bound \(U_{\rm raw}\). The fallback may return when \(L_{\rm raw}>0\) and \(20U_{\rm raw}\le21L_{\rm raw}\), even if the numerical solve is not highly precise; handle exactly zero objectives by the separate zero proof. Otherwise permit one second solve using the explicitly permitted HiGHS highs-ipm method after highs-ds. The entire fallback has a 30-second ceiling, with at most 15 seconds for each attempt, including construction, solve, repair, and verification; a second attempt may reuse already constructed within-call data under the same rule in both policies. Global and stage budgets may forbid the second attempt. Then return unresolved. Record the returned distribution, bound interval, repair mass, and solver status.

For offline reference diagnostics aim for \(U_{\rm raw}-L_{\rm raw}\le10^{-7}D\), where \(D\) is anchor diameter. This accuracy is far smaller than a 5% objective tolerance on the intended geometries. It is a reference-quality target, not an extra requirement imposed on only one policy. If the verified interval remains wider, retain it and report interval-valued errors instead of inventing an exact optimum.

Use the same fallback function, backend, precision, time limit, and validation for both policies. An explicit exact-solver fallback is the scope of the latency claim; it does not establish superiority over all possible target-gap iterative solvers.

## 4. Implementation contract

### 4.1 Minimal stack and environment inspection

Use native arm64 CPython 3.12 if available, NumPy, SciPy, python-flint, pytest, psutil, and Matplotlib. Ordinary scripts and JSON/CSV/NPZ files suffice. No service, dashboard, database, container cluster, distributed runner, or training framework is needed.

Initialize a persistent session-start timestamp before setup, and inspect the interpreter with only the standard library before installing packages. Select a native arm64 interpreter; the commands below use “python3” to mean that inspected interpreter, not an unchecked system alias. Before installation or timing record:

* machine model, CPU architecture, macOS version, Python architecture and version;
* physical RAM, currently available memory, free project-volume storage, power state;
* NumPy/SciPy/BLAS/HiGHS versions and configuration, actual solver method and thread behavior;
* installed Arb/FLINT versions, interval precision, and the success of an exact-rational/square-root enclosure smoke check.

Do not infer RAM or backend capabilities from “M3 Max.” Do not run the primary experiment under Rosetta without explicitly reporting it and narrowing the target-hardware claim. Prefer a native interpreter; allow one bounded installation repair.

Suggested initial dependency ranges are Python 3.12, NumPy 2.x, SciPy 1.15–1.18, python-flint 0.8–0.9, and currently compatible pytest/psutil/Matplotlib wheels. These are compatibility intentions, not a tested Mac lockfile. Request binary wheels, inspect actual APIs, and freeze exact installed versions after tiny correctness. Hosted planning probes used NumPy 2.3.5 and SciPy 1.17.0; that is not evidence for Mac compatibility or performance.

Set common numerical-library thread limits to one before importing them, record whether they are honored, and use one experiment worker. Do not run competing benchmark jobs in parallel. If SciPy exposes unsupported-option warnings, fix the configuration rather than ignoring them. Pin the chosen explicit HiGHS method; automatic backend switching would complicate timing attribution.

If validated arithmetic cannot be installed within the repair allowance, finish the analytic derivation and float correctness diagnostics, label the machine-certified engineering study blocked, and stop before primary performance claims. Do not substitute an arbitrary epsilon margin or unvalidated high precision.

### 4.2 Repository responsibilities

Create this small structure; file names below are interfaces to implement, not files already supplied:

~~~text
README.md
pyproject.toml
requirements.lock
configs/study.json
src/fusioncert/
    __init__.py
    data.py          # deterministic instances; exact masses; metadata/provenance
    geometry.py      # distances; three-point median values and candidate points
    verify.py        # exact conversions; Arb enclosures; decision predicate
    tent.py          # float greedy; exact witness; verified tent lower bound
    candidates.py    # q0; two-target coupling/dual; input medoid interfaces
    compressed.py    # rational basis cache; two compression variants
    reference.py     # sparse MM/pair LPs; dual and primal repairs; fallback
    policies.py      # shared generic prefix; baseline; late tent extension
    ledger.py        # cumulative time/call/memory/disk budget enforcement
    cli.py           # stage runner and report commands
tests/
results/
    environment.json
    ledger.jsonl
    runs.jsonl
    instances/
    witnesses/
    tables/
    figures/
report/
    RESEARCH_REPORT.md
    protocol_frozen.json
    review_and_deviations.md
~~~

Do not implement later-stage plotting or abstractions until the corresponding evidence gate passes. A stopped study needs only the modules and evidence for stages reached.

Required logical interfaces:

~~~text
Instance:
    id, phase, block_id, geometry_stratum, sigma, imbalance
    anchors[3,2], points[3,n_i,2], integer_counts[3,n_i], denominator
    two_cloud_labels, omitted_anchor_mapping, exact aggregation weights
    seed/provenance, actual dimensions, validation status

BoundedCandidate:
    support coordinates, exact rational output masses
    objective_upper, optional objective_lower
    feasible transport witness or explicit composition recipe

Certificate:
    lower_bound, upper_bound, precision, exact acceptance predicate
    bound_type, witness payload, input/config hashes, verifier version

PolicyResult:
    status, candidate, certificate, branch trace, elapsed_ns
    stage timings, solver calls/statuses, precision retries, repair statistics

solve_compressed(instance, centers_variant) -> bounds + candidate
tent_witness(instance, reusable_distances) -> lower bound + exact coefficients
solve_reference(instance, quality_target, budget) -> verified interval + candidate
run_policy(instance, policy_name, frozen_config) -> PolicyResult
verify_saved_certificate(instance, result) -> pass/fail + explanation
~~~

Witness files must be independently replayable without rerunning the floating optimizer. Record exact rational coefficients/counts, the binary64 candidate coordinates, and sufficient input identifiers. A checksum of an untrusted float objective is not a certificate.

### 4.3 Intended commands

Implement and document these commands. They are **proposed interfaces**, not commands already run by the research lead:

~~~bash
python3 - <<'PY'
import json, platform, sys, time
from pathlib import Path
Path("results").mkdir(exist_ok=True)
p = Path("results/session.json")
if not p.exists():
    with p.open("x", encoding="utf-8") as f:
        json.dump({"started_unix_ns": time.time_ns(),
                   "python": sys.version, "architecture": platform.machine(),
                   "platform": platform.platform()}, f, indent=2)
print(p.read_text(encoding="utf-8"))
PY
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --only-binary=:all: 'numpy>=2,<3' 'scipy>=1.15,<1.19' 'python-flint>=0.8,<0.10' pytest psutil matplotlib
python -m pip install -e .

python -m fusioncert.cli preflight --config configs/study.json
python -m pytest -q
python -m fusioncert.cli smoke --config configs/study.json
python -m fusioncert.cli calibrate --config configs/study.json
python -m fusioncert.cli pilot --config configs/study.json
python -m fusioncert.cli freeze --config configs/study.json
python -m fusioncert.cli confirm --protocol report/protocol_frozen.json
python -m fusioncert.cli verify-results --protocol report/protocol_frozen.json
python -m fusioncert.cli report --protocol report/protocol_frozen.json
~~~

The setup example requests the proposed ranges; enforce the same constraints in the package configuration and inspect what was actually selected. Once correctness passes, save a lockfile. Future reproduction should install that lockfile, not fetch unconstrained latest versions. Do not upgrade dependencies during confirmation.

For an authorized stop before protocol freeze, implement and use:

~~~bash
python -m fusioncert.cli verify-results --config configs/study.json --reached-only
python -m fusioncert.cli report --config configs/study.json --reached-only
~~~

These modes read the environment, session, ledger and completed-stage outputs; they must not invent confirmation results or require a nonexistent frozen protocol. If implementation stops before a CLI is usable, write the analytic/blocker report directly from the preserved evidence.

Make the CLI resumable by immutable run IDs. Re-execution must not overwrite failures or reset the budget ledger. Retrying a failed run creates a linked new row and consumes the remaining budget. The confirmation command must refuse to run if the frozen protocol hash differs.

Use results/session.json and results/ledger.jsonl as the shared session/ledger paths for CLI commands and pytest. Implement a pytest fixture or wrapper that records numerical instances, method/LP starts, failures and outer elapsed test time, so direct test reruns cannot evade accounting. Record outer execution intervals once rather than double-counting nested helper timers; operation counters remain cumulative. Installation and coding time count toward the 12-hour session even when they are outside the four-hour experimental envelope.

### 4.4 Configuration and reproducibility

The initial configuration must contain the scientific constants in this brief, all caps, seed roots, precisions, solver options, candidate rules, timing boundaries, and baseline-selection rule. After the pilot, finish the commands needed for confirmation, verification and reporting, then freeze the protocol. Hash the scientific source, tests, configuration, package metadata and dependency lock; explicitly exclude generated results, logs, witnesses, figures, reports, session files and the protocol file itself. Writing results must not invalidate the protocol. Changes to scientific code after freezing require the deviation/fresh-confirmation rule in §7.1. Record generated-instance hashes and all exact input sizes.

Use a seed hierarchy based on NumPy SeedSequence with components
\((20261023,\text{phase},\text{stratum},\text{block},\text{substream})\).
Assign phase codes: 10 correctness, 20 calibration, 30 pilot, 40 confirmation, 50 stress, 60 scaling. Save the component tuple and generated arrays. Use explicit PCG64 and record NumPy version; saved arrays are the definitive data if generator-version behavior changes.

Do not use planning seed 70291 or any planning-probe arrays for confirmation. Do not select or reject a confirmation instance according to certificate behavior.

## 5. Experiments and independent checks

### 5.1 Tiny correctness: establish trusted behavior first

Use the following checks because they address plausible scientific errors:

| Check | Required expected behavior | Error detected |
| --- | --- | --- |
| Exact 3–4–5 complementary inputs | \(C=7/6\); mixture \(4/3\); medoid and theorem candidate coincide; centroid certificate about 3.45185% | Wrong indexing, normalization, or attribution |
| Exact equilateral complementary inputs | \(C=D/3\); tent reaches \(C\); centroid gap about 15.47%; multiple ideal minimizers permitted | Incorrect uniqueness assumptions; weak baseline control |
| Collinear, coincident-point, zero-weight, and dominant-weight cases | Appropriate scalar/degenerate solution or exact majority rule; no unsafe division | Invalid assumptions near boundaries |
| Three Dirac inputs at an equilateral triangle | Continuous optimum \(D/\sqrt3\), while restricting output to vertices gives \(2D/3\) | Confusing finite support with free support |
| 128 randomized feasible moment arrays | Greedy and its reconstructed witness agree with independent 12-variable epigraph LP to calibrated numerical precision | Tent optimizer/indexing defects |
| At least 24 random compressed two-atom problems, including zero masses and distinct six centers | 58-basis solver agrees with HiGHS; exact marginal feasibility; certified bounds contain reference | Baseline incorrectly specialized to equal masses/common centers |
| Two-target transport on 24 small particle cases | Coupling feasible; upper and scalar-dual lower agree within interval/numerical targets; agree with ordinary OT LP | Sort, split-mass, or cost errors |
| Tiny raw MM problems with at most 3 atoms/input | Agreement with a separately assembled joint-transport LP using all tuple-median output sites, allowing the rounding enclosure | Wrong marginal tensor/tuple costs |
| Deliberately damaged witnesses | Coefficient-box violations, incorrect output masses, and omitted transport costs are rejected | Verifier that only rechecks the implementation's assertions |
| Permutations, translations, scale factors \(2^{-20},2^{20}\) | Objective scales correctly; equivalent cases preserve certificate decisions away from an enclosure boundary | Conditioning, unit, or reproducibility errors |

For the moment epigraph oracle, introduce \(t_1,t_2,t_3\ge0\), nine \(v_{ij}\ge0\), \(\sum t_j\le1\), and constraints \(v_{ij}\ge\pm(a_jt_j-h_{ij})\); minimize \(\sum_{ij}\lambda_i v_{ij}\). Generate nonnegative arms and rows \(h_i=a\odot r_i\) with \((r_{i1},r_{i2},r_{i3},r_{i4})\) drawn from Dirichlet(1,1,1,1). Include exact rational/tied arrays as deterministic cases.

The tiny joint-transport oracle has shared output masses and separate transports from each input, minimizing their weighted total cost. When the output sites include every exact tuple median, the continuous optimum is attainable there. With rounded sites it is an upper reference whose small support-rounding error must be bounded. Do not use an arbitrary grid and assert equality.

Unequal-weight correctness checks apply to the theorem scores, tent/epigraph relaxation, and generic majority argument. Do not pass unequal weights to the equal-weight three-point cost primitive or call that an unrestricted weighted reference.

Set float comparison tolerances during calibration from independent residuals and verified objective enclosures. A starting diagnostic tolerance of \(10^{-9}D\), with an intended maximum \(10^{-7}D\), is reasonable for these normalized small problems; these are diagnostics, not substitute certificate margins. Investigate discrepancies larger than the justified tolerance. The interval acceptance predicate itself has no empirical epsilon padding.

### 5.2 Main synthetic population

Use **16 particles per input**, eight per labelled cloud, and exactly three inputs. The four geometry strata below have equal primary weight. Create eight independent blocks per stratum, for 32 blocks in total.

Before normalization, draw prototypes as follows:

| Stratum | Prototype construction |
| --- | --- |
| Near equilateral | \(x_1=(0,0), x_2=(1,0), x_3=(u,v)\), \(u\sim U[0.4,0.6]\), \(v\sim U[0.75,0.95]\) |
| Near 3–4–5 | \(x_1=(0,0), x_2=(0.6,0), x_3=(u,0.8+v)\), \(u,v\sim U[-0.03,0.03]\) |
| Slender | \(x_1=(0,0), x_2=(1,0), x_3=(u,h)\), \(u\sim U[0.2,0.8]\), \(\log_{10}h\sim U[-4,-1.5]\) |
| Irregular, including obtuse shapes | \(x_1=(0,0), x_2=(1,0), x_3=(u,v)\), \(u\sim U[-0.3,1.3]\), \(v\sim U[0.15,1.2]\) |

Normalize each triangle by its maximum pairwise distance, then apply one block-specific random planar rotation and translation in \([-1,1]^2\). Store the resulting binary64 anchors. Recompute their actual diameter \(D\) for bounds and scaling; do not assume its stored value is exactly one.

For input \(i\), its designated clouds correspond to the other two anchors, in increasing anchor-index order. Generate independent standard-normal 2D displacement arrays for all input/cloud/particle combinations. **Do not enforce antithetic symmetry or zero sample means**: empirical centroid drift is part of the comparison, and centroid compression may exploit it.

Within each block cross:

* position noise \(\sigma/D\in\{0.001,0.004,0.016,0.064\}\);
* imbalance magnitude \(\beta\in\{0,1/32,1/8\}\).

For each input draw one independent sign \(\epsilon_i\in\{-1,+1\}\), fixed within the block. Its two cloud masses are \(1/2+\epsilon_i\beta\) and \(1/2-\epsilon_i\beta\). Particle positions are \(x_{\ell}+\sigma g_{ik}\); distribute each cloud's exact mass over its eight particles. Reuse standardized displacements and imbalance signs across a block's twelve cells to isolate perturbation effects.

This gives **384 primary instances**. They are a declared synthetic mechanism population, not a model fitted to a tracker. The logarithmically spaced noise levels cover very tight through diffuse clouds, and imbalance is varied separately from position noise. All cells have nonzero position noise; noiseless exact cases are controls, not part of the primary mixture.

The independent statistical unit is the geometry/noise **block**, not a particle, cell, policy repeat, or solver iteration. Preserve a block's correlated cells in every resampling calculation.

The only pre-lock workload reduction is from eight to six complete blocks per stratum, yielding 288 primary instances. Use it solely if the cost forecast requires it. Do not remove a noise level, imbalance level, or geometry stratum to make the result look better.

### 5.3 Development and calibration instances

Use different phase seeds from confirmation:

* Correctness cases as above, with small supports.
* Calibration: at least four representative geometry/perturbation cases at each of \(n=8,16,32\), covering a likely acceptance and a likely rejection. Stop larger allocations if the smaller observations already exceed a cap.
* Pilot: one independent block per geometry stratum, all four noise levels, and \(\beta\in\{0,1/8\}\): 32 cases.

The primary support size remains \(n=16\). A run at \(n=8\) can support a bounded diagnostic if \(n=16\) is infeasible, but cannot satisfy the planned primary scale claim. Do not enlarge the main LP merely to inflate fallback cost and manufacture a speed ratio.

During the pilot, compare the permitted raw-medoid baseline variants. A raw input medoid uses the three pairwise \(W_1\) intervals, selects an input with minimal verified upper score, and uses \(L_{\rm pair}\) plus the other generic lower bounds. Verify pairwise LPs using the same primal/dual principles with two marginals. Include ordinary probability mixing as an objective-only diagnostic on the exact controls and eight preselected pilot cases; it is not a certified-accuracy competitor.

### 5.4 Small robustness set

If confirmation is reached, reserve up to 32 separately reported stress instances, eight per group. They do not change the primary mixture.

1. **Metadata errors:** at \(n=16,\sigma/D=0.004,\beta=0\), flip labels of the first two particles in each input, or shift each supplied anchor by \(0.05D\) while retaining the true generated particles. Use four geometry strata and both modifications. Charge actual distances to every chosen compression center.
2. **Additional modes/outliers:** take the same four strata and move one equal-mass particle from each cloud either to the omitted anchor or to \(x_{\ell}+(3D,0)\). Keep its assigned label and all its mass. These eight cases test honest accounting for noncomplementary structure.
3. **Unrelated distributions:** eight seeds with each input containing 16 independent uniform points in the square \([-1,1]^2\), equal masses, anchors fixed at \((0,0),(1,0),(0,1)\), and alternating two-cloud labels. The bounds remain valid; usefulness is not assumed.
4. **Numerical boundaries:** one duplicate-anchor case; one exactly collinear case; two triangles at a \(120^\circ\) angle perturbed by \(\pm2^{-40}\) radians; two tiny-arm cases with heights \(2^{-20}\) and \(2^{-30}\); and two translated/scaled copies of a normal case using the factors in §5.1. Invalid numeric inputs such as NaN or negative mass belong in validation tests, not performance averages.

For duplicate anchors, disable the tent component or merge its degenerate features conservatively and let the generic/fallback policy proceed. Do not divide by an uncertified tiny arm. A rejected shortcut is correct behavior; a failed complete policy is a separately reported failure.

Clustering/hypothesis discovery is not implemented in this primary study. As a cost sensitivity, report the maximum additional preprocessing time the measured benefit could tolerate. A small optional nearest-of-the-two-given-anchors assignment timing may illustrate label-assignment cost; it does not measure discovery of the anchors themselves.

### 5.5 Ablations and controls

Reuse already computed validated quantities wherever these are offline diagnostics; do not count diagnostic reuse as deployable timing:

| Comparison | Purpose |
| --- | --- |
| Original \(C\pm E\) interval and \(2E\) rule | Reproduce the supplied acceptance logic without claiming novelty |
| \(C-E\) with the common tight \(U_0\) | Separate better upper-bound evaluation from a better lower bound |
| Centroid certificate alone | Test the established cheap mechanism, including the headline 3–4–5 case |
| Anchor compression alone versus added per-cloud centroid compression | Detect gains caused by a weak choice of cluster representatives |
| Generic pipeline versus the same pipeline plus late tent | Primary incremental comparison |
| Same candidate pool, \(L_{\rm generic}\) versus \(\max(L_{\rm generic},L_{\rm tent})\) | Attribute acceptance changes to the lower bound |
| Eight-variable HiGHS versus basis enumeration | Identify generic solver-overhead effects |
| Raw input medoid | Show exact-family equivalence and assess a stronger generic candidate on perturbed inputs |

Use the same raw inputs, metadata, exact weights, tolerance, verification, and candidate budgets for matched comparisons. No method receives ground-truth cloud centers or labels that its competitor does not. There is no “localization accuracy” metric without a separate generative truth model and justified statistical objective; do not invent one here.

### 5.6 Primary measurements and analysis

Time the fusion call from in-memory input arrays and metadata to a returned distribution plus verified certificate/fallback result. Include input validation, exact conversions, all geometry, feature extraction, compression, sorting, basis work, solver model construction, interval arithmetic, repairs, and branch overhead. Exclude deterministic data generation, input-file loading, and report plotting, but record their total costs separately.

Run three paired repetitions per primary instance, randomizing baseline/proposed order within each pair. Warm imports and immutable basis tables once; reset per-call workspaces and caches between calls. No reuse of a previous policy's distances, LP solution, features, or certificate is allowed in primary timing. Retain all three observations; use their per-instance median for the primary aggregate. CPU execution needs no GPU synchronization.

Record cold process/import/basis setup separately. Only small stage microbenchmarks may use batching: choose a batch count during calibration to reach roughly 20–100 ms, cap it at 512, and charge every invocation to the ledger. Do not multiply the expensive fallback by a microbenchmark batch count.

Measure wall time with a monotonic nanosecond timer. Measure peak RSS for the process tree, including native solver/Arb allocations, with a separate instrumented calibration/run. Check macOS maximum-RSS units; Python allocation tracers alone miss native memory. Record memory-profiling overhead separately and do not include a profiler in only one policy's timing.

Let \(T_B(s),T_P(s)\) be per-instance median times. The primary effect is
\[
\mathcal R=1-\frac{\sum_sT_P(s)}{\sum_sT_B(s)}.
\]
The balanced design gives equal cell weights. Do not average individual speedup ratios.

Compute a paired stratified block-bootstrap 95% interval with 2,000 resamples: independently resample blocks within each geometry stratum, preserving all twelve cells and their paired policy measurements. Timing repeats are not additional independent scientific samples. Report total time, medians, p90/p95 tail latency, fallback rates, verified acceptance uplift, verification-time share, peak memory, solver calls, and per-stratum outcomes.

For a returned candidate with objective interval \([U_q^{-},U_q^{+}]\) and continuous reference interval \([L_{\rm ref},U_{\rm ref}]\), report relative-gap bounds
\[
\max(0,U_q^{-}/U_{\rm ref}-1)
\ \le\ \frac{F_p(q)-F_p^\star}{F_p^\star}
\ \le\ U_q^{+}/L_{\rm ref}-1
\]
when denominators are positive. A feasible-coupling upper bound alone does not give \(U_q^{-}\); use the two-target dual or a separately verified OT evaluation for that lower endpoint. For compressed/fallback outputs, a missing tight candidate evaluation means report the valid upper bound only.

An accepted candidate is empirically contradicted if a verified candidate lower bound exceeds \(1.05\) times a verified feasible reference upper bound. Conversely, agreement with an approximate solver is not the proof of certification: the analytic bound and independently replayed numerical witness provide that proof.

Report both thresholds 1% and 10% as secondary acceptance sensitivity using the frozen bound data. They are descriptive; the primary claim remains at 5%. Do not retune the policy to those thresholds.

### 5.7 Rejections, failures, and uncertainty

Rejected shortcuts remain in every policy-latency denominator and pay their overhead. Invalid inputs receive explicit validation status. Timeouts, allocation failures, unresolved reference intervals, certificate failures, and nonconvergence receive distinct statuses; preserve their elapsed time and solver output.

A policy that cannot return a validated result is not a fast success. Do not drop it or substitute zero time. Report capped/censored times and failure rates. A supported primary speed claim requires every included primary case to return a validated result in both policies; otherwise the full-workload conclusion is inconclusive or infeasible, with completed-case timing clearly secondary.

Reference intervals may remain wider than the diagnostic target without invalidating an independently verified shortcut. Report their coverage and refrain from precise actual-gap claims on those cases. A solver timeout is not a refutation of the theorem.

The generic policy and its late-tent extension should have identical successful outputs on paths accepted by the generic prefix. On other paths they may return different distributions, but both must satisfy the same objective requirement. Inspect branch traces to verify that the claimed saved work really is a raw fallback.

### 5.8 Break-even interpretation

For baseline-unresolved cases, the added tent stage costs time on every attempt and saves fallback only when it certifies. With observed tent cost \(c_s\), saved raw-fallback time \(t_s\), and incremental acceptance indicator \(I_s\), the extra work is
\[
\Delta T=\sum_s c_s-\sum_s I_s t_s .
\]
Compare this decomposition with actual paired timing; discrepancies can reveal caching or timing-boundary errors.

Do not infer a real eligibility frequency from the synthetic mixture. Report a sensitivity curve for hypothetical fractions of the tested case types, and the preprocessing-cost allowance implied by the measured saving. If fusion is only fraction \(f\) of system runtime, a fusion speedup \(S\) would imply at most
\[
S_{\rm system}=\frac1{(1-f)+f/S}
\]
before new preprocessing overhead. This is a conditional calculation, not a measured robot-level benefit.

## 6. Resource ceilings, calibration, and evidence gates

### 6.1 Hard cumulative limits

The following are **planning ceilings**, not measured M3 performance estimates:

| Resource | Ceiling and enforcement |
| --- | --- |
| Experimental execution envelope | 4 hours cumulative wall time across correctness runs, calibration, pilot, confirmation, verification, failed runs, and report-regeneration jobs |
| Mandatory remaining reserve | 30 minutes of that envelope for evidence preservation and report generation; stop starting new experiments when reserve is reached |
| End-to-end working session | 12 hours including setup, implementation, debugging, analysis, and experimental execution |
| Expected setup/implementation/analysis effort | Approximately 4–7 hours in addition to up to 4 hours of execution; total expected 8–11 hours, unmeasured estimate |
| Peak process-tree RSS | \(M_{\max}=\min(4\text{ GiB},0.20\,\text{physical RAM},0.50\,\text{available RAM at start})\); stop allocating at 80% forecast utilization |
| Project disk | 2 GiB, additionally at most 20% of initial free space; require at least 1 GiB free after projected outputs |
| Distinct numerical problem instances | 1,000, including randomized moment tests, development, failed/repaired cases, and stress cases |
| Total method evaluations | 30,000, including timing batches and repeated calls |
| Total LP starts | 12,000, including moment, pairwise, compressed-reference, raw, retry, and failed starts |
| Raw/fallback call limit | At most two attempts of 15 seconds each and 30 seconds total per fallback, including construction, solve, reconstruction and verification; remaining global/stage budget may be smaller |
| Raw pairwise LP attempt | At most 3 seconds, including verification |
| Repairs | At most two ordinary repair episodes of 30 minutes each, charged to the 12-hour session and applicable execution/stage budgets |
| Precision escalations | One per certificate computation, 128 to 256 bits; no indefinite precision loop |

The ledger must record counters and elapsed work before each new allocation. Changing stage, method, configuration, or fallback does not reset a cap. An actual memory/disk shortage overrides the nominal ceiling. No remote compute, paid compute, or additional machine is authorized.

Avoid an unnecessary dense \(n^3\)-by-\(3n\) matrix. At equal support size \(n\), the raw LP has \(n^3\) columns and \(3n^3\) nonzeros:

| Particles/input | Variables | Constraint nonzeros before removing redundant rows |
| ---: | ---: | ---: |
| 8 | 512 | 1,536 |
| 16 | 4,096 | 12,288 |
| 32 | 32,768 | 98,304 |

These are exact size counts, not runtime predictions. For \(n=16\), the objective vector alone is 32 KiB in float64; the sparse coefficients, tuple geometry, interval objects, solver copies/presolve/factorization, Python objects, and logs add substantially more. Measure peak RSS. Do not store a Python object graph for all possible input-output transports when sparse tuples suffice.

The tent features require \(O(\sum_i n_i)\) distance/accumulation work and constant-size witness optimization; the two-target candidate costs \(O(\sum_i n_i\log n_i)\). Compressed optimization has fixed 58-basis work after linear-time preprocessing. Raw cost construction and verification scale as \(O(n_1n_2n_3)\), and LP solve time has no assumed cubic law. Use measured scaling, including native memory, to forecast it.

### 6.2 Calibration and workload resizing rule

At the calibration stage measure construction, verified geometry, each cheap stage, both raw-medoid variants, raw solve/repair/verification, and total policy latency on representative \(n=8,16,32\) cases. Run each at least twice after a warm-up, unless a cap is already threatened.

Use the worst observed representative cost and a factor-two time contingency to forecast the remaining exact count of planned calls, including three timing repetitions, reference audits, possible raw-medoid calls, and a 10% retry allowance. Check instance/evaluation/LP counters explicitly as well as time and memory. For example, if every main case reaches an enabled medoid stage and one raw fallback per call, 384 cases require \(384[6(3+1)+1]=9{,}600\) LP starts for the two policies' three repetitions and one independent reference each, before development, stresses, retries or second fallback attempts. If the remaining count budget cannot cover this workload plus the declared retry allowance, reduce complete blocks or stop; do not assume all cases will certify. Forecast memory using observed RSS, sparse sizes, and solver scaling; do not extrapolate only the NumPy input buffer.

Proceed with 384 primary instances only if the forecast fits both the remaining global execution budget/reserve and stage 4's 120-minute allocation, as well as operation/memory/disk caps. Otherwise select 288 complete instances with six blocks per stratum. Reduce optional scaling work and stage microbenchmark batches before reducing primary blocks. Unused allocations do not enlarge a later stage's ceiling. If the 288-instance \(n=16\) study does not fit, report primary-study infeasibility and optionally complete the declared \(n=8\) diagnostic within the remaining budget. A diagnostic at a smaller scale is not a successful primary study.

Do not change scientific tolerances or use float32 to fit the machine. If 128-bit verification dominates, measure that finding; switching to unverified arithmetic is not an optimization. No job should start when its calibrated upper time or memory estimate threatens the reserve/cap. Enforce a watchdog around solver **and** postprocessing.

### 6.3 Stage gates

Stage allocations sum to the four-hour execution envelope. Implementation effort is separate but constrained by the 12-hour session. Repairs consume the remaining allocations; they add no time.

| Stage | Question and prerequisites | Minimum work and evidence | Execution allocation | Decision and next action |
| --- | --- | --- | ---: | --- |
| 0. Constructive feasibility | Are the mathematical mechanism, exact inputs, dependencies, and target hardware available? | Re-derive the scalar witness; inspect environment; verify Arb conversion/endpoints; enumerate basis structure; record sources and missing archives | 15 min | Continue if the witness and required backend are valid. One ordinary dependency repair is allowed. A proof counterexample stops the affected claim with a reproducible derivation. Missing Mac blocks Mac timing, not unaffected analytic work. |
| 1. Tiny correctness | Do independent formulations and verifiers agree? Requires stage 0 | Implement only core geometry, candidates, compressed/reference solvers, tent and certificate replay; execute §5.1; preserve raw discrepancies | 30 min | Continue only after all essential invariants pass and no unresolved false certification remains. Use a bounded repair for a located implementation defect. An unexplained disagreement makes the comparison invalid; stop before benchmarking. |
| 2. Resource calibration | Does a scientifically useful \(n=16\) workload fit? Requires correct tiny implementation | Timing/memory profile, backend/version record, operation counts, forecast, justified tolerances, selected block count | 15 min | Freeze feasible scale using §6.2. If main scale fails, take only the declared smaller diagnostic or stop for practical infeasibility. |
| 3. Mechanism pilot | Can the new bound change decisions, and can the measurement distinguish useful savings? Requires feasible costs | 32 development cases; paired policy timing; raw-medoid baseline choice; same-candidate bound table; verify rejection accounting | 30 min | Freeze baseline, precision, code, generator and counts. Zero extra acceptances in this small pilot is not a scientific refutation. Permit at most 8 extra development cases solely to diagnose a named numerical/measurement uncertainty, within the same budget; no generator retuning. |
| 4. Locked comparison | Does the predeclared engineering claim hold? Requires frozen protocol and enough remaining budget | Main paired runs, all saved witnesses, continuous reference intervals, failures, primary aggregate and block interval; up to 32 stresses if feasible | 120 min | Do not tune after seeing results. Stop new allocations before a cap; preserve partial evidence and classify incomplete primary comparison honestly. Optional scaling only with all primary work complete and reserve intact. |
| 5. Diagnosis and report | What conclusion is justified? Always execute after any prior stop | Replay certificates, regenerate tables/plots, reconcile ledger, report evidence/status/deviations/limitations, package runnable reached-stage code | 30 min | Deliver a supported contribution, informative negative, practical infeasibility, inconclusive result, or invalid experiment, using §7. |

If a pilot is noisy, first check timing boundaries, paired order, thermal/background contention, and certificate replay. One bounded diagnostic may resolve such uncertainty. Do not repeatedly collect pilot data until the result becomes favorable.

If a mathematical argument proves redundancy on the entire proposed regime, a verifiable derivation plus minimal confirming diagnostic can complete the study without stage 4. A few unfavorable pilot examples do not constitute such a proof. Conversely, a positive pilot is only justification to run the locked evaluation.

### 6.4 Bounded optional scaling

Only after the main study and its certificate audit are complete, and with at least 50 minutes remaining including the mandatory 30-minute reserve, run at most 24 preselected cases at \(n=8\) and \(n=32\). Select case IDs by stratum and cell indices before viewing their outcomes. Keep all other settings fixed.

These measurements assess the crossover between feature overhead and raw LP cost. They cannot rescue a failed primary hypothesis or replace its result. Do not claim larger-particle applicability from asymptotic symbols alone.

## 7. Outcome rules and required final artifacts

### 7.1 Classify the result explicitly

| Classification | Required evidence and permissible conclusion |
| --- | --- |
| **Supported contribution** | Valid proofs/implementation and all primary policy outputs verified; lower endpoint of the 95% block interval for \(\mathcal R\) is at least 10%. State the measured gain only for the frozen synthetic workload, hardware, backend and pipeline. Historical novelty and deployment relevance remain unresolved. |
| **Informative negative result** | Valid implementation and adequate locked evaluation, with the upper interval endpoint below 10%, or a proved obstruction/redundancy under stated assumptions. A positive but smaller gain fails the chosen investment hurdle; it is not universal inferiority. An upper endpoint below zero supports overhead harm in this workload. |
| **Practical infeasibility** | Measured costs/scaling or unavailable indispensable resources prevent the specified study or application within caps. Show the forecast, failing stage, and what smaller diagnostic still means. |
| **Inconclusive evidence** | Interval straddles the 10% hurdle, primary cases remain unresolved, evidence coverage is inadequate, or access/budget prevents the specified decision. Report what is and is not identified. |
| **Invalid experiment** | Unresolved implementation, baseline, input, certificate, or measurement defects invalidate the attempted empirical comparison. Preserve the defect and reproduction; do not call it evidence against the mathematical hypothesis. |

Numerical tolerance violations must first be diagnosed. An accepted shortcut with an invalid witness suspends all affected performance claims until repaired and rerun on fresh confirmation instances, if the remaining budget permits. Otherwise classify those comparisons as invalid. A timeout, a bug, or failure to observe a gain is not automatically a mathematical counterexample.

If confirmation changes are necessary, retain the original protocol/results, mark the revised work exploratory, and use fresh phase/substream IDs for any new confirmation. No favorable subset or post-hoc noise/geometry mixture may replace the declared primary result.

### 7.2 Required report content

The final **RESEARCH_REPORT.md** must stand alone and lead with the decision:

1. Research question, reached gates, chosen outcome classification, and the exact scope of the conclusion.
2. The mathematical certificate and machine-validation contract, including what is derived, sourced, and actually verified.
3. Environment, versions, device/backend/precision, frozen configuration, actual elapsed work, memory/disk peaks, and complete ledger totals.
4. Main baseline/proposed comparison with uncertainty, per-stratum results, failures and rejected-case overhead; exact versus approximate objective distinctions.
5. Attribution: acceptance changes from the tent lower bound, the effect of centroid compression/raw medoid, and verification overhead.
6. Limitations: conditional synthetic data, supplied hypothesis information, finite empirical inputs, lack of downstream accuracy/calibration evidence, fallback implementation dependence, and unresolved priority.
7. One specific worthwhile next step if justified; otherwise say why further work is not warranted under these assumptions.

Required tables/figures for stages reached:

* claim-to-evidence and stage-status tables;
* primary latency/certificate/failure table with per-stratum breakdown;
* acceptance versus noise and imbalance, with exact controls separate;
* verified certificate bound versus continuous reference gap interval where available;
* total time decomposition, including interval work and rejected attempts;
* resource ledger and a conditional break-even/preprocessing-cost plot.

Do not make plots that imply exact actual error when only a one-sided or wide interval is available. Standard Matplotlib files are sufficient; include the underlying CSV/JSON.

### 7.3 Deliverable package

For the stages actually reached, supply runnable code, concise README setup/reproduction commands, a pinned environment, frozen configuration, generated inputs or exact regeneration plus hashes, immutable raw run logs, all failure/retry records, replayable accepted certificates, essential tables/plots, and the report. Do not require unavailable attachments or old archives to reproduce the implementation.

For an analytic stop, provide the checked derivation/counterexample and only the minimal diagnostic code needed to reproduce it. Do not invent benchmarks to fill a checklist. For a hardware/access block, complete unaffected mathematics and local diagnostics and say exactly which Mac measurements were not performed.

The report command must regenerate essential tables from saved raw results without rerunning solvers. The verifier must reject a tampered witness. Check all README commands on a clean virtual environment if time permits within the cap; otherwise state exactly which reproduction step remains unverified.

## 8. Research basis, planning checks, and review disposition

### 8.1 Compact claim-to-evidence map

| Central claim | Source or derivation | Evidence status at handoff | Resolving execution check |
| --- | --- | --- | --- |
| An exact complementary input is a global median | [R1], Theorem 2.1, pp. 3–4 | Proof inspected; no essential defect found; not formally verified | Analytic controls and tiny continuous reference |
| The exact algorithm equals the input medoid | Pairwise formula in [R1], Eq. (5) | Direct identity | Equality/control tests |
| \(2E\) stability and compressed interval nesting | Triangle-inequality derivations in this brief | Mathematically derived | Exact/perturbed comparisons and explicit coupling checks |
| Tent moment relaxation and scalar witness are valid | §2.4–2.5, derived from [R1]'s map | Independent mathematical audit and float probes; no deployed implementation | Independent epigraph LP, exact coefficient feasibility, Arb replay |
| Generic centroid certification can already pass the headline example | Established centroid inequality [R3, R4], calculation here | Numerical calculation performed | Reproduce 3.45185% control |
| Eight-variable generic compression can avoid LP startup | [R2] formulation plus exact-basis adaptation | 58-basis count independently checked in planning | Rational enumeration versus HiGHS |
| Additional tent certification yields useful net speed | Study hypothesis | Unmeasured; could be false | Locked paired comparison including verifier cost |
| Ambiguous poses motivate distribution-valued methods | [R5] | Literature motivation only | No claim about deployment prevalence; future real workload needed |
| M3 Max implementation is feasible within caps | Size counts and simple CPU design | Planning estimate; actual Mac uninspected | Environment and resource calibration gates |
| Historical originality | Targeted searches and supplied note | Unresolved | Conservative claims; do not present search failure as proof of priority |

### 8.2 Small planning checks actually performed

These checks took place on **hosted Linux x86_64**, Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0. They are not Mac timings, validated-arithmetic tests, confirmation data, or a completed PoC.

* The centroid bound for the exact 3–4–5 case was calculated as above.
* The rank-four eight-column marginal matrix was enumerated independently: 58 nonsingular four-column bases, with determinant magnitudes one or two. This establishes an implementable strong baseline.
* With seed 70291, 250 randomized feasible moment arrays were checked against a separately assembled 12-variable epigraph HiGHS LP. The final probe's maximum absolute greedy-versus-LP discrepancy was \(8.88\times10^{-16}\); the maximum greedy-versus-reconstructed-dual discrepancy was also \(8.88\times10^{-16}\).
* Twelve small design cases crossed three geometries with exact inputs, balanced symmetric cloud noise, imbalance, and both. A normalized 3–4–5 balanced-noise case gave relative bounds of about 6.108% for original \(2E\), 5.206% for anchor compression with the same \(q_0\) upper, 5.613% for centroids, and 4.539% for the tent witness.
* The same probes showed no tent lower-bound improvement over compression on balanced equilateral noise; a centroid advantage on a thin triangle; and a roughly 7–8% error for the original \(q_0\) on imbalance-only cases where compressed optimization could return a better answer.

The illustrative noisy case establishes plausibility, not prevalence or a fair full-pipeline win. It did not compare the complete strengthened pipeline now specified, and its antithetic particle construction is not used in confirmation. No timing or global superiority claim is based on it.

Planning scratch scripts and numerical outputs are not required execution inputs. The equations, tiny-test generators, and expected controls in this brief specify how to rebuild the necessary checks independently. Arb and the full primal/dual verification pipeline remain implementation tasks.

### 8.3 Search coverage and limits

Research was conducted on 23 October 2026 using both available search engines, primary papers, official code, and official numerical-library documentation. Queries included complementary/leave-one-out Wasserstein medians, Fréchet median certificates, centroid EMD lower bounds, tent/Lipschitz-feature bounds, moment relaxations, compression, and continuous multimarginal solvers.

Relevant portions of the principal \(W_1\) paper, its official code description, Rubner's lower-bound proof, Mémoli's explicit centroid statement, the rotation-synchronization paper, the cited 2026 \(W_2\) paper, and software documentation were inspected. Backward references exposed the established centroid filtering principle; recent related work confirmed that “Wasserstein median” also names materially different \(W_2\) problems.

The 2026 free-support paper [R6] is real and its objective uses \(W_2\), with fixed support weights. Its published speed numbers are not an admissible baseline for this \(W_1\) study. The official implementation accompanying [R2] targets one-dimensional sorting and two-dimensional grid/flow methods; it is useful context, but a direct particle/free-support implementation is required here [R9].

Broader observable-Wasserstein and moment-relaxation work shows that Lipschitz projections and moment lower bounds are established frameworks [R10]. No matching tent/complementary construction was located in the sources checked. The search was targeted, not exhaustive; inaccessible material and terminology differences remain priority risks. A historical centroid technical report was located through citations but its direct PDF was inaccessible; its proof is not relied upon because the accessible later paper provides it.

### 8.4 Three review passes and repaired risks

The brief was developed with separate mathematical/novelty, experimental, and execution reviews. The material findings and their dispositions are:

| Stress scenario / review finding | Disposition |
| --- | --- |
| The theorem might be nonconstructive | The exact three-score construction is explicit; the extension uses constant-size greedy search and a separately checked scalar witness. |
| A realistic input violates exact complementarity | The direct tent lower bound is valid for arbitrary actual particle measures; compression charges all mismatch. Applicability is an efficiency question, not an ignored assumption. |
| The mechanism is equivalent to known work | Attribute medoids, centroids, stability, compression, and dual/moment principles to prior work. Limit the candidate contribution to this map's incremental engineering value. |
| Generic cheap methods already solve the headline example | Include the centroid 3–4–5 calculation and the 58-basis mass-preserving solver, with per-cloud centroid compression. |
| A better candidate removes the apparent gain | Both policies share the candidate pool; compressed and raw-medoid baselines can return their own distributions. Tent is inserted only after the generic prefix. |
| Only favorable synthetic cases work | Freeze multiple shapes, four noise scales, three imbalances, non-antithetic clouds, and separate unrelated/outlier stresses. Claims remain conditional on the declared population. |
| Preprocessing dominates | Time all supplied-metadata processing; do not assume anchor discovery is free in deployment. Report preprocessing break-even allowance. |
| Finite precision reverses a bound | Exact masses/coefficient constraints, Arb enclosures, explicit feasible couplings, full dual checks, and a conservative rejection path replace arbitrary epsilon padding. |
| Numerical repair itself creates an infeasible witness | Reject nonfinite solver entries, clamp floored mass counts at zero before exact capacity repair, and use an exact rational 120-degree branch test. The affected mathematical specification was independently rechecked. |
| A finite-grid reference appears exact | Use the continuous multimarginal oracle and an equilateral Dirac countercontrol. A restricted support optimum is only an upper comparison. |
| Verification or memory is too expensive | Calibrate total verified cost and native RSS; keep caps and allow an honest infeasibility result. Float-only timing cannot rescue the certified claim. |
| The empty-project workflow has missing accounting or stop paths | Initialize the session before dependency setup; share the cumulative ledger with tests; support reporting before freeze; exclude outputs from the scientific hash; enforce both stage and global budgets. The repaired execution workflow was independently rechecked. |
| A small pilot is inconclusive | A zero/uncertain pilot effect is not a refutation. Permit one bounded diagnostic for a named uncertainty, then freeze or report inability to decide. |
| An essential archive or dataset is inaccessible | Historical archives are unverified and unnecessary for the independent construction; no real workload is claimed. Core sources/definitions are included here. |
| The main hypothesis is false | Preserve negative results and classify redundancy/overhead as a useful scoped engineering finding; do not switch projects. |

Residual risks remain: undocumented library behavior on the actual Mac, implementation defects in validated geometry, uncertain costs of verifying all raw tuple constraints, a synthetic workload unrepresentative of a real system, and unresolved historical novelty. Each is an explicit gate or limitation. No claim of absolute completeness or correctness is made.

## References

**[R1] Supplied working note.** *Complementary Measures and Median Geometry of Finite Wasserstein Simplices.* Attributed in the PDF to “Autonomous mathematical investigation,” 23 October 2026. Attachment **wasserstein_geometry_research(1).pdf**, 17 pp.; location and SHA-256 in §2.1. Critical locations: Theorem 2.1, pp. 3–4, Eqs. (1)–(5); Corollary 2.2, pp. 4–5; experimental claims and limitations, §6, pp. 11–12; priority discussion, §7, pp. 12–14. Unpublished working note; no public URL or DOI supplied.

**[R2] Guillaume Carlier, Enis Chenchene, Katharina Eichinger.** *Wasserstein Medians: Robustness, PDE Characterization, and Numerics.* SIAM Journal on Mathematical Analysis 56(5), 6483–6520 (2024). DOI: [10.1137/23M1624786](https://doi.org/10.1137/23M1624786). Inspected preprint: [arXiv:2307.01765v2](https://arxiv.org/html/2307.01765v2), 3 February 2025 revision of the 2023 preprint. Critical locations: Example 2.6 (majority threshold); §5.1, Eq. (39)–(41), Theorem 5.1 (continuous multimarginal representation); §5.2, Eq. (45), Proposition 5.4 (dual); §7.1 (numerical formulations).

**[R3] Yossi Rubner, Carlo Tomasi, Leonidas J. Guibas.** *The Earth Mover's Distance as a Metric for Image Retrieval.* International Journal of Computer Vision 40(2), 99–121 (November 2000). DOI: [10.1023/A:1026543900054](https://doi.org/10.1023/A:1026543900054). [Accessible author manuscript](https://www.cs.cmu.edu/~efros/courses/LBMV07/Papers/rubner-jcviu-00.pdf). Critical locations in that 20-page manuscript: §4, pp. 8–9 (cheap bounds/filtering), Appendix B, p. 20 (norm-of-centroids lower-bound proof). Manuscript and journal pagination differ.

**[R4] Facundo Mémoli, Zane Smith, Zhengchao Wan.** *The Wasserstein Transform.* Proceedings of the 36th International Conference on Machine Learning, PMLR 97 (2019). [Paper and metadata](https://proceedings.mlr.press/v97/memoli19a.html); [PDF](https://proceedings.mlr.press/v97/memoli19a/memoli19a.pdf). Critical location: Remark 2.2, PDF p. 2, recalling the centroid lower bound and its prior attribution.

**[R5] Tolga Birdal, Michael Arbel, Umut Şimşekli, Leonidas Guibas.** *Synchronizing Probability Measures on Rotations via Optimal Transport.* CVPR 2020; arXiv submitted 1 April 2020. [arXiv:2004.00663](https://arxiv.org/abs/2004.00663); [full text](https://arxiv.org/html/2004.00663v1). Relevant locations: §3 definitions of transport/Sinkhorn divergences and §4 measure-valued pose hypotheses. Application motivation only; not evidence of a deployed three-input \(W_1\)-median workload.

**[R6] Kisung You, Mauro Giuffrè, Dennis Shung.** *Fast Computation of Free-Support Wasserstein Medians.* arXiv:2606.19148v1, 17 June 2026. [Full text](https://arxiv.org/html/2606.19148). Critical locations: introduction Eq. (1), §2.4 Eq. (4) (\(W_2\) objective), and §3 (fixed support weights). Excluded as an incompatible timing baseline.

**[R7] SciPy developers.** *linprog(method='highs') documentation.* Documentation retrieved 23 October 2026; the inspected page identified SciPy 1.18.0. [Official documentation](https://docs.scipy.org/doc/scipy/reference/optimize.linprog-highs.html). Relevant entries: method selection, status, time_limit, primal/dual tolerances, residuals, and marginals. Consult the installed-version documentation during execution.

**[R8] Fredrik Johansson and python-flint/FLINT contributors.** *Python-FLINT documentation and implementation.* Retrieved 23 October 2026. [Installation and supported platforms](https://python-flint.readthedocs.io/en/latest/install.html), [Arb real-number API](https://python-flint.readthedocs.io/en/latest/arb.html), [official repository](https://github.com/flintlib/python-flint). Critical entries: exact rational conversion, rigorous real error tracking, lower/upper endpoints, enclosing minimum/maximum, precision context, and macOS ARM binary support. Actual target installation remains to be checked.

**[R9] TraDE-OPT / Carlier–Chenchene–Eichinger.** *wasserstein-medians: example code.* [Official repository](https://github.com/TraDE-OPT/wasserstein-medians), README inspected 23 October 2026. One-dimensional sorting and two-dimensional Douglas–Rachford examples; not treated as an optimized free-support particle baseline. Record a commit if code is reused.

**[R10] Edivaldo Lopes dos Santos, Leandro Vicente Mauri, Washington Mio, Tom Needham.** *The Observable Wasserstein Distance.* arXiv:2605.09916v1, May 2026. [Full text](https://arxiv.org/html/2605.09916v1). Relevant locations: introduction and §§2, 4–5, Lipschitz observables and lower-bound constructions. Framework-level context only; this brief does not claim that paper proves the particular tent median relaxation.

**[R11] Supplied application/task text.** **Pasted text(20261023-165551).txt**, attached 23 October 2026; location and SHA-256 in §2.1. Contains the original hypothesis, \(2E\) proposal, historical synthetic results, and the requested execution-brief requirements. Its historical runtime and accuracy numbers are unverified here.

---

**Execution instruction:** proceed autonomously through authorized gates, make ordinary bounded repairs, preserve adverse evidence, and stop when the gate rules require it. Your deliverable is the reached-stage reproducible study and its honest conclusion, not a guaranteed positive result.
