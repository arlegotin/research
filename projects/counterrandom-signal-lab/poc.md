# POC execution brief: certified clock-grid limits of bounded moment completion

**Recipient:** autonomous coding agent configured as `gpt-6.1-sol` with `xhigh` reasoning.  
**Research and protocol date:** 4 November 2026.  
**Deliverable you must produce:** a small, runnable local research implementation, independently checked certificates, the measurements justified by the stages reached, and a candid research report. This document is the execution instruction; the end-to-end PoC has not already been completed.  
**Target:** one MacBook with an M3 Max; CPU computation only. Inspect the actual machine before making hardware claims.

## 1. Project decision and smallest defensible contribution

Implement a **certificate-driven diagnostic of the extra duration imposed by a uniform sample clock on bounded, prescribed-prefix moment completion**. Keep the original samples unchanged. Treat each output sample as a zero-order hold over one complete clock period. Establish feasibility and impossibility using exact rational certificates, with ordinary floating-point linear programming used to discover candidates.

The selected question is:

> For constant bounded prefixes and cancellation orders three and four, does rounding the sharp continuous-time horizon upward to a whole number of samples suffice for exact moment cancellation, and, where it fails, what is the certified additional sample count on the declared finite set of clock resolutions?

The universal rounding claim is **already false**, as demonstrated by the exact planning-stage counterexample in §3.4. Your task is to reproduce that result independently and, if the evidence gates permit, establish its finite-grid extent. A secondary, explicitly finite-set hypothesis is that **one additional sample beyond upward rounding suffices on the locked evaluation set**. This hypothesis remains untested on that set. Do not promote it to a universal theorem.

### 1.1 Why this project

The supplied continuous theorem provides a constructive, amplitude-preserving completion and a sharp worst-case duration. It does not provide a uniform-clock theorem, a fixed-wordlength theorem, an energy optimum, or a useful-signal estimator. The existing executable reports sampling residuals that do not precisely describe its exported WAV files. Resolving this translation is a necessary, inexpensive prerequisite to credible hardware claims.

The closest mechanisms are established: Uhrig sine-squared timing, Chebyshev/Fourier moment annihilation, polynomial spectral nulls, bounded discrete-time reachability, and exact verification of floating-point LP candidates [R1], [R2], [R3], [R4], [R5], [R6], [R7], [R8], [R9]. In particular, the identical duration formula and quadratic asymptotic already appear in Uhrig and Lidar, §VI, Eqs. (36)–(37) [R4]. The exact prescribed-prefix minimax/equality theorem in the attachment was not located in the inspected literature, but its historical originality remains unresolved.

**Smallest defensible contribution:** a reproducible, exact counterexample to a tempting digital extension of the supplied theorem, corrected measurement of the supplied files, and—if completed—a certified finite table of the clock-grid penalty for this precise constraint set. Claim a limitation or scoped characterization, not invention of moment cancellation, polynomial duality, LP, or a new quantum-control method. The particular certificate is independently derived here; historical priority for that example is not established.

### 1.2 The decision the study informs

Determine whether future digital implementations may use the continuous duration rounded upward, require a measured clock guard, or require an input-specific feasibility solve. A one-tick guard is a meaningful **representation-level** threshold: it is the smallest additional output-duration increment the device clock permits. Its usefulness for an actual sensor or actuator still depends on a deadline supplied by that application. No such deadline is available, so do not manufacture a clinical, acoustic, industrial, or commercial benefit threshold.

The study can support a useful negative result even if every attempted improvement over conventional optimization disappears. A correct analytic stop is a completed outcome; a benchmark is not required to rescue a false premise.

### 1.3 Fixed scope and exclusions

The primary model permits **arbitrary real amplitudes** within the same peak bound as the input. Exact rational witnesses certify that model. They are not necessarily representable by float64 or a fixed-bit DAC. An impossibility certificate applies to those narrower amplitude alphabets too; a rational feasibility witness does not automatically transfer to them.

Do not build live microphone capture, actuator playback, a GUI, a dashboard, a service, GPU acceleration, a compressor, a graph experiment, or a quantum simulation. Do not claim instantaneous noise cancellation, improved sensor precision, optimum energy, bandwidth compliance, or superiority over feedback control. The entropy and graph notes are reviewed context, not additional project branches. The infinite Abel-regularized extension in the moment note is excluded: its moments are not ordinary integrals.

This revision preserves the practical purpose of discovering what a bounded continuation can deliver after digitization. It postpones the original same-time/energy/bandwidth hardware comparison because neither a plant nor a useful-signal/noise measurement objective is specified. Appending a continuation changes selected observables of the whole record; it cannot alter observations already made on the prefix.

### 1.4 Required execution order

1. Verify the exact obstruction and feasible witness before installing or building substantial benchmark infrastructure.
2. Implement one small LP-guided, exact-certificate oracle and the independent checker.
3. Calibrate on development cases; freeze the protocol before examining the locked set.
4. Run the bounded constant-prefix frontier, then the small secondary diagnostics only if their gates pass.
5. Preserve all failures and produce the report for the stages actually reached.

Proceed autonomously through these authorized stages. Repair ordinary implementation problems within the stated limits. If target hardware or inputs are absent, complete unaffected work and identify the precise unperformed validation. Never substitute execution on another machine for M3 Max measurements.

## 2. Inputs, evidence status, and source fidelity

### 2.1 Files to receive

Receive this brief and, preferably, both original ZIP files below. Extract into fresh directories without modifying the originals. Reject archive members that escape the extraction directory. Record SHA-256 hashes and retain the original names in the provenance manifest.

| Input | Relevant members and role | SHA-256 of supplied archive |
|---|---|---|
| `counterrandomness_research(1).zip` | `moment_completion_theorems.txt` is the essential mathematical source; `moment_completion_experiment.py` and its JSON/PNG record the source's finite checks. Also contains `RESEARCH_README.txt`, the entropy note/code/results/figure, and the inverse-heat note/code/results. | `a854eeb5cc31c862a8c6f376b0b1a449f7d3a71f3e4f150cb582e2523ae53ff8` |
| `Counterrandom_Signal_Lab_Mac_M3_Max(1).zip` | Root directory `counterrandom_mac_poc/`; essential files are `README.md`, `counterrandom_poc.py`, `optimality_certificate.py`, `verify_theorems.py`, `requirements.txt`, and `example/` plus `high_rate/` WAVs, reports, and plots. `entropy_demo.py` and `entropy_gap.png` are contextual. | `9109d45ecb45a6b65fbc4e86a6af0eaecb7f0da32f894da175c92d7eb8d2e70e` |

The originating task was `Pasted text(20261104-134413).txt`, SHA-256 `a5d65704ab47f51b09390a8b90bb931426031c2bc969aabb41cdbaab9e9d62f3`; you do not need it because the actionable specification is here. Source notes are user-supplied research documents dated 4 November 2026 with no named author or verified peer-review status.

During planning, archives were available at `upload/` and extracted under `/workspace/scratch/8359c65e753c/inputs/research/` and `/workspace/scratch/8359c65e753c/inputs/signal_lab/counterrandom_mac_poc/`. These are provenance locations, not paths to assume on the Mac. Use your new project's `inputs/research/` and `inputs/signal_lab/counterrandom_mac_poc/`.

If the archives are unavailable, the exact fixture and all primary definitions below are sufficient for the core diagnostic. Mark the legacy-file audit unavailable; do not fabricate or reconstruct the supplied WAVs and call them the originals. If a source archive hash differs, record the mismatch, inspect the relevant contents, and distinguish a changed source from a numerical discrepancy. External papers are research references, not runtime services or mandatory data downloads.

### 2.2 Compact claim-to-evidence map

| Central claim | Source or derivation | Present status | Resolving action or remaining limit |
|---|---|---|---|
| Every measurable bounded prefix has a norm-preserving continuous completion at `H_r` | Moment note §1, Theorem A(a), construction §2, Newton-sum proof §3 | Source proof inspected; essential proof independently checked during planning | Reimplement only the finite-step construction and verify tiny cases; do not replace the proof with numerical residuals |
| `H_r` is the minimum universal continuous horizon; constant prefixes attain it uniquely | Moment note §4, Theorem A(b,c); dual polynomial `q` | Essential proof checked | Reproduce the constant lower-bound argument used for discrete minimality |
| Only constant saturated prefixes require the worst continuous horizon | Moment note §§5–6, Theorem A(d) | Source's compactness/separation proof is consistent | Do not treat `H_r` as the continuous optimum for nonconstant inputs; the proof supplies no numerical amount of slack |
| The supplied ideal residual and energy numbers describe one implemented example | `high_rate/report.json`; `counterrandom_poc.py` | Source-reported observations; not universal guarantees | Separate ideal segments, floating held samples, and decoded WAV measurements |
| Reported sampled residual is the exported WAV residual | Code `sample_segments`, `main`, `export_wav` | Incorrect identification: endpoint is adjusted and residual is computed before PCM encoding | Re-read the actual WAV; use every complete cell and actual duration |
| Upward rounding of `m H_r` always gives a feasible bounded clock-grid completion | New proposed inference, absent from the source theorem | Refuted by exact `r=4,m=2,n=21` certificate below | Independently verify both certificates; scope the conclusion to the digital extension |
| One further tick suffices on the locked set | Secondary hypothesis introduced here | Not established; exploratory evidence only | Exact primary frontier; any certified `delta >= 2` refutes it on that set |
| Moment-null construction or dual-polynomial obstruction is historically new | [R1], [R2], [R3], [R4], [R5], [R6], [R7], [R8], [R9], especially [R4], [R5] | Unsupported; close mechanisms are established | Use the narrow diagnostic contribution; do not assert global novelty |
| A small digital penalty implies a useful physical noise-control device | No source or model establishes this | Unsupported | Explicitly excluded; would require a separate application and plant model |

### 2.3 Audit corrections to carry forward

The source's center-set counts describe different enumeration scopes, not a simple arithmetic correction. `anti_random_combinatorics_experiment.py` includes singleton sets (`k=1,...,d+1`): `129 + 696 + 679120 = 679945`. Signal Lab's `verify_coverage_gap` starts at `k=2`, omitting `9 + 16 + 64 = 89` singletons and producing `679856`. Likewise, 640 versus 240 moment cases and 164,031 exhaustive rooted graphs versus 570 random rooted cases refer to different suites. Do not combine them or state that all were rerun in this project.

The entropy note's results assume a uniform product source, `q>d` for the complete mex construction, and appropriate causal information restrictions; there is no lossless-reconstruction obligation. The graph result is a finite-graph, short-time expansion with positive edge weights, and its own §6 supplies a counterexample to a global distance-parity sign rule. Neither provides a baseline or application guarantee for this project.

## 3. Mathematical specification and verified obstruction

### 3.1 Continuous source theorem

Let `r >= 1` be the number of moments, `D > 0` the prefix duration, and `A > 0` its essential peak. After normalization, let `f:[0,1] -> [-1,1]` be measurable. A completion `z:[0,H] -> [-1,1]` must equal `f` almost everywhere on `[0,1]` and satisfy

\[
\int_0^H t^k z(t)\,dt=0,\qquad k=0,\ldots,r-1.
\tag{1}
\]

The source theorem uses the complete prefix and imposes no continuity, slew, spectral, or wordlength restriction. Put

\[
N=r+1,\qquad H_r=\csc^2\!\left(\frac{\pi}{2N}\right),\qquad
a_j=H_r\sin^2\!\left(\frac{j\pi}{2N}\right),\quad j=0,\ldots,N.
\tag{2}
\]

Thus `a_0=0`, `a_1=1`, `a_N=H_r`. For `u in [0,1]`, define

\[
\theta(u)=2\arcsin\sqrt{u/H_r},\quad
\theta_j(u)=
\begin{cases}
j\pi/N+\theta(u),&j\text{ even},\\
(j+1)\pi/N-\theta(u),&j\text{ odd},
\end{cases}
\quad \tau_j(u)=H_r\sin^2(\theta_j(u)/2).
\tag{3}
\]

The extension on branch `j` is `(-1)^j f(u)` at time `tau_j(u)`. Even branches preserve order; odd branches reverse it. The first branch is exactly the original prefix. Multiply time by `D` and amplitude by `A` to return to physical units. Implement step inputs by transporting their boundaries, not by evaluating a high-degree Chebyshev polynomial and numerically inverting it.

Why cancellation holds: the `N` branch values `tau_j(u)` are roots of `T_N(1-2t/H_r)-T_N(1-2u/H_r)`. Newton's identities make their power sums through degree `N-1` independent of `u`; differentiation gives `sum_j tau_j(u)^k tau_j'(u)=0` for `k<r`. Branch orientation cancels the alternating sign during change of variables. Endpoint derivative singularities are integrable. This is an explicit finite-step algorithm, not an oracle.

For the saturated constant prefix, the unique optimal continuation is alternating `+1,-1,...` between the `a_j`. The source's sharpness polynomial is

\[
q(t)=(-1)^{r-1}\prod_{j=2}^{r}(t-a_j).
\tag{4}
\]

It is positive on the prefix and has the opposite sign to the tail. Therefore `integral_0^1 q = integral_1^{H_r} |q|`. At any shorter horizon the tail cannot provide enough opposing `q`-moment, regardless of how it is chosen. This proves the continuous lower bound needed below. The source's stronger nonconstant-prefix result is not needed to certify primary cases.

For the orders used here:

| `r` | `H_r` | Continuous constant-prefix total duration |
|---|---|---|
| 1 | `2` | `2 D` |
| 2 | `4` | `4 D` |
| 3 | `4 + 2 sqrt(2)` | approximately `6.82842712474619 D` |
| 4 | `6 + 2 sqrt(5)` | approximately `10.47213595499958 D` |

### 3.2 Uniform-clock model

Let the sample period be `Delta=1/f_s`; let `m` input samples be rational numbers `x_0,...,x_(m-1)`; and let `D=m Delta`. The output has **integer** length `n >= m`, with

\[
z(t)=z_j\quad\text{for }j\Delta\le t<(j+1)\Delta,\qquad
z_j=x_j\ (j<m),\quad |z_j|\le A=\max_i|x_i|.
\tag{5}
\]

Every cell, including the last, has width `Delta`; the output ends at `n Delta`. There are no fractional last cells, no stretched final hold, no hidden resampling of the prefix, and no extra settling interval included or excluded opportunistically.

If `A=0`, return the all-zero prefix at `n=m` and mark energy ratios undefined. Otherwise normalize exactly by `A` for the solver and preserve that scale in the records. All methods use this same bound. An LP with a unit bound must receive `x/A`; it must not gain extra headroom over a smaller-amplitude fold.

For any finite output,

\[
M_k(z)=\int_0^{n\Delta}t^kz(t)\,dt
=\frac{\Delta^{k+1}}{k+1}
\sum_{j=0}^{n-1}z_j\big((j+1)^{k+1}-j^{k+1}\big).
\tag{6}
\]

The polynomial `((j+1)^(k+1)-j^(k+1))/(k+1)` has degree `k` and leading coefficient one. The resulting triangular change of basis is invertible. Consequently the first `r` ordinary held-signal moments vanish **if and only if**

\[
\sum_{j=0}^{n-1} j^k z_j=0,\quad k=0,\ldots,r-1.
\tag{7}
\]

Use `j^0=1`, including `j=0`. Equation (7) is the exact certificate representation. Use Python integers and `fractions.Fraction`, not fixed-width integer powers that might overflow. Equivalent polynomial language is `(w-1)^r` dividing `P(w)=sum_j z_j w^j`; this is established spectral-null/bounded-coefficient mathematics [R5, R9]. The equivalence does not make the physical quantum and radar applications interchangeable with this problem.

The main output is

\[
n_{\min}(x,r)=\min\{n\ge m:\text{(5) and (7) are feasible}\}.
\tag{8}
\]

For `x_j=1`, the continuous theorem implies `n_min >= n_0=ceil(m H_r)`. Compute this ceiling **exactly**, without floating-point `ceil`:

```python
from math import isqrt

def continuous_ceiling(r, m):
    if r == 1:
        return 2*m
    if r == 2:
        return 4*m
    if r == 3:
        return 4*m + isqrt(8*m*m) + 1
    if r == 4:
        return 6*m + isqrt(20*m*m) + 1
    raise ValueError("Study is restricted to orders 1 through 4")
```

For positive integer `m`, the square roots in the last two lines are irrational. Hence the final `+1` is exact. The lower bound is specific to the saturated constant primary inputs; do not use it as an input-specific lower bound for other prefixes.

### 3.3 Exact primal and dual certificates

A **primal certificate** is a rational output vector satisfying the prefix, every box constraint, and all `r` equations (7) exactly. This proves feasibility in the real-amplitude model.

A **dual certificate** is a rational polynomial `Q(j)` of degree at most `r-1` such that

\[
G(Q)=\left|\sum_{j=0}^{m-1}x_j Q(j)\right|
- A\sum_{j=m}^{n-1}|Q(j)|>0.
\tag{9}
\]

Every admissible tail has opposing `Q`-sum of magnitude at most the second term, so (9) proves impossibility. Verify the strict inequality using exact arithmetic. The certificate checker must not call the LP solver or reuse its floating-point constraint matrix. This is ordinary finite-dimensional support-function duality, closely related to the bounded-coefficient polynomial obstruction in [R5]; its mechanism is not new.

Feasibility is monotone in `n`: append a zero sample to a feasible vector. Thus a primal at `n` plus a dual at `n-1` proves minimality. For a constant prefix, a primal at `n_0` plus the continuous lower bound is sufficient. For `n=m`, an exactly zero prefix moment vector proves minimality by the domain bound; otherwise any nonzero prefix moment supplies a no-tail obstruction.

### 3.4 Mandatory exact fixture: upward rounding fails

Use `r=4`, `m=2`, `x=(1,1)`, `A=1`, and measure time in sample periods. The continuous lower bound is

\[
2H_4=12+4\sqrt5\approx20.94427191,
\quad n_0=21.
\]

Take

\[
Q(j)=-(j-7)(j-13)(j-19).
\]

Direct integer sums give

\[
Q(0)+Q(1)=3025,\qquad
\sum_{j=2}^{20}|Q(j)|=3016,
\qquad G(Q)=9>0.
\tag{10}
\]

Therefore **no** amplitude-one tail with 21 total uniform samples can cancel all four moments. This excludes arbitrary real-valued tails, not merely binary tails or the supplied construction.

At 22 samples the following rational vector, indexed `j=0,...,21`, is feasible:

```text
[1, 1, -1, -1, -1, -1, -1, 72/143,
 1, 1, 1, 1, 1, 13/35, -1, -1, -1, -1,
 -4/55, 1, -73/91, 1]
```

Every amplitude lies in `[-1,1]` and all four integer-power sums are exactly zero. Check (6) independently as well. By zero-padding monotonicity, (10) and this witness establish `n_min=22`. With a one-second prefix, `Delta=1/2 s`: the grid optimum is 11 seconds, while the continuous optimum is about 10.472 seconds. This coarse example establishes an obstruction, not its importance at 48 kHz.

These sums, the full rational witness, and the ordinary held-signal moments were independently checked during planning with integer/Fraction arithmetic. Your checker must reproduce them from the numbers above. Do not use decimals for the fractions.

An additional development stress case is `r=4,m=36,n_0=377`, with `Q(j)=-(j-130)(j-246)(j-341)`. Its prefix sum is `301956570`, tail absolute sum is `301953510`, and gap is `3060`. It illustrates why testing only powers-of-two prefix lengths can miss clock alignment effects. Produce your own primal at 378 if you use this case; the certificate alone proves only infeasibility at 377.

### 3.5 Mechanism, limiting cases, and what does not follow

At the exact continuous optimum the constant completion is unique. If one of its internal sign switches falls inside a uniform hold cell, a grid waveform cannot equal that completion almost everywhere. Merely rounding the end time upward may leave too little room to compensate for this restriction. More generally, for a polynomial `p`, clock holds replace the continuous support `integral |p|` with `sum_cells |integral_cell p|`, which is smaller on cells containing sign changes. This explains sensitivity to clock phase. It is not a proof of a one-tick bound.

Expect no penalty beyond the exact continuous horizon for the constant `r=1,2` controls, whose switches align at integer multiples of the prefix length. For `r=3,4`, the penalty need not be monotone with `m`. Smaller physical sample periods do not make irrational switch alignment disappear. Any asymptotic rate or universal bound would require a separate proof; do not fit a few points and label it a theorem.

A supplementary planning derivation, not a new experiment branch, is that the fold is an `L_infinity` isometry, not an energy isometry. A prefix concentrated near a folded endpoint can incur a much larger energy ratio than the source's 9.38× example; for `r>=2`, inverse-branch derivatives near `u=0` can scale as `u^(-1/2)`, giving an unbounded energy ratio for indicator prefixes shrinking toward that endpoint. Treat energy as an input-specific descriptive cost. Do not infer a uniform energy bound of `H_r`.

Also avoid the blanket claim that any stable low-pass filter destroys moment zeros. When the required absolute moments exist and the complete convolution output is observed, `M_k(h*z)=sum_l binom(k,l) M_l(z) M_(k-l)(h)`, so the first `r` zeros survive. Truncation, saturation, initial conditions, time variation, and unsupported moment assumptions can change that conclusion. No plant experiment is part of this study.

## 4. Implementation specification

### 4.1 Stack and numerical conventions

Use Python 3.11 or newer, NumPy, SciPy's CPU HiGHS LP interface, and Matplotlib for static figures. Use standard-library `fractions`, `decimal` only if needed for presentation, `json`, `csv`, `hashlib`, `subprocess`, `resource`, and `time`. No CVXPY, commercial optimizer, exact-LP package, neural model, GPU, or online service is necessary.

Initial dependency constraints may be `numpy>=1.26,<3`, `scipy>=1.11,<2`, and `matplotlib>=3.8,<4`. These are compatibility intentions, not a claim that every combination was tested. Select an installable combination on the actual Mac, run the smoke checks, then record and freeze the exact versions. The planning environment was Linux x86_64, Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0, and Matplotlib 3.10.8. It establishes neither Mac performance nor availability of those exact wheels there.

Use float64 for candidate search and plotting; use arbitrary-precision integers/Fractions for acceptance. Do not rely on `longdouble` providing extra precision on Apple Silicon. Verify that the requested SciPy method and options are accepted without warnings; capture solver version/build information when exposed. Use `highs-ds` explicitly for basic solutions [R10]. Establish correctness before any vectorization beyond simple array operations.

All primary fixture data are exact integers. Secondary data are integer codes divided by 16. Do not convert those source rationals through decimal strings or a rounded display. Store certificates as numerator/denominator strings or integers; store float candidates separately. Reject NaN, infinity, empty input, unsupported order, or malformed certificate records explicitly.

### 4.2 Independent certificate checker first

Create a checker with only standard-library dependencies. It must accept a prefix, `r`, `n`, amplitude bound, and either a primal or dual certificate. It must not import the discovery solver.

For every certificate, independently recompute `A=max |x_i|` from the authoritative prefix and verify it agrees with the recorded amplitude metadata and any normalization. For a primal, expand the vector or its disjoint half-open constant runs, check exact coverage of indices `0,...,n-1`, verify the exact prefix and bounds, and independently evaluate (6) with `Delta=1`. For a constant run `[a,b)` of value `v`, its contribution is `v*(b^(k+1)-a^(k+1))/(k+1)`. Check all moments equal the rational number zero. Also check (7) on tiny fixtures as a cross-check of the triangular equivalence. The production checker can use run integration to reduce cost.

For a dual, parse rational coefficients in ascending powers of `j`, check degree `<r`, evaluate `Q(j)` by exact Horner arithmetic, and check (9). Store both exact sums and their exact positive difference. A zero polynomial, nonpositive gap, invalid index range, or missing prefix is a rejected certificate.

For an optimum claim at upper length `n`, independently check its primal and a lower bound of exactly `n-1`: either a dual at `n-1`, the constant-prefix ceiling when and only when `n=continuous_ceiling(r,m)`, or the trivial domain bound when `n=m`. A ceiling below `n` plus an upper witness supplies only a bracket, not minimality. This last step prevents a correct feasible waveform from being mislabeled minimum duration.

Required negative checks: alter a nonzero tail entry by `1/16`; corrupt one certificate index; alter the prefix; reverse the sign of only one nonzero component; and submit the fixture's `n=21` dual at `n=22`, where its gap is `3025-3240=-215`. These fixtures must be rejected for the relevant property. The same dual remains valid at shorter horizons, so do not demand rejection merely because the horizon changes. Do not demand that a sign reversal of the entire valid waveform fail—it is a true symmetry. Tests must exercise scientific failure modes, not mirror implementation internals.

### 4.3 Candidate LP, conditioning, and exact recovery

Work with normalized prefix `v=x/A` and bound one. For a candidate `n>1`, use the shifted Legendre point basis

\[
B_{kj}=\frac{1}{n}L_k\!\left(\frac{2j}{n-1}-1\right),
\quad k=0,\ldots,r-1,\quad j=0,\ldots,n-1,
\tag{11}
\]

where `L_0(s)=1`, `L_1(s)=s`, and `(k+1)L_(k+1)=(2k+1)s L_k-k L_(k-1)`. This is a basis change for (7), **not a quadrature rule**. Its rows remain bounded and avoid the raw-power scale disparity. Keep a rational representation of each basis polynomial for dual conversion. Handle `n=1` separately by exact arithmetic.

Let `b=B_prefix v`. First solve the feasibility LP

\[
\min_g 0\quad\text{subject to }B_{tail}g=-b,
\quad -1\le g_j\le1.
\tag{12}
\]

Use `method='highs-ds'`, explicit bounds, and initially `primal_feasibility_tolerance=dual_feasibility_tolerance=1e-9`. These settings guide discovery only. The independent exact checker decides correctness.

**Primal reconstruction:** for each classification threshold in `[1e-6, 1e-8, 1e-10]`, regard samples strictly inside `[-1+threshold,1-threshold]` as potentially free. If more than `r` are free, that attempt has no basic reconstruction. Otherwise include every such index, then add distinct tail indices in increasing order until there are `r` pivots or all tail indices are used. Set all nonpivot tail samples exactly to their indicated bounds, `+1` or `-1`. Solve the remaining at-most-`r`-variable equations using integer-power rows and exact Fraction Gaussian elimination. When fewer than `r` tail indices exist, solve the overdetermined system and check every equation. Reject rank inconsistency, a pivot outside the box, or any nonzero exact residual. Successful classification is not an acceptance tolerance: the result must pass the exact checker. Multiply the recovered normalized tail by the exact original peak `A`, prepend the unchanged physical prefix, and pass that physical-amplitude vector to the independent checker and certificate store.

**Dual discovery:** if no exact primal is recovered, solve

\[
\max_{g,\alpha}\ \alpha\quad\text{subject to }
B_{tail}g+\alpha b=0,\quad |g_j|\le1,\quad\alpha\ge0.
\tag{13}
\]

In SciPy minimize `-alpha`. If the exact prefix moment vector is nonzero, (13) is feasible at `alpha=0` and bounded. Therefore an infeasible/unbounded solver status is a numerical or implementation problem, not a scientific result. If the exact prefix moment vector is zero, return the prefix followed by zero and skip (13), whose scale can be unbounded.

The equality marginals in (13), up to overall sign, supply a polynomial separator candidate. Convert each finite marginal to its **exact dyadic value** with `Fraction.from_float`; combine it with the rational basis (11); clear coefficient denominators and divide their integer greatest common divisor. Check (9) exactly. No continued-fraction approximation is required. Overall sign is immaterial because (9) uses an absolute prefix sum. As a compact first candidate, when the scale LP has exactly `r-1` distinct unsaturated tail indices, try the product polynomial with those indices as roots; accept it only if its exact gap is positive. The root pattern is a convenience, not a general assumption.

If the exact dual fails and there is still per-case and global budget, permit **one** additional `highs-ds` call with presolve disabled to repair candidate discovery, followed by the same bounded reconstruction checks. Repeat the primal LP (12) if the scale LP returned a finite `alpha >= 1`; otherwise repeat the scale LP (13) for another dual candidate. That floating branch choice only selects a discovery attempt; it never accepts a scientific claim. Otherwise return `unknown`, retaining the solver status and rejected candidates. Never convert `unknown` into `infeasible`.

Each candidate horizon may use at most three LP discovery calls in total: the initial primal call, the scale/dual call, and the one repair call. This limit is cumulative for the exact prefix hash, order, bound, and horizon across search paths and resumes; cache exhausted unknown attempts as well as certificates. The six explicitly declared timing replays of one already-certified development case in §6.3 are the only separate repeat measurements, count toward every global/stage budget, and cannot be used to rescue an unresolved case. Each call has a five-second solver time limit and an external process watchdog; the complete horizon oracle has a 20-second limit. Exact checking is also inside the limit. A timeout retains an unresolved bracket. Do not increase precision, change solver families, install new optimizers, or relax exact acceptance indefinitely.

### 4.4 Horizon search and termination

A simple constructive upper bound is provided by Prouhet–Thue–Morse (PTM) copies [R8]. Set

\[
s_j=(-1)^{\operatorname{popcount}(j)},\quad 0\le j<2^r,
\qquad z_{jm+i}=s_jx_i.
\tag{14}
\]

The polynomial `sum_j s_j w^j=product_(l=0)^(r-1)(1-w^(2^l))` has a root of order `r` at one. Expanding `(jm+i)^k` proves (7) for every `k<r`. The first block is exactly the prefix, the peak is unchanged, and `n_PTM=2^r m` is always feasible. Its total energy is exactly `2^r` times prefix energy in the unquantized model. This is a positive control and a finite search bracket, not an optimal method.

Create an exact PTM witness and verify it before searching. If it fails, stop comparison work and repair correctness; do not increase the search horizon.

Use the following logic, caching every certified result and every exhausted unknown attempt, and never changing a bound on a floating-point status alone:

```text
certified_minimum(prefix x, order r):
    validate input; A = exact max absolute prefix amplitude
    if prefix's first r power moments are exactly zero:
        return n=m, exact prefix witness, domain lower bound
    upper = 2**r * m, with independently checked PTM witness
    if x is the saturated constant +A or -A:
        lower = continuous_ceiling(r,m) - 1  # proven infeasible
        n0 = lower+1
        try n = n0, n0+1, n0+2 in that order
        retain every certified feasible upper or infeasible lower
        an unknown at n0 may be followed by the predetermined n0+1 guard query
        do not move either bound because of an unknown
        stop immediately if upper == lower+1
    else:
        lower = m  # nonzero prefix moments; no tail is infeasible
    while upper > lower+1 and case/global budgets allow:
        n = floor((lower+upper)/2)
        status, certificate = horizon_oracle(x,r,n)
        if exact_feasible: upper = n; keep witness
        if exact_infeasible: lower = n; keep certificate
        if unknown: stop this case with its current bracket
    independently recheck the final bracket and certificates
    return exact minimum if adjacent; otherwise [lower+1, upper]
```

Do not query outside `m <= n <= 2^r m`; cap the search at 16 distinct candidate horizons and 90 seconds per instance. For the constant scan, skip candidates already outside the current bracket. An `unknown` horizon cannot justify binary-search direction. A case may remain unresolved even when the floating-point solver appears confident. If bisection revisits a horizon whose discovery allowance was exhausted, reuse `unknown` and retain the bracket; do not grant it another three calls. In particular, an unknown at `n_0` plus a feasible `n_0+1` gives the bracket `[n_0,n_0+1]` and guard support, not a new retry campaign.

The direct LP is the closest strong feasible comparator: it searches all real-valued tails in the exact model, rather than only a folding family. There is no need to implement Bako et al.'s sparse relaxation [R7] for this tiny-row problem. A certified minimum from the ordinary LP is already the strongest possible duration result for that instance under the stated constraints.

### 4.5 Continuous and sampled reference implementations

Implement or carefully adapt the source's transported-segment fold. For `m` uniform prefix cells, use boundaries `u_i=i/m`, map them through (3), reverse boundaries and values on odd branches, and attach signs. Set the first branch boundaries and values directly from the input to preserve them exactly. Use the `arcsin(sqrt(...))` expression, clamp only tiny roundoff excursions of valid arguments, and fail on nonmonotone boundaries. Do not silently merge or reorder invalid segments. At `r<=4` and the planned sizes, segment count is `(r+1)m`.

Create two digital projections at `n_0=ceil(m H_r)`, extending the ideal continuous waveform by zero after `H_r D`:

| Method | Definition | Purpose |
|---|---|---|
| `fold_midpoint` | Sample at each actual cell center; return zero when the center lies after the ideal endpoint | Reproduce the source's basic projection mechanism with honest full-cell timing |
| `fold_average` | Average the ideal segments over each full output cell, including zero beyond the ideal endpoint | Targeted ablation of midpoint sampling; preserves area in exact arithmetic and the amplitude bound, but does not generally preserve higher moments |
| `ptm` | Signed whole-prefix copies from (14) | Simple exact-grid reference, positive control, finite upper bracket |
| `lp_certified` | Exact primal returned by the certified search | Strong minimum-duration comparator; no energy-optimality claim |
| `zero_tail` | Prefix followed by zeros | Negative control; detect an evaluator accidentally ignoring the prefix |

Use a linear sweep over sorted ideal segments and output cells for averaging; avoid an all-pairs overlap matrix. Both projections retain the prefix cells exactly. Do not use the source's `round(H*m)` or its clamping of the last center into the ideal support. The corrected projection may differ from the legacy file; retain that distinction.

For same-horizon residual comparisons, evaluate the two folds and a certified LP feasibility result at `n_0`, and, if that horizon is certified infeasible, at `n_0+1` when available. Pad the fold with a zero sample at the longer common horizon; do not alter its existing samples. PTM has a different constructive duration and belongs in the duration/exactness table, not a misleading equal-latency performance ranking. Record an infeasible common-horizon LP as infeasible, not a missing successful sample.

Do not include detrending or ordinary high-pass filtering as competitors: they change the preserved prefix and solve a different problem. A constrained FIR formulation that retains the prefix and imposes these moment/box equations is another representation of the same feasible set, not an omitted stronger benchmark.

### 4.6 Measurement functions and quantization boundary

For every output record its representation: `ideal_segments`, `rational_grid`, `float64_grid`, or `decoded_pcm16`. Keep these domains in separate columns.

Exact feasibility is decided by rational equalities, not a tolerance. For descriptive floating residuals, use a **fixed per-input reference duration** `T_ref=2^r D` shared by every method, and report

\[
\mu_k(z)=\frac{k+1}{A T_{ref}^{k+1}}M_k(z),\qquad
R=\max_{k<r}|\mu_k|.
\tag{15}
\]

Also retain the individual raw physical moments, actual duration, sample period, and scale. The fixed normalization prevents a method from appearing more accurate solely by dividing by a longer output duration. For legacy audit tables only, additionally reproduce the source's ideal-horizon normalization and a separately labeled actual-duration normalization.

Calculate floating held-sample moments with exact cell antiderivatives and compensated summation (`math.fsum` is adequate here). For small adjacent normalized endpoints `a,b`, evaluate the power difference as `(b-a)*sum_(ell=0)^k b^(k-ell)*a^ell` to reduce subtraction error. No trapezoidal integration, FFT inference, or midpoint quadrature is an acceptance oracle.

Calibration sets a reporting/float-validation threshold `tau_float=max(1e-12,20*e_cal)`, where `e_cal` is the largest disagreement on the declared tiny/calibration controls between float measurement and independently evaluated exact rational hold moments. Require `tau_float <= 1e-9`; otherwise repair the evaluator or mark float comparisons unavailable. Freeze this value before evaluation. It does not replace any exact certificate or become a success target for the sampled fold.

Additional descriptive quantities are actual `n Delta`, `max |z_j|`, exact prefix preservation, total energy `Delta sum z_j^2`, energy ratio to the nonzero prefix, and total variation of the held sequence. Energy and total variation are diagnostics only: no energy optimization or physical bandwidth guarantee is being tested. The LP may return a high-energy minimum-duration solution.

For the small wordlength diagnostic, use secondary prefixes `x_i=a_i/16` without changing their amplitudes. Map physical normalized amplitude to symmetric signed PCM codes with scale `S=32752=16*2047`: `code=round(S*z)`, with nearest-even ties, and decode as `code/S`. This preserves those input samples exactly and keeps `+A` and `-A` representable, avoiding the asymmetric `-32768` negation problem. Restrict the codes to the shared bound `max |a_i|*2047`; valid inputs/outputs should not require clipping. Implement rounding from rationals exactly for LP/PTM witnesses. For a float fold sample, first use `Fraction.from_float(float(z_j))` and then the same exact nearest-even integer rounding; do not let a floating multiplication by `S` introduce an additional rounding ambiguity. A `.wav` file is unnecessary for this synthetic diagnostic; integer arrays suffice.

Quantized tail error is at most `1/(2S)` per sample under this convention. With prefix unchanged and `n <= 2^r m`, (15) gives the independent bound `|delta mu_k| <= 1/(2SA)` when `A>0`; the actual weighted tail fraction can tighten it. A violation indicates a codec/evaluator defect. This is an error bound, not exact fixed-wordlength feasibility. Do not attempt integer programming for the optimum PCM horizon.

For the supplied WAVs, retain their actual signed-16-bit convention and decode as integer code divided by 32768. Do not impose the synthetic scaling on them. Read the file again after any export [R11], and check prefix sample identity in decoded integer codes.

## 5. Experiment set and fairness

### 5.1 Development, exploration, and locked evaluation

The planning team already explored **all constant-prefix cases `r in {3,4}`, `m=1,...,64`**, using floating LP discovery and exact certificates. All 128 were independently checked; only `(r,m)=(4,2)` and `(4,36)` required one extra tick beyond upward rounding. This is development evidence, not unseen confirmation. The supplied examples and every number in this document are also development material.

For your development and calibration, use only constant `m in {1,2,4,16,32,36,64}`; `r=1,2` analytic controls; and the secondary generators below at `m=16` with development seeds. Additionally permit development-seeded `iid_codes` at `m=32,64` for calibration. This avoids reusing deterministic evaluation prefixes during calibration. You need not rerun the entire planning sweep. The planning sweep's scripts are not required inputs and are not a substitute for your independent implementation.

The **locked core** is exactly `r in {3,4}`, `m=65,...,128`, `x_j=1`: 128 instances. The **locked scaling extension**, reached only if the calibration gate permits, is `r in {3,4}`, `m in {256,512,1024}`: six instances. Do not inspect these results while selecting solver settings, reconstruction thresholds, or tolerances.

Freeze the code revision, configuration, complete case manifest, generator definitions, tolerance calibration, numerical backend, and a hash of each before opening locked results. The primary oracle and acceptance criteria are fixed by this brief. Later scientific changes must be labeled exploratory; they cannot quietly redefine the locked result. An implementation repair with unchanged mathematical protocol may rerun affected cells once, preserving the first attempt and the reason. If a hypothesis or distribution changes, no fresh confirmatory expansion is authorized under this budget; report the changed work as exploratory.

### 5.2 Minimal experiment matrix

| Block | Cases | Claim or uncertainty resolved | Required evidence |
|---|---:|---|---|
| Analytic fixture and tiny checks | Two exact obstruction fixtures; small controls specified below | The digital extrapolation fails; checker/LP/basis are correct | Standard-library exact verification, independent moment integration, rejected corruptions |
| Development and calibration | At most 24 case executions plus timing repeats | Candidate recovery works; cost and precision fit limits | Raw solver records, exact certificates, time/memory estimates |
| Locked constant core | 128 | Quantifies the extra clock ticks on untouched contiguous resolutions | Exact `n_min` or honest certified bracket for every declared cell |
| Locked scaling extension | 6, conditional | Checks whether certificate generation and the finite one-tick observation extend to larger grids | Same certificates; no 48 kHz extrapolation |
| Secondary nonconstant prefixes | 16 prefixes × 2 orders = 32 instances | Rules out a constant-only implementation; compares universal fold with input-specific duration | Prefixes, common-bound LP minima/brackets, fold/PTM residuals and costs |
| Synthetic wordlength diagnostic | Two prefixes × 2 orders × 3 methods = 12 quantizations | Separates real-amplitude feasibility from representation error | Exact code arrays, residual changes, independent quantization bound |
| Legacy accounting audit | Two supplied example directories | Corrects endpoint, PCM, and energy interpretation | Decoded sample counts, rate, identity, held moments, energy |

Every case and failed attempt counts against the global caps. A block skipped by a gate remains explicitly unperformed. Do not generate a large random benchmark or seek a favorable application after a negative result.

### 5.3 Tiny controls before comparative interpretation

Required controls are:

- Constant `r=1,2`, `m in {1,2,7,16}`: exact minima `2m` and `4m`, respectively. PTM provides exact feasible witnesses; the continuous lower bound certifies minimality.
- The `r=4,m=2` 21/22 fixture, including a deliberately corrupted witness.
- The `r=4,m=36` dual gap; solve and verify a 378-sample witness during the pilot.
- Zero input; a nonzero prefix whose area is zero but a higher moment is not; and sign-negation invariance. For the area-only case, `(1,-1)` is adequate.
- A nonzero measurement oracle: with `A=D=1`, constant prefix `+1`, and any zero continuation, (15) must give `mu_k=2^(-r*(k+1))` for every `k<r`. Verify this exact reference and its float measurement before calibrating `tau_float`; include its discrepancy in `e_cal`. This catches an evaluator returning zero, omitting the prefix, or using the wrong normalization.
- Exhaustive tiny prefix-grid oracle at `r=2,m=2,n=4` for prefixes from `{-1,0,1}^2`: solve the at-most-two-variable linear equations exactly and check bounds; compare with the LP-guided oracle. This is nine deterministic prefixes, not Monte Carlo.
- PTM identity for at least one nonconstant rational prefix at each `r=1,...,4`; independently check ordinary held moments.
- One reference fold at each `r=1,...,4` on the development input `(1,-1/2,1/4,0)`, with unchanged prefix, unchanged peak, valid segment ordering, and small ideal integrated residual.

The tiny oracle is independent of LP discovery and catches incorrect bounds, omitted prefix terms, indexing, order conventions, and row scaling. An intentionally invalid certificate must never be accepted because a floating residual is small.

### 5.4 Secondary data, exactly specified

Use two evaluation lengths, `m=64` and `m=256`. For each length generate eight integer-code prefixes below and set `x_i=a_i/16`. Use the actual common peak `A=max |a_i|/16` for every method. No mean removal, clipping of valid deterministic data, resampling, or per-method normalization is allowed. Store the integer arrays before running any method.

| ID | Integer generator for `i=0,...,m-1` | Reason |
|---|---|---|
| `biased_ramp` | `a_i = 8 + floor(8*i/(m-1))` | Persistent positive offset plus drift |
| `bipolar_ramp` | `a_i = -16 + floor(32*i/(m-1))` | Low-frequency signed trend |
| `alternating` | `a_i = 16*(-1)^i` | Fast sign changes and little net area |
| `early_impulse` | `a_0=16`, all other codes zero | Concentrated endpoint energy; exposes folding cost |
| `middle_burst` | `a_i=16` for `m//3 <= i < m//3 + max(1,m//8)`, otherwise zero | Localized signal away from the endpoint |
| `signed_step` | `a_i=16` for `i<m//3`, otherwise `a_i=-8` | Asymmetric step with substantial low-order moments |
| `iid_codes` | Independent uniform integers in `[-16,16]`, inclusive | Broad irregular finite-alphabet input |
| `bounded_walk` | `a_0=8`; `a_i=clip(a_(i-1)+e_i,-16,16)`, independent uniform integer `e_i in [-2,2]` | Correlated, slowly varying bounded disturbance |

For the two random generators explicitly instantiate `numpy.random.Generator(numpy.random.PCG64(seed))`. Let family index be 7 for `iid_codes` and 8 for `bounded_walk`. Use evaluation seed `2026110500 + 1000*family_index + m`, development seed `2026110400 + 1000*family_index + m`. Seeds identify reproducible examples; these 16 prefixes are not a population sample with inferential confidence intervals. `bounded_walk` clipping is part of the generator, not a method-specific preprocessing step.

Run both orders three and four on each prefix. For the wordlength diagnostic use the evaluation `biased_ramp` and `early_impulse` at `m=64`, both orders, and the three methods `fold_average`, `ptm`, and the certified LP witness when available. If an LP optimum is unresolved, a separately labeled certified feasible upper witness may be quantized; it must not be called minimum duration. No additional seed search is allowed.

These inputs isolate exact linear constraints and implementation behavior. They do not stand in for environmental audio, calibrated sensors, human subjects, or a realistic stochastic noise model. A new external dataset is unnecessary for this question.

### 5.5 Metrics, comparisons, and decisions

The primary metric for a constant instance is

\[
\delta(r,m)=n_{\min}-\lceil mH_r\rceil\quad\text{(integer extra clock ticks)}.
\tag{16}
\]

Report also `n_min/m`, total continuous-to-grid excess `n_min/m-H_r`, its decomposition into ordinary upward rounding plus `delta/m`, and `delta/f_s` only at an explicitly stated illustrative sample rate. An illustration is a unit conversion, not a measurement at a new prefix length. In particular, testing `m<=1024` does not validate the one-second `m=48000` frontier.

Track **minimum status** separately from **guard status**. One checked dual at `n_0` contradicts rounded-horizon sufficiency, even before the exact minimum is located; that fact is already established by the fixture. A checked primal at any `n <= n_0+1` proves that the one-tick guard suffices for that instance, even if its minimum remains in `[n_0,n_0+1]`. A checked dual at `n_0+1`, or a checked infeasible lower bound at a larger horizon, proves that the guard fails, even before the exact minimum is located. Record `guard_supported`, `guard_refuted`, or `guard_unknown` accordingly.

Full characterization of the frontier requires exact minima for every declared cell. Support for the secondary guard on the complete locked finite set requires a sufficient checked upper witness for every cell; it does **not** require exact minima. Report both coverage counts. A single guard-refuting certificate defeats the finite-set rule despite unresolved cells elsewhere. If neither sufficient upper witnesses for the entire set nor a refuting certificate exists, the guard conclusion is inconclusive. No finite-set outcome establishes a universal guard theorem. Report core guard coverage against exactly 128 cases and scaling guard coverage against exactly six cases separately. Skipping the conditional scaling set does not prevent a justified conclusion about a fully checked core, but no combined core-plus-scaling claim is then permitted.

For nonconstant inputs, report `n_min/m` or a bracket relative to the fold's universal horizon and the PTM duration. A value below `H_r` measures input-specific opportunity, not a negative digital penalty: the continuous input-specific optimum is unknown here. Do not infer that constants are the worst prefixes on every digital grid merely because they are worst in the continuous theorem.

No p-values, bootstrap intervals, or repeated seeds are needed for exact deterministic certification. Report finite-set counts, ranges, all exceptions, and coverage. If reporting exploratory aggregate residuals across the 16 prefixes, label them descriptive, retain per-instance values, and do not treat each sample, moment, or solver iteration as an independent trial.

All methods receive the same complete prefix, the same peak budget, the same clock and moment definitions, and comparable numerical evaluation. The duration is the quantity being optimized, so it may differ across construction families; common-horizon residual comparisons use §4.5. Do not compare incompatible published performance numbers.

## 6. Hardware feasibility, accounting, and evidence gates

### 6.1 Inspect the actual Mac

Before allocations, record macOS version, CPU model, architecture, physical RAM, available memory, free project-volume space, Python architecture/version, NumPy/SciPy versions, BLAS configuration, process/thread settings, and power state. Suitable local commands include `sw_vers`, `uname -m`, `sysctl -n machdep.cpu.brand_string`, `sysctl -n hw.memsize`, and Python's platform/configuration APIs. Capture failure of an inspection command rather than assuming a value. Do not infer RAM, OS, installed libraries, or precision from “M3 Max.”

Prefer a native arm64 Python. If only an emulated environment is available, complete correctness work and label timings accordingly; do not call them native M3 Max performance. Use CPU float64 and exact Python arithmetic. Verify the selected solver actually runs; no CUDA, MPS, or GPU synchronization is involved.

Set one computational thread for comparable timing where supported; record rather than hide any backend that ignores the setting. Start with `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, and `VECLIB_MAXIMUM_THREADS=1`. Do not claim determinism from these variables alone. Close unrelated high-load jobs if the user has already authorized that ordinary action; otherwise merely record observed load and do not terminate their processes.

### 6.2 Concrete ceilings and scaling

These are planning ceilings, not measured M3 Max estimates:

| Resource | Ceiling and enforcement |
|---|---|
| Total local experimental wall time | **240 minutes**, cumulative across all stages, failed runs, repeats, tuning, preprocessing, certificate checking, and repairs |
| Protected finishing budget | Last **45 minutes** reserved for verification, preserving evidence, analysis, and report regeneration; stop launching new experiments when it would consume this reserve |
| Peak memory | Lesser of **4 GiB** and **25% of measured physical RAM**; also require measured currently available memory to cover the calibrated job with a twofold margin |
| Project disk | **2 GiB** including inputs, environments/results attributable to the study, logs, and figures; check before starting each block |
| Maximum output length in frontier/secondary search | `2^4*1024 = 16384` samples; no LP on the 48 kHz legacy waveform |
| LP calls | **1,600** total, including repairs, cold/warm timing repeats, development, and failed calls |
| Candidate horizons per instance | At most **16**, within the 90-second instance limit |
| Horizon oracle / individual solve | **20 seconds** including exact checking / **5 seconds** solver limit; watchdog kills and records overrun |
| Repair allowance | At most **30 minutes** cumulative local repair/test computation and no more than the remaining global LP quota; this is part of, not additional to, 240 minutes |

Candidate matrices have `r` rows and `n-m` columns, so storage is `O(rn)` and assembly/evaluation is `O(rn)`. A 4×16,384 float64 matrix is about 0.5 MiB; solver workspace and imports can dominate that. Never allocate an `n×n` matrix. Fold storage is `O((r+1)m)` and cell averaging is linear in segments plus samples. A certificate needs only an at-most-4-variable exact solve and linear-size checking, but integer bit lengths and solver pivots can grow; calibration must measure them. Do not assume a linear wall-time law for simplex.

Measure calibration at `m=16,32,64`, `r=4`, using a constant and the development-seeded `iid_codes` prefix. Fit only a conservative cost envelope from observed cases; use the largest observed adjacent-size exponent with a floor of two for projecting LP time, and a twofold safety factor. This is an allocation heuristic, not a complexity claim. Include certificate reconstruction/verification, serialization, and process overhead. Project the locked core and six scaling jobs separately.

If the core cannot fit in its stage budget with that margin, retain the analytic result and do not silently replace the core with a favorable subset. A partial started core is reported with its original denominator of 128. The conditional scaling extension may be skipped entirely before evaluation if its projection exceeds its allocation; list all six skipped cases. Do not shrink `r` or change exact acceptance to obtain success. Secondary diagnostics may be skipped as a whole when their gate fails.

The actual anticipated work is small: an informative analytic/tiny pilot should take minutes of compute, with a one-hour cumulative ceiling for G0–G3 including any repairs in those gates. Shared repair time does not extend that pilot ceiling; reduce remaining pilot work or stop with the valid analytic result when it is reached. The full matrix is expected to fit in hours, but this is unmeasured until calibration. Expected setup is 10–30 minutes, implementation/review roughly 3–6 hours, and human/agent analysis roughly 30–60 minutes, separate from the experimental cap. A reasonable end-to-end planning estimate is one focused working day, roughly 5–10 hours, not days of compute. Dependency download time and hosted-agent reasoning are not experimental computation; disclose them separately. If setup exceeds 30 minutes, complete the standard-library analytic path and identify the dependency blocker rather than spending an unbounded repair session.

### 6.3 Timing and memory protocol

Run scientific cases sequentially in bounded worker processes. Start the timed boundary before input parsing/normalization and matrix assembly, and end after certificate generation and independent verification. Separately report generation/preprocessing, candidate solve, reconstruction/checking, and serialization times. Also measure one cold command including Python imports and five warmed repetitions on a fixed development case at `r=4,m=64`; use median and range, not a performance-significance test. All six executions count toward the caps. Do not repeat deterministic correctness results for statistical effect.

Record process peak resident memory using a verified platform-specific method, e.g. macOS `/usr/bin/time -l` or `resource.getrusage`, documenting the units [R12]. A parent can poll the child RSS with system tools for the watchdog; polling may miss short peaks, so retain the kernel high-water measurement when available. Measure a baseline import process separately. Never compare a Python-only allocation counter with total solver RSS or report one machine's resource values as another's.

Use `time.perf_counter` for wall time, retain raw records, and record process exit status, timeout, memory kill, and solver message. The resource ledger is cumulative and durable; resuming, changing stages, or restarting a process never resets it. Resolve its canonical location from the project script directory as `results/budget.json`, independently of `--out` or the caller's working directory. The analytic/tiny checker, smoke run, calibration, timing repeats, worker processes, and final verification must all use it and the adjacent `attempts.jsonl`. Charge each top-level job interval once; nested timing components are a breakdown, not a second charge. Initialize this standard-library ledger before the first analytic check. Before launching a job, require its allowed maximum to fit the remaining stage/global budget while preserving the finishing reserve.

### 6.4 Evidence gates and authorized next actions

| Gate | Question, prerequisites, and minimum work | Ceiling | Decision and next action |
|---|---|---:|---|
| **G0 — constructive feasibility** | Inspect inputs/environment; reproduce (10), the 22-vector moments, and the exact ceiling with a standalone script; establish real-amplitude scope and source hashes | 10 min | If exact arithmetic verifies, the analytic negative result is available. If it does not, stop broader work and investigate the discrepancy; no benchmarking of an unverified fixture. Missing archives block only their audit; missing Mac blocks only Mac claims. |
| **G1 — tiny correctness** | Independent checker, PTM, tiny oracle, LP candidate path, corruption checks, and `r=1,2` controls | 20 min | Continue only when every mandatory exact control passes. Allow one bounded repair cycle. If unresolved, report an invalid computational implementation plus any independently verified analytic result. |
| **G2 — resource calibration** | Development-only costs, memory, exact-recovery rates, float evaluator error, and projected workload | 10 min | Freeze a feasible allocation and `tau_float`; skip conditional blocks that fail projection. If the fixed core is infeasible, stop at the analytic diagnostic with measured resource evidence. |
| **G3 — mechanism pilot** | Reproduce `m=2,36` infeasibilities and next-tick primals; sample-average area control; small nonconstant/PTM controls; inspect solver/checker disagreements | 20 min | Continue if the experiment distinguishes exact impossibility from candidate failure. A numerical ambiguity is not rejection; one specifically targeted repair is allowed within shared limits. Otherwise preserve an unresolved or invalid extension, not a false refutation. |
| **G4 — locked comparison** | Freeze protocol; run 128 core cells; conditional six scaling cells; up to 32 secondary cases, 12 quantizations, two legacy audits | 105 min: core 75, scaling/secondary/audit together 30 | Publish every result/status. Any certified `delta>=2` settles the finite one-tick hypothesis negatively; finish remaining affordable core cells only to characterize the declared set, not to reverse the finding. At the cap, retain certified brackets and stop new jobs. |
| **G5 — diagnosis and report** | Recheck all asserted certificates from saved files in a fresh process; regenerate essential tables/figures; audit exclusions, failed runs, units, and provenance | 45 min protected | Deliver the strongest valid conclusion for completed stages. No new scientific experiments in this reserve. |
| **Shared bounded repair reserve** | Repairs and their tests across G0–G4; no additional scientific branch | 30 min | Once exhausted, mark affected work blocked/invalid/unknown as appropriate. Do not consume the G5 reserve. |
| **Total** | All local experimental work above | **240 min** | Stop at the earliest decisive gate or resource boundary; the caps are not targets to consume. |

An analytic stop after G0 is appropriate if only the rounded-horizon implication needs resolution, or if indispensable implementation/hardware resources are blocked. Under normal available resources, execute the authorized finite extension through its gates; do not stop merely because the first counterexample is convenient. If an essential theorem defect is discovered, re-evaluate the lower bound before using it to certify any frontier; direct primal/dual certificates can still stand on their own.

## 7. Repository, interfaces, commands, and reproducibility

### 7.1 Minimal project layout

Create one ordinary local directory, `counterrandom_grid_study/`. Keep the implementation small; do not create a general optimization framework.

| Path | Responsibility |
|---|---|
| `README.md` | Exact setup, command sequence, result locations, model and limitations |
| `requirements.txt`, `requirements-lock.txt` | Initial constraints and the exact smoke-tested environment |
| `run_study.py` | CLI, stage gates, worker limits, freeze/resume, raw results, report generation |
| `model.py` | Input generators, exact ceilings, continuous fold, midpoint/average projection, PTM, independent float metrics |
| `solver.py` | Scaled LP construction, bounded candidate recovery, horizon search; no authority to declare unchecked certificates valid |
| `check_certificates.py` | Standard-library-only primal, dual, and minimality checking; direct held-signal integration |
| `checks.py` | Mandatory exact fixtures, tiny oracle, corrupted controls, and smoke checks |
| `study.json` | Fixed scientific parameters and resource caps |
| `inputs/` | Unmodified supplied files or extracted copies with a manifest; never overwrite original evidence |
| `results/environment.json`, `results/input_manifest.json` | Actual hardware/software/backend and input hashes |
| `results/protocol_lock.json`, `results/cases.jsonl` | Frozen settings/code hashes, exact case identities and input hashes |
| `results/attempts.jsonl`, `results/budget.json` | Canonical repository-root attempt ledger and cumulative resource accounting, independent of `--out` |
| `results/certificates/`, `results/prefixes/` | Exact rational witnesses/separators and raw generated integer prefixes |
| `results/tables/`, `results/figures/`, `REPORT.md` | Reproducible analysis and final research interpretation |

Suggested callable interfaces, which you must implement rather than assume already exist:

```text
exact_ceiling(r: int, m: int) -> int
make_prefix(family: str, m: int, seed: int | None) -> list[int]  # codes / 16
fold_segments(prefix, r) -> (edges, values, ideal_duration)
project_segments(edges, values, delta, n, mode) -> array
ptm_completion(prefix, r) -> rational_vector
horizon_oracle(prefix, r, n, limits) -> status_and_certificate
minimum_horizon(prefix, r, limits) -> minimum_or_certified_bracket
verify_primal(prefix, r, n, bound, certificate) -> verification_record
verify_dual(prefix, r, n, bound, certificate) -> verification_record
measure_output(output, representation, delta, bound, reference_duration) -> metrics
```

Module boundaries are engineering guidance; you may simplify them. The model, inputs, comparisons, exact acceptance, locked set, failure interpretation, and cumulative caps are scientific requirements and must not be silently changed.

### 7.2 Configuration to create

Use these defaults and explicitly record any gate-based omission. The configuration is concrete; its schema is yours to implement.

```json
{
  "protocol_version": "2026-11-04-clock-grid-1",
  "primary_orders": [3, 4],
  "development_constant_m": [1, 2, 4, 16, 32, 36, 64],
  "calibration_m": [16, 32, 64],
  "calibration_nonconstant_family": "iid_codes",
  "calibration_seed_phase": "development",
  "locked_core_m_inclusive": [65, 128],
  "conditional_scaling_m": [256, 512, 1024],
  "secondary_m": [64, 256],
  "secondary_families": ["biased_ramp", "bipolar_ramp", "alternating", "early_impulse", "middle_burst", "signed_step", "iid_codes", "bounded_walk"],
  "amplitude_bound": "actual_prefix_peak",
  "moment_model": "complete_uniform_zero_order_hold_cells",
  "primary_amplitude_model": "bounded_real",
  "exact_acceptance": true,
  "lp_method": "highs-ds",
  "lp_primal_tolerance": 1e-9,
  "lp_dual_tolerance": 1e-9,
  "active_set_thresholds": [1e-6, 1e-8, 1e-10],
  "solver_seconds_per_call": 5,
  "oracle_seconds": 20,
  "case_seconds": 90,
  "max_horizons_per_case": 16,
  "max_lp_calls_total": 1600,
  "experiment_minutes_total": 240,
  "protected_finish_minutes": 45,
  "repair_minutes_total": 30,
  "maximum_output_samples": 16384,
  "disk_gib_cap": 2,
  "memory_gib_cap": 4,
  "memory_fraction_cap": 0.25,
  "synthetic_pcm_scale": 32752,
  "secondary_one_tick_rule": "delta <= 1 on the complete locked finite set only"
}
```

`tau_float`, observed device values, frozen dependency versions, and conditional-block admission belong in the generated `protocol_lock.json` after calibration. They are not unresolved scientific choices: their selection rule is specified in §§4.6 and 6.2. Persist the original declaration even if the admitted workload is smaller.

### 7.3 Intended command sequence

The following are **interfaces to create**, not commands already executed on a Mac. Run from the new project directory. Standard setup commands are ordinary Python workflow; their successful execution on the target still must be verified.

First create the small dependency-free fixture/checker path. Defer all NumPy/SciPy/Matplotlib imports until their non-analytic functions are invoked; the analytic command below must not import `solver.py` or a numerical-library-dependent `model.py` at module import time.

```bash
python3 checks.py --analytic-only

python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip freeze > requirements-lock.txt

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

python run_study.py inspect --inputs inputs --out results
python checks.py --tiny
python run_study.py calibrate --config study.json --out results
python run_study.py pilot --config study.json --out results
python run_study.py freeze --config study.json --out results
python run_study.py run --config study.json --out results --resume
python check_certificates.py --results results
python run_study.py report --results results --out REPORT.md
```

For a minimal smoke run, implement `python run_study.py smoke --out results_smoke` to execute the 21/22 fixture, one `r=2` control, and one nonconstant PTM/fold case. Its attempts count in the same study ledger; a new output directory does not create a new budget. The standard-library `checks.py --analytic-only` path must work without NumPy/SciPy/Matplotlib so the exact diagnostic survives dependency failure.

`run --resume` must read the existing global ledger, reuse verified cached certificates, retain failed attempts, refuse to run locked evaluation before a valid freeze, and refuse a mismatched configuration hash unless an explicit deviation record is created. No remote compute or paid service may be used for an experimental fallback.

### 7.4 Raw record contract

Every scientific case must have a stable ID based on phase, family, `r`, `m`, and seed when applicable. Record exact prefix identity, amplitude scale, candidate `n`, representation, method, and status. Every attempt must retain start/end time, wall time, LP count, peak memory measurement/unit, solver status/message, basis/scaling choice, certificate path/hash, verification status, and reason for timeout or rejection.

Allowed scientific statuses include `exact_feasible`, `exact_infeasible`, `exact_minimum`, `certified_bracket`, `unknown_numeric`, `timeout`, `resource_limit`, `invalid_input`, and `invalid_implementation`. Keep discovery status separate from verification status. Also record `minimum_status` independently of `guard_status`, with the exact sufficient upper or refuting lower certificate that supports the guard decision. A successful LP may yield a rejected candidate; an unsuccessful LP does not prove the instance infeasible.

Store both the best certified lower and upper bound even when a case fails. Unresolved cases remain in every coverage denominator. Never compute a success percentage only on cases where the solver returned a usable vector. No silent exclusions or dropped failed jobs are allowed.

A compact certificate format may encode constant runs as `[start_inclusive,end_exclusive,"numerator/denominator"]`; alternatively store all fractions. Duals must include coefficients and exact gap sums, not merely roots inferred from a plot. Retain source code hashes, generated-prefix hashes, dependency versions, actual platform, solver configuration, and the frozen case manifest. This is sufficient for independent regeneration without a container image or workflow service.

## 8. Analysis, honest outcomes, and required deliverables

### 8.1 Essential analysis only

Generate a primary table with one row per declared constant-grid case: phase, `r`, `m`, `n_0`, exact `n_min` or interval, `delta` or interval, certificate status, separate one-tick guard status, and total solve/check cost. Separate development from locked evidence visually. Give a compact summary listing every `delta>0` case and every unresolved case; do not average away clock alignment exceptions.

For secondary cases, show latency, exactness/representation, peak preservation, floating residuals at the common reference normalization, and descriptive energy. A residual chart must indicate exact rational zeros as exact values, not replace them with arbitrary epsilon bars. Distinguish missing/unknown from zero. If a log scale needs a plotting floor, annotate it and retain actual zeros in the table.

Produce at most three essential figures: (1) extra ticks versus `m`, with unresolved intervals and phases marked; (2) a held-step view of the exact 22-sample fixture and its infeasibility certificate summary; (3) secondary residuals by method and representation if that block ran. Static publication-quality PNG or PDF output is sufficient; an interactive app adds no scientific value here.

The legacy audit table must distinguish reported ideal numbers, reported floating sampled numbers, recomputed floating projections when performed, and the actual decoded files. Use actual sample counts/rates and identify the normalization used. Do not modify the old reports in place.

### 8.2 Outcome classifications

Classify each material claim and then the overall study. Different subclaims may have different outcomes.

| Outcome | Evidence required | Wording and consequence |
|---|---|---|
| **Supported contribution** | Exact fixture verified; declared scope backed by independently checked primal/lower certificates; no unresolved cells hidden | A scoped clock-grid limitation or finite frontier is supported. The known ingredients and uncertainty about historical priority remain explicit. |
| **Informative negative result** | An exact obstruction refutes a proposed digital implication or finite one-tick rule; alternatively an adequate valid comparison removes a claimed advantage | Identify exactly which claim failed. The continuous theorem and all other applications are not refuted. |
| **Practical infeasibility** | Measured resource ceiling or supported calibrated scaling blocks the intended computational extent | Retain the analytic result and completed cells; state whether the implementation, exact certification, or target workload caused the limit. This is not mathematical impossibility. |
| **Inconclusive evidence** | Unresolved certificate intervals, missing target hardware/input, or insufficient budget prevent the claimed scope | Report the exact unresolved portion and why; do not convert absence of a counterexample into support for the full rule. |
| **Invalid experiment** | Incorrect input, baseline, constraints, checker, units, or measurement invalidates attempted comparisons | Withdraw affected numerical conclusions. Independent valid arithmetic fixtures may remain, with the separation explained. |

A bug, timeout, or inability to reconstruct a rational basis is not evidence against feasibility. A feasible solution at some duration is not evidence of minimality. A finite exact table is stronger than an approximate table for its listed cases but is not a proof for all `m`, all prefixes, higher orders, or finite-wordlength hardware.

The expected useful result is a corrected model and an exact account of a specific obstruction. If ordinary LP provides the appropriate completion, say so; the research does not require the Chebyshev construction to beat it. If the one-tick rule fails, report the new exact counterexample without searching for a revised favorable threshold.

### 8.3 Final report contents

Write `REPORT.md` with these substantive elements:

1. The question, model, reached stages, and outcome classification, stated first.
2. The 21/22 proof in enough detail for a reader to verify the arithmetic; source-theorem assumptions and why the obstruction does not contradict them.
3. Primary finite-set results, complete coverage/failure accounting, and the status of the secondary one-tick hypothesis.
4. Secondary comparison and legacy-file corrections only where actually executed; exact, floating, and quantized domains separated.
5. Actual device/backend/precision, elapsed computation, peak memory, disk use, development/failed-run accounting, and deviations from this brief.
6. What prior work already establishes, what this execution added, and the limits of any novelty claim.
7. What remains unresolved and **one** justified next step, if warranted. For a purely digital result, a sensible next step is a specified actuator/sample alphabet or an application deadline, not an unspecified sensor campaign. If no such next step is justified, say so.

Use evidence-backed language and ordinary source links. Include any invalidated preliminary results in the attempt record and explain their exclusion. Do not describe unexecuted commands, unaccessed papers, or target-hardware checks as completed.

### 8.4 Required artifacts by reached stage

| Stage reached | Required final artifacts |
|---|---|
| Analytic diagnostic only | Standalone standard-library checker; exact witness and separator; concise usage; verifiable derivation and scoped report; blocker/resource record where relevant |
| Candidate implementation | Above plus runnable source, dependencies/lock, tiny checks, raw candidate failures, and model documentation |
| Pilot or locked experiments | Above plus frozen configuration/manifests, raw attempts, exact certificate files, generated prefixes, resource ledger, complete result tables, and essential figures |
| Legacy/quantization diagnostics | Above plus decoded-file metadata or synthetic integer codes, their measurement conventions, and before/after representation accounting |

Provide a single convenient project archive only after inspecting its contents. Exclude the virtual environment, caches, unrelated files, and duplicate regenerated artifacts. The report and raw results must be available separately in the project. Re-run the independent certificate checker from the saved artifacts in a fresh process, regenerate the tables/figures, and verify the README commands before delivery. If the Mac was unavailable, state exactly which commands ran elsewhere and which validation remains unperformed.

## 9. Review requirements and stress tests

Before your final delivery, perform distinct mathematical, experimental, and execution reviews. A separate reviewer is useful when available, but record only review that actually occurred. Repair material defects within the shared limits, rerun only affected checks, and do not launch a new benchmark to compensate for an unfavorable conclusion.

The following dispositions are part of the design, not optional warnings:

| Stress scenario | Required response |
|---|---|
| Theorem turns out to be nonconstructive or a needed proof fails | The fold itself is explicit; the strict shorter-horizon result for nonconstant prefixes is nonquantitative and unused. If the constant lower proof fails, stop relying on it and retain only direct certificates. |
| A realistic signal violates amplitude, timing, or input assumptions | Reject/record unsupported input or explicitly model preprocessing; do not silently change the prefix. Primary synthetic inputs avoid this ambiguity. |
| Mechanism is equivalent to known work | Already accounted for by [R1], [R2], [R3], [R4], [R5], [R6], [R7], [R8], [R9]; retain the diagnostic scope and avoid an algorithmic novelty claim. |
| Only favorable synthetic cases work | The primary constants are worst-case continuous inputs, and the nonconstant set includes endpoint impulses and irregular sequences. Restrict the conclusion to completed cases; no real-device inference follows. |
| A properly formulated baseline removes a gain | The direct LP searches the full feasible set. Report its result without inventing a weaker comparator or changing the objective. |
| Preprocessing or exact checking dominates | Timing includes both; report the breakdown and apply the same total budget. |
| Finite precision changes a result | Accept only exact certificates for frontier claims; separate float/PCM residuals. Do not use a smaller numerical tolerance as proof. |
| Memory or runtime exceeds estimates | Preserve brackets and evidence; skip predeclared conditional blocks or stop, with measured resource reasons. |
| Small pilot is inconclusive | Distinguish solver uncertainty from impossibility; permit only a specific bounded repair, then retain `unknown`. |
| A paper or archive cannot be retrieved | The brief contains the essential model/fixture. Mark inaccessible source checks and omit only blocked legacy claims; do not invent contents. |
| The one-tick hypothesis is false | An exact `delta>=2` certificate is a useful negative result. Keep the original hypothesis and threshold in the report. |
| Hardware is unavailable | Complete portable exact work and report native-Mac execution as blocked, without claiming target performance. |
| Quantized playback breaks exact zeros | Report representation error; the primary rational upper witness was never a fixed-wordlength guarantee. |

### 9.1 Material planning-review findings already incorporated

Three separable reviews informed this brief: the continuous/discrete mathematics and certificates; adversarial experiment/input/measurement design; and source/novelty/executable-handoff review. Their material findings were the endpoint/PCM measurement mismatch, the different enumeration scopes, known prior art for both timing and polynomial duality, the real-amplitude versus finite-wordlength distinction, common-peak fairness, lack of instance-optimality for the fold, and the need for exact unresolved states rather than interpreting solver failures as impossibility. The final post-draft review disposition is recorded in §11.

Residual uncertainties are explicit: historical originality of the exact source theorem or particular counterexample, the untouched finite frontier, robust certificate recovery on the actual Mac, fixed-wordlength minimum duration, and application-level utility. None is to be silently filled by an assertion of superiority.

## 10. Supporting research and references

### 10.1 Search scope and stopping decision

Research was conducted on **4 November 2026** using two web search engines and primary author/publisher sources. Relevant source content was opened, not inferred solely from search snippets. Search terms and equivalent formulations included bounded moment completion, Chebyshev folds, Uhrig timing, Hobby–Rice sign selection, polynomial concentration, chains of integrators, discrete minimum-time control, constrained-coefficient polynomials, spectral-null sequences, Prouhet–Thue–Morse, and exact rational LP verification.

Backward references from the supplied note led to [R1], [R2], [R3], [R4]. Searching discrete spectral nulls led to [R8], [R9]; searching constrained polynomial coefficients led to [R5] and its later literature. Exact-LP research supplied [R6], and discrete reachability supplied [R7]. These findings materially reduced the novelty claim. Further unrelated entropy/graph literature expansion would not change the selected diagnostic, so it was not pursued.

Some leads were inaccessible in full, including the original Hobby–Rice paper and the full 2003 Zanasi–Morselli discrete-three-integrator article. The latter's publisher metadata/introductory material was available, but its detailed results were not verified and are not relied on here. The selected protocol does not depend on either inaccessible text. Author-hosted PDFs can differ in pagination from final journal versions; precise locations below explicitly identify PDF pages when necessary. This was a targeted opportunity assessment, not an exhaustive global novelty search or a patent search.

### 10.2 References and precise relevance

**[A1] User-supplied research note.** *Sharp Bounded Moment Completion by Signed Chebyshev Folding*, 4 November 2026. File `moment_completion_theorems.txt` in `counterrandomness_research(1).zip`. Essential locations: §1 Theorem A; §§2–3 construction/cancellation; §4 dual sharpness and uniqueness; §§5–6 equality and nonconstant-prefix slack; §7 Abel-only extension; §9 reported experiments; §10 prior-art caveat. No public publication URL or named author is supplied. Use the archive hash in §2.1.

**[A2] User-supplied executable.** *Counterrandom Signal Lab — macOS / Apple Silicon PoC*, undated README within the supplied November 2026 materials. `Counterrandom_Signal_Lab_Mac_M3_Max(1).zip`, directory `counterrandom_mac_poc/`. Critical locations in `counterrandom_poc.py`: `build_completion` lines 26–67; `normalized_moments` lines 70–75; `sample_segments` lines 78–85; `export_wav` lines 134–138; main measurements lines 196–236, especially 203–207 versus 233–234. `high_rate/report.json` is a reported example, not a universal result or a decoded-WAV measurement.

**[R1] Götz S. Uhrig (2007).** “Keeping a Quantum Bit Alive by Optimized π-Pulse Sequences.” *Physical Review Letters* **98**, 100504. [DOI](https://doi.org/10.1103/PhysRevLett.98.100504); [primary preprint](https://arxiv.org/pdf/quant-ph/0609203). Inspected location: preprint p.3, Eq. (12), and the preceding derivative-cancellation conditions. Establishes the sine-squared timing sequence; does not establish this prescribed-prefix digital theorem.

**[R2] Wen Yang and Ren-Bao Liu (2008).** “Universality of Uhrig dynamical decoupling for suppressing qubit pure dephasing and relaxation.” *Physical Review Letters* **101**, 180403. [DOI](https://doi.org/10.1103/PhysRevLett.101.180403); [primary preprint](https://arxiv.org/pdf/0807.4058). Inspected locations: Eq. (1); Eq. (20), PDF p.3; symmetry discussion pp.3–4. General transformed-time modulation is established; the stronger nonlinear quantum conditions differ from the scalar constraints here.

**[R3] N. Cody Jones, Thaddeus D. Ladd, and Bryan H. Fong (2012).** “Dynamical decoupling of a qubit with always-on control fields.” *New Journal of Physics* **14**, 093045. [DOI](https://doi.org/10.1088/1367-2630/14/9/093045); [primary preprint](https://arxiv.org/pdf/1205.2402). Inspected locations: §§2.2–2.3 and Appendix A, especially Eq. (A.5), PDF p.19, reducing conditions to polynomial-moment annihilation. Supplies close Chebyshev/Fourier prior art; no physical performance result is transferred into this study.

**[R4] Götz S. Uhrig and Daniel A. Lidar (2010).** “Rigorous Bounds for Optimal Dynamical Decoupling.” *Physical Review A* **82**, 012301. [DOI](https://doi.org/10.1103/PhysRevA.82.012301); [primary preprint](https://arxiv.org/pdf/1002.1484). Inspected location: §VI, Eqs. (36a,b) and (37), PDF p.5. Gives `T=t_1*csc^2(pi/(2N+2))` and the quadratic asymptotic with `+1/3`, under a minimum-pulse-interval interpretation. This limits novelty claims for the formula while leaving the exact prescribed-prefix formulation distinct in scope.

**[R5] Peter Borwein, Tamás Erdélyi, and Géza Kós (2013).** “The multiplicity of the zero at 1 of polynomials with constrained coefficients.” *Acta Arithmetica* **159**(4), 387–395. [DOI](https://doi.org/10.4064/aa159-4-7); [author PDF](https://people.tamu.edu/~terdelyi/papers-online/zeros12_rev.pdf). Inspected locations in that PDF: Theorem 2.1 p.2; Lemma 3.4 p.3; Lemma 3.5 p.4 and its proof p.7. Connects root multiplicity to polynomial annihilation and gives bounded-coefficient dual obstructions. Particularly close to the one-prescribed-coefficient ancestor of this discrete problem; the technique itself is established.

**[R6] David L. Applegate, William Cook, Sanjeeb Dash, and Daniel G. Espinoza (2007).** “Exact solutions to linear programming problems.” *Operations Research Letters* **35**(6), 693–699. [DOI](https://doi.org/10.1016/j.orl.2006.12.010); [author PDF](https://www.math.uwaterloo.ca/~bico/papers/exact_simplex.pdf). Inspected locations: §3, author-PDF pp.2–3, Algorithm 1 and rational verification discussion. Motivates floating-point candidate discovery followed by exact tests. The brief uses a tiny specialized certificate checker, not that paper's complete solver.

**[R7] Laurent Bako, Dulin Chen, and Stéphane Lecoeuche (2011).** “A numerical solution to the minimum-time control problem for linear discrete-time systems.” arXiv:1109.3772v2, 11 October 2011. [Stable record](https://arxiv.org/abs/1109.3772); [primary PDF](https://arxiv.org/pdf/1109.3772). Inspected locations: §2, Problem 1 and Eqs. (1)–(4), PDF pp.3–4. Establishes the bounded discrete-reachability/minimum-time context. Its sparse relaxation is unnecessary for the low-row-count exact-feasibility search here.

**[R8] Ali Pezeshki, A. Robert Calderbank, William Moran, and Stephen D. Howard (2008).** “Doppler Resilient Golay Complementary Waveforms.” *IEEE Transactions on Information Theory* **54**(9), 4254–4266. [DOI](https://doi.org/10.1109/TIT.2008.928292); [primary published PDF](https://www.ese.wustl.edu/~nehorai/MURI/publications/IT08_Pezeshki.pdf). Inspected location: §III-C, Theorem 1 (Prouhet), p.4258, including visual inspection of the equations. The classic power-sum identity supports the signed-copy control; the adaptation to a preserved arbitrary prefix is derived directly in (14).

**[R9] Wenbing Dang, Ali Pezeshki, Stephen D. Howard, William Moran, and Robert Calderbank (2020 preprint).** “Coordinating Complementary Waveforms for Suppressing Range Sidelobes in a Doppler Band.” arXiv:2001.09397, 26 January 2020; manuscript records submission in August 2019. [Stable record](https://arxiv.org/abs/2001.09397); [primary PDF](https://arxiv.org/pdf/2001.09397). Inspected locations: §IV-C, Remark 10, Eqs. (35)–(39), PDF p.6; §V objective formulations, PDF p.7. Establishes integer-Vandermonde spectral-null formulations and coefficient-performance tradeoffs. Its radar SNR problem is not the benchmark objective here.

**[R10] SciPy contributors.** `scipy.optimize.linprog` and `linprog(method='highs-ds')`, official documentation accessed 4 November 2026. [HiGHS dual simplex documentation](https://docs.scipy.org/doc/scipy/reference/optimize.linprog-highs-ds.html); [general interface](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html). Inspected sections: bounds, statuses, equality marginals, presolve, time limits, feasibility tolerances. The online pages reported SciPy 1.18.0; record the actual installed version rather than assuming it matches the current website.

**[R11] SciPy contributors.** `scipy.io.wavfile.read`, official documentation accessed 4 November 2026. [Documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.wavfile.read.html). Use actual integer dtype and sample rate in decoded-file auditing. This study needs only the supplied signed-16-bit files, not a general audio ingestion library.

**[R12] Python Software Foundation.** `resource` — Resource usage information, official documentation accessed 4 November 2026. [Documentation](https://docs.python.org/3/library/resource.html). Platform-specific resource interfaces require checking availability and units; the document does not substitute for actual Mac measurement.

[R1]: https://doi.org/10.1103/PhysRevLett.98.100504
[R2]: https://doi.org/10.1103/PhysRevLett.101.180403
[R3]: https://doi.org/10.1088/1367-2630/14/9/093045
[R4]: https://doi.org/10.1103/PhysRevA.82.012301
[R5]: https://doi.org/10.4064/aa159-4-7
[R6]: https://doi.org/10.1016/j.orl.2006.12.010
[R7]: https://arxiv.org/abs/1109.3772
[R8]: https://doi.org/10.1109/TIT.2008.928292
[R9]: https://arxiv.org/abs/2001.09397
[R10]: https://docs.scipy.org/doc/scipy/reference/optimize.linprog-highs-ds.html
[R11]: https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.wavfile.read.html
[R12]: https://docs.python.org/3/library/resource.html

## 11. Planning work actually performed and final review disposition

### 11.1 Performed before this handoff

The planning team read all supplied text notes, READMEs, scripts, and result JSONs, inspected the supplied plots, and decoded the supplied demonstration WAVs. Separate agents reviewed the moment mathematics, non-moment claims/experimental design, and literature/novelty. The lead reconciled their findings and independently checked the decisive exact arithmetic. This was planning work; the repository/CLI specified in §7 and its Mac evaluation have not been executed.

The following small probes were actually performed in Linux x86_64:

| Probe | Result | Evidentiary scope |
|---|---|---|
| Integer/Fraction verification of `r=4,m=2` | Dual gap 9 at 21; exact feasible 22-vector with all four ordinary held moments zero | Exact counterexample and discrete optimum in the real-amplitude model |
| Independent check of `r=4,m=36` dual | Exact gap 3060 at 377 | Exact second rounding obstruction |
| Exploratory constant sweep `r=3,4`, all `m=1,...,64` | 128 exact primal/minimality certificates; extra tick only at `r=4,m=2,36` | Development finite-set result; lead independently rechecked every archived witness and lower certificate; no held-out/Mac claim |
| Center-set scope arithmetic | Difference is 89 singleton sets | Reconciles source counts; no re-execution of the large entropy enumeration is claimed |
| High-rate WAV readback | Actual duration `502663/48000 = 10.472145833333334 s`; integer prefix preserved | Audits the provided file, not a new device experiment |

For the high-rate file, the source's reported sampling maximum is `1.9949490337563963e-5`. Reintegrating its decoded PCM over the actual uniform cells, normalized by the **actual file duration**, gave a maximum approximately `1.950542805569458e-5`. Using the source's **ideal duration** for normalization while still integrating the actual cells gives approximately `1.9505446457823013e-5`. These are intentionally different conventions; your audit should label them rather than “correcting” one into the other. The decoded digital energy ratio is approximately `9.384674206830804`, while the source's ideal-segment ratio is `9.383834628261521`.

For the lower-rate `example/`, the actual duration is `54627/8000 = 6.828375 s`, compared with the reported ideal `6.82842712474619 s`. Its decoded-WAV maximum using actual-duration normalization was approximately `4.7740090955039616e-5`, compared with the source's floating sampled value `4.340876209314032e-5`; decoded energy ratio was approximately `6.242871312588519`. These floating numbers are audit observations, not acceptance or tuning targets.

The 48 kHz waveform was not used in a large LP or an M3 Max performance benchmark. The original full entropy and graph exhaustive suites were inspected, not rerun. No hardware playback, actual sensor experiment, fixed-wordlength optimum, or final confirmatory study was performed during planning.

### 11.2 Final review record

Three independent post-draft passes were completed, and affected sections were rereviewed after material repairs. No held-out or scaling cases were run during those reviews.

| Review pass | Material findings | Disposition in the delivered brief |
|---|---|---|
| Mathematical and novelty | A valid dual can also apply at shorter horizons; a continuous ceiling plus an arbitrary feasible upper does not prove adjacency; changing search bounds could make an initial scan ambiguous; unknown horizons could accidentally regain a retry allowance | The negative fixture uses `n=22` explicitly; minimality requires adjacent certified bounds; the initial `n_0` is frozen; failed discovery allowances persist across searches/resumes. Continuous proof indexing, exact ceilings, PTM, dual conversion, quantization scope, and close prior-art attribution were rechecked. |
| Experimental | Guard sufficiency and exact-minimum recovery require different evidence; calibration reused a deterministic secondary input; zero-only metric controls could miss a broken evaluator; amplitude metadata and core/scaling coverage needed independent checks | Separate minimum/guard fields and denominators; development-seeded calibration inputs; a nonzero analytic residual oracle; independent recomputation of the peak; complete unknown/failure accounting. The affected design was rereviewed and the identified issues were resolved. |
| Execution | Analytic verification originally followed dependency installation; multiple output directories could reset budgets; pilot repair accounting and the third LP choice were ambiguous | Dependency-free analytic-first commands; one canonical ledger; a pilot ceiling including repairs; a deterministic bounded repair formulation. Exact rounding of stored float samples and the timing-repeat exception were clarified and rereviewed. |

The final checks confirmed additive budgets, consistent declared case sets, an independently usable input fallback, explicit unverified target dependencies, and no claim of completed M3 Max validation. The reviewers found no remaining known material defect in the sections they reviewed. This is a bounded review conclusion, not a guarantee of absolute correctness; implementation, actual machine behavior, untouched cases, and historical originality remain to be established or scoped as stated above.

## 12. Completion contract

Your job is to execute the justified stages, inspect their evidence, obey the stop/repair rules, and deliver an honest result. Success does not require a positive algorithmic advantage. It requires that each retained conclusion be traceable to a source, exact derivation, verified certificate, or actual measurement under the declared model.

Before final delivery, verify all linked local artifacts exist, the checker accepts every published certificate, the report matches the raw ledger and frozen manifest, skipped/failed work remains visible, and no instructions or result cells contain unresolved placeholders. Report exactly what was learned and what remains outside the tested scope.


