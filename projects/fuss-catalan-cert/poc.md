# PoC execution brief: certified real-shape comparison for Fuss–Catalan laws

**Executor:** an autonomous coding agent configured as `gpt-6.1-sol` with `xhigh` reasoning.  
**Research and planning date:** 31 October 2026.  
**Target:** one locally available MacBook with an M3 Max; inspect its actual configuration.  
**Deliverable of this instruction:** execute the bounded research study below and report the evidence, including an early stop if justified. This brief is a research plan, not a claim that the PoC or Mac benchmark has already run.

## 1. Project decision and smallest defensible contribution

Build an **exact certificate and falsification tool for the noninteger-shape-gap extension of the supplied Fuss–Catalan stochastic-comparison theorem**. Preserve the original mathematical purpose: decide when subtracting a deterministic shift from a larger convolution power makes it stochastically smaller, and therefore determine a sharp bounded-displacement transport distance. The original note already resolves its motivating centered-sum examples; repeating those examples is a correctness control, not the new contribution.

The supplied note proves the sharp criterion for every real pair of shapes when the branching degree is two, and for integer shape differences at higher branching degrees. It explicitly leaves arbitrary real differences at degree at least three open [A1, Theorem 3.1; Appendix A.1–A.4]. The selected study addresses that gap with three parts of one mechanism:

1. Independently replay and document the short **continuum extension for degree three and smaller shape at most five** derived during planning in §4. This is a specific extension of the supplied manuscript; historical originality remains provisional.
2. Implement the finite, exact rational-instance decision procedure in §5. Its correctness does not assume the general real-gap conjecture. It combines a coefficient-sign certificate, an exact polynomial reference, and a finite CDF fallback whose unexamined suffix is proved safe.
3. Evaluate the locked 144-case matrix in §7, including larger shapes and degrees outside the proved subdomain. Seek a certified counterexample to the proposed sharp threshold, or certify its validity for each completed case. Retain the larger-shape cases even if the continuum result succeeds.

The useful outcome is a verifiable extension, a counterexample, or a precisely scoped computational limitation. **Do not claim a universal real-shape theorem from a finite collection of certificates. Do not claim a new probability distribution, a new general single-crossing method, or a generic transport speedup.**

### 1.1 Primary question, hypothesis, and decision

For each predeclared rational parameter tuple outside the manuscript's integer-gap restriction, is the smallest tail-admissible integer shift also sufficient for stochastic order at **every** threshold?

With definitions in §3, let

\[
b_{\rm tail}=\min\{b\in\mathbb Z_{\ge0}:L_d(s,t,p,b)\le1\}.
\]

The primary hypothesis is

\[
N_t^{(d)}-b_{\rm tail}\le_{\rm st}N_s^{(d)}.
\tag{H}
\]

The predicted mechanism is that the shifted mass ratio has one trough, so its necessary limiting-tail bound controls every tail. The coefficient-sign shortcut is a way to prove this mechanism; it is not assumed to be necessary. A finite-prefix certificate can establish (H) even if this shape condition fails.

**Main quantities:** the number of exact counterexamples to (H), the number of independently verified all-threshold decisions, and unresolved cases by reason. A single verified finite CDF violation with `L <= 1` refutes the unrestricted conjecture. Zero counterexamples together with successful certificates for every active matrix case supports the formula on that finite set only. The separately verified §4 derivation supports its stated continuum domain. There is no approximate accuracy threshold: the meaningful mathematical difference is one wrong stochastic-order decision, or one count of error in the proposed minimal shift.

**Practical target, fixed before benchmarking:** complete the full 144-case study and controls within the 4-hour cumulative experimental wall-time cap and the memory ceiling in §8. No minimum percentage speedup is required. Runtime and the fraction handled by the cheap certificate explain feasibility; they are secondary to correctness. Do not tune a practical effect threshold to observed timings.

### 1.2 Scope and exclusions

Use integer branching degrees, a shared subcritical branching probability, positive shapes, and integer shifts. Exact computational inputs are rational numbers; §4 is an analytic real-parameter statement. The finite matrix is designed to expose small-shape, large-shape, small-gap, large-gap, and near-critical behavior. It is not a sample of a real-world population.

Exclude parameter fitting, proprietary data, application-specific risk claims, direct optimal-coupling samplers, GPU acceleration, generalized-Poisson or inverse-Gaussian benchmark projects, critical/supercritical branching, noninteger branching degree, and a search for an arbitrary new convolution family. Do not silently broaden or substitute these problems. These exclusions retain the supplied mechanism while keeping the scientific question identifiable and executable locally.

## 2. Inputs, source audit, and evidence status

### 2.1 Files to receive

The execution agent should receive this brief and these original attachments. Put the originals in `inputs/`; extract the ZIP into `inputs/package/` without modifying it.

| Attachment | Required use | SHA-256 of bytes reviewed during planning |
|---|---|---|
| `catalan_transport_note(1).pdf` | Fourteen-page mathematical note, including Appendix A | `c9446d405a533fd8880632a2b6601c5275124d7ec68d15a61e2ea196f4ce2f22` |
| `catalan_research_reproducibility(1).zip` | TeX sources, verification scripts, recorded outputs, figures | `16274c24f4585e9fdb161aaf04b389e12bcfa9251f4ca03045ca71ce458a7069` |

The ZIP contains `README.txt`, `catalan_transport_note.tex`, `fuss_poisson_appendix.tex`, `verify_catalan.py`, `verify_extensions.py`, `verification_results.json`, `extension_results.json`, `verification_summary.txt`, `catalan_tail_scan.csv`, `make_figures.py`, and the two illustrations in PDF/PNG form. There are no bundled external research articles. The pasted task instructions are not a computational dependency; this brief incorporates the necessary requirements.

If the attachments are unavailable, the equations, controls, and algorithms below suffice for independent implementation and the new diagnostic. Mark attachment-reproduction work blocked and request only the missing ZIP/PDF if that block matters. Never substitute an unrelated paper. A hash mismatch requires comparing the actual version and recording the difference; it is not permission to assume the mathematics is unchanged.

### 2.2 What the supplied mathematics supports

The planning review read both complete TeX files, the PDF, all supplied scripts and result files, and the figures. Independent mathematical and computational reviewers were used. No material defect was identified in the proofs listed below. This is a research audit, not peer review or proof-assistant verification.

| Claim | Precise source and assumptions | Evidence status and use |
|---|---|---|
| Positive-shape Catalan convolution laws | [A1, §2, equations (1)–(4), pp. 1–2]; `0 < r < 1/4`, real shape `u > 0` | Analytic generating-function/Lagrange-inversion construction. Classical family, not new. |
| Sharp shifted Catalan comparison | [A1, Theorem 3.1 and Lemma 3.2, pp. 2–4, equations (5)–(7)]; real `t > s > 0`, integer `b >= 0` | Quadratic sign proof plus one-crossing tail argument; includes equality. Valid positive control. |
| Exact Catalan extended `W_infinity` | [A1, Corollary 4.1, p. 4, equation (8)] | Uses stochastic monotonicity and common quantiles, not an independent-sample coupling. |
| Centered sums, activation, catalysis | [A1, §5, pp. 4–6] | Exact consequences for mixed-sign variables; no reversal of ordinary arithmetic or nonnegative constants. |
| Inverse Gaussian analogue and diffusion agreement | [A1, §6, pp. 6–7] | Analytic companion; convergence of pairwise distances does not assert `W_infinity` convergence of each discrete law to its continuous limit. Background only. |
| Fuss–Catalan sharp formula | [A1, Theorem A.1, Lemma A.2, pp. 10–12, equations (12)–(17)] | Requires positive **integer** `t-s`, though the individual shapes may be real. Finite-product proof cannot be reused for a fractional gap. |
| Generalized Poisson formula | [A1, Theorem A.3, pp. 12–13, equations (18)–(21)] | Strict comparison follows by a Fuss limit; boundary by continuity; necessity by asymptotics. A separate proved family, not the selected experiment. |
| Arbitrary real Fuss gap at `d >= 3` | [A1, Appendix A.4, p. 14] | Explicit open question in the supplied source. §4 resolves only one subdomain; the broader assertion remains a hypothesis. |

### 2.3 What was actually computed during planning

These are **already performed planning checks**, not future Mac results:

- Both original verification scripts ran successfully in a hosted Linux x86_64 environment using Python 3.12.14. Core verification took about 4.35 seconds and extension verification about 0.78 seconds. Their two JSON outputs and the tail-scan CSV were byte-identical to the supplied outputs. These timings are provenance only and must not be presented as M3 Max performance.
- The core program checked 86,240 integer and 1,785 rational instances of the Catalan quadratic identities, 1,519 exact coefficient convolutions, and 3,001 exact tail thresholds. The extension program checked 3,024 exact sorted-grid instances, all with integer shape gap. Its generalized-Poisson and inverse-Gaussian checks are ordinary floating-point illustrations, not rigorous infinite-tail certificates [A2, scripts and recorded outputs].
- Three exploratory exact-polynomial checks were made: 1,200 seeded cases by one reviewer, 3,000 by another, and four explicitly reconstructed examples by the lead. Every tested polynomial had at most one nonzero-coefficient sign variation. These pools were exploratory and are not the locked evaluation. A separate 4,000-case floating derivative probe found no robust shape reversal; cancellation-sensitive signs were excluded, so it has no proof status.
- The degree-three identities in §4 were independently expanded with exact integer polynomial arithmetic by separate implementations. An additional 251-term transformed positivity certificate was checked but is unnecessary for execution: the shorter derivation below is the required evidence.
- The exact fractional-gap equality fixture in §6.2 was checked using rational arithmetic, including both neighboring probabilities. No full new solver, locked comparison, or target-device benchmark was implemented during planning.

The original figures use finite numerical tails and evaluations of derived formulas. They illustrate the source; they are not independent empirical proof. In particular, a plot made by evaluating the proposed formula is not a validation of that formula.

### 2.4 Closest prior art and contribution boundary

Jain–Consul introduced the relevant generalized negative binomial family [R1]. Its original publisher abstract was accessible; its full article was not. Faroughi–Li–Ren provide an accessible matching mass formula and the generalized-Poisson limit [R2, §1]. Match equations when searching: other distributions also use the name “generalized negative binomial.”

Klenke–Mattner's half-monotone likelihood-ratio criterion is the closest general method [R3, author manuscript §2.1, Definition 2.1 and Proposition 2.3]. The published article renumbers that proposition as 2.1. Their method is prior art; so are exact polynomial root isolation and Descartes' rule. Belzunce–Martínez-Riquelme–Pereda apply related shape methods to other discrete orders, including unshifted generalized Poisson laws [R4, Theorem 1 and Example 3]. Neither inspected result states the required noninteger-gap shifted Fuss formula.

For the analytic subdomain, Młotkowski's nonnegative-support moment representation implies coefficient log-convexity for `0 < s <= d` [R5, Theorem 5.1, pp. 948–949; the implication follows by Cauchy–Schwarz]. At `d=3`, this covers `s <= 3`, not the entire new interval `3 < s <= 5`. The same paper records the necessary first-minor condition `s <= 2d-1`; that condition alone is not a proof of all log-convexity inequalities. Nearby gamma-quotient results were checked but did not directly cover the whole two-shape shifted problem [R6–R8].

The smallest defensible contribution is therefore **an independently verified extension of the supplied note on a specified continuum domain, plus exact certificates or counterexamples in the remaining rational-input regime**. The finite decision construction below is a transparent specialization of standard algebraic methods; do not market the machinery itself as an unprecedented algorithm. If equivalent prior work is found, credit it and frame the result as reproducibility, an independently checked derivation, or a limitation study. Finding prior art does not invalidate the calculation, but it can remove a novelty claim.

## 3. Mathematical specification and constructive bridge

### 3.1 Laws, dimensions, and conventions

All quantities here are scalar. For integer `d >= 2`, real `u > 0`, and integer `k >= 0`, set

\[
A_{d,u}(k)=\frac{u\Gamma(dk+u)}{\Gamma(k+1)\Gamma((d-1)k+u+1)},\qquad A_{d,u}(0)=1.
\tag{1}
\]

Choose `0 < p < 1/d` and define

\[
\alpha=1-p,\quad r=p\alpha^{d-1},\quad
\rho_d=\frac{(d-1)^{d-1}}{d^d},\quad
\tau_d=\frac d{d-1},\quad c_d=\alpha^{-1},
\]
\[
\Pr(N_u^{(d)}=k)=A_{d,u}(k)r^k\alpha^u.
\tag{2}
\]

The generating function is `[C_d(rz)/C_d(r)]^u`, where `C_d=1+z C_d^d` is the branch with `C_d(0)=1`. Lagrange inversion gives (1) and normalization. Convolution adds the shape parameters, so `N_s <=st N_t` for `s < t`. These facts hold for all positive real shapes, with no need to construct a forest with a fractional number of trees [A1, Appendix A.1].

**Constructive simplification:** input `p`, not `r`. Then `c_d=1/(1-p)` exactly; there is no algebraic-root selection or numerical generating-function oracle. For rational `u` and integer `k`, the coefficient can also be computed directly as

\[
A_{d,u}(k)=\frac{u}{k!}\prod_{j=1}^{k-1}((d-1)k+u+j),\qquad k\ge1.
\tag{3}
\]

Use (3) as an independent tiny oracle, not the main large-index implementation.

Usual stochastic order means `P(U >= x) <= P(V >= x)` for every real `x`. For the present integer-valued laws, examine integer thresholds. The desired comparison is equivalent to

\[
\Pr(N_t\ge k+b)\le\Pr(N_s\ge k),\quad k=0,1,2,\ldots.
\tag{4}
\]

Negative thresholds impose no extra constraint because the comparator is nonnegative. The original independent increments establish stochastic monotonicity; the bounded optimal coupling is generally dependent.

### 3.2 Necessary tail constant and exact transport consequence

Write `delta=t-s > 0` and

\[
H=\tau_d/c_d=\frac{d(1-p)}{d-1}>1,\qquad
q=r/\rho_d\in(0,1),
\]
\[
L_d=\frac ts H^{\delta}q^b.
\tag{5}
\]

Direct Stirling expansion, valid without integer-gap assumptions, gives

\[
A_{d,u}(k)\sim
\frac{u\tau_d^u}{\sqrt{2\pi d(d-1)}}\rho_d^{-k}k^{-3/2}.
\tag{6}
\]

Thus the ratio `R_k=P(N_t=k+b)/P(N_s=k)` tends to `L_d`. If `L_d > 1`, all sufficiently far-right mass differences are positive, and summing them proves that (4) fails. This is an exact analytic rejection even if a concrete remote threshold is not enumerated.

If sufficiency at `b_tail` is certified, then

\[
W_\infty(\mathcal L(N_t),\mathcal L(N_s))=b_{\rm tail}
=\left\lceil\frac{\log(t/s)+\delta\log H}{-\log q}\right\rceil.
\tag{7}
\]

Reason: common quantiles realize `0 <= Q_t(U)-Q_s(U) <= b`; any coupling with displacement at most `b` implies the shifted stochastic order. Integer-valued differences make the minimal bound an integer [A1, Corollary 4.1 and Appendix A.1]. The logarithmic expression is for interpretation only; compute its integer value by exact comparisons below. Do not estimate `W_infinity` from finitely many sampled quantiles.

### 3.3 Exact recurrence polynomial

For real `x >= 0`, define positive polynomials

\[
P_u(x)=\prod_{j=0}^{d-1}(dx+u+j),\qquad
Q_u(x)=(x+1)\prod_{j=1}^{d-1}((d-1)x+u+j).
\]

The coefficient recurrence is

\[
M_u(k):=\frac{A_{d,u}(k+1)}{A_{d,u}(k)}=\frac{P_u(k)}{Q_u(k)}.
\tag{8}
\]

For `h_k=A_{d,t}(k+b)/A_{d,s}(k)`, the sign of `h_(k+1)-h_k` is exactly the sign of

\[
D(x)=P_t(x+b)Q_s(x)-Q_t(x+b)P_s(x).
\tag{9}
\]

For rational `s,t`, construct this polynomial over the rationals. The coefficients of degrees `2d` and `2d-1` cancel. Its exact degree is `n=2d-2`, and its leading coefficient is

\[
a_n=d^d(d-1)^{d-1}
\left[\frac{3b}{2}+\frac{\delta(t+s+2d-1)}{2d(d-1)}\right]>0.
\tag{10}
\]

Verify these identities at the constructive gate. One derivation expands

\[
\log A_{d,u}(k)=-k\log\rho_d-\tfrac32\log k+\text{constant}
+\frac{c_1(u)}k+O(k^{-2}),
\]
\[
c_1(u)=-\frac{u(u+2d-1)}{2d(d-1)}-
\frac{d^2-d+1}{12d(d-1)}.
\]

The shifted ratio approaches its limit from below with first-order correction
`-3b/2-delta(t+s+2d-1)/(2d(d-1))`; (10) follows, or can be obtained by expanding the leading product coefficients directly. A floating asymptotic approximation must never decide a finite coefficient sign.

### 3.4 What survives implementation

The exact algorithm retains infinite-support correctness because it verifies a mathematical suffix certificate, not because the omitted mass is small. It has no Monte Carlo error or convergence tolerance. Its trust assumptions are correct rational arithmetic, correct polynomial identities and sign verification, and the stated analytic asymptotics/normalization. Resource limits may prevent a decision; they cannot change the order relation.

The rational-input restriction makes all operational comparisons finite algebraic calculations. Approximating an irrational input by a nearby rational is not a certificate for the original input, especially at `L=1`. No guarantee of small computational cost follows from finite decidability: coefficient growth, exact powers, root separation, and a long prefix can dominate.

## 4. Analytic subdomain to verify before benchmarking

**Planning-stage derived result.** For `d=3`, `0 < s <= 5`, every real `t > s`, every integer `b >= 0`, and `0 < p < 1/3`,

\[
N_t^{(3)}-b\le_{\rm st}N_s^{(3)}\quad\Longleftrightarrow\quad L_3\le1.
\tag{11}
\]

Consequently (7) holds when the smaller of the two positive shapes is at most five. The proof below was developed and algebraically checked during planning; your first scientific task is to reproduce the identities independently and write a readable proof. Do not describe it as a theorem from the attachments.

For degree three,

\[
M_u(k)=\frac{(3k+u)(3k+u+1)(3k+u+2)}
{(k+1)(2k+u+1)(2k+u+2)}.
\]

Let `Bbar=(2k+u+1)(2k+u+2)`. Its derivative with respect to shape satisfies

\[
\partial_uM_u(k)=\frac{J(k,u)}{(k+1)\overline B^2}>0,
\]
\[
\begin{aligned}
J(k,u)={}&(u+1)^2(u+2)^2\\
&+(30+60u+39u^2+8u^3)k\\
&+(65+78u+21u^2)k^2+(45+18u)k^3.
\end{aligned}
\tag{12}
\]

Every coefficient is positive. Next, the numerator of `M_s(k+1)-M_s(k)`, over the positive product `Q_s(k+1)Q_s(k)`, is `V(k,s)=sum_(i=0)^4 c_i(s) k^i`, where

\[
\begin{aligned}
c_4&=162,\\
c_3&=708+234s-18s^2,\\
c_2&=1056+771s+72s^2-21s^3,\\
c_1&=618+767s+230s^2-11s^3-8s^4,\\
c_0&=(s+1)(s+2)(s+3)(s+4)(5-s).
\end{aligned}
\tag{13}
\]

For `0 < s <= 5`,

\[
c_3\ge708+144s,\quad c_2\ge1056+606s,\quad
c_1\ge618+642s,\quad c_0\ge0.
\]

For example, use `s^3 <= 5s^2`, `s^4 <= 25s^2`, and then `s^2 <= 5s` to get the stated lower bound for `c_1`. Thus `M_s(k)` is nondecreasing in integer `k`. Combining this with strict shape monotonicity,

\[
\frac{h_{k+1}}{h_k}=\frac{M_t(k+b)}{M_s(k)}>
\frac{M_s(k+b)}{M_s(k)}\ge1.
\]

Therefore `R_k` increases to `L_3`; if `L_3 <= 1`, every common-support mass difference is nonpositive and every required tail is nonpositive. Equation (6) proves necessity; quantiles prove the distance statement.

The endpoint five is sharp for **uniform global increase of the coefficient ratio**, not necessarily for stochastic order. At `b=1`,

\[
h_1/h_0=(t+5)/(2s)<1
\quad\text{if }s>5\text{ and }0<t-s<s-5.
\tag{14}
\]

This supplies an essential adverse control. A ratio can decrease initially and still have one trough. Never call (14) a counterexample to (H). The study's larger-shape cases test precisely the regime where this easy proof is insufficient.

## 5. Exact algorithm, failure behavior, and certificates

### 5.1 Numerical conventions and exact tail threshold

Parse every rational from an integer or a string such as `"13/2"`. Reject binary floating input to the exact interface. Reduce fractions, require `d >= 2`, `0 < s < t`, `0 < p < 1/d`, and integer `b >= 0`. Use arbitrary-size integers and exact rational arithmetic on CPU. The display layer may produce decimals, but no displayed decimal authorizes a sign decision.

Write `delta=A/B` in lowest terms with positive integers `A,B`. Then

\[
L_d\le1\iff (t/s)^B H^A q^{bB}\le1.
\tag{15}
\]

All bases are positive rational numbers. Cross-multiply positive integer numerators and denominators, using size guards before exponentiation. This also handles exact equality. The exponent `A` is the numerator of the **gap**, not the numerator of `t`.

Find `b_tail` by exact exponential bracketing and integer bisection: start with `lo=0` (known inadmissible) and `hi=1`; double `hi` while (15) is false; then bisect until `hi=lo+1`. Keep the exact proof of both endpoint decisions. A float logarithmic estimate is optional only as a hint; the initial implementation should omit it. If a cap is reached, return an unresolved resource status rather than a rounded answer.

### 5.2 Cheap shape certificate

Clear positive denominators in `D`, divide by the positive gcd of integer coefficients, and retain the positive leading sign. Store coefficients in ascending powers and record that convention. Strip trailing zero high-degree coefficients and handle initial zero coefficients explicitly.

Count sign variations after removing zero coefficients. By Descartes' rule:

- Zero variations and positive leading coefficient imply `D(x) >= 0` for `x >= 0`.
- One variation implies one positive root counted with multiplicity. The polynomial is negative before that root and positive after it, apart from a possible zero at the origin. Hence `h_k` is nonincreasing then nondecreasing.

If `L <= 1`, either pattern permits the source's single-crossing argument. To make that bridge explicit, put `d_k=P(N_t=k+b)-P(N_s=k)` and `T_k=sum_(j>=k)d_j`. The signs of `d_k` can change only from positive to nonpositive, while `T_0=P(N_t>=b)-1 <= 0`. Before the sign change, `T_(k+1)=T_k-d_k <= T_k`; afterwards it is a sum of nonpositive terms. Thus every `T_k <= 0`, including the boundary `L=1` [A1, §3.2; R3].

More than one variation means **shortcut unresolved**. Do not infer either multiple positive roots or failure of stochastic order from it.

### 5.3 Strong exact reference and ablation

Implement a root-based route using a polynomial explicitly over `QQ` or `ZZ`: exact real-root isolation, multiplicities, and rational sign evaluation between isolated roots. SymPy's documented `Poly.intervals`, `count_roots`, and Sturm functionality are suitable [R9]. `roots()` may return only roots expressible in radicals; `nroots()` is approximate. Neither is a complete exact substitute.

Construct a sign chart on `[0,infinity)`, treating a root at zero, rational roots, and even multiplicities explicitly. Rational interval endpoints must not be roots unless the root is separately represented exactly. Refine intervals only enough to order them and determine their side of zero. Certify a single trough whenever the chart has no positive-to-negative transition. Multiple roots can still be compatible with the shape; do not reject merely because a root count exceeds one.

This route is both the strongest feasible reference and the **ablation with the coefficient-sign shortcut disabled**. It receives the same exact tail precheck, parameters, accuracy requirement, prefix fallback, and limits. Do not run it on cases already rejected by `L>1` merely to inflate baseline cost.

### 5.4 Complete finite-prefix fallback

If neither shape route certifies the order, the following finite algorithm remains valid. Let `D(x)=a_n x^n+...+a_0`, `a_n>0`. Find an integer `K>=1` such that

\[
a_nK^n>\sum_{i:a_i<0}|a_i|K^i.
\tag{16}
\]

Dyadic doubling from one eventually succeeds. Dividing by `x^n` proves `D(x)>0` for all real `x>=K`. Save the exact inequality as a suffix certificate. Alternatively, the exact root chart may supply a smaller integer cutoff above its last nonnegative real root. A root-derived cutoff must carry a complete root-bound verification, not just a numerical root list. Try only the bounded alternatives allowed by §8.

Since `R_k` then increases to `L<=1`, each `R_k<=1` for `k>=K`, so all tails starting there are safe. Only `k=1,...,K-1` remain; `k=0` is automatic.

Define rational unnormalized prefixes

\[
U_u(n)=\sum_{j=0}^{n-1}A_{d,u}(j)r^j,\quad U_u(0)=0.
\]

For each remaining threshold, compute the equivalent exact comparison

\[
\alpha^\delta U_t(k+b)\ge U_s(k)
\iff \alpha^A U_t(k+b)^B\ge U_s(k)^B.
\tag{17}
\]

It is legitimate to raise to `B` because all quantities are nonnegative. This compares the finite CDFs without subtracting two rounded values close to one. Initialize terms at `A(0)r^0=1`; update `term_(j+1)=term_j*r*P_u(j)/Q_u(j)`. Stream both prefixes and the shifted index. Account for generating the initial `b` terms; it is not free preprocessing.

If any comparison in (17) fails, return its exact finite witness. If all pass, return a prefix-plus-suffix proof of global order. If a time, memory, bit-size, or prefix-length cap is reached first, return unresolved with completed work and the limiting resource. Mathematical finite termination does not promise termination under the allocated cap.

### 5.5 Main pseudocode

```text
decide(d, s, t, p, b, route, budget):
    validate exact inputs and budget
    derive alpha, r, rho, H, q, reduced gap A/B
    compare E = (t/s)^B * H^A * q^(b*B) with 1 exactly
    if E > 1:
        return CERTIFIED_FALSE_TAIL_LIMIT with E certificate

    construct D from recurrence products; verify degree and leading coefficient
    if route == "descartes_first":
        if nonzero-coefficient sign variations(D) <= 1:
            return CERTIFIED_TRUE_SHAPE with D and sign certificate

    isolate roots / form exact sign chart within the per-case budget
    if chart certifies negative-then-positive or everywhere-nonnegative D:
        return CERTIFIED_TRUE_SHAPE with exact root/sign certificate

    obtain verified eventual-positive cutoff K
    if generating K+b terms would exceed a declared resource cap:
        return UNRESOLVED_RESOURCE
    initialize and stream U_s, U_t using exact recurrence
    for k = 1,...,K-1:
        check alpha^A * U_t(k+b)^B >= U_s(k)^B
        if false:
            return CERTIFIED_FALSE_FINITE with k and exact inequality witness
    return CERTIFIED_TRUE_PREFIX with cutoff and complete finite-prefix evidence
```

Implement `find_tail_shift` separately and test `b_tail-1` as well as `b_tail`. Root work is subject to a killable worker deadline; after a root timeout, the simple bound (16) may still permit a small prefix under the **remaining** per-case allowance. Never restart the allowance when changing routes.

### 5.6 Independent verification and serialization

Use a separate verifier entry point that never trusts the producer's Boolean decision. It must rebuild `D` from the input by an independent polynomial construction, re-evaluate (15), and replay the appropriate certificate. A simple `Fraction` convolution implementation and a SymPy product expansion provide two distinct constructions. Sharing exact integer primitives is acceptable; calling the same production decision function twice is not independent verification.

For root certificates, check all claimed root intervals and completeness with exact Sturm counts, including multiplicities/signs; the sign chart must cover the whole half-line. The planned paired root reference retains root-based conclusions, so **its independent rational Sturm verifier must be complete at Gate 2, before reference calibration or matched timing**. A small rational polynomial division/gcd/square-free/Sturm implementation is sufficient; a second computer-algebra system is unnecessary. If this verifier is blocked, retain independently verified Descartes/prefix conclusions but mark the paired root comparison blocked. Until verification exists, root outputs and their timings are provisional and cannot enter a valid matched comparison. Cheap coefficient and bound-(16) certificates can be verified entirely with standard-library integer/rational operations. Counterexamples must also be recomputed from the direct coefficient formula (3) when the witness index is affordable; otherwise use a separately implemented recurrence and retain the limitation.

Certificate records must include exact input rationals, polynomial coefficients and normalization, exact tail-test representation, route, cutoff or root data if used, and finite witness or deterministic prefix-replay specification. A hash alone is not the mathematical witness. Store large integers as hexadecimal strings or in compressed text with an explicit encoding to avoid Python's decimal-string length limit; do not silently truncate them. Include a SHA-256 manifest for integrity, separate from proof verification.

Allowed result statuses are `certified_true_shape`, `certified_true_prefix`, `certified_false_tail_limit`, `certified_false_finite`, `unresolved_time`, `unresolved_memory`, `unresolved_size`, `blocked_dependency`, and `invalid`. Verification failure changes a result to `invalid`; it cannot remain in the success count.

## 6. Correctness and adverse controls

Complete these before interpreting any comparison. Exact assertions have zero numerical tolerance. Use explicit exceptions/check results in production; reject `python -O` and a nonempty `PYTHONOPTIMIZE` for the supplied assertion-based scripts. Every check must address an independent identity, a known answer, a certificate branch, or a plausible scientific failure.

### 6.1 Algebra and original-domain controls

1. Compare recurrence coefficients with (3) for `d in {2,3,4,8}`, `u in {1/3,1,5/2,9}`, and `k=0,...,12`. Verify positivity and exact convolutions through `k=12` for shape pairs `(1/3,2/3)`, `(1,5/2)`, and `(5/2,9)`. This tests both the law and shape addition.
2. Recover the paper's quadratic at `d=2`. In its notation,
   `D(k) = [delta(t+s+3)+6b]k^2 + [6b^2+(6t+8-2s^2)b+delta(s+2)(t+2)]k + (s+1)[(4-s)b^2+((4-s)t+2-2s)b+delta(t+1)]`.
   Compare exact coefficients, not sampled polynomial values.
3. Reproduce the supplied scripts with their default complete ranges into new output directories, comparing the recorded results as data. Do not overwrite the original archive outputs. If the original scripts fail, diagnose before importing any of their numerical conclusions.
4. Verify four new-tool controls at `d=2` with fractional gaps and four at `d in {3,4,8,16}` with integer gaps. Use `s=3/2`, gap `2/5` for degree-two controls with load `dp in {1/10,1/3,2/3,9/10}`; use `s=7/3`, gap `2`, load `dp=1/2` for the four integer-gap controls. Check `b_tail` and `b_tail-1` against the proved source theorems.
5. Reconstruct (12)–(13) from rational differentiation and product expansion; verify each identity coefficientwise. Explain the inequalities and quantile consequence in prose. Include `s=5` and the adverse case `d=3,s=6,t=13/2,b=1`, for which `h_1/h_0=23/24<1`.

Planning-stage fixed polynomial fixtures, after clearing denominators and taking the positive primitive part, are:

| `(d,s,t,b)` | Primitive coefficients in ascending powers |
|---|---|
| `(3,1/2,3/4,1)` | `[27075, 105985, 143128, 78472, 15024]` |
| `(3,16,33/2,1)` | `[-244647, -81830, -4891, 705, 63]` |
| `(4,7/3,19/6,2)` | `[42249631840, 153115294658, 218532190113, 156195261360, 58790479185, 11110856832, 830570112]` |

These exercise zero and one sign variation and noninteger gaps. They are development fixtures, not final confirmation.

### 6.2 Exact equality and boundary rounding

Use this rational fractional-gap fixture, derived and checked during planning:

\[
d=3,\quad s=11664/54797,\quad
t=78125/109594=s+1/2,\quad p=1/25,\quad b=1.
\]

Then `H=36/25`, `q=3888/15625`, and

\[
(t/s)^2Hq^2=1.
\]

The §4 theorem proves the order at equality, and the minimal shift is exactly one. Keep `s,t,b` fixed and replace `p` by `1/25 - 1/10^12` and `1/25 + 1/10^12`: exact tail tests must be respectively less than and greater than one. The latter rejects shift one. These are not tolerance tests: every comparison is exact. A `ceil` applied to rounded logs is not an acceptable oracle.

Also test the same-shape wrapper separately: equal laws have distance zero. It should bypass the strict `t>s` core; unordered shapes can be swapped by a distance-only interface. Invalid domains (`p=0`, `p=1/d`, noninteger `d`, negative shift, zero shape) must be explicitly rejected. They must not accidentally enter (15).

### 6.3 Remote-tail and incorrect-generalization controls

**Remote-tail failure:** reproduce the paper's `d=2,p=1/8,s=1,t=9,b=8` case. The first failure in the nonnegative-count threshold is `k=616`, equivalent to centered-sum threshold 615. The reference tail is approximately `3.97935059032695e-226`, with tail ratio approximately `1.000095372332306`. At `k=615`, the ratio is still below one. Preserve the exact rational sign. Run a deliberately limited exact scan through `k=100`; its honest answer is “no violation found in this prefix,” not “order certified.” The all-threshold algorithm rejects immediately from `L>1`. This control demonstrates the insufficiency of short enumeration; it is not a competing infinite-tail solver [A1, §7; A2, `exact_rare_tail_scan`].

**Globally increasing-ratio failure:** reproduce the paper's `d=2,s=9,t=10,b=1` normalized ratio `h_0/limit=9/8` and `h_1/limit=13/16`. Together with (14), this prevents replacing a one-trough argument by an unqualified increasing-ratio assertion.

**Tail-only logic is not generic:** record the source's analytic compound-Poisson counterexample [A1, §7]. Its jump PGF is

\[
J(z)=\tfrac9{10}z^4+\tfrac1{10}\frac{C(z/8)-1}{C(1/8)-1},\qquad
\mathbb E z^{V_u}=e^{u(J(z)-1)}.
\]

At `s=0.001,t=0.002,b=2`, its limiting shifted mass ratio is less than `0.5 exp(0.014)<1`, but

\[
\Pr(V_t\ge4)\ge1-e^{-0.0018}>1-e^{-0.001}\ge\Pr(V_s\ge2).
\]

No new compound-Poisson engine is needed: verify the strict scalar bounds and explain why the generic inference fails. This is a logic control, not an input inside the selected Fuss family.

### 6.4 Exercise fallbacks even if all research cases use the shortcut

Test the root/sign utility on polynomials with a zero at the origin, a repeated positive root, no positive roots, and several sign changes: `x(x-1)`, `(x-2)^2`, `x^2+1`, and `(x-1)(x-2)(x-3)`. These are software fixtures only. A negative-to-positive-to-negative pattern must not produce a one-trough certificate.

Force the exact-prefix route on the valid source-domain case `d=2,s=9,t=10,p=1/16,b=1`; its small cutoff makes a complete independent prefix check cheap. Also force it on the exact equality fixture in §6.2 if its measured cutoff fits the cap. Test cutoff endpoints `k=K-1` and `k=K`, the empty prefix, and a simulated kill/size-limit response. Do not invent a probabilistic counterexample from an arbitrary test polynomial.

Tamper with one saved coefficient, one rational input, and one finite inequality witness in temporary copies; the verifier must reject them. Keep these tests narrow. Do not build a broad testing framework unrelated to the actual certificate trust boundary.

## 7. Smallest sufficient experiment and analysis

### 7.1 Development and freeze

Use the fixtures above and this pilot grid for debugging and calibration:

\[
d\in\{3,4,8,16\},\quad s\in\{1/4,9\},\quad
\delta=3/8,\quad \vartheta=dp\in\{1/5,4/5\}.
\]

This is 16 base instances; test `b_tail` and `b_tail-1` as needed. They differ from the final matrix. Add exactly one calibration stress instance `(d,s,delta,vartheta)=(16,9,3/8,99/100)` to assess exact-power growth. The polynomial exploratory pools described in §2.3 are already development data. Do not call them held-out tests.

After correctness and calibration, write `protocol.lock.json` containing the exact case list, route definitions, selected matrix tier, all ceilings, dependency versions, analysis rules, and source/git hashes. Generate the complete case list before executing it. Do not use final outcomes to tune polynomial representations, choose cases, or change the primary question. Necessary later repairs invalidate the affected runs; preserve them and rerun affected cases once only if the remaining time and call caps permit. A defect in shared core logic invalidates every dependent result and requires a full active-matrix rerun to restore that claim; if it cannot fit, report the affected confirmation as invalid. The repair reserve does not guarantee a full rerun. A scientific protocol change is exploratory and requires a fresh predeclared confirmation list before any new confirmatory claim.

### 7.2 Locked main matrix

Take the full Cartesian product

\[
\begin{aligned}
d&\in\{3,4,8,16\},\\
s&\in\{1/10,3/2,25\},\\
\delta&\in\{1/7,2/5,7/3,13/2\},\\
\vartheta=dp&\in\{1/10,1/2,9/10\}.
\end{aligned}
\]

Set `t=s+delta` and `p=vartheta/d` exactly. There are **144 base tuples**, with two shifts each, for **288 order decisions per route**. Of the 144 base tuples, 24 lie in the degree-three `s<=5` analytic subdomain and serve as embedded positive controls; the other **120 are outside that subdomain and outside the source's integer-gap theorem**. Report those groups separately. Independence of arithmetic checks does not make them independent random samples.

For every tuple:

1. Find and certify `b_tail`, counting threshold-search cost.
2. Run both `descartes_first` and `root_reference` at `b_tail`; independently verify all retained certificates.
3. Run the common exact tail decision at `b_tail-1`, which must reject. Count these as necessary-tail controls, not as informative evidence that the shortcut outperforms root isolation.
4. If a verified finite counterexample at `b_tail` appears, stop the remaining large comparison, independently reproduce that witness, and complete the diagnostic report. The universal extension is already false. Finish only the cheap controls needed to rule out an invalid experiment.

The grid is not optimized to produce a gain: it spans degrees 3–16, both sides of the analytic shape boundary, gaps below and above one, and branching loads from 0.1 to 0.9. It isolates a mathematical mechanism without assuming a real application distribution. Its finite coverage cannot justify extrapolating to arbitrary degree, irrational inputs, or the critical limit.

**Predeclared reduced tier:** if Gate 3's measured projection exceeds the main-stage allocation, use `d in {3,4,8,16}`, `s in {1/10,25}`, `delta in {1/7,13/2}`, and the same three loads: **48 base tuples, 96 decisions per route**, with 42 base tuples outside the analytic subdomain. Choose this tier before confirmation. It retains large shapes, degree 16, fractional gaps, and all load regimes. If even this tier cannot be independently certified within the cap, report practical infeasibility of the intended comparison; do not replace it with only degree-three easy cases.

### 7.3 Twelve bounded stress cases

Only after the main comparison, unless a fatal result ends the study, run these fixed groups as separately labeled robustness work:

| Group | Four cases, one for each `d in {3,4,8,16}` | Purpose |
|---|---|---|
| A | `s=25`, `delta=13/2`, `vartheta=99/100`; determine `b_tail` | Near-critical exact-power and threshold cost |
| B | `s=1/1000`, `delta=1/1000`, `vartheta=9/10`; determine `b_tail` | Small shapes/gaps and a larger rational denominator |
| C | `s=250`, `delta=2/5`; shape certificates at `b=1` and `b=65536`, without a `p` input | Failure of uniform monotone increase at shift one, and large-shift coefficient growth |

Groups A and B have eight probability tuples. Test both adjacent shifts; Group C has eight shape-certificate requests. Thus this stress block has at most **24 decisions per route**. Group C establishes only a ratio-shape fact; a conditional all-`p` order implication follows with `L<=1`, but no numerical distance is claimed without a specified `p`.

Do not replace over-budget stress cases with easier ones. Report their exact parameters and failure reasons. Do not enlarge the stress set after inspecting favorable results.

### 7.4 Baseline roles and fair costs

| Role | Implementation | Legitimate conclusion |
|---|---|---|
| Simple credible reference | Direct tiny coefficient formula, exact finite CDFs, source theorem controls | Checks laws, indices, signs, and boundary handling |
| Strong all-threshold reference | Exact tail precheck, forced root/sign chart, exact prefix fallback | Same mathematical decision with a standard algebraic method |
| Proposed route | Same pipeline with Descartes coefficient-sign bypass | Tests whether this family's algebra permits cheap certificates |
| Targeted ablation | Disable only the bypass; identical to strong reference | Isolates the effect without a redundant third solver |
| Deliberately incomplete control | Scan only the first 101 thresholds in the rare-tail example | Demonstrates that a finite scan cannot certify the target property |

No published runtime from another machine is a comparator. No Monte Carlo, truncated optimal-transport solver, or fixed-tail-padding method is an adequate reference for exact `W_infinity`; excluding them narrows the computational claim to exact decision methods.

Both routes use the same inputs, exact arithmetic, threshold definition, per-case budgets, and fallback. First establish a correct reference. Tuning allowance is at most two ordinary representation/implementation changes on pilot data; record each. Do not alter the algorithm, domain, or tolerances under the name of optimization.

Measure the complete base-tuple workflow, including exact shift search, polynomial construction, certificate production, and verification. Also report these components separately. For end-to-end timing, execute each route from original inputs in its own fresh worker without sharing precomputed polynomials or `b_tail`. A separately labeled shared-preprocessing analysis may show costs with a reused threshold or polynomial, but must state the reuse count and cannot replace total cost. No amortization claim is needed for this study.

For each locked tuple, run each route **once in a fresh worker; this same execution provides both correctness evidence and its first timing**. The two adjacent-shift decisions and their shared shift search belong to that one tuple/route execution and share its deadline. Do not add a separate full correctness pass. For timing stability, preselect **12 tuples** by the lowest SHA-256 of canonical exact case strings, stratified as one tuple per `(d,vartheta)` pair; repeat each route three times in addition to its original execution. Use alternating route order by sorted case index. Include an import warm-up outside the inner timer but inside the parent/global elapsed accounting, then record both parent launch-to-exit time and inner computation-plus-verification time. Report startup separately rather than hiding it. If an inner time is below 10 ms, report an instrument-resolution warning and do not infer a precise speedup from it; no automatic unlimited batching.

This repetition policy is for measurement stability, not repeated mathematical evidence. Report paired medians and ranges on the selected repeats; no significance test or population confidence interval is required for this deterministic grid. Any speed ratio must be restricted to matched, independently valid completed cases and accompanied by both methods' timeout/failure counts. Do not replace censored runtimes by zero, the timeout value, or a success-only overall average.

For call accounting, a tuple/route execution with both shifts counts as **two** order decisions. The full original main pass consumes 576 decisions, its extra repeats 144, the 16 pilot tuples 64, the extra calibration tuple 4, and the final stress block 48: 836 before other controls and any repairs. Independent verifier calls and polynomial utility assertions are not new order decisions, but their wall time and memory still count. Enforce the 1,200-decision ceiling on the actual ledger; do not assume it reserves enough room for a full repeated study.

### 7.5 Interpretation rules fixed before results

- **Supported restricted theorem:** (12)–(13) are independently verified and the proof chain is correct. State exactly `d=3,min(s,t)<=5`, not all real shapes.
- **Supported finite extension:** every reported accepted case has a verified infinite-tail certificate; lower adjacent shifts are rejected exactly. List the actual scope, including any chosen reduced tier.
- **Informative mathematical negative:** an exact finite witness violates (17) despite (15) being true, and independent reconstruction confirms it. One such witness settles the universal conjecture negatively. Finding `L>1` at `b_tail-1` is an expected control, not a new negative finding.
- **Mechanism limitation:** the polynomial is not one-trough, but full CDF comparison still succeeds or remains unresolved. This concerns the proof shortcut, not automatically the sharp criterion.
- **Practical infeasibility:** measured or conservatively projected exact arithmetic costs exceed declared resources. Identify whether the barrier is shift search, root isolation, bit growth, prefix generation, or certificate verification.
- **Inconclusive:** some necessary decisions remain unresolved without a counterexample; retain the analytic subresult if independently valid. A noisy timing pilot is not a refutation of a mathematical hypothesis.
- **Invalid experiment:** an implementation, baseline, index, input, or measurement defect compromises the claimed result. Preserve it as invalid, repair only within the allowance, and never merge its output into valid summaries.

## 8. Target environment, ceilings, and calibration

### 8.1 Inspect the actual Mac

Before target timing, record `sw_vers`, `uname -m`, relevant `sysctl` hardware fields including physical memory and CPU model, Python version/build/architecture, free disk space, dependency versions, whether running through Rosetta, power/battery state, and concurrent heavy workloads. Do not infer memory, storage, OS, or precision support from “M3 Max.” Use native arm64 Python if available; record any translation. Do not install a new system Python or system service without need.

Use one CPU worker at a time. Exact Python/SymPy arithmetic is the primary backend; Metal, MPS, CUDA, and GPU floating precision are irrelevant to this exact task. No remote experimental compute is authorized. Browsing and ordinary dependency downloads are separate from experiment execution. A hosted agent may write code elsewhere, but target performance claims require actual execution on the Mac.

Use Python 3.11 or newer, a local virtual environment, `sympy==1.14.0`, and its `mpmath==1.3.0` dependency as a reproducible initial stack. These are proposed pins, supported by the official API documentation inspected during planning, not a verified installation on the user's Mac [R9]. `mpmath` is not an exact sign oracle. Standard-library `unittest`, `fractions`, `json`, `csv`, `hashlib`, `subprocess`, `resource`, and `time` suffice otherwise. A static CSV/Markdown table is sufficient; use an optional plotting package only if a small figure resolves a specific interpretive question.

If package installation fails, finish the dependency-free certificate and source-reproduction work. Spend at most 20 minutes on one ordinary installation repair. A replacement exact-root implementation is allowed only within the existing engineering/repair budget and must pass the same root controls; otherwise mark the root comparison blocked. Do not silently substitute approximate roots or loosen arithmetic.

### 8.2 Hard caps and their rationale

All failed runs, development probes, source-script reproductions, tuning attempts, timing repeats, verification passes, and fallback branches count cumulatively. Changing routes or stages does not reset a limit.

| Resource | Ceiling | Rationale/action |
|---|---:|---|
| Experimental wall time | 240 minutes total; stop launching new research jobs after 200 minutes | Leaves 40 minutes for preserving/rechecking evidence and report production |
| Total end-to-end agent effort | Target 6–10 hours; hard stop at 12 hours, with final hour reserved for reporting | Includes setup, implementation, reading, debugging, analysis, and compute; this is an estimate, not measured delivery time |
| Worker peak resident memory | `min(4 GiB, 20% of observed physical RAM)` | Conservative local arithmetic study; do not consume the machine's full memory |
| Project disk | 2 GiB total new files; require at least 4 GiB free before starting | Enough for source, environment, exact certificates, and raw logs; stop growing certificates before the cap |
| Parallel workers | 1 | Predictable memory, fair local timing, simple failure accounting |
| Main degree | `d <= 16`, hence polynomial degree at most 30 | Matches the fixed matrix; no high-degree rescue branch |
| Polynomial coefficient size | 65,536 bits per integer after normalization | A generous guard against accidental symbolic blowup |
| Exact intermediate integer size | 20,000,000 bits | Bound large rational powers before allocation; monitor actual RSS as well |
| Tail-shift search | `b <= 1,000,000`, at most 64 exact comparisons per tuple | More than sufficient bracketing/bisection steps at this cap; over-cap is unresolved |
| Prefix generation | `K+b <= 10,000`; total generated prefix terms at most 200,000 for the study | Counts shifted initialization and both laws; controls expensive rational accumulation |
| Per main tuple per route | 120 seconds total, including threshold search, fallback, and verification | Killable subprocess; root phase gets at most 45 seconds of the same allowance |
| Per stress tuple per route | 180 seconds total | Stress is bounded diagnosis, not an indefinite rescue attempt |
| Solver calls | At most 1,200 route-level order decisions over all stages, including repeats and repairs | Prevents unnoticed proliferation; source script internal assertions are separately logged, not counted as new solver calls |
| Repairs/tuning | At most two bounded repair cycles; 20 experimental minutes total | No repeated search for favorable results; engineering time also counts toward 12 hours |

If a local per-process cap cannot be enforced by the OS, the parent must poll worker RSS and terminate near the limit. State the polling interval and possible overshoot. On macOS, `resource.ru_maxrss` is measured in bytes; on Linux it is typically KiB. Detect/document the platform convention instead of copying the hosted audit's unit. A fresh worker avoids cumulative high-water marks contaminating case memory measurements. No device synchronization is needed for CPU-only arithmetic.

Before a power `a^e`, conservatively estimate output bits using `e * bit_length(a)`, and account for the **sum** of bit estimates when multiplying factors or cross-multiplying fractions. Check both numerator and denominator. Intermediate fraction gcd work and symbolic roots can require more memory than the final number, so this bound complements, rather than replaces, the RSS limit. A 20-million-bit integer occupies roughly 2.5 MB of raw bits, but Python objects, temporaries, and several such integers multiply that cost.

### 8.3 Scaling and workload selection

Polynomial construction requires products of `O(d)` linear factors and roughly `O(d^2)` coefficient arithmetic with a simple convolution implementation; operand bit sizes determine the actual cost. Root isolation can suffer coefficient growth and close roots, so do not infer its runtime from degree alone. The prefix fallback uses `O(d(K+b))` factor operations with growing rationals. It can dominate even when the stored arrays are small; stream it.

Writing `vartheta=dp=1-epsilon`,

\[
q=\vartheta\left(1+\frac{1-\vartheta}{d-1}\right)^{d-1},\qquad
-\log q=\frac{d}{2(d-1)}\epsilon^2+O(\epsilon^3).
\]

For fixed unequal shapes, `b_tail` therefore grows on the order of `epsilon^-2`. This is a planning derivation, not a target timing model. Main-grid shifts are expected to be below roughly one thousand, whereas the 0.99 stress load can produce shifts of order one hundred thousand; verify actual exact values. The large exponent in (15), not a floating probability array, is a real hidden cost.

At calibration, record for each of the 17 pilot/calibration tuples: threshold-search cost, coefficient degree and bit sizes, root cost, verification cost, peak memory, and any cutoff/prefix estimate. For each route separately, take the **largest complete tuple/route elapsed time across all 17 calibration tuples**, including the load-0.99 case, and use three times that maximum for every planned tuple/route execution and timing repeat. If a calibration attempt is unresolved, charge its applicable per-tuple ceiling in the projection; never project from successful attempts alone. Invalid correctness or missing verification blocks the paired comparison. Sum those projected costs for the selected matrix and its repeats. This explicitly maps the different development loads to the final loads; no unobserved degree/load stratum is assigned a zero cost. It is a conservative scheduling heuristic, not a statistical or worst-case mathematical bound. Reuse the completed calibration evidence at Gate 4 rather than rerunning the whole pilot automatically. Recheck elapsed/remaining budget before every worker launch.

Choose the full matrix only if that projection, repeats, and required verification fit the 100-minute main-stage allocation and memory/disk caps. Otherwise lock the reduced tier. If a representative pilot already needs the per-case ceiling, do not assume favorable average scaling will rescue the full matrix. If the reduced tier still fails, finish the analytic/diagnostic result and report the blocked computational scope. A stop based only on this conservative projection means allocation was not justified under this protocol; it is not proof that the main matrix is intrinsically infeasible, especially if the harder load-0.99 case alone drove the forecast. Reserve a strong practical-infeasibility claim for measured representative costs or a supported scaling obstruction. Do not reduce degree, larger-shape coverage, or exactness after outcomes are known.

## 9. Evidence gates and autonomous decisions

Experimental allocations below total 240 minutes. They are ceilings, not instructions to consume time. No later-stage infrastructure should be built before the preceding fatal uncertainty is resolved. Engineering effort is tracked separately against the end-to-end cap.

| Gate | Question and minimum work | Required evidence | Experimental allocation | Continue, repair, fallback, or stop |
|---|---|---|---:|---|
| 1. Constructive feasibility | Inspect target/access; reconcile source version; derive recurrence, exact tail test, and finite-prefix bridge; replay §4 identities | Environment record, source hashes, `derivation.md`, exact algebra output | 15 min | Continue if definitions and constructive steps agree. Repair one ordinary transcription issue. A genuine essential mathematical contradiction stops affected claims and triggers a diagnostic report. Missing Mac blocks target benchmarking, not unaffected math/code. |
| 2. Tiny correctness | Implement minimum reference and certificate checker, including independent rational Sturm verification; execute §6 controls and original scripts | Raw controls, source-output comparisons, verified equality/rare-tail fixtures, root completeness/multiplicity controls | 20 min | Require all applicable exact checks and corruption tests to pass. A discrepancy is invalid evidence until diagnosed. An unavailable root verifier blocks the paired reference; unaffected verified mathematics may continue. No timing of an unverified reference is a valid comparison. |
| 3. Resource calibration | Run 16 pilot tuples plus one fixed stress calibration; measure separate stages; project full/reduced matrix costs | Calibration CSV, memory/bit measurements, selected tier and exact ceilings | 10 min | Continue if full or reduced tier fits with headroom. One bounded representation repair allowed. If neither tier is meaningful and feasible, stop empirical comparison with a practical-limit report. |
| 4. Mechanism pilot and freeze | Verify both routes agree on pilot cases; exercise forced prefix and root branches; confirm timings can distinguish setup from core work | Pilot certificates, branch checks, `protocol.lock.json` | 15 min | Freeze if mechanisms and measurement are valid. If timing is noisy, retain descriptive timings and exact scientific question; do not reject it. A true counterexample moves directly to diagnosis. |
| 5. Locked comparison | Execute frozen matrix, paired routes, limited repeats, and independent verification | Complete case manifest with every status, raw timings, certificates | 100 min | Continue to bounded stress only if controls remain valid and no decisive counterexample exists. Stop on a verified counterexample or threatened reserve. Timeouts remain unresolved, never silently excluded. |
| 6. Predeclared robustness | Twelve stress cases only; inspect numerical/size limits and all unfamiliar statuses | Separate stress table and exact evidence | 20 min | Stop at this allocation. No added parameter search to rescue a desired narrative. |
| Repair reserve | At most two repairs across all preceding gates; preserve invalid runs; retest affected controls and at most one frozen-matrix rerun | Deviation/repair log linked to replaced runs | 20 min | If the allowance is exhausted, report invalid or inconclusive affected evidence. A rerun also counts against global calls/time. |
| Diagnosis and report | Recheck retained certificates, regenerate tables, explain scope and uncertainty, package runnable artifacts | Final report, manifest, reproducibility commands | 40 min | Finish even after an early stop; preserve remaining evidence rather than launching new experiments. |

You are authorized to implement and execute these stages autonomously on the available target, make ordinary reversible engineering choices, and repair routine problems within the caps. Do not ask for confirmation at every gate. Ask only if an indispensable missing resource cannot be handled by the specified fallback. Do not claim stages that were not executed. If hardware, permissions, or dependencies block a stage, complete unaffected derivations, code, and checks and identify exactly what remains blocked.

## 10. Minimal implementation and executable handoff

### 10.1 Repository and interfaces to create

Use a single small local Python project. These paths and commands are **interfaces you must implement**; they were not existing executables run during planning. Do not create services, dashboards, a database, a plugin architecture, or distributed orchestration.

| Path | Responsibility |
|---|---|
| `README.md` | Setup, exact input syntax, commands, result meanings, reached-stage summary |
| `requirements.txt` | Initial exact dependency pins; record resolved environment separately |
| `config.json` | Development/control cases, full/reduced matrices, stress list, all budgets |
| `protocol.lock.json` | Immutable evaluation choice, case manifest, analysis rules and source/config hashes |
| `inputs/` | Original attachments and unchanged extracted package |
| `fuss_cert/__main__.py` | Thin command-line dispatcher |
| `fuss_cert/law.py` | Rational parsing, parameter validation, direct coefficient oracle, recurrence, exact tail test, shift search |
| `fuss_cert/polynomial.py` | Recurrence-product polynomial construction, normalization, sign variations, exact cutoff, optional root chart |
| `fuss_cert/certify.py` | Decision pipeline, bounded prefix checks, typed result/certificate records |
| `fuss_cert/verify.py` | Independent coefficient construction and certificate replay; exact root verification when used |
| `fuss_cert/derivations.py` | Reconstruct degree-three identities and general leading-term checks |
| `fuss_cert/run.py` | Fixed case generation, killable workers, resource monitoring, cumulative ledger, safe resume |
| `fuss_cert/report.py` | Deterministic aggregation into CSV/Markdown, coverage/failure summaries and report scaffold |
| `tests/` | Focused correctness and corruption controls from §6 |
| `results/` | Raw records, certificates, environment, locked/stress/pilot separation, logs, budget ledger |
| `REPORT.md`, `derivation.md` | Final evidence-backed interpretation and the mathematical argument |

Keep core functions small: `coefficient(d,u,k)`, `recurrence(d,u,k)`, `tail_compare(d,s,t,p,b)`, `find_tail_shift(...)`, `ratio_polynomial(d,s,t,b)`, `decide(...,route,budget)`, and `verify_certificate(record)`. Return structured results rather than silently throwing away unresolved cases. Domain errors are not numerical nonconvergence. A verification exception must retain the offending exact input.

`run.py` must launch expensive work in fresh subprocesses, write a result/status immediately after each completed or terminated job, and checkpoint the global budget ledger. Resume may run only missing or explicitly invalidated work; it must preserve earlier records and charge every attempt. Avoid a giant final-only results write that could lose the study at a timeout.

### 10.2 Concrete default configuration

Use decimal strings only for presentation. The authoritative configuration represents all nonintegers as rational strings. A compact configuration must at least include the following values and the fully expanded case manifest generated from them:

```json
{
  "study_id": "fuss_real_gap_2026_10_31",
  "routes": ["descartes_first", "root_reference"],
  "main": {
    "d": [3, 4, 8, 16],
    "s": ["1/10", "3/2", "25"],
    "gap": ["1/7", "2/5", "7/3", "13/2"],
    "load": ["1/10", "1/2", "9/10"]
  },
  "reduced": {
    "d": [3, 4, 8, 16],
    "s": ["1/10", "25"],
    "gap": ["1/7", "13/2"],
    "load": ["1/10", "1/2", "9/10"]
  },
  "limits": {
    "experiment_seconds": 14400,
    "stop_new_jobs_seconds": 12000,
    "main_stage_seconds": 6000,
    "main_tuple_route_seconds": 120,
    "root_phase_seconds": 45,
    "stress_tuple_route_seconds": 180,
    "max_integer_bits": 20000000,
    "max_polynomial_coefficient_bits": 65536,
    "max_shift": 1000000,
    "max_shift_search_comparisons": 64,
    "max_prefix_endpoint": 10000,
    "max_total_prefix_terms": 200000,
    "max_solver_decisions": 1200,
    "max_project_bytes": 2147483648,
    "max_workers": 1
  },
  "timing": {
    "repeated_base_cases": 12,
    "additional_repeats_per_route": 3,
    "selection": "one lowest canonical-case SHA-256 per degree/load stratum"
  }
}
```

Fill the environment-dependent memory ceiling from the inspection formula in §8, not from a guessed RAM value. Add the explicit controls, development tuples, stress groups, stage allocations, and encoding convention; these are specified above, not choices to optimize later. Store a canonical case string as `d|s_num/s_den|gap_num/gap_den|load_num/load_den`, using reduced fractions, with literal separators and no whitespace. Sort case strings lexicographically for deterministic run ordering. Select timing-repeat cases by the smallest SHA-256 digest of that string within each degree/load stratum, breaking a digest tie lexicographically. No random seed is required for the deterministic main matrix; any incidental randomized software test must save its seed and remain development-only.

### 10.3 Commands to implement and then execute

From the new project root, after creating the files:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m fuss_cert inspect --out results/environment.json
.venv/bin/python -m fuss_cert source-check --package inputs/package --out results/source
.venv/bin/python -m fuss_cert derive --out results/derivations
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m fuss_cert smoke --config config.json --out results/smoke
.venv/bin/python -m fuss_cert calibrate --config config.json --out results/calibration
.venv/bin/python -m fuss_cert freeze --config config.json --calibration results/calibration --out protocol.lock.json
.venv/bin/python -m fuss_cert run --protocol protocol.lock.json --phase main --out results/locked
.venv/bin/python -m fuss_cert run --protocol protocol.lock.json --phase stress --out results/stress
.venv/bin/python -m fuss_cert verify-all --results results
.venv/bin/python -m fuss_cert report --protocol protocol.lock.json --results results --out REPORT.md
```

`smoke` should run the fixed polynomial, equality, monotonicity-failure, and forced-prefix fixtures, not generate an arbitrary small case. `freeze` must check the gate criteria and emit the exact chosen full or reduced case list; it cannot choose from final outcomes. `run --phase stress` must refuse to run after a decisive counterexample, a failed correctness gate, or an exhausted reserve. `report` must operate on partial results when an early stop prevents later stages. If no protocol was frozen, provide a report mode that explicitly records the last reached gate instead of inventing a lock file.

The original scripts and their `--outdir` interface **were** reproduced during planning. In the new project, the `source-check` wrapper should use these concrete project-root commands, writing separate outputs without modifying the originals:

```bash
.venv/bin/python inputs/package/verify_catalan.py --outdir results/source/core
.venv/bin/python inputs/package/verify_extensions.py --outdir results/source/extensions
```

No network access is needed for those source checks. These project-relative paths become available after the prescribed input extraction; they are not claims that this new repository already exists.

### 10.4 Raw record requirements

One JSONL record per attempt must contain: unique case/attempt ID; phase; exact parameters and shifts; route; code/config hash; start/end times and monotonic elapsed durations; cold total and inner/component times; peak RSS and units; integer/coefficient maxima; polynomial degree; shortcut sign-variation count; prefix terms/cutoff or root statistics if relevant; status and failure reason; verification status; certificate path/hash; and cumulative budget consumed.

Use append-only raw records, not a single mutable table that erases failed attempts. Build summary CSVs from them. Preserve stderr for exceptions and terminated workers. Hash the input attachments and every locked configuration. Record exact dependency versions with `pip freeze`, OS, Python architecture, actual backend, and the fact that integer/rational precision is unbounded subject to resource guards. A displayed decimal error tolerance belongs only to an optional illustration and must be labeled nonrigorous.

## 11. Final artifacts and honest research report

Deliver the artifacts justified by the stages actually reached:

- Runnable code and concise usage instructions for every implemented operation, plus the pinned dependency file and environment record.
- `derivation.md` with the precise new subdomain theorem, proof, independent identity checks, finite-instance algorithm correctness, and the distinction between one-trough failure and order failure.
- Exact certificates/witnesses, raw attempted-run logs, source-output comparisons, and cumulative budget/deviation records. Store actual exact evidence or a deterministic verifier sufficient to reconstruct it; hashes alone are insufficient.
- The frozen configuration and expanded case list if confirmation ran. Keep exploration, confirmation, and stress outputs separate.
- Essential tables: case coverage and status by degree/shape/load; exact decision disagreement count; certificate-route usage; total/component timing and peak memory; all unresolved/invalid cases with reasons. A small optional plot may show bit growth against load, but a plot of the conjectured formula does not add validation.
- `REPORT.md` with a claim-by-claim verdict and enough reproduction instructions to regenerate all retained evidence. Package the project without the virtual environment, caches, or unrelated source files if a single archive is useful; keep the total disk cap.

Lead the report with what was learned and which decision it supports. State whether to invest next in a general proof, investigate a concrete counterexample, improve a specific exact-arithmetic bottleneck, or stop. Recommend at most one next scientific step, justified by the actual result.

The report must separately identify:

1. **Supported contribution:** exact continuum or finite-domain result, prior-art status, and the validated conditions.
2. **Informative negative result:** a verified obstruction or adequate valid evidence against the stated hypothesis; include the smallest reproducible witness found within the budget, without an unbounded minimization search.
3. **Practical infeasibility:** measured resource obstruction and the intended scope it blocks.
4. **Inconclusive evidence:** unresolved cases, missing hardware/access, or inadequate measurement coverage.
5. **Invalid experiment:** defects that prevent interpreting the attempted comparison.

More than one category can apply to different claims. For example, a verified degree-three theorem can coexist with inconclusive higher-degree benchmarking. A baseline timeout does not erase a mathematically valid certificate, but it limits a comparative efficiency claim. An implementation bug does not refute the theorem. A zero-counterexample finite study is not a universal theorem or a bound on the prevalence of counterexamples.

For an analytic early completion, produce the independently checkable derivation and the small verification code actually used; omit unperformed benchmark tables. For a resource or access stop, preserve the completed evidence and state the exact command/stage that remains blocked. Do not fabricate a benchmark to fill a reporting heading.

## 12. Compact claim ledger and planning-review disposition

### 12.1 Claim-to-evidence map

| Central claim | Source or derivation | Current status | Remaining resolving check |
|---|---|---|---|
| The chosen mass function is a normalized positive-shape convolution family | [A1, §2/A.1], [R1–R2]; Lagrange inversion | Established construction | Independent recurrence/direct formula/convolution controls |
| The manuscript proves arbitrary real gaps for `d=2`, but only integer gaps for larger `d` | [A1, Theorem 3.1 and Appendix A.1–A.4] | Source scope verified | Do not misapply its finite product |
| `L>1` rules out the shifted order for all positive real shapes | Equation (6) and eventual mass comparison | Direct analytic consequence | Verify formula mapping and exact tail arithmetic |
| A degree-`2d-2` polynomial controls consecutive ratio signs | Equations (8)–(10), planning exact probes | Derived and checked on small instances | Exact product identity and leading-coefficient controls |
| A certified suffix plus finite CDF comparisons decides a rational instance | Equations (16)–(17) and proof in §5.4 | Constructive derivation, not yet a full implementation | Independent cutoff/prefix verifier and resource calibration |
| Arbitrary real gaps work for `d=3,min(s,t)<=5` | New derivation (12)–(14) | Independently algebra-checked during planning | Executor replays identities and full inference; external novelty remains provisional |
| The formula also works at the 120 new-domain main tuples | Hypothesis (H) and fixed matrix | Untested confirmatory claim | Execute and independently verify all active cases |
| Descartes bypass reduces practical cost | Algorithmic hypothesis; cheap exact planning probes | Plausible, no Mac evidence | Fair paired reference/ablation and component accounting |
| The study fits one M3 Max | Size analysis and bounded design | Unmeasured planning assumption | Actual environment and calibration gate |
| The contribution is globally new | Focused literature review | **Not established** | Cite closest work; a later publication novelty review is separate |

### 12.2 Three review passes and material revisions

The planning process used distinct mathematical/novelty, experimental, and execution reviews. The final brief was revised to address the following material issues; this is a disposition summary, not a claim of exhaustive correctness.

| Adversarial concern | Disposition in this brief |
|---|---|
| Infinite-support theorem hides an oracle or a nonconstructive step | Input `p` removes generating-function inversion; explicit recurrence, exact sign polynomial, tail bound, and finite prefixes give a constructive decision. Exact optimal sampling is excluded. |
| Integer-gap proof is silently reused for fractional gaps | Source restriction is explicit; the new polynomial method and §4 proof are separately derived. |
| A positive result merely reproduces known machinery | Closest half-monotone and gamma/moment literature is cited; novelty is assigned only to a verified family-specific extension or limitation, provisionally. |
| Only a favorable subdomain is studied | The easy degree-three subdomain is a proof/control; the main matrix keeps `s=25` and degrees up to 16. Reduced scope must retain them. |
| A stronger baseline removes the apparent gain | The root reference is also the ablation, with identical exact prechecks and fallbacks. Scientific value does not depend on a speedup. |
| Threshold search or verification dominates | Both are timed in total cost and separately; no unexplained amortization or free preprocessing. |
| Finite precision changes the integer answer | Rational powers and an exact equality fixture replace rounded logarithmic ceilings. No numerical tail sign is authoritative. |
| Short scans miss rare-tail violations | The known `k=616` witness is mandatory; suffix certification replaces arbitrary truncation. |
| Failure of a proof tactic is mistaken for falsity | Multiple coefficient changes or roots trigger the finite-prefix route or unresolved status. Only a verified order witness refutes (H). |
| Memory/time exceeds the forecast | Exact bit, RSS, prefix, worker, global-time, and call caps; full/reduced tiers selected before confirmation. |
| Duplicate passes or ambiguous calibration exceed the budget | One original execution serves correctness and timing; repeats alone add timing runs. The global per-route pilot maximum maps explicitly to all final loads. Repairs require remaining capacity and do not promise a full rerun. |
| Small pilot is inconclusive | Use it to assess feasibility and instrumentation; retain the mathematical test unless a valid obstruction exists. No significance-based premature rejection. |
| Essential source/dependency/hardware is unavailable | Complete unaffected exact work; identify the blocked source reproduction/root comparison/target timing without fabricating it. |
| Main hypothesis is false | One independently verified finite counterexample is a valid early completion; preserve it and the valid analytic subresult. |
| A source statement itself is unreliable | All operational identities are derived directly. The inspected Qi–Cerone preprint's increasing-sequence assertion conflicts with `A(0)=1,A(1)=u<1` for small `u`; this brief does not use it. The final publisher text was not fully accessed, so no defect is asserted about that version. |

Residual risks are historical novelty, correct implementation of exact root/certificate machinery, and target-machine costs. They are handled by attribution limits, independent checks, and early resource gates. They cannot honestly be removed by stronger wording.

## 13. Supporting research record and references

### 13.1 Search coverage

Research was conducted on **31 October 2026**, using two available web search engines and primary-source retrieval. Searches covered Catalan/Raney/Fuss–Catalan coefficients, Jain–Consul/Lagrangian negative binomial laws, generalized Poisson, shifted stochastic order, half-monotone likelihood ratios, dispersive order, bounded transport, gamma quotients, and coefficient log-convexity. Backward references were followed where distributional identity or proof applicability mattered; a useful forward connection was the 2022 Belzunce–Martínez-Riquelme–Pereda paper.

Representative query groups were `"generalized negative binomial" "stochastic order"`, `"Fuss-Catalan" "transport"`, `"generalized Poisson" "stochastic ordering"`, `"half-monotone likelihood"`, `"inverse Gaussian" "shifted" "stochastic order"`, `"Raney" "log-convex"`, and `"Fuss-Catalan" "2p-1"`. These terms are not equivalent families: compactly supported free-probability laws whose **moments** are Fuss–Catalan coefficients are not the exponentially tilted discrete laws studied here. Their moment results are relevant only when explicitly transferred to coefficient inequalities.

The review did not establish exhaustive historical novelty. Full access to the original 1971 paper and all chapters of Consul–Famoye's *Lagrangian Probability Distributions* was unavailable. Several publisher requests failed; accessible primary manuscripts were used with their version stated. No external dataset is required. External documents are for attribution and audit; the executable mathematical definitions are included in this brief.

General activation/catalytic phenomena are also prior art: Aubrun–Nechita study equal powers of two different measures [R10], whereas the supplied centered-power example compares different powers within one family. Their main all-exponential-moment setting cannot automatically be transferred to a subcritical Catalan law with a finite positive convergence boundary. This is why an activation-reproduction project was not selected.

### 13.2 References and precise access status

**[A1] Supplied mathematical note.** *Exact stochastic comparison and bounded transport for Catalan convolution powers*. Author line: “Research developed in an autonomous mathematical investigation.” Dated 31 October 2026, unpublished attachment `catalan_transport_note(1).pdf`, 14 pages. Full text and figures reviewed. Source `catalan_transport_note.tex` and included `fuss_poisson_appendix.tex` are in [A2]. Critical locations: Theorem 3.1/Lemma 3.2, pp. 2–4; Corollary 4.1, p. 4; §§5–7, pp. 4–8; Appendix Theorem A.1/Lemma A.2, pp. 10–12; Theorem A.3, pp. 12–13; open real-gap question, Appendix A.4, p. 14. No public DOI supplied.

**[A2] Supplied reproducibility archive.** `catalan_research_reproducibility(1).zip`, dated 31 October 2026. All fourteen entries reviewed, including source, scripts, exact/numerical outputs, and figures. Critical functions: `verify_catalan.py`—`verify_polynomial_instance`, `coefficient_checks`, `exact_rare_tail_scan`, `inverse_gaussian_checks`; `verify_extensions.py`—`exact_sorted_product_checks`, `gp_shift_checks`, `poisson_limit_checks`. Both scripts reproduced in the planning environment; new proof-of-concept and Mac timings not run. Original input hashes are in §2.1.

**[R1] G. C. Jain and P. C. Consul (1971).** “A Generalized Negative Binomial Distribution.” *SIAM Journal on Applied Mathematics* 21(4), 501–513. DOI: [10.1137/0121056](https://doi.org/10.1137/0121056). [Publisher record](https://epubs.siam.org/doi/10.1137/0121056). Publisher abstract inspected; full article not accessed. Supports foundational attribution, not a claim that every historical ordering result was excluded.

**[R2] Pouya Faroughi, Shu Li, and Jiandong Ren (2025; online 18 September 2024).** “Generalized Poisson random variable: its distributional properties and actuarial applications.” *Annals of Actuarial Science* 19(1), 140–158. DOI: [10.1017/S1748499524000198](https://doi.org/10.1017/S1748499524000198). [Open full text](https://www.cambridge.org/core/journals/annals-of-actuarial-science/article/generalized-poisson-random-variable-its-distributional-properties-and-actuarial-applications/F0311FC1D9574924BBA8130547733D36). Relevant content read: §1 GNB mass and GP limit; §2 distributional representations. Map their `(a,b,alpha)` to this brief's `(u,d,p)`. Their risk-measure FFT comparison is a different computational task.

**[R3] Achim Klenke and Lutz Mattner (2010).** “Stochastic ordering of classical discrete distributions.” *Advances in Applied Probability* 42(2), 392–410. DOI: [10.1239/aap/1275055235](https://doi.org/10.1239/aap/1275055235). [Author manuscript](https://www.staff.uni-mainz.de/klenke/papers/binomial.pdf); [arXiv:0903.1361](https://arxiv.org/abs/0903.1361). Relevant full text read: §1.1 common-quantile characterization; §2.1 half-monotone criterion and examples showing shape sufficiency is not necessity. Author-manuscript Proposition 2.3 corresponds to published Proposition 2.1, pp. 398–399. Cite the version-specific numbering.

**[R4] Félix Belzunce, Carolina Martínez-Riquelme, and Magdalena Pereda (2022).** “Sufficient Conditions for Some Stochastic Orders of Discrete Random Variables with Applications in Reliability.” *Mathematics* 10(1), 147. Published 4 January 2022. DOI: [10.3390/math10010147](https://doi.org/10.3390/math10010147). [Accessible full article](https://arxiv.org/pdf/2601.19389). Relevant content read: §3 Theorem 1, pp. 4–5; §4 Example 3, pp. 9–10, generalized-Poisson conditions (7)–(10). The 2026 arXiv deposit contains the 2022 article. It is adjacent prior art on unshifted likelihood/hazard/mean-residual-life orders, not the specific shifted Fuss formula.

**[R5] Wojciech Młotkowski (2010).** “Fuss-Catalan Numbers in Noncommutative Probability.” *Documenta Mathematica* 15, 939–955. DOI: [10.4171/DM/318](https://doi.org/10.4171/DM/318). [Publisher full PDF](https://ems.press/content/serial-article-files/26099). Relevant full text read: equation (2.1) for coefficients, equation (2.4) for convolution; Theorem 5.1, pp. 948–949, for nonnegative-support moments; following paragraph, p. 949, for the first-minor necessary condition. These are coefficient/moment facts, not an identification of the discrete count law with the continuous moment measure.

**[R6] Feng Qi and Pietro Cerone (2018).** “Some Properties of the Fuss–Catalan Numbers.” *Mathematics* 6(12), 277. DOI: [10.3390/math6120277](https://doi.org/10.3390/math6120277). Read the [2017 author preprint](https://www.preprints.org/manuscript/201708.0056/v1/download), DOI [10.20944/preprints201708.0056.v1](https://doi.org/10.20944/preprints201708.0056.v1): equation (1.6), Theorems 1.1–1.2. Final publisher full text was not completely accessible. Its stated coefficient log-convexity domain is `0<r<=p`; do not use the preprint's separate increasing-sequence assertion without checking its endpoint convention.

**[R7] Feng Qi and Dongkyu Lim (2020).** “Monotonicity properties for a ratio of finite many gamma functions.” *Advances in Difference Equations* 2020, 193. Published 1 May 2020. DOI: [10.1186/s13662-020-02655-4](https://doi.org/10.1186/s13662-020-02655-4). [Open full text](https://link.springer.com/article/10.1186/s13662-020-02655-4). Relevant content read: equation (2.7), Theorem 3.1. Provides structured gamma-quotient monotonicity; no direct applicable specialization resolving the full shifted two-shape problem was established in this review.

**[R8] Bao-Xuan Zhu (2015 version; first posted 2013).** “Monotonicity and log-behavior of some functions related to the Euler Gamma function.” [arXiv:1309.5693v3](https://arxiv.org/abs/1309.5693v3), dated 28 April 2015; [full PDF](https://arxiv.org/pdf/1309.5693). Relevant content read: Theorem 4.2, pp. 11–12, and Corollary 4.4, p. 13. The theorem's integer-offset restrictions and direct substitution condition do not immediately establish the real `3<s<=5` interval used here.

**[R9] SymPy development team.** *SymPy 1.14.0 documentation*, accessed 31 October 2026. [Polynomial manipulation reference](https://docs.sympy.org/latest/modules/polys/reference.html), especially `Poly.intervals`, `count_roots`, `sturm`, and the warning implicit in the restricted output of radical-root routines; [polynomial domains](https://docs.sympy.org/latest/modules/polys/domainsref.html), especially `QQ` and `ZZ`; [root-finding guidance](https://docs.sympy.org/latest/guides/solving/find-roots-polynomial.html). Relevant API content inspected. The actual target installation, API behavior on root fixtures, and resolved versions remain execution-gate requirements.

**[R10] Guillaume Aubrun and Ion Nechita (2009).** “Stochastic domination for iterated convolutions and catalytic majorization.” *Annales de l'Institut Henri Poincaré, Probabilités et Statistiques* 45(3), 611–625. DOI: [10.1214/08-AIHP175](https://doi.org/10.1214/08-AIHP175). [arXiv:0707.0211](https://arxiv.org/abs/0707.0211); [full PDF](https://arxiv.org/pdf/0707.0211). Relevant content read: Example 1.4 and Proposition 1.5, manuscript pp. 2–3; §2 integrability assumptions. Background for activation and catalysis, not an automatic solution of the selected sharp shifted comparison.

---

**Execution completion rule:** stop when the declared scientific decision is supported, a fatal mathematical/resource/access gate has been resolved as a limitation, or the cumulative budget is exhausted. Preserve and interpret the evidence actually obtained. Do not prolong the study to force a favorable result, and do not leave an authorized ordinary implementation repair unfinished while budget and a clear remedy remain.
