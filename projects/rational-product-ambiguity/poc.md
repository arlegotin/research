# Execution brief: certify the dimension of infinite rational-product ambiguity

**Recipient:** autonomous coding agent configured as `gpt-6.1-sol`, reasoning effort `xhigh`.

**Prepared:** 29 October 2026. Research retrieval date: 29 October 2026, Europe/Amsterdam.

**Your assignment:** implement and execute the bounded study below on one MacBook with an M3 Max. Produce a reproducible mathematical decision, including an informative early stop if warranted. This document is the planning deliverable; the proposed PoC has not already been implemented or run. Small planning probes actually performed are identified in Section 12.

## 1. Project decision and smallest defensible contribution

### 1.1 Selected project

Build an **exact-arithmetic certificate and independent numerical audit of the infinite-fiber dimension theorem** for the following five specified rational product systems. Fix `c = 1` throughout the implementation.

| System | Rational factor at index n >= 1 | Proposed dimension of the infinite-fiber value set |
|---|---|---:|
| A | `P_1(2^(-n))` | 1/2 |
| B | `P_2(2^(-n))` | 1/3 |
| C | `P_3(2^(-n))` | 1/4 |
| D | `P_4(2^(-n))` | 1/5 |
| E | `P_(n+4)(2^(-n))` | 0 |

Here `P_m(t) = sum(t^j/j!, j=0,...,m)`. The word *precision* in the motivating problem means polynomial truncation degree, not decimal arithmetic precision. The previously unspecified increasing-degree system is fixed to `d_n=n+4`, matching the paper's numerical example. General positive `c`, other degree schedules, and arbitrary rational factor collections are outside the implementation's certification domain.

**Primary falsifiable question:** Can the five stated dimensions, and the local infinite critical Hausdorff measure claim for A-D, be supported by a complete, independently audited chain of all-level inequalities and an exactly verified finite certificate, with correctly checked hypotheses of the published mass transference principle, or does an identifiable mathematical obstruction invalidate a necessary step?

The planning audit found no fatal gap in the essential source proofs. It also derived explicit rational bounds that make the proof substantially easier to audit and execute. Your job is to reproduce and scrutinize that derivation, build a rejecting verifier, check it through independent computations, and report exactly what was established. Do not accept this brief's derivations merely because they are supplied here.

### 1.2 Contribution and attribution

The smallest credible contribution is a **quantitative certificate with explicit bounds valid at every index**, plus a transparent verification and falsification package for the specified rational families. It replaces unspecified asymptotic remainder constants and numerical slope fitting with finite algebraic obligations and reviewed analytic deductions.

The underlying mathematical novelty remains unresolved. Rational radix products already occur in Knopfmacher [R2]. Infinite subsum multiplicity is an established cardinal-function topic [R3]. Earlier expansion-approximation work already obtains reciprocal-exponent dimension laws [R5, R6]. The candidate distinction is the identification of infinite fibers with the actual child-overlap limsup in this nonstationary interval-filling regime, together with explicit rational families. New code, a new name, or the number `1/alpha` alone is not a new theorem.

A successful audit is sufficient to justify a specialist mathematical review and a better reproducibility supplement. It is not proof of historical priority, publication readiness, a new general mass transference theorem, or a computational advantage over binary encoding. If an equivalent theorem is found, report a reproduction/quantitative refinement and withdraw the originality claim. Do not invent a different project to rescue novelty.

### 1.3 Deliberate scope revision

The motivating request asks to independently compute the overlap structure and test a dimension law. Retain that purpose, but make the decisive evidence a proof audit with finite exact certificates. Large finite prefix enumerations cannot identify an infinite-fiber Hausdorff dimension; Section 4 explains why. Numerical work is a correctness diagnostic, not a dimension estimator.

**Exclude the 10,000-complex-number, 100-decimal-digit codec benchmark.** It would answer a separate question about encoding cost and error, for which no efficiency mechanism or advantage is established. The present result will make no compression, speed, storage, or complex-codec claim. The paper's complex extension, prime-like interpretations, separate dimensions of countably infinite versus continuum fibers, exact finite multiplicity ranges, and height-versus-dimension tradeoffs are also excluded. Do not add GPU work, a proof-assistant port, a dashboard, or a general symbolic theorem prover.

### 1.4 What counts as a decisive result

The primary outcome is a proof-obligation status, not a fitted accuracy score. A supported result requires all applicable algebraic checks, analytic deductions, and source-theorem hypotheses to be resolved without contradiction. One failed necessary obligation blocks the corresponding claim. There is no acceptable percentage of failed identities.

The exact all-level covering margin and a quantified Hausdorff cover are the meaningful thresholds. Numerical agreement to a chosen tolerance is insufficient by itself. Failure of a sufficient certificate is also insufficient to refute the theorem: first distinguish a coding error, a false new bound, a gap in the source argument, and an actual counterexample to a source claim.

## 2. Inputs, provenance, and evidence status

### 2.1 Supplied files

Preserve the supplied bytes under `input/` and record hashes before editing anything. The attachment names are authoritative; the workspace paths below describe the planning environment, not the future Mac.

| ID | Attachment and original planning location | Role |
|---|---|---|
| P | `rational_product_ambiguity.pdf`, `/workspace/scratch/037bb97f5059/upload/rational_product_ambiguity.pdf` | Thirteen-page research note, *Rational product expansions with controlled infinite ambiguity*, dated 29 October 2026; attributed on its title page to an autonomous investigation prepared with ChatGPT. Treat its proofs as claims to audit. |
| S | `ambiguity_theory_stress_test(2).py`, `/workspace/scratch/037bb97f5059/upload/ambiguity_theory_stress_test(2).py` | Supplied finite stress test using `Fraction`, `mpmath`, and NumPy. It is evidence of the previous experimental design, not a trusted implementation. |
| A | `Pasted text(20261029-164203).txt`, `/workspace/scratch/037bb97f5059/upload/Pasted text(20261029-164203).txt` | Original assignment and reported experimental observations. This brief includes the decisions necessary for execution without that file. |

SHA-256 hashes computed during planning:

```text
P: 6719006788244e6671faf77b89215630beb70b6a166a6a60d734651785b1f65f
S: 613476ec945b183685ec1e6b0d2bb0811a3c8aa263ddfa40398519a97c85183e
A: a98e9f6dd2f3b84186226905cb5f4cbc1ac56ed4c131e98413ec1fe9f7a46a62
```

The paper mentions `reproduce.py`, but that file was not supplied. Do not invent its contents or claim to reproduce its full experiments. The supplied stress test is a different file. Its absence does not block this study: the definitions, proofs, and required diagnostics are specified here. If P or S is missing on the Mac, complete the independent certificate from this brief, but mark source-byte comparison or literal script reproduction unavailable. Retrieve the published MTP statement before claiming its invocation has been source-checked. If retrieval fails, the lower-bound result remains explicitly conditional; E's elementary upper-bound work can continue.

### 2.2 Central claim-to-evidence map

Page references below are the paper's printed pages.

| Claim | Source or derivation | Planning status | Resolving check in your study |
|---|---|---|---|
| Every finite prefix has an interval of attainable continuations | P, Lemma 2.1, p.3 | Proof audited; depends on all-level `w_n <= R_n` | Reproduce interval-cover induction and verify coverage for each family |
| Prefix counts increase to fiber cardinality, with both infinite cardinalities encoded as +infinity | P, Lemma 2.2, pp.3-4 | Proof audited, including duplicate words and endpoints | Exact tiny tree/sweep controls and written argument |
| Infinite fibers equal the limsup of child overlaps | P, equation (6), p.4 | Essential audited identity | Derive from pointwise counting; do not substitute a finite count criterion |
| Integral of multiplicity is 1 when c=1 | P, equations (1), (4), pp.2-4; Section 3 below | Audited analytic consequence | Exact control integrals and certified tail bounds |
| A-D have dimensions 1/(m+1), with infinite critical measure locally | P, Theorem 4.1, p.6; Theorem 1.2, p.2 | No fatal defect found; requires published MTP | Exact constants, common-center identity, complete MTP applicability ledger |
| Overlap leading coefficient D_r is correct | P, equation (10), p.7; explicit remainder derived here | Symbolically rederived; small exact checks performed | Independently recompute D_r and residual bounds; two numerical routes |
| E has dimension zero and a comeager continuum-fiber subset | P, Theorem 1.3 and Section 5.3, pp.2-3,7-8 | Proof audited; no MTP needed for dimension zero | Universal degree argument and quantified cover bound |
| Strict overlaps imply generic continuum multiplicity | P, Theorem 3.1, pp.4-5 | Category/compact-fiber proof audited | Written topological argument; finite counts do not establish this |
| All-level rational inequalities and K=4 below | New planning derivation | Exact constants checked; not yet independently implemented | Restricted certificate generator and separate verifier |
| Reported slopes and prefix maxima | A; S; P Section 7, p.10 | Finite observations; original script not executed in planning | Re-evaluate only the scoped diagnostics, with certified arithmetic |
| Mathematical originality | P Section 8, p.11, and research in Section 13 | Unresolved; important closer prior art located | Preserve qualification; bounded source refresh only if decision-relevant |

## 3. Mathematical specification and the proof to audit

### 3.1 Objects, domains, and conventions

For each selected family let `q_n>1` be its rational factor and set

\[
w_n=\log q_n,\quad S=\sum_{n\ge1}w_n,\quad
R_n=\sum_{k>n}w_k,\quad R_0=S,\quad
\delta_n=R_n-w_n.
\]

All logarithms are natural. The index starts at 1; `R_n` excludes `w_n`. These are scalar real quantities. Products are limits in increasing index order. The binary coding map is

\[
\pi(\varepsilon)=\sum_{n\ge1}\varepsilon_n w_n,
\qquad \varepsilon\in\{0,1\}^{\mathbb N}.
\]

Define `M(x)=#pi^(-1)(x)` on `[0,S]`, counting distinct binary sequences even when finite prefix values coincide. For integration and the infinite-fiber set, record either infinite cardinality as `+infinity`. Define

\[
E_\infty=\{x\in[0,S]:M(x)\text{ is infinite}\}.
\]

For a word `u` of length n, set `p(u)=sum(u_j w_j,j=1,...,n)` and `I(u)=[p(u),p(u)+R_n]`. All cylinders and overlaps are **closed**. For the empty prefix, `I(empty)=[0,S]`.

Product values are `y=exp(x)` in `[1,exp(S)]`, represented by `product(q_n^epsilon_n)`. Since `exp` has derivative between 1 and `exp(S)` on this compact interval, it is bi-Lipschitz there. The dimensions and local infinite positive-order Hausdorff measure conclusions transfer to product values. Mean multiplicity uses `dx=dy/y`, not uniform measure in y. No separate finite multiplicity histogram in y should be labeled a logarithmic mean.

### 3.2 Coverage, counting, and the infinite-fiber identity

If `w_n <= R_n` at every index, then

\[
[0,R_{n-1}]=[0,R_n]\cup[w_n,w_n+R_n].
\]

Recursively choose a child containing a target. The remaining error is between 0 and `R_n -> 0`, proving that every tail is an interval and every counted prefix extends to an infinite representation. This is a mathematical construction; it does not supply an exact comparison oracle for an arbitrary real input. The selected study does not need such an encoder.

Let `N_n(x)=sum_{|u|=n} 1_(I(u))(x)`. For a parent with prefix value p,

\[
\mathbf1_{[p,p+R_n]}+\mathbf1_{[p+w_n,p+w_n+R_n]}
=\mathbf1_{[p,p+R_{n-1}]}+
\mathbf1_{[p+w_n,p+R_n]}.
\]

This holds at endpoints, including a singleton overlap when `delta_n=0`. Summing proves `N_n` is nondecreasing. If there are finitely many distinct representations, their prefixes eventually separate and exhaust all extendable prefixes. If the fiber is infinite, any finite number of its members eventually have distinct prefixes. Thus `N_n -> M`, in the extended counting convention, and

\[
M(x)=1+\sum_{n\ge1}\sum_{|u|=n-1}
\mathbf1_{B_{u,n}}(x),\qquad
B_{u,n}=[p(u)+w_n,p(u)+R_n].
\tag{1}
\]

Each level has finitely many words, even when positions repeat. Therefore

\[
E_\infty=\limsup_{n\to\infty}\bigcup_{|u|=n-1}B_{u,n}.
\tag{2}
\]

The integral identity is

\[
\int_0^S M(x)\,dx=\lim_n 2^nR_n
=S+\sum_{n\ge1}2^{n-1}\delta_n.
\tag{3}
\]

This follows from `integral N_n=2^n R_n` and monotone convergence. For the selected systems the limit is 1. In particular, `M` is finite almost everywhere and `E_infty` is Lebesgue-null. These facts are compatible with topological genericity.

### 3.3 Exact all-level bounds for A-D

Put `t=2^(-n)`, `r=m+1`, and `a_m(t)=t-log(P_m(t))`. Do not implement the main certificate as a floating-point Taylor fit. Use the exact identities

\[
P_m-P_m'=t^m/m!,\qquad
a_m'(t)=\frac{t^{r-1}}{(r-1)!P_m(t)}.
\tag{4}
\]

Every coefficient of `P_m` is nonnegative and its constant coefficient is 1. Moreover,

\[
(1-t)P_m(t)-1
=-\sum_{j=2}^{m}\frac{j-1}{j!}t^j-\frac{t^{m+1}}{m!}\le0
\quad(t\ge0).
\tag{5}
\]

The empty sum for m=1 is zero. Consequently `1-t <= 1/P_m(t) <= 1`. Define

\[
B_r=\frac1{(r-1)!(r+1)},\qquad
D_r=\frac{2^r-2}{r!(2^r-1)}.
\]

Integrating (4)-(5), with `a_m(0)=0`, gives the explicit remainder

\[
0\le\frac{t^r}{r!}-a_m(t)\le B_rt^{r+1}.
\tag{6}
\]

Since `sum_{j>=1} t/2^j=t`, the overlap is `a_m(t)-sum_{j>=1}a_m(t/2^j)`. Apply (6) and the exact geometric sums:

\[
D_rt^r-B_rt^{r+1}\le\delta_n
\le D_rt^r+\frac{B_r}{2^{r+1}-1}t^{r+1}.
\tag{7}
\]

Also,

\[
t-\frac{t^r}{r!(2^r-1)}\le R_n\le t.
\tag{8}
\]

The inequality `B_r<=D_r` is equivalent to `2^r>=r+2`, true for every integer r>=2 by induction. Since t<=1/2,

\[
L_r2^{-rn}\le\delta_n\le U_r2^{-rn},\qquad
L_r=D_r/2,\quad U_r=1/r!.
\tag{9}
\]

The upper bound uses `delta_n<=a_m(t)<=t^r/r!`. These are bounds for **all n>=1**, with no unspecified eventual cutoff. They establish strict coverage and yield `R_n~2^(-n)` and `delta_n~D_r 2^(-rn)` with the explicit residual in (7).

| m | r | D_r | B_r | L_r | U_r | `4^r D_r/4` |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2 | 1/3 | 1/3 | 1/6 | 1/2 | 4/3 |
| 2 | 3 | 1/7 | 1/8 | 1/14 | 1/6 | 16/7 |
| 3 | 4 | 7/180 | 1/30 | 7/360 | 1/24 | 112/45 |
| 4 | 5 | 1/124 | 1/144 | 1/248 | 1/120 | 64/31 |

Recompute this table; do not use it as the implementation's source of truth.

### 3.4 Dimension and the exact MTP invocation

There are `2^(n-1)` word-indexed overlap intervals at level n. For every s>1/r, (9) gives a convergent Hausdorff covering series

\[
\sum_n2^{n-1}\delta_n^s
\le\frac{U_r^s}{2}\sum_n2^{n(1-rs)}<\infty.
\tag{10}
\]

Tail unions from (2) have diameters tending to zero and costs tending to zero, proving the upper bound `dim_H E_infty <= 1/r`.

For the lower bound, the center and radius of `B_(u,n)` are

\[
z_{u,n}=p(u)+\frac{w_n+R_n}{2}
=p(u)+\frac{R_{n-1}}2,
\qquad \rho_n=\delta_n/2.
\tag{11}
\]

The center is exactly the parent cylinder's center. Its parent radius is at most t by (8) and `w_n<=t`. Choose `s=1/r` and the dimension function `f(rho)=4 rho^(1/r)`. By (9),

\[
f(\rho_n)\ge4(D_r/4)^{1/r}t\ge t\ge R_{n-1}/2.
\tag{12}
\]

The middle inequality is checked with exact rational arithmetic as `4^r D_r/4>=1`, avoiding a numerical root. The table shows a strict margin for all four degrees. Enlarged overlaps therefore contain their entire parents **at every level**. Parents cover `[0,S]`; enlarged balls consequently have a full-cover limsup there.

Use Beresnevich and Velani [R1], **Section 6.1, pp.990-991, Theorem 3, “A general Mass Transference Principle.”** In their setting, small metric balls have Hausdorff g-measure comparable to g(radius), g is doubling, and f/g is monotone. Full local g-measure for the limsup of balls enlarged to radius `g^(-1)(f(radius))` implies full local f-measure for the original limsup. Cite the section and pages: the same paper also labels another theorem “Theorem 3” on p.992.

Audit the following application explicitly:

| Hypothesis | Concrete verification here |
|---|---|
| Locally compact metric space | `X=[0,S]` with ordinary distance is compact |
| Doubling g and regular small balls | `g(rho)=rho`; interval metric-ball lengths lie between rho and 2rho for sufficiently small rho, including endpoints; the source's Hausdorff normalization only changes fixed comparison constants |
| Dimension function and monotonic ratio | `f(rho)=4rho^(1/r)` tends to zero and increases; `f/g=4rho^(1/r-1)` is decreasing for rho>0 |
| Radii tend to zero | (9); enumerate by increasing level, with finitely many words per level |
| Balls and centers belong to X | Actual overlaps lie inside their parents; repeated centers/words are permitted |
| Full local measure of enlarged limsup | (11)-(12) give full coverage at every level, a stronger assertion |
| Correct target limsup | Equation (2), counting all words and closed endpoints |

It follows that `H^(1/r)(E_infty intersect J)=infinity` for every nonempty open interval J inside `(0,S)`, because s<1 and a nondegenerate interval has infinite s-dimensional Hausdorff measure. Multiplying the gauge by 4 does not change that conclusion. Together with (10), this gives the claimed exact dimension. **Do not invoke the whole-real-line formulation while proving coverage only inside `[0,S]`.** The compact metric formulation removes that mismatch. No separation or probabilistic independence assumption is needed.

### 3.5 System E: an elementary zero-dimensional certificate

Set `d_n=n+4`, `t_n=2^(-n)`, and

\[
a_n=t_n-\log P_{n+4}(t_n),\qquad
b_n=\frac{2^{-n(n+5)}}{(n+5)!}.
\]

Applying (4)-(6) at each degree gives

\[
(1-t_n)b_n\le a_n\le b_n.
\tag{13}
\]

The lower bound is slightly weaker than the integrated expression but convenient. Exactly,

\[
q_n^*=\frac{b_{n+1}}{b_n}
=\frac{2^{-(2n+6)}}{n+6}\le\frac1{1792},
\tag{14}
\]

and these ratios decrease. The symbol `q_n^*` is a defect-bound ratio, not the rational factor `q_n`. Therefore

\[
\sum_{k>n}a_k\le\frac{b_{n+1}}{1-q_n^*}\le\frac{b_n}{1791}.
\]

Since `a_n>=b_n/2`,

\[
\frac{b_n}{3}\le\delta_n=a_n-\sum_{k>n}a_k\le b_n.
\tag{15}
\]

The exact inequality `1/2-1/1791>1/3` supplies the lower constant. This proves strict coverage directly for E and supplies a cheap numerical tail strategy. An independent coverage derivation from the paper is also available: `P_m(x)P_m(y)>=P_m(x+y)` for nonnegative x,y, strictly for positive x,y; later degrees are at least the current degree. Include that coefficient proof as an audit cross-check, not an additional computational project.

For every integer j>=1 and N>=2j,

\[
\sum_{n\ge N}2^{n-1}\delta_n^{1/j}
\le\frac12\sum_{n\ge N}2^{n-n(n+5)/j}
\le\frac12\sum_{n\ge N}2^{-n}=2^{-N}.
\tag{16}
\]

Thus `H^(1/j)(E_infty)=0` for every j, and `dim_H E_infty=0` once nonemptiness is established below. This is a quantified, universal argument. A finite list of j values is not a proof of it. No MTP is needed for E. The stronger factorial bound in b_n can be retained for numerical work; the simpler exponent suffices in (16).

For all five systems, `P_d(t)>=1+t` and `P_d(t)<=exp(t)`, giving `t-t^2/2 <= w_n <= t`. Consequently `2^n R_n -> 1`, proving the multiplicity integral in (3).

### 3.6 Category and nonemptiness

Reproduce P's Theorem 3.1, pp.4-5, in the report. For every starting index N, any nonempty relatively open interval in the tail interval contains a sufficiently small prefix cylinder. A later strict overlap inside that cylinder gives an open subinterval with at least two tail representations. The unique-tail set is therefore nowhere dense. An isolated point of a compact fiber is isolated by a finite prefix and hence has a unique tail. Values with any isolated fiber point lie in a countable union of translated unique-tail sets. Outside that meager set, fibers are nonempty perfect compact subsets of Cantor space and have cardinality continuum.

All selected systems have strict overlaps at every level. This proves that E's exceptional set is nonempty and contains a comeager continuum-fiber subset. For A-D it does **not** determine the separate dimensions of countably infinite and continuum-sized fibers. Preserve that distinction.

### 3.7 The certificate's trust boundary

The output is a **computer-checked certificate of finite algebraic/numerical obligations, accompanied by an audited analytic proof**. It is not a complete proof-assistant formalization.

Code can independently verify polynomial coefficients and signs, rational constants, tail-ratio arithmetic, containment margins, finite interval enclosures, and structural algebra. The analytic bridges—differentiation/integration, infinite geometric limits, representation trees, monotone convergence, Baire category, the Hausdorff covering argument, the published MTP, and bi-Lipschitz transfer—must be stated and reviewed explicitly. Do not make a program print `proved=true` for these because their names appear in a configuration file.

Record each dependency as `exactly_checked`, `analytically_reviewed`, `source_verified`, `blocked`, or `contradicted`, with its evidence. A dimension conclusion is `supported_conditional_on_listed_mathematical_theorems` only when every necessary dependency is resolved. A claim of complete machine verification would require a different, separately scoped formalization project.

## 4. Why the original numerical challenge needs this reformulation

### 4.1 Audit of the supplied stress test

| Function in S | Valid interpretation | What it does not establish |
|---|---|---|
| `exact_coverage_tests` | Twenty finite rational tail products exceed their current factors, certifying those twenty inequalities | Coverage at every index or for E |
| `scaling_tests` | A two-point overlap-decay slope at n=12 and n=20 | Hausdorff dimension or a rigorously enclosed numerical slope |
| `center_covering_tests` | A floating numerical multiplier at n=20 | All-level containment or MTP applicability; the function prints but does not assert that condition |
| `prefix_search_tests` | A maximum for rounded finite cylinder intervals | A certified exact maximum or any infinite fiber |
| `tail_log(extra=360)` | The mathematical omitted log tail is at most `2^(-(n+360))` | Rounding-error control, or adequacy for a much smaller overlap |

The `searchsorted` principle is valid for exactly known equal-length intervals. Its float64 endpoints are the problem. At n=16, leading overlap widths for m=1,2,3,4 are approximately `7.76e-11`, `5.08e-16`, `2.11e-21`, and `6.67e-27`. Prefix positions are of order one. Late sibling overlaps for degrees 3 and 4 are far below ordinary double-precision absolute spacing there. This does not prove the reported maxima are wrong; it makes them uncertified.

### 4.2 An exact truncation trap to reproduce

S does not implement E. If someone extends its fixed `extra=360` log-tail procedure naively to E, let `K=n+360`. Since every degree is at least 1,

\[
R_K\ge2^{-K}-2^{-2K}/6.
\]

At n=16, `b_n=2^(-336)/21!`, while the omitted tail has that lower bound with K=376. Exact fraction comparison shows the omitted tail exceeds b_n by more than `4.6466941e7`. Since `0<delta_n<=b_n`, the truncated difference `R_n^(K)-w_n=delta_n-R_K` is necessarily negative, even with exact arithmetic.

Preserve this as a negative numerical-method control. It diagnoses a truncation mistake, not a false theorem. E must use defect tails (13)-(15), not a fixed number of ordinary logarithmic tail terms.

### 4.3 Wrong estimators and finite-data nonidentifiability

Because E_infty contains a comeager subset of an interval, it is dense. Its lower and upper box-counting dimensions are therefore 1, even when its Hausdorff dimension is 0 or 1/5. A pixel occupancy plot estimates the wrong invariant. Uniform random sampling also misses E_infty with probability one because the multiplicity integral is finite. Neither method belongs in the primary evaluation.

A stronger obstruction: keep any first N weights and the exact remaining tail sum R_N. Append either (i) the binary tail `R_N 2^(-j)`, j>=1, or (ii) a rescaled strict perturbed-dyadic tail from Section 7. Both preserve every cylinder through depth N. The first has only finite fibers; the second has infinite-fiber dimension 1/r by the audited argument. A finite union of translations preserves that dimension. Thus even exact finite cylinders and an exact tail sum cannot determine the infinite-fiber dimension without an all-level structural assumption. This obstruction concerns general positive summable weight completions; it does not claim that both completions preserve rational factors or the prescribed P_m family. Include this derivation; do not implement an artificial statistical contest to demonstrate it.

## 5. Implementation: minimum architecture and exact interfaces

Use CPU-only CPython, initially version 3.11 or newer, with the standard library: `fractions`, `math`, `decimal` for optional display checks, `json`, `hashlib`, `time`, `resource`, `subprocess`, `platform`, and `unittest`. Record the actual interpreter and OS. Standard-library rational arithmetic avoids a mandatory symbolic algebra system or high-precision native dependency [R11]. Do not mix Python floats into certificate arithmetic. No CUDA, MPS, GPU, remote compute, paid service, or proprietary dataset is needed.

The following is a required structure to create, not an existing repository:

```text
README.md
config/study.json
input/                         # original files, immutable
sources.json                   # URLs, locations, access status, file hashes
proof/audit.md                 # derivations and MTP applicability ledger
proof/claims.json              # dependency statuses and evidence paths
rpa/__init__.py
rpa/families.py                # exact factors and rational reference families
rpa/certificates.py            # generate finite proof obligations
rpa/intervals.py               # independent log/defect enclosure paths
rpa/prefixes.py                # exact closed-interval counts and witnesses
rpa/run.py                     # inspection, calibration, pilot, freeze, confirmation
rpa/report.py                  # deterministic tables and report assembly
verify_certificate.py          # separate restricted arithmetic verifier
tests/                         # meaningful oracle and mutation checks
certificates/                  # rational JSON, no executable expressions
results/                       # append-only run ledger, raw intervals, failures
report.md                      # final research result
```

A few modules may be combined if that improves clarity. Preserve separation between generator and verifier, and between the two numerical log evaluators. Do not create a general plugin framework. The verifier must not import the generator, load cached answers from it, or use `eval` on certificate fields. It may use `Fraction` and integer operations. It must recompute the expected coefficients and constants from the family identifier and parameters.

Required interfaces, with rational endpoints represented as reduced integer numerator/positive denominator pairs:

```python
factor_fixed(m: int, n: int) -> Fraction
factor_growing(n: int, offset: int = 4) -> Fraction
generate_certificate(family_spec: dict) -> dict
verify_certificate(path: str) -> dict
log_atanh_bounds(q: Fraction, width: Fraction) -> tuple[Fraction, Fraction]
log1p_alternating_bounds(x: Fraction, width: Fraction) -> tuple[Fraction, Fraction]
overlap_bounds(family_spec: dict, n: int, route: str, eta: Fraction) -> dict
prefix_max_bounds(family_spec: dict, depth: int, cutoff: int) -> dict
closed_sweep(intervals: list) -> dict
parent_contained(left: Fraction, right: Fraction, center: Fraction,
                 radius: Fraction, r: int, multiplier: int) -> bool
```

Interfaces are proposals you must implement. They have not been tested as commands or code during planning.

### 5.1 Restricted certificate format and independent checker

For A-D, store a schema version, family identifier, `c=[1,1]`, m, r, all polynomial coefficients, D/B/L/U, K=4, the claimed exponent `[1,r]`, and identifiers of the reviewed analytic lemmas. Reject unsupported c, degrees outside 1-4, mismatched r, missing fields, inconsistent coefficients, or unsupported proof templates. An arbitrary supplied polynomial is not silently treated as P_m.

Pseudocode for the arithmetic core:

```text
read certificate; validate schema and bounded family domain
reconstruct P[j] = 1/j! using independent integer factorial code
check each supplied coefficient exactly
differentiate coefficient list; check P - P' = t^m/m!
multiply P by (1-t), subtract 1; check all coefficients <= 0
check P[0]=1 and remaining coefficients >= 0
recompute r, D, B, L=D/2, U=1/r! from integer formulas
check supplied constants equal reconstructed constants
check supplied K == 4 and claimed dimension pair == [1,r]
check B <= D, L > 0, K^r*D/4 >= 1 using the checked K
check formal center identity using R_previous = w + R
emit arithmetic results and unresolved analytic dependency IDs
never infer an analytic dependency's truth from its presence in JSON
```

For E, store the degree rule `n+4`, b_n's factorial shift 5 and exponent polynomial `n^2+5n`, ratio rule (14), lower constant 1/3, claimed dimension pair `[0,1]`, and cover template (16). Check every supplied rule and conclusion field against this restricted specification; verify the ratio algebra and `1/2-1/1791>1/3`. The universal polynomial inequality behind the cover can be checked after substituting `n=2j+h`: `n(n+5)-2jn=(2j+h)(h+5)` has nonnegative coefficients for j>=1,h>=0. The general-degree coefficient identity still needs its stated symbolic/inductive derivation; checking finitely many degrees does not prove it for all degrees.

Produce separate fields for arithmetic verification and analytic audit status. A rejected certificate is an ordinary output with a precise failing obligation, not an exception that gets dropped from the report. Unanticipated exceptions are implementation failures and must be recorded separately.

The CLI consumes `certificates/manifest.json`. Define schema version 1 with exactly five entries keyed A, B, C, D, E. Each entry records its family identifier and a path relative to the manifest directory (`A.json` through `E.json`), plus the actual SHA-256 of that certificate. For an early stop, an entry may instead have `status=not_created` and a nonempty reason. The manifest never supplies a trusted success flag. Verify each file and hash independently, preserve each family's result, and return aggregate arithmetic success only when all five required entries exist and pass. Missing, extra, duplicated, malformed, or failed entries cannot produce an aggregate pass. An aggregate arithmetic pass is still distinct from the analytic dimension conclusion in the claim ledger.

### 5.2 Numerical conventions and failure semantics

Use exact integers/Fractions for all trusted signs, endpoints, tolerances, and stopping tests. Serialize fractions exactly; decimal strings are display-only. Retain closed endpoints and duplicate words. No epsilon-based merging is allowed in the trusted paths.

The default relative enclosure target is `eta=2^(-40)`. For A-D set `tau=eta*L_r*2^(-rn)`; for E set `tau=eta*b_n/3`. This gives roughly twelve relative decimal digits against a proved positive lower bound, comfortably finer than the reported slope deviations around `1e-5`. The dimension decision does not depend on this display precision. Do not tune eta after inspecting favorable or unfavorable results.

Every numerical routine must terminate with one of `certified`, `unresolved_width`, `timeout`, `resource_limit`, `invalid_input`, or `implementation_error`. Never replace an unresolved interval by its midpoint and continue as though its sign were known. Use one bounded refinement when a diagnostic interval straddles a comparison boundary. If still unresolved, retain that status.

## 6. Algorithms for independent finite checks

### 6.1 Route A: separate rational log enclosures

For q>1 rational, put `z=(q-1)/(q+1)`. For J>=1,

\[
L_J=2\sum_{j=0}^{J-1}\frac{z^{2j+1}}{2j+1},\qquad
L_J\le\log q\le L_J+
\frac{2z^{2J+1}}{(2J+1)(1-z^2)}.
\tag{17}
\]

This follows from the positive atanh series and a geometric bound on its remainder. Choose J by exact comparison with the requested width. Handle q=1 separately as `[0,0]`; reject q<1 in this restricted routine. Prove the remainder formula in `proof/audit.md` rather than treating it as a numerical library oracle.

To prevent denominator growth when adding many unrelated fractions, outward-round the final log brackets to a common dyadic grid using integer floor/ceiling. This is exact interval arithmetic. For M logarithms and final target tau, choose grid spacing h=2^(-p) <= tau/(16M); make each raw series remainder <=tau/(8M). Each rounded bracket then has width <=tau/(4M). Compute integer grid sums exactly.

For A-D:

```text
tau = eta*L_r*2^(-r*n)
choose smallest cutoff H >= n+1 with 2^(-H) <= tau/4
M = H-n+1; enclose log q_k for k=n,...,H with the allocation above
delta_lower = sum(lower[k], k=n+1,...,H) - upper[n]
delta_upper = sum(upper[k], k=n+1,...,H) - lower[n] + 2^(-H)
assert enclosure width <= tau; return endpoints, H, p, term counts
```

The omitted tail bound uses `0<log P_m(t)<=t`; it is separate from numerical rounding. Inspect actual integer bit lengths during calibration.

For E, use defects:

```text
tau = eta*b_n/3
choose H >= n with b_(H+1)/(1-q_H^*) <= tau/4
M = H-n+1; enclose log q_k for k=n,...,H by (17)
enclose a_k = 2^(-k) - log q_k with exact outward interval operations
delta_lower = a_n.lower - sum(a_k.upper,k=n+1,...,H)
              - b_(H+1)/(1-q_H^*)
delta_upper = a_n.upper - sum(a_k.lower,k=n+1,...,H)
assert enclosure width <= tau; return endpoints and accounting
```

Although the defect subtracts close quantities, it uses exact rational enclosures with a budget selected for the defect size. There is no floating cancellation. The defect tail decays so quickly that very few later indices should be needed; that is a planning expectation to calibrate, not a measured Mac claim.

### 6.2 Route B: a rational product and an independent log series

For the designated cross-check cases, construct factors independently with the term recurrence `term_0=1`, `term_j=term_(j-1)*t/j`. Verify agreement with the factorial-sum implementation. Form

\[
G=\frac{\prod_{k=n+1}^{H}q_k}{q_n}
\]

as a single exact rational. Choose H by the same **absolute** omitted-tail criterion `2^(-H)<=tau/4`; E uses this route only at its one small designated level. The positive lower bound and this tail tolerance imply G>1. Check `1<G<2` exactly. Failure is a diagnostic to investigate, not permission to change the expected sign.

Set x=G-1. For 0<x<1 the alternating log1p series has even partial sums below `log(1+x)` and odd partial sums above it. Use an even partial sum and the next odd partial sum; refine until their difference is <=tau/4. Add `[0,2^(-H)]` for the remaining log tail and, if needed, outward dyadic rounding with a separately budgeted width. Require total width <=tau.

Route B must not call Route A's logarithm routine. Their certified delta intervals must intersect. A disjoint pair indicates a defect in at least one path, a factor definition, or a proved bound; stop interpretation and isolate it. A Decimal high-precision estimate may help debugging, but cannot overrule a rigorous enclosure.

For every case compare the numerical enclosure with (7), (9), or (15). If wholly contained, record finite corroboration. If disjoint from a claimed bound, preserve the rational witness and diagnose it. If it straddles a boundary, refine once, then report unresolved rather than declaring a theorem false.

### 6.3 Exact finite multiplicity in product coordinates

For A-D, generate all 2^N prefix products `P_u=product(q_j^u_j,j<=N)` exactly. Preserve the word identifier, even when products are equal. At a cutoff H>N, let `T=product(q_k,k=N+1,...,H)`. Since the remaining logarithmic tail is <=2^(-H),

\[
T\le Q_{\mathrm{tail},N}\le T/(1-2^{-H}).
\tag{18}
\]

The upper bound follows from `exp(v)<=1/(1-v)` for 0<=v<1. Prove it using the nonnegative power series coefficient comparison. Sweep both interval families

\[
[P_u,P_uT],\qquad [P_u,P_uT/(1-2^{-H})].
\]

Their maxima bound the true finite prefix maximum. If equal, the maximum is certified. If not, report the bracket and apply at most one predeclared cutoff refinement. Begin with `H=r*N+32`; refine by adding 16. Use starts before ends at equal event positions, because intervals are closed. For tiny exact controls, compare the event sweep with direct membership counts at every endpoint and one rational midpoint between adjacent distinct endpoints.

Return a rational witness in the intersection of the reported lower-bound intervals, preferably in its interior, with the corresponding distinct prefix words. A witness with k valid prefixes certifies at least k representations after the coverage proof. It does not certify infinitely many. Do not label a floating witness or an outer-interval witness a lower-bound certificate.

Do not perform full E prefix enumeration: it adds expensive tiny-width endpoint work without helping its elementary dimension certificate. Its numerical evidence is the defect-route checks and the exact truncation counter-control.

### 6.4 Integral and center checks

Use the exact perturbed-dyadic control below to verify `integral N_N=2^N R_N` by integrating the sweep's piecewise constant counts. Separately verify the telescoping increment `2^n R_n-2^(n-1)R_(n-1)=2^(n-1)delta_n`. Singletons contribute zero measure but retain their pointwise multiplicity. For actual log families use certified R_n intervals; do not integrate product-coordinate counts against ordinary dy and compare them to a dx identity.

Center equality is a symbolic identity in p,w,R and an exact arithmetic check in the rational controls. It is not a regression on approximate centers.

## 7. References, controls, and adversarial checks

These are mathematical/correctness comparators. They are not contestants in an encoding-speed benchmark.

| Role | Construction | Required expected behavior |
|---|---|---|
| Simple reference; zero-overlap control | `w_n=2^(-n)`, `R_n=2^(-n)`, `S=1` | Infinite-fiber set is empty. Interior dyadic rationals can have two representations; zero-width closed overlaps must be counted. Maximum finite prefix count is at most two. |
| Strong exact positive control | For r=2,3,4,5, `w_n=2^(-n)-(1/r!)2^(-rn)` | Exact rational tails and widths; formulas below permit an independent sweep/count oracle. |
| Duplicate-word and finite-only ambiguity control | `w_1=w_2=1/2`; `w_n=2^(-(n-1))` for n>=3 | Only the first inequality is strict; the tail is binary and all fibers are finite. At depth two, prefixes 01 and 10 coincide and must both be counted. The prefix count at x=1/2 is three. |
| Invalid-hypothesis control | `w_n=2^(-n)+(1/r!)2^(-rn)` | Exact negative `delta_n`; reject the interval-filling inference with a rational witness at n=1. Do not assign the theorem's dimension. |
| Geometry ablation | Artificial intervals with the same level counts and radii, all centered at one fixed interior point | Their limsup is one point. Enlarged balls fail full coverage. This isolates why rate and count alone are insufficient. It is an interval-family diagnostic, not another valid representation system. |
| Precision/truncation control | Naive E ordinary-log cutoff n+360 at n=16 | Exact calculation in Section 4.2 must flag unavoidable tail domination, not a theorem counterexample. |

For the strong control, with b=1/r!,

\[
S=1-\frac b{2^r-1},\quad
R_n=2^{-n}-\frac b{2^r-1}2^{-rn},\quad
\delta_n=b\frac{2^r-2}{2^r-1}2^{-rn}=D_r2^{-rn}.
\]

These weights are rational in log coordinates; their exponentials are not claimed to be rational factors. They isolate the overlap mechanism and provide exact finite answers, not an independent proof of MTP or an application result.

Make the geometry control executable. For a rational parent `[a,b]`, supplied center z, radius rho>=0, integer r>=2, and positive multiplier K, `parent_contained` must check

\[
\max\{|z-a|,|z-b|\}^{r}\le K^{r}\rho
\]

by exact arithmetic. This is containment by the enlarged ball of radius `K*rho^(1/r)` without evaluating a root. On the r=2 perturbed control, check the true centers for all parents through level 6 with K=4. Then replace every level-6 center by S/2 while retaining the original radii; require at least one parent-containment rejection. The structural center identity in the valid-family certificate does not substitute for this test on supplied coordinates.

Use a short, fixed mutation suite:

1. Alter D_r's numerator: the arithmetic verifier must reject the formula mismatch.
2. Set the fixed-degree covering multiplier to 1: reject the unsupported parameter; independently check that its rational containment margin would fail. This does not refute a theorem that works with multiplier 4.
3. Delete a duplicate prefix or process end events before start events: the duplicate/binary endpoint oracles must detect the changed counts.
4. Omit the tail allowance: the explicit E truncation diagnostic must prevent treating the resulting sign as a true overlap sign.
5. Remove the MTP source/applicability evidence: the inference layer must withhold A-D's lower-bound status even if arithmetic checks pass. E's independent upper-bound work remains available.
6. Replace actual centers by the geometry ablation: the full-cover/parent-containment obligation must fail.

Do not add dozens of arbitrary perturbations. These cases target distinct plausible errors. A test suite that accepts all positive inputs but cannot reject these mutations is invalid evidence.

## 8. Experiment matrix, metrics, and confirmation protocol

### 8.1 Smallest sufficient case set

No dataset download or random generator is needed. Inputs come from the exact definitions above. There are four fixed degrees and one specified increasing schedule, not random samples from an undefined population of rational systems.

| Workstream | Development/calibration cases | Locked confirmation cases | Purpose |
|---|---|---|---|
| Exact certificate | All five families, analytic dependency review, fixed mutation suite | Rebuild and verify all five from a clean result directory using frozen code/configuration | Establish all-level obligations and reject malformed inference |
| Route A, fixed degrees | m=1..4 at n=1,2,4,8, distributing these cases across tiny/pilot stages | m=1..4 at n=3,5,12,16,20,24,32: 28 cases | Check widths, constants, numerical implementation, and growth of required precision |
| Route B, fixed degrees | m=1..4 at n=1; m=1,4 at n=4; m=4,n=32 for resource calibration | m=1..4 at n=5,12,20,32: 16 cases paired with Route A; m=4,n=32 is a frozen replication | Independent factor/log path on the same targets |
| E defect route | n=1,2,4; n=16 for resource calibration | n=3,6,8,12,16: five cases; n=16 is a frozen replication | Validate rapidly shrinking defects without ordinary-tail cancellation |
| E product route | n=1 only | n=3 only, paired with defect route | Independent small-case cross-check; no need for large product intermediates |
| Actual-family finite prefix sweep | A and D at N=6,8 | A-D at N=10; reduce all to N=8 before freezing if calibration requires | Certified finite multiplicity brackets and rational witnesses |
| Exact-control sweep | Binary and duplicate controls at N=2,3,6; perturbed controls at N=6 | Perturbed r=2,5 at selected final N; all negative/mutation controls rerun | Endpoint, integral, center, and rejection behavior |

Do not expand to N=16 solely to match the old script. No E full-prefix enumeration, raster dimension estimate, random-target multiplicity survey, or countably-infinite-fiber classifier is required. The main arithmetic certificate has no exponential prefix enumeration cost.

There is a hard ceiling of **160 scientific case executions**, counting development cases, repeats, failures, controls, confirmation, and repairs cumulatively. A case is one configured family/level/route or one sweep/control execution. Unit assertions within a test process are not separate scientific cases, but the entire test process's time and memory count. Keep a ledger so rerunning a case never resets its cost. The initial matrix is comfortably below the cap; do not consume the remainder without a specific unresolved risk.

### 8.2 Freeze and independence

Development cases may be used to debug arithmetic, choose bounded data structures, and select N=8 or N=10 according to resource calibration. Before confirmation, save:

- The exact code revision/hash, interpreter, configurations, case list, eta, cutoff rules, resource ceilings, failure classifications, and all exclusions.
- A `protocol_lock.json` with a timestamp and hashes of the code/configuration/proof ledger.
- A list of all already evaluated cases, including planning values if copied into the repository.

Known deterministic mathematical expectations do not require secret test data. Held-out index values primarily catch implementation mistakes. Do not claim statistical independence from the theorem's expected answers. The independent protection comes from different arithmetic paths, analytic controls, and a separately implemented rejecting verifier.

Any later scientific change—families, claimed theorem, exponent, baseline, tolerance policy, or required evidence—makes affected results exploratory and requires a new lock and fresh rerun of the affected confirmation cases. A purely mechanical repair must still be logged and the affected confirmations rerun. All attempts remain in the budget. No repeated search for a favorable slope window is allowed.

### 8.3 Metrics and their interpretation

| Metric | Meaning and required treatment |
|---|---|
| Proof-obligation status | Primary decision variable; exact arithmetic, reviewed analytic deductions, and source checks are separate columns |
| Rational covering margin `4^r D_r/4` | Must be >=1 for the chosen certificate; its exact value is more informative than a numerical p-value |
| Numerical enclosure width and positivity | Width <=tau; report endpoints, cutoff, term counts, grid precision, and bit lengths |
| Cross-route intersection | Must be nonempty; disjoint intervals block further interpretation |
| Scaled overlap enclosure `2^(rn)delta_n` | Compare against the explicit residual band in (7); this is a corroborative coefficient check |
| Prefix maximum bracket and witnesses | A finite count statement only; unequal bounds remain unresolved |
| Source-method failure | Record the naive E cutoff's exact tail-domination witness separately from theorem outcomes |
| Resource usage | Cold elapsed time, selected warm timings, peak RSS, output size, and failure counts establish local feasibility only |

Do not report a “measured Hausdorff dimension,” and do not implement fitted slopes in this PoC. The scaled-overlap enclosures already check the finite behavior that matters. The historical planning slopes in Section 12 are explicitly noncertifying context. Omitting a new slope routine also avoids accidentally applying the atanh series to enormous ratios with impractically slow convergence.

No randomized repetitions, significance tests, or confidence intervals are appropriate for exact deterministic equalities. Numerical uncertainty is expressed by rigorous intervals. Timing variability may be summarized by the individual measurements and a median/range of the small declared repeat set; do not invent population-level confidence from those runs.

### 8.4 Fairness, failures, and scope of comparison

Routes A and B must use identical family definitions, n values, and final absolute tolerances. Report factor construction and any shared preprocessing separately; cold costs include them. An exact rational product route and a float64 sweep do not provide equivalent correctness guarantees, so speed differences between them cannot establish algorithmic superiority.

Retain every timeout, failed assertion, unresolved enclosure, rejected certificate, and missing source as a result row. Report attempted/completed/failed counts by workstream. Do not compare only successful cases or relabel missing runs as passes. A deterministic mismatch requires diagnosis, not averaging. A resource timeout means practical infeasibility for that workload, not a mathematical refutation. If the core proof succeeds but an optional finite maximum remains unresolved, state both facts separately.

## 9. Mac feasibility, ceilings, and calibration

### 9.1 Initial inspection

Inspect the actual machine before experiment execution. Record macOS version, CPU identification, architecture/Rosetta status, physical RAM, available memory/headroom, free storage on the project volume, Python version/build, and the active power mode. Conventional inspection commands include `sw_vers`, `uname -m`, `sysctl -n machdep.cpu.brand_string`, `sysctl -n hw.memsize`, `df -k .`, and `python3 --version`. Verify availability on the actual host; these commands have not been executed on the user's Mac during planning.

Do not infer RAM or precision from “M3 Max.” Use a native CPU Python interpreter if available. No accelerator backend needs to be validated because this design does not use one. Do not change exact arithmetic to float64 to improve speed.

If the target Mac is unavailable, complete the mathematical audit, source checks, code, and documentation that do not depend on it. Any bounded portability/syntax checks elsewhere must be labeled with their real environment. Leave Mac experimental execution and feasibility measurements explicitly blocked; do not replace them with hosted-machine benchmarks or claim the Mac gates passed.

### 9.2 Hard limits

| Resource | Ceiling |
|---|---|
| Cumulative experimental computation | 180 minutes of local elapsed process time, including calibration, tests, development runs, repeats, failures, repairs, and report regeneration |
| Peak resident memory per experimental process | `min(2 GiB, 0.25 * measured physical RAM)`; also require current available headroom before launching |
| Total new project/result disk use | 1 GiB, including temporary intermediates; original inputs are separately recorded |
| Scientific case executions | 160 cumulatively |
| Per numerical case/sweep | 120 seconds initially; raise to at most 300 seconds only after calibration justifies it and before protocol freeze |
| Repairs | At most two cycles, at most 25 cumulative minutes of experimental computation, within the 180-minute total |
| Scope-expanding fallback | None; fallback is certificate-only reporting or a specified analytic stop |

Run experimental processes sequentially. Parallelism is unnecessary here and complicates peak-memory attribution. Use child-process timeouts and a parent watchdog; record native maximum RSS and raw measurement output. On macOS, `/usr/bin/time -l` or an equivalent verified local facility can supply process resource data. Check the local units rather than assuming Linux `ru_maxrss` conventions. If using a sampling watchdog for live memory, record the interval and possible overshoot; the native high-water mark remains the reported peak.

Timing boundaries start before factor/preprocessing construction and end after result serialization for cold end-to-end cases. Warm diagnostic timings may reuse the explicitly recorded cache, but must state what was excluded. During calibration, use one cold and two warm executions for at most four representative workloads. These repeats count toward the case/time limits. No device synchronization is needed for CPU-only work.

### 9.3 Scaling and resizing rules

The family polynomials have degree at most four for A-D, and the certificate expressions have degree at most five; their arithmetic work is tiny and independent of the number of binary prefixes. Numerical costs depend on required bit precision and intermediate integer sizes, not merely the requested number of decimal digits. Fixed-degree direct tails have H roughly proportional to rn plus tolerance bits. E's defect route avoids constructing a long product of increasing-degree factors.

Prefix sweeps hold O(2^N) intervals/events and sort O(2^N) rational endpoints. Rational cross-products and temporary arrays matter; an 8-byte-per-number estimate is invalid for `Fraction` objects. Measure actual high-water memory at N=6 and N=8 with the intended cutoffs and arithmetic.

For two calibration sizes, project the next workload using the larger of the observed growth ratio and the structural increase in event count/bit length, then multiply by a safety factor of two. Permit N=10 only if projected remaining sweep work consumes less than half its stage's remaining time allowance and projected peak RSS is below half the memory cap. Otherwise freeze N=8 for all actual-family sweeps. Do not change degrees, eta, or the analytic question.

Use comparable calibration for Route B at m=4,n=32 and E's n=16 defect calculation. They are explicitly calibration cases in Section 8.1; their later frozen runs are replications, not previously unseen inputs. If the planned finite matrix projects beyond its allowance, omit unnecessary repeats and apply the predeclared certificate-only fallback in Section 10.1 before protocol freeze, preserving completed diagnostics. Do not keep reducing finite samples and call them an empirical substitute for the full theorem.

Maintain at least 35 minutes of the experimental budget and 45 minutes of end-to-end elapsed time for preservation, diagnosis, and reporting. Do not launch a case whose worst-case timeout plus that reserve would exceed the remaining allocation.

### 9.4 Expected effort versus measured runtime

The mathematical and exact-arithmetic workloads are expected to take minutes to less than an hour when the implementation is sound; this is an **unmeasured planning estimate**, not an M3 Max benchmark. The generous 180-minute ceiling covers failures and bounded repairs rather than a required workload duration.

Plan roughly 5-9 hours for setup, implementation, source/proof review, and interpretation, plus the actual experimental runtime. A successful end-to-end pass should fit in roughly 6-10 hours; uncertainty in implementation and source access justifies a hard 12-hour elapsed work-session ceiling. Stop new implementation/experiments with 45 minutes left and produce the best scoped report. Dependency/source setup has a 45-minute allowance; do not spend a workday installing an optional package. These effort estimates are distinct from the workload cap and are not claims about hosted-model latency.

## 10. Evidence gates and autonomous decisions

The experimental allocations below sum to 180 minutes including the shared repair reserve. Unused allowance is not a reason to run more cases. Do not build later-stage infrastructure before its prerequisites pass.

| Stage | Question and prerequisites | Minimum work and evidence | Experimental allocation | Decision and next action |
|---|---|---|---:|---|
| 0. Constructive feasibility | Are the definitions, essential sources, and all-level proof route coherent? | Inspect environment; hash inputs; read P/S; retrieve MTP statement; audit (1)-(16); write claim/dependency ledger and source access record | 10 min | Continue if no fatal gap. Repair a transcription error within shared reserve. If a necessary claim is contradicted, preserve the derivation and take analytic-stop reporting. If only MTP access is blocked, proceed with independent upper bounds/E and mark lower bounds conditional. |
| 1. Tiny correctness | Can a minimal verifier and interval/count routines distinguish true and false cases? | Implement restricted generator/verifier; binary, duplicate, gap, and small perturbed controls; independent tiny sweep; mutation suite; first few development log cases | 25 min | Continue only with resolved oracle discrepancies and effective rejection tests. At most one early repair cycle. A false new bound requires analytic diagnosis before numerical expansion. |
| 2. Resource calibration | Does exact arithmetic fit the measured Mac limits? | Representative A/B routes, E defects, N=6/8 sweeps; one cold/two warm measurements for at most four workloads; scaling projection and chosen final N | 15 min | Freeze N=8 or 10 and per-case timeout. If primary arithmetic itself exceeds limits, report infeasibility. If only enumeration is costly, use certificate-only fallback with the limitation stated. |
| 3. Mechanism pilot | Can the independent paths and controls expose the relevant errors? | Finish the declared development cases; verify common centers/parent containment; reproduce the naive-E truncation failure; reconcile any cross-route disagreement | 25 min | Continue if dependencies and diagnostics are consistent. A noisy/unresolved finite maximum is not a theorem rejection. Allow the remaining bounded repair only for a named uncertainty. No change of exponent window or study question. |
| 4. Locked comparison | Do frozen implementations satisfy the planned exact and numerical checks? | Save protocol lock; regenerate all certificates; run confirmation matrix and controls; preserve every attempt | 45 min | Support only the claims whose obligations pass. Disjoint rigorous intervals stop interpretation pending diagnosis. Missing optional counts are reported as missing, not silently excluded. |
| Shared repair reserve | Can an isolated implementation defect be fixed without changing the scientific question? | At most two documented cycles across all stages; rerun affected cases and record old/new code hashes | 25 min | If exhausted, stop rescue work and classify the remaining result as invalid, blocked, or inconclusive as appropriate. No clock reset. |
| 5. Diagnosis and report | What conclusion does the available evidence justify? | Verify evidence paths; regenerate tables; explain proof trust boundary, failures, resource use, novelty limits, and one justified next step | 35 min | Deliver the completed scoped research outcome, including a valid early stop. Do not start a codec benchmark. |

### 10.1 Predeclared fallback and analytic stop

If a reproducible mathematical obstruction settles a necessary claim, stop later numerical work. Supply the exact counterexample or derivation, its independent check, and the narrowly affected statement. A failed convenient bound may still permit the paper's original proof; spend at most the remaining repair reserve checking that specific alternative. Do not embark on a new theorem search.

The **certificate-only fallback** means a certificate audit with limited diagnostics. Its retained minimum is all five finite certificates and the separate verifier, the complete analytic dependency ledger, the fixed mutation suite, resolved binary/duplicate/gap and small perturbed-weight controls, one paired Route A/B calculation at n=1 for each fixed degree, and one paired E calculation at n=1. Preserve all attempts and explicitly list every unexecuted or unresolved matrix row. If even this minimum cannot be validated, do not claim a validated certificate implementation; report the analytic work separately.

Select this fallback before freezing if the full finite matrix or prefix work cannot fit. Its report may support a mathematically resolved theorem while clearly saying that the full planned numerical audit was not performed. If resources fail after freezing, retain the original plan and label confirmation partial; do not retrospectively remove failed cases. Unresolved optional prefix maxima can coexist with a supported analytic proof. If a critical analytic obligation remains unresolved, label the corresponding dimension assertion inconclusive even if every numerical case agrees. E may be resolved independently of A-D's MTP lower bounds.

If equivalent prior art removes originality, finish only the useful reproduction/certificate work already justified. Do not treat absence of novelty as mathematical falsity, and do not inflate the new explicit bounds into a major breakthrough without a substantive comparison.

## 11. Commands, reporting, and reproducibility contract

### 11.1 Commands the agent must create

The shell commands below define intended interfaces. The environment inspection commands are conventional; the `rpa` commands and verifier do not yet exist and have not been executed. Implement them so that a reader can follow this sequence from an empty project. The trusted core has no third-party package installation step.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m rpa.run inspect --out results/environment.json
python -m rpa.run check --config config/study.json
python -m rpa.run smoke --config config/study.json
python -m rpa.run calibrate --config config/study.json
python -m rpa.run pilot --config config/study.json
python -m rpa.run freeze --config config/study.json --out results/protocol_lock.json
python -m rpa.run confirm --lock results/protocol_lock.json
python verify_certificate.py certificates/manifest.json
python -m rpa.report --results results --out report.md
```

Each command must return a nonzero exit code for implementation failure, a clear status for an intentional blocked/unresolved outcome, and a path to recorded evidence. The `check` command supervises `python -m unittest discover -s tests -v` as a child process, enforces the applicable limits, and appends its resource record to the cumulative ledger. Bootstrap/debug test invocations outside the driver must also be recorded; they are not free work. `confirm` must check the code/configuration hashes against the lock. The separate verifier consumes certificate data and its own fixed mathematical templates, not a generator-provided success flag. Document how to resume after a bounded repair without deleting old rows.

The uploaded NumPy/mpmath script need not be installed or executed to complete the primary study. If you choose to reproduce it literally for context, pin and record its actual dependencies, charge the work to the same budgets, and label its output noncertifying. There is no required SymPy, Lean, Sage, Arb, native extension, or GPU dependency. Do not make access to any of them a hidden blocker.

### 11.2 Raw record schema

Use ordinary JSON/JSONL/CSV. Each run row should contain a unique run ID, stage, case ID, attempt number, code/configuration hashes, start/end times, actual host/interpreter, family and parameters, arithmetic route, tolerance, cutoff, series terms, precision/grid bits, relevant maximum integer bit length, exact interval endpoints or certificate path, expected condition, observed status, wall time, peak RSS with units/source, and error/timeout details. Include preprocessing/cache status where relevant.

For a sweep, also record N, number of words (including duplicates), event count, inner/outer maximum, cutoff refinement, and rational witness/prefix IDs. For an analytic obligation, record its derivation, exact source location if any, reviewer status, and dependent conclusions. A source that was not accessed must not be labeled read.

Retain the original input hashes, downloaded source identifiers/hashes where permitted, retrieval dates, actual dependency versions, all configurations, the protocol lock, and failed attempts. No random seeds are necessary because inputs are deterministic; if you add incidental randomized implementation checks, record their seeds and keep them outside the primary scientific evidence.

Generate tables from raw files rather than hand-copying convenient rows. Recommended essential outputs are the certificate constants/margins table, the claim-status table, interval cross-check summary, finite-count brackets/witness summary, and cumulative resource/failure ledger. A scaled-overlap plot is optional; label it as a finite diagnostic. Do not produce a plot called “measured fractal dimension.”

### 11.3 Required artifacts by reached stage

Always provide `report.md`, `proof/audit.md`, `proof/claims.json`, source/provenance records, and a concise README explaining what was and was not executed. If implementation occurred, provide runnable code and the independent verifier. If experiments occurred, provide their raw results, failures, environment record, and the command that regenerates essential tables. If confirmation occurred, include its frozen protocol. An analytic stop requires the verifiable derivation or exact witness and any minimal reproduction code used; it does not require later benchmark infrastructure.

The final report must answer the primary question first, then explain:

1. Which of the five family claims are supported, contradicted, conditional, or unresolved, and the exact evidence for each.
2. What is checked by exact arithmetic and what remains dependent on reviewed mathematical arguments or published theorems.
3. Whether any observed discrepancy belongs to the paper, the new bounds, the implementation, the original numerical procedure, or access/resource limits.
4. What the closest prior work already establishes and why originality remains unresolved or has been withdrawn.
5. Actual Mac resources and runtime, including failures and setup/preprocessing; or the precise absence of Mac validation.
6. Whether one next step is justified. Prefer specialist review of the exact theorem/novelty comparison after success, or resolution of one named obligation after an inconclusive result. Do not recommend a broad application program without evidence.

### 11.4 Honest outcome labels

| Label | Apply when |
|---|---|
| Supported contribution | The specified certificate/proof claim survives its required checks, with a clear trust boundary and tested scope; mathematical priority is a separate statement |
| Informative negative result | A valid derivation or exact counterexample rules out a necessary claim or a proposed numerical method under stated conditions |
| Practical infeasibility | Measured costs or supported scaling exceed the explicit limits for a required workload |
| Inconclusive evidence | A proof obligation, essential source, hardware validation, or enough diagnostic resolution remains unavailable |
| Invalid experiment | A coding, endpoint, interval, baseline, or measurement defect invalidates an attempted comparison; preserve it and do not recast it as a scientific negative |

These labels can differ by workstream. For example, the naive E cutoff can be invalidated while the E dimension theorem is supported. A finite floating-point disagreement never automatically falsifies an infinite-set theorem. A successful certificate with unverified MTP applicability never automatically establishes the lower dimension bound.

## 12. Planning work actually performed; work still to do

### 12.1 Source and proof audit performed

During preparation, the full 13-page supplied PDF was extracted and read, including its definitions, proofs, numerical section, limitations, references, and appendix. Pages containing the main dimension and tail formulas were rendered and inspected when layout mattered. The extraction was recovered with Python PDF tools after the available Poppler binary failed to load a shared library. The stress script and complete pasted assignment were read. The missing `reproduce.py` was identified rather than reconstructed.

Three independent agent subtasks examined (i) the essential mathematics and MTP application, (ii) the closest literature and novelty, and (iii) numerical design/feasibility. Their findings were reconciled in this brief. This was independent analytical review within the planning session, not external peer review. No proof-assistant formalization or complete PoC was performed.

The derivative certificate, uniform K=4 margin, strengthened E defect bound, finite-data obstruction, and estimator limitations were derived during planning. They are supplied arguments to verify, not preexisting published results whose priority has been established.

### 12.2 Small probes actually performed

The planning environment was **Linux x86_64, CPython 3.12.14**, not an M3 Max. Standard-library `Fraction` calculations checked D/B/L/U and all four exact containment margins in Section 3.3. Exact fraction comparisons checked the tail-domination counter-control for E at n=16,17,20. No Mac timing inference is justified from these probes.

A bounded exact-arithmetic review probe also constructed E's Route B product at n=3 with eta=2^(-40): the cutoff was H=83, its numerator and denominator each had 224,338 bits, and the even-2/odd-3 log bracket already met the tau/4 series-width allowance. The largest bracket integer had 673,015 bits. This checks a concrete intermediate-size risk for that one case; it does not measure target-machine performance or execute the planned confirmation matrix.

A separate standard-library Decimal calculation at 160 significant digits evaluated fixed-degree log factors through index 480 and obtained the following rounded values:

| m | Two-point decay slope, n=12 to n=20 | `1/S_m(1)` | `2^(20(m+1))*delta_20` |
|---:|---:|---:|---:|
| 1 | 1.999962413081 | 1.150911348510 | 0.333333060855 |
| 2 | 2.999964184670 | 1.017290954829 | 0.142857031595 |
| 3 | 3.999963622183 | 1.001899992749 | 0.038888858125 |
| 4 | 4.999962834989 | 1.000179011200 | 0.008064509611 |

These reproduce the reported scale behavior and means to the shown digits. The omitted mathematical log tail is <=2^(-480), but the complete rounding error of this planning calculation was not propagated. It is a high-precision corroborative calculation, not a rigorous enclosure or a dimension measurement. Python documents `Decimal.ln` as correctly rounded, but that does not by itself certify an entire multi-step calculation [R12]. The uploaded script itself was **not** executed during planning.

Everything specified as a repository command, certificate-verifier output, locked experiment, finite multiplicity witness, runtime/memory result on the Mac, or final research report remains work for you. Do not copy planning observations into `confirmed` result rows.

## 13. Decision-relevant literature and novelty assessment

### 13.1 Closest comparisons

The following is a targeted primary-source assessment, not an exhaustive priority search. Avoid repeating it indefinitely during implementation; only follow a new lead if it could change the theorem's attribution or validity.

| Source | Relevant inspected location and established content | Difference that matters here |
|---|---|---|
| Beresnevich–Velani [R1] | Section 3, Theorem 2; Section 6.1, Theorem 3, pp.990-991 | Supplies the trusted dimension tool; the new work must establish the correct representation-specific target and hypotheses |
| Knopfmacher [R2] | Theorem 2.3 and Remarks 2.4, p.292; relevant algorithm discussion p.294 | Binary rational radix products and nonuniqueness already occur; the m=1 building-block construction is not new |
| Głąb–Marchwicki [R3] | Theorem 3.8 and proof | Under all-level interval filling, infinitely many strict inequalities ensure an infinite fiber; this is a direct qualitative predecessor |
| Komornik–Loreti–Pedicini [R4] | Theorem 1.9(ii), PDF p.4 | Their every-interior-point continuum criterion requires `w_n<R_(n+1)`; our ratio `w_n/R_(n+1)` tends to 2, so it does not apply |
| Persson–Reeve [R5] | Introduction and Theorem 1, preprint pp.1-3 | Shrinking neighborhoods of finite expansion sums can have dimension 1/alpha, but also strict drops for exceptional parameters; distribution/coverage cannot be omitted |
| Baker [R6] | Section 3, printed pp.13-14, Theorems 3.1-3.4 | MTP already yields critical Hausdorff results for digit-prefix approximation limsups; our target is infinite multiplicity of a different coding |
| Głąb–Marchwicki [R7] | Introduction and relevant Section 7 statements | Cantorval category behavior is different; the present interval-filling result does not settle broader uniqueness questions |
| Głąb–Prus-Wiśniowski [R8] | Section 6, “Cardinal function” | Current overview of cardinality questions; no equivalent quantitative law was located in that inspected section |
| Marchwicki–Żmija [R9] | Introduction and interval-filling/bounded-range sections | Classifies bounded finite ranges, especially maximal value six; not an identified exceptional-set dimension result |
| Pratsiovytyi–Ratushniak [R10] | Theorem 1, p.111; Theorem 3, p.115 | Perturbed binary overlaps/gaps are established objects, but their alternating gap inequalities violate all-level interval filling |

The most material addition to the supplied paper's bibliography is the pair [R5, R6]. They narrow any novelty claim about the reciprocal-exponent mechanism. Ordinary binary coding's infinite-fiber set is empty, while a limsup of unusually close finite binary approximations can have dimension 1/alpha. Those are different sets; a comparison that conflates them is invalid.

The exact mean-multiplicity identity is a straightforward counting consequence. It is useful as an invariant, not a stand-alone major novelty claim. Explicit finite rational bounds such as (7), (9), (12), and (15)-(16) are the practical improvement selected here. Whether their mathematical packaging merits publication must be judged against the literature, not the fact that a verifier is new.

### 13.2 Search coverage and limitations

Searches covered rational/radix products, achievement-set cardinal functions, infinite subsum representations, multiple Kakeya expansions, perturbed binary/dyadic weights, overlap limsups, mass transference, and beta/lambda expansion approximation. Backward citations from the expansion literature and the current achievement-set survey were followed where relevant, with searches for recent work. General Kakeya searches were filtered to expansion/subsum terminology to avoid the unrelated geometric Kakeya-set problem. No exhaustive citation-database, multilingual, or unpublished-work search was completed.

The MTP publisher PDF, the relevant accessible cardinal-function papers, and the Baker/Persson–Reeve author texts were read in the cited locations. Repeated full-PDF opens of Knopfmacher timed out, but relevant primary-PDF text—including the theorem, nonuniqueness remarks, and algorithm material—was returned by indexed retrieval. Do not describe that as a complete eight-page review. The Pratsiovytyi–Ratushniak paper's English abstract and mathematical statements were checked; its body is Ukrainian, so coverage is narrower than a full-language scholarly review.

Ghenciu, Munday, and Roy [R13] was accessible only through its publisher abstract/introduction/preview. Its complete propositions and proofs were not inspected. The preview concerns general series representations and IFS dimension spectra; an equivalent result hidden in inaccessible material has not been ruled out. The issue is dated December 2026, while the preview was accessible on 29 October; do not confuse issue date with first online publication. The survey [R8] has differing version-header and internal dates; identify the exact retrieved URL and date rather than reconstructing a revision history.

These are originality/access limitations, not automatic blockers for verifying the supplied argument. The defensible wording after success remains: **“The specified theorem has an independently audited proof and reproducible exact certificates; historical originality remains unresolved after a targeted literature review.”**

## 14. Adversarial review findings and dispositions

Three distinct review passes were required in preparing this handoff: mathematical/novelty, experimental, and execution. The following summarizes material findings and the decisions they caused. Your final report should repeat the relevant checks against the code and evidence actually produced.

| Challenge | Disposition in this brief |
|---|---|
| The lower bound invokes a nonconstructive theorem | Keep MTP as an explicit published dependency; verify its precise compact-space hypotheses; do not claim a Python proof of MTP or require a constructive Frostman measure |
| The paper's “complete proof” label could be accepted uncritically | Re-derive the essential counting, coverage, coefficient, category, and dimension steps; planning found no fatal gap but does not waive execution review |
| The mechanism may already be known | Add the closer metric prior art [R5, R6]; narrow the candidate contribution and preserve unresolved priority |
| An assumption can fail for apparently similar weights | Include a gap-producing exact control and reject the theorem's applicability; the result is not for arbitrary rational collections |
| Only favorable finite data can be selected | Freeze the full specified family set, small/large levels, controls, and failures; no fitted dimension or selected slope window is decisive |
| A stronger reference removes an apparent gain | No speed/accuracy superiority is claimed; exact control formulas and an independent rigorous log/product route are the relevant comparators |
| Rate and formal interval count might masquerade as a proof | Require the common-center/full-cover argument; include the concentrated-center ablation and finite-data obstruction |
| Finite precision or tail truncation changes the sign | Use rational enclosures, explicit tail budgets, E defect tails, and the exact n=16 truncation failure control |
| Duplicate prefixes or endpoints corrupt multiplicity | Count words, use closed intervals, and compare an event sweep with exact tiny membership oracles |
| Preprocessing or hidden integer size dominates | Include factor construction and output in cold costs; measure bit lengths, RSS, and scaling before large products/sweeps |
| Memory/runtime exceeds forecasts | Cap cases/time/RSS/disk; resize sweeps before freezing; retain the scientifically sufficient certificate-only fallback |
| A small pilot is unresolved | Diagnose a named uncertainty within two bounded repairs; unresolved finite counts do not refute an infinite theorem |
| Essential source or target hardware is unavailable | Complete unaffected proof/code work and identify the blocked claims; no invented source access or Mac performance |
| The main claim is false | Preserve an exact derivation/counterexample, isolate the affected claim, and stop later work; an honest analytic stop completes the project |
| A report might overclaim formal verification | Separate machine-checked arithmetic from analytically reviewed lemmas and published theorem dependencies in every conclusion |
| Correct arithmetic might accompany an unchecked stored conclusion | Explicitly validate K and claimed dimension fields, require all five manifest entries, and compute margins from the validated multiplier |
| Calibration could be mislabeled as fresh confirmation | Mark the costly preflight cases as development and their later runs as frozen replications |
| Fallback or unmetered tests could hide missing work | Define the retained diagnostic minimum, keep incomplete rows visible, and run tests through the resource supervisor |

Residual risks are the correctness of the analytic glue and verifier implementation, the incompleteness of the novelty search, and unmeasured target-machine costs. They are handled by explicit proof obligations, independent calculations, rejection controls, and resource gates. No assertion of absolute perfection or exhaustive originality is warranted.

## 15. References and exact source locations

All online sources below were retrieved or attempted on 29 October 2026. “Inspected” is limited to the locations discussed above. No long source excerpts need to be redistributed with the code; retain bibliographic metadata and stable links, and respect source download/reuse terms.

- **[P] Supplied research note.** *Rational product expansions with controlled infinite ambiguity*. “An autonomous mathematical investigation prepared with ChatGPT,” 29 October 2026, 13 pages. Attachment `rational_product_ambiguity.pdf`. Essential locations: Definition 1.1 and Theorems 1.2-1.3, pp.2-3; Lemmas 2.1-2.2 and equations (3)-(6), pp.3-4; Theorem 3.1, pp.4-5; Theorem 4.1, p.6; Section 5, pp.6-8. Experiments p.10; literature p.11; limitations/appendix pp.12-13. No public publication identifier was supplied.
- **[S] Supplied script.** `ambiguity_theory_stress_test(2).py`, no author/version metadata beyond its source docstring. Exact bytes identified by the SHA-256 in Section 2. It checks finite necessary consequences and explicitly disclaims proving an infinite-set dimension.
- **[A] Supplied assignment.** `Pasted text(20261029-164203).txt`, exact bytes identified in Section 2. Contains the original motivating problem and execution-brief requirements.
- **[R1] Victor Beresnevich and Sanju Velani (2006).** “A Mass Transference Principle and the Duffin–Schaeffer Conjecture for Hausdorff Measures.” *Annals of Mathematics* 164(3), 971-992. [DOI: 10.4007/annals.2006.164.971](https://doi.org/10.4007/annals.2006.164.971). [Publisher PDF](https://annals.math.princeton.edu/wp-content/uploads/annals-v164-n3-p06.pdf). Critical invocation: Section 6.1, Theorem 3, pp.990-991; dimension-function definitions p.974; Euclidean formulation Theorem 2, p.977. The theorem numbered 3 on p.992 is different.
- **[R2] Arnold Knopfmacher (1990).** “A Radix Product Representation for Real Numbers.” *Fibonacci Quarterly* 28(4), 290-297. [Primary PDF](https://www.fq.math.ca/Scanned/28-4/knopfmacher.pdf). Relevant text: Theorem 2.3 and Remarks 2.4, p.292; algorithm/error discussion p.294. Relevant primary text recovered through indexed retrieval; full PDF opens timed out.
- **[R3] Szymon Głąb and Jacek Marchwicki (2020).** “Cardinal Functions of Purely Atomic Measures.” *Results in Mathematics* 75, article 141. [DOI and full publisher text](https://doi.org/10.1007/s00025-020-01260-x). Critical comparison: Theorem 3.8 and proof; discussion following Corollary 3.9.
- **[R4] Vilmos Komornik, Paola Loreti, and Marco Pedicini (online 2025).** “Multiple Kakeya Expansions.” *Annali della Scuola Normale Superiore di Pisa, Classe di Scienze*. [DOI: 10.2422/2036-2145.202404_009](https://doi.org/10.2422/2036-2145.202404_009). [Publisher/author PDF](https://journals.sns.it/index.php/annaliscienze/article/download/6996/2544/4989). Theorem 1.9(ii), PDF p.4; related Theorem 1.11 and Section 3 proofs.
- **[R5] Tomas Persson and Henry W. J. Reeve (2013; preprint 2012).** “On the Diophantine Properties of λ-Expansions.” *Mathematika* 59(1), 65-86. [DOI: 10.1112/S0025579312001076](https://doi.org/10.1112/S0025579312001076). [Author preprint, arXiv:1202.4904v1](https://arxiv.org/pdf/1202.4904). Inspected introduction and Theorem 1, preprint pp.1-3; definitions distinguish approximation sets from representation fibers.
- **[R6] Simon Baker (2018; online 2017).** “Approximation Properties of β-Expansions II.” *Ergodic Theory and Dynamical Systems* 38(5), 1627-1641. [DOI: 10.1017/etds.2016.108](https://doi.org/10.1017/etds.2016.108). [Accessible author manuscript](https://wrap.warwick.ac.uk/id/eprint/81658/7/WRAP-approximation-properties-beta-expansions-iiBaker-2017.pdf); [arXiv:1506.07782](https://arxiv.org/abs/1506.07782). Section 3, printed pp.13-14 / PDF pp.14-15, Theorems 3.1-3.4. The manuscript's definitions/introduction were also inspected.
- **[R7] Szymon Głąb and Jacek Marchwicki (2023; online 2022).** “Set of Uniqueness for Cantorvals.” *Results in Mathematics* 78, article 9. [DOI and publisher text](https://doi.org/10.1007/s00025-022-01777-3). Inspected introduction and relevant Section 7 statements, including Theorem 7.8 and surrounding category discussion.
- **[R8] Szymon Głąb and Franciszek Prus-Wiśniowski (2026 text).** “Achievement Sets—Current Results and Open Problems.” *Real Analysis Exchange*, advance publication. [DOI: 10.14321/realanalexch.1766383782](https://doi.org/10.14321/realanalexch.1766383782). [Retrieved text, arXiv:2512.17285v1](https://arxiv.org/html/2512.17285v1). Section 6, “Cardinal function.” The retrieved header and internal manuscript dates differ; the URL and retrieval date identify the reviewed text.
- **[R9] Jacek Marchwicki and Błażej Żmija (2025).** “Bounded Ranges of Cardinal Functions.” [arXiv:2508.13016v1](https://arxiv.org/html/2508.13016v1), 18 August 2025. Relevant introduction and interval-filling/bounded-range sections inspected; no equivalent metric dimension result located there.
- **[R10] Alexandr Pratsiovytyi and Sofiya Ratushniak (2025).** “On the Properties of a Certain Perturbed Binary Sequence.” *Bukovinian Mathematical Journal* 13(1), 109-117. [DOI: 10.31861/bmj2025.01.10](https://doi.org/10.31861/bmj2025.01.10). [Publisher PDF](https://media.chnu.edu.ua/media/fqbcveip/109-117_pratsovytyi-ratushniak.pdf). Theorem 1, p.111; Theorem 3, p.115; English abstract. Body-language review limitation stated in Section 13.2.
- **[R11] Python Software Foundation, current documentation retrieved 2026.** [`fractions` — Rational numbers](https://docs.python.org/3/library/fractions.html). Use integer-pair construction and exact rational operations; constructing a Fraction from a binary float preserves that float's value, not the intended decimal rational. Record the actual local Python version.
- **[R12] Python Software Foundation, current documentation retrieved 2026.** [`decimal` — Decimal fixed-point and floating-point arithmetic](https://docs.python.org/3/library/decimal.html), especially `Decimal.ln` and contexts/rounding. This documents the planning display calculation; it is not the trusted numerical certification path.
- **[R13] Andrei E. Ghenciu, Sara Munday, and Mario Roy (2026).** “Representations of Numbers Using Series and the Hausdorff Dimension Spectrum of Self-Similar IFSs.” *Journal of Mathematical Analysis and Applications* 564(2), 130902; issue dated December 2026. [DOI: 10.1016/j.jmaa.2026.130902](https://doi.org/10.1016/j.jmaa.2026.130902). [Publisher preview](https://www.sciencedirect.com/science/article/pii/S0022247X26005147). Only accessible abstract/introduction/preview inspected; full proof content remains an explicit novelty-search limitation.

---

**Execution instruction:** Carry out the authorized stages autonomously, inspect the evidence, and follow the gates. Resolve ordinary implementation choices without requesting permission. Preserve unfavorable evidence and stop when a valid decision or a resource/assumption gate requires it. Deliver the completed scoped research result; do not replace missing proof or hardware validation with optimistic language.

