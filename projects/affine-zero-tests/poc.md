# PoC execution brief: signed affine zero tests and syndrome-decoding state complexity

**Recipient:** an autonomous coding agent configured as `gpt-6.1-sol` with `xhigh` reasoning.  
**Prepared:** 29 October 2026. **Research cutoff:** 29 October 2026.  
**Target:** one local MacBook with an M3 Max, using exact CPU arithmetic.  
**Assignment:** execute the bounded study below, inspect its evidence, and produce the reproducible result. This document is the research plan, not a claim that the PoC has already been completed.

## 1. Project decision and smallest defensible contribution

### 1.1 Execute one focused limitation study

Determine whether the supplied signed polynomial encoding can reduce the state space of **exact forward dynamic programming over its parity features**, compared with dynamic programming over the original syndrome. Permit the signed encoding to remove a common parity factor before the comparison. This inexpensive transformation is essential to a fair test.

The primary question is:

> For a polynomial whose Boolean-cube zero set is exactly the nonempty affine system `He = s`, can its independent Fourier-parity features, after the best common-character shift, yield fewer reachable states than ordinary syndrome coordinates at the same variable prefix and Hamming-weight cap?

The derivation in Section 3 answers **no for this specified mechanism**. Your task is to check that derivation, make it executable, independently validate the necessary constructions and identities, and report the resulting limitation. A small exact diagnostic is sufficient. Do not launch a general decoder-performance campaign after this question has been resolved.

The useful contribution is a reproducible connection between the paper's degree savings and the actual state information required by this particular search method. It comprises:

1. A proof that the best character-shifted feature rank is at least the syndrome rank.
2. An exact projection from the feature DP to the ordinary syndrome DP, including sparse reachable states and weight truncation at every prefix.
3. A degree/rank tradeoff: a representation with syndrome-rank Fourier support cannot retain a degree below the paper's nonnegative threshold.
4. A small, independently checked implementation exposing the difference between polynomial degree, support size, feature rank, and search work.

Treat this as a **supported, route-specific limitation derived from established Fourier principles**, not as a historically new lower bound. Fourier dimension and translation invariance are prior art [R3, Fact 5.2]. The project-specific synthesis and executable diagnostic are the deliverable; publication-level novelty is unestablished.

### 1.2 Why this revision preserves the practical purpose

The original practical question is whether a signed exact-zero representation helps decide

\[
\exists e\in\{0,1\}^n:\quad He=s\pmod 2,\quad \operatorname{wt}(e)\le t.
\]

The paper supplies an exact representation of the affine constraints, but no low-weight zero-search algorithm. A broad benchmark of an unspecified new solver would leave the central mechanism undefined. This brief makes one natural bridge executable: track the parity features that determine the polynomial while processing error coordinates.

The revision tests that bridge and stops if it only reproduces or enlarges the original syndrome state. It does **not** settle whether another algorithm could use the encoding beneficially. It does not address general Gröbner/SOS/SAT/ISD comparisons, cryptographic parameter attacks, or universal polynomial-time decoding. No conclusion here establishes `P = NP` or `P != NP`.

General decoding hardness is established context [R2]; the proof in Section 3 does not rely on a complexity conjecture. Merely expressing syndrome decoding with low-degree equations is already established: quadratic algebraic models exist [R7, Section 4.1]. A lower input degree alone would not be a defensible new algorithmic contribution.

### 1.3 Decision to inform

**Should further implementation effort be invested in a complete-parity-signature DP as the route from the signed degree theorem to faster bounded-weight decoding?**

If the derivation and diagnostic pass, recommend against that route as a source of state compression. A later project would need a specified mechanism outside the scope of this proof, with a new cost argument and appropriate baselines. Do not invent such a mechanism or pursue an unbounded rescue branch under this brief.

## 2. Inputs, source audit, and evidence status

### 2.1 Files and provenance

The planning inputs were:

| Input | Provenance and required handling |
|---|---|
| `affine_zero_tests.pdf` | Ten-page research note, *Exact degree of polynomial zero tests for binary affine systems: Signed witnesses, nonnegative witnesses, and short parity checks*, dated 29 October 2026. No author is identified in the PDF. Receive it as `sources/affine_zero_tests.pdf` if available. |
| `Pasted text(20261029-164659).txt` | User's problem statement and requirements. Its essential mathematical example, purpose, constraints, and requested outcomes are incorporated here; it is not required to run this brief. |

The supplied PDF's SHA-256, computed during planning, is:

```text
18c6be6ab46fc00d8ec0886111959dbb0ad19e3b2bde155a2b5a203d156c4fe1
```

The planning workspace copy was `/workspace/scratch/412ff2cdf931/upload/affine_zero_tests.pdf`. This is provenance, not a path you should assume exists on the Mac. Check the received PDF's hash and record any difference. A different version requires checking the relevant theorem statements before attributing them to the supplied note.

All ten pages, including both appendices and the experiment/literature sections, were read during planning. Important displayed formulas were also inspected in rendered pages. An independent mathematical reviewer found no essential defect in the proofs used here. This is a human-readable mathematical audit, not a formal proof-assistant certification.

The verifier scripts mentioned in the paper were **not attached**. Do not claim to rerun them. Implement the new, smaller diagnostics specified here. Their absence does not block this project: the needed equations and proofs are reproduced below. If the PDF itself is unavailable, complete the self-contained derivation and diagnostics, mark source-version verification unavailable, and request only that PDF if source fidelity must be certified. No private dataset is required.

### 2.2 What the paper actually supports

Use these conventions throughout:

- `H_in` is an `m x n` binary matrix; `s_in` is an `m`-bit right-hand side.
- After consistency checking, `H` consists of `r = rank(H_in)` independent original rows and `s` their corresponding right-hand sides.
- `L = ker(H)` and `C = {e: He=s} = a+L` for any one solution `a`.
- `D = L^perp = rowspan(H)` has dimension `r` over `F_2`.
- A character is `chi_v(e) = (-1)^(v dot e) = product_{i:v_i=1}(1-2e_i)`.
- Polynomial coefficients are real in the theorem and exact integers in this implementation. Fourier masks, dot products, row operations, and coordinate maps use `F_2`.
- Polynomial degree means the degree of the unique multilinear function on the Boolean cube, equivalently the largest Hamming weight of a nonzero Fourier frequency. It is not a count of stored terms or a solver complexity parameter.

For `D_w = span_{F_2}{h in D: wt(h)<=w}`, define

\[
w_{\rm gen}(D)=\min\{w:D_w=D\}
=\min_{\mathcal H\text{ a basis of }D}\max_{h\in\mathcal H}\operatorname{wt}(h).
\]

This is not generally the minimum distance of `D` or the maximum weight of an arbitrary supplied basis.

| Source result | Exact scope and implication |
|---|---|
| Theorem 1, pp. 2–3 | For nonempty affine `C`, common zeros of all degree-`d` real polynomials vanishing on `C` are `a + D_(2d)^perp`; restricting those polynomials to be nonnegative on the cube gives `a + D_d^perp`. |
| Theorem 2, pp. 3–4 | For allowed frequencies `A` containing zero, the corresponding spans are generated by `D intersect (A+A)` and `D intersect A`. This characterizes allowed interactions; it is not an optimization algorithm. |
| Theorem 3, pp. 4–5 | For nonempty proper `C`, minimum signed exact-zero-test degree is `ceil(w_gen(D)/2)`; minimum cube-nonnegative degree is `w_gen(D)`. Given a suitable check basis, witnesses have at most `2r` character terms and coefficients of `O(r)` bits. |
| Theorem 4, p. 5 | `w_gen(D)` equals the maximum, over nonmembers, of the minimum size of a coordinate certificate of nonmembership. The associated nondeterministic quantum-query interpretation does not provide a classical zero-search algorithm. |
| Fixed-degree closure algorithm, p. 6 | Enumerates up to `sum_{j=0}^{min(2d,n)} binom(n,j)` masks and tests membership in `D`. Polynomial for fixed `d`; potentially prohibitive when `d` grows. |
| Appendix A, pp. 8–9 | Gives Fourier concentration and exact-support formulas involving cosets and covering radius. It supplies no bounded-weight intersection routine; computing an optimizing coset is an additional problem. |
| Appendix B, pp. 9–10 | Proves a union-closure/matrix-rank identity. It is unrelated to the selected DP and supplies no decoder guarantee. |

The proof mechanisms are sound under these assumptions: restricted characters are independent across dual cosets; translation averaging removes frequencies outside `D`; nonnegative summands with zero average must each vanish; powers-of-two separator coefficients prevent cancellation. Do not extend nonnegativity beyond the cube or move to a finite coefficient field without a separate proof [R1].

The paper reports 16,484 exact closure checks on 2,823 affine sets, 172 numerical LP checks, and 6,622 checks for each of the two prescribed-frequency closures. These remain **source-reported results**, not reproduced experiments. Numerical LP feasibility outcomes are not exact infeasibility certificates. The planning probes in Section 4 are separate and much smaller.

### 2.3 Compact claim-to-evidence map

| Central claim | Evidence status | Remaining issue / resolving check |
|---|---|---|
| Signed minimum degree can be about half the nonnegative minimum | Proven in the supplied note under its assumptions; audited, not formally verified [R1, Theorem 3] | Exact construction checks; known `w_gen` controls. Do not infer optimality from one chosen basis. |
| A sparse exact signed witness is constructible from given checks | Explicit construction and no-cancellation proof [R1, pp. 4–5] | Test affine phases, redundant rows, signs, and arithmetic. |
| The supplied result implies faster low-weight existence testing | Unsupported by the source | This brief tests only the specified parity-feature DP bridge. |
| Even an optimally shifted Fourier signature has rank at least `r` | Derivation in Section 3, using established Fourier invariance [R3] | Independently check proof and matrix-containment certificates. |
| Feature-DP fibre minima equal syndrome-DP entries at every prefix and cap | Exact derivation in Section 3 | Compare full tiny tables against independent enumeration. |
| The state result is a universal lower bound for all decoders | False interpretation | Explicitly exclude minimal trellises, other sufficient statistics, and other algorithms [R8]. |
| This is a historically new theorem | Not established; core principles are known | Attribute the result as a project-specific corollary/diagnostic. No global novelty claim. |
| The proposed audit fits an M3 Max | Planning estimate only | Inspect the actual laptop and perform the bounded calibration in Section 8. |

## 3. Mathematical mechanism and obstruction to verify

### 3.1 Construct the signed and nonnegative witnesses

For each independent check `h_j`, split its support into disjoint masks `u_j,v_j` with `h_j=u_j+v_j` over `F_2`. Put

\[
\sigma_j=(-1)^{s_j},\qquad
q_j(e)=\chi_{u_j}(e)-\sigma_j\chi_{v_j}(e),
\]

\[
P(e)=\sum_{j=0}^{r-1}2^j q_j(e),\qquad
Q(e)=\sum_{j=0}^{r-1}\bigl(1-\sigma_j\chi_{h_j}(e)\bigr).
\]

Each `q_j` is `0` or `+/-2` and vanishes exactly when its check holds. If the largest nonzero separator index is `j`, its weighted magnitude is `2^(j+1)`, while all preceding magnitudes sum to at most `2^(j+1)-2`. Consequently

\[
P(e)=0\iff Q(e)=0\iff He=s.
\]

With balanced splits, `deg(P) <= ceil(max_j wt(h_j)/2)`. This becomes the minimum signed degree only when the supplied basis achieves `w_gen`. For general inputs, report the degree actually constructed and an upper bound, not global optimality.

Store a dictionary `mask -> integer coefficient`, merge repeated frequencies exactly, and delete zero coefficients **before** measuring support or degree. Do not expand characters into monomials in the `0/1` variables. The sparse description uses polynomial space, approximately `O(nr+r^2)` bits for masks and coefficients; that fact does not bound subsequent search [R1, p. 5].

### 3.2 Allow an optimal common-character shift

Let `A = {v: c_v != 0}` be the actual merged Fourier support of an arbitrary exact witness

\[
P(e)=\sum_{v\in A}c_v\chi_v(e).
\]

For nonempty proper `C`, `P` is neither identically zero nor a nonzero constant, so its support is nonempty. Define

\[
U=\operatorname{span}(A),\quad q_{\rm raw}=\dim U.
\]

Choose the numerically smallest mask `a_0 in A`, including zero if present. This deterministic choice avoids tuning. Define

\[
P^\star(e)=\chi_{a_0}(e)P(e),\qquad
V=\operatorname{span}\{v+a_0:v\in A\},\quad q=\dim V.
\]

Multiplication by a character never introduces or removes a cube zero. Moreover, `V` is independent of which support point is chosen as anchor. For any possible character multiplier `chi_b`,

\[
\operatorname{span}(A+b)=V+\operatorname{span}\{a_0+b\}.
\]

Therefore `q` is the **minimum Fourier-support rank over all character multipliers**, attained by every `b in a_0+V`. No search over `2^n` multipliers is needed. Also `q <= q_raw <= q+1`.

This minimizes rank, not polynomial degree. The chosen shift may increase degree, which must be recorded. Do not call it a globally minimum-degree normalization. Use this best-rank representation for the primary comparison; the raw representation is an ablation.

### 3.3 Rank obstruction

For every `z in V^perp`, every character in `P^star` is unchanged by translation by `z`. Thus `P^star(e+z)=P^star(e)` for every cube point. Choose any `a in C`. Then `P^star(a+z)=0`, hence `a+z in a+L` and `z in L`. Taking orthogonal complements gives

\[
\boxed{D\subseteq V,\qquad q\ge r.}
\]

This is a direct application of Fourier-support/translation-invariance principles [R3, Definition 5.1 and Fact 5.2], with a character shift that preserves the zero predicate. The conclusion is about sufficient **linear parity information**. It remains true even if an oracle supplied the optimal-degree signed witness for free.

The nonempty assumption matters: an inconsistent affine system has empty zero set and can be represented by a constant nonzero polynomial of rank zero. Process consistency first. The rank-zero whole-cube case is also separate.

### 3.4 Exact coordinate projection

Let independent rows of `B in F_2^(q x n)` form a basis of `V`. Write each shifted frequency as `v+a_0 = lambda_v B`, where `lambda_v` is a `q`-bit coordinate vector. Then

\[
P^\star(e)=F(Be),\qquad
F(y)=\sum_{v\in A}c_v(-1)^{\lambda_v\cdot y}.
\]

Because `D subset V`, solve for `T in F_2^(r x q)` satisfying

\[
H=TB.
\]

Check that equality explicitly, bit for bit. `T` has rank `r`. Since `B` has full row rank, every `y in F_2^q` is attainable as `Be`. Global exactness of the zero set therefore implies

\[
\boxed{F(y)=0\iff Ty=s.}
\]

Exactly `2^(q-r)` complete signatures are accepting. That count concerns the entire signature space; a particular Hamming ball need not reach every accepting signature.

Also construct a raw basis `B_raw` for `U` and a map `S` satisfying `B = S B_raw`. This supplies a checkable chain from raw to shifted to syndrome coordinates.

### 3.5 Forward-DP projection, including sparse states and the weight bound

Fix the same ordering of the `n` variables for every arm. Let `B_i,H_i` contain the first `i` columns in that order. For a nonnegative integer cap `t`, define

\[
d_{B,i,t}(y)=\min\{\operatorname{wt}(x):x\in\{0,1\}^i,
B_i x=y,\ \operatorname{wt}(x)\le t\},
\]

with the minimum of an empty set equal to infinity. Define `d_H` and `d_raw` similarly.

Every prefix assignment with signature `y` has syndrome `Ty`; every retained feature state is realized by a prefix assignment. Grouping those same assignments by syndrome gives the exact identity

\[
\boxed{d_{H,i,t}(z)=\min_{y:Ty=z}d_{B,i,t}(y).}
\]

This holds for every prefix, including zero, and for every cap. It also holds from raw to shifted coordinates using `S`. Taking minima and then dropping entries above `t` commutes with projection, so weight truncation does not create a loophole.

Let `N_R(i,t)` count finite entries for representation `R`. Then

\[
\boxed{N_{\rm raw}(i,t)\ge N_{\star}(i,t)\ge N_H(i,t).}
\]

The same domination holds for entries with weight strictly below `t`, which permit an include transition. Thus the specified sparse recurrence cannot save state entries or include-transition opportunities by using these features. Quotienting by `Ty` exactly recovers syndrome DP. An invertible change of basis only relabels states.

This is stronger and fairer than comparing dense arrays of size `2^q` and `2^r`: it compares the actually reachable states at each prefix and cap. It is **not** a theorem that every implementation has greater wall time or actual RAM use; implementations, data structures, and coordinate bit widths can differ.

### 3.6 Degree/rank tradeoff

For any globally exact zero test with support span `W`, the invariance argument gives `D subset W`. If `dim(W)=r`, then `W=D`. Its nonzero supporting frequencies lie in `D`, generate `D`, and all have weight at most its degree `d`. Therefore

\[
\dim(W)=r\quad\Longrightarrow\quad d\ge w_{\rm gen}(D).
\]

Consequently a witness with `d < w_gen(D)` has raw feature rank greater than `r`. A character shift may reduce the rank to `r`, but if it does, its transformed degree cannot remain below `w_gen(D)`.

Do **not** incorrectly infer that the original degree saving always forces `q > r` after shifting. Section 4's motivating example is a counterexample to that stronger assertion.

### 3.7 Limits that must accompany the result

The obstruction covers forward DP retaining complete independent parity signatures, the same variable order, and the same Hamming cap. It allows merging identical signatures, best common-character shifts, invertible basis changes, and exact projection onto syndrome coordinates.

It does not cover arbitrary nonlinear state summaries, a scalar-value algorithm with a separately justified update rule, target-specific or suffix pruning, adaptive search, optimized minimal trellises, different-order comparisons, ISD, SAT, MILP, Gröbner bases, or SOS relaxations. It also need not apply to a polynomial required to be correct **only inside one Hamming ball**. Such a polynomial would be a different construction from the globally exact witness used here.

In particular, `2^r` is not a universal lower bound on decoder memory or minimal trellis width. Minimal code trellises can have smaller state spaces at individual coordinate cuts; Forney's state-space theorem describes them [R8, Theorem 10.2]. No optimized-trellis implementation is needed because this study makes no claim to outperform practical decoders.

## 4. Planning probes already performed and fixed controls

The following are **small calculations performed during planning**, not the requested Mac PoC. The root planning environment was Linux x86-64, Python 3.12.14. Independent reviewers also ran small exact checks. No M3 Max runtime, memory, dependency installation, or full evaluation suite has been measured.

### 4.1 Reproduce the user's 12-bit example

Use zero-based coordinates with group masks

```text
a = 0x00f       # bits 0..3
b = 0x0f0       # bits 4..7
c = 0xf00       # bits 8..11
H = [a XOR b, b XOR c]
s = 0
```

Writing `A=chi_a(e)`, `B=chi_b(e)`, `C=chi_c(e)`, the construction is `P=A+B-2C`. Exact enumeration gave:

| Value of `P` | Assignments |
|---:|---:|
| -4 | 512 |
| -2 | 1,024 |
| 0 | 1,024 |
| 2 | 1,024 |
| 4 | 512 |

All 4,096 assignments were checked; 1,024 satisfied the original equations; there were zero zero-set mismatches. The three nonzero dual words all have weight eight, so `w_gen=8`; the signed degree is four and the minimum nonnegative degree is eight.

The raw frequency rank is three. Multiplication by `C` gives `AC+BC-2`, of rank two and degree eight. The implementation's canonical anchor is instead `a`, giving `1+AB-2AC`; its rank and degree are the same. Do not fail a test because these equivalent gauges have different coefficient dictionaries.

All four right-hand sides must be checked using

\[
P_{s_0,s_1}=A-(-1)^{s_0}B+2\bigl(B-(-1)^{s_1}C\bigr).
\]

An independent planning probe found 1,024 zeros and zero equivalence mismatches for each phase. The minimum weight is zero for `(0,0)` and one for every nonzero phase. The homogeneous example is therefore a representation control; every decoding query with `s=0,t>=0` is trivially YES because `e=0` is allowed.

### 4.2 A control where inflation survives the best character shift

Take `n=4`, `H=[0b0011,0b1100]`, `s=0` and

\[
P=z_1-z_2+2z_3-2z_4,\qquad z_i=(-1)^{e_i}.
\]

There are four zeros among sixteen assignments. Here `r=2`, `q_raw=4`, `q=3`, and the raw degree is one. Gauge by `z_1` to obtain

\[
P^\star=1-z_1z_2+2z_1z_3-2z_1z_4,
\]

of degree two. Exact planning checks verified the zero set, the gauge equality, and all-prefix fibre-min identities for every cap from zero through four. This is a state-inflation control, not a difficult decoding family: independent pair checks are easy to solve directly.

The following final-prefix signature counts were independently obtained by enumerating all error masks with weight at most `t`:

| Instance | Cap `t` | Raw | Best-rank gauge | Syndrome |
|---|---:|---:|---:|---:|
| Three four-bit groups | 0 | 1 | 1 | 1 |
| Three four-bit groups | 1 | 4 | 4 | 4 |
| Three four-bit groups | 2 | 7 | 4 | 4 |
| Three four-bit groups | 12 | 8 | 4 | 4 |
| Two pairs | 0 | 1 | 1 | 1 |
| Two pairs | 1 | 5 | 5 | 3 |
| Two pairs | 2 | 11 | 8 | 4 |
| Two pairs | 4 | 16 | 8 | 4 |

These controls distinguish “degree saving exists,” “a removable sign causes apparent overhead,” and “some overhead remains after that removal.” None demonstrates a speed advantage.

### 4.3 Exact arithmetic and deliberately wrong constructions

Add two error-detection controls:

1. **Cancellation mutation.** Replacing the two-pair coefficients `1,2` by `1,1` gives `P_bad=z_1-z_2+z_3-z_4`. At error mask `0b0110`, both checks are violated but `P_bad=0`. The independent oracle must reject this construction. Record the expected rejection as a passing negative control, not a failed experiment.
2. **Floating-point false zero.** Set `H=I_55`, use `s_j=0` for `j<54` and `s_54=1`, and evaluate at `e_j=1` for `j<54`, `e_54=0`. With unit/empty splits, `q_j=-2` for `j<54` and `q_54=2`. Exact `sum_j 2^j q_j=2`. During planning the following naive binary64 loop returned `0.0`:

```python
qs = [-2] * 54 + [2]
exact = sum((1 << j) * q for j, q in enumerate(qs))
approx = 0.0
for j, q in enumerate(qs):
    approx += float(1 << j) * q
# Planning observation: exact == 2; approx == 0.0.
```

The built-in floating-point `sum` need not reproduce this failure; summation algorithms vary. This fixture is a scalar arithmetic check only: never enumerate `2^55` assignments or run its signature DP. A platform-specific difference in the float result is not a failure of the exact method. Record it and retain the exact-integer requirement.

## 5. Implementation specification

### 5.1 Stack and nonnegotiable conventions

Use CPython 3.11 or newer and its standard library: `int`, `collections`, `itertools`, `hashlib`, `json`, `csv`, `time`, `resource`, `subprocess`, `unittest`, and ordinary file operations. No third-party dependency is necessary. The actual Python installation is unverified until target inspection.

Use arbitrary-precision integers for coefficients, masks, parity, ranks, costs, and equality. There is no numerical tolerance, optimization convergence criterion, floating zero test, modular coefficient shortcut, GPU, Metal, CUDA, remote compute, or paid solver in the scientific path. No expanded polynomial, Macaulay matrix, all-coset table, or SDP matrix is required.

Bit `i` of an error mask is variable `i`; bit `j` of a packed syndrome is row `j`. Compute parity as `(mask & error).bit_count() & 1`. Coefficients are characteristic-zero integers even though masks combine by XOR. Preserve original inputs separately from reduced bases.

### 5.2 Preprocessing and edge cases

Validate binary right sides, row widths, and integer `n,t`. For this study allow `0<=n<=100`. Treat a negative cap as an immediate NO; normalize a cap above `n` to `n` for structural diagnostics. Malformed data are `INVALID_INPUT`, not NO.

Perform exact augmented elimination on `(H_in,s_in)`:

- If a dependent row reduces to zero with right side one, return `INCONSISTENT` and the corresponding XOR-of-input-rows certificate. This is a legitimate NO for decoding; do not enter the nonempty-set theorem.
- Retain the indices of independent **original** rows and their original right sides as `(H,s)`. Elimination may use residual rows internally, but do not silently replace the construction basis with RREF rows of different weights.
- Record how every omitted row is represented by retained rows and verify its right side. This guards against changing the affine set.
- For `r=0` and consistency, use `P=Q=0`, empty support, `q=q_raw=0`, and answer YES for nonnegative caps. Mark degree as not applicable for the zero polynomial; do not invent an ordinary finite degree.
- For `s=0`, ordinary decision mode may return the zero witness immediately. Structural audit mode deliberately completes the requested prefixes, since early stopping would make state comparisons incompatible.

Selecting a basis or computing coordinate maps uses only polynomial-time binary linear algebra. Finding an optimal `w_gen` basis is not required. On locked cases with `r<=6`, you may compute `w_gen` by enumerating all `2^r` dual words, sorting by `(weight,mask)`, and greedily adding independent words until rank `r`. This step is required only when the report uses “optimal degree” for that instance. Charge it to preprocessing and label its exponential dependence on `r`.

### 5.3 Deterministic polynomial construction

```text
build_witnesses(H, s):
    P = empty integer coefficient dictionary
    Q = {0: number_of_rows(H)}
    for j, h in input-preserving independent row order:
        support = increasing list of set-bit indices of h
        u = mask of first ceil(len(support)/2) indices
        v = h XOR u
        sigma = +1 if syndrome bit j is 0 else -1
        P[u] += 2^j
        P[v] -= sigma * 2^j
        Q[h] -= sigma
    merge and delete zero coefficients
    return P, Q, all (h,u,v,sigma) records
```

For any coefficient dictionary, direct evaluation is

```text
evaluate(coefficients, e):
    total = 0                         # exact integer
    for (mask, coefficient) in dictionary:
        sign = -1 if popcount(mask AND e) is odd else +1
        total += coefficient * sign
    return total
```

Gauge by `a_0=min(P.keys())` and replace every frequency `v` by `v XOR a_0`, retaining its coefficient. Compute the raw and gauged bases, frequency coordinates, and matrices `S,T` described in Section 3. Check `B=S B_raw`, `H=T B`, and all frequency-coordinate reconstructions exactly. Failure is a correctness error; never quietly fall back to a guessed map.

### 5.4 Exact bounded-weight DP

Use a sparse dictionary keyed by the packed signature. A value is `(minimum_weight, witness_mask)`, with lexicographic tie-breaking on that pair. A single witness mask per entry makes positive answers independently checkable without storing every predecessor table. Its storage is part of the measured memory.

```text
forward_dp(rows R, n, cap t, order, budget):
    d = {0: (0, 0)}
    emit completed-prefix record for prefix 0

    for i in order:
        c = packed column i of R
        next = shallow_copy(d)           # exclude this variable
        for y, (w, witness) in d.items():
            if w < t:
                y2 = y XOR c
                candidate = (w+1, witness OR (1 << i))
                if y2 not in next and len(next) == budget.state_cap:
                    return STATE_LIMIT, last_completed_prefix
                if y2 not in next or candidate < next[y2]:
                    next[y2] = candidate
            periodically check deadline, state cap, and memory budget
        d = next
        emit completed-prefix record
    return COMPLETE, d
```

Use two buffers to preserve the specified 0/1 recurrence and avoid mutating a dictionary while iterating over it. Check the state ceiling before every new-key insertion; check deadline and memory at least every 1,024 attempted updates and after each prefix. Preserve the last completed prefix if interrupted. Do not silently convert a timeout or state limit to NO.

For syndrome DP, read the entry at the packed target `s`. For the raw/gauged arm, evaluate its polynomial in the corresponding parity coordinates on each final retained signature; take the lexicographically smallest value among zero signatures. Independently compare the gauged zero predicate with `Ty=s`. For a YES result, verify the returned error against **every original row**, its weight cap, and the polynomial. A NO result is justified only by a complete exhaustive DP or the independent exhaustive oracle, never by resource exhaustion.

```text
quotient_min(table, linear_map M):
    projected = {}
    for signature y, value in table:
        z = M*y over F_2
        projected[z] = lexicographic_min(projected.get(z), value)
    return projected
```

At each completed prefix, require equality of the weight maps after raw-to-gauge and gauge-to-syndrome projection. Lexicographic witnesses can also be compared if all arms use the prescribed tie rule, but the mathematical claim only needs the weights. Stream the three current tables through one audit driver; retain at most current/next tables per arm, not a list of all historical tables. Keep prefix counts and equality results in logs.

The nonnegative `Q` has frequency span `D`, so its feature DP is a relabeling of syndrome DP. Verify that mapping and its terminal zero predicate. Reuse the syndrome recurrence/table for this control after the isomorphism is checked; do not build a fourth optimized solver.

### 5.5 Independent finite oracle

For `n<=16`, enumerate every error mask once and evaluate the original system directly. Independently build, for each arm, maps from full signatures to the lexicographically least `(weight,witness)`. Filtering those maps by `weight<=t` gives the expected final DP maps for every requested cap without rerunning the enumeration. Compute the true minimum weight for the requested syndrome and retain one witness.

For the fixed four-variable cases, independently enumerate every prefix assignment as well; compare all prefix maps. Use scalar per-coordinate parity in these tiny tests to guard against a common packed-bit convention error. For larger tiny cases, full-cube zero-set comparison, final oracle maps, and all-prefix projection checks are sufficient; a second implementation of the same DP is not an independent oracle.

For `q<=12`, enumerate **all** signature masks to test `F(y)=0 iff Ty=s` and count `2^(q-r)` accepting signatures. This includes signatures excluded by a small cap. Do not infer this count from only a truncated DP table.

### 5.6 Interfaces to create

Keep the repository small:

```text
README.md
config.json
src/__init__.py
src/gf2.py          # augmented consistency, bases, coordinates, certificates
src/zero_tests.py   # sparse P/Q, exact evaluation, character gauge
src/diagnostic.py   # signature maps, DP, quotient, independent enumeration
src/instances.py    # fixed fixtures and deterministic generators
src/run.py          # stage runner, subprocess limits, cumulative ledger
src/report.py       # regenerate tables and the research report
tests/test_fixed.py
tests/test_oracle.py
sources/           # supplied paper if received; source/access manifest
results/           # append-only records and compact instance data
REPORT.md
```

Use ordinary functions and dictionaries; typed dataclasses are optional. The essential interfaces are:

```text
reduce_system(rows, rhs, n) -> consistency, kept_rows, kept_rhs, certificates
build_witnesses(rows, rhs) -> P, Q, split_records
compile_signatures(P, rows) -> raw_basis, gauge_basis, anchor, S, T, coordinates
audit_instance(instance, config, budget) -> summary, prefix_records, failure
enumerate_oracle(instance, representations) -> exact full maps and truth
render_report(results_directory, output_path) -> Markdown report and CSV tables
```

You must create these CLI commands; they have **not** been implemented or verified during planning:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m src.run inspect --config config.json
python -m src.run fixed --config config.json
python -m src.run smoke --config config.json
python -m src.run pilot --config config.json
python -m src.run freeze --config config.json
python -m src.run locked --config config.json
python -m src.report --results results --output REPORT.md
```

The stage runner must invoke the relevant `unittest` tests and count their compute in the same ledger. Document `python -m unittest discover -s tests` as a manual diagnostic, but manual executions must also be entered in the ledger. `inspect` implements Stage 0's environment checks; `fixed` runs Stage 1; `smoke` runs exactly the three Stage 2 calibration jobs; `pilot` runs Stage 3; `locked` runs Stage 4. `freeze` writes a manifest containing the code hash, configuration hash, generation keys, enabled cells, and analysis rules; later runs validate it. `src.report` must itself open and close a reporting-stage ledger entry so the standalone report command also counts toward the common budget.

## 6. Experiment design, references, and ablations

### 6.1 Primary quantity and decision threshold

The primary quantity is the **exact** per-prefix difference `N_star(i,t)-N_H(i,t)` after optimal character shifting. The hypothesized benefit would be a negative difference. The practical motivation is to reduce the number of states that the proposed recurrence must retain; reducing feature rank by one would halve a full dense table.

No arbitrary percentage speedup threshold is appropriate: the proof rules out even one fewer reachable state under the specified conditions. A valid derivation plus zero exact diagnostic discrepancies supports the scoped limitation. A reproducible negative difference with a globally valid witness would contradict the derivation and requires investigation; do not average it away or declare success from timing.

Secondary quantities explain the mechanism: degree before/after gauge, support count, coefficient bit lengths, `r,q_raw,q`, complete accepting-signature count, prefix state/transition counts, and construction/diagnostic resource use. Report timed costs only as feasibility measurements, not as evidence of solver superiority.

No inferential statistical test, confidence interval, timing repetition campaign, or random train/test performance estimate is needed for these exact identities. Generated matrices are implementation-coverage units; their seeds are not repeated measurements of an unknown mathematical inequality. A finite suite supports implementation confidence; the proof supplies the general result.

### 6.2 Comparator roles

| Role | Implementation | Why sufficient for the retained claim |
|---|---|---|
| Independent simple reference | Full enumeration of at most `2^16` errors using original equations | Gives exact yes/no answers, witnesses, minimum weights, and final maps. |
| Exact algorithmic comparator | Sparse bounded-weight syndrome DP with identical order/cap | The closest algorithm to the tested feature-DP mechanism; projection is exact. |
| Proposed route | DP over best-character-shifted Fourier coordinates | Gives the signed approach its inexpensive rank-minimizing normalization. |
| Targeted ablation | Raw Fourier-coordinate DP | Isolates apparent overhead caused solely by an irrelevant common sign. |
| Nonnegative representation control | `Q`, verified isomorphic to syndrome DP | Separates degree differences from a different search recurrence. |

Do not include ISD, a general exact SAT solver, a minimal-trellis solver, or published runtime numbers in this diagnostic. They would be required for a future claim about decoder speed, but they cannot change the exact quotient identity and are not weak comparators being substituted for such a claim. No claim about fastest practical decoding is retained.

### 6.3 Mandatory fixed suite

Run the controls in Section 4 and the following edge cases. Each tests a specific possible implementation or applicability failure:

| Fixture | Required observation |
|---|---|
| `n=3`, no equations, empty RHS | Whole cube; zero polynomials; ranks zero; YES for `t>=0`. |
| `n=2`, rows `[1,1]`, RHS `[0,1]` | Inconsistent; explicit dependency contradiction; no application of Section 3. |
| `n=4`, identity rows, syndrome `0b1011` | Singleton; true minimum weight three; decisions NO at two, YES at three. |
| `n=6`, one all-ones row, each RHS bit | Signed degree three; nonnegative degree six; raw/gauge ranks two/one. |
| `n=6`, rows `[0b000001,0b111110]`, packed RHS `0b11` | Dual nonzero weights 1,5,6; `w_gen=5`, not its minimum distance one. |
| `n=5`, rows `[0b00011,0b01100,0b01111]`, RHS `[1,0,1]` | Consistent redundant row, repeated column patterns, and a zero column; preserve the affine set and the unused coordinate. |

For all consistent fixed fixtures with `n<=12`, enumerate the entire cube and run caps `{0,1,2,n}` plus any explicitly named boundary caps, deduplicated. Run all caps `0..4` for the two-pair fixture. Apply every original row in the oracle, including redundant rows. The deliberate bad polynomial must be rejected before comparing state claims for it.

### 6.4 Development, pilot, and locked implementation audit

The locked suite is a bounded regression audit, not an empirical discovery search. Use two families:

- **Dense:** sample `n`-bit rows uniformly, retaining successive linearly independent rows until rank `r` is reached.
- **Sparse-overlapping:** sample a uniformly chosen four-element subset of coordinates for each candidate row, retaining independent rows. These are generic sparse checks, not a claim to model every LDPC construction.

Use cells `(n,r) = (8,3), (12,4), (16,6)`. Generate four locked instances per family/cell, for **24 locked matrices**. These include dense inputs unrelated to the hand-crafted examples and sparse overlapping inputs where supplied checks are short. They establish coverage of signs, dependencies, overlaps, and feature maps; they do not estimate cryptographic-code performance.

Generation must be deterministic and independent of results:

```python
key = f"affine-signature-audit-v1|{split}|{family}|{n}|{r}|{index}"
seed = int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest(), "big")
rng = random.Random(seed)
```

Use the exact family strings `dense` and `sparse4`. Use `split=dev` and indices `0,1` only for development/pilot; use `split=locked` and indices `0,1,2,3` for confirmation. Store the actual rows and packed right sides, not only seeds. Draw a nonzero syndrome uniformly by rejection from `rng.getrandbits(r)`; the full-row-rank system is always consistent. Do not plant a secret error, filter for favorable degrees/ranks, or select by observed decoding cost.

Stop generation after 10,000 candidate draws for any instance; record `GENERATION_LIMIT` and retain its prescribed slot as missing. Do not search more seeds to replace an unfavorable instance. The cap is a generous implementation guard, not a statistical filtering rule.

Use development instances only in the two smaller cells, with at most eight generated development matrices total. The single additional large calibration instance is exactly `dev|dense|16|6|0` under the key format above; its full cube is counted in the resource budget and it never becomes a locked instance. The two smaller calibration jobs are `dev|dense|12|4|0` and `dev|sparse4|12|4|0`. The pilot uses `dev|dense|8|3|1` and `dev|sparse4|8|3|1`; inspect existing fixed-fixture records for its other controls. Reuse an already completed identical development diagnostic by checking its code/config/input hashes; any deliberate rerun counts again in the ledger.

For every generated matrix, including development, calibration, pilot, and locked matrices, run caps `{0,1,2,n}` and the oracle-derived boundary caps `{max(0,w_min-1),w_min}`, deduplicated. This deliberate boundary construction tests both sides of an exact decision; the solver receives only `(H,s,t)`, never `w_min` or the oracle witness. It is not evidence of unbiased average-case runtime. One exhaustive pass supplies the oracle for all caps.

Use natural coordinate order for the locked suite. As a single robustness check, rerun the fixed 12-bit and two-pair cases in reverse coordinate order. All arms must share the changed order. The proof applies to any common order; no order is tuned separately for an arm.

### 6.5 Freeze and analysis rules

Before the locked run, freeze the hypothesis, proof, source mapping, arithmetic, canonical split/gauge, generator, cells, caps, comparator roles, resource policy, and report rules. Calibration may reduce the number of locked indices to `{0,1}` in **all** cells before freezing, for twelve matrices total. This is the only predeclared workload reduction. Do not remove a family or a cell because its result is inconvenient.

Do not inspect held-out outcomes to make implementation or scientific choices. If a correctness defect appears during confirmation, preserve the failed attempt, repair only within the common repair allowance, and rerun affected checks. Label repaired same-instance results as regression verification. If the hypothesis, generator, primary metric, or algorithm changes materially, the old confirmation is exploratory; fresh confirmation would require the reserved `locked2` keys and available budget. All distinct confirmation matrices, including abandoned slots and `locked2` matrices, count cumulatively toward the 24-matrix ceiling; repeating identical inputs also consumes job-attempt and time allowances. After a full 24-matrix run, fresh confirmation is unavailable under this brief. There is no automatic budget extension, and normally the proper action is an honest partial report.

Every requested instance/cap/arm must appear in the result accounting. Distinguish `COMPLETE`, `EXPECTED_MUTATION_REJECTION`, `INVALID_INPUT`, `INCONSISTENT`, `CORRECTNESS_FAILURE`, `TIME_LIMIT`, `MEMORY_LIMIT`, `STATE_LIMIT`, `GENERATION_LIMIT`, and `NOT_RUN`. A resource-limited arm has no NO answer. Never compare only completed favorable runs.

## 7. Resource feasibility and measurement on the Mac

### 7.1 Initial inspection

Before allocating experiments, record `sw_vers`, `uname -m`, `sysctl -n machdep.cpu.brand_string`, `sysctl -n hw.memsize`, Python version/executable, logical CPU count, free project-volume space, and current memory pressure. If a particular command is unavailable, record that fact and use `platform`, `os`, or `shutil.disk_usage` for the corresponding information. Do not infer RAM, storage, OS, architecture mode, or backend from the phrase “M3 Max.”

Use one foreground worker process at a time, CPU-only. Record whether Python is native `arm64` or running through translation. Check `sys.float_info` only for the deliberately wrong floating-point fixture; scientific arithmetic uses integers. No acceleration/backend fallback is necessary.

If the actual target Mac is unavailable, complete source review, derivation, code, and tiny correctness work on the available host, identifying that host explicitly. Mark M3 inspection/calibration and target resource validation blocked. Do not relabel local-host timings as M3 results. Missing target access does not invalidate a hardware-independent proof.

### 7.2 Hard ceilings and planning estimates

These are project limits, not measured performance predictions:

| Resource | Ceiling / policy |
|---|---|
| Cumulative experimental execution | **60 minutes** across all local checks, development trials, setup diagnostics, failed attempts, calibration, repairs, locked work, and report computation. One persistent ledger; stages and retries never reset it. |
| Reporting reserve | Last **10 minutes** of that total are reserved for preserving evidence and generating the report. Stop new experimental jobs at 50 minutes. |
| Per child job | **60 seconds**, including its preprocessing, oracle, DP, and checks. A job is one fixed fixture or one generated matrix with its prescribed caps, not one tiny arm used to evade the limit. |
| Process memory | `M = min(1 GiB, floor(0.10 * inspected physical RAM))`; if current memory pressure leaves less safely usable memory, lower `M` and record it. Stop on pressure or cap violation. |
| DP states | At most **65,536** entries in any current or next table. The locked maximum raw rank is at most twelve, so this guard is deliberately above the intended state count. |
| Disk | **250 MiB** for code, inputs, logs, and results. Stream summaries; do not save full cubes or every historical DP map. |
| Exhaustive oracle | `n<=16`; never extend it for the identity-55 fixture or a large illustrative instance. |
| Matrix/run counts | At most 24 locked matrices, 8 smaller development matrices, 1 larger calibration matrix, the specified fixed fixtures, and the two reverse-order checks. At most 80 child-job attempts total, including failures/repeats. |
| Repairs | At most **two** ordinary implementation-repair cycles, **15 minutes** of experimental reruns and **60 minutes** of engineering effort in total, all within the other caps. |

Expected engineering/setup/analysis effort is roughly **4–6 hours**, including the bounded repair allowance; this is an unmeasured planning estimate. Reserve the last hour of a six-hour autonomous work session for interpretation and delivery. The 60-minute compute cap is separate from that effort estimate. Stop work on new machinery if it jeopardizes the report. Research browsing and ordinary software downloads are not experimental compute, but setup time is part of the end-to-end effort estimate.

For nonempty proper systems (`r>0`), the intended sparse tables are small: with at most `2r` Fourier terms, `q_raw<=min(n,2r)` and `q<=min(n,2r-1)`. Thus locked `r<=6` gives at most 4,096 raw signatures. Rank-zero systems use the separate conventions in Section 5.2. Python object overhead and witness masks still must be measured. Keeping all prefix tables would introduce unnecessary `O(n 2^q)` storage and is prohibited.

By contrast, a dense full table with `r=32` contains `2^32` entries: even four bytes per entry would require 16 GiB for one table, before working copies or Python overhead. This is an exact capacity calculation, not a claim that every decoder must allocate that table. It explains why the proposed `n=40–100` broad decoding campaign is not an appropriate default for this diagnostic.

### 7.3 Calibration and bounded resizing

Calibrate the three exact generation keys specified in Section 6.4: dense and sparse `n=12,r=4`, then dense `n=16,r=6`. Measure construction time, oracle time, DP/update counts, validation overhead, peak RSS, and output bytes. Do not extrapolate a speedup between arms from these measurements.

Predict the locked suite conservatively as twice the sum of per-cell costs estimated from measured `2^n` oracle work and observed DP update costs. For unmeasured cells use the measured upper-cell cost and the worst-case state envelope `2^(2r)`, rather than an optimistic average-rank assumption. Include all caps, startup, quotient validation, logging, and the two reverse-order checks. The factor of two is headroom for ordinary process and workload variability, not a confidence interval.

Freeze the full 24-matrix suite only if its projected execution is at most **8 minutes** within Stage 4's 10-minute ceiling, its projected total use leaves the reporting/repair reserves, each job is predicted below 30 seconds, and calibration peak RSS is below half `M`. Apply the same admission limits to the predeclared twelve-matrix suite if resizing is necessary. If even that fails the limits, complete the fixed-fixture analytic diagnostic and report why the larger regression audit was omitted. The theoretical conclusion does not depend on broad random coverage.

Do not reduce exactness, alter integer arithmetic, weaken original-equation checks, or turn the problem into a planted-only benchmark to fit the machine. If the mandatory small fixtures cannot be validated within the caps, classify implementation validation as incomplete; do not claim a completed computational result.

### 7.4 Timing and memory boundaries

The purpose of measurement is feasibility and reproducibility. Run jobs in fresh serial subprocesses. Record parent-observed `perf_counter_ns` wall time for the entire child and child-measured phases: construction/preprocessing, oracle, each DP, projection checking, and output serialization. Include basis selection, optimal-degree enumeration when used, and signature compilation in total costs. Report cold process startup separately; do not mix warmed and cold figures into a speed claim.

Record raw `resource.getrusage` peak RSS and the OS-specific units. On macOS normalize bytes; on Linux normalize the reported KiB to bytes. Cross-check the convention once against the platform's process/time tool if available, and retain the raw value if units cannot be verified. The parent should sample current child RSS periodically to enforce the budget, while the child's final high-water mark supplies the peak. Use process exit/timeout status as well as the last completed prefix. No device synchronization is applicable.

There is no amortized multi-syndrome speed claim. Each instance's total preprocessing is charged to that instance. Reusing one exact oracle pass across its caps is allowed and must be disclosed; all DP arms receive the same cap schedule. Validation costs may exceed solver costs and must remain visible.

## 8. Evidence gates and autonomous actions

The stage allocations below sum to the 60-minute execution ceiling, including a shared repair reserve. They are maxima, not targets to spend. Unused time may remain unused; it does not authorize extra projects.

| Stage | Question and prerequisites | Minimum work / evidence | Compute cap | Decision and next action |
|---|---|---|---:|---|
| 0. Constructive feasibility | Is the global zero-set construction valid, and is the proposed state bridge fully specified? | Inspect host and source version; review Section 3 proof; implement exact masks/coefficients and check small matrix certificates. Write a short proof note. | 5 min | Continue if assumptions and proof hold. If an essential mathematical step fails, investigate within one repair allowance; otherwise stop with the counterexample or unresolved step. Do not build later benchmark infrastructure first. |
| 1. Tiny correctness | Do construction, affine phases, gauge, and DP agree with independent truth? | Mandatory fixed suite, mutation rejection, scalar precision fixture; exact expected counts and original-row checks. | 10 min | Zero unexplained discrepancies required. Fix ordinary bugs within the two-cycle shared allowance. A validated theorem counterexample requires an analytic report, not benchmarking. |
| 2. Resource calibration | Can the bounded audit run on the actual host? | Three calibration jobs specified in Section 7.3; time/memory/work estimates; choose 24, 12, or fixed-only scope. | 5 min | Continue with the frozen feasible size. If target hardware is absent, retain host-specific results and mark the target step blocked. If even fixed work fails, report incomplete validation. |
| 3. Mechanism pilot | Does the diagnostic detect both true degree savings and removable/residual state inflation? | Inspect raw/gauged/native tables for the two fixed examples; one dense and one sparse development case; verify every prefix projection and acceptance map. | 5 min | If exact identities hold, the mechanism question is resolved analytically. Continue only to the small locked regression audit if feasible. A noisy runtime comparison is irrelevant and cannot justify extra tuning. |
| 4. Locked audit | Does the implementation remain correct on previously unused matrices? | Freeze manifest; run 24 or 12 matrices once; reverse-order fixed checks; preserve every status. | 10 min | Zero discrepancies supports implementation validity for executed cases. On a defect, preserve runs and use remaining bounded repair capacity; no open-ended reseeding. Omit this stage after a valid fixed-only analytic stop. |
| Shared repair reserve | Can a specific ordinary defect be repaired without changing the project? | At most two cycles; reproduce, fix, rerun affected tests, record deviation. | 15 min | If exhausted, stop and report invalid/incomplete empirical work. It never converts a bug or timeout into a mathematical refutation. |
| 5. Diagnosis and report | What is actually established, and what remains open? | Regenerate tables from raw records; proof/certificate audit; outcome classification; concise recommendation and limitations. | 10 min | Deliver the completed or properly scoped early-stop result. Always preserve this stage. |

When calibration passes, execute the frozen small locked audit as implementation verification. Choose fixed-only analytic completion when the stated resource/access gate or engineering deadline prevents that audit, or end earlier when a decisive verified mathematical defect/counterexample resolves the project. A fixed-only stop after the derivation and essential exact checks is a valid completed research outcome for the selected question. State clearly that the locked random audit was not run. Such a stop does not establish target-machine performance or broad empirical reliability. This policy never requires a broader decoder benchmark.

If you think you found an advantage, first check that the polynomial is globally exact, dependencies and phases are correct, masks were merged, the common-character shift was allowed, caps/orders match, and projection checks pass. Then identify whether the actual mechanism falls outside Section 3.7. A different mechanism is a proposal for later work, not permission to redesign this study silently.

## 9. Required records, analysis, and interpretation

### 9.1 Raw records and reproducibility

Use append-only JSONL for attempts and prefix records, a compact JSONL instance manifest, and derived CSV/Markdown tables. Store at least:

- Attempt ID, stage, timestamp, code/config/freeze hashes, instance ID, generator key, actual rows/RHS, `n,m,r,t`, and variable order.
- Consistency status, retained original-row indices, dependency certificates, split masks, sparse coefficient dictionaries, anchor, bases, and `S,T` certificates.
- Exact degree before/after gauge and for `Q`; support sizes; maximum coefficient bit length; `w_gen` only when actually computed; `q_raw,q,r`.
- Per-prefix finite-state counts, include-transition opportunities, attempted updates, quotient-equality result, and maximum resident entries.
- Complete signature zero counts when enumerated; oracle minimum/witness; each arm's decision/witness; original-row validation; mismatch details.
- Wall/CPU measurements and timing boundaries, peak RSS raw/normalized, output bytes, exit status, resource limit, last completed prefix, and any omitted or failed check.

Serialize arbitrary-precision coefficients and masks as decimal or hexadecimal strings where portability could otherwise lose precision. Small count values may be JSON integers. Store byte-level hashes of generated instance files. Record the actual Python/OS/architecture, source hashes, and no-third-party-dependency fact; do not label an unavailable dependency as installed.

For a mismatch, save a compact reproducible counterexample: input rows/RHS, cap/order, polynomial dictionaries, coordinates, failing prefix/state, expected and observed weight, and an explicit witness if available. Do not dump millions of irrelevant rows. Keep unsuccessful attempts and fixes linked by parent attempt ID.

### 9.2 Essential analysis outputs

Generate these directly from raw files:

1. A fixed-control table with expected/observed degree, rank, zero count, and decision outcomes.
2. A per-instance table containing `r,q_raw,q,degree(P),degree(P_star),degree(Q)`, support/coefficient sizes, and all-prefix inequality/identity status.
3. One concise prefix-count table for the 12-bit case and one for the two-pair case. Exact tables are more informative than a decorative plot here.
4. A completeness/resource table accounting for every prescribed job, failure, omission, repair, and measured phase cost.
5. The verified proof and the precise model scope, including why quotienting recovers ordinary syndrome DP.

Do not publish a headline speedup or geometric-mean runtime comparison. A faster Python loop in one arm would be implementation evidence only, not a mathematical consequence of degree halving. No p-values or confidence intervals should be fabricated for exact deterministic identities.

### 9.3 Outcome categories

| Category | Appropriate use in this study |
|---|---|
| **Supported contribution** | The scoped proof, certificates, and reproducible diagnostic are complete. Describe the contribution as a limitation/synthesis; do not imply historical novelty or a new decoder. |
| **Informative negative result** | The rank and DP-projection arguments are verified, with the essential exact controls passing. They rule out state compression by this mechanism, including the best common-character shift. This is the anticipated substantive conclusion. |
| **Practical infeasibility** | Measured cost prevents a specified audit or proposed application within the limits. Identify the failed workload and retained evidence; a timeout alone does not refute an algorithm. |
| **Inconclusive evidence** | A proof step remains unresolved, source version cannot be checked where material, target validation is missing, or computational coverage is insufficient for an empirical claim. A valid analytic result may coexist with incomplete target validation. |
| **Invalid experiment** | An implementation, oracle, data, phase, measurement, or fairness defect remains unresolved. Preserve the attempted runs and separate any independently valid proof from invalid empirical evidence. |

The theorem's correctness, this route's limitation, the implementation's correctness, historical novelty, and laptop feasibility are separate judgments. Report each explicitly. A false global zero-set equivalence is an encoding defect; a false claim about every decoder is an interpretation defect; neither should be hidden by a passing subset of tests.

If the expected limitation is confirmed, the recommendation is to stop developing this signature-DP route for state compression. The only justified next step is to obtain a concrete alternative zero-search mechanism with its own correctness and full-cost argument before committing to a new comparison. Do not suggest that larger instances alone can overturn the algebraic identity.

## 10. Final artifacts and completion contract

Deliver only artifacts justified by the stages reached:

- Runnable exact code and a short `README.md` with setup, smoke, audit, and report commands if implementation occurred.
- `config.json`, freeze/environment/source manifests, actual generated inputs, raw JSONL records, failures, and regeneration scripts for any experiments performed.
- Essential CSV/Markdown tables and `REPORT.md`, including the proof, exact scope, source attribution, observed results, outcome classification, deviations, and remaining blockers.
- For an analytic early stop, a complete verifiable derivation or concrete counterexample, plus whatever minimal diagnostic was actually executed. Do not create empty benchmark tables suggesting that omitted stages ran.

The report must distinguish planning observations copied from Section 4 from newly executed checks, and measured target results from estimates or other-host runs. Its first paragraph should state the decision and its scope. Its conclusion must not assert that the supplied degree theorem is false merely because the selected search route cannot compress states.

Run the authorized stages autonomously within these bounds. Resolve ordinary implementation choices yourself. If essential hardware, permissions, or a received file is missing, finish unaffected work and identify the smallest missing item. Never claim unperformed validation or request approval for routine reversible implementation work already covered by this assignment.

## 11. Supporting opportunity assessment and rejected shortcuts

This section explains why the chosen diagnostic is scientifically preferable to immediately building a broader solver.

### 11.1 Prior art and mathematical distinctions

The signed/nonnegative degree separation and full affine closure formulas are the supplied note's substantive claims. Its own originality assessment is cautious [R1, p. 8]. The single-parity phenomenon and nondeterministic-degree viewpoint are established [R4, Section 2.1 and Theorem 2.3]. Code ideals already use split binomials whose degree relates to roughly half a codeword's weight [R5, Section 4, Proposition 1]; real vanishing-ideal/coset constructions also appear in binary-matroid work [R6, Lemmas 3.2–3.3, Theorem 3.4].

Be precise about duality. In this brief the affine solution code is `L=ker(H)` and the polynomial frequencies are built from `D=L^perp`. Standard code-ideal decoding often uses an ideal associated with the **primal** code `L` to reduce exponent vectors modulo `L`. These are related uses of code algebra, not automatically the same executable algorithm. A low-degree generating set is not a computed Gröbner basis [R5, Sections 3 and 5].

The real-valued Fourier invariance identity in Gopalan et al. directly supplies the foundation for the rank obstruction [R3, Fact 5.2]. It prevents claiming a new universal parity-information theorem. Our character-shift normalization and per-prefix minimum-weight projection are explicit elementary adaptations to the proposed zero-test application.

Recent quadratic modeling work distinguishes input degree, solving degree, and matrix size, and already models both bounded and exact weights [R7, Sections 2.2 and 4.1]. Its comparison with Prange uses estimated algebraic costs and observed step degrees; it is not a transferable matched wall-time baseline. Therefore this brief neither repeats its numerical comparison nor assumes that a degree reduction predicts a solve-time improvement.

### 11.2 Why the obvious alternatives are not additional PoCs here

| Alternative bridge | Material issue and disposition |
|---|---|
| Evaluate `P(e)` while enumerating candidate errors | Exact but leaves the enumeration problem unchanged. Membership in `He=s` was already easy. Not a useful new decoder claim. |
| Find an optimal short-check basis first | May improve the displayed degree, but the paper gives no general efficient basis optimizer. The rank obstruction survives even a free optimal basis, so do not hide this task in preprocessing. |
| Square the signed constraints and minimize a penalty | On the cube, `q_j^2=2(1-sigma_j chi_(h_j))`, hence `(1/2)sum_j q_j^2=Q`. This exact identity returns the ordinary nonnegative parity penalty. Also `P^2` has the same zero set and is nonnegative, so its reduced degree is at least `w_gen` by Theorem 3. This limits these particular transformations, not every SOS approach. |
| Invoke a general polynomial solver because degree is small | Solving degree, number of variables, coefficient growth, auxiliary encodings, and intermediate matrices must be charged. Existing quadratic syndrome-decoding models already show that low input degree alone is not a new mechanism [R7]. No unbudgeted Gröbner/SOS implementation is authorized. |
| Use the fixed-degree closure routine as a fast general decoder | Its candidate count grows with `d`; it enforces short implied checks but does not decide intersection with a Hamming ball. It also is not compositional under conjunction [R1, pp. 6–7]. |
| Scale the original homogeneous example | `e=0` always solves it. It is a correct representation control and cannot support hard-decoding claims. |

For an order-of-magnitude check, `sum_{j=0}^6 binom(64,j)=83,278,001` and `sum_{j=0}^8 binom(100,j)=203,366,882,996`. These exact integer counts were checked during planning; they are candidate counts for the paper's degree-three/degree-four closure construction, not measured runtimes. They justify avoiding that hidden enumeration as a “cheap” preprocessing step.

### 11.3 Search coverage and access limitations

Research on 29 October 2026 used two web search systems and direct primary-source reads. Searches covered signed/nonnegative affine zero tests; code ideals and split binomials; nondeterministic polynomial degree; Fourier dimension, parity queries, invariance, and linear sketches; syndrome/minimal trellises; and recent algebraic syndrome-decoding models. The paper's citations were followed where they could affect the selected mechanism or novelty. Later algebraic-model papers were checked to avoid treating low-degree modeling itself as new.

Relevant full content was read for [R3]–[R8], and the supplied note [R1] was read completely. For [R2], the authoritative repository metadata and abstract were verified, but attempts to retrieve the original PDF failed; the original NP-completeness reduction was not audited line by line. Recent September/October 2026 work [R9]–[R10] was located through primary records/abstracts, but full-paper retrieval was unsuccessful. Do not attribute detailed bounds, implementation performance, or superiority to those papers. These gaps do not affect the self-contained state obstruction; they do limit any broader novelty survey of algebraic decoding.

This was targeted decision-relevant research, not an exhaustive global priority search. No inaccessible source is used as the sole justification for the selected proof or execution dependency.

## 12. Review findings and residual risks

Three distinct review passes must be completed on the implementation/report before final delivery: mathematical/novelty, experimental, and empty-project execution. The planning brief itself was reviewed along those same dimensions with independent agents; material findings incorporated here include:

| Review finding or stress scenario | Disposition in this protocol |
|---|---|
| Raw support rank unfairly counts an irrelevant common sign | Added globally rank-optimal character shifting. The motivating example now correctly ties syndrome rank after the shift. |
| The lower bound resembles established Fourier/linear-sketch theory | Added direct attribution and removed historical-novelty claims. |
| Full `2^r` tables could be mistaken for a universal decoder lower bound | Proved sparse per-prefix domination and explicitly excluded minimized trellises/other algorithms. |
| The theorem might require a hidden optimal-basis oracle | Construction uses supplied independent rows; optimality is claimed only when checked; the obstruction does not depend on optimal basis discovery. |
| Favorable homogeneous/synthetic examples could manufacture success | Classified them as controls, included nonzero phases and dense/sparse coverage, and eliminated unsupported runtime claims. |
| A fairer baseline could remove an apparent gain | The signed route receives the best common-character normalization; its exact quotient is the native baseline. No advantage is inferred from raw timing. |
| Preprocessing or intermediate representations could dominate | Charge preprocessing and gauge/coordinate construction; store factored characters; prohibit large expanded systems and retain explicit caps. |
| Floating point could change zero/nonzero decisions | Exact integers mandatory; a reproduced cancellation/rounding stress fixture documents the hazard. |
| A small pilot could be noisy or uninformative | The primary evidence is exact identity plus proof. No noisy timing pilot is used to accept or refute a speed hypothesis. |
| Essential sources or target machine could be unavailable | Mathematics and minimal code remain self-contained; source-version and M3 validation blockers are reported separately. |
| Memory/runtime exceeds estimates | Calibrate, use the sole predefined suite reduction, or stop after fixed analytic validation. Do not weaken correctness. |
| Draft admission policy exceeded its locked-stage budget, and some execution details were ambiguous | Corrected admission to an eight-minute projection within the ten-minute stage; fixed generation keys, CLI/ledger responsibilities, cumulative confirmation counts, and the default post-calibration path. |
| Main hope of state compression is false | The verified obstruction is an informative negative result and the planned completion path. No rescue campaign is required. |

Residual risks are ordinary implementation defects, an incorrect interpretation of the proof's scope, unverified target hardware performance, and uncertain historical originality of the precise corollary. The protocol addresses them with independent finite oracles, explicit matrix certificates, scoped language, target inspection, and cautious attribution. It does not claim absolute mathematical or experimental perfection.

## References

**[R1] Supplied research note.** Author not identified. *Exact degree of polynomial zero tests for binary affine systems: Signed witnesses, nonnegative witnesses, and short parity checks*. 29 October 2026, 10 pages, attachment `affine_zero_tests.pdf`; target location `sources/affine_zero_tests.pdf`. Critical locations: definitions p. 1; Theorems 1–2 pp. 2–4; Theorem 3 pp. 4–5; Theorem 4 p. 5; closure algorithm and reported tests pp. 6–7; limitations p. 7; originality discussion p. 8; appendices pp. 8–10. No public URL supplied. Source hash is in Section 2.1.

**[R2] Elwyn R. Berlekamp, Robert J. McEliece, Henk C. A. van Tilborg.** *On the inherent intractability of certain coding problems*. IEEE Transactions on Information Theory 24(3), 384–386, May 1978. [DOI: 10.1109/TIT.1978.1055873](https://doi.org/10.1109/TIT.1978.1055873). [Author-repository record and abstract](https://authors.library.caltech.edu/records/aw9vs-ann16). Used for the established general-decoding hardness context. Record/abstract verified; full original reduction not accessed successfully during planning.

**[R3] Parikshit Gopalan, Ryan O'Donnell, Rocco A. Servedio, Amir Shpilka, Karl Wimmer.** *Testing Fourier Dimensionality and Sparsity*. SIAM Journal on Computing 40(4), 1075–1100, 2011. [DOI: 10.1137/100785429](https://doi.org/10.1137/100785429). [Accessible author manuscript](https://www.cs.columbia.edu/~rocco/Public/4-27-2011-gossw.pdf). Critical locations in that manuscript: Definition 1.1, p. 2; Definition 5.1, pp. 10–11; Fact 5.2, p. 11, giving Fourier-support span as the orthogonal complement of translation invariance for real-valued functions. Relevant full text read.

**[R4] Ronald de Wolf.** *Nondeterministic Quantum Query and Communication Complexities*. SIAM Journal on Computing 32(3), 681–699, 2003. [DOI: 10.1137/S0097539702407345](https://doi.org/10.1137/S0097539702407345). [Author PDF](https://homepages.cwi.nl/~rdewolf/publ/qc/nqj_final.pdf). Critical locations: Section 2.1, printed p. 685, nondeterministic degree and signed parity example; Theorem 2.3, printed pp. 687–688, `NQ(f)=ndeg(f)`. This is query-model context, not a classical search guarantee. Relevant full text read.

**[R5] M. Borges-Quintana, M. A. Borges-Trenard, P. Fitzpatrick, E. Martínez-Moro.** *Gröbner bases and combinatorics for binary codes*. arXiv:math/0509164, submitted 2005. [Stable record](https://arxiv.org/abs/math/0509164); [PDF](https://arxiv.org/pdf/math/0509164). Critical locations: Section 3, Definition 2 and equations (6)–(7), pp. 3–4; Section 3.1, Theorem 1, pp. 4–5; Section 4, Proposition 1, p. 6; Section 5, basis construction. The accessed PDF displays a later internal date; cite the identifier/submission year without inventing a publication date. Relevant full text read.

**[R6] João Gouveia, Monique Laurent, Pablo A. Parrilo, Rekha Thomas.** *A new semidefinite programming hierarchy for cycles in binary matroids and cuts in graphs*. arXiv:0907.4518, version 2, 15 October 2010; originally submitted 2009. [Record](https://arxiv.org/abs/0907.4518); [PDF](https://arxiv.org/pdf/0907.4518). Critical locations: Lemmas 3.2–3.3 and Theorem 3.4, pp. 8–9 of the accessed manuscript, on split binomials, dual cosets, and real vanishing ideals. Relevant full text read; no SDP implementation from this source is required.

**[R7] Alessio Caminata, Ryann Cartor, Alessio Meneghetti, Rocco Mora, Alex Pellegrini.** *Quadratic Modelings of Syndrome Decoding*. PQCrypto 2025, pp. 35–70. [DOI: 10.1007/978-3-031-86599-2_2](https://doi.org/10.1007/978-3-031-86599-2_2). [arXiv:2412.04848v2, 28 January 2025, full HTML](https://arxiv.org/html/2412.04848v2); [ePrint 2024/1975](https://eprint.iacr.org/2024/1975). Critical locations: Section 2.2, solving degree/Macaulay matrices and Remark 3; Section 4.1, Modeling 1; Section 5.1, Table 3 and comparison with combinatorial attacks. Relevant full text read. No proprietary MAGMA dependency is adopted.

**[R8] G. David Forney, Jr.** *Trellis representations of binary linear block codes*, Chapter 10, MIT 6.451, *Principles of Digital Communication II*, Spring 2005. [Course resource](https://ocw.mit.edu/courses/6-451-principles-of-digital-communication-ii-spring-2005/resources/chap10/); [PDF](https://ocw.mit.edu/courses/6-451-principles-of-digital-communication-ii-spring-2005/1b67ae9f5c124c3bd56c8649afa5af47_chap10.pdf). Critical locations: Section 10.2.1, Lemma 10.1 and Theorem 10.2, printed pp. 137–139; Section 10.2.2, canonical minimal trellis. Used to delimit full-signature DP claims, not as a newly implemented comparator. Relevant full text read.

**[R9] Elisa Gorla, Simone Trebiani.** *Symmetric Models for Syndrome Decoding*. arXiv:2609.14052, September 2026. [Primary record](https://arxiv.org/abs/2609.14052); [author talk abstract, University of Zurich, 14 October 2026](https://www.math.uzh.ch/en/ve-mfs-sem-vor0?L=1&key1=0&key2=1816&key3=6082). Located as recent related work on symmetric polynomial models. Abstract/topic verified; full text unavailable during planning. No detailed theorem or performance claim from it is used.

**[R10] Roberto La Scala, Marco Marchesin, Sharwan K. Tiwari.** *Hamming Ideals and Gröbner Bases for ISD-like Syndrome Decoding*. arXiv:2609.18866; IACR ePrint 2026/2062, 2026. [arXiv record](https://arxiv.org/abs/2609.18866); [ePrint record](https://eprint.iacr.org/2026/2062). Primary landing-page/abstract material was inspected by the literature reviewer; full-paper retrieval failed. Listed as a coverage limitation for broader algebraic-decoding novelty, not as evidence for a detailed result.

---

**Completion rule:** deliver a verified scoped limitation with executable evidence, or an honestly classified blocker/defect with preserved evidence. Do not substitute more experiments for a missing proof, and do not substitute a lower polynomial degree for a decoding algorithm.
