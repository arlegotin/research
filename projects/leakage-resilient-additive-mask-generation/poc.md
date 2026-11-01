# Execution brief: leakage-resilient arithmetic mask generation

**Recipient:** autonomous coding agent configured as `gpt-6.1-sol`, reasoning `xhigh`  
**Research and planning date:** 23 October 2026  
**Target:** one MacBook with an M3 Max; all experimental computation stays on that machine  
**Deliverable from you:** the executed, reproducible study and its research report, including a justified early-stop report if a gate fails

## 1. Project decision and smallest defensible contribution

Implement and evaluate **one component**: generation of a fresh pair of arithmetic masks in

\[
G_w=(\mathbb Z/2^w\mathbb Z)^2,
\]

remaining jointly uniform when any two of the component's logical source bits are disclosed. The primary width is **32 bits per mask**. Compare complete generation costs in one fixed, three-party, semi-honest replicated-sharing configuration of MP-SPDZ, with all three processes on the target laptop.

**Use the stronger construction derived during this planning investigation**, rather than limiting the study to the attachment's 144-bit block lift. Draw four independent uniform words \(U_1,\ldots,U_4\in\mathbb Z/2^w\mathbb Z\) and one independent fair bit \(B\). Return

\[
R_1=U_1+U_3+2U_4+B\pmod {2^w},\qquad
R_2=U_2+U_3+U_4\pmod {2^w}.
\tag{C129}
\]

At \(w=32\), this uses **129 logical independent source bits**. Section 3 supplies two executable proofs of its leakage property. It is an additive map of the individual source bits and has a particularly small implementation using native arithmetic words.

The strongest immediate additive comparator is equally important: draw five independent uniform words and return

\[
S_1=V_1+V_2+V_5,\qquad S_2=V_3+V_4+V_5\pmod {2^w}.
\tag{N160}
\]

This is the 160-bit separate-level construction **regrouped into five native word draws**, with the same leakage guarantee. Do not weaken it by using 160 arithmetic-bit draws in the primary comparison.

### Primary falsifiable question

> On the inspected M3 Max, in the frozen MP-SPDZ configuration and workload, does C129 reduce the complete time to generate independent pairs of 32-bit arithmetic masks by at least 5% against N160 and a properly implemented shortened-Hamming alternative, including source generation, required preprocessing, and representation conversion?

The **5% reduction in elapsed time** is a planning choice for an engineering adoption decision: a smaller speed difference alone would not justify maintaining a specialized component in this study. It is not a universal economic threshold, a published industry requirement, or a threshold chosen from observed results. Report absolute time saved as well. Exact entropy savings are a separate result, with no statistical threshold.

The smallest defensible contribution is a verified, simple \(4w+1\)-bit additive construction and a measured answer about whether its native implementation is useful at this source interface. The possible contribution includes a clear limitation: **one secret arithmetic bit can cost more than the native word it replaces**. The construction's historical novelty and optimality for arbitrary \(w\) are unestablished. Do not call 129 bits optimal or call the component a new secure MPC protocol.

### Why the project has been refined

The supplied nine-bit construction is valid and optimal for two modulo-four outputs. Its 32-bit lift is valid but not the strongest candidate. Small exact-cover probes during planning suggested lengths \(5,9,13,17\) at widths \(1,2,3,4\); a subsequent elementary derivation produced C129 for all widths. This removes a weak comparison and exposes a much cleaner practical question. The refinement preserves the original purpose: leakage-resilient arithmetic masks with local additive combination. It does not attempt general mask generation, arbitrary side-channel protection, optimality at 32 bits, or a full cryptographic application benchmark.

| Mechanism, for one pair at width 32 | Logical source bits | Best relevant source interface | Role |
|---|---:|---|---|
| C129, equation above | 129 | Four native arithmetic words + one arithmetic bit | Selected candidate |
| N160, equation above | 160 | Five native arithmetic words | Required strong additive comparator |
| Shortened Hamming map H71 | 71 | Native Boolean sources + arithmetic output conversion, or optimized arithmetic evaluation | Required before any broad positive performance conclusion |
| Attachment's nine-bit block lift A144 | 144 | Arithmetic bits, unless an equally valid better supplier is explicitly demonstrated | Historical/mechanism control |
| Separate-word protection A192 | 192 | Six native words, grouped as three per output | Correct but dominated reference; correctness/counts only |
| Two direct native masks U64 | 64 | Two native words | Unprotected cost floor; **not leakage matched** |

C129 saves \(31/160=19.375\%\) of logical source bits against N160 and \(15/144\approx10.417\%\) against A144. These are derivations, not runtime measurements. H71 remains much more economical in logical source bits. Do not conceal that comparison.

## 2. Fixed scientific contract, scope, and assumptions

### 2.1 Output and leakage contract

For each request, the source is a vector of independent fair logical bits \(Z\). A full-range native word is shorthand for its \(w\) independent binary coordinates. For every fixed set \(D\) of at most two coordinates and every assignment \(z_D\), require

\[
\Pr(R=r\mid Z_D=z_D)=2^{-2w}\quad\text{for all }r\in G_w.
\tag{1}
\]

For C129, the eligible coordinates are all bits of \(U_1,\ldots,U_4\), plus \(B\). For N160, they are all bits of its five words. For H71, they are its 71 source bits. **Every entropy-bearing logical source used in the output must appear in this inventory.** Relabeling an extra output-bearing source as preprocessing does not remove it from the leakage model.

For an independent secret \(X\), observing \(X+R\) and the allowed disclosures gives no information about \(X\) in the ideal-source model. This follows from (1), not from a hardware leakage experiment. Choices of a second coordinate based only on the first disclosed value are also covered: each resulting transcript fixes at most two coordinates. Outcome-dependent selection based on undisclosed source information is outside the contract.

Each request uses disjoint logical sources. A benchmark batch consists of many independent requests and permits **two disclosures per request**. Batching changes scheduling and vectorization, not the security contract. No source reuse is permitted across requests.

This per-request contract matters. If the application instead permits only two disclosures across a whole batch, joint coding is stronger: two pairs can use a binary \([7,4,3]\) code at each of 32 levels, costing 224 bits total, or 112 per pair. That construction tolerates two global disclosures; it does not establish protection against four disclosures distributed as two per request. State this limitation in the report. Do not market the per-pair method as optimal for globally bounded bulk leakage.

### 2.2 What the implementation study establishes

The experiment measures the cost of realizing specified ideal functionalities with an existing MPC implementation. Its protocol boundary is three-party semi-honest Rep3 over the ring \(\mathbb Z/2^{32}\mathbb Z\), with the implementation's normal computational assumptions. Running all parties locally is a benchmarking arrangement, not a secure deployment against an adversary controlling that laptop.

The exact mathematical guarantee assumes independent fair logical sources. MP-SPDZ's PRG-based realization does not turn an experiment into proof of independent physical coin flips. Its internal seeds, correlated randomness, shares, preprocessing tapes, and transcripts are not the named leakage coordinates. Count all their costs, but do not equate logical source-bit counts with physical entropy or total PRG output bytes. Composition with any real leakage-bearing supplier remains an explicit application assumption requiring its own security argument.

No concrete deployed caller or leakage-bearing source was supplied. Therefore a positive result supports this **component and interface**, not an improved production cryptosystem. A caller that only needs ordinary secure random shared words may not need any of these wrappers.

Explicit exclusions are arbitrary intermediate-wire probing, glitches, power/EM leakage, malicious parties, compromised servers, PRG-seed leakage, arbitrary functions of the source, entropy repair for correlated or biased sources, adaptive omission after inspecting all inputs, and randomness beacons. Do not add a side-channel measurement campaign or claim those properties.

### 2.3 Autonomy and fixed versus adjustable choices

Execute the authorized stages without repeatedly asking for routine decisions. Keep the output functionality, two-disclosure contract, primary width, candidate, mandatory N160 comparator, 5% adoption threshold, statistical decision rules, and cumulative caps fixed. You may choose local filenames, ordinary coding details, an available supported Python minor version, and a documented build correction inside the stated bounds. The bounded calibration rules select workload and repetition count.

A change to the security model, arithmetic domain, candidate formula, primary timing boundary, comparison set needed for a positive claim, or threshold is a scientific deviation. Record it; do not silently redefine success. If target hardware or a necessary permission is unavailable, complete exact mathematics and unaffected code, then identify precisely which performance claims remain blocked.

## 3. Mathematics that must govern the implementation

### 3.1 Audit of the attachment

The supplied PDF [A1] is a 17-page research note, not evidence of peer review or certified priority. Its relevant statements are:

| Source location | What is established under its assumptions | Use here |
|---|---|---|
| Theorem 1.2, pp. 3-4 | For independent uniform indexed dice and an additively separable sum, one-omission uniformity implies \(|G|\exp(G)\mid\prod q_i\); equality is constructively attainable | Context and a check against confusing arbitrary resilient maps with additive ones |
| Corollary 2.1, p. 4 | For an abelian \(p\)-group of order \(p^m\), exponent \(p^k\), exactly \(m+k\) fair \(p\)-symbol inputs suffice for one omission | Do not use as a two-omission or 32-bit optimality bound |
| Lemma 3.1, pp. 5-6, equation (6) | Binary omission resilience is equivalent to covering each nontrivial character with sufficiently many zero Fourier factors | Exact verification oracle |
| Proposition 3.2, p. 6 | Biased independent Bernoulli choices cannot improve the minimum number of summands for exact uniformity | Bias is an assumption failure, not an optimization |
| Theorem 4.1, pp. 6-8, equations (7)-(10) | Exact omission profile of \(C_4\oplus C_4\), including \(L_3=9\) | Validates the supplied small construction |
| Section 4.4, p. 8 | Nine coefficients beat the ten-coefficient separate-level construction | Historical comparison |
| Section 5, pp. 8-9 | Reports integer verification scripts and checks | Author-reported evidence; scripts/archive were not supplied |
| Sections 6-7, pp. 9-10 | Discuss known frameworks, novelty uncertainty, and limitations | Preserve those qualifications |

The note's independent spanning-tree and median-closure appendices were inspected. Neither supports this mask-generation mechanism; exclude them from the implementation. Do not spend the PoC budget reproducing their experiments.

The one-omission fiber proof is constructive and internally consistent: every output fiber has size \(M=N/|G|\) and identical coordinate marginals; summing the additive output identity over a fiber and subtracting the zero fiber gives \(Mg=0\) for every \(g\), hence \(\exp(G)\mid M\). The small nine-bit lower bound is also independently checkable. In the note's notation, \(2S\ge9\) and \(S+4H\ge18\); therefore

\[
S+H\ge\left\lceil(18+3\lceil9/2\rceil)/4\right\rceil=9.
\]

The displayed nine-coefficient construction attains it. No optimizer is needed for this claim.

### 3.2 Exact Fourier criterion

For \(q=2^w\), a coefficient list \(a_i=(a_{i1},a_{i2})\), and independent fair bits \(Z_i\), let

\[
R=\sum_i Z_i a_i\pmod q.
\]

For the character indexed by \(b=(x,y)\), its Fourier coefficient factors as

\[
\mathbb E\chi_b(R)=\prod_i\frac{1+\exp(2\pi i(xa_{i1}+ya_{i2})/q)}2.
\]

A factor vanishes exactly when

\[
xa_{i1}+ya_{i2}\equiv q/2\pmod q.
\]

Consequently, (1) holds exactly when every nonzero \(b\) has **at least three** such coefficients. Conditioning on disclosed bits only adds a fixed translation to the omitted sum. This is [A1, Lemma 3.1] specialized to the experiment; the Fourier/resilient-function framework is established prior art [R1-R3].

Use integer congruences. Never evaluate complex roots numerically and decide that a tiny number is zero.

### 3.3 Planning derivation: the \(4w+1\) family

This construction and its following proof were derived during preparation of this brief; they are not attributed to the attachment or to the cited papers. Define the coefficient **multiset**

\[
A_w=\{2^j(1,0),2^j(0,1),2^j(1,1),2^j(2,1):0\le j<w\}
\cup\{(1,0)\},
\tag{2}
\]

reducing coordinates modulo \(q\) and retaining repetitions as distinct independent source bits.

For a nonzero scalar \(z\in\mathbb Z/q\mathbb Z\), write \(z=2^v u\) with \(u\) odd. Exactly one \(j\in\{0,\ldots,w-1\}\) satisfies \(2^jz=q/2\pmod q\), namely \(j=w-1-v\). For \(z=0\), none does. Thus the four geometric families in (2) contribute one zero factor for each nonzero scalar among

\[
x,\quad y,\quad x+y,\quad2x+y.
\]

Every pair of these forms has trivial common kernel except the pair \(y,2x+y\). That pair has the additional common zero \((q/2,0)\). Therefore every nonzero character has at least three zero factors from the four families, except this one character, which has two. The appended \((1,0)\) supplies its third. Equation (1) follows for every \(w\ge1\).

Grouping the \(w\) bits in each direction into a uniform word gives C129. The formula is explicit, finite, and linear over the final arithmetic ring once sources are available. There is no search oracle, integer program, convergence assumption, floating-point approximation, or exponentially large preprocessing step in the implemented mechanism.

**Second proof, useful for the native implementation.** The word directions are

\[
v_1=(1,0),\ v_2=(0,1),\ v_3=(1,1),\ v_4=(2,1).
\]

If any two whole word sources are fixed, the two remaining words give a uniform pair whenever their direction determinant is odd. The only exceptional surviving pair is \(v_2,v_4\), whose determinant is \(-2\). Its image is all pairs with even first coordinate, uniformly; the independent bit in direction \((1,0)\) selects the missing parity coset. If the bit and one word are fixed, the three remaining word directions contain an odd-determinant pair. Thus the construction even tolerates fixing any two entire source symbols among its four words and bit. Fixing two constituent bits is weaker. Keep the main comparison on the common bit-level contract; the stronger property does not change H71's required guarantee.

**Sensitive invalid control.** Delete the extra bit. Then disclose the least significant bits of \(U_1\) and \(U_3\). The first output's parity is fixed. At \(w=2\), the conditional distribution is uniform on eight of sixteen outputs, giving total variation distance exactly \(1/2\) from uniform. A validator that accepts this control is invalid.

### 3.4 Required baselines and their proofs

**N160.** Its word directions are \(e_1,e_1,e_2,e_2,e_1+e_2\). Two disclosed bits touch at most two words. Among the other three words, two directions have odd determinant. Their sum already provides an independent uniform pair; adding other contributions cannot bias it. Expanded into individual bits, the coefficients at each level are

\[
2^j(1,0),2^j(1,0),2^j(0,1),2^j(0,1),2^j(1,1).
\]

This establishes that the bitwise and native-word baseline implement the same map and source law. Within maps that use only fixed linear combinations of whole uniform words, four word sources cannot suffice: reduction to their least significant bits would give a binary dimension-two, distance-three code of length four; its three nonzero codeword weights sum to at most eight, below the required nine. The extra bit in C129 changes that restriction.

**H71.** Let \(m=2w\), and select the smallest \(r\) satisfying \(m+r\le2^r-1\). For the primary case, \(m=64,r=7,n=71\). Let \(p_0,\ldots,p_{m-1}\) be the first \(m\) nonzero, nonbasis \(r\)-bit vectors ordered by `(Hamming weight, integer value)`, with least significant coordinate first. Form

\[
H=[P\mid I_r],\qquad G=[I_m\mid P^T],
\tag{3}
\]

where \(P\)'s columns are those vectors. Then \(HG^T=0\), \(G\) has full row rank, and the columns of \(H\) are distinct and nonzero. Its kernel code has minimum distance at least three. For independent source bits \(Z\in\mathbb F_2^{m+r}\), output \(Y=GZ\). Deleting any two columns of \(G\) retains rank \(m\): otherwise a nonzero row combination of \(G\) would have support at most two. Therefore fixing any two source bits leaves all \(m\) outputs uniform. This is the standard linear-code extractor construction [R2, Theorem 13; R4, §3.1].

Write \(Z=(d_0,\ldots,d_{m-1},c_0,\ldots,c_{r-1})\). Evaluate

\[
Y_i=d_i\mathbin{\mathrm{XOR}}\bigoplus_{j:(p_i)_j=1}c_j,
\quad
R_1=\sum_{i=0}^{w-1}2^iY_i,\quad
R_2=\sum_{i=0}^{w-1}2^iY_{w+i}.
\]

For \(w=2\), this is the seven-input Hamming construction. These counts are optimal even among arbitrary balanced two-resilient maps for 4 and 64 output bits: an output fiber has \(N=2^{n-m}\) rows; the constant column and the \(n\) input-coordinate sign columns are mutually orthogonal nonzero vectors in \(\mathbb R^N\), hence \(N\ge n+1\). This rules out \((m,n)=(4,6)\) and \((64,70)\); Hamming attains the next integer. This elementary strength-two orthogonal-array bound is established mathematics, not a novelty claim [R3].

**A144.** Use the attachment's nine coefficients

```text
(2,0), (0,2), (2,2), (1,0), (1,2), (0,1), (2,1), (1,1), (1,3)
```

and multiply all of them by \(4^j\) for \(j=0,\ldots,15\), evaluating every contribution directly modulo \(2^{32}\). Do **not** reduce each secret block sum modulo four before shifting. Such truncation can introduce nonlinear operations. The raw lift is correct: conditioned on allowed disclosures, each block residue is independent and uniform modulo four; carries from lower blocks translate the next independent uniform digit pair. Induct over positions. Nine-bit optimality at width two does not imply optimality of the lift.

**A192 and U64.** Three independent native words per coordinate give the 192-bit protected reference; two directly drawn native words give U64. Revealing one source bit of U64 reveals a mask bit, so it is excluded from the protected performance ranking.

## 4. Algorithms, exact references, and correctness obligations

### 4.1 Candidate and baseline interfaces

Implement a clear reference interface and a backend interface with the same semantics:

```python
coefficients(method: str, width: int) -> list[tuple[int, int]]
evaluate_bits(method: str, width: int, bits: Sequence[int]) -> tuple[int, int]
evaluate_words(method: str, width: int, words: Sequence[int], extra_bit=None)
    -> tuple[int, int]
make_hamming(width: int) -> (P, H, G, column_order)
verify_exact(config) -> structured_result
run_trial(method, request_count, chunk_size, boundary, config) -> trial_record
```

Reject wrong dimensions, nonbinary logical bits, widths outside the configured range, and unsupported methods. Public reference words are canonical integers in \([0,2^w)\). Reference arithmetic uses Python integers followed by `& ((1 << w) - 1)`. Do not silently accept a field backend in place of a ring backend.

The optimized native arithmetic candidate needs only five additions by sharing a subexpression:

```text
generate_C129(N, chunk, ring_source, bit_source):
    allocate output shares R1[0:N], R2[0:N]
    for each public chunk of q <= chunk requests:
        U1, U2, U3, U4 = four independent ring_source(q) draws
        B = bit_source(q), independent of the four draws
        T = U3 + U4
        R2_chunk = U2 + T
        R1_chunk = U1 + T + U4 + B
        store both output chunks at their public output offsets
    return the two output-share arrays
```

All operations occur in the final ring. There are no secret branches, secret-secret products, comparisons, truncations, or reconstruction in the combination. Sources and required preprocessing may themselves be costly.

```text
generate_N160(N, chunk, ring_source):
    for each public chunk:
        V1, V2, V3, V4, V5 = five independent ring_source(q) draws
        store (V1 + V2 + V5, V3 + V4 + V5)
```

For the supplied-bit ablations, draw exactly \(4w+1\) or \(5w\) arithmetic bits, form the words with public powers of two, and use these same formulas. Do not reconstruct bits. These are different supplier implementations of the same mathematical source law.

### 4.2 Exact omission and conditional-distribution checks

Implement integer convolution independently of the optimized formulas:

```text
omitted_histogram(coefficients, w, D):
    q = 2^w
    H = array(q*q, integer zero); H[0,0] = 1
    for index, a in coefficients:
        if index in D: continue
        NEW = copy(H)
        for (x,y) in [0,q)^2:
            NEW[(x+a.x) mod q, (y+a.y) mod q] += H[x,y]
        H = NEW
    assert sum(H) == 2^(len(coefficients)-len(D))
    return H
```

Uniformity requires every entry to equal \(2^{n-|D|-2w}\), exactly. Use old `H` for every update; an in-place update would reuse a source bit and invalidate the oracle.

Required checks follow. G1 completes all clear mathematical checks and C129/N160 backend integration before interpreting their timings. Full H71 backend checks are deferred until that route is implemented in G4b and must pass before interpreting any H71 route-selection or confirmation timing. A tiny conversion-capability smoke in G0 does not require the full Hamming program.

1. **C129 family:** all omission sets of sizes zero through two at widths 1-4; all nontrivial integer character tests at widths 1-6. At widths 1, 2, 3, the omission-set counts are 16, 46, 92; at width 4 it is 154. At width 2, every two-omission histogram has eight preimages per output.
2. **Independent truth-table oracle at width 2:** enumerate all input assignments and all disclosed values for C129's nine-bit instance and the attachment's original nine-bit instance. Each has 163 conditioning cases. Do not call the convolution helper to construct this oracle. Test every input for packed/expanded equivalence.
3. **N160 at width 2:** 56 omission sets, corresponding by translation to 201 conditioning cases. Check packed/expanded equality. Check A192 at width 2 and A144's base block exactly; for the two-block width-4 lift, check all 153 two-omission sets plus the zero/one cases using convolution.
4. **Hamming:** at width 2, enumerate all 128 input assignments and all 99 conditioning cases. At width 32, verify matrix dimensions, \(HG^T=0\), rank 64, distinct nonzero columns of \(H\), and full row rank of \(G\) after all 2,557 deletion sets of sizes zero through two. Use bit-packed Gaussian elimination over \(\mathbb F_2\), not floating rank. Cross-check the systematic formula against generic binary matrix multiplication.
5. **Negative controls:** the four-word-only candidate must fail with the stated parity witness and TV \(1/2\) at width 2; U64 must fail after one disclosure. With only C129's extra bit biased to \(\Pr(B=1)=1/4\), disclosing the two stated least significant bits produces a first-output parity bias and TV \(1/4\). Use exact rational counts. These are expected assumption failures, not candidate bugs.
6. **Backend integration:** compare revealed dummy outputs with the independent reference for at least 256 deterministic width-32 input vectors, covering zero, one, \(2^{31}-1\), \(2^{31}\), \(2^{32}-1\), alternating patterns, and overflow. Normalize signed display modulo \(2^{32}\). For each H71 route reached in G4b, include one-hot data/check inputs and nontrivial mixed patterns before timing that route. These input tests are outside performance timing and are not used as real masks afterward.

Preserve a minimal counterexample for any unexpected failure: method, width, coefficients or matrix, omission set, disclosed values if relevant, histogram, expected counts, and code/config hashes. One ordinary implementation repair is allowed before rerunning the affected checks; shared repair limits still apply. Never weaken an exact condition to a numerical tolerance.

No exhaustive enumeration at width 32 is required or feasible. The general proof, exact small cases, symbolic coefficient generation, and integer backend checks are the evidence bridge. Monte Carlo uniformity tests, entropy estimators, and randomness-test suites are unnecessary for the exact guarantee.

### 4.3 H71 implementation specification and bounded optimization

Build the mathematical H71 reference immediately. Defer full performance infrastructure for it until the N160 gate justifies it.

Consider two execution routes, with the deterministic matrix (3) fixed throughout:

- **H71-A:** 71 arithmetic source bits; evaluate XOR by \(u+v-2uv\). Build a shared parity DAG for subsets of the seven check bits, using memoized prefixes: remove the highest set bit, recursively obtain the prefix parity, then XOR the last bit. Reuse nodes across all 64 outputs. Each chosen subset has size at most four. Include compiler simplifications that follow exactly from the final ring, such as eliminating a product multiplied by zero modulo \(2^{32}\); do not assume the compiler did so without inspecting counts.
- **H71-B:** native Boolean source bits; packed XOR evaluation of the same DAG; compose two 32-bit Boolean output words and convert both to arithmetic shares using supported MP-SPDZ primitives. Count all source generation and conversion preprocessing. This route is required to avoid treating expensive arithmetic input conversion as intrinsic to Boolean coding.

For H71-B, start with the documented Rep3 local-conversion setting `-Z 2`; test `-X` as one alternative if supported. Do not implement daBits, edaBits, or a new conversion protocol. At most **three H71 route/settings combinations** are allowed. First use up to four dedicated integration jobs to validate them. Then use at most **12 development MPC jobs**, including failures: three randomized development blocks, each containing C129 and every viable H71 variant (at most four jobs per block). Freeze the viable H71 variant with the smallest median complete time; break exact numerical ties by a recorded deterministic route name order. Report close or unstable rankings as selection uncertainty, not proof of a globally fastest implementation. Development measurements are never confirmatory evidence. G4b subsequently uses a fixed 24 fresh paired confirmation blocks, so it does not estimate a precision-based sample size from only three route-selection measurements.

The official Boolean random-bit API is scalar. Do not invent `sbit.get_random_bit(size=N)` or assume that requesting a wider type generates multiple independent bits. A safe initial route creates independent scalar bits, packs fixed-size lanes with the documented bit-composition API, evaluates bit-sliced parities, and converts the resulting word vectors. Use runtime loops over fixed chunks so compilation does not unroll \(71N\) operations. Verify the actual pinned APIs and supported lane lengths in a tiny smoke test. A high-level source bottleneck is an implementation limitation; it cannot establish that Boolean methods are inherently slow. Allow one bounded repair using a documented efficient source path if available; otherwise narrow the conclusion.

## 5. Environment, dependencies, repository, and intended commands

### 5.1 Inspect the actual Mac before allocating work

Record OS version, architecture, chip identifier, physical RAM, available memory, free disk, power state, Python version/architecture, compiler version, and dependency availability. Useful local commands include `sw_vers`, `uname -m`, `sysctl -n machdep.cpu.brand_string`, `sysctl -n hw.memsize`, `df -h .`, `xcrun clang++ --version`, and `python3 --version`. Inspect the machine; the chip name does not specify RAM or storage.

Require native ARM64 execution for target-machine performance claims. Do not silently run under Rosetta, use a remote host, or replace the target with the hosted planning environment. If the target is unavailable, exact reference work remains useful and executable elsewhere, but clearly label it and stop before claiming M3 results.

Use CPU integer arithmetic. Metal, CUDA, GPU frameworks, floating-point kernels, and hardware accelerators are unnecessary. A correct reference comes before optimization.

### 5.2 Minimal stack and fixed backend

Use Python 3.11 or later for references, orchestration, JSON/CSV, and standard-library tests; `psutil` for process-tree resource monitoring. `matplotlib` may be added for two static report figures after correctness, if not already available. No solver is required. Keep the implementation as ordinary scripts, one small JSON configuration, and local result files.

Use MP-SPDZ **v0.4.3**, resolving and recording its full commit; the release page identifies abbreviated commit `26a6053`. The documented requirements include Apple Silicon support and a C++20-capable toolchain [R5-R6]. Pin dependency versions actually installed. Do not silently use an older release after a failure: the release notes contain security fixes. A later revision is allowed only to repair a verified compatibility problem within the shared repair budget, recording the reason, commit, and affected checks.

The selected backend is `replicated-ring-party.x`, three parties, actual ring width 32. No malicious protocol, WAN emulation, Docker deployment, extra MPC framework, or remote worker is part of the primary study. Do not use fake preprocessing, insecure preprocessing modes, or cached uncharged material for the primary total measurement.

For native words use **`sint.get_random(size=q)`**, whose documented source emits `randomfulls`. For arithmetic bits use **`sint.get_random_bit(size=q)`**. Do not substitute `sint.get_random_int(32, size=q)` for native words: that method can compose bits or use edaBits and would change the baseline's supply cost [R7]. Inspect the pinned source, emitted instructions, and runtime metadata. `--bits-from-squares` is a prime-field option and is not a ring32 optimization here [R9].

The manual-conversion prose has a known naming inconsistency: verify the actual `sbitvec` / `sbitintvec` classes in the checkout rather than copying `sbitvecint`. Verify bit ordering, transposition, and batch shape in tests [R8]. These are dependency gates, not APIs verified by executing the PoC during planning.

### 5.3 Small repository layout

```text
README.md                         setup, commands, scope, reached gates
requirements.txt                  exact installed Python package versions
config/study.json                  budgets, model, workload candidates, seeds
config/locked_native.json          immutable G4a protocol and hashes
config/locked_hamming.json         immutable G4b protocol, only if reached
src/reference.py                  integer maps, coefficient lists, Hamming matrices
src/exact_checks.py                convolution, independent truth tables, GF(2) ranks
src/mpc_programs.py                bounded source generation for MP-SPDZ programs
src/runner.py                      process lifecycle, timers, budgets, raw records
src/analysis.py                    paired ratios, intervals, tables and figures
scripts/study.py                   the command interface below
tests/test_exact.py                deterministic meaningful correctness tests
tests/test_backend.py              dummy input and representation checks
vendor/MP-SPDZ/                    pinned shallow checkout, only selected backend
inputs/                           optional provenance copies of the two attachments
results/manifest.json              environment, versions, input/source hashes
results/ledger.jsonl               every job, including failures and repairs
results/raw/                      per-party logs, timing/resource data, counters
results/checks/                    exact certificates and counterexamples
results/derived/                   regenerated tables and static figures
REPORT.md                         complete research conclusion and limitations
```

Do not build a general benchmarking framework or dashboard. A single driver should enforce caps and create machine-readable records. Use monotonic time for durations and UTC timestamps for reproducibility; describe local scheduling times in the actual user's timezone if needed.

Create the following initial `config/study.json`. It is a concrete starting configuration, not an alternative source of scientific rules. The first `inspect` call creates `results/budget_state.json` with the original start timestamp, absolute deadline, reporting boundary, counters, and resolved memory/disk caps. It refuses to overwrite an existing budget state. If the driver is not yet written, record the start/deadline in that file manually before coding and have `inspect` import it. Lock files resolve all choices to actual numbers and record hashes.

```json
{
  "schema_version": 1,
  "width": 32,
  "parties": 3,
  "protocol": "replicated-ring",
  "mp_spdz_tag": "v0.4.3",
  "compiler_args": ["-R", "32"],
  "runtime_args": ["-R", "32"],
  "preprocessing": {"mode": "live", "batch_size": 10000, "file_preprocessing": false},
  "request_ladder": [256, 4096, 65536, 1048576],
  "chunk_candidates": [64, 256],
  "native_confirmation_n_choices": [12, 16, 20, 24],
  "hamming_confirmation_n": 24,
  "adoption_time_ratio": 0.95,
  "pilot_ratio_precision_target": 0.025,
  "development_seed": 20261023,
  "confirmation_order_seed": 20261024,
  "sizing_target_seconds": 20,
  "job_deadline_seconds": 30,
  "termination_grace_seconds": 5,
  "compile_deadline_seconds": 180,
  "budget_state": "results/budget_state.json",
  "budget_minutes": {"total": 480, "setup": 90, "implementation": 150, "experiments": 195, "report": 45},
  "stage_minutes": {"G0": 10, "G1": 15, "G2": 15, "G3": 20, "G4a": 40, "G4b": 45, "robustness": 15, "repair": 35},
  "stage_mpc_jobs": {"G0": 6, "G1": 6, "G2": 20, "G3": 24, "G4a": 48, "G4b": 64, "robustness": 8, "repair": 20},
  "max_mpc_jobs": 196,
  "max_compilations": 24,
  "max_exact_invocations": 12,
  "rss_gib_cap": 8,
  "rss_physical_fraction_cap": 0.25,
  "disk_gib_cap": 10,
  "disk_initial_free_fraction_cap": 0.20,
  "minimum_free_disk_gib": 10,
  "raw_logs_mib_cap": 512
}
```

Explicitly resolve the documented preprocessing batch setting in the pinned backend and record any compiler-driven reduction and actual generated/consumed amounts. If unused material or a buffer boundary appears to determine a result, use the single named robustness allowance to test that explanation; label the outcome as configuration-dependent. Do not mistake a frozen default for a universally optimized supplier.

### 5.4 Setup and commands

The following are **interfaces you must create**, not commands already tested during planning:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install psutil
python scripts/study.py inspect --config config/study.json
python scripts/study.py setup --config config/study.json
python scripts/study.py smoke --config config/study.json
python scripts/study.py check --config config/study.json
python scripts/study.py calibrate --config config/study.json
python scripts/study.py pilot --config config/study.json
python scripts/study.py confirm-native --config config/locked_native.json
python scripts/study.py develop-hamming --config config/locked_native.json
python scripts/study.py confirm-hamming --config config/locked_hamming.json
python scripts/study.py report --results results --output REPORT.md
```

The `check` command invokes standard-library unit tests under the same budget context and writes exact certificates. `pilot` creates `locked_native.json` only after the G3 requirements pass. `develop-hamming` checks the supported G4a gain, executes the bounded integration/development schedule, and creates `locked_hamming.json`; it must refuse to proceed when that gate failed or the complete remaining schedule cannot fit. The Hamming confirmation command must refuse to run without its prerequisite gate and lock file. The report command must work after any earlier stop. Add `--resume` only to load the existing ledger and remaining budgets; it must not reset them.

Documentation-supported MP-SPDZ command shapes, **also not executed in this planning work**, are:

```bash
git clone --branch v0.4.3 --depth 1 https://github.com/data61/MP-SPDZ.git vendor/MP-SPDZ
git -C vendor/MP-SPDZ rev-parse HEAD
make -C vendor/MP-SPDZ -j2 replicated-ring-party.x
```

Run the framework's SSL setup and compiler/driver from the proper project directory, after installing only dependencies required by this backend:

```bash
Scripts/setup-ssl.sh 3
./compile.py -R 32 component_smoke
Scripts/ring.sh component_smoke -R 32
```

These last commands are relative to the MP-SPDZ working directory. They are examples to execute through the setup/smoke supervisor, not unaccounted alternative workflows. Your setup script must install/copy the generated `.mpc` program there, capture the actual compiled program name, and inspect `--help` at the pinned revision. **Compile with `-R 32` and pass `-R 32` to every party at runtime** (the documented runtime default is 64), unless the pinned driver demonstrably adds that exact setting. Inspect actual party arguments and runtime ring metadata before canonicalizing revealed integers. Load `2**32 - 1` through a runtime secret input/register, add one inside the MPC program, and check that the raw revealed result is zero before any Python modulo normalization. Ensure this addition executes in the VM: a literal Python expression can be constant-folded before execution. Merely reducing a 64-bit result modulo 32 bits in the Python oracle could conceal the wrong backend and wrong source consumption. Use the official macOS build instructions; inspect `make setup` before using it because it may install unrelated backends [R6]. Do not install libOTe solely for Rep3 unless the selected build actually requires it.

If the system toolchain or a required install needs unavailable permission, complete unaffected work and report the exact blocker. Do not spend the whole budget installing unrelated tooling. A scalar cleartext benchmark is not a substitute for MPC preprocessing measurements.

### 5.5 Inputs, provenance, and development separation

No private dataset or paid service is needed. Sources are synthetic because the object under study is a random generator with a specified distribution. This is appropriate for mechanism and cost isolation, not evidence of application demand.

Use deterministic public test vectors with development seed `20261023` and confirmation-order seed `20261024`. These seeds control tests, bootstrap sensitivity checks, and run ordering only. Do not seed the MPC randomness with a common public test seed and then claim secrecy. Allow the framework's normal fresh source state for performance jobs; record that exact raw secret streams are intentionally not retained.

Generate all coefficients and code matrices deterministically. Save their hashes and full small matrices. Store committed source/config hashes before confirmation. Development and confirmation use separate job IDs and separate measurements; never pool pilot measurements into confirmatory intervals.

The brief is mathematically self-contained. If the attachments are transferred to the Mac, preserve them under `inputs/` with their original names and the hashes in Section 12. Their absence does not block the implemented mechanism or verification, but must be disclosed as a provenance limitation. The reproduction archive mentioned in [A1] was not received and is not a dependency.

## 6. Fair benchmark implementation and cost accounting

### 6.1 Primary measurement boundary

The primary observation is **complete batch wall time**, from launching the first of the three fresh party processes until all three exit successfully, with live source generation, actual required preprocessing, output-share materialization, and common fixed validation/sink work included. Compilation is outside that observation but is reported as setup cost and charged to the project budget. Start from normal fresh protocol state, without previously generated offline files supplying free work.

Also record framework timers around source generation through stored mask outputs, after normal initialization, and separate supplier-only and combination-only diagnostics. These explain startup and preprocessing effects; they never replace the complete measurement. Any preprocessing completed before an internal timer still belongs in the complete cost. Record cold first-run behavior separately from later warmed executable/OS-cache behavior. A claim restricted to an amortized boundary must be labeled as such.

All methods must use the same ring, party protocol, request count, output sink, logging level, and resource accounting. They may use different legitimate source representations and compiler conversion flags; this is precisely the end-to-end design choice being evaluated. Repeats use fresh randomness and independent requests. Do not force all methods to consume the same bit count or pay the same preprocessing bill artificially.

### 6.2 Materialize the requested outputs and inspect the program

Use runtime loops over chunks and write both mask-share arrays at all requested public offsets. Keep at most a chunk of intermediate sources live where practical. Do not replace the requested array of masks with one accumulated linear checksum that could be algebraically simplified. Inspect emitted instruction counts and a small integration trace to establish that all requested sources and outputs are evaluated.

For performance jobs, reveal only a small fixed number of dummy output entries after the internal generation timer; use the same count and sink pattern for every method. External complete time includes this common work. Correctness testing reveals enough dummy outputs separately to compare the whole small test. Benchmark outputs are disposable and never used to protect actual secrets.

For C129 require, per request, four logical full-word sources and one logical arithmetic bit. For N160 require five word sources. The combination must have no secret-secret multiplication or conversion. Count source-generation instructions and required material separately from local additions. Verify the Hamming input count and output conversion. Report actual generated versus consumed preprocessing where observable, including unused tails and buffering; otherwise state the missing counter and retain direct total measurements.

### 6.3 Mechanistic model and its limits

For explanatory purposes only, write

\[
T_C=4c_W+c_B+A_C,\qquad T_N=5c_W+A_N.
\]

The predicted break-even condition is

\[
c_W>c_B+(A_C-A_N).
\tag{4}
\]

The optimized C129 circuit uses five additions; N160 uses four. The source cost of the extra bit is the main unknown. Measure batched primitive costs and the complete circuits to see whether (4) explains the result. Buffering, shared setup, scheduling, and overlapped preprocessing can make these terms nonadditive; do not predict the winner solely by summing isolated timers.

The supplied-bit factorial diagnostic is:

| Construction | Individual arithmetic-bit supplier | Native word supplier |
|---|---|---|
| \(4w+1\) | Expanded C129 | Packed C129 |
| \(5w\) | Expanded N160 | Packed N160 |

This separates logical entropy reduction from the cost of representing the source. The fair practical comparison is the right-hand column. A gain only in the left-hand column supports a conditional existing-bit-interface observation, not a full-generation win.

### 6.4 Hardware and failure accounting

Run one MPC job at a time; the three parties of that job run concurrently. Keep the laptop plugged in, use a stable ordinary power mode, and record background load and thermal/power information available without privileged instrumentation. Do not change these conditions selectively between methods. Randomize and balance method order within blocks to reduce drift.

The supervisor owns each job's entire process group. On a deadline, cancellation, or failed party, terminate and then kill remaining descendants if necessary, record their statuses, and wait/reap all processes before another job can start. A lingering party or occupied port must not contaminate a subsequent trial. The wrapper records `perf_counter_ns` durations, exit codes, source/config hashes, method, width, request count, chunk size, compiler flags, party count, ring, preprocessing settings, communication bytes, available round counts, and resource peaks. Do not report both sent and received bytes as though they were distinct network traffic; define the aggregate convention, preferably the sum of sent bytes across parties.

Sample the sum of RSS of the complete process tree at about 50 ms intervals using `psutil`, including all three parties and helpers. Record compiler/build peaks separately and include them in project maxima. On macOS, `/usr/bin/time -l` can corroborate per-process peak RSS, but one process's peak is not aggregate concurrent memory. Report sampling limitations. Monitor workspace growth, new dependency/preprocessing files, and the project's free-space headroom.

Record every failed job, timeout, interrupted build, and repair. Do not silently exclude slow successful trials. A timeout is censored resource evidence, not a completed runtime and not a proof that the mathematical method fails. An invalid baseline blocks a broad positive claim. A valid loss against one mandatory comparator can establish the scoped negative without measuring every other comparator.

## 7. Minimal experiment matrix and calibration

### 7.1 Required work and purpose

| Work | Methods / cases | Scientific purpose | Timing status |
|---|---|---|---|
| Exact reference | C family, N160, Hamming, A144/A192, invalid controls | Validate functionality, assumptions, and the oracle | Deterministic; no repetitions for statistical significance |
| Backend smoke | Word/bit draws, wraparound, small conversion | Validate actual ring and APIs | Feasibility only |
| Calibration | C129/N160, primitive suppliers, empty harness | Size the meaningful workload and expose hidden costs | Development only |
| Mechanism pilot | C129/N160; small 2-by-2 supplier diagnostic; A144 and U64 controls | Isolate entropy, packing, preprocessing, and startup | Development only |
| G4a locked comparison | C129 versus N160 | First necessary condition for practical adoption | Confirmatory |
| G4b, only if justified | C129 versus frozen best viable H71 route | Test whether a positive additive result survives strong Boolean competition | Fresh confirmatory data |
| One bounded robustness check | Neighboring batch size or buffering boundary | Resolve a named mechanism/measurement risk | Secondary; never rescues the primary result |

The primary output is always two 32-bit masks per independent request. Widths 1-6 are proof checks, not substitutes for the practical workload. Do not add multiple protocols, large width sweeps, or general coefficient searches.

### 7.2 Cheap calibration before allocation

Begin with chunk size 64 and request count 256. Measure both primary native methods, the empty harness, and source primitives. The empty harness uses the same startup and fixed validation structure without source generation; use it diagnostically and **do not subtract it from the primary observation**.

Then consider request counts in the fixed ladder

```text
256, 4096, 65536, 1048576
```

subject to the resource guards. Probe chunk size 256 at one representative feasible size if the backend supports the required shapes. Choose a common chunk size by the lower sum of C129 and N160 development times, with ties within 2% keeping 64; this choice cannot be based on the candidate's relative gain. H71 may use multiple supported 64-bit packed lanes inside the same chunk.

Choose the largest ladder request count predicted to keep **both C129 and N160 below 20 seconds per complete job**, within memory/disk caps, and leave the remaining fixed confirmation within its time allowance. Prefer a generation interval of at least 0.5 seconds and startup below 10% of complete time. These latter preferences reduce timer/startup dominance; if they cannot be achieved, retain complete batch latency as the actual quantity, disclose the limitation, and use pilot precision to decide whether it is informative. Do not fabricate a steady-state throughput conclusion by subtracting noisy startup estimates.

Use adjacent measurements to estimate scaling, rather than assuming perfect linearity. Before a larger job, require predicted time and peak aggregate memory multiplied by **1.5** to fit their caps. The multiplier is a planning safety margin for buffering/intermediates, not a confidence bound. If behavior is superlinear, use the measured worse slope. Stop growth at the first infeasible step. Never allocate arrays proportional to \(2^{64}\), compile an unrolled program proportional to all requests, or enumerate width-32 output states.

If required workloads do not fit, smaller request counts preserve the same functionality but may lose timing precision. Use them only with appropriately narrowed latency/inconclusive conclusions. Reducing word width from 32 to make the practical result work is not an authorized substitute.

### 7.3 Pilot, freeze, and bounded tuning

Run six paired C129/N160 pilot blocks at the selected workload. In a block, run each method once in balanced randomized order. Run at most two complete trials of each supplier-factorial cell and two each of A144 and U64 at a feasible common diagnostic size. Those controls diagnose the mechanism; they are not candidates for a favorable performance headline.

Inspect supplier paths, costs, instruction counts, memory, and process logs. A large native-word advantage is evidence to test, not license to stop on an unconfirmed pilot. A noisy pilot is not a scientific rejection.

Compute the sample standard deviation \(s\) of paired log time ratios. Select the smallest \(n\in\{12,16,20,24\}\) with

\[
2.21s/\sqrt n\le\log(1.025).
\tag{5}
\]

The 2.5% approximate ratio half-width is chosen to resolve the 5% adoption threshold, independently of the observed mean gain. If none qualifies, allow one calibration-driven change in batch size or noise control, with at most six further pilot pairs charged to the repair reserve. If precision still appears unattainable within the remaining caps, classify the 5% timing decision as inconclusive, retain exact results and measured ranges, and report. Do not repeat pilots until a favorable mean appears.

Freeze `locked_native.json`: code and dependency hashes, protocol, all compiler/runtime/preprocessing options, primary timing boundary, request count, chunk size, planned \(n\), randomized run order, exact rejection rules, and remaining budget. No candidate-specific tuning after this point. In G4b, freeze the independently developed H71 route in a second lock file before collecting its fresh confirmation; keep the candidate and primary endpoint unchanged.

## 8. Confirmatory analysis and stopping rules

### 8.1 Independent unit and estimand

One **paired execution block**, not one generated mask, is the independent performance unit. For competitor \(J\), record

\[
\ell_k=\log(T_{C,k}/T_{J,k}).
\]

The primary estimate is the geometric mean time ratio \(\widehat\rho_J=\exp(\bar\ell)\). Report complete-time medians, the ratio, absolute time differences, and per-pair time as descriptive quantities. Do not treat a million masks in one job as a million independent timing samples.

Use the paired log-ratio interval

\[
[L_J,U_J]=\exp\left(\bar\ell\pm t_{n-1,0.975}\frac{s_\ell}{\sqrt n}\right).
\tag{6}
\]

For the allowed \(n=12,16,20,24\), the critical values are respectively approximately `2.200985`, `2.131450`, `2.093024`, `2.068658`. Each upper endpoint is a one-sided 97.5% bound; requiring it against two mandatory competitors gives a conservative familywise 95% superiority decision by the union bound. These are conventional sampling intervals under approximately independent, stable log ratios, not mathematical guarantees about runtime.

Inspect order/time plots and log-ratio dispersion. As a sensitivity check, bootstrap whole paired blocks with 10,000 fixed-seed resamples using the same endpoint probabilities. Do not select whichever interval is favorable. If the decision changes between the declared interval and bootstrap sensitivity, or evident uncontrolled drift undermines the block model, classify that threshold decision as inconclusive. Report the raw evidence. No post hoc outlier removal is allowed.

### 8.2 G4a: native comparator first

Collect exactly the frozen \(n\) valid paired C129/N160 blocks, within the failure rules below. There is no optional stopping based on intermediate ratios.

- If \(L_N>0.95\), the data support failure to achieve the predeclared 5% improvement against N160 in this setting. If \(L_N>1\), they support a runtime disadvantage. Finish the negative/limitation report; full H71 performance infrastructure is unnecessary.
- If \(U_N<0.95\), the native comparison supports a practically meaningful gain. Proceed to the bounded H71 development and fresh G4b confirmation if budget remains.
- If the interval touches or crosses 0.95, classify the primary adoption decision as inconclusive. Do not rerun confirmation to force a conclusion. Retain the exact construction and cost diagnosis. Additional work is limited to a predeclared robustness check that addresses a specific defect; it cannot replace the primary result.

This gate makes a well-supported early negative a completed study. It does not turn a timeout, unsupported API, or noisy pilot into a negative result.

### 8.3 G4b: strong Boolean comparison

Only after a supported G4a gain, validate and tune the bounded H71 routes as specified in Section 4.3, then freeze the fastest observed valid route. Use **exactly 24 fresh paired C129/H71 confirmation blocks**. This fixed maximum avoids sizing confirmation from a three-block route-selection sample. Do not recycle G4a timings. Check feasibility of the complete schedule before starting; a wide final interval remains inconclusive, and no additional confirmation is authorized.

The primary positive claim requires **both** \(U_N<0.95\) and \(U_H<0.95\), valid exact/integration checks, and no material missing cost or baseline defect. If \(L_H>0.95\), report a scoped negative for the overall adoption hypothesis despite the N160 gain. An interval touching or crossing 0.95 leaves the overall result inconclusive at the 5% threshold.

If H71-B is unsupported or its high-level source path is demonstrably inefficient and no fair bounded repair fits, a win against H71-A alone cannot establish superiority over the strongest Boolean alternative. Report the supported native-comparator result and the blocked broader conclusion. Do not quietly drop the missing comparator from the headline.

If a valid H71 job exceeds the cap, record its runtime as a lower bound and its status as censored. You may report that tested implementation's resource failure, but do not insert the timeout as an exact time in (6), or turn it into a universal claim against Hamming methods.

### 8.4 Failures, exclusions, and repairs

Infrastructure failure eligible for replacement means a documented method-independent external interruption, not merely an unusually slow run or an unexplained lost connection. An intrinsic resource timeout, method-specific failure, or cause-unknown censoring blocks ordinary complete-case inference for that phase: preserve it and report the relevant result as inconclusive, infeasible, or invalid. Do not repeatedly replace censored jobs until only successful runs remain. Preserve it and its reason. Allow at most **two replacement paired blocks per confirmatory phase**, using new predeclared-order slots and the shared job/time reserve. The failed blocks remain in the ledger and failure-rate table. If more are needed, or a method systematically fails, stop confirmation and classify the relevant performance evidence as inconclusive, infeasible, or invalid according to its cause.

A substantive correctness defect invalidates affected timings. Repair within the shared cap, rerun the affected independent checks, and discard affected timings from inference while retaining them as invalid records. Fresh confirmation is permitted only if its full preregistered schedule fits the original budgets; otherwise report the limitation. Changing an algorithm or primary configuration after seeing confirmation makes subsequent work exploratory unless a fresh, fully recorded confirmation is possible.

## 9. Cumulative resource ceilings and end-to-end effort

These are **unmeasured planning ceilings**, not M3 Max performance predictions. The study should normally reach an informative pilot within about one hour of executable readiness and resolve its central cost question in hours. A useful early stop is preferred to an over-budget benchmark.

### 9.1 Hard global limits

| Resource | Ceiling and enforcement |
|---|---|
| Total elapsed project time | **8 hours** from starting the coding task; includes setup, implementation, runs, failures, repairs, and analysis |
| Dependency/build/setup allowance | **90 minutes**; one normal build plus one bounded correction; include downloads/install time |
| Implementation allowance | **150 minutes** for scripts, references, backend programs, and fixes outside experimental execution; do not spend later-stage effort before gates justify it |
| Experimental/diagnostic allowance | **195 minutes**, including pilot, tuning, confirmations, failed runs, preprocessing, and execution repairs |
| Protected analysis/report allowance | **45 minutes**, reserved from the start; never spend it on more benchmark attempts |
| Aggregate process memory | \(M_{cap}=\min(8\text{ GiB},0.25\times\text{actual physical RAM})\), including concurrent parties; inspect actual availability as well |
| New project/dependency/preprocessing disk | \(D_{cap}=\min(10\text{ GiB},0.20\times\text{initial free disk})\); preserve at least 10 GiB free outside project growth |
| Raw results/log budget | 512 MiB within the disk cap; aggregate measurements, not every generated mask |
| Complete MPC jobs | **196** maximum, each containing three parties; include smoke, development, failures, repairs, and confirmations |
| Bytecode compilations | 24 maximum, each at most 3 minutes; charged to setup/implementation and global elapsed time |
| Complete performance job | 20 seconds is the workload-sizing target; **30 seconds is the hard execution deadline for every performance job**. Successful jobs taking 20-30 seconds remain in inference. At 30 seconds terminate the whole job, allow at most 5 further seconds to reap/kill it, and charge all elapsed time; record the observation as censored |
| Exact reference batches | 12 invocations maximum; individual deterministic character/histogram assertions are not separate experiments |

Persist a start timestamp and absolute eight-hour deadline before any setup or implementation, even before the full driver exists. Never launch new experimental work after deadline minus the protected 45-minute report reserve. Track all clocks and counts in one append-only ledger. Categories are disjoint for accounting; a minute cannot disappear between them. Overlapping activity does not authorize more work: no simultaneous independent MPC jobs, and both the global elapsed cap and each applicable allowance must hold. Any repair counts against the relevant category and the shared experimental repair allocation when it runs experiments. Restarting the driver, switching approaches, or reaching another gate does not reset a counter.

For memory, preflight at 1.5 times measured projected use; terminate a growing job before 90% of the cap if the next allocation would exceed it. If sampled RSS unexpectedly crosses the hard cap, terminate, preserve evidence, and reduce only a permitted workload parameter. Do not use swap as an unreported extension of RAM. If free disk headroom cannot be maintained, do not download/build more material. Clean only reproducible files created by this study after retaining manifests/results; never delete unrelated user data.

The source/reference formulas use negligible memory. The practical hidden costs are MP-SPDZ build/compiler memory, preprocessing buffers, three concurrent parties, retained output arrays, and Hamming intermediate registers. With chunking, source/intermediate storage should scale with chunk size, and output storage with request count. Verify actual scaling; Python/VM object overhead can dominate raw word sizes.

### 9.2 Stage runtime and job allocation

The following runtime allocations sum to the 195-minute experiment allowance. Setup and implementation are separate as above; every stage still observes the global deadline and protected report reserve.

| Stage | Execution/diagnostic minutes | MPC jobs, maximum | Evidence priority |
|---|---:|---:|---|
| G0 environment/source feasibility | 10 | 6 | Actual backend, ring and source API |
| G1 tiny correctness/integration | 15 | 6 | Exact checks and sensitive invalid controls |
| G2 resource calibration | 15 | 20 | Safe size, real supplier cost, memory/startup |
| G3 mechanism pilot | 20 | 24 | Cause of effects, precision, first protocol lock |
| G4a locked native comparison | 40 | 48 | Necessary adoption comparison |
| G4b Hamming integration + development + confirmation | 45 | 64 | Strong Boolean competitor if justified |
| Named robustness check | 15 | 8 | One concrete remaining risk |
| Shared repair/retry allocation | 35 | 20 | Bounded corrections, replacements, one extra pilot |
| **Total** | **195** | **196** | No stage or approach resets these totals |

Do not run all allocated jobs merely because capacity remains. The expected engineering commitment is roughly 2-4 hours for setup/implementation plus up to 3.25 hours of experimental work and 45 minutes of reporting; these are estimates. The eight-hour hard cap overrides estimates. If setup or implementation consumes its allowance, finish the unaffected exact analysis and report a practical execution limitation rather than starting a multi-day rescue effort.

Before every large job or new stage, require enough remaining time for its entire fixed schedule plus 45 minutes for analysis, allowing measured startup and 1.5 times projected experimental cost. If not, stop or follow the specified narrow fallback. Do not start half a confirmatory schedule whose completion is predictably impossible.

## 10. Evidence gates and autonomous next actions

| Gate | Question and prerequisites | Minimum artifacts | Decision and next action |
|---|---|---|---|
| **G0: constructive feasibility** | Can the exact source contract and ring32 backend be realized on the actual target? Read this brief; inspect hardware/toolchain and source APIs before a large build. | Environment manifest; source inventory; resolved MP-SPDZ commit; tiny word/bit/conversion smoke records; clear statement of absent real deployment. | Continue if native word and bit sources, ring arithmetic, and caps are credible. One bounded documented build/API repair is allowed. If only target access/backend is blocked, complete exact reference/proofs and report blocked performance. If a source-law assumption is false, do not replace it silently. |
| **G1: tiny correctness** | Are the map, code, representation, and security oracle correct? Requires clear reference and feasible backend smoke. | Exact histograms, integer character minima, Hamming rank certificate, invalid-control failures, backend dummy-vector comparisons. | Continue only with zero unexplained mismatches. Repair one ordinary defect inside shared limits. A verified counterexample to an essential mathematical claim is an analytic stop; preserve it and revise the conclusion. A persistent implementation defect is invalid/inconclusive, not a theorem refutation. |
| **G2: calibration** | What really costs time and can meaningful width-32 work fit? Requires G1. | Supplier/complete timings, emitted instructions, memory/disk scaling, startup fraction, workload choice and projection. | Choose workload by Section 7, not observed relative gain. Resize request count/chunking once when justified. If no meaningful target workload/precision fits, report practical infeasibility or inconclusive timing; retain exact results. |
| **G3: mechanism pilot** | Can the intended effect be separated from packing, supply, startup and noise? Requires calibrated workload. | Six paired pilot blocks; small factorial/control table; source/preprocessing diagnosis; frozen native protocol. | Select fixed \(n\) using variance alone. One bounded extra pilot is allowed only to resolve precision/measurement uncertainty. Noisy or unfavorable pilot means no scientific verdict yet. If corrected measurement remains inadequate, stop inconclusive. |
| **G4a: locked native comparison** | Is C129 materially faster than N160 under full costs? Requires G3 lock. | Frozen schedule; all paired raw records and failure statuses; interval (6). | Apply Section 8.2 exactly. Supported failure against N160 completes a useful negative. A threshold overlap is inconclusive. Only a supported gain advances to full H71 development. |
| **G4b: locked Hamming comparison** | Does a native-comparator gain survive strong Boolean coding? Requires supported G4a gain and remaining budget. | Verified H71 route selection; second lock; fresh paired data; interval and source-path limitations. | Both required upper bounds below 0.95 support the scoped performance contribution. A supported H71 failure to meet the gain rejects overall adoption. Missing strong baseline or threshold overlap leaves the broad claim inconclusive. |
| **G5: diagnosis and report** | What conclusion is justified by the reached gates? Always execute, including after a stop. | Reproducible report, available code/checks/raw results, regenerated tables, deviations and one justified next step if any. | Stop further experiments at the reserved reporting boundary. State exact supported scope, resources consumed, blockers and unresolved questions. |

A justified analytic or resource stop satisfies the task when its evidence is reproducible. Do not invent an empirical comparison or additional algorithm simply to fill unreached stages.

## 11. Outcome interpretation and final artifacts

Classify separate claims separately. For example, the mathematical construction can be supported while the application hypothesis is negative or untested.

| Outcome | Required basis | Wording to avoid |
|---|---|---|
| **Supported contribution** | General derivation plus exact implementation checks; for a speed claim, both mandatory frozen comparisons satisfy the predeclared criterion with all costs and valid baselines | “Optimal 129 bits,” “best cryptosystem,” or universal speed superiority |
| **Informative negative result** | Valid adequate comparison rules out the selected 5% gain in the measured scope, or a verified mathematical obstruction/counterexample resolves a claim | Treating a bug, missing API, or noisy pilot as a refutation |
| **Practical infeasibility** | Measured resource costs or a justified scaling argument prevent the intended study/application within the fixed limits | Claiming an irrelevant small-width success instead |
| **Inconclusive evidence** | Precision, coverage, target access, strong-baseline access, or budget prevents a justified decision | Presenting failure to prove a gain as proof of no gain |
| **Invalid experiment** | Defective map, baseline, source law, measurement boundary, or accounting invalidates attempted comparisons | Retaining favorable invalid timing results |

### Required final report

Produce `REPORT.md`, runnable code and concise usage instructions for the stages reached, the environment/config/source manifest, exact certificates/counterexamples, raw measured results and failure ledger if any, and a single command that regenerates essential tables from saved local files. Do not require remote computation to reproduce the study.

The report must contain:

1. The decision, outcome category, and precise supported scope first.
2. The source/leakage contract, per-request batching rule, ideal-versus-implementation distinction, and all material assumptions.
3. The C129 derivation, source-bit comparison, and clear lack of general optimality/priority claims.
4. Correctness evidence, including the failing controls and any attachment claims not reproduced.
5. Hardware/software versions, actual ring/backend, complete timing boundary, workload, tuning allowances, locks, repetitions, and all deviations.
6. A table of source costs, total/inner timings, uncertainty, startup, preprocessing, communication, peak aggregate RSS, failure rates, and setup cost. Mark unperformed cells explicitly; do not enter zero.
7. A compact end-to-end cost plot and supplier/combination breakdown if timing data exist. Show raw paired ratios or a table sufficient to inspect variability. If no timings exist, omit performance plots and include the exact diagnostic instead.
8. An explanation of whether logical entropy savings, source packing, conversion cost, or implementation overhead caused the result. Discuss alternative explanations supported by the data, not just the intended mechanism.
9. Resource totals including unsuccessful work; reached/failed gates; unresolved limitations; and whether one specific bounded next step is worth doing. A speculative hardware/protocol port is not automatically recommended.

Preserve unfavorable results. Keep reproducible public test inputs and exact certificates; do not publish live protocol seeds, credentials, or reusable secret preprocessing as research artifacts. Record that performance source streams are fresh and not reproducible bit-for-bit, while program behavior, settings, and statistical protocol are reproducible.

## 12. Planning evidence, provenance, and claim status

### 12.1 What was actually done before this handoff

The planning investigation read both supplied documents, inspected the PDF's relevant formulas directly, researched decision-relevant primary literature and official implementation documentation, used three independent review agents for mathematics, systems, and experimental design, and reconciled their findings. These were preparation activities, **not the completed PoC**.

Small hosted-Linux probes actually executed:

| Probe | Observed result | Limit |
|---|---|---|
| Exact character check of supplied nine-bit list | All 15 nontrivial characters covered at least 3 times; maximum 4 | Confirms this finite list, not the general proof by experiment |
| Small integer multicover exploration | Solver reported lengths 5, 9, 13, 17 optimal at widths 1-4; widths 5 and 6 hit 20-second limits with feasible lengths 21 and 25 | Floating MILP bounds are not independent exact optimality certificates; no general optimality inferred |
| General family (2), independent exact character checks | All characters at widths 1-6 passed; counts 3, 15, 63, 255, 1,023, 4,095; minimum coverage always 3 | Width 32 rests on the analytic proof |
| Independent convolution of family (2) at widths 1-3 | All 16, 46, 92 omission sets passed | Remaining required agent checks still need execution |
| Four-word-only invalid control at width 2 | Eight reachable outputs of sixteen after fixing two designated bits; TV 1/2 | Confirms test sensitivity to this failure |
| Native five-word baseline at width 2 | All 56 omission sets passed | No MPC implementation timing |
| Joint four-output-word global-budget example at width 2 | All 106 omission sets passed for the 14-bit lifted Hamming-layer construction | Illustrates batching-contract limitation only |

An independently checked 17-coefficient width-4 solver witness also preceded the general formula. Do not implement a separate MILP search in the PoC: it is no longer needed. No MP-SPDZ backend was installed, no target Mac was inspected, no source primitive was timed on that Mac, and none of the proposed project commands was executed during planning. All resource/performance expectations remain estimates until your measurements.

### 12.2 Supplied files

| Identifier | File and received location | SHA-256 |
|---|---|---|
| A1 | `robust_uniform_sums_research_note.pdf`; received as `upload/robust_uniform_sums_research_note.pdf` | `4dffebf59644b586de9ebdf7821ab6cdc92f2caf6b303db8a3826f5dfa36e4f8` |
| A2 | `Pasted text(20261023-165633).txt`; received as `upload/Pasted text(20261023-165633).txt` | `69e974e6a0cfedbe3426d1e260b659bae248eee56ba33da0838795cca9ff9174` |

At planning time these were under `/workspace/scratch/f5a9d1e8f5ba/`. That hosted path is provenance, not a path you should expect on the Mac. If transferred, use the repository's `inputs/` directory. A1 is titled *Exact Uniform Sums After Omissions: Sharp sample-space bounds and a complete profile on (Z/4Z)²*, attributed to “Autonomous mathematical research investigation,” dated 23 October 2026. A2 contains the application proposal and requirements for this brief.

### 12.3 Central claim-to-evidence map

| Claim | Basis and status | Remaining uncertainty / resolving action |
|---|---|---|
| Nine-bit modulo-four construction is valid and additive-optimal | A1 Theorem 4.1, §4.4; lower bound and integer check audited | Recreate exact implementation checks; priority not established |
| A144 lift is valid, but not strongest | Carry argument in A2 and Section 3.4; stronger explicit family derived here | Recreate raw-lift tests; do not insert nonlinear truncation |
| C129 has the required leakage property for all widths | Two complete derivations in Section 3.3 and finite independent checks | Verify generated coefficients and backend; no width-32 optimum claimed |
| N160 is a strong native baseline | Exact regrouping and odd-determinant proof | Measure actual native supplier, not literal bit-by-bit supply |
| H71 gives the same bit-leakage protection using 71 logical bits | Standard coding/extractor result, explicit matrix, rank argument | Exact rank/integration checks; fair native Boolean source/conversion |
| Arithmetic words and bits may have different costs | Official MP-SPDZ source paths, not a numerical cost assumption | G2/G3 actual measurements and counters |
| A practical gain may exist | Hypothesis; equation (4) explains conditions | G4a and, if justified, G4b |
| Real systems benefit or have this leakage-bearing interface | Not established by the supplied materials or research | No production claim; a future concrete supplier/caller would need validation |

## 13. Research coverage, novelty, and review disposition

### 13.1 Decision-relevant research

Research was conducted on **23 October 2026** using exact mixing, robust/omission-tolerant sums, separable resilient functions, bit-fixing extractors, correlation immunity, orthogonal-array fibers, quaternary/additive codes, mixed-alphabet arrays, and mixed arithmetic/Boolean MPC terminology. Primary content was inspected where accessible, rather than relying solely on search snippets.

Ordinary exact mixing is established: finite 2-groups have mixing length \(\log_2|G|\) without the omission requirement [R1, Corollary 3.1]. Resilient functions, Fourier/OA characterizations, and the linear-code extractor construction are also established [R2-R4]. They explain why the seven-/71-bit Boolean comparisons are mandatory and why mere renaming is not novelty. The application's distinct constraint is additive evaluation in the arithmetic ring.

Backward references from the attachment identify older uniform-convolution and simultaneous-factorization work; the note itself records partial access to Sherstnev and uncertain historical priority. That historical search was not fully replicated. The closest operational alternatives were investigated through MP-SPDZ's official source, current release, mixed-circuit documentation, and linked daBit/edaBit/ABY3 papers [R5-R12]. Official docs/source were available; several full publisher/ePrint PDFs returned access errors. Their landing-page abstracts and metadata were checked, but their cryptographic proofs were not independently audited here.

No equivalent statement of the explicit \(4w+1\) construction was located in this bounded search. This is not evidence sufficient to certify novelty. Mixed-alphabet orthogonal arrays and ring erasure/resilient constructions are plausible alternative formulations. Describe the result as **derived and verified in this investigation**, with historical novelty unresolved. The executed study can still contribute a useful implementation characterization or limitation even if the formula is later found in prior literature.

Do not extend the literature survey indefinitely during implementation. Allow at most 20 minutes within G0/setup to resolve a new source fact that could change the selected mechanism or a mandatory comparator. If a directly equivalent implementation is found, credit it and frame the work as a new measured regime/limitation only if that characterization remains useful. If no credible new conclusion remains, perform the smallest reproducible diagnostic and say so.

### 13.2 Three review passes and material resolutions

The following records the substantive review findings, not a claim of perfection. The final brief was subjected to distinct mathematical/novelty, experimental, and execution reviews; residual dependency questions are deliberately early gates.

| Review / stress scenario | Material finding | Disposition in this brief |
|---|---|---|
| Mathematical: larger-word extrapolation | The 144-bit lift is valid but a stronger \(4w+1\) family exists | Selected C129; complete proof and exact-check obligations supplied |
| Mathematical: nonconstructive theorem or hidden oracle | Essential mechanisms have explicit lists/formulas; no optimizer is needed | Solver exploration excluded from required implementation |
| Mathematical: carries / finite precision | Per-block secret modulo can add hidden nonlinear work; floating Fourier zeros would be unreliable | Raw final-ring lift; exact integer congruences and reference arithmetic |
| Novelty: equivalent known method | Resilience, coding, OA bounds and exact mixing are known; construction priority unresolved | Credit foundations; no optimality/priority headline; useful measured result remains possible |
| Experimental: weak additive baseline | The 160-bit method packs into five native words | N160 mandatory; literal-bit version only an ablation |
| Experimental: strongest Boolean method erases gain | Hamming uses 71 bits and may convert efficiently | Native Boolean route required for broad positive conclusion; optimized arithmetic route also considered |
| Experimental: preprocessing dominates | One arithmetic bit may cost more than one native word | Supplier audit and complete timing; early locked N160 gate |
| Experimental: source information is unequal | Logical leakage coordinates differ from internal preprocessing, and extra output entropy could be hidden | Explicit source inventory for every method; count all costs |
| Experimental: only favorable synthetic/batched cases | A global two-disclosure batch permits stronger joint coding | Fixed per-request contract; disclose 112-bit-per-pair two-pair counter-comparison |
| Experimental: bias/correlation breaks assumptions | Exact theorem does not repair a source | Biased-bit invalid control; ideal/computational distinction and supplier gate |
| Experimental: small pilot inconclusive | Pilot noise does not refute the mechanism | Variance-based fixed confirmation; one bounded precision repair; inconclusive category |
| Execution: Mac/dependencies unavailable | M3 Max is specified but actual RAM, software and availability are unknown | Inspect first; caps; exact-only fallback with performance explicitly blocked |
| Execution: undocumented/incorrect API assumption | Scalar Boolean random source and conversion naming can mislead | Pinned-source smoke, explicit safe packing route, bounded repair |
| Execution: runtime ring differs from compiler | Runtime defaults to 64 bits and public normalization could conceal it | Explicit ring32 compiler and party arguments, raw wraparound and metadata check |
| Experimental/execution: deferred H71 work and sampling | Full H71 integration was initially placed too early; three selection blocks cannot support precise sample-size planning | Defer route integration to G4b; three selection blocks followed by fixed 24 fresh paired confirmation blocks |
| Experimental: timeout censoring | Replacing intrinsic timeouts could retain only favorable complete runs | One 30-second hard deadline; retain successful slower runs; no complete-case inference after intrinsic or unexplained censoring |
| Execution: runtime/memory exceeds estimates | Compiler unrolling, preprocessing and all-party memory are hidden costs | Runtime chunks, process-tree monitor, 1.5 scaling margin and cumulative ceilings |
| Execution: missing source/archive | Attachment reproduction archive absent; some papers inaccessible | Complete formulas/checks in brief; no dependency on missing archive or unverified proof |
| Main hypothesis false | Logical entropy improvement need not save complete generation time | Valid negative is a completed result; no forced large benchmark or repeated rescue |

## 14. References and access notes

Use these ordinary references in the final research report. Add resolved software commits and exact dependency versions from execution. All web access dates below are **23 October 2026**. “Read” distinguishes substantive content access from abstract-only access.

- **[A1]** *Exact Uniform Sums After Omissions: Sharp sample-space bounds and a complete profile on (Z/4Z)².* Autonomous mathematical research investigation, 23 October 2026. Supplied `robust_uniform_sums_research_note.pdf`, 17 pages; locations and hash in Sections 3.1 and 12.2. Read in full; relevant equations visually inspected. Not assumed peer reviewed or historically novel.
- **[A2]** *Pasted text(20261023-165633).txt.* Supplied application proposal and execution-brief instructions, 23 October 2026. File location/hash in Section 12.2. Read in full. Its previous computational claims remain author-reported except where this brief explicitly records new checks.
- **[R1]** Gideon Amir, Guy Blachar, Subhajit Ghosh, Uzi Vishne. *Mixability of Finite Groups.* Random Structures & Algorithms 67(4), e70037, 2025. [DOI: 10.1002/rsa.70037](https://doi.org/10.1002/rsa.70037); [open full text](https://onlinelibrary.wiley.com/doi/full/10.1002/rsa.70037); [preprint](https://arxiv.org/abs/2501.17806). Relevant locations: Definition 2.1; Corollary 3.1; §5 decomposition/representation criterion. Relevant primary content read. Does not establish the omission optimization claimed here.
- **[R2]** Mahdi Cheraghchi, Frédéric Didier, Amin Shokrollahi. *Invertible Extractors and Wiretap Protocols.* arXiv:0901.2120v2, revised 22 July 2011; earlier conference version, ISIT 2009. [Full text](https://arxiv.org/html/0901.2120v2). Relevant location: §IV-B1, **Theorem 13**, linear-code generator matrices as exact symbol-fixing extractors; §III for the related but distinct wiretap model. The theorem and surrounding primary text were read.
- **[R3]** Paul Camion, Anne Canteaut. *Correlation-Immune and Resilient Functions Over a Finite Alphabet and Their Applications in Cryptography.* Designs, Codes and Cryptography 16, 121-149, 1999. [DOI: 10.1023/A:1008337029047](https://doi.org/10.1023/A:1008337029047); [author publication page](https://www.rocq.inria.fr/secret/Anne.Canteaut/Publications/Camion_Canteaut98.html). Relevant subject: orthogonal-array, Fourier and matrix characterizations. Author/publisher abstract and indexed passages were available; full official manuscript retrieval failed. Essential facts used here have explicit elementary derivations in Sections 3.2 and 3.4; no uninspected detailed theorem is a dependency.
- **[R4]** Venkatesan Guruswami. *Introduction to Coding Theory: Notes 1, Introduction and Linear Codes.* Carnegie Mellon University, January 2010. [Author-hosted notes](https://www.cs.cmu.edu/~venkatg/teaching/codingtheory/notes/notes1.pdf). Relevant locations: §1, p. 3, explicit Hamming example; §3.1, p. 6, Lemma 9 on parity-check columns and distance; §3.2, pp. 7-8, Hamming construction. Relevant text read. Use rank-nullity and the stated code parameters rather than an apparent dimension typo in a proof sentence.
- **[R5]** Marcel Keller. *MP-SPDZ: A Versatile Framework for Multi-Party Computation.* ACM CCS 2020. [DOI: 10.1145/3372297.3417872](https://doi.org/10.1145/3372297.3417872); [official repository](https://github.com/data61/MP-SPDZ); [v0.4.3 release, 6 July 2026](https://github.com/data61/MP-SPDZ/releases/tag/v0.4.3). Official software citation, release and implementation documentation read; no independent audit of the full framework's security is claimed.
- **[R6]** MP-SPDZ contributors. *Getting Started*, current documentation. [Documentation](https://mp-spdz.readthedocs.io/en/latest/readme.html). Precise headings: Requirements; Arithmetic modulo 2^k; Honest majority; Mixed circuits; Local share conversion; Finding the most efficient variant; Compiling and running programs from external directories. Read substantively. Apple Silicon support and command shapes are documentation evidence, not an executed M3 validation.
- **[R7]** MP-SPDZ contributors. *Compiler.types source documentation.* [Rendered source](https://mp-spdz.readthedocs.io/en/latest/_modules/Compiler/types.html). Exact symbols: `sint.get_random`, `sint.get_random_int`, `get_random_bit`, `get_dabit`, `get_edabit`, `bit_compose`. Read substantively; distinguish native `randomfulls` from bit/edaBit composition. Pin and recheck the executable revision.
- **[R8]** MP-SPDZ contributors. *Compiler.GC.types* and *Compiler.GC.instructions source documentation.* [Types](https://mp-spdz.readthedocs.io/en/latest/_modules/Compiler/GC/types.html); [instructions](https://mp-spdz.readthedocs.io/en/latest/_modules/Compiler/GC/instructions.html). Exact symbols: `sbits.get_random_bit`, `sbitvec.from_vec`, `sbitintvec`, and scalar `bitb`. Relevant source read. This is the basis for the Boolean-source API gate.
- **[R9]** MP-SPDZ contributors. *Run-Time Options.* [Documentation](https://mp-spdz.readthedocs.io/en/latest/runtime-options.html). Relevant headings: Efficiency trade-offs, preprocessing batch sizes, file preprocessing, bits from squares. Read substantively. Defaults and overproduction must be verified at execution, not inferred from unrelated protocols.
- **[R10]** Daniel Escudero, Satrajit Ghosh, Marcel Keller, Rahul Rachuri, Peter Scholl. *Improved Primitives for MPC over Mixed Arithmetic-Binary Circuits.* CRYPTO 2020, pp. 823-852. [DOI: 10.1007/978-3-030-56880-1_29](https://doi.org/10.1007/978-3-030-56880-1_29); [ePrint 2020/338](https://eprint.iacr.org/2020/338); [author-institution record](https://pure.au.dk/portal/en/publications/improved-primitives-for-mpc-over-mixed-arithmetic-binary-circuits/). Abstract/metadata and framework usage documentation read; full PDF access failed. Establishes the relevance of edaBit conversion machinery, not a claim that C129 improves those protocols.
- **[R11]** Dragos Rotaru, Tim Wood. *MArBled Circuits: Mixing Arithmetic and Boolean Circuits with Active Security.* INDOCRYPT 2019. [ePrint 2019/207](https://eprint.iacr.org/2019/207). Abstract/metadata read; full PDF access failed. MP-SPDZ's documented daBit route points to §4.1. No implementation of this protocol from scratch is required.
- **[R12]** Payman Mohassel, Peter Rindal. *ABY3: A Mixed Protocol Framework for Machine Learning.* ACM CCS 2018. [DOI: 10.1145/3243734.3243760](https://doi.org/10.1145/3243734.3243760); [ePrint 2018/403](https://eprint.iacr.org/2018/403). Abstract/metadata and official MP-SPDZ local-conversion guidance read; full PDF access failed. Use the supported framework route rather than claiming an independently verified conversion proof.

---

**Execution instruction:** start with G0, create only the infrastructure needed for the next evidence gate, execute each reached stage, inspect the results, and follow the declared decisions autonomously. Preserve evidence and write the final report even when the most useful result is a limitation, blocked comparison, or early stop.
