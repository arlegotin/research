# POC_EXECUTION_BRIEF

## Decision: independently verify a scoped structural no-go before building a defect benchmark

**Execution target:** an autonomous coding agent configured as **gpt-6.1-sol with xhigh reasoning**.  
**Prepared:** 1 November 2026.  
**Project:** Reachability information after conventional interaction coverage.  
**Deliverable from you:** a small reproducible diagnostic, its independently checkable evidence, and an honest research report.

You are to implement and execute the diagnostic specified here, inspect the results, repair ordinary implementation faults within the stated limits, and finish the report autonomously. You do not need the preceding conversation. The essential mathematics, inputs, algorithms, expected observations, and decision rules are below.

**The research decision is to stop the proposed reachability-prioritization branch if the supplied structural obstruction is independently confirmed.** Planning-stage calculations found that a conventional greedy pairwise seed, on eight Boolean flags, naturally produces an eight-row catalogue whose complemented-diagonal outputs contain every nonrow configuration. Consequently, adding reachability as a tie-break to an otherwise identical coverage selector cannot change any additional test choice.

This is an exact, useful limitation of a specific proposed integration. It is **not a rejection of all theorem-inspired testing, a new matching theorem, or an empirical result about real feature-flag defects**. Do not manufacture a software benchmark to measure a mechanism that has no selection effect. Do not turn the constructed counterexample later in this brief into a favorable application benchmark.

### 1. Research decision, contribution, and scope

#### 1.1 The practical purpose and the retained research question

The original purpose is to find additional consequential configuration-dependent bugs after mandatory tests, at the same total release-testing cost, using information from the supplied diagonalization note. Mandatory tests, application requirements, workloads, and correctness oracles must remain fixed. A distinction between mathematically reachable and unreachable configurations is valuable only if it improves that engineering outcome.

The first integration considered here is deliberately modest: retain a conventional coverage objective and use the note's reachability information only to resolve equal-coverage candidate choices. All supported configurations remain selectable. No competitor is artificially restricted to diagonal outputs.

**Primary question to resolve:**

> After the exact eight-flag, naturally square, greedy pairwise catalogue specified below, can a complemented-diagonal reachability tie-break change a conventional three-way coverage selector's sequence of additional configurations?

The answer anticipated from the planning checks is **no**. Your task is to verify the inputs, construction, theorem assumptions, exhaustive certificates, and selector equivalence independently. The stronger practical consequence follows conditionally: identical configuration sequences cannot reveal different root defects at the same completed-test prefix when the application, workload, and oracle are identical.

The practical effect threshold motivating the original project is at least **one additional independently confirmed consequential root defect within the same total testing budget**. This is a minimal decision threshold, not a statistical estimate or a claim that every extra defect has equal business value. For this added tie-break under the additive-cost model in §4.4, changing the selected configurations is necessary for an information-driven detection gain. The diagnostic tests that condition exactly. For the primary catalogue, the expected selection difference is exactly zero for every possible deterministic failure map.

#### 1.2 Why the project was narrowed

Small mathematical probes were authorized during preparation of the brief. They found a zero-information regime before an application harness was justified. A subsequent review also found that forcing a coverage catalogue to contain exactly as many rows as flags could append zero-gain rows. The final primary construction therefore stops immediately at full pairwise coverage; its square shape is an observed output, not a requirement imposed by padding.

This is an **exploratory design revision informed by those observations**. Do not describe the later reproduction as a prospectively blinded discovery or treat the chosen case as a random sample of production practice.

| Original aim | Treatment in this project | What the conclusion can cover |
|---|---|---|
| Use theorem information to improve additional configuration selection | Retained as a precisely specified reachability tie-break | Whether this integration has any selection effect for the fixed catalogue |
| Begin after mandatory tests | Represent their configuration set by \(M\), with all catalogue rows already tested | A conditional statement; no actual application suite is claimed to have passed |
| Use one component and real historical bugs | Deferred after the structural gate fails | No claim about measured bug detection, production severity, or particular release safety |
| Use roughly 8–12 Boolean flags | Primary case has 8; a 10-dimensional literal matrix is a mathematical control | Exact finite configurations, not scale extrapolation |
| Compare with established interaction testing | Use conventional greedy coverage and its otherwise identical augmented version | Exact attribution of the proposed extra information |
| Count total cost | Retain a precise additive-cost implication and measure diagnostic cost | No unmeasured claim of wall-clock slowdown or application savings |

**Smallest defensible contribution:** an independently reproducible applicability test showing that this particular reachability augmentation is redundant after the specified conventional seed, together with a counterexample preventing overgeneralization. It is an R&D decision aid. Publication-level novelty has not been established.

**Decision informed:** do not spend the remaining R&D allocation implementing or benchmarking this exact augmentation on a software subject once the obstruction is verified. The original empirical opportunity remains open for independently justified, nonsaturated catalogues or materially different uses of the theorem.

#### 1.3 Fixed scientific choices and permitted engineering discretion

The primary matrix, catalogue-generation rule, stopping rule, bit convention, score, tie priorities, reference checks, controls, outcome categories, and resource ceilings are fixed. You may organize functions, choose efficient integer representations, improve diagnostics, or repair implementation errors. You may not change a matrix, seed, stop rule, score, or expected result to obtain a preferred outcome.

Do not search for a replacement application or a nonsaturated catalogue during this project. A disagreement with an expected result triggers bounded diagnosis, not a rescue campaign. A verified counterexample to this brief's central claim is a valuable result and must be reported.

The work excludes adaptive catalogue replacement, stateful flag transitions, races, context drift, multivalued flags, production traffic, dynamic test assertions, automated deployment decisions, and claims of whole-application correctness.

---

## 2. Mathematical contract and source audit

The supplied note is *How far finite diagonalization can reach*, in **diagonal_escape_note.pdf**, 9 pages, dated 1 November 2026 [1]. Its equations were inspected in rendered pages because ordinary text extraction omits much of the mathematics.

The attachment *Pasted text(20261101-120259).txt* contains the original application problem and execution requirements [2]. The note mentions accompanying Python and JSON verification artifacts, but those artifacts were **not supplied**. Do not claim to possess or have reproduced its larger exhaustive run.

### 2.1 Objects, dimensions, and meanings

Let \(A=(a_{ij})\in\{0,1\}^{n\times n}\). Rows are indexed; let \(C=\{a_i:0\le i<n\}\) be their set of distinct values. Define

\[
D(A)=\left\{x\in\{0,1\}^n:
x_j=1-a_{\pi(j),j}\quad\text{for some permutation }\pi\in S_n
\right\}.
\]

Each output uses exactly one entry from every row and column, then complements it. Every output differs from every indexed row, so \(D(A)\cap C=\varnothing\) [1, §1, p. 1].

For application interpretation, \(V\subseteq\{0,1\}^n\) is the independently specified supported domain and \(M\subseteq V\) is the already tested mandatory configuration set. The equivalence claim requires \(C\subseteq M\). The primary diagnostic uses the full cube \(V=\{0,1\}^8\) and \(M=C\). This is a formal test-selection scenario; no real deployment catalogue is being asserted.

Keep three concepts separate:

- **Not yet selected:** the selector may select the configuration but has not done so.
- **Unreachable by the frozen diagonal construction:** the configuration is not in \(D(A)\).
- **Invalid or outside scope:** the configuration is not in \(V\).

The selectors in this project can select any point in \(V\setminus M\). “Unreachable” describes only the reference diagonal construction, not those selectors.

### 2.2 Exact matching and repair distance

For target \(x\), form a bipartite graph \(G_x\) with row vertices \(i\), coordinate vertices \(j\), and an edge precisely when \(a_{ij}\ne x_j\). Let \(\nu(G_x)\) be the maximum matching size. Then

\[
x\in D(A)\ \Longleftrightarrow\ \nu(G_x)=n,
\]

and

\[
\delta_A(x)
=\operatorname{dist}_H(x,D(A))
=n-\nu(G_x)
=\max_{I\subseteq\{0,\ldots,n-1\}}
\left(|I|-|N_x(I)|\right),
\tag{1}
\]

where \(N_x(I)=\{j:\exists i\in I,\ a_{ij}\ne x_j\}\); the empty subset contributes zero [1, Proposition 1, Eq. (1), p. 2]. Repeated rows are allowed in this result, but their indices and multiplicities must be retained.

The computational bridge is constructive:

1. Compute a maximum disagreement matching.
2. Flip \(x\) in each unmatched coordinate.
3. Match the unmatched rows bijectively to those coordinates.

No unmatched row can have an existing disagreement edge to an unmatched coordinate, since that would enlarge the maximum matching. Binary complementation creates all such missing edges and preserves matched coordinates. Exactly \(n-\nu(G_x)\) flips suffice. Conversely, changing \(s\) coordinates of a matched output can remove at most \(s\) matching edges, proving optimality.

This is a nearest-point statement in the **unconstrained cube**. The returned point can be invalid, and \(D(A)\cap V\) can be empty. It does not solve nearest valid diagonal output. The selector does not use repair; its only required repair execution is the two-bit constraint guard in §4.2. The exhaustive tiny checks verify the exact distance formula independently without needing to execute repair on every target.

The set-valued tableau permanent and its matching formulation are established prior art. Brlek et al. give the permanent definition and the polynomial matching test in §1 and Theorem 8, preprint pp. 6–7 [3]. The note uses the complemented tableau. Do not claim that maximum matching or polynomial membership is a new algorithm.

### 2.3 Independent face formula

For a nonempty row-index subset \(I\), let \(v(I)\) count coordinates on which its rows vary, and let \(Q_I\) be their smallest coordinate face. Hall's condition gives

\[
x\notin D(A)
\Longleftrightarrow
\exists Q\ni x:
\#\{i:a_i\in Q\}>\dim Q.
\tag{2}
\]

Also,

\[
\operatorname{rad}(D(A))
=\max_{\varnothing\ne I}(|I|-v(I)).
\tag{3}
\]

These are Corollary 2, Eqs. (2)–(3), pp. 2–3 of the note [1]. The quantity \(v(I)\) agrees with established \(r\)-wise Hamming distance terminology [5, Definitions 31 and 33].

Use the following independently derived form as a reference implementation:

\[
\delta_A(x)=
\max\left\{0,\ 
\max_{\varnothing\ne I}
\left[|I|-v(I)-\operatorname{dist}_H(x,Q_I)\right]\right\}.
\tag{4}
\]

Every varying coordinate belongs to \(N_x(I)\). Each constant coordinate belongs to it exactly when \(x\) disagrees with that common value. This proves Eq. (4) directly.

A high deficit measures concentration in or near a face containing many catalogue rows. It is not inherently a diversity score or a predictor of software failure.

### 2.4 Saturation theorem used by this project

For **distinct** rows and \(n\ge2\), define

\[
d=\min_{i\ne i'}d_H(a_i,a_{i'}),\qquad
b=\min_j\min\left(\#\{i:a_{ij}=0\},\#\{i:a_{ij}=1\}\right).
\]

The note proves

\[
2d+b>n
\quad\Longrightarrow\quad
D(A)=\{0,1\}^n\setminus C.
\tag{5}
\]

This is Theorem 3, Eq. (4), pp. 3–4 [1]. The implication is sufficient, not necessary.

The essential proof was reviewed during planning. If a nonrow target has a deficient row subset of size \(r\), write \(u=|N_x(I)|\le r-1\). Pairwise-distance counting gives

\[
d\binom r2
\le \sum_{\{i,i'\}\subset I}d_H(a_i,a_{i'})
\le \frac{u r^2}{4}
\le \frac{(r-1)r^2}{4},
\]

hence \(2d\le r\). A coordinate outside \(N_x(I)\) is constant on those rows, so \(r\le n-b\). Thus a nonrow obstruction would imply \(2d+b\le n\).

No essential proof defect was found in the review. This is ordinary mathematical inspection and finite checking, not proof-assistant formalization.

### 2.5 Other supplied results: what they do and do not support

| Source result | Audited content | Role here |
|---|---|---|
| Theorem 3 equality case, pp. 3–4 | At \(2d+b=n\), extra obstructions are faces with \(\lvert I\rvert=2d\), \(v(I)=2d-1\), carrying a local punctured Hadamard structure | Context and assumption audit; not needed for the primary conclusion |
| Proposition 4, p. 4 | Balanced sharp examples with additional obstructing faces | Shows saturation is not automatic |
| Proposition 5, p. 4 | \(\operatorname{rad}(D(A))\le\max\{1,n-b-2d\}\), under distinct-row assumptions | Optional assertion in tiny checks; no bug-detection guarantee |
| Theorem 6, pp. 5–6 | Cluster-plus-antipode examples have far smaller diagonal separation than ordinary covering radius | Refutes equating diagonal escape with maximum Hamming diversity |
| Theorem 7, p. 6 | Exact maximal output-set covering radii under distinctness and optional nonconstant-column assumptions | Mathematical context; no required implementation |
| §6, p. 7 | Source reports exhaustive distinct-row checks through \(n=5\) and specific larger family checks | Source-reported evidence only beyond the checks described in §9 below |
| §§7–8, pp. 7–8 | Historical novelty is explicitly uncertain; a 2007 paper was unavailable | Do not promote the note's novelty status to an established priority claim |

For specificity, Theorem 6 takes \(n=2^k+1\), \(q=n-k\), \(k\ge3\), and
\(C_k=(\{0,1\}^k\times\{0^q\})\cup\{1^n\}\). It gives
\(|D|=2^k(q+1)-1\), ordinary row-set radius \(2^{k-1}\), and best diagonal separation \(k+1\). These concern geometry, not application defects.

All retained computations are exact Boolean or integer operations. There are no numerical tolerances, conditioning issues, convergence criteria, hidden optimization oracles, or GPU requirements. The costly general operation avoided is enumeration of all \(n!\) permutations. Deliberate enumeration at \(n=8\) is a bounded correctness reference, not the proposed scalable implementation.

---

## 3. Fixed cases and reproducible data

### 3.1 Bit and serialization conventions

A configuration is an integer in \([0,2^n)\), serialized as exactly \(n\) ASCII binary digits, most significant bit first. Coordinate \(j=0\) is the leftmost bit:

\[
x_j=(x\mathbin{\texttt{>>}}(n-1-j))\mathbin{\texttt{\&}}1.
\]

Row indices are zero-based. Never use a mixture of left-to-right coordinate indices and low-bit indices in certificates.

A catalogue's digest is SHA-256 over its binary rows in generation order, with one ASCII LF after every row, including the last. Do not sort rows before this digest. Row permutation preserves \(D(A)\), but the digest records provenance.

Reject malformed lengths, nonbinary symbols, empty inputs, and dimensions incompatible with the requested theorem. For a general matching test, repeated binary rows are legitimate. For the distinct-row saturation certificate, report repeated rows as “assumption not satisfied”; do not silently deduplicate.

### 3.2 Ordinary coverage seed: stop at completion, not at a row count

For configuration \(x\), let

\[
T_t(x)=\{(J,x|_J):J\subseteq\{0,\ldots,n-1\},\ |J|=t\}.
\]

For a supported domain \(V\), feasible tuples are \(\mathcal T_t(V)=\bigcup_{x\in V}T_t(x)\). In the full cube, \(|\mathcal T_2|=4\binom n2\).

Use this exact catalogue generator for **n = 8**:

~~~text
GREEDY_PAIR_SEED(n, seed):
    V = all integers 0 through 2^n - 1
    rows = [0, 2^n - 1]
    covered = T2(0) union T2(2^n - 1)
    all_pairs = union(T2(x) for x in V)
    priority(x) = SHA256_ASCII("catalogue|" + decimal(seed)
                              + "|" + n_bit_MSB_string(x))

    while covered != all_pairs:
        candidates = V minus set(rows)
        gain(x) = cardinality(T2(x) minus covered)
        choose x by:
            largest gain,
            then smallest priority as hexadecimal string,
            then smallest integer x
        require gain(x) > 0; otherwise report an implementation error
        append x
        record x, gain(x), cumulative coverage, tie key
        covered = covered union T2(x)

    return rows, full trace
~~~

The generator never consults reachability, software behavior, defects, row distance, or column balance. It never stops because it has reached \(n\) rows. After generation, check its dimensions. A rectangular result is not an input to the square theorem and must not be padded or truncated.

The primary case is seed 0. A fixed descriptive cohort uses seeds 0–19, inclusive. These are deterministic tie-priority variants, not independent software projects or a population sample.

### 3.3 Primary catalogue

Expected seed-0 output:

| Row index | Binary word | Decimal |
|---:|---|---:|
| 0 | 00000000 | 0 |
| 1 | 11111111 | 255 |
| 2 | 10010011 | 147 |
| 3 | 01101100 | 108 |
| 4 | 01001011 | 75 |
| 5 | 10110100 | 180 |
| 6 | 00011101 | 29 |
| 7 | 11100010 | 226 |

Expected SHA-256:

~~~text
e81c1cba8db1ba0f81ff36caca4dc0567d66cc1ab789465cca7d9bebfcb16ffc
~~~

The initial two rows cover 56 of 112 pair assignments. The six additions must contribute respectively **16, 16, 8, 8, 4, 4**, reaching 112. Every addition has positive gain. The procedure naturally stops at eight rows.

Expected independently verifiable facts:

\[
d=4,\quad b=4,\quad 2d+b=12>8,\quad |D(A)|=248.
\]

All eight rows have deficit 1. Every one of the 248 nonrow words has deficit 0. In particular,

\[
U(A,V,M)=V\setminus(M\cup D(A))=\varnothing.
\]

The source theorem establishes this after \(d,b\) are verified. Direct permutation enumeration and perfect-matching certificates must also verify it without relying on the theorem.

For all 20 declared eight-flag catalogue seeds, planning checks found exactly eight rows, full pair coverage, \(d=b=4\), and saturation. Reproduce and report every member; do not retain only the cases with the expected result.

### 3.4 Essential counterexample: pairwise coverage alone is insufficient

Use this separate mathematical control:

~~~text
00000000
00000001
00000010
11111111
00011100
01100100
10101000
11010000
~~~

Decimal rows are \([0,1,2,255,28,100,168,208]\). It covers all 112 pair assignments, but

\[
|D(A)|=247,\qquad
\{0,1\}^8\setminus(C\cup D(A))=\{00000011\}.
\]

For target \(x=00000011\), indexed rows \(I=\{0,1,2\}\) have disagreement neighbors only at **coordinates \(\{6,7\}\)** under this brief's MSB convention. Therefore \(3>2\) is an explicit Hall witness.

Pairwise coverage can be checked directly. The first three rows and the all-one row cover all patterns on the two low-order coordinates. Across the six other coordinates, the last four rows encode all six two-element subsets of four row positions; this supplies the asymmetric patterns, while the extreme rows supply 00 and 11. Mixed coordinate pairs likewise have all four patterns.

This control must pass. Otherwise a broken implementation could report saturation for every input. Its existence also forbids the conclusion “pairwise coverage implies diagonal saturation.”

### 3.5 Control for a failed sufficient condition

Use the following **literal mathematical input**, not a claim about natural covering-array completion:

~~~text
0000000000
1111111111
1001110010
0110001101
0111100011
1000111100
1001000101
0010011010
0101111000
0000001100
~~~

Here \(d=2\), \(b=4\), \(2d+b=8\le10\), yet \(|D|=1014\) and there are no nonrow omissions. Check this using matching and the independent subset/face formula, not \(10!\) permutation enumeration.

During planning this matrix came from a preliminary fixed-ten-row construction. Its last row added **zero** new pairs; the natural pairwise procedure would already have stopped at nine rows. That is why it is retained only as a mathematical control and is not the primary practical seed.

### 3.6 Positive control for the selector wiring

In the counterexample of §3.4, define the already selected set for this unit control as every cube word except \(3\) and \(244\). This is a constructed instrument check, not an application test suite.

All three-way tuples remain covered: each has 32 full-cube completions and only two words are withheld. Thus both candidate gains are zero. The exact reachability labels are \(u_A(3)=1\), \(u_A(244)=0\).

Under the selection priority defined in §4.3 with seed 0:

~~~text
priority(244) = 01b022711379030fb27dbda024bf5db30438bb130b713fecd7ad824e3c1983b6
priority(3)   = 90f553e67f1d1d34e0d9aa88ef4360b9b3e0a1468ceead50d0217f5807079916
~~~

The coverage-only selector must choose 244 first; the reachability tie-break must choose 3 first. This verifies that the proposed branch is capable of affecting a genuine equal-gain decision when the information differs. It establishes no defect-detection advantage.

---

## 4. Implementation specification

Use Python 3.11 or newer and its standard library. No solver, NumPy, GPU, database, service, dashboard, or external application package is required. Record the actual interpreter version. Prefer straightforward implementations whose outputs can be checked independently.

### 4.1 Exact matcher and certificates

A deterministic augmenting-path bipartite matcher suffices. Traverse rows and coordinates in ascending index order. Reset visited coordinates for each attempted augmentation.

~~~text
MATCH(A, x):
    verify A is nonempty, square, and binary
    right_owner = [-1] * n

    AUGMENT(i, seen):
        for j = 0 .. n-1:
            if A[i,j] == bit(x,j) or seen[j]:
                continue
            seen[j] = true
            if right_owner[j] == -1 or AUGMENT(right_owner[j], seen):
                right_owner[j] = i
                return true
        return false

    size = 0
    for i = 0 .. n-1:
        if AUGMENT(i, [false] * n):
            size += 1

    return size, right_owner, n - size
~~~

The conservative running-time bound is \(O(n^3)\) per target; graph storage is \(O(n^2)\), or implicit from the matrix. There is no need to optimize further at the retained sizes.

For every reachable primary target, write a certificate assigning one distinct row to each coordinate. A separate checker must verify:

1. the catalogue digest and target representation;
2. exactly \(n\) row indices, all in range, each used once;
3. \(a_{\pi(j),j}\ne x_j\) for each coordinate.

The checker must not import or call the matcher. It must also check **global completeness**: the primary catalogue has the expected digest from §3.3, every target has the canonical eight-bit format, target keys are unique, and the exact set of certified targets equals \(\{0,1\}^8\setminus C\). Reject duplicate, missing, row, and extraneous targets. A count of 248 records alone is insufficient. Only this complete set of independently valid certificates establishes that every primary nonrow target is reachable. Row exclusion follows directly from the diagonal definition.

For unreachable targets, obtain a Hall witness either from alternating-path traversal or, at these sizes, exhaustive row subsets. Report a nonempty subset of **distinct row indices** and the actual neighbor set. The independent checker must reconstruct the entire neighborhood from the matrix and target, require equality with the reported coordinate set, and verify \(|I|>|N_x(I)|\). A claimed subset of the real neighborhood is not a valid substitute. Distinct indices may still have equal row values when repeated rows are permitted.

Do not declare unreachability from a search timeout, a partial matching, an exception, or a failed sufficient condition.

### 4.2 References that do not share the matching implementation

Implement both of these references:

- **Definition reference:** enumerate every row permutation and construct its complemented diagonal directly. Use this for all distinct-row inputs at \(n\le4\), all ordered \(3\times3\) binary matrices including duplicates, and the two specified \(8\times8\) matrices. There are only 40,320 permutations per eight-row matrix. Do not enumerate permutations at \(n=10\) or \(n=12\).
- **Subset reference:** independently enumerate nonempty row-index subsets and evaluate Eq. (4), using common bits and varying bits rather than augmenting paths. Use it for the primary and both controls, including all 1,024 targets of the ten-row control.

For each tiny case, compare membership, exact deficit, and direct distance to the enumerated \(D(A)\). Where assumptions hold, check the strict saturation implication and Eq. (3). Execute constructive repair only for the required two-bit guard below and verify that its output is in the unconstrained diagonal set at the claimed distance. Optional checks of the note's equality or radius bounds must not delay delivery.

Required exhaustive tiny counts are 2, 6, 56, and 1,820 distinct row sets for \(n=1,2,3,4\), respectively: 29,596 target checks in total. Separately test all 512 ordered \(3\times3\) matrices and their 4,096 target checks. These scopes overlap; do not claim their combined count represents disjoint matrices.

**Constraint guard:** take the distinct-row matrix with rows 00 and 10 and supported domain \(V=\{00,10\}\). Its diagonal set is \(\{01,11\}\), so \(D(A)\cap V=\varnothing\). A nearest diagonal repair from 00 is invalid. This is an exact two-bit control against silently inheriting constrained repair optimality.

### 4.3 Coverage baseline and exact matched augmentation

Build feasible tuple IDs independently, with coordinate subsets in lexicographic order and bit values in lexicographic order. A bit mask for \(T_t(x)\) is convenient. Verify masks against a separate set-based tuple enumerator on every primary configuration.

Use \(t=3\), giving \(8\binom83=448\) feasible triples. Let

\[
g_3(x;S)=|T_3(x)\setminus\bigcup_{z\in S}T_3(z)|,\qquad u_A(x)=\mathbf 1[\delta_A(x)>0].
\]

The proposed mechanism uses the **binary** label \(u_A\), not a tuned deficit weight or a new objective. Deficits are still computed for correctness and interpretation.

Define stateless tie priority

~~~text
SHA256_ASCII("selection|" + decimal(seed) + "|" + n_bit_MSB_string(x))
~~~

Do not consume a shared random-number stream during matching or validation. Different RNG consumption would create an implementation difference unrelated to the proposed information.

~~~text
ORDER(V, M, A, seed, use_reachability):
    require set(rows(A)) subset of M subset of V
    S = set(M)
    covered = union(T3(x) for x in S)
    order = []

    while V minus S is nonempty:
        for x in V minus S:
            gain = bit_count(T3_mask[x] AND NOT covered)
            extra = u_A(x) if use_reachability else 0
        choose x by:
            largest gain,
            then largest extra,
            then smallest selection priority,
            then smallest integer x
        append x to order
        S = S union {x}
        covered = covered union T3(x)
        record selected x, gain, extra, coverage, priority

    return order
~~~

For the diagnostic, continue after full three-way coverage so the complete remaining order is comparable. Also record the first prefix attaining full three-way coverage. If a practical policy stopped there, both versions would stop at the same point in the primary case.

Run paired complete orders for primary selector seeds 0, 1, and 2. The proof below covers every priority seed; these three are implementation checks, not statistical replicates. Run the positive wiring control once with seed 0.

This baseline is an exact finite-candidate instance of the established greedy “cover most previously uncovered interactions” approach [8, §2.2 and Algorithm 1]. It is not presented as a new covering-array algorithm.

### 4.4 Equivalence proof the report must contain

Let \(A\) be the primary catalogue. Verification gives \(D(A)=\{0,1\}^8\setminus C\). Assume \(C\subseteq M\subseteq V\). Every remaining candidate \(x\in V\setminus M\) is a nonrow and therefore has \(u_A(x)=0\), and also \(\delta_A(x)=0\).

Initially, both selectors have the same candidate set and covered tuples. Their coverage gains and stateless tie priorities are identical. The treatment's extra score is constant, so they choose the same first configuration. Updating both sets with that same configuration preserves the premises. Induction gives the same entire additional order.

The proof applies to any further validity restriction and any additional mandatory configurations satisfying the containment conditions. It does not assert that an arbitrary constrained application could actually use the primary catalogue: its rows must first be supported and already tested.

For an arbitrary deterministic root-defect map \(F(x)\), both selected prefixes yield the same union of defect IDs:

\[
\bigcup_{x\in S_{\rm treatment}(k)}F(x)
=
\bigcup_{x\in S_{\rm baseline}(k)}F(x).
\]

Thus no failure distribution, including a real one, can make this constant tie-break improve discovery through different configuration choices.

**Cost qualification:** under identical downstream execution and additive nonnegative extra analysis cost, the treatment cannot complete more of the common sequence within a fixed budget. This is a stated cost-model implication. It is not a measured claim about hardware caching, wall-clock noise, or a separately optimized implementation. An implementation that recognizes the constant score and skips it can be equivalent in cost. Do not report “slower on M3 Max” without measuring that quantity.

The argument does not cover algorithms that use saturation to switch to a different policy, choose a different catalogue, change workloads, or sample permutations with a different probability distribution. Those are different mechanisms.

---

## 5. Evidence gates and bounded execution

Follow these stages in order. Their purpose is to verify a known structural candidate result, not to discover a favorable software benchmark.

| Gate | Question and minimum work | Required evidence | Decision and next action | Automated-work cap |
|---|---|---|---|---:|
| G0: Constructive feasibility | Can exact inputs and the ordinary seed rule be represented? Inspect environment; validate literal inputs; regenerate primary seed and its positive-gain trace | Environment record, exact rows/digest, 112-pair coverage, generation trace | Continue if exact. A shape discrepancy is a construction failure to diagnose, never permission to reshape. Missing target hardware blocks only target measurements | 5 min |
| G1: Tiny correctness | Do independent definitions, matching, distance, and face calculations agree? Run the specified tiny suites and malformed-input/constraint guards | Raw check counts; any failing matrix/target; passing independent certificate checks | Continue with zero discrepancies. Otherwise use bounded repair; after exhaustion classify the attempted implementation as invalid | 10 min |
| G2: Resource calibration | Does the actual machine comfortably execute a representative exact check? Time one primary matching scan, direct permutation scan, and paired order calculation in separate measured commands | Wall time, CPU time if available, peak RSS, result sizes | Continue if predicted remaining work fits. Do not shrink the primary case. Skip only the descriptive cohort if it cannot fit, with explicit incompleteness | 5 min |
| G3: Structural mechanism test | Do the primary theorem certificate and exhaustive outputs agree? Does the tie-break do nothing on the primary but act in the wiring control? | \(d=b=4\), 248 reachable nonrows, 248 checked certificates, paired order equality, counterexample witness, positive control difference | If confirmed, **close the real-defect benchmark branch**. If contradicted, preserve the minimal counterexample and diagnose; do not launch a replacement study | 10 min |
| G4: Fixed robustness checks | Does the natural-stop result reproduce for all 20 declared eight-flag catalogue seeds? Does the ten-row control invalidate misuse of the sufficient criterion? | All cases, including failures/rectangular results; exact coverage and saturation summaries; ten-row control results | Report the complete finite cohort. These checks delimit the finding; they do not establish population prevalence | 5 min |
| G5: Diagnosis and report | Is the conclusion precisely scoped and independently reproducible? Run the standalone checker from clean results, regenerate tables, inspect report | Code, instructions, input records, certificates, raw logs, report and limitation statement | Deliver the completed diagnostic or documented early stop | 10 min |
| Shared repair reserve | At most two diagnosed implementation-repair attempts; no parameter searching or changed scientific design | Original failing artifacts, change description, rerun evidence | Stop at reserve exhaustion; do not reset a gate clock to obtain more budget | 15 min |

**Total automated-work ceiling: 60 minutes**, including environment commands, all exploratory runs, failed attempts, repetitions, repairs, and report generation. Caps are ceilings, not targets. An informative run should take minutes; that is an unmeasured M3 planning expectation. Keep the last 10 minutes for evidence preservation and reporting. Stop launching optional work once less than that reserve remains.

No independent “locked empirical comparison” stage remains after G3. The replacement is a locked, deterministic reproduction of the specified claim and its controls. No application data need to be acquired, and no postponed software benchmark should be built speculatively.

### 5.1 Calibration and resizing rules

Use the measured primary work, not the hosted planning timings, to forecast the remaining finite checks. A conservative cohort estimate is 20 times the measured eight-row classification-plus-generation cost, plus the independently timed controls. If that exceeds G4's remaining allowance, report a partial descriptive cohort and retain the exact primary conclusion if verified.

Do not reduce eight flags, drop catalogue rows, replace difficult controls, weaken equality checks, or omit failed cases to meet a budget. If the indispensable primary checks do not fit after ordinary implementation repair, classify practical infeasibility of this implementation or inconclusive verification, as appropriate.

Each command must have an outer deadline. Launch every timed command, including direct test, certificate-checker, and report invocations, through the same small subprocess wrapper applying the remaining stage/global deadline. Alternatively, measure such an invocation under an outer deadline and append its elapsed time and status to the shared ledger before starting another stage. A subprocess timeout and a JSONL ledger entry suffice; no service is needed. Reproduction commands do not reset or evade the cumulative budget. On timeout, terminate the process, save the last completed case and its inputs, and mark remaining work unperformed. A timeout does not mean a configuration is unreachable and does not refute the theorem.

### 5.2 Target hardware and measurement boundaries

All PoC computation is intended for one MacBook with an M3 Max. First record actual chip, architecture, macOS version, Python version, physical RAM, available disk, and whether execution is native arm64. For example, inspect the appropriate outputs of:

~~~bash
uname -m
sw_vers
system_profiler SPHardwareDataType
sysctl -n hw.memsize
df -k .
python3 --version
~~~

Do not infer installed RAM, free storage, or interpreter architecture from the chip name. Do not publish hardware serial numbers or other unrelated device identifiers in the report.

Use at most one computation worker. Set a **512 MiB peak resident-memory ceiling** for that worker and a **256 MiB project-output ceiling**. Require enough free storage for that allowance plus a 256 MiB reserve; inspect actual memory availability before work. These are conservative planning caps for a tiny standard-library diagnostic, not measured requirements. If the machine cannot spare them, record the constraint rather than assume it away.

Measure command wall time from before starting the worker through completion and result serialization. Record setup, seed generation, matching, reference checking, selection, and report work separately. Use a monotonic timer. Record cold process-start measurements; do not add a timing competition or repeated warm-run campaign because the scientific claim is exact selection identity.

On macOS, /usr/bin/time -l can supply process peak RSS; alternatively use resource.getrusage in a worker and normalize units explicitly. macOS and Linux report ru_maxrss in different units. Do not add child-process peaks as if their maxima necessarily occurred simultaneously. The single-worker design avoids most process-tree ambiguity.

There is no GPU backend, mixed precision, device synchronization, or approximate solver. Record backend as native CPU / exact Python integers. A hosted Linux run may verify portable mathematics but must never be labeled M3 Max performance.

**Expected effort:** roughly 2–4 hours for implementation, independent review, and reporting, plus minutes of execution; allow up to 6 hours of active agent work including ordinary repairs. These are estimates, not promises. The 60-minute automated-work cap is cumulative and separate from the active coding/analysis estimate.

---

## 6. Minimum repository, interfaces, and intended commands

Create a small directory such as:

~~~text
reachability-diagnostic/
  README.md
  study.json
  run.py
  matcher.py
  tuple_coverage.py
  reference_checks.py
  verify_certificates.py
  tests/
    test_core.py
  results/
    ledger.jsonl
    run-0001/
  report/
    REPORT.md
~~~

Responsibilities:

- **matcher.py:** input validation, deterministic matching, deficit, certificate production, and the small repair routine needed by the two-bit guard.
- **tuple_coverage.py:** tuple enumeration, verified masks, natural-stop catalogue generation, and matched selectors.
- **reference_checks.py:** permutation definition and face/subset oracle. Do not implement these by delegating to the matcher.
- **verify_certificates.py:** standalone input-and-certificate validation, including exact primary target-set completeness, using only basic indexing and set checks. It must not import matcher.py or trust its membership labels.
- **run.py:** small CLI, environment inspection, cumulative budget ledger, stage orchestration, results, and report tables.
- **tests/test_core.py:** meaningful mathematical equivalence, malformed-input behavior, bit-orientation checks, constraint guard, certificate corruption and duplicate/missing-target rejection, and the selector wiring control.

No package installation is necessary beyond an available Python 3.11+ interpreter. A virtual environment is optional. Avoid a packaging step that downloads build tools for a standard-library-only project.

The configuration must fix the following values; document any engineering-only additions:

~~~json
{
  "schema_version": 1,
  "bit_order": "msb-first",
  "primary_n": 8,
  "primary_catalogue_seed": 0,
  "catalogue_seeds": [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19],
  "catalogue_stop": "complete-feasible-pair-coverage",
  "square_policy": "check-after-generation-never-reshape",
  "selector_strength": 3,
  "selector_score": "binary-unreachable",
  "selector_seeds": [0,1,2],
  "automated_wall_cap_seconds": 3600,
  "report_reserve_seconds": 600,
  "peak_rss_cap_bytes": 536870912,
  "output_cap_bytes": 268435456,
  "maximum_repair_attempts": 2
}
~~~

Suggested callable interfaces:

~~~python
validate_matrix(rows: list[str], require_square: bool = True) -> None
match_target(rows: list[str], target: str) -> dict
face_deficit(rows: list[str], target: str) -> int
permutation_outputs(rows: list[str]) -> set[str]
greedy_pair_seed(n: int, seed: int) -> tuple[list[str], list[dict]]
configuration_order(rows, valid_words, mandatory_words,
                    seed: int, use_reachability: bool) -> list[dict]
verify_matching_certificate(rows, target, row_for_coordinate) -> bool
verify_hall_certificate(rows, target, row_indices, coordinate_indices) -> bool
~~~

Return enough evidence to diagnose a failure; do not return only a Boolean “passed.”

**The commands below are interfaces you must create. They were not executed during preparation of this brief.** Make them work from the repository root:

~~~bash
python3 run.py environment --out results/environment.json
python3 -m unittest discover -s tests -v
python3 run.py verify-small --out results/small-checks.json
python3 run.py smoke --config study.json --out results/smoke-0001
python3 run.py study --config study.json --out results/run-0001
python3 verify_certificates.py --catalogues results/run-0001/catalogues.json --certificates results/run-0001/certificates.jsonl
python3 run.py report --input results/run-0001 --out report/REPORT.md
~~~

“Study” must run required stages not already covered by valid recorded artifacts and must charge earlier smoke/check work to the same ledger. Reuse completed correctness evidence by input/code digest; do not silently duplicate a large test run. Refuse to overwrite an existing result directory. A rerun gets a new directory and consumes the remaining shared budget.

If Python or target hardware is unavailable, finish the unaffected derivation, input specification, and code. State exactly which commands and target measurements remain blocked. Do not claim execution merely because commands are written.

---

## 7. Measurements, analysis, and honest outcomes

### 7.1 What to measure

This project has exact finite endpoints:

| Endpoint | Primary expected value | Meaning |
|---|---:|---|
| Natural catalogue row count | 8 | The theorem applies without reshaping |
| Pair tuples covered | 112/112 | Correct conventional seed construction |
| Minimum row distance / minimum column minority | 4 / 4 | Strict saturation certificate |
| Reachable nonrow words | 248/248 | No omitted nonrow region |
| Extra unreachable nonrow words | 0 | No reachability signal after mandatory rows |
| Primary paired-order mismatches | 0 for each selector seed | Specified augmentation is behaviorally redundant |
| Counterexample extra omissions | Exactly word 3 | Checks the diagnostic can identify a real obstruction |
| Positive wiring-control first choices | Baseline 244; augmentation 3 | Checks the intervention is implemented |
| Ten-row control nonrow omissions | 0 despite \(2d+b\le n\) | Guards against inverting a sufficient implication |
| Diagnostic time, peak RSS, output bytes | Measured, no required performance win | Local feasibility and reproducibility |

Do not report Hamming distance, pair coverage, number of distinct configurations, or an obstruction count as “bugs found.”

No statistical significance test, confidence interval, bootstrap, or power calculation is appropriate for these exhaustive finite identities. The 20 catalogue variants are a bounded descriptive cohort chosen during exploration; they do not estimate production prevalence. Deterministic checks need not be repeated to manufacture sample size.

No application defect count, severity estimate, or time-to-defect curve is retained. The baseline and augmentation are compared at their actual interface: ordered configuration selection. The proof is stronger than trying a finite collection of fabricated failure maps, so do not generate a synthetic bug benchmark.

### 7.2 Failure and exclusion accounting

Every run has a status such as complete, failed-check, invalid-input, timed-out, resource-stop, skipped-not-applicable, or blocked-target-hardware. Record exceptions, attempted work, and omitted cases. Summaries must show expected and completed case counts.

A rectangular naturally generated catalogue is a valid observation that the square theorem is inapplicable, not a malformed result to repair by padding. An implementation error in the generation rule is different and should be repaired within bounds.

If the matcher and direct reference disagree, suspend interpretation until resolved. Preserve the smallest matrix, target, graph edges, and conflicting outputs. If you cannot resolve the discrepancy, report an invalid experiment. Do not count only the successful cases.

If certificate bytes are corrupted, a row is reused, an index is out of range, a disagreement edge is absent, or a catalogue digest differs, the independent checker must reject the certificate. It must also reject a duplicate target replacing a missing primary target, even when the total record count is still 248, and reject a Hall witness whose reported neighborhood omits a real neighbor. Verify at least one intentional example of each material corruption category.

### 7.3 Outcome categories

| Outcome | Evidence required | Required interpretation |
|---|---|---|
| Supported contribution | Verified primary construction, independent saturation evidence, selector-equivalence proof, correct controls | A reproducible scoped limitation is supported; no claim of a useful defect-detection improvement |
| Informative negative result | The same evidence rules out a selection-information advantage for the exact integration | Do not build the proposed application benchmark for this catalogue |
| Practical infeasibility | Measured indispensable checks exceed actual resources after bounded repair | State the implementation/machine limitation; do not infer that mathematics or practical testing is false |
| Inconclusive evidence | Incomplete essential mathematical checks or unresolved provenance prevent verification | If mathematics is independently verified elsewhere but the M3 Max is unavailable, report “analytic conclusion supported; target execution and measurements blocked,” rather than downgrading the proof |
| Invalid experiment | An unresolved implementation, reference, input, or measurement defect affects the conclusion | Preserve evidence and identify what prevents interpretation |

“Supported contribution” and “informative negative” can both describe the final result: the supported contribution is the limitation. Avoid a positive-results vocabulary suggesting improved software assurance.

A successful primary verification permits this conclusion:

> For the exact naturally generated eight-flag catalogue and the specified coverage selector, all remaining configurations have identical diagonal-reachability labels. The proposed reachability tie-break cannot alter configuration selection. Under identical application execution, it therefore cannot improve defect discovery through a different test sequence. This closes this integration branch; it does not resolve usefulness for other catalogues or mechanisms.

Do not claim that all pairwise suites saturate, all releases are sufficiently tested, conventional coverage finds every bug, or reachability information is always useless.

### 7.4 One justified next step, outside this project

Only if further work is requested: obtain an independently motivated **actual supported configuration catalogue** and inspect its shape and exact nonrow reachability before building an application harness. A nonsaturated catalogue would establish only that information exists, not that it predicts failures.

Any later empirical protocol would need a pinned component and workflow, independently specified expected behavior, unchanged mandatory tests, unique root-defect attribution, equal total cost, an established constrained interaction-testing implementation, a matched score-disabled control, and separation of selector inputs from failure truth. Existing known failures receive no discovery credit. That is a new research decision, not an automatic fallback authorized by this brief.

---

## 8. Required final artifacts

Deliver only what the reached stages support:

1. **Runnable code and README:** exact Python version, commands, input conventions, scientific scope, and how to reproduce the decision from a clean directory.
2. **Canonical inputs and provenance:** literal matrices, generated catalogue traces, seed list, row digests, code/configuration digests, and environment metadata.
3. **Raw evidence:** all checks, failures, timings, resource usage, paired orders, coverage traces, and budget ledger. Keep records from failed attempts.
4. **Independent certificates:** at minimum every primary nonrow perfect matching and the counterexample's Hall witness, with a standalone checker.
5. **Essential tables:** primary construction/coverage, exact reachable-set counts, control outcomes, finite-cohort summary, and performed-versus-unperformed work.
6. **REPORT.md:** project decision; source mathematics versus new derivations; exact evidence; the equivalence proof; limitations; resource results; outcome classification; deviations; and the one scoped next step.

No plot is required: exact tables and short proofs communicate this result more clearly. Do not create a dashboard or claim a performance frontier.

For an early stop, deliver the minimal verifiable derivation or counterexample and all available code/evidence. Missing target timing is an explicit missing measurement, not a reason to erase a valid mathematical certificate.

Before finishing, run the certificate checker in a separate process, regenerate the essential tables from raw files, inspect the report for inconsistent counts and bit conventions, and ensure it distinguishes executed work from planned or blocked work.

---

## 9. Evidence already obtained during preparation

This section records what the research lead and independent reviewers actually did. It is background evidence, not a substitute for your implementation checks. Their scratch programs are not required inputs; recreate the specified checks from this brief.

### 9.1 Source inspection and small mathematics

All nine rendered pages of the note were read, including proofs, assumptions, experiments, limitations, and references. An independent mathematical reviewer found no essential proof error.

One planning probe exhaustively checked all distinct row sets for \(n=1,\ldots,4\), totaling 29,596 target checks, and all ordered \(3\times3\) matrices, totaling 4,096 target checks. It compared permutations, matching, direct distances, Hall deficiency, and constructive repair. All assertions passed. The note's exhaustive \(n=5\) claim was not reproduced.

Additional source-family probes obtained:

| Mathematical construction | \(\lvert D\rvert\) | Ordinary row-set covering radius | Best diagonal separation |
|---|---:|---:|---:|
| Cluster family \(n=9,k=3\) | 55 | 4 | 4 |
| Generalized cluster \(n=12\), first 11 four-bit prefixes plus antipode | 143 | 6 | 5 |
| Balanced Hadamard boundary \(n=8,d=2,b=4\) | 240 | Not needed | Not needed |

The boundary example had eight extra nonrow omissions. These are mathematical controls, not software experiments.

### 9.2 Complete disclosure of exploratory catalogue probes

A label-free random probe used Python Random(20261101), sampling \(n\) distinct cube words without replacement, 20 catalogues each at \(n=8,10,12\), in that order. Exact saturation occurred in 17/20, 17/20, and 20/20, respectively. None satisfied the strict sufficient threshold. Nonzero omission counts were \(1,1,1\) at \(n=8\), and \(1,26,502\) at \(n=10\); the largest had a constant column. These are random formal matrices, not realistic release distributions.

A second exploratory probe initialized all-off/all-on and greedily added pair coverage until **exactly \(n\) rows**, for seeds 0–19 at each of \(n=8,10,12\). All 60 fixed-size matrices had full pair coverage and no nonrow omissions. This observation was not hidden when the design changed.

Review found that this fixed-size stopping condition could add zero-gain rows:

- the displayed ten-row seed-0 matrix completed pairwise coverage after nine rows;
- natural completion at ten flags occurred in 8–10 rows across the explored seeds;
- natural completion at twelve flags occurred in 9–10 rows.

The final primary policy therefore uses natural coverage completion. At eight flags, all 20 examined seeds naturally stopped at eight rows, each added row had positive gain, and all had \(d=b=4\). This corrected policy is the one you must implement.

The final primary scope was chosen after those observations. The finding is exploratory and exact for the supplied instances; it is not an unbiased frequency estimate or a global inference.

### 9.3 Independent cross-checks of the retained inputs

A different reviewer directly enumerated all 40,320 permutations for each eight-row primary/control matrix, without using the matching implementation. This independently confirmed \(|D|=248\) and \(|D|=247\), respectively, with the control's sole nonrow omission 3.

The research lead separately evaluated the row-subset face formula for every target of the primary, the counterexample, and the ten-row control. Results agreed:

| Case | Deficit 0 | Deficit 1 | Extra nonrow omissions |
|---|---:|---:|---|
| Primary eight-row matrix | 248 | 8 | None |
| Pairwise-complete counterexample | 247 | 9 | Exactly 3 |
| Ten-row literal control | 1,014 | 10 | None |

The primary digest and the selector wiring-control priorities were also calculated.

These probes took seconds in the hosted execution environment. **None was measured on the user's M3 Max.** No Terser installation, application bug benchmark, production data access, or end-to-end release test occurred during preparation.

---

## 10. Research context, closest alternatives, and unresolved opportunity

### 10.1 What the literature changes

The practical problem is real, but it does not validate this particular construction. The Office work studies feature-flag interdependencies using query logs, not complemented diagonals [10]. LaunchDarkly recommends meaningful production, upcoming, fallback, and historically problematic states [11]. OpenFeature specifies that evaluated values depend on context [12]; prerequisites can also constrain achievable values [13]. Consequently, geometric omissions cannot be treated as relevant application omissions without a supported-domain model.

The closest computational prior art has two parts:

- The tableau permanent and matching-membership mechanism are established [3,4]. The paper's face-distance terminology overlaps established coding-theoretic quantities [5].
- Greedy interaction coverage and configuration prioritization are established [7–9,14,18]. The proposed extra score must earn its value beyond those mechanisms; changing terminology does not establish novelty.

No inspected primary source established that production testing tools commonly use the frozen square complemented-diagonal generator. Its prevalence remains unverified. Accordingly, this brief evaluates an explicit proposed descriptor and does not diagnose a widespread defect in existing tools.

The exact no-op result is not surprising enough to justify a novelty claim by itself. Its value is avoiding an unjustified application build and identifying a necessary applicability gate.

### 10.2 Established baselines considered and why they are not run here

Microsoft PICT provides an openly available interaction-testing implementation with higher strengths, constraints, and seeded rows [9]. NIST ACTS is another established alternative; its official download page currently makes basic ACTS directly available but requests contact for the advanced constraint-supporting tool [7]. No email request was sent.

If the retained project still made a real-defect improvement claim, a strong feasible comparison would include PICT at strength 3, ordinary coverage without the reachability score, uniform sampling without replacement, and an ordinary Hamming-distance prioritization control. Invalid rows must be modeled correctly, not removed after generation in a way that loses valid tuple coverage.

Those implementations are unnecessary in the present branch because the exact score-disabled ablation already proves the proposed information has no effect. Running a different covering-array tool would compare unrelated generator choices; it could not establish incremental benefit from a constant tie-break. This is an analytic stop, not substitution of a weak random benchmark for a strong one.

Wu et al. explicitly study conventional coverage, random, and adaptive random generation and their costs [8]. Halin et al. demonstrate the importance of distinguishing failing configurations from underlying faults and of including testing cost [14]. Neither paper implies that pairwise or three-way coverage guarantees all real faults are found.

### 10.3 Application subjects investigated but not admitted

Terser was investigated as a feasible local **static-configuration transfer** subject: a fixed JavaScript minify-then-execute workflow with Boolean compressor settings and historical wrong-code fixes [15–17]. That would already narrow the original deployed-feature-flag problem.

Two concrete primary-source examples were inspected:

- Issue #997, reported in 2021, concerns lexical declarations moved out of a loop initializer. Its fix adds a regression expecting output 3 [16].
- Issue #1306, reported on 13 November 2022, concerns assumptions about a variable before initialization; the fix adds a corresponding regression [17].

Both examples were publicly reported on the dates above. Discovery credit in a retrospective release simulation would require an independently verified historical information cutoff; the initially considered Terser 5.16.0 cutoff was not fully established from accessible release records. They are appropriate harness-control candidates, and neither is admitted as a discovery here. Any defect already known at an eventual release cutoff must remain mandatory and receive no discovery credit. A case caught by mandatory states would likewise get no additional-discovery credit.

The source investigation did not establish a full, immutable, locally reproduced historical corpus, a natural square upstream catalogue, or residual independently attributed defects after mandatory testing. Some tag/raw retrieval routes were unavailable. None of those gaps is silently treated as solved.

After the structural result, further Terser acquisition became unnecessary. **Do not install or benchmark Terser for this diagnostic.** Its investigation explains the practical alternative considered; it is not a second project hidden in the handoff.

The published JHipster study is highly relevant but its full rebuild workload used substantial distributed computation [14]. Replaying its published outcomes would be a different historical-table study, not a local reproduction of one pinned workflow. It was not selected.

### 10.4 Limits of an analytic application claim

For any two valid untested configurations with different geometric labels, one can define a deterministic failure map that fails only at the first, and another that fails only at the second. These maps favor opposite orderings. This elementary derivation rules out a distribution-free promise of better bug detection from catalogue geometry alone.

It does not rule out an empirical relationship in a justified software population. Likewise, the present saturation result rules out an effect for the selected catalogue, not for the counterexample or every real catalogue.

A permutation can also reach a word with very small positive probability. Membership and maximum matching only distinguish zero from nonzero reachability. Counting witnesses is a different problem; do not infer uniform sampling of \(D(A)\), favorable probabilities, or practical sampling coverage from these certificates.

---

## 11. Claim-to-evidence map and review disposition

### 11.1 Central claims

| Claim | Evidence status and basis | Remaining uncertainty | Resolving action |
|---|---|---|---|
| Membership and exact repair distance are constructively computable | Source proof [1, Prop. 1]; classical membership machinery [3]; independent tiny probes | Implementation correctness in the delivered code | G1 plus standalone certificates |
| The strict saturation condition is valid under distinctness | Audited source proof [1, Thm. 3] and tiny checks | Not formally mechanized; historical priority uncertain | Verify assumptions and independently enumerate the primary |
| The primary matrix is natural output of the declared coverage rule | Planning trace, positive gains, exact digest | Independent reproduction of construction | G0 |
| Its remaining candidates all have score zero | Theorem certificate, matching, permutation enumeration, independent face calculation | Portability and artifact correctness | G3 |
| The augmentation cannot change test order | Inductive derivation in §4.4 | Accidental implementation differences such as RNG consumption | Paired full orders and wiring control |
| Pairwise coverage alone does not imply saturation | Explicit counterexample and Hall witness | None for that finite claim after independent validation | G3 control |
| This resolves real feature-flag testing generally | **Unsupported; not claimed** | Actual catalogue prevalence, valid domains, defects, cost, other mechanisms | A separately justified future empirical project |
| The note or this specialization is globally novel | **Unverified; not claimed** | Inaccessible prior work and broader literature | Not a dependency of this diagnostic |

### 11.2 Material review findings incorporated into the design

Three distinct review perspectives were used during preparation: mathematical/novelty, experimental validity, and execution feasibility. The principal findings and dispositions are:

| Finding | Disposition in this brief |
|---|---|
| A correct diagonal theorem supplies no defect oracle | Use a necessary-condition diagnostic and make no empirical assurance claim |
| A sufficient condition can fail even when every nonrow is reachable | Require exact checks; include the ten-row saturated control |
| Pairwise coverage might be incorrectly generalized to saturation | Include a pairwise-complete nonsaturated matrix and explicit Hall witness |
| Forcing exactly \(n\) seed rows can alter a normal coverage workflow | Primary generation stops at coverage completion; square shape is checked afterward |
| The chosen regime was identified through exploration | Disclose all relevant preliminary probes and the post-observation scope revision |
| Nearest diagonal repair can violate validity constraints | Exclude repair from selection; include the exact two-bit constraint guard |
| A shared RNG could make a null intervention appear to change choices | Use stateless candidate priorities and identical candidate/tie inputs |
| Public historical cases could be miscounted without a verified cutoff | Admit no defect benchmark; distinguish verified issue dates from an unverified release chronology |
| A small favorable synthetic example could masquerade as practical success | Use constructed cases only as instrument or mathematical controls |
| Missing proof code or inaccessible older literature could be hidden | State supplied-file and source-access limits; include all executable definitions directly |
| A full software harness would add work without testing a live mechanism | Stop that branch after G3; keep only the diagnostic implementation |
| Hosted timings could be mislabeled as M3 results | Require actual environment inspection and separate target measurements |
| Individually valid certificates could omit most targets or understate a Hall neighborhood | Check exact primary target-set equality and reconstruct each complete Hall neighborhood independently |
| Direct CLI examples could bypass cumulative budget accounting | Apply the same outer deadlines and shared ledger to every invocation |

Stress cases are explicitly handled: nonconstructivity is absent because matching supplies witnesses; failed assumptions cause inapplicability, not reshaping; known-mechanism equivalence limits novelty; synthetic-only success is not practical support; a stronger baseline cannot rescue a constant extra score; preprocessing is charged; precision is exact; resource overruns trigger bounded stops; a small pilot is not used for noisy statistical rejection; unavailable application sources are no longer essential; and a false primary claim must produce a preserved counterexample rather than forced agreement.

Before final delivery, repeat the affected review after any material correction. Do not assert absolute perfection. Report remaining limitations plainly.

---

## 12. References and source-access record

Search and source inspection date: **1 November 2026**. Searches used two web search systems and direct primary-source reading, plus the supplied attachments. Coverage was targeted at the mathematical mechanism, conventional configuration testing, accessible real-defect subjects, and feasibility; it was not an exhaustive global novelty review. Backward references from the supplied note and the 2012 tableau paper, and later configuration-prioritization work, were examined where relevant.

1. **Supplied research note, author not explicitly identified in the document.** *How far finite diagonalization can reach: Matching defects, a distance-balance threshold, and an explicit linear versus logarithmic separation.* 1 November 2026. Attachment **diagonal_escape_note.pdf**, 9 pages. Preparation workspace location: /workspace/scratch/e70f426ab12f/upload/diagonal_escape_note.pdf. Critical locations: §1 p. 1; Proposition 1 and Corollary 2 p. 2; Theorem 3 pp. 3–4; Propositions 4–5 p. 4; Theorem 6 pp. 5–6; Theorem 7 p. 6; computational claims p. 7; novelty and limitations pp. 7–8. All rendered pages inspected. If this attachment is transferred, place it under inputs/diagonal_escape_note.pdf; it is not required at runtime because the relevant definitions are reproduced here.

2. **User-supplied problem and execution requirements.** *Pasted text(20261101-120259).txt.* 1 November 2026. Preparation workspace location: /workspace/scratch/e70f426ab12f/upload/Pasted text(20261101-120259).txt. Includes the feature-flag testing problem, permitted narrowing, evidence-gate requirements, and permission to stop on a verified obstruction. Read in full. No external URL.

3. **Srečko Brlek, Michel Mendès France, John Michael Robson, and Martin Rubey.** *Cantorian Tableaux and Permanents.* L'Enseignement Mathématique 50 (2004), 287–304; preprint 8 August 2003. [arXiv record](https://arxiv.org/abs/math/0308081); [full preprint](https://arxiv.org/pdf/math/0308081). Inspected §1 definitions and Theorem 8/proof, preprint pp. 6–7. Establishes the tableau permanent and perfect-matching approach.

4. **Srečko Brlek, Jean-Philippe Labbé, and Michel Mendès France.** *Combinatorial Variations on Cantor's Diagonal.* Journal of Combinatorial Theory, Series A 119 (2012), 655–667. [DOI: 10.1016/j.jcta.2011.11.006](https://doi.org/10.1016/j.jcta.2011.11.006); [preprint](https://arxiv.org/pdf/1104.1083). Inspected introduction, finite permanent definition, scope, and references. Its enumeration/equivalence focus does not establish the proposed application.

5. **Hsuan-Yin Lin, Stefan M. Moser, and Po-Ning Chen.** *Weak Flip Codes and their Optimality on the Binary Erasure Channel.* IEEE Transactions on Information Theory 64(7) (2018), 5191–5218. [DOI: 10.1109/TIT.2018.2834924](https://doi.org/10.1109/TIT.2018.2834924); [preprint](https://arxiv.org/pdf/1711.03310). Inspected Definitions 31 and 33, Eqs. (46)–(50), preprint pp. 16–17, and introductory coding-theory context.

6. **Michel Mendès France.** *Cantorian tableaux revisited.* Functiones et Approximatio Commentarii Mathematici 37(2) (2007), 329–336. [DOI: 10.7169/facm/1229619656](https://doi.org/10.7169/facm/1229619656). Bibliographic identity verified through the primary literature. Full text was not retrieved: DOI/access attempts and a Project Euclid security check prevented inspection. No claim is made about its unseen contents. This is a material historical-novelty gap.

7. **NIST Computer Security Resource Center.** *Combinatorial Methods for Trust and Assurance: FAQs; Downloadable Tools.* Undated live documentation, accessed 1 November 2026. [FAQs](https://csrc.nist.gov/projects/automated-combinatorial-testing-for-software/faqs); [tools](https://csrc.nist.gov/projects/automated-combinatorial-testing-for-software/downloadable-tools). Read the definition of t-way coverage and current ACTS availability. Used as authoritative practical context, not as proof that a chosen strength finds all faults.

8. **Huayao Wu, Changhai Nie, Justyna Petke, Yue Jia, and Mark Harman.** *An Empirical Comparison of Combinatorial Testing, Random Testing and Adaptive Random Testing.* IEEE Transactions on Software Engineering, accepted manuscript 2018. [DOI: 10.1109/TSE.2018.2852744](https://doi.org/10.1109/TSE.2018.2852744); [author-repository PDF](https://discovery.ucl.ac.uk/id/eprint/10053044/8/Petke_08405609.pdf). Inspected §§2.2–2.4, Algorithm 1, study-design sections, and relevant threats to validity. The greedy uncovered-tuple objective is prior art; the exact full-candidate implementation here is a small-domain design choice.

9. **Microsoft PICT contributors.** *Pairwise Independent Combinatorial Tool; PICT documentation.* Undated maintained repository, accessed 1 November 2026. [Repository](https://github.com/microsoft/pict); [documentation](https://github.com/microsoft/pict/blob/main/doc/pict.md). Read usage, constraints, seeding, and build guidance. No PICT binary was installed or benchmarked for this brief.

10. **Michael Schröder, Katja Kevic, Dan Gopstein, Brendan Murphy, and Jennifer Beckmann.** *Discovering Feature Flag Interdependencies in Microsoft Office.* ESEC/FSE 2022 Industry Track, November 2022. [Conference abstract and paper links](https://2022.esec-fse.org/details/fse-2022-industry/4/Discovering-Feature-Flag-Interdependencies-in-Microsoft-Office); [DOI: 10.1145/3540250.3558942](https://doi.org/10.1145/3540250.3558942). Conference abstract and method summary inspected; no proprietary Office data accessed. Supports the practical motivation, not diagonal-generator prevalence.

11. **LaunchDarkly.** *Testing code that uses feature flags.* Undated live guide, accessed 1 November 2026. [Guide](https://launchdarkly.com/docs/guides/flags/testing-code). Inspected recommended current/upcoming/fallback/problematic scenarios. No account or paid service is required by this project.

12. **OpenFeature project.** *Specification, §3: Evaluation Context.* Live specification, accessed 1 November 2026. [Specification](https://openfeature.dev/specification/sections/evaluation-context/). Inspected overview and context requirements. Establishes why modeled Boolean values must be actual evaluations under a fixed context.

13. **LaunchDarkly.** *Flag prerequisites.* Undated live documentation, accessed 1 November 2026. [Documentation](https://launchdarkly.com/docs/home/flags/prereqs). Inspected prerequisite/dependent evaluation behavior. Used to motivate valid-domain care.

14. **Axel Halin, Alexandre Nuttinck, Mathieu Acher, Xavier Devroey, Gilles Perrouin, and Benoit Baudry.** *Test them all, is it worth it? Assessing configuration sampling on the JHipster Web development stack.* Empirical Software Engineering 24 (2019), 674–717; online 17 July 2018. [DOI and open full article](https://doi.org/10.1007/s10664-018-9635-4); [preprint](https://arxiv.org/abs/1710.07980). Inspected abstract, configuration-sampling discussion, §6.2, and conclusions. Supports fault/failure distinctions, comparison design, and cost awareness.

15. **Terser contributors.** *Options: Compress options; CHANGELOG.md.* Maintained documentation, accessed 1 November 2026. [Options](https://terser.org/docs/options/); [changelog](https://github.com/terser/terser/blob/master/CHANGELOG.md). Inspected Boolean options, defaults/unsafe caveats, and v5.16.1–v5.16.9 entries. Historical effective defaults and release SHAs were not fully reconstructed; this is not a verified subject package.

16. **Lorenzo Dalla Vecchia; Fábio Santos and Terser contributors.** *Name collision caused by let in loop initializer*, issue #997, opened 24 May 2021; fix committed 27 January 2023. [Issue](https://github.com/terser/terser/issues/997); [fix 79d76135344c8eb0b5bda8e311878888ec3047ab](https://github.com/terser/terser/commit/79d76135344c8eb0b5bda8e311878888ec3047ab). Issue and actual diff inspected; regression location test/compress/collapse_vars.js, label issue_997. No reproduction was executed.

17. **hyfdev; Fábio Santos and Terser contributors.** *[Bug]: hoisted var declaration will causing unexpected simplification*, issue #1306, opened 13 November 2022; fix 2 February 2023. [Issue](https://github.com/terser/terser/issues/1306); [fix 7f6187a461340451c2f6bfd2cfd6544de24844a8](https://github.com/terser/terser/commit/7f6187a461340451c2f6bfd2cfd6544de24844a8). Issue and diff inspected; relevant regression test/compress/reduce_vars.js, label defun_reference_2. Its original option profile includes toplevel=true; no silent projection onto a different study profile is justified.

18. **Son Nguyen, Hoan Nguyen, Ngoc Tran, Hieu Tran, and Tien N. Nguyen.** *Feature-Interaction Aware Configuration Prioritization for Configurable Code.* ASE 2019; preprint November 2019. [Full preprint](https://arxiv.org/pdf/1911.07779). Abstract and introduction inspected. CoPro uses program-derived interaction information; this is relevant prior art for prioritization, not an implementation reproduced here.

### Final instruction

Complete the authorized diagnostic and follow its evidence gates. If the expected obstruction is verified, deliver the scoped negative conclusion and stop the application-benchmark branch. If it is not verified, preserve the exact contradictory evidence, complete the bounded diagnosis, and report the resulting status honestly. Do not substitute a favorable synthetic demonstration or an unperformed real-defect study for the decision this brief specifies.

