# PoC execution brief: certified qubit budgets for repeated reversible simulation

**Recipient:** an autonomous coding agent configured as `gpt-6.1-sol` with `xhigh` reasoning.  
**Planning and research date:** 1 November 2026.  
**Deliverable of this assignment:** execute the authorized stages below and produce a reproducible research result. This document is the execution instruction, not a claim that the PoC has already been implemented or benchmarked.  
**Target experimental machine:** one MacBook with an M3 Max; inspect its actual configuration before allocation.

## 1. Project decision and contribution

Build a small **certificate-aware compiler diagnostic for finite deterministic dynamics**. Given an explicit function table `f`, a horizon `T`, and a logical-qubit budget, it must separate mathematically impossible budgets from valid compiled implementations and unsuccessful searches. Then test whether the supplied exact-capacity result helps a bounded synthesis pipeline find useful circuits.

The primary research question is:

> Under a fixed total budget of eight logical qubits, can an exact capacity certificate plus constructive permutation candidates increase the number of verified finite-horizon simulations meeting predeclared gate and depth limits, relative to an independently implemented search of the same behavioral contract with the same compilation allowance?

The comparison includes a certificate-only ablation. It distinguishes gains from ruling out impossible widths, gains from the constructive representation, and gains from ordinary synthesis choices.

### 1.1 Selected scope and honest contribution

The smallest defensible contribution is a reproducible connection between:

1. an independently checked mathematical capacity certificate;
2. concrete phase-correct circuits with complete width accounting;
3. the feasibility boundary of specified compilation procedures; and
4. the additional memory imposed by requiring one unchanged block.

This may yield a supported conditional compiler improvement, a useful limitation of the construction, or a negative result in which exact certificates add no useful circuit benefit. Implementing a known bound, confirming it numerically, or beating only a history tape is not by itself a novel engineering contribution.

**Material revision of the original objective:** no supplied application or inspected source establishes a deployed workload requiring all of the exact fixed-block, every-input, every-intermediate-output, finite closed-environment conditions. Therefore this is initially a **conditional compiler-method study**, with synthetic compiler regression workloads. Do not advertise application-level engineering success without independent architecture evidence. This restriction preserves the resource-decision purpose while withholding claims about adoption, production quantum algorithms, and hardware advantage.

At Stage 0, record any supplied design document, workload owner requirement, or integration specification that independently requires the contract. “Repeated circuits are common,” a Floquet analogy, or a requirement introduced to favor this theorem is insufficient. If no evidence exists, record `application_status: conditional_microbenchmark` and continue the diagnostic; do not invent a customer or stop useful mathematical work.

### 1.2 Fixed requirements and exclusions

The main study uses exact deterministic tables on all `2^b` visible bit patterns, common blank initialization, one repeated permutation block, no resets or measurements, and correctness after every block. It uses an ideal logical gate model with unrestricted connectivity. Persistent auxiliaries remain allocated throughout the run.

Exclude stochastic transitions, approximate visible dynamics, restricted initial states, changing visible encodings, externally supplied iteration controls, indefinite operation, fault-tolerant synthesis, noise, energy claims, quantum speedup, single-cycle constraints, and physical-qubit estimates. The source's cycle-counting results are not a work package.

Broad certificate coverage includes 4–64 visible states. The main compiled comparison uses 4–16 visible states, with four explicit scaling cases at 32 and 64 states. The hard eight-qubit envelope, rather than visible-state count alone, bounds quantum verification. Do not claim compiled coverage for all broad certificate cases.

Use ordinary Python modules, local JSON/JSONL/CSV files, and a small TOML configuration. No service, dashboard, distributed execution, GPU requirement, or external compute is needed.

## 2. Evidence available before you begin

### 2.1 Supplied inputs and provenance

Place received inputs in `inputs/`. The core specification below is self-contained; the PDF is needed for source fidelity, not as a hidden algorithm oracle.

| Input | Provenance and relevant content |
|---|---|
| `finite_reversible_simulation.pdf` | Nine-page research note, *Exact finite reversible simulation: State capacities, optimal constructions, and cycle constraints*, dated 1 November 2026. PDF metadata credits “Research investigation prepared with ChatGPT”; no individual author or peer-review venue is supplied. Use [S1]. |
| `verification_results.json` | Aggregate results claimed by that note. It reports 608 map–horizon cases, 19,224,927 labeled permutations, and additional constrained enumerations. Use [S2] as reported evidence, not as independently reproduced results. |
| `Pasted text(20261101-120148).txt` | Original assignment and problem contract. Its operational requirements have been incorporated here. It is not needed to execute this brief. |

SHA-256 identifiers:

~~~text
finite_reversible_simulation.pdf
25bad91f15075fea2f8d072529780400d60231dd384740b794f9a0ec9c1913c8

verification_results.json
29e90ba27362073be4d30442fd0a74dfd6e28625ba07067520c8fc31b450c433

Pasted text(20261101-120148).txt
dbb9ff9715e96300e59b73aa3de4df69b588ac14028d8b9d45271fb13304e514
~~~

During planning, these were available under `/workspace/scratch/bafe825be3a4/upload/`. That path is historical provenance, not a path to assume exists on the Mac.

The note says a companion verification program and document source were supplied together. They were **not present in the received attachments**. This does not prevent the proposed study: implement independent checks specified below. Do not claim to reproduce the original program, its timings, or its exact enumeration counts. If the PDF is missing from your handoff, complete work supported by the fully stated mathematics here and mark your own source inspection unavailable; ask for that one file only if a source discrepancy cannot otherwise be resolved.

### 2.2 Work actually performed during planning

The planner and separate mathematical reviewer read all nine PDF pages; equations were inspected in rendered pages because text extraction omitted them. The core proofs in Theorem 2, Corollary 3, and Theorem 4 were checked. No defect was found in the capacity result under its stated assumptions. This is mathematical review, not formal proof-assistant verification.

A separately written, standard-library probe did the following:

| Planning check | Actual result |
|---|---|
| Every function on `n=1,2,3,4` states and `T=1,2,3,4`: direct profile, excess-indegree profile, constructive product realization, time-dependent realization | 1,152 map–horizon cases; all checks agreed |
| Independent partial-permutation search on those cases at each product capacity `M=1,2,3,4`, without theorem pruning | 4,608 feasibility cases; 972,814 search nodes; no mismatches or capped cases |
| Largest individual feasibility search | 61,597 nodes |
| Re-execution by the planner | Same discrete results; roughly 0.28 seconds in the hosted environment |

The probe searched arbitrary integer product capacities, including nonpowers of two. You must additionally verify binary padding and register conventions. The planning probe is evidence supporting this design, not a substitute for the implementation's independent tests. Its search algorithm is fully specified in Section 5.3.

A second small NumPy probe checked the four-sweep controlled-gate construction in Section 6.3 for all matrix columns with 0–7 controls. Maximum absolute entry error was at most `8.92e-16`; gate counts agreed with the stated formulas. That first probe checked CP/CRX-level templates.

During adversarial review, a separately translated H/P/CX primitive implementation was checked on all columns with 0–7 controls; maximum absolute entry error was approximately `1.39e-14`. The transformation synthesizer in Section 6.2 was also independently translated and checked on all 40,347 permutations of domains of size 1, 2, 4 and 8, plus twenty deterministic random 256-state permutations. Every row-fixing invariant and exact integer circuit replay passed. These checks took roughly 3.2 seconds in the hosted environment. Reimplement and rerun the relevant checks on the target; these are planning diagnostics, not PoC results.

**No compiler comparison, target dependency installation, M3 Max performance measurement, or complete PoC was performed during planning.** All runtime estimates and success hypotheses below remain unmeasured. Literature and official source-code inspection are distinguished from executable validation.

## 3. Mathematics and its executable consequences

### 3.1 Behavioral contract

Let

\[
X=\{0,\ldots,n-1\},\qquad n=2^b,\quad b\in\{2,\ldots,6\},
\]
\[
f:X\to X,\qquad T\in\mathbb Z,\quad T\ge1.
\]

The public compiled input is the entire integer table `f[0:n]`. Reject wrong length, out-of-range entries, noninteger entries, or unspecified values.

Internal mathematical routines and the tiny feasibility oracle support **any positive integer n**, including 1, 2 and 3. Non-power-of-two cases test the finite-set theorem and do not represent a qubit register. Only the public compiled tier requires `n=2^b` in the stated range; apply binary-register tests only to power-of-two instances. Keep these validators separate.

An implementation has `b` visible qubits and `a` persistent auxiliary qubits. The same circuit `U` is applied `T` times, and

\[
U^t|x,0^a\rangle=|f^t(x),g_t(x)\rangle
\quad(x\in X,\;0\le t\le T).
\]

Here `g_t(x)` is a bit string, and the complete block is a phase-free permutation on **every** computational-basis state, including states not reached from blank initialization. No assertion is made for `t>T`. The visible register, its physical mapping at block boundaries, and its integer interpretation stay fixed.

A change in the hidden encoding across times is permitted only as behavior of the same fixed block. An external controller may repeat the identical block, but may not choose a different update, swap an answer register externally, reset workspace, or supply new qubits between applications. Quantum gates may create intermediate superpositions inside a block; its completed action must be the specified permutation.

Primary experiments have no end-of-run cleanup requirement. Persistent garbage is part of the output state. If an actual application needs cleanup, declare and count that use case before its experiment; Section 8.5 gives the appropriate accounting.

### 3.2 Exact capacities

Define

\[
d_y=|\{x:f(x)=y\}|,\qquad
L=X\setminus f(X),\qquad \ell=|L|.
\]

Leaves mean zero indegree, not terminal states. Define the exact observation-fiber capacity

\[
q_T(y)=|\{x:f^T(x)=y\}|
+\sum_{j=0}^{T-1}|\{\lambda\in L:f^j(\lambda)=y\}|.
\tag{1}
\]

[S1, Theorem 2, equations (3)–(5), p. 2] establishes that prescribed fiber capacities `m_y` are feasible exactly when `m_y≥q_T(y)` for every `y`. Consequently,

\[
N_{\min}=n+T\ell,\qquad
M_T=\max_y q_T(y),\qquad
a_{\min}=\lceil\log_2 M_T\rceil.
\tag{2}
\]

`N_min` is the minimum total cardinality of an unrestricted hidden state space. `M_T` is the minimum dimension of a common product environment, by [S1, Corollary 3, p. 3]. For qubits, compute `a_min=(M_T-1).bit_length()` with integer arithmetic. In particular, `M_T=1` gives zero auxiliaries.

**Never estimate the auxiliary requirement by rounding `N_min/n`.** Product fibers all have the same capacity, so the largest fiber controls width. Report `M_T`, `a_min`, the binary capacity `2^a_min`, and padding separately. A capacity reduction that does not cross a power of two saves no qubit.

For an efficient exact evaluator, put

\[
e(y)=\max(d_y-1,0),\qquad
(f_*w)(y)=\sum_{x:f(x)=y}w(x).
\]

Then [S1, Section 3.4, equation (6), p. 3]

\[
q_T=\mathbf1+\sum_{j=0}^{T-1} f_*^j e.
\tag{3}
\]

~~~text
capacity_profile(f, T):
    validate f, T
    d = indegrees(f)
    mass[y] = max(d[y] - 1, 0)
    q[y] = 1
    repeat T times:
        q += mass
        next_mass = zeros(n)
        for x in 0..n-1:
            next_mass[f[x]] += mass[x]
        mass = next_mass
    return q, max(q), bit_length(max(q)-1)
~~~

This costs `O(nT)` integer additions and `O(n)` working memory. Values are at most `1+T(n-1)`. There is no numerical conditioning, convergence tolerance, discretization, inverse-function oracle, or hidden exponential counting problem for these explicit tables. Succinct-program inputs would be a different complexity regime.

### 3.3 Why the certificate is sound

The source proof counts

\[
U^T s(X),\qquad U^j s(L)\quad(0\le j<T),
\]

where `s(x)=(x,0)`. These sets are disjoint. A collision between different layers can be shifted backward by the common power of the permutation. Observation then says that a positive iterate of `f` equals a leaf, impossible because every positive iterate lies in `f(X)`. Within one layer, permutation and initialization injectivity prevent collisions. Counting required states with visible label `y` gives (1).

For arbitrary unitary hidden dynamics, the same argument makes the corresponding initialized vectors orthogonal. Their projections lie in orthogonal visible-label subspaces. Therefore arbitrary quantum hidden states cannot beat this dimension bound when initialized basis inputs have exact visible outputs [S1, Theorem 4 and proof, pp. 3–4].

This stronger lower bound does **not** specify a channel on arbitrary superpositions after discarding the environment. The present circuit contract instead fixes the coherent linear extension of a particular full permutation. Do not conflate these statements.

### 3.4 Constructive sufficiency and binary embedding

Use the following explicit construction from [S1, Section 3.2, pp. 2–3].

1. For each `v∈f(X)`, choose one predecessor `p(v)∈f^{-1}(v)`. The canonical choice is the smallest integer predecessor.
2. Retain the edges `p(v)→v`. These form disjoint paths and cycles. Let `D=X\p(f(X))` be the path ends; `|D|=|L|=ell`.
3. Choose a bijection `beta:D→L`, canonically sorted order.
4. For each `d∈D`, add `T` distinct states `(d,1),...,(d,T)` with labels `f(d),...,f^T(d)`.
5. Replace the unassigned successor of `d` by the chain
   \[
   d\to(d,1)\to\cdots\to(d,T)\to\beta(d).
   \]
   Retained edges remain. Each original state `x` is its own initialized representative with visible label `x`.

Every state has exactly one predecessor and successor. From any representative, an incorrect return after a path end is at least `T+1` steps away. The construction has exactly `n+T*ell` states and exactly `q_T(y)` states of each visible type.

For any `a≥a_min`, let `K=2^a`. Assign original representative `y` hidden label zero in fiber `y`. Give the remaining states of that fiber distinct labels in `1..K-1`. The canonical order is original construction order; later variants may permute those labels. Use the external integer convention

\[
\mathrm{enc}(y,h)=(y\ll a)\;|\;h.
\tag{4}
\]

Conjugate the constructed permutation by this encoding. All unassigned padded basis states are fixed points. This produces a permutation array of length `nK` with common all-zero initialization.

~~~text
embed_product(nodes, labels, successor, a):
    K = 2^a
    for each visible y:
        assign original node y -> (y, 0)
        assign other nodes labeled y to distinct (y, h), h != 0
        fail if the fiber exceeds K
    P = identity permutation on n*K states
    for every constructed node z:
        P[enc(z)] = enc(successor[z])
    return P
~~~

Construction before padding is `O(nT)`; a dense product table costs `O(n2^a)` time and space. For broad certificate cases, a sparse nonidentity mapping is sufficient: require identical key and value sets and treat omitted states as fixed. Materialize dense arrays only for the compiled tier.

**Search-space limitation:** Theorem 5 classifies minimum **total-cardinality** simulators. Minimum-qubit realizations have `n2^a` states and may use their extra states in more helpful ways than fixed-point padding. Predecessor choices, path joins, fiber labelings, and padding therefore give a sufficient candidate family, not all minimum-width circuits. Failure of that family to compile cheaply proves no gate/depth impossibility.

### 3.5 A sharp alternative-architecture calculation

The following is a derivation made for this brief, not a claim of historical novelty. If separate permutation blocks `U_t` are allowed at each step, with the **same visible register** and every intermediate output still required, the exact auxiliary capacity is

\[
m_T=\max_y|\{x:f^T(x)=y\}|,\qquad
a_{\mathrm{vary}}=(m_T-1).\mathrm{bit\_length}().
\tag{5}
\]

Necessity is injectivity at the final time. For sufficiency, at each `t` give each input `x` a distinct hidden rank among the inputs with visible value `f^t(x)`. This defines injections

\[
h_t(x)=(f^t(x),r_t(x)),\qquad h_0(x)=(x,0).
\]

Maximum fiber multiplicity is nondecreasing in `t`, because a fiber can only merge into a later fiber. Thus all `h_t` fit in capacity `m_T`. Each partial bijection `h_t(x)→h_{t+1}(x)` extends separately to a full permutation. Pad to a binary environment as in (4).

Final-only computation has the same capacity bound. Its circuit-cost problem is different again. Record the exact **capacity cost of the fixed-block restriction**:

\[
\Delta a=a_{\min}-a_{\mathrm{vary}}\ge0.
\]

A positive `Delta a` is a reason to challenge the architecture, not evidence that the proposed compiler improves an application. Compile selected relaxed alternatives using the same synthesis backend in the bounded architecture challenge, and count their entire sequence of gates. Do not compare one varying block against `T` applications of the fixed block.

### 3.6 Analytic fixtures

These formulas are source consequences or direct derivations; recheck them independently.

| Function | Fixed-block `M_T` | Varying-block/final-only `m_T` | What it tests |
|---|---:|---:|---|
| Any permutation | `1` | `1` | Zero-auxiliary control |
| Constant map | `1+T(n-1)` | `n` | Maximal merging and a long-horizon obstruction |
| Countdown `max(x-1,0)` | `T+1` | `min(T+1,n)` | Nonconstant structured loss |
| Clear `r` low bits | `1+T(2^r-1)` | `2^r` | Idempotence does not authorize reset |
| Right shift `x>>1`, `T≤b` | `2^T` | `2^T` | No capacity advantage from relaxing autonomy before saturation |
| Right shift, `T≥b` | `n+(T-b)n/2` | `n` | Continuing after settling |
| `f=(0,0,0,3)` | `1+2T` | `3` | Isolated recurrent component plus merging branches |

For countdown, `q_T(0)=T+1` and every other coordinate is one. For clear-low-bits, every image coordinate has the displayed maximum and every other coordinate is one. At `T=1`, `M_1=max_y d_y`, the established one-step embedding bound [R1]; that control cannot establish novelty.

## 4. Certificates, statuses, and interfaces

### 4.1 Impossibility certificate

For a proposed persistent auxiliary budget `a`, choose a visible `y` with `q_T(y)>2^a`. Emit `b,T,a,y`, the full `f` table and its hash, and disjoint-layer identifiers:

- `(T,x)` for each `x` with `f^T(x)=y`;
- `(j,leaf)` for each leaf and `0≤j<T` with `f^j(leaf)=y`.

The two identifier types must be tagged explicitly even when their integer fields look similar. A subset with `2^a+1` distinct identifiers is sufficient; emit the complete list at this scale.

A separate verifier must recompute leaf membership and trajectories directly, check all identifiers and their count, and use the disjoint-layer argument in Section 3.3. It must not call the producer's recurrence routine. Attach a short human-readable explanation of the witness.

This is a checkable combinatorial certificate backed by an elementary proof, not a proof-assistant object. A solver's `unsat` answer or timeout is not a replacement.

### 4.2 Feasibility and circuit certificates

The abstract certificate is the permutation array, or an explicitly described sparse permutation, with the fixed encoding and common blank representative. An independent checker verifies range, bijectivity, and every `(x,t)` requirement by repeated application. It must not accept a construction merely because its reported capacity is correct.

A compiled witness additionally includes:

- total allocated wires and their visible/auxiliary roles;
- a phase-free X/MCX circuit with exact integer replay;
- the primitive symbolic circuit and exact decomposition trace;
- gate/depth metrics, source and dependency identifiers;
- phase-sensitive verification results and their numerical error;
- any global phase or wire-mapping metadata, explicitly resolved.

Default physical wire `i` corresponds to bit `i` of the integer in (4): auxiliary wires `0..a-1` and visible wires `a..a+b-1`. A toolkit may serialize in a different order. The adapter must demonstrate the conversion using asymmetric basis states, not infer it from a drawing. Logical naming alone does not authorize an implicit SWAP at a repeated-block boundary.

### 4.3 Status model

Store separate mathematical, search, circuit, and budget statuses; do not compress them into one success flag.

| Reported finding | Required evidence |
|---|---|
| Budget impossible under the model | Independently accepted witness with `2^a<M_T` |
| Abstractly feasible | Verified permutation, irrespective of synthesis result |
| Valid implementation within budget | Accepted abstract and circuit checks, plus all width/gate/depth limits |
| Above abstract minimum | Measured implementation width exceeds `b+a_min`; explain any useful extra memory |
| Unresolved search | Timeout, incomplete enumeration, missing backend, or no accepted circuit |
| Invalid candidate | A concrete contract, permutation, phase, mapping, or accounting failure |
| Not run under resource gate | A preregistered resource decision; remains in coverage tables |

For peak width `Q` with temporary workspace, `Q<b+a_min` is impossible. If some workspace is forced back to zero at every block boundary, it does not enlarge the boundary capacity available to store history. Count both persistent capacity and temporary peak allocation. The main exact backend uses no additional workspace, but assert this from the emitted circuit.

## 5. Candidate generation and the independent baseline

### 5.1 Three comparison arms

All arms use identical input tables, encoding, gate library, connectivity, tuning data, completion methods, synthesis portfolio, verification, and per-instance time/candidate ceilings.

| Arm | Information and candidates |
|---|---|
| `G: generic` | Direct trajectory CSP, known final-fiber lower bound (5), reversible control shortcut, and the preallocated ring reference below. No use of (1)–(3), normal-form choices, or theorem-derived seeds. |
| `C: certificate_only` | Exactly G, with independently verified capacity certificates pruning impossible auxiliary widths. No paper construction. |
| `F: certificate_and_construction` | C plus the finite-path construction and bounded encoding variants; generic CSP remains available to escape that construction family. |

G is an independently implemented solver of the full contract, not a circuit designed to retain every state. Its correctness and available search space do not depend on the supplied theorem.

The arms answer different causal questions: `C-G` measures the effect of certification; `F-C` measures constructive candidates; `F-G` measures the complete intervention. Preserve all three even when one is unfavorable.

### 5.2 Exact trajectory CSP

For a chosen `a`, set `K=2^a` and precompute `v[x,t]=f^t(x)`. Introduce integer variables

\[
0\le h_{x,t}<K,\quad h_{x,0}=0,\quad
s_{x,t}=K\,v[x,t]+h_{x,t}.
\]

For every pair of required edges `(s[x,t],s[x,t+1])` and `(s[z,u],s[z,u+1])`, impose

\[
s_{x,t}=s_{z,u}
\quad\Longleftrightarrow\quad
s_{x,t+1}=s_{z,u+1}.
\tag{6}
\]

The forward direction makes the partial edge relation functional; the reverse makes it injective. These conditions are necessary and sufficient for that relation to extend to a full permutation of `nK` states. This is an elementary derivation for this brief, not a claim of a new CSP method.

Use `z3-solver` with finite integer domains, one thread, fixed recorded random seeds, and explicit timeouts. The unoptimized formulation has `O((nT)^2)` pair constraints. Simplify comparisons whose visible coordinates are unequal. Adding per-time all-distinct states is redundant and sound. Do not add restrictions derived from the paper to G.

Do not use an optimizer pretending to minimize gate count through a proxy without measuring the resulting circuits. Enumerate at most four distinct hidden-label models per instance-arm stream. Block the complete nonconstant hidden-label assignment after an accepted model. Use at most two fixed solver seeds, shared by the arms. Avoid speculative symmetry constraints in the first implementation; if added during development, prove they only remove per-fiber relabeling and test them against the tiny exhaustive oracle.

From a model, deduplicate identical required edges and check functionality and injectivity directly. Complete the unused source/target sets using each of these three fixed policies:

1. Fix every state that is both an unused source and an unused target, then sorted-pair the residual sets.
2. Greedily assign each unused source its unused target of smallest Hamming distance; break ties by integer value.
3. Sorted sources paired with a deterministically shuffled target list.

All three are true bijections. They may change unrestricted padding behavior and can find realizations outside the theorem's normal-form family. Their cost, including optimization, belongs to their arm.

### 5.3 Tiny independent feasibility oracle

Write a small standard-library depth-first search separate from Z3 and the capacity producer:

~~~text
exact_product_search(f, T, K, node_limit):
    succ = unassigned array of length n*K
    predecessor_used = false array of length n*K
    process input x=0..n-1, each beginning at state x*K

    recursively follow T required steps of that input:
        wanted_visible = next classical trajectory label
        if succ[current] already assigned:
            reject this branch if its visible label is wrong
            otherwise follow the assigned successor
        else:
            try every state with wanted_visible whose predecessor is unused
            assign current -> candidate, recurse, undo on failure

    after all inputs succeed:
        complete unused source/target sets to a permutation
        independently verify it and return SAT witness

    exhaustive failure -> UNSAT
    node or time cap -> UNKNOWN
~~~

The search maintains assignments across inputs, including complete rollback on a failed branch. It uses no capacity formula. Search all maps on `n=1..4`, `T=1..4` and `K=1..4` with a per-case cap of 500,000 nodes and a global Stage 1 deadline. This matches the planning check's finite domain. Also test the binary widths immediately below and at the formula where `nK≤16`. Any capped case is unknown, not an inconsistency.

For Z3 cross-checks, use all `n≤3,T≤3,K≤4` cases and a deterministic sample of at least 32 `n=4` cases spanning feasible/infeasible capacities. Retain the full selected list. A mismatch is a stop-and-debug event, never a benchmark result.

### 5.4 Independent constructive reference: preallocated ancilla ring

The ring is a valid fixed block, inspired by established power-dilation constructions [R4, Section III.C]. It is a reference and a portfolio candidate, not the sole strong baseline.

Let `r=ceil(log2(max_y d_y))`. Construct a one-step permutation `V` on `b+r` wires satisfying `V(x,0)=(f(x),rank(x))`, where ranks distinguish predecessors of each output. Complete the partial bijection using the same three policies as above.

Allocate **T slots of r qubits at the beginning**, all zero. A block applies `V` to the visible register and the first slot, then cyclically rotates the slots. Implement the rotation with explicit SWAPs; each SWAP costs three CX. The first `T` invocations each encounter a previously unused zero slot, all already present in the initial register. No memory is supplied between blocks.

This has `b+Tr` wires and fixed visible output. For a permutation, `r=0` and the reference is simply `V=f`. Validate the rotation direction and all trajectories. Do not treat output garbage as erasable or silently remove the rotation.

If the ring fits the primary width, evaluate its naturally structured exact circuit (the commonly synthesized one-step V plus explicit slot SWAPs) and its full permutation through the common synthesis portfolio. These two routes share one candidate slot for the same full permutation; their measured work is charged to that arm. The structured shortcut is identically available in G, C and F. Apply all ordinary phase and resource checks to the selected result.

If the ring exceeds the primary budget, record its valid width. A compositional gate estimate is optional and must be labeled as a derived estimate unless its one-step circuit was actually compiled within an existing allowance. Do not spend an additional synthesis allowance on this reference. For the broad certificate matrix, report ring width only; never expand its potentially enormous full permutation or unitary. It cannot alone support a claim of F outperforming a functioning independently implemented baseline.

### 5.5 Bounded search policy

For the main eight-qubit envelope, `A=8-b` is the largest permitted persistent auxiliary width. G's lower bound is `a_vary`; C and F use `a_min`.

Use the following fixed serial schedule. The limit is `D=90` seconds in confirmation. The parent sets the deadline before starting the worker; imports, preparation, constraint construction, solver calls, compilation, lowering, cost measurement, verification, and witness serialization all count.

**Generation phase**

1. Allow at most `min(30,D/3)` cumulative seconds for all generation work. A solver call, including constraint construction, has at most five seconds and cannot exceed the remaining generation or arm allowance. Stop after four accepted hidden-label models across all widths.
2. Visit width A first, then smaller admissible widths in ascending order; repeat this width order in subsequent rounds. Keep model-blocking clauses separately for each width. Skip a width proved UNSAT after its existing blocking clauses; a timeout leaves it unresolved. Alternate the two fixed solver-seed indices by round. Derive generic seeds from `(instance_digest,width,solver_seed_index)`, without the arm name, so the shared generic streams agree whenever the arms reach the same width and model number.
3. Keep separate queues for common ring/reversible references, generic CSP candidates, and F's constructive candidates. Prepare common references first; F then prepares its at most four construction candidates; use the remaining generation allowance for the generic stream. F pays for this added preparation rather than receiving extra time.
4. F's four candidates are canonical and seeded constructions at `a_min` and A. If these widths coincide, use the canonical construction and three separately seeded variants at that width. Seeded variants choose predecessors uniformly within each fiber, an endpoint bijection, and nonzero fiber labels; all preserve visible labels and blank representatives. Use separate purpose tags for construction, completion and solver randomness.
5. For accepted generic models, enqueue completion policy 1 for every model in width-visit order, then policy 2 for every model, then policy 3. Completion work shares the generation ceiling; retain a reserve of `min(2,D/15)` seconds of that ceiling for completing accepted models, and stop solver calls before that reserve. If completion itself reaches the ceiling, retain only completed valid permutations and log the rest as not completed.
6. Admit at most 12 distinct full permutations by round-robin over nonempty source queues, starting with common references, then generic, then construction. Deduplicate by full permutation digest before consuming a slot. The construction source supplies at most four slots. Store all provenance tags on a duplicate, but synthesize it once. Record generated candidates that never reach admission or synthesis; they are not circuit results.

**Synthesis and verification phase**

Use this common compiler order, breadth first over admitted candidates: owned forward/normal, TKET Cycle, owned inverse/normal, owned forward/reversed, owned inverse/reversed. Thus try the first route on each candidate before the second route. For a ring candidate, try its structured route immediately before its full-permutation owned forward/normal route. This extra common route uses the same candidate slot, time limits and accounting in every arm.

A synthesis attempt includes permutation synthesis or toolkit extraction, exact integer replay, primitive lowering, audited simplification, and resource measurement. Its limit is `min(synthesis_limit,D/6)` seconds, where `synthesis_limit` is 10 seconds, or 5 only under the prelock calibration rule. Before a fully verified fit exists, its deadline is also capped at `arm_start+D-min(30,D/3)`. This preserves verification time even for an attempt launched just before the cutoff.

As soon as a completed circuit passes exact checks and the gate/depth/width limits, attempt full phase verification, with a cap of `min(30,D/3,arm_deadline-now)` seconds. A verification timeout leaves the candidate unverified; another attempt is allowed only if the deadline permits. A failed phase or mapping check makes the candidate invalid. If it exposes a material implementation defect, follow Section 10's bounded repair and affected-run invalidation rules before interpreting comparative results. Once an incumbent fit is fully verified and saved, remaining time may seek improvements; an unverified improvement never replaces it. Retain completed nonfitting circuits as resource evidence, without spending mandatory phase-verification time on circuits already known to exceed the limits.

Return the **best fully verified circuit among completed attempts**, not a promise that every candidate received every portfolio strategy. Record time to first verified fit, its remaining deadline margin, and best verified width/cost. Do not promote a circuit to an in-budget fit using postdeadline verification. The parent may record a timeout after terminating a worker, but only witnesses completed and saved before the deadline qualify. Stop early when the finite stream is exhausted.

The same formulas apply to `D=45` pilot runs, `D=30` fallback runs and `D=10` timing microbenchmarks; shorter runs therefore reserve a proportionate interval rather than an impossible fixed 30 seconds. Microbenchmark timeouts remain censored observations and never replace confirmation outcomes.

C and G use identical generic code, apart from certificate pruning. F pays for its certificate and construction inside its allowance. Shared preprocessing may be performed once for efficiency, but charge its measured standalone cost to every arm's total and report shared and unshared timings separately. No arm receives an uncharged precomputed candidate or a successful circuit from another arm. Log attempted and skipped widths. If all widths in C/F are certified impossible, issue certificates immediately; G may still use its allowance, and its timeout remains a search status.

Freeze this schedule and any allowed timeout reduction on development cases. The baseline is a functioning independently implemented search with four models, three completion policies, and the specified allowance; its expressive feasible space does not make that bounded sample globally optimal.

## 6. Exact circuit synthesis and verification

### 6.1 Common logical basis and synthesis portfolio

The primary primitive gate library is

\[
\{X,\ H,\ CX,\ P(\alpha):\alpha\in\mathbb Z[1/2]\},\qquad
P(\alpha)=\operatorname{diag}(1,e^{i\pi\alpha}),
\]

with exact dyadic rational coefficients, all-to-all connectivity, unit cost per listed elementary gate, and no additional synthesis ancillas. It is an ideal logical model, not a Clifford+T or physical fault-tolerance estimate.

For every admitted permutation candidate, attempt the following common routes under the schedule in Section 5.5, and retain the best fully verified circuit among completed attempts:

1. an owned transformation-based X/MCX synthesizer, with forward/inverse and normal/reversed wire-order variants;
2. TKET's `ToffoliBoxSynthStrat.Cycle` route, converted to the same exact X/MCX representation.

The owned transformation route is related to established truth-table synthesis [R8], not a proposed novelty. The current TKET API uses `ToffoliBox`, **not** `BasisPermutationBox`. Its Cycle source generates X/CnX operations on the existing wires [R10]. Verify that property in the installed version; reject unexpected operations from this adapter. Supply a complete basis mapping, including fixed states, so an identity or empty-support permutation does not lose its declared width.

Do not count a permutation box as one gate. Lower every MCX to the exact primitives below, then apply only audited exact local rewrites: adjacent inverse cancellation, adjacent same-wire phase merging modulo two, and cancellation across operations on disjoint wires. No floating-point KAK/peephole resynthesis enters the primary exact result.

Select circuits lexicographically by: meets all budget limits; smallest allocated width; smallest CX count; smallest total depth; smallest total gate count; stable circuit hash. Also retain the nondominated width/CX/depth frontier so useful extra memory is visible.

### 6.2 Unambiguous transformation reference

A simple output-side reduction is sufficient; add its inverse and wire-order variants as the bounded portfolio.

~~~text
synthesize_transform(P, width):
    work = copy(P)
    reductions = []
    for i in 0..2^width-1:
        current = work[i]
        # All already fixed outputs are j < i.
        for bit in increasing order where i has 1 and current has 0:
            G = MCX(target=bit, controls=all 1-bits of current)
            apply G to every output entry of work
            append G to reductions
            current = G(current)
        for bit in increasing order where current has 1 and i has 0:
            G = MCX(target=bit, controls=all 1-bits of i)
            apply G to every output entry of work
            append G to reductions
            current = G(current)
        assert work[i] == i and all earlier rows remain fixed
    assert work is identity
    return reverse(reductions)
~~~

An empty control set means X. Neither control set includes the target. Before processing row `i`, its output is at least `i`, since smaller outputs have already been used by fixed rows. The first type of gate cannot change a smaller fixed integer containing all current one-bits; the second cannot change a smaller integer containing all one-bits of `i`. This proves progress without disturbing prior rows. Inverting the reduction order implements the original permutation.

For the inverse variant, synthesize `P^-1` and reverse its gate order. For a reversed wire-order variant, conjugate the table by that bit-position permutation, synthesize, and translate every gate's wire indices back. This is an internal synthesis ordering, not a free change in the visible output encoding.

For `N=2^w`, this reference uses at most `O(wN)` corrections and `O(wN^2)=O(N^2 log N)` time with scalar full-array updates. Do not present it as scalable to succinct large programs. Verify every resulting X/MCX network by integer simulation over its complete basis before lowering it.

### 6.3 Exact no-workspace MCX lowering

Use the quadratic da Silva–Park construction as a practical exact option [R11, Theorem 1 and equation (3)], following the four-sweep specialization in the inspected Qiskit implementation [R12]. No Qiskit installation is required: implement this small rational-angle algorithm locally and preserve source/license attribution for any translated code.

For `k` controls indexed `0..k-1` and target `k`, use X for `k=0` and CX for `k=1`. Otherwise surround the following sweeps with H on the target. Set

\[
\alpha(c,t)=2^{-(t-c-\mathbf1_{c=0})}.
\]

| Sweep | Index pairs; order | Operation |
|---|---|---|
| A | `0≤c<t≤k`; decreasing `c+t` | `CP(+alpha)` if `t=k`, otherwise `CRX(+alpha)` |
| B | `1≤c<t≤k`; increasing `c+t` | Same gate choice, coefficient `-alpha` |
| C | `0≤c<t<k`; decreasing `c+t` | `CRX(-alpha)` if `c=0`, otherwise `CRX(+alpha)` |
| D | `1≤c<t<k`; increasing `c+t` | `CRX(-alpha)` |

Generate pairs target-first then control-first and use stable sorting. `CRX(alpha)` means controlled `Rx(pi*alpha)`. Exact **temporal** primitive sequences are

\[
CP(\alpha):
P_c(\alpha/2),P_t(\alpha/2),CX_{c,t},P_t(-\alpha/2),CX_{c,t},
\]
\[
CRX(\alpha):
H_t,P_t(\alpha/2),CX_{c,t},P_t(-\alpha/2),CX_{c,t},H_t.
\]

No global phase is dropped. Before exact simplification, there are `2k-1` CP and `2(k-1)^2` CRX gates, hence `4k^2-4k+2` CX for `k≥2`. Count the H and P gates too.

Use an independent Gray-parity template as a small reference. For `m=k+1` involved bits,

\[
\prod_{i=1}^{m}x_i
=2^{1-m}\sum_{\varnothing\ne S\subseteq[m]}
(-1)^{|S|-1}\bigoplus_{i\in S}x_i.
\tag{7}
\]

Applying each parity phase with coefficient `(-1)^(|S|-1)/2^(m-1)` gives controlled-Z; H on the target gives MCX. Compute and uncompute each parity with CXs. An efficient enumeration groups subsets by their highest-index bit and cycles through the lower bits in reflected Gray order, restoring the pivot after each group. This uses `2^m-1` phase gates and `2^m-2` CX before simplification. Check (7) by integer enumeration for every `m` used.

Before any evaluation, generate both templates for each control count 0–7, verify them, and freeze the lower-CX template, breaking ties by depth and total gates. This deterministic choice depends on control count, not on observed benchmark gains. The polynomial option prevents an exponential lowering rule from manufacturing an advantage for narrower registers.

The exact symbolic IR stores phase coefficients as numerator/denominator integers; never infer an exact angle by rounding an arbitrary numerical optimizer's output.

### 6.4 Parity, phase, and mapping traps to test

On `w≥4` wires, X, CX, and CCX induce respectively `2^(w-1)`, `2^(w-2)`, and `2^(w-3)` transpositions, all even. NCT-only circuits therefore cannot realize an odd full-basis permutation at that width. This is a gate-library restriction, not a defect of the capacity theorem. The chosen quantum primitive basis handles it without extra wires.

Required adversarial tests:

- One transposition on four wires: accepted by exact MCX/quantum lowering; an NCT-only parity check must reject it.
- Replace a correct gate by a relative-phase variant or insert a Z: a measurement-only check could pass, but the phase check must fail.
- Reverse bit order in an adapter: the asymmetric basis-state test must fail.
- Omit a slot rotation or apply a one-step embedding repeatedly to its used garbage: on the constant four-state map with `T=2`, the multi-step contract test must fail for some initialized input. Use this irreversible fixture; a reversible map would not detect the omission.
- Force every auxiliary bit back to zero after an irreversible step: the claimed implementation must fail.
- Insert a measurement, reset, discarded wire, or an undeclared workspace qubit: reject or count and reclassify it.
- Perturb an impossibility witness, a permutation entry, and a declared width: the respective independent verifiers must reject them.

Do not repair phase errors by comparing only magnitudes, ignoring input-dependent phases, or minimizing over a different global phase for each input column.

### 6.5 Verification layers and numerical conventions

1. **Mathematical layer:** integer certificates, independent direct-layer counts, tiny exhaustive search, analytic fixtures.
2. **Permutation layer:** exact bijection and all initialized trajectories, including `t=0` and `t=T`.
3. **X/MCX layer:** exhaustive integer evaluation on all `2^w` basis states, not only initialized states.
4. **Template layer:** independent complex128 matrix construction for every control count and primitive lowering used; compare all columns with the exact MCX permutation, including phase.
5. **Primitive whole-circuit layer:** independently simulate each retained headline fit, including the best circuit of each arm that is reported as fitting. Compare its full matrix with the intended permutation for `w≤8`. A candidate whose verification cannot finish remains unverified and cannot count as a fit.

Use a NumPy gate interpreter independent of TKET for the matrix checks. Convert rational angles to floating values only at this verification boundary. Include circuit global-phase metadata and explicit wire mappings. Compare `V` directly with the real permutation matrix `P`; do not quotient out an unrecorded phase.

Record maximum absolute entry error, maximum column 2-norm error, and the maximum absolute entry of `U.conj().T @ U - I` as the unitarity residual. Initial numerical acceptance limits are `1e-10` for entry error and `1e-9` for column error and unitarity residual. Calibration must show reference-template errors at least 100 times smaller. If it does not, investigate operation order, precision, and mapping. Do not relax these thresholds after observing which method wins. Any justified change before freezing must include an error analysis and all-arm revalidation.

Exactness is supplied by the integer circuit and symbolic decomposition identities. Floating-point matrix tests detect implementation errors; they do not prove literal exact equality by themselves and are not evidence that approximate dynamics are allowed. Preserve the exact symbolic circuit alongside any numerical toolkit export.

Full-matrix verification costs `O(G4^w)` operations in a direct implementation. Memory for one complex128 matrix is `16*4^w` bytes. Use row-wise one/two-qubit updates or column batches, not a dense `2^w × 2^w` matrix multiplication for every gate. Do not simulate `U^T` by multiplying dense matrices: the verified block plus exact trajectory check establishes repeated behavior compositionally.

## 7. Workloads, development, and locked evaluation

### 7.1 Deterministic workload definitions

For each `n=2^b`, define:

| Name | Exact function |
|---|---|
| `identity` | `f(x)=x` |
| `increment` | `f(x)=(x+1) mod n` |
| `countdown` | `f(x)=max(x-1,0)` |
| `shift` | `f(x)=x>>1` |
| `clear_low1` | `f(x)=x & (n-2)` |
| `constant` | `f(x)=0` |
| `branch_cycle` | `f(0)=1; f(1)=0; f(x)=x//2 for x≥2` |
| `split_cycle` | Let `m=n//2`. For `x<m`, `f(x)=max(x-1,0)`; otherwise `f(x)=m+((x-m+1) mod m)`. |
| `uniform_s` | For each x, hash the canonical string specified below, take its first eight digest bytes as a big-endian integer, and mask with `n-1`. |

Canonical hash generation, with no trailing newline:

~~~python
label = f"qbcert-v1|{split}|b={b}|seed={seed}|x={x}"
value = int.from_bytes(hashlib.sha256(label.encode("utf-8")).digest()[:8], "big")
f_x = value & (n - 1)
~~~

`branch_cycle` has merging branches feeding a two-cycle. `split_cycle` combines a countdown component and an isolated pure cycle. The hash-generated functions provide unfiltered pseudorandom tables, not samples selected by favorable leaf count or qubit savings. Do not regenerate a seed because a map is constant, reversible, duplicates another map, or is hard.

Every method receives the exact same generated integer table. Store that table and its SHA-256; do not rely on names or seeds alone. Deduplicate run instances by **(full function table, T, behavioral contract)**, retaining all provenance tags. Different horizons remain different runs but share one function-level statistical unit. Identical instances with two family tags cannot provide two independent wins or satisfy the two-family threshold by themselves.

### 7.2 Development and calibration data

Use structured `b=2,3` cases with horizons `T=1,3`, and `uniform_s` with `SPLIT=dev` and seeds 0 and 1. Calibration may also use `b=4,T=1` and the fixed four-wire transposition.

Do not inspect confirmatory circuit outcomes while tuning. Structured families are intentionally shared across splits; this tests new size/horizon instances, not generalization to unknown program families. Hash-generated development and confirmation tables are separate.

Freeze the following before generating confirmation outputs: code revision, dependency lock, seed rule, candidate scheduler, per-method caps, certificate verifier, numeric tolerances, template table, primary limits, instance manifest, exclusions, and analysis rules. Save a hashed `protocol_lock.json` with timestamp. A material correction after freezing makes affected results exploratory until rerun under a new lock; charge the rerun against the same cumulative budget.

### 7.3 Broad certificate matrix

Use `b=2,3,4,5,6`; all eight named structured/control functions above plus `uniform_s` with `SPLIT=confirm` and seeds 0 and 1; horizons

\[
T\in\operatorname{unique}\{1,2,b,2b,n,2n\}.
\]

This is at most **280 map–horizon cases before exact-table deduplication**. For each, compute both capacity formulas, the time-varying bound, analytic checks when available, a constructive realization, and certificates for auxiliary budgets immediately below/at/above `a_min` when nonnegative. Build sparse embeddings for large padded domains. The theorem says feasible capacities exist above the bound; the at/above cases must be witnessed, not mislabeled impossible because synthesis was not attempted.

This tier answers capacity and architecture questions. Its results do not establish feasible circuit cost at 64 visible states.

### 7.4 Locked compiled matrix

**Primary set: 30 case positions before deduplication**

- `b∈{2,3,4}`;
- `T∈{2,8}`;
- functions `countdown, shift, branch_cycle, split_cycle, uniform_s` with `SPLIT=confirm, seed=0`.

**Scaling set: four cases**

- `b∈{5,6}`, `T=2`;
- `countdown` and `shift`.

**Controls: two cases**

- `increment` at `b=4,T=8`;
- `constant` at `b=3,T=8`.

Total: **36 instances × three arms = at most 108 initial arm runs**. The constant control is deliberately impossible at eight total qubits: `M=57`, `a_min=6`, `b+a_min=9`. The reversible control needs no auxiliary qubits. An independent planning calculation found 15 distinct primary function tables and 28 of 30 primary positions abstractly feasible at eight wires; the exceptions are `b=4,T=8,shift` (10 wires) and `b=4,T=8,branch_cycle` (9 wires). Recompute this from the literal generator and certificates before execution; it is a workload check, not a compiler result.

Additional edge cases occur in development and the broad certificate tier; do not enlarge the locked matrix just to include every possible variant. The targeted paired replays in Section 8.2 add at most six arm runs and do not change the primary case set.

Keep scaling cases and controls separate from the 30-case primary endpoint. Horizons of the same function are related observations. Duplicate tables and conjugated encodings do not increase the count of independent mathematical functions.

### 7.5 Predeclared resource envelope and effect threshold

For the primary experiment, a circuit fits only if the complete forward execution satisfies all of:

| Resource | Limit |
|---|---:|
| Peak allocated logical qubits | **8** |
| CX gates over all T block applications | **20,000** |
| Total elementary gates over all T applications | **100,000** |
| Total unit-duration depth over all T applications | **30,000** |
| Arm compilation/search/verification wall time | **90 seconds** |
| Measurements, resets, discarded wires, fresh supplied qubits | **0** |

These are **benchmark decision thresholds**, chosen before comparative outputs, not calibrated claims about useful physical hardware. A real application may supply different limits only before a separate protocol lock.

One whole qubit is the smallest meaningful memory effect. The primary empirical endpoint is the paired difference in the number of independently verified fits, `fits(F)-fits(G)`, over all primary case positions, accompanied by exact counts of impossible, invalid, unverified, and resource-capped cases. Show the same difference on distinct tables and per family.

The predeclared threshold for a **promising conditional pipeline contribution** requires a net increase of at least two verified fits both over the full primary set and over its 20 eligible positions with `b≥3`. It also requires F-only fits on distinct function tables from at least two noncontrol families in that eligible subset, no unresolved correctness defects, and no loss concealed by selective reporting. All primary horizons already satisfy `T≥2`. Disclose every G-only fit and report net and gross wins. Deduplicate identical instances before counting; multiple family tags do not create extra wins. This threshold guards against a lone favorable toy and has no population-significance interpretation.

An F-only or G-only win requires both arms to have validly attempted the instance under the locked allowance, including legitimate timeouts or predeclared resource-cap terminations. A not-started run, external interruption, missing required backend or unresolved implementation defect cannot supply the losing side; it is missing or invalid comparative evidence. A confirmatory threshold verdict requires valid attempted outcomes for all three arms on every primary instance. Otherwise report the available paired observations and coverage, while leaving the primary threshold unresolved.

For each F-only fit, classify the evidence as lower measured width, lower gate/depth cost at the same width, certificate/search-time benefit, or unresolved cause; multiple mechanisms may apply. Fit counts alone establish no qubit saving. A qubit-reduction claim requires at least one fewer measured allocated wire than a valid, phase-checked comparison circuit under the same contract. A baseline timeout supplies no comparison width.

If F finds a fit while G times out, call it a **bounded-search advantage**, not proof of a better achievable qubit minimum. If an actual comparison circuit is over budget, report its measured resources; do not claim every possible baseline implementation is over budget. A robust positive pipeline conclusion also requires the predeclared paired replays in Section 8.2 to preserve their selected F-only outcomes. Application-level engineering success additionally requires independent architecture justification and surviving the stronger synthesis/architecture challenges below. It cannot be established in the currently evidence-free application setting merely by meeting this microbenchmark threshold.

## 8. Resource measurement and targeted challenges

### 8.1 Count the complete repeated computation

For a block with `G_CX` CX gates, `G_all` total gates, and depth `D`, report forward values `T*G_CX`, `T*G_all`, `T*D`. Block boundaries are explicit barriers: correctness is required after each block, so cross-boundary rewrites must not silently change the observable program. The primary schedule does not overlap successive block invocations.

Within a block, compute ASAP unit-duration depth from wire dependencies. Count CX depth separately. Count H, X, P, and any explicitly lowered SWAPs. Distinguish all allocated wires from wires merely touched by a gate; all `b` visible wires remain part of the contract even if idle. Spare declared wires count toward peak width.

Report the actual memory/time cost of table building, candidate generation, solver time, synthesis, lowering, verification, and reporting separately. The theorem's `O(nT)` evaluation is not the cost of compiling its permutation.

### 8.2 CPU measurements and reproducibility

Use one active benchmark worker and one solver thread. Set relevant BLAS thread counts to one before imports; record the actual BLAS backend. No CUDA, Metal, or GPU path is needed. If later acceleration is explored, keep it outside the primary comparison and verify its actual backend and precision.

Measure process wall time with a monotonic high-resolution clock. For each arm, include subprocess startup/import and any warmup in its applicable D-second cold end-to-end deadline—90 seconds for confirmation. Separately record warm algorithm time after import; any explicit warmup is measured and charged to that same deadline and the global ledger even when excluded from the warm algorithm interval. Do not mix a cold G measurement with a warm F measurement. Use an independent parent watchdog to enforce deadlines and sample aggregate parent-plus-child RSS at approximately 100 ms intervals. Record the platform convention for peak RSS; macOS and Linux units differ.

Gate counts and correctness are deterministic conditional on a circuit. Whether bounded search generates and verifies that circuit before its deadline can depend on execution timing. Candidate seeds remain fixed algorithm inputs. A runtime-speedup claim is optional: use at most two preselected development instances, three repetitions and three arms, with a 10-second cap per repetition (at most three minutes total), rotating arm order. Report every timing, median and range; capped repetitions support only censored-time statements. These repeats consume the 15-minute calibration allocation and do not establish robustness on confirmation cases.

For confirmation, use a deterministic seeded shuffle of instance order and rotate G/C/F order by instance index. Log power source, background load, observed thermal/memory-pressure conditions when available, and whether the laptop slept. Do not silently discard slower runs; a genuine interruption is a recorded invalid timing with at most one budgeted retry.

**Targeted paired replay:** if the initial primary comparison meets the threshold in Section 7.5, select the first eligible F-only position in locked manifest order and then the first later eligible F-only position with both a different function table and a different family. Repeat all three arms once on those two positions, with unchanged seeds and configuration and reversed arm order. Do not replace a selected case if its result fails to reproduce. The six additional runs cost at most nine minutes and remain within Stage 4's 180-minute allocation.

Keep the original outcome table unchanged and report replays separately, including time to first fit and margin to each deadline. Do not choose the better run, merge candidate pools, or reuse a successful circuit across runs. If either selected F-only outcome is absent on replay, report the positive effect as **timing-sensitive and not robustly confirmed**. If budget or interruption prevents the required replay, robustness remains unresolved. These two checks support only the selected effects, not reproducibility of every untested win.

The independent unit for descriptive cross-case summaries is the function table; horizons are a within-function factor. Report paired counts and per-family distributions. Do not produce p-values pretending these purposive synthetic workloads are a random sample of all quantum programs.

### 8.3 Stronger synthesis challenge

Do not launch a broad second compiler study. Within the 20-minute combined challenge allowance:

1. Select the first six apparent F-only primary positions in locked manifest order, or all if fewer exist. For each position, challenge exactly one **already generated** candidate from G and one from F using TKET `Matching` plus its documented exact-intent decomposition, with no measurement/reset assumptions and no implicit output permutations. Choose the candidate with the best completed primary exact circuit under the locked selection rule; if none has a completed circuit, choose the first generated valid permutation in that arm's locked queue. If none exists, record that arm's challenge as unavailable.
2. Allow at most 30 seconds per candidate for Matching synthesis, decomposition, resource measurement and phase verification together. At most `6*2*30=360` seconds of candidate work are authorized. Keep a total challenge ledger and do not transfer F candidates to G. Missing G candidates are missing robustness evidence, not a successful challenge.
3. Decompose all boxes and controlled operations before counting. Normalize the numerical counterpart to `X,H,CX,P(alpha)`, allowing real alpha here, using verified one-qubit identities and explicit output mapping. A general one-qubit U3/TK1 operation is not one primitive in this count. Include diagonal correction and global phase. Use the independent full-matrix comparison. Record these as **numerically validated toolkit circuits** unless their symbolic exactness is independently established. If numerical angles cannot be certified dyadic, label that basis relaxation; the result cannot satisfy the primary exact-basis endpoint, but can challenge a broad practical interpretation. An opaque box never counts as one elementary gate.
4. If the challenge supplies a plausible baseline fit, withdraw the broad practical-advantage interpretation even if the exact primary pipeline result remains true. If all apparent wins cannot be challenged within the cap, those wins have untested robustness.

This challenge addresses the known weakness of fixing an embedding and of comparing only two X/MCX synthesis strategies [R3]. It tests one completion per selected arm, not every embedding or compiler. A missing or broken mainstream backend leaves the robustness claim unresolved, not favorable by default.

If no F-only fit exists, run the challenge only on one fixed primary case per nonrandom structured family at `b=3,T=8`, subject to the same cap, to diagnose whether the exact backend itself explains general failure. Stop optional challenge work when its decision question is resolved.

### 8.4 Architecture challenge

Always report (5) for the broad matrix. For at most four primary cases, chosen before confirmation as `b=3,T=8` on `countdown, shift, branch_cycle, split_cycle`, construct and compile the time-varying permutation extensions and a final-only embedding using the common exact backend. Each alternative gets at most 60 seconds total, including construction, all its block circuits, measurement and verification. This is at most eight minutes of candidate work. Combined with Matching's six minutes, at least six minutes of the 20-minute challenge allocation remain for setup and records. Neither cap authorizes extra cases if the other finishes early.

Keep the visible encoding fixed. Count the sum of gate counts and depths of all varying blocks, and the width actually allocated. Final-only results go in a separate table because they do not provide intermediate observations.

If such a relaxation removes the memory bottleneck, state that the theorem certifies a cost of the imposed architecture. A favorable fixed-block comparison does not establish that this architecture should be selected. Do not treat external control as free in an application claim without the application's control-cost assumptions; in this diagnostic it is explicitly the changed model.

### 8.5 Cleanup accounting if a real use case requires it

The primary study retains garbage. For an optional coherent “compute, copy final classical label, uncompute” use case, allocate an additional `b`-qubit output register from the start, apply `U^T`, use `b` parallel CXs to copy the basis label, and apply `U^{-T}`. The original input and zeroed simulation auxiliaries remain, and the copied result can be entangled with the original input.

The complete resource accounting is:

\[
Q_{\rm peak}=2b+a+w,\quad
G_{CX}=2T\,G_{CX}(U)+b,\quad
G_{\rm all}=2T\,G_{\rm all}(U)+b,
\]

where `w` is any extra decomposition workspace. With block barriers, depth is `2T D(U)+1`. This is not a method for erasing the input. If this use case is selected, its new output wires and cleanup must be included in the budget before comparison. Do not bolt cleanup onto a completed experiment and retain its old “fits” classification.

## 9. Mac feasibility, dependencies, and hard ceilings

### 9.1 Inspect the actual target

Create `environment.json` using actual observations: `uname -m`, `sw_vers`, `sysctl -n hw.memsize`, CPU identification, Python version, available memory, free disk, installed package versions, NumPy configuration, and git revision. Verify native ARM64 execution rather than silently running an x86 environment through translation.

Do not infer RAM, storage, macOS release, or package compatibility from “M3 Max.” Current inspected metadata lists native ARM64 wheels for `pytket 2.18.5` requiring macOS 14 or later, while generic documentation states a broader OS range [R13]. Check the actual wheel and import on the target. The package versions and source snapshots researched on 1 November 2026 are evidence of an available route, not proof of installation.

Use a local virtual environment with Python 3.12 if available; compatible Python 3.10+ is acceptable if all required wheel/import checks pass. The core stack is:

- Python standard library, especially `fractions`, `hashlib`, `json`, `tomllib` or a compatible TOML reader;
- NumPy for independent complex128 verification;
- `z3-solver` for the generic baseline;
- `psutil` for process/memory observation;
- `pytket==2.18.5` for the Cycle adapter and bounded Matching challenge.

Resolve compatible versions of the other packages once, record them, and freeze the entire environment before confirmation. Do not depend on Qiskit at runtime; [R12] is an inspected algorithm/source reference for the locally implemented exact template. No cloud account, simulator service, or proprietary dataset is required.

Allow at most 60 minutes for setup/download troubleshooting, reported separately from experimental compute. Do not build TKET from source as a rescue project. If its wheel is unavailable, the owned exact compiler and certificate work can proceed, but the strong synthesis portfolio and mainstream challenge are incomplete; report that limitation and withhold the corresponding engineering claim. If Z3 is unavailable, tiny exhaustive checks can proceed, but no main G/C/F comparison is valid.

### 9.2 Memory, disk, and width limits

Set the experiment's aggregate RSS ceiling to

\[
R=\min(8\ \text{GiB},\,0.25\times\text{physical RAM},\,
0.5\times\text{available RAM at start}).
\]

Use one worker. If `R<2 GiB`, perform the certificate and tiny diagnostic stages only and report the main study resource-blocked. The parent watchdog stops the worker before the cap; a memory kill is a failure record, not a successful negative proof. Do not swap heavily to finish a run.

Require at least 4 GiB free local storage; cap project-created experiment data and temporary outputs at 2 GiB, with final retained evidence targeted below 500 MiB. Save compact circuit IR and compressed permutation arrays, not every dense unitary. Check available disk before each stage and reserve space for reporting.

All primary and challenge circuits are capped at eight total wires. A structured over-budget ring may have its width and compositional gate counts recorded, but is not authorized for a larger full-matrix simulation. No dense unitary wider than eight wires is authorized. This ceiling does not permit enlarging the experiment.

For scale: one complex128 matrix needs 1 MiB at eight wires, 16 MiB at ten, 256 MiB at twelve, and 64 GiB at sixteen. For constant reset on 64 visible states at `T=64`, `M_T=4033` and total width is 18; one full unitary would need 1 TiB. Its permutation table is far smaller. This derivation is why broad capacity coverage and compiled coverage are separate.

### 9.3 Cumulative time and work ceilings

**Hard experimental-compute wall budget: 360 minutes.** Run serially so this ledger is meaningful. Every development trial, failed attempt, tuning run, preprocessing step, repair rerun, repeat, challenge, and report-generation computation counts. Restarting the script, changing an algorithm, or taking a fallback does not reset it.

| Allocation | Maximum |
|---|---:|
| Stage 0: environment and constructive feasibility | 10 min |
| Stage 1: tiny correctness and certificate checks | 20 min |
| Stage 2: resource calibration | 15 min |
| Stage 3: mechanism pilot | 15 min |
| Stage 4: locked comparison, verification and selected paired replays | 180 min |
| Stage 5: combined synthesis and architecture challenges | 20 min |
| Stage 6: analysis, evidence preservation, report | 45 min |
| Bounded repair/rerun reserve across all stages | 30 min |
| Protected contingency, usable only for preservation/reporting | 25 min |
| **Total** | **360 min** |

The first informative pilot is targeted within 60 minutes of experimental execution. Stage 4's base maximum is `108*90=162` minutes; at most six replays add nine minutes, leaving nine minutes for broad certificates and bookkeeping. These allocations are planning ceilings, not measured M3 Max timings. Stage budgets may end early; unused experimental budget does not authorize extra favorable-search branches. Maintain a persistent ledger with start/end, stage, instance, arm, attempted operation, elapsed time, status, and cumulative total.

Do not start a new expensive job unless its maximum timeout plus the protected reporting reserve fits. At 290 cumulative experimental minutes, stop launching new comparisons; complete safe bounded verification and preserve evidence. Reserve the final 70 minutes for diagnosis/reporting/contingency. A run can therefore finish earlier than its nominal stage cap.

Estimated human-equivalent engineering effort, separate from experimental runtime: roughly 4–8 hours for the small implementation and tests, up to one hour for setup, and 1–2 hours for interpretation and writeup. Including nonoverlapped experiments, plan approximately 8–16 elapsed working hours. This is an unmeasured planning estimate, not a promise about hosted-agent speed. Do not spend days adding infrastructure to rescue a failed gate.

## 10. Evidence gates and autonomous decisions

Execute gates in order. Build only what the next gate needs. You may repair ordinary implementation problems within the shared 30-minute experimental rerun reserve and the implementation-effort estimate. At most two repair cycles are allowed for any material correctness defect; if it remains unresolved, stop the affected comparison and preserve an invalid/inconclusive outcome.

| Stage | Question and minimum work | Required evidence | Decision and next action |
|---|---|---|---|
| **0. Constructive feasibility** | Inspect hardware and inputs; write the capacity and direct-layer checkers; implement canonical abstract construction; verify a reversible map, countdown, constant map, and the supplied disconnected example. Record architecture evidence or its absence. | Environment/provenance files, exact fixture results, architecture status. | Continue if the contract and constructive implementation are coherent. A reproducible counterexample to an essential theorem triggers analytic stop. Missing application evidence limits claims but does not block the diagnostic. Missing target hardware blocks Mac claims. |
| **1. Tiny correctness** | Run the independent exhaustive oracle and Z3 cross-checks; verify binary padding; implement exact X/MCX replay, both synthesis routes, and primitive templates; run the deliberate faulty controls. | Tiny results, rejected mutations, template matrices/errors/counts, validated bit mapping. | All actual mismatches must be explained and fixed before comparison. A timeout is unknown. If the theorem fails independently, emit the minimal witness and stop. If only the compiler fails, preserve mathematical results and stop circuit claims after bounded repair. |
| **2. Calibration** | Measure table, CSP constraint construction and solving, synthesis, primitive generation, and whole-circuit verification on development representatives at widths 4, 6, and 8. Use an actual lowered development circuit of meaningful size, not only a single template. Establish that G solves nontrivial feasible development cases, including one eight-wire attempt. Measure RSS and disk. | Cost breakdown, baseline witnesses, estimated work at main caps, template table and fixed timeout settings; record every tried scheduler/timeout choice. | Keep the primary envelope unchanged. Lower the per-synthesis timeout from 10 to 5 seconds only if startup/verification still leaves useful work inside 90 seconds; apply to every arm before lock. If width-eight verification cannot fit 30 seconds or R, use the reduced diagnostic below. If G cannot produce any eight-wire development witness within its allowance after bounded repair, use that fallback and withhold the full primary claim. |
| **3. Pilot** | Six cases: `b=2,3`, `T=3`, functions countdown/shift/branch_cycle; three arms, at most 45 seconds per arm. Exercise every status and timing boundary. | Pilot fit/status table, evidence that G can solve feasible cases and that all verifiers operate. | Continue if the study can distinguish outcomes and baseline validity is established. Noisy timings or equal fit counts do not refute the hypothesis. If G never solves any feasible nontrivial development case, investigate the baseline; do not declare F superior to a broken implementation. Freeze after repairs. |
| **4. Locked comparison** | Run the fixed 36-case matrix, three arms, with the locked common pipeline and budgets. Run the broad certificate matrix within the allocation if not already completed; execute the selected paired replays if the primary threshold triggers them. | Append-only original runs and separate replays, accepted/rejected certificates, fit witnesses, failure records, actual resource use. | Complete the manifest within caps. Do not add cases or change limits after observing results. Preserve all incomplete entries as not run/unresolved. A material bug invalidates affected results; one bounded corrected rerun is allowed only under a new lock and the same cumulative budget. |
| **5. Diagnosis challenges** | Execute only the bounded checks in Sections 8.3–8.4 that still answer a live uncertainty. | Exact architecture comparison and labeled numerical toolkit challenge results. | Withdraw broad interpretation if a stronger baseline or relaxed architecture removes the apparent advantage. Missing challenge evidence narrows the claim; it does not count as passing. |
| **6. Report** | Recompute tables from raw files; review witnesses, overhead, exceptions, and coverage; assign outcome categories. | Reproducible report, concise usage, manifests, code, essential data, or analytic stop evidence. | Finish with a scoped conclusion and at most one justified next step. A supported negative or early stop is a completed result. |

**Reduced diagnostic fallback:** if the main resource/calibration gate fails, retain all broad capacity cases and run at most the 12 structured primary positions with `b=2,3`, functions countdown/shift/branch_cycle, and `T=2,8`, subject to the same eight-qubit and scientific correctness contract. Give each arm at most 30 seconds. Label this a limited diagnostic, not the full confirmatory comparison, and do not use its selected subset to claim the primary improvement threshold. If even it cannot provide trustworthy phase-checked witnesses, stop after certificates and measured scaling evidence. One fallback only; no further narrowing campaign.

A mathematically valid construction that produces expensive circuits is a meaningful result. A defective baseline, unverified phase behavior, or missing essential backend is not a negative result about the theorem's engineering value.

## 11. Implementation handoff

### 11.1 Minimal repository

~~~text
qubit_budget_poc/
  README.md
  pyproject.toml
  requirements.lock
  config/study.toml
  inputs/
  src/qbcert/
    __init__.py
    cli.py
    model.py
    capacity.py
    certificate_check.py
    construct.py
    generic.py
    synth.py
    exact_gates.py
    verify.py
    resources.py
    runner.py
    report.py
  tests/
    test_capacity.py
    test_oracle.py
    test_circuits.py
  results/
    environment.json
    source_manifest.json
    protocol_lock.json
    budget_ledger.jsonl
    instances.jsonl
    runs.jsonl
    certificates/
    circuits/
    tables/
    REPORT.md
  THIRD_PARTY_NOTICES.md
~~~

This is an intended structure, not an existing repository. Combine small modules if doing so improves clarity; preserve independence between the producer and checker.

Responsibilities:

- `model` validates explicit instances and deterministic generation; declares bit order and hashes.
- `capacity` implements only the fast producer and derived resource summaries.
- `certificate_check` implements direct witness/trajectory checking without importing the producer.
- `construct` builds the paper family, one-step embeddings, ring, and varying-block alternatives.
- `generic` implements Z3 and a separate tiny brute-force oracle.
- `synth` implements transformation synthesis and the TKET adapter.
- `exact_gates` owns rational-angle IR, verified template expansion and safe rewrites.
- `verify` owns independent integer/matrix checks.
- `resources/runner` enforce the budget and write records even on failure.
- `report` derives all analysis tables from stored measurements.

Use git locally for code/version provenance. No remote publication or pull request is required.

### 11.2 Core interfaces and records

Implement equivalent typed interfaces:

~~~python
capacity_profile(f: tuple[int, ...], T: int) -> tuple[int, ...]
make_impossibility_certificate(instance, a: int) -> dict | None
check_impossibility_certificate(cert: dict) -> CheckResult
construct_permutation(instance, a: int, variant_seed: int) -> Permutation
solve_trajectories(instance, a: int, deadline, seed: int) -> SearchResult
complete_partial_bijection(edges, domain_size, policy, seed) -> Permutation
verify_permutation(instance, permutation, a: int) -> CheckResult
synthesize(permutation, width: int, strategy: str, deadline) -> ExactCircuit
lower_exact(circuit, template_table) -> ExactCircuit
verify_compiled(permutation, exact_circuit, numeric_policy) -> CheckResult
measure_circuit(exact_circuit, T: int) -> ResourceRecord
run_arm(instance, arm, locked_config, shared_budget) -> RunRecord
~~~

`CheckResult` must include pass/fail/unknown, diagnostic, checked domain, and errors if numerical. `SearchResult` distinguishes SAT, UNSAT, UNKNOWN, exception, and killed. Z3's UNSAT is recorded as solver evidence; final impossible-budget classification still requires the independent mathematical certificate.

Each run record contains instance digest, function/family tags, b/n/T, arm, candidate provenance, proposed persistent a, actual peak Q, q-profile/M/a_min, relaxed bound, source/dependency/config hashes, all stage timings, RSS, primitive counts, block and total depth, verification tier/errors, outcome, and timeout/failure reason. A failed or killed run has a record even without a circuit.

Use a canonical JSON representation for hashes (sorted keys, fixed separators, UTF-8) and store explicit tables. Store phase fractions as integer pairs. A floating circuit export is supplementary.

### 11.3 Commands to implement and verify

The following are **intended interfaces you must create**. They were not executed during planning and must not be described as existing working commands until you implement them.

~~~bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip freeze --exclude-editable > requirements.lock

python -m qbcert.cli inspect --config config/study.toml
python -m qbcert.cli check --suite tiny --config config/study.toml
python -m qbcert.cli smoke --config config/study.toml
python -m qbcert.cli calibrate --config config/study.toml
python -m qbcert.cli pilot --config config/study.toml
python -m qbcert.cli freeze --config config/study.toml
python -m qbcert.cli run --stage confirm --lock results/protocol_lock.json
python -m qbcert.cli run --stage challenge --lock results/protocol_lock.json
python -m qbcert.cli report --results results
python -m qbcert.cli verify-artifacts --results results
~~~

The CLI must refuse confirmation without a matching protocol lock, refuse an already exhausted budget, and resume a stopped run without overwriting existing evidence or restarting its cumulative timer. An explicit `--exploratory` flag may permit a changed configuration, but it must change the label and cannot restore confirmatory status.

In `pyproject.toml`, pin TKET as specified and declare the other necessary packages. Use `tomli` only for Python versions lacking `tomllib`. Record installation resolution failures. For reproduction in a fresh environment, install `requirements.lock` with `python -m pip install -r requirements.lock`, then install this project with `python -m pip install --no-deps -e .`. Excluding the editable project from the dependency lock prevents an absolute local source path from making the lock nonportable. The source paper and JSON are inputs; generated results must never overwrite them.

## 12. Analysis and outcome interpretation

### 12.1 Required analysis tables

Produce these from raw records, not hand-entered summaries:

1. **Certificate coverage:** cases/unique tables, exact profile agreement, witness acceptance, construction acceptance, and all unknown/missing cases.
2. **Architecture capacities:** `N_min, M_T, a_min, m_T, a_vary, Delta a` by instance/family/horizon. Include plateau examples where capacity differences save no qubit.
3. **Primary paired outcomes:** G/C/F fit and status for every case; net and gross wins on the full and eligible sets; distinct-table and per-family summaries; controls and scaling cases separate. Show the selected replays in a separate table without replacing initial outcomes. Classify each apparent improvement by width, gate/depth, search-time or unresolved mechanism; report actual qubit-saving witnesses separately from fit-count changes.
4. **Resource frontier:** actual allocated width, CX/total gates/depth, ratio to the abstract minimum, and time to verified fit. Preserve over-budget valid circuits.
5. **Costs and failures:** preprocessing and verification costs, timeouts, errors, invalid circuits, peak RSS/disk, and cumulative budget.
6. **Challenges:** exact relaxed-architecture results and separately labeled numerical toolkit results, including whether an apparent advantage survived.

One or two static plots can help: width versus CX/depth Pareto points and capacity/qubit counts versus horizon. Use standard plotting only if it clarifies a decision. No chart should omit failures or make a capacity reduction look like a whole-qubit saving.

Do not calculate gate/runtime ratios only on a favorable intersection and call them general performance. If a conditional ratio is useful, explicitly state its paired-success denominator and report failure rates beside it. Do not impute a timeout as a measured maximum cost or as infinity in a claim of mathematical superiority.

### 12.2 Outcome categories

The final report may assign different categories to mathematical certification and engineering utility.

| Outcome | Justified interpretation |
|---|---|
| **Supported contribution** | Independently checked certificates and a scoped empirical effect meeting the locked criterion against a functioning independently implemented baseline with the specified search allowance; disclose required replay and challenge results. State the exact pipeline, workload, and architecture condition. Historical novelty remains separate. |
| **Informative negative result** | Valid construction and adequate comparison show no specified useful effect under the tested conditions; or a verified mathematical obstruction settles a proposed budget. Report whether a stronger compiler or relaxed architecture removes the opportunity. |
| **Practical infeasibility** | Measured cost or a concrete scaling bound prevents the intended study/application inside its resource envelope. Identify the dominant stage, not merely “too slow.” |
| **Inconclusive evidence** | Missing hardware/source/backend, insufficient coverage, timeouts, or incomplete phase verification prevents the relevant decision. |
| **Invalid experiment** | A method, baseline, accounting, data-generation, or verification defect invalidates comparative results. Preserve them as invalid; do not reinterpret them as unfavorable science. |

If the only result is successful reproduction of known one-step embedding behavior, say so. If the source theorem survives but offers no useful compiler improvement, that is an allowed negative engineering result. If only G fails to search in time, report a bounded-search observation. If F performs worse because fixed padding destroys synthesis freedom, that is a meaningful limitation of the chosen constructive family.

For a theorem contradiction, minimize the explicit f/T/capacity counterexample, show the full alleged permutation or violated proof assumption, and verify it through an independent direct checker. Stop the main benchmark until resolved. Do not “fix” the theorem by silently excluding the troublesome inputs.

### 12.3 Required final artifacts

For the stages actually reached, deliver:

- runnable code, exact setup and CLI instructions, dependency lock, and third-party attributions;
- input/function/source/config identifiers and target environment report;
- exact witnesses, circuit IR and checks for reported fits;
- raw successful, unsuccessful, killed and invalid run records;
- regenerated essential tables and optional plots;
- `results/REPORT.md`, explaining the decision, evidence, deviations, limitations, and whether one specific next step is worthwhile;
- a concise reproduction command for the essential conclusion.

For an analytic early stop, a minimal reproducer or independently checkable derivation/counterexample is sufficient; do not build a benchmark framework to fill an inapplicable deliverable list. For unavailable Mac hardware, complete unaffected implementation and mathematical work, clearly identify blocked Mac measurements, and never label another machine's timing as M3 Max performance.

Execute these stages autonomously. Repair ordinary reversible implementation issues inside the bounds. Ask for input only for an indispensable missing resource or a real scientific choice not covered here. Do not publish, contact outside people, purchase compute, or broaden the experiment without authorization.

## 13. Research basis and claim-to-evidence map

### 13.1 Closest work and what this study changes

The opportunity is narrower than a new reversible compiler. Minimum one-step garbage is established [R1]. Exact/succinct reversible embeddings and the effects of fixing garbage assignments before synthesis are also established [R2–R3]. The finite-horizon source specializes a familiar dilation idea to exact per-visible-state capacities. Set-theoretic defect constructions [R5], finite operator dilations [R6], and autonomous Markov dilations [R7] are close conceptual antecedents.

The differing requirement tested here is simultaneous exact correctness of every initialized trajectory at every required time under **one permutation**, combined with actual binary product-register encoding and measured compilation costs. The source's closed capacity formula is the mechanism; this study's candidate contribution is evidence about its resource-decision value and limitations.

The finite-operator formula `n+Nd` in [R6] is an analogy, not a direct proof: a merging deterministic transition matrix is generally not a contraction. The hardness results in [R2] concern succinct formula/BDD input representations, not explicit 4–64-state tables. Do not borrow either claim outside its assumptions.

Sequential reversible synthesis already studies state encodings and next-state embeddings [R9], but external control, constant inputs, output garbage and mutable internal encodings differ from this closed finite-horizon contract. Reqomp [R14] and SyReC [R15] establish the broader importance of width/gate trade-offs. They do not license uncomputing the persistent distinctions that make an irreversible visible update reversible. No incompatible published benchmark numbers are used as this study's baseline.

### 13.2 Central claims and resolving checks

| Claim | Evidence status | Remaining uncertainty / resolving check |
|---|---|---|
| Exact capacities are (1)–(3). | Supplied theorem; proof reviewed; independent tiny checks support it. | Your independent producer/checker/oracle tests; a mismatch stops comparison. |
| `ceil(log2 M_T)` is the binary auxiliary minimum. | Source corollary plus power-of-two padding derivation. | Explicit blank-label/bit-order and padding validation. |
| General unitary hidden states do not reduce that exact dimension. | [S1, Theorem 4]. | No claim about noisy or arbitrary-channel simulation. |
| Capacity computation is cheap for the chosen input representation. | Direct `O(nT)` algorithm. | Measure actual preprocessing and storage; do not transfer to succinct inputs. |
| Minimal-total construction is a complete circuit-optimization search space. | **False as a general inference.** | Compare with unconstrained partial-bijection completion; never claim optimal gate/depth synthesis. |
| Certification can improve bounded compilation. | Hypothesis. | G/C/F comparison with equal allowances and exact witnesses. |
| Smaller hidden capacity saves qubits or gates. | Not automatic. | Binary rounding and actual decomposed resource measurements. |
| Fixed-block memory can exceed varying-block memory. | Derivation (5), analytic fixtures, tiny constructive checks. | Exact architecture table plus limited compiled challenge. |
| The target is useful for a deployed application. | Unverified. | Independent application contract; currently conditional. |
| The theorem/formula is historically new. | Unresolved. | Specialist literature review beyond this bounded study; do not claim priority. |
| Work fits the user's M3 Max. | Planning hypothesis; no Mac measurement yet. | Environment inspection and calibration, then hard caps. |

### 13.3 Search coverage and limits

Planning research on **1 November 2026** used two web search systems, primary papers, author-hosted manuscripts and official software documentation/source. Searches covered finite/set-theoretic/permutation dilations, defect sets, autonomous invertible dynamics, channel powers, finite unitary dilations, reversible simulation, minimum garbage, explicit and succinct embeddings, sequential reversible synthesis, uncomputation, MCX decomposition, and phase/register semantics. Backward references were followed where they affected the model; the current Reqomp publication page's cited-by entries and current official compiler APIs were checked for adjacent developments.

No exact duplicate of the supplied finite-horizon fiber formula was found in that targeted coverage. This is not an exhaustive priority search. Monographs, nonindexed and unpublished work, and comprehensive forward-citation databases were not exhausted. Cycle-rook literature was not independently revalidated because the cycle-enumeration results are excluded. The source note itself already limits its novelty claim.

Some direct source routes failed; accessible author/arXiv copies or downloaded public source snapshots were inspected instead. A source's existence, source-code inspection, package wheel availability and successful execution are separate evidence levels. Do not convert any of them into a performance claim.

## 14. Review findings and residual risks

Three distinct review passes are required before you finalize your report: mathematics/novelty, experiments, and execution. Separate reviewers performed those passes on this planning brief, and the planner reconciled their findings. The brief incorporates the following material findings; repeat affected checks after implementation changes.

| Review finding or stress scenario | Disposition in this design |
|---|---|
| A theorem might be nonconstructive or rely on an inaccessible oracle. | The exact predecessor/chain construction and `O(nT)` profile algorithm are explicit. The independent CSP checks feasibility without using the construction. |
| Total hidden-state count was mistaken for product-register width. | Separate `N_min`, `M_T` and binary rounding; largest fiber controls qubits. |
| The attachment's reported computation cannot be reproduced from its JSON alone. | Treat it as source-reported; fresh independent exhaustive checks are specified and a separate planning probe was performed. |
| The mechanism may be equivalent to prior work. | One-step and general dilation antecedents are credited; priority and publication novelty are not asserted. |
| Only constant reset or favorable encoding gives a win. | Primary nonconstant transient/cyclic/disconnected/random families; fixed visible encoding; constant reset excluded from the primary success endpoint. |
| A weak history baseline manufactures a gain. | Exact direct-contract CSP and equal synthesis opportunity; ring is reference only; certificate-only ablation isolates pruning. |
| A better completion or tuned baseline erases improvement. | Generic completion explores beyond padded minima; fixed portfolio and Matching challenge; withdraw broad interpretation if gain disappears. |
| An exponential MCX backend artificially rewards narrow circuits. | Polynomial exact four-sweep template plus independently verified small templates. |
| Floating-point agreement is confused with exact mathematics. | Exact rational IR and structural identities; numeric matrices are separate implementation checks with fixed tolerances. |
| Correct probabilities conceal phases or wire swaps. | Full-basis integer checks, phase-sensitive matrix checks, negative controls and explicit mapping restoration. |
| Extra temporary qubits evade budget accounting. | Actual allocated width is counted; primary lowering has no extra workspace; boundary persistent capacity is reported separately. |
| Preprocessing dominates or synthesis scales badly. | End-to-end timing includes setup/preprocessing/verification, explicit total-width caps and all failures; no universal scaling extrapolation. |
| A pilot is uninformative. | A valid pilot can continue to the locked matrix; baseline failure triggers repair/inconclusive, not automatic superiority or refutation. |
| Execution rules mixed finite sets with qubit inputs, or left no time to verify a candidate. | Separate abstract/compiled validators; preserve horizons when deduplicating; charge warmups; cap timing repeats; reserve phase-verification time inside each arm's deadline. |
| A fit improvement was mistaken for a memory reduction. | Classify width, gate/depth and search-time mechanisms separately; require a valid measured comparison circuit for a whole-qubit saving. |
| Deadline variation or queue arrival creates a fragile apparent gain. | Deterministic source/portfolio schedule, all preparation charged, verified witnesses saved before cutoff, and two predeclared paired replays without replacement or best-run selection. |
| Missing or invalid baseline evidence is counted as a losing comparison. | Only valid attempted pairs supply wins; missing primary outcomes leave the confirmatory threshold unresolved. Phase/mapping defects trigger the bounded repair and affected-run invalidation rules. |
| Robustness checks silently become a larger favorable-search campaign. | One already-generated G and F candidate per selected Matching case, explicit per-alternative architecture caps, and one shared 20-minute allowance. |
| The essential application permits varying blocks or final-only output. | Exact relaxed capacity comparison and separate architecture challenge; conditional application claim. |
| A required source/backend/hardware is unavailable. | Self-contained mathematics and owned exact routes preserve unaffected work; missing strong baseline or Mac timing blocks corresponding claims. |
| The main hypothesis is false. | Informative negative result with verified circuits and fair comparisons is an authorized completion; no search-until-success behavior. |

Residual risks remain: the generic CSP is a bounded heuristic search at practical sizes; state encoding and completion affect gate cost; even the common synthesis portfolio is not globally optimal; synthetic inputs do not establish demand; and real applications may need locality, fault tolerance, different cleanup or a different architecture. These limit conclusions rather than justify altering the experiment after seeing its results.

## References

All web sources below were consulted during planning on 1 November 2026. Use stable links and exact versions/commits in your source manifest. The supplied note and JSON have the hashes in Section 2.1.

**[S1]** *Exact finite reversible simulation: State capacities, optimal constructions, and cycle constraints*. Research note, 1 November 2026; PDF metadata author: “Research investigation prepared with ChatGPT.” Supplied `finite_reversible_simulation.pdf`, expected handoff location `inputs/finite_reversible_simulation.pdf`. Critical locations: Definition 1/equation (1), p. 1; Theorem 2/equations (3)–(5), p. 2; Section 3.2 construction and Section 3.4/equation (6), pp. 2–3; Corollary 3/equation (7), p. 3; Theorem 4, pp. 3–4; Theorem 5, p. 4; examples Section 7, pp. 6–7; reported computation Section 8, p. 7; novelty/limitations Sections 9–10, pp. 7–9. Not identified as peer reviewed.

**[S2]** *verification_results.json*. Supplied aggregate verification output accompanying [S1], 1 November 2026; expected handoff location `inputs/verification_results.json`. Generating program not received. Source reports finite checks, not a proof.

**[R1]** Dmitri Maslov and Gerhard W. Dueck. *Garbage in Reversible Designs of Multiple Output Functions*. 6th International Symposium on Representations and Methodology of Future Computing Technologies, March 2003. [Author-hosted manuscript](https://www.cs.unb.ca/~gdueck/reversible/garbage.pdf). Theorem 1, PDF p. 3: maximum-output-multiplicity garbage bound.

**[R2]** Mathias Soeken, Robert Wille, Oliver Keszocze, D. Michael Miller and Rolf Drechsler. *Embedding of Large Boolean Functions for Reversible Logic*. Published online 9 December 2015; ACM Journal on Emerging Technologies in Computing Systems 12(4), article 41, pp. 1–26, July 2016 issue. [arXiv:1408.3586](https://arxiv.org/abs/1408.3586), inspected 2014 version; [DOI:10.1145/2786982](https://doi.org/10.1145/2786982). Section II-A, arXiv PDF p. 2; Section IV, Propositions 2–3, pp. 4–5; Sections V–VI algorithms.

**[R3]** Alwin Zulehner and Robert Wille. *Skipping Embedding in the Design of Reversible Circuits*. ISMVL 2017, pp. 173–178. [DOI:10.1109/ISMVL.2017.19](https://doi.org/10.1109/ISMVL.2017.19); [author-affiliated PDF](https://www.cda.cit.tum.de/files/eda/2017_ismvl_main.pdf). Section I and Figure 1, p. 1: pre-embedding fixes synthesis freedoms and expands tables.

**[R4]** Francesco Buscemi, G. Mauro D'Ariano and Massimiliano F. Sacchi. *Physical realizations of quantum operations*. Physical Review A 68, 042113 (2003). [DOI:10.1103/PhysRevA.68.042113](https://doi.org/10.1103/PhysRevA.68.042113); [arXiv:quant-ph/0305180](https://arxiv.org/abs/quant-ph/0305180). Section III.C, equations (39)–(43), PDF pp. 5–6: power interacting dilations using preallocated ancilla copies and cyclic shifts. Its full-channel contract differs from this study.

**[R5]** B. V. Rajarama Bhat, Sandipan De and Narayan Rakshit. *A caricature of dilation theory*. 2020. [arXiv:2004.09255](https://arxiv.org/abs/2004.09255). Definition 1.6, p. 3; Definition 1.8 and following construction, p. 4; Theorem 1.11, pp. 4–5. Infinite defect rays and orbit-generated minimality.

**[R6]** Eliahu Levy and Orr Moshe Shalit. *Dilation theory in finite dimensions: the possible, the impossible and the unknown*. Rocky Mountain Journal of Mathematics 44 (2014), 203–221; inspected arXiv v2, 15 July 2011. [arXiv:1012.4514](https://arxiv.org/abs/1012.4514). Definition 1.2/Theorem 1.3, p. 2; Corollary 2.2, p. 4.

**[R7]** Matteo Gregoratti. *Classical dilations à la Quantum Probability of Markov evolutions in discrete time*. arXiv 2007; Theory of Probability and Its Applications publication, 2009. [arXiv:math/0702690](https://arxiv.org/abs/math/0702690); [DOI:10.1137/S0040585X97983997](https://doi.org/10.1137/S0040585X97983997). Standard dilation, PDF pp. 6–7; Theorem 1 and nonminimality observation, p. 8; cocycle alternative, pp. 8–9.

**[R8]** D. Michael Miller, Dmitri Maslov and Gerhard W. Dueck. *A Transformation Based Algorithm for Reversible Logic Synthesis*. DAC 2003, pp. 318–323. [DOI:10.1145/775832.775915](https://doi.org/10.1145/775832.775915); [author-hosted PDF](https://webhome.cs.uvic.ca/~mmiller/publications/DAC03.pdf). Section 3 transformation algorithm; Section 4 template simplification. The simple reference in this brief is an explicitly stated variant, not a claim to reproduce every optimization in that paper.

**[R9]** Mathias Soeken, Robert Wille, Christian Otterstedt and Rolf Drechsler. *A Synthesis Flow for Sequential Reversible Circuits*. ISMVL 2012. [DOI:10.1109/ISMVL.2012.72](https://doi.org/10.1109/ISMVL.2012.72); [author-hosted PDF](https://msoeken.github.io/papers/2012_ismvl_1.pdf). Section III external control/feedback; Section IV.C embedding; Section V state encodings.

**[R10]** Quantinuum TKET development team. *pytket Circuit Construction and API documentation*, inspected documentation for 2.18.5, 2026. [Circuit Construction](https://docs.quantinuum.com/tket/user-guide/manual/manual_circuit.html), sections on global phase, ToffoliBox, serialization and implicit qubit permutations; [circuit API](https://docs.quantinuum.com/tket/api-docs/circuit.html), `ToffoliBoxSynthStrat`. [Inspected ToffoliBox.cpp source snapshot](https://github.com/Quantinuum/tket/blob/dbf6e1a49dcb96863bbd7c877d625bc8998a5d83/tket/src/Circuit/ToffoliBox.cpp): `generate_circuit`, `get_bitstring_circuit`, `gen_circuit_using_toffoli_gates`. The snapshot was inspected; confirm the installed adapter's output rather than assuming the wheel is that commit.

**[R11]** Adenilton J. da Silva and Daniel K. Park. *Linear-depth quantum circuits for multiqubit controlled gates*. Physical Review A 106, 042602 (4 October 2022). [DOI:10.1103/PhysRevA.106.042602](https://doi.org/10.1103/PhysRevA.106.042602); [arXiv:2203.11882](https://arxiv.org/abs/2203.11882). Theorem 1/equation (3), pp. 1–2; Theorem 2 and Lemma 1, pp. 2–3.

**[R12]** Qiskit contributors. *Multi-controlled gate synthesis implementation*, inspected source snapshot, 2026. [mcx.rs at commit ae0faebd701f7ceb1b77d814079d2409311dabcd](https://github.com/Qiskit/qiskit/blob/ae0faebd701f7ceb1b77d814079d2409311dabcd/crates/synthesis/src/multi_controlled/mcx.rs). `synth_mcp_noaux_sp22` and module `sp22`, particularly `step_1` through `step_4`. Credits [R11] and QCLib. [Official synthesis API](https://quantum.cloud.ibm.com/docs/en/api/qiskit/synthesis) also documents newer no-auxiliary MCX routes; wire-permutation routines there are not arbitrary basis-permutation synthesis.

**[R13]** Quantinuum TKET development team. *pytket 2.18.5 release metadata*, released 8 October 2026. [PyPI release](https://pypi.org/project/pytket/2.18.5/); [Getting Started](https://docs.quantinuum.com/tket/api-docs/getting_started.html). Inspected ARM64 macOS 14+ wheel metadata; runtime compatibility remains to be checked on the actual Mac.

**[R14]** Anouk Paradis, Benjamin Bichsel and Martin Vechev. *Reqomp: Space-constrained Uncomputation for Quantum Circuits*. Quantum 8, 1258 (19 February 2024). [DOI:10.22331/q-2024-02-19-1258](https://doi.org/10.22331/q-2024-02-19-1258); [full paper](https://quantum-journal.org/papers/q-2024-02-19-1258/pdf/); [official implementation](https://github.com/eth-sri/Reqomp). Definition 2.1; Section 4 strategies/limitations; Section 6 evaluation. Motivation and related work, not a matched repeated-block baseline.

**[R15]** Munich Quantum Toolkit contributors. *MQT SyReC Synthesizer: Description and Features*, current documentation inspected 1 November 2026. [Official documentation](https://mqt.readthedocs.io/projects/syrec/en/latest/description_and_features.html). Cost-aware and line-aware synthesis descriptions establish that width/gate trade-offs are existing compiler objectives.

---

**Execution directive:** preserve correctness and the study's ability to reach an unfavorable conclusion. Finish the authorized evidence gates that remain feasible, retain failures, and report exactly what the evidence settles.

