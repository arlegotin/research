# Exact context laws for partition-valued quota circuits

## Research assessment and executable proof-of-concept plan

**Research date:** 30 October 2026.  
**Target:** a MacBook Pro with an M3 Max; actual memory, storage, and macOS version must be discovered.  
**Deliverable type:** a standalone mathematical research and experimental-design handoff, not a trained model or a claim of an established practical speedup.  
**Primary direction:** prove, independently verify, and document an exact depth/arity/quota law for retained-object witnesses when raw partition blocks contain at most three objects.

### Reading and execution guide

Read Sections 1–4 before implementing anything. The equations and constructor contracts in Section 4 are the scientific specification. Section 5 defines what counts as evidence; Section 7 gives the execution order. Commands labelled **proposed** refer to modules the coding agent must create. Only the four reproduction commands explicitly identified in Section 7 already exist in the supplied archive. The core project requires no external dataset, GPU, model weights, paid API, or second computer.

The original attachments are useful replication assets, but the selected mechanism, proofs, construction rules, interfaces, and evaluation protocol are specified here. If the archive is unavailable to the executing agent, reproduce the mathematical fixtures from this document and mark the original-bundle replication step unavailable. Do not silently claim that its saved results were reproduced.

## 1. Executive assessment and recommended direction

The supplied idea is a discrete symbolic computation model: a signal is an equivalence relation, a gate accepts object pairs supported by a quota of its inputs, and transitive closure turns the accepted graph into a partition. The strongest part of the supplied work is an exact theorem about the number of objects needed to preserve an output relationship after deleting objects from **all raw inputs and recomputing the circuit**. It is not evidence for a learning architecture, numerical efficiency, or an alternative to language models.

**Recommendation:** pursue a theorem-led PoC, centered on the exact witness law for **all homogeneous quota gates with raw blocks of size at most three**. This preserves the original computational primitive and its context-dependence question. It explicitly extends the supplied three-of-four theorem rather than renaming it. Use a small, independently checked circuit evaluator and witness analyzer as experimental infrastructure. Keep a solver optional.

The investigation produced five consequential findings:

1. **The supplied three-of-four proof survived the present review, and its deterministic computational reports reproduced.** For positive depths `d≥1`, the exact pair-witness bound is 2 for raw blocks of size at most two, `(4**d + 14) / 6` for raw blocks of size at most three, and unbounded at depth one when size-four raw blocks are allowed. The finite checks support the code; the proof, not those checks, supports the universal statement. [A1]
2. **The operators themselves have substantial prior art.** Lattice federation functions, partition quota rules, equivalence aggregation, and nonnumeric lattice networks already exist. The exact object-restriction/depth invariant is the candidate contribution. [R1–R7]
3. **A broader exact recurrence can be derived.** For an `m`-input gate accepting at least `q` inputs, with `q < m` and `3*q > 2*m`, define `s=m-q`, `h=2*q-m`, and start `p=t=0`. At each depth, update `t_new=h*t+2*s*p` and `p_new=max(q*p, 1+t_new)`. Then the exact worst pair-witness size is `P=2+p` and the exact worst triple-witness size is `Q=3+t`. Section 4 gives both the upper proof and matching constructions. This is a new deduction in this investigation; historical originality remains unresolved.
4. **The most tempting software novelty did not survive a fair baseline.** A specialized two-hop provenance formula for size-three blocks is identical to a generic reachability compiler after the same full-domain block pruning. A small implementation probe found identical normalized formulas for all 3,665 tested related pair events. Generic provenance circuits and solver-based smallest-subset extraction also predate this project. Do not claim a new compiler, a cubic-to-linear competitive advantage, or a solver breakthrough from this specialization. [R8–R11]
5. **Small intermediate blocks do not make global minimum-witness search easy.** A Set Cover reduction gives NP-completeness for the unrestricted-depth three-of-four problem. Conversely, the depth bound gives finite-size certificates at fixed depth. These are compatible statements about different parameters. The reduction is included here and was checked on small instances; its novelty is not asserted.

The intended advance is therefore an **exact mathematical characterization**, with reproducible constructive verification and honest resource accounting. It improves the scope and precision of the supplied result; it does not demonstrate better prediction or a practical application advantage. If the general quota law is already in the literature, the correct outcome is a replication and synthesis, followed by the narrowly specified extension gate in Section 10.

### What would make this worthwhile?

For researchers studying relational computation, the law distinguishes local output block size from the amount of underlying context a computation can depend on. It also exposes a pair/triple transition that the original symmetric case hides. For developers of exact symbolic systems, it supplies adversarial fixtures, correct witness semantics, and limits on what an explanation extractor can promise. Demand for a deployed application remains an assumption; there is no evidence here that this gate is preferable to existing clustering or inference systems.

The strongest case against the project is that the useful mathematical result may already be known under lattice-consensus or relation-identity terminology, while the implementation reduces to standard provenance and optimization. That possibility is a legitimate stopping outcome, not a reason to manufacture an empirical win.

## 2. Problem, scope, assumptions, and hypotheses

### 2.1 Objects, partitions, gates, and circuit depth

Let `V` be a finite set of objects. A partition is identified with an equivalence relation `E` on `V`. Write `Eq_≤b(V)` for partitions whose individual blocks have cardinality at most `b`; this does **not** bound the number of blocks or total storage.

Meet is intersection of equivalence relations. Join is the equivalence closure of their union. For a nonempty antichain `H` of nonempty subsets of input positions `[m]`, define

\[
F_H(E_1,\ldots,E_m)=\bigvee_{S\in H}\bigwedge_{i\in S}E_i.
\]

An unordered distinct pair `{u,v}` is an accepted edge exactly when some coalition `S∈H` groups the pair in every input position in `S`. The output blocks are connected components of this accepted graph. Empty coalitions, complements, negation, and arbitrary relabeling operations inside a circuit are outside the model.

The selected family is

\[
T_{q,m}(E_1,\ldots,E_m)
=\bigvee_{|S|=q}\bigwedge_{i\in S}E_i,
\qquad 1\le q\le m.
\]

The convention is **at least `q` agreeing positions**, not strictly more than `q`. Some consensus papers use the latter convention or a fractional quota; translate explicitly. A repeated child occupies multiple positions and contributes multiple votes. The circuit is a finite directed acyclic graph. A raw input has depth zero; a gate has depth one plus the maximum child depth. The primary theorem uses the **same fixed `(q,m)` at every computed node**. It permits sharing, repetitions, unrestricted width, and any number of raw partitions.

### 2.2 Restriction semantics and witness objectives

For `Y⊆V`, restrict **every raw partition** to `Y`, retain the same gate wiring, and recompute. Let `C[Y]` be the resulting output. A pair witness for a true output relation `x C[V] y`, with `x≠y`, is a set `Y` containing `x,y` such that `x C[Y] y`.

Define

\[
w(C,E,x,y)=\min\{|Y|:x,y\in Y\subseteq V,\ xC[Y]y\}.
\]

For three specified distinct output-equivalent objects, the triple objective requires all three to remain in one output block. Let `P_d(q,m)` and `Q_d(q,m)` be the suprema of these minimum cardinalities over finite domains, raw blocks of size at most three, and circuits of depth at most `d`. The object count includes terminals. At depth zero, `P_0=2`, `Q_0=3`.

These notions must remain distinct:

| Quantity | Meaning | What it does not establish |
|---|---|---|
| Cardinality-minimum witness | No smaller retained-object set works | A unique witness, causal explanation, or optimal runtime |
| Inclusion-minimal witness | No proper subset of that witness works | Global minimum cardinality |
| Small proof tree | Few derivation nodes or low derivation height | Few distinct retained objects |
| Minimum edge or tuple witness | Few relational facts are kept | Valid induced restriction of all raw partitions |
| Restriction of the final output | Retain output pairs between remaining objects | Correct recomputation after deletion |

An absent object and a present singleton are different states in the implementation. A leaf partition containing `{a,b,c}` still relates `a,b` after deleting `c`; storing that block as a fixed spanning tree and deleting tree vertices would implement the wrong semantics.

### 2.3 Facts, assumptions, and speculation

**Established or derived with an explicit proof:** the algebraic gate definition; monotonicity under enlarging a retained domain; the supplied fixed-gate theorem; the general-quota argument in Section 4; the reduction and compiler-equivalence observations in Section 4. These are mathematical claims with stated hypotheses, not performance measurements.

**Mechanism assumptions:** raw inputs really are equivalence relations; the same retained-object set applies everywhere; closure is exact; gate positions retain multiplicity; all raw blocks satisfy the declared size bound; no hidden learned preprocessing changes the partitions after deletion.

**Application assumptions:** a user benefits from minimum retained-object explanations; raw blocks of size at most three occur naturally in a relevant task; a gate-depth hierarchy is useful under realistic input and storage budgets. None has yet been established by an application study.

**Speculative extensions:** heterogeneous gate families, blocks of size four in a quota regime that preserves four, bounded numbers of raw inputs, edge deletion, learning circuit wiring, and group-word pooling. They are not necessary for this PoC. A general neural-network or language-learning project is outside the justified scope.

### 2.4 Hypotheses and discriminating observations

| ID | Claim or hypothesis | Supporting basis | Alternative explanation or failure | Discriminating evidence |
|---|---|---|---|---|
| H0 | The attached three-of-four theorem and evaluators have the stated semantics | Inspected proofs and reproduced checks | Both implementations repeat one convention error | Independent restriction-before-evaluation oracle, boundary counterexamples, mutation tests |
| H1 | The general-quota recurrence is an exact worst-case law | Upper argument plus recursively sharp pair/triple constructions | Incompatible extremizers, incorrect overlap counting, or hidden vertex identification | Type-specific constructors, proof review, small complete restriction checks, every deletion of larger constructed witnesses |
| H2 | Pair and triple extrema must sometimes be tracked separately | For `(q,m)=(4,5)`, depth 2 predicts `(P,Q)=(6,5)` | A pair-only estimate is accidentally treated as an equality | Independently construct both extremizers and verify their different sizes |
| H3 | The specialization offers a new generic provenance speedup | Initially plausible small-block structure | The generic compiler makes the same simplification | Already weakened decisively: identical formulas after fair pruning; stop this claim unless a distinct transformation is later specified |
| H4 | Minimum-witness optimization can remain difficult despite size-three components | Explicit polynomial Set Cover reduction | Counting occurrence costs rather than distinct objects, or allowing invalid deletions | Exact truth-table agreement for the reduction and direct objective matching |
| H5 | A hierarchy matters under a fixed resource budget | Possible circuit-design relevance | Construction growth pays for all additional context | Report gates, wires, raw incidences, active block events, and bytes alongside depth; do not infer a size advantage |

H1 is a universal theorem target: a valid counterexample refutes it. Finite positive tests alone cannot establish it. H5 is open and is deliberately separated from H1.

## 3. Research landscape, baselines, and novelty assessment

### 3.1 Closest prior art

The search covered lattice federation and quota consensus, partition/equivalence aggregation, Nakamura numbers, near-unanimity terms and relation identities, Datalog provenance, minimum subinstances, exact optimization, and implementations. Important sources were opened; access limitations are recorded in Section 12.

| Prior work | Problem and mechanism | Exact overlap and difference | Evidence, cost, implementation implications |
|---|---|---|---|
| Monjardet federation consensus; Hudry–Monjardet survey; Leclerc quota discussion [R1,R2] | Lattice functions formed as joins of coalition meets | The general `F_H` operator family already exists. The selected quantity deletes domain objects and tracks circuit depth | Mathematical consensus results; older original texts partly inaccessible. No new-gate claim |
| Dimitrov–Marchant–Mishra [R3] | Axiomatic join/meet aggregation of equivalence relations | Same partition objects and lattice operations; different separability question | Reuse formal definitions and compare assumptions; no empirical network benchmark |
| Endriss–Grandi and Filmus [R4,R5] | Graph/equivalence aggregation preserving consistency, including issuewise impossibility results | Closure depends on other pairs, unlike issuewise aggregation | These impossibility theorems do not directly refute the present closure-based operator; no claim to evade an identical theorem |
| Freixas–Kurz [R6] | Nakamura number of winning coalitions | Same minimum empty-intersection invariant and quota formula | The invariant is classical; the supplied bounded-block connection must be credited and separately assessed |
| Maximov–Goncharenko–Legovich [R7] | Lattice-valued associative memory and a learning rule | Establishes that nonnumeric/lattice network ideas pre-exist | Their lattice assumptions and task differ; not a directly comparable minimum-witness method |
| Deutch–Milo–Roy–Tannen [R8] | Shared Boolean circuits for Datalog provenance | Presence variables, positive Boolean expressions, sharing, and compact provenance are established | Reuse the paradigm; avoid expanding all proof trees or all minimal sets |
| Miao–Roy–Yang [R9] | Small query counterexamples via provenance and SAT/optimizing SMT | Presence provenance plus min-ones optimization is already an implemented method | Their tuple-deletion task differs from coupled vertex deletion, but is decisive algorithmic prior art |
| Rankooh–Rintanen [R10] | SAT encodings of reachability using vertex elimination | A strong generic closure baseline adapted from vertex elimination | After pruning to blocks of size three, the per-pair formula becomes the same two-hop formula used here |
| Calautti–Milani–Pieris, October 2026 [R11] | Diverse Datalog explanations using graph/hypergraph encodings and ILP | Current provenance optimization, support pruning, and encoding tradeoffs | Different objective and published server/Gurobi environment; use as context, not comparable laptop timings |
| QuickXplain [R12] | Irreducible subsets for monotone predicates | Useful inclusion-minimal witness machinery | Does not certify cardinality minima; fair role is a fast baseline or an upper-bound provider |
| Lipparini [R13] | Sharp congruence/relation identities involving lattice reducts and near-unanimity | Close algebraic neighborhood, but different quantified structures | Do not confuse congruences of an algebra with changing the object domain of `Eq(V)` |
| Supplied research note [A1] | Exact three-of-four depth law and raw-guard law | Direct starting point and required special-case baseline | Complete source and standard-library checks available; no public publication claim or independent peer review |

Consensus-clustering papers also define the same pair-agreement-then-closure quota rule [R14]. Thus even the full high-quota family is established as an operator. The proposed difference is its **sharp retained-object witness law under a raw block bound and composition by depth**.

### 3.2 Six explicit novelty questions

1. **What already exists?** The partition lattice, coalition/quota lattice operations, nonnumeric networks, the Nakamura invariant, provenance DAGs, exact minimum-subset optimization, and efficient reachability encodings.
2. **What can be reused?** The supplied fixtures and proofs; canonical partition representations; union-find or graph connectivity; Boolean DAG techniques; an existing exact solver if needed. A separate learning framework or custom GPU kernel is unnecessary.
3. **What is substantively different?** The exact two-state depth law for all quota gates in the size-three-preserving regime, including the divergence and later equality of pair/triple extrema. This formulation was not located in inspected sources.
4. **Why might it have value?** It gives an exact answer to a compositional context-dependence question, makes an otherwise hidden state distinction explicit, and provides sharp tests for symbolic computation systems.
5. **What isolates the difference?** Verify `(q,m)=(4,5)` at depth 2, where the two extrema differ, then verify quotas with more than one omitted vote, such as `(5,7)` and `(6,8)`. The original `(3,4)` case alone cannot distinguish the extension.
6. **What invalidates the novelty claim?** An earlier theorem with the same domain-restriction semantics and equivalent sharp recurrence, even under different notation; or a known theorem from which this is an immediate acknowledged corollary. A counterexample invalidates correctness, which is a separate question.

No search result establishes absence of prior art. Publication positioning remains blocked on specialist review of older lattice-consensus and universal-algebra literature. This is not a patentability or freedom-to-operate opinion.

### 3.3 Direction selection

| Direction | Assessment | Decision |
|---|---|---|
| Repeat only the supplied three-of-four result | Useful replication, little additional scope | Required baseline, not primary new contribution |
| General quota depth law for blocks of size at most three | Precise, falsifiable, constructive, locally verifiable; meaningful extension | **Primary** |
| Specialized two-hop provenance optimizer | Equivalent to a strong generic compiler after shared preprocessing | Keep one reference implementation; drop novelty/speedup claim |
| Train partition circuits for predictive tasks | No learning objective, corpus, or demonstrated bottleneck in the assets | Defer until an application and defensible learning hypothesis exist |
| Group-word pooling appendix | Different state algebra and independent symmetry theorem | Preserve as a separate research branch; do not combine its evidence with the partition claim |
| Four-object blocks under a preserving quota | Potentially richer terminal-state combinatorics | Evidence-triggered theoretical extension only after the primary gates pass |

## 4. Selected mechanism: exact law and implementation specification

### 4.1 Persistence and the single-gate boundary

For any circuit of partition meets and joins, `Y⊆Z` implies

\[
C[Y]\subseteq C[Z]\!\restriction_Y.
\]

Raw relations satisfy this directly; intersections preserve it; a connecting path on `Y` remains a connecting path on `Z`. Induction over the circuit proves persistence. It permits unioning witnesses for several child obligations.

For the coalition gate, let `ν(H)` be the minimum number of distinct members of the minimal-winning antichain `H` whose intersection is empty, or infinity if no such family exists. The supplied argument gives:

- If `ν(H)>b`, every output component has size at most `b` on raw blocks bounded by `b`. Otherwise a tree on `b+1` vertices has `b` accepted edges; their supporting coalitions share an input position, forcing an illegal input block of size `b+1`.
- If `ν(H)≤b`, label the edges of an arbitrarily long path periodically by a minimum empty-intersection family of distinct members of `H`. Input `i` is the partition into components of the path edges whose labels contain `i`. Its blocks have at most `ν(H)` vertices. Antichainhood prevents accepted nonconsecutive edges, so all path vertices are necessary for the endpoints.

For a quota gate with `q<m`, `ν=ceil(m/(m-q))`; unanimity has `ν=∞`. The invariant and quota formula are classical [R6]; the partition interpretation is in the supplied source [A1]. Thus the entire quota classification for positive depths, up to raw block size three, is:

| Raw block cap | Quota condition | Exact worst pair witness |
|---|---|---|
| 1 | Any | No distinct output pair is possible |
| 2 | `2*q > m` | 2 |
| 2 | `2*q ≤ m` | Unbounded at depth one |
| 3 | `q=m` | 2; triple witnesses are 3 |
| 3 | `q<m` and `3*q ≤ 2*m` | Unbounded at depth one |
| 3 | `q<m` and `3*q > 2*m` | The recurrence below |

For arbitrary `b≥2`, unanimity remains an intersection and pair witnesses remain two. The original three-of-four gate has `ν=4`, so its size-four regime is unbounded. Do not use that fact to conclude that every high-quota gate is unbounded at size four.

### 4.2 Working theorem: the exact finite quota recurrence

**Status:** derived and proof-reviewed in this investigation; finite constructions independently checked; not formally proof-assistant certified or externally peer reviewed; historical priority unresolved.

Fix integers `1≤q<m` satisfying `3*q>2*m`. Put

\[
s=m-q,\qquad h=2q-m,
\qquad p_0=t_0=0.
\]

For every `d≥1`, compute

\[
\boxed{t_d=h\,t_{d-1}+2s\,p_{d-1},\qquad
p_d=\max\{q\,p_{d-1},\ 1+t_d\}.}
\]

Then

\[
\boxed{P_d(q,m)=2+p_d,\qquad Q_d(q,m)=3+t_d.}
\]

Every bound is attained by a finite circuit. Intermediate blocks remain at most three on every restriction. Depth means at most `d`; padding by `T(E,…,E)=E` permits an exact-depth representation if necessary.

#### Upper proof

Assume bounds `P=2+p`, `Q=3+t` at the previous depth. A desired pair in an output block of size at most three is either a directly accepted edge or the endpoints of a two-edge path `a-c-b`.

For a direct edge, choose `q` supporting input positions. Their pair witnesses share the two endpoints, giving

\[
2+q(P-2)=2+qp.
\]

For a two-edge path, choose size-`q` supports `A,B` for its edges. Let `r=|A∩B|≥2q-m=h`. Each of the `r` shared children must preserve the triple `a,c,b`. Each of the `q-r` positions exclusive to each edge must preserve a pair. The witness union has size at most

\[
3+r(Q-3)+2(q-r)(P-2)=3+rt+2(q-r)p.
\]

At depth zero, `p=t=0`; after depth one the recurrence maintains `p≥t+1`. Therefore `t-2p≤0`, so this expression is largest at the smallest allowable overlap `r=h`. It becomes `3+ht+2sp`.

For a triple obligation, choose two accepted edges spanning its three vertices. If their supporting sets agree, `q` triple witnesses give `3+qt`. Otherwise the mixed bound applies. The mixed bound also dominates the identical-support bound because

\[
(ht+2sp)-qt=s(2p-t)\ge0.
\]

Thus `Q_new≤3+ht+2sp`, and the pair bound is the maximum of the direct and mixed expressions. These are precisely the recurrence. The first step yields `p_1=1,t_1=0`; induction maintains the invariant. The proof applies to DAGs as well as trees, because any accidental sharing of witness vertices only decreases the union size.

#### Matching lower proof: two constructor contracts

The distinction between these contracts is essential:

- **P gadget:** designated endpoints `a,b`; the full output has only one nonsingleton block, either the pair `{a,b}` or a triple containing it. On any restriction retaining `a,b`, their relation holds **if and only if every gadget vertex remains**.
- **Q gadget:** designated terminals `a,c,b`; the full output has exactly that triple and otherwise singletons. On restrictions retaining the three terminals, all three are related **if and only if every gadget vertex remains**. A Q gadget need not make a particular pair individually critical.

At depth zero use a raw pair block for P and a raw triple block for Q. The stated criticality properties hold because their entire domains are their terminals.

**Direct P construction.** Take `q` P gadgets of depth `d-1`, identify only their two terminals, and make all other vertices private. Extend all raw inputs by implicit singletons outside their own copy. Add `s=m-q` discrete input positions. At the parent, a nontrivial pair can pass only if all `q` nondiscrete children support it, so the output is their meet. The full output is the endpoint pair, and every copy must preserve that pair. Its size is `2+q*p`.

**Direct Q construction.** Similarly, take `q` Q gadgets sharing the designated triple, with private remaining vertices, plus `s` discrete positions. Preserving the output triple forces all child triples. The size is `3+q*t`. This branch is useful for the general proof and ties; after the base step the mixed Q expression dominates it.

**Mixed construction.** Use exactly `m=h+2s` children:

| Child role | Number | Shared terminals | Required event |
|---|---:|---|---|
| Q copies | `h` | `a,c,b` | All three remain related |
| Left P copies | `s` | `a,c` | Endpoint relation |
| Right P copies | `s` | `c,b` | Endpoint relation |

All other vertices of every copy are private, including an extra terminal appearing in a P gadget's full output triple. The parent pair `a,c` has exactly `h+s=q` supporters; `c,b` has exactly `q`; `a,b` has only `h<q`. Pairs involving private vertices have at most one supporter, which is below the quota in this regime. The accepted parent graph is exactly `a-c-b`.

After any restriction, persistence prevents new accepted edges. Preserving the endpoint relation requires both path edges. Their `q` required supporters force every shared Q triple and every exclusive P pair, hence every vertex in every child. Conversely, retaining all vertices makes the path. The construction is both P-critical and Q-critical and has

\[
3+h(Q-3)+2s(P-2)=3+ht+2sp
\]

vertices. Choose the larger direct or mixed P construction, and the larger direct or mixed Q construction. This realizes both upper bounds independently and completes induction.

This proof establishes a family of sharp examples and bounds. It does not establish that the generated circuits use the fewest possible gates, wires, or raw inputs.

### 4.3 Consequences, transition, and reference arithmetic

| Quota | `(P_1,Q_1)` | `(P_2,Q_2)` | `(P_3,Q_3)` | `(P_4,Q_4)` |
|---|---:|---:|---:|---:|
| 3 of 4 | (3,3) | (5,5) | (13,13) | (45,45) |
| 4 of 5 | (3,3) | (6,5) | (18,17) | (77,77) |
| 5 of 7 | (3,3) | (7,7) | (35,35) | (231,231) |
| 6 of 8 | (3,3) | (8,7) | (43,43) | (327,327) |

The 4-of-5 example is a decisive check against conflating the two states. Pair and triple extrema refer to different extremizing circuits, so a worst pair witness can exceed a worst triple witness.

An optional closed form exposes a change in the extremizing construction. Let

\[
D=1+\min\{j\ge1:q^j-2h^j+1\ge0\}.
\]

The minimum exists because `q>h>0`. For `1≤d<D`,

\[
P_d=2+q^{d-1},\qquad
Q_d=3+2(q^{d-1}-h^{d-1}).
\]

Here `D≥2` is the first depth at which the mixed P branch catches the direct branch and equality persists thereafter. Both extrema already equal three at depth one and can subsequently diverge. At this persistent crossover `D`, `P_D=Q_D=3+2(q^(D-1)-h^(D-1))`. Thereafter, writing their common value as `N_d`,

\[
N_d=mN_{d-1}-m-2q+3.
\]

At the persistent crossover, `p_D=t_D+1≥q`, so `t_D≥q-1`. At a later merged state the mixed-minus-direct excess in offsets is `s(t+2)+1-q≥0`, proving that the merge persists. Use integer arithmetic to locate `D`; a logarithmic approximation can misclassify equality cases.

For `q=3,m=4`, this reduces to `N_1=3`, `N_d=4N_(d-1)-7`, giving `(4**d+14)//6`. The numeric sequence was already catalogued as OEIS A321358; the object-restriction interpretation is the relevant question. [R15]

The following is complete executable Python for the arithmetic oracle, independent of the proposed repository CLI:

```python
def finite_quota_bounds(m: int, q: int, depth: int):
    if not (isinstance(m, int) and isinstance(q, int)
            and isinstance(depth, int)):
        raise TypeError("integer parameters required")
    if not (1 <= q <= m and depth >= 0):
        raise ValueError("require 1 <= q <= m and depth >= 0")
    if q == m:
        return [(2, 3)] * (depth + 1)
    if 3 * q <= 2 * m:
        if depth == 0:
            return [(2, 3)]
        raise ValueError("unbounded at positive depth for raw block cap 3")
    s, h = m - q, 2 * q - m
    p = t = 0
    rows = [(2, 3)]
    for _ in range(depth):
        t_new = h * t + 2 * s * p
        p_new = max(q * p, 1 + t_new)
        p, t = p_new, t_new
        rows.append((2 + p, 3 + t))
    return rows

assert finite_quota_bounds(4, 3, 4) == [
    (2, 3), (3, 3), (5, 5), (13, 13), (45, 45)]
assert finite_quota_bounds(5, 4, 3)[-2:] == [(6, 5), (18, 17)]
assert finite_quota_bounds(7, 5, 3)[-1] == (35, 35)
assert finite_quota_bounds(8, 6, 3)[-1] == (43, 43)
```

Do not evaluate `p_new` after overwriting `p` or `t`; the update is simultaneous. Python integers avoid overflow. A port to fixed-width integers must check growth explicitly.

### 4.4 Circuit representation and constructor interface

Use canonical immutable records, with IDs assigned in topological order:

```text
Profile:
  schema_version: 1
  vertices: sorted distinct integer IDs
  raw_nodes: [{id, blocks: [[v1,v2,...], ...]}]
  gates: [{id, q, inputs: [child_id, ...]}]
  output: node_id
  query: {kind: "pair" | "triple", terminals: [IDs]}
  metadata: {family, seed, depth, block_cap, constructor_policy}
```

`blocks` stores only nonsingleton raw blocks; all other present vertices are implicit singletons. Raw blocks must be disjoint, and every listed object must belong to `vertices`. The length of `inputs` is the arity. Its order is immaterial mathematically, but multiplicity is not. Each child ID must precede its parent. Construction-copy renaming must be an injective map except for explicitly shared terminals.

Implement these shared interfaces:

```text
bounds(m, q, d) -> [(P_i, Q_i)]
build_extremizer(m, q, d, kind, terminals, allocator) -> Gadget
evaluate(profile, retained_vertices) -> output_partition
evaluate_trace(profile, retained_vertices) -> node_partitions
query_holds(partition, query) -> bool
verify_witness(profile, retained_vertices) -> bool
minimum_exhaustive(profile) -> {size, vertices, subsets_tested}
criticality_certificate(profile) -> {full_true, deletion_results}
compile_presence(profile) -> BooleanDAG
solve_minimum(profile, options) -> SolveRecord
```

The builder must return its complete vertex set, designated terminals, all nodes, output ID, expected depth, contract P/Q, and exact size. At depth zero allocate the raw pair or triple. At larger depths obtain the arithmetic table, compare the direct and mixed cardinalities, and instantiate the corresponding construction from Section 4.2. Resolve ties by choosing **direct**, record that policy, and never treat different branch choices as theorem failures. A shared discrete raw node is allowed. Do not deduplicate arbitrary isomorphic subcircuits unless their globally named raw partitions and wiring are actually identical.

For the mixed case, allocate one shared middle terminal `c` if building P; Q already supplies `a,c,b`. Build `h` fresh Q copies on those three terminals, `s` fresh P copies on `a,c`, and `s` fresh P copies on `c,b`. Each call gets fresh private vertices. If a P child has a third output object, it remains private. This is the main place where an incorrect constructor can accidentally reuse context and lose sharpness.

### 4.5 Two independent evaluators

**Simple oracle:** for small instances, restrict each raw block by intersection, assign component labels, scan all unordered retained pairs at each gate, count supporting input positions, and run a graph search or union-find on accepted edges. It may be slow; its purpose is independent semantics.

**Sparse evaluator:** enumerate all unordered pairs within restricted raw blocks; for each gate count pair occurrences across each child position; accept counts at least `q`; compute connected components; return canonical blocks. In the finite size-three regime a raw/gate block contributes at most three pairs. Use sparse maps for active objects, not a dense length-`n` array at every node.

Pseudocode:

```text
for node in topological_order:
    if raw:
        blocks[node] = canonicalize(B intersect Y for B in raw.blocks)
    else:
        votes = empty counter
        for child_position in node.inputs:  # preserve repetitions
            for pair in pairs(blocks[child_position]):
                votes[pair] += 1
        edges = {pair for pair,count in votes.items() if count >= node.q}
        blocks[node] = nonsingleton_components(edges)
        if contract_is_size_three_preserving:
            assert max_block_size(blocks[node]) <= 3
return blocks[output]
```

Return each partition as the retained domain plus canonical nonsingleton blocks; the domain may be shared across node results. `query_holds` returns false if any terminal is absent. Store all intermediate node results only when needed for a trace or provenance. For ordinary evaluation, release a child's result after its last consumer, counting repeated input positions consistently or releasing only after the entire parent has been processed. Queries require distinct terminals; the raw restriction may be empty for general evaluation but not for a valid positive pair/triple witness.

### 4.6 Certificates: cheap sharpness checks and exact minima

For an extremizer with terminal set `T`, evaluate the full domain and every set `V\{v}` for `v∈V\T`. If the full query is true and every single deletion makes it false, then the full domain is the only witness. Any proper terminal-containing set omits some `v` and is contained in `V\{v}`; persistence forces it to be false.

This is a deterministic certificate of criticality, conditional on correct evaluation and the proven monotonicity. It is much stronger than random deletion sampling and requires `1+n-|T|` evaluations instead of `2^(n-|T|)`. Use exhaustive restrictions on small cases to cross-check the evaluator and the certificate logic. Larger all-required constructions are easy by this certificate; they are not evidence of difficult optimization performance.

For general profiles, first validate the query and evaluate the full domain. A false full-domain query conclusively returns `status="negative_full_query"`, with null witness size and vertices, by persistence; it is not solver `UNKNOWN`. For positive small instances, enumerate terminal-containing subsets in increasing cardinality until the first success. For larger instances, an optional exact presence-provenance solver uses one variable `z_v` per **global object**, with the terminals fixed true. A raw pair event is `z_u AND z_v` when the pair is co-block in that full raw partition, and false otherwise. At a gate, compile a threshold over child pair events. The two-hop compiler must first verify that every relevant full-domain block has size at most three. Otherwise return `unsupported_block_size` or use a genuine general-reachability encoding; boundary fixtures must not silently enter this fast path. For a full-domain pair block, the event is the accepted edge; for a triple block `{u,v,w}` it is

\[
R_{uv}=A_{uv}\lor(A_{uw}\land A_{wv}).
\]

Pairs absent in the full-domain node partition are constant false by persistence. This also gives the proper generic vertex-elimination result. Use a shared Boolean DAG; do not expand formulas into a list of all minimal witnesses. A threshold may use the same native reified integer count on both compared methods; enumerating all `q`-subsets is unnecessary for large arity.

For CP-SAT, implement Boolean equivalences, not one-way permissions. For an AND node `y`, enforce `y≤x_i` for every child and `y≥sum(x_i)-(k-1)`; for OR enforce `y≥x_i` and `y≤sum(x_i)`. For a threshold `a` over `m` Boolean input positions, enforce `sum(inputs)≥q` when `a` and `sum(inputs)≤q-1` when `not a`. Repeated inputs keep repeated coefficients. For a pair assert its root event; for a triple `(a,c,b)`, fix all three presence variables and assert `R_ac AND R_cb`. Minimize `sum(z_v)`. Empty Boolean AND is true and empty OR is false. All coefficients are integers.

Every solution is reevaluated independently: use the simple dense oracle on small fixtures and, after cross-validation, a separately implemented sparse graph verifier for large witnesses. Include its cost in the total. `OPTIMAL` establishes a minimum under a validated model; `FEASIBLE` gives an upper bound; `UNKNOWN` gives neither falsity nor optimality. Also solve the feasibility problem with `sum(z_v)≤k-1` after a claimed size-`k` optimum on selected independent checks. This is a solver cross-check, not a machine-checkable mathematical proof unless a separately validated proof certificate is exported. [R18]

### 4.7 Hardness, overlap, and resource interpretation

For the original gate `T=T_(3,4)`, encode Set Cover as follows. Let the cover universe be nonempty and every element covered by at least one candidate set `S_i`. Use terminals `x,y` and one optional object `v_i` per set. The raw path gadget with blocks `{x,v_i,y}`, `{x,v_i,y}`, `{x,v_i}`, `{v_i,y}` relates `x,y` exactly when `v_i` is kept. Let raw guard `G` have the sole nonsingleton block `{x,y}`.

The identities

\[
T(P,P,Q,Q)=P\wedge Q,
\qquad T(G,G,P,Q)=(G\wedge P)\vee(G\wedge Q)
\]

implement conjunction and disjunction of the endpoint events. Make one OR clause for the sets covering each universe element, then AND the clauses. A one-input clause can be clipped with `T(G,G,P,P)` if a pair-only signal is desired. There are polynomially many gates, all raw blocks have size at most three, and

\[
w=2+\min\{|I|:\bigcup_{i\in I}S_i=U\}.
\]

Verification is polynomial, so the unrestricted-depth decision problem is NP-complete. If the universe is empty, the cover problem is trivial and handled separately; if an element is uncovered, the full query is false. The reduction uses an NP-hard problem as a mathematical starting point, not as an assertion that the tiny fixtures below are hard.

This does not contradict the depth theorem. With fixed `q,m,d`, exhaustive search only needs cardinalities through `P_d`, taking an upper bound of `O(n^(P_d-2))` oracle calls for pair queries. This is an exponent depending on depth, not a generally efficient or fixed-parameter algorithm.

Never add minimum child costs as scalars to find a minimum global witness: nonminimum child choices can share objects and give a smaller union. Distinct-object provenance must retain overlap information.

For resource accounting at `d≥1`, the original full four-copy tree has `g=(4**d-1)//3` gate occurrences and `N=(g+5)/2` vertices. This is not an optimal-size lower bound. The branch-selected constructors in this plan can have fewer gates. For instances whose computed full-domain blocks all have size at most three, an optional diagnostic counts those triple blocks as `B3`. Merge backward obligations by `(node, full-domain block)`; witness extraction introduces at most one new object at each such block event, giving a conservative bound `min(n,P_d,2+B3)`. Record the explicit backward proof if using this bound; bare gate count is insufficient when a gate has many blocks. Treat this diagnostic as a secondary deduction to audit, not a substitute for the main exact law.

## 5. Evaluation designed around the actual claim

### 5.1 Primary evidence and appropriate baselines

The primary claim is exactness of the recurrence, including its domain of applicability. Its units are retained objects. The universal evaluation population is all finite valid profiles and homogeneous quota circuits at the specified `(m,q,d)`. The primary evidence is a complete upper proof plus matching lower constructions. Computational metrics are disagreement counts, exact integer witness cardinalities, and verified criticality certificates. Statistical significance is not a substitute for a proof.

Use these comparisons:

| ID | Comparator | Role and fairness requirements |
|---|---|---|
| B0 | Supplied `3-of-4` exact theorem and scripts | Required special-case reproduction; there is no improvement over its exact answer at the same parameters |
| B1 | Supplied raw-guard circuits | Exact restricted-class control: `(b-1)^d+1`; for `b=3`, `2^d+1`. Computed guards are a different class |
| B2 | General path-witness union bound | Analytic diagnostic `U_0=2`, `U_d=3+2q(U_(d-1)-2)`, hence `U_d=2+((2q)^d-1)/(2q-1)`. It ignores shared child triple obligations; label it a coarse bound, not the strongest published theorem |
| B3 | Full subset enumeration with simple evaluator | Exact local oracle; increasing cardinality, one global retained set |
| B4 | Generic block-pruned vertex elimination + exact min-ones solver | Strong conceptual software comparator, using the same full-domain information, threshold encoding, simplifier, and backend |
| B5 | Single-pass greedy deletion; optionally QuickXplain | Fast inclusion-minimal output and upper bounds; not equal-quality competition with exact minima |
| B6 | Direct Set Cover objective on translated instances | Mechanism oracle for the reduction; not a fair general partition-solver runtime competitor because it receives an additional decomposition |

Current Datalog diversity systems and lattice associative-memory learning are relevant context but solve different objectives. Published timings on server hardware or commercial solvers are not head-to-head results. There is no need to port a different task merely to create a longer baseline list. If a future claim concerns clustering quality or prediction, this evaluation is insufficient and must be redesigned around that task.

For B4, first compare canonical formulas. If they are identical—as the present probe found—collapse the nominal methods into one implementation. Do not run a large experiment to turn random timing differences between equivalent formulas into an architectural claim. An independent encoding, such as bounded-stage reachability, is useful for correctness cross-checking, not as a deliberately weaker competitive baseline.

### 5.2 Deterministic coverage and falsification suite

Freeze the following **functional obligations** before implementing the generalized constructor. Exact maximum instance sizes are then chosen from calibration; the obligations may not be dropped merely because a preferred method loses.

1. **Boundary classes:** singleton inputs; raw pairs above and at the `2q=m` boundary; raw triples above and at `3q=2m`; unanimity; empty/full domains; negative full queries; malformed inputs rejected.
2. **Original result:** the supplied fixed gate at depths 0–3 with exhaustive restrictions; at least one larger instance with all single nonterminal deletions. Reproduce the standard-library reports when the archive is available.
3. **Generalized law:** both P and Q contracts; `(q,m)=(4,5)` at depths 2 and 3; `(5,7)` and `(6,8)`; branch ties; a direct-to-mixed transition; raw/identity padding; shared DAGs and equivalent tree expansions.
4. **Independent arithmetic:** compare the offset recurrence, the unsimplified two-state max recurrence, and the transition formula using exact integers. Include a larger-arity parameter range without materializing its exponentially large witness domains.
5. **Independent semantics:** exhaustive retained subsets on small generated circuits, comparing all node outputs between simple and sparse evaluators. Input partition generation and circuit wiring must not call the tested constructor.
6. **Minimum versus minimal:** the six-vertex Set Cover fixture `U={0,1,2}`, sets `[U,{0},{1},{2}]`. Its optimum is three including terminals; deleting the universal-set vertex first yields an inclusion-minimal witness of five.
7. **Overlap controls:** alternatives sharing the same object across several child events; duplicated child positions; independent copies with private vertices; raw triple restriction where the middle object of a stored tree is deleted.
8. **Non-distributivity:** preserve the supplied matching-based example distinguishing join-of-meets from meet-of-joins. Boolean identities may simplify a presence formula, but not replace a lattice gate by a different lattice polynomial.

For a standalone non-distributivity fixture on `V={0,1,2,3,4,5}`, use `E={{0,3},{1,4},{2,5}}`, `F={{0,4},{1,5},{2,3}}`, and `H={{0,5},{1,3},{2,4}}`. Every pairwise meet is discrete; every pairwise join is universal. Thus the join of the three pairwise meets is bottom, while the meet of the three pairwise joins is top.

The raw-guard control is also constructible without the archive. Take a path of `(b-1)^d` edges. Recursively split each interval into `b-1` equal subintervals; its raw guard has the `b` subdivision endpoints as its sole nonsingleton block. Recurse on the subintervals and use raw edge-pair leaves. The parent computes `join_j(G meet child_j)`. For `b=3`, each parent is `T(G,G,left,right)`.

For larger extremizers, full-domain truth and every single nonterminal deletion are a complete criticality test, not a sampled proxy. For arbitrary circuits this only finds mandatory objects; it does not usually identify a minimum witness.

### 5.3 Generated data and provenance

No public dataset is required for the theorem claim. Generate exact combinatorial profiles; preserve their bytes and generator revision. This avoids data access and license uncertainty without pretending synthetic evidence establishes application utility.

The non-extremal generator must state its distribution. A simple semantic generator shuffles `0..n-1`, repeatedly draws a requested block size uniformly from `{1,2,3}`, truncates at the remaining length, and forms a raw partition. This is **not uniform over partitions**. For homogeneous safe-quota tests, cycle gate roles deterministically by index modulo three: AND, raw-guarded OR, and unrestricted position sampling. For AND, repeat a sampled earlier child `P` in `floor(m/2)` positions and `Q` in the remaining positions; both multiplicities are below the quota. For guarded OR, choose a raw guard `G`, repeat it `h=2q-m` times, and repeat sampled earlier `P,Q` each `s=m-q` times. Since `2s<q`, this accepts exactly `G AND (P OR Q)` at each pair before closure. For the third role, draw all `m` positions independently from earlier nodes. Freeze the role cycle, distributions, and seeds in the protocol; do not insert a four-input gate into a different-arity circuit. Record each family separately; independently drawn small partitions frequently produce discrete outputs, and that itself must be retained in the report.

For positive-query evaluation, first generate and retain the whole profile, evaluate it, then sample a positive pair from its output if one exists. Report the positive-profile rate and the selection rule. Do not silently reject negative or easy instances until a difficult-looking dataset remains. Random labels of one extremal gadget are equivariance tests, not independent structural instances.

Optimization stress instances, if the optional solver is used, should include covering systems with multiple alternatives, overlapping sets, dominated and duplicate sets, forced choices, irrelevant objects, and disconnected query-irrelevant components. Generate them directly, save the cover system and its translated circuit, and distinguish any planted optimum from an optimum independently established by exhaustive search or a solver.

The uploaded bundle contains no explicit software redistribution license in its listed files. Local inspection and reproduction of user-supplied material are available for this task; do not invent an open-source license for redistribution. The new implementation can be written from the specification, with authorship and licensing chosen explicitly before public release. Papers being downloadable does not license their code or text for unrestricted redistribution.

### 5.4 Exploratory versus confirmatory work

The derivation and finite checks reported in this plan are **exploratory evidence already observed**. They cannot become an unseen confirmatory sample by changing their label. The executing agent should freeze a `protocol.json` containing the theorem statement, constructor contracts, check families, expected values, comparator semantics, and failure policy before running its independent confirmation suite.

Confirmation comprises an independent code path and fresh cases chosen to cover the mathematical branches, plus the proof audit. Use separate random seeds for exploratory fuzzing and confirmation, recorded before results. If a theorem or assumption changes, retain the failed instance, make a methodological amendment, and choose new independent cases. This is a discipline against adaptive interpretation, not an assertion that random tests prove a theorem.

A counterexample found by a sound evaluator is decisive. A solver timeout or inability to materialize a huge construction is not a counterexample. A construction that is not critical may indicate a broken builder even when the upper bound remains valid; separately test the proof obligation it violates.

### 5.5 Conditional performance protocol

Do this only if a **distinct, justified transformation** remains after equivalence analysis, or if resource characterization itself is the declared objective. The current plan does not predict a compiler win.

For exact witness extraction, the primary performance endpoint is wall-clock seconds from parsing the saved profile through required preprocessing, compilation, exact optimization, and independent witness validation. Cold process startup is included in a separately reported user-facing total; also report the execution-only boundary. Report every phase, peak resident memory, formula/constraint counts, exact solve fraction, witness size, and certified lower/upper gaps. For repeated-query use, report the break-even query count rather than hiding compilation cost.

Use the same profiles, queries, solver release, worker count, seeds, threshold representation, simplifier, query-support pruning, and initial upper bounds. Any generally useful optimization must be shared. Primary measurements use one worker/process; an identical matched multicore operating point may be secondary. Do not infer GPU speed from CPU data or include numerical approximations.

Pilot timing determines the comparison horizon `T`, repetitions, and sample size. Fix them before protected results. Enforce `T` with an outer process deadline covering required compilation, search, and verification, not just a solver time limit. Wrong witnesses and invalid models are correctness failures that block confirmation and require repair; they are not ordinary censored timings. For a horizon `T`, assign each instance the capped time to certified optimum `min(t,T)`; every unproved result contributes `T`, even if a backend stops early. Report the paired mean difference in these capped times and the certified-solve fraction. This estimates performance within the declared budget, not uncensored time-to-solution. Also show per-family distributions and worst regressions; a solved-only speed ratio is secondary.

The independent unit is the generated circuit/profile, with all its queries, repetitions, and solver seeds nested. Average or otherwise pre-specify aggregation within an instance before inference. Use a stratified paired bootstrap over independent instances for confidence intervals when the sample is adequate; do not count each deletion, query, or timing repetition as a separate independent observation. Fix family weights before measurement.

Let `s_delta` be the pilot standard deviation of paired instance-level capped-time differences. If a practical or precision target `delta` is justified by an actual workflow, use approximately `N=((1.96+0.84)*s_delta/delta)^2` as an initial 80%-power planning calculation, then assess censoring and non-normality by simulation from the pilot. Alternatively target a fixed interval half-width with `N≈(1.96*s_delta/half_width)^2`. These are planning approximations, not a universal test prescription. If no application supplies a meaningful `delta`, report achieved precision without inventing a minimum percentage gain.

Choose repetition counts to make within-instance timing uncertainty small relative to between-instance variation. Randomize paired method order, warm the interpreter/solver on excluded fixtures, use monotonic timers, and keep the laptop on the same power policy with background load recorded. CPU completion is synchronous here; no GPU synchronization is relevant. Verify cache handling and preserve genuine outliers, crashes, and thermal slowdowns unless a predeclared measurement-invalidity rule applies. Freeze a fixed confirmatory sample; do not keep running until a confidence interval favors the proposal. Adjust secondary multiple comparisons or label them exploratory.

Practical superiority requires an interpretable tradeoff: report absolute seconds/bytes as well as relative differences, and disclose a lower solve rate or memory regression. A small reliable effect may be real but immaterial. A wide interval is inconclusive; an interval that excludes effects important to the declared task supports absence of a useful benefit in that population.

## 6. M3 Max feasibility and hardware-adaptive configuration

### 6.1 Discovery before dependencies or scaling

Run these existing system commands and record stdout, errors, and units in `environment.json`:

```bash
uname -m
sw_vers
sysctl -n hw.memsize
sysctl -n hw.physicalcpu
sysctl -n hw.logicalcpu
df -k .
python3 --version
python3 -c 'import platform, sys; print(platform.machine()); print(sys.executable)'
```

Use Activity Monitor or an available read-only system interface to record memory pressure and available headroom. Detect Rosetta translation if applicable; primary runs require a native `arm64` interpreter. Do not infer memory from the chip name. Apple lists several M3 Max memory/CPU configurations [R20]; actual discovery governs the run. No personal serial number or unrelated machine data is needed.

### 6.2 Software stack and dependency policy

The **core** uses native CPython, its standard library, exact integers, `unittest`, JSON, `hashlib`, and monotonic timing. Select an installed supported native Python 3.12 or 3.13 if available; otherwise install a compatible signed official macOS distribution and record the exact patch release and artifact checksum. Official Python macOS installers support native Apple Silicon through universal2 builds [R19]. Do not modify Apple's system Python.

The supplied scripts require Python 3.10 or later. Their successful run here used 3.12.14 on Linux; repeat locally. Create a virtual environment even for a standard-library project so the interpreter choice is explicit. No NumPy, PyTorch, MLX, MPS, CUDA, model downloads, or floating-point training is needed.

The **optional exact optimizer** uses `ortools==9.15.6755`, the inspected release, with a matching native arm64 wheel. The official package metadata lists macOS 11+ arm64 CPython 3.12 and 3.13 wheels and Apache 2.0 licensing [R16]. Google's installation page documents testing on macOS Ventura/M1, which supports plausibility but does not certify this user's macOS/M3 configuration [R17]. Smoke-test import, integer Boolean constraints, status handling, and the size-three versus size-five fixture before any dependent experiments.

After choosing the interpreter, substitute its exact validated absolute `sys.executable` for `VALIDATED_PYTHON` below. These are real package commands after that substitution, not proposed project commands. Recheck the created environment's architecture and version; the default `python3` may point elsewhere:

```bash
"VALIDATED_PYTHON" -m venv .venv
.venv/bin/python -m pip install --only-binary=:all: ortools==9.15.6755
.venv/bin/python -m pip freeze --all
.venv/bin/python -m pip check
```

The install line is needed only for the solver branch. Save exact direct and transitive versions plus wheel hashes in the lock manifest. If no compatible wheel is available, continue the standard-library theorem PoC and small exhaustive oracle. A solver is not a circular prerequisite for proving the construction. Do not silently run an emulated solver and label its timing native. Source builds or a different exact backend require a recorded compatibility and fairness amendment, not a change to witness semantics.

### 6.3 Resource model

Let `n` be object count, `r` raw nodes, `g` computed gates, `W` input-position references, `B` nonsingleton raw membership entries, `E` stored related-pair events across relevant nodes, `F` Boolean DAG nodes, and `L` encoded literal/constraint occurrences. For blocks of size at most three, a partition has at most `n` nonreflexive unordered related pairs, so `E≤(r+g)n`; sparse constructed gadgets are much smaller.

A complete memory model is

\[
M_{peak}=M_{runtime}+M_{profile}(B,W,n,r,g)
+M_{evaluation}(E_{live},N_{live})+M_{provenance}(F)
+M_{solver\ model}(F,L)+M_{search}+M_{buffers/logs}.
\]

`N_live` counts live result/node records, including empty partitions. Solver-model and search terms vanish when the solver is unused; provenance vanishes separately when it is not compiled. Buffers and logs remain in all branches. Python objects and dictionaries cost substantially more than packed integers; measure both per-node/intercept cost and bytes per stored event on the selected representation, including all-discrete profiles. Native solver search memory is not determined solely by model size. Process concurrency multiplies several terms.

Dense singleton expansion is avoidable overhead. For `m>1`, a full `m`-ary formula has at most `(m**d-1)/(m-1)` gate occurrences and `m**d` leaves, but an extremizer stores only a small raw block per leaf. For unary unanimity use at most `d` gate occurrences and one leaf instead. Repeating all `n` singleton labels in every raw input creates artificial quadratic growth. Implicit singletons preserve the actual sparse input description and must be used by all relevant implementations.

Actual counts obtained during the present construction checks are below. They use direct-branch preference on ties, a shared discrete node, and no further identical-subexpression elimination. These are **counts, not estimates of an optimal circuit or M3 measurements**.

| `(m,q,d)` | Pair witness objects | Computed gates | Raw membership entries |
|---|---:|---:|---:|
| (4,3,1) | 3 | 1 | 10 |
| (4,3,2) | 5 | 4 | 30 |
| (4,3,3) | 13 | 19 | 136 |
| (4,3,4) | 45 | 77 | 544 |
| (4,3,5) | 173 | 309 | 2,176 |
| (4,3,6) | 685 | 1,237 | 8,704 |
| (5,4,4) | 77 | 130 | 1,286 |
| (7,5,3) | 35 | 49 | 679 |
| (8,6,3) | 43 | 65 | 1,088 |

The branch-distinguishing core fixtures are small enough that CPU and memory needs should be modest, but this is an estimate until local calibration. The supplied and new finite checks ran in seconds to tens of seconds on the research environment; that is not a prediction of the same M3 runtime. No fixed completion-time promise is made.

### 6.4 Stage-specific calibration and configuration rules

| Stage | Principal cost | Calibration and feasible fallback |
|---|---|---|
| Integer bounds/transition table | Integer arithmetic; bit length grows with `d log m` | Measure one large arithmetic case; keep symbolic counts when materialization would be huge |
| Constructor | Raw entries, nodes, private-vertex allocation | Predict exact counts before allocation; stream JSON or use compact arrays if object overhead dominates |
| Full evaluation | Pair visits, support counts, DSU work | Measure active pair visits and peak live state; release dead node outputs |
| Criticality certificate | `1+n-|T|` evaluations | Measure one full run and a few deletion runs; project total before executing all |
| Complete restrictions | `2^(n-|T|)` evaluations | Use for small cross-checks selected by measured cost; switch to logically complete single-deletion certification for extremizers |
| Optional provenance | `F,L` and compiler caches | Measure before solver creation; collapse identical compiler variants |
| Optional exact solver | Model plus unpredictable search | One process initially; track incumbent and bound; timeouts remain censored/inconclusive |

Choose a process memory budget from measured free headroom, with an explicit reserve for macOS and other work. One conservative starting policy is the lesser of 40% of physical memory and 60% of measured available headroom; these are operational safety margins, not scientific scale limits. Refine them after observed RSS and memory pressure. Project using the largest measured bytes-per-event ratio and a documented margin. Avoid a run projected to exceed the budget; reduce simultaneous processes or instance expansion, preserving the same mathematical mechanism.

Record the exact source and definition of any available-memory estimate. If no reliable headroom estimate is available, start serially with a conservative 25% of physical memory as the process budget and monitor pressure; this fallback must not block the small theorem fixtures. Save peak RSS both in its native unit and normalized bytes. For `getrusage`, macOS reports bytes and Linux reports KiB; verify the platform/unit conversion with a known-size allocation before comparing research-environment and laptop records. [R21]

Storage is dominated by saved profiles, traces, and solver models, not datasets. Estimate serialized bytes from pilot fixtures, multiply by the planned manifest, and include temporary-write and failed-run space. Save compact deletion verdicts and hashes by default; retain a full trace for every failure and representative passing case. Generate comprehensive traces on demand. Do not delete unfavorable runs to recover space.

Select scale by the uncertainty it resolves. Materialize a deeper gadget if it exercises a new constructor branch or exposes a resource-scaling question. Repeating the same inductive shape at a huge depth is not necessary to establish the theorem. A low-memory laptop can complete the core small cases and arithmetic proof checks faithfully; inability to instantiate enormous sharp examples limits materialization claims, not the theorem's logical range.

### 6.5 Interruption and recovery

Each case has a hash-derived ID and atomic completion record. Checkpoint after each deletion, exhaustive-mask batch, or completed instance. Store the next mask/deletion index and accumulator hash. Resume only if the profile, code revision, protocol, and environment compatibility fields match. A changed implementation starts a new run lineage.

For a solver, save the serialized model, best witness, best known bound, parameters, and cumulative elapsed time. Restarting with a saved incumbent is **not** restoration of internal search state. Label it a restart, include prior spent time in total resource accounting, and independently validate the incumbent before supplying it as a hint. On memory exhaustion, preserve the partial record and classify the run as a resource failure; do not turn it into a negative mathematical result.

## 7. Ordered work packages and decision gates

The coding agent may make routine, reversible implementation choices. A change to witness semantics, theorem domain, constructor contract, evaluation population, primary endpoint, or baseline fairness requires a recorded amendment and downstream revalidation. The project is not authorized to contact authors, publish, buy services, or use remote compute merely by this plan.

### WP0 — Establish a reproducible local starting point

**Objective/risk:** remove hardware, source, and dependency ambiguity before substantive work. **Inputs:** this document, the laptop, and the optional supplied archive. **Entry:** none.

Create the repository structure in Section 9. Record the discovery commands from Section 6, interpreter architecture, available storage, source checksums, and any missing archive. Validate archive member paths before extraction. Keep an immutable copy under `vendor/original/`, with working reruns under `artifacts/replication/`; do not overwrite the supplied saved reports. Create a native virtual environment. Verify a standard-library smoke run before considering the optional solver.

**Outputs:** `environment.json`, `sources.json`, `decisions.md`, dependency manifest, initial git revision. **Resource:** negligible relative to later stages; record actual environment footprint. **Acceptance:** a native interpreter runs, the source provenance is explicit, and the core plan has sufficient disk/headroom for its smallest fixtures. **Next:** proceed to WP1; if the solver is unavailable, defer only WP6's solver branch. If the archive is missing, record the limitation and reconstruct from Section 4.

### WP1 — Implement and challenge the semantic kernel

**Objective/risk:** prevent every later check from sharing one wrong interpretation. **Dependency:** WP0.

Implement schema validation, raw restriction, canonical block labeling, `query_holds`, and the simple evaluator. In a separate module, implement the sparse support-count/DSU evaluator. Avoid sharing the raw-restriction implementation between the two initially. Add exact tests for repeated children, `T(P,P,Q,Q)`, guarded OR, raw triple deletion, identity, unanimity, absent terminals, malformed blocks, and object renaming.

Deliberately inject three faults in isolated mutation tests: restrict final output instead of raw inputs, deduplicate child positions, and delete a vertex from a stored spanning tree instead of restricting its partition block. Each mutation must be rejected by a known fixture.

**Outputs:** `model.py`, `eval_reference.py`, `eval_sparse.py`, `tests/test_semantics.py`, JSON test report. **Resource:** tiny exact cases; calibrate one `n=6` run. **Acceptance:** complete node-output agreement on the small enumeration set and all intentional faults detected. **Failure:** save the smallest discrepant profile and trace; stop theorem/solver measurements until resolved. Do not patch expected values to fit the implementation.

### WP2 — Reproduce the supplied claims faithfully

**Objective/risk:** establish the exact starting point and detect asset drift. **Dependencies:** WP0; use WP1 for independent checks.

In a working copy of the supplied directory, these **existing scripts** run its standard-library checks. Set `PARTITION_POC_PYTHON` to the absolute path of the native virtual-environment interpreter validated in WP0, then invoke that same interpreter for every script:

```bash
"$PARTITION_POC_PYTHON" check_partition_threshold.py --output rerun_partition_checks.json
"$PARTITION_POC_PYTHON" fixed_gate_fourfold.py
"$PARTITION_POC_PYTHON" guarded_depth_general.py
"$PARTITION_POC_PYTHON" check_group_words.py
```

The last command writes a report next to itself; hence the working copy. Compare deterministic JSON fields and text results, excluding elapsed time and explicitly documented machine metadata. Check source claims against actual loop limits. The group appendix is reproduced for artifact integrity but remains outside the primary contribution.

**Outputs:** exact command/stdout/stderr records, comparison diff, independent checks of selected fixed-gate and raw-guard fixtures. **Resource:** measure locally; present Linux elapsed times are context only. **Acceptance:** all mathematical counts/assertions match, or a specific source discrepancy is explained and retained. **Next:** WP3 if clean. A contradiction in a proof or semantics blocks dependent claims; unavailable original assets block only the claim of reproducing those assets.

### WP3 — Freeze and check the quota theorem specification

**Objective/risk:** validate the general law separately from its construction code. **Dependencies:** WP1 and the proof text; WP2 should be complete where available.

Implement `bounds.py` using both the offset recurrence and the unsimplified P/Q maxima. Implement the exact transition test. Assert unanimity and boundary dispatch. Write `proof_obligations.md` with the persistence lemma, coalition-overlap argument, invariant `p≥t+1`, direct/mixed upper cases, and P/Q constructor contracts. Verify every algebraic identity with integer examples, including tie cases.

Freeze a `protocol.json` covering the confirmation suite and expected small tables before inspecting its results. Distinguish this freeze from the already observed exploratory tests in Section 11.

**Outputs:** arithmetic tables, classification tests, frozen protocol hash, proof obligations. **Resource:** exact integer arithmetic only; no large vertex domains required. **Acceptance:** arithmetic forms agree, the special case matches A1, and a proof review finds no unresolved implication. **Next:** WP4. If the claimed exact recurrence fails, preserve valid bounds, identify which constructor/upper step fails, and revise the theorem before more scale.

### WP4 — Build sharp pair and triple constructions

**Objective/risk:** verify attainability without confusing the two criticality contracts. **Dependencies:** WP1 and WP3.

Implement `gadgets.py` exactly as Section 4.4, with fresh private allocations, terminal sharing, a shared bottom, and direct tie preference. Predict object/gate/raw-entry counts before allocation. Validate raw disjointness and topological wiring. For each functional case in Section 5.2, verify the expected size, full output shape, depth, all intermediate block caps, and contract-specific criticality. Use full subset enumeration when the calibrated cost permits; otherwise use full truth plus every single nonterminal deletion.

**Outputs:** canonical fixture files, deletion certificates, independent evaluator traces for representative cases and all failures, size tables. **Resource:** constructor prediction and measured evaluator throughput; no arbitrary maximum depth defines theorem validity. **Acceptance:** both P and Q examples attain the recurrence, including the 6-versus-5 case. **Next:** WP5. If a constructed example fails, distinguish naming collisions, an invalid contract, and a genuine lower-bound defect; do not assume the upper theorem failed automatically.

### WP5 — Independent confirmation and adversarial coverage

**Objective/risk:** expose correlated code errors and unsupported generalization. **Dependencies:** WP1–WP4.

Run the frozen confirmation suite with the independent evaluator. Include fresh generated profiles and complete retained-subset checks on the locally calibrated small range. Test b=2/b=3/b=4 boundaries using only regimes whose hypotheses actually apply. Test equivalent DAG and tree presentations, renamings, negative queries, and profiles containing many output blocks. Run mutation tests and compare the recurrence with construction certificates and exact enumeration.

**Outputs:** `confirmation.jsonl`, coverage report, failure/counterexample corpus, proof amendment log if needed. **Resource:** choose complete-enumeration sizes using the measured cost; do not change the mathematical population to hide failures. **Acceptance:** no unresolved semantic mismatch; all required mathematical branches covered; every encountered failure classified. **Next:** WP7 for the core. WP6 is optional and useful if general witness optimization or the hardness reduction is being included in the final research artifact.

### WP6 — Optional exact witness analyzer and hardness diagnostics

**Objective/risk:** characterize global minimum extraction without claiming an invented solver advantage. **Dependencies:** WP1, small exact oracle, and a passed local solver smoke test for the solver portion.

Implement one shared presence compiler, generic block-pruned elimination, formula normalization, and exact objective as in Section 4.6. Check structural formula equality first. Collapse duplicate implementations when equal. Implement the Set Cover translation and compare every assignment on small cases, including the minimal-versus-minimum trap. Implement greedy deletion as an upper-bound provider. If using CP-SAT, record every solver status, witness, lower/upper bound, and independent reevaluation; use a `k-1` infeasibility check on selected optima.

**Outputs:** semantic-reduction report, compiler-equivalence report, exact/heuristic witness records. **Resource:** standard-library truth checks first, then pilot-sized models. **Acceptance:** exact agreement with enumeration; no false optimum status; no artificial compiler distinction. **Next:** if the compilers coincide, record the null result and stop the comparative speedup branch. If a later distinct transformation exists, specify its mechanism and discrimination test before WP7 timing.

### WP7 — Resource characterization and conditional controlled comparison

**Objective/risk:** establish what the actual laptop can execute and avoid depth-only efficiency claims. **Dependencies:** validated semantics and constructors; WP6 only for solver comparisons.

Measure construction, evaluation, certificate verification, and optional compilation/search phases in isolated subprocesses. Record object/gate/wire/raw-entry/event counts, file bytes, peak RSS, and wall time. Include original and branch-selected fixtures with their different construction policies. Increase scale only to resolve a specified growth or branch question. Freeze any confirmatory timing design using Section 5.5; do not time two canonically identical formulas as different architectures.

**Outputs:** hardware-specific capacity table, phase costs, resource forecast with residual uncertainty, optional paired benchmark report. **Acceptance:** observed peaks fit the declared budget; projected growth is checked on another informative size; all costs and failures are retained. **Next:** WP8. Resource limitation narrows executable claims; it does not falsify the uniform theorem. A generic baseline matching or winning terminates the corresponding improvement claim.

### WP8 — Final prior-art, scientific, and execution audit

**Objective/risk:** prevent a correct implementation from becoming an overstated contribution. **Dependencies:** proof, results, full source ledger.

Compare the exact theorem statement with the known federation/quota operators and neighboring relation-identity results. Resolve any newly encountered exact match before a novelty assertion. Mark the unavailable older sources explicitly. Recheck all claims against the correct baseline, metric, and evidence type. Walk through the repository as if only this plan and its artifacts were available. Verify interruption/resume consistency and confirm that a missing optional solver does not block the core.

**Outputs:** `audit.md`, claim/evidence matrix, residual-prerequisite list. **Acceptance:** no unaddressed defect that changes correctness or interpretation. External expert review is a prerequisite for confident publication priority, not a fabricated completed step. **Next:** revise affected work packages and rerun their dependent checks; otherwise WP9.

### WP9 — Synthesize and decide

**Objective:** produce an interpretable outcome including negative findings. **Dependencies:** WP8.

Generate a standalone research report with theorem statement/proof, exact construction specification, reproduction extent, source/provenance ledger, raw result links, actual M3 capacity measurements, all failures, and the continuation decision from Section 10. Include a single-command reproduction manifest after the proposed CLI exists. Separate known results, deductions, observations, estimates, and open questions.

**Acceptance:** another agent can regenerate the main tables and inspect every claim without chat history; the report does not call a finite test a proof, a heuristic minimum, a known operator new, or a slow baseline state of the art. **Outcome:** continue, narrow, revise, or stop with the recorded evidence. No positive result is required for completion.

## 8. Ablations, adversarial scenarios, and failure analysis

| Challenge | Required test or comparison | Interpretation and response |
|---|---|---|
| Exact result already exists under another name | Map its quantifiers, restriction operation, depth, arity, block cap, and sharpness to this statement | Reclassify as replication/corollary; do not rename it as a new architecture |
| Strongest published method is not locally runnable | Separate its task/hardware/license from the local comparison; use the strongest faithful local generic method | Limit direct claims; no incomparable published-number win |
| Available memory is low | Keep implicit singletons, one process, small branch-complete fixtures, symbolic recurrence counts | Core survives; large materialization/solver claims may remain untested |
| Critical package operation is unavailable | Run the standard-library evaluator and exhaustive small oracle; test an alternative backend only under amendment | Preserve exact semantics; no approximate kernel substitution |
| Asset license or access is unclear | Use the self-contained generators; retain source assets locally only as authorized | Do not invent redistribution rights or make unavailable data a silent dependency |
| Synthetic result does not transfer | Require naturally bounded raw partitions and a real user decision before an application claim | Keep the theorem result; withdraw application benefit claims |
| Small tests do not express the mechanism | Cover pair/triple divergence, multiple exclusive supports, and merge transition | An uninformative toy failure is not a theorem rejection; a valid counterexample is |
| Tiny or unstable speed gain | Canonical formula check, equal preprocessing, noise-calibrated intervals, phase costs | Report no established practical advantage if indistinguishable or erased |
| Primary gain hides regression | Report certified-solve fraction, memory, preprocessing, and per-family failures | Describe the tradeoff; do not declare overall superiority |
| Resume state is inconsistent | Match code/profile/protocol hashes, compare resumed and clean deterministic outputs | Restart a new lineage; never combine incompatible partial results |
| Quota convention is off by one | Explicit accepted-edge truth tables at support `q-1,q,q+1` | Correct schema/convention and rerun all dependent results |
| Raw block restriction is replaced by graph deletion | Triple `{a,b,c}` with `c` removed must still relate `a,b` | Treat any mismatch as a semantic bug, not a model discovery |
| Intermediate output is restricted instead of recomputed | Three-object path gadget; delete its middle object | Must lose endpoint relation; otherwise reject the evaluator |
| Child multiplicities are erased | Test `T(P,P,Q,Q)` and `T(G,G,P,Q)` identities | Fix positional vote counting |
| P and Q extremizers are conflated | 4-of-5 depth 2 has different exact extrema and different contracts | Correct builder types; retain separate tables |
| Private vertices accidentally shared | Compare generated IDs and counts with constructor recursion | Fix naming; a smaller witness caused by reuse does not refute the upper bound |
| Proof cost is added per occurrence | Shared-object alternative witnesses, plus Set Cover translation | Use globally coupled presence variables and set unions |
| General quota is tested by enumerating exponentially many coalitions | Compare support counting with a tiny exhaustive-coalition oracle | Use counts/native threshold encoding; coalition enumeration is only a small check |
| A cyclic SAT encoding invents reachability | Negative query with no supporting path; compare full truth tables | Use the acyclic two-hop/vertex-elimination DAG; a recursive alternative must enforce least-reachability semantics via a sound bounded-stage or ranking encoding |
| Many blocks or DAG sharing change costs | Disjoint unions, shared subcircuits, tree-unrolled controls | Report event count and memory; gate count alone is not state size |
| Worst-case context is mistaken for typical behavior | Unconditioned random-profile positive-rate report | Keep worst-case theorem separate from distributional observations |
| A theorem is correct but novelty/value is weak | Exact prior-art comparison and resource-normalized report | A documented replication/null result is valid completion |

Mutation tests are tests of the harness, not extra production features. No broad fault-injection framework is required beyond these concrete semantic risks.

## 9. Reproducibility, repository, and result schema

### 9.1 Minimal repository layout

All paths below are **proposed project paths**, not files already delivered with this plan.

| Path | Purpose |
|---|---|
| `README.md` | Setup, actual commands after implementation, and claim limits |
| `pyproject.toml` | Package entry point and supported interpreter range |
| `requirements-solver.lock` | Optional solver and transitive dependency hashes |
| `src/partition_poc/__init__.py` | Package initialization |
| `src/partition_poc/__main__.py` | Dispatch `python -m partition_poc` to `cli.main()` |
| `src/partition_poc/model.py` | Schema, validation, canonical partitions |
| `src/partition_poc/eval_reference.py` | Independent small-instance oracle |
| `src/partition_poc/eval_sparse.py` | Sparse exact evaluator |
| `src/partition_poc/bounds.py` | Classification, recurrences, transition |
| `src/partition_poc/gadgets.py` | P/Q, path, raw-guard, and cover constructors |
| `src/partition_poc/witness.py` | Enumeration, criticality certificates, greedy deletion |
| `src/partition_poc/provenance.py` | Optional shared Boolean DAG and generic closure |
| `src/partition_poc/solve.py` | Optional exact backend adapter and status handling |
| `src/partition_poc/cli.py` | Config-driven commands; no implicit cloud access |
| `tests/` | Semantic, constructor, mutation, and resume tests |
| `configs/` | Frozen JSON protocols and generator specifications |
| `vendor/original/` | Immutable attachment bytes, if available |
| `fixtures/` | Canonical generated profiles with manifests |
| `artifacts/runs/` | Append-only result records and failed runs |
| `artifacts/certificates/` | Deletion verdicts, optima, solver models as applicable |
| `research/` | Proof, source ledger, amendments, audits, decisions |
| `reports/` | Regenerated tables and final findings |

Proposed CLI surface, to implement and test rather than assume exists. Configure package discovery for the `src/` layout in `pyproject.toml`, implement `__main__.py` with `from .cli import main` and `raise SystemExit(main())`, and install the project into the validated environment. Run these commands from the repository root:

```bash
.venv/bin/python -m pip install -e .
.venv/bin/python -m partition_poc env --output artifacts/environment.json
.venv/bin/python -m partition_poc replicate --source vendor/original --output artifacts/replication
.venv/bin/python -m partition_poc bounds --m 5 --q 4 --depth 4
.venv/bin/python -m partition_poc build --m 5 --q 4 --depth 2 --kind pair --output fixtures/q4m5d2p.json
.venv/bin/python -m partition_poc verify --input fixtures/q4m5d2p.json --mode critical
.venv/bin/python -m partition_poc run --config configs/confirmation.json
.venv/bin/python -m partition_poc report --manifest artifacts/runs/manifest.json
```

Expose `--resume` only after implementing the hash checks. No command should silently change method settings to obtain a successful result.

### 9.2 Configuration and experiment records

Use JSON rather than adding a YAML dependency. A protocol must specify schema version, family, `m,q,d`, block bound, query kind, generator distribution, seed, constructor tie policy, exact/heuristic method, evaluator, resource policy, and expected mathematical obligation. Randomized configurations must distinguish profile-generation seed, timing-order seed, and solver seed.

Each JSONL record must include at least:

```json
{
  "schema_version": 1,
  "run_id": "hash-derived identifier",
  "phase": "exploration or confirmation",
  "profile_sha256": "exact saved-profile hash",
  "protocol_sha256": "exact frozen-protocol hash",
  "git_commit": "actual source revision",
  "environment_id": "linked environment record",
  "family": "extremal_quota",
  "m": 5,
  "q": 4,
  "depth": 2,
  "query_kind": "pair",
  "n_vertices": 6,
  "n_gates": 5,
  "method": "single_deletion_certificate",
  "status": "verified_critical",
  "witness_size": 6,
  "lower_bound": 6,
  "upper_bound": 6,
  "timing_seconds": {},
  "peak_rss_bytes": null,
  "failure_reason": null,
  "resume_parent_run_id": null
}
```

This illustrates field meanings; hash strings and timing values are placeholders in the schema, not actual results. Use explicit statuses such as `invalid_input`, `negative_full_query`, `verified_critical`, `optimal`, `feasible_only`, `unknown`, `timeout`, `memory_limit`, `semantic_disagreement`, and `counterexample`. A full positive query must always have the full domain as a feasible witness; an exact model declaring it infeasible is an implementation defect.

Save witnesses as sorted object IDs, the objective definition, raw solver status, and validation verdict. Preserve raw stdout/stderr and exact parameters. Record elapsed time separately for generation, parsing, full evaluation, compilation, solving, and verification. All displayed tables are derived from raw records; never hand-edit a result table to hide a failure.

### 9.3 Source integrity and standalone reproduction

The supplied PDF matched byte-for-byte the PDF inside the archive in this investigation. Relevant attachment hashes:

| Artifact | SHA-256 |
|---|---|
| `partition_network_reproducibility.zip` | `4f6cb5304f475908ca52e1c177519cc92837733251f5764acd57beaacd039813` |
| `partition_network_research.pdf` | `f1c0fd202e961b2142d2f0fa851e3041c3109883f112c9230c99f04079ed9e4b` |
| Main LaTeX source inside archive | `10f1c8619adb0b06556b17baa9fd394ca7f9d9e965964b010248d6563e5f45bf` |
| Main supplied checker | `0363e4e9520c051b09456fa43cbfe1685e8617c0fa180b18ddfe06fc1f46bcd4` |

Hashes identify the inspected artifacts; they do not establish authorship or correctness. If any input differs, record the difference before comparing results. Pin source revisions for reused code and retain license text. Record failed runs, methodological amendments, and decision dates alongside successful runs.

The final project report must reproduce the theorem and essential algorithms, not simply link to this conversation or to temporary research notes. A coding agent can recreate the selected construction from Sections 4.2–4.4 even without the attached archive.

## 10. Outcome interpretation and evidence-triggered continuation

The general law is a direct extension of the supplied overlap-counting method, with a useful pair/triple distinction. Even if historically original, it may be best positioned as a concise mathematical contribution. Neither its breadth nor the amount of checking makes it a new computational architecture or a demonstrated practical breakthrough.

| Observed outcome | Defensible conclusion | Next action |
|---|---|---|
| Upper proof and sharp constructors survive; no equivalent source located | Candidate exact quota-depth theorem with reproducible corroboration | Prepare precise theorem/related-work manuscript; seek specialist assessment of priority through a separately authorized channel |
| An equivalent theorem is found | Replication, reformulation, or corollary | Credit it and identify whether any nontrivial extension remains; otherwise stop the novelty project |
| Upper proof valid, proposed extremizers fail | A bound may survive; exactness unresolved | Reduce failing construction, repair contracts or narrow statement; do not publish exact values as proved |
| Correct evaluator finds a valid violation of upper bound | The theorem statement or proof is wrong | Preserve minimal counterexample; locate failed lemma before changing scale |
| Evaluators disagree | Implementation/evaluation defect | Stop dependent results, keep both traces, repair and revalidate |
| Finite checks pass but a proof step remains unresolved | Computational evidence only | State the open implication and solve it analytically; more random tests do not close it |
| Compiler formulas coincide with generic preprocessing | No distinct compiler mechanism | Keep one conventional implementation; stop comparative speedup claim |
| Solver is slow or times out on cover-derived cases | Consistent with hard minimum optimization, not a theorem refutation | Report gaps/resources; use exact small controls and certificates on extremizers |
| Resource-normalized analysis shows no advantage | Depth/context hierarchy does not imply efficiency | Preserve theoretical result; drop efficiency positioning |
| Actual application data violates the block cap | Current theorem does not apply to that application | Do not split blocks artificially to claim applicability; redesign or abandon application branch |

### The single prioritized extension if the primary result is known or complete

Investigate **four-object blocks under a quota that preserves four**, beginning with the smallest near-unanimous case `q=4,m=5`, whose Nakamura number is five. This preserves the original context question and introduces more accepted-tree shapes and terminal obligations. It is a theoretical extension, not a workaround for a negative application result.

Entry requires the size-three proof and verifier to be stable and either (a) a confirmed prior-art match that makes the primary result a replication, or (b) a completed size-three result with a specific reason that four-object interactions add information. First enumerate connected accepted graphs on four labeled vertices and feasible input partitions of those vertices. Track obligations as terminal partitions/connectivity patterns, not merely cardinalities `P,Q,R`; different four-terminal patterns may require different certificates. Derive a sound upper-bound state transition and search for a sharp compositional construction on the smallest depth where it differs from a coarse path bound.

Do not presume the two-state recurrence generalizes. Stop or narrow if the required state abstraction loses sharing information, a purported lower construction cannot realize simultaneous obligations, or the only new finding is already a direct known result. Use the existing exact evaluator and small solver only to find counterexamples or candidate patterns. A new universal equality still needs a proof. Broader learning, entity-resolution deployment, or the group appendix requires a separately justified research question and remains deferred.

## 11. Work performed now, audits, and remaining limitations

### 11.1 Evidence actually obtained during this investigation

The research environment was **Linux x86_64, CPython 3.12.14**, not the user's Mac. The supplied PDF was read alongside its LaTeX source; the first page was visually inspected and the standalone PDF matched the archived PDF. All four supplied scripts were inspected before execution. Separate small evaluators and probes were written for the extension and compiler comparison. No training, target-laptop run, production implementation, solver benchmark, or real application evaluation was performed.

| Check performed | Extent | Result and limit |
|---|---|---|
| Supplied main checker | 2,549,708 bounded-side cases; 1,342 unbounded paths; 152 bounded sharp paths; 250,051 endpoint-containing restrictions | Assertions passed; deterministic saved fields matched, excluding elapsed time |
| Supplied additional evaluators | Fourfold evaluator through depth 5; raw-guard checks; group-word checker with 5,766 cyclic-word cases | Passed; group appendix remains a separate project |
| General-quota P/Q constructors | All safe nonunanimous quotas for `m=4,…,8`, depths 1–3, both query kinds: 42 cases | 3,853 restrictions passed, including intermediate block-bound checks |
| Restriction detail for the preceding row | 3,489 exhaustive restrictions when there were at most 11 nonterminals; 364 full/single-deletion evaluations otherwise | Every tested construction was critical; finite corroboration, not a universal proof |
| Arithmetic crossover/post-crossover checks | All safe quotas for `m=4,…,64`, through six post-crossover depths: 8,058 checks | Exact-integer agreement |
| Independent Set Cover fixtures | Three explicit cover systems, 176 assignments | Truth/optimum mapping passed; six-vertex fixture exhibited minimum 3 versus greedy minimal 5 |
| Sparse versus direct circuit evaluation | 100 six-vertex, eight-raw/eight-gate **three-of-four** circuits; all 64 retained subsets per circuit | 6,400 masks, representing 1,536,000 node/pair truth comparisons, all agreed |
| Specialized versus generic compiler | 3,665 full-domain related pair events in those circuits, including raw-node events | All normalized formulas identical; these are dependent conformance checks, not 3,665 independent trials or a timing result |
| Generated cover reduction probe | 50 six-variable monotone CNFs, all 64 assignments each | 3,200 truth checks and exact witness/cover minima agreed |

The supplied main checker took approximately 4.01 seconds, the general-quota construction checks approximately 15.41 seconds, and the combined semantic/compiler/cover probe approximately 1.44 seconds in this environment. These are recorded execution context, not controlled comparative measurements, and cannot certify M3 timing or memory requirements.

The generalized-construction coverage is completely specified by Section 4's direct-tie policy, the `(m,q,d,kind)` range above, and the exhaustive-versus-single-deletion rule. The semantic/compiler probe used `random.Random(593118)`, six vertices, eight shuffled-block raw partitions, and eight gates. Gate index modulo three selected a `(a,a,b,b)` pattern, a `(a,a,b,c)` pattern, or four unrestricted earlier-node positions, after drawing four positions independently. This historical probe was for `3-of-4`; the homogeneous general-quota confirmation generator is specified separately in Section 5.3. The same RNG stream then generated 50 CNFs, each with 2–8 clauses and 1–6 distinct sampled variables per clause. These recipes describe the observed exploratory work; independent confirmation must use fresh cases.

### 11.2 Three distinct challenge audits and revisions

The draft underwent separately tasked mathematical/prior-art, experimental-validity, and implementation reviews, followed by targeted corrections. These were model-assisted reviews within this investigation, **not external peer review or formal proof certification**.

| Audit | Material weakness found | Revision made | Residual issue |
|---|---|---|---|
| Scientific/prior-art | The exact gate family is an existing federation operator, and exact min-ones provenance is established | Reframed contribution as a sharp object-restriction depth law; added closest sources | Historical equivalence to older results remains open |
| Scientific | A specialized two-hop compiler appeared novel before generic block pruning | Analytic equivalence and 3,665 formula matches retained as a null result; timing branch gated | No practical compiler improvement established |
| Mathematical | P/Q criticality contracts could be conflated | Separate constructor types and sharpness proofs; 4-of-5 divergence fixture | Uniform proof not proof-assistant formalized |
| Mathematical | “First merge” wording ignored equality already at depth one | Defined the persistent crossover at depth at least two | No remaining arithmetic issue found in reviewed formulas |
| Experimental | A greedy irreducible set could be misreported as minimum | Explicit trap fixture, exact oracle, solver statuses, objective separation | Large general instances may remain unproved within resources |
| Experimental | All-required extremizers could yield an artificial solver win | Mandatory-deletion certificate and alternative-rich cover controls; no standalone performance claim from extremizers | Real application workload absent |
| Implementation | Two-hop code might be applied to an unbounded/boundary case | Explicit full-domain block-size guard and general/reject fallback | General large-block compiler optional |
| Implementation | Random tests could accidentally mix four-input and other-arity gates | Homogeneous general-quota repetition patterns and exact role cycle | Generator is not uniform over partitions; explicitly disclosed |
| Implementation | Memory formula omitted empty-node overhead and incorrectly removed logs without a solver | Added node/live-record terms and corrected solver/provenance branches | Actual Mac peaks require measurement |
| Implementation | Package setup, interpreter path, negative queries, triple root, domain representation, and large-witness verification were underspecified | Added package entry point/install, exact path selection, explicit statuses/root formula, retained-domain result, independent sparse verifier | Local package smoke test remains necessary |
| Standalone handoff | Non-distributivity and raw-guard controls depended on the archive | Included complete constructions in Section 5.2 | Original saved-report reproduction still requires the original archive |
| Measurement | Solver-only limits and correctness failures could contaminate end-to-end timing | Outer deadline; invalid witnesses/models block confirmation | Any future practical effect needs an independently frozen comparison |

Remaining risk is not removed by the number of checks. The proofs depend on the exact restriction model, input-position multiplicity, raw block cap, and homogeneous gate family. The source archive itself was generated as a research note and is not independent published validation. The quota extension is closely related to the supplied argument and may be a modest theorem-level extension even if original. Worst-case certificates do not show typical behavior, learning dynamics, optimization advantage, information compression, or real-world causal explanation.

### 11.3 Delivery and claim checklist

The intended final project report should be accepted only when each central claim has a proof or appropriate measured evidence; every minimum has an exact certificate/status; each expensive task was preceded by a relevant gate; all source/version and failure records exist; and the actual laptop's constraints are reflected in the executed manifest. This plan itself supplies the mathematical specification, research findings, small validation results, and executable work order. The future project commands and M3 measurements are deliberately marked unperformed.

## 12. References and search-coverage record

References use stable bibliographic identifiers and ordinary URLs. Access descriptions refer to this investigation. A source that was only discovered or partly accessible is not described as fully read. Primary papers and official software documentation ground consequential technical claims; the author survey is also direct evidence for its authors' federation definition and points back to the original priority source.

### Supplied source

**[A1]** *Exact Context Bounds for Partition-Valued Networks: A mathematical research investigation prompted by computation without scalar states.* “ChatGPT research session,” dated 30 October 2026. User-supplied `partition_network_research.pdf`, 12 pages, and `partition_network_reproducibility.zip`. Main LaTeX, appendix, all four Python checkers, and saved reports inspected; checks rerun. No public DOI, publication status, or redistribution license was supplied. Exact identity is recorded in Section 9.3. Supports the original threshold, fixed-gate depth theorem, raw-guard theorem, and separate group appendix; it is the work being investigated, not external confirmation.

### Prior art

**[R1]** Olivier Hudry and Bernard Monjardet. *Consensus Theories. An Oriented Survey.* Mathématiques et Sciences humaines 190 (2010), 139–167. [Author PDF](https://micro.econ.kit.edu/downloads/Monjardet_ConsensusTheories.pdf). Inspected federation/quota definitions and the lattice formula, especially printed pp.144–145 and 153. Establishes exact primitive overlap; it does not supply the selected depth law in the inspected discussion. Original antecedent: Bernard Monjardet, *Arrowian characterizations of latticial federation consensus functions*, Mathematical Social Sciences 20(1) (1990), 51–71, [DOI](https://doi.org/10.1016/0165-4896(90)90077-K); original full text was not accessible.

**[R2]** Bruno Leclerc. *The median procedure in the semilattice of orders.* Discrete Applied Mathematics 127(2) (2003), 285–302. [DOI](https://doi.org/10.1016/S0166-218X(02)00211-1). Publisher-indexed primary text in Sections 3.2–3.3 inspected for quota/federation definitions; direct full-text opens failed. Supports established operator terminology; no full-paper novelty clearance is claimed.

**[R3]** Dinko Dimitrov, Thierry Marchant, and Debasis Mishra. *Separability and Aggregation of Equivalence Relations.* Economic Theory 51 (2012), 191–212. [Author manuscript, 18 October 2010](https://users.ugent.be/~tmarchan/joinmeet.pdf). Definitions and join/meet characterization theorems inspected. Same objects and lattice operations, different axiomatic target. Fishburn–Rubinstein's 1986 result was traced through this and R4/R5; the original was not fully inspected.

**[R4]** Ulle Endriss and Umberto Grandi. *Graph Aggregation.* Artificial Intelligence 245 (2017), 86–114. [arXiv:1609.03765](https://arxiv.org/abs/1609.03765). Opened full preprint; inspected framework, rule definitions, and equivalence-aggregation discussion. Supports the issue-independence/closure distinction, not the exact witness recurrence.

**[R5]** Yuval Filmus. *Classification aggregation: a quantitative impossibility theorem.* arXiv:2605.17136v2, 20 May 2026. [Full text](https://arxiv.org/html/2605.17136v2). Sections 1.3 and 4 inspected. Exact/approximate issuewise equivalence preservation differs from threshold followed by closure; recent overlap check, not a direct contradiction.

**[R6]** Josep Freixas and Sascha Kurz. *Bounds for the Nakamura Number.* Social Choice and Welfare 52 (2019), 607–634. [DOI](https://doi.org/10.1007/s00355-018-1164-y); [author manuscript](https://epub.uni-bayreuth.de/3629/1/bounds_nakamura_arxiv.pdf). Definition, minimal-coalition formulation, and quota formula inspected. The invariant is established; the supplied partition interpretation is separately assessed.

**[R7]** Dmitry Maximov, Vladimir I. Goncharenko, and Yury S. Legovich. *Multi-Valued Neural Networks I: A Multi-Valued Associative Memory.* Neural Computing and Applications 33 (2021), 10189–10198; corrected preprint [arXiv:2302.11909](https://arxiv.org/abs/2302.11909), 2023. Full preprint opened, lattice-state definitions and assumptions inspected. Establishes nonnumeric network precedent; its Brouwer-lattice assumptions and associative-memory task are not the present theorem.

**[R8]** Daniel Deutch, Tova Milo, Sudeepa Roy, and Val Tannen. *Circuits for Datalog Provenance.* ICDT 2014, 201–212. [DOI](https://doi.org/10.5441/002/icdt.2014.22); [author PDF](https://www.cs.tau.ac.il/~milo/projects/bpq/papers/icdt14b.pdf). Positive Boolean circuits, shared construction, and semi-naive evaluation inspected. Its polynomial representation statements concern fixed-query data complexity; do not transfer them to arbitrary circuit depth.

**[R9]** Zhengjie Miao, Sudeepa Roy, and Jun Yang. *Explaining Wrong Queries Using Small Examples.* SIGMOD 2019, 503–520. [DOI](https://doi.org/10.1145/3299869.3319866); [full author manuscript](https://pmc.ncbi.nlm.nih.gov/articles/PMC6705612/). Definitions, Section 4, Algorithms 1–2, and implementation/experiments inspected. The min-ones provenance architecture is a close predecessor. Tuple-deletion counterexamples and the historical Z3 implementation are not a direct run of the partition task.

**[R10]** Masood Feyzbakhsh Rankooh and Jussi Rintanen. *Propositional Encodings of Acyclicity and Reachability by Using Vertex Elimination.* AAAI 2022. [DOI](https://doi.org/10.1609/aaai.v36i5.20530); [author PDF](https://users.aalto.fi/~rintanj1/papers/RankoohRintanen22aaaiVE.pdf). Reachability encodings, soundness/completeness, and structural cost inspected. Informs a strong generic baseline. The small compiler implemented here adapts the vertex-elimination principle; the authors' software was not executed.

**[R11]** Marco Calautti, Mostafa Milani, and Andreas Pieris. *Diverse Why-Provenance for Datalog Queries.* arXiv:2610.24707v1, 17 October 2026. [Full text](https://arxiv.org/html/2610.24707v1). Graph/hypergraph encodings, implementation, and experiments inspected. Objective is diversity among explanations. Reported server/Gurobi conditions are not controlled M3 comparisons. Research repository was identified but not built or license-audited.

**[R12]** Patrick Rodler. *A formal proof and simple explanation of the QuickXplain algorithm.* Artificial Intelligence Review 55 (2022), 6185–6206. [DOI](https://doi.org/10.1007/s10462-022-10149-w); [full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC9622537/). Monotone-predicate and irreducible-subset formulation inspected. Supports the minimal-versus-minimum distinction; not an exact cardinality baseline.

**[R13]** Paolo Lipparini. *The distributivity spectrum of Baker's variety.* Journal of the Australian Mathematical Society 110(1) (2021), 119–144; first online 2020. [DOI](https://doi.org/10.1017/S1446788720000373); [preprint](https://arxiv.org/abs/1709.05721). Publisher full text, definitions, Theorem 2.1, and relevant near-unanimity relation identities inspected. Close algebraic neighborhood, different carrier/relation quantifiers. The quaternary lattice term discussed there must not be assumed equal to the present term on nondistributive partition lattices.

**[R14]** Jyrko Correa-Morris. *Reduction of Search Space for the Mean Partition Problem.* WSEAS Transactions on Mathematics 22 (2023), published 20 October 2023. [Publisher page](https://www.wseas.com/journals/articles.php?id=8456); [full PDF](https://wseas.com/journals/mathematics/2023/b645106-041%282023%29.pdf). Partition quota-rule definition and references inspected. Exact gate-level precedent; the paper's consensus-quality discussion is not evidence of benefits for the current circuit architecture.

**[R15]** P. Curtz and contributors. *A321358: a(n)=(2·4^n+7)/3.* The On-Line Encyclopedia of Integer Sequences, entry begun 2018; inspected 30 October 2026. [Entry](https://oeis.org/A321358). Confirms the original numeric sequence already exists; no partition-circuit interpretation was found in the inspected entry.

### Software and hardware sources

**[R16]** Google. *OR-Tools package metadata and release files*, release 9.15.6755, 14 January 2026. [PyPI](https://pypi.org/project/ortools/). Inspected release, license, Python requirements, and native macOS arm64 wheels. This supports a dependency choice, not measured laptop compatibility or optimal solver performance.

**[R17]** Google. *Using OR-Tools Pip for Python on macOS.* [Official installation documentation](https://developers.google.com/optimization/install/python/pkg_mac). Inspected tested Intel/M1 macOS configurations. The example branch is older than the selected package release; pinning follows R16 and still requires a local smoke test.

**[R18]** Google. *CP-SAT Solver.* [Official documentation](https://developers.google.com/optimization/cp/cp_solver). Integer modeling and the OPTIMAL/FEASIBLE/INFEASIBLE/MODEL_INVALID/UNKNOWN distinction inspected. A solver status is only as meaningful as the validated encoding and objective.

**[R19]** Python Software Foundation. *Using Python on macOS.* [Official documentation](https://docs.python.org/3/using/mac.html). Native universal2 distribution and system-Python guidance inspected. Exact local interpreter selection remains a discovery task.

**[R20]** Apple. *MacBook Pro (16-inch, November 2023): Technical Specifications.* [Official specifications](https://support.apple.com/en-us/117737). M3 Max CPU/memory variants inspected. The user's actual machine may be another chassis/configuration; discovery takes precedence.

**[R21]** Python issue tracker. *Issue 30513: getrusage returns platform-dependent value*, 2017, migrated 2022. [Primary project issue](https://bugs.python.org/issue30513); [resource documentation](https://docs.python.org/3/library/resource.html). Indexed issue text and the resource documentation were inspected; direct retrieval of the issue failed. Used for the RSS-unit caveat. Verify local behavior and preserve raw units rather than relying on a cross-platform default.

### Search coverage, saturation, and blind spots

The search expanded beyond “partition network” into federation consensus, lattice polynomial quota rules, equivalence aggregation, block preservation, empty-intersection winning coalitions, Nakamura numbers, near-unanimity, local consistency, relation-composition identities, witness/certificate/support/locality terminology, Datalog provenance, min-ones optimization, and vertex-elimination reachability. Both general search and targeted primary-source retrieval were used. Citation tracing went backward to 1980s/1990s consensus and forward to 2024–October 2026 provenance and aggregation papers. Exact-formula and terminology-neighborhood searches were repeated after selecting the quota extension.

The materially different discoveries were the exact federation-operator precedent, established min-ones witness optimization, the strong generic closure baseline, and the distinction from fixed-carrier near-unanimity identities. Further targeted searches did not identify a different exact vertex-restriction depth law or a stronger directly matching mathematical result. This is a stopping point for this bounded investigation, not proof that no such source exists.

Important remaining gaps are the original Monjardet 1990 paper; Hans-Jürgen Bandelt and Gerasimos C. Meletiou, *An algebraic setting for near-unanimity consensus*, Order 7 (1990), 169–178, whose metadata/citation trail were checked but full text was not inspected; portions of older French consensus literature; and specialist equivalence to universal-algebra locality results. Original Baker–Pixley and every source cited by the supplied group appendix were not independently audited in full. No software repository other than the supplied archive and newly written probes was executed as part of this investigation.

A limited patent/deployed-system search produced adjacent lattice-network and equivalence/record-linkage material, but no patent result was used to establish novelty or implementation freedom. The task is presently a mathematical characterization, not a product-clearance exercise. There is no evidence here of a deployed system using the proposed exact witness law.

Before a publication-priority assertion, compare the full theorem and critical constructions with an expert familiar with lattice consensus and relation identities. The question is specific: whether deleting the carrier objects and recomputing yields an existing bounded-support/depth theorem equivalent to the recurrence. This unresolved comparison should not be delegated as a substitute for the completed research in this plan; it is the remaining specialist review gate.

[R1]: https://micro.econ.kit.edu/downloads/Monjardet_ConsensusTheories.pdf
[R2]: https://doi.org/10.1016/S0166-218X(02)00211-1
[R3]: https://users.ugent.be/~tmarchan/joinmeet.pdf
[R4]: https://arxiv.org/abs/1609.03765
[R5]: https://arxiv.org/html/2605.17136v2
[R6]: https://doi.org/10.1007/s00355-018-1164-y
[R7]: https://arxiv.org/abs/2302.11909
[R8]: https://doi.org/10.5441/002/icdt.2014.22
[R9]: https://doi.org/10.1145/3299869.3319866
[R10]: https://doi.org/10.1609/aaai.v36i5.20530
[R11]: https://arxiv.org/html/2610.24707v1
[R12]: https://doi.org/10.1007/s10462-022-10149-w
[R13]: https://doi.org/10.1017/S1446788720000373
[R14]: https://www.wseas.com/journals/articles.php?id=8456
[R15]: https://oeis.org/A321358
[R16]: https://pypi.org/project/ortools/
[R17]: https://developers.google.com/optimization/install/python/pkg_mac
[R18]: https://developers.google.com/optimization/cp/cp_solver
[R19]: https://docs.python.org/3/using/mac.html
[R20]: https://support.apple.com/en-us/117737
[R21]: https://bugs.python.org/issue30513




