# POC execution brief: when shuffled channel data identify chronology

**Recipient:** an autonomous coding agent configured as `gpt-6.1-sol` with `xhigh` reasoning.  
**Research decision and literature cutoff:** 29 October 2026.  
**Deliverable from you:** a small, independently checked, reproducible research result on the target MacBook with M3 Max.  
**Status of this document:** an execution instruction informed by proof review and small planning probes; it is not a report of a completed PoC on that Mac.

## 1. Project decision

### 1.1 Execute one exact identifiability study

Determine whether complete **reference-to-checkpoint channel data**, together with their recovery statistics, identify the chronological order of shuffled observations when reversible dynamics and hidden environmental memory are allowed.

The selected project is an **exact diagnostic of the boundary between informational ordering and actual chronology**. Implement and independently verify the counterexample and controls specified below. Do not develop a new chronological classifier or a large benchmark unless this brief's central derivation is found invalid. If it is invalid and the bounded repair cannot resolve the defect, report that outcome; do not invent a replacement research project.

The expected, useful result is a rigorously scoped limitation:

> Several physically admissible histories can produce exactly the same allowed observations while visiting the same checkpoints in different orders. The paper's postprocessing relation identifies possible information degradation, not the realized chronological sequence. Under an additional promise of an invertible, divisible observed-system history, an elementary determinant score already orders every strict loss step.

This is a statement to verify, not an answer to hard-code. You must implement the explicit finite mechanisms, check their admissibility and observational equivalence, and preserve the evidence. An exact obstruction can complete the research objective without an empirical accuracy contest.

### 1.2 Why this is the right revision of the original problem

The supplied paper studies
\[
A\preceq B \quad\Longleftrightarrow\quad B=QA
\]
for some stochastic postprocessor \(Q\). This is the established Blackwell comparison of experiments. The paper itself cautions against interpreting this existential simulation relation as actual causal precedence [S1, §§1, 12; R1, Definition 3 and Theorem 4].

Consequently:

1. On identical full-channel data, the proposed relation and a complete Blackwell baseline have identical information. A disagreement is a correctness problem, not evidence of discovery.
2. The paper's global obstruction to finitely many monotone scores does not imply an advantage in sorting one realized chain. Section 4.2 proves why a single determinant score suffices on an invertible divisible chain, modulo reversible relabelings.
3. Environmental memory can restore information to the observed system. A transition that fails a system-only garbling test may still occur under a bijection of the system and its environment.
4. Ordered-trajectory time-reversal classification, as studied by Roldán et al. and Seif et al., has different inputs and assumptions [R4, R5]. It cannot fairly be benchmarked by substituting one-time marginal entropy for path entropy production.

The revision preserves the central practical question: **what direction, if any, is identifiable from physical transformation data after ordering information is removed?** It deliberately stops short of estimating elapsed times, reconstructing an arbitrary photographed movie, proving a thermodynamic arrow, or resolving Loschmidt's paradox. These objectives require additional operational or physical assumptions.

### 1.3 Contribution and novelty boundary

The smallest defensible contribution is a reproducible **application-level limitation and conditional positive result**:

- An explicit finite, positive, invertible three-state example with identical unordered observations and multiple chronological histories.
- A distinction between actual chronology, a compatible divisible ordering, and thermodynamic forward/reverse inference.
- Exact verification that the proposed channel comparison equals its strongest established comparator.
- A control showing that retaining correlations involving the preparation reference and environment causes the apparent reversal of the system's informational ordering.

The conceptual ingredients are established. In particular, general realization of channel sequences through open-system evolution is already discussed in [R2, §II, PDF p. 2]. The particular finite construction and combined diagnostic were derived for this brief; **historical originality of this construction is not established**. Do not market it as a new physical law, a new Blackwell algorithm, or a publication-ready novelty claim. The source's fixed-\(3\times3\) monotone-dimension theorem may be a mathematical strengthening of prior work, but its historical novelty remains separate and unresolved [S1, §12; R6].

### 1.4 Primary question, hypothesis, and decision criterion

**Question.** Over the declared class of unknown reversible system–environment dynamics, does the allowed observation law determine a unique chronological permutation of three checkpoints?

**Hypothesis under challenge, \(H_{\mathrm{chron}}\).** Full endpoint channels and all their system-only recovery/postprocessing statistics suffice to identify the actual checkpoint order, without an additional direction-bearing assumption.

**Primary quantity.** The number
\[
N_{\mathrm{compatible}}(D)
=
\left|\{\pi:\text{an admissible model realizes order }\pi
\text{ and produces the same delivered data law }D\}\right|.
\]

The scientific threshold is exact: finding **two** distinct admissible orders with the same data law refutes universal identifiability under that data contract. The specified six-model construction should certify all \(3!=6\) orders. This threshold follows from the definition of identifiability, not from an observed gain.

For a uniform prior over those six models, any single-order estimator has success probability at most \(1/6\); for the specified opposite-direction pair, at most \(1/2\). These are analytic bounds under stated priors, not frequentist confidence intervals or claimed average performance on natural systems.

**Conditional positive result.** Under a separately supplied promise that the observed system follows an invertible divisible chain, the three distinct depolarizing channels have exactly one compatible order. The full channel comparator and determinant baseline must agree on it.

**Expected disappearance of identifiability:** hidden correlations, reversible relabelings, or missing ordering information.  
**Expected restoration in the chosen control:** a valid divisible-chain promise and distinct Blackwell equivalence classes.  
**Evidence against the proposed diagnostic:** an incorrect microscopic bijection, different delivered data across histories, or a supposedly hidden ordering variable reaching an estimator. Such a finding invalidates the attempted counterexample until repaired; it does not prove \(H_{\mathrm{chron}}\).

## 2. Operational data contract

### 2.1 What “complete transformation data” means here

Use a three-valued preparation label \(X\in\{0,1,2\}\). A laboratory can repeatedly prepare each value and identify the same checkpoint across repetitions. Each checkpoint is a distinct experimental condition with an arbitrary token, not a timestamp.

For checkpoint \(r\), supply the entire column-stochastic channel
\[
A_r(y,x)=\Pr(Y_r=y\mid X=x),\qquad A_r\in\mathbb Q^{3\times3}.
\]

All three preparations are available. Input and output labels have a common, known alignment. The environment is initially independent of \(X\). For entropy illustrations only, use \(\Pr(X=x)=1/3\).

The known preparation supplies a distinguished earlier boundary. The unknown chronology concerns the later checkpoints. This assumption is part of the observation protocol and must not be described as deriving every temporal orientation from entirely unstructured observations.

This is stronger than a bag of photographs or a single passive trajectory. A passive set of state distributions does not generally identify these channels. Do not claim the PoC solves that data-acquisition problem.

Give all methods the same matrices. Recovery error and postprocessing deficiency are functions of those matrices and supply no additional measured information:
\[
D(A,B)=\frac12\max_x\sum_y |A_{yx}-B_{yx}|,
\]
\[
\Delta(A,B)=\min_{Q\in\mathrm{St}_3}D(B,QA),\qquad
e(A)=\min_{R\in\mathrm{St}_3}D(RA,I).
\]
Here \(\mathrm{St}_3\) is the set of nonnegative \(3\times3\) matrices with every column summing to one. \(Q\) or \(R\) may use the observed system output only; they cannot consult \(X\), the hidden environment, or the true history.

### 2.2 Additional records and explicit exclusions

The primary proof also establishes equality of the **unoriented joint checkpoint law**
\[
J(y_1,y_2,y_3\mid x).
\]
Its coordinates refer to fixed checkpoint identities. Their order in a JSON object is not a chronological order. This is additional information beyond the endpoint channels, and equality must be verified separately.

| Information | Use in the study | Consequence |
|---|---|---|
| Complete \(A_r(y\mid x)\) | Available to every comparator | Main operational input |
| System-only \(\Delta\), \(e\), determinant, entropy scores | Computable from the same input | No privileged information |
| Unoriented joint law of fixed checkpoint variables | Checked as a stronger indistinguishability result | Still identical across the six histories |
| Full joint checkpoint endpoints \((S,E,r)\) for the 144-state construction | Certificate-level strengthening | Also identical across its six models; their transition laws differ |
| True chronological permutation, visit index, filename carrying the order | Evaluator only | Never estimator input |
| Known microscopic transition law together with its fully specified boundary preparation | Outside the primary unknown-law problem | Can distinguish histories; no failure claim here |
| Oriented transitions, intervention sequences, timestamps, signed durations, ordered paths | Excluded | These can carry the information the diagnostic removes |
| Append-only records with a known earlier-to-later retention rule | Excluded from the inference input | The retention rule already supplies a directed assumption |
| Energies, work, reservoir temperatures, thermal preparation, forward/reverse protocols | Not specified | No entropy-production or thermodynamic conclusion |

Do not quietly interpret a successful system-only postprocessor as the physical mechanism that actually occurred. Do not interpret its failure as a prohibition on all system–environment dynamics.

The 288-state control in §5.5 has a separate nuance: its microscopic law is fixed, but the initial hidden direction bit differs. Its indistinguishability statement excludes that bit. It does not claim nonidentifiability when both the complete microscopic law and complete initial environment state are supplied.

### 2.3 Three meanings of “order”

Use these terms consistently in code and reports:

- **Postprocessing order:** the mathematical relation \(A\preceq B\).
- **Divisible-compatible checkpoint order:** a permutation admitting stochastic maps between successive endpoint channels. It certifies the existence of a matching Markov realization, not the actual multitime coupling.
- **Actual chronology:** the sequence generated by a specified admissible system–environment model.

Buscemi–Datta's classical corollary concerns a Markov embedding with matching initial-to-time marginals. It does not identify the actual process as that embedding [R2, Corollary 1]. The distinction between divisibility and multitime Markovianity is also established [R3].

## 3. What the supplied mathematics supports

### 3.1 Source inventory and authority

The supplied assignment is `Pasted text(20261029-162651).txt`. The principal mathematical presentation is `research_note(1).html`. Its identical HTML and LaTeX source occur inside `research_reproducibility(1).zip` under `research_reproducibility/`.

The consolidated note is authoritative when supplemental working notes contain extra branches. During planning, the full principal source, README, three supplemental notes, four Python scripts, and two supplied JSON result files were inspected. The HTML copy was byte-identical to the archived HTML.

| Attachment material | Relevant content | Treatment |
|---|---|---|
| `research_note.tex` / `research_note.html` | Main theorems, proofs, scope, references | Primary mathematical source [S1] |
| `stochastic_proof_note.txt` | Recovery rigidity and nearly returning histories | Reviewed; secondary, not a chronology algorithm |
| `records_research_note.txt` | Retained records and cardinality bounds | Reviewed; assumptions already contain direction |
| `quantum_pauli_order_notes.txt` | Qubit extension, classical audit, literature boundaries | Reviewed; quantum implementation excluded |
| `quantum_pauli_order_verify.py` | Exact classical and qubit checks | Some small cases rerun during planning |
| `stochastic_audit_order.py` and its JSON | Generic rational inverse audit | Code and recorded result reviewed |
| `records_verify.py` | Integer flow/composition checks | Code reviewed; not rerun during planning |
| `stochastic_recovery.py` and its JSON | LP recovery and near-return checks | Code and reported results reviewed; not rerun |

The source reports 52,216 exact pair checks per main classical/qubit construction, 3,116 generic classical checks, 9,408 record comparisons, 13,955 record upper-bound checks, and numerical recovery experiments [S1, §11]. These are **source-reported results**, not newly reproduced results of this PoC.

### 3.2 Main family and executable comparison

The source uses
\[
P(u,v)=
\begin{pmatrix}
(1+2u)/3&(1-u)/3&(1-u)/3\\
(1-u)/3&(2+u+3v)/6&(2+u-3v)/6\\
(1-u)/3&(2+u-3v)/6&(2+u+3v)/6
\end{pmatrix},
\]
with
\[
0<u<1,\qquad 0<v<(2+u)/3.
\]

These matrices are strictly positive, symmetric, doubly stochastic, and invertible, with common eigenvectors \((1,1,1),(-2,1,1),(0,1,-1)\) and eigenvalues \(1,u,v\). For positive parameters,
\[
P(u,v)P(U,V)=P(uU,vV).
\]
If \(A=P(u,v)\) and \(B=P(U,V)\), the only possible postprocessor is
\[
Q=BA^{-1}=P(U/u,V/v).
\]
It is stochastic exactly when
\[
\boxed{U/u\le1,\qquad 3V/v\le2+U/u.}
\tag{1}
\]

This is a complete finite pairwise algorithm on this family [S1, §2.1, equations labelled `eq:product` and `eq:order-test`]. It tests every stochastic \(Q\), because invertibility makes \(Q\) unique. It is not a restriction of the comparator to symmetric or commuting postprocessors.

### 3.3 What the infinite-monotone result actually says

For \(n\ge2\), set
\[
d=\frac1{20n},\quad q=1-d,\quad c=1-d^2,
\]
\[
A_i=P(q^{4i}/4,q^i/4),\qquad
B_i=P(cq^{4i}/6,2q^i/9),\quad i=0,\ldots,n-1.
\]
The only nonidentity comparisons are
\[
A_i\preceq B_j\quad\Longleftrightarrow\quad i\ne j.
\tag{2}
\]

The construction follows from
\[
3+cz^4-4z=(z-1)^2(z^2+2z+3)-d^2z^4,
\quad z=q^{j-i},
\]
together with \(19/20<z<20/19\). Diagonal cross comparisons fail; off-diagonal cross comparisons hold. The within-family and reverse comparisons fail as proved in [S1, §§2.3–2.4].

A single weak monotone cannot witness two failed diagonal comparisons: the two allowed cross comparisons would force a strict inequality cycle. Therefore arbitrarily many monotones are needed for a **fixed universal list of unary scores compared componentwise over the entire family** [S1, Theorem 1 and §3].

This is not a claim that:

- one channel needs infinitely many numerical parameters;
- one pair needs infinitely many calculations;
- every restricted family or actual history needs infinitely many scores;
- the \(2n\) channels in (2) can all lie along one divisible history.

They cannot all lie on one such history: many pairs are incomparable. Use this construction only as a source-fidelity and comparison check, never as a shuffled Markov movie.

### 3.4 Other results and limits that matter to this project

| Source result | Assumptions and audit status | Consequence for execution |
|---|---|---|
| Countably many continuous, 1-Lipschitz scores are complete | Compact stochastic simulation sets; dense target channels; proof checked [S1, §4] | Do not enumerate an infinite score family. Pairwise comparison is directly constructive. |
| Two weak scores suffice for square binary channels | Output relabeling equivalence; singular class treated separately [S1, §5] | Concerns channel-order dimension, not a physical two-versus-three-state arrow threshold. |
| False-positive deficiency \(>1/[25200(k+1)^2]\) for some pair missed by \(k\) monotones | Exact global quantifiers; conservative bound; proof checked [S1, §6] | Does not ensure a practical statistical gap in a fixed trajectory or at finite sample size. |
| Two scores suffice under the common-basis continuous-time restriction | \(x=-\log u,\ y=-\log v\); allowed generator increments satisfy \(x\ge0,\ 3y-x\ge0\) [S1, §7] | Do not transfer the unrestricted theorem to continuous-time restrictions. |
| Record capacity \(F(c)=\lfloor2c/3\rfloor+\lfloor c/2\rfloor\) | Nested observation partitions; exact early record sizes \(2,3\); older record recoverable from newer record alone [S1, §8] | A combinatorial cardinality theorem. Retention direction is an assumption, not inferred time. |
| Recovery region \(\delta/(1+\delta)\le e\le\min(\delta,1-1/n)\) | Square \(n\times n\) channels, system-only randomized worst-case recovery [S1, §9] | Does not apply to correlated access to a hidden environment or automatically to unequal alphabets. |
| Fixed-qubit extension | Pauli-channel positivity and postprocessing [S1, §10] | Mathematically relevant; quantum software and experiments are unnecessary here. |

No essential proof defect was found in planning review. This is a finite review, not formal proof certification. Source-reported numerical experiments support checks, not theorems or physical claims.

The source's continuous-time restriction is especially material: all its main \(A_i,B_i\) are individually embeddable within that commuting family, while the cross postprocessors used by the construction are not. Do not confuse individual endpoint embeddability with admissibility of the transitions being inferred.

### 3.5 Constructive and numerical audit

For the selected rational instances, no hidden oracle, asymptotic optimizer, convergence criterion, or discretization is needed. Generic exact inversion and finite permutation enumeration suffice.

The countable-score completeness proof is an existence/separation result. A search through targets has no uniform finite stopping certificate for equality or arbitrarily small deficiency. It is unnecessary and would be a poor implementation choice.

Floating-point inversion near singularity could change signs and comparison outcomes. The primary study avoids that issue by using rational arithmetic. In the central example \(u\ge1/4\), the Euclidean condition number is at most four. In source checks with \(n=3\), a failed diagonal quotient has entry \(-1/32400\). These facts are useful diagnostics, not reasons to replace exact certification with a tolerance.

The finite experiments do not prove the universal source theorem. Conversely, the counterexample below does not falsify it; it challenges a stronger proposed physical interpretation.

## 4. Derived facts used as independent oracles

The results in this section are elementary derivations for this brief. Attribute them as such, with conceptual context from the cited channel literature. They are not claims copied from the source paper.

### 4.1 Exact depolarizing-channel comparison and recovery

Let
\[
J=\frac13\mathbf1\mathbf1^\top,\qquad
C_u=uI+(1-u)J=P(u,u),\qquad 0<u\le1.
\]
Then
\[
C_u C_v=C_{uv},\qquad \det C_u=u^2,
\]
and
\[
\boxed{\Delta(C_u,C_v)=\frac23\max(v-u,0).}
\tag{3}
\]

Proof: if \(v\le u\), \(C_{v/u}\) is a stochastic postprocessor. If \(v>u\), every stochastic \(Q\) satisfies
\[
\operatorname{tr}(QC_u)
=u\operatorname{tr}Q+(1-u)\operatorname{tr}(QJ)
=u\operatorname{tr}Q+1-u\le1+2u.
\]
For each column, total variation dominates its diagonal-coordinate difference. Taking the maximum and then the mean gives
\[
D(C_v,QC_u)\ge
\frac{\operatorname{tr}C_v-\operatorname{tr}(QC_u)}3
\ge\frac23(v-u).
\]
The choice \(Q=I\) attains the bound.

Therefore
\[
C_u\preceq C_v\iff u\ge v,\qquad
e(C_u)=\frac{2(1-u)}3.
\tag{4}
\]
The same value is the distance to the nearest permutation: every diagonal entry is the largest entry in its column, so identity matching is optimal.

For labels \(a,b,c\), use \(u_a=1/4,\ u_b=1/2,\ u_c=3/4\).

| Channel | Diagonal entry | Off-diagonal entry | \(\lvert\det\rvert\) | \(e\) |
|---|---:|---:|---:|---:|
| \(A_a=C_{1/4}\) | \(1/2\) | \(1/4\) | \(1/16\) | \(1/2\) |
| \(A_b=C_{1/2}\) | \(2/3\) | \(1/6\) | \(1/4\) | \(1/3\) |
| \(A_c=C_{3/4}\) | \(5/6\) | \(1/12\) | \(9/16\) | \(1/6\) |

The directed deficiency table has source rows and target columns:

| \(\Delta(\text{source},\text{target})\) | \(a\) | \(b\) | \(c\) |
|---|---:|---:|---:|
| \(a\) | 0 | \(1/6\) | \(1/3\) |
| \(b\) | 0 | 0 | \(1/6\) |
| \(c\) | 0 | 0 | 0 |

These are exact values. A generic LP is not needed to certify them. Do not introduce SciPy solely to reproduce an analytic table.

### 4.2 A scalar sorts any strict invertible divisible chain

For every square stochastic matrix \(Q\), Hadamard's inequality gives
\[
|\det Q|
\le\prod_j\|Q_{\cdot j}\|_2
\le\prod_j\|Q_{\cdot j}\|_1=1.
\]
Equality requires each column to be a unit basis vector and the columns to be distinct. Thus \(|\det Q|=1\) exactly for permutation matrices.

If \(B=QA\) with \(A\) invertible,
\[
|\det B|=|\det Q|\,|\det A|.
\]
It strictly decreases unless \(Q\) is a permutation. Hence \(1-|\det A|\) is a weak monotone, and descending \(\lvert\det\rvert\) orders a finite invertible divisible chain up to permutation-equivalent checkpoints.

This is a **conditional sorting result**, not a complete pairwise Blackwell test: incomparable channels can have different determinants. Equal determinants alone also do not imply equivalence outside a promised chain. For singular snapshots the determinant may vanish at many stages; the strict sorting argument then does not apply.

### 4.3 What an informational entropy control can say

For uniform \(X\), all \(C_u\) yield uniform output, so \(H(Y)=\log 3\) for every checkpoint. It is only a deliberately uninformative negative control.

A credible scalar reference is
\[
I(X;Y_u)=\log 3-H(p_u),\qquad
p_u=\left(\frac{1+2u}3,\frac{1-u}3,\frac{1-u}3\right).
\]
Use natural logarithms and label units as nats, with \(0\log0=0\) at endpoints. For \(0<u<1\),
\[
\frac{d}{du}H(p_u)=\frac23
\log\frac{(1-u)/3}{(1+2u)/3}<0,
\]
so mutual information increases with \(u\). It gives the same loss ordering \(c,b,a\) here. Do not call \(H(Y)\), \(H(p_u)\), or \(I(X;Y_u)\) entropy production.

### 4.4 Sampling cannot rescue exact observational equivalence

If two models induce the same law \(D\), every measurement, estimator, randomized postprocessing, or sampling scheme whose conditional distribution depends only on \(D\) has the same distribution under both models. This includes arbitrary numbers of independently sampled endpoint observations or unoriented checkpoint records.

Thus the primary impossibility does not require a finite-sample benchmark. It holds even with exact probabilities. It does not address an experiment that records extra ordered interventions.

For future numerical use only, the stable bound
\[
|\Delta(A,B)-\Delta(A',B')|
\le D(A,A')+D(B,B')
\]
follows from contraction and the triangle inequality. A statistical branch would require simultaneous uncertainty bounds on estimated channels; it is outside this execution scope. Do not use a noisy inverse plus an arbitrary sign tolerance to assert exact accessibility.

## 5. The explicit reversible counterexample

### 5.1 Fixed shift functions

Let the observed state be \(S\in\mathbb Z_3\). Let \(E\in\{0,\ldots,11\}\) be uniform, initially independent of \(X\). Use checkpoint labels \(r\in\{0,a,b,c\}\); these are identities, not ordered numbers.

Define fixed functions \(f_r(E)\in\mathbb Z_3\):

| Label | \(f_r=0\) | \(f_r=1\) | \(f_r=2\) |
|---|---|---|---|
| \(0\) | all \(E\) | none | none |
| \(a\) | \(0,\ldots,5\) | \(6,7,8\) | \(9,10,11\) |
| \(b\) | \(0,\ldots,7\) | \(8,9\) | \(10,11\) |
| \(c\) | \(0,\ldots,9\) | \(10\) | \(11\) |

The initial state is \(S=X,\ r=0\). Since the shift counts are \((6,3,3)\), \((8,2,2)\), and \((10,1,1)\), the channel \(X\mapsto X+f_r(E)\pmod3\) is respectively \(C_{1/4}, C_{1/2}, C_{3/4}\).

These are strictly positive, invertible, commuting channels in the paper's family. The example does not rely on complete erasure, rank deficiency, numerical ill-conditioning, or zero-probability system-channel entries.

### 5.2 One autonomous bijection for each chronological order

For any permutation \(\pi=(\pi_1,\pi_2,\pi_3)\) of \((a,b,c)\), let \(\sigma_\pi\) be the cycle
\[
0\mapsto\pi_1\mapsto\pi_2\mapsto\pi_3\mapsto0.
\]

On the \(3\cdot12\cdot4=144\) microstates define
\[
F_\pi(s,e,r)=
\left(
s+f_{\sigma_\pi(r)}(e)-f_r(e)\pmod3,\ e,\ \sigma_\pi(r)
\right).
\tag{5}
\]
This is a single time-homogeneous deterministic map for that model. Its inverse is
\[
F_\pi^{-1}(s,e,r)=
\left(
s+f_{\sigma_\pi^{-1}(r)}(e)-f_r(e)\pmod3,\ e,\ \sigma_\pi^{-1}(r)
\right).
\tag{6}
\]
The shifts telescope, so \(F_\pi^4=I\) on all 144 microstates.

Verify bijectivity and the inverse by exhaustive enumeration, independently of merely substituting (6) back into (5).

### 5.3 Identical data, different histories

Starting from \((x,e,0)\), every visit to checkpoint \(r\) has
\[
S_r=x+f_r(e)\pmod3.
\]
Consequently its full joint endpoint channel is
\[
T_r(s,e',r'\mid x)
=
\frac1{12}\mathbf1\{r'=r,\ s=x+f_r(e')\pmod3\},
\tag{7}
\]
independent of \(\pi\). Marginalizing \(E,r'\) yields \(A_r\).

The unoriented joint checkpoint law is
\[
J(y_a,y_b,y_c\mid x)=
\frac1{12}\sum_{e=0}^{11}
\prod_{r\in\{a,b,c\}}
\mathbf1\{y_r=x+f_r(e)\pmod3\}.
\tag{8}
\]
It also does not depend on \(\pi\).

The same checkpoint labels and the same \(f_r\) assignments must be used in every model. Changing the assignment of channels to numerical clock phases would spoil the stronger equality of full joint endpoints; the cycle formulation above avoids that error.

All six \(\pi\) are physically admissible in this finite bijective model class. All six produce the same \(A_r\), \(T_r\), and \(J\). Yet their actual chronological orders differ. In particular, \(F_\pi^{-1}\) visits the same checkpoints in reverse order.

The microscopic transition law is unknown in this six-model statement. A known \(F_\pi\) and complete boundary preparation would be extra information. State this qualification prominently.

### 5.4 Exact inference consequence

For any single-order estimator \(g(D)\), the distribution of its output is the same in all six models. With a uniform prior on \(\pi\),
\[
\Pr[g(D)=\pi]
=\frac16\sum_{\pi}\Pr[g(D)=\pi\mid\pi]
=\frac16.
\]
An estimator that sometimes abstains can have smaller unconditional single-order success; no procedure can exceed \(1/6\) with only the delivered data under this prior.

Do not describe the six histories as six independent statistical samples. They are an exhaustively enumerated observational-equivalence class.

A set-valued result that guarantees coverage over this class must include all six orders. Report
`chronology_status="nonidentifiable"`,
with the six certified witnesses. You may additionally report the unique **divisible-compatible** order \((c,b,a)\), explicitly conditional on the divisible-system promise.

The source comparison falsely interpreted as actual chronology would select \((c,b,a)\) even in the admissible history \((a,b,c)\). That is a failure of the interpretation, not a failure of the comparison algorithm.

### 5.5 Fixed-law time-reversal control

Include this small control to distinguish bijectivity from a specified time-reversal symmetry.

Fix \(\sigma=(0,a,b,c)\), add a hidden direction bit \(v\in\{-1,+1\}\), and use the same 288-state law
\[
\widetilde F(s,e,r,v)=
\left(
s+f_{\sigma^v(r)}(e)-f_r(e)\pmod3,\ e,\ \sigma^v(r),\ v
\right).
\tag{9}
\]
Define
\[
\Theta(s,e,r,v)=(s,e,r,-v).
\]
Verify on all 288 states that
\[
\Theta^2=I,\qquad
\Theta\widetilde F\Theta=\widetilde F^{-1}.
\tag{10}
\]

From the common observable preparation \(S=X,r=0\), the two hidden initial ensembles \(v=+1\) and \(v=-1\) give orders \((a,b,c)\) and \((c,b,a)\). After omitting \(v\), their allowed checkpoint observations are identical. Thus an equal prior over these two hidden preparations gives the \(1/2\) bound even with a fixed microscopic law.

Their complete 288-state endpoint channels are **not** identical if \(v\) is observed. Their complete initial environments are also different. Do not conflate this control with the six-model same-preparation, different-law result.

Equations (9)–(10) establish an involutive time-reversal symmetry of a finite model. No energy function, bath, work protocol, detailed balance, or thermodynamic entropy production is specified. Do not infer any such properties.

## 6. Controls and ablation

### 6.1 Memoryless positive control with the same endpoint channels

Construct a separate reduced Markov history
\[
X\xrightarrow{C_{3/4}}Y_c
\xrightarrow{C_{2/3}}Y_b
\xrightarrow{C_{1/2}}Y_a.
\tag{11}
\]
Its endpoint maps are exactly \(C_{3/4},C_{1/2},C_{1/4}\). It has fresh independent randomness in each step.

The data consisting only of the three endpoint channels are identical to the memoryful construction. Its joint checkpoint law need not equal (8), and must not be claimed to do so.

Under the explicit promise that the observed-system chronology is divisible, the unique compatible order is \((c,b,a)\). Both full channel methods, determinant, mutual information, and the recovery score must return that loss ordering. The marginal-output-entropy control must tie.

This control demonstrates conditional usefulness and detects reversed conventions. It cannot establish that a real process is memoryless from its endpoint data.

### 6.2 Targeted ablation: erase environmental correlations

Use the ascending-memory history \((a,b,c)\) from (5). After each step and before the next, redraw \(E\) independently from the uniform distribution while keeping the current system state and checkpoint label. This removes the retained correlation relevant to decoding the preparation.

At a checkpoint of the **unablated retained-memory history**, \(S=X+f_r(E)\). For fixed \(X=x\), the system and environment are correlated through that identity. With uniform \(X\), \(S\) and \(E\) are marginally independent, but \(I(X;S,E)=\log3\) while \(I(X;S)<\log3\). A reset at such a checkpoint makes the fresh environment independent of \((X,S)\), removing its additional information about the preparation. The perfect joint encoding need not persist after earlier resets. Do not interpret this ablation as a decrease in marginal \(I(S;E)\), which is already zero in the uniform unablated ensemble.

The effective system transition from \(r\) to \(t\) is
\[
K_{t\leftarrow r}(y,s)=
\frac1{12}\#\{e:y=s+f_t(e)-f_r(e)\pmod3\}.
\tag{12}
\]
Compute it from the shift table, not from the expected answer. For both \(a\to b\) and \(b\to c\), it should equal
\[
K=
\begin{pmatrix}
3/4&1/4&0\\
0&3/4&1/4\\
1/4&0&3/4
\end{pmatrix},
\qquad \det K=7/16.
\tag{13}
\]

The ablated endpoint channels are
\[
C_{1/4},\quad KC_{1/4},\quad K^2C_{1/4},
\]
whose determinants are exactly
\[
1/16,\quad 7/256,\quad 49/4096.
\]
They decrease, whereas the unablated ascending history has determinants
\(1/16,1/4,9/16\).

The ablation deliberately changes the physical process and its endpoint data. Interpret it as a mechanism check showing the role of retained correlation with the preparation reference, not as a same-input accuracy comparison. The reset operation is not asserted reversible.

### 6.3 Nonsymmetric, noncommuting chain control

Symmetric commuting examples cannot expose every indexing or multiplication-order bug. Include the following independent generic control:
\[
T_1=\frac1{10}
\begin{pmatrix}8&1&2\\1&7&1\\1&2&7\end{pmatrix},\quad
T_2=\frac1{10}
\begin{pmatrix}7&2&1\\2&7&3\\1&1&6\end{pmatrix},\quad
T_3=\frac1{10}
\begin{pmatrix}6&1&2\\3&8&1\\1&1&7\end{pmatrix}.
\]
Their determinants are \(9/25,\ 1/4,\ 3/10\), respectively; \(T_1T_2\ne T_2T_1\).

Use snapshots
\[
G_1=T_1,\qquad G_2=T_2T_1,\qquad G_3=T_3T_2T_1.
\]
Their determinants are \(9/25,\ 9/100,\ 27/1000\). Generic Blackwell comparison and descending determinant must recover \(G_1,G_2,G_3\) under the divisible-chain promise.

The source-family shortcut must return `out_of_domain` for these matrices rather than projecting them into \(P(u,v)\). Its absence here is not a comparative failure: the formula's domain is explicitly restricted. The generic exact comparator remains applicable.

### 6.4 Reversible relabeling ambiguity

Let \(R\) be the permutation matrix sending state \(x\) to \(x+1\pmod3\). Use snapshots \(I,R,R^2\). They are mutually postprocessing-equivalent, have \(|\det|=1\), and have zero recovery error.

All six orders are compatible with the divisible promise. The output must preserve this ambiguity. An arbitrary lexicographic tie-break may be used for display only, never presented as recovered chronology.

### 6.5 Source standard examples are not histories

For development use the source construction with \(n=2\); for the locked source check use \(n=3,4\). Check all ordered pairs using both generic rational inversion and formula (1), with expected relation (2).

For \(n=3\), a whole-set divisible-history request must return `incompatible_with_divisible_chain`, because the six snapshots are not a chain. It must not turn the partial order into a purported admissible movie by ordinary topological sorting. Incomparability is not an unconstrained adjacent transition.

This control audits the source mechanism without pretending to test the universal theorem empirically.

## 7. Implementation specification

### 7.1 Stack and exact arithmetic

Use ordinary CPython, preferably an installed native Apple-silicon interpreter, version 3.12 or later. A compatible earlier 3.11 interpreter is acceptable if the implemented standard-library features work and its version is recorded. No third-party package is required.

Use:

- `fractions.Fraction` for all probabilities, channel entries, matrix operations, and exact decisions;
- `itertools` for finite permutations and state enumeration;
- `json`, `csv`, `hashlib`, `pathlib`, `argparse` for local data and commands;
- `unittest` for the substantive mathematical/certificate checks;
- `time.perf_counter_ns` and the platform's resource tools for measured cost;
- `math.log` only for secondary entropy illustrations.

Construct fractions from integer numerator/denominator pairs or rational strings. Never use `Fraction(0.1)` to mean \(1/10\). Store rational outputs as canonical strings such as `"1/12"`. This makes equality and certificate replay exact [R8].

No GPU, Metal, MPS, CUDA, distributed execution, web service, database, training framework, or paid computation is needed. Do not accelerate a 3-by-3 rational problem.

### 7.2 Core interfaces to create

These are required interfaces, **not existing verified commands or supplied implementation**.

| Interface | Input | Output and failure behavior |
|---|---|---|
| `validate_channel(A)` | Rational matrix | Dimensions, exact nonnegativity/column sums; explicit invalid-input error |
| `inverse_fraction(A)` | Square rational matrix | Exact inverse or `singular`; never silently perturb |
| `blackwell_exact(A,B)` | Valid invertible source, matching input alphabet | Boolean, exact \(Q=BA^{-1}\), and negative-entry witness when false |
| `family_parameters(A)` | Rational 3-by-3 matrix | Exact \((u,v)\) only if reconstructed \(P(u,v)=A\) and domain holds; otherwise `out_of_domain` |
| `source_family_compare(A,B)` | Two exact family matrices | Equation (1), quotient parameters, comparison result |
| `compatible_orders(channels, relation)` | At most six snapshots, cached exact pair relation | Every permutation whose adjacent pairs are allowed; empty set if incompatible |
| `build_reversible_model(pi)` | Permutation of fixed checkpoint tokens | 144-state permutation mapping and independently constructed inverse table |
| `observe_model(model)` | Model and common preparation | Exact \(A_r,T_r,J\); truth kept outside estimator data |
| `certify_time_reversal()` | Fixed §5.5 definitions | Exhaustive 288-state checks and equality of allowed records for the two initial ensembles |
| `reset_environment_channels()` | Shift table and ascending schedule | Kernels (12) and exact cumulative channels |
| `verify_certificates(path)` | Saved rational matrices, maps, counts | Independent replay; nonzero exit on any mismatch |
| `generate_report(path)` | Preserved results and protocol | Evidence-backed Markdown report and tables; no fabricated missing values |
| `supervise(command, config)` | One local experimental command | Parent-process timing, timeout, raw resource capture, and an attempt record even after child failure |

For singular input, the generic inverse comparator must return `unsupported_singular`. None of the required comparisons has a singular source. Do not misclassify a singular case as an impossible transition. A general singular-channel LP is outside scope.

The exact generic comparator is the strongest feasible baseline on this domain. A separate all-decision-problem optimizer, infinite guessing-probability family, or LP would produce the same order and adds no necessary evidence.

### 7.3 Generic comparator pseudocode

~~~text
BLACKWELL_EXACT(A, B):
    validate both channels and matching dimensions
    if A is singular:
        return unsupported_singular
    Inv = generic Gauss–Jordan elimination over rational numbers
    assert A*Inv == Inv*A == I
    Q = B*Inv
    assert Q*A == B
    assert each column of Q sums exactly to 1
    if every Q entry is nonnegative:
        return allowed, Q
    return forbidden, Q, at least one exact negative-entry witness
~~~

Gauss–Jordan must use pivot search and row swaps; no floating tolerance. Cache one inverse per source matrix. Do not use the spectral formula inside this generic implementation: it is the independent check of that formula.

For a failed comparison, a negative entry of the **unique** quotient is a complete infeasibility certificate. It is not necessary to search over other stochastic matrices.

### 7.4 Compatible-order pseudocode

~~~text
COMPATIBLE_ORDERS(channels):
    compute every directed Blackwell comparison exactly
    if any comparator result is unsupported or invalid:
        return explicit unsupported/invalid status, not an empty set
    candidates = []
    for pi in all permutations of the checkpoint tokens:
        if every adjacent ordered pair in pi is allowed:
            append pi
    if candidates is empty:
        return incompatible_with_divisible_chain
    if candidates has one member:
        return unique_divisible_compatible_order, that member
    return multiple_divisible_compatible_orders, candidates
~~~

Composition of stochastic postprocessors ensures nonadjacent comparisons along an accepted chain. Verify this transitivity in the saved relation as a consistency check. Mutual accessibility is an equivalence relation, not a reason to arbitrarily orient equivalent checkpoints.

Do not describe this routine as enumerating all physically possible histories with hidden memory. It enumerates **divisible-compatible** orders only.

For the hidden-memory diagnostic, admissibility is instead certified by building and checking each of the six explicit \(F_\pi\). The complete set of all three-label permutations is finite, so these witnesses establish six physically possible orders for the delivered data.

### 7.5 Reversible-model pseudocode

~~~text
CERTIFY_MEMORY_WITNESS:
    define the fixed shifts f_0, f_a, f_b, f_c
    reference_data = none
    for pi in permutations(a,b,c):
        define sigma as cycle (0, pi[0], pi[1], pi[2])
        build map F on all (s,e,r)
        check its image has exactly 144 distinct states
        construct inverse by reversing the enumerated mapping
        verify inverse against the separately derived formula
        verify F^4 is identity on all 144 states

        initialize integer count tables A, T, J
        for x in {0,1,2}:
            for e in {0,...,11}:
                start at (x,e,0)
                simulate four applications of F
                record the first visit to each checkpoint by identity
                verify visits occur in order pi
                increment A_r(y,x), T_r(s,e,r | x), J(y_a,y_b,y_c | x)
        normalize each conditional count by 12
        compare A to the explicit C_u matrices
        compare observed tables exactly to equations (7) and (8)
        compare complete tables across all six models
        save exact maps, counts, order, and equality results separately

    only after all checks pass:
        certify nonidentifiability over the declared model class
~~~

The simulation route must not fill tables directly from equations (7)–(8). Those equations are independent expected results. Conversely, the certificate verifier should reconstruct marginals from saved counts rather than trust a stored `"pass"` field.

### 7.6 Fair baselines and interpretation

| Method/control | Delivered information | Required result | What it tests |
|---|---|---|---|
| Source-family formula | Same endpoint matrices, exact family membership | Same relation as generic comparator | Fidelity to paper |
| Generic exact Blackwell comparator | Same endpoint matrices | Complete relation on invertible domain | Closest strong baseline |
| Descending \(\lvert\det\rvert\) | Same matrices | Correct strict-chain order; explicit tie handling | Strong simple scalar reference |
| Mutual information / recovery | Same matrices; uniform prior stated for MI | Same ordering on \(C_u\) | Scalar usefulness on a restricted family |
| Output entropy \(H(Y)\) | Same matrices and uniform prior | Complete tie for \(C_u\) | Negative control only |
| Explicit reversible witnesses | Declared hidden-memory model class, same delivered law | All six orders certified | Actual-chronology nonidentifiability |
| Environment reset | Same shift definitions; changed correlation retention | Loss of rebound as in §6.2 | Mechanism ablation |

No baseline may receive timestamps, true \(\pi\), known microscopic adjacency, or a different sample budget. Do not report the witness enumeration as an inference advantage over a channel baseline: it establishes the admissible model class and the need to abstain.

A lower runtime for the specialized formula would only reflect a closed-form shortcut on a restricted family. There is no runtime-superiority claim in this study.

### 7.7 Data and repository layout

Create a small repository in an isolated local project directory:

~~~text
README.md
POC_EXECUTION_BRIEF.md
config.json
protocol.lock.json
src/
    exact_channels.py
    reversible_models.py
    comparators.py
    certificates.py
    supervise.py
    run_study.py
    make_report.py
tests/
    test_mathematics.py
    test_certificates.py
source/
    provenance.json
    supplied/                 # supplied archive material, when available
results/
    attempts.jsonl
    development/
    locked/
        environment.json
        channel_tables.json
        model_certificates.json
        relation_tables.csv
        outcome_table.csv
        verification.json
        resource_usage.json
report/
    REPORT.md
~~~

These are module responsibilities, not a demand for a framework. Combine adjacent modules if it improves clarity, while keeping independent formulas and generic matrix checks genuinely distinct. A few hundred lines to roughly a thousand lines of ordinary Python should be sufficient; this is an estimate, not a productivity requirement.

Use JSON configuration containing at least:

~~~json
{
  "schema_version": 1,
  "arithmetic": "fractions",
  "system_states": 3,
  "environment_states": 12,
  "checkpoint_ids": ["opal", "reed", "quartz"],
  "development_source_n": [2],
  "locked_source_n": [3, 4],
  "max_comparison_dimension": 3,
  "max_microstates": 288,
  "max_order_enumeration_snapshots": 6,
  "max_pairwise_fixture_snapshots": 8,
  "max_command_seconds": 60,
  "max_experimental_seconds": 1200,
  "max_process_rss_mib": 512,
  "max_project_disk_mib": 512,
  "max_experiment_invocations": 40,
  "max_full_suite_starts": 6,
  "max_repair_cycles": 2,
  "float_display_abs_tolerance": 1e-12
}
~~~

Map the three neutral tokens to \(a,b,c\) in the generator manifest, not to chronology. The mapping may be public: the matrices already reveal their informational ordering. The true visit permutation must remain evaluator-only. Enumerate all six input presentation orders to check that serialization order does not affect results.

Store probabilities as rational strings and integer counts. Keep exact comparisons separate from floating display values. An entropy display discrepancy never changes an exact accessibility decision.

### 7.8 Intended commands

Create and verify these commands. They have **not** been executed as a complete project during brief preparation.

~~~bash
python3 -m venv .venv
.venv/bin/python src/supervise.py --config config.json -- .venv/bin/python src/run_study.py inspect --out results/development/environment.json
.venv/bin/python src/supervise.py --config config.json -- .venv/bin/python -m unittest discover -s tests -v
.venv/bin/python src/supervise.py --config config.json -- .venv/bin/python src/run_study.py smoke --config config.json
.venv/bin/python src/supervise.py --config config.json -- .venv/bin/python src/run_study.py freeze --config config.json --out protocol.lock.json
.venv/bin/python src/supervise.py --config config.json -- .venv/bin/python src/run_study.py run --protocol protocol.lock.json --out results/locked
.venv/bin/python src/supervise.py --config config.json -- .venv/bin/python src/run_study.py verify --results results/locked
.venv/bin/python src/supervise.py --config config.json -- .venv/bin/python src/make_report.py --results results/locked --out report/REPORT.md
~~~

No `pip install` is needed. If a suitable interpreter is unavailable, complete the analytic report and code preparation; record the exact blocked execution step. Use a permitted existing installation route if available, but do not claim validation on an interpreter or machine you did not access.

The source archive is useful for provenance and re-auditing. All formulas necessary for the primary diagnostic are included here, so an unavailable external paper or attachment does not block the independent counterexample. Label any omitted source reproduction explicitly.

## 8. Minimal experiment and verification matrix

Every row has a specific scientific or correctness purpose. Do not add random seeds, grids, or training runs merely to make the study appear larger.

| ID | Fixed work | Expected evidence | Decision |
|---|---|---|---|
| E1 | Source \(S_n\): \(n=2\) development; \(n=3,4\) locked | 116 exact ordered-pair comparisons across the three sizes; source formula equals generic inverse; relation (2) | Any disagreement blocks source/comparator-dependent conclusions until diagnosed; preserve an independently certified core witness |
| E2 | All six 144-state models | Six bijections, inverse checks, \(F^4=I\), identical \(A,T,J\), six different visit sequences | Core identifiability certificate |
| E3 | Fixed-law 288-state control | Involution (10) on all states; two hidden preparations give identical allowed records and opposite order | Scope of microscopic time-reversal claim |
| E4 | Memoryless realization (11) | Exact products and unique compatible order; scalar and channel agreement | Conditional positive control |
| E5 | Retain versus reset \(E\) | Exact kernels and determinant sequences in §6.2 | Isolates environmental correlation |
| E6 | Generic noncommuting chain | Exact products, inverse orientation, unique compatible order, source shortcut rejects domain | Detects symmetry-masked bugs |
| E7 | \(I,R,R^2\) | Mutual accessibility, zero recovery, six compatible permutations | Detects false certainty |
| E8 | Presentation-order and certificate-integrity checks | All six input-list permutations preserve results; deliberately corrupted certificate is rejected | Detects leakage and ineffective verification |

For E1, the total \(16+36+64=116\) is a logical pair count per comparator, not an independent sample size. For the locked E1 subset alone the count is \(36+64=100\). Do not confuse these with the source's much larger reported checks.

For E8, corrupt a copy of one saved transition target or one count numerator and verify that validation fails. Preserve the valid original. This tests a meaningful risk: a verifier that only echoes stored status flags.

### 8.1 Primary metrics

Report exact values:

- Number of certified different chronological orders sharing the data law: expected six.
- Maximum entrywise difference across their rational \(A,T,J\) tables: exactly zero.
- Number of state-map failures, inverse failures, or nonreturning states: exactly zero.
- Number of mismatches between source and generic comparisons: exactly zero.
- Size of the divisible-compatible order set for each fixture.
- Presence of false certainty: any claimed unique actual chronology in the hidden-memory fixture is unacceptable.
- Resource use and every failed invocation.

A mathematical assertion requires exact equality. There is no rounding threshold for the primary conclusion.

The only numerical tolerance in the default project is an absolute \(10^{-12}\) check for optional float entropy display identities. It is far below the displayed scalar differences and does not affect scientific decisions. If platform arithmetic violates this display check, keep the exact result and report the display issue.

### 8.2 Independence and uncertainty

The independent object is a specified physical model and observation contract, not a pairwise entry or repeated execution of deterministic code. The six models are an exhaustive finite construction; they are not a sample of real physical processes.

No confidence intervals, null-hypothesis significance tests, bootstrap, or repeated random seeds are warranted for exact identities. The universal limitation follows from an admissible counterexample plus the indistinguishability argument. No prevalence claim is made.

Do not fit a six-class classifier to identical features. Its chance performance would be a weaker illustration of a result already proved exactly and could introduce spurious label leakage.

### 8.3 Exploration, locking, and confirmation

Use the \(n=2\) source fixture and one 144-state order for debugging. The remaining fixtures are specified in advance; do not tune any threshold or select histories based on favorable outcomes.

Before the locked run, save `protocol.lock.json` containing:

- configuration and code hashes;
- exact fixture definitions and expected scientific claims;
- source hashes and versions;
- data-access restrictions;
- primary criteria and failure classifications;
- cumulative budget already consumed.

A locked run is a reproducibility/correctness check, not an independent statistical holdout. Expected mathematical answers are public and appropriately used as oracles. Independent implementation of the generic inverse and certificate replay guards against merely reproducing a faulty formula.

If code changes after locking, preserve the failed run, record the reason, and create a new lock version. Rerun affected checks only, within the two-cycle repair allowance. Any new hypothesis, distribution, or physical assumption is exploratory and cannot be folded into the original conclusion.

### 8.4 Failure accounting and fair measurement

Every attempt appends one record to `attempts.jsonl` with command, configuration hash, start/end time, exit status, elapsed time, peak-memory raw value and units, and failure reason. Never overwrite unfavorable results. A small standard-library parent runner, `src/supervise.py`, starts its stopwatch before spawning the experimental command, captures stdout/stderr and raw system-resource output when available, and writes the attempt record in a `finally` path. It must record killed or crashed children and explicitly mark any unavailable resource measurement. Start the child in a process group so a timeout terminates the experimental command and any resource-utility wrapper, rather than leaving a worker running.

The supervisor checks cumulative caps before launch and sets the child timeout to the minimum of 60 seconds, the remaining stage allowance, and the remaining total compute allowance. Withhold the two-minute report reserve from non-report work, and do not launch when the applicable remaining allowance is nonpositive. Count each child invocation of `inspect`, the test suite, `smoke`, `freeze`, `run`, `verify`, or report generation once toward the 40-invocation limit; count failures and repeats. Do not count the supervisor as a second invocation. Virtual-environment setup, code editing, and reading belong to the separate active-work budget, while their disk use still counts toward the project cap. The official experimental elapsed time includes the child's interpreter startup and imports; record supervisor overhead separately if material.

A timed-out, unsupported, or crashed computation is not a forbidden physical transition. An invalid certificate does not count as evidence of nonidentifiability. If a method result is missing, report it as missing; do not silently compute comparison rates on successful cases alone.

Time complete command execution, including imports, rational construction, pair precomputation, enumeration, serialization, and validation. Also record core computation separately if convenient. The official cost is the complete command cost. These tiny workloads do not justify an amortized-cost claim.

Use one cold fresh-process locked run. At most one warm repeat is permitted if needed to diagnose an unexpectedly large resource cost. There is no performance ranking, so confidence intervals for runtime and extensive repeated timing are unnecessary.

## 9. Concrete Mac feasibility and resource policy

### 9.1 Inspect the actual machine first

The planning environment was Linux x86-64 with Python 3.12.14. It was not the user's Mac. Its checks establish algebraic and finite-construction plausibility, **not M3 Max performance**.

On the target inspect and record:

- `sw_vers`, `uname -m`, and chip model from an appropriate `sysctl` query;
- actual physical RAM and available memory pressure;
- free disk space in the intended project directory;
- Python executable, version, architecture, and whether translation is active;
- available process-resource measurement method;
- access to the supplied archive and project write permissions.

Do not infer RAM, free storage, macOS version, or backend behavior from the chip name. Avoid collecting machine serial numbers or unrelated system information.

Use exact CPU arithmetic. Native arm64 is preferred; a translated interpreter can still verify mathematics, but its timings must be labeled and are not native M3 Max performance. If no target Mac is available, complete unaffected code/proof work and explicitly mark Mac execution blocked.

### 9.2 Workload and scaling

The largest microscopic state set has 288 elements. The largest channel inversion is 3-by-3. Full joint observation tables may be rectangular and are not fed to that inversion routine. The most expensive requested ordering enumeration is \(6!=720\) candidates for the \(n=3\) six-channel source fixture, using cached pair results. The \(n=4\) source fixture has eight channels and is used only for its 64 ordered-pair comparisons; never send it to `compatible_orders`. Source exponents stop at \(n=4\).

Expected main costs are Python startup, Fraction normalization, and JSON serialization. There is no large tensor, optimization history, trajectory archive, or hidden GPU allocation. Store permutations as integer maps, not dense 288-by-288 rational transition matrices.

For \(m\) snapshots and channel dimension \(d\), generic comparison uses \(m\) inversions and \(m^2\) small products, with arithmetic operation counts \(O(md^3+m^2d^3)\) plus rational bit costs. Pairwise fixtures allow \(m\le8\); order enumeration is factorial and separately capped at \(m\le6\). Do not extrapolate the implementation to long movies.

Unmeasured estimates: the complete exact computation should take seconds to a few minutes and use well below 512 MiB. These are estimates to be checked, not promised measurements.

### 9.3 Hard caps and cumulative accounting

| Resource | Planning target | Hard cap / action |
|---|---|---|
| All experimental command wall time | Under 5 minutes | 20 minutes cumulative, including failed runs, retries, tuning, and report computation |
| Individual experimental invocation | Under 10 seconds for most checks | 60 seconds; terminate, preserve evidence, investigate |
| Peak resident memory per experimental process | Under 128 MiB | 512 MiB; stop further execution if exceeded |
| Project disk, including environment and results | Under 100 MiB | 512 MiB; remove only reproducible temporary copies, preserve evidence |
| Experiment invocations | About 10–20 | 40 total, including failures and repeats |
| Full-suite starts | Normally 2 or fewer | 6 total across all stages and repairs |
| Repair cycles | Zero or one | Two cycles, at most 45 minutes of engineering effort total |
| End-to-end setup, implementation, review, analysis | Approximately 2–4 hours | 6 hours of active project work; document any blocking access delays separately |

These caps are deliberately generous for the finite diagnostic. Exhausting them would indicate a tooling, implementation, or access issue, not intrinsic computational difficulty of the specified mathematics.

Reserve the final two minutes of the 20-minute compute cap and the final 30 minutes of active work for preserving evidence and producing the report. Branches and restarts never reset a cap.

If the project directory lacks 1 GiB free space, or system pressure makes the memory budget unsafe, complete the analytic work and reassess the smallest standard-library execution. Do not infer adequate capacity from installed RAM alone.

### 9.4 Calibration and resizing rules

The `smoke` command implements both G2 checks and G3 calibration: it runs one source \(n=2\) case, one 144-state map, one generic inverse, and serialization/replay. Measure total time and resource usage.

- If it finishes below 10 seconds and 128 MiB, continue with the fixed protocol.
- If it takes 10–60 seconds or 128–512 MiB, inspect the dominant cost. Permit one bounded repair if the cause is concrete, such as dense state matrices or repeated uncached inversion.
- If it exceeds 60 seconds or 512 MiB, stop the computational progression and diagnose. Do not replace rational decisions with float tolerances.
- Before each remaining stage, require a conservative estimate of twice the measured comparable cost to fit within the remaining compute allowance and report reserve.

No scale-up is needed after successful calibration. If budgets become tight, first omit optional entropy displays and any warm timing repeat. Next omit repeated source checks already certified, while stating what was not rerun. Do not reduce the three checkpoints, alter the data contract, or drop the distinction between system and environment to manufacture a successful result.

The exact analytic obstruction remains available if local execution is blocked. Report mathematical evidence and hardware validation status separately.

### 9.5 Resource measurement

Use `time.perf_counter_ns()` for elapsed durations [R8]. On macOS, run the locked command under the available system resource utility, such as `/usr/bin/time -l`, and retain its raw output. Record `resource.getrusage` values if used, but verify the platform's units before converting; peak-RSS units differ across systems [R8].

Do not use `tracemalloc` alone as process peak memory: it omits allocations outside its tracked Python allocator. No device synchronization is required because the study uses CPU-only rational arithmetic.

Avoid elaborate monitoring infrastructure. A simple parent subprocess timeout, raw system-resource output, and per-stage disk check are proportionate to the task.

## 10. Evidence gates and autonomous decisions

Use the gates in order. They combine the original constructive, correctness, calibration, pilot, comparison, and reporting purposes without forcing an unnecessary large benchmark.

The compute allocations below sum to 20 minutes including the shared repair reserve. Active engineering time is separately bounded in §9.3.

| Gate | Question and prerequisites | Minimum work and evidence | Compute allowance | Decision and next action |
|---|---|---|---:|---|
| G1: constructibility and scope | Does the data contract match the intended claim? Is a target interpreter accessible? | Restate unknown-law/hidden-environment assumptions; reproduce the short indistinguishability derivation; inspect actual Mac and sources | 1 min | Continue if admissible. If a required datum or machine is missing, finish unaffected proof/code and mark the exact block. If counterexample assumptions do not match the intended physical claim, report the scope mismatch. |
| G2: tiny correctness | Are arithmetic, matrix conventions, and microscopic construction sound? | Source \(n=2\); one 144-state model and its inverse; formulas (3)–(4); nonsymmetric multiplication check | 4 min | Zero exact discrepancies required. Repair a concrete defect within reserve; persistent contradiction stops scientific interpretation. |
| G3: resource calibration | Can the fixed finite study run locally? | Calibration in §9.4, resource record, remaining-budget forecast | 1 min | Continue if within caps. Bounded repair or analytic-only completion if not; never silently weaken arithmetic. |
| G4: mechanism pilot | Is the memory mechanism and conditional usefulness distinguishable? | Ascending retained-memory history, reset ablation, divisible positive control, permutation ambiguity control | 3 min | Exact expected distinctions must hold. A failure triggers diagnosis, not an immediate rejection of the source theorem. |
| G5: locked exact certification | Do independent checks certify the declared result? | Freeze protocol; all six 144-state models; 288-state reversal control; source \(n=3,4\); generic chain; presentation invariance; independent certificate replay | 4 min | If at least two distinct same-data admissible histories are certified, universal chronology inference is disproved for the contract. Verify the planned six where feasible. Do not launch an accuracy sweep afterward. |
| G6: diagnosis and report | What is established, conditional, untested, or blocked? | Final report, raw evidence, commands, limitations, resource totals, one justified next step | 2 min | Complete the study, including an honest early-stop result if appropriate. |
| Shared repairs | A specific defect with a plausible bounded remedy | Preserve failed attempts; at most two repair cycles | 5 min | All repair and fallback compute counts here and toward the same total. Stop when the reserve is exhausted. |

An exact counterexample is an allowed analytic stop against the universal application claim. G4–G5 are small verification controls that make the interpretation trustworthy; they are not a requirement to proceed to a large benchmark after the no-go result is established.

If the 288-state strengthening alone fails while the six-model witness remains valid, isolate the failure and narrow the final statement to the verified 144-state unknown-law result. Do not discard a valid core result because an explicitly stronger control is unresolved. Conversely, do not claim the fixed-law symmetry was checked if it was omitted.

If a bounded stop leaves only two distinct same-data witnesses computationally certified, report the computational result as \(N_{\mathrm{compatible}}\ge2\), with the six-model enumeration incomplete. That already establishes the universal-identifiability obstruction. Keep the general analytic derivation separate from the number of witnesses actually replayed.

If source-family reproduction fails but the independently specified counterexample passes, report the source defect separately, preserve the independent limitation, and avoid attributing the limitation to a reproduced source mechanism.

## 11. Outcome interpretation and required deliverables

### 11.1 Outcome classes

| Classification | Required evidence | Appropriate conclusion |
|---|---|---|
| **Supported contribution** | Verified scoped diagnostic or conditional boundary; credible exact comparisons agree | The reproducible application-level result is supported. Historical novelty is still unconfirmed. |
| **Informative negative result** | At least two same-data, different-order admissible histories plus the indistinguishability proof | \(H_{\mathrm{chron}}\) is false for the stated model class and observation contract. The source theorem is not refuted. |
| **Practical infeasibility** | Measured costs or supported resource/access analysis block the intended execution | State what cannot run on the target; separate this from mathematical validity. |
| **Inconclusive evidence** | A material assumption, source, hardware check, or construction detail remains unresolved | No final empirical or computational validation claim for that portion. |
| **Invalid experiment** | Leakage, wrong matrix convention, defective comparator, differing data laws, or ineffective certificate checks | Preserve and explain the failed attempt; no scientific inference from it. |

The expected final result is an informative negative answer to the stronger physical inference, together with a supported conditional ordering result. These labels can coexist because they refer to different claims.

Do not report “the approach is universally useless.” It can test possible stochastic simulation. Do not report “time has been reconstructed” from a divisible-compatible order unless the additional promise is explicit. Do not report that microscopic reversibility eliminates every thermodynamic arrow; the study does not specify the thermal protocols needed to ask that question.

### 11.2 Required artifacts for completed stages

Deliver:

1. Runnable standard-library code and concise `README.md` with the actual verified commands.
2. The frozen protocol and environment record.
3. Exact rational channels, relation tables, microscopic permutation certificates, raw count tables, and replay results for work actually performed.
4. All failed-attempt records and cumulative resource accounting.
5. A standalone `report/REPORT.md` containing the derivation, results, interpretation, and limitations.

The final report must include:

- the precise data contract;
- the 144-state construction and why its endpoints are identical across different histories;
- the distinction between the six-model unknown-law result and fixed-law hidden-bit control;
- the exact three-channel deficiency table and determinant/recovery values;
- the memoryless positive control and retained-versus-reset comparison;
- agreement or disagreement between the source shortcut and generic Blackwell comparator;
- what was actually run on M3 Max, with measured costs;
- source-reported results versus newly reproduced results;
- historical-novelty limitations and explicit exclusions;
- any material deviations, failed runs, or blocked checks.

For an analytic-only completion, provide the verifiable derivation and clearly state which code or Mac validation was not performed. Do not create empty result files implying completed experiments.

Use compact tables. A figure is optional: the exact order/deficiency table and a table of identical data versus different visit sequences are sufficient. Do not build a dashboard. If a figure adds value, generate it locally from preserved exact results and keep its meaning independent of decorative layout.

### 11.3 One next-step decision

If the expected limitation is confirmed, the next useful step is **to specify one concrete physical system and an additional direction-bearing observation protocol**, such as known driven dynamics with stated preparation and ordered intervention records. Reassess identifiability before choosing an estimator.

Do not pursue this extension within the present budget. It changes the data contract and needs a separate study. Further work on the source's fixed-alphabet order-dimension novelty is also a separate mathematical project.

## 12. Planning evidence, claim map, and review disposition

### 12.1 What was actually done before this brief

The following planning work occurred in a hosted Linux x86-64 environment, not on the target Mac:

- Read the complete supplied principal note and supplemental notes, code, and result files.
- Checked the principal algebra, global monotone quantifiers, quantitative bound, continuous-time scope, recovery assumptions, and record-retention assumptions through separate mathematical review.
- Reran the supplied exact classical and qubit checker functions at \(n=2,3,4,5,10\): 616 ordered pairs per class; all passed. This reused supplied checking functions and is not an independently implemented proof.
- Independently constructed and checked a small controlled-shift reversible example.
- Independently enumerated all six 144-state maps, checked bijectivity, prepared-state returns, common joint checkpoints, reduced channels, and reverse-order traversal. All passed.
- Derived the fixed-law 288-state involution analytically; its exhaustive computational check remains a coding-agent task.
- Derived the determinant strictness and exact deficiency/recovery formulas.
- Checked the reset kernels and determinant sequence with exact fractions.
- Checked the nonsymmetric control matrices' determinants and noncommutativity.
- Researched the primary literature in §13 and performed separate mathematical, experimental, and execution reviews of the design.

The large source recovery and record experiments were not rerun. No fitted model, finite-sample benchmark, target-Mac timing, or completed PoC repository was produced during planning.

You must still execute the required stages on the target and independently verify the evidence. The planning probes reduce design risk; they do not authorize marking unperformed checks complete.

### 12.2 Central claim-to-evidence map

| Claim | Source or derivation | Evidence status at handoff | Remaining check |
|---|---|---|---|
| Source relation is Blackwell postprocessing | [S1, §1]; [R1, Definition 3/Theorem 4] | Source and primary literature inspected | Exact comparator equivalence in E1 |
| Source theorem concerns global unary monotone completeness | [S1, §§1–4]; [R6] for broad predecessor | Proof and quantifiers reviewed; novelty unresolved | Source-fidelity tests; no universal proof claim from them |
| A strict invertible divisible chain has scalar determinant ordering | §4.2 derivation | Analytically checked | Generic/noncommuting positive control |
| Six different histories have the same delivered law | Equations (5)–(8) | Proof and small exact planning enumeration passed | Independent target implementation and certificate replay |
| Fixed law has time-reversal involution but hidden preparations remain ambiguous | Equations (9)–(10) | Analytically checked | Exhaustive 288-state check |
| Environmental retention enables rebound | §6.2 derivation | Exact reset-kernel planning probe passed | Retain/reset control on target |
| More samples of the same data cannot resolve identity | §4.4 / §5.4 | Exact information-theoretic argument | Check actual serialization contains no extra order data |
| Ordering here is different from thermodynamic path classification | [R4, R5] | Relevant full-text sections inspected | Report scope accurately; no surrogate “entropy-production baseline” |
| Workload fits local Mac | §9 operation counts and estimates | Plausible, not measured on M3 Max | Actual inspection, calibration, resource record |

### 12.3 Material review findings and repairs

Three distinct passes were performed with independent reviewer assistance; their findings were reconciled into this instruction.

**Mathematical and novelty pass.** The source's central mechanism equals the strongest known baseline. Its global theorem does not imply chain-sorting superiority. The source's possible fixed-alphabet novelty was separated from the proposed physical interpretation. The primary project was changed to an exact identifiability diagnostic, and determinant sorting was added as a strong scalar control. A phase-labeling error that could have changed full joint endpoint data was prevented by using fixed checkpoint identities and variable cycles.

**Experimental pass.** Marginal entropy alone would be an artificially weak comparison. It is now labeled only as a negative control; mutual information, recovery, determinant, and exact Blackwell comparison are included where applicable. The data contract excludes order leakage. Unknown-law and known-law/hidden-preparation claims are separate. The reset ablation changes the process and is interpreted only as a mechanism control.

**Execution pass.** A large noisy benchmark, infinite monotone enumeration, numerical optimizer, GPU stack, and general process-tensor reconstruction were unnecessary. The protocol now uses finite exact arithmetic, explicit certificates, a nonsymmetric indexing control, fixed case counts, cumulative budgets, and an analytic completion route. Review separated the eight-channel pairwise-fixture cap from the six-checkpoint enumeration cap, made `smoke` the calibration command, and specified an outer supervisor so crashes and startup costs are recorded. Required commands are identified as interfaces to implement, not commands already verified.

### 12.4 Stress cases and honest consequences

| Stress case | Disposition |
|---|---|
| The theorem's completeness construction is countable/nonconstructive as an algorithm | Bypass it using unique exact quotient on the invertible domain |
| Memorylessness fails | This is the central tested limitation; report actual chronology as nonidentifiable |
| Mechanism is equivalent to an established method | Treat comparator agreement as correctness, not novelty |
| Only specially constructed examples work | An admissible counterexample suffices against a universal claim; make no prevalence or generic-performance claim |
| A stronger scalar or channel baseline removes the gain | Already anticipated; there is no superiority claim |
| Preprocessing dominates | Count all startup, construction, enumeration, and serialization; no hidden amortization |
| Finite precision changes the result | Exact rationals decide all primary claims |
| Runtime or memory is unexpectedly large | Bounded repair and analytic completion; no unbounded rescue |
| Small pilot gives unclear evidence | Exact disagreement requires diagnosis; it is not statistical noise or a scientific refutation |
| A paper, archive, or dataset is inaccessible | No dataset is required; primary formulas are included; label omitted source reproduction |
| Main chronology hypothesis is false | A verified limitation is the intended valuable completion |
| Main counterexample is wrong | Preserve failure, bounded repair, then inconclusive/invalid status; do not hard-code expected result |
| Full dynamics or oriented records are supplied | Different data contract; this no-go claim does not apply unchanged |
| Append-only records appear to recover direction | Identify the directed retention assumption rather than claiming direction emerged without one |

Residual uncertainties are historical novelty, unmeasured M3 Max runtime, and the relation of this finite model to any specific laboratory system. They do not weaken the exact conditional mathematics, but they limit claims about research originality and real-world prevalence.

## 13. Research record and references

### 13.1 Search scope

Searches were conducted on 29 October 2026 using two search engines and primary-source follow-up. Topics included Blackwell/garbling order, stochastic-matrix order dimension, fixed output alphabets, divisibility, guessing probability, multitime Markovianity, reversible environmental memory, and arrow-of-time inference. Backward comparison included the standard-example construction in Kuramochi and the foundational Blackwell relation; later adjacent work on divisibility and signal comparison was checked where it affected interpretation.

Relevant source content, rather than snippets alone, was read for the core references below. The source paper's own literature notes were treated as claims and independently checked for the decision-relevant comparisons.

Coverage is not exhaustive. Blackwell's 1953 publisher full text and Dahl's 1999 matrix-majorization full text were not available in the inspected routes; recent SSRN geometric Blackwell work was available only incompletely. No global novelty certification follows. Seif et al.'s publisher identity was verified, and an accessible copy of the published paper was used when the arXiv PDF route failed.

### 13.2 References

**[S1] Supplied mathematical working note.** Author credited as “Mathematical investigation developed in this conversation.” *Finite channels with no finite system of monotone clocks*. 29 October 2026. Attachments: `research_note(1).html`; `research_reproducibility(1).zip` → `research_reproducibility/research_note.tex` and `research_note.html`. Critical locations: Theorem 1, §§1–4; equations labelled `eq:matrix`, `eq:product`, `eq:order-test`, `eq:AB`, `eq:keyfactor`; binary result §5; quantitative bound §6; continuous-time restriction §7; records §8; recovery §9; computational record §11; novelty/physical limits §12. Supplemental files are identified in §3.1. This is supplied working research, not independently verified publication.

**[R1] Nils Bertschinger and Johannes Rauh.** *The Blackwell relation defines no lattice*. 2014 IEEE International Symposium on Information Theory, 2014. [DOI: 10.1109/ISIT.2014.6875280](https://doi.org/10.1109/ISIT.2014.6875280). [arXiv:1401.3146](https://arxiv.org/abs/1401.3146), [full text](https://arxiv.org/pdf/1401.3146). Inspected Definitions 1–3 and Theorem 4, PDF pp. 2–4. The row-stochastic convention there is transposed in this brief. This is the operational reference for the distinction between an actual coupling and an existential Markov replica.

**[R2] Francesco Buscemi and Nilanjana Datta.** *Equivalence between divisibility and monotonic decrease of information in classical and quantum stochastic processes*. Physical Review A **93**, 012101, 2016. [DOI: 10.1103/PhysRevA.93.012101](https://doi.org/10.1103/PhysRevA.93.012101). [Full text, arXiv:1408.7062v3](https://arxiv.org/pdf/1408.7062). Inspected §III, Eq. (3), §V Proposition 1 and Lemma 2, and §V.A Corollary 1, PDF p. 5. The corollary's matching initial–time marginals are the critical scope qualification.

**[R3] Simon Milz, M. S. Kim, Felix A. Pollock, and Kavan Modi.** *Completely Positive Divisibility Does Not Mean Markovianity*. Physical Review Letters **123**, 040401, 22 July 2019. [DOI: 10.1103/PhysRevLett.123.040401](https://doi.org/10.1103/PhysRevLett.123.040401). [Institutional full text](https://researchmgt.monash.edu/ws/portalfiles/portal/291269926/281200867_oa.pdf). Inspected pp. 1–2, especially classical Eq. (1), the Chapman–Kolmogorov discussion, and Definition 1. Used for the distinction between divisible descriptions and multitime memory.

**[R4] Édgar Roldán, Izaak Neri, Meik Dörpinghaus, Heinrich Meyr, and Frank Jülicher.** *Decision Making in the Arrow of Time*. Physical Review Letters **115**, 250602, 16 December 2015. [DOI: 10.1103/PhysRevLett.115.250602](https://doi.org/10.1103/PhysRevLett.115.250602). [Full text, arXiv:1508.02018v4](https://arxiv.org/pdf/1508.02018). Inspected introduction and PDF p. 2, Eqs. (1)–(3). The input is an ordered path and a forward/reverse probability model; it is not the present endpoint-channel data contract.

**[R5] Alireza Seif, Mohammad Hafezi, and Christopher Jarzynski.** *Machine learning the thermodynamic arrow of time*. Nature Physics **17**, 105–113, 2021; online 21 September 2020. [DOI: 10.1038/s41567-020-1018-2](https://doi.org/10.1038/s41567-020-1018-2). [Publisher page](https://www.nature.com/articles/s41567-020-1018-2). [Inspected full-text copy](https://aiichironakano.github.io/cs596/Seif-LearningTimeArrow-NPhys20.pdf). Inspected “Thermodynamics and the arrow of time,” pp. 1–2, Eqs. (2)–(5), and the forward/reverse data-labeling setup. Thermal preparation and specified protocols differ materially from this diagnostic.

**[R6] Yui Kuramochi.** *Infinite dimensionality of the post-processing order of measurements on a general state space*. Journal of Physics A: Mathematical and Theoretical **55**, 435301, 2022. [DOI: 10.1088/1751-8121/ac99b0](https://doi.org/10.1088/1751-8121/ac99b0). [Full text, arXiv:2111.14129v2](https://arxiv.org/pdf/2111.14129). Inspected Theorem 1 and Remarks 2–4, PDF pp. 10–11, and Lemma 7, pp. 16–19. Its displayed standard-example construction has three outcomes on one side and \(2n-1\) on the other. It establishes the broad predecessor without by itself resolving simultaneous fixed-\(3\times3\) novelty.

**[R7] Pedro Hack, Daniel A. Braun, and Sebastian Gottwald.** *Majorization requires infinitely many second laws*. Proceedings of the Royal Society A **480**, 20230464, 2024. [DOI: 10.1098/rspa.2023.0464](https://doi.org/10.1098/rspa.2023.0464). [Full text, arXiv:2207.11059](https://arxiv.org/pdf/2207.11059). The introduction, published §4(b) and Theorem 5.1 were inspected in the novelty review; these correspond to arXiv §4.2 and Theorem 1 (PDF p. 11). The strict-monotone requirement for state majorization differs from the supplied weak-monotone channel result. This reference is included to avoid conflating similar three-state thresholds.

**[R8] Python Software Foundation.** *Python documentation: fractions, time, and resource*. Official documentation, accessed 29 October 2026. [Rational arithmetic](https://docs.python.org/3/library/fractions.html), [performance counters](https://docs.python.org/3/library/time.html#time.perf_counter_ns), [resource usage](https://docs.python.org/3/library/resource.html). Use the documentation matching the actual interpreter version. API inspection supports the simple stack; it does not establish target-machine performance.

**Historical attribution, not directly inspected full text:** David Blackwell, *Equivalent comparisons of experiments*, Annals of Mathematical Statistics **24**(2), 265–272, 1953. [DOI: 10.1214/aoms/1177729032](https://doi.org/10.1214/aoms/1177729032). The operative theorem was checked through [R1] and [R2].

### 13.3 Source integrity and handoff

The following hashes were computed from the supplied bytes during planning:

| File | SHA-256 |
|---|---|
| `Pasted text(20261029-162651).txt` | `e03fd13940c6562618d88bb44e32cc6a8ea5bca6e0121a9fb2293b25b96337de` |
| `research_note(1).html` | `96227e3573cac07474ed95b3dcc529824c7cc13be1fd4c33774d6bbeb2b3986c` |
| `research_reproducibility(1).zip` | `9801d2c9654a629987d359cc71cee707b1847b22c9dfa4e4dad1febd1f6cefe5` |
| Archived `research_note.tex` | `f3a2bbf1981c6e4671f69513fe98609767c15f87d99e72b7c5839c5d0910e23a` |

Receive this brief as `POC_EXECUTION_BRIEF.md`. Ideally also receive the HTML and reproducibility archive above, preserve their bytes, and extract the latter beneath `source/supplied/` with path validation. Do not require access to the originating chat, hosted scratch paths, or the planning probe scripts: the central construction, all fixed fixtures, assumptions, equations, expected results, and commands to implement are specified here.

If a source hash differs, record the difference and identify the version used. A source mismatch does not authorize pretending the original source was reproduced.

## 14. Completion checklist

Before delivering your research result, verify:

- The primary conclusion follows from an explicit data contract and certified admissible models.
- Source theorems, planning derivations, source-reported experiments, and newly executed checks are distinguished.
- The generic comparator uses all stochastic postprocessors on its stated invertible domain.
- The six-model result and the fixed-law hidden-bit control are not conflated.
- The memoryless promise is stated wherever a unique order is claimed.
- The environment-reset ablation is presented as a changed-process mechanism control.
- No data file or identifier supplies the true chronology to a comparator.
- Exact rational certificates replay independently; the verifier rejects a corrupted copy.
- All attempts and cumulative costs are preserved; no timeout or bug is labeled a physical impossibility.
- Mac execution, interpreter architecture, precision, and resource claims are supported by actual measurements.
- Any omitted or blocked check is explicit.
- The report contains a scoped outcome, its evidence, limitations, and the one next-step decision in §11.3.

Execute the authorized stages autonomously. Resolve routine implementation choices yourself, repair ordinary problems within the stated bounds, and stop once the evidence supports the scoped research conclusion. A verified limitation is a completed and valuable result.

