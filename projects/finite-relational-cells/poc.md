# Research and proof-of-concept plan: tiny finite-relation cells

**Research date:** 14 October 2026  
**Target:** one MacBook Pro with an M3 Max; actual memory, storage, operating system, and usable accelerators must be discovered.  
**Status:** researched proposal and executable experimental specification, not a demonstrated improvement.  
**Primary research question:** can training through a tiny, bidirectional, possible-value constraint cell learn reusable local Boolean operations better than strong directed logic/LUT learners and symbolic rule synthesis, when information and evaluation are matched?

## 1. Executive assessment and recommended direction

### 1.1 Decision

Build **one low-level PoC around a three-port finite-relation cell**. Each cell contains a learned two-input Boolean truth table and maintains possible values at its three connected wires. Unlike a directed numerical neuron, it can restrict any port using information at the others. Compose many copies through explicit wire bindings. Train the small tables from circuit input/output examples; then test unfamiliar compositions, reverse queries, incomplete evidence, and contradictions.

The coherent subset of the brainstorm is:

- **Constraint neurons:** remove unsupported possibilities rather than always emit an answer.
- **Temporary variable bindings:** compact wire identities connect reusable rules to particular operands.
- **Tiny executable rules:** an operation is an eight-tuple relation, not a large controller.
- **Explicit unknown and inconsistent states:** lack of information and incompatible information have different representations.

This is an **explicit narrowing** of the documents, not a claim to implement their complete “computational ecology.” Persistent world editing, learned question scheduling, neuron birth, macro discovery, counterfactual branches, and self-repair are outside the core. In particular, the proposal is **not QCR**: there are no learned recurrent clipped weighted sums, signed-change messages, reversal propagation, or query-conditioned approximation certificates. The core repeatedly computes ordinary finite-domain supports on a fixed graph.

### 1.2 Verdict on novelty

**The core primitive is substantially known.** Finite-domain constraint propagation, partial-information propagator networks, soft constraint semirings, logical neural networks, learned factor graphs, and differentiable logic/LUT networks cover its important ingredients. Terminal-agnostic logic also predates this proposal. Calling a table constraint a neuron does not create a new computational capability. [R1–R8, R11–R13]

The defensible research opportunity is narrower: **an empirical and training-method comparison of a particularly small support-propagation parameterization**, with explicit separation of learning quality from the reasoning backend. There is no demonstrated methodological novelty at present. A possible contribution would be a reproducible improvement in learning reusable rules or in the quality–cost frontier, attributable to the support-based learning mechanism rather than to an existing solver added afterward. A negative result establishing equivalence or lack of advantage is a valid completion.

The decisive design choice is to evaluate learned tables through a **common exact inference backend**. A directed LUT learner is allowed the same SAT or table-constraint wrapper as the proposed cell. Therefore, “our network can reason backward” cannot by itself win the comparison. The proposed learner must recover better transferable rules, or reach comparable quality at lower total cost. Native propagation is assessed separately.

### 1.3 Why this direction, despite the caveat

The cell preserves the documents’ strict small-unit ambition, has a complete mathematical specification, and permits inexpensive counterexamples and attribution tests. Its limitations can be exposed without perception, language, robotics, or a hidden planner. There is real scientific headroom in the **learning behavior**, but its existence in the selected workload is an assumption to test, not a finding. The strongest simple competitor—SAT/MaxSAT synthesis of the same small tables—may leave no headroom at all.

A living graph of facts plus requests has a plausible application, but it requires unresolved entity identity, update semantics, routing, and delayed credit assignment. A rule engine can supply most apparent intelligence. A self-growing ecology adds more confounds before establishing a useful primitive. The finite-relation direction is therefore the best **risk-adjusted first investigation**, not a prediction that it will outperform the others at all scales.

### 1.4 Claims this PoC can and cannot establish

It can establish correctness of the cell implementation, its known local-consistency properties, learning/generalization differences on specified finite rule families, and measured laptop quality–cost tradeoffs. It can investigate transfer to different circuit topologies.

It cannot establish general intelligence, open-ended abstraction, autonomous concept formation, superiority to conventional neurons on arbitrary tasks, physical neuromorphic efficiency, globally exact reasoning from local propagation on arbitrary graphs, or commercial value. Boolean circuits are a controlled test of reusable local computation; they are not evidence that perception, grounding, or natural-world reasoning has been solved.

### 1.5 Checks actually performed during this research

A small independent reference program was executed on **Linux x86-64, Python 3.13.5, NumPy 2.3.5, PyTorch 2.10.0+cpu; MPS was unavailable**. It passed:

| Check | Executed result | What it establishes |
|---|---:|---|
| All 16 two-input Boolean functions × all 64 three-port domain configurations | 1,024 cases matched exhaustive local support enumeration | Correct local truth-table/domain semantics in that reference implementation |
| Random connected factor trees with one, two, or three factors | 900 cases matched exhaustive global solutions | A finite sanity check of the tree-exactness property, not a proof for all graphs |
| Three-XOR contradictory reconvergent example | No global solution, but local propagation retained nonempty domains | A concrete completeness failure that the evaluation must preserve |
| Repeated-wire test `XOR(x,x)=1` | Contradiction detected when alias equality was enforced | Why binding identities cannot be treated as independent ports |
| One-cell, non-tied float64 automatic-differentiation check | Passed finite-difference gradient check | Local differentiability at that point only |

No M3 Max benchmark, deep training run, end-to-end superiority result, baseline reproduction, or independent external review was executed. Appendix A contains a standalone reproduction of the central checks; the handoff does not require access to the research session’s files.

---

## 2. Analysis of the source documents, problem statement, and hypotheses

### 2.1 Source inventory and evidential status

The inputs were three Markdown brainstorms, read in full. They contain proposals, examples, warnings, and links to research, but **no controlled measurements, complete learning algorithm, or demonstrated novel primitive**. Their tentative rankings are the authors’ judgments, not empirical comparisons.

| ID | Provided filename and title | Distinctive contribution to the discussion |
|---|---|---|
| D1 | `1(9).md`, *Beyond numerical neurons: Rethinking the fundamental unit of intelligence* | Moves from scalar signals to typed consequences; distinguishes knowledge-graph edits from computational rewiring; favors pointers, relationship edits, and questions. |
| D2 | `2(10).md`, *Beyond neurons: What if intelligence is not made of signals?* | Questions the processor itself; examines reactions, rewriting, constraints, equivalence, fields, compartments, differences, and obligations; emphasizes learning reusable transformations. |
| D3 | `3(9).md`, *Beyond artificial neurons: Exploring the entire design space* | Organizes six design axes, three notions of simplicity, five research programs, four learning levels, and a micro-event interface; makes hidden interpreter intelligence and credit assignment explicit risks. |

Source fingerprints, for provenance rather than as access requirements:

```text
D1 SHA256 46d6edc92e232b8b1a782d99f94adec2e7bd2a3142041661a60c9a952c165ff2
D2 SHA256 5e48d99f409302a7684e1110197741994db8d1c43e0f9daaa6131f6b9bf01e6a
D3 SHA256 ac1f1aafc346c3e1da05caab493906eaf11c6984ea9895c16ed6241607a03876
```

The evolution matters. D1 changes what a unit outputs. D2 changes which interactions count as computation. D3 assembles the ideas but recognizes that a vocabulary of actions is not a learning principle. This plan follows the most testable common thread: **a tiny learned local transformation, bound to explicit variables, whose composition has measurable consequences**. It does not treat biological metaphors or computational universality as evidence of learnability.

### 2.2 Detailed disposition of the brainstorm

“Defer” means not included in this PoC, not judged intrinsically unimportant. The table preserves the source terminology while distinguishing our engineering decision from the source’s preferences.

| Source concepts | Intended mechanism in the documents | Decision and technical reason |
|---|---|---|
| Pointer, reference, ownership | Share identity without copying representations; coordinate access | **Keep only compact wire references.** Binding identity is indispensable to composition. General entity resolution and mutable ownership are separate problems. |
| Variable-binding neurons; analogy through relational roles | Reuse one rule on different entities | **Keep exact binding and renaming tests.** No semantic analogy claim. Repeated ports must refer to the same variable. |
| Constraint, veto, negative-information, restriction units | Exclude incompatible candidates | **Primary primitive.** Small finite tables make soundness relative to the stored rule decidable and allow exact local tests. |
| Explicit unknown, contradiction, unresolved possibility | Distinguish absent evidence from conflicting assumptions | **Keep set semantics.** Do not confuse these states with probabilities or paraconsistent beliefs. |
| Operator neurons; local pattern → permitted transformation | Compose small learned operations | **Keep fixed-arity learned Boolean relations.** This gives a precise, small action family without an interpreter that supplies task rules. |
| Relationship edits, temporary truths, assertion/withdrawal, expiration | Maintain changing, evidence-linked world facts | **Defer persistence.** Initial inputs are immutable evidence per episode. Retraction requires resetting domains; otherwise monotone inference is unsound. |
| Witness and proof-fragment neurons | Preserve immediate justification and dependencies | **Keep diagnostic deletion records only.** These justify a local removal relative to a learned table, not the table’s truth. Full proof graphs are not needed for the learning test. |
| Question, obligation, attention, future conditional computation | Activate only needed work and install dormant rules | **Defer learned scheduling.** A fixed full-sweep schedule isolates the primitive and avoids QCR-like demand/certificate mechanisms. Fixed query masks are task inputs, not learned questions. |
| Router, temporary teams, active computational edges | Choose communication structure at runtime | **Defer learned routing.** Supplied graph topology is shared with every method. Otherwise routing capacity can explain a gain. |
| Birth, distinction, abstraction, chunking, reusable subcircuits | Grow a useful vocabulary or compile repeated behavior | **Defer.** Shared type tables provide reuse without creation. Growth must later beat fixed libraries, caching, synthesis, and compilation. |
| Equivalence, merging, e-graphs | Compress interchangeable computations without premature commitment | **Use only exact test equivalences and common compiler simplifications.** Learned approximate equivalence cannot be treated as logical identity. Equality saturation already supplies extensive machinery. [R21] |
| Hypothesis, counterfactual, compartment, local world-difference units | Maintain competing alternatives with sparse edits | **Defer explicit branches.** A possible-value domain is an overapproximation, not a population of consistent worlds; correlations are intentionally not represented. |
| Prediction, deadline, absence, curiosity | Detect violated expectations or acquire missing observations | **Defer.** The test has no real-time deadlines or observation policy. Missing evidence is explicitly represented rather than inferred from silence. |
| Resource bidders, computational economy, competing populations | Allocate effort according to anticipated benefit | **Defer.** Allocation itself needs a learning objective and calibrated cost model. It would confound a cell-level comparison. |
| Artificial chemistry and reactive blackboards | Computation through local encounters or event matching | **Retain only the local-relation perspective.** No random collision simulator or scheduler is required; those mechanisms must earn their own overhead. |
| Graph rewriting and interaction combinators | Change structure using a tiny vocabulary | **Defer topology rewriting.** Fixed bindings allow a cleaner falsifiable learning experiment; nontermination and rewrite selection remain unresolved in the source. |
| Synchronization and temporal binding | Encode relationships through timing | **Separate research direction.** Timing precision, collision capacity, and simulation cost are not tested by static tables. |
| Fields, morphogenesis, self-maintenance, repair | Distributed patterns preserve and restore computation | **Separate research direction.** Pattern regeneration is established; functional repair requires a different damage model and redundancy baseline. [R20] |
| Energy minimization and distributed negotiation | Reconcile preferences or supporting/opposing evidence | **Use as prior-art perspective, not extra machinery.** Factor-graph inference and weighted constraints already formalize related objectives. Correlated evidence must not be multiplied as independent. |
| Gradient units and representational failure | Change learning rules or introduce new representational primitives | **Defer.** The core uses conventional automatic differentiation and a fixed vocabulary. It does not claim fully local learning or ontology discovery. |

D1’s preferred relationship-plus-question architecture, D2’s fact/need/reaction/constraint ecology, and D3’s relational organism are therefore **not implemented literally**. Their higher-level operations are reduced to an explicitly identified finite relational substrate. This sacrifices persistence and open-ended reasoning to test the low-level idea requested here. [D1 §§3,6,8; D2 §§4,12,15–16; D3 §§8–12,13,16–17]

### 2.3 Beneficiaries, workflow, and bottleneck

The immediate beneficiary is a researcher or engineer designing small reusable computational layers for structured binary tasks. The reference workflow is:

1. Receive known wiring and examples of a system’s input/output behavior.
2. Learn the local operation associated with each opaque reusable type.
3. Deploy those operations in new compositions.
4. Answer both forward and partially specified inverse queries.

A scalar feedforward layer normally commits to one direction and one output representation. A relation layer makes possible-value restrictions available at every port. This may change inductive bias and training signals. It is **not an expressiveness impossibility for numerical networks**: an ordinary network can encode domains, and a learned directed function can be wrapped in an existing constraint solver. Those are the main alternatives.

The documents identify plausible bottlenecks—repeated reencoding, weak composition, conflated uncertainty, and irregular control—but provide no workload measurements establishing which dominates. This PoC tests **rule learning and composition**, not repeated memory reencoding. Practical demand for such a component remains unverified because no application was supplied.

### 2.4 Formal task

A circuit is a directed acyclic graph with `P` primary-input wires, `G` gate occurrences, `V=P+G` wires, and `K` reusable opaque types. Gate occurrence `f` binds ports `(a_f,b_f,c_f)` to wire IDs, has type `k_f`, and obeys

\[
x_{c_f}=h_{k_f}(x_{a_f},x_{b_f}),\qquad x_v\in\{0,1\}.
\]

Each type has four hidden truth bits. Topology, port roles, and type IDs are given to **all** methods; truth bits are not. Main training examples contain complete primary-input assignments and final-layer output labels, never intermediate wire labels. Type identities persist within a world, but are independently reassigned in independently generated worlds.

For query evidence `E`, let

\[
\mathcal S(C,h,E)=\{x:\text{all gates and evidence in }E\text{ are satisfied}\},
\quad D_q^*=\{x_q:x\in\mathcal S(C,h,E)\}.
\]

The answer is an exact domain: `∅`, `{0}`, `{1}`, or `{0,1}`. On inconsistent evidence, every exact query domain is empty. The evaluation knows the teacher tables; the learner does not. A query’s label is never included in its evidence.

Learning a good `h` and computing `D_q^*` are **different problems**. The primary comparison holds the second problem’s backend fixed. The native-cell comparison studies its incomplete but sound domain approximation relative to a specified hard `h`.

### 2.5 Established facts, assumptions, and speculation

**Established or mathematically checkable:** there are 16 two-input Boolean functions; table constraints implement them; generalized arc consistency removes locally unsupported values; local consistency is incomplete on general cyclic factor graphs; the proposed hard cell is a particular implementation of this established operation. Logical neural networks and learned logic gates already exist. [R1–R8]

**Assumptions requiring experiments:** end-to-end examples identify useful local tables; support-based gradients are trainable beyond one gate; the selected workload contains nontrivial query information; a measurable quality–cost advantage survives tuned LUT/logic and symbolic baselines; a faithful tensor implementation runs efficiently enough on the actual laptop.

**Speculation excluded from the initial claim:** higher-arity relational neurons might support richer reasoning; learned constraints might aid uncertain perception; adaptive composition might discover abstractions; custom hardware might exploit two-bit domains. None follows from a Boolean PoC.

### 2.6 Hypotheses and falsifiers

| ID | Hypothesis and predicted observation | Evidence/rationale | Alternative explanation | Observation that weakens or falsifies the local claim |
|---|---|---|---|---|
| H0 | Hard cells preserve every solution of their stored relations and are exact on acyclic factor graphs after closure. | Standard support reasoning; proof sketch in §4; executed small checks. | An oracle or labels accidentally entered inference. | Any counterexample to soundness on a correctly specified instance is an implementation defect; any claimed arbitrary-cycle exactness is a theoretical error. |
| H1 | Support-based training learns tables that produce lower unfamiliar-composition domain error than strong directed learners at matched information. | Different relaxation and backward information paths; related differentiable reasoning literature. No direct supporting benchmark exists for this exact comparison. | Different parameterization, loss, initialization, discretization, tuning, or extra compute. | Gain disappears under same-table parameterization, shared exact inference, or stronger directed training. |
| H2 | H1, if present, yields a useful quality–cost frontier against symbolic table synthesis as well. | Small learned local models may offer an optimization alternative; entirely unverified. | Symbolic baseline was timed out early or encoded poorly. | Exact or adequately converged synthesis matches quality sooner or cheaper; no compensating tradeoff remains. |
| H3 | Native cells are useful partial-information components under their explicitly limited semantics. | Bounded state and known consistency properties. | A generic table propagator supplies exactly the same functionality. | Equality with the generic propagator invalidates a new-primitive claim, though not correct implementation; worse performance invalidates an engineering advantage. |
| H4 | Effects transfer beyond favorable random graphs. | Shared local types ought not depend on wire names or exact graph size. | Saturated Boolean functions, memorized topology, small independent-input space. | Failure under renaming, increasing independent inputs, different fanout, or controlled reconvergence; synthetic-only success narrows the result. |

### 2.7 Strongest case for and against

**For:** the representational state is small and interpretable; exact identity binding enables reuse; a single relation can be queried in several directions; inconsistent and underdetermined situations remain distinguishable. A small truth-table learner can be compiled and inspected. The evaluation can locate a learning advantage without requiring a huge task system.

**Against:** most of the benefit is an existing constraint solver. Four-parameter LUTs already match the local function class. Max/min gradients can be sparse, tied, or bottlenecked. Repeated evidence can constrain all scores without teaching the correct internal rule. Partial domains lose correlations, so larger circuits can be globally inconsistent while locally stable. Hardening an imperfect rule can remove the true answer. Irregular support computation may be slower than a forward lookup and exact symbolic synthesis can dominate the whole task. These are central competing explanations, not appendix caveats.

**Headroom:** a library drawn uniformly from all 16 gates has `4K` bits of prior uncertainty and `16^K` possibilities. At least that much informative supervision is necessary to identify it in the unrestricted case; correlated outputs and unreachable input combinations can require much more or make identification impossible. For `K=4`, exhaustive dictionary enumeration has only 65,536 candidates and is a required diagnostic. Larger `K` is not automatically harder in practice: SAT may exploit the examples very effectively. If the strongest baseline already reaches zero exact-domain error, there is no accuracy headroom in that condition.

---

## 3. Research landscape and precise novelty assessment

### 3.1 Closest-prior-art comparison

The important comparison is mechanism and assumptions, not similarity of names. “Native” below refers to the proposed cell’s support propagation, not to novel theory.

| Alternative | Problem/mechanism and assumptions | Evidence and computational character | Implementation/access | Exact overlap and consequence |
|---|---|---|---|---|
| Classical consistency/table constraints [R1] | Variables carry finite domains; relations remove unsupported values. Rules are given. | Established local-consistency algorithms; global search can still be required. | Straightforward independent reference and mature SAT/constraint solvers. | **Exact overlap with hard inference.** This is the equivalence control, not a weaker baseline to outperform by renaming. |
| Propagation Networks [R2] | Cells accumulate partial information; propagators communicate consequences. | General substrate, including dependencies and incomplete knowledge. | Thesis and framework descriptions inspected; no performance replication here. | Direct conceptual antecedent of small information-accumulating units. Learning a four-bit relation is an added task, not a new substrate. |
| Semiring/fuzzy constraints [R3] | Combine and project soft relations using algebraic operations. | Covers families of soft consistency/optimization semantics. | Primary bibliographic/abstract evidence; our max/min equations are specified independently. | Max/min support relaxation is not new merely because automatic differentiation is applied. |
| Logical Neural Networks [R4] | Meaningful logical neurons, truth bounds, omnidirectional inference, learned weighted logic. | Explicit unknown/contradiction treatment; propositional/first-order scope differs from random hidden tables. | Paper algorithms and selected figures inspected; IBM Apache-2.0 implementation and installation metadata inspected. | Directly invalidates novelty of bidirectional, uncertainty-aware logical neurons. Use as a semantics reference; do not claim full-framework superiority from this task. |
| DLGN [R5,R6] | Learn a categorical choice among 16 two-input gates on supplied sparse connections; relax during training, harden for inference. | Efficient compiled inference; learning/discretization limitations. Published image workloads are not our controlled task. | MIT repository; Python mathematics available, but main module unconditionally imports CUDA extension. | Same local function class. Needs faithful CPU/MPS adaptation, correct gate-index mapping, and contemporary tuning. |
| Gumbel/STE logic gates [R7] | Hard sampled forward gates with soft surrogate gradients. | Studies discretization and training on image tasks. | Paper method and pseudocode sections inspected. | Mandatory strong training alternative; comparing only with old softmax training would be weak. |
| Fully trainable LUT/logic networks [R8] | Direct sigmoid LUT entries, annealing, STE, optional learned connections. | 2026 paper reports deeper training and connection optimization. | Methods inspected; no release or local reproduction assumed. | Four parameters per two-input LUT already exist. We supply connections equally, so connection learning is outside the controlled claim. |
| Task-aware discretization [R9] | Use downstream task impact rather than only local confidence to freeze gates. | Recent preprint emphasizes limitations of large first-order interventions. | Method inspected; no local results reproduced. | Requires a shared task-aware hardening control. On this tiny library, exact candidate interventions are cheaper and less ambiguous than a large-network approximation. |
| GLIDR [R10] | Learn graph-like logical rules; propagate variable domains through differentiable consistency approximations. | Closely related learned-domain mechanism; cycles and large open-query costs limit it. | Detailed method text inspected; a matching official implementation was not established in this search. | Strong overlap with learned relations and consistency. No “first differentiable constraint neuron” claim is defensible. |
| Belief propagation / learned factor inference [R11] | Local factor-to-variable reasoning; learned message operators can preserve selected BP properties. | Probability/counting tasks differ from exact possible-value domains. | Primary paper/abstract available; not a complete local reproduction. | Factor graphs and learned local inference are established. Probability messages must not be treated as exact support certificates. |
| SATNet [R12] | Differentiable smoothed MAXSAT through a semidefinite relaxation. | Learns logical structure; heavier global solver-based layer. | Paper and official code link inspected; target build not tested. | Replaces a layer with logical optimization already. Not required as the microscopic matched competitor when exact table synthesis is stronger and simpler for this task. |
| Learned constraints plus exact solver [R13] | Predict discrete graphical-model costs/constraints and solve separately; E-PLL improves constraint learning. | Puzzles, partial observation, optimization/protein-design settings with different data assumptions. | Method/assumptions inspected; source contains placeholder journal metadata, so cite arXiv ID, not the placeholder DOI. | Separating learning from an exact reasoning backend is established. It motivates fairness rather than a novelty claim. |
| SAT/MaxSAT table synthesis [R14,R15] | Shared unknown truth bits plus per-example hidden wires; solve or minimize label disagreements. | Exact finite hypothesis-class competitor, with difficult instances possible. | PySAT and RC2 documented; macOS build prerequisites inspected. | Strong simple competitor for the actual learning problem. Must not receive less information or an artificial failure deadline. |
| Self-organizing logic gates [R16,R17] | Terminal-agnostic gate constraints implemented with dynamical circuit elements and memory. | Physics-inspired simulation and patent disclosure, not proof of our learning claims. | Original abstract and patent description inspected. | “A gate usable backward” is old. No physical implementation or complexity claim is adopted. |
| Neural Production Systems / NPI / NLM [R18,R19,R22] | Learned entity-conditioned rules, reusable procedures, or relational operations. | Relevant composition precedents with different representations and supervision. | Primary abstracts inspected; not reproduced. | Broader brainstorm combinations already have neural-symbolic precedents; these are contextual, not mandatory unrelated benchmark implementations. |

### 3.2 What can be reused

Reuse exact truth-table semantics, automatic differentiation, standard optimizers, PySAT/RC2, existing logic-gate relaxation mathematics, CPU compilers, and established statistical tools. Adapt the official DLGN Python operations rather than reinstalling a CUDA-dependent package blindly. Preserve required license notices. A generic finite-table propagator should be implemented independently as a small correctness oracle, not imported from the same optimized code path.

Do not rebuild a general theorem prover, graph database, differentiable SAT solver, neural compiler, or experiment-tracking service. The repository and JSONL artifacts specified below are sufficient. No pretrained model is required. All project-specific learning costs therefore remain visible.

### 3.3 Six explicit novelty answers

1. **What exists?** Every core inference operation, the small LUT function family, several differentiable logical learning approaches, and bidirectional logical neurons.
2. **What is reusable?** Their mathematics and permitted code, plus mature exact solvers and ordinary tensor tooling.
3. **What is substantively different?** At most, the particular controlled coupling of four-entry relation learning, max/min support training, shared type bindings, and solver-normalized out-of-composition evaluation. This is a candidate empirical/training contribution, not a demonstrated new algorithmic class.
4. **Why might it matter?** A different relaxation could favor reusable discrete rules or make certain credit paths easier. It might instead train worse. The claim is about observed learning, not representational magic.
5. **What isolates it?** Same topology, same labels, same table family, same hardening controls, and same exact query solver; change only the training executor/relaxation. Then ablate reverse messages and table parameterization.
6. **What invalidates novelty?** An equivalent published training/update scheme; equality with the generic implementation without learning gains; an improvement fully explained by an existing optimizer or hardening technique; or no advantage against symbolic synthesis. The correct result label would then be replication, engineering study, or negative finding.

### 3.4 Reconciliation of apparently conflicting evidence

DLGN-related papers report different discretization and training outcomes under different depths, losses, architectures, initialization, and hard-forward training. Their numbers are not contradictory controlled measurements of the same configuration. The new plan therefore includes soft, deterministic-STE, and Gumbel-STE variants, with task-aware hardening shared across methods. It does not import a published speedup into the M3 estimate. [R5–R9]

SATNet’s original visual-task claims were challenged when intermediate symbol labels were removed; later symbol-grounding work changed the training method. This does not establish that all SATNet variants fail, nor that a symbolic layer alone solves grounding. Our task deliberately supplies identities and wiring, discloses that assistance, and never presents it as learning grounding. [R12,R23,R24]

Local constraint propagation can be reliable and useful while failing global consistency. Both statements are true. Likewise, exact inference of an incorrectly learned model is not exact inference of the true process. These distinctions determine the two evaluation tracks below.

### 3.5 Licensing and patent caution

Open-source code permission, downloadable research text, dataset redistribution, and patent rights are different matters. The inspected DLGN repository labels its code MIT and also carries a patent-pending notice; IBM LNN is Apache-2.0. A related terminal-agnostic-logic patent was located. These observations are not a legal status assessment, patentability determination, or freedom-to-operate opinion. Preserve notices, verify the exact revision’s terms, and do not redistribute assets with unverified terms. [R6,R17,R25]

---

## 4. Proposed mechanism and implementation specification

### 4.1 Hard state, parameters, and graph invariants

Represent a domain with a two-bit unsigned integer:

| Code | Domain | Meaning |
|---:|---|---|
| 0 | `∅` | No consistent value remains |
| 1 | `{0}` | Known false |
| 2 | `{1}` | Known true |
| 3 | `{0,1}` | Underdetermined |

Bit `b` indicates that Boolean value `b` is possible. This is a possible-value encoding, not IEEE/Verilog high-impedance logic and not a probability. It is not a claim of a new dual-rail representation.

Canonical gate code `g∈{0,…,15}` has

\[
h_g(a,b)=(g\gg(2a+b))\ \&\ 1.
\]

Thus canonical `AND=8`, `XOR=6`, and `OR=14`. The official DLGN implementation uses a different numbering: its `AND=1`. **Generate an explicit mapping from the four Boolean evaluations; never copy integer gate IDs across implementations without conversion.** [R6]

Each relation contains the four tuples

\[
R_g=\{(a,b,h_g(a,b)):a,b\in\{0,1\}\}.
\]

An arbitrary ternary Boolean relation would need eight independent bits, but that larger family is not necessary for the initial PoC. The restriction to deterministic two-input functions matches the directed baselines exactly.

Graph invariants:

- Wire IDs are integers `0…V−1`; each non-input wire has one producer in the forward circuit.
- Every gate output has a higher topological index than both inputs.
- A type ID selects one shared table; occurrences do not have hidden extra parameters.
- Repeated input ports are legal. If ports refer to the same wire, tuples assigning them different values are illegal.
- An episode’s evidence is intersected once into fresh domains. Duplicate contradictory observations produce an empty domain, not last-write-wins behavior.
- There is no cross-episode hidden state. Evidence removal or correction starts a new propagation state.

### 4.2 Hard local update

For a factor `f` and current domains `D`, compute the compatible tuple set

\[
A_f(D)=\{t\in R_{k_f}:t_p\in D_{v_{fp}}\ \forall p,
\quad v_{fp}=v_{fq}\Rightarrow t_p=t_q\}.
\]

For every **unique bound wire** `v` in the factor, replace

\[
D_v\leftarrow D_v\cap\{t_p:t\in A_f(D),v_{fp}=v\}.
\]

Compute all port supports from the same pre-update factor snapshot. Intersect simultaneous constraints on an aliased wire. Do not mutate one port and then evaluate another from partly updated tuples inside the same cell unless the implementation is separately proved equivalent at closure.

The minimum complete implementation uses deterministic sweeps over all factors until a sweep removes nothing. A synchronous batched version must reach the same closure. No learned queue or query-dependent scheduler is part of the contribution.

```python
# Specification pseudocode; project code must validate shapes and indices.
def revise_cell(domains, ports, relation_tuples):
    allowed = []
    for t in relation_tuples:                  # no more than 8 tuples
        aliases_ok = all(
            ports[p] != ports[q] or t[p] == t[q]
            for p in range(3) for q in range(p)
        )
        domains_ok = all(domains[ports[p]] & (1 << t[p]) for p in range(3))
        if aliases_ok and domains_ok:
            allowed.append(t)
    updates = {}
    for v in set(ports):
        p = ports.index(v)
        support = 0
        for t in allowed:
            support |= 1 << t[p]
        updates[v] = domains[v] & support
    return updates
```

At closure, return both local domains and `proved_inconsistent = any(domain == 0)`. If any component is inconsistent, the **global exact-domain answer** is empty even for a query in a disconnected component. Do not require an empty domain to physically propagate across disconnected components. If no empty domain occurs on a cyclic factor graph, return “not refuted,” not “satisfiable.”

### 4.3 Mathematical properties and limits

**Soundness relative to stored hard relations.** A global satisfying assignment supplies a compatible tuple at every incident factor. Its value therefore cannot be removed by a support update. Induction over updates proves preservation of every global solution.

**Termination.** Hard domains only lose values. At most `2V` value deletions can occur. Every nonterminal full sweep must delete at least one value, so `2V+1` sweeps is a conservative bound for an implementation that scans every factor fairly. This is not a bound on SAT search and not a guarantee of efficient global reasoning.

**Tree exactness.** In an acyclic bipartite factor graph, a retained value can be extended through supporting tuples recursively without reconvergent conflicts; nonempty arc-consistent domains therefore give exact marginal supports. Treat disconnected components separately and aggregate inconsistency globally. A forward DAG is **not necessarily** an acyclic factor graph.

**Counterexample.** With all `x,y,z` initially unknown, impose `x XOR y=0`, `y XOR z=0`, and `x XOR z=1`. Every individual relation has local supports, yet no global assignment exists. Native closure does not establish satisfiability. The executed reference reproduced this counterexample.

**No information creation.** A nonempty two-value domain is not a hidden guess. Local unary domains cannot encode arbitrary correlations between wires. Recovering those correlations may require larger clusters, branching, or an exact solver. Adding such machinery later changes the resource model and must be compared with existing stronger consistency/search methods.

### 4.4 Differentiable relation parameterization

For type `k`, train four logits `θ[k,ab]`:

\[
p_k(a,b)=\sigma(\beta\theta_{k,2a+b}),\qquad
r_k(a,b,1)=p_k(a,b),\quad r_k(a,b,0)=1-p_k(a,b).
\]

`β=1` is the initial relaxation. A validation-selected annealing schedule is a shared option for both the relation learner and the direct LUT baseline. There are `4K` parameters; no hidden controller. A categorical 16-gate relation parameterization is an attribution control, not a separate invention. Direct sigmoid LUT entries are prior art. [R8]

Let `s[v,b]∈[0,1]` be a **support score**, not a calibrated probability. Known evidence `x_v=b` initializes scores to the corresponding one-hot vector; unknown evidence initializes `[1,1]`. For target port `p`:

\[
m_{f\to v_{fp}}(b)=
\max_{t:t_p=b,\;\text{alias-consistent}}
\min\left(r_{k_f}(t),\min_{q\ne p}s[v_{fq},t_q]\right).
\]

An empty maximum is zero. All new messages are computed from the previous synchronous state, then

\[
s^{t+1}[v,b]=\min\left(s^t[v,b],\min_{f,p:v_{fp}=v}m_{f\to v}(b)\right).
\]

The minimum over no incident messages is one. Alias consistency is applied before projection. For a repeated bound wire, using its current score in another aliased port is harmless under the final intersection, but inconsistent tuples must still be excluded.

Use fixed, explicitly recorded unroll length `T` during training. Do not use a floating-point “convergence certificate.” Hard inference uses discrete closure after table extraction. Max/min are piecewise differentiable; tie gradients depend on the chosen reduction and must be tested. Preserve the same tie convention on CPU and MPS or use the CPU path.

A generic soft table-constraint implementation with these equations is **mathematically equivalent**. It must match, not serve as a deliberately slower straw man. The proposed investigation is whether this training computation is useful relative to other learners.

### 4.5 Learning objective and gradient conventions

For an output label `y`, the target membership vector is `[1−y,y]`. The initial shared objective is mean squared membership error over observed final outputs:

\[
L_{\rm mse}=\frac{1}{2|Q|}\sum_{q\in Q}\sum_{b=0}^1(s[q,b]-\mathbf 1[b=y_q])^2.
\]

A directed learner returns scalar `z_q` and is evaluated in the same objective through `[1−z_q,z_q]`. Strong baselines must also be allowed their better validation-performing binary-cross-entropy objective, and the relation learner receives the same choice. This prevents a convenient shared loss from weakening a baseline.

For membership BCE, avoid a hard numerical clamp that kills gradients at hard-forward outputs. Use `u=ε+(1−2ε)s` inside logarithms, with an explicitly recorded `ε` (initial numerical test value `1e−6` in float32; verify representability and finite gradients). This is a bounded surrogate loss, not a claim that the two support scores form a joint probability. Report sensitivity to the numerical floor if it changes learned tables.

Initialize four-entry logits with small independent, zero-centered noise, not all exact zeros; the latter creates pervasive max/min ties. Include both unbiased and pass-through-biased initialization where valid, giving the same prior to equivalent directed methods. Bias strength is validation-selected. No teacher gate identity may influence initialization.

Track per-type gradient norms, fraction of zero gradients, score sums, min/max saturation, hardening margin, and the number of factors on active gradient paths. Deep trainability is unresolved. Automatic differentiation is global training machinery; this PoC does **not** claim a fully local credit-assignment rule.

### 4.6 Shared directed learners

**Four-entry LUT.** For scalar inputs `a,b∈[0,1]`, use

\[
z=\sum_{u,v\in\{0,1\}}p_k(u,v)
[a^u(1-a)^{1-u}][b^v(1-b)^{1-v}].
\]

For Boolean inputs this selects the corresponding entry. This is a multilinear relaxation, not an exact probability calculation when reconvergent inputs are dependent. Use ordinary float32 operations without claiming probabilistic independence.

The `lut4` baseline family must include both soft entries and **entrywise hard-forward/soft-backward entries**, not just the soft multilinear model. Set `p_hard = (p >= 0.5)` and use `p_hard + p - p.detach()` in the forward circuit for the STE variant; retain the sigmoid path for backward differentiation. Test both fixed and validation-selected annealed sigmoid slope. All four hard entries must be shared consistently by type. This is an explicit local adaptation of contemporary LUT learning, not a reproduced large-network architecture. The analogous hard-entry support variant is an attribution option for `frc4`, with its selection privilege and cost matched. [R8]

**Categorical logic gate.** Let `π_k=softmax(α_k/τ)`, and compute `Σ_g π[k,g] f_g(a,b)`, where `f_g` is the same multilinear extension of each gate. Include:

- standard soft training;
- deterministic hard-forward/soft-backward selection;
- Gumbel hard-forward/soft-backward selection. [R5–R8]

Implement straight-through selection as `hard + soft − soft.detach()`. Sample Gumbel noise on the CPU from a saved generator. Because type parameters are shared, sample **one selection per type per minibatch and reuse it across occurrences and examples**. Independent per-occurrence samples would change the hypothesis class into inconsistent copies of the same rule.

**Tiny numerical cell.** A shared type-specific `2 → H → 1` tanh or ReLU MLP followed by sigmoid, initially `H=8`, is the strong simple conventional alternative. Permit validation expansion/shrinkage of `H` when capacity or cost is material. A single linear threshold cannot represent XOR, so it is only a negative capacity control. Export the MLP’s four Boolean responses as a hard table; it then receives the same inference wrapper.

### 4.7 Hardening and a mandatory common refinement

Initial extraction uses `p≥0.5` for four-entry models and argmax for categorical gates. Resolve ties deterministically and log margins. For MLPs, evaluate the four Boolean input pairs before thresholding.

Then run a **shared discrete coordinate-refinement control**: cycle through type IDs in a seeded fixed order; try all 16 hard tables for that type; evaluate exact forward training-label disagreement; accept only a strict improvement; repeat until a full pass has no improvement. Preserve the current table on equal loss. The finite objective and strict decrease prevent cycles. Record cost, all candidate evaluations, and results before/after refinement.

For the tiny shared library, exact downstream interventions are a practical stronger check than relying only on local confidence or a first-order freeze score. This is an engineering control inspired by existing task-aware discretization, not a claimed reproduction of its full large-network algorithm. [R9] Choose raw versus refined checkpoints using validation only, for every learner. A gain erased by this common refinement is not evidence for the proposed primitive.

### 4.8 Exact teacher and common query solver

Two independent implementations must agree:

1. Enumerate all `2^P` primary-input assignments, execute the hard DAG, retain assignments matching query evidence, and OR their query-value bits.
2. Encode the same hard tables in CNF and ask satisfiability under `E∧(q=0)` and `E∧(q=1)`.

A false answer to both queries proves an empty domain only if the solver finished. Timeout or interruption is `UNRESOLVED`, never false or empty. Amortize CNF construction per circuit, but report construction separately and include it in cold/end-to-end measurements.

At query time, **every learned-table method uses this same backend in the primary track**. Teacher tables are loaded only by the isolated labeling/evaluation process. They cannot enter training, checkpoint selection, or learner-side simplification.

### 4.9 Symbolic rule synthesis, specified rather than deferred

Create shared Boolean variables `z[k,u,v]` for the four truth bits of each type. Each training example has its own wire variables. For each gate occurrence, each input pattern `(u,v)` implies that output `c` equals `z[k,u,v]`:

\[
(a=u\land b=v)\Rightarrow(c\leftrightarrow z_{k,u,v}).
\]

Encode this as two clauses. If `L(a,u)` means literal `a` when `u=1` and `¬a` otherwise, the clauses are

```text
¬L(a,u) ∨ ¬L(b,v) ∨ ¬c ∨ z[k,u,v]
¬L(a,u) ∨ ¬L(b,v) ∨  c ∨ ¬z[k,u,v]
```

There are eight such clauses per gate/example before simplification. Deduplicate literals; tautologies caused by aliased ports may be removed. Clamp training primary inputs with hard unit clauses. In the noiseless condition, output labels are hard clauses. With label noise, output labels are unit-weight soft clauses and all circuit semantics remain hard; RC2 minimizes disagreements. No hidden state label is added.

For `K≤4`, enumerate the complete dictionary hypothesis class as a correctness and optimal-loss check. For larger cases, use SAT/MaxSAT. Record whether optimum is proved, incumbent cost, and available bounds; never label an unfinished incumbent optimal.

Identifiability is tested by fixing the optimum training cost and asking whether each shared truth bit can take both values in an optimal model. Use complete enumeration when feasible, otherwise additional solver calls. Non-identifiable tables are not counted as recovery failures; evaluate predictive behavior and report the ambiguity. Use a strong, explicit symbolic model-selection rule rather than an arbitrary prefix of tied models: among training-optimal dictionaries, minimize validation-label disagreement with a second exact solve. For clean training data, keep training labels hard and make validation labels unit-weight soft. For noisy training data, assign each training-label soft clause weight `N_val_bits + 1` and each validation-label clause weight one; this implements lexicographic training-then-validation minimization. Include separate wire variables for validation examples but share the same truth bits. The validation labels were already equally available to all methods for selection; this stronger search is not extra information, and all of its compute is counted. Record whether each optimization stage completed; an unfinished selection remains an incumbent, not an optimum. Break any remaining ties by a fixed seed/solver policy independent of the teacher and protected tests. Never choose the candidate closest to the hidden teacher. Training-only identifiability and post-validation identifiability are distinct diagnostics and must be labeled separately.


---

## 5. Baselines, data, metrics, and confirmatory evaluation

### 5.1 Two tracks that must not be conflated

**Track A — learning contribution, primary.** Train each learner on identical circuit input/output examples. Extract its reusable tables. Answer held-out queries with the same exact solver. The primary claim is a difference in learned-model quality under unfamiliar composition, not a difference in solver power.

**Track B — native primitive behavior, secondary.** Given the same hard tables, compare the proposed native propagator, an independent generic table propagator, and exact inference. Measure soundness, incompleteness, state size, and latency. The first two should agree. A speed difference is an implementation result; identical behavior is not a new scientific mechanism.

A combined deployment may use native propagation followed by exact completion when necessary. Its complete cost includes both stages and it must compete with direct exact solving. Never compare native approximate answers with an exact baseline as though their semantics were identical.

### 5.2 Mandatory locally executable comparisons

| Method ID | Role | Required implementation/configuration | Fairness requirement |
|---|---|---|---|
| `frc4` | Proposed learner | Four-entry sigmoid relation, max/min support unroll, shared types | Report soft and hardened behavior; no extra type-specific network. |
| `lut4` | Closest capacity-matched directed learner | Four-entry multilinear LUT; same type sharing and optional sigmoid annealing | Same labels, initialization family, parameter count, optimizer opportunities, and exact query wrapper. |
| `logic16` | Established logic-gate learner | Softmax gate mixture; standard, deterministic STE, and Gumbel STE training variants | Include high-learning-rate and deep-gradient options; adapt current methods rather than only the weakest original setup. [R5–R8] |
| `mlp_cell` | Strong simple numerical component | Type-shared two-input MLP, starting hidden width 8; four-input truth-table export | A nonlinear MLP, not an incapable single threshold, is the competitive baseline. |
| `synth` | Strong simple symbolic learner | Exact enumeration for small `K`; SAT/RC2 encoding in §4.9 | Equal observations, correct shared parameters, optimizer/solver status recorded, no deliberately tiny timeout. |
| `generic_table` | Equivalence control | Independently written tuple-support propagator | Same tables and closure; discrepancy blocks conclusions. |
| `teacher_exact` | Attainable-quality ceiling and label oracle | True hidden tables plus exhaustive/SAT query support | Extra information clearly labeled; never described as a learned competitor. |
| `unknown`, `majority` | Sensitivity controls | Always return `{0,1}`; or validation-majority query-domain class | Reveal trivial query distributions and label collapse. |

Use a validation-selected variant within `logic16`, but report every variant tried, including failed runs. Preserve the family-level comparison and the best competent result, not just a convenient individual configuration. Common discrete coordinate refinement is applied to all table-exporting learned methods and evaluated as a separate declared option.

**Additional required attribution models:** `frc16` (categorical tables with support execution) and `frc4_forward_only` (same tables with only output-port support updates). They separate parameterization, support algebra, and reverse messages. A known-function one-threshold model is a diagnostic capacity negative control only.

### 5.3 Relevant broader baselines and justified exclusions

IBM LNN is a **required small semantics-reference check** if its isolated CPU environment passes its own tests: construct known AND/OR/NOT/XOR formulas and check forward, reverse, unknown, and contradictory inputs. Do not declare general LNN inferential completeness based on one local example or assume its truth-bound semantics are identical to set domains. A failed installation is recorded; the independent hard table control still provides a faithful core correctness path. No performance claim against LNN is permitted without its functioning implementation. [R4,R25]

GLIDR is a close conceptual/learning comparison, but its knowledge-base rule language and entity enumeration are not the same task as discovering four truth bits on supplied circuit wiring. A matching official implementation was not established here. Reimplementing a subset would be labeled an adaptation, not a faithful full replication. There is **no claim of outperforming GLIDR** in this PoC. [R10]

SATNet, E-PLL graphical-model learning, Neural Logic Machines, and Recurrent Relational Networks provide broader context. Their original benchmark numbers are not controlled head-to-head results here. Exact table synthesis is a more directly matched and often stronger executable competitor than forcing an unrelated large puzzle architecture into this small hypothesis class. [R11–R13,R22,R26]

A generic recurrent factor-graph network becomes mandatory **only if a later claim is made about native learned masked-query answering rather than exported-rule learning**. Its specification is: variable nodes receive two evidence bits and a query flag; factor nodes receive type embeddings and ordered port indicators; shared MLP messages plus GRU updates, initial hidden width 64, two output membership logits per query, same supervision and unroll opportunities. Such training requires valid domain labels for masked queries; it is not silently added to the core. Its parameters, runtime, and privileged label generation must be counted. This branch requires a methodological amendment and fresh confirmation.

No LLM, Transformer, retrieval system, paid model, remote accelerator, or pretrained perception asset is needed.

### 5.4 Core synthetic dataset, completely specified

Name the project-generated dataset `typed_boolean_circuits`; its version is the generator source commit plus a SHA256 of the generation configuration, not an invented external release.

**Independent world.** Sample a new hidden type library and new training/validation/test topologies. One world is the principal independent statistical unit. Models are trained afresh within it; no learned parameter transfers from exploratory to confirmatory worlds.

**Initial local calibration setting.** `K=8`, width `W=8`, training depths `d∈{2,4}`. These give 32 hidden truth bits, 8 independent primary inputs, and 16 or 32 gate occurrences. This is a starting scientific scale, not a maximum. `K=4` is the earlier exhaustive-synthesis diagnostic. Increasing `K`, width, or depth must address a specific remaining uncertainty, not merely seek a favorable result.

**Layered DAG generator.** The `W` primary inputs form layer zero. Each later layer has `W` gates. For each gate, independently draw two ordered predecessor indices uniformly from the preceding layer, with replacement. Draw its type uniformly from `0…K−1`. Output ports are new unique wire IDs; all final-layer wires are labeled outputs. Replacement permits exact alias tests; sharing permits reconvergence. Retain constant functions, unused primary inputs, repeated gates, and unbalanced output frequencies. Record them instead of deleting inconvenient examples.

**Two predeclared teacher strata.** Half the independently generated worlds draw each type uniformly from all 16 Boolean functions. Half draw from the 10 functions that genuinely depend on both inputs. Compute the latter set by exhaustively checking dependence, not by a handwritten name list. Both strata are equally weighted in the primary summary and separately reported. They test raw breadth and genuinely two-input composition without selecting only parity. The learner’s allowed table family remains all 16 in both; regime metadata is not a predictive input. The design and mixture are public and identical for every baseline.

**Training records.** For each topology, sample primary-input vectors uniformly without replacement until all `2^W` are used; further examples require a new topology. Supply all final-layer output labels. No internal activations, proof traces, teacher truth tables, or masked-query support labels are supplied. Count supervised **output bits**, not just records. Also report unique topologies, input vectors, and type-occurrence counts.

**Label ladder.** Begin the pilot with `K` topologies, half at each training depth, and 16 distinct input assignments per topology. Double the number of topologies to add labels, preserving previous data as a prefix, until the pilot characterizes both the low-information regime and the practical learning/identifiability plateau. Smaller prefixes are retained. When adding assignments rather than topologies, record the change as a separate acquisition policy: repeated examples on one topology and new compositions provide different information.

**Validation.** Generate disjoint topologies at the training depths, with one-quarter as many supervised output bits as the associated training corpus, rounded up to whole records. These labels are equally accessible for model/configuration selection and are included in the information budget. No validation topology is reused for testing. The ratio is an initial allocation for separating fitting from selection, not a performance target; pilot variance can justify changing it before confirmation.

**Primary held-out topology depths.** `2d_max` and `4d_max`, initially 8 and 16, at the same width. These specifically test reuse beyond the trained composition depth. Report the complete depth curve, not only the best depth. The final confirmation configuration fixes all sizes before protected labels are inspected.

**Independent-input scaling is mandatory robustness.** Repeat a diagnostic at doubled width, initially 16, and at larger widths when the oracle/resource pilot supports them. Increasing depth with width fixed keeps only `2^W` possible input assignments; it is not evidence of scaling to arbitrarily many independent inputs.

### 5.5 Query generation and ground truth

Generate a full teacher-consistent wire assignment by sampling primary inputs and executing the hard teacher DAG. Then form three equally weighted query strata:

1. **Forward:** observe all primary inputs; query a uniformly selected final-layer wire.
2. **Reverse:** observe all final-layer outputs; query a uniformly selected primary input; observe no other primary input.
3. **Mixed:** select a query wire uniformly; independently reveal each other wire with probability one-half, using the teacher-consistent value. The query wire is always unobserved.

Truth is the full possible-value domain under this evidence, not the particular sampled assignment’s value. A reverse or mixed query can legitimately have two answers. Training a point label from that one sampled hidden state would falsely punish legitimate ambiguity; do not do it.

Generate exact labels by the independent methods in §4.8. Keep the sampled full assignment in the oracle-side record only. Learner-facing records contain topology, type IDs, evidence, and query IDs. Multiple query masks from one circuit are dependent observations, not independent replicates.

For contradiction diagnostics, add an explicitly contradictory observation, or flip a selected revealed bit and ask the exact oracle whether the resulting evidence is inconsistent. A flip is not automatically a contradiction. Report separately the mechanically contradictory and globally inconsistent-but-locally-plausible cases. Preserve the XOR counterexample as a mandatory adversarial fixture.

### 5.6 Protected evaluation and leakage controls

Separate namespaces and directories:

```text
explore/*        implementation, debugging, tuning, sample-size pilot
confirm/train/* learner-visible fitting data for new worlds
confirm/val/*   learner-visible checkpoint/configuration selection data
confirm/test/*  protected topology/evidence records and oracle-only labels
replicate/*     independent confirmation after a positive or amended claim
```

Generate confirmatory seeds after freezing the protocol, store their hash before execution, and do not use their outcomes to choose depth, labels, method variants, hyperparameters, or stopping rules. The evaluator may generate protected labels on demand, but must not print per-method scores before the frozen run batch finishes. This is an operational separation, not a claim that filesystem permissions can prevent a determined researcher from looking.

Rename type IDs and wires independently of functions. Strip function-revealing names from any imported circuits. Hash topologies and compare train/validation/test for exact duplicates; report near-duplicate structural motifs rather than pretending all motifs can be disjoint when the task is reuse. Shared local rule types are intended; repeated whole test circuits are not.

Pilot changes to query balance, supervision, noise, teacher family, topology, or the primary metric require a written amendment before confirmation. After test inspection, changes create a new exploratory result and require fresh worlds for confirmation.

### 5.7 Primary metric and claim

For world `i`, method `m`, and query `j`, define

\[
e_{imj}=\mathbf1[\widehat D_{imj}\ne D^*_{ij}].
\]

Average first over queries within a circuit/stratum, then over circuits, then equally over the three query strata and two held-out depths. Finally average equally over independent worlds, with equal aggregate weight for the two teacher strata. Call this **OOD exact-domain error**, in fraction or percentage points; lower is better.

The **primary comparison** is `frc4` against each competent mandatory learner family (best validation-selected `lut4` variant, best validation-selected `logic16` variant, `mlp_cell`, and `synth`) in Track A at the frozen label/optimization setting. A broad superiority claim requires simultaneous positive evidence against all of them, not merely the weakest. A narrower pairwise improvement is labeled accordingly. A model with an unfinished solver cannot silently be dropped; the unresolved result blocks that complete comparison or is treated using worst/best-case bounds.

Choose the primary label setting from the independent pilot’s information/optimization transition, not from a protected positive effect: it should expose a nontrivial learning question without requiring unavailable local resources. Record the exact selection and rationale in `confirm.yaml`. If the strongest methods already have zero error across the informative pilot regimes, do not manufacture headroom by withholding useful information from them. Complete the diagnostic study and report no accuracy advantage in the investigated setting.

Also report the entire nested label-learning curve, including total validation labels and tuning cost. A small effect at one point does not establish sample efficiency over the whole curve.

### 5.8 Secondary metrics and guardrails

| Quantity | Definition/units | Interpretation |
|---|---|---|
| Absolute and relative error effect | `error_baseline − error_frc`, percentage points; relative reduction only when denominator >0 | Primary effect with uncertainty; do not divide by zero or emphasize unstable relative percentages. |
| False elimination | Fraction of truly possible query values omitted | Most consequential native soundness failure; distinguish wrong learned rules from propagation bugs. |
| Retained impossibility | Fraction of impossible query values retained | Measures incompleteness rather than unsound elimination. |
| Query-domain confusion matrix | Four-by-four counts, plus per-class error | Prevents always-unknown or always-singleton behavior from hiding in averages. |
| Conflict detection | Precision/recall of proven inconsistency on oracle-labeled cases | Native “not refuted” is not a proof of consistency. |
| Identifiable-bit recovery | Accuracy on truth bits fixed across all optimal training-consistent models | Avoids penalizing unrecoverable teacher details; give solver proof status. |
| Rule reuse | Error versus depth, fanout, independent-input width, and renaming | Tests composition; do not attribute trivial constant-output generalization to reasoning. |
| Training cost | Wall seconds, CPU seconds where available, optimizer updates, label bits, peak memory | Includes failed tuning, coordinate refinement, and solver compilation/encoding. |
| Inference cost | Median and tail latency in ms/query or μs/query, queries/s, peak bytes | Warm and cold; native and exact semantics separately. |
| Native work | Factor revisions, sweeps, removed bits, tuple tests | Mechanistic explanation, not a substitute for wall time. |
| Model/storage | Trainable scalar count, hard-table bytes, topology/index bytes, checkpoint bytes | Tiny parameter count does not mean tiny training memory. |
| Numerical behavior | Zero/nonfinite gradients, CPU–MPS deviations, extracted-table disagreements | Blocks an accelerator claim when unresolved. |

Report worst relevant strata and the distribution of world-level effects, not only the pooled mean. An improvement accompanied by more false elimination, much higher training cost, or a large reverse-query regression is a tradeoff, not an unqualified advance. Soft support scores do not receive calibration claims unless a separate probabilistic interpretation and appropriate scoring protocol are introduced.

### 5.9 Pilot-calibrated replication and statistical analysis

The independent unit is the **world/library plus its independently generated datasets**. Seeds and repeated masks are nested within a world. Repeated timing calls estimate measurement noise; they are not additional task samples.

Use the following procedure, implemented in `analysis/design.py`:

1. Begin with a small pilot block containing both teacher strata and at least two optimization seeds per world. An initial eight-world block is a variance-estimation starting point, not a final sample-size quota.
2. Estimate between-world, seed-within-world, circuit, and query variance using paired method differences. Add pilot worlds when uncertainty in the between-world variance makes the projected confirmation size unreliable. Add inner queries/seeds instead only when their variance reduction per measured second is larger.
3. Store the empirical distribution of paired world means and use it to simulate fixed-sample confirmation power and interval width. Use a conservative variance estimate, including its pilot uncertainty. Never count all query rows as independent.
4. Select a fixed number of fresh worlds before opening confirmatory outcomes. For a nonzero pilot effect, use the observed effect magnitude as the **planning alternative**, not as a minimum success criterion. The initial analytic estimate is `n ≈ ((z_(1−α/(2M)) + z_0.9) s/δ_plan)^2`, then refine by paired-world simulation and stratification. Here `M` is the frozen number of primary comparisons. Record projected time and memory from actual local measurements.
5. If the pilot effect is near zero or its sign/magnitude is unstable, do not invent a favorable target. Produce a precision-versus-cost curve and either select a justified estimation study or declare that superiority confirmation is not yet warranted. Lack of power is not evidence of equivalence.
6. The primary uncertainty procedure is a paired, teacher-stratified **world-cluster bootstrap with simultaneous max-t intervals** across the frozen primary comparisons. Resample complete world records, not individual masks. Use a fixed bootstrap seed and enough resamples that Monte Carlo uncertainty of interval endpoints is negligible relative to the reported effect; determine this by doubling resamples and checking endpoint stability. This is an approximate inferential method, not a finite-sample coverage theorem.
7. Before confirmation, use pilot-based null simulations, skewed/rare-error stress cases, and duplicate-method controls to check interval behavior. If few worlds or extreme discreteness make the planned approximation unreliable, increase worlds or use conservative bounded-outcome intervals and accept the loss of power. Do not quietly retain an unreliable significance test.
8. Execute the frozen number of worlds without significance-driven optional stopping. Operational failure can pause or terminate the study; an incomplete batch is marked incomplete and is not selectively reported as confirmed. A fresh run after an amendment uses new independent worlds.

Use familywise confidence level 95% for the primary comparison family. Other effects are descriptive with intervals, or enter separately preregistered families. Multiple seeds, depths, label budgets, and ablations do not each become an undisclosed chance to claim a discovery.

There is **no arbitrary percentage-improvement requirement**. A confirmed nonzero effect can be small. Its practical value is assessed separately through the quality–cost frontier and the affected query count. With no downstream application specified, do not invent a utility threshold or call a statistically detectable tiny effect a breakthrough.

### 5.10 Tuning and optimization fairness

All methods receive the same data and validation labels. Match parameterization where attribution requires it; compare realistic cost frontiers where matching updates would conceal actual overhead.

Initial neural tuning must include learning rates `0.001`, `0.01`, and `0.1`, with decade expansion when the best setting lies at a boundary or all settings underfit. These are a starting log-scale probe, not a final search quota. Include loss choice, initialization, annealing/no annealing, and required hard-forward variants. Deep DLGN trials must include the documented gradient-amplification option as well as the ordinary gradient, with the altered gradient explicitly recorded. Do not automatically transfer a helpful implementation trick only to the proposed method. [R6–R8]

Use common pilot optimization checkpoints on a logarithmic update/time schedule. Extend still-improving trials. An unchanged hard dictionary and unchanged validation mistakes across a doubled training interval is an operational plateau, not proof of optimum; diagnose loss, gradients, restarts, and the exact synthesis lower bound before treating it as a competent baseline. Freeze final step counts/schedules and restart counts from these pilots.

Report both (a) converged/best competent quality and (b) checkpoints at matched elapsed training costs. Hyperparameter-search costs are separate from single selected-model fitting costs, and total project cost includes both. A baseline that is intrinsically slower may legitimately lose a cost comparison, but must not be called less accurate simply because it was stopped before improvement ceased.

### 5.11 Timing protocol and overhead

Use CPU execution for exact symbolic solving and hard integer support propagation. Compare float training on both CPU and MPS during calibration; select the fastest faithful path, not the nominally more powerful device.

For timing, exclude graph generation from hot kernel timing but include it in the appropriate end-to-end construction figure. Warm up until compilation/allocation transients settle and block medians stop drifting beyond measured repeat noise. For MPS, synchronize immediately before and after the measured region. Synchronize **outside** throughput batches, not after every tiny cell. CPU timers use a monotonic high-resolution clock.

Measure cold process/import/model-load/encode/query cost separately from warm repeated queries. Report batch-one latency and the throughput-maximizing batch that fits the same safe memory policy. Rotate method order in paired blocks; record AC power, low-power mode, background load, thread count, memory pressure, and thermal state when available. Retain outliers with reasons rather than deleting slow blocks automatically.

If timing uncertainty is material, increase independent timing blocks or batch duration. Repeating a microsecond operation many times within one block gives one block estimate, not thousands of independent measurements. A native cell timing claim requires a competent compiled or vectorized generic-table baseline; Python dispatch overhead alone cannot establish a computational advantage.

For deployment horizon `N` queries,

\[
C_m(N)=C_{\rm data}+C_{\rm tune}+C_{\rm fit}+C_{\rm harden}+C_{\rm encode}+N C_{\rm query}.
\]

Show break-even query counts where costs differ and quality is comparable. If a stage occupying fraction `p` of baseline time is accelerated by `s`, the ideal total speedup is at most `1/[(1−p)+p/s]` before new overhead. Thus a tiny support-kernel gain need not survive exact completion, data preparation, or training.

### 5.12 Optional real-topology transfer, not a disguised application claim

The EPFL Combinational Benchmark Suite supplies public circuit netlists and a stable archived record, DOI `10.5281/zenodo.2572934`. The inspected archive is approximately 2.0 GB; downloading the complete archive is unnecessary for the core. [R27]

This is a **gated extension**, not a core dependency. First verify the license text of the exact selected files and their relation to the archived version. The parsed archive page did not expose a usable license field during this research; therefore no redistribution permission is assumed here. Select small combinational netlists from the official repository after inspecting actual paths and sizes. Do not invent filenames or silently call a live repository snapshot the archived release.

Prefer a small, explicitly supported BLIF subset containing combinational `.names` tables of at most two inputs. Reject sequential elements, unsupported directives, and wider LUTs unless a semantics-preserving converter is independently checked and its cost disclosed. Validate parsed tables against hand-built fixtures and exact simulation.

Test name-redacted real **topology** with known or deliberately hidden local functions. If functions are changed, call the workload synthetic functions on real circuit topology. If functions are merely hidden, acknowledge that the netlist source originally contained them and enforce a clean split. Split entire source circuits before extracting cones; cones from one original circuit cannot cross train/test boundaries.

This experiment can support structural transfer. Simulated observations on published netlists are **semi-synthetic**, not evidence of physical hardware repair, real sensor uncertainty, or a new deployed circuit-design workflow.

---

## 6. M3 Max feasibility and hardware-adaptive configuration

### 6.1 What is known and what remains to be measured

The core uses tiny tables, integer indices, gather operations, reductions, automatic differentiation, and CPU SAT/MaxSAT. It needs neither CUDA nor a large pretrained model. This makes a local PoC plausible, but **no execution on the target M3 Max has been verified**.

Official Apple/PyTorch documentation describes the MPS backend, while current pages can show different release-version labels. The inspected Apple page states Apple Silicon, a sufficiently recent macOS, supported Python, and command-line developer tools as prerequisites. Treat the installed wheel and its documented requirements as authoritative for the selected environment; do not hard-code a purported “latest version” from a page heading. [R28–R30]

The official `difflogic` main module imports `difflogic_cuda` at top level, despite offering a Python execution path. A naïve `pip install difflogic` is therefore not the Mac plan. Vendor or independently adapt the permitted pure tensor mathematics in a CUDA-free module, record the source revision, preserve notices, and test all truth tables and gradients. [R6]

### 6.2 Initial discovery commands

These are **existing macOS/system commands**, unlike the proposed project commands in §7:

```bash
uname -m
sw_vers
sysctl -n hw.memsize
sysctl -n hw.logicalcpu
system_profiler SPHardwareDataType SPDisplaysDataType
xcode-select -p
clang --version
python3 --version
df -h .
vm_stat
sysctl vm.swapusage
pmset -g
```

Record outputs with timestamps, except redact serial numbers, hardware UUIDs, usernames, and unrelated personal paths from sharable logs. Verify `arm64`, the actual M3 Max, actual bytes of unified memory, free disk on the project volume, and whether Python runs natively rather than through translation. Do not assume a maximum-memory configuration. Do not automatically change power settings or install an operating-system upgrade.

Build `hardware.json` containing physical memory bytes, initial available-memory estimate, swap use, free disk bytes, OS/build, architecture, CPU count, Python executable/version/architecture, compiler version, selected torch version, MPS built/available flags, and timestamp.

### 6.3 Dependency strategy

Core dependencies: a stable native-arm64 Python, NumPy, PyTorch, `python-sat`, a YAML parser, pytest, and analysis packages such as SciPy/pandas. Select a mutually compatible Python minor from actual supported wheel metadata; IBM LNN’s inspected current checkout requires Python 3.11 or newer, so isolate it if its lock conflicts with the core. Do not install a similarly named unrelated `pysat` package. [R14,R25]

Use a virtual environment and a lock/hashed requirements file generated after a successful smoke test. Store the interpreter, compiler, package versions, solver backend, source commit IDs, and wheels/source hashes. A stable release is preferred over a nightly; moving to a nightly is a recorded compatibility amendment and does not justify mixing numerical implementations across methods.

PySAT’s documented source-build path requires a POSIX environment, compiler, and build tools; its Mac instructions discuss using the system clang rather than an incompatible GCC setup. Start with the basic `python-sat` package, without unnecessary optional extras. Enumerate actually available solver backends in the installed package and run small SAT/UNSAT/MaxSAT fixtures before selecting one. [R14,R15]

Reuse no pretrained weights. Software dependencies are reusable assets; fitting, hyperparameter searches, solver encodings, and labels generated specifically for this project are project costs. Retain third-party license notices and list licenses from the exact installed revisions, including bundled solver licenses; the outer Python package’s license does not automatically describe every bundled solver.

### 6.4 Critical-operation compatibility gate

Before training or large data generation, test the actual chosen stack on CPU float64, CPU float32, and MPS float32 where available:

- sigmoid and optional softmax/STE paths;
- indexed gathers with repeated indices;
- min/max reductions with and without ties;
- port-to-wire aggregation and its backward pass;
- loss and gradient behavior at near-binary values;
- forward/backward through several unrolled steps;
- CPU hardening producing identical four-bit tables;
- save/load and deterministic CPU-generated random draws.

Use dense padded incidence tensors and masked `min` as the first faithful implementation, avoiding a dependency on a particular sparse or scatter-reduction MPS kernel. Padding must be the correct neutral element: one for minima, zero for maxima, with invalid tuples masked to zero. Never turn an unsupported operation into a different mathematical relaxation without an amendment.

Reference gradients use CPU float64. MPS compatibility is not inferred from importing torch. Disable silent CPU fallback while testing acceleration; inspect failures explicitly. If a fallback mode is later used, label it mixed CPU/MPS and include transfers and synchronization in its timing. Do not present it as pure GPU performance.

Use empirical CPU–MPS tolerances scaled by float32 precision, operation count, and reference magnitude; initial test tolerances such as `atol=1e−6, rtol=1e−4` are diagnostic starting values, not permission to ignore a changed hard table. Away from ties compare finite-difference gradients; at ties compare declared subgradient behavior and resulting updates separately. A discrepancy that changes conclusions forces a CPU-only primary run until resolved.

### 6.5 Memory model

Unified memory is shared by CPU, GPU, operating system, and other applications. Do not add GPU-allocated bytes and process resident bytes as though they were always disjoint physical allocations. Use them as diagnostic views and record system pressure/swap as the operational guard.

Let `B` be batch size, `T` unroll steps, `G` factors, `V` wires, `K` types, and `Pθ` trainable parameters.

**Static core:** three port indices per factor, type indices, incidence maps, `4K` logits, and hard domains. A compact hard representation is on the order of `12G + 4G + V + K` bytes before alignment/incidence bookkeeping, using int32 indices and one byte per domain/table code. Implementations may use int64 indices for tensor compatibility; count actual bytes.

**Training parameters:** float32 parameters, gradients, and two Adam moments alone require approximately `16Pθ` bytes, before framework overhead. Here activations, not parameters, dominate.

**Soft activations:** states require `8BV` bytes per retained state. Tuple support intermediates scale as `O(BTG)` because arity and domain size are fixed. For planning, use

\[
M_{\rm soft}\approx c\,BTG+16BTV+16P_\theta+M_{\rm graph}+M_{\rm batch}+M_{\rm runtime},
\]

with an initial **unmeasured** coefficient `c=256 bytes` per example/factor/step, covering several tuple-sized float tensors and masks. Replace this coefficient with measured peak scaling over `(B,G,T)`; autograd and allocator behavior can make it substantially larger.

For illustration only, `B=32,G=128,T=32` gives about **32 MiB** for the `cBTG` term; `V=136` adds about **2.1 MiB** for the displayed state term. These are not peak-process estimates or M3 measurements. A recurrent generic baseline with hidden width 64 can use hundreds of MiB or more because it stores hidden messages and GRU activations at every step.

**Synthesis:** for `M` training assignments of similar size, approximately `4K + M(P+G)` Boolean variables and `8MG + M(P+W)` base clauses before simplification. With `M=512,P=8,G=32,K=8,W=8`, this is 20,512 variables and about 139,264 clauses. Learned clauses, watchers, Python objects, and MaxSAT bookkeeping may dominate the raw literal storage; profile them rather than assuming a fixed multiplier.

**Teacher enumeration:** storing all assignments costs approximately `2^P(P+G)` bytes with byte-valued wires, or less with careful packing. This grows exponentially in independent inputs. Choose enumeration versus SAT from measured latency and memory, not a universal input-count cutoff.

### 6.6 Safe-memory and configuration rules

Derive an initial project budget from actual physical memory and idle system use. Reserve the larger of (a) observed nonproject use plus a measured fluctuation margin and (b) 25% of physical memory; initially limit the project to the remaining memory and avoid persistent swap. The 25% is a conservative operational reserve, not a requirement that every laptop have spare capacity; lower the project allocation when pressure measurements warrant it. Record the policy.

Probe `(B,G,T)` beginning with batch one, then double batch size while measuring peak memory and step time. Select the largest throughput-efficient batch that stays below the safe budget with a measured margin. Stop increasing batch when throughput plateaus; larger is not inherently better.

On low available memory, in this order:

1. Run one training/solver process at a time and stream datasets.
2. Reduce microbatch size with gradient accumulation that preserves the effective loss and optimizer-step schedule.
3. Use exact activation checkpointing/recomputation; account for extra time.
4. Use CPU execution if accelerator allocations or unsupported operations cause the problem.
5. Reduce **exploratory** graph size to debug, but do not silently reduce a frozen confirmatory width/depth or unroll horizon. If the scientific condition cannot run faithfully, mark that condition locally infeasible and narrow the claim.

Lower precision, quantization, detached iterations, truncated backpropagation, or replacing min/max with smooth approximations can alter the mechanism. None is a default memory workaround. A proposed replacement is a separate method with separate validation.

### 6.7 Stage resource estimates and calibration

| Stage | Expected resource character, not a measured duration | Required calibration/output |
|---|---|---|
| Environment and licenses | Package downloads and isolated environments, plausibly several GB including caches | Measure exact download/environment footprint and free-space reserve before optional installs. |
| Truth-table and tree checks | Tiny CPU workloads; no accelerator needed | Actual runtime and peak RSS; exhaustive case counts and checksums. |
| Oracle/synthesis validation | Small `K≤4` enumeration, small circuits, CNF build/solve | Cost versus candidate count, labels, and graph size; solver agreement. |
| Tensor smoke tests | Small batches and short unrolls; CPU/MPS comparisons | Supported operations, tolerance report, measured scaling coefficient. |
| Neural pilot | Small parameter sets but repeated unroll activations and tuning | Seconds/update and fit, peak memory, optimization/variance curves. |
| Confirmatory batch | Independent worlds can run serially; total time follows calibrated replication | Forecast total time from run manifest, plus uncertainty and retry storage. No unsupported “overnight” promise. |
| Robustness/transfer | Larger input space, deeper graphs, exact-label generation can dominate | Gate each expansion on oracle and memory pilot. Optional netlists only after license/access verification. |
| Reporting/reproduction | Modest tabular data unless all intermediate tensors are retained | Retain essential raw predictions, not every training activation; project/storage manifest. |

Use `total_seconds = Σ(measured_seconds_per_run × planned_runs)` with separate label-generation, tuning, fitting, query-solving, and analysis terms. Measure variability across at least different run blocks rather than reporting a single optimistic extrapolation. Long runs are acceptable when this forecast and expected information gain justify them. Resource exhaustion is not a negative scientific result.

### 6.8 Resumability and recovery

Checkpoints contain model parameters, optimizer moments, scheduler/annealing state, exact training step, data-prefix IDs, sampler state, Python/NumPy/CPU-torch RNG states, validation-selection history, configuration hash, and source/environment hashes. Generate stochastic choices on the CPU so GPU RNG support is not a hidden resume dependency.

Write to a temporary path, flush, then atomically rename; include a content hash and parent-checkpoint ID. Test resumed execution against an uninterrupted run before long jobs. Exact bitwise equality may not be available for all accelerator kernels; the minimum accepted equivalence is declared numerical agreement and identical selected hard tables/metrics in the resume fixture. A material change requires CPU execution or an explicitly stochastic reproducibility claim.

On out-of-memory failure, retain the failed record. Restart from a valid checkpoint with a smaller microbatch preserving effective optimization semantics. A changed effective batch, unroll length, loss, or data order requires an amendment rather than silent continuation. For a solver interrupted mid-search, preserve the input CNF/WCNF, incumbent if available, and status; rerunning the same problem is valid, but do not claim the solver’s internal search state was checkpointed unless it actually was.


---

## 7. Repository, interfaces, and ordered work packages

### 7.1 Minimal repository layout

```text
finite-relation-poc/
  README.md
  research_poc_plan.md
  pyproject.toml
  requirements.lock                 # generated from the tested environment
  LICENSE                           # project owner selects terms for new code
  THIRD_PARTY_NOTICES.md
  configs/
    smoke.yaml
    pilot.yaml
    confirm.yaml                    # created and frozen after pilot design
    robustness.yaml
  src/frc_poc/
    __main__.py                     # proposed CLI below
    schema.py                       # validated dataclasses and serialization
    hardware.py
    data/generate.py
    data/oracle.py
    data/cnf.py
    data/import_blif.py              # only for the licensed optional extension
    cells/hard_reference.py
    cells/hard_fast.py
    cells/soft.py
    cells/lut.py
    cells/logic.py
    cells/mlp.py
    baselines/synthesis.py
    baselines/lnn_checks.py          # optional isolated environment adapter
    train.py
    harden.py
    evaluate.py
    timing.py
    checkpoint.py
    analysis/design.py
    analysis/statistics.py
    analysis/report.py
  native/                           # only after Python semantics pass
    table_propagation.c
    build.py
  tests/
    test_truth_tables.py
    test_aliases.py
    test_tree_exactness.py
    test_cycle_limits.py
    test_gradients.py
    test_teacher_solver.py
    test_discretization.py
    test_no_leakage.py
    test_resume.py
    test_statistics.py
  provenance/
    source_inventory.json
    licenses/
    source_revisions.json
  decisions/
    progress.jsonl
    amendments/
    audit.md
    freeze.json
  data/
    explore/
    confirm/train/
    confirm/val/
    confirm/test/
    replicate/
  runs/<run_id>/
    config.resolved.yaml
    environment.json
    manifest.json
    metrics.jsonl
    predictions.parquet             # CSV/JSONL allowed if fully typed
    timings.jsonl
    failures.jsonl
    checkpoints/
  reports/
    pilot.md
    confirm.md
    final.md
    tables/
```

Do not create services, a database server, or a general event engine. Use ordinary immutable files. The package name `frc_poc` is a **proposed project module**, not an existing downloadable package.

### 7.2 Shared interfaces and schemas

All arrays have explicit dtypes and dimensions. Serialization must not require arbitrary object deserialization for data files; use JSON metadata and numeric arrays, with `allow_pickle=False` for NPZ.

```python
@dataclass(frozen=True)
class Circuit:
    circuit_id: str
    n_inputs: int
    n_wires: int
    ports: NDArray[np.int64]       # [G, 3], ordered input,input,output
    type_id: NDArray[np.int64]     # [G]
    layer: NDArray[np.int64]       # [G]
    final_wires: NDArray[np.int64] # [W]

@dataclass(frozen=True)
class TrainingRecord:
    world_id: str
    circuit_id: str
    assignment_id: str
    inputs: NDArray[np.uint8]     # [P], values 0/1
    targets: NDArray[np.uint8]    # [W], observed labels only

@dataclass(frozen=True)
class Query:
    query_id: str
    circuit_id: str
    evidence: NDArray[np.uint8]   # [V], domain codes 0..3
    query_wire: int
    stratum: str                 # forward/reverse/mixed/diagnostic

class TableLearner(Protocol):
    def fit(self, train, validation, config) -> FitReport: ...
    def export_tables(self) -> NDArray[np.uint8]: ...  # [K], canonical codes
    def save(self, path) -> None: ...

@dataclass(frozen=True)
class QueryResult:
    query_id: str
    domain: int | None            # None means unresolved execution, not unknown
    status: str                   # exact/local_not_refuted/proved_inconsistent/unresolved
    elapsed_ns: int
    factor_revisions: int | None
    solver_calls: int | None
```

The `teacher_tables` field belongs only in oracle/provenance records, never in `TrainingRecord`, model configuration, input features, or a `TableLearner` constructor. Distinguish `domain=3` (a meaningful two-value answer) from `domain=None` (a computation did not finish).

The soft-cell interface is `forward(scores[B,V,2], circuit_batch, steps:int) -> scores[B,V,2]`; gradients flow only through trainable tables and intermediate scores. Padding uses separate valid-wire/factor masks. `−1` padding indices must never accidentally gather the final real wire. Ragged circuits may initially be grouped by shape to avoid this hazard.

Every result row contains `world_id`, `method_id`, `seed_id`, `circuit_id`, `query_id`, teacher stratum, graph depth/width, data-prefix ID, inference track, predicted/true domain, and execution status. Timing records state `boundary` (cold end-to-end, encode, fit, warm query, native kernel), device, threads, batch size, warmup policy, and synchronization policy.

### 7.3 Configuration convention

The following is a **schema illustration with exploratory defaults**, not a frozen confirmation protocol:

```yaml
phase: explore
worlds:
  seed_manifest: data/explore/world_seeds.json
  type_count: 8
  teacher_strata: [all16, depends_both]
graph:
  family: layered_dag
  width: 8
  train_depths: [2, 4]
  test_depth_multipliers: [2, 4]
data:
  initial_topologies_per_world: 8
  assignments_per_topology: 16
  query_strata: [forward, reverse, mixed]
  mixed_reveal_probability: 0.5
  output_label_noise: 0.0
method:
  id: frc4
  parameterization: sigmoid_table4
  relaxation: max_min_support
  optimizer: adam
  loss: membership_mse
  learning_rate: 0.01
  unroll_rule: twice_depth_plus_one
  effective_batch_size: null       # doctor/calibration must resolve
  microbatch_size: null
  steps: null                      # pilot optimization gate must resolve
inference:
  primary: common_exact
  native: full_hard_closure
  harden: validation_select_raw_or_coordinate_refined
runtime:
  device: auto_verified
  dtype: float32
  silent_fallback: false
analysis:
  primary_metric: world_macro_ood_exact_domain_error
  independent_unit: world
  confidence_level: 0.95
  confirm_world_count: null        # must be fixed before confirmation
```

`twice_depth_plus_one` is an initial finite-unroll hypothesis, not a soft-convergence theorem. Compare it with shorter and longer unrolls during the pilot. Oracle-hard tests must show that the chosen computation horizon can express the relevant dependency paths. Final confirmation disallows unresolved `null` fields, moving source revisions, unrecorded seeds, and automatic per-test tuning.

### 7.4 Proposed CLI

The agent must implement these commands; they do not exist until the repository is built:

```bash
python -m frc_poc doctor --out provenance/hardware.json
python -m frc_poc check --suite semantics --config configs/smoke.yaml
python -m frc_poc check --suite numerics --config configs/smoke.yaml
python -m frc_poc generate --config configs/pilot.yaml
python -m frc_poc baseline-check --config configs/smoke.yaml
python -m frc_poc pilot --config configs/pilot.yaml
python -m frc_poc design --pilot reports/pilot.md --out configs/confirm.yaml
python -m frc_poc freeze --config configs/confirm.yaml
python -m frc_poc run --config configs/confirm.yaml --resume
python -m frc_poc robustness --config configs/robustness.yaml
python -m frc_poc report --manifest decisions/freeze.json
```

`design` must read machine-readable pilot artifacts referenced by the report, not scrape prose for decisions. `freeze` validates all blocking gates and writes hashes; it does not confer statistical validity by itself. `run --resume` must refuse an incompatible checkpoint rather than silently changing the method.

### 7.5 Work-package sequence and gates

Each package below has a functional scientific purpose. Optional packages are reached only by their stated evidence trigger.

#### WP00 — Establish inputs, rights, and actual local capacity

**Risk addressed:** unavailable hardware/dependencies, hidden CUDA assumptions, unsupported asset rights.  
**Entry/dependencies:** this plan and a fresh target laptop; no experiments need have run.

**Actions:** create the repository and provenance files; copy the supplied-source inventory and hashes without redistributing source text unnecessarily; run the discovery commands; choose a compatible native Python environment; resolve dependencies and licenses; smoke-test basic SAT/UNSAT and RC2; inspect current source revisions for the adapted logic kernels; store the complete environment lock.

**Files/command:** `hardware.py`, `schema.py`, `provenance/*`, `requirements.lock`; proposed `doctor`.

**Outputs/tests:** sanitized `hardware.json`; `compatibility.json` with `operation`, `device`, `dtype`, `forward_pass`, `backward_pass`, `max_abs_error`, `status`; solver identity and known-fixture answers; exact free-space forecast.

**Resources:** environment/download footprint measured before optional packages; no model/data download beyond permitted software.

**Acceptance:** a faithful CPU path and exact teacher/synthesis path exist; available disk and memory support the small semantic tests. MPS is not required for this gate.

**Outcomes:** CPU works/MPS fails → continue CPU-first; source build fails → use documented compatible build/interpreter, record attempts; no exact solver yet → enumeration permits WP01 only, not larger competitive claims; no viable local environment → report a blocking resource/dependency limitation, not a failure of the mechanism. Optional EPFL license uncertainty blocks only that extension.

#### WP01 — Validate indispensable semantics and theoretical limits

**Hypothesis:** H0.  
**Entry:** basic Python/NumPy for hard semantics; the final numerical check in Appendix A additionally requires CPU PyTorch. Solver availability is helpful but unnecessary for that appendix.

**Actions:** implement hard reference cells and independent exhaustive enumeration; reproduce all 1,024 single-cell cases, 900 seeded factor-tree cases, alias contradiction, disconnected inconsistency aggregation, and the cyclic XOR failure. Add all 16 gate-index conversions, evidence intersection, empty/no-factor graphs, and arbitrary wire-renaming tests.

**Files/command:** `cells/hard_reference.py`, `data/oracle.py`, semantic tests; `check --suite semantics`.

**Outputs:** exact case counts, failures with minimal witness graphs, canonical table mapping, result JSON and source hashes.

**Resources:** small CPU work; measure actual runtime but do not optimize it yet.

**Acceptance:** zero soundness/indexing discrepancies; native incompleteness on the known counterexample is correctly reported rather than “fixed” by consulting the oracle invisibly.

**Outcomes:** soundness failure → debug before all learning; unexpected global completeness claim → revise claim, preserve counterexample; correct known behavior → proceed; equality with classical constraints is expected and narrows novelty rather than invalidating correctness.

#### WP02 — Validate labels, task sensitivity, and identifiability

**Risks:** a trivial query population, incorrect teacher, leakage, or unlearnable hidden rules.  
**Entry:** WP01 passes; a tested exact solver or small exhaustive path exists.

**Actions:** implement layered graph and evidence generators; cross-check enumeration against CNF for small instances; create separate truth-table/library storage; compute domain class frequencies, output entropy diagnostics, active-input dependence, and type occurrence coverage. Implement complete dictionary enumeration at `K=4`, symbolic synthesis, and bit-identifiability checks. Insert known rule errors and verify that the evaluator detects them.

**Files/command:** `data/generate.py`, `data/cnf.py`, `baselines/synthesis.py`, `test_teacher_solver.py`, `test_no_leakage.py`; `generate`, `baseline-check`.

**Outputs:** `dataset_manifest.json` with counts/hashes; `sensitivity.json` with exact-domain frequencies and fixed-predictor error; `identifiability.parquet` containing bit IDs, possible values at optimum, and solver proof status.

**Resources:** estimate enumeration as `16^K × examples × circuit work`; profile before scaling. CNF storage/solve time measured independently.

**Acceptance:** two labeling paths agree; no target/teacher data enters learners; the workload has enough nontrivial forced and underdetermined queries to measure the proposed behavior at a feasible precision. All-unknown dominance is documented and not cured by post-hoc selective reporting.

**Outcomes:** label disagreement → broken harness; no signal because all outputs constant → retain that raw stratum, use the already declared essential-function stratum to assess sensitivity; parameters non-identifiable → expand examples/topology diversity or narrow to predictive claims; exact synthesis already saturates all informative settings → document absent quality headroom and decide whether a cost-only investigation is still justified.

#### WP03 — Reproduce competent local baselines before the candidate comparison

**Risk:** apparent novelty produced by weak or incompatible baselines.  
**Entry:** WP02 valid datasets/oracles.

**Actions:** adapt the inspected DLGN pure tensor mathematics without CUDA imports; test all gate values and gradients; implement LUT4 and the numerical MLP cell; implement soft/STE/Gumbel variants, consistent shared-type sampling, initialization/learning-rate options, and table export. Validate symbolic optimum against enumeration at small `K`. Run IBM LNN known-formula semantics in an isolated compatible environment when available.

**Files/command:** `cells/lut.py`, `cells/logic.py`, `cells/mlp.py`, `baselines/*`, `harden.py`; `baseline-check`.

**Outputs:** baseline reproduction card per method: source revision, adaptation list, license, truth-table agreement, gradient agreement, shallow-fit result, optimum status, and device.

**Resources:** shallow single-/few-gate tests first; only then pilot-width circuits. Published CUDA image training is not required or claimed reproduced.

**Acceptance:** each competitive baseline can fit a fully observed truth table and representative shallow composition within its function class. A documented full-framework baseline that cannot run remains contextual, not a fabricated executed comparator.

**Outcomes:** XOR failure in single threshold → expected negative control; XOR failure in sufficient MLP/LUT/DLGN → implementation/optimization diagnosis; CUDA import failure → repair the explicitly allowed adaptation, not replace it with a weaker model; incomplete synthesis → retain incumbent/status and adjust local test size for validation, without misreporting optimality.

#### WP04 — Implement minimum complete differentiable cells

**Hypothesis/risk:** the support-based learner is numerically meaningful and sufficiently simple.  
**Entry:** WP01 and WP03 pass.

**Actions:** implement `frc4`, `frc16`, and forward-only support controls from the equations; use explicit tuple ordering, alias masks, synchronous snapshots, neutral padding, and fixed unroll lengths. Implement finite-difference gradient tests away from ties, tie/subgradient diagnostics, nonfinite detection, and finite hard-forward losses. Test several connected cells, not only one.

**Files/command:** `cells/soft.py`, `tests/test_gradients.py`; `check --suite numerics`.

**Outputs:** CPU64 reference comparisons, CPU32/MPS32 error and gradient reports, parameter counts, per-type gradient coverage, hardening agreement.

**Resources:** start `B=1` and tiny graphs, then calibrate selected `(B,G,T)` points. Store measured memory scaling.

**Acceptance:** correct hard limit; expected local support equality with the generic implementation; finite gradients where a smooth path exists; no silent semantic change on accelerator fallback.

**Outcomes:** hard-limit mismatch → implementation defect; gradients fail away from ties → implementation defect; genuine tied/zero-gradient bottleneck → mechanistic learning risk, compare alternative established relaxation only as a named ablation; CPU correct/MPS divergent → CPU primary; no learning signal even on informative shallow cases after competent diagnosis → evidence against this relaxation locally, not against every possible constraint neuron.

#### WP05 — Pilot optimization, resources, and the hypothesis

**Hypotheses:** H1/H2 plausibility; planning variance and scale.  
**Entry:** correct harness, candidate, and competent baseline cards.

**Actions:** execute the exploratory label ladder and optimization extensions; train all mandatory families on identical world data; export/harden/refine tables; compute Track A and Track B separately. Record zero-gradient paths, hardening gaps, solver optimality, and each cost component. Compare finite unroll lengths to longer ones on informative circuits. Fit resource and variance models. Test interruption/resume.

**Files/command:** `train.py`, `checkpoint.py`, `analysis/design.py`; `pilot`.

**Outputs:** `reports/pilot.md` plus machine-readable world/seed/circuit metrics; capacity/resource curves; complete tuning ledger; candidate hypothesis amendments; projected confirmation precision and duration.

**Resources:** measured; parallelism is optional and must not compromise memory pressure or timing. Avoid broad searches before gradient/identifiability risks are resolved.

**Acceptance:** enough optimization and replication evidence to distinguish implementation failure, missing signal, and a plausible method effect. A positive effect is **not** required to complete this package.

**Outcomes:** gain vanishes with tuned/refined LUTs or synthesis → record no differentiated benefit; all methods at floor → do not run an underinformative superiority study; gradient horizon too short → revise finite-unroll specification before confirmation and rerun affected pilots; variance dominates → allocate additional independent worlds; mechanism works only on a proxy → narrow the intended claim; local capacity insufficient for the needed phenomenon → record local inconclusiveness and the exact missing scale.

#### WP06 — Freeze the confirmatory design

**Risk:** adaptive selection creates a manufactured win.  
**Entry:** WP05 report with solved numerical/label prerequisites.

**Actions:** choose exact primary setting, data counts, training variants and schedules, baseline families, inference backend, hardening policy, seeds/replication, metric aggregation, uncertainty method, and operational recovery rules. Compute the required fixed sample size from pilot variance/power simulations or choose a justified precision study. Verify every unresolved configuration field is resolved. Store protected seed/hash manifest and methodological amendments.

**Files/command:** `configs/confirm.yaml`, `decisions/freeze.json`, `analysis/design.py`; `design`, `freeze`.

**Outputs:** a machine-verifiable frozen manifest with source/data/environment hashes, planned run IDs, primary comparisons, alpha family, effect-planning assumptions, and resource forecast.

**Resources:** analysis and configuration generation are cheap; a forecast may reveal that a meaningful confirmation needs prolonged serial work.

**Acceptance:** no protected scores have been used; design detects the pilot-relevant effect when feasible; a materially underpowered design is explicitly an estimation/inconclusive study, not a promised confirmation.

**Outcomes:** no credible differentiated hypothesis survives → skip superiority runs and complete the diagnostic report; excessive but finite workload → reduce nuisance variance or choose a narrower justified claim before freezing, not an arbitrary favorable subset; fundamental hypothesis/population change → amendment and new exploratory calibration.

#### WP07 — Execute the controlled comparison

**Hypotheses:** frozen H1/H2 claims.  
**Entry:** WP06 frozen and all blocking gates pass.

**Actions:** run the exact planned worlds and seeds; isolate teacher labels; preserve all outputs/failures; apply the same selected per-family training procedure; evaluate common exact-domain queries; collect cold/hot and training costs according to the declared boundaries. Do not tune after viewing protected results.

**Files/command:** `evaluate.py`, `timing.py`, run manifests; `run --resume`.

**Outputs:** one status row for every planned run, complete prediction/timing records, fixed-sample primary analysis, per-world differences, simultaneous intervals, and unresolved-result bounds.

**Resources:** use WP05 measured forecast; actual totals and any drift are reported. Resume only compatible states.

**Acceptance:** all planned runs accounted for; exact backend finished or unresolved cases explicitly bound the conclusion; no unfair baseline exclusions; no hidden testing amendments.

**Outcomes:** confirmed pairwise/joint improvement → WP08 attribution and fresh confirmation; interval crosses zero → inconclusive at this precision, not “no effect”; credible negative effect → report it; unfinished/resource-limited runs → incomplete comparison; materially changed software/numerics → rerun affected paired block under a common environment and retain the superseded results.

#### WP08 — Attribution, boundaries, and targeted transfer

**Hypotheses:** H1 mechanism attribution and H4 robustness.  
**Entry:** completed controlled comparison, or a negative/inconclusive result needing a specific diagnosis.

**Actions:** execute the prioritized ablations in §8. Start with same tables/solver and same parameterization; then reverse-message removal, hardening, generic optimization sharing, input-width/fanout/renaming, and label-noise tests. Native soundness is tested with oracle tables separately from learned tables. The licensed real-topology extension is reached only after a transferable component signal or a specifically justified structural diagnostic.

**Files/command:** `configs/robustness.yaml`, `analysis/*`, optional importer; `robustness`.

**Outputs:** attribution table linking each observed gain/regression to a discriminating control, failure witnesses, cost/quality Pareto plots, clearly exploratory versus preregistered labels.

**Resources:** prioritize controls that can erase the interpretation before large-width or real-topology runs. Measure oracle cost before expanding independent inputs.

**Acceptance:** a positive interpretation survives the simpler explanations relevant to its scope; otherwise it is revised. Negative controls behave as predicted.

**Outcomes:** effect explained by existing hardening/extra compute → reclassify; only raw synthetic graphs work → narrow scope; expected finite-domain limitation appears → report boundary, not bug; new hypothesis emerges → log amendment and confirm on new worlds rather than relabel the old test.

#### WP09 — Independent rerun and final synthesis

**Risk:** a one-run artifact or an irreproducible handoff.  
**Entry:** interpretation and scope are explicitly stated; positive results have completed attribution.

**Actions:** rebuild the environment from its lock in a fresh local environment; regenerate fixtures; reproduce a frozen selected run and resume test; for a positive scientific claim, execute an independent new-world confirmation with the amended final claim already frozen. Produce a report including unfavorable results, unsuccessful variants, unresolved cases, costs, source/provenance records, and exact claim language.

**Files/command:** `reports/final.md`, exported tables, `decisions/audit.md`; `report`.

**Outputs:** standalone reproducibility instructions, evidence table, outcome classification, residual-risk register, next-step decision with rationale.

**Resources:** a fresh environment requires disk capacity; serial reruns avoid another machine. A local independent rerun is not external independent review.

**Acceptance:** another coding agent can recover the configuration, data, code, and result provenance without chat memory. No unexecuted test is described as passed.

**Outcomes:** repeat confirms → report the precise supported improvement; inconsistent repeat → report instability and investigate, not average selectively; no advantage → publish a useful replication/negative result; missing resource/prerequisite → explicit incomplete scope.

---

## 8. Ablation, robustness, and failure-analysis program

### 8.1 Priority-ordered attribution tests

| Priority/test | Intervention | Required comparison and interpretation |
|---|---|---|
| 1. Solver-normalized evaluation | All exported tables use the same exact query solver | Removes the easiest fake win: giving only the candidate inverse reasoning. Any remaining effect concerns learning/hardening. |
| 2. Identical hard tables | Feed exactly the same tables to both propagators | Must agree at closure. Different learned-table quality cannot explain a kernel discrepancy. |
| 3. Four versus sixteen parameters | `frc4`/`frc16` versus `lut4`/`logic16` | Separates relational execution from parameter-count/optimization differences. |
| 4. Reverse messages removed | Keep the same support parameterization and only compute output-port supports | Tests whether bidirectional execution matters during training rather than merely being an available inference interface. |
| 5. Shared discretization/refinement | Compare raw hardening and coordinate-refined tables for every learner | An improvement erased by generic refinement is not uniquely attributable to the cell. |
| 6. Shared optimizer/init/loss options | Give helpful generally applicable choices to competitors | Prevents a training-trick advantage being mislabeled architectural. |
| 7. Oracle versus learned tables | Test native propagation with true and estimated rules separately | Distinguishes propagation unsoundness from wrong learned constraints. |
| 8. Equal labels / equal compute / best competent quality | Three complementary views, not one forced matching rule | Extra labels, capacity, or training time must remain visible. |
| 9. Teacher-library identifiability | Compare fixed and ambiguous bits; exact small-library optimum | Avoids attributing unavoidable uncertainty to a weak learner or scoring arbitrary hidden parameters. |
| 10. Finite unroll sensitivity | Compare causally adequate horizons, gradient paths, and longer runs | A too-short horizon is an unsuitable experiment; increasing it indefinitely without learning gain is not justified. |

### 8.2 Mechanistic adversaries

**Reconvergence:** test the explicit unsatisfiable XOR cycle, then increase shared ancestors/fanout while holding forward depth and table complexity approximately fixed. Local nonempty domains are allowed; false claims of global satisfiability are not.

**Alias and identity collision:** use `XOR(x,x)`, AND with repeated input, renumber all wires, permute type IDs with table rows consistently, and batch graphs with overlapping local IDs but different world identities. Cross-example or cross-world binding is a severe implementation defect.

**Disconnected contradiction:** add an inconsistent component unrelated to the query. The global exact answer is empty; a native local result must still surface the global inconsistency flag. Do not silently use only the query cone when evidence elsewhere can make the entire model inconsistent.

**Duplicate constraints:** duplicate an identical factor. Hard closure must be unchanged. Max/min support values should be idempotent under exact duplicate constraints; inspect gradient tie behavior separately. A product-message ablation may shrink repeated “evidence” and is explicitly a different semantics, not an innocuous acceleration.

**Retraction:** reveal `x=0`, propagate, then remove or replace that evidence. Reinitialize and recompute; never reuse shrunk domains as if removed values would recover spontaneously. This is also a deliberate boundary separating the project from incremental-change/QCR mechanisms.

**Distractors:** add disconnected satisfiable factors, unused inputs, and constant gates. Account for their scheduling cost. Do not award a sparse execution benefit unless a generic propagator receives the same graph pruning/indexing optimization and the same consistency semantics.

**Unreachable rows:** create gates whose input combinations never occur in the training corpus. Their truth bits may be non-identifiable; no method receives their teacher labels. Test predictions and uncertainty under later compositions that make those rows reachable.

**Constraint error amplification:** flip one learned truth bit and measure the resulting false elimination, depth dependence, and conflict propagation. A compact, transparent model can still be confidently wrong.

**Relaxation pathology:** initialize symmetric tables, make almost-equal min/max candidates, include high-degree variables, and compare CPU/MPS reduction order. Track whether a nominally small numerical difference changes a hard gate or final claim.

**Parameter-sharing violation:** deliberately sample a different gate choice at each occurrence in a diagnostic broken model. The harness should reveal inconsistency relative to the shared-type teacher. The production Gumbel implementation must not do this accidentally.

**Equivalent functions, different internal rules:** construct observationally equivalent dictionaries on training circuits. Evaluate whether claimed “rule recovery” incorrectly uses inaccessible teacher identity rather than identifiable behavior.

### 8.3 Robustness axes

Change one important axis at a time before combining them: depth; number of independent inputs; type count; fanout/reconvergence; tree versus layered-DAG topology; output entropy; available evidence fraction; and distribution of primitive functions.

Add training-label noise only after the noiseless task is understood. Initial exploratory rates of 0.05 and 0.15 expose mild and substantial corruption; they are workload definitions, not improvement targets. Validation/test truth remains clean unless a separate explicitly noisy-observation question is posed. The symbolic baseline changes from hard labels to MaxSAT soft labels as specified, not to an unfair unsatisfiable hard encoding. Noisy point observations must not be interpreted as sound learned constraints.

For query evidence, vary the mixed reveal probability around the primary one-half setting. Track domain-class composition and exact oracle difficulty. A gain at one evidence density need not generalize to others.

### 8.4 Harness validation

Use equivalent-method pairs with identical parameters and predictions to check zero differences. Insert known table mistakes to check sensitivity. Swap method labels in stored records and verify the sign of paired effects reverses. Shuffle labels in a negative learning control and verify that generalization does not remain inexplicably high beyond fixed-predictor performance.

Validate aggregation with a tiny hand-calculated dataset in which one world has many more queries than another: world weighting must not turn into query weighting. Validate teacher-stratum balancing and cluster resampling. Construct missing/timeout rows and verify they are neither silently excluded nor confused with `{0,1}`.

### 8.5 Failure taxonomy used in every report

| Code | Meaning | Appropriate action |
|---|---|---|
| `IMPLEMENTATION_INVALID` | Broken truth semantics, labels, gradients, aliasing, or serialization | Repair; invalidate affected comparisons; rerun necessary paired data. |
| `MEASUREMENT_INVALID` | Timing boundary, synchronization, leakage, or statistical aggregation defect | Repair measurement; retain superseded results as invalid, not negative. |
| `PROXY_INSENSITIVE` | Trivial domains, insufficient horizon, or no identifiable information | Redesign the exploratory proxy with recorded rationale; do not claim mechanism falsified. |
| `RESOURCE_LIMITED` | Faithful required scale or solver completion unavailable locally | Report local infeasibility/inconclusiveness; preserve the exact scientific gap. |
| `UNDERPOWERED` | Valid noisy experiment cannot distinguish effects at achieved precision | Report intervals and detectable effects; no equivalence claim. |
| `MECHANISM_NEGATIVE` | Correct, sensitive tests contradict a locally expressed prediction | Narrow or stop that mechanism; do not invoke unspecified future scale as an excuse. |
| `KNOWN_EQUIVALENT` | Behavior is an established propagator or existing training method | Reclassify as replication/engineering; novelty claim removed. |
| `NO_ADVANTAGE` | Mechanism functions but competent alternatives match or dominate | Report negative comparative result; avoid adding speculative modules to rescue it. |
| `TRADEOFF` | One benefit with a material uncompensated regression | Report Pareto position and scope, not universal superiority. |


---

## 9. Reproducibility, experiment records, and reporting

### 9.1 Immutable inputs and resumable execution

Store the plan itself in the repository and assign it a content hash. Every run references that hash, its configuration hash, the source commit, and the dependency lock. Archive the original configuration even when a corrected configuration supersedes it. Never reuse a run ID for changed code, data, precision, or scientific settings.

Keep generator seeds, teacher-stratum labels, topology hashes, observed-label hashes, and oracle-label hashes separately. Teacher tables belong in the oracle namespace, not in a convenience object passed to a learner. A learner must be executable in a process that cannot read teacher parameters or test labels. Unit tests should exercise that restricted interface.

At a safe optimizer-step boundary, write a checkpoint to a temporary file on the same filesystem, flush it, and atomically rename it. Include model and optimizer states; learning-rate, temperature, and annealing state; global step and completed examples; CPU, NumPy, Python, and available accelerator random states; data-sampler position; selected hardening state; effective-batch policy; configuration/data hashes; and software/hardware fingerprint. Log any interrupted partial step as uncommitted. Test interruption and recovery against an uninterrupted CPU reference before relying on checkpoints.

MPS bitwise replay is not presumed. Test its actual reproducibility locally. When exact replay is unavailable, resume from a committed step with the documented tolerance and retain the interruption record. A version, backend, or numerical-convention change creates a new run branch; rerun affected paired competitors rather than silently combining incomparable segments. Solver objects need not be serialized: save the complete CNF/WCNF, variable map, assumptions, incumbent, proved bound, and solver configuration, then reconstruct. A saved incumbent is not a proof of optimality.

### 9.2 Required record schemas

Use UTF-8 JSON for small records and JSONL or compressed columnar files for repeated observations. No database server is necessary. All byte quantities are integers; summarized durations are monotonic-clock seconds, while raw `elapsed_ns` fields retain integer nanoseconds; wall-clock timestamps are UTC ISO 8601. Represent unavailable quantities as `null` plus a reason, never as zero.

A run manifest must contain at least:

```json
{
  "schema_version": 1,
  "run_id": "content-addressed-or-unique-id",
  "phase": "explore|confirm|replicate",
  "protocol_sha256": "...",
  "config_sha256": "...",
  "source_commit": "...",
  "source_dirty": false,
  "dependency_lock_sha256": "...",
  "method_id": "frc4",
  "world_id": "...",
  "teacher_stratum": "all16|depends_both",
  "optimization_seed": 0,
  "data_manifest_sha256": "...",
  "hardware_manifest_sha256": "...",
  "backend": "cpu|mps",
  "dtype": "float32",
  "effective_batch_records": 0,
  "microbatch_records": 0,
  "status": "created|running|complete|interrupted|failed|invalid",
  "failure_code": null,
  "parent_checkpoint": null
}
```

The zeros and ellipses above describe fields, **not executable experiment defaults**. Validation must reject missing hashes and nonpositive required sizes.

Each query record contains `world_id`, `circuit_id`, `query_id`, query stratum, topology size/depth, evidence hash, true domain code, predicted domain code or `null`, exact-backend completion status, native inconsistency flag where relevant, raw error indicators, domain-class labels, and timings with a named measurement boundary. Each fit record contains train/validation loss, hard forward error, table margin, gradient diagnostics, label count, optimization work, hardening work, solver status, memory measurements, and the checkpoint hash.

Each resource record distinguishes process resident memory, sampled Metal tensor memory, sampled Metal driver allocation, observed system pressure, stored artifact bytes, training seconds, hardening seconds, cold query seconds, and amortized query seconds. Do not add overlapping unified-memory counters into a fictional total. Keep individual synchronized timing blocks, their order, and their batch size; a summary percentile alone is insufficient.

Store a task/decision journal with these fields:

```text
timestamp_utc, work_package, entry_evidence, action,
outputs_and_hashes, observed_outcome, failure_class,
decision, rationale, next_work_package, amendment_id
```

A progress report should make it possible to resume from the last accepted gate without chat memory. Record unsuccessful installation attempts and incompatible versions in the environment log, but do not repeatedly retry a known unsupported path without new evidence.

### 9.3 Statistical implementation details

The independent unit is the **new hidden-library world**, not an optimizer seed, circuit, query mask, epoch, or timing repeat. Average the prescribed within-world quantities first. For each baseline `m`, define the paired effect in world `i` as

\[
d_{im}=\operatorname{error}_{im}-\operatorname{error}_{i,\mathrm{frc4}}\in[-1,1].
\]

Positive values favor the proposed learner. With the two equally weighted teacher strata,

\[
\widehat\Delta_m=\tfrac12(\bar d_{0m}+\bar d_{1m}),\qquad
\widehat{se}_m^2=\tfrac14(s_{0m}^2/n_0+s_{1m}^2/n_1).
\]

Implement the planned simultaneous bootstrap intervals as follows. Resample complete paired world rows **within each teacher stratum**. Recompute each effect and its standard error. For resample `b`, compute the maximum absolute studentized deviation across the entire frozen primary comparison family:

\[
Z_b=\max_m\left|\frac{\widehat\Delta_m^{*(b)}-\widehat\Delta_m}
{\widehat{se}_m^{*(b)}}\right|.
\]

Use its frozen `1−α` quantile `q` to form `Δ̂_m ± q sê_m`. The default family error level is `α=0.05`. Fix the number of resamples and quantile convention after pilot Monte Carlo stability checks, before confirmation; changing numerical precision of a quantile must not become significance-driven optional stopping. This is an approximate cluster-bootstrap procedure, not a finite-sample theorem for a few worlds. Validate coverage empirically on simulated bounded world-effect distributions resembling the pilot, including skew and many exact zeros.

**Degenerate or unstable bootstrap:** do not divide by an arbitrary epsilon and announce certainty. Report the failure and use the following conservative bounded-data fallback, which is also a useful audit interval. For `S=2` strata and `M` frozen baseline comparisons, set

\[
h_s=\sqrt{\frac{2\log(2SM/\alpha)}{n_s}},\qquad
h=\tfrac12(h_0+h_1).
\]

Each stratum effect lies in `[-1,1]`; the bounded-mean exponential tail bound followed by a union bound gives simultaneous intervals `Δ̂_m±h`, clipped to `[-1,1]`. This bound is often wide, appropriately exposing insufficient independent replication. It does not assume seeds or queries are independent. The finite-sample bound assumes independently generated worlds and a fixed learning/evaluation protocol. Report both interval types when used; never select whichever looks favorable. Appendix B supplies a hand-calculable aggregation check.

For incomplete exact-oracle or method results, retain the affected population weights. Give lower and upper error bounds by assigning unresolved errors zero and one, respectively. A superiority statement must survive the corresponding worst-case proposed/best-case baseline assignments as well as statistical uncertainty. Otherwise classify it as incomplete. A solver timeout is not the domain `{0,1}`.

Primary confidence intervals concern the frozen effect family only. Learning curves, per-function breakdowns, ablations, and robustness analyses remain exploratory unless separately prespecified and multiplicity-adjusted. An independent repeat on fresh worlds confirms the same claim; it cannot repair an undisclosed change in the question.

### 9.4 Reporting contract

Generate `reports/final.md` and machine-readable tables from raw records, not hand-entered preferred numbers. The report starts with the exact claim and population, followed by a disposition: confirmed comparative improvement, narrower component improvement, tradeoff, no advantage, negative mechanism result, or inconclusive.

Include absolute errors, absolute paired differences in percentage points, relative changes only when denominators are meaningful, simultaneous intervals, world and query counts, all primary comparators, and domain-class frequencies. Report per-depth, per-teacher-stratum, and ambiguity/contradiction outcomes even when the macro average is positive. Include both raw and task-refined tables, tuning and failed-run costs, measured memory, and both cold and amortized inference boundaries.

Plot quality against labels and total project-specific compute, rather than displaying one favorable checkpoint. Distinguish completed runs from invalidated, interrupted, or resource-limited runs. A Pareto improvement requires that relevant cost/quality regressions are not hidden. With no specified deployment, estimate useful operating ranges and break-even query counts instead of inventing an economic value.

A negative report must contain the strongest valid baseline result, evidence that the harness was sensitive, identifiability/optimization diagnostics, and the particular prediction weakened. An inconclusive report must identify what uncertainty remains and the measured cost of resolving it. Neither should be rewritten as evidence that arbitrary large-scale intelligence is impossible or inevitable.

### 9.5 Agent authority and methodological amendments

The coding agent may choose routine filenames, internal factoring, equivalent vectorization, logging libraries, and non-destructive caching, provided interfaces and tests remain satisfied. It may reduce microbatch size while preserving the effective update and choose the validated CPU path instead of unsupported MPS operations. Such choices are logged.

Changes to teacher distribution, information access, supervision, primary metric, comparison family, tuning privilege, hardening objective, confirmatory sample size/stopping rule, or inferential semantics require an amendment with: reason, evidence available at the time, affected hypotheses, consequences for fairness, and whether fresh confirmation is required. Post-test changes cannot retain the old confirmatory label. Never erase unfavorable runs or retroactively rename a primary endpoint.

---

## 10. Outcome interpretation and evidence-triggered continuation

### 10.1 Decision table

| Observed outcome | Defensible conclusion | Next action |
|---|---|---|
| Exact hard cell matches the generic propagator; no learning improvement | A correct implementation of known machinery | Finish the replication/negative comparative report. Do not add neuron birth or an ecology to manufacture a contribution. |
| `frc4` beats directed learners but not exact synthesis | A possible optimization advantage over those continuous learners, not the strongest matched alternative | Report the narrower effect and synthesis costs. Continue only where the measured quality–cost frontier gives a concrete use case. |
| Gains disappear with modern STE/Gumbel training or shared hardening | The apparent advance was baseline weakness or an existing training technique | Revise the result to no distinct contribution; retain the attribution evidence. |
| Gains survive all locally competent baselines, matched information, and attribution controls | Evidence for the specified small-rule learning contribution | Freeze the causal interpretation, independently replicate on fresh worlds, then test width/topology transfer before expanding architectural scope. |
| All competent methods achieve the same exact answers at an information/optimization plateau | No remaining accuracy headroom in that regime | Compare real costs at equal quality only when sensitive timing measurements justify doing so. Do not select an artificially undertrained competitor. |
| Learned tables are correct but native closure often stalls | Local consistency is incomplete on the chosen topology | Report completeness versus cost. Use the common exact backend for exact answers; no claim that more sweeps will solve arbitrary cycles. |
| Training gets stuck while local numerical tests pass | A learning failure, not necessarily an inference failure | Check the bottleneck-gradient/parameterization controls below. A changed relaxation is a new method, not vindication of the original one. |
| Rule recovery is ambiguous even under exact synthesis | The training observations do not identify the hidden library | Score observable predictions; expose ambiguity. More independent compositions may resolve it; privileged internal labels are a separate experiment. |
| Synthetic gain does not survive licensed real-topology transfer | A synthetic-family benefit only | Narrow the applicability claim and inspect the differing structural conditions. Do not imply deployment value. |
| Precision is inadequate or a fair baseline cannot complete locally | Inconclusive/local resource limitation | Report valid bounds and measured scaling; do not call an omitted comparator a defeated one. |

### 10.2 What justifies more scale

Increase a dimension only when the current experiment identifies a mechanism that the larger setting can express or distinguish. Examples include a rising gap between native and exact domains as reconvergence grows; measurable optimization failure beyond a depth where the forward dependency horizon is still covered; or exact synthesis becoming costly while a learner retains quality. Estimate the next experiment’s runtime and memory from calibration, choose independent-world replication sufficient for its claim, and retain the smaller-scale controls.

A hypothetical eventual advantage is not sufficient. Before escalating after a negative result, write the quantitative prediction that differs at the next scale, explain why the current scale could not test it, and state the observation that would further weaken the hypothesis. If repeated sensitive tests contradict that prediction, stop escalating that mechanism. Conversely, a one-gate success or failure says little about compositional credit assignment, which requires actual compositions.

### 10.3 Narrow alternatives, activated only by evidence

**Gradient obstruction with correct hard semantics.** The particular max/min relaxation can pass a one-cell gradient check yet provide sparse or tie-diluted deep gradients. First run the fixed-chain diagnostic: vary depth, independently perturb one truth logit, measure the Jacobian of the supervised output membership, and compare `frc4`, `frc4_forward_only`, and `lut4` with identical tables. Inspect whether repeated min ties divide gradients and whether only bottleneck factors learn. An untied local max/min branch has derivative zero or one before the sigmoid; this avoids multiplication by many fractional branch derivatives along some active paths but can starve almost all other paths. Neither behavior proves a net advantage.

Only when this diagnostic explains a measured learning obstruction should a smooth support aggregation or a different surrogate gradient be tested. Specify its mathematical meaning, alias handling, limiting hard behavior, and matched application to the relevant baseline. Existing soft-constraint and logic-learning work already covers broad versions of this idea. [R3,R7,R8] Mark the change as an exploratory amendment and retain the original relaxation’s result.

**Uncertainty in the learned relation, rather than uncertain wire values.** If sparse or noisy examples make hard rule mistakes the dominant failure, investigate a conservative version-space prediction from multiple consistent libraries, or a probabilistic factor model. This is not the same as a domain within one fixed relation. First use the exact small-library enumeration to quantify how much error comes from model uncertainty. Only then consider a learned approximation, compared against exact/solver version-space or probabilistic baselines under matched information. This is outside the initial core and cannot rescue a failed soundness claim retrospectively.

**Equal quality but possible runtime value.** When all learning methods give equivalent correct tables, the remaining question is engineering: generic compiled table propagation versus a specialized compact lookup implementation and direct forward evaluation. Apply equivalent optimizations to all methods. A demonstrated compact kernel can be useful without methodological novelty. No accelerator or physical energy claim follows from CPU instruction counts alone.

These alternatives remain branches of the same finite-relation investigation. Persistent graph editing, request scheduling, parallel imagined worlds, self-growing units, and QCR are not automatic fallback projects.

### 10.4 Strongest locally meaningful positive result

The strongest justified result would say, in substance: under a frozen distribution of unseen typed Boolean compositions and matched supervision, a specified support-trained four-entry cell reduces world-averaged exact-domain error relative to all competent local comparators, with uncertainty quantified; the gain survives task-aware hardening, parameterization controls, a fresh independent repeat, and at least one structural shift; total learning and inference costs identify a useful operating range.

Even that result would establish a low-level component/training advantage, not a new universal computational paradigm. Broader scaling, non-Boolean representations, grounding, topology discovery, autonomous local learning, and real applications would remain open research questions.

---

## 11. Audit record, revisions, and unresolved risks

These were three **self-audits of this plan**, not independent review. Actual validation was limited to the checks in §1.5 and Appendix A. The audits changed the design rather than awarding it a quality score.

### 11.1 Scientific and prior-art audit

| Challenge or defect found | Revision made | Residual risk and acceptance condition |
|---|---|---|
| Bidirectional, uncertainty-aware neurons already exist | Removed broad novelty language; included classical propagation, LNN, GLIDR, factor methods, and terminal-agnostic logic | A close equivalent may remain undiscovered. Any subsequent equivalent implementation reclassifies the contribution before publication. |
| A four-parameter Boolean LUT is already contemporary prior art | Added four-entry LUT, current hard-forward learning, and parameterization controls | Faithful reproduction still requires local tests and frozen source revisions. |
| “Backward reasoning” could be supplied entirely by a solver wrapper | Made the same exact backend available to every learned table | Track A can support learning claims only, not a novel reasoning algorithm claim. |
| The primitive may simply be generalized arc consistency | Required independent generic table propagation and exact equality tests | Native semantic equivalence is expected. A speed difference needs equal-quality optimized implementation comparison. |
| Small hidden libraries may make symbolic synthesis decisive | Added exact dictionary search and SAT/MaxSAT shared-table synthesis before broad neural tuning | There may be no headroom. That is a valid endpoint, not a reason to cripple the solver. |
| The source documents favor broader ecologies rather than this exact cell | Added a full disposition table and explicit narrowing | This PoC does not validate the deferred memory, imagination, growth, or chemistry proposals. |
| Attractive deep-gradient intuition could conceal dead paths | Added depth/Jacobian and tie diagnostics; no deep-trainability claim from one-cell gradcheck | Learning may fail despite correct inference. A different surrogate is an amendment. |

### 11.2 Experimental-validity audit

| Challenge or defect found | Revision made | Residual risk and acceptance condition |
|---|---|---|
| Sampled hidden values are not valid unique answers under partial evidence | Exact support domains are labels; core training uses fully observed primary inputs | Oracle completeness is a blocking gate; unresolved queries remain censored, not guessed. |
| A forward DAG can have a cyclic factor graph | Added XOR reconvergence counterexample and native-versus-exact distinction | Larger reconvergence can erase native utility; report rather than discard it. |
| Repeated wires could be treated as independent operands | Mandatory alias equality filters, renaming tests, and cross-world isolation | All exported/SAT/native paths must agree on repeated ports. |
| Extra labels, capacity, or hardening could explain improvement | Shared topology/information, parameter-count controls, same discrete refinement, total cost accounting | Supervision or optimizer changes need recorded fairness analysis. |
| Seeds and masks inflate apparent sample size | Independent library worlds, teacher-stratified paired aggregation, frozen sample design | Few worlds can remain underpowered; conservative intervals are provided. |
| Tuning could search for a favorable protected slice | Pilot/confirm/replicate separation; fixed primary strata and publication of all outcomes | Operational blinding relies on discipline; hashes and amendments make deviations visible. |
| Fixed-width depth generalization overstates scaling | Mandatory independent-input width diagnostic | The local suite still cannot establish unlimited compositional scaling. |
| Noisy data could make a hard symbolic baseline unfairly infeasible | Use soft output clauses and exact small-case optimum checks | MaxSAT may cost more; that cost is measured, not hidden. |
| Improvements might consist of invalid fast answers | Exact-domain error, false-elimination and contradiction guardrails, unresolved-result bounds | A native speedup with unsound pruning is not accepted. |

### 11.3 Implementation and feasibility audit

| Challenge or defect found | Revision made | Residual risk and acceptance condition |
|---|---|---|
| Nominal CPU code in the official logic package imports CUDA unconditionally | Adapt inspected pure mathematics, preserve notices, test gate mapping; do not install blindly | Adaptation is not a CUDA benchmark reproduction. Unsupported dependencies do not block the independent core. |
| Current web installation pages can describe different release states | Select the actual compatible native wheel and lock it only after local smoke tests | Actual macOS/MPS behavior remains unmeasured. CPU32 is the faithful fallback. |
| M3 Max memory configuration was unspecified | Hardware discovery and measured memory model precede scaling | Available memory, concurrent pressure, and autograd peaks decide batches; no maximum-memory assumption. |
| MPS max/min backward or reduction ties may differ | CPU64 reference, CPU32/MPS32 comparisons, fallback without silent mixed execution | Backend-specific results require disclosure and paired reruns when interpretation changes. |
| Microbatch splitting can change gradient scaling or shared stochastic gates | Appendix B fixes normalization and one-sample-per-effective-batch behavior | Test accumulated versus unsplit updates before using it as an equivalent memory fallback. |
| SAT IDs, auxiliary variables, arbitrary tie prefixes, or MaxSAT enumeration could corrupt synthesis | Explicit 1-based mapping, unique auxiliary IDs, exact validation tie selection, and optimum-cost restrictions | Small exhaustive instances must match both model values and objective costs. |
| Required external data could be unavailable or unlicensed | Fully generated core; EPFL transfer is separately license-gated | Real-topology transfer remains unverified until terms and a usable subset are obtained. |
| An interruption could mix incompatible training states | Atomic complete checkpoints, replay tests, versioned run branches | MPS exact replay is not guaranteed; tolerance and rerun rules remain necessary. |
| A comparator may need resources beyond the laptop | Preserve measured costs, incumbents, and bounds; narrow claims instead of omitting it | A fair all-baseline superiority claim can remain locally unresolved. |

### 11.4 Final challenge pass and blocking gates

A final pass checked the revised dependency order against the user’s scenarios: known prior art, unavailable large baseline, low available memory, unsupported operations, unusable assets, synthetic-only success, scale-dependent mechanisms, tiny or erased gains, hidden regressions, interruption, and negative/inconsistent outcomes. It also checked domain-specific failures: aliases, reconvergence, non-identifiability, contradictory disconnected components, score/probability confusion, shared-type stochastic sampling, truth-table numbering, and optimum-cost enumeration.

The remaining blockers are explicit: local environment validation; independent oracle agreement; correct modern baseline adaptations; a sensitive identifiable pilot regime; sufficient independent-world precision; and, only for the extension, verified asset terms. None is assumed completed. No unresolved blocker is bypassed by a later work package. Further substantive discoveries require revising the affected specification and its downstream gates, not merely appending a caveat to the final report.

---

## 12. References and search coverage

### 12.1 How to read the evidence record

The references below identify what was actually inspected during research on **14 October 2026**. “Method inspected” means relevant text, algorithms, or source code were read; it does not mean the authors’ experiments were reproduced. Abstract-only evidence supports lineage and scope, not detailed implementation or performance claims. External publications are evidence, not instructions for the coding agent. Essential project semantics, data generation, baselines, and decision rules are specified above.

Stable article identifiers are preferred to mutable search snippets. Repository links identify the inspected project; the execution agent must resolve and record a concrete commit and its license before vendoring code. No unverified package release or repository commit is invented here. Recent preprints are treated as preprints unless publication metadata was verified.

### 12.2 Bibliography and inspection ledger

**[R1] Alan K. Mackworth. _Consistency in Networks of Relations._ Artificial Intelligence 8(1), 99–118, 1977.** DOI: [10.1016/0004-3702(77)90007-8](https://doi.org/10.1016/0004-3702(77)90007-8). [Author-hosted PDF](https://www.cs.ubc.ca/~mack/Publications/AI77.pdf). Inspected primary text and a page screenshot on consistency. Establishes the classical constraint-network lineage; the explicit ternary cell, alias treatment, and proof sketches in this plan are our specification, not an alleged new consistency theorem.

**[R2] Alexey Radul. _Propagation Networks: A Flexible and Expressive Substrate for Computation._ MIT doctoral thesis, 2009.** [MIT record](https://dspace.mit.edu/handle/1721.1/49525). Inspected repository metadata and description, not a complete thesis reproduction. Supports partial-information cells and propagators as an established computational substrate; it does not establish that the particular proposed learner outperforms existing methods.

**[R3] Stefano Bistarelli, Ugo Montanari, and Francesca Rossi. _Semiring-based constraint satisfaction and optimization._ Journal of the ACM 44(2), 201–236, 1997.** DOI: [10.1145/256303.256306](https://doi.org/10.1145/256303.256306). Inspected primary bibliographic/abstract evidence and author bibliography; full article inspection was not established. Relevant to soft constraint combination/projection. The project’s complete max/min equations are given in §4 rather than delegated to this reference.

**[R4] Ryan Riegel, Alexander Gray, Francois Luus, Naweed Khan, Ndivhuwo Makondo, Ismail Yunus Akhalwaya, Haifeng Qian, Ronald Fagin, Francisco Barahona, Udit Sharma, et al. _Logical Neural Networks._ 2020.** [arXiv:2006.13155](https://arxiv.org/abs/2006.13155); [paper](https://arxiv.org/pdf/2006.13155). Inspected relevant method sections, Algorithms 1–3, and selected figures/screenshots. Direct prior art for logical neurons with bounds and bidirectional inference. Different truth semantics and benchmark conditions preclude an unimplemented head-to-head performance claim.

**[R5] Felix Petersen, Christian Borgelt, Hilde Kuehne, and Oliver Deussen. _Deep Differentiable Logic Gate Networks._ NeurIPS, 2022.** [arXiv:2210.08277](https://arxiv.org/abs/2210.08277). Inspected the primary method and related-work text, with implementation details checked in R6. Supplies the 16-gate relaxed/hard learning baseline. Published image-task results and compiled execution are not measurements on this laptop or this rule-discovery task.

**[R6] Felix Petersen and contributors. _difflogic_, official implementation. Accessed 14 October 2026.** [Repository](https://github.com/Felix-Petersen/difflogic); [main module](https://raw.githubusercontent.com/Felix-Petersen/difflogic/main/difflogic/difflogic.py); [functional operations](https://raw.githubusercontent.com/Felix-Petersen/difflogic/main/difflogic/functional.py). Inspected README, license label, initialization, Python operations, gradient factor, and unconditional CUDA-extension import. Supports a faithful mathematical adaptation, not an assumption that the package installs on M3. Freeze an actual source commit before reuse.

**[R7] Shakir Yousefi, Andreas Plesner, Till Aczel, and Roger Wattenhofer. _Mind the Gap: Removing the Discretization Gap in Differentiable Logic Gate Networks._ 2025; inspected version 2.** [arXiv:2506.07500](https://arxiv.org/abs/2506.07500); [versioned HTML](https://arxiv.org/html/2506.07500v2). Inspected the hard-forward/Gumbel method and relevant experiment descriptions. Motivates competent contemporary discretization baselines; image-model findings are contextual, not a guaranteed improvement for shared tiny tables.

**[R8] Wout Mommen, Lars Keuninckx, Matthias Hartmann, Werner Van Leekwijck, and Piet Wambacq. _Fully Trainable Deep Differentiable Logic Gate Networks and Lookup Table Networks._ Preprint, 10 July 2026.** [arXiv:2607.09399](https://arxiv.org/abs/2607.09399); [version 1 HTML](https://arxiv.org/html/2607.09399v1). Inspected methods for LUT entries, hard-forward learning, annealing, and connection learning. Invalidates novelty of the four-entry sigmoid LUT. The controlled task supplies connections equally; no verified software release or reproduction is assumed.

**[R9] Thore Gerlach. _Task-Aware Discretization of Differentiable Logic Gate Networks._ Preprint, 27 September 2026.** [arXiv:2609.33747](https://arxiv.org/abs/2609.33747); [version 1 HTML](https://arxiv.org/html/2609.33747v1). Inspected downstream-impact discretization and discussion of approximation limits. Informs the mandatory shared hardening control. This plan’s exhaustive 16-choice coordinate refinement is a small-library adaptation, not the paper’s full algorithm or an independently established novel method.

**[R10] Blair Johnson, Clayton Kerce, and Faramarz Fekri. _GLIDR: Graph-Like Inductive Logic Programming with Differentiable Reasoning._ 8 August 2025.** [arXiv:2508.06716](https://arxiv.org/abs/2508.06716); [version 1 HTML](https://arxiv.org/html/2508.06716v1). Inspected domain-propagation, learned-relation, and limitation discussions. Strong overlap with differentiable local consistency. A matching official code release was not established; unrelated LiDAR projects with similar names are not substitutes. No GLIDR superiority claim is made.

**[R11] Jonathan Kuck, Shuvam Chakraborty, Hao Tang, Rachel Luo, Jiaming Song, Ashish Sabharwal, and Stefano Ermon. _Belief Propagation Neural Networks._ 2020.** [arXiv:2007.00295](https://arxiv.org/abs/2007.00295). Inspected the primary abstract and scope. Supports learned factor-graph inference as prior art. Probabilistic inference/counting differs from exact Boolean support; this reference does not justify treating support scores as probabilities or claiming a faithful local BPNN replication.

**[R12] Po-Wei Wang, Priya Donti, Bryan Wilder, and Zico Kolter. _SATNet: Bridging deep learning and logical reasoning using a differentiable satisfiability solver._ ICML/PMLR 97, 6545–6554, 2019.** [Official proceedings](https://proceedings.mlr.press/v97/wang19e.html). Inspected primary description and official code link. Establishes differentiable logical optimization layers. Its global relaxation and original tasks differ from the proposed microscopic shared-table comparison; local build and reported benchmark results were not reproduced.

**[R13] Marianne Defresne, Romain Gambardella, Sophie Barbe, and Thomas Schiex. _Efficient Neuro-Symbolic Learning of Constraints & Objective._ Preprint, 28 August 2025.** [arXiv:2508.20978](https://arxiv.org/abs/2508.20978); [version 1 HTML](https://arxiv.org/html/2508.20978v1). Inspected constraint-learning assumptions and E-PLL methodology. Relevant to separating learned constraints from exact solution. The inspected manuscript contains placeholder journal DOI metadata; the stable arXiv identifier is used instead. Different supervision and problem families limit direct comparison.

**[R14] PySAT maintainers. _Installation documentation._ Accessed 14 October 2026.** [Official installation page](https://pysathq.github.io/installation/). Inspected native build prerequisites and macOS/compiler guidance. Informs the minimal `python-sat` installation and solver availability tests. It does not prove any particular solver build works on the user’s unspecified macOS; that is an early blocking local check.

**[R15] PySAT maintainers; RC2 algorithm by Alexey Ignatiev, António Morgado, and João Marques-Silva. _RC2 MaxSAT solver API documentation._ Accessed 14 October 2026.** [Official RC2 documentation](https://pysathq.github.io/docs/html/api/examples/rc2.html). Inspected constructor, usage examples, computed cost, and enumeration behavior. Cost is unsatisfied soft-clause weight; enumeration may continue into worse-cost models. This informs optimum restrictions and objective tests, not a promise of solver runtime or installation success.

**[R16] Sean R. B. Bearden, Haik Manukian, Fabio L. Traversa, and Massimiliano Di Ventra. _Instantons in Self-Organizing Logic Gates._ Physical Review Applied 9, 034029, 2018; preprint 2017.** DOI: [10.1103/PhysRevApplied.9.034029](https://doi.org/10.1103/PhysRevApplied.9.034029); [arXiv:1708.08949](https://arxiv.org/abs/1708.08949). Inspected primary abstract. Terminal-agnostic gate behavior is relevant prior art, but physical dynamics, simulated performance, and learning claims are not transferred to this software PoC.

**[R17] Massimiliano Di Ventra and Fabio Lorenzo Traversa. _Self-organizing logic gates and circuits and complex problem solving with self-organizing circuits._ Patent publication US20170316309A1, 2 November 2017.** [Publication record and description](https://patents.google.com/patent/US20170316309A1/en). Inspected bibliographic metadata and terminal-agnostic description. Relevant to novelty caution only. No current legal-status, enforceability, claim-scope, patentability, or freedom-to-operate conclusion is offered.

**[R18] Anirudh Goyal, Aniket Didolkar, Nan Rosemary Ke, Charles Blundell, Philippe Beaudoin, Nicolas Heess, Michael Mozer, and Yoshua Bengio. _Neural Production Systems: Learning Rule-Governed Visual Dynamics._ NeurIPS, 2021; revised arXiv version 2022.** [arXiv:2103.01937](https://arxiv.org/abs/2103.01937). Inspected primary abstract and publication metadata. Establishes learned entity-conditioned rules and binding as precedents for the broader brainstorm. Its visual-dynamics experiments do not validate our circuit-cell hypothesis.

**[R19] Scott Reed and Nando de Freitas. _Neural Programmer-Interpreters._ 2015 preprint; ICLR 2016.** [arXiv:1511.06279](https://arxiv.org/abs/1511.06279). Inspected primary abstract. Supports reusable learned procedures as an established lineage. No claim is made that this architecture’s training assumptions, program representation, or published generalization are reproduced by the finite-relation experiment.

**[R20] Alexander Mordvintsev, Ettore Randazzo, Eyvind Niklasson, and Michael Levin. _Growing Neural Cellular Automata._ Distill, 2020.** [Article](https://distill.pub/2020/growing-ca/); DOI: [10.23915/distill.00023](https://doi.org/10.23915/distill.00023). Inspected the local-update, training, and repair description. Supports the source documents’ self-maintenance lineage, but pattern regeneration is not evidence of learned logical composition. This is a deferred direction rather than a required core component.

**[R21] Max Willsey, Chandrakana Nandi, Yisu Remy Wang, Oliver Flatt, Zachary Tatlock, and Pavel Panchekha. _egg: Fast and Extensible Equality Saturation._ 2020 preprint; POPL 2021.** [arXiv:2004.03082](https://arxiv.org/abs/2004.03082). Inspected primary abstract. Supports non-destructive rewriting and equivalence representation as established mechanisms. It does not support merging approximately similar meanings as exact truth; equality discovery is outside the selected core.

**[R22] Honghua Dong, Jiayuan Mao, Tian Lin, Chong Wang, Lihong Li, and Dengyong Zhou. _Neural Logic Machines._ ICLR, 2019.** [arXiv:1904.11694](https://arxiv.org/abs/1904.11694). Inspected primary abstract. Relevant to learned relational operations and controlled size generalization. This broader architecture is context, not an unexecuted benchmark victory or an essential replacement for the exact matched table-synthesis baseline.

**[R23] Oscar Chang, Lampros Flokas, Hod Lipson, and Michael Spranger. _Assessing SATNet’s Ability to Solve the Symbol Grounding Problem._ arXiv posting, 13 December 2023.** [arXiv:2312.11522](https://arxiv.org/abs/2312.11522). Inspected primary abstract. Questions what intermediate supervision contributes to SATNet-style visual tasks. The archive posting date is not assumed to be the first circulation date of the critique. Its result is interpreted alongside R24, not generalized to every logical learning system.

**[R24] Sever Topan, David Rolnick, and Xujie Si. _Techniques for Symbol Grounding with SATNet._ NeurIPS, 2021.** [Official proceedings](https://proceedings.neurips.cc/paper_files/paper/2021/hash/ad7ed5d47b9baceb12045a929e7e2f66-Abstract.html). Inspected primary abstract and metadata. Describes training changes addressing symbol-grounding difficulties. Different training interventions explain why it should not be treated as directly contradicting an unchanged-condition critique. Grounding is explicitly supplied, not learned, in the core PoC.

**[R25] IBM and contributors. _Logical Neural Networks_, official implementation. Accessed 14 October 2026.** [Repository](https://github.com/IBM/LNN). Inspected README, Apache-2.0 license label, and current installation/Python requirements. Informs the isolated CPU semantics-reference branch. The exact revision’s environment must be locked; a successful repository read is not a successful local installation or a reproduced paper result.

**[R26] Rasmus Berg Palm, Ulrich Paquet, and Ole Winther. _Recurrent Relational Networks._ 2017 preprint; NeurIPS 2018.** [arXiv:1711.08028](https://arxiv.org/abs/1711.08028). Inspected primary abstract. Establishes iterative learned relational message passing as a broader alternative. A matched recurrent factor-network control is required if the project later claims native learned masked-query performance; the current exported-table claim does not silently imply superiority to RRN.

**[R27] EPFL/LSILS. _The EPFL Combinational Benchmark Suite._ Archived record, version 1, 19 February 2019.** DOI: [10.5281/zenodo.2572934](https://doi.org/10.5281/zenodo.2572934); [archive](https://zenodo.org/records/2572934); [official repository](https://github.com/lsils/benchmarks). Inspected archive and format/repository metadata. A usable license field was not exposed in the parsed record; exact file terms remain an extension gate. Real netlist topology with simulated observations is not physical deployment evidence.

**[R28] Apple. _Accelerated PyTorch training on Mac._ Accessed 14 October 2026.** [Official developer page](https://developer.apple.com/metal/pytorch/). Inspected requirements and installation guidance. Informs native Apple Silicon/MPS discovery. Its displayed release guidance and the PyTorch documentation release should not be assumed synchronized. The implementation must select a compatible available wheel and verify relevant operations locally before locking dependencies.

**[R29] PyTorch maintainers. _MPS backend and torch.mps documentation._ Accessed 14 October 2026.** [Backend notes](https://docs.pytorch.org/docs/stable/notes/mps.html); [API documentation](https://docs.pytorch.org/docs/stable/mps.html); [inspected versioned API page](https://docs.pytorch.org/docs/2.14/mps.html). Inspected synchronization, RNG, and allocated-memory APIs. The stable pages resolved to versioned documentation during research; this is not a requirement to install that release. API existence must be checked in the chosen environment.

**[R30] PyTorch maintainers. _MPS Environment Variables._ Accessed 14 October 2026.** [Stable page](https://docs.pytorch.org/docs/stable/mps_environment_variables.html); [inspected versioned page](https://docs.pytorch.org/docs/2.14/mps_environment_variables.html). Inspected CPU fallback, fast-math, and allocator-limit behavior. Supports explicit fallback disclosure and retaining memory safety limits. Do not disable allocator safeguards or silently enable a changed numerical path to make a benchmark run.

### 12.3 Search scope, refinement, and stopping record

Research began with the documents’ own families, then searched equivalent mechanisms rather than only their proposed names. The principal search clusters were:

| Cluster | Representative terms and citation paths | Consequence for the design |
|---|---|---|
| Tiny local transformations | learned rules, neural production systems, neural program composition, graph rewriting, equality saturation, neural cellular automata | Separated reusable small computations from a large interpreter or uncontrolled structural growth. |
| Partial information and constraints | propagation networks, finite-domain table constraints, generalized arc consistency, semiring constraints, logical neural networks | Found established semantics covering the proposed primitive; removed broad novelty claims. |
| Differentiable relation inference | differentiable domain propagation, GLIDR, belief propagation neural networks, learned constraints, exact-solver learning, E-PLL | Found close overlap and imposed a learning-versus-inference separation. |
| Learnable Boolean neurons/layers | differentiable logic gates, four-entry LUT learning, straight-through logic, Gumbel gates, annealing, task-aware discretization | Added the strongest relevant inspected 2025–2026 refinements, not just the 2022 baseline. |
| Bidirectional logic outside neural terminology | terminal-agnostic logic, self-organizing logic gates, reverse gate computation, related patent disclosures | Rejected novelty based solely on usable input/output reversal. |
| Competing exact methods | shared truth-table synthesis, SAT/MaxSAT, PySAT native builds, RC2 optimal models | Made exact synthesis an early headroom test and a direct competitor. |
| Local execution and assets | Apple MPS support, PyTorch gather/reduction/backward, memory/fallback APIs, official package source, EPFL netlist formats and archive terms | Identified CUDA import, mutable-version, numerical, and license gates before dependent work. |
| Adverse evidence | SATNet grounding critiques and revised training, discretization gaps, cyclic propagation limitations | Prevented favorable-only interpretation and required explicit supervision/correctness controls. |

Backward tracing covered classical consistency, propagators, semiring constraints, symbolic composition, and terminal-agnostic logic. Forward tracing from differentiable gates and logical layers found recent Gumbel/STE, directly trainable LUT, domain-propagation, constraint-learning, and task-aware discretization work. The last targeted passes checked equivalent domain semantics, current hardening/training alternatives, solver API traps, and the actual CPU/MPS path. They did not yield another materially distinct mandatory comparator for the **frozen small shared-table learning question** beyond those incorporated here.

This is **scoped search saturation**, not proof that no prior art exists. Important blind spots remain: unpublished work; poorly indexed or non-English sources; exact mathematical equivalence to an uninspected soft-propagation implementation; full reproduction of GLIDR/LNN/BPNN; solver-specific scaling; and the actual target laptop. Some abstracts were inspected without full papers, an OpenReview access path was obstructed, and exact EPFL asset permissions were unresolved. Recent preprint findings may change after the research date. A future materially equivalent method or stronger executable comparator must update the novelty statement and evaluation before a strong public claim.

The patent search was contextual and non-exhaustive. Nothing here is a legal patentability or freedom-to-operate opinion. The core requires no unavailable proprietary asset, remote accelerator, paid API, or specialized physical equipment.


---

## Appendix A. Reproduce the small checks executed during research

Save the following code as `checks_reference.py` in the tested Python environment and execute `python checks_reference.py`. This is standalone reference code, not a complete project implementation or a laptop performance benchmark. It uses only NumPy and CPU PyTorch. The hard-domain checks could run without PyTorch after omitting the last numerical check.

The expected semantic fields are `single_cell_cases=1024`, `factor_tree_cases=900`, `cycle_counterexample.solutions=0`, `cycle_counterexample.sweeps=1`, the six retained domains `[[0,1],[0,1],[0,1],[0],[0],[1]]`, `alias_xor_domains=[[],[]]`, and `one_cell_gradcheck=true`. Environment fields will differ on the target laptop. These test counts establish only the stated finite checks.

```python
"""Small semantic checks; not a benchmark and not an M3 Max validation."""
from itertools import product
import json, platform, sys
import numpy as np
import torch

TUPLES = np.array(list(product((0, 1), repeat=3)), dtype=np.int64)
DOMAINS = [set(), {0}, {1}, {0,1}]

def table(g):
    return np.array([c == ((g >> (2*a+b)) & 1) for a,b,c in TUPLES])

def revise(dom, scope, allowed):
    good=[]
    for t, ok in zip(TUPLES,allowed):
        if not ok: continue
        if any(t[i]!=t[j] for i in range(3) for j in range(i) if scope[i]==scope[j]): continue
        if all(int(t[p]) in dom[v] for p,v in enumerate(scope)): good.append(t)
    out=[set(x) for x in dom]
    for p,v in enumerate(scope): out[v] &= {int(t[p]) for t in good}
    return out

def closure(dom,factors):
    d=[set(x) for x in dom]; sweeps=0
    while True:
        old=[set(x) for x in d]
        for scope,allowed in factors: d=revise(d,scope,allowed)
        sweeps +=1
        if old==d:return d,sweeps

def exact(dom,factors):
    sol=[]
    for x in product(*[sorted(s) for s in dom]):
        if all(allowed[4*x[s[0]]+2*x[s[1]]+x[s[2]]] for s,allowed in factors):sol.append(x)
    return [{x[i] for x in sol} for i in range(len(dom))],len(sol)

single=0
for g in range(16):
    for d in product(DOMAINS, repeat=3):
        a=revise(d,(0,1,2),table(g));b,_=exact(d,[((0,1,2),table(g))])
        assert a==b;(single:=single+1)

# Trees of ternary factors, with each new factor introducing two new variables.
rng=np.random.default_rng(20261014)
trees=0
for nfac in (1,2,3):
    for _ in range(300):
        nv=1+2*nfac; factors=[]
        for j in range(nfac):
            factors.append(((int(rng.integers(1+2*j)),1+2*j,2+2*j),table(int(rng.integers(16)))))
        ds=[DOMAINS[int(rng.integers(1,4))] for _ in range(nv)]
        a,_=closure(ds,factors);b,ns=exact(ds,factors)
        # Connected factor tree: GAC detects all inconsistency, eventually empties all.
        assert a==b
        trees+=1

# Reconvergent constraints: x=y, y=z, x!=z, implemented as XOR cells with pinned output.
factors=[((0,1,3),table(6)),((1,2,4),table(6)),((0,2,5),table(6))]
ds=[{0,1},{0,1},{0,1},{0},{0},{1}]
a,steps=closure(ds,factors);b,nsol=exact(ds,factors)
assert all(a) and nsol==0

# Binding alias within a cell: XOR(x,x)=1 is locally impossible when port identity is honored.
a_alias=revise([{0,1},{1}],(0,0,1),table(6))
assert a_alias == [set(),set()]

# A CPU float64 gradient check on a one-factor soft, observed-input case away from ties.
torch.manual_seed(17)
tt=torch.tensor(TUPLES)
def soft_output(theta):
    p=theta.sigmoid()
    ridx=2*tt[:,0]+tt[:,1]
    r=torch.where(tt[:,2].bool(),p[ridx],1-p[ridx])
    # Select known a=1,b=0 by a Boolean admissibility mask; take max by output value.
    r=r*((tt[:,0]==1)&(tt[:,1]==0)).to(torch.float64)
    return torch.stack([r[tt[:,2]==c].amax() for c in (0,1)])
theta=torch.tensor([-1.3,0.7,-0.4,1.1],dtype=torch.float64,requires_grad=True)
gradok=torch.autograd.gradcheck(soft_output,(theta,),eps=1e-6,atol=1e-6,rtol=1e-4)
result={'environment':{'platform':platform.platform(),'python':sys.version.split()[0],'numpy':np.__version__,'torch':torch.__version__,'mps_available':torch.backends.mps.is_available()},'single_cell_cases':single,'factor_tree_cases':trees,'cycle_counterexample':{'solutions':nsol,'gac_domains':[sorted(d) for d in a],'sweeps':steps},'alias_xor_domains':[sorted(d) for d in a_alias],'one_cell_gradcheck':gradok,'seed':20261014}
print(json.dumps(result,indent=2))
```

## Appendix B. Implementation details that must not be left implicit

### B.1 SAT variable numbering and optimum handling

Circuit wire IDs are zero-based, but SAT literals use nonzero signed integers. For an encoding with `K` shared types, assign

```text
truth_id(k, a, b) = 1 + 4*k + 2*a + b
wire_id(example_i, v) = 1 + 4*K + sum(V_j for j < i) + v
```

The second sum uses the actual wire count of each earlier example. Keep truth IDs and wire IDs in a serialized variable map. Allocate auxiliary variables strictly above every base ID using a single ID pool; verify that solver/cardinality helpers do not reuse occupied IDs. Repeated ports map to the **same** SAT variable. For a query against an already fixed dictionary, truth variables can be eliminated; freeze the chosen encoding convention and validate it against enumeration.

With clean data, first ask SAT for training-consistent models, then apply the exact validation tie selection in §4.9; validate every returned dictionary by independent forward execution. For noisy labels, RC2 minimizes unsatisfied soft-label weight, with the explicit lexicographic weights when validation selection is included. Recompute that cost independently from the exported dictionary. Its enumeration API can yield higher-cost models after exhausting an optimum; do not treat all yielded models as optimal. [R15]

For identifiability after an optimum training-only cost `c*` is proved, constrain the number of violated unit-weight training labels to `≤c*` and ask the underlying SAT problem with each truth bit forced to each value. Represent a label-violation indicator exactly: if the observed bit is one, violation is `¬output`; if zero, violation is `output`. A documented cardinality encoding over these literals is sufficient; give its auxiliaries fresh IDs and validate on enumerated small cases. At a proved optimum `≤c*` is equivalent to `=c*`. Without an optimum proof, report incumbent-relative ambiguity, not global identifiability.

For the `K≤4` exhaustive diagnostic, iterate canonical dictionaries in lexical order, chunk candidates to fit measured memory, and evaluate identical output-label disagreement. Prune a candidate only once its nonnegative accumulated loss already exceeds the incumbent; equal-loss candidates remain eligible for the recorded tie analysis. This is an optimization of exact enumeration, not privileged knowledge. Training/validation ties are never broken with hidden test outcomes or teacher distance.

### B.2 Microbatching, stochastic choices, and checkpoint recomputation

Let one effective minibatch have `N` supervised membership entries, where each observed output contributes two entries. For microbatch `r` with loss sum `L_r` and `N_r` entries, backpropagate `L_r/N`, not the mean loss independently for every microbatch. Equivalently, multiply its mean by `N_r/N`. Take one optimizer step after all microbatches, and advance the optimizer-dependent schedule once. Empty-supervision batches are invalid and must not produce division by zero.

Draw type-shared Gumbel noise once per **effective minibatch**, reuse it across microbatches and repeated occurrences, and preserve it during activation-checkpoint recomputation. A recomputed block that resamples gates changes both the function and the gradient. Save the generator state and effective-batch index; interruption midway through an uncommitted effective batch replays the whole batch. CPU initializations and data ordering must likewise be paired when a comparison depends on identical starting tables.

Test accumulated versus unsplit CPU64 and CPU32 updates on unequal microbatch sizes. Compare loss, parameters, optimizer moments, selected hard tables, and the next random draw. Bitwise identity is not generally expected under changed floating reduction order, but discrepancies exceeding the declared numeric tolerance or changing the substantive result are not an equivalent fallback. The primary run then uses a fixed tested accumulation policy for all relevant competitors.

### B.3 A competent compact native baseline

Only after semantic tests pass and native timing could change the interpretation, implement an optional small compiled full-sweep kernel. The local state space is finite. Precompute support updates for each gate code, each legal alias pattern, and each three-port domain-code combination. For a fully general three-port interface, there are five equality partitions, 16 gates, and 64 domain triples. Three one-byte output codes occupy `5×16×64×3 = 15,360` bytes. The forward-DAG generator permits only distinct ports or equal input ports, so a two-pattern specialization needs 6,144 bytes. These are derived lookup sizes, not measured speed or a new algorithm.

Mask invalid alias combinations explicitly; a shared wire cannot simultaneously enter with two different domain codes. Use the same snapshot/alias rules as the reference implementation. A C function can receive contiguous port/type arrays and `uint8_t` domains, sweep to closure, and return revision count plus inconsistency status. Use the detected native compiler and a reproducible build manifest; a standard library loaded through `ctypes` or a standalone executable is sufficient. The agent must supply and test the build script—no fictional prebuilt binary is required.

Apply this precomputation and equivalent batching to the generic propagator as well. Compare a compiled direct forward evaluator for fully observed inputs. Count Python/FFI overhead in end-to-end latency, excluding it only in a separately labeled kernel microbenchmark. Without this competent baseline, a Python-versus-C timing gap is not evidence for a new primitive. The dense native sweep remains outside QCR: it does not propagate signed numerical deltas or certify an approximate recurrent fixed point.

### B.4 Small aggregation and uncertainty fixtures

Create a two-world fixture in one teacher stratum. World A has 100 query records with baseline error 0.2 and candidate error 0.1; world B has 10 records with baseline error 0 and candidate error 0.2. With equal world weights, the paired effect is `(0.1−0.2)/2 = −0.05`, **not** the query-weighted positive effect. Construct both teacher strata explicitly and verify their aggregate weights remain one-half even when their world counts differ.

For implementation of the max-t procedure, a NumPy array of shape `[worlds_in_stratum, M]` is sufficient after all nested averaging. Resample its rows, not its columns or constituent queries; preserve the correspondence of every baseline with the same candidate/world. Use sample variance with one degree of freedom correction. Fewer than two worlds in either stratum makes the variance estimate undefined and must block a studentized interval. The bounded fallback can still be reported, usually with uninformative width.

Test identical columns, sign-reversed method labels, missing outcomes, zero variance, one extreme world, unequal stratum sizes, and nonzero optimizer variance averaged within worlds. Confirm that every reported interval and outcome can be regenerated from immutable raw records.

### B.5 Final completion checklist for the executing agent

Before publishing a comparative conclusion, verify: the source-derived narrowing is explicit; no QCR mechanism has entered the implementation; every table-exporting learner has the same exact query backend; modern LUT/logic training and common hardening are competent; aliases and exact labels pass independent checks; the protected protocol and all run statuses are accounted for; costs include material overhead; hardware estimates have been replaced by local measurements where claimed; unsuccessful results remain visible; and the stated conclusion is no broader than the supported population and component.

The project is complete when it produces a reproducible, interpretable answer to the specified question—even when that answer is that the primitive is known, its training is inferior, or no useful advantage was detected at meaningful local scales.
