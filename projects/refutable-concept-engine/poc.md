# Research and local proof-of-concept plan
## Refutable concepts: learning reusable experiments as well as reusable programs

**Research date:** 14 October 2026  
**Target:** one MacBook Pro with an M3 Max; actual memory, storage, macOS, and sustained throughput must be discovered locally.  
**Deliverable status:** research and implementation handoff, not a report of a completed architecture or a demonstrated speedup.  
**Core requirement:** no remote accelerator, paid model API, proprietary training corpus, or specialized physical equipment.

**Navigation:** [Assessment](#1-executive-assessment) · [Hypotheses](#2-problem-scope-and-falsifiable-hypotheses) · [Prior art](#3-research-findings-and-novelty-assessment) · [Mechanism](#4-formal-mechanism-and-implementation-specification) · [Evaluation](#5-evaluation-populations-baselines-and-measurement-boundaries) · [Statistics](#6-confirmatory-design-and-statistical-decisions) · [Executed check](#7-small-checks-executed-during-this-research) · [M3 feasibility](#8-m3-max-feasibility-and-hardware-adaptive-execution) · [Work packages](#9-risk-ordered-work-packages-and-decision-gates) · [Ablations](#10-attribution-robustness-and-validation-of-the-harness) · [Agent contracts](#11-coding-agent-handoff-repository-contracts-configuration-and-records) · [Decisions](#12-outcome-interpretation-escalation-and-evidence-triggered-alternatives) · [Audits](#13-audit-record-changes-made-and-residual-risks) · [References](#14-references-and-search-coverage-record) · [Script](#appendix-a-complete-executed-finite-domain-check)

---

## 1. Executive assessment

### 1.1 Verdict

The motivating idea is scientifically legitimate, but its broad form is substantially established. Learning executable rules, revising hypotheses after counterexamples, inventing reusable abstractions, choosing informative queries, and remembering distinguishing experiments all have substantial prior art. A new name for their combination would not constitute a new learning principle. Particularly important counterexamples to an overly broad novelty claim are Popper, STEVIE, oracle-guided synthesis, interactive synthesis, adaptive automata learning, and Boolean functional synthesis [R01], [R02], [R03], [R04], [R05], [R06], [R11], [R12], [R13], [R14], [R15], [R20], [R21], [R22].

The strongest research direction identified here is a **neuron-free learner whose reusable knowledge includes both an executable concept and a procedure for distinguishing it from competing concepts**. The proposed implementation is called the **Refutable Concept Engine**, and its reusable objects are called **contrastive concept capsules**. These are working project labels, not novelty claims.

A capsule does not merely say “this program explains these observations.” It also stores, in executable and checked form, “under these conditions, this experiment separates this explanation from that alternative.” The experiment is parameterized so that it can transfer to different constants and unfamiliar surrounding computations. It must be checked in the complete new context before use.

**The proposed scientific contribution is narrow:** discover useful local contrast schemas from a history of synthesis ambiguities, learn compact experiment constructors for them, and determine whether context-sensitive reuse reduces the cost of acquiring new executable models beyond strong existing synthesis, caching, functional-synthesis, and query-selection methods. The key uncertainty is not whether such objects can be represented. They can. It is whether the learned decomposition and reuse policy save more work than they introduce.

The mathematical subproblem of constructing a parameter-dependent witness is already functional/Skolem synthesis. Learning such witnesses from samples with proof-guided repair also already exists, including in non-neural form. Therefore, **neither the witness equation nor a sample–verify–repair loop is claimed as a new algorithm** [R20], [R21], [R22]. Methodological novelty of the proposed cross-task packaging and selection is unresolved. A reproducible engineering improvement, a new empirical finding about transfer, or a well-supported negative result would each be legitimate outcomes.

### 1.2 What to build first

Build an exact, finite, typed program domain and the **experiment-construction component**, then integrate it into a small complete active program learner. Use Boolean and fixed-width bit-vector programs first. This is an explicit refinement of the original object-world proposal: finite semantics permit exhaustive checks and solver-checked counterexamples, allowing the proposed learning operation to be isolated without confounding it with perception or simulator design.

The first confirmatory claim is component-level:

> On a frozen population of previously unseen synthesis-ambiguity workloads, does a learned contrast library reduce the time to a valid distinguishing input, relative to the strongest validated local comparison, after lookup, context lifting, validation, failed attempts, and fallback are included?

The second claim, requiring separate evidence, is system-level:

> With the same target information, program search backend, forward library, and total acquisition budget, does the complete learner recover correct target behavior faster or with fewer queries on new tasks?

A component result does not have to wait for a large general-purpose system to be meaningful. Conversely, it must not be promoted into a claim about general intelligence, raw perception, language understanding, a universally superior neuron replacement, or reduced physical energy consumption.

### 1.3 Evidence available now

Current primary literature, official documentation, and selected public implementation files were inspected. A small pure-Python finite-domain check was executed in the research environment. It recovered a simple parameterized diagnostic input and exposed two context-transfer failure modes. It did **not** compare performance against solvers or train the complete engine. The environment was Linux x86-64, not an M3 Max. Section 7 and Appendix A give the complete check and its output.

Three blocking uncertainties precede expensive work: whether contemporary local baselines reproduce correctly; whether real synthesis traces contain recurrent useful contrasts rather than only hand-designed examples; and whether savings survive ordinary caching, direct functional synthesis, and full-context validation.

### 1.4 Architectural choice, without assuming that symbolic means better

| Candidate direction | Mechanism and relevance to the brief | Main unresolved issue | Decision |
|---|---|---|---|
| New scalar/neuron-like primitive | Could alter representation, optimization, or hardware efficiency. | The brief specifies no concrete low-level update law from which distinctive predictions follow. | Do not invent a nominal neuron replacement merely to satisfy a metaphor. Revisit only with a specific operation and measurable advantage. |
| Clause/automaton learning | A genuinely different small learning unit already exists in Tsetlin machines [R26]. | A fresh clause implementation alone is not original, and classification gains would not establish concept invention. | Credible alternative for a future Boolean-classification objective, not the selected hypothesis. |
| Constraints plus program invention | Directly preserves the brief's emphasis on revisable explanations and reusable concepts. | Search and hypothesis-language bias can dominate; close prior art is strong. | Select, with the narrow contrast-transfer mechanism below. |
| Memory and analogy | Reuse can reduce repeated search. | Ordinary cached solutions, symbolic summaries, and adaptive test reuse may explain the whole benefit [R12], [R13], [R14], [R15]. | Make these mandatory competitors rather than call them novel components. |
| Evolutionary program populations | Can discover executable functionality without neural units; digital-organism experiments establish a limited precedent [R27]. | Search evaluation cost and within-lifetime adaptation are separate questions. | An optional constructor-search operator only if enumeration is the measured bottleneck. No evolutionary ecosystem in the core. |
| Physical/local dynamics | Could change the physical implementation of computation. | Software simulation on this laptop cannot establish the energy or latency of an unbuilt physical device. | Not selected for this rule-discovery hypothesis. This is a scope judgment, not evidence against physical computation. |

---

## 2. Problem, scope, and falsifiable hypotheses

### 2.1 Intended beneficiaries and existing workflow

The initial beneficiaries are developers and researchers building interactive program learners, small software-model discovery tools, and exact black-box diagnostic systems. A user supplies examples of an unknown transformation, or a local executable oracle supplies answers to chosen inputs. A synthesizer searches for programs consistent with the observations, resolves ambiguity with more examples, and eventually returns a predictive program.

Interactive and oracle-guided synthesis are existing workflows, not newly identified markets [R01], [R02]. This project has **not** established customer demand, a deployment query price, or a real-world bottleneck. Its first practical question is whether repeated experiment construction is a material part of acquisition cost in the chosen workflows. Profile it before claiming headroom.

The potential bottleneck is repeated rediscovery of essentially the same distinguishing experiment across new program parameters and contexts. For example, several candidate explanations may differ only at an arithmetic boundary. Learning the boundary once might help later, but a solver or a few ordinary boundary probes may already solve the problem cheaply.

### 2.2 Inputs and outputs

A task has an unknown deterministic target function

\[
f^* : \mathcal X \longrightarrow \mathcal Y,
\]

an explicitly declared input-validity predicate \(V(x)\), a public typed grammar \(\mathcal G\), and observed examples \(D_t=\{(x_i,f^*(x_i))\}_{i=1}^t\). Candidate programs are executable members of \(\mathcal G\), possibly using learned macros that expand into the original grammar.

The learner may evaluate its own candidates freely, paying computation cost. It may request a target answer only through the oracle interface, paying one query and its measured execution cost. It does **not** see the target source, a target-equivalence oracle, hidden evaluation inputs, or ground-truth counterexamples. The evaluator may use target source privately to score frozen snapshots, but its results must not flow back into the learner.

The component's input is a pair of known candidate programs \(h,h'\), their type/width, and \(V\). Its output is either a validated \(x\) satisfying

\[
V(x)\land h(x)\ne h'(x),
\]

an independently checkable equivalence result, or an explicit non-answer such as `unknown`, `timeout`, or `unsupported`. In the component benchmark the pair programs are inputs to every method; this is not target-source leakage. In the active learner those programs are hypotheses, not privileged knowledge of the target.

### 2.3 Definition of “neuron-free” for this project

The selected learner uses typed syntax trees, finite symbolic states, counters, exact arithmetic, program enumeration, library refactoring, and SAT/SMT-based checks. It has no learned neural representation, neural proposal distribution, neural query policy, or pretrained model. A solver may use ordinary numerical heuristics; a counter is not an artificial neuron merely because it is a number.

This is a distinction in representation and learning algorithm, not a claim about which universal machine can simulate which other machine. A Boolean program can be implemented by many substrates. The project is not attempting to prove that its computations cannot be emulated by a neural network.

The learned macros in the core **do not enlarge the extensional function class** beyond the base grammar and resource bound. They alter description and search organization. Discovery of new sensory variables, new mathematical primitive semantics, or an unbounded self-expanding language is outside the initial claim.

### 2.4 Assumptions and boundaries

| Category | Core assumption | Consequence when violated |
|---|---|---|
| Oracle | Deterministic, stationary, and callable on valid inputs. | Noise and drift require a different inference contract; test rejection/detection, not silent elimination of contradictory observations. |
| Representation | Inputs are structured values, not pixels or natural language. | Success says nothing about object discovery or perception. |
| Hypothesis class | Synthetic in-class tasks have a solution in the disclosed grammar and expanded-size bound. | Out-of-class tasks must expose model misspecification; no unconditional learning guarantee. |
| Recurrence | Some local ambiguities recur across different full tasks. | If trace recurrence is negligible, a learned contrast library may have no useful application in that distribution. |
| Transfer | A local witness sometimes remains reachable and observable in new contexts. | Lifting and validation may consume all savings. |
| Compute | Native CPU symbolic execution is available; MPS is optional for neural controls. | Do not depend on GPU support for search, solver calls, or exact bit operations. |
| Resources | Actual RAM/storage are unknown. | Select concurrency, grammar size, and candidate counts from measurements, not the maximum advertised M3 configuration. |

### 2.5 Hypotheses, alternatives, and falsifiers

| ID | Testable statement | Supporting evidence or rationale | Strong alternative explanation | Evidence against it |
|---|---|---|---|---|
| H0: sound transfer | Every emitted capsule-generated input distinguishes the complete supplied candidate pair on a valid input. | Direct validation makes this a checkable invariant; the small check shows why context matters. | No learned mechanism is needed to guarantee this; validation itself is standard. | Any accepted invalid witness, false equivalence claim, or stale certificate is an implementation defect and blocks evaluation. |
| H1: reusable diagnostic structure | Held-out task contexts exhibit useful matches to automatically discovered local contrast schemas. | Reusable program abstractions and test reuse are established separately [R05], [R07], [R12]. | Repeated exact tasks, constants, or deliberately chosen mutations create artificial recurrence. | Matches vanish after full-task/context splitting; only exact input replay works; independent traces show no recurrence. |
| H2: component benefit | With equal online budgets and access to source assets, capsules improve restricted-mean valid-witness time against the strongest validated baseline. | Replacing repeated search with a small executable constructor may save work; no positive measurement yet. | Generic constraint caching, a Skolem function, a hand-written boundary heuristic, or preprocessing explains the gain. | Confidence interval rules out the decision-relevant saving, or stronger shared optimizations erase it. |
| H3: concept coupling | Attaching contrasts to learned forward concepts is more useful than an independently learned contrast library of matched storage and training cost. | Common learned program boundaries might expose transfer sites. This is the most differentiated architectural hypothesis. | Anti-unification on raw syntax exposes the same sites; forward macros are only cosmetic. | Decoupled local templates match performance; an unlinked library works equally well. In that case do not claim a new learning unit. |
| H4: acquisition benefit | The full active learner improves exact target recovery versus time or queries without a concealed regression. | Better experiments can reduce acquisition effort; selection overhead is known to matter [R11]. | More target information, extra search, a different candidate pool, or target-aware stopping. | Paired runs with the same backend/library show no gain; saved query time is swamped by synthesis or library costs. |
| H5: robustness envelope | Useful gains persist beyond the motivating algebraic fixture and have identifiable failure boundaries. | A testable extrapolation, not established evidence. | The implementation encodes the benchmark's answer or exploits a narrow DSL. | Gains occur only for the explicitly demonstrated `<` versus `<=` motif, disappear on natural ambiguity traces, or fail independent benchmark provenance. |

H2 is the first confirmatory hypothesis. H3 is the architecture-specific attribution test. H4 requires its own frozen experiment. H1 is a diagnostic precursor, not a significance-filter that permits quietly discarding a negative population. H0 is a correctness requirement, not a statistical achievement.

---

## 3. Research findings and novelty assessment

### 3.1 Closest prior art

The table records exact overlap, rather than describing every neighboring field at equal length. “Available” means publicly inspectable, not locally executed or necessarily redistributable. Published results on different tasks are not head-to-head comparisons for this project.

| Work | Problem and mechanism | Evidence/conditions and computational implications | Exact overlap and consequence here | Reuse/implementation status |
|---|---|---|---|---|
| Oracle-guided component synthesis, Jha et al. [R01] | Infer a program through oracle interaction and distinguish candidates. | Component-structured synthesis; requires an appropriate oracle and hypothesis vocabulary. | Candidate disagreement plus chosen input is old. | Reuse the formulation; implement with current solvers rather than an unsupported historical binary. |
| Interactive Program Synthesis, Le et al. [R02] | Incremental hypothesis spaces and proactive disambiguating questions. | Evaluated in example-driven transformation settings; domain-specific structure matters. | “Hypothesis → question → refinement” is not a new loop. | Reuse incremental-interface ideas; not an assumed drop-in bit-vector engine. |
| FlashMeta, Polozov and Gulwani [R03] | Operator-specific inverse/witness semantics guide synthesis. | Structured DSLs and supplied semantics; decomposition can substantially constrain search. | Backward reasoning and witness propagation are old. | Reuse semantic decomposition ideas; preserve a generic solver fallback. |
| Popper, Cropper and Morel [R04] | Generate, test, then constrain logic-program hypotheses from failure. | ASP/Prolog implementation and bounded language bias; original paper's feature set must not be confused with later versions. | Contradiction-driven structural revision is already implemented. | Official repository is the baseline for a later relational-domain branch, not a forced bit-vector translation. |
| STEVIE, Hocquette et al. [R05] | Discover higher-order abstractions for logic learning. | Refactoring and downstream induction, with explicit background language. | Non-neural abstraction invention is not an empty research area. | Reuse/compare in a compatible logic domain; do not call DreamCoder the only abstraction baseline. |
| Playgol, Cropper [R06] | Learn reusable background programs through play. | Self-generated tasks and cross-task reuse; bounded search remains central. | Self-play followed by concept reuse is established. | Source-task generation here must be charged and shared, not treated as free experience. |
| DreamCoder, Ellis et al. [R07] | Jointly develop program libraries and a neural recognition/search model. | Library and recognition ablations are explicitly studied. | Program-library learning is established; removing neural guidance alone is not novel. | Useful broader reference. No unsupported claim that every experiment or variant requires a neural guide. |
| Stitch, Bowers et al. [R08] | Top-down library learning for compression. | Efficient structural abstraction search; compression is the objective, not a proof of future acquisition savings. | Reuse its abstraction machinery rather than rebuild a large library learner. | MIT repository; Python bindings/native Rust build require local checks. |
| babble, Cao et al. [R09] | Abstraction discovery using e-graphs and anti-unification. | Exploits equivalences to expose shared structure. | Equivalence-aware concept discovery is not a novel add-on. | Reference for stronger canonicalization; full e-graph integration is not required before the basic test. |
| Learning Abstractions for Program Synthesis, Wang et al. [R10] | Learn abstraction predicates and transformers to assist synthesis. | Abstract interpretation/interpolation setting. | “Invent constraints instead of only adjusting weights” is already a technical research program. | Limits high-level novelty; no unsupported perception claim. |
| LearnSy, Ji et al. [R11] | Efficiently choose informative questions for oracle-guided synthesis. | Grammar-aware estimation and dynamic programming; original build has Linux-specific dependencies. | The query-cost bottleneck has a strong direct competitor. | Port gate for query-selection comparisons; do not substitute simple entropy and call it LearnSy. |
| Adaptive automata learning [R12], [R13] | Reuse prior observations and distinguishing information while relearning changed systems. | Stateful/automata assumptions differ from pure functions. | Remembering tests and reusing separators is established. | Use as novelty constraint and inspiration for a trace-replay control, not an incomparable score. |
| GreenTrie [R14] | Reuse constraint solutions beyond exact syntactic identity. | Symbolic execution and implication-aware retrieval. | Exact-cache-only comparisons would be weak. | Implement a disclosed implication/subset-aware control; do not call a minimal port a full reproduction. |
| Compositional dynamic test generation, Godefroid [R15] | Reuse function summaries during test generation. | Path/context semantics and summaries are essential. | Context-sensitive test reuse and decomposition are old. | Reuse summary principles and charge their computation for both methods. |
| EcoSearch [R16] | Efficient cost-ordered program enumeration. | CPU experiments reported on a 16 GB machine; this supports plausibility of a local search study, not M3 performance. | A naive Python enumerator is not enough for competitive synthesis claims. | Native/compatible implementation gate; common optimized backend for attribution. |
| CrossBeam and AbstractBeam [R17], [R18] | Neural search, and neural search combined with learned abstractions. | Domain-specific encoders, operations, and training; original CrossBeam setup names CUDA/torch-scatter. | A tiny untuned MLP would not establish superiority to neural program learning. | Broader reference and conditional faithful local port; not essential to a solver-component claim. |
| SmartLabel [R19] | Active learning with uncertain neural program components. | Different noise and semantic assumptions from an exact deterministic DSL. | Informative-query learning is contemporary and extends beyond exact logic. | Relevant only after a noisy/neurosymbolic branch is explicitly introduced. |
| Manthan / Manthan2 [R20], [R21] | Learn Boolean Skolem functions and repair/verify them with reasoning. | Known relation → executable witness functions; non-neural learning can be used. | The capsule's parameterized witness is an instance of an existing synthesis problem. This invalidates the broadest proposed novelty. | Current repository describes macOS builds, but also an explicit reproducibility/implementation-bug notice. Reproduce and independently check; never import headline performance as settled fact. |
| Symbolic Boolean realizability/synthesis [R22] | Construct realizability sets and witnesses using symbolic decomposition. | Boolean representation and decomposition quality determine cost. | Compiling a relation into a reusable witness is also established outside sample-based learning. | Include a direct functional-synthesis baseline, not only ground SMT calls. |
| Clouseau, Zhou et al., 2026 [R23] | Synthesize effectful test generators from underapproximate type/trace specifications. | Black-box effectful systems and supplied specification language. | “Learning executable experiments” by itself is not novel, including beyond pure functions. | Closest newer neighboring test-generator work; core has a different cross-task contrast-discovery objective. |
| Abstraction selection by bounded search cost, Cuculich, 2026 [R24] | Study selection criteria beyond compression alone. | Recent non-peer-reviewed preprint; independently unreplicated here. | Even “select concepts for search value rather than compression” is not a safe novelty claim. | Treat as relevant preliminary evidence and a search lead, not established superiority. |

### 3.2 What remains potentially different

The proposed contrast library is **discovered from actual ambiguities**, not supplied as a hand-written catalogue of semantic inverses. It is local rather than compiled independently for each complete ground formula. Its entries are linked to learned forward abstractions and retained according to held-out future acquisition cost, including reachability and observability in new contexts.

That conjunction could matter because the same local distinction may recur inside many differently parameterized tasks. It could also be entirely explained by familiar compositional summaries and memoization. The only defensible initial position is:

* **Established:** executable concept learning, active synthesis, witness/Skolem synthesis, learned test generators, and reuse of tests or constraints.
* **Plausible differentiation:** jointly selecting forward concepts and local contrast procedures for cross-task acquisition, with verified context lifting.
* **Unresolved:** whether this conjunction is methodologically new, whether coupling helps, and whether any advantage survives stronger baselines.
* **Not supported:** “a new fundamental law of intelligence,” “the first neuron-free concept learner,” or “an alternative that will outperform neural networks generally.”

The name “capsule” has no evidentiary role. It is simply a serialization unit and a causal hypothesis about what should be reused.

### 3.3 Discriminating tests and novelty invalidators

| Candidate contribution | Decisive comparison | What invalidates or narrows the claim |
|---|---|---|
| Parameterized witnesses | Direct cvc5 synthesis of the same witness relation; Manthan when validated locally. | Comparable performance means witness learning itself adds no new advantage. |
| Local context transfer | Ground witness replay, common-context summaries, global-schema functional synthesis. | Identical cache keys or fixed full schemas explain all gains. |
| Forward/contrast coupling | Same source assets and resource budget; coupled versus independent contrast libraries; 2×2 forward-library/contrast-library experiment. | No interaction or no extra value over raw-syntax anti-unification. |
| Learning from ambiguity history | Automatically collected traces versus hand-designed contrast pairs; source-to-target split. | Only specially selected hard motifs produce gains. |
| End-to-end acquisition | Same synthesizer, same public grammar, same oracle, same budget; then independent strong synthesizers. | Extra information or stronger search, not contrast transfer, explains the result. |

Finding an earlier method with the same schema-discovery, context-transfer, and cost-selection operations would change the project to replication, optimization, or a new evaluation. It would not make the experiments useless. Record that change rather than continue advertising a methodological invention.

### 3.4 Reconciling apparently conflicting evidence

Compression can reduce the number of syntactic steps needed to express a solution while increasing the number of possible next steps in search. Thus a good compressor need not improve acquisition. Neural guidance may improve which abstractions are discovered, while non-neural library systems demonstrate that neural guidance is not logically required. These are compatible findings, not evidence that one tradition has disproved the other [R05], [R07], [R08], [R09], [R24].

Likewise, a small number of target queries need not imply low wall-clock cost: selecting a query can be harder than answering it. LearnSy is directly relevant to that distinction [R11]. Formal correctness of a witness algorithm does not validate a particular implementation's performance numbers; Manthan's current maintainers explicitly warn about reproducibility problems [R21]. These distinctions determine the evaluation below.

---

## 4. Formal mechanism and implementation specification

### 4.1 Exact core language

Implement the following pure, first-order DSL. Let `B` denote Boolean values and `W` denote unsigned bit vectors of a single width `w` within a task.

```text
W ::= input[j] | const[c]
    | bvnot(W) | bvneg(W)
    | bvadd(W,W) | bvsub(W,W) | bvand(W,W) | bvor(W,W) | bvxor(W,W)
    | shl_const(W,k) | lshr_const(W,k)
    | ite(B,W,W)
B ::= true | false | eq(W,W) | ult(W,W) | ule(W,W)
    | not(B) | and(B,B) | or(B,B) | xor(B,B)
    | ite(B,B,B)
```

Use one input for the first exhaustive fixtures and one or two for the acquisition experiments. The task declares its output type. `k` is an integer in `[0,w]`, not a bit-vector operand. For `k=w`, both shifts yield zero. Every arithmetic result is masked with `M = 2**w - 1`. Comparisons are explicitly **unsigned**. Python `~`, negation, and addition must be masked; solver overloads for signed comparisons must not be used accidentally. Boolean values and bit-vector values are distinct types, even though Python's `bool` subclasses `int`.

Exclude division, arbitrary recursion, floating point, mutable state, and variable shifts from the first version. These exclusions make semantics total and solver correspondence straightforward. They are not evidence that the broader proposal cannot use those operations.

Widths `4`, `8`, and `16` have different roles: 4-bit exhaustive validation; 8-bit finite acquisition diagnostics; 16-bit scale/transfer tests when calibrated. Do not interpret a certificate at one width as valid at another. A later 32-bit condition is justified only by measured headroom and recurrence, not by a desire for larger numbers.

The public literal alphabet is stored in each task specification. For exhaustive 4-bit fixtures it may include every literal. For larger synthesis tasks, use a task-independent corpus alphabet comprising `0`, `1`, `M`, powers of two and their complements, plus pseudorandom literals selected before generating targets. Record the alphabet generation seed and size. Use the **same alphabet for all methods**. Never supply just the target's constants to the proposed method. Parameter-transfer component tests can use all bit-vector parameters because both candidate programs already disclose their literals.

### 4.2 Syntax, types, and semantics

Represent programs as immutable hash-consed DAGs with records:

```text
Node(op, result_type, width, child_ids, literal_or_input_index)
Program(root_id, input_types, output_type, semantics_version, base_expanded_nodes)
Macro(id, typed_arguments, body, dependency_ids, provenance, expanded_cost)
```

Canonicalize only exact, safe transformations: alpha-renaming; child ordering for commutative operators; constant folding; and a separately proved rewrite set. Preserve the uncanonicalized source for auditing. Hashes include width, types, operator semantics, and library revision. Do not use floating-point hashes or Python process-randomized hashes for artifact identity; use SHA-256 of a canonical byte encoding.

A macro is a typed acyclic function over earlier macros and primitives. Full expansion must terminate, preserve semantics, and have a recorded base-primitive size. Count both stored definitions and references in storage/description metrics. A one-token macro call does not make a huge definition free.

Provide two independently written interpreters: a scalar reference interpreter and a batch interpreter. Use solver translation as a third semantic implementation. Differential tests must cover every operator exhaustively at small widths, both comparison boundaries, all shift values, nested `ite`, and overflow.

### 4.3 Forward concepts

Begin with no learned macros. Run source tasks or consume source programs generated under a declared common pretraining protocol. Feed solved-program bodies to Stitch as a reusable candidate abstraction generator [R08]. Serialize the DSL into typed S-expressions with an adapter; reject abstractions that fail type checking or expansion equivalence.

Stitch's compression score is a proposal mechanism, not the final admission objective. For each candidate macro, re-run source-validation tasks with and without it under the same search budget and expanded-size bound. Keep it only if the selected library policy improves the chosen validation cost–quality objective, or place it in a separately labeled compression-only baseline. Eliminate duplicates by proved equivalence where feasible.

For a minimal fallback when Stitch cannot be built faithfully, implement repeated-subtree abstraction and typed anti-unification. This is a reference implementation, not “Stitch reproduced.” That fallback permits correctness and component work, but blocks claims about superiority to optimized library learning until the appropriate baseline exists.

**Source access modes must be separate.** In `trace_learned`, source programs are the learner's own solved hypotheses; no gold target program is exposed. In `public_program_corpus`, explicit source programs are a common training asset for every method. The first tests autonomous acquisition; the second tests library reuse under supplied executable experience. Never pool their results or call the second unsupervised discovery. In `trace_learned`, “solved” means the source learner reached its declared operational stopping condition. It does not mean the private evaluator told it the gold target was recovered. Do not filter which source hypotheses enter the library using hidden gold correctness. Report any such oracle-curated source-corpus condition separately as supplied supervision.

### 4.4 Contrast schema discovery

Record every candidate-pair comparison attempted by the source acquisition engine, including failed separator searches and equivalent pairs. Use neither a match to the proposed library nor a favorable runtime as a logging/filtering criterion.

To mine a pair:

1. Expand macros only to the frozen comparison representation, keeping mappings back to macro boundaries. Find structurally common surrounding contexts and the positions where the two terms differ.
2. In version 1, admit **one differing subtree position** after structural alignment. Pairs with multiple independent differences go to the generic fallback; they remain in evaluation denominators.
3. Identify local alternative functions `a(z; theta)` and `b(z; theta)` with the same input/output type. Local input `z` may initially be one bit vector; later allow a tuple only after the one-input path works.
4. Replace corresponding literal roles with shared parameter variables, preserving equalities and disequalities implied by role identity. Parameterize values, not arbitrary operators. Operator replacement is a different schema.
5. Group alpha-equivalent typed pairs, with symmetric pair ordering. Repeated literal roles must share the same parameter. Never group by target ID, answer, protected test membership, or observed success.
6. Retain the occurrence paths, matched forward-macro IDs, surrounding contexts, widths, observed parameter bindings, and full provenance. Do not require a minimum recurrence number chosen after seeing test gains. Generate candidates from repeated training structure, then decide retention using validation cost.

Plain typed anti-unification and subtree alignment are established mechanisms [R09]. What is under test is which discovered contrasts are worth retaining and whether attachment to learned forward concepts changes their transfer value.

### 4.5 Capsule definition and witness learning

A capsule is

\[
K=(a,b,\tau,w,G,g,\Pi,P,S),
\]

where `a,b` are local schemas, `tau` is the signature, `w` is width, `G(theta)` is an applicability guard, `g(theta)` returns a proposed local distinguishing input, `Pi` is verification evidence, `P` is provenance, and `S` is training/validation cost statistics.

Its local correctness condition is

\[
\forall\theta:\quad G(\theta)\Rightarrow
 a(g(\theta);\theta)\ne b(g(\theta);\theta).
\tag{1}
\]

This is a guarded functional-synthesis problem, not a new logical formulation [R20], [R21], [R22]. The guard may be incomplete; a capsule need not cover every distinguishable parameter setting. It must never advertise more coverage than has been certified.

**Minimal constructor grammar.** Reuse `W` and `B` operations from Section 4.1, replacing `input[j]` with parameter variables `theta[j]`; permit `ite` and the fixed width constants `0,1,M`. No oracle calls, program-ID lookup, task-ID lookup, file access, or hidden evaluation data are legal inside a constructor. Constructors are pure total programs.

**Version-1 synthesis algorithm.** Enumerate constructors in increasing typed AST cost, tie-broken by canonical serialization. On source parameter bindings, require the candidate to produce a separating input. Do **not** require equality to a solver's arbitrary chosen witness: many valid witnesses may exist. Verify Equation (1) by asking an independent SMT backend for a parameter assignment satisfying its negation. If SAT, add that assignment to constructor training and continue. If UNSAT, store the formula, backend/version/options, result, and a replay hash, then apply the evidence-level gate in Section 10.3; a single solver log is not a checked proof certificate. If UNKNOWN, do not call the capsule certified.

Initially attempt `G=true`. On failure, enumerate guards from the Boolean parameter grammar in increasing cost, requiring nonempty source support and satisfiable `G`. Score coverage on source-validation bindings. A constant-false guard is forbidden, and a guard that merely enumerates training parameter tuples is retained only as an explicitly labeled memorization baseline. A general guard may legitimately exclude some parameter values; report both source-validation and target match coverage.

For bounded synthesis time, keep `unknown` candidates in an experimental queue rather than silently granting correctness. They may be attempted as **unverified proposals** only if every actual input is fully validated before use and are reported separately from certified capsules. The confirmatory primary method initially uses certified capsules only.

Always include cvc5 direct SyGuS generation of `g` under the identical relation and grammar as a competing constructor backend. Where the local Manthan build passes independent checking, include bit-blasted functional synthesis without restricting it to the proposed high-level constructor syntax. This avoids rewarding the proposal merely for solving an already known subproblem.

### 4.6 Context lifting: the indispensable check

Suppose a candidate pair can be represented as

\[
h(x)=C_x[a(t(x);\theta)],\qquad
h'(x)=C_x[b(t(x);\theta)].
\]

The notation `C_x` allows the common outer context also to use the original input. A local witness `z=g(theta)` is not automatically a valid full experiment. Solve

\[
V(x)\land t(x)=g(\theta)\land h(x)\ne h'(x).
\tag{2}
\]

If a local input is a tuple, replace equality by conjunction over components. If the two candidates do not share the same local-input expression, decline this version-1 match rather than make an unsound substitution.

The last conjunct is necessary: a common suffix can mask the local output difference. The middle conjunct is necessary: the new prefix may not reach the proposed local witness. A satisfiable result is independently re-evaluated by the scalar interpreter on both **complete** programs and `V` before it can leave the component.

Equation (2) may still be as hard as the original problem. Record its time separately. Common-prefix simplification, formula DAG reuse, constant propagation, and generic summaries must be available to the baselines as well. Do not attribute these engineering optimizations to learning.

If Equation (2) is UNSAT, it proves only that this local witness does not lift. It does **not** prove that the full programs are equivalent. Fall back to

\[
V(x)\land h(x)\ne h'(x).
\tag{3}
\]

Only an appropriate UNSAT result for Equation (3), with checked translation, can justify `equivalent`. Unknown and timeout remain unknown.

### 4.7 Dispatch and safe fallback

Each pair request has a total online deadline `B_pair`. The same deadline applies to the proposal and every comparator. A learned attempt consumes part of that budget; fallback does not receive a fresh full budget.

```text
separate(pair, deadline, frozen_library, baseline_dispatcher):
    start inclusive timer
    validate_types_and_semantics(pair)
    run shared cheap exact-cache / boundary / random-probe prefix
    if a valid witness is obtained: return it

    for capsule in matched_capsules_sorted_by_frozen_validation_value:
        if remaining_budget <= reserved_fallback_budget: break
        if not applicable_guard(capsule, pair.parameters): continue
        z = evaluate(capsule.constructor, pair.parameters)
        attempt solve(lifting_formula(pair, z), allocated_attempt_budget)
        log match, guard, reachability, observability, and cost
        if result is SAT and independent_full_pair_validation(result.x):
            return valid_witness(result.x)
        # local UNSAT is not global equivalence

    return baseline_dispatcher.continue_from_shared_prefix(pair, deadline)
```

The shared cheap prefix is selected on development data. It can be empty when lookup/probing adds overhead. The capsule method and baseline receive identical generic caches and canonicalization, with identical byte budgets. Report the additional capsule bytes separately and in a matched-total-memory condition.

All validation must finish before the absolute deadline for an answer to count as timely. Use a watchdog or a tested solver interruption mechanism; record wall-clock overshoot separately even when the primary score is capped. The fallback continues the shared baseline schedule with its existing prefix/cache state rather than needlessly repeating it. A ground witness or equivalence result arriving after the deadline remains a timeout for the capped endpoint.

Choose the attempt/fallback schedule on development data by minimizing validated capped runtime; freeze it. An always-try policy, an always-fallback policy, and a random/shuffled ranking are ablations. All are compared under the same total deadline. A positive component result that depends on spending more total time is not a positive result.

### 4.8 Library admission and coupling

Let `L` denote a proposed library and `C_B(task; L)` the observed capped acquisition or separator cost under a fixed budget. A cost-oriented validation objective is

\[
J_N(L)=T_{\mathrm{construct}}(L)+T_{\mathrm{load}}(L)
       +N\,\widehat{\mathbb E}_{\mathrm{validation}}[C_B(\mathrm{task};L)].
\tag{4}
\]

`N` is an explicit intended reuse horizon, not a hidden constant. Because no deployment horizon was supplied, report the **library-size/training-cost/online-cost Pareto frontier and break-even horizon**, rather than choose a favorable `N` and call it universally useful. For the first conditional-online experiment, select the lowest validation online cost subject to the calibrated memory envelope, then separately report its full training and loading costs. This separates reusable pretraining from project expenditure without making it disappear.

Coupled admission evaluates a forward macro together with contrast capsules whose schemas are exposed at its occurrences. Decoupled admission sees the same raw expanded source programs, pair logs, and total storage/training allowance, but does not use the learned macro identity as its grouping/prioritization signal. Shared anti-unification, proof checks, and generic optimizations remain identical. The comparison tests the **benefit of the learned concept boundary**, not access to different data.

A greedy admission policy is sufficient initially: propose one addition or deletion, evaluate paired validation workloads, retain the lower-cost library subject to correctness/storage constraints, and stop when the fixed candidate set has no improving move. Log every attempted move. This is a heuristic, not a claim of global optimality. Validation-selection uncertainty is resolved only by the protected test, not by declaring a tiny validation difference decisive.

### 4.9 The complete active learner

The minimal complete learner has a hypothesis generator, shared example store, forward library, pair/experiment selector, oracle interface, and output ranking. It need not include a language model or an agent framework.

Use a typed cost-ordered enumerator for the exact small reference mode. For the scalable local mode, integrate a validated optimized enumerator or cvc5 SyGuS candidate generation. cvc5 documents `checkSynth` and `checkSynthNext`; the latter requires incremental mode and immediately follows a successful synthesis call [R29]. Rebuild its synthesis state after a new example unless the exact pinned API's constraint-update behavior is tested. Rebuilding cost belongs in acquisition time.

The candidate generator interface returns all examples-consistent candidates up to a configured pool/time limit, with an explicit `complete_for_bound` flag. Cache structural candidates and safely filter by new examples. Preserve semantic alternatives.

**Critical invariant:** never merge programs only because they agree on the currently observed examples. They are supposed to agree there; their differences elsewhere are the reason to ask a question. Globally proved equivalents may be merged. A full-domain truth-table signature is safe only in a domain actually exhaustively covered. A packed representation may preserve multiple alternatives, but selecting one representative and discarding the rest is unsafe for active identification.

For the reference pool, use uniform candidate weights to avoid calling an uncalibrated ranking a posterior. Optional description-length weights `p(h) ∝ 2**(-L(h))` define a disclosed prior, not a calibrated confidence. The same weights are used across selector variants. For a candidate query `x`, score partitioning with

\[
U(x)=1-\sum_y\left(\sum_{h:h(x)=y}p(h)\right)^2.
\tag{5}
\]

This Gini disagreement score is a heuristic for the finite retained pool. Generate query proposals from deterministic high-weight pair traversal, the shared cheap probes, and the selected separator. Deduplicate valid inputs; evaluate all proposals against the same candidate pool; choose maximum `U`, with a canonical tie-break. Pair traversal order and query-proposal budget are identical in attribution comparisons. LearnSy is a distinct comparator rather than mislabeled as Equation (5).

After asking the oracle, add the answer and regenerate/filter candidates. If the pool is empty, expand search within the disclosed grammar/budget or return `model_misspecification_or_search_failure`; do not erase the inconvenient example. If all retained candidates are equivalent but search is incomplete, return the best candidate with `pool_agreement_only`, not global certainty.

The learner stops on its configured resource/query budget, true complete-bound agreement, or a declared operational criterion. **The hidden evaluator must never tell it when its program matches the gold target.** Save snapshots after each query and predetermined wall-time boundary so that the evaluator can later measure when correct behavior first appeared.

### 4.10 Minimal theoretical properties and limits

**Sound-witness proposition.** If the interpreter and solver translation implement the declared DSL, and the component emits an input only after evaluating `V(x)` and unequal full outputs, every emitted witness is valid for the supplied pair. This follows directly from the postcondition, independently of the constructor's learning procedure. A defective capsule can waste time but cannot invalidate the postcondition when this validation is mandatory.

**Finite identification proposition.** For a finite complete hypothesis set containing the deterministic target, if each query separates two remaining non-equivalent hypotheses, at least one represented functional hypothesis is eliminated per answered query. Therefore identification terminates after at most `H-1` such queries, up to functional equivalence. This is a simple finite-set bound, not an efficiency guarantee, and does not apply to an incomplete top-`H` pool. The requirements on oracle and hypothesis classes are substantive [R25].

**No information-free advantage.** A capsule computes from hypotheses and source knowledge; it does not obtain new target answers without querying. Its plausible advantage is computational amortization and inductive bias, not privileged information. An optimal baseline with the same information could choose equally informative experiments.

**Headroom calculation.** If experiment construction accounts for fraction `f` of baseline end-to-end time, accelerating only that part by factor `s` gives the idealized bound

\[
S_{\mathrm{end}}\le\frac{1}{(1-f)+f/s},
\]

before new overhead. For example, if measured `f=0.1`, even eliminating this cost entirely would give at most about `1.11×` speedup. This is an algebraic illustration, not a measurement of this project. Query-count reductions may create a second benefit; report them separately.

The search space remains combinatorial. Learned macros can increase branching, schemas can proliferate, and compiled witness functions can be larger than the original relation. Finite-width certificates do not prove arbitrary-width laws. No theorem here establishes favorable scaling, global optimality of library selection, or general intelligence.

---

## 5. Evaluation populations, baselines, and measurement boundaries

### 5.1 A risk-ordered workload hierarchy

Do not construct only examples for which a capsule is known to work. Use the following distinct populations and retain their separate identities in every report.

**P0 — exact diagnostic fixtures.** Exhaustively check small typed programs and automatically generated one-operator substitutions at width 4. Include identical programs, different syntax with equivalent behavior, valid and invalid guards, comparison boundaries, unreachable local states, suffix masking, and overflow. These fixtures validate mechanisms; they are not the primary performance population.

**P1 — constructed transfer diagnostics.** Mechanically generate typed program pairs by replacing one operator with every other operator of the same signature, replacing literals, or replacing a subtree with another well-typed subtree. Surround the differing subterm with independently sampled common prefixes and suffixes. Enumerate all permitted mutation types rather than selecting only arithmetic boundaries. Include unmodified-equivalent and masking contexts explicitly. Report the uniform generated mixture and each mechanism-defined stratum; do not select a favorable stratum after timing.

**P2 — primary synthesis-ambiguity traces.** Generate hidden target programs from the disclosed grammar, then run a frozen **baseline** active learner to collect candidate-pair requests. Record every attempted request, not just solver successes, capsule matches, or pairs known to be hard. The trace-generating learner must not use capsules. This yields an actual ambiguity workload instead of a synthetic test-maker's choice of contrasts. Freeze the generator, starting examples, query policy, trace budgets, and target distribution before the primary test. This is still synthetic program-learning evidence, not real-world deployment evidence.

**P3 — complete acquisition tasks.** On a separate target sample, compare live active learners rather than replaying their pairs. Replay controls isolate separator cost; they cannot establish how a changed selector alters later hypothesis pools. P3 is therefore necessary for H4.

**P4 — independent benchmark provenance.** Attempt a SyGuS benchmark import only after access and per-file licensing have been verified [R30]. Accept a benchmark into the active protocol only if it defines a deterministic executable reference transformation in the supported grammar, for example a universally quantified equality to a reference expression not containing the unknown synthesized function. A collection of PBE examples alone does not supply a membership oracle. General specifications with multiple valid solutions do not define a unique hidden target. Do not invent one silently. Use those tasks only in a separately labeled specification-satisfaction experiment, or exclude them with recorded reasons.

The **core PoC** consists of P0, P1, P2, and a minimal P3 integration. P4 is required before a claim about benchmark-provenance transfer beyond the project's generator; it is not required to establish a narrow component effect.

### 5.2 Sampling and leakage controls

Create disjoint namespaces `source_train`, `source_select`, `pilot`, `confirm_component`, `confirm_acquisition`, and `independent_replication`. Each has an independently generated seed manifest. Generate full target/context skeletons before filling literals. Remove duplicate full skeletons across namespaces, including duplicates exposed by alpha-renaming and the safe rewrite set. Record semantic duplicate checks at exhaustive scales; at larger scales, record the residual possibility that syntactically different programs are equivalent.

**Important distinction:** the intended transfer condition holds out **complete tasks and surrounding context skeletons**, while permitting learned **local contrast schemas** to recur. Completely unseen local schemas form a negative/coverage control; the version-1 matcher is not expected to recognize arbitrary new operators. Calling ordinary local-schema recurrence “generalization to unseen concepts” would be incorrect.

For parameter-transfer diagnostics, use disjoint source and test parameter tuples. Whole parameter values need not be disjoint in every multivariate position; declare which condition is tested. Add a stricter novel-literal condition where feasible. Certification may quantify over all parameters because operator semantics are public; it does not use hidden target labels. Disclose that this is deduction from a known DSL, not experiential proof of arbitrary real-world laws.

Group all constant variants, mutation relatives, and trace steps sharing a full target/context skeleton into a **family**. Assign the entire family to one namespace. Never split individual observations or solver calls from one family across train/test. For the primary conditional-library estimand, independently generated families are the units of replication. Variation in source libraries is a separate outer replication, not manufactured by repeating target seeds.

In the task generator, sample output type, expanded-size level, and operator categories under a declared distribution. Start with small size levels to validate the search, then choose larger levels from measured baseline coverage and cost. Do not condition task inclusion on which method wins. Remove ill-typed or invalidly constructed tasks before method timing and retain rejection counts. Constant functions and easy pairs are legitimate members of the declared mixture; do not quietly eliminate them to create headroom.

For the smallest task instances, use a complete grammar enumeration so target inclusion in the bounded hypothesis set is verified. At larger scales use constructive target generation followed by an exact syntax/expanded-size check. The target's generator trace and AST live only in the evaluator's directory.

Starting examples are sampled by a frozen common pseudorandom rule over valid inputs. They are identical across methods on a task. Do not choose them to hide exactly the motivating `<`/`<=` boundary. A separate adversarial rare-boundary condition may do that explicitly and must not replace the general mixture.

**Executable default target sampler.** For P2/P3, use exact-size typed-tree sampling from Section 4.1 rather than an unspecified “random program” routine. First select the declared width/arity/size stratum with the frozen mixture weights, and select `B` versus `W` output with equal probability among types admitting that size. Count possible ordered syntax trees `N(type,size)` by dynamic programming: for each production, sum over child-size compositions of `size-1` the product of its children's counts; terminal alternatives include every public literal and input index. Choose a production/size partition with probability proportional to that count, then recurse. Arbitrary-size integer counts and integer random draws avoid floating-point sampling overflow. At tiny sizes, compare observed samples/counts with complete enumeration.

Sample before canonicalization, then canonicalize and apply the declared cross-split root-family exclusion. Record both distributions and all rejection counts; the resulting distribution is the specified sampler conditioned on those exclusions, not uniform over semantic functions. Do not reject constant functions, solver-easy targets, or targets with no capsule match. For the primary population, validity is `V=true`; constrained validity is a separately stratified robustness condition. Width/size/arity weights are equal over feasible registered strata unless a real workload provides an independently justified frequency distribution. The pilot selects feasible size levels using baseline coverage and local cost, never capsule advantage; freeze the resulting explicit levels and weights.

Define a root family by the alpha-normalized operator/type skeleton with literal values replaced by equality-preserving parameter roles, before width-specific instantiation. Any widths, literals, repeated timings, or trace steps derived from that same root stay in its cluster and split. Store the generator algorithm/version and all rejected candidate IDs. P1 separately applies every permitted type-correct one-site mutation to sampled roots and independently sampled one-hole contexts; it does not replace the baseline-driven P2 primary.

### 5.3 Ground-truth status for pair workloads

For small domains, independently enumerate valid inputs to label each pair `satisfiable_difference` or `equivalent_on_valid_domain`. For larger domains, obtain a concrete checked witness or an independently replayed equivalence proof using evaluation-only resources. Keep these artifacts hidden from timed methods.

Pairs unresolved by the evaluation checker are retained as `truth_unresolved`, reported by family and size, and omitted only from metrics that logically require ground truth. Report the omitted fraction and its covariates. If unresolved cases materially concentrate in a difficult stratum, do not extrapolate the confirmed result to that stratum. Never use “solved by the proposed method” as a ground-truth inclusion filter.

For exact small domains, **exhaustive input enumeration is also a serious baseline**. At 8 bits with one input there are only 256 possible inputs; pretending a heavyweight solver is the only alternative would create an artificial win. At two 8-bit inputs there are 65,536 combinations. At two 16-bit inputs exhaustive enumeration is no longer a small truth table; use budgeted vectorized enumeration/probing and record the coverage limit.

### 5.4 Required component baselines

All component methods receive the same pair ASTs, validity predicate, semantics, source corpus/pair logs, online deadline, and resource accounting. None receives hidden pair labels or evaluator witnesses.

| ID | Baseline | Exact configuration and purpose | Role |
|---|---|---|---|
| B0 | Scalar and vectorized input search | Canonical input enumeration at exhaustive scales; budgeted pseudorandom/boundary probes at larger scales. Include full-program interpretation and validation. | Strong simple comparison, not merely a sanity check. |
| B1 | Z3 ground separation | Solve Equation (3) with a native supported release. Compare fresh and persistent/incremental contexts on development data. Share formula DAGs and simplification. | Mandatory current solver baseline. |
| B2 | cvc5 ground separation | Same formula, semantics, and deadline; tune only documented applicable solver settings on development data. | Mandatory independent solver baseline. |
| B3 | Shared reuse portfolio | Best validated development schedule combining B0–B2, exact canonical constraint caching, replay of prior valid inputs, incremental state, and shared common-context simplification. | Minimum competitive baseline, not a cold-solver strawman. |
| B4 | Implication-aware constraint reuse | Store checked solutions keyed by canonical conjunctions; test cached assignments against the new formula; reuse UNSAT only with an actually established implication/subset relation. Include index/check cost. Inspired by GreenTrie, not mislabeled as its full reproduction. | Excludes “better than exact caching” as the entire explanation. |
| B5 | Direct functional-synthesis reuse | Discover full-pair and local raw-syntax parameter schemas using the same anti-unifier. Ask cvc5 to synthesize witness functions for the same guarded relations. Store and validate them under the same budget. No attachment to learned forward concepts. | Mandatory strongest simpler explanation for learned constructors. |
| B6 | Verified Manthan functional-synthesis adapter | Bit-blast the same relations, synthesize witnesses, independently validate them, and measure compilation plus reuse. Use the current maintained build only after correctness/reproduction gates. | Closest additional prior-art implementation; blocking for claims against functional-synthesis learning broadly. |
| B7 | Hand-written grammar-derived test heuristics | Probe constants, adjacent values modulo width, all-zero/all-one vectors, single-bit masks, and comparison boundaries. Permit inverse affine offsets only when derived from the visible candidate syntax; publish rules before confirmation. | Tests whether a few ordinary expert rules explain the gain. |

The primary comparator `B*` is the **lowest capped-runtime validated baseline or fixed baseline portfolio selected on source-selection/development data**, including B5 and B6 when runnable. Select it without seeing protected outcomes. Report all baselines, not only the chosen comparator. Do not use the fastest baseline per protected test instance as an executable method; such an oracle portfolio is at most a labeled lower-bound diagnostic.

If B6 cannot be made valid locally, the confirmatory primary comparison may proceed against B0–B5/B7, but the result must be described as superior to that tested local set, not the strongest functional-synthesis approach in general. A failing Manthan implementation is a baseline-availability problem, not evidence that its algorithm loses. Record whether B6 is unavailable, unsupported for the encoding, invalid, or simply slow.

### 5.5 Functional-synthesis adapter details

For a local schema, define `F(theta,z) = G(theta) => (a(z;theta) != b(z;theta))`. A candidate guard must make the guarded relation realizable. For direct comparison to the proposal, give B5 the same local schema and guard as a matched-control condition as well as letting it discover its own schemas. This deliberately gives the baseline a strong advantage in isolating whether the proposed constructor-search procedure adds anything.

In cvc5, synthesize a function from parameter bit vectors to `z` with the declared constructor grammar and assert the universal relation. Also run a less restricted grammar condition when feasible, reporting representation and search costs. Keep `no_solution_in_grammar`, `unknown`, and `invalid_solution` distinct.

For Manthan, convert the finite Boolean relation to CNF using a tested Tseitin translator, with universal parameter bits followed by existential witness and gate variables in QDIMACS. Preserve the correspondence between witness output bits and original `z`; an existential gate variable must not accidentally become an input parameter. Validate the CNF against the DSL exhaustively at small widths. An encoding bug invalidates the experiment, not the solver.

The current repository emits Verilog witness artifacts and provides a checker [R21]. Initially implement a strict parser/interpreter for the **combinational subset actually emitted by the pinned build**: module declarations, input/output/wire declarations, continuous assignments, bit constants, negation, Boolean binary operations, conditional expressions, and acyclic named-wire dependencies. Reject any unrecognized construct or cycle. Map output bits using the repository's explicit variable order, never lexical guessing. Cross-check the parsed function with the supplied checker and the independent DSL relation checker. Do not execute arbitrary Verilog or shell content from a result file.

If the actual emitted syntax exceeds that subset, extend the adapter with fixture tests or use the repository's established circuit tooling to obtain a verified combinational representation. Until representation equivalence is validated, report the adapter as blocked rather than timing a substitute. This is a bounded interface task, not permission to replace Manthan with a home-grown decision tree and call it reproduced.

### 5.6 Complete-learner baselines and comparison scope

For H4, first compare live systems sharing exactly the same candidate backend and forward library: B* query construction versus capsules. This isolates the mechanism. Then compare to cvc5-based PBE acquisition with its own best validated configuration and an optimized enumerative learner such as EcoSearch when the port gate passes [R16], [R29]. All get identical starter examples and oracle rights.

Implement random valid queries and finite-pool disagreement queries as diagnostic controls. Include a LearnSy adapter for its supported overlapping domain and grammar if a faithful local port is successful [R11]. Its published build uses Linux-specific Z3/Gurobi paths; remove dependencies only when the selected solver/selector code path demonstrably does not need them. Preserve operator semantics and the original selection algorithm. If the supported task representation differs, compare on a separately declared common domain. Do not silently translate to a friendlier task for one method.

A neural-versus-non-neural **general superiority** claim is not part of the core. For predictive context, a local MLP and a tree ensemble may be added to structured-state or passive examples using the same inputs and label budget. Their losses on exact unseen bit-vector functions do not establish superiority to neural synthesis. A stronger neural program-learning comparison requires a faithful CrossBeam-like implementation and matched source training assets, or a supported higher-order/list domain with a relevant abstraction learner [R17], [R18]. CrossBeam's original CUDA/torch-scatter instructions are not proof of a native MPS training path. An unvalidated port remains a reference, not a scored weak baseline.

Do not force Popper/STEVIE into a bit-vector benchmark by a translation that removes their strengths. They become direct baselines in the explicitly relational branch in Section 12. Their prior art already constrains novelty regardless of whether that branch is run.

### 5.7 Baseline tuning and reproduction

Begin with official defaults and the documented reproduction commands on tiny examples. Verify outputs with independent interpreters, not only upstream exit codes. Then tune development settings that plausibly affect the workload: solver persistence, simplification, probe count, batch width, candidate-pool size, search cost prior, macro count, cache capacity, and per-stage schedule.

Use a shared development-instance manifest and disclose each method's tuning compute. A baseline is not considered adequately tuned while a cheap documented setting still materially improves its validation frontier. Equal numbers of trials are not intrinsically fair when one method has more knobs; report both an equal-project-compute comparison and a best-validated feasible configuration. The proposed method must not get unlimited design iteration while a baseline gets one default run.

Select between candidate configurations using paired validation cost and correctness. Keep a record of all configurations attempted. Freeze the configuration set before confirmation. Generic improvements discovered during implementation must be offered to every applicable method and may require rebuilding the freeze.

### 5.8 Primary and secondary metrics

For known non-equivalent pair instance `j` in family `i`, define

\[
T_{ijm}=\text{elapsed time until method }m\text{ emits a validated witness},
\quad C_{ijm}(B)=\min(T_{ijm},B),
\]

with `T=∞` when no witness is emitted before the common deadline. The timer begins with the canonical in-memory pair supplied to the method and includes method-specific matching, formula construction, cache lookup, probing, constructor execution, context lifting, independent validation, failed attempts, and fallback. Common corpus parsing is measured separately and included in cold/end-to-end runs.

The primary endpoint is the **family-weighted mean capped valid-witness time**, in milliseconds:

\[
R_m(B)=\frac1n\sum_{i=1}^n\frac1{|J_i|}\sum_{j\in J_i}C_{ijm}(B),
\qquad \Delta=R_{B^*}(B)-R_{\mathrm{capsule}}(B).
\tag{6}
\]

Positive `Delta` favors the capsule method. Root-family weighting prevents long traces or many constant variants from masquerading as more independent evidence. Families with no known non-equivalent pair are excluded only from Equation (6), with counts and their equivalent-pair results reported separately.

Report absolute milliseconds and relative effects with intervals; also show the full solved-by-time curve, failure/timeout counts, median and upper-tail successful latency, equivalent-pair decision cost, false-equivalence count, invalid-witness count, match coverage, lifting success, fraction of budget spent in failed matches, peak memory, disk footprint, and library construction cost. Conditional-on-success speed alone is not the primary metric.

For P3, record exact functional correctness of every saved program snapshot, target queries, time, and all failure states. A suitable separately frozen primary endpoint is family-weighted capped time to the **first correct saved snapshot**, where correctness is checked privately after the run. This measures acquisition progress without granting a stopping oracle. Also report final correct-task fraction at the common budget, capped queries to first correct snapshot, and prediction accuracy on a declared held-out input distribution. Exact correctness and average predictive accuracy are not interchangeable.

### 5.9 Full cost and practical significance

Report four cost boundaries:

1. **Hot component:** library already resident, generic method state initialized; all per-request work included.
2. **Cold component:** includes library loading, parsing/translation, and initial caches. Process startup is reported separately and in cold end-to-end measurements for both methods.
3. **Acquisition:** all candidate search, query selection, oracle execution, bookkeeping, and output creation for a new task.
4. **Lifecycle:** source generation, source solving, failed library searches, tuning, compilation/proof checks, storage, and online operation. Distinguish research-development expenditure from reproducible training expenditure.

For an online saving `s = E[T_baseline] - E[T_proposal] > 0` and extra reusable preparation cost `A`, report `N_break_even = max(0,A)/s`, with uncertainty. If `s <= 0`, there is no finite break-even under that model. If the confidence interval crosses zero, do not present a single confident break-even figure. Show a curve over reuse horizons and task-mix sensitivity.

No query price was supplied. Report measured oracle cost and an analytical cost curve `compute_time + c_query * target_queries` for explicitly labeled query-price scenarios; do not insert sleeps to manufacture an advantage. A statistically detectable microsecond effect, a large memory regression, or a favorable isolated component does not by itself justify a breakthrough claim.

---

## 6. Confirmatory design and statistical decisions

### 6.1 Separate exploration from confirmation

Use source training to construct libraries, source-selection data to tune them, and pilot data to estimate noise and determine resources/sample size. None enters the protected primary analysis. Before creating or reading protected result summaries, freeze a signed/hash-addressed `confirmatory_protocol.yaml` containing hypotheses, population generator, baseline `B*`, budgets, configuration hashes, library hashes, independent unit, metrics, failure treatment, sample size, confidence procedure, and multiplicity policy.

The coding agent may generate and execute protected tasks automatically after freezing. During execution, expose only operational diagnostics necessary to identify crashed processes, corrupted files, invalid witnesses, or resource failures. Avoid inspecting comparative performance dashboards until the frozen sample is complete. A material change after unblinding creates an exploratory amendment and requires fresh independent confirmation.

### 6.2 Pilot-calibrated size and detectable effects

The pilot must cover the predeclared structural strata and realistic cache states. Estimate between-family paired-difference variability separately from repeated-timing variability. Increase timing repetitions until timer/thermal variability is small relative to between-family variation or report that timing resolution is limiting. Do not count those repetitions as new families.

Choose the common pair deadline from the development solved-by-time curves and the intended interactive/compute regime. Report the curve rather than select a deadline at the largest apparent effect. Choose the confirmatory deadline before protected evaluation. If no deployment constraint exists, use a development-selected point at which extending the deadline changes baseline coverage enough to justify its cost, and retain nearby deadline values as secondary sensitivity analyses.

Let `s_d` be the pilot standard deviation of family-level paired capped-time differences. Before freezing, declare an absolute precision target `epsilon_ms`. Ground it in a decision: the saving that would change library admission or break-even within an explicitly considered reuse horizon. If no application horizon is available, use a declared measurement-precision target and report practical importance as unresolved. Do not invent a percentage improvement that defines success.

A planning approximation for a two-sided 95% interval half-width is

\[
n_{\mathrm{precision}}\approx (1.96s_d/\epsilon)^2.
\]

For 80% power to detect a specified paired effect `delta`, a normal approximation gives

\[
n_{\mathrm{power}}\approx ((1.96+0.84)s_d/\delta)^2.
\]

Use the pilot upper uncertainty bound on variability for conservative planning, and simulate the proposed capped/hierarchical data model to check these approximations. These formulas are planning tools, not guarantees for heavily skewed small samples. As an **unmeasured arithmetic example**, `s_d=8 ms` and `epsilon=2 ms` imply about 62 independent families for that interval precision; detecting `delta=2 ms` with the stated power implies about 126. Neither number is a mandated experiment count.

Compute the resulting total wall-time/storage forecast using Section 8. If the necessary design cannot be run locally, reduce redundant timing repetitions or concurrency first, improve measurement efficiency, or narrow the population **before** confirmation with a recorded claim change. Do not silently reduce sample size and retain the original sensitivity claim. A feasible but underpowered run is inconclusive, not negative evidence of the desired strength.

### 6.3 Analysis

For the primary test, calculate paired differences at the **family** level. Obtain a two-sided 95% confidence interval using a stratified family-cluster bootstrap, preserving all within-family instances and trace dependence. For very small family counts, report the limitation and supplement with a paired randomization/sign-based analysis where its assumptions are appropriate. A p-value alone is not the decision rule. The first analysis is conditional on the frozen source library and source corpus. Generalization over the training procedure requires independent outer source-corpus/library replications and a hierarchical analysis; changing timing seeds on one learned library does not provide that evidence.

Bootstrap computation is numerical analysis, not extra data collection. Use enough resamples that interval endpoint Monte Carlo variation is negligible relative to the requested precision; save RNG state and resampling code. For a fixed benchmark collection rather than sampled target families, report finite-benchmark effects and timing uncertainty without pretending the benchmark is a random population sample.

The primary planned comparison is one method against the development-selected `B*`. Other baselines remain visible. H3 attribution comparisons form a separate predeclared family with Holm correction if formal significance claims are made. P3 acquisition confirmation is a separate experiment with its own primary endpoint. Secondary slice plots and robustness sweeps are descriptive unless separately frozen and adjusted. Never search many widths, budgets, libraries, and task subsets and report only the best nominal p-value.

### 6.4 Decision vocabulary

* **Correctness failure:** an invalid emitted witness, false equivalence, corrupt artifact, or interpreter discrepancy. Stop scoring, fix, and rerun affected data. It is not evidence for or against the scientific mechanism.
* **Confirmed component improvement:** primary interval excludes zero in the favorable direction, all correctness gates pass, and baseline/tuning/cost controls support attribution. State the tested population and cost boundary.
* **Practically useful improvement:** the component or acquisition effect additionally gives an acceptable disclosed lifecycle tradeoff, with regressions reported. This is not implied by statistical significance.
* **Evidence against a specified advantage:** uncertainty excludes the prespecified decision-relevant effect on a valid sensitive experiment. This is local evidence, not a universal impossibility result.
* **Inconclusive:** interval too wide, resource truncation, unresolved baseline validity, inadequate recurrence, or an unsuitable proxy. Name which uncertainty prevents interpretation.
* **Mechanism works without differentiated value:** constructors transfer correctly but simpler reusable witnesses/caches do as well. Retain it as an engineering option or report a negative architectural result; do not rename it into novelty.

Use a fixed-sample confirmation. Do not extend it because the effect is “almost significant.” New experiments require a scientific amendment and an independently generated confirmation sample.

---

## 7. Small checks executed during this research

### 7.1 What was run

A pure-Python enumerator considered simple constructor expressions using parameter leaves `theta`, `delta`, `0`, and `1`, with modular `add`, `sub`, `xor`, and `and`. The task pair was

\[
h_0(x)=\mathbf1[((x+\delta)\bmod 16)<\theta],\quad
h_1(x)=\mathbf1[((x+\delta)\bmod 16)\le\theta].
\]

The code selected a constructor using 16 parameter pairs formed from `theta,delta ∈ {0,2,5,9}`. It then checked the remaining 240 pairs and the entire 4-bit parameter domain. This is a deliberately simple mechanism fixture. The program selected the term automatically, but the task family and constructor grammar were hand-designed.

### 7.2 Results

| Quantity | Observed result |
|---|---:|
| Runtime environment | Linux 6.18.44, x86-64, Python 3.13.5 |
| Constructor selected | `sub(theta,delta)` |
| Constructors examined in the fixed enumeration order | 22 |
| Source parameter pairs | 16 |
| Held-out parameter pairs | 240 |
| Successful held-out witnesses | 240 / 240 |
| Successful witnesses over all parameters | 256 / 256 |
| Naive unlifted input `x=theta` | 16 / 256 |
| Distinctions preserved after an output-erasing suffix | 0 / 256 |
| Odd thresholds unreachable after clearing the input low bit | 8 / 8 tested odd thresholds |
| Performance comparison against a solver | Not performed |
| Complete engine training or M3 compatibility test | Not performed |

The last two failure cases directly changed the design: a local certificate is insufficient without input reachability and full-output observability. A separate assertion checked that `h(x)=x` and `h'(x)=0` agree on the observed input zero but disagree at one, demonstrating why observational-equivalence merging is unsafe for the active candidate pool.

This check does not show superiority over `x=theta-delta` supplied by a hand-written heuristic, cvc5, Z3, or a functional synthesizer. It demonstrates the intended transfer equation and the necessity of its guards. Appendix A contains the complete runnable script; no conversation artifact is required to reproduce it.

### 7.3 What was not run or verified here

No target-Mac benchmark, native package installation, full repository build, complete learner, statistical performance comparison, energy measurement, or large data-generation job was executed. Public code was inspected through web-accessible files, not through a successful local clone. The research container had no working external network access; Z3 and cvc5 were not installed there. Current wheel availability is documentary evidence, not an installation test.

---

## 8. M3 Max feasibility and hardware-adaptive execution

### 8.1 Discover the actual machine before selecting workload sizes

The M3 Max family has more than one CPU/GPU and memory configuration; the largest advertised configuration is not a safe assumption [R28]. The core relies on native CPU computation. GPU availability is not a prerequisite for exact symbolic learning.

Run the following **existing macOS commands**, capture stdout/stderr, and redact personal serial numbers before sharing the manifest:

```sh
uname -m
sw_vers
sysctl -n hw.memsize
sysctl -n hw.logicalcpu
sysctl -n hw.physicalcpu
vm_stat
sysctl vm.swapusage
df -h .
system_profiler SPHardwareDataType SPDisplaysDataType
xcode-select -p
pmset -g custom
```

Record machine architecture, total physical memory in bytes, current free disk space, macOS build, CPU/GPU configuration, power source, installed developer tools, Python architecture, and whether a process is running under translation. Use native `arm64` Python and native dependencies. Do not mix x86-64 and arm64 wheels or count a translated run as a native M3 measurement. Lack of developer tools blocks source builds, not necessarily wheel-based smoke tests. Installing system tools or changing the OS requires the machine owner's approval; ordinary project-local environments do not.

### 8.2 Software and dependency strategy

| Component | Selection and evidence | Required local validation / fallback |
|---|---|---|
| Core implementation | Native CPython 3.13 is the initial candidate; NumPy for batched finite semantics; standard library for canonical ASTs, hashing, processes, JSON, and timing. Use pytest/Hypothesis for tests and a native memory-inspection package when useful. | Resolve compatible releases on the actual OS; freeze exact versions and wheel hashes. If an optional package forces a different Python minor version, isolate that adapter instead of destabilizing the core. |
| Z3 | `z3-solver` 5.1.0.0 was available at research time, with a macOS 13+ arm64 wheel [R31]. | Install in a clean native environment, run SAT/UNSAT/timeout and unsigned-bit-vector tests. Older macOS requires a validated source build or an explicitly approved alternative version; do not silently compare against an obsolete solver. |
| cvc5 | Version 1.4.2 was available at research time, with macOS arm64 wheels. Official Python documentation includes synthesis and subsequent-solution enumeration [R29]. | Test the actual wheel, bit-vector encoding, synthesis grammar, `checkSynthNext`, and resource interruption. The Python package is not assumed to install a command-line executable. Use its verified Python API. |
| Forward abstraction proposals | Stitch, pinned to an actual recorded commit; Rust or Python binding path according to local build success [R08]. | Run its examples, then verify every imported abstraction by expansion in this project's interpreter. If unavailable, a small anti-unification proposal generator is a labeled fallback; it cannot support a claim against Stitch. |
| Direct Boolean functional synthesis | Manthan's maintained native setup script has OS-specific build handling; its checker uses circuit tooling [R21b], [R21c]. | Build in an isolated adapter environment. Run upstream fixtures, independent small-domain truth checks, and the project's encoded relations. The repository's current reproducibility warning is a real gate, not a footnote [R21a]. |
| Optimized program enumeration | Reuse EcoSearch in its supported representation or make a tested semantic-preserving adapter [R16]. | Reproduce ordering and tiny exhaustive results. A port must retain the algorithm and published semantics; a custom enumerator is named separately. |
| Optional neural controls | PyTorch with native MPS support, full-precision operations appropriate to the control [R32]. | Verify every operation and backward path actually used. Compare CPU/MPS losses and gradients on a fixed fixture. Do not assume CUDA extensions run on MPS. CPU is a faithful fallback only if the full model and evaluation budget remain comparable. |

The versions above are **inspected availability snapshots, not versions installed or tested here**. At execution, first try those versions; a newer release is allowed only as a recorded update followed by all semantic/reproduction checks. Record release, source commit where relevant, downloaded artifact URL, checksum, architecture, compiler, and build flags. Do not invent a source SHA in advance.

Use an isolated project environment and a lock with exact versions and hashes. Keep a wheelhouse or other permitted local dependency archive for repeatable installation. The official pip documentation describes version pinning, hash checking, and wheelhouse-based installs [R33]. For locally built wheels, save the source revision, build log, wheel hash, and the build environment; a package version alone is insufficient. Separate optional baseline environments when dependencies conflict. No test run may install or update dependencies automatically.

For MPS measurements, synchronize before and after timed regions using the supported synchronization API [R32]. Initially disable silent CPU fallback so an unsupported operation is visible. A deliberately enabled fallback must be declared and its transfer/CPU costs measured. MPS is not used to claim acceleration of Z3, cvc5, or arbitrary Python syntax search. Do not change precision or quantize only one baseline to make it fit; record a separate matched comparison if such a change is scientifically appropriate.

### 8.3 Licensing and permitted assets

The core can operate entirely on newly generated task specifications and code written for this project. Public assets improve external validity but do not remove licensing obligations.

| Asset | Inspected status | Action before redistribution/use |
|---|---|---|
| Z3 package | MIT license metadata [R31]. | Preserve notice and record version; inspect bundled distributions as needed. |
| cvc5 | BSD-style core license with separately described bundled dependencies [R29c]. | Retain applicable notices; do not label every dependency BSD by association. |
| Stitch | MIT repository license [R08b]. | Preserve attribution and pinned source provenance. |
| Manthan | MIT root license; separately built dependencies have their own conditions [R21b], [R21d]. | Audit downloaded dependencies and notices; do not assume the root license covers all bundled tools. |
| CrossBeam | Apache-2.0 repository [R17b]. | Preserve notices; inspect dataset/training-asset terms separately. |
| SyGuS benchmark collection | Repository structure and README inspected; a single blanket redistribution permission was not established [R30]. | Inspect the license and provenance of each selected subcollection/file. Exclude unresolved assets from redistributed packages. A downloadable benchmark is not automatically openly licensed. |
| LearnSy / other research artifacts | Public source inspection does not itself establish all use/redistribution conditions [R11b], [R16b]. | Resolve terms before packaging. Record an unavailable-license gate rather than quietly copying the artifact. |

The final report may describe excluded sources and cite their papers without redistributing their contents. A locally generated benchmark is not a substitute for an external-evidence claim; report the narrower evidence when an external asset is unavailable.

### 8.4 Memory model and calibration

Track both process-specific allocation and system pressure. A useful accounting model is

\[
M_{\mathrm{peak}}\approx M_{\mathrm{runtime}}+M_{\mathrm{AST}}+M_{\mathrm{frontier}}
+M_{\mathrm{eval\ cache}}+M_{\mathrm{indices}}+M_{\mathrm{library}}
+\sum_{k=1}^{p}M_{\mathrm{solver},k}+M_{\mathrm{queues}}+M_{\mathrm{optional\ neural}}.
\]

This is an accounting guide, not an additive identity for overlapping shared-memory measurements. In particular, do not add GPU unified-memory allocation to a system-used-memory figure that already includes it. Record solver child-process memory as well as the Python parent.

**Illustrative storage calculation, not a measurement:** a compact million-node DAG at 192 bytes/node uses 192 MB; a `1,000,000 × 128` result matrix stored as unsigned 16-bit values uses 256 MB; an assumed 64 MB index and 19.2 MB frontier bring those four terms to about 531 MB. Python object graphs, allocator fragmentation, extra copies, solver state, and wider result types can multiply this. Measure actual bytes/node, bytes/candidate, and cache growth using fixture sizes that successively double. Do not extrapolate from the raw array alone.

Never materialize every assignment of a multi-input 16-bit task. A single 16-bit input has 65,536 values; two have over four billion. Enumerate in bounded batches where exact enumeration is justified, or use solvers with explicit time/memory budgets. Candidate execution caches need capacity limits and an eviction policy shared across applicable methods.

Start with one solver worker and one task at a time. Establish an engineering memory ceiling from actual idle application use plus a safety reserve, initially the larger of 4 GiB or 10% of physical memory in addition to observed non-project use. This is a conservative safety setting, not an experimental result or a mandatory performance threshold. Adjust it from measured memory pressure and documented user constraints. Stop growth before sustained swapping or critical pressure. Do not intentionally benchmark through out-of-memory failure.

Calibrate one stage at a time: interpreter throughput; candidate expansion rate; solver resident-set growth; constructor synthesis; library indexing; source-trace storage; and optional training. Increase concurrency only after measuring useful throughput and peak memory, and only up to actual physical/resource limits. Use single-task execution for confirmatory latency unless throughput contention is itself the declared workload. Reduce concurrency and batch size before reducing the hypothesis language or model capacity. Any latter change affects the scientific claim and requires a manifest amendment.

### 8.5 Workload estimates from pilot measurements

| Stage | Resource driver | Forecast to calculate before expansion |
|---|---|---|
| Exact fixtures | Domain cardinality × program pairs × interpreter implementations | Measured evaluations/second and bytes/batch; all intended small-domain truth checks must fit. |
| Source search and trace generation | Tasks × candidates evaluated × mean interpretation/search cost | Pilot distribution by structural family, including failures and timeouts; not just successful median time. |
| Forward-library construction | Source AST count/size, proposal arity, accepted library size | Measured construction peak and total time; store compressed corpus size and indexing overhead. |
| Contrast discovery and synthesis | Candidate alignments × parameter assignments × constructor/verifier cost | Separate match rate, constructor attempts, verifier calls, and accepted/rejected entries. |
| Component confirmation | Families × pairs/family × methods × timing repetitions × common deadline | Upper-bound runtime plus setup and cold-start costs; tighten with measured solved-by-time distributions. |
| Live acquisition | Tasks × acquisition deadline × methods, plus evaluation of snapshots | Measure evaluator overhead separately so private scoring is not charged as learner computation. |
| Optional baseline ports/training | Source compilation, parameter/optimizer states, native kernels, source examples | Profile before large downloads or training. Exclude invalid ports rather than silently simplify them. |

For scale intuition only, 126 families × 10 pairs × 8 methods × 3 timing repetitions × a 0.5-second cap yields a cap-based total of 15,120 seconds, or 4.2 hours, **before** setup, preparation, cold starts, or live acquisition. Actual instance counts, deadlines, and repetitions must come from Section 6; this arithmetic is not a promised duration or a prescribed experiment. Calibration can justify substantially smaller or larger work.

Estimate storage from a pilot's bytes per trace/result/checkpoint, include failed-run logs and independent replication, and maintain free-space headroom for atomic checkpoints. Retain essential raw evidence; compress verbose solver logs or regenerate deterministic intermediates only when their hashes and reconstruction procedure are preserved. Do not fill the disk to complete a predetermined run count.

### 8.6 Timing and power boundaries

Use a monotonic high-resolution timer, preferably `time.perf_counter_ns()`, and absolute deadlines. Report elapsed wall time and CPU time separately. Standardize garbage-collection policy, interpreter warm-up, formula caches, process startup, and solver persistence per declared timing regime. A cold run must actually clear/recreate the relevant state; do not call an already warm process cold. A hot run must not include hidden cross-test answers that a real deployed process would not possess.

Randomize or counterbalance method order within family blocks, run on stable power settings, record competing load and thermal/power conditions where available, and repeat enough to quantify variability. Do not launch all competitors concurrently and infer intrinsic latency from contention. OS sleep or a process crash invalidates the affected timing block; retain it as failed and rerun the whole paired block under the same frozen configuration, with the reason recorded.

No physical-energy claim is part of the core. Elapsed time and CPU time are not joules. Optional system power telemetry requires a clearly defined measurement boundary, baseline subtraction, uncertainty, and appropriate permissions; otherwise omit energy results.

### 8.7 Checkpointing and recovery

Checkpoints contain source/data/library/configuration hashes, RNG states, task queue, candidate DAG and preserved alternatives, current observations, source-library selection state, and completed-result identifiers. Write to a temporary file, flush, atomically rename, and verify a checksum on load. Keep the previous valid checkpoint until the new one is verified.

Do not pickle opaque solver state as the only recovery path. Rebuild solver contexts from canonical constraints and recorded versions, then revalidate cached witnesses. Reuse an equivalence certificate only under identical semantics, width, validity predicate, and expanded macro hashes. An interrupted timed call is not resumed halfway and scored as if uninterrupted; rerun the paired timing block. An interrupted untimed training stage may resume if its logical state is reproduced exactly. If deterministic recovery cannot be achieved, restart that stage with the same input manifest and record the duplicate expenditure.

---

## 9. Risk-ordered work packages and decision gates

**Command convention:** every `python -m rce ...` command below is a **proposed project command that the coding agent must implement**. It is not an existing package interface. Upstream commands explicitly identified as existing in Section 8 or the references are different. Commands produce machine-readable artifacts and a short decision record rather than relying on chat memory.

### WP00 — Prerequisites, rights, and native dependency checks

**Objective/risk:** establish that exact semantics and relevant baselines can be executed legally and faithfully on the actual laptop. **Inputs/dependencies:** this plan, permission to create a project-local environment, no prior experiment results.

**Actions:** implement `rce doctor`; capture Section 8 discovery; resolve and lock core dependencies; run Z3 and cvc5 SAT/UNSAT/timeout/bit-vector/SyGuS fixtures. Inspect asset licenses. Create isolated baseline environments. Test Manthan's documented native build and small upstream examples, then the independent checker; record its maintained warning and exact revision. Test Stitch expansion and available optimized enumeration. Do not begin a broad port or download a training corpus before a small supported path succeeds.

**Files/commands:** `src/rce/doctor.py`, `tests/test_dependencies.py`, `manifests/hardware.json`, `manifests/dependencies.json`, `manifests/licenses.csv`; `python -m rce doctor --out artifacts/prerequisites`.

**Outputs/resources:** versions, architecture, RAM/disk bytes, raw build/check logs, status per dependency, actual peak memory for each smoke test. **Acceptance:** native core interpreter plus two independent validated solver encodings; a clear available/unavailable/invalid status for each additional comparator. **Branches:** fix encoding failures before science; isolate dependency conflicts; if only one solver works, use exhaustive small-domain truth checking but block larger cross-solver verification claims. If a strong extra comparator is unavailable, apply the restricted claim in Section 5.4 rather than declare victory over it. Resource/permission blockers are recorded as such.

### WP01 — Exact domain, invariants, and transfer counterexamples

**Objective/risk:** prevent semantic or context-transfer errors from masquerading as learning. **Entry:** WP00 core works. **Inputs:** grammar and capsule equations in Section 4; Appendix A.

**Actions:** implement typed immutable ASTs, scalar and batched evaluation, canonical serialization, Z3/cvc5 translators, validity checks, and deterministic generation. Reproduce Appendix A exactly. Add exhaustive tests for overflow/underflow, unsigned comparisons, Boolean/word separation, shifts at zero/width, input arity, empty-validity domains, context masking, unreachable local witnesses, and multiple differing sites. Test parameter renaming and macro expansion. Construct proof-evidence records as in Section 10.3.

**Files/commands:** `src/rce/dsl.py`, `semantics.py`, `solvers/`, `tests/test_semantics.py`, `tests/test_contexts.py`; `python -m rce fixtures --config configs/fixtures.yaml`.

**Outputs/resources:** exhaustive truth tables and discrepancy counts, machine-readable expected/actual assertions, bytes/node and evaluation throughput. **Acceptance:** zero discrepancies over the frozen fixture set; masking/unreachability produce fallback or non-answer, never a false equivalence. **Branches:** any discrepancy blocks later packages; a grammar operation lacking exact support is removed only through an explicit grammar/version amendment applied to all methods. The fixture passing proves correctness only within its tested scope.

### WP02 — Strong separator baselines and a complete minimal learner

**Objective/risk:** ensure the comparison is competitive and produce real ambiguities before designing a large library system. **Entry:** WP01 passes. **Inputs:** shared DSL, fixtures, baseline source revisions.

**Actions:** implement B0–B5/B7 and the validated B6 adapter; use independent interpreters to check all returned witnesses. Implement the minimal PBE candidate generator, observation update, fixed disagreement-query policy, oracle boundary, and snapshots from Section 4.9. Add an optimized enumeration adapter where valid. Demonstrate whole tasks on small finite examples without a learned contrast library. Reproduce relevant upstream examples before reporting a port as faithful. Record baseline performance frontiers and generic optimizations; share every applicable optimization with the proposal.

**Files/commands:** `baselines/`, `src/rce/synthesis.py`, `oracle.py`, `tests/test_baselines.py`, `tests/test_active_loop.py`; `python -m rce reproduce --manifest manifests/baselines.yaml`; `python -m rce acquire --config configs/dev_baseline.yaml`.

**Outputs/resources:** checked witness/equivalence/non-answer results; candidate pool and query traces; exact-match snapshots; preparation/online cost records. Calibrate solver/candidate memory and timeout handling. **Acceptance:** baselines produce valid results, candidate updates preserve all alternatives needed for the declared search, and the learner receives no private target information. **Branches:** a slow correct baseline is tuned, not discarded; an invalid port is blocked, not scored; if exact tiny tasks cannot be recovered, diagnose coverage versus implementation before adding capsules.

### WP03 — Source population, recurrence, and headroom

**Objective/risk:** determine whether the proposed repetition exists outside the motivating fixture and can matter end to end. **Entry:** WP02 generates valid complete-task traces. **Inputs:** development task generator and source-only seeds.

**Actions:** generate the P1 structural diagnostic population and P2 ambiguities from baseline-driven acquisition. Record every attempted separator call, including unmatched and timed-out pairs. Build source-training and source-selection splits by the specified skeleton/group rules. Measure single-site alignment rate, recurring local schemas, distinct contexts per schema, source-learning coverage, fraction of acquisition spent separating, and cost of ordinary cache/heuristic hits. Inspect artifacts for accidental source/target leakage and special-case DSL design.

**Files/commands:** `src/rce/data/generate.py`, `traces.py`, `splits.py`; `python -m rce generate --config configs/source.yaml`; `python -m rce profile --manifest manifests/source.json`.

**Outputs/resources:** provenance/split manifests, raw ambiguity traces, recurrence by stratum, Amdahl/headroom and preparation-cost estimates. Grow source collection only while it changes the uncertainty about recurrence/coverage or the stable source-library frontier enough to justify cost. **Acceptance:** the population is defined independently of capsule hits, traces are valid, and a measured headroom statement is possible. **Branches:** low recurrence on a valid intended population is evidence against the proposed transfer value there; report it. An unsuitable generator may be replaced only with a substantive workflow justification and fresh splits. High recurrence but trivial baseline separation suggests a narrow/negative component result, not a reason to secretly remove easy cases.

### WP04 — Minimum contrast learner and safe reuse

**Objective/risk:** implement the indispensable contribution without a full architecture. **Entry:** WP01–WP03 pass; source corpus is frozen for this development cycle. **Inputs:** observed candidate pairs, shared anti-unifier, constructor grammar.

**Actions:** mine single-site schemas; enumerate/synthesize constructors and guards; apply exhaustive or solver-based verification with evidence labels; store capsules; implement full-context lifting, independent validation, absolute deadlines, and B* fallback. Train B5 on the same schema/guard assets. Start with true guards and one differing subtree; add restricted nontrivial guards only when true-guard failures are an observed material limitation. Guard synthesis and rejected candidates count as preparation cost.

**Files/commands:** `src/rce/contrast/{align,construct,verify,lift}.py`, `src/rce/library/registry.py`, `tests/test_capsules.py`; `python -m rce fit-contrasts --config configs/contrast_dev.yaml`.

**Outputs/resources:** capsule registry with provenance/evidence, constructor/guard complexity, match/validation/fallback counts, training and online memory/time. **Acceptance:** zero invalid emitted witnesses, no local-UNSAT-to-global-equivalence mistake, and deterministic replay. **Branches:** if constructors cannot be verified, keep them as untrusted proposals with mandatory online validation and label that distinct mode; do not assert certificates. If direct functional synthesis is equally good, preserve it as the likely simpler solution and continue only to test the separate coupling hypothesis or document the result.

### WP05 — Forward concepts, coupled/decoupled selection, and pilot freeze

**Objective/risk:** test the potentially differentiated concept-coupling mechanism and select a sensitive feasible confirmation. **Entry:** correct WP04 pipeline and measured costs. **Inputs:** source-training/selection data only.

**Actions:** obtain forward abstractions from Stitch or the explicitly labeled fallback; validate expansions. Build coupled and decoupled libraries with the same proposal pool, source access, training-cost accounting, and storage constraints. Select entries by the lifecycle objective and report the reuse-horizon frontier. Tune all baselines fairly. Run the pilot, estimate family/timing variance, deadline sensitivity, detectable effects, and total memory/time. Select `B*` on development only. Freeze the final library, grammar, population generator, sample size, budgets, statistical procedure, and amendments.

**Files/commands:** `src/rce/library/{forward,select}.py`, `analysis/pilot.py`, `configs/confirmatory_protocol.yaml`; `python -m rce pilot --config configs/pilot.yaml`; `python -m rce freeze --protocol configs/confirmatory_protocol.yaml`.

**Outputs/resources:** paired pilot results, all tuning attempts, Pareto/break-even plots or tables, sensitivity calculation, signed/hash-addressed freeze manifest. **Acceptance:** all blocking fields resolved; no protected data used; planned precision can be supported or its local limitation explicitly stated. **Branches:** no measurable saving after strong baselines yields a negative/engineering finding and may justify a smaller confirmatory equivalence/bound study, not a guaranteed-win redesign. Memory problems reduce concurrency first; population changes require an amendment. A pilot is never relabeled confirmation.

### WP06 — Protected component confirmation

**Objective/risk:** test H2 without adaptive selection or hidden overhead. **Entry:** WP05 frozen protocol and complete correct harness; protected component targets not inspected. **Inputs:** immutable protocol, baseline/library hashes, private generator seeds.

**Actions:** generate P2 confirmation families; obtain truth labels independently of tested methods; execute randomized paired timing blocks and all registered methods; retain unmatched pairs and non-answers. Monitor only operational failures until completion. Run the frozen family-level analysis and report every baseline, correctness outcome, cap curve, and cost boundary.

**Files/commands:** `src/rce/bench/component.py`, `analysis/confirm.py`; `python -m rce run --protocol configs/confirmatory_protocol.yaml --phase component`; `python -m rce analyze --run RUN_ID`.

**Outputs/resources:** raw JSONL, block logs, family-level data, intervals, unresolved truth-label fractions, preparation/online totals, failed-run manifest. Resource ceiling comes from the pilot, not a target p-value. **Acceptance:** completed registered sample, intact blinding/freeze, valid results, interpretable uncertainty. **Branches:** classify confirmed benefit, bounded negative, inconclusive, or invalid measurement using Section 6.4. Correctness fixes invalidate affected runs and require a documented fresh confirmation when exposed performance could influence design.

### WP07 — Attribution and complete acquisition

**Objective/risk:** separate genuine concept/experiment coupling from caching and determine whether the component matters to learning. **Entry:** correct component system; an interesting positive, bounded negative, or unexplained effect that an attribution experiment can resolve. A positive p-value is not the sole entry criterion. **Inputs:** new protected acquisition families and predeclared ablation protocol.

**Actions:** run Section 10's essential matched controls and the forward/contrast factorial design. For H4, hold candidate search, forward library, examples, and total time constant while replacing the separator; then compare independently tuned complete baselines. Keep target-aware equivalence evaluation private and offline. Freeze one acquisition primary endpoint, such as exact-recovery-by-budget or capped time to first privately verified correct snapshot, before execution. Report both compute and target-query effects.

**Files/commands:** `src/rce/bench/acquisition.py`, `configs/attribution.yaml`, `configs/acquisition_protocol.yaml`; `python -m rce run --protocol configs/acquisition_protocol.yaml --phase acquisition`.

**Outputs/resources:** per-task query/compute trajectories, exact-recovery results, interaction/coupling estimates, all regressions. Pilot this experiment separately. **Acceptance:** attribution boundaries are respected and end-to-end costs close. **Branches:** component-only benefit is reported as component-only; no advantage over an unlinked contrast library rejects the new-unit interpretation on this population; worse queries despite faster computation is an explicit tradeoff, not an unqualified win.

### WP08 — Boundary conditions and externally grounded transfer

**Objective/risk:** determine how narrow the result is. **Entry:** a correct mechanism and a specific uncertainty from preceding results. **Inputs:** predeclared stress tests and licensed external assets where available.

**Actions:** run width/context/recurrence and validity stress tests; import eligible SyGuS tasks through the exact-function criteria of Section 5.1; alternatively implement the separately scoped relational world in Section 12 when the hypothesis concerns structured world rules. Confirm that external tasks were not used to tune the initial library. Select only scientifically motivated branches, not every neighboring architecture.

**Files/commands:** `src/rce/data/import_sygus.py`, `configs/robustness.yaml`, optional `worlds/`; `python -m rce stress --config configs/robustness.yaml`.

**Outputs/resources:** coverage and excluded-asset records, native/translated semantics checks, stratified transfer/failure results. Calibrate new domain sizes before confirmation. **Acceptance:** domain changes and claims are explicit; no external source or unobserved target rule leaks to the learner. **Branches:** synthetic-only success stays synthetic-only; unavailable licenses block external redistribution; lack of raw perception remains a limitation, not an implementation bug.

### WP09 — Independent replication, adversarial audit, and final report

**Objective/risk:** make the finding reproducible without this conversation and retain unfavorable outcomes. **Entry:** a frozen interpretable result or a documented stopping condition. **Inputs:** saved source revision, locks, manifests, raw data, decision log.

**Actions:** rebuild in a clean native environment; use independent root families/seeds and fresh timing blocks for any repeated scientific claim; reproduce correctness and the frozen analysis. Test interrupted recovery and hash mismatches. Conduct the three audits in Section 13, record defects and fixes, and generate an evidence-bound report with all methods and failed runs. A second execution by the same agent is replication, not independent human peer review.

**Files/commands:** `reports/final.md`, `state/decisions.md`, `state/amendments.jsonl`, `reproduce.sh`; `python -m rce audit --manifest manifests/final.json`; `python -m rce report --manifest manifests/final.json`.

**Outputs/resources:** complete reproduction manifest, result tables, uncertainty, resource totals, novelty classification, continuation/stop decision, residual risks. **Acceptance:** a fresh checkout plus permitted assets can reconstruct the reported result; unresolved blockers are visible. **Branches:** nonreplication triggers diagnosis of variance, drift, hidden state, or invalid inference before scaling. Negative or inconclusive results are finished research outcomes when their evidence and limitations are fully recorded.

---

## 10. Attribution, robustness, and validation of the harness

### 10.1 Essential mechanism controls

Run the following controls on the same family splits, source information, and resource boundaries. These are planned experiments, not findings.

| Control | What changes / what stays fixed | Interpretation |
|---|---|---|
| No learned contrast library | Same forward macros, search backend, query policy, generic caches, and deadline; use B*. | Isolates the added experiment-reuse component. |
| Ground-witness memory | Replay previously useful concrete inputs, with identical validity/output checks and index budget. | Separates parameterized transfer from remembering good test values. |
| Hand-written boundary probes | Use B7's frozen grammar-derived probes and visible syntax. | Tests whether the mechanism merely rediscovers a cheap standard heuristic. |
| Direct functional synthesis | Supply the same discovered schema/guard to cvc5 and validated Manthan, then use the same lifting and validation code. | Isolates constructor search from already established witness synthesis. |
| Decoupled raw-syntax contrasts | Same source pairs, proposal pool, anti-unifier, memory/training budget; no learned forward-concept identity. | Tests the most differentiated claim: forward concept boundaries improve discovery or selection. |
| Shuffled attachment | Permute capsule-to-forward-concept labels while preserving available capsules and measured lookup opportunities. Never change their executable semantics. | Detects whether concept attachment is doing useful work rather than serving as a name. |
| Compression-only forward library | Select abstractions by compression, not acquisition cost, with measured preparation and storage. | Tests the selection objective; does not make cost-aware selection novel by itself. |
| Shared-optimization control | Give exact caching, safe rewrites, incremental solvers, batched interpretation, and context simplification to all applicable methods. | A gain erased here is an implementation optimization, not evidence for a new learning mechanism. |

For a forward-library factor `F ∈ {0,1}` and contrast-library factor `C ∈ {0,1}`, run a matched 2×2 design. Let `Y_FC` be a lower-is-better cost. The interaction

\[
I=(Y_{10}-Y_{11})-(Y_{00}-Y_{01})
\]

measures whether contrasts help more when forward concepts are present. It is not by itself proof that the capsule abstraction is novel. Control for source access, expanded program capacity, candidate ordering, and memory. If forward macros alter the search trace, report that as part of the factorial system effect; separately use frozen common pair traces to isolate lookup/lifting effects.

### 10.2 Boundary conditions and adversarial cases

**Recurrence and novelty of contexts.** Vary recurrence using generator probabilities frozen before evaluation, not by cherry-picking matching pairs. Include exact-repeat tasks, new constants with a known schema, new common prefixes/suffixes, new operator schemas, and no-source-overlap tasks. Distinguish generalization over parameters from generalization to a previously unseen operator relation. A library may reasonably fail on new schemas; that is its coverage boundary.

**Context reachability and observability.** Include non-injective prefixes, invalid local witnesses, constant/masking suffixes, nested conditionals whose branch guard changes relevance, multiple differing subterms, and empty input-validity sets. A separating local value may be unreachable; a reachable local distinction may not affect the full output. Generic fallback must receive its remaining budget. Local failure never proves global equivalence.

**Representation confounds.** Rename inputs, reorder commutative operands, introduce semantically redundant expressions, change macro factoring, and preserve the full function while changing syntax. Match safe canonicalization across methods. If semantic equivalence recognition dominates runtime, report that bottleneck rather than assume an e-graph solves it for free.

**Width and arithmetic.** Relearn or reverify at each width. Stress wrap-around, all-one values, high-bit unsigned comparisons, and shift boundaries. A 4-bit pattern need not preserve the same useful search cost at 16 bits. Validate full type/width metadata in every cache key.

**Library growth and interference.** Plot hit rate, false proposal rate, lookup time, validation time, memory, and amortized preparation against stored entries and source diversity. More concepts can enlarge search and slow retrieval. Include deletion/reselection and record whether the reported policy discards useful rare schemas. Do not assess memory by macro count alone.

**Noisy or changing observations.** The core assumes a deterministic total oracle. Injected contradictory labels should produce `inconsistent_observations`, not silently revise a logically correct rule to fit an error. A separate noisy experiment may use repeated oracle measurements, an explicit noise model, and probabilistic scoring, but requires a new hypothesis/protocol and loses the core's exact-identification interpretation. After deliberate target drift, old target answers cannot be treated as current truth; reset or time-index them under a declared protocol. A source test constructor is not an old answer.

**Misspecification and budget censoring.** Include targets outside the grammar and search bounds, and cases with no valid input. Return search/misspecification uncertainty rather than claim the world violates logic. Report all timeout and unknown rates by family. Do not count a timeout as an equivalence proof or as a successful zero-cost abstention.

### 10.3 Verification evidence is not a binary “proved” flag

Store one of the following evidence levels, with the exact claim and trusted components:

- `exhaustive_checked`: the finite declared domain was completely enumerated by an independent interpreter; store domain cardinality and enumeration hash.
- `proof_checked`: a solver-produced proof/certificate was accepted by a separately identified checker for the exact formula; store both artifacts and checker version.
- `cross_solver_checked`: independent supported encodings/backends agree on the relevant UNSAT/equivalence claim, with successful small-domain differential tests. This increases confidence but is not an independently checked formal proof object.
- `solver_reported`: one validated solver reports UNSAT; retain the formula, options, revision, and result, but do not describe the log as a proof certificate.
- `unverified_proposal`: no accepted universal correctness evidence; only individually validated emitted inputs are safe to use.

For the first confirmatory **certified-library mode**, require `exhaustive_checked`, `proof_checked`, or `cross_solver_checked` evidence according to the frozen protocol. The exact word “certified” in this plan denotes this declared operational gate, not a claim that every entry has a machine-checked mathematical proof object. Prefer exhaustive checks for small parameter domains. For larger domains without a proof checker, disclose the solver trust boundary. A separate single-solver mode can be explored but must be labeled accordingly.

Every actual emitted witness still passes full-program interpretation and validity checks. An equivalence result, unlike a witness, cannot be verified by evaluating just one input; its evidence level and complete domain/formula matter. If universal checks cannot be completed locally, the method may still safely emit individually checked proposals, but the universal-reuse claim remains unverified.

### 10.4 Test the evaluator itself

Before unblinding, run an A/A comparison of identical implementations under different method IDs with randomized order. The expected systematic difference is zero; uncertainty must reflect observed timing noise. Inject a known extra computation or a calibrated test-only delay into either side to verify that the harness detects differences in both directions and includes the correct timing boundary. These injected fixtures are not scientific method results.

Use deliberately incorrect witnesses, stale width hashes, expired deadlines, invalid input predicates, and false-equivalence records to check that the harness rejects them. Create two candidates agreeing on all observed examples but disagreeing elsewhere; verify neither disappears through an observational-equivalence shortcut. Add a trap target ID and private evaluator field that raise an error if accessible through the learner interface. This is a software isolation test, not a formal operating-system security proof.

Check that reordering family rows, duplicating timing repetitions, or changing JSON file order does not change the independent sample count. Check interval code on constant effects, zero effects, censored data, unequal cluster sizes, and intentionally dependent traces. Verify Holm adjustment against a hand-computed small example. Retain all failed records, and ensure report generation does not silently filter them.

Interrupt a development run before, during, and after atomic checkpoint creation. Verify deterministic logical reconstruction, correct rerun accounting, and refusal to load mismatched source/library/semantics hashes. Recovered untimed training and uninterrupted training must produce equivalent recorded state under the declared determinism policy.

---

## 11. Coding-agent handoff: repository, contracts, configuration, and records

### 11.1 Minimal repository

```text
refutable-concepts/
  README.md                         # purpose, environment, exact reproduction steps
  research_poc_plan.md               # this handoff, retained verbatim with amendments
  pyproject.toml
  requirements.lock                 # actual resolved versions and hashes
  configs/
    fixtures.yaml
    source.yaml
    contrast_dev.yaml
    pilot.yaml
    confirmatory_protocol.yaml
    attribution.yaml
    acquisition_protocol.yaml
  src/rce/
    __init__.py
    __main__.py                      # proposed CLI entry point
    doctor.py
    dsl.py
    semantics.py
    synthesis.py
    oracle.py
    contracts.py
    solvers/                        # independent Z3 and cvc5 translators/adapters
    contrast/                       # align, construct, verify, lift
    library/                        # forward proposals, selection, indexing
    data/                           # generation, splits, traces, import_sygus
    bench/                          # component, acquisition, timing, recovery
    analysis/                       # pilot, fixed confirmation, report
  baselines/                        # adapters; do not vendor unlicensed sources
  tests/
    test_semantics.py
    test_contexts.py
    test_baselines.py
    test_active_loop.py
    test_capsules.py
    test_statistics.py
    test_recovery.py
  manifests/                        # hardware, software, licenses, data, freezes
  artifacts/
    source/                         # immutable source corpora and libraries
    runs/RUN_ID/                    # raw outputs, checksums, logs, checkpoints
  state/
    progress.md
    decisions.md
    amendments.jsonl
  reports/
    final.md
  reproduce.sh                      # implemented only after commands are validated
```

Keep the implementation modular but small. No distributed scheduler, vector database, agent-orchestration service, or elaborate web UI is required. Add machinery only for an observed correctness/reproducibility/resource need.

### 11.2 Shared interfaces

The following is an interface sketch to implement with checked dataclasses/protocols; it is **not** a claim that these classes already exist. Canonical serialization must avoid pickling arbitrary executable objects from downloaded assets.

```python
from typing import Literal, Protocol, Sequence

EvidenceLevel = Literal[
    "exhaustive_checked", "proof_checked", "cross_solver_checked",
    "solver_reported", "unverified_proposal"
]
ResultStatus = Literal[
    "witness", "equivalent", "unknown", "timeout", "unsupported",
    "invalid_result", "resource_exhausted", "internal_error"
]

class Separator(Protocol):
    def prepare(self, source: "SourceBundle", budget: "ResourceBudget") -> "ModelArtifact":
        """Source-only preparation; return hashes and all preparation costs."""
        ...

    def separate(self, pair: "PairSpec", deadline_ns: int) -> "WitnessResult":
        """Absolute monotonic deadline, inclusive of lookup/validation/fallback."""
        ...

class CandidateGenerator(Protocol):
    def reset(self, public_task: "PublicTask", library: "ForwardLibrary") -> None: ...
    def add_observation(self, x: tuple[int, ...], y: int | bool) -> None: ...
    def candidates(self, budget: "ResourceBudget") -> "CandidateBatch":
        """Includes complete_for_bound, bound definition, and search accounting."""
        ...

class Oracle(Protocol):
    def query(self, x: tuple[int, ...]) -> "OracleReply":
        """Checks validity; returns only the answer and permitted cost metadata."""
        ...

class ActiveLearner(Protocol):
    def reset(self, task: "PublicTask", models: "PreparedModels") -> None: ...
    def update(self, observation: "Observation") -> None: ...
    def propose_query(self, deadline_ns: int) -> "QueryProposal": ...
    def snapshot(self) -> "LearnerSnapshot": ...
```

Define the referenced records as follows:

| Record | Required fields and invariants |
|---|---|
| `PublicTask` | Task ID, input/output types, width, public grammar/literals, `V`, starter examples, permitted budgets. **No** target AST, hidden truth labels, target-family answer hints, or evaluator callback. |
| `PairSpec` | Two canonical expanded program roots, validity predicate, semantics/library hashes, input/output types. Caller-side trace metadata is retained for analysis but not exposed as predictive features. |
| `ResourceBudget` | Absolute deadline or declared stage seconds, memory ceiling bytes, permitted query count where applicable, candidate/proposal bound, and budget origin (pilot/protocol). Zero and unlimited are represented distinctly. |
| `WitnessResult` | Status, optional input tuple, elapsed/CPU nanoseconds, phase timing map, method/config hashes, validation result, evidence level for equivalence, reason code, attempted capsule IDs. A witness without an independently validated postcondition is invalid. |
| `CandidateBatch` | Canonical program IDs, weights, complete-for-bound flag, exact bound/grammar, preserved alternatives, construction/filter cost. Never infer completeness from a full returned pool. |
| `SourceBundle` | Split/provenance manifest, common source assets, schema pairs, permitted solved programs, mode `trace_learned` or `public_program_corpus`, no protected examples. |
| `CapsuleRecord` | Local schemas, parameter order and equality roles, constructor/guard ASTs, evidence, source occurrence IDs, forward links, width/semantics, preparation cost, validation coverage, bytes on disk/in memory. |
| `LearnerSnapshot` | Current ranked executable hypotheses, observations/query count, elapsed acquisition time, resource status, declared confidence/completeness. No private correctness flag. |

Validate inputs at every adapter boundary. Booleans are serialized as JSON booleans; words as nonnegative integers with explicit width; canonical byte order is fixed. Malformed records yield typed errors and logs rather than coerced semantics. Separate `target not in searched bound`, `no candidate found`, and `inconsistent observations` where the available evidence permits; otherwise return an explicitly ambiguous reason.

### 11.3 Configuration conventions

Use YAML or JSON with a versioned schema. A **development template**, not an executable confirmatory default, can be:

```yaml
schema_version: 1
experiment_id: null                 # required before any run
phase: pilot
semantics_version: bv-pure-v1
source_mode: trace_learned
population_manifest: null           # required immutable path + SHA-256
widths: [4, 8, 16]                  # roles defined in this plan; not pooled blindly
input_arities: [1, 2]
method_manifest: null
library_manifest: null
resource:
  memory_ceiling_bytes: null        # WP00/WP05 measurements
  workers: 1
  pair_deadline_ms: null            # development coverage/cost decision
  acquisition_deadline_s: null
  source_training_budget_s: null
analysis:
  independent_unit: root_family
  primary_metric: family_mean_capped_valid_witness_ms
  family_count: null                # pilot-calibrated fixed sample
  timing_repetitions: null          # pilot-calibrated measurement variability
  precision_target_ms: null
  confidence_level: 0.95
  bootstrap_seed: null
freeze:
  protocol_sha256: null
  source_revision: null
  dependency_lock_sha256: null
```

`null` means **blocking unresolved input**, never “pick a convenient default.” Fixture configurations are separate and completely specified: width 4, the Appendix A parameter set, the stated interpreter tests, and known expected outputs. No confirmatory run starts until every applicable field is populated and validated. Unsupported options are errors, not ignored keys. Quantities have units in names; seeds have named purposes rather than one undocumented global integer.

### 11.4 Run and event schemas

Every run has `run.json` with: run ID; UTC timestamp; local time zone; phase; source revision and dirty-tree diff hash; OS/hardware manifest; dependency lock; task/source/library/config hashes; seeds and RNG algorithms; method versions; timing regime; power/load notes; start/end state; exit reason; resource limits; and links to raw files.

Store `results.jsonl` with at least:

```text
run_id, phase, method_id, config_hash, root_family_id, task_id, pair_id,
structural_stratum, width_bits, input_arity, split, timing_block, repetition,
truth_status, truth_evidence_level, outcome_status, reason_code,
wall_ns, cpu_ns, deadline_ns, peak_rss_bytes, solver_child_peak_bytes,
witness, witness_valid, emitted_equivalence, equivalence_evidence_level,
lookup_ns, constructor_ns, lifting_ns, validation_ns, fallback_ns,
cache_hits, capsule_hits, capsules_attempted, target_queries,
source_model_hash, observation_hash, checkpoint_hash
```

Phase counters must be nonoverlapping or explicitly marked inclusive; their sum must reconcile with the measured boundary up to recorded overhead. `null` means not measured/not applicable, not zero. `truth_status` and private correctness are filled by the evaluator **after** learner output is committed. The learner never reads the augmented result file.

Store construction and training expenditures separately by asset ID, with source inputs, rejected candidates, verifier outcomes, total time, peak memory, and bytes. Do not charge shared preparation twice in lifecycle totals or omit method-specific preparation. Record wall-clock costs of failed attempts and tuning even when they are not part of a deployment inference timer.

Maintain `events.jsonl` for queries, candidate updates, timeouts, restarts, invalid outputs, and checkpoints. Preserve raw solver statuses and original error messages. A summarized `unknown` rate without the underlying reasons is insufficient for diagnosis.

### 11.5 Reporting and execution authority

The coding agent may choose routine non-destructive details such as internal function names, compact data containers, test organization, and equivalent build tooling, while preserving the specified contracts and evidence. Changes to hypotheses, population, target access, grammar, baseline algorithm, cost boundary, primary endpoint, exclusion rules, or statistical procedure require a dated methodological amendment. After protected results are visible, such changes cannot be presented as preregistered.

After every work package, update `state/progress.md` with completed artifacts, hashes, unresolved blockers, next executable action, and entry condition. Add a decision record containing evidence, alternatives considered, selected action, and which scientific claim it changes. Do not rely on session memory. Resumption begins by validating these manifests and the latest checkpoint, not by guessing what a previous agent intended.

The final report must show all registered methods, population counts, exclusions, failures, uncertainty, absolute and relative effects where defined, runtime/memory/query tradeoffs, reusable preparation costs, and practical break-even sensitivity. Include an explicit table separating inspected literature, checks executed, measured target-hardware results, deductions, and unresolved hypotheses. Preserve unfavorable results and publish the source/analysis needed to reproduce them within asset licenses.

---

## 12. Outcome interpretation, escalation, and evidence-triggered alternatives

### 12.1 What each outcome would justify

| Result | Defensible conclusion | Next action |
|---|---|---|
| Valid constructors, no speed advantage over direct reusable functional synthesis | The representation works, but the new constructor/search claim is unsupported. | Prefer the simpler backend; test coupling only if its uncertainty remains material. Report replication/engineering, not a new principle. |
| Component improvement, no acquisition improvement | A useful test-generation optimization may exist; synthesis or querying dominates the full system. | Use measured Amdahl/break-even analysis to identify an appropriate workflow. Do not claim improved intelligence. |
| Coupled and decoupled libraries tie within a sensitive interval | Learned concept attachment has no established value in this population. | Retain a modular unlinked contrast cache or stop the capsule-as-new-unit claim. |
| Faster acquisition but more queries | A compute/query tradeoff, not unqualified dominance. | Report the cost curve and applicable query-price regime; verify that extra target information does not explain the entire effect. |
| Better synthetic results, no external transfer | Evidence for a controlled-domain mechanism only. | Investigate representation/task mismatch on new development data; require fresh external confirmation for any broadened claim. |
| Clear gains after strong controls and independent replication | A reproducible local contribution with a stated domain and resource boundary. | Expand along measured recurrence/context complexity and test broader transfer. Novelty classification still depends on the exact comparison to prior art. |
| Tiny, unstable, or underpowered result | Insufficient evidence for a useful advantage. | Improve measurement or state the unresolved bound; never extend a frozen sample merely to cross significance. |
| Invalid outputs or unreproducible state | Implementation or measurement failure. | Repair and revalidate before scientific interpretation. |

### 12.2 Conditions for scale escalation

Increase program/context complexity only when correct smaller runs show that useful contrasts recur, context lifting remains feasible, and the baseline cost component being attacked grows materially. Measure how library size and verifier cost scale; do not extrapolate a constant-time lookup story from a small table while ignoring context solving.

A negative tiny-domain result can be insensitive because brute force is nearly free, source concepts do not repeat, or the target language cannot express the intended reuse. A larger test is justified by a measured mechanism-specific reason, such as rising baseline separator cost with stable verified schema coverage. It is not justified by the generic statement that “intelligence appears at scale.” Conversely, repeated failure on well-covered, correctly measured, adequately sensitive populations is real evidence against the proposed practical advantage there.

### 12.3 If the dominant bottleneck is elsewhere

If forward search rather than experiment construction dominates, the evidence-triggered pivot is to **cost-aware forward abstraction selection and optimized enumeration**, reusing Stitch/EcoSearch and comparing against the cost-oriented prior art [R08], [R16], [R24]. That is a changed contribution, not success of H2. Preserve the contrast result as a negative or limited component finding.

If actual target-query cost dominates and construction is cheap, prioritize a faithful grammar-aware query selector such as LearnSy on a compatible domain [R11]. A more informative question may matter more than a faster witness. This creates a separate active-learning hypothesis and protocol.

If generic functional synthesis produces excellent reusable local witnesses, use that existing tool inside the engine rather than rebuilding it to protect the project's name. The remaining research question is which relations to learn and reuse, not how to claim invention of Skolem functions.

### 12.4 Optional relational-world branch preserving the original motivation

This branch is not required for the core component claim. Enter it when the specific remaining question is whether explicit rule/test reuse transfers to structured interactive worlds rather than pure program tasks.

Implement a **fully specified symbolic simulator**, not raw perception. Use a one-dimensional ring of eight positions with one mover and one blocker, a direction in `{-1,+1}`, and a binary observable mover attribute. Initial positions are distinct. An action asks for one transition. Targets differ by explicit hidden rule choices: movement in the supplied direction versus opposite direction; blocking versus passing; and an attribute-controlled choice between those alternatives. The target generator, not the learner, composes these primitive dynamics. All methods receive the same public object/state/action schema and validity constraints, not the target's rule identity. Start with deterministic wrap-around semantics; a later bounded-line/clamping variant is a separately declared shift.

The transition oracle returns the complete next symbolic state. Querying an invalid state is rejected for every method. Train on some compositions and evaluate unseen compositions and attribute/direction remappings. Do not claim that learning the visible object schema has occurred; it is supplied.

A small faithful comparison includes an exact enumerative/solver learner, compatible Popper/STEVIE-style relational learners where their official builds pass, a tuned tree ensemble on identical symbolic features, and a small neural predictor with a calibrated CPU/MPS path. Fix operator/background-predicate availability and count source tasks and oracle calls. Do not give the concept engine `blocked` as an oracle feature while requiring other models to infer it. Either derive it from the same positions for everyone or require everyone to infer it.

Primary outcomes are exact transition recovery over the finite valid state/action domain and query/compute cost on held-out rule compositions. Add spurious attribute correlations to training and break them in testing as a disclosed compositional/causal diagnostic. This is still a simulator result, not evidence of raw visual perception, physical embodiment, general causal discovery, or real-world deployment value. Any extension beyond this exact specification needs its own source-informed design and audit.

---

## 13. Audit record, changes made, and residual risks

The following are **self-audits of this research and plan**, not independent peer review or claims that the proposed implementation has run. Concrete defects found during the research changed the design.

### 13.1 Scientific and prior-art audit

| Challenge / defect found | Revision in this plan | Residual risk / acceptance condition |
|---|---|---|
| The original hypothesis–test–revision loop already exists. | Replaced the broad invention claim with a narrow cross-task cost/transfer question; added direct OGIS, ILP, abstraction, and automata prior art. | Novelty remains unresolved; cite an empirical or engineering contribution when that is all the evidence supports. |
| A learned parameterized diagnostic input is a Skolem function, not a new primitive. | Added cvc5 functional synthesis and verified Manthan as essential competitors; matched schema/guard inputs and full reuse costs. | A direct functional-synthesis baseline may erase the gain. That is a valid result. |
| Test-generator synthesis and compositional reuse have close existing equivalents. | Added FlashMeta, compositional dynamic testing, adaptive test reuse, and Clouseau; restricted the possible difference to discovery/coupling/selection and measured transfer. | Unindexed or differently named work may cover even that combination. No absence-of-prior-art proof is claimed. |
| Selecting abstractions for search cost is itself not safely new. | Added recent cost-oriented abstraction-selection evidence and classified it as preliminary where appropriate. | Further matching prior art may change the novelty category without invalidating a useful experiment. |
| No empirically established application bottleneck or query price. | Added source-trace profiling, Amdahl bounds, lifecycle/break-even curves, and explicit conditional value. | A valid speedup may have negligible practical impact. |
| Symbolic inputs could conceal the hardest part of learning. | Declared the supplied DSL/object representation and prohibited perception/general-intelligence claims. | Representation learning remains an untested long-term issue. |

### 13.2 Experimental-validity audit

| Challenge / defect found | Revision | Residual risk / check |
|---|---|---|
| The 4-bit fixture makes exhaustive probing a formidable baseline. | Mandatory batched enumeration at small widths, not just cold SMT comparisons. | Larger-domain gains must include their verification/search costs. |
| Hand-designed threshold examples could guarantee a win. | Primary population comes from baseline-generated ambiguity traces, including unmatched cases; threshold fixture is correctness-only. | Source generator remains synthetic and may still favor recurrent algebraic structure; report strata and external transfer separately. |
| Local witness validity does not imply full-context usefulness. | Added reachability and full-output separation, generic fallback, and explicit masking/unreachable fixtures verified in the executed check. | Lifting may consume all the alleged saving. |
| Observational-equivalence pruning can destroy the active version space. | Require proved full equivalence or preserved alternatives; add a concrete counterexample test. | Optimized third-party enumerators may need careful adapters to avoid hidden pruning. |
| Extra fallback time or free preparation could create a false gain. | One absolute deadline, inclusive phase timers, separate preparation ledger and lifecycle totals. | High-resolution timing and shared-cache boundaries still require local calibration. |
| Source/target schema splits could either leak the task or remove the very recurrence being tested. | Hold out full task/context skeletons while declaring recurring local schemas; label unseen-schema tests separately. | Parameter transfer is not discovery of an unseen operation. |
| Target-aware stopping would advantage the proposed learner. | Only private offline snapshot scoring; identical learner-accessible stopping rules and oracle rights. | Process/interface tests must confirm no private fields flow back. |
| Trace calls and timing repetitions are dependent. | Family-level primary endpoint and cluster analysis; pilot-driven fixed sample and distinct replication. | Too few root families gives weak population evidence regardless of raw call count. |
| A favorable primary metric could conceal query/memory regressions. | Require secondary guardrails, full frontiers, all baselines, and qualified outcome vocabulary. | Practical importance remains application-dependent. |

### 13.3 Implementation and feasibility audit

| Challenge / defect found | Revision | Residual risk / blocking gate |
|---|---|---|
| Advertised Apple Silicon support does not prove required paths work. | Native API/semantics/build tests before dependent work; optional neural ports isolated. | Only execution on the actual laptop establishes compatibility. |
| A contemporary closest baseline has an explicit implementation/reproduction warning. | Manthan's independent verification and tiny reproduction are a gate; an invalid/unavailable baseline is not scored as losing. | Broad superiority to functional-synthesis learning remains blocked when that comparator is unvalidated. |
| A solver UNSAT log is not a checked proof object. | Evidence levels distinguish exhaustive, checked proof, cross-solver, single-solver, and unverified claims. | Solver trust remains in large-domain results lacking a proof checker. |
| Unknown memory could make a nominally local plan impossible. | Actual hardware discovery, measured bytes/work unit, bounded caches, single-worker start, and adaptive concurrency. | Grammar/capacity reductions change the population and require amendments. |
| Public assets may not be redistributable. | Per-asset license manifests and generated-core fallback; no blanket SyGuS permission assumed. | External-validity claims may remain blocked. |
| Baseline APIs or output encodings could be invented by a handoff. | Project commands marked proposed; actual interfaces checked locally; strict Manthan circuit parser/reproduction gate. | Adapter work is real expenditure and may reveal unsupported subsets. |
| An underspecified task sampler or source “solved” filter could introduce hidden supervision. | Added exact-size typed-tree sampling, recorded rejection conditioning, and an explicit prohibition on gold-correctness filtering in `trace_learned`. | Generator bias remains a declared synthetic limitation; outer source-library replication is separate. |
| Timer placement and duplicated fallback work could distort costs. | Start timing before method-side validation, require completion before the absolute deadline, continue the shared baseline prefix state, and log overshoot. | Native interruption and timing calibration still need target-hardware tests. |
| Checkpoint recovery can reuse stale semantic assumptions. | Hash width/grammar/library/validity/source, rebuild solver state, revalidate cache, rerun interrupted timing blocks. | Full recovery tests remain to be executed. |
| The plan could depend on intermediate chat artifacts. | Complete grammar, equations, interfaces, decision rules, bibliography, and executed-check script are embedded here. | The coding agent still must perform the clearly labeled local tests and future experiments. |

### 13.4 Final adversarial pass and unresolved issues

Additional adversarial scenarios to retain are a correct-but-slow verifier, a guard covering only source coincidences, a hidden manual inverse in the anti-unifier, a library hit rate calculated after dropping unmatched cases, an existential/universal variable-order error in QDIMACS, solver `unknown` counted as UNSAT, and multiple observationally identical hypotheses silently collapsed by a borrowed enumeration library. These can create false claims even when aggregate accuracy looks good; the preceding gates explicitly test or expose them.

A low-memory machine, unavailable strong baseline, unusable external license, or unsupported native operation does not become evidence against the mathematical idea. It limits the runnable claim. Conversely, a hypothetical future scale or unavailable comparator does not excuse a measured lack of advantage over strong valid local methods. The final outcome must distinguish those cases.

No complete engine, target-hardware benchmark, baseline port, statistical study, or independent review has been executed during this planning task. The checked finite example supports only its stated algebraic observations. The central research question—whether learned, context-aware experimental reuse provides a reproducible advantage beyond established alternatives—remains open and directly testable by this plan.

---

## 14. References and search-coverage record

### 14.1 How to read the references

References identify inspected evidence, not experiments reproduced during this assignment. `Paper text` means relevant sections were opened and read, not that every appendix or implementation was audited. `Abstract/metadata` marks a more limited inspection. Public repositories were inspected through their README and selected source files; no successful local clone/build is claimed. Versioned arXiv links and DOIs are preferred for papers. Repository default branches and “latest” documentation are mutable: the coding agent must replace them with actual commit/release hashes in the execution manifest.

The bracketed IDs in the plan resolve to the primary source or a source bundle below. Multiple IDs in one bracket refer to multiple entries; they are not a claim that every cited source establishes every proposed design choice. Equations and protocols explicitly introduced as this project's design are deductions/proposals rather than reported literature results.

**R01 — Susmit Jha, Sumit Gulwani, Sanjit A. Seshia, and Ashish Tiwari. _Oracle-Guided Component-Based Program Synthesis_. ICSE 2010.** DOI: [10.1145/1806799.1806833](https://doi.org/10.1145/1806799.1806833). [Berkeley report record](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2010/EECS-2010-15.html). Inspected report abstract/metadata; formulation also traced through later interactive-synthesis literature. Establishes the oracle-guided candidate-disambiguation precedent. Component language and oracle assumptions constrain applicability; no original performance numbers are imported.

**R02 — Vu Le, Daniel Perelman, Oleksandr Polozov, Mohammad Raza, Abhishek Udupa, and Sumit Gulwani. _Interactive Program Synthesis_. 2017.** [arXiv:1703.03539v1](https://arxiv.org/html/1703.03539v1). Paper text inspected for incremental synthesis, feedback, and question generation. Supports the prior-art boundary around the learning loop, not a claim of neuron-free general intelligence.

**R03 — Oleksandr Polozov and Sumit Gulwani. _FlashMeta: A Framework for Inductive Program Synthesis_. OOPSLA 2015.** DOI: [10.1145/2814270.2814310](https://doi.org/10.1145/2814270.2814310). [Author-hosted paper](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/12/oopsla15-pbe.pdf). Relevant paper text inspected. Operator-specific witness semantics inform the overlap with backward decomposition; they are not automatically learned contrast capsules or a drop-in bit-vector implementation.

**R04 — Andrew Cropper and Rolf Morel. _Learning Programs by Learning from Failures_. Machine Learning 110, 801–856, 2021.** DOI: [10.1007/s10994-020-05934-z](https://doi.org/10.1007/s10994-020-05934-z). [Paper](https://andrewcropper.com/pubs/popper.pdf); [Popper repository](https://github.com/logic-and-learning-lab/Popper). Paper text and first-page rendering, plus current README, inspected. Establishes generate/test/constrain learning; original and current capabilities must be distinguished. A relational-language baseline, not an assumed faithful bit-vector port.

**R05 — Céline Hocquette, Sebastijan Dumančić, and Andrew Cropper. _Learning Logic Programs by Discovering Higher-Order Abstractions_. IJCAI 2024.** [Proceedings paper](https://www.ijcai.org/proceedings/2024/0379.pdf). Paper text and page rendering inspected. STEVIE provides a close non-neural abstraction-discovery precedent. Its logic/background-language conditions differ from this finite pure-function PoC.

**R06 — Andrew Cropper. _Playgol: Learning Programs through Play_. 2019.** [arXiv:1904.08993v2](https://arxiv.org/html/1904.08993v2). Relevant paper text inspected. Supports the established role of self-generated tasks and reusable learned programs; the plan accordingly charges and shares source experience.

**R07 — Kevin Ellis, Catherine Wong, Maxwell Nye, Mathias Sablé-Meyer, Lucas Morales, Luke Hewitt, Luc Cary, Armando Solar-Lezama, and Joshua B. Tenenbaum. _DreamCoder: Bootstrapping Inductive Program Synthesis with Wake-Sleep Library Learning_. PLDI 2021.** DOI: [10.1145/3453483.3454080](https://doi.org/10.1145/3453483.3454080). [Paper](https://www.neurosymbolic.org/papers/EllisWNSMHCST21.pdf); [implementation](https://github.com/ellisk42/ec). Paper text, recognition/library discussion, and page rendering inspected. Main architecture combines neural guidance and library learning; ablations matter. Published results are broader context, not a matched local score.

**R08 — Matthew Bowers, Theo X. Olausson, Catherine Wong, Gabriel Grand, Joshua B. Tenenbaum, Kevin Ellis, and Armando Solar-Lezama. _Top-Down Synthesis for Library Learning_. POPL 2023.** DOI: [10.1145/3571234](https://doi.org/10.1145/3571234). [arXiv:2211.16605v2](https://arxiv.org/html/2211.16605v2); [paper PDF](https://mlb2251.github.io/stitch.pdf). Relevant text and page rendering inspected. Reusable compression-oriented abstraction search, not proof that compression minimizes future acquisition time.

**R08b — Stitch maintainers. _Stitch repository and binding tutorial_. Inspected 14 October 2026.** [Repository](https://github.com/mlb2251/stitch); [binding tutorial](https://stitch-bindings.readthedocs.io/en/stable/intro/tutorial.html); [Cargo manifest](https://raw.githubusercontent.com/mlb2251/stitch/master/Cargo.toml). README, API examples, manifest, and license status inspected. Native build and adapter correctness remain local gates.

**R09 — David Cao, Rose Kunkel, Chandrakana Nandi, Max Willsey, Zachary Tatlock, and Nadia Polikarpova. _babble: Learning Better Abstractions with E-Graphs and Anti-Unification_. POPL 2023.** DOI: [10.1145/3571207](https://doi.org/10.1145/3571207). [arXiv:2212.04596](https://arxiv.org/abs/2212.04596). Abstract/metadata inspected. Establishes close equivalence-aware abstraction machinery; no uninspected implementation behavior is assumed.

**R10 — Xinyu Wang, Greg Anderson, Isil Dillig, and K. L. McMillan. _Learning Abstractions for Program Synthesis_. 2018.** [arXiv:1804.04152](https://arxiv.org/abs/1804.04152). Abstract/metadata inspected. Predicate/transformer abstraction learning limits broad claims about inventing constraints; the specific abstract-interpretation machinery is not required by this PoC.

**R11 — Ruyi Ji, Chaozhe Kong, Yingfei Xiong, and Zhenjiang Hu. _Improving Oracle-Guided Inductive Synthesis by Efficient Question Selection_. OOPSLA 2023, article 103.** DOI: [10.1145/3586055](https://doi.org/10.1145/3586055). [Author-hosted paper](https://xiongyingfei.github.io/papers/OOPSLA23.pdf). Relevant paper text, including the efficient selection construction, inspected. LearnSy is a substantive question-selection competitor; its published task assumptions and algorithm must survive any port.

**R11b — LearnSy authors. _LearnSy implementation_. Inspected 14 October 2026.** [Repository](https://github.com/jiry17/LearnSy); [CMakeLists.txt](https://raw.githubusercontent.com/jiry17/LearnSy/master/CMakeLists.txt). README and build configuration inspected. Hard-coded platform/library assumptions inform the native-port gate. Public access is not a license determination; performance was not reproduced.

**R12 — Carlos Diego N. Damasceno, Mohammad Reza Mousavi, and Adenilso da Silva Simão. _Learning to Reuse: Adaptive Model Learning for Evolving Systems_. IFM 2019.** [Publisher record](https://link.springer.com/chapter/10.1007/978-3-030-34968-4_8). Abstract and reference context inspected. Reuse in evolving automata models overlaps with remembered distinguishing information, while stateful assumptions differ from pure functions.

**R13 — Tiago Ferreira, Gerco van Heerdt, and Alexandra Silva. _Tree-Based Adaptive Model Learning_. 2022.** [arXiv:2209.00122v2](https://arxiv.org/html/2209.00122v2). Relevant paper text inspected. Supports the prior-art treatment of observation/test reuse; its oracle structure is not silently supplied to this learner.

**R14 — Xiangyang Jia, Carlo Ghezzi, and Shi Ying. _Enhancing Reuse of Constraint Solutions to Improve Symbolic Execution_. 2015.** DOI: [10.1145/2771783.2771806](https://doi.org/10.1145/2771783.2771806). [arXiv:1501.07174](https://arxiv.org/abs/1501.07174); [GreenTrie repository](https://github.com/jiaxy/GreenTrie). Abstract and repository description inspected. Motivates implication-aware reuse beyond exact caching. The proposed B4 is disclosed as inspired, not a full artifact reproduction.

**R15 — Patrice Godefroid. _Compositional Dynamic Test Generation_. POPL 2007.** DOI: [10.1145/1190215.1190226](https://doi.org/10.1145/1190215.1190226). [Author-hosted paper](https://patricegodefroid.github.io/public_psfiles/popl2007.pdf). Relevant paper text inspected. Function summaries and compositional testing are direct conceptual predecessors to context-aware reuse; the present cost/transfer question is narrower than inventing compositional tests.

**R16 — Théo Matricon, Nathanaël Fijalkow, and Guillaume Lagarde. _EcoSearch: A Constant-Delay Best-First Search Algorithm for Program Synthesis_. arXiv 2024; AAAI 2025 publication.** [arXiv:2412.17330](https://arxiv.org/html/2412.17330); [publication DOI 10.1609/aaai.v39i18.34139](https://doi.org/10.1609/aaai.v39i18.34139). Paper text inspected. Algorithm and reported CPU setting motivate a stronger enumerator, but do not establish M3 throughput.

**R16b — SynthesisLab. _DeepSynth2, EcoSearch artifact branch_. Inspected 14 October 2026.** [Repository branch](https://github.com/SynthesisLab/DeepSynth2/tree/eco_search_aaai); [README](https://raw.githubusercontent.com/SynthesisLab/DeepSynth2/eco_search_aaai/README.md). Artifact instructions inspected. Resolve actual dependencies/license and native semantics before a scored port.

**R17 — Kensen Shi, Hanjun Dai, Kevin Ellis, and Charles Sutton. _CrossBeam: Learning to Search in Bottom-Up Program Synthesis_. ICLR 2022.** [arXiv:2203.10452](https://arxiv.org/abs/2203.10452). Paper record and implementation context inspected. A stronger neural synthesis reference than a generic MLP, with task-specific training and search assumptions.

**R17b — Google Research. _CrossBeam implementation_. Inspected 14 October 2026.** [Repository](https://github.com/google-research/crossbeam); [README](https://raw.githubusercontent.com/google-research/crossbeam/main/README.md). README, dependencies, and Apache-2.0 status inspected. Original CUDA/torch-scatter setup is a port risk; no MPS training success is claimed.

**R18 — Janis Zenkner, Lukas Dierkes, Tobias Sesterhenn, and colleagues. _AbstractBeam: Enhancing Bottom-Up Program Synthesis using Library Learning_. 2024.** [arXiv:2405.17514](https://arxiv.org/abs/2405.17514); [paper text](https://arxiv.org/html/2405.17514). Relevant text inspected. Neural search with library learning supplies a contemporary comparison scope, not a directly comparable bit-vector benchmark number.

**R19 — Celeste Barnaby, Qiaochu Chen, Ramya Ramalingam, Osbert Bastani, and Isil Dillig. _Active Learning for Neurosymbolic Program Synthesis_. 2025.** DOI: [10.1145/3763102](https://doi.org/10.1145/3763102). [arXiv:2508.15750v2](https://arxiv.org/html/2508.15750v2). Relevant paper text and metadata inspected. SmartLabel addresses uncertain neural components; use it to delimit the deterministic core and inform an explicitly changed noisy setting.

**R20 — Priyanka Golia, Subhajit Roy, and Kuldeep S. Meel. _Manthan: A Data-Driven Approach for Boolean Function Synthesis_. CAV 2020.** [arXiv:2005.06922v1](https://arxiv.org/html/2005.06922v1). Paper text inspected. Non-neural learned Skolem functions with verification/repair are decisive prior art for the witness-learning subproblem. This plan imports no headline speedup from the paper.

**R21 — Priyanka Golia, Friedrich Slivovsky, Subhajit Roy, and Kuldeep S. Meel. _Engineering an Efficient Boolean Functional Synthesis Engine_. ICCAD 2021.** [arXiv:2108.05717](https://arxiv.org/abs/2108.05717). Abstract/metadata and maintained implementation inspected. Establishes the later Manthan engine as an important comparator; implementation correctness and reproducibility must be assessed separately from algorithm descriptions.

**R21a — Manthan maintainers. _Current README, native setup instructions, and reproducibility notice_. Inspected 14 October 2026.** [Repository](https://github.com/meelgroup/manthan). The notice dated **3 October 2026** reports implementation/reproduction problems. This is why baseline validity is a gate, not why the algorithm is excluded or assumed defeated.

**R21b — Manthan maintainers. _scripts/setup.sh_. Inspected 14 October 2026.** [Source](https://raw.githubusercontent.com/meelgroup/manthan/master/scripts/setup.sh). Native OS dispatch and dependency-build handling inspected. Supports trying a native build; it is not evidence that the research environment or target Mac completed one.

**R21c — Manthan maintainers. _checkSkolem.py_. Inspected 14 October 2026.** [Source](https://raw.githubusercontent.com/meelgroup/manthan/master/checkSkolem.py). QDIMACS/Verilog checking interface and circuit-tool use inspected. Informs the adapter contract; independent DSL checking is still required.

**R21d — Manthan maintainers. _LICENSE_. Inspected 14 October 2026.** [License file](https://raw.githubusercontent.com/meelgroup/manthan/master/LICENSE). MIT root license inspected; separately fetched dependencies require their own notices.

**R22 — Yi Lin, Lucas M. Tabajara, and Moshe Y. Vardi. _Dynamic Programming for Symbolic Boolean Realizability and Synthesis_. CAV 2024.** [arXiv:2405.07975v4](https://arxiv.org/html/2405.07975v4). Relevant paper text and metadata inspected. BDD/decomposition-based realizability and witness construction provide another established route. The core's local comparison set does not automatically include a reproduced implementation of every symbolic-synthesis approach.

**R23 — Zhe Zhou, Ankush Desai, Benjamin Delaware, and Suresh Jagannathan. _Trace-Guided Synthesis of Effectful Test Generators_. 2026.** DOI: [10.1145/3808264](https://doi.org/10.1145/3808264). [arXiv:2604.04345v1](https://arxiv.org/html/2604.04345v1). Paper text and metadata inspected. Clouseau broadens the relevant test-generator prior art to effectful systems; it uses a different specification/trace setup from the proposed source-ambiguity transfer experiment.

**R24 — Adam Cuculich. _On the Value of Abstractions: Abstraction Selection in Bounded Program Synthesis_. 20 September 2026, preprint.** [Author page](https://adamcuculich.com/research/on-the-value-of-abstractions/); [paper](https://adamcuculich.com/research/on-the-value-of-abstractions/on-the-value-of-abstractions.pdf); DOI [10.13140/RG.2.2.17726.24647](https://doi.org/10.13140/RG.2.2.17726.24647). Page and relevant text inspected. Non-peer-reviewed and unreplicated here. It narrows novelty claims about selecting abstractions for search value; its empirical claims are not adopted as established facts.

**R25 — Susmit Jha and Sanjit A. Seshia. _A Theory of Formal Synthesis via Inductive Learning_. 2017 version.** [arXiv:1505.03953v4](https://arxiv.org/html/1505.03953v4). Relevant text inspected. Oracle capabilities and hypothesis-class assumptions are central; the elementary finite-set bound in Section 4.10 is this plan's stated deduction under explicit assumptions.

**R26 — Ole-Christoffer Granmo. _The Tsetlin Machine—A Game Theoretic Bandit Driven Approach to Optimal Pattern Recognition with Propositional Logic_. 2018, later revisions.** [arXiv:1804.01508](https://arxiv.org/abs/1804.01508). Abstract/metadata inspected. Clause/automaton learning is an established alternative computational-learning unit. No broad superiority or optimization guarantee from the title is asserted here.

**R27 — Richard E. Lenski, Charles Ofria, Robert T. Pennock, and Christoph Adami. _The Evolutionary Origin of Complex Features_. Nature 423, 139–144, 2003.** DOI: [10.1038/nature01568](https://doi.org/10.1038/nature01568). [Publisher page](https://www.nature.com/articles/nature01568). Accessible abstract/metadata inspected; full paywalled text not claimed inspected. Digital-organism functionality is a limited precedent, not evidence for general intelligence or efficient individual adaptation.

**R28 — Apple. _MacBook Pro (16-inch, November 2023): Technical Specifications_. Inspected 14 October 2026.** [Official specifications](https://support.apple.com/en-us/117737). Configuration information inspected. Used only to avoid assuming a single M3 Max memory/core configuration; actual machine discovery remains necessary.

**R29 — cvc5 maintainers. _cvc5 package release and Python synthesis documentation_. Inspected 14 October 2026.** [PyPI](https://pypi.org/project/cvc5/); [SyGuS Python example](https://cvc5.github.io/docs/latest/examples/sygus-fun.html); [Solver API](https://cvc5.github.io/docs/latest/api/python/base/solver.html). Version 1.4.2 release/wheel metadata and synthesis APIs inspected. Availability does not establish a native installation, a working timeout implementation, or compatibility of every latest-documentation call with a pinned release.

**R29c — cvc5 maintainers. _COPYING_. Inspected 14 October 2026.** [License text](https://raw.githubusercontent.com/cvc5/cvc5/main/COPYING). Core and dependency notices inspected. Use the actual pinned source's license manifest when distributing binaries.

**R30 — SyGuS Organization. _SyGuS benchmark collection_. Inspected 14 October 2026.** [Repository](https://github.com/SyGuS-Org/benchmarks); [README](https://raw.githubusercontent.com/SyGuS-Org/benchmarks/master/README.md). Collection structure and README inspected. Individual task functions, semantics, provenance, and licenses require import-time verification; partial PBE data do not become an unrestricted query oracle.

**R31 — Z3 maintainers. _z3-solver package metadata_. Inspected 14 October 2026.** [PyPI](https://pypi.org/project/z3-solver/). Version 5.1.0.0, native macOS arm64 wheel requirements, and MIT metadata inspected. This is availability evidence, not a local performance or correctness measurement.

**R32 — PyTorch maintainers. _MPS backend notes and synchronization API_. Documentation snapshot inspected 14 October 2026.** [MPS notes](https://docs.pytorch.org/docs/2.14/notes/mps.html); [synchronize API](https://docs.pytorch.org/docs/2.14/generated/torch.mps.synchronize.html). Official backend/timing documentation inspected. Actual operators, numerical behavior, extensions, and installed package version remain local checks.

**R33 — pip maintainers. _Repeatable Installs_. Inspected 14 October 2026.** [Official documentation](https://pip.pypa.io/en/stable/topics/repeatable-installs/). Version pinning, hash verification, and wheelhouse guidance inspected. Informs reproducibility rather than any learning claim.

**R34 — Patrice Godefroid, Rishabh Singh, and Hila Peleg. _Machine Learning for Input Fuzzing_. US10983853B2, published 20 April 2021; original assignee Microsoft Technology Licensing LLC.** [Patent publication text](https://patents.google.com/patent/US10983853B2/en). Bibliographic record and relevant description inspected. It concerns learned input generation/fuzzing, including neural sequence models, rather than establishing this exact capsule contribution. No claim construction, legal-status verification, patentability, or freedom-to-operate conclusion was performed.

**R35 — Arya Grayeli, Atharva Sehgal, Omar Costilla-Reyes, Miles Cranmer, and Swarat Chaudhuri. _Symbolic Regression with a Learned Concept Library_. NeurIPS 2024.** [arXiv:2409.09359v2](https://arxiv.org/html/2409.09359v2). Abstract and method framing inspected. LaSR combines evolutionary search and LLM-assisted textual concept libraries. It confirms that broad “concepts plus evolution” is crowded prior art, but is not a neuron-free/local-core baseline under the present exact-witness hypothesis. No claimed regression accuracy is imported.

### 14.2 Search scope, tracing, and stopping judgment

Searches used both the brainstorm's language and more precise equivalents: counterexample-guided inductive synthesis; oracle-guided and interactive synthesis; learning from failures; predicate invention and higher-order abstraction; library learning, anti-unification, and e-graphs; learned abstraction predicates; question selection and experimental design; adaptive automata learning and distinguishing-test reuse; compositional dynamic testing and constraint caching; Boolean functional/Skolem synthesis; reusable witness functions; effectful test-generator synthesis; cost-aware abstraction selection; symbolic regression with concept libraries; clause/automaton learning; and native Apple Silicon solver/build support.

Backward tracing linked the proposed loop to OGIS/FlashMeta, logic-learning predecessors, and compositional testing. Forward and adjacent searches located the later abstraction/search systems, efficient question selection, contemporary Boolean synthesis, SmartLabel, Clouseau, and the September 2026 cost-selection preprint. The Skolem-synthesis connection materially changed the design: a supposedly new constructor learner became a subproblem with mandatory existing competitors. Repository inspection then changed feasibility and fairness gates, especially for Manthan and older Linux/CUDA-specific artifacts.

A final neighboring-domain search also inspected LaSR and a learned-fuzzing patent publication [R34], [R35]. They reinforce the broad prior-art boundary but do not introduce a new executable comparison for the selected pure-function component. Patent searching was preliminary publication screening, not a jurisdiction-wide or claim-level analysis. Search results unrelated to the causal mechanism were not used as evidence.

Research was stopped after targeted checks around the selected refinement returned established functional synthesis, abstraction learning, cached/compositional testing, or adjacent neural/evolutionary systems without changing the core mechanism, blocking gates, or required local comparator set. This is a task-specific coverage judgment, not proof that all prior art has been found. Further citation tracing during implementation is required only when a concrete new competing mechanism, unsupported interface, or changed source materially affects the claim; essential design research is not being deferred wholesale.

Remaining blind spots include unpublished/private systems, unindexed theses or non-English material, comprehensive patent claims, complete reproduction of every source, all variants of physical computing or microscopic neuron replacement, and current support on the user's exact machine. Some sources were accessible only as abstracts or metadata, as marked above. No conclusions rely on inaccessible numerical tables. This search supports a cautious experimental direction, not guaranteed novelty, patentability, or superiority.

---

## Appendix A. Complete executed finite-domain check

Save the following as `capsule_check.py` and run `python3 capsule_check.py`. It needs only Python's standard library. The platform string will differ on a Mac; the mathematical counts should not. This script is a correctness fixture, not the proposed architecture or a speed benchmark.

```python
"""Small finite-domain checks, NOT a performance benchmark or complete learner."""
from itertools import product
import json, platform, sys

W = 4
M = (1 << W) - 1
train_values = (0, 2, 5, 9)
train = list(product(train_values, repeat=2))
all_parameters = list(product(range(1 << W), repeat=2))
heldout = [p for p in all_parameters if p not in set(train)]

# A constructor is evaluated on known parameters theta and delta, not oracle labels.
leaves = [('theta', lambda t,d: t), ('delta', lambda t,d: d),
          ('0', lambda t,d: 0), ('1', lambda t,d: 1)]
ops = [('add', lambda a,b: (a+b)&M), ('sub', lambda a,b: (a-b)&M),
       ('xor', lambda a,b: a^b), ('and', lambda a,b: a&b)]
candidates = list(leaves)
for opname, op in ops:
    for (aname, a), (bname, b) in product(leaves, repeat=2):
        candidates.append((f'{opname}({aname},{bname})',
                           lambda t,d,op=op,a=a,b=b: op(a(t,d),b(t,d))))

def distinguish(x: int, theta: int, delta: int) -> bool:
    z = (x + delta) & M
    return int(z < theta) != int(z <= theta)

winner = None
for tested, (name, f) in enumerate(candidates, start=1):
    if all(distinguish(f(t,d),t,d) for t,d in train):
        winner = name, f
        break
assert winner is not None
name, f = winner
assert all(distinguish(f(t,d),t,d) for t,d in all_parameters)
naive = sum(distinguish(t,t,d) for t,d in all_parameters)
# A suffix that discards the output erases every local distinction.
masked = sum(((int(((f(t,d)+d)&M) < t) & 0) !=
              (int(((f(t,d)+d)&M) <= t) & 0)) for t,d in all_parameters)
# A prefix that clears the low bit cannot reach an odd threshold.
unreachable = sum(not any(int((x & (M ^ 1)) < t) !=
                          int((x & (M ^ 1)) <= t)
                          for x in range(1 << W))
                  for t in range(1, 1 << W, 2))
# Observational equality on one input is not functional equivalence.
assert (lambda x: x)(0) == (lambda x: 0)(0)
assert (lambda x: x)(1) != (lambda x: 0)(1)
# A solved future contrast does not depend on the old task's oracle response.
res = {
 'environment': {'platform': platform.platform(), 'python': sys.version.split()[0]},
 'width_bits': W, 'training_parameter_pairs': len(train),
 'constructor': name, 'constructors_tested_in_fixed_order': tested,
 'heldout_parameter_pairs': len(heldout),
 'heldout_successes': sum(distinguish(f(t,d),t,d) for t,d in heldout),
 'exhaustive_successes': len(all_parameters),
 'naive_unlifted_witness_successes': naive,
 'masked_context_successes': masked,
 'unreachable_odd_thresholds': unreachable,
 'observational_aliasing_counterexample_checked': True,
 'performance_comparison_performed': False,
}
print(json.dumps(res, indent=2))
```

Observed output in the research environment:

```json
{
  "environment": {
    "platform": "Linux-6.18.44-x86_64-with-glibc2.41",
    "python": "3.13.5"
  },
  "width_bits": 4,
  "training_parameter_pairs": 16,
  "constructor": "sub(theta,delta)",
  "constructors_tested_in_fixed_order": 22,
  "heldout_parameter_pairs": 240,
  "heldout_successes": 240,
  "exhaustive_successes": 256,
  "naive_unlifted_witness_successes": 16,
  "masked_context_successes": 0,
  "unreachable_odd_thresholds": 8,
  "observational_aliasing_counterexample_checked": true,
  "performance_comparison_performed": false
}
```

The count 22 depends on the displayed enumeration order and deliberately small constructor grammar. The algebraic result can also be checked directly: at `x=(theta-delta) mod 16`, `(x+delta) mod 16 = theta`, so `< theta` is false and `<= theta` is true. A suffix that always returns zero erases that distinction; a prefix whose range excludes odd values cannot reach an odd boundary. These deductions explain the fixture without treating it as comparative empirical evidence.

---

**End of standalone handoff.** Start at WP00. Do not start a long confirmatory experiment until the preceding correctness, baseline, population, and calibration gates have produced their required artifacts.

[R01]: https://www2.eecs.berkeley.edu/Pubs/TechRpts/2010/EECS-2010-15.html
[R02]: https://arxiv.org/html/1703.03539v1
[R03]: https://doi.org/10.1145/2814270.2814310
[R04]: https://doi.org/10.1007/s10994-020-05934-z
[R05]: https://www.ijcai.org/proceedings/2024/0379.pdf
[R06]: https://arxiv.org/html/1904.08993v2
[R07]: https://doi.org/10.1145/3453483.3454080
[R08]: https://arxiv.org/html/2211.16605v2
[R08b]: https://github.com/mlb2251/stitch
[R09]: https://doi.org/10.1145/3571207
[R10]: https://arxiv.org/abs/1804.04152
[R11]: https://doi.org/10.1145/3586055
[R11b]: https://github.com/jiry17/LearnSy
[R12]: https://link.springer.com/chapter/10.1007/978-3-030-34968-4_8
[R13]: https://arxiv.org/html/2209.00122v2
[R14]: https://arxiv.org/abs/1501.07174
[R15]: https://doi.org/10.1145/1190215.1190226
[R16]: https://arxiv.org/html/2412.17330
[R16b]: https://github.com/SynthesisLab/DeepSynth2/tree/eco_search_aaai
[R17]: https://arxiv.org/abs/2203.10452
[R17b]: https://github.com/google-research/crossbeam
[R18]: https://arxiv.org/html/2405.17514
[R19]: https://arxiv.org/html/2508.15750v2
[R20]: https://arxiv.org/html/2005.06922v1
[R21]: https://arxiv.org/abs/2108.05717
[R21a]: https://github.com/meelgroup/manthan
[R21b]: https://raw.githubusercontent.com/meelgroup/manthan/master/scripts/setup.sh
[R21c]: https://raw.githubusercontent.com/meelgroup/manthan/master/checkSkolem.py
[R21d]: https://raw.githubusercontent.com/meelgroup/manthan/master/LICENSE
[R22]: https://arxiv.org/html/2405.07975v4
[R23]: https://arxiv.org/html/2604.04345v1
[R24]: https://adamcuculich.com/research/on-the-value-of-abstractions/
[R25]: https://arxiv.org/html/1505.03953v4
[R26]: https://arxiv.org/abs/1804.01508
[R27]: https://doi.org/10.1038/nature01568
[R28]: https://support.apple.com/en-us/117737
[R29]: https://cvc5.github.io/docs/latest/api/python/base/solver.html
[R29c]: https://raw.githubusercontent.com/cvc5/cvc5/main/COPYING
[R30]: https://github.com/SyGuS-Org/benchmarks
[R31]: https://pypi.org/project/z3-solver/
[R32]: https://docs.pytorch.org/docs/2.14/notes/mps.html
[R33]: https://pip.pypa.io/en/stable/topics/repeatable-installs/
[R34]: https://patents.google.com/patent/US10983853B2/en
[R35]: https://arxiv.org/html/2409.09359v2
