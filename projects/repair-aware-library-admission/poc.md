# Research and proof-of-concept plan: repair-aware executable hypotheses

**Research date:** 14 October 2026  
**Target:** one MacBook Pro with an M3 Max; actual memory, storage, macOS and toolchain must be discovered  
**Deliverable type:** standalone research and implementation handoff, not a report of a completed architecture experiment  
**Primary direction:** non-neural program induction with immutable reusable abstractions, tested for the value of selecting abstractions for subsequent correction rather than compression or one-shot transfer alone.

## 1. Executive assessment

### Verdict

Proceed with a **narrow, falsifiable study of repair-aware library admission in an executable-hypothesis learner**. Do not launch a project premised on having invented a new alternative to neural intelligence. The broad architecture—learning executable theories, imposing consistency constraints, revising explanations and reusing invented predicates or procedures—has substantial prior art. The Apperception Engine is particularly close; inductive logic programming, truth-maintenance systems, incremental theory revision and library learning cover much of the rest. [R01][R02][R03][R04][R05][R06][R07]

The proposed research question is:

> On streams of small, fully observed worlds with stable hidden rules, does selecting a reusable program library by its performance during sequential correction produce better subsequent learning than compression-only selection and strong generic held-out transfer selection, once all search, selection, storage and inference costs are counted?

This is a plausible **empirical contribution and possibly an objective/application refinement**, not established methodological novelty. Selecting abstractions for anticipated utility is itself already studied. A result attributable merely to standard cross-validation, caching, a favorable grammar or more compute must be reported that way. [R08][R09][R10]

The smallest complete experiment uses a typed, terminating symbolic language; independently implemented world simulators; a recoverable set of competing hypotheses; shared memoized search and constraint checks; immutable learned macros; and a passive, identical sequence of observations for all learners. Active experimentation is a later, separately controlled extension. No pretrained language model, remote service or physical computing apparatus is required.

**What a positive result would establish:** a resource-bounded advantage for a particular abstraction-selection policy in symbolic transition learning, with its operating regime and costs. **What it would not establish:** general intelligence, replacement of contemporary neural AI, autonomous perception, discovery of genuinely new computational primitives, or advantages of unconventional physical hardware.

### Why this direction, rather than the other proposed principles?

| Direction | Assessment and decision |
|---|---|
| Program discovery plus constraints | Preserves the original proposal most directly. Offers exact semantic tests, transferable executable knowledge and a laptop-sized experiment. Selected. |
| An independently “self-correcting constraint field” | Recast as explicit hypothesis/evidence consistency and provenance, not a new physical field. Latent-state, cyclic and probabilistic inference remain outside the first experiment. |
| Evolutionary computational ecosystem | Retain typed mutation as a possible search control after search bottlenecks are measured. A separate population architecture would introduce too many simultaneous causal changes. AlphaEvolve is not evidence that neural proposal mechanisms have been eliminated. [R11] |
| Associative memory and analogy | Include retrieval and previously learned-program reuse as inexpensive baselines. Distinguish exact reuse, parameterized reuse and new synthesis. |
| Physical dynamics | A laptop simulation can study an algorithm, but cannot substantiate physical-substrate energy or throughput benefits. Hardware transduction and readout costs would be indispensable in that claim. Physical reservoir research is relevant context, not the selected local PoC. [R12] |

### Evidence status at handoff

- **Inspected evidence:** original papers or author/publisher records, selected full texts, official source repositories and current dependency documentation, recorded in Sections 3 and 12.
- **Executed here:** small standard-library semantic fixtures on Linux x86_64/Python 3.13.5. Their complete source, output and limited interpretation appear in Appendix A.
- **Not executed:** the proposed learner, comparative benchmarks, baseline installations, M3 Max tests, power measurements or a complete reproduction of any cited paper.
- **Estimates and deductions:** complexity bounds, memory examples, break-even equations and experimental predictions. These are explicitly not laptop measurements.
- **Blocking unknowns:** actual laptop capacity; native compatibility of the full Popper dependency chain; original Apperception implementation licensing/buildability; useful search depth and curriculum scale; whether any advantage survives fair baselines.

## 2. Problem, scope and hypotheses

### 2.1 Intended task and beneficiaries

The initial beneficiary is a researcher or engineer needing an inspectable, incrementally corrected transition model from few observations: for example, a symbolic simulator, a small rule-driven workflow or a game-state model. These are **candidate applications**, not demonstrated user demand. No application-specific return on investment is assumed.

The workflow is: receive a state and action, predict the successor, observe the actual successor, update an executable explanation, and reuse knowledge on a new world. The suspected bottleneck is the combinatorial cost of discovering and correcting programs when useful operations have been composed into a growing library. Search-space growth and the need to manage learned background knowledge are established concerns; the particular claim that correction-aware admission alleviates this bottleneck remains untested. [R07][R13]

The original ambition is preserved at the level of learning executable explanations rather than neural weights. Three deliberate restrictions make its causal mechanism testable:

1. Inputs are structured objects, not pixels or text. Perception has not been solved or hidden inside an unreported model.
2. Initial abstractions are **definitional macros** over a declared language. They enlarge the convenient vocabulary, not the language's expressivity after inlining.
3. The first experiment uses stable deterministic laws. A contradiction means a hypothesis failed, unless the data-consistency checks establish otherwise. Changing laws, label noise and unobserved state are separate conditions.

### 2.2 Formal task

A world has a fixed hidden transition function \(F_w\). An observation is \(e_t=(s_t,a_t,y_t)\), where \(y_t=F_w(s_t,a_t)\). Inputs need not form one uninterrupted trajectory: the core task provides independent, valid reset states. This is **resettable transition learning**, not exploration in an inaccessible physical environment.

After observing \(D_t=\{e_1,\ldots,e_t\}\), a learner returns a point prediction, a status, and optionally a finite set of alternatives for a new input. Its executable representation is a tuple of typed expressions \(P\), with a library \(\mathcal L\) of acyclic macro definitions. Let:

\[
V_t(\mathcal L)=\{P\in\mathcal H(\mathcal L):\forall e\in D_t,\;\operatorname{run}(P,e.s,e.a)=e.y\}.
\]

The implemented frontier \(\widehat V_t\) usually covers only part of \(V_t\). “No candidate found” is not a proof that \(V_t\) is empty. Grammar bounds, exhausted search regions and omitted alternatives are part of every result.

A **curriculum** is one independently generated collection of donor worlds, library-selection worlds and protected target worlds. Shared motifs within a curriculum are intentional. The independent statistical unit is the curriculum, not its objects, transitions, program candidates, timings or optimizer seeds.

### 2.3 Established capabilities, assumptions and speculation

Established precedents include program-based concept learning on a specific handwritten-character task, executable causal-theory induction, neural-guided library learning, purely symbolic program search, failure-derived constraints and incremental theory revision. These do not jointly prove a general non-neural intelligence architecture. [R01][R03][R04][R06][R14][R29]

The selected mechanism assumes that useful rules are reasonably short in the declared language; observations contain distinguishing cases; related tasks share reusable structure; and search-cost savings can exceed library-selection overhead. Full observability and stable object identities are supplied by the task. None of these assumptions should be silently attributed to the learner.

Speculative extensions—learning new semantics outside the language, raw perception, latent causal state, recursive planners, open-ended self-modification and physical realization—are not prerequisites for the core result.

### 2.4 Hypothesis register

| ID | Claim and supporting rationale | Alternative explanation | Observation that weakens or falsifies the local claim |
|---|---|---|---|
| H1 — primary | Correction-aware library admission lowers protected prequential exact-state error relative to generic all-prefix transfer admission, at matched information and resource envelopes. Compression and prospective utility can differ; whether correction adds value is unknown. [R08][R09] | More validation computation, smaller library, favorable prefix ordering, generic objective alignment rather than a new mechanism. | No reproducible benefit after equal selection resources; generic admission ties or wins; benefit disappears under cardinality matching or fresh ordering. |
| H2 — attribution | Any H1 gain is associated with faster or more reliable correction, not just initial synthesis. | Better initial hypotheses, cached evaluations, additional candidates, a different grammar. | Gains occur only before the first counterexample; shared caching eliminates them; no interaction with correction difficulty. |
| H3 — transfer | Learned macros improve new combinations of seen motifs beyond exact retrieval. | Hidden teacher primitives, duplicated programs, memorized states or task IDs. | Gain vanishes after equivalent-program deduplication, opaque-ID permutation or unseen compositions. |
| H4 — integrity | Immutable definitions and complete dependency handling prevent accidental changes to unrelated stored predictions. This is an engineering invariant, not novel learning theory. [R05] | Only positive traces were indexed; negation or an empty quantifier was omitted. | Differential tests find stale outputs, changed unrelated roots or resumed-state divergence. |
| H5 — secondary active track | Candidate disagreement obtains useful observations more efficiently than random admissible queries in an identifiable regime. Oracle-guided synthesis already exists. [R15] | Privileged access to teacher rules, test inputs, free query-scoring compute or unreachable states. | Savings disappear after matching valid query access and charging acquisition cost; selected queries reveal no new distinctions. |
| H6 — practical headroom | Total savings eventually repay construction and selection costs on a finite, realistic-length curriculum. | Only the search kernel improved; setup was excluded. | No observed break-even before the measured useful task horizon, or resource/accuracy regressions dominate. |

H1 is not a claim about noisy observations, law changes or all possible program languages. H4 must pass before interpreting H1. H5 is not required for a successful passive PoC.

### 2.5 Strongest case for and against

**For:** programs can share structure across many instances; explicit counterexamples can eliminate whole hypotheses; typed semantic constraints prune impossible explanations; and immutable factoring can allow local revisions without destroying useful components. The finite symbolic setting permits exact correctness and information-access audits.

**Against:** the language may encode most of the problem; short programs may remain hard to find; missing observations may leave hypotheses unidentifiable; a growing library increases branching; and replay-based admission can cost more than it saves. A tuned decision tree, cached synthesizer or ordinary validation-based selector may obtain the same benefit. Exact consistency with observations is neither truth nor calibrated uncertainty. An individually useful component need not improve an end-to-end system.

Two deductions guide escalation. With full-rebuild time \(T_s+T_r\), unchanged shared work \(T_s\), retained repair fraction \(\rho\), and bookkeeping cost \(T_o\):

\[
\text{speedup}=\frac{T_s+T_r}{T_s+\rho T_r+T_o};\qquad
\text{benefit requires }T_o<(1-\rho)T_r.
\]

For library construction and selection cost \(C_L\), downstream per-task saving \(\Delta_j\), and extra execution/storage cost \(C_{x,j}\), break-even requires
\(\sum_{j=1}^{J}(\Delta_j-C_{x,j})>C_L\).
Report this curve and its observed horizon; do not extrapolate an unobserved perpetual saving.

## 3. Research landscape and novelty assessment

### 3.1 Closest prior art

“Local” below means a credible native execution route to be verified, not a completed installation. Published results in different domains are not head-to-head scores.

| Work | Problem, mechanism and assumptions | Evidence, cost and implementation | Exact overlap and implication |
|---|---|---|---|
| Apperception Engine, Evans et al. [R01][R02] | Small symbolic causal theories constrained to explain sensory sequences; structural unity requirements and domain templates. | Full evaluation paper and Haskell/ASP repository inspected. Its reported setup includes multi-hour per-task limits on a Unix desktop. Native build and license gates are unresolved. | Very close to the broad executable-explanation engine. Do not claim symbolic theories plus consistency as new. |
| Popper, Cropper and Morel [R03][R16] | Generate, test and constrain logic programs using failed examples; explicit hypothesis bias. | Original paper record and current source/configuration inspected. Local CPU route, but current dependencies require a separate environment. | Direct strong program-induction competitor; learned failure constraints are established. |
| MaxSynth, Hocquette et al. [R04] | MDL-style logic learning under noisy positive/negative evidence. | Full paper inspected; built on Popper. Different noisy objectives can require different sound pruning rules. | “Handle contradictions” is not novel. Clean and noisy pruning must not be conflated. |
| ILED, Katzouris et al. [R06] | Incremental event-definition revision with support information preserving previous coverage under stated assumptions. | Full article and repository inspected; older Event Calculus infrastructure. Original scope is not arbitrary functional programs. | Revision without wholesale forgetting and support-set compression already exist. Mandatory conceptual comparison; reproduce compatible examples when feasible. |
| Truth maintenance, Doyle [R05] | Belief justifications, dependency-directed revision and alternative assumptions. | MIT archival abstract inspected; no new performance reproduction. | Provenance and contradiction propagation are established architecture ideas. |
| DreamCoder, Ellis et al. [R14] | Program search and reusable abstractions, with learned neural guidance. | Full preprint inspected. Domain-specific libraries and recognition training matter to cost. | Reusable executable knowledge and multiple candidate programs are not new; not a purely non-neural baseline. |
| Stitch, Bowers et al. [R07][R17] | Corpus-guided top-down abstraction synthesis optimizing program/library compression. | Full paper, Rust repository and Python API inspected. CPU implementation, MIT repository. | Reuse it rather than build another compressor. Its utility framework also limits claims that merely changing a utility is methodologically novel. |
| babble, Cao et al. [R18] | E-graphs and anti-unification seek abstractions across equivalent program representations. | Original abstract and artifact availability inspected; full method not reproduced. | Syntactic compression can be an unfairly weak control when equivalent syntax varies. Add this route if representation sensitivity is material. |
| Probe, Bee Search and Eco Search [R33][R34][R35] | Just-in-time probabilistic guidance and cost-ordered bottom-up synthesis, rather than naive uniform enumeration. | Probe/Eco Search full texts and author repositories inspected; Bee Search record and repository inspected. Original benchmark languages differ from these worlds. | Modern guided synthesis is a necessary competitor; cheap candidate generation is not constant-time learning. |
| AbstractBeam, Zenkner et al. [R36] | Neural execution-guided synthesis combined with library learning in an integer-list domain. | Full preprint and author repository inspected. Historical PyTorch/scatter setup and GPU-oriented launcher require native compatibility work. | Closer neural-symbolic comparison than a generic neural predictor; conditional local reproduction, otherwise explicitly contextual. |
| STEVIE, Hocquette et al. [R37] | Higher-order abstraction discovery by refactoring logic programs. | Full IJCAI 2024 paper, rendered first page and author repository inspected. Higher-order scope differs from G0. | Inventing useful executable vocabulary is already developed beyond first-order macros; relevant escalation comparator. |
| Knorf, Dumančić et al. [R09] | Refactors learned background knowledge using invented predicates and optimization. | Full preprint and publisher record inspected; tasks include structured program-learning domains. | Background-library restructuring and reducing subsequent search cost are established. |
| Playgol and Forgetgol, Cropper [R13][R19] | Acquire reusable procedures through play; manage or forget learned background knowledge. | Original publication records inspected. Symbolic implementations, task-specific logic bias. | Non-neural accumulation and selective retention predate this proposal. |
| Oracle-guided synthesis, Jha et al. [R15] | Component-based loop-free synthesis using examples and SMT-guided oracle interaction. | Original author record inspected. Different executable language and oracle setting. | Choosing distinguishing experiments is established, not a separate novelty claim. |
| Prospective Compression, Hernandez Cano et al., 2026 [R08] | Studies abstraction utility under anticipated, potentially changing future demands. | Full preprint inspected; human Pattern Builder task and computational models. | Prospective rather than retrospective usefulness is already explicit prior art; correction-specific empirical value must be isolated. |
| TheoryCoder-2, Ahmed et al., 2026 [R10] | LLM-synthesized reusable abstractions and hierarchical planning. | Full preprint inspected; pretrained/API-model context differs from the local requirement. | Strong current overlap at the architecture level. Published results are contextual, not comparable laptop measurements. |
| ADVENT, Yu et al., 2026 [R20] | LLM-driven predicate proposals with logical checking and reusable knowledge. | Original abstract inspected, not full reproduction. | Predicate invention plus verification is not an unoccupied area; neural dependence distinguishes it from this implementation. |
| SMPMA, Paatur, 2026 [R21] | Online library evolution in a program-synthesis system. | Author repository/README inspected; full preprint access unsuccessful. Results are not independently verified here. | Recent direct overlap that prevents sweeping “first online library” claims. |
| SmartLabel, Barnaby et al., 2025 [R22] | Active selection for neurosymbolic synthesis, including uncertainty from neural components. | Full preprint inspected. Different noise source and task language. | Acquisition quality and acquisition overhead require explicit controls. |

### 3.2 Reuse, differentiation and invalidation

**Reuse:** Stitch for candidate macros; cvc5 for an independent bounded constraint-based learner and a native SyGuS comparison on a compatible grammar; Popper for relational ILP; standard trees, XGBoost and PyTorch for non-symbolic comparisons; standard typed ASTs, content addressing, SQLite and differential testing for infrastructure. None of these components is a claimed invention. [R16][R17][R23][R24][R25]

**Selected difference:** evaluate library admission by **chronological correction replay**, while keeping the proposal language, candidate macro pool, program search, local-repair operators, evidence access, cache implementation and resource envelopes shared. Compare it to retrospective compression and generic held-out terminal- and all-prefix transfer utility; R–G is the primary contrast. Attribute only the difference remaining after those controls.

**Why it could matter:** two equally compressive libraries may favor different search paths. A macro that hides a usually correct guard may accelerate initial solutions but impede the correction triggered by a rarer relation. Conversely, additional parameters may support cheap correction but enlarge search. The proposed study measures that tradeoff; it does not assume it is favorable.

**What would invalidate novelty:** a prior system using the same correction-conditioned admission objective and evaluation regime; an implementation already exposing the same policy; or results fully explained by known prospective selection or knowledge refactoring. The contribution then becomes replication, engineering or a new controlled empirical comparison. Search absence is not proof of novelty.

**What would invalidate practical value:** a cheaper tuned baseline matching performance; selection costs exceeding downstream savings; or gains confined to a human-designed primitive that all useful target concepts already instantiate.

### 3.3 Research refinements made before implementation

The original architecture-level claim was narrowed after the Apperception/ILP comparison. A second search for “theory revision,” “predicate invention,” “knowledge refactoring,” “prospective compression,” “abstraction repair,” “online library evolution,” and “oracle-guided synthesis” found further direct overlap. Consequently, the contribution is not described as a new learning principle.

The older and newer results do not imply a contradiction: success on sparse, structured symbolic tasks is compatible with hard program search, useful neural search guidance, or an advantage for simpler methods on easy instances. Different representations, learned assets and accounting conventions explain much of the apparent conflict. No published cross-domain score is imported into the local leaderboard.

A final search for guided bottom-up synthesis added Probe, Bee Search, Eco Search and AbstractBeam to avoid treating uniform search or a direct neural predictor as the strongest relevant alternatives. Higher-order predicate invention and reusable synthesis frameworks were also checked through STEVIE and Herb.jl. [R33][R34][R35][R36][R37][R38]

There is no justified claim that the selected refinement is entirely unprecedented. The experiment is valuable only if it answers a specific, currently unresolved question about correction-aware selection under fair costs.

## 4. Mechanism and implementation specification

### 4.1 Exact world representation

Use JSON records with a versioned schema:

```json
{
  "schema_version": 1,
  "world_instance_id": "opaque-observation-group",
  "width": 6,
  "height": 6,
  "objects": [
    {"id": "opaque-a", "x": 1, "y": 2, "color": 0, "solid": true},
    {"id": "opaque-b", "x": 2, "y": 2, "color": 1, "solid": true}
  ],
  "action": {"actor_id": "opaque-a", "dx": 1, "dy": 0, "paint": 2}
}
```

The example illustrates the schema, not a supplied law. Coordinates are integers; colors are the finite set `{0,1,2,3}`; actions are the four cardinal unit directions or zero movement, plus a supplied paint color. Object IDs support equality only. IDs and world identifiers must never become numeric features, tokens that carry a rule class, or synthesized constants. Serialize in canonical ID order, but computation must be permutation-equivariant.

The core has a fixed object set during each transition. The successor contains every object with the same ID and solid flag; x, y and color may change. All updates read the **old** state and occur simultaneously. Co-location is representable; whether movement is blocked is a learned law, not a universal constraint. Inputs have valid coordinates. A predicted out-of-bounds coordinate is an incorrect prediction, not silently clamped by the evaluator.

World size and object count vary. Missing fields, duplicate IDs, an absent actor, unsupported colors or an invalid action are schema errors. They are not noisy training examples. Padding is masked and cannot create fictitious objects. No raw-source filenames, teacher ASTs, motif IDs, generator branch names or hidden future labels are exposed through this schema.

### 4.2 Typed language

Implement a pure, first-order AST, not Python `eval` or generated unrestricted code. Types are `Bool`, `Int`, `Color`, `Step`, `Object`, `World`, and `Action`. `Step` takes values −1, 0 or 1. The top-level rule is `(step_x, step_y, next_color)` evaluated separately for each object `i`:

\[
x'_i=x_i+\text{step}_x(s,a,i),\quad y'_i=y_i+\text{step}_y(s,a,i),\quad c'_i=\text{next_color}(s,a,i).
\]

The initial grammar is:

```text
I ::= 0 | 1 | width | height | dx | dy
    | x(i) | y(i) | x(j) | y(j) | I + I | I - I
    | count j in objects: B
B ::= true | false | solid(i) | solid(j) | actor(i) | actor(j)
    | same_object(i,j) | I == I | I < I | C == C
    | not B | B and B | B or B
    | any j in objects: B
C ::= color(i) | color(j) | paint | c0 | c1 | c2 | c3
    | if B then C else C
D ::= -1 | 0 | 1 | dx | dy | if B then D else D
P ::= (D, D, C)
```

`j` exists only inside the closest `any` or `count` binder. Initially allow one binder level; no nested aggregation, general recursion, multiplication, unbounded loops, hidden state or object creation. The bound is a scope choice to be tested, not a claim about general expressivity. Empty `any` is false; empty `count` is zero. Equality is typed. Integers are mathematical integers in the reference interpreter and SMT encoding. Fixed-width optimized code must demonstrate equivalence or detect overflow and fall back.

All candidate macros have explicit typed parameters for their free variables, including any world/action/object context. Use de Bruijn indices for bound variables; no string substitution that can capture `j`. A macro may call only earlier immutable macros. Type-check its inlined body, parameter occurrences and binder scope. Reject invalid, cyclic or non-total definitions.

Useful domain concepts such as “blocked,” “touching,” “trigger color,” “safe move,” or “wall exception” are **not primitives**. They must be represented using the language above. However, even arithmetic, object fields and quantifiers are substantial supplied bias; disclose them in every result.

### 4.2.1 Mandatory finite core profile G0 and escalation profile G1

The unrestricted grammar above is the semantic language, not a promise that unconstrained enumeration is cheap. The **mandatory local PoC uses G0**, a declared normal-form search profile that retains conjunction learning, relational guards, reuse and correction while making candidate generation explicit. G1 expands toward the full grammar only after measured evidence supports it. This restriction is part of the supplied inductive bias, not learned intelligence.

For G0, construct ten pair atoms from the primitive interpreter:

```text
same_object(i,j), solid(j), actor(j), color(j)==color(i),
color(j)==paint, x(j)==x(i)+dx, y(j)==y(i)+dy,
x(j)==x(i), y(j)==y(i), color(j)==c0
```

A quantified predicate is `any j: conjunction(literals)`, with zero through four distinct pair atoms, each positive or negated. Four literals suffice to express identity exclusion, solidity and two-coordinate destination matching; this is why that bound is present. Before simplification there are exactly
`sum(comb(10,k)*2**k for k in range(5)) = 4521` possible bodies. Include both polarities of a quantified predicate. Its expanded AST, not a free named “blocked” feature, determines structural cost.

Unary atoms are `actor(i)`, `solid(i)`, `color(i)==paint`, and `color(i)==c0/c1/c2/c3`. A root guard is a conjunction of at most two distinct unary literals and at most one quantified predicate. Root programs are decision lists with zero, one or two guards; leaves are the declared Step terminals or `color(i)`, `paint`, and color constants. Two guards express a main case plus an exception. Use this same search profile for R/C/U/G/B0 and their learned macros; fully expanded macro use must remain in the profile. A macro is a cheaper syntax route, not permission for one method to leave G0.

Generate G0 movement inputs so the actor's proposed step is inside the grid. This avoids silently giving the learner boundary clipping as part of the update interpreter. G0 includes obstruction, color and one-exception composition; boundary learning, counts and richer arithmetic belong to G1. The size-shift test still changes distances, locations and distractor counts within the declared G0 input domain.

Represent each candidate guard's truth values over current `(example, object)` rows as bitsets. For a decision-list branch, discard a proposed leaf when its predicted values disagree with any row on which that branch fires. Recurse on the remaining rows, with at most two guards, using cost-ordered search. Cache remaining-row subproblems, but retain derivations for splitting after new evidence. This is ordinary hypothesis-space pruning applied to all variants. It avoids materializing every pair of guards. Chunk guard evaluation and record actual candidate counts; do not allocate all guard/program combinations at once.

For a finite core correctness test, exhaust the restricted grammar and state space at a reduced atom set, compare enumeration to the independent SMT learner, and verify identical minimal consistent costs. For the full ten-atom G0 profile use differential samples plus exhaustive tests of each primitive. G0 failures can distinguish invalid implementation, insufficient evidence and lack of repair-selection value. They cannot by themselves reject every richer non-neural architecture.

### 4.3 Program cost and hypothesis status

Use a reproducible **MDL-inspired structural cost**, not a claim of a calibrated Bayesian prior. Every primitive/operator/call AST node costs one; count nodes recursively, including macro arguments. A library costs the sum of definition-body node counts plus one per typed parameter and one per definition. References to library definitions cost a call node plus argument costs. Charge the library once when reporting total representation cost. Report fully inlined node counts separately.

Fixed-library root costs are additive. Store common subexpressions in a DAG to accelerate evaluation, but do not change the objective to count shared DAG nodes without declaring a new condition. For equal cost use canonical serialization as the deterministic tie-breaker.

Return one of:

- `CONSISTENT_FOUND`: the returned program passes all revealed examples, without a global truth claim.
- `INCONSISTENT_DATA`: two identical canonical inputs have different observed outputs in the deterministic task.
- `UNSAT_WITHIN_BOUND`: a complete search/solver certificate excludes the explicitly reported finite grammar bound.
- `BUDGET_EXHAUSTED`: no sufficient result within a compute or memory envelope.
- `UNSUPPORTED` or `ERROR`: a capability or implementation problem; never relabel these as a learning failure.

A retained frontier can contain multiple consistent programs. Normalize finite weights such as `exp(-lambda * cost)` only for an explicitly named **frontier heuristic**; tune lambda on development data, include uniform weighting, and never call the resulting confidence the posterior over all programs. Point prediction uses the most-weighted full successor, then canonical tie-breaking. Abstention is permitted but counts as an error for the primary endpoint; coverage-risk is secondary.

### 4.4 Reference search and independent constraint learner

**Reference enumerator, B0:** bottom-up typed enumeration by structural cost. Cache each expression's output vector over current evidence and its type/context. Apply globally valid rewrites (`not not b`, identical conditional arms, arithmetic identities) to every symbolic method. Never equate two expressions permanently just because they agree on the observed states.

Observational-equivalence buckets must retain either all members compactly or a regenerable derivation/backpointer representation. When evidence grows, split affected buckets. A capped frontier reports truncation and retains the enumerator checkpoint needed to explore omitted derivations. Learned failure pruning is valid only under its stated grammar and clean/noisy assumptions. Appendix A gives an executable counterexample to lossy deduplication.

Because every output reads the old state, the three roots can be fitted separately to revealed field targets, then joined by an ordered Cartesian-product heap. This factorization is valid for the declared additive objective and synchronous semantics, not for arbitrary mutually recursive next-state programs. Validate joint predictions against the complete successor state. All methods receive the same field-level supervision derived from the observed transition.

**Independent SMT learner:** encode a bounded, typed AST with shared operator-choice variables and child indices ordered topologically. For each input/object pair add evaluation variables. For an operator alternative, constrain its value to the corresponding child operation using an implication or `ite`. Unroll the finite `j` domain with the same operator choices reused at every object index; mask padding. Add observed-output equalities. A selector may not choose a different program for each example. Increase structural bounds in the same schedule as enumeration. Use cvc5 as the default independent solver; Z3 is an optional cross-check. Save SMT-LIB, results and all bounds. [R23][R26]

The broad SMT encoding is a **project adapter using an existing solver**, not a claimed faithful reimplementation of a particular synthesis paper. Separately reproduce cvc5's native SyGuS example and run native SyGuS on the finite-feature subtrack below. Do not claim its sample API is incremental: the inspected official SyGuS example explicitly disables incremental solving. Reconstruct a fresh SyGuS job from the accumulated transcript when required. [R23]

### 4.5 Evidence, provenance and repair

Store an append-only evidence table; immutable AST nodes; macro definitions; root-to-definition references; expression-output caches; and evidence-to-root obligations. A cache key includes `(DSL hash, definition hash, input hash, bound-variable environment)`. Never key only by a mutable macro name or world ID.

An execution trace records read fields, binder domains and branch predicates. `not`, an empty `any`, `count`, and a branch not taken can still influence an obligation. Absence checks depend on the relevant collection, not on “no object” as an empty dependency set.

The repair process is:

```text
observe(e):
    validate schema and canonicalize input/output
    if same input already has a different output:
        return INCONSISTENT_DATA in the deterministic track
    append evidence and create output obligations
    evaluate old hypotheses on e; update all evidence-consistency scores
    keep every still-consistent candidate or regenerable alternative
    identify failed roots and relevant control/data dependencies
    propose typed subtree replacements in failed roots, smallest edits first
    copy changed nodes; never overwrite a shared definition
    interleave local edits with resumed global enumeration
    test candidates against ALL revealed evidence, using valid cached values
    atomically commit the best consistent frontier and its search checkpoint
```

Local replacement enumerates any well-typed subtree of increasing cost; include condition insertion, guard replacement and branch replacement as ordering heuristics, not the only allowed changes. The entire failed root must eventually be eligible. The local/global allocation is selected on development workloads and shared by all library-admission variants. A globally inconsistent local patch cannot be accepted just because it fits the newest example.

The dependency slice prioritizes search; it is **not** a general theorem that the globally best repair lies inside the slice. Full search is the fallback. Disjoint immutable root dependencies imply unchanged executable values for those roots on the same inputs, but do not imply that their explanations are true on unseen data.

### 4.6 Library proposals and admission policies

For each curriculum divide already observed source worlds into donor set **E** and selection set **V** before fitting anything. Mine candidate macros only from evidence-consistent programs learned on E. Teacher programs are never the corpus. Log multiple-frontier mining as a separate setting; use the same frontier inputs for every selector.

Use Stitch to propose a common ordered pool of candidate abstractions. Its inspected Python API includes `compress(programs, iterations=..., max_arity=..., threads=...)`, and rewriting with the returned abstractions. Bindings are not a type checker for this language: validate capture-safe translation and expansion independently. [R17]

A candidate addition is a dependency-closed bundle: include every unadmitted ancestor definition it calls, and charge their full cost. A selector may not admit a dangling macro reference. Cardinality-matched controls count the whole dependency closure.

Stitch's native compression weights need not equal the core structural cost. Recompute matched-policy compression scores with the declared cost; validate translation and rewriting explicitly. Also retain an upstream-native-cost Stitch condition as a competitive baseline, with its own reported cost units. Do not call the rescaled condition an unchanged native reproduction or compare compression ratios with different units.

For the minimum study, the pool contains compressive candidates produced under development-selected arity/iteration settings. This may miss non-compressive but useful abstractions; record the limitation. A later anti-unification proposal extension requires a fresh matched comparison, not an unreported advantage for the proposed selector.

Compare these admission policies using the same initial library, candidate pool and target search engine:

| ID | Selection criterion on source worlds | Purpose |
|---|---|---|
| **C — compression** | Maximize donor corpus structural saving, including definition cost. Tune library size and compressor settings on development curricula. | Strong retrospective library baseline. |
| **U — terminal transfer utility** | On V, synthesize from the complete revealed support set under the allotted budget; score error on a disjoint source-validation probe set. Break ties by total search work, then library cost. | Strong simple held-out terminal-utility baseline. |
| **R — correction-aware utility** | On V, replay revealed transitions in their original order, predicting each next transition before revealing it. Minimize mean prequential exact-state error under the allotted update budgets. Break ties by total executed interpreter-node work, then library cost. | Selected proposed policy. |
| **G — generic prefix utility** | Score the same prefixes as R, but reconstruct each prefix's program without a previous-program edit warm start; retain generic semantic caches. Use the selected library in the same target repair engine as R. | Primary H1 comparator; separates sequential warm-start/revision alignment from merely validating more prefixes. |

All policy scores are finite-sample selection criteria, **not confirmatory statistical tests**. Admission uses forward selection: start empty; evaluate every affordable remaining candidate addition; choose the lexicographically best strictly improving addition; repeat until no addition improves or the common selection-work envelope is exhausted. Store the best visited library and all evaluations. Break score ties deterministically. A validation-selected removal pass is a separately matched control, not default extra power for R.

For C, use its native compression optimization rather than forcing it through an unnecessarily expensive replay implementation. Compare both equal-envelope performance and actual total cost. Unused selection compute must be reported, not falsely charged as performed work. In the equal-total-budget condition it may be reassigned by a development-frozen policy to source search or later adaptation.

**Leakage prohibition:** a V world's eventual solution may be used as revealed source evidence during admission scoring, but must not have been used to generate that candidate before replaying its earlier prefixes. E supplies candidates; V supplies prospective evaluation. Replaying a program on the prefix of the very task from which that program was mined is not prospective evidence. Protected target worlds supply neither proposals nor admission decisions.

Freeze the learned library before the initial protected target phase. Per-target hypothesis revision remains online. Online addition/removal of library definitions during protected evaluation is a later extension, using a predeclared rolling donor/selection schedule and fresh confirmatory curricula.

### 4.7 Active extension, not part of the primary comparison

Offer every learner the same independently generated pool of valid reset states/actions, excluding protected probe inputs. At a fixed observation budget select the query maximizing weighted successor disagreement, such as entropy over frontier predictions. Include uniform random sampling without replacement and a query scorer based on a tree or neural ensemble. Charge pool construction, all candidate executions and query selection.

Do not ask the teacher to produce a counterexample to an arbitrary program unless **all methods** receive that stronger oracle in a separately labeled experiment. Exact SMT disagreement can find a state on which two hypotheses differ, but it must obey public state-validity rules and be answered through the same charged oracle. No true hidden rule is a solver constraint.

Stop acquiring when the common observation budget is exhausted or all candidates agree on every admissible pool item. Agreement is not certainty outside that pool. Report query savings only alongside acquisition and adaptation costs. The four-site ring in Appendix A verifies the information mechanism, not benchmark superiority.

## 5. Fair and sensitive evaluation

### 5.1 Evaluation population and generator specification

Create an original, public synthetic benchmark; do not present it as a real-world dataset. Release project-authored code under an explicit permissive license and generated data under an explicitly chosen data license, after confirming ownership. Imported examples retain their original licenses.

A curriculum contains disjoint donor worlds E, selection worlds V, and protected target worlds T. Development curricula and confirmatory curricula are independently generated. A world's hidden law remains fixed throughout its stream. Draw all random choices from named RNG streams derived from a master seed with a cryptographic hash; never Python's process-randomized `hash()`.

Implement two teacher routes:

**Teacher A — DSL-generated:** sample typed programs from a declared finite grammar configuration, reject non-total/ill-typed programs, and retain only programs whose generated benchmark states have valid outputs. Deduplicate by canonical syntax plus a fixed, independently seeded behavioral fingerprint. A finite fingerprint is not proof of global equivalence. The hidden teacher AST is retained in the evaluator-only provenance store.

**Teacher B — independently written simulator:** write imperative implementations of the following rule families without calling the learner's interpreter. Derive oracle outputs from old-state snapshots, not in-place update order. Where a family belongs to the declared grammar, supply a separate witness AST only to correctness tests. This is independence of implementation, not a claim of independent scientific authorship.

| Family | Imperative rule recipe | Distinction to expose |
|---|---|---|
| Free movement | The actor takes its action step, except a boundary condition that keeps coordinates valid; others stay. | Persistence versus movement; boundaries. |
| Conditional obstruction | Actor moves unless an object at the destination satisfies a chosen conjunction over solid/color/equality relations. A blocked actor stays. | “All objects block” versus a qualifying subset. |
| Conditional color effect | A selected object changes to the action paint color when a chosen relation holds; otherwise retains color. | Unchanged position does not imply unchanged state. |
| Composed effects | Combine independently selected movement and color rules, both reading the old state. | Reuse of a guard across output roots; synchronous semantics. |
| Context-dependent exception | Add a second spatial/color relation that overrides an otherwise shared rule. | A common early explanation is correct on ordinary cases and wrong on an exception. |
| Count/context boundary | A guard depends on whether the number of qualifying objects is 0, 1 or greater than 1. | Existential versus count-sensitive behavior; optional richer-grammar stratum. |

A “chosen conjunction” is generated mechanically from the public pairwise atoms available in the grammar, not from a library of named correct predicates. Enumerate permitted atom lists in `configs/grammar.yaml`; draw their inclusion patterns from the curriculum seed. Choose paint colors, directions, guard polarity and relation combinations independently where valid. Reject only for preregistered semantic reasons, not because a method succeeds or fails. Log every rejection and its reason.

Use curriculum-level motif pools: E and V share some guard fragments; T contains new combinations and exception placements, not copies of complete source programs. Include a **no-reuse control** with disjoint fragments. Metadata that identifies the pool or family remains evaluator-only.

For each world, generate an initial support block followed by a prequential stream and independent frozen probe sets. Ordinary reset states are sampled uniformly from the declared construction distribution: choose dimensions and object count; sample distinct IDs, positions, colors and solid flags; sample actor and action. Duplicate positions are allowed when the selected family admits them. Apply the family validator before oracle evaluation.

For an exception-rich stream, generate ordinary and exception-triggering states by rejection sampling using teacher-only predicates. The observer sees only states/actions/outputs. Its ordering is fixed before any learner runs. Include randomly interleaved order and ordinary-first order; do not report only the latter. Rejection sampling changes the evaluation distribution and must be disclosed. Report natural-trigger-frequency results separately from trigger-balanced diagnostic results.

The ordinary-first condition tests correction of a plausible but wrong explanation; it is **not actual law drift**. In the primary population, average ordinary-first and randomly interleaved curricula with weights frozen in the manifest. Include both reuse-rich and no-reuse curricula, producing four primary design cells. Report every cell, not only the pooled mean.

Initial development anchors may use 4–6-wide grids and 1–4 objects, with 8–12-wide grids and 8–16 objects for size shift. These are starting points that expose size generalization, not final limits or a claim of sufficient scale. Increase complexity before freezing when the baseline curves show ceilings, floors or absent distinguishing cases. Do not increase it because a particular method is losing.

Use the G0 family/input restrictions in Section 4.2.1 for the mandatory core; label G1 families separately. The final manifest must specify exact E/V/T counts, stream lengths, support sizes, dimensions, object-count distributions, motif/exception distributions, grammar bounds, atom lists, all distribution weights and seed derivation. W05 determines these values from measured search capacity and information availability; W07 freezes them. No unspecified randomness remains in confirmation.

### 5.2 Splits and leakage defenses

A random transition split alone is inadequate. Enforce all of the following:

- Split complete hidden rule programs and composition families; share only explicitly designated transferable motifs. Remove exact and behaviorally fingerprinted full-program duplicates across E/V/T. Preserve a list of unresolved equivalences.
- Separate all donor inputs, selection inputs, target inputs and diagnostic probes by canonical input hash within each world. A test input may be seen only when its scheduled prediction is requested; its label is then revealed only according to the declared stream protocol.
- Never select the best hypothesis using hidden probes, full teacher traces or a learned-program/teacher syntax match. Structural recovery is evaluator-only secondary analysis and must quotient known equivalences.
- Keep teacher metadata and protected labels out of model-process files and function arguments. Use a small oracle interface; record every call, returned observation and acquisition cost. Protect results from manual inspection until the fixed confirmation completes.
- Train all variants on the same revealed source examples. A source program supplied by a human or pretrained model is an additional asset, not free information. The main experiment uses neither.
- Share any generic feature extractor, normalization rule, semantic simplifier or cache optimization when applicable. Count its build/execution cost and record its source.

### 5.3 Baseline panel and faithful comparison scope

No single published system is “the strongest” across an original workload before running it. The following panel combines close prior-art methods with strong simple methods. Select and freeze configurations using development curricula, then report all applicable confirmatory results.

| ID | Baseline and implementation contract | Direct claim scope |
|---|---|---|
| B0 | Typed memoized enumeration, no learned macros; includes resumable search, safe equivalence handling and the same local repair proposals as R. Also run global-search-only as an ablation. | Entire declared grammar; strong matched simple explanation. |
| C, U, G | Section 4.6 selectors, all using the same target engine and candidate pool as R. | Entire grammar; primary mechanism comparison. |
| B-SMT | Independent cvc5 bounded-AST encoding, with symmetry reduction, reusable constraints where supported and warm-start opportunities. Tune it rather than comparing only to a cold naive solver. | Entire grammar if differential tests pass; otherwise report exact capability coverage. |
| B-guided | Eco Search with a source-fitted probabilistic grammar, plus its uniform-cost condition; native DeepSynth integration preferred. Probe is a targeted adaptive-guidance extension if static guidance is a material bottleneck. | Mandatory cost-guided comparison on the common feature track; full G0 only after translation tests. A custom PCFG heuristic is not called Probe. |
| B-neural-synth | AbstractBeam/LambdaBeam-style execution-guided synthesis with library learning, using the author implementation when feasible. | Conditional local compatible-language comparison; otherwise a broader reference and an explicit limit on neural-superiority claims. |
| B-SyGuS | Native cvc5 syntax-guided synthesis on a common finite-feature grammar. Run the official example before the adapter. | Feature subtrack only; not represented as the full relational-quantifier implementation. |
| B-ILP | Current Popper in its own locked environment; relational facts, positive and negative output examples, explicit type/direction/body bounds. Reproduce its bundled game example first. | Preregistered expressible relational subset. Count-sensitive or negation-heavy cases require a validated encoding; do not quietly drop hard cases. |
| B-AE | Original Apperception implementation on its supported examples and a validated common transition subset, conditional on license and build gates. | Closest architecture reproduction/context. A clean-room adapter is labeled as such, never called the original system. |
| B-memory | Exact transition lookup plus objectwise nearest-neighbor prediction on the shared feature track; tune neighbors, distance normalization and tie-breaking on development data. | Rapid-update/retrieval baseline; no learned executable abstraction. |
| B-tree | Decision tree, extremely randomized trees and XGBoost CPU histogram trees on shared, mechanically derived features. Select the strongest validation configuration rather than one arbitrary tree. | Strong tabular/relational-feature comparison, not equal inductive bias. |
| B-neural | Task-conditioned relational predictor with source training and target adaptation, specified below; CPU or tested MPS float32. | Strong local neural comparison within measured training-data/compute ranges, not the best possible neural architecture. |
| D-controls | Persistence, a simple action-only rule, a deliberately wrong model and a teacher-oracle upper bound. | Harness diagnostics. The teacher oracle is privileged and excluded from competitive rankings. |

**Finite-feature subtrack.** Mechanically enumerate public unary/pairwise predicates to a development-selected structural bound, then aggregate permitted pair predicates by `any` and `count`. Export all features and definitions, including those never used by the teacher. Do not include only helpful predicates. Every method in this subtrack receives identical features. Native SyGuS builds Boolean combinations and conditional output expressions over these features; trees and nearest neighbors use the same matrix. This subtrack trades explicit feature construction for simpler learners. Its performance does not prove autonomous primitive discovery.

**Cost-guided synthesis adapter.** First run the pinned DeepSynth implementation on an included permitted small example and validate enumeration order on an exhaustible typed grammar. Then translate the common finite-feature grammar, preserving all output leaves and constraints. Use uniform positive production costs and, separately, a non-neural source-fitted prior
\[
p(r\mid A)=\frac{n_E(r)+\alpha}{\sum_{r':A}n_E(r')+\alpha |R_A|},\qquad c(r)=1-\log_2 p(r\mid A),
\]
where counts come only from learner-acquired donor programs, not teacher programs, and alpha is tuned on development data. The added 1 is an explicit positive-cost convention. Freeze probabilities within an enumeration pass; rebuilding after an update is charged. Search cost is not the structural complexity reported for the resulting program. Keep all previously observed examples as constraints, and use recoverable semantic equivalence rather than forgetting discarded alternatives across counterexamples. Export the same features and source assets to other methods. Reproduce upstream behavior before labeling an adapter as Eco Search; custom queues with merely similar ordering receive a separate name. No constant-delay guarantee is asserted for interpretation, filtering or the entire adaptation loop. [R35]

Compare without a library and with compression-selected source macros. If the guided ordering materially improves the shared engine during development, make it available to R/C/U/G before freezing rather than retaining a weakened control. A Probe experiment is justified when partial-solution guidance, rather than enumeration mechanics, is the unresolved bottleneck: reuse the original or a validated implementation, freeze its update schedule on development data, and charge grammar updates/restarts. Do not relabel the static frequency prior above as Probe. [R33]

**Neural-guided synthesis gate.** AbstractBeam is not replaced by an arbitrarily smaller predictor and then called reproduced. The original repository lists Python 3.8.19, PyTorch 2.2.1 and torch-scatter 2.1.2, with a GPU-oriented launch configuration. Test the actual required scatter, search, training and checkpoint paths in an isolated native environment. A CPU port requires forward/backward and search-order equivalence tests; an incompatible kernel, unlicensed asset or infeasible training budget blocks the faithful comparison, not the main non-neural mechanism study. Train any local guide on learner-acquired or explicitly self-generated programs with charged generation costs, never concealed teacher solutions. State whether results are original-domain reproduction, a compatible-language adaptation or a new neural-synthesis baseline. [R36]

**Popper adapter.** Represent each `(state, action, object)` as an opaque example entity with field/relation facts. For each root, the observed output is positive; other values in its finite output domain are negative. Share any mechanically constructed aggregation predicates with competitors. Mapping a conditional to logic clauses must preserve functional single-output semantics; a hypothesis returning zero or multiple outputs is invalid for exact prediction. Set type, direction, variable and body-size bounds to cover all preregistered witness programs. A small exhaustive truth-table test must verify the translation. An unsupported language feature is a scope limitation, not a timeout.

**Trees.** Tune depth from shallow through unbounded, minimum leaf size, number of estimators and feature subsampling. For XGBoost tune depth/leaf complexity, learning rate and regularization with early stopping on source-development worlds. Use categorical encoding agreed in the feature manifest. Train per-target predictors on that world's revealed support; optionally add source-learned macro features as an explicitly charged transfer condition. Do not pool examples from incompatible hidden world laws without task conditioning.

**Neural specification.** Use a permutation-equivariant relational network, not a flattened fixed-size MLP that cannot represent larger worlds. Encode each object using normalized and raw relative coordinates, color one-hot, solid flag, actor flag, action and dimensions. Use dense masked pair messages based on `(h_i,h_j,x_j-x_i,y_j-y_i)` with sum/mean/max aggregation and residual node updates. A support encoder sees each revealed before/action/after transition; pool its object embeddings and attend from query objects to support-transition embeddings. Use dot-product attention and standard dense operations to reduce backend risk. Predict categorical x-step, y-step and next color; decode their joint successor without teacher constraints.

Development candidates use one to three message-passing blocks and widths spanning a doubling sequence from a small working model; extend until validation or resource evidence shows the relevant capacity range. Train source episodes with varying support sizes and unseen query transitions, cross-entropy over the three output heads, Adam-style optimization and float32. Tune learning rate, batch size, support encoder capacity, source-training steps and per-target adaptation versus context-only inference. The learner receives no teacher program or family ID. Record both the selected model and its learning curves. A persistently improving neural baseline that cannot reach a stable regime locally makes a broad “better than neural learning” interpretation inconclusive, even if the matched-budget comparison remains valid.

Baselines may exploit the whole permitted laptop resource envelope, including multiple CPU cores or tested GPU operations. Give all methods the same tuning objective, comparable development opportunity and their implementation-appropriate optimizations. Keep reusable dependency builds separate from project-specific source fitting, library admission and neural training; disclose both. Development tuning cost is not deployment adaptation cost, but must appear in the project compute ledger.

### 5.4 Primary endpoint and resource accounting

The primary contrast is **R versus G**. This challenges the stronger simple explanation that ordinary held-out evaluation of all prefixes already suffices. C and U are mandatory attribution comparisons; the other applicable baselines test practical competitiveness.

For target world w, let an initial support block be revealed without scoring. For each remaining transition t, provide its input, obtain a prediction, then reveal its output and permit an update. Define:

\[
\ell_{mwt}=\mathbf 1\{\widehat y_{mwt}\ne y_{wt}\},\qquad
E_{mc}=\frac1{|T_c|}\sum_{w\in T_c}\frac1{|J_w|}\sum_{t\in J_w}\ell_{mwt},
\]

where comparison is the entire successor object state, IDs aligned, and \(J_w\) is the scored portion of the stream. Failed, invalid or absent predictions have loss one. Each world receives equal weight, and each curriculum contributes one paired difference \(d_c=E_{Gc}-E_{Rc}\). Positive \(d_c\) favors R. Report mean error as a proportion and as errors per 100 predictions, plus the paired absolute difference and relative reduction when the G denominator is nonzero.

**Primary resource regime: matched total curriculum envelope.** Choose a common wall-time budget B and memory envelope from development calibration. Include source fitting, program discovery, library construction, admission replay, serialization/loading, target adaptation, feature extraction and prediction. Framework-only oracle generation/scoring is reported separately and identical access is enforced. Do not hide source setup by starting the stopwatch at target inference.

Each method freezes a source/selection allocation policy on development data. After setup, distribute remaining learner wall time over the remaining target operations using a deterministic, logged schedule. Cheaper setup therefore permits more target computation. Do not charge C for validation replay it did not perform. Record actual setup time, target allocations and unused budget. Once the envelope is depleted, the method may return a cached output only if that operation is included; otherwise return no prediction and score an error.

**Secondary regime: matched per-update envelopes.** Give R/C/U/G identical target update budgets and disclose their different source costs. This isolates the correction mechanism and yields break-even curves. It cannot independently support an equal-total-cost claim.

Use a deterministic interpreter-work budget for a further R/C/U/G sensitivity check. It removes wall-clock threshold jitter but does not replace realistic timing against compiled solvers or neural methods, whose work units are different.

Calibrate a reference update scale from the median time for B0 to obtain a consistent hypothesis on informative development prefixes. Explore budgets geometrically around it and freeze the primary operating point without using protected results. Select a regime with measurable headroom rather than one where every learner is perfect or universally fails. Budget choice must not maximize R's observed advantage. Show neighboring budgets to expose fragile operating-point dependence.

### 5.5 Secondary metrics and guardrails

Report exact-state accuracy on independently frozen probes for familiar sizes, larger sizes and unseen motif combinations. Also report per-object/field accuracy, but do not let easy unchanged distractors substitute for whole-state success. Separate actor, changed-object and unchanged-object performance using evaluator-only labels.

Correction metrics include evidence-consistency restoration, executed node evaluations, candidate count, total repair time and post-counterexample probe error. Define trigger positions by the preregistered stream, not by selecting only failures of R. A method-specific “time after its own error” is descriptive and not an unbiased paired endpoint.

For time to recovery, return a consistency/prediction-quality curve up to a common horizon, including non-recoveries. Use restricted mean time or a deadline-penalized time with that same horizon; never average time only over successful runs. Provide success rate alongside latency.

Track total and inlined program size, macro count/arity, source-to-target call reuse, initial-solve versus revision work, active oracle calls, peak memory, disk growth, solver timeouts, invalid predictions, retained/omitted frontier counts and achieved compute. Report conditional frontier calibration/coverage only as an approximate model diagnostic.

A primary-error benefit does not imply overall superiority if it produces an important memory, latency, source-cost, out-of-distribution or retained-knowledge regression. Present a Pareto comparison rather than a single composite score with invented business weights.

### 5.6 Pilot, sample size and statistical analysis

Development and pilot results are exploratory. The confirmatory protocol is fixed before reading protected outcomes, including the primary contrast, population weights, endpoint, resource regime, selected baseline configurations and missing-result rules.

Use independent pilot curricula covering all four primary design cells. Add complete balanced pilot batches until the variance estimate and predicted sample requirement are sufficiently stable for an explicit decision; store the batch sequence and never label pilot p-values confirmatory. A useful stability check is whether deleting any pilot batch changes the planned sample requirement by more than one balanced batch. Zero pilot variance caused by universal success/failure is a workload-sensitivity problem, not evidence that two confirmation samples suffice.

Let \(s_d\) be the pilot standard deviation of curriculum-level paired differences. Choose the desired 95% confidence half-width h before confirmation. With no external application utility supplied, use an interpretable planning convention: precision of half of one target-world contribution to the curriculum's normalized loss, \(h=1/(2|T_c|)\) for equal-size curricula. This is a precision target, **not a minimum improvement required for success**. A more demanding application-derived precision may replace it in the recorded freeze decision.

Compute an initial fixed sample requirement
\[
N=\left\lceil(z_{0.975}s_{d,\mathrm{upper}}/h)^2\right\rceil,
\]
where \(s_{d,\mathrm{upper}}\) is a conservative upper variability estimate from pilot curriculum resampling; iterate with a t critical value for small N. Round up to balanced design-cell batches. Simulate the planned paired analysis using pilot residuals to estimate detectable effects and assess the normal-approximation planning formula. Save the full power/precision curve rather than asserting guaranteed power.

If the required N is not locally feasible at the calibrated workload, report the achievable confidence width and the unresolved effect range. Revise an insensitive proxy before confirmation when justified; do not silently reduce N, shrink the task until it flatters R, or keep running until significance. Missing disk space or excessive measured runtime is a resource limitation, not evidence against H1.

For the primary analysis compute a curriculum-paired mean difference and a 95% interval using a stratified cluster bootstrap that resamples entire curricula within their preregistered cells. Use a paired mean/t interval as a sensitivity check, and report distribution/skewness and all individual curriculum differences. Fix bootstrap RNG and use enough resamples that endpoint Monte Carlo uncertainty is negligible relative to h. For very small samples or extreme bounded outcomes, do not treat an unstable bootstrap interval as decisive; report exact finite-instance counts and the sensitivity limitation.

The primary R–G comparison has one prespecified endpoint. Apply Holm familywise adjustment to the declared secondary superiority contrasts against C, U and the other applicable competitive baselines. An assertion of superiority to **all** tested applicable baselines requires simultaneous support, not merely beating the development-selected favorite. Descriptive ablations are identified as such; new confirmatory claims require a fresh independent run.

Repeat stochastic method fits within each curriculum only to quantify algorithmic variation; average or hierarchically analyze them according to the frozen rule. They are not additional independent curricula. Likewise, repeated timings are measurements of one workload, not extra learning samples. A fresh confirmation should change both source and target curriculum seeds; repeating the same learned library on more target states establishes only conditional precision for that library.

### 5.7 Measurement hygiene and harness validation

Warm up dependencies and the relevant execution path before steady-state timing; report cold-start separately. For GPU work synchronize before starting and after completing a timed region. Time complete subprocess trees, not only the Python caller. Include model compilation/loading and cache construction where they occur in deployment. Record wall time, CPU time, thread settings, cache condition and whether output decoding is inside the boundary.

Run methods in counterbalanced order on the same powered laptop with the same power mode. Avoid concurrent experiments. Record ambient/thermal warnings and background-load indicators when accessible; do not invent sensor readings. Repeat timing blocks until their uncertainty is small enough to distinguish the claimed timing effect, or classify that timing effect as unresolved. Report median and tail measurements with the raw repetitions retained.

Harness tests must show: identical method copies have no systematic paired difference; a deliberately corrupted predictor is detected; the teacher oracle has zero error; input-ID permutations preserve results; withheld labels cannot be requested early; padding changes do not change semantics; timeout/abstention scores are correct; and interrupted deterministic-work runs resume identically. A broken harness blocks all scientific claims.

### 5.8 External construct-validity check

After the controlled synthetic result, attempt a compatible public symbolic transition task. The inspected Popper repository includes `examples/iggp-rps-next-score`; reproduce that existing example and, subject to its data license and validated schema conversion, use the relevant game-transition representation for a separately frozen adaptation task. [R16]

Do not report the repository example as a held-out discovery benchmark after tuning on it. Additional external task files, licensing, split provenance and representability must be established before use. The original Apperception examples can provide another faithful reproduction check. Neither an elementary cellular automaton nor a game-score rule by itself demonstrates general intelligence; lookup tables and standard logic learners remain essential controls.

A synthetic-only positive outcome remains a synthetic transition-learning result. A failed external transfer test can reflect representation mismatch or missing primitives; distinguish those from an advantage that genuinely fails outside the designed distribution.

## 6. M3 Max feasibility and hardware-adaptive execution

### 6.1 Hardware discovery before dependencies or large runs

Apple lists multiple M3 Max configurations; maximum RAM is not an appropriate working assumption. [R27] Execute and save these existing macOS commands:

```bash
uname -m
sw_vers
sysctl -n hw.memsize
sysctl -n hw.logicalcpu
system_profiler SPHardwareDataType SPDisplaysDataType
df -h .
vm_stat
```

Require native `arm64` for the selected toolchain, or explicitly label a Rosetta condition and keep it out of native performance claims. Record physical memory bytes, storage availability, OS version, CPU/GPU identity, Python architectures, power mode and relevant compiler versions. Measure idle memory and working-set variability before assigning a safe process-group memory envelope. Do not infer free memory from installed RAM alone.

The core symbolic learner is CPU-based. GPU acceleration is useful for a baseline but not required for the proposed mechanism. No paid API, external machine or large model download is part of the core path.

### 6.2 Dependency and licensing strategy

| Component | Evidence inspected on 14 October 2026 | Installation/validation decision |
|---|---|---|
| Main environment | Standard Python plus numeric/testing packages; no dependency run was performed here. | Start with native CPython 3.12 or 3.13 according to available compatible wheels. Resolve, test, then lock the exact successful interpreter and packages. |
| cvc5 | PyPI showed 1.4.2, released 8 October 2026, with Apple Silicon wheel listings; current SyGuS API inspected. [R23] | Test the exact wheel, official example, bounded-AST encoding and interruption behavior. A listed wheel is not proof of this workload's compatibility. |
| Z3, optional | PyPI showed `z3-solver` 5.1.0.0, released 16 August 2026. [R26] | Use only if its native wheel and required integer/Boolean operations pass smoke tests. It is not required when cvc5 suffices. |
| Stitch | Rust implementation and Python docs labeled `stitch_core` 0.1.11 were inspected; repository is MIT. [R17] | Try bindings, then a native Rust build at a recorded commit if needed. Test capture-safe rewriting and inlining, not just successful import. |
| Popper | Inspected `pyproject.toml` declares version 5.0.0 and **Python >=3.14**, with clingo, Janus/SWI and compiled dependencies. [R16] | Separate locked environment and subprocess adapter. Do not force it into the main environment or use an obsolete PySwip setup. |
| SWI-Prolog/Janus | Official macOS instructions and Janus notes inspected. Janus may require source compilation and correct SWI library paths. [R28] | Verify `swipl`, compiler and Python architectures agree; test a Prolog call from the exact Popper interpreter. |
| Apperception | Haskell/clingo requirements in original repository inspected; no clear root license was established. [R02] | License/build gate before reuse or redistribution. Preserve failure logs. No unsupported assertion of an available permissive license. |
| Guided search | DeepSynth/Eco Search repository and project metadata inspected; repository declares MIT. Probe's repository states CRAPL; do not assume it is MIT. [R33][R35] | Separate resolved environment if needed; native grammar, enumeration and evaluator tests before target runs. No current release tag is assumed. |
| Neural-guided synthesis | AbstractBeam author repository declares Apache-2.0 and historical torch/scatter dependencies. [R36] | Optional isolated environment; operation-level compatibility gate before data generation or training. |
| PyTorch | Apple documents the MPS route; operation-level support remains to be tested. [R24] | Float32 CPU/MPS forward, backward, optimizer, padding and dense-attention parity tests. Explicit CPU fallback, no silent unsupported-op fallback. |
| XGBoost | Official installation documentation supports macOS/Apple Silicon distribution routes; CPU setup selected. [R25] | Test import, CPU histogram training and threading. Do not assume CUDA or a Metal training backend. |

Create separate lockfiles for the main environment and external baselines. Pin git commits using the actual checked-out SHA, not a guessed release tag. Save upstream licenses and notices for every reused code/data asset, including transitive native dependencies. Publicly downloadable source or weights are not automatically permissively licensed. No pretrained weights are required here. Herb.jl is a reusable alternative for a difficult baseline port, not a mandatory second main implementation; adopt it only after a small semantic comparison shows that the added Julia environment reduces work without changing the comparison. [R38]

The following are **existing upstream example commands**, not generated project commands:

```bash
# Official SWI-Prolog macOS route, when Homebrew is present:
brew install swi-prolog

# In a checked-out Popper repository, using its own environment:
uv run popper.py examples/iggp-rps-next-score

# In the original Apperception repository, after its documented build:
~/.cabal/bin/solve misc predict_1.lp
```

Do not execute installer changes destructively or replace an existing user's environment. Determine whether GNU `timeout` is available as `timeout` or `gtimeout` when an upstream script requires it; a project-owned subprocess deadline is safer than assuming a Linux command name. Baseline ports that alter search semantics need a separate label and an amendment.

### 6.3 Memory model

Use measured coefficients, not just asymptotic complexity. Approximate peak memory as

\[
M_{\mathrm{peak}}=M_{\mathrm{runtime}}+M_{\mathrm{AST}}+M_{\mathrm{cache}}+
M_{\mathrm{evidence}}+M_{\mathrm{solver}}+M_{\mathrm{model}}+M_{\mathrm{activation}}+
M_{\mathrm{checkpoint\ buffer}}.
\]

Material contributors:

- Packed AST storage: approximately `node_count * measured_bytes_per_node`; Python object overhead can be much larger than a packed layout. Measure both before choosing Rust/C++ work.
- Expression cache: `cached_expressions * evidence_inputs * evaluated_objects * bytes_per_value`, with an additional object factor for materialized pairwise/binder values. Bit-pack Booleans only after semantic tests. Stream or recompute instead of assuming the full tensor fits.
- Solver memory: syntax-choice variables plus per-example evaluation constraints; a single finite binder can require work proportional to `AST_nodes * examples * objects^2 * operator_alternatives`. Native solver overhead must be measured.
- Neural parameters/optimizer: float32 weights, gradients and two Adam moment arrays alone are approximately `16 * parameter_count` bytes, before temporary buffers, activations and backend overhead. Dense pair messages scale with batch size, object count squared and width; support encoding adds another contributor.
- Apple unified memory is shared. CPU RSS and GPU allocator numbers can overlap conceptually; do not simply sum all reported metrics as if they were independent physical allocations.

**Illustrative arithmetic, not measurements:** 100,000 cached expressions evaluated on 256 inputs with 8 objects at 8 bytes/value require 1,638,400,000 bytes (about 1.53 GiB) for values alone. One million packed 64-byte AST records require about 0.060 GiB before indexes; actual Python objects may require substantially more. A one-million-parameter float32 Adam model needs about 15.3 MiB for those four parameter-sized arrays, but activations and libraries may dominate. These examples show why a symbolic search can exhaust memory despite small task states.

### 6.4 Calibration and conditional configurations

Before large runs, sweep expression count, evidence count, object count, grammar size and neural batch size independently over geometric steps. Measure candidate evaluations/second, solver tasks/second, peak resident memory, disk bytes and model step time. Stop a sweep before unsafe memory pressure; extrapolate conservatively only within a tested range.

Set a safe project envelope from physical memory minus measured OS/application working set and a recorded operational reserve. Start conservatively; adjust only from calibration. The reserve is an operational safety constraint, not a scientific model-size limit. No experiment should deliberately force the user's machine into sustained swap or disk exhaustion.

Choose cache blocks, frontier storage, solver-worker count and neural batches to fit that envelope. Hardware adaptation follows this order:

1. Reduce simultaneous workers and materialized cache blocks; preserve the same search space and evidence.
2. Spill recoverable AST/frontier records to indexed disk; include I/O in cost.
3. Use faithful CPU execution for unsupported neural GPU operations, keeping precision and architecture fixed.
4. If a different grammar bound, model capacity or dataset scale is necessary, create a new explicitly labeled condition. It cannot inherit the original comparison claim.

Increasing frontier caps changes approximation quality even when inlining semantics remain intact; report it as a configuration change. Quantization, reduced precision, lossy evidence deletion and replacing a solver are not silent memory fixes.

### 6.5 Resource estimates and scheduling rule

No M3 wall-time estimate is asserted here. Estimate each stage on the actual laptop:

| Stage | Work estimate | Dominant resource and scheduling decision |
|---|---|---|
| Data generation/validation | Number of oracle evaluations × measured evaluation cost; retain rejection rate. | CPU and small serialized data. Generate deterministic shards lazily. |
| B0/R/C/U/G search | Measured candidate and subtree evaluations at each grammar bound. | Expression cache/AST growth. Stop at the common scientific budget, retaining the frontier. |
| Macro proposals | Measured Stitch cost versus corpus size, total syntax nodes and maximum arity. | Corpus/index memory. A runaway compressor blocks scaling until a justified configuration is selected. |
| Admission replay | Candidate additions × validation prefixes × measured update work. | Often the main hidden cost. Reuse correct shared results and count all cache construction. |
| SMT/ILP | Empirical solve-time distributions, including failures, across pilot strata. | Heavy-tailed runtime and native memory. Do not extrapolate from successful easy examples alone. |
| Neural training | Steps × measured complete step time; source learning curves determine useful steps. | Activations/backend memory. Batch adaptation must preserve the declared optimizer semantics. |
| Confirmation | Sum of calibrated per-curriculum envelopes × fixed N × repetitions. | Elapsed laptop use and persistent storage; execute sequentially and resumably. |

A feasible small mechanism test is strongly suggested by its finite states and CPU implementation. Useful comparative scale remains an empirical prerequisite. If the selected evidence precision cannot be reached locally, deliver the exact confidence/power limitation rather than pretending the laptop validated a broader claim.

### 6.6 Checkpointing and interruptions

Checkpoint after each revealed observation, accepted library decision and completed work chunk. Persist RNG states, evidence hash, definitions, root references, frontier/derivation state, cache schema, spent budget, configuration hash and pending oracle position. Write to a temporary file, fsync, then atomically rename; use transaction boundaries for SQLite.

Resume only if source revision, dependency lock, DSL semantics, dataset hash and budget accounting match. Otherwise create a new run linked to the old one. Killing a solver may require rebuilding its process from the saved transcript; do not assume opaque solver state is serializable. Preserve the killed job and actual spent time.

On memory pressure stop accepting new work, checkpoint and exit with `RESOURCE_LIMIT`. Do not silently discard evidence or finish with a smaller model. A checkpoint corruption test and interruption/restart test are mandatory before confirmation. Deterministic symbolic runs must match exactly; stochastic/GPU training must satisfy the frozen reproducibility tolerance and disclose any unavoidable nondeterminism.

## 7. Repository, interfaces and executable conventions

### 7.1 Minimal repository

All files below are **to be implemented by the coding agent**, except the semantic fixture reproduced in Appendix A. The plan does not imply they already exist.

```text
executable-hypotheses/
  README.md
  pyproject.toml
  uv.lock
  LICENSE
  configs/
    grammar_g0.yaml
    grammar_g1.yaml
    development.yaml
    methods/
    confirmatory.yaml
  src/ehe/
    cli.py
    schemas.py
    protocol.py
    budgets.py
    dsl/{ast,types,interpreter,normalize,serialize}.py
    worlds/{teacher_ast,teacher_imperative,sampling,splits,oracle}.py
    search/{enumerate,bitsets,frontier,repair,constraints}.py
    library/{stitch_adapter,proposals,admission,costs}.py
    methods/{base,enumerator,selectors,smt,sygus,ilp,apperception,trees,neural,memory}.py
    storage/{evidence,dag,checkpoint}.py
    evaluation/{runner,metrics,statistics,timing,report}.py
  baselines/
    manifests/
    environments/
    patches/
  tests/
    test_semantics.py
    test_translation.py
    test_version_space.py
    test_dependencies.py
    test_information_access.py
    test_harness.py
    test_resume.py
    semantic_checks.py
  data/{development,source,protected,external}/
  artifacts/
    hardware.json
    dependency_manifest.json
    bias_inventory.yaml
    calibration/
    decisions/
    freezes/
    runs/<run_id>/
    reports/
```

Use one Python package and a CLI before adding orchestration infrastructure. SQLite plus compressed JSONL is sufficient initially. Add a compiled hot loop only when profiling shows it changes the useful experimental scale; apply it to all methods sharing that operation.

### 7.2 Shared method protocol

The following is an interface sketch to implement, not an importable released library:

```python
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

@dataclass(frozen=True)
class Budget:
    wall_seconds: float
    memory_bytes: int
    work_units: int | None = None  # only meaningful for a declared work counter

@dataclass(frozen=True)
class Prediction:
    successor: dict[str, Any] | None
    status: str
    alternatives: tuple[dict[str, Any], ...] = ()
    metadata: dict[str, Any] | None = None

class Learner(Protocol):
    def fit_source(self, source_bundle: dict[str, Any], budget: Budget) -> dict[str, Any]: ...
    def start_world(self, public_schema: dict[str, Any], budget: Budget) -> None: ...
    def predict(self, state_action: dict[str, Any], budget: Budget) -> Prediction: ...
    def observe(self, transition: dict[str, Any], budget: Budget) -> dict[str, Any]: ...
    def select_query(self, pool: Sequence[dict[str, Any]], budget: Budget) -> int | None: ...
    def checkpoint(self, path: Path) -> None: ...
```

`source_bundle` explicitly separates E transcripts, V replay transcripts and V probe sets; every method has the same allowed access. U's probes follow the same declared distributions as target probes, not a specially easy terminal distribution. During simulated replay only the current prefix is passed to the simulated learner. Cache ownership and source/target boundaries are checked by the wrapper.

`start_world` receives no hidden family metadata. It resets per-world evidence while retaining only the frozen allowed source assets. `predict` has no access to the successor. The harness measures resources independently of method-reported counters. Subprocess adapters communicate through validated JSON files or stdin/stdout with unambiguous status records, never by parsing an undocumented human message as a proof certificate.

### 7.3 Proposed project commands

These commands do not exist until the agent implements `src/ehe/cli.py`:

```bash
python -m ehe.cli doctor --out artifacts/hardware.json
python -m ehe.cli check --suite semantics
python -m ehe.cli check --suite baselines
python -m ehe.cli generate --config configs/development.yaml
python -m ehe.cli calibrate --config configs/development.yaml
python -m ehe.cli pilot --config configs/development.yaml
python -m ehe.cli freeze --config configs/confirmatory.yaml
python -m ehe.cli run --freeze artifacts/freezes/CONFIRM_ID.json
python -m ehe.cli resume --run artifacts/runs/RUN_ID
python -m ehe.cli analyze --freeze artifacts/freezes/CONFIRM_ID.json
python -m ehe.cli report --freeze artifacts/freezes/CONFIRM_ID.json
```

`CONFIRM_ID` and `RUN_ID` are documented CLI metavariables, not actual asset identifiers. The freeze command must reject missing calibrated values, unsupported method/population combinations, unresolved mandatory dependencies and mismatched grammar/data hashes.

### 7.4 Configuration and record schemas

Configuration has these required groups: `schema_version`, `grammar_profile`, `grammar_hash`, `data_manifest_hash`, `phase`, `curriculum_cells`, `source_counts`, `target_counts`, `stream_schedule`, `method_configs`, `resource_policy`, `primary_contrast`, `metrics`, `statistical_plan`, `seed_derivation`, `dependency_locks`, `source_revision`, and `amendment_id`.

The confirmatory file contains actual numbers and immutable hashes produced by calibration; no `auto`, unspecified default, runtime hyperparameter search or unresolved placeholder is permitted. Configurations for development can contain explicit search grids and calibration rules. Confirmatory mode must not silently run those searches again.

Each JSONL event records:

```text
schema_version, run_id, curriculum_id, method_id, phase, world_id,
event_index, event_type, input_hash, revealed_label_hash,
prediction_hash, status, grammar_hash, library_hash, state_hash,
seed_stream, budget_granted_seconds, elapsed_wall_ns, cpu_ns,
work_counter_name, work_units, peak_rss_bytes, backend_memory_bytes,
cache_hits, cache_misses, candidate_count, frontier_truncated,
checkpoint_hash, error_code, source_revision, config_hash
```

Use `null` for unavailable measurements, not zero. Store full inputs/predictions/outputs in content-addressed payload files linked by the hashes. Timing units are nanoseconds; resource sizes are bytes; accuracy is a dimensionless proportion. Event ordering is monotonic and checked on resume.

A run summary also records upstream repository SHAs, exact installed package versions, compiler/backend details, asset licenses and checksums, complete stdout/stderr, failed attempts, budget overruns, warm/cold conditions and method execution order. Source/setup, per-target adaptation and final inference have separate timing fields that sum to the reported boundary without double-counting.

### 7.5 Human-supplied bias inventory

Record every input attribute, action/output domain, language operator, G0 atom, normal-form constraint, type rule, feature extractor, invariant and teacher template in `bias_inventory.yaml`. For each entry identify its author, introduction date, source, rationale and whether it changed after observing results. Count domain-specific code and manually supplied concepts separately from generic infrastructure. Log agent/developer effort when measured; do not invent human-hour estimates.

In particular, the seven unary atoms, ten pair atoms, quantifier-body bound and decision-list form are supplied prior structure. Learning a macro over them is not the same as discovering objects, arithmetic or new physical relations from unstructured sensation. A result whose useful concepts are effectively all supplied by this inventory has a narrower interpretation even when the implementation is non-neural.

## 8. Risk-ordered work packages and decision gates

Each package produces a decision record with `continue`, `revise`, `stop_local_claim`, `blocked`, or `inconclusive`, together with evidence and the next permitted action. Routine implementation decisions are delegated to the coding agent. Changes to the claim, population, primary endpoint, information access, baseline fairness or interpretation require a methodological amendment and, after unblinding, new confirmation.

### W00 — Prerequisites, provenance and native smoke tests

**Objective/risk:** eliminate licensing, dependency and hardware blockers before expensive work. **Inputs/dependencies:** this plan, fresh target laptop and permitted public downloads. **Actions:** implement `doctor`; run Section 6.1 discovery; establish isolated environments; capture upstream SHAs/licenses; test Stitch rewrite, cvc5 integer/Boolean and official SyGuS examples, SWI/Janus call, Popper bundled example, and neural CPU/MPS forward/backward parity. Also validate an included DeepSynth/Eco Search example and record the AbstractBeam scatter/training compatibility result. Attempt the original Apperception build only after its reuse conditions are resolved.

**Outputs:** `hardware.json`, dependency/license manifests, smoke-test logs and exact environment locks. **Resources:** measured small examples; no dataset-scale downloads or model weights. **Acceptance:** native core interpreter/toolchain and at least one independent solver path work; baseline capability table distinguishes validated, blocked and unsupported entries. **Branches:** a neural backend failure uses faithful CPU; a Popper/Apperception blocker narrows its direct-comparison scope and retains the gap; failure of both independent solver paths blocks competitive synthesis claims until fixed. Do not repair dependency issues by changing the experiment without a record.

### W01 — Semantics and finite indispensable mechanism

**Objective/risk:** establish that the learner's abstraction is coherent and can represent the distinctions being tested. **Dependencies:** W00's main environment. **Actions:** implement typed AST/interpreter/G0 normalization; port Appendix A; implement macro expansion and immutable hashing; exhaust a reduced G0 atom/state space; build the shared G0 guard bitset evaluator. Add the bounded SMT profile using literal selectors with shared syntax across all examples.

**Outputs:** interpreter, grammar JSON/YAML, golden fixtures and minimum-cost tables. **Tests:** exact agreement across interpreter, bitset evaluator and SMT; all totality/scope/negation/empty-quantifier cases; minimal costs match on exhaustible instances; known alternatives survive new evidence. **Resources:** bounded finite fixtures; record primitive execution counts and peak memory. **Acceptance:** zero semantic discrepancies and at least one instance with genuine competing explanations and a distinguishing observation. **Branches:** mismatch means implementation repair, not a hypothesis result; no distinguishability means redesign the proxy before building the learner; an unrepresentable teacher rule is moved to a declared missing-primitive stratum.

### W02 — Independent data, evaluator and leakage barriers

**Objective/risk:** prevent teacher/interpreter agreement from masquerading as correctness, and prevent data leakage. **Dependencies:** W01 semantics. **Actions:** implement both teacher routes, exact schemas, data splits, hidden oracle process, input/behavior fingerprints and task-cell generation. Construct ordinary-first and interleaved streams without method feedback. Implement scoring and timeout/abstention handling.

**Outputs:** development manifest, independent witness fixtures, generator rejection ledger and evaluator tests. **Tests:** imperative versus witness AST predictions; permutation/padding invariance; oracle zero error; known-bad predictor detected; identical copies tie; hidden labels inaccessible before their event. **Resources:** lazy small shards, measured bytes per transition and oracle latency. **Acceptance:** complete reproducible input distributions and information-access logs. **Branches:** leakage invalidates and regenerates affected splits; semantic disagreement returns to W01; a trivial ceiling/floor returns to development generator design, with reasons recorded.

### W03 — Strong baselines before proposed admission

**Objective/risk:** establish competitiveness and headroom without a weak comparison. **Dependencies:** W00–W02. **Actions:** implement B0, B-SMT, B-guided, memory and tree baselines; run native-cost Stitch baseline; build compatible Popper/SyGuS/Apperception adapters where permitted; implement the task-conditioned neural baseline and attempt the conditional neural-synthesis route only after its compatibility gate. Tune implementation-appropriate configurations on development curricula. B-SMT should be tested both without a learned library and with a source-derived compression library when that translation is valid.

**Outputs:** baseline manifests, capability matrix, reproduction logs, tuning traces and error/compute curves. **Tests:** all algorithms consume the same revealed labels; translation agrees on exhaustive small cases; published example behavior is reproduced where specified; no incompatible laws are pooled without conditioning. **Resources:** increase tuning only while validation curves or implementation fixes can change the baseline decision. **Acceptance:** a strong nontrivial baseline panel with observed ceilings/floors and a verified independent competitor on G0 and a validated cost-guided competitor on the common feature track. Unsupported conditional baselines remain visible, not silently omitted. **Branches:** universal success means insufficient headroom; universal failure means language/search/information diagnosis; a continuously improving undertrained baseline prohibits claims about its converged potential.

### W04 — Shared revision engine and library bridge

**Objective/risk:** isolate correction from generic caching and test H4. **Dependencies:** W01–W03. **Actions:** implement append-only evidence, recoverable frontiers, provenance, immutable root copying, typed subtree repair and full-search interleaving. Integrate Stitch through typed, capture-safe translation; implement dependency-closed candidate bundles and exact structural-cost accounting.

**Outputs:** shared engine used by R/C/U/G, candidate-library corpus and cache/repair traces. **Tests:** incremental versus cold full evaluation on every small fixture; no unrelated-root changes; negative dependencies handled; semantic equality before/after rewriting; no dangling definitions; resumed deterministic runs identical. **Resources:** profile cache and AST growth versus evidence; no performance result is accepted before semantic checks. **Acceptance:** zero differential errors and generic optimizations available to all relevant variants. **Branches:** bugs return to W01/W04; repair provides no gain over cached global search at current size means retain the simpler route as a strong baseline and investigate whether the task actually requires revision.

### W05 — Admission policies and resource pilot

**Objective/risk:** determine whether selection overhead and local capacity permit a meaningful H1 test. **Dependencies:** W03–W04. **Actions:** implement C/U/G/R exactly as specified, plus random and size-matched controls; enforce E/V separation; log every replay input and candidate origin. Measure cost components across grammar size, library size, evidence count, object count and batch size. Generate balanced pilot curricula and run both resource regimes.

**Outputs:** candidate/admission logs, peak-memory models, budget curves, source/target learning curves and preliminary variance estimates. **Tests:** candidate generation never sees V solutions; replay never sees future labels; source selection cost is charged; native versus structural compression units are distinct; library dependencies are closed. **Resources:** empirical Section 6 model; fail safely before memory pressure. **Acceptance:** an informative operating region exists and its total-cost/precision requirements are measurable. **Branches:** no correction opportunities means revise the task; setup dominates without plausible measured break-even means narrow/stop the practical claim; a tiny/inconsistent effect means plan adequate precision or label the prospective test underpowered, not intensify favorable slicing.

### W06 — Adversarial attribution and scope checks

**Objective/risk:** explain away apparent pilot improvements before confirmation. **Dependencies:** W05. **Actions:** run Section 9 controls, particularly R versus G, cardinality matching, generic-cache sharing, random order, no reuse, foreign syntax and increasing distractor counts. Recheck the refinement against the closest sources if the actual implementation differs from the planned one.

**Outputs:** attribution matrix, corrected specification and explicit claim boundary. **Tests:** determine whether pilot gains require the claimed correction path or merely a smaller library/better initial fit. **Resources:** targeted low-cost counterexamples before broad sweeps. **Acceptance:** either a defensible remaining claim or a documented negative/replication direction. **Branches:** an explained-away gain is renamed or rejected; a new mechanism requires new development and a revised hypothesis, not relabeling the old result as confirmation.

### W07 — Freeze the confirmatory experiment

**Objective/risk:** prevent adaptive evaluation and unpowered claims. **Dependencies:** W00–W06 accepted or explicit scope exceptions recorded. **Actions:** compute the fixed N/precision plan; select all method configurations and allocation policies; freeze exact data distributions, exclusions, metrics, multiplicity treatment, timing procedure and failure rules; record independent protected seeds and all hashes. The primary contrast is R–G.

**Outputs:** machine-readable freeze manifest, human-readable preregistration and its SHA-256. **Tests:** validator rejects unresolved fields, missing capability coverage or a mismatched source revision; dry-run the analysis on fake/known-null results. **Resources:** analytic/resampling work plus a complete runtime/disk projection. **Acceptance:** confirmation is both scientifically interpretable and locally executable, or is explicitly declared precision-limited before results. **Branches:** unresolved baseline or power issue blocks the corresponding claim; do not access protected outcomes to choose a workaround.

### W08 — Controlled confirmation

**Objective/risk:** test H1 and assess competitiveness under the frozen protocol. **Dependencies:** W07. **Actions:** execute the fixed balanced curriculum set; counterbalance method order; collect all outputs and resource traces; monitor only correctness/resource failures, not treatment-effect significance; checkpoint after each observation/chunk. Run the frozen analysis once the dataset is complete.

**Outputs:** complete raw runs, primary/secondary estimates and intervals, budget accounting, per-cell results and failure rates. **Resources:** calibrated envelopes and fixed N; sequential local scheduling. **Acceptance:** all scheduled runs accounted for and analysis matches the freeze. **Branches:** implementation failure invalidates affected paired comparisons and invokes a recorded correction/fresh confirmation; legitimate timeouts/unsolved cases remain outcomes; an interval covering useful gains and losses is inconclusive; negative results are retained.

### W09 — Fresh confirmation, robustness and external transfer

**Objective/risk:** test reproducibility beyond the exact pilot/first-confirmation assets. **Dependencies:** a valid W08 analysis; not necessarily a positive result. **Actions:** repeat the frozen primary contrast on fresh source and target curricula when needed to substantiate a positive claim; run prespecified boundary conditions; execute the external symbolic-task gate; add active queries only as a separate experiment. Any exploratory post-W08 change gets new confirmation.

**Outputs:** independent-seed replication, robustness tables, external capability report and active-query cost curves when run. **Resources:** informed by W08 uncertainty and the scientific claim; no automatic large extension. **Acceptance:** replication direction/effect range is compatible with the claim, or the inconsistency is explicitly reported. **Branches:** synthetic-only success narrows scope; external mismatch prompts a representation diagnosis; a reversal requires explanation and prevents a general advantage claim.

### W10 — Reproducibility and interruption audit

**Objective/risk:** ensure results survive outside the original process state. **Dependencies:** any complete candidate release after W08/W09. **Actions:** reconstruct environments from locks on the same laptop in fresh directories; replay recorded inputs; interrupt and resume symbolic, solver and neural runs; verify hashes, ledger totals and statistical outputs. This is a local repeatability audit, not independent third-party review.

**Outputs:** reproducibility report, manifest checksums, failure reproductions and a minimal release script. **Tests:** exact symbolic outputs under deterministic budgets; stated stochastic tolerances; all failed/unfavorable runs present; no unpublished dependency state. **Resources:** a representative stratified run subset plus complete analysis rerun. **Acceptance:** another agent can reproduce the reported tables from raw files and the release command. **Branches:** nondeterminism/serialization problems block the corresponding numerical claim until repaired or bounded transparently.

### W11 — Synthesis and decision

**Objective/risk:** avoid converting a narrow or negative result into a sweeping architectural story. **Dependencies:** available valid results and all audit records. **Actions:** apply Section 10 decision rules; write the final scientific report with exact local claims, confidence intervals, costs, failure regions, supplied biases and prior-art classification. Update novelty assessment only from evidence.

**Outputs:** `artifacts/reports/final_report.md`, machine-readable conclusions, reproducible figures/tables and next-decision record. **Resources:** analysis/reporting, not additional result-seeking experiments. **Acceptance:** every claim maps to a baseline, metric, population and run manifest; estimates and unperformed extensions remain labeled. **Branches:** continue only when the next experiment resolves a named uncertainty; simplify when ordinary methods suffice; stop the selected mechanism when valid adequately sensitive evidence is against its value.

## 9. Ablation, robustness and failure-analysis program

Do not run an unbounded factorial experiment by default. Begin with targeted matched contrasts on development data, then preregister the subset necessary for each intended claim.

| Challenge | Matched experiment | Interpretation |
|---|---|---|
| Compression alone suffices | R versus C and native-cost Stitch, same disclosed information/cost boundaries. | A loss or tie defeats the corresponding added-complexity claim. |
| Generic validation explains the gain | R versus U and especially G. | Beating C/U but not G supports generic prefix-aware validation, not a correction-specific distinction. |
| Smaller library explains it | Match dependency-closed library node cost, definition count and candidate pool; include random admission and validation-pruned C. | Surviving gain must exceed simple capacity reduction. |
| Caching explains it | Same evaluator, indexes, memoization and warm-start opportunities; include cached global solver. | A kernel optimization is engineering, not evidence for new intelligence. |
| Better initial fit explains it | Separate first-fit and post-trigger portions using common preregistered trigger positions. | A gain only in first-fit cannot support H2. |
| Favorable curriculum order | Interleaved, ordinary-first and reversed source orders, with no hidden-rule change. | An order-specific result needs that restriction in its claim. |
| Transfer is memorization | Exact lookup and parameterized retrieval; new program compositions; ID/color relabeling with transformed outputs. | Loss of benefit exposes leaked identity or narrow reuse. |
| Representation favors the miner | Equivalent algebraic/logical forms and alpha-renaming; shared canonicalization; consider babble-style equivalence-aware mining if material. [R18] | A syntactic artifact cannot be sold as robust abstraction discovery. |
| Hidden teacher bias | Independent imperative teacher; expose full bias inventory; common finite-feature and raw-relational tracks. | Success confined to G0 is a G0 result, not autonomous perception or universal rule discovery. |
| Sparse versus tangled dependency | Vary proportion of outputs/motifs depending on shared guards while holding evidence and cost envelopes fixed. | Repair should lose its expected local-work advantage when dependencies become dense; measure rather than assume. |
| More compute/information | Matched total budget; same source labels; exact acquisition ledger; larger baseline budgets on a tradeoff curve. | Reclassify any unequal-resource gain. |
| Approximate frontier overconfidence | Expand recoverable frontier/search bounds and inject a late distinguishing case. | Confidence collapse or lost alternatives exposes search truncation, not epistemic certainty. |
| Tiny, unstable effects | Fresh curricula, paired uncertainty, neighboring budgets and setup-inclusive break-even. | A real but negligible effect is not a breakthrough. |

### Noise, drift and latent state are three different adversaries

**Label noise:** corrupt a declared fraction of successor labels using a seeded mechanism. Preserve clean labels only in the evaluator. Use a soft objective such as `number of incorrect observed output fields + lambda * structural_cost`, with lambda and corruption model chosen on development data; compare to a validated noisy ILP route and robust statistical baselines. Do not reuse clean-data elimination rules that become unsound for this objective. The noisy branch is an additional experiment, not a fix applied only to R. [R04]

**Law drift:** change the hidden rule at a logged but observer-hidden boundary. Compare explicit windowed/discounted evidence, segmented models and a change-detection baseline. An append-only deterministic version space is expected to become inconsistent. Keeping all contradictory labels while demanding one static deterministic function is a modeling error, not evidence of weak intelligence.

**Latent state/partial observability:** mask a causally relevant variable or add an unobserved timer. Construct identical observed state/action pairs with different outputs. The correct diagnosis is insufficient state information. A memory/state-inference extension needs its own declared hypothesis language and equal observation history for competitors; do not retroactively invent an unobserved feature.

Additional adversarial tests include: all examples agreeing while hypotheses disagree elsewhere; logically equivalent teacher programs; rare exceptions never observed; a macro valid on source data but harmful on targets; cyclic macro references; empty collections in generalized schemas; negative evidence about absence; object reordering; stale caches after a definition version changes; source libraries much larger than target demand; constraints that admit many solutions; and solver `unknown` confused with `unsat`.

## 10. Outcome interpretation and evidence-triggered pivots

| Observed result | Valid conclusion | Next action |
|---|---|---|
| R improves the primary endpoint with supported uncertainty, beats strong applicable controls, and repays setup cost | A correction-aware admission policy provides a measured advantage in the stated symbolic setting. | Fresh-source replication; then a separately frozen external or richer-grammar test. Do not claim general AI replacement. |
| R beats compression but not generic prefix selection G | Sequential validation is useful; no isolated correction-specific advantage has been established. | Prefer the simpler best policy. Report an empirical objective-alignment result. |
| All learned libraries tie retrieval or a small tree | The task does not justify synthesis/library complexity at this scale. | Keep the simpler learner; expand the task only if a specified capability is absent from the proxy. |
| Correct implementation, informative workload, precise null/negative result | Evidence against the selected policy's local value. | Stop or revise this policy, not all symbolic intelligence. A new policy needs a new hypothesis and evaluation. |
| Wide intervals or too few independent curricula | Inconclusive effect range, not “no difference.” | Improve sensitivity or obtain the fixed additional confirmation justified by a new plan; do not claim significance by optional continuation. |
| Search cannot express/find the necessary rule | Distinguish missing primitive, insufficient bound, solver failure and lack of observations. | Diagnose with witness ASTs and exhaustive small cases; amend G1 only when the mechanism requires it. |
| Gains exist only after large library growth or dense interaction | A scale-dependent regime may exist, but local evidence is bounded. | Escalate only when measured scaling curves show an emerging effect and predicted cost fits the laptop. Do not invoke hypothetical future scale to excuse every failure. |
| Mechanism works but overhead dominates | Component capability without end-to-end benefit. | Retain only if an observed reuse horizon or application can repay the cost; otherwise simplify. |
| Local baseline unavailable | Comparison coverage is incomplete. | Make a narrower direct claim using validated strong local competitors; keep nonlocal published systems contextual. Do not replace a large API model with a small local model and call it faithful reproduction. |
| Synthetic success does not transfer externally | The supplied representation/curriculum may be essential. | Diagnose domain bias and narrow the claim. A perception adapter becomes a distinct hybrid project with separately charged assets. |

An ambitious continuation, if justified, is to learn reusable structures that improve both discovery and revision across independently designed symbolic domains, with source and target distributions no longer sharing a hand-built motif generator. A further move toward recursive procedures or new state representations must specify termination, observation access and verification. These are research directions, not promised capabilities of this PoC.

## 11. Audit record and residual limitations

The following are **self-audit findings from preparing this plan**, not independent peer review or completed target-laptop validation.

### 11.1 Scientific and prior-art audit

| Defect found | Revision made | Residual risk/acceptance criterion |
|---|---|---|
| The original executable-theory architecture closely overlaps Apperception, ILP and theory revision. | Narrowed the claim to correction-aware admission; retained replication/negative outcomes. | Methodological novelty remains unresolved; an exact predecessor reclassifies the contribution. |
| Prospective abstraction utility is also prior art. | Added generic terminal and all-prefix utility controls; made **R–G**, not R–compression, the primary contrast. | A positive result may still be ordinary validation-policy engineering. |
| Definitional macros could be described misleadingly as new primitives. | Required expansion, language-bias disclosure and inlined cost reporting. | No evidence here of learning semantics outside the supplied language. |
| The broad grammar might make the laptop experiment an intractable search project. | Specified finite G0 atoms, bounded normal forms, bitset pruning and a separate G1 escalation. | G0 supplies strong bias; useful throughput still requires local calibration. |
| Physical/evolutionary references could be mistaken for non-neural general intelligence evidence. | Kept them contextual and did not treat API-based evolutionary proposals as purely non-neural. | No hardware-energy or broad capability claim is validated. |

### 11.2 Experimental-validity audit

| Defect found | Revision made | Residual risk/acceptance criterion |
|---|---|---|
| Mining from a task's final solution then replaying its prefix leaks the answer. | Separate E donors from V replay; audit candidate provenance and all replay calls. | No candidate may depend on its replay task's solution before acquisition. |
| Warm-cache speedups could explain a result. | Share caches and include cached global synthesis; separate full-system and deterministic-work comparisons. | An advantage erased by shared optimization is not attributed to admission. |
| An ordinary-first stream could manufacture a win. | Add random interleaving/no-reuse cells and explicit population weights. | Order-specific gains must be reported as such. |
| Multiple programs agreeing now could be irreversibly merged. | Recoverable equivalence buckets and exhaustive finite checks. | No unsound removal of alternatives; truncation must be visible. |
| Initial ring fixture allowed a repeated query to an already observed support state. | Removed that input from the acquisition pool before the reported check; random-query expectation is now 16/5, not 17/5. | The fixture remains illustrative, not evidence of benchmark superiority. |
| Ignoring selection/training costs or failures would flatter the proposal. | Primary matched total budget; retained timeouts/abstentions; setup-inclusive break-even. | Every scheduled run and cost boundary must reconcile in the ledger. |
| Uniform enumeration and a direct neural predictor omit stronger search alternatives. | Added Eco Search/Probe and the conditional AbstractBeam comparison, with common-language and native-compatibility gates. | No broad synthesis/neural superiority claim from a panel missing its applicable strongest competitors. |
| A neural baseline could be weakened by lack of world conditioning. | Specified support-conditioned relational learning and source/adaptation curves. | Insufficient locally achievable training prevents a broad neural-superiority claim. |
| Treating timings, seeds or transitions as independent exaggerates evidence. | Curriculum-level inference, fixed sample design and multiplicity handling. | Small or dependent sample uncertainty must remain visible. |

### 11.3 Implementation and feasibility audit

| Defect found | Revision made | Residual risk/acceptance criterion |
|---|---|---|
| Assuming one Python environment would conflict with current Popper requirements. | Separate environment; source-inspected Python >=3.14 requirement and native Janus test. | Compiled dependency chain remains an early blocking gate. |
| Nominal Apple Silicon support does not guarantee the needed kernel. | Operation-level CPU/MPS and solver smoke tests; faithful CPU fallback. | No M3 performance claim before actual measurements. |
| Mutating a shared definition or omitting negative dependencies could corrupt prior knowledge. | Immutable hashes, dependency-closed bundles, full-evidence checks and non-interference tests. | Scope prioritizes repair only; no unjustified global-optimum guarantee. |
| Stitch's native weighting can differ from declared structural units. | Explicit rescore, native-objective comparator and separate units. | Adapter must pass semantic and cost-accounting tests before comparisons. |
| Library proposals can refer to ancestors a selector rejected. | Admission is dependency-closed and charges all ancestors. | Dangling/cyclic references block acceptance. |
| Original public repositories do not establish reuse permission or easy native builds. | License/access gates and clearly labeled conditional reproductions. | Closest-original comparison may remain unavailable. |
| Interruptions can reset clocks, reuse future labels or change RNG state. | Transactional checkpoints, remaining-budget ledger and strict resume hashes. | Replay-equivalence tests block confirmation. |
| Configuration placeholders could make the handoff non-executable. | Explicit calibration-to-freeze path and a validator that rejects unresolved confirmatory fields. | Workload sizes and N are intentionally not fabricated before hardware/pilot measurements. |

A final cross-check of the written protocol must verify that the primary contrast, data profile, units, stopping rule and baseline scope agree across configurations, code and report. Any new defect discovered during implementation enters the same ledger; scientific gates are not bypassed to preserve a schedule.

### 11.4 Remaining limits

This plan cannot establish that a new architecture already outperforms existing methods; no such benchmark has been run. The current evidence is strongest about **prior-art overlap and implementable experimental distinctions**, not positive performance. There is no exhaustive novelty proof, patentability conclusion or freedom-to-operate opinion. The user supplied no application utility function, so practical significance is reported through error/cost/reuse-horizon tradeoffs rather than invented financial values.

The local experiment supplies object structure, action semantics, a finite color/step alphabet and a substantial grammar bias. G0 omits boundary learning, general recursion, open-ended state invention, raw sensory interpretation and non-resettable exploration. A valid negative G0 result can reject this selected local policy, but not every possible non-neural learning mechanism. Conversely, a positive G0 result cannot validate those omitted capabilities.

## 12. References, inspection record and search coverage

### 12.1 Search scope and stopping assessment

Research was conducted for this handoff on **14 October 2026**. The search followed the original proposal into program induction, inductive logic programming, causal-theory induction, truth maintenance, incremental theory revision, program repair, counterexample-guided/oracle-guided synthesis, active learning, abstraction and predicate invention, knowledge refactoring, library growth/forgetting, prospective abstraction utility, and cost- or neural-guided synthesis. Physical reservoir computing and evolutionary algorithm discovery were checked to delimit the unselected alternatives rather than to claim a hardware comparison.

Representative query groups included “Apperception Engine incremental learning,” “inductive logic programming theory revision predicate invention,” “program synthesis library learning repair,” “knowledge refactoring Knorf STEVIE,” “prospective compression abstraction learning,” “online library evolution,” “active oracle guided synthesis,” “bottom up synthesis Probe Bee Eco Search,” and Apple Silicon/native installation queries for the actual dependencies. Citation tracing connected Apperception to ILP/theory revision, DreamCoder to Stitch/babble and AbstractBeam, knowledge refactoring to higher-order invention, and Bee Search to Eco Search. Current searches found 2025–2026 work on prospective compression, planning abstractions, online libraries and active neurosymbolic learning. These newer sources reduced, rather than strengthened, the claim of broad architectural novelty.

A limited patent search included program inference, library learning and incremental program revision. One inspected example, Microsoft’s *Multi-modal program inference*, concerns language-model/component-based program inference and ranking, not this exact non-neural admission policy. Its independent claim was inspected only to identify adjacent scope. This is not a patent search opinion, legal claim construction, patentability finding or freedom-to-operate assessment. [R31]

Programming-by-example work on spreadsheet transformations provides an established application precedent, not proof that the proposed learner solves a validated customer problem. A more recent responsive Apperception implementation is further evidence that faster theory induction is already an engineering objective. Neither source supplies controlled evidence for H1. [R30][R32]

The last targeted search produced an important correction: a modern guided-search comparator was missing, and has now been added. Subsequent design checks did not require another architectural change. This is a **bounded coverage assessment**, not proof that no equivalent policy exists. Remaining blind spots include unpublished work, non-indexed code and dissertations, terminology outside the searched traditions, detailed patent-family histories, and research appearing after the search date. The exact correction-aware policy may already be an application of an existing general utility framework.

### 12.2 Inspection and access boundaries

Full text was inspected for the Apperception evaluation, MaxSynth, ILED, Stitch, Knorf, DreamCoder, prospective compression, TheoryCoder-2, SmartLabel, Probe, Eco Search, AbstractBeam, STEVIE and Herb.jl. PDF pages were rendered where needed for consequential visual/table inspection, including the Apperception evaluation, Stitch, MaxSynth and STEVIE. Other entries explicitly identify abstract, publisher-record or repository-only access. A reference's presence does not imply full-text inspection, independent validation of its reported results, or reproduction.

Implementation inspection included the Popper README, `pyproject.toml`, launcher and solver notes; the Apperception README/build route; Stitch's repository and Python API tutorial; DeepSynth's README/project metadata; Probe's README; AbstractBeam's README/dependency route; and selected ILED/STEVIE/online-library repository descriptions. Repositories were **not cloned, built or benchmarked here**. Some direct downloads and full-paper access attempts failed; no missing source contents are invented. The only executed code is the independent semantic check in Appendix A.

Each reference below identifies its role and access level. Mutable repository URLs are intentional discovery locations; W00 must record the actual checked-out commit and applicable licenses. Dates on mutable documentation are inspection dates unless a release date is explicitly stated.

### 12.3 Bibliography and evidence roles

**[R01]** Richard Evans, José Hernández-Orallo, Johannes Welbl, Pushmeet Kohli and Marek Sergot. *Making sense of sensory input*. 2019, arXiv:1910.02227. [Author preprint record](https://arxiv.org/abs/1910.02227). **Inspected:** abstract/record. **Role:** closest broad architecture, executable theories and explanatory constraints. **Limit:** conceptual overlap does not establish performance in this proposed task.

**[R02]** Richard Evans and colleagues. *Evaluating the Apperception Engine*. 2020, arXiv:2007.05367. [Record](https://arxiv.org/abs/2007.05367); [author-hosted full paper](https://www.doc.ic.ac.uk/~re14/evaluating-the-apperception-engine.pdf); [original implementation](https://github.com/RichardEvans/apperception). **Inspected:** full evaluation, rendered page, repository/build instructions. **Role:** exact architecture comparison and conditional reproduction. **Limit:** reported symbolic-task conditions and Unix implementation are not M3 results; reuse license unresolved here.

**[R03]** Andrew Cropper and Rolf Morel. *Learning programs by learning from failures*. 2020 preprint, subsequently Machine Learning, 2021; arXiv:2005.02259. [Record](https://arxiv.org/abs/2005.02259). **Inspected:** original record/abstract, with implementation separately inspected in R16. **Role:** failure-derived constraints and a strong symbolic competitor. **Limit:** hypothesis bias and logic semantics must match the comparison.

**[R04]** Céline Hocquette, Andreas Niskanen, Matti Järvisalo and Andrew Cropper. *Learning MDL Logic Programs from Noisy Data*. AAAI 2024. DOI: [10.1609/aaai.v38i9.28925](https://doi.org/10.1609/aaai.v38i9.28925). [Publisher record](https://ojs.aaai.org/index.php/AAAI/article/view/28925); [full paper](https://ojs.aaai.org/index.php/AAAI/article/view/28925/29759). **Inspected:** full paper and rendered page. **Role:** noisy-data objective and pruning distinctions. **Limit:** does not license applying clean elimination rules to arbitrary noisy objectives.

**[R05]** Jon Doyle. *A Truth Maintenance System*. MIT Artificial Intelligence Memo 521, 1979. [MIT archival record](https://dspace.mit.edu/entities/publication/5377b306-4ecc-4687-b1f5-78cbb4a0543a). **Inspected:** archival abstract. **Role:** established belief dependencies and revision. **Limit:** no target-performance inference from this historical precedent.

**[R06]** Nikos Katzouris, Alexander Artikis and Georgios Paliouras. *Incremental learning of event definitions with Inductive Logic Programming*. Machine Learning, 2015. DOI: [10.1007/s10994-015-5512-1](https://doi.org/10.1007/s10994-015-5512-1). [Article](https://link.springer.com/article/10.1007/s10994-015-5512-1); [ILED repository](https://github.com/nkatzz/ILED). **Inspected:** full article and repository. **Role:** incremental revision and support information. **Limit:** Event Calculus assumptions and older infrastructure do not automatically transfer to arbitrary AST learning.

**[R07]** Matthew Bowers, Theo X. Olausson, Catherine Wong, Gabriel Grand, Joshua B. Tenenbaum, Kevin Ellis and Armando Solar-Lezama. *Top-Down Synthesis for Library Learning*. POPL 2023. DOI: [10.1145/3571234](https://doi.org/10.1145/3571234). [Author full paper](https://mlb2251.github.io/stitch.pdf). **Inspected:** full paper and rendered page. **Role:** abstraction candidate generation and established utility-based search. **Limit:** compression results are not evidence for correction-aware admission.

**[R08]** Leonardo Hernandez Cano and colleagues. *Prospective Compression in Human Abstraction Learning*. May 2026 preprint, arXiv:2605.09985. [Full version inspected](https://arxiv.org/html/2605.09985v1). **Inspected:** full text. **Role:** anticipated future usefulness already motivates abstraction choice. **Limit:** human Pattern Builder/computational-model setting, not this engine or a direct laptop baseline.

**[R09]** Sebastijan Dumančić, Tias Guns and Andrew Cropper. *Knowledge Refactoring for Inductive Program Synthesis*. AAAI 2021. DOI: [10.1609/aaai.v35i8.16893](https://doi.org/10.1609/aaai.v35i8.16893). [Publisher record](https://ojs.aaai.org/index.php/AAAI/article/view/16893); [full preprint, titled “Knowledge Refactoring for Program Induction”](https://arxiv.org/html/2004.09931v1). **Inspected:** full preprint and publisher record. **Role:** refactoring learned background knowledge is prior art. **Limit:** reported domains/objective differ from sequential correction admission.

**[R10]** Zergham Ahmed, Kazuki Irie, Joshua B. Tenenbaum, Christopher J. Bates and Samuel J. Gershman. *Learning Abstractions for Hierarchical Planning in Program-Synthesis Agents*. 2026 preprint, arXiv:2602.00929. [Full version inspected](https://arxiv.org/html/2602.00929v1). **Inspected:** full text. **Role:** current planning/library architecture overlap. **Limit:** TheoryCoder-2 uses pretrained/API language-model capabilities; not a faithful purely non-neural local comparison.

**[R11]** Google DeepMind, AlphaEvolve team. *AlphaEvolve: A Gemini-powered coding agent for designing advanced algorithms*. 14 May 2025. [Official technical announcement](https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/). **Inspected:** official account. **Role:** distinguishes evolutionary evaluation from the neural proposal mechanism. **Limit:** no independent reproduction or controlled comparison in this handoff.

**[R12]** Gouhei Tanaka and colleagues. *Recent Advances in Physical Reservoir Computing: A Review*. Neural Networks 115, 2019, pp. 100–123. DOI: [10.1016/j.neunet.2019.03.005](https://doi.org/10.1016/j.neunet.2019.03.005); [author preprint record](https://arxiv.org/abs/1808.04962). **Inspected:** abstract/record. **Role:** contextualizes the unselected hardware branch. **Limit:** a review, not direct primary experimental evidence of a proposed substrate advantage; no such advantage is asserted.

**[R13]** Andrew Cropper. *Forgetting to Learn Logic Programs*. AAAI 2020. [Publisher record](https://ojs.aaai.org/index.php/AAAI/article/view/5776). **Inspected:** record/abstract. **Role:** library retention and search-space management precedents. **Limit:** exact algorithm and benchmark reproduction not performed.

**[R14]** Kevin Ellis and colleagues. *DreamCoder: Growing generalizable, interpretable knowledge with wake-sleep Bayesian program learning*. 2020 preprint, arXiv:2006.08381. [Full version inspected](https://arxiv.org/html/2006.08381v1). **Inspected:** full preprint. **Role:** neural-guided synthesis and abstraction learning. **Limit:** paper version named explicitly; its learned guidance and task distributions are not equated to this non-neural system.

**[R15]** Susmit Jha, Sumit Gulwani, Sanjit A. Seshia and Ashish Tiwari. *Oracle-Guided Component-Based Program Synthesis*. ICSE 2010, pp. 215–224. DOI: [10.1145/1806799.1806833](https://doi.org/10.1145/1806799.1806833). [Author publication record](https://people.eecs.berkeley.edu/~sseshia/pubs/b2hd-jha-icse10.html). **Inspected:** author record/abstract. **Role:** interactive distinguishing examples and SMT synthesis. **Limit:** loop-free component language and oracle assumptions differ.

**[R16]** Logic and Learning Lab. *Popper*, current public implementation, inspected 14 October 2026. [Repository](https://github.com/logic-and-learning-lab/Popper); [project metadata](https://raw.githubusercontent.com/logic-and-learning-lab/Popper/main/pyproject.toml). **Inspected:** README, project metadata, launcher and solver notes. **Role:** baseline execution, Python/native dependency gate and examples. **Limit:** inspected source, not installed software; exact commit must be pinned locally.

**[R17]** Stitch contributors. *Stitch* and Python bindings documentation, inspected 14 October 2026. [Repository](https://github.com/mlb2251/stitch); [Python tutorial](https://stitch-bindings.readthedocs.io/en/stable/intro/tutorial.html). **Inspected:** repository and API tutorial. **Role:** concrete candidate-mining/rewrite reuse. **Limit:** documented native cost units and API must be tested against the installed version; this plan's structural objective is distinct.

**[R18]** David Cao, Rose Kunkel, Chandrakana Nandi, Max Willsey, Zachary Tatlock and Nadia Polikarpova. *babble: Learning Better Abstractions with E-Graphs and Anti-Unification*. POPL 2023. DOI: [10.1145/3571207](https://doi.org/10.1145/3571207); [preprint record](https://arxiv.org/abs/2212.04596); [artifact location](https://github.com/dcao/babble). **Inspected:** abstract and artifact availability, not a full reproduction. **Role:** representation-sensitive library learning control. **Limit:** conditional implementation only after that failure mode is observed.

**[R19]** Andrew Cropper. *Playgol: Learning Programs Through Play*. IJCAI 2019. [Proceedings record](https://www.ijcai.org/proceedings/2019/841). **Inspected:** publication record. **Role:** non-neural accumulated reusable procedures. **Limit:** not direct evidence for this selection policy.

**[R20]** Tingting Yu, Pei-Cing Huang, Chan Hsu, Chan-Tung Ku and Yihuang Kang. *ADVENT: LLM-Driven Automatic Predicate Invention for Inductive Logic Programming*. July 2026 preprint, arXiv:2607.01585. [Record](https://arxiv.org/abs/2607.01585). **Inspected:** abstract. **Role:** current predicate-invention/verification overlap. **Limit:** neural proposals and unreplicated preprint evidence.

**[R21]** Chahel Paatur. *Self-Modifying Program Synthesis via Online Library Evolution: A Mechanistic Approach to Few-Shot Abstract Reasoning with Aerospace Validation*. 2026 preprint. DOI: [10.36227/techrxiv.176784644.48617663](https://doi.org/10.36227/techrxiv.176784644.48617663); [author repository](https://github.com/chahelpaatur/Self-Modifying-Program-Synthesis-via-Online-Library-Evolution). **Inspected:** repository/README; direct full-preprint access failed. **Role:** flags recent online-library overlap. **Limit:** reported results are not verified or used quantitatively.

**[R22]** Celeste Barnaby, Qiaochu Chen, Ramya Ramalingam, Osbert Bastani and Işıl Dillig. *Active Learning for Neurosymbolic Program Synthesis*. PACMPL 9, OOPSLA2, 2025. DOI: [10.1145/3763102](https://doi.org/10.1145/3763102); [full preprint](https://arxiv.org/html/2508.15750v1). **Inspected:** full text. **Role:** active acquisition and uncertainty controls. **Limit:** SmartLabel's neural uncertainty is not the same as a complete symbolic version space.

**[R23]** cvc5 developers. *Syntax-guided synthesis example, Python API and distribution metadata*. Inspected 14 October 2026. [Official example](https://cvc5.github.io/docs/latest/examples/sygus-fun.html); [Python API](https://cvc5.github.io/docs/latest/api/python/python.html); [PyPI release metadata](https://pypi.org/project/cvc5/). **Inspected:** example/API and release/wheel listings. **Role:** solver encoding and native installation gate. **Limit:** wheel availability does not verify the target encoding or interruption semantics.

**[R24]** Apple. *Accelerated PyTorch training on Mac*. [Official developer documentation](https://developer.apple.com/metal/pytorch/), inspected 14 October 2026. **Inspected:** official support documentation. **Role:** MPS route and backend checks. **Limit:** no blanket inference about scatter, attention, precision or every optimizer operation.

**[R25]** XGBoost developers. *Installation Guide*. [Official documentation](https://xgboost.readthedocs.io/en/stable/install.html), inspected 14 October 2026. **Inspected:** installation/platform guidance. **Role:** CPU tree baseline on macOS. **Limit:** exact package build and runtime remain to be tested.

**[R26]** Z3 developers/package maintainers. *z3-solver distribution metadata*. [PyPI](https://pypi.org/project/z3-solver/), inspected 14 October 2026. **Inspected:** release and wheel information. **Role:** optional independent solver fallback. **Limit:** not a successful local installation or execution.

**[R27]** Apple. *MacBook Pro (16-inch, Nov 2023): Technical Specifications*. [Official specification](https://support.apple.com/en-us/117737). **Inspected:** specification page. **Role:** memory configurations vary; use actual hardware discovery. **Limit:** the user's size/configuration is not established from this reference.

**[R28]** SWI-Prolog/Janus maintainers. *Building/installing SWI-Prolog on macOS* and *janus-swi*. [macOS documentation](https://www.swi-prolog.org/build/macos.html); [Janus package record](https://pypi.org/project/janus-swi/), inspected 14 October 2026. **Inspected:** installation and build requirements. **Role:** Popper's native bridge. **Limit:** interpreter, architecture and library-path compatibility are local blocking tests.

**[R29]** Brenden M. Lake, Ruslan Salakhutdinov and Joshua B. Tenenbaum. *Human-level concept learning through probabilistic program induction*. Science 350, 2015. DOI: [10.1126/science.aab3050](https://doi.org/10.1126/science.aab3050); [PubMed record](https://pubmed.ncbi.nlm.nih.gov/26659050/). **Inspected:** abstract/record; publisher full text unavailable in this session. **Role:** verifies the original idea's specific character-learning precedent. **Limit:** not evidence of general-purpose non-neural parity.

**[R30]** Sumit Gulwani. *Automating String Processing in Spreadsheets Using Input-Output Examples*. POPL 2011. [Microsoft Research publication record](https://www.microsoft.com/en-us/research/publication/automating-string-processing-spreadsheets-using-input-output-examples/). **Inspected:** original organization publication record. **Role:** useful programming-by-example application precedent. **Limit:** not a dataset or user-demand study for this PoC.

**[R31]** Kiarash Rahmani, Mohammad Raza, Sumit Gulwani and colleagues; assignee Microsoft Technology Licensing LLC. *Multi-modal program inference*. US11934801B2, publication 19 March 2024. [Patent text and metadata](https://patents.google.com/patent/US11934801B2/en). **Inspected:** metadata and independent claim 1. **Role:** adjacent patent landscape example. **Limit:** no conclusion about validity, enforceability, current legal status or the legal scope of the proposed system.

**[R32]** Jean-François Cloutier. *Finding causal theories quickly enough — building a responsive Apperception Engine*. Version 2, 24 May 2025. DOI: [10.5281/zenodo.15512255](https://doi.org/10.5281/zenodo.15512255); [record](https://zenodo.org/records/15512255); [implementation location](https://github.com/jfcloutier/karma_apperception). **Inspected:** record/description and repository. **Role:** responsive non-neural theory induction is already an implementation objective. **Limit:** no independent performance verification.

**[R33]** Shraddha Barke, Hila Peleg and Nadia Polikarpova. *Just-in-Time Learning for Bottom-Up Enumerative Synthesis*. OOPSLA 2020. DOI: [10.1145/3428295](https://doi.org/10.1145/3428295); [full preprint](https://arxiv.org/html/2010.08663v1); [Probe implementation](https://github.com/shraddhabarke/Probe). **Inspected:** full text and original README/license statement. **Role:** online probabilistic guidance from partial solutions. **Limit:** original Scala/CVC4-era artifact is not established as an M3-native ready-to-run package.

**[R34]** Saqib Ameen and Levi H. S. Lelis. *Program Synthesis with Best-First Bottom-Up Search*. Journal of Artificial Intelligence Research 77, 2023, pp. 1275–1310. DOI: [10.1613/jair.1.14394](https://doi.org/10.1613/jair.1.14394); [preprint record](https://arxiv.org/abs/2310.04327); [Bee Search repository](https://github.com/lelis-research/BeeSearch). **Inspected:** record/abstract and repository. **Role:** best-first predecessor and additional implementation alternative. **Limit:** GPL-3.0 repository conditions and original task language must be respected; no published speedup is transferred here.

**[R35]** Théo Matricon, Nathanaël Fijalkow and Guillaume Lagarde. *Eco Search: A Constant-Delay Best-First Search Algorithm for Program Synthesis*. 2024 preprint, arXiv:2412.17330; AAAI 2025 proceedings title uses “No-delay.” DOI: [10.1609/aaai.v39i18.34139](https://doi.org/10.1609/aaai.v39i18.34139). [Full preprint](https://arxiv.org/html/2412.17330v1); [DeepSynth implementation](https://github.com/SynthesisLab/DeepSynth); [project metadata](https://raw.githubusercontent.com/SynthesisLab/DeepSynth/main/pyproject.toml). **Inspected:** full text, algorithm sections, README and metadata. **Role:** modern guided enumeration comparator/reuse. **Limit:** enumeration-delay claims have specified cost assumptions and do not bound arbitrary semantic evaluation or total learning time.

**[R36]** Janis Zenkner, Lukas Dierkes, Tobias Sesterhenn and Christian Bartelt. *AbstractBeam: Enhancing Bottom-Up Program Synthesis using Library Learning*. 2024 preprint, arXiv:2405.17514v2. [Full text](https://arxiv.org/html/2405.17514v2); [author implementation](https://github.com/jzenkner/AbstractBeam). **Inspected:** full text and repository README. **Role:** neural execution-guided library learner as a closer comparison class. **Limit:** integer-list domain, historical scatter dependencies and GPU launcher; no faithful laptop reproduction claimed.

**[R37]** Céline Hocquette, Sebastijan Dumančić and Andrew Cropper. *Learning Logic Programs by Discovering Higher-Order Abstractions*. IJCAI 2024, pp. 3421–3429. [Full paper](https://www.ijcai.org/proceedings/2024/0379.pdf); [STEVIE implementation](https://github.com/celinehocquette/ijcai24-stevie). **Inspected:** full text, rendered first page and repository. **Role:** higher-order abstraction/refactoring prior art. **Limit:** G0 does not exercise its higher-order language; no contrived head-to-head on an inexpressible subset.

**[R38]** Tilman Hinnerichs, Reuben Gardos Reid, Jaap de Jong, Bart Swinkels, Pamela Wochner, Issa Hanou, Nicolae Filat, Tudor Magurescu and Sebastijan Dumančić. *Extended Paper: An Introduction to Herb.jl: A Unifying Program Synthesis Library*. arXiv:2510.09726v2, 30 January 2026. [Full text](https://arxiv.org/html/2510.09726v2). **Inspected:** full text and Probe customization example. **Role:** reusable framework alternative and engineering-comparison caution. **Limit:** tutorial pseudocode is not copied as a verified command; Julia/native compatibility and exact port fidelity require testing.

[R01]: https://arxiv.org/abs/1910.02227
[R02]: https://arxiv.org/abs/2007.05367
[R03]: https://arxiv.org/abs/2005.02259
[R04]: https://doi.org/10.1609/aaai.v38i9.28925
[R05]: https://dspace.mit.edu/entities/publication/5377b306-4ecc-4687-b1f5-78cbb4a0543a
[R06]: https://doi.org/10.1007/s10994-015-5512-1
[R07]: https://doi.org/10.1145/3571234
[R08]: https://arxiv.org/html/2605.09985v1
[R09]: https://doi.org/10.1609/aaai.v35i8.16893
[R10]: https://arxiv.org/html/2602.00929v1
[R11]: https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/
[R12]: https://arxiv.org/abs/1808.04962
[R13]: https://ojs.aaai.org/index.php/AAAI/article/view/5776
[R14]: https://arxiv.org/html/2006.08381v1
[R15]: https://people.eecs.berkeley.edu/~sseshia/pubs/b2hd-jha-icse10.html
[R16]: https://github.com/logic-and-learning-lab/Popper
[R17]: https://stitch-bindings.readthedocs.io/en/stable/intro/tutorial.html
[R18]: https://arxiv.org/abs/2212.04596
[R19]: https://www.ijcai.org/proceedings/2019/841
[R20]: https://arxiv.org/abs/2607.01585
[R21]: https://doi.org/10.36227/techrxiv.176784644.48617663
[R22]: https://arxiv.org/html/2508.15750v1
[R23]: https://cvc5.github.io/docs/latest/examples/sygus-fun.html
[R24]: https://developer.apple.com/metal/pytorch/
[R25]: https://xgboost.readthedocs.io/en/stable/install.html
[R26]: https://pypi.org/project/z3-solver/
[R27]: https://support.apple.com/en-us/117737
[R28]: https://www.swi-prolog.org/build/macos.html
[R29]: https://pubmed.ncbi.nlm.nih.gov/26659050/
[R30]: https://www.microsoft.com/en-us/research/publication/automating-string-processing-spreadsheets-using-input-output-examples/
[R31]: https://patents.google.com/patent/US11934801B2/en
[R32]: https://zenodo.org/records/15512255
[R33]: https://arxiv.org/html/2010.08663v1
[R34]: https://arxiv.org/abs/2310.04327
[R35]: https://arxiv.org/html/2412.17330v1
[R36]: https://arxiv.org/html/2405.17514v2
[R37]: https://www.ijcai.org/proceedings/2024/0379.pdf
[R38]: https://arxiv.org/html/2510.09726v2


## Appendix A. Small checks actually executed during planning

These tests were executed in this research session on **Linux x86_64, Python 3.13.5**, not on the user's laptop. They use only the Python standard library. They resolve limited semantic issues; they do **not** implement the proposed architecture or benchmark any baseline.

The ring fixture contains 16 valid states and excludes the one already observed state from the query pool. Four of the 15 eligible states distinguish the two supplied hypotheses. Maximum disagreement selects a distinguishing state immediately; uniform sampling without replacement requires an exact expected 16/5 queries. This calculation demonstrates why acquisition can help in a deliberately tiny, fully enumerated hypothesis set, not that the proposed system has an acquisition advantage in realistic search.

The Boolean fixture checks all 256 functions on three input bits. An initial observation leaves 128 functions; a counterexample leaves 64, while an arbitrarily retained representative fails. All 16,384 incremental/full filtering comparisons agree. This is the reason recoverable equivalence is mandatory. Contradictory outputs for the same input leave no deterministic hypothesis in this fixture.

The immutable-cache fixture performs 32 definition revisions and verifies exact agreement with full evaluation and unchanged values of an unrelated root. Its finite-table setting supports the implementation invariant, not a claim of superior continual learning.

To reproduce, save the following block as `semantic_checks.py` and run `python semantic_checks.py`. Environment strings will differ on the Mac; all semantic fields and assertions should agree. Unexpected differences block dependent work until explained. No third-party package installation is needed.

```python
"""Small deterministic design checks; not an architecture benchmark.
Standard library only; run: python semantic_checks.py
"""
from __future__ import annotations
import hashlib
import json
import platform
import sys
from fractions import Fraction
from itertools import product
from math import comb


def main() -> dict:
    # A four-site ring. Wall value 4 denotes no wall; actor and wall differ.
    states = [(x, w) for x in range(4) for w in range(5) if x != w]
    def move(x: int, w: int) -> int:
        return (x + 1) % 4
    def blocked(x: int, w: int) -> int:
        y = (x + 1) % 4
        return x if y == w else y
    assert move(0, 2) == blocked(0, 2)
    pool = [s for s in states if s != (0, 2)]  # do not query known support again
    distinguish = [s for s in pool if move(*s) != blocked(*s)]
    n, d = len(pool), len(distinguish)
    expected = sum((Fraction(comb(n-d, k), comb(n, k))
                    for k in range(n-d+1)), Fraction(0))
    assert (n, d, expected) == (15, 4, Fraction(16, 5))

    # All Boolean functions on three bits, encoded as eight-bit truth tables.
    hypotheses = tuple(range(256))
    def out(h: int, x: int) -> int:
        return (h >> x) & 1
    first = [h for h in hypotheses if out(h, 0) == 0]
    second = [h for h in first if out(h, 1) == 1]
    lossy = min(first)
    assert len(first) == 128 and len(second) == 64 and lossy not in second
    checked = 0
    for truth in hypotheses:
        for a, b in product(range(8), repeat=2):
            inc = [h for h in hypotheses if out(h, a) == out(truth, a)]
            inc = [h for h in inc if out(h, b) == out(truth, b)]
            full = [h for h in hypotheses if all(
                out(h, x) == out(truth, x) for x in (a, b))]
            assert inc == full and truth in inc
            checked += 1
    impossible = [h for h in hypotheses if out(h, 0) == 0 and out(h, 0) == 1]
    assert not impossible

    # Pure, immutable finite-function nodes and version-keyed memoization.
    # Roots: A=f(x)+g(x); B=2*g(x). Repair f only; B must not change.
    def node(table: tuple[int, ...]) -> str:
        return hashlib.sha256(json.dumps(table).encode()).hexdigest()
    cache: dict[tuple[str, int], int] = {}
    def evaluate(table: tuple[int, ...], x: int) -> int:
        key = (node(table), x)
        if key not in cache:
            cache[key] = table[x]
        return cache[key]
    g = tuple(x % 3 for x in range(8))
    f = tuple(range(8))
    unchanged_before = tuple(2*evaluate(g, x) for x in range(8))
    for revision in range(1, 33):
        old_hash = node(f)
        new_f = tuple(x + revision for x in range(8))
        assert node(new_f) != old_hash
        incremental_a = tuple(evaluate(new_f, x)+evaluate(g, x) for x in range(8))
        full_a = tuple(new_f[x]+g[x] for x in range(8))
        assert incremental_a == full_a
        assert tuple(2*evaluate(g, x) for x in range(8)) == unchanged_before
        f = new_f
    return {
        "environment": {"platform": platform.platform(), "python": sys.version.split()[0]},
        "ring": {"states": len(states), "eligible_queries": n, "distinguishing_states": d,
                 "max_disagreement_queries": 1,
                 "random_without_replacement_expected_queries": str(expected)},
        "version_space": {"initial_viable": len(first), "after_counterexample": len(second),
                          "lossy_representative_survives": lossy in second,
                          "incremental_vs_full_checks": checked,
                          "contradictory_pair_viable": len(impossible)},
        "immutable_cache": {"revisions_checked": 32, "unrelated_root_unchanged": True},
        "scope": "Semantic fixtures only; no M3 Max or proposed-architecture performance measurement."
    }

if __name__ == "__main__":
    print(json.dumps(main(), indent=2, sort_keys=True))
```

**Output actually recorded:**

```json
{
  "environment": {
    "platform": "Linux-6.18.44-x86_64-with-glibc2.41",
    "python": "3.13.5"
  },
  "immutable_cache": {
    "revisions_checked": 32,
    "unrelated_root_unchanged": true
  },
  "ring": {
    "distinguishing_states": 4,
    "eligible_queries": 15,
    "max_disagreement_queries": 1,
    "random_without_replacement_expected_queries": "16/5",
    "states": 16
  },
  "scope": "Semantic fixtures only; no M3 Max or proposed-architecture performance measurement.",
  "version_space": {
    "after_counterexample": 64,
    "contradictory_pair_viable": 0,
    "incremental_vs_full_checks": 16384,
    "initial_viable": 128,
    "lossy_representative_survives": false
  }
}
```

Source SHA-256: `92309d2023b16e0ff13ad98dfb6b4774205c80d65379ad957981bb07e520d0db`. This hash refers only to the exact script block above.


## Appendix B. Coding-agent start and final reporting contract

Begin at W00 and record its decision before acquiring large assets. Do not treat the research date's dependency observations as an installed environment. Then complete the finite semantics and harness gates before interpreting learning results. Follow W03–W07 to reproduce competitors, calibrate capacity and freeze the experiment; only W08 may open protected results under that freeze.

The final execution report must contain the tested claim and its scope; all scheduled methods, including blocked and failed runs; primary and secondary effects with uncertainty; setup-inclusive resource accounting and measured reuse break-even; negative controls and important regressions; supplied language bias; fresh-confirmation status; exact provenance; and a continuation decision tied to Section 10. Every unpublished methodological amendment must be visible. A negative or inconclusive report that meets these requirements is a valid project outcome.

This Markdown document is the complete planning handoff. No other file from the originating conversation is required to understand the selected mechanism, implement the work packages or reproduce the limited planning checks. Public upstream assets are required only where the relevant work package and access/license gate say so.
