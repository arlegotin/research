# Learned Reaction Ecology: research assessment and executable proof-of-concept plan

**Research cut-off:** 14 October 2026.  
**Target:** one MacBook Pro with an M3 Max; actual memory, macOS, storage and supported software must be discovered.  
**Status:** research and experimental design, not a demonstrated new architecture. No LRE model has been trained. Small deterministic design checks described in §3 were executed on Linux/x86-64, not on the target Mac.  
**Primary direction:** a typed, demand-driven, budget-conditioned reaction system, with learned reaction outcomes and learned creation, scheduling and retirement of computational objects.  
**Working implementation name:** `LRE-T`. This is a project label, not a claim of novelty.

## 1. Executive assessment

### 1.1 Recommendation

Proceed with a **mechanism-isolating study**, not with a claim that an unprecedented artificial chemistry or a replacement for neural networks has been discovered. The defensible question is:

> Can jointly learning local computational reactions and the allocation and lifetime of their intermediate objects produce more reliable short-to-long compositional execution, at a measured resource cost, than the same learned reactions controlled by strong conventional and learned schedulers?

The strongest prospect is **conditional computation with a changing demand frontier**: some intermediate computations become unnecessary after other values are obtained; several queries can share intermediates; retaining an object competes with creating another. Strict, equal-cost expression reduction alone does not establish this prospect. Include it as a compositional test and a deliberately unfavorable allocation control.

The closest competition is not a generic feed-forward network. FastNRS already learns local reductions and their locations, including parallel reductions and length extrapolation [R1]. NeuRewriter learns where and which existing rewrite to apply [R3]. Aurora learns resource-sensitive scheduling of existing graph rewrites [R4]. Agoric computation and Hayek already connect computational agents, competition and credit [R7–R8]. Recent artificial chemistry work also obtains useful programs under endogenous selection [R6]. Therefore, **“learned chemistry plus resources” is not a defensible standalone novelty claim**.

The proposed differentiated result, if obtained, would be an **empirical finding about joint learning and compositional transfer under resource constraints**, supported by a matched factorial experiment. Stronger claims about a new methodological mechanism require surviving the stateful-controller equivalence test in §4.4. It is entirely possible that the appropriate conclusion is that LRE is an unnecessarily expensive implementation of a learned interpreter and scheduler.

### 1.2 What success would and would not mean

A locally meaningful success has three parts: accurate extrapolation to protected longer compositions; a reproducible advantage over strong resource-matched controls; and an advantage that survives actual execution, including bookkeeping, model evaluation, memory management and preprocessing. A gain only against random allocation is insufficient. A gain only against a fixed-depth GNN is insufficient. A gain in completed reactions without a benefit in correctness or elapsed time is insufficient.

The adaptive-capability target is a **single frozen set of learned parameters** that reorganizes demands and intermediate lifetimes when composition length, query reuse or resource constraints change. Faster execution at one familiar configuration alone does not satisfy that target. The core experiment cannot establish open-ended evolution, autonomous discovery of arbitrary algorithms, physical chemical computation, thermodynamic efficiency, biological plausibility, or the usefulness of replacing every neuron in a large neural system. Conventional neural modules parameterize the reaction kernel. The runtime objects replace a fixed **execution structure**, not the arithmetic inside those parameterizing modules. State this narrowing in every report.

### 1.3 Immediate decisions

| Decision | Reason | Blocking evidence or next test |
|---|---|---|
| Use typed graph-local reactions, not unrestricted random collisions | Makes locality, object lifetimes, correctness and costs explicit without an all-pairs matching explosion | Compare representation- and information-matched controllers; retain an explicit locality boundary |
| Separate strict compositions from conditional/shared-demand compositions | Strict reductions can have no allocation headroom | Prove/check reduction-count invariant before training |
| Include a learned finite rule-table baseline and an exact interpreter | Finite primitive semantics may be learned perfectly and then compose indefinitely | Establish this ceiling before extensive neural tuning |
| Make a stateful learned scheduler a mandatory control | Resource bids alone are a conventional policy parameterization | Compile the LRE policy into a conventional controller and test equivalence |
| Start with local supervision and short execution traces | Isolates execution and allocation from the separate difficulty of discovering primitive semantics from terminal rewards | Give every eligible baseline the same information; terminal-only training is a later, separately labeled experiment |
| Use native CPU execution plus optional dense PyTorch MPS kernels | Dynamic small-object runtimes may not benefit from GPU launches | Measure the complete path on the actual Mac; CPU is a faithful fallback |
| Do not make official FastNRS reproduction a fictitious accomplished prerequisite | A public official repository was not located, and a newer journal text could not be fully accessed during this research | Use an explicitly named paper-specification implementation and a stronger parser-assisted control; qualify claims accordingly |

## 2. Problem, scope and hypotheses

### 2.1 Intended beneficiary and bottleneck

The immediate beneficiary is a researcher building **small learned algorithmic components**: evaluators, symbolic manipulation modules and graph-local reasoning systems that must run more steps on larger inputs than they saw during training. A possible later beneficiary is a resource-limited incremental computation service. No customer requirement or deployed workload was provided. Thus practical demand and commercial value are **unverified**, not established bottlenecks.

The conventional workflow is to choose a fixed computation graph or a hand-designed interpreter/scheduler, train its local predictors, and execute under a declared traversal policy or iteration budget. A good conventional interpreter can already adapt its traversal to intermediate values. The hypothesized limitation is not merely limited model capacity. It is a mismatch between how computation and intermediate storage are allocated during training and the changing demands of longer, branching or shared computations. Whether this mismatch remains after a good lazy interpreter, learned priority queue and memoization is the central uncertainty.

### 2.2 Formal task

An episode contains a typed acyclic expression graph \(G=(V,E)\), literal values, and one or more root queries. An operator vocabulary and its arities are fixed within a training world, but the model must learn its semantics from the provided training examples. Operator identifiers presented to models are opaque symbols. A root query asks for the exact value of a composition. Test graphs change composition length, shape, operator context, query reuse and, in designated experiments, memory pressure.

Let \(S_t\) denote the mutable multiset of computational objects and their connections; \(a_t\) a local reaction; \(\theta\) reaction-kernel parameters; \(\phi\) allocation/lifetime parameters; \(B\) an execution-work allowance; \(M\) a mutable-state byte limit; and \(D\) an elapsed-time deadline. The runtime is

\[
S_{t+1}=T(S_t,a_t),\qquad a_t\sim\pi_{\theta,\phi}(\cdot\mid\mathcal O(S_t),B_t,M_t).
\]

`T` enforces types, bounds, ownership and atomicity. It does **not** compute an operator's correct answer, validate a predicted answer against an oracle, or decide semantic sufficiency during inference. The observation function supplies the permitted local view and declared budget information. A training-only oracle supplies labels and scores outcomes.

For a method \(m\), define \(A_m(d;\mathcal P)\) as exact query accuracy under deadline \(d\) on evaluation population \(\mathcal P\). Abstention, malformed output and missed deadlines count as incorrect. The primary competitive comparison integrates accuracy over a **pilot-fixed log-deadline grid** (§7), with separate reports by graph size and workload. Work counters and memory distinguish possible causal explanations but are not substitutes for elapsed-time measurement.

### 2.3 Established, assumed and speculative elements

**Established from inspected literature:** learned local rewriting, graph-based algorithmic learning, computational ecologies, local learned growth and resource-aware rewrite scheduling exist [R1–R12]. These are antecedents, not evidence that their combination outperforms a simpler scheduler.

**Necessary assumptions to test:** shared local rules can be learned accurately enough to survive longer execution; there is useful demand/storage headroom; local observations contain enough information to exploit it; scheduling overhead does not consume the benefit; training on short graphs teaches policies whose resource behavior transfers.

**Not required for the core PoC:** free-form molecule languages, emergence from random initial programs, evolution of a rule vocabulary, reproduction, neural architecture search, higher-order macro discovery, inter-episode online learning, physical reaction simulation and external tools or language-model APIs.

### 2.4 Hypothesis ledger

| ID | Testable hypothesis and predicted observation | Supporting antecedent, not proof | Credible alternative explanation | Evidence against / weakening observation |
|---|---|---|---|---|
| H1: compositional execution | Frozen learned local reactions maintain exactness as compositions get longer | FastNRS and discrete algorithmic reasoning [R1,R11] | Finite truth-table recovery plus a supplied interpreter | Learned table or discrete GNN matches/exceeds LRE; local error accumulation explains failures |
| H2: useful allocation | Adaptive reactions spend less time on unnecessary work under changing demand | Resource-sensitive rewriting and learned priorities [R4,R13] | Ordinary lazy evaluation, memoization or a learned queue | Gain disappears against lazy/cost-aware/stateful learned controls, or on measured wall time |
| H3: joint-learning interaction | Joint reaction/allocation training beats independently optimized components at matched information and expenditure | Plausible co-adaptation; **no direct evidence located for this exact experiment** | More training, better initialization, extra state or extra labels | No positive interaction in the factorial contrast, or advantage disappears with matched retuning |
| H4: object lifecycle contribution | Learned creation/cancellation/retention generalizes to larger frontiers and changed memory capacity | Local growth and developmental models [R9,R10] | A learned cache or recurrent hidden state with different names | Equivalent conventional implementation reproduces results, or cache/prefetch controls explain gain |
| H5: practical local feasibility | The complete method reaches a useful accuracy/time/memory frontier on the Mac | Supported dense CPU/MPS paths exist, not this runtime [R17–R20] | Work-counter savings conceal synchronization/bookkeeping cost | Stronger simple baselines dominate after all costs are included |

An H1 success without H2/H3/H4 is a learned-execution result, not evidence for a novel ecology. H4 requires causal interventions, not attractive visualizations of changing object populations.

## 3. Analytical constraints and checks already performed

### 3.1 A necessary no-headroom result

**Proposition.** Consider a finite acyclic strict expression tree with \(n\) internal operations. Each legal reaction evaluates exactly one previously unevaluated internal node, after its children are available. Every internal node is required, no result is recomputed, and each operation has equal unit cost. Every complete legal serial schedule performs exactly \(n\) evaluations.

**Reason.** The number of unevaluated internal nodes decreases by exactly one on every reaction, starts at \(n\), and must end at zero. Reordering cannot change this count. With correct deterministic primitive rules, structural induction also gives the same final result for every legal schedule.

This does not establish equal wall time, parallel makespan, maximum live memory or correctness with imperfect learned rules. It also does not cover short-circuiting, speculative work, shared queries, cache eviction or alternative derivations. Those are the legitimate places to look for headroom. Never manufacture a scheduling claim from this strict case by comparing against a baseline that performs gratuitous work.

### 3.2 Finite-rule ceiling and partial certificates

For the core modular-arithmetic domain, three binary operators on values `0..6` require only \(3\times7^2=147\) full-input rows. A learner given all rows can implement exact primitive semantics and, with a correct interpreter, generalize to arbitrary finite composition length. This is a substantive baseline, not an unfair oracle.

For Boolean operators `NOT`, `AND`, `OR`, `XOR`, `IF`, there are 22 full-input rows. A partial tuple uses `?` for an unavailable operand. Define a **certificate** from a learned, complete table:

\[
\operatorname{cert}(o,x)=v\quad\text{iff every full tuple consistent with }x\text{ has learned output }v.
\]

An incomplete table may not silently treat missing completions as evidence. It must abstain or use a separately declared inductive model. Neural reaction heads learn the same `value-or-unknown` target; they do not receive certified answers from this table at inference.

### 3.3 Executed small checks

Executed with Python 3.13.5 on Linux/x86-64. The check enumerated all ordered full binary tree shapes with one through six internal nodes and every legal reduction order for each shape. It used fixed deterministic literal values and modular arithmetic operator assignments, not every possible semantic assignment.

| Internal nodes | Tree shapes | Complete legal schedules checked | Different answers | Reductions per schedule |
|---:|---:|---:|---:|---:|
| 1 | 1 | 1 | 0 | 1 |
| 2 | 2 | 2 | 0 | 2 |
| 3 | 5 | 6 | 0 | 3 |
| 4 | 14 | 24 | 0 | 4 |
| 5 | 42 | 120 | 0 | 5 |
| 6 | 132 | 720 | 0 | 6 |

A second exhaustive check enumerated partial Boolean tuples:

| Operator | All partial/full patterns | Certifiable patterns | Certifiable patterns containing `?` |
|---|---:|---:|---:|
| NOT | 3 | 2 | 0 |
| AND | 9 | 6 | 2 |
| OR | 9 | 6 | 2 |
| XOR | 9 | 4 | 0 |
| IF | 27 | 14 | 6 |

These results establish small harness facts and an obvious exact symbolic competitor. They are **not** evidence of learned extrapolation or LRE superiority. The reproducible check is included in Appendix A; no external intermediate file is needed to repeat it.

### 3.4 Additional deductions that guide the design

**Error accumulation.** If each of \(n\) required local predictions has error probability at most \(\epsilon\), a union bound gives whole-trace error at most \(n\epsilon\); independence is unnecessary. The often-used \((1-\epsilon)^n\) success expression requires independent errors and should only be a diagnostic approximation. Root accuracy can conceal canceling local errors, so log both root and local correctness.

**Allocation equivalence.** An ecology that assigns a learned score to each eligible reaction and selects by softmax is exactly a learned scheduling policy. Object names and an “energy” label do not change that fact. A finite bounded multiset state can be encoded as a conventional controller state. The PoC seeks an inductive-bias or optimization advantage, not a representational impossibility for conventional systems.

**Locality and information.** A local controller cannot choose optimally between two subgraphs indistinguishable in its current view but with different unseen costs or outcomes. Include such pairs. Global structural preprocessing may help conventional methods; either provide the same features to LRE and charge the preprocessing, or report a separate local-information comparison. Do not cripple practical competitors to preserve a locality advantage.

**Matching cost.** Unrestricted pairwise collisions generate \(O(N^2)\) potential pairs. Typed parent/child ports restrict the core to \(O(N+E)\) candidates. High fan-out still costs \(O(E)\); bounded child arity does not imply bounded parent fan-out.

**Overhead headroom.** If an optimization affects fraction \(f\) of baseline time and speeds that part by \(s\), before new overhead the speedup is at most \(1/[(1-f)+f/s]\). Add LRE-specific time to the denominator. For example, a hypothetical 2× improvement in a component occupying 20% of runtime yields only about 1.11× overall before overhead. This is arithmetic, not a performance estimate.

**Termination versus correctness.** Positive charged work and finite credits bound runtime even if a policy loops. They do not prove semantic correctness or confluence. Learned wrong publications can terminate quickly and incorrectly.

## 4. Prior art and novelty assessment

### 4.1 Closest alternatives

The matrix separates substantive overlap from differences. “Local” describes likely small controlled reproductions, not an executed compatibility test.

| Prior art | Problem and mechanism | Empirical conditions / assumptions | Compute, implementation and access | Exact overlap and implication |
|---|---|---|---|---|
| NRS / FastNRS [R1,R2] | Learned selection and reduction of subexpressions; FastNRS batches disjoint reductions | Supplied algorithmic reduction template; supervised modules; mathematical formula domains | Small neural modules; full paper and appendix inspected; official code not located in this search | Direct overlap with learned local reactions and short-to-long composition; mandatory controlled baseline |
| NeuRewriter [R3] | Learns region and rewrite selection using reinforcement learning | Existing rewrite vocabulary; expression optimization, scheduling and routing | Public PyTorch code; repository states CC-BY-NC, so not an unrestricted open-source dependency | Scheduling plus rewriting is old; use paper-based region/rule control and honor code restrictions |
| Aurora [R4] | GNN/recurrent policy schedules equality-saturation rewrites, including extraction/reset decisions | Predefined rewrite rules; query optimization and resource limits | Paper inspected; complete database reproduction is not core-laptop evidence | Resource-aware learned rewriting is not new; learned rule outcomes distinguish the proposed test, not automatically its novelty |
| Fontana / AlChemy [R5] | Computational molecules interact and self-organize | Algorithmic chemistry rather than supervised end-task learning | Original institutional working-paper description inspected | Rules-as-molecules and functional self-organization are established |
| Prebiotic Functional Programs [R6] | Endogenous selection through program/test interactions produces useful functional programs | Lambda-calculus chemistry; experiments include scales beyond a laptop toy | Full preprint inspected; author repository identified, not executed | Productive computational ecology itself is not a new claim; current, particularly relevant counterexample |
| Agoric systems / Hayek [R7,R8] | Computational agents, markets, incentives and credit assignment | Economic organization; Hayek includes learned agent cooperation in a task setting | Original agoric text and Hayek publisher abstract/metadata inspected | Competition and computational credit are old; a bid/credit mechanism must be distinguished precisely |
| Growing NCA / learned developmental programs [R9,R10] | Local learned updates, growth and changing computational structure | Morphogenesis or developmental/behavioral settings, not this compositional benchmark | Primary articles inspected; different task and training assumptions | Object birth/death and local self-organization are not independently novel |
| DNAR [R11] | Discrete algorithmic states with learned transitions and numeric operations | Algorithm-specific interfaces and hints; graph algorithm tasks | Official Apache-2.0 repository; PyTorch/PyG dependencies and processor code inspected | A strong discrete graph baseline can extrapolate without an ecology |
| Size-generalizing MPNNs, 2026 [R12] | Conditions for learnability/extrapolation; architecture and regularization matter | Guarantees have explicit task/data/function-class conditions | Full current preprint inspected, including method discussion | Do not generalize a weak continuous GNN's failure to all GNNs; not an off-the-shelf expression evaluator |
| Neural priority queues [R13] | Learned priority/memory interfaces for graph networks | Graph-learning settings, not this ecology | Primary abstract inspected | Learned queues are an obvious alternative explanation and control |
| Popper [R14] | Inductive logic program learning from examples and background knowledge | Search-language bias is part of the input | MIT repository; current dependency manifest inspected; native build compatibility unresolved | A serious symbolic learner must receive comparable primitives and supervision |
| DreamCoder [R15] | Learns reusable program libraries with neural-guided search | Explicit program language and induction/search budget | Primary abstract inspected; full reproduction outside core scope | Reusable computational abstractions are not a new claim; relevant if macro discovery is added |
| Graph rewriting for GNNs [R16] | Formalizes GNN computation as graph rewriting | Representation/formalization rather than this adaptive resource experiment | Primary abstract inspected | Changing graph terminology does not create an architectural contribution |
| Learned compiler graph optimization patent [R21] | Learning to guide computational graph optimizations | Patent description spans graph optimization/scheduling mechanisms | Public patent text inspected; no legal scope assessment | Additional evidence against broad “learned graph scheduling” originality; not a freedom-to-operate opinion |

### 4.2 What can be reused

Reuse the **ideas and clearly licensed implementations** of local predictors, recurrent message passing, discrete bottlenecks, typed graph data structures, program generators and symbolic induction. Reuse PyTorch/NumPy/test infrastructure rather than building an accelerator compiler. Use the DNAR repository as a reference and, where compatible, an isolated baseline. Use Popper in a separate environment. Implement the tiny finite-table learner directly; it is both stronger and cheaper than unnecessary large symbolic search for the finite core domains.

Do not copy NeuRewriter code into a generally redistributable project without satisfying its noncommercial terms. A fresh implementation from the paper's algorithmic description avoids making that repository a required dependency, but does not erase the need to cite the method. Do not redistribute third-party datasets, code or model assets merely because a browser can access them.

### 4.3 FastNRS findings that change this plan

FastNRS is sufficiently close that a renamed local rewriting layer would not be original. The study inspected its full arXiv v1 text and PDF appendices, including screenshots of the architecture-ablation and hyperparameter pages. Its evaluation and module-training choices are useful precedents, not controlled results for LRE. The paper itself identifies the supplied algorithmic blueprint as a limitation [R1].

A 2026 *Artificial Intelligence* version was indexed at the publisher and the authors' institution, but its full text could not be fetched here. No claim is made that v1 captures every later change. Searches by method name, paper title, authors and repository combinations did not locate an official public FastNRS implementation. This is an access limitation, not proof that none exists. The executable baseline below is therefore named `fastnrs_paper_spec`, not “official FastNRS reproduced.”

### 4.4 Selected refinement and the equivalence challenge

Three directions were considered.

**Unrestricted artificial chemistry:** closest to open-ended ecology, but long collision searches, uncontrolled growth and sparse terminal credit would entangle too many uncertainties. It is not rejected as impossible. It is deferred until a controlled learned reaction system demonstrates a reason to incur those costs.

**Strict learned bottom-up reduction:** easiest and useful for verification, but already close to FastNRS and a learned table interpreter, with little allocation headroom. It is a diagnostic and baseline, not the primary contribution.

**Selected: typed demand reactions with learned lifecycle and budget conditioning:** allows local demand creation, partial-result publication, cancellation, intermediate retention and budget competition. It preserves a controlled version of the intended changing computation substrate while giving clean interventions. Its value is plausible when demand and storage needs change; its novelty remains unresolved and likely narrow.

Before attributing a benefit to ecology, implement `compiled_stateful_controller`: encode the same object multiset in conventional arrays and run the identical learned policy. It should give the same deterministic trace. If it does not, one implementation is wrong or the alleged match is incomplete. This intentionally establishes representational equivalence. The substantive comparison is then whether **training with the structured reaction/object organization** is a useful inductive bias versus independently trained, equally expressive conventional controllers—not whether conventional computation cannot implement it.

The term “self-organization” is permitted only operationally: a task-conditioned object/communication structure is produced by the learned policy rather than a fixed traversal. It must not imply autonomous rule invention or emergence without engineered interfaces.

### 4.5 Novelty invalidators

Invalidate or narrow a novelty claim if earlier work already jointly learns reaction outcomes and resource/lifecycle control with comparable locality and transfer tests; if a standard recurrent scheduler exactly reproduces the claimed effect after matching state and training; if a learned cache explains the gain; or if the only difference is software packaging. A new benchmark, a careful negative result, or an efficient implementation can still be useful, but must be labeled accordingly.

The refinement search included learned rewrite scheduling, equality saturation under budgets, program economies and credit, demand-driven computation, learned priority queues, learned memory/caching, developmental computation and reusable program libraries. It did not establish absence of prior art. Patent searching was limited to publicly indexed descriptions and was not a legal review.

## 5. Selected mechanism: implementation specification

### 5.1 Deliberate limits of the initial chemistry

The first implementation is a **typed port-graph interpreter with a learned control policy**. It does not learn the input language, parser, arities, type system, memory allocator or atomic transaction semantics. It does learn operator outcomes, when to request operands, when a partial result is sufficient, which eligible reaction receives a processing opportunity, and whether an intermediate should survive.

The input syntax graph is immutable. The mutable reaction graph is not: work objects, pending subscriptions and value objects are created and destroyed. An episode can therefore build a different computation graph for different inputs even when the syntax has the same shape. This is the smallest controlled version that preserves the intended lifecycle/competition mechanism. Arbitrary invention of syntax edges or new operator types is not part of the initial claim.

A single global arbiter awards finite processing opportunities to locally described reactions. This is an engineering implementation of competition, **not a decentralized market**. Adding per-object wallets without a discriminating hypothesis is specifically excluded.

### 5.2 Static input representation

Use a structure-of-arrays representation, initially on CPU:

```text
Graph:
  graph_id: UTF-8 string
  domain_id: integer enum
  op_id[N]: int16                 # includes literal type; opaque operator IDs
  arity[N]: uint8                 # 0..3 in the core grammar
  child[N,3]: int32               # -1 only for unused ports
  literal[N]: int16               # -1 on nonliteral nodes
  parents: CSR lists of (parent_id, port)
  roots[Q]: int32
  arrival_ns[Q]: int64            # private to service wrapper until each arrival
  semantic_version: string       # held by evaluator, not a semantics API
```

Validate acyclicity, child types, port order and root existence. Node IDs must not encode traversal order, answer, depth or split membership: randomly permute IDs after generation and remap every pointer. Preserve child-port order for noncommutative operations. Padding is not a node and must not receive messages or be selectable.

Hash-consing repeated syntax is a shared optional preprocessing optimization. Its time and memory are charged, and either all eligible methods receive it or a separate raw-input comparison is reported. Never expose a hash of a subtree's *answer*.

### 5.3 Mutable objects and ownership

Use indexed slots with generation counters, not a Python object per scalar in the optimized path. Begin with a readable reference implementation before optimizing.

| Object | Fields | Role and ownership |
|---|---|---|
| `Work(v)` | node ID, generation, hidden vector `z`, three operand slots, pending-port mask, demand reference count, creation/update counters | One live work object per syntax node per graph/epoch; holds unfinished computation |
| `Wait(p,v,k)` | parent work generation, child ID, port, subscription ID | Delivers a child value into a particular parent operand slot; represents a learned demand edge |
| `Value(v,y)` | node ID, value, epoch, pin count, access counters | Reusable intermediate; may survive after its current consumers finish |
| `Root(q,v)` | query ID, root ID, active/completed flag | External demand; cannot be deleted to evade a query |
| `Delivery(q,y)` | query ID, predicted value, completion timestamp | Final output record; immutable once submitted |

Literal values are input data. Reading a literal port does not run an oracle: it is a charged local read of the provided input. A `REQUEST` on a literal can fill its parent operand directly without allocating a child work object. A cached value can similarly be delivered, but the lookup and write are recorded.

Parent operand slots store **copies** of delivered small values, not dangling references to evictable objects. A `Wait` is removed after delivery or cancellation. A completed parent enqueues its subscriptions for explicit charged cleanup; `CLEAN` removes one obsolete subscription per event. Required cleanup is drained before the next learned selection by the same deterministic housekeeping policy for every shared-runtime method; it consumes credits and wall time. There is no invisible recursive free traversal. Orphan work may be canceled, but never deletes a different consumer's required value or subscription. Deduplicate work by `(graph_id, epoch, node_id)`, not by unverified semantic equivalence.

A value is pinned while any live waiter or not-yet-delivered root requires that value, as well as during an atomic read. Compute/update its pin count from those references when publishing, delivering, canceling and cleaning. An obsolete waiter remains a pin until its charged cleanup removes the reference. Eviction must not strand a waiter after the producing work object has retired. Unpinned cached values are the learned retention choice; active consumer storage is not silently reclaimable. Apply this ownership rule to every shared-runtime baseline.

Root admission is deterministic, not a learned semantic shortcut. At initialization and each visible arrival, allocate a `Root` demand. If its node is a literal or has a same-epoch cached value, enqueue a charged `ROOT_DELIVER` event; otherwise create/reuse its `Work` object and increment its root demand count. Root-admission allocation, lookup and input inspection are included in timed setup/arrival handling and the byte ledger; do not double-charge them as an extra learned reaction. Admission failure returns an explicit resource failure. A literal root is copied from the input, never predicted by an arity-zero neural publication. Empty-query inputs return an empty result with measured setup time. Newly arriving roots cannot observe future root identities.

For multiple roots, maintain consumer reference counts. Canceling one query's branch cannot cancel a shared demand still needed elsewhere. Cross-graph caches are empty in the core. Across roots of the same immutable graph, cache lifetime is explicitly part of the streaming episode; across changed inputs an epoch mismatch invalidates values.

### 5.4 Reaction grammar

The following are project-defined actions, not pre-existing package APIs.

| Action | Syntactic preconditions | Transaction; learned part |
|---|---|---|
| `REQUEST(p,k)` | parent work live; port valid, unknown and not pending | Read literal/cache if present; otherwise create/reuse child work and create a waiter. Policy chooses port and timing |
| `ROOT_DELIVER(q)` | root awaits an input literal or same-epoch cached value | Copy that known value into a delivery record; release its root pin; one deterministic charged event |
| `DELIVER(w)` | waiter live and child value available | Copy value to parent slot and delete waiter. Policy may prioritize delivery; no predicted semantics involved |
| `PUBLISH(p,y)` | work live, result type valid; strict-track readiness rule below | Create a value and satisfy matching root demands; retire parent work and enqueue subscription cleanup. Policy predicts `y` and timing |
| `CANCEL(p,k)` | pending subscription owned by this parent exists | Delete that subscription and clear pending bit; update reference counts. Policy decides unnecessary demand |
| `CLEAN(w)` | waiter belongs to a retired parent generation and has not already been removed | Delete one obsolete subscription and update child reference count; one charged housekeeping event |
| `EVICT(v)` | cached value live; no pending waiter/root pins or atomic read | Delete the value; delivered operand copies remain valid. Policy chooses lifetime |
| `RETIRE(v)` | orphan work has no root or waiter demand | Delete work and enqueue cleanup of its owned subscriptions. Policy chooses whether to abandon speculative work |
| `REFINE(v)` | work live | Apply one learned recurrent update to local hidden state without producing a value. Counts as processing; optional in the minimum implementation and always ablated |

The strict domain requires all operand slots before `PUBLISH`; the conditional domain does not. This distinction belongs to the declared workload protocol. In the conditional domain the engine must not consult semantic sufficiency to mask publishing actions. Otherwise the supposed learned short-circuit behavior would be supplied by the evaluator.

For a publication with existing cached value, normally no work for that node should remain. Treat inconsistent duplicates as an invariant failure in debug mode, not as an opportunity to select the correct answer. A wrong value is never silently replaced with a ground-truth one. Core inference permits one submitted output per query; a wrong answer is final.

`REFINE` is disabled for the first correctness implementation. Enable it only after a pilot shows that repeated local state updates, rather than primitive-data deficiency, could improve the measured tradeoff. Its inclusion changes the declared method configuration, not the result population; retain the version without it.

### 5.5 Local neural kernel and state

An initial implementation specification is a shared MLP encoder with two hidden layers, followed by separate action/value heads and a GRU state update. Start at hidden width 64 for smoke tests and compare 64/128 at the first capacity pilot. These are calibration starting points, not mandatory model-size ceilings. Increase width only when primitive/generalization learning curves and measured resource capacity justify it. Match or explicitly account for parameter and state differences in controls.

The features for a work object are:

```text
opaque operator embedding; domain/type embedding;
for each ordered child port:
  valid-port flag, pending flag, known-value one-hot including UNKNOWN,
  child's static operator/type embedding, literal flag;
root-demand flag; log1p(active demand count); log1p(object age);
remaining event credits / initial credits;
mutable bytes used / mutable byte allowance;
local hidden state z.
```

Use fixed value vocabularies and an explicit unknown symbol. Never encode an unknown as the numeric value zero. Clamp/log-transform unbounded counters consistently; do not introduce a learned embedding indexed by absolute graph size or node ID. The age convention is number of committed interpreter events since creation; timers are not semantic features. At the start of a work object, set `z=0` unless a learned initializer using only its permitted features is declared.

A waiter/eviction candidate uses the same local endpoint features and its own type/age flags. Extra global structural features—subtree size, critical-path estimates, fan-out summaries—are absent from the main local variant but offered to a stronger conventional control and to a separately labeled augmented LRE variant, with their costs included.

At each event, score each eligible local object once and derive its action logits. A simple reference implementation rescans all active objects. This cost is intentionally visible. The GRU state updates exactly once on a committed reaction involving that still-live object, not during a speculative scoring pass. `REFINE` uses this same update phase rather than applying a second update. During training, keep versioned/out-of-place hidden tensors until backward; do not overwrite autograd-required storage when recycling an object slot. Publish-value logits are part of the policy; auxiliary semantic supervision trains them. Concretely, let the semantic head predict `K_value + 1` classes including UNKNOWN, where `K_value` is the domain's value-vocabulary size. Form each eligible publication logit as an action-type logit plus `log p(value)` from that head, using a numerical floor declared in the configuration. Other action logits come from the control head. Normalize over the final candidate-action list once; record that actual log probability for policy gradients. Do not renormalize away UNKNOWN before deciding whether publication is attractive. There is no global hidden answer vector.

### 5.6 Competition and conservation

Let \(\mathcal A_t\) be the syntactically eligible action set. A reference policy is

\[
\pi(a\mid S_t)=\frac{\exp(\ell_a/\tau)}{\sum_{b\in\mathcal A_t}\exp(\ell_b/\tau)}.
\]

Training samples one action per policy-selection event; deterministic root delivery and mandatory cleanup are charged environment events without a policy sample. Evaluation uses deterministic maximum logit with a documented tie rule based on a stable randomized input ordering, or a separately measured stochastic policy whose seeds are fixed before testing. Subtract the maximum logit before exponentiation. If all candidates are masked or nonfinite, return a typed failure/abstention; never produce a uniform distribution over invalid actions.

Each attempted interpreter reaction burns one event credit, including failed resource reservations, canceled proposals, refinement, and cleanup events. With initial allowance \(B_0\),

\[
B_t + C_t = B_0,\qquad B_t\geq0,
\]

where \(C_t\) counts charged events. Rewards do not create credits. Object creation does not create credits. Event credits are a finite-execution control, **not joules, FLOPs, or equal wall-clock compute**.

The actual scarce resources are time and mutable memory. In controlled tests, the mutable-state cap is an algorithmic working-set constraint, not a claim that the laptop has run out of RAM. Every run records scoring rows, candidate enumeration, graph reads, writes, allocations, deletions, semantic predictions, rejected actions, resident bytes and elapsed time. Equal event caps support matched-runtime diagnostics, not cross-architecture efficiency claims. No controller is allowed arbitrarily many free policy evaluations between charged events.

Memory reservation is atomic. If an action would exceed its mutable-byte limit, reject it, charge the attempt, and leave the prior state intact. Do not silently evict objects using an LRE-only oracle. A conventional eviction policy can be used only as an explicitly named baseline or shared safety policy. Count arena backing buffers and unused reserved capacity as well as live payload when enforcing a physical mutable-byte cap. A separate live-object occupancy cap is permitted only as an explicitly labeled algorithmic diagnostic. Deleting an object makes a slot reusable; it does not necessarily reduce RSS. Model parameters, framework workspaces, immutable input and total process memory are additionally reported; they are not hidden by the mutable-state cap.

### 5.7 Serial reference pseudocode

```python
# Proposed project algorithm; observation and transaction APIs must be implemented.
def execute(graph, policy, limits, clock, ledger):
    state = initial_state(graph, limits)
    # clock/deadline was created by the wrapper before parsing and initial setup.
    while not state.episode_finished():
        state.accept_visible_arrivals(clock, ledger)
        if state.event_credits == 0 or clock.deadline_reached():
            return state.finish_unanswered_as_abstentions(ledger)
        if state.required_housekeeping_pending():
            action = state.next_housekeeping_action()  # CLEAN / ROOT_DELIVER, fixed rule
        else:
            candidates = enumerate_syntactic_candidates(state, ledger)
            if not candidates:
                if state.only_waiting_for_future_arrivals():
                    clock.wait_for_next_arrival_or_deadline()
                    continue
                return state.finish_unanswered_as_deadlocked(ledger)
            logits = policy.score(local_views(state, candidates, ledger))
            action = choose_action(logits, candidates, state.rng)
        state.burn_one_credit()           # also on rejected reservation / stale proposal
        delta = prepare_transaction(state, action)
        if not delta.valid_types_and_versions:
            ledger.record_rejection(action, "stale_or_invalid")
            continue
        if not reserve_mutable_bytes(state, delta):
            ledger.record_rejection(action, "memory")
            continue
        commit(state, delta, ledger)      # includes pointer/refcount correctness checks
        policy.update_touched_state(state, action)  # no-op if no live neural object is touched
        ledger.record_completion_times(state.new_outputs, clock.now_ns())
        # Outputs finishing after the deadline remain misses, even after overshoot.
    return state.outputs_and_ledger()
```

`required_housekeeping_pending` includes obsolete-subscription cleanup and root-delivery events; use cleanup-first, then stable admission order for root delivery. A normal `episode_finished` requires the declared service horizon, settled query outcomes, and no required housekeeping. A deadline/credit exit with pending housekeeping is recorded as `unclean_termination`, not a successful normal exit. Preserve raw root-publication timestamps separately. The primary single-query B1 metric uses the complete call's finish time, including required cleanup and output materialization, and treats an unclean deadline exit as a miss. Streaming may report earlier user-visible delivery latency, but must also report total service/cleanup time and backlog; no post-output work disappears from throughput accounting. Episode-wide teardown or scratch-arena reuse is an optional shared optimization, not an LRE-only shortcut, and its charged/timed boundary must be identical across eligible methods.

The initial B2 condition serves a declared number of queries sequentially: reveal the next root only after the previous query terminates and its required housekeeping settles, with caches retained. The input may disclose the query count, not future root identities. Causal streaming is a separate condition with a predeclared external wall-time arrival sequence. Logical-budget debug tests may use a separate simulated arrival clock, never method-dependent reaction counts presented as physical time. Waiting is not deadlock; an episode is complete only when its service horizon and all relevant query outcomes are settled.

The first trained version should vectorize **independent environments**, not conflicting events within one graph. Later parallel commits may select a conflict-free set, but only after comparison with every serial ordering of small disjoint transactions. Shared writes, reference-count changes, root deliveries and memory reservations are conflicts. Comparing an aggressively batched LRE only with an unbatched baseline is invalid.

### 5.8 Learning procedure

**Phase A: primitive and partial-state learning.** Supply a training table of complete primitive tuples to all methods. For Boolean tasks, generate partial-state labels by enumerating completions in the training-only complete table. Target is one of the legal values or `UNKNOWN`. Learn cross-entropy on full results and partial sufficiency. This removes semantic sparsity as an explanation in the main experiment. Keep a reduced-data condition separate; it tests primitive induction, not pure compositional transfer.

**Phase B: trace warm start.** Generate short-graph executions from multiple conventional schedules: lazy depth-first, breadth/ready-first and a training-only cost-to-go oracle for tiny cases. Supply their union to eligible methods, not only a privileged LRE demonstration. Supervise requesting, publication and learned delivery choices. Include deterministic root delivery and cleanup in traces for runtime and cost validation, not as learned action targets. For states with several equally good next actions, minimize negative log probability of the allowed action set rather than imposing one arbitrary order. Cache decisions use only past and currently visible demands; future-informed Belady-style decisions are oracle diagnostics, not demonstrations for a causal streaming claim unless every learner is explicitly given the same privileged training condition.

**Phase C: joint rollout training.** Alternate a primitive-supervision minibatch with on-policy batches of short graph episodes. Use a shared encoder and separate semantic, action and critic heads. A simple actor–critic/REINFORCE implementation is preferable to a complex training framework:

\[
L = L_{\mathrm{semantic}} + \lambda_I L_{\mathrm{imitation}}
   -\lambda_P\sum_t\log\pi(a_t\mid S_t)\,\operatorname{stopgrad}(G_t-V(S_t))
   +\lambda_V\sum_t(V(S_t)-G_t)^2 -\lambda_H\sum_t H(\pi_t).
\]

Policy-gradient, imitation-action and entropy sums range only over actual policy-selection events. Deterministic `CLEAN`/`ROOT_DELIVER` events and external arrivals have no policy log probability, but their time, memory and credit effects contribute to the subsequent return. The critic may also be trained at those states under a separately normalized state-value loss. Do not fabricate log probabilities for forced housekeeping.

A terminal reward is the fraction of arrived queries answered exactly under the declared completion contract, including failures as zero. Optional shaped rewards penalize measured event/byte-time costs using training-only coefficients. They must never use test answer correctness online. Use discount 1 for the finite horizon; an exponential discount would introduce an additional preference for short executions. Include value predictions chosen by `PUBLISH` in the action log probability. For hierarchical selection, the log probability is the sum of the conditional probabilities actually sampled.

Initially normalize each loss by its relevant count and use unit weights for semantic and policy terms, critic coefficient 0.5, and entropy coefficient 0.01 as **pilot defaults**, not presumed optima. Tune relative gradients and validation behavior equally for joint and control methods; record gradient norms and a declared search. Anneal imitation only after autonomous short-graph execution works; retain semantic supervision to detect rule degradation. Start AdamW at `3e-4`, no weight decay, FP32, clipping global gradient norm at 1; these are project starting choices, not paper reproduction parameters. Verify several neighboring learning rates in the pilot before attributing failure to the idea.

No gradient passes through an integer pointer mutation. The policy-gradient term handles discrete choices; local supervised losses and the GRU's differentiable computation train the neural parts. Do not claim an unbiased gradient through hard rewrites. Avoid straight-through top-k as the unexamined primary estimator. A critic may have a global training-only view, but must never supply features to the deployed policy; a matched privileged critic must be available to control learners.

For jointness, retain both **frozen-kernel allocation training** and **separate-kernel/separate-scheduler training**, with the same total project-specific examples and update/time expenditure. Also allow a strong conventional stateful controller to train jointly end-to-end. Otherwise “joint learning wins” could merely mean that one competitor was forbidden to co-adapt.

### 5.9 Essential correctness and numerical tests

Test a publish–pending-waiter–eviction attempt (eviction must be rejected), root literal/cache admission, and an output published before its cleanup crosses a deadline (primary B1 must count a miss). Test parser/printer round trips; all 147 modular primitive rows; all 22 Boolean full rows and 57 partial patterns; all-literal and empty-query cases; high fan-out; duplicate root queries; identical subexpressions with different IDs; cache epoch changes; unknown-versus-zero; all-invalid logits; nonfinite gradients; and exact resumption across a transaction boundary.

For every event, check credit conservation, nonnegative reference counts, unique work/value ownership, generation-safe pointers, byte-accounting reconciliation and no answer changes after submission. Test repeated cancel/request loops, attempts to delete required root demands, duplicate delivery and simultaneous shared-child completion. Use fault injection to ensure wrong predictions are recorded as wrong rather than repaired.

FP32 is the primary neural precision. Ground-truth Boolean and modular arithmetic use exact integer operations on CPU. Use float64 CPU only for numerical diagnostics or a baseline that requires it. A different precision path needs output, gradient and threshold-decision parity tests; speed alone does not justify the substitution.

## 6. Workloads, data and baseline specification

### 6.1 Core generated workloads

The core uses newly generated synthetic data; there are no required pretrained weights or private datasets. Publish the generator and manifest under a license selected for code the project owns. Treat third-party benchmark reconstructions as separately licensed/provenanced assets.

| Track | Grammar / outputs | Why it is included | What it cannot establish |
|---|---|---|---|
| A: strict composition | `E := digit(0..6) | ADD7(E,E) | SUB7(E,E) | MUL7(E,E)`; ordered operands; arithmetic modulo 7; root output, strict readiness | Short-to-long execution and no-headroom allocation control | A resource-competition advantage from avoiding mandatory equal-cost nodes |
| B1: conditional composition | `E := bit | NOT(E) | AND(E,E) | OR(E,E) | XOR(E,E) | IF(E,E,E)`; `IF(c,a,b)=a` for `c=1`, else `b` | Learned partial sufficiency, request ordering and cancellation | Real-world utility or unrestricted algorithm invention |
| B2: shared/streaming demand | Same Boolean grammar as B1, represented as a DAG with successive root queries | Object retention, reuse, memory pressure and changing demand | An online adaptation claim unless parameter updates are explicitly enabled |
| C: realistic-format transfer, conditional | Pure Boolean predicates exported from a locally constructed tabular filter or a documented public query workload | Checks whether synthetic benefits survive parsing, real data distributions and integration | Production database acceleration without an actual controlled engine integration |

Track C is a gated extension, not evidence already obtained. Its first version must specify actual predicate cost and source provenance, compare with an optimized direct evaluator, and avoid adding expensive black-box leaf calls only to make scheduling look attractive. If no lawful, appropriate real workload is available, retain the synthetic claim and stop that transfer branch.

### 6.2 Generation and splits

Count size by **internal operator nodes**, and also report depth, total nodes, token length, fan-out, number of roots and number of distinct subgraphs. Do not use “length” ambiguously.

An initial development configuration trains on 1–8 internal operations and validates on independent 1–8 and 9–16 operation graphs. Eight is a starting scale that permits multiple operator interactions and branching, not a required universal boundary. Confirm the intended lifecycle choices occur at this scale before freezing it. Protected sizes start beyond the longest validation size, using geometric growth (for example 32, 64, 128, then further sizes if calibration permits). Freeze the tested levels before protected labels/results are opened. Add larger levels only as a new exploratory/confirmatory tranche, not after looking for a favorable stopping point.

Use three explicit shape strata: balanced, comb-like deep, and random branching. For binary random trees, sample shape weights with Catalan-count dynamic programming if claiming uniform ordered binary shapes; a uniform random split of subtree size is a different distribution and must be named accordingly. For mixed arity, sample from a documented operator-count grammar and record rejection rates. Do not let one generator's easy shape distribution stand in for all compositions.

Build a deterministic holdout set of **three-operator path motifs** using a versioned hash of opaque operator IDs and ordered ports. Ensure every individual operator and every permitted two-operator adjacency remains represented in training; repair the *split rule before data generation*, not by moving test examples into training. Main training graphs exclude designated holdout motifs. Protected evaluation has separate longer graphs with familiar motifs and longer graphs containing held-out motifs. Common primitive rows and tiny subexpressions are deliberately shared; do not claim complete subtree disjointness.

Partition full-template canonical hashes before assigning literals. Canonicalization strips arbitrary node IDs but preserves topology, port order and operator identity. Keep templates, their literal instantiations and roots in the same partition. For B2, keep the entire graph and arrival sequence in one partition. No future root sequence may appear in a training cache trace for the same test episode.

For Boolean literals, include a balanced marginal condition and preregistered shifts estimated from a separately generated development pool. Balanced literals do not imply balanced answers: primary B1 also uses the root-balanced conditional sampler in §11.2, with a separate natural-distribution report. Do not select only very short-circuit-friendly inputs. Include all-XOR/no-short-circuit slices, nearly decisive AND/OR slices, deep dependent conditionals, and paired subgraphs indistinguishable locally but with different unseen costs. Report these strata separately, not only an average that hides them.

For B2, derive shared DAGs by controlled reuse of syntactically identical subexpressions. Include zero reuse as a boundary. Distinguish sequential known-root batches from causal streaming roots. Match total root count and graph content across methods. Query arrival times and memory limits are input constraints; no method sees future arrivals unless in the explicit clairvoyant ceiling.

Select memory-pressure levels from measured storage requirements of a correct lazy evaluator and full memoization on pilot graphs: a feasible small working-set regime, an intermediate regime with a real eviction choice, and enough space to retain all values. Keep the algorithmic byte cap distinct from actual laptop memory pressure. Never force the machine into swapping to simulate a scientific memory constraint.

### 6.3 Training information and leakage rules

The primary comparison supplies the full primitive dataset, short graph inputs/outputs and the same optional intermediate labels to all methods able to consume them. A separate terminal-only condition removes local labels for everyone. Training-data generation and trace construction count toward total expenditure, even when cheap. No pretrained assets are needed.

A symbol learner may use graph traversal, equality, list/port access and training facts as background. Providing the true truth table as executable background only to that learner would turn it into an oracle; providing it as training examples to everyone is the intended primary condition. Conversely, do not handicap it by forbidding rule generalization or by making it learn a parser that other models receive for free.

Training worlds may permute operator IDs to test reliance on names. A world's mapping is fixed across its train/validation/test partitions; unseen semantic remappings without demonstrations are not a fair generalization test. Separately trained worlds are independent replication units, not a way to secretly change the task at test time.

Protected input manifests and hidden labels are generated before training. Hash their contents, maintain access logs, and expose aggregate protected results only through the frozen runner. This is a procedural safeguard, not a cryptographic guarantee against a coding agent reading local files. A methodological amendment after accessing results requires fresh independent confirmation.

### 6.4 Mandatory baseline portfolio

| ID | Baseline | Required implementation and tuning | Status of comparison |
|---|---|---|---|
| O | Exact interpreter | Iterative stack/DAG evaluator; strict and short-circuit variants; reference-count cleanup and ordinary memoization | Correctness ceiling and overhead lower reference; hand-specified semantics disclosed |
| P0 | Majority / calibrated output-prior predictor | Training-only class/structure priors, measured prediction time; matched deadline fallback options for other methods | Mandatory anti-guessing control; not a compositional executor |
| S | Learned table + certificate interpreter | Fit rows only from shared primitive examples; exact lookup; derive partial certificates only with complete coverage; lazy traversal and cached variants | Strong simple learned baseline, directly competitive in finite domains |
| ILP | Popper or exhaustive tiny-rule induction | Use the same examples and declared language bias; export learned clauses to the shared executor; include search cost | Serious symbolic learner; separate environment and compatibility gate |
| G | Recurrent MPNN | Tied recurrent processor, max and sum/mean aggregation candidates, ordered port features, root demand and hints; sufficient inference steps allowed | Strong continuous graph baseline; not a shallow fixed-depth straw man |
| GD | Discrete graph processor | Discrete node/edge state and typed value slots; DNAR-based adapter or explicitly labeled paper-inspired implementation | Mandatory discrete alternative; exact provenance/changes reported |
| F | `fastnrs_paper_spec` | Learned local span segmentation, learned local solver, parallel nonoverlapping replacement; project grammar adapter | Closest rewriting baseline; not an official-code reproduction claim |
| FP | Parser-assisted FastNRS-style solver | Exact syntax identification plus the same learned solver; efficient ready queue and batching | Stronger engineering control removes segmentation as a source of apparent LRE benefit |
| Q | Shared learned reactions + conventional scheduling | Lazy DFS, ready FIFO, dependency/critical-path priority, cost-aware benefit priority, round-robin between active roots; fixed total event/time/memory limits | Mandatory causal controls; select/tune on validation, not protected data |
| LQ | Shared learned reactions + learned priority/lifecycle controller | Same observations, state capacity, actions, labels and training budget; recurrent controller; joint and frozen-kernel training | Main alternative explanation; removing the chemical names must not remove capabilities |
| NR | Region/rule policy | Paper-based NeuRewriter-style two-stage selection on the project runtime; fixed learned/certified rule bank | Additional closest-mechanism control where not already identical to LQ |
| R | Random or uniform allocation | Same engine and hard limits | Diagnostic only, never the basis of a competitive claim |

Collapse mathematically identical controls after demonstrating equivalence; do not spend the laptop budget training duplicates merely to enlarge a table. Do not exclude S, FP or LQ because they are inconveniently strong. Nonlocal published benchmark numbers appear only as context, never in a controlled head-to-head result table.

### 6.5 Concrete baseline algorithms

**Conventional demand scheduling.** Lazy DFS requests one required operand at a time and publishes immediately when its learned certificate says sufficient; try both port orderings where meaningful. Ready FIFO batches values whose inputs are present. A critical-path heuristic uses structural depth/remaining-work estimates, counting their computation. A cost-aware rule chooses the child with highest predicted probability of making the parent decidable per estimated evaluation time, using training-only estimates; after a value arrives, recompute the decision. For AND/OR with independent child outcomes and known costs, this includes the natural falsity/truth-probability-to-cost ordering. Independence is a heuristic assumption, not used to restrict competitors or declare optimality.

Across roots, include fair round-robin and a throughput-oriented priority. Report per-root tail latency and starvation, not just completed root count. Baselines use the same ordinary cache optimizations; compare LRU, reuse-count priority and a learned retention head. A clairvoyant cache/scheduler on small episodes is a headroom ceiling, never a causal competitor.

**MPNN.** Initialize node state from operator/literal/type/port information; include reverse edges and root-demand flags. At each tied step, compute ordered-edge messages with a two-layer MLP, aggregate by masked max or sum/mean, and update with a GRU or residual MLP. Decode value and ready/active states; train node hints plus root loss using the shared traces. Permit a number of test iterations that grows with graph depth or an adaptive halt head. Charge full node/edge sweeps. A version using exact ready masks should be labeled an algorithm-informed control, not denied to preserve LRE's advantage. Match hidden width/state and explore validation-selected regularization. The 2026 theory [R12] warns against extrapolating conclusions from one unsuitable aggregation/regularization choice; it does not provide an automatic guarantee for this expression grammar.

**Discrete processor.** The directly inspected DNAR processor includes discrete state encodings, selection and scalar-update components; its code also uses PyG operations and some double-precision paths [R11]. First execute it on CPU in an isolated environment and reproduce a tiny supported algorithm's behavior. Then add the expression-domain input/hint/output adapter, documenting any changes. A custom categorical-state MPNN is an acceptable additional baseline but must be named `discrete_mpnn`, not falsely presented as an unchanged DNAR reproduction. Never pass ground-truth ready/value hints during autonomous test rollout. Prove the test forward path does not call the teacher-forcing branch.

**Tiny symbolic induction.** The default finite-table learner stores all observed complete tuples and rejects conflicts. For partial certificates it enumerates all legal completions. An alternative compact rule learner enumerates conjunctions of `slot_i=value` predicates, keeps only rules consistent with every complete training row, and selects a minimum-description covering set with a declared tie rule. Include all negative labels. This is implementable without a native solver and is not a mock symbolic baseline. Popper then tests whether a stronger general-purpose language/search strategy changes the result; no extra semantic background is permitted.

### 6.6 FastNRS-style adapter, without outsourcing essential design

Implement an encoder that labels which tokens belong to immediately reducible subexpressions, a small encoder–decoder that maps such a subexpression to its value, and a deterministic position-based combiner. Process nonoverlapping spans in a batch; apply replacements from right to left or with an index map. A malformed/nonreducing replacement is a failure or threshold rejection, not an oracle-corrected answer. The project adapter may use a boundary-label scheme such as `B/I/O`; record it as an implementation choice where the inspected specification is incomplete.

The inspected paper uses local attention and sorted, sampled sinusoidal position labels. For the adapter, sample distinct position indices from a declared finite range large enough for all frozen test token lengths, sort them, and add their sinusoidal vectors. State this maximum input assumption; do not sample labels outside the implementation's range at test time. Predeclare evaluation label seeds and do not average additional samples for free.

For the closest Logic/ListOps starting configurations, the paper reports selector width 256, four heads, three/four layers and attention half-window one; solver widths 64/128 and one/two encoder and decoder layers [R1]. Use these as an anchor, not as a reason to avoid tuning the new grammar. The source's selected selector learning rates are around `3.6e-5`, whereas a stated search range is `1e-6..6e-6`; preserve this discrepancy in the reproduction notes rather than silently correcting it. For the project adapter, tune a declared log-spaced range covering both on shared validation data.

Train segmentation with token cross-entropy on all valid local spans; train the solver with teacher-forced output cross-entropy. Include end-of-output tokens and bounded malformed-output handling. Use atomic-token input segmentation and a separately recorded solver tokenizer. Tune confidence thresholds only on development data; count abstentions and every extra solver/selector invocation. Test token-label correctness, span disjointness, perfect-module reconstruction, repeated-subexpression indexing and long-input positional behavior.

Implement **two attention paths** if needed: a dense masked reference and an equivalent gathered local-window path. Masking an `L×L` attention matrix does not remove its quadratic memory allocation. Apply any equivalent locality/batching optimization to the baseline, verify FP32 forward/backward parity, and count conversion costs. This is particularly important before claiming an LRE memory advantage.

The adapter's reproduction record must distinguish paper-stated choices, project substitutions, unavailable details, and validation observations. If the adapter is weak but the parser-assisted version is strong, report the cause; do not call the comparison a victory over published FastNRS.

## 7. Confirmatory evaluation and statistical protocol

### 7.1 Claims and hierarchy

**Primary competitive estimand:** the difference in exact-accuracy area over a common, frozen log-deadline grid between LRE-T and each mandatory credible local contender, on protected longer B1 compositions. The most important matched alternative is LQ; S and FP prevent a specialized learned-interpreter comparison from overstating practical value.

**Mechanism estimand:** joint-learning interaction on B1 under common event and mutable-memory limits, with full work counters reported. This supports causal interpretation but cannot establish elapsed-time efficiency alone.

**Secondary/generalization populations:** strict A; familiar versus held-out B1 motifs; B2 lifecycle/streaming; memory and input-distribution shifts; and gated C. B2 is necessary for a retention claim, not a substitute for a failed B1 claim. Amend the primary claim and use fresh confirmation if exploratory evidence justifies a different population.

Use the following outcome tiers, rather than an arbitrary minimum percentage improvement:

1. Correct execution and reproducible length transfer only.
2. A statistically supported mechanism-specific gain over matched controls.
3. A measured Pareto improvement against the strongest local practical alternatives, with relevant regressions exposed.
4. Independently confirmed transfer to a useful workload.

Do not label tier 1 or 2 a breakthrough in adaptive capability. With no specified application, practical importance remains a reasoned statement about saved time, accuracy, memory and training amortization—not an invented business threshold.

### 7.2 Deadline grid and score

Pilot the fastest correct learned-table implementation, the strongest neural baseline and LRE without touching protected results. Choose common absolute deadlines **by input-size stratum**, spanning the transition from frequent misses to saturation. Use geometric spacing and refine where the pilot curve changes too sharply for stable integration. All methods share these deadlines. Record the endpoint and interpolation choices before confirmation.

For frozen deadlines \(d_1<\dots<d_K\), let \(a_k\) be exact query accuracy. Define

\[
U=\frac{1}{\log d_K-\log d_1}\sum_{k=1}^{K-1}
\frac{a_k+a_{k+1}}2(\log d_{k+1}-\log d_k).
\]

`U` is dimensionless in `[0,1]`. Report it with per-deadline accuracy, absolute percentage-point differences, raw deadlines in milliseconds, and all per-size strata. The aggregation population gives equal weight to the preregistered shape/motif strata, not to whichever generator produced more examples. Also publish natural-frequency results when a realistic workload supplies such frequencies.

Use separate fresh executions for each deadline when the policy conditions on its budget; truncating a long-budget trace would evaluate a different policy. For budget-unaware deterministic methods, a single trace may support a diagnostic curve, but confirmation uses the same declared measurement procedure or explicitly validates equivalence.

The maximum deadline is a workload horizon, not a claim that no benefit exists outside it. Include unrestricted-correctness/long-horizon diagnostics to distinguish deadline misses from semantic errors. Do not silently drop a method that is slow at the chosen horizon.

### 7.3 Guardrails and secondary metrics

Record exact root accuracy, all-internal-value accuracy where observed, malformed output, abstention/deadlock, event/byte-limit rejection, loops, wrong early publication, invalid cancellation, cache hit/miss/recomputation, primitive prediction count, total policy-scoring rows, bytes and peak total memory, setup time, inference time, total time and training/tuning time. Use milliseconds, seconds, bytes/MiB/GiB and explicitly named event counts.

For streaming, record each query's arrival-to-completion time, median and upper-tail latency, fraction never served, and accuracy by root position. For memory, report peak mutable state and the process/device view separately. For extrapolation, show error versus operation count and depth; a single OOD average is insufficient. Parameter count and learned-state dimension are explanatory covariates, not an automatic fairness criterion.

A primary-AUC gain accompanied by a severe accuracy or latency regression in a prespecified boundary population is a tradeoff. Use simultaneous uncertainty on these guardrails; do not hide a regression behind a composite utility score.

### 7.4 Factorial attribution

Run a primary 2×2 design with reaction kernel training (`separate/frozen` versus `jointly adapted`) and allocation (`strong fixed conventional` versus `learned`). Keep the local reaction architecture, initial semantic checkpoint, exposed examples and total optimization expenditure matched. Measure

\[
I = U_{\mathrm{joint,learned}}-U_{\mathrm{separate,learned}}
    -U_{\mathrm{joint,fixed}}+U_{\mathrm{separate,fixed}}.
\]

For the “joint/fixed” cell, update semantic/lifecycle prediction on the states induced by the fixed scheduler using the same loss opportunities, while keeping its selection order fixed. This avoids an undefined cell in which a nonexistent scheduler is supposedly trained. Separate lifecycle-policy ablations are necessary because a fixed ordering need not fix cache retention.

Here “fixed conventional” means a fixed, nonlearned scheduling algorithm under a fixed total allowance, not a predetermined input-independent event sequence. Lazy evaluation and cost-aware conventional scheduling may adapt to observed values. A positive main effect of a learned scheduler is not evidence of interaction. A positive interaction is not proof of ecological novelty. Follow with LQ's jointly trained recurrent controller, equal-memory hidden state, and the exact compiled-controller equivalence test. If the best fixed scheduler differs across cells, choose it on validation and freeze it; do not let test-specific scheduler selection contaminate the contrast.

### 7.5 Tuning and stopping before confirmation

Give every principal learner a declared search space, shared supervision tracks, and a common **tuning-expenditure envelope**, measured in actual local device-hours and examples processed. A baseline may need more time to reach a faithful configuration; report both equal-expenditure and adequately tuned practical comparisons rather than quietly undertraining it. Reuse generic optimizations across methods.

Use pilots to identify capacity bottlenecks, optimizer instability, semantic ceiling and overhead. Extend a learning curve while it shows unresolved systematic improvement worth the measured cost. Stop semantic training on complete finite rows when correctness is certified or validation/probe failures are explained, not at a round-number epoch. Do not terminate an expensive baseline before its relevant learning curve is understood simply to give LRE a favorable comparison.

Before accessing protected outcomes, freeze the training-data recipe, seeds, methods, hyperparameters, stopping/checkpoint-selection rules, deadline grid, memory strata, primary contrasts, multiplicity procedure and sample-size calculation. Record amendments with reasons, timestamps and a new confirmation set. A failed or unfavorable run is retained.

### 7.6 Independent units, pilot sizing and fixed-sample confirmation

There are at least two sources of uncertainty: **independently trained systems** and **independent program-template/episode clusters**. Several literal assignments of one template, several roots in one graph, repeated timing of one query, and repeated checkpoints of one training run are not independent units. Seeds alone are not a population definition; they instantiate independent training procedures or input generation.

Use paired independent training blocks: in block `r`, methods share the training world/data recipe and test episodes, but each receives independently initialized model randomness as declared. A block may include separately generated training data to support a claim beyond one fixed training dataset. Also report the conditional-on-training-data result when relevant.

During exploration, estimate between-block, between-template and interaction variance from crossed pilot data, plus timing-induced variation in the AUC. Determine a fixed number of training blocks and protected template clusters before confirmation. A useful planning approximation for paired differences is

\[
\mathrm{Var}(\bar\Delta)\approx\sigma^2_{\mathrm{train}}/R+
\sigma^2_{\mathrm{template}}/N+
\sigma^2_{\mathrm{interaction}}/(RN).
\]

For a planning effect \(\delta\), a simple independent-pair approximation is
\(n\approx(z_{1-\alpha/2}+z_{1-\beta})^2\sigma_d^2/\delta^2\).
Use simulation from the crossed pilot to choose the actual `R,N`, since the simple expression omits hierarchy. Validate interval coverage on null simulations and injected known effects. Include multiplicity in the sizing simulation.

No fixed success percentage or arbitrary seed quota is imposed. The pilot report must give **a detectable-effect versus device-time table**, confidence-width estimates and timing-resolution limits. Choose the least-cost design that can resolve the claimed advantage beyond those measured limits and the opportunity cost of the required operation savings. The scientific planning effect and confidence-width target must be written down before protected evaluation; they are not a threshold below which an actual effect is declared nonexistent. If the effect is smaller than the laptop can resolve with the justified workload, report it as inconclusive or too small to substantiate, not as a win.

Use a fixed-sample confirmatory analysis with 95% intervals, treating this confidence level as a declared error-control convention. Use a crossed/hierarchical bootstrap resampling training blocks and template clusters while preserving method pairing, all literal instantiations, all deadlines and all roots within each selected cluster. With very few training blocks, bootstrap uncertainty is unreliable: report block-level paired results and a conservative paired analysis, and do not imply thousands of queries compensate for missing training replication. Add replication before confirmation when the pilot shows training variance dominates.

Use Holm correction for the finite family of primary superiority contrasts and prespecified interaction tests, or an explicitly validated simultaneous bootstrap procedure. Report estimates and unadjusted/adjusted uncertainty rather than only a p-value. A superiority claim requires the relevant corrected interval above zero; a claim to beat **all** credible local baselines requires all those comparisons, not just the easiest one. Treat extensive ablations as exploratory unless individually preregistered or confirmed on new data.

Do not keep adding seeds until a p-value crosses a threshold. After the fixed confirmatory run, further work is a new experiment with its own protected data and analysis. Negative evidence, practical dominance by a simple method, inadequate precision, and implementation failure are different outcomes.

### 7.7 Timing protocol

Use an idle, plugged-in laptop with recorded power mode and thermal/system context. Do not claim exclusive hardware control without measuring it. Exclude downloads, compilation and package installation from inference, but report them as setup costs; include task-specific parsing, graph construction, candidate indexing, model input assembly, CPU/device transfers, selection, commits, cleanup and output materialization in end-to-end inference.

Warm up every actual shape/backend path until latency stabilizes under a preregistered criterion derived from pilot variation. Run CPU and MPS in separate measured blocks when contention would distort results. Synchronize MPS before starting and after completing measured work. A CPU timer around asynchronous launches without synchronization is invalid [R18]. Fix thread settings after a throughput pilot and give baselines appropriate settings too.

Randomize/interleave method order in paired blocks. Record raw timings, warm-ups, caching state, temperature/pressure information available without special equipment, and background interruptions. Treat repeated timings as measurement replicates, not extra independent accuracy samples. Report median and relevant tail summaries with paired-block uncertainty. Cold parsing/setup and warm reusable-model serving are separate modes; no hidden precomputed evaluation graph or persistent cache is allowed in a supposedly cold comparison.

For primary B1 deadlines, record the end-to-end completion time after required housekeeping, output materialization and final synchronization. Retain earlier raw publication times only as diagnostics. For separately labeled streaming response latency, record when the operation/synchronization makes an answer available and account for subsequent maintenance in service time and backlog. An action started before the deadline but completed afterward is a miss. Check wall time at reaction boundaries; record overshoot. Do not make an LRE atomic action arbitrarily large to evade checks. Define equivalent checkpoints for batched baselines. A watchdog enforces a separate safety limit without converting an overrun into a valid prediction.

## 8. MacBook M3 Max feasibility and resource adaptation

### 8.1 Hardware discovery comes first

Apple's M3 Max configurations differ in unified memory and processor configuration [R17]. Do not assume a maximum-memory machine. The following are **existing macOS commands**, not scripts already present in this project:

```bash
uname -m
sw_vers
sysctl -n hw.memsize
sysctl -n hw.ncpu
df -h .
vm_stat
sysctl vm.swapusage
pmset -g
xcrun --find clang
system_profiler SPHardwareDataType SPDisplaysDataType
```

Save a sanitized inventory: remove serial numbers, hardware UUIDs and personal paths before publishing it. Record actual free storage, process architecture, macOS build, power mode, CPU thread count and GPU configuration. Confirm that Python is native arm64 rather than running through Rosetta. Discovery may identify a machine other than the expected target; report that rather than labeling its timings M3 Max results.

### 8.2 Stack and dependency policy

Use Python, NumPy, PyTorch, PyYAML, pytest and a process-memory utility such as psutil for the main implementation. Standard-library JSON/SQLite is sufficient for records; a heavy experiment platform is unnecessary. PyTorch has a documented MPS backend and memory/synchronization interfaces [R18–R20], but this does not prove that every baseline kernel, dtype or backward path works.

Select the newest **released** PyTorch/CPython combination with an actual native arm64 wheel supporting the discovered macOS version, verify installation and critical tests, then lock exact versions and hashes. The inspected documentation identified itself as PyTorch 2.14; that is documentation evidence, not proof that a compatible wheel is installed on this laptop. Do not hardcode a nightly build to appear current. The inspected backend page's example requires macOS 14 or later; verify the selected release's own requirements [R19].

Use separate environments for main training, DNAR reproduction when necessary, and Popper. The current Popper manifest inspected on the research date requires Python `>=3.14` and lists native/solver-related dependencies including `janus_swi`, `clingo`, `python-sat`, `pypblib` and `ortools` [R14]. This is a concrete reason not to assume a single convenient Python environment will run all baselines. Pin a lawful supported revision, build/test it, and record the exact revision; do not silently use an older algorithm under a current label.

DNAR's inspected requirements include PyTorch, PyG, NumPy and tqdm; processor code uses grouped scatter/selection and some double-precision operations [R11]. Its CPU route is the faithful first route. Replacing those paths with dense gathers or FP32 can be a useful adapter but requires equivalence checks and explicit labeling. SALSA-CLRS reinforces the need to avoid unnecessary dense graph representations [R22]; its published results are not expression-task baselines.

MLX is a possible later optimization, not a required second implementation stack. It is not justified merely by Apple Silicon branding. Only add it after profiling identifies a bottleneck for which equivalent operators and training behavior can be demonstrated. Do not require CUDA, Triton, remote GPUs, paid APIs, containers that emulate a different accelerator, or downloaded large model weights.

On a fresh laptop, first locate a trusted native CPython and Apple command-line tools. If absent, install from their official distributions with any required local approval; do not assume administrative access or silently upgrade macOS. Existing shell commands after a compatible Python has been selected are:

```bash
python3 -c "import sys, platform; print(sys.version); print(platform.machine())"
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
```

The agent must then create the project's `pyproject.toml`, including the selected runtime dependencies and a test/development extra, and install that local project using the locked environment. `pip install -e '.[dev]'` becomes meaningful only after that file/extra exists. Record installer sources and lock resolution, and stop for explicit approval on blocked administrative prerequisites rather than pretending setup succeeded.

### 8.3 Operation compatibility gate

Before any training run, execute a CPU reference and candidate MPS test for the actual operations used: embedding lookup, linear layers, activation, GRU, masked softmax, cross-entropy, gather/indexing, variable-length padding, local attention, loss reduction, optimizer updates, and serialization. Test at least one complete forward/backward/update cycle with the actual shapes, plus several successive steps to expose state/gradient errors.

Start with MPS fallback disabled and set environment variables **before importing PyTorch**. The documented fallback option can move unsupported operations to CPU [R20]; it must not silently contaminate a claimed all-MPS measurement. An explicitly hybrid implementation is permitted if it is labeled and all transfers/synchronizations are timed. Never disable memory high-watermark safety to force a run to fit.

Check `torch.backends.mps.is_built()` and `is_available()`; inspect API availability before using `torch.mps.synchronize`, memory counters or RNG-state accessors. Compare finite outputs and gradients with the CPU implementation under preregistered FP32 tolerances derived from a tiny exact/reference test. For categorical decisions, also compare the chosen actions and inspect near-tie logits. A small tensor difference that changes a discrete trajectory is material even if `allclose` passes.

If an operation is unsupported, choose in order: a mathematically equivalent supported implementation with parity tests; a fully CPU implementation; or a documented blocked baseline/claim. Do not silently remove an action, detach its gradients, change its precision, or reduce its problem size.

### 8.4 Peak-memory model

For a dense FP32 Adam-family model with \(P\) parameters, weights, gradients and two optimizer states alone are approximately \(16P\) bytes. Add any master copies, temporary update buffers, saved activations, framework allocation reserves, static graphs, dynamic slots, candidate tensors, output buffers and checkpoint serialization buffers.

A useful first model is

\[
M_{\rm peak}\approx16P+\alpha\,(4BTN_a d)+M_{\rm graph}
+M_{\rm objects}+M_{\rm candidates}+M_{\rm workspace}+M_{\rm checkpoint},
\]

where `B` is minibatch environments, `T` retained differentiable steps, `N_a` active objects per step, `d` hidden width, and \(\alpha\) is a **measured** activation multiplier. It is not valid to assume \(\alpha=1\) for a multilayer recurrent model. Shared long-lived tensors and graph copies require separate accounting.

For FastNRS-style dense masked attention, an attention-score tensor alone is \(4BHL^2\) bytes. For an equivalent local-window implementation it is approximately \(4BHL(2k+1)\), excluding projections and other activations. Thus locality in the mathematical mask must be reflected in the implementation to realize linear rather than quadratic workspace.

Calculated examples, not measurements:

| Allocation | Assumptions | Size |
|---|---|---:|
| FP32 weights + gradients + Adam states | 1 million parameters | 15.26 MiB |
| One retained activation tensor | `B=16,T=8,N_a=256,d=64` | 8 MiB |
| One dense attention-score tensor | `B=8,H=4,L=2048` | 512 MiB |
| Corresponding local-window scores | same, half-window 1 | 0.75 MiB |
| One recurrent activation tensor | `B=8,T=16,N=4096,d=128` | 256 MiB |

These examples suggest small local models need not require maximum unified memory. They do **not** estimate full peak usage or runtime. Dynamic Python representations can dominate small tensors; measure both reference and packed forms.

Unified CPU/GPU memory is shared. Process RSS, framework allocated memory and driver allocated memory are overlapping views, not automatically additive physical usage. Record them separately with system pressure/swap behavior. Use a conservative physical-memory envelope below observed pressure onset and the device's recommended limit where exposed; retain headroom for the operating system, peak workspaces and checkpoints. Derive the envelope from a controlled ramp, not by multiplying advertised memory by an unexplained constant.

### 8.5 Calibration and scale rules

For each substantial stage, run a geometric shape/batch ramp beginning with one short episode and one optimizer step. Record throughput, complete-step latency, peak memory, allocation growth, system pressure, swap changes and checkpoint size. Fit a simple empirical cost model separately for training, inference and data generation; validate it at a withheld intermediate configuration.

Increase batch size while throughput improves without approaching the measured safe envelope. Prefer microbatching and gradient accumulation to changing the semantics, data or model. Changing accumulation can affect stochastic optimization; preserve effective batch size and document any numerical difference. Use full short-episode unrolls initially. Truncated backpropagation changes credit assignment and is a separate training variant, not a silent memory fix.

Increase protected graph sizes by geometric steps only in the pre-freeze pilot and only when exact/table baselines and the comparison harness remain valid. Run long inference without storing autograd graphs. If object count or fan-out exceeds a padding bucket, move to a larger bucket or streaming representation; never drop edges or nodes.

Estimate total training time from measured optimizer updates and rollout generation:

\[
T_{\rm campaign}=\sum_{m,c,r}(u_{mcr}\,t^{\rm update}_{mc}
+e_{mcr}\,t^{\rm rollout}_{mc})+T_{\rm tuning}+T_{\rm evaluation}+T_{\rm checks}.
\]

Do not double-count rollouts already included in measured update time. Publish assumptions, a central estimate and a range from observed variability. For illustration only, 10,000 measured-complete updates at 20, 100 or 500 ms would require about 3.3, 16.7 or 83.3 minutes **before other campaign costs**. Neither 10,000 updates nor these speeds is an asserted requirement or measurement.

If runtime is dominated by serial Python/CPU-device crossings, prefer a CPU-only policy, packed arrays, batched independent episodes and shared batching optimizations before larger models. An accelerator is not mandatory for feasibility. If a faithfully tuned baseline needs longer local execution, a longer run is acceptable when the information gained justifies it; incomparable remote results do not replace it.

### 8.6 Stage resource envelopes

| Stage | Main scaling / storage | Calibration rule and implication |
|---|---|---|
| Discovery and checks | Small Python/native environments; no large weights | Measure environment disk size and tiny operation tests; block unsupported dependencies early |
| Data generation | `O(total generated nodes)` plus trace length; conditional-generation overhead | Generate deterministically in shards; benchmark nodes/s and compressed bytes/episode; retain seeds and protected manifests |
| Exact/table baselines | `O(N+E)` graph storage; evaluator frontier/cache | Establish throughput and minimum feasible working set before learned allocation |
| Primitive learners | 147 arithmetic rows or 22 Boolean full rows plus partial labels; small model | Train to verified finite-table behavior; expose local failures before long rollouts |
| Neural rewrite/GNN baselines | Parameters, activations, attention or message workspace | Per-method ramp; use faithful local-window/gather paths or CPU; no hidden batch/model reductions |
| LRE rollout training | Active object count × episode steps × policy width, plus saved state | Measure full episode generation and backward cost; batch independent episodes; checkpoint boundaries |
| Tuning and confirmation | Number of independent fits × calibrated fit cost, plus paired inference blocks | Select using learning curves, variance and the fixed analysis design; report infeasible precision honestly |
| Optional transfer | Actual workload preprocessing, execution, and provenance | No download/integration until core headroom and attribution gates pass |

Store full event traces for correctness fixtures, failure cases and a fixed sampled subset of routine episodes. Store complete per-query outcomes/counters for every confirmatory run. Do not fill the disk with every intermediate tensor. Estimate disk as environment copies + generated shards + checkpoints + raw outcomes + selected traces; verify remaining space before each campaign tranche.

### 8.7 Recovery

Write checkpoints to a temporary path, flush, then atomically rename. Save model/optimizer/scheduler state, Python/NumPy/framework RNGs, data sampler position, curriculum state, event budget, object tables, generation counters, pending deliveries/cleanup, cache epoch and experiment manifest hash. Save an episode-level checkpoint only at a completed transaction boundary.

After interruption, verify source/config/lock/data hashes before resuming. If device-specific bitwise replay is not supported, restart the affected paired measurement block and keep the interrupted raw records. Do not splice incompatible halves of a trajectory or restart only an unfavorable method. Retry memory failures at a smaller microbatch with the same effective batch when valid; changing model, precision, sequence population or loss requires a recorded amendment and a new comparable run.

## 9. Repository, interfaces and execution records

### 9.1 Minimal layout

```text
lre-poc/
  README.md
  research_poc_plan.md
  pyproject.toml
  locks/                         # exact versions/hashes; separate baseline environments
  configs/
    tasks/{strict,bool,stream}.yaml
    methods/{table,lre,lq,mpnn,discrete,fastnrs}.yaml
    pilot.yaml
    confirmatory.yaml             # created and frozen only at the freeze gate
  src/lre_poc/
    cli.py
    data/{grammar,sampling,splits,oracle,manifest}.py
    engine/{objects,transactions,candidates,budget,reference,packed}.py
    models/{local_kernel,controller,mpnn,discrete,fastnrs}.py
    baselines/{interpreter,table,rule_induction,schedulers,adapters}.py
    train/{supervised,rollout,losses,checkpoint}.py
    eval/{runner,timing,metrics,statistics,leakage}.py
    platform/{doctor,opcheck,calibrate}.py
  tests/
    test_grammar.py
    test_certificates.py
    test_transactions.py
    test_equivalence.py
    test_backend_parity.py
    test_no_teacher_leakage.py
    test_statistics_null.py
    test_resume.py
  data/{train,validation,protected}/
  runs/<run_id>/{manifest.json,config.yaml,metrics.jsonl,queries.jsonl,events/,checkpoints/}
  artifacts/{hardware,calibration,provenance,decisions,audits,reports}/
  docs/{METHOD_AMENDMENTS.md,DECISIONS.md,PROGRESS.md}
```

This is a proposed repository, not an existing downloadable implementation. Create only modules needed by the next gate; the layout is a contract, not a requirement for excessive scaffolding.

### 9.2 Shared method interface

Use a small adapter interface so baselines are not forced to share an unfavorable implementation. Data visibility and output accounting are shared; internal execution can differ.

```python
from dataclasses import dataclass
from typing import Protocol, Iterable

@dataclass(frozen=True)
class Limits:
    deadline_ns: int | None
    mutable_bytes: int | None
    event_cap: int | None            # meaningful for shared-runtime controls only
    safety_timeout_ns: int

@dataclass(frozen=True)
class QueryResult:
    query_id: str
    prediction: int | None
    status: str                     # answered / abstained / timeout / invalid / deadlock
    completion_ns: int | None
    counters: dict[str, int | float]

class Method(Protocol):
    def fit(self, train_manifest: str, config: dict, run_dir: str) -> None: ...
    def load(self, checkpoint: str) -> None: ...
    def predict_episode(self, episode, limits: Limits) -> Iterable[QueryResult]: ...
    def save(self, checkpoint: str) -> None: ...
    def provenance(self) -> dict: ...
```

The evaluator owns ground truth and computes correctness **after** prediction. It must not inject truth-dependent fields into `episode`, `Limits`, candidate masks or runtime callbacks. For streaming, a service wrapper reveals roots only when due. A fixed graph, device, policy checkpoint and random seed must be sufficient to replay a non-timing trajectory.

For shared-runtime methods, use the same event-cap mapping at a given deadline. Select that mapping on validation from the fastest correct shared-runtime execution, freeze it, and give LRE/LQ the same budget information. Non-runtime baselines are controlled by actual deadlines and memory, with their inference-step limits chosen on validation. Do not interpret a GNN sweep, a token decode and an object reaction as equal compute units.

### 9.3 Required record schemas

Each `manifest.json` includes:

```json
{
  "schema_version": "1",
  "run_id": "unique-id",
  "phase": "exploratory-or-confirmatory",
  "source_revision": "actual-git-commit",
  "dirty_patch_sha256": "hash-or-null",
  "config_sha256": "actual-hash",
  "dependency_lock_sha256": "actual-hash",
  "dataset_manifest_sha256": "actual-hash",
  "method": "registered-method-id",
  "baseline_revision": "actual-commit-or-paper-specification-id",
  "training_world_seed": 0,
  "model_seed": 0,
  "evaluation_seed": 0,
  "backend": "cpu-or-mps-or-explicit-hybrid",
  "precision": "float32-or-declared-alternative",
  "hardware_record": "relative-path",
  "started_at_utc": "ISO-8601",
  "status": "running-complete-failed-interrupted",
  "amendment_id": null
}
```

The strings and zero seeds above are schema examples, not real results. Populate actual values.

Each query row includes episode/template/graph/query IDs; domain/size/depth/shape/motif/label stratum; arrival time; deadline; memory/event limit; prediction; ground truth stored only by evaluator; correctness; failure category; setup/inference/total latency; completion time; full counter vector; training block; timing block; and raw artifact paths. Use `null`, not zero, for unavailable measurements. Units belong in field names (`latency_ns`, `peak_mutable_bytes`).

Each decision record includes date, work-package ID, evidence files, decision (`continue/revise/stop/inconclusive`), reason, affected claims, next action and whether protected results had been accessed. A progress record should let a fresh coding agent resume without chat memory.

### 9.4 Command conventions

The following are **proposed commands to implement**, not commands that exist today:

```bash
python -m lre_poc.cli doctor --out artifacts/hardware
python -m lre_poc.cli opcheck --backend cpu --out artifacts/calibration/cpu
PYTORCH_ENABLE_MPS_FALLBACK=0 python -m lre_poc.cli opcheck --backend mps --out artifacts/calibration/mps
python -m lre_poc.cli generate --config configs/tasks/bool.yaml --split development
python -m lre_poc.cli verify --suite semantics,transactions,equivalence,leakage
python -m lre_poc.cli calibrate --config configs/pilot.yaml
python -m lre_poc.cli train --method table --config configs/pilot.yaml
python -m lre_poc.cli train --method lre --config configs/pilot.yaml
python -m lre_poc.cli freeze --config configs/confirmatory.yaml
python -m lre_poc.cli evaluate --manifest artifacts/decisions/frozen_manifest.json
python -m lre_poc.cli report --manifest artifacts/decisions/frozen_manifest.json
```

`freeze` must reject missing baseline provenance, unresolved method substitutions, absent power/precision justification, unsupported backend paths, and changed protected manifest hashes. `evaluate` must refuse a mismatched checkpoint/config or an unregistered method unless explicitly in exploratory mode. `report` must include failed and unfavorable runs, not just the best checkpoints.

The coding agent may make routine non-destructive implementation decisions such as module organization, test names and equivalent container types. Changes to the hypothesis, task population, observable information, reward, primitive supervision, primary metric, baseline configuration class or interpretation require a method amendment.

## 10. Risk-ordered work packages and decision gates

All work packages inherit the logging, unit, provenance and recovery contracts above. Report costs measured locally; do not replace a gate with an arbitrary elapsed-time deadline. The dependency order is intentional.

### WP00 — Discover hardware, lawful assets and viable numerical paths

**Risk/objective:** nominal Apple Silicon compatibility may conceal unsupported operations or incompatible baseline environments.  
**Dependencies/entry:** this document and the target laptop; no training required.

**Actions/files:** implement `platform/doctor.py` using §8.1 and save sanitized hardware/storage records. Create isolated main and baseline environments with exact locks. Record licenses for copied code and assets. Implement `opcheck.py`; execute the actual dense/GRU/indexing/local-attention forward and backward paths with fallback disabled. Inspect and pin baseline repository revisions; retain current source/version access limitations rather than inventing identifiers.

**Tests/outputs:** `hardware.json`, environment locks, `license_manifest.json`, `opcheck.json`, numerical parity fixtures, actual available bytes and disk space. Verify CPU/MPS availability separately from correctness. Attempt current Popper and original DNAR CPU smoke tests before relying on them.

**Resources:** small installs and tiny tensors; measure disk usage and op-test time. No large weights or benchmark download is justified here.

**Gate:** continue on at least one faithful local path. If MPS fails but CPU works, use CPU and recalibrate. If a baseline cannot run, isolate whether its adapter can be faithfully reproduced; otherwise block claims against that published system and keep stronger directly runnable controls. If no faithful core numerical path works, stop implementation and document the missing prerequisite rather than altering the mechanism silently.

### WP01 — Establish semantic ceilings and a sensitive harness

**Risk/objective:** the task may be solved perfectly by a tiny learner, or the proposed allocation benefit may be mathematically absent.  
**Dependencies/entry:** WP00 CPU path; no trained neural models.

**Actions/files:** implement `grammar.py`, `oracle.py`, `sampling.py` and `table.py`; run Appendix A; enumerate primitive tables and partial certificates; implement strict and lazy oracle interpreters. Add tests of output priors, equal-method comparisons and deliberate wrong-answer/slowdown injections. Generate development-only shape/motif examples and verify labels with independently structured reference evaluators.

**Tests/outputs:** exact primitive coverage; zero oracle disagreement; correct strict reduction counts; `headroom_initial.json`; null/effect-detection harness results; generator distribution and rejection-rate reports.

**Resources:** tiny exact enumerations initially; extend structural tests based on measured cost rather than enumerating exponentially large spaces blindly.

**Gate:** any semantic/harness discrepancy blocks all training. If a perfect learned table dominates the entire measured finite task, record that ceiling immediately. Continue only to answer a still-open allocation/learning interaction question with demonstrated headroom, or conduct a bounded diagnostic of why no advantage remains. A no-headroom strict result directs attention to B1/B2, not to a weakened baseline.

### WP02 — Implement and verify the reaction engine before learning

**Risk/objective:** birth/death, shared demands and conservation may be incorrect or expensive.  
**Dependencies/entry:** WP01 semantic fixtures.

**Actions/files:** implement reference object tables, syntactic candidate generation, charged cleanup, atomic transactions and the budget ledger. Drive the engine with an exact table and hand-coded schedules. Implement `compiled_stateful_controller` using a distinct state encoding but the same action choices. Add generation-counter, cancel/shared-child, byte-reservation and deadline-overshoot tests.

**Tests/outputs:** deterministic event traces; reference-versus-compiled equality; valid finite termination; exact budget reconciliation; no ownership leaks; correct epoch invalidation. Log all reads/scoring/allocation costs. Provide a packed-array version only after reference correctness.

**Resources:** pilot active-object/edge ramps; measure bytes/object, bytes/waiter, bytes/value and event overhead. Compare direct interpreter versus shared reaction runtime to expose the structural tax before neural scoring.

**Gate:** continue only with correct transactions and an explained overhead profile. If the engine is slower but a measurable scheduling opportunity remains larger than the overhead, continue. If bookkeeping already exceeds the upper-bound opportunity, test equivalent shared optimizations, then stop/narrow the practical claim if dominance persists. This is not evidence that every possible chemical substrate is useless.

### WP03 — Establish credible competitors and reproduction records

**Risk/objective:** apparent improvement may be due to weak/outdated baselines.  
**Dependencies/entry:** WP01 datasets; WP02 runtime for shared controls; WP00 environments.

**Actions/files:** implement O/S/ILP, conventional schedules, G/GD, F/FP and LQ as in §6. Native baselines may use their own optimized executor. Build small supported-task DNAR checks before adapting the domain; implement autonomous hint-free test rollout. Build perfect-module and span-replacement tests for F. Run Popper or the exact small rule inducer with explicit background/bias. Create a baseline ledger listing actual revisions, parameter counts, substitutions, tuning spaces and data access.

**Tests/outputs:** baseline unit/integration tests; primitive and ID/OOD development learning curves; exact parser/table accuracy; tiny autonomous graph-algorithm check; FastNRS adapter error taxonomy; license/reproduction notes. Compare at least the strongest simple and learned control before training an elaborate ecology.

**Resources:** per-method calibration; reuse one generator/split and shared semantic checkpoint where the contrast requires it. No full SQL engine, large evolutionary soup or foundation model is needed.

**Gate:** inadequate baseline convergence triggers debugging/tuning, not a premature comparison. A missing official FastNRS implementation is reported explicitly; F/FP remain controlled adaptations. If an important baseline is not faithful enough, do not claim to beat the published method. Proceed with a narrower reproducible comparison only if S, FP, strong G/GD and LQ are credible.

### WP04 — Train the minimum complete LRE-T and factorial controls

**Risk/objective:** discrete lifecycle choices may not learn, or joint training may degrade local semantics.  
**Dependencies/entry:** WP02 invariants and WP03 credible basic competitors.

**Actions/files:** implement the kernel and losses in §5. Train primitive/certificate predictions, warm-start from shared traces, then run on-policy short episodes. Keep `REFINE` off initially. Implement frozen/separate/joint cells and the stateful conventional controller with equal information and explicit expenditure accounting. Probe every primitive row after each checkpoint selection stage.

**Tests/outputs:** full semantic probe tables; autonomous short-graph success; gradient and policy-entropy diagnostics; birth/cancel/retention distributions; work/time breakdown; checkpoint and resume tests. Confirm that no inference path calls the training oracle or supplies true hidden values.

**Resources:** measured full short-episode training cost; adjust microbatching before changing semantics. Retain learning curves instead of choosing a fixed epoch count as proof of convergence.

**Gate:** failure to imitate short exact executions is an implementation/optimization problem until ruled out, not compositional evidence. Once short execution works, test longer development graphs. Add `REFINE` only for a measured residual computation-depth hypothesis. If LRE does not exceed simple controls despite adequate fitting, retain the negative finding; do not remove those controls.

### WP05 — Pilot headroom, power, timing and hardware limits

**Risk/objective:** either the mechanism has no usable opportunity, or a prospective experiment cannot distinguish its effect from noise.  
**Dependencies/entry:** WP03–WP04 functioning development methods; protected results unopened.

**Actions/files:** run development-only B1/B2 and strict A. On tiny graphs compute a valid-trace, same-action-grammar clairvoyant scheduling/storage ceiling; it may know hidden/future information but may not create an unwitnessed certificate or skip charged transactions. Compare the best practical schedule with that ceiling. Run backend/shape calibration, fit cost models, estimate crossed variance, and simulate proposed confidence procedures. Produce detectable-effect/device-time and memory/deadline reports.

**Tests/outputs:** `pilot_analysis.json`, headroom by condition, baseline Pareto curves, operation-cost breakdown, null interval coverage, sample-size/precision plan, proposed deadline/memory grids and campaign cost range. Diagnose whether any apparent gain is just output guessing or semantic-confidence calibration.

**Resources:** expand only the pilot dimension that resolves a decision—training replication for unstable learners, templates for input variance, or repeated timings for measurement noise. Do not launch a broad Cartesian sweep without information-gain justification.

**Gate:** if credible headroom and an attributable development signal remain, freeze a powered local test. If the proxy offers no headroom, retain the null result and use an evidence-backed B2/transfer amendment rather than more of the same data. If estimated precision is infeasible locally, state the exact detectable range and narrow or declare inconclusive; do not claim a tiny noisy gain.

### WP06 — Freeze the confirmatory contract

**Risk/objective:** adaptive selection or hidden information changes could invalidate inference.  
**Dependencies/entry:** WP05 cost/variance plan, credible baselines and all correctness gates.

**Actions/files:** create `configs/confirmatory.yaml` and a signed/hash-stamped manifest containing hypotheses, populations, model configs, selected checkpoints or training rules, independent blocks, seeds, deadline grids, budget mapping, memory limits, precision, primary tests, multiplicity and resumption behavior. Generate/hash protected inputs/labels from reserved seeds. Audit all feature and teacher-forcing pathways.

**Tests/outputs:** successful `freeze` validation; frozen source and lock hashes; `PRE_REGISTRATION.md`; protected access log; explicit unresolved limitations. Another local code path verifies that duplicate templates do not cross partitions.

**Resources:** small manifests and deterministic generation; no reason for expensive training here.

**Gate:** missing fairness, provenance, power/precision or compatibility information blocks confirmation. Routine corrections before protected access are logged; changes after access require a new confirmation tranche.

### WP07 — Execute the controlled comparison

**Risk/objective:** determine whether the predeclared advantage is genuine.  
**Dependencies/entry:** WP06 frozen contract.

**Actions/files:** train independent blocks using frozen rules or load the preregistered checkpoints; run paired/interleaved end-to-end timing and exact evaluation; enforce limits and preserve every result. Use the fixed statistical procedure, no test-driven retuning. Monitor system pressure and apply the declared invalid-block/restart rule symmetrically.

**Tests/outputs:** complete query rows and counters; paired estimates/intervals; corrected primary tests; boundary regressions; failed-run records; exactness versus size/depth; total campaign cost.

**Resources:** use WP05 measured cost and checkpoint schedule. If interruption or thermal/system variation invalidates a block, rerun that block according to the preregistration and retain the original data.

**Gate:** classify confirmed improvement, non-superiority/negative, tradeoff or inconclusive. A failed deadline does not become an excluded sample. Do not extend the sample because significance is just out of reach.

### WP08 — Attribute, stress-test and independently confirm

**Risk/objective:** a positive result may not arise from the proposed mechanism.  
**Dependencies/entry:** WP07 result or a clearly labeled exploratory diagnosis.

**Actions/files:** execute the ablations in §11 with priority determined by the observed effect. Match state, caching, batching, semantic supervision and optimization. Replay exact compiled-controller equivalence. Test held-out motifs, label shifts, high fan-out, very low/high memory, locality-impossible pairs and changed node IDs. Conduct independent confirmation with fresh training/input blocks for any revised or expanded claim.

**Tests/outputs:** causal contrast tables, intervention traces, regression/failure taxonomy, compiled-equivalence checks and fresh confirmation report. Quantify reuse and recurring reaction motifs; do not infer computational abstraction from mere cache hits.

**Resources:** prioritize the cheapest plausible explanation of the effect; do not blindly run all possible ablations. Newly confirmatory tests have their own fixed sample plans.

**Gate:** a gain erased by ordinary caching, state matching or better tuning narrows the claim accordingly. A stable interaction surviving those tests justifies transfer work. Inconsistent results require explicit heterogeneity analysis or an inconclusive verdict, not selective reporting.

### WP09 — Gated transfer or a justified architectural extension

**Risk/objective:** synthetic success may not be useful, or the smallest proxy may omit the hypothesized mechanism.  
**Dependencies/entry:** WP08 explains the gain, or WP05 documents a specific missing headroom mechanism without evidence against it.

**Actions/files:** choose **one** evidence-triggered branch from §12. For transfer, fix actual predicate/query data, licensing, input costs and optimized conventional execution before evaluating. For macro or richer-value experiments, specify the added language, supervision and exact symbolic competitor before coding. Preserve the original core results.

**Tests/outputs:** amendment, new task/interface specification, source/license manifest, new power/feasibility calibration and fresh confirmation. Demonstrate that the new setting expresses the mechanism missing from the old one.

**Resources:** recalibrate actual workloads; no remote GPU requirement. Do not download or integrate large assets until the branch is justified.

**Gate:** successful transfer upgrades only the supported claim. No transfer, unavailable assets or local resource blockage limits scope; it does not license a production-performance assertion. A pivot that changes the mechanism must be named as a pivot.

### WP10 — Reproducibility package and research decision

**Risk/objective:** the result may be impossible to reproduce or overstated.  
**Dependencies/entry:** completed or stopped program with all decision records retained.

**Actions/files:** generate the final report from raw manifests/outcomes; include all preregistered methods and unsuccessful runs. Audit one clean-environment rerun of a small exact fixture, one learned checkpoint and the statistics pipeline. Assemble licensed code/configs, environment locks, seeds, manifests, relevant checkpoints, raw per-query outcomes and sampled traces. Do not redistribute third-party assets beyond their permissions.

**Tests/outputs:** `REPORT.md`, machine-readable result tables, `REPRODUCE.md`, final limitations and continuation decision, clean-rerun hashes and differences. Separate confirmed, exploratory and unexecuted results visibly.

**Resources:** disk/hash/replay calibration; no requirement to rerun every training fit merely to create a package.

**Gate:** publish only claims traceable to the completed comparisons. Stop with a useful negative or inconclusive report when warranted. Resume work only through an evidence-backed decision record, not because the desired narrative remains unproven.

## 11. Attribution, robustness and failure analysis

### 11.1 Required controls, ordered by explanatory power

| Control / intervention | Keep matched | What a disappearing gain means |
|---|---|---|
| Replace learned semantics with exact learned-table semantics in every shared-runtime method | Scheduling/lifecycle architecture, inputs, limits | Gain was semantic fitting/calibration rather than resource organization |
| Replace allocation with best validation-selected conventional schedule | Kernel checkpoint, state/lifecycle policy where feasible, total limits | Identifies the allocation main effect; does not itself establish novelty |
| Train LQ jointly with equal per-node recurrent state | Supervision, actions, memory, initialization opportunities and tuning cost | Extra hidden state or generic co-adaptation explains LRE |
| Compile the identical LRE state/policy into conventional arrays | Every state transition and random draw | Must preserve behavior; discrepancies are implementation defects, not scientific evidence |
| Give all contenders exact parsing, hash-consing and ordinary memoization | Input transformations and their cost | Parsing/caching/representation engineering explains the gain |
| Remove learned retention; use LRU, reuse-count or learned-cache-only control | Value semantics, scoring budget and available observations | A cache-policy result, not a broader reaction ecology |
| Remove cancellation / force full strict demand | Same inputs and model where meaningful | Indicates benefit specifically from short-circuit computation |
| Remove persistent `z`; then add equal-capacity recurrent state to Q/LQ | Total parameter and mutable-state footprint | Separates object state from more ordinary memory capacity |
| Remove `REFINE` or permit it equally in controllers | Event/time caps, losses and depth opportunities | Identifies repeated local compute rather than ecology |
| Replace local features with global structural features for both | Preprocessing and information access | Establishes whether locality is useful, harmless or an avoidable handicap |
| Equalize batching, kernel implementation and device choices | Exact computations and outputs | Reveals implementation-only speed differences |
| Match fallback guessing behavior and class priors | Output information and deadline policy | Reveals accuracy gains from guessing rather than compositional execution |

The first three and ordinary parsing/caching controls are indispensable. Remaining tests are chosen from the observed failure/effect profile; expensive extensions require a specific causal question.

### 11.2 Output-prior and certificate guard

Boolean formulas can become overwhelmingly true or false as they grow, even when literal bits are balanced. This can make a constant answer appear to “generalize.” Add mandatory baseline `P0`: constant-majority and a small calibrated output-prior predictor trained only on development examples and permitted structural features. Its prediction time counts. Give table and neural baselines a validation-selected fallback prediction option rather than forcing one to abstain while allowing another to guess.

The primary B1 population is **root-answer balanced within size/shape/motif strata**, with a separately reported naturally generated population. For tree templates, construct balanced assignments without exponentially inefficient rejection: compute bottom-up counts of leaf assignments giving each node output; sample a target root bit and then sample child-value tuples with probabilities proportional to the product of their counts, recursively assigning literals. Use exact Python integers for counts and an integer-weight sampler. Declare and separately report templates constant under all leaf assignments; do not pretend both labels exist for them.

For shared DAGs, this tree counting procedure is not generally valid because child assignments can be correlated. Do not reuse it silently. B2 may use natural episodes with stratified reporting and prior controls, or a separately validated conditional sampler. Record its sampling bias and whole-episode grouping. This distinction blocks a subtle dataset correctness error.

Include counterfactual pairs: same tree/operator template, different input assignments, opposite correct roots, kept in the same cluster. A conditional-computation claim should persist on these pairs and against P0. For the reaction methods, audit each publication against the set of values consistent with the **actually observed operand slots**, using the evaluation oracle only after execution. Report the fraction of root answers supported by a valid chain of local publications. Unsupported lucky answers remain correct for primary task accuracy, but cannot be evidence of learned compositional execution. GNNs are evaluated on the common root metric; do not impose a post hoc chemistry-specific proof format on them.

A claimed LRE mechanism gain must survive equal fallback/readout opportunities and cannot be explained mainly by an increase in lucky unsupported publications. A root-AUC gain that fails this guard is a predictive/calibration tradeoff, not the proposed adaptive capability.

### 11.3 Boundary and adversarial scenarios

Test long combs that require deep credit assignment; broad frontiers that stress candidate scanning; high fan-out where a single value has many consumers; memory sufficient for everything (retention should cease to matter); a working set barely sufficient for a correct lazy traversal; zero reuse; repeated roots; changing epochs; all-XOR graphs; cheap decisive versus expensive nondecisive branches; and left/right port permutation.

Include opposing cost/prior shifts: a policy trained that one operator/branch is usually cheap or decisive should not be rewarded for blindly retaining that bias. Keep primitive semantics fixed when testing allocation shift. Test locally indistinguishable options with different hidden costs to bound what local observations could achieve.

Adversarial runtime cases include an object that repeatedly requests/cancels a child, duplicate proposals to publish one node, stale slot reuse, deletion of a shared dependency, a cache hit from the wrong epoch, a policy that evicts useful values immediately, a near-tie logit whose ordering differs by backend, and a batched commit that accidentally writes two different values into one parent port. Every case needs an explicit invariant/error category.

Inject equivalent methods under different names into the evaluation pipeline; the analysis must not manufacture a method effect. Inject a known delay or a known probability of wrong output and verify detection at the designed precision. Deliberately include a leaked answer field in a development fixture and ensure the observation/manifest checker rejects it.

### 11.4 Failure taxonomy and actions

| Observation | Category | Required action |
|---|---|---|
| Oracle or table violates exact known semantics | Broken implementation | Stop; fix data/runtime before any scientific interpretation |
| Identical compiled controllers diverge | State/measurement defect | Inspect RNG, ordering, mutation, precision and time boundaries |
| Short supervised local cases do not fit | Optimization or implementation | Inspect losses, labels, masking and gradients; compare an exact table; do not call it a scaling failure |
| Strict schedules have identical required reductions | Correct no-headroom control | Do not seek an allocation win there; analyze B1/B2 separately |
| Even a valid-trace clairvoyant schedule has no material headroom | Unsuitable proxy or strong conventional solution | Narrow the question or make a justified task amendment |
| Clairvoyant headroom exists but local observations cannot reveal it | Information limitation | Add shared observable features or acknowledge the locality limit; do not promise learning will recover absent information |
| MPS path fails but CPU works | Local implementation limitation | Use CPU faithfully, report it and recalibrate |
| Required campaign cannot resolve effects locally | Resource/precision limitation | Report detectable ranges and inconclusive result; do not weaken baselines |
| LRE works but a learned table/lazy evaluator dominates | Mechanism works, no advantage | Publish the negative comparison; consider a richer task only with evidence and a new claim |
| Gain disappears after fair tuning/caching | Alternative explanation established | Attribute to that component or withdraw the ecology claim |
| Positive mean hides large shape/domain regressions | Heterogeneous tradeoff | Report each stratum and limit the claim; no blanket superiority |
| Synthetic advantage fails realistic-format transfer | External validity failure | Retain only the synthetic component claim |
| Fresh independent confirmation fails | Unstable/exploratory result | Report inconsistency; diagnose selection/variance without relabeling the first result confirmed |

## 12. Interpretation and evidence-triggered continuations

### 12.1 Decision table

| Evidence at the end of the core | Defensible conclusion | Next action |
|---|---|---|
| Exact local rules generalize, no scheduler/lifecycle gain | Learned interpreter behavior; no evidence for the proposed ecology advantage | Stop the central claim; release the benchmark/negative result if useful |
| Adaptive scheduling helps, jointness/lifecycle do not | A scheduling result under these inputs and budgets | Reframe as a scheduler study and compare directly with LQ/priority/cache literature |
| Joint training improves matched event-budget results but loses wall time | A component/optimization effect, not practical efficiency | Profile whether removable shared implementation overhead explains the loss; no end-to-end superiority claim |
| Wall-time and accuracy gain survives strong simple, rewriting, discrete and stateful controls | A credible local empirical advantage under the frozen workload | Independent confirmation, then WP09 transfer; novelty still needs mechanism-specific comparison |
| Benefit depends only on cache retention | A learned intermediate-retention result | Compare directly against strong caching policies; remove broad chemistry claims |
| Benefits appear only after adding global information | Locality hypothesis weakened; augmented controller may still be useful | Report the information requirement and its cost |
| Results too noisy or campaign precision infeasible | Inconclusive at the resolved effect scale | Preserve artifacts; target the variance source or narrower mechanism, not arbitrary extra runs |

### 12.2 Continuation branches with explicit triggers

**Branch T — Realistic incremental Boolean evaluation.** Trigger only after a measured B1/B2 advantage survives prior/guessing, cache and stateful-controller controls. Use a local tabular filter workload with pure deterministic predicates, repeated filter roots and measured per-predicate costs. Fix actual expressions, data source/license, row distributions, caching semantics and optimizer baseline before testing. Compare against vectorized and lazy conventional evaluation, not an intentionally slow per-row Python loop. This tests practical value at the component level; it is not automatically a database-system result.

**Branch V — Richer primitive outputs.** Trigger when finite-table saturation removes semantic learning headroom **and** there is a specific scientific reason to study co-adaptation of expensive learned primitive approximators. One concrete next task is compositions of list transformations over alphabet `0..3`: `MAP_INC4`, `REVERSE`, `DROP_ZERO`, `CONCAT` and `TAKE2`, with exact list outputs. Training compositions remain short; list length and composition length are separate shifts. Compare with typed recursive symbolic programs, a table of local state transitions where applicable, recurrent GNN/DNAR adapters and a strong learned rewriter. Shared terminal/local supervision must be redesigned explicitly. This is not part of the frozen Boolean claim and needs a new interface/memory/power specification before execution.

**Branch M — Reusable macro reactions.** Trigger only if recurring multi-event motifs account for substantial measured cost after conventional batching and caching. Extract candidate parameterized motifs from **training traces only**, validate their semantics on the finite domain, charge their discovery/verification/storage and offer equivalent peephole compilation to all methods. Compare with simple frequency/description-length motif selection, not only no-macro LRE. Count primitive work separately from macro dispatches; calling ten operations one reaction is not a compute saving. This would test learned abstraction selection, an area already overlapping program-library learning [R15], not prove new abstraction discovery by itself.

**Branch F — Freer computational chemistry.** Trigger only when fixed ports demonstrably prevent an otherwise useful learned local reorganization. Add a bounded, typed register/list language with explicit fuel, allowed edge edits and no host-language `eval` or execution. Preserve a semantic checker and a conventional graph-program/controller baseline. This is a new, more ambitious architectural experiment; it cannot be validated by merely showing object growth. Require a new risk/cost study before replacing the selected core with it.

A failed small proxy does not refute benefits that require a missing mechanism or scale. Conversely, scaling is justified only when measured learning/resource trends, an identified representational limitation or a demonstrated headroom calculation predicts what larger scale could change. “Perhaps emergence happens later” is not an escalation criterion.

### 12.3 Training amortization and significance

Report extra training/tuning time alongside inference savings. If LRE adds training cost \(T_{\rm extra}\) and saves \(\Delta t>0\) per query at comparable accuracy and memory, the simple break-even query count is \(T_{\rm extra}/\Delta t\), ignoring deployment overhead. If \(\Delta t\le0\), there is no inference-time break-even. Do not choose an unsupported expected deployment volume to make amortization look favorable.

A tiny, reproducible effect can be scientifically real while practically unimportant. A large effect on an engineered synthetic distribution can be practically unverified. Keep these two distinctions explicit.

## 13. Audit record and remaining limitations

These audits were conducted as part of drafting this plan, not by independent reviewers. The implementation/training/evaluation packages are proposed work. Only the deterministic checks in §3/Appendix A were run.

### 13.1 Scientific and prior-art audit

| Defect/challenge found | Revision incorporated | Residual risk / acceptance criterion |
|---|---|---|
| Chemical naming could obscure known rewriting and computational-market mechanisms | Added FastNRS, NeuRewriter, Aurora, Agoric/Hayek and recent endogenous-selection chemistry; narrowed novelty to a tested interaction | Prior-art absence remains unproven; any matched predecessor requires reassessment |
| Ordinary exact interpreters may dominate finite tasks | Added learned table/certificate and symbolic controls as mandatory, plus an early ceiling gate | A negative result is likely and scientifically acceptable |
| Strict trees cannot show the advertised allocation mechanism | Proved the count invariant and separated strict from conditional/shared-demand workloads | Conditional workloads can still have no headroom after strong conventional control |
| A softmax bid is just a scheduler | Added exact compiled-controller equivalence and independently trained stateful controls | A positive result may be an inductive-bias/optimization effect, not methodological novelty |
| Typed execution is narrower than “replace neurons” | Stated the engineered grammar, parameterizing neural modules and fixed substrate assumptions | Core cannot establish radical neuron redesign or open-ended structure invention |
| Earlier/current work already learns growing structures or reusable programs | Added developmental and library-learning antecedents; macro/free-chemistry branches gated | Literature search is broad but not exhaustive |

### 13.2 Experimental-validity audit

| Defect/challenge found | Revision incorporated | Residual risk / acceptance criterion |
|---|---|---|
| Root-label skew can reward constant guessing | Added P0, balanced B1 strata, counterfactual pairs, natural-distribution reports and matched fallback behavior | B2 requires correlation-aware sampling/reporting; trace support must accompany capability claims |
| A semantic oracle could leak through “valid reaction” masks | Inference masks restricted to syntax/resource conditions; conditional publication correctness checked only afterward | Observation-path tests must verify this in code |
| Dynamic objects might merely implement caching | Shared memoization/hash-consing, LRU/learned-cache controls and lifecycle ablations | If caching explains the effect, report a cache result |
| Eviction could strand a pending consumer after producer retirement | Values pinned by pending waiters/root delivery; eviction allowed only after references clear | Shared-cache implementations must pass the explicit eviction/waiter test |
| Forced housekeeping could acquire a fictitious policy gradient or disappear after output | Excluded forced events from action losses; specified root admission, normal termination and cleanup-inclusive primary timing | Streaming response and complete service time remain distinct metrics |
| Event counts might masquerade as real compute | Made elapsed-time AUC primary; event cap only a matched-runtime diagnostic; record all scoring/bookkeeping | Timer noise and discrete deadline effects require pilot calibration |
| Large test sets might hide too few independent training fits | Crossed training/template units, block-preserving intervals and power/precision planning | Small-block bootstrap remains unreliable; replication may be locally expensive |
| Repeated protected inspection could manufacture significance | Fixed confirmatory sample and freeze gate; new data after amendments | Access isolation is procedural, not adversarially secure |
| FastNRS could be weakened by poor parsing or quadratic masked attention | Parser-assisted comparator and equivalent local-window implementation | Official reproduction status remains limited by code/text access |
| A composite average might conceal regressions | Per-size/shape/motif, latency, exactness and memory guardrails | No universal application utility exists to resolve all tradeoffs |

### 13.3 Implementation and feasibility audit

| Defect/challenge found | Revision incorporated | Residual risk / acceptance criterion |
|---|---|---|
| Nominal MPS support is not whole-model compatibility | Operation-level forward/backward gate, CPU path, explicit hybrid timing and dtype audit | Actual laptop checks are still unperformed |
| Popper's current dependencies conflict with a casual single-environment plan | Isolated environment; actual manifest requirements recorded; native-build gate | Build failure may require a documented revision or exact small-rule alternative |
| Shared child cancellation can corrupt unrelated roots | Reference counts, generation-safe subscriptions, charged cleanup and epoch checks | Stress tests must pass before learning |
| Object deletion does not necessarily return physical memory | Separate live state, reserved arena, framework/driver and process memory reporting | A physical-memory advantage requires measurements, not object counts |
| Streaming loops could terminate before future queries arrive | Explicit episode completion/waiting logic and externally controlled arrival visibility | Wall-time streaming and logical-budget simulation must not be conflated |
| In-place hidden-state slot reuse can break autograd or resumption | Versioned/out-of-place training state and complete checkpoint contract | CPU/MPS trajectory replay needs backend-specific checks |
| Frozen configs might omit budget-to-deadline mapping or fallback guesses | Shared mapping and fallback/readout rules included in freeze schema | Freeze must reject absent entries |
| Huge all-pairs candidates or dense attention could exceed memory | Typed local ports, fan-out accounting, local-window/gather parity path | High fan-out and long-token cases remain calibration dimensions |

### 13.4 Final challenge pass and honest limits

The plan was revised to address concrete defects found in these passes; it has not been certified correct by execution or independent review. Remaining blocking gates are actual Mac compatibility, faithful baseline adaptation, measured headroom, training stability and feasible confirmatory precision. The exact numerical role of joint training is unresolved. Synthetic success would not establish a broader deployment claim. A legal novelty/patent clearance conclusion is expressly outside this research.

The current source search reached coverage of the mechanisms most likely to change the design: chemistry/endogenous selection, learned rewriting, budgeted graph rewrite scheduling, computational economies, developmental dynamics, continuous/discrete algorithmic GNNs, symbolic induction, learned memory/priorities, and native hardware paths. Later targeted searches mainly reinforced scheduler/cache/controller equivalences and implementation constraints. This is qualitative coverage, not proof that no overlooked approach exists.

## 14. References and inspection/search record

References below are ordinary, stable links for a standalone handoff. The inspected scope is specified so that an abstract, repository manifest or inaccessible newer text is not mistaken for a reproduced experiment. Repository links are starting references; execution must record the actual pinned commit and license contents.

**[R1]** Flavio Petruzzellis, Alberto Testolin, Alessandro Sperduti. *Learning neuro-symbolic convergent term rewriting systems*. arXiv:2507.19372v1, 25 July 2025. [Abstract](https://arxiv.org/abs/2507.19372); [full v1 HTML](https://arxiv.org/html/2507.19372v1); [PDF](https://arxiv.org/pdf/2507.19372). Inspected full methods, experiments, limitations and appendix; visually checked PDF pages 28, 34–36 for errors/ablation and configuration material. Informs the closest learned-rewriting baseline. Newer indexed 2026 journal entry: [publisher PII](https://www.sciencedirect.com/science/article/pii/S0004370226001487), [institutional record](https://www.research.unipd.it/handle/11577/3615227). Full newer text was inaccessible in this session; no unverified DOI or equivalence to v1 is asserted.

**[R2]** Flavio Petruzzellis, Alberto Testolin, Alessandro Sperduti. *A Neural Rewriting System to Solve Algorithmic Problems*. 2024; arXiv:2402.17407, v2 dated 12 July 2024; ECAI DOI [10.3233/FAIA240854](https://doi.org/10.3233/FAIA240854). [Full preprint](https://arxiv.org/html/2402.17407v2). Inspected architecture and task discussion. Establishes the earlier modular selection/solution/combination lineage; not a reproduced local score.

**[R3]** Xinyun Chen, Yuandong Tian. *Learning to Perform Local Rewriting for Combinatorial Optimization*. NeurIPS 2019; arXiv:1810.00337 (earlier version titled *Automatic Local Rewriting for Combinatorial Optimization*). [Paper](https://arxiv.org/abs/1810.00337); [inspected HTML version](https://arxiv.org/html/1810.00337v3); [official repository](https://github.com/facebookresearch/neural-rewriter); [license](https://github.com/facebookresearch/neural-rewriter/blob/master/LICENSE). Inspected region/rule policy formulation, repository task instructions and license statement. Informs a mandatory learned-scheduling alternative; code's CC-BY-NC terms are a reuse constraint.

**[R4]** George-Octavian Bărbulescu, Taiyi Wang, Zak Singh, Eiko Yoneki. *Learned Graph Rewriting with Equality Saturation: A New Paradigm in Relational Query Rewrite and Beyond*. arXiv:2407.12794, 2024. [Full paper](https://arxiv.org/html/2407.12794v1). Inspected Aurora's rewrite scheduling and resource/graph-state discussion. Relevant overlap is learned resource-sensitive selection among supplied rewrite rules; published database outcomes are not directly comparable to this task.

**[R5]** Walter Fontana. *Algorithmic Chemistry: A Model for Functional Self-Organization*. Santa Fe Institute working paper, 1990. [Institutional record](https://www.santafe.edu/research/results/working-papers/algorithmic-chemistry-a-model-for-functional-self-). Inspected institutional description, not a new execution of the chemistry. Establishes the historical computational-chemistry antecedent.

**[R6]** Devansh Vimal, Cole Mathis, Westley Weimer, Stephanie Forrest. *Prebiotic Functional Programs: Endogenous Selection in an Artificial Chemistry*. arXiv:2509.03534, 2025. [Full v1 paper](https://arxiv.org/html/2509.03534v1); [author-linked implementation](https://github.com/AgentElement/functional-supercollider). Inspected methods, targeted-selection experiments and limitations; repository identified, not run. The text contains a May 2025 manuscript date and a late-August version header; the reference is identified by arXiv ID rather than pretending those are one submission date. Supports overlap with productive ecologies, not LRE's specific learning advantage.

**[R7]** Mark S. Miller, K. Eric Drexler. *Markets and Computation: Agoric Open Systems*. In *The Ecology of Computation*, 1988, pp. 133–176. [Author/organization-hosted full text](https://papers.agoric.com/papers/markets-and-computation-agoric-open-systems/full-text/). Inspected discussion of computational resources, agents and economic organization. Do not confuse the host webpage's date with the original publication date.

**[R8]** Eric B. Baum. *Toward a Model of Intelligence as an Economy of Agents*. *Machine Learning* 35, 155–185, 1999. DOI [10.1023/A:1007593124513](https://doi.org/10.1023/A:1007593124513). Inspected publisher abstract and bibliographic information, not a full local Hayek reproduction. Relevant to agent cooperation and credit; limits broad novelty claims.

**[R9]** Alexander Mordvintsev, Ettore Randazzo, Eyvind Niklasson, Michael Levin. *Growing Neural Cellular Automata*. Distill, 2020. [Primary article](https://distill.pub/2020/growing-ca/). Inspected local update, growth/life-mask and training discussion. Supports antecedents for learned local growth, not symbolic compositional performance.

**[R10]** Erwan Plantec, Joachim W. Pedersen, Milton L. Montero, Eleni Nisioti, Sebastian Risi. *Evolving Self-Assembling Neural Networks: From Spontaneous Activity to Experience-Dependent Learning*. arXiv:2406.09787v1, 14 June 2024. [Full paper](https://arxiv.org/html/2406.09787v1). Inspected architecture and developmental/plasticity framing. Similarity is changing learned computational structure; tasks differ from this PoC.

**[R11]** Gleb Rodionov, Liudmila Prokhorenkova. *Discrete Neural Algorithmic Reasoning*. ICML 2025, PMLR 267, pp. 51851–51865; arXiv:2402.11628. [Proceedings](https://proceedings.mlr.press/v267/rodionov25a.html); [official code](https://github.com/yandex-research/dnar); [requirements](https://github.com/yandex-research/dnar/blob/main/requirements.txt); [processor implementation](https://github.com/yandex-research/dnar/blob/main/processors.py). Inspected proceedings abstract, repository, requirements and processor code. Apache-2.0 repository license observed. Supports a discrete graph comparator; code inspection exposed PyG and precision compatibility issues. No M3 execution was performed.

**[R12]** Solveig Wittig, Antonis Vasileiou, Robert R. Nerem, Timo Stoll, Floris Geerts, Yusu Wang, Christopher Morris. *Which Algorithms Can Graph Neural Networks Learn?* arXiv:2602.13106; inspected v2, 23 August 2026. [Full paper](https://arxiv.org/html/2602.13106v2); [abstract](https://arxiv.org/abs/2602.13106). Inspected sufficient-condition/regularization discussion and experimental section. Limits blanket claims about GNN length failure; does not establish guarantees for LRE or this grammar. OpenReview access encountered a browser verification page; the arXiv primary text was accessible.

**[R13]** Rishabh Jain, Petar Veličković, Pietro Liò. *Neural Priority Queues for Graph Neural Networks*. arXiv:2307.09660, 2023. [Primary abstract](https://arxiv.org/abs/2307.09660). Abstract-level inspection only. Informs the need for a learned priority/memory control, not a reproduced architecture claim.

**[R14]** Andrew Cropper, Rolf Morel. *Learning programs by learning from failures*. *Machine Learning*, 2021. DOI [10.1007/s10994-020-05934-z](https://doi.org/10.1007/s10994-020-05934-z). [Author paper](https://andrewcropper.com/pubs/popper.pdf); [official Popper repository](https://github.com/logic-and-learning-lab/Popper); [current manifest](https://github.com/logic-and-learning-lab/Popper/blob/main/pyproject.toml). Repository README, license indicator and manifest inspected; full paper was located but not used for uninspected detailed claims. MIT license observed. Current manifest details are dated observations, not a promise of local installation success.

**[R15]** Kevin Ellis, Catherine Wong, Maxwell Nye, Mathias Sablé-Meyer, Luc Cary, Lucas Morales, Luke Hewitt, Armando Solar-Lezama, Joshua B. Tenenbaum. *DreamCoder: Growing generalizable, interpretable knowledge with wake-sleep Bayesian program learning*. arXiv:2006.08381, 2020. [Primary abstract](https://arxiv.org/abs/2006.08381). Abstract inspected. Informs the prior-art boundary for reusable libraries and the macro-extension branch; full-system reproduction is not required by the core.

**[R16]** Adam Machowczyk, Reiko Heckel. *Graph Rewriting for Graph Neural Networks*. arXiv:2305.18632, 2023. [Primary abstract](https://arxiv.org/abs/2305.18632). Abstract inspected. Used only for the representation-level overlap, not an empirical performance claim.

**[R17]** Apple. *MacBook Pro (16-inch, Nov 2023) — Technical Specifications*. [Official specifications](https://support.apple.com/en-ie/117736). Inspected hardware configuration/memory information. Informs why actual hardware must be measured; does not supply measured ML throughput.

**[R18]** PyTorch contributors. *torch.mps*, documentation page identifying itself as version 2.14, accessed 14 October 2026. [Official API documentation](https://docs.pytorch.org/docs/2.14/mps.html). Inspected synchronization/memory API availability. Installed-release checks remain necessary.

**[R19]** PyTorch contributors. *MPS backend*, version 2.14 documentation, accessed 14 October 2026. [Official backend documentation](https://docs.pytorch.org/docs/2.14/notes/mps.html). Inspected setup/check example and OS requirement. Also inspected Apple's [Accelerated PyTorch training on Mac](https://developer.apple.com/metal/pytorch/). These are support documents, not end-to-end compatibility tests.

**[R20]** PyTorch contributors. *MPS Environment Variables*, version 2.14 documentation, accessed 14 October 2026. [Official environment documentation](https://docs.pytorch.org/docs/2.14/mps_environment_variables.html). Inspected fallback and memory-watermark controls. Informs honest hybrid timing and safe resource gates.

**[R21]** Yanqi Zhou and co-inventors; original applicant Google LLC. *Learned graph optimizations for compilers*. WIPO publication WO2021248138A1, 9 December 2021. [Public patent text](https://patents.google.com/patent/WO2021248138A1/en). Inspected bibliographic record and technical description. Used only as adjacent prior-art discovery; no claim about enforceability, legal status, patentability or freedom to operate.

**[R22]** Julian Minder, Florian Grötschla, Joël Mathys, Roger Wattenhofer. *SALSA-CLRS: A Sparse and Scalable Benchmark for Algorithmic Reasoning*. arXiv:2309.12253v2, 20 November 2023. [Full paper](https://arxiv.org/html/2309.12253v2); [author code](https://github.com/jkminder/SALSA-CLRS). Inspected benchmark/scalability and architecture discussion; repository identified, not reproduced. Supports sparse recurrent graph-baseline design and warns against conflating dense execution overhead with reasoning capability.

### 14.1 Search-coverage log

| Search family | Representative searches / tracing | Material result |
|---|---|---|
| Exact proposal and equivalent language | learned artificial chemistry; local computational reactions; resource competition; functional self-organization | Chemical terminology is not a novelty boundary |
| Closest learned rewriting | FastNRS; NRS; paper title/authors/code; neural local rewriting; reference/backward tracing | FastNRS is central; older NRS/NeuRewriter overlap; official FastNRS code not located |
| Current/forward work | 2025–2026 artificial chemistry, size-generalizing graph algorithms, recent rewriting/query optimization | Found recent endogenous-selection chemistry, current graph-learnability work and indexed newer FastNRS journal record |
| Allocation equivalents | learned graph rewrite scheduling; equality saturation memory budget; agoric computation; Hayek; learning classifier systems/bucket brigade; priority queues | Resource scheduling and credit are established; stateful learned scheduling is indispensable |
| Reusable structure / growth | neural cellular automata; developmental programs; program libraries; DreamCoder; e-graph abstraction learning | Growth and reusable abstractions already have close antecedents |
| Implementations and constraints | DNAR requirements/processor; Popper manifest; NeuRewriter license; PyTorch MPS operators/memory; Apple specifications | Native-build, grouped-operation, precision, fallback and licensing gates changed the plan |
| Public patent discovery | artificial chemistry computational resources; learned graph rewriting; compiler scheduling | Relevant broad graph-optimization patent description found; no legal clearance inference |
| Refinement challenge | demand-driven learned rewriting; local state/lifetime/caching; joint scheduler training; graph-controller equivalence | No convincing reason found to omit ordinary lazy/cache/stateful controls |

Research used original papers, author/institutional sources, official documentation and public implementation files for consequential technical claims. Secondary/indexed results were used for discovery and access status, not as substitutes for uninspected methods. This was not a complete citation-graph crawl, patent landscape, systematic-review protocol or proof of absence. Some candidate later papers and implementations were only discovered; they are not claimed as reproduced or fully audited. No private repositories, paid APIs or remote accelerator experiments were used.


## Appendix A. Reproducible design checks executed during research

Save the following as `tools/design_checks.py` (creating `tools/` first) and run `python tools/design_checks.py`. It uses only the Python standard library. It writes `results.json` beside the script. The output on the research environment is summarized in §3.3; a different environment field on the Mac is expected. This is a small mathematical/runtime-design check, not training or an M3 benchmark.

```python
"""Small design checks; not an LRE implementation or hardware benchmark."""
from functools import lru_cache
import itertools, json, platform, sys
from pathlib import Path

@lru_cache(None)
def shapes(n):
    if n == 0:
        return (None,)
    return tuple((a,b) for k in range(n) for a in shapes(k) for b in shapes(n-1-k))

def build(tree):
    children, vals, ops = [], [], []
    def visit(t):
        if t is None:
            j=len(children); children.append(()); vals.append((3*j+1)%7); ops.append('lit'); return j
        left,right = visit(t[0]),visit(t[1])
        j=len(children); children.append((left,right)); vals.append(None); ops.append(('add','sub','mul')[j%3]); return j
    root=visit(tree)
    return children,vals,ops,root

def operation(op,a,b):
    return {'add':a+b,'sub':a-b,'mul':a*b}[op] % 7

strict=[]
for n in range(1,7):
    schedules=0
    for tree in shapes(n):
        children,initial,ops,root=build(tree)
        internals=tuple(i for i,c in enumerate(children) if c)
        oracle=list(initial)
        for i in internals:
            a,b=children[i]; oracle[i]=operation(ops[i],oracle[a],oracle[b])
        def walk(values,done):
            global schedules
            if len(done)==n:
                assert values[root]==oracle[root]
                schedules+=1
                return
            ready=[i for i in internals if i not in done and all(values[c] is not None for c in children[i])]
            assert ready
            for i in ready:
                a,b=children[i]; v=list(values); v[i]=operation(ops[i],v[a],v[b]); walk(v,done|{i})
        walk(initial,set())
    strict.append(dict(internal_nodes=n,tree_shapes=len(shapes(n)),complete_schedules=schedules,semantic_disagreements=0,reductions_per_schedule=n))

ops={'not':(1,lambda a:1-a),'and':(2,lambda a,b:a & b),'or':(2,lambda a,b:a | b),'xor':(2,lambda a,b:a ^ b),'if':(3,lambda c,a,b:a if c else b)}
cert={}
for name,(arity,fn) in ops.items():
    total, partial = 0,0
    for pattern in itertools.product((None,0,1),repeat=arity):
        completions=itertools.product(*[(0,1) if x is None else (x,) for x in pattern])
        results={fn(*x) for x in completions}
        if len(results)==1:
            total+=1; partial+=int(None in pattern)
    cert[name]={'patterns':3**arity,'certifiable':total,'certifiable_partial':partial}

out={'environment':{'python':sys.version.split()[0],'system':platform.system(),'machine':platform.machine()},'strict_tree_checks':strict,'boolean_partial_certificates':cert,'memory_arithmetic':{'fp32_adam_1M_params_MiB':16_000_000/2**20,'one_activation_B16_T8_N256_D64_MiB':4*16*8*256*64/2**20},'notes':['Enumeration uses fixed deterministic leaf values and operator assignment, not exhaustive semantic assignments.','No neural models trained. No M3 Max timings or compatibility tests performed.']}
Path(__file__).with_name('results.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
print(json.dumps(out,indent=2))
```


## Appendix B. Freeze checklist and complete stopping record

The frozen manifest must contain concrete values, not phrases such as “tune appropriately,” for the following fields. They are selected from the already completed pilots, not guessed in this research document:

```yaml
schema_version: 1
phase: confirmatory
source_revision: REQUIRED_ACTUAL_COMMIT
dependency_locks: REQUIRED_HASHED_PATHS
hardware_record: REQUIRED_HASHED_PATH
claim_ids: [H1, H2, H3]                 # revise before freeze to match intended claims
primary_population: REQUIRED_B1_STRATA
training_max_operator_count: REQUIRED
validation_operator_counts: REQUIRED
protected_operator_counts: REQUIRED
shape_weights: REQUIRED
motif_split_hash: REQUIRED
root_balance_sampler_version: REQUIRED
natural_distribution_secondary: REQUIRED
training_information_track: REQUIRED
training_blocks_and_seeds: REQUIRED
protected_template_clusters_and_seeds: REQUIRED
methods_and_exact_configs: REQUIRED
baseline_provenance_and_substitutions: REQUIRED
checkpoint_selection_rules: REQUIRED
deadlines_ns_by_size: REQUIRED
deadline_to_event_cap_mapping: REQUIRED
mutable_byte_limits: REQUIRED
physical_vs_live_state_accounting: REQUIRED
fallback_prediction_rules: REQUIRED
streaming_arrival_visibility: REQUIRED
precision_and_backend_by_method: REQUIRED
policy_randomness_and_tie_rules: REQUIRED
warmup_and_measurement_blocks: REQUIRED
invalid_block_and_interruption_rules: REQUIRED
primary_contrasts_and_multiplicity: REQUIRED
crossed_variance_and_sample_size_justification: REQUIRED
interval_and_bootstrap_procedure: REQUIRED
planning_effect_and_precision_target: REQUIRED
known_regression_guardrails: REQUIRED
protected_manifest_hash: REQUIRED
oracle_isolation_test_hash: REQUIRED
resume_test_hash: REQUIRED
estimated_campaign_time_and_peak_memory: REQUIRED
```

This is a **schema template**, not an executable experiment configuration. The freeze command must reject unresolved `REQUIRED` fields. A coding agent is authorized to populate them from measured development evidence; it is not authorized to choose them after observing protected outcomes.

For power/precision selection, enumerate candidate training-block/template allocations, simulate the declared estimator using the pilot's crossed variance and bounded-score structure, and choose the least-cost allocation meeting the recorded precision/power goal. Set the goal in units of the primary score, relate it to the measured timing-resolution floor and a plausible operation/time saving, and publish a sensitivity table for smaller effects. Use a fixed confirmatory allocation. If no allocation is justified by the local cost/precision tradeoff, the output is an explicit inconclusive/negative-scope decision, not an arbitrary training cap or a conveniently underpowered win.

Each stopping record must state:

```text
1. Which claim was tested, and on which population/limits?
2. Were correctness, provenance, fairness and local compatibility gates satisfied?
3. What was observed, with absolute effects, uncertainty and costs?
4. Is the outcome a defect, missing headroom, information limit, resource limit,
   negative evidence, explained component benefit, confirmed improvement, or inconclusive?
5. What narrower claim is supported, if any?
6. What specific evidence would justify another experiment?
7. Were protected data accessed, and does continuation require new confirmation?
```

**Final handoff rule:** preserve the original hypothesis, the strong unfavorable baselines and all failed runs. The project succeeds as research when it makes the mechanism's contribution—or lack of contribution—clear enough to reproduce and act on, not when it produces an attractive ecological visualization.
