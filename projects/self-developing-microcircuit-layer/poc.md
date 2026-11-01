# SDML: research assessment and executable proof-of-concept plan

**Research date:** 14 October 2026  
**Target:** one MacBook Pro with an M3 Max; actual memory, storage and macOS version must be discovered  
**Status:** researched experimental design, not a demonstrated architecture or an M3 benchmark  
**Working implementation name:** `sdml-local`  
**Primary deliverable of execution:** an attributable, resource-accounted comparison of learned local microcircuit organization against strong reusable-library, modular and nonmodular alternatives.

**Navigation:** [Assessment](#1-executive-assessment-and-decision) · [Problem](#2-problem-hypotheses-and-analytical-constraints) · [Prior art](#3-prior-art-and-precise-novelty-assessment) · [Mechanism](#4-implementable-sdml-specification) · [Evaluation](#5-data-baselines-and-confirmatory-evaluation) · [Hardware](#6-m3-max-feasibility-and-hardware-adaptive-execution) · [Work packages](#7-risk-ordered-work-packages-and-decision-gates) · [Robustness](#8-attribution-robustness-and-adversarial-failure-analysis) · [Repository](#9-repository-interfaces-configuration-and-reproducibility-contract) · [Decisions](#10-outcome-interpretation-and-evidence-triggered-alternatives) · [Audits](#11-audit-record-revisions-and-residual-risks) · [Executed checks](#12-targeted-checks-actually-executed-for-this-research) · [References](#13-references-and-inspected-evidence-register) · [Search scope](#14-search-coverage-limitations-and-handoff-completion)

## 1. Executive assessment and decision

**Proceed with a bounded experiment, not with a claim that self-developing networks are new.** The broad proposal combines established ideas: growing neural structures, local developmental rules, structural plasticity, modular transfer, program-library learning and learned skill management. Particularly close precedents include evolved reusable Cartesian-GP modules, HOUDINI, modular meta-learning, DreamCoder/Stitch, neural developmental programs and lifelong local module selection [R03]–[R14]. Current work also removes two tempting novelty claims: prospective abstraction selection is explicit in *Prospective Compression in Human Abstraction Learning*, and counterfactual evaluation of learned skill-bank edits is explicit in SkillMaster [R17], [R18].

The defensible question is narrower and still worth answering:

> Can a very small, shared policy, using bounded local circuit observations and trained on outcome-based structural interventions, discover and retain useful differentiable microcircuit templates—while achieving a better transfer-versus-total-cost tradeoff than strong heuristic, symbolic-library and conventional modular learners?

The proposed contribution is **not** “counterfactual credit,” “networks that grow,” “future utility,” or “invented reusable functions.” It is a test of **local-information sufficiency and the economics of amortizing structural decisions in a tiny numerical circuit substrate**. A positive finding would initially be an empirical component result. A new learning principle or broad architectural breakthrough would require additional evidence.

### Selected direction

Implement small scalar DAGs with three primitive operation types, a learned local action scorer, temporary task-specific graphs, exact reusable structural templates, and bounded library replacement/retirement. Numerical parameters learn by ordinary gradients. The action scorer learns from paired intervention outcomes during the training prefix. At protected evaluation, its weights are frozen; it receives neither target programs nor future query outcomes. Macros initially reuse **structure with freshly initialized call parameters**. Warm-started and shared-parameter macros are separate variants, not hidden advantages.

Start with an **isolated consolidation assay** using an identical task solver for all selectors. This is a necessary diagnostic, not the complete SDML demonstration. Only proceed to the complete learned assembly-and-consolidation loop after demonstrating measurable reuse headroom and a learnable utility signal. Use synthetic hidden-program tasks for causal identification; follow with an independent teacher family and, conditionally, a small real sensor-drift case study.

### What must be true before reporting success

Three results must not be conflated:

1. **Mechanism:** allowing selected macro templates changes subsequent learning beneficially relative to the same solver without them.
2. **Learned organization:** the local learned selector improves on well-tuned frequency/description-length selection under matched information and resource limits.
3. **Practical advantage:** the best complete SDML configuration improves the relevant error/time/memory tradeoff against the strongest development-selected, locally executable competitor, with all structural-learning costs disclosed.

A macro that merely names an existing computation does not reduce its arithmetic. A gain solely from parameter initialization is transfer learning, not evidence for structural abstraction. A gain that vanishes after charging for counterfactual training is not an end-to-end efficiency improvement. These are decision rules, not objections to doing the experiment.

### Boundaries and assumptions

The primary beneficiaries are researchers building small adaptive numerical models or reusable computational components, not users of an already identified product. Repeated related tasks and costly rediscovery of their structure are **assumed bottlenecks**. No supplied evidence establishes that this is a commercial or deployment bottleneck. The first experiments measure whether that headroom exists at all.

The core requires no pretrained foundation model, cloud accelerator, paid API, robot, spiking hardware or large dataset. The plan does not require the maximum M3 Max memory configuration. Self-repair, recurrent execution, autonomous code generation, approximate macro distillation and open-ended vocabulary invention are deferred, explicitly, because they would obscure the first causal test.

## 2. Problem, hypotheses and analytical constraints

### 2.1 Objects and information

An independent experimental **world** contains a hidden distribution over numerical tasks. A task is a function observed through a labeled support set

\[
S_t=\{(x_j,y_j)\}_{j=1}^{n_s},\qquad x_j\in[-1,1]^d,\quad y_j\in[-1,1].
\]

The learner produces a fitted function \(\hat f_t\). The evaluator holds an independent query set \(Q_t\). A learner may know the public primitive vocabulary and task-distribution specification, but not the sampled teacher graph, motif identities, split labels or query targets. Knowing the vocabulary is a common inductive bias, not privileged access to the answer.

A task graph \(G_t\) is an acyclic, typed numerical circuit. A library \(L\) holds reusable graph **templates**, not examples or cached answers. A template has one output, at most two external scalar inputs and a small number of primitive nodes. A call instantiates it on new signals. A **unit** is one primitive operation; a **macro** is a composition of units, not a newly discovered primitive operator. A **temporary circuit** is the fitted graph for a task. Structural development consists of changes to these graphs and to the reusable library.

Denote numerical parameters by \(\theta\), local-policy parameters by \(\phi\), active expanded primitive-node count by \(N_e\), all persistent model/registry/replay bytes by \(M\), and elapsed charged wall time by \(T\). `expanded` means that a macro call is charged for every primitive it actually executes. Library references do not make this work disappear.

### 2.2 Objective and estimands

For bounded regression, use normalized mean squared error

\[
R_t=\frac{1}{4|Q_t|}\sum_{(x,y)\in Q_t}(\operatorname{clip}(\hat f_t(x),-1,1)-y)^2.
\]

This normalization is a fixed unit conversion, not division by query variance. All methods receive the same final clipping convention; unclipped error is also reported. The primary objective is lower mean \(R_t\) at a preselected adaptation deadline \(B_*\), support-label budget \(n_s\), persistent-memory envelope and measured training-resource disclosure. Section 5 defines aggregation and inference.

Use **hard resource constraints** for the primary experiment rather than inventing business-specific exchange rates between accuracy and memory. The original utility can still be studied as a sensitivity analysis:

\[
U(a)=E[R_{\rm noop}-R_a]-\lambda_T\Delta T-\lambda_M\Delta M,
\]

where \(\lambda_T\) has units of normalized error per second and \(\lambda_M\) error per byte. Set both to zero in the hard-budget primary condition; any nonzero values and their meaning must be chosen on development data and frozen. Never report an arbitrary weighted score as unqualified superiority.

For a library intervention \(a\), train on the finite-horizon paired target

\[
\hat v_H(a)=\frac1H\sum_{h=1}^{H}\left[R_{t+h}^{\operatorname{noop}}(B_{\rm probe})-R_{t+h}^{a}(B_{\rm probe})\right].
\]

Each arm has the same allowed data and adaptation budget, and starts from the same nonintervened state. The intervention can include explicit eviction. The target is conditional on the chosen solver, training distribution, horizon and budget. It is not the intrinsic, context-free value of a circuit.

### 2.3 Hypotheses and discriminating observations

| ID | Hypothesis and supporting precedent | Alternative explanation | Evidence against, or limiting the claim |
|---|---|---|---|
| H1: reuse headroom | Small reusable structures reduce rediscovery cost on related held-out tasks. Library learning and modular meta-learning establish precedents, not a guarantee for this substrate [R05]–[R09]. | Extra parameters, copied coefficients, favorable teacher grammar or simply a larger mutation step. | A correct oracle-template control offers no advantage over primitive-only search at nonsaturated budgets; or structure-only reuse loses its gain when expanded work is matched. |
| H2: local sufficiency | Bounded local activity, gradient and history features contain useful information about intervention value. NDP/LNDP and LMC motivate testing local decisions [R11]–[R13]. | The controller encodes task identity, receives global graph information, or learns only frequency. | A global controller or fresh direct-probe selector works while the local policy fails on held-out worlds/contexts; shuffled targets work equally well. |
| H3: learned consolidation | A learned outcome-based selector chooses more useful libraries than frequency or description length. | More probe data, more search, retention of warm-start parameters, or tuning advantage. | Gains disappear under equal development expenditure, structure-only parameters or strong Stitch/MDL selection. |
| H4: total economics | Amortized local scoring repays the cost of training it over a feasible number of future tasks. | Counting only cheap deployment while omitting paired-rollout generation and failed proposals. | The breakeven number of tasks is unsupported by the tested reuse horizon, or no matched-quality cost saving exists. |
| H5: bounded adaptation | Replacement/retirement helps after a change in motif utility without unacceptable loss on old tasks. | Unlimited library growth, a supplied shift marker, or unequal replay. | A bounded fixed library plus replay wins; removal causes unrecovered forgetting; old motifs never became obsolete. |

**Established:** gradients optimize parameters on a fixed differentiable graph; exact macro expansion preserves its mathematical function; local neural development and reusable library induction already exist. **Assumed:** meaningful recurring substructure, sufficiently informative support data, a stable relationship between local features and future benefit, and useful search headroom. **Proposed tests:** H1–H5. **Speculative:** self-repair, biologically local credit assignment, invention outside the primitive closure, unbounded cumulative development and million-unit scaling.

### 2.4 Strongest case for and against

The strongest case is an intermediate regime: primitive-only search is difficult but not hopeless; useful motifs recur with different coefficients and input bindings; no fixed handcrafted decomposition is consistently ideal; small macros reduce the number of structural decisions required; and cheap local scoring generalizes enough to avoid repeating expensive selection experiments. Reuse may improve optimization or sample efficiency even without faster inference.

The strongest countercase is that this is learned program-library search with an expensive neural wrapper. A fixed modular learner may have the right factorization already. Stitch or a simple frequency rule may select equally good motifs. Tiny dynamic operations may lose badly to dense kernels. A local observation can be insufficient for future utility; a macro can increase branching; and short-horizon marginal values miss interactions between two jointly useful structures. The plan deliberately includes these cases.

### 2.5 Analytical limits and headroom calculations

**No representational magic.** With exact bounded-depth macro expansion, the function class is still contained in the primitive-DAG class at the same expanded-size bound. Macros alter a parameterization, search prior and learning trajectory. They do not prove a larger representational class. Expanding the allowed size or vocabulary is a different intervention.

**No universal local predictor.** Consider two future worlds with identical observable histories and opposite preferences for storing macro A versus B, with room for only one. Under an equally likely, unobserved future choice, any history-only selector has expected success 1/2. A local predictor cannot overcome missing information; a global predictor without additional information cannot either. Use this as a negative control, not as a claim that local learning is generally impossible.

**Finite examples do not identify an arbitrary program.** For support inputs \(x_i\), adding \(c\prod_i(x-x_i)\) leaves all support predictions unchanged but can change unseen outputs. This is a general identifiability witness, not a statement that this exact polynomial belongs to every restricted SDML configuration. Support size, teacher restrictions and query distribution therefore matter. Exact teacher-graph recovery is not required for predictive success and must not be inferred from it.

**Arithmetic versus structural description.** A reusable definition can reduce stored structural tokens. Executing it at two different inputs still requires two evaluations. Identical-input common-subexpression elimination and kernel fusion are separate optimizations, shared with baselines. A literal macro-to-inline equality test is mandatory.

**Stability.** For the proposed bounded affine-tanh primitive, each input partial derivative has magnitude at most 2; for product and mean on the bounded domain, at most 1 and 1/2 respectively. A path of length \(D\) can still amplify derivatives by a factor up to \(2^D\), and multiple paths add contributions. Bounded outputs do not guarantee well-conditioned training. Log gradient norms, saturation, active depth and optimizer failures; compare clipping choices fairly.

**Search cost.** For \(N\) two-input nodes with \(d\) terminals and \(|\mathcal O|\) operation types, a crude upper envelope is \(\prod_{i=1}^N |\mathcal O|(d+i-1)^2\), before parameters. Restricting local proposals avoids enumerating this space, not its existence. A library increases choices but may shorten effective construction paths. Do not turn grammar-count ratios into claimed runtime speedups.

**Breakeven.** If the extra development cost is \(C_{\rm extra}\) seconds and measured saving at matched error is \(s>0\) seconds per future task, then \(K_{\rm break}=C_{\rm extra}/s\). Report uncertainty, the measured range of task reuse and library degradation. When \(s\le0\), there is no observed time breakeven. Error improvements with higher cost are a separate tradeoff.

**Amdahl-style warning.** Even eliminating a fraction \(p\) of original runtime cannot exceed speedup \(1/(1-p)\) before new overhead. Measure how much time is actually spent on reducible structural search before optimizing its proxy.

## 3. Prior art and precise novelty assessment

### 3.1 Closest alternatives

Evidence below is from inspected primary papers, author sites, official repositories or documentation except where an access limitation is explicitly stated. Published numbers from different tasks or hardware are not controlled SDML comparisons.

| Work | Existing mechanism and overlap | Conditions / limits of evidence | Implementation and local relevance |
|---|---|---|---|
| Cascade-Correlation; PathNet [R01], [R02] | Growth of learned features; selection/reuse of paths through a larger network. | Growth and pathway reuse alone are old; not the proposed learned tiny-template policy. | Historical controls, not the strongest complete comparisons. |
| Automatically defined functions; evolved CGP modules [R03], [R04] | Evolution invents reusable subroutines/modules from primitive computations. Very close to the broad circuit-abstraction idea. | GP/module mechanisms are not necessarily local learned utility predictors. The 2008 ECGP full article was not accessible here. | Include a clearly labeled CGP-style mutation baseline; do not call an unverified port a faithful ECGP reproduction. |
| DreamCoder [R05] | Alternates program search and abstraction/library learning with learned search guidance. | Primarily program synthesis; does not establish practical scalar-DAG local policy economics. | Official MIT repository inspected; Python/OCaml/Rust dependencies require a native build gate. |
| Stitch [R06] | Efficient, corpus-guided discovery of reusable abstractions under a compression objective. | Program-corpus compression is not predictive utility on unseen numerical tasks. | Official MIT implementation and Python binding API inspected. Required direct library comparator when buildable. |
| babble [R07] | Library learning modulo supplied equational theories using e-graphs and anti-unification. | Stronger treatment of semantic equivalence, with theory and search costs. | Conditional comparator on equivalence-heavy workloads; not an excuse to call syntactic matching new. |
| HOUDINI [R08] | Typed differentiable programs with neural components, synthesis and lifelong reuse. | A predefined typed language and a different search/learning architecture. | Closest conceptual neural-symbolic predecessor; a restricted reimplementation must be labeled as such. |
| Modular meta-learning / BounceGrad / MOMA [R09] | Learns reusable numerical modules and searches their compositions across tasks. | Module vocabulary/structure family prescribed; no autonomous template library as specified here. | Required compact numerical baseline; equations inspected, implement algorithm-level replication in the common harness. |
| RigL [R10] | Changes sparse connectivity through magnitude-based removal and gradient-based growth. | Rewiring does not itself invent reusable graph templates. Sparse FLOP accounting is not wall-clock speed. | Official TensorFlow code inspected; a tested PyTorch port is `RigL-style`, with update costs counted. |
| NDP; LNDP [R11], [R12] | Learned local development, replication/weights, and ongoing activity/reward-dependent structural plasticity. | LNDP uses a more elaborate graph/GRU mechanism and control-task outer optimization. | Key novelty predecessors. A new platform is not a new developmental principle. |
| Local Module Composition [R13] | Local module relevance decisions, growth and continual composition. | Uses richer modules; local selection can struggle on longer sequences. | Required local-selection conceptual control; code available, license not established by the inspected README. |
| Discovering modular solutions [R14] | Identifies conditions supporting modular recovery and compositional generalization in a teacher/student setting. | Conditional identifiability, not a theorem for arbitrary SDML circuits. | MIT JAX code inspected; small CPU teacher case is an independent robustness family. |
| Growing NCA; structural-plasticity theory [R15], [R16] | Local shared rules can maintain/regenerate patterns; structural connectivity can support learning/memory. | Different objectives and model assumptions; no evidence for arbitrary task-abstraction transfer. | Context and limits, not required end-to-end baselines. |
| Prospective Compression [R17] | Explicit expected-future compression criterion for promoting abstractions. | Human Pattern Builder experiments and computational models; not a complete general generative curriculum learner. | Defeats “future usefulness rather than past frequency” as a standalone novelty claim. |
| SkillMaster; ALMA [R18], [R19] | Learned skill edits with counterfactual downstream utility; meta-learned executable memory designs. | LLM-agent settings, different information and substantial model/API infrastructure. | Strong novelty precedents, not fair locally executable numerical benchmarks. |
| SMGrNN, revised July 2026 [R20] | Local update-statistic-driven growth/pruning with gradient-trained numerical weights. | Policy distillation and within-task capacity adaptation; not demonstrated open-ended cross-task library development. | Include its simple instability/frequency ideas as heuristic controls, not as a claimed exact replication. |
| Numerical superoptimization; REFACTOR-VLA [R21], [R22] | Optimizing library primitives for numerical execution; typed motor-program library learning. | Different mechanisms/workloads; a primitive implementation improvement is distinct from naming a circuit. | Additional current overlap; no use of their published task scores as SDML evidence. |
| LEGO-Prover library-learning analysis [R23] | Tests whether a claimed learned library is actually directly reused. | A negative result in one theorem-proving system is not a universal impossibility. | Motivates actual invocation, removal and counterfactual reuse tests. |
| Counterfactual contribution analysis, COCOA [R24] | Outcome-conditioned counterfactual credit for delayed rewards. | Reinforcement-learning credit, not this numerical library mechanism. | Establishes another reason not to claim counterfactual utility itself as new. |
| Function Encoders [R35] | Learned neural basis functions with fast least-squares task adaptation. | Fixed learned basis, no circuit creation; numerical transfer is nevertheless directly relevant. | Required locally executable algorithm-level comparison; source and licensing gates below. |
| Compositional open-ended framework [R25] | Formalizes primitives, composition, motifs and a proposed next-primitive objective. | A framework/agenda, not a measured M3 numerical microcircuit advantage. | The long-term vision also has contemporary conceptual overlap. |
| US10019470B2 [R26] | Describes construction, use and reuse of ANN components/structures. | Patent descriptions are not empirical validation or a legal novelty analysis. | Broad patent search found relevant terminology; no patentability/FTO conclusion is offered. |

### 3.2 What to reuse rather than rebuild

Reuse PyTorch automatic differentiation and optimizers, official Stitch compression, ordinary CPU numerical baselines, and the published modular teacher generator when applicable. Reuse established paired-intervention reasoning and supervised regression for structural utility; do not rebrand them. Implement only the bounded DAG runtime, local observations/actions, a clear solver adapter, library transactions and evaluation instrumentation that this question requires.

Use the original RigL source to validate mask-update semantics. The inspected implementation computes gradients with respect to masked-weight tensors, not only already-masked parameter gradients; uses pre-removal growth scores; and handles optimizer state for new connections [R10]. This is a consequential implementation distinction.

### 3.3 Selected refinement versus alternatives

**Original all-mechanism SDML:** ambitious, but too many coupled causal explanations for a first experiment. Keep it as a longer-term program.

**Selected local-policy / exact-template SDML:** preserves tiny units, learned structural decisions, formation, consolidation, reuse and bounded retirement. It makes locality and accounting observable. It removes arbitrary executable code, recurrence, approximate distillation and automatic semantic merging from the first claim.

**Simpler default alternative:** a strong shared solver plus Stitch/MDL selection or a fixed modular meta-learner. Prefer this if it is as good or better. A negative result can establish that the speculative local mechanism is unnecessary for this setting.

**Global learned controller:** an evidence-triggered alternative if useful utility is predictable only with wider context. This preserves learned structural organization but explicitly relinquishes the strong local-information claim. It is not to be silently labeled the original local-neuron mechanism.

**Kernel/compiler optimization:** justified only if profiling identifies execution overhead and semantics-preserving optimization improves all methods. This is an engineering contribution, not proof of newly learned abstraction.

### 3.4 Novelty questions answered

What already exists? Every broad ingredient and several close combinations. What may differ? The precise bounded local observation/action contract, structure-only numerical reuse and measured amortization in one small circuit substrate. Why might that matter? A cheap policy could make useful structural adaptation viable without a large global controller or repeated expensive library evaluation. What isolates it? Identical candidate pools/solver/data, outcome-trained versus MDL/frequency/global selection, and causal disabling of learned macros. What would invalidate novelty? Earlier work with the same effective observation, action, learning target and evaluation contribution; or a result fully explained by existing library induction or parameter reuse.

**No absence-of-prior-art claim survives this search.** Even a successful experiment should first be described as a researched synthesis and new empirical evidence. A methodological novelty claim requires a fresh, precise comparison at the level of the actual rule, not the working name SDML.

## 4. Implementable SDML specification

All interfaces, module names and `sdml` commands below are **proposed project interfaces to implement**, not existing package APIs. Existing third-party commands are identified separately in Section 6.

### 4.1 Primitive numerical contract

Use `float32` for primary training on CPU or MPS. Use CPU `float64` as the correctness reference. Inputs and primary teacher outputs are bounded. Three initial two-input primitives suffice:

\[
\begin{aligned}
A_{u,v,b}(a,c)&=\tanh\left(2\tanh(u)a+2\tanh(v)c+\tanh(b)\right),\\
P(a,c)&=ac,\\
M(a,c)&=(a+c)/2.
\end{aligned}
\]

`A` has three trainable raw scalars; `P` and `M` have no parameters. Zero, the input coordinates, and optionally a fixed one terminal are legal inputs; the exact terminal set is frozen per experiment. Use `{x_0,...,x_(d-1),0,1}` in version 1. All primitives map the bounded domain to itself. Comparisons are smooth numerical functions expressible by `A`; there is no hard comparison oracle or Boolean target-program supervision.

For new `A` units, sample raw parameters independently from `Normal(0,0.4²)`. This is an engineering initialization to tune on development data, shared with controls. The teacher uses a separate documented coefficient prior. Do not transfer a teacher's sampled coefficients. There is no claim that defining product in code constitutes learning multiplication.

Version 1 has no recurrence, division, logarithm, arbitrary code, data-dependent loops, hidden vector-valued networks inside units, batch normalization or stateful prediction. Per-node activity history is for structural decisions, not an unbounded hidden computational state. `predict` must not update these histories or any parameter.

### 4.2 Graph and library structures

```python
@dataclass(frozen=True)
class NodeSpec:
    uid: str                   # stable within a lineage; never a policy feature
    op: str                    # A, P, M
    parents: tuple[str, str]    # earlier node UIDs or typed terminals
    parameter_keys: tuple[str, ...]  # exactly three for A, otherwise empty
    origin_template: str | None      # audit metadata only

@dataclass(frozen=True)
class GraphSpec:
    schema_version: int
    nodes: tuple[NodeSpec, ...]       # canonical topological order
    output: str                      # a node or terminal

@dataclass(frozen=True)
class TemplateSpec:
    content_hash: str
    arity: int                      # 1 or 2 in the initial library
    primitive_body: GraphSpec       # flat, no recursive template references
    parameter_slots: tuple[str, ...]
    # Numerical initial values are absent in structure-only mode.

@dataclass
class TrainState:
    graph: GraphSpec
    parameters: dict[str, Tensor]
    adam_state: dict[str, dict]      # moments and step indexed by stable keys
    local_statistics: dict
    library: dict[str, TemplateSpec]
    proposal_state: dict
    rng_state: dict
    counters: dict
```

Store policy parameters separately, with an immutable policy-version hash in every state. Implement serialization with explicit JSON metadata and a numerical tensor container; load only project-created, trusted checkpoints. Do not use Python `eval` to execute DSL text or arbitrary downloaded pickle files.

**Initial engineering envelope:** 64 expanded primitive nodes per task, up to 16 available templates, 2–4 primitive nodes per template, maximum template boundary arity 2, and 16 sampled legal candidate actions per decision. These are small starting configurations, not scientifically mandated maxima. Section 6 specifies measured enlargement rules. Include pressure conditions with fewer template slots than recurring motifs. Keep byte caps as well as slot counts, because a slot is not a byte.

Templates are flat in the core. They can be composed with each other in task graphs, but a template does not recursively call itself or other templates. A later hierarchical-library experiment can permit immutable calls to older versions, with explicit dependency closure, maximum expanded size and maximum call depth. It is not required for the first result.

### 4.3 Invariants

Every transaction must preserve the following:

- Every parent exists, types and arities match, the graph is acyclic, and every output is defined. Numeric batch axes are execution dimensions, not node state.
- Count expanded primitive nodes and all instantiated parameter slots. A macro call is not one unit of numerical work.
- Existing parameter identities and optimizer moments are preserved for unchanged nodes. New parameters receive fresh initialized moments; removed parameters cannot remain as usable hidden memory.
- Template definitions are immutable and versioned. Wrapping an existing group preserves its current parameters and outputs. A **new** structure-only call initializes fresh parameters. These are different operations.
- Retirement removes a template from future proposal availability; it does not corrupt graphs already expanded from it. Archived audit records are inaccessible to the learner except through an explicitly budgeted replay interface.
- No overlapping in-place autograd writes into a shared activation tensor. Build a list of output tensors in topological order; stack only where needed.
- Snapshot/rollback includes parameters, optimizer steps/moments, statistics, library, RNG, candidate sampler and counters. Both arms of an intervention begin from the same relevant snapshot.

For a future hierarchical extension, a retired-but-referenced template is not freed until its dependency reference count reaches zero. Count the entire retained closure. In version 1, executable task graphs contain their expanded bodies and do not depend on a mutable library lookup.

### 4.4 Canonicalization and candidate groups

Implement canonical hashes from operation tags, graph edges, boundary-port order and parameter-slot sharing—not from fitted numeric values, source task names, raw input-coordinate IDs, timestamps or human-readable motif names. Alpha-rename external ports in first-use order during a deterministic depth-first traversal. Preserve repeated-port identities. Canonicalize commutative `P` and `M` inputs by structural key; do not reorder `A` inputs without swapping its corresponding parameter slots.

Use exact structural equivalence only. Do not round learned coefficients to declare two functions equal. Numerically similar functions on a probe set are candidates for an explicitly approximate later experiment, not grounds for exact merging.

A candidate group is a connected, single-output subgraph whose internal nodes have no external consumers except through that output. Its incoming boundary contains at most two distinct signals. Enumerate legal groups of 2–4 primitive nodes within three ancestor hops of a root. For the main assay, exclude repeated internal sharing that would be duplicated by the tree adapter to Stitch; retain a separate shared-DAG stress test. This makes the baseline serialization faithful rather than silently comparing a DAG against an inflated tree.

Enumeration supplies possibilities, not task-specific intelligence. Sample roots uniformly and then legal groups uniformly, using a counter-based RNG and deterministic canonical ordering. Charge enumeration and hashing time. The local policy ranks only the sampled candidates. An unrestricted Stitch competitor may search a larger corpus; account for its cost rather than artificially restricting its strongest implementation.

### 4.5 Structural actions

Task-assembly actions:

| Action | Exact effect |
|---|---|
| `REWIRE(node, port, source)` | Replace one parent with a legal earlier node or terminal. `source=0` is disconnection. |
| `INSERT(node, port, op, second_source)` | Insert one primitive on the chosen input edge; one parent is the original source, the other an eligible earlier source. |
| `RETYPE(node, op)` | Change a primitive type, preserving parents; preserve only compatible parameter slots and initialize new ones. |
| `SET_OUTPUT(source)` | Choose an existing signal as output. |
| `CALL(node, port, template, bindings)` | Instantiate the template on legal signals and replace the selected input edge by its output. Parameters follow the declared reuse mode. |
| `NOOP` | Retain topology and spend the same numerical-refinement opportunity. |

Library actions after a completed training task:

| Action | Exact effect |
|---|---|
| `ADD(group)` | Preserve a legal structural template when both slot and byte caps permit. |
| `REPLACE(group, old_template)` | Atomically add the proposed template and retire the named available template. The scorer evaluates the **joint** action. |
| `DROP(template)` | Remove it from availability; record actual freed persistent bytes. |
| `NOOP` | Keep the existing library. |

No policy may exceed a resource cap and rely on a later cleanup. When a graph loses unreachable nodes, garbage collection is mechanical; deleting a still-useful pathway requires an evaluated structural action. Log illegal proposals separately rather than evaluating undefined behavior.

### 4.6 Local observations and policy capacity

Implement a versioned **56-feature** candidate vector. The feature extractor may read the candidate's root, two boundary signals, up to three ancestor hops, and that candidate/template's own accumulated history. It may not read an arbitrary graph embedding, hidden teacher metadata or a task identifier.

| Feature block | Width | Definition |
|---|---:|---|
| Action kind | 9 | One-hot: rewire, insert, retype, set-output, call, add, replace, drop, no-op. |
| Root and proposed primitive type | 6 | Two three-way one-hots; zeros for absent types. |
| Candidate size/cost | 5 | Expanded nodes, edges, arity, signed persistent-byte delta, signed trainable-slot delta. |
| Root and two boundary signal statistics | 12 | For each: mean activation, standard deviation, mean absolute output-gradient, saturation fraction `abs(h)>0.95`. |
| Bounded-neighborhood summaries | 8 | Mean absolute raw parameter, raw-parameter standard deviation, mean absolute parameter gradient, RMS last update, mean in-degree, mean out-degree, mean age, mean invocation count. |
| Past measured utility | 4 | Own utility EMA, squared-utility EMA, observation count, age since last observation. |
| Own usage history | 4 | Total uses, use EMA, root fan-out, occurrences encountered in past learner graphs. |
| Presence flags | 4 | Boundary-0 present, boundary-1 present, previous utility observed, library-phase. |
| Local shape summary | 4 | Three primitive-type frequencies and local expanded depth. |

For `REPLACE`, candidate summaries cover the proposed group, while signed size/cost deltas and past utility/use fields describe the replaced template; record this convention. `DROP` uses the retired template's stored local summaries; no fresh global evaluation enters its feature vector. Missing numeric observations are zero with flags. The feature schema must explicitly state these cases in tests.

Use signed `log1p(abs(x))` for unbounded counts/costs. Standardize using training-prefix statistics only, with an epsilon for zero variance, and clip standardized features to a development-fixed range. EMA decay is initially 0.9 and is tuned against 0.5 and 0.99 on development worlds; count it as a hyperparameter, not a biological constant.

The initial scorer is `Linear(56,32) -> SiLU -> Linear(32,2)`, **1,890 parameters**. One head estimates current-task edit value; the other estimates finite-horizon library value. This is a shared policy, not a 1,890-parameter network inside every unit. A unit still performs only its small primitive computation.

Gradients and scalar intervention outcomes originate from task-level losses. Accordingly, the claim is **local policy observations with global teaching signals**, not biologically local learning or decentralized discovery without an external trainer. Training/evaluating the policy is a centralized experimental procedure. At execution the registry only enforces legality, accounting and deterministic arbitration among policy scores.

### 4.7 Numerical fitting and task solver

Use a single-parent stochastic graph search with numerical refinement. Start with one `A` unit on two legal input terminals. Permit neutral and occasionally worse moves to cross valleys; maintain a separate best support-validation incumbent. A graph-search comparator uses exactly this solver with different action selection.

Split the 64-example initial support configuration into 48 fitting and 16 selection examples with a public split seed. These values are development starting points. All methods get the same complete 64 examples; independent baselines may choose another training-only allocation or cross-validation strategy. No baseline is forced into a weak 48-example fit solely to mimic SDML.

Use Adam on raw parameters, no weight decay initially, with learning rate selected on development data. Begin with 8 full-fit-set gradient steps per structural proposal. Compare smaller and larger bundles when profiling; the frozen confirmatory setting follows numerical progress and overhead, not a desired SDML result. A no-op also receives numerical fitting, so a structural action does not get additional free optimization.

Metropolis acceptance for the temporary search state is

\[
p_{\rm accept}=\min\{1,\exp[-(\ell'_{\rm sel}-\ell_{\rm sel})/\tau]\}.
\]

At zero temperature accept nonworsening proposals only. Initialize \(\tau\) from the median absolute proposal loss change in the development warm-up and decay continuously to zero over the allowed adaptation budget. Tune a greedy alternative. This annealing is an established search device, not the proposed contribution. Best-incumbent updates use selection loss; ties prefer lower expanded cost, then deterministic hash order. Return the incumbent, not the last sampled graph.

The deadline includes initialization, feature extraction, proposals, graph compilation/expansion, candidate fits, validation and any within-task library operations. A complete update crossing the deadline is ineligible for the returned incumbent. Report overrun work as consumed cost. Maintain a ready prediction from initialization so every method can return a result.

**Autograd edge cases.** Call `retain_grad()` only on differentiable nonleaf signals needed for local observations. For a parameter-free graph, compute input/signal gradients in a separate tracked forward pass when requested; do not instantiate an optimizer with an empty parameter list or call backward on an untracked scalar. Missing/unreachable gradients are zero with explicit presence metadata, not NaN. Numerical fitting for a parameter-free graph is a no-update evaluation; account for its real cost rather than fabricating eight optimizer steps. Global/local controls must share this convention. Reject nonfinite candidate values/gradients and restore the entire parent snapshot; record the failure as consumed work.

### 4.8 Training structural credit without a circular start

**Bootstrap:** run the uniform-action solver on training-prefix tasks. Collect all valid snapshots, fitted graphs, errors, action proposals and numerical statistics—not only successes. No teacher structure labels are used. The bootstrap cost is charged. The isolated assay gives all selectors the same bootstrap corpus and charges that shared corpus acquisition to each complete pipeline.

**Assembly labels:** choose a legal action uniformly from the recorded candidate pool. Compare candidate versus no-op from the same snapshot, with equal additional numerical-fitting budget. Target the difference on the support-selection split. This trains the first head. On a separate training-only check, test whether this target predicts independent within-prefix query performance; failure indicates support overfitting rather than useful structural credit.

**Library labels:** after a source task, choose one add/replace/drop/no-op intervention. Fork the state and run the same frozen continuation solver on `H` training-only probe tasks in each arm. Allow neither branch to see the probe query before its adaptation completes. Only the experimenter computes the difference afterward. Begin with `H=4`; estimate label noise by repeating paired probes on fresh task sets, and increase `H` only when noise materially limits selection. Probe tasks must be from the training prefix distribution, never protected recombinations or future evaluation batches.

For the clean marginal-value assay, freeze other library edits during the continuation horizon. Later allow the full continuation policy in both arms as a separate closed-loop test. The resulting labels answer different questions and must not be pooled without an indicator and a declared objective.

Use separate task folds for source fitting, utility probes and policy validation. Every baseline has access to the same total observed task data; a baseline need not adopt SDML's split. Fit the scorer with Huber regression; set the Huber transition from the training target scale, handling a zero-scale target by a constant predictor. Select checkpoints on policy-validation worlds/tasks, not on the label-training loss.

After the bootstrap, collect additional intervention data from a mixture of uniform and current-policy trajectories. Begin with equal mixture probability to preserve coverage; freeze the final mixture after development. Keep the behavior-policy probability and action pool in every record. This is supervised policy improvement on intervention values, not an unbiased off-policy policy-gradient theorem. A fresh-policy evaluation is required because its state distribution can differ from bootstrap data.

**Test-time decision:** score the sampled legal actions, compare the appropriate head against no-op value zero, and select the highest positive value. Use a development-calibrated uncertainty/threshold option only if it improves held-out decisions; otherwise use zero. During assembly, a no-op still refines numerical parameters. During library maintenance, no positive proposal means no change. Exploration can remain in assembly's shared stochastic solver, but its rate is frozen and charged; protected evaluation never trains the scorer on its queries.

### 4.9 Core pseudocode

```text
TRAIN_PREFIX(tasks, config):
    partition tasks into source / utility_probe / policy_validation folds
    corpus, snapshots = uniform_solver(source_tasks, charged_budget)
    repeat development-selected training rounds:
        for snapshot selected independently of its future outcome:
            pool = sample_legal_candidates(snapshot, counter_rng)
            action = behavior_policy(pool)  # record its probability
            features = LOCAL_FEATURES(snapshot, action)
            if action is an assembly edit:
                value = paired_current_task_trial(snapshot, action, NOOP)
            else:
                probes = draw_training_probe_tasks_without_test_access()
                value = paired_future_trial(snapshot, action, NOOP, probes)
            append record including both arms, failures, cost and provenance
        fit value scorer on allowed training records
        validate decisions on separate prefix-validation tasks
        collect next trajectories with frozen current scorer plus exploration
    freeze scorer, feature normalization, solver and maintenance hyperparameters
    develop final prefix library using only allowed prefix observations
    return frozen policy and immutable prefix library

FIT_TASK(support, prefix_library, policy, deadline):
    state = initial_graph_and_fresh_parameters(support)
    incumbent = ready_initial_prediction(state)
    while charged_time_remaining():
        pool = sample_legal_candidates(state)
        action = select_with_policy_or_baseline(pool, state)
        trial = apply_transaction_to_snapshot(state, action)
        fit_parameters(trial, support.fit, same_refinement_budget)
        evaluate_on_support_selection_only(trial)
        if update_completed_before_deadline:
            update_temporary_state_by_shared_annealing_rule()
            update_incumbent_if_better()
    return stateless_predictor(incumbent), full_cost_log

PAIRED_FUTURE_TRIAL(snapshot, action, noop, probes):
    start two logically identical snapshots
    change only the declared library intervention in the treatment
    hold continuation policy and every other permitted setting fixed
    randomize physical execution order of arms
    for probe_task in probes:
        use coupled random streams and identical observed support examples
        fit each arm within its own equal charged adaptation deadline
        experimenter scores independent training-only queries after fitting
    return mean(control_error - treatment_error), both raw trajectories
```

A coupled RNG means common underlying random variates and stable semantic keys where possible, not forcing identical actions from different legal libraries. Record where branching changes the realized sequence. Candidate and no-op **must not share mutable tensors**.

### 4.10 Reuse modes and compulsory separation

Run structure-only mode first: parameters of every new macro call are fresh draws from the common primitive prior. It isolates reusable organization. Then, separately:

- `copy_init`: initialize a call from the source circuit's learned values, with independent subsequent parameters;
- `frozen_shared`: calls use frozen shared parameters;
- `tied_trainable`: calls share trainable values, with replay and interference measured;
- `inline_alias`: replace macro notation by its expanded body, preserving exact parameter identities and computation.

Every instantiated scalar and optimizer state counts. Keep the structure-selection policy and evaluation protocol fixed when isolating these choices. A benefit only in `copy_init` is valuable parameter transfer but does not establish H3 in structure-only form.

## 5. Data, baselines and confirmatory evaluation

### 5.1 Synthetic workload: hidden numerical compositions

The main dataset is generated locally by project code and needs no external asset. Publish the generator, configuration, seeds, code license and split manifests. The protected teacher interpreter lives in the evaluator package, never in a learner process. The benchmark is deliberately synthetic: it can establish controlled compositional transfer, not general intelligence or commercial usefulness.

**Generator version 1:**

1. Set `d=4`. Draw eight unique two-input motif skeletons per world by uniformly sampling legal flat graphs with 2–4 primitive nodes. Require both external ports to affect the output syntactically, at least one `A`, at least one `P` or `M`, and every internal node to be reachable from the output. Canonicalize as in Section 4.4. This is a public grammar, not a list of handpicked “winning” motifs.
2. Draw task programs by composing motif calls in ordered binary trees/chains, with terminals bound to input coordinates or constants. Sample both balanced and chain-like shapes, with equal mixture weight. Draw fresh raw `A` coefficients from `Normal(0,0.7²)` **for each task instance**, including each motif occurrence. Thus recurring structure does not imply reusable exact coefficients.
3. Training-prefix tasks have outer motif depth 1–3. Same-depth held-out recombination tasks have depth 2–3. Depth extrapolation tasks have depth 4–6. Reject only syntactically illegal, duplicate or resource-incompatible structures by declared rules; do not reject tasks because a method performs badly. Record output-variance and saturation strata, including near-constant tasks.
4. Split ordered parent/child motif pairs using a stable hash of `(world_seed, parent_id, child_id) mod 5`: values 0–2 train, 3 development, 4 protected. Force a randomly permuted directed ring and its reverse into the training set so all motifs occur in a connected set of training contexts. Record those forced edges before task generation. Training uses only training pairs; development contains a development pair and no protected pair; recombination tests contain a protected pair and otherwise training pairs.
5. Generate enough training tasks that every motif occurs under at least two input bindings and in at least two legal training contexts. Determine the task count from this coverage condition and a prefix learning-curve pilot, not an arbitrary large corpus. Use distinct parameter draws and support/query RNG streams throughout.
6. Verify no exact parameter-free task skeleton crosses the declared structural split after input-port alpha-renaming. Log accidental function similarity using evaluator-only random probes, without pretending this is a complete semantic-equivalence test. When the same mathematical function has distinct syntactic forms, keep a separately labeled semantic-overlap analysis rather than secretly modifying the split after seeing results.
7. Draw support and query inputs independently from uniform `[-1,1]^4`. The primary noiseless condition measures structure/optimization without label-noise confounding. Add a separately frozen noise condition, with noise variance set from a training-only signal-to-noise specification; clip noisy targets consistently and disclose the changed distribution.

**Outer-tree capacity convention.** The teacher's expanded node cap is the same registered representational cap as the circuit learners for the primary test. Never let a nominal depth setting silently create an unrepresentable full binary tree. Generate chains with exactly `D` motif calls. For the branching component, sample a call count between `D` and `min(2**D-1, floor(N_cap/4))`, start with a depth-`D` spine, then add calls at available shallower terminal positions until reaching that count. Prefer minimum-depth available positions uniformly to make this component bushier; randomly orient the spine. Fill remaining leaves with uniformly sampled public terminals. Sample motif identities conditionally on the required pair split and record rejection counts. This is a bounded branching distribution, not uniform sampling of all trees. Both shape strata remain explicit. If no legal protected pair or tree exists for a sampled world, regenerate it based solely on grammar feasibility and log the reason. Do not use numerical learner performance to reject a world.

Initial support size is 64 and query size 256. Pilot support sizes on a geometric grid starting at 8; choose a confirmatory size where an informed oracle and credible baselines are distinguishable and the task is neither unidentified from support nor saturated. Increase query count until query Monte Carlo variation contributes substantially less than between-world variation in paired method differences. A quantitative initial rule is query Monte Carlo standard error below one fifth of the between-world standard deviation; document residual query noise. This controls measurement precision, not the improvement needed for success.

**Primary population:** same-depth held-out recombinations, equally weighted across the two predefined library-pressure conditions `K=4` and `K=16`. Report each stratum as well. Depth extrapolation is a predefined secondary population because it changes difficulty and can suffer derivative attenuation. Do not mix the two after observing which favors SDML.

**Independent worlds:** development worlds and confirmatory worlds have different motif libraries and seeds. Within each confirmatory world, every method receives the same allowed training prefix and support/query realization. It trains its own permitted state from that prefix using the frozen recipe. Consequently, the experimental unit includes the entire learned developmental history, not just a random restart of one already-trained model.

**Negative/control generators:** no recurring motifs; motif-frequency reversal; identical past but uninformative future; very small support; constant/near-constant targets; and teachers whose operation vocabulary includes a withheld primitive. Do not combine these with the primary population. They establish the mechanism's limits.

### 5.2 Five evaluation tracks, with separate claims

**E0 — correctness and information audit.** Test primitive values/gradients, interpreter expansion, transactions, split protection and timing. No performance claim.

**E1 — reuse-headroom assay.** Give a diagnostic oracle the true *structural* motif templates but not task-specific coefficients or final program composition. Compare against primitive-only and fixed-module search at identical expanded bounds and deadlines. This extra-information oracle is not a competitive baseline. Separately fit the known teacher graph to support to test numerical optimization and identifiability. Neither oracle may be included in a “fair head-to-head” ranking.

**E2 — isolated consolidation assay.** Give each selector the identical primitive-only learner-generated corpus and sampled legal group pool. Compare no-library, frequency, MDL, local learned value, global learned value and a training-only fresh-probe upper reference. Use identical continuation search. This isolates selector quality conditional on the shared solver and corpus. The direct-probe reference evaluates choices on allowed probe data and scores final choices on fresh data; it cannot select and score on the same noisy probe realization.

**E3 — complete developmental loop.** Enable learned assembly, actual birth/rewiring, library consolidation/replacement and retirement during the training prefix. Compare whole pipelines, each with its own allowed training history and cost. This is the primary architectural test. During held-out task evaluation, freeze the prefix library across independent query tasks: adaptation constructs task-specific graphs but does not leak one task's hidden query into another. Test-time library updates belong to E4.

**E4 — online shift/adaptation.** After a prefix, change motif probabilities in an explicit synthetic stream. Evaluate each task's query before any outcome could influence subsequent updates. Only support labels are revealed to the learner; query labels remain withheld. Permit library maintenance using support evidence and frozen policy weights. Compare equal replay and memory budgets; report adaptation area under the error curve and old-task forgetting. A shift can be detectable from recent experience but cannot be anticipated from an identical past without an observable signal.

### 5.3 Locally executable baseline basket

Use the strongest credible versions permitted by the information and hardware, not only the most convenient weak neural baseline. Every method has a `fit_task` adapter and is evaluated through the same protected evaluator. The final baseline manifest records exact versions, source revisions, implementation status and tuning results.

| ID | Required implementation/configuration family | What it tests / reproduction check |
|---|---|---|
| B0 | Support-mean predictor; ridge linear model; RBF kernel ridge with training-only regularization/bandwidth selection. | Essential low-cost controls for four-dimensional smooth regression. Kernel ridge solves `(K + alpha I)c=y`; test an analytic linear target and an exact small kernel system. |
| B1 | Dense tanh MLP, from-scratch and Reptile-style meta-trained initialization [R34]; independently tune depth, width, learning rate and regularization. Include a compact residual MLP if it improves development performance. | Ordinary function approximation and parameter transfer. Verify convergence on simple primitive targets; compare against the separately fitted known-teacher-graph diagnostic. Charge prefix training. |
| B2 | Fixed reusable-module learner using BounceGrad-style alternating structural search and gradient updates; add post-selection parameter adaptation, and test the MOMA-style inner-adaptation objective where it improves development performance. | Strong numerical modularity without library growth [R09]. Use the same primitive family and also small learned two-input modules, with capacity/time frontiers rather than forcing equivalently tiny modules. |
| B3 | RigL-style sparse MLP port: fixed active parameter budget; magnitude removal; pre-removal dense candidate-gradient growth; update interval/drop schedule tuned on development. Also run the corresponding static sparse network. | Whether ordinary rewiring explains a gain. Validate mask cardinality, off-mask growth signals and optimizer-state behavior against a hand-computed example and inspected source [R10]. |
| B4 | Shared circuit solver + frequency/LRU selection; shared solver + exact description-length selection; `CGP-style` neutral stochastic mutation with the same legal edits. | Cheap structural alternatives. Frequency counts actual occurrences in learned graphs; it receives no teacher motifs. MDL counts definitions plus rewritten corpus, including call tokens and parameter-slot structure. |
| B5 | Official Stitch compressor on learner-produced programs, paired with the same continuation solver. Evaluate its preferred unconstrained discovery pool as well as the pool-matched assay. | The closest strong runnable library component [R06]. Parse/rewrite/execute round-trip must preserve numerical and parameter-slot semantics. It is a **Stitch-based composite baseline**, not a claimed full DreamCoder reproduction. |
| B6 | Official PySR on CPU, with the public numerical operator set represented faithfully and constants optimized; allow its native search and tune within the same data/time envelope. | Strong independent symbolic regression, not merely another variant of the SDML solver [R32]. Export selected expressions and independently verify predictions. |
| B7 | Global learned utility selector, using the same intervention labels; one approximately parameter-matched pooled-context variant and one stronger global encoder within the common resource envelope. | Whether apparent “local discovery” actually needs global context. Not an oracle and not proof that all possible global learners have been exhausted. |
| B8 | Function Encoder with learned neural bases and least-squares task coefficients; official algorithm/reference behavior, locally adapted regression harness [R35]. | A strong fixed-representation numerical transfer alternative requiring no structural search. Validate basis/coefficient reconstruction and rank-deficient support handling. |

**Required SDML attribution controls:** full model without consolidation; full model with the development-winning heuristic selector; and inline aliases with identical parameter sharing. The first two participate in the confirmatory contrast family. Others are diagnostic unless separately registered.

**MLP initialization baseline.** Pooled regression across conflicting task functions is not adequate meta-learning. For B1, implement Reptile's first-order update: clone a shared initialization, take several gradient steps on one source task, then move the initialization toward that task's fitted weights, `theta <- theta + eta*(theta_task-theta)` [R34]. Tune inner steps, inner/outer learning rates and task batching on development worlds; at test adapt only to support. Include scratch training so a poor prior does not handicap the baseline.

**Function Encoder baseline.** The published method learns neural basis functions and fits task coefficients by least squares, with basis-norm regularization [R35]. Our regression adapter uses a multi-output MLP `g(x) in R^k`; solve `(Phi.T@Phi/n + alpha*I)c = Phi.T@y/n` on support and predict `g(x)@c`. Use a stable solve, not an explicit inverse. Train bases with source-task fitting/validation examples, differentiating through the regularized solve. Include the norm penalty `sum_j(mean_x g_j(x)^2-1)^2` and a residual-mean-function option. Freeze bases at test; also compare a separately tuned fine-tuning variant. These adapter/configuration choices are proposed here, not claimed paper settings.

Begin development search at basis counts 8, 32 and 128 and tanh MLP hidden widths 32 and 128; vary ridge and norm-penalty strengths logarithmically, expanding useful boundaries. All basis parameters and training are charged. Test against a hand-built polynomial basis and singular/duplicate columns. The official experimental scripts assume multiple GPUs; do not run their full sweep or infer that the underlying method requires them. The inspected repository links the `FunctionEncoder` package and an experimental version pin; resolve the actual source/license and reproduce a small original regression check before selecting a locally pinned dependency. No permissive license was established in the inspected repository view: do not vendor unlicensed code. A clean equation-based implementation is feasible and must be labeled as an algorithm-level adaptation.

**Global-policy challenge specification.** The near-capacity-matched variant appends eight whole-graph summaries to the local 56 features: expanded nodes, edges, output mean/std, fitting loss, selection loss, mean absolute output-gradient and fraction of nodes originating in macros. Use `64 -> 28 -> 2` (1,878 parameters versus local 1,890), the same activation and training targets. The stronger variant embeds every node's 12 features (operation one-hot; activation mean/std/absolute-gradient/saturation; in/out degree; age; macro-origin flag; mean absolute raw parameter) into width 32. Apply two rounds of message passing using separate mean parent/child embeddings, concatenated with the current embedding and projected back to 32. Concatenate graph mean/max pooling, the candidate features, fitting loss and selection loss; score with a width-64 two-head MLP. Tune widths/layers on the same resource frontier. Charge full-graph extraction and training. Permuting nonsemantic node IDs must not change either score. No global version receives teacher metadata, final queries or unseen future tasks.

**Fixed-module numerical baseline specification.** Maintain a bank of learned two-input scalar modules and a legal composition skeleton family. Alternate simulated-annealing structure moves on task fitting examples with gradient updates to shared module parameters on disjoint training-task validation examples. At meta-test search structures using only support; compare frozen modules versus adapted copies. The original BounceGrad algorithm supplies this separation [R09]. Tune bank size and module width independently. Match total persistent bytes and report a parameter-matched slice, but do not handicap the baseline by giving SDML additional source-task data or optimization passes.

**RigL port detail.** If `W_eff = W * mask`, then `W.grad` is zero off-mask and is not the full regrowth signal. Retain the gradient of `W_eff`, or compute the mathematically equivalent unmasked candidate gradient, before selecting new connections. Record whether a topology-update step also performs a weight update; preserve the chosen source schedule or label the deviation. Masked dense kernels are acceptable for correctness but do not justify sparse wall-time claims. Count inactive storage and dense candidate-gradient work.

**Stitch adapter.** Serialize closed parameter-free structural skeletons with tokens `A`, `P`, `M` and lexical input variables. Each `A` occurrence creates fresh parameter slots on later instantiation in structure-only mode. Restrict the pool-matched assay to groups with no repeated internal subgraph sharing so the tree representation does not duplicate computation. Do not insert fitted floating-point constants as task-specific token names. Import each abstraction into the typed template runtime, preserving repeated argument identities and validating arity. Reject unsupported abstractions explicitly and record their frequency; widespread rejection invalidates the comparator and blocks a broad claim until the adapter is fixed.

**PySR adapter.** A faithful numerical expression can represent `A` through `tanh`, multiplication, addition and bounded constants; charge its complete expression complexity, not an artificial one-token equality with SDML. Use two views: the exact common DSL where the engine permits it, and native symbolic-regression performance under the common wall-time/data/memory envelope. This distinction prevents both artificially handicapping PySR and hiding its richer search space. Run source-supported custom-operator smoke tests before the benchmark; do not guess an unsupported API.

### 5.4 Wider references and exclusions

Attempt the official DreamCoder native smoke build early enough to affect baseline planning. Its inspected repository has older OCaml/Python/Rust and platform-specific instructions [R05]. If a faithful compact numeric-domain integration works and remains competitive, include it. If it fails, record the actual build failure and use the strong Stitch-based comparator for the local claim; do not assert that DreamCoder cannot run on Apple Silicon merely from the age of its instructions.

HOUDINI, LNDP and LMC are important mechanism precedents. A purpose-built local port is not a reproduction of their published scores. Include an algorithm-level local-growth/module-relevance control where it is straightforward, with differences documented. The current learned-value proposal must still be compared directly with the stronger independent library and modular baselines above.

SkillMaster, ALMA and full VLA experiments are broader reference results, not local head-to-head candidates: their task information and large-model/agent infrastructure answer different questions [R18], [R19], [R22]. Excluding them from this numerical benchmark does not restore novelty to their established ideas.

### 5.5 Baseline tuning and fairness

Choose all hyperparameters using development worlds. First validate reference behavior, then tune. Use log-spaced initial learning rates/regularizers and geometric capacities; expand any boundary on which the winner sits. For solver methods tune primitive/refinement balance, temperature, restart strategy, candidate count, template capacity and library-update frequency. For MLP/kernel methods tune their own relevant parameters rather than borrowing SDML's preferred settings.

Include B8 and Reptile in WP03 baseline readiness and tuning; their inexpensive test adaptation can remove the assumed search headroom. Use the same initial development wall-time allowance per method and the same observed data. Extend a method's tuning when its learning curve is still improving and the expected decision value justifies it; report both equal-budget and extended-tuned results. The final comparator must not be known to be undertrained. Freeze tuning search spaces, source versions and final configurations before protected evaluation. Never choose a baseline solely because it is easier to beat.

Maintain two resource views:

**Controlled component view:** same corpus, candidate pool, continuation solver and expanded-size limits; changes only selector or library availability. This identifies mechanism.

**Complete-pipeline view:** same observed prefix/support data and local resource envelope, but each method may use the most effective implementation, model size and solver it can afford. Count preprocessing, adaptation, structural searches, all counterfactual branches, compilation, persistent assets and final inference. Report error/time/memory Pareto frontiers rather than one parameter count.

Prefix-trained assets are not “free” because every test task shares them. Report development cost separately and amortized over the actually evaluated task counts, with a breakeven calculation. Show both data-matched and total-training-compute-matched slices if the learned policy consumes much more training work. A fast pretrained local policy is not an equal-compute win unless that additional cost is repaid or its value proposition is explicitly different.

### 5.6 Selecting the primary budget without selecting a winner

On development worlds, characterize the credible baselines' error-versus-adaptation-time curve on geometric budgets. Choose `B_*` in a region where the best baseline is neither equivalent to the support-mean floor nor effectively saturated. Operationally, choose the earliest budget at which the best baseline has achieved half of its observed improvement from initialization to the pilot's numerical plateau. Extend the pilot if no plateau has been measured. This rule uses baseline curves, not SDML's most favorable crossing point.

Treat that half-progress rule as a **budget-selection convention**, not a success threshold. Register additional smaller/larger budgets as secondary. When baselines have incompatible cold-start costs, report cold one-task and warmed repeated-task regimes separately. Set `B_*` only after include/exclude decisions for mandatory baselines, so a Julia startup or compiler cannot be silently omitted.

For an analytic/one-solve baseline whose curve reaches its plateau in one completed update, do not manufacture an intermediate point or repeatedly extend an already saturated pilot. Use its first valid completion time as the cold minimum and examine support-size/teacher-complexity headroom on development data. If no informative common budget exists, conclude that this workload does not support an adaptation-speed claim; retain equal-label quality and total-cost diagnostics with a separately registered estimand. Do not keep increasing task difficulty solely until SDML wins.

The scorer's probe deadline may initially be smaller than `B_*` for cost calibration, but validate that its action rankings predict benefit at `B_*`. A failure of ranking transfer triggers matched-horizon training or a narrower claim; it is not repaired by reporting only the cheap probe score.

### 5.7 Primary metrics, aggregation and inference

**Primary outcome:** task-balanced mean normalized MSE at `B_*` on same-depth protected recombinations. Within each world, first average tasks within each registered pressure stratum, then average strata equally. Compare methods using paired world-level differences

\[
d_w=R_{w,\rm comparator}-R_{w,\rm SDML}.
\]

Positive values favor SDML. A world includes independently generated motifs, prefix data and a complete independent training/developmental trajectory under a frozen recipe. Repeated queries, timing repetitions, bootstrap samples and multiple restarts within that world are **not independent worlds**. Extra training seeds with the same world are nested replications, averaged before the world-level analysis or modeled explicitly as crossed/nested effects.

**Confirmatory contrasts:** full SDML versus each required independently tuned competitor in the frozen eligible baseline basket, plus full versus no-consolidation and full versus the development-winning heuristic selector. Remove exact duplicate contrasts before freezing the manifest. Identify the development-best complete competitor as the focal contrast, but report all eligible competitors. Control family-wise error across this finite family with Holm's procedure at 0.05. A claim of outperforming the entire tested basket requires evidence against every relevant competitor; otherwise name only the supported comparisons.

Report absolute error differences, relative changes when the denominator is not near zero, paired uncertainty intervals and raw per-world outcomes. Use two-sided paired t inference on independent world means as the registered primary approximation, with whole-world bootstrap intervals and distributional diagnostics as sensitivity analyses. Investigate heavy tails or one-world domination during the pilot; increase independent replication or pre-register a more robust estimand before confirmation rather than switching tests afterward. Bootstrap confidence is not a cure for a handful of worlds.

**Precision and sample-size calibration:**

1. Collect independent development worlds until a useful variance/heterogeneity estimate is available; adding more query points cannot substitute for more worlds. Inspect stability of the estimated paired standard deviation and its uncertainty as pilot worlds accumulate.
2. Establish an operational error-resolution scale `delta_design`: the development-best baseline's measured improvement from one additional calibrated numerical/search update bundle around `B_*`. Increase that bundle geometrically if its effect is buried in noise. This is a unit of practically interpretable effort, not a required SDML improvement. If the baseline has no measurable remaining improvement, the budget is saturated and must be revised before confirmation.
3. Target a confidence half-width of `delta_design/2` and 90% power to detect `delta_design`, using a conservative upper estimate `sigma_U` of the paired-world standard deviation. For planning, with `J` registered contrasts, use the Bonferroni critical value `z_(1-.05/(2J))` even though the final procedure is Holm:
   `n_precision >= (z * sigma_U / (delta_design/2))^2`,
   `n_power >= ((z + z_.90) * sigma_U / delta_design)^2`.
   Refine with t quantiles or world-resampling power simulation; choose the larger requirement. Publish the resulting detectable-effect curve, not only a nominal power label.
4. Freeze that independent-world sample size before accessing protected outcomes. The planned number is an output of variance, resolution and measured cost, not “five seeds.” If it cannot be completed, report the achieved interval and the resource limitation. Do not keep adding runs until a p-value crosses a threshold.

Failure to reject zero is not equivalence. An interval excluding useful gains at the prespecified resolution is evidence against a useful effect in this setting; a wide interval is inconclusive. Any positive effect smaller than the operational resolution can still be real, but its practical relevance must be evaluated separately. There is no arbitrary percentage improvement needed to qualify.

**Secondary measurements:** error/time and error/label curves; total prefix and adaptation seconds; end-to-end inference latency at batch sizes 1 and the relevant workload batch; peak process/device memory; persistent library/replay bytes; expanded primitive work; active and allocated parameter counts; accepted/rejected/illegal edits; number and diversity of contexts per macro; utility-prediction calibration; template survival; forgetting; and failures/timeouts.

A scalar speedup is permitted only at a common quality target within the overlapping empirical ranges. Do not extrapolate an unobserved curve intersection. A quality gain concealing a latency or memory regression is reported as a tradeoff, not dominance.

### 5.8 Attribution tests and leakage protection

Perform **causal macro removal** at deployment: disable learned templates as atomic proposal actions, leaving already fitted predictions and numerical capacity unchanged, and re-run task adaptation from the same initial snapshot. This tests their role in learning. Replacing an already invoked macro by identical inlined operations is the semantic null test; it should preserve predictions and may change dispatch overhead only.

Also reinitialize macro call parameters; shuffle intervention targets across matched size/frequency strata; swap local/global feature access; match added parameter counts with unused/random templates; and compare to a generic multi-node mutation with the same expanded size. The last test distinguishes a useful learned abstraction from merely permitting larger moves.

Protected query targets exist only in the evaluator's process/files. Learners receive support examples and task-local opaque IDs. Dataset hashes, filenames and logging fields must not encode teacher motifs or split labels. A `SpyLearner` test must fail when attempting to access teacher ASTs or query labels. The evaluator may use teacher metadata for stratified reporting only after outputs are committed.

Do not train the local scorer on a candidate's same future outcomes and then report those outcomes as successful generalization. Selection probes and final validation probes must be different. Likewise, repeatedly reusing support-selection data is allowed task adaptation, but its generalization is assessed only on independent protected queries and its overfitting is measured.

### 5.9 Independent family and realistic extension

**Independent synthetic family:** reproduce a compact teacher/student condition from the official modular-hyperteacher repository [R14]. Use its documented task-support/connectedness settings, expose only the same support information to competing learners, and keep its teacher generator separate from SDML's motif code. Begin on CPU JAX or a numerically checked NumPy/PyTorch port; do not claim its published results have been reproduced until an original small reference case is matched. If SDML's scalar primitive closure cannot approximate that teacher at the allowed size, first diagnose representation limits using the teacher-graph/fixed-capacity controls. This test is not a permission to redesign the benchmark until SDML wins.

**Real-data case study, conditional on licensing and core gates:** UCI Gas Sensor Array Drift, dataset 224, DOI `10.24432/C5RP6W` [R33]. The inspected record describes 13,910 examples, 128 features from 16 sensors, six classes and ten batches; the compressed asset is small. Use the corrected `batch10.dat` provenance and record the exact downloaded archive hash. Parse the LIBSVM-style features explicitly into 128 columns; verify labels, finiteness, duplicates, counts and missingness locally rather than trusting metadata alone.

A material provenance conflict exists: the page's license footer says CC BY 4.0, while its descriptive terms say research-only and exclude commercial use. Do not silently choose the more permissive interpretation. For this research PoC, use conservatively for research, retain attribution, do not redistribute raw data, and block any incompatible/commercial deployment pending clarification. This is not a legal opinion. A blocked extension does not block the asset-free core.

Use batches 1–5 for training, 6–7 for development, and 8–10 for protected chronological testing. Learn all scalers, imputers if needed, hyperparameters and feature selection from revealed data only. Do not use the UCI page's batch-specific optimal SVM settings for the protected batches. Exact within-batch temporal order is not established by this plan: report drift at batch granularity, and label any random within-batch training episodes as such.

For the neural extension, use a bounded feature transform fitted on training data, a shared scalar-DAG layer with a fixed number of selected output signals, and an ordinary six-class linear readout. This requires a declared multi-output `GraphSpec` extension but does not enlarge individual units. Train local credit from revealed training-batch episodes; classify on cross-entropy and evaluate balanced accuracy/error and log loss. Freeze the policy before protected batches. Use uniform seeded support sampling within each test batch at registered label budgets; do not balance support by reading unrevealed class labels. Queries are the disjoint remaining examples, with class counts disclosed.

Required real-data competitors are RBF SVM, a strong tree ensemble, dense MLP with replay, fixed modular learner and the corresponding no-structure/no-consolidation variants. Permit independently tuned standard preprocessing for the practical comparison; keep identical preprocessing only for the component-control view. Account for all replay and prefix training. Select support and time budgets on batches 6–7 using the same principles as above.

This is one physical dataset/chronological stream. Random splits, seeds and thousands of sensor records do not create independent drift environments. Report it as an external case study with per-batch outcomes and sensitivity, not as a p-value-backed claim of general real-world adaptation. Synthetic success followed by failure here limits the application claim; it does not erase a valid component result.

## 6. M3 Max feasibility and hardware-adaptive execution

### 6.1 What is verified and what is not

Official PyTorch documentation exposes an MPS backend, synchronization, allocator-memory reporting and MPS RNG-state APIs [R27]–[R29]. The inspected documentation's stable links resolved to versioned 2.14 pages; this is **not** a claim that a particular release is installed on the user's laptop. A compatible official wheel, operating system, exact operations and gradients must be tested locally.

No M3 Max was available for this research. The only executed numerical checks were on Linux x86_64 CPU with PyTorch 2.10.0+cpu, described in Section 12. No throughput, battery, thermal, MPS-gradient or peak-memory claim has been measured on the target. Feasibility is supported by a deliberately small memory/compute substrate and faithful CPU fallback, not by an invented laptop benchmark.

### 6.2 Initial discovery: existing system commands

Run these **existing macOS commands**, saving stdout/stderr and exit codes to `artifacts/environment/`. They are read-only discovery, not an instruction to alter the machine:

```bash
uname -m
sw_vers
sysctl -n hw.memsize
sysctl -n hw.physicalcpu
sysctl -n hw.logicalcpu
system_profiler SPHardwareDataType SPDisplaysDataType
df -h .
vm_stat
python3 --version
```

Confirm native `arm64`, not an accidental x86/Rosetta environment. Record actual unified memory, free disk, macOS build, GPU identity, Python executable/architecture, available CPU cores, power mode and whether the machine is plugged in. Do not expose serial numbers or personal machine identifiers in a public report; redact those fields from `system_profiler` output.

Use an isolated environment. Select a supported, non-prerelease native CPython minor with compatible official PyTorch and numerical-package wheels; start from an already available supported version, and record any reason for choosing another. Do not upgrade the user's global Python or system packages silently. Resolve exact dependency versions, then freeze them before meaningful runs. Run tests before choosing a release merely because it is newest.

### 6.3 Stack and dependency gates

**Primary numerical stack:** Python, PyTorch eager execution, NumPy/SciPy, a small testing framework, and a process-memory reader such as psutil. Use a CPU graph registry and CPU scalar-control logic. Batch examples within each primitive. Avoid requiring sparse tensor kernels, custom Metal code, differentiable topology changes or compiler support for variable-shaped graphs.

**Library comparator:** official `stitch_core` bindings and their Rust implementation [R06], [R31]. **Independent symbolic comparator:** PySR and its Julia dependency [R32], in a separate environment if necessary. **Independent teacher:** official modular-hyperteacher code, initially JAX CPU [R14]. Install conditional stacks only after core correctness and the corresponding baseline gate, not as a large up-front environment project.

Official Stitch documentation shows the following real API; this is a smoke example, not the complete numerical adapter:

```python
from stitch_core import compress
result = compress(["(f (g a))", "(f (g b))"],
                  iterations=1, max_arity=2, threads=1)
print(result.abstractions)
print(result.rewritten)
```

Confirm this interface against the installed version and save `result.json` when supported as documented. The documentation lists `pip install stitch_core --upgrade`; for reproducible experiments, resolve once and then install the pinned version rather than continually upgrading. A missing arm64 wheel triggers a native Rust source-build attempt from the official repository, not a claim of incompatibility. The official repository documents `cargo run --release --bin=compress -- ...` and tests; use its current checked-out instructions rather than inventing a binary interface.

PySR's inspected official README exposes `PySRRegressor`, `.fit(X, y)`, `.predict(X)` and custom Julia operators [R32]. Smoke-test the operator mapping and exported expression. Record Julia startup/compilation separately and include it in the cold-pipeline cost. Its persistent search state is not a free warm start from protected tasks.

MLX is an optional optimization route, not a second mandatory implementation. Its compiler documentation warns that shapes/state affect compilation behavior [R30]. Frequent graph changes can defeat compilation amortization. A backend change requires primitive values, gradients, parameter-sharing and optimizer-step tests, plus repeated baseline measurements. Do not switch only SDML to a faster implementation and leave a generically optimizable comparator unoptimized.

### 6.4 Required local operation matrix

The proposed `python -m sdml doctor` must test, on CPU and then MPS if available:

| Operation/path | Required check |
|---|---|
| Bounded affine-tanh, product, mean | Outputs and first derivatives against CPU float64; finite differences for sampled parameters and inputs. |
| Gather/index selection and concatenation | Correct shapes, indices and gradients; repeated-source accumulation. |
| Full DAG backward + Adam | Loss decreases on a known graph; unchanged parameters retain their correct optimizer moments after a structural transaction. |
| Exact macro aliases | Same outputs and shared/copy parameter gradients within a dtype-calibrated tolerance. |
| Snapshot/resume | Same committed action sequence and predictions on deterministic CPU tests; declared tolerance and state restoration checks on MPS. |
| Timing | `torch.mps.synchronize()` before/after timed asynchronous work; no MPS calls on an unavailable backend. |
| Memory | Process RSS in bytes; MPS allocated and driver-allocated bytes where those functions exist; recommended memory where exposed. |
| Hidden fallback | Run capability tests without `PYTORCH_ENABLE_MPS_FALLBACK=1`; an unsupported path must fail visibly rather than silently becoming a mixed CPU benchmark. |

Use feature detection for `torch.backends.mps.is_built/is_available` and MPS utility functions. A successful import is not an operation-level validation. Keep CPU float64 reference tests separate from the primary float32 training. Do not introduce half precision, quantization or fast-math switches in the core. Never disable the MPS high-watermark safety limit with a zero ratio; official documentation warns that this can cause system failure [R28].

If a required operation cannot be implemented faithfully on MPS, run the complete affected method on CPU. This is a valid local fallback. Do not replace the operation with a mathematically different approximation without a separate experimental condition. For tiny graphs, CPU can be the better implementation; choose by measured complete-step throughput, not GPU branding.

### 6.5 Memory model and numerical examples

An initial planning model, in bytes, is

\[
M_{\rm peak}\approx M_{\rm runtime}+M_{\rm registry}+M_{\rm replay}+M_{\rm cache}
 +\sum_{\rm live\ copies}(4P+4P_{\rm grad}+8P_{\rm Adam})
 +4\,a\,B\,N_e\,V+M_{\rm feature\ buffer}+M_{\rm temporary}.
\]

Here `P` counts stored float32 parameters, `P_grad` those with gradients, `P_Adam` those with two float32 moments, `B` is example batch size, `V` simultaneously live candidate graphs and `a` an empirically calibrated saved-activation multiplier. Frozen parameters do not require Adam state. Add allocator fragmentation, Python object overhead, framework workspace and compiled caches; do not use this tensor-only sum as a safety limit.

**Arithmetic illustrations, not measured peaks:**

| Contributor | Assumptions | Bytes / approximate scale |
|---|---|---:|
| Trainable model plus gradients and Adam | 1,000,000 float32 parameters | 16,000,000 bytes before workspace |
| Saved graph activations | `B=128`, `N_e=64`, `a=8`, one live candidate | 262,144 bytes |
| Same activation estimate, 16 parallel candidates | All candidate graphs retained concurrently | 4,194,304 bytes |
| Support replay | 1,024 tasks × 64 examples × `(4 inputs + 1 target)` × 4 bytes | 1,310,720 bytes |
| Local-policy feature buffer | 10,000 candidates × 56 float32 features | 2,240,000 bytes |
| Gas-drift feature matrix only | 13,910 × 128 float32 values | 7,121,920 bytes |

The 1,890-parameter local scorer and a 64-node task graph are far smaller than a million-parameter baseline; interpreter objects and stored trajectories can dominate them. Paired rollouts should be serialized with copy-on-write metadata and one active graph per numerical arm where possible. Do not retain all horizons' autograd tapes. Store historical graphs as compressed structural deltas plus parameter snapshots only when needed for reproducibility.

A practical initial working-memory target is the smaller of 40% of physical memory and current available memory minus a reserve of at least 4 GiB or 20% of physical memory, whichever reserve is larger. This is a conservative engineering margin, not a measured Apple limit. If that target is too small for the chosen packages, use one CPU worker and the minimum compatible stack, measure actual peaks and revise the operational envelope before dependent experiments. The core tensor sizes should not require a high-memory M3 Max configuration.

### 6.6 Calibration, scaling and faithful fallback

Measure complete forward/backward/refinement/search steps for graph sizes 8, 16, 32 and 64, and example batches spanning the intended supports. Compare CPU thread counts and MPS after warm-up. Record medians, tails, process/driver memory and end-to-end task fit times. Test candidates sequentially first. Only batch candidates if measured throughput improves and peak memory remains safe.

Fit a simple throughput model from measured components, then verify it on a full small developmental episode. For `I` library interventions and horizon `H`, the dominant probe workload is approximately

\[
T_{\rm labels}\approx 2 I H\,\bar T_{\rm fit}(B_{\rm probe})+T_{\rm snapshots}+T_{\rm scoring}+T_{\rm policy\ fit}.
\]

Add assembly-label trials, bootstrap search and every rejected intervention. A cheap deployment scorer can have an expensive training multiplier. In a confirmatory design with `W` worlds and methods `m`, estimate

\[
T_{\rm confirm}\approx \sum_{w,m}(T_{\rm prefix,w,m}+n_{\rm tasks,w}T_{\rm adapt,w,m}+T_{\rm query,w,m}).
\]

**Stage-specific workload planning:**

| Stage | Work and memory to budget before launching | Calibration output required |
|---|---|---|
| Bootstrap / baseline training | Number of source tasks times measured fit cost; all optimizer and replay state, independent baselines' native workspaces | Prefix learning curves, full-state peak bytes, seconds per task and data coverage |
| Assembly credit | Two refinements per sampled edit plus cloning/scoring | Seconds per paired edit, target variance and rejection/failure rate |
| Library credit | `2*I*H` probe fits plus snapshot/policy costs, serialized arms | Measured seconds per probe fit and label precision versus `H` |
| Policy fitting | Batches of 56-feature records and the 1,890-parameter scorer; full-graph baseline has additional extraction/model cost | Held-out utility/ranking performance per training second, feature-buffer peak |
| Protected comparison | Independent full prefixes plus every method/task adaptation and committed query prediction | Fixed planned world count, total measured-cost projection, failure-inclusive storage |
| Robustness / external branch | Only the registered discriminating conditions; external feature matrix about 7.1 MB before framework copies | Pilot of the actual multi-output model, replay and preprocessing; not reuse of scalar timings |

For cost sensitivity only, **not a prediction of M3 throughput**, `I=1,000` and `H=4` imply 8,000 probe fits for one policy-training workload. If local calibration measures 0.1, 1 or 10 seconds per probe fit, that term alone is 800, 8,000 or 80,000 seconds, respectively, before any other training. These intervention/horizon numbers are arithmetic examples, not the prescribed sample size. This illustrates why label efficiency and amortization gates precede independent-world replication.

The Function Encoder baseline additionally requires a supported, differentiable regularized linear solve. Test both its forward solution and gradients on rank-deficient/regularized small systems against CPU float64. Use a whole-method CPU path when MPS lacks a faithful kernel; do not silently substitute another coefficient estimator. Its support design matrix and Gram matrix scale as `4*n_s*k` and `4*k*k` bytes in float32, apart from gradients/workspace.

Substitute measured local quantities and report projected device-hours, storage and uncertainty before execution. These are workload estimates, not delivery promises. Do not manufacture absolute M3 throughput in advance.

Scale in the following order: increase measurement precision; improve a demonstrably underoptimized baseline; increase data/curriculum coverage; only then enlarge graphs, template sizes or horizons when evidence indicates a binding expressiveness/credit limit. Increasing model size because an uninformative pilot failed is not justified.

A larger template is warranted if an oracle expanded-size analysis shows useful motifs beyond four nodes and the existing solver can learn their components. A longer credit horizon is warranted if immediate value is weak but fresh-probe delayed value is reproducibly positive. A larger policy is warranted if labels are predictably learnable with wider context or capacity, not because training error alone decreases.

For memory pressure, reduce parallelism and serialize branches before changing scientific parameters. Microbatch examples only with weighted gradient accumulation that implements the same full-batch loss and **one optimizer step**, verified against the unchunked reference. If batch size, precision, topology cap, optimizer schedule or baseline envelope changes during confirmation, record an amendment and rerun affected comparisons under the same recipe; do not silently merge incompatible runs.

### 6.7 Timing and operational protocol

Use `time.perf_counter_ns()` for local elapsed intervals. Warm up until rolling timing summaries stabilize; report the convergence rule and number of warm-up iterations. For MPS, synchronize at both boundaries. Record cold first-task cost separately from warm repeated-task cost, including compilation and library construction. Do not flush only a competitor's caches or prewarm only SDML.

Run timing competitors in randomized paired blocks on the same laptop, with the same power mode and no deliberately unequal background load. Use sequential tasks when concurrency changes thermal or CPU/GPU contention. Repeated timings estimate system variability, not scientific sample size. Include initialization, search, fitting, mandatory registry operations and minimal experiment logging in the declared charged boundary. Report optional verbose audit/report rendering separately, identically for all methods.

Archive system-state indicators and timing outliers. A shared hardware interruption can justify rerunning the entire paired block under a predeclared rule; a method-specific slowdown, compilation cost or memory failure is an outcome. Do not discard it as an “outlier.” Energy claims require actual energy measurements; process time or GPU utilization is not joules.

### 6.8 Checkpointing and recovery

Checkpoint after each completed task and committed library change, and periodically within long training stages at a measured I/O-overhead/lost-work tradeoff. Write to a temporary path, flush, validate hashes, and atomically rename. Store schema and dependency hashes, policies, parameters, optimizer moments/steps, local histories, library versions, candidate sampler, all RNG states, data position, used budgets and experiment amendment ID.

On restart, refuse incompatible schemas or changed method/config hashes unless a deliberate migration is recorded and tested. Restore CPU/Python/NumPy and MPS RNG states where used. If a crash loses the timing telemetry needed to reconstruct consumed work, mark that timing record unusable and rerun the registered paired block from its pre-block checkpoint; do not fabricate the missing cost or quietly grant extra adaptation. Never reset the clock budget, erase rejected actions or regenerate a “luckier” task stream. A partially completed transaction is discarded but its consumed work is retained in the cost ledger. Idempotent event IDs prevent applying a library edit twice.

Distinguish interrupted experiments from algorithmic or resource failures. Resource-failing methods return the last committed valid predictor where available, otherwise the declared support-mean fallback; report the failure flag and rate alongside risk. Do not claim superiority by averaging only their successful cases.

## 7. Risk-ordered work packages and decision gates

The packages below are the execution contract. The agent may make routine non-destructive code/organization choices, but changes to hypotheses, populations, information access, primary metrics, baseline eligibility or resource matching require a recorded methodological amendment. No package may bypass a blocking correctness or evaluation gate.

### WP00 — Environment, provenance and baseline feasibility

**Objective/risk:** establish local prerequisites and prevent a nominally runnable plan from depending on an unavailable kernel, asset or license.  
**Dependencies/entry:** fresh project directory and this document. No numerical result required.  
**Actions/files:** create `pyproject.toml`, lock strategy, `src/sdml/doctor.py`, `provenance/dependencies.json`, `provenance/assets.json` and `artifacts/environment/`; execute Section 6 discovery; test official PyTorch CPU/MPS; resolve Stitch/PySR native installation; attempt the compact DreamCoder build path without modifying global dependencies. Record exact repository commits and license files for reused code.  
**Tests/output:** machine-readable `doctor.json` with architecture, versions, operations, tolerances, memory units and pass/fail reasons; `baseline_manifest.json` labeling faithful/original/ported/composite/reference-only.  
**Resources:** one worker, tiny arrays; measure process baseline RSS and disk installation footprint.  
**Acceptance:** CPU primitive/autograd stack works; core synthetic assets require no download; strong library/numerical baseline path is identified. MPS is optional.  
**Outcomes:** unsupported MPS → CPU; missing wheel → native official source-build attempt; unclear code license → do not vendor it, implement a documented algorithm-level control or exclude with a narrowed claim; no credible close baseline → diagnostics may continue, but broad competitiveness is blocked.

### WP01 — Primitive, graph, template and transaction correctness

**Objective/risk:** falsify semantic, gradient and state-management defects before learned development.  
**Dependencies/entry:** WP00 CPU gate.  
**Actions/files:** implement `runtime/{primitives,graph,execute,templates,transactions,canonical}.py`; explicit parameter/moment identity; flat template expansion; bytes/expanded-node counters. Implement tests in `tests/test_runtime.py`, `test_alias.py`, `test_transactions.py`.  
**Tests/output:** analytic values and finite-difference gradients; randomly generated legal DAGs; alias/inline/copy/tied gradients; cycle and type rejection; unreachable-node cleanup; retired-template use in existing expanded graphs; exact serialization hashes; rollback after induced exception. Save test report and failing seeds.  
**Resources:** tiny CPU float64 batches, then float32/MPS matrix if available; no training corpus.  
**Acceptance:** semantic errors within reference-derived tolerances, no invalid graph accepted, snapshots independent and optimizer identities correct.  
**Outcomes:** incorrect values/gradients → repair, not a hypothesis test; platform-only mismatch → isolate operation and use faithful CPU path; only alias dispatch differs in runtime → record engineering overhead, not abstraction benefit.

### WP02 — Protected generator and evaluation harness

**Objective/risk:** prevent target-program leakage, easy constant-task domination and pseudoreplication.  
**Dependencies/entry:** WP01.  
**Actions/files:** implement `data/{motif_world,splits,task_store}.py`, isolated `evaluator/teacher.py`, `evaluator/runner.py`, `analysis/estimands.py`; generate split manifests and canonical hashes; learner/evaluator process boundary.  
**Tests/output:** teacher graph outside learner namespace; hidden-pair and full-skeleton separation; all motifs covered in prefix; reproducible fresh coefficients; `SpyLearner` blocked; support-mean and identical-method null outputs; deliberately better known predictor detected. Produce `harness_audit.json` with dependency units and leakage attempts.  
**Resources:** stream data by seed; calibrate query Monte Carlo precision rather than storing an unnecessarily large dataset.  
**Acceptance:** equivalent methods produce equivalent outcomes; known differences are detected; no teacher/query access; estimand aggregation matches hand-calculated examples.  
**Outcomes:** leakage or biased exclusions → invalidate and regenerate affected data; collapsed teacher outputs → revise generator on development only and create new manifests; excessive query noise → more query examples, not extra fake independent tasks.

### WP03 — Baseline reproduction and numerical/search headroom

**Objective/risk:** determine whether the task is learnable and whether a reusable library could plausibly help.  
**Dependencies/entry:** WP00–WP02.  
**Actions/files:** implement `methods/{mean,ridge,kernel,mlp,modular,rigl_style,circuit_search,stitch_adapter,pysr_adapter}.py`; reference checks for each; tune baseline pilots; implement teacher-graph and structural-template oracle diagnostics.  
**Tests/output:** analytic kernel/linear cases; numerical convergence on known graph; RigL off-mask gradient toy; Stitch round-trip; PySR export agreement; error/time curves and `headroom_report.md`.  
**Resources:** profile complete fits; start sequential and geometrically extend budgets only until useful baseline progress/plateau can be characterized.  
**Acceptance:** credible baselines are optimized enough to be meaningful; an oracle library shows informative search headroom at some nonsaturated local scale, or a clear diagnosis explains why not.  
**Outcomes:** all methods including known graph fail → optimizer/information issue; oracle graph works but oracle motifs do not help → reconsider search substrate/target claim; simple baseline dominates at all informative budgets → retain as strongest competitor and proceed only for a justified narrower component question; no plausible reuse headroom after faithful diagnostics → stop the structural-efficiency claim for this family, not all possible SDML scales.

### WP04 — Isolated consolidation and causal intervention data

**Objective/risk:** establish whether actual learned circuits have useful future marginal value and whether evaluation is affordable.  
**Dependencies/entry:** WP03 correct solver and measured headroom.  
**Actions/files:** `learning/{interventions,probe_runner,credit_dataset}.py`; gather common bootstrap corpus, candidate groups and paired add/replace/drop/no-op outcomes; implement frequency and exact MDL selectors; split source/probe/policy-validation tasks.  
**Tests/output:** independent mutable state per arm; randomized arm order; fresh probe scoring; preserved failure records; all branch time counted; `credit.jsonl` schema in Section 9; error of repeated value estimates; actual label-generation device-hours.  
**Resources:** serialize pairs; initial horizon four, calibrate noise; choose intervention count from action/state coverage and held-out learning curves.  
**Acceptance:** some source-derived groups produce reproducible positive or negative values on fresh probes, beyond numerical noise; direct-probe selection beats an uninformed choice when headroom exists.  
**Outcomes:** no value despite oracle headroom → source discovery/candidate coverage limitation; noisy value → enlarge independent probe sets or stabilize continuation; probe cost dominates plausible breakeven → keep a diagnostic-only study or pivot to cheaper validated credit, not an uncosted “efficient” model.

### WP05 — Local utility learning and locality challenge

**Objective/risk:** test H2/H3 before integrating every mechanism.  
**Dependencies/entry:** WP04 reliable, appropriately split intervention data.  
**Actions/files:** `learning/{features,local_value,train_value}.py`; exact 56-feature schema and 1,890-parameter network; global-context and constant/frequency/MDL predictors; behavior-policy logging and held-out decision evaluation.  
**Tests/output:** feature-access allowlist; zero target and shuffled-target controls; unseen-context calibration; selected-action regret measured on **fresh** probes; matched-cost selector comparison; `locality_report.md`.  
**Resources:** scorer itself is tiny; record the much larger upstream data cost and learning curve versus intervention count.  
**Acceptance:** local decisions improve fresh outcomes over simple selectors or reveal a precise, useful limit. There is no pass based only on regression training loss or correlation.  
**Outcomes:** global succeeds/local fails → reject locality for that observation contract, test a declared wider-context variant; both fail but direct probes work → improve target estimation or policy representation on development only; shuffled targets match performance → investigate hidden heuristics/caching and do not credit learning; heuristic ties within narrow uncertainty → favor the simpler method unless another measured tradeoff justifies complexity.

### WP06 — Minimum complete SDML loop

**Objective/risk:** test whether learned assembly and consolidation coexist without hidden centralized search or destructive maintenance.  
**Dependencies/entry:** WP05 provides a useful or specifically motivated local-policy signal.  
**Actions/files:** `methods/sdml_local.py`, `learning/develop.py`; implement all actions, shared annealed solver, numerical refinement, on-policy data refresh, exact prefix-library freezing and bounded replacement. Add structural ledgers and per-task checkpoints.  
**Tests/output:** births, rewires, calls and retirements actually occur in legal transactions; all decisions trace to permitted policy inputs and scores; no automatic capacity creep; no test-query updates; checkpoint/resume agreement; complete-lifecycle smoke run.  
**Resources:** measured prefix/label/fit cost model from earlier packages; one independent world at a time.  
**Acceptance:** full loop runs from an empty library without teacher structure labels and preserves every resource/information invariant. Performance is not assumed.  
**Outcomes:** dead policy always chooses no-op → inspect target scale/coverage and compare sensible no-op optimum; unstable restructuring → fix snapshot/acceptance bugs or report genuine forgetting; only a sophisticated external proposer yields useful structures → relabel the tested mechanism or reject the local-discovery interpretation.

### WP07 — Pilot, tuning completion and preregistration

**Objective/risk:** resolve sensitivity, sample size and realistic local workload before protected results.  
**Dependencies/entry:** WP03–WP06, tested complete pipelines.  
**Actions/files:** `configs/exploratory/`, `analysis/{power,timing,frontier}.py`; complete baseline tuning; select support size, `B_*`, template pressure, prefix length, probe horizon, independent-world count and statistical contrasts using Section 5 rules; measure device-hour/storage projection. Create immutable `configs/confirmatory.yaml`, `preregistration.md` and hashes.  
**Tests/output:** no winning hyperparameter on an unexplored boundary; matched information/data and cost review; falsification controls detect known effects; estimated precision and budget table; baseline manifest frozen.  
**Resources:** explicitly measured, potentially substantial; discontinue uninformative combinations rather than escalating everything.  
**Acceptance:** every consequential choice and contrast is specified, hardware capacity verified, and required precision/workload is understood.  
**Outcomes:** inadequate power locally → state attainable resolution and narrow the claim before confirmation; boundary optima → further development tuning; changed central hypothesis → amendment and new protected data; all mechanisms unnecessary → publish diagnostic/negative result without a ritual large run.

### WP08 — Protected controlled comparison

**Objective/risk:** estimate the registered primary effect without adaptive selection.  
**Dependencies/entry:** signed/hash-frozen WP07 manifest; fresh untouched confirmatory worlds.  
**Actions/files:** `experiments/run_confirmatory.py`; sequential paired-world jobs with randomized method order; commit predictions before evaluator reads queries; stop at the predetermined independent-world count; retain failures and cost ledgers.  
**Tests/output:** `run_manifest.json`, per-task predictions, world-level outcomes, timings/memory, Holm-adjusted comparisons, CI tables and learning/resource frontiers. Analysis cannot mutate methods or generator.  
**Resources:** calibrated sum of full prefixes and adaptation work; CPU fallback configuration already frozen where needed.  
**Acceptance:** completed planned comparison with valid provenance, or a clearly labeled incomplete/underpowered result. A positive result is not required.  
**Outcomes:** valid improvement → WP09 attribution and fresh confirmation; no improvement with narrow interval → evidence against the registered useful effect; wide/heterogeneous interval → inconclusive, diagnose without test-set retuning; method-specific failure → report failure and fallback outcome, not deletion.

### WP09 — Attribution, robustness and independent confirmation

**Objective/risk:** attempt to explain any gain without learned structural abstraction.  
**Dependencies/entry:** a result from WP08 worth interpreting, including a useful negative boundary.  
**Actions/files:** run registered ablations and Section 8 adversarial cases; independent teacher-family test; replicate the exact main recipe on a new batch of worlds and fresh training seeds; verify all generic optimizations were shared.  
**Tests/output:** causal no-consolidation/heuristic/global/multi-node-mutation comparisons; inline semantic null; structure/copy/tied separation; deeper compositions; fresh confirmation report; negative controls.  
**Resources:** choose robustness cases that discriminate live explanations; do not run every combinatorial configuration automatically.  
**Acceptance:** attribution supported on fresh data, or the conclusion is explicitly narrowed to the actual explanation.  
**Outcomes:** only copied parameters matter → parameter-transfer result; generic multi-node moves explain gain → search-operator result; only one generator works → synthetic family-specific result; optimizer sharing erases gain → engineering correction, not a surviving architectural advantage.

### WP10 — Drift and external usefulness

**Objective/risk:** test bounded maintenance under shift and whether a useful component transfers beyond its favorable synthetic generator.  
**Dependencies/entry:** correct E3 loop and reason to study H5; real-data branch additionally requires the licensing/data gate.  
**Actions/files:** `data/drift.py`, `experiments/run_stream.py`, optional `data/uci_gas.py`; execute synthetic prequential shift, then Section 5.9 real-data protocol if warranted. Fit preprocessing only to revealed data; keep replay/capacity equal.  
**Tests/output:** per-step risk, recovery time/area, old-task forgetting, template turnover, failure rate; per-batch external balanced accuracy/log loss; explicit “case study” label.  
**Resources:** small asset and modest model; local throughput still calibrated because 128 input dimensions and a multiclass head change the workload.  
**Acceptance:** interpretable adaptation comparison; no requirement that every workload improve.  
**Outcomes:** shift has no observable precursor → do not demand impossible anticipation; retirement hurts retention → report stability/plasticity tradeoff; external failure → withhold application claim; license uncertainty incompatible with use → omit raw asset and stop that branch.

### WP11 — Reproducible release and decision

**Objective/risk:** make the conclusion survive a fresh process and unfavorable-result audit.  
**Dependencies/entry:** completed or explicitly terminated prior packages with recorded reasons.  
**Actions/files:** `analysis/report.py`, `REPORT.md`, `DECISIONS.md`, `limitations.md`, release manifest; regenerate every table from raw immutable outputs; audit bibliography/source versions, data access, labels and costs; run one clean-environment smoke reproduction.  
**Tests/output:** hash-consistent report; no missing failed runs; no chat-dependent instructions; provenance includes licenses and exact commit/lock files; reproduction instructions distinguish small smoke from full confirmation.  
**Resources:** report generation plus a calibrated smoke run, no hidden retraining.  
**Acceptance:** a third party can reconstruct methods, uncertainty and decision from artifacts.  
**Outcomes:** missing state/cost data → result not reproducible and claims withheld; successful narrow effect → continue only on its supported scope; no advantage → release the negative result and prefer the simpler method.

## 8. Attribution, robustness and adversarial failure analysis

Run the essential attribution controls with the primary recipe. Other cases are conditional tests of a live alternative explanation, not a requirement to exhaust every combination. Register which tests are confirmatory before viewing their protected outcomes; label the rest exploratory and obtain fresh confirmation for a revised claim.

| Challenge | Controlled intervention and observation | Interpretation / required action |
|---|---|---|
| Naming or compiler caching masquerades as abstraction | Replace every macro by its exact inlined body, retaining parameter identities; compare values, gradients and expanded execution counts. Separately disable macros as future search actions. | Output changes indicate a bug. Learning changes under action removal but not inlining support a search/organization effect. Dispatch-only improvements are engineering. |
| Extra capacity | Match expanded nodes, instantiated parameter slots and persistent bytes; add equally sized random templates to the control. | A capacity-matched effect is required for structural attribution. Do not match only the number of macro names. |
| Larger search moves | Allow generic multi-node mutations with the same size distribution as macro calls, no learned template reuse. | If these explain the gain, report a search-operator result, not acquired abstraction. |
| Hidden coefficient transfer | Reinitialize all future call parameters; compare structure-only, copied, frozen-shared and tied-trainable modes. | Structure-only is the primary claim. A copy-only benefit is numerical transfer. |
| Frequency is sufficient | Compare frequency, LRU and MDL; permute utility labels within frequency/size strata. | A learned-policy benefit must exceed well-tuned simple selection. A shuffled-label win demands leakage and noise investigation. |
| Unobserved future | Use identical source histories followed by equally likely incompatible future motif demands. | Expected anticipation above chance without an available signal indicates leakage. Do not penalize a legitimate inability to predict this shift. |
| Local observations are insufficient | Compare the two B7 global variants using the same labels and action pools; permute node IDs and task identifiers. | Global-only success narrows or rejects H2, not necessarily useful structural organization. |
| Counterfactual winner's curse | Select using one probe set and estimate actual value on another; increase probes based on measured label noise. | Reusing the selection maximum as its own evaluation overstates utility. |
| Interactions between templates | Compare separately estimated additions with joint add/replace actions under tight capacity; measure whether pairs help only together. | Myopic utility can miss complementarity. A longer/joint-action experiment requires evidence and a new declared action space, not retroactive relabeling. |
| Saturation/vanishing signal | Stratify teachers by output variance, saturation and path-gradient magnitude; fit the known teacher graph. | If the oracle cannot fit, inspect numerical conditioning/identifiability before judging structure learning. Keep hard strata visible. |
| Generator vocabulary advantage | Independent modular teacher; withheld primitive; additional input mixing with transforms disclosed to all methods only when part of public data. | Failure on an unrepresentable teacher is a scope limit. Generator-specific success is not general adaptation. |
| Spurious semantic novelty | Detect exact skeleton duplicates and inspect function similarity on independent evaluator probes. | Syntactic novelty alone does not prove a new mathematical function; report both notions. |
| Retirement destroys prior competence | After replacing templates, adapt to old-task supports under the same budget; compare frozen-library and equal-replay controls. | Quantify forgetting versus recovery. Lower recent risk with severe retention loss is a tradeoff. |
| Mutable-state contamination | Run each paired arm in isolated serialized state; reverse arm order; compare restored snapshots and RNG counters. | Any order-dependent state reuse invalidates intervention estimates until repaired. |
| Timing/caching advantage | Randomized paired timing blocks; cold and warm regimes; all compilation/probe costs charged; shared generic optimizations. | Reclassify a vanished effect rather than defending a biased timing boundary. |
| Sparse-work accounting error | Compare masked dense RigL-style runtime and actual allocated matrices, not just mask density. | Sparsity is not measured speed; count candidate gradients and dense storage. |
| Failure selection | Include every planned task/world; record fallback predictions, OOM, NaN, crash and timeout outcomes. | Successful-run-only analysis is invalid. Report failure risk with accuracy. |
| Tiny or unstable gain | Inspect paired intervals, per-world effects, stratum regressions and operational resolution; repeat the frozen recipe on fresh worlds. | Distinguish real but negligible benefit, underpowered uncertainty, heterogeneity and reproducible useful advantage. |

### 8.1 Domain-specific correctness tests

Write small deterministic tests before any long run:

1. A two-input template with repeated boundary arguments must preserve aliasing; two distinct calls must have independent parameters in structure-only mode. Tied mode must sum gradients exactly as the equivalent expanded shared graph, within floating-point tolerance.
2. Replacing an `A` by `P` removes three parameter slots and their moments; replacing `P` by `A` creates exactly three. Reordering `A` inputs without swapping its first two raw parameters must be detected as nonequivalent.
3. A candidate subgraph with two externally consumed outputs is not a legal single-output macro. A seemingly small nested future macro must be checked by expanded dependency closure, not surface size.
4. Retirement frees only actually unreferenced storage. Keeping a full archive accessible to the learner is replay and counts toward memory. Administrative logs are not a free hidden model.
5. Graph hashes are stable under nonsemantic UID renaming, but not under parameter-sharing changes. Hash collisions must be checked by structural equality before merging.
6. Identical null methods run through different method names must yield the same deterministic CPU outcomes and centered timing differences after paired randomization. Inject a known corrupted predictor to ensure the risk evaluator detects it.
7. A query-access spy must fail before fitting. A leaked-label diagnostic, executed only in a separate harness-test mode, should perform implausibly well and be visibly marked invalid; this validates that the harness recognizes its own positive leakage control.
8. Kill execution between prepare and commit; resume once; confirm the same library, optimizer steps, output and consumed-work ledger as the uninterrupted deterministic reference. Reject a checkpoint with a changed feature schema or library hash.

### 8.2 When to increase scale

An uninformative smallest case does not settle a scale-dependent hypothesis. Escalation is justified by a positive oracle headroom result, reliably measurable delayed utility, a clear coverage/representation bottleneck, or a trend on development scales with uncertainty small enough to distinguish it from noise. Enlarge the specific bottleneck and keep all baselines competitive at that scale.

Conversely, repeated correct experiments in which oracle templates help but outcome-trained local selection does not improve on MDL are evidence against the selected local mechanism. “It may work when enormous” is not an unlimited exemption from that evidence. Record the range tested and the precise untested prediction needed to justify another scale.

## 9. Repository, interfaces, configuration and reproducibility contract

### 9.1 Minimal repository

The following is a proposed layout. Small adjacent modules may be combined without changing their interface or experimental meaning.

```text
sdml-local/
  research_poc_plan.md
  README.md
  pyproject.toml
  requirements.lock                  # exact chosen dependency solution
  LICENSE                            # license for newly authored project code
  THIRD_PARTY_NOTICES.md
  preregistration.md
  DECISIONS.md
  STATUS.json
  src/sdml/
    __init__.py
    __main__.py
    config.py
    doctor.py
    types.py
    primitives.py
    graph.py
    canonical.py
    transactions.py
    executor.py
    optimizer_state.py
    observations.py
    candidates.py
    policy.py
    credit.py
    solver.py
    library.py
    budget.py
    checkpoint.py
    methods/
      base.py
      sdml.py
      simple.py
      mlp_reptile.py
      modular.py
      rigl_style.py
      heuristic_library.py
      stitch_adapter.py
      pysr_adapter.py
      global_policy.py
      function_encoder.py
    data/
      public_schema.py
      support_stream.py
      drift.py
      uci_gas.py
  evaluator/                         # not importable in learner process
    teacher.py
    worlds.py
    protected_store.py
    metrics.py
    score.py
  experiments/
    calibrate.py
    build_credit.py
    run_pilot.py
    run_confirmatory.py
    run_stream.py
  analysis/
    power.py
    timing.py
    frontier.py
    report.py
  tests/
    test_primitives.py
    test_macros.py
    test_transactions.py
    test_optimizer_state.py
    test_locality.py
    test_splits.py
    test_no_leakage.py
    test_baselines.py
    test_budget.py
    test_resume.py
  configs/
    smoke.yaml
    exploratory/
    confirmatory.yaml
  provenance/
    dependencies.json
    assets.json
    baseline_manifest.json
    search_log.md
  artifacts/
    environment/
    calibration/
    pilot/
    confirmatory/
    reports/
  data_public/                       # supports accessible to learners
  data_protected/                    # evaluator-only, never in model inputs
```

Files shown as `data/...`, `analysis/...` or a bare module name in work-package descriptions resolve to the corresponding package/module in this layout; they are not additional external dependencies.

The directory name alone is not a security boundary. Run learners with only the allowed support/model directory mounted or copied into a restricted working directory, and keep the evaluator in a separate process that commits and scores predictions. Do not import the teacher generator in a learner and rely on programmer self-restraint. Do not place protected target values in exception messages or debugging logs readable by the learner.

### 9.2 Shared method interface

These are project contracts, not claims about a third-party package API:

```python
from dataclasses import dataclass
from typing import Any, Protocol
import numpy as np

@dataclass(frozen=True)
class SupportTask:
    opaque_id: str
    x: np.ndarray       # shape [n_support, d], finite, float32
    y: np.ndarray       # shape [n_support, output_dim], finite
    split_seed: int     # no motif, teacher, pressure or test-label information

@dataclass(frozen=True)
class ResourceBudget:
    wall_seconds: float
    persistent_bytes: int
    peak_working_bytes: int
    support_labels: int
    expanded_nodes: int | None  # applies where graph work is meaningful

class FittedPredictor(Protocol):
    def predict(self, x: np.ndarray) -> np.ndarray: ...
    # Deterministic with fixed state; no updates or access to y/query labels.

class Method(Protocol):
    def train_prefix(self, tasks, budget: ResourceBudget, rng) -> Any: ...
    def adapt(self, prefix_state: Any, support: SupportTask,
              budget: ResourceBudget, rng) -> tuple[FittedPredictor, dict]: ...
    def maintain(self, prefix_state: Any, revealed_support: SupportTask,
                 fit_record: dict, budget: ResourceBudget, rng) -> Any: ...
    def save_state(self, path: str) -> None: ...
    def load_state(self, path: str) -> None: ...
```

**Training-label access:** all methods receive the same finite prefix observations, including outcome/probe examples that SDML uses to learn utility. The prefix manifest records their counts, roles and reveal times. SDML probe arms must commit predictions before the experimenter scores those outcome examples; other baselines may use the same revealed labels for their own training thereafter. Protected evaluation queries never enter this manifest. Equalizing only source-task supports while giving SDML additional labeled probes would be an unfair comparison.

`train_prefix` receives a finite manifest-backed stream or documented replayable view of allowed source data, never the evaluator object. A method may use fewer labels or less compute, but cannot request more without the harness recording it. The prefix budget and per-task budget are separate ledgers. `maintain` is forbidden in the independent held-out E3 task loop; it is enabled only in the declared training and E4 stream phases.

`adapt` returns a ready predictor and a record with fitting status, valid-incumbent timestamp, failures, model/graph hashes and costs. The evaluator supplies query inputs after that state is committed. `predict` may batch inference, but cannot train on query inputs, fit batch statistics, adapt the graph, or read other query predictions unless a separate transductive condition was registered for every method. Stable task random streams are derived from `(world, task, phase, semantic purpose, counter)`; do not derive choices from Python's process-randomized `hash()`.

### 9.3 Configuration conventions

Start from an exploratory configuration like this. The values explicitly labeled calibration outputs are deliberately unresolved until their stated early gates; a confirmatory validator **must reject** nulls, exploratory flags or missing hashes. This is a hardware/data-dependent procedure, not permission to choose values after test access.

```yaml
schema_version: 1
study: sdml_local_structure_only
mode: exploratory
seed_namespace: sdml-v1
precision: float32
backend: auto_calibrate
teacher:
  input_dim: 4
  motif_count: 8
  motif_nodes: [2, 4]
  train_depth: [1, 3]
  test_recombination_depth: [2, 3]
  extrapolation_depth: [4, 6]
  initial_support: 64
  initial_query: 256
  primary_noise: 0
  coefficient_sd: 0.7
runtime:
  initial_expanded_nodes: 64
  initial_candidate_pool: 16
  template_arity_max: 2
  template_nodes_max: 4
  pressure_slots: [4, 16]
  template_mode: structure_only
  nested_templates: false
  allow_prediction_updates: false
solver:
  initial_fit_fraction: 0.75
  initial_refinement_steps: 8
  learning_rate: null             # WP03 development tuning
  temperature: calibrated_median_delta
policy:
  features: local56-v1
  hidden_width: 32
  heads: 2
  initial_probe_horizon: 4
  initial_behavior_uniform_fraction: 0.5
  target: paired_risk_difference
  test_weights_frozen: true
primary:
  outcome: clipped_mse_divided_by_4
  population: protected_same_depth_recombinations
  strata_weighting: equal_pressure
  unit: independent_world_with_full_prefix_training
  adaptation_seconds: null       # WP07 baseline progress rule
  independent_worlds: null       # WP07 precision/power calculation
  contrasts: null                # frozen eligible baseline basket
  alpha_familywise: 0.05
  multiplicity: holm
  target_power: 0.90
resources:
  working_bytes: null            # WP00/07 measured hardware envelope
  persistent_bytes: null        # complete-method Pareto calibration
  prefix_seconds: null           # WP07 matched-resource views
provenance:
  generator_commit: null
  dependency_lock_sha256: null
  baseline_manifest_sha256: null
  preregistration_sha256: null
```

Use typed validation for dimensional consistency, positive budgets, valid action/feature versions, unique split IDs, complete source pins, appropriate study phase and exact pressure weights. When support size changes from 64, apply the frozen fitting fraction with at least one selection observation; choose an explicit minimum useful split during development. Store both requested and resolved configurations. No implicit device, dtype, optimizer or seed default may change between methods without disclosure.

### 9.4 Commands and execution order

First implement the CLI dispatcher and the work-package modules; these commands do **not** exist before the coding agent writes them. They define the desired executable entry points:

```bash
python -m sdml doctor --output artifacts/environment/doctor.json
python -m pytest -q tests/test_primitives.py tests/test_macros.py tests/test_transactions.py
python -m sdml generate --config configs/smoke.yaml --split development
python -m sdml calibrate --config configs/smoke.yaml --output artifacts/calibration
python -m sdml baselines --config configs/exploratory/baselines.yaml
python -m sdml credit --config configs/exploratory/credit.yaml
python -m sdml pilot --config configs/exploratory/full_loop.yaml
python -m sdml freeze --pilot artifacts/pilot --output configs/confirmatory.yaml
python -m sdml validate --config configs/confirmatory.yaml --require-confirmatory
python -m sdml confirm --config configs/confirmatory.yaml
python -m sdml resume --run-dir artifacts/confirmatory/RUN_ID
python -m sdml report --run-dir artifacts/confirmatory/RUN_ID
```

`freeze` prepares a proposal with measured sample-size/resource calculations; it must not select a favorable test result or automatically bypass a failed gate. `confirm` refuses to run without completed prerequisite manifests and an immutable registration. `resume` verifies hashes before loading. `report` reads raw results and produces tables, figures and a decision without retraining. The actual existing commands in Section 6 are distinct from this proposed CLI.

The agent may choose a normal environment manager and adjust source-file grouping. It must not silently download large training assets, enable a remote API, change the primary population, remove a strong baseline, redistribute an uncertain-license repository, or replace a required primitive with an approximation.

### 9.5 Artifact schemas and raw records

Use append-only JSONL for events and a tabular format such as CSV/Parquet for outcomes. Units belong in field names. Include schema versions and fail validation on nonfinite numeric fields unless the failure schema explicitly allows a missing value.

**Run manifest:** `run_id`, `study_phase`, UTC start/end, `world_ids`, planned/completed counts, method/config/source/lock hashes, machine ID (nonpersonal), backend, precision, thread count, power state, baseline eligibility, hypothesis IDs, amendment IDs, status and termination reason.

**Task outcome row:** `run_id`, `world_id`, opaque `task_id`, method, seed lineage, evaluation stratum (evaluator-only until scoring), support count, query count, normalized risk, unclipped MSE, fit seconds, prediction seconds, initialization seconds, maximum valid-incumbent elapsed seconds, overrun seconds, prefix-allocation seconds, peak RSS bytes, peak device bytes, persistent bytes, expanded nodes, actual primitive invocations, active/allocated parameters, status and fallback flag. Do not add CPU RSS to overlapping unified-memory allocations and label the sum physical memory; retain separately measured views.

**Structural event:** `event_id`, parent-state hash, candidate-pool hash, action payload, legal/rejected reason, before/after graph/library hashes, feature schema and vector, behavior probability, predicted value, observed value when available, both paired raw outcomes, probe IDs, resource deltas, optimizer remapping summary, accepted flag, policy hash and transaction commit state. The observed future target is absent at protected execution. It can be joined into analyst records afterward, never fed back into the frozen method.

**Cost ledger:** category (`bootstrap`, `assembly_probe`, `library_probe_control`, `library_probe_treatment`, `policy_train`, `prefix_develop`, `adapt`, `predict`, `compile`, `preprocess`, `checkpoint`, `optional_report`), elapsed seconds, CPU/device memory view, input/output artifact hashes and whether already included in a parent timing boundary. Parent/child boundaries are not double-counted. Costs of failed trials stay present.

**Decision entry:** date, work package, question, evidence artifact, decision (`continue`, `revise`, `stop`, `inconclusive`), reason, scope affected, next permitted package and amendment requirement. Keep `STATUS.json` pointed at the last committed artifact and next action so execution does not require chat memory.

### 9.6 Reporting requirements

Every report starts with the registered claim and its outcome, not the most flattering graph. Include all eligible baselines, uncertainty, individual worlds, failure rates, both pressure strata, total training cost, cold/warm adaptation, memory and unchanged-versus-adapted protocol details. Put raw negative results in the same tables as positive ones.

For primary inference, report individual 95% intervals with that label, Holm-adjusted p-values and conservative simultaneous Bonferroni intervals for the registered family. Do not imply that unadjusted intervals have family-wise coverage. Report an effect estimate and interval even when the corresponding test is not significant.

Provide representative learned circuits only after quantitative results, with their source and distinct reuse contexts, removal effect, parameter mode and expanded cost. A visually attractive graph is not causal evidence. State whether a macro represents a teacher motif, a different useful factorization or merely a frequent pattern; evaluator-only matching is analysis, not training supervision.

Separate exploratory findings from protected confirmation. Retain a complete run index, including aborted/failed runs, and explain every exclusion. Commit the generator, test harness, configurations, exact dependencies and source revisions. Release data only under established conditions; otherwise supply permitted retrieval instructions and hashes. Do not claim bitwise reproducibility across operating systems or PyTorch releases; define deterministic same-environment tests and tolerance-based cross-device checks [R29].

## 10. Outcome interpretation and evidence-triggered alternatives

### 10.1 Decision table

| Observed result after valid measurement | Supported conclusion | Next action |
|---|---|---|
| Numerical, leakage, rollback or timing test fails | No scientific conclusion about SDML. | Repair the implementation/harness and rerun affected tests before further comparison. |
| Teacher graph cannot fit, or informative support is absent | Representation/optimization/identifiability limits obscure the mechanism. | Diagnose with noiseless/simple cases and an explicit support/capacity intervention; keep the original failed regime recorded. |
| Oracle templates do not help on nonsaturated learnable tasks | No reuse headroom in this setting. | Prefer the simpler baseline; test a different prejustified recurrence/complexity regime only with a declared change of population. |
| Oracle/global/direct-probe choices help, local policy does not | Local observations, training distribution or policy capacity are inadequate. | Separate estimation error from insufficient information; a global-policy pivot relinquishes H2 explicitly. |
| Local policy beats uniform but not MDL/Stitch | Learned structure can work, but no advantage over simpler existing abstraction selection is established. | Use the simpler method and report the boundary; investigate only a specifically evidenced failure mode. |
| Structure-only reuse helps but full SDML loses to Function Encoder, kernel or modular meta-learning | A component effect, not best overall numerical adaptation. | Report the component result without claiming a superior general architecture. |
| Only copied/tied weights help | Parameter transfer or shared-parameter regularization explains the result. | Reclassify the contribution; compare directly with the corresponding conventional transfer learner. |
| Warm adaptation improves but training cost is not repaid | An amortized deployment advantage with an unproven or unfavorable total cost. | Report measured breakeven and usage assumptions; no unqualified efficiency claim. |
| Primary metric improves but memory, retention or failure risk regresses | A Pareto tradeoff, not dominance. | State the regression, measure its scope and evaluate an explicit constrained use case. |
| Frozen recipe wins on fresh worlds, survives attribution, and beats the locally executable basket within declared costs | Reproducible benefit of this selected mechanism and scope. | Extend to independent teachers, drift and larger components; distinguish synthesis/empirical contribution from genuinely new methodology. |
| Synthetic gain does not transfer externally | The real-world application claim is unverified or contradicted by that case study. | Preserve the valid synthetic conclusion; do not market deployment readiness. |
| Wide uncertainty or infeasible replication cost | Inconclusive at the stated resolution. | Report attainable resolution and the exact additional information needed; do not call it either a breakthrough or a falsification. |

### 10.2 Alternatives that preserve purpose, with explicit changes

**Global structural credit:** justified when a stronger global selector consistently predicts future utility that the bounded local features cannot predict. Keep the same tiny numerical units and legal actions; state that the controller is centralized. Compare its extra training/inference cost. This is not a local learning principle.

**Prospective library learning without neural growth:** justified when consolidation is helpful but learned rewiring adds no value beyond a strong common solver. Retain reusable computational development but publish it as a library-selection contribution, with DreamCoder/Stitch and prospective/counterfactual prior art central to the comparison.

**Parameterized abstraction or coefficient sharing:** justified when exact structural templates are useful only after coefficient transfer. Separate the structural and statistical priors; use modular meta-learning and Function Encoder as primary comparators rather than continuing to claim structure-only advantage.

**Hierarchical/long-horizon development:** justified only when flat-template oracle tests and delayed-value measurements identify a specific missing scale. Add immutable dependency DAGs, explicit closure accounting, longer probe horizons and a fresh registration. Do not infer open-ended development from a two-level example.

**Numerical execution optimization:** justified by measured dispatch or repeated arithmetic bottlenecks. Pursue compilation/fusion with exact output/gradient tests and identical access for all methods. A learned approximate primitive additionally requires error bounds/distributional validation, out-of-distribution testing and a separate accuracy/latency tradeoff. GrowLibm is relevant prior art [R21].

Self-repair remains a later research question. A legitimate test would compare targeted/random pathway damage against equal-retraining, redundancy and restored-checkpoint controls at equal repair cost, withholding the original graph from methods not entitled to it. Restoring a stored checkpoint is not evidence for emergent repair. This extension is not a prerequisite for the present PoC.

## 11. Audit record, revisions and residual risks

The following audits were performed while drafting this plan. They are the author's adversarial review, not independent peer review, a reproduced benchmark, or executed validation on the target machine.

### 11.1 Scientific and prior-art audit

**Defect found:** future-oriented consolidation and counterfactual downstream skill credit initially looked like potential differentiators. Inspected 2026 prospective-compression, SkillMaster and numerical-library-learning work already contains these concepts; older GP/library/modular work covers the larger architecture family. **Revision:** removed standalone novelty claims and selected a bounded locality/economics question. **Residual:** the exact combination may also exist; this search cannot certify absence or legal novelty.

**Defect found:** an exact macro alias could be described as a cheaper computational unit. **Revision:** require expanded primitive accounting, structure-only mode and explicit search-versus-execution claims. Added exact inlining and multi-node-mutation controls. **Residual:** a useful inductive bias can remain difficult to distinguish from favorable search geometry; report both rather than assign an inflated abstraction label.

**Defect found:** a fixed-MLP-only comparison would miss strong low-cost transfer methods; pooling different task labels would understate meta-learning. **Revision:** added Reptile and Function Encoder, retained BounceGrad/MOMA, kernels and independent symbolic search, and revised budget selection for one-solve baselines. **Residual:** a tuned baseline may remove all available headroom, which is an acceptable outcome.

**Defect found:** local features do not imply biologically local credit or an intelligent-free external training process. **Revision:** state global outcome/gradient teaching and centralized experimental evaluation explicitly. Restrict only policy observations and deploy-time arbitration. **Residual:** locality is a defined information constraint, not a claim about biological neurons.

### 11.2 Experimental-validity audit

**Defect found:** teacher AST exposure, program-identity metadata or accidental structural duplicates could manufacture transfer. **Revision:** evaluator-only teachers, opaque task IDs, hash-defined protected pair splits, exact skeleton checks and a query-access spy. **Residual:** semantic equivalence is not decided completely by finite probes; report that limitation.

**Defect found:** source-circuit coefficients, extra gradient steps or stronger macro mutations could explain apparent reuse. **Revision:** fresh call parameters, equal no-op refinement and a size-matched multi-node control. **Residual:** wall-time matching and work matching answer different questions; both are retained rather than conflated.

**Defect found:** evaluating the same probes used to choose a library overestimates utility; many task/query observations could masquerade as independent replications. **Revision:** fresh selection/evaluation folds, paired complete-world units, fixed-sample planning, multiplicity correction and a separate confirmation batch. **Residual:** expensive independent prefixes can limit precision; label resource-limited intervals honestly.

**Defect found:** a memory-constrained library can exhibit complementary or destructive interactions not captured by isolated candidate scores. **Revision:** evaluate atomic replacement, test joint effects and keep the marginal versus closed-loop targets distinct. **Residual:** a short horizon cannot establish long-term usefulness; no such claim is made.

**Defect found:** a high-depth balanced teacher may exceed a 64-node learner's representational cap. **Revision:** added an explicit bounded branching generator and expanded-size feasibility rule before training. **Residual:** these constraints define a limited synthetic population, not the space of all compositions.

### 11.3 Implementation and feasibility audit

**Defect found:** nominal Apple support does not establish correctness of the actual kernels, gradients or native dependencies. **Revision:** CPU-first eager DAGs, operation-by-operation doctor, whole-method CPU fallback, no silent MPS fallback, no assumed maximum RAM. **Residual:** target throughput, compatible dependency pins and native Stitch/PySR builds still require WP00 execution.

**Defect found:** a naive RigL port would regrow from zero masked gradients; mutable graph changes can lose optimizer moments; parameter-free graphs can break backward/optimizer creation. **Revision:** specify retained effective-weight gradients, stable parameter-key state remapping, transactional rollback and parameter-free conventions. **Residual:** all must pass the proposed tests; only the small alias test below has been executed.

**Defect found:** unlicensed public source code and conflicting dataset terms were being treated as routine downloadable assets. **Revision:** explicit license gates for uncertain repositories and conservative research-only handling of the UCI conflict; no raw-data redistribution. **Residual:** a closest implementation may be unavailable under suitable terms, narrowing the comparison rather than justifying unauthorized copying.

**Defect found:** restoring only model weights could repeat edits or reset consumed budgets. **Revision:** atomic event IDs, complete optimizer/RNG/library/history serialization, source/config hash checks and kill-resume equivalence tests. **Residual:** cross-device nondeterminism requires tolerance-based rather than promised bitwise reproduction.

**Defect found in an executed check:** a bitwise gradient equality assertion was stricter than mathematically equivalent floating-point accumulation justified. **Revision:** use recorded maximum errors and a tight float64 tolerance rather than claiming bitwise identity. This was a test-criterion correction, not evidence for SDML performance.

A subsequent cross-check of work-package dependencies, baseline inclusion, generator limits, inference units and artifact interfaces produced the consolidated plan here. No claim of perfection follows: the remaining blocking prerequisites are local environment/license checks, baseline reproduction, oracle headroom, utility learnability and calibrated confirmation cost.

## 12. Targeted checks actually executed for this research

### 12.1 Environment and results

Executed on **Linux 6.18.44 x86_64**, Python **3.13.5**, PyTorch **2.10.0+cpu**, NumPy **2.3.5**, CPU, one PyTorch thread, seed **731**, float64. This is not the user's M3 Max and does not validate MPS, macOS, native-package builds or architecture training.

| Check actually run | Result | What it does and does not establish |
|---|---|---|
| Macro versus inline forward/gradient comparison | 100 random trials, batch 64; maximum output difference 0; input-gradient difference `3.469446951953614e-18`; shared-parameter-gradient difference 0. | Validates this small expression's aliasing semantics, not a complete graph runtime or learned consolidation. |
| Expanded work in that expression | Five high-level primitive nodes in both forms. | A function name does not remove its underlying computation. Not a timing benchmark or a universal FLOP count. |
| Finite-support ambiguity witness | `q(x)=x(x+1)(x-1)` is zero at `[-1,0,1]` and `q(0.5)=-0.375`. | Illustrates nonidentification from finite supports in a broader function class; does not prove ambiguity for every bounded SDML instance. |
| Indistinguishable-past witness | For 101 action probabilities, expected success is exactly 0.5 with equally likely future A/B and one available choice. | Confirms elementary arithmetic of the counterexample, not a learned policy's result. |
| Memory arithmetic | Tensor/state byte calculations used in Section 6. | Estimates only; no peak RSS or M3 memory measurement. |

The executed alias expression used an ordinary affine-tanh operator to isolate parameter-sharing behavior; it is not a trained implementation of the final bounded-parameter primitive. The implementation agent must run the analogous checks on `A`, `P`, `M` and every transaction in the selected runtime.

No SDML training, comparative benchmark, real-data fit, full dependency installation or M3 profiling was executed during this assignment. Public code/README inspection is not an installation test. Attempts to obtain immutable repository snapshots through a separate network path failed; therefore this document identifies inspected branch/source URLs, not invented commit hashes. WP00 must record actual checked-out revisions before execution.

### 12.2 Self-contained reproducer for the semantic checks

The following reproduces the essential executed checks without requiring an intermediate file from this conversation. Save it as `tests/research_semantics.py` in an environment with NumPy and PyTorch. It is a small sanity script, not the project implementation.

```python
import platform
import sys
import numpy as np
import torch

torch.set_num_threads(1)
torch.manual_seed(731)

def macro(a, b, p):
    return torch.tanh(p[0] * a + p[1] * b + p[2]) * b

def inline(x, p):
    u = torch.tanh(p[0] * x[:, 0] + p[1] * x[:, 1] + p[2])
    v = torch.tanh(p[0] * x[:, 1] + p[1] * x[:, 2] + p[2])
    return torch.tanh(u * x[:, 1] + v * x[:, 2])

def aliased(x, p):
    return torch.tanh(macro(x[:, 0], x[:, 1], p)
                      + macro(x[:, 1], x[:, 2], p))

max_y = max_dx = max_dp = 0.0
for _ in range(100):
    x = (2 * torch.rand(64, 3, dtype=torch.float64) - 1).requires_grad_()
    p = torch.randn(3, dtype=torch.float64, requires_grad=True)
    y0 = inline(x, p)
    g0 = torch.autograd.grad(y0.square().mean(), (x, p))
    y1 = aliased(x, p)
    g1 = torch.autograd.grad(y1.square().mean(), (x, p))
    max_y = max(max_y, (y0 - y1).abs().max().item())
    max_dx = max(max_dx, (g0[0] - g1[0]).abs().max().item())
    max_dp = max(max_dp, (g0[1] - g1[1]).abs().max().item())
assert max(max_y, max_dx, max_dp) < 1e-12

support = np.array([-1.0, 0.0, 1.0])
def q(x):
    return np.prod(np.asarray(x)[..., None] - support, axis=-1)
assert np.max(np.abs(q(support))) == 0.0
assert float(q(0.5)) == -0.375
for probability_a in np.linspace(0, 1, 101):
    assert 0.5 * probability_a + 0.5 * (1 - probability_a) == 0.5

print({"platform": platform.platform(), "python": sys.version.split()[0],
       "torch": torch.__version__, "numpy": np.__version__,
       "max_output_error": max_y, "max_input_gradient_error": max_dx,
       "max_shared_parameter_gradient_error": max_dp})
```

## 13. References and inspected-evidence register

References identify the source actually used, not a claim that its results were independently reproduced. “HTML inspected” means the relevant methods/limitations were read in an HTML rendering; “repository inspected” means the specified source or README was examined, not installed. Access date for all web material: **14 October 2026**. Preprints are identified as such; later publication status must not be inferred. Code licenses apply to the checked version, not automatically to every linked dataset or dependency.

### R01 — Growing fixed-function networks

Scott E. Fahlman and Christian Lebiere. *The Cascade-Correlation Learning Architecture*. NeurIPS 1989 proceedings. [Official proceedings record](https://papers.nips.cc/paper_files/paper/1989/hash/69adc1e107f7f7d035d7baf04342e1ca-Abstract.html). Inspected the proceedings abstract. Establishes neural growth as longstanding prior art; not evidence for this cross-task library rule.

### R02 — Reusable pathways

Chrisantha Fernando et al. *PathNet: Evolution Channels Gradient Descent in Super Neural Networks*. 2017. [Author/Google Research record](https://research.google/pubs/pathnet-evolution-channels-gradient-descent-in-super-neural-networks/). Inspected the research description. Relevant to pathway discovery and transfer; no local implementation or controlled SDML comparison was executed.

### R03 — Automatically defined reusable functions

John R. Koza. *Genetic Programming II: Automatic Discovery of Reusable Programs*. 1994. [Author's 1994 bibliography](https://www.genetic-programming.com/jkpubs94.html). Inspected the author bibliography and descriptions, not the complete book. Historical prior art for evolving reusable computational subroutines; not a specification of the chosen optimizer.

### R04 — Evolved Cartesian-GP modules

James A. Walker and Julian F. Miller. *The Automatic Acquisition, Evolution and Reuse of Modules in Cartesian Genetic Programming*. IEEE Transactions on Evolutionary Computation 12(4), 397–417, 2008. [DOI: 10.1109/TEVC.2007.903549](https://doi.org/10.1109/TEVC.2007.903549); [bibliographic/abstract record](https://gpbib.cs.ucl.ac.uk/gp-html/Walker_2008_TEC.html). Publisher access was limited; the bibliographic abstract was available. Accordingly, this plan does not claim a faithful ECGP replication or detailed theorem inspection. The broader novelty conclusion also rests on fully inspected library/modular papers below.

### R05 — DreamCoder

Kevin Ellis et al. *DreamCoder: Growing generalizable, interpretable knowledge with wake-sleep Bayesian program learning*. Preprint 2020, subsequently revised; published program-learning work. [arXiv:2006.08381](https://arxiv.org/abs/2006.08381); [official implementation](https://github.com/ellisk42/ec). Inspected the paper record and repository README/build guidance. MIT repository license observed. The legacy OCaml/Rust/native stack requires a local build check; it was not proved either runnable or impossible on M3 Max.

### R06 — Stitch

Matthew Bowers et al. *Top-Down Synthesis for Library Learning*. POPL 2023; preprint 2022. [HTML paper](https://arxiv.org/html/2211.16605v1); [DOI: 10.1145/3571234](https://doi.org/10.1145/3571234); [official Rust implementation](https://github.com/mlb2251/stitch). Inspected methods and repository; MIT license observed. Important runnable library-learning component, but the numerical parameter-slot adapter is proposed work, not an official SDML integration.

### R07 — Library learning modulo equivalence

David Cao, Rose Kunkel, Chandrakana Nandi, Max Willsey, Zachary Tatlock and Nadia Polikarpova. *babble: Learning Better Abstractions with E-Graphs and Anti-Unification*. POPL 2023; preprint 2022. [HTML paper](https://arxiv.org/html/2212.04596v1). Inspected the equational-theory/e-graph abstraction mechanism and conditions. Shows why syntactic deduplication is not semantic library learning. No full local reproduction performed.

### R08 — Neural lifelong program synthesis

Lazar Valkov, Dipak Chaudhari, Akash Srivastava, Charles Sutton and Swarat Chaudhuri. *HOUDINI: Lifelong Learning as Program Synthesis*. 2018. [HTML paper](https://arxiv.org/html/1804.00218v2); [stable record](https://arxiv.org/abs/1804.00218). Inspected the typed neural-functional representation and lifelong transfer design. Close conceptual overlap; its published setting is not interchangeable with this numerical benchmark.

### R09 — Modular meta-learning

Ferran Alet, Tomás Lozano-Pérez and Leslie Pack Kaelbling. *Modular Meta-Learning*. 2018. [HTML paper](https://arxiv.org/html/1806.10166v2). Inspected BounceGrad, structure/parameter optimization and adaptation variants. Informs the required fixed-module baseline. The plan's small scalar-module implementation is a declared algorithm-level adaptation, not a reproduction of published task scores.

### R10 — RigL and consequential source semantics

Utku Evci et al. *Rigging the Lottery: Making All Tickets Winners*. ICML 2020. [Official proceedings](https://proceedings.mlr.press/v119/evci20a.html); [official repository](https://github.com/google-research/rigl); [inspected optimizer source](https://raw.githubusercontent.com/google-research/rigl/master/rigl/sparse_optimizers_base.py). Inspected mask removal/growth and optimizer-state behavior, in addition to the paper record. Repository uses Apache-2.0. Its TensorFlow source is a semantic reference for the proposed PyTorch port; dense masking is not an actual sparse-speed reproduction.

### R11 — Neural developmental programs

Elias Najarro, Shyam Sudhakaran and Sebastian Risi. *Towards Self-Assembling Artificial Neural Networks through Neural Developmental Programs*. 2023. [HTML paper](https://arxiv.org/html/2307.08197); [official repository](https://github.com/enajx/NDP). Inspected the developmental mechanism and repository/dependency information. Establishes learned local neural growth as direct prior art; no M3 installation was attempted.

### R12 — Lifelong neural developmental programs

Erwan Plantec, Joachin W. Pedersen, Milton L. Montero, Eleni Nisioti and Sebastian Risi. *Evolving Self-Assembling Neural Networks: From Spontaneous Activity to Experience-Dependent Learning*. 2024. [HTML paper](https://arxiv.org/html/2406.09787v1). Inspected activity/reward-dependent structural plasticity, controller design and control-task setting. Supports plausibility of adaptive development, not a theorem or direct baseline score for tiny numerical DAG templates.

### R13 — Local module composition

Oleksiy Ostapenko, Pau Rodriguez, Massimo Caccia and Laurent Charlin. *Continual Learning via Local Module Composition*. 2021. [HTML paper](https://arxiv.org/html/2111.07736); [official repository](https://github.com/oleksost/LMC). Inspected the module-selection/growth mechanism and repository README. No permissive code license was established in the inspected repository view. Do not equate public access with redistribution permission.

### R14 — Conditions for modular compositional generalization

Simon Schug et al. *Discovering modular solutions that generalize compositionally*. ICLR 2024; preprint 2023. [HTML paper](https://arxiv.org/html/2312.15001v2); [official modular-hyperteacher code](https://github.com/smonsays/modular-hyperteacher). Inspected teacher/task-support conditions and JAX implementation documentation; MIT license observed. Provides an independent teacher family and conditional evidence, not universal identifiability for SDML.

### R15 — Growing neural cellular automata

Alexander Mordvintsev, Ettore Randazzo, Eyvind Niklasson and Michael Levin. *Growing Neural Cellular Automata*. Distill, 2020. [Article](https://distill.pub/2020/growing-ca/). Inspected local-rule growth, maintenance and regeneration descriptions. Pattern regeneration is not equivalent to learning reusable task abstractions; included to delimit the self-repair analogy.

### R16 — Structural-plasticity theory

Tiddia, Sergi and Golosio. *Theoretical framework for learning through structural plasticity*. Physical Review E 110, 044311, 2024. [DOI: 10.1103/PhysRevE.110.044311](https://doi.org/10.1103/PhysRevE.110.044311). Inspected publisher abstract, not a full derivation. Context for structural plasticity; no theorem is transferred to the selected DAG architecture.

### R17 — Prospective abstraction selection

Leonardo Hernandez Cano et al. *Prospective Compression in Human Abstraction Learning*. Preprint, 11 May 2026. [HTML paper](https://arxiv.org/html/2605.09985v1); [stable record](https://arxiv.org/abs/2605.09985). Inspected the future-compression criterion, Pattern Builder experiments and computational models. Directly overlaps the proposed prospective objective; human/model evidence does not establish a trainable local numerical circuit policy.

### R18 — Counterfactual skill-edit training

Min Yang, Jinghua Piao, Xu Xia, Xiaochong Lan, Jiaju Chen, Yongshun Gong and Yong Li. *SkillMaster: Toward Autonomous Skill Mastery in LLM Agents*. Preprint, 9 May 2026; latest inspected revision **v3, 30 September 2026**. [HTML v3](https://arxiv.org/html/2605.08693v3); [stable record](https://arxiv.org/abs/2605.08693); [author-linked code](https://github.com/sduyangmin/Skill-Master). Inspected counterfactual probe utility and separate action/skill optimization in earlier and current versions. Important novelty overlap; code was not executed and agent results are not a local numerical head-to-head comparison.

### R19 — Meta-learned memory design

Xiong, Hu and Clune. *ALMA: Learning to Continually Learn via Meta-learning Agentic Memory Designs*. Preprint, February 2026. [HTML paper](https://arxiv.org/html/2602.07755v1); [official repository](https://github.com/zksha/alma). Inspected memory-design method and repository/API setup. A broader learned-organization precedent, not a requirement to introduce foundation-model agents into this PoC.

### R20 — Local growth and pruning

Yiyang Jia and Chengxu Zhou. *Self-Motivated Growing Neural Network for Adaptive Architecture via Local Structural Plasticity*. Preprint, December 2025; revised **27 July 2026**. [HTML v2](https://arxiv.org/html/2512.12713v2). Inspected local growth/pruning rules and policy-distillation setting. Overlaps simple-unit structural adaptation, but does not by itself demonstrate the proposed cross-task template-learning effect.

### R21 — Numerical library learning

Jonas Regehr, Mitch Briles, Zachary Tatlock and Pavel Panchekha. *Numerical Superoptimization for Library Learning*. Preprint, 25 March 2026. [HTML paper](https://arxiv.org/html/2603.24812); [stable record](https://arxiv.org/abs/2603.24812). Inspected GrowLibm's candidate/utility mechanism and distinction between discovered primitives and their expert numerical implementations. Counters a broad numerical-library novelty claim; its application gains cannot be attributed to mere macro naming.

### R22 — Typed motor-program libraries

Riyaaz Shaik and Chandru Venkataraman. *REFACTOR-VLA: Unsupervised Library Learning of Typed Motor Programs*. Preprint, 1 September 2026. [HTML paper](https://arxiv.org/html/2609.01215v1). Inspected typed library learning, admission criteria and stated evaluation scope. Relevant contemporary overlap, with substantially different models/tasks. No broad VLA performance claim is imported into this plan.

### R23 — Negative evidence about alleged library reuse

Berlot-Attwell, Rudzicz and Si. *LLM Library Learning Fails: A LEGO-Prover Case Study*. Preprint, April 2025. [HTML paper](https://arxiv.org/html/2504.03048v1). Inspected the reuse-focused analysis. Motivates direct invocation/removal tests; it does not establish impossibility of useful library learning elsewhere.

### R24 — Counterfactual contribution analysis

Alexander Meulemans et al. *Would I have gotten that reward? Long-term credit assignment by counterfactual contribution analysis*. 2023. [HTML paper](https://arxiv.org/html/2306.16803v2). Inspected the counterfactual credit formulation and reinforcement-learning context. Further prior art against claiming counterfactual structural credit as an entirely new general principle.

### R25 — Compositional open-ended learning framework

Ida Momennejad and Roberta Raileanu. *A Compositional Framework for Open-ended Intelligence*. Preprint, June 2026. [HTML paper](https://arxiv.org/html/2606.15386v1). Inspected the compositional primitives/motifs framework and proposed objective. A conceptual precedent, not an experimentally validated SDML implementation.

### R26 — Relevant patent terminology

United States patent publication **US10019470B2**. *Method and apparatus for constructing, using and reusing components and structures of an artifical neural network* (title spelling in indexed record). Priority 16 October 2013. [Patent record](https://patents.google.com/patent/US10019470B2/en). Inspected description/metadata for broad overlap. This is not a legal claim chart, patentability assessment, validity opinion or freedom-to-operate analysis; legal status and scope are not inferred here.

### R27 — PyTorch MPS backend

PyTorch documentation maintainers. *MPS backend*. Documentation inspected 14 October 2026. [Stable documentation](https://docs.pytorch.org/docs/stable/notes/mps.html). Inspected availability/build/operating-system guidance. Stable links resolved to versioned documentation during research; installed wheel and actual operator support remain local gates, not assumptions.

### R28 — MPS environment and allocation controls

PyTorch documentation maintainers. *MPS Environment Variables*. [Inspected versioned documentation](https://docs.pytorch.org/docs/2.14/mps_environment_variables.html). Informs fallback logging and allocator safety. Disabling allocation safety limits is not a valid feasibility strategy. No target allocator behavior was measured.

### R29 — Device state, memory and reproducibility

PyTorch documentation maintainers. [Recommended maximum MPS memory](https://docs.pytorch.org/docs/2.14/generated/torch.mps.recommended_max_memory.html), [MPS RNG state](https://docs.pytorch.org/docs/2.14/generated/torch.mps.get_rng_state.html), and [reproducibility guidance](https://docs.pytorch.org/docs/2.14/notes/randomness.html). Inspected API/documentation text. Informs logging/checkpoint requirements; does not promise reproducibility across releases/platforms.

### R30 — MLX compilation constraints

Apple MLX documentation maintainers. *Compilation*. [Official documentation](https://ml-explore.github.io/mlx/build/html/usage/compile.html). Inspected shape/state/compilation guidance. MLX is an optional measured engineering path, not required infrastructure and not assumed faster for changing graphs.

### R31 — Stitch bindings and actual API

Stitch maintainers. [Bindings repository](https://github.com/mlb2251/stitch_bindings), [installation](https://stitch-bindings.readthedocs.io/en/latest/intro/install.html), [tutorial](https://stitch-bindings.readthedocs.io/en/latest/intro/tutorial.html). Inspected `stitch_core.compress`, result/abstraction fields and native installation requirements. The displayed smoke API is documented; numerical-template conversion and M3 execution remain proposed work.

### R32 — PySR symbolic regression

Miles Cranmer and PySR contributors. *PySR*. [Official repository](https://github.com/astroautomata/PySR). Inspected README configuration, custom operators, fitting/prediction and Julia dependency. Informs a strong independent CPU baseline. Exact package/source versions and transitive license compatibility must be frozen locally; no successful M3 build is claimed.

### R33 — External sensor-drift data

Alexander Vergara / UCI Machine Learning Repository. *Gas Sensor Array Drift Dataset*, deposited 2012, dataset 224. [Official record](https://archive.ics.uci.edu/dataset/224/gas+sensor+array+drift+dataset); [DOI: 10.24432/C5RP6W](https://doi.org/10.24432/C5RP6W). Inspected dimensions, batch descriptions, corrected batch-10 notice and inconsistent usage/license statements. These conflicts are retained as a gate in Section 5.9, not silently resolved. No asset was downloaded or fitted during research.

### R34 — Strong first-order meta-learning baseline

Alex Nichol, Joshua Achiam and John Schulman. *On First-Order Meta-Learning Algorithms*. 2018. [HTML paper](https://arxiv.org/html/1803.02999v3). Inspected Reptile's parameter update and inner-task adaptation. Replaces an inadequate pooled-regression interpretation of the prefix-trained MLP baseline. The selected numerical configuration is project-specific and must be tuned fairly.

### R35 — Function Encoder baseline

Tyler Ingebrand, Adam J. Thorpe and Ufuk Topcu. *Function Encoders: A Principled Approach to Transfer Learning in Hilbert Spaces*. ICML 2025. [HTML paper](https://arxiv.org/html/2501.18373v2); [author project](https://tyler-ingebrand.github.io/FEtransfer/); [experimental repository](https://github.com/tyler-ingebrand/FEtransfer); [algorithm package repository](https://github.com/tyler-ingebrand/FunctionEncoder). Inspected least-squares training, regularization, basis architecture and README setup. The multi-GPU sweep is not needed for a scalar-regression adaptation. Repository views did not establish a permissive code license; no code was installed or vendored.

## 14. Search coverage, limitations and handoff completion

### 14.1 Search record

The search was mechanism-driven rather than limited to the SDML name. It covered historical neural growth, automatically defined functions and Cartesian GP; dynamic sparsity and optimizer semantics; neural developmental programs and structural plasticity; lifelong modular composition and numerical meta-learning; typed neural program synthesis; library induction, compression and semantic equivalence; prospective/counterfactual abstraction value; continual skill/memory learning; negative evidence for claimed library reuse; contemporary numerical-library and motor-program work; patents; and current Apple-compatible numerical tooling.

Representative query families included `neural developmental programs local growth`, `lifelong neural developmental program structural plasticity`, `Cartesian genetic programming automatic modules reuse`, `library learning future utility prospective compression`, `counterfactual skill library learning`, `numerical library superoptimization`, `modular meta-learning compositional generalization`, `function encoder transfer least squares`, and `PyTorch MPS dynamic graph gradients memory`. Backward tracing connected SDML's named precedents to GP, HOUDINI, BounceGrad and related library work. Forward/current searches added 2025–2026 work that materially narrowed novelty and strengthened baselines.

The final refinement searches combined local policy observations, amortized library learning, numerical microcircuits, consolidation and counterfactual utility. They largely returned already covered library/credit mechanisms or neighboring memory/orchestration settings, rather than establishing a missing new principle. The last consequential baseline addition was Function Encoder; the latest SkillMaster revision was checked before finalization. This is practical coverage saturation for the selected design, not proof of exhaustive coverage of all publications.

Primary sources supported consequential decisions. Secondary bibliographic records and public research notes were used as discovery leads only; their novelty judgments were not adopted as authority. In particular, a public note about prospective abstraction credit led to primary-source checks of SkillMaster/related work. No private repository, inaccessible dataset or uninspected benchmark result is required to understand this handoff.

### 14.2 Remaining blind spots

The search is mainly English-language and web-indexed. Full patent claim analysis, all unpublished workshop work, every genetic-programming implementation and all non-English literature are outside its evidentiary reach. Some older publisher material was accessible only as abstracts; this is recorded in the relevant references. HTML access was used for paper inspection; no unperformed PDF/image analysis is claimed. Public repository pages can change; immutable revisions still need local capture.

No evidence currently establishes a favorable SDML effect size, the number of worlds required, an M3 throughput rate, a positive breakeven, application demand or broad open-ended development. These are outcomes or explicit prerequisites, not hidden assumptions. Exact dependency pins, resource caps, time budgets and final sample size become resolved artifacts before confirmation using the rules specified above.

### 14.3 Handoff checklist

The selected implementation preserves tiny operations, learned structural proposals, temporary circuits, exact consolidation, reuse and bounded retirement. Departures from the full vision—smooth numerical tasks, flat templates, global teaching signals, no recurrence, no repair and no approximate primitives—are explicit. Claims map to hypotheses, strong baselines, metrics and causal controls. Expensive work depends on correctness, baseline readiness, headroom and local calibration. Every work package specifies inputs, actions, outputs, acceptance and branches. Proposed commands are distinguished from existing APIs; failures, uncertainty and unperformed checks are retained.

**The coding agent's first action is WP00, not architecture expansion. Its first scientific question is whether useful reusable-template headroom exists against the strong numerical and library baselines. Its final authority is the measured, reproducible comparison—not the appeal of the SDML name.**

[R01]: https://papers.nips.cc/paper_files/paper/1989/hash/69adc1e107f7f7d035d7baf04342e1ca-Abstract.html
[R02]: https://research.google/pubs/pathnet-evolution-channels-gradient-descent-in-super-neural-networks/
[R03]: https://www.genetic-programming.com/jkpubs94.html
[R04]: https://doi.org/10.1109/TEVC.2007.903549
[R05]: https://arxiv.org/abs/2006.08381
[R06]: https://arxiv.org/html/2211.16605v1
[R07]: https://arxiv.org/html/2212.04596v1
[R08]: https://arxiv.org/html/1804.00218v2
[R09]: https://arxiv.org/html/1806.10166v2
[R10]: https://proceedings.mlr.press/v119/evci20a.html
[R11]: https://arxiv.org/html/2307.08197
[R12]: https://arxiv.org/html/2406.09787v1
[R13]: https://arxiv.org/html/2111.07736
[R14]: https://arxiv.org/html/2312.15001v2
[R15]: https://distill.pub/2020/growing-ca/
[R16]: https://doi.org/10.1103/PhysRevE.110.044311
[R17]: https://arxiv.org/html/2605.09985v1
[R18]: https://arxiv.org/html/2605.08693v3
[R19]: https://arxiv.org/html/2602.07755v1
[R20]: https://arxiv.org/html/2512.12713v2
[R21]: https://arxiv.org/html/2603.24812
[R22]: https://arxiv.org/html/2609.01215v1
[R23]: https://arxiv.org/html/2504.03048v1
[R24]: https://arxiv.org/html/2306.16803v2
[R25]: https://arxiv.org/html/2606.15386v1
[R26]: https://patents.google.com/patent/US10019470B2/en
[R27]: https://docs.pytorch.org/docs/stable/notes/mps.html
[R28]: https://docs.pytorch.org/docs/2.14/mps_environment_variables.html
[R29]: https://docs.pytorch.org/docs/2.14/notes/randomness.html
[R30]: https://ml-explore.github.io/mlx/build/html/usage/compile.html
[R31]: https://stitch-bindings.readthedocs.io/en/latest/intro/tutorial.html
[R32]: https://github.com/astroautomata/PySR
[R33]: https://doi.org/10.24432/C5RP6W
[R34]: https://arxiv.org/html/1803.02999v3
[R35]: https://arxiv.org/html/2501.18373v2
