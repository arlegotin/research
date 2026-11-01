# SIML: research assessment and executable proof-of-concept plan

**Research date:** 14 October 2026  
**Target:** one MacBook Pro with an M3 Max; actual memory, operating system, and storage must be discovered locally  
**Status:** researched experimental specification, not a validated architecture or a claim of novelty  
**Primary direction:** small, gradient-informed local circuit proposals with transfer-validated, immutable, hierarchically reusable microcircuits  
**Working method identifier:** `siml_local_transfer` (an experiment identifier, not a novelty claim)

## 1. Executive assessment

SIML is worth a controlled investigation, but its broad premise is already established research territory. Automatically constructing reusable subprograms, learning libraries, combining differentiable modules with structural search, and assessing whether learned knowledge actually speeds up later problem solving all have substantial precedents. DreamCoder, HOUDINI, modular meta-learning, local module composition, and the older utility-problem literature are particularly consequential [R01, R04–R06, R08]. Calling an extracted subgraph a “new primitive” does not establish an original method.

The recommended contribution to investigate is narrower:

> **Can a small policy, restricted to a bounded neighborhood of a differentiable microcircuit and its learning signals, propose reusable circuit boundaries and compositions that improve later adaptation more economically than strong centralized library discovery and fixed modular learning, after paying for discovery, validation, storage, and execution?**

This preserves the proposed causal mechanism: tiny computational units; learned structural changes; experience-derived circuits; explicit consolidation; and future learning using the resulting vocabulary. It does not substitute an LLM, prompting pipeline, or an unrestricted textual program synthesizer for the proposed layer. Its first implementation is nevertheless a hybrid differentiable/discrete learner. Numerical execution belongs to the layer; a generic CPU runtime validates topology and performs bounded search and empirical acceptance tests.

Three qualifications govern the project. First, the available evidence does **not** establish that this particular local mechanism is new. Second, selecting abstractions for held-out search benefit rather than past compression is itself prior art, including a directly relevant September 2026 preprint [R09] and related DSL optimization work [R10]. Third, no M3 Max benchmark or complete SIML experiment has been performed for this plan. Only limited semantic and analytical checks were executed, on a Linux CPU; their exact scope appears in Section 4.6.

### Recommended research sequence

Build the typed runtime and an unbiased task/evaluation harness first. Reproduce a native Stitch compression example and a competent fixed-module learner before implementing the learned consolidation policy. Then run the minimum complete comparison: `siml_local_transfer`, `stitch_transfer`, `fixed_modular`, and `primitive_only`, with shared numerical execution and genuinely comparable costs. Use an offline common-donor experiment to isolate candidate quality before spending on end-to-end development. Advance to hierarchical and out-of-generator transfer only when the earlier experiment has sufficient sensitivity.

The strongest locally supportable positive conclusion would be **a reproducible component-level advantage in cumulative computational abstraction on specified task distributions and resource budgets**. Neither a good-looking circuit hierarchy nor synthetic success establishes general intelligence, universal self-improvement, biological locality, a universally superior neural layer, or real-world deployment value.

### What is already available, what is proposed

| Category | Assessment |
|---|---|
| Existing capabilities | Learned libraries, modular composition, neural-guided synthesis, network construction, dynamic connectivity, and utility-based acceptance already exist. |
| Reusable implementation | Native Stitch compression; ordinary automatic differentiation; typed graph representations; first-order meta-learning; PySR for appropriate numerical comparisons. |
| Proposed mechanism | A bounded local feature/action interface; learned edit and consolidation scores; independent training-task transfer gates; exact versioned circuits; controlled multigeneration reuse. |
| Plausible differentiation | The quality/cost tradeoff of this particular local neural proposal mechanism, not the existence of libraries or validation. |
| Main alternative explanation | A common solver plus a well-chosen reusable library is sufficient; local self-organization adds no benefit. |
| Decisive direct test | Local proposals versus native Stitch proposals, both receiving the same donor information and the same transfer gate, with all costs counted. |
| Practical decision | Continue only when an identifiable uncertainty remains and the next experiment can resolve it. A small genuine effect is evidence, but not automatically useful or groundbreaking. |

## 2. Problem, scope, assumptions, and hypotheses

### 2.1 Beneficiary and bottleneck

The initial beneficiary is a researcher or developer building a learner for a **stream of related small numerical tasks**, where new functions repeatedly require combinations of previously useful computations. The existing workflow is to train or adapt a dense network, select from a fixed module collection, or search expressions using a fixed or learned language.

The proposed bottleneck is repeated discovery of computational structure. It is an assumption about the target task distribution, not a universal criticism of neural networks. Before claiming this bottleneck, measure whether a known useful macro library actually improves a competent baseline under a finite budget. If an oracle library cannot help at any scientifically meaningful local depth, the benchmark cannot establish the proposed advantage. If all methods already solve everything easily, there is no measurable headroom; if no method can acquire the constituent functions, the experiment is testing failure of the substrate instead.

No end-user application was supplied. Do not invent one. Scientific derivative regression is a later, explicitly simulated transfer probe, not a promised product.

### 2.2 Learning setting and notation

A **world** `w` is an independently generated task ecosystem: its own latent reusable functions, task grammar, curriculum, parameters, and random seeds. Methods train separate persistent states in each world. Development worlds and protected confirmatory worlds are disjoint.

For task `t`, observations are `D_t = {(x_i, y_i)}`, with `x_i ∈ [-1,1]^d` and a scalar bounded target `y_i ∈ [-1,1]` in the core experiments. A learner observes a support set `S_t`; a separate evaluator owns query set `Q_t`. The latent generating program, motif identities, and query labels never enter the learner's adaptation API.

A persistent state is

\[
\mathcal{S}_t=(\mathcal{P},\mathcal{L}_t,\phi_t,\psi_t,\mathcal{A}_t),
\]

where `P` is the designer-provided elementary operation vocabulary, `L_t` is an immutable-version library, `φ_t` contains small proposal-policy weights, `ψ_t` contains permitted numerical initializations, and `A_t` is the charged archive of donor graphs, metadata, and replay information. An adapted model is a typed graph `G_t` and task-owned numerical parameters `θ_t`:

\[
\hat y = F_{G_t,\mathcal L_t}(x;\theta_t).
\]

Distinguish **primitive unit**, **candidate circuit**, **accepted library entry**, **call site**, and **task-owned specialization**. A macro call is one *structural search choice* but still executes its expanded elementary operations. Library generation is zero for designer primitives and `1 + max(dependency generations)` for a newly accepted circuit containing earlier entries.

The target is expected future adaptation quality under resource constraints, not maximization of library size:

\[
\min_{\text{learner}}\; \mathbb{E}_{w,t}[\operatorname{AULC}(w,t)]
\quad\text{subject to}\quad
T_{\rm development}\le B_{\rm meta},\;
T_{\rm adapt}\le b,\;
M_{\rm peak}\le M_{\rm safe}.
\]

Here “development” means that method's within-world learning, including structural exploration and validation, not the research team's software development time. Both research tuning expenditure and method expenditure are reported separately. AULC is defined precisely in Section 7.

### 2.3 Explicit refinements of the original proposal

The first version uses scalar, differentiable, state-free operations and bounded acyclic expression graphs. Recurrent state, chemical-style dynamics, economic allocation, damage repair, and approximate circuit replacement are excluded. These are scope reductions, not evidence against those extensions.

The first controlled runtime restricts internal nodes to single-consumer expression trees; input leaves may recur. A tree is a restricted DAG. This permits lossless comparison to established expression-library tools without an unequal tree-to-DAG translation. A later DAG experiment must count common-subexpression savings and expansion costs explicitly. The core result cannot be advertised as validating arbitrary graph self-organization.

Policies receive local activations and **backpropagated gradients**. Those gradients can carry task-wide information. The correct description is “gradient-informed local observability,” not strictly decentralized or biologically local learning. Global acceptance tests are explicitly part of the algorithm and its expense.

### 2.4 Hypotheses and falsifiers

| ID | Testable hypothesis and supporting motivation | Main alternatives | Discriminating observation / weakening result |
|---|---|---|---|
| H0: substrate | Tiny units can fit and compose the intended bounded functions reliably. Differentiable composition is established; this particular vocabulary is a design choice. | Numerical conditioning, not structure, dominates difficulty. | Known-graph fitting and input/parameter-gradient tests succeed across required depths. Failure here blocks interpretation of invention experiments. |
| H1: local proposal value | Bounded local learning signals predict which candidate edits/cuts will improve future learning. LMC shows local modular decisions are plausible, but is not this mechanism [R06]. | Frequency, a generic global proposer, or random proposals work as well. | Higher paired transfer utility per discovery second than controls, on unseen donor contexts; disappearance without gradient features localizes the source of information. |
| H2: transferable consolidation | Accepted circuits help novel tasks rather than repeated examples. Library learning gives precedent, not assurance [R01, R07]. | Cached solutions, extra parameters, teacher grammar bias, or gate leakage explain gains. | Protected AULC benefit survives matched library, replay, cost, and information controls. No-macro *adaptation* removes the benefit. |
| H3: locality adds value | Local proposals yield a better quality/cost frontier than centralized candidates under a common gate. | Stitch plus the gate captures all useful structure. | `siml_local_transfer` beats `stitch_transfer` or achieves comparable quality at lower total cost; otherwise narrow the claim to a replication or empirical comparison. |
| H4: cumulative abstraction | Earlier accepted circuits causally facilitate discovery and transfer of later circuits. | A flat library, weight warm starts, or increasing capacity suffices. | History interventions disabling earlier calls reduce subsequent discovery/transfer, and generation-two entries add benefit beyond a flat library. A hierarchy diagram alone is insufficient. |
| H5: net value and robustness | Benefits amortize over new tasks and survive at least one independent generator. | Discovery overhead, vocabulary distraction, or distribution shift erases value. | Report positive or negative break-even horizon and G2 transfer. Synthetic G1-only success narrows scope rather than proving generality. |

The formal primary confirmation tests H2/H3 as an **algorithm-level** comparison. H1 is first assessed in a controlled donor experiment; H4 and H5 require additional evidence. This prevents one attractive result from being used to claim the entire original vision.

### 2.5 Strongest case for and against SIML

**For:** repeated bounded subcomputations create exploitable task structure. A stable callable circuit can shorten future structural decisions, preserve useful numerical behavior, and transfer across input roles. A local proposer could avoid repeatedly scanning a large corpus and focus attention on newly useful regions. Immutable versions can protect previous computations while new specializations adapt.

**Against:** local evidence may be insufficient to predict future usefulness; extraction may simply reproduce ordinary library compression; numerical snapshots may be too specific; more callable operations increase the branching factor; parameter gradients can vanish despite bounded activations; and CPU/GPU dispatch, candidate fitting, and repeated transfer tests can dominate any search savings. A fixed module learner may acquire a better shared representation without discrete invention. Cheap centralized code can outperform a conceptually elegant local policy. These are competing explanations to test, not reasons to predetermine rejection.

The proposal depends on repeated functional structure, accessible learning signals, a manageable macro interface, a distribution stable enough for reuse, and sufficient future tasks to recover discovery costs. None is guaranteed.

## 3. Research findings and novelty assessment

### 3.1 Closest prior-art comparison

The matrix summarizes inspected sources. “Local candidate” refers to feasibility at reduced, declared experimental scale, not reproduction of every published result.

| Work | Mechanism and exact overlap | Evidence/assumptions and limitations | Implementation / resource position | Consequence for SIML |
|---|---|---|---|---|
| DreamCoder [R01] | Joint library discovery and learned search; recursively reusable abstractions. | Program-synthesis tasks with a provided base language; results do not directly establish scalar neural-layer superiority. | Public original code; toolchain reproduction is nontrivial. | Broad primitive invention and cumulative libraries are not new. |
| Stitch [R02, R03] | Efficient top-down extraction of reusable parameterized expressions from program corpora. | Compression objective; shorter descriptions need not mean cheaper future search. | Rust implementation and Python bindings; a mandatory local baseline. | Use the native engine, not a weak imitation; compare both its native objective and a shared transfer gate. |
| HOUDINI [R04] | Typed differentiable programs, structural search, and reusable learned neural components. | Lifelong program-learning setting with a typed language. | Public historical implementation; dependency modernization needs validation. | “Inside differentiable computation rather than symbolic code” does not by itself escape prior art. |
| Modular meta-learning / BounceGrad [R05] | Learns numerical modules while searching compositions; task adaptation can reuse and refine components. | Fixed composition/module design, rather than SIML-style boundary invention. | A small CPU/MPS adaptation is locally credible; original MIT-licensed code is available. | Strong fixed-module comparison; do not cripple it by banning composition or fine-tuning. |
| Local Module Composition [R06] | Local structural relevance estimates and modular expansion/reuse for continual learning. | Primarily modular continual-learning workloads; relevance decisions differ from scalar microcircuit extraction. | Public implementation; native-domain reproduction is separate from a numerical adaptation. | Locality plus growth and reuse is already known. |
| AbstractBeam [R07] | Execution-guided bottom-up synthesis with learned abstractions. | Integer/list domains; library growth also increases search width. | Public Apache-2.0 code. Inspected training configuration targets multiple GPUs/processes; full configuration is not a laptop baseline. | Closest broader modern reference; distinguish its native reproduction from a port. |
| COMPOSER [R08] | Empirical acceptance of learned control knowledge according to utility, considering interference and cost. | Planning/control-rule representation, not neural circuits. | Historical algorithm; no need to rebuild its original planner. | Utility-aware consolidation and the knowledge-utility problem predate SIML. |
| Cuculich 2026 [R09] | Compares compression and held-out search-cost selection from the same candidate pool. | PatternBuilder; unreviewed September 2026 preprint. | Author provides code/data link; not independently reproduced here. | Transfer/cost validation alone is not a defensible novelty claim. |
| AMaze [R10] | Optimizes a synthesis language using a learned estimate of synthesis cost. | Syntax-guided synthesis and its cost model, not local differentiable micro-units. | Inspected paper; no local reproduction asserted. | A learned, economically useful vocabulary is also an established direction. |
| Prospective compression / E-Stitch [R11, R12] | Future-oriented abstraction utility; equivalence-aware library extraction. | Respectively a 2026 preprint and workshop material; different experimental settings. | Limited directly reusable implementation verification. | Include nonstationarity and algebraic-duplicate controls; do not assume syntactic novelty equals functional novelty. |
| PySR / FePySR [R23, R29] | Strong numerical expression search; FePySR adds neural discovery of useful features before search. | Per-dataset symbolic regression differs from a persistent multi-task library. FePySR is a 2026 preprint. | PySR Apache-2.0; FePySR repository declares CC BY-NC 4.0, not unrestricted open-source reuse. | Numerical feature invention has close competitors. Use PySR locally; license-gate FePySR rather than ignoring it. |
| Cascade-Correlation / RigL [R17, R18] | Constructive units / dynamic connectivity under constraints. | Neither alone is a reusable library mechanism. | Small faithful algorithmic controls are practical; kernel-speed claims require native sparse execution. | Growth or rewiring alone cannot establish invention. |
| Compositional Interfaces / Shan et al. [R20, R21] | Shared modular interfaces / decomposed recurrent computation and component reuse. | Different control/task assumptions; Shan's setting includes task information. | Broader neural reference results, not comparable published scores. | Neural composition and continual reuse do not establish SIML novelty. |

Historical automatically defined functions, OOPS, and Neural Programmer-Interpreters reinforce the same boundary: stable reuse, incremental computation, and compositional programs are not new concepts [R15, R16, R19]. LILO and LaSR demonstrate language-model-assisted abstraction approaches, but using such controllers here would change both the mechanism and the local resource contract [R13, R30].

### 3.2 What the investigation changed

The original idea correctly singled out DreamCoder, but a stronger plan must also face HOUDINI, BounceGrad, LMC, native Stitch, and execution-guided library learners. The 2026 search further removed “future-utility rather than compression” as a stand-alone novelty claim. The selected direction therefore tests the **information restriction and cost of the proposal mechanism**, with the consolidation rule shared by a strong competitor.

A negative-results paper found that claimed library benefits in two LLM-based systems could instead come from other components [R14]. Its domain does not justify concluding that library learning generally fails. It does justify recording actual call reuse and using causal removals rather than relying on module counts or success anecdotes.

There is no contradiction between fast compression, harmful library expansion, and useful learned abstractions. They measure different stages and budgets. Compression may be efficient while adaptation becomes slower; adding a useful macro can still make a shallow search worse; and a module can preserve a function without improving discovery of new tasks. All three effects must be separated here.

### 3.3 Novelty classifications and invalidation rules

**Not novel:** constructing units; rewiring; differentiable composition; extracting reusable subgraphs; immutable software versions; hierarchical naming; neural proposal scores in general; held-out utility tests in general; CPU/MPS execution.

**Plausibly differentiated, unresolved:** a particular bounded gradient-informed proposal interface that learns which local circuit boundaries to consolidate, has low overhead, and demonstrably improves the transfer/search frontier relative to centralized extraction under the same gate. This could be a methodological contribution, or simply a useful empirical finding about when local information suffices. Evidence, not naming, decides.

**Engineering contribution only:** faster batching, graph packing, or a better Mac implementation that benefits all methods. Apply these optimizations to baselines before claiming a learning advantage.

**Novelty would be weakened or invalidated by:** a prior system with essentially the same observations, actions, update rule, and acceptance mechanism; a reduction showing the method is an existing library learner with a different encoding; or evidence that the apparent advantage comes entirely from a generic shared optimization. Continue as a replication or narrow empirical study where useful, rather than relabeling the mechanism.

A relevant Fang et al. identifier was located, but full text could not be accessed through the publisher during this research [R22]. Its apparent modular expansion overlap remains an unresolved lead, not a verified basis for detailed claims. An “InceptionSR” citation encountered inside FePySR could not be independently located by targeted exact-title/name search; it is **not included as an established source**. Do not propagate that reference as verified.

### 3.4 Selected direction versus alternatives

Pure MDL extraction is simpler and should be adopted if it matches performance at lower total cost. A global proposer is easier to inform but weakens the local-organization claim; it is a valuable control and an evidence-triggered fallback. Unrestricted reinforcement learning over long structural histories adds severe credit-assignment uncertainty without being necessary to test H1–H3. Recurrent self-organizing populations expand the problem before the basic transfer mechanism is established.

Choose the bounded local method now because it preserves the distinctive research question while permitting close controlled alternatives. Do not condition the choice on an assumed positive result.

## 4. Theory, headroom, and checks performed

### 4.1 Expressivity and exact wrapping

For a finite acyclic macro library, recursively replacing each call by its body yields a graph in the original vocabulary. This follows by induction on library dependency generation. A versioned exact macro therefore adds no new function beyond those compositions; it changes the search representation, access pattern, parameter reuse, and possibly implementation overhead.

If expansion preserves primitive evaluation order, numerical parameters, and alias identities, it also preserves the underlying differentiable computation. Floating-point equivalence still requires a defined execution order; algebraic rewrites can change rounding. The core compiler does not reassociate arithmetic or replace expressions with approximations.

This statement concerns the selected finite vocabulary. No claim that these bounded primitives are computationally universal is needed or made.

### 4.2 Search depth versus branching factor

For an illustrative complete search tree, `N(b,d)=Σ(i=0..d)b^i`. Adding `K` macros can help only when useful depth reduction outweighs added choices and evaluation cost. For example, with branching 8 and depth 3 there are 585 nodes; adding four choices without reducing depth gives 1,885. With branching 8/depth 6 there are 299,593; increasing branching to 40 while reducing depth to 4 gives 2,625,641. These are exact arithmetic illustrations, **not measured solver runtimes**.

Measure actual legal-action counts, candidate evaluations, duplicate rates, and expanded execution. Token shortening is a mechanism proxy, not the outcome metric. The archive and selector can become the bottleneck even when a solution uses fewer macro calls.

### 4.3 Limits of local information

Suppose two task ecosystems produce exactly the same local feature/history distribution for a candidate during development, but their future tasks reward opposite reuse choices. A policy using only those features must make the same distribution of decisions in both. It cannot be optimal in both. Thus no distribution-free claim that local activity reveals future abstraction utility is justified.

The practical claim must depend on regularities across tasks and learning from wider outcomes. The experiment tests whether a bounded interface retains enough information on declared distributions, and measures the gap to a global-information control. Removing gradient features tests whether task-wide credit arriving through backpropagation is indispensable.

### 4.4 Numerical conditioning

Bounded outputs prevent activation overflow for the core primitives, but do not prevent vanishing or exploding derivatives through compositions. Products can suppress signal variance; sigmoid/tanh can saturate; TEST can amplify an input derivative by at most its selected slope divided by four. For a chain, Jacobian norms multiply; fan-out and shared parameters require summed contributions. Track input-Jacobian norm, parameter-gradient norm, activation variance, and saturation by expanded depth, not macro depth.

Naming a deep circuit does not improve its Jacobian. Improvements due to frozen weights, warm starts, or reduced adaptation dimensionality must be separated from structural discovery. Use float64 CPU finite differences away from numerical extremes and float32 training; do not infer model failure from a faulty low-precision kernel.

### 4.5 Net headroom and amortization

Let extra discovery expense relative to baseline be `D` seconds and average later saving at a fixed quality target be `s` seconds/task. If `D>0` and `s>0`, the illustrative break-even horizon is `ceil(D/s)`. For example, two extra hours and ten seconds saved per task imply 720 tasks. These numbers are a calculation, not a forecast. If quality never matches, `s` is not defined; show the error/time frontier instead. If `s≤0`, there is no positive time-amortization claim.

At an equal meta-training budget, a method spending heavily on validation has less time for ordinary learning. Report this opportunity cost. In a separate amortized regime, allow different meta-training costs and plot cumulative cost against the number of new tasks. Neither regime replaces the other.

### 4.6 Targeted checks actually executed for this plan

Environment: Linux x86_64, Python 3.13.5, PyTorch 2.10.0+cpu; MPS unavailable. These checks are not M3 compatibility evidence and do not constitute an implementation of SIML.

A small differentiable macro `tanh(θ0*a+θ1*b)*a`, called twice with shared parameters, was compared with its explicitly expanded expression on 257 random input pairs. Maximum forward difference and maximum parameter-gradient difference were both exactly zero in that check for float32 and float64. A separate mutation check kept a cloned numerical snapshot unchanged while an aliased live parameter tensor changed the output by approximately `0.0885093`. This verifies the small example and illustrates the versioning risk; it is not a universal property test or an efficiency measurement.

The following self-contained reference check reproduces the semantic setup; the coding agent should extend it to the actual runtime rather than treating it as sufficient coverage:

```python
import torch

def macro(a, b, theta):
    return torch.tanh(theta[0] * a + theta[1] * b) * a

for dtype in (torch.float32, torch.float64):
    torch.manual_seed(20261014)
    x = 2 * torch.rand(257, 2, dtype=dtype) - 1
    p = torch.tensor([0.3, -0.4], dtype=dtype, requires_grad=True)
    q = p.detach().clone().requires_grad_(True)
    wrapped = macro(x[:, 0], x[:, 1], p) + macro(x[:, 1], x[:, 0], p)
    expanded = (torch.tanh(q[0]*x[:, 0] + q[1]*x[:, 1])*x[:, 0]
                + torch.tanh(q[0]*x[:, 1] + q[1]*x[:, 0])*x[:, 1])
    gp, = torch.autograd.grad(wrapped.square().mean(), p)
    gq, = torch.autograd.grad(expanded.square().mean(), q)
    assert torch.equal(wrapped, expanded)
    assert torch.equal(gp, gq)
```

Research also inspected public documentation, paper text, selected PDF pages visually, repository READMEs, licenses, and selected source/configuration files. Attempts to fetch repository revisions directly from the research container failed because of network resolution. Web inspection succeeded, but no source checkout, exact commit pin, end-to-end benchmark, or native baseline build was completed. Resolving and recording revisions is consequently a setup gate, not a completed deliverable.

## 5. Mechanism and implementation specification

Everything in this section is the **proposed implementation**, except where an existing algorithm is cited. Numerical values labeled “pilot start” are executable starting configurations, not validated optima or permanent experimental limits. Changes to the scientific comparison must follow Section 11.

### 5.1 Elementary vocabulary and numerical conventions

Use two signal types: `R` for real values in `[-1,1]` and `P` for values in `[0,1]`. Use scalar signals initially; batches do not change the signal type. Each primitive has at most two signal inputs and at most two trainable scalar parameters. Parameters are metadata, not additional signal wires.

| Primitive | Signature | Exact computation |
|---|---|---|
| `MIX` | `(R,R) -> R` | `sigmoid(alpha)*a + (1-sigmoid(alpha))*b` |
| `MUL` | `(R,R) -> R` | `a*b` |
| `NEG` | `R -> R` | `-a` |
| `SHIFT` | `R -> R` | `tanh(a + 2*tanh(theta))` |
| `TEST` | `R -> P` | `sigmoid((1+7*sigmoid(beta))*(a-tanh(theta)))` |
| `GATE` | `(P,R) -> R` | `p*a` |
| `NOT` | `P -> P` | `1-p` |
| `AND` | `(P,P) -> P` | `p*q` |
| `CONST_R` | `() -> R` | `tanh(theta)` |

Inputs are typed terminals, not learned modules. Probability constants are fixed `0` and `1` terminals. `CALL` is a reference to a typed library entry. A `P` signal may be used at an `R` port by an explicit identity cast; do not insert implicit casts that make the search grammar method-dependent. The primary target is `R`; `P` results can be returned through that cast. Include cast cost consistently.

These operations are supplied by the designer. Discovering a composition of them is the experimental question. The experiment must not count `MUL`, `TEST`, or a hand-written difference detector as an invented primitive.

Use raw parameter initialization `Normal(0,0.5²)` in the pilot, except a zero-centered `MIX` parameter. Learn per-op initializations on meta-training tasks as a common optional facility for controlled graph methods; all updates count toward the budget. Parameter transforms enforce range bounds without hard parameter clipping. Training loss is mean squared error; gradient clipping, weight decay, and complexity penalties are development-tuned settings shared where applicable and logged. No dropout, mixed precision, stochastic arithmetic, division, logarithms, or discontinuous comparisons are needed in the core runtime.

Do not clamp every intermediate to conceal an implementation error. The definitions already give bounds. Assert finite values; log a violation with graph and input hashes. Numerical mismatch tolerance is calibrated from CPU float64 references and depth, not selected to excuse a favorable result.

### 5.2 Graph and library objects

Use immutable serialized records and an explicit numerical store. A suitable representation is:

```text
Node:
  local_id: integer                     # canonical topological order
  op: enum | CALL
  inputs: list[InputRef | NodeRef]      # typed; stable left-to-right order
  parameter_slots: list[integer]
  callee_version: sha256 | null

CircuitVersion:
  format_version: integer
  input_types: list[R | P]
  output_type: R | P
  nodes: list[Node]
  output_ref: reference
  parameter_layout: list[slot metadata]
  parameter_snapshot: float32 bytes
  dependency_versions: sorted list[sha256]
  structure_hash: sha256
  version_hash: sha256                 # graph + exact parameter bytes + dependencies
  generation: integer
  expanded_ops: integer
  expanded_parameter_slots: integer
  serialized_bytes: integer
  provenance: donor/task/candidate IDs, creation event, fit and gate partitions

LibraryIndex:
  callable_versions: set[sha256]
  retired_versions: set[sha256]
  dependency_refcounts: map[sha256, integer]
  usage_and_utility_records: append-only records
```

`structure_hash` excludes numeric values but includes parameter layout and dependencies' structural identities. `version_hash` includes every behavioral dependency. Canonicalize variable names and topological numbering only. Do not apply unverified algebraic identities. Behavioral fingerprints are approximate diagnostics, not proofs of equality and not the authoritative key.

**Invariants:** every edge is type-correct; all internal references point backward; no dependency cycles; every output is reachable from declared inputs/constants; all numeric slots are defined; no dead nodes remain in accepted bodies; expanded size is computable; retirement cannot delete a version needed by a stored model; and loading a library gives the same behavior before any new learning. Reject malformed candidates rather than repairing them with task-specific logic.

A task owns its scratch graphs and mutable parameter arrays. The library owns read-only snapshots. Default calls use the frozen snapshot. `SPECIALIZE` clones a version's numerical slots into task-owned storage, preserving the original behavior at the instant of cloning. Subsequent gradients may change the clone but never the library or another call's clone. A specialized result can later be proposed as a **new** version. Whether frozen calls, task-local specialization, or both are available is tuned on development data, frozen for confirmation, and equalized across controlled library baselines.

Higher-generation entries reference exact lower-generation versions. Initially prohibit recurrence, cyclic calls, state, and hidden global variables. Exact consolidation wraps a body; it does not distill it. Approximate replacement would require a separate error budget and experiment.

### 5.3 Runtime and common structural solver

Implement a slow CPU reference interpreter first and a packed execution path second. For trees of the same topology, batch parameters and examples. For mixed topology, group by expanded signature; compile/cache a topological operation tape. Do not launch one GPU kernel per scalar neuron. The compiled path expands calls using the same evaluation order as the interpreter and can reuse an execution tape, not previously computed outputs for new inputs.

The common graph solver supports:

- replacing an operation with a signature-compatible operation; rewiring an input to a legal input terminal or compatible subtree;
- inserting a typed operation, or deleting a node by a type-correct bypass;
- inserting a library call and its boundary connections, and explicitly specializing a call;
- proposing a connected backward cut as a candidate circuit; and `NO_CHANGE`.

For the tree restriction, rewiring an internal subtree copies its structure and task-owned parameters rather than creating undisclosed sharing. Count the copy. `COMPOSE` is insertion of a macro call into an ordinary legal composition, not a hand-coded motif template. `GROUP`/`CONSOLIDATE` change representation only after validation.

Use a population of parameterized candidate graphs with mutation and annealed acceptance. A reproducible pilot starts with 16 candidates, elementary input/constant roots and typed random graphs of one to four operations, a maximum of 32 expanded operations, and short batches of eight Adam steps per proposed edit. These values only make the pilot executable. The required graph cap must exceed the expanded size of the deepest protected targets and permit a meaningful search margin; it must not be chosen so that only macros fit a target within the cap.

For each search iteration: select a parent using the configured score/temperature; produce a legal edit; copy unchanged slots and optimizer moments by stable identity; initialize new slots; fit on support data; evaluate the configured support-only selection score; update the population and archive. A simple score is `support_MSE + λ_size*expanded_ops + λ_param*mutable_parameters`. Include zero penalties in tuning; choose penalties on development data, not hidden targets. Lower loss dominates unless a prespecified penalty selects a simpler model. Every baseline can return its best valid incumbent if time expires.

Keep a fraction of random proposals, with a pilot starting value of 0.2, to avoid eliminating unexplored actions. Annealing, restarts, population size, fit steps, and final refinement are tunable within matched research budgets. No feasibility or positive result is assumed from this solver; successful baseline reproduction is a gate. Common optimizations and immediate-edit learning are shared with controlled library baselines. A separate ablation disables edit learning to identify its effect.

### 5.4 Local proposal policy: exact information boundary

One shared small policy evaluates local opportunities; do not give each primitive a large controller. A pilot network is `88 -> 32 -> 32 -> 3` with tanh hidden activations: 4,003 trainable scalars. Its three heads predict immediate fit benefit, future transfer benefit, and log evaluation cost. Output heads are not a language model or an unrestricted graph decoder.

For each node, form a 24-value feature vector: operation-family one-hot (nine base families plus `CALL`), output-type one-hot (2), two input-type one-hots (4, masked for missing inputs), in/out degree (2), activation mean/variance (2), mean absolute/rms support gradient with respect to the node output (2), saturation fraction (1), and parameter rms (1). Fixed casts/terminals use a declared mapping/mask, never a task-dependent code. For `R`, saturation means `abs(value)>0.95`; for `P`, value below 0.025 or above 0.975. These thresholds are diagnostic feature definitions, not success thresholds.

Collect these statistics after a support-loss backward pass on the current donor. Retain gradients with respect to intermediate outputs during that pass, then detach all feature values. Never store the autograd graph in the archive. Absent gradients, missing input features, and absent numerical slots contribute zero under a fixed public convention; the no-gradient ablation zeros all gradient features. Library weights remain frozen, while derivatives through a call with respect to its signal inputs are still available. Gate-query losses are never the source of local features.

A local neighborhood contains the root and up to two predecessor hops plus its immediate consumer, capped at eight nodes with a deterministic canonical-order tie break. Concatenate root features (24), masked neighborhood mean (24), masked maximum (24), and 16 candidate features: eight action-family indicators, `R/P` boundary-port counts, expanded size, dependency generation, mutable parameter count, storage estimate, donor age, and prior occurrence count. Values are normalized using running meta-training statistics; freeze those statistics for protected adaptation. Task IDs, global target descriptions, hidden teacher graphs, query labels, global archive embeddings, and full-corpus attention are prohibited.

Legal action generation is generic and deterministic conditional on its RNG. For consolidation, enumerate or sample connected backward cuts within this neighborhood, requiring at least two elementary operations and at most three external signal ports. The pilot caps candidate bodies at eight expanded operations. Later generations may contain earlier calls, but still obey the declared expanded-size cap. Before selecting final caps, compute a public worst-case representability bound from the generator specification without inspecting protected task realizations. A higher-level target may exceed the eight-operation pilot candidate cap; the 32-operation scratch cap may also be inadequate. Raise the neighborhood/body cap when development diagnostics show it cannot express the tested hierarchy, and allow all controlled primitive-IR methods the same expanded-size envelope. Record this change before confirmation; a pilot with inadequate caps cannot confirm or falsify H4.

The local policy **ranks legal candidates**. It does not receive a precomputed correct cut from the benchmark. A uniform root reservoir prevents processing only hand-selected successful regions. Include successful and unsuccessful donor attempts in the candidate-label dataset. The runtime supplies legality, types, reference management, and resource counters, not the right architecture.

### 5.5 Learning proposal values without unrestricted long-horizon RL

Let a paired immediate-edit trial fit two clones of the same incumbent for the same additional fit allowance: one unchanged and one edited. The label is `Δ_edit = L_unchanged - L_edited` on support-only selection data. At meta-training time, an inner split may be used; all examples remain within the common training allowance. This pairing avoids crediting an edit for gradient steps that the unchanged graph did not receive.

For consolidation candidate `c`, obtain `Δ_transfer` from the gate below. Log the candidate features **before** seeing its label. Fit the corresponding value head by Huber regression on standardized labels; fit the log-cost head to measured gate seconds. Replay a bounded reservoir of all evaluated candidates, including rejected ones. Log proposal probabilities. Evaluation subsets include uniformly sampled candidates so that apparent policy quality is not inferred only from its own selected positives. Sampling weights and any importance correction must be frozen and documented; the initial method does not claim unbiased off-policy policy-gradient estimation.

Select candidates through an epsilon mixture of uniform selection and softmax ranking by predicted positive transfer divided by predicted gate cost. Clamp predicted costs below a measured positive timing floor, and do not divide by arbitrarily tiny predictions. A separate random-evaluation quota supplies labels when all predictions are nonpositive. The no-policy baseline uses the identical legal proposal generator with uniform ranking. The global-information control replaces only the feature interface, retaining a size/cost-matched model where possible and reporting its additional input-encoding expense.

Train these policies only during within-world meta-training. During protected task adaptation, the persistent policy and library are frozen; task-owned graph and parameter fitting remain allowed. An online-continual variant that updates policies on new task support is a separate experiment with its own information contract.

### 5.6 Transfer gate and library maintenance

A candidate goes through four steps:

1. **Legality and exactness:** determine its typed boundary, build an exact snapshot from donor computation, test expansion equivalence, and reject duplicates with the same version hash. Do not fit it using hidden benchmark labels.
2. **Optional parameter fitting:** fit task-owned numerical parameters on donor training support only, then freeze the proposed snapshot. A refined snapshot is a distinct candidate with charged cost.
3. **Paired transfer panel:** draw previously unused meta-training tasks from the current permitted training composition strata. For each task, clone the same persistent solver state and RNG streams; adapt once with `L` and once with `L ∪ {c}` at the same support size and adaptation budget. Neither adaptation sees query labels. The gate evaluator computes their bounded query-error difference. Candidate/control random streams are indexed by event, not assumed to stay identical after divergent control flow.
4. **Acceptance:** accept only if average panel benefit is positive, at least two distinct task IDs benefit and actually invoke the candidate, and the configured storage/expanded-execution constraints are respected. Sort ties by lower total storage and then deterministic hash. Recheck the resource envelope before committing the version.

Start with a four-task gate panel for a mechanics pilot; use development variance and total gate expense to choose the final panel size. Tasks must be separate from donor fitting tasks, not merely different examples of the same target. Do not require each accepted entry to pass a conventional significance test: repeated candidate selection makes that interpretation false without additional correction. Acceptance is a **learned algorithm's heuristic**; its whole future performance is tested on untouched worlds. Retain panel results and all rejected candidates in the log.

Accept entries sequentially. Once the library changes, cached gate utility for another candidate is conditional on an obsolete library; invalidate or remeasure it. Candidate interactions are not additive. Stop accepting when the meta-training budget or the callable/storage cap is reached. For the core comparison, use a bounded append-only callable library rather than inventing a complex retirement policy. A later maintenance experiment may retire zero-use or harmful entries on fresh training panels, but keep dependency versions on disk. Storage savings require dependency-aware garbage collection and recovery tests.

All gate tasks belong to the **meta-training data allowance**, even though they have internal support/query partitions. Fixed-module and dense baselines may use the same labeled allowance for meta-learning rather than for gates. Each method receives the same allowed task/example pool and time allowance, not additional private supervision for SIML. Gate labels never come from protected development or confirmation evaluation tasks.

### 5.7 Main loop pseudocode

```text
META_TRAIN(method, world_training_access, budget, config):
    initialize persistent state, RNG streams, empty immutable library
    initialize numerical op priors and small policies as configured
    while budget permits the next atomic task:
        task = world_training_access.next_donor_task()
        model, traces = ADAPT(state.snapshot(), task.support, task_budget)
        archive typed graphs and bounded detached local statistics
        update numerical initializations / immediate-edit policy if enabled
        if scheduled consolidation and sufficient budget remains:
            candidates = method.propose(archive_or_local_stream, state)
            for candidate in acquisition_order(candidates):
                validate types, exactness, versions, and expanded costs
                if valid and affordable:
                    panel = world_training_access.next_unused_gate_panel()
                    outcomes = PAIRED_GATE(state, candidate, panel)
                    update transfer/cost policy using pre-outcome features
                    if acceptance_rule(outcomes, candidate, state):
                        atomically append immutable version and update index
                        invalidate utilities conditioned on old library
        record cost counters; atomically checkpoint at a safe boundary
    return frozen snapshot

CONFIRM(snapshot, task_support, evaluator, frozen_config):
    for k in frozen_config.support_sizes:
        state = snapshot.read_only_clone()
        model = ADAPT(state, task_support.prefix(k), frozen_config.adapt_budget)
        commit predictions, hashes, timings, and resource record
        evaluator scores predictions without returning query labels to ADAPT
    do not update persistent snapshot or carry task-specific state to next task
```

A gate runs the same adaptation procedure as testing, except that its tasks are explicitly training data. Do not substitute an oracle macro matcher, a correct-program enumerator, or full future task knowledge inside the gate.

### 5.8 The minimum discriminating experiment

Use a shared archive of **learner-produced**, not teacher-produced, short differentiable circuits from development training tasks. Freeze the donor archive, execution backend, numerical snapshots, fit schedule, and gate task panels. Compare local learned ranking, uniform local ranking, frequency ranking, native Stitch candidates, and global ranking. Evaluate their selected libraries on new tasks under equal discovery-plus-adaptation cost.

This experiment separates candidate quality from feedback through different donor solvers. It can falsify H1/H3 cheaply. It does **not** demonstrate autonomous end-to-end development: the subsequent full loop must acquire its own donor graphs and pay for them. A positive offline result is an entry condition for the full experiment, not its replacement. An inconclusive offline result with poor candidate coverage justifies improving the proposal interface only when diagnostics identify a missing expressible opportunity.

## 6. Data and task construction

### 6.1 Data firewall and world separation

Create three nonoverlapping sets of **world seeds**: exploratory, pilot/calibration, and protected confirmation. A world includes separate donor tasks, fresh gate tasks, task-level development probes, and final tasks. Exploratory worlds may be inspected fully. Protected worlds are generated only after the protocol is frozen; the training process receives only their allowed training stream. The same procedure and exposure apply to all methods.

Place hidden teacher definitions and final query labels in an evaluator-only directory/process. The `TaskSupport` object contains only numerical support, public signal types, and opaque IDs. It must not contain a deserializable teacher, motif identifier, target length, correct source code, or a path that reveals the target. A coding agent can inspect generators while developing software, but learned models and search routines cannot read hidden programs at runtime. This is an experimental information boundary, not an adversarial security guarantee against the human experimenter.

Use independent RNG streams for world structure, numerical parameters, support examples, query examples, candidate search, policy initialization, and timing order. Do not construct test seeds as trivially reused training seeds. Store cryptographic hashes of manifests and use exact tuple membership to verify separation. A test label accessor called from a method should raise an exception and create an audit event.

### 6.2 G1: hidden reusable differentiable motifs — primary controlled workload

**Purpose:** test whether experienced computations become useful building blocks for novel combinations. This is deliberately synthetic and aligned with the primitive language. Its conclusions are conditional on that alignment.

The pilot uses `d=6`, eight level-zero latent motifs and four level-one latent motifs. These are **generator objects**, not modules supplied to any method. Each level-zero motif is a typed random expression with two `R` inputs, `R` output, and two to four nonterminal operations from Section 5.1. Draw internal raw parameters independently from `Normal(0,0.5²)`. Require both inputs to affect its output on independent diagnostic probes. Construct each level-one motif from two previously generated level-zero motifs and an ordinary typed coupling operation, with two or three boundary inputs. The generator may share fixed motif parameters across task occurrences, but supplies no intermediate supervision.

Generate trees using a size-conditioned typed grammar: enumerate feasible operator/root signatures for the remaining size and desired output type, sample a legal root uniformly, sample a legal partition of remaining size over children, and recurse. Resolve signature infeasibility by rejecting that generation attempt, not by inserting a desired motif. Keep all generator rejection counts. The exact grammar and parameter distributions are part of the dataset version.

A task embeds one or more latent motifs in different input-coordinate roles and in a short randomly generated outer context. Contexts draw fresh outer parameters; shared internal motif parameters remain fixed within a world. Input coordinates are sampled without replacement where an interface permits it; repeated coordinates are an explicitly separate condition. This separates reuse from memorization of a whole numerical mapping.

Use three curriculum phases: elementary/short context tasks; tasks exposing recurring level-zero and level-one behavior; and tasks composing familiar level-one motifs in permitted combinations. The minimum exposure count is determined by motif/context coverage, not by whether SIML accepts a circuit. Pilot start: at least four distinct contexts per latent motif, with gates drawn from additional contexts. Increase exposures if competent baselines cannot learn constituent behavior; do not inspect protected scores to decide this.

**Combination split:** with four level-one motifs, choose a world-specific random permutation. Directed outer-composition edges of offset `+1 mod 4` are allowed in donor/gate tasks; offset `-1 mod 4` provides task-level development probes; offset `+2 mod 4` is protected within-world combination holdout. Identity/self-compositions form a separately reported stratum. For larger motif dictionaries, partition ordered pairs by a fixed salted hash while ensuring every motif appears in training in both roles. The final generator must reject any target whose prohibited outer adjacency leaks through a training template. Inspect fully expanded structures for accidental exact target overlap as an additional check.

Protected tasks include new contexts on unseen allowed-edge instances, held-out ordered combinations, and greater outer composition depth. The initial depth range is one/two outer motifs in development and three/four in depth extrapolation, subject to numerical sensitivity checks. Core graph limits apply to **expanded operations**, never only macro count, and must accommodate these functions for all methods.

Sample clean inputs independently from `Uniform([-1,1]^6)`. Generate a diagnostic probe set separately from training/query data. Public, method-independent exclusions: nonfinite output; output variance below `1e-4`; or absence of observable dependence on a declared input under coordinate perturbations at numerical tolerance. The variance cutoff removes nearly constant tasks that do not test structural learning; it changes the evaluation population and must be reported. Repeat conclusions with lower cutoffs on development data. Never exclude a task because a method fails on it. If most deep tasks are rejected, the grammar is an unsuitable depth proxy, not evidence of SIML failure.

Record: world seed, hidden motif/parameter hashes, task skeleton hash, full expanded structure hash, exact parameter hash, input roles, context family, expanded depth/size, split reason, exclusion reason, and diagnostic output variance. The evaluator retains hidden fields; methods receive only public support objects.

**Pilot data sizes:** donor support 128 examples/task; gate adaptation 16 examples and gate scoring 128 independent queries/task; final support prefixes `[4,8,16,32,64]` and an initial query count of 1,024. These are starting values. Set final counts using within-task query noise, between-task variance, grammar coverage, and measured cost before confirmation. Every label used in donor fitting, gates, or policy training is included in the per-world training-data allowance. Unused training labels remain available to competing methods within their budgets.

### 6.3 G2: independent teacher construction — required scope check

Construct latent functions from tiny dense neural teachers rather than from the SIML grammar:

\[
m(z)=\tanh\bigl(w_2^\top\tanh(W_1z+b_1)+b_2\bigr),
\]

with two input values, hidden width four, weights `Normal(0,1/fan_in)`, and biases `Normal(0,0.25²)`. Interpret the second argument as variance; implement standard deviation `sqrt(1/fan_in)`. Teachers are frozen within a world. Apply the same independent dependence/variance screening, context sampling, and support/query protocol. Hidden teacher functions are not available to any learner.

Two experiments answer different questions. **Within-G2 library learning** trains a new persistent state on G2 training tasks and tests G2 compositions; it tests whether success depends on exact native-grammar generation. **Strict G1-to-G2 transfer** freezes a G1-trained library and adapts only task-owned state on new G2 tasks; it tests portability and may reasonably show negative transfer. Never describe retraining on G2 as zero-shot library transfer.

G2 is still synthetic, and its numerical representability differs from G1. Test known-topology approximation capacity for each architecture and include fixed neural modules. If the SIML vocabulary cannot approximate G2 constituent behavior even with ample task-owned fitting, a failure here bounds the vocabulary rather than disproving all reusable abstraction mechanisms. Changing the vocabulary becomes a new, explicitly documented experiment.

### 6.4 G3: simulated scientific functions — evidence-triggered extension

Use this only after a core mechanism result or a clear need to test application scope. No external dataset is required. Define small vector-field regression tasks directly from equations, with independently sampled coefficients in `[0.5,1.5]`:

- predator–prey-form fields: `f1(u,v)=a*u-b*u*v`, `f2(u,v)=d*u*v-c*v`, on `[0,1]^2`, divided by 3 for the bounded target convention;
- oscillator-form fields: `f1(u,v)=v`, `f2(u,v)=-delta*v-alpha*u-beta*u^3`, on `[-1,1]^2`, with the second output divided by 4.5.

Treat each scalar component as a task; do not supply the equation family label to the learner. Coefficient combinations and input regions are held out by manifest, not by random row splitting alone. This is regression of simulated field values, not discovery from measured physical trajectories. It does not test stateful SIML or causal identification.

Mandatory simple comparator here: ordinary ridge regression on all monomials up to total degree three, including the constant, with regularization chosen on development data. These fields lie in that feature space; a failure to match this simple method limits the application case. Add PySR and, when licensing permits, FePySR. Do not select equations after observing a favorable SIML result.

ODEBench was inspected as a possible broader source, but its usable license was not established in this investigation [R26]. The core plan does not depend on it. Importing its data/code is blocked until permissions and provenance are resolved. Independently implementing explicitly specified mathematical functions avoids that dependency; it does not turn synthetic data into real-world evidence.

### 6.5 Required negative and boundary populations

Generate no-reuse worlds by redrawing latent motif parameters/structures independently for every task; predict that unnecessary consolidation should not help. Generate vocabulary-distraction conditions by adding legal, irrelevant library entries with matched sizes. Include reversed task order, a distribution change that makes earlier motifs unhelpful, observation noise, and input-role permutation. For noise, add the declared noise before any common output clipping and preserve the unclipped values for a secondary metric.

Use the same public transformation for every method. A benchmark that defeats only the fixed neural model, or a teacher that hands one method the desired internal boundaries, is not an acceptable test.

## 7. Baselines and confirmatory evaluation

### 7.1 Comparison tiers

**Mandatory controlled competitors:** primitive-only graph learning, a strong fixed modular learner, native Stitch topology/parameter mining with its usual compression objective, and Stitch candidates with the same transfer gate as SIML. Dense/meta-learned networks and growth controls are also required before a broad architecture claim. PySR is a mandatory numerical reference where its operator space is suitable. The research pipeline should implement these in risk order, not demand full-scale execution of every historical paper.

**Broader references:** original DreamCoder, HOUDINI, LMC, AbstractBeam, AMaze, and recurrent compositional systems. Their published scores are not numbers to place in a controlled SIML table. A faithful native-domain sanity run is valuable where affordable; a domain port must be labeled as an adaptation. The AbstractBeam configuration inspected here includes two GPUs, 30 data-generation processes, and a large training-step allowance; reducing that configuration is a new scale, not a reproduction of its published experiment [R07].

### 7.2 Exact local baseline specifications

| Method ID | Implementation and permitted information | Development tuning and reproduction checks |
|---|---|---|
| `primitive_only` | Same typed executor, graph search, op initializations, immediate-edit policy, archive allowance, and task-local fitting; never adds callable abstractions. May replay whole prior solutions as initial candidates. | Tune search/fitting jointly; verify it solves known simple tasks and uses all available training budget. Separately compare with no replay. |
| `dense_reptile` | Tanh MLP with bounded output, trained from scratch and with first-order meta-learned initialization. Reptile update is `ψ <- ψ + ε*(θ_adapted-ψ)` over training tasks [R28]. | Pilot depth 2–4, width 32–128; expand range if the best is on a boundary and memory permits. Tune inner steps/rate, outer rate, regularization, and restarts. Compare ordinary joint pretraining too. |
| `fixed_modular` | BounceGrad-style learned library with a fixed count and fixed small neural module architectures; task-specific typed compositions and optional numerical fine-tuning. Modules are not fixed random functions [R05]. | Pilot 4/8/16 modules, hidden widths 4/8/16; `R,R->R` and `R->P` signatures, common composition solver. Alternate simulated-annealing structural choices and gradient updates on training-task query partitions. At test, use support only. Check that shared useful modules are learned on a known-module positive control. |
| `stitch_mdl` | Native Stitch candidate extraction from the method's learner-produced archive; add compression-selected templates/versions under the same library envelope. | Reproduce native documented compression/rewrite examples. Tune arity, iterations, candidate counts, extraction cadence, and appropriate code-length costs. Both parameter-aware and topology-first encodings must be tried on development data. |
| `stitch_transfer` | Same native Stitch candidate pool mechanism but the **identical** fit, paired transfer gate, specialization choices, and acceptance criterion as SIML. | Pay extraction and every rejected gate cost. This is the main centralized comparison; choose its strongest encoding on development only. |
| `grow_no_library` | Permit equivalent scratch-graph/parameter growth and numerical initialization learning, with donor/replay storage, but no persistent subgraph calls. | Match parameter/storage envelopes or show a frontier. Full solved graphs can seed later search, making this a strong warm-start rather than an artificially forgetful control. |
| `sparse_rewire` | Dense-matrix mask implementation of periodic small-weight removal and gradient-informed regrowth at a fixed nonzero budget, using the same MLP base. | Check degree/nonzero accounting; tune density and update interval; retain fixed-mask version. Label a RigL-style numerical adaptation, not native sparse-kernel speed reproduction [R18]. |
| `constructive` | Cascade-Correlation-style incremental hidden units: fit a candidate to residual correlation, freeze incoming weights on acceptance, refit readout [R17]. | Unit tests for residual objective and freezing; tune candidate pool and growth penalty. Report this as the specified small numerical implementation, not the original benchmark reproduction. |
| `pysr` | Native PySR evolutionary expression search and constant fitting; no hidden teacher code. Allow its strongest reasonable numerical grammar and public support-only selection [R23]. | Fix repository/package revision after local discovery. Start with `+,-,*,tanh` and constants; permit a stable sigmoid implementation for G1, and development-justified additional functions for G2. Tune native populations/complexity/cost settings using documented interfaces. Include Julia startup/precompilation costs separately and in cold totals. |

Give `fixed_modular` access to the same base primitive vocabulary as well as its learned fixed modules; otherwise a deliberately incomplete module bank would be an unfair competitor. The expanded-primitive cap is a controlled-IR constraint, not a fiction that a dense neural module executes in one arithmetic operation. Neural baselines receive sufficient compositional depth for the declared target scope and are constrained by measured time, memory, and their recorded parameter/operation costs. Report neural multiply-add conventions separately rather than equating them with primitive-call counts.

For fixed modular learning, implement the alternating procedure explicitly: maintain a composition for each training task; propose a legal module/connection change; accept based on training-support score and an annealing temperature; update the shared module weights using that task's disjoint training-query partition. At test, neither task identity nor training-query access is available. Both frozen shared-module selection and task-local fine-tuning are valid development candidates. This captures the relevant mechanism without pretending the numerical benchmark is an unchanged original experiment.

For PySR, the symbolic hypothesis space and optimizer differ from SIML. This is a **practical competitor**, not a parameter-by-parameter causal control. Common bounded predictions are used for the primary metric, and unclipped errors are reported too. Nonfinite symbolic predictions are failures, not values to hide by clipping. Its unused meta-training time is not artificially spent; include its lower preparation expense in the amortized comparison.

FePySR is a particularly relevant 2026 neural-feature-discovery competitor [R29]. Its repository is publicly readable but declares a noncommercial license. First determine whether the intended use is permitted; do not silently classify it as unrestricted open source or copy it into an incompatible project. Where permitted, run native feature extraction plus PySR on the same data/budget, with its full extraction expense and development tuning. Otherwise retain it as an explicit unrun competitor and restrict any “best numerical method” claim. A documented clean reimplementation of its described algorithm would be an adaptation requiring separate reproduction checks, not an excuse to ignore the closest result.

### 7.3 Stitch adapter: prevent an artificial disadvantage

An important fairness hazard is that independently fitted numerical constants destroy exact textual repetition. Do not serialize arbitrary floating-point values as unrelated opaque operators and then claim native compression cannot find structure.

Implement **two** public, generic encodings, selected on development data:

1. **Literal encoding:** ordinary expression structure and explicit parameter leaves, allowing native abstraction over such leaves. Preserve the distinction between signal ports and parameter arguments when rebuilding the typed circuit. Set and report arity limits honestly; do not count a lifted parameter as free memory.
2. **Topology-first encoding:** serialize operation/type structure while keeping each occurrence's numerical slots in a separate decorated tree. Native Stitch mines structure without seeing numeric values. For every returned abstraction, structurally align each occurrence with its original decorated tree, recover its complete numeric snapshot, and propose those donor-derived versions through the same gate. Reject a reconstruction that is ambiguous or not exact. Keep structure-only templates distinct from accepted numerical versions.

The second encoding is an explicit adaptation of Stitch, not a claim that its compressed topology is already a behaviorally stable module. It prevents a nuisance encoding choice from handicapping the baseline. Count the template, per-version numerical storage, mapping, and reconstruction work. Test extracted version outputs and input derivatives against their donor subtrees. For parameter-derivative equivalence tests only, create temporary differentiable clones of both snapshots with the same parameter-identity mapping. Ordinary frozen-library execution does not update snapshot parameters. Never choose the best donor version using protected labels.

Use the same donor frontier size and retention allowance in the common-donor diagnostic. In the full loop, methods may acquire different archives because their learning differs; archive acquisition cost remains part of the method. Supply generic algebraic normalization only if it is verified, available to every method, and paid for. A rewrite bug that breaks a baseline is a blocking correctness failure, not a SIML win.

The inspected binding exposes `stitch_core.compress(programs, iterations=..., max_arity=...)`, abstractions with `name/body/arity`, and a `rewrite` function [R03]. Pin and smoke-test the actual installed version before relying on that API. Native documentation also supplies `cargo run --release --bin=compress -- data/basic/ex1.json --max-arity=3 --iterations=1`; this is an existing upstream example, not a proposed SIML command.

### 7.4 Resource and information fairness

Two regimes are required.

**Controlled regime:** give each method the same within-world labeled task/example allowance, the same measured meta-training wall-time ceiling `B_meta`, the same peak-memory envelope, and the same task-adaptation ceiling `b` at each support size. All candidate generation, failed fits, policy learning, validation, archive processing, and checkpoint activity during method execution are counted. A baseline may use its allowance for ordinary meta-learning rather than gates. Materialize the allowed task/example pool and its phase boundaries before each method run; gate allocation cannot expose a later curriculum phase early. Each method may use the same union of labels available in its current phase, subject to its time budget. Report actual usage, not just the ceiling. All methods may retain their best valid incumbent.

**Amortized regime:** allow different preparation times and report total cost for serving a sequence of new tasks at common quality, or the full quality/cost frontier when common quality is not reached. Reusable public software installation and project-specific compilation are disclosed; method-specific learned assets count as preparation. Do not grant an externally pretrained search model free learning while charging all SIML learning, or erase its original pretraining requirement.

Record common task generation and evaluator-only scoring as shared experimental infrastructure expense, separately from method-specific expense; never charge that common work selectively to one method. Include it in the full project-resource forecast. Method-specific data preparation, feature extraction, validation inference, and archive construction remain charged to the corresponding method.

Match exact expanded primitives and parameterization in causal graph comparisons. For neural and symbolic competitors, parameter equality may be inappropriate; show parameter/storage/latency frontiers within the same physical envelope. A larger dense baseline is necessary if growth explains an improvement. All generic optimizations, caching of execution tapes, vectorization, and numerical initialization improvements should be shared where meaningful.

Tuning uses only development/pilot worlds and a recorded equal wall-time allocation per method family. Use random or successive-halving search over the declared ranges, not many manual SIML trials and one baseline default. When a baseline's optimum is at a search boundary or its reproduction test fails, expand/repair **before** final freeze. Retain every trial and failure. Add baseline-specific strong settings from its native documentation rather than forcing an identical optimizer where that would weaken it.

### 7.5 Primary estimand and metrics

For each query set, clip finite predictions to `[-1,1]` for the common bounded evaluation; all methods use the same rule. Define

\[
e_{wtmk}=\frac{1}{4|Q_t|}\sum_{i\in Q_t}(\hat y_i-y_i)^2\in[0,1].
\]

Nonfinite/invalid-model failures receive error 1 and a failure flag. Report raw unclipped MSE and clipping fraction as guardrails. This policy must be frozen before confirmation.

With support sizes `k_j`, define normalized log-support area under the learning curve:

\[
A_{wtm}=\frac{1}{\log k_J-\log k_1}
\sum_{j=1}^{J-1}\frac{e_{wtmk_j}+e_{wtmk_{j+1}}}{2}
(\log k_{j+1}-\log k_j).
\]

Lower is better. Reset the frozen persistent snapshot independently for every task/support-size run. Use nested support prefixes and common queries across methods, but no cross-task adaptation leakage. Set the same adaptation ceiling `b` at each support size in the primary regime; this measures sample efficiency under a fixed per-task compute allowance. An additional time-curve experiment varies `b` separately.

Average tasks within predefined strata, then strata with fixed equal weights, then learner seeds within a world. Do not let a stratum with more generated variants dominate unintentionally. For comparator `c`, define the paired world effect

\[
D_{wc}=A_{wc}-A_{w,\mathrm{SIML}}.
\]

The **primary estimand** is the mean of `D_wc` across independently sampled worlds for held-out G1 combinations at the frozen resource setting. Positive means lower SIML error. Report the absolute bounded-error difference, corresponding MSE difference (`4D`), and relative reduction only when the comparator error is not approximately zero. G1 depth extrapolation, G2, and cumulative tests are separately labeled secondary or confirmatory extensions.

Important secondary metrics: error versus adaptation seconds; training+search+gate seconds; unique labeled examples; expanded primitive evaluations; active/trainable/stored parameter counts; library and archive bytes; maximum process/unified-memory use; inference median and tail latency; solver failure rate; task worst-decile error; retention; and break-even horizon. Module usage, generation, compression ratio, and proposal hit rate are mechanism diagnostics, not substitute success metrics.

### 7.6 Calibration, sample size, and inference

The independent unit is a **world**, including its independently learned library. Hundreds of tasks sharing one library are not hundreds of independent replications. Multiple learner seeds within a world quantify training variability and are averaged or modeled as nested observations. Repeated timings are measurement repeats, not independent learning experiments.

Use pilot worlds excluded from confirmation to estimate between-world variance, within-world task variation, training-seed variation, and query Monte Carlo error. A pilot starts with a few independently generated worlds and grows only to estimate these quantities reliably; no significance claim is made from that pilot. Allocate further work to the largest uncertainty contributor rather than automatically adding tasks to the same library.

Choose a target confidence-interval half-width `h` in bounded-error units from the scientific resolution needed to distinguish practically different error/cost frontiers. Document why that resolution is useful. It is **not** a minimum effect that SIML must exceed. An initial approximation is

\[
W\approx (z_{1-\alpha/(2C)}\,s_D/h)^2,
\]

where `C` is the number of declared primary comparisons. Use a conservative variance estimate and then simulate the planned paired/nested design using pilot residuals to check precision and coverage. Alternatively calculate power for a separately justified effect using `(z_{1-alpha/(2C)}+z_power)^2*s_D²/delta²`. Do not size the study around an optimistically selected pilot effect. Set final world/task/seed/query counts and the sample-size rule **before** protected results exist.

Use paired world-level mean differences with Student-t intervals as a transparent primary analysis. For simultaneous primary intervals, use a Bonferroni-adjusted confidence level; report a world-cluster bootstrap sensitivity analysis that resamples whole worlds and preserves pairings. If pilot world effects are strongly skewed or too few worlds permit credible inference, increase worlds or preregister a robust bootstrap-based primary analysis before confirmation. Do not switch tests after seeing which gives a positive result. Timing uncertainty can use paired block medians/bootstraps separately.

The mandatory primary comparator set contains `stitch_transfer`, `fixed_modular`, and the strongest remaining locally validated development competitor. If the third is already one of those, use the strongest distinct competitor. Select configurations using development results only; retain all baseline outcomes in the report. To claim superiority over this set, each simultaneous interval must exclude zero in SIML's favor, and material regressions must be addressed. A single favorable pairwise comparison does not establish “strongest-baseline superiority.”

Freeze a fixed final `W`. No repeated peeking and continuing until significance. If local cost cannot achieve the desired precision, report the achievable interval and classify the result as inconclusive when appropriate. A larger later confirmation requires a fresh registered batch with an appropriate combined-analysis rule, not recycling exploratory discoveries as an untouched test.

### 7.7 Preconfirmation freeze and harness validation

Before opening final scores, commit `protocol/confirmatory.yaml`, data-generator revision, split hashes, methods, configurations, comparator selection, budgets, hardware/backend settings, primary strata/weights, analysis code, failure handling, and run order. Write an explicit freeze event and hash. Generate protected seeds afterward, or have a separate evaluator retain them unopened until this point.

The harness must pass: identical methods under different labels produce no systematic effect; known wrapped and expanded executions agree; deliberately degraded predictions produce a detectable difference; an oracle useful-library diagnostic can improve at least one nontrivial budget/depth setting; test labels cannot influence adaptation or gate decisions; and restart ordering cannot change which task enters which split. The oracle is a sensitivity check, not a baseline eligible for superiority claims.

Protected adaptation can use its supplied support labels only. It cannot request more labels, early-stop on queries, choose among seeds using queries, or select a symbolic expression using final error. Holdout-based selection must use a subset of the supplied support, with its cost and reduced fitting set counted; fixed development-selected schedules are the simplest core choice.

## 8. M3 Max feasibility and hardware-adaptive execution

### 8.1 Feasibility assessment and blocking unknowns

A meaningful local experiment should be possible because the core numerical state is small and graph search can run on the CPU. That is an architectural estimate, not a measured M3 result. The principal risks are combinatorial search, validation expense, interpreter/dispatch overhead, and the cost of statistically adequate independent worlds—not billions of weights.

The actual unified memory, free storage, macOS release, Python architecture, and supported kernels are unknown. No assumption about a maximum-memory M3 Max is allowed. Resolve these before choosing experiment scale or downloading large optional assets.

Apple's inspected PyTorch page advertised a stable 2.11.0 configuration with macOS 14+ and Python 3.10+, while the PyTorch stable documentation inspected during this research resolved to version 2.14 [R24, R25]. These are not interchangeable version pins. Select an actual supported arm64 wheel for the installed OS/Python, validate it, and lock the exact resolved version. Do not install a nightly merely to match a page label or assume a newest wheel supports an older macOS.

### 8.2 Software stack and setup discovery

Core: native arm64 Python virtual environment; PyTorch for numerical execution/autograd; NumPy for immutable arrays; SciPy for statistical routines if needed; pytest for correctness tests; psutil for memory/process observations; YAML or JSON configuration; standard-library JSONL/SQLite for records. Rust/native Stitch or its verified binding is required for the compression baseline. Julia/PySR is a separate environment if dependency isolation requires it. No pretrained model, remote GPU, paid API, torchvision, CUDA, or LLM is required.

The following are **existing system commands**, not project commands:

```bash
uname -m
sw_vers
sysctl -n hw.memsize
sysctl -n hw.logicalcpu
df -h .
xcode-select -p
python3 --version
```

Record OS build, CPU/GPU model, physical memory in bytes, free disk bytes, power mode, process architecture, and package/driver information. Avoid storing the laptop's serial number in public artifacts. Check that Python and native dependencies are arm64 rather than silently running under translation.

After selecting the compatible Python installation, use an isolated environment. Resolve dependencies once, run tests, then save exact package versions and hashes in a lock/manifest. Pin source dependencies to recorded commits and preserve their licenses. A usable manifest records package name, version, source URL, commit where relevant, wheel/source hash, license, and the smoke test performed. No exact commits were obtained in this research; confirmation must fail to start until they are recorded locally.

Use the native Stitch release build and documented tests. Its MIT license, the modular meta-learning repository's MIT license, AbstractBeam's Apache-2.0 license, and PySR's Apache-2.0 license were visible in inspected sources [R03, R05, R07, R23]. Verify the downloaded revisions and dependency licenses again. FePySR's noncommercial restriction and ODEBench's unresolved permission are separate gates [R29, R26]. Do not infer a data license from a paper's availability.

### 8.3 Backend selection and faithful fallbacks

Implement all core primitive operations and their backward paths on CPU float32, with CPU float64 as a correctness reference. Test MPS explicitly for tensor creation, indexing/gather used by the tape, tanh, sigmoid, multiplication, matrix multiplication used by baselines/policies, loss/backward, optimizer updates, and serialization round trips. Verify actual gradient parity on random and adversarial depth cases. Test whole fitting/search loops; individual operation support is insufficient.

MPS is optional. For tiny scalar graphs, dispatch cost may outweigh acceleration; decide using measured end-to-end candidate fitting and adaptation, not a large-matrix benchmark. Keep discrete topology management on CPU. Group graphs of the same topology, keep examples on device during fitting, and avoid per-node `.item()`/host transfers.

The primary accelerated path uses float32 without fast-math or reduced precision. Unsupported MPS operations should be detected with automatic fallback disabled during the smoke test. The documented `PYTORCH_ENABLE_MPS_FALLBACK` mechanism can be evaluated as a separately measured hybrid path, but it must not conceal CPU execution [R25]. The simplest faithful fallback is to run the same graph and learning semantics entirely on CPU. Do not replace the mechanism with a smaller unrelated network, a dense relaxation, or a CUDA-only dependency merely to claim acceleration.

Do not depend on dynamic `torch.compile`, sparse MPS kernels, or a particular fused optimizer. They are optional optimizations requiring their own semantics and end-to-end timing checks. Re-run controlled baselines with any generally useful acceleration. If a critical operation has no faithful local execution path, block the dependent experiment and report that limitation rather than substituting silently.

### 8.4 Memory model

Measure the interpreter/framework baseline and fit a peak-memory model from pilot blocks. A starting accounting model is

\[
M_{\rm peak}\approx M_{\rm runtime}
+16P_{\rm active}+4P_{\rm frozen}
+c_{\rm save}\,4CBNv
+M_{\rm candidate\ outputs}
+M_{\rm graph/archive}
+M_{\rm compiler/cache}
+M_{\rm temporary}.
\]

Here `P_active` counts float32 parameters requiring gradients plus Adam's two moments (16 bytes/parameter including the parameter and gradient, before extra temporaries); `P_frozen` is frozen numerical storage; `C` is simultaneously fitted candidates; `B` is examples; `N` is expanded operations; `v=1` initially; and `c_save` is a measured autograd/storage multiplier. Sequential fitting reduces `C` without changing the candidate population. Never assume all branches of a large masked supernetwork cost only their active slots.

For illustration, `C=16, B=128, N=32, v=1` gives 262,144 bytes for one raw float32 activation array; autograd, topology objects, and runtime allocations are additional. One million active parameters would require about 16 MB for the listed Adam state, not including activations. These are arithmetic estimates, not predicted whole-process peaks. Imported runtimes and Julia/compiler processes can dominate small models and must be measured.

Record packed tensor sizes and actual resident/unified-memory behavior. On Apple unified memory, CPU RSS and GPU allocation counters can overlap; do not sum them and call that physical usage. PyTorch exposes MPS allocation/driver counters, which should be retained alongside process and system memory-pressure observations [R25]. Multi-process totals require attention to shared pages.

Start with an operational reserve `R=max(6 GiB, 0.2*M_physical)` and a conservative cap no greater than `min(M_physical-R, 0.75*initial_available_memory)`. These are safety margins for calibration, not scientific constraints on model capacity. Validate them under representative thermal/load conditions and record amendments. Keep enough disk for the projected archive, dependency caches, raw results, and at least two checkpoints; the initial requirement is derived from measured file growth rather than an arbitrary fixed download quota.

If memory pressure or swap growth appears, first reduce simultaneous candidate batches, use sequential fitting, checkpoint/recompute activations, and evict recomputable execution tapes. Keep example counts, effective optimizer batch, graph semantics, and candidate population fixed. Changing the scientific population size, graph cap, precision, or module capacity requires a declared new configuration. Apply equivalent operational changes to baselines where possible.

### 8.5 Runtime calibration and scale selection

Measure these separately: primitive forward/backward by expanded size; candidate parameter fitting; legal proposal construction; local policy inference/update; native Stitch extraction by archive size; paired gate adaptation; final task adaptation; model loading/compilation; and prediction. Fit empirical cost surfaces rather than extrapolating from FLOPs alone.

A useful forecast for one world is

\[
T_w\approx T_{\rm startup}+\sum_t T_{\rm solve,t}
+\sum_c\bigl(T_{\rm propose,c}+T_{\rm fit,c}+2r_cT_{\rm gate\ adapt,c}\bigr)
+T_{\rm policy}+T_{\rm IO}.
\]

Count rejected candidates. As an illustration, a four-task paired panel with two-second adaptations costs about 16 seconds per candidate before overhead; 200 such candidates would consume about 53 minutes. This is why gate cost must be measured before building a complex library. It is not a forecast of the actual laptop workload.

Choose the primary `B_meta` and `b` using development learning curves: find a nontrivial regime where competent primitive/fixed-module baselines learn constituents, oracle macro access has measurable headroom, and the methods have not all saturated. Use the smallest scientifically informative regime, then include lower/higher budget sensitivity. Do not select only a narrow budget where SIML happens to win. Publish how the main budget was selected and show the development curves for competitors.

Choose maximum graph size from target representability; choose callable-library capacity from measured selection/storage curves; choose archive size from candidate-yield versus extraction-cost curves; choose independent worlds from Section 7.6. Starting values such as 16 library entries or 32-node scratch graphs are merely calibration points. If a required scale exceeds the local time/storage envelope, narrow the claim or report insufficient sensitivity. A longer justified laptop run is acceptable; unbounded search for a favorable result is not.

### 8.6 Timing protocol

Use monotonic wall-clock time for practical budgets. Before and after measured MPS regions, synchronize the device; asynchronous dispatch is not completed computation [R25]. Warm up each operation/backend and run a representative block until initialization and thermal transients are characterized. Separate cold-start compilation/import time from steady-state timings and include both in the appropriate end-to-end totals.

Run methods serially in randomized paired blocks on the same laptop, plugged in, with the same power settings and controlled CPU thread counts. Measure CPU/Rust/Julia concurrency to avoid oversubscription. Record thermal/power/memory-pressure indicators that are available without unsupported privileges. Do not discard slow runs merely because they hurt one method. Predefine environmental invalidation criteria, such as system sleep or unrelated heavy workload, and retain invalid records with reasons.

For inference, use identical inputs, batch sizes, warm-up, and model-loading boundaries. Repeat enough to make the timing interval narrower than the relevant observed difference, based on pilot variability. Repeated timer samples are nested measurement repeats. Reporting median latency alone must not hide tail regressions or one-time compilation costs.

### 8.7 Checkpoint, interruption, and recovery

At atomic task or gate boundaries, save graph/library hashes, parameters, optimizer state keyed by parameter identity, numerical priors, policies, normalization statistics, Python/NumPy/PyTorch RNG states, any supported MPS RNG state, donor reservoir, pending candidate/panel IDs, consumed data IDs, and monotonic expenditure counters. Store metadata as JSON and arrays/tensors in a non-executable format; for NumPy archives use `allow_pickle=False`. Do not load untrusted downloaded pickles.

Write to a temporary path, flush, atomically rename, then append a commit event. Keep the prior committed checkpoint until the new one passes hash/readback checks. Library updates and index updates are one transaction. A restart must not silently regenerate gate data, forget rejected trials, mutate an older version, or reset the compute ledger.

Deliberately interrupt and resume a CPU run. Require exact state/prediction agreement at the safe boundary and equivalent subsequent deterministic decisions. MPS may not provide identical trajectories; verify snapshot equality immediately, RNG restoration where supported, and resumed behavior against its measured repeat-run variability. Declare that limitation instead of promising bitwise determinism.

An OOM recovery first rolls back to a safe transaction and reduces only concurrency, keeping the algorithm unchanged. Record wasted work and recovery cost. Repeated OOM at the minimum concurrency is a resource failure; it is not permission to shrink a model only for one method during confirmation. OS interruptions, implementation crashes, algorithmic failures, and budget exhaustion receive different status codes.

## 9. Attribution, robustness, and failure analysis

### 9.1 Required causal controls

| Intervention | What stays matched | What it distinguishes | Interpretation |
|---|---|---|---|
| No consolidation | Solver, numerical learning, storage allowance, training budget | Dynamic search versus persistent vocabulary | A gain that survives unchanged is not attributable to consolidation. |
| No library calls during fresh adaptation | Same learned snapshot and sunk acquisition cost; expanded primitive search still permitted | Future access to learned abstractions | Loss of transfer supports usefulness of callable structures, not yet superiority of the local proposer. |
| Exact inlining at execution only | Same fitted graph, numerical values, aliases, operation order | Runtime correctness and naming effects | Predictions and input derivatives should match; parameter derivatives match under corresponding differentiable test clones. This is a null control, not a test that reuse is unimportant. |
| Whole-solution replay / cached warm starts | Equal archive bytes and preparation budget | Reuse of subcomputations versus cached solved tasks | If replay explains all gains, narrow the claim. No memoization of test answers is allowed. |
| Topology versus parameter reuse | Same bodies; compare frozen snapshots, fresh initializations, and task-owned clones | Learned computation versus numerical warm starts | Weight randomization alone can harm any model and is not sufficient causal evidence. |
| Random/frequency/MDL/transfer acceptance | Candidate pool, fit allowance, library envelope | Quality of consolidation criterion | Gate contribution is separate from proposal contribution. |
| Local versus Stitch versus global proposals with common gate | Donor data, executor, gate, budget | Value of the local structural mechanism | A Stitch win invalidates superiority of locality on that setting, not library learning generally. |
| Learned versus frozen/random local scores | Same legal cuts, feature extraction, cost envelope | Learning versus a hand-coded search bias | Candidate ranking must improve held-out outcomes, not just its own training labels. |
| No-gradient local features | Policy size, activity statistics, proposal pool | Reliance on task-wide credit entering through gradients | A collapse rules out strong claims of activity-only local organization. |
| Matched capacity / larger dense model | Actual active/stored parameters and compute frontiers | Extra capacity versus invention | A size-matched or cheaper dense match weakens the architecture case. |
| Flat versus hierarchical library | Comparable storage and expanded compute; matched lower-generation entries | Higher-order discoveries | Removing generation-two-and-higher calls must affect later learning to support cumulative benefit. |
| History intervention | Earlier training tasks and total cost; disable early macro calls while learning later circuits | Whether old discoveries facilitate new discoveries | A diagram with nested references alone does not establish H4. |

The history intervention is conducted from an earlier saved training checkpoint and reruns subsequent learning. Merely expanding a finished generation-two body still leaves the same function available and cannot test whether earlier knowledge helped discover it. Charge the extra experimental runs; do not conflate their research cost with a single deployed method's cost.

Measure both **per-entry** and **group** interventions. Redundant entries can compensate for a single removal; no individual effect does not prove the whole library is irrelevant. Conversely, removing every useful parameter at once exaggerates the special contribution of structural invention. Report interactions rather than assuming additive utility.

### 9.2 Retention versus immutable storage

Re-executing an old frozen graph with its exact dependency versions should preserve its outputs; this is a software integrity property. It is not, by itself, evidence of superior continual learning. Separately measure reacquisition of old task families from support without revealing their identities, and performance of the current learner after later tasks. Report the error increase relative to prior checkpoints, replay/storage cost, and the possibility that a larger vocabulary makes retrieval harder.

Do not compare SIML with stored old task graphs against a baseline forbidden to retain equivalent information. If task IDs are supplied for a separate routing experiment, supply them to every method and label the assumption.

### 9.3 Adversarial cases and numerical robustness

Test near-constant and saturated units; opposing gradients from different examples; duplicated algebraic functions with different syntax; permutations of ports; irrelevant high-frequency subgraphs; identical local summaries with different distant contexts; library entries useful only at deeper budgets; rare but important functions; shared-parameter aliasing; dependency retirement; and a circuit useful only on the donor input distribution.

Evaluate noise levels and shifted input domains separately from the clean primary result. Recompute target bounds or retain an unbounded secondary metric when a shift leaves `[-1,1]`; do not blindly apply the clean bound to a different problem. Public diagnostic fingerprint inputs cannot become the only equality test, because distinct functions can agree on finitely many probes. No approximate functional deduplication is allowed to silently replace exact semantics.

Use task-order reversals, fresh world dictionaries, support-size/time sweeps, different candidate/library caps, and a competent larger baseline. Generic compilation must benefit all applicable methods. Inspect worst-case task strata as well as means; a mean gain can conceal failures on deep or no-reuse tasks.

### 9.4 Failure taxonomy

| Status | Evidence | Required next action |
|---|---|---|
| `implementation_invalid` | Wrong gradients, broken baseline adapter, leaked labels, inconsistent versions, invalid timer | Repair and rerun from a valid checkpoint/data freeze; do not interpret scientific scores. |
| `proxy_insensitive` | All methods saturate; all fail constituents; oracle useful library cannot help; deep targets numerically constant | Redesign/calibrate the workload on development data, retain the failed diagnostic, and use fresh confirmation. |
| `resource_limited` | Faithful configuration cannot fit or cannot obtain meaningful precision locally | Use semantic-preserving concurrency changes; otherwise narrow scope and report the untested scale. |
| `evidence_against_locality` | Correct/sensitive comparison favors uniform/global/Stitch proposals | Reject or narrow H1/H3 for that regime; a global-library result is not a local-invention result. |
| `mechanism_without_advantage` | Learned reusable circuits causally help, but simpler methods match at lower cost | Report a replication/negative competitiveness result; adopt the simpler method where practical. |
| `inconclusive` | Intervals cover meaningful benefit and harm, or results vary strongly across worlds | Report uncertainty; justify any new independent batch by expected information, not desire for significance. |
| `supported_local_effect` | Valid paired benefit survives mandatory controls and cost accounting | Proceed to independent generator and hierarchy tests; keep the claim at the demonstrated abstraction level. |

A scale-dependent mechanism is not rejected merely because an unexpressive toy fails. Escalation nevertheless needs evidence: constituent learning, oracle headroom, growing candidate utility with depth/reuse, a feasible resource forecast, and a clear next discriminating test. “It might work at scale” is not an unlimited exemption from negative evidence.

## 10. Risk-ordered work packages and decision gates

All `python -m siml ...` commands below are **proposed project commands that the coding agent must implement**. They are not installed tools and were not executed for this plan. Upstream/native commands are identified separately. Every package writes a decision record with `continue`, `revise`, `blocked`, `stop_claim`, or `inconclusive`, and the supporting artifact hashes.

### WP00 — Establish provenance, permissions, and local capacity

**Objective/risk:** determine whether the actual laptop and required baseline dependencies can support a faithful experiment. **Dependencies/inputs:** this plan, target laptop, public upstream assets; no model training required.

**Actions/files:** implement `siml doctor`; collect Section 8 system information into `artifacts/hardware.json`; create isolated environments; install a compatible native PyTorch; resolve and pin Stitch and optional PySR; record sources, commits, hashes, licenses, and exact commands in `provenance/assets.jsonl` and dependency locks. Verify Python architecture. Run small forward/backward and native Stitch example smoke tests. Record the AbstractBeam native configuration as a broader-reference resource constraint rather than attempting its full training setup. Block FePySR copying/execution until its use conditions are resolved.

**Tests/outputs/resources:** `tests/test_environment.py`; capability matrix by operation/device/dtype; startup seconds, baseline RSS bytes, free disk bytes, and imported runtime memory. Measure actual install/cache footprint. No large pretrained assets are needed.

**Gate:** proceed with CPU even if MPS is unavailable, provided all core semantics and the native compression baseline work. Repair architecture/wheel mismatches before proceeding. If native Stitch cannot be built, document the failure and restrict comparison claims; a weak pure-Python substitute is not the mandatory baseline. Administrative OS upgrades or unrelated system changes require explicit authorization, not silent execution.

### WP01 — Establish the computational substrate and invariants

**Objective/risk:** test H0 and eliminate semantic errors before learning structure. **Dependencies:** WP00. **Inputs:** primitive specification and explicit tiny test graphs.

**Actions/files:** implement `ir.py`, `types.py`, `ops.py`, `execute.py`, `library.py`, and `costs.py`; construct the CPU reference and packed executor; add copy-on-write specialization, content hashes, exact expansion, and dependency-safe retirement. Measure numerical behavior versus expanded depth. Add graphs for `x0*x1`, bounded shifted tanh, nested gates, repeated inputs, and repeated macro calls with intentional shared/frozen parameter identities.

**Tests/outputs/resources:** `tests/test_types.py`, `test_gradients.py`, `test_macro_equivalence.py`, `test_versions.py`; output max forward/gradient deviations, finite-difference tolerances, invalid-graph reasons, and microbenchmarks in seconds/bytes. Run a size sweep whose maximum covers the generator's declared target bounds. Use CPU float64 checks and CPU/MPS float32 execution parity where available.

**Gate:** zero type/version/alias violations; numerical differences explained by the specified tolerance; known-graph fitting succeeds at the required depths. If conditioning fails, revise the primitive/depth regime on development data. Do not proceed to invention and blame its policy for a broken differentiable substrate.

### WP02 — Build datasets, evaluator, and a sensitive measurement harness

**Objective/risk:** eliminate leakage, trivial tasks, and uninformative proxies. **Dependencies:** WP01. **Inputs:** G1/G2 definitions and public split rules.

**Actions/files:** implement `data/g1.py`, `data/g2.py`, `data/manifests.py`, `eval/firewall.py`, and `eval/metrics.py`; produce exploratory worlds only; test exact/approximate overlap diagnostics; keep teacher definitions out of `TaskSupport`; validate that queries are inaccessible to adaptation. Implement AULC and stratified aggregation. Build identical-method, degraded-prediction, and known-useful-library diagnostic cases.

**Tests/outputs/resources:** `test_splits.py`, `test_no_query_access.py`, `test_metric_nulls.py`; data reports with task counts, motif/context coverage, exclusions, variance/dependence, and query Monte Carlo error. Estimate generation cost and disk bytes/task. Probe increasing depth and reuse frequency before picking final scale.

**Gate:** the evaluator detects a known improvement, produces no systematic improvement for identical methods, and has a regime with constituent learnability plus oracle-library headroom. All-fail/all-perfect regimes are `proxy_insensitive`; revise them rather than report a substantive negative result. Protected worlds remain unopened/uncreated.

### WP03 — Implement and validate strong local baselines

**Objective/risk:** avoid engineering a win against weak competitors. **Dependencies:** WP01–WP02. **Inputs:** exploratory worlds, upstream sources, baseline specification.

**Actions/files:** implement `search/solver.py`, `methods/primitive.py`, `methods/dense.py`, `methods/modular.py`, `methods/stitch.py`, `methods/growth.py`, `methods/sparse.py`, `methods/constructive.py`, and `methods/pysr_adapter.py` as applicable. Implement both Stitch encodings and reconstruction tests. Share execution/cost accounting. Establish fixed-module meta-training rather than a random-module baseline. Record native/ported distinctions in `provenance/baseline_adaptations.md`.

**Tests/outputs/resources:** reproduce an upstream native Stitch example; reconstruct every returned test abstraction exactly; verify fixed-module learning on a known-module task family; verify growth/freeze/nonzero-budget mechanics; run PySR on a simple polynomial in its native environment. Produce support-only learning curves, losses versus time, parameter counts, and original implementation/configuration references. Time one complete world before launching broad tuning.

**Gate:** relevant baselines pass correctness and achieve nontrivial performance at an informative scale. A broken or obviously undertrained baseline blocks competitiveness claims. If a closest broader system cannot run locally, retain it as an explicit reference limitation; do not claim to outperform its published result from incomparable numbers.

### WP04 — Test whether local proposal information is useful

**Objective/risk:** test H1/H3 cheaply before an end-to-end developmental loop. **Dependencies:** WP03. **Inputs:** common learner-produced donor archive; fresh training gate panels; uniform candidate generator.

**Actions/files:** implement `structure/local_features.py`, `proposals.py`, `value_model.py`, `training_pairs.py`, and `experiments/common_donor.py`. Generate candidate labels with paired unchanged/edited fitting and paired transfer panels. Divide donor contexts into policy-fit and policy-evaluation groups. Compare learned local, uniform, frequency, global, and native Stitch proposals under the same gate; log all proposal and gate seconds. Include unsuccessful donor attempts and uniformly evaluated candidates.

**Tests/outputs/resources:** feature-access audit; labels timestamped after features; candidate coverage by body size/arity; ranking calibration; top-selected transfer utility and library AULC versus total discovery seconds. Estimate label acquisition expense before increasing candidate counts.

**Gate:** proceed to full SIML when local ranking has credible predictive utility or a clearly diagnosed, cheaply testable limitation remains. If uniform/Stitch dominates with adequate coverage and uncertainty, stop the local-superiority claim or choose the simpler method. If labels are too noisy, calibrate panel/repeat counts; do not keep only successful candidates to manufacture learnability.

### WP05 — Implement the minimum complete developmental loop

**Objective/risk:** test whether the offline signal survives autonomous donor acquisition and consolidation. **Dependencies:** WP04 plus all required runtime invariants. **Inputs:** frozen exploratory configuration and training-only data access.

**Actions/files:** implement `methods/siml.py`, `structure/gate.py`, `training/meta_loop.py`, `training/checkpoint.py`, and `experiments/developmental.py`. Execute the loop in Section 5.7 with empty learned libraries, conditional numerical fitting, sequential gates, exact snapshots, and generation tracking. Count each donor, candidate rejection, fitting step, gate task, and library call. Make snapshots read-only during evaluation. Implement no-consolidation/no-call controls alongside the full method.

**Tests/outputs/resources:** complete event trace from primitives to proposed/accepted circuits; round-trip saved predictions; dependency graphs; candidate utility conditional on library hash; kill/resume tests; gate-label isolation; total wall time and memory model residuals. Store only detached statistics unless a diagnostic trace is explicitly retained and charged.

**Gate:** a complete correct run is necessary, not sufficient. If no candidates are accepted, distinguish lack of useful structures, a poor proposer, and an overexpensive/noisy gate using the common-donor diagnostics. A visually attractive accepted circuit is not an entry ticket to costly confirmation.

### WP06 — Calibrate resources, sensitivity, and fair tuning

**Objective/risk:** choose an informative and feasible test, and assess baseline tuning sensitivity. **Dependencies:** WP03–WP05. **Inputs:** separate pilot worlds, timing/memory measurements, tuning spaces.

**Actions/files:** implement `experiments/calibrate.py`, `experiments/tune.py`, and `analysis/design.py`. Fit resource models from representative runs; compare CPU versus MPS end-to-end; tune every method with recorded comparable research expenditure; measure effect variance by world, seed, task, and query. Check operator/graph/library cap boundaries. Determine target representability before setting caps. Select `B_meta`, `b`, stratum weights, precision target, and comparator set from development criteria only.

**Tests/outputs/resources:** `artifacts/calibration.json`, `artifacts/tuning_trials.jsonl`, a full-run forecast, and `protocol/design_rationale.md`. Forecast total time as measured per-world/per-method cost times planned counts plus uncertainty, not a claimed M3 throughput from another machine. Include expensive gates and baseline compilation.

**Gate:** proceed only with a credible precision/cost plan and informative workload. Expand inadequate baseline tuning before freezing. If the attainable local design is underpowered, narrow the question or explicitly plan an inconclusive interval estimate. Do not choose a dramatic but unstable effect as the power target.

### WP07 — Freeze the confirmatory protocol

**Objective/risk:** prevent adaptive selection and undisclosed methodological drift. **Dependencies:** WP06. **Inputs:** pinned code, assets, resolved configurations, analysis tests, and hardware manifest.

**Actions/files:** implement `siml freeze`; write `protocol/confirmatory.yaml`, seed-generation commitments, baseline configurations, all metrics/strata/failure rules, fixed sample size, paired run-order plan, and analysis code hash. Run a validator that rejects unresolved numerical settings, missing version pins, unavailable required baselines, test-data access paths, and insufficient representational caps. Record preplanned ablations and which secondary claims require multiplicity handling.

**Tests/outputs/resources:** a signed or hashed immutable protocol bundle and `FREEZE.json`; dry-run analysis on synthetic null/effect data; no protected model evaluation. Work is metadata-sized plus an inexpensive harness run.

**Gate:** every required field resolves. Amendments affecting the estimand, baseline fairness, population, or interpretation must be recorded before new protected data. Routine formatting, log rotation, or nonsemantic performance engineering can proceed with regression tests and recorded revisions.

### WP08 — Run the protected paired comparison

**Objective/risk:** obtain the primary algorithm-level evidence. **Dependencies:** WP07. **Inputs:** protected worlds, frozen protocol, pinned environment.

**Actions/files:** implement `siml run --protocol protocol/confirmatory.yaml`; execute randomized method blocks, within-world training, frozen-snapshot adaptation, and evaluator-only scoring. Save predictions before score aggregation. Monitor environment/resource flags; resume only from validated transactions. Do not expose interim comparison dashboards to tuning logic. The primary script must stop at the fixed planned sample count, not when a p-value crosses a threshold.

**Tests/outputs/resources:** raw prediction files, task/world metrics, complete cost and failure records, environment history, final simultaneous intervals, and run-completeness audit. Requirements are the measured WP06 forecast; run worlds sequentially when memory requires it.

**Gate:** classify positive, negative, mixed, or inconclusive results using Section 7.6 and guardrails. A primary gain with a material failure/latency regression is a tradeoff. Any implementation/data-validity failure invalidates affected results and requires a documented repair and fresh appropriate confirmation, not selective deletion.

### WP09 — Attribute the result and test cumulative learning

**Objective/risk:** rule out alternative explanations and test H4. **Dependencies:** correct WP05 checkpoints and the preplanned attribution protocol; some cheap controls run before WP08. **Inputs:** paired snapshots, stage checkpoints, fixed support/query panels.

**Actions/files:** implement `experiments/ablations.py` and `experiments/history_interventions.py`; execute Section 9 controls, including same-gate local/Stitch comparisons, no-call fresh adaptation, replay/capacity controls, and rerunning later learning with earlier macro access disabled. Compare stage-one, stage-two, and final library transfer at equivalent resource accounting. Report generation-two marginal benefit and the effect of historical access separately.

**Tests/outputs/resources:** intervention manifests with unchanged/changed fields; exact-inlining null test; per-entry/group effects; interaction estimates; call and expanded-compute counts. Budget is measured from the relevant subset of full-world costs. Selection of entries for intervention uses a predeclared rule, not whichever removal makes the largest final effect.

**Gate:** claim cumulative abstraction only when earlier discoveries causally facilitate later useful discoveries and additional higher-generation calls improve protected learning. Otherwise report flat transfer, numerical warm starts, or a library-use effect as appropriate. Unplanned post-result ablations are exploratory and require independent confirmation for new claims.

### WP10 — Test scope beyond the native generator

**Objective/risk:** evaluate H5 and prevent a grammar-specific result from becoming a general claim. **Dependencies:** WP06-calibrated G2 and an interpretable core result, or a targeted boundary question. **Inputs:** independent G2 worlds; optionally G3 and cleared optional assets.

**Actions/files:** implement `experiments/cross_generator.py`; distinguish within-G2 learning from strict G1-library transfer. Apply the same resource/protocol rules and fixed neural competitors. For G3 implement the exact fields in Section 6.4, cubic-feature ridge, and PySR; include FePySR when use permissions and local reproduction permit it. Register any newly chosen workload before scoring it.

**Tests/outputs/resources:** independent-generator learning curves, negative-transfer cases, capacity/approximation diagnostics, and scope-specific intervals. Calibrate one G2/G3 world before estimating total cost. No external data download is a prerequisite for G1/G2 or the specified G3.

**Gate:** extend scope only where evidence supports it. G1-only success is a synthetic compositional result. Failure on G2/G3 may reveal vocabulary mismatch, absent reuse, or no practical advantage; report which diagnosis the controls support. Do not swap in a friendlier application and call it independent confirmation.

### WP11 — Reproduce, audit, and publish the decision record

**Objective/risk:** make the result usable without chat memory or selective reporting. **Dependencies:** all executed packages; may run partially after any stop. **Inputs:** full artifact tree including unfavorable and invalid runs.

**Actions/files:** implement `siml report`; generate `reports/final.md`, machine-readable results, effect/cost plots, limitations, baseline reproduction status, and a claim-to-evidence table. Rerun a declared independent world from a clean environment or clean process using saved assets/locks, without tuning on its result. Run the scientific/prior-art, experimental, and implementation audits in Section 13. Keep progress and decisions in append-only artifacts.

**Tests/outputs/resources:** reproduce metric calculations from raw predictions; validate all links/hashes; verify exact seed/data/config coverage; rerun a saved model's predictions; retain failed trials. Clean-environment reproduction cost is forecast from WP06, not hidden.

**Gate:** final outcome is one of: supported narrow contribution; component benefit with unfavorable tradeoff; replication/engineering result; evidence against the selected mechanism; or inconclusive. State what was not run and why. No unexecuted baseline or optional experiment may appear as completed.

## 11. Repository, interfaces, configuration, and provenance

### 11.1 Minimal repository layout

```text
siml-poc/
  README.md
  research_poc_plan.md
  pyproject.toml
  requirements.lock
  provenance/
    assets.jsonl
    baseline_adaptations.md
    source_notes.md
  protocol/
    exploratory.yaml
    pilot.yaml
    confirmatory.yaml
    design_rationale.md
    amendments.jsonl
    FREEZE.json
  src/siml/
    __main__.py
    ir.py  types.py  ops.py  execute.py  library.py  costs.py
    data/              # generators, public support objects, manifests
    search/            # generic legal edits and common solver
    structure/         # local features, proposers, value learning, gate
    methods/           # SIML and baseline adapters
    training/          # meta loop, budgets, checkpoint transactions
    eval/              # evaluator-only scoring and access firewall
    experiments/       # pilot, common-donor, confirmation, interventions
    analysis/          # AULC, paired inference, plots, report
  tests/
  data/public/         # allowed training streams and support sets
  data/evaluator/      # teacher definitions and protected query labels
  artifacts/
    hardware.json
    calibration.json
    runs/<run_id>/
      config.resolved.json
      environment.json
      events.jsonl
      costs.jsonl
      predictions/
      checkpoints/
      library/
      status.json
  decisions/
    progress.jsonl
    decisions.jsonl
  reports/
```

Use ordinary package entry points and type-checked dataclasses/protocols. The repository is proposed; it has not been implemented for this handoff. Avoid unnecessary distributed orchestration or a large experiment platform.

### 11.2 Shared method interface

```python
from dataclasses import dataclass
from typing import Protocol
import numpy as np

@dataclass(frozen=True)
class TaskSupport:
    opaque_id: str
    x: np.ndarray      # [k, d], finite float32; no hidden teacher fields
    y: np.ndarray      # [k, 1], finite float32
    input_types: tuple[str, ...]
    output_type: str

class Predictor(Protocol):
    def predict(self, x: np.ndarray) -> np.ndarray: ...
    def manifest(self) -> dict: ...

class Method(Protocol):
    def meta_train(self, training_access, budget, logger) -> object: ...
    def snapshot(self) -> object: ...
    def adapt(self, snapshot, support: TaskSupport,
              budget, rng, logger) -> Predictor: ...
```

`training_access` supplies only the permitted within-world training pool and tracks every label. `budget` exposes monotonic time/cost counters and an explicit `can_start(estimated_atomic_cost)` check; actual expense, overshoot, and best incumbent are always logged. `Predictor.predict` receives only inputs, not query labels. The evaluator checks shape, bounds, finiteness, and budget status after prediction. Baseline adapters cannot read hidden files or bypass the cost ledger.

Snapshot objects must not expose mutable shared tensors to adaptation. `meta_train` and `adapt` return status/manifest information identifying valid completion, timeout, OOM, invalid numerical output, and interrupted execution. All methods use the same task IDs and evaluator code, but RNG namespaces remain distinct and reproducible.

### 11.3 Configuration convention

Use one resolved configuration per run. A pilot may contain values awaiting calibration; a confirmatory configuration may not. The exact fields include:

```yaml
schema_version: 1
phase: pilot
method: siml_local_transfer
data:
  generator: g1
  input_dim: 6
  primitive_version: bounded_scalar_v1
  support_sizes: [4, 8, 16, 32, 64]
  query_count: 1024
structure:
  graph_kind: tree
  pilot_population: 16
  pilot_max_expanded_ops: 32
  pilot_max_candidate_expanded_ops: 8
  max_signal_ports: 3
  pilot_library_entries: 16
  policy_hidden: [32, 32]
  random_proposal_probability: 0.2
numerics:
  dtype: float32
  reference_dtype: float64
  backend: cpu
  fast_math: false
validation:
  pilot_gate_tasks: 4
  gate_support_count: 16
  gate_query_count: 128
  final_queries_visible_to_method: false
```

This is **not** a complete confirmatory file: measured budgets, final representability caps, method-specific tuned settings, counts of worlds/tasks/seeds, frozen source revisions, device settings, and analysis parameters are filled by WP06–WP07. The freeze validator must reject use of a `pilot_*` field as an unexplained final value. A change to an operation definition increments `primitive_version` and invalidates cached graph hashes.

### 11.4 Experiment-record schema

Every run manifest includes `run_id`, parent/checkpoint ID, UTC timestamp, phase, world seed, method, learner seed, code commit, dirty-tree patch hash if any, dependency-lock hash, data-manifest hash, config hash, hardware/backend, allowed budgets, and terminal status.

Every event record includes `event_id`, prior event hash, event kind, task/candidate/library version IDs, support/gate partition IDs, pre-outcome feature hash, result, elapsed seconds, cumulative charged seconds, expanded primitive evaluations, active/stored parameters, serialized bytes, memory observations, RNG stream identifier, and exception/rejection reason. Fields not applicable are explicit nulls, not invented values.

Every prediction record includes `world`, `task`, `method`, `seed`, `k`, query-input hash, prediction array hash, adapted-model hash, fit seconds, inference seconds, budget overshoot, and validity. Query labels remain evaluator-owned. Raw predictions and scores, not only aggregate plots, are required for reproduction.

Record both logical work and wall time. A macro with ten expanded operations is not one primitive evaluation. Store node execution counts separately from approximate FLOP estimates; the latter require an explicit convention and cannot replace measured latency.

### 11.5 Resumability, authority, and amendments

The coding agent may choose routine implementation details, file formatting, safe batching, test organization, and non-destructive logging changes. It must record a methodological amendment for changes affecting primitive semantics, hypothesis, evaluation population, final metrics/weights, primary comparator, information access, precision, resource matching, or interpretation. No deletion of unfavorable results or replacement of failed seeds is permitted.

Append progress and decisions after every work package and interruption. A resume command reads artifacts rather than relying on this conversation. Report schema migrations, preserve previous manifests, and validate old snapshot compatibility before rewriting stored state.

For an environmental failure, retain the failed record and rerun under the predeclared rule. For an algorithmic failure at a valid budget, retain its primary failure score. For a detected implementation defect, mark all affected runs invalid with a machine-readable reason and perform a documented repair; do not quietly retain only unaffected favorable results.

## 12. Outcome interpretation and evidence-triggered pivots

| Outcome | Supported conclusion | Next justified action |
|---|---|---|
| Exact runtime works, but learned proposals do not predict utility | The computational substrate is viable; the tested local information/update rule is insufficient. | Compare added local information versus the global control; stop calling frequency-based extraction learned invention. |
| Libraries help, but native Stitch plus the common gate matches or wins | Reusable abstraction helps this workload; the local mechanism is not justified. | Adopt the simpler/faster library learner or investigate a specifically measured local-scaling advantage with fresh tests. |
| Local proposals save discovery time but final error is unchanged | An efficiency result, conditional on equal quality and full cost accounting. | Quantify break-even and robustness; do not claim better representational capacity. |
| Better error but substantially higher preparation/inference cost | A quality–cost tradeoff, not general superiority. | Show the Pareto frontier and application horizon; stop if no plausible beneficiary values it. |
| Flat reusable functions help, but hierarchy interventions do not | Transferable abstraction without demonstrated cumulative higher-order learning. | Examine task reuse depth/coverage and credit assignment; do not use nested names as evidence. |
| G1 succeeds and G2 fails from vocabulary mismatch | A native-language result with a bounded scope. | Test a justified richer primitive family or vector interface as a new experiment, keeping the original negative result. |
| G1/G2 succeed, G3 loses to cubic regression/PySR | A mechanism finding without evidence of scientific-workload advantage. | Do not market the system as a better symbolic-regression tool. |
| No-reuse worlds incur large overhead | Selective consolidation/maintenance is inadequate. | Add a training-only no-change/retirement policy only after measuring where the overhead originates. |
| Confidence intervals remain wide | Inconclusive evidence, not equality or success. | Publish uncertainty or plan a fresh information-justified batch; no significance-driven continuation. |
| Strong matched effects recur on independent worlds/generators with H4 support | A credible narrow architectural/learning contribution. | Expand vector/DAG/stateful scope incrementally, with new priors, baselines, and resource calibration. |

The most ambitious locally testable development is not just two generations of names. It is a causal chain: an early learned computation changes later search, enables a new useful computation, and that new computation improves further protected learning at an acceptable total cost. Even this remains a finite-distribution result. Larger-scale possibilities deserve investigation when their prerequisites appear in local evidence, not because the label “self-inventing” implies them.

## 13. Audit record, residual risks, and limitations

These are three distinct author-conducted challenge passes on the research and proposed protocol. They are not independent peer review, a completed experimental audit, or evidence that the unbuilt system works. The revisions below are incorporated into the preceding specification.

### 13.1 Scientific and prior-art audit

| Defect or credible objection | Revision made and affected sections | Residual uncertainty / required evidence |
|---|---|---|
| “Inventing reusable primitives” duplicates established library learning and automatically defined functions. | Replaced the broad novelty claim with H1/H3: the specific information-restricted proposal mechanism and its measured quality/cost contribution (Sections 1–3). Added historical and neural/program-library comparisons. | A literature search cannot establish absence of equivalent work. If a matching mechanism is found, label the project a replication or differentiated empirical evaluation. |
| Cost-aware consolidation could be relabeled existing utility-based speed-up learning. | Added COMPOSER, AMaze, and the September 2026 bounded-synthesis preprint; made `stitch_transfer` with the identical gate a primary comparator (Sections 3, 7). | Whether the local policy improves on a good centralized proposal pool is completely unmeasured. |
| A symbolic expression learner could be presented as a validated neural self-organizing layer. | Declared the hybrid numerical/discrete architecture, scalar/tree restriction, limited CPU runtime, and excluded autonomous recurrence/economies (Sections 2, 5). | The PoC does not validate arbitrary neuron populations, recurrent computation, or a general-purpose drop-in neural layer. |
| Backpropagated gradients are not strictly local information. | Specified gradient-informed local observability and a no-gradient ablation; prohibited claims of fully decentralized credit assignment (Sections 2, 5, 9). | Task-wide information may be doing most of the work. A larger local neighborhood could erase the supposed locality advantage. |
| Recent numerical-feature learning could be an overlooked close alternative. | Added FePySR, its noncommercial permission gate, native PySR, and the LLM-based LaSR reference; separated per-dataset feature invention from cumulative transfer (Sections 3, 7, 14). | FePySR was not reproduced, and some recent sources are preprints. Numerical superiority cannot be claimed over unrun relevant competitors. |
| A small failing example might not express the proposed hierarchy. | Made graph/candidate caps follow public target-size bounds and oracle sensitivity, not convenient pilot constants (Sections 5, 7–8). | Local experiments cannot settle arbitrarily larger-scale effects; escalation needs observed prerequisites rather than an unfalsifiable scaling promise. |

A targeted patent-landscape check also found historical automatically defined-program mechanisms, including US6532453B1 [R27]. This reinforces the need to distinguish implementation and empirical contributions from broad invention language. It is not a claim-construction analysis, legal patentability assessment, or freedom-to-operate opinion; no present legal status is asserted.

### 13.2 Experimental-validity audit

| Alternative explanation of a positive result | Revision made | Acceptance check / remaining risk |
|---|---|---|
| SIML gets extra examples through validation panels. | Gate examples count as meta-training; all methods receive the same phase-specific allowed pool and may use it for their own learning (Sections 5, 7). | Audit label provenance and temporal access. A gate must never consume protected examples or future-phase labels. |
| Numeric constants make native Stitch artificially unable to find repeated structure. | Required literal and topology-first encodings with exact occurrence reconstruction; selected encoding is development-tuned (Section 7.3). | Native compression/rewrite and derivative-equivalence tests must pass before comparison. The topology adapter is an explicit adaptation, not native numerical-module support. |
| Fixed neural modules are too small, random, or compositionally handicapped. | Added learned modules, fine-tuning, base primitives, adequate composition depth, and separate measured neural-resource accounting (Section 7.2). | A failed known-module positive control blocks a competitiveness claim. No tiny pilot width is a permanent baseline limit. |
| Removing a macro name changes nothing, so reuse appears irrelevant. | Distinguished execution inlining, removal of calls during fresh search, and interventions on the history of later library construction (Section 9). | Exact inlining is a null control. H4 needs a changed developmental trajectory and later protected-learning effect. |
| Stable old snapshots are mistaken for continual learning. | Separated version integrity from old-task reacquisition without privileged task identity; charged stored models and replay (Section 9.2). | Retention of every saved solution can be a storage tradeoff, not better general learning. |
| Hundreds of related tasks create spurious precision. | World-level paired inference, nested seeds/tasks, pilot variance calibration, simultaneous primary intervals, and fixed-sample confirmation (Section 7.6). | Too few independent worlds or excessive variance can make the feasible experiment inconclusive. |
| Repeated inspection, entry selection, or stronger late baseline tuning manufactures a win. | Protected freeze, independent confirmation after changed claims, all trials retained, and preregistered intervention selection (Sections 7, 10–11). | Human access cannot be cryptographically undone. Log any accidental exposure and replace affected protected cohorts. |
| Positive error change conceals longer search, expensive inference, or worse rare tasks. | Full-cost ledgers, error/time frontiers, tail/stratum guardrails, no-reuse worlds, and simple polynomial competitors on G3 (Sections 7–9). | A primary-quality win is only a resource-conditioned claim; important regressions remain tradeoffs. |

### 13.3 Implementation and feasibility audit

| Execution defect or adverse scenario | Revision / recovery path | Blocking gate |
|---|---|---|
| A source or package version is invented or incompatible with the actual Mac. | No fabricated pins; discover native arm64 packages, record resolved revisions, isolate baseline environments, and test the actual kernels (Section 8). | WP00 must produce a valid lock and smoke-test record. Web documentation is not an installed-kernel test. |
| MPS dispatch is slow or an operation/backward path is unsupported. | CPU float32 is the faithful core; MPS is optional and chosen by whole-loop measurements, with no silent fallback or reduced-precision substitution. | Parity and measured-loop tests precede backend selection. A missing acceleration path does not justify changing the scientific semantics. |
| Unified memory is much smaller than anticipated. | Discover physical/available memory, reserve headroom, measure autograd/archive overhead, fit candidates serially before shrinking scientific scope (Section 8.4). | OOM and swapping invalidate affected timings. A scope reduction after freeze requires an amendment and fresh confirmation. |
| A dense neural module is counted as one cheap primitive, or is excluded by a symbolic node cap. | Separated primitive-IR caps from actual neural time/memory/operation costs (Section 7.2). | Confirm comparator representability and cost conventions before WP07. |
| Frozen weights cannot have parameter gradients in an equivalence test. | Parameter-gradient checks use corresponding temporary differentiable clones; actual immutable library parameters remain frozen. | WP01 and the Stitch adapter test must distinguish input derivatives, cloned parameter derivatives, and training permissions. |
| A checkpoint resumes with a stale acceptance panel or inconsistent optimizer/library state. | Atomic content-addressed snapshots, RNG/counter/event transactions, obsolete-panel invalidation, and forced kill/resume tests (Sections 5, 8, 11). | CPU semantic resumption tests pass before long runs. MPS bitwise identity is not promised. |
| Optional assets have uncertain or unsuitable licenses. | Core data are independently generated; ODEBench is blocked without a verified license; FePySR use is conditional (Sections 6–8). | Do not download-and-assume permission. An unavailable asset narrows the comparison rather than silently creating a weak substitute. |
| The strongest published implementation is too large or cannot be faithfully installed locally. | Run the strongest credible small native component/adaptation, label it precisely, and retain the unrun full method as a scope limit (Section 7.1). | No claim of a faithful full-method victory without a faithful run. Failure of a custom port is not negative evidence about the original method. |
| Synthetic results do not transfer to an independent generator or field-regression workload. | G2 separates new-generator learning from strict library transfer; G3 retains ridge/PySR controls (Sections 6, 10). | Report a narrower mechanism result. No empirical result in this project alone supports real-world data, deployment, or general-intelligence claims. |

The final cross-check revisited these changes against the shared method interface, stage ordering, budget ledger, and freeze requirements. The plan retains unavoidable empirical questions—learnability, task reuse frequency, candidate utility, numerical conditioning, statistical precision, and actual Mac throughput—as measured gates. It does not conceal them behind implementation instructions or assert that a plausible future scale will rescue every negative result.

## 14. References, inspected evidence, and search coverage

### 14.1 Reading this evidence ledger

Reference identifiers in the text resolve to the entries below. URLs and DOIs are stable source pointers, not dependencies on this conversation. “Inspected” means the stated paper sections, abstract, documentation, or repository material was actually read during this research; it does **not** mean the associated code was run. Dates are publication/preprint dates where established, otherwise the access date. Online-first and future issue dates are distinguished. Repository revisions must still be resolved and pinned in WP00.

### R01 — DreamCoder

Kevin Ellis and collaborators. **DreamCoder: Bootstrapping Inductive Program Synthesis with Wake-Sleep Library Learning.** PLDI, 2021. [DOI: 10.1145/3453483.3454080](https://doi.org/10.1145/3453483.3454080). [Author-hosted paper](https://www.neurosymbolic.org/papers/EllisWNSMHCST21.pdf); [implementation](https://github.com/ellisk42/ec).

**Inspected:** paper text, selected PDF pages, repository README. **Informs:** library induction, learned search, recursive abstraction, strongest conceptual overlap. Demonstrations concern specified synthesis domains; they do not establish the value of SIML's local neural interface. No reproduction or code checkout was completed.

### R02 — Stitch: top-down library learning

Matthew Bowers and collaborators. **Top-Down Synthesis for Library Learning.** POPL, 2023. [DOI: 10.1145/3571234](https://doi.org/10.1145/3571234). [Full text](https://arxiv.org/html/2211.16605v2); [author-hosted PDF](https://mlb2251.github.io/stitch.pdf).

**Inspected:** algorithm/evaluation text and PDF pages. **Informs:** efficient centralized abstraction extraction and compression-based objective. Reported extraction improvements are not whole-system SIML comparisons. Expression/DAG and numerical-parameter differences motivate an explicit, tested adapter rather than presumed equivalence.

### R03 — Stitch implementation and binding documentation

Stitch authors/maintainers. [Native repository](https://github.com/mlb2251/stitch); [binding documentation](https://stitch-bindings.readthedocs.io/en/stable/); [tutorial](https://stitch-bindings.readthedocs.io/en/stable/intro/tutorial.html); [installation](https://stitch-bindings.readthedocs.io/en/stable/intro/install.html). Accessed 14 October 2026.

**Inspected:** README, MIT license indication, documented examples/API, selected native exports. **Informs:** executable compression/rewrite baseline and release-build path. Documentation labeling a binding version is not a verified latest release or installed binary. Native compilation and exact revision pinning remain gates.

### R04 — HOUDINI

Lazar Valkov and collaborators. **HOUDINI: Lifelong Learning as Program Synthesis.** NeurIPS, 2018. [Full text](https://arxiv.org/html/1804.00218v2); [proceedings](https://papers.nips.cc/paper/8086-houdini-lifelong-learning-as-program-synthesis); [repository](https://github.com/trishullab/houdini).

**Inspected:** paper text and repository README/dependency context. **Informs:** typed composition of differentiable neural functions and lifelong reuse; a particularly close precedent. Original software has old dependencies. No modernized implementation is silently counted as a faithful reproduced baseline.

### R05 — Modular meta-learning

Ferran Alet, Tomás Lozano-Pérez, and Leslie P. Kaelbling. **Modular meta-learning.** CoRL, 2018. [Paper](https://proceedings.mlr.press/v87/alet18a/alet18a.pdf); [HTML](https://arxiv.org/html/1806.10166); [repository](https://github.com/FerranAlet/modular-metalearning); [license](https://raw.githubusercontent.com/FerranAlet/modular-metalearning/master/LICENSE).

**Inspected:** method text, repository, MIT license. **Informs:** BounceGrad-style alternating structure selection and numerical module learning. It supports a strong learned, compositional baseline—not randomly initialized frozen modules. The proposed scalar benchmark port is an adaptation requiring its own reproduction checks.

### R06 — Local module composition

Oleksiy Ostapenko, Pau Rodriguez, Massimo Caccia, and Laurent Charlin. **Continual Learning via Local Module Composition.** NeurIPS, 2021. [Full text](https://arxiv.org/html/2111.07736); [repository](https://github.com/oleksost/LMC).

**Inspected:** method text and implementation documentation. **Informs:** locally assessed module relevance, composition, and expansion without requiring task identity at inference. This substantially narrows claims about “local modular organization,” but is not an established demonstration of SIML's tiny-unit boundary invention.

### R07 — AbstractBeam

Janis Zenkner, Lukas Dierkes, Tobias Sesterhenn, and Christian Bartelt. **AbstractBeam: Enhancing Bottom-Up Program Synthesis using Library Learning.** 2024 preprint. [Full text](https://arxiv.org/html/2405.17514v2); [repository](https://github.com/jzenkner/AbstractBeam); [training configuration](https://raw.githubusercontent.com/jzenkner/AbstractBeam/master/crossbeam/experiment/deepcoder/configs/train/abstractbeam.py).

**Inspected:** paper, README/license indication, native configuration. **Informs:** neural bottom-up synthesis plus learned abstractions. The inspected configuration lists two GPU processes, 30 data-generation processes, and a ten-million-step maximum; this is not a measured runtime or proof that all configurations require those resources. Full native training was not reproduced.

### R08 — Historical utility problem / COMPOSER

Jonathan Gratch and Gerald DeJong. **COMPOSER: A Probabilistic Solution to the Utility Problem in Speed-Up Learning.** AAAI, 1992, pp. 235–240. [Paper](https://cdn.aaai.org/AAAI/1992/AAAI92-037.pdf).

**Inspected:** paper text and algorithm page visually. **Informs:** acquired knowledge can slow search; retention should consider utility and cost; interacting candidates make stale utility estimates problematic. Its formal assumptions and statistical procedures are not automatically valid for SIML's adaptive neural task stream.

### R09 — Abstraction selection under bounded search

Adam Cuculich. **On the Value of Abstractions: Abstraction Selection in Bounded Program Synthesis.** 20 September 2026, author-posted preprint. [Project/paper page](https://adamcuculich.com/research/on-the-value-of-abstractions/); [PDF](https://adamcuculich.com/research/on-the-value-of-abstractions/on-the-value-of-abstractions.pdf); DOI: **10.13140/RG.2.2.17726.24647**.

**Inspected:** paper page, full-text passages, PDF pages. **Informs:** direct comparison of compression and held-out search-cost selection from a common candidate pool, with budget/library effects. This is unusually close to the proposed gate; it is not peer-reviewed evidence that SIML's local mechanism succeeds. Code and data were not executed.

### R10 — AMaze

Zhentao Ye, Ruyi Ji, Yingfei Xiong, and Xin Zhang. **AMaze: Accelerating Syntax-Guided Program Synthesis by Optimizing Domain-Specific Languages.** POPL, 2026. [Paper](https://xinpl.github.io/papers/popl26b.pdf); DOI: **10.1145/3776679**.

**Inspected:** method/evaluation text and PDF pages. **Informs:** optimizing a synthesis language using learned costs and combinatorial optimization rather than rewarding syntax compression alone. Its solver setting and information access differ from a local differentiable layer. No direct laptop reproduction is claimed.

### R11 — Prospective compression

Leonardo Hernandez Cano and collaborators. **Prospective Compression in Human Abstraction Learning.** 11 May 2026 preprint, arXiv:2605.09985. [Full text](https://arxiv.org/html/2605.09985).

**Inspected:** full-text passages on the task/model and interpretation. **Informs:** future usefulness and nonstationary demands matter for abstraction selection. Evidence concerns human abstraction behavior and models in its specified environment; it is not a neural module-invention benchmark or independent support for SIML performance.

### R12 — E-Stitch

Kavi Gupta, Maddy Bowers, and Armando Solar-Lezama. **E-Stitch: Top-Down Library Learning for E-Graphs.** EGRAPHS workshop, 2026. [Official workshop abstract](https://pldi26.sigplan.org/details/egraphs-2026-papers/7/E-Stitch-Top-Down-Library-Learning-for-E-Graphs).

**Inspected:** official abstract only. **Informs:** ongoing work on equivalent-expression-aware library learning; syntax-level uniqueness is not functional novelty. Full paper/artifact and implementation conditions were not verified, so this is a relevant boundary/lead rather than a reproduced comparator.

### R13 — LILO

Gabriel Grand and collaborators. **LILO: Learning Interpretable Libraries by Compressing and Documenting Code.** ICLR, 2024; arXiv:2310.19791. [Full text](https://arxiv.org/html/2310.19791v2); [repository](https://github.com/gabegrand/lilo).

**Inspected:** paper text. **Informs:** library discovery can be combined with LLM-guided synthesis and documentation. Such external model machinery is outside this project's primary mechanism and local-resource scope. No comparison to its published scores or free use of its pretrained capabilities is proposed.

### R14 — Negative evidence about purported reuse

Ian Berlot-Attwell, Frank Rudzicz, and Xujie Si. **Library Learning Doesn't: The Curious Case of the Single-Use “Library”.** 26 October 2024 preprint, arXiv:2410.20274. [Full text](https://arxiv.org/html/2410.20274v1).

**Inspected:** methods, reuse analysis, and ablation discussion. **Informs:** some LLM-library gains can arise without meaningful reuse. The examined systems/tasks include LEGO-Prover/TroVE-style settings; this is not a general disproof of DreamCoder or all library learning. It motivates direct usage and causal-intervention tests rather than hierarchy visualizations.

### R15 — Automatically defined reusable programs

John R. Koza. **Genetic Programming II: Automatic Discovery of Reusable Programs.** MIT Press, 1994. [Author bibliography and description](https://www.genetic-programming.com/jkpubs94.html).

**Inspected:** author bibliography/description, not the full book. **Informs:** automatic reusable subprogram discovery has a long history. No numerical comparison, full algorithm reproduction, or claim that this reference solves SIML's proposed neural-local problem is inferred.

### R16 — Optimal Ordered Problem Solver

Jürgen Schmidhuber. **Optimal Ordered Problem Solver.** Preprint 2002; *Machine Learning* 54, 211–254, 2004. [Abstract and bibliographic record](https://arxiv.org/abs/cs/0207097).

**Inspected:** abstract and metadata. **Informs:** incremental problem solving can exploit previously found programs, preceding modern neural library systems. Abstract-level evidence supports the historical overlap only; its conditional theoretical results are not transferred to the proposed stochastic learner.

### R17 — Cascade-Correlation

Scott E. Fahlman and Christian Lebiere. **The Cascade-Correlation Learning Architecture.** NIPS 1989 proceedings, published 1990. [Official record](https://proceedings.neurips.cc/paper/1989/hash/69adc1e107f7f7d035d7baf04342e1ca-Abstract.html); [paper](https://proceedings.neurips.cc/paper/1989/file/69adc1e107f7f7d035d7baf04342e1ca-Paper.pdf).

**Inspected:** publisher abstract and relevant mechanism description. **Informs:** constructive growth and freezing of acquired units are established. The source proposal also cited the later recurrent variant; this plan deliberately uses the foundational feed-forward construction for its state-free baseline. Neither variant establishes reusable functional abstraction by growth alone.

### R18 — RigL

Utku Evci and collaborators. **Rigging the Lottery: Making All Tickets Winners.** ICML, 2020, PMLR 119. [Publisher paper page](https://proceedings.mlr.press/v119/evci20a.html).

**Inspected:** publisher abstract/method summary. **Informs:** dynamic sparse connectivity at a controlled nonzero budget. The planned dense-mask numerical adaptation tests rewiring as an alternative explanation; it cannot reproduce native sparse-kernel throughput or the original large-scale experimental results merely by using the name RigL.

### R19 — Neural Programmer-Interpreters

Scott Reed and Nando de Freitas. **Neural Programmer-Interpreters.** 2015 preprint, arXiv:1511.06279. [Abstract and record](https://arxiv.org/abs/1511.06279).

**Inspected:** abstract and bibliographic metadata. **Informs:** learned compositional program execution and subprogram reuse are not new. This record alone is not evidence of autonomous invention of the specific tiny numerical primitives proposed here.

### R20 — Compositional interfaces

Jelena Luketina, Jack Lanchantin, Sainbayar Sukhbaatar, and Arthur Szlam. **Compositional Interfaces for Compositional Generalization.** PMLR 274, 2025. [Publisher record](https://proceedings.mlr.press/v274/luketina25a.html).

**Inspected:** official abstract and metadata. **Informs:** compositional differentiable modules and shared control are relevant contemporary alternatives. It narrows broad novelty claims about modular interfaces; full experimental/implementation parity with SIML was not established.

### R21 — What/how separation in recurrent computation

Haozhe Shan, Sun Minni, and Lea Duncker. **Separating the what and how of compositional computation to enable reuse and continual learning.** NeurIPS, 2025; arXiv:2510.20709. [Full text](https://arxiv.org/html/2510.20709v1); [record](https://arxiv.org/abs/2510.20709); [proceedings](https://proceedings.neurips.cc/paper_files/paper/2025/hash/82d07a9f247048b85f78786ac80e6fbf-Abstract-Conference.html).

**Inspected:** abstract and full-text method passages. **Informs:** reusable recurrent computations, context-dependent organization, and continual-learning comparisons. Task/context information assumptions must be preserved in any adaptation; an abstract's high-level unsupervised language does not grant a baseline or SIML unrestricted task information.

### R22 — Recent modular expansion lead, access-limited

Ziye Fang, Bo Wan, Shangqi Guo, and Jian K. Liu. **A scalable complexity-regularized modular network for continual learning.** *Neurocomputing*, DOI: **10.1016/j.neucom.2026.135079**. [DOI](https://doi.org/10.1016/j.neucom.2026.135079); [publisher abstract page](https://www.sciencedirect.com/science/article/abs/pii/S092523122602477X).

**Access limitation:** search metadata surfaced September 2026 online availability and a 2027 issue assignment; direct publisher access returned an error. Full text was not verified. This remains a close modular-expansion lead, not evidence for detailed implementation or performance claims. The future issue date is not treated as an experiment already published in January 2027.

### R23 — PySR and SymbolicRegression.jl

Miles Cranmer. **Interpretable Machine Learning for Science with PySR and SymbolicRegression.jl.** 2023 preprint, arXiv:2305.01582. [Full text](https://arxiv.org/html/2305.01582v3); [repository and documentation](https://github.com/astroautomata/PySR).

**Inspected:** paper sections, README/install context, Apache-2.0 license indication. **Informs:** a strong numerical expression-search alternative, constant optimization, and a native Python/Julia local implementation path. Neither Julia installation nor a PySR run was performed for this plan. Symbolic grammar and complexity settings require recorded development tuning.

### R24 — Apple PyTorch acceleration documentation

Apple Developer. **Accelerated PyTorch training on Mac.** [Official documentation](https://developer.apple.com/metal/pytorch/). Accessed 14 October 2026.

**Inspected:** requirements and installation/backend guidance. **Informs:** Apple Silicon/MPS setup and the published compatibility conditions described in Section 8. Page labels are not an installed software guarantee or a measured M3 Max workload. Hardware, OS, Python, wheel availability, and operation/backward support must all be checked locally.

### R25 — PyTorch MPS documentation

PyTorch contributors. **MPS backend notes, API, and environment variables.** [Backend notes](https://docs.pytorch.org/docs/2.14/notes/mps.html); [API](https://docs.pytorch.org/docs/2.14/mps.html); [environment variables](https://docs.pytorch.org/docs/2.14/mps_environment_variables.html). Accessed 14 October 2026; version 2.14 pages served during inspection.

**Inspected:** official backend/API documentation. **Informs:** explicit synchronization, memory observations, backend availability, and fallback controls. These pages do not prove determinism, parity, or efficiency for the proposed dynamic graph loop. Disabling allocation protections is not an acceptable workaround for memory pressure.

### R26 — ODEBench

GPBench organization. **ODEBench repository.** [Repository](https://github.com/GPBench/ODEBench). Accessed 14 October 2026.

**Inspected:** repository README and available metadata. **Informs:** an adjacent differential-equation benchmark lead. A suitable data/code license was not established in this research, so it is not a required asset. G3 uses independently specified analytical vector fields and does not claim to reproduce ODEBench.

### R27 — Illustrative historical patent record

**US6532453B1**, genetic-programming problem solving with automatically defined stores, loops, and recursions. [Patent record](https://patents.google.com/patent/US6532453B1/en).

**Inspected:** public patent record for historical mechanism overlap. **Informs:** broad automatic-subprogram concepts also appear in patents. This was a targeted landscape check, not exhaustive patent searching, legal-status verification, novelty clearance, or freedom-to-operate advice. No legal conclusion is needed for the experimental recommendation.

### R28 — Reptile / first-order meta-learning

Alex Nichol, Joshua Achiam, and John Schulman. **On First-Order Meta-Learning Algorithms.** 2018 preprint, arXiv:1803.02999. [Paper record](https://arxiv.org/abs/1803.02999); [author explanation](https://openai.com/index/reptile/); [reference repository](https://github.com/openai/supervised-reptile).

**Inspected:** abstract and author material. **Informs:** a strong simple dense adaptation baseline using within-task optimization and initialization updates. The proposed current-PyTorch numerical implementation is an adaptation; no old repository environment or published benchmark was reproduced.

### R29 — FePySR

Zhiming Yu, Wangtao Lu, and Xin Lai. **FePySR: A Neural Feature Extraction Framework for Efficient and Scalable Symbolic Regression.** 12 May 2026 preprint, arXiv:2605.12704. [Full text](https://arxiv.org/html/2605.12704v1); [repository](https://github.com/laixn/FePySR); [repository license](https://raw.githubusercontent.com/laixn/FePySR/main/LICENSE).

**Inspected:** methods/appendix, README, and explicit CC BY-NC 4.0 repository license. **Informs:** neural mathematical-feature extraction before PySR is a relevant local numerical competitor. Per-dataset extraction differs from SIML's persistent cross-task hierarchy. Native runtime/compatibility were not measured; permission and faithful reproduction are gates. A referenced “InceptionSR” lead was not independently verified and is not used as established evidence.

### R30 — Symbolic regression with a learned concept library

Arya Grayeli, Atharva Sehgal, Omar Costilla-Reyes, Miles Cranmer, and Swarat Chaudhuri. **Symbolic Regression with a Learned Concept Library.** NeurIPS, 2024; arXiv:2409.09359. [Paper record](https://arxiv.org/abs/2409.09359); [project](https://trishullab.github.io/lasr-web/).

**Inspected:** abstract and project/metadata context. **Informs:** LaSR combines language-model concept generation with evolutionary symbolic regression. It is a relevant library-learning reference, not the same mechanism or a free local baseline; no full LLM-backed reproduction is claimed.

### 14.2 Search-coverage record

The investigation was organized by mechanisms that could change the decision, not by an arbitrary number of papers. Search and citation tracing covered the following axes:

| Search axis / representative terminology | Material result for this plan |
|---|---|
| Program/library learning, abstraction invention, DreamCoder successors, differentiable program synthesis | DreamCoder, Stitch, HOUDINI, AbstractBeam, LILO, E-Stitch; these establish overlap and the need for a native centralized baseline. |
| Local neural modularity, constructive networks, dynamic expansion, compositional interfaces, continual reuse | Modular meta-learning, LMC, Cascade-Correlation, RigL, Luketina, Shan, and the access-limited Fang lead; these prevent attributing growth, rewiring, or ordinary reuse to invention. |
| Utility problem, speed-up learning, automatically defined functions, incremental program search | COMPOSER, Koza, OOPS, and a historical patent; these challenge claims that preserving computations or considering their costs is new. |
| Future transfer utility, bounded synthesis, abstraction selection, DSL optimization | AMaze, Cuculich's September 2026 preprint, prospective compression; these make same-gate comparison indispensable. |
| Neural mathematical operators, learned symbolic features, numerical concept libraries | PySR, FePySR, LaSR; these strengthen numerical baselines and distinguish low-level feature invention from cross-task accumulation. |
| Reuse ablations, negative evidence, implementation configurations, local compatibility | The single-use-library study, native AbstractBeam configuration, Stitch interfaces/licenses, Apple/PyTorch MPS docs; these shape causal tests and laptop feasibility gates. |

Original references were checked against primary records where accessible; close works were followed backward for mechanisms and forward for successors. Consequential implementation constraints were checked in official documentation and selected repository files rather than inferred from paper abstracts. Selected PDF algorithm/figure pages were inspected visually. Search-result snippets were used to locate sources, not as substitutes for accessible technical evidence; the Fang access limitation is explicitly retained.

Later targeted checks consolidated the selected comparison around local proposals versus centralized extraction under a common transfer gate. The numerical-feature pass added FePySR as a relevant permission-gated competitor but did not justify changing the core research question. Remaining blind spots include unpublished/poorly indexed work, very recent proceedings, access-limited papers, unverified implementation behavior, full patent coverage, and domain-specific module systems outside the searched scope. This is a current research assessment, not proof that no additional prior art exists.

### 14.3 Evidence boundary at handoff

**Verified from inspected sources:** the mechanisms and implementation/license facts with the scope stated in the ledger. **Actually executed:** the limited Linux CPU semantic, snapshot, and analytical checks in Section 4.6. **Derived:** bounded-range arguments, search/headroom calculations, memory formulas, and proposed statistical design. **Proposed, not executed:** all SIML learning experiments, native baseline reproduction, Mac profiling, confirmatory comparisons, and downstream transfer. **Unresolved blocking gates:** actual laptop capacity/support, pinned compatible dependencies, optional-asset permissions, baseline reproduction, pilot sensitivity, and affordable independent-world precision.

The coding agent must preserve these distinctions in its final report. A complete implementation with negative or inconclusive results is a valid outcome; an attractive result with compromised information access, missing costs, or weak baselines is not.

[R01]: https://doi.org/10.1145/3453483.3454080
[R02]: https://doi.org/10.1145/3571234
[R03]: https://github.com/mlb2251/stitch
[R04]: https://arxiv.org/html/1804.00218v2
[R05]: https://proceedings.mlr.press/v87/alet18a/alet18a.pdf
[R06]: https://arxiv.org/html/2111.07736
[R07]: https://arxiv.org/html/2405.17514v2
[R08]: https://cdn.aaai.org/AAAI/1992/AAAI92-037.pdf
[R09]: https://adamcuculich.com/research/on-the-value-of-abstractions/
[R10]: https://xinpl.github.io/papers/popl26b.pdf
[R11]: https://arxiv.org/html/2605.09985
[R12]: https://pldi26.sigplan.org/details/egraphs-2026-papers/7/E-Stitch-Top-Down-Library-Learning-for-E-Graphs
[R13]: https://arxiv.org/html/2310.19791v2
[R14]: https://arxiv.org/html/2410.20274v1
[R15]: https://www.genetic-programming.com/jkpubs94.html
[R16]: https://arxiv.org/abs/cs/0207097
[R17]: https://proceedings.neurips.cc/paper/1989/hash/69adc1e107f7f7d035d7baf04342e1ca-Abstract.html
[R18]: https://proceedings.mlr.press/v119/evci20a.html
[R19]: https://arxiv.org/abs/1511.06279
[R20]: https://proceedings.mlr.press/v274/luketina25a.html
[R21]: https://arxiv.org/html/2510.20709v1
[R22]: https://doi.org/10.1016/j.neucom.2026.135079
[R23]: https://github.com/astroautomata/PySR
[R24]: https://developer.apple.com/metal/pytorch/
[R25]: https://docs.pytorch.org/docs/2.14/notes/mps.html
[R26]: https://github.com/GPBench/ODEBench
[R27]: https://patents.google.com/patent/US6532453B1/en
[R28]: https://arxiv.org/abs/1803.02999
[R29]: https://arxiv.org/html/2605.12704v1
[R30]: https://arxiv.org/abs/2409.09359
