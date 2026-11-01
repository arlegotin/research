# Research and proof-of-concept plan: representation diversity for reliable local LLM reasoning

**Research cutoff:** 27 October 2026. **Target:** one MacBook Pro with an M3 Max; actual memory, storage and macOS must be discovered. **Deliverable type:** executable research plan, not a report of completed model experiments. **Status:** ready for prerequisite checks and implementation; positive results and methodological novelty are unproven.

## Reading and execution guide

This document is a complete handoff for a coding agent. It includes the research assessment, selected experiment, algorithms, evaluation contract, hardware gates, work packages and reporting rules. All commands under “proposed project commands” are interfaces the agent must implement; they do not already exist. Commands identified as operating-system or upstream commands are real commands, subject to checking the installed version. References use ordinary links and identifiers in §12.

The agent may make routine, reversible implementation choices, fix defects and resume interrupted work without additional approval. It must record a methodological amendment before changing a hypothesis, population, primary outcome, baseline information, resource comparison or confirmatory analysis. An amendment made after inspecting protected results requires fresh confirmation data. Do not spend money, require cloud inference, publish results externally or accept third-party account conditions on the user's behalf as a hidden prerequisite.

## 1. Executive assessment and recommended direction

### 1.1 Verdict

**The broad idea is scientifically plausible and substantially established.** Translating language into programs, constraints, logical forms or other representations; comparing independent solutions; and checking translations with counterexamples already exist. A particularly close October 2026 preprint, *Validating Autoformalized Specifications via Contrapuntal Languages*, explicitly connects language-induced biases to structurally different formal languages [R08]. CLOVER already uses solver-generated separating interpretations and source-language judgments to choose between formalizations [R04]. Mixture-of-Thought already combines natural-language, code and truth-table reasoning in training and inference [R11]. None of these general principles can be claimed as new here.

**Recommended primary direction:** an exact, budget-controlled study of whether independently inferred **intensional formulas** and **extensional relation tables** reduce correlated semantic errors enough to improve a local LLM system over strong single-representation and synthesized-language baselines. Implement a small, fixed representation portfolio and a shared semantic selector. Test it on fresh bounded logic problems with exact semantics, then attempt transfer to human-authored logical reasoning. This is initially a replication and empirical extension, with a possible engineering contribution. A new scientific mechanism has not been established.

The substantive unresolved question is narrower than “do multiple representations help?”:

> Under equal information and measured local inference budgets, does changing the semantic representation of independently generated candidates improve the probability of a correct answer, beyond resampling, compositional translation, source-grounded checking and automatically synthesized representation diversity?

An affirmative answer must survive all translation costs, rejected candidates, common mistakes, strong baseline tuning and independent confirmation. A negative answer is useful if it identifies whether the missing ingredient is representation competence, complementary errors, reliable selection, or sufficient scale.

### 1.2 What the PoC can and cannot establish

The core can establish a causal effect of the specified format, prompt and serialization intervention on a fixed model and a defined problem distribution, relative to validated **LLM-based** methods. It can distinguish wrong translations from wrong inference, and test whether lower shared error explains a gain beyond more compute. It cannot isolate a distinct internal cognitive mode from this intervention alone. It can establish an actual M3 Max accuracy–latency–coverage tradeoff after local measurements.

It cannot establish a universal escape from language, better general intelligence, human-like thought, independent semantic correctness from agreement alone, or a new foundation-model architecture. The primary intervention improves an **LLM-plus-formal-reasoner system**; the frozen base weights remain unchanged. Training and latent-channel research are separate, evidence-triggered branches (§10), with different claims and controls.

Synthetic success alone supports only the declared bounded setting. Human-authored benchmark transfer supports a broader benchmark claim, not deployment readiness or professional decision reliability. A deterministic parser for the generator's controlled language is a legitimate strong simple comparator; if it solves that setting perfectly, the LLM study remains mechanistic rather than a useful replacement for that parser.

### 1.3 Why this direction was selected

| Direction | Value and distinctive uncertainty | Principal difficulty | Decision |
|---|---|---|---|
| More prose prompts, paraphrases or majority votes | Cheap strong controls; can expose framing sensitivity | Does not isolate a change in representation | Mandatory baselines |
| Formula/table portfolio with exact semantics | Directly tests complementary semantic errors; no training prerequisite | Translation and source-judge errors may be shared; tables grow exponentially | Primary experiment |
| Automatically synthesized contrapuntal languages | Closest contemporary representation-diversity alternative | Language synthesis, grammar following, seed bias and extra inference cost | Mandatory closest-method comparison or explicit claim restriction |
| Learned representation router or multi-representation fine-tuning | Could amortize successful representation choices | Existing typed-thinking/MoT/routing work; training confounds | Only after fixed portfolio evidence |
| Continuous latent feedback or recurrent computation | Tests a legitimate lower-level channel/compute hypothesis | Existing methods, training dependence, causal attribution and Mac porting | Separate mechanistic branch, not dismissed on scale alone |
| Images or diagrams | Appropriate when spatial information is indispensable | Rasterization/vision introduces a different bottleneck; current task has exact symbolic structure | Reserve for a later spatial task, not cosmetic diagrams |

This selection follows the proposed causal idea—flexible representations plus empirical checking—and the ability to isolate it. It is not a claim that an orchestration layer is inherently more promising than neurons, circuits, learning rules or architectures.

## 2. Problem statement, assumptions and hypotheses

### 2.1 Beneficiary and workflow

The immediate beneficiary is a developer using a local open-weight language model to interpret explicit rules and answer deductive questions. A representative workflow is: receive a natural-language rule set, extract its semantics, determine whether a query follows, and return the answer with a machine-checkable intermediate artifact. The initial workload concerns bounded Boolean policies and constraints, not legal advice, unrestricted commonsense or noisy real-world causal inference.

The proposed bottleneck is **representation-dependent semantic error**: the model may omit a condition, reverse an implication, mistake inclusive for exclusive disjunction, or mishandle negation when converting prose into an intermediate form. Multiple samples in one familiar representation may repeat the same mistake. A different representation could require a different local computation and produce a complementary error pattern. This is a hypothesis; it must be measured.

Three other bottlenecks must remain distinguishable: serialization errors, insufficient computation on correctly represented facts, and inherent ambiguity or missing information in the source. A better serializer does not by itself demonstrate better reasoning; an exact solver can correctly solve an incorrectly translated problem.

### 2.2 Definitions

Let a source instance be x=(V,S,q), where V is a public atom registry, S is a list of source sentences, and q is a public query literal. Each atom denotes a Boolean state. A complete world w belongs to Ω={0,1}^|V|. The hidden intended premise formula is G; its denotation is the set of worlds satisfying it. The evaluator alone receives G and the generation/transformation metadata.

Representation r has a partial model translator E_r(x), a deterministic parser and compiler C_r, and generated candidate F_r=C_r(E_r(x)). Partial means malformed, truncated or unsupported outputs are possible and recorded. The exact reasoner computes the answer from F_r and q. The selector may use source text, generated candidates, solver equivalence tests and source-conditioned witness judgments; it never sees G.

Two candidates are semantically equivalent when their truth functions agree on **all worlds of the declared domain**, not merely on the current query. For the finite core, equivalence is established by exhaustive truth vectors or UNSAT of XOR. “Verified” must be qualified: the compiler/solver can verify a mathematical relation between formal artifacts; it cannot prove that those artifacts capture unrestricted human intent.

The primary labels are ENTAILED, CONTRADICTED and UNDETERMINED for satisfiable premises. A separate INCONSISTENT status handles unsatisfiable premises. A method failure or solver timeout is never the semantic label UNDETERMINED.

### 2.3 Established evidence versus assumptions

**Established capabilities:** formal solvers can decide bounded Boolean entailment; code/logic-assisted LLM reasoning and representation ensembles are existing methods [R01–R05, R08, R11]. Transformers already use continuous hidden states. Emitting text constrains the next supplied token embedding, but does not erase earlier continuous information from their computation or caches. Latent-feedback, pause-token and recurrent-depth alternatives already exist [R15–R20].

**Human evidence:** the cited 2026 PNAS logical-reasoning/aphasia study is real; its accessible abstract supports a dissociation on the tested tasks. Full participant-level methods were not available in the inspected text, so this plan makes no sample-size claim. The 2024 Nature perspective supplies broader context. Neither establishes the corresponding mechanism in an LLM [R21–R22]. The other psychological anecdotes are motivations, not premises required for this PoC.

**Assumptions requiring tests:** the chosen local model can use both representations; their errors are usefully different; the selector can exploit that difference; overhead does not erase it; exact synthetic effects persist under unfamiliar wording and at least one human-authored workload. None follows from linguistic relativity.

### 2.4 Hypothesis and falsification ledger

| ID | Testable hypothesis and predicted observation | Strong alternative explanation | Observation that weakens it / discriminating test |
|---|---|---|---|
| H1 | Independent formula/table candidates have useful semantic complementarity at whole-task or per-clause level | The second format is simply more accurate, or there are more samples | Compare F/F, T/T and F/T at matched cost; report marginal/joint errors and both whole-candidate and composed oracle headroom |
| H2 | A shared source-grounded semantic selector turns complementarity into higher unconditional accuracy than validated LLM-based alternatives at a fixed budget | Extra checking, longer prompts, more favorable rejection, or weak baseline generation | Compare with the best single format, prompt-diverse same-format ensemble, tuned CLOVER and contrapuntal adapters; count abstentions as failures |
| H3 | The gain persists under known semantic transformations of presentation or query | Memorized renderer, label priors or copying a stereotyped chain | Held-out templates and structures; invariance under renaming/order; equivariance under query negation |
| H4 | The selected system improves a useful cost/accuracy operating point | Better accuracy is purchased with disproportionate inference or preparation | Local end-to-end frontier, coverage, cost per correct answer and amortized development costs |
| H5 | The original factor-table effect transfers to supported human-authored propositional text | The generator effectively teaches the answer or supplies privileged structure | A preregistered propositional FOLIO subset; the separately named full-FOL gate variant does not establish unchanged-mechanism transfer |
| H6, optional | A trained internal/latent channel carries causally useful problem information beyond extra compute | Pause slots, answer already encoded, privileged supervision | Explicit-chain/filler controls, matched supervision and targeted content interventions (§10.3) |

H2 is the confirmatory primary claim. H1 and H3 are necessary attribution evidence for the representation explanation. H4 is needed for practical advantage. H5 is needed for transfer claims. H6 is a separate project branch.

### 2.5 Strongest case for and against

The strongest favorable case is not that tables are universally better. It is that implication syntax and explicit admissible cases invite different local computations. A formula decoder may reverse “only if,” while a case-based decoder may correctly identify the one forbidden assignment. Exact composition then handles long chains without requiring the LLM to carry every inference in prose. Independent source parsing can expose omissions that a deterministic rewrite of one wrong formula would preserve.

The strongest adverse case is that both outputs come from the same pretrained model, source interpretation and vocabulary. They may inherit the same wrong assumption. New syntax can reduce individual competence, increase prompt length or shift errors rather than fix them. Formal inference may already dominate direct language reasoning, leaving little room beyond a strong formalizer. The exact benchmark may be better solved by a small parser. A stronger local checkpoint or more same-format samples may erase the entire advantage.

### 2.6 Headroom and limiting cases

Measure three ceilings on development data: exact solver with gold formula; best candidate under an evaluator-only oracle; and best per-clause candidate followed by exact composition. The first diagnoses the harness. The second bounds selection gains for a fixed pool. The third bounds gains from compositional selection and is explicitly privileged. Never expose these choices to deployable methods.

If B is baseline accuracy and O is the relevant oracle accuracy of a cost-matched portfolio, O−B is an empirical ceiling, with uncertainty. Use the **composed** oracle for a selector that mixes clauses; lack of a wholly correct candidate does not remove that headroom. A fixture must include F with only clause 1 correct and T with only clause 2 correct, whose correctly composed result is valid. If even the applicable ceiling's upper bound is tiny, expensive selection work is poorly justified. If headroom is large but selection does not improve, investigate source judgment rather than adding formats blindly.

In the limiting case of identical denotations or deterministically translated candidates, representation agreement adds no independent evidence. In the limiting case of a perfect controlled-language parser, no LLM can improve accuracy. In the limiting case of a uniformly wrong shared interpretation, arbitrary syntactic diversity cannot restore missing source meaning.

## 3. Research findings and novelty assessment

### 3.1 Closest prior art and reuse matrix

The entries below summarize inspected primary sources. Published outcomes are evidence about those sources' settings, not controlled comparisons with this project. §12 records stable links and access limitations.

| Prior work | Problem/mechanism and overlap | Evidence, assumptions and costs | Reuse and implication |
|---|---|---|---|
| PAL / Program of Thoughts [R01–R02] | Generate executable programs; move computation out of prose | Numerical/reasoning benchmarks with older model/API settings; executable programs can still encode wrong semantics | Reuse separation of translation/execution and prompt ideas; core principle established |
| SatLM / Logic-LM [R03] | Declarative constraints or logical forms plus solvers and repair | Solver correctness is conditional on generated specification; repeated translation consumes inference | Strong simple formalization baseline; reuse solver adapters where licensed |
| CLOVER [R04] | Compositional translation, semantic voting and counter-interpretation adjudication against source | Mainly GPT-family experiments; decomposition and checking add cost | Closest selector baseline; public repository, source-license and backend-port gate |
| Semantic Self-Verification [R05] | Positive/negative instantiations test translated constraints | Selective verification can have low coverage; source-generated tests are fallible | Paper-defined adapter; linked repository was empty when inspected |
| COVER / roundtrip repair [R06–R07] | Typed repair with witnesses; cyclic NL/formal checking | Oracle/test-assisted variants can have extra information; roundtrips can converge to consistent errors | Diagnostic alternatives; do not award hidden reference models to treatment |
| Contrapuntal Languages [R08] | Synthesized formal languages intended to reduce shared errors; cross-language equivalence | Recent preprint; seed-dependent construction, behavioral specification evaluation, substantial generation overhead | Closest representation-diversity baseline; fixed tables do not reproduce its synthesis or satisfy its theorem automatically |
| Semantic triangulation [R09] | Inversion/decomposition/enumeration yield independently checked code solutions | Code-oriented setting; full implementation not audited here | Further defeats broad novelty; relevant if extending to program synthesis |
| VERGE / adaptive solver composition [R10, R12] | Semantic consensus, routing, localization or solver selection | Nonlocal published setups; small-model formalization can fail despite correct routing | Routing alone is not new; avoid unnecessary controller training |
| Mixture-of-Thought / TypedThinker [R11, R14] | Train/use several reasoning forms, including truth tables or thinking types | Different training and inference recipes; additional supervision and sampling | Multi-format inference/training is established; compare inference controls fairly |
| Self-consistency / adaptive verification [R13] | More independent reasoning paths and budget allocation | Additional computation can explain gains without representational change | Mandatory generic controls |
| Coconut / CODI / pause and filler tokens [R15–R17] | Continuous feedback, distillation or extra computation positions | Task/training-dependent; CUDA research launchers are not Mac-ready by declaration | Architectural alternatives and compute controls, not primary external-representation baselines |
| Recurrent depth / soft thinking / switching / invariance [R18–R20] | Reuse computation, mixture embeddings, latent/text switching, aligned trajectories | Existing methods and mixed evidence; equivalence to useful parallel reasoning is not guaranteed | No blanket novelty for “nonverbal,” “adaptive” or “invariant” reasoning |
| Metamorphic and competence-based evaluation [R23–R25] | Check known relations among transformed inputs and correctness profiles | Agreement can be consistently wrong; compiling subsets can hide failure | Reuse evaluation principles; robust scoring itself is not a new method |
| AWS Automated Reasoning checks [R26] | Translate policy language into formal variables/rules and validate stages | Deployed cloud system; translation quality remains a dependency | Deployment precedent, excluded from required local workload |

### 3.2 What already exists; what differs

1. **Exists:** symbolic intermediates, executable reasoning, multi-format ensembles, semantic equivalence voting, source-grounded witness checking, adaptive selection, latent representations and metamorphic evaluation.
2. **Reuse:** pretrained local models; MLX or a validated Metal backend; Z3; safe AST/JSON parsing; source-algorithm recipes; licensed dataset text; ordinary paired inference. Do not build a new neural architecture or large orchestration framework merely to test H2.
3. **Proposed difference:** a fixed, complete intensional/extensional representation pair, independently parsed from source, evaluated with exact denotational ground truth and explicit error-dependence measurements at local compute parity. Its usefulness against strong synthesis-based diversity is unresolved.
4. **Potential value:** simpler representation construction could reduce grammar-synthesis overhead and common seed dependence. Better semantic complementarity could improve accuracy or selective reliability. These are deductions to test, not results.
5. **Isolating experiment:** substitute representation family while holding model, source, selector, budget, generated-candidate count controls and evaluation instances constant. Separate selection-only analysis from full pipeline cost.
6. **Invalidating novelty:** a prior implementation of the same fixed pair/selector/evaluation or a broader method encompassing it eliminates a method claim. Existing MoT and contrapuntal work already eliminate broad claims. A successful replication can still be useful; do not rename it to manufacture novelty.

The appropriate starting label is **replication plus empirical extension**. A reproducible reduction in shared semantic errors under this exact intervention would be a new empirical result only after comparing against prior results and artifacts. Lower latency with equal accuracy would be an engineering result. Neither is automatically groundbreaking.

### 3.3 Reconciliation of apparently conflicting evidence

Latent reasoning papers can report benefits while later intervention studies find inactive chains: the task, scale, training recipe, supervision and causal question differ. The October 2026 taxonomy study distinguishes chain presence from chain content; some logical chains were dispensable while tested arithmetic chains depended on content [R19]. This argues for targeted interventions, not a universal rejection of continuous reasoning.

Representation ensembles can improve accepted-answer precision while reducing coverage or increasing compute. Those results do not establish higher unconditional utility. Likewise, compilation instability and semantic instability are different: a Lean autoformalization study reports a setting where much surface sensitivity arose before successful compilation [R24]. The present plan measures both stages and shares generally useful formatting improvements across arms.

The contrapuntal paper's structural condition is not a theorem that semantic errors become independent. Shared seeds, source ambiguity and model priors can survive different syntax. Conversely, a fixed table need not satisfy that paper's structural condition to be empirically useful. Test dependence directly, and retain both possibilities.

## 4. Selected mechanism and implementation specification

### 4.1 Minimal complete experiment

Implement three candidate-generation conditions first: two formula candidates (F/F), two table candidates (T/T), and one independently generated candidate of each kind (F/T). Use the same source, vocabulary, solver and selector. Run this on a development slice where both formats can express every problem. Compare sample-matched conditions for mechanism diagnosis and time-matched conditions for end-to-end claims. Add the strongest local baseline adapters before confirmation.

The minimum complete F/T run contains two independent source-to-representation generations, safe parsing/compilation, exact comparison, source-grounded resolution when useful, and final inference. Merely rendering a generated formula as a table tests presentation, not independent semantic translation.

### 4.2 Public input and private oracle schema

~~~json
{
  "task_id": "opaque-id",
  "group_id": "opaque-base-instance-id",
  "view_id": "opaque-render-id",
  "atoms": [{"id": "p0", "meaning": "the access badge is valid"}],
  "sentences": [{"id": "s0", "text": "..."}],
  "query": {"atom": "p0", "negated": false},
  "semantics": {"logic": "classical_boolean", "closed_world": false}
}
~~~

The atom registry and sentence boundaries are **public task inputs** in the finite core, available to every baseline, including direct reasoning. They make the initial question conditional on a known vocabulary and segmentation; they are not inferred from gold for only one arm. IDs must be opaque and must not reveal label, split, difficulty or formula structure. Neither scope lists nor correct clauses are supplied. A separate end-to-end extraction diagnostic removes this assistance (§8).

The scorer-only record contains the gold AST, denotation hash, label, generator version/seed, structural identity, renderer transformation map and diagnostic tags. The inference process must not receive or be able to open that record. Do not place it in prompt metadata, filenames or retrieved examples.

### 4.3 Intensional representation F

Use a typed JSON expression tree with Boolean constants, declared atom references and operators NOT, AND, OR, IMPLIES, IFF and XOR. AND/OR accept two or more children; all other arities are fixed. Output exactly one expression per source sentence. Every sentence ID must occur exactly once. The conjunction of these expressions is the premise set. Unknown symbols, unexpected properties, duplicate IDs, invalid arities and partial JSON are errors. **Default validation is all-or-nothing per generation:** if any required record fails, that generation contributes no clauses. This policy also applies to T and matched compositional outputs; record-level salvage is a separately matched ablation, never an implicit treatment advantage.

Example: “If the badge is valid, the gate is open” with atom IDs b and g:

~~~json
{"sentence_id":"s0","expr":{"op":"implies","args":[{"atom":"b"},{"atom":"g"}]}}
~~~

Do not evaluate generated Python. Convert whitelisted AST nodes to an internal immutable tree and then to solver expressions. Choose an explicit JSON schema version. Parse literal true/false as Booleans; reject strings such as "false" where a Boolean is required.

### 4.4 Extensional representation T

The independent table decoder outputs one complete Boolean relation per source sentence. For the core distribution, each sentence references at most four distinct atoms. The decoder must choose its ordered scope from the public registry, then give a Boolean outcome for every assignment in lexicographic binary order, with the last scope position changing fastest.

~~~json
{"sentence_id":"s0","scope":["b","g"],"allowed":[true,true,false,true]}
~~~

This represents rows 00, 01, 10, 11, so only a valid badge with a closed gate violates the rule. Scope order is part of the contract; it must not be sorted without permuting the outcome vector. Duplicate variables or an outcome count different from 2^k are invalid. Zero scope requires exactly one Boolean, representing a constant. Missing rows have **no default**. All-true and all-false factors are valid semantic objects, not parser failures.

For scope (v1,…,vk) and allowed vector a, compile deterministically:

\[
T(v)=\bigvee_{j:a_j=1}\ \bigwedge_{\ell=1}^{k}
\begin{cases}v_\ell,&\text{bit}_{\ell}(j)=1\\\neg v_\ell,&\text{bit}_{\ell}(j)=0.\end{cases}
\]

An empty outer disjunction is false; an empty inner conjunction is true. A factor may omit a relevant source atom and still be syntactically valid; that is a semantic error for the evaluator to detect. Rejecting it using hidden gold scope would leak information. For higher-arity stress tests, either allow the full vector and charge its cost or record an unsupported outcome; never silently truncate it.

This fixed representation pair does not claim the contrapuntality theorem. Its expressivity on the declared core follows from ordinary Boolean truth functions, not from syntax-shape diversity.

### 4.5 Solver semantics and invariants

For premise P and query q, first check SAT(P). If UNSAT, return INCONSISTENT. If SAT, check SAT(P AND NOT q) and SAT(P AND q). Return ENTAILED when the first is UNSAT, CONTRADICTED when the second is UNSAT, and UNDETERMINED when both are SAT. Any solver UNKNOWN, timeout or internal error is an execution status, not a semantic answer. Never infer entailment from an inconsistent premise set by classical explosion.

For equivalence, check XOR(F,G) over exactly the same declared vocabulary and assumptions. UNSAT establishes equivalence; SAT supplies a separating world; UNKNOWN establishes neither. Do not add gold premises as assumptions during comparison. For finite tasks, complete every witness assignment explicitly and reevaluate it in the independent interpreter.

The evaluator computes translation fidelity by comparing full denotations with G, not by comparing surface ASTs or the answer to q. A lucky correct answer can coexist with a wrong translation. Report both.

### 4.6 Candidate generation and shared selection

Each generation starts a fresh conversational context containing only the source, public registry, representation contract, allowed examples from development and budget parameters. It must not see another candidate, its answer, the hidden transformation map or the oracle. Use independent derived seeds and log all raw outputs. Fresh contexts do not imply statistically independent model errors.

Implement an immutable candidate object with representation family, source-sentence outputs, compiled AST, parsing status, prompt/output hashes, sampler seed, generation time and tokens. All valid candidates enter semantic classes. Also retain invalid outputs in the cost and failure denominator.

Use a shared **CLOVER-style bounded selector** for the key attribution contrast. Retain a separate native compositional CLOVER adapter to avoid weakening the closest baseline. The shared selector's exact project definition is:

~~~text
for each source sentence, in a deterministic recorded order:
    collect valid candidate clauses and group exact-equivalent denotations
    order classes by support count, then a seeded permutation fixed before inference
    current = first class
    for challenger in remaining classes while audit budget remains:
        find witnesses in current AND NOT challenger, and challenger AND NOT current
        for each existing witness in the fixed order:
            complete the world over the public atom registry
            ask a fresh source-only judge whether this world obeys this sentence
            judge returns ALLOWED, FORBIDDEN, or UNCERTAIN
            if an unambiguous judgment contradicts current's truth value:
                current = challenger
                break
            if an unambiguous judgment contradicts challenger's truth value:
                retain current
                break
        unresolved comparison: retain higher original support; deterministic tie rule
    select current; mark whether the choice had unresolved disagreement
conjoin selected clauses; run exact query classification
~~~

The judge sees the sentence, atom meanings and complete world, but neither candidate syntax nor the eventual query answer. Ask “Does this complete state obey the stated rule?” Its judgment is fallible evidence, not a proof. Fix witness order, tournament ordering, tie behavior and uncertainty policy in configuration. A cached judgment is reusable only for identical source/world/judge configuration; charge its original generation cost in cold-task comparisons.

If no valid clause exists for a sentence, the method cannot compose a complete premise set. Its standard full-coverage mode uses the same development-frozen fallback as its matched comparator: spend remaining budget on one fresh formula translation, then one direct reasoning answer if feasible. Its selective mode abstains. All fallback cost and outcomes count. Do not drop a missing sentence or quietly replace it with true.

The tournament above is the project's shared selector, not a claim of byte-for-byte reproduction of CLOVER. The native adapter must preserve the source algorithm separately (§5.2). Distinguish selector-caused harms from candidate limitations by replaying identical saved pools through several selectors offline.

### 4.7 Allocation and stopping inside an instance

Start with a fixed portfolio, not a learned router. On development data, compare equal candidate shares, cost-balanced shares and the best single-format allocation. Reserve audit/fallback time using observed costs. Freeze the selected allocation before confirmation. Do not give F/T more LLM calls or a better source judge without a matched arm.

A method receives a per-instance **active elapsed-time** allowance and a token safety cap derived from it. The budget starts after the shared model has been loaded/warmed; all task-specific preparation and computation count. Before any operation, check its conservative completion estimate and pass the actual remaining time to solver/synthesis watchdogs. After each complete reasoning result, commit an answer snapshot and its active-time timestamp. At budget expiry, return the latest fully computed snapshot committed within the allowance, or abstain if none exists. Late outputs cannot alter it. A deadline failure therefore means **no timely complete answer**, not the loss of an earlier timely answer because optional work overran.

Metal kernels are not assumed to be preemptible. Poll streamed generation at token boundaries, cancel further work on expiry and log the overshoot; stop/synchronize unfinished work before the next timing block. Report observed delivery latency and overshoot separately, plus the prespecified strict-delivery sensitivity analysis. This is a measured anytime operating contract, not a hard-real-time guarantee. No selection rule can use the correct answer.

Adaptive allocation is a later ablation: compare “spend the next call on the other representation” against “spend it on the same representation,” using only observable disagreement, invalidity and remaining cost. A learned policy requires its own development split and comparison with a fixed portfolio. Its novelty is not assumed.

### 4.8 Reference prompts, sufficient to start implementation

**Formula translator:** “Translate every source sentence into the specified Boolean JSON AST. Use only the declared atoms. Preserve negation and the direction of implication. The absence of a fact does not imply its negation. Return one record for every sentence ID, exactly once. Do not solve the query or infer extra source premises. [Schema, development examples, registry, source sentences].”

**Table translator:** “Independently describe which complete local truth assignments obey each source sentence. Choose the atoms that the sentence constrains. For the ordered scope, list all 2^k allowed/not-allowed Booleans in binary order, last atom fastest. Include every row. Do not first copy a supplied formula; none is provided. Return one record per sentence ID. [Schema, development examples, registry, source sentences].”

**Witness judge:** “Consider only the stated rule and this complete state. Is this state permitted by the rule? Treat 'or' as inclusive unless the text explicitly says exactly one. Return ALLOWED, FORBIDDEN, or UNCERTAIN. Do not infer unstated facts. [Original sentence, registry meanings, complete state].”

These are initial templates, not secretly optimized winning prompts. Tune the same scope of wording, example count and decoding options for all relevant arms on development only. Save exact final bytes.

## 5. Baselines, data and confirmatory evaluation

### 5.1 Define the evaluation before scaling implementation

The evaluation unit is an independently generated underlying task, not a sentence, candidate, witness, prompt variant or repeated timing. The main estimand is average correct-answer probability under the declared task/renderer mixture at a fixed local time budget. For confirmation, sample the stratum independently with **uniform probability over the feasible cells frozen after generator validation**, then draw the underlying task and one renderer. Do not force balanced confirmatory cell counts; the unweighted task mean estimates this mixture. The pilot remains balanced. A separate preregistered robustness sample evaluates multiple views per task and clusters them together.

The primary intervention is the full F/T system, including selection, fallback and overhead. Select its comparator **from validated LLM-based methods B01–B09 and A01–A02**, using development data. Also include the closest validated contrapuntal method and the best single-format/shared-selector method when different from that winner. Merge duplicate comparators. B00 remains prominently reported, but is a practical ceiling, not a member of this restricted H2 comparison: its expected perfect performance makes overall superiority on controlled syntax impossible. A claim against “the strongest evaluated local LLM-based methods” requires success against every declared primary comparator with multiplicity control. H4 practical competitiveness requires supported human text on which B00 is not a complete solution.

### 5.2 Baseline register

| ID | Exact local condition | Tuning and reproduction requirements | Role |
|---|---|---|---|
| B00 | Deterministic parser for the documented controlled-language grammar, then exact solver | Include all published renderer rules; fail explicitly on unsupported novel text | Strong task-specific simple method and ceiling; not an oracle hidden from users |
| B01 | Direct answer and the model's supported native reasoning mode | Compare documented mode settings on development; enough output budget to finish | Strong single-call baseline |
| B02 | Independent native reasoning/CoT samples with answer plurality | Tune temperature and sample allocation; count reasoning tokens and invalid outputs | Generic test-time compute control |
| B03 | One formula translation and exact solver | Same schema, examples and parser repairs as treatment | Formalization baseline |
| B04 | Same-format repeated translations, semantic plurality or answer plurality | Compare both selectors; best local configuration chosen on development | Strong resampling baseline |
| B05 | Compositional CLOVER adapter with native semantic/counter-interpretation selection | Preserve decomposition and tournament; replace API transport with local backend | Closest source-grounded verification baseline |
| B06 | Contrapuntal-language synthesis and equivalence acceptance | Preserve seed generation, synthesis, independent grammar-conditioned draws, failure handling and acceptance; charge every stage | Closest representation-diversity baseline |
| B07 | SSV paper-based instantiation/repair adapter | Positive and negative source-generated cases, solver checks; document deviations from source recipe | Strong independent semantic-checking alternative |
| B08 | Formula→language→formula roundtrip, equivalence checking and bounded repair | Reverse translation must be independent; all calls count | Cycle-consistency control |
| B09 | MoT-style inference portfolio: natural language, executable/formal reasoning and truth-table reasoning with answer vote | Same frozen model; no claim to reproduce trained MoT; compare its best single mode | Multi-format voting control |
| A01 | F/F through the exact shared selector in §4.6 | Identical judge, fallback and implementation improvements as F/T | Principal attribution arm |
| A02 | T/T through the same selector | Same information and budget | Controls stronger single representation |
| P | F/T through the shared selector | Independently generated source interpretations | Proposed treatment |

All baseline definitions, including failed ports and screened configurations, remain in the report. Screening is for efficient allocation, not deletion of unfavorable comparisons. If B00 is perfect on controlled syntax, report that openly; the empirical LLM question concerns semantic parsing beyond that grammar and error structure within the constrained experiment.

**Native baseline adaptations.** B05 preserves CLOVER's decomposition, accumulated translation, satisfiability filtering and two-direction selection [R04].

Its project records store (id, kind, source_spans, head_id) plus ordered accumulated sentences. Units represent atomic predicates; couplers are conjunctions or merge (joining units sharing a term without conjunction); dependents are other components. Require one acyclic root, one head per nonroot and no dependent→coupler. Accumulate units, separate then joint modifiers, and couplings until the source is reconstructed. Translate accumulated sentences sequentially using earlier translations; retain the final formula. Filter syntactically invalid/theory-unsatisfiable candidates before native selection. This filter is distinct from the shared selector's policy.

Start with a seeded random winner. For each challenger, inspect both directional witnesses against source. Replace the winner if any current-only witness is judged forbidden or challenger-only witness allowed. Commit only after both directions; uncertain/unavailable judgments leave it unchanged. Do not add the shared selector's early break.

The transport schema and uncertainty/timeout rules are project choices. Keep alternative decompositions separately charged. Validate accumulation, filtering, semantic voting and two-direction selection against source fixtures before results count. A Mac transport port must not inherit CUDA/Triton assumptions from the source environment.

For B07, use the common sentence segmentation. Generate one source-conditioned positive and one negative concrete instance per constraint; the candidate constraint must permit the positive instance and reject the negative. Use two repair attempts and the reference temperature sequence 0, 0.3, 0.4, 0.5. The reference repair branch runs only when no executable answer has been found. If verification fails throughout, retain an available executable answer or use the common direct-answer fallback; charge every call. Keep any broader repair policy as a separately named matched variant. The linked code repository was empty, so this is a paper-based adapter [R05].

For B06, first audit the linked artifact without downloading model weights. Its landing page advertised a 1.89 GB archive and CC BY 4.0; individual code/dependency terms still need inspection. The reference candidate allocation is ten seeds and ten outputs per synthesized language. Run this on development at native cost, then tune its allocation within the same search resources as competing methods. Do not claim equal compute from equal response counts [R08].

Synthesize B06 languages for **whole-task premise ASTs**, not separately for every atomic source fact. The query remains the same public literal in the core. For the answer-producing local adapter, desugar every valid language-conditioned draw and group by exact premise semantics. A class is cross-language accepted only if at least one draw from each language belongs to it. Define its score as min(c1/N1,c2/N2), with Nr counting all attempted draws in language r, including invalid ones. Rank accepted classes by this score, then combined support, then a frozen seeded tie order; solve the query using the selected representative. The selective mode requires score≥τ, with τ and any support requirement chosen on development and frozen. If no class qualifies, abstain in selective mode; in full-coverage mode use semantic plurality among valid already-generated seeds, then the same common remaining-budget fallback. Record this answer adapter separately from the source's specification-acceptance evaluation. Different accepted classes with different query answers are not silently merged.

Calibrate the full native B06 allocation first on representative short, median and longest development inputs, extending only until its cost/synthesis behavior is understood. Do not interpret the 108-task pilot as a command to run every expensive native setting on all 108 tasks before this calibration.

A restricted B06 port must expose these internal stages: parse seed trees; cut subtrees into constructor definitions with holes; synthesize two vocabularies using all integrated seeds; verify seed representability and the required shape separation; independently generate in each vocabulary; desugar exactly; compare denotations; accept or abstain using the frozen support threshold. A bounded exhaustive checker must enumerate **all** possible sugarings of each small seed, including constructor recombinations. For each node, match each constructor pattern, recursively combine possible child shapes at holes, and union the resulting ordered shapes. Each language's set must be nonempty and their intersection empty. Checking only the originally selected cuts is insufficient. One-node seeds, recursion/cyclic grammar attempts, variable capture and synthesis timeout are mandatory failure fixtures.

If the artifact or a faithful restricted port fails its gate, B06 is marked unavailable, not replaced with fixed F/T under its name. Continue the informative local study, but withdraw superiority over the closest method. No published cloud-model number can substitute for this missing comparison. This is a bounded access/implementation contingency, not deferred research into whether the prior art exists.

### 5.3 Fair tuning and resource comparison

Separate two comparisons:

1. **Mechanism-controlled:** same model, source, examples, candidate counts and selector; record actual costs. F/F versus F/T isolates representation family, but does not establish efficiency if lengths differ.
2. **End-to-end:** same model and per-task active elapsed-time allowance under the anytime snapshot contract in §4.7, including generation, syntax repair, solver calls, source judgments and controller work. Each method chooses its development-tuned allocation. Report actual delivery/overshoot and the strict-delivery sensitivity separately. Also plot the natural-cost frontier; do not force a good one-call method to spend its entire allowance unnecessarily.

All methods get the same public information and access to generally useful primitives. A JSON-constrained decoder, if supported and validated for both schemas, must be offered to both formats and compatible baselines; it must enforce syntax only, never semantic gold. Shared prefix caching, batching, quantization and parser repairs must be available symmetrically. Prompt examples must contain the same underlying training examples and information, although their encoding differs. Record example token lengths rather than pretending equal example counts imply equal context.

Run a common development search over mode, temperature, few-shot count, sample/audit allocation and repair policy. Use documented model defaults as the initial configuration; add alternatives only to resolve observed truncation, diversity or formatting problems. Allocate comparable tuning **time or evaluation calls**, retain all trials, and use nested development splits when choosing many configurations. Do not tune treatment repeatedly while accepting the first baseline prompt. A baseline with unexplained weak behavior blocks the competitiveness claim.

Choose the primary budget B* before protected evaluation. Measure each viable baseline's and treatment's minimum complete algorithm cost, and evaluate development cost–accuracy curves around those costs. Select B* in the common nontruncated operating region that includes the strongest close algorithm. If B06 only operates at a larger budget, include that native-cost comparison and either raise B* or explicitly limit the primary claim; do not starve it and call the result a win. Freeze B* in seconds, token safety limits, solver allocation and the selected model. Other budget points are secondary unless separately preregistered.

When development curves show a newer/larger checkpoint dominates the proposed method on a smaller model at the same latency, report this practical alternative. Within-model attribution remains useful, but it is not an overall best-local-system result.

### 5.4 Core task population and generator

Create a fresh generator rather than recycle known benchmark answers. Use three public structural families: implication networks with conjunction/disjunction antecedents; equivalence/exclusion/cardinality-one rules; and mixtures with disconnected or redundant substructures. Every sentence is a classical Boolean relation over at most four public atoms. No defaults, anaphora across sentences, implicit quantification, temporal operators, deontic obligations or unstated commonsense implications are permitted in the core. Atom meanings describe independently assignable Boolean flags; logical relationships must be stated as rules.

The initial development grid uses atom counts 4, 8 and 16, premise counts n and 2n, and all three structural families. These are declared bounded experimental conditions, not an assertion that larger tasks are impossible. They span short to compositional instances while allowing independent exhaustive checks at the largest core size. Include arities one through four in every applicable stratum. Probe n=32 and arities five through eight separately to locate scaling failures; they are not silently pooled into the core population.

Generate formulas from a documented, versioned operator distribution; reject unsatisfiable premises for the primary three-label population. Choose query literals and sample to balance the three answer labels within each feasible stratum. If a stratum cannot support a requested label, detect that during generator validation and amend its weight before freeze, rather than repeatedly searching for rare convenient test cases. Record proposal counts and rejection rates. The declared population is the resulting stratified distribution, not an estimate of real policy frequency. Include an unbalanced naturally sampled distribution as a sensitivity check.

Maintain at least two independently written renderers. Use unambiguous constructions such as “if A then B,” “A only if B,” “at least one of A and B,” and “exactly one,” with explicit classical semantics. Place unfamiliar wording and logical composition patterns in held-out partitions. A renderer's inverse parser and a separate Boolean interpreter must agree. Additionally, supply hand-specified truth-table fixtures for every language template, independent of the renderer/inverse code: “A only if B” forbids only A=true,B=false; inclusive OR permits 01,10,11; exactly-one permits only 01,10. A shared mutation of renderer and inverse must still fail these fixtures. This is deterministic semantic review, not a claimed independent human annotation study. Ambiguous constructions and unverified LLM paraphrases cannot inherit gold labels.

Partition underlying structures before generating wording variants, using a fixed salted hash of a canonical operator/incidence graph with atom names erased. Freeze the salt, partition boundaries and canonicalization implementation before confirmation. Use exact isomorphism checks for suspected hash/canonicalization collisions; a nonunique graph fingerprint is not proof of distinct structure. Keep every wording variant, query and candidate seed of a designated structural group in the same partition. Generator parameters that leave that structure unchanged cannot move it to another partition. Save partition, rejection and collision records, and use independent generation seed namespaces.

Within confirmation, draw tasks **independently with replacement** from the frozen mixture and partition-conditional generator. Do not reject a draw because it duplicates a previous confirmation draw. Chance structural collisions remain in the sample, with distinct draw IDs and independently derived model seeds, and are reported. Rejection is allowed only for frozen task-validity, requested-cell and partition rules, never in response to the history of confirmation draws. Separate replication uses its own previously fixed partition/seed policy. If the intended finite population is too small for this experiment, choose and preregister a finite-population or cluster design instead; do not retain the IID binomial coverage claim after changing the sampling scheme.

A challenge suite outside the primary population includes inconsistent premises, tautologies, contradictions, irrelevant atoms, redundant clauses, missing conditions, ambiguous source language and unsupported modalities. Its role is failure diagnosis; it cannot be filtered using treatment results or silently change the confirmatory denominator.

### 5.5 Transfer dataset and licensing contract

Use **FOLIO** for the first human-authored transfer attempt. The original Yale-LILY repository identifies v0.0 and CC-BY-SA-4.0, and links a corrected dataset. The linked Yale-NLP Hugging Face dataset is MIT-tagged but gated; its data bytes were not inspected here. These are different provenance/access conditions [R34].

Prefer the corrected version only if it is already accessible under authorized conditions. Otherwise use the publicly downloadable v0.0, explicitly name it, count its actual files and preserve its license. Do not fabricate counts from a differently versioned paper. Store repository commit, file SHA-256, exact split and every exclusion. Development uses the training partition; final transfer uses the designated held-out partition, frozen by story ID. Multiple conclusions from one story are dependent; bootstrap by story, not row.

Inference inputs are only premise text and conclusion text. Gold FOL, labels and explanations remain scorer-only. Do not create a public vocabulary from gold. Freeze the following common extraction policy: a source-only preliminary call produces a dictionary of predicate IDs/arities, entity IDs and exact supporting text spans. For T1 it also enumerates ground atoms as unique tuples (predicate_id, ordered_argument_entity_ids), each with an opaque atom_id and text meaning. Thus loves(Alice,Bob) and loves(Bob,Alice) remain distinct, while repeated mentions of an extracted tuple share one atom ID; explicit negation is a literal flag, not a new atom. Validate arity, ID references and tuple uniqueness. All symbolic methods receive the same dictionary and are charged its cost. Do not repair it from annotations. Independently generated clause/query representations must use these IDs. Record incomplete or wrong dictionaries as a common bottleneck. Independent extraction/alignment is a later matched ablation, not a hidden advantage.

**Transfer T1: direct but restricted human-text test of the original F/T mechanism.** Before any held-out model results, identify the dataset fragment whose gold syntax and source review support only ground predicate atoms, classical Boolean connectives, at most four distinct atoms per premise and a literal conclusion. Exclude quantifiers, equality, function terms and unsupported semantics by this fixed criterion, retaining a full eligibility ledger and counts. Gold annotations may determine evaluator-only eligibility, never inference inputs. This is a support-based subset, not an outcome-based subset. The inference adapter must translate the natural-language conclusion to a registry literal; an invalid or ambiguous mapping is a method outcome. F and T still independently produce complete per-premise Boolean relations. This tests the original mechanism with the added practical extraction burden. If the eligible sample is too small, H5 is inconclusive; do not broaden its semantics silently or count paraphrases as new stories.

**Transfer T2: separately named quantified variant, not unchanged-mechanism transfer.** Full FOLIO requires explicit first-order binders and predicate arguments. Use an ordinary formula AST F and an alternative G that expresses Boolean connectives as complete truth-table gates over subformulas. G is an intensional expression tree with a different gate vocabulary; it is **not** the core's complete extensional factor representation. Success of F/G supports only this related intervention. State this change in every transfer conclusion.

For T2, use one nonempty entity sort and ordinary predicates for categories by default. “Unicorn” as a unary predicate may have an empty extension; do not invent a nonempty unicorn sort. Do not assume different constants denote different objects, introduce existential import, or close the domain to named entities. Additional sorts/functions are permitted only under an explicitly justified source semantics and a validated adapter; otherwise record unsupported. F/G schemas must distinguish constant IDs from bound-variable IDs and reject unbound variables: every premise and conclusion must be a closed formula, with binder scopes checked before compilation. Each family independently translates **both premises and conclusion** using the shared source dictionary. Select query candidates separately using their source conclusion text, then classify the selected premise/conclusion pair with the same logical rules.

The first-order witness path is a separate blocking gate. A usable witness must be serialized as a finite interpretation with an explicit domain, constant assignments and total predicate extensions/defaults. Enumerate predicates over the supplied model domain, resolve equality as identity, and re-evaluate both candidate formulas in an independent interpreter to verify the advertised separation. If Z3 gives UNKNOWN, an infinite/unrenderable model, an unhandled function or an interpretation too large for the frozen representation budget, the comparison stays unresolved. Never fabricate a named-entity approximation. The source judge sees the full text for referent context, the designated target sentence/conclusion and the certified interpretation; it is asked only whether that target holds. Other premises are context, not additional hidden constraints on the witness. Charge all serialization and judging costs and measure judge failures separately.

Quantified reasoning may time out or return UNKNOWN; count that as method failure/abstention, never the dataset's semantic Unknown label. Fixtures must cover empty predicate extensions, existential import, quantifier negation, variable shadowing, co-referring constants, equality, inconsistent premises, conclusion mistranslation and unrenderable witnesses. Full semantic fidelity to human text is not automatically certified by annotations.

Each transfer study is exploratory unless its eligibility, model/configuration and analysis were frozen separately. Contamination of public benchmarks cannot be excluded. A gain is benchmark evidence, not proof on unseen professional tasks. If T1 is too small or T2 support fails, report the precise block. Neither finite grounding nor an outcome-selected subset is an acceptable substitute.

LogiQA2.0 and CLUTRR are possible later shifts, not core dependencies: their noncommercial/share-alike terms and different inference assumptions require separate contracts. ProofWriter archive/license contents were not verified in this research and are not essential to the executable path [R35].

### 5.6 Primary and secondary metrics

For task i and method m, let Z_im=1 if the final returned answer is correct and was committed within the fixed operating contract, else 0. Terminal abstention, invalidity, solver uncertainty, OOM or expiration **without a usable timely snapshot** are zero. A failed subordinate or optional call does not erase an earlier valid snapshot: return that snapshot with final status ok, retain the call failure events, and score its actual answer. For the frozen mixture weights w_h=1/H over H feasible cells, the population accuracy and primary estimator are:

\[
A_m=\sum_h w_h\mathbb E[Z_{im}\mid h],\qquad
\widehat A_m=\frac1N\sum_i Z_{im},\qquad
\widehat\Delta=\widehat A_P-\widehat A_B.
\]

Report Δ in percentage points, the baseline and treatment accuracies, and relative error reduction (1−A_P)/(1−A_B) subtracted from one when baseline error is nonzero. Never call a relative change an absolute percentage-point gain.

Secondary metrics, all with task-clustered uncertainty where applicable:

- Full-denotation translation accuracy; per-clause fidelity; validity, truncation and missing-sentence rates.
- Joint semantic error Pr(F wrong and T wrong); at-least-one-correct rate; and wrong-equivalence coincidence Pr(F≡T | both wrong). Report denominator counts and unconditional wrong agreement as well. Improved conditional coincidence alone is insufficient.
- Selector recovery and harm: correct candidate available but lost; all candidates wrong but answer lucky; correct initial answer changed to wrong.
- Coverage, selective accuracy, unconditional false-acceptance rate, and risk–coverage curves at development-calibrated thresholds. No post hoc threshold chosen on test labels.
- Mean/median/p95 whole-method latency in seconds, prompt/decode token counts, solver/controller time, peak memory in GiB, failure recovery cost and cold-start time.
- Correct answers per hour and mean seconds per correct answer, with the caveat that these depend on the workload and coverage policy. Include development/adaptation amortization separately.
- Robust all-view correctness and canonicalized output agreement on the dedicated robustness set. Agreement without correctness is never success.

Error dependence is estimated across **task-level summaries** of candidate pairs. Thousands of pairs from a few source tasks do not create thousands of independent observations. Conditional error analysis is descriptive unless its hypotheses and selection were preregistered.

### 5.7 Pilot, sample size and inferential procedure

Use a separate balanced pilot to measure failure rates, discordance, costs, task variance and covariance across views. The initial generator grid has 54 family/size/density/label cells; two independent tasks per cell provide 108 tasks for the first calibration batch. This is an exploratory coverage design, not a powered accuracy claim. Add pilot tasks only to resolve a concrete unstable estimate or missing regime. Never include them in confirmation.

Choose a **fixed-sample confirmatory design**. Before running protected data, freeze the method set, model, precision, primary budget, total N, mixture probabilities, partition/generation rules, randomization, multiplicity procedure and analysis code. Cell counts arise from IID draws; do not force fixed counts per stratum. For a single paired binary outcome let D_i=Z_iP−Z_iB, d=E[D_i], and q=Pr(D_i≠0). Then Var(D_i)=q−d². A useful planning approximation for two-sided 5% size and 80% power is:

\[
N\approx\frac{(1.96+0.842)^2(q-\delta^2)}{\delta^2}.
\]

Use simulation with pilot-derived joint outcomes, the frozen mixture and the **exact interval decision below** to finalize N. The equation is an initial normal approximation, not the final power guarantee. Multiple comparators and conservative intervals generally increase the necessary sample. Every declared primary comparator must be beaten; retain all intervals.

Calculated examples for q=0.20, one comparator and one primary observation per task:

| Detectable absolute difference | Approximate independent paired tasks | Interpretation |
|---|---:|---|
| 1 percentage point | 15,690 | Small effects can require substantial local inference |
| 2 percentage points | 3,917 | Still thousands of tasks, not dozens of seeds |
| 3 percentage points | 1,737 | Useful intermediate sensitivity illustration |
| 5 percentage points | 621 | Detects larger effects; a null result does not exclude 1–2 points |

These are sensitivity examples, **not improvement requirements**. Any reliably positive effect is a statistical finding; practical value depends on cost and application. No universal “breakthrough percentage” is specified.

The default resolution is a transparent **planning convention**, not a practically valuable effect size. For the balanced independent pilot of n_p tasks, let q_hat be observed discordance and d_p the paired mean. Set q_U=min(1, q_hat+sqrt(log(1/0.05)/(2 n_p))), a conservative Hoeffding upper bound that does not require identical per-cell probabilities. Set s_plus=sqrt(q_U/n_p) and δ_plan=max(|d_p|−s_plus,s_plus). This remains positive even when no discordance is observed. Use the largest required N across comparators, then simulate the final procedure conservatively. A superiority sensitivity simulation assumes a hypothetical **positive** effect +δ_plan with feasible joint probabilities; it must not search forever for 80% positive-superiority power under a negative simulated effect. A negative pilot may instead motivate a preregistered interval-width target around the signed estimate, using the same fixed-sample analysis. Publish the chosen objective, resolution, N and cost grid before protected generation. A near-zero or negative pilot motivates estimation/falsification, not a presumption of superiority. If a real application supplies a minimum valuable effect, plan for that instead. Never interpret this convention as excluding smaller benefits.

**Executable primary inference rule:** let M be the number of distinct frozen primary comparators. For each comparator, count wins w (P correct, B wrong) and losses l (P wrong, B correct) among N tasks. Construct Clopper–Pearson binomial intervals for p_win and p_loss, each with total error α/(2M), where α=0.05. Use beta quantiles with lower endpoint zero at count zero and upper endpoint one at count N. Form the difference interval [L_win−U_loss, U_win−L_loss]. By a union bound these M difference intervals have simultaneous coverage at least 1−α under the specified independent-mixture sampling. Declare the primary superiority result only if **every lower bound is above zero**. These intervals are conservative but defined at zero events and near ceilings. Freeze the algorithm and simulate its actual power; do not use the optimistic normal table as the completed power analysis.

The finite-sample coverage statement is conditional on independent, stationary task outcomes under the frozen mixture and operating conditions. Randomize method order within paired blocks and record session, temperature/memory pressure and deadline failures. If material session dependence or operating drift remains, report a prespecified session-cluster sensitivity analysis and qualify the exact-coverage claim; randomizing task inputs alone does not remove device-state dependence.

Report exact paired McNemar tests and paired bootstrap intervals as secondary efficiency/consistency checks, clearly distinguished from the primary decision. Report per-stratum estimates and a descriptive macro-average as secondary results. For story-level transfer and multiple-view robustness, use a separately preregistered task/story-cluster bootstrap or appropriate cluster model, resampling whole records including all methods; check sparse-event coverage and report bounds rather than degenerate zero-width bootstrap intervals when necessary. Do not pool stochastic seeds as independent tasks. The primary uses one frozen seed schedule per task/method; a multi-seed study averages within root and requires its own cluster analysis [R36].

No optional stopping for significance. Interim operational dashboards may expose throughput, parse errors and crashes but must hide comparative correctness on protected data. If a flaw is discovered, stop, record the amendment, invalidate affected runs and restart with fresh protected tasks when interpretation could have been influenced. An additional study after an inconclusive result is a newly preregistered independent replication, not an extension until a desired p-value appears.

### 5.8 Robustness, harness validation and timing

For each robustness root generate a fixed set: original wording; atom-renamed/fact-reordered wording; an independently validated equivalent connective rendering; and a query-negated variant. The first three test invariance. Query negation is **equivariance**, mapping ENTAILED↔CONTRADICTED while leaving UNDETERMINED and INCONSISTENT unchanged. Invert that map before scoring canonical agreement. Never label a changed query meaning-preserving.

Validate the harness before models: identical method outputs must yield exactly zero difference; a deliberately injected error must be detected; shuffled labels must remove apparent performance; the gold solver must answer all valid generator instances; and equivalent compiled tables/formulas must agree. Inject the same wrong implication into both formats and verify that agreement remains high while semantic correctness falls. Test all failure statuses in aggregation so rejected cases cannot disappear.

Warm the loaded model and kernels using non-evaluation inputs. Measure cold load separately. Use monotonic time from source submission through final answer, including all calls. Force lazy work and synchronize the actual generation stream before stopping GPU-stage timers. Fully consuming a streaming API and inspecting its version's synchronization behavior is required; timing graph construction alone is invalid. Randomize method order within task blocks, keep the laptop on mains power, record power mode, background pressure and swap, and balance long runs across sessions. Repeated timing trials characterize variability; they do not increase the number of independent reasoning examples.

## 6. M3 Max feasibility and hardware-adaptive configuration

### 6.1 Current support and chosen stack

Use native arm64 Python, MLX-LM for frozen local text inference, Z3 on CPU for formal reasoning, a small JSON/AST layer and NumPy/SciPy for analysis. Use a virtual environment and dependency lock. This avoids requiring remote GPUs or paid APIs. MLX documentation and model-source inspection support the relevant local text path, but do not substitute for target-device tests [R27–R33].

The inspected base MLX documentation requires Apple Silicon, native Python at least 3.10 and macOS at least 14. **The current MLX-LM package metadata is stricter: Python at least 3.11**, with Darwin MLX at least 0.32.3 and Transformers at least 5.7.0 in the inspected source. These are observed dependency declarations, not a target-tested lock. MLX-LM separately documents macOS 15 for large-model memory wiring. Use a compatible native interpreter, discover actual support, and freeze the tested resolution. Do not silently upgrade the OS or raise wired-memory limits. Older stacks require a separately validated older-model branch and narrower competitiveness claims [R28].

Use current model cards to screen Qwen3.5-4B, Qwen3.5-9B and Qwen3.8-27B, with Gemma 4 E4B as an independent-family replication candidate. These official assets and license metadata were inspected as of the cutoff. Qwen3-8B/14B remain simpler dense diagnostic fallbacks, not sufficient alone for a contemporary strongest-local comparison. Choose the best validated development performance at the measured operating budget; the newest or largest name is not automatically the winner.

The qwen3_5 MLX implementation includes hybrid attention/recurrent-state operations and a text-only conversion path that omits vision/MTP components. This plan tests ordinary text generation only. Use documented chat-template mode parameters; do not assume informal '/think' strings control newer checkpoints. Pin actual resolved revisions rather than inventing release numbers.

### 6.2 Hardware discovery: existing commands

Run these separately and save outputs, redacting serial numbers from shareable reports:

~~~bash
uname -m
sw_vers
sysctl -n hw.memsize
sysctl -n machdep.cpu.brand_string
system_profiler SPHardwareDataType SPDisplaysDataType
df -h .
vm_stat
sysctl vm.swapusage
pmset -g batt
python3 -c 'import platform; print(platform.machine(), platform.python_version())'
~~~

Record physical memory, available disk, CPU/GPU configuration, macOS build, Python architecture, baseline memory pressure and power state. M3 Max configurations vary; no maximum-memory assumption is allowed [R27]. A Rosetta/x86 Python process fails the native-backend gate even if the machine is Apple Silicon.

After choosing a compatible interpreter, create a project virtual environment and install the minimal requirements. Check the actual upstream help before using flags:

~~~bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install mlx-lm z3-solver numpy scipy jsonschema pyyaml pytest
.venv/bin/mlx_lm.generate --help
.venv/bin/mlx_lm.convert --help
.venv/bin/python -m pip freeze
~~~

These are a discovery installation, not the final reproducibility lock. Preserve resolved versions, wheel hashes and repository commits after the smoke gate. If a named console entry point differs in the installed release, inspect its registered entry points and record the actual command; do not invent flags. Existing Python entry points inspected include MLX-LM load/generate/stream_generate. The project wrapper must adapt to the installed sampler/token-response signatures.

### 6.3 Readiness tests before large downloads or dependent experiments

First read small model cards/configs/manifests and licenses. Estimate disk requirements before downloading. Begin with the smallest viable current checkpoint and a short source-to-JSON smoke test. Then:

1. Verify exact prompt/chat-template bytes, EOS handling, reasoning mode, token counts and answer extraction.
2. Compare cached versus uncached teacher-forced outputs, and whole versus chunked prefill, on short fixed fixtures. Compare finite logits, maximum numerical deviations, top-token agreement and actual parsed outputs. Establish tolerances from same-precision references and repeated runs; do not choose a tolerance just to pass.
3. Test interrupted/recreated cache behavior for the hybrid architecture. If serialization cannot be proven equivalent, resume only at complete call boundaries by regenerating the prompt.
4. Inspect resolved architecture fields and sensitive tensor dtypes; ensure the backend loads the intended text weights and layer types. Do not assume a successful import proves faithful inference.
5. Compare four-bit and higher-precision inference on a paired development calibration subset when it fits. Report output and validity differences. All within-model method comparisons use identical weights and quantization; a precision-dependent result is scoped accordingly.
6. Stress the longest planned prompt/output length, one model at a time, with repeated peak-memory/pressure measurements. No recurrent swap growth, unexplained nonfinite values or cache divergence may be hidden by shrinking only treatment inputs.

If a critical operator is unsupported, use a validated equivalent backend or smaller supported model. A replacement kernel or CPU fallback requires numerical and cost revalidation. Do not silently change the proposed computation or report CPU speed as Metal speed.

### 6.4 Memory model

For one process, estimate:

\[
M_{peak}=W_{quant}+W_{unquant}+KV+S_{recurrent}+A_{prefill}
+B_{logits}+B_{allocator}+B_{host/solver}.
\]

Quantized weights require more than P·b/8 bytes: add group scales/biases, mixed-precision tensors, padding and metadata. Calculate actual stored bytes from tensor headers. Conversion may temporarily hold both originals and outputs. Unified-memory GPU allocations and process RSS overlap; report them separately rather than adding them into a false physical total.

For full-attention GQA layers at batch B, sequence T, H_KV heads, width D and d bytes/value:

\[
KV=2B T L_{full}H_{KV}D d.
\]

Calculated examples at batch one, 8,192 tokens and two bytes/value from inspected model configurations:

| Model | Full-attention KV only | Additional qualification |
|---|---:|---|
| Qwen3-8B | 1.125 GiB | 36 layers, 8 KV heads, width 128 |
| Qwen3-14B | 1.250 GiB | Dense architecture; config-specific dimensions |
| Qwen3.5-9B | 0.250 GiB | Add hybrid recurrent and convolution states |
| Qwen3.8-27B | 0.500 GiB | 16 full-attention layers; add 48 recurrent-layer states |

Shape-based fp32 recurrent-state estimates are approximately 48 MiB for the 9B and 144 MiB for the 27B configuration, before convolution states and workspaces. Validate actual implementation shapes. Cache truncation/rotation can change answer quality and is not an invisible optimization.

Planning envelopes for batch-one, moderate-context four-bit inference are roughly 3–7 GiB for 4B, 6–11 GiB for 9B and 18–25 GiB for 27B. These are **estimates, not measured requirements**. The inspected Qwen3.8-27B-oQ4 conversion lists a 16.6 GB asset; that is neither peak memory nor a guarantee of numerical equivalence [R30]. Use actual measured peaks plus a reserve covering observed OS variation, allocator spread and solver memory. Do not infer fit solely from nominal model parameters.

On the smallest M3 Max memory configuration, screen 4B/9B first and admit 27B only when its measured envelope fits comfortably without pressure. On larger configurations, still choose according to accuracy/cost, not capacity alone. Load independent model families sequentially so their weights do not coexist unnecessarily.

### 6.5 Storage, runtime and cost calibration

Disk requirement equals original assets plus converted assets plus temporary/download-cache duplication plus logs/checkpoints and a free-space reserve. A 27B two-byte parameter lower bound alone is about 54 GB decimal; multimodal/auxiliary weights and conversion copies add more. A preconverted checkpoint saves conversion space but requires provenance checks. Do not download all candidates in advance.

Measure prefill rate r_p as a function of input length and decode rate r_d as a function of cache length. Estimate complete experiments using:

\[
T_{run}\approx T_{load}+T_{data}+\sum_{calls}(n_{in}/r_p+n_{out}/r_d+t_{solver}+t_{controller}).
\]

Use empirical distributions, not one optimistic tokens/second figure. Include invalid outputs, judges, repairs and grammar synthesis. As an arithmetic illustration only, two million output tokens at 40 tokens/second require 13.9 hours before prefill and other work. For 1,737 tasks, three arms and 8,000 output tokens per arm/task, decoding alone would be about 289.5 hours at that assumed rate. These are not M3 measurements. They show why pilot screening and focused confirmation matter.

Produce a local duration/precision table before long runs. No arbitrary wall-time ceiling or promised overnight success is imposed. If the desired sensitivity requires an impractical amount of time, the scientifically valid result is a narrower measured claim or an explicitly inconclusive study. Shorten the method only through a recorded development change; never truncate one baseline selectively to fit a schedule.

### 6.6 Checkpointing and recovery

Save each completed call and task atomically, with immutable hashes of source, model revision, quantization, prompt, decoder settings, seed, method version and budget. A run manifest maps planned calls to statuses pending/running/complete/failed. Never overwrite failed outputs with successful retries without preserving the original event.

There are two explicit recovery paths, chosen by recorded execution state while outcomes remain blinded:

1. **Completed-call resume:** if no call was partially executed, restore the same candidate order, seeds and completed artifacts. Subtract **all previously charged active time** from each method's allowance. Continue with only its remaining budget; no candidate regeneration or budget refresh is permitted. Downtime is excluded from active cost. This preserves an algorithmic-budget observation but is flagged as discontinuous, so it is not an uninterrupted live-latency observation.
2. **Whole-block restart:** a partially executed call may be discarded only for an independently logged infrastructure interruption. Invalidate the entire paired task block for the primary comparison, including already completed arms. Restart every arm from empty task caches, with the original task, method order and seeds, under fresh full allowances. Do not carry any interrupted-attempt candidates or judgments into the new attempt. A complete replacement block supplies the primary observations; all consumed work from every attempt remains in the total-work ledger. This is a replacement experiment, not a resumed task with free extra reasoning. Use this same whole-block rule when an uninterrupted timing replacement is needed after path 1; retain only the replacement block in the primary paired analysis and label the superseded algorithmic record.

Method-caused OOM/timeouts are outcomes, not infrastructure retries. Repeated infrastructure failure pauses the run for diagnosis; correctness must never determine retries or which attempt is retained. Any model/prompt/context/precision change creates a new run and requires the affected comparison to be refrozen. Never restore an unvalidated partial model cache.

Maintain three distinct ledgers: live uninterrupted task latency; algorithmic active cost charged to a resumed task; and total compute actually consumed including failed infrastructure attempts. Primary live latency uses empty task-specific caches and comparable uninterrupted blocks. Offline replay of saved pools can diagnose selectors but cannot be reported as live speed. Cross-method development caches may save research time only if their original costs are charged and the result is labeled replay. Native cold-start and model loading are reported separately.

Fallback sequence: one concurrent inference stream; smaller prefill chunks without discarding context; release unused caches; validated smaller current model; independently validated llama.cpp/Metal backend. Every substantive fallback triggers recalibration and limits the resulting claim. Nothing in the core requires training or specialized physical equipment.

## 7. Ordered work packages and decision gates

### 7.1 Dependency map

| Package | Depends on | Risk resolved before more expensive work |
|---|---|---|
| WP00 Scope, licenses and hardware | This plan | Assets, permitted use, actual machine and claim boundary |
| WP01 Exact semantics and harness | WP00 | Incorrect labels, compiler bugs, leakage and deceptive metrics |
| WP02 Local backend/model calibration | WP00, WP01 fixtures | Unsupported architecture, precision/cache failure, capacity |
| WP03 Baseline adapters | WP01–02 | Weak or unfaithful comparisons; missing closest method |
| WP04 Fixed representation portfolio | WP01–02; parallel with WP03 | Missing indispensable mechanism or false independence |
| WP05 Pilot and configuration selection | WP03–04 | Competence, headroom, cost, variance and baseline tuning |
| WP06 Preregistration and freeze | WP05 | Adaptive selection, insufficient sensitivity, undefined budget |
| WP07 Protected comparison | WP06 | Main causal comparison under locked conditions |
| WP08 Attribution, robustness and replication | WP05; frozen designs before each new test | Alternative explanations and instability |
| WP09 Human-text transfer | WP01–05 plus T1 eligibility/extraction; T2 additionally needs quantified-logic/interpretation gates | Synthetic-only explanation and practical relevance |
| WP10 Synthesis and handoff | All attempted packages | Honest interpretation, reproducibility and continuation decision |

WP01 can proceed while small asset metadata are reviewed. No large model download is needed to validate the exact semantics. WP09's data/provenance gate can run early, but no unplanned transfer result should tune the primary confirmatory method.

### WP00 — Establish the executable contract

**Objective/risk:** prevent a project whose data, model or machine assumptions are false. **Entry:** this document and the actual target laptop; no research results assumed.

**Actions/files:** create the repository layout in §9; run hardware discovery; write hardware.json, assets.yaml, decisions.md and a baseline register. Inspect each selected asset's model/data/code terms separately. Record the exact CPL artifact availability and source-license status before promising its reuse. Choose the core task interpretation and explicitly document the conditional scope of public vocabulary/segmentation. Keep this plan under version control.

**Tests/outputs:** readable machine inventory; disk/memory estimates in bytes/GiB; source URLs and hashes or a status of not-yet-downloaded; explicit license/access field per asset. Confirm no cloud/API dependency is required by the core.

**Resources:** metadata downloads and local inspection only; use manifests to forecast substantial downloads.

**Acceptance/branches:** proceed when the core has an authorized local path. If a corrected dataset is gated, select the declared public legacy path or record transfer blocked. If a baseline lacks reusable code terms, independently implement its published algorithm and record provenance; do not silently copy it. If actual hardware is not M3 Max, record the mismatch and limit any speed claim. Missing optional assets do not block exact core development.

### WP01 — Build exact semantics and an adversarial evaluation harness

**Objective/risk:** test mathematical consistency, label handling and information boundaries before LLM cost. **Dependencies:** WP00.

**Actions/files:** implement schema.py, boolean_ast.py, factor_table.py, oracle/enumerate.py, oracle/z3_check.py, data/generate.py, data/render.py, data/split.py and tests/test_semantics.py. Use independent recursive evaluation and solver compilation. Keep the scorer executable separate from inference. Implement failure-inclusive aggregation and identical-method checks. Publish the exact source grammar and deterministic parser B00.

**Tests/outputs:** enumerate every Boolean function for arity zero through four and verify table compilation; test random composed formulas on all worlds at small n; test De Morgan, implication orientation, scope permutations, zero-scope tables, constants, inconsistent premises, missing rows, missing clauses and query-label mappings. Verify transformations and inverse mappings. Emit harness_validation.json with counts, failures and software versions.

**Resources:** CPU only; memory for a truth vector is 2^n bits when packed, plus interpreter/solver work. Profile before increasing n. Exhaustive tests for the local factor compiler are finite and cheap compared with model inference.

**Acceptance/branches:** all exact fixtures agree and injected defects are detected. Any mismatch blocks model evaluation; fix and rerun only affected invariants plus a regression set. If full-world enumeration becomes expensive, retain it for independent small cases and use Z3 on larger cases; do not call unvalidated solver output an oracle. A meaningless or trivially parsed population changes claim scope, not the harness's arithmetic.

### WP02 — Validate inference and select feasible checkpoints

**Objective/risk:** establish actual Apple Silicon support and costs. **Dependencies:** WP00 and WP01's public fixtures.

**Actions/files:** implement backends/mlx_backend.py and calibration.py. Install a candidate lock, then test the smallest current model; admit larger/current and independent-family candidates only after manifest checks. Run the readiness suite in §6.3. Record model SHA, tokenizer/template hash, quantization, memory, prefill/decode distributions and exact runtime version. Provide a separately validated alternate backend only if needed.

**Tests/outputs:** backend_validation.json, hardware.json additions, calibration.csv and candidate_models.csv. Confirm valid JSON generation, finite outputs, cache parity, context completion, reproducible seed behavior to the extent the backend supports it, and no unexplained pressure/swapping.

**Resources:** one loaded model at a time; sequential generation; actual measured reserve governs fit. Quantization comparison uses the largest higher-precision checkpoint that fits; lack of a full-precision comparison remains a limitation.

**Acceptance/branches:** proceed with a checkpoint/backend pair that passes correctness and resource gates. A support failure is not evidence against representation diversity. If only an older/smaller model works, continue the bounded mechanism study and restrict contemporary competitiveness claims. If no local inference path is valid, deliver the exact harness and a blocked feasibility report rather than invented model results.

### WP03 — Reproduce the strongest relevant local baselines

**Objective/risk:** prevent false gains against weak implementations. **Dependencies:** WP01–02.

**Actions/files:** implement methods/direct.py, sample_vote.py, formula_solver.py, clover_port.py, cpl_port.py, ssv_port.py, roundtrip.py and mot_vote.py. Start with B01–B05; perform B06 artifact/source inspection and port before broad runs. Add B07–B09 to development screening. Record algorithm deviations in baseline_register.csv. Reuse prompts only with appropriate attribution and access terms.

**Tests/outputs:** source-example traces where accessible, synthetic fixtures with known candidate pools, equivalence groups, witness adjudication and tie outcomes. For CPL, add the all-sugarings checker and synthesis-failure cases. Compare source reference behavior on small deterministic inputs; matching old published accuracy on a different model is not required or asserted. Produce baseline_reproduction.md and raw development trials.

**Resources:** begin with no model or recorded mock pools for selector tests; then use short development runs to characterize calls/cost. Expensive native CPL allocation runs only after synthesis tests succeed.

**Acceptance/branches:** each admitted baseline has a documented faithful algorithm or explicitly named adaptation, correct accounting and sensible development behavior. Investigate an anomalously weak baseline before comparison. If the closest baseline cannot be validated, the project may proceed only with an explicit narrower claim. If a simple baseline already dominates the practical frontier, test whether representation diversity has any remaining headroom before implementing a more complex controller.

### WP04 — Implement the minimum independent representation mechanism

**Objective/risk:** ensure the system actually tests H1/H2 rather than reformatting one answer. **Dependencies:** WP01–02; may proceed alongside WP03, whose baseline gates still block comparative claims.

**Actions/files:** implement methods/fixed_portfolio.py, selection/semantic_groups.py, selection/witness_tournament.py and prompts/formula.txt, table.txt, witness.txt. Generate candidate sets from independently seeded fresh contexts. Enforce per-sentence completeness and exact truth-table ordering. Implement common fallback and deadline behavior.

**Tests/outputs:** traces containing both successful and disagreeing candidates; independent prompt hashes; complete cost ledger. Inject an incorrect shared formula/table and ensure the solver accepts their equivalence while the hidden scorer marks mistranslation. Demonstrate a complementary candidate pool where the selector can recover the right clause, and a deliberately wrong judge where it harms selection.

**Resources:** at least two completed translations and optional audit calls; cost measured against same-format conditions.

**Acceptance/branches:** the pipeline executes fully with no oracle information. If table competence is poor, first distinguish schema errors, semantic errors and output truncation. A shared parser improvement is applied to competitors too. If independently generated candidates are never complementary within uncertainty, keep the method simple and move to headroom analysis; do not add an unmotivated third representation.

### WP05 — Pilot, diagnose and choose configuration

**Objective/risk:** calibrate headroom, cost and statistical sensitivity. **Dependencies:** WP03–04.

**Actions/files:** run the balanced development/pilot design, with immutable split assignments. Tune all methods comparably; screen current checkpoint size and reasoning mode. Estimate marginal fidelity, joint error, oracle headroom, judge reliability, common-format versus mixed-format outcomes and full costs. Include two differently worded formula prompts as a same-format heterogeneity control. Produce pilot_report.md, tuning_trials.csv and resource_forecast.json.

**Tests/outputs:** forecasts reproduce observed elapsed time within documented error; candidate counts and costs reconcile to call logs; no result uses the protected partition. Record which baseline won and uncertainty among contenders. Retain table-only winners if they beat the mixture.

**Resources:** the first 108 task batch covers the declared grid; expand only for a specific estimation/diagnosis need. The purpose is not a significance claim. Larger model screening is justified by observed capacity and expected information, not by artificially keeping a weak model.

**Acceptance/branches:** if there is usable complementarity and an exploitable budget, freeze a confirmatory comparison. If there is complementarity but poor selection, run a bounded judge diagnostic before additional candidate sampling. If all methods are at ceiling, preserve that result and use a predefined higher-complexity diagnostic; do not select tasks on P's wins. If all methods fail, assess task/model competence before rejecting H1. If no advantage remains after tuning, prepare a negative or estimation-focused confirmation, not a favorable benchmark slice.

### WP06 — Freeze hypotheses, sample size and protected evaluation

**Objective/risk:** prevent adaptive analysis and ambiguous success. **Dependencies:** WP05.

**Actions/files:** write preregistration.yaml with H2, exact model/config, primary comparator set, budget, N and stratum weights, seeds, tests, correction, exclusions, failure policy and precision forecasts. Freeze prompts and dependency/model/data revisions. Generate protected source inputs and scorer-only files in separate directories/process access. Hash the manifest and commit it before inference. Save statistical simulation scripts and their results.

**Tests/outputs:** freeze command rejects placeholders, missing hashes, overlapping groups, unfrozen prompts, unequal information fields, unnamed fallback and impossible budgets. Verify no protected data were used in tuning. Run simulated null and injected-effect coverage/power checks using the final estimator.

**Resources:** CPU/data generation and analysis, not LLM runs. Estimate complete confirmatory runtime from WP05; include planned robustness and replication costs separately.

**Acceptance/branches:** start only when the comparison is valid and its achievable sensitivity is recorded. If required N exceeds feasible resources, state the unresolved effect range and select an estimation study through a pre-result amendment. Do not redefine a small positive threshold as “meaningful” to fit a short run. A changed scientific question creates a new preregistration.

### WP07 — Execute the protected paired comparison

**Objective/risk:** test H2 under locked conditions. **Dependencies:** WP06.

**Actions/files:** run the planned methods on the same root tasks, interleaving order and preserving call-boundary checkpoints. Collect outputs without exposing comparative correctness in progress dashboards. At the planned N, score all outcomes once using the frozen analysis. Generate primary_results.csv, paired_outcomes.jsonl, intervals.json and resource_summary.csv.

**Tests/outputs:** completed/failed task counts equal the planned manifest; costs include discarded attempts; every outcome has a status; no duplicate completed key; no model/config drift. Quantify any budget overshoot and repeat conclusions using the preregistered strict-deadline rule.

**Resources:** forecasted local duration and disk, updated only by observed throughput without using correctness. Interruptions are resumable. Warm/cold costs are reported separately.

**Acceptance/branches:** a valid completed run produces a positive, negative or inconclusive result. Implementation corruption invalidates affected evidence and triggers a logged repair/fresh confirmation. Infrastructure interruption is resumed with the same contract. A valid negative is retained. Statistical significance without acceptable cost/coverage does not pass the practical-value gate.

### WP08 — Attribute, stress-test and independently confirm

**Objective/risk:** eliminate alternative explanations and test reproducibility. **Dependencies:** WP05 and a frozen design for each new study; use WP07 to select clearly labeled exploratory follow-ups only.

**Actions/files:** execute the ablation register in §8; measure source-judge accuracy on hidden-labeled witness cases; compare best single format and same-format prompt diversity; test exact transformations, higher arity and size. Replicate the primary contrast with new generator seeds and, if support passes, an independent model family. Resume an intentionally interrupted small run and compare to uninterrupted execution.

**Tests/outputs:** ablations.csv, robustness.csv, failure_taxonomy.jsonl and replication.md. Preserve task clusters and confidence intervals; record whether any generic optimization erases the gain. For reproducibility, rerun sampled saved cases from a clean environment and compare outputs/metrics or the declared stochastic distribution.

**Resources:** prioritize ablations capable of overturning attribution. Use saved candidate pools for selector diagnostics, but charge full production costs when making end-to-end claims. Replication N follows its own uncertainty target and is not automatically a full repeat of every baseline.

**Acceptance/branches:** scale only if the mechanism survives the decisive controls. If gains are model-specific or erased by tuned single-format methods, narrow or reject the representation-diversity explanation. If reproducibility fails, resolve environment/RNG/cache causes before reporting confirmation.

### WP09 — Attempt human-authored transfer

**Objective/risk:** test whether synthetic gains have broader utility. **Dependencies:** dataset/license gates plus WP01–05. T1 requires its eligibility/extraction gate; T2 additionally requires quantified semantics and interpretation-serialization gates.

**Actions/files:** implement data/folio.py and predicate_registry.py for the preregistered T1 fragment. Add fol_ast.py, fol_interpretation.py and tests/test_fol_semantics.py only for the separately named T2 variant. Preserve version/story grouping and the full eligibility ledger. Build configurations from training text only. Evaluate every designated held-out story in the declared population, including unsupported method cases in its denominator. Report T1 eligibility coverage over the full dataset separately. Compare direct/native reasoning, strongest symbolic baseline and the appropriate F/T or F/G method; include deterministic parser coverage/failures.

**Tests/outputs:** transfer_manifest.json, quantified_fixtures.json, transfer_results.csv and an error review separating source ambiguity, annotation issues, unsupported logic, mistranslation, solver uncertainty and selection. Review errors blind to method identity when feasible. Report official-label and separately adjudicated sensitivity scores without rewriting the main labels.

**Resources:** small data relative to model assets; quantified solver timeouts are calibrated on development and charged. No external judge/model API is required.

**Acceptance/branches:** transfer gain with fair coverage/cost warrants a scoped human-text benchmark claim and further domain research. Synthetic-only gain remains a restricted result. If dataset quality, licensing or solver completeness prevents a meaningful test, report the precise block. Do not use favorable subsets or finite grounding to manufacture transfer.

### WP10 — Synthesize and make the experiment independently reproducible

**Objective/risk:** prevent partial evidence being presented as a breakthrough. **Dependencies:** all attempted work packages, including failed ones.

**Actions/files:** produce report.md, decision.md, reproducibility_manifest.json, environment locks, results tables and the full failure/limitation ledger. Link every conclusion to a run and comparison. Include the strongest competing explanation and what would change the conclusion. Archive complete configs, raw model outputs and method amendments. Retain unfavorable data and screened baselines.

**Tests/outputs:** a clean analysis run rebuilds tables from immutable raw artifacts; hashes match; no protected file leaked into inference; file count/cost totals reconcile. Produce explicit continue/revise/stop/inconclusive decisions using §10. Add a practical-cost calculation only for a specified deployment volume; do not invent a user business case.

**Resources:** CPU analysis and artifact storage. Re-run model inference only to resolve a concrete reproducibility issue.

**Acceptance/branches:** the final report clearly distinguishes verified facts, performed experiments, deductions, unperformed checks and future hypotheses. A negative result can complete the project successfully. A positive result with unresolved baseline or leakage defects cannot.

## 8. Ablation, robustness and failure-analysis program

### 8.1 Required attribution matrix

| Comparison / intervention | Alternative explanation tested | Interpretation and required record |
|---|---|---|
| F/T versus best of F/F and T/T | A better single representation explains the gain | A mixture losing to the selected single format is not a diversity advantage |
| F/T versus two different formula prompts | Generic instruction heterogeneity, not semantic form | Tune both prompts fairly; record whether the simpler ensemble matches P |
| Same candidate counts, then same wall time | More samples versus cheaper computation | Neither alone establishes the other claim |
| Shared selector versus plurality, source critique and native CLOVER | Verification or better decomposition explains the gain | Hold candidate pool fixed for selector-only analysis; count full costs separately |
| Independently inferred table versus deterministic table of formula | Reformatting or shared translation explains behavior | Deterministic version is a diagnostic, not independent evidence |
| Pure F and pure T without solver; with solver | Tool access or exact inference dominates | A solver gain is distinct from multi-representation gain |
| Shared constrained-output decoding on/off | Better JSON serialization explains improvement | If the effect disappears, report a formatting/engineering result |
| Full cost versus generation-only cost | Hidden synthesis, preprocessing or judge expense | Practical claims use full cost |
| Selective acceptance at common coverage | Rejecting more difficult cases explains precision | Report unavailable operating points instead of extrapolating |
| Correct candidate pool with deliberately wrong judge | Selection can destroy useful candidates | Quantify harm; agreement is not a semantic certificate |
| Shared source omission in both formats | Common causes survive diverse syntax | Expect high agreement and low fidelity; harness must expose it |
| Current larger model / independent family | Small-model or family-specific weakness explains gain | Restrict scope if absent on a stronger feasible model |
| Higher precision and longer full context | Quantization or truncation artifacts | Treat configuration as part of the experimental condition |

The full factorial need not be run on every protected item. Use development to identify decisive contrasts and preregister focused independent samples. However, the best-single-format, equal-cost, same-format prompt-diversity and failure-inclusive comparisons are indispensable to attributing any claimed diversity benefit.

### 8.2 Boundary and adversarial cases

Include all of the following with explicit expected behavior:

- An unconstrained atom: absence of proof is not proof of negation.
- Reversed “only if,” negated antecedents, inclusive/exclusive disjunction and nested negation.
- A factor whose scope order changes; outputs must permute accordingly.
- A zero-scope tautology/contradiction and duplicate/redundant clauses.
- Contradictory premises: return INCONSISTENT, not an arbitrary query answer.
- An irrelevant sentence and a semantically decisive sentence with similar wording; importance must not be judged from fluency.
- A query whose answer is unchanged despite mistranslation; full-denotation scoring must still fail.
- Two candidates agreeing on the same wrong answer but different semantics; answer voting must not be mistaken for semantic equivalence.
- Two candidates with different syntax and identical wrong semantics; the source-fidelity score must fail.
- Solver UNKNOWN, expired deadline, malformed/truncated output and grammar-synthesis failure; all remain in denominators.
- High-arity clauses where table size becomes expensive; no covert scope reduction.
- Deliberately ambiguous or contradictory atom meanings; these are source-contract violations, not solvable by syntactic diversity.
- Source text containing imperative-looking content; treat it as task data, never as instructions to bypass schemas or access scorer files.
- Cache reuse across task IDs; no response contamination, leaked prefix or uncharged hidden prior work.

For end-to-end extraction, remove the supplied registry and segmentation from every LLM method. Charge the extraction stage, compare independent versus shared extraction, and evaluate its errors separately. If that destroys the gain, state that the original result required curated source structure. Do not treat this as a failure of exact Boolean inference.

### 8.3 Failure taxonomy and response rules

| Failure class | Evidence | Required response |
|---|---|---|
| Broken implementation or measurement | Oracle disagreement, leaked gold, missing costs, cache mismatch | Block affected evidence, repair, rerun validation and fresh confirmation as needed |
| Inadequate proxy or competence | Both formats cannot express/learn the task; easy template ceiling | Repair scope or use a declared diagnostic; no broad mechanism rejection |
| Local capacity limitation | Unsupported operation, memory pressure, impractical sensitivity | Smaller faithful setting or alternate validated backend; report scope limit |
| Evidence against representation complementarity | Cost-matched joint errors/headroom do not improve with adequate precision | Retain negative finding; do not add formats without a new causal reason |
| Complementarity without useful selection | Oracle headroom improves but actual answers do not | Investigate source judge and selection; bound additional work by measurable headroom |
| Mechanism works without competitive advantage | Attribution positive but tuned baseline/frontier dominates | Report empirical insight; stop practical-performance claim |
| Transfer failure | Synthetic effect vanishes on properly supported human text | Restrict claim, investigate translation/ambiguity; no universal LLM improvement claim |
| Statistical ambiguity | Interval includes gain and harm at achieved sensitivity | State unresolved range; plan independent higher-information study only with justification |

## 9. Repository, interfaces, logging and reporting

### 9.1 Minimal repository layout

Use a normal local Git repository. The following paths are proposed project paths, not supplied downloadable code.

| Path | Responsibility |
|---|---|
| README.md; research_poc_plan.md | Setup, claims, command map and this plan |
| pyproject.toml; requirements.lock; assets.yaml | Package metadata, resolved dependencies, model/data/license manifest |
| configs/ | Development, pilot, confirmation, robustness and transfer YAML files |
| prompts/ | Exact source-controlled templates and allowed development examples |
| src/reprlab/schema.py | Strict public inputs, candidate records and status enums |
| src/reprlab/logic/ | Boolean/FOL ASTs, table compiler, solver wrappers and witness serializers |
| src/reprlab/data/ | Generator, renderers, group splits, FOLIO adapters and public/private separation |
| src/reprlab/backends/ | MLX adapter and optional separately validated backend |
| src/reprlab/methods/ | Named baseline and treatment implementations |
| src/reprlab/selection/ | Semantic grouping, source-witness adjudication and fallback policies |
| src/reprlab/budget.py; runner.py | Deadline ledger, atomic checkpoints and randomized scheduling |
| src/reprlab/analysis/ | Paired estimators, uncertainty, power simulation and plots/tables |
| tests/ | Meaningful semantic, accounting, leakage, recovery and baseline fixtures |
| data/public/; data/oracle/ | Inference-visible inputs versus isolated scorer metadata |
| artifacts/runs/<run_id>/ | Immutable manifest, raw calls, outcomes, timing and errors |
| artifacts/reports/ | Generated tables, reports and decision records |
| preregistrations/; amendments/ | Frozen designs and timestamped changes |
| decisions.md; progress.json | Resumable project state and evidence-based next action |

Use JSONL and CSV for core logs/tables to avoid an unlisted storage dependency. If Parquet is later useful, add and lock an actual writer such as PyArrow, and keep a simple export route. Raw outputs may be compressed losslessly; retain exact uncompressed SHA-256.

### 9.2 Shared interfaces

~~~python
from dataclasses import dataclass
from typing import Literal, Protocol

Answer = Literal['ENTAILED', 'CONTRADICTED', 'UNDETERMINED', 'INCONSISTENT']
Status = Literal['ok', 'abstain', 'invalid', 'unsupported', 'timeout',
                 'oom', 'backend_error', 'infrastructure_interrupted']

@dataclass(frozen=True)
class Outcome:
    task_id: str
    method_id: str
    answer: Answer | None
    status: Status
    committed_at_active_seconds: float | None
    candidate_ids: tuple[str, ...]
    call_ids: tuple[str, ...]
    unresolved_disagreement: bool
    failure_reason: str | None

class Method(Protocol):
    def run(self, public_instance, backend, budget, seed_schedule) -> Outcome: ...

class Backend(Protocol):
    def generate(self, messages, generation_config, call_key, budget): ...

class SemanticEngine(Protocol):
    def compile(self, parsed_representation): ...
    def classify(self, premises, query, timeout_seconds): ...
    def equivalent(self, left, right, timeout_seconds): ...
    def witness(self, left, right, direction, timeout_seconds): ...
~~~

These are project interfaces. They intentionally do not invent upstream MLX response fields. A single backend adapter converts the installed API's actual objects to a documented local GenerationRecord. Method implementations cannot import oracle modules or receive answer labels. Solver results are typed SAT/UNSAT/UNKNOWN/TIMEOUT/ERROR, distinct from answer labels.

Strict JSON parsing rejects repeated object keys before schema validation, nonfinite numeric values, unexpected properties, duplicate IDs and invalid types. Boolean fields accept only actual JSON Booleans. Define public byte/node/depth limits and check them before recursive traversal or table expansion. Limits are determined from supported input/schema sizes and applied to every compatible method; they are not changed to rescue one protected result.

### 9.3 Configuration conventions

~~~yaml
schema_version: 1
phase: confirmatory
experiment_id: resolve_before_freeze
model:
  repository: resolve_in_WP02
  revision: resolve_before_freeze
  tokenizer_sha256: resolve_before_freeze
  precision: resolve_before_freeze
  backend_lock_sha256: resolve_before_freeze
population:
  generator_revision: resolve_before_freeze
  strata_manifest_sha256: resolve_before_freeze
  seed_namespace: confirm_v1
  primary_view_policy: one_preselected_renderer_per_root
methods:
  treatment: fixed_formula_table
  primary_comparators: []
budget:
  active_seconds_per_task: null
  max_output_tokens_per_call: null
  solver_seconds_per_call: null
  cache_policy: empty_task_cache
analysis:
  alpha_familywise: 0.05
  planned_root_count: null
  protected_results_visible_during_run: false
failure_policy:
  invalid_or_abstain_scores_zero: true
  preserve_failed_attempts: true
  resume_at: completed_call_boundary
~~~

All placeholders and null decision fields cause freeze to fail. Store final configuration in a canonical serialization and hash it. Derive seeds from a cryptographic hash of master seed, task ID, method ID, role and draw index, rather than relying on mutable global RNG order. Do not use Python's process-randomized hash for reproducibility. Log backend nondeterminism where seeds cannot ensure identical outputs.

### 9.4 Experiment-record schema

Every call record must include:

~~~json
{
  "schema_version": 1,
  "run_id": "...", "attempt_id": "...", "call_id": "...",
  "phase": "pilot_or_confirmatory_or_replication",
  "task_id": "...", "group_id": "...", "view_id": "...",
  "method_id": "...", "role": "translate_or_judge_or_repair",
  "representation": "F_or_T_or_native",
  "source_sha256": "...", "prompt_sha256": "...",
  "model_revision": "...", "quantization_sha256": "...",
  "config_sha256": "...", "source_commit": "...",
  "seed": 0,
  "input_tokens": 0, "output_tokens": 0,
  "prefill_seconds": 0.0, "decode_seconds": 0.0,
  "solver_seconds": 0.0, "controller_seconds": 0.0,
  "active_seconds_before": 0.0, "active_seconds_after": 0.0,
  "wall_start_utc": "...", "wall_end_utc": "...",
  "backend_peak_bytes": 0, "process_peak_rss_bytes": 0,
  "swap_delta_bytes": 0,
  "status": "...", "termination_reason": "...",
  "raw_output_path": "...", "raw_output_sha256": "...",
  "cache_hit": false, "infrastructure_retry_of": null
}
~~~

Use null for unavailable measurements rather than a misleading zero. Do not log credentials, unrelated local files or hardware serial numbers. Gold labels and fidelity measurements are appended by the isolated scorer in separate scored records only after the protected run is complete. Call metrics can be summarized during execution without exposing outcome comparisons.

Progress records contain completed work packages, artifact paths/hashes, unresolved blockers, current run/manifest, last completed call and next permitted action. Decision records include the evidence, chosen branch, discarded alternatives and whether a claim changed. A fresh agent must be able to resume from these records without chat memory.

### 9.5 Proposed project commands

The coding agent must implement these under a small CLI. They are **not existing package commands**. After creating pyproject.toml and the package layout in §9.1, install the project into the validated environment; always invoke its explicit interpreter:

~~~bash
.venv/bin/python -m pip install -e .
.venv/bin/python -m reprlab doctor --config configs/discovery.yaml
.venv/bin/python -m reprlab validate-semantics --config configs/core.yaml
.venv/bin/python -m reprlab calibrate --config configs/pilot.yaml
.venv/bin/python -m reprlab reproduce-baselines --config configs/development.yaml
.venv/bin/python -m reprlab pilot --config configs/pilot.yaml
.venv/bin/python -m reprlab plan-sample --config configs/pilot.yaml
.venv/bin/python -m reprlab freeze --config configs/confirm.yaml
.venv/bin/python -m reprlab run --config configs/confirm.yaml
.venv/bin/python -m reprlab resume --manifest artifacts/runs/RUN_ID/manifest.json
.venv/bin/python -m reprlab score --manifest artifacts/runs/RUN_ID/manifest.json
.venv/bin/python -m reprlab report --manifest artifacts/runs/RUN_ID/manifest.json
~~~

Doctor writes hardware/dependency readiness; validate-semantics runs exact fixtures; calibrate writes costs; reproduce-baselines writes source-adaptation checks; pilot writes exploratory results; plan-sample computes cost/sensitivity and proposed fixed N; freeze validates and locks; run refuses unfrozen confirmation; resume enforces hashes and remaining budgets; score checks completion and opens oracle data; report regenerates all tables from raw records. Exit nonzero with an interpretable error if a gate fails. Never implement freeze as a superficial filename change.

### 9.6 Required final result tables

1. Asset/model/backend provenance, access limits and exact executed configurations.
2. Every screened baseline, validation status, tuning cost and reason for inclusion/exclusion.
3. Primary per-method accuracy, pairwise differences/intervals and multiplicity decision at the fixed budget.
4. Natural-cost frontier, latency distribution, memory, token and preparation costs.
5. Coverage/false acceptance and abstention/failure decomposition.
6. Marginal fidelity, whole-candidate and compositional headroom, joint errors and wrong agreement.
7. Required ablations, robustness, independent replication and transfer, with their exploratory/confirmatory status.
8. All methodological amendments, invalidated evidence, remaining limitations and the next decision.

Report improvement in one dimension as such. A tiny positive gain with excessive cost, strong regressions, unavailable closest baseline or weak transfer does not justify a breakthrough claim. Preserve raw unfavorable outputs and include uncertainty even when the point estimate is appealing.

## 10. Outcome interpretation and evidence-triggered extensions

### 10.1 Decision rules

| Outcome | Defensible conclusion | Next action |
|---|---|---|
| P improves over validated LLM comparators, survives decisive controls and independent replication | Scoped representation-intervention benefit on the tested model/population/budget | Test practical transfer and amortization; consider training only if justified |
| P beats weak direct reasoning but not strong formalization, T/T or CPL | Existing technique explains the performance | Use the better method; retain diagnostic insight, withdraw novelty/performance claim |
| Joint errors fall but no end-to-end gain | Complementarity exists, exploitation or cost is inadequate | Improve selection only within measured oracle headroom; compare against simpler allocation |
| Accepted accuracy rises but coverage/cost deteriorates | A selective tradeoff, not unconditional superiority | Report frontier and whether a real application values that operating point |
| Strong synthetic gain; direct human-text transfer absent/underpowered | Bounded mechanistic finding | No general usefulness claim; diagnose vocabulary, source complexity or proxy mismatch |
| Valid negative with narrow interval | Evidence against the tested effect at that precision and scale | Stop that configuration; escalate only with a distinct, evidence-based hypothesis |
| Wide interval, weak model competence or missing baseline | Inconclusive or incomplete comparison | State exactly what remains unresolved and the cheapest discriminating next experiment |
| New prior art matches the contribution | Replication or engineering adaptation | Update attribution and reuse it; do not manufacture a new name |

A valuable result need not be positive. A well-controlled finding that structured diversity adds no benefit over strong semantic resampling is a successful answer to the research question.

### 10.2 Optional work package T — Amortize a verified representation advantage

**Entry:** H1/H2 survive controls, translation cost is material, and a simpler single-format method does not dominate. **Objective:** determine whether supervised adaptation can retain the benefit at lower inference cost; this is related to existing MoT/typed-thinking work, not presumptively new.

Create matched training sets from development-only generated worlds. Gold semantics may supervise training, with equal access for all trained comparators. Compare mixed-format supervised adaptation with single-format adaptation and direct-answer/explicit-chain adaptation using matched examples, unique task groups, training tokens/compute and target inference budget. Filter traces by full semantics where feasible rather than by a lucky final answer. A source judge's approval is not a gold training label.

Implement training_data.py, training_config.yaml and an MLX-LM LoRA adapter only after its actual supported model path passes forward/backward and save/reload tests. LoRA rank, target modules, optimizer and sequence length are development choices recorded in a bounded search. Include untrained base and equal-parameter adapter controls. Charge trace generation, filtering, training and storage.

Measure accuracy/cost on fresh structures and new renderers; test that gains persist after removing inference ensembles. Compute the amortization break-even query volume V*=extra_adaptation_cost/(baseline_per_query_cost−adapted_per_query_cost) only when the denominator is positive and accuracy is acceptable. If error covariance improves only on seen templates, reject the generalization claim. If training consumes more local resources than a credible deployment would amortize, retain the frozen-system result.

**Resources/gate:** measure actual peak and tokens/second before choosing training size; quantized LoRA support is model/path-specific. Full pretrained-model retraining is not required. **Outputs:** locked data/config, checkpoint hashes, training costs, OOD evaluation and an explicit proceed/stop/inconclusive decision. A new training claim requires fresh confirmation.

### 10.3 Optional work package L — Test a genuine continuous-channel hypothesis

**Entry:** the project explicitly targets an internal computational mechanism, or primary diagnostics reveal a precise channel hypothesis that cannot be answered by external representations. Mere failure of the primary method is not sufficient reason to start architecture work.

**Hypothesis:** with matched task information, supervision and computation, a continuous intermediate channel carries problem-specific information that improves performance beyond explicit tokens or extra empty computation positions. Existing Coconut/CODI/pause/recurrent/soft-switching work defines the baseline landscape [R15–R20].

Use exactly generated state-update tasks with known intermediates. An affine regime, st+1=(a_t·st+b_t) mod p, is a **compressible control**: updates compose associatively as parameter pairs (A,B), so recurrence notation does not establish sequential computational difficulty. Add a nonlinear regime such as st+1=(a_t·st²+b_t·st+c_t) mod p, or small explicit lookup transitions, and demonstrate on development that the chosen lengths offer learnable headroom beyond direct-answer/filler controls. No claim of theoretical strict sequentiality follows from these choices. Publish the initial state, coefficients or lookup tables, modulus and sequence; intermediate states are training/scorer-only. Render the same instances as text and arrays. Split by generator groups and held-out lengths, with fresh symbol mappings. This is a component test, not evidence of general LLM improvement.

Compare a small decoder trained for direct answers; explicit correct intermediate states; fixed filler positions; hidden-state feedback with a matched curriculum; and a distillation-based latent baseline. Equalize examples and disclose extra per-step labels. Match training and inference compute where the claim demands it; equal epochs alone are not equal compute. A pretrained small model may be reused, but do not confuse its advantages with the new channel.

Test chain deletion, same-length constant fillers, donor states from independent examples and targeted correct/incorrect intermediate-state patches. Preserve position/cache structure. A probe decoding the answer before the chain does not prove chain necessity; a noisy off-manifold perturbation hurting accuracy does not prove faithful computation. Require content-specific, semantics-predicted intervention effects beyond filler controls, alongside the relevant accuracy/cost effect and independently trained seed replication.

Start CPU FP32 reference forward/gradient fixtures, then port to native MPS or MLX and validate the actual recurrent operations. The inspected Coconut launcher uses CUDA/NCCL and is not directly Mac-ready. Keep a separate environment for old research dependencies. A 124M FP32 Adam model has a rough 16P≈1.98 GB decimal parameters/gradients/moments lower-level estimate before activation retention and workspaces; recurrent unrolls may dominate. Calibrate before scaling.

**Outputs:** task generator, matched training configs, forward/gradient parity report, checkpoints, intervention outcomes and compute ledger. **Failure branches:** if no baseline learns, repair task/supervision; if filler matches latent, reject a continuous-feedback performance advantage at the tested compute, while conclusions about content use require deletion, donor-state and targeted intervention evidence. Constant filler embeddings can still induce problem-specific hidden states. If local precision/support fails, report resource/implementation limits; if success appears only at easy lengths, restrict the claim. Do not reject a scale-dependent architecture because an unexpressive toy failed, and do not excuse every negative result by invoking hypothetical future scale.

## 11. Performed checks, audit record and residual uncertainty

### 11.1 What was actually done during this research

Primary papers, official proceedings, model cards/configs, selected implementation source, dataset documentation and current backend documentation were inspected. Three separate research/audit passes addressed prior art, experimental validity and local implementation feasibility. These were assistant-led parallel reviews, not external peer review or empirical replication.

A small **CPU-only analytical check** ran in Python 3.12.14 on Linux x86_64, not on the target laptop. It enumerated 65,814 Boolean functions across arities zero through four and checked that an allowed-row relation and its disjunctive compilation agree on every assignment. It also checked label distinctions and a consistent-wrong example:

\[
G=(a\Rightarrow b)\land(b\Rightarrow c)\land a,
\quad F=(b\Rightarrow a)\land(b\Rightarrow c)\land a.
\]

G entails c; F leaves c undetermined. Another differently written candidate equal to F agrees with it on every world. The assignment a=1,b=0,c=0 is a separating witness: F permits it while G rejects it. This validates the need to score semantic correctness separately from agreement; it is not evidence that the proposed LLM method improves anything.

The paired sample-size illustrations and KV arithmetic were calculated. An initial planning assumption of 32 Qwen3-8B layers was corrected to the inspected configuration's 36 layers before delivery, yielding 1.125 GiB at 8,192 two-byte-cache tokens. No model inference, training, target-device throughput, baseline reproduction or protected benchmark was executed during preparation of this plan.

The finite check can be reproduced with this compact reference snippet; it is intentionally independent of the future project code:

~~~python
from itertools import product

checked = 0
for k in range(5):
    rows = list(product((False, True), repeat=k))
    for mask in range(1 << len(rows)):
        allowed = [r for i, r in enumerate(rows) if (mask >> i) & 1]
        for x in rows:
            direct = x in allowed
            compiled = any(all(v == b for v, b in zip(x, row))
                           for row in allowed)
            assert direct == compiled
        checked += 1
assert checked == 65814
~~~

This establishes a finite compiler property only. The project still needs tests of its actual implementation, parser, solver integration and natural-language templates.

### 11.2 Audit findings and revisions

| Audit | Concrete defect found | Revision / acceptance condition |
|---|---|---|
| Scientific/prior art | Broad representation/verification proposal already exists | Primary labeled replication plus empirical extension; CPL, CLOVER and MoT explicitly included |
| Scientific | Query negation was grouped with meaning-preserving changes | Separate invariant transformations from equivariant label mapping |
| Scientific | Boolean sentences could quietly import defaults, modality or anaphora | Explicit core grammar and source contract; broader language requires a distinct adapter |
| Experimental | A perfect generator parser conflicted with “strongest baseline” selection | Primary scientific claim scoped to LLM-based methods; parser retained as practical ceiling |
| Experimental | F/T beating F/F might merely reflect superior T | Best single-format comparator and same-format prompt-diversity control required |
| Experimental | Candidate-level headroom missed compositional recovery | Whole-candidate and per-clause/composed oracle ceilings recorded separately |
| Experimental | Agreement/accepted precision could hide wrong semantics or rejection | Full denotation, unconditional success, false acceptance, coverage and full cost all reported |
| Experimental | Zero-discordance pilot and underspecified multiplicity could invalidate inference | Conservative finite-sample uncertainty, explicit frozen analysis and null/power simulation |
| Experimental | Within-confirmation deduplication and fixed cell counts conflicted with IID analysis | Fixed structural partition, with-replacement mixture draws, retained collisions and total N; device-dependence sensitivity |
| Experimental | Negative-pilot planning could seek impossible positive-effect power | Explicit hypothetical positive-effect sensitivity or a signed-effect interval-precision objective |
| Implementation | Missing/duplicate records had ambiguous salvage behavior | Strict parsing and common explicit response/record policy |
| Implementation | Baseline stage/aggregation contracts were incomplete | Native versus shared adapters distinguished; decomposition and synthesis acceptance gates specified |
| Implementation | Full-FOL transfer changed the representation and witness semantics | Direct propositional transfer distinguished from quantified variant; separate query/model serialization gate |
| Implementation | Transfer symbol alignment and binders were underspecified | Ground-atom tuple IDs for T1; closed, scope-checked F/G formulas for T2 |
| Feasibility | Base MLX requirements were weaker than current MLX-LM requirements | Current package metadata checked; compatible native interpreter and lock required |
| Feasibility | Cached/resumed calls could refresh time budgets | Separate active-time, replay and total-consumed-compute ledgers; no free additional reasoning |
| Feasibility | Parquet outputs required an unlisted dependency | Core outputs use CSV/JSONL; optional writer must be pinned explicitly |
| Reproducibility | Partial cache restart could alter answers silently | Resume at completed calls unless cache parity is demonstrated |
| Architectural | Affine recurrence was mistaken for intrinsic sequential difficulty | Affine task labeled a compressible control; nonlinear/lookup regime requires measured learnable headroom |
| Architectural | Equal filler/latent accuracy was overinterpreted as unused content | Performance conclusion separated from content-specific causal intervention evidence |

After these revisions, targeted scientific/experimental, architectural and implementation challenge passes found no further substantive correction within their inspected scope. That is a document-review result, not proof of implementation correctness, global novelty or successful execution on the target device. Further methodological edits must be followed through to schemas, work packages and reporting and challenged again.

### 11.3 Residual risks and blocked claims

The most consequential remaining uncertainties are model competence in tables, actual source-judge reliability, faithful local availability of the closest baseline, current package/model compatibility, the laptop's real capacity, human-text subset size/quality and achievable statistical resolution. Each is a gate above. There is no guarantee of novelty or positive results.

Recent 2026 preprints are informative competing approaches, not settled replicated evidence. Search cannot prove global absence of prior art. Artifact landing-page access does not prove its code works locally. Formal equivalence does not prove human-intent fidelity. Open-weight availability is not the same as permissive code, dataset and redistribution licensing.

The core's curated vocabulary, clause arity and controlled renderer are deliberate causal controls and substantial limits. A useful empirical result may remain narrow. Architectural or real-world improvements require their own evidence; this plan does not imply them from a successful Boolean demonstration.

## 12. References and search-coverage record

### 12.1 Evidence ledger and bibliography

All web materials below were checked during research on 27 October 2026. “Inspected” describes the evidence actually available to the research process, including parallel research; it does not imply code execution or full replication. Mutable repositories/cards must be pinned by revision when used. Some entries group closely related original sources; links remain ordinary URLs so the handoff does not depend on chat citation tokens.

- **R01 — Luyu Gao et al. (2023), *PAL: Program-aided Language Models*. ICML.** [Paper](https://proceedings.mlr.press/v202/gao23f.html); [official code](https://github.com/reasoning-machines/pal). Inspected official paper record and repository documentation. Establishes executable intermediate programs; older model/benchmark conditions do not establish present local performance. Apache-2.0 repository label inspected.

- **R02 — Wenhu Chen, Xueguang Ma, Xinyi Wang and William W. Cohen (2023), *Program of Thoughts Prompting: Disentangling Computation from Reasoning for Numerical Reasoning Tasks*. TMLR.** [Paper](https://arxiv.org/abs/2211.12588); [code](https://github.com/TIGER-AI-Lab/Program-of-Thoughts). Inspected paper record and official implementation documentation. Numerical reasoning and program execution are existing mechanisms; MIT repository label inspected.

- **R03 — Xi Ye, Qiaochu Chen, Isil Dillig and Greg Durrett (2023), *SatLM: Satisfiability-Aided Language Models Using Declarative Prompting*; Liangming Pan et al. (2023), *Logic-LM*.** [SatLM paper](https://arxiv.org/abs/2305.09656), [code](https://github.com/xiye17/SAT-LM); [Logic-LM paper](https://aclanthology.org/2023.findings-emnlp.248/). Inspected original records and SatLM repository. These inform the strong single-formalization baseline and the distinction between parsed-specification correctness and source fidelity. SatLM MIT label inspected; do not assume every related repository has identical terms.

- **R04 — Hyun Ryu, Gyeongman Kim, Hyemin S. Lee and Eunho Yang (2025), *Divide and Translate: Compositional First-Order Logic Translation and Verification for Complex Logical Reasoning*. ICLR.** [Paper/PDF](https://proceedings.iclr.cc/paper_files/paper/2025/file/3e592c571de69a43d7a870ea89c7e33a-Paper-Conference.pdf); [official code](https://github.com/Hyun-Ryu/clover). Inspected method, Algorithm 1, accumulation rules, examples, comparison tables, README and requirements. Direct overlap with counterexample adjudication. Repository subdirectory fetches were partly blocked; no reusable license was visible in the inspected root. Native source behavior and the local adapter must remain distinguishable.

- **R05 — Mohammad Raza and Natasa Milic-Frayling (2025), *Instantiation-based Formalization of Logical Reasoning Tasks Using Language Models and Logical Solvers*. IJCAI, 4633–4641.** [DOI](https://doi.org/10.24963/ijcai.2025/516); [proceedings](https://ijcai.org/proceedings/2025/516); [advertised code](https://github.com/mohammadraza4/ssv). The research pass inspected the paper's method/recipe; direct PDF retrieval was inconsistent across calls. The repository rendered empty. Supports the SSV comparator and warns that high selective precision can coexist with weak coverage.

- **R06 — Ziang Wu (2026), *COVER: Counterexample-Guided Repair for Natural-Language Constraint Modelling*. Published 16 September 2026.** [DOI](https://doi.org/10.3233/FAIA260737). Inspected publisher method/results sections in the prior-art pass. Typed repair and semantic witnesses overlap; oracle mode and lightweight-test mode have different information assumptions. No source repository was located in the inspected article. Not treated as a turnkey local baseline.

- **R07 — Daneshvar Amrollahi, Jerry Lopez and Clark Barrett (2026), *Faithful Autoformalization via Roundtrip Verification and Repair*. Preprint, 27 April.** [Paper](https://arxiv.org/abs/2604.25031); [inspected full text](https://arxiv.org/html/2604.25031v1). Method, failure modes and traffic-rule evaluation inspected. Cyclic agreement can retain a consistent error. Code/data availability in the inspected version was prospective, so no downloaded implementation is claimed.

- **R08 — Yihan Dai, Xuanlin Liu, Haoxiang Jia, Tianchi Li, Xin Zhang and Sergey Mechtaev (2026), *Validating Autoformalized Specifications via Contrapuntal Languages*. Preprint, 21 October.** [Paper](https://arxiv.org/abs/2610.30059); [inspected v1](https://arxiv.org/html/2610.30059v1); [artifact](https://figshare.com/s/be97e0291c8ce608bcd4). Full definitions, synthesis, evaluation/cost/failure sections inspected. Artifact landing page was accessible in one research pass, unavailable in another; archive not downloaded. It listed 1.89 GB and CC BY 4.0. Closest novelty-overturning source. Cite the arXiv record, not the manuscript's placeholder publisher DOI/2018 template metadata.

- **R09 — Yihan Dai, Sijie Liang, Haotian Xu, Peichu Xie and Sergey Mechtaev (2025–2026), *Reducing Hallucinations in LLM-Generated Code via Semantic Triangulation*.** [Paper, revised 8 September 2026](https://arxiv.org/abs/2511.12288); [author publication list](https://mechtaev.com/publications/). Abstract/version and author record inspected; full HTML exceeded the retrieval limit. It establishes close conceptual overlap through dissociative task variants; detailed reproduction is not claimed.

- **R10 — Vikash Singh, Darion Cassel, Nathaniel Weir, Nick Feng and Sam Bayless (2026), *VERGE: Formal Refinement and Guidance Engine for Verifiable LLM Reasoning*.** [Paper](https://arxiv.org/abs/2601.20055), first posted 27 January, revised 11 October. Method and appendix inspected in the research pass. Supports overlap in semantic consensus/routing/localization; published hosted-model/judge conditions are not local head-to-head evidence.

- **R11 — Tong Zheng, Lichang Chen, Sophia Han, R. Thomas McCoy and Heng Huang (2026), *Learning to Reason via Mixture-of-Thought for Logical Reasoning*. ICLR.** [Proceedings](https://proceedings.iclr.cc/paper_files/paper/2026/hash/af80ce1011eb35fe1023c320158c2ad9-Abstract-Conference.html); [full paper](https://proceedings.iclr.cc/paper_files/paper/2026/file/af80ce1011eb35fe1023c320158c2ad9-Paper-Conference.pdf). Inspected training/inference algorithms, modality definitions, results and compute details. Direct prior art for natural language/code/truth-table mixtures. Its training used a multi-GPU setup; inference-only local controls must not be represented as reproduction of trained MoT results.

- **R12 — Lei Xu, Pierre Beckmann, Marco Valentino and Andre Freitas (2026), *Adaptive LLM-Symbolic Reasoning via Dynamic Logical Solver Composition*. EACL.** [Paper](https://aclanthology.org/2026.eacl-long.54/); [code](https://github.com/idiap/adaptive_symbolic_reasoning). Full PDF and official record inspected. Solver-type prediction and successful formalization are separate capabilities, particularly for smaller models. Informs the decision to defer routing complexity.

- **R13 — Xuezhi Wang et al. (2023), *Self-Consistency Improves Chain of Thought Reasoning in Language Models*; Wan et al. (2025), *Derailer-Rerailer: Adaptive Verification for Efficient and Reliable Language Model Reasoning*.** [Self-consistency](https://arxiv.org/abs/2203.11171); [adaptive verification](https://aclanthology.org/2025.findings-acl.18/). Original records inspected; the latter's full PDF exceeded access limits. Establish generic sample/verification controls; no uninspected detailed algorithm is made a required dependency.

- **R14 — Danqing Wang, Jianxin Ma, Fei Fang and Lei Li (2025), *TypedThinker: Diversify Large Language Model Reasoning with Typed Thinking*. ICLR.** [Proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/hash/494ad6d9b36d9a1528dbf3baeb4e8a72-Abstract-Conference.html); [preprint](https://arxiv.org/abs/2410.01952). Official abstract and metadata inspected after OpenReview's forum presented an access challenge. Reasoning-type selection is already studied; it is related to, but not identical with, changing semantic encodings.

- **R15 — Shibo Hao et al. (2024–2026), *Training Large Language Models to Reason in a Continuous Latent Space* (Coconut).** [Paper v4, 23 August 2026](https://arxiv.org/html/2412.06769v4); [official code](https://github.com/facebookresearch/coconut). Method/results and selected run.py/coconut.py paths inspected. MIT code; source launcher uses CUDA/NCCL. Supports a legitimate latent-channel alternative and conditional benefits, not a universal replacement for explicit reasoning.

- **R16 — Shen et al. (2025), *CODI: Compressing Chain-of-Thought into Continuous Space via Self-Distillation*. EMNLP.** [Paper](https://aclanthology.org/2025.emnlp-main.36/); [code](https://github.com/zhenyi4/codi). Method/limitations and repository overview inspected. Distillation is an existing stronger latent baseline; repository redistribution terms were not established from the inspected root.

- **R17 — Sachin Goyal et al. (2024), *Think before you speak: Training Language Models With Pause Tokens*; Jacob Pfau, William Merrill and Samuel R. Bowman (2024), *Let's Think Dot by Dot: Hidden Computation in Transformer Language Models*.** [Pause tokens](https://arxiv.org/abs/2310.02226); [filler tokens](https://arxiv.org/abs/2404.15758). Original task/training discussions inspected. These motivate matched extra-computation and supervision controls, not the assumption that an untrained pause prompt is effective.

- **R18 — Jonas Geiping et al. (2025), *Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach*; Zhen Zhang et al. (2025), *Soft Thinking*.** [Recurrent depth](https://arxiv.org/abs/2502.05171), [Huginn model](https://huggingface.co/tomg-group-umd/huginn-0125); [Soft Thinking](https://arxiv.org/abs/2505.15778), [code](https://github.com/UCSB-AI/Soft-Thinking). Methods, model/runtime requirements and limitations inspected. Reusing a checkpoint differs radically in cost from reproducing its pretraining; embedding mixtures do not automatically implement parallel reasoning.

- **R19 — Jingjie Qiu et al. (2026), *Does Latent Chain-of-Thought Actually Reason? A Single, Unifying Taxonomy for Latent Reasoning*; Junhong Wu et al. (2025), *LLMs are Single-threaded Reasoners: Demystifying the Working Mechanism of Soft Thinking*.** [Taxonomy preprint, 15 October 2026](https://arxiv.org/abs/2610.21182); [soft-thinking analysis, revised 16 October 2025](https://arxiv.org/abs/2508.03440). PDFs/intervention definitions inspected in the architectural pass; original metadata rechecked. Findings vary by task/recipe; they motivate content-specific causal tests rather than blanket claims for or against latent computation.

- **R20 — Dachuan Shi et al. (2026), *SwiReasoning*; Xuan Li et al. (2026), *LaTER*; Malarkkan et al. (2026), *Invariant Reasoning Directions in Latent Trajectories of Language Models*.** [Switching](https://arxiv.org/abs/2510.05069); [latent/external stages](https://arxiv.org/abs/2605.07315); [invariance](https://arxiv.org/abs/2606.29164). Relevant methods/limitations inspected in the architectural pass; some later retrievals failed. These defeat broad novelty claims for adaptive switching and invariant latent trajectories. No local implementation is claimed.

- **R21 — Hope Kean et al. (2026), *Evidence from formal logical reasoning reveals that the language of thought is not natural language*. PNAS 123(28), e2520095123.** [DOI](https://doi.org/10.1073/pnas.2520095123); [PubMed](https://pubmed.ncbi.nlm.nih.gov/42406951/). Online 6 July; issue 14 July. Abstract/significance and bibliographic record inspected. Full participant methods were unavailable in the retrieved text; PMC availability was listed for 6 January 2027. Supports the human-task motivation only.

- **R22 — Evelina Fedorenko, Steven T. Piantadosi and Edward A. F. Gibson (2024), *Language is primarily a tool for communication rather than thought*. Nature 630, 575–586.** [DOI](https://doi.org/10.1038/s41586-024-07522-w); [author-hosted PDF](https://mcgovern.mit.edu/wp-content/uploads/2024/08/s41586-024-07522-w.pdf). Perspective and dissociation/scaffolding discussion inspected. Provides context, not a causal result about transformers.

- **R23 — Jize Jiang and Chengxiang Zhai (2026), *Competence-Based Evaluation of Logical Reasoning in Large Language Models*; Zhou et al. (2026), *LGMT: Logic-Grounded Metamorphic Testing for evaluating the reasoning reliability of LLMs*.** [Competence evaluation, 16 October](https://arxiv.org/html/2610.22894v1); [LGMT DOI](https://doi.org/10.1016/j.knosys.2026.116324). CBE full text inspected; LGMT publisher abstract was accessible but full article returned 403. Invariance/equivariance testing has prior art; detailed inaccessible LGMT procedures are not assumed.

- **R24 — William Feng, Ethan Lou and Aryan Sharma (2026), *Characterizing Paraphrase-Induced Failures in Lean 4 Autoformalization*.** [Paper v2, 17 May](https://arxiv.org/abs/2604.23135); earlier v1 was titled *Surface Sensitivity in Lean 4 Autoformalization*. Original/full-text research inspection and revised metadata checked. Informs the distinction between compilation failures and semantic differences in successfully compiling outputs; do not generalize its conditional finding universally.

- **R25 — Iman Mirzadeh et al. (2024), *GSM-Symbolic: Understanding the Limitations of Mathematical Reasoning in Large Language Models*.** [Paper](https://arxiv.org/abs/2410.05229). Original record and research findings inspected; some full-text retrievals failed. Supports controlled perturbation as a diagnostic. Reported brittleness is not proof that models cannot reason.

- **R26 — Amazon Web Services, *What are Automated Reasoning checks in Amazon Bedrock Guardrails?* Current documentation.** [Official documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails-automated-reasoning-checks.html). Workflow, translation dependency and limitations inspected. Deployed precedent; excluded from the local workload.

- **R27 — Apple, *MacBook Pro (14-inch, M3 Pro or M3 Max, November 2023): Technical Specifications*.** [Official specifications](https://support.apple.com/en-us/117736). Memory/configuration information inspected. Establishes hardware variability; no measured target-machine configuration is inferred.

- **R28 — MLX and MLX-LM contributors, current official documentation and source.** [Install](https://ml-explore.github.io/mlx/build/html/install.html); [MLX-LM README](https://github.com/ml-explore/mlx-lm); [package requirements](https://github.com/ml-explore/mlx-lm/blob/main/pyproject.toml); [qwen3_5 implementation](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/qwen3_5.py); [qwen3_next implementation](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/qwen3_next.py). Requirements, console entries and selected model/dtype/cache/conversion code inspected; not run. MIT repository metadata inspected. [Peak memory](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.get_peak_memory.html), [cached memory](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.get_cache_memory.html), [memory limit](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.set_memory_limit.html) and [synchronization](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.synchronize.html) inform measurement boundaries. A memory limit is not a hard allocation sandbox.

- **R29 — Qwen Team, official model cards/configurations (2025–2026).** [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B), [9B](https://huggingface.co/Qwen/Qwen3.5-9B), [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B); [9B config](https://huggingface.co/Qwen/Qwen3.5-9B/raw/main/config.json), [27B config](https://huggingface.co/Qwen/Qwen3.8-27B/raw/main/config.json); [Qwen3-8B config](https://huggingface.co/Qwen/Qwen3-8B/raw/main/config.json), [14B config](https://huggingface.co/Qwen/Qwen3-14B/raw/main/config.json). Dimensions, license metadata and inference-mode guidance inspected. Cards/configs justify candidate selection and arithmetic, not local capability or benchmark claims. Apache-2.0 official model metadata inspected; final asset manifests remain required.

- **R30 — MLX Community, *Qwen3.8-27B-oQ4*, conversion updated 16 August 2026.** [Conversion card](https://huggingface.co/mlx-community/Qwen3.8-27B-oQ4). Quantization description and displayed asset size inspected; complete tensor/provenance audit not performed. Mixed precision and parent revision must be recorded before use.

- **R31 — Google DeepMind, *Gemma 4 E4B instruction-tuned model*, 2026.** [Official model card](https://huggingface.co/google/gemma-4-E4B-it); [license documentation](https://ai.google.dev/gemma/docs/gemma_4_license). Official card, Apache-2.0 reference and total/effective-parameter distinction inspected. Independent-family candidate only; backend support is an unperformed local gate.

- **R32 — Z3 contributors, official solver repository, packaging and API.** [Repository](https://github.com/Z3Prover/z3); [Python packaging](https://github.com/Z3Prover/z3/blob/master/src/api/python/setup.py); [API](https://z3prover.github.io/api/html/z3.html). Repository/package metadata and macOS arm64 packaging support inspected. MIT source; actual installation and solver cross-check remain local prerequisites. Solver uncertainty must be handled explicitly.

- **R33 — PyTorch and llama.cpp contributors, current backend documentation/source.** [MPS documentation](https://docs.pytorch.org/docs/stable/notes/mps.html); [llama.cpp](https://github.com/ggml-org/llama.cpp); [qwen35 implementation](https://github.com/ggml-org/llama.cpp/blob/master/src/models/qwen35.cpp). General availability and selected implementation details inspected. These are fallback leads, not evidence of equivalence for an arbitrary custom operation or model conversion.

- **R34 — Simeng Han et al. (2024), *FOLIO: Natural Language Reasoning with First-Order Logic*. EMNLP; Yale dataset repositories.** [Paper](https://aclanthology.org/2024.emnlp-main.1229/); [original data](https://github.com/Yale-LILY/FOLIO), [license](https://github.com/Yale-LILY/FOLIO/blob/main/LICENSE); [updated repository](https://huggingface.co/datasets/yale-nlp/FOLIO). Paper, schema and original version/license inspected; updated bytes gated/unavailable. Supports human-authored transfer with story-level clustering and explicit version distinctions.

- **R35 — Original authors' adjacent dataset repositories.** Saparov et al., [PrOntoQA](https://github.com/asaparov/prontoqa) (Apache-2.0; generator/bug-fix notes inspected); Liu et al., [LogiQA2.0](https://github.com/csitfun/LogiQA2.0) (2023; examination sources and CC-BY-NC-SA-4.0 notice inspected); Sinha et al., [CLUTRR](https://github.com/facebookresearch/clutrr) (2019; generator/split guidance and CC-BY-NC-4.0 inspected). These inform diagnostics/alternatives; none is silently substituted for the declared core. ProofWriter archive contents/license were not verified and are not required.

- **R36 — Rotem Dror, Gili Baumer, Segev Shlomov and Roi Reichart (2018), *The Hitchhiker's Guide to Testing Statistical Significance in Natural Language Processing*. ACL.** [Paper](https://aclanthology.org/P18-1128/); [PDF](https://aclanthology.org/P18-1128.pdf). Original methodological discussion inspected. Supports matching inference to metric/design. The exact interval construction, power arithmetic and project decision rules in this plan are specified directly and are not claimed as a new statistical method.

- **R37 — Tsinghua University-associated patent publication CN121118877A (12 December 2025), *Automatic Formalization Method and Apparatus for Mathematical Propositions Based on Reinforcement Learning*.** [Published record](https://patents.google.com/patent/CN121118877A/en). Metadata/description inspected during a limited adjacent-art scan; priority field 6 August 2025. Relevant overlap is syntax/semantic consistency feedback for autoformalization. No validity, claim-scope, patentability or freedom-to-operate conclusion is offered.

### 12.2 Search scope and stopping rationale

The research used two web search engines, original proceedings/arXiv records, official repositories/model cards, author materials and documentation. Current and recent queries were checked through 27 October 2026; old headline results were not accepted as current local baselines.

| Search family / representative terms | Material finding or design change |
|---|---|
| “language reasoning representations,” “program/logic aided language model,” “declarative prompting” | Established PAL/PoT/SatLM/Logic-LM; removed broad novelty claim |
| “semantic self verification,” “counterexample autoformalization,” “roundtrip semantic repair” | CLOVER/SSV/COVER/roundtrip; source-conditioned witnesses already exist |
| “representation diversity correlated errors,” “contrapuntal languages,” “semantic triangulation” | October 2026 closest match; added synthesis baseline and shared-seed/coverage controls |
| “truth table LLM reasoning,” “mixture of thought,” “typed thinking” | Existing truth-table/multiformat training and inference; narrowed contribution to exact local empirical comparison |
| “adaptive symbolic solver composition,” “verification budget routing” | Routing already studied; small-model formalization failures argue against starting with a learned controller |
| “continuous latent reasoning,” “pause/filler tokens,” “recurrent depth,” “soft thinking,” “latent causal interventions” | Architectural alternatives and negative/conditional findings; separated channel, representation and compute hypotheses |
| “invariance/equivariance,” “metamorphic logical reasoning,” “paraphrase autoformalization failures” | Known robustness evaluation; separated agreement, correctness, compilation and semantics |
| Official Apple/MLX/MPS/Metal and Qwen/Gemma configurations | Current candidates, dependency minima, cache/state calculations and actual-device gates |
| Dataset authors' schemas, licenses, versions and split warnings | FOLIO version/access distinction, story clustering and explicit direct-versus-quantified transfer |
| Patent/deployed reasoning systems | Adjacent automatic-formalization record and AWS deployment precedent; no legal novelty inference |

Backward tracing followed close formalization methods to solver/program baselines and ensemble principles. Forward tracing found 2026 synthesized-language, multimodal/latent, adaptive and causal studies. After selecting the fixed intensional/extensional refinement, targeted searches for truth-table portfolios, semantic error correlation, source-grounded witness selection and representation invariance repeatedly returned the same competing families or adjacent domain-specific variants. No additional distinct indispensable baseline emerged in the final focused passes; the major unresolved issues became execution/access and empirical performance, which are explicit gates.

This is a scoped technical search, not proof that no prior art exists. Blind spots include inaccessible full texts, unindexed or private industrial systems, the unexamined large CPL archive, gated corrected data and non-English patent literature. Recheck the narrow novelty statement and asset revisions at the actual execution freeze if substantial time has passed. That is a freshness check, not delegation of this plan's essential prior-art investigation.

### 12.3 Completion contract

The research project is complete when a reproducible report can state which of the following occurred: a scoped reproducible gain; an accuracy/cost/coverage tradeoff; evidence against the proposed contribution; or a precisely bounded inconclusive result. It must identify the strongest executed comparison, any missing closest alternative, actual target-device measurements, achieved uncertainty, transfer scope and all material failures. The intended result is an auditable answer to the hypothesis, not a predetermined endorsement.

[R01]: https://proceedings.mlr.press/v202/gao23f.html
[R02]: https://arxiv.org/abs/2211.12588
[R03]: https://arxiv.org/abs/2305.09656
[R04]: https://proceedings.iclr.cc/paper_files/paper/2025/file/3e592c571de69a43d7a870ea89c7e33a-Paper-Conference.pdf
[R05]: https://doi.org/10.24963/ijcai.2025/516
[R06]: https://doi.org/10.3233/FAIA260737
[R07]: https://arxiv.org/abs/2604.25031
[R08]: https://arxiv.org/abs/2610.30059
[R09]: https://arxiv.org/abs/2511.12288
[R10]: https://arxiv.org/abs/2601.20055
[R11]: https://proceedings.iclr.cc/paper_files/paper/2026/hash/af80ce1011eb35fe1023c320158c2ad9-Abstract-Conference.html
[R12]: https://aclanthology.org/2026.eacl-long.54/
[R13]: https://arxiv.org/abs/2203.11171
[R14]: https://arxiv.org/abs/2410.01952
[R15]: https://arxiv.org/abs/2412.06769
[R16]: https://aclanthology.org/2025.emnlp-main.36/
[R17]: https://arxiv.org/abs/2310.02226
[R18]: https://arxiv.org/abs/2502.05171
[R19]: https://arxiv.org/abs/2610.21182
[R20]: https://arxiv.org/abs/2510.05069
[R21]: https://doi.org/10.1073/pnas.2520095123
[R22]: https://doi.org/10.1038/s41586-024-07522-w
[R23]: https://arxiv.org/abs/2610.22894
[R24]: https://arxiv.org/abs/2604.23135
[R25]: https://arxiv.org/abs/2410.05229
[R26]: https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails-automated-reasoning-checks.html
[R27]: https://support.apple.com/en-us/117736
[R28]: https://github.com/ml-explore/mlx-lm
[R29]: https://huggingface.co/Qwen/Qwen3.8-27B
[R30]: https://huggingface.co/mlx-community/Qwen3.8-27B-oQ4
[R31]: https://huggingface.co/google/gemma-4-E4B-it
[R32]: https://github.com/Z3Prover/z3
[R33]: https://docs.pytorch.org/docs/stable/notes/mps.html
[R34]: https://github.com/Yale-LILY/FOLIO
[R35]: https://github.com/asaparov/prontoqa
[R36]: https://aclanthology.org/P18-1128/
[R37]: https://patents.google.com/patent/CN121118877A/en
