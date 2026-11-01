# Megabyte-Scale Intelligence: Research and Local Proof-of-Concept Plan

**Research date:** 16 October 2026  
**Target machine:** one MacBook Pro with an M3 Max; actual unified memory, storage and macOS version must be discovered.  
**Execution constraint:** no remote GPU, paid model API, second computer or special equipment. Public downloads are permitted.  
**Status:** researched implementation and experimental plan, not a trained-model result. The small checks reported in §11.4 ran on a Linux CPU, not Apple Silicon.  
**Primary question:** does persistent continuous state give a tiny semantic compiler a better accuracy–storage–computation trade-off than strong discrete synthesis and repair?

## Reading and execution map

The coding agent should read §§1–6 before implementing anything, execute the dependency-ordered work packages in §7, and use §§8–11 as mandatory interpretation and audit rules. Section 12 contains primary-source references, inspection notes and search coverage. All `microbrain` commands below are **proposed project commands to implement**, not existing package commands. External installation commands are explicitly identified separately.

Evidence labels used throughout: **Verified** means inspected source content, not independently reproduced performance; **Measured here** means the limited checks in §11.4; **Derived** means arithmetic or a stated argument; **Proposed** means an experiment or design choice; **Unknown** means a blocking gate or unresolved question.

---

## 1. Executive assessment and recommended direction

### 1.1 Verdict

The original ambition is worth testing, but its main ingredients are not new. Reusing neural weights across computation steps, neural program interpretation, execution-guided program repair, learning reusable libraries, retrieval augmentation and low-bit deployment all have substantial prior art. Even loop-specific quantization now has direct competing work, including an October 2026 paper. Merely combining these ingredients would not establish a new scientific method. [R01], [R03], [R09], [R12], [R14], [R18], [R20], [R22]

**Select one primary direction: a tiny recurrent semantic compiler for bounded arithmetic and list-processing tasks, with exact execution and an explicitly controlled persistent latent workspace.** Begin around five million parameters, approximately 10 MB of FP16 weights. Compare it to an otherwise matched compiler that resets its latent workspace between attempts but retains the same observable program and execution history. Also challenge both with a tuned conventional sequence-to-sequence model, typed search, neural-guided search and contemporary local language models.

The proposed contribution is initially **a falsifiable empirical claim about the value of continuous state under a real storage and end-to-end compute budget**, not an assertion that recurrence or tool use has been invented. A convincingly favorable frontier on unseen compositions, surviving simple search and discrete-repair controls, would be useful. Failure would tell us whether the bottleneck is interpretation, search, training, representation or deployment overhead.

Do not begin with web-scale language pretraining, extreme ternary training, a general autonomous agent, a retrieval corpus or a learned tool library. Those additions multiply confounds before the indispensable mechanism has been tested. Do not demand that a component experiment prove general intelligence; conversely, do not advertise a component success as a general assistant.

### 1.2 Explicit refinement of the original proposal

The original proposal contemplated a recursive micro-LLM, exact tools, optional knowledge access and skill distillation, with roughly 16 MB and 32 MB ambitions. This plan preserves the separation of weights, computation and external information, but narrows the first useful task to **compiling a request into a small, executable, typed program**. That is a deliberate specialization, not an equivalent replacement for open-ended language generation.

There are three separately reported size claims:

| Claim tier | Definition | Initial role |
|---|---|---|
| Learned-weight budget | All deployed learned tensors, including embeddings, adapters, heads, learned stopping policies and quantization metadata | Target at or below 32,000,000 bytes; initial configuration is much smaller |
| Portable inference payload | Weights plus tokenizer, configuration, interpreter, rules, program libraries, indexes and inference source needed by this project | Attempt 16,000,000 bytes with the small model; count actual files |
| Complete installed system | Payload plus Python/native runtime, dependent libraries and every required external asset, with shared dependencies separately itemized | Measure honestly; this plan does **not** establish a complete 16 MB installed system |

Use decimal MB in these constraints and report MiB alongside memory measurements. A large compressed file that expands into large weights is not a small resident model. A 20 MB controller plus a 2 GB database is not a 20 MB complete intelligence artifact.

### 1.3 What a meaningful result would establish

A local positive result can establish that a specific learned compiler, trained locally, generalizes more effectively per stored byte or per inference budget on a specified task population. It can also establish practical task parity with a much larger locally run model **on that population**, when both receive the same inputs and tools.

It cannot by itself establish broad conversational competence, arbitrary code generation, factual knowledge, general-purpose reasoning, superiority to every current model, or that a 16 MB isolated file contains the capabilities of a multi-gigabyte assistant. These are distinct future claims with distinct data and resource requirements.

**Recommended first executable experiment:** on freshly generated typed-program tasks, train matched persistent-state and state-reset models with identical data, optimizer-update counts, feedback access and loop schedules. Start without semantic checker feedback, then add a separate three-visible-example condition. First validate the executor, split integrity and a conventional baseline. Evaluate additional cycles on held-out program structures, not just familiar templates.

---

## 2. Formal problem, scope, assumptions and hypotheses

### 2.1 Beneficiaries and present workflow

The assumed initial beneficiary is someone needing private, offline, repeatable transformations of small numeric datasets: totals, ratios, filtering, sorting, transformations and simple conditional calculations. Today that person can write a short program, use spreadsheet formulas, use a rule-based assistant, or ask a larger local model to generate code. This plan tests whether a much smaller learned front end can remove some of the programming burden without sacrificing correctness.

**Unknown:** the user has not supplied a production application, service-level objective or evidence that storage rather than latency is their actual bottleneck. Therefore report resource frontiers instead of assigning an invented monetary value to saved megabytes. The M3 Max is the experimental machine, not proof that the eventual deployment needs such a small model.

### 2.2 Task definition

A public task is

\[
x=(d,\sigma,b_q,\mathcal E),
\]

where `d` is an English description, `σ` is a typed input/output signature, `b_q` supplies the query bindings, and `E` is either empty or three visible input/output examples. The solver produces one typed program `p` and an answer obtained by executing it on `b_q`. The evaluator alone has a reference specification `g` and independent hidden bindings `H`.

For generated tasks, define strict finite-test semantic success as

\[
Y(p,x)=\mathbf 1\{p\text{ is valid and }\operatorname{Exec}(p,b)=g(b)
\text{ for } b\in H\cup\{b_q\}\}.
\]

This is a **test-based** semantic criterion, not a proof of universal program equivalence. For small finite domains, exhaustive equivalence is an additional diagnostic. A string mismatch between two correct programs is not a task failure.

The deployed solver never receives `g`, the reference program, hidden outputs, hidden bindings or evaluator diagnostics during inference. Its permitted checker sees only public examples and the query inputs. Public test success is evidence, not a certificate of the intended meaning.

### 2.3 Resource quantities

For method `m`, record:

- `W_m`: serialized learned bytes; `A_m`: portable payload bytes; `I_m`: installed footprint, with deduplicated shared files reported separately.
- `C_train,m`: project-specific training operations and accelerator/CPU wall time, including rollouts, failed runs, tuning and quantization. Separate inherited pretraining from this expenditure.
- `C_inf,m(x)`: inference operations, model calls, decoder tokens, recursive block executions, search nodes and interpreter calls.
- `T_m(x)`: end-to-end elapsed time, including parsing, tokenization, device synchronization, execution, candidate selection and answer serialization.
- `M_m`: peak measured host/process and accelerator allocations, with unified-memory overlap explicitly acknowledged.

The scientific object is an accuracy–bytes–time frontier, not “effective parameter count.” Executing a five-million-parameter block eight times does not make it a forty-million-parameter knowledge store.

### 2.4 Established facts versus assumptions

**Established by prior work:** shared recurrent computation can be trained; exact interpreters and bounded symbolic search can solve combinatorial classes; neural proposal models can guide synthesis; execution feedback can support repair; stored weight size and active compute are distinct resources. [R01], [R02], [R09], [R10], [R11], [R12]

**Assumptions to test here:** a locally trainable tiny model can understand enough unfamiliar wording; the task distribution rewards reusable computation rather than memorized templates; additional recurrent steps improve useful intermediate representations; execution feedback contains actionable information; those gains survive synchronization and interpreter overhead.

**Speculative extensions, not prerequisites:** autonomous discovery of new library procedures, retrieval-grounded factual answers, general-language pretraining from scratch, sub-three-bit recurrent training, learned halting, implicit fixed-point solvers and retrofit of a pretrained model into a new recurrence pattern.

### 2.5 Hypotheses and falsifiers

| ID | Hypothesis and predicted observation | Supporting evidence | Strong alternatives and observations against it |
|---|---|---|---|
| H1: latent state | At fixed parameters, training regimen and recurrent-call count, carrying continuous state improves held-out semantic success relative to resetting it while preserving all observable history | Universal Transformers, latent-depth models and TRM make this plausible, not proven for this task [R01], [R03], [R04] | Same gain from an autoregressive repair model; benefit only from extra supervision; no benefit after baseline tuning; unstable or reversed effects across training seeds |
| H2: compute use | Increasing inference cycles improves difficult unseen compositions without requiring new weights | Controlled loop studies show task-dependent generalization and failures [R06], [R07] | Improvement only from trying more discrete candidates; gains disappear under equal latency; final states saturate or regress; only lengths encountered in training improve |
| H3: useful feedback | Public execution feedback improves hidden-case success, not just visible-test pass rate | Synthesize–Execute–Debug and UPI–TRM provide direct precedent [R12], [R08] | Visible examples underdetermine the task; feedback creates shortcut programs; mismatch with language; no gain beyond typed search given the same examples |
| H4: practical compactness | A compact compiler offers a useful accuracy–storage–time point relative to strong tiny and larger local systems | Neural synthesis and program-aided language systems support the division of labor [R10], [R11], [R15] | Rules or enumerative search dominate; natural-language understanding consumes more capacity than the budget allows; runtime outweighs weight savings |
| H5: deployment preservation | Ordinary FP16 or 8-bit deployment retains the observed advantage | Low-bit deployment is established, but recurrent trajectories require care [R19], [R20], [R21], [R22] | Quantization shifts recurrent states, breaks late-loop behavior or saves disk bytes without saving resident memory or time |

H1 is the primary causal hypothesis. H4 is the separately gated competitiveness claim. H2, H3 and H5 are not silently promoted to confirmed claims after exploratory selection.

### 2.6 Strongest case for and against

**For:** compact software encodes reusable algorithms without storing each answer. Shared neural computation can spend effort on interpreting and composing those algorithms. The output space can be made much more structured than unrestricted prose, and generated tasks offer inexpensive ground-truth supervision and strong failure diagnosis. A persistent workspace may retain uncertainty and partial structure lost by committing to discrete programs after every attempt.

**Against:** most apparent intelligence may come from the supplied language, examples and hand-written operator library. A recurrent model can be slower than a larger feed-forward model, harder to optimize and no more robust. A discrete program already provides compact persistent memory; a neural latent state may add nothing. The model may lack the language representations needed for unfamiliar requests. Extra cycles cannot recover facts not present in weights, inputs or tools. Bounded synthetic success may have little practical value.

A finite artifact of `B` bits cannot distinguish more than `2^B` arbitrary encoded alternatives without external information. Computation can exploit structure but cannot conjure independent unavailable facts. This counting argument does not preclude compact general algorithms; it does rule out using a tiny seed as an unexplained source of arbitrary knowledge.

### 2.7 Headroom and end-to-end limits

Estimate headroom in the pilot from three diagnostics: an oracle reference-program executor, an oracle program selector over each method's proposals, and the actual deployable selector. The first identifies harness/coverage ceilings; the second separates proposal quality from selection failures. Oracle scores are diagnostics only and must never enter the deployment result.

If `a` is the fraction of tasks whose required function is in the DSL, `b` the conditional probability of correct interpretation and synthesis, and `c` the probability of correct execution/reporting given a correct program, then overall success is `a*b*c` by the chain rule, not by an independence assumption. If the neural mechanism only improves `b`, its maximum absolute headroom is at most `a*c*(1-b)` when the other terms remain fixed.

Measure `T = T_input + T_encode + Σ(T_core + T_decode + T_sync + T_check) + T_select + T_output`. A reduction in neural arithmetic that is smaller than checker and synchronization costs is not an end-to-end acceleration. Shared parameters save storage, not automatically this sum.

---

## 3. Research findings, prior art and novelty assessment

### 3.1 Corrections and qualifications to the starting idea

The approximate original parameter arithmetic is sound. For two ordinary attention/SwiGLU blocks, hidden width 768, feed-forward width 2,048 and a tied 8,192-token embedding, the dominant matrices contain **20,447,232 parameters**, excluding norms and other small tensors. Raw eight-bit storage is about 20.45 MB. The storage formula is idealized and does not include scales, packing, tokenizer or runtime. This was recalculated here, not inferred from a trained artifact.

The official TRM paper reports a small specialized recursive network and strong puzzle results; it is not a general language-model demonstration. Inspection of its implementation matters: it contains no-gradient outer iterations, detached carries and a training-time halting rule, while evaluation uses the configured maximum steps. A new full-backpropagation compiler using similar weight sharing is **not** a faithful TRM replication. [R03]

Parameter Golf is real and useful, but its central task is byte-normalized language prediction under a 16 MB artifact constraint and a training regime involving eight H100 GPUs. Its competitive entries already combine recurrence, compression and other techniques. The root MLX implementation explicitly is not the state-of-the-art submission. Its int8 serialization/decompression path is not evidence of packed int8 GPU inference. Therefore “recurrent model versus the strongest conventional 16 MB model” is not a clean characterization of that leaderboard. [R16]

The July 2026 Parameter Golf retrospective is observational: a technique's association with good submissions is not its causal effect after controlling all interacting engineering choices. Treat n-gram blending as a useful candidate for a future language-modeling track, not proof that it improves semantic compilation. [R17]

The SmolLM2-135M card confirms 2T pretraining tokens and its reported base-model MMLU-cloze and five-shot GSM8K numbers of 31.5 and 1.4, respectively. These are evidence about that model and evaluation, not a contemporary lower bound or the strongest small-model baseline. [R23]

### 3.2 Closest-alternative matrix

“Local” below means the relevant experiment could be implemented or an inference asset could potentially run locally after compatibility tests; it does not mean it was run here.

| Prior art | Problem, mechanism and assumptions | Evidence / resource conditions | Exact overlap, reuse and limitation |
|---|---|---|---|
| Universal Transformers [R01] | Shared-depth self-attention and optional adaptive computation | Sequence tasks; algorithmic expressivity under stated assumptions | Recurrence and halting are established. Reuse the tied-depth control; universality is not an optimization guarantee |
| Neural GPU [R02] | Gated recurrent spatial computation for algorithm learning | Synthetic algorithm tasks with length generalization | Non-Transformer alternative at the circuit/topology level. Older implementation needs a port; no general language claim |
| TRM [R03] | Repeated refinement of latent/answer states with deep supervision | Approximately 7M parameters, supervised grids; reported ARC results depend on that setting | Small recursive reasoning is known. Reuse state-update and supervision ideas; separately label adaptations and original-recipe diagnostics |
| Recurrent-depth language models / MoR [R04], [R05] | Reuse a core; vary depth, sometimes token-wise routing and caching | Language-model experiments much larger than the proposed compiler; substantial pretraining | Counters novelty of reused neural depth. Reuse training/stability ideas; full original training is nonlocal |
| Tiny Autoregressive Recursive Models; Loop, Think, Generalize [R06], [R07] | Controlled placement of recurrence and compositional tests | Task-dependent results; small or synthetic evaluation populations | Makes recurrence placement, deep supervision and length shift necessary controls. Published gains cannot be imported into this task |
| UPI–TRM [R08] | Checker-feedback learning for tiny recursive policies using policy/value updates | Author poster includes Sudoku; reported 9×9 success is low, not a solved general feedback-learning problem | Feedback plus tiny recurrence is already explicit prior art. Poster inspected; full OpenReview paper inaccessible in this session |
| Neural Programmer-Interpreters [R09] | Shared recurrent interpreter with program memory and supervised traces | Structured algorithmic demonstrations | “Store procedures rather than answers” has close historical precedent; program recognition and invocation remain the hard parts |
| DeepCoder [R10] | Neural predictions guide typed program search | Program synthesis from examples in a restricted DSL | A strong simple hybrid baseline. Implement operator-prior-guided enumeration rather than assuming a large language model is necessary |
| RobustFill [R11] | Recurrent synthesis from examples | String-transformation synthesis; example/noise assumptions differ | Compact neural synthesis is established. Its task and output language differ; do not present our numerical DSL as a reproduction |
| Synthesize–Execute–Debug [R12] | Generate programs, execute tests, repeatedly repair from feedback | Learned discrete synthesis and repair | Closest functional alternative to the proposed controller. Full-history state-reset repair is indispensable, not an optional weak baseline |
| Latent Execution for Neural Program Synthesis [R13] | Learn latent execution information during synthesis | Program-synthesis tasks | Latent computation inside synthesis is not new. Persistent cross-attempt state must be distinguished from this overlap |
| DreamCoder / LILO [R14] | Learn reusable abstractions and libraries; LILO couples language and library learning | Domain-specific program collections; teacher/search costs can be large | Library discovery is a potential later experiment, not a novelty claim. Count learned library bytes and discovery compute |
| PAL [R15] | Model generates programs, interpreter computes answers | Larger pretrained models, program-aided tasks | Beating a larger **tool-free** model would not establish a new architecture advantage. Give larger references the same tools |
| Parameter Golf [R16], [R17] | Optimize code plus compressed weights for next-byte predictive efficiency | 16 MB artifact, accelerator-intensive training; engineering mixtures | Reuse packaging checks and inspect MLX code. Separate LM metric and adaptation rules from compiler task success |
| BitNet / ParetoQ [R19], [R20] | Quantized representations and training/deployment methods | Substantially different training and kernel conditions | Compression is established; 1.58-bit entropy is not automatically a physical GPU format or preserved competence |
| LoopQ / feedback-exposure analysis / LoopQuant [R21], [R22] | Quantization aware of evolving recurrent activations and propagated errors | Recent looped-model work, primarily larger models | Directly invalidates a blanket claim that combining recurrence and quantization is novel. Reuse sensitivity diagnostics before inventing new calibration |
| RETRO [R18] | External retrieval supplements parametric knowledge | Large model and retrieval-corpus experiments | Relevant later to factual knowledge; does not validate a self-contained tiny artifact |
| Ouro / Nanbeige4.2 [R24], [R25] | Released looped language models, including trained reasoning/agent behavior | Billion-scale weights and very large inherited training budgets | Relevant contemporary reference systems. Their existence narrows the claimed novelty; they are not laptop-trainable baselines from scratch |
| Solve the Loop [R26] | Fixed-point/attractor-style computation and implicit differentiation | Different optimization assumptions and model scales | An evidence-triggered architectural alternative, not an initial dependency. Solver benefits and train-time effects must be disentangled |
| Patent disclosures [R36], [R37] | Neural DSL synthesis from examples; multimodal specifications and synthesis | Published technical disclosures, not controlled performance evidence | Prior-art search is wider than paper terminology. No legal status, patentability or freedom-to-operate conclusion is made |

### 3.3 Reconcile apparent disagreements

Puzzle success, text-prediction quality and semantic program synthesis measure different abilities. They use different inputs, supervision, data scales, inference budgets and opportunities to verify answers. There is no valid numerical leaderboard combining them.

Some recurrent work attributes improvements to iterative computation; other analyses emphasize deep supervision, recurrence placement or training regularization. The remedy is not to select the favorable paper: match losses and optimizer updates, include last-loss-only training, compare an untied executed-depth control, and measure what happens when inference depth changes after training. [R03], [R06], [R07], [R26]

Recent larger looped models support parameter efficiency, but their pretraining is not remotely comparable to laptop-generated supervised tasks. Ouro also describes training instabilities and changes in recurrent-step schedules. This is reason to monitor optimization and depth extrapolation, not to assume all additional loops help. [R24]

A recent Apple-Silicon Nanbeige deployment report documents concrete compatibility problems and useful checks, but its general argument that repeating attention necessarily doubles **peak** memory does not follow merely from doubling work: inference intermediates can be released or fused. Treat its observed implementation behavior as a regression target, not a universal law. The paper also contains differing date/repository statements; this plan does not rely on its speed numbers or promise that its patches are available or necessary in today's revision. [R30]

### 3.4 Exact novelty position and discriminating tests

**Already exists:** the broad architecture family and each major resource-redistribution strategy. **Reusable:** tokenizer/runtime infrastructure, exact arithmetic, typed synthesis algorithms, recurrence recipes, public datasets, model cards and public local-model assets. **Substantively different, provisionally:** the particular controlled test of persistent continuous state versus a fully informed discrete repair history at roughly 10–20 MB, with measured Apple-Silicon end-to-end costs and structural holdouts.

That difference may have value because a discrete candidate throws away unexpressed alternatives while a continuous workspace can retain them. It may have no value because a well-trained discrete repair model and search already retain all useful information. The reset/full-history ablation, common-transcript diagnostic and search baselines distinguish these explanations.

This is **not a demonstrated methodological novelty claim**. Until additional evidence supports a new mechanism, classify the project as replication-informed experimental research plus implementation engineering. An unexpected, reproducible resource frontier or transfer effect could be a new empirical finding. A new dataset is a benchmarking contribution only if its validity and difficulty are justified independently of the candidate's results.

The novelty claim would be invalidated by prior work that already establishes the same continuous-state benefit with equivalent information, sufficiently strong discrete alternatives and comparable resource accounting. Search the final selected formulation again before a publication claim. If an equivalent is found, reproduce it and report a replication or extension rather than rename it.

---

## 4. Proposed mechanism and implementation specification

### 4.1 Minimum complete system

Implement only: deterministic input adapter; training-only tokenizer construction; compact encoder; shared recurrent program workspace; typed program decoder; bounded exact interpreter; public-feedback generator; deployable output selection; independent hidden evaluator. No web access or external knowledge database is used during inference.

The interpreter executes a small DSL, never arbitrary Python, shell commands, SQL against user databases or generated package imports. Public model outputs are data. The evaluation process must not execute arbitrary code from either a model or a dataset.

### 4.2 DSL v1: exact semantics

Types are `N` (rational number), `L` (immutable tuple of rational numbers), and `B` (Boolean, distinct from integer). Rational values are reduced numerator/positive-denominator pairs. Parse decimal literals through their strings into exact rationals, never through binary floating point.

A public signature contains at most eight named inputs and at most one input list. Built-in constants are `-1, 0, 1, 2`. Other numeric constants must be present in public bindings or extracted lexically from the public description and assigned a binding index; this extraction is identical for every method and never consults a reference equation. Input lists initially contain at most 32 items. These are disclosed interface bounds, not claims about arbitrary programs.

Programs have 24 instruction rows, each four fields: `(op, arg1, arg2, arg3)`. Each argument is an input reference, built-in constant reference, earlier register reference or `NULL`. Instructions write one register. A `RETURN` row selects a value of the declared output type; all following rows are `PAD`. Thus the maximum is 23 computational instructions plus `RETURN`. Initial train/test lengths stay below this output capacity so length generalization is not blocked by serialization.

| Operators | Signature | Edge cases and conventions |
|---|---|---|
| `ADD, SUB, MUL, DIV, MIN2, MAX2` | `N,N -> N` | `DIV` by zero returns a structured domain error; no infinity/NaN |
| `NEG, ABS` | `N -> N` | Exact rational results |
| `LT, LE, EQ` | `N,N -> B` | Exact comparison; no Boolean-to-number coercion |
| `SUM, MEAN, LENGTH` | `L -> N` | Empty sum is 0; empty mean is a domain error; length is an integer rational |
| `SORT, REV, UNIQUE` | `L -> L` | Ascending numeric sort; reverse preserves all elements; unique preserves first occurrence order |
| `FILTER_GT` | `L,N -> L` | Retain elements strictly greater than the threshold |
| `MAP_ADD, MAP_MUL` | `L,N -> L` | Apply the scalar operation element-wise |
| `TAKE, DROP` | `L,N -> L` | Count must be a nonnegative integer; counts above length are clamped to length |
| `GET` | `L,N -> N` | Index must be an integer with `0 <= i < len`; no negative indexing |
| `SELECT_N, SELECT_L` | `B,N,N -> N`; `B,L,L -> L` | Select second argument when true, third otherwise; earlier rows have already executed |
| `RETURN` | one value -> output | Type must match public signature |

All values have a 1,024-bit numerator/denominator bound. Execution also enforces instruction count and list-length bounds. The generator only admits reference tasks whose specified public and hidden domains are valid under these rules. On external workloads, out-of-bound inputs count as coverage failures rather than disappearing from the denominator. Empty lists, duplicate values, negative numbers, zero denominators, invalid indices and equal comparisons receive dedicated tests.

**Important:** `SELECT` is not lazy branching. A previous division by zero cannot be hidden by selecting the other branch. Do not silently implement Python's lazy conditional semantics instead.

### 4.3 Program decoding and error behavior

Predict logits in parallel for the 96 fields. Decode rows in order with deterministic type/arity/reference masks. At each row, select the highest-scoring legal operator with at least one legal argument completion, then legal arguments in field order. Break exact ties by token ID. Mask unavailable registers and future references. After `RETURN`, emit only `PAD`. If no return has appeared, row 24 is forced to a legal return when one exists; otherwise emit `INVALID_PROGRAM` and count failure. The signature's available inputs make a trivial return possible in most, but not all, cases.

This greedy grammar projection is part of inference, not a proof that the neural model understood syntax. Report raw-field validity as a diagnostic. Strong baselines may use beam search over the same legal language; the practical comparison must include its cost rather than artificially prohibiting it.

Training uses static field-class masks and unmasked-with-respect-to-predicted-types logits for supervised cross-entropy. **Do not compute gold-target cross-entropy after applying legality masks derived from an incorrect predicted prefix:** a correct gold reference can then be masked to negative infinity. Gold-prefix legality masks are permitted as a separately identified training option, consistently applied across methods.

### 4.4 Public execution feedback and information equality

A candidate is executed on the three public examples, or only on query inputs when examples are absent. The compact feedback for each public example has six categorical fields:

1. execution status (`OK`, parse/type/domain/bound error);
2. pass/fail/unknown;
3. first mismatching list position, `LENGTH_MISMATCH`, scalar mismatch or `NONE`;
4. sign of numeric comparison at the mismatch (`LT`, `EQ`, `GT`, `UNKNOWN`);
5. actual list length, or `SCALAR`/`UNKNOWN`;
6. expected list length, or `SCALAR`/`UNKNOWN`.

For Boolean outputs, exact equality determines pass/fail, a mismatch uses the scalar-mismatch position code, and numeric comparison is `UNKNOWN`; Booleans are never coerced to numbers. If execution fails, mismatch/comparison/actual-length fields are `UNKNOWN`, while a public expected-length field may still be supplied. When no expected output is public, pass and comparison fields are `UNKNOWN`. It is forbidden to substitute the evaluator's expected output. All exact outputs are retained in logs; the model-facing API returns this same compact feedback to all methods in the matched experiment. Wider-feedback experiments are separate information conditions.

Every solver gets the same public description, signature, query bindings and examples. For the latent-state comparison, both methods can attend to **the complete sequence of their own previous candidate programs and public feedback**, not just the latest failure. There is no hidden extra memory advantage from withholding history from the reset baseline. Histories differ naturally when actions differ; a common-transcript diagnostic in §8 isolates that consequence.

### 4.5 Compact architecture

Use a conventional, auditable encoder–workspace design rather than inheriting a CUDA-specific training stack.

| Component | Small initial configuration | Evidence-triggered wider configuration |
|---|---|---|
| Total tokenizer vocabulary, including reserved DSL symbols | 4,096; byte fallback; reversible normalization | Same |
| Hidden width / SwiGLU intermediate width | 256 / 768 | 384 / 1,024 |
| Heads / head dimension | 4 / 64 | 6 / 64 |
| Input encoder | Two independent bidirectional attention blocks | Same structure |
| Shared recurrent core | Two blocks, each workspace self-attention, input/history cross-attention, SwiGLU | Same structure |
| Workspace | 96 fixed field positions | Same |
| Input length | Buckets 256, 512, 1,024, 2,048; no silent truncation | Same unless an amended coverage experiment requires more |
| Position representation | Fixed sinusoidal positions; separate fixed row/field/history-step coordinates | Same |
| Embedding/output projection | One shared vocabulary matrix; output selects reserved DSL rows | Same |
| Training loops | Uniform draw from `{1,2,4,8}` per effective minibatch after the one-loop debug phase | Same initially |
| Evaluation loops | `{1,2,4,8,16}`; 16 is extrapolation | Same |
| Precision | FP32 parameter/optimizer reference; FP16 compute only after parity tests | Same |

The matrix parameter estimate is

\[
P_{mat}=Vd+2(4d^2+3df)+2(8d^2+3df)=Vd+24d^2+12df.
\]

With trainable RMSNorm scales, one learned initial workspace vector and three segment embeddings, the specified small-tensor convention adds approximately `17d` parameters. This gives **4,985,088** parameters / **9,970,176 FP16 bytes** for the small configuration and **9,836,928** / **19,673,856 bytes** for the wider one. These are architecture calculations, not measured model files. The implemented parameter tree and serialized manifest are authoritative; record every difference.

Use bias-free projections, pre-RMSNorm residual sublayers, epsilon `1e-5`, FP32 norm/softmax reductions, and no dropout in the initial mechanism experiment. Two encoder blocks have two norm scales each; each core block has three; encoder final, core final and entry-mix norms provide the remaining three. Fixed positions add no learned tensors. All segment embeddings and the learned initial vector must be counted.

### 4.6 State update

Let `E = Encoder(x)` be immutable input memory, `S` fixed workspace positions plus a learned initial vector, `p_0` an all-`MASK` program, and `τ_<t` the candidate/feedback history. History tokens use the same token embeddings with fixed field and step coordinates; they are not hidden gold traces.

\[
Z_t=\operatorname{concat}(E,\operatorname{EmbedHistory}(\tau_{<t})),
\]
\[
U_t=\operatorname{RMSNorm}(\widetilde H_{t-1}+\operatorname{EmbedProgram}(p_{t-1})),
\qquad H_t=F_\theta(U_t,Z_t),
\]
\[
p_t=\operatorname{GrammarDecode}(\operatorname{Head}(H_t)),\qquad
f_t=\operatorname{CheckPublic}(p_t,x),\quad \tau_t=(p_t,f_t).
\]

For the persistent method, `H_0=S` and `H_tilde_(t-1)=H_(t-1)`. For the matched reset method, `H_tilde_(t-1)=S` at every cycle. All other interfaces and parameters are the same. Train the reset method with resets; resetting a trained persistent model only at test time is an out-of-distribution diagnostic, not the main baseline.

At most 16 candidate histories contain `96+18` fields each. Together with a 2,048-token input, cross-attention memory is below approximately 3,872 positions, excluding a few separators. Declare and test the exact serialized limit. Beyond the configured limit, return an explicit coverage/budget failure; never silently discard old history for just one method.

Input cross-attention keys/values can be reused after projection; newly appended history projections can be appended. **Workspace self-attention keys/values must be recomputed each cycle because all workspace states change.** No cross-task latent state survives between independent requests. Incorrect stale-cache reuse can create an apparent speedup while changing the model.

### 4.7 Training: local supervision, full state gradients, discrete feedback

Generate canonical reference programs and descriptions from training-only grammars. Start with one-loop supervised fitting solely as an implementation check. The main training loss is

\[
\mathcal L=\frac1R\sum_{t=1}^{R}\frac1{N_{active}}
\sum_{j\in active}\operatorname{CE}(\ell_{t,j},p^*_j).
\]

Include the return row and ignore padding. Each task's loss is normalized by active fields, then averaged across tasks. Compute **one optimizer update per effective minibatch**, not one per loop. All matched recurrent methods use the same sampled loop counts and deep-supervision schedule. Canonical-program supervision can penalize an alternative correct program; log semantic success separately and do not equate this loss with semantics.

Execution and grammar decisions are discrete and receive no gradient. The latent trajectory remains differentiable through all sampled cycles. Do not quietly substitute detached carries, truncated backpropagation or TRM's no-gradient outer updates when memory becomes inconvenient. They are distinct training treatments.

**Recommended implementation for host feedback: two-pass rollout/replay.** First collect discrete actions and feedback with fixed weights. Then recompute the continuous trajectory with those discrete observations held constant and backpropagate. With deterministic forwards, unchanged weights and no dropout, this implements the intended gradient through continuous states while stopping gradients through decisions. It avoids requiring a Python rational interpreter to run inside a compiled autodiff graph. Count both passes in training cost.

```python
# Algorithmic pseudocode; these are project interfaces, not package APIs.
def train_minibatch(model, batch, optimizer, R, state_mode):
    # No parameter update occurs between collection and replay.
    tape = collect_public_rollout(model, batch.public, R, state_mode)
    # tape contains only candidate tokens and public feedback, never gold programs.
    def objective(params):
        E = encode(params, batch.public)
        H = initial_workspace(params, batch.size)
        total = 0.0
        for t in range(R):
            base = H if state_mode == "persistent" else initial_workspace(params, batch.size)
            H = core_step(params, base, E, tape.prefix(t))
            logits = field_head(params, H)
            total += supervised_field_loss(logits, batch.gold_program) / R
        return total
    loss, grads = value_and_grad(objective, model.parameters())
    grads = global_norm_clip(grads, 1.0)
    optimizer.update_once(model, grads)
    materialize_parameters_optimizer_and_loss()
    return loss, tape.summary()
```

The replay function must use the recorded previous program injection, not recompute a different discrete candidate during backpropagation. Test collection/replay equality before enabling compilation or activation checkpointing. Checkpoint only deterministic tensor subfunctions; do not repeat interpreter side effects during backward recomputation.

Use AdamW initially with betas `(0.9,0.95)`, epsilon `1e-8`, gradient clipping 1.0 and weight decay candidates `{0,0.01,0.1}`. Pilot learning rates `{1e-4,3e-4,1e-3}` with warmup and a common schedule in optimizer updates. These are starting search points, not privileged settings: expand a range when the best point is on a boundary or optimization diagnostics demand it. Tune each serious baseline independently under an equal search-resource ledger. Preserve stable checkpoints and learning curves.

### 4.8 Output selection and stopping

The primary causal test uses the **final candidate at a fixed preregistered loop count**, so a hidden “best iterate” selector cannot manufacture an effect. Also report each iterate's performance and a public-only selection wrapper.

For practical frontier comparisons, allow each method its strongest validation-selected policy under the common information and deadline constraints: final candidate, first public-test-passing candidate, or best public score with model-confidence tie-breaking. Give the candidate the same opportunity. Freeze these choices before confirmation, log every proposal and include selection cost. Never select using hidden tests. In no-example tasks, a successful execution does not provide a semantic pass signal.

Learned halting is deferred. Its later objective must trade observed task success against measured cost, with calibration on development data. Report false-confidence and selective-risk curves; a low residual or a high learned stop score is not a correctness proof.

---
## 5. Baselines, datasets, metrics and confirmatory protocol

### 5.1 Baseline hierarchy and reproduction rules

Do not claim competitiveness from a single weak tiny Transformer. The following methods have different scientific roles; preserve those labels.

| ID | Method and implementation | Matching / role | Required reproduction or acceptance check |
|---|---|---|---|
| B0 | Oracle reference-program executor | Harness ceiling, never a competitor | Exactly correct on all in-domain generated tasks; deliberate corruptions must fail |
| B1 | Rule parser plus template/program retrieval | Strong simple practical alternative | Rules learned/written only from training grammar; retrieval index stores only training programs; count rule/index bytes |
| B2 | Typed bottom-up enumerator with observational-equivalence pruning | Strong non-neural synthesis from visible examples | Exhaustively matches a brute-force enumerator on small domains; same visible tests, same deadline; no hidden-case pruning |
| B3 | DeepCoder-style operator predictor guiding B2 | Strong compact neural–symbolic alternative [R10] | Predictor estimates operator presence from public input; order search by prior without deleting low-prior operations; count network, index and search memory |
| B4 | Conventional autoregressive encoder–decoder with grammar-constrained beam/sample search | Strong parameter/payload-matched neural baseline | Same tokenizer, data and public checker; independently tune depth/width, beam and decoding policy; demonstrate fit and validation convergence |
| B5 | Shared-core state-reset iterative repair, complete candidate/feedback history | **Primary H1 comparator** and close SED-style alternative [R12] | Same initial parameter distribution, encoder, core, supervised losses, minibatch sequence, update count and loop schedule; trained with resets |
| B6 | Persistent shared-core without semantic feedback | Ordinary recurrence control | Same architecture; feedback is unknown/structural only; no gold test signals |
| B7 | Attention-based GRU encoder/decoder | Non-Transformer local alternative | Use the same tokenization and DSL; tune hidden width/layers to the payload limit; retain if competitive on development data |
| D1 | Untied core with identical executed depth | Capacity/weight-sharing diagnostic, not same-size competitor | Clone shared weights: initial forward outputs match; parameter increase disclosed; repeat at matched training compute |
| D2 | Same-size narrower/deeper feed-forward network; final-loss-only recurrent model | Depth and supervision controls | Match actual bytes and train exposure in one panel, train work in another; no claim that every resource is simultaneously equal |
| L1–L4 | Contemporary larger pretrained local models, same task/tool access | Practical references, not from-scratch causal controls | Correct tokenizer/template, validated backend, preserved loop count, disclosed pretrained assets, deadline and thought-token accounting |

For B1, use a transparent token/BM25-style or sparse TF–IDF retriever over canonical training descriptions, with candidate bindings substituted only by deterministic public-name/number matching. A failed substitution is failure, not a request to an uncounted large model. Run a zero-neural rule-only variant. Training-case retrieval is external memory: report the full index and source payload, and compare both within-cap and unconstrained-index versions separately.

For B2, enumerate well-typed expression DAGs by increasing structural cost. Memoize each candidate's vector of outputs on **public** examples and keep the cheapest representative per type and output vector. This pruning is sound for finding some public-consistent expression in the compositional DSL, but not for discovering the intended hidden semantics: observationally identical candidates can differ elsewhere. In language-only tasks, there are no expected outputs to synthesize against; enumeration must be ranked by a language-derived prior, and cannot claim oracle filtering. Save explored nodes, equivalence collisions, depth reached and timeout status. Avoid exponential precomputation disguised as free inference.

For B3, train a small pooled token encoder or GRU plus sigmoid operator-presence head on generated training programs. Tune its capacity and prior temperature on development data. Use prior-weighted enumeration with a queue that remains complete as budget grows. A reference operator set extracted from the hidden program is an oracle diagnostic only.

For B4, begin with the same two-block encoder and two-block decoder, causal autoregressive decoding over active DSL fields, and tied embedding. Tune a deeper/narrower configuration under the **actual** payload cap. Evaluate greedy, beam widths 2/4/8 and stochastic proposals as pilot starting points; expand a useful frontier rather than imposing those as permanent limits. Teacher forcing, grammar constraints and public-example reranking are not withheld from this baseline.

B7 screens whether a cheaper recurrent sequence model already captures the benefit. It is not mandatory to train every historical algorithm to convergence; its exclusion from confirmation requires recorded development evidence, not architectural preference. A Neural-GPU-like port is warranted only if a separate pure algorithm-learning diagnostic identifies spatial recurrence as the relevant unresolved mechanism.

### 5.2 Contemporary pretrained reference assets

Use exact upstream model identifiers, then resolve and lock a commit SHA and file hashes at W00/W06. Model names are not immutable versions. Search for a material upstream replacement again at the freeze date, since this plan is a dated snapshot.

| Reference | Why include | Local path and caveat |
|---|---|---|
| `Nanbeige/Nanbeige4.2-3B` | Particularly close: a released looped agentic model, Apache-2.0 card [R25] | Its “3B” excludes embeddings; card lists about 4B total. Inspect pinned custom code and preserve loop configuration. Use a verified Apple-compatible upstream/fork path; do not copy CUDA quickstarts. Deployment bugs are a known regression risk, not proof current code is broken [R30] |
| `Qwen/Qwen3.5-4B` | Stronger contemporary general/reference baseline than historical 135M models [R27] | Official MLX-LM source includes its hybrid text architecture. Test recurrent-state cache correctness and text-only conversion; count actual tensors, not nameplate size |
| `google/gemma-4-E2B-it` | Another independent contemporary small local reference [R28] | Card reports 2.3B effective but 5.1B including embeddings. MLX-LM has a Gemma4 implementation; correct model conversion remains a gate. Card license is Apache-2.0 |
| `ByteDance/Ouro-2.6B` and a pinned officially documented thinking variant if selected | Direct looped-language reference [R24] | Inspect upstream license, supported inference path and trained depth. Full pretraining is not local. Do not undertake a large unvalidated runtime port merely to claim a comparison |

Screen these on public development tasks after metadata and compatibility checks. Download and load one at a time. Select the strongest successfully validated model for a full practical comparison, retaining screening results. The larger-reference ladder may add `Qwen/Qwen3.5-9B` or `google/gemma-4-E4B-it` when measured memory, storage and expected information gain justify it. The lower-memory branch still has a serious multi-gigabyte reference; it does not assume maximum M3 Max RAM.

`LiquidAI/LFM2-1.2B-Tool` is an optional tool-specialist reference. Its card uses a custom `lfm1.0` license, unlike an unconditional Apache/MIT assumption. Inspect actual terms before use or redistribution; exclusion for unresolved terms is explicit. An official MLX-LM LFM2 implementation exists, but no compatibility or performance test was executed here. [R29], [R31]

All larger references receive the same request, signature, public examples and DSL tool. Use model-native chat/tool templates and validate a few-shot DSL tutorial on development tasks. Also report a native answer-only or natural tool-call condition when the common DSL interface materially disadvantages a model; it is a separate interface comparison. Do not allow larger models to read hidden tests or use extra databases, and do not deny them public feedback allowed to the tiny model.

Follow each model's documented thinking and sampling conventions as starting points, then tune on development data. Count thinking tokens, failed parses and retries in the deadline. A forced short greedy answer from a reasoning model is not its strongest attainable performance. Report both the common interactive deadline and a separately measured relaxed-budget frontier so a tiny system's speed does not masquerade as a universal quality advantage.

If a stronger reference cannot run faithfully, label it nonlocal/unvalidated and retain its published results only as context. Do not compare its published benchmark number numerically with the project's different test. A full Parameter Golf, MoR, RETRO, Ouro or Nanbeige training reproduction is outside the core local workload.

### 5.3 Generated dataset: construction and provenance

Create `dsl_tasks_v1`, released with project-authored generator code and a permissive license chosen explicitly in the repository. Each task has a typed canonical program, a public signature, descriptions, public examples, query bindings and hidden tests. Use independent RNG streams for program structure, values, surface realization and hidden cases. Seeds and generator revision are sufficient to regenerate each partition; preserve sampled manifests as well.

**Generator algorithm:** choose a target output type and a typed operator DAG; sample public input types; ensure every computed row influences the return; normalize register names; reject dangling references, semantically vacuous constant-only tasks and domain-invalid reference instances. Detect obvious identities and redundant rows with a fixed training-independent rewrite list. Do not use candidate model behavior to choose tasks.

Generate six declared mechanism strata: scalar arithmetic/comparisons; reductions and aggregates; list ordering/filtering; element-wise transformations; indexing/slicing; conditional selection. Classify a program by its final non-return operator group. Fix equal stratum weights for the primary macro average; also report unweighted micro accuracy and every stratum separately. These are fixed task strata, not six independent samples of all possible reasoning domains.

Descriptions come from a compositional English grammar that refers to named inputs and intermediate results. Examples of semantic templates are “keep the values greater than {threshold},” “add {amount} to each remaining value,” “return the total,” and “when {condition}, use {left}; otherwise use {right}.” Include both procedural and goal-oriented phrasing in development, with explicit entity/binding references and controlled pronouns. The procedural language track is easier and must be labeled as such.

For surface shift, reserve entire realization grammars, not just random synonyms: reordered conditions, relative clauses, passive constructions, irrelevant factual distractors and changed variable names. Include counterfactual pairs where a single operator or comparison word changes the answer. New wording written by the same generator is still synthetic evidence, not independent human language understanding.

No paid teacher is necessary for the core. A local larger model may later generate training-only paraphrases, but this is an amendment with its own filtering cost and contamination risks; compare both tiny architectures on exactly the same additions and keep human-written external tests untouched.

### 5.4 Split rules and hidden cases

Store structural hashes before rendering text. A structural key must include operator DAG, types and argument dependency positions, normalize register/input names, and replace varying literals with role placeholders while retaining distinguished built-in constants and semantically relevant signs. Canonicalize commutative operand order without equating noncommutative programs. Store both the complete AST hash and a coarser skeleton hash. Numeric variations and paraphrases of the same skeleton form one dependence cluster.

Use separate partitions:

| Partition | Purpose | Construction |
|---|---|---|
| Train | Fit all locally trained methods/tokenizer | Initially 1–6 computational instructions; all primitive operators covered; lengths and strata balanced in sampling |
| Development | Hyperparameters, resource calibration, baselines | Separate structural keys and value/surface seeds; includes representative shifted development motifs that will not be reused for confirmation |
| Protected structural confirmation | Primary population | Unseen type-compatible operator/dependency motifs at lengths 1–6, with every primitive seen in training; no skeleton overlap |
| Length shift | Test compositional depth | 7–12 computational instructions, below 23-row capacity; separate structure/value seeds |
| Surface shift | Test language robustness | Familiar operation support but an entirely reserved realization grammar and new skeletons |
| Joint shift | Hard boundary condition | Reserved surface grammar plus longer and structurally new programs |
| Numeric/list-size shift | Test algorithm reuse | Values and list sizes outside training ranges, within disclosed execution bounds |

Construct the holdout motif assignment before training from the typed grammar using a deterministic hash partition of compatible operator–argument-role pairs. Repair the partition, **before generating data**, only to ensure each primitive and input/output type remains represented in training and test generation is possible. Record repairs. Then verify no protected motif appears in a training DAG. Do not “repair” a split because a model scores poorly. Fix split proportions and motif lists in the dataset manifest; their purpose is structural separation, not tuning the candidate's advantage.

Training starts with integer numerators in `[-100,100]`, denominators `1..10`, and input-list lengths `0..8`, subject to valid domains. Shift sets extend magnitudes and list length to the declared bounds. Report rejection rates and the resulting, not merely intended, distribution; high rejection can bias difficult operations out of the test.

Each generated task initially has three randomly drawn valid public examples and 32 independent hidden cases plus query evaluation. Increase the hidden suite **during harness development**, using an independent bank of mutated wrong programs, until undetected wrong-program rates reach the precision required for the proposed semantic claim. Include adversarial edge cases and, for small domains, exhaustive checks. Freeze this procedure before protected evaluation. More tests do not prove equivalence on an infinite domain.

Select public examples without hidden-test knowledge. Measure ambiguity with bounded enumeration: how often do multiple public-consistent programs disagree on independent evaluator cases? This is a property of the information supplied, not necessarily a model defect. Compare empty-example and three-example conditions separately; do not average them into an unexplained score.

### 5.5 External transfer: mandatory arithmetic, conditional list tasks

**SVAMP is the mandatory realistic-language transfer test.** Use the official `SVAMP.json`, freeze repository revision/hash and evaluate the whole 1,000-problem challenge set without training or prompt selection on its answers. The official repository and its MIT license were inspected. The source distinguishes SVAMP evaluation from other training datasets; do not repurpose the challenge set as a local training corpus. [R33]

The model receives only `Body` plus `Question`, public numeric bindings extracted by the shared lexical adapter, and a numeric output signature. `Equation` and `Answer` remain in the evaluator process. Decimal parsing is exact; numeric answer equivalence uses exact rational evaluation where possible and a separately specified serialization tolerance only for a reference value represented by an imprecise decimal. The default tolerance is zero for finite exact decimal references; any exception list must be documented before model outputs are inspected. Report full-set accuracy, number-extraction coverage, DSL coverage, parse failures and all exclusions. Unsupported cases remain failures in the full-set metric.

SVAMP provides single-instance word problems, not public input/output examples for a reusable function. Therefore it tests the **no-semantic-feedback** mode. Do not fabricate checker feedback from the reference equation. Report answer correctness rather than universal program equivalence on this dataset. Public benchmark contamination in pretrained reference models is possible and must be acknowledged; newly generated structural tests provide a separate, less contamination-prone axis.

**MBPP is a conditional second transfer workload.** Official material supplies English programming problems, tests and split IDs; the dataset card lists CC-BY-4.0. Use the sanitized subset with inherited splits. Test IDs remain in `11..510`; IDs `1..10` are prompting examples, `511..600` validation and `601..974` training in the original grouping. [R34]

Before any candidate outputs, build and freeze a coverage manifest of test tasks expressible in DSL v1 with its one-list/input and bounded scalar semantics. Base selection on a written semantic rubric, not convenience or benchmark scores. Implement trusted DSL reference translations for those tasks and validate them against the original tests and generated edge cases. Do not execute arbitrary dataset Python; manually/structurally translate admitted pure operations. When reliable reference translation is unavailable, exclude that task with a reason, but count it as unsupported in the whole-test coverage-adjusted metric.

The conditional MBPP score is an **adapted restricted-subset** result, never “MBPP accuracy” without qualification. Report the eligible denominator, all rejected IDs/reasons and a whole-test metric treating unsupported tasks as failures. Here “whole-test” means every test-split task present in the frozen sanitized dataset; additionally report its coverage of the original 500 test IDs, without implying that removed tasks were evaluated. Keep original test assertions hidden in the no-example condition. A separate example-conditioned track may reveal a fixed subset, but it needs newly generated private tests and is no longer the standard benchmark protocol. If too few eligible tasks exist, this arm is inconclusive; do not expand the DSL after seeing which candidate fails.

### 5.6 Primary estimands and guardrails

**Primary causal estimand:** average paired difference in strict finite-test semantic success between the small persistent model and B5, at eight cycles, final candidate, three visible examples, on the protected structural population. Both are trained with the same randomized `{1,2,4,8}` schedule and the same number of optimizer updates and task exposures. Eight is the maximum trained depth, not a test-selected best point. Report measured training and inference work; matched regimen is not a claim of identical seconds or FLOPs.

Average first over task instances within structural clusters, then over clusters within each of the six strata, then equally over strata, and finally over paired training seeds. Preserve those weights in resampling. State the population as this generator's structural holdout, not arbitrary real-world requests.

**Practical estimand:** at a common, preregistered wall-clock deadline `T*` and payload limit, compare the selected candidate policy with the strongest compact development baseline `B*`, selected from B1–B7 without viewing protected outcomes. Set `T*` from the development p95 end-to-end time of the eight-cycle B5 reference on the declared deployment workload, before selecting a favorable candidate frontier point. This anchors a reproducible envelope rather than inventing a user utility function. Also show complete measured frontiers around that envelope and relaxed budgets for larger models. Do not claim a generally optimal latency preference.

A request succeeds under a deadline only if the **submitted** answer is correct and completed by the deadline. Enforce elapsed time at each model/tool boundary; unfinished work and timeouts count as failure. Do not let an earlier hidden-correct proposal rescue an incorrect final submission.

Secondary metrics: public versus hidden success, final-iterate versus public-selected success, per-family regressions, no-example accuracy, SVAMP accuracy, coverage, invalid programs, oracle proposal coverage, calls/nodes/tokens, p50/p95 latency, throughput at fixed batch, cold start, peak allocations, all three footprint tiers, training wall time/operations, and quantization loss. Report absolute percentage-point differences and relative changes only when denominators are nonzero and meaningful. Do not rename a quality–latency trade-off an unqualified improvement.

### 5.7 Exploratory development versus confirmation

Development can change architecture, data composition, training schedules and selection policies. Its results are exploratory. W06 writes an immutable `confirmatory.yaml` containing hypotheses, primary contrast, populations, hashes, method revisions, trained configurations, sample/seed counts, deadline, selection policy, statistical procedure and stopping rules. Hash it before protected tasks are opened.

The protected evaluator can access labels; solvers and training code cannot. Merely hiding a filename is not a security guarantee: enforce process/interface separation and use an explicit embargo on reading test results during tuning. Store the generating key under the evaluation manifest and release it after freeze/run completion so results remain reproducible.

If a protected result causes a method or claim to change, mark that result exploratory, record an amendment and use newly generated independent confirmation data. SVAMP and MBPP cannot be made fresh by changing a random seed; a revised method's re-evaluation on them is adaptive reuse and must be labeled accordingly.

### 5.8 Replication and uncertainty

The independent data unit is the canonical structural cluster, not each paraphrase, numeric variation, test case or repeated timing. Training randomness is a second crossed source of variation. Pair methods on the same generator sample order, loop draws and initialization seed when architectures permit, but train each method independently.

Run a small multi-seed pilot on development-only data to estimate task discordance and between-training-seed variability. Three pilot seeds are a starting variance diagnostic, not sufficient evidence of stable generalization. Choose the final number of training seeds and clusters from a written precision/power calculation before confirmation. If seed variability dominates, more test cases cannot solve the uncertainty; train more independent seeds or report conditional-on-trained-model results with the narrower claim.

For a simple paired binary comparison with discordance probability `q`, the variance of a task-level difference is `q - delta^2`. A rough normal-approximation planning formula is

\[
n\approx (z_{1-\alpha/2}+z_{1-\beta})^2 q/\delta_{detect}^2.
\]

For illustration only, `q=0.20`, a detectable difference of 0.02, two-sided 5% error and 80% power give approximately **3,925 independent tasks before clustering**. This number was calculated here. A two-point improvement is **not** a required success threshold. Use pilot cluster/seed resampling simulations to plan the real design, and publish its smallest detectable effects. Detecting a small effect and finding it practically worthwhile are separate questions.

Use a crossed bootstrap for the principal uncertainty interval: resample paired training seeds globally and structural clusters within each fixed stratum, preserving all observations and methods inside a sampled unit. Compute the weighted estimand in each replicate. Check bootstrap behavior on simulated correlated data and against an exact paired test in the single-trained-model, unclustered special case. Report seed-wise effects as well as pooled uncertainty. With too few seeds for credible population inference, state that limit rather than hiding it behind many task resamples.

Use one protected fixed-sample primary comparison. For the primary and the named practical confirmatory claim, apply a preregistered Holm correction across the two tests/interval decisions, or a fixed-sequence gate where the practical claim is tested only after H1 passes; choose one in W06, not after results. This plan recommends the **fixed-sequence gate**. All other loop counts, feedback conditions, quantization choices and subgroup searches are descriptive/exploratory unless allocated a separate multiplicity plan and fresh confirmation.

A confidence interval crossing zero is inconclusive about sign. A tightly estimated interval near zero can establish a lack of meaningful local headroom relative to an application-specific value threshold chosen independently of protected results. A negative effect is evidence against the tested configuration and population, not every conceivable recurrent architecture. Never keep adding seeds until the result becomes significant.

### 5.9 Fair tuning and resource panels

Keep a common development ledger: every tried configuration, cost, success and failure. Give serious baselines comparable search effort and the same opportunities for grammar constraints, checkpointing, batching, quantization and public-only selection. An unsuccessful tuning run still consumes the budget and stays in the record.

Produce three distinct comparison panels:

1. **Mechanism panel:** same architecture and training examples/updates/loop schedule, only state persistence changed. Record any actual work difference.
2. **Training-budget panel:** equal measured project-specific training work, allowing faster methods more updates. Separately show equal-data exposure; do not pretend both are identical.
3. **Deployment panel:** actual serialized size, deadline, memory and task success, with each method's validated best configuration. Pretrained references have inherited assets explicitly separated.

Match FLOPs with an auditable operator counter where reliable, but retain wall time because kernels and host synchronization differ. Do not derive exact training FLOPs from `6 * parameters * tokens` for this encoder/recurrent/history architecture; encoder work, repeated core work, attention and replay require separate accounting.

### 5.10 Timing protocol

Benchmark plugged in, with recorded power mode, background load, thermal state and macOS version. Do not disable safety mechanisms, raise global wired-memory limits or require administrative changes. Randomize method order across timing blocks. Warm each shape/loop/precision path until compilation and lazy initialization have completed; preserve cold-start timings separately.

Synchronize before the start and after the final included operation. In MLX, explicitly materialize lazy results and use documented synchronization; in MPS use `torch.mps.synchronize()`. Host-to-device work, JSON handling, public execution and candidate selection are inside the end-to-end boundary. Report accelerator-only submeasurements separately. [R31], [R32]

Use multiple complete task blocks to estimate variability, with paired workloads and both warm-cache and cold-load measurements. Repeated timings are not extra independent accuracy samples. Report measured repetition counts and confidence intervals; use pilot variability to determine repeats. Do not discard slow blocks without a predeclared environmental failure rule. A speedup that depends on excluding time spent in the interpreter is not the claimed system speedup.

---

## 6. M3 Max feasibility and hardware-adaptive execution

### 6.1 Hardware discovery before downloads or training

These are existing macOS/system commands, not project commands:

```bash
uname -m
sw_vers
sysctl -n hw.model
sysctl -n hw.memsize
sysctl -n hw.ncpu
df -h .
system_profiler SPDisplaysDataType
python3 --version
```

Record exact bytes for RAM and free disk using programmatic APIs in `hardware.json`, not only rounded command output. Record the active Python architecture and executable to catch Rosetta/x86 environments. Confirm an M3 Max and working Metal device. Available RAM is not the advertised maximum configuration; measure normal OS/process use and memory pressure before setting a limit.

The official MLX installation documentation inspected requires native Apple Silicon, macOS 14 or newer and Python 3.10 or newer. Resolve a compatible stable environment on the actual machine, then freeze package versions and hashes. This is compatibility information, not proof that the proposed training graph runs. [R31]

### 6.2 Software stack and dependency strategy

**Primary:** native ARM Python, MLX for the custom neural models, NumPy/standard-library exact arithmetic for reference tests and DSL execution, SentencePiece or an equivalently verified byte-fallback tokenizer, safetensors-compatible tensor storage, YAML/JSON configuration, pytest-style tests, and an analysis stack based on NumPy/SciPy. Use MLX-LM in a separate optional environment for larger reference inference.

**Faithful fallback:** PyTorch with MPS for the same equations after CPU-versus-MPS checks. A CPU FP32/FP64 reference remains available for correctness and tiny debugging. If an essential operation is unsupported or numerically different, use an equivalent unfused tensor implementation and apply it to both compared methods; log the change and remeasure costs. A CUDA/Triton/FlashAttention dependency is not a valid core prerequisite.

Do not hard-code invented versions. Create `requirements.in` with direct dependencies, resolve a native environment, record exact versions in a lock plus `pip freeze`, and archive upstream SHAs. Maintain separate locks when a pretrained model's custom code requires a conflicting Transformers version. A pip freeze file alone is not a complete cross-platform reproducibility guarantee; preserve wheel/source hashes, Python/macOS versions and model code.

An existing generic environment bootstrap, after selecting the actual native Python, is:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Install the explicitly reviewed requirements.in, then lock exact resolutions.
python -m pip install -r requirements.in
python -m pip freeze > requirements.resolved.txt
```

The agent must create `requirements.in`; it is not an available file yet. Adding a new dependency requires a license and compatibility entry. Never run unreviewed `trust_remote_code=True` simply because a model card suggests it; download/pin/inspect custom code first and restrict it to the reference-model environment without secrets.

Verified MLX capabilities relevant to the plan include checkpoint recomputation, quantization APIs and quantized linear layers. Crucially, the inspected `QuantizedLinear` documentation says its parameters are frozen and excluded from gradients. Do not assume native packed low-bit training. For later quantization-aware training, use an explicitly specified floating-point master-weight/fake-quantization path and test it separately. [R31]

### 6.3 Critical local smoke tests

Before a substantial training run, test forward/backward for embedding lookup and tied output use, masked self/cross-attention, RMSNorm, SwiGLU, recurrent weight reuse, dynamic history lengths, gradient accumulation, clipping, optimizer serialization, lazy graph materialization, activation checkpointing and host-feedback rollout/replay. Test batches of one and more than one; variable program lengths; every loop depth; and all-padding tails.

Compare a deterministic toy network to CPU FP64, then compare proposed FP32 and FP16 compute on a representative minibatch. Inspect per-layer relative and absolute errors, finite gradients, token argmax agreement away from ties and optimizer-step agreement. Set numerical tolerances from well-conditioned toy references before evaluating model quality; do not loosen them to make an unexplained bug pass. Floating-point equivalence need not be bitwise across backends, but any systematic semantic discrepancy must be resolved.

For larger references, additionally compare cached versus uncached short-sequence logits, chunked versus full prefill, the trained number of loop applications, tokenizer/chat-template round trips and deterministic repeated requests. Qwen3.5's source uses hybrid recurrent/full-attention states; a generic KV cache is not automatically correct. Nanbeige and Ouro require particular care around reused layers. [R24], [R25], [R27], [R30]

### 6.4 Memory model

For `P` FP32 parameters with FP32 gradients and Adam first/second moments, a baseline persistent training allocation is approximately `16P` bytes. Optional compute casts, master copies, exponential moving averages and allocator buffers add to this. At the calculated small and wider sizes, that base is approximately **79.8 MB** and **157.4 MB**, respectively. These are not peak training-memory estimates.

Peak memory also includes

\[
M_{peak}\approx M_{params+opt}+M_{encoder\ activations}+M_{recurrent\ activations}
+M_{history}+M_{logits}+M_{temporaries}+M_{runtime}+M_{data}.
\]

Material terms to instrument:

- Encoder hidden/MLP activations scale roughly with `B*L*(d+f)` per layer; explicitly materialized attention scores scale with `B*h*L^2`.
- A workspace-core pass uses self-attention `B*h*M^2`, cross-attention `B*h*M*(L+history_t)`, and feed-forward activations `B*M*f`, with `M=96`.
- Full backpropagation can retain per-loop activations; history length grows with the loop count, so cross-attention work and naive saved activations can grow faster than linearly over a rollout. Activation checkpointing trades extra compute for smaller retention; measure both.
- Output logits need only reserved DSL vocabulary rows, not all 4,096 tokens at all 96 positions. Verify that implementation choice in the profiler.
- Interpreter workers, tokenizer batches, stored tapes and unmaterialized MLX graphs can dominate these small weights. Stream data and bound queues.

For orientation, at `B=1`, `h=4`, `L=512`, one explicitly stored FP32 encoder attention-score tensor is about 4.2 MB; at `L=2048`, about 67.1 MB. Fused implementations may avoid this tensor. A last-loop cross-attention score tensor with `M=96` and memory length near 3,872 is about 5.95 MB in FP32. These are individual-tensor calculations, not measured peaks or additive guarantees.

MLX device allocations, process resident memory and macOS physical memory use overlap on unified memory. Report them as different views; do not sum them as if they were independent dedicated RAM pools. Include swap/memory-pressure observations. Do not call a swapping run representative interactive performance.

### 6.5 Calibration and configuration rules

Begin with the small configuration, batch one, 256-token bucket, one loop and FP32. Then test representative and maximum lengths, eight loops and the 16-loop inference path. Materialize every step. Record peak active/cached allocations and step time for a stable window, including collection/replay and interpreter work.

Determine a usable memory envelope from actual available RAM minus observed OS/background peaks and a reserve for input spikes/checkpoint writes. Start conservatively below that envelope and increase microbatch only when repeated peak measurements and memory pressure permit. Do not set the reserve from a presumed 64/96/128 GB configuration. Estimate uncertainty from multiple measurement windows and keep headroom above the observed worst case.

When memory is tight, use this order: eliminate accidental retained graphs and unbounded queues; reduce microbatch and accumulate gradients to preserve effective batch; enable exact activation checkpointing; bucket lengths without truncation; serialize model evaluations so only one model is resident. Only then consider a smaller model or altered context length as a documented new experimental condition, not a silent replacement.

The wider configuration is permitted when the small model demonstrably underfits, optimization is sound, the data support a capacity test, and pilot measurements show a useful next scale within the local envelope. If a larger baseline cannot coexist in memory with training, unload the training process first. Disk capacity must cover source plus converted reference weights and temporary files, not just final quantized payloads.

### 6.6 Workload estimates and how to replace them with measurements

| Stage | Resource model / illustrative scale | Gate before committing |
|---|---|---|
| DSL and generator validation | CPU; small exhaustive domains and streamed generated cases; metadata/text normally much smaller than model training assets | Property tests and measured case-generation rate |
| Small neural debugging | About 5M parameters; batch 1, short sequences; seconds-per-step measured locally | Forward/gradient/replay parity and tiny-set overfit |
| Controlled training | `U * measured_step_seconds`, with exact realized loop schedule and both rollout/replay passes; data exposure `U * effective_batch` | Pilot learning curves, statistical seed plan and memory stability |
| Confirmation inference | `N_tasks * N_seeds * mean_request_seconds` per method, plus generation/checking/analysis | Frozen test sizes and timing distributions |
| Larger references | Download bytes from repository metadata; FP16 tensor bytes approximately `2P`, plus state/runtime; quantized size measured rather than inferred | License, code review, conversion/parity tests and actual free disk |
| Deployment packaging | Weights/tokenizer/source/metadata manifest; training checkpoints excluded only from deployment, not project cost | Reload and byte-for-byte manifest verification |

For an illustrative arithmetic example, 10,000 updates at a **measured** 0.2/1/5 seconds per update would require approximately 0.56/2.78/13.89 hours for that run. These are conditional calculations, not predictions for this MacBook, and omit evaluation/checkpoint costs unless the measured boundary includes them. Longer runs are allowed when justified; there is no fixed overnight or ten-minute requirement.

A pilot should estimate training speed at several microbatches and lengths, learning-curve slope, seed variability and total cost of the planned comparison. Use successive **predeclared training allocations on development data** to see whether longer training remains informative. Do not select a favorable protected checkpoint. If expected runtime exceeds available uninterrupted sessions, use resumable stages, not an unapproved cloud fallback.

### 6.7 Quantization and packing

FP16 small-model deployment may already meet the chosen payload target; prove the mechanism before making it numerically harder. Then evaluate eight-bit weight-only deployment with the same trained checkpoint and calibrated recurrent trajectories. MLX affine quantization supports documented bit/group options, but operation availability does not establish fast M3 Max kernels for every option. Validate the exact format and matrix dimensions selected. [R31]

Measure serialized weights plus scales/biases and remaining higher-precision tensors. Test reload equality for the stored format, accuracy across every loop depth, prefill/core latency, resident memory and cold-load time. State explicitly whether weights remain packed during inference or are dequantized to floating point at load. The latter can save distribution bytes without saving resident weight memory.

Four-bit, two-bit or ternary experiments require a separate design justified by a storage bottleneck or a favorable quality frontier. Use train/development calibration only, covering early and late recurrent states. Compare ordinary per-group quantization, recurrent-depth-aware calibration and selective higher precision for input/state interfaces. Do not claim these ideas are novel; current papers directly address them. Ternary entropy `log2(3)` is not a claim that an implementation stores each parameter in exactly 1.58 bits. [R19], [R20], [R21], [R22]

### 6.8 Checkpointing, interruption and recovery

Checkpoint at optimizer-update boundaries. Save model, optimizer moments, scheduler position, gradient-scaling state if used, all RNG streams, generator cursor/sample-order manifest, loop draws, tokenizer hash, config/hash, training counters, backend versions and previous checkpoint hash. Saving only weights is not resumable training.

Use atomic temporary-file writes followed by fsync/rename and checksums; retain at least the last known valid checkpoint plus the new one until verification completes. Check free disk before a write. Do not save a half-accumulated minibatch unless gradients and its complete data/tape state are also serialized; the simpler default restarts that uncommitted minibatch.

W03 must compare uninterrupted and interrupted/resumed short runs on the same backend. Compare next sample IDs, next loop count, optimizer counters, losses and parameter checksums/tolerances. Record any nondeterministic kernel limitations. On OOM or process termination, preserve logs, mark the run failed/interrupted, restart from the last valid boundary, and record a new attempt ID. An altered microbatch with mathematically equivalent accumulation is a configuration amendment; retest numerical equivalence and do not rewrite the failed run's history.

---
## 7. Risk-ordered work packages and decision gates

Every package writes `decisions/Wxx.md`, updates `STATUS.json`, and records commands, inputs, outputs, checksums, costs and failed attempts. Entry conditions are blocking. Routine implementation choices are delegated to the agent; changes to the population, hypothesis, primary metric, information access, training treatment or baseline fairness require a numbered methodological amendment.

### W00 — Establish provenance, permissions and hardware

**Risk:** an unavailable asset, wrong architecture, incompatible backend or misleading size claim invalidates later work.  
**Inputs/dependencies:** this plan; a fresh target laptop; public network access for downloads.  
**Actions:** run §6.1 discovery; create the repository skeleton; review dependency/model/data licenses; record the cited source versions; resolve the native environment; inspect custom model code before executing it. Save metadata before large downloads. Confirm that public downloads do not require paid services.  
**Proposed commands/files:** `python -m microbrain doctor --out manifests/hardware.json`; `manifests/assets.jsonl`, `manifests/licenses.jsonl`, `locks/`, `decisions/W00.md`.  
**Tests/outputs:** ARM Python, functioning Metal, exact RAM/free disk bytes, a working minimal MLX or faithful MPS tensor forward/backward, source/license hashes.  
**Resource rule:** metadata and tiny tensors first; download weights only after disk/headroom checks.  
**Gate:** proceed when the custom-model path is valid. If a reference asset is blocked, document the exclusion and select another current validated reference. If Metal is unavailable, fix the environment or mark local acceleration blocked; CPU debugging may continue, but no M3 performance claim is possible.

### W01 — Implement the exact DSL and independent evaluator

**Risk:** a weak verifier or unsafe execution creates fictitious competence.  
**Dependencies:** W00.  
**Actions:** implement §4.2 semantics, grammar codec, typed decoder masks, deterministic feedback, public/private schemas and a separate evaluator process. Create independent reference implementations for arithmetic and list operators. Add output/bit/instruction/time limits.  
**Commands/files:** `python -m microbrain test-harness --config configs/harness.yaml`; `dsl.py`, `feedback.py`, `evaluator.py`, `tests/test_dsl.py`, `tests/test_information_boundary.py`.  
**Tests/outputs:** exact agreement across finite small domains; generated mutations rejected; add-versus-multiply ambiguity detected; malformed programs rejected without host execution; `SELECT` eager semantics; every exception gets a typed status; changing hidden outputs cannot change a solver's public transcript.  
**Resources:** CPU, bounded exhaustive domains and measured mutation-suite cost.  
**Gate:** require correct oracle execution and functioning negative controls. Any discrepancy is a harness failure, not evidence for/against H1. Resolve it before dataset generation or model fitting.

### W02 — Build splits, adapters and the first simple frontier

**Risk:** leakage, an artificial output ceiling or a trivial task makes the mechanism uninformative.  
**Dependencies:** W01.  
**Actions:** implement the typed generator and six strata; freeze motif/skeleton separation rules; create training/development corpora; keep confirmation generation keys sealed. Use an initial deterministic motif allocation of hash modulo ten: bucket 0 protected, bucket 1 development shift, buckets 2–9 training support, with only pre-generation coverage repairs. Implement shared lexical binding extraction on invented development examples. Record external dataset schemas/licenses without tuning on test labels. Build B1 and B2.  
**Commands/files:** `python -m microbrain make-data --config configs/data.yaml --partition development`; `python -m microbrain baseline --method enumerator --config configs/pilot.yaml`; `data/manifests/`, `data/grammars/`, `tests/test_splits.py`.  
**Tests/outputs:** zero protected motif/skeleton collisions, complete primitive coverage, rejection/length distributions, positive hidden-test mutation sensitivity, rule/search accuracy–cost curves and input coverage.  
**Resources:** stream examples; choose corpus size from throughput, structural diversity and subsequent learning-curve needs, not a large speculative download.  
**Gate:** if search already solves the population cheaply, investigate harder predeclared compositions or a real-language task **on development data**; retain the easy result and disclose the new population. If the generator cannot express a needed mechanism, revise the proxy before training. Do not make confirmation harder after seeing candidate results.

### W03 — Validate neural equations, gradients and resumability

**Risk:** an incorrect recurrence, cache, loss mask or training graph.  
**Dependencies:** W00–W02.  
**Actions:** implement the small encoder/core/head and state-mode flag; full-gradient replay; loop-depth loss normalization; parameter counting; atomic checkpoints; timing boundaries. Start with 32 distinct short training programs and enough repeats to test memorization, not generalization.  
**Commands/files:** `python -m microbrain smoke --config configs/small.yaml`; `python -m microbrain calibrate --config configs/small.yaml`; `models/core.py`, `training/replay.py`, `training/checkpoint.py`, `tests/test_gradients.py`.  
**Tests/outputs:** tied-versus-cloned forward/gradient checks, deterministic collection/replay parity, finite gradients at all depths, zero semantic training error on the tiny memorization set after successful optimization, no hidden-information flow, resume continuity, actual parameter and memory report.  
**Resource rule:** batch one and short shapes first; expand through §6.5.  
**Gate:** proceed only after correctness and a usable stable training path. Failure to fit the debug set requires diagnosis of optimization/labels/architecture; it is not evidence that recurrence cannot generalize. An unsupported kernel triggers a parity-tested equivalent implementation, not a silent change in model semantics.

### W04 — Establish competitive local baselines

**Risk:** an apparent advance is merely a weak baseline.  
**Dependencies:** W02–W03.  
**Actions:** train B3–B5 and screen B7; tune B4's width/depth and decoding under actual payload limits; validate B2's completeness on small cases; run no-semantic-feedback controls. Use identical generated data where the scientific panel calls for it, and record every training/tuning cost.  
**Commands/files:** `python -m microbrain train --config configs/baselines/seq2seq.yaml`; analogous configurations for `reset`, `operator_prior` and `gru`; `reports/baseline_pilot.json`.  
**Tests/outputs:** fitted debug tasks, stable development curves, payload manifests, beam/search costs, all failed configurations and a justified shortlist.  
**Resources:** use pilot step timings to allocate staged training; expand useful boundary hyperparameters. Avoid simultaneous residency of large models.  
**Gate:** do not advance a competitiveness claim until a serious conventional and discrete-repair baseline has converged or its explicit unresolved limitation is recorded. If a simple baseline dominates, retain it as the principal challenger rather than dropping it.

### W05 — Minimum mechanism pilot

**Risk/Hypotheses:** H1/H2/H3; persistent state may add no useful information or merely amplify training cost.  
**Dependencies:** W03–W04.  
**Actions:** train persistent and reset models from matched initialization seeds on the same task/update/loop sequence. Run a no-semantic-feedback pilot first, then the three-example pilot. Collect per-cycle hidden-state norms, loss, gradient norms, program changes, visible/hidden-development success, timings and training work. Do not open protected results.  
**Commands/files:** `python -m microbrain pilot --config configs/pilot.yaml`; `reports/mechanism_pilot.json`, `runs/*/trajectory.jsonl`.  
**Tests/outputs:** learning and quality-versus-cycles curves, seed-wise effects, loop-specific resource estimates, public/hidden mismatch rate and baseline headroom.  
**Resources:** staged development-only allocations justified by learning-curve slope; at least a multi-seed variance pilot before power planning.  
**Gate:** if improvements are plausible and detectable locally, proceed to W06. If the model underfits and gradients/labels are sound, extend training or test the wider size as a new development condition. If it learns the tasks but added state repeatedly loses to reset/search, narrow or reject the local hypothesis rather than escalating blindly. Insufficient sensitivity is recorded as inconclusive.

### W06 — Freeze the experiment and stronger references

**Risk:** adaptive choices or missing contemporary baselines invalidate confirmation.  
**Dependencies:** W04–W05.  
**Actions:** review current reference revisions and licenses; run one-at-a-time reference smoke/development tests; select the strongest validated practical reference and compact B*. Fit the sample/seed plan from pilot variance; define `T*`; choose the fixed-sequence testing gate specified in §5.8; freeze configurations, final checkpoint-selection rule, payload format, selectors and analysis code.  
**Commands/files:** `python -m microbrain preregister --config configs/confirmatory.yaml`; `manifests/confirmation.lock.json`, `reports/power_plan.json`, `decisions/W06.md`.  
**Tests/outputs:** hashes of every asset/config/split, simulated analysis under null and planted effects, zero hidden-result reads in the development log, current baseline inclusion/exclusion reasons and a resource budget with uncertainty.  
**Resources:** full downloads only for references passing metadata/code checks; prefer a smaller faithful reference over a larger broken one, while labeling the limitation.  
**Gate:** proceed when the locked design is affordable under measured local capacity and has interpretable sensitivity. If not, reduce the *claim scope* or report an underpowered plan; do not silently lower replication or drop a difficult stratum.

### W07 — Execute protected controlled comparisons

**Hypotheses:** primary H1 and conditionally the named practical H4 claim.  
**Dependencies:** W06.  
**Actions:** execute the frozen training seeds/budgets or frozen pilot-independent retraining procedure; generate/open protected tasks only after lock; run methods under the specified policies; write immutable predictions, trajectories and measurements; run the locked paired analysis once. Retain all unsuccessful runs and environment failures.  
**Commands/files:** `python -m microbrain confirm --lock manifests/confirmation.lock.json`; `python -m microbrain analyze --lock manifests/confirmation.lock.json`; `reports/confirmatory.md`.  
**Tests/outputs:** full predicted-program records, exact denominators, primary interval/effect, seed-wise effects, corrected/gated decisions, all required resource panels and regression summaries.  
**Resources:** fixed sample/seed sizes from W06; suspend/resume only under §6.8.  
**Gate:** positive, negative and inconclusive outcomes all advance to synthesis. New architectural changes start a new exploratory branch, not a repaired version of this confirmation.

### W08 — Evaluate realistic transfer and deployment references

**Risk:** synthetic gains have no useful-language counterpart.  
**Dependencies:** W06; execution can follow W07 without depending on its sign.  
**Actions:** evaluate the full SVAMP set in no-example mode; run the selected larger model with identical public tool access and documented templates; execute the frozen MBPP coverage/subset arm only if its reference translations are valid. Measure common-deadline and relaxed-budget curves, including preprocessing and checker overhead.  
**Commands/files:** `python -m microbrain evaluate --suite svamp --lock manifests/confirmation.lock.json`; `reports/transfer.md`, `manifests/mbpp_coverage.jsonl`.  
**Tests/outputs:** full-set and coverage-conditioned results, runtime failures, contamination caveats, interface variants and realistic-language error categories.  
**Resources:** one large reference resident at a time; timing blocks calibrated locally.  
**Gate:** failed transfer forbids a general assistant claim. It can justify a language-data or task-specific pivot only through a recorded amendment; the original transfer result remains unfavorable evidence.

### W09 — Attribute gains and test deployable packing

**Hypotheses:** H2/H3/H5 and mechanism attribution, not automatic confirmation of every exploratory slice.  
**Dependencies:** W05 for development ablations; W07 for interpreting protected effects.  
**Actions:** execute the prioritized ablations in §8; compare FP16/8-bit reloads and real packed/dequantized modes; assess selected larger/wider configurations only when earlier evidence justifies them; run fresh confirmation for any newly selected quantized or modified claim.  
**Commands/files:** `python -m microbrain ablate --config configs/ablations.yaml`; `python -m microbrain export --config configs/export.yaml`; `reports/attribution.md`, `artifacts/*/manifest.json`.  
**Tests/outputs:** matched-information/compute results, trajectory diagnostics, serialization hashes, precision-by-loop accuracy and RAM/time/size changes.  
**Resources:** prioritize tests that discriminate an observed effect; no mandatory large factorial grid when a cheap control explains it.  
**Gate:** if benefit disappears under a fair control, retract the stronger interpretation. If quantization erases the effect, report the FP16 result and failed compression separately. A packed file must reload correctly before any byte-budget claim.

### W10 — Reproducible decision and final report

**Dependencies:** completed or explicitly blocked prior packages.  
**Actions:** regenerate every table from immutable raw records; audit selection and resource accounting; perform a fresh-process replay of a stratified subset; produce the claim/outcome table in §10 and an executable README; retain a complete unfavorable-results appendix.  
**Commands/files:** `python -m microbrain report --manifest manifests/study.json`; `reports/final.md`, `reports/audit.md`, `STATUS.json`, deployment bundle plus manifest if valid.  
**Tests/outputs:** table/record consistency, no uncited inherited assets, no missing failures, hashes resolve, independent-process predictions match the documented numerical policy, public and hidden files remain separated in the bundle.  
**Resources:** bounded analysis/replay calibrated from previous stages.  
**Gate:** release conclusions matching the actual evidence: positive mechanism, engineering-only gain, narrow specialist, no advantage, resource-limited or inconclusive. Completion does not require a positive result.

---

## 8. Attribution, robustness and failure analysis

### 8.1 Priority-ordered controls

| Test | What remains fixed | What a positive result would and would not imply |
|---|---|---|
| Persistent versus trained reset, full history | Parameters, examples, update schedule, losses, loop count, public information | Main H1 test. Does not alone establish lower latency or broad task transfer |
| Common-transcript replay | Both models receive the same externally collected public candidate/feedback sequence; never hidden labels | Distinguishes state processing from better earlier action selection. It is an off-policy diagnostic, not deployment accuracy |
| No semantic feedback versus three public examples | Architecture and training allocation; separate explicit input conditions | Tests information/feedback value. Extra examples are extra information, not free reasoning |
| Final candidate versus public-selected candidate | Proposal trajectory | Separates iterative refinement from search/selection benefit; hidden best-of remains oracle-only |
| More loops versus more B4/B5 proposals or search nodes | Actual latency/compute envelope and public tool access | Tests whether the gain is simply more opportunities to propose an answer |
| Deep supervision versus last-loss-only | Data, optimizer updates and architecture; training work disclosed | Tests whether a training objective explains the result rather than inference state |
| Untied executed depth; same-size deeper/narrower baseline | One panel matches executed work, another stored bytes | Separates shared capacity, optimization and inference depth; cannot match every resource at once |
| Same training examples versus same training work | Each matching rule in its own panel | Prevents extra recurrent backpropagation/replay from being hidden project expenditure |
| GRU or operator-prior search | Interface, task population and payload accounting | Tests a simpler architecture/algorithmic explanation |
| FP16 versus packed/dequantized int8 | Checkpoint, public examples and loop schedule | Separates numerical and packaging changes from an architectural result |

Do not implement every expensive control before a signal exists. The indispensable controls are valid information boundaries, B5/full history, B4, B2/B3, training-exposure accounting and end-to-end timing. The remaining tests are ordered by which alternative explanation best fits the pilot. Their prioritization and omitted uncertainties must be recorded.

### 8.2 Numerical and representation diagnostics

For each cycle, log latent RMS norm, relative state change `||H_t-H_(t-1)||/(||H_(t-1)||+eps)`, finite/nonfinite counts, logit margins, program edit distance and public/development success. Measure sensitivity to small state perturbations in development. A local finite-difference or Jacobian-vector probe can estimate amplification; it is a diagnostic of the sampled trajectory, not a global contraction proof.

The recurrence linearization is `delta_H_(t+1) ≈ J_t delta_H_t + epsilon_t`, so quantization or roundoff can accumulate through products of step Jacobians. If their norms are large, repeated application can amplify error; if too contractive, useful distinctions can disappear. Neither stable state norms nor a small update residual implies the correct answer. Avoid imposing a contraction penalty before seeing the failure regime: it may remove useful computation. A gated recurrent update or residual-scaling experiment is allowed as a separately trained, counted refinement when instability is observed.

Compare two programs that agree on visible examples but disagree elsewhere. Deliberately include fixed points that are wrong, cycles between candidates, late-loop regressions, and correct programs changed into incorrect ones. Report these transition rates; an aggregate “more thinking helps” plot can conceal them.

### 8.3 Adversarial and boundary cases

Test contradictory descriptions/examples, underspecified descriptions, extra irrelevant numbers, swapped variable bindings, negation, changed comparison direction, unusual entity names, Unicode minus signs, decimal formatting, empty lists, duplicate values, equal thresholds, invalid indices and division by zero. Define expected behavior for contradictions as “unsupported/ambiguous” in a separate abstention stress set; do not mix them into ordinary exact-success labels without a specification.

Check prompt injection in task text: instructions to reveal hidden tests, execute host commands, contact an external service or bypass the DSL must remain inert data. Verify the solver process has no hidden-label handle or network tool. Add resource bombs within the parser, such as huge integer literals, excessive nesting and oversized lists; reject before expensive execution and report the rejection.

Shuffle case order and restart the process to catch accidental state leakage between requests. Batch identical tasks with different neighbors to detect cross-example attention/masking contamination. Replace hidden labels while preserving public inputs: a solver's output distribution and transcript must not change. Remove retrieval files at deployment to ensure no supposedly weight-only result silently depends on them.

### 8.4 Failure taxonomy and next diagnostic

| Observed failure | Classification | Next action |
|---|---|---|
| Oracle fails or identical solvers produce systematic scored differences | Invalid implementation/measurement | Stop scoring; repair harness and invalidate affected runs |
| Tiny model cannot fit short training cases | Implementation/optimization/capacity unresolved | Inspect gradients, masking, labels and learning rates; no generalization conclusion |
| Fits training but all models fail shifted language | Proxy/data/interface limitation possible | Inspect independent linguistic errors; test better training language under amendment; retain failed transfer |
| Strong models and search solve easily; tiny methods indistinguishable | Ceiling/insensitive workload | Declare insufficient headroom; redesign development workload before a new confirmation |
| Recurrent benefit vanishes with equal training work or beam search | No demonstrated mechanism advantage under fair resources | Report training/search explanation; consider simpler deployment |
| Recurrence benefits only familiar templates | Narrow distribution-specific effect | Reject systematic-generalization claim |
| More cycles increase visible pass rate but not hidden success | Verifier/ambiguity overfitting | Quantify ambiguity and shortcut selection; no semantic-improvement claim |
| Correct model cannot fit local memory with faithful optimization | Local resource limitation | Use exact checkpointing/accumulation; otherwise narrow scale and preserve unresolved larger-scale claim |
| Tiny effect, broad uncertainty or inconsistent seeds | Inconclusive | Report precision; follow preregistered replication plan, not post-hoc significance chasing |
| Strong quality gain with unacceptable time or a severe subgroup regression | Trade-off, not unqualified improvement | Report frontier and affected users/tasks; no blanket superiority claim |

---

## 9. Repository, interfaces, logging and reporting

### 9.1 Minimal repository layout

```text
microbrain-poc/
  README.md
  pyproject.toml
  requirements.in
  locks/                         # Resolved environments, package/source hashes
  configs/
    data.yaml  harness.yaml  small.yaml  pilot.yaml
    confirmatory.yaml  ablations.yaml  export.yaml
    baselines/  references/
  src/microbrain/
    __main__.py  cli.py  schema.py  provenance.py
    dsl.py  feedback.py  evaluator.py  timing.py
    data/                        # Generator, grammar, split and external adapters
    models/                      # Shared core, seq2seq, GRU, operator predictor
    solvers/                     # Common adapter, rules, retrieval, enumeration
    training/                    # Rollout/replay, optimizer loop, checkpoints
    analysis/                    # Locked estimands, bootstrap, tables
  tests/
    test_dsl.py  test_splits.py  test_information_boundary.py
    test_gradients.py  test_replay.py  test_resume.py  test_export.py
  data/
    manifests/  grammars/  train/  development/  protected/
  manifests/
    assets.jsonl  licenses.jsonl  hardware.json  study.json
    confirmation.lock.json
  runs/<run_id>/
    config.yaml  provenance.json  events.jsonl  metrics.jsonl
    predictions.jsonl  trajectory.jsonl  timing.jsonl
    checkpoints/  failures.jsonl
  artifacts/<artifact_id>/        # Only audited deployment files
  decisions/                     # W00.md ... W10.md, amendments, search updates
  reports/                       # Generated tables and narrative conclusions
  STATUS.json
```

Protected data are separated by interfaces/process permissions; merely creating a directory does not enforce the embargo. Version code and small manifests. Use content-addressed storage or an asset manifest for large binaries; do not commit secrets or replicate gigabyte weights unnecessarily.

### 9.2 Shared interfaces and data schemas

Use immutable public task objects and explicit private evaluation records. Below is an implementable interface skeleton; serialization validation, the tensor backend and error handling must be supplied in the named modules.

```python
from dataclasses import dataclass
from typing import Literal, Protocol, Callable

Scalar = tuple[int, int]  # reduced numerator, positive denominator
Value = Scalar | tuple[Scalar, ...] | bool

@dataclass(frozen=True)
class PublicTask:
    task_id: str  # opaque; not a generator seed or semantic hint
    description: str
    signature: dict[str, object]
    query_bindings: dict[str, Value]
    examples: tuple[tuple[dict[str, Value], Value], ...]

@dataclass(frozen=True)
class Budget:
    deadline_ns: int | None  # absolute monotonic deadline; None for fixed-loop test
    max_cycles: int
    max_proposals: int | None
    max_search_nodes: int | None

@dataclass(frozen=True)
class Program:
    rows: tuple[tuple[str, str, str, str], ...]

@dataclass(frozen=True)
class PublicFeedback:
    fields: tuple[tuple[str, ...], ...]
    elapsed_ns: int

@dataclass(frozen=True)
class SolveResult:
    status: Literal['ok', 'timeout', 'invalid', 'unsupported', 'error']
    program: Program | None
    answer: Value | None
    elapsed_ns: int
    proposal_count: int
    core_calls: int
    tool_calls: int
    log_path: str

class Solver(Protocol):
    def solve(self, task: PublicTask, budget: Budget,
              check_public: Callable[[Program], PublicFeedback]) -> SolveResult: ...
    def artifact_manifest(self) -> dict[str, object]: ...
```

Because raw Python tuples make a rational pair potentially ambiguous with a list representation, the **wire format must use tagged values**: `{"type":"N","num":3,"den":2}`, `{"type":"L","items":[...]}` or `{"type":"B","value":true}`. Validate denominator positivity, integer bounds and the distinction between `bool` and `int`. Convert tags to internal types only after validation. Do not deserialize arbitrary pickle objects from models or datasets.

Private evaluation records add `cluster_id`, `stratum`, reference specification, hidden bindings and expected outputs. They must never be fields on `PublicTask`. The evaluator joins them to predictions by opaque task ID after the solver returns. Do not let a public tool error message include a file path, hidden expected value or reference-program fragment.

Every prediction record includes: schema version, run/method/artifact IDs, task ID, training seed, status, exact serialized program/answer, submitted iteration, elapsed nanoseconds, counters, public feedback transcript hash and error category. Scoring joins add hidden correctness and coverage flags in an evaluator-only output. Raw task contents are stored once with provenance, not copied into every log unnecessarily.

### 9.3 Configuration convention

All experiments are driven by a fully resolved YAML saved before execution. Reject unknown fields and unresolved mandatory values. This is a **template for the agent to resolve from calibration**, not a runnable trained configuration:

```yaml
schema_version: 1
study: microbrain_v1
phase: exploratory
method: persistent
backend: mlx
model:
  vocab_size: 4096
  hidden: 256
  intermediate: 768
  heads: 4
  encoder_blocks: 2
  shared_core_blocks: 2
  program_rows: 24
  fields_per_row: 4
  max_input_tokens: 2048
  norm_epsilon: 1.0e-5
  dropout: 0.0
training:
  loop_values: [1, 2, 4, 8]
  loop_probabilities: [0.25, 0.25, 0.25, 0.25]
  replay: deterministic_two_pass
  gradient_through_continuous_state: full
  optimizer_updates_per_effective_batch: 1
  microbatch: null              # W03 measured fit; unresolved blocks training
  accumulation: null            # Preserve chosen effective batch
  total_updates: null           # W05 learning curves, then W06 lock
  seeds: null                   # W06 variance/precision plan
  parameter_dtype: float32
  compute_dtype: float32        # FP16 only after parity gate
  optimizer: adamw
  learning_rate: null           # Independently tuned on development data
  betas: [0.9, 0.95]
  epsilon: 1.0e-8
  gradient_clip_norm: 1.0
  weight_decay: null
  checkpoint_interval_updates: null  # Measured write/recovery trade-off
information:
  visible_examples: 3
  full_public_history: true
  hidden_evaluator_access: false
inference:
  primary_cycles: 8
  primary_selector: final
  deadline_ns: null             # Only required for practical deadline panel
accounting:
  learned_weight_cap_bytes: 32000000
  portable_payload_target_bytes: 16000000
provenance:
  data_manifest_sha256: null
  tokenizer_sha256: null
  code_commit: null
  environment_lock_sha256: null
```

The normal CLI accepts `--config`, `--out`, `--resume`, and `--dry-run`; confirmation accepts only a locked manifest plus operational output/resume paths. `--dry-run` validates schema, entry gates, assets, disk and resolved budgets without launching training. Any command must return nonzero on an unmet gate, not quietly substitute a default. Native external commands in §6 remain distinct from this project CLI.

### 9.4 Experiment records and reporting invariants

An event record contains UTC timestamp, run ID, attempt ID, phase, config hash, code revision, device/backend, seed/cursors, event name, elapsed seconds, optimizer update, cumulative task/token/core-call counts, loss and memory/timing fields with explicit units. Use JSON `null` for unavailable measurements, not fabricated zeros. Save raw stdout/stderr and stack traces for failures.

Store an asset ledger with stable source URL, exact revision, download date, SHA-256, byte size, license identifier/text hash, intended role, preprocessing/quantization actions and redistribution restrictions. For a downloaded teacher/reference model, distinguish inherited pretraining from local fine-tuning/inference. Log local teacher generation and filtering if an amended experiment uses them.

`STATUS.json` records current completed/blocked packages, latest valid checkpoints, next gate, unresolved issues and references to decision records. It must be sufficient to resume without chat history. Keep unfavorable results in the same machine-readable tables as successful runs. Tables must be generated from raw records rather than manually edited screenshots or selected summaries.

Every final table includes denominators, training-seed count, loop/call policy, public-information condition, payload definition, precision, latency boundary and whether results are exploratory or confirmatory. A deployment package contains a manifest of **all** required files and a clean-load test; the report additionally inventories runtime dependencies and any optional memory assets excluded from the payload tier.

---
## 10. Outcome interpretation and evidence-triggered continuation

### 10.1 Decision rules after the protected comparison

Interpret the causal and practical results separately. A useful compiler need not validate persistent latent state; a real latent-state effect need not produce a useful product.

| Observed outcome | Permitted conclusion | Required next action |
|---|---|---|
| H1 improves held-out success, survives fair tuning, and the gated H4 comparison improves the compact frontier | Evidence for the tested persistent-state mechanism and a task-specific resource advantage | Repeat on a fresh structural holdout, inspect per-family regressions, complete external transfer and package verification before making a deployment claim |
| H1 is positive, but discrete search or a conventional compiler offers a better deployment frontier | A component-level effect, not overall superiority | Publish the cost decomposition; retain the simpler winner for deployment. Escalate only if profiling identifies a removable implementation bottleneck rather than inherently greater work |
| H1 is null/negative, but the reset/compiler system is competitive | Tools or structured prediction are valuable; persistent state has not earned its complexity | Use the best simpler controller. Treat this as an engineering or specialization result, not a recurrent-reasoning breakthrough |
| Gains appear only with three examples and disappear without them | Evidence for example-conditioned synthesis, not broad request understanding | Narrow the application to programming by example; keep SVAMP and no-example failures visible |
| Visible-test success rises while independent hidden success does not | Feedback exploitation or underdetermination, not reliable semantic improvement | Analyze ambiguity and proposal selection. Redesign the information condition only through an amendment and fresh confirmation |
| Synthetic structural transfer succeeds but SVAMP/eligible external transfer fails | A synthetic compositional result with unproven practical language transfer | Diagnose number binding and wording coverage; evaluate any additional language training as a separate intervention with equal access for baselines |
| Strong baselines and candidate are all at ceiling or floor | The proxy is insensitive or the training/representation is inadequate | Use the already defined development difficulty axes and oracle diagnostics to locate a sensitive regime; do not select protected tasks to manufacture separation |
| Confidence interval is wide or training seeds disagree | Inconclusive at the achieved precision | Report the detectable-effect limit. Start a separately planned independent replication only if its information gain justifies its measured local cost |
| All relevant intervals are tight and simpler methods dominate on the intended workload | Evidence against the selected mechanism's local value | Stop this branch or retain the simpler system; do not invoke unspecified future scale as an unfalsifiable excuse |
| Hardware or faithful backend support blocks the matched experiment | A local feasibility failure, not a mechanism falsification | Apply the declared faithful fallback; otherwise report the blocked comparison without substituting incompatible published scores |

Do not call parity with a larger model a confirmed result merely because a difference is nonsignificant. A noninferiority claim requires a practical margin specified independently of protected outcomes and an appropriate confidence bound. With no application-specific margin, report the observed difference and uncertainty, not “equivalent intelligence.” Similarly, a smaller file at materially worse accuracy is a trade-off, not dominance.

### 10.2 Conditions for increasing model size, data or recurrent depth

Increase resources only to resolve a diagnosed uncertainty. More data is warranted when development learning curves and novel-template errors indicate underexposure, with training cost forecast from measured throughput. Greater width is warranted when optimization is sound, training performance shows a capacity limit, and the wider FP16 or validated 8-bit package still meets the stated size tier. More cycles are warranted when held-out development trajectories show useful progress that is not replicated by equal-cost discrete search. Use all these rules symmetrically when improving baselines.

For a suspected scale-dependent effect, preregister a small scaling study across feasible settings, preserving architecture family, data construction and accounting. Report what remains unverified beyond the laptop. Failure at a scale that cannot represent the required program does not refute the mechanism; success on a tiny task also does not establish its larger-scale behavior. A trend suggesting a crossover is a hypothesis, not evidence that an unrun crossover occurs.

### 10.3 Evidence-triggered alternatives, in priority order

**Simpler program controller.** Trigger: B4, B5 or neural-guided typed search matches or dominates persistent state. Preserve the tiny-model objective and deploy the simpler approach. Search for a distinct remaining bottleneck—such as language binding, calibrated abstention or cost-aware search—before proposing another architectural addition. Each new contribution needs its own close comparator and protected test.

**Better language exposure or local distillation.** Trigger: oracle bindings/structured inputs succeed, unfamiliar wording fails, and a locally runnable reference model materially improves that interpretation step. Generate teacher supervision only from training tasks, validate it independently, count teacher inference and filtering, and give the same supervised assets to the relevant baselines. The deployed model must not call the teacher. This tests distillation, not pure from-scratch recurrent learning.

**Learned compact procedure library.** Trigger: many solved training tasks repeatedly contain the same nontrivial subprograms, and baseline search/profile data identifies repeated composition as the limiting cost. Compare a library compressed from training programs with hand-written primitives, ordinary common-subexpression extraction and DreamCoder/LILO-style library induction. Count stored library bytes and discovery cost; hold out new combinations of library procedures. This is a separate library-learning study, not a new name for established program induction. [R14]

**Retrieval-separated knowledge.** Trigger: a specified application actually needs unavailable factual information and exact reasoning is already adequate. Begin with a fixed local text corpus and a transparent lexical index; compare retrieval-only templates and a larger model with the identical corpus. Evaluate source support and updateability, count corpus/index/runtime bytes, and label the outcome “small weights with external memory.” This does not meet a strict all-information-in-16-MB goal. [R18]

**The original self-contained neural/language track.** Trigger: the project explicitly prioritizes broad language prediction or neural-only reasoning rather than the selected specialist's task utility. Start a separately scoped track using the Parameter Golf evaluation rules and current verified submission artifacts as appropriate references. Its BPB metric, training corpus, tokenizer and artifact definition must be fixed independently of the compiler task. Port a relevant implementation faithfully and compare a tied-depth version with tuned conventional and classical/neural hybrids at matched local budgets. Report local-budget results separately from original challenge training conditions. Do not present improved BPB as evidence of instruction following or broad reasoning. [R16], [R17]

**Lower-level recurrent mechanisms.** Trigger: trajectory diagnostics identify a specific optimization or stability failure, rather than merely disappointing end-to-end accuracy. A gated recurrent cell, modified state update, local learning rule or fixed-point formulation is a legitimate candidate. Isolate its effect with matched representational capacity and training, test gradients/stability and then task behavior. Contraction to the wrong output is not progress; a solver that immediately internalizes a fixed point need not benefit from more iterations. No transformer-specific assumption should override a better-supported mechanism. [R02], [R26]

None of these alternatives may consume protected results as training or development data without a recorded amendment. They are continuation rules, not an instruction to implement all branches.

### 10.4 Required final research report

Write `reports/final.md` from the run records. Its first page states the tested claim, best supported conclusion, strongest counterevidence, actual device, direct baselines, three footprint tiers and the distinction between fresh synthetic and public external evaluations. Include accuracy/resource frontiers, paired uncertainty, per-seed and per-family results, public-versus-private gaps, training/tuning expenditure, the failure ledger and all methodological amendments.

The conclusion must choose **continue**, **narrow**, **use a simpler alternative**, **inconclusive**, or **stop this branch**, with the evidence supporting that choice. Preserve trained artifacts and scripts sufficient for another owner of a comparable laptop to reproduce the result. Do not claim successful execution merely because all work packages have implementation files; protected results and clean artifact loading must actually exist.

---

## 11. Audit findings, performed checks and residual uncertainty

These are three distinct self-audits of this plan, not independent peer review. The revisions below are incorporated into the specifications and gates above. They do not replace execution-time testing.

### 11.1 Scientific and prior-art audit

| Defect or challenge identified | Revision incorporated | Residual risk or acceptance test |
|---|---|---|
| Weight reuse, tool use, execution-guided repair and checker-feedback recursive training already exist | Recast the main contribution as an empirical state-persistence comparison; include SED-style repair, full-history reset and typed search | An equivalent controlled finding may exist outside the inspected corpus; §3.4 defines what would invalidate novelty |
| Puzzle success and language-model BPB do not establish a tiny general assistant | Separate specialist semantic success, external-language transfer and any later BPB track | Broad intelligence remains untested even after specialist success |
| A “conventional 16 MB baseline” could secretly use recurrence, adaptation or external memory | Inspect actual submission mechanisms and distinguish Parameter Golf's root MLX example from its competitive records | A future strict-payload language study needs its own faithful baseline reproduction |
| Loop-specific quantization might look like an original refinement | Include LoopQ, feedback-exposure calibration and LoopQuant, including October 2026 material | No quantization novelty claim without a genuinely different mechanism and direct comparison |
| Published loop benefits and negative autoregressive TRM results appear inconsistent | Separate tasks, training, recurrence placement, supervision and compute matching; test rather than assume a transfer | Local compiler results will not settle every scale/task regime [R06], [R07] |
| Neural-symbolic success could be entirely explained by the hand-written operator set | Add oracle coverage, rules, enumerative search, neural-guided search and conventional compiler baselines | Domain choice is still a modeling assumption; external utility is unspecified |
| A small model's label might omit stored parameters | Count complete tensors and packaging; distinguish Nanbeige's non-embedding count and Gemma's effective count from total weights | Exact resident and serialized sizes require pinned artifact inspection [R25], [R28] |

Historical deployed precedent also weakens any broad “tiny front end plus programs” novelty story: Microsoft's Flash Fill project documents an Office 2013 programming-by-example feature. That is contextual precedent, not a direct accuracy baseline for this DSL. Patent disclosures further demonstrate long-standing neural/multimodal program-inference ideas; this is not a patentability or freedom-to-operate opinion. [R35], [R36], [R37]

### 11.2 Experimental-validity audit

| Defect or alternative explanation | Revision incorporated | Required evidence before accepting a result |
|---|---|---|
| Resetting latent state while withholding history creates an information/memory advantage | Both methods get their complete public action/feedback histories; add common-transcript diagnostics | Persistent-state gain survives the properly trained reset comparator |
| Passing visible tests is mistaken for semantic correctness | Independent hidden cases, public ambiguity measurement, mutation calibration and no hidden oracle feedback | Hidden-case and external answer gains, not merely visible pass-rate gains |
| More recurrent losses or optimizer updates can masquerade as better reasoning | Normalize loop losses; use one update per effective batch; separate final-only and equal-work controls | The claimed effect is not fully explained by extra supervision or training expenditure |
| Numeric variants/paraphrases and repeated timing measurements inflate sample size | Structural clusters and training seeds are explicit uncertainty units | Resampling preserves dependence; precision plan is frozen before protected results |
| Candidate selection can use a hidden-correct proposal in hindsight | Primary uses the final iterate; deployable selection uses public information only | Reported answer and deadline match the actual submitted output |
| External benchmark adaptation can cherry-pick representable tasks | Freeze the semantic coverage rubric and all eligible/rejected IDs before model outputs | Publish eligible and whole-population coverage-adjusted denominators |
| A size gain can hide an unacceptable latency, accuracy or runtime regression | Report multiple resource axes, three storage tiers and full end-to-end boundaries | Any dominance or parity wording satisfies its stated conditions |
| Tuning, repeated inspection and switching metrics can manufacture a favorable result | Development/confirmation separation, fixed sample design and fixed-sequence testing | Every amendment and unfavorable run is retained; changed claims get fresh confirmation |

A remaining limitation is that canonical syntax does not perfectly identify semantic equivalence classes. The generator removes declared identities and duplicates, but discovering every equivalent program is not required or claimed. Report residual near-equivalence diagnostics and do not describe a hash split alone as proof of concept-level novelty.

### 11.3 Implementation and feasibility audit

| Defect or fresh-laptop failure | Revision incorporated | Blocking test or faithful fallback |
|---|---|---|
| The target machine's RAM, OS and native architecture were unknown | Discovery precedes downloads; batch, context and reference-model admission use measured memory | W00 fails cleanly when requirements are unmet; no maximum-memory assumption |
| Host-side exact feedback cannot be placed naively inside a compiled differentiable graph | Deterministic rollout followed by fixed-action replay; no dropout or intervening updates | W03 checks conditional-gradient and action equality; count both passes |
| Gold labels can become illegal under an incorrect decoded prefix | Separate training masks from predicted-prefix inference masks | A test where an incorrect early type choice does not produce an infinite supervised loss |
| Quantized modules may not support training gradients | FP32 reference training and ordinary floating deployment first | Quantization is a separately validated frozen deployment path [R31] |
| Reusing changing workspace attention keys can silently change the algorithm | Cache only invariant input projections; recompute mutable workspace state | Cached/uncached predictions agree within frozen numerical tolerances |
| Unified host/device memory can be double-counted | Report allocation views separately rather than summing overlapping metrics | Independent OS-pressure observation and stable repeated peak measurement |
| A 12-step length test can exceed a small fixed program output | Use 24 rows, with at most 23 computational instructions | All protected reference programs fit the declared interface without truncation |
| JSON rational pairs, lists and Booleans can be confused | Tagged wire values and explicit validation before internal conversion | Round-trip every value type, including a two-element list and Boolean values |
| A training restart can alter optimizer, random loop draws or data ordering | Atomic batch-boundary checkpoints with complete state and replayed next-batch checks | Resumed and uninterrupted runs agree under the declared deterministic/tolerance contract |
| Current hybrid/looped model support is not guaranteed by general “Apple Silicon support” | Model-specific cache, position, recurrent-state and generation parity gates | Use a faithful alternate backend or mark that reference blocked; never substitute a different model silently |
| Unresolved example configuration fields can become accidental defaults | CLI rejects unresolved fields at the appropriate gate; W05/W06 resolve quantities from measurements | `--dry-run` identifies the exact missing decision and its responsible work package |

The strongest published model may not be runnable locally at its published precision or context. The direct claim then names the strongest **faithfully executed** admitted local reference, while the unrun result remains context only. Low memory changes batching/checkpointing before model semantics; an unavailable dataset license blocks that optional arm rather than encouraging undocumented replacement. Synthetic-only success is reported as such. These scenarios are not reasons to mislabel a convenient substitute as the original experiment.

### 11.4 Checks actually performed during preparation

**Environment:** Linux `6.18.44-x86_64`, Python `3.13.5`, PyTorch `2.10.0+cpu`, CPU float64, one Torch thread, seed 137. These checks were executed during plan preparation. They are mathematical/implementation sanity checks, not trained-model results or evidence of M3 compatibility.

| Check | Result measured or calculated here | Interpretation |
|---|---|---|
| Original two-block dominant matrix count | 20,447,232 parameters | Confirms the starting storage arithmetic before small tensors/metadata |
| Proposed small/wide specified count | 4,985,088 / 9,836,928 parameters | Includes the specified `17d` small-tensor estimate; an implemented inventory must replace estimates |
| Corresponding raw FP16 bytes | 9,970,176 / 19,673,856 | Does not include file headers, tokenizer or runtime |
| Four tied tanh steps versus four identical cloned matrices | Maximum forward absolute difference `0.0` | Confirms the comparison identity in the small CPU check |
| Tied gradient versus sum of corresponding cloned gradients | Maximum absolute difference `6.938893903907228e-18` | Checks the weight-sharing chain rule numerically |
| Detached-decision rollout versus fixed-decision replay | Maximum gradient absolute difference `0.0` | Supports the conditional-gradient replay construction, not all future MLX implementations |
| Addition and multiplication on two public examples | Both pass `(0,0)` and `(2,2)`; disagree at hidden `(2,3)`, producing 5 versus 6 | Constructive counterexample to treating public consistency as semantic proof |
| Paired-binary planning example in §5.8 | `ceil((1.96 + 0.841621)^2 * 0.20 / 0.02^2) = 3925` | Illustrative unclustered sample-size arithmetic, not the final design |

The following standalone snippet reproduces the two gradient sanity checks; it is not the proposed model implementation:

```python
import torch

torch.set_num_threads(1)
torch.manual_seed(137)
dtype = torch.float64
x = torch.randn(2, 5, dtype=dtype)
w = (torch.randn(5, 5, dtype=dtype) * 0.1).requires_grad_()
u = torch.randn(2, 5, dtype=dtype) * 0.1
h = x
for _ in range(4):
    h = torch.tanh(h @ w + u)
g_tied, = torch.autograd.grad(h.square().sum(), w)
clones = [w.detach().clone().requires_grad_() for _ in range(4)]
h_clone = x
for wi in clones:
    h_clone = torch.tanh(h_clone @ wi + u)
g_clones = torch.autograd.grad(h_clone.square().sum(), clones)
assert torch.allclose(h, h_clone, atol=1e-12, rtol=0)
assert torch.allclose(g_tied, sum(g_clones), atol=1e-12, rtol=0)

def rollout(weight, forced_actions=None):
    state, actions = x, []
    loss = torch.zeros((), dtype=dtype)
    for t in range(4):
        action = ((state.detach().sum(1, keepdim=True) > 0).to(dtype)
                  if forced_actions is None else forced_actions[t])
        actions.append(action)
        state = torch.tanh(state @ weight + 0.05 * action + u)
        loss = loss + state.square().sum() / 4
    return loss, actions

loss, tape = rollout(w)
g_online, = torch.autograd.grad(loss, w)
loss_replay, _ = rollout(w, tape)
g_replay, = torch.autograd.grad(loss_replay, w)
assert torch.allclose(g_online, g_replay, atol=1e-12, rtol=0)
```

**Not performed:** M3 execution, MLX/MPS kernel parity, full interpreter implementation, model training, benchmark scoring, throughput measurement, quantization evaluation or independent replication. No empirical success rate, latency or laptop completion time in this plan is a measured result of the proposed system.

### 11.5 Residual risks and limits

The most consequential unresolved question is whether a locally trained five-million-parameter model acquires sufficient language grounding. The synthetic generator may be too narrow, the public feedback may underdetermine meaning, and persistent state may be less useful than discrete search. Local optimizer behavior and accelerator support remain measured gates. The exact installed footprint will exceed raw weight size; the application has not supplied a utility function for exchanging memory, latency and accuracy.

Pretrained references have unequal inherited data and possible public-benchmark contamination. A versioned reference run is more informative than copying its model card, but cannot erase those differences. Recent papers may be preliminary; repositories and model cards are mutable. The bibliography records inspection scope and dates, not a guarantee of reproducibility or exhaustive prior-art discovery.

A final challenge pass incorporated explicit Boolean feedback behavior, external-test denominator definitions, latest source-version metadata and the distinction between a nominal parameter label and complete tensors. Remaining unknowns are either scientific hypotheses or named entry gates, rather than claims that have already been resolved.

---

## 12. References and search-coverage record

### 12.1 How to read the evidence record

All sources below were accessed during preparation on **16 October 2026**. “Full text inspected” means relevant accessible paper sections were opened and read, not that the research was independently reproduced or every theorem checked. “Abstract/card inspected” supports only the stated scope. Repository inspection refers to the accessible revision at research time; the execution agent must pin a commit/hash and preserve license text before reuse. An asset being downloadable is not evidence that redistribution is unrestricted.

Stable arXiv identifiers, project URLs and patent publication identifiers are supplied so the handoff does not depend on chat citations. Some model/runtime pages are moving documents rather than fixed releases. Their exact executed revision belongs in the W00/W06 asset ledger.

### 12.2 Recurrent computation and learning dynamics

**[R01] Mostafa Dehghani, Stephan Gouws, Oriol Vinyals, Jakob Uszkoreit and Łukasz Kaiser. _Universal Transformers._ 2018; ICLR 2019.** arXiv:1807.03819. [Paper][R01]; [inspected HTML v3](https://arxiv.org/html/1807.03819v3). Full-text mechanism inspected. Establishes shared-depth recurrence and adaptive computation as prior art; it does not establish this plan's tiny compiler advantage.

**[R02] Łukasz Kaiser and Ilya Sutskever. _Neural GPUs Learn Algorithms._ 2015; ICLR 2016.** arXiv:1511.08228. [Paper][R02]; [inspected HTML v3](https://arxiv.org/html/1511.08228v3). Full-text mechanism inspected. Relevant non-transformer precedent for learned iterative algorithms and length generalization; different tasks and training.

**[R03] Alexia Jolicoeur-Martineau. _Less is More: Recursive Reasoning with Tiny Networks._ 6 October 2025.** arXiv:2510.04871. [Paper][R03]; [HTML v1](https://arxiv.org/html/2510.04871v1); [official repository](https://github.com/SamsungSAILMontreal/TinyRecursiveModels); [inspected `trm.py`](https://raw.githubusercontent.com/SamsungSAILMontreal/TinyRecursiveModels/main/models/recursive_reasoning/trm.py). Paper and recurrence/halting/detach implementation inspected. Supports specialized small-model recursion, not general language competence or equivalence to full BPTT.

**[R04] Jonas Geiping and colleagues. _Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach._ 7 February 2025.** arXiv:2502.05171. [Paper][R04]. Abstract inspected. Relevant large-scale recurrent-depth language-model precedent; its pretraining regime is not reproduced by this local plan.

**[R05] Sangmin Bae and colleagues. _Mixture-of-Recursions: Learning Dynamic Recursive Depths for Adaptive Token-Level Computation._ 14 July 2025.** arXiv:2507.10524. [Paper][R05]; [HTML v1](https://arxiv.org/html/2507.10524v1); [implementation link](https://github.com/raymin0223/mixture_of_recursions). Full-text mechanism inspected; linked code was not executed. Adaptive shared computation already exists; reported base-model scales and GPU engineering differ from the proposed small compiler.

**[R06] Paulius Rauba, Claudio Fanconi and Mihaela van der Schaar. _Tiny Autoregressive Recursive Models._ 9 March 2026; ICLR 2026 Workshop RSI.** arXiv:2603.08082. [Paper][R06]; [HTML v1](https://arxiv.org/html/2603.08082v1). Full text inspected. Important negative evidence: controlled character-level algorithmic tests did not find reliable gains from the full autoregressive TRM formulation, despite some useful simpler refinements. This motivates, rather than settles, the present controls.

**[R07] Harsh Kohli, Srinivasan Parthasarathy, Huan Sun and Yuekun Yao. _Loop, Think, & Generalize: Implicit Reasoning in Recurrent-Depth Transformers._ 9 April 2026; revised 11 August 2026, COLM 2026.** arXiv:2604.07822. [Paper][R07]; [inspected full text v1](https://arxiv.org/html/2604.07822v1). Full v1 and current v2 abstract/metadata inspected. Supports task-dependent systematic/depth generalization and an overthinking limitation; it is not a universal guarantee of beneficial extra cycles.

**[R08] Bahram Behzadian, Brett Daley, Gopeshh Subbaraj and Houssam Nassif. _UPI–TRM: Unrolled Policy Iteration for Tiny Recursive Models._ ICLR workshop, 2026.** [Author-hosted poster][R08]; [OpenReview record](https://openreview.net/forum?id=rIzREKws05). The poster text and rendered PDF page were inspected; the OpenReview full-paper route was not accessible in this research session. Checker-feedback training is close prior art; limited Sudoku experiments do not establish transfer to semantic compilation. This plan does not claim a faithful UPI–TRM replication.

**[R26] Jacob Fein-Ashley and Paria Rashidinejad. _Solve the Loop: Attractor Models for Language and Reasoning._ 12 May 2026.** arXiv:2605.12466. [Paper][R26]; [HTML v1](https://arxiv.org/html/2605.12466v1). Full text inspected. Adjacent implicit/fixed-point alternative and warning that recurrent formulations can internalize solutions rather than exploit ever more iterations.

### 12.3 Program synthesis, tools, libraries and external memory

**[R09] Scott Reed and Nando de Freitas. _Neural Programmer-Interpreters._ 2015.** arXiv:1511.06279. [Paper][R09]. Abstract inspected. Establishes recurrent program interpretation with reusable program structure; not evidence that the present language front end works.

**[R10] Matej Balog, Alexander L. Gaunt, Marc Brockschmidt, Sebastian Nowozin and Daniel Tarlow. _DeepCoder: Learning to Write Programs._ 2016; ICLR 2017.** arXiv:1611.01989. [Paper][R10]. Abstract inspected. Informs the learned-operator-prior/typed-search baseline. The plan specifies its local baseline algorithm directly; it does not claim to reproduce an inspected official implementation.

**[R11] Jacob Devlin, Jonathan Uesato, Surya Bhupatiraju, Rishabh Singh, Abdel-rahman Mohamed and Pushmeet Kohli. _RobustFill: Neural Program Learning under Noisy I/O._ 2017.** arXiv:1703.07469. [Paper][R11]. Abstract inspected. Relevant neural synthesis from examples; different domain and information condition.

**[R12] Kavi Gupta, Peter Ebert Christensen, Xinyun Chen and Dawn Song. _Synthesize Execute and Debug: Learning to Repair for Neural Program Synthesis._ 2020; NeurIPS 2020.** arXiv:2007.08095. [Inspected full text v2][R12]. Direct precedent for execution-guided candidate repair; motivates a serious discrete-feedback competitor, not an artificially weak one-shot baseline.

**[R13] Xinyun Chen and colleagues. _Latent Execution for Neural Program Synthesis._ 2021.** arXiv:2107.00101. [Paper][R13]. Abstract inspected. Adjacent precedent for learned latent execution representations; no claim that the specific persistent-state contrast is absent from all follow-up work.

**[R14] Kevin Ellis and colleagues. _DreamCoder: Growing generalizable, interpretable knowledge with wake-sleep Bayesian program learning._ 2020/2021.** arXiv:2006.08381. [Paper][R14]. Abstract inspected. **Gabriel Grand, Lionel Wong, Maddy Bowers, Theo X. Olausson, Muxin Liu, Joshua B. Tenenbaum and Jacob Andreas. _LILO: Learning Interpretable Libraries by Compressing and Documenting Code._ 2023; ICLR 2024.** [Inspected full text v4](https://arxiv.org/html/2310.19791v4). Both establish reusable-library learning as prior art; library discovery, recognition and storage remain separate costs.

**[R15] Luyu Gao, Aman Madaan, Shuyan Zhou, Uri Alon, Pengfei Liu, Yiming Yang, Jamie Callan and Graham Neubig. _PAL: Program-aided Language Models._ 18 November 2022; revised 27 January 2023.** arXiv:2211.10435. [Paper][R15]. Abstract inspected. Supports separating semantic interpretation from exact execution; its comparison conditions do not justify an unequal-tools experiment here.

**[R18] Sebastian Borgeaud and colleagues. _Improving Language Models by Retrieving from Trillions of Tokens._ 2021/2022.** arXiv:2112.04426. [Paper][R18]. Abstract inspected. RETRO is evidence for external-memory augmentation at much larger scale, not a 20 MB standalone-knowledge demonstration.

**[R35] Microsoft Research, including Sumit Gulwani. _Flash Fill (Excel Feature in Office 2013)._ Project established 2012; deployed feature described for Office 2013.** [Official project page][R35]. Project description inspected. Historical deployed programming-by-example precedent; not a benchmark-equivalent implementation for this task.

**[R36] Abdelrahman S. A. Mohamed and co-inventors; original assignee Microsoft Technology Licensing LLC. _Neural network for program synthesis._ US20180275967A1, published 27 September 2018.** [Patent disclosure][R36]. Description and bibliographic record inspected through Google Patents. Used only as technical prior-art context; no legal-status or claim-scope opinion.

**[R37] Kiarash Rahmani and co-inventors; original assignee Microsoft Technology Licensing LLC. _Multi-modal program inference._ US11934801B2, published 19 March 2024.** [Patent disclosure][R37]. Description and bibliographic record inspected. Relevant combination of program inference and multiple input modalities; not a freedom-to-operate assessment.

### 12.4 Compression and strict artifact budgets

**[R16] OpenAI. _Parameter Golf_ and _What Parameter Golf taught us._ 2026; retrospective published 12 May 2026.** [Official repository][R16]; [retrospective](https://openai.com/index/what-parameter-golf-taught-us/); [inspected MLX training example](https://raw.githubusercontent.com/openai/parameter-golf/main/train_gpt_mlx.py). README, challenge conditions, retrospective and relevant MLX serialization/dequantization code inspected. MIT repository. The challenge's multi-H100 training setting and BPB objective are not this laptop's task protocol; the root example is not automatically the strongest submission.

**[R17] Prashanna Mani Paudel and Shivanand Venkanna Sheshappanavar. _Parameter Golf: What Really Works?_ 1 July 2026; revised 5 September 2026.** arXiv:2607.01517. [Paper][R17]; [inspected full text v1](https://arxiv.org/html/2607.01517v1). Full v1 and current v2 abstract/metadata inspected. Retrospective associations motivate classical/neural controls; they are not randomized causal evidence that an n-gram addition always helps.

**[R19] Microsoft researchers. _BitNet b1.58 2B4T Technical Report._ 16 April 2025.** arXiv:2504.12285. [Inspected full text v1][R19]. Evidence for trained low-bit models and specialized inference implementations; not proof of this proposed model's M3 training path or accuracy after quantization.

**[R20] Zechun Liu and colleagues. _ParetoQ: Improving Scaling Laws in Extremely Low-bit LLM Quantization._ 4 February 2025.** arXiv:2502.02631. [Paper][R20]. Abstract inspected. Informs the caution that lower bitwidth regimes can have different training behavior. No numerical gain is assumed here.

**[R21] Rui Fang, Hsi-Wen Chen, Chia-Ming Cheng and Ming-Syan Chen. _LoopQ: Quantization for Looped Language Models._ 8 May 2026; revised 12 October 2026.** arXiv:2605.16343. [Inspected v2 full text][R21]. **Nux Li. _Quantizing Looped Transformers: Feedback Exposure and Calibration Blindness._ 25 September 2026.** [Full text](https://arxiv.org/html/2609.30820); [stable record](https://arxiv.org/abs/2609.30820). Full-text mechanisms inspected. Direct prior art for recurrent-trajectory calibration and loop-entry sensitivity; quantitative outcomes are not imported into the laptop forecast.

**[R22] Zhixiong Zhao, Zukang Xu, Xing Hu, Lifeng Liu, Guangyu Sun and Dawei Yang. _LoopQuant: Accurate and Efficient Post-Training Quantization for Looped Transformers._ 12 October 2026.** arXiv:2610.16368. [Inspected full text v1][R22]. Very recent direct prior art. Its existence changes novelty assessment; implementation availability and reported efficiency were not independently verified.

### 12.5 Model assets and local execution

**[R23] Hugging Face TB. _SmolLM2-135M model card._ 2024 release; card accessed 2026.** [Card][R23]. Card evaluation, training and Apache-2.0 license fields inspected. Historical warning about small-model breadth; not selected as the strongest contemporary reference.

**[R24] Rui-Jie Zhu and colleagues. _Scaling Latent Reasoning via Looped Language Models_ (Ouro). 2025; v2 accessed 2026.** [Inspected full text v2][R24]; [project](https://ouro-llm.github.io/); [Ouro-2.6B asset](https://huggingface.co/ByteDance/Ouro-2.6B). Paper and project inspected; the asset link was identified, not executed. Different pretraining expenditure and recurrence implementation; gate its exact artifact/license/backend before a direct comparison.

**[R25] Nanbeige LLM Lab. _Nanbeige4.2-3B: Unlocking Agentic Capabilities in a Compact Model._ Model card and linked technical report, July 2026.** [Inspected card][R25]; [report record](https://arxiv.org/abs/2607.22083). Card inspected, including the distinction between non-embedding and total parameters, Apache-2.0 label, native thinking/tool format and loading instructions. Linked report not independently reproduced; model-name size is not complete tensor size.

**[R27] Qwen Team. _Qwen3.5-4B._ 2026 model card.** [Card][R27]; [inspected MLX implementation](https://raw.githubusercontent.com/ml-explore/mlx-lm/main/mlx_lm/models/qwen3_5.py). Card and relevant recurrent/attention source inspected. A contemporary hybrid local reference; support must be verified for the pinned model and runtime together.

**[R28] Google. _Gemma 4 E2B instruction-tuned model card._ 2026.** [Card][R28]; [inspected MLX implementation](https://raw.githubusercontent.com/ml-explore/mlx-lm/main/mlx_lm/models/gemma4.py). Card and model wrapper inspected. The card distinguishes effective from total parameters and lists Apache-2.0; complete tensor accounting matters. No M3 execution was performed.

**[R29] Liquid AI. _LFM2-1.2B-Tool model card._ 2026.** [Card][R29]; [inspected MLX implementation](https://raw.githubusercontent.com/ml-explore/mlx-lm/main/mlx_lm/models/lfm2.py). Card/source inspected. Optional specialized tool-use reference; its custom license requires explicit review rather than a blanket open-source assumption.

**[R30] John T. Halloran. _Nanbeige4.2-3B on Apple Silicon: Fixing Deployment Bugs and Decreasing Looped Transformer Memory Overhead._ August 2026.** arXiv:2608.13987v1. [Inspected full text][R30]. The record/header and body contain differing August dates. Useful cache/position/memory regression warnings from a different Apple machine; the paper's architectural interpretation of peak memory is not accepted as a universal law, and patch availability remains a revision-specific gate.

**[R31] Apple/MLX contributors. _MLX and MLX-LM documentation and source._ Continuously updated; accessed 16 October 2026.** [MLX-LM repository][R31]; [installation requirements](https://ml-explore.github.io/mlx/build/html/install.html); [checkpoint](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.checkpoint.html); [quantize](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.quantize.html); [QuantizedLinear](https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.QuantizedLinear.html). These official pages/source were inspected. They inform native installation, recomputation and frozen quantized-module constraints; they do not replace per-operation local tests. MLX-LM repository lists MIT.

**[R32] PyTorch contributors. _MPS backend and synchronization API documentation._ Version 2.14 documentation accessed 16 October 2026.** [Backend/API index][R32]; [synchronize](https://docs.pytorch.org/docs/2.14/generated/torch.mps.synchronize.html). Official documentation inspected after following the stable-page redirect. Describes available APIs, not proof that every proposed operation has a faithful MPS implementation on the user's installed version.

### 12.6 Evaluation datasets

**[R33] Arkil Patel, Satwik Bhattamishra and Navin Goyal. _Are NLP Models really able to Solve Simple Math Word Problems?_ NAACL 2021; SVAMP repository.** [Official repository][R33]; [license inspected](https://raw.githubusercontent.com/arkilpatel/SVAMP/main/LICENSE); [paper](https://aclanthology.org/2021.naacl-main.168/). Repository documentation and MIT license inspected; paper linked for provenance, not claimed as full-text inspection. Informs the whole-challenge evaluation and no-training-on-SVAMP rule.

**[R34] Jacob Austin and colleagues / Google Research. _Program Synthesis with Large Language Models_ and MBPP dataset. 2021.** [Dataset card][R34]; [official README inspected](https://raw.githubusercontent.com/google-research/google-research/master/mbpp/README.md). Card and original split instructions inspected; card lists CC-BY-4.0. The proposed sanitized DSL-compatible adaptation has a new, explicitly limited evaluation protocol and must not be reported as standard unrestricted MBPP.

### 12.7 Search scope, updates and blind spots

Searches covered equivalent mechanisms and terminology rather than only the phrase “micro-brain”: shared-depth/Universal Transformers; Neural GPU and program interpreters; recursive latent reasoning, halting, fixed-point and attractor models; TRM follow-up studies and negative results; program induction, typed synthesis, execution-guided repair, latent execution, library induction and program-aided models; strict-artifact compression, classical/neural blending and loop-aware quantization; contemporary compact/looped local models; Apple-Silicon operation support, cache semantics, memory and quantized training; dataset splits/licenses; and a bounded patent/deployed-system search.

Backward tracing identified foundational neural synthesis and recurrence work. Forward tracing materially changed the plan by adding the 2026 autoregressive-TRM negative study, checker-feedback prior art, current looped agent references and multiple loop-quantization competitors. Final version checks identified the 12 October LoopQ revision, the August revision of Loop/Think/Generalize and the September Parameter Golf retrospective revision. The full texts and abstracts actually inspected are distinguished above; no uninspected later full text is silently treated as read.

The final targeted searches did not justify replacing the main controlled question, but they did change baseline selection, quantization novelty claims and compatibility gates. This is a coverage judgment, not a proof that no more prior art exists. Remaining blind spots include very recent unindexed work, non-English literature, private industrial implementations, undocumented benchmark contamination, complete patent families/claim construction and execution on the exact target device. Access to one OpenReview paper was limited; its author poster was used instead. Public pages and raw source were read through the research tools, but repositories were not built or model weights downloaded during preparation.

The essential research decisions are made in this document. Execution-time version pinning and hardware discovery remain necessary because they concern the actual machine and mutable assets, not because the coding agent is expected to redo the landscape investigation from scratch.

### 12.8 Stable reference links

[R01]: https://arxiv.org/abs/1807.03819
[R02]: https://arxiv.org/abs/1511.08228
[R03]: https://arxiv.org/abs/2510.04871
[R04]: https://arxiv.org/abs/2502.05171
[R05]: https://arxiv.org/abs/2507.10524
[R06]: https://arxiv.org/abs/2603.08082
[R07]: https://arxiv.org/abs/2604.07822
[R08]: https://pages.cs.wisc.edu/~hous21/presentations/ICLRW_26.pdf
[R09]: https://arxiv.org/abs/1511.06279
[R10]: https://arxiv.org/abs/1611.01989
[R11]: https://arxiv.org/abs/1703.07469
[R12]: https://arxiv.org/html/2007.08095v2
[R13]: https://arxiv.org/abs/2107.00101
[R14]: https://arxiv.org/abs/2006.08381
[R15]: https://arxiv.org/abs/2211.10435
[R16]: https://github.com/openai/parameter-golf
[R17]: https://arxiv.org/abs/2607.01517
[R18]: https://arxiv.org/abs/2112.04426
[R19]: https://arxiv.org/html/2504.12285v1
[R20]: https://arxiv.org/abs/2502.02631
[R21]: https://arxiv.org/html/2605.16343v2
[R22]: https://arxiv.org/html/2610.16368v1
[R23]: https://huggingface.co/HuggingFaceTB/SmolLM2-135M
[R24]: https://arxiv.org/html/2510.25741v2
[R25]: https://huggingface.co/Nanbeige/Nanbeige4.2-3B
[R26]: https://arxiv.org/html/2605.12466v1
[R27]: https://huggingface.co/Qwen/Qwen3.5-4B
[R28]: https://huggingface.co/google/gemma-4-E2B-it
[R29]: https://huggingface.co/LiquidAI/LFM2-1.2B-Tool
[R30]: https://arxiv.org/html/2608.13987v1
[R31]: https://github.com/ml-explore/mlx-lm
[R32]: https://docs.pytorch.org/docs/2.14/mps.html
[R33]: https://github.com/arkilpatel/SVAMP
[R34]: https://huggingface.co/datasets/google-research-datasets/mbpp
[R35]: https://www.microsoft.com/en-us/research/project/flash-fill-excel-feature-office-2013/
[R36]: https://patents.google.com/patent/US20180275967A1/en
[R37]: https://patents.google.com/patent/US11934801B2/en

---

**End of standalone handoff.** Begin execution at W00. No positive result, universal novelty claim or target-laptop compatibility result is assumed.
