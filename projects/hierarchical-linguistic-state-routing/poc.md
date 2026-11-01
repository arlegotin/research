# Explicit linguistic structure inside a language model: research and local proof-of-concept plan

**Research date:** 23 October 2026.  
**Target machine:** one MacBook Pro with an Apple M3 Max; actual memory, storage, operating system, and sustained performance must be discovered.  
**Status:** researched experimental design, not a demonstrated new architecture or a positive experimental result.  
**Deliverable contract:** this file is a standalone handoff to a coding agent. All commands invoking `-m linglab` and all `linglab/` files below are **proposed project interfaces to implement**, not existing packages or already executed commands. External package APIs must be checked against the dependency lock created on the target machine.

## 1. Executive assessment and selected direction

### Verdict

The broad idea is substantially established: neural systems have explicitly supervised linguistic tasks at different depths, passed predicted linguistic labels between layers, incorporated syntactic and semantic graphs into attention, and generated language through structured intermediate representations. A model with six named modules would not, by that fact alone, constitute a new scientific contribution. Historical hierarchical multitask models, LISA, structured language models, concept bottlenecks, and the recent GiLT model are consequential prior art. [R01], [R02], [R03], [R04], [R07], [R08], [R09], [R10], [R11]

**Selected direction:** test whether **finite, predicted linguistic side channels, used at different depths of an autoregressive language model, improve held-out language prediction and controlled generalization beyond equally supervised, well-tuned simpler alternatives**. The primary implementation uses four text-supported channels: morphology, coarse dependency syntax, entity type, and discourse information status. It explicitly routes their predictions into subsequent computation. Hard categorical communication is compared with soft prediction feeding, auxiliary heads alone, generic adapters, and tree-based attention supervision.

This is initially an **experimental synthesis of established ideas**. The intended contribution is a reproducible finding about when explicit linguistic communication helps, what explains the effect, and what it costs. A narrowly differentiated method might emerge, but neither architectural novelty nor improvement is currently established. Stop describing it as a new method if a closer equivalent is found; a carefully controlled replication or negative result remains valuable.

The primary real-data experiment adapts the same public **`Qwen/Qwen3-0.6B-Base`** checkpoint across methods, using GUM annotations and official document splits. GENTLE supplies a separate distribution-shift evaluation. A small controlled language experiment establishes causal correctness and compositional sensitivity without relying on unknown pretraining contamination. A contemporary **`Qwen/Qwen3.5-0.8B-Base`** challenger must be screened; Qwen3 is the clean attribution platform, not an assumed current winner. [R23], [R24], [R31], [R32]

### What this preserves and what it deliberately narrows

The original intent—make linguistic distinctions affect computation at different architectural locations—is preserved. This is not merely prompting a model to discuss grammar. Predictions enter the model's forward computation, and interventions can remove or replace those routes.

The initial experiment does **not** implement a complete six-level theory of language. Entity type is a narrow semantic representation; information status is a narrow discourse/pragmatics representation. Phonology and acoustic phonetics require a separate modality-aware branch. Text can support pronunciation-related predictions, but an experiment without acoustic observations cannot establish sensitivity to distinctions between waveforms with the same transcript. An explicit audio continuation is specified in Section 11.3; it is not silently counted as completed by the text experiment.

### Claims the program can support

| Claim | Required evidence | Claim it does not license |
|---|---|---|
| A finite linguistic route is useful in this adaptation setting | Better protected-test score than the development-selected strongest eligible comparator; matched information; uncertainty and cost accounting | All LLMs need explicit linguistic modules |
| Direct cross-level connections add value | Routed versus no-direct-route comparison, plus active-capacity controls | Linguistic levels are independent human-like faculties |
| Named states influence computation | Valid state interventions, route removal, and predicted-direction behavior | The whole model is interpretable or exclusively uses those states |
| Improvement transfers beyond a toy language | Real GUM test result and separately reported GENTLE result | General improvement across languages, deployments, or model scales |
| The design is locally feasible | Target-machine forward/backward, memory, timing, and resume checks | Estimated memory or Linux arithmetic proves M3 compatibility |
| All six linguistic levels are computationally useful | Additional acoustic/phonological experiments, then a controlled combined system | Four text channels establish a six-level architecture |

### The first consequential decisions

1. Reproduce the pinned corpus checks reported below. The released data does contain resolved information status, but also contains `auto`; mask the unresolved intervals and verify tokenizer-specific alignment.
2. Validate causal alignment, the loss harness, and target-machine backward support before downloading or adapting multiple large assets.
3. Establish headroom against ordinary adaptation, auxiliary supervision, and Tree-Planted Transformer supervision before expanding the proposed mechanism.
4. Treat a failure to beat those controls as an informative outcome. Larger scale is justified only by a diagnosed limitation and a prediction it can resolve.

## 2. Problem statement, scope, assumptions, and hypotheses

### 2.1 Beneficiaries, task, and the possible bottleneck

The immediate beneficiaries are researchers and developers of compact local language models who need reliable language prediction under limited adaptation data, unusual constructions, or changes in genre. No specific commercial application or user population was supplied. Accordingly, this plan does not assume that an annotation-heavy design has economic value in a product.

The existing workflow is to adapt a pretrained causal language model on text and evaluate its predictions. The proposed change adds supervised internal representations and uses their **predictions**, rather than their gold labels, during ordinary inference.

The suspected bottleneck is inefficient or fragile use of reusable linguistic relationships under limited adaptation data. Existing papers demonstrate both benefits from structure and conditions where a stronger ordinary model removes those benefits. They establish a credible research question, not evidence that this bottleneck exists in the selected checkpoint/corpus pair. The pilot must establish that pair's actual errors and improvement headroom. [R05], [R11], [R12], [R13]

Inputs are a raw text prefix and the normal model/tokenizer configuration. Training additionally supplies linguistic labels. Outputs are a normalized next-token distribution and optional diagnostic state predictions. **Normal inference receives no gold parses, entity boundaries, document summaries, discourse labels, or future text.**

### 2.2 Operational definitions

| Original linguistic level | Meaning for this project | Initial treatment and limitation |
|---|---|---|
| Phonetics | Acoustic realization: phones, timing, spectral/prosodic properties, speaker-conditioned variation | Outside text core; requires waveforms and independent acoustic evaluation |
| Phonology | Contrastive sound categories and permitted sound patterns | Audio continuation; a dictionary/G2P-only result would be a symbolic pronunciation result |
| Morphology | Inflectional features and grammatical word class | UPOS plus selected UD morphological attributes; this does not cover all derivation or nonconcatenative morphology |
| Syntax | Relationships between words | Dependency relation and a coarse head-location category; full graph structure is represented by a stronger comparator, not assumed to be captured by these coarse states |
| Semantics | Meaning-bearing distinctions | Entity type in real text; roles, negation, and denotation in controlled diagnostic language; no claim of complete semantics |
| Pragmatics | Interpretation conditioned on discourse and communicative context | Entity information status in real text; given/new reference choices in diagnostics; no claim about all intent, implicature, politeness, or social competence |

The six labels are analytical categories, not a proof that language or brains decompose into six serial stages. Layers can share information, and syntax and meaning can constrain each other. Quarter-depth placement is a testable design choice. Investigate reverse/shared placement as an ablation; do not describe it as known neural anatomy. Hierarchical structure can also emerge under an ordinary language-modeling objective. [R13], [R14], [R15]

### 2.3 Known facts, assumptions, and open questions

**Source-supported facts:** explicit hierarchical supervision and prediction feeding exist; structured generation has strong predecessors; soft concept channels can carry information beyond their intended concepts; some syntactic bottlenecks hurt document modeling; M3 Max configurations vary; small open-weight base checkpoints and Apple GPU training frameworks exist. [R01], [R02], [R07], [R19], [R30], [R31], [R33], [R34]

**Design assumptions to test:** the chosen labels are predictable enough from prefixes; they capture useful distinctions missing from ordinary adaptation; hard communication preserves enough ambiguity; the shared backbone learns to use the channels; benefits can exceed training and inference overhead; corpus annotations are sufficiently consistent.

**Noncritical defaults:** local research use, English first, no paid API or remote GPU, native Python, existing open assets, frozen backbone weights plus trainable adapters for the main adaptation experiment. These are scope choices, not claims that English or adapters are intrinsically best.

**Consequential unknowns:** actual RAM/OS/storage; tokenizer-aligned label coverage; compatible released dependency versions; effective long-context support; baseline error rates; variance across documents and training runs; pretraining contamination; utility of cross-level routes. They are explicit gates below.

### 2.4 Hypotheses and falsifiers

| ID | Hypothesis and predicted observation | Supporting rationale | Alternative explanation | Evidence weakening it |
|---|---|---|---|---|
| H1 | Typed predicted side channels reduce protected-test bits per byte relative to the strongest development-selected local baseline | Linguistic supervision/structure has helped some tasks | More trainable parameters, compute, annotation information, or tuning | Effect disappears under active-capacity, annotation, and compute controls; interval excludes the pilot-relevant benefit |
| H2 | Direct cross-level communication helps beyond independent linguistic feedback at the same taps | Dependencies between grammatical and discourse properties | Earlier residual injection already supplies everything; extra projections add capacity | No-direct-route matches or wins; generic routing matches typed routing |
| H3 | Finite categorical states give a useful accuracy/intervention tradeoff | Finite messages exclude arbitrary real-valued probability-shape communication | Hard decisions discard uncertainty and propagate mistakes | Soft feeding reliably wins, especially on ambiguous prefixes, without compensating intervention value |
| H4 | The model uses a named route in a linguistically appropriate way | Designed computational access to the states | Off-manifold corruption, generic activation disruption, or residual bypass | Corrected states fail to improve expected decisions; norm-matched/random controls behave identically |
| H5 | Gains survive a meaningful distribution shift | Explicit features may be reusable across text types | GUM-specific style or repeated source documents | GENTLE/independent diagnostic results regress; benefit depends on one genre or annotation artifact |
| H6 | Local cost is compatible with a practical compact-model benefit | Small additions may be cheap relative to the backbone | Large logits, lost fusion, annotation cost, or sequential state work dominate | End-to-end cost is dominated with no defensible benefit; contemporary ordinary model offers a better tradeoff |

H1 is the primary predictive claim. H2–H4 are attribution claims. H5 is external validity. H6 is practical value. A positive H1 alone does not establish the others.

### 2.5 Theoretical limits and headroom

Let the available prefix be \(X_{\le t}\), the next token be \(Y\), and a predicted state be \(Z=g_\theta(X_{\le t})\). Once parameters are fixed, a deterministic state adds no new observation:

\[
H(Y\mid X_{\le t},Z)=H(Y\mid X_{\le t}).
\]

The potential benefit is therefore an inductive bias, a better use of finite computation/capacity, or a change in optimization and sample efficiency. It is not extra information at inference. Supervised labels do add information during training; that is why equally supervised baselines are essential.

For categorical fields with cardinalities \(C_f\), a single message has at most \(\sum_f\log_2 C_f\) bits of discrete identity. This bounds that message, **not the information in the residual model**. Hard states remove the continuous probability-shape channel but do not guarantee semantic purity, eliminate all unintended coding through errors, or make the whole network identifiable.

An oracle-label experiment is not an achievable headroom estimate. A label computed using a suffix can reveal the answer. In a two-continuation example with the same prefix, a future-derived label supplies one bit and reduces loss from \(\ln 2\) to zero; a prefix-only predictor cannot achieve that reduction. This exact arithmetic was checked during preparation (Appendix A). Use oracle interventions only to diagnose sensitivity, clearly separated from normal inference.

Measure useful headroom in the pilot using: ordinary adaptation errors; held-out linguistic prediction quality; development losses of predicted-only versus auxiliary-only variants; stratification by feature predictability and ambiguity; and the strongest baseline's learning curve. If the baseline is at the known optimum on a synthetic task, that task cannot discriminate improvements. Increase task complexity only on development data and freeze a fresh test generator afterward.

For cost, let \(C_{prep}\) be project-specific annotation/preprocessing cost, \(C_{train}\) adaptation cost, and \(c_{infer}\) cost per processed token. Total cost at usage \(U\) is \(C_{prep}+C_{train}+Uc_{infer}\). Report the frontier over \(U\); do not claim that a small accuracy gain automatically compensates for any overhead. Pretrained asset creation is a reusable external investment and must be disclosed separately.

## 3. Research findings and precise novelty assessment

### 3.1 Closest prior art

Published scores below are used to understand mechanisms and limitations, not as uncontrolled numbers to beat on a different dataset.

| Prior art | Mechanism and exact overlap | Evidence and limitation | Reuse / local comparator role |
|---|---|---|---|
| Deep multitask learning; JMT [R01], [R02] | Supervise different linguistic tasks at different depths; feed predicted label embeddings upward | Mostly bidirectional, older tagging/parsing/NLU settings | Reimplement the principles in the same causal backbone; label this a causal adaptation, not an exact original reproduction |
| LISA [R03] | Joint POS, syntax and semantic-role tasks; syntactically supervised attention | Full-input SRL rather than causal text prediction | Defeats novelty of named linguistic heads or syntax-aware attention |
| Sluice networks; structured attention [R04], [R06] | Learned task/layer sharing; structured intermediate distributions | Different tasks and older implementations | Gates, soft structure, and learned placement are established alternatives |
| RNNG; Transformer Grammars [R05], [R07] | Explicit tree structure and recursive composition inside generative models | Structure inference costs; TG's document-memory tradeoff | Reference mechanisms; no comparison of oracle/joint probabilities with raw conditional probabilities |
| Pushdown Layers; DTG [R08], [R09] | Incremental stack/dependency state changes Transformer computation | Requires structural training and careful inference | Strong graph/state alternatives; implementation/platform fidelity must be established |
| Tree-Planted Transformers, TPT [R10] | Train causal attention against next-word tree-distance targets; ordinary inference | Syntactic evaluation benefits do not prove all-level competence | Mandatory cheap structured comparator; no test-time parser |
| GiLT, 2026 [R11] | Incrementally predicted syntactic or semantic dependency graphs modulate attention, without structural tokens | Latent-graph search; published experiments use different data/hardware | Most important recent graph comparator; joint use of several graph formalisms is already suggested future work |
| Semantic dependency infusion and oracle graph ensembles [R16], [R17] | Incorporate semantic graphs or several linguistic formalisms into pretrained models | Full-input/oracle availability can explain gains | Compare information availability; do not call semantic graph fusion new |
| Concept bottlenecks, leakage work, CB-LLM [R18], [R19], [R20] | Named concepts affect outputs; Havasi et al. already use dependent concept predictions and a latent side channel; CB-LLM generation combines supervised concepts with unsupervised residual features | Havasi trains annotated concept conditionals and marginalizes samples; CB-LLM's generation proxy is another LM scoring generated text, not held-out likelihood | Our predicted-only hard chain and multiple depth injections differ in detail; neither dependent concepts nor residual routes are new |
| Steerling technical report, 2026 [R21] | Concept-mediated language prediction with an additive concept/residual design | Different scale, training recipe, and interpretability claims; not reproduced here | Contemporary context; does not establish originality of concept-mediated generation |
| SupCBM, 2024 [R53] | Hierarchical concept inventories, hard-concept leakage analysis and supervised concept-based prediction | Image-classification setting with a specified concept/class mapping | Further rules out novelty from the phrase “hard hierarchical concepts”; no direct causal-LM comparison |
| Hierarchical Audio Codec, 2025 [R22] | Factorizes acoustic, phonetic, and lexical speech representations | Audio codec/reconstruction setting | Strong prior art for the proposed later sound-level branch |
| Hashimoto/Xiong/Socher patent publication [R29] | A linguistic hierarchy of neural tasks with inter-layer embeddings | Patent text inspected as technical prior art only | Broad hierarchical-task idea is also described in patent literature; no legal conclusion is offered |

The practical strongest simple comparison is not merely an unmodified language model. It is **ordinary adaptation plus the same auxiliary labels, and TPT plus those labels**, with suitable tuning. The closest architectural comparison is soft hierarchical prediction feeding. The strongest recent structural reference is GiLT. These answer different questions and should not be collapsed into one weak baseline.

### 3.2 Conflicting results and why they matter

There is evidence both for and against adding explicit structure. Syntax-augmented extraction models can depend heavily on parse quality; strong ordinary adaptation can erase syntax gains in other tasks. TG shows why an isolated syntactic improvement can coexist with worse document prediction. These results are compatible: they differ in full-input versus prefix information, gold versus predicted structure, amount of pretraining, sentence versus document context, and objective. [R07], [R12], [R13a]

Theoretical and controlled-language studies also show that ordinary Transformers can learn hierarchical relationships without a prescribed linguistic decomposition. A 2026 subgrammar study reports that a pretraining advantage may be limited to models small relative to the grammar. This motivates matched objectives and multiple meaningful local scales; it does not prove explicit structure is useless. [R13], [R14], [R15]

A September 2026 structural-transfer study finds that nonlanguage pretraining can lower LM loss while additional language data is stronger and broader linguistic transfer remains inconsistent. Its GPT-2/multilingual setup is different, but reinforces the need to separate prediction quality from competence and to charge additional training data/computation. [R54]

### 3.3 Six explicit answers about novelty

1. **What already exists?** Hierarchical linguistic supervision, prediction feeding, syntax-aware attention, semantic-graph fusion, explicit generative grammar state, concept bottlenecks, and multilevel acoustic factorization.
2. **What can be reused?** Public base weights/tokenizers; released GUM/GENTLE annotations; UD and Universal Anaphora specifications; PyTorch/Transformers/PEFT; established attention-supervision objectives; standard evaluation and bootstrap components. Reuse source code only when its license permits it.
3. **What is different here?** A specific controlled investigation of predicted finite channels spanning morphology, coarse syntax, entity type and information status, with matched annotation access, direct-route ablation, prefix-causality tests, and measured local costs. The exact combination may differ from inspected implementations, but combination alone is not demonstrated methodological novelty.
4. **Why might it be valuable?** It can determine whether explicit communication adds anything beyond supervision and generic extra capacity, and identify predictable conditions under which it helps or fails. That empirical distinction can prevent expensive, unjustified six-module systems.
5. **What isolates the contribution?** Hard routed versus hard no-direct-route, soft routed, auxiliary-only, active untyped routing, and TPT comparisons; predicted-only normal evaluation; valid state interventions; shared training data and token budgets.
6. **What invalidates a novelty claim?** A prior implementation or publication with the same information flow and objectives; gains fully explained by a known method; a best variant equivalent to JMT/TPT/CB-LLM; or a difference consisting only of checkpoint, corpus, or hardware. Revise the claim to replication/application/engineering accordingly.

**Novelty status:** broad concept—known; selected implementation—plausibly distinct engineering synthesis; scientific contribution—unresolved until experiments; patentability/freedom to operate—not assessed.

### 3.4 Alternatives considered and selection rationale

| Direction | Scientific value and main risk | Decision |
|---|---|---|
| Six hard serial stages from sounds to intent | Strong structural commitment; annotation scarcity, information loss, difficult attribution | Defer; no evidence supports this mandatory serial order |
| Linguistic auxiliary losses only | Cheap, strong, straightforward causal training | Mandatory baseline and likely outcome if architecture adds no value |
| Training-only structural attention supervision | Strong structure-sensitive baseline with low inference cost | Mandatory TPT implementation |
| Finite typed side channels at several depths | Directly tests architectural use while preserving a residual path | **Primary mechanism** |
| Full multi-formalism graph inference | More expressive; close to GiLT; expensive search and licensing/implementation complications | Comparator/conditional extension, not the first infrastructure project |
| Tokenizer/morphological segmentation redesign | Relevant to morphology but changes sequence length and representation budget | Separate later study if tokenization is diagnosed as the bottleneck |
| Neuron constraints, new learning rules, or non-Transformer primitives | Legitimate architectural levels, but no specific mechanism in the supplied idea requires them | Do not invent a neuron-level hypothesis merely to appear fundamental; test later only with a causal prediction |
| Prompting/orchestrating linguistic agents | Cheap to prototype but does not test the intended internal architecture | Outside the selected mechanism |

This selection follows the causal question, not a claim that Transformers are the only valid architecture. The backbone is held fixed to isolate the proposed communication mechanism. A recurrence-based or hybrid replication becomes valuable if the mechanism survives the initial controls.

## 4. Evaluation contract established before implementation

### 4.1 Datasets and provenance

#### Main real-text data: GUM

Use **UD r2.18** from `UniversalDependencies/UD_English-GUM` and matching `UniversalDependencies/UD_English-GENTLE`, together with GUM **V12.1.0** split metadata. Public files were downloaded and inspected during this research. Their document sets exactly match that GUM release's train/dev/test lists after excluding Reddit, and its GENTLE `test2` list. Freeze the SHA256 values in Appendix A and resolve full repository commit IDs at WP0; do not follow mutable `master`. The GUM website and some release prose retain older version labels, so the actual files and checksums govern. [R23], [R25], [R28]

The dataset provides morphology, dependencies, entity type and information status on the same text. This is preferable to combining unrelated datasets and claiming a joint hierarchy. The measured starting counts are below; final usable subtoken supervision and canonical UTF-8 byte counts must still be computed after alignment.

| Pinned split | Documents | Sentences | Syntactic words | Surface tokens | Mention openers | Unresolved `auto` mentions |
|---|---:|---:|---:|---:|---:|---:|
| GUM train | 197 | 11,314 | 200,223 | 196,696 | 56,551 | 508 |
| GUM dev | 30 | 1,575 | 28,119 | 27,602 | 7,909 | 71 |
| GUM test | 30 | 1,464 | 28,397 | 27,986 | 7,955 | 78 |
| GENTLE test | 26 | 1,334 | 17,799 | 17,619 | 5,681 | 11 |

These are **measured corpus/schema counts**, not model scores or independent statistical sample sizes. The inspection did not evaluate candidate models on protected text.

Licenses are heterogeneous across underlying texts. Assume internal, noncommercial research initially; record source-specific terms and the intended use. Exclude unavailable Reddit text rather than reconstructing it from another service. Exclude any document whose use cannot be established. An exclusion changes the named evaluation population and must appear in the split manifest before scores are inspected. Do not redistribute corpus text or model checkpoints solely because annotations have a permissive license. [R25], [R26]

`MISC.Entity` is described by a `# global.Entity` header: `GRP-etype-infstat-salience-centering-minspan-link-identity` in all inspected documents. Parse field names, not fixed offsets. Use only entity type and resolved information status for the proposed channels. Do not feed salience, Wikification identities, summaries, complete discourse trees, or gold entity IDs to the model. The released files still contain `infstat=auto`; these targets must be masked. [R25], [R27]

#### External shift set: GENTLE

Use `UniversalDependencies/UD_English-GENTLE` at the matching release, cross-checked with the 26 `GENTLE_*` document names in the GUM `test2` split. It follows the GUM annotation design and has eight unusual genres. It is a **test-only** set; never manufacture a development split from it after seeing model performance. Its small document count limits inference about arbitrary new genres. The downloaded aggregate license says **CC BY-NC-SA 4.0**, while the project's website grants CC BY 4.0 to annotations and preserves underlying-source restrictions. Treat these as distinct layers of rights; the local noncommercial research assumption does not establish unrestricted redistribution. [R24], [R26], [R28]

#### Controlled mechanism data

WP2 implements a bounded, explicitly synthetic English-like event language with complete derivations, described in Section 5.5. It tests causal alignment, agreement, role binding, negation and given/new reference. It is not evidence of natural-language fluency. Generate protected samples only after the development grammar, held-out combinations, and generator seed policy are frozen.

#### Optional external benchmarks

SyntaxGym provides an established syntactic diagnostic, conditional on verifying the selected suite's license and scoring protocol. BLiMP is relevant context, but availability does not by itself establish redistribution permission or contamination freedom. ReCOGS is a useful later semantic-composition experiment because its semantic exact-match evaluation avoids some original COGS representation artifacts. These are **extensions of the claim**, not replacements chosen after a negative primary result. [R40], [R41], [R42]

### 4.2 Splits and access discipline

1. Preserve official document IDs and split membership. Keep GUM test and all GENTLE results protected until the protocol lock.
2. Divide GUM development documents deterministically into `dev_tune` and `dev_calibration`, stratified by genre where possible, using a published seed and hash manifest. Select configurations on `dev_tune`, freeze those configurations, then use `dev_calibration` for variability/resource planning. Its one-document-per-genre strata cannot support within-genre resampling; Section 7 uses pooled source clusters. If calibration feedback changes configurations, record that it became additional tuning data. Neither subset is a final test.
3. A data steward script may read test bytes to validate syntax, hashes, legal availability and duplicate relationships. It must not run candidate models, select difficult examples, or return test scores. Record every such access.
4. Detect exact and near duplicates, shared source URLs, and repeated source documents across splits. Cluster related documents for uncertainty estimates. If a leakage repair moves/removes an item, do it by a rule frozen before modeling and disclose the departure from the official population.
5. Unknown pretraining contamination remains a limitation. Same-backbone comparisons reduce one confound but do not prove that public text was unseen. Fresh synthetic samples help establish mechanism behavior, not contamination-free real-world performance.
6. Learn feature vocabularies, class weights, normalization and token budgets from training/development only. Never select layers, task subsets, or a winning seed using protected data.

### 4.3 Primary metric and estimand

For method \(m\), training seed \(s\), and document \(d\), define

\[
b_{m,s,d}=\frac{-\sum_{t\in\mathcal S_d}\log p_{m,s}(x_t\mid x_{<t}\text{ in allowed context})}{B_d\ln 2},
\]

where \(\mathcal S_d\) contains each scored raw-text token exactly once and \(B_d\) is the number of UTF-8 bytes represented by those targets. Artificial BOS/EOS, padding and metadata are not bytes of text; report their losses separately. Maintain exact byte accounting when a token contains whitespace or multibyte characters. Within the primary comparison, all methods use the same tokenizer, text and context windows.

**Primary score:** mean document BPB over the preregistered retained GUM test documents, then mean over independent training seeds. Each document has equal weight. Also report byte-weighted BPB and genre-stratified scores, since equal-document weighting answers a different question from aggregate compression.

**Primary contrast:** \(\Delta=\overline b_{B^*}-\overline b_{typed}\), where \(B^*\) is the strongest eligible same-backbone comparator selected on development data under the declared cost regime. Positive \(\Delta\) favors the proposed method. Units are bits per byte; report absolute difference, relative loss reduction, and confidence intervals. If reporting token perplexity, compare it only within a common tokenizer.

The primary claim is conditional on the specified checkpoint, adaptation corpus, tuning policy and retained document population. A stronger statement that the method beats **every** mandatory baseline requires simultaneous pairwise evidence, not just beating the development winner.

GENTLE BPB is a separately reported external-shift endpoint. It cannot rescue a failed GUM primary claim without a new explicitly scoped confirmation. Synthetic challenge accuracy, grammatical preference, state quality and interventions are secondary/attribution endpoints.

### 4.4 Mandatory baselines and controls

Architecture-versus-supervision comparisons receive the same permissible annotations, text, tokenizer, target masking and context. B0/B1 deliberately establish frozen/text-only reference points. A historical method adapted to this setup must be named `project reimplementation/adaptation`, never presented as its published model.

| ID | Exact comparison role | Required configuration |
|---|---|---|
| B0 | Frozen base diagnostic | No adaptation; same scoring harness. Establish whether adaptation itself helps |
| B1 | Strong ordinary adaptation | LoRA and LM loss only; increase useful adapter capacity to bracket the proposal's active trainable parameter count; tune it independently |
| B2 | Equal-information auxiliary baseline | Evaluate independent heads (`u=0, A=0`) and routed heads (`u` retained, `A=0`); the latter keeps dependent prediction without LM residual feedback. Give freed capacity to useful LoRA/generic adapters. Select the stronger on development |
| B3 | TPT, then TPT plus auxiliary labels | Correct next-word distance supervision; tuning includes the combination with all proposed auxiliary labels. The stronger development variant is eligible as \(B^*\) |
| B4 | Soft hierarchical prediction feeding | Same tap schedule and named heads; feed probability-weighted embeddings; compare both ordinary joint gradients and a detached-soft variant when gradient flow is a confound |
| B5 | Hard states without direct cross-level links | Keep every side-state injection and tap; remove only the explicitly defined head-to-head inputs. Later layers may still receive earlier states through the residual stream; this limitation is intentional and reported |
| B6 | Active untyped routing | Same taps and auxiliary supervision, with generic learned continuous residual messages instead of categorical messages; match/bracket useful active parameters and record FLOPs/seconds |
| B7 | Current generic model challenger | Plain and, if tractable, auxiliary adaptation of `Qwen/Qwen3.5-0.8B-Base`; same raw-text endpoints and public information; separate from same-backbone attribution |
| C1 | Shuffled-label negative control | Shuffle training labels across eligible source events within length/frequency strata; preserve missingness and marginal counts. Never shuffle test targets |
| C2 | State/route removals | Inference interventions on the trained proposed model; distinguish destructive sensitivity tests from separately retrained ablations |

**Strong structural comparator beyond TPT:** GiLT must be addressed explicitly. First inspect its pinned code/license and attempt the supported mathematical path on a tiny CPU reference and MPS. Its published environment is CUDA-oriented, so native installation is not established. If source reuse is not licensed or supported, independently implement the published algorithm only for a separately declared structural-comparison extension. Validate tiny joint probabilities by enumeration; calibrate graph beams and include their full cost. A greedy graph-path score is not the published marginal likelihood. If faithful comparison is unavailable, the final claim must say **“better than the specified locally reproduced baselines”**, not “better than structural language models” or “state of the art.” Do not weaken GiLT and silently count that as a win. [R11]

A full reproduction of large-scale TG/GiLT/Steerling training is not required for the local component claim and is not assumed feasible. Their published results remain references with different resources and information. Inference feasibility of a large checkpoint is not the same as reproducing its training experiment.

### 4.5 Tuning and resource fairness

Use two declared views, because they answer different questions:

- **Information/token-matched attribution:** same pretrained snapshot, train text/labels, data order seeds, total LM target tokens, effective batch and checkpoint-selection policy. Report parameters, actual training compute and elapsed time; do not pretend added modules are free.
- **Measured cost frontier:** allow each method its efficient faithful backend and useful hyperparameters, and compare accuracy at actual end-to-end adaptation/inference costs. Include preprocessing and unsuccessful tuning expenditure. A baseline may use its saved overhead for more training.

Initial tuning actions are concrete starting points, not permanent search limits. For the main backbone, start LoRA rank 16 on attention projections, then evaluate active-capacity brackets using ranks 8 and 32 if required. Start learning-rate candidates \(10^{-5},5\times10^{-5},2\times10^{-4}\). If the best stable candidate is at a boundary, extend one multiplicative step in that direction. Select shared optimization defaults with B1 first, then permit each method to choose its own rate and auxiliary coefficient. A larger method must not get a systematically better search.

For each auxiliary group, use a normalized mean loss. Pilot coefficients corresponding to aggregate auxiliary-to-LM gradient-norm ratios near 0.1, 0.3 and 1 on fixed training calibration batches, measured on the **shared LoRA parameter set**, not classifier-only weights with structurally zero LM gradients. A zero shared LM norm makes that batch unusable for the ratio; report it and select another predeclared batch. Include zero in B1/B3. These are optimization probes, not required improvement targets. Freeze selected coefficients before confirmation. Diagnose conflicting gradients or collapsed heads rather than simply increasing their weight.

During development, evaluate at full-pass checkpoints, extending a horizon by doubling complete training passes while the best validation curve is still improving beyond its document uncertainty. Two consecutive clearly worse pass checkpoints can terminate a trial; a noisy plateau calls for variance calibration, not indefinite retraining. Report all evaluated checkpoints. For the main token-matched comparison, jointly select the strongest eligible baseline and its best common horizon from development, then tune the typed configuration at that horizon. Freeze **one common full-pass count, owned-target total and logical-step partition** for every method in that comparison, and score the checkpoint at that common horizon. Method-specific best horizons/checkpoint choices belong to the separate cost frontier. There is no protected-test early stopping.

At WP6 allocate method-specific tuning using measured run cost and observed development uncertainty. Every mandatory comparator gets the same base search opportunity and an extension when its optimum lies on an explored boundary. If resources prevent adequate baseline tuning, record an inconclusive competitiveness assessment. No wall-clock cutoff or paper quota substitutes for this decision.

### 4.6 Guardrails and practical significance

Report state prediction loss, balanced accuracy/F1, calibration, per-feature coverage, next-token BPB, syntax diagnostic accuracy, semantic denotation accuracy in the synthetic language, document copying/coreference-sensitive diagnostics, latency, throughput, sampled/verified peak memory, artifact size and total project cost.

Avoid a universal percentage required for success. Statistical evidence of a small effect and practical value are distinct. Without an application-specific utility function, report a Pareto tradeoff. “No statistically significant regression” does not demonstrate noninferiority. A claim of no meaningful regression requires a prespecified acceptable margin justified by an application; otherwise show the full interval and narrow the claim.

## 5. Mechanism and implementation specification

### 5.1 Computational graph

The default main model has frozen pretrained weights plus trainable attention LoRA, four state predictors and four message projections. Frozen weights still transmit activation gradients. Do not wrap their forward pass in `no_grad()` when trainable computation below them needs gradients.

For \(L\ge8\) blocks, use zero-based tap indices

\[
\ell=(\lfloor L/4\rfloor-1,\lfloor L/2\rfloor-1,\lfloor3L/4\rfloor-1,L-2).
\]

For Qwen3's 28 blocks this gives `(6, 13, 20, 26)`, leaving a block after the last injection. These taps are defaults to freeze or amend during development, not universal linguistic locations. Taps operate on **block outputs before the next block**. The normal final norm and LM head remain intact.

For group \(k\), field \(f\), prefix position \(t\), residual vector \(h_t\in\mathbb R^d\), and previous messages \(e_t^j\in\mathbb R^r\), define

\[
u_t^k=\sum_{j<k}R_{kj}e_t^j,
\qquad q_{t,f}^k=\operatorname{softmax}(W_{kf}\operatorname{LN}(h_t)+U_{kf}u_t^k+b_{kf}),
\]

\[
c_{t,f}^k=\operatorname{argmax}_c q_{t,f,c}^k,
\qquad e_t^k=\frac{1}{\sqrt{|F_k|}}\sum_{f\in F_k}E_{kf}[c_{t,f}^k],
\qquad \widetilde h_t=h_t+A_ke_t^k.
\]

Default message width is \(r=32\); its adequacy is a pilot question. \(R_{kj}\) is an \(r\times r\) learned matrix for every earlier group. \(U_{kf}\) maps the summed message to field logits. The **hard IDs are detached**: no straight-through estimator is used in the primary version. Auxiliary cross-entropy trains \(W,U,R,b\) and shared differentiable computation. LM loss directly trains \(E,A\) and shared adapters; it has no differentiable path through hard IDs to \(W,U,R\). Later auxiliary losses can also influence earlier embeddings. State predictions always come from the actual model, in training and inference.

Initialize \(A_k=0\) for exact branch-zero identity; initialize embeddings and predictor weights with the backbone's documented scale or a recorded small variance. Keep routing matrices small at initialization. At step zero, zero \(A\) can temporarily make some downstream gradients zero; require their expected activation after an optimizer step, not an impossible all-parameters-nonzero gradient test. Never initialize both sides of every learnable path to zero.

The no-direct-route condition sets \(u_t^k=0\) while retaining every injection. Soft feeding replaces the embedding lookup by \(\sum_cq_{t,f,c}^kE_{kf}[c]\). The detached-soft control blocks task gradients through \(q\); ordinary soft feeding does not. This separates finite representation from gradient-flow effects.

For B6, use a generic message \(z_t^k=\operatorname{GELU}(B_k\operatorname{LN}(h_t)+D_ku_t^k+a_k)\), with \(u_t^k=\sum_{j<k}R_{kj}z_t^j\), and inject \(A_kz_t^k\). Keep the same auxiliary heads on \((h_t,u_t^k)\), but do not use their categorical outputs in \(z\). Initialize \(A=0\), other matrices nonzero. Adjust generic width or active LoRA rank to bracket the typed model's useful trainable parameters; report the exact counts and costs. B6 tests a continuous routing explanation, not absence of annotation supervision.

B2's routed version computes the same hard predicted messages and auxiliary dependencies as the typed model but sets all residual injections to zero throughout training and inference. Its independent version removes those head-to-head routes as well. Removing injections can leave some embeddings without a task role; do not count inert tensors as useful capacity. Compensate with active adapters and report both the faithful removal and capacity-bracketed comparison.

**There is no confidence gate, external parser, global graph search, extra structural token, or separate LLM agent in version 1.** Add uncertainty gating only after hard-decision failures are observed and after comparing established soft/abstaining alternatives. This avoids making an untested combination unnecessarily hard to interpret.

### 5.2 State fields and dense target semantics

The labels are **prefix-indexed predictions of annotated properties**, not facts guaranteed to be recoverable from the prefix. Even whether a mention has ended can be ambiguous without its suffix. All groups are computed at every input token. No gold annotation, availability mask, boundary, or category count for that example enters the forward pass.

| Group | Fields | Training target at prefix position \(t\) |
|---|---|---|
| M: morphology | UPOS; Number, Person, Tense, Mood, Voice, Case, Gender when annotated | Features of the latest fully consumed aligned UD word within the visible input context |
| S: syntax | Universal dependency relation with subtypes preserved if supported; head-location class | Relation of that word; `ROOT`, `RIGHT_UNSEEN`, `LEFT_OUTSIDE_CONTEXT`, or visible left distance in sentence-local integer UD words: 1, 2, 3, 4–7, 8–15, 16–31, 32+ |
| E: entity meaning | Entity type | Type of the latest fully closed supported mention within the visible context |
| P: discourse | Information status | Resolved `infstat` of the same selected mention |

Before any completed word/mention, use `NONE_YET` except for a span crossing the window start as specified below. On a well-formed ordinary UD word row, an absent selected FEATS key has the explicit target `ABSENT_IN_ANNOTATION`; this describes the file, not a linguistic judgment of inapplicability. Malformed/unavailable fields are masked. Learn categories from training, treating multi-valued feature strings as sorted atomic tuples. Out-of-training-vocabulary labels get an evaluation-only `UNKNOWN_LABEL` marker, excluded from closed-set accuracy/CE and separately counted as coverage failures. Do not add an untrained output class. Unresolved Entity values such as `auto` and `_` remain masked, never `new`.

Targets are **carried forward in the training-label sequence** until a new source event; the model predicts them afresh at every token. Every completed selected event updates the target-building event identity, even if one or all of its fields are invalid. Invalid fields become masked for that interval, rather than reviving an older event. This constrains more positions than anchor-only supervision but does not eliminate unintended information in errors or masked fields. The model is not given event positions/carry decisions. Full-sentence targets may be future-informed; they never become runtime observations. Measure their irreducible prefix ambiguity.

For repeated targets from one source event, assign weights summing to one over its eligible, owned target positions in a complete training pass. The context overlap policy in Section 8.1 gives each target one owner; context-only copies have zero auxiliary weight. If an event participates in several scored blocks, normalize across all its eligible positions, not separately within each block. For `NONE_YET`, use one synthetic event per field and window, and separately report its frequency/loss so reset states cannot hide weak linguistic accuracy. Report event-weighted and token-weighted state evaluation separately.

M/S refer to the latest completed word; E/P refer to the latest selected completed mention. These may be different linguistic objects and have different ages. This is a multilevel summary, not a chain describing one shared object. Stratify route effects by word/mention overlap, mention gap, and event age.

A coarse head-location category deliberately substitutes for a full syntax graph in the cheapest mechanism. Therefore the proposed model cannot claim to implement all syntactic composition. TPT and the GiLT extension test whether stronger structural information is more effective. Do not add word identities, antecedent IDs, or raw selected hidden vectors to a supposedly finite linguistic message.

### 5.3 Text reconstruction, alignment, and edge cases

1. Parse CoNLL-U integer word IDs, multiword ranges and empty nodes separately. Validate each sentence's dependency references. Preserve document order, IDs, `SpaceAfter=No`, sentence text and available document boundaries.
2. Reconstruct one deterministic raw-text representation per document, using supplied raw text when exact alignment is available and verified. If only sentence text is available, join sentences with a documented newline rule. Call this the **canonical corpus text**, not a byte-identical copy of the original web source.
3. Tokenize the entire canonical document with the pinned tokenizer and no chat template. Obtain token-to-byte spans through verified offset mappings and UTF-8 conversion. Character offsets are not automatically byte offsets. Require exact round-trip text equality, including whitespace and non-ASCII cases.
4. Build gold word/mention spans strictly in the target-building path. A source event must start and end inside the visible context and have completed by \(x_t\). An event crossing the window start has unknown auxiliary state until an unambiguous eligible event, rather than a fabricated `NONE_YET`. A fully visible word whose head is left of the window gets `LEFT_OUTSIDE_CONTEXT`; a later head gets `RIGHT_UNSEEN`; a visible earlier head gets its sentence-local UD-word distance bin. Root remains `ROOT`. No head token is dereferenced into a forward feature.
5. Multiword tokens, empty nodes, normalization mismatches and overlapping subword offsets need explicit handling. In version 1, mask only affected auxiliary fields until the next unambiguous event; retain all valid LM targets. Do not silently drop the full document or treat an unparseable mention as `NONE_YET`.
6. Parse `Entity` bracket events with a stack of open mention records per **document-local coreference GRP**; GRP is not a unique mention ID. Process raw delimiters before unescaping `&#45;`. Detect `[i/n]` discontinuity markers. Version 1 masks unsupported/discontinuous mention targets rather than inventing contiguous spans.
7. For several supported mentions closing at the same UD word endpoint, choose the shortest span. Deduplicate equal spans/attributes; if equal innermost spans have conflicting labels, mask conflicting fields and report them. Do not choose a semantic value using an arbitrary GRP. Carry E/P from the same event. If one LM token spans multiple different word endpoints, treat that as a projection collision and mask affected auxiliaries. Save alternative-mention/collision counts.
8. Reset targets and caches at a new document or scoring window. No target event outside the visible context is treated as an observed state. Information status may still be intrinsically difficult when its antecedent is outside the window; report that case separately and do not equate it with a coding failure.
9. A missing P label must not erase a valid E label. Once a selected mention has unresolved P, mask P for that entire carry interval until the next event; do not retain an older mention's P while E changes. Apply the same interval rule to alignment failures and other unknown fields. Masks influence losses/diagnostics only; the forward method cannot receive them.

The preparation pipeline must report exact/masked coverage by split, genre, feature and cause. Four-channel real-data claims are blocked if a required field has no reliably resolved, nontrivial usable supervision. A three-channel result may proceed only after a recorded scope amendment; it is not a successful four-level experiment.

### 5.4 Losses, interfaces, and numerical conventions

Use a causal LM loss on every valid next raw-text token:

\[
\mathcal L_{\mathrm{pass}}=\frac{\sum_t-\log p(x_{t+1}\mid x_{\le t})}{N_{LM}}
+\sum_k\lambda_k\frac{1}{|F_k|}\sum_{f\in F_k}
\frac{\sum_tw_{t,f}m_{t,f}\,CE(q_{t,f},y_{t,f})}{\sum_tw_{t,f}m_{t,f}}.
\]

The sums above define the **full training-pass objective over unique owned positions**, not a separately renormalized weighted mean for each short microbatch. Omit a field whose full-pass denominator is zero; do not divide by an epsilon and manufacture a label. Use stable log-softmax/cross-entropy and FP32 loss accumulation when reduced precision is used. Label smoothing is off initially; if tuned, share the option with comparable baselines.

Precompute a complete pass's logical optimizer-step schedule independently of microbatch size. Let it contain \(K\) steps, \(N_{LM}\) LM targets and \(W_f=\sum_t w_{tf}m_{tf}\) valid event weight per field. For step \(b\), backpropagate

\[
\widehat{\mathcal L}_b=
\frac{K}{N_{LM}}\sum_{t\in b}NLL_t+
\sum_k\frac{\lambda_k}{|F_k|}\sum_{f\in F_k:W_f>0}
\frac{K}{W_f}\sum_{t\in b}w_{tf}m_{tf}CE_{tf}.
\]

At fixed parameters, the mean of these \(K\) step losses is exactly the declared full-pass objective, even when event density or target counts vary across steps. **Do not renormalize each step by its own event mass**, which would undo equal-event weighting across long carry intervals. A step without labels for a field contributes zero for that field. TPT uses \(K/N_{\mathrm{tree}}\) times the step's sum of owned word-event KL values, with the prescribed average over selected heads; \(N_{\mathrm{tree}}\) is the full pass's eligible event count. This fixed-parameter identity does not imply identical training trajectories if the logical optimizer-step partition itself changes.

During accumulation, sum the already scaled numerators over microbatches in the same logical step; do not average separately normalized microbatch losses. Test against a concatenated logical-step reference and a two-event/three-step density fixture. Save \(K,N_{LM},W_f\), schedule and cursor for exact resumption. Keep these totals/ownership common across same-information methods and recalibrate pilot gradient ratios using this estimator.

Proposed interfaces:

```python
@dataclass
class TextBatch:
    input_ids: Tensor       # int64 [B,T]
    attention_mask: Tensor  # bool [B,T], padding only
    position_ids: Tensor   # int64 [B,T], if required by pinned backbone

@dataclass
class Supervision:
    lm_targets: Tensor     # int64 [B,T], shifted once; ignore_index for padding
    lm_mask: Tensor        # bool [B,T]
    fields: dict           # name -> {targets, valid_mask, event_weights}
    tree_targets: object   # training-loss targets only, never model.forward

@dataclass
class ForwardResult:
    logits: Tensor         # [B,T,V], or exact chunkable head representation
    state_logits: dict     # field -> [B,T,C_field]
    state_ids: dict        # field -> [B,T], diagnostics
    cache: object | None
    selected_attention: dict | None  # differentiable [B,H_selected,T,T], B3 only

class Method:
    def forward(self, text: TextBatch, cache=None,
                intervention=None) -> ForwardResult: ...
    def losses(self, result: ForwardResult,
               supervision: Supervision) -> dict: ...
    def checkpoint_state(self) -> dict: ...
```

`intervention` is normally `None` and permitted only in declared diagnostic runs. It may replace a named categorical ID/message after prediction at a specified position. It must not become a hidden normal-inference oracle interface. Remove `Supervision` from `forward` at the type and callsite levels.

TPT requests selected attention through the fixed method configuration. The forward pass does not receive gold word mappings; `losses` performs the differentiable aggregation using training targets. Do not detach attention or retain all heads/layers unnecessarily. Detached-soft B4 detaches probabilities only when constructing messages; its auxiliary CE still trains the predictors.

Reference forward pseudocode:

```python
h = backbone.embed(input_ids)
messages = {}
for layer_index, block in enumerate(backbone.blocks):
    h = block(h, causal_mask, positions, cache_for_layer)
    if layer_index in tap_to_group:
        k = tap_to_group[layer_index]
        routed = zeros_like_message_shape(h, width=r, dtype=head_dtype)
        for j in earlier_groups(k):
            routed = routed + R[k,j](messages[j])
        q = heads[k](layer_norm(h), routed)
        ids = {f: logits.argmax(-1).detach() for f, logits in q.items()}
        ids = apply_declared_diagnostic_intervention(ids)
        e = sum(embedding[k,f](ids[f]) for f in fields[k]) / sqrt(len(fields[k]))
        h = h + injection[k](e)
        messages[k] = e
        save_state_outputs_for_loss_or_diagnostics(k, q, ids)
return lm_head(backbone.final_norm(h)), state_outputs
```

Respect the actual block return type/cache/RoPE implementation of the pinned model. Prefer a wrapper over rewriting attention. Do not assume attention head dimension equals hidden size divided by head count: the inspected Qwen3 configuration explicitly sets `head_dim=128`.

**Causality invariant:** with parameters fixed and evaluation mode, any two inputs with the same token prefix must have identical state/logit outputs on that prefix within validated numerical tolerance, regardless of suffix length/content or auxiliary labels. Every block is causal, each message uses only its current causal hidden state and earlier-depth causal messages, and no label enters forward; induction over blocks establishes the property. This must also hold with padding and caching.

**Complexity:** beyond the backbone, per-token work is approximately \(O(d\sum_fC_f+Kr d+K^2r^2+r\sum_fC_f)\), with small fixed \(K=4\). There is no new quadratic token-pair tensor in the primary channels. Actual overhead can nevertheless be dominated by dispatches, lost fusion, activation storage and logits; measure it.

### 5.5 Smallest complete controlled mechanism experiment

Implement a finite event-language generator, not an unrestricted natural-language simulator. Its purpose is to supply known dependencies and counterfactuals without an external annotator.

**World:** entities have a noun kind with semantic type (animal, person or object), color, number (one/two), and stable internal identity; events have an agent, transitive predicate, patient and polarity. The IDs never appear as extra model features. A discourse state records introduced entities and their last mention. Different types prevent the synthetic E target from being a constant `animal` label.

**Grammar:** noun phrases are `a/the + color + noun` for singular and `two/the + color + plural-noun` for plural; optional modifiers use `near + NP` or a relative clause `that + VP`. Clauses include subject agreement, active/passive variants, and positive/negative assertions. Contexts first introduce entities, then assert events, then ask a fixed-form question about an agent, patient, event truth or a previously introduced entity. Answers use the same public word vocabulary. Store a derivation tree, dependency edges, entity spans, denotation and given/new state with every example.

**Required families:** (a) subject agreement with opposite-number distractors; (b) agent/patient swap and passive voice; (c) negation changing denotation with syntax preserved; (d) given/new reference with the same entity vocabulary; (e) increasing modifier/relative-clause depth; (f) identical ambiguous prefixes with different valid suffixes. For example, changing the number of the main subject while keeping a nearby distractor fixed should change the agreement target. Changing an event's polarity should change its truth answer, not its entity type.

**Development versus test:** use one shared lexicon-to-meaning bijection within each complete run and hold out combinations of subject/object kind, voice, polarity, reference status and depth. Every individual factor and lexical item must appear in training; designated combinations do not. Repeat complete train/dev/test worlds with independently sampled lexicon bijections to test lexical robustness; do not change a word's meaning only in test. Partition a whole derivation/counterfactual family together. Balance lexical and length marginals where possible; report residual imbalance and add a bag-of-words/n-gram diagnostic to detect shortcuts.

**Exact tiny case:** enumerate a vocabulary/grammar subset small enough to compute all valid derivations and next-token probabilities. Verify normalization and ambiguity; compare a plain LM's achievable loss with the known distribution. An oracle solver validates the dataset, not a claimed neural improvement. All neural methods get the same tokens and annotations; no baseline is deliberately deprived of the full LM objective.

Start with an eight-block decoder of width 256 and four attention heads, tied embeddings and a training-only vocabulary. Its exact parameter count is computed from the implemented vocabulary; no invented model asset is required. Scale to width 512 or an additional recursion depth only if the first model cannot express the hypothesized dependency and development tests diagnose that limitation. A failure on a task outside the model's effective context/capacity is not a global falsification; a failure on a verified expressive, learnable task is evidence against the tested mechanism.

### 5.6 TPT baseline specification

The published method aggregates all attention rows used to predict the next word's subtokens, not just the final row of the preceding word. Let \(T_j\) be the input-token positions for sentence word \(j\), and \(\alpha_{v-1,u}\) the attention used to predict target subtoken \(v\) from position \(v-1\). For selected layer/head \(h\), use:

\[
C^{h}_{ij}=\sum_{v\in T_{i+1}}\sum_{u\in T_j}\alpha^{h}_{v-1,u},
\qquad
W^{h}_{ij}=\frac{C^{h}_{ij}}{\sum_{k\in J_i}C^{h}_{ik}},
\qquad
S_{ij}=\frac{\exp[-D(i+1,j)]}{\sum_{k\in J_i}\exp[-D(i+1,k)]}.
\]

Here \(D\) is undirected distance in the annotated dependency tree and \(J_i\) contains eligible complete earlier words \(j\le i\) in the **same sentence** and visible window. Exclude special tokens, whitespace-only tokens, ambiguous cross-word tokens, incomplete keys and partial keys from the target word. Normalize both distributions over the same \(J_i\). Skip cases with fewer than two keys, incomplete target-word alignment, missing predictive rows, or a sentence boundary instead of a next word. Own each tree event in the block scoring the next word's final target subtoken; its one aggregate may include available warmup predictive rows. Never duplicate the tree loss in another block. Whole-sentence tree distances are training targets, not inference inputs.

Average \(KL(S_i\Vert W_i^h)\) over eligible words for each selected head, then average over heads. Do not take KL of a head-averaged map. Start with one head at the syntax tap; development candidates include one head at each tap separately and the combination, with expansion if the best tested choice is at a boundary. Tune the coefficient independently and include TPT plus all auxiliary labels. This is a sentence-local, dependency-tree **project adaptation** within a document LM. The query shift and aggregation were checked against the paper and official training source. Validate both one-token-per-word parity and a hand-computed multi-subtoken next-word fixture. [R10]

Requesting attention outputs can disable efficient attention kernels. Share the same faithful path for token-matched timing comparisons, and also report each method's optimized inference separately. A training-only attention loss must not be charged an unnecessary parser at inference.

## 6. M3 Max feasibility and hardware-adaptive execution

### 6.1 Discover the actual laptop before selecting a run shape

Apple lists multiple M3 Max configurations. A 14-core CPU/30-core GPU variant has 300 GB/s memory bandwidth; the 16-core/40-core variant has 400 GB/s. Available memory starts at 36 GB, not at the largest advertised configuration. Neither bandwidth nor nominal memory predicts this custom training path's sustained speed. [R30]

The following are **existing macOS commands**, not project commands:

```bash
uname -m
sw_vers
sysctl -n hw.memsize
sysctl -n hw.ncpu
system_profiler SPHardwareDataType SPDisplaysDataType SPSoftwareDataType
vm_stat
sysctl vm.swapusage
df -h .
python3 -c "import platform; print(platform.machine(), platform.python_version())"
```

Record physical bytes, GPU variant, macOS/build, native `arm64` interpreter, free disk, idle memory/swap, power mode, external workload, and whether the laptop is on power. Redact machine serial numbers from shared reports. Do not change OS settings or disable protection mechanisms to force a run to fit. If Python reports `x86_64` under Rosetta, select a native interpreter before installing acceleration packages.

### 6.2 Software and asset selection

**Primary implementation:** native Python 3.12, a mutually compatible released PyTorch/Transformers/PEFT stack, NumPy, a standard YAML/JSON parser, safetensors, and a local CoNLL-U reader with the explicitly required Entity support. Use PyTorch CPU FP32 as the small numerical reference and MPS for training after parity tests. Reuse the backbone's official attention/cache implementation and PEFT's supported adapter path; wrap block outputs to add the proposed modules.

The inspected stable PyTorch 2.14 MPS documentation requires macOS 14 or newer; newer/main documentation and the 2.15 release-candidate announcement have different requirements. The announcement is not a reason to pin an unreleased stack. MLX documentation currently requires macOS 14 or newer and Python 3.10 or newer. Resolve a **released** compatible set on the actual OS, record wheel versions/hashes and repository commits, then freeze it. If no released stack supports a needed model, either validate a pinned source revision explicitly or take the documented fallback and narrow the claim. [R33], [R34], [R35]

Do not supply a fabricated ready-made lockfile. The execution agent must create `requirements.in` with the package roles above, resolve compatible versions in an isolated environment, run the operation tests below, then write `requirements.lock` and `environment.json`. Record the exact installation commands. A lock that imports successfully but fails backward is not accepted.

| Asset | Role and rationale | Conditions |
|---|---|---|
| `Qwen/Qwen3-0.6B-Base` | Main same-backbone comparison; pretrained-only checkpoint; public Apache-2.0 model card | Pin weights, tokenizer and config to the same full revision. Do not substitute the instruction/reasoning-tuned `Qwen3-0.6B` checkpoint. [R31] |
| `Qwen/Qwen3.5-0.8B-Base` | Contemporary generic challenger; architecture differs, so it cannot alone identify the side-channel effect | Apache-2.0 card; validate text-only loading and hybrid recurrent/attention training. Current MLX and Transformers sources contain training paths, so a CUDA-oriented example is not proof of impossibility. [R32], [R36] |
| `HuggingFaceTB/SmolLM2-135M` | Cheap pretrained plumbing/calibration model | Apache-2.0 card; a negative result here does not decide the main hypothesis. `SmolLM2-360M` is a replication/fallback scale, not a replacement silently labeled 0.6B. [R37] |
| Eight-block scratch model | Controlled language mechanism experiment | Locally implemented, trained only on the declared generator; vocabulary and parameter count recorded |

A small model is a scientific scale choice as well as a resource choice. Hold the objective, annotation access, split discipline and mechanism fixed across scale comparisons. Do not invent nonexistent small SmolLM3 variants. Other current models with gated or noncommercial licenses are not mandatory prerequisites when public permissive baselines already answer the attribution question.

**Faithful fallback:** if MPS has a material unsupported operation, use a mathematically equivalent implementation and prove small CPU/MPS loss and gradient parity. If that is too slow, port all relevant compared conditions to MLX and establish parity on identical weights/text. CPU-only execution can complete the tiny mechanism experiment. A CPU-only toy is not evidence that the main laptop workload is practical.

Quantization is unnecessary for the core design. Current bitsandbytes documentation lists Apple Silicon support, so “bitsandbytes never works on Apple” would be outdated. Specific quantized backward kernels remain a separate gate. Introducing quantization changes the comparison and requires shared validation; it is not the first recovery step. [R38]

### 6.3 Critical operation and numerical gates

Before loading several checkpoints, run a small block with the same operations and then one complete main-model optimizer step:

- Embedding lookup, layer/RMS normalization, linear projections, RoPE, attention, causal masks, gather/argmax, categorical embedding feedback, cross-entropy, backward, AdamW, adapter updates, and activation checkpointing.
- Padded and unpadded batches; B=1 and B>1; unequal lengths; context boundaries; inference prefill and one-token cached continuation.
- Prefix-suffix invariance, zero-injection identity, consistent label-permutation identity, summed-denominator accumulation parity, and saved/resumed update parity.
- FP32 first. Assess BF16/FP16 with actual loss, gradient direction, state decisions, overflow, underflow and downstream score drift. Compare precision error against the effect the study aims to resolve. Use the same accepted precision for direct competitors or report the difference explicitly.

PyTorch SDPA boolean masks use `True` for allowed attention; other APIs can use the opposite convention. The documented `enable_gqa` path is not a blanket MPS guarantee. An explicit KV-head repeat can preserve the equation but change memory/cost. Cached rectangular attention also needs correct absolute-position alignment. Validate against an explicit small causal attention matrix, not only two equally buggy implementations. Float64 reference calculations belong on CPU. [R39]

Use the pinned backbone's true dimensions. The Qwen3 config has 28 layers, hidden width 1,024, FFN width 3,072, 16 query heads, 8 KV heads, `head_dim=128` and vocabulary 151,936 with tied input/output embeddings. Therefore `hidden_size / num_heads` is **not** its head dimension. Source inspection also found separate recurrent and KV state in the Qwen3.5 path; do not force a uniform Transformer cache abstraction onto it. [R31], [R36]

Disable silent CPU fallback during compatibility tests. If fallback is deliberately accepted later, log every affected operation and include transfers in end-to-end timing. Check the selected MPS environment variables; never set the allocator high-watermark ratio to zero as a workaround for insufficient memory. [R45]

### 6.4 Memory model and calculated planning envelope

Let `P` be backbone parameters, `A` trainable adapter/module parameters, `B` microbatch, `T` input tokens, `L` layers, `Hq/Hkv` query/KV heads, `dh` head width and `V` vocabulary. Use:

\[
M_{\rm peak}=M_{\rm weights}+M_{\rm grads}+M_{\rm optimizer}
+M_{\rm saved\ activations}+M_{\rm attention\ work}
+M_{\rm logits/loss}+M_{\rm states}+M_{\rm cache}
+M_{\rm allocator}+M_{\rm CPU\ data}+M_{\rm checkpoint\ copy}
+M_{\rm OS}.
\]

This equation is a whole-machine accounting envelope. Compare its **process components excluding `M_OS`** against an available-process ceiling that already subtracts idle system use; do not charge the operating system twice. Components' maxima need not occur at the same instant. A frozen base still needs activation gradients through later blocks when lower adapters are trainable.

| Component | Planning expression / caveat |
|---|---|
| Full FP32 AdamW static state | Approximately `16P` bytes for weights, gradients and two moments; temporaries/master copies extra |
| Frozen base + trainable additions | Approximately `4P + 16A` in FP32, or `2P + 16A` for BF16 base with FP32 additions; verify actual state dtypes |
| Vocabulary logits | `B*T*V*bytes_per_logit`; cross-entropy may create FP32 intermediates |
| Eager attention | One score tensor is `B*Hq*T*T*bytes` per layer; probabilities/backward can add copies |
| Full-attention inference KV | `2*L*B*T*Hkv*dh*bytes_per_cache_element`; not a training-cache requirement when cache is disabled |
| Diagnostics | Avoid retaining every layer/token tensor; sample a fixed diagnostic subset and move outputs to CPU |
| Checkpointing | Include serialization copies and atomic temporary checkpoint; do not count only model weights |

Configuration arithmetic gives **596,049,920** Qwen3 parameters before adapters. Confirm with loaded `numel()`; the following numbers are **calculated, not measured peaks**:

| Quantity | Calculated GiB |
|---|---:|
| BF16 backbone weights | 1.11023 |
| FP32 full-tuning weights + gradients + AdamW moments | 8.88184 |
| FP32 logits, B=1, T=512 | 0.28979 |
| One FP32 score tensor per layer across all 28 layers, B=1, T=512 | 0.43750 |
| Same score term at T=1,024 | 1.75000 |
| BF16 inference KV, B=1, T=512 | 0.05469 |

This supports a **plausible**, unquantized 0.6B local path even at the lower M3 Max memory configuration; it does not prove that every backend or long-context batch fits. A 10–50M-parameter scratch model needs roughly 0.16–0.80 decimal GB for FP32 AdamW static state, plus activations. Preprocessing the inspected approximately 30 MB of CoNLL-U inputs is small compared with model state; tokenized caches and diagnostics still need measured accounting.

Set an operating ceiling using observed memory, for example:

```text
reserve = max(4 GiB, measured checkpoint transient + uncertainty margin)
usable = min(0.8 * recommended_GPU_working_set,
             physical_memory - idle_system_usage - reserve)
```

The reserve and 0.8 factor are operational safeguards, not scientific thresholds. Test worst-length batches against this ceiling; sustained swap growth or unstable memory pressure invalidates the selected shape even if a short step completes.

### 6.5 Calibration, runtime and storage

Calibrate `T` from linguistic context requirements and train/dev length/dependency distributions, initially probing 256, 512 and 1,024 tokens. Freeze the selected context before confirmation. Do not automatically choose the longest context or reduce it after OOM without acknowledging the changed task. Determine microbatch from measured memory, then set token-correct accumulation to preserve the chosen effective batch.

Warm up until loading/compilation/cache effects separate from sustained execution. Measure consecutive blocks of **complete optimizer steps**, synchronizing only at block boundaries. Include forward, backward, update and required transfers; separately time validation, saving, and data preparation. Alternate conditions' timing order to reduce thermal/order bias, keep power state consistent, and retain raw block measurements.

PyTorch exposes `torch.mps.synchronize()`, `current_allocated_memory()`, `driver_allocated_memory()` and `recommended_max_memory()`. It does not document a CUDA-style `max_memory_allocated()` in this API. Samples of current memory can miss transient peaks: label them as sampled maxima and use a trace or conservative margin when this matters. Current MLX memory functions are exposed through `mlx.core`, including `get_peak_memory` and `reset_peak_memory`; materialize lazy results and call `mx.synchronize()` around timings. [R43], [R44]

\[
t_{\rm train}=\frac{\text{nonpadding processed training tokens}}
{\text{measured sustained tokens/second}}+t_{\rm validation}+t_{\rm save}.
\]

Compute the full program forecast by summing every tuning trial, confirmation seed, baseline, failure, evaluation and preprocessing stage. Log both **unique target tokens** and **processed input tokens** when context overlap causes extra work. For illustration only, one million tokens at a future measured 100 tokens/s takes 2.78 hours before validation; at 500 tokens/s it takes 0.56 hours. Neither throughput was measured here or forecast as expected.

Storage must cover one immutable base snapshot per model, original/processed data, caches, configurations, raw scores, logs, and recoverable checkpoints. A full FP32 resumable Qwen3 checkpoint with weights plus two moments is approximately `12P` bytes, or 7.15 decimal GB, before metadata. Two saved versions plus a temporary third can exceed 20 GB. Adapter checkpoints are much smaller but still require the frozen base. Before any large download/save, calculate free space against the actual format; do not download all optional assets in advance.

### 6.6 Recovery and exact resumption

Save only at completed optimizer steps unless accumulated gradients are explicitly included. A resumable checkpoint contains model/adapters, optimizer, scheduler, scaler if used, Python/NumPy/framework/MPS RNG state, sampler order and index, completed token counters, backend/precision, dependency/config/data/model hashes, and cumulative costs.

Write to a temporary path, flush/fsync, calculate checksum, load/validate metadata, then atomically rename. Keep the previous known-good state until validation succeeds. The inspected MLX-LM trainer saves trainable model weights, not all of the scientific resume state above; build the missing state handling rather than assuming an adapter file is an exact checkpoint. [R46]

On OOM or interruption: retain the failed-run record and cost, terminate the affected process, recover the last valid checkpoint, and diagnose the phase of peak use. First reduce microbatch while preserving effective token batch; then use validated activation recomputation or exact chunked loss. A dtype/kernel change needs parity checks. A context/model/data change requires a methodological amendment and a new run identity. Never continue partially corrupted optimizer state or silently restart with a new sampler.

## 7. Confirmatory statistics and measurement protocol

### 7.1 Units, pairing and fixed-sample design

The real-data unit is a document, or a larger source cluster if the duplicate/source audit links documents. Training seeds are a separate source of algorithmic variation. Tokens, overlapping windows, dense repeated state targets and timing repetitions are **not independent test examples**. In the synthetic study, the unit is the context-overlap component defined in Appendix B, keeping its derivation/counterfactual families, alternative questions and lexical variants together.

Use the same seed list, data order, batch boundaries and eligible initialization seeds across competing methods; tensors with compatible roles share initialization where meaningful. Different model graphs cannot have identical random consumption, so use separately named RNG streams for sampler, backbone adapters, side modules and dropout. Pair results by training replicate and document. Use every completed preregistered replicate, not the best seed.

Begin pilot variance estimation with four paired development runs of the selected typed configuration and the strongest candidates. Four is a starting point for estimating variation, not a confirmation quota. If the variance estimate is unstable or learning curves differ materially, add pilot runs in a recorded batch and reassess planning uncertainty. Do not inspect protected outcomes during this process.

Estimate variance components for paired differences:

\[
D_{sd}=b_{B^*,s,d}-b_{\mathrm{typed},s,d}
=\mu+a_s+b_d+\epsilon_{sd}.
\]

The model is a **planning approximation**, not a claim that document effects are exactly Gaussian. Simulate planned analyses using empirical development residuals, whole-source clustering and seed variation, with genre as a reporting covariate. Splitting 30 development documents can leave only one document per genre in each half: do **not** bootstrap within such singleton strata. Pool source clusters for planning, show the resulting genre uncertainty, and use leave-one-genre-out sensitivity. Examine small positive effects, zero and negative effects over a grid. More seeds reduce training variance; they cannot create more independent natural documents.

Before protected evaluation, freeze `n_confirm` and the reason for it. With no application-specified minimum worthwhile effect, use a precision-versus-cost curve rather than inventing a required percentage gain. A default allocation rule is to increase seed count until its estimated contribution to mean-effect variance no longer dominates the document/source sampling floor, then evaluate whether the resulting interval width can resolve the conservative pilot-supported effect range. Use variance-component uncertainty, not only point estimates. If that rule yields too few seeds for credible seed uncertainty, collect more pilot information or restrict inference to descriptive results. If document variance is effectively zero, choose a fixed precision justified by the synthetic optimum or application utility; do not divide by zero. Record the chosen count and detectable-effect curve. If affordable local replication cannot achieve useful precision, the outcome is inconclusive. Expansion of independent data must precede protected scores. [R47], [R48]

### 7.2 Primary estimator and uncertainty

Compute the paired mean difference across seeds and equally weighted retained documents, preserving the original weighting in every resample. For the primary interval, use a crossed seed-and-document cluster bootstrap:

```python
# D has one value per paired seed and document; no token-level resampling.
for b in range(bootstrap_replicates):
    sampled_seeds = sample_with_replacement(seed_ids, len(seed_ids))
    sampled_clusters = sample_with_replacement(source_cluster_ids,
                                              len(source_cluster_ids))
    # Apply each cluster's draw multiplicity to all its member documents.
    # Original estimand remains the equal-document mean, not equal-cluster mean.
    effects[b] = mean_over_seeds_and_weighted_documents(
        D, sampled_seeds, sampled_clusters
    )
interval = percentile(effects, [2.5, 97.5])
```

Use a fixed analysis RNG and enough bootstrap draws that endpoint Monte Carlo variation is small relative to reported precision; begin at 10,000 and verify endpoint stability using independent resampling batches. The bootstrap itself is cheap CPU analysis, not additional model training. With few seeds or only two documents per genre, nominal 95% intervals can be unreliable. Validate this procedure on simulated null/known-shift data with the observed clustering; report a seed-level paired-t interval conditional on this finite corpus and per-document/genre effects as sensitivity analyses. If those disagree materially, call the population inference fragile and narrow it; do not select the favorable procedure.

Sample each source cluster as a whole even if it spans several genres; genre is a reporting stratum, not permission to split dependent sources. The **finite-corpus** estimand conditions on these documents and generalizes over training randomness. The **observed-mixture** interpretation also treats source clusters as sampled instances from this corpus's mixture; it is more assumption-dependent and does not hold every genre proportion fixed. Do not generalize either interval to arbitrary languages, genres or deployments.

For primary H1, a positive two-sided 95% interval supports improvement against the preregistered development-selected `B*`. An interval spanning zero is inconclusive at the achieved sensitivity. An interval entirely below zero supports harm in this setup. A narrow interval near zero can rule out benefits larger than its upper bound; it cannot prove the mechanism is universally useless.

The **crossed seed/source-cluster interval** is the prespecified H1 decision interval, conditional on passing the pilot coverage/sensitivity checks. The finite-corpus seed interval is a reported sensitivity analysis, not an alternative chosen after observing scores. If simulation shows that the decision interval is unreliable at the available units, amend the inferential scope at WP6 before protected results, or report descriptive estimates without a confirmatory population claim.

### 7.3 Selection, multiplicity and repeated inspection

Freeze one typed configuration, one primary comparator `B*`, one primary score and one primary contrast. Selection uses development only. All mandatory comparators' results remain visible, including those stronger on protected data; do not call `B*` the test winner merely because it won development.

H2–H4 form a separately declared attribution family. Use Holm-adjusted tests or simultaneous bootstrap intervals for the finite listed contrasts, and report raw effect sizes/intervals as well. GENTLE, per-genre slices, alternative context lengths and optional benchmarks are secondary unless given their own independently justified confirmatory contract. A claim of superiority over **all** baselines requires each relevant pairwise contrast to pass a simultaneous procedure.

Do not increase seeds after looking at a borderline primary result. A fixed-sample confirmation runs the locked number; if a training failure occurs, follow the preregistered replacement/recovery rule without examining scores. Numerically divergent runs are reported as failures, with success rate and failure-inclusive interpretation. A software bug invalidates affected runs; correct it, record the amendment and rerun the affected comparison. Favorable partial results cannot survive a known invalid harness.

If test feedback changes the method, hypotheses or endpoints, label further work exploratory and obtain fresh independent confirmation. Existing GENTLE outcomes cannot be reused as an untouched rescue test. A future sequential design is allowed only if its alpha-spending/e-value rule is specified before its first outcome; “continue until significant” is prohibited.

### 7.4 Harness validation before model claims

**Fresh seeds and failures:** confirmation uses a newly generated, locked seed list disjoint from all configuration-selection runs. Infrastructure failures (power loss, external process termination, disk failure) may resume/rerun the same seed from verified state. Algorithmic divergence, NaNs without an established infrastructure cause, or repeated valid-shape OOM are outcomes of the tested procedure, not replaceable unlucky seeds. Under the default no-retry algorithm, any missing paired outcome from such divergence blocks an unconditional H1 superiority claim; report failure rate and successful-run scores conditionally, without substituting new seeds. A more robust training/retry procedure must be tuned and locked as a new procedure before fresh confirmation. If cause is uncertain, retain the failure classification until evidence resolves it.

Required deterministic and statistical checks:

1. Identical predictions produce exactly zero paired differences, correct byte counts, and tie behavior.
2. A known probability distribution returns its analytically known cross-entropy/BPB. A controlled log-probability shift yields the intended effect in the report.
3. Sliding/scoring windows cover every eligible target exactly once; padding, artificial special tokens and repeated context never enter the byte denominator.
4. Whole-document and cached evaluation agree where context is identical. State heads see no gold boundaries, masks, labels or suffixes.
5. CPU FP32 full-batch and accumulated gradients match within predeclared numerical tolerance.
6. Simulated clustered nulls and known shifts test interval coverage/sensitivity; report simulation uncertainty. The simple identity check performed during research is not a statistical coverage test.
7. Learned feature vocabularies and hyperparameters cannot access protected outputs; logs distinguish schema validation from model scoring.

Set numeric tolerances from FP32 reference behavior and the size of an effect the experiment seeks to distinguish. For example, a synthetic parity check can begin at `atol=1e-6, rtol=1e-5` on CPU FP32, but a backend discrepancy must be investigated rather than made to pass by arbitrarily relaxing tolerances.

## 8. Repository, data flow and reproducibility contract

### 8.1 Exact text/scoring-window convention

Canonical text is the verified sentence text joined by a single LF between sentences and no trailing LF, in corpus document order. Normalize neither Unicode nor whitespace. Preserve the original CoNLL-U files alongside this derived representation. Let raw tokenizer IDs be \(x_0,\ldots,x_{N-1}\), produced with no automatic special tokens/chat template. Verify decoding the **complete sequence** exactly reconstructs canonical text.

Let `START` be the pinned tokenizer's documented BOS ID, or its documented EOS ID used as a start sentinel if BOS is absent. Record this choice. The sentinel is conditioning context only, never a scored target. If neither exists, add a shared explicit start token to all compared methods and treat that as an amended model/tokenizer configuration.

Set \(z=[START,x_0,\ldots,x_{N-1}]\). With context length \(T\) and stride \(S=\lfloor T/2\rfloor\), define:

```python
for a in range(0, N, S):
    b = min(a + S, N)
    c = max(0, a + S - T)
    input_ids = z[c:b]
    shifted_targets = z[c+1:b+1]
    # Row with global input index i predicts raw token x_i.
    lm_mask = [(a <= i < b) for i in range(c, b)]
    # Only these owned rows contribute ordinary auxiliary losses.
    # Context-only rows are computed, but have zero LM/auxiliary weight.
```

This scores every raw token, including the first, exactly once. The context start depends on block start/stride, **not on the unknown final document length**. Row `i` consumes `z_i` (START when `i=0`, otherwise `x_(i-1)`) and predicts `x_i`; auxiliary labels describe only that consumed prefix. Reset caches per window/document; position IDs are `0..len(input_ids)-1`. Train with the same ownership rule, deterministic whole-window shuffling, and full-pass owned-event weights. All conditions share logical steps/windows/data order. TPT may aggregate available context-only predictive rows for its uniquely owned event.

For document BPB, \(B_d=\mathrm{len}(\mathrm{canonical\_text.encode("utf-8")})\) because all raw tokens are scored. **Do not** add independently decoded token byte lengths: byte-BPE tokens can split a UTF-8 character and have overlapping character offsets. Mask ambiguous auxiliary alignment without changing LM target coverage or the document byte denominator. Empty-text documents are invalid inputs and must be reported during the pre-model data audit.

For cross-backbone B7, report two regimes: each model's practical token-budget frontier, and an information-matched diagnostic using common, nonoverlapping canonical-text blocks. Use a fixed number `C` of Unicode code points per block, with boundaries `0,C,2C,...` independent of suffix contents or final document length; preserve every byte including separators and the final short block. Choose `C` conservatively on development, verify the common token-cap bound in the pre-model schema audit, and freeze it. Do not greedily move a block start after inspecting an unseen sentence remainder. Retokenize each block independently, reset to the same role of start sentinel, score every raw block token, and divide by the common raw bytes. This measures prediction on identical raw-text blocks, with both models losing cross-block context equally. Report it separately from the main overlapping-document protocol. Do not present equal token limits across tokenizers as equal text access. Primary H1 always uses the same tokenizer/backbone.

### 8.2 Minimal project layout

The following are files to create; no existing `linglab` distribution is assumed.

```text
README.md
pyproject.toml
requirements.in
requirements.lock
configs/{base,methods,pilot,confirm,synthetic}.yaml
linglab/cli.py
linglab/__init__.py
linglab/__main__.py
linglab/data/{conllu,entities,alignment,windows,synthetic}.py
linglab/models/{backbone,typed,auxiliary,continuous,tpt}.py
linglab/{train,evaluate,checkpoint,calibrate}.py
linglab/analysis/{paired,bootstrap,interventions,report}.py
tests/{test_alignment,test_causality,test_tpt,test_loss,test_resume}.py
manifests/{assets,data,splits,labels,environment}.json
protocol/{exploration,confirmation,amendments,decisions}.md
artifacts/schema_audit/
runs/<run_id>/{config,resume_state,metrics,failures,timings,predictions}/
reports/{results.csv,coverage.csv,costs.csv,report.md}
```

Use a local Git repository for source/config history. Keep large assets out of Git; store immutable revision/hash manifests. Nothing may depend on a path from this planning session or an unpublished intermediate note.

### 8.3 Fresh-environment sequence and proposed CLI

After selecting an installed native interpreter, these are **existing Python/pip/Git commands**. `python3.12` is illustrative only if discovery confirms that executable:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install torch transformers peft numpy pyyaml safetensors huggingface-hub
.venv/bin/python -m pip check
.venv/bin/python -m pip freeze > requirements.resolved.txt
git ls-remote https://huggingface.co/Qwen/Qwen3-0.6B-Base HEAD
git ls-remote https://github.com/UniversalDependencies/UD_English-GUM.git 'refs/tags/r2.18' 'refs/tags/r2.18^{}'
```

The unconstrained install is a **compatibility probe**, not a reproducible environment. Inspect installed versions, run WP1, then generate the final exact lock including package hashes with a supported lock tool, recreate a second environment from it and rerun the small checks. Use the recorded full model revision for the subsequent hub download; repository `HEAD` output is metadata resolution, not the weights themselves. For an annotated Git tag, record the peeled `^{}` commit rather than the tag-object hash as the repository commit. Native Python installation and disk location are routine local choices. Do not assume a preinstalled Homebrew, CUDA, compiler, or paid service.

Implement `linglab/__main__.py` with `from .cli import main` and `if __name__ == "__main__": main()`, and provide `cli.main()`. The commands below run from the repository root and explicitly use the created environment; they are **proposed CLI contracts to implement**:

```bash
.venv/bin/python -m linglab discover --output manifests/environment.json
.venv/bin/python -m linglab prepare --config configs/base.yaml --schema-only
.venv/bin/python -m linglab validate --suite core
.venv/bin/python -m linglab generate --config configs/synthetic.yaml --split development
.venv/bin/python -m linglab calibrate --config configs/pilot.yaml
.venv/bin/python -m linglab train --config configs/pilot.yaml --method B2 --seed 101
.venv/bin/python -m linglab freeze --config configs/confirm.yaml
.venv/bin/python -m linglab train --config configs/confirm.yaml --method typed --seed 1009
.venv/bin/python -m linglab evaluate --protocol protocol/confirmation.md --split gum_test
.venv/bin/python -m linglab analyze --protocol protocol/confirmation.md
```

`freeze` validates that mandatory fields are filled, hashes the protocol/config/manifests, and refuses unresolved revision placeholders. `evaluate` requires that hash and writes a test-access record before computation. This is audit discipline, not an assertion that a local file permission can prevent deliberate misuse. `train --resume <path>` must refuse incompatible provenance unless an explicitly recorded new-run amendment is provided.

### 8.4 Configuration and optimizer defaults

Store resolved configurations, never just command-line overrides. Required fields include:

```yaml
protocol_version: 1
phase: exploratory              # changed only by a protocol lock
method: typed
model_id: Qwen/Qwen3-0.6B-Base
model_revision: REQUIRED_FULL_COMMIT
tokenizer_revision: REQUIRED_FULL_COMMIT
data_release: ud-r2.18_gum-v12.1.0
data_manifest_sha256: REQUIRED
context_tokens: CALIBRATE
score_stride: half_context
effective_target_tokens_per_step: CALIBRATE
precision: fp32                 # change only after parity gate
backend: mps                    # conditional on WP1
cache_during_training: false
lora:
  target_modules: [q_proj, k_proj, v_proj, o_proj]
  rank: 16
  alpha: 16
  dropout: 0.0
messages:
  width: 32
  taps: [6, 13, 20, 26]
  hard_ids: true
  straight_through: false
  initial_injection: zero
optimizer:
  name: AdamW
  learning_rate: 0.00005
  betas: [0.9, 0.999]
  epsilon: 0.00000001
  weight_decay: 0.01
  clip_global_grad_norm: 1.0
  scheduler: cosine_after_linear_warmup
  warmup_fraction: 0.05
seed_streams: [sampler, adapters, messages, dropout, analysis]
```

These are starting optimization settings, not assertions of optimality. Exclude bias and normalization scale/offset from weight decay; apply the same rule to all comparable methods. Use `alpha/rank=1` in the initial LoRA brackets to avoid accidentally changing adapter scaling when rank changes. Freeze the main base weights and output/input embeddings unless a separate full-tuning comparator is activated. Full tuning becomes a justified comparator if the pilot shows adapter capacity is limiting and measured memory supports it; report that comparison separately from a fixed-adaptation-budget claim.

Choose effective target tokens per optimizer step from a shared B1 pilot: start with one corpus-pass partition giving stable, nondivergent gradients and test a doubled/halved batch when the learning curve or gradient variance suggests dependence. Record **target**, input, and event counts separately. Microbatch is an execution detail; the effective objective must be unchanged by accumulation. In the main matched comparison, lock a common full-pass count, total owned targets and logical-step partition as specified in Section 4.5. Learning rates and auxiliary coefficients may be tuned per method; schedules use that common horizon. Independently optimized training horizons are reported only in the separately labeled cost frontier.

For mixed precision, calculate heads/softmax and message projections in the validated explicit dtype, normally FP32, then cast the injected \(A_ke\) to `h.dtype` before residual addition. Ensure the following backbone block receives its expected dtype. Implement checkpointed blocks functionally: no append-only hidden-state lists or cache mutation on recomputation. `use_cache=False` during training.

Argmax uses a deterministic field-category order for ties. Log top-two logit margins and CPU/MPS category mismatch rates. Permutation-invariance fixtures must have strict unique maxima or permute the tie rule too. Finite differences test only smooth paths away from category boundaries; a missing LM gradient into a detached predictor is expected, not a failed implementation.

### 8.5 Experiment records and retained outputs

Every run has an immutable `run.json`:

```json
{
  "run_id": "unique-id",
  "phase": "exploratory-or-confirmatory",
  "method": "B2",
  "parent_run": null,
  "source_commit": "full-sha",
  "protocol_sha256": "sha256",
  "config_sha256": "sha256",
  "data_manifest_sha256": "sha256",
  "model_revision": "full-sha",
  "environment_sha256": "sha256",
  "seed_streams": {},
  "started_utc": "ISO-8601",
  "status": "running-complete-failed-invalidated",
  "failure_reason": null
}
```

Required per-document prediction rows: `run_id, split, doc_id, source_cluster, genre, n_raw_tokens, n_utf8_bytes, nll_nats, bpb, context_policy, score_coverage`. Keep sufficient raw log-probabilities or deterministic recomputation metadata to audit aggregation. Diagnostic rows include field, valid events, unknown/masked counts, confusion counts, event/token-weighted scores, gap/age strata and intervention identity. Do not expose raw licensed text in a public report.

Synthetic rows additionally retain `component_id, family_id, member_id, context_key, query_family, endpoint, realized_depth, answer_class, denotation_correct, answer_log_probability` and the generator/lexicon hashes. Aggregate and resample at `component_id`, keeping member-level outputs for paired counterfactual checks.

Required timing rows: operation, warmup/steady-state flag, start/end UTC, synchronized elapsed seconds, target/input tokens, full updates, precision/backend, memory API, sampled/trace peak bytes, CPU RSS, swap, checkpoint bytes and thermal/power notes. Memory categories must distinguish **process workload** from total machine usage: subtract idle OS usage only once when applying the Section 6 ceiling.

Preserve failed trials, boundary hyperparameter extensions, rejected checkpoints, and negative results. `protocol/decisions.md` records the question, evidence paths, decision, owner/agent, date and next action. `amendments.md` records any change to hypothesis, data population, primary score, baseline fairness, context, precision or interpretation, plus which outcomes were already known. The coding agent may make routine non-destructive implementation choices autonomously; methodological changes require this record and an appropriate fresh confirmation, not silent edits.

## 9. Risk-ordered work packages and decision gates

The main dependency path is WP0 → WP1/WP2 → WP3 → WP4 → WP5 → WP6 → WP7 → WP8 → WP9. WP1's small numerical work and WP2's generator can proceed independently after prerequisites; full-model downloads wait for the small compatibility checks. Tests below are necessary scientific invariants, not a demand for redundant unit tests of every helper.

### WP0 — Freeze prerequisites, data rights and source manifests

**Objective/risk:** prevent an unsupported machine, wrong release, unusable labels or incompatible rights from invalidating later work. **Entry:** this plan, local M3 Max access, permitted internet downloads. **Inputs:** Section 6 commands; URLs/hashes in Appendix A; official licenses.

**Actions/files:** create `manifests/environment.json` and `assets.json`; discover actual hardware/native Python/storage; resolve full revision IDs before large downloads; fetch the four pinned corpus files and source metadata; record licenses and intended internal research use. Implement the schema-only CoNLL-U/Entity reader and reproduce document/sentence/mention counts. Run source/duplicate checks without model scores. Write `splits.json`, document clusters and protected-access log.

**Outputs/tests:** exact SHA256/count equality with Appendix A, or a documented explanation of any changed release; no unmatched Entity brackets; per-ID stack support; exclusions by explicit pre-model rules; no underscored Reddit placeholders in training. Expected units are bytes, documents, words and events. Resource cost is CPU parsing plus small public downloads, measured rather than guessed.

**Gate:** proceed if rights for intended use and required labels are usable, metadata is reproducible and the laptop has a compatible local route. A licensed-data exclusion changes the population before modeling. If data cannot support E/P, do not substitute sentence mood and call it pragmatics: amend to a narrower study or select a documented compatible corpus before any confirmation. If hardware is inaccessible, produce a blocked feasibility record; do not claim M3 validation.

### WP1 — Validate the backend, loss harness and alignment

**Objective/risk:** detect leakage, incorrect gradients, byte errors and unsupported operations cheaply. **Dependencies:** WP0. **Inputs:** tiny random model/arrays, pinned tokenizer files, declared window policy.

**Actions/files:** implement `alignment.py`, `windows.py`, shared `Method` interfaces, exact CE/BPB aggregation, small causal attention, functional checkpointing and the core tests. Use minimal numerical routing/gradient fixtures and shared primitives here; the integrated production typed model is implemented in WP4 and is not a prerequisite for WP1. Download only the tokenizer/config and small plumbing model after elementary operations pass. Implement CPU FP32 reference checks and MPS forward/backward tests.

**Outputs/tests:** unique target ownership; exact complete-text round trip; documented cross-word/subtoken masks; prefix invariance including contradictory gold targets for the same prefix; branch-zero identity; strict-margin label permutation; correct gradients with LM-only versus auxiliary-only loss; sum-denominator accumulation; cached/uncached parity; deliberate interrupt/resume parity. Report numeric tolerances and discrepancies.

**Resources:** single short batches; CPU reference plus measured MPS memory. **Gate:** any causal, normalization, checkpoint or byte failure blocks scientific runs. A backend gap triggers faithful fallback/parity, not a different unrecorded mechanism. A tokenizer issue masks only demonstrably ambiguous auxiliary fields; coverage must remain scientifically adequate. Passing this package establishes harness correctness, not model improvement.

### WP2 — Build the controlled language and validate its semantics

**Objective/risk:** test an expressible causal mechanism with known answers and expose shortcut/ambiguity confounds. **Dependencies:** WP0 and the window/metric conventions from WP1. **Inputs:** Section 5.5 and Appendix B's concrete grammar contract.

**Actions/files:** implement `synthetic.py`; serialize grammar version, lexicon mapping, factor split, seeds and component/family/member IDs. Generate development worlds; compute a deterministic interpreter and annotation projection; enumerate the tiny limiting case. Create agreement, role, negation, reference and recursion counterfactual families, then form the context-overlap components before partitioning.

**Outputs/tests:** normalization where enumerated; every variant's truth answer consistent with its ledger; unique referents or explicitly flagged ambiguity; zero cross-partition context overlap; nonempty promised cells and answer classes; all atomic factors represented in training for composition splits; no lexical remapping between splits; token-identical final mentions for neutral history contrasts; achieved class/depth counts, component sizes, rejection causes/rates and shortcut baseline scores. Publish generator seeds/config, not a claim that templates are natural language.

**Resources:** CPU generation and a small scratch model after WP1. **Gate:** revise generator if answers are ambiguous, the baseline is at ceiling, or an unintended lexical cue solves the claimed dependency. Increase complexity only in development with a concrete diagnosed reason. A working toy harness is sufficient to proceed; a toy win is not required. Freeze protected generator families only at WP6.

### WP3 — Reproduce strong simple and structural baselines

**Objective/risk:** avoid an apparent gain caused by weak competitors. **Dependencies:** WP1 and prepared data; WP2 for controlled checks. **Inputs:** B0–B3 equations/configs, Qwen3 base.

**Actions/files:** implement ordinary LoRA, both B2 auxiliary variants, and exact TPT aggregation. Reproduce backbone branch-zero logits and hand-computed tree loss before adaptation. Tune B1 first to establish a usable ordinary learning curve. Attempt the GiLT local feasibility gate using permitted official code or a separately documented faithful implementation; inspect license before copying.

**Outputs/tests:** same input/scoring pipeline, attention-row fixture including multiple target subtokens, valid selected-head gradients, per-baseline learning curves, active parameter counts, and a GiLT feasibility/exclusion record. If GiLT is attempted, reproduce its incremental state and compare tiny enumerated marginal probabilities with beam estimates; no gold future graph at inference.

**Resources:** first measured main-model step and sustained calibration. GiLT graph search may dominate; do not scale beam blindly. **Gate:** baseline correctness and adequate tuning must precede competitiveness claims. An unavailable GiLT implementation blocks claims of beating the full current structured-LM frontier, not the narrower B1–B6 mechanism study. Log measured incompatibility or license blocker rather than describing an unattempted port as impossible.

### WP4 — Implement the minimum typed mechanism and controls

**Objective/risk:** establish that the intended channels, rather than extra generic capacity, are the experimental treatment. **Dependencies:** WP3. **Inputs:** Sections 5.1–5.4, frozen feature vocabularies and tap schedule.

**Actions/files:** implement `typed.py`, `continuous.py` and method registry for B4–B6/C1/C2; implement diagnostic intervention injection with normal inference defaulting to none. Add colocated/reversed-placement configurations for later attribution. Build dense targets with interval masks and globally normalized owned-event weights.

**Outputs/tests:** repeat WP1's primitive invariants on the integrated typed model before any pilot comparison: initial identity, expected isolated loss gradients, zero gold-data dependencies in `forward`, causal cache parity, whole-interval invalidation of unresolved P, and correct dtype casts. Check that category predictions have nontrivial coverage and that projections begin learning after initialization. Count trainable/useful/inert parameters separately.

**Resources:** tiny model then one main-model step; no broad tuning yet. **Gate:** if hard predictions collapse, diagnose labels, initialization and optimization against B2/B4. Do not compensate with gold feedback at test. If the implementation is correct but hard uncertainty is consistently inferior, keep the negative finding and allow soft feeding to become the selected practical outcome.

### WP5 — Pilot learning, headroom and local cost

**Objective/risk:** learn whether the chosen setting can distinguish mechanisms and what a rigorous run costs. **Dependencies:** WP3–WP4. **Inputs:** train/dev only, operationally safe model shape.

**Actions/files:** calibrate context, microbatch, accumulation, sustained throughput and checkpoint cost; run matched pilot learning curves for mandatory candidates; evaluate state predictability/ambiguity, generic capacity brackets, TPT+aux, soft versus hard feeding, and Qwen3.5 generic challenger. Record every trial. Compare train/dev divergence, rare-class behavior, gradient conflicts and event-gap strata.

**Outputs/tests:** `cost_forecast.csv`, memory traces/samples, learning curves, error taxonomy, real coverage, and a pilot variance table. Test checkpoint/resume under the selected actual optimizer/backend. Check that any apparent advantage survives adequate B1/B2/B3 tuning and shared optimizations.

**Gate/outcomes:** improve implementation if numerical/label faults exist; revise proxy if task ceiling or ambiguity makes H1 untestable; amend scale only when capacity/context evidence predicts a resolution. Correctly implemented absence of benefit at an adequately sensitive setting is evidence against that setting. Local inability to run a comparator is a scope limitation. Cost estimates alone never count as measured feasibility.

### WP6 — Lock the confirmatory experiment

**Objective/risk:** prevent adaptive selection and inadequate sensitivity. **Dependencies:** completed pilot and mandatory feasible baselines. **Inputs:** development evidence, cost/variance curves, protected tests still unscored.

**Actions/files:** select `B*` and one typed config; freeze exact training tokens/steps, seeds, data/context, tuning record, primary estimand, all planned contrasts, bootstrap algorithm and failure/replacement rules. Determine `n_confirm` by Section 7; make total local-time/storage forecast explicit. Freeze synthetic test combinations/seeds and any secondary benchmark subset. Write/hash `protocol/confirmation.md` and `configs/confirm.yaml`.

**Acceptance:** no unresolved placeholder affects execution, all expensive runs depend on passed gates, planned inference is supported by available independent units, and no test outcome has shaped selection. If precision is inadequate, increase independent evidence before inspection or explicitly register an estimation-only/inconclusive scope. There is no requirement to predict a positive result.

**Next:** run confirmation once. A changed primary hypothesis, comparator or evaluation population after this point requires an amendment; protected results already seen cannot become unseen again.

### WP7 — Execute fixed confirmation and preserve raw evidence

**Objective:** estimate H1 and the locked secondary contrasts without selective reporting. **Dependencies:** WP6 hash. **Inputs:** immutable configs/assets/seed list.

**Actions/files:** execute all locked paired training runs at the fixed token budget, store complete checkpoints, then evaluate GUM test and GENTLE with identical harness settings. Run the protected synthetic families. Produce per-document/per-family scores before aggregate analyses.

**Outputs/tests:** complete run inventory, failure rate, raw NLL/bytes, simultaneous comparison table where applicable, paired intervals and secondary results. Validate hash identity, target coverage and denominator checks automatically before computing a claim.

**Resources:** measured forecast from WP5 multiplied by locked trials/seeds, plus uncertainty and disk margin. **Gate:** bugs invalidate affected comparisons; hardware failures follow frozen recovery rules; normal unfavorable scores remain in results. Do not stop early on a favorable score or expand seeds on a borderline one. Unfinished fixed-sample studies are incomplete, not statistically confirmed.

### WP8 — Attribute effects and challenge their robustness

**Objective/risk:** distinguish explicit linguistic communication from labels, generic capacity, disruption and favorable slices. **Dependencies:** trained locked conditions; analyses either preregistered in WP6 or clearly exploratory. **Inputs:** Section 10's ablation/intervention matrix.

**Actions/files:** conduct route/placement/channel controls, predicted-state interventions, cost-frontier analysis, genre/source sensitivity and context/scale replication where justified. Use fresh held-out counterfactual families for new mechanistic hypotheses. Correct for the declared contrast family; retain harms.

**Outputs:** effect decomposition, natural-state accuracy, intervention validity/margin tables, errors by ambiguity/age/context and cost/quality frontier. **Gate:** if gains vanish under stronger baselines or generic routing, credit that explanation. If interventions merely damage all representations, do not claim named causal reasoning. A real BPB gain with unresolved attribution remains a predictive result only.

### WP9 — Reproduce, report and decide

**Objective:** produce an auditable outcome and an evidence-based continuation decision. **Dependencies:** WP7 and required WP8 analyses, or an earlier documented stopping condition.

**Actions/files:** recreate the locked environment in a fresh local directory, reproduce a selected full checkpoint's deterministic scores and one interrupted/restarted training segment, verify manifest/checksum provenance, and generate `reports/report.md` plus CSV tables. Include all runs, costs, uncertainty, failures, departures, limits and negative results. Apply Section 11's outcome rules.

**Acceptance:** another coding agent can execute the documented commands from the saved repository and public permitted assets; published tables derive from retained records; no external paid API/GPU is needed. If full-training replication is prohibitively costly, reproduce scoring/resume and explicitly label full retraining unverified. Consider the audio branch only when its separate hypothesis and annotation access pass Section 11's gate.

## 10. Ablations, robustness and failure analysis

### 10.1 Required attribution matrix

Run enough of this matrix to support the claims actually made. Predeclare confirmatory contrasts at WP6; later investigations remain exploratory until independently confirmed.

| Question / confound | Discriminating comparison | Interpretation |
|---|---|---|
| Is the effect just ordinary adaptation? | Typed versus tuned B1; B0 establishes starting loss | Beating B0 alone is insufficient |
| Is supervision sufficient? | Typed versus both B2 variants, including useful capacity brackets | If equal, prefer the simpler auxiliary method |
| Is structure-aware training sufficient? | Typed versus TPT and TPT+all auxiliary labels | A cheaper training-only objective may dominate |
| Is a real-valued generic route sufficient? | Typed versus B6 at active parameter and end-to-end cost brackets | Equal performance weakens a claim about named finite states |
| Is the difference caused by gradient policy? | Hard versus detached-soft versus joint-soft B4 | Separate finite messages from blocked task gradients |
| Do direct routes help? | Typed versus B5, plus extra useful generic capacity replacing deleted R/U | H2 concerns the direct route; residual communication remains |
| Does placement across depth matter? | Distributed taps versus all four sequential groups after the syntax tap, and reversed group order P→E→S→M across the four taps | Retain causal connections only from already computed groups. A depth-hierarchy claim requires beating suitable colocated/reordered controls |
| Is any one label group doing everything? | Retrain M-only, S-only, E/P-only when pilot attribution points to them; full model minus each consequential group | Test a specific suspected dependency, not every combinatorial subset by default |
| Are labels meaningful, or just regularization? | Event-level shuffled training labels C1, stratified by genre, length and missingness; preserve each field vector and its interval weights | A benefit from shuffled labels suggests a regularization/capacity explanation |
| Is “meaning” merely the category numbering? | Consistently permute category IDs, predictor outputs and embedding rows at evaluation | Must preserve outputs under strict maxima; this is an invariance test, **not** a negative semantic control |
| Are hard errors costly? | Error/entropy/margin strata; valid single-field correction and matched donor substitution | Distinguish predictable states from unresolvable prefix ambiguity |
| Do old mention states help? | Word/mention overlap, gap and event-age strata, selected before test scores | Detect stale-state shortcuts; never add gold age as a forward feature |
| Does document memory regress? | Whole-document BPB, repeated-entity/copy-sensitive slices, long-gap/context strata | Syntactic success can coexist with document loss |
| Does the effect transfer? | GUM genres, GENTLE, independent lexicon worlds, and a second calibrated model scale | Transfer is a separate claim; do not average a harmful setting out of view |
| Is the result driven by implementation cost? | Shared exact optimizations and each method's best validated implementation | Report both scientific control and practical cost frontier |

The E field encodes entity type, not predicate roles or polarity. A positive role/negation/denotation score in the synthetic task is a **system-level generalization result**. It does not prove that E contains a role-binding or truth representation. Version 1 has no type-sensitive query: H4's directional correctness claims are therefore restricted to declared M/S/P interventions with interpreter-defined expected outcomes. E supports prediction, ablation and sensitivity results only; an E intervention changing output is not evidence of appropriate semantic use. Adding a balanced type-sensitive query or role/polarity fields would be a separately named extension with equal supervision for all comparators.

### 10.2 Intervention protocol

Intervention experiments ask whether a named channel changes behavior in the intended way, not merely whether perturbing a neural network changes its output.

1. Use held-out synthetic counterfactual families with a known interpreter to define the expected direction of the output change. Keep other relevant factors matched.
2. Intervene on one declared field/message at one declared position or interval. Record its natural prediction, target, confidence margin, replacement, source donor and intervention norm. Recompute downstream groups consistently; do not splice cached states from incompatible prefixes.
3. Compare natural inference, route removal, a valid corrected state, a valid alternative state from a matched donor, and a norm-matched generic perturbation. Include interventions that should have no effect on the queried property.
4. Report both intended changes and collateral harm, conditioning on whether the natural state was correct. Do not report only cases selected because an intervention succeeded.
5. On real text, replacing a prediction with a gold future-informed label is an **oracle diagnostic**. Its benefit cannot be added to normal-inference performance or used as attainable headroom. Matched donors reduce but do not eliminate off-distribution concerns.
6. If the residual stream reconstructs or overrides the state, small intervention effects may reflect redundancy. If every perturbation causes comparable harm, large effects may reflect generic disruption. Neither result alone establishes an interpretable linguistic mechanism.

Concept leakage work and the inspected concept/residual systems motivate these precautions; they do not prove that hard messages are automatically faithful or that correct partial interventions always help. [R18], [R19], [R20]

### 10.3 Domain-specific adversarial fixtures

Maintain small explicit fixtures for all of the following before expensive training:

- Identical prefixes with suffix-dependent entity boundaries or different full-tree labels: forward outputs must remain identical.
- A mention with valid E but `P=auto` following a known-P mention: E updates and P stays masked across the entire new interval.
- Same-GRP nested spans, an equal-span conflicting label, an escaped hyphen, a discontinuous mention, a multiword token and an empty node.
- Two UD words covered by one LM token, a UTF-8 character split across byte tokens, a leading whitespace token and a word cut by a scoring window.
- A selected word with a root, an unseen right head, and a left head outside the context; distinguish all cases.
- A TPT target word with several subtokens and a prior key word with several subtokens, including a block-boundary owner.
- A state-logit tie, a strict-margin ID permutation, a lower-precision near-tie flip, and checkpoint recomputation that must not duplicate state outputs.
- Two source-linked documents in different genres; bootstrap them together.
- Synthetic agent/patient reversal with unchanged word multiset, polarity flip with unchanged entity type, and reference-history change with the same final mention.
- A document that yields only `NONE_YET` or absent-feature targets in a channel: state accuracy must not appear strong solely because these default classes dominate.

### 10.4 Failure taxonomy

| Observation | Classification | Required response |
|---|---|---|
| Future suffix changes earlier logits beyond numerical effects; duplicate targets; bad resume | Invalid implementation/measurement | Stop dependent runs, repair, rerun affected comparisons |
| Token/label mismatch or inadequate resolved coverage | Data/interface failure | Repair projection or narrow scope before confirmation |
| Synthetic baseline at optimum; all labels constant; confidence intervals too wide | Unsuitable proxy or insufficient sensitivity | Redesign development task or gather independent evidence; no hypothesis rejection |
| Main model cannot train within stable memory on any faithful local path | Local resource limitation | Report blocked main claim; retain narrower validated component result |
| Typed equals auxiliary or generic routes after adequate sensitivity | Mechanism adds no demonstrated advantage | Prefer simpler method; record bound on possible improvement |
| Typed is worse with narrow paired intervals and credible controls | Evidence against tested design/setting | Stop that configuration; consider only a concrete diagnosed revision |
| Synthetic succeeds, real data fails | Limited mechanism result | Report both; investigate annotation/prefix ambiguity, do not call full proposal validated |
| Larger local scale reverses a smaller-scale negative | Scale dependence | Confirm the interaction; do not select only the favorable scale |
| Strong primary score but costly inference or important genre harm | Tradeoff, not unqualified win | Report frontier/regression and restrict practical claim |

## 11. Outcome interpretation and evidence-triggered continuation

### 11.1 Decision rules

**Continue with a differentiated empirical contribution** if protected H1 is supported, effect estimates survive strong baseline tuning and matched information/capacity controls, important regressions are visible and acceptable for a stated use, and some predicted mechanism/robustness evidence survives independent confirmation. Describe exactly the checkpoint, data, scale and cost regime. A paper-worthy result could be a small reliable gain, a useful cost tradeoff, or a strong negative result; the word “breakthrough” requires broader evidence than this local study supplies.

**Prefer an existing simpler method** if B2, TPT+aux or soft feeding matches/wins at better cost. This still answers the original question: explicit linguistic training may help even when the proposed architectural communication does not. Do not rename that outcome as invention of multitask learning or concept bottlenecks.

**Narrow the claim** if the mechanism works only synthetically, on one genre, under unusually clean labels, or on one local scale. State the boundary and the smallest new experiment capable of challenging it.

**Declare inconclusive** if the data, prefix ambiguity, variance or local resources prevent resolving plausible effects. Give the achieved interval and missing evidence. “Not significant” is not evidence of equivalence; a positive point estimate is not evidence of improvement.

**Stop the tested direction** if a correct implementation and sensitive comparison repeatedly show no useful gain over credible simpler alternatives, or if unavoidable overhead dominates the measured benefit. This is a conclusion about the tested design and scale. Escalate only when an observed capacity/context/optimization limitation predicts a specific larger-scale change that can itself be tested.

### 11.2 Conditional pivots

| Evidence | Justified next experiment | What must remain fixed / newly controlled |
|---|---|---|
| B2 gains; all feedback variants fail | Study annotation-efficient auxiliary supervision or label selection | Same labels/budgets; compare learned task weighting and established multitask methods |
| TPT dominates coarse S | Test a fuller predicted dependency mechanism, including GiLT if locally faithful | Prefix-only inference and marginal probability accounting; no oracle graph |
| Soft feeding beats hard channels on ambiguous states | Test calibrated uncertainty/abstention as an explicit new hypothesis | Match gradient policy and capacity; acknowledge soft-channel leakage |
| E/P states are predictable but too coarse | Add an independently annotated role/reference task, such as a licensed ReCOGS extension | Strong structured semantic baselines, semantic exact-match scorer and fresh confirmation |
| Morphological failures track tokenizer fragmentation | Compare segmentation/character or byte-level alternatives | Match raw text, vocabulary capacity and end-to-end processed-token cost |
| Practical Qwen3.5 comparator wins decisively | Replicate channels on its verified hybrid path or adopt the stronger generic model | Same-backbone attribution still needed; separate cache/kernel validation |
| Only extreme recursion/context reveals a benefit | Calibrate a locally feasible larger-context/width replication | Establish that the smaller setting lacked the tested dependency; no indefinite scale excuse |

ReCOGS/COGS-style experiments would require a structural semantic parser/constrained-decoding comparator, not just a weak historical flat seq2seq baseline. Their representation and split conventions differ; do not combine published scores as if they measured the same task. [R42]

### 11.3 Explicit phonology/phonetics continuation

The text PoC does not demonstrate all six levels. A continuation is justified if the remaining scientific question is specifically whether explicit sound-related states improve a local speech/language task, not simply to complete a six-row diagram.

For a **phonological** branch, a public pronunciation lexicon such as CMUdict supports phoneme, syllable/stress or rhyme targets. Start with dictionary lookup and a simple grapheme-to-phoneme baseline, split by lexeme/family rather than random pronunciation entries, and report unseen-word generalization. Symbolic pronunciation supervision does not constitute acoustic phonetics. [R49]

For an **acoustic** branch, use permitted public recordings rather than requiring a microphone or new physical equipment. `facebook/wav2vec2-base` supplies a public pretrained speech encoder; its card specifies 16 kHz input and Apache-2.0 licensing. LibriSpeech supplies CC BY 4.0 audio/transcripts, but not gold phone boundaries. The selected acoustic labels/contrasts must therefore be independently available or generated by an explicitly audited alignment pipeline. [R50], [R51]

First freeze the same encoder in every condition, cache frame features in bounded shards, and compare generic acoustic adapters, the same auxiliary sound labels without feedback, and explicit finite sound-state feedback. Define one sound-sensitive hypothesis: for example, improved within/across-speaker phone discrimination when transcripts are held constant, or improved phoneme recognition on unseen speakers. A ZeroSpeech-style ABX measure is a possible endpoint once its dataset/label license is resolved. Cluster uncertainty by speaker and lexical item; reused contrast triplets are not independent. ASR word error rate may be secondary but does not establish acoustic discrimination by itself. [R52]

Before adding a joint six-level pipeline, show that the sound branch contributes information unavailable in text, survives speaker/channel controls, and beats the same-encoder generic/auxiliary baselines. Then test whether connecting it to the text channels adds value beyond parallel multitask supervision. Multilevel acoustic/phonetic/lexical factorization already exists, so search and compare that work before claiming originality. [R22]

This branch is **optional and unexecuted**. No audio data was downloaded or speech model run during preparation. Its memory/time forecast must come from local calibration. If suitable independent phone labels cannot be used, stop the phonetics claim; transcripts alone cannot replace them.

### 11.4 Required final research report

The execution report must state, in this order: what claim was tested; which gates passed; what was actually run; primary paired effect and uncertainty; strongest baseline and practical challenger; attribution/robustness; cost and regressions; failures/amendments; remaining limits; and a continue/revise/narrow/stop decision.

Separate **source facts**, **executed measurements**, **derived estimates**, **exploratory observations**, and **confirmed findings**. Provide the original seed/document-level data and programmatic aggregation. Keep estimates and unknowns visible. A report with no positive result can still satisfy the research objective if it resolves the question honestly.

## 12. Audit record and unresolved risks

Three parallel agent reviewers challenged the draft from scientific/prior-art, experimental-validity and implementation/feasibility perspectives, followed by targeted rechecks after revisions. These were AI-assisted research/design audits, not independent human peer review and not experimental replication. The final targeted passes found no further material design correction within their scopes after the listed fixes and synchronization of the synthetic statistical unit. That conclusion does not certify the unimplemented generator, target backend or future measurements. The preparation also included actual public-file inspection and small CPU algebra checks; Appendix A separates those from proposed execution.

| Audit | Concrete defect found | Revision applied | Remaining execution gate |
|---|---|---|---|
| Scientific/prior art | Dependent hard concepts and side channels could be misread as novel | Added Havasi/CB-LLM/SupCBM overlap; framed contribution as a causal, depth-distributed empirical adaptation | No method novelty claim without a further exact-equivalent check if the method changes |
| Scientific/prior art | Generic semantic/six-level claims exceeded annotations | Restricted E/P meaning, separated role/negation outcomes, specified acoustic continuation | Four text channels cannot establish all six levels |
| Scientific/prior art | A reverse/colocated depth claim lacked a control | Added explicit causal placement variants | Run them before claiming that the chosen depth hierarchy matters |
| Experimental validity | Initial synthetic wording changed lexical meaning across splits | Shared lexicon within each world; independent bijections across retrained worlds | Generator balance, answerability and shortcut audit |
| Experimental validity | TPT could be implemented with the wrong attention rows or undefined cross-sentence distances | Inspected paper and official source; specified all shifted next-word subtoken rows, sentence-local keys, unique word ownership and selected attention output | Hand-computed multi-subtoken fixture and baseline tuning |
| Experimental validity | Repeated dense events could be overweighted, and stepwise normalization undid equal-event weighting | Full-pass event weights with fixed `K/W_f` scaling, common logical steps and two-event density fixture | Production gradients and target-machine accumulation parity |
| Experimental validity | Unknown P could retain an old mention | Invalidating interval masks; same-event E/P | Tokenizer alignment and coverage on target software |
| Experimental validity | Held-out relations could appear in truth questions; different questions could split the same world | Asserted-and-queried tuple exclusion, tested-tuple requirement and connected context-family components | Execute generator split/duplicate audit before neural training |
| Scientific/experimental validity | Counterfactuals could contradict facts; a depth cap was mistaken for realized depth; neutral determiners left a numeral shortcut | Whole-family validation, exact achieved-depth strata, explicit `a/the/two` replacement and scaffold masks | Verify nonempty sampling cells, valid answers and token-identical history contrasts |
| Experimental validity | Method-specific best checkpoints could violate token matching; cross-tokenizer blocks could depend on future text | Common locked horizon for matched comparisons; fixed code-point blocks for B7 | Report independent-horizon cost frontier separately |
| Experimental validity | Singleton genre strata and cross-genre source clusters invalidated naïve resampling | Pooled whole-source crossed bootstrap, finite-corpus sensitivity and coverage gate | Limited document count and distribution assumptions |
| Experimental validity | Divergent runs/seeds could be selectively replaced | Fresh confirmation seeds; explicit infrastructure-versus-algorithm failure contract | Sufficient successful unconditional replication |
| Implementation/feasibility | Undefined first-token/window/byte convention; final-window start depended on document ending | Exact start-sentinel/ownership, prefix-independent context start, canonical-document bytes; 3,937 arithmetic window cases | Byte-BPE round trip and cache parity |
| Implementation/feasibility | Missing CLI module, unselected interpreter and WP1/WP4 gate cycle | Explicit package entrypoint/environment; primitive tests in WP1 repeated on integrated WP4 model | Fresh-machine execution of the proposed project CLI |
| Implementation/feasibility | Hard gradients, ties, mixed dtypes and recomputation had ambiguous behavior | Gradient ownership, strict-margin tests, dtype casts and functional checkpointed outputs | Actual MPS backward and precision checks |
| Implementation/feasibility | Nominal Apple support and adapter saves could be mistaken for a usable training/resume path | Current docs/source inspection, full-state checkpoints, measured local calibration | No M3 measurement or compatible final lock yet |
| Implementation/feasibility | Cross-tokenizer context and memory accounting could be unfair | Common raw-text diagnostic and explicit process/system memory distinction | Actual contemporary-challenger training/cost |

**Residual scientific risks:** annotations may be predictable but irrelevant; hard predictions may discard too much ambiguity; residual routes may absorb the whole task; GUM may be too small or contaminated for broad conclusions; stronger generic models may dominate; a genuinely scale-dependent effect may remain beyond the laptop. The plan turns each into a measured limitation, falsifiable claim or blocking gate rather than promising a result.

**Stress-case dispositions:** already-known mechanism → replication/empirical claim; strongest reference unavailable → narrowed named-comparator claim; low memory/unsupported kernel → faithful fallback or local-feasibility failure; unavailable/licensed-out assets → pre-model population amendment; toy-only success → component result; tiny/unstable gain → uncertainty and practical frontier; hidden regression → restricted claim; inconsistent resume → invalidate affected runs; negative outcomes → retained evidence and a stop/revise decision.

## 13. References and search-coverage register

### 13.1 Search scope, inspection and stopping rationale

Research was conducted through **23 October 2026**, using two web-search systems, direct primary-paper retrieval, author repositories, official model/framework/corpus documentation, and targeted public-file/code inspection. Recent model/framework checks and final 2026 searches used date-sensitive queries. Secondary pages were discovery aids; consequential technical claims above rely on the primary sources listed below.

| Search family / representative query terms | Material result |
|---|---|
| “linguistic levels neural architecture”, “low level tasks lower layers”, “joint many-task”, “predicted label embeddings” | Direct overlap with deep supervision, JMT and LISA |
| “sluice networks”, “structured attention”, “semantic dependency finetuning” | Learned sharing and graph-mediated computation already exist |
| “causal syntactic language model”, “Transformer Grammars”, “pushdown”, “dependency Transformer”, “GiLT” | Explicit generation-state alternatives; sentence/document and marginal-likelihood caveats |
| “tree planted transformer next word distance”, followed through official source | Correct predictive shift and multi-subtoken attention aggregation |
| “hard linguistic bottleneck”, “autoregressive concept bottleneck”, “hierarchical concepts side channel”, “concept bottleneck LLM generation” | Havasi, CB-LLM, SupCBM and Steerling substantially constrain novelty |
| “syntax strong pretrained baseline”, “syntax noisy parses”, “learning syntax without trees”, current structural-transfer work | Counterexamples to guaranteed benefit and to equating lower LM loss with broad competence |
| “GUM infstat auto Entity”, UD r2.18 / GUM V12.1.0 / GENTLE split and license | Exact data/schema counts, unresolved-label masking and source-specific rights |
| Official Apple M3 Max, PyTorch MPS, MLX training/source, Qwen3.5 Base, bitsandbytes Apple | Conditional hardware path; current challenger; obsolete support assumptions corrected |
| Compositional semantics, ReCOGS/SyntaxGym/BLiMP, pronunciation/ABX, hierarchical audio codec | Appropriate optional diagnostics and a genuine sound-modality boundary |
| Hashimoto/Xiong/Socher linguistic-task patent text | Technical patent prior art; no legal clearance conclusion |

Backward/forward tracing connected GiLT to RNNG, TG, Pushdown, DTG and oracle graph models, and connected concept-feedback variants to classifier chains, latent side channels and generative concept models. A final refined search surfaced SupCBM, a new structural-transfer counterexample and COCO-Tree, an external vision-language concept-reasoning pipeline. These reinforce known method families rather than establish a new one. COCO-Tree's use of a separate linguistic reasoner is outside the chosen internal causal mechanism. [R53], [R54], [R55]

Search stopped expanding the core architectural comparison once the selected treatment had a close hierarchical-feeding equivalent, a close dependent-concept equivalent, a strong cheap structural comparator, and a contemporary graph/reference and generic-model challenger, while follow-up queries no longer changed the selected test or its information assumptions. This is a reasoned coverage judgment, **not proof of exhaustive prior art**, novelty, patentability or freedom to operate.

**Blind spots/access limits:** no private/unpublished systems; no exhaustive patent claims analysis; some raw GitHub/publisher requests failed and were replaced with accessible primary mirrors/HTML; GiLT's full implementation/license and several optional benchmark reuse terms remain unresolved; no local Mac, trained model benchmark, final environment lock or acoustic run. A late-2026 discrete-bottleneck search lead whose full text could not be inspected is not used to support a material claim. Bibliographic dates distinguish preprints from publications; mutable documentation is cited by inspected version/access date.

### 13.2 Source register

All online sources below were accessed during the 23 October 2026 research session. “Inspected” states what was actually retrieved/read; it does not mean the experiments were reproduced. Paper URLs are primary author/publisher repositories. Numbered citations in the main text resolve to these stable URLs.

- **R01.** Anders Søgaard and Yoav Goldberg (2016), *Deep multi-task learning with low level tasks supervised at lower layers*, ACL. [Paper][R01]. Inspected method/architecture. Direct precedent for depthwise linguistic supervision; recurrent tagging tasks, not causal modern LM adaptation.
- **R02.** Kazuma Hashimoto, Caiming Xiong, Yoshimasa Tsuruoka and Richard Socher (2017), *A Joint Many-Task Model: Growing a Neural Network for Multiple NLP Tasks*, EMNLP. [Paper][R02]. Inspected hierarchy and probability-weighted label embeddings. Direct prediction-feeding overlap; bidirectional/task-specific setting.
- **R03.** Emma Strubell, Patrick Verga, Daniel Andor, David Weiss and Andrew McCallum (2018), *Linguistically-Informed Self-Attention for Semantic Role Labeling*, EMNLP. [Paper][R03]; [code](https://github.com/strubell/LISA). Inspected architecture/results and README. Syntax-supervised attention and multilevel tasks; full-input SRL and legacy TensorFlow.
- **R04.** Sebastian Ruder, Joachim Bingel, Isabelle Augenstein and Anders Søgaard (2019; preprint 2017), *Latent Multi-task Architecture Learning*, AAAI, originally Sluice Networks. [Paper][R04]. Inspected sharing formulation and legacy repository requirements. Learned task/layer sharing precedent; not a direct causal-LM score comparison.
- **R05.** Chris Dyer, Adhiguna Kuncoro, Miguel Ballesteros and Noah A. Smith (2016), *Recurrent Neural Network Grammars*, NAACL. [Paper][R05]; [code](https://github.com/clab/rnng). Inspected generative formulation and corrigendum notice. Recursive structural generation; latent-structure marginalization matters.
- **R06.** Yoon Kim, Carl Denton, Luong Hoang and Alexander M. Rush (2017), *Structured Attention Networks*, ICLR. [Paper][R06]. Inspected structured marginalization and method overview. Differentiable structural distributions are established; tasks and costs differ.
- **R07.** Laurent Sartran, Samuel Barrett, Adhiguna Kuncoro, Miloš Stanojević, Phil Blunsom and Chris Dyer (2022), *Transformer Grammars: Augmenting Transformer Language Models with Syntactic Inductive Biases at Scale*, TACL, DOI 10.1162/tacl_a_00526. [Paper][R07]. Inspected mechanism and sentence/document tradeoff. Supports syntax benefits and the copying/document guardrail; no local reproduction.
- **R08.** Shikhar Murty, Pratyusha Sharma, Jacob Andreas and Christopher D. Manning (2023), *Pushdown Layers: Encoding Recursive Structure in Transformer Language Models*, EMNLP. [Paper][R08]; [code](https://github.com/MurtyShikhar/Pushdown-Layers). Inspected formulation/limitations and repo landing. Incremental stack-state attention precedent; runtime/license not validated.
- **R09.** Yida Zhao, Chao Lou and Kewei Tu (2024), *Dependency Transformer Grammars: Integrating Dependency Structures into Transformer Language Models*, ACL. [Paper][R09]. Inspected transition-system and modeling sections. Dependency-based architectural bias; structural-token/search differences prevent uncontrolled PPL comparison.
- **R10.** Ryo Yoshida, Taiga Someya and Yohei Oseki (2024), *Tree-Planted Transformers: Unidirectional Transformer Language Models with Implicit Syntactic Supervision*, Findings ACL. [Paper][R10]; [official training source](https://github.com/osekilab/TPT/blob/main/src/train.py). Inspected equations, controls, limitations and downloaded training/preprocessing source. Informs mandatory B3 and predictive-row aggregation. Project adaptation uses different corpus/backbone; code-reuse license still requires confirmation.
- **R11.** Tianyu Huang, Yida Zhao, Chuyan Zhou and Kewei Tu (15 May 2026), *GiLT: Augmenting Transformer Language Models with Dependency Graphs*, arXiv:2605.15562v1; repository labels ACL 2026. [Paper][R11]; [repository](https://github.com/cookie-pie-oops/GiLT-LM). Inspected method, probability/search conventions, results and environment. Most consequential recent structural reference. Published GPU cost is not an M3 forecast; some source paths/license were inaccessible.
- **R12.** Devendra Singh Sachan, Yuhao Zhang, Peng Qi and William L. Hamilton (2021), *Do Syntax Trees Help Pre-trained Transformers Extract Information?*, EACL. [Paper][R12]. Inspected method/findings. Predicted parse quality can limit gains; full-input extraction, not universal evidence against syntax.
- **R13.** Kabir Ahuja, Vidhisha Balachandran, Madhur Panwar, Tianxing He, Noah A. Smith, Navin Goyal and Yulia Tsvetkov (2025), *Learning Syntax Without Planting Trees: Understanding Hierarchical Generalization in Transformers*, TACL, DOI 10.1162/tacl_a_00733. [Paper][R13]. Inspected objective/dataset argument. Challenges necessity of explicit structure; controlled-language conditions.
- **R13a.** Junqi Dai, Hang Yan, Tianxiang Sun, Pengfei Liu and Xipeng Qiu (2021), *Does syntax matter? A strong baseline for Aspect-based Sentiment Analysis with RoBERTa*, NAACL. [Paper][R13a]. Inspected motivation/method findings. Strong ordinary adaptation can erase gains in ABSA; do not generalize beyond that setting.
- **R14.** Zhiling Zhou, Tianhao Wang and Zhuoran Yang (2026), *How Transformers Represent Hierarchies: A Local-to-Global Mechanism*, ICML/PMLR 306. [Paper page][R14]. Inspected abstract and construction summary. Ordinary Transformers can represent grammar structure under controlled PCFG assumptions.
- **R15.** Laura Ying Schulz, Daniel Mitropolsky and Tomaso Poggio (2026), *Unraveling Syntax: Language Modeling and the Substructure of Grammars*, ICML/PMLR 306. [Paper page][R15]. Inspected theorem/experimental summary. Grammar-relative scale affects pretraining benefits; alignment is not equivalent to predictive improvement.
- **R16.** Zhaofeng Wu, Hao Peng and Noah A. Smith (2021), *Infusing Finetuning with Semantic Dependencies*, TACL, DOI 10.1162/tacl_a_00363. [Paper][R16]. Inspected semantic graph fusion/SIFT-light discussion. Semantic supervision precedent; full-input NLU and parser costs differ.
- **R17.** Jakob Prange, Nathan Schneider and Lingpeng Kong (2022), *Linguistic Frameworks Go Toe-to-Toe at Neuro-Symbolic Language Modeling*, NAACL. [Paper][R17]. Inspected oracle graph setting and local graph encoding. Seven-formalism comparison; oracle-conditioned gains are not autonomous prefix-only gains.
- **R18.** Pang Wei Koh, Thao Nguyen, Yew Siang Tang, Stephen Mussmann, Emma Pierson, Been Kim and Percy Liang (2020), *Concept Bottleneck Models*, ICML. [Paper][R18]. Inspected formulation/overview. Foundational concept-supervision/intervention precedent; our residual side route is not a complete bottleneck.
- **R19.** Marton Havasi, Sonali Parbhoo and Finale Doshi-Velez (2022), *Addressing Leakage in Concept Bottleneck Models*, NeurIPS. [Paper][R19]; [code](https://github.com/dtak/addressing-leakage). Inspected sections 4.1–4.2 and inference/training distinctions. Direct dependent-concept and latent-side-channel overlap; sampled concept inference differs from our deterministic predicted chain.
- **R20.** Chung-En Sun, Tuomas Oikarinen, Berk Ustun and Tsui-Wei Weng (2025), *Concept Bottleneck Large Language Models*, ICLR. [Paper][R20]; [project](https://lilywenglab.github.io/CB-LLMs/). Inspected generation method, residual/adversarial design and metric definitions. Direct concept-plus-residual generation overlap; external-LM likelihood of generated text is not our endpoint.
- **R21.** Guide Labs Team—Andreas Madsen, Aya Abdelsalam Ismail, Giang Nguyen, Isaac Plant, Muawiz Chaudhary, Nathaniel Monson, Saqib Azim, Zhichen Guo and Julius Adebayo (2026), *Scaling Inherently Interpretable Language Models*, arXiv:2608.07594v1. [Report][R21]. Inspected concept module and residual ablation. Contemporary concept-mediated LM context; different scale/diffusion recipe and unreplicated interpretability claims.
- **R22.** Sameer Khurana, Dominik Klement, Antoine Laurent et al. (2025), *Factorized RVQ-GAN For Disentangled Speech Tokenization*, Interspeech, DOI 10.21437/Interspeech.2025-2612. [Paper][R22]. Inspected abstract/method motivation. Hierarchical Audio Codec separates acoustic/phonetic/lexical representations; codec results do not establish the proposed LM benefit.
- **R23.** Amir Zeldes (2017), *The GUM Corpus: Creating Multilayer Resources in the Classroom*, Language Resources and Evaluation 51, DOI 10.1007/s10579-016-9343-x; GUM contributors (2026), V12.1.0. [Repository][R23]; [project](https://gucorpling.org/gum/). Inspected documentation/releases and downloaded pinned splits/license/sample. Informs joint-layer data; current counts are measured here.
- **R24.** Tatsuya Aoyama, Shabnam Behzad, Luke Gessler, Lauren Levine, Jessica Lin, Yang Janet Liu, Siyao Peng, Yilun Zhu and Amir Zeldes (2023), *GENTLE: A Genre-Diverse Multilayer Challenge Set for English NLP and Linguistic Evaluation*, LAW-XVII. [Paper][R24]. Inspected composition/annotation sections. External genre challenge; test-only and small.
- **R25.** Universal Dependencies/GUM contributors (2026), *UD English GUM*, r2.18. [Repository][R25]; [format specification](https://universaldependencies.org/format.html). Inspected README/format and all four relevant released data files across GUM/GENTLE. Supports schema and measured manifest; tokenizer alignment remains untested.
- **R26.** GUM/GENTLE contributors (2026), corpus licensing. [GUM license][R26]; [GENTLE annotation/source terms](https://gucorpling.org/gum/cite_gentle.html); [UD GENTLE aggregate license](https://raw.githubusercontent.com/UniversalDependencies/UD_English-GENTLE/r2.18/LICENSE.txt). Inspected/downloaded terms. Annotation permission differs from underlying text; no blanket redistribution or model-license conclusion.
- **R27.** Universal Anaphora contributors, *CoNLL-U Entity format proposal* (current document, accessed 2026). [Specification][R27]. Inspected headers, bracketing, escapes and discontinuities. Informs parser; actual pinned subset audited separately.
- **R28.** GUM/UD contributors (2026), *GUM V12.1.0 splits* and *UD English GENTLE r2.18*. [Split map][R28]; [GENTLE repository](https://github.com/UniversalDependencies/UD_English-GENTLE/tree/r2.18). Downloaded and compared document IDs; exact no-Reddit/test2 equality verified.
- **R29.** Kazuma Hashimoto, Caiming Xiong and Richard Socher / Salesforce (10 October 2023 publication; listed priority 3 November 2016), US11783164B2, *Joint many-task neural network model for multiple natural language processing (NLP) tasks*. [Patent text][R29]. Inspected bibliographic record/description, not an exhaustive claim chart. Technical prior-art lead only.
- **R30.** Apple (2023 hardware; current support page), *MacBook Pro 14-inch, M3 Pro or M3 Max—Technical Specifications*. [Specs][R30]. Inspected memory/chip variants/bandwidth. Hardware facts only, no training performance evidence.
- **R31.** Qwen Team (2025), *Qwen3-0.6B-Base model card and config*. [Card][R31]; [config](https://huggingface.co/Qwen/Qwen3-0.6B-Base/raw/main/config.json). Inspected pretraining/license/config and source shape implications. Mutable revision must be frozen; parameter arithmetic is our deduction.
- **R32.** Qwen Team (2026), *Qwen3.5-0.8B-Base*. [Card][R32]. Inspected pretraining-only statement and Apache-2.0 metadata. Current challenger, not a measured winner on GUM or M3.
- **R33.** PyTorch contributors (2026), *MPS backend*, version 2.14 documentation. [Docs][R33]. Inspected availability/OS requirements; no custom training execution.
- **R34.** Apple/MLX contributors (2026), *Build and Install*, rendered documentation 0.32.4. [Docs][R34]. Inspected OS/Python/native architecture requirements. MLX fallback still requires model and gradient parity.
- **R35.** Irakli Salia and Khushi Agrawal, Hugging Face PyTorch team (20 October 2026), *MPS in PyTorch 2.15*. [Announcement][R35]. Inspected contributor role, release-candidate status and requirements. Context for version selection; not a validated released lock or M3 benchmark.
- **R36.** MLX-LM and Transformers contributors (2026), Qwen model implementations. [MLX Qwen3.5 source][R36]; [Transformers source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py); [MLX Qwen3](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/qwen3.py). Inspected training-mode delta paths, text-weight filtering and caches. No execution; source commits must be pinned.
- **R37.** HuggingFaceTB (2024 checkpoints; 2025 paper), *SmolLM2-135M/360M*. [135M card][R37]; [360M card](https://huggingface.co/HuggingFaceTB/SmolLM2-360M). Inspected configs/license. Calibration/replication assets, not assumed contemporary winners.
- **R38.** Bitsandbytes contributors (2026), *Installation Guide*, version 0.50.2. [Docs][R38]. Inspected Apple support and arm64 wheel listing. Corrects obsolete blanket claims; does not prove every quantized training kernel works.
- **R39.** PyTorch/MLX contributors (2026), scaled-dot-product attention APIs. [PyTorch docs][R39]; [MLX docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.scaled_dot_product_attention.html). Inspected mask/GQA/cache alignment conventions. Mathematical/kernel parity is an execution gate.
- **R40.** Jennifer Hu, Jon Gauthier, Peng Qian, Ethan Wilcox and Roger P. Levy (2020), *A Systematic Assessment of Syntactic Generalization in Neural Language Models*, ACL; Jon Gauthier et al. (2020), *SyntaxGym*, ACL demos. [Study][R40]; [platform paper](https://aclanthology.org/2020.acl-demos.10/). Inspected task/metadata and code landing. Optional syntax endpoint; selected-suite license/harness still a gate.
- **R41.** Alex Warstadt et al. (2020), *BLiMP: The Benchmark of Linguistic Minimal Pairs*, TACL, DOI 10.1162/tacl_a_00321. [Repository][R41]. Inspected benchmark organization. Optional diagnostic; license/contamination conventions not established for this project.
- **R42.** Zhengxuan Wu, Christopher D. Manning and Christopher Potts (2023), *ReCOGS: How Incidental Details of a Logical Form Overshadow an Evaluation of Semantic Interpretation*, TACL, DOI 10.1162/tacl_a_00623. [Paper][R42]; [code](https://github.com/frankaging/ReCOGS). Inspected evaluation discussion/SEM and README. Optional semantics continuation; no Mac port or comparable structured-parser reproduction.
- **R43.** PyTorch contributors (2026), *torch.mps*, version 2.14. [API][R43]. Inspected synchronization, allocator, working-set and RNG functions. Sampled memory is not a proven transient peak.
- **R44.** MLX contributors (2026), *Metal/memory management* and checkpointing. [API][R44]; [checkpoint](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.checkpoint.html). Inspected current memory/evaluation APIs. No measured memory result here.
- **R45.** PyTorch contributors (2026), *MPS Environment Variables*, version 2.14. [Docs][R45]. Inspected allocation/fallback flags. Supports safe resource configuration; no need to disable allocator limits.
- **R46.** MLX-LM contributors (2026), *tuner/trainer.py*. [Source][R46]. Inspected accumulation, checkpoint wrapper and saving. Model-weight saves omit required full experiment state; our resume requirements are a deduction from code inspection.
- **R47.** Rotem Dror, Gili Baumer, Segev Shlomov and Roi Reichart (2018), *The Hitchhiker's Guide to Testing Statistical Significance in Natural Language Processing*, ACL. [Paper][R47]. Inspected methodological overview. Informs unit/test selection; does not specifically validate this proposed crossed bootstrap.
- **R48.** Xavier Bouthillier, Pierre Delaunay, Mirko Bronzi et al. (2021), *Accounting for Variance in Machine Learning Benchmarks*, MLSys. [Paper page][R48]. Inspected variance/benchmarking argument. Motivates data/seed/tuning uncertainty; the project's sampling plan needs its own simulation check.
- **R49.** Carnegie Mellon/CMUdict contributors, pronunciation dictionary and license (current source). [License][R49]. Inspected BSD-style terms. Optional symbolic pronunciation resource; not acoustic labels.
- **R50.** Alexei Baevski, Henry Zhou, Abdelrahman Mohamed and Michael Auli / Meta (2020), *wav2vec 2.0 / wav2vec2-base*. [Model card][R50]. Inspected 16 kHz, audio-pretraining/tokenizer and license statements. No waveform inference or training performed.
- **R51.** Vassil Panayotov, Guoguo Chen, Daniel Povey and Sanjeev Khudanpur (2015), *LibriSpeech*, ICASSP, DOI 10.1109/ICASSP.2015.7178964. [Official assets][R51]. Inspected downloads/license. Transcripts do not supply gold phone boundaries; no audio downloaded.
- **R52.** Zero Resource Speech Challenge organizers, *ABX metrics* (current official documentation). [Metric][R52]. Inspected within/across-speaker discrimination definitions. Optional acoustic evaluation; benchmark access/labels remain a gate.
- **R53.** Ao Sun, Yuanyuan Yuan, Pingchuan Ma and Shuai Wang (3 February 2024), *Eliminating Information Leakage in Hard Concept Bottleneck Models with Supervised, Hierarchical Concept Learning*, arXiv:2402.05945v1. [Paper][R53]. Inspected design sections 2–3. Adjacent hard/hierarchical concept work; classification with a concept/class mapping, not this LM.
- **R54.** Yana Veitsman, Jonas Mayer Martins, Jonathan Lautenschlager and Lisa Beinborn (10 September 2026), *Structural priors for data-efficient language learning*, arXiv:2609.11505v1. [Paper][R54]. Inspected setup and results sections 4–5. Structural-transfer gains depend on metric and stronger language-data control; distinct multilingual initialization experiment.
- **R55.** Sanchit Sinha, Guangzhi Xiong and Aidong Zhang (2025), *COCO-Tree: COmpositional Hierarchical COncept Trees for Enhanced Reasoning in Vision Language Models*, EMNLP. [Paper][R55]. Inspected concept-tree construction. Adjacent externally generated concept reasoning, not a finite internal causal-language-model route; no score comparison used.

[R01]: https://aclanthology.org/P16-2038.pdf
[R02]: https://aclanthology.org/D17-1206.pdf
[R03]: https://aclanthology.org/D18-1548.pdf
[R04]: https://arxiv.org/abs/1705.08142
[R05]: https://aclanthology.org/N16-1024.pdf
[R06]: https://arxiv.org/abs/1702.00887
[R07]: https://aclanthology.org/2022.tacl-1.81.pdf
[R08]: https://aclanthology.org/2023.emnlp-main.195.pdf
[R09]: https://aclanthology.org/2024.acl-long.84.pdf
[R10]: https://aclanthology.org/2024.findings-acl.303.pdf
[R11]: https://arxiv.org/html/2605.15562v1
[R12]: https://aclanthology.org/2021.eacl-main.228.pdf
[R13]: https://aclanthology.org/2025.tacl-1.6/
[R13a]: https://aclanthology.org/2021.naacl-main.146.pdf
[R14]: https://proceedings.mlr.press/v306/zhou26at.html
[R15]: https://proceedings.mlr.press/v306/schulz26a.html
[R16]: https://aclanthology.org/2021.tacl-1.14.pdf
[R17]: https://aclanthology.org/2022.naacl-main.325.pdf
[R18]: https://proceedings.mlr.press/v119/koh20a.html
[R19]: https://papers.neurips.cc/paper_files/paper/2022/file/944ecf65a46feb578a43abfd5cddd960-Paper-Conference.pdf
[R20]: https://proceedings.iclr.cc/paper_files/paper/2025/file/de4ce91dfe56b919ee1c228d6a78f866-Paper-Conference.pdf
[R21]: https://arxiv.org/html/2608.07594v1
[R22]: https://www.isca-archive.org/interspeech_2025/khurana25_interspeech.pdf
[R23]: https://github.com/amir-zeldes/gum/tree/V12.1.0
[R24]: https://aclanthology.org/2023.law-1.17.pdf
[R25]: https://github.com/UniversalDependencies/UD_English-GUM/tree/r2.18
[R26]: https://raw.githubusercontent.com/amir-zeldes/gum/V12.1.0/LICENSE.md
[R27]: https://github.com/UniversalAnaphora/UniversalAnaphora/blob/main/documents/UA_CONLL_U_proposal_compact.md
[R28]: https://raw.githubusercontent.com/amir-zeldes/gum/V12.1.0/splits.md
[R29]: https://patents.google.com/patent/US11783164B2/en
[R30]: https://support.apple.com/en-us/117736
[R31]: https://huggingface.co/Qwen/Qwen3-0.6B-Base
[R32]: https://huggingface.co/Qwen/Qwen3.5-0.8B-Base
[R33]: https://docs.pytorch.org/docs/2.14/notes/mps.html
[R34]: https://ml-explore.github.io/mlx/build/html/install.html
[R35]: https://huggingface.co/blog/mps-pytorch-2-15
[R36]: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/qwen3_5.py
[R37]: https://huggingface.co/HuggingFaceTB/SmolLM2-135M
[R38]: https://huggingface.co/docs/bitsandbytes/v0.50.2/en/installation
[R39]: https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html
[R40]: https://aclanthology.org/2020.acl-main.158/
[R41]: https://github.com/alexwarstadt/blimp
[R42]: https://aclanthology.org/2023.tacl-1.96/
[R43]: https://docs.pytorch.org/docs/2.14/mps.html
[R44]: https://ml-explore.github.io/mlx/build/html/python/metal.html
[R45]: https://docs.pytorch.org/docs/2.14/mps_environment_variables.html
[R46]: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/tuner/trainer.py
[R47]: https://aclanthology.org/P18-1128/
[R48]: https://proceedings.mlsys.org/paper_files/paper/2021/hash/0184b0cd3cfb185989f858a1d9f5c1eb-Abstract.html
[R49]: https://github.com/cmusphinx/cmudict/blob/master/LICENSE
[R50]: https://huggingface.co/facebook/wav2vec2-base
[R51]: https://www.openslr.org/12
[R52]: https://zerospeech.com/tasks/task_1/metrics/
[R53]: https://arxiv.org/html/2402.05945v1
[R54]: https://arxiv.org/html/2609.11505v1
[R55]: https://aclanthology.org/2025.emnlp-main.135.pdf

## Appendix A. Checks actually performed during preparation

### A.1 Public corpus inspection and immutable file identifiers

The following public files were downloaded in the research environment and read for metadata/schema validation. URL bases:

- GUM: `https://raw.githubusercontent.com/UniversalDependencies/UD_English-GUM/r2.18/`
- GENTLE: `https://raw.githubusercontent.com/UniversalDependencies/UD_English-GENTLE/r2.18/`
- Split map: `https://raw.githubusercontent.com/amir-zeldes/gum/V12.1.0/splits.md`

Append the filename in the table to the corresponding base URL. Exact file SHA256 values:

| File | Bytes of CoNLL-U file | SHA256 |
|---|---:|---|
| `en_gum-ud-train.conllu` | 20,996,445 | `88c410ca6a51ccc79f9536d75feb1633421691165cd191f8137f8b37569a0329` |
| `en_gum-ud-dev.conllu` | 2,970,960 | `61d02ce27a143e9749118b1b9c314ac3da40f64ec17a6aaff5fc7b42704f7910` |
| `en_gum-ud-test.conllu` | 2,950,684 | `cd96a285e7339f401f4803dd0f4f61109c51692e3ef6f9186a1f977544080b69` |
| `en_gentle-ud-test.conllu` | 1,949,238 | `42d12cdef99bcd3160a3241ba1811fe9e7fd4b37a77f9b9e8c2852f04d69f3fa` |

File byte sizes above include annotations and are **not LM text byte denominators**.

The read-only audit counted sentence/word/MWT/empty rows, parsed all Entity openers/closers using document-local per-GRP stacks, reconstructed each sentence from surface tokens and `SpaceAfter`, and compared IDs with the pinned split map. Measured results:

| Audit output | GUM train | GUM dev | GUM test | GENTLE |
|---|---:|---:|---:|---:|
| Multiword token rows | 3,527 | 517 | 411 | 180 |
| Empty-node rows | 125 | 14 | 15 | 0 |
| Same-GRP nested openers | 234 | 59 | 39 | 18 |
| Retained innermost closing events | 47,752 | 6,671 | 6,530 | 4,896 |
| Retained closing events with `P=auto` | 396 | 54 | 57 | 11 |
| Unmatched/unclosed Entity events | 0 | 0 | 0 | 0 |
| Sentence text reconstruction mismatches | 0 | 0 | 0 | 0 |
| Nonliteral MWT constituent concatenations | 0 | 0 | 0 | 0 |
| Conflicting innermost-label ties | 0 | 0 | 0 | 0 |
| Detected discontinuity markers | 0 | 0 | 0 | 0 |

All documents had the same eight-field Entity header. GUM train/dev/test document sets exactly matched V12.1.0 after removing `GUM_reddit_*`; GENTLE exactly matched `test2`. The checks resolve release/schema uncertainty but **not tokenizer-specific byte/word alignment**, nor whether the annotations improve prediction. No candidate model scores were examined. Recreate these counters with the parser described in Section 5.3; any mismatch is investigated before modeling.

### A.2 Small algebra/forward checks

Environment: **Linux 6.18.55, x86_64 CPU, Python 3.12.14, NumPy 2.3.5**. There was no M3 GPU and no PyTorch training in this check.

An untrained eight-layer NumPy causal-attention toy with four hard predicted message injections was evaluated on a 12-token sequence. It is a simplified graph fixture, not the production architecture.

| Check | Measured result | What it establishes |
|---|---:|---|
| Same-length suffix changes at 11 cut positions | Maximum earlier-logit absolute difference `0.0` | The tested toy forward graph is causal under those fixtures |
| Consistent category-output/embedding permutation | Maximum logit absolute difference `0.0` | Category numbering is not a semantic negative control |
| Identical score arrays subtracted | Paired mean difference `0.0` | Basic identity arithmetic only; not interval coverage |
| Equally likely two continuations | Prefix-only NLL `0.6931471805599453` nats; future-label oracle `0` | An unavailable label can manufacture one bit of apparent gain |
| Qwen3 configuration parameter formula | `596,049,920` parameters | Arithmetic estimate, pending loaded-model count |

The memory numbers in Section 6 were calculated from that config. No optimizer behavior, learning gain, MPS kernel, training speed, checkpoint replay, corpus BPB or acoustic ability was measured. The production checks remain in WP1–WP9.

The following compact snippet reproduces the arithmetic portion without model downloads:

```python
import math
P = 151936 * 1024 + 28 * (
    1024 * (16 * 128 + 2 * 8 * 128)
    + (16 * 128) * 1024
    + 3 * 1024 * 3072
    + 2 * 1024 + 2 * 128
) + 1024
assert P == 596049920
gib = 2**30
print({
    "bf16_weights_GiB": 2 * P / gib,
    "fp32_full_AdamW_static_GiB": 16 * P / gib,
    "fp32_logits_B1_T512_GiB": 512 * 151936 * 4 / gib,
    "one_score_per_layer_GiB": 28 * 16 * 512**2 * 4 / gib,
    "bf16_KV_B1_T512_GiB": 2 * 28 * 512 * 8 * 128 * 2 / gib,
    "future_label_spurious_gain_bits": -math.log(0.5) / math.log(2),
})
```

### A.3 Reproducible simplified forward fixture

Copy this standalone CPU-only fixture if reproducing the preparation check. It intentionally omits normalization, real tokenizer alignment, optimization and the production cache. The full architecture is specified in Section 5.

```python
import math
import numpy as np

rng = np.random.default_rng(20261023)
T, D, V, C = 12, 16, 23, 5
taps = {1: 0, 3: 1, 5: 2, 6: 3}
weights = [tuple(rng.normal(0, .12, (D, D)) for _ in range(3))
           for _ in range(8)]
emb = rng.normal(0, .2, (V, D))
out = rng.normal(0, .2, (D, V))
heads = [rng.normal(0, .15, (D, C)) for _ in range(4)]
codes = [rng.normal(0, .1, (C, D)) for _ in range(4)]

def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

def forward(tokens, permutations=None):
    h = emb[np.asarray(tokens)].copy()
    messages = []
    n = len(tokens)
    for layer, (wq, wk, wv) in enumerate(weights):
        scores = (h @ wq) @ (h @ wk).T / math.sqrt(D)
        scores[np.triu_indices(n, 1)] = -np.inf
        h = h + softmax(scores) @ (h @ wv)
        if layer in taps:
            k = taps[layer]
            q, e = heads[k], codes[k]
            if permutations is not None:
                p = permutations[k]
                q, e = q[:, p], e[p, :]
            prior = sum(messages) if messages else np.zeros_like(h)
            ids = np.argmax((h + prior) @ q, axis=-1)
            message = e[ids]
            messages.append(message)
            h = h + message
    return h @ out

tokens = rng.integers(0, V, size=T)
errors = []
for cut in range(1, T):
    changed = tokens.copy()
    changed[cut:] = rng.integers(0, V, size=T-cut)
    errors.append(np.abs(forward(tokens)[:cut] -
                         forward(changed)[:cut]).max())
permutations = [rng.permutation(C) for _ in range(4)]
permutation_error = np.abs(forward(tokens) -
                          forward(tokens, permutations)).max()
assert max(errors) == 0.0
assert permutation_error == 0.0
print(float(max(errors)), float(permutation_error))
```

### A.4 Scoring-window and event-density checks

Additional CPU checks used every integer `T=2..32` and `N=1..127`: **3,937 parameter pairs and 251,968 owned target positions**. Each raw token had exactly one owner; every input fit its context limit; the consumed-prefix and relative-position assignment for every earlier target stayed identical when its document was extended by `T+7` tokens. These are index/arithmetic checks, not tokenizer or neural-cache tests.

For an event A with CE 2 carried over two steps with weights 1/2 and 1/2, and event B with CE 8 and weight 1 in a third step, the correct full-pass equal-event loss is **5**. Independent per-step normalization incorrectly gives **4**. The specified fixed normalization yields step losses 1.5, 1.5 and 12, whose average is **5**. Unequal LM target counts were checked by the same fixed-denominator identity. The embedded A.2/A.3 snippets were rerun successfully after the final revisions.

This compact standalone fixture reproduces the added index and density checks:

```python
from fractions import Fraction

def rows(N, T):
    S = T // 2
    assert N > 0 and T >= 2
    z = [-1] + list(range(N))  # -1 stands for START.
    owned = {}
    for a in range(0, N, S):
        b = min(a + S, N)
        c = max(0, a + S - T)
        inputs, targets = z[c:b], z[c+1:b+1]
        assert 0 < len(inputs) <= T
        for i in range(a, b):
            assert i not in owned and targets[i-c] == i
            assert inputs[i-c] == (-1 if i == 0 else i-1)
            owned[i] = (c, i-c, tuple(inputs[:i-c+1]))
    assert set(owned) == set(range(N))
    return owned

cases = positions = 0
for T in range(2, 33):
    for N in range(1, 128):
        original, extended = rows(N, T), rows(N+T+7, T)
        assert all(original[i] == extended[i] for i in original)
        cases += 1
        positions += len(original)
assert (cases, positions) == (3937, 251968)

masses = [Fraction(1, 2), Fraction(1, 2), Fraction(1)]
numerators = [Fraction(1), Fraction(1), Fraction(8)]
K, W = 3, sum(masses)
fixed = [K * numerator / W for numerator in numerators]
wrong = [numerator / mass for numerator, mass in zip(numerators, masses)]
assert sum(fixed) / K == 5
assert sum(wrong) / K == 4
print(cases, positions, [float(x) for x in fixed])
```

The final document also passed UTF-8 decoding, balanced-fence and reference-definition checks: 56 unique source IDs, no undefined numbered reference, and no chat-specific citation marker. This verifies artifact structure, not the truth of every future experimental assumption.

## Appendix B. Version 1 controlled-language contract

This specification resolves the central generator choices. It is a **proposed** experimental distribution, not a claim that its neural results are known. Store every rule in `configs/synthetic.yaml` and hash that file. Changes following development diagnostics require a new generator version and fresh protected families.

### B.1 World and interpretation

Use six noun kinds: `fox, hare` (animal), `pilot, chef` (person), `robot, cart` (object); colors `red, blue, green, yellow`; count `one/two`; predicates `follow, inspect, admire`. A plural entity is a **single group of two**, not two separately addressable individuals. All predicates are formal relations between groups; do not infer real-world affordances or introduce unstated commonsense truth.

A world contains four distinct `(kind, color, count)` descriptors, with all three semantic types represented. No two entities share a complete descriptor. IDs are private generator metadata. First introduce two entities with one bare, unmodified `NP .` fragment each, then generate four main assertions involving distinct agent/patient IDs. Each introduction adds that entity to mention history but adds no predicate fact. The remaining two entities may be introduced by their first assertion mention. Sample unused entity descriptors uniformly conditional on these constraints; log the sampling seed.

The fact ledger maps `(agent_id, predicate, patient_id)` to positive or negative. Do not assert both polarities of the same tuple, including facts contributed by relative clauses. Absence is **unknown**, not false. Positive assertions set true; explicit negative assertions set false. A relative clause contributes its own ledger fact; a `near` modifier supplies an unqueried relation. Reject/resample a conflicting proposal before rendering, with rejection counts logged.

At each mention's start, information status is `new` if that group has never appeared earlier in the text, otherwise `given`. E is its noun kind's semantic type. P is only this two-way generator status; it is not presented as equivalent to all six GUM information-status classes.

### B.2 Rendering grammar and annotations

Render new singular NPs as `a + color + singular noun`, new plural NPs as `two + color + plural noun`, and given NPs as `the + color + number-marked noun`. Plurals are `foxes, hares, pilots, chefs, robots, carts`. Capitalize only sentence beginnings using a recorded reversible token rule.

Main assertions are chosen uniformly among voice/polarity combinations:

```text
Active positive: NP_agent + agrees(predicate, agent_number) + NP_patient .
Active negative: NP_agent + does/do + not + base_predicate + NP_patient .
Passive positive: NP_patient + is/are + past_participle + by + NP_agent .
Passive negative: NP_patient + is/are + not + past_participle + by + NP_agent .
```

Inflections are `follows/inspects/admires`, base forms, and `followed/inspected/admired`. In active forms the subject is the agent; in passive forms the subject is the patient. Relative clauses are active `that + VP` with the modified NP as subject, generated under the same number rule.

To propose an NP below a chosen depth cap, choose no modifier with probability 0.5, `near + NP` with probability 0.25, and `that + VP` with probability 0.25. At the cap, choose no modifier with probability 1. Count a recursive NP expansion as one additional level; an unmodified NP has depth 0. Forbid an entity from recursively modifying itself along a path. The sampled factor `depth` is the **realized maximum NP recursion depth in the example**, not merely the cap: condition generation on reaching that value and record rejection rates. Train/dev composition depths are 0, 1 and 2; the separately named extrapolation endpoint requires realized depth **exactly 3**. These values define the first controlled distribution, not a ceiling on later scientifically justified tests.

The derivation carries head indices and UD-like relations by construction: lexical predicate is clause head; active agent `nsubj` and patient `obj`; passive patient `nsubj:pass` and agent `obl:agent`; auxiliary `aux/aux:pass`, determiner `det`, numeral `nummod`, color `amod`, negation `advmod`, preposition `case`, NP modifier `nmod`, relative predicate `acl:relcl`. The relative marker `that` is a `PRON` with `nsubj` pointing to its relative predicate; its coindexation with the modified noun is private derivation metadata. In an introduction fragment, the noun is the root. Attach punctuation to the clause/fragment root.

Assign `NOUN` to entity nouns, `ADJ` to colors, `DET` to `a/the`, `NUM` to `two`, `VERB` to lexical predicates, `AUX` to `does/do/is/are`, `PART` to `not`, `ADP` to `by/near` and `PUNCT` to punctuation. Nouns and relative pronouns receive their entity's Number; finite lexical verbs and auxiliaries receive subject Number, Person=3, Tense=Pres and Mood=Ind. Active lexical forms receive Voice=Act and passive participles Voice=Pass; uninflected verbs after `do/does` have no finite Number/Tense/Mood target. Selected features without a value use `ABSENT_IN_ANNOTATION`. The exact labeled trees and features are derivation outputs, not an external parser. Do not claim this tiny annotation grammar is a full implementation of UD.

Render mention spans from NP derivations, preserve nesting, and apply the same shortest-closing-mention projection as the real experiment. The generated field inventory can be smaller; report it. Retain private logical roles/polarity only for the interpreter and diagnostic outcomes; the initial E message still contains entity type only.

### B.3 Queries and scoring

Choose query family uniformly before selecting an answerable instance:

| Query family | Fixed form / answer rule |
|---|---|
| Agent role | `In the last main assertion, which group is the agent?` → canonical descriptor of the last main assertion's agent, irrespective of voice/polarity |
| Patient role | Same form with `patient` → its patient descriptor |
| Truth | `Is it true that <canonical agent> <base predicate> <canonical patient>?` → `true/false/unknown` from the full ledger |
| Discourse status | `Was the last mentioned group already mentioned before that mention?` → `yes/no` computed from history before that mention |

Sample the desired truth answer class uniformly **before** constructing an eligible world/query, then balance accepted base examples across true/false/unknown. Do not require every world or counterfactual variant to admit all three classes. Recompute each variant's actual answer and report achieved class counts after family acceptance; counterfactual diagnostic members remain paired secondary examples and need not share the base example's answer. Balance discourse yes/no by selecting the last mention/world accordingly. “Last mentioned” always refers to the context **before the query**. Role answers use canonical `one/two + color + singular noun` strings, yielding one unambiguous answer. Serialize the answer after a fixed `Answer:` delimiter; train **all methods** on the same complete base-example LM objective, and report answer-span exact match/denotation separately. Whether diagnostic variants also enter training must be fixed identically for every method before the generator lock; the default reserves them for diagnostic scoring.

Questions, the `Answer:` delimiter and answer strings add neither facts nor mention-history events. For version 1, mask **all auxiliary fields** at positions whose consumed input has entered this scaffold, including a token overlapping its start; do not carry the context's last annotated state as a supposedly annotated query/answer target. Heads and messages still run densely. Apply these masks identically to all supervised conditions, retain every LM target, and report scaffold versus context coverage. Named-state correctness interventions use only context positions with defined derivation targets. A later fully annotated query grammar requires a versioned amendment.

Discourse status can be trivial from NP markers in the basic distribution. Therefore include a prespecified counterfactual diagnostic that replaces **every NP-initial marker in `{a, the, two}`**, regardless of UPOS, with the fixed token `the` before capitalization, in **every** method's train/dev/test variant. Give that marker the same `DET/det` annotation with absent morphological fields in every case; preserve number only in noun morphology. Assert token-identical final mention spans across the paired earlier-history contrast. Replacing only POS-tagged determiners would leave the numeral `two` as a shortcut and is prohibited. A basic determiner-driven success alone is not evidence of discourse memory.

Score agreement with paired grammatical/ungrammatical continuations of the same prefix, using summed continuation log probability under the same tokenizer. Score role/truth/reference answers by constrained enumeration of the finite answer vocabulary and by free greedy answer generation separately. The constrained scorer is shared across methods and is not a privileged parser for the typed model.

### B.4 Partition and sampling policy

Hold out these **unordered noun-kind pair + predicate** combinations, in either agent direction and every voice/polarity:

- Composition development: `{hare, robot}+follow` and `{chef, fox}+inspect`.
- Composition protected test: `{fox, chef}+follow` and `{robot, hare}+admire`.
- Training: other combinations; no asserted **or queried** predicate tuple may instantiate either held-out set.

Check split eligibility against **every asserted and queried tuple**, including relative-clause facts, unknown-answer truth questions and every counterfactual variant. Protected-test relations are excluded from both training and development text; development holds are excluded from training. For a composition endpoint, designate the tested relation: it must be the last main assertion for role questions, or the actual queried tuple for truth questions, and must belong to that endpoint's holdout. A held-out relation appearing only in an unrelated context clause does not qualify. Include matched training-seen relation controls. Discourse, IID and depth outcomes are separately labeled endpoints.

Held-out pairs may co-occur as separate mentions or in unqueried `near` modifiers; the held-out unit is the specified predicate relation, not word occurrence. Verify that every kind, predicate, voice, polarity, count and training depth occurs in training and that each held-out constituent occurs in other relations. A family that would mix development and protected relation holds is not generated. The realized-depth-3 extrapolation set uses otherwise training-eligible asserted **and queried** relations so depth and relation novelty are not conflated.

Group all counterfactual versions of one abstract world/derivation skeleton together, including role swap, number change, polarity change and voice change. Designate the assertion and entity to be changed in private metadata before transforming it. A number change applies consistently to **every mention** of that entity. Role, polarity and voice operations act on the designated main assertion; the queried tuple remains fixed unless the specified diagnostic explicitly changes the query. Unknown-truth controls must remain unasserted in variants labeled unknown. Do not interpret an inapplicable operation as a demonstrated intervention effect.

Construct the complete family closure under the declared operations before accepting the skeleton. Recompute each variant's ledger, referring expressions, mention histories/statuses, annotations, answers, achieved depth and split eligibility. Reject/resample the **whole family** if any required variant produces duplicate descriptors, contradictory facts, inaccessible referents, malformed derivations or split violations. For clean role/polarity interventions, prohibit another fixed fact from constraining the designated tuple or its reversal. Record rejection causes and rates; a generator that cannot sample its promised cells is a WP2 failure, not permission to silently omit them.

Family identity has two levels. A member-family key serializes the complete symbolic variant set, sorts it, minimizes its serialization over all 24 renamings of the four private entity IDs, then applies SHA256; it includes query definitions and modifiers but excludes seeds/counters. For the **partition and statistical unit**, also hash every canonical context variant with query, answer, probe-designation metadata and unused entity records removed. Build connected components of families sharing any canonical context or rendered context. Thus agent/patient/truth questions derived from the same context, and different counterfactual closures with an overlapping context, remain together. The component ID hashes its sorted context-key set. Private-ID renaming and the chosen counterfactual starting member must not alter these identities.

Construct these components across the finite candidate pool **before** assigning partitions. Assign IID components with published hash boundaries and salt; choose planned partition sizes from development learning curves and the Section 7 precision analysis, then freeze them. Explicit composition holds take precedence; discard a component whose members impose incompatible split constraints. Reject cross-partition duplicate contexts, canonical families and rendered examples. When expanding a development pool, recompute overlap components; after the lock, a fresh pool must also be checked against all existing contexts and cannot move an exposed component into protected evaluation. Disjoint seeds alone do not prevent finite-grammar duplication. Store exact component/family/member IDs, partitions, hashes and seeds.

Start development generation with one balanced traversal of all permitted main-clause factor cells `(agent kind, patient kind, predicate, voice, polarity, subject number, object number, depth)`, sampling color/context under the constraints above. Expand examples per cell and training passes using baseline learning curves and held-out uncertainty; do not pick a fixed corpus size merely because it finishes quickly. Lock the resulting finite dataset/generator policy before confirmation. Sample size for protected families follows the Section 7 precision analysis.

For lexicon robustness, replace surface noun/predicate stems with a bijection in a separately trained replicated world; preserve singular/plural/verb forms consistently and keep that mapping across all of that world's partitions. Do not expose the original semantic IDs as input tokens or metadata.

For the exact tiny case, enumerate two kinds × two colors × two numbers in `the COLOR NOUN agrees(run, number) .` with no modifiers and a uniform distribution over the eight strings. Verify total probability 1 and expected sequence NLL \(\ln 8\) under the exact grammar. This validates arithmetic and Bayes headroom; it does not establish the full world's learnability.
