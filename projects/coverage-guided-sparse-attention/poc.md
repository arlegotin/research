# Recamán-inspired neural computation: research assessment and executable local PoC plan

**Research date:** 17 October 2026. **Target machine:** a MacBook Pro with an M3 Max; its actual memory, storage, macOS version, and thermal conditions must be discovered. **Status:** researched experimental specification, not a report of a trained model or an M3 benchmark. All proposed project commands below must be implemented unless explicitly identified as existing system/package commands.

## 1. Executive assessment and selected direction

The central idea—allocate a sparse neural computation budget using relevance, previous selections, and a preference for useful new information—is scientifically reasonable but substantially anticipated. In particular, K-VEC already combines coverage across layers/heads with relevance protection, and the October 2026 VASC preprint combines block selection with an execution-dependent memory across layers. A different name, the literal Recamán sequence, or the addition of decay would not establish a new architectural principle. [R19, R20]

**Recommended primary experiment:** add a small, reversible, query-local coverage selector to an existing pretrained sparse-attention router, then test whether it improves the quality–read-budget tradeoff over the best locally executable relevance, coverage, diversity, and reuse controls. Use the published SeerAttention-R gate for **Qwen/Qwen3-4B**, port its mathematical inference path to MLX, and retain the entire KV cache. Use Qwen3-0.6B only for early integration checks or a clearly labelled reduced-scope fallback. The published 4B gate avoids making a weak hand-built selector the main competitor and avoids large-scale training. [R24, R40–R43]

The proposed selector changes the support of attention during single-query decoding. It resets coverage at each new input/generated token, reads previous-layer coverage, protects leading relevance candidates, and limits the loss of a cheap relevance score. It does not alter pretrained weights, evict KV entries, or claim that position coverage measures information gained. Its exact definition appears in Section 4.

The intended contribution is initially **an empirical finding and a portable implementation**, with a possible narrow methodological contribution only if the particular state/reset/constraint rule survives the comparison with K-VEC, VASC, and equivalent constrained-diversification rules. No methodological novelty is established by this review. If the best existing rule wins, reproduce and explain that result rather than relabelling it.

Retain one inexpensive **Recamán-specific graph diagnostic** before large downloads. Compare a precisely defined sequence-to-offset mapping against random, dilated, and layer-coordinated alternatives, including identical boundary repairs. This addresses the original mathematical intuition. It is a separate experiment: dense prefill makes the causal ancestor graph of the primary decoder fundamentally different, so its results cannot be used as evidence for the decode selector.

### 1.1 Why this direction

| Direction | Assessment | Decision |
|---|---|---|
| Literal Recamán weights or offsets | No inspected source establishes a neural efficiency advantage; repeats and finite-prefix scale mismatch require explicit handling. | Cheap structural diagnostic; no special status in the primary architecture. |
| Coverage-aware sparse decoding | Preserves the history-dependent allocation idea; pretrained routers and a frozen small model make a controlled local intervention possible. Close prior art sharply narrows novelty. | **Primary.** |
| Training a new sparse transformer | Can test learnability and jointly adapted representations, but would entangle topology, routing, optimization, and kernel maturity before the indispensable allocation mechanism is established. | Evidence-triggered extension, not the core dependency. |
| Active external-memory traversal | Valuable when later queries depend on earlier reads, but requires a semantic index and a separate controller/task definition. | Conditional pivot after query-dependent retrieval evidence, not a synonym for sparse attention. |
| Structured channel mixing | Butterfly and block-structured alternatives already address hardware utilization. No particular channel-mixing bottleneck has been supplied. | Defer until profiling identifies a relevant workload. |
| Low-discrepancy kernel features | Performer and DiJiang already cover substantial territory, including quasi-Monte Carlo conversion. | A different approximation research question, not a novelty escape route. |

The smallest complete primary result is a controlled comparison on one frozen backbone, one declared context distribution, and one matched read budget, including the strongest licensed local learned router and close coverage baselines. A convincing positive result need not be large. It must have uncertainty small enough to distinguish it from noise, survive attribution controls, and disclose any cost or guardrail regression.

### 1.2 Claims the local PoC can and cannot establish

It can establish that a specific selection rule improves a specified frozen decoder at a measured budget on specified synthetic and/or natural-language workloads; that its actual Metal execution is faster or slower; and whether its benefit is explained by ordinary diversification, relevance protection, or reuse.

It cannot establish that Recamán's sequence is generally superior, that coverage equals semantic novelty, that training will become cheaper, that a full sparse-prefill model will benefit, that KV capacity decreases, or that larger models on other accelerators will improve. A negative frozen-model intervention also cannot by itself reject an architecture trained to use coverage from the outset.

## 2. Problem, assumptions, hypotheses, and headroom

### 2.1 User, workload, and bottleneck

The initial beneficiary is a researcher or developer performing local, long-context inference under a fixed attention-read budget. No product use case has been specified. The immediate workload is answering questions whose evidence is dispersed through a supplied context, with controls requiring repeated use of the same evidence region.

At each layer a dense decoder reads all past K/V vectors. A sparse decoder first chooses a subset and then evaluates attention only on that subset. The proposed limitation is **redundant selection across depth under a tight budget**, not merely the existence of a long sequence. This is a hypothesis: different layers can extract different functions from the same position, and repeated selection can be beneficial. IndexCache and related reuse methods are a concrete counterweight to an unconditional exploration preference. [R9, R10]

For batch size one let:

- \(N\): cached tokens visible to the current query, including its newly appended KV;
- \(b\): tokens per KV block; \(M=\lceil N/b\rceil\);
- \(L\): layers; \(H_q\): query heads; \(H_{kv}\): KV heads; \(d_h\): head dimension;
- \(S_{\ell,g,t}\): selected block IDs for layer \(\ell\), KV group \(g\), query position \(t\);
- \(B\): an attended-token budget per group/layer, including mandatory blocks and partial-block accounting;
- \(c_{\ell,t,j}\): previous-layer selection coverage for absolute block \(j\), reset at each \(t\).

The primary path uses GQA-aligned selections because that is the published SeerAttention-R interface. A separate per-query-head Quest port preserves Quest's finer selection granularity. Do not silently coarsen that comparator to make the candidate appear competitive. [R23, R24]

### 2.2 Established facts, assumptions, and proposed hypotheses

| ID | Statement and predicted observation | Strong alternative explanation | Evidence that weakens it |
|---|---|---|---|
| H-GRAPH | A declared Recamán-derived offset family offers a better causal, layer-ordered structural tradeoff at equal executed edges. | Longer average jumps, boundary fill, or more tuning produced the apparent benefit. | The advantage disappears after matching those factors or fails at held-out lengths/phases. This rejects the tested family, not every conceivable mapping. |
| H-COVER | Given the same backbone, scorer, and executed reads, bounded query-local coverage improves all-required-facts accuracy on dispersed-evidence tasks. | A protected top-relevance set, generic MMR, random near-tie perturbation, or extra resources explains the gain. | Close controls match it; repeat-evidence tasks regress; score changes do not translate to answers. |
| H-STATE | The useful part is information in the actual previous-layer selection history. | Any deterministic reshuffling regularizes a brittle selector. | Zero, permuted, lagged-wrong-query, or shuffled-layer history performs equally well. |
| H-REVISIT | A relevance constraint and short memory preserve important revisits better than unconditional coverage avoidance. | The constraint simply disables nearly all exploration. | Constraint-active runs improve only by returning to ordinary top-k, or hurt multi-attribute evidence in one block. |
| H-TRANSFER | A synthetic improvement persists on an independently frozen natural-language population. | The generator was built to reward spreading selections, or the model exploits template cues. | Natural evaluation is negative, insensitive, or contradictory; the claim remains synthetic/component-specific. |
| H-TIME | The selected implementation improves measured latency on an identified workload after all routing and memory costs. | Faster incomplete timing, omitted prefill, CPU fallback, or a weakened dense kernel. | Total latency does not improve under shared optimized execution and quality constraints. |

Established building blocks include causal attention, learned block routing, relevance/diversity selection, coverage tracking, and reuse. Assumptions requiring measurement include a nontrivial sparse-vs-dense quality gap, affordable router overhead, a predictive connection between block history and redundant computation, adequate base-model competence, and local operator compatibility. None follows from Recamán's irregular trajectory.

### 2.3 Mathematical limits and accounting

**Sequence definition.** Implement \(a_0=0\); at step \(n>0\), subtract \(n\) if the result is nonnegative and unvisited, otherwise add \(n\). The fallback can repeat: \(a_{20}=a_{24}=42\). Deduplication, modulo mapping, rescaling, and boundary filling each change the proposed attention construction and must be separately named. [R1]

**Reachability is only a necessary structural opportunity.** For a sparse-prefill graph, define ancestor sets by

\[
A_0(i)=\{i\},\quad
A_\ell(i)=A_{\ell-1}(i)\cup\bigcup_{h}\bigcup_{j\in S_{\ell,h}(i)}A_{\ell-1}(j).
\]

Attention \(i\to j\) means an information edge from \((\ell-1,j)\) to \((\ell,i)\). Computing paths in a union of all layers permits paths in the wrong temporal order. Bidirectional expander guarantees do not automatically apply after orienting/clipping edges to a causal prefix. Dense-prefilled cached representations may already encode distant information, so even correct ancestor counts do not measure evidence selected during decoding. [R5, R12]

**Coverage is not information gain.** Cached vectors at the same position differ between layers. Similar-looking blocks can contain distinct answer-critical attributes. Conversely, many distinct blocks can repeat one fact. A visit count cannot resolve either case. No mutual-information or accuracy guarantee is inferred from the number of unique positions selected.

**No content-free search miracle.** If one relevant item is uniformly hidden among \(M\) exchangeable blocks and the selector has no signal about it, inspecting \(k\) distinct blocks finds it with probability \(k/M\). A deterministic sequence can alter position bias but cannot beat this expectation without useful information or assumptions about the distribution.

**Output error depends on values.** If exact attention drops mass \(m\), its renormalized selected output is \(y_S\), and all values have norm at most \(V_{\max}\), then

\[
y=(1-m)y_S+my_{\bar S},\quad
\|y-y_S\|\le 2mV_{\max}.
\]

This elementary bound is a diagnostic, not a certificate from proxy block scores. Constant values yield the same output under many different masks; small mass can still matter for a decision boundary. Approximation error, attention mass, and answer correctness must be reported separately. Value-aware approximation and verified sparse attention supply relevant alternatives. [R27, R52]

**Cost is stage-specific.** Dense prefill is quadratic in attention pairs; dense single-token decode is linear in \(N\). A block scorer scanning all blocks remains \(O(Md_r)\) per group/query even if the selected attention stage costs \(O(Bd_h)\). Token-to-block scoring during prefill is \(O(N^2d_r/b)\); block-to-block scoring is \(O(N^2d_r/b^2)\), still quadratic for fixed block size. Never call the whole pipeline constant or linear in context merely because its selected-attention term is fixed.

For the original illustration \(N=65,536,b=128,k=8\) including a causal current block, exact pair counting under the stated boundary convention gives

\[
P_{dense}=N(N+1)/2=2,147,516,416,
\]
\[
P_{sparse}=512\frac{128\cdot129}{2}+128^2\sum_{u=0}^{511}\min(7,u)
=62,488,576.
\]

The ratio is about 34.37; the simpler \(Nkb\) estimate is 67,108,864, giving about 32. These are different counting conventions, both unrelated to a measured model speedup.

**Amdahl limit.** If the stage replaced by sparse execution occupies measured fraction \(f\) of end-to-end time, its effective speedup is \(r\), and extra overhead is \(o\) times original total time, then

\[
\text{speedup}=\frac{1}{1-f+f/r+o}.
\]

For illustration, \(f=0.25,r=4,o=0.05\) gives only 1.16×. For \(f=0.60,r=4,o=0.05\), it gives 1.67×. These are assumptions, not measurements. Dense prefill, MLPs, projections, LM head, indexing, gathers, and tokenization all restrict a decode-only intervention. Short-output question answering may show a component benefit with negligible total latency gain.

### 2.4 Checks actually performed while preparing this plan

Small checks ran in **Linux x86_64, Python 3.12.14, NumPy 2.3.5**, not on Apple hardware:

1. Recomputed the sequence and verified its first repeated value: 42 at indices 20 and 24.
2. Recomputed the exact pair counts above.
3. Constructed a three-position, two-layer counterexample: layer 1 lets query 2 read position 1, while layer 2 lets query 1 read position 0. A union graph suggests a 0→1→2 route, but the final layer's position 2 has ancestors only {1,2}.
4. Checked 500 small randomized selector cases for exact cardinality, uniqueness, protection of the highest score, the score-regret constraint, and equality to top-k for zero penalty or zero coverage.
5. After the audit corrected local-index/absolute-ID ambiguity, extracted the final executable NumPy selector from Section 4.4 and ran 1,000 randomized cases, including gapped IDs and ties. Cardinality, protected IDs, regret, zero-penalty and zero-history controls passed. A final rerun also checked 400 anchor/regret-switch cases. The fractional-state bound passed 400 updates across gamma 0, 0.5, 0.9 and 1.
6. Traced the inspected compressed-cache counter ordering with two layers and four-slot blocks in NumPy. With one prefilled slot, the later layer's intended `[11,12,13,14]` became `[14,0,12,13]`, changing its mean from 12.5 to 9.75. An aligned prefix produced a harmless permutation, explaining why aligned-only tests miss the problem. This emulated control flow; it did not execute upstream PyTorch or measure model quality.
7. Reproduced SketchWalk's decode-threshold shape issue in a NumPy fixture: the released assignment changes a `[1,1,1]` threshold to zero and selects all 16 positive-scored blocks. Retaining the quantile threshold and explicitly adding endpoint support selects 4 of 16 on that fixture. This isolates the bypass; the required exact-total-budget adapter still needs its separate implementation/count tests. No upstream model execution is implied.

These checks validate arithmetic, reference invariants and a small cache-state counterexample. They are not efficacy experiments, graph-family comparisons, performance results, or proof of M3 compatibility. All model, full gate-port, dataset and Metal validation remains to be executed.

## 3. Research findings and novelty assessment

### 3.1 Closest prior-art comparison

The matrix distinguishes published mechanisms from proposed local adaptations. Reported empirical findings remain the authors' results; no published benchmark was reproduced during preparation.

| Prior work | Problem and mechanism | Evidence, assumptions, and costs | Reuse and exact overlap |
|---|---|---|---|
| Sparse Transformer; Longformer; BigBird [R2–R4] | Factorized, local/dilated, and local/random/global sparse patterns. | Different pretraining/tasks and masks; sparse connectivity alone does not guarantee useful retrieval or Metal speed. | Reuse pattern definitions and boundary controls. Generic mixed local/long-range connections and coordinated heads/layers are established. |
| Exphormer [R5] | Expander-based sparse attention for graph transformers. | Results/theory concern graph settings; causal clipping changes properties. | Structural comparator; not a faithful causal LLM baseline after adaptation. |
| SPAttention and boundary-repair analysis [R11, R12] | Complementary distance ranges across heads; explicit layerwise causal-reachability diagnostics. | Coverage claims depend on pattern and layer order; complete head union can still cost quadratic pairs. | Defeats novelty of generic complementary heads and motivates phase/boundary tests. |
| MoBA, NSA, DSA [R6–R8] | Content-routed block/token attention; NSA also combines compressed, selected, and local paths. | Native training and accelerator-specific implementations matter; indexers and extra branches cost computation. | Use mean-block scoring as a mathematical control. Full native models are nonlocal reference results here. |
| Quest [R23] | Query-dependent page selection from coordinatewise key extrema. | No learned gate; page size, retained dense layers, and head granularity matter. Original implementation is CUDA-oriented. | Required mathematical local port; preserve native page-16/dense-early-layer alternatives as well as a common page-64 comparison. |
| SeerAttention-R [R24] | Learned block router distilled from attention with GQA-compatible sparse decode. | Published 4B adapters exist; inference needs the exact base model, key preprocessing, positions, and gate math. | Required strong learned local baseline and shared scorer for the proposed selector. |
| SAS [R25] | Same gate family trained through downstream LM loss rather than solely attention matching. | Released 4B gate; training and reference inference use a different hardware stack. Inspected release did not establish usable code/weight licensing. | Conditional direct baseline if its license is verified; otherwise named reference gap. Do not replace it with invented 0.6B published weights. |
| K-VEC [R19] | Coverage-aware KV eviction with relevance protection. | Prefill-time retention using observation-window attention; published large-model/A100 setting. Eviction changes storage and future access. | Closest coverage precedent. Compare its selection rule inside the reversible harness, clearly labelled an adaptation. |
| VASC [R20] | Value-sensitive block demand plus unserved-demand memory across layers. | October 2026 geometry-model preprint; spatial block pairing and special-token treatment matter. | Closest execution-history precedent. Include a defined deficit-controller adaptation; 3D results are not LLM evidence. |
| Vortex [R56] | Programmable sparse decode includes proposals penalizing an EMA of past relevance scores. | Score history crosses tokens; the final search winners converge to ordinary block top-k plus system tuning. CUDA serving implementation; the proposed penalty is not established as a winner. | Required cheap score-EMA adaptation separates historical relevance from actual executed service. Full serving framework is reference-only. |
| Self-Indexing Attention [R59] | Shared rotations, sign matches and key norms provide inexpensive token relevance scores. | Later Qwen models and hardware-specific packed kernels; no independently verified original-Qwen3-4B release. An independently authored mathematical decode adapter is locally plausible. | Required current-position token comparator with extra index memory counted. It tests whether a stronger relevance approximation removes apparent coverage headroom. |
| Sketch-and-Walk / SketchWalk [R60] | Sketched relevance and walk-based block ranking; released decode retains per-layer state across tokens. | Published paper and released decode recurrences differ. Accessible MIT source has a threshold bypass requiring declared correction; native GPU timing does not transfer. | Required corrected-source exact-budget comparator, with dense early layers, window/matrix costs and source conventions retained. A sparse walk across layers is already prior art. |
| SPIN; DoSA [R57, R58] | Predict which native-indexer inputs to score, or preserve its top-k using bounds/selection optimization. | Native DSA-family models and NVIDIA serving; SPIN observes only scored slots, while DoSA's exactness depends on its scorer arithmetic. | Nonlocal references. Become required if the project claims cross-token predictive pruning or exact indexer acceleration; neither is the current final-support intervention. |
| MMR, coverage NMT/summarization, visual diversification [R16–R18, R21, R22] | Balance relevance, previously covered content, and diversity; some preserve a relevance anchor. | Domains and objectives differ; semantic diversity costs more than identity coverage. | Required relevance/diversity controls. A visit penalty is a member of this family, not a new principle. |
| IndexCache; CLSA [R9, R10] | Amortize selection by reusing indices across layers. | Depend on architecture, layer similarity, calibration, and sometimes shared KV. | Reuse is an opposing hypothesis; a calibrated refresh schedule is a local control, not full CLSA on Qwen. |
| Uncertainty-gated selection; vAttention [R28, R27] | Spend budget on ambiguous cutoffs; combine approximation with error-aware allocation. | Additional scoring/read work must be counted; fixed- and variable-budget claims differ. | Test random vs uncertainty-selected extra reads at matched mean cost; no novelty claim for conditional fallback itself. |
| RetroAttention [R26] | Update previous attention outputs when later reads expose previously missed KV. | Adds an output cache and changes past/deeper KV state; long-generation focus. | Reversible access and history are known. A substantive long-output claim needs this comparator or a disclosed comparison gap. |
| Performer; DiJiang; Pixelated Butterfly [R13–R15] | Kernel features, quasi-Monte Carlo conversion, and structured sparse/low-rank mixing. | Different approximation or linear-layer targets; precision, training, and hardware dominate comparisons. | Adjacent alternatives; no basis for presenting low-discrepancy sampling or butterfly replacement as unexplored. |
| DNC; RAT+ [R30, R31] | Learned external memory and recurrent content augmentation. | Require different state/learning mechanisms; RAT+ content memory is not a visit ledger. | Evidence-triggered architecture pivots, not lightweight frozen-decoder controls. |

### 3.2 Six explicit novelty questions

**What already exists?** Every broad ingredient in the idea: sparse communication, structured long jumps, relevance-aware selection, diversification, coverage memory, protected anchors, cross-layer coordination, index reuse, and learned external memory. The close overlap with K-VEC and VASC is material. Vortex supplies score-history anti-redundancy proposals, and Sketch-and-Walk already composes sketched block interactions through walks; neither a generic history penalty nor a sparse walk is a new principle. [R19, R20, R56, R60]

**What can be reused?** A public frozen Qwen3 model; SeerAttention-R's published router, MIT-licensed implementation, and configuration; MLX-LM's Qwen3 normalization, RoPE, and cache code; Quest's scoring equation; the accessible SketchWalk Qwen3 implementation with the declared correction; established dataset scoring; and ordinary paired-evaluation tools. Port only the inference components needed by the experiment. Do not rebuild tokenization or pretrained gate training.

**What might differ?** The proposed conjunction is a query-reset, reversible selection intervention with finite-memory cross-layer coverage and a hard **proxy-score regret** limit. These distinctions define a testable configuration, not established novelty. Reversibility already exists in non-evicting sparse attention; finite-memory history and constrained diversification are established ideas.

**Why might it have value?** Under a tight budget, near-equivalent relevance candidates may include different necessary evidence. Short history can reallocate some reads while allowing previously useful blocks back into later queries. Retaining the full cache preserves that option, at the cost of retaining its storage.

**What isolates the difference?** Use identical model/scorer/block size/kernel/information, compare the candidate against top-k, K-VEC-style multiplicative coverage, a VASC-style deficit state, score-EMA history, embedding-similarity MMR, generic constrained random selection, and calibrated reuse; remove decay and the score constraint separately. Stronger native-style scorers, including SIA and corrected-source SketchWalk, test competitiveness but do not replace these matched-scorer attribution controls. Measure answer quality and total costs, not only coverage.

**What invalidates novelty?** A prior work with equivalent state, admissible set, and ranking under a parameter transformation; equal outputs from an existing selector on all relevant inputs; or gains explained entirely by shared generic optimizations. Then classify the work as replication/adaptation. An absence of a source in this search is not evidence that no such prior art exists anywhere.

### 3.3 Reconciling apparently conflicting findings

Coverage and index reuse can both help: they optimize different budgets and operate on different models and objectives. Eviction reduces future access and storage; reversible selection preserves both. A learned router trained on final task loss can outperform one trained to imitate attention even when it matches less dense attention mass. A synthetic retrieval win need not transfer to natural multi-hop or aggregation tasks; the Sparse Frontier study motivates reporting task families separately. [R9, R19, R25, R29]

Treat K-VEC's information-theoretic motivation cautiously: positional union coverage alone does not establish useful mutual information in contextualized representations. Treat VASC's very recent results as relevant prior art and a source of a comparator, not a settled demonstration on language models. Treat published CUDA speedups as implementation-specific evidence, not portable ratios.

Score memory, execution memory and content memory observe different quantities. Vortex's candidate score penalties do not establish that anti-redundancy outperforms top-k; SPIN's partially observed score prediction is a different problem from a fresh all-block proxy scorer. DoSA preserves its native selection target, whereas coverage deliberately changes support. SketchWalk's accessible decode recurrence and threshold behavior must be read from the pinned implementation rather than inferred from its paper or another paper's earlier code-availability statement. These differences determine both the local comparator set and the limits of a positive finding. [R56–R60]

A scoped patent search identified existing sparse-attention disclosures, including US11238332B2 and dynamic head-pattern selection applications. Their existence reinforces the crowded prior-art landscape; this plan draws no conclusion about infringement, claim scope, validity, patentability, or freedom to operate. [R53]

## 4. Selected mechanism and implementation specification

### 4.1 Fixed backbone and intervention boundary

Use **Qwen/Qwen3-4B**, the original checkpoint expected by the selected Seer gate, not a later similarly named 4B release. Read and pin the base/gate revisions independently and verify they agree. The inspected base configuration has 36 layers, 32 query heads, 8 KV heads, and head dimension 128. The head dimension must come from the configuration; hidden-size divided by query-head count is not a valid substitute here. [R41, R42]

Two explicit evaluation boundaries prevent the first answer from bypassing the intervention:

1. **Primary query-suffix condition:** dense-prefill the complete supplied memory/context and fixed task instructions, ending before the question. Then process every token of the question and assistant answer prefix one at a time through the selected sparse decoder, followed by answer generation. Every method sees identical text and boundaries. This is sparse processing of a known suffix as well as generated tokens; report that fact.
2. **Standard-decode companion:** dense-prefill all prompt tokens except the final one. Route that final token sparsely to compute the first answer distribution, then continue generation. Never use dense final-prompt logits to generate a one-token answer attributed to sparse attention.

Store `dense_prefix_token_count`, `routed_prompt_token_count`, and `generated_token_count` per example. For latency, also measure the best ordinary fully batched dense prompt path. Equal split boundaries are a causal control; they must not be used to handicap the optimized dense production reference.

Serialize the complete chat prompt **once**, tokenize it once, then select a split index in that token array. Do not tokenize prefix and suffix separately and concatenate: tokenizer boundary merges and chat-template tokens can change the input. If a token spans the text boundary, assign it to the routed suffix and record the resulting index. Verify full token-ID equality across methods and parity of dense split ingestion against dense ingestion of that same array. Prompt roles, assistant prefix and end-of-turn tokens are part of the immutable template.

Each method builds or restores its own prefix-equivalent cache and then uses its own updated Q/K/V and generated history. Dense-teacher traces may be shared only in the explicitly labelled replay diagnostic. Do not replay dense hidden states while claiming an autonomous sparse-model result.

### 4.2 Common router and attention interface

Use absolute zero-based positions. At layer \(\ell\), append current K/V once, with the original model's Q/K normalization and RoPE. Blocks are contiguous absolute positions \([jb,\min((j+1)b,N))\). Block identity is stable across layers. No modulo wrapping or sequence-position reset after selection.

Proposed project interfaces:

```python
@dataclass
class RouteInput:
    q_pre_rope: Array        # [1,Hq,1,dh], after base Q norm; Seer gate input
    q_post_rope: Array       # [1,Hq,1,dh], actual attention and Quest input
    block_features: object   # scorer-specific cached features, only visible prefix
    valid_block_ids: Array   # [M], sorted unique absolute IDs
    token_counts: Array      # [M], excludes padding
    layer: int
    query_position: int
    kv_group_count: int

@dataclass
class RouteResult:
    support_ids: Array       # [R,Kmax], block/page IDs or absolute token IDs
    support_kind: str        # "blocks" or "tokens"
    route_axis: str          # "kv_group", "query_head", or "shared"
    block_tokens: int        # native page size; 1 for token support
    valid: Array             # [R,Kmax], -1 IDs only in invalid padding
    token_reads: Array       # [R], actual valid tokens, mandatory included
    diagnostics: dict       # optional; excluded from normal timed path

class BlockScorer(Protocol):
    def begin_request(self, model_config, block_size): ...
    def observe_prefill(self, layer, q_pre_rope, q_post_rope,
                        k_pre_rope, k_post_rope, values, positions): ...
    def finalize_prefill(self, layer): ...
    def update_cache(self, layer, q_pre_rope, q_post_rope,
                     k_pre_rope, k_post_rope, values, positions): ...
    def score(self, route_input) -> Array: ...  # [G,M], before top-k

class Selector(Protocol):
    def begin_query(self, position, valid_block_ids): ...
    def select(self, scores, mandatory, token_counts, budget, state) -> RouteResult: ...
    def after_layer(self, selections, layer_is_dense: bool): ...
    def end_query(self): ...

class AttentionBackend(Protocol):
    def attend(self, q, full_k, full_v, route, scale) -> Array: ...
```

The adapter owns the model's learned transformations. The selector gets only cheap scores, public position/count metadata, and its own earlier selection history. It receives no answer labels, supporting-fact labels, dense attention probabilities, future tokens, or scores from future layers. All routing tensors remain on-device in the production timing path.

`observe_prefill` consumes each dense-prefix chunk with absolute positions; `finalize_prefill` constructs any whole-prefix scorer state after its last chunk. During routed suffix/generation, `update_cache` observes each current Q/K/V exactly once per layer/query; `score` must not append that query again. This supports Seer summaries, SIA token indexes, and SketchWalk's Q-window/matrix cache without changing their state lifetimes. Replay uses cloned or separate state.

Before attention, `lower_to_tokens(route, N)` expands valid native blocks/pages into sorted unique absolute token IDs per route, preserving partial lengths. It does not expand a native token selection to larger blocks. GQA block selectors use `route_axis=kv_group`; Quest uses query-head routes; LessIsMore uses one shared token route. The attention backend maps query head h to KV head `h // (Hq/Hkv)`, and gathers the corresponding route. A shared route broadcasts over all heads. Count query-token pairs after that mapping and unique KV fetches separately. This lowering contract permits strong native-granularity baselines without changing the candidate's internal block-state definition.

For common-comparison runs, form `mandatory` as **absolute token IDs** in the current, preceding and first **canonical 64-token blocks**, deduplicated and clipped to the visible prefix. Lower that token set to each method's native support units; `block_tokens=1` must not turn it into just three tokens. All forced tokens consume budget. Native-policy runs retain each baseline's separate declared policy, with true reads reported; in particular SIA's exact first-64/latest-64 policy is distinct. At short contexts return all tokens if the budget covers them. If mandatory support exceeds a budget, mark the configuration invalid; do not overspend silently.

### 4.3 Coverage state

The initial candidate coordinates **across layers**, pooling selection events across KV groups. It does not impose an arbitrary order on simultaneous heads. At each new query token set `v=zeros(M)` and `z=0`. Before layer \(\ell\):

\[
c_{\ell,j}=\begin{cases}0&z=0,\\v_j/z&z>0.\end{cases}
\]

After all groups in that layer finish selection/execution:

\[
e_{\ell,j}=\frac{1}{H_{kv}}\sum_g\mathbf1\{j\text{ selected by group }g\text{ as an optional block}\},
\qquad v\leftarrow\gamma v+e_\ell,\quad z\leftarrow\gamma z+1.
\]

Here \(0\le\gamma\le1\), so \(0\le v_j\le z\) inductively and \(c\in[0,1]\). \(\gamma=1\) is historical fractional service; \(\gamma=0\) remembers the immediately preceding processed-layer event, including zero after a dense layer. Mandatory blocks are excluded from optional-history events. Dense layers use `e=0` and advance the recurrence. This is the candidate's convention, not a claim that a dense layer read nothing. Preserve each native controller's separately specified transition in source-form comparisons.

Fractional service is the primary event because an any-group union can quickly cover almost every block and erase useful variation. It still does not measure semantic novelty. Retain any-group union and per-group history as explicit controls. During the pilot record event dispersion, coverage variance near the relevance cutoff, support overlap, fraction of routes changed, and fraction repaired all the way back to top-k. Do not choose confirmatory examples by these diagnostics.

Reset the state for every new token and every independent request. Cache selected IDs for logging only when requested; no semantic claim attaches to a visit. Cross-head coordination, persistent history across generated tokens, and accumulated attention mass are separate ablations, not hidden additions to the initial method.

### 4.4 Relevance-constrained selection

For each group, first remove mandatory blocks from the optional universe \(U\). Initially use fixed-size full optional blocks; the newest partial block is mandatory. Let \(k\) be the remaining number of affordable optional blocks. Normalize finite scorer logits within \(U\):

\[
u_j=(s_j-\operatorname{mean}(s_U))/\max(\operatorname{std}(s_U),10^{-6}).
\]

Use FP32 for score normalization, coverage arithmetic, and selection comparison. A stable tie chooses the smaller absolute block ID; never perturb scores with a tiny position-dependent number to fake stability.

Let \(T\) be ordinary top-\(k\) by \(s\). Protect the first
\(a=\min(k,\max(1,\lceil\beta k\rceil))\) blocks of \(T\) when \(k>0\). The proposed default \(\beta=0.25\) is a tunable relevance-protection convention, not a measured optimum.

Rank unprotected candidates by \(u_j-\lambda c_j\). Add enough to fill \(k\), obtaining \(S\). Impose

\[
R(S)=\sum_{j\in T}u_j-\sum_{j\in S}u_j\le\epsilon k.
\]

If this fails, repeatedly remove the lowest-\(u\) member of \(S\setminus T\) and add the highest-\(u\) member of \(T\setminus S\), with deterministic ID ties, until it holds. Protected blocks cannot be removed because they belong to \(T\). This is at most \(k\) repairs and always terminates at a feasible set. It is a transparent heuristic, not a solver claiming optimal constrained coverage.

```python
import math
import numpy as np

def choose_optional(scores, coverage, ids, k, lam, eps, beta,
                    anchors_enabled=True, regret_enabled=True):
    # Arrays align over optional U; do not index them by absolute block ID.
    ids = np.asarray(ids, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float32)
    c = np.asarray(coverage, dtype=np.float32)
    assert ids.ndim == 1 and s.shape == c.shape == ids.shape
    assert np.all(ids[1:] > ids[:-1]) and 0 <= k <= len(ids)
    assert np.all(np.isfinite(s)) and np.all(np.isfinite(c))
    assert np.all((0 <= c) & (c <= 1))
    assert lam >= 0 and eps >= 0 and 0 <= beta <= 1
    if k == 0:
        return ids[:0]
    # All ranks/sets below contain LOCAL positions in the aligned arrays.
    top = np.lexsort((ids, -s))[:k]
    if lam == 0 or np.all(c == 0) or k == len(ids):
        return np.sort(ids[top])
    u = (s - s.mean(dtype=np.float32)) / max(
        float(s.std(ddof=0, dtype=np.float32)), 1e-6)
    anchor_n = min(k, max(1, math.ceil(beta * k))) if anchors_enabled else 0
    protected = top[:anchor_n]
    remaining = np.setdiff1d(np.arange(len(ids)), protected)
    priority = u - np.float32(lam) * c
    ranked = remaining[np.lexsort((ids[remaining], -priority[remaining]))]
    chosen = np.concatenate((protected, ranked[:k-anchor_n]))
    def total(positions):
        # Fixed summation order makes set equality an exact check.
        return float(np.sum(u[np.sort(positions)], dtype=np.float32))
    for _ in range(k if regret_enabled else 0):
        if total(top) - total(chosen) <= eps * k:
            break
        drops = np.setdiff1d(chosen, top)
        adds = np.setdiff1d(top, chosen)
        drop = drops[np.lexsort((ids[drops], u[drops]))[0]]
        add = adds[np.lexsort((ids[adds], -u[adds]))[0]]
        chosen[chosen == drop] = add
    if regret_enabled:
        assert total(top) - total(chosen) <= eps * k
    return np.sort(ids[chosen])
```

An implementation may optimize the repair loop by sorting differences once and computing cumulative score recovery on-device. It must return exactly the reference result, including ties. Do not insert `.item()`, host sets, or device-to-host synchronization inside each layer of a timed implementation. The reference loop is a CPU correctness specification.

The constraint limits only a **proxy relevance score**, not true attention mass, information loss, or accuracy. Its scientific value must be demonstrated. \(\lambda=0\) or zero history reproduces top-k; full budget reproduces dense support. At \(\epsilon=0\), unequal-score replacements cannot reduce total relevance, though score ties can still change IDs. An explicit no-op baseline bypasses the coverage code to quantify overhead even when outputs agree.

The primary has `anchors_enabled=True` and `regret_enabled=True`. Anchor-off sets the former false, not merely beta=0; regret-off sets the latter false, not an arbitrarily large epsilon. These switches apply identically to wrapped controls. They are explicit ablations of the primary definition.

### 4.5 Budget, head, and padding conventions

Record two read counts: query–token pairs, including repeated reads by separate query heads; and unique KV tokens fetched per KV group. Native per-head Quest may fetch repeated KV entries even when mathematical pairs are matched. Both matter for GQA latency. Logical reuse is not a guaranteed physical memory-traffic reduction.

At a common \(b=64\), compare budgets on a declared lattice such as 256, 512, 1024, 2048, and 4096 visible tokens per group, retaining only budgets feasible and informative at the selected context length. These are starting operating points, not success thresholds. Baseline native page-16 results use their true token counts. Do not compare eight 16-token pages with eight 64-token pages as equal computation.

No selected token appears twice in a group. Padding has zero effective attention probability; its K/V contents cannot affect the result. Partial blocks use actual valid lengths. Gathered K/V retain already-applied absolute-position RoPE. The selected output is ordinary softmax attention **renormalized over selected tokens**, followed by the unchanged model output projection. No hidden gate multiplier is retained at inference unless the original method explicitly requires it.

### 4.6 Attention execution

Implement a trusted FP32 dense-mask reference for short contexts, then a gathered implementation. For GQA-shared routes, gathered tensors have shapes `q=[1,Hq,1,dh]` and `k,v=[1,Hkv,Bmax,dh]`. MLX SDPA supports GQA without repeating KV heads; preserve this. Invalid/padded keys require an appropriate mask, or separate equal-length groups into valid calls. Test fused-kernel availability rather than assuming it. [R35]

For the per-head Quest reference, either group independent query heads as the batch dimension with one KV head, or use a correctly masked vectorized implementation. Count the repeated gathers and compare latency honestly. A later kernel optimization must be applied to every method using the same support representation.

A custom Metal sparse kernel is optional and gated by profiling: implement it only if gathered execution is correct, a quality advantage survives, and measured gather/launch overhead blocks a plausible useful speedup. An unfused masked dense kernel is a correctness reference, not the speed baseline. Fewer attention pairs alone do not authorize a runtime claim.

### 4.7 Exact invariants and indispensable tests

- Full selection equals the unmodified dense attention output and full model logits within the precision envelope calibrated against a trusted reference.
- Zero penalty, zero coverage, all-equal inputs, one optional block, no optional blocks, context shorter than a block, and contexts just below/above block boundaries behave as specified.
- Changing any future input token leaves earlier outputs **and routing decisions** unchanged. Query-tile averages/maxima that inspect later queries are forbidden even if the final attention matrix is causally masked.
- Reordering simultaneous heads does not alter shared history. State updates occur after selection and reset exactly at the next query.
- The same absolute block retains its ID through all layers; GQA maps the correct query heads to their KV group; padding and request batching cannot leak state.
- The highest protected relevance candidates survive; the score-regret constraint is satisfied; selection cardinality and actual token reads match the manifest.
- Raw max/min/sum/count accumulators of the growing final block equal a fresh recomputation after every token. Projected/normalized/rotated summaries match the declared source reference when a block becomes complete; the source's inactive partial-block placeholder is tested separately. Do not incrementally pool an already transformed mean.
- An intentionally broken local-only selector loses on remote-evidence fixtures; equivalent selectors do not produce a false quality or latency gain after appropriate measurement.

### 4.8 Published gate port and source discrepancies

The gate must be specified below the paper-summary level. For each Qwen3-4B layer, the inspected Seer implementation groups four normalized **pre-model-RoPE** query heads for each of eight KV heads. Its query projection has shape `[8,4,128,128]`. For each 64-token block it concatenates key maximum, minimum, and mean, producing 384 features per KV head, and projects with weights `[8,384,128]`. Learned gate RMS normalization uses epsilon `1e-6`; gate RoPE and a dot product scaled by `1/sqrt(128)` then produce block scores. There are four learned tensors per layer: query/key projections and query/key norm weights. The expected total is 33,039,360 gate parameters. Preserve tensor orientation from the source; do not transpose by guesswork. [R43]

A small CPU reference must execute the same operations from the four tensors without importing the CUDA-dependent package. Convert the official `.pth` with `torch.load(..., map_location="cpu", weights_only=True)`, validate every key/shape, and preserve values in safetensors. Record the source commit and exact conversion hash. A file of matching dimensions is not sufficient proof that the correct gate was loaded.

Source inspection found two material ambiguities to settle **before model comparisons**:

- The inspected Microsoft code uses block-start positions for compressed prefill keys, while a newly completed decode block receives the current/end-token position. Preserve this as an explicitly named source-compatible mode; also test one internally consistent position convention as a separately named repair. Neither may silently replace the other. Cache-transition fixtures must cross a 64-token boundary with sparse budgets below the context length.
- Mandatory first/current blocks can be added outside the nominal top-k mask. Keep a source-compatible budget mode with true read counts and a common **total-budget** mode that reserves mandatory slots. The latter is a defined adaptation. A zero-coverage comparison always uses the same mode as its coverage treatment.

Resolve the installed source version's behavior by tracing actual tensors and masks. Keep both mathematically specified modes if the discrepancy remains. Source-compatible unmodified-gate results belong in the baseline envelope; an improvement that exists only against an accidentally broken convention is invalid. The optional SAS export has related but distinct forced-block accounting and must not inherit the Microsoft mode without verification. [R43, R44]

The inspected compressed-cache implementation also advances one shared remainder counter in layer 0 before later layers write. A small control-flow trace shows that a non-block-aligned prefix can corrupt later layers' first completed block. The canonical port therefore uses a per-layer or absolute-position-derived write slot and uncorrupted raw statistics. **Source-compatible** here means gate math, positions, and budget conventions; it never requires retaining a demonstrated mutable-cache bug. An optional legacy-bug diagnostic is excluded from the competitive baseline. Test a 65-token prefix followed by 63 routed tokens in at least two layers, as well as aligned prefixes. This finding is about the inspected source revision, not a measurement of upstream model accuracy. [R43]

Dense prefix chunks must append to summaries by absolute block position, preserving partial-block raw accumulators across chunks. Do not reuse a source `prefill` initializer that overwrites the compressed cache on every chunk. Compare chunked construction with a one-shot reference for the same full prefix. Prefix-completed blocks receive the chosen prefix position convention; later decode completion follows the separately declared mode.

## 5. Baselines, data, metrics, and confirmatory protocol

### 5.1 Comparison tiers and baseline obligations

**Tier A, required local competitors:** optimized dense MLX-LM; source-compatible and common-budget SeerAttention-R; native-granularity Quest; LessIsMore's Qwen3 selection/reuse algorithm; a Self-Indexing Attention decode adapter; corrected-source SketchWalk decode; and the close coverage/diversity/history adapters below. All use the same frozen Qwen3-4B base, precision, tokenizer, prompts and tested attention backend. Published CUDA implementations are not reproduced by a mathematical Metal port. [R59, R60]

**Tier B, conditionally local:** SAS's published 4B gate if permission to use the released code/weights is established; additional original-method reproduction where its data and faithful implementation fit. A missing license is an asset gate, not a reason to claim a win over SAS. No paid service, remote GPU, or external machine is needed for the required path.

**Tier C, context only:** native NSA/DSA/MoBA training, CLSA's different architecture, full VASC geometry models, K-VEC's original large-model eviction results, and large CUDA kernel results. List them to delimit the contribution; do not put their published scores in a controlled local result table.

**Tier D, diagnostics:** dense score oracle, fixed local/random/dilated support, answer-support oracle, exact-history equality implementation, and intentionally broken masks. They isolate mechanisms and validate the harness; beating them does not establish competitiveness.

The required local methods are:

| Project method ID | Exact behavior and role | Configuration/fidelity requirements |
|---|---|---|
| `dense` | Unmodified full attention with the best valid dense backend. | Benchmark both common split and ordinary batched prompt ingestion; never a deliberately materialized quadratic reference for speed. |
| `seer_native`, `seer_total` | Original pretrained gate top-k with its native conventions; identical gate with explicit total-read budget. | Block 64; exact projection/normalization/RoPE and source discrepancies from Section 4.8; no coverage. |
| `quest_native`, `quest_common` | Score page \(j\) by \(\sum_r\max(q_r K^{min}_{jr},q_r K^{max}_{jr})\). | Native page 16 and first two dense layers are eligible; also test page 64/common mandatory policy. Per-query-head port is required. Group max/sum aggregation is an additional adaptation. [R23] |
| `less_is_more` | Dense early/selection layers choose a unified token set that later sparse layers reuse. | Published Qwen3-4B defaults start with dense layers 0–1 and selection at 2 and 20; reserve 25% recent tokens. Charge all full scoring/attention layers. Its exact union rule is specified below. [R54, R55] |
| `seer_reuse` | The same Seer scorer refreshes selected IDs at designated layers; intervening layers reuse those IDs on their own layer's KV. | Refresh the first sparse layer of every query; period 1 equals top-k. Compare periods 2/4 and a loss-calibrated schedule; count skipped scoring. This is an IndexCache-inspired adaptation, not a native DSA reproduction. [R9] |
| `sia_decode_total` | A transformed key sign/norm index ranks individual tokens per KV group. | Required mathematical decode adapter; same frozen 4B model, full KV retained, exact total-token budget. No native grouped-prefill or KV-compression claim. [R59] |
| `sketchwalk_release_total` | Sketch probabilities and a cached within-layer walk rank shared blocks. | Required corrected-source Qwen3 decode adapter; retain its scorer/cache recurrence, repair density selection, and count all state/scoring work. Distinct from the paper's cross-layer product. [R60] |
| `mean_key` | Query dot pooled block key, followed by common top-k. | A MoBA-inspired scoring control, not a trained MoBA model. |
| `kvec_adapter` | Relevance-weighted low-coverage bonus plus hard protected relevance set. | Query-local, block-level reversible adaptation; its original prefill eviction and cross-head observation-window module are not reproduced. |
| `deficit_adapter` | Current normalized demand plus backlog of unserved demand; update using actual selections. | Explicit language adapter of the VASC controller, initially value-neutral; defined below. |
| `mmr_current`, `mmr_history` | Greedy relevance minus maximum embedding similarity to selected/current or previous-layer candidates. | Mean-token-embedding similarity is a cheap proxy, not established semantic redundancy. Include its cache and comparisons in cost. |
| `identity_sum` | Independently implemented relevance minus decayed summed identity similarity. | Include the same group factor 1/Hkv and normalizer z; then it is an equality control, not max-similarity MMR or a contribution. |
| `score_ema_adapter` | Penalize normalized prior relevance scores for all blocks, including never-selected blocks. | Same query reset and wrapper; separates score history from executed-service history. Vortex-inspired cross-depth adaptation. [R56] |
| `constrained_random` | Add seeded random ranks in place of coverage, then apply identical protected anchors and regret repair. | Same candidate set, state size where meaningful, tuning allowance, and randomness protocol. Tests generic perturbation. |
| `permuted_history` | Apply a fixed random permutation to actual past coverage before scoring; update the underlying state from actual executed selections. | Preserve its histogram within frozen relevance-rank/position strata; same wrapper and coefficient. This tests whether the identity of previously read addresses matters. |
| `bounded_coverage` | Section 4, applied to the best selected gate/scorer. | The primary experimental treatment; compare to that exact scorer with coverage off as well as the baseline champion. |

LessIsMore's `UnionFlatten` is not mean attention or voting: sort historical candidates within each query head, visit them rank first and head second, deduplicate stably, retain the required historical count, then add recent/sink tokens according to the original budget convention. Pin the reference implementation; its ordinary accuracy path can compute dense scores before masking in every layer, so that path is not evidence of sparse speed. The Metal port must gather only at truly sparse layers and charge full scoring at selection layers. Tune recent fraction and allowed refresh locations on development data, while preserving the published default in the evaluated set. [R55]

For all probability-based adapters below, \(p_{g}=\operatorname{softmax}(s_{g,U})\) over **optional blocks only**. For `kvec_adapter`, let \(I_j=\max_g p_{g,j}\) and rank \(p_{g,j}+\eta I_j(1-c_j)\). The source-form state is \(c_{\ell,j}=n_j/(\ell+1)\), with zero-based physical layer \(\ell\) and \(n_j\) counting preceding sparse layers that selected j in any group; at a dense layer increment no optional count. This is explicitly a decoding adaptation of the original prefill-eviction state. A second variant uses Section 4.3's fractional/decayed state instead. Its \(\eta=0\) control is top-k; zero history alone need not be. [R19]

For `deficit_adapter`, pair consecutive IDs in the sorted optional universe after removing mandatory blocks, fixed for the entire query, with an odd singleton. This pairing is a language adaptation. Let
\(D_{g,j}=\min(1,k p_{g,j})\), or value-aware \(D=\Pi_k(p\odot I^{value})\) only in the labelled extra ablation, where \([\Pi_k(r)]_j=\min(1,kr_j/\sum r)\) and \(\Pi_k(0)=0\). Do not renormalize after clipping. Average demand and final executed binary masks over each pair to obtain \(a_{g,p},x_{g,p}\). Rank \(D_{g,j}+\Pi_{\sum_p a_{g,p}}(M_g)_{p(j)}\), then update \(M\leftarrow[M+a-x]_+\). Reset per query; leave backlog unchanged at a dense layer in this named adapter. No raw decay is in the original recurrence. For the optional value-aware version, cache each block's mean value at the current layer, use half the RMS difference of the two pair means for both children, and zero contrast for a singleton; charge the cache construction. [R20]

K-VEC, deficit, and MMR must each have both their raw proposal rule and a version with the **identical protected-anchor and relevance-regret wrapper** used by the candidate. Execution-dependent histories observe the final executed mask after repair; score-EMA and SketchWalk retain their separately defined updates. Otherwise a gain from safeguards could be misattributed to a new state rule. The K-VEC raw control still retains its hard anchor adaptation, explicitly replacing the paper's score-boost protection.

For embedding-similarity MMR, construct one L2-normalized feature per block from mean **input token embeddings**, using zero features when the norm is below 1e-8. Greedily choose by normalized relevance minus \(\alpha\max_{i\in H\cup S}\max(0,f_j^Tf_i)\), with an empty-set maximum of zero. `H` is empty for within-layer MMR and previous-layer selected optional blocks for history MMR. Include feature construction and comparisons in cost. If claiming superiority over diversity methods broadly, also include an AnchorPrune-inspired adapter: reserve the same anchors, then greedily rank \(p_j\min_{i\in H\cup S}(1-\max(0,f_j^Tf_i))\), with empty-set distance one, followed by the same wrapper. This is a fixed-anchor text adaptation, not the original visual model. A conditional-DPP challenger is optional unless an observed redundancy failure makes it decisive. [R16, R21, R22]

For `seer_reuse`, period 1 refreshes every sparse layer. Periodic alternatives and greedy removal of refresh layers are calibrated on identical development answer NLL: begin with all layers refreshing, evaluate each allowable single removal, retain the one with smallest loss, and repeat until the prespecified refresh count. The first sparse layer stays a refresh. Skip scores at reuse layers, and recompute mandatory support consistently. No new query may inherit the preceding token's support. Count calibration forward passes; downstream development accuracy validates a chosen schedule.

For `permuted_history`, use a deterministic keyed PRNG from `(method_seed, episode_id, query_position, layer)` and Fisher–Yates shuffle within fixed quartiles of relative block position crossed with current relevance-rank quartiles. Scores are current-layer information, so this is causal. Singleton strata remain unchanged and their rate is logged. Use the same shuffled map for shared groups after defining the rank stratum by mean standardized group score. Also run unrestricted permutation as a diagnostic if strata make the sham nearly identical. Random controls use multiple paired method seeds during development and a frozen seed-averaged protocol for confirmation; the episode remains the independent unit.

For `constrained_random`, generate a random permutation rank r in 0,...,|U|-1 and use `u - lambda*r/max(|U|-1,1)` before the common wrapper. This keeps the perturbation in [0,1], matching coverage's scale across context lengths. Use the same keyed RNG convention. Raw ranks increasing with M are not a comparable penalty.

For `score_ema_adapter`, reset `a=zeros(M), z=0` at each query; before each sparse layer set `h=0` if z=0 else a/z and propose by `u[g,j] - lambda*h[j]`. After scoring, update `a=gamma*a+mean_g(u[g,:])` for **all optional blocks**, and `z=gamma*z+1`. At a dense layer decay a and advance z without a new score event. Reuse the same anchors and regret wrapper. Signed standardized score history is not bounded in [0,1], so calibrate its lambda independently and record achieved perturbation/exchange/regret rates. Pass its proposed ranking to the common wrapper; do not feed signed h into `choose_optional`'s bounded-coverage input or clip it to satisfy that validator. This is a specified cross-layer adaptation of earlier score-history proposals, not a native Vortex reproduction. [R56]

For `mean_key`, pool post-RoPE keys within each valid block. Compute every query head's dot product with its KV group's mean key divided by sqrt(dh), then take the arithmetic mean of the four query-head scores to obtain that group's block score. This declared mean aggregation is a control, not native MoBA's learned behavior.

#### Self-Indexing Attention: exact local decode adapter

Use a fixed independently seeded sign vector r in {-1,+1}^128 per layer/KV head, shared by that group's four query heads. Derive its bits from SHA-256 of the UTF-8 string `sia-v1|seed|layer|kv_head`, using the first 128 digest bits in most-significant-bit order and mapping 0/1 to -1/+1; store the vectors/hash. This is a reproducible adapter RNG convention, not a claim to reproduce a framework's native seed. For row vectors define `rot(x)=(x*r) H128/sqrt(128)` after base Q/K normalization and model RoPE. Signs precede the transform; applying shared signs afterward would cancel in sign agreement. MLX exposes this transform, but WP-00 must probe dispatch. [R59]

At the current query only, pool each transformed coordinate across the four heads by largest absolute magnitude, retaining its sign; break ties by canonical head ID and define sign(0)=+1. For token i cache transformed-key sign bits and FP32 norm n_i. Rank by
\[
s_i=\max\left(\sum_{d=1}^{128}\mathbf1[\operatorname{sgn}(\hat q_d)=\operatorname{sgn}(\hat k_{i,d})]-64,\,0\right)n_i.
\]
Reserve the deduplicated first 64 and latest 64 tokens, including current, **inside** total B; then stable top-k fills the remainder. This 64/64 decode policy is a declared default, not a separately verified native override. Include the paper-style logical budget `min(N,max(768,ceil(0.05*N)))`; absent documented kernel rounding, label it an unaligned adaptation. Lower fixed budgets are additional matched-cost adaptations.

Retain ordinary BF16 K/V and add this index; do not claim free metadata or cache compression. A packed 128-bit sign plus FP32 norm costs 20 bytes/token/KV head/layer, about 90 MiB at 16K for this model; an unpacked 128-byte sign representation plus norm is about 594 MiB. Count the actual representation and sign-matching work. A boolean-vector reference is acceptable for quality; it is not evidence of a fast bit-packed Metal kernel. Do not import grouped-prefill routing that pools later query positions and only masks keys afterward. The adapter uses one causal query position at a time. [R59]

#### SketchWalk: corrected-source decode comparator

Pin the Qwen3 release in R60; use **dense prefill** and preserve dense layers 0–1 during decoding (`layer_idx>1` activates the source walk). At each remaining layer, stream post-RoPE Q/K into absolute block means (block 64), preserving its zero-padded partial-block mean. Build head-aggregated block-probability matrix A once after the full dense prefix, and seed the last W post-RoPE queries. Each new query is appended once, retaining W; construct current block probabilities a from this window and pooled keys. This request-local state persists across tokens, unlike candidate coverage.

The released projection uses fixed random input signs, an **unnormalized** Walsh–Hadamard transform, a fixed coordinate permutation/subset of size p, and division by sqrt(dh/p). Retain that scaling; SIA's normalized transform changes the temperature. It casts block/query-window Q/K inputs to BF16 **before** the Hadamard projection, computes dot logits divided by sqrt(p) and then by 100, applies softmax per head/query, sums over heads/window, and normalizes to a. Initialize A analogously. Include p=128, W=32, damping beta=0.25 and degree 3 from the released decode configuration, recording constructor defaults separately. Tune fairly; store each layer's realized sign/permutation arrays so RNG-library differences do not masquerade as a method effect.

On a new block, zero-extend old A columns and append rows equal to a. Let i be the current block and alpha=1/(1+position mod 64). Mix A's row i with a using alpha, normalize that row, then mix its column i with a using alpha and normalize that column, preserving this order. This column operation need not preserve every row sum; do not silently “correct” it into a different recurrence. Start v=a and iterate `v=(1-beta)*a+beta*(v@A)` for the selected degree. Rank blocks by v, shared across heads. Incremental mean updates can avoid re-reading all K if parity with the zero-padded reference holds; share applicable caching optimizations.

The released decode threshold has shape `[batch,1,1]` and is then set to zero, admitting every positive walk score. Treat that as a density diagnostic; replace it with deterministic exact-k selection after reserving mandatory support. This is **corrected-source total-budget SketchWalk**, not an unqualified reproduction of reported sparse speed. Use the common gathered backend for timing; the source's masked SDPA is a correctness reference. A trace-proven unused full-token score computation can be removed with the optimization documented.

Run the common mandatory policy and a separately named **absolute-position source-parameter policy** where its cost floor is feasible. The pinned config uses `random_walk_window=0.05`, `random_walk_sink=0.025`, `random_walk_kblocks_frac=0.125`, and block 64. For the declared corrected policy, reserve the first `ceil(max(1,floor(0.025*N))/64)` blocks and the latest `ceil(max(1,floor(0.05*N))/64)` blocks, clipping to available blocks and deduplicating; fill remaining total-budget slots by v. The released token-window backend uses zero-based query coordinates even during one-token decode and ties its sink union to the window size, so it does not faithfully implement this intended absolute policy. Record those repairs too. The 0.125 quantile is a fraction of nonmandatory candidates, not a total-budget guarantee; a source-density diagnostic may use `ceil(0.125*|U|)` optional blocks plus mandatory support with true reads reported. It must not be called equal-cost to total B unless its measured count agrees. [R60]

The paper's equation `R_l=R_(l-1) W_l` and its sketch-score power are a separate cross-layer walk construction. The inspected release's decode recurrence above is different. Do not conflate either with executed-selection history or exact causal ancestors. A paper-product adaptation is conditional on a new fully declared convention, but the accessible corrected-source competitor is required now. At 16K, its 36 head-aggregated FP32 256-by-256 matrices occupy 9 MiB; Q windows and block representatives add memory and preprocessing. Memory alone is not grounds for excluding it. [R60]

### 5.2 Fair resource matching

Run two complementary evaluations:

1. **Mechanism attribution:** identical scorer, block size, layer allocation, mandatory support, preprocessing information, and exact per-layer executed reads. This isolates selection history and its constraint.
2. **Best useful implementation:** each baseline may use its justified native block size, calibrated dense/refresh layers, reuse, and optimized execution. Compare the resulting quality–total-read and quality–wall-time frontiers.

For a baseline with \(d\) fully dense layers and a sparse-layer budget \(B_s\), its effective average read budget is
\(B_{eff}=[dN+(L-d)B_s]/L\), before other exact scoring passes. LessIsMore may be unable to occupy the tightest total budgets because its selection layers impose a real floor. Report that limitation and its full frontier; do not silently omit dense-layer cost or remove those layers to create a crippled baseline. Account separately for scorer operations that read summaries rather than full KV.

Preprocessing cost includes gate-feature construction, semantic-feature caches, static-mask construction, and tokenization. Training/adaptation expenditure is zero for the frozen primary treatment; published pretrained base/gates are reusable assets with their origins disclosed. A local trained extension has its own compute ledger. Generally useful gather, sorting, compilation, summary caching, or batching improvements go to all compatible methods.

Tune on the same development population and disclose candidate counts, scored tokens, wall time, and failed configurations. Include published defaults. Use a common maximum development-evaluation expenditure, set after calibration, and allow unused budget to the simpler baseline rather than forcing pointless trials. A baseline still materially improving when its allocation ends is under-tuned: expand development resources before freezing, or narrow the competitive claim. Never use the protected test results to choose a baseline, scorer, budget, or constraint.

### 5.3 Synthetic populations

Create natural-language record tasks for a frozen pretrained model; do not feed unfamiliar integer-token MQAR directly to Qwen and interpret failure as an architecture result. Zoology/MQAR is useful for an optional trained synthetic model, not an interchangeable zero-shot dataset. [R49]

Every generated episode has a latent record set, explicit gold dependencies, unique ID, independent seed, family, prompt template version, length stratum, evidence spans, and answer. Store evidence labels outside model input. Use disjoint generators/seed namespaces for development and confirmation. Sample names and answer values from tokenizer-checked pools with balanced lengths; the value pool must not correlate with location or difficulty.

| Family | Required construction and gold output | What it establishes |
|---|---|---|
| `independent_facts` | Ask for an ordered tuple of 2, 3, or 4 attributes from separate randomly placed records. Output all values, with each necessary. Distractor copies vary independently of answer values. | Whether limited reads recover distinct required evidence. |
| `pointer_chain` | Records encode two- or three-link entity relationships; follow the chain and return a terminal value. Place links independently; avoid unique formatting of relevant links. | Content-dependent discovery after earlier clues. A model must pass the dense competence gate. |
| `same_block_attributes` | Put all required attributes in one remote block, using the same answer format/number of facts as `independent_facts`. | More positional spread is not intrinsically useful; checks costly suppression of the same location. |
| `anchor_return` | A chain returns to an earlier record to request a different field; same answer arity as a matched no-return chain. | Revisiting can be necessary at the task level. It does not prove which neural layer must perform the revisit. |
| `single_fact` | One remote lookup plus irrelevant or duplicate records. | Negative control for indiscriminate exploration. |
| `many_fact_aggregate` | Bounded counting/comparison across records, with a gold computation and difficulty independently controlled. | A distributed-attention failure regime; include only if dense model competence is demonstrated. |

Assign every underlying latent record world and **all** derived questions, task families, positions, lengths, and paraphrases to one split and one `cluster_id`. Task-family categories remain represented in both splits. Retain every selected variant regardless of method outcome. Prefer independently generated worlds for each family for simplicity; when worlds cross families, preserve their joint dependence in analysis.

Start length calibration at 4K and 8K tokens, then 16K if memory/throughput and task sensitivity justify it. Place evidence uniformly over declared relative-position strata, including near block boundaries and in early context. Do not always arrange it at exact block starts. Scaling context adds independent distractors without changing required facts; record the transformation. Confirmatory length strata and family weights are frozen after the pilot.

Validate gold answers with a deterministic record interpreter. Remove one required record/attribute and verify the latent problem becomes underdetermined; create two completions with different correct answers and identical remaining evidence. This establishes task necessity without assuming that attention maps reveal causal reasoning. Model ablation tests then assess whether the actual model uses the evidence.

### 5.4 Natural data and licenses

**Required natural transfer population: HotpotQA v1.1 distractor.** Use the original public train and development JSONs, preserve original paragraph/sentence content and question IDs, and cite the CC BY-SA 4.0 terms. Train examples provide prompt and selector development; a protected subset of official development provides natural confirmation. The local subset is not the official test set. Use answer exact match and token F1 from the original evaluator. The method does not predict supporting sentences, so do not report joint answer/support score. Gold support labels are available solely for diagnostics. [R45, R46]

Use every valid item admitted by a predeclared deterministic ID-hash rule, including dense-model errors. Do not require “dense answered correctly” for inclusion. Parsing failures, duplicate IDs, and context-limit exclusions are defined and counted before scoring treatments. Supporting paragraphs must not be truncated away. Natural examples remain their original lengths; any appended distractor condition is explicitly **HotpotQA-derived**, statistically clustered with the original question, and reported separately from unmodified data.

**Optional independent confirmation: MuSiQue-Ans v1.0.** The original repository states CC BY 4.0 and supplies the answerable development file and evaluator. Keep decomposition/support fields hidden from methods and use the component-question overlap information to document potential contamination. Fetch availability and exact file hashes remain local gates. If this asset is unavailable or the dense model is at floor, record that limitation without replacing a frozen population after inspecting results. [R47]

Public pretrained weights may have seen benchmark text. This prevents a claim of uncontaminated general reasoning. Controlled new synthetic records, evidence-removal/shuffle controls, and held-out sources address different risks; none proves training-data absence. Test lexical and structural generator shift, not only fresh random seeds from one template.

### 5.5 Primary endpoint, guardrails, and claim hierarchy

**Primary endpoint:** paired difference in **all-required-facts exact accuracy**, in percentage points, on the confirmatory synthetic episode population, macro-averaged first over declared variants within an episode and then over the equally weighted `independent_facts`, `pointer_chain`, `same_block_attributes`, and `anchor_return` families. If a family cannot pass the dense sensitivity gate, amend the population during development, state why, and freeze a narrower claim before confirmation. Never drop a bad family afterward.

The primary contrast is `bounded_coverage` against a single development-selected baseline champion under a predeclared **per-routed-token read policy**, including its layer allocation, length strata, and block granularity. It is not automatically a fixed total-request-cost comparison: methods can generate different numbers of tokens. Report actual complete-request reads and latency separately. Also report the matched-scorer top-k contrast for attribution. Choose the champion only among policies with the same per-token cost envelope; methods with an irreducible dense-layer floor remain on the full frontier. If integer block sizes prevent exact matching, use the largest feasible budget no greater than the common envelope and report the unused capacity. Do not round an overspend down in the analysis.

For a confirmatory frontier claim, freeze the budget lattice and aggregation before evaluation. The default frontier output is a descriptive set of paired quality/cost points without interpolation. A declared area-under-quality-versus-log-read curve may use linear interpolation only between observed bracketing points on a common cost range; choose that range and rule on development data. No extrapolation or post-hoc winning point receives primary significance. Request-wide cost equivalence needs its own predeclared stopping/allocation policy and is outside the initial fixed-per-token claim.

Required secondaries and guardrails:

- Individual task-family accuracies and error/truncation rates; single-fact and same-block/return performance cannot be hidden by a global average.
- Answer negative log-likelihood in bits per gold answer token, with all methods teacher-forced on the same answer prefix in a separate run; this is a likelihood diagnostic, not free-generation accuracy.
- HotpotQA answer F1 and EM on the unmodified protected natural subset, with paired uncertainty; natural transfer is a distinct claim.
- Actual query–token pairs, KV tokens read per group, scorer/feature work, full-cache bytes, routing state bytes, and all offline/project-specific expenditure.
- Request latency, routed-token latency, dense-prefix time, time to first answer token, generated answer length, and peak memory. Token latency alone cannot hide longer/worse answers.
- Replay retained mass and relative output error \(\|y_{sparse}-y_{dense}\|_2/\max(\|y_{dense}\|_2,10^{-8})\), plus absolute error near zero norms; these never replace downstream endpoints.

For the primary quality-at-fixed-per-token-policy claim, a positive lower confidence bound establishes evidence of improvement on that population. It is not sufficient for “dramatically more efficient.” Practical interpretation includes complete-request cost and regressions. For a lower-cost-at-same-quality claim, prespecify an acceptable quality-loss margin from an actual use requirement or a declared resolution question. If no such requirement exists, report the paired quality interval and cost reduction as an estimated tradeoff rather than proven equivalence.

An improvement of one extra correct answer per hundred may be real; whether its overhead is worthwhile depends on use. There is no mandatory 2× speed threshold and no arbitrary percentage gain required for scientific success. Report the entire supported tradeoff, including tiny or negative effects.

### 5.6 Development, freeze, and independent confirmation

1. **Exploration:** implement parity tests, tune prompts/scorers/coverage/baselines, inspect failures, and estimate variance using development episodes only. Synthetic gold labels are public to the evaluator but inaccessible to methods. The agent may inspect development failures freely.
2. **Candidate choice:** choose the scorer and strongest baseline using development accuracy under the matched per-token read policy, with realized request latency/memory reported. If treatment only beats a weak scorer, test it on Seer and the strongest available scorer; do not promote the weak result.
3. **Freeze:** write a content-hashed `confirmatory_manifest.json` containing hypotheses, methods and revisions, prompt/template hashes, data membership, family/length weights, budgets, all hyperparameters, generation/extraction rules, sample size, CI/test procedure, exclusions, and timing design. Commit the manifest before running the protected split.
4. **Confirmation:** execute the fixed matrix once, retaining failed/unfavorable runs. Inspect completeness/technical errors without changing the scientific choices. Correct demonstrable implementation bugs by an amendment and fresh confirmation; an execution repair is not permission for result-driven retuning.
5. **Replication:** use a new synthetic generator template/seed namespace and, when available, an independent natural corpus. A second seed applied to the same protected outcomes is not fresh confirmation.

Proposed development hyperparameter families are \(\lambda\in\{0,0.25,0.5,1,2\}\), \(\gamma\in\{0,0.5,0.9,1\}\), \(\epsilon\in\{0,0.1,0.25,0.5\}\), and \(\beta\in\{0,0.25,0.5\}\), with the definition still preserving one anchor when \(k>0\). The actual zero-anchor ablation requires a separate switch. Do not run the entire Cartesian product blindly. Begin with one-factor diagnostic sweeps on cached development replay, retain technically valid configurations, then compare a prespecified shortlist on actual closed-loop development episodes. Replay is a cheap screen; it cannot select a winner without downstream confirmation. Supply comparable tuning effort to K-VEC/deficit/MMR/Quest/refresh schedules.

### 5.7 Statistical design and sensitivity

The independent quality unit is the underlying latent world or natural question/source cluster, not a token, head, layer, timing repetition, or derived permutation. Compute paired method differences for every variant, then average within each cluster/family cell. Resample **entire clusters jointly**, retaining their observations across families/lengths, and recompute the fixed macro weights. Stratify only by attributes that partition clusters, never by a family that splits one cluster. For independent single binary pairs, exact McNemar testing is a useful secondary check. Do not treat model seeds as new test examples. [R50]

The preceding cluster/family macro point estimator is for synthetic worlds. Natural HotpotQA EM/F1 is the mean over admitted **original question IDs**, with variants averaged within an ID only in derived conditions. Resample whole source clusters for uncertainty, then recompute that question-weighted ratio in each replicate; do not replace it by equal weighting of differently sized source components. Source-macro accuracy is an optional separately labelled statistic.

Use the pilot's paired standard deviation \(s_d\) to plan precision. A rough independent-episode sample size for CI half-width \(w\) on an accuracy fraction is
\(n\approx(1.96s_d/w)^2\); use simulation with the actual strata/clusters for the final choice. Publish a power/precision curve for feasible sample sizes instead of an arbitrary count. For illustration only, \(s_d^2=0.10\) and \(w=0.01\) imply about 3,842 independent pairs; that may be expensive, and the measured runtime determines whether that precision is feasible. This example is neither an observed variance nor a required success margin.

Before confirmation, select one fixed sample size from the precision/resource frontier and record which effect sizes it can resolve. If the attainable interval would be wider than the question of interest, label the proposed comparison underpowered; increase measured allocation if feasible, or report it as a diagnostic. Do not stop when significance first appears. No optional extension of the confirmatory sample after inspecting the direction of the effect is allowed under this fixed-sample protocol.

A single frozen primary contrast needs no correction for exploratory tuning on separate data. Freeze a hierarchical claim tree: test the primary synthetic contrast at 0.05; only if it passes may the declared secondary claim family be promoted. Apply Holm within that family, which includes standard-decode accuracy, natural answer F1 (EM descriptive), true-versus-permuted history, and any promoted revisit or source-mode effect. Additional family/budget/precision analyses remain descriptive unless included in that family before confirmation. Report both unadjusted effect intervals and adjusted decisions with the hierarchy. Failure of the primary does not erase secondary observations; it makes their positive interpretation exploratory. Bootstrap replicate count controls only numerical quantile error, not sample size.

Ordinary bootstrap intervals can collapse spuriously when observed paired differences are all zero. Never interpret `[0,0]` as proof of equality. With independent binary pairs and zero discordances, a conservative 95% bound on the possible accuracy difference is ±\([1-0.05^{1/n}]\); at n=100 this is approximately ±2.95 percentage points. With general cluster-level differences \(d_i\in[-1,1]\), use a conservative bounded-difference interval, clipped to [-1,1], when bootstrap support is degenerate or discordance is too rare for validated coverage. For independent equally weighted clusters, the Hoeffding half-width is \(\sqrt{2\log(2/0.05)/n}\); for fixed normalized cluster weights \(w_i\), replace 1/n with \(\sum_i w_i^2\). Freeze the rare-event switch using pilot simulations. Always report discordant clusters and changed-support rates. This fallback may be wide; that is honest lack of information.

Zero-crossing intervals are inconclusive about small effects unless their width rules out a prespecified effect of interest. Tight intervals around zero support “no practically detectable advantage at this setting,” not universal impossibility. A negative mean with a wide interval is not decisive evidence against the underlying mechanism.

### 5.8 Generation and scoring

Use the base tokenizer/chat template at its pinned revision with `enable_thinking=False` for the primary short-answer study. Use deterministic greedy decoding as a deliberate controlled experimental setting; it is not the model card's recommended stochastic production configuration. Fix the extraction rule and cap from development answer lengths and task requirements, and count cap hits as failures with their rate reported. Do not silently remove malformed outputs. [R40]

The synthetic evaluator accepts the exact ordered tuple/terminal value after only declared whitespace/case normalization; do not grant semantic credit through another model. Unit-test tokenization and extraction with correct, partial, duplicate, reordered, and wrong answers. Natural answer scoring matches the official Hotpot normalization, including yes/no handling. A sampled non-thinking companion can use the model's documented settings with paired random streams and all draws retained; it is a robustness endpoint, not a source of extra independent questions.

Long-thinking or long-generation claims require an additional population, sampling policy, completion-length budget, and suitable competitors such as RetroAttention; no short-output result establishes those claims. If such a workload is selected later, it receives its own pre-registration and resource calibration.

## 6. M3 Max feasibility and hardware-adaptive configuration

### 6.1 What is verified and what is not

The primary computation is frozen inference, batch one, with dense chunked prefix ingestion and single-query gathered attention. This avoids requiring sparse training gradients or reproducing a CUDA/Triton runtime. Current MLX documentation exposes GQA-capable SDPA, gather operations, stable sorting, and custom Metal kernels; it also exposes sparse matrix primitives, so it would be incorrect to say Metal has no sparse operations. None of these facts proves that the proposed shapes, masks, gate port, or entire model are fast on the user's M3 Max. [R34–R38]

**No M3 Max execution, model download, end-to-end quality measurement, or local throughput measurement was performed for this plan.** The laptop compatibility claims below are hypotheses gated by WP-00/03. The analytical checks in Section 2.4 ran in a different Linux environment.

Run these existing read-only system commands and save relevant output without serial numbers or unrelated personal configuration:

```bash
uname -m
sw_vers
sysctl -n hw.memsize
sysctl -n machdep.cpu.brand_string
system_profiler SPDisplaysDataType
df -h .
vm_stat
pmset -g batt
python3 -c 'import platform; print(platform.python_version(), platform.machine())'
```

Use the measured memory and available storage, not Apple's maximum configuration. An M3 Max can have different GPU/memory configurations; model name alone does not determine capacity. Do not infer capacity from the advertised family maximum. [R39]

### 6.2 Stack, versions, and assets

Use a project-local native arm64 Python environment, initially Python 3.11 or newer. The package metadata inspected on the research date lists **MLX 0.32.4** and **MLX-LM 0.32.2**. These are candidate resolution pins, not a tested compatible environment. Resolve their actual macOS/Python wheels and transitive requirements locally; the contemporary MLX-LM package may require a newer Transformers than old Qwen model-card examples. Do not force an old Transformers version merely because an older example used it. Record a resolved lock and hashes only after the critical operator probes pass. [R34, R36, R40]

Core dependencies are MLX/MLX-LM, NumPy, safetensors, Hugging Face Hub, PyYAML, psutil, and the minimum statistical/test packages actually used. Use PyTorch in a separate CPU reference/conversion environment if needed for trusted `.pth` gate loading; do not import the full CUDA Seer package. PyTorch 2.14.1 was visible in release metadata, but no particular PyTorch/MLX combined environment has been validated. MPS is a fallback backend only after its exact operators pass; it is not required for the MLX path. [R34, R38]

Existing environment commands, with Python minor version and dependency pins finalized by WP-00:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --dry-run 'mlx==0.32.4' 'mlx-lm==0.32.2'
```

These commands do not constitute a complete project installation. Create `pyproject.toml` and the resolved lock after inspecting the dry run and wheel compatibility; install that exact environment. Do not modify global packages, automatically enable CPU fallback, or use floating `main` assets in a confirmatory run.

Required assets are the **original** `Qwen/Qwen3-4B` checkpoint/tokenizer and the corresponding `SeerAttention/SeerAttention-Decode-Qwen3-4B-AttnGates` export. Later Qwen3 releases with similar names are different models. Record Hub commit, per-file SHA-256, license text, model/config IDs, and tokenizer/chat-template hashes. The Qwen repository declares Apache 2.0; Microsoft SeerAttention declares MIT. Check the exact gate export's accompanying metadata before use and retain notices. Downloadability alone is insufficient licensing evidence. SAS remains a conditional challenger because an adequate license was not established in the inspected release. [R40–R44]

The default is BF16 weights and BF16 KV, with FP32 score/state/reduction arithmetic where specified. If BF16 execution is unsupported or unsuitable on the actual stack, test an FP16 variant on **all** compared methods and report its parity envelope. Quantization changes rankings, logits, memory traffic, and possibly conclusions: a common 4-bit/8-bit base is a separately calibrated branch, with an unquantized overlap study. Do not compare a quantized sparse candidate against an unquantized dense baseline and attribute the whole difference to routing.

### 6.3 Memory and storage model

The inspected original Qwen3-4B configuration has 36 layers, 32 query heads, 8 KV heads, head dimension 128, hidden size 2560, intermediate size 9728, vocabulary 151936, and tied embeddings. The derived parameter count is 4,022,468,096; BF16 parameter payload is about **7.49 GiB**, before loader copies or allocator overhead. Its configured position limit is 40960; the core experiment stays at 4K–16K and makes no extrapolation claim. [R41]

For batch one and two-byte KV elements,

\[
\mathrm{KVBytes}(N)=2\times36\times8\times128\times N\times2
=147456N\ \text{bytes}.
\]

Full KV remains resident for every reversible selection method. Memory savings from eviction are not part of this proposal.

| Contributor | 4K tokens | 8K tokens | 16K tokens | Interpretation |
|---|---:|---:|---:|---|
| Full BF16 KV payload | 0.5625 GiB | 1.125 GiB | 2.25 GiB | Exact from config and element size, excluding allocation overhead. |
| One fully materialized FP32 dense score tensor, 32 heads | 2 GiB | 8 GiB | 32 GiB | An unacceptable speed-baseline implementation; softmax temporaries add more. |
| FP32 scores for a 512-query fallback prefill chunk | 0.25 GiB | 0.5 GiB | 1 GiB | Upper-order tensor estimate at full prefix; fused SDPA should avoid this materialization. |
| All-layer raw max/min/mean FP32 summaries, block 64 | 27 MiB | 54 MiB | 108 MiB | Conservative fully retained summary payload; streaming may lower it. |
| All-layer projected BF16 gate keys | 4.5 MiB | 9 MiB | 18 MiB | Gate dimension 128, 8 groups. |

The gate adds about 0.062 GiB of parameter payload. One 1024-token gathered KV buffer across eight groups is 4 MiB per layer; temporary duplication and masks depend on implementation. Shared FP32 coverage for 256 blocks is 1 KiB; per-group coverage is 8 KiB. State size is cheap, but repeated launches and gathers may still dominate its runtime.

A **planning estimate**, not a measurement, is roughly 10–16 GiB peak application memory for the 4B base, one gate, a 4K–16K context, fused/chunked prefill, no gradients, and no attention dumps. Loader duplication, compilation, allocator retention, logit arrays, and operating-system pressure can exceed that range. Measure active/peak MLX allocation, MLX cache allocation, process RSS, system memory pressure, and swap changes separately; they are not interchangeable and must not be summed blindly. [R37]

Inspect asset metadata before downloading. The original model payload is roughly 8 GB decimal, but temporary conversion files, an optional 0.6B checkpoint, wheels, dataset files, logs, and replay arrays add storage. Use the sum of exact asset sizes plus explicitly planned temporary/output sizes and measured working headroom as the download gate. An initial 10–15 GB asset envelope is only a planning aid. Keep only selected replay queries/layers and sufficient statistics by default; never archive all \(N^2\) attention maps.

### 6.4 Indispensable operator and parity probes

Before the 4B download, create synthetic arrays with the actual GQA dimensions. Test `mx.fast.scaled_dot_product_attention` for BF16/FP16, query lengths 1 and 512, KV lengths on both sides of block boundaries, repeated gathers, and nonuniform padding. Compare to an FP32 NumPy reference at short lengths, report max absolute/relative error, and verify a common output/logit error envelope. Test stable ID tie handling and gather index dtype. Calibrate tolerance from numerical scale and backend-to-reference differences; do not loosen it selectively until a candidate passes. [R35, R37]

For a single current query whose gathered keys are all visible, no causal mask is needed. MLX's string causal mask uses a lower-right alignment for nonsquare attention, whereas PyTorch's documented `is_causal` reference uses an upper-left alignment. Never swap them blindly for one-query decoding. Use a boolean/additive mask only for actual invalid positions and test all-masked situations as invalid inputs. `force_fused=True` is a diagnostic that may reject unsupported shapes; select the fastest correct backend rather than assuming forcing fusion is always best. [R35, R38]

The CPU gate reference and MLX port must agree before decoding comparisons. Probe contexts **larger than the selected budget**, or dense fallback can make a broken sparse path appear correct. Probe transitions at 63/64/65, 127/128/129 and a long prefix followed by more than one completed block. Preserve pre- and post-RoPE tensors explicitly as in Section 4.2. If padding or GQA cannot be implemented faithfully in the available fused kernel, use a correct gathered unfused path for quality; postpone the speed claim.

### 6.5 Calibration, timing, and adaptive rules

Measure stage costs separately: load/compile; tokenization; dense prefix; block features; each routed query's score/selection/gather/attention/other-layer work; and complete answer. Also time the original dense model with normal batched prompt ingestion. Serial sparse question-suffix processing may improve quality under a read policy while losing to batched dense ingestion in total latency.

Materialize outputs and updated state with `mx.eval`, synchronize before the timer and after the final evaluated work, and use `time.perf_counter_ns`. MLX's lazy graph-construction time is not execution time. Record warm-up until compilation/allocation and latency stabilize according to a pilot-frozen rule; save the stabilization trace. Randomize or alternate method order in paired time blocks on identical prompts. Include ordinary cache reuse that the stated workload permits, and separately report cold-load and steady-state timings. Do not silently give the treatment a reused prefix while rebuilding the baseline's prefix. [R37]

Pilot timing determines the number of repetitions needed for a stable interval, using time-block/request pairing and accounting for thermal drift. Record power source, low-power mode, concurrent load, sequence/generation lengths, package versions, and timing boundaries. Do not force a particular cooling wait; begin the next comparable block when the observed state is comparable. Report per-request distributions and paired ratios, not the fastest individual trial. Repetitions quantify measurement noise, not the number of independent quality examples.

Choose configuration through these gates:

| Observation | Action | Permitted interpretation |
|---|---|---|
| Comfortable measured headroom, no rising swap, correct fused execution | Advance from 4K to 8K, then 16K if the longer task adds information. | Local feasibility at measured settings only. |
| Peak approaches available memory or swapping rises | Reduce prefill chunk, remove diagnostic caches, serialize methods, then shorten context while preserving tasks. | Resource adaptation; record the new claim scope. |
| Base weights alone are unsuitable | Test common quantization with overlap parity, or use 0.6B for a component study. | Does not establish competitiveness against the unavailable 4B gate. |
| Gathering is correct but slower than dense | Continue the minimal quality comparison if its read-allocation question is sensitive; use profiling to bound a kernel opportunity. | No runtime efficiency claim. |
| Critical operator cannot reproduce the reference | Try a faithful MLX primitive composition or separately validated CPU/MPS reference; halt dependent speed tests. | Implementation/local-resource blocker, not a negative mechanism result. |
| Measured sample precision is unaffordable | Reduce nonessential matrix breadth before the primary sample; report remaining uncertainty. | Underpowered evidence, not equality. |

Estimate each run using measured cells rather than a guessed tokens/second number:

\[
\hat T=\sum_{m,f,n}E_{m,f,n}\,[t_{prefix}(n,m)+Q_{f,n}t_{suffix}(n,m)+A_{m,f,n}t_{gen}(n,m)]
+T_{warmup}+T_{analysis}.
\]

Here E counts **actual variant-by-method-seed requests**, Q is routed suffix length, and A is the **method-specific** measured answer-length distribution. Independent cluster count n is a different quantity used for precision. Add explicit request counts for timing repetitions, teacher-forced NLL, replay, replication, failures and reruns. A long run is justified when its expected precision resolves a live question and the calibrated laptop budget supports it; there is no arbitrary overnight or hour cutoff.

### 6.6 Checkpointing and recovery

The default checkpoint unit is one completed episode/method/seed. Write its result to a temporary file, flush, then atomically rename; append the indexed ledger only after validation. Resume by hashing immutable inputs and skipping only complete matching records. A partial request is rerun from its original prompt with its original method/RNG, which is cheaper and safer than maintaining a bespoke KV snapshot format.

Within-request resumption is optional. If implemented, save every layer's exact KV, raw and projected summaries, partial-block counters/positions, method state, current layer/query, generated tokens, RNG, precision, and software hashes. Reconstructing a sparse-generated history by dense replay of the same tokens is **not** equivalent because hidden states and caches can differ. A mismatch invalidates the partial result. On OOM, persist the failure record, release arrays, restart a fresh process, and change only a development configuration; do not silently rerun protected failures at a smaller context.

## 7. Ordered work packages and decision gates

All `python -m recaman_poc ...` commands below are **proposed project interfaces to implement**, not commands that currently exist. Files named here are outputs of the future coding project. Section 9 defines the minimum layout and schemas. Do not build a general framework before the early scientific gates pass.

### WP-00 — Discover prerequisites without large downloads

**Objective/risk:** determine whether the target and licensed assets support the selected frozen experiment. **Inputs/dependencies:** this plan, target laptop, network access to permitted public metadata; no prior project stage.

**Actions/files:** create `env/discover.py`, `env/probe_ops.py`, `configs/assets.yaml`, and `reports/prerequisites.json`. Run hardware discovery, isolated wheel resolution and exact asset/license inspection. Probe GQA, gather, sorting, BF16/FP32 length-128 normalized and unnormalized Hadamard transforms, and sign-match reductions against CPU references. Verify norm/dot preservation only for the normalized transform; SketchWalk's source scaling is different. Record revisions, telemetry and exact asset sizes.

**Tests/outputs:** machine JSON with bytes and versions; operator table with dtype/shape/mask/fused status/error; asset table with URL/revision/license/use status/bytes. Use tiny synthetic tensors first, then actual head dimensions. No model-quality claim is possible yet.

**Resource/gate:** seconds-to-minutes are plausible for discovery, but record actual time; package downloads and compilation vary. Continue only if a correct inference path and sufficient free disk/memory are identified. If a package version conflicts, resolve and relock a compatible version, repeating relevant probes. If the gate is unusable, retain graph and mathematical controls, and label the later 0.6B path reduced-scope; do not proceed to a claimed 4B learned-router comparison. SAS uncertainty does not block licensed Tier A, but limits strongest-baseline claims.

### WP-01 — Test the literal graph proposal and analytical invariants

**Objective/hypothesis:** H-GRAPH and the correctness of layer-ordered reachability; reject an unhelpful literal construction cheaply. **Inputs:** NumPy/Python CPU, Section 2 equations; independent of model assets. **Files:** `graphs/recaman.py`, `graphs/patterns.py`, `graphs/reachability.py`, `tests/test_graphs.py`, `configs/graphs.yaml`, `reports/graphs.json`.

**Precise construction:** require k>=2; include self/current and one preceding local block when it exists. Fix q=k-2, even at the first query. For zero-based layer ell and head h, use sequence indices `phase + (ell*H+h)*q + r`, r=0,...,q-1, with offset `a_index` and source `i-offset`. Zero, duplicate, future, and negative candidates are invalid. The target cardinality is `min(k,i+1)`; no construction can read more distinct causal blocks. Choose phase on development sizes only. The unfilled raw construction is a utilization diagnostic, not an equal-budget competitor.

For the repaired family, keep valid distinct proposed edges and fill to `min(k,i+1)` using a deterministic random permutation of remaining causal prefix IDs, seeded from `(global_seed,ell,h,i)`. Use the **identical fill routine and seed convention** for all families. Keep repair fraction and executed edge count as outcomes. Modulo and rescaling variants require separate registration. A tuned phase is an extra degree of freedom that comparators also receive.

**Comparators:** uniform sampling without replacement; exponentially dilated offsets with layer/head phase rotation; causalized butterfly toggles `j=i xor 2^r` retaining only j<i; and an expander-inspired random undirected regular graph subsequently oriented/clipped to the causal prefix. Name the latter a causal adaptation with no inherited expander theorem. Include a layer-coordinated greedy coverage baseline whose pool is **all causal nonmandatory IDs**, maximizing incremental correctly ordered ancestors with lower-ID ties; charge its broader structural search. Compare native masks and common repaired masks separately.

**Measurements:** executed edges, fill fraction, lag distribution, incoming-degree concentration, fraction of reachable legal source/query pairs after each depth, minimum layer depth to reach each legal pair, unreachable fraction, boundary/position strata, and mask-construction time/memory. Unreachable distances are not assigned a finite favorable average. Match lag distribution as an additional diagnostic to determine whether longer jumps alone explain a gain. Use Python integer bitsets at block scale; test exact brute-force enumeration at tiny sizes.

**Resource/calibration:** begin with small block counts to validate, then candidate powers-of-two and nearby off-grid sizes corresponding to 4K–64K token contexts at block 64/128. Bitset work depends on blocks/layers/heads; measure growth before broadening. Hold out lengths and phases before tuning.

**Gate/branches:** correct analytical invariants are mandatory. A Recamán family dominated in reachability, concentration, and construction cost at equal executed edges on held-out conditions is retired in this project. A nondominated family earns only a structural result, not an accuracy/speed claim. Its optional training test is WP-09. A wrong graph recurrence is an implementation failure requiring repair. The coverage decoder proceeds independently of H-GRAPH because its scientific mechanism differs.

### WP-02 — Specify controllers and verify identities on small tensors

**Objective/risk:** remove ambiguity in state, anchors, regret, budgets, and causality before integration. **Dependencies:** WP-00 numerical environment; Section 4/5. **Files:** `routing/reference.py`, `routing/state.py`, `routing/controllers.py`, `tests/test_routing.py`, `reports/controller_checks.json`.

**Actions:** implement the FP32 CPU selector, common wrapper, K-VEC/deficit/MMR/reuse/sham controls, token-accounting helper, and an independent summed-identity implementation. Implement stable ties and all-masked/empty/partial-block behavior explicitly. A group first computes its mask from old state; update after all groups, never midway through a head loop. Emit proposal and final-mask diagnostics.

**Tests/outputs:** enumerate tiny score/state cases and randomized inputs to verify cardinality, protected IDs, regret, bounded state, group permutation invariance, `lambda=0` and no-history equality, `identity_sum` equality, period-1 reuse equality, padding invisibility, and reset. A constructed high-relevance previously visited anchor must be selectable again. The method has no novelty benefit if a controller is algebraically identical; report that identity rather than inventing a new comparison.

**Resource/gate:** CPU-only fixtures; no learned weights. Continue only when identities hold exactly apart from declared floating arithmetic. Property failures block all downstream method comparisons. Numerical ties may change selected IDs under different precisions; document their rate and output significance before freezing a tolerance.

### WP-03 — Reproduce the frozen backbone and strongest accessible routers

**Objective/risk:** faithful inference, true sparse execution, baseline competence, and M3 feasibility. **Dependencies:** WP-00 licenses/operator gates and WP-02 support representation. **Files:** `models/qwen_adapter.py`, `routing/seer.py`, `routing/quest.py`, `routing/less_is_more.py`, `routing/sia.py`, `routing/sketchwalk.py`, `attention/reference.py`, `attention/mlx_gather.py`, `assets/manifest.json`, `tests/test_gate_parity.py`, `tests/test_model_parity.py`, `tests/test_index_walk.py`.

**Actions:** download and hash permitted assets. Optionally use 0.6B for first cache/model plumbing, but load the 4B base and actual gate for required comparisons. Implement exact gate tensor mapping, RoPE modes and mandatory accounting. Reproduce Quest page-16/per-head behavior and LessIsMore selection layers/union logic. Preserve the best unmodified dense MLX-LM path. Freeze base weights and assert gradients are disabled. Port common optimizations to all compatible methods.

Add SIA's independently authored decode selector and corrected-source SketchWalk from Section 5.1. Test sign orientation, absmax head pooling/ties, token versus block lowering, source/total forced support, transform scale/rounding, matrix/window lifecycle, first-two-dense cost, partial blocks and true non-dense masks. The uncorrected SketchWalk threshold must trigger the density-bypass fixture; its corrected exact-budget route must pass the count test. Chunked-prefix caches must equal the one-shot reference before quality evaluation.

**Tests/outputs:** dense full-support parity on logits and selected answer outputs; short CPU gate parity; beyond-budget sparse cache-transition fixtures; short dense-mask versus gathered attention; no future-token influence; cache state after a split prompt versus an equivalent token-by-token reference; native read accounting; model/gate configuration assertions. Inspect actual supports so a dense fallback cannot pass unnoticed. Native CUDA wall time cannot be reproduced locally; claim mathematical port fidelity only after these tests.

**Resource/gate:** roughly 8 GB base payload plus Section 6's conditional memory/storage budget; measured load, prefix and query times. Pass only with a defensible numerical envelope and no hidden CPU transfer in timed paths. If a source ambiguity changes outputs, retain named source-compatible and repaired variants in the baseline envelope. If only dense full-support parity passes, fix the sparse path before proceeding. A licensed but unportable strong baseline creates a comparison gap; do not substitute its published score as if measured locally.

### WP-04 — Build the evaluation population and demonstrate sensitivity

**Objective/risk:** a correct task with neither floor/ceiling nor leaked evidence; primary hypotheses must be expressible. **Dependencies:** WP-03 dense and sparse inference; generator/evaluator can be written earlier. **Files:** `data/generate_records.py`, `data/hotpot.py`, `eval/answers.py`, `tests/test_tasks.py`, `configs/populations.yaml`, `reports/sensitivity.json`.

**Actions:** implement Section 5 families, gold interpreter, split/cluster generator, official natural scoring, query-suffix and standard-decode splits, evidence-span tracking hidden from methods, and malformed-output handling. Before inspecting treatment results, calibrate task/length/budget using dense and unchanged baseline methods. Keep all calibration failures in the report.

**Development admission criteria:** before treatment inspection, record the development sampling rule, unchanged Seer/Quest configurations, endpoint and controls in `populations.yaml`. For each proposed core family/length, the lower confidence bound of the **paired dense-minus-no-evidence accuracy difference** must exceed zero; necessary-evidence deletion must similarly show a resolved paired degradation. These are adaptively calibrated development diagnostics, not nominal 95% error-controlled discoveries. Retain the accuracy primary only when an unchanged-sparse-vs-dense accuracy gap or another documented accuracy sensitivity makes that endpoint resolvable at feasible sample size. If only NLL/output error is sensitive, freeze a separately labelled component study and exclude that regime from the accuracy-primary population before protection. Attention error is not accuracy sensitivity. A zero gap with a wide interval is insufficient, as is an apparent floor with few examples; use pilot precision to distinguish them.

**Tests/outputs:** gold underdetermination fixtures; task-format and scorer fixtures; equivalent-method null; deliberate wrong/local-only mask loses on remote tasks when the task is sensitive; outputs change when known required evidence changes; protected membership hashes; dense and sparse accuracy/NLL by family and length. Do not select individual confirmatory examples by any method's success.

**Resource/gate:** begin at 4K with the smallest number of development clusters needed to locate a sensitive regime, then expand using measured variance. If dense cannot solve a task, repair formatting or simplify required steps during development and narrow the claim. If sparse is already at dense performance, tighten its budget or use a longer justified context before adding sophisticated coverage; retain the original regime as a negative boundary. If no relevant sensitive regime fits locally, report an unsuitable local proxy rather than falsifying the general mechanism. Natural data remains intact even when it is insensitive; such a result limits transfer.

### WP-05 — Integrate and calibrate the minimal treatment

**Objective/hypotheses:** H-COVER/H-STATE/H-REVISIT with measured overhead. **Dependencies:** WP-02/03/04 gates. **Files:** `routing/bounded_coverage.py`, `experiments/pilot.py`, `configs/pilot.yaml`, `reports/pilot.json`, `reports/tuning_ledger.jsonl`.

**Actions:** connect the selected gate to the fractional state and common wrapper. Run cached query/layer replay for cheap parameter screening, then closed-loop development inference. Tune required controls with comparable evaluation expenditure, include native defaults, and select a baseline champion before protection. Measure read counts, route changes, repairs, retained mass/output error, answer accuracy, and full timing. Compare candidate against same-wrapper K-VEC/deficit/MMR, reuse, and sham history. Record all tried configurations and reasons for rejection.

**Tests/outputs:** integrated `lambda=0` reproduces baseline; candidate state changes route support in a nontrivial subset of appropriate pilot cases; actual token budget and first-answer boundary are verified. If routing never changes, determine whether coverage is flat, anchors consume all slots, the constraint always repairs, or top candidates dominate. Do not add claim-changing complexity just to force a difference.

**Resource/gate:** use measured cell costs to size the development matrix and proposed confirmatory sample. Continue to confirmation when a valid and sensitive question remains, including when development estimates are negative or uncertain. A clearly inferior candidate already explained by close controls may be stopped with a development-only negative report, without pretending confirmatory certainty. Optional kernel work needs a surviving quality/read tradeoff and a profile showing addressable overhead, not simply enthusiasm for sparsity.

### WP-06 — Freeze and execute the controlled comparison

**Objective:** obtain nonadaptive evidence for the primary claim and declared secondary family. **Dependencies:** all correctness, sensitivity, baseline, precision and resource gates; immutable manifest. **Files:** `experiments/freeze.py`, `experiments/run.py`, `eval/statistics.py`, `configs/confirmatory.yaml`, `confirmatory_manifest.json`, `results/<run_id>/`.

**Actions:** choose one fixed sample size using pilot cluster simulations; finalize family/length weights, per-token cost envelope, candidate/champion, source conventions, seeds, extraction, CI fallback, claim tree, failures and timing blocks. Hash and commit before viewing protected outcomes. Run paired complete episodes with atomic per-episode checkpoints. Record all failures and missing cells; never drop a method's hard examples. Analyze exactly the frozen primary contrast and secondary family.

**Tests/outputs:** manifest/code/data hash match; every expected cell has success or a classified failure; complete paired analysis table in accuracy percentage points, request milliseconds, reads, memory bytes, intervals and adjusted decisions. Verify the frozen report can be regenerated from raw immutable records. Interrupted partial requests restart faithfully as in Section 6.6.

**Resource/gate:** total runtime and disk budget come from WP-05's measured model; no fixed arbitrary run count. A technical bug requires a recorded amendment and re-execution/fresh protected data when scientific choices were exposed. A memory failure at a protected setting is an observed feasibility outcome, not permission to shrink that setting unnoticed. Incomplete or underpowered comparisons are reported as such. Proceed to attribution/reproduction without changing the protected winner.

### WP-07 — Attribute, stress, and independently confirm

**Objective:** distinguish actual history value from safeguards, perturbation, tuning, and favorable task structure. **Dependencies:** WP-05 selects meaningful ablations; WP-06 supplies primary evidence. **Files:** `experiments/ablations.py`, `configs/robustness.yaml`, `reports/attribution.json`, `results/replication/`.

**Actions:** execute the frozen necessary controls in Section 8. Add new exploratory stress tests only with that label; a claim-changing discovery receives a new manifest and fresh independent data. Use a second synthetic template namespace and required natural evaluation. Promote a cross-query persistence or long-generation extension only with its own protocol and comparators.

**Tests/outputs:** effect-by-condition table; true/sham and wrapped-controller contrasts; family regressions; source-mode and block-boundary sensitivity; numerical precision overlap; new-data confirmation; explicit explanation of which causal hypotheses survived. Rerun a small, prespecified sample from a clean environment and reproduce its masks/logits/results within the established envelope.

**Resource/gate:** prioritize ablations capable of changing interpretation; calibrate extra runs with measured cell costs. A candidate tied with a wrapped known method yields an empirical/engineering result, not distinctive-state novelty. Synthetic-only evidence stays synthetic. A standard-decode null with query-suffix success limits the claim to the latter. Inconsistency widens uncertainty or narrows scope; it is not repaired by selecting a favorite seed.

### WP-08 — Synthesize and release the reproducible result

**Objective:** an auditable positive, negative, or inconclusive result. **Dependencies:** complete records and classified gaps. **Files:** `reports/final_report.md`, `reports/result_tables.csv`, `reports/figures/`, `reproduce.md`, final dependency/asset manifests.

**Actions:** regenerate tables/figures from immutable records; separate source facts, measurements, derivations and unverified extrapolations. Report the whole baseline envelope and failures, not only the best candidate curve. Explain absolute effects, precision, actual request costs, guardrails, and fidelity limits. Select a Section 10 outcome with evidence. Keep licensed model/data downloads external to the code release; provide exact fetch manifests and required notices. Do not redistribute assets without their actual permission.

**Tests/outputs:** one documented command reconstructs the report from recorded results; a clean environment can recreate a sample; every promoted claim has an appropriate comparison and analysis. Record an audit and amendment log. Stop optional testing when remaining questions cannot change the supported claim without a new study.

### WP-09 — Conditional architectural escalation

**Objective:** test representation learning or larger-scale effects only when prior evidence identifies what the frozen intervention could not express. **Entry:** a held-out graph family is structurally promising, or a coverage allocation effect is credible but pretrained representations/short suffix make the broader hypothesis unresolved. A vague possibility of scaling benefit is insufficient.

**Actions/files:** create a separate `extensions/trained_model/` proposal before code. Use an openly specified small causal transformer trained from scratch on integer associative recall/pointer tasks, common tokenizer/vocabulary, exact layers/width/context, and matched parameter/training-token/attention budgets. Begin with measured CPU/MLX gradient/operator probes; if sparse backward is unavailable, dense-mask training can test quality at matched mathematical masks but cannot establish sparse-training speed. Compare dense, best graph alternatives, top-k/content routing, and close coverage/reuse rules with comparable optimization tuning. Full-batch masks remain causal. Use independent training seeds and held-out task instances, treating seed and instance variation explicitly.

**Gate/resources/outcomes:** first measure steps/second, activation/optimizer peak, learning curves, and projected tokens-to-competence; commit a concrete architecture/sample/power budget from that calibration. This extension is not a hidden prerequisite of the core PoC. If capacity/sequence length cannot express the effect, report a component limitation. If matched alternatives learn as well, no new topology advantage is supported. Remote training, new large learned routers, and claims about production-scale models remain outside the authorized local core.

## 8. Ablation, robustness, and failure analysis

### 8.1 Minimal discriminating matrix

The table specifies questions, not a requirement to run an enormous Cartesian product. Choose the confirmatory subset using development evidence and the frozen claim tree. Controls essential to the primary mechanism cannot be omitted merely because they perform well.

| Contrast or intervention | Hold fixed | Interpretation of a null or adverse result |
|---|---|---|
| Candidate vs same-scorer top-k and `identity_sum` | Scores, support budget, source mode, numerical path | Top-k tie means no demonstrated allocation gain; identity mismatch is a bug or a different rule. |
| Candidate vs same-wrapper K-VEC/deficit/MMR | Anchor IDs/count, regret threshold, candidate universe | A tie removes evidence for a distinctive history controller even if both beat raw top-k. |
| Real vs permuted history and constrained random | Penalty strength, support cost, safeguards, paired seeds | Equivalent gains support generic perturbation/position diversity rather than useful actual history. |
| Fractional vs any-group vs per-group service | Scorer, memory horizon, cost envelope | Detect pooling saturation and harmful suppression across specialized heads. Per-group state may be a useful named refinement. |
| Gamma 0, finite gamma, gamma 1; explicit history off | Other controller terms | Separates preceding-layer service, decayed history, cumulative history and true no-op. Gamma 0 is not “coverage disabled.” |
| Anchors on/off; regret bound on/off | Proposal scores and executed budget | Identifies safeguards as the source of benefit; a zero-slot exploration path cannot support an exploration explanation. |
| Standardized scores vs temperature-calibrated probability penalties | Same histories and wrapper | Score scale can explain a state-rule advantage. Calibrate temperatures on development data, including close baselines. |
| Reuse period 1, periodic reuse, loss-calibrated reuse | Same learned scorer and per-token support | Determines whether coordination is better achieved by repeating reads or reducing router work. |
| Distributed facts vs same-block and anchor-return pairs | Answer arity, latent world, distractor load | A gain purchased by loss on concentrated/reused evidence is a task tradeoff, not an unconditional improvement. |
| Query-suffix vs standard-decode boundary | Full token array, generation policy, populations | Success only during sparse question processing is limited to that intervention. |
| Native vs total-budget gate conventions; source-compatible vs consistent RoPE | Exact asset and precision | A gain against only one flawed convention is a reproduction issue, not robust contribution. |
| Best implementation with shared optimizations vs initial path | Correctness envelope and workload | Gain lost after fair optimization is not an algorithmic runtime advantage. |

For the per-group state, maintain independent `v[g,j]` with `e[g,j]=1` only when group g selected the optional block, using the same z, decay and reset. For any-group state, replace fractional e with a binary union. No head-order dependence is introduced. Normalize history MMR's scope separately: it uses a maximum over its set, so it is not equivalent to additive frequency. Report state dimensions and update cost.

### 8.2 Distribution and boundary conditions

Vary relevant-evidence position, block alignment, answer length, distractor density, duplicated but irrelevant passages, lexically similar contradictory facts, unique facts sharing an address, and different query phrasing. Include non-power-of-two context lengths and partial final blocks. Data variants inherit their latent-world cluster. Position shifts must not accidentally alter the logical answer or truncate evidence.

Stress multi-group disagreement, all-equal/near-equal scores, one dominant repeatedly relevant anchor, a changing decisive block across layers, budget smaller than mandatory support, full budget, zero valid optional blocks, and query reset immediately after a block completion. The low-budget-invalid case must produce a declared invalid configuration rather than a hidden extra read. A state memory inherited from a different request is a correctness failure.

For natural data, present unmodified-source and derived-length conditions separately. Use paragraph-order permutations only as labelled variants; a new order can change model difficulty without changing gold truth. If score improvements appear only on generated layouts with many equal-quality candidates, call that the supported operating regime. If a longer context merely increases distractors without adding useful scale-dependent routing behavior, it does not justify a general scaling claim.

### 8.3 Diagnostics without oracle leakage

On a bounded development replay subset, compute full attention probabilities and outputs from the same captured q/K/V to measure mass, output error, proxy-score agreement, and selected evidence spans. This is a local component comparison. On real inference, rerouting changes later hidden states and caches; replay cannot supply the primary answer result.

Gold supporting blocks never enter a real selector. A **privileged support-selection diagnostic** exposes them only in a separate run with reads counted. Its success shows that particular support can suffice; failure does not prove a budget insufficient. Other contextualized KV may carry useful information, and a different mask or normalization can perform better. This is not an optimized oracle upper bound. For a causal neural-revisit claim, add a matched-mask intervention removing previously selected addresses at prespecified layers, with matched random exclusions; distinguish fixed replay from downstream rerouting. Otherwise report task-level return accuracy and observed revisits without claiming causation.

### 8.4 Failure triage

| Symptom | First checks | Classification and next action |
|---|---|---|
| Full-budget outputs differ | Q/K normalization, RoPE offsets, masks, GQA grouping, tokenizer, cache layout | Implementation error; halt comparisons. |
| Controller never changes support | Coverage dispersion, k/anchor slots, lambda, tie policy, repair rate | Inactive mechanism or unsuitable regime; diagnose during development. |
| More unique blocks but worse answers | Same-block guardrail, value error, required evidence, near-cutoff replacements | Positional coverage is a poor proxy here; reduce claim or test a named value/relevance refinement. |
| Improvement vanishes under shared wrapper | Raw and final masks, normalization, baseline tuning | Safeguard/adaptation finding, not a distinctive state rule. |
| Candidate wins against weak baseline only | Seer/Quest/reuse defaults and tuning trajectory | No competitiveness claim; use strongest credible envelope. |
| Faster attention but slower request | Routing/gather, dense prefix, MLP/weight reads, completion length | Component improvement only, unless a measured removable bottleneck justifies kernel work. |
| Natural transfer null | Dense competence, source length, standard-decode sensitivity, contamination | Synthetic result retained with scope limits; no silent benchmark replacement. |
| Tiny effect, wide or unstable interval | Discordant clusters, rare-event CI, sample size, timing drift | Inconclusive; report detectable-effect frontier rather than calling a breakthrough. |
| Aggregate gain hides return-family loss | Frozen per-family intervals and required-use guardrails | Report an uncompensated tradeoff; “preserved” requires a declared margin. |
| Resume produces different results | Manifest, partial request restart, RNG, layer KV/state restoration | Invalid recovery; quarantine and rerun matching records. |

## 9. Reproducibility, logging, and reporting

### 9.1 Minimum repository

Use ordinary modules with explicit data flow; no service deployment is needed. The repository is created by the coding agent and is not contained in this research-plan deliverable.

```text
recaman-poc/
  pyproject.toml
  requirements.lock
  README.md
  reproduce.md
  src/recaman_poc/
    env/                 # discovery, operators, timing and memory telemetry
    graphs/              # Recaman and matched graph constructions
    models/              # Qwen adapter and exact prompt/token boundary
    attention/           # trusted references and MLX gather execution
    routing/             # scorers, controllers, state, common wrapper
    data/                # record generator and official-data loaders
    eval/                # answer scoring, clusters, statistics, reporting
    experiments/         # pilot, freeze, run, ablations, report CLI
  configs/
    assets.yaml
    graphs.yaml
    populations.yaml
    pilot.yaml
    confirmatory.yaml
    robustness.yaml
  assets/manifest.json
  data/manifests/         # IDs, hashes, split/cluster mapping; no model access
  tests/
  results/<run_id>/       # immutable config, records, failures and telemetry
  reports/               # generated tables, figures, audit and final prose
```

### 9.2 Configuration contract

Reject unknown configuration keys. Store fully resolved values in each run, including inherited defaults. The following is an **illustrative development configuration**, not the frozen winning parameters or final sample allocation:

```yaml
schema_version: 1
phase: development
model:
  repo: Qwen/Qwen3-4B
  revision: REQUIRED_RESOLVED_COMMIT
  weights_dtype: bfloat16
  kv_dtype: bfloat16
  quantization: null
  thinking: false
router:
  scorer: seer
  gate_revision: REQUIRED_RESOLVED_COMMIT
  block_tokens: 64
  position_mode: source_compatible
  mandatory_policy: first_previous_current_total_budget
  dense_layers: []
  controller: bounded_coverage
  history_scope: per_query_across_layers
  event: fraction_of_kv_groups
  lambda: 0.5
  gamma: 0.9
  epsilon: 0.1
  protected_fraction: 0.25
  anchors_enabled: true
  regret_enabled: true
execution:
  backend: mlx_gather
  batch_size: 1
  prefill_chunk_tokens: 512
  prompt_boundary: question_suffix
  diagnostic_attention_dump: false
evaluation:
  token_budget: 1024
  context_strata: [4096, 8192]
  independent_unit: latent_world_cluster
  generation: greedy
  max_new_tokens: REQUIRED_FROM_DEVELOPMENT_ANSWER_DISTRIBUTION
  extraction_version: REQUIRED_VERSION
  sample_size: REQUIRED_FROM_PILOT_PRECISION_AND_RUNTIME
```

The `REQUIRED_...` values are template placeholders that block actual execution; each has a procedure: asset revisions from WP-00/03, caps from WP-04 and confirmatory sample size from WP-05/06. Validation is **phase-specific**. A development run instead requires a finite `development_batch_clusters` and a provisional cap derived from the permitted answer format/gold-token lengths; it does not require a future confirmatory sample size. A small plumbing batch is not a scientific sample limit. Expand development batches and refine the cap using observed sensitivity/precision/cap-hit diagnostics, then freeze the distinct confirmatory fields. No circular dependence on a completed pilot is permitted to start that pilot.

### 9.3 Minimal experiment record

Each episode/method/seed record is valid UTF-8 JSON with at least:

```json
{
  "schema_version": 1,
  "run_id": "content-derived-id",
  "manifest_sha256": "...",
  "code_commit": "...",
  "environment_sha256": "...",
  "asset_manifest_sha256": "...",
  "split": "development_or_confirmatory",
  "cluster_id": "latent-world-or-source-cluster",
  "episode_id": "unique-variant-id",
  "family": "independent_facts",
  "method_id": "bounded_coverage",
  "method_config_sha256": "...",
  "method_seed": 0,
  "prompt_token_sha256": "...",
  "prefix_tokens": 0,
  "suffix_tokens": 0,
  "generated_tokens": 0,
  "answer_exact": null,
  "answer_f1": null,
  "answer_nll_bits_sum": null,
  "answer_nll_token_count": null,
  "query_token_pairs": 0,
  "unique_kv_tokens_by_group_sum": 0,
  "summary_score_operations": 0,
  "prefix_ms": null,
  "suffix_ms": null,
  "first_answer_ms": null,
  "generation_ms": null,
  "request_ms": null,
  "mlx_peak_bytes": null,
  "rss_peak_bytes": null,
  "swap_delta_bytes": null,
  "status": "pending_success_or_classified_failure",
  "failure_stage": null
}
```

Zero/null entries above are schema examples, not observations. Completed records add output token IDs/text, parsing status, all actual counts, timing-block ID, read-budget violations, and optional aggregated routing diagnostics. Store raw answer strings securely with provenance; public synthetic fixtures may be redistributed, while natural-data licensing controls publication. Avoid full attention dumps and duplicating private environment details.

Use two cost ledgers: mathematical operations/support and measured execution. Name the **primary common resource unit** as query–token attention pairs summed over all query heads/layers per routed token. A common GQA block budget implies equal pairs when head counts/length are fixed; per-head Quest also matches this unit, but can have different unique KV gathers. Record those gathers and summary scoring separately. Dense selection layers count all their pairs and any extra exact scoring pass; a fused computation reused for both score selection and output is counted once, a separate recomputation twice. Native methods retain their justified minimum cost and do not become artificially cheap through this accounting.

For natural source dependence, connect each question ID to both a canonical document/title node and a paragraph-content-hash node for every supplied context. Shared title **or** shared paragraph joins components. Assign connected components to source clusters and report their sizes. Do not split a component just to obtain more “samples.” If connectedness leaves too few independent components for the pilot-validated inference/precision procedure, make no source-generalization significance claim: report the admitted benchmark descriptively, or preregister a distinct finite-benchmark sampling estimand before results. Screen train/development document/paragraph overlap and report it; pretrained-weight contamination remains unknown.

### 9.4 Failures and aggregation

Parsing failures, cap hits, and method-caused numerical failures count as incorrect answers in the quality denominator and are reported by type. A baseline failure cannot be removed to improve the candidate's comparison. Infrastructure failure before a valid model request may be rerun identically; retain its failure ledger. If a method is incapable of a complete declared population owing to memory/runtime, report its coverage and feasibility; do not claim a controlled quality win from a favorable complete-case subset. Use paired lower/upper sensitivity bounds for genuinely missing outcomes, assigning missing paired scores their worst/best admissible values; if the conclusion changes, the comparison is incomplete.

Answer NLL uses identical gold answer tokens under identical teacher-forced prefixes, excluding prompt tokens and any undeclared special-token suffix. Sum negative log probabilities in base two, divide by answer-token count per example, then apply synthetic variant/world/family macro weights or natural question-mean weights as appropriate. Also provide token-weighted corpus NLL descriptively; it answers a different question. An empty **gold** answer or zero gold-token count is a dataset/formatter error, not a zero-loss sample. An empty model output remains an incorrect-generation outcome. NLL and free generation are separate runs/costs.

The primary inference concerns the **frozen development-selected champion**. Display every required competitor's protected performance, but do not claim superiority to all from that one test. A claim that the candidate beats the whole required baseline set needs a predeclared family of paired contrasts with simultaneous/multiplicity control and adequate power. Likewise, absence of a significant guardrail regression is not proof of preserved performance: provide its interval and a justified noninferiority margin, or use descriptive language.

### 9.5 Reporting and reproducibility commands

Implement and document these project CLI operations with fail-closed validation:

```bash
python -m recaman_poc discover --out reports/prerequisites.json
python -m recaman_poc graph --config configs/graphs.yaml
python -m recaman_poc verify --suite core
python -m recaman_poc pilot --config configs/pilot.yaml
python -m recaman_poc freeze --config configs/confirmatory.yaml
python -m recaman_poc run --manifest confirmatory_manifest.json --resume
python -m recaman_poc report --manifest confirmatory_manifest.json
```

Produce a main results table with population, model/assets, precision, prompt boundary, method/fidelity variant, actual resource policy, n independent clusters, accuracy/F1 effect and interval, request latency and interval, peak memory, and classified failures. Draw quality versus actual per-token reads and quality versus request latency, plus family effects, coverage/repair diagnostics, and calibrated precision-versus-runtime. Use ordinary plotting libraries and save standalone SVG/PDF/PNG as appropriate; include raw tabular data and code. Do not use a theoretical pair-reduction factor as an observed speedup label.

Record exact package/source commits and reproducible fetch commands. Hash every transformation and frozen split. Reproduction means masks/state/outputs agree within the declared numerical envelope, statistical analysis regenerates exactly from records, and timing conclusions persist within system variability; it does not require identical nanoseconds. Keep technical failures and unfavorable runs visible. A report with no measured quality advantage is a complete outcome when the experiment was valid and informative.

## 10. Outcome interpretation and evidence-triggered continuation

The original ambition is efficient structured exploration. The selected experiment tests history-dependent allocation while allowing revisits, replacing the literal sequence with an explicitly tested controller. That is a disclosed refinement, not evidence that Recamán itself caused any improvement.

| Observed outcome | Supported conclusion | Next action |
|---|---|---|
| Literal graph is dominated after matching edges, lags and repairs | This tested Recamán mapping has no demonstrated structural advantage. | Retire it. Continue coverage only on its separate hypothesis; do not rescue the sequence through endless post-hoc mappings. |
| Graph is nondominated but no trained task evaluated | A structural opportunity exists at the tested sizes. | A WP-09 trained component study may be justified; no language-quality or runtime claim yet. |
| Candidate beats the frozen champion with a resolved paired effect and acceptable disclosed guardrails | Better accuracy at that fixed per-token read policy on that population. | Confirm on fresh data, report full request cost, and test whether close wrapped controls explain it. |
| Candidate ties a same-wrapper known controller | A useful adaptation may exist; the distinctive state rule is unsupported. | Prefer the simpler/faster existing rule and publish the comparative finding. |
| More coverage or retained mass, no answer gain | A diagnostic changes; the downstream hypothesis is unconfirmed. | Examine sensitivity and values. If accuracy was well powered and active, stop that proposed accuracy claim. |
| Only NLL/component metrics are sensitive | A lower-level approximation/allocation question can be studied. | Freeze a component-specific endpoint before confirmation; do not relabel it as improved reasoning. |
| Synthetic gain, natural or standard-decode null | Benefit is restricted to the demonstrated task/intervention. | Report the scope explicitly. A new deployment claim requires a new sensitive natural/standard-decode study. |
| Accuracy improves but total latency or memory worsens | A measured resource/quality tradeoff, possibly useful for a particular constraint. | Quantify its value; consider kernel work only if profiling identifies a removable cost. |
| Runtime improves at comparable quality with uncertainty and an appropriate margin | Local end-to-end efficiency improvement at the measured workload. | Repeat under comparable thermal conditions and another length/task; separate engineering from controller value. |
| Robust negative accuracy effect on an active, sensitive, powered study | Evidence against this frozen controller in that regime. | Stop it or propose a narrowly motivated new mechanism; do not invoke hypothetical scale as an unlimited escape. |
| Wide interval, too few independent clusters, or technical failures | Inconclusive or invalid comparison, according to the recorded cause. | Report resolution/resource frontier or repair correctness; do not call a tie a proof of equivalence. |
| Strongest relevant asset is unlicensed, unavailable or unportable | A comparison boundary remains. | Complete valid local work and restrict the competitiveness claim; never insert incomparable published numbers as a substitute. |

### 10.1 When a larger or different experiment is warranted

Increase context only when the measured regime suggests a scale-dependent opportunity—such as more distant competing evidence, a growing attention share, or increased distinct-block redundancy—and the calibration predicts enough statistical resolution. A flat state at 4K does not falsify an effect at 64K, but a larger experiment needs a causal prediction about how the state/quality gap should change. The local core can report this prediction without pretending to validate it.

Training is warranted when evidence indicates the pretrained decoder cannot use the redistributed reads but a controlled learner could: for example, a structurally superior graph combined with a task whose solution is reachable only over multiple sparse layers. Match optimization and training resources as in WP-09. Do not use extra training only for the candidate and attribute the result to its topology.

A memory-search pivot requires a task in which an intermediate read changes the next retrieval query and repeated retrieval demonstrably wastes budget. Specify the external index, per-read observations, stopping rule and full index-building cost. Compare against query reformulation plus ordinary retrieval, MMR and an established learned memory controller at matched reads. DNC/RAT+ and recent predictive indexers delimit this space; a visit ledger does not supply a semantic index. [R30, R31]

A value- or uncertainty-aware allocation pivot is justified if diagnostics show that missed high-impact values or ambiguous relevance cutoffs explain failure. Match the extra summary/scoring work and variable average reads. A content-memory change, a better relevance scorer, and an executed-history selector are different mechanisms; isolate them before claiming a combination advantage. vAttention, value-aware approximation and uncertainty-gated selection are relevant predecessors. [R27, R28, R52]

Structured channel mixing and low-discrepancy approximation remain legitimate alternatives if an actual linear-layer/kernel approximation bottleneck emerges. Their current prior art means they need their own discriminating component tests rather than being offered as unexamined novelty substitutes. [R13–R15]

## 11. Audit record, unresolved gates, and limits

### 11.1 Audit method and important revisions

This plan received distinct **scientific/prior-art**, **experimental-validity**, and **implementation/feasibility** challenge passes by assistant subagents, followed by revision and targeted rechecks. These are separate analytical perspectives within the same assistant workflow, not external human peer review. The record below concerns defects in draft specifications; it does not claim the future M3 experiment passed.

| Draft weakness identified | Revision incorporated | Acceptance/recheck target |
|---|---|---|
| Coverage/relevance protection treated as potentially broad novelty | K-VEC/VASC and older coverage/diversity explicitly bound novelty; known-controller adapters required. | A positive claim must survive same-wrapper close controls; no Recamán causal attribution. |
| Any-group coverage could flatten state or punish specialization | Fractional group service primary; any-group/per-group alternatives; dispersion/no-op diagnostics. | Bounded-state derivation and 400 update checks passed; usefulness remains experimental. |
| K-VEC denominator, VASC optional normalization/dense transitions, safeguards ambiguous | Explicit source-form versus common-state variants; fixed pairs; final-mask updates; matched wrappers. | Controller identity/transition fixtures in WP-02; no accidental weakened baseline. |
| Reuse was discussed but insufficiently controlled | Added LessIsMore and same-scorer periodic/loss-calibrated reuse. | Count dense/refresh work and compare the opposing allocation policy. |
| Score history and executed-service history were insufficiently separated | Added same-wrapper score-EMA with all-block score updates and a separately calibrated signed-state scale. | First-query/lambda-zero identities and actual-service-versus-score-history contrast in WP-02/07. |
| Later citation tracing exposed stronger local scorers and a released density bypass | Added SIA and corrected-source SketchWalk; specified transforms, native support units, early dense layers, persistent caches and fixed-budget correction. | SIA sign/GQA fixtures and SketchWalk density/window/matrix fixtures; source-shape bypass reproduced in NumPy. |
| One-token answers could bypass sparse attention | Route question suffix; standard-decode companion routes final prompt token; serialize/tokenize once. | Token-array equality and first-answer route trace. |
| Nominal sparse budget hid mandatory blocks and different output lengths | Mandatory support counted; attention-pair resource explicit; per-token policy separated from request cost. | Exact read ledger and method-specific complete-request timing. |
| Wrong independent units, natural weighting, rare-event intervals | Joint latent-world clusters; question-mean natural endpoint; source clusters for uncertainty; nondegenerate CI fallback. | Pilot simulation and frozen statistical manifest; no artificial sample inflation. |
| A component-sensitive task could slip into an accuracy study | Separate admission and claim branches before treatment inspection. | Accuracy endpoint must be resolvable at feasible sample size. |
| Absolute block IDs confused with compact array positions | Executable aligned-index CPU reference, stable ties and summation. | 1,000 extracted-reference randomized cases passed in Linux; Metal parity pending. |
| Source gate pre/post-RoPE tensors and partial summaries ambiguous | Explicit dual representations, source-position modes, raw-accumulator/partial-placeholder tests. | CPU gate/MLX transition parity beyond budget and block boundaries. |
| Shared compressed-cache counter can corrupt nonaligned prefixes | Canonical position-derived writes; raw source bug excluded from competition; chunked prefix accumulation specified. | Small control-flow counterexample reproduced; full target-model correction gate remains. |
| Graph boundaries and “support oracle upper bound” overclaimed | Early-prefix cardinality/pool specified; privileged support-mask result is only a diagnostic. | Tiny graph brute force and scoped interpretation. |
| Partial resume could silently change model state | Atomic episode checkpoints and full rerun; exact full-state restoration only if implemented and tested. | Deliberate interruption/restart fixture. |

Following revision, the scientific recheck found no additional substantive correction in the mechanism, comparison scope and outcome interpretation. The experimental recheck closed its remaining execution-count, paired-admission and empty-answer issues. The implementation recheck closed the selector/interface/cache, phase-specific configuration, SIA and SketchWalk issues; the final source-config defaults are explicitly pinned below. No further substantive correction was identified within those reviewed scopes. This is closure of identified specification defects, not a claim of perfect completeness or external independent validation. It cannot resolve unknown empirical effectiveness, asset changes after the research date, or actual laptop capacity.

### 11.2 Explicit unresolved prerequisites

- The laptop's RAM, free storage, OS, power state and exact MLX operator path are unknown. WP-00 blocks dependent work until measured.
- The full learned-gate/base port, corrected compressed cache and native comparator supports have not been executed on M3. WP-03 blocks quality/runtime comparisons until parity.
- Data availability, precise hashes and all asset permissions must be checked at download time. SAS licensing remains a particular comparison gap.
- A sensitive task/budget population and affordable confirmatory precision have not been measured. WP-04/05 determine whether the accuracy study is valid locally.
- The broad idea is known; narrow novelty is unestablished. Even a positive local result may be best described as replication, adaptation, engineering or a new empirical finding.
- Synthetic necessity, graph connectivity, coverage, attention mass and replay errors are different evidence. None alone demonstrates improved reasoning or end-to-end efficiency.
- The full KV cache is retained. This project does not validate cache-capacity savings, training efficiency, production serving throughput, million-token inference, or other hardware.

These are executable gates and claim limits, not reasons to skip the already specified inexpensive checks. The study is complete when it answers its calibrated question with an honest uncertainty/resource account, including a negative or inconclusive result.

## 12. References and search-coverage record

### 12.1 How to read the evidence record

References are primary papers, author repositories, official model/data records, documentation, and patent records unless explicitly labelled metadata-only. “Inspected” means source reading during preparation, including delegated source inspection; it never means reproduced benchmark results. Repository branches and documentation are mutable. Pin the indicated versions/commits or resolve immutable replacements in WP-00, and record any changed behavior. The complete core mechanism is specified above; the bibliography supplies provenance rather than delegating its central design.

### 12.2 Bibliography

**[R1] OEIS Foundation, “A005132: Recamán's sequence.”** Living reference, accessed 17 October 2026. [https://oeis.org/A005132](https://oeis.org/A005132). Inspected the definition and listed terms, including the repeated value 42. The sequence's subtraction test avoids previously visited values only when subtraction succeeds; addition can revisit. The entry supplies a mathematical definition, not evidence for neural efficiency, low discrepancy, or useful causal connectivity.

**[R2] Rewon Child et al., “Generating Long Sequences with Sparse Transformers.”** 23 April 2019. [https://arxiv.org/abs/1904.10509](https://arxiv.org/abs/1904.10509). Inspected the paper's factorized patterns and information-path discussion, plus the official announcement. Local, strided, and fixed sparse constructions establish coordinated sparse connectivity as prior art. Their training results do not establish that a Recamán-derived mask preserves quality or runs efficiently on Apple hardware.

**[R3] Iz Beltagy, Matthew E. Peters, and Arman Cohan, “Longformer: The Long-Document Transformer.”** 10 April 2020; revised 2 December 2020. [https://arxiv.org/html/2004.05150v2](https://arxiv.org/html/2004.05150v2). Inspected local, dilated, and global attention mechanisms, including variation across heads and layers. This is direct precedent for combining nearby and distant access. The main bidirectional architecture differs from a strictly causal, frozen-decoder adaptation.

**[R4] Manzil Zaheer et al., “Big Bird: Transformers for Longer Sequences.”** 28 July 2020; NeurIPS 2020. [https://arxiv.org/abs/2007.14062](https://arxiv.org/abs/2007.14062). Inspected the abstract and official repository, not a complete theorem-level reproduction. Local, random, and global connections are established structural controls. Published expressivity guarantees have specific assumptions; they cannot automatically be transferred to causal clipping, altered global-token policies, or an irregular deterministic mask.

**[R5] Hamed Shirzad et al., “Exphormer: Sparse Transformers for Graphs.”** ICML 2023. [https://proceedings.mlr.press/v202/shirzad23a.html](https://proceedings.mlr.press/v202/shirzad23a.html). Inspected Section 4 and Corollary 4.4 on interaction depth, alongside the graph-edge, expander, and virtual-node construction. Its undirected regular-expander reasoning motivates a structural comparator. Orienting or removing edges to enforce causality changes the premises, so undirected diameter is insufficient evidence for layered causal information flow.

**[R6] Enzhe Lu et al., “MoBA: Mixture of Block Attention for Long-Context LLMs.”** 18 February 2025. [https://arxiv.org/html/2502.13189v1](https://arxiv.org/html/2502.13189v1). Inspected the method and official repository instructions. Queries route using mean-key block summaries, then attend within selected blocks with causal handling. A frozen-model mean-key scorer is a useful mathematical control; it does not reproduce MoBA's trained conversion, quality results, or native kernels.

**[R7] Jingyang Yuan et al., “Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention.”** 16 February 2025. [https://arxiv.org/html/2502.11089v1](https://arxiv.org/html/2502.11089v1). Inspected token compression, fine selection, local-window branches, and grouped-query selection. The work establishes learned hierarchical sparse attention as prior art. Its native training and NVIDIA-oriented implementation are outside the proposed local replication; a single borrowed scoring operation would be an adaptation.

**[R8] DeepSeek-AI et al., “DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models.”** 2 December 2025. [https://arxiv.org/html/2512.02556v1](https://arxiv.org/html/2512.02556v1). Inspected the lightning indexer and its training: weighted rectified query–key scores select tokens for main attention. The native large-model system is not locally reproduced. Sparse main attention alone does not remove the indexer's full-history scan or its cumulative quadratic prefill work.

**[R9] Yushi Bai et al., “IndexCache: Accelerating Sparse Attention via Cross-Layer Index Reuse.”** 12 March 2026. [https://arxiv.org/html/2603.12201v1](https://arxiv.org/html/2603.12201v1). Inspected the reuse mechanism, loss-based calibration, and training-aware alternative. Reusing earlier selected indices is a strong competing allocation policy, not merely a weak ablation. Applying refresh schedules to Seer gates remains an IndexCache-inspired adapter, rather than a reproduction of its native DSA experiments.

**[R10] Yutao Sun et al., “You Only Index Once: Cross-Layer Sparse Attention with Shared Routing.”** 4 June 2026. [https://arxiv.org/abs/2606.06467](https://arxiv.org/abs/2606.06467). Inspected Sections 2–3 and the Microsoft research publication page. CLSA shares both cached keys/values and routing across cross-decoder layers in a KV-sharing architecture. Its architectural and training assumptions differ from reusing selections over a frozen Qwen model's separate layer caches.

**[R11] Mingkuan Zhao et al., “Making Every Head Count: Sparse Attention Without the Speed-Performance Trade-off.”** 12 November 2025; AAAI 2026. [https://arxiv.org/html/2511.09596v1](https://arxiv.org/html/2511.09596v1). Inspected SPAttention's disjoint head-specific distance bands. This directly precedes generic claims that heads should cover complementary positions. Total attention over all heads can remain quadratic, and early causal rows can leave distant-band heads empty; structural complementarity alone does not establish semantic specialization.

**[R12] Zhibo Yang, “Locality Does Not Imply Reachability: Boundary Repair in Block-Sparse Causal Attention.”** 1 June 2026. [https://arxiv.org/html/2606.02680v1](https://arxiv.org/html/2606.02680v1). Inspected the method and limitations. Layer-ordered reachability and boundary probes directly inform the graph diagnostic. Its fixed disconnected-block barrier assumes the stated communication and positionwise operations; changing partitions, adding global memory, or using already contextualized dense-prefill caches prevents a direct transfer of that argument.

**[R13] Krzysztof Choromanski et al., “Rethinking Attention with Performers.”** 30 September 2020; ICLR 2021. [https://arxiv.org/abs/2009.14794](https://arxiv.org/abs/2009.14794). Inspected the primary abstract and FAVOR+ description, without independently auditing every theorem. Positive orthogonal random features approximate kernel attention and supply a relevant approximation baseline. This concerns feature-space approximation rather than choosing token supports; its accuracy and complexity guarantees do not establish a Recamán construction's properties.

**[R14] Hanting Chen et al., “DiJiang: Efficient Large Language Models through Compact Kernelization.”** 29 March 2024. [https://arxiv.org/html/2403.19928v1](https://arxiv.org/html/2403.19928v1). Inspected Section 3's weighted quasi-Monte Carlo kernelization and discrete cosine transform construction. It is close prior art for deterministic or structured feature sampling. Its approximation assumptions and required model adaptation cannot be replaced by an unsupported claim that any irregular integer sequence has low discrepancy.

**[R15] Tri Dao et al., “Pixelated Butterfly: Simple and Efficient Sparse training for Neural Network Models.”** 30 November 2021; ICLR 2022. [https://arxiv.org/abs/2112.00029](https://arxiv.org/abs/2112.00029). Inspected Section 3 and the hardware-cost discussion. Blocked and flattened butterfly structure with low-rank components provides a substantial structured-sparsity comparator. It addresses sparse model training and matrix structure; it does not supply an executed-coverage controller or demonstrate arbitrary gather efficiency.

**[R16] Jaime Carbonell and Jade Goldstein, “The Use of MMR, Diversity-Based Reranking for Reordering Documents and Producing Summaries.”** SIGIR 1998. [https://doi.org/10.1145/290941.291025](https://doi.org/10.1145/290941.291025). Inspected bibliographic and institutional repository records; the publisher's full text was inaccessible. MMR is foundational relevance-versus-redundancy prior art. Standard maximum-similarity reranking must be distinguished from the plan's explicitly derived sum-history identity-kernel equivalence; sharing the general idea does not make their formulas identical.

**[R17] Zhaopeng Tu et al., “Modeling Coverage for Neural Machine Translation.”** ACL 2016. [https://aclanthology.org/P16-1008/](https://aclanthology.org/P16-1008/). Inspected the coverage mechanisms, including cumulative, fertility-related, and neural coverage used by later attention. Tracking previously attended source content is established neural-method prior art. The original dense translation system differs from a fixed-budget, query-local selector and provides neither a sparse-decoding speed claim nor evidence that all revisits are harmful.

**[R18] Abigail See et al., “Get To The Point: Summarization with Pointer-Generator Networks.”** ACL 2017. [https://aclanthology.org/P17-1099/](https://aclanthology.org/P17-1099/). **Simeng Han et al., “Resurrecting Submodularity for Neural Text Generation.”** 8 November 2019; revised 4 October 2020. [https://arxiv.org/html/1911.03014v3](https://arxiv.org/html/1911.03014v3). Inspected their coverage methods: accumulated attention with an overlap loss, and diminishing attention from submodular coverage. Both predate generic anti-repetition claims; neither establishes the proposed frozen-model sparse allocation's value.

**[R19] Shuvendu Roy, Mengyao Zhai, Hossein Hajimirsadeghi, Golnoosh Samei. _Coverage-Driven KV Cache Eviction for Efficient and Improved Inference of LLM_ (K-VEC), 28 June 2026.** [arXiv:2606.29563](https://arxiv.org/abs/2606.29563). Full method/algorithm and experimental setup inspected. Direct coverage/protected-relevance precedent; original prefill eviction, 7B/8B models and A100 conditions differ from reversible decode. An official executable release was not established; the proposed controller is explicitly adapted.

**[R20] Junyi Wu, Fanqing Kong, Leyang Chen, Shaoqiu Zhang, Yulun Zhang. _VASC: Value-Aware Sparse Attention with Cross-Layer Memory for Efficient 3D Reconstruction_, 1 October 2026.** [Version 1](https://arxiv.org/html/2610.01013v1); [author repository](https://github.com/kosakayamahoo-design/VASC). Full method, appendices and ablations inspected; repository README, not its implementation files, inspected. Direct demand/service-memory precedent. Geometry models, spatial pairs and Linux/CUDA implementation do not establish LLM or M3 performance.

**[R21] Kyuan Oh, Bumsoo Kim. _AnchorPrune: Relevance-Anchored Contextual Expansion for Visual Token Pruning_, 8 July 2026.** [arXiv:2607.07033](https://arxiv.org/abs/2607.07033); [author repository](https://github.com/MULTI-cau/AnchorPrune). Primary method and README/API inspected; individual selection code not retrieved. Protected relevance plus contextual diversity narrows novelty. Its visual representations and adaptive anchor rule differ from the specified text adapter; venue claims beyond repository metadata were not independently established.

**[R22] Qizhe Zhang et al. _Beyond Attention or Similarity: Maximizing Conditional Diversity for Token Pruning in MLLMs_ (CDPruner), June 2025, version 2 inspected.** [Paper](https://arxiv.org/html/2506.10967v2); [author repository](https://github.com/Theia-4869/CDPruner). Primary method exposition inspected; code not executed. Relevance-conditioned DPP diversity is a conceptual challenger, not an executed-history decoder. The local plan requires MMR and conditions a broader diversity claim on stronger adapters.

**[R23] Jiaming Tang, Yilong Zhao, Kan Zhu, Guangxuan Xiao, Baris Kasikci, Song Han. _Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference_, ICML 2024.** [Proceedings](https://proceedings.mlr.press/v235/tang24l.html); [paper](https://arxiv.org/html/2406.10774v1); [official code](https://github.com/mit-han-lab/Quest). Full selection method, conditions and repository material inspected. Supplies coordinate-extrema page scoring and a strong training-free comparator; native page/head/dense-layer conventions must survive the port. CUDA code availability is not proof of a Qwen GQA or Metal implementation.

**[R24] Yizhao Gao et al. _SeerAttention-R: Sparse Attention Adaptation for Long Reasoning_, 10 June 2025.** [Paper](https://arxiv.org/html/2506.08889v1); [metadata](https://arxiv.org/abs/2506.08889); [Microsoft repository](https://github.com/microsoft/SeerAttention). Full gate/method and evaluation inspected. Reusable learned decoder gates make the primary frozen experiment credible; published long-generation/GPU findings are not reproduced here. Exact asset and implementation provenance follow in R42–R43.

**[R25] Zhiwei Li, Lei Zhu et al. _SAS: Simple Attention Sparsification via End-to-End Optimization of Context Ranking_, 11 September 2026.** [Version 1](https://arxiv.org/html/2609.13141v1); [official release](https://github.com/Tencent-Hunyuan/Simple-Attention-Sparsification). Full method, exported assets and relevant inference code inspected. Gate training through LM loss is a stronger alternative to attention imitation; released 4B gates are potentially local. Adequate code/weight permission was not established, so this is an explicit conditional comparator.

**[R26] Seonghwan Choi, Beomseok Kang, Dongwon Jo, Jae-Joon Kim. _Retrospective Sparse Attention for Efficient Long-Context Generation_ (RetroAttention), August 2025; version 2, 20 May 2026.** [Paper](https://arxiv.org/html/2508.09001v2); [author code](https://github.com/csh3695/RetroAttention). Full retrospective mechanism and evaluation inspected. Revises earlier outputs using newly fetched KV, with additional state and long-generation focus. It does not reduce to the candidate's query-local selection frequency; required comparison if that broader long-output claim is pursued.

**[R27] Desai et al. _vAttention: Verified Sparse Attention_, October 2025; version 2 inspected, May 2026.** [Paper](https://arxiv.org/html/2510.05688v2). Full approximation and validation method inspected. Combines important items with weighted random-tail estimates; numerator and denominator estimation and confidence assumptions matter. A random support mask with ordinary selected softmax is not this method, and its practical verification assumptions are not the candidate's proxy-score guarantee.

**[R28] Thomas Rossi. _Uncertainty-gated selection for block-sparse attention: A value-of-information view of the SSA-style selector_, 2026, version 1.** [Paper](https://arxiv.org/html/2607.07724v1). Method and experiments inspected; code not inspected. Near-cutoff margins trigger extra reads, motivating matched variable-budget controls. Displayed manuscript/submission dates disagree; cite the version and recheck metadata when fetching. More average blocks are not a free same-budget improvement.

**[R29] Piotr Nawrot, Robert Li, Renjie Huang, Sebastian Ruder, Kelly Marchisio, Edoardo M. Ponti. _The Sparse Frontier: Sparse Attention Trade-offs in Transformer LLMs_, Findings of ACL, July 2026.** [Proceedings paper](https://aclanthology.org/2026.findings-acl.1926.pdf); [official code](https://github.com/PiotrNawrot/sparse-frontier). Full methods/conditions and task-family findings inspected. Its 4B–72B models, 16K–128K contexts and six training-free methods support sensitivity and fairness decisions; its frontier is not a universal ranking of all later routers.

**[R30] Alex Graves et al. _Hybrid computing using a neural network with dynamic external memory_, Nature 538, 471–476, 2016.** [DOI/publisher](https://doi.org/10.1038/nature20101); [DeepMind implementation](https://github.com/google-deepmind/dnc). Publisher/abstract metadata and architecture description inspected; direct publisher retrieval was intermittent. Established learned external read/write memory bounds the memory-controller novelty claim. It is a distinct trained architecture, not a drop-in sparse Qwen selector or a reproduced local baseline.

**[R31] Xiuying Wei, Caglar Gulcehre. _Augmenting Attention with Exponentially Decaying Memory Improves Query-Aware KV Sparsity_ (RAT+), 27 May 2026.** [Version 1](https://arxiv.org/html/2605.28640v1). Full mechanism and experimental conditions inspected. Decayed content augmentation changes K/V and uses substantial training; it is not a cheap visit counter. Its limited task/length conditions constrain extrapolation. Cited as a content-memory pivot boundary, not a required frozen-controller reproduction.

**[R32] Tri Dao et al. _FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness_, 2022.** [Paper](https://arxiv.org/abs/2205.14135). Original architecture/IO analysis and metadata inspected at a conceptual level. Establishes why dense references need memory-aware exact kernels; an explicitly materialized score matrix is not a fair speed baseline. No CUDA-to-Metal performance transfer is assumed.

**[R33] Ted Zadouri, Markus Hoehnerbach, Jay Shah, Timmy Liu, Vijay Thakkar, Tri Dao. _FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling_, 5 March 2026.** [Paper](https://arxiv.org/abs/2603.05451). Primary abstract and hardware-target description inspected. Blackwell-specific pipelining illustrates why operation counts do not determine elapsed time. This is hardware context, not a locally executable M3 baseline or evidence of the candidate's speed.

**[R34] MLX/MLX-LM and PyTorch maintainers. Package release metadata, accessed 17 October 2026.** [MLX](https://pypi.org/project/mlx/), [MLX-LM](https://pypi.org/project/mlx-lm/), [PyTorch](https://pypi.org/project/torch/), [MLX-LM dependency declaration](https://github.com/ml-explore/mlx-lm/blob/main/pyproject.toml). Inspected release/version and dependency metadata supporting candidate pins 0.32.4/0.32.2 and optional PyTorch 2.14.1. These sources establish availability, not a tested combined environment; WP-00 resolves actual native wheels.

**[R35] Apple MLX maintainers. `mlx.core.fast.scaled_dot_product_attention` API, accessed 17 October 2026.** [Official documentation](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.scaled_dot_product_attention.html). Inspected GQA/MQA support, input/mask conventions, lower-right causal alignment and fusion controls. Establishes candidate operations, not successful dispatch or speed for the user's dtype/padding/shape. Frozen inference avoids unsupported sparse-backward assumptions.

**[R36] MLX-LM maintainers. Qwen3 model and generation implementation, accessed 17 October 2026.** [Qwen3](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/qwen3.py); [generation](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/generate.py). Inspected normalization, head dimensions, KV/update and prefill/generation boundaries. Reuse these components while preserving a trusted unmodified baseline. Branches are mutable; hash the exact implementation and adapted diff at execution time.

**[R37] MLX maintainers. Gather, sorting, sparse primitives, Metal execution and memory APIs, accessed 17 October 2026.** [Gather](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.take_along_axis.html); [stable sort](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.argsort.html); [block-masked matmul](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.block_masked_mm.html); [gathered matmul](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.gather_mm.html); [custom Metal](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.metal_kernel.html); [synchronization](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.synchronize.html); [memory](https://ml-explore.github.io/mlx/build/html/python/memory_management.html); [Metal SDPA source](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/scaled_dot_product_attention.cpp). APIs and relevant source dispatch/backward branches inspected; no target benchmark executed.

**[R38] PyTorch maintainers. MPS and SDPA documentation, accessed 17 October 2026.** [MPS guide](https://docs.pytorch.org/docs/2.14/notes/mps.html); [SDPA](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html); [MPS environment variables](https://docs.pytorch.org/docs/main/mps_environment_variables.html). Inspected backend availability, causal-mask reference and fallback controls. MPS presence is not proof of the required fused GQA path; avoid silent CPU fallback in measured acceleration claims.

**[R39] Apple. _Apple unveils M3, M3 Pro, and M3 Max_, 30 October 2023.** [Official announcement](https://www.apple.com/newsroom/2023/10/apple-unveils-m3-m3-pro-and-m3-max-the-most-advanced-chips-for-a-personal-computer/). Product configurations inspected as hardware context. The actual laptop configuration is unknown and must be measured; no maximum-memory assumption or advertised bandwidth-to-runtime conversion is used.

**[R40] Qwen team. Original Qwen3 model cards and licenses, 2025 releases, accessed 17 October 2026.** [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B); [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B). Cards, chat/thinking settings and Apache-2.0 declarations inspected. Supplies exact frozen base identity and reduced-scope smoke model. Controlled greedy non-thinking evaluation is a declared experimental choice, not a claim to follow the recommended stochastic production policy.

**[R41] Qwen team. Original Qwen3-4B `config.json`, accessed 17 October 2026.** [Exact configuration path](https://huggingface.co/Qwen/Qwen3-4B/blob/main/config.json). Full configuration inspected. Grounds layer/head/vocabulary/dtype/position settings and parameter/KV-memory derivations. Memory arithmetic is this plan's calculation; loader peak and M3 execution remain unmeasured. Resolve an immutable Hub commit before downloading.

**[R42] SeerAttention authors. Qwen3-4B decode-gate export, accessed 17 October 2026.** [Asset](https://huggingface.co/SeerAttention/SeerAttention-Decode-Qwen3-4B-AttnGates); [config](https://huggingface.co/SeerAttention/SeerAttention-Decode-Qwen3-4B-AttnGates/blob/main/config.json). Asset tree/config/size inspected. Grounds block 64, gate dimension 128, projection/pooling/norm settings and compatible original base. Asset permission and exact weights/tensor hashes remain download-time gates; no fabricated smaller-model gate is assumed.

**[R43] Microsoft SeerAttention maintainers. Decode inference implementation, commit `aba03e3f2caefd0ccd21e576670aa830b748c84e`, 3 July 2025.** [Gate](https://github.com/microsoft/SeerAttention/blob/aba03e3f2caefd0ccd21e576670aa830b748c84e/seer_attn/decode_sparse/attn_gate_inf.py); [Qwen integration](https://github.com/microsoft/SeerAttention/blob/aba03e3f2caefd0ccd21e576670aa830b748c84e/seer_attn/decode_sparse/qwen3/modeling_qwen3_seerattn_inference.py); [cache](https://github.com/microsoft/SeerAttention/blob/aba03e3f2caefd0ccd21e576670aa830b748c84e/seer_attn/decode_sparse/cache_utils.py); [common math](https://github.com/microsoft/SeerAttention/blob/aba03e3f2caefd0ccd21e576670aa830b748c84e/seer_attn/modules/common.py); [sparse dispatch](https://github.com/microsoft/SeerAttention/blob/aba03e3f2caefd0ccd21e576670aa830b748c84e/seer_attn/decode_sparse/attention_forward_sparse.py). Exact source contents and commit metadata inspected through the repository connector when web retrieval failed. Supplies math, forced-block and cache/RoPE discrepancies. Five fetched files matched their Git blob hashes. No upstream package execution is claimed.

**[R44] Tencent SAS maintainers. Release and exported gates, September 2026.** [Release repository](https://github.com/Tencent-Hunyuan/Simple-Attention-Sparsification); [gate asset tree](https://huggingface.co/tencent/Simple-Attention-Sparsification); [SGLang backend](https://github.com/rayleizhu/sglang/blob/2a090d525cc52dc8eb70059543271baef34c5696/python/sglang/srt/layers/attention/blocksparse/seer_attn/backend.py); [gate implementation](https://github.com/rayleizhu/sglang/blob/2a090d525cc52dc8eb70059543271baef34c5696/python/sglang/srt/layers/attention/blocksparse/seer_attn/attn_gate.py). Release commit `c7d7101bbf5d52edf89f6b02de86f6a0fbde4d6e`, Qwen3-4B asset and inference details inspected. Native historical-budget-plus-current accounting differs from Microsoft. Short golden fixtures below the budget do not validate sparse execution. License remained unresolved.

**[R45] Zhilin Yang et al. _HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering_, EMNLP 2018; original v1.1 data.** [Official data and license](https://hotpotqa.github.io/); [paper](https://aclanthology.org/D18-1259/). Official downloads, format and CC BY-SA 4.0 terms inspected. Grounds required natural transfer data. Public development is used as a locally protected subset, not the official hidden test; pretrained contamination and shared sources remain limitations.

**[R46] HotpotQA authors. Original answer evaluator.** [Code](https://github.com/hotpotqa/hotpot/blob/master/hotpot_evaluate_v1.py). Source inspected for answer exact match, F1 and normalization/yes-no handling. Use answer metrics only unless supporting-sentence predictions are actually implemented. Pin its revision/hash; a custom lenient parser must not replace official scoring unnoticed.

**[R47] Harsh Trivedi et al. _MuSiQue: Multihop Questions via Single-hop Question Composition_, 2022; MuSiQue-Ans v1.0 assets.** [Official README/data/license](https://github.com/StonyBrookNLP/musique/blob/main/README.md); [download script](https://github.com/StonyBrookNLP/musique/blob/main/download_data.sh). Data/evaluator instructions and CC BY 4.0 terms inspected. Optional independent natural confirmation; component-question overlap and downloadable-file availability require checking. No decomposition/support labels enter model prompts.

**[R48] Cheng-Ping Hsieh et al. _RULER: What's the Real Context Size of Your Long-Context Language Models?_, 2024.** [Paper](https://arxiv.org/abs/2404.06654). Primary task descriptions and use in inspected sparse-attention evaluations inform the synthetic/natural distinction. It is contextual evidence, not the required population or a reproduced score; a needle/retrieval result alone cannot establish natural multi-hop transfer.

**[R49] Hazy Research. Zoology synthetic sequence-model research code, accessed 17 October 2026.** [Repository](https://github.com/HazyResearch/zoology); [license](https://github.com/HazyResearch/zoology/blob/main/LICENSE.md); [basic experiment](https://github.com/HazyResearch/zoology/blob/main/zoology/experiments/basic_examples/basic.py). Repository/license and associative-recall setup inspected. Reusable for conditional trained synthetic studies. Integer-token MQAR is not an equivalent zero-shot input distribution for pretrained Qwen.

**[R50] Rotem Dror, Gili Baumer, Segev Shlomov, Roi Reichart. _The Hitchhiker's Guide to Testing Statistical Significance in Natural Language Processing_, ACL 2018.** [Paper/metadata](https://aclanthology.org/P18-1128/). Primary evaluation guidance inspected. Informs paired inference and suitable tests; this plan's world/source clusters, hierarchical claims, rare-event fallback and precision calculations are explicit design choices, not claims that the paper prescribes this exact protocol.

**[R51] Hugging Face maintainers. Hub download and repository metadata APIs, accessed 17 October 2026.** [Download guide](https://huggingface.co/docs/huggingface_hub/guides/download); [Hub API](https://huggingface.co/docs/huggingface_hub/package_reference/hf_api). Official revision/download/metadata guidance inspected. Supports pinned public-asset manifests and size/license discovery; permissions and file content must be verified rather than inferred from a successful HTTP download.

**[R52] Ankit Gupta, Jonathan Berant. _Value-Aware Approximate Attention_, EMNLP 2021.** [Proceedings](https://aclanthology.org/2021.emnlp-main.753/); [paper](https://arxiv.org/abs/2103.09857). Primary method/abstract inspected. Establishes that attention values matter to approximation quality, supporting separate mass/output/task metrics. The elementary output-error bound in Section 2 is derived here and is not a guarantee supplied by the proposed cheap gate.

**[R53] Sparse-attention patent screening.** [US11238332B2, _Attention neural networks with sparse attention mechanisms_](https://patents.google.com/patent/US11238332B2/en), Google, published 1 February 2022; [US20260004125A1](https://patents.google.com/patent/US20260004125A1/en); [US10521465B2, _Deep reinforced model for abstractive summarization_](https://patents.google.com/patent/US10521465B2/en). First document's description/claims and summarization description inspected; dynamic-pattern application screened at metadata level. Evidence of a populated disclosure landscape only. This limited search is not claim construction, patentability analysis or freedom-to-operate advice.

**[R54] Lijie Yang et al. _Less Is More: Fast and Accurate Reasoning with Cross-Head Unified Sparse Attention_, 2025; version 2, 27 April 2026.** [Revised paper](https://arxiv.org/html/2508.07101v2). Full method, evaluation and refresh-layer appendix inspected. Cross-head token selection with cross-layer reuse is a strong opposing mechanism on the intended model family. Published long-reasoning GPU results cannot be imported into the M3 short-answer comparison.

**[R55] LessIsMore authors. Qwen3 implementation, commit `fa3d23a45fde680ac220394e7c569f5e98ca144e`, 31 August 2025.** [Exact source](https://github.com/DerrickYLJ/LessIsMore/blob/fa3d23a45fde680ac220394e7c569f5e98ca144e/src/tidal_build/modify_qwen3_lim.py); [repository](https://github.com/DerrickYLJ/LessIsMore). Source, commit metadata and Apache-2.0 declaration inspected. Resolves rank-first/head-second stable union, defaults, recent/sink allocation and dense scoring costs. The original batch-one host-list implementation must be lowered faithfully for timing; a generalized optimized Metal port is new engineering.

**[R56] Zhuoming Chen, Xinrui Zhong, Qilong Feng, Ranajoy Sadhukhan, Yang Zhou, Michael Qizhe Shieh, Zhihao Jia, Beidi Chen. _Vortex: Efficient and Programmable Sparse Attention Serving for AI Agents_, 4 June 2026.** [Version 1](https://arxiv.org/html/2606.06453v1); [official repository](https://github.com/Infini-AI-Lab/vortex_torch). Full method, evaluation and generated-flow appendix inspected; repository root/README and Apache-2.0 declaration inspected, individual generated flows not retrieved. Past-score EMA penalties motivate a cheap control; the final optimized winners used ordinary block top-k plus system tuning. The native Qwen3-1.7B/H200 agent-serving setting and cross-token history do not establish the local adapter's effectiveness.

**[R57] Yao Fu et al. _SPIN: Shadow Predictive Indexer for Sparse Attention_, 6 October 2026; version 2, 8 October 2026.** [Version 2](https://arxiv.org/html/2610.09025v2); [metadata](https://arxiv.org/abs/2610.09025). Full prediction/update algorithm and ablations inspected; no official implementation located in the primary source or focused search. Absolute-slot/relative-offset EMA predicts inputs to a native indexer; an in-budget random quota refreshes observations. Native evaluation uses DeepSeek-V4 and multi-GPU NVIDIA serving. It is a required comparison boundary for a future cross-query indexer-pruning claim, not a reproduced final-support controller here.

**[R58] Fanxu Meng, Taosong Fang. _DoSA: Dots Sparse Attention_, version 1, 2026.** [Primary method](https://arxiv.org/html/2610.13990v1); [metadata](https://arxiv.org/abs/2610.13990); [official implementation link](https://github.com/llm-sys/dosa-sglang). Primary method, numerical qualifications and decode section inspected; repository roots inspected, implementation not audited or executed. Both arXiv metadata pages display 26 September 2026 despite the October-form identifier; preserve this discrepancy rather than inventing a corrected date. Native DSA score bounds and selection optimizations contextualize exact acceleration. They do not automatically apply to Seer's different gate or prove Metal speed.

**[R59] Xu Yang, Jiapeng Zhang, Yuxin Chen, Feiqiang Sun, Chengguang Xu, Feng Jin, Zhuo Tang. _Self-Indexing Attention for Compression-Compatible Sparse Long-Context LLM Inference_, September 2026; version 2 revised 7 October 2026.** [Version 2](https://arxiv.org/html/2609.13205v2); [metadata](https://arxiv.org/abs/2609.13205); [MLX Hadamard API](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.hadamard_transform.html). Full method/appendices and CC BY 4.0 paper declaration inspected. No official code, code license or original-Qwen3-4B release was verified after targeted searching; this is not a claim that none can exist. The independently authored decode adapter in Section 5 specifies its own RNG, support policy and d=128 conventions. Native later-Qwen/d=256 packed-kernel results are context, not local measurements. Official MLX documentation supports probing the transform, not assuming dispatch performance.

**[R60] Hoang Anh Duy Le, Sahil Joshi, Zeyu Yang, Zhaozhuo Xu, Anshumali Shrivastava. _Scout Before You Attend: Sketch-and-Walk Sparse Attention for Efficient LLM Inference_, arXiv version 1, 7 February 2026; ICML 2026, PMLR 306:63033–63059.** [Proceedings](https://proceedings.mlr.press/v306/le26c.html); [final paper](https://proceedings.mlr.press/v306/le26c/le26c.pdf); [arXiv](https://arxiv.org/abs/2602.07397); [exact Qwen3 source](https://github.com/Escanord/SketchWalk/blob/35048bb7f3eed2f0e15c557a7f54a36c71e0f2f9/pipeline/sketchwalk/modeling/modeling_qwen3.py); [pinned released decode config](https://github.com/Escanord/SketchWalk/blob/35048bb7f3eed2f0e15c557a7f54a36c71e0f2f9/config/pipeline_config/SketchWalk/Qwen3-8B/Qwen3-8B-inference-decoding.json). Final primary PDF, repository/license/config and exact source contents inspected; source Git blob verified as `ca150787995b3ef916b31c466083750f66797a80`. Repository MIT license and inherited Qwen source Apache-2.0 header were present. Paper and released decode recurrences differ; source threshold/window issues motivate the named correction, not an unqualified reproduction. The config comes from Qwen3-8B; applying its settings to original 4B is a declared model-size adaptation. NumPy control-flow checks were executed; upstream PyTorch/model/GPU benchmarks were not.

### 12.3 Search coverage and stopping rationale

Research was conducted through **17 October 2026**, using two search engines, primary full papers and official documentation, model/data records, and author repositories. Search terms were broadened beyond “Recamán neural network” to cover sparse attention topology, causal reachability, boundary repair, coverage vectors, inhibition of return, novelty/diversity, executed-selection history, score EMA, cross-layer reuse, random walks, indexer pruning, value-aware approximation, structured mixing, external memory and quasi-Monte Carlo features. These are representative query families, not a claim to have exhaustively searched every terminology or database.

| Search wave / citation tracing | Material discovery or decision |
|---|---|
| Recamán definition; sparse-transformer and expander references, followed forward to complementary-head/reachability analysis | Separated literal sequence tests from useful routing, corrected layer-order reachability, and required common boundary repairs. |
| Coverage/diversity history backward to MMR, NMT, summarization and inhibition of return; forward to K-VEC, VASC and relevance-anchored pruning | Broad novelty substantially anticipated; required source-informed coverage/diversity controls and a narrower empirical contribution. |
| Current sparse LLM routing and its references: MoBA/NSA/DSA, Quest, SeerAttention-R, SAS, IndexCache, LessIsMore, RetroAttention and Sparse Frontier | Selected the frozen 4B learned gate, retained reuse as an opposing hypothesis, and separated scorer, controller, budget and hardware claims. |
| Score-history, exploration and exactness follow-ups through Vortex, SPIN and DoSA | Added score-EMA; clarified partial observation, cross-token state and native-scorer bounds. No native serving result became an unmeasured local baseline. |
| Later sign-index and walk-based alternatives: SIA and backward/forward checking of Sketch-and-Walk's final proceedings/release | Added two feasible mathematical competitors. A prior statement of unavailable code was superseded by the accessible SketchWalk release; source inspection exposed paper/release and density discrepancies. |
| Exact code paths, official package/operator documents and model/data assets | Identified Seer's gate/cache/RoPE issues; pinned source paths; derived memory bounds; made M3 operators, licenses and achieved precision explicit early gates. |
| Final targeted combinations of execution/selection history, coverage, sparse attention, exponential decay and score EMA | Returned already mapped controller families or representation-memory mechanisms with different causal targets; no additional primary method requiring a new core comparator emerged in that bounded pass. |

The search stopped after the last material discoveries had changed the design and their code/definition follow-ups no longer produced a distinct missing core mechanism or unresolved design decision that could be settled by another targeted lookup. This is a **diminishing-information stopping judgment**, not a paper quota or proof of novelty. The bibliography intentionally records decisive sources, rather than every duplicate search result.

Access limits remain visible in individual entries: some publisher pages were blocked or intermittent; primary PDFs or exact author-repository files were used when available. Several repositories were read only at README/metadata level, and no published end-to-end result was reproduced. Remaining blind spots include unpublished/unindexed work, non-English and broader patent collections, software or asset changes after the cutoff, private implementations, and exact-equivalent rules published under terminology not located here. Newly discovered equivalent prior art should narrow or remove a novelty claim; it must not cause favorable results to be rebranded as a new principle.
