# PoC execution brief: evidence-supported numerical interpretation

**Prepared:** 1 November 2026.  
**Executor:** an autonomous coding agent configured as `gpt-6.1-sol`, reasoning effort `xhigh`.  
**Experimental machine:** one MacBook with an M3 Max; inspect its actual memory, operating system, storage, and backend before allocation.  
**Deliverable of this document:** an instruction for implementing and executing the study. The planning work described here is not a completed PoC or an M3 Max benchmark.

## 1. Project decision and smallest defensible contribution

Build a **certificate-scored diagnostic of numerical identifiability**, then use it for one bounded comparison of three prompts applied to one local language model. The diagnostic asks whether anonymous addition relationships determine a complete numerical dictionary. A successful response must either recover the unique dictionary or demonstrate ambiguity with two different, complete, compatible dictionaries. A third response, `unresolved`, reports insufficient analysis; it is not a property of the evidence.

The primary comparison is **explicit search for a competing interpretation versus strong generic caution**, under the same model, observations, output contract, and generation limits. Include a neutral task prompt as a simple reference. Use an exact, independently checked mathematical decoder as the scoring oracle and a deterministic positive reference. Do not give its answers, theorem, algorithm, or feedback to the language model.

The practical decision is whether the witness-seeking configuration merits further evaluation on application data. The minimum useful research output is a reproducible diagnostic that exposes the difference between correct classification, constructive interpretation, demonstrated ambiguity, and unsupported claims. A favorable prompt result is not assumed.

### Primary falsifiable question

> On the locked interval-addition panel and its controlled presentations, does a prompt that explicitly searches for a second compatible dictionary reduce unsupported claims of uniqueness by at least 10 percentage points relative to generic caution, while losing no more than 5 points of useful complete performance?

“Useful complete performance” means a valid full recovery with correct unique status, or correct ambiguous status with two valid distinct full assignments. Neither classification alone nor a plausible dictionary receives this credit. Section 6 defines the exact estimand, conservative handling of ungradable responses, usefulness floors, and screening decision.

**Mechanism hypothesis, not an established result:** directing attention to an alternative complete interpretation may prevent promotion of a possible assignment to a unique one. The effect may disappear when the baseline already avoids unsupported uniqueness, when the model cannot construct even one dictionary, when witnesses exceed the resource budget, or when generic caution already performs the same checks. It may reverse if extra search distracts the model or induces unjustified ambiguity claims.

**Contribution boundary.** This is not a new general theory of abstention, symbol grounding, counterexample prompting, or solver-assisted reasoning. WorldSense already tests completeness of evidence; LogicSkills already tests semantically verified countermodels; AbstentionBench studies abstention and caution prompts; symbol-remapping and counterexample-search benchmarks also exist [R1–R6]. The project-specific extension is the unified decision-and-construction task on one numerical inverse model, with a sharp boundary, complete positive and negative evidence, exact witness checking, and exact assessment of individual fixed values. Historical priority of the supplied interval theorem remains unconfirmed. Describe a successful result as a scoped diagnostic or empirical contribution, not a first-ever benchmark or universal reliability improvement.

### Fixed scope and exclusions

| Included | Excluded from this PoC |
|---|---|
| Complete, clean inventories and complete unordered distinct addition triples | Learning what addition means, learning the hypothesis family, recovering a damaged or incomplete table |
| Consecutive positive integer values with lower endpoint at least 6 | Gaps, repeated values, arbitrary common scaling, negative values, modular arithmetic |
| Exact full-dictionary and individual-value identifiability | Physical units, currency, business semantics, alien intentions, internal model understanding |
| One fixed local 4B model, three prompt configurations | Hosted experimental inference, paid computation, model training, a broad model leaderboard |
| Deterministic solver as oracle and practical upper reference | Crediting the language model for answers substituted by the solver |
| A finite-panel screening decision with uncertainty reported | A claim of deployment readiness, universal noninferiority, or transfer to real application data |

These restrictions preserve the original reliability question. They limit its generality; they do not replace dictionary recovery with an easier classification problem. The notebook's binary-marker theorem is a separate communication problem and supplies no mechanism needed here.

## 2. Mathematical evidence, assumptions, and executable consequences

### 2.1 Source status and provenance

The supplied research source is `alien_communication_research(1).ipynb` [N1], a research notebook containing proofs, code, recorded outputs, and a novelty discussion. It is not an independently refereed paper. Its author is not identified in the supplied metadata. Cell numbers in this brief are **zero-based JSON cell indices**; the accompanying headings make the locations independently findable.

| Material in the notebook | Status and use |
|---|---|
| Cell 1, Theorem A, Lemmas 1–2, and proof | Source proof of rigidity iff `n >= 3a - 1`, for `a >= 6`, `n >= a + 1`; audited during planning without finding a defect |
| Cell 1, “Corollary: endpoints can also be inferred” | Source endpoint formula, applicable only when at least one edge exists |
| Cell 1, “Algorithmic corollary”; cell 11, `reconstruct` | Constructive unique-regime algorithm; the source code recomputes degrees and does not implement the optimized quadratic bound |
| Cell 6, `automorphism_count`; cells 9 and 12 | Source reports 135 capped automorphism checks and 90 scrambled reconstructions; stopping at two automorphisms cannot determine all orbits |
| Cell 2 | Exact smallest unique example, including one isolated but uniquely determined symbol |
| Cells 3–8 | Separate binary-marker results; read for scope, not used or independently reproduced for this project |
| Cells 13–14 | Conditional interpretation limits and unconfirmed historical novelty; preserve these limitations |

The complete task specification is `Pasted text(20261101-130307).txt` [N2]. The notebook refers to a Golden Board roadmap and a standalone `schur_reconstruct.py`; neither was separately attached. Neither is required: the relevant mathematics and algorithm specification are reproduced below, and the reconstruction code is inside notebook cell 11. Do not invent or retrieve an unrelated replacement paper.

### 2.2 Exact object and interpretation space

An observation is a pair `G = (V, E)`:

- `V` is a complete finite inventory of `m >= 2` distinct opaque symbols, including isolated symbols.
- `E` is a set of unordered three-element subsets of `V`. Duplicate records, repeated symbols inside a triple, and unknown symbols are invalid input syntax.
- A compatible dictionary is a bijection `f: V -> {a, a+1, ..., n}` for integers `a >= 6`, `n = a + m - 1`, such that for every three distinct symbols,

  \[
  \{u,v,w\}\in E
  \quad\Longleftrightarrow\quad
  \text{the two smaller values among }f(u),f(v),f(w)
  \text{ sum to the largest.}
  \]

Let `F(G)` be the set of all compatible dictionaries over **all** allowed endpoints. A clean generated case has `F(G)` nonempty. It is unique exactly when `|F(G)| = 1`; otherwise it is ambiguous. `Unresolved` is never a gold mathematical class. For a symbol `v`, the numerical claim `f(v) = t` is justified exactly when every member of `F(G)` assigns `t` to `v`.

The canonical hypergraph is

\[
H[a,n]=\bigl([a,n],\{\{x,y,x+y\}:a\le x<y,\ x+y\le n\}\bigr).
\]

All oracle arithmetic uses Python integers. There are no rounding tolerances, numerical optimization, convergence criteria, or floating-point approximations in the truth definition.

### 2.3 The source theorem and the endpoint bridge

The source's Theorem A states

\[
\operatorname{Aut}(H[a,n])=\{\mathrm{id}\}
\quad\Longleftrightarrow\quad n\ge3a-1,
\qquad a\ge6,\ n\ge a+1.
\tag{1}
\]

This is initially a statement about automorphisms of a fixed interval, not automatically about alternative intervals. The endpoint corollary supplies the necessary bridge. With `E != empty`, write `d = max degree`, `m = |V|`. Then

\[
d=n-2a,\qquad a=m-1-d,\qquad n=2m-2-d.
\tag{2}
\]

Consequently every compatible nonempty interpretation has the same endpoints. If `f0` is one compatible dictionary, every other one is `alpha composed with f0` for an automorphism `alpha` of `H[a,n]`. Dictionary uniqueness follows from (1), while individual fixed values are exactly singleton automorphism orbits. This is why a whole-dictionary threshold cannot be used as an individual-symbol oracle.

The source degree formula is

\[
\deg(v)=\max(0,n-v-a+1)-\mathbf1_{2v\le n}
 +\max\left(0,\left\lfloor\frac{v-1}{2}\right\rfloor-a+1\right).
\tag{3}
\]

The first part counts occurrences as an addend, excluding a repeated-addend triple; the last counts occurrences as a sum. For `a >= 3` and `k = n - 2a >= 4`, the vertices `a` and `a+1` are the unique first and second degree maxima, of degrees `k` and `k-1` [N1, cell 1, Lemma 1].

Equations (1)–(2) imply the observable classification rule `2*d >= m-2` when the promised family is nonempty. Degree and count shortcuts are legitimate evidence. Their availability is a reason to require full constructive answers, not to suppress those statistics.

### 2.4 Empty evidence is an infinite ambiguity case

**Planning derivation.** When `E` is empty, (2) does not apply. The least sum of distinct interval values is `2a+1`; hence the compatible starts are exactly

\[
a\ge\max(6,m-1),\qquad n=a+m-1.
\tag{4}
\]

There are infinitely many compatible dictionaries, and no symbol has a fixed numerical value. To construct two witnesses, choose `a0 = max(6,m-1)` and `a0+1`, then assign the inventory bijectively to each interval in any fixed internal order. Validate both. This is a symbolic resolution of an unbounded hypothesis space; do not replace it with an arbitrary endpoint search cap.

### 2.5 Why the positive theorem is constructive

Here is the essential proof structure, sufficient to audit the implementation [N1, cell 1]:

1. In each boundary interval `[b,3b-1]`, `[b,3b]`, `[b,3b+1]`, Lemma 1 fixes `b` and `b+1`.
2. The union of their link graphs exposes a path, plus explicitly identifiable anchors and, in the first case, a single isolate. A link at `u` has edge `{v,w}` whenever `{u,v,w}` is a hyperedge.
3. At endpoints `3b-1` and `3b`, the path has two possible orientations. The edge `{b+2,b+3,2b+5}` exists for `b >= 6` and rejects reversal. At endpoint `3b+1`, the path is anchored at the already fixed value `2b+1`.
4. For a longer interval set `b = floor((n+1)/3)`. Peel the unique maximum-degree vertex, assigning successive values `a, ..., b-1`. Each deletion leaves the next interval hypergraph. Decode the remaining boundary and restore the peeled labels.
5. Below the threshold, the interval is either edgeless or has at least two isolated values, so an isolated swap proves nonuniqueness.

The optimized incidence-update implementation uses `O(m^2)` indexed operations and storage because there are `O(m^2)` edges. A simpler implementation that recomputes degrees during peeling has an `O(m^3)` upper bound and is acceptable at this project's sizes. Report the implementation actually timed. String parsing and tokenization have their own byte costs; the indexed-operation bound is not a complete LLM pipeline bound.

### 2.6 A complete ambiguity decoder and exact partial-value oracle

**New derivation during this planning task; historical novelty unverified.** This complements the notebook's unique-only decoder. Recheck its proof and the independent finite tests before using it as the scorer.

For nonempty ambiguous evidence define

\[
k=n-2a,\quad1\le k\le a-2,\quad
A_i=a+i\ (0\le i\le k),\quad S_j=2a+j\ (1\le j\le k).
\]

The nonisolated vertices are precisely the `2k+1` values `A_0,...,A_k,S_1,...,S_k`. The isolates form

\[
I=[a+k+1,2a],\qquad r=|I|=a-k\ge2.
\tag{5}
\]

Every edge is `{A_i,A_j,S_(i+j)}`, where `0 <= i < j` and `i+j <= k`. These addend and sum ranges are disjoint.

For `k >= 4`, Lemma 1 fixes `A_0,A_1`; their common triple fixes `S_1`. Remove these three and the isolates from the union of the two anchor links. What remains is the path

\[
S_2,A_2,S_3,A_3,\ldots,S_k,A_k.
\tag{6}
\]

Test its two orientations against the full relation. Both work at `k=4`; only one works at `k>=5`. For the latter, reversal is `v -> 3a+k+2-v`. It sends the edge `{A_2,A_3,S_5}` to `{2a+k,2a+k-1,a+k-3}`, a nonedge because the smaller two sum to `3a+2k-4 > 2a+k`.

For `k=1,2,3`, enumerate bijections only on the nonisolated support, of size 3, 5, or 7. At most `7! = 5040` candidates suffice. Assign isolates arbitrarily and never enumerate their factorial permutations. Obtain a second full dictionary by swapping two isolated values in any valid dictionary.

The exact value-orbit table is:

| `k` | Nonisolated orbits | Nonisolated automorphism count |
|---:|---|---:|
| 1 | `{A0,A1,S1}` | 6 |
| 2 | `{A0}`, `{A1,A2,S1,S2}` | 8 |
| 3 | `{A0}`, `{A1,A2,S3}`, `{A3,S1,S2}` | 6 |
| 4 | `{A0}`, `{A1}`, `{S1}`, `{A2,S4}`, `{A3,S3}`, `{A4,S2}` | 2 |
| At least 5 | Every active value is a singleton | 1 |

`I` is one additional orbit in every row; total automorphism count is the last column multiplied by `r!`. The small rows follow from the explicit edge patterns: one triple; two triples sharing a center; and, at `k=3`, three interchangeable degree-two vertices with their determined leaves. The path argument proves the remaining rows. Convert these value orbits to symbol orbits through a validated dictionary.

**Critical sentinel:** `H[6,16]` has four automorphisms and orbits

\[
\{6\},\{7\},\{13\},\{8,16\},\{9,15\},\{10,14\},\{11,12\}.
\]

Only 6, 7, and 13 are fixed. Conversely, the isolated value 12 in `H[6,17]` is uniquely determined. These two examples invalidate both “all nonisolated values are fixed” and “isolated values are never fixed.”

## 3. Exact implementation specification

### 3.1 Separate observations, generation secrets, and truth computation

The model receives only the common task statement and JSON containing `inventory` and `triples`. The generator may retain endpoints, numeric-to-symbol mappings, seeds, and transformation maps in a private scoring record. The decoder must accept **only observations**, and independently infer its answer. Never pass private metadata into the prompt builder or the decoder.

Use distinct implementations for generation, decoding, and validation where their independence matters:

- Main generation: enumerate ordered addend pairs `a <= x < y`, retaining `x+y <= n`, then rename.
- Production certificate validation: validate the value domain, check each listed triple, and use the exact edge count below.
- Independent tiny check: enumerate sorted three-element subsets and test `x+y=z`; enumerate degree-preserving bijections without calling the theorem or production decoder.

Sharing JSON readers and basic types is acceptable. Reusing one edge-generation routine in every correctness path is not sufficient independent validation.

### 3.2 Complete dictionary validation, including absent triples

For a proposed dictionary `f`:

1. Require exactly the inventory's keys, all with actual integer values. Reject JSON booleans, floats, missing or extra keys, and duplicate keys.
2. Require distinct values forming exactly one consecutive interval, with minimum at least 6 and at least two values.
3. For each observed triple, sort its assigned values and test the two-smallest-sum equality.
4. Compute `k = max(f.values()) - 2*min(f.values())` and require the number of distinct input triples to be

   \[
   e(k)=\begin{cases}0,&k\le0,\\
   \left\lfloor (k+1)^2/4\right\rfloor,&k\ge1.
   \end{cases}
   \tag{7}
   \]

Equation (7) is a planning derivation: at sum `2a+s` there are `ceil(s/2)` distinct-addend pairs, so sum this quantity for `s=1,...,k`. Every checked observed edge belongs to the expected relation; equal cardinality then proves equality, including all absences. Reject duplicate input triples **before** converting records to a set. Otherwise a count-based validator can be fooled.

During correctness checks also compare against an independently generated full expected edge set. There is no need to materialize all `m choose 3` nonedges in production.

Two distinct dictionaries passing this validator constitute an ambiguity certificate, even if neither equals the generator's dictionary. One passing dictionary proves only possibility. A valid dictionary plus the established uniqueness theorem proves unique recovery in this family; the response does not need a free-text proof.

### 3.3 Decoder pseudocode

Implement these operations directly; use exact equality and raise distinct internal errors for invalid input, a failed promise, and resource exhaustion.

```text
decode(inventory, triples):
    validate_input_syntax_without_silently_deduplicating()
    if triples is empty:
        construct witnesses using starts max(6,m-1) and max(6,m-1)+1
        validate both; return AMBIGUOUS, witnesses, fixed_values={}

    compute degrees including zero-degree inventory members
    infer a=m-1-max_degree and n=2*m-2-max_degree
    require a>=6 and n=a+m-1

    if n>=3*a-1:
        f = decode_unique(a,n,inventory,triples)
        validate f; return UNIQUE, [f], fixed_values=f
    else:
        f = decode_ambiguous(a,n,inventory,triples)
        g = f with two isolated values exchanged
        validate f and g; require f!=g
        compute exact fixed values from Section 2.6's orbit table
        return AMBIGUOUS, [f,g], fixed_values
```

`decode_unique` must implement the following boundary details, not rely on finding an absent standalone source file:

1. Set `b=(n+1)//3`. On the live induced hypergraph, repeatedly remove the unique maximum-degree vertex for values `a,...,b-1`, recording each assignment. A tie or inconsistent degree is a failed promise, not demonstrated ambiguity.
2. In the remaining graph require unique first and second maxima of degrees `n-2b` and `n-2b-1`. Assign their symbols to `b,b+1`.
3. Their unique common triple supplies the symbol for `2b+1`.
4. Form the union of their links. Set `t=n-3b`, which must be `-1,0,1`. Remove the two anchors. Also remove the common-triple center when `t<1`. When `t=-1`, identify the unique remaining zero-degree symbol, assign it `2b`, and remove it from the path search.
5. Require the remaining graph to be a connected path with exactly two endpoints and all other degrees two. Traverse from either endpoint. The expected value sequence is the concatenation of `(2b+j,b+j)` for `j=2,...,b+t` inclusive.
6. Try both path directions. Preserve already assigned values, including the center when `t=1`; do not silently overwrite a conflicting anchor. Restore peeled assignments and validate the entire dictionary against the **original** observations. Exactly one candidate must survive.

`decode_ambiguous` must require the isolate count and active-support size in (5). For `k<=3`, try all active-support bijections to the explicit canonical active values, with a fixed arbitrary isolate assignment. For `k>=4`, identify the anchors, center, and path (6), test both directions, and require respectively two or one accepted active labelings for `k=4` or `k>=5`. Choose a valid dictionary only after full validation.

Under the clean-family contract these algorithms terminate without a hidden isomorphism oracle. For inputs outside the family, failure means the contract was not validated. Do not claim a general theorem about noisy data. A damaged table can coincidentally be a valid table for a different interval; the observation cannot reveal its generating history.

### 3.4 Independent correctness work required before any model comparison

Use ordinary `unittest`; these are scientific correctness checks, not redundant tests of low-impact formatting.

| Check | Required evidence and pass condition |
|---|---|
| Source reproduction | Reproduce 90 opaque-label cases: `a=6..15`, `n=3a-1..3a+7`; every recovered dictionary and inferred endpoint correct |
| Complete decoder sweep | `a=6..20`, `n=a+1..3a+6` inclusive: 480 intervals; fresh deterministic renamings, both branches, all returned certificates valid |
| Degree and endpoint audit | Independently enumerate triples for `a=1..30`, `n=a+1..4a+10`; 1,695 intervals; (3) correct and (2) correct whenever nonempty |
| Tiny exact search | All `H[6,n]`, `n=7..19`, plus `[7,18]`, `[7,19]`, `[7,20]`, `[8,21]`, `[8,22]`, `[8,23]`; compare complete orbit partitions and counts with the production oracle |
| Independent factorial sanity check | For `[6,13]`, enumerate all `8!` permutations once; confirm 720 automorphisms, including isolate permutations |
| Ambiguous-path extension | `a=6..40`, `k=4..a-2`; 630 cases; isolate set/path exactly correct and reversal compatible iff `k=4` |
| Edgeless cases | `m=2,8,12,31`; verify two different starts and prove no fixed values using (4), without finite endpoint exhaustion |
| Metamorphic checks | Rename all symbols; reorder inventory, triples, and triple members; pull results back and check unchanged truth and supported-value sets |

Factor isolate permutations in the tiny exact search, but enumerate **all** active automorphisms when computing orbits. A search capped after two solutions cannot certify fixedness. Check the sentinel orbit table in Section 10. Use a 10-second per-case watchdog for independent search; a timeout is an incomplete check. Do not convert it to a uniqueness result.

Mutation tests must reject: a repeated inventory member, a duplicate triple, a triple with repeated/unknown members, a duplicate dictionary key, equal values, a nonconsecutive or scaled dictionary, incorrect addition, two identical ambiguity witnesses, and a false partial fixed-value claim. Include a 12-symbol observation with just one triple and a proposed `[6,17]` dictionary that realizes that triple: positive-edge checking alone accepts it, but the additional eight expected triples make it invalid. This catches omission of negative evidence.

If any mathematical mismatch survives a bounded implementation repair, stop model evaluation and retain the smallest reproducible counterexample. Determine whether the defect is in the implementation, source theorem, or new orbit derivation before interpreting anything empirically.

## 4. Model configuration, response contract, and prompts

### 4.1 Fixed local candidate

Use `mlx-community/Qwen3-4B-Instruct-2507-4bit`, derived from `Qwen/Qwen3-4B-Instruct-2507`, on native Apple-silicon Python with MLX-LM. The source model has 4.0B parameters and supports only non-thinking mode; its card states that `enable_thinking=False` is unnecessary. The conversion's card/configuration specify 4-bit weights with group size 64 [R9–R10]. This is a deliberately compact fixed candidate, not a claim that it is the strongest available local model.

Pin `mlx-lm==0.32.2`, whose package requires Python at least 3.11. Prefer native arm64 Python 3.11 or 3.12 already available on the machine. The release corresponds to source commit `93e6d4a6447b3080861985947f0820e76fcb957d` [R11]. Resolve and record all transitive versions once; freeze them before experiments.

The full immutable model revision was **not resolved during planning**. Resolve it during setup, then download that exact snapshot:

```python
from huggingface_hub import HfApi, snapshot_download

repo = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
revision = HfApi().model_info(repo).sha
assert len(revision) == 40
snapshot_path = snapshot_download(repo_id=repo, revision=revision)
```

Record the returned full revision and SHA-256 of the weight, configuration, tokenizer, chat-template, and license files. Thereafter load only that local snapshot. Hub documentation supports version-specific downloads [R14]. If the repository cannot be accessed or the selected checkpoint cannot run within the caps, complete the exact diagnostic and report the model experiment blocked or infeasible. Do not silently substitute a different model, use hosted inference, or change quantization.

Set batch size one, explicit greedy sampling (`make_sampler(temp=0.0)`), a maximum of **1,536 new tokens**, and a maximum of **8,192 total tokens after chat templating**. Use the documented MLX-LM sampler interface [R11]. Use no tools, retrieval, feedback loop, few-shot solved examples, grammar-constrained decoding, rotating/truncated KV cache, or cross-case conversation state. Start each case with a fresh context and reset its KV cache. Record actual dtypes and backend rather than assuming a model-card dtype describes every loaded tensor.

The coding agent's hosted reasoning configuration is not the subject model and is not an experimental inference budget. Do not replace local model responses with answers from the coding or research agent.

### 4.2 Response JSON

Require exactly three top-level fields:

```json
{"status":"unresolved","assignments":[],"fixed_values":{}}
```

The example is the unresolved shape, not a recommended response. The schema is:

- `status`: exactly `unique`, `ambiguous`, or `unresolved`.
- `assignments`: a list of complete symbol-to-integer dictionaries. Exactly one for `unique`; exactly two different dictionaries for `ambiguous`; zero or one for `unresolved`. Every supplied assignment asserts compatibility with the complete declared model. An assignment under `unresolved` asserts compatibility but not uniqueness. An ambiguous response additionally asserts that its two witnesses are different.
- `fixed_values`: a possibly empty symbol-to-integer dictionary containing only values asserted to hold in **every** compatible interpretation. Empty means no partial claims offered, not “nothing can be learned.” It may repeat entries from a complete assignment.

Do not require an exhaustive list of fixed symbols. Evaluate each volunteered claim separately. Do not infer certainty from assignments explicitly marked as possibilities. No free-text rationale is required or scored; this avoids a second semantic judge and preserves output space for certificates.

The deterministic parser may remove surrounding whitespace and one complete outer Markdown JSON fence. It must reject trailing prose, multiple candidate objects, duplicate keys, undeclared fields, invalid types, and incomplete JSON. Status may be retained only from a syntactically valid single top-level JSON object with exactly one enum-valued top-level `status`, even if other schema fields fail. Truncated JSON, duplicate keys, trailing prose, multiple objects, and matches inside nested values leave status unknown; never use regex recovery. Do not use an LLM to repair or judge outputs. Preserve raw text and distinguish parsing, schema, certificate, and semantic failures. Section 6 prevents malformed responses from appearing to improve certainty merely by escaping classification.

### 4.3 Shared task text

Use the following system instruction for every arm:

> Solve the mathematical interpretation task using the supplied observations and declared model. Return one JSON object in the required response format. Do not use external information or tools.

Use the following common user preamble before the arm-specific paragraph and observation JSON:

> Every symbol represents a different integer in an unknown consecutive interval from a to n, where a is at least 6 and n is at least a+1. Every integer in the interval has exactly one symbol. The inventory is complete, including symbols in no triple. The list contains every unordered triple of three distinct symbols for which one represented integer is the ordinary sum of the other two, and contains no other triples. An absent triple therefore does not satisfy this relation. A triple does not identify its sum symbol. Symbol spellings and all presentation orders carry no intended numerical information. Determine whether the complete dictionary is uniquely established. Return status unique with one complete dictionary if it is; status ambiguous with two different compatible complete dictionaries if ambiguity is established; or status unresolved if you have not established either. A possible dictionary is not proof of uniqueness, and failure to solve is not proof of ambiguity. You may report individually guaranteed values in fixed_values even when the whole dictionary is ambiguous. Return exactly status, assignments, and fixed_values as described by the response schema. Use integer values and the exact inventory symbol strings.

Append a concise mechanical description of the three fields from Section 4.2 identically to all arms, including the sentence: “Every supplied assignment asserts that it satisfies the complete declared model; an assignment under unresolved asserts compatibility but not uniqueness.” The same full observation appears in every arm; no endpoint, theorem, source filename, case ID, seed, answerability label, or hidden dictionary appears.

### 4.4 Fixed prompt configurations

| Arm | Role | Additional paragraph |
|---|---|---|
| `N` | Simple reference | No additional paragraph. |
| `C` | Main strong caution baseline and search-procedure ablation | The caution paragraph below. |
| `W` | Witness-search intervention | The witness paragraph below. |

**Caution paragraph (`C`):**

> Make only claims that the observations establish. Carefully distinguish a possible assignment, a uniquely established assignment, and an unresolved analysis. Review the numerical values, inventory coverage, and every declared assumption before choosing your final status. Check that any complete dictionary respects the full relationship table, including absent triples. Do not let familiarity with a pattern, apparent plausibility, or pressure to finish substitute for sufficient evidence. If your analysis is insufficient, explicitly report unresolved. Preserve useful supported conclusions, including individually fixed values, without extending them to claims you cannot justify.

**Witness paragraph (`W`):**

> Before choosing your final status, actively search for a different complete dictionary compatible with the same observations. Start from a candidate assignment and try changing or exchanging numerical roles, checking that the entire declared model and all present and absent triples still hold. If you find two different valid dictionaries, report both as ambiguity witnesses. If you establish that exactly one dictionary works, report it as unique. Failure to find an alternative is not proof that none exists; report unresolved when necessary. Preserve individually fixed values even if other assignments remain interchangeable.

The main comparison is `W-C`; `N` is not the sole comparator. All arms already know the task requires witnesses. The treatment adds a procedure, not privileged information. Record actual token lengths of `C` and `W`; both are short and have similar word counts. Do not tune either on final outcomes or pad it with arbitrary material. Their input-token costs are part of the configuration cost; equal output/time caps do not imply equal realized computation. No arm gets additional samples or repair turns.

## 5. Smallest sufficient experiment set

### 5.1 Development and calibration data

Use separate seed namespaces for `development`, `evaluation`, `stress`, and `analysis`, derived from a recorded master seed of `20261101` using SHA-256 of the namespace and case coordinates. Do not use Python's process-randomized `hash()` for seeds. Explicitly sort sets before any seeded shuffle.

The six mechanism-pilot structures are fixed:

| Gold type | Pilot intervals |
|---|---|
| Unique | `[6,19]`, `[6,20]`, `[6,21]` |
| Ambiguous | `[8,20]`, `[9,23]`, `[10,26]` |

Use two independently rendered views of each and all three arms: **36 calls**. These intervals do not occur in the main panel or the locked stress set. They exercise both boundary decoding and peeling, and ambiguous instances with different orbit structures.

Reserve `[6,17]`, the smallest unique instance, and `[7,15]`, a nonempty ambiguous instance, as two additional **development diagnoses**. Only if all 36 pilot calls have zero useful success and at least 90% are syntactically and schema valid, run one fresh view of each diagnosis in all three arms: six calls. If any of these succeeds usefully, proceed with the fixed main protocol. If none does, stop with zero observed useful performance on the tested pilot/diagnostic cases and an inconclusive primary comparison. Neither diagnosis appears in the main or stress panels. Do not tune prompts or merge these diagnostic outcomes into the primary score. The pilot has a maximum of **42 calls**.

For backend calibration use `[6,19]` and `[10,41]`, each in all three arms: six calls, plus one identical repeat of each for a total of **12 calibration calls**. The latter is an off-panel, relatively dense 32-symbol case; it conservatively stresses prompt processing. Use calibration answers only to diagnose the interface and resource use, not to tune the scientific hypothesis or select the better prompt.

The baseline and treatment text, model, schema, generation settings, and practical effect thresholds are fixed in this brief. Ordinary fixes to parser, generator, resource monitoring, or backend invocation are allowed before the lock. Do not launch a prompt search. A single bounded repair of an actual implementation defect is allowed; all failed calls remain in the ledger. If a wording or algorithm change becomes necessary, record it as a protocol deviation and do not present the resulting comparison as the original locked test without a fresh uncontaminated panel.

### 5.2 Main paired panel

The default panel has **20 size blocks**, `m=13,...,32`. The smallest unique 12-symbol case is reserved for the development diagnosis above. For each main size generate:

\[
a_U=\lfloor m/2\rfloor,\quad n_U=a_U+m-1,
\qquad
a_A=\lfloor m/2\rfloor+1,\quad n_A=a_A+m-1.
\tag{8}
\]

The first interval is unique and the second ambiguous. This yields 40 distinct nonempty structures. Because size and maximum degree determine endpoints, these distinct intervals are not duplicate isomorphism classes. Matching size prevents inventory length alone from supplying the class; relational counts and degrees remain valid evidence.

| Size | Unique case / edge count | Ambiguous case / edge count |
|---:|---|---|
| 13 | `[6,18]` / 12 | `[7,19]` / 9 |
| 20 | `[10,29]` / 25 | `[11,30]` / 20 |
| 32 | `[16,47]` / 64 | `[17,48]` / 56 |

These counts are exact integer calculations, not runtime estimates. The panel deliberately concentrates around the identifiability boundary. It does not represent all intervals. All ambiguous main-panel cases have fixed nonisolated values and interchangeable isolates. Nonisolated ambiguity, sparse cases, and empty cases are covered by correctness sentinels and the separate stress set; do not generalize the main panel's orbit pattern to all ambiguous inputs.

Create three controlled views per structure:

1. **Base:** fresh opaque symbols; independent random inventory order, triple order, and order within each triple.
2. **Order-only:** preserve the base symbol-to-value mapping; independently reorder all three presentation components.
3. **Rename-only:** uniformly rename every symbol in the base view, preserving every list position and consistently changing every occurrence. If a later extension adds a queried symbol, it must undergo the same renaming.

These are related presentations of the same structure, **not three independent mathematical examples**. Compare their pulled-back outputs to isolate ordering and spelling sensitivity. In ambiguous cases two valid returned witnesses need not be the same pair after pullback; compare their validity, status, and supported claims, not equality to one canonical witness pair.

All three arms receive all views: `20 * 2 * 3 * 3 = 360` calls. If resource calibration rejects this panel, the only predeclared reduction is `m=13,...,24`, giving 12 blocks, 24 structures, and **216 calls**. Preserve both statuses, all three views, and all arms. Select the panel from token/memory/time measurements before examining held-out model outputs. If even the reduced panel cannot fit under a reliable measured projection, stop the model comparison as practically infeasible under these caps. If calibration itself remains uncertain, report that uncertainty instead. Do not shrink below the meaningful unique regime, remove the ambiguity task, or change the output cap to manufacture a result.

### 5.3 Opaque symbols and leakage controls

For a base view, draw `m` distinct strings of the form `z` followed by four uniformly sampled lowercase ASCII letters, with rejection only for duplicate strings. Sample the symbol-to-value bijection independently of spelling and all presentation permutations. The string pool is independent of endpoints and truth class. Token lengths may vary; record them, but do not condition the mapping on numeric values. The rename-only view uses a disjoint freshly sampled pool and a uniform bijection.

Check the following before lock:

- The prompt builder accepts only public observation fields, the shared contract, and the arm text. Private endpoints, map, truth, seed, filename, and structural IDs cannot be serialized into it.
- Every inventory symbol is preserved, including zero-degree symbols. All triples and only triples are included. No lexicographic or insertion order inherited from numeric generation remains in the base rendering.
- A symbol's character suffix, numeric rank, token count, and inventory position have no intentional relation to its value. Inspect rank/position associations over at least 10,000 inexpensive generated renamings as a pipeline diagnostic; this is not an empirical language-model sample or a proof of zero correlation.
- True class is balanced within every size. No arm sees an explanation identifying a case as a test of ambiguity or uniqueness.
- Private scoring data are separate from serialized observations, and final model prompts are archived verbatim.

Do not attempt to hide mathematical statistics or forbid a learned algorithm. Fresh names prevent direct dictionary lookup, not prior exposure to the theorem, task family, or general arithmetic. Record this residual contamination limit.

### 5.4 Locked stress set and scorer controls

After a completed main comparison, run the following six additional structures with two fresh jointly renamed/reordered views and all arms: **36 calls**, subject to the separate 15-minute ceiling. Their results are secondary and cannot rescue a failed primary hypothesis.

| Structure | Purpose |
|---|---|
| `[6,7]` | Minimal edgeless inventory; two endpoints are not identifiable |
| `[11,22]` | Edgeless 12-symbol case; many output entries but no relational constraints |
| `[6,13]` | One triple, active and isolated orbits, no fixed values |
| `[6,16]` | Nonisolated ambiguity sentinel |
| `[9,28]` | Unique boundary residue `n=3a+1` |
| `[6,26]` | Unique case requiring peeling beyond the boundary |

The pilot-floor branch uses the separately reserved nonempty development diagnoses in Section 5.1, not these stress cases. The stress set remains held out from model development and cannot rescue a failed primary comparison.

The following deterministic controls are mandatory and cost no language-model calls:

| Control response | Required scoring behavior |
|---|---|
| Exact decoder output on every clean case | Useful success 100%, unsupported claims 0 |
| Always `unresolved`, no assignments or fixed claims | Useful success 0; explicit unresolved, not demonstrated ambiguity |
| Hidden generator dictionary with `unique` on every case | Correct on unique cases; unsupported uniqueness on every ambiguous case even when all values match the hidden map |
| Correct class only, without required dictionaries | No complete-success credit |
| Valid ambiguous witnesses differing by an allowed automorphism | Full ambiguity credit even if neither matches the generator |
| Two identical or partially invalid witnesses | No demonstrated-ambiguity credit |
| Whole dictionary ambiguous but some singleton orbits | Individually correct fixed claims receive partial credit; variable values do not |

The exact solver is also the strongest feasible computational reference for this declared model. Time it on the same observations, including parsing and certificate construction. Its success is expected from the mathematics and cannot be counted as a learned-model improvement. If the practical use case permits this exact algorithm, direct use of the solver is preferable to relying on unaided numerical guesses. The empirical study instead measures how faithfully prompt-only configurations handle the distinction.

## 6. Scoring, uncertainty, and the locked research decision

### 6.1 Retain the outcome distinctions

For every response preserve separate fields for declared status, parse/schema validity, dictionary validity, gold class, fixed-claim correctness, and resource failure. Do not collapse them prematurely into one accuracy number.

| Metric | Definition |
|---|---|
| `R_U` | On true unique cases: correct unique status and one valid complete dictionary |
| `D_A` | On true ambiguous cases: correct ambiguous status and two distinct valid complete dictionaries |
| `Q` | Balanced useful complete rate `(R_U + D_A)/2`; distinct from caution arm `C` |
| `U_A` | On true ambiguous cases: explicit claim of `unique`, regardless of whether its dictionary matches hidden labels |
| Wrong unique-dictionary rate | On true unique cases: `unique` with an invalid full dictionary; distinguish this from false uniqueness |
| False-ambiguity rate | On true unique cases: `ambiguous`; distinguish it from unresolved analysis |
| Undemonstrated ambiguity | Gold ambiguous and status ambiguous, but without two valid distinct complete witnesses |
| Explicit unresolved rate | A mechanically recognized `unresolved` declaration; do not relabel timeouts or malformed outputs as voluntary abstention |
| `B` | Case-level indicator of any established unsupported/false claim: false uniqueness, false ambiguity, any supplied assignment failing complete compatibility under any status, an incorrect guaranteed fixed value, or an ambiguous response presenting identical witnesses as different |
| Format and resource failures | Parse failure, schema failure, token limit, timeout, backend error, or OOM; report each and their union |
| Partial-value performance | Number of correct fixed claims, number of incorrect fixed claims, claim precision, and claimed-symbol coverage; zero claims gives undefined precision, not 100% |

A valid full dictionary on an ambiguous case is a correct **possibility**. It becomes unsupported uniqueness only when the response asserts it is unique. A wrong fixed claim includes assigning the generator's value to a symbol that can legitimately vary. Explicitly test that a wrong dictionary under `unique`, an invalid witness under `ambiguous`, and an invalid possible dictionary under `unresolved` each set `B=1`; two identical ambiguity witnesses also set `B=1`. Missing or unparseable material follows the unknown-response rules unless an offending claim is already established.

Report declared decision coverage and useful verified coverage separately. If reporting risk among definite attempts, state its denominator and mark the result undefined when there are no such attempts. A low error rate purchased by universal unresolved responses does not satisfy the project decision.

### 6.2 Conservative treatment of malformed and missing responses

All scheduled calls stay in the denominator for useful performance, with failure scoring zero useful success. Preserve failures and partial output. A timeout or malformed object does not establish that the model made a definite claim; do not fabricate one.

For `U_A` and `B`, report **lower and upper bounds** where semantic grading is unavailable:

- The lower bound counts mechanically established offending claims.
- The upper bound also counts each ungradable response as potentially offending.
- A parsed top-level `status=unique` on an ambiguous case is an established false-uniqueness claim even when another schema field is invalid. When no reliable status can be extracted from the permitted parser, the response is unknown for `U_A`.
- For `B`, a parse/schema failure that prevents complete inspection remains unknown unless an error has already been established. A valid structured unresolved response with no claims is a known nonoffense.

The conservative observed reduction is `U_A(C)_lower - U_A(W)_upper`. Use the same worst-case direction for the broad claim-error guard `B(C)_lower - B(W)_upper`. This prevents an arm from winning by producing more ungradable output. Also show the raw known-status counts so readers can distinguish actual certainty errors from conservative missingness bounds.

Take a fixed, blinded diagnostic sample of up to 12 malformed/truncated responses, selected by the smallest hash of their anonymous raw-record IDs, and inspect why parsing failed. Do not repair primary outcomes or add an unbudgeted judge model. The coding agent may describe obvious format failure patterns; do not present its retrospective interpretation as independently validated semantic labels.

### 6.3 Aggregation and units

For each size block compute rates over its three views separately for its unique and ambiguous structures. Compute paired `W-C` differences within the block, then average blocks equally. The two statuses, every view, and every prompt must remain together in paired analyses. Report `N`, `C`, and `W` by true class, size, and view, as well as the macro result.

The main estimand is the **realized performance difference on this fixed finite panel**, using a declared equal mixture of the three presentation types and greedy decoding. It is a screening experiment. It is not a random sample of all integer intervals, and it does not justify a population-level claim about all fresh names or models.

Use a 10,000-draw paired size-block bootstrap with a separate fixed analysis seed to display sensitivity to the panel's composition. Resample whole size blocks, never triples, individual symbols, or the three dependent views as independent observations. Label these intervals as empirical block-resampling sensitivity intervals, not distribution-free guarantees over unseen intervals. Also show all block differences; a plot should not hide adverse blocks.

A degenerate zero-width bootstrap interval is not proof of equivalence or absence of future failures. As a planning illustration, even zero losses among 20 independent Bernoulli trials leaves a one-sided 95% loss-probability bound of `1 - 0.05**(1/20)`, approximately 13.9%; at 30 trials it is approximately 9.5%. Those are not confidence bounds for this panel's utility difference, but they show why a five-point general noninferiority claim is not supported merely by seeing no losses. The reduced panel is still less informative.

Do not run significance tests on repeated deterministic oracle outputs. Calibration duplicates diagnose backend stability; they are not extra independent benchmark samples. If nominally identical greedy runs differ, document the differences and maintain the fixed first-run rule. Do not search seeds for favorable answers.

### 6.4 Prespecified screening rule

Recommend `W` for further application-data testing only if the selected main panel is complete, all validity gates pass, and all of the following hold:

1. The conservative observed reduction in `U_A` is at least **0.10**.
2. `R_U(W) >= R_U(C) - 0.05` and `D_A(W) >= D_A(C) - 0.05`.
3. Both `R_U(W)` and `D_A(W)` are at least **0.20**. Avoiding mistakes without useful success in both tasks is insufficient.
4. The conservative broad claim-error reduction `B(C)_lower - B(W)_upper` is at least **0.05**. On the balanced panel, a ten-point ambiguous-case improvement corresponds to five overall points; this guard prevents merely moving errors into bad witnesses or partial claims.
5. The conservative `U_A(C)_lower - U_A(W)_upper` reduction is nonnegative in each of the three separately reported view types. A fully graded reversal, or a negative optimistic bound, establishes presentation sensitivity. Bounds that straddle zero leave presentation robustness inconclusive; either case withholds an unqualified recommendation.

These are **planning policy thresholds**, not established application tolerances. Ten points selects a material change in this diagnostic; five points limits the sacrificed utility; a one-in-five success floor prevents a nearly useless abstainer from qualifying. Do not alter them to fit observed outcomes. If `C` has too little unsupported uniqueness for a ten-point reduction, report a baseline floor for this criterion; a smaller effect can be descriptive but does not satisfy the prespecified screen.

A pass supports the statement “promising on the tested finite panel and worth further testing.” It does **not** establish a five-point noninferiority guarantee on future data. If the sensitivity intervals include materially worse utility or no certainty improvement, make that uncertainty prominent and describe broader effectiveness as inconclusive. There is no automatic upgrade to a population claim.

A screen failure means **no recommendation under this policy**, but its scientific interpretation depends on why it failed. A fully graded completed panel that fails a necessary criterion, or a failure even under optimistic admissible error bounds, provides an informative negative for the corresponding finite-panel criterion. Compute optimistic reductions as `U_A(C)_upper - U_A(W)_lower` and analogously for `B`. If conservative bounds fail but optimistic bounds could pass, the certainty effect is inconclusive because grading is insufficient. A useful-performance failure is a definite observed protocol failure because invalid/missing constructions score zero by definition, but dominant format failures should be described as protocol usability rather than inability to reason. A floor discovered only in the pilot or an incomplete locked panel cannot reject the held-out hypothesis. Report the failed criterion and its evidence rather than using one generic failure label.

### 6.5 Required diagnostic displays

Produce only what helps the decision:

- A rate table with numerator/denominator, bounds for ungradable responses, and `R_U`, `D_A`, `U_A`, `B`, unresolved, and failure rates for each arm.
- A plot of useful complete rate against unsupported-uniqueness rate, including all-unresolved and exact-solver controls; show bounds and identify the finite panel.
- A paired per-size plot of `W-C` useful performance and certainty errors, separated by view if necessary.
- A compact latency/token/memory table and a partial-value claim table. Do not give symbols or triples the status of independent trials.

No dashboard, service, or elaborate visualization framework is needed. Save ordinary CSV tables and Matplotlib PNG or PDF figures, plus the raw measurements needed to recreate them.

## 7. M3 Max feasibility, cumulative budgets, and gates

### 7.1 Inspect the actual target

Before installing or timing anything, record `sw_vers`, `uname -m`, `sysctl -n machdep.cpu.brand_string`, `sysctl -n hw.memsize`, Python version/architecture/executable, disk free space, available system memory, swap use, and the intended backend. MLX's installation documentation requires Apple silicon and macOS at least 14; the selected MLX-LM package imposes the stricter Python-at-least-3.11 requirement [R11–R12]. Verify Metal availability and `mx.default_device()`/device information through the installed API. A CPU fallback must be explicit and does not count as a verified M3 Max Metal run.

Record power connection, low-power mode, and competing workloads. Do not infer RAM from the chip, terminate other user applications, increase OS wired-memory limits, or change system settings to rescue the experiment. If the Mac is unavailable, the coding agent may complete source and exact correctness work elsewhere, but must label that environment and leave Mac-specific performance and local-model validation unperformed.

### 7.2 Memory, disk, and numerical conventions

| Resource | Ceiling or rule |
|---|---|
| Single inference worker | One loaded model, one sequence, no concurrent model jobs |
| Inference working-set policy | `min(12 GiB, 0.5 * physical RAM)`; retain at least 8 GiB of available system headroom |
| Exact reference/checker process | 2 GiB; 10 seconds per independent tiny search; 1 second per ordinary main-panel decode after warm-up |
| Project disk | 12 GiB including environment, model, caches, and results; require at least 14 GiB initially free |
| Input + output | 8,192 tokens total; 1,536 new tokens; no truncation of observations |
| Per inference attempt | 120 seconds including prompt processing and generation, excluding an amortized model load that is separately recorded |
| Total inference attempts | 456, including calibration, repeated calls, failures, pilot diagnoses, stress cases, and permitted repairs |
| Experimental elapsed compute budget | 6 hours cumulatively across all gates, pauses aside; reserve analysis and evidence preservation time |

The conversion is reported at roughly 2.26 GB on the model repository, but the working set is larger [R10]. A full BF16 KV cache using the documented 36 layers, eight KV heads, and head dimension 128 would require

\[
2\cdot36\cdot8\cdot128\cdot2=147{,}456\text{ bytes/token},
\]

or about **1.125 GiB at 8,192 tokens**, excluding weights, logits, intermediate activations, Python, and allocator cache. This is a derived planning estimate, not a measurement. Inspect actual cache types and measure actual memory.

Use an external watchdog sampling the inference process and system approximately once per second. Record sampled RSS or physical footprint, system available memory, swap change, and MLX active, cache, and peak-active counters when supported. MLX's memory limit is a guideline, not an enforced process ceiling [R13]. Do not add overlapping RSS and MLX counters as if they were independent allocations. Stop on either reported process footprint or MLX active-plus-cache exceeding the policy cap, sustained available headroom below 8 GiB, or at least 512 MiB of new swap within a block. These are operational protection rules, not performance theorems. Retain partial output and the stop reason.

Use exact integers for all mathematics, with no floating-point equality tolerance. Quantization and floating-point inference affect the candidate model, not the oracle. Do not change quantization, cache precision, or decoding midway to fit memory. If the selected configuration cannot fit, the empirical comparison is unavailable under this protocol.

### 7.3 Timing boundaries and calibration

Measure separately: initial import/load and first warm-up; prompt formatting/tokenization; prefill; generation; parse/validation; and end-to-end case latency. Synchronize the actual MLX generation stream/device through the installed API before ending timing, and force lazy results to materialize. Archive the exact invocation and backend version. Stream responses so a watchdog can preserve partial output on termination.

Model weights may remain loaded across cases, but no prompt/KV state may be reused. Balance and randomize arm order within blocks using a locked schedule so thermal drift and cache effects do not systematically favor an arm. Report both cold-start total cost and warm repeated-use cost, including model loading divided by the stated number of cases only when discussing amortization. Include all preprocessing, gold computation, calibration, and failed calls in total project cost.

Before the main run, tokenize every planned prompt and a complete valid reference response. Require each prompt plus the 1,536-token cap to fit 8,192. Require each compact reference response, including all oracle fixed values, to fit within 80% of the response cap. This is a capacity check, not a hint shown to the model. If the full panel fails, evaluate the predeclared reduced panel; never omit dictionary entries or triples.

Use calibration measurements conservatively. Estimate a capped call as the larger of the maximum observed all-in latency and `maximum measured prefill + 1536 / minimum reliable decode_tokens_per_second`, plus parsing/IO. Use decode-rate measurements from streams with at least 64 generated tokens. If all calibration responses are shorter, spend at most one of the six shared extra-call slots on a **resource-only** generation requesting at least 256 words while retaining a prompt at least as long as the largest calibration prompt; the same 1,536-token and 120-second limits apply. Exclude this output from scientific scores. If this still fails to provide a reliable rate, report calibration as inconclusive, not measured hardware infeasibility; finish the exact diagnostic and report without a large model run. Multiply a reliable call estimate by the proposed remaining call count and a **1.25 scheduling margin**. Compare with the remaining main allocation after reserving later stages and the global reserve. This is intentionally conservative; observed performance may permit the full panel, but no token-per-second figure is assumed in advance.

At 30 seconds per call, 360 calls would take three hours. At the output cap they generate at most 552,960 tokens. These are arithmetic planning scenarios, not measured M3 Max throughput. The maximum 456 attempted calls would generate at most 700,416 tokens. Use actual timing rather than either scenario to admit the locked run.

### 7.4 Evidence gates and stop rules

The compute allocation below sums to **360 minutes**. The repair allowance is shared across all stages; unused time does not authorize a larger scientific matrix or extra tuning. Count every failed attempt and fallback cumulatively.

| Gate | Question and minimum work | Ceiling | Decision and next action |
|---|---|---:|---|
| G1: constructive and tiny correctness | Inspect environment; implement syntax, validator, both decoder branches, independent checks, orbit tests, and deterministic scorer controls | 30 min compute, 0 model calls | Continue only with zero unexplained correctness mismatches. Repair a concrete bug within the shared reserve. A persistent mathematical discrepancy stops empirical work and yields a counterexample report. |
| G2: local feasibility | Resolve immutable weights; verify backend/dtypes; run 6 calibration calls and their 6 identical repeats; measure full-prompt tokens, memory, and latency | 20 min, 12 calls; one resource-only extra call may use the shared reserve | Admit full or reduced panel using Section 7.3. Separate dependency blocks, measured infeasibility, and inconclusive calibration; complete exact work and report the actual reason. |
| G3: mechanism pilot | Run the fixed 6 structures, 2 views, 3 arms; inspect schema usability, useful construction, and bottlenecks | 30 min, 36 calls plus at most 6 reserved diagnosis calls | Continue if the interface is valid. Do not select a prompt by pilot gain. If every arm has zero useful complete success and at least 90% of outputs are syntactically and schema valid, run the two nonempty development diagnoses in Section 5.1. Continue if any succeeds usefully; otherwise stop with the primary effect inconclusive and the exact observed pilot limitation documented. If format/backend defects dominate, repair only an identified implementation fault or report a protocol-usability limitation. A noisy difference alone is no efficacy test. |
| G4: locked comparison | Freeze hashes/configuration/schedule; run every cell of the selected panel once | 225 min, 360 or 216 calls | Complete the selected panel without changes. If a limit is reached, preserve every scheduled/missing outcome and stop; the incomplete primary comparison is inconclusive. Do not drop slow sizes or only analyze successful calls. |
| G5: bounded robustness | Run the 6 fixed stress structures, 2 views, 3 arms | 15 min, at most 36 calls | Report secondary results separately. Budget expiry affects coverage, not primary denominators. No new experiment is admitted merely because it might reverse the result. |
| G6: diagnosis and report | Recompute scores from raw files; generate essential tables/plots; interpret and archive | 20 min compute, 0 model calls | Finish the report for the stages actually reached, including stopped or blocked branches. |
| Shared bounded repair reserve | Repair ordinary implementation/backend problems, preserve originals, and rerun only affected infrastructure checks | 20 min, at most 6 extra calls | No model timeout, invalid answer, or unfavorable result gets an extra semantic attempt. When exhausted, report the unresolved defect or limitation. |

The total maximum is `12 + 42 + 360 + 36 + 6 = 456` calls. Calibration repeats and conditional pilot diagnoses are already counted. A resource-only calibration fallback or smoke call uses the shared six-call reserve, reducing the slots available for infrastructure retries. Time ceilings still apply if a stage has unused call slots.

Install/download troubleshooting has a separate **60-minute elapsed ceiling** because network transfer is not experimental inference. Record it in total end-to-end cost. Implementation and analysis effort are estimated at **4–8 active hours**, separate from the up-to-six-hour experimental run; actual elapsed completion may span one or two working sessions. These are unmeasured estimates. Stop adding infrastructure when a fatal gate fails. Preserve enough time to explain the failure rather than spending the reserve on repeated rescues.

### 7.5 Locking, failures, and deviations

Write `protocol_lock.json` before any held-out response is generated. It must include the selected size range, all observation hashes, seeds, prompts, response schema, model full revision, dependency versions, numerical settings, metrics/thresholds, schedule, budget counters, and source-code commit/hash. Once locked:

- Do not change model, prompts, output/time caps, scorer rules, panel, or primary contrast in response to outcomes.
- A verified scoring bug requires rescoring **all** affected raw responses identically, with both old and corrected scores preserved. If it influenced generation or selection, the locked comparison is invalid; do not silently reuse it as confirmation.
- A model timeout, truncation, or invalid answer remains that scheduled attempt's outcome. Infrastructure retries are allowed only for a documented failure before usable model execution, at most six globally; retain both attempts and their costs.
- If the main run ends early, retain the full planned ledger with explicit missing-run reasons. Completed-block summaries are descriptive and cannot replace the complete primary panel.
- Any new scientific hypothesis or changed evaluation distribution is exploratory and requires a newly authorized, fresh confirmatory study; this brief does not authorize an indefinite rescue campaign.

## 8. Repository, interfaces, commands, and required artifacts

### 8.1 Minimal repository to create

Use ordinary Python scripts and local files. A suitable layout is:

```text
arithmetic_identifiability/
  README.md
  config.json
  requirements.in
  requirements.lock
  poc.py
  core.py
  independent_check.py
  runner_mlx.py
  score.py
  report.py
  prompts/neutral.txt
  prompts/caution.txt
  prompts/witness.txt
  tests/test_correctness.py
  inputs/
  data/public/
  data/private/
  results/
  report/
```

This is an intended layout, not an existing project. Keep mathematical code in `core.py`, the independent brute-force relation/search in `independent_check.py`, inference/watchdog code in `runner_mlx.py`, response grading in `score.py`, and analysis/report generation in `report.py`. `poc.py` is a small CLI, not a general orchestration framework. The source notebook can be archived in `inputs/` when available; the mathematics in this brief is sufficient to implement the specified mechanism without the chat or a missing companion script.

Define these interfaces:

| Interface | Input and output |
|---|---|
| `generate_case(a, n, render_seed)` | Public observation plus separate private generation record |
| `validate_observation(obs)` | Validated inventory/edge structure or explicit syntax error |
| `validate_dictionary(obs, mapping)` | Boolean validity plus specific failed constraint, with no hidden-label comparison |
| `decode(obs)` | Gold status, one/two validated dictionaries, exact fixed-value map, internal derivation metadata |
| `parse_response(raw)` | Strict response or parse/schema error; retains mechanically recoverable status for error bounds |
| `score_response(obs, gold, parsed)` | Separate claim, construction, fixed-value, and missingness fields |
| `run_case(prompt, locked_model_config, limits)` | Raw completion, token counts, termination reason, timing, memory, and backend identifiers |
| `analyze(locked_manifest, raw_records)` | Auditable per-case scores, paired aggregates, bounds, plots, and stage-aware outcome |

Do not let a response grader import the generator's hidden dictionary as its compatibility criterion. A hidden dictionary may be used as one sanity check on unique cases, never as the definition of a correct ambiguous answer.

### 8.2 Dependencies and intended command sequence

Use the standard library for the exact core and tests. Use MLX-LM, `huggingface-hub`, and `psutil` for the local runner; NumPy/Matplotlib for small analysis tables and plots. Do not add NetworkX, Z3, a database, a web service, or a second model unless a concrete defect makes the specified approach impossible; such a scientific-method change must be reported, not silently treated as the same protocol.

The following commands are **interfaces you must implement**, not commands verified during planning. Run from the new repository root using a compatible native Python. The package installation form is standard; actual resolution and hardware support remain G2 gates.

```bash
python3 poc.py inspect --config config.json
python3 -m unittest discover -s tests -v
python3 poc.py verify --config config.json

python3 -m venv .venv
.venv/bin/python -m pip install "mlx-lm==0.32.2" huggingface-hub psutil matplotlib
.venv/bin/python -m pip freeze --all > requirements.lock

.venv/bin/python poc.py inspect --config config.json
.venv/bin/python poc.py prepare --config config.json
.venv/bin/python poc.py calibrate --config config.json
.venv/bin/python poc.py pilot --config config.json
.venv/bin/python poc.py freeze --config config.json
.venv/bin/python poc.py run --lock results/protocol_lock.json
.venv/bin/python poc.py stress --lock results/protocol_lock.json
.venv/bin/python poc.py analyze --lock results/protocol_lock.json
.venv/bin/python poc.py report --lock results/protocol_lock.json
```

`inspect` must also work with standard-library Python before dependencies are installed, so environment constraints can be discovered first. `prepare` generates both development and held-out observations, computes exact gold, downloads the pinned model if necessary, and records hashes; it does not invoke the model. `calibrate` selects the permitted full/reduced panel from resources. `freeze` refuses to lock if correctness or resource gates fail. `run` refuses to execute when the lock differs from current artifacts. `report` must work after any stop, even when no main-run lock exists; provide `--results results/` for that branch. Do not require nonexistent model results to generate a negative or blocked report.

Provide a `poc.py smoke --config config.json` alias for one exact decode/score and one local inference on a development case. Charge its inference to the six-call shared repair allowance unless it replaces an already scheduled calibration call; do not double-count it as a new source of empirical evidence. Provide `poc.py reproduce-analysis --results results/` to regenerate tables and figures without model inference or network access.

Store source revision information and resolved dependency versions before inference. An already working system Python can run exact tests even when the local MLX environment is blocked. No access credential, proprietary dataset, or external inference endpoint is required by the design.

### 8.3 Configuration fields and execution discipline

`config.json` must explicitly contain the model repository and resolved revision, master seed and namespace rules, arm prompt files, greedy sampler, token caps, per-case and global limits, memory policy, development intervals, full/reduced size ranges, the fixed stress intervals, primary contrast, metric thresholds, bootstrap count, and output paths. Use a null/unresolved model revision only before `prepare`; `freeze` must reject it. This is a state transition, not an unresolved scientific choice.

Maintain an append-only JSONL attempt log with unique run/attempt IDs and a separate planned-call ledger. Write a start record before inference and an end record after it. A crash must remain visible. For each attempt archive the exact prompt, raw text, public observation digest, arm, view, model revision, completion/EOS/limit reason, tokens, times, memory samples or summaries, and budget counters. Flush partial streamed text periodically; avoid a large in-memory run accumulator.

The gold record includes inferred endpoints, validated witness dictionaries, exact fixed symbols/orbits, and source of the truth rule. The private generator record remains distinct. Save code/configuration hashes and a short deviation log. Do not publish model weight files inside the result archive; retain the pinned download recipe and hashes instead.

### 8.4 Final artifacts for the stages actually reached

Deliver a runnable repository and concise README once implementation occurs. At minimum preserve:

1. `RESEARCH_REPORT.md` with the project decision, exact tested scope, source status, stage reached, outcome classification, evidence, limitations, and one justified next step if any.
2. Code, tests, configuration, exact prompts, dependency lock, environment manifest, model revision/hash manifest, and regeneration commands.
3. Exact correctness results, including the independent orbit checks and all scorer-control outcomes.
4. Public synthetic instances and separate private truth/provenance, with deterministic regeneration instructions.
5. Raw responses, planned and attempted call ledgers, failed runs, timing/memory data, and material deviations for every experimental stage performed.
6. Primary and secondary CSV tables, essential figures, and an analysis command that reproduces them without inference.

For an analytic stop, replace nonexistent benchmark artifacts with the smallest verified counterexample, derivation, failing input, and reproduction command. For blocked model access or absent hardware, preserve the completed exact diagnostic and identify exactly which empirical claims remain untested. Never fabricate empty “successful” result files or report unperformed validation.

## 9. Honest outcome interpretation and autonomy

Execute the authorized stages, inspect their actual outputs, and follow the gates autonomously. Repair ordinary implementation problems within the shared reserve. Do not stop at producing code when execution is possible, and do not force later stages after a fatal gate fails.

| Outcome | Evidence required | Appropriate conclusion |
|---|---|---|
| Supported contribution | Valid exact diagnostic; completed fair comparison; prespecified finite-panel criteria met, with all uncertainty disclosed | A reproducible diagnostic and a promising tested configuration within this model/panel/resource scope; no transfer or population noninferiority claim |
| Informative negative result | Valid complete panel rejects the specified finite-panel effect, or a verified mathematical obstruction defeats the proposed mechanism | Identify the failed claim and conditions. Preserve any useful diagnostic contribution and unfavorable outputs. |
| Practical infeasibility | Measured memory/time/access constraints prevent the admitted protocol, despite bounded ordinary repairs | State the measured limit and which stage cannot fit; do not call a slow run a mathematical refutation. |
| Inconclusive evidence | Pilot floor, inadequate grading, incomplete held-out coverage, or uncertainty too broad for a broader claim | State precisely what can and cannot be inferred. A correct class prediction without construction does not resolve the target question. |
| Invalid experiment | Leakage, faulty truth, unequal unauthorized information, a material scoring bug influencing selection, or corrupted protocol | Withdraw affected comparisons. Repair only within the rules; retain the invalid evidence and explain why it cannot decide the question. |

A zero-success pilot supports a narrow observed capability limitation, not universal inability. A timeout, a bug, or failure to improve is not automatically a scientific refutation. Do not infer internal understanding, adversarial intent, or a real-world semantic meaning from numerical task performance.

If the prompt effect fails but the exact solver succeeds, the result still isolates a gap between mathematically available interpretations and this local model's usable output. If generic caution matches or beats witness search on the relevant metrics, describe it as the stronger configuration within the primary pair; report all three arms, and keep any recommendation involving the neutral arm separate from the prespecified `W-C` hypothesis. Do not replace caution with the weaker neutral prompt in the headline comparison. If the solver is the practical best solution under the declared model, say so plainly.

The default next step after a positive screen is **one independently designed application-data evaluation with an explicit hypothesis family and independent ground truth**, not another round of favorable synthetic tuning. After a floor or infeasibility result, recommend at most one concrete next step justified by the diagnosed bottleneck, such as a separately budgeted stronger local model or common algorithm assistance. Neither is part of this PoC.

## 10. Supporting research and planning evidence

### 10.1 Central claim-to-evidence map

| Central claim | Evidence and status | Remaining uncertainty / resolving check |
|---|---|---|
| Unique full labeling has boundary `n=3a-1` in the declared family | Supplied elementary proof [N1, cell 1]; audited with no identified defect; finite independent checks corroborate it | No external peer review or proof-assistant certificate; G1 independently checks the executable consequences and stops on a counterexample |
| Nonempty observations determine endpoints | Source corollary plus degree formula; independently checked on 1,695 intervals during planning | Input must actually belong to the family; full dictionary validation, not size/degree alone, validates that promise |
| Empty observations have infinitely many interpretations and no fixed symbols | Derivation (4), from the least distinct sum | No finite search can replace the unbounded argument; explicit different-start witnesses are the executable check |
| All ambiguous cases admit a cheap decoder and exact fixed-value oracle | New path/small-support derivation in Section 2.6; exact small orbit enumeration and a 735-case observation-only constructive probe | Historical novelty unverified; new implementation must pass G1; the finite probe is not an all-size proof |
| Full compatibility can include negative evidence without storing every nonedge | Subset-plus-cardinality derivation (7) | Duplicate records and domain mistakes can invalidate the reasoning; independent edge-set comparisons and mutation tests address these |
| Witness-search prompting may reduce unsupported certainty | Research hypothesis; nearby studies motivate the question but do not establish the effect here [R1–R6] | Requires the local model experiment; no model responses were evaluated during planning |
| The experiment can fit the M3 Max | Small exact algorithms; public compact MLX model; explicit memory calculation and measured-admission plan [R9–R14] | Actual machine, backend, immutable snapshot access, throughput, and response lengths remain G2 gates |
| A positive finite-panel result is useful but does not establish transfer | Defined practical screening decision and limited synthetic distribution | Application data and stronger generalization require a separate experiment |

### 10.2 What was actually done during planning

The research lead read both supplied files, including all notebook cells, recorded outputs, proof assumptions, and scope statements. The marker results were read but excluded as unrelated to the selected mechanism. Three independent agent tasks supported mathematical auditing, primary-source research, and local feasibility; three further draft-review passes challenged mathematics/novelty, experimental validity, and execution. These were agent reviews, not external peer review.

The following bounded calculations were performed in the planning environment:

| Completed work | Actual coverage and result |
|---|---|
| Independent degree/endpoint probe | 1,695 intervals: `a=1..30`, `n=a+1..4a+10`; exhaustive sorted-triple generation agreed with degree formula and nonempty endpoints. The lead reran the probe. |
| Independent exact automorphism/orbit probe | 19 selected intervals listed in G1: six edgeless cases handled analytically, 13 nonempty cases searched exactly with isolate permutations factored; threshold and sentinel orbits agreed. The lead reran the probe. |
| Source unique decoder reproduction | All 90 cases `a=6..15`, `n=3a-1..3a+7` decoded correctly with deterministic opaque names; executed by the lead. |
| Ambiguous-path check | 630 intervals: `a=6..40`, `k=4..a-2`; isolate sets and link paths agreed; reversal valid exactly at `k=4`. |
| Observation-only ambiguity construction probe | All 735 intervals `a=6..40`, `k=1..a-2`; fresh opaque labels with shuffled inventory, triple, and within-triple orders; inferred endpoints, valid distinct witnesses, and derived fixed/orbit sets agreed. Active-labeling counts were 6, 8, 6, 2, and 1 in the five table regimes. |
| Small feasibility calculations | Recomputed representative main-panel edge counts, case/call totals, KV-cache arithmetic, worst-case generated-token totals, and the illustrative zero-event uncertainty bound. |

The expanded ambiguity probe checks the finite construction independently of hidden endpoints at its interface. Its complete-orbit claim still uses the proved restriction to the small candidates/path orientations; the separate brute-force tiny check supplies an independent check of those orbit patterns. Do not overstate these checks as a formal proof of every implementation or every size.

For provenance, the temporary planning scripts had SHA-256 values:

- `math_audit_probe.py`: `94e7482b5b528b348b1293d13b6c362a13e30aff09b5ce98400c026c0661711e`.
- `ambiguity_decoder_probe.py`: `07b9245176352fa1e9b1d42f03fec1437edded2148851e9bba7070df60ec21df`.
- The ambiguity probe's deterministic per-case evidence rows had SHA-256 `2e25472c9d24b34f4921c1ab1655316342eaa2645243bcad91d091c9d4d9af02`.

Those temporary scripts are not required inputs to the execution brief. Recreate the specified independent checks in the new repository rather than assuming the planning workspace is available. Do not claim to have reproduced a byte-identical probe merely because its mathematical results match.

**Not performed:** language-model inference, MLX execution, model-weight download, inspection of the user's Mac, M3 Max timing/memory measurement, the full PoC, the source marker enumeration, or global historical-priority verification. The source notebook's recorded 135 automorphism checks remain source-reported; the independent rerun here was the separate 19-case exhaustive/orbit probe. The benchmark result and hardware feasibility are still hypotheses/gates.

### 10.3 Exact orbit sentinels for independent checking

| Interval | Full automorphism count | Value orbits |
|---|---:|---|
| `[6,13]` | 720 | `{6,7,13}`; `{8,9,10,11,12}` |
| `[6,14]` | 192 | `{6}`; `{7,8,13,14}`; `{9,10,11,12}` |
| `[6,15]` | 36 | `{6}`; `{7,8,15}`; `{9,13,14}`; `{10,11,12}` |
| `[6,16]` | 4 | `{6}`; `{7}`; `{13}`; `{8,16}`; `{9,15}`; `{10,14}`; `{11,12}` |
| `[6,17]` | 1 | Every value singleton, including isolated 12 |
| `[7,18]` | 12 | `{7}`; `{8}`; `{15}`; `{9,18}`; `{10,17}`; `{11,16}`; `{12,13,14}` |
| `[7,19]` | 2 | `{13,14}`; every other value singleton |
| `[8,21]` | 6 | `{14,15,16}`; every other value singleton |

The ambiguity witnesses need not enumerate these groups. The orbit table is needed for evaluating partial claims, not for requiring the subject model to output every possible dictionary.

### 10.4 Closest work and what this study adds

| Primary work | Relevant precedent | Project-specific distinction |
|---|---|---|
| WorldSense [R1], Sections 3.1–3.4 | Consistency/completeness of described worlds; alternative states; controlled linguistic skins | Here the output must construct a full numeric interpretation or two full compatible alternatives, not only choose an answerability label |
| LogicSkills [R2], Section 3.1, Section 4, Appendices B–C | Finite countermodel construction and semantic verification; separate construction and validity tasks | Here the same inverse problem requires deciding uniqueness and constructing its answer; the status is not supplied to the model |
| AbstentionBench [R3], Section 3.4, Section 4.4, Appendix D.5 | Insufficient-evidence failures, useful answering versus abstention, caution prompts | Here correctness and claims are checked exactly from structured output rather than through a language-model judge |
| Ma et al. [R4], Sections 3–4, Appendix A.5 | Structure-preserving symbol remapping | Renaming is a control, not a new contribution or proof about internal abstraction |
| REFUTE [R5], Section 5.2 and Table 2 | Executable counterexamples and explicit search prompting; benefit is not assured | Two compatible numerical worlds directly certify nonuniqueness; the prompt effect remains empirical |
| Logic-LM [R6], Section 3 | LLM formalization followed by deterministic symbolic reasoning and bounded feedback | The exact solver is kept as reference/scorer; no novelty is claimed for solver augmentation |
| Cooper–Poirel [R7]; Friedgut et al. [R8] | Schur triple systems and related hypergraph/Ramsey mathematics | These sources provide context, not an independently located proof of this notebook's interval-rigidity threshold |

Do not compare published accuracy numbers from these different tasks/models to this local study. The strong feasible operational comparison is the exact solver; the strong feasible prompt comparison is the fixed caution arm. A tuned large-model or solver-agent benchmark is outside the resource and information regime, so no claim of superiority over those methods is made.

**Search date:** 1 November 2026. Research used both available search engines, opened relevant primary-paper content and official documentation, and followed decision-relevant related-work leads to WorldSense, LogicSkills, REFUTE, and solver-based reasoning. Queries included combinations of “Schur hypergraph automorphism,” “restricted Schur triples interval rigidity,” “sum-preserving permutations,” “sum hypergraph reconstruction,” “Freiman isomorphism,” “abstention underspecified questions,” “symbol remapping abstract reasoning,” “countermodel construction LLM,” and “counterexample search prompting.”

This was a bounded opportunity search, not a systematic exhaustive survey or complete forward-citation census. No independent publication of the exact `n>=3a-1` classification was located. That absence does not establish novelty. The two Schur-context works were read at the relevant definitions/sections; their entire mathematical programs were not independently re-proved. An OpenReview access obstacle for LogicSkills was bypassed using the accessible ACL primary PDF. The MLX-LM release-tag sampler file could not be retrieved through the web reader; the package documentation and current sampler source were inspected, and the pinned installed signature must be verified in G2. The full model revision and actual checkpoint loading remain unresolved **setup gates**, with an exact retrieval procedure in Section 4.1; they are not claimed as verified.

## 11. Review findings, stress tests, and residual risks

Three distinct draft-review passes were completed and reconciled: mathematical/novelty, experimental, and execution. The following material findings shaped the final instructions.

| Finding or stress scenario | Disposition |
|---|---|
| The theorem might be nonconstructive or conceal an isomorphism oracle | Audited the degree-peeling/boundary construction and supplied a complete ambiguous branch; no generic GI/SMT subproblem is required |
| Rigidity on fixed endpoints might not imply unique unknown-interval meanings | Included the nonempty endpoint bridge and a separate infinite edgeless branch |
| Whole-dictionary ambiguity might be misused as an individual-value answer | Derived exact orbits, independently checked small cases, and included the `[6,16]` and `[6,17]` sentinels |
| Validation of only present triples could accept false interpretations | Complete domain validation plus edge-count equality; independent full-edge and negative-evidence mutation tests |
| The mechanism may already be known | Narrowed the contribution after finding WorldSense, LogicSkills, and counterexample-search precedents; no global priority claim |
| A favorable result could merely reflect blanket abstention or shifted errors | Per-class usefulness floors, utility-loss limits, two-witness requirements, and broad claim-error guard |
| Broad error semantics were ambiguous for wrong unique dictionaries or duplicate witnesses | Defined every returned assignment as a compatibility claim under every status; identical “different” witnesses explicitly fail `B` |
| Malformed responses could appear safer; status recovery could be permissive | Explicit lower/upper error bounds and top-level JSON-only status salvage; no regex or judge repair |
| Conservative bounds might fail even though the true effect could pass | Separated no-recommendation from informative negative; optimistic admissible bounds preserve inconclusive outcomes |
| A small pilot could miss the smallest meaningful unique case | Reserved `[6,17]` and nonempty `[7,15]` for a bounded development diagnosis; shifted the held-out main panel to `m=13..32` to preserve separation |
| Short calibration responses could spuriously imply nine-hour inference cost | Added one budgeted resource-only long-generation fallback; unsuccessful rate estimation yields inconclusive calibration, not measured infeasibility |
| A strong baseline could remove the gain | Fixed caution as the primary comparator; report all arms and accept a valid negative |
| Only favorable synthetic cases may work | Fixed near-threshold panel plus separately reported empty, sparse, nonisolated-ambiguity, and deeper-unique stress cases; restrict conclusions to the observed scope |
| Missing/corrupted evidence or a different numerical model | Excluded from the clean theorem's guarantee; syntax mutations test the implementation, not universal damage detection or model discovery |
| Finite precision could change the result | Exact integer oracle; fixed, logged quantized inference; no transfer of floating-point model behavior into mathematical truth |
| Preprocessing, prompt length, memory, or runtime could dominate | Count total cost, capacity-check complete reference outputs, measure the actual backend, admit only a bounded full/reduced panel |
| Essential source, dependency, or target machine may be unavailable | Self-contained algorithm and independent tests; explicit source-status records, public pinned download gate, and an honest blocked report |
| The main hypothesis may be false | Predeclared negative, infeasible, inconclusive, and invalid outcomes; no opportunistic model/prompt/tool switch |

The remaining risks are empirical: this compact model may have little constructive success; strict output requirements may expose a format bottleneck; a small fixed panel provides limited evidence about future instances; and mathematical historical priority remains unconfirmed. These are explicit limits and gates, not reasons to manufacture a positive result. No claim of absolute proof completeness, implementation perfection, or exhaustive novelty review is made.

## 12. References and attached-source identifiers

Bracketed identifiers in this document refer to the entries below. URLs are ordinary source links; no chat-specific citation tokens are needed. All online material was accessed on 1 November 2026 unless stated otherwise. Primary-paper locations identify the portions relevant to the design, not a claim to have verified every theorem in the cited work.

### Supplied material

**[N1] Author unspecified.** *Anonymous arithmetic, robust binary markers, and the limits of alien understanding.* Research notebook dated 1 November 2026; uploaded as `alien_communication_research(1).ipynb`. Planning copy: `/workspace/scratch/54cea597d30f/upload/alien_communication_research(1).ipynb`. Size: 63,604 bytes. SHA-256: `2928b7f3cda599d9364d17fc507c40cbe5f7e3ee87bd2a0c33b8da8da8635607`. Critical locations: cell 1, Theorem A, Lemmas 1–2, endpoint and algorithmic corollaries; cell 2, smallest example; cell 6, exact capped search; cell 11, reconstruction implementation; cells 9/12, recorded outputs; cells 13/14, limitations and novelty. If available to the coding agent, archive the original bytes under `inputs/` and record the hash. The planning path is provenance, not an assumed path on the Mac. There is no supplied public DOI or author record.

**[N2] User-provided task specification.** *Problem: unsupported certainty in AI interpretation of unfamiliar numerical data* and execution-brief requirements. Uploaded as `Pasted text(20261101-130307).txt`. Planning copy: `/workspace/scratch/54cea597d30f/upload/Pasted text(20261101-130307).txt`. Size: 33,484 bytes. SHA-256: `a2697b2daafbda817b25537c2253783a19aee7125d3d09ed3f8c157f958e760e`. Defines the model, practical purpose, local-only compute constraint, requested executor, and required research standards. The necessary requirements are incorporated in this brief.

### Research literature

**[R1] Youssef Benchekroun, Megi Dervishi, Mark Ibrahim, Jean-Baptiste Gaya, Xavier Martinet, Grégoire Mialon, Thomas Scialom, Emmanuel Dupoux, Dieuwke Hupkes, and Pascal Vincent.** *WorldSense: A Synthetic Benchmark for Grounded Reasoning in Large Language Models.* arXiv:2311.15930, first submitted 27 November 2023. [Primary paper](https://arxiv.org/pdf/2311.15930). Relevant: Sections 3.1–3.4, especially Section 3.3 “Completeness problems,” Table 1 and Figure 2. [Official code](https://github.com/facebookresearch/worldsense).

**[R2] Brian Rabern, Philipp Mondorf, and Barbara Plank.** *LogicSkills: A Structured Benchmark for Formal Reasoning in Large Language Models.* Findings of EMNLP 2026, pp. 2035–2047, 24–29 October 2026; arXiv:2602.06533. [Publication record](https://aclanthology.org/2026.findings-emnlp.98/), [primary PDF](https://aclanthology.org/2026.findings-emnlp.98.pdf). Relevant: Section 3.1, Section 4, Appendices B–C. [Official code](https://github.com/brianrabern/LogicSkills).

**[R3] Polina Kirichenko, Mark Ibrahim, Kamalika Chaudhuri, and Samuel J. Bell.** *AbstentionBench: Reasoning LLMs Fail on Unanswerable Questions.* arXiv:2506.09038, 10 June 2025; NeurIPS 2025 Datasets and Benchmarks. [Primary HTML, version 1](https://arxiv.org/html/2506.09038v1), [paper](https://arxiv.org/pdf/2506.09038). Relevant: Section 3.4 on evaluation, Section 4.4 and Appendix D.5 on caution prompts. [Official code](https://github.com/facebookresearch/AbstentionBench).

**[R4] Qingchuan Ma, Yuhang Wu, Xiawu Zheng, and Rongrong Ji.** *Benchmarking Abstract and Reasoning Abilities Through A Theoretical Perspective.* ICML 2025, Proceedings of Machine Learning Research 267:42209–42235. [Publication record](https://proceedings.mlr.press/v267/ma25u.html), [primary preprint HTML](https://arxiv.org/html/2505.23833v1). Relevant: Sections 3–4 and Appendix A.5, symbol-mapping procedure. This brief uses the experimental control, not a theorem inferring internal understanding from aggregate remapping scores.

**[R5] Shiven Sinha, Shashwat Goel, Ponnurangam Kumaraguru, Jonas Geiping, Matthias Bethge, and Ameya Prabhu.** *Can Language Models Falsify? Evaluating Algorithmic Reasoning with Counterexample Creation.* arXiv:2502.19414, 26 February 2025. [Primary HTML, version 1](https://arxiv.org/html/2502.19414v1), [paper](https://arxiv.org/pdf/2502.19414). Relevant: Section 4 for REFUTE construction; Section 5.2, Table 2, and Appendix A for explicit search prompting. [Official code](https://github.com/falsifiers/REFUTE).

**[R6] Liangming Pan, Alon Albalak, Xinyi Wang, and William Yang Wang.** *Logic-LM: Empowering Large Language Models with Symbolic Solvers for Faithful Logical Reasoning.* Findings of EMNLP 2023, pp. 3806–3824, December 2023. DOI: [10.18653/v1/2023.findings-emnlp.248](https://doi.org/10.18653/v1/2023.findings-emnlp.248). [Primary PDF](https://aclanthology.org/2023.findings-emnlp.248.pdf). Relevant: Section 3 and Figures 1–2, formulation/solver/interpretation and bounded refinement. [Official code](https://github.com/teacherpeterpan/Logic-LLM).

**[R7] Joshua Cooper and Chris Poirel.** *Pythagorean Partition-Regularity and Ordered Triple Systems with the Sum Property.* arXiv:0809.3478, first submitted 20 September 2008; the retrieved PDF carries an internal later date. [Primary PDF](https://arxiv.org/pdf/0809.3478). Relevant: Section 1 “Preliminaries,” Example 1, PDF page 3, defining the Schur triple system. This is mathematical context, not a source for equation (1).

**[R8] Ehud Friedgut, Eden Kuperwasser, Wojciech Samotij, and Mathias Schacht.** *Sharp thresholds for Ramsey properties.* Forum of Mathematics, Sigma 14 (2026), e32; online 27 February 2026. DOI: [10.1017/fms.2026.10178](https://doi.org/10.1017/fms.2026.10178). [Publisher full text](https://www.cambridge.org/core/journals/forum-of-mathematics-sigma/article/sharp-thresholds-for-ramsey-properties/37118888483709FB8C79D6FDB0F7F254). Relevant: Section 7.3, Schur hypergraphs in cyclic groups. Its random Ramsey threshold is different from this notebook's interval reconstruction threshold.

### Runtime and model documentation

**[R9] Qwen Team.** *Qwen3-4B-Instruct-2507 model card.* 2025, accessed 2026. [Official model card](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507). Relevant: “Model Overview,” non-thinking-only behavior, architecture and Apache-2.0 license. No model-card benchmark number is used as evidence of performance on this task.

**[R10] MLX Community.** *Qwen3-4B-Instruct-2507-4bit model card and configuration.* 2025–2026. [Conversion repository](https://huggingface.co/mlx-community/Qwen3-4B-Instruct-2507-4bit), [configuration](https://huggingface.co/mlx-community/Qwen3-4B-Instruct-2507-4bit/blob/main/config.json), [revision history](https://huggingface.co/mlx-community/Qwen3-4B-Instruct-2507-4bit/commits/main). Relevant: conversion provenance, artifact size, `num_hidden_layers`, `num_key_value_heads`, `head_dim`, and `quantization`. Planning verified the public card/configuration, not a successful download or full immutable revision.

**[R11] MLX-LM contributors.** *MLX-LM 0.32.2 package, release, and Python API.* Released 16 October 2026. [Package metadata](https://pypi.org/project/mlx-lm/0.32.2/), [release](https://github.com/ml-explore/mlx-lm/releases/tag/v0.32.2), [source commit](https://github.com/ml-explore/mlx-lm/commit/93e6d4a6447b3080861985947f0820e76fcb957d), [sampler source at the release tag](https://github.com/ml-explore/mlx-lm/blob/v0.32.2/mlx_lm/sample_utils.py). Relevant: Python requirement, `load`, streaming generation, sampler argument, and greedy sampling. Inspect the installed signatures during implementation rather than assuming unrelated generation APIs accept the same arguments.

**[R12] Apple / MLX contributors.** *MLX Build and Install.* Documentation version 0.32.4 as accessed. [Official installation documentation](https://ml-explore.github.io/mlx/build/html/install.html). Relevant: “Python Installation,” native architecture, macOS requirements. Actual dependency resolution and target compatibility remain measured setup checks.

**[R13] Apple / MLX contributors.** *Memory and synchronization API documentation.* Documentation version 0.32.4 as accessed. [Memory-limit semantics](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.set_memory_limit.html), [peak-memory API](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.get_peak_memory.html), [synchronization API](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.synchronize.html). Related [MLX issue 3896](https://github.com/ml-explore/mlx/issues/3896), opened 23 July 2026 with a project-member explanation on 8 August 2026, clarifies active versus cached footprint. Issue workload numbers are not measurements of this proposed study.

**[R14] Hugging Face.** *Download files from the Hub.* Current official documentation accessed 1 November 2026. [Download guide](https://huggingface.co/docs/huggingface_hub/guides/download). Relevant: “From specific version” and “Download an entire repository”; use a full commit hash with `snapshot_download`, archive it, and load locally thereafter.

## 13. Completion requirements for the coding agent

Finish the authorized work up to the justified terminal gate. Verify the report against the raw records and actual commands. State which claims are sourced, derived, measured on the target, estimated, untested, or blocked. Provide the complete result even when it is negative, infeasible, inconclusive, or invalid. The research decision must follow the evidence and the prespecified rules; a positive prompt result is optional, a trustworthy result is required.


