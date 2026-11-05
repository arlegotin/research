# PoC execution brief: certified sparse guards on real LLM features

**Recipient:** an autonomous coding agent configured as `gpt-6.1-sol`, reasoning effort `xhigh`.

**Research and protocol date:** 5 November 2026. **Target:** one local MacBook with an M3 Max; inspect its actual memory, operating system, storage, and software before allocating work. **Deliverable from you:** an executed, reproducible research result, including a justified early stop when warranted. This document is the research plan, not evidence that the LLM experiment has already succeeded.

## 1. Project decision and research contract

### 1.1 Execute this one project

Test whether the notebook's 13-coordinate projective-plane polynomial can provide **useful, selective correction of a frozen Gemma 3 1B instruction model on a compositional ticket-escalation task**, while its conditional preservation bound remains informative on unrelated text. Use pretrained Gemma Scope 2 features, an explicitly constructed two-layer SwiGLU auxiliary branch, and a bounded positive correction to two answer logits. Discover features using development data; do not train the base model or the SAE.

The primary falsifiable question is:

> With a fixed, bounded search over two matching pretrained SAEs, can a 13-feature compiled guard improve positive-case accuracy by at least five percentage points, preserve at least 95% accuracy on each type of three-of-four near miss, and change at most 1% of unrelated binary decisions, while supplying a useful sparse-tube certificate on at least 95% of those unrelated inputs?

All percentages refer to the precisely defined distributions, independent units, and confidence rules below. They are requirements to test, not expected results. The main comparison is the proposed patch against its own frozen, explicitly prompted base. Comparisons with the symmetric polynomial and learned gates explain the cost and limitations of obtaining the certificate; superiority to every baseline is not required for a useful finding.

The decision this study informs is whether this **particular certified guard family warrants further investment as a selective behavior-patching mechanism**, or whether feature geometry, correction strength, simpler alternatives, or local resource costs make it unpromising in this setting.

### 1.2 The contribution and closest alternatives

The smallest defensible contribution is a reproducible characterization of the **certificate–utility tradeoff on actual frozen-model features**: whether the necessary geometry occurs, how much correction the geometry permits, what preservation the certificate actually explains, and what it costs relative to credible simple alternatives. A scoped negative result is useful. Neither generic local patching nor supervised classification of SAE activations is new: REASSURE, SERAC, GRACE, Dynamic SAE Guardrails, and ordinary SAE classifiers already cover substantial parts of those ideas [R6–R10]. Program-to-weight compilation and the complementary-SiLU multiplication identity also predate this application [R5].

The supplied notebook's specific minimum-degree sparse-tube minimax theorem and converse to design balance are the mathematical foundation [N, §§2–5]. This review found no essential defect under its stated assumptions. Historical priority is not established; targeted searches are not an exhaustive novelty proof. Do not spend this PoC trying to establish priority or invent another theorem. A real-feature validation or limitation is enough.

### 1.3 Explicit revisions to the initial idea

1. **Supply the same four-condition policy in the primary base prompt and in every patched primary condition.** The study repairs execution of an already supplied rule. It does not demonstrate acquisition of a missing capability or persistence of a learned rule after removing the instruction. Withholding the rule from a capable prompt baseline would confound the comparison.
2. **Use a separate unnormalized branch and a direct output correction.** Gemma's native gated feed-forward activation is not the notebook's assumed SwiGLU architecture [R2]. This is a frozen-model auxiliary patch, not an in-place rewrite of existing Gemma feed-forward weights.
3. **Use one bounded positive write direction.** A nonnegative guard can raise escalation preference; it cannot fix existing false-positive escalations. Inspect that obstruction early. Do not silently add a second guard, reverse the edit direction, remove difficult negatives, or weaken the prompt to manufacture room for improvement.
4. **Study one-token decisions.** Preservation is tested at the declared answer boundary, including on an unrelated real question-answering dataset. This is not a claim about arbitrary multi-turn conversation, generated continuations, identical samples, or all model capabilities.
5. **Use controlled synthetic ticket text and real unrelated text.** No accessible, independently validated corpus with these exact four ticket predicates was established in this review. The ticket generator isolates composition and near misses; it does not represent production support traffic. Keep that limitation in the title or abstract of the resulting empirical report.
6. **Do not include LoRA in the core matrix.** The selected question concerns the applicability and cost of a specified preservation mechanism. Strong inexpensive learned-gate controls isolate it within the budget. This exclusion gives up claims that the method is more accurate, cheaper overall, or generally preferable to fine-tuning. LoRA is not categorically unavailable on Apple silicon; the exclusion is a scope decision, not a hardware impossibility claim [R13, R14].

The original ≥95% positive accuracy remains a separately reported aspiration. The principal useful-effect threshold is a five-point positive-case improvement with the preservation conditions above. A model already above 95% positive accuracy may have too little headroom to establish that improvement; this is a valid opportunity limitation, not a reason to degrade the base.

### 1.4 Fixed requirements and engineering discretion

The model family, matching IT SAEs, two candidate layers, feature position, normalization, polynomial, edit direction, amplitude cap, data split, primary metrics, inference scope, and cumulative budgets below are fixed scientific choices. You may choose ordinary implementation details—batch size within the specified options, file compression, vectorization, and a supported precision/backend path—after correctness and resource checks. Log every material deviation. Do not change the hypothesis, distribution, baseline family, or success threshold after viewing confirmation results.

Execute the authorized stages autonomously. Repair ordinary bugs within the shared repair allowance. A gate failure ends the affected experiment; it does not prevent you from preserving evidence and writing the report. If the Mac, model access, or another indispensable resource is unavailable, complete unaffected analytic and software work and identify precisely what remains unvalidated.

## 2. Evidence and mathematics you must preserve

### 2.1 Supplied material and source locations

The mathematical source is **`Optimal_Guards_for_Weight_Space_Programming(1).ipynb`**, dated 5 November 2026, with 22 cells. No separate mathematical PDF was supplied. The task specification was **`Pasted text(20261105-135718).txt`**. In the planning workspace they were at:

```text
/workspace/scratch/2155d5887c42/upload/Optimal_Guards_for_Weight_Space_Programming(1).ipynb
/workspace/scratch/2155d5887c42/upload/Pasted text(20261105-135718).txt
```

Those are provenance locations, not paths expected to exist on the Mac. If supplied, copy them unchanged to `inputs/` and record SHA-256 hashes. The definitions, construction, and essential checks below are sufficient to implement the selected mechanism without this chat. If the notebook is unavailable to you, say that you could not independently compare its complete source; do not invent an attachment or claim a rerun of it.

Notebook citations use zero-based cell numbers: Theorem A is cell 2 with proof in cell 3; Theorem B is cell 2 with proof in cell 4; the concrete compiler is cell 5 and code cells 13–14; the signed-design counterexample is cells 15–16; output insertion is cell 19; limitations are cell 20. There is no Theorem C.

### 2.2 Exact mathematical statement

For integers $1\le s<n$, $1\le r\le n-s$, let $d=s+r$. Define

\[
\Sigma_s=\{z\in\mathbb R^n:|\operatorname{supp}(z)|\le s\},\qquad
T_{s,\varepsilon}=\{z\in[-1,1]^n:\#\{i:|z_i|>\varepsilon\}\le s\},\quad0<\varepsilon<1.
\]

An admissible real polynomial has degree at most $d$, satisfies $p(\mathbf1)=1$, and has $D^\beta p=0$ on every point of $\Sigma_s$ for every multi-index with $|\beta|<r$. These are derivatives with respect to feature coordinates. Under these assumptions it must have the form

\[
p(z)=\sum_{|A|=d}c_A\prod_{i\in A}z_i,\qquad \sum_Ac_A=1.
\]

There is no nonzero admissible polynomial of smaller degree. To see the key restriction, a monomial with exponents $\gamma_i$ must have outside degree $\sum_{i\notin S}\gamma_i\ge r$ for every size-$s$ support $S$. Taking its $s$ largest exponents and using total degree at most $s+r$ forces every positive exponent to be one. This uses $s\ge1$ [N, §3.1].

The exact optimum, even allowing signed coefficients, is

\[
\min_p\sup_{z\in T_{s,\varepsilon}}|p(z)|=
L_{n,s,r}(\varepsilon)=
\frac{\sum_{j=r}^{\min(d,n-s)}\binom{s}{d-j}\binom{n-s}{j}\varepsilon^j}{\binom nd}.
\]

The normalized elementary symmetric polynomial $E_d=\binom{n}{d}^{-1}\sum_{|A|=d}z_A$ attains it. For any output norm and target vector $u$, the corresponding vector-correction optimum is $\|u\|L$, attained by $uE_d$. The proof averages evaluations at all positive corners with $s$ coordinates equal to one and the others equal to $\varepsilon$; positivity of the symmetric polynomial bounds the entire signed tube [N, Theorem A, §§3.2–3.3].

Every scalar minimizer at even one fixed noise level obeys

\[
\sum_{A\supseteq S}c_A=\frac{\binom ds}{\binom ns}\quad (|S|=s).
\]

Nonnegative coefficients satisfying these design equations are sufficient. Signed balanced coefficients are not sufficient in general. For the converse, with inclusion matrices $W_{i,j}$ and $t=1/\varepsilon$, boundary evaluation factors as

\[
M=\varepsilon^dHW_{s,d},\qquad
H=\sum_{i=0}^{s}\frac{(t-1)^i}{\binom{d-i}{s-i}}W_{i,s}^{\mathsf T}W_{i,s}.
\]

The $i=s$ summand is a positive multiple of the identity; thus $H$ is invertible. Equal corner values at the optimum force equal design marginals [N, §4]. Every minimizer needs at least $\lceil\binom ns/\binom ds\rceil$ supported monomials. Equality with the unrounded ratio occurs precisely for equally weighted Steiner blocks. This is a monomial-support result, not a lower bound on general neural-network size or arithmetic work.

### 2.3 The exact guard to implement

Fix $n=13,s=2,r=2,d=4$. Use the projective plane $PG(2,3)$. Enumerate nonzero triples over $\{0,1,2\}$, retaining those whose first nonzero coordinate is 1; sort lexicographically. A line is the set of point indices whose dot product with one such nonzero normalized triple is 0 modulo 3. The resulting zero-based blocks are:

```python
BLOCKS = [
    (1, 4, 7, 10), (0, 4, 5, 6), (3, 4, 9, 11),
    (2, 4, 8, 12), (0, 1, 2, 3), (1, 6, 9, 12),
    (1, 5, 8, 11), (0, 10, 11, 12), (3, 6, 8, 10),
    (2, 5, 9, 10), (0, 7, 8, 9), (2, 6, 7, 11),
    (3, 5, 7, 12),
]
```

Independently regenerate and compare these blocks; assert that every pair of points occurs in exactly one block. For an ordered selected feature vector $z\in[0,1]^{13}\subset[-1,1]^{13}$, define

\[
G(z)=\frac1{13}\sum_{A\in\mathrm{BLOCKS}}\prod_{i\in A}z_i,
\qquad L(\varepsilon)=\frac{\varepsilon^2+6\varepsilon^3+6\varepsilon^4}{13}.
\]

Then $G(\mathbf1)=1$; $G$ and its feature-coordinate gradient vanish on every two-sparse state; and $\sup_{T_{2,\varepsilon}}|G|=L(\varepsilon)$. Thirteen monomials are necessary and sufficient in this minimax degree class. On the nonnegative cube, $0\le G\le1$.

Do not reinterpret this as a four-condition logical AND or as a threshold on total activation count. Exact enumeration performed during planning showed that only 13 of the 715 binary supports of size four activate this guard. The complement of any projective line has **nine active coordinates but $G=0$**, whereas $E_4=126/715\approx0.176224$. Consequently the assignment of SAE coordinates to the 13 points is part of feature search and must be frozen and charged as tuning.

A second obstruction is useful as a negative control. Suppose each of 13 coordinates is merely a copy of one of four Boolean semantic predicates. Exact zero on every missing-one-predicate input would require every projective line to contain all four predicate types. Every pair of points lies on a line, so no two points could have the same type; 13 points cannot meet that requirement with four types. This rules out that particular copied-predicate implementation, not real contextual SAE features or successful classification using favorable base margins. Also, an AND of four actual Boolean predicates is already linearly separable by a sum threshold of 3.5; this task cannot establish that a polynomial is necessary for conjunction.

### 2.4 Compile real matrices, alongside a direct reference

Use the established identity $\operatorname{SiLU}(a)-\operatorname{SiLU}(-a)=a$, so

\[
ab=\operatorname{SiLU}(a)b-\operatorname{SiLU}(-a)b.
\]

For a batch-by-input matrix $X$, one layer is

\[
\operatorname{SwiGLU}(X)=\big(\operatorname{SiLU}(XW_g^\mathsf T)\odot XW_u^\mathsf T\big)W_d^\mathsf T.
\]

To produce $x_a x_b$ in output slot $j$, allocate hidden slots $2j,2j+1$. Set gate rows to $+e_a,-e_a$, both up rows to $e_b$, and down coefficients for output $j$ to $+1,-1$. All other entries are zero. No biases, normalization, dropout, residual addition, activation substitution, or trainable parameters are allowed inside the guard.

The first layer computes two pair products for each ordered block; the second computes their product and averages the 13 results. Shapes are:

| Layer | Gate and up matrices | Down matrix | Hidden units |
|---|---|---|---:|
| 1 | each $52\times13$ | $26\times52$ | 52 |
| 2 | each $26\times26$ | $1\times26$ | 26 |

The second down matrix has alternating $+1/13,-1/13$. The branch has 78 hidden units, 39 product operations in its construction, 234 nonzero scalar weight entries, and 4,082 entries when the six matrices are stored densely. This does not count the encoder, feature selection, or output write.

Two layers are minimal for a globally exact quartic in this unnormalized affine-SwiGLU architecture, by its growth bound $O((1+\|z\|)^{2^L})$. This is not a lower bound for arbitrary guards or normalized architectures [N, §5].

Maintain three independent evaluators: direct FP64 quartic products, the actual matrix branch, and normalized $E_4$ by a dynamic program. The symmetric reference must not enumerate 715 monomials in timed inference. A normalized sequential update for $k$ processed coordinates is

\[
E_j^{(k)}=\frac{k-j}{k}E_j^{(k-1)}+\frac{j}{k}z_kE_{j-1}^{(k-1)},\quad E_0=1,
\]

using old values, $j\le\min(k,4)$, and absent states as zero. A balanced normalized convolution tree is another valid implementation. Planning execution of the notebook's tree at $n=13,d=4$ used **33 bilinear products at depth 4**, versus the project's 39 at depth 2. Measure actual work and latency; do not claim a 55-fold speedup from 715 versus 13 monomials.

### 2.5 Feature map and output correction

Let $h(x)\in\mathbb R^{1152}$ be the output of the specified transformer block at the last nonpadding prompt position, immediately before the answer is generated. For the matching SAE, with $W_{\rm enc}\in\mathbb R^{1152\times16384}$, use

\[
b(x)=h(x)W_{\rm enc}+b_{\rm enc},\qquad
a_j(x)=\max(b_j(x),0)\,\mathbf1\{b_j(x)>\theta_j\}.
\]

This is the released JumpReLU encoder convention. Do not subtract decoder bias or add an input normalization; the release folds its training scale into the weights [R3, R4]. Inspect actual tensors and config before accepting those shapes.

For selected ordered IDs $j_1,\ldots,j_{13}$, compute scales on **fit-positive examples only**,

\[
q_j=Q_{0.90}\{a_j(x):x\in D_{\rm fit}^{+}\},\qquad
z_i(x)=\min(1,\max(0,a_{j_i}(x)/q_{j_i})).
\]

Use NumPy's `quantile(method="linear")`; drop any coordinate with nonpositive or nonfinite $q_j$. This quantile includes zeros. Keep the exact scales. There is no additional learned offset, threshold subtraction, top-k operation, or supervised feature synthesis. Clipping ensures the cube domain; it does not establish useful separation. Report saturation and raw as well as normalized activation statistics.

Let $v_Y,v_N$ be verified single-token answer IDs and let

\[
m_0(x)=\ell_{v_Y}(x)-\ell_{v_N}(x),\quad
\ell'_Y=\ell_Y+\alpha G(z)/2,\quad
\ell'_N=\ell_N-\alpha G(z)/2,
\quad 0\le\alpha\le8.
\]

Every other logit is unchanged. The binary prediction is Yes exactly when (m>0); ties are No. The two logits may be modified after the frozen final projection, in FP32 or higher. Do not apply another softcap or change temperature afterward. The scalar α is calibrated as specified below; the 78-unit weights are compiled, not fitted.

The cap 8 is a **predeclared engineering bound**, not a theorem or measured optimum: it allows a large change of binary odds while preventing arbitrarily amplified tiny guards. Always report the actual α. A smaller signal cannot be rescued by an unbounded readout. At $z_i=0.5$ for every coordinate, $G=1/16$, so even α=8 shifts the margin by only 0.5.

For ε equal to the third-largest coordinate, every observed $z$ belongs to its corresponding tube; use limiting values (L(0)=0,L(1)=1). This fact alone is vacuous. The fixed useful tube is **ε₀=0.1**. In it,

\[
L(0.1)=0.001276923076923077,
\quad |m'-m_0|\le8L(0.1)=0.0102153846153846,
\quad\|\ell'-\ell\|_\infty\le0.0051076923076923.
\]

Thus a binary base margin with magnitude greater than $\alpha L(\varepsilon)$ remains unchanged in real arithmetic. A full-vocabulary top-two base margin greater than the same quantity is sufficient for full argmax preservation. Exact zero correction gives exact value preservation of the mathematical composition without requiring the feature encoder to be smooth. The gradient theorem does **not** give global derivative preservation through JumpReLU thresholds or clipping kinks. Do not claim sampled-token or sequence preservation from an argmax bound.

### 2.6 What was actually checked during planning

This work occurred on the planning environment's CPU, not on an M3 Max. No Gemma checkpoint was loaded, no ticket activations were measured, and no LLM edit was evaluated.

| Actual check | Result and scope |
|---|---|
| Complete reading of both attachments, notebook code and saved outputs, and its plotted interference figure | The figure depicts the stated mathematical bound, not empirical LLM behavior. |
| Independent agent rerun of every original notebook code cell | 168 degree cases and 70 signed-coefficient LPs passed; maximum LP/formula relative discrepancy was $6.66\times10^{-16}$. |
| Actual matrix branch on 11,001 notebook cases | Max absolute error versus quartic reference: FP64 $4.44\times10^{-16}$, FP32 $1.696\times10^{-7}$; tested sparse values and gradients were zero. FP32 target output was 1.0000001192, not exactly 1. |
| Independent exact rational inclusion-factorization checks | All entries passed in five parameter cases, including 55,770 entries for $13,2,2,\varepsilon=1/10$. |
| Exhaustive enumeration of all 8,192 Boolean feature vectors | Established the line-count interpretation and the nine-active zero-response example above. |
| Exact count/byte arithmetic and binomial-interval calculations | Informed the memory plan and sample-size requirements below; they are not measured model throughput or evidence of natural-language generalization. |

The rerun environment reported NumPy 2.3.5, SciPy 1.17.0, and Matplotlib 3.10.8. The planning audit files are not required inputs to your implementation; recreate the essential checks below. Theoretical proofs establish real-arithmetic claims. Numerical successes do not certify all machine-precision inputs.

## 3. Concrete resources and initial inspection

### 3.1 Retrieve only the necessary public files

| Resource | Exact choice and access rule |
|---|---|
| Frozen model/tokenizer | `google/gemma-3-1b-it`; download the standard nonquantized checkpoint and tokenizer after confirming authorized access to its Gemma terms. Record resolved commit and hashes. |
| Primary SAE | `google/gemma-scope-2-1b-it`, `resid_post/layer_13_width_16k_l0_medium/{config.json,params.safetensors}`. |
| Single predeclared alternative | Same repository, `resid_post/layer_17_width_16k_l0_medium/{config.json,params.safetensors}`. No extra layer, width, PT model, or SAE training search. |
| Retention data | BoolQ: official training split for fit/calibration, official validation/development split for confirmation [R15]. |

The SAE repository revision observed during research was `b0fa29457c3601df0a70c48a15534c738d7c10e0`; use it if retrievable. The layer-13 parameter file was 151,131,000 bytes, with SHA-256 `6be6fbfed9850588adb1826bcc9fc89551d9eaf573311e95b21307fd4a281cd5`. If this immutable revision cannot be retrieved, resolve one available revision once, record the reason and actual hashes, verify the same config/shape contract, and freeze it before collecting model outcomes. A changed architecture or encoder convention is a blocker, not an equivalent silent substitution.

The layer-13 config specifies `model_name=google/gemma-3-1b-it`, `hf_hook_point_in=model.layers.13.output`, width 16,384, JumpReLU, target L0 60, `affine_connection=false`. These are zero-based block outputs after the residual updates. Verify the equivalent layer-17 fields. Target L0 60 is a training/configuration value, not a guarantee that 13 chosen coordinates are two-sparse on protected inputs.

**Do not download the entire SAE repository or its example datasets.** The observed whole repository was about 2.89 TB; the layer directory included a large examples file. Fetch only the two named files per permitted layer. Scope 2 uses `safetensors`, not the older Scope `npz` examples. Some overview-card examples are stale; prefer checkpoint config, tensor shapes, and actual loader source [R3, R4]. SAEs are licensed CC-BY-4.0; retain attribution. The model has separate Gemma usage conditions. Never accept new legal terms or invent access credentials on the user's behalf.

### 3.2 Inspect the actual Mac

Create an environment report with timestamp, hardware model, chip, physical RAM, available memory estimate, free disk, macOS version, architecture, power mode, Python, package versions, BLAS configuration, and MPS availability. Useful native commands are `sw_vers`, `uname -m`, `sysctl -n hw.memsize`, `system_profiler SPHardwareDataType SPDisplaysDataType`, and `df -h .`; avoid storing hardware serial numbers. Query `torch.backends.mps.is_built()` and `is_available()`. Record the results instead of inferring capabilities from the chip name.

Use Python 3.11 or 3.12 in an isolated local environment, with PyTorch, Transformers, huggingface_hub, safetensors, NumPy, SciPy, scikit-learn, Matplotlib, psutil, and pytest. Use official supported wheels. Resolve mutually compatible versions, run the smoke checks, then freeze them. The commands in §10 are interfaces you must create; they were not executed on the Mac during planning.

Primary engineering path: nonquantized PyTorch model on MPS, FP32 initially; CPU FP64 for mathematical oracles; FP32 encoder and guard with an explicitly recorded device. An auxiliary CPU path is acceptable and simple, but its transfers/synchronization count in latency. Select the auxiliary device once during calibration. Do not silently mix feature-extraction paths between fitting and deployment.

Disable implicit MPS-to-CPU fallback initially and expose unsupported operations. If a bounded repair enables a documented fallback, record every affected operation and measure end-to-end time. If FP32 model inference is too costly, one predeclared BF16 model attempt is permitted only when supported and verified on the installed stack; keep encoder/guard arithmetic FP32 and re-collect affected development caches. Do not quantize, change the model, or loosen numerical criteria to fit the budget. A CPU-only model run is a diagnostic fallback; proceed to the full study only if measured scaling fits the same caps. Current official docs differ in stated OS minima; actual backend checks and a real forward pass decide feasibility [R13, R14].

### 3.3 Avoid hidden large intermediates

Use `eval()`, frozen parameters, `torch.inference_mode()`, `use_cache=False`, `output_hidden_states=False`, and the official final-position `logits_to_keep=1` argument, after checking its installed signature [R2]. Capture the selected position from a forward hook; do not retain all hidden states or all-token SAE activations. Handle block outputs that are either a tensor or a tuple with the residual tensor first, and assert shape/position. Use fixed padded length buckets 256 and 512; left-pad so the final position is always real. Check single-item versus padded-batch equivalence before using batches.

Known architecture dimensions from official source are 26 layers, residual width 1,152, intermediate width 6,912, and vocabulary 262,144; assert against the downloaded config [R2]. These byte calculations explain the implementation choices:

| FP32 object | Calculated size |
|---|---:|
| Full $1152\times16384$ encoder | 72 MiB |
| Encoder, decoder, biases and thresholds | about 144.13 MiB |
| 5,000 final-position SAE rows | 312.5 MiB |
| 5,000 final-position residual rows | 21.97 MiB |
| 5,000 examples × 256 tokens × full SAE activations | **78.125 GiB** |
| Batch 8 × 256 tokens × vocabulary logits | **2 GiB** |
| Batch 8 × one position × vocabulary logits | 8 MiB |
| 13 selected encoder columns | 58.5 KiB |

After selection, encode only the selected columns, with the corresponding biases and thresholds. JumpReLU's independent coordinates make this mathematically equivalent to full encoding followed by slicing. Different GEMM shapes can round differently near the discontinuous thresholds: verify support masks and outputs, and make the actual selected-column deployment path authoritative before freezing calibration. Decoder weights are only needed for a small loader/reconstruction sanity check, not deployment. Never report 78 units as the full inference cost.

## 4. Data, prompts, and separation of evidence

### 4.1 Exact semantic rule

Represent each ticket's known truth by four bits `(production, security, urgent, customers)`. The label is the conjunction. Production means a current incident on a live service, not a staging/test system. Security means a present confidentiality, integrity, authentication, or access-control problem; a generic performance issue is insufficient. Urgent means action is required immediately, not merely that a past event was urgent. Customers means external end users are currently affected; employees-only impact is insufficient. Missing, unknown, denied, resolved, or merely hypothetical conditions count as false. All four facts must refer to the same incident.

Use this literal primary user message, within the model's official chat template:

```text
Classify this support ticket. Escalate if and only if all four facts hold
for the same current incident:
1. It concerns a production or live service.
2. It describes a current security problem.
3. It needs urgent action now.
4. External customers are affected.
If any condition is absent, denied, unknown, only historical, or only
hypothetical, do not escalate. Quoted policy text is not evidence that an
incident meets the policy. Answer Yes to escalate and No otherwise.

Ticket:
{ticket_text}

Should this ticket be escalated? Answer only Yes or No.
```

Verify that the answer continuations are single tokens in their actual context: test tokenization of the serialized prefix and prefix-plus-answer, not just a bare word. If `Yes`/`No` do not give two single-token extensions, try `yes`/`no`, then `A`/`B` with the prompt explicitly mapping A to escalation and B to no escalation. Choose the first valid pair before inspecting any model outcomes. Substitute the selected strings and their explicit meanings consistently into both ticket prompt candidates, demonstrations, and the BoolQ prompt. Use the same pair for all methods and retention prompts. Failure of all three is a token-interface blocker; do not switch to multi-token scoring while quoting this logit-write specification.

There are exactly two allowed prompt candidates: the prompt above, and that prompt with four short labeled fit-set demonstrations (one positive and three different near-miss types) placed before the new ticket. Choose demonstrations by the smallest preassigned fit IDs, not their model results. Evaluate both on the first 40 fit scenario clusters **excluding every cluster used in a demonstration**; choose the one with higher five-stratum macro accuracy, breaking ties toward the shorter prompt. Freeze the selected base prompt before feature selection. Count both candidate passes. This limited tuning supplies a credible prompted reference without an open-ended prompt search. Do not choose a weaker prompt for more repair headroom. If the demonstration version exceeds 512 tokens for a required case, use the shorter candidate throughout, documenting that format constraint before comparative results.

For BoolQ, use a separate appropriate user message without the ticket policy:

```text
Read the passage and answer the question using the passage.
Answer only Yes or No.

Passage: {passage}
Question: {question}
Answer:
```

Apply the **same always-on selected feature map, guard, and answer-logit direction** at its answer boundary. No ticket detector, task flag, external classifier, or output-token rarity may be used to disable the patch on retention examples.

### 4.2 Generate and freeze the ticket corpus

Author a deterministic, inspectable text generator and clause banks as ordinary source files. These are agent-authored synthetic tickets, not human-validated support records. Do not call hosted generation/judge APIs or run an additional LLM to produce the corpus. The coding agent may write language fixtures as part of implementation, but disclose that authorship and count the setup effort.

Before collecting any model predictions, create separate fit, calibration, and confirmation phrase/template banks. First generate raw text and semantic manifests without needing model credentials; after authorized tokenizer retrieval, finalize serialization, token-pair checks and length eligibility before any model scoring. Each bank must contain at least 12 distinct positive and 12 distinct negative realizations per predicate, with fixed metadata indicating truth and rhetorical type. Include affirmative statements, explicit denial, resolved history, hypothetical situations, and internal/external or production/staging distinctions. Include several multi-sentence and reordered forms, and quoted policy/distractor keywords in both classes. Keep phrase IDs and template roots disjoint across splits. Entity and product names cannot predict labels. Never emit the four-bit tuple, labels, generation metadata, or an explicit count of satisfied conditions into the input.

For each independently sampled scenario cluster, draw its context, entities, clause realizations, sentence order and distractor choices once. Produce one all-four-positive ticket and four minimally changed versions, each denying exactly one condition. Preserve irrelevant details across the five. Negation must change factual scope correctly, not merely add the word “not” to an ambiguous sentence. Reject malformed examples through generator invariants and a text audit **before** model outputs are visible; record all replacements and reasons.

Use fixed seeds: generator 20261105, split/ordering 104729, search 271828, bootstrap 314159. Seeds identify deterministic samplers, not repeated independent LLM experiments.

| Split | Scenario clusters | Core presentations | Extra all-mask presentations | Ticket total |
|---|---:|---:|---:|---:|
| Fit | 250 | 1,250 | 220: the other 11 masks for first 20 clusters | 1,470 |
| Calibration | 150 | 750 | 220: other masks for first 20 clusters | 970 |
| Confirmation | 600 | 3,000 | 440: other masks for first 40 clusters | 3,440 |
| Total | 1,000 | 5,000 | 880 | **5,880** |

The 5,000 core presentations are exactly 1,000 positives plus 4,000 missing-one negatives across splits. Extra masks are a secondary mechanism/stress panel, not equally powered evidence for all 16 assignments. They share their parent cluster; do not count them as independent positive cases.

Use a fixed sampling mixture rather than quotas: independently choose plain current-factual prose with probability 1/2; allocate the other 1/2 equally among resolved history/hypotheticals, irrelevant quoted policy, condition words attached to a different incident, and nuanced internal/external or production/staging scope. A simple isolated denial alone does not qualify as this difficult rhetorical stratum. Sample each scenario independently from its split's declared banks and mixture. Report realized counts; do not resample to force favorable template balances or results. This gives substantial stress coverage in expectation and preserves the stated generator-based inference. Include at least four confirmation template roots with sentence organization different from development in the sampling bank. Also include explicit policy-only/neutral-ticket controls during development, a negative ticket repeating all policy keywords, and unrelated keyword-bearing text. They detect an encoder responding to the common instruction rather than the ticket. Report these controls separately; do not include them as extra independent main samples.

Manually inspect at least 10 clusters from every split plus every clause/template definition before freezing; the coding agent's inspection is an internal audit, not independent human annotation. Save the complete JSONL, source generator, clause banks, split manifest, and hashes. Keep confirmation outputs inaccessible to tuning commands until the protocol lock. The agent can see generator code; it must not use confirmation predictions or feature statistics for decisions.

### 4.3 Retention corpus and accessible fallback

Retrieve BoolQ through `google/boolq`, or the official `https://storage.googleapis.com/boolq/train.jsonl` and `https://storage.googleapis.com/boolq/dev.jsonl` files if the dataset wrapper is unavailable [R15]. Use a minimal JSONL/parquet reader; no large evaluation framework is needed. Record source revision, file hashes and license. The official released training and development counts are 9,427 and 3,270; the latter is this study's held-out retention pool.

Normalize whitespace only for duplicate detection, group by normalized-passage SHA-256, and select at most one question per passage by stable hash. Discard passages that overlap across train and development. Filter serialized inputs longer than 512 tokens **before** model scoring; do not truncate evidence from passages. Report the exclusion count, length distribution, and resulting short-context scope. From the remaining groups, take a seeded hash order with the natural label mixture:

| Use | Source pool | Independent passages |
|---|---|---:|
| Fit protected examples | official training | 400 |
| Calibration protected examples | disjoint official training | 300 |
| Confirmation retention | official development/validation | 1,000 |

During scope-classifier fitting, every retention example has edit-scope label 0 regardless of its true Yes/No answer. Its actual answer remains the task-accuracy target for reporting. Do not balance the confirmation labels after looking at results. Report results separately for true label and base prediction.

If insufficient eligible passage groups remain, the resource decision rules may reduce confirmation to 600, then 300; report the reduced scope and exact interval requirements. If both public retrieval routes fail, complete the ticket/math work but classify the preservation claim as blocked/inconclusive. Synthetic retention text cannot silently replace this real-text check.

### 4.4 Independent units and leakage control

The independent unit for ticket inference is the sampled **scenario cluster**, not each of its counterfactual variants. Six hundred independent draws from the fixed confirmation grammar can support conditional inference for that specified generator. They do not create 600 independent linguistic styles. Report per-template-root performance and a template-root resampling sensitivity analysis; generalization to arbitrary support prose remains unestablished. If the claim is about new template populations, the template root becomes the independent unit and intervals will be wider.

Fit data determine feature scales, candidate rankings, discrete group search and classifier coefficients. Calibration chooses only among the finite, already fitted finalists and their scalar operating points. It is development data, not independent confirmation. Freeze everything—including deployment precision, selected columns, IDs and point assignment, thresholds, amplitudes, prompts, example IDs, and report code—before a single confirmation run. Test-driven changes require a fresh untouched dataset and are outside this study's fixed confirmation budget; otherwise label them exploratory.

## 5. Computational specification

### 5.1 Inference and cache contract

For each prompt, record: sample/cluster/split IDs; token count and padded length; checkpoint and prompt hashes; device/dtype; baseline answer logits; full-vocabulary log-sum-exp; top-two logits/IDs; highest logit outside the answer pair; and final-position residual vectors at layers 13 and 17 during development. Collect both permitted layer hooks in the same frozen-model forward when cheap. Do not run two entire model passes merely to obtain two layers.

During fit/calibration, encode full SAEs in bounded chunks or store final residuals and encode sequentially. Never retain all-token data. Store nonfinite flags and errors, not just successful rows. The primary feature outputs are `[N,16384]` only during discovery, followed by `[N,64]` candidate-pool and `[N,13]` selected-column caches; residuals are `[N,1152]`. Use local `.npy`/`.npz` and JSONL, not a database.

After selecting the layer, confirmation requires only that layer's residual, the final 13-column guard path, the frozen 64-column baseline path, and the raw-residual baseline. Recompute fit and calibration selected columns using the exact deployment path before choosing final operating points. Do not treat full-versus-sliced threshold disagreements as negligible because preactivation differences are small.

### 5.2 Bounded feature search

This is an explicitly heuristic, supervised search, not an oracle and not a theorem-backed discovery algorithm.

1. For each allowed layer, compute valid $q_j$ and normalized fit features. Define activation prevalence as the fraction of examples with $a_j>0$; use positive prevalence $p_j^+$ on fit positives, and $p_{jg}^-$ on each of the four near-miss groups, the pooled extra-mask negatives, and fit retention examples. Rank by $p_j^+-\max_g p_{jg}^-$, descending; break ties by larger mean normalized positive activation, then smaller feature ID. Exclude nonfinite scales/features. Retain the first 64 valid coordinates, or all if fewer; fewer than 13 is a layer-specific failure.
2. Use four initial ordered 13-tuples: top 13; 13 evenly spaced ranked coordinates at zero-based indices `floor(k*(K-1)/12)` for k=0,...,12 in a pool of K≥13; and two uniform without-replacement selections with fixed search seeds. Preserve order as point assignment. Do not duplicate a coordinate within a tuple.
3. For each initial tuple, perform at most four sweeps. In each sweep propose 64 moves: 32 replacements of a uniformly selected position by a uniformly selected currently unused pool feature, and 32 swaps of two distinct positions. When the pool has exactly 13 features, skip the unavailable replacement moves without allocating extra swaps. Generate proposals deterministically from the recorded seed. Remove duplicates without adding replacement proposals. Evaluate all proposals on fit data and accept the best strict improvement over the incumbent according to the objective below. Stop that start when no strict improvement exists.
4. Save one best tuple per layer. Ties between tuples use lexicographic feature-ID order. There are at most $4(1+4\times64)=1,028$ tuple evaluations per layer and 2,056 total. Count early evaluations, duplicates, failed trials, and repairs in the ledger; do not expand the pool or restart campaign after calibration disappoints.

For a tuple, let $g_i=G(z_i)$, $\varepsilon_i=z_{i,(3)}$. For each scalar candidate α below, compute fit positive repair rate, the number of previously No predictions flipped to Yes divided by **all examples in that near-miss stratum**, retention disagreement, and useful-tube coverage. A feasible fit setting requires maximum near-miss flip rate ≤1%, retention disagreement ≤1%, and at least 90% of fit retention examples in the fixed 0.1 tube. These fit constraints are search preferences, not statistical guarantees. Among feasible settings, maximize positive repair rate; then minimize retention disagreement; then maximum near-miss flip rate; then mean retention $\alpha L(\varepsilon)$; then α. If none is feasible, minimize $v=\max(0,\max_g h_g/0.01-1,h_U/0.01-1,(0.90-c_U)/0.10)$, where $h_g$ is a near-miss flip rate, $h_U$ retention disagreement, and $c_U$ fixed-tube coverage; then apply the same utility tie-breaks, retaining its explicit infeasible status. This retains a diagnostic candidate without relabeling it a success.

Pilot tuple evaluations decrement the same 2,056-evaluation counter. Prefer pilot ranking, support and envelope diagnostics; if a top-13 guard is evaluated, record that trial and truncate later proposal batches in their prescribed order when the global cap is reached. Changing scale estimates or data size does not erase the pilot trial. The search is allowed to learn coordinates and their assignment, not polynomial coefficients. Every evaluated candidate remains the same positive weighted design. Do not infer that selected features are causal semantic units from their activation statistics or human-readable labels.

### 5.3 Exact finite scalar search and obstruction checks

Use no gradient steps for the proposed patch. For fixed nonnegative scores $g_i$, ideal classification changes only at $\alpha=-m_{0i}/g_i$. Define the allowed scalar candidate set as 0, 8, every breakpoint strictly inside the interval, and one midpoint of each adjacent interval. Add the nearest representable values immediately below and above each breakpoint, within [0,8], using `nextafter` in the deployed output-write dtype. Form candidates in FP64, cast them to the deployed dtype, deduplicate, and evaluate **every retained candidate in the actual output-addition arithmetic** with the strict-Yes tie rule. This is an exhaustive search over that declared finite candidate set; it is not a claim of a minimum over all real amplitudes or global machine-precision optimality. All continuous tie-breaks, including smaller amplitude, are evaluated only on that finite set. Never divide by $g=0$.

For a fixed guard, sort ideal events $-m_{0i}/g_i$ once and aggregate tied events by stratum to obtain cumulative ideal repair/flip counts in $O(N\log N)$. These counts can order work, but ideal infeasibility alone must not discard a candidate: evaluate every retained candidate's deployed-precision feasibility in bounded vectorized chunks rather than storing a giant candidate-by-example tensor. The finite allowed set and actual-precision checks remain authoritative.

This yields a cheap empirical feasible-range diagnostic. A positive example needs $\alpha >(-m_0)/g$ to flip, while a previously correct negative requires $\alpha \le(-m_0)/g$ to stay No. To achieve a diagnostic target margin μ=0.01, use $\alpha \ge(\mu-m_0)/g$. If $g=0$ and the base is wrong, this guard cannot repair that example. If $8g\le-m_0$, the amplitude cap prevents repair. Existing base false positives are immutable under this patch. Report these counts before benchmarking.

Also compute an optimistic **candidate-pool upper bound**: for each positive, take its 13 largest normalized coordinates from the permitted pool, enumerate their 715 four-products, and average the 13 largest products. Every possible ordered 13-feature projective guard from that pool is at most this value: each of its 13 distinct products is bounded by the corresponding top-product envelope. This bound allows per-example choices and ignores incidence restrictions, making it optimistic; it is a diagnostic, not an implementable predictor. If even α=8 times that bound cannot repair an example, no allowed group drawn from **that fixed cached feature vector** can repair it. This is stronger than local-search failure for the cached representation. Full and selected-column GEMMs can differ near JumpReLU thresholds, so the cached envelope alone cannot rule out every selected-column deployment path. Do not use it as an implementation-wide analytic stop unless all candidate groups provably share that coordinate evaluation or conservative coordinate upper bounds cover every permitted path. Otherwise retain it as a scoped cached-representation diagnostic and evaluate the chosen finalists with their actual deployment encoder.

### 5.4 Calibration and freeze

Evaluate the two fit-derived layer finalists on calibration once. Refit no scales, pools, tuples or assignments there. Choose each operating point with the scalar procedure above, subject to:

- no more than 1% new false positives in each near-miss stratum, with denominator **all** examples in that stratum, and total observed specificity at least 95% separately in every stratum;
- no more than 1% retention disagreement;
- for a nonzero intervention, at least 95% calibration retention **useful margin-qualified certificate coverage** as defined in §6.2, which includes membership in the fixed 0.1 tube;
- α≤8.

Keep α=0 as the no-edit reference, but it cannot qualify as a useful nonzero finalist. Choose the feasible nonzero finalist with the greatest positive-case gain, then smaller retention disagreement, smaller worst near-miss flip rate, larger useful certificate coverage, and smaller α; final ties favor layer 13. Full near-miss accuracy, including pre-existing mistakes, remains a separate requirement. If no nonzero finalist is feasible, retain a diagnostic candidate by minimizing $V_{\rm cal}=\max(0,f_{\max}/0.01-1,(0.95-s_{\min})/0.05,d_U/0.01-1,(0.95-c_U)/0.05)$, where $f_{\max}$ is the largest new-false-positive rate, $s_{\min}$ the smallest total specificity, $d_U$ retention disagreement, and $c_U$ useful certificate coverage. Break equal maxima by the smaller sum of these nonnegative violations, then the same utility/tie-break sequence. This candidate is eligible only for the locked diagnostic branch; it cannot become a success by relaxing constraints. If the base already has specificity below 95% in a calibration stratum, the one-sided patch cannot repair that observed deficit; record the operational infeasibility and its development-distribution scope.

These finite-sample constraints do not prove ≤1% population risk. In particular, 150 clean calibration scenario clusters are insufficient for that claim. Confirmation alone supports rate claims. Store a hash-locked configuration containing every fitted object, calibration selection result, candidate count, data hash and code commit before confirmation.

### 5.5 Baselines and controls

Use the same cached frozen-model logits, prompts, fit/calibration IDs and correction cap. The only additional model forward passes are prompt selection, verification, confirmation feature collection, and genuine end-to-end timing. The reference methods are:

| Method | Specification | Question it resolves |
|---|---|---|
| Prompted frozen base | Selected explicit-policy prompt, α=0 | Is there a useful repair opportunity? |
| Uniform shift | (g(x)=1), same scalar calibration | Does selectivity add anything over answer-bias calibration? |
| Compiled projective guard | The frozen 13 features, point assignment and 78-unit branch | Primary mechanism. |
| Direct projective reference | Same polynomial computed by direct products | Detects compiler/precision artifacts; not a separate scientific treatment. |
| Symmetric $E_4$ | Same 13 features, normalized DP, own calibrated α | Same worst-case minimax certificate, different target response and computation. |
| Logistic gate, SAE13 | Fitted scope score on the same normalized 13 features | Is a simple decision surface sufficient? |
| Logistic gate, SAE64 | Fitted on the fixed pool from the selected layer | Is the 13-coordinate restriction or group search the bottleneck? |
| Logistic gate, residual | Fitted on all 1,152 coordinates at the same hook/position | Has the SAE/normalization discarded relevant information? |
| Tube-cut projective control | $g=0$ when $z_{(3)}\le0.1$, otherwise $g=G(z)$ | Could an inexpensive hard boundary achieve the practical preservation result? |

The hard tube cut is outside the theorem's polynomial class and is discontinuous at its boundary. It preserves the declared tube exactly in real arithmetic. Give it full operational credit if it wins; do not imply the polynomial lower bound excludes it. Test boundary perturbations around 0.1. There is no need to build it as a SwiGLU network or to call it a new method.

For each logistic family, fit a CPU FP64 L2-regularized classifier with an intercept and a weighted `StandardScaler` fitted on fit data only using the same sample weights as the classifier, and scope label 1 for ticket positives, 0 for all ticket negatives and retention. Give each of seven strata equal total fit weight: positives, four missing-one groups, extra-mask negatives, and retention. For $N_{\rm fit}$ total records and stratum size $n_g$, set every row's weight to $N_{\rm fit}/(7n_g)$, so the mean weight is one and the regularization scale is defined. Record these exact weights. Require every stratum to be nonempty. The scaler mean and variance use those weights divided by their sum; zero-variance columns use scale 1. This choice applies identically to all three classifier families. Use $C\in\{0.01,0.1,1,10,100\}$, `lbfgs`, max_iter 2,000, tolerance $10^{-8}$. One increase to 4,000 iterations within the repair budget is allowed for reported nonconvergence; otherwise mark that fit failed. Check a tiny separable reference problem and agreement with an independent logistic-loss evaluation. Do not remove the intercept to make the baseline weaker.

Use the classifier's finite raw decision function $r(x)$ for hard scope gating, $g(x)=\mathbf1\{r(x)>t\}$. For each fitted $C$, calibrate $t$ over all unique calibration decision-score boundaries/midpoints plus $-\infty,+\infty$, explicitly including always-on and always-off gates. Let $s(x)=\operatorname{sigmoid}(r(x))$ for the separate soft-gate sensitivity. For hard gates, build one common finite amplitude grid by applying §5.3's candidate construction to all calibration examples with hypothetical $g=1$; use that same grid for every scope threshold. This declared grid keeps amplitude choice independent of which rows the gate activates. Sort it and the scope scores once. For each base-No example, find its first **actual rounded-write** Yes bin on that grid with $g=1$, using a binary search justified by the monotone positive write; use a sentinel if it never flips. Construct stratum-specific integer counts from these actual bins and scope thresholds, using a two-dimensional prefix table or a scope sweep. This evaluates all retained operating points in actual decision arithmetic with $O(N^2)$ simple count work per fitted classifier, without an $O(N^3)$ example loop. Check the count-table results against explicit evaluation on a small case and verify selected operating points directly. Apply the same new-false-positive, total-specificity, retention-disagreement, amplitude constraints and positive-gain tie-breaks to every operational baseline. The useful-certificate constraint applies to the projective and symmetric certified methods, not to learned gates or the uniform shift. The tube-cut control can report its own exact-zero region; do not require it to inherit the polynomial theorem. Also report a sigmoid gate $g=s(x)$ for the selected $C$ as a secondary cached sensitivity, with §5.3's finite scalar search and without another classifier fit. Select $C,t,\alpha$ on calibration; freeze all. There are at most 15 core logistic fits, plus bounded convergence repairs. These are learned controls; “no base-model gradients” is true for them too and is not a differentiator.

SAE13 must consume the exact same authoritative 13-column outputs as the guard, not a separately rounded slice of the 64-column GEMM. Its scaler is fitted on those finalized fit features. SAE64 has its own recorded 64-column path. The SAE13 comparison matches inputs, while SAE64 and residual comparisons deliberately grant wider information to expose upstream bottlenecks. Report their larger parameter counts and feature costs. All get the same labels and protected examples; none sees confirmation data. Count the proposed method's 2,056 group-search trials and scale estimation as tuning cost instead of pretending it has zero training cost. Do not equate these simple controls with complete reproductions of SERAC, GRACE, DSG, or LoRA.

### 5.6 End-to-end pseudocode

```python
# Constants and all dataset hashes are fixed before confirmation.
inspect_environment_and_access()
generate_and_audit_raw_data()
verify_math_and_compiler_on_cpu()
fetch_authorized_tokenizer_model_and_only_named_sae_files()
finalize_serialized_prompts_answer_tokens_eligible_manifests_and_hashes()
load_base_and_only_named_saes()
verify_template_tokens_hooks_padding_encoder_and_precision()
calibrate_resources_or_stop()
base_prompt = choose_between_two_predeclared_prompts(
    first_40_fit_clusters_excluding_demonstration_clusters
)
freeze_chosen_prompt_hash_and_update_resource_forecast()

for split in ["fit", "calibration"]:
    for batch in bounded_batches(split):
        logits, h13, h17 = frozen_forward(batch, final_logits_only=True)
        append_cache_and_cost_ledger(logits, h13, h17, batch)

finalists = []
for layer in [13, 17]:
    fit_a = encode_full_sae(cached_fit_h[layer])
    scales, pool = fit_scales_and_rank_pool(fit_a, fit_labels_and_strata)
    envelope = optimistic_pool_response_bound(fit_a, scales, pool)
    candidate = bounded_ordered_tuple_search(pool, fit_data, scales)
    finalists.append(candidate_with_fit_obstruction_counts(candidate, envelope))

verify_and_recompute_selected_column_path(finalists, fit_and_calibration_h)
main = select_finalist_and_amplitude_on_calibration(finalists)
controls = fit_and_calibrate_fixed_baseline_families(main, fit, calibration)
apply_mechanism_gate_or_write_scoped_stop_report()
verify_online_wrapper_matches_cached_evaluation(on_development_examples=32)
freeze_and_hash_dependency_environment()
write_protocol_lock(main, controls, hashes, budget_remaining)

for batch in confirmation_in_predeclared_order():
    base_logits, selected_h = frozen_forward(batch)
    z13, pool64 = frozen_feature_maps(selected_h)
    evaluate_all_frozen_methods_and_certificates(base_logits, z13, pool64, selected_h)
    append_raw_results_and_cost_ledger()
    enforce_resource_stop_without_changing_methods()

verify_online_wrapper_matches_cached_evaluation()
run_bounded_precision_controls_and_actual_timing()
compute_intervals_and_failure_accounting()
write_report_and_regeneration_commands()
```

NaN, infinity, corrupt cache/hash, missing required feature, or shape mismatch is an explicit failure. Do not replace invalid scores with zero, drop them from denominators, or let a harmless-looking zero patch conceal a broken encoder.

## 6. Correctness, experiment matrix, and analysis

### 6.1 Tiny correctness before model interpretation

Implement these checks as ordinary fast scripts/tests. They test actual scientific risks; do not build a general theorem-prover or reproduce unrelated notebook experiments.

| Check | Required result or interpretation |
|---|---|
| Projective incidence | 13 distinct blocks of size 4; each of 78 pairs covered once; equal weight sum exactly one symbolically. |
| Independent Boolean enumeration | All 8,192 vectors agree with line counts/13; recover 13 active size-four supports and the line-complement counterexample. |
| Sparse protection | Enumerate all size-0/1/2 supports, test signed amplitudes, and test gradients from an independent direct-polynomial derivative. Values and gradients must be zero on exactly sparse inputs. Three-sparse values are also zero, but their gradients need not be. |
| Tube corners | All 78 positive supports at ε in {0.01, 0.1, 0.4, 0.9} equal the formula. For the smaller seven-feature $PG(2,2)$ case, enumerate all signed box vertices to exercise sign handling. |
| Matrix branch | Correct shapes, 234 nonzeros, 52/26 hidden widths; compare actual SiLU matrices with direct products on sparse, random signed, near-target and threshold-adjacent inputs. |
| Symmetric reference | DP equals explicit enumeration on tiny vectors; $E_4(\mathbf1)=1$, same tube corners, and $126/715$ on the nine-active line complement. |
| Signed-design negative control | $p=z_1z_2+z_3z_4-\tfrac12(z_1z_3+z_2z_4)$ has balanced one-subset marginals but exceeds its optimum at ((1,ε,0,0)). This detects an invalid signed-sufficiency claim. |
| Positive synthetic control | Supply $z=\mathbf1$, a known negative base margin, and a valid α that flips it; sparse protected vectors must stay unchanged. This tests the write path, not real feature discovery. |
| Semantic-copy negative control | Copy four predicate bits onto 13 point groups; demonstrate nonzero near-miss responses. Never use those copied bits as the real feature map. |
| Loader and position | Assert matching IT config/site/shapes; compare the hand encoder against the published convention and an independent implementation on fixed residuals. Verify final real token under padding. |
| Full versus selected encoder | Compare values, active masks and distance to thresholds on development residuals; freeze the actual selected path and repeat affected development calibration if needed. |
| Wrapper integration | On at least 32 diverse development examples, online patched logits/predictions agree with cached recomputation and all other logits are unchanged. A zero α and a zero guard must reproduce the base. |

Run direct references in CPU FP64. For the guard on its fixed cube, require maximum absolute discrepancy ≤$2\times10^{-6}$ in FP32, and ≤$10^{-12}$ in FP64; additionally record relative error where |G|≥0.001. Exactly sparse zero-input structure should yield exact zeros in the selected implementation. Use at least the notebook's 5,000 random signed points, 1,000 near-target points, and 5,000 sparse points once; deterministic reruns do not require many seeds. Do not divide by the measured compiled value at the dense target to hide its rounding error.

These tolerances are conservative engineering checks relative to the 0.0102 margin scale, not universal floating-point error theorems. If a guard path fails, allow one correction/backend repair; the predeclared correct fallback is CPU FP64 guard arithmetic, timed and recorded. If even the reference fails, stop the empirical study as invalid pending a mathematical/implementation diagnosis. Never widen a tolerance because the comparative result becomes favorable.

For model-backend checks, use 16 representative short prompts and the same weights. Compare MPS against a small CPU FP32 reference when feasible, including actual logits and captured residuals; record discrepancies and answer disagreements. An absolute logit discrepancy over 0.05, a nonfinite tensor, or disagreements on base margins above 0.1 triggers investigation before proceeding. These are screening thresholds, not proof of equivalence. Use the same accepted model/backend/precision for all comparative methods, so cross-backend differences cannot masquerade as an edit effect.

### 6.2 Per-example numerical certificates

Save the actual FP32 normalized coordinates, deployed α, base logits and patched logits. On all confirmation retention examples, independently compute (G(z)) and $L(z_{(3)})$ using Python `Fraction` on the exact represented floating-point inputs and the exact coefficient $1/13$. Rationalize the saved individual base and patched logits, normalized coordinates, and deployed α before subtracting; do not first round a difference. For ideal fixed-tube membership compare with `Fraction(1,10)`, not a rounded FP32 0.1. There are only 13 quartic terms per example; this is a bounded small calculation. It verifies a finite set of observed inputs without assuming that empirical maximum errors bound every future input.

For each such example define δₘ as the absolute discrepancy between the actual observed binary-margin change and the ideal real-arithmetic change α(G(z)). Then

\[
|m'_{\rm actual}-m_{0,\rm actual}|\le \alpha L(z_{(3)})+\delta_m
\]

is checked pointwise. Use exact or outward-rounded comparisons when classifying a point as certified. Record δₘ, the guard discrepancy, and the two output-addition errors separately. This does not establish a universal FP32/Metal implementation bound or immunity to a changed feature map. Retention results must state “real-arithmetic guarantee with numerical agreement checked on the evaluated inputs,” not “formally verified model.”

Apply the same procedure to the symmetric baseline using an independently computed exact normalized $E_4(z)$ and its own observed margin discrepancy in place of $G(z)$ and δₘ. Its theoretical tube bound is the same $L$, but its floating-point discrepancy must be measured against its own polynomial.

Define useful binary certificate coverage as the proportion of retention examples satisfying all of:

\[
z_{(3)}\le0.1,\qquad |m_0|>\alpha L(z_{(3)})+\delta_m,\qquad
\alpha>0.
\]

The α>0 requirement prevents the no-edit control from qualifying as a successful certified intervention. Also report exact sparse coverage, fixed-tube coverage without the margin condition, and the analogous full-vocabulary sufficient-margin check. For the latter, bound the maximum actual perturbation of either answer logit, including rounding, and require the base top-two margin to exceed twice that bound.

If a putatively certified example changes its binary decision, investigate a certificate, ordering, precision, or logging bug. It is an invalid result, not an empirical exception to the theorem. A point outside the useful region can still be unchanged; do not credit that event to this certificate.

### 6.3 Smallest sufficient experiment set

1. **Algebra/compiler verification:** the tests above; no language model required.
2. **Prompt/headroom and feature-geometry pilot:** up to 50 fit clusters initially, 100 fit retention passages, and the policy/distractor controls. Inspect base errors, support counts, $G$, $E_4$, ε₃, correction thresholds and pool envelopes. The pilot can reveal a broken hook, format detector, dead guard, or prohibitive cost; it cannot establish a 1% rate.
3. **Development and locking:** finish fit/calibration caches, bounded search and baseline fitting. Use only these splits for decisions.
4. **One locked comparison:** all frozen methods on the same confirmation examples, with one primary comparison and secondary diagnostic comparisons. All-mask panels and per-rhetorical-type breakdowns are predeclared robustness checks.
5. **Implementation controls and timing:** direct versus compiled equivalence, actual selected-column inference, fixed-count coordinate-permutation diagnostic, precision sensitivity on a fixed subset, and end-to-end timing. No new feature search follows confirmation.

The coordinate-permutation diagnostic uses 16 seeds fixed before confirmation and reports the distribution of direct-polynomial responses/accuracy on the already cached vectors, not the best permutation. Every permutation has the same minimax bound, and its expectation is $E_4(z)$. This probes assignment sensitivity. It is not a second opportunity to choose a deployed guard or an ensemble for the headline result.

Use a fixed 128-example development subset for FP32 versus FP64 auxiliary-branch comparisons and for small perturbations around JumpReLU and tube-cut thresholds. Perturbations are mechanism stress tests, not draws from the language distribution. If the main dtype changes after these checks, repeat all affected development work before lock and charge the cost.

### 6.4 Metrics and primary decision rule

Report counts before percentages. Main task summaries are positive sensitivity and the specificity of each missing-condition stratum. The balanced five-stratum score is their equal-weight mean; raw overall accuracy on a 1:4 artificial mixture is secondary. Extra masks are reported individually and as a separate panel, never pooled to inflate the primary sample size.

For every method record:

- base and patched positive accuracy, repaired positives, unrepairable positives ((g=0), insufficient α cap, or interference constraint), and positive-margin changes;
- all four near-miss specificities, their pre-existing base errors, new false positives, and corrected negatives if a comparator permits them;
- retention base/patched accuracy, correct-to-wrong and wrong-to-correct transitions, total decision disagreement, and class-specific results;
- exact sparsity, ε₃ quantiles, useful certificate coverage, raw/normalized feature saturation, $G$/$E_4$ quantiles, and gain-amplified leakage bounds;
- full-vocabulary argmax changes separately from restricted Yes/No decisions, plus probability-distribution drift;
- setup, acquisition, search/fitting, confirmation, and deployment costs; numerical failures, omitted/failed runs, and remaining budgets.

A **material activation** is a margin shift of at least 0.1, declared in advance and reported as a secondary rate. The main unintended-behavior event is a changed restricted decision on retention. Do not call every (G>0) a false activation: continuous guards can have small nonzero values without a meaningful effect.

The primary practical-success claim requires the frozen projective patch to satisfy all of the following on confirmation:

| Requirement | Evidence threshold | Rationale |
|---|---|---|
| Useful positive repair | One-sided 95% lower bound on positive-accuracy gain ≥0.05 | At least a five-point improvement is the predeclared practical benefit. |
| Near-miss rule fidelity | One-sided 95% lower specificity bound ≥0.95 for **each** of four missing-condition types | No missing condition may be hidden in a favorable pooled score. |
| Retained decisions | One-sided 95% upper bound on total retention disagreement ≤0.01 | Bounds loss of prior behavior, regardless of whether some flips improve labels. |
| Useful certificate | One-sided 95% lower bound on useful binary certificate coverage ≥0.95, with α>0 | Most retained behavior must lie in a quantitatively informative protection region. |
| Valid implementation | All essential correctness, input-split, precision and online-wrapper checks pass | Performance without the specified mechanism is not evidence. |
| Feasible local execution | Measured study and deployment paths stay within the caps | No unseen remote computation or uncounted preprocessing. |

The positive patch is monotone toward Yes, so its positive-accuracy gain is the fraction of positives that change from No to Yes; there can be no positive-class regressions in the defined binary interface. Use an exact binomial interval for that repair indicator, conditional on the fixed generator. A supported primary success claim requires valid base and primary-patch outcomes for **every prescheduled primary confirmation row**. Unresolved missing or numerical outcomes make the primary comparison incomplete/invalid; report failure rates and worst-case sensitivity bounds, and do not fold a potentially regressed unknown case into a successful Bernoulli repair calculation. With complete valid outcomes on 600 independent positive scenarios, **40 repairs** give a one-sided 95% lower bound about 0.05070; 30 repairs merely give an observed 5% gain, with lower bound about 0.03622. An observed 5% gain alone does not pass the chosen threshold.

At 600 examples per near-miss type, 580 correct are needed for its one-sided 95% lower accuracy bound to reach 95%. At 1,000 independent retention passages, at most four disagreements are compatible with a one-sided 95% upper bound ≤1%; five give about 1.0484%. For useful-coverage success, at least 962/1,000 are needed to place the one-sided lower bound at 95%. Recompute all thresholds for actual sample counts using beta quantiles; do not hardcode rounded percentages as decisions.

For zero events the exact upper bound is $1-0.05^{1/n}$: approximately 1.98% at $n=150$, 0.994% at $n=300$, and 0.299% at $n=1000$. This explains why the calibration set does not certify the retention target and why fewer than 300 independent retention observations cannot support it even with no failures.

The ≤1% total-disagreement requirement also gives a conservative ≤1 percentage-point bound on accuracy degradation, because accuracy loss cannot exceed disagreement. Still report the paired accuracy difference and its one-sided uncertainty interval; do not compare two independent accuracy intervals. A nonsignificant degradation test is not a preservation proof. Report the original 95% positive-accuracy aspiration and whether its lower bound reaches 95%, independently of the repair endpoint.

### 6.5 Uncertainty, comparison fairness, and failure accounting

Use one-sided 95% Clopper–Pearson intervals for the specified independent Bernoulli rates. For paired five-stratum summaries and method differences, use 5,000 deterministic bootstrap resamples of whole scenario clusters; for retention, resample passage groups. Preserve stratum structure and paired method outcomes in each draw. Include the template-root sensitivity described earlier. No repeated model seeds are needed for deterministic cached scoring; repeated timings measure hardware variability, not statistical accuracy uncertainty.

There is one primary compound success claim. Requiring every component threshold to pass is an intersection-union decision; do not describe the displayed individual intervals as jointly simultaneous 95% intervals. Comparisons among the other methods are secondary estimates, not a winner selected from many test-set p-values. If claiming superiority to a named baseline, require the paired uncertainty interval to support it and account explicitly for any multiple confirmatory comparisons; otherwise state the observed difference descriptively.

An α, feature group, classifier $C$, or threshold chosen on calibration stays fixed even if another test operating point looks better. You may display a test tradeoff curve for understanding only if labeled retrospective/descriptive; it cannot replace the locked endpoint. Highlight when simple gates or $E_4$ equal or outperform the guard. The mathematically optimal worst-case bound does not promise the best empirical decision boundary.

Failed forwards remain rows with failure status. A common base-forward failure prevents all methods from receiving that input; report its rate and worst-case bounds instead of silently shrinking the denominator. A method-specific numerical failure counts against that method in conservative sensitivity analysis; unresolved failures preclude the primary success claim as specified above. A failed classifier fit is listed as failed, not omitted as if it had never been attempted. One retry of a failed batch is allowed after an identified bounded repair, and both attempts count toward compute. If a required strong baseline is invalid and cannot be repaired, the comparative conclusion is invalid or limited, not proof of superiority.

Preprocessing exclusions must be based only on predefined length, malformed-schema, duplicate, or semantic-generator criteria. Record the number before/after each filter. If late label defects are discovered, flag all affected template roots, provide sensitivity with and without them, and classify the affected claim as invalid unless an untouched, budget-compliant confirmation can resolve it. Do not silently relabel examples because the model disagreed.

For a nonzero result with wide intervals, distinguish a promising estimate from supported practical benefit. If a valid test places the upper bound on positive gain below 5%, it rules out the chosen useful effect under that protocol. If the interval crosses 5%, it is inconclusive about the practical threshold. Similar logic applies to preservation constraints. Failure of a heuristic search alone is not a theorem of nonexistence.

### 6.6 Probability drift without storing every logit

For the output write, cache base probabilities $p_Y,p_N$ using log-sum-exp. Compute the **actual observed** shifts $\Delta_Y=\ell'_Y-\ell_Y$ and $\Delta_N=\ell'_N-\ell_N$ from the stored individual logits in FP64 or exact rational arithmetic. Rounded FP32 additions need not give exactly symmetric shifts, even when the ideal write is symmetric. The normalization ratio is

\[
Z=1+p_Y\operatorname{expm1}(\Delta_Y)+p_N\operatorname{expm1}(\Delta_N).
\]

Then $D_{\rm KL}(p\|p')=\log Z-p_Y\Delta_Y-p_N\Delta_N$. Total variation is

\[
\mathrm{TV}=\tfrac12\left[
p_Y|e^{\Delta_Y}/Z-1|+
p_N|e^{\Delta_N}/Z-1|+
(1-p_Y-p_N)|1/Z-1|
\right].
\]

Use stable FP64 `expm1`/`log1p` evaluation and verify against explicit softmax vectors on tiny cases. The real-arithmetic special case is $\Delta_Y=-\Delta_N=\alpha g/2$. These are derivations for this two-logit write, not claims from the notebook. Planning independently checked the actual-shift formulas on 100 small randomly generated FP32 logit/write cases; maximum discrepancies against explicit FP64 softmax were approximately $5.1\times10^{-16}$ for KL and $3.4\times10^{-16}$ for TV. This was a small arithmetic probe, not an LLM test.

Store compact statistics rather than every vocabulary logit. Full argmax is reconstructed from the **actual** two patched logits and the highest unchanged competitor. None of these metrics establishes long-form generation preservation.

## 7. Local feasibility, resource calibration, and evidence gates

### 7.1 Cumulative hard ceilings

These are planning limits, not measured M3 Max performance estimates. Keep a single append-only budget ledger spanning all development, failed attempts, baselines, fallback paths and repairs. Changing a stage or backend does not reset it.

| Resource | Ceiling / rule |
|---|---|
| Active experimental wall time | **6 hours total**, including extraction, search, fitting, correctness runs, repeats, numerical controls, analysis computation and 20 minutes of repairs. Unattended waiting for a remote checkpoint download is tracked separately, not hidden. |
| Informative pilot | Aim for ≤55 minutes after the environment/data assets are prepared; do not promise an unmeasured speed. |
| Model forward work | At most **12,000 sequence-forwards** and **4,000,000 padded input tokens**, whichever binds first; includes prompt trials, retries, calibration, timing and precision rechecks. No autoregressive generation campaign. |
| Feature search | ≤2,056 ordered tuple evaluations across two SAEs; 16 frozen permutation controls only; no extra semantic-feature searches. |
| Learned controls | 15 core logistic fits; at most one documented convergence retry per failing fit within the common repair/time budget. |
| Backend/model attempts | FP32 MPS primary; at most one supported BF16 model attempt and one CPU diagnostic path. All changed caches and retries count. |
| Memory | Conservative live-workload cap $M=\min(20\ \mathrm{GiB},0.60\times\mathrm{physicalRAM},0.75\times\mathrm{availableRAM\ at\ start})$. Keep at least 6 GiB outside that allowance. If (M<8) GiB, do only the calibration needed to decide feasibility before a full allocation. |
| Disk | New project/download/cache use ≤15 GiB and leave ≥10 GiB free. Use one base checkpoint copy and at most two SAE files. If access/download tools would duplicate the model beyond the cap, fix the cache plan first. |
| Repair allowance | **20 minutes total**, at most two material repair episodes after initial implementation; runtime/experimental debugging edits count toward the same time. Initial code authoring has the separate effort estimate below, but every executed probe/test, failed run and retry still consumes the experimental compute ledger. Do not start a new algorithm after exhausting it. |
| Evidence/report reserve | Preserve at least **35 minutes** of the 6-hour cap for analysis, failure accounting, manifest checks and report generation. |

Monitor process RSS and MPS allocated/driver memory; on unified memory they overlap and must not simply be added. Also record system memory pressure and swap growth. The cap is enforced conservatively by either relevant reading approaching it, not by inventing an exact combined metric. Stop or reduce the next batch if peak usage exceeds 90% of the cap. Sustained memory pressure, >1 GiB new swap, or repeated OOM is a resource gate, not a reason to keep retrying. Never change the user's system configuration to defeat memory limits.

Setup/download/install time is separately capped at 90 minutes of active attempts; log actual elapsed download time. Expected implementation and analysis authoring effort is an **estimate of 4–8 active hours**, plus setup and up to 6 hours of experiments: roughly 10–16 hours end to end if the full study proceeds. Early fatal gates can finish much sooner. This is not a multi-day compute study, and these estimates are not Mac timings.

### 7.2 Calibration and permitted resizing

After one correctness-forward batch, time 32 diverse development examples at batch sizes 1, 4 and 8 that fit memory, covering both 256- and 512-token buckets. Warm each accepted path, synchronize MPS immediately before and after measurement, and count warmups. Measure prompt processing, residual transfer, full SAE encoding, selected encoder, guard, output handling and end-to-end latency separately. Observe peak memory and checkpoint-load time.

Estimate remaining work by measured bucket-specific seconds per batch, including padding, with a factor **1.5 safety margin**, plus observed search/fitting times and the report reserve. Maintain parallel forecasts for sequence-forwards and padded tokens. Before collecting large development caches and again before lock, reserve all remaining required prompt trials, verification, numerical rechecks, confirmation and timing against **all three compute caps**, not just wall time. Do not extrapolate from a tiny 32-token example to 512-token batches. The full-feature decoder is unnecessary for large extraction. Preserve the same backend, shape policy and final-position scoring path for fair timing.

The default data plan contains 7,580 model-scored records. The fixed four-path warm timing matrix, if both buckets and batch sizes 1 and 8 are used, requires 2,088 additional sequence-forwards and 801,792 padded tokens (29 batches per path/size/bucket, including warmups). If every data record used 512 tokens, data plus timing alone would be 4,682,752 tokens, exceeding the cap before prompt trials or verification. Therefore the default sample size is conditional on the measured bucket mix; it is not automatically authorized by the headline 6-hour allowance.

Resize only before lock, whenever projected wall time, sequence-forwards, or padded tokens exceed their remaining cap, using this order:

1. Choose the fastest accepted batch size that stays below 80% of the memory cap.
2. Remove redundant cache materialization and use selected-column deployment; keep the scientific algorithm and all required methods.
3. If projected work still exceeds any remaining compute cap after reserving verification/timing/report work, reduce confirmation ticket clusters from 600 to 400, then 300; retain one positive and all four near misses per cluster. Reduce the all-mask confirmation subset proportionally to 30, then 20 clusters.
4. Reduce retention confirmation from 1,000 to 600, then 300 independent passages, recomputing attainable uncertainty and expected decision power. At 300, zero disagreements are necessary to support a 1% upper bound. Do not describe this as an equally precise study.
5. If the minimum 300 ticket clusters and 300 retention passages, the required comparators, fixed verification/timing workload and report reserve still do not fit every applicable cap, stop the confirmatory claim as practically infeasible. Preserve development evidence. Do not replace Gemma 1B with a smaller model or replace natural text with artificial feature vectors and call the original question answered.

Using pilot paired-error rates, calculate the confidence width or power expected at the planned size. If even optimistic plausible effects cannot be distinguished from the five-point threshold, retain the geometry diagnostic but classify practical benefit as inconclusive. The practical threshold itself never moves to accommodate available compute or observed gains.

### 7.3 Evidence gates and exact next actions

The stage allocations below sum to 340 minutes plus the shared 20-minute repair reserve. They are maxima, not targets to spend. Reserve unused time; do not expand experiments merely because an early stage was fast.

| Stage / budget | Question and minimum work | Evidence | Decision and next action |
|---|---|---|---|
| **A. Constructive feasibility — 10 min experimental** | Do source mathematics, model access, exact SAE files and hardware constraints permit the intended construction? Inspect actual environment; run projective incidence/Boolean checks; check asset metadata. | `environment.json`, input/source hashes, analytic counterexamples, initial budget. | Continue if construction and resource access are credible. One bounded path/config repair is allowed. Missing model permissions or absent target Mac blocks empirical validation; finish unaffected checks/report. A verified essential theorem defect triggers analytic stop. |
| **B. Tiny correctness and resource calibration — 15 min** | Are tokens, hooks, encoder convention, compiler and timing boundaries correct? Run tiny tests and representative timed batches. | `correctness.json`, tensor/config audit, timing/memory calibration. | Continue only after essential checks pass and minimum study fits. Adjust batch/cache/device as prescribed. Persistent correctness failure is invalid experiment; measured failure to fit is practical infeasibility. |
| **C. Mechanism pilot — 30 min** | Does the prompted base have repairable misses, and do actual features carry usable conjunction/selectivity signals? Use up to 50 fit clusters and 100 retention passages initially; run policy-only controls. | Base confusion/headroom, full/pool support counts, guard responses, correction-feasibility and cost projections. | Gross hookup/format failures may use the shared repair reserve. Sparse negatives plus zero positive response is failure, not success. Do not reject a noisy small pilot solely on its point estimate. Expand only into already budgeted fit/calibration data to resolve a specific uncertainty. |
| **D. Complete development and freeze — 85 min** | Can the bounded search produce a usable operating point, and can credible controls be fitted? Finish fit/calibration; evaluate at most two finalists; verify selected deployment path; fit controls. | Full search ledger, candidate-pool upper bounds, baseline convergence records, frozen config and all selected IDs. | If a valid finalist has plausible five-point headroom and acceptable calibration tradeoff, lock confirmation. If no candidate is feasible but a locked diagnostic can distinguish lack of response from excess interference within budget, permit one such diagnostic run. If the base itself leaves insufficient headroom, report opportunity limitation/inconclusive mechanism benefit; do not weaken it. If neither useful confirmation nor a discriminating diagnostic fits, stop and report. |
| **E. Locked comparison — 140 min** | Does the frozen method meet the primary claim on untouched data, and what do the controls explain? Evaluate one predeclared sample size in fixed order. | Per-example outcomes, certificates, compact logits, failures and costs for every method. | Complete the fixed run unless a correctness/resource safety gate fails. Do not stop for a favorable p-value, extend until significance, or change thresholds. Incomplete runs remain incomplete; give bounds and an inconclusive/infeasible classification. |
| **F. Precision and actual deployment costs — 25 min** | Does the saved online patch match cached results, and what does it really cost? Run fixed subset checks, finite-point rational comparisons, permutation and tube-cut sensitivities, actual timing. | Numerical errors, online equality check, timing blocks, peak memory, boundary sensitivities. | A compiler/path discrepancy requires diagnosis; if unresolved, invalidate the affected empirical conclusion. A simple comparator winning is a valid result. No implementation optimization campaign after sufficient verification. |
| **G. Diagnosis and report — 35 min** | What conclusion is supported and reproducible? Recompute tables/intervals from raw rows, inspect failures, package only reached-stage artifacts. | Final report, figures/tables, README, manifests, one-command regeneration. | Classify the outcome using §9. State what was not run. Recommend at most one evidence-motivated next step. |

The cumulative time and forward-token caps override individual stage caps. Do not borrow the reporting reserve to complete optional robustness work. A stage transition must write its decision, evidence files, cumulative budget and next action to the ledger.

Headroom decisions must respect uncertainty. A few clean pilot positives do not prove absence of a repair opportunity. The exact one-sided upper bound on the development false-negative rate can be compared with 5%, but any stop on that basis concerns the development distribution, not all unseen prose. Similarly, a candidate-pool envelope obstruction is an actual bound for the specified normalized features and examples; heuristic failure without such a bound has weaker interpretation. Keep those reasons distinct in the report.

## 8. Timing, memory, and fair cost reporting

Measure both **total acquisition cost** and **post-selection inference cost**. Total acquisition includes model/SAE setup, data authoring/preprocessing, prompt trials, all frozen forwards, full SAE computation, scale/ranking work, every tuple proposal, classifier fitting, calibration, failed runs, and numerical verification. Report download bytes/time separately from compute. State that pretrained model/SAE creation costs were external prerequisites; they are not zero, but this PoC is reusing public weights rather than retraining them.

For deployment compare at least base, selected-column compiled guard, selected-column $E_4$, and the best calibration-selected learned gate. Include uniform and tube-cut arithmetic in cached microbenchmarks. Time full-SAE encoding once during development to quantify the discovery/deployment difference. Do not run full model inference once per cached comparator merely for accuracy scoring.

Cold-start timing includes process start, model/tokenizer loading from the local cache, SAE/selected-state loading, and first forward. Record one cold start per retained deployment path if budget permits; otherwise report the single measured common cold load and mark path-specific cold numbers unavailable. Download time is not cold inference latency.

For warm timing, use exactly the four deployment paths named above, both fixed sequence buckets, and batch 1 plus the selected throughput batch (skip duplicate batch sizes). Warm each path five batches, then collect three blocks of eight matched batches per path. Rotate method order between blocks, and synchronize around timed regions. The shared forward-count/token budget includes all timing calls. Report median, observed p95, dispersion across blocks, examples/s and actual tokens/s; p95 from a small sample is descriptive, not a stable production-tail estimate. Report both model-plus-patch time and isolated encoder/guard time with transfers included at their actual boundaries. Host scheduling and thermal state may limit small percentage claims.

Use `time.perf_counter()` with `torch.mps.synchronize()` when appropriate, and standard CPU timing for CPU-only regions. Log RSS, MPS current/driver allocation, cache behavior, swap and thermal/power observations if accessible. CPU and accelerator time overlap cannot simply be summed. Do not subtract a noisy isolated overhead from total time and claim a measured speedup. Exclude initial compilation/warmup only from warm measurements, not from total setup cost.

If comparing amortized costs, write the equation explicitly: $T(N)=T_{\rm acquire}+N\,t_{\rm deployment}$, using measured quantities and a stated number of future requests. Report break-even only when the signs and uncertainty permit it. Parameter count, monomial count, or zero base-gradient steps alone do not establish efficiency. No published CUDA runtime may stand in for an M3 Max measurement.

## 9. Honest outcome interpretation

Use one primary classification, with separate labels for any subsidiary findings:

| Outcome | What qualifies | What does not qualify |
|---|---|---|
| **Supported contribution** | Valid frozen protocol, useful real-feature geometry, primary benefit/preservation/certificate thresholds supported, credible controls and actual costs reported. A narrower supported certificate-geometry finding may be identified separately. | Claiming a missing skill was taught; claiming universal preservation or superiority to excluded fine-tuning; calling generic SAE classification new. |
| **Informative negative result** | A valid, adequately precise locked study rules out the practical effect/tradeoff for this implementation, or a verified mathematical/pool obstruction explains the specified failure. Simpler methods may remove its practical advantage. | “No suitable features exist” after a bounded local search; a bug, missing access, or a timeout. |
| **Practical infeasibility** | Measured calibration or actual resource consumption prevents the minimum meaningful study/deployment under the explicit caps. | An unsupported guess that Apple silicon is too slow or a report based on another machine's runtime. |
| **Inconclusive evidence** | Intervals cross decision thresholds, there is too little base error/headroom, too little independent coverage, insufficient search reach, missing model/data/Mac access, or incomplete valid runs. | Treating failure to reject degradation as proof of preservation, or hiding an inadequately sized pilot. |
| **Invalid experiment** | Wrong site/token mapping, train/test leakage, invalid labels, broken numerical/compiler path, hidden routing, missing required baseline correctness, or measurement defects invalidate the intended comparison. | Treating a favorable point estimate as a result before fixing the invalidating defect. |

Interpret mechanism diagnoses specifically. Low support even in the whole candidate pool implicates the readout position/feature representation under this search. Enough support but zero line products implicates the block assignment/guard shape. Adequate $G$ but no feasible α implicates baseline margins and interference. $E_4$ working where $G$ fails implicates sparse design response rather than the shared minimax theorem. SAE64 or residual logistic success with SAE13 failure points to selection or information restrictions. Tube-cut superiority shows the practical value of leaving the polynomial class. All are useful findings; none licenses a universal claim about all SAEs or LLM editing.

If prompting already achieves the desired behavior and leaves inadequate repair headroom, say so. You may still report completed geometry/cost diagnostics, but do not claim a nonzero quality improvement or an opportunity that was not demonstrated. If a learned baseline matches quality with simpler engineering, report that plainly; a conditional certificate can remain mathematically interesting without being the preferred deployment mechanism.

## 10. Minimal repository, commands, and deliverables

### 10.1 Implement a small script-based project

Use this layout; omit downstream result directories when their stages are not reached. Create a valid `pyproject.toml` with a standard build system, `src` package discovery, declared runtime dependencies and a `dev` optional-dependency group containing pytest. Include `src/guard_poc/__init__.py` and a `cli.py` `if __name__ == "__main__"` dispatch so every module command below works from an editable install:

```text
guard_poc/
  README.md
  pyproject.toml
  requirements.lock.txt
  config.json
  inputs/                         # unchanged optional source attachments
  data/
    clauses.json
    templates.json
    tickets.jsonl
    retention_manifest.jsonl
    splits.json
  src/guard_poc/
    __init__.py
    cli.py                        # stage orchestration and stop/resume gates
    environment.py                # access/backend/memory/calibration checks
    data.py                       # deterministic text, grouping and hashes
    model.py                      # official chat template, final-token hooks
    sae.py                        # full and selected-column JumpReLU encoder
    guards.py                     # blocks, direct/compiled/DP/cut evaluators
    selection.py                  # bounded tuple search and scalar sweep
    baselines.py                  # weighted logistic fits and threshold sweep
    evaluate.py                   # frozen metrics, finite-point certificates
    benchmark.py                  # synchronized actual deployment timing
    report.py                     # all tables/plots from saved results
  tests/
    test_math.py
    test_interfaces.py
    test_metrics.py
  artifacts/
    environment.json
    correctness.json
    assets.json
    protocol_lock.json
    selected_patch.npz
    baseline_states.npz
  results/
    ledger.jsonl
    search_trials.jsonl
    predictions.jsonl
    timings.jsonl
    summary.json
    figures/
  REPORT.md
```

Required configuration keys include model/SAE revisions and paths; layers `[13,17]`; SAE width; token-position rule; precision/backend and allowed fallback; answer strings/IDs; seeds; scale quantile; pool size 64; search starts/sweeps/proposals; α cap 8; ε₀ 0.1; selected sample sizes; max context 512 and buckets `[256,512]`; memory/disk/time/token ceilings; baseline C grid; and report thresholds. Resolved runtime values belong in the protocol lock. No unfilled placeholders may remain in a locked run.

Define interfaces with shape assertions:

```python
frozen_forward(batch) -> logits_final[B,V], residuals[layer][B,1152]
encode(h, feature_ids=None) -> activations[B,F_or_selected]
normalize(a, scales) -> z[B,K]
compile_guard(blocks, dtype) -> two_fixed_layer_state_dicts
guard(z, implementation) -> g[B]
write_logits(logits, g, alpha, yes_id, no_id) -> patched_logits
fit_group(fit_cache, config) -> ids[13], scales[13], search_record
calibrate(method_state, calibration_cache) -> frozen_operating_point
evaluate_locked(lock, confirmation_manifest) -> append_only_rows
make_report(raw_results, lock) -> summary_tables_and_REPORT_md
```

`write_logits` must preserve the input tensor or clone explicitly, so one method's write cannot contaminate another method's base. Copy-on-write/immutability tests are essential. Model and SAE weights must have gradients disabled and hashes unchanged before/after the experiment. Do not share mutable optimizer objects across baseline fits.

### 10.2 Commands to create and verify

The following commands specify the interface you must implement; they are **not commands already verified on the user's laptop**. Choose an existing supported Python executable and use its actual version in the report.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
python -m guard_poc.cli inspect --config config.json
python -m guard_poc.cli prepare-data --config config.json
python -m guard_poc.cli verify-math --config config.json
python -m pytest -q
python -m guard_poc.cli fetch-assets --config config.json
python -m guard_poc.cli finalize-data --config config.json
python -m guard_poc.cli smoke --config config.json
python -m guard_poc.cli calibrate-resources --config config.json
python -m guard_poc.cli pilot --config config.json
python -m guard_poc.cli develop --config config.json
python -m pip freeze > requirements.lock.txt
python -m guard_poc.cli lock --config config.json
python -m guard_poc.cli confirm --lock artifacts/protocol_lock.json
python -m guard_poc.cli benchmark --lock artifacts/protocol_lock.json
python -m guard_poc.cli report --results results --lock artifacts/protocol_lock.json
```

`inspect`, raw `prepare-data`, and mathematical verification must work without model credentials. `prepare-data` writes raw text/semantic manifests only. `finalize-data` runs after authorized tokenizer retrieval: it serializes both prompt candidates, selects the valid answer pair, checks length eligibility without model outcomes, finalizes retention passage sampling, and hashes the scored manifests. `pilot` later selects between the two length-valid prompts using the designated fit examples; the final prompt hash is then frozen before feature search. Model-dependent pytest tests must skip with an explicit reason until assets are present. `fetch-assets` must honor exact filenames and disk caps, use existing authorized Hugging Face authentication, and never print tokens. If terms require a user's acceptance, report that specific access blocker and continue unaffected work. Do not accept terms or fetch an unofficial reupload to bypass the gate. `confirm` refuses changed hashes or an absent lock. All model-running commands update the same resource ledger; `--resume` may reuse hash-verified caches but cannot reset limits or silently overwrite outcomes.

For an early stop before lock, `report --results results` must work without a lock and state the reached stage. A one-command `run --config config.json` may orchestrate gates, but do not add a service, scheduler, dashboard, distributed worker, experiment server, or container requirement.

### 10.3 Final artifacts, proportional to work actually done

If implementation occurred, provide runnable code, fixed coefficients and feature-map parameters, dependency lock, concise setup/run instructions, and meaningful tests. If empirical stages ran, provide all per-example outcome rows, failed-run records, immutable configuration, asset/data hashes, selection/tuning ledger, raw timing blocks, environment facts, and scripts that regenerate every essential table/plot. Do not redistribute gated model checkpoints; supply retrieval identifiers and hashes.

The report must stand alone and include: question/scope; exact mechanism; what was performed; source-based facts versus new derivations versus measured findings; gate decisions; main effect with uncertainty; every required comparator; sample/exclusion/failure counts; certificate coverage and gain amplification; actual costs; deviations; limitations; outcome classification; and one specific justified next step if any. Explicitly state base-model gradient steps = 0 and separately enumerate data-informed selection/fitting. A zero base-gradient count does not mean no learning or zero optimization.

Essential figures are limited to (1) positive repair versus retention disagreement for frozen methods, clearly identifying the locked points; (2) observed ε₃/guard response and useful-certificate coverage by cohort, including near misses; and (3) measured acquisition/deployment cost breakdown if timing was reached. Plot scales, units and independent sample counts must be explicit. Do not create figures for unperformed experiments. An analytic stop needs the verifiable derivation/counterexample, source locations, and a minimal reproducer, not fabricated benchmark tables.

Before delivery, rerun report generation from raw files, check that primary counts reconcile with the manifests, check that all stated methods/stages actually ran, inspect figures, and verify that README commands match implemented entry points. Preserve unfavorable and failed results. Remove neither them nor the stop reason to make the result look complete.

## 11. Research record, claim map, and review dispositions

### 11.1 Compact claim–evidence map

| Central claim | Status/source | Remaining uncertainty | Resolving check |
|---|---|---|---|
| Exact minimax sparse-tube bound and design necessity | Supplied proofs checked; notebook rerun and independent rational probes [N] | Historical priority; not a machine-precision theorem | Tiny independent verification; qualified novelty statement |
| Explicit 13-block, 78-unit construction | Constructive source code and checked matrices [N, §5, cells 13–14] | Target backend's SiLU/reduction behavior | Actual matrices versus independent direct/rational references |
| Natural SAE coordinates satisfy useful protection geometry | **Hypothesis**, not established in supplied work | Target response, near misses, scale and assignment sensitivity | Bounded search, pool envelope, untouched geometry/utility evaluation |
| Matching 1B IT SAE exists | Official repository/config and loader inspected [R3, R4] | Local access, pinned files and exact runtime loading | Metadata/hash/shape gate and independent encoder check |
| New rule can be installed without base gradients | Only a narrower prompted-rule repair is tested | No demonstrated absent capability or instruction-free retention | State revision; compare with strong prompted base |
| Four predicates imply the required guard response | **False in general**, by explicit copied-predicate obstruction | Contextual SAE conjunctions might work | Keep all missing-one cases; measure rather than assume |
| Small monomial support implies lower runtime | **Unsupported**; $E_4$ tree used fewer products in planning check | Actual kernel/transfer/batch costs | DP baseline and real synchronized M3 measurements |
| Feature-conditioned frozen-base editing is new | **Not a defensible claim** [R6–R10] | Specific minimax theorem's priority remains uncertain | Attribute predecessors; study its conditional practical utility |
| Preservation below 1% follows from a clean small pilot | **False** without adequate independent sample size | True rare-event rate and language-distribution shift | Held-out passage/cluster intervals; fixed generator scope |
| Local execution fits an M3 Max | Planning estimate only [R13, R14] | Actual memory, OS, operations, throughput and thermal behavior | Initial inspection and representative calibration |

### 11.2 Research coverage and limits

Research was conducted on **5 November 2026**, using both available search engines, actual primary-paper content, official repository/configuration pages, implementation source, and backend documentation. Searches included sparse-jet/vanishing-ideal formulations, minimax polynomials and block designs, inclusion matrices and symbolic powers, weight-space programming and exact neural compilation, SAE-gated editing, simple SAE/probe baselines, and Gemma/MPS deployment details. Backward references from the matroid paper identified older uniform-matroid degree results; applied searches also located newer evaluations challenging and refining claims about SAE steering [R11, R12]. These checks changed baseline selection and narrowed novelty claims.

This was targeted, not an exhaustive global priority search or a reproduction of published LLM results. The original official Scope 2 Colab encountered a sign-in wall; the report and actual loader source supplied the needed convention. Some unrelated full PDFs could not be opened; no uninspected detailed results from them are needed here. The official SAE model-card examples contained copy/paste inconsistencies, resolved using per-checkpoint configuration. No natural-language feature geometry, dataset label quality, Mac timing, installed package compatibility, or permission state has been assumed verified merely because a webpage exists.

### 11.3 Three review passes and material fixes

The planning work used three independent agents for mathematical audit, tooling/source verification, and adversarial experimental design. Their findings were reconciled into this brief. All three agents also read the complete first draft in distinct mathematical/novelty, experimental, and empty-project execution review passes. Material findings were repaired in the final brief; this is a record of those reviews, not evidence of unperformed Mac execution.

| Review | Material issue | Disposition in this protocol |
|---|---|---|
| Mathematical and novelty | Sparse activation count does not determine line products; nine active coordinates can give zero. | Ordered feature assignment, exhaustive Boolean control, $E_4$ comparison and pool-response envelope are explicit. |
| Mathematical and novelty | Monomial minimality was liable to be mistaken for circuit/runtime minimality; paired-SiLU compilation and symbolic ideals have prior art. | Restricted theorem class and correct attributions retained; efficient symmetric DP and actual timing required. |
| Mathematical and novelty | JumpReLU and clipping break a blanket smooth-composition argument; float32 target is not exactly 1. | Claim value preservation only without extra smoothness; finite-point arithmetic checks; no hidden target renormalization. |
| Experimental | Prompt withholding, all-zero guards, one-sided false-positive limits, and cheap hard gates could create misleading wins. | Shared explicit policy, headroom gate, nonzero utility condition, full near-miss accounting, learned gates and tube-cut control. |
| Experimental | Template reuse, counterfactual dependence, clean calibration sets, and many tuning trials could overstate evidence. | Cluster/passage units, fixed-generator scope, held-out banks, locked confirmation and exact interval thresholds. |
| Execution | Wrong SAE format/normalization or whole-repository download could invalidate or overwhelm the project. | Exact safetensors files/config, no extra encoder normalization, narrow downloads and hashes. |
| Final-draft mathematical review | Ideal symmetric logit shifts, continuous scalar optima, and full-cache pool envelopes could be overclaimed after rounding/path changes. | Actual-shift KL/TV; finite scalar candidate-set semantics; exact finite-point checks; cached-envelope scope and proper math formatting. |
| Final-draft experimental review | Calibration omitted total specificity/useful coverage; demonstration examples overlapped prompt scoring; failures could break the repair-rate model. | Aligned calibration constraints, disjoint prompt-selection clusters, fixed iid rhetorical mixture, explicit denominators/infeasibility ranks, complete-outcome success requirement. |
| Final-draft execution review | Tokenizer access, unspecified fit-weight scale, small-pool initialization, late dependency freezing, and mandatory timing counts could block reproducibility. | Raw/finalized data stages, mean-one weighted scaling, evenly spaced initializer, pre-lock online/environment checks, and all-cap timing reservation. |
| Execution | All-token logits/features and unmeasured MPS assumptions could hide fatal costs. | Final-position-only capture/logits, selected-column deployment, actual hardware inspection and cumulative caps. |

Stress-test consequences are predeclared: a nonconstructive concern is resolved by explicit finite blocks/matrices; realistic assumption failure produces a geometry diagnostic; known generic mechanisms narrow the contribution; success only on synthetic text limits external validity; a stronger simple baseline winning limits deployment value; preprocessing dominance is reported in total cost; precision failure triggers bounded repair or invalidation; memory/time overflow gives measured infeasibility; a noisy pilot expands only within its already budgeted development set; inaccessible assets block affected claims; a false hypothesis produces an honest negative or inconclusive report. There is no indefinite rescue branch.

## References

All web resources were consulted or specifically checked for the stated purposes on 5 November 2026. Mutable repositories/docs must be pinned again at execution. References are evidence for the described facts, not a claim that their full experimental results were independently reproduced.

**[N]** *Optimal guards for weight-space programming: A proved connection between selective polynomial edits and combinatorial block designs.* Supplied research notebook, 5 November 2026; author not identified in the attachment. File: `Optimal_Guards_for_Weight_Space_Programming(1).ipynb`. Critical locations: §2 Theorems A/B; §§3–4 proofs; §5 construction; code cells 7–18; §7 output insertion; §§8–9 limitations/attribution. Input paths are recorded in §2.1.

**[R1]** Rikiya Takehi, Jayden Teoh, Alex Zhang, Omar Khattab. *Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights.* 2 November 2026. [Primary article](https://rikiyat.github.io/blog/weight-space-programming/). Read Experiment 8 and “Raising math capability of a model that does not loop”: the reported Qwen change addresses looping, while the additional Gemma experiment did not establish a capability gain. Motivation only; do not transfer its performance numbers to this study.

**[R2]** Google / Hugging Face Transformers contributors. *Gemma 3 model and implementation documentation*, living sources accessed 2026-11-05. [Model repository](https://huggingface.co/google/gemma-3-1b-it); [Gemma 3 forward API](https://huggingface.co/docs/transformers/en/model_doc/gemma3); [decoder implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma3/modeling_gemma3.py); [Google reference configurations](https://github.com/google/gemma_pytorch/blob/main/gemma/config.py), `get_config_for_1b`. Relevant: model dimensions, native gated activation, final-position `logits_to_keep`, block-output location. Inspect current installed code rather than relying on a mutable API snapshot.

**[R3]** Callum Stuart McDougall et al., Google DeepMind. *Gemma Scope 2 – Technical Paper.* PDF displays 16 September 2025; accessed 5 November 2026. [Technical paper](https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/gemma-scope-2-helping-the-ai-safety-community-deepen-understanding-of-complex-language-model-behavior/Gemma_Scope_2_Technical_Paper.pdf); [official 1B IT SAE repository](https://huggingface.co/google/gemma-scope-2-1b-it); [layer-13 configuration](https://huggingface.co/google/gemma-scope-2-1b-it/blob/main/resid_post/layer_13_width_16k_l0_medium/config.json); [layer-17 configuration](https://huggingface.co/google/gemma-scope-2-1b-it/blob/main/resid_post/layer_17_width_16k_l0_medium/config.json). Critical paper locations: §2.1 equations (1)–(2), §2.3 equation (3), §2.5 IT adaptation, §3.1 folded activation scaling. Checkpoint metadata was read; no model inference was performed during planning.

**[R4]** SAELens contributors. *Pretrained SAE loaders*, living official implementation, accessed 2026-11-05. [Source](https://github.com/decoderesearch/SAELens/blob/main/sae_lens/loading/pretrained_sae_loaders.py), functions `get_gemma_3_config_from_hf` and `gemma_3_sae_huggingface_loader`. Relevant: lowercase tensor keys, encoder orientation, `apply_b_dec_to_input=False`, and no extra activation normalization. Use as an independent specification; a large SAELens dependency is not required for this five-tensor encoder.

**[R5]** Rob Porter. *Introducing torchwright.* 23 July 2026. [Author article](https://ood.dev/posts/torchwright-intro/), nonlinearity and bounded-approximations discussions. Explicit prior source for paired complementary-Swish multiplication and program-to-weight compilation; does not supply the notebook's sparse-tube theorem.

**[R6]** Feisi Fu, Wenchao Li. *Sound and Complete Neural Network Repair with Minimality and Locality Guarantees.* ICLR 2022; arXiv submitted 2021, v3 July 2022. [Paper](https://arxiv.org/html/2110.07682v3), repair construction and locality results; [DOI](https://doi.org/10.48550/arXiv.2110.07682). Establishes prior localized ReLU patch-network repair; its architectural assumptions differ from this LLM branch.

**[R7]** Eric Mitchell, Charles Lin, Antoine Bosselut, Christopher D. Manning, Chelsea Finn. *Memory-Based Model Editing at Scale.* ICML 2022, PMLR 162:15817–15831. [Paper](https://proceedings.mlr.press/v162/mitchell22a/mitchell22a.pdf), §3.1 equation (1), §3.2 and scope-error discussion. SERAC separates scope prediction from counterfactual prediction while retaining a frozen base. The lightweight controls here are not a reproduction of SERAC.

**[R8]** Thomas Hartvigsen, Swami Sankaranarayanan, Hamid Palangi, Yoon Kim, Marzyeh Ghassemi. *Aging with GRACE: Lifelong Model Editing with Discrete Key-Value Adaptors.* NeurIPS 2023. [Paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/95b6e2ff961580e03c0a662a63a71812-Paper-Conference.pdf), §2.2 equation (1) and Algorithm 1. Prior activation-space local editing with frozen base weights and learned replacement values; no lifelong-edit claim is tested here.

**[R9]** Aashiq Muhamed, Jacopo Bonato, Mona Diab, Virginia Smith. *SAEs Can Improve Unlearning: Dynamic Sparse Autoencoder Guardrails for Precision Unlearning in LLMs.* arXiv:2504.08192v1, 11 April 2025. [Full text](https://arxiv.org/html/2504.08192v1), §§3.2–3.3 and Algorithm 1. Prior feature selection, input-dependent scope classification, and conditional SAE intervention. Cite its mechanism as prior art; this brief does not import its causal or statistical-optimality claims or reproduce its unlearning benchmarks.

**[R10]** Liam Dugan, Callum Stuart McDougall, Matthew Denton, Philip Pham, Christine Kaeser-Chen, Chris Callison-Burch. *Building Auditable AI Text Detectors with Sparse Autoencoders.* EMNLP 2026, pp. 17640–17689. [Proceedings page](https://aclanthology.org/2026.emnlp-main.762/); [DOI](https://doi.org/10.18653/2026.emnlp-main.762); [official DetectorScope implementation](https://github.com/google-deepmind/detectorscope), models/reproduction documentation. Lightweight learned classifiers on SAE features are established. The landing page and implementation were read; the entire paper was not independently audited.

**[R11]** Zhengxuan Wu, Aryaman Arora, Atticus Geiger, Zheng Wang, Jing Huang, Dan Jurafsky, Christopher D. Manning, Christopher Potts. *AxBench: Steering LLMs? Even Simple Baselines Outperform Sparse Autoencoders.* ICML 2025, PMLR 267:67035–67080. [Proceedings](https://proceedings.mlr.press/v267/wu25a.html); [official implementation](https://github.com/stanfordnlp/axbench). Read for baseline selection and limits of SAE-steering claims. Its reported Gemma-2 comparisons are not results about this guard. Do not import its remote generation/judging or distributed training harness.

**[R12]** Mikkel Godsk Jørgensen, Lars Kai Hansen. *Steering LLMs? Actually, Sparse Autoencoders can outperform simple baselines.* arXiv:2605.31183v1, 29 May 2026. [Full text](https://arxiv.org/html/2605.31183v1), §§3.1, 3.3 and 4. Reports that feature-selection/labeling choices materially affect SAE steering. This motivates counting selection cost and avoiding blanket conclusions, not reproducing its larger hardware pipeline.

**[R13]** PyTorch contributors. *MPS backend*, *MPS API*, and *MPS environment variables*. Living official documentation, accessed 2026-11-05. [Backend](https://docs.pytorch.org/docs/2.15/notes/mps.html); [API](https://docs.pytorch.org/docs/2.15/mps.html); [environment variables](https://docs.pytorch.org/docs/2.9/mps_environment_variables.html). Relevant: actual backend availability, synchronization, allocator/driver memory, and explicit fallback. Versioned URLs document inspected sources; the installed supported version must be recorded.

**[R14]** Hugging Face contributors. *Apple Silicon*, Transformers v5.22.1 documentation. [Official guide](https://huggingface.co/docs/transformers/en/apple_silicon), accessed 2026-11-05. Relevant: MPS execution, supported precision and shape/cache considerations. Edward J. Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, 2021 / ICLR 2022, [paper](https://arxiv.org/abs/2106.09685), is the excluded fine-tuning reference. No LoRA superiority conclusion is authorized by this study.

**[R15]** Christopher Clark, Kenton Lee, Ming-Wei Chang, Tom Kwiatkowski, Michael Collins, Kristina Toutanova. *BoolQ: Exploring the Surprising Difficulty of Natural Yes/No Questions.* NAACL 2019. [Paper](https://arxiv.org/abs/1905.10044); [official dataset](https://github.com/google-research-datasets/boolean-questions); [Hugging Face mirror/card](https://huggingface.co/datasets/google/boolq). Relevant: natural question/passage/Boolean-answer records, 9,427 training and 3,270 released development examples, CC-BY-SA-3.0. The archive status does not itself make its released files unavailable.

**[R16]** Paolo Mantero, Vinh Nguyen. *The Structure of Symbolic Powers of Matroids.* arXiv:2406.13759v1, 19 June 2024. [Full text](https://arxiv.org/html/2406.13759v1), Theorem 1.2 / 3.7, Theorem 6.2, and §7 preceding Proposition 7.10. Relevant prior algebra/initial-degree machinery; it cites older uniform-matroid formulas, including Paolo Mantero, *The structure and free resolutions of the symbolic powers of star configurations of hypersurfaces*, Transactions AMS 373 (2020), 8785–8835, [DOI](https://doi.org/10.1090/tran/8208), Theorem 7.7. The older theorem was identified through citations/metadata, not independently reread in full.

**[R17]** Codruţ Grosu, Yury Person, Tibor Szabó. *On the rank of higher inclusion matrices.* Journal of the London Mathematical Society 90 (2014), 333–349. [Author PDF](https://page.mi.fu-berlin.de/szabo/PDF/inclusionmatrices.pdf); [DOI](https://doi.org/10.1112/jlms/jdu029). Inclusion-matrix definitions and classical rank context. This is the paper at the notebook's linked URL; the URL is not Gottlieb's 1966 paper itself.

**[R18]** Peter J. Cameron, Leonard H. Soicher. *Block intersection polynomials.* Bulletin of the London Mathematical Society 39 (2007), 559–564. [Author manuscript](https://webspace.maths.qmul.ac.uk/l.h.soicher/bip.pdf), dated 8 September 2006, §2 equations (1)–(2); [DOI](https://doi.org/10.1112/blms/bdm034). Classical intersection-counting/design identities, rather than the notebook's particular optimization statement.

**[R19]** Nico Pelleriti, Max Zimmer, Elias Samuel Wirth, Sebastian Pokutta. *Approximating Latent Manifolds in Neural Networks via Vanishing Ideals.* ICML 2025, PMLR 267:48734–48761. [Proceedings page](https://proceedings.mlr.press/v267/pelleriti25a.html). The inspected abstract describes polynomial vanishing-ideal representations of latent manifolds. Relevant background against claiming algebraic latent features in general are new; no detailed result beyond the inspected source is used in this protocol.
