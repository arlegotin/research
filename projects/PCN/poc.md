# Local learning for language models: research assessment and executable proof-of-concept plan

**Research date:** 11 October 2026.  
**Target:** one MacBook Pro with an M3 Max; actual memory, storage, macOS, and software support must be discovered.  
**Deliverable status:** researched experimental plan, with limited CPU mathematical checks completed. No language-model training, M3 benchmark, or positive research result is claimed.  
**Working project name:** `local-contract-lm`. The proposed “Predictive Contract Network” name is descriptive; it is not an established method or a novelty claim. Avoid the abbreviation PCN in code because it also commonly denotes predictive coding networks.

**Contents**

1. [Executive assessment and decision](#1-executive-assessment-and-decision)
2. [Problem, scope, assumptions, and hypotheses](#2-problem-scope-assumptions-and-hypotheses)
3. [Research findings and novelty assessment](#3-research-findings-and-novelty-assessment)
4. [Evaluation contract, fixed before implementation choices are finalized](#4-evaluation-contract-fixed-before-implementation-choices-are-finalized)
5. [Exact minimum mechanism](#5-exact-minimum-mechanism)
6. [Data, baselines, attribution, and harness sensitivity](#6-data-baselines-attribution-and-harness-sensitivity)
7. [Tuning, measured budgets, and statistical decision procedure](#7-tuning-measured-budgets-and-statistical-decision-procedure)
8. [M3 Max implementation and resource plan](#8-m3-max-implementation-and-resource-plan)
9. [Executable project specification and work packages](#9-executable-project-specification-and-work-packages)
10. [Adversarial interpretation, failure regimes, and continuation decisions](#10-adversarial-interpretation-failure-regimes-and-continuation-decisions)
11. [Independent plan audits and revision record](#11-independent-plan-audits-and-revision-record)
12. [Mathematical checks actually executed during planning](#12-mathematical-checks-actually-executed-during-planning)
13. [Source ledger, research scope, and access limits](#13-source-ledger-research-scope-and-access-limits)

## 1. Executive assessment and decision

### 1.1 Verdict

**Pursue a controlled investigation of locally trained predictive refinements, with a protected input path and sparse forward evaluation of early-block auxiliary-task updates. Treat the contribution initially as a new empirical investigation and application of established techniques.** There is a credible question to answer, but neither foundational architectural novelty nor an advantage over backpropagation is established.

The original proposal correctly identifies an architectural and coordination problem: an early module can improve its own predictions without preparing features that later computation needs. Nevertheless, residual boosting, information-preserving local objectives, context supply, learned feedback, and adaptive auxiliary-task selection already cover most of its ingredients. Particularly close antecedents are DiffusionBlocks for independently trained predictive refinement, ContSup/InfoPro for preserving useful inputs, and HydaLearn for evaluating provisional task-specific updates before setting loss weights. [R02], [R04], [R05], [R06], [R07], [R10], [R11], [R12], [R13]

The selected research question is:

> **At a fixed complete training-compute budget, can an additive prediction interface with protected input access make sparse, forward-only downstream feedback useful enough to improve a small autoregressive language model beyond strong fixed local objectives and established lookahead controls?**

The strongest local result this project could establish is a reproducible quality/compute/memory tradeoff for the measured model, corpus, context length, and training regime. It cannot establish frontier-LLM feasibility, biological synaptic locality, distributed speedup, or a general solution to credit assignment.

### 1.2 Selected mechanism

Use a small, modern causal Transformer partitioned into gradient-isolated groups. Each group produces a working representation and a correction to an accumulated predictive state. Later groups can also read the original input embeddings. Train with ordinary autograd **inside** each group, detaching every cross-group state.

At the first group boundary, provide a deliberately small menu of contracts: next-token prediction alone, or next-token prediction plus an auxiliary prediction of token `t+2` or `t+4`. Occasionally make exact, temporary optimizer updates for these choices. With the remaining network frozen, score each resulting model on a separate training-audit minibatch using forward passes only. Commit the chosen early-group update and reuse that objective between audits.

This is a **block-constrained lookahead task-selection experiment**, explicitly related to HydaLearn and AutoSeM. It is not an invention of counterfactual optimization or autonomous semantic “feature demand.” The possible new empirical finding is that the predictive interface improves the usefulness, persistence, or economics of this feedback. That requires an interaction test against the same controller on a simpler interface. [R12], [R13]

Start with direct audit-based selection. Add a learned utility predictor only if direct audits demonstrate useful decisions and their cost becomes the limiting factor. This ordering prevents a failed critic from obscuring whether there was useful feedback to predict.

### 1.3 What changes from the original idea

| Original element | Decision and reason |
|---|---|
| Prediction corrections | Retain, with explicit residual-equivalence controls. Their novelty is not presumed. |
| Protected information | Retain access to input embeddings, while distinguishing access from useful feature computation. |
| Adaptive contracts | Operationalize as a finite menu of causal future-token objectives. This is a narrow, implementable test of the intended coordination principle. |
| General differentiable utility critic | Defer. Test direct, on-manifold optimizer proposals first; subsequently test a small discrete action-value predictor. |
| Simultaneously changing modules during audits | Replace with an atomic audit transaction that freezes every non-audited owner. Normal training remains synchronous local training. |
| Shared vocabulary projection | Retain but assign a single owner. Its coupling and possible distributed broadcast costs are disclosed. |
| Frontier-scale victory and a mandatory 400M gate | Remove from the laptop deliverable. Escalation depends on measured sensitivity and resources, not a predetermined model size. |
| Biological learning, recurrent societies, self-organizing specialists | Keep as separate research questions; they are not required to test this mechanism. |

### 1.4 What has actually been done

Primary papers, proceedings, author repositories, dataset documentation, and framework documentation were inspected. Initial access failures for the October greedy-LM paper were resolved through its versioned arXiv HTML. Several other exact versions or repository licenses remain unavailable or require verification before code reuse; these are recorded below.

Small mathematical checks were executed on **Linux x86_64, Python 3.12.14, NumPy 2.3.5**. PyTorch was unavailable in that environment. The checks establish limited algebraic examples and explicit chain-rule properties, not compatibility with a training framework. Section 12 contains the results and a self-contained reproduction snippet.

No data corpus was downloaded or tokenized for this plan; no training throughput, laptop memory peak, accuracy, or NLL improvement was measured. All resource numbers below are estimates or calculation examples until WP01/WP06 replace them with measurements.

## 2. Problem, scope, assumptions, and hypotheses

### 2.1 Intended beneficiaries and task

The immediate beneficiary is a researcher trying to train useful language-model representations without an end-to-end backward graph. A longer-term beneficiary might be a developer training models across memory-constrained workers. The latter benefit is conditional: deleting backward activation traffic does not by itself reduce total traffic or improve utilization.

The task is **autoregressive pretraining from random initialization**, with a fixed tokenizer and a fixed training corpus. The conventional workflow trains a causal Transformer jointly using terminal next-token cross-entropy. The candidate changes the state interface and training signals; it does not change the prediction target, obtain a pretrained teacher, or use a paid API.

Local learning can reduce retained activation depth. Whether this is a material bottleneck on the particular laptop must be measured. On a small model, vocabulary projections, optimizer states, memory allocation, or kernel efficiency may dominate instead. Published multi-worker memory savings do not predict whole-laptop savings. [R01], [R03], [H06]

### 2.2 Definitions

For document tokens `x_0,…,x_(n−1)`, predict `x_(t+1)` from `x_≤t`. Let `K` be the number of groups, `L` the number of Transformer layers, `d` the hidden width, `V` the vocabulary size, `B` the microbatch sequence count, and `T` the input length. `sg` denotes a value-preserving stop-gradient operation.

The primary risk is

\[
\mathcal L(\theta)=\mathbb E_{(x,t)\sim\mathcal D}
[-\log p_\theta(x_{t+1}\mid x_{\leq t})],
\]

measured in **nats per valid target token**. Documents, not independently sampled token positions, provide the natural held-out clusters. A training replicate is a complete run with independent initialization and data-order randomness.

**Block-local:** a loss may differentiate through the current group and its own readout, but not through another group. **Audited local:** block-local derivatives plus occasional terminal scalar scores from frozen-suffix forward passes. **Hybrid:** occasional true cross-group derivatives. **Global:** terminal loss backpropagates through the complete network. Report these categories separately.

“No global backpropagation” does not mean no global information, no autograd, no synchronization, or strictly synaptic learning. This candidate uses labels, a shared readout, and downstream scalar feedback.

### 2.3 Evidence versus assumptions

| Category | Statement |
|---|---|
| Established capability | Gradient-isolated training, auxiliary prediction, residual feature boosting, and learned local feedback have working implementations in relevant research families. [R01], [R02], [R03], [R04], [R05], [R06], [R07], [R08], [R09], [R10], [R11], [R12], [R13] |
| Directly relevant evidence | Locally trained autoregressive Transformers can learn, but strong likelihood results still show a gap and auxiliary-compute costs. [R01], [R03] |
| Architectural assumption | A separate predictive state may make useful downstream corrections easier to learn than repeated independent prediction. This is not implied by addition alone. |
| Information assumption | Some future-token auxiliary gradients improve features consumed by later groups. High auxiliary accuracy alone is insufficient. |
| Feedback assumption | The ranking of one-step, fixed-suffix updates persists long enough to guide subsequent training. It may fail under rapid suffix drift. |
| Feasibility assumption | A meaningful small model can express at least some local/global gap and useful contract choice within measured local resources. |
| Speculation | Any advantage persists at long contexts, large vocabularies, large models, or frontier token budgets. |

### 2.4 Hypotheses, alternatives, and discriminating observations

| ID | Hypothesis and prediction | Alternative explanation | Observation that weakens or rejects the stated claim |
|---|---|---|---|
| H1: interface | Corrections plus input access improve terminal NLL or useful later-group contributions relative to appropriately equipped greedy models. | Extra projections, added context, head sharing, initialization, or optimization explain the change. | No benefit after the factorial and matched-capacity controls; algebraic identity in the restricted residual control invalidates a distinct correction principle. |
| H2: feedback | Sparse downstream-scored contract choices improve final NLL beyond the best fixed contract at the same full compute budget. | Additional data exposure, larger updates, or generic lookahead explains the gain. | Benefit disappears against static, random, shuffled-reward, norm-matched-CE, or established lookahead controls. |
| H3: interaction | Accumulation changes the incremental benefit of the same contract controller relative to an exact reset interface. Easier utility prediction or longer ranking lifetime are separately tested explanations. | The controller helps both interfaces equally, or the interaction reflects different available action gains without improved predictability. | The controller gain is equal or smaller with accumulation under a precise paired interaction estimate. A positive interaction alone does not prove easier prediction or longer persistence. |
| H4: economics | The best candidate offers a useful point on the NLL/time/memory/inference-cost frontier compared with tuned end-to-end training. | Saved graph depth is offset by heads, audits, and parameter states. | It is dominated by tuned end-to-end or checkpointed training over the examined range. |
| H5: amortization, conditional | A small critic can replace some direct audits without losing more quality than its saved compute buys. | It merely memorizes training phase or repeats a good static policy. | No advantage over a constant, phase-only, or linear predictor and direct use of the same audit observations. |

A failure to detect H1–H3 at a scale where neither the baseline gap nor auxiliary utility is measurable is **inconclusive about larger scales**. Conversely, a clear gap, a sensitive harness, adequate training, and a precise null/negative result are evidence against that mechanism in the tested regime. Do not invoke hypothetical scaling indefinitely to rescue a negative result.

### 2.5 Two important theoretical limits

For a frozen predecessor and a fixed shared readout, a correction class containing zero has a training-risk infimum no worse than the predecessor: zero correction is a feasible point. This says nothing about finite optimization, held-out risk, moving readouts, changing earlier groups, or the final loss after later transformations.

For binary label 1, let an early block predict logit `a` and a frozen suffix add `−2a`. Its local loss is `softplus(−a)` and terminal loss `softplus(a)`. At `a=0`, local and terminal gradients are `−1/2` and `+1/2`. Local descent therefore worsens the final prediction. This counterexample survives the presence of an additive predictive state.

With a linear readout, `W(b+c)=Wb+Wc`. If `b=h` and `c` is the residual hidden update, the proposal is a residual network with local deep supervision. Splitting the same computation into two names does not create a new hypothesis. A protected stream also does not solve credit assignment: it makes information available, while learning to compute with it remains necessary.

## 3. Research findings and novelty assessment

### 3.1 What the recent evidence actually supports

The October greedy-LM study is the most direct likelihood-training anchor. It examines 125M/400M models and two/four groups, with one synchronous seed per configuration and matched tokens. Its stronger first-boundary MTP configuration adds approximately 97%/48% analytical training FLOPs at four groups. Collapsed local losses are evidence consistent with myopia, not a demonstration that a particular feature was erased. [R01]

DiffusionBlocks is a close architectural competitor. Its AR experiments evaluate generated text with MAUVE and external-model generative perplexity; they do not establish ordinary held-out NLL parity. Its released repository describes a vision implementation, making a faithful language port a separate undertaking. [R02]

The apparent contrast between successful vision local learning and weaker language results is not a contradiction. Tasks, readouts, supervision, compute matching, and evaluation populations differ. Likewise, a perturbation method approaching backpropagation with a large population does not establish equal-compute efficiency. Dust explicitly leaves substantial efficiency work unresolved. [R01], [R02], [R08], [R17], [R18], [R27]

Improving a layer-health diagnostic is not automatically useful. A recent cumulative-goodness study finds that repairing deeper-block separation need not improve final accuracy. Its FF vision setting differs from this proposal, but it is a direct warning against making “healthy depth” the success metric. [R22]

A targeted public-patent check also found synthetic-gradient training and validation-guided auxiliary scheduling disclosures. These reinforce the technical ancestry; they do not establish legal scope or patentability. A deployed adjacent example, DeepSeek-V3, uses multi-token training but retains backward computation in its documented training pipeline. Neither patent documents nor deployment of auxiliary prediction establish strict-local pretraining at frontier scale. [PAT01], [PAT02], [DEP01]

### 3.2 Prior-art comparison matrix

“Runnable” below means there is a credible local implementation path, **not** that the published workload has been reproduced on M3.

| Family/source | Problem and mechanism | Assumptions/evidence and computational cost | Implementation/reuse | Exact overlap and comparison role |
|---|---|---|---|
| Greedy LM [R01] | Detached groups; stronger auxiliary predictors and selectively placed future-token loss. | Autoregressive pretraining; auxiliary work is material. | Implement inspected equations in the common harness; released-code URL/license not located. | Essential contemporary local baseline; scaled adaptation, not reproduction of its published numbers. |
| DGL/local parallelism [R03], [R14] | Local heads remove backward update dependencies; chunking/overlap varies gradient reach. | Vision and small LM evidence; modeled parallelism is distinct from serial laptop time. | Reuse partition/readout structure; no need for a distributed framework. | Locality itself is established; include simple local and overlapping-group diagnostics. |
| BoostResNet/ResFGB [R04], [R05] | Greedily grow residual feature predictors under a readout. | Boosting assumptions; earlier stages commonly fixed. | Mathematical reduction is sufficient; legacy framework port is unnecessary. | Anticipates additive predictive refinement; requires exact-equivalence and stagewise controls. |
| InfoPro [R06] | Preserve task-useful information through reconstruction plus local supervision. | Vision results; surrogate information objectives. | Public PyTorch reference; check exact license before copying. | Anticipates information-preserving contracts; compare feature reconstruction if token contracts fail. |
| ContSup [R07] | Supply context across isolated modules. | Vision benchmarks; information access is the intervention. | Author code available; inspect license at chosen commit. | Protected input is not new. Give the greedy comparator the same input access. |
| DiffusionBlocks [R02] | Independent noise-conditioned denoisers, original-input conditioning, iterative prediction state. | Changes training and sampling; ordinary likelihood comparison is unresolved. | Apache-2.0 vision code; language port and doubled clean/noisy sequence are substantial costs. | Closest broad architecture reference, excluded from the ordinary-NLL leaderboard with explanation. |
| Synthetic gradients [R09] | Predict downstream derivatives to decouple modules. | Feedback model must learn useful errors; supervision may use real derivatives. | Paper/reusable small predictor pattern. | Learned credit is established; stronger-information diagnostic, not strict-local evidence. |
| Perturbation-trained feedback [R10] | Train approximate feedback from activity perturbations and global rewards. | Small feedforward/CNN evidence; estimator variance matters. | Author code link; license not verified. | Direct antecedent to amortized counterfactual local utility. |
| Local Critic Training [R11] | Local critics approximate downstream losses and provide local gradients. | Convergence depends on alignment/descent assumptions. | Public reference; license gate. | Low value-regression error is insufficient; require useful interventions. |
| HydaLearn [R12] | Provisional task updates are scored on the main metric before weighting tasks. | Two-task supervised examples; more forward work and optimizer sensitivity. | Equations inspected; use a labeled LM adaptation with explicit edge cases. | Closest controller ancestor. No novelty for shadow-update selection alone. |
| AutoSeM/Auto-Lambda/AuxiLearn [R13], [R19], [R20] | Adapt task selection or weighting from main-task performance. | Bandit or meta-gradient mechanisms; validation use and optimization cost differ. | Author code exists for the latter two; use only after license/operation checks. | Adaptive objectives are established. Compare cheap selection and stronger-information weighting when relevant. |
| AugLocal/feature reconstruction [R15], [R21] | Give local losses more downstream structure or reconstruct incoming features. | Vision Transformers/CNNs; replicated auxiliary computation. | Public reference implementations in varying states. | “Successor needs” has close antecedents; an extra auxiliary attention layer is a strong simple alternative. |
| PGL/GrAPE [R16], [R17] | Periodic global guidance or aligned learned feedback. | GrAPE includes BP calibration and transformer-specific derivative paths. | Simple periodic-BP control is implementable; full GrAPE port is separate. | Sparse feedback is known; compare its cost with occasional true gradients. |
| Dust/forward gradients [R08], [R18] | Activity perturbations or directional derivatives estimate learning signals. | Population and derivative costs differ; published implementations target other hardware. | Dust MIT minimal code is Linux/NVIDIA oriented. | Use as estimator/reference evidence; a partial port must be named “inspired,” not faithful. |
| Deep predictive coding [R23] | Local iterative latent relaxation and balanced errors. | Vision/deep-network evidence with extra inference dynamics. | Public code; license unresolved in inspection. | Shares motivation, not the proposed one-pass mechanism; not a mandatory laptop LM baseline. |
| Titans/MIRAS/Hope/LoPT [R24], [R25], [R26] | Online memory or localized post-training of pretrained models. | Different stage of learning and often outer/global optimization. | Public assets vary; no pretrained asset required here. | Credible alternatives only if the goal changes to adaptation or memory. They do not validate local pretraining. |

### 3.3 Explicit novelty answers

1. **What exists?** All major component ideas, including forward-scored provisional auxiliary updates. The broad proposal is substantially anticipated.
2. **What can be reused?** Standard causal blocks, exact detached training, established optimizers, tokenization/Parquet readers, auxiliary MTP structure, lookahead weighting logic, bootstrap/statistical tools, and source-linked numerical fixtures. Avoid rebuilding a tokenizer, attention kernel, or distributed engine.
3. **What is potentially different?** The empirical interaction among a separate accumulated predictive state, repeated original-input access, and sparse feedback restricted to actual early-group optimizer proposals in from-scratch autoregressive training. The present search does not establish that this exact experiment has never been performed.
4. **Why might it help?** The local prediction gives a useful default direction; preserved inputs reduce irreversible deprivation; a finite proposal menu reduces the credit signal's dimensionality. These are deductions, not measured advantages.
5. **What isolates it?** Apply the identical controller and audit budget to both the predictive and simpler interfaces; compare fixed objectives, constant/phase-only policies, direct lookahead weighting, and norm-matched CE proposals; measure terminal NLL and realized action utility.
6. **What invalidates novelty?** A prior method with the same computational graph and update/information flow; exact parameter-mapping equivalence; a result explained by generic context supply or task weighting; or no interface-dependent effect. In those cases report a replication, port, optimization, or negative empirical result accurately.

No new theorem, mathematical framework, “first” claim, or patentability claim is proposed. A carefully demonstrated improvement over strong baselines could still be scientifically valuable without an unprecedented primitive.

### 3.4 Direction selection and contingent alternatives

| Direction | Judgment and next condition |
|---|---|
| Audited contracts on the predictive/protected scaffold | Primary. Small enough to specify exactly, attributable with controls, and able to test the intended coordination principle. |
| Corrections/protection with a fixed early MTP objective | Mandatory simpler alternative; becomes the recommended result if adaptation adds no value. |
| Strong auxiliary attention or one-group overlap | Mandatory escape from a weak local baseline; prefer it if generic lookahead capacity wins. |
| Unrestricted perturbation-trained critic | Defer until direct audit decisions help and amortization is the bottleneck. High-dimensional action coverage and critic exploitation are serious risks. |
| Independent denoising blocks | Scientifically credible different formulation. Pursue only with a separate generation/likelihood protocol and a verified language implementation. |
| Continual-memory architecture | Pivot only if actual demand shifts to adaptation or a local pretraining result specifically motivates fast-memory updates. |
| Recurrent/specialist graph | Not justified for this PoC: routing and temporal credit introduce additional unisolated questions. |

## 4. Evaluation contract, fixed before implementation choices are finalized

The following decisions define the scientific comparison. WP06 supplies measured budgets, selected configurations, and sample sizes; it may not quietly change these definitions.

### 4.1 Three comparison panels

| Panel | What is matched | What it can establish |
|---|---|---|
| Mechanism | Same data order, effective batch, deployed shape, precision, common parameter initialization, head bank, audit events where applicable; report remaining active-parameter differences. | Attribution to accumulation, protected input, or decision policy. It does not establish efficiency. |
| Training-compute | Same total counted arithmetic budget including auxiliaries, audits, rejected proposals, recomputation, controller fitting, and run-specific preparation. Same eligible data pool and tuned implementations. | Primary conditional NLL comparison at a fixed project-defined compute budget. |
| Laptop economics | Same laptop conditions; measure complete elapsed time and whole-process/accelerator memory. Allow each method its appropriate microbatch/checkpointing configuration. | Real end-to-end tradeoffs on this M3, including training, model storage, and inference. |

Also report both deployed parameter count and all trainable/frozen training-only parameters. Parameter equality, token equality, FLOP equality, and elapsed-time equality are different comparisons. Do not force one artificial allocation to answer all four questions.

### 4.2 Primary estimand and success language

Select the strongest eligible **no-audit local** configuration on development data, call it `B_local`. Select a strongest established lookahead adaptation separately, `B_lookahead`. Freeze both before confirmatory evaluation. The candidate is the single selected audited configuration `A`.

For independent paired training replicate `s`, compute

\[
D_s=\operatorname{NLL}(B_{\rm local},s;C_*)-
\operatorname{NLL}(A,s;C_*).
\]

Positive `D` favors the candidate. The primary estimand is the mean paired difference over training randomness on the frozen held-out population at total budget `C_*`. Report the NLLs, difference in nats/token, confidence interval, and perplexity ratio `exp(−D)`; do not report only a percentage. `C_*` is chosen using the calibration procedure, not a guessed token budget.

There are two prespecified superiority comparisons for an “adaptive local improvement” claim: `A` versus `B_local`, and `A` versus `B_lookahead`. Use simultaneous familywise inference as specified in Section 7. An architecture-attribution claim additionally requires the corresponding fixed-scaffold and interaction controls. A claim against end-to-end learning requires a separate comparison to the selected end-to-end frontier; winning only against greedy local training does not meet it.

There is **no arbitrary minimum percentage gain**. A statistically resolved small gain is reported as small. Practical value is evaluated by equivalent baseline compute, elapsed time, memory, and inference costs. An NLL improvement with a material regression is a tradeoff, not an overall victory.

### 4.3 Guardrails and diagnostic outcomes

Measure final NLL, training FLOPs, training wall time, peak memory, checkpoint size, inference prefill time, steady-state decode time, and numerical failures. Evaluate `T=512` initially and length/distance stress tests only where training can express them.

Measure stage NLLs with the same deployed shared head, incremental `NLL_(k−1)−NLL_k`, and the actual final effect of bypassing a correction. For greedy models with different private heads, do not interpret unequal head quality as unequal information content: add identical-capacity diagnostic probes trained on frozen features with a common budget. Probe results are secondary and conditional on that decoder class.

Report synthetic retrieval/algorithmic exact-match by dependency distance and difficulty; real-text NLL by article and prespecified token-frequency/position/context-length bins. Synthetic success alone does not establish real-text value or reasoning competence.

## 5. Exact minimum mechanism

Everything in this section is a **proposed implementation**, except the explicitly identified algebraic checks in Section 12. The working name `local_contract_lm` describes this experiment; it is not a claim to a newly established architecture.

### 5.1 State, causal blocks, and parameter ownership

The initial shape is `L=8, K=4, d=384, heads=6, head_dim=64, d_ff=1024, V=8192, T=512`. Each group contains two pre-RMSNorm causal Transformer layers with RoPE and SwiGLU. Use untied input/output embeddings, no linear biases, RMS epsilon `1e-5`, RoPE base `10000`, and zero dropout initially. These are explicit starting choices to be calibrated, not claims of optimal hyperparameters. Keep four groups in the smaller `d=256, heads=4, d_ff=704` fallback so that the depth-coordination question survives.

Inputs contain only tokens through the position being predicted. Position enters through causal attention's RoPE. Let `r=E(x)` be the original embedding sequence, `h_0=r`, and `b_0=0`, all shaped `[B,T,d]`. At group `k`, define

\[
\begin{aligned}
\bar h_{k-1}&=\operatorname{sg}(h_{k-1}),\quad
\bar b_{k-1}=\operatorname{sg}(b_{k-1}),\quad
\bar r=\operatorname{sg}(r), && k>1,\\
z_k&=\bar h_{k-1}+I_{\rm raw}P_k\bar r+I_{\rm condition}Q_k\bar b_{k-1},\\
h_k&=G_k(z_k),\\
c_k&=C_k\operatorname{RMSNorm}_k(h_k),\\
b_k&=\alpha\bar b_{k-1}+c_k,\\
\ell_k^{(1)}&=\operatorname{CE}(Wb_k,x_{t+1}).
\end{aligned}
\]

For `k=1`, use `z_1=r`, with no `P_1` or `Q_1`; the input embedding has the restricted gradient ownership below. For the candidate, `I_raw=I_condition=1` and `alpha=1`. The `alpha=0` control resets the predictive state at each boundary. The `I_raw=0` and `I_condition=0` controls separate original-input access, conditioning on the prediction, and accumulation. Every CE is the sum over eligible target positions divided by their global effective-batch valid-token count, not a mean of differently padded microbatch means.

`P_k`, `Q_k`, and `C_k` are `d × d` linear maps owned by group `k`. No normalization follows the accumulated `b_k` before `W`; this preserves the exact additive-logit identity used in the interpretation. The norm before `C_k` belongs to that group. Initialize standard dense weights with Xavier uniform, `E` and `W` with normal standard deviation `0.02`, RMS scales at one, layer residual-output matrices with an additional `1/sqrt(2L)` factor, and correction matrices with an additional `1/sqrt(K)` factor. Store the explicit initialization implementation and random streams. Common tensors receive identical initialization across paired arms. Alternative initialization is a development-only sensitivity, shared where the tensor roles match.

The hidden representation is not an untrained side branch: `c_k` is computed **from** `h_k`, so its local prediction loss trains the representation passed onward. A parallel hidden branch unused by any local loss would receive zero gradient. A zero-initialized correction can also initially block trunk gradients; use nonzero common initialization for synchronous training. A separate frozen-stage diagnostic may use zero correction to test the limited feasible-solution property.

| Parameter owner | Permitted training signal | Forbidden paths / special rule |
|---|---|---|
| Input embedding `E` | First-group next-token CE only on normal steps | No later-group gradients, auxiliary-contract gradients, or audit mutation |
| Group `theta_k` = Transformer layers, its norms and active `P/Q/C` | Its own next-token CE; group 1 may also receive the selected auxiliary loss | No derivative through another group; audited actor excludes `E`, readout, and auxiliary bank |
| Shared vocabulary readout `W` | Terminal-group next-token CE only on normal steps | Early groups use `W.detach()` but still differentiate the linear operation with respect to `b_k` |
| Auxiliary head bank `phi` | Both future-token losses on `sg(h_1)` on normal steps | Never update the actor through the bank-training loss; freeze bank parameters when computing actor auxiliary gradients |
| Optional discrete utility predictor | Past training-audit records only | No confirmation data, no gradients across the suffix, no derivative through an unconstrained value extrapolation |

Give every tensor exactly one optimizer owner. Assert disjoint ownership and full coverage. An intentionally frozen or inactive parameter must be reported as such, rather than counted as useful capacity. Earlier local losses see the current shared readout but cannot update it. This is a common numerical convention and a source of nonstationarity, not completely independent learning. A frozen-readout diagnostic tests whether moving `W` explains an effect.

### 5.2 Local loss and auxiliary bank

Provide two train-only early-group predictors, for offsets `j ∈ {2,4}`:

\[
a_j(h)=h+\operatorname{FFN}_{\phi_j}(\operatorname{RMSNorm}_{\phi_j}(h)),\qquad
\ell_1^{(j)}=\operatorname{CE}\left(W_{\rm aux}\operatorname{RMSNorm}_{\rm aux}(a_j(h_1)),x_{t+j}\right).
\]

Each FFN is a one-layer residual SwiGLU branch of the same `d_ff`; the branches share a separate auxiliary vocabulary projection and output norm. This bank is removed at inference. Targets may extend beyond the input window when they remain in the same training document; target tokens are never supplied as future inputs. Mask positions without an eligible target and normalize each horizon separately. `j=2` predicts one token beyond the usual next-token target; `j=4` predicts three beyond it.

The action menu is

\[
J_0=\ell_1^{(1)},\qquad J_2=\ell_1^{(1)}+\lambda\ell_1^{(2)},\qquad
J_4=\ell_1^{(1)}+\lambda\ell_1^{(4)}.
\]

The action called `none` or `0` means **a real CE-only optimizer step**, never “do nothing.” All other groups use next-token CE alone in the minimum candidate. Initial development candidates for `lambda` are `{0.1, 0.3, 1.0}`; expand the bracket only when the development trend at an edge warrants it, under the same search-accounting rule as competing methods.

On each ordinary step, train **both** bank branches on detached early features regardless of the chosen action, with `L_bank = sum_CE_2/N_2 + sum_CE_4/N_4`, where each horizon is separately normalized over the complete effective batch. This prevents a rarely selected objective from losing because its readout never learned. Actor gradients use the bank's pre-step parameters as constants while retaining derivatives with respect to its input. Implement this by a functional call with detached bank parameters, not by wrapping the entire bank call in `no_grad`.

For an entirely invalid auxiliary horizon (`N_j=0`), omit its branch loss and leave exclusive branch parameters with `grad=None`; skip their optimizer/moment/decay update. Shared bank parameters may still update from the other valid branch. A valid but mathematically zero gradient remains a zero tensor and follows the declared optimizer rule; it is not interchangeable with `None`. An effective batch with no valid next-token targets performs no model update. Preserve actual per-parameter optimizer counters as well as aggregate owner-update counts. Inactive ablation parameters are explicitly frozen/excluded from active ownership.

The diagnostic policy comparison uses the identical bank and training exposure for static, random, phase-only, and adaptive policies. A mandatory static competitor also uses the fixed mixture `CE1 + lambda * [omega*CE2 + (1−omega)*CE4]`, with `omega` initially bracketed at `{0,0.5,1}` and refined on development if needed. This tests whether combining two useful tasks explains the benefit of alternating them. In the economic comparison, the fixed CE-only policy may remove the bank, and a fixed single-horizon policy may remove the unused branch. Charge these honest implementations their actual costs, including both actor auxiliary paths for a mixture. Conversely, the contemporary multi-token local baseline may use its own tuned auxiliary architecture; it need not inherit a deliberately expensive bank.

Set the original proposal's utility-loss and stability coefficients `mu=rho=0` initially. There is no unspecified extra regularizer. Add a stability penalty only after a measured drift failure and a registered ablation specifying its reference state, normalization, cost, and failure criterion.

### 5.3 Ordinary synchronous local step

Use AdamW as the initial optimizer with independently owned parameter groups. Initial development defaults are `beta=(0.9,0.95)`, `eps=1e-8`, weight decay `0.1` on dense non-embedding weights, no decay on norms, and an owner-wise gradient-norm clip of `1.0`. Embedding/readout decay is a declared tuning choice common across applicable methods. The conventional end-to-end baseline uses its customary whole-model clipping; also test whether clipping conventions explain a comparison. Accumulate gradients over microbatches with exact token weighting before clipping and stepping.

Within a microbatch, evaluate groups in order, differentiate only the current owner's loss, detach the outgoing values, and release that group's graph. Compute `E`'s gradient from CE1 separately, and `W`'s gradient from terminal CE together with the final group's gradient. The first group's auxiliary gradient is requested only with respect to its internal owner. Sum bank-training gradients from a separate call on detached features. Apply **all** owner updates only after the entire effective batch and all group forwards have used one parameter version. This synchronous barrier avoids accidental stale or mixed-version semantics while still permitting sequential graph release.

Represent the ordinary actor objective uniformly as `main_weight*CE1 + aux_weight[2]*CE2 + aux_weight[4]*CE4`. Discrete action 2 means weights `(1,lambda,0)`; the mixture and lookahead controls use their specified coefficients. Omit a zero-coefficient term from the graph. Parameters exclusive to omitted terms have `grad=None` and skip that optimizer transaction; participating parameters with a numerically zero derivative retain zero gradients. The embedding's separate CE1 update is unweighted by the actor policy. This distinction matters for decoupled weight decay and existing Adam moments.

If the actor has no active valid loss term, represent its objective as absent and skip actor autograd/update, rather than differentiating the Python scalar zero. The embedding and other owners may still receive their independently permitted losses. Bank and owner helpers must handle this case explicitly.

```text
ordinary_step(batch, policy, state):                    # proposed project pseudocode
    zero_all_owner_grad_buffers()
    version = state.model_version
    for microbatch in exactly_weighted_microbatches(batch):
        r = embed(microbatch.inputs)
        h, b = r, zeros_like(r)
        for k in 1..K:
            h, b = group_forward(k, h, b, r, detach_boundaries=(k > 1))
            ce = token_sum_ce(readout(b, detach_weights=(k < K)), target_1)
            objective = ce / batch.valid_count_1
            if k == 1:
                objective = policy.main_weight * objective if policy.main_weight != 0 else absent
                for horizon, weight in policy.aux_weights.items():
                    if weight != 0 and batch.valid_count[horizon] > 0:
                        objective = add_valid_term(objective, weight * frozen_bank_ce(h, horizon) / batch.valid_count[horizon])
            if k == 1:
                if policy.is_exact_ce_only():
                    accumulate_gradients(ce / batch.valid_count_1, owners=[theta[1], E])
                else:
                    if objective is not absent:
                        accumulate_gradients(objective, owner=theta[1], retain_graph=True)
                    accumulate_gradients(ce / batch.valid_count_1, owner=E)
                accumulate_gradients(bank_loss(stop_gradient(h)), owner=phi)
            elif k == K:
                accumulate_gradients(ce / batch.valid_count_1, owners=[theta[K], W])
            else:
                accumulate_gradients(objective, owner=theta[k])
            h, b = stop_gradient(h), stop_gradient(b)
            release_finished_graphs()
    assert state.model_version == version
    clip_each_owner_then_apply_its_optimizer_once()
    state.model_version += 1
```

The pseudocode expresses exact ownership and the intended joint gradient requests. It applies to the primary `K>=2` regime; a `K=1` E2E diagnostic requests all permitted owners jointly. The bank call is omitted in a pruned-bank configuration. Unit tests must verify the realized graph and memory behavior. Do not retain every group's graph in a list and then claim activation-depth savings.

### 5.4 Exact downstream audit transaction

An audit uses an update minibatch `S` from the ordinary-update pool and a separate minibatch `Q` from the training-only audit-document pool. Neither is development or confirmation data. `S` and `Q` cannot share an article/duplicate cluster. Audit `Q` is additional training information and compute; its exposure is logged and made available on the same terms to relevant controls.

For each action, start from the **same** early-group parameters, AdamW first/second moments, owner step count, learning rate, clipping rule, and random-number state. Form the same local effective-batch gradient that a real update would use. Apply its exact optimizer transaction to a temporary copy of `theta_1`. Freeze `E`, `W`, the bank, all other groups, and their optimizer states during the entire transaction.

Let `U_j(theta_1, state_1; S)` be this parameter-and-optimizer-state update. Score

\[
q_j=\operatorname{NLL}_Q\left(f_{\theta_{>1},E,W}
\left(U_j(\theta_1,\mathrm{state}_1;S)\right)\right),\qquad
R_j=q_0-q_j.
\]

`q_0` is the **updated CE-only** model, so `R` measures value beyond an ordinary local update. Also record the unupdated model's score for diagnostic compatibility with lookahead weighting. Run the suffix in forward-only evaluation mode; do not create a derivative graph across it.

Use the following exact numerical tie rule with tolerance `tau` established by repeated identical/no-op audit probes. Evaluate action 0 first and retain its evaluated state. Visit actions 2 then 4; replace the retained winner only if the new score is lower than the retained score by **more than `tau`**. Otherwise retain the earlier action. Apply this same rule during serial state retention and final selection, so a tie never chooses a discarded state. The choice is an approximate argmin up to the calibrated numerical resolution; it is not a test that the action's population benefit is positive. Sampling uncertainty is resolved by independent diagnostics and the held-out experiment.

Commit the **already evaluated** winner's actor parameters and optimizer state. An audit step replaces an ordinary all-owner step with this actor-only update: non-actor owners remain unchanged. Holding them fixed is essential because otherwise the scored action differs from the committed model. Between audits, use the last selected action for ordinary synchronous local steps. Choose audit period and `Q` size on development cost/utility curves, then freeze them. Starting periods `{16,64,256}` ordinary update opportunities are a search bracket, not a mandatory maximum or minimum frequency; extend when the measured ranking lifetime or overhead puts the optimum outside it.

Number effective-batch update opportunities from one. For period `p`, audit at `p,2p,3p,…`, use the initial CE-only policy before the first audit, and hold the selected policy afterward. An interrupted transaction resumes/replays the same opportunity; it does not advance the schedule twice. Store this index separately from per-owner/per-parameter optimizer counters.

```text
audit_step(S, Q, state):                                # proposed project pseudocode
    require disjoint_document_clusters(S, Q)
    base = snapshot_actor_parameters_moments_counters_rng()
    frozen_signature = hash_non_actor_parameters_and_states()
    cached_Q = cache_immutable_input_and_prefix(Q)        # includes r; E is frozen
    q_before = forward_terminal_nll(base.parameters, cached_Q)
    best_action, best_score, best_state = 0, +infinity, null
    records = []
    for action in [0, 2, 4]:
        restore_actor_and_rng_exactly(base)
        grad = actor_local_gradient(S, action, frozen_bank=True, exclude_E=True)
        proposal = exact_adamw_step(base, grad, current_lr, clipping_rule)
        score = forward_terminal_nll(proposal.parameters, cached_Q)
        records.append(action, score, update_norm, grad_norm, costs)
        if action == 0 or score < best_score - numerical_tolerance:
            best_action, best_score = action, score
            best_state = copy_complete_actor_state(proposal)
    chosen = best_action
    restore_complete_actor_state(best_state)
    assert hash_non_actor_parameters_and_states() == frozen_signature
    advance_data_cursor_once_for_S_and_once_for_Q()
    write_audit_record_and_increment_model_version()
    return chosen
```

The first-group audit has no nontrivial trainable prefix; caching mainly reuses immutable embedded audit inputs. The interface supports a later interior-group experiment only if all prefix-produced inputs are truly frozen and both predictive and hidden states are cached. Do not make interior selection part of the minimum search space.

Use one immutable actor snapshot, one temporary proposal, and at most one retained winner, processed serially. Never instantiate three full models. A winner may be saved/restored from host memory, but unified host memory still counts. An implementation can instead deterministically recreate the winning actor transaction; if so, count that extra gradient/update work and verify bitwise or tolerance-qualified equivalence before committing.

Adam state is part of the counterfactual. A trial SGD update followed by a committed AdamW update, resetting moments for trials, clipping the trial differently, or allowing `W/E` to move invalidates this test. When an owner is frozen on an audit step, its optimizer step counter must not advance. Schedule progress follows the comparison's declared global budget axis; bias correction follows each owner's actual update count.

### 5.5 Established lookahead control

Implement a clearly named **HydaLearn-derived block-local LM adaptation**, alongside the source's small analytical fixture. HydaLearn scores task-specific provisional updates against a main-task metric before weighting the losses. This already anticipates the feedback principle. Its derivation does not guarantee arbitrary optimizer geometry or every possible sign pattern. [R12]

For one development-selected auxiliary horizon, evaluate pure CE and pure auxiliary actor proposals from the same state using the unupdated `Q` model as the common reference. Let their improvements in negative NLL be `g_main` and `g_aux`. The bounded adaptation uses

\[
u_i=\max(0,g_i)^\beta,\quad
w_i=u_i/(u_{\rm main}+u_{\rm aux}),\qquad
J=w_{\rm main}\ell^{(1)}+w_{\rm aux}\lambda\ell^{(j)}.
\]

Use `beta=1` initially; if the sum is numerically zero, use main-only. The pure auxiliary proposal is generated from `lambda*CE_j`, consistently with the combined loss. Compute the actual combined-gradient AdamW proposal from the original state; do not interpolate the two updated parameter vectors or assume the scored gains add. Store `{horizon, w_main, w_aux, lambda, selected_at_version}` as the policy state, and use this same weighted actor objective on every ordinary step until the next audit. `E` retains its separately specified CE1-only ownership, the bank trains both branches, and later groups retain ordinary CE.

Admit **both** cost-faithful and safeguarded variants to development selection. The source-style variant commits the combined proposal without paying for an extra combined `Q` score that does not affect its decision; optional diagnostic scoring is separated and charged. The safeguarded variant scores the combined proposal and commits it only when it beats the already evaluated CE-only proposal by more than the common numerical tolerance; otherwise it commits CE-only and stores main-only weights until the next audit. Both use the same frozen-owner transaction. Select the stronger complete-cost implementation as `B_lookahead`. The source-compatible positive-gain SGD fixture should agree with the inspected weighting relation. The sign handling, AdamW use, block restriction, dataset, safeguards and finite budget are explicit adaptations; do not call them reproductions of published HydaLearn results.

This control can be more or less expensive than the discrete selector. Charge every proposal and score. Add a cheaper local-score controller that replaces terminal `Q` NLL with first-group `Q` CE: if it performs equally well, the particular downstream-information claim weakens. A tuned constant weight and training-phase-only schedule test whether adaptivity is necessary at all.

Also admit a **three-task extension** so the strongest lookahead comparison has access to the same horizons as the candidate. Score pure `CE1`, `lambda*CE2` and `lambda*CE4` actor proposals against the unupdated model; normalize the positive gains (with `beta>0`) over all three tasks. The combined actor objective is `w_main*CE1 + lambda*(w_2*CE2+w_4*CE4)`. When all gains are numerically zero, use main-only. Test both source-style unscored-combination and CE-safeguarded variants as above, retain their full three-weight policy between audits, and charge the extra actor path and scores. This is explicitly a project extension of a two-task antecedent, not a published HydaLearn result. `B_lookahead` is selected from the valid two- and three-task adaptations under the common tuning budget.

For any claim that discrete selection itself is superior to continuous weighting, run a matched-menu diagnostic: restrict the candidate to `{0,j}` and the lookahead control to that same development-selected horizon `j`, with common actor/bank/exposure and complete costs. A full-menu win over a single-horizon controller alone cannot support that attribution.

### 5.6 Conditional utility predictor and broader credit assignment

Activate this branch only if direct audits choose actions with useful **independently scored** terminal effects, outperform appropriate static/mixed-task controls, and their overhead is a demonstrated bottleneck. Winning reward on the same `Q` used to choose the action is optimistically selected and cannot satisfy this gate. On a predeclared development subset of audit events, evaluate the chosen and CE-only proposals on a fresh training-only `Q_check` from disjoint article clusters, unused in that event's selection or gradient. Record the signed NLL difference, including negative values. Keep `Q_check` results out of actor/controller fitting and action decisions; their diagnostic work belongs in the development ledger. This does not turn training-audit data into protected confirmation.

First fit a regularized linear/ridge predictor of each action's `q_0−q_j` using detached local CE, auxiliary CE, entropy, available past gradient/update norms, learning rate, optimizer age, and training progress. Use all scored actions, not only selected winners. No article IDs, raw future tokens, evaluation features, future audit records, or post-decision measurements may enter it. Features must be available before that action is chosen; any extra forward/backward work to obtain them is charged. Scale features using past records only.

Evaluate chronologically: predict each next audit before observing its outcome; report chosen-action regret against the directly scored menu, ranking accuracy with ties separated, calibration, and downstream realized utility. Compare a constant-action policy, phase-only policy, linear predictor, and direct selection using the same history. A learned nonlinear critic is admitted only if it beats these simpler alternatives under complete cost accounting. Reduce audit frequency only through a development comparison that checks whether saved compute buys more quality than prediction errors lose. The primary candidate then requires a new protected confirmation if its policy changes.

Do not differentiate an actor through an unrestricted scalar critic merely because value MSE is small. For a random spherical direction `u` in `d` dimensions and exact linear response `uᵀg`, the unbiased estimator `d(uᵀg)u` has mean-squared error `(d−1)||g||²`; averaging `m` independent directions divides that variance by `m`. This calculation is a warning about dimension, not a lower bound for every structured estimator. If revisiting continuous perturbations, use antithetic pairs, identical suffix/RNG states, radius sensitivity, a CPU gradient oracle on tiny cases, on-policy coverage tests, and direct improvement tests. Include all rejected probes. The sparse finite action menu avoids claiming to recover a full downstream gradient.

The original ambition of downstream modules requesting semantic distinctions remains open. Future-token objectives are a deliberately narrow proxy. Success here supports economical task-selection feedback in this regime, not arbitrary communication of future feature needs.

### 5.7 Required invariants

1. Before any learning update, changing future input tokens leaves every group's states/logits at positions `<=t` unchanged. Changing target labels alone leaves forward features/logits unchanged. Audit decisions may depend on authorized `S/Q` target labels and terminal scores, but never on protected splits or future audit records; test this information boundary separately from causal attention.
2. In strict-local arms, a local loss has zero gradients on all earlier/later groups and forbidden owners; both `h` and `b` boundaries and repeated `r` access are detached.
3. Every active hidden trunk receives a nonzero gradient in a nondegenerate numerical fixture; branch disconnection is detected.
4. All proposals start from identical parameters, moments, counters, LR, clipping convention, and RNG; non-actor signatures remain unchanged.
5. For the **discrete** menu, `lambda=0` makes all three CE-plus-auxiliary proposals/scores agree to calibrated tolerance; forcing action `0` reproduces actor-only CE. Lookahead pure-auxiliary probes need not equal CE at zero lambda: test their declared omitted-loss behavior and main-only fallback/guard separately.
6. Cached-suffix and complete-model forwards agree before and after each proposal. Target labels cannot enter a cache used as features.
7. The discrete selector and safeguarded lookahead commit exactly their scored winner, including optimizer state. The source-style unscored lookahead commits exactly its constructed weighted-gradient proposal from the common snapshot and records the combined score as unmeasured. All variants roll back rejected probes completely; never fabricate a score to satisfy a schema.
8. Changing microbatch partitioning preserves effective gradients within calibrated floating-point tolerance, including horizon-specific masks.
9. Saving and resuming preserves model, optimizer ownership, controller history, RNG, sampler cursors, and the next planned transaction.
10. Costs include auxiliaries, snapshots/copies, audits, rejected work, preparation, and any diagnostic derivatives. Unsupported or unmeasured counters are `null`, never fabricated zeros.

## 6. Data, baselines, attribution, and harness sensitivity

### 6.1 Real-text corpus and immutable assets

Use **WikiText-103 raw**, fetched from the Salesforce-maintained Hugging Face dataset repository at immutable revision `b08601e04326c79dfdd32d625aee71d232d685c3`. Read the Parquet files directly with PyArrow; no remote dataset script is required. This provides real encyclopedia text at a manageable download size and a stable article structure. It is a narrow language domain, so a result is not automatically a general web-language result. [D01], [D02], [D03]

The following are **source-declared LFS hashes and sizes inspected during planning**, not hashes of files downloaded in this session. WP02 must compute local hashes and stop on a mismatch.

| Path within repository | Bytes | Declared SHA-256 |
|---|---:|---|
| `wikitext-103-raw-v1/train-00000-of-00002.parquet` | 156987808 | `74da360f23826045b3e6ac6375411fdb15f003030aa74f2596ed08b857cb9212` |
| `wikitext-103-raw-v1/train-00001-of-00002.parquet` | 157088770 | `ba090ac30dbf5461e8dcbdd1a1b8e6f3cf9c2c756d64f0c1220450acd514f720` |
| `wikitext-103-raw-v1/validation-00000-of-00001.parquet` | 657209 | `204929b7ff9d6184953f867dedb860e40aa69c078fc1e54b3baaa8fb28511c4c` |
| `wikitext-103-raw-v1/test-00000-of-00001.parquet` | 732610 | `5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91` |

The dataset card's license metadata and body disagree: the former lists CC BY-SA 3.0/GFDL while the latter states CC BY-SA 4.0. Preserve the exact notices and attribution in `data/NOTICE.md`; do not simplify them into “public domain” or unrestricted rights. The intended activity is local research. Resolve conditions material to the intended use before dependent use or redistribution; do not bundle source text or assert a trained model's redistribution license from a package license alone. If necessary use an explicitly permitted replacement corpus, record a population amendment, and redo the split/tokenizer freeze. The code/aggregate-result deliverable need not redistribute the corpus. [D01]

Existing download API example, to run only after WP01 prerequisites:

```python
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id="Salesforce/wikitext",
    repo_type="dataset",
    filename="wikitext-103-raw-v1/train-00000-of-00002.parquet",
    revision="b08601e04326c79dfdd32d625aee71d232d685c3",
    local_dir="data/source/wikitext",
)
```

The function is a documented library API; the surrounding project downloader and checksum validation are to be implemented. Record the actual installed package and API probe. [D04]

### 6.2 Article parsing, contamination control, and splits

Stream the two training shards in filename/row order before reconstructing articles; a shard boundary is not necessarily an article boundary. Identify top-level title lines of the form `= title =`, excluding multi-equals section headings. Keep the title and original text, including dataset-specific punctuation artifacts. Do not detokenize `@-@`, lowercase, or silently normalize whitespace. Assert conservation of input text and row coverage, and manually inspect a small **training-only** parsing fixture. Compare resulting article counts with the author's reported split counts as a diagnostic, not permission to discard unexplained rows. [D03]

Build exact-content and near-duplicate clusters before project splitting. Exact hashes use the raw reconstructed article bytes. For near-duplicate discovery, use whitespace-collapsed, case-folded word 5-shingles, 128-permutation MinHash with recorded seed, and candidate pairs from 32 bands of four hashes; verify candidates with actual shingle Jaccard at `>=0.90`. Use the established `datasketch` package on CPU: after a tiny installed-API probe, set `MinHash(num_perm=128, seed=..., gpu_mode="disable", scheme="affine32")` and `MinHashLSH(num_perm=128, params=(32,4))`. Save/hash the actual permutations, scheme, hashing convention and package version. Stream `update_batch` in bounded chunks; do not allocate all article shingles times all permutations at once. Setting only an LSH threshold is not equivalent to fixed banding. [D09]

For articles with fewer than five normalized words, use exact normalized-text matching only and record that near-duplicate coverage limit. These are transparent deduplication engineering settings, not guarantees of exhaustive semantic decontamination. Record uncovered risks and retained/removed counts. Apply identical decisions to every method. Use connected components so near-duplicate members cannot be assigned independently. Automated structural parsing/deduplication may process protected text to build the frozen manifest; do not inspect protected model outputs or use those documents to choose preprocessing rules.

Assign official-training clusters using a fixed salted SHA-256 hash of the sorted member raw hashes: first 90% of the integer hash range to a training pool, next 5% to `D_dev_internal`, last 5% to protected `D_confirm`. Within the training pool, use a second independent salt to allocate 95% to ordinary-update documents and 5% to the controller's `Q` audit documents. Add the official validation split to development. Reserve the official test split as `D_reserve`. Priority for conflicting cross-split clusters is `D_reserve > D_confirm > development > training-audit > ordinary-update`; remove lower-priority copies and record the exclusion. Preserve all salts, cluster memberships, article IDs, and final counts in a manifest before training.

Both update and audit pools are **training data**. Audit documents may influence the policy repeatedly and must never be described as test or validation evidence. The strongest non-audit methods may train on the union of eligible training pools under the compute-matched economic panel; retain update-pool-only controls to isolate information exposure. No configuration choice, early stopping, critic feature fitting, normalization, or tuning may use `D_confirm` or `D_reserve` outputs.

Do not substitute WikiText-2 as an independent confirmation corpus: the inspected releases share validation/test assets with WikiText-103. A future second real-text domain requires its own immutable shard manifest and contamination checks. FineWeb-Edu is a credible extension, but its named `sample-10BT` release is not a small laptop sample; choose and record bounded actual shards only after the first result warrants a domain extension. TinyStories would test synthetic narrative data, not replace the real-text claim. OpenWebText's packaging and source-text rights must also be distinguished. [D05], [D06], [D07]

### 6.3 Tokenizer, training examples, and exact scoring

Train one byte-level BPE tokenizer of vocabulary size 8192 on the entire ordinary-update pool only. Use the `tokenizers` package, no Unicode normalization or lowercasing, no implicit prefix space, the complete byte alphabet, minimum merge frequency two, and explicit `PAD`, `BOS`, `EOS`, and `UNK` tokens. The count 8192 includes specials. Verify the actual stable wheel's trainer API with a tiny round-trip fixture; do not assume a prerelease documentation signature. Freeze `tokenizer.json`, special IDs, package version, training article order, token counts, and SHA-256. Decoding must preserve representative Unicode/whitespace examples, and all bytes must be representable without unexpected UNK. [D08]

Store token IDs as little-endian `uint16` and article offsets as `uint64`; convert batch indices/targets to the backend's supported integer type. Arrays are memory-mapped, with only the current CPU batch transferred to MPS. Do not duplicate the whole corpus in device memory. `BOS` begins an article and `EOS` ends it. Track injected specials/padding by structural masks, not token-ID equality alone; literal text resembling a special-token spelling must round-trip and must not silently disappear from scoring. No training attention or label crosses an article boundary. Use separate padded article windows initially; packing can be added only with verified block-diagonal masks and identical semantics for all arms.

For training, form each article as `BOS, content tokens, EOS`. Enumerate input-window starts `0,T,2T,…` through the last position that has a next-token target; exclude EOS as an input with no successor. Take at most `T` actual input tokens, right-pad to `T`, and use the next tokens as targets. Thus content and EOS are each next-token targets exactly once per enumerated epoch; no target crosses a document boundary. Reset RoPE positions to `0..T−1` within each window. Future-offset targets use the same source article and the same valid input positions, clipped at EOS; future labels never enter features. This is deliberately truncated training context, with richer overlapping context used only by the fixed evaluation convention below.

Assign a stable window ID from article ID, start and preprocessing hash. Shuffle all eligible window IDs once per epoch using a named RNG seeded by `(data_seed,pool_id,epoch)` and the locked RNG implementation; consume every window once before reshuffling. Separate update/audit pools have independent streams. Sampling windows this way gives every valid next-token target one exposure per complete epoch rather than equalizing article weights. Preassemble each effective batch from **whole windows** until the configured `effective_valid_tokens` minimum is reached. Keep the final whole window, normalize by the **actual** valid count, and record overshoot, which is less than `T`; no partial-window remainder is hidden. Microbatching partitions this already fixed window list and cannot change it. Carry an incomplete batch across an epoch boundary using the next epoch's permutation, and checkpoint the exact next epoch/cursor and any preassembled-but-uncommitted batch IDs.

The same paired data seed/rule yields the same update stream; compute-matched methods may consume different prefixes because their costs differ. The audit batch follows the same whole-window rule using its own valid-token minimum. If an auxiliary horizon has no valid targets in an effective batch, omit that objective and follow the `None`/owner rules in Section 5.2; do not divide by zero or use cross-document targets. Assert `max_token_id < 65536` before uint16 storage, and verify one-epoch target coverage and the next batch after restart.

Primary evaluation is a fixed overlapping-window protocol with `T=512` and scoring stride `S=T/2`. Let a document be `x_0=BOS, x_1,…,x_n, x_(n+1)=EOS`. Partition target indices `1..n+1` into consecutive blocks `[a,b]` of at most `S` targets. For each block, set `start=max(0,b−T)`; input tokens are `x_start..x_(b−1)`. Score only predictions for targets `a..b`; all earlier context-target losses and padding are masked. Thus every content token and EOS is scored exactly once, with at most `T` preceding input positions and at least approximately half a window of context away from the document start. This is a fixed windowed likelihood protocol, not a claim that every token receives exactly `T` context tokens.

Aggregate `sum_nll_nats / valid_target_count` over all held-out articles. Include EOS in primary NLL; exclude BOS and PAD targets. Report `exp(NLL)` as tokenizer-specific perplexity. For a secondary bits-per-byte measure, exclude artificial special-token losses and divide the remaining negative log-likelihood by `ln(2)` times the raw UTF-8 content-byte count; report the exact special-token and reconstruction conventions. Do not compare perplexities from different tokenizers as equal units.

Test one-token/empty-content documents, articles spanning source shards, Unicode, all-padding batches, horizon clipping, and incomplete final windows. The evaluator must reproduce a hand-calculated tiny distribution and return identical article sums regardless of evaluation batch partitioning. Store per-article loss sums and token counts, not only a scalar leaderboard.

### 6.4 Baseline register and implementation scope

All train-from-scratch arms use the same core causal-layer code, tokenizer, eligible training pools, evaluation masks, initialization pairing, and numerical conventions unless a recorded difference is part of the method. Default configurations live in versioned YAML; commit hashes and dependency locks determine implementations. There is no fabricated upstream release tag. Reimplement a paper's inspected equations when released code is absent or unsuitable, and call the result a **recipe adaptation at local scale**.

| ID | Implementation and tuning | Role / reproduction standard |
|---|---|---|
| `U0` | Smoothed training-token unigram predictor | Harness floor; not a competitive baseline. Check hand-computed probabilities and held-out aggregation. |
| `E2E` | Conventional eight-layer pre-RMSNorm/RoPE/SwiGLU model, final norm and untied head, terminal CE through all layers | Strong global reference. Tune AdamW and, if the MPS probe succeeds, Muon for eligible 2-D hidden matrices with AdamW for remaining parameters. Test checkpointing and batch choices on the memory/time frontier. |
| `E2E_scaffold` | Exact candidate inference architecture and initialization, terminal CE with no boundaries detached, no train-only contracts | Separates architecture capacity/conditioning from the local training rule. It is not interchangeable with the conventional E2E reference. |
| `G_linear` | `K=4` detached groups, own RMSNorm/linear vocabulary heads at early boundaries, true final head | Simple local diagnostic only; beating it is insufficient. |
| `G_mlp` | Each early auxiliary head uses one residual pre-norm SwiGLU FFN before its norm and private vocabulary head | Stronger local decoder; derived from the contemporary greedy recipe. |
| `G_attn` | Each early head adds one causal Transformer layer **before** the residual pre-norm SwiGLU FFN, output norm, and private vocabulary head | Strong local comparator; charge both auxiliary attention and residual FFN at every early boundary. |
| `G_mtp` | `G_attn` with a first-boundary two-token predictor: separate residual FFN branches for offsets 1 and 2, sharing that boundary's vocabulary head/output norm; later boundaries remain next-token only | Principal contemporary local recipe adaptation. Tune its MTP weight and head width/depth fairly. Auxiliary heads are discarded at inference. [R01] |
| `G_context` | Best greedy head recipe with the same protected original-input access as the candidate | Mandatory control against explaining the gain entirely through extra context. |
| `P_static` | Candidate scaffold with actions 0/2/4, a tuned fixed convex mixture of horizons 2 and 4, and a phase-only schedule | Primary simple alternative. Include identical-bank attribution and pruned-bank economic variants. |
| `P_audit` | Exact Section 5 discrete controller | Candidate; all shadows, `Q` accesses, freezes, and bank costs included. |
| `P_lookahead` | Section 5.5 two-task and same-horizon-menu three-task HydaLearn-derived adaptations, cost-faithful/safeguarded variants, plus constant and phase-only weights | Closest adaptive comparison; select the strongest valid complete-cost version and disclose extensions. |
| `P_local_score` | Same menu and transaction, scored using early CE instead of terminal CE | Tests the need for downstream information. |
| `H_periodic_bp` | Local training with occasional full terminal-loss BP replacing a local step; budget frequency calibrated to the candidate's extra work | Stronger-information hybrid reference. Charge its full graph, checkpointing, and real extra cost; do not label it strict-local. |
| `H_overlap` | Best greedy configuration with a one-layer boundary overlap and explicit parameter-update ownership | Development diagnostic if auxiliary depth is insufficient. Report the larger derivative reach, update schedule, and any duplicate work. |

`G_mlp/G_attn/G_mtp` implement the functional recipe at the project's common shape, tokenizer, and budget. They do not reproduce the source's published scale or numerical results. Verify the two-token branch/target arrangement against the inspected equations and a local fixture before using that attribution. Avoid indiscriminate MTP at every boundary as the only “strong” comparator. [R01]

The code reference `nanoGPT` is useful for a readable baseline, but its repository marks it deprecated in favor of `nanochat`. Current `nanochat` includes an MPS-oriented path with caveats; its full production recipe is not thereby validated on this laptop. Use inspected reference components and a small common harness, not an untested wholesale port or an old untuned model presented as strongest contemporary training. Muon is an optional common optimizer branch subject to the actual installed release and MPS tests. [B01], [B02], [B03]

Nonlocal references remain outside the controlled NLL table: full-scale greedy-LM runs; DiffusionBlocks' language generation experiments; GrAPE's particular feedback/derivative implementation; Dust's population workload; vision-only local SSL/predictive-coding results; and pretrained memory/post-training systems. They inform design and ceiling questions but are not substitutes for a run on the same corpus and budget. A partial implementation must say `inspired` or `adapted`, and cannot inherit an upstream performance claim.

### 6.5 Minimum attribution matrix and negative controls

Do not run a blind Cartesian product. First establish reliable `E2E`, `G_mtp/G_context`, and `P_static` learning curves. Then run the following targeted comparisons on development, retaining unsuccessful arms:

| Question | Required comparison | Confound addressed |
|---|---|---|
| Is correction more than renamed residual supervision? | Restricted `b=h` equivalent fixture; reset (`alpha=0`) versus accumulation; same non-accumulating scaffold dimensions | Algebraic equivalence, readout/extra-map capacity |
| Is protected access responsible? | Raw path on/off in both greedy and predictive arms; shared-token-input control | Added access mistaken for better credit assignment |
| Does conditioning on `b` matter? | `Qb` on/off with `alpha` fixed | Conditioning versus simple additive output |
| Is the policy useful? | Audited versus best static mixture, phase-only and direct lookahead on identical scaffold/bank; a matched single-horizon selector/weighting comparison for controller attribution | Good fixed hyperparameters, task-menu breadth, or established weighting |
| Is downstream feedback useful? | Terminal scoring versus local scoring; permutation of score-to-action labels within an audit; random actions with same empirical frequencies | Additional evaluation, phase, or chance effects |
| Does the interface enable feedback? | Use an exact `alpha=0` reset scaffold with every other path, head, bank, owner and optimizer convention unchanged; cross `alpha in {0,1}` with static versus identical audit policy | Generic controller benefit mistaken for an accumulation-dependent interaction |
| Are updates simply larger? | CE-only proposals norm-matched to the selected update, plus gradient/update-norm diagnostics | Step size rather than auxiliary direction |
| Is audit freeze overhead responsible? | Fixed-choice and random policies with the identical actor-only audit freeze schedule and dummy measurement cost | Different suffix-update count or data exposure |
| Is shared-head drift responsible? | Shared `W` frozen after a common warm-up; separately owned readout variant with counted capacity | Moving common target or head capacity |
| Does later computation use the state? | Bypass each correction, shuffle only protected training-fixture inputs, and compare frozen diagnostic probes | Stage-head health without terminal causal contribution |

Norm matching is a diagnostic, not an invitation to tune a baseline on confirmation outcomes. Apply the chosen norm rule using only information available within the same training transaction. For shuffled reward controls, record the permutation seed and ensure the score multiset and work remain identical. Do not leak the true best action through tie-breaking or auxiliary logs consumed by the policy.

For the primary interaction diagnostic, hold raw access and `Qb` conditioning on in all four cells. Define `gain_alpha = NLL(static, alpha) − NLL(audit, alpha)` and estimate `gain_1−gain_0` directly. Use the same chosen static mixture/weight, bank and audit parameters for causal attribution; also report independently tuned cell frontiers for practical competition. `G_context` remains a strong algorithm comparator, not an undefined interchangeable factorial cell. A later controller port to greedy heads must separately specify actor/private-head ownership and avoid adding the existing MTP twice.

Use the same actor-only audit freeze schedule in all four attribution cells. At each such event, construct/score the common menu and the fixed-mixture proposal in every cell; the static cell commits its fixed proposal and the adaptive cell uses its registered selector. This extra common diagnostic work and its costs are explicit; it is removed from the pruned economic baselines. With common update opportunities, data, bank and work, the interaction cannot be attributed solely to skipped suffix updates or one cell receiving an extra audit workload.

Make the other diagnostic policies concrete before running them. The raw-path capacity control replaces `P*r` with the same active `P*h_previous` while preserving shape and arithmetic; it does not claim equality of every notion of function-class capacity. The random policy samples frequencies frozen from development, or a registered estimator of **past** events only; never use a run's future final action histogram. For norm matching, construct the CE-only AdamW proposal from the same snapshot and scale its common LR multiplier by `||Delta_selected||/||Delta_CE||`, including decay and moments in each full displacement. Recreate the proposal with that LR and verify its norm. If the CE displacement is numerically unresolved/zero and the selected displacement is nonzero, mark the comparison unavailable; if both are unresolved, retain CE. Charge all work. This is an attribution diagnostic with access to the selected transaction's norm, not an information-restricted competitive baseline.

### 6.6 Smallest faithful tasks and sensitivity checks

Before the real-text pilot, use two generated causal tasks whose generator is part of the repository. They are **mechanistic diagnostics**, not evidence of general language competence.

**Associative retrieval:** generate a random mapping from a fixed inventory of distinct single-token keys to single-token values, place several key/value pairs with distractors, and later query one key. Sample values independently so unigram or key identity cannot predict the answer. Vary query distance and distractor count; balance answer values. Hold out mapping combinations, rename key/value symbols between split fixtures, jitter positions, and include a position-only baseline. Score the answer token separately from filler CE and show the entire distance curve.

**Compositional modular arithmetic:** generate prefix-style expressions from digits and addition/multiplication modulo a fixed small modulus, with explicit parentheses and answer delimiter. Compute labels with an independent interpreter. Hold out expression templates/compositions and depth ranges, not just token sequences. Compare against unigram, shallow/local, and E2E models; use balanced answers and avoid train/test duplicates. Report exact-match by expression depth. The generator must produce enough data to prevent memorizing a tiny fixed set, and every model receives the same generated stream at the declared budget.

The first diagnostic retains four groups, nonzero contracts, a nontrivial suffix, a train/audit split, and exact shadow transactions; reducing to a single linear layer would not test the interface/controller. Width and vocabulary can shrink for these checks without making a real-text claim. Validate a known informative auxiliary direction and an explicitly harmful one using the binary counterexample or a constructed two-feature task; the evaluator must distinguish them. Equivalent graph variants and duplicated checkpoint evaluations must show no manufactured advantage.

Escalate beyond a diagnostic only if the harness learns the generated task, the relevant baseline gap or controlled harmful-update effect is visible, and numerical/ownership tests pass. If neither a global model nor the intended task learns, debug or replace the unsuitable proxy. If the global model learns but local methods systematically fail under a sensitive protocol, that is useful evidence; do not automatically demand a larger model to avoid the result.

## 7. Tuning, measured budgets, and statistical decision procedure

### 7.1 Development and tuning fairness

Maintain a ledger of **every** tuning run, candidate variant, failed configuration, compilation/profiling cost, and selection decision. Separate shared reusable preparation from method-specific preparation; show both amortized per-run costs and total project expenditure. No pretrained model or teacher is used by the primary methods. A downstream-gradient oracle used for diagnosis is additional development expenditure and cannot secretly train the strict-local candidate.

Start with a shared AdamW learning-rate bracket `{1e-4,3e-4,1e-3}` and weight-decay choices `{0,0.1}`, with the numerical defaults in Section 5. These are search starting points. Expand an edge by a factor of three when adjacent development results show a stable improving trend and the numerical checks remain valid. Use cosine decay with an explicitly recorded warm-up fraction; initial warm-up candidates `{0.02,0.05}` are likewise a bracket. Tune objective weights, auxiliary head capacity, and the audit period/size only after basic optimization works. Evaluate a probed Muon/AdamW split as a common eligible optimizer branch; a method using it must have a matching, tested shadow-transaction implementation before entering confirmation.

Use the same measured **development compute allowance per method family**, and the same successive-budget promotion rule, for `E2E`, greedy local, static scaffold, discrete audit, and lookahead families. Determine that allowance from the pilot's cost and the configurations needed to resolve LR/learning-curve uncertainty; do not allocate many candidate trials while giving a baseline one default run. If the candidate needs a larger search, extend baseline search on comparable unresolved dimensions or disclose a distinct search-budget frontier. Report each family's quality versus cumulative tuning cost. Existing good published defaults can initialize a baseline; they do not replace local checks when tokenization and scale change.

Promote a configuration because it is competitive on the fixed development metric across paired seeds, has valid numerical behavior, and a longer run can resolve a learning-curve uncertainty. Do not select from a single lucky checkpoint. Preserve nearby budgets and unfavorable results. Choose the strongest eligible no-audit local configuration—including pruned static objectives and context-supplied greedy heads—as `B_local`, and the best valid lookahead adaptation as `B_lookahead`. Freeze one candidate `A`. A result applies to those selected pipelines; fresh seeds do not replicate the entire model-selection process. [S01]

Before each search round, write `development_runs.json` with its finite candidate list, common rung budgets, paired seed set, per-family allowance and promotion rule. Begin with optimizer/learning-rate questions before combining objective/frequency grids. Set the allowance from measured costs of the listed scientifically necessary probes and promoted rungs, giving every family the same available arithmetic budget; unused allowance is reported. At a rung, rank valid configurations by mean development NLL over the same registered seed set. Promote the better half to twice the training budget; when the boundary difference is within the development-estimated comparison resolution, treat it as a tie and retain both if the allowance permits, or choose by lower complete cost and then a frozen configuration-ID order. Never break a tie using protected results. A new round or expanded edge must name the unresolved optimization/learning-curve question it can answer and give competing families the same opportunity. This is a transparent search allocation rule, not a claim of globally optimal tuning or a fixed quota after which scientific uncertainty disappears.

### 7.2 Cost ceiling, operating regime, and headroom

Let `C` denote the project's counted arithmetic work, with two floating-point operations per multiply-add and the explicit backward-role ledger in Section 8. Count CPU controller fitting and method-specific preparation as well as neural operations; separately retain native wall time for operations, copies, I/O, and scheduling that this arithmetic proxy does not represent well. Report `C` as an analytic estimate with a versioned estimator, not a hardware instruction counter. Use real elapsed time as a second frontier so an accounting convention cannot manufacture practical value.

Freeze the accounting boundary in `cost_convention.json`: `C_*` includes all intrinsic run-specific learning work—forward/backward/optimizer operations, trained auxiliaries, rejected shadows, controller fitting/features, and any learned run-specific preparation. Shared corpus/tokenizer construction is reported once and under a declared amortization count; it is not charged repeatedly to only one method. Development evaluation used to select configurations enters the family tuning ledger. Protected final evaluation and reporting do not affect training and are recorded separately in total project cost. If a diagnostic changes an online policy, its work becomes training work rather than an excluded evaluation.

Count logical scalar additions/multiplications/divisions/reductions element by element, with fused multiply-add counted as two; record exp/log/sqrt and other nonlinear primitive evaluations as separate scalar-operation counts. The scalar convention is a reproducible arithmetic proxy, not equal hardware cost per primitive. Explicitly track optimizer loops, repeated embedding-gradient accumulation and CPU controller fitting. Log copying, serialization, hashing, indexing and I/O in bytes/operations and measured seconds; do not invent FLOPs from their elapsed time. Publish both the dominant matrix/attention subtotal and the scalar/controller subtotals so conclusions can be checked against alternative reasonable conventions. A method cannot be declared cheaper overall from this proxy alone when the measured time frontier disagrees.

For deterministic budget admission, define `C = matrix_attention_flops + scalar_arithmetic_ops + nonlinear_primitive_ops + controller_arithmetic_ops`, with one unit per listed nonlinear scalar primitive and no double-counting of operations already inside a matrix subtotal. Call this an **analytic FLOP-equivalent convention** in reports; publish all components and actual elapsed time. Any alternative weighting is a prespecified sensitivity analysis, not a post-hoc way to declare a winner.

WP06 measures learning curves over nested budgets, initially doubling a short calibrated run after warm-up. Extend only to resolve an identified uncertainty: adequate baseline learning, useful auxiliary head competence, a task expressivity limit, changing curve order, audit utility persistence, thermal behavior, or variance estimation. The first complete update is not evidence of useful learning, and “fully converged” is not required for a claim explicitly limited to a fixed budget.

Select a single common `C_*` from development curves, preferring the smallest **validated nontrivial regime** that supports a fair comparison at attainable precision. Record why a smaller regime is inadequate and publish neighboring development-budget results so a transient crossing remains visible. Choosing this operating point during development is legitimate; it does not support all-budget or scale-independent superiority.

At `C_*`, use the last complete permitted optimizer transaction whose counted cost fits the ceiling. Record unused remainder and never interpolate or choose a final checkpoint using confirmation NLL. In the compute panel, LR/warm-up progress is indexed to declared cumulative compute; in the token-matched mechanism panel it is indexed to main-update valid tokens. Owners' Adam bias-correction counters always count their actual updates. Account for any objective-dependent transaction cost before admission, using a conservative bound if necessary. Do not overspend one arm's ceiling and round it down afterward.

Estimate available headroom from development:

\[
G(C)=\operatorname{NLL}(B_{\rm local},C)-\operatorname{NLL}(E2E,C),
\]

and from the directly scored action regret, audit-cost fraction, and persistence of action rankings. If `G` is indistinguishable from noise at the smallest shape, a local/global-gap explanation cannot yet be tested there. A candidate can still improve local economics, but it must not claim to solve an unobserved gap. If direct action selection rarely differs meaningfully from the best fixed choice, the critic has little demonstrated opportunity.

Fit a monotone development baseline quality-versus-compute curve only over the observed trained range, showing raw checkpoints and uncertainty. For a candidate NLL, report the baseline compute needed to reach it when interpolation is supported. If the target lies outside the measured curve, report a bound or extend the baseline on development; do not extrapolate a dramatic speedup from a short slope. The same treatment applies to wall time. All audit, auxiliary, preparation, and inference overhead remains in the economic comparison.

### 7.3 Pilot precision and sample-size calibration

The independent training unit is a complete **replicate triad** `(A,B_local,B_lookahead)`. Pair common initialization tensors and update/audit streams within a triad using named RNG streams; keep triads independent. Extra auxiliary initialization must not shift common tensor seeds. Different-cost methods may consume different data prefixes at equal compute, so identical seed IDs do not imply identical exposure.

During exploration, start with four independent triads as an initial variability check and add another batch only to resolve a documented planning uncertainty. Four is not a sufficient-evidence rule or the confirmatory sample size. After pipeline, `C_*`, and precision selection, predeclare a **fixed-size fresh variance-calibration batch** of `m` triads. Choose `m` from the exploratory variance/cost forecast before observing this batch: tabulate predicted calibration-plus-confirmation cost over `m` using the formula below, and choose the smallest-cost adequate plan, with tail-diagnosis requirements recorded. All exploratory/calibration runs are excluded from final confirmation.

Choose an interval-resolution target `h_j` in **nats/token** for each primary contrast. The default planning scale is the NLL improvement that comparator `j` could buy near `C_*` with extra baseline compute equal to the candidate's measured audit/auxiliary overhead. Estimate it from the development curve and report its uncertainty. It is a decision-relevant precision yardstick, not a minimum gain the method must achieve. If this slope is flat or unresolved, publish a table of attainable precision versus fresh-run count and elapsed time, select a concrete resolution justified by the measured local/global gap or intended compute tradeoff, and record the choice before confirmation. A plan with no defensible attainable resolution becomes a sensitivity-limited study; it must not promise to detect arbitrarily small effects.

For pilot standard deviation `s_p,j` from `m` independent triads, a conservative planning value under approximately normal paired differences is

\[
\sigma_{U,j}=s_{p,j}\sqrt{\frac{m-1}{\chi^2_{0.025,m-1}}}.
\]

At the predeclared fixed `m`, these upper bounds have simultaneous 95% planning coverage for two variances under that model. They do not protect against arbitrary heavy tails. Do not extend this batch after inspecting its variance and continue to claim the same coverage. Adaptive exploratory variance estimates are planning heuristics only. If the fixed calibration reveals instability or unaffordable confirmation, report it, revise development openly if justified, and predeclare a new independent calibration/protocol. Retain every old batch and decision; none becomes protected confirmation.

The bound concerns variance on the fixed **development** population measured in that batch. It is not automatically an upper bound for another article population. Use prespecified development article resamples/strata matching the protected manifest's size/composition to forecast plausible transfer of seed variability, without opening protected losses. Include that sensitivity in the resource plan. Actual confirmation intervals may be wider than planned; do not extend the protected study opportunistically to attain the target width.

Choose the smallest integer final triad count `S >= 2` satisfying both

\[
t_{S-1,0.975}\frac{\sigma_{U,j}}{\sqrt S}\le h_j.
\]

The lower mathematical bound of two does not validate a tiny-sample normal approximation; development must support the intended inference. Also report the minimum detectable-effect/power curve. A planning approximation for one-sided level `0.025` and power `1−beta` is

\[
S\gtrsim\left[\frac{(z_{0.975}+z_{1-\beta})\sigma_{U,j}}{\delta}\right]^2.
\]

Use 90% power as a displayed reference, simulate sensitivity at the proposed `S` using pilot-supported skew/tails, and show several small absolute `delta` values. Do not choose `delta` from the most favorable selected pilot effect or compute “observed power” as a substitute for a final interval. Confidence/error probabilities here control inference; they are not arbitrary performance targets.

Stop development when the correctness/learning gates pass and a conservative fixed `S`, `C_*`, and measured runtime/memory plan can be frozen. If the plan is physically or operationally infeasible, first reduce irrelevant overhead or narrow the regime while preserving the mechanism and repeat development. Otherwise report resource-limited sensitivity. Longer runs are allowed when justified; there is no invented universal number of hours, model-size quota, or “run until significant” loop.

### 7.4 Frozen confirmation and multiple comparisons

Before producing **any** protected evaluation result, write and hash `confirmatory_protocol.json`, containing hypotheses, method/optimizer configurations, code/dependency/data hashes, `C_*`, final checkpoint rule, `S`, all seeds, primary population/masks, precision assumptions, failure/recovery policy, analysis code hash, and familywise error allocation. The runner must reject a confirmation job whose hashes differ. A routine fix that changes outputs after unblinding requires an amendment and fresh evidence; preserve the original result.

For article `d`, retain NLL sum `l[m,s,d]` and valid count `n[d]`. On fixed `D_confirm`, calculate

\[
D_{s,j}=\frac{\sum_d\{l[B_j,s,d]-l[A,s,d]\}}{\sum_d n[d]},\quad
L_j=\bar D_j-t_{S-1,0.975}\frac{s_j}{\sqrt S}.
\]

Each `L_j` is a one-sided 97.5% paired-t lower confidence bound. Bonferroni provides at least 95% simultaneous **lower** coverage for the two comparisons when their individual assumptions hold, even though they share a candidate run. The prespecified adaptive-improvement claim requires **both** `L_j>0`, valid measurements, and transparent guardrail results. If only one passes, state only that comparison. Quality superiority does not erase a time, memory, or inference regression.

Show all seed-level differences and ordinary 95% two-sided intervals as **per-comparison** intervals. Their complete intervals are not jointly covered at 95%. If jointly covered two-sided intervals are wanted, use per-comparison 97.5% two-sided intervals, with critical value `t_(S−1,0.9875)`, and use that critical value consistently in planning.

Paired-t inference is exact only under its normal model and otherwise approximate. Predeclare a bootstrap-t sensitivity analysis over whole triads if development supports it; show outliers rather than deleting them. If feasible `S` cannot support credible mean inference, label the result exploratory/inconclusive. Shared seed identifiers do not make a sign-flip/permutation test automatically exact; changing to a sign/median estimand is a methodological amendment, not a silent fix.

Randomize the within-triad run order, balance methods across comparable laptop time/thermal blocks, and retain environmental covariates. Do not stop early for favorable protected loss, repeatedly inspect checkpoint significance, or search subgroups after seeing a winner and call them primary. Hardware/numerical aborts follow the frozen failure policy, with all costs and outcomes retained.

### 7.5 Document uncertainty and independent confirmation

The primary interval conditions on the training dataset, tokenizer, selected pipelines, and fixed held-out articles. Millions of tokens are not millions of independent training replicates. To explore source-population generalization, use a **secondary paired crossed bootstrap**: independently resample triad IDs and article-cluster IDs, use the same resampled IDs/multiplicities for all methods, recompute the ratio of summed NLL difference to summed valid tokens, and average over sampled triads. Preserve source strata and duplicate-family clustering. This approximate product-factor procedure addresses two uncertainty sources but cannot make a convenience corpus representative or a tiny seed count reliable. [S02], [S03]

Do not resample individual tokens, resample articles independently for each method/seed, or average article perplexities. Set bootstrap RNG/count in the protocol. Calibrate the count until Monte Carlo variation of interval endpoints is negligible relative to reported precision, using development analysis fixtures; bootstrap draws are not extra experimental observations.

`D_reserve` remains untouched during the first study. For an independent replication, freeze the same method, data processing, `C_*`, and analysis, use **fresh training seeds**, and evaluate on the reserve with its sample-size rule specified before opening it. Re-evaluating the original checkpoints on new articles tests document transfer only. Report the two cohorts separately and require success in both for a replicated claim; never replace an unfavorable first result with a favorable reserve.

If instead the project wants two opportunities where either study can establish a claim, it must allocate the error budget across cohorts before either is opened—for example familywise `0.025` per cohort and one-sided `0.0125` per contrast. The default here is one primary study followed by a separately reported, same-method replication, with no “either wins” rule. A revision prompted by the first confirmation makes that result development evidence for the revised method; a frozen revision can use the reserve once, but it is a new study, not confirmation of the original fixed-corpus claim.

### 7.6 Failures and interpretation

External interruption permits resumption of the same seed from a verified complete checkpoint. Algorithmic NaNs, divergence, inability to fit the workload, or repeated invalid transactions are outcomes. Do not discard such seeds or silently substitute new ones. A predeclared, method-agnostic finite-output recovery policy may be part of the evaluated training pipeline; its cost and behavior must be identical in principle across arms. Otherwise report failure frequency and do not estimate an unconditional superiority claim from successful survivors only.

An interval spanning meaningful positive and negative effects is inconclusive. A precise interval excluding gains of practical interest is evidence against useful advantage in this regime, without proving universal equality. A reproducible regression or loss of a gain after stronger baseline tuning is evidence against the selected method or attribution. An incomplete fixed-sample study is reported with planned versus achieved precision and the reason, not relabeled confirmatory at an outcome-dependent stopping point.

The H3 interface-by-policy interaction and any E2E superiority claim are separate, prespecified follow-up families. Do not infer an interaction from “significant in one arm, not significant in the other.” Estimate the paired difference of policy gains directly, with a fresh/familywise-adjusted protocol if a confirmatory claim is intended. Secondary plots, probes, and task bins remain descriptive unless explicitly assigned an inference budget.

## 8. M3 Max implementation and resource plan

### 8.1 Discover the actual laptop before selecting a workload

The planning environment is not the target Mac. The following are **existing macOS/Python commands**, not commands executed here. Run them separately, capture stdout/stderr/exit status in `artifacts/environment/`, and redact serial numbers/identifiers from shareable reports:

```bash
uname -m
sw_vers
sysctl -n hw.memsize
sysctl -n hw.logicalcpu
system_profiler SPHardwareDataType SPDisplaysDataType -json
df -k .
vm_stat
sysctl vm.swapusage
pmset -g custom
python3 -c 'import platform,sys; print(platform.platform()); print(platform.machine()); print(sys.version)'
xcode-select -p
```

Record physical unified memory, available memory estimate and estimation method, free disk, GPU/core configuration, native `arm64` versus translated Python, macOS version, power mode, and battery/AC state. A missing developer-tools path is a finding, not automatic permission to install a large toolchain. Binary wheels should avoid a compiler dependency for the primary path. Do not assume the maximum M3 Max memory configuration or overwrite the user's system Python.

### 8.2 Primary stack and exact dependency strategy

Use native ARM Python in a new project virtual environment and **PyTorch MPS eager FP32** first. The inspected stable documentation resolves to PyTorch 2.14 and documents MPS availability checks with a macOS 14+ condition. These are observed documentation labels, **not a claim that a particular wheel has been installed or validated here**. Select the newest stable binary wheel compatible with the discovered native Python/macOS, record the actual version and wheel hashes, then freeze it after the operation probes. Prefer a supported Python 3.12/3.11 interpreter when already available; do not invent package compatibility or require a nightly build. [H01]

Existing bootstrap commands, subject to the discovered interpreter:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install --only-binary=:all: torch numpy scipy tokenizers huggingface_hub pyarrow datasketch PyYAML pytest
.venv/bin/python -m pip install -e . --no-deps
.venv/bin/python -m pip freeze --all
```

These unpinned installation commands are for discovery only. After successful probes, write an exact lock containing versions, wheel filenames/SHA-256, platform/Python tags, resolver output and package licenses. Save compatible wheels if disk allows; verify a clean-environment reinstall before confirmation. A package without a compatible binary wheel is a dependency decision, not an invitation to silently compile unknown native code. Use only needed packages; no full distributed training framework, pretrained model stack, or cloud service is required.

Use `.venv/bin/python` for the remaining project commands, or explicitly activate this venv before using the shorter `python` examples. Include the chosen packaging/build backend in the lock. A confirmation run requires a clean frozen source tree or a complete hashed source manifest including the exact diff and relevant untracked code; a Git commit ID alone does not detect edits in an editable install.

Set these environment variables **before importing PyTorch** in the runner:

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=0
export PYTORCH_MPS_FAST_MATH=0
```

Unsupported operations must surface during discovery. Do not silently fall back to CPU during “MPS” benchmarks or disable allocator safeguards to fit an oversized job. Keep default high-watermark protection. Begin without autocast, compilation, quantization, sampled softmax, or a custom attention kernel. Float64 is a CPU numerical-oracle type, not an MPS training assumption. Current MPS source rejects Float64 conversion. [H02], [H03]

Use `torch.nn.functional.scaled_dot_product_attention` only after checking forward **and backward**, mask semantics and causal behavior on the installed build. Explicitly set dropout probability zero in evaluation; do not assume module evaluation mode overrides the functional argument. A square training attention check does not establish correctness of non-square cached decoding. Use non-reentrant activation checkpointing (`use_reentrant=False`) for relevant baseline/memory comparisons, with a detached-boundary test. [H04], [H05]

**MLX is a conditional alternative, not an assumed faster replacement.** Its inspected current documentation labels version 0.32.3 and exposes stop-gradient, selected-parameter differentiation, checkpointing, explicit evaluation/synchronization, and memory counters. Lazy execution and fused-attention backward details require an actual operation probe. If PyTorch has a demonstrated blocking bottleneck, port the complete comparison and rerun numerical/ownership tests; do not benchmark only the candidate in MLX. For FP32 reference work, disable reduced-precision matrix multiplication with `MLX_ENABLE_TF32=0` where supported and verify the installed API. A fast-forward path alone does not establish fast or equivalent gradients. [H08], [H09], [H10], [H11]

### 8.3 Operation and transaction probes

Implement `local_contract_lm.cli probe-backend` before downloading the full corpus. Start with `B=2, T in {16,33}, d=128, heads=2, V=257`, plus repeated embedding indices and odd sequence lengths. Copy identical weights to a CPU FP64 reference and MPS FP32 path. Include:

- Embedding lookup/backward with repeated indices, non-contiguous Q/K/V layouts, RMSNorm, real-valued RoPE, SwiGLU, exact cross-entropy, causal/padding masks, and gradient clipping.
- At least three real AdamW steps to allocate lazy moments; default portable settings `foreach=False, fused=False` until alternatives are separately probed. Inspect parameter/moment/counter equality for the custom shadow transaction against the installed optimizer on the same fixture.
- All Section 5.7 ownership and transaction invariants, including early readout **input** gradients with detached readout weights and the separate CE-only embedding derivative.
- Full-forward versus cached-suffix equality, action-zero and `lambda=0` identities, deliberate stale-prefix failure, losing-proposal rollback, and audit-step owner counters.
- Activation-checkpoint output/gradient parity, microbatch accumulation, finite loss/gradients after repeated steps, and exact-scoped save/resume behavior.
- Full-sequence versus incremental cached inference, including non-square causal alignment and every group's two-state/raw-input path.

Initial numerical tolerances are `atol=1e-5, rtol=1e-4` for bounded FP32 forward fixtures and relative gradient L2 error `<=5e-4` away from near-zero reference gradients. These are engineering probe settings, not statistical success criteria. Report absolute errors, worst tensor/location, and denominator conventions. Investigate systematic discrepancies or changed action rankings; document a justified tolerance amendment if operation ordering explains a harmless difference. Never loosen tolerance merely to make a broken derivative pass.

Run zero-change and duplicate-proposal audits to estimate the numerical score floor separately from minibatch sampling noise. If FP32 cannot resolve relevant proposal differences, use larger `Q`, stable reduction, or a higher-precision CPU **small diagnostic**; do not call a numerical tie scientific evidence. Introduce BF16/FP16 only after a complete step/audit/ranking comparison to FP32, finite-state checks, and full cost measurement. Apply the same eligible precision to baselines. If precision changes the selected policy, treat it as a methodological variable requiring development and refreezing.

Probe logs must include actual device/dtype per tensor, installed versions, operation names, fallback state, errors, counters, and exception text. A missing counter is `null` with a reason. Smoke tests on Linux or a different Mac are not substitutes for these target-device checks.

### 8.4 Parameter and memory estimates

The following are **analytical starting-shape calculations**, excluding optimizer implementation padding/temporary buffers. Count the actual implementation before using them in a run forecast.

| Model component | Parameters at `L=8,d=384,d_ff=1024,V=8192,K=4` |
|---|---:|
| Conventional model, including untied `E/W` and final RMSNorm | 20,453,760 |
| Candidate deployed model: core, `E/W`, four `C`, three `P`, three `Q`, correction norms | 21,929,472 |
| Candidate two-branch train-only auxiliary bank | 5,506,176 |
| Candidate total resident parameters | 27,435,648 |
| Strong greedy attention/MTP model: deployed conventional core | 20,453,760 |
| Its three auxiliary attention heads and first-boundary MTP branches | 19,469,184 |

For FP32 AdamW, parameter, gradient, first-moment and second-moment arrays alone require approximately `16P` bytes after lazy state allocation. Candidate total is therefore about **418.6 MiB**, before activations or audit copies. Checkpoint weights and moments alone are about `12P` bytes; gradients need not be saved at a transaction boundary. Every local group's optimizer state remains resident on one laptop: splitting the graph does not divide these arrays by `K`.

Model peak memory as

\[
M_{\rm peak}\approx16P_{\rm resident}+M_{\rm activations}+M_{\rm attention}
+M_{\rm logits}+M_{\rm audit}+M_{\rm data/staging}+M_{\rm allocator/workspace}+M_{\rm checkpoint\ transient}.
\]

Fit rather than guess the dominant terms:

- Saved activation storage is roughly `a B T d L_live s`, with implementation-dependent factor `a` initially bracketed around 12–24 and element size `s`. In correctly sequential local training `L_live` is approximately one group plus its auxiliary path; global training retains more depth unless checkpointed. The factor is a calibration aid, not a measured peak.
- Materialized attention can contain multiple `B × heads × T × T` arrays per live layer. A fused forward or backward may avoid some arrays, but that depends on actual backend/path. Do not promise quadratic-memory elimination from a function name.
- One FP32 vocabulary-logit tensor is `4 B T V` bytes: at `B=2,T=512,V=8192`, about 32 MiB. Avoid keeping all local/MTP logits alive. Exact token-chunked CE can reduce this term after forward/gradient parity tests; count recomputation and offer it to baselines.
- A saved actor includes weights and both moments, approximately `12P_actor` bytes, plus gradients during proposals. Budget the immutable snapshot, current proposal and retained winner if all coexist. Host snapshots consume the same unified memory; serial proposals avoid three complete model copies but do not make copying free.
- Include detached `h/b/r` buffers, audit inputs, tokenizer/dataloader staging, NumPy-to-torch copies, framework caches, model-version diagnostics, and temporary checkpoint serialization buffers. File caches/RSS and accelerator allocations overlap on unified memory; do not add them as disjoint physical memory.

Measure `torch.mps.current_allocated_memory()`, `driver_allocated_memory()`, and `recommended_max_memory()` where available, process RSS and system memory/swap samples. These are not interchangeable; the first two are current readings, not built-in high-water peaks. Report a maximum of samples as **maximum sampled allocation**. Use a native Metal trace for a true transient peak claim if necessary. Synchronize before stable snapshots, and record allocator-cache policy. [H06], [H07]

Initial job memory budget is `min(0.60 × physical_memory, 0.80 × estimated_available_memory, 0.80 × recommended_device_memory)` using available terms only. These conservative margins are operational starting settings, not theoretical hardware limits or performance success criteria. Calibrate against actual pressure and prevent sustained swap growth; a run that “fits” by swapping is not a credible accelerator-throughput result.

### 8.5 Arithmetic ledger and total cost

Use dense-equivalent attention/matrix arithmetic for a reproducible proxy: a multiply-add counts as two FLOPs, even if a fused causal kernel skips part of a masked triangle. Label this convention and separate executed-device timing. Layer forward cost per processed token is approximately

\[
F_{\rm layer}=8d^2+6d\,d_{ff}+4Td,
\qquad F_{\rm head}=2dV.
\]

Fully trained dense layers usually need approximately three forward-equivalents including both input and weight derivatives; a frozen vocabulary head still needs forward plus input VJP, about `4dV`, when training its input. An embedding lookup is not a dense `6Vd` operation per token. Count trainable/frozen roles, auxiliary branches, norm/softmax/reduction work, and optimizer work explicitly in the estimator rather than using `6 × total_parameters × tokens` indiscriminately.

At the default shape, the following **estimated dense-equivalent matrix/attention** costs provide a check on the implementation ledger. They exclude scalar norm/softmax/optimizer work and copies; those must be separately estimated/logged and captured by wall time.

| Transaction | Approximate million matrix/attention FLOPs per processed update token |
|---|---:|
| Conventional E2E training | 122.683 |
| Candidate local scaffold, CE only, no auxiliary bank | 167.510 |
| Training the complete two-branch bank on detached features, additional | 51.905 |
| Ordinary candidate step with bank, action 0 | 219.415 |
| Ordinary candidate step with selected auxiliary, portable separate-embedding-gradient implementation | 253.526 |
| Greedy auxiliary-attention recipe with first-boundary two-token MTP | 265.421 |
| Three-proposal audit plus four terminal `Q` forwards, equal processed token counts in `S/Q`, independently recomputed actor gradients | 328.237 |

The selected-auxiliary ordinary step includes an additional frozen-bank input derivative and a repeated CE-only path to `E`. A straightforward pair of `autograd.grad` calls traverses the first group twice. With action 0, request `(theta_1,E)` jointly; with terminal CE, request `(theta_K,W)` jointly. An optimized adjoint or cached-component-gradient implementation is allowed after parity checks, but its savings cannot be assumed in advance. The strong greedy recipe's first embedding may receive its first-boundary MTP loss, unlike the candidate's deliberate CE-only embedding ownership; disclose and ablate that policy difference rather than weakening the source recipe.

An audit replaces an ordinary step. For `n` total opportunities and `a` audits,

\[
C_{\rm run}=\sum_{i\notin\mathcal A}C_{\rm ordinary}(i)
+\sum_{i\in\mathcal A}C_{\rm audit}(i)
+C_{\rm run\ specific\ preparation}+C_{\rm controller}.
\]

Do not add audits on top of ordinary updates that never occurred. Separately report main-update valid tokens, audit valid tokens, total label presentations over all losses/proposals, processed/padded token positions, owner update counts, and full-model forward/backward equivalents. Only update-token count is insufficient when a method repeatedly scores the same labels.

As a magnitude example, 100 million processed tokens at the conventional default cost are about 12.27 PFLOPs under this convention. If a future **measured** full-step throughput were 500 or 2,000 update tokens/s, that workload alone would take about 55.6 or 13.9 hours respectively. These are hypothetical conversions, not M3 performance predictions or required training budgets. Multiply actual per-run forecasts by tuning, variance-pilot, confirmation, reserve, and robustness workloads; include evaluation and checkpointing in total project time. The dominant expense may be repeated experiments, not fitting one model.

### 8.6 Calibration, timing, and configuration branches

After operation probes, benchmark actual complete ordinary and audit transactions at microbatch one, first the smaller fixture and then the default model. Allocate optimizer states before measurement. Warm until compiled kernels/allocator behavior and rolling timing stabilize; record the warm-up count and stability rule. Start with repeated windows of at least 20 warmed transactions and extend measurement only if the interval or thermal trend remains unresolved. Include a sustained run spanning normal/audit cycles to detect temperature/power drift; do not extrapolate a burst benchmark into an overnight forecast.

For a timed region, synchronize MPS, record `perf_counter`, execute the defined region, synchronize again, then stop the timer. The production boundary includes data loading/transfer, ordinary updates, audits, snapshots, controller work, and amortized checkpoint writes. Report kernel-only or cached-batch microbenchmarks separately. Include first-use/compilation and evaluation costs in the project ledger even when excluded from steady-state training throughput. Avoid logging every scalar via device-to-host transfer outside the measured boundary.

Repeat whole timing blocks in randomized method order with AC power and the same documented power mode, active workload and display/background conditions. Record raw block times, median and interval, CPU/GPU/system pressure, cache state, and thermal trends. Repeated steps characterize this run's timing noise, not independent model-quality evidence. Native profiler signposts/Instruments are optional diagnostic tools; no privileged energy measurement or specialized equipment is required. If energy is not measured, mark it unmeasured. [H07]

Choose effective-batch valid tokens by a short gradient-noise/throughput pilot: initially bracket 4096, 8192 and 16384 next-token targets, accumulating exact weighted microbatches. Keep the selected effective batch common in mechanism comparisons. In economic comparisons, each method may use a locally efficient microbatch/checkpointing choice while reporting effective-batch/optimization changes and retuning as needed.

Apply the following fallback order, recording the reason and rerunning dependent probes:

1. Reduce microbatch and increase accumulation, preserving effective tokens, context, architecture and objective.
2. Release local graphs/logits promptly; use parity-tested exact token-chunked CE and baseline activation checkpointing. Count recomputation.
3. If memory still fails, reduce width to `d=256, heads=4, d_ff=704`, keeping `L=8,K=4,T=512` and applying the shape consistently to direct comparisons.
4. If throughput makes the experiment uninformative, calibrate a shorter or narrower common operating regime while checking that the baseline gap/contract utility and required dependencies remain expressible. A shorter context changes the claim; never use `T=128` to claim preservation of a 2,000-token dependency.
5. If the MPS path has an operation-specific limitation, fix a mathematically equivalent common operation or use the fully probed MLX branch. If only CPU toy checks are viable, deliver a mechanism diagnostic and a resource-blocked real-text result, not a completed laptop LM PoC.

Increase to `L=12,d=512,K=4` or a longer context only if the smaller validated regime reveals a relevant expressivity/baseline-gap limit, or a promising effect with a measured cost/precision forecast justifies checking scale. No mandatory 400M model is needed to answer this local question, and a 20M result cannot imply a frontier-scale result.

### 8.7 Storage, checkpoints, interruption and recovery

Before downloads, forecast source files (about 315 MB), token arrays from measured tokenizer output, provenance/logs, temporary data-build files, wheel cache, and checkpoint retention for the planned run count. The candidate's weights/moments are about 329 MB decimal before metadata and serialization overhead; allow simultaneous old and new checkpoints during atomic replacement. Large per-article logits are unnecessary; retain NLL sums and token counts. Audit records contain scalars/features and identities/hashes, not every full activation.

Checkpoint only at completed ordinary/audit transactions. Save all parameter owners, optimizer states/counters, schedule progress, cumulative cost ledger, selected action, bank/critic state, audit history needed for learning, Python/NumPy/CPU/MPS RNG states when supported, sampler cursors, data/config/code hashes, and pending next transaction. Serialize to a temporary file, flush, hash, then atomically rename; retain the last known-good checkpoint until the replacement is verified. Restore into the same locked environment and reject incompatible ownership/configuration hashes. Never unpickle an untrusted external checkpoint.

Calibrate write duration and choose cadence from acceptable lost work versus measured write overhead, recording that tradeoff; it is an operational setting, not a scientific stopping rule. Also save at milestones/final state and before long diagnostic branches. Do not delete failed-run evidence to save space; retain compact failure records and the final recoverable state even when intermediate checkpoints are pruned according to a recorded policy.

An external interruption resumes the same replicate. On resource exhaustion, stop cleanly when possible, preserve the error and memory trace, restart the process to clear uncertain allocator state, reduce only a preauthorized equivalent microbatch setting, and verify the next transaction. Any effective-batch, precision, model, dataset, or loss change is a new configuration/development amendment. Do not resume it as if the original confirmatory run completed unchanged.

Maintain separate `intrinsic_admitted_work`, `external_replay_work`, and total actual elapsed/work records. The learning schedule resumes from the last complete checkpoint's intrinsic progress so an external interruption does not silently alter the frozen pipeline; replay waste is added to the actual economic report, with bounds if an interrupted operation's completion is unknown. Algorithm-caused divergence, resource exhaustion and its predeclared recovery work are **not** reclassified as external replay to make a method cheaper. Checkpoints preserve the next opportunity index, held scalar/mixture/lookahead policy, pending whole-window batch IDs, cost-admission state and every named random stream. Verify these identities on restart, not only similar loss.

### 8.8 Scope of the economic claim

Inference removes every auxiliary head and controller. Prefill/decode still computes the deployed groups, protected projections and correction states. Implement the same causal cache strategy and measure batch-one prefill at declared prompt lengths, decode over a fixed token count, memory, and artifact size with identical vocabulary/precision. Validate logits before timing optimized inference. A larger deployed scaffold can lose an inference tradeoff even if local training improves.

This is one serial laptop. There is no measured distributed speedup. A conventional partition sends roughly a hidden activation forward and its gradient backward; the candidate may send both `h` and `b`, retained raw inputs, scalar audits, and readout synchronization. Shared `W` has `Vd` entries. Depending on caching and ownership, the forward-only protocol can transmit as much or more data and synchronize just as often. A communication benefit requires a separate byte/message/latency model and later actual measurement; it is not inferred from detachment.

## 9. Executable project specification and work packages

### 9.1 Repository, commands, and shared interfaces

Create a local Git repository for code/configuration/protocols, without committing corpus text, credentials, giant checkpoints or machine identifiers. All paths below are relative to its root. These are **proposed files and project CLI contracts to implement**, not existing packages or commands verified during this research.

| Path | Responsibility |
|---|---|
| `README.md`, `pyproject.toml`, `requirements.lock`, `THIRD_PARTY_NOTICES.md` | Setup, supported execution commands, pinned environment and code-license provenance |
| `src/local_contract_lm/cli.py`, `config.py` | Strict CLI/config validation, command dispatch, resolved-config hash |
| `src/local_contract_lm/data.py`, `tokenization.py`, `synthetic.py` | Pinned downloads/checksums, articles/splits, byte BPE, deterministic example generators |
| `src/local_contract_lm/model.py`, `heads.py`, `methods.py` | Common causal layers, boundary states, auxiliary heads, E2E/local adapters |
| `src/local_contract_lm/ownership.py`, `optim.py`, `audit.py` | Explicit derivative/optimizer ownership, exact snapshots/proposals, frozen suffix scoring and commit |
| `src/local_contract_lm/train.py`, `checkpoint.py` | Budget-aware transactions, accumulation, schedules, interruption and resume |
| `src/local_contract_lm/evaluate.py`, `costs.py`, `measure.py` | Article NLL, versioned arithmetic/label ledger, synchronized timing and memory sampling |
| `src/local_contract_lm/analysis.py`, `critic.py` | Frozen paired/crossed inference and optional chronological discrete predictor |
| `tests/test_math.py`, `test_ownership.py`, `test_audit.py`, `test_data_eval.py`, `test_backend.py`, `test_resume.py`, `test_analysis.py` | Meaningful numerical, leakage, transaction, recovery, and analysis fixtures |
| `configs/` | `smoke.yaml`, `e2e.yaml`, `greedy_mtp.yaml`, `predictive_static.yaml`, `predictive_audit.yaml`, `lookahead.yaml`, and resolved immutable run configs |
| `protocols/` | Hypotheses, data split manifest, baseline register, confirmation JSON, amendments and access policy |
| `data/source/`, `data/processed/`, `data/NOTICE.md` | Unmodified source assets, hashed token/offset arrays, attribution and exclusions |
| `artifacts/environment/`, `artifacts/calibration/`, `artifacts/pilots/` | Hardware/dependency probes, cost fits, raw development evidence |
| `runs/<run_id>/` | Immutable config, metadata, metrics/audits JSONL, article losses, checkpoints, failures, final status |
| `reports/`, `decisions.jsonl`, `progress.md` | Complete favorable/unfavorable result synthesis and resumable decision/progress history |

Use a minimal common adapter instead of independent training scripts whose loss masks and costs diverge:

Derive named 64-bit seeds from SHA-256 of the UTF-8 tuple `(master_seed, phase, replicate_id, stream_role)`, using a recorded master seed (initially `20261011`) and fixed serialization. Use separate roles for common tensor initialization, each auxiliary/projection initialization, data permutations, audit sampling, random controls, and bootstrap analysis. The method name must not alter paired common streams; method-specific extra tensors use separate keys. Confirmatory/replication phases use disjoint seed namespaces, and all concrete derived values are saved before their runs.

```python
# Proposed interfaces; implement with typed dataclasses and validated tensors.
Batch = {
    "input_ids": "integer[B,T]",
    "position_ids": "integer[B,T]",
    "attention_mask": "causal + padding, tested backend convention",
    "targets": "dict[offset -> integer[B,T]]",
    "valid": "dict[offset -> bool[B,T]]",
    "document_cluster_ids": "list[str]",
    "effective_batch_valid_counts": "dict[offset -> int]",
}

class MethodAdapter:
    def forward_inference(self, batch, cache=None): ...
    def parameter_owners(self): ...            # owner -> unique tensor names
    def ordinary_step(self, batch, state): ... # StepResult with complete cost
    def audit_step(self, update_batch, audit_batch, state): ...
    def state_dict_complete(self): ...
    def load_state_dict_complete(self, state): ...

class AuditEngine:
    def snapshot(self, actor_owner, state): ...
    def propose(self, snapshot, batch, action, optimizer_spec): ...
    def score(self, proposal, frozen_model, audit_batch, immutable_cache): ...
    def commit(self, proposal, expected_model_version): ...
```

`BoundaryState` explicitly contains `h`, `b`, `r`, positions, masks, group index and model version. A cache records its producing parameter/version signature, tokenizer/data identity and evaluation mode. `Proposal` contains actor parameter/moment/counter state, action, pre-state hash, gradient/update norms, consumed RNG state and cost; it cannot contain a silently changed shared parameter. A stale version causes an exception. `StepResult` contains loss sums/counts, model/owner versions, consumed example IDs, cost ledger, timings, memory readings and status.

Resolve a configuration to a fully specified immutable object. An illustrative **development starting configuration**, not a confirmation configuration, is:

```yaml
schema_version: 1
phase: development
method: P_audit
model:
  layers: 8
  groups: 4
  width: 384
  heads: 6
  ffn_width: 1024
  vocabulary: 8192
  context: 512
  raw_access: true
  predictive_conditioning: true
  accumulation_alpha: 1.0
  dropout: 0.0
training:
  device: mps
  dtype: float32
  microbatch_sequences: 1
  effective_valid_tokens: 8192
  optimizer: adamw
  learning_rate: 0.0003
  betas: [0.9, 0.95]
  epsilon: 0.00000001
  owner_clip_norm: 1.0
  dense_weight_decay: 0.1
  embedding_readout_weight_decay: 0.0
  warmup_fraction: 0.02
  schedule: cosine
contract:
  actions: [0, 2, 4]
  lambda: 0.3
  bank_train_all_actions: true
  initial_action: 0
  audit_period: 64
  audit_valid_tokens: 8192
  freeze_non_actor: true
  numerical_tie_tolerance: null
budget:
  axis: analytic_total_flops
  ceiling: null
  cost_model_hash: null
provenance:
  data_manifest_hash: null
  tokenizer_hash: null
  code_commit: null
  dependency_lock_hash: null
  seed_manifest_hash: null
```

`effective_valid_tokens` is the Section 6.3 whole-window **minimum**; every loss uses the actual batch count. `null` means **unresolved and forbidden for a training/confirmation job**, never zero/unlimited. The resolver fills fields from measured artifacts and records decisions; it may not choose hidden defaults. For action 0, a bank-pruned economic configuration is a different named config, not a silent runtime optimization.

Two narrowly bounded commands may run before a data/cost/tolerance hash exists. `probe-backend` uses only the fixed small generated tensor inventory in Section 8.3 and three optimizer steps; it cannot load a corpus or produce a trained-model result. `calibrate --config PATH --fixture synthetic --transactions normal0,normal2,normal4,audit --repetitions 3 --memory-budget-bytes VALUE --output PATH` executes exactly that finite inventory from disposable states, with the Section 8.4 measured-memory safety budget. Three repetitions are a startup/calibration bound, not adequate performance evidence. Later explicitly sized calibration runs may extend the inventory to resolve timing/thermal uncertainty; they still require a concrete repetition count/memory budget and cannot access protected data. WP01 certifies primitives and miniature transaction fixtures; WP04 certifies the complete model/ownership/cache/resume implementation. This exception does not unlock an arbitrary unbudgeted training job.

### 9.2 Required records and units

Every JSONL record has `schema_version`, UTC timestamp, `run_id`, phase, source code/config hashes and event type. Encode finite numbers only; use `null` plus an error/status field for unavailable quantities.

| Record | Required fields beyond common identity |
|---|---|
| `metadata.json` | Method/architecture/optimizer versions, owner map, active/deployed/train-only parameter counts, hardware/OS/package locks, data/tokenizer/license manifests, named seeds, protocol hash |
| `metrics.jsonl` | Transaction index/model version/owner counters; loss sums in nats and valid counts by objective; cumulative processed/update/audit tokens and label presentations; matrix/attention, scalar/optimizer, controller and preparation FLOPs; elapsed seconds; RSS and maximum sampled MPS bytes; swap/pressure status |
| `audits.jsonl` | `S/Q` article-cluster IDs or resolvable hashes, owner/pre-state/cache signatures, action menu or continuous policy weights, all permitted pre/post scores, gradient/update norms, numerical tie floor, chosen action/state, proposal/commit equality, age since prior audit, per-action work/time; independent `Q_check` diagnostic identity/score in a separate flagged record |
| `article_losses.parquet` | `run_id`, checkpoint/protocol hash, split, article/cluster ID, token count, byte count, NLL sum, optional special-excluded NLL, prespecified strata |
| `failures.jsonl` | Exception/reason, numerical or external category, last complete transaction, memory trace, recovery action, counted wasted work, resumed/new-config status |
| `decisions.jsonl` | Decision ID, question/hypothesis, available evidence hashes, action, alternatives, rationale, owner, affected configs and next gate |
| `amendments.jsonl` | Old/new protocol hashes, exact changed assumptions/population/metrics/configuration, why, protected data already accessed, required fresh confirmation |
| `evaluation_access.jsonl` | Requested/used split, command/protocol hash, checkpoint IDs, timestamp, approved experiment phase; refuse protected split access before freeze |

Preserve raw observations before formatting tables. A report must be regenerable from immutable configs and records without reading this conversation. Each work package updates `progress.md` with completed checks, unresolved questions, next command and entry condition.

### 9.3 Work-package graph and stage gates

The ordered program is `WP00 → WP01 → WP02 → WP03 → WP04 → WP05 → WP06`. Only a frozen, valid WP06 enters WP07 confirmation. WP08 attribution uses development or separately registered evidence, and can run after WP06; it must not secretly modify WP07. WP09 replication follows an interpretable frozen result. WP10 is conditional mechanism expansion and returns any changed method to development/freeze. WP11 synthesizes every branch, including early stops. No later stage is required when an earlier informative result resolves its premise.

#### WP00 — Establish the research contract and reproducible skeleton

**Objective/risk:** Preserve the selected question, source limits and experiment provenance; prevent an implementation from quietly changing the research problem. **Dependencies/inputs/entry:** This complete plan, a project directory and ordinary local development permissions; no corpus or accelerator needed.

**Actions/files/commands:** Create the Section 9.1 layout and a minimal installable CLI; copy the hypothesis/baseline/data policies into `protocols/`. Define strict config/record schemas, named seed streams and append-only decision/progress records. Extract the Section 12 mathematical script into `tests/fixtures/mechanism_math_checks.py`. Record source reuse and license status before copying third-party code. Implement `python -m local_contract_lm.cli validate-config --config PATH` and `status --project .`.

**Tests/outputs:** Invalid unresolved confirmation configs and unknown fields fail; valid development fixtures round-trip to the same canonical hash. Produce a code commit, `protocols/research_contract.md`, source register and initial status. **Resources:** Negligible model work; local disk for code/metadata only.

**Acceptance/next action:** If the contract and schemas are explicit, enter WP01. If an essential assumption is contradictory, amend before infrastructure. A missing optional package or unavailable upstream code triggers the documented clean implementation path, not a false reproduction claim.

#### WP01 — Verify laptop, dependencies, math, and critical operations

**Objective/risk:** Detect incompatible backend paths, broken detachment or optimizer semantics before data/model investment. **Dependencies/inputs/entry:** WP00; actual target laptop, native Python discovery, public package access and licenses relevant to use.

**Actions/files/commands:** Execute Section 8.1 discovery; create the venv and dependency lock; implement `probe-backend --output artifacts/environment/backend_probe.json`, `probe-optimizer --output artifacts/environment/optimizer_probe.json`, and `python -m pytest tests/test_math.py tests/test_backend.py`. Run the provided CPU checks and MPS forward/backward/three-step probes. Test portable AdamW state cloning and rollback. Record optional Muon eligibility; it is not required to pass the primary AdamW path.

**Tests/outputs:** Detailed device/dtype/gradient ownership, causal mask, optimizer parity, numerical-error and memory-counter reports; wheel/code license inventory; reproducible lock. **Resources:** Tiny generated tensors; seconds/minutes are expected qualitatively, but log actual time. No full corpus required.

**Acceptance/next action:** Compatible, numerically valid primary operations enter WP02. An implementation discrepancy is repaired and the affected probe rerun. An MPS-specific limitation activates an equivalent common operation or the whole-comparison MLX branch. If only CPU diagnostics remain, record target feasibility blocked and complete the diagnostic/report branch; do not label the LM PoC complete.

#### WP02 — Build immutable data and a leakage-resistant evaluator

**Objective/risk:** Eliminate split, mask, tokenization and likelihood artifacts. **Dependencies/inputs/entry:** WP01 pass; verified dataset notices, revision and hashes in Section 6.

**Actions/files/commands:** Implement `prepare-data --spec protocols/data_spec.json`, article reconstruction/deduplication, deterministic split manifests, tokenizer training and memory-mapped arrays. Implement `evaluate --checkpoint PATH --split development --output PATH` with the exact target-window convention. Protected split access is disabled. Run `python -m pytest tests/test_data_eval.py tests/test_analysis.py`.

**Tests/outputs:** Local source SHA-256 matches; text/row conservation; duplicate clusters do not cross pools; tokenizer byte round-trips; target counts invariant to batching; tiny uniform/unigram CE agrees with arithmetic; empty/masked/horizon cases pass. Produce `data_manifest.json`, tokenizer hash, notices, article statistics and hand-checked loss fixtures. **Resources:** Approximately 315 MB source download plus measured token/temp arrays; stream parsing/tokenization and forecast disk before work.

**Acceptance/next action:** Valid manifests/evaluator enter WP03. A hash mismatch halts that asset use; resolve upstream routing/version, not expected-hash editing. A parsing/count discrepancy is diagnosed before training. Unresolved material use conditions or missing assets require an approved-source population amendment and a fresh freeze, not silent dataset substitution.

#### WP03 — Establish strong baseline learning and cost accounting

**Objective/risk:** Ensure the candidate is compared with competent current local/global learning. **Dependencies/inputs/entry:** WP02 harness; common model/heads and owner interfaces; validated primary optimizer.

**Actions/files/commands:** Implement `U0/E2E/G_linear/G_mlp/G_attn/G_mtp/G_context` from Section 6, starting with synthetic and a fixed training-only real-text fixture. Implement `train --config configs/e2e.yaml`, `train --config configs/greedy_mtp.yaml`, `profile --config PATH --output PATH`, and the versioned cost estimator. Resolve smoke budgets from target measurements before these runs. Verify branch targets/readout sharing against source equations, then begin development LR/head checks. Add `E2E_scaffold` once the scaffold exists in WP04.

**Tests/outputs:** Causality/ownership maps; train/development loss curves better than the floor where expected; ability to overfit a tiny training fixture; analytical operation counts checked on tiny shapes; first real timing/memory records; a baseline implementation-fidelity table. **Resources:** Common small fixture and calibrated short runs; auxiliary heads counted in memory/work. No protected evaluation.

**Acceptance/next action:** Well-behaved global and strong local runs enter WP04. If no model learns, debug data/optimizer/masks. If only the local implementation fails a source-consistent fixture, repair it. A real local/global gap after tuning is useful experimental headroom; absence of a gap informs scale/sensitivity without invalidating the harness.

#### WP04 — Implement the minimum predictive interface and audit controller

**Objective/hypotheses:** Test H1–H3 with exact on-policy counterfactuals, without introducing a learned critic. **Dependencies/inputs/entry:** WP03 competent baselines; Section 5 state/ownership contract.

**Actions/files/commands:** Implement `P_static`, `P_audit`, `P_lookahead`, `P_local_score`, shared bank, synchronous barrier, full state snapshots and serial audit proposals. Add `E2E_scaffold`. Implement `audit-fixture --config configs/smoke.yaml --output artifacts/calibration/audit_fixture.json`. Run `python -m pytest tests/test_ownership.py tests/test_audit.py tests/test_resume.py`. Compare the proposed custom optimizer to the actual installed implementation, including nonzero moments and counter offsets.

**Tests/outputs:** Every applicable Section 5.7 invariant; scored-state parity for discrete/safeguarded policies and constructed weighted-update parity for unscored source-style policies; no-gradient escape through `b/r/W/E`; bank training independent of actor gradients; correct frozen-owner/per-parameter counters and complete loss/cost records. Include the discrete forced-action-zero/`lambda=0` cases, lookahead omitted-loss/main-only behavior, a near-tie retaining CE, stale-cache rejection and losing-proposal rollback. **Resources:** Tiny models first, then one full-shape normal/audit cycle including allocated moments/snapshots.

**Acceptance/next action:** Passing core transactions enter WP05. A graph/state failure is an implementation defect, not evidence against the scientific idea. If the two-state graph reduces to a tested equivalent control, narrow the novelty claim immediately. If snapshots/costs are material, optimize only with parity tests and remeasure; do not remove them from accounting.

#### WP05 — Demonstrate sensitivity and diagnose indispensable mechanisms

**Objective/hypotheses:** Determine whether the smallest workload expresses local/global conflict, useful contracts and observable suffix consequences. **Dependencies/inputs/entry:** WP04 invariants and WP02 evaluator.

**Actions/files/commands:** Implement `diagnose --suite mechanism --output artifacts/pilots/mechanism/`. Run the restricted residual identity, harmful-local-update fixture, two generated task families and initial real-text learning curves. Measure terminal/stage losses, bypass effects, local-versus-terminal update alignment on tiny diagnostic copies, and independently scored action utility using Section 5.6's `Q_check`. At prespecified later lags, form fresh proposals from that later model state and score them on fresh independent `Q_check` batches; compare ranking/policy regret without mutating the live run. Charge every replay/score. Any global gradients used to diagnose alignment are isolated from candidate updates and charged as diagnosis.

**Tests/outputs:** Known difference detected, equivalent model difference absent, synthetic answer balance/split integrity, exact-match by distance/depth, real-text unigram/global/local curves, decision record on expressivity and proxy limitations. **Resources:** Calibrated short workloads; diagnostic graphs may be larger but must fit measured memory. No broad architecture sweep.

**Acceptance/next action:** A sensitive harness and meaningful trained regime enter WP06 regardless of whether the candidate wins. If the proxy cannot express the intended dependency, repair its design or move to justified real-text/shape context. If a sensitive task shows harmful or useless contracts, record that evidence and favor a static/simple alternative; do not add a critic to hide absent signal. Resource failure remains a separate outcome.

#### WP06 — Calibrate economics, tune fairly, and freeze confirmation

**Objective/hypotheses:** Resolve H4 feasibility and measurement sensitivity; select pipelines without protected-data feedback. **Dependencies/inputs/entry:** WP05 interpretable diagnostics and stable baseline/candidate implementations.

**Actions/files/commands:** Implement `pilot --manifest protocols/development_runs.json`, `analyze --phase development`, and `freeze-protocol --input protocols/proposed_confirmation.json --output protocols/confirmatory_protocol.json`. Run full normal/audit timing cycles, measured memory/disk forecasts and neighboring-budget learning curves. Tune per Section 7; select `A/B_local/B_lookahead/E2E`, audit period/size and final checkpoint rule. Run fresh variance-pilot triads, calculate `C_*`, precision, `S`, power/MDE and complete project cost. Document pruning/precision/checkpoint optimizations and exposure differences.

**Tests/outputs:** Cost identity including replaced ordinary steps; thermal/cache stability; invariant masks/hashes; analysis recovery of planted mean/null cases; baseline search parity; `calibration.json`, complete run ledger, selected configs, sample-size calculation and frozen protocol. **Resources:** Measured per-stage estimates times actual planned runs; forecast uncertainty explicitly. No guessed overnight throughput or mandatory model-size gate.

**Acceptance/next action:** A valid attainable frozen protocol enters WP07; attribution experiments may proceed in WP08 with their declared status. A clear precise negative pilot can justify stopping/diagnostic synthesis if the intended question is answered; do not call it a protected confirmation. An inconclusive proxy triggers a justified regime change and new development. Excessive projected resources yield a documented narrower or sensitivity-limited claim, not selective loss of seeds.

#### WP07 — Execute the protected primary comparison

**Objective/hypotheses:** Confirm or reject the two prespecified H2 comparisons at `C_*`. **Dependencies/inputs/entry:** WP06 frozen protocol, fresh seed list, clean reinstall/checkpoint reproducibility and adequate disk/memory.

**Actions/files/commands:** Implement `run-protocol --protocol protocols/confirmatory_protocol.json` and `analyze --protocol protocols/confirmatory_protocol.json`. Verify hashes before each run, randomize run order, execute the fixed triads and final-checkpoint evaluations on `D_confirm`, log access, and apply the frozen analysis without interim selection. Preserve failures and all raw article sums. Do not change hyperparameters after protected outcomes arrive.

**Tests/outputs:** Complete/failed run inventory, primary NLLs/differences/lower bounds, per-comparison intervals, cost/time/memory guardrails, seed scatter, achieved precision and protocol deviations. **Resources:** `S ×` measured sum of three run costs plus analysis/evaluation/checkpoint overhead, with measured uncertainty.

**Acceptance/next action:** Both lower bounds positive supports the defined conditional adaptive claim, subject to validity and honest tradeoffs; proceed to attribution/replication if not already resolved. One passes supports only that comparison. A precise regression/null against useful gain informs stopping/pivot. A wide interval or incomplete sample is inconclusive. Any invalidating bug pauses interpretation, preserves original outputs and returns the affected branch to development with an amendment.

#### WP08 — Attribute and stress-test a result

**Objective/hypotheses:** Determine whether H1/H3 explain a gain or a simpler intervention does. **Dependencies/inputs/entry:** Valid WP06 pipelines; WP07 outcomes may motivate **exploratory** work only, unless the follow-up was already frozen.

**Actions/files/commands:** Implement `ablate --manifest protocols/attribution_runs.json` and registered diagnostics from Section 6.5. Prioritize fixed scaffold/raw access, identical-controller interface interaction, freeze-schedule/information/capacity controls, norm matching, action shuffling, and shared-readout drift. Check context/dependency distance, `K=2` versus `K=4`, nearby width/budget, and a declared corpus slice only when each resolves an observed explanation. Measure inference with `benchmark-inference --manifest PATH` and validate full versus cached logits first.

**Tests/outputs:** Paired interaction estimates rather than significance comparisons; training/head parameter/work exposure tables; terminal bypass effects; subgroup plots including regressions; complete deployed inference timings/memory. **Resources:** Pilot-calibrated targeted arms; common RNG and information where applicable. Do not run every optional factor combination by default.

**Acceptance/next action:** A persistent interaction and robust terminal benefit support the mechanism-specific interpretation. Equal controller gains across interfaces support a generic controller result. A fixed/context-only/extra-head control explains the gain, so adopt the simpler method and narrow attribution. A post-hoc revised candidate returns to WP06/fresh confirmation, never overwrites WP07. Untested robustness dimensions remain limitations.

#### WP09 — Independently replicate the frozen result

**Objective/risk:** Distinguish a reproducible effect from a selected run/corpus outcome. **Dependencies/inputs/entry:** Interpretable frozen result, unused `D_reserve`, unchanged same-method pipeline for a replication claim, a prespecified fresh-seed/sample rule.

**Actions/files/commands:** Create `protocols/reserve_protocol.json`, freeze before reserve evaluation, use new replicate triads and the same method/configuration/`C_*`/tokenizer. Run the same CLI/analysis against the reserve. Perform clean-environment reconstruction from locks and a verified uninterrupted-versus-resumed comparison; record tolerance-qualified reproducibility where backend nondeterminism prevents bitwise identity.

**Tests/outputs:** Separate replication table, original/result agreement or disagreement, new seed/article uncertainty, resource measurements, reproducibility instructions and hashes. **Resources:** Recalculate precision for the reserve's article population and measured variance without opening its loss outcomes; actual repeated-training cost is explicit.

**Acceptance/next action:** Consistent effects in both cohorts support a replicated local claim. Disagreement is reported and investigated on development without concealing either result. A revised method tested on the reserve is labeled a new study. If reserve precision or resources are insufficient, state that independent confirmation remains incomplete.

#### WP10 — Conditional critic, horizon, or scale extension

**Objective/hypotheses:** Test H5 or a specifically supported boundary of H1–H4. **Dependencies/inputs/entry:** Direct audits show actionable utility and a measured overhead/persistence problem, or validated small-scale evidence identifies an expressivity limitation. This package is not automatically admitted.

**Actions/files/commands:** For amortization, implement `fit-critic --audit-manifest PATH --chronological` and prequential linear/constant/phase-only comparison, then test saved-audit-cost economics. For delayed suffix adaptation, replay identical saved states with a prespecified short horizon of suffix relearning and count every replay; test whether immediate ranking is systematically misleading. For scale/domain, freeze the concrete new configuration/data manifest and resource forecast. Change one unresolved mechanism at a time, then return to WP06 for a new protected design.

**Tests/outputs:** Out-of-time action regret/calibration and realized utility, complete cost comparison, horizon/radius/drift diagnostics as applicable, admission decision and newly bounded claim. **Resources:** Explicit measured extra storage/replay/fitting/training work; no remote GPU/API. No hidden default nonlinear critic or unbounded architecture search.

**Acceptance/next action:** A simple predictor saving more useful compute than it loses advances as an empirical extension. Constant-policy equivalence removes the critic. No useful direct feedback rejects the critic premise in this regime. A new scale failing to express the effect after validated sensitivity strengthens a negative finding; hypothetical future scale alone does not justify indefinite escalation.

#### WP11 — Synthesize and make the execution independently reviewable

**Objective/risk:** Deliver conclusions proportionate to evidence, including negative or resource-limited outcomes. **Dependencies/inputs/entry:** Any completed informative branch and all associated raw records; a positive confirmation is not required.

**Actions/files/commands:** Implement `report --manifest protocols/report_manifest.json --output reports/final_report.md` and `verify-artifacts --report reports/final_report.md`. Produce an executive verdict, novelty classification, method/evaluation specification, favorable/unfavorable results, absolute effects/uncertainty, full economic frontier, protocol deviations, failed runs, source/data/code notices and reproduction commands. Compare completed work to the original frontier ambition explicitly. Package code/config/locks and permissible aggregate outputs without corpus text or unverified third-party assets.

**Tests/outputs:** Every table regenerates from raw records; no protected-development mixing; units/parameter counts/token counts reconcile; all claims reference a run/protocol or primary source; file hashes and expected outputs are complete. **Resources:** CPU analysis and compact reporting; no new training unless it resolves a concrete missing validity requirement.

**Acceptance/next action:** Issue one of: replicated scoped advantage; single-study scoped advantage; useful simpler known method; sensitive negative result; methodologically invalid result requiring repair; or resource/sensitivity-limited inconclusion. Identify the evidence that would change the decision and the cheapest justified next experiment. Do not present unfinished required confirmation as a completed breakthrough.

### 9.4 Coding-agent authority and amendment policy

The coding agent may autonomously make routine reversible implementation choices, install compatible packages in the project environment, fetch the specified permitted public assets, choose memory-safe microbatches, optimize equivalent shared operations, fix bugs before unblinding, and execute the scientifically justified measured program. It need not ask for confirmation for each routine step. Preserve user/system access controls, source licenses and machine constraints.

Changes to hypotheses, primary metric/aggregation, evaluation population, preprocessing semantics, protected-data use, locality/feedback definition, baseline information or tuning fairness, checkpoint selection, or the interpretation of a claim require a **written methodological amendment** before dependent runs. Record whether protected outcomes were already observed and designate fresh evidence when needed. This is a scientific record, not a mandatory permission ceremony for every engineering fix. Actual destructive operations or new external commitments remain outside this plan's routine authority.

## 10. Adversarial interpretation, failure regimes, and continuation decisions

### 10.1 Strongest case for and against the direction

**The strongest case for it** is architectural conditioning plus a deliberately low-dimensional coordination signal. A group starts from an existing prediction, can recover original inputs, and chooses between a few real optimizer directions. Terminal forward scores are sufficient to observe those directions' immediate consequences. If a few useful demands persist across updates, the controller may spend less than a continuously learned/global credit mechanism while improving the representations consumed by its suffix. The complete method is small enough to compare with strong greedy objectives on one machine; even a negative result could identify whether interface, signal or economics is the real obstacle.

**The strongest case against it** is that the mechanism largely repackages known deep supervision, context supply and auxiliary optimization while adding expensive heads and state coordination. Immediate final loss may reward an early block for fitting a currently inadequate suffix rather than preparing a more capable future suffix. The protected path can let later groups ignore learned early features altogether. Shared readouts and Adam moments drift, so yesterday's winning objective may become harmful. Restricting the menu makes credit cheap but may exclude every useful direction. At the starting shape, the analytic training overhead is material and activation memory may not even be the laptop bottleneck. A stronger auxiliary head, better optimizer, checkpointed E2E training or occasional true gradient could dominate the result.

These objections are not answered by lower local loss, attractive generated examples, a high critic correlation, or a larger perturbation population. They are answered, if at all, by final likelihood and complete economic comparisons with the specified controls.

### 10.2 Diagnosing particular failures

| Observed pattern | Interpretation to test | Cheapest discriminating next action |
|---|---|---|
| Local CE falls but terminal NLL worsens | Misaligned directions or moving suffix, not proof of implementation failure | Validate gradients/transactions, compare immediate versus delayed realized utility and strongest static policy |
| `h` probe improves but final model does not | Probe decoder or redundant information explains the diagnostic | Terminal bypass/intervention and equally budgeted frozen probes; keep terminal NLL primary |
| Audit choices repeatedly select the same objective | A good constant may capture most value | Tune/prune that static objective; compare phase-only policy and full cost |
| Direct oracle menu helps but sparse reuse hurts | Rankings stale between audit events | Measure lagged ranking/regret and cost-optimal shorter period; do not add a critic before checking drift |
| One-step scoring rejects features useful after relearning | Frozen-suffix horizon is myopic | A small prespecified saved-state replay with suffix adaptation; charge replay and treat changed horizon as a new method |
| Training-audit gain fails on untouched articles | Controller overfits `Q` or exploits shortcuts | Increase/rotate only training-audit documents; chronological utility evaluation, duplicate/template controls, fresh confirmation |
| Raw path explains the complete gain | Improved information access is useful, specific prediction/credit claim is unsupported | Adopt `G_context` or a simpler static scaffold; report the engineering/empirical result accurately |
| Gain disappears after norm matching or LR tuning | Update scale or optimization explains it | Retain better tuned simple baseline; no contract-direction claim |
| Training is better per update but worse per FLOP/time | Auxiliary/audit work consumes more than it returns | Prune bank, use parity-tested gradient reuse, or prefer static/E2E economic frontier |
| Memory is lower only per graph/worker | Whole-laptop states and buffers were excluded | Reconcile RSS/driver/snapshot inventory and compare checkpointed E2E |
| Small real-text model cannot distinguish baselines | Inadequate expression or precision, possibly too easy a regime | Use diagnostic headroom and measured neighboring shape/context, or report limited sensitivity |
| Higher scale still shows a precise loss after valid baselines | Accumulating evidence against this regime's mechanism | Stop or adopt the demonstrated simpler method; do not invoke unspecified frontier magic |

### 10.3 Allowed conclusions

A defensible positive statement would name the corpus, tokenizer, `L/d/K/T`, complete training budget, selected strong comparators, absolute NLL improvement/interval, repeated seeds, memory/time/inference effects and replication status. A new interaction result additionally reports the directly estimated interface-by-policy effect. Call the work an empirical finding, an application of established control ideas, or an engineering optimization according to what the ablations establish.

A negative result can be equally decisive: a well-trained local baseline with measurable headroom, validated actions, adequate precision and a consistent failure of the audited interface at equal complete cost would argue against this specific approach locally. An unavailable kernel, a failed evaluator, an unexpressive toy task or an incomplete sample is a different outcome. Publish enough raw evidence to tell them apart.

The open long-term question remains whether a low-bandwidth vocabulary of **learned future computational demands** can coordinate modules at frontier scale. The proposed finite future-token menu does not answer that broad question. It tests one necessary practical premise: that a small set of locally executable directions can be selected using sparse downstream evidence in a way that beats competent alternatives after paying for the evidence.

## 11. Independent plan audits and revision record

Three separately tasked reviewers inspected the completed draft: one challenged its science and prior-art claims, one challenged experimental validity, and one walked through implementation and laptop feasibility. These were separate agent reviews of the document, not external peer review or execution of the proposed training system. The lead revised the plan, and each reviewer then reread the affected sections and downstream dependencies. A final targeted pass on 11 October 2026 produced no further substantive correction in the requested scope.

The common version inspected in the final three rechecks had SHA-256 `cbd1ee46d3b81befdd6f82c349c40f8bd09162cc26b18e842997f8b6f7767797`. This identifies the substantive plan before insertion of this audit record and final citation/navigation formatting; it is not the hash of the delivered file.

| Audit and material defect found | Revision incorporated into this plan | Targeted recheck and disposition |
|---|---|---|
| **Scientific / prior art:** additive predictions, protected context, utility feedback and provisional task updates have close antecedents; the proposal could be an assembly of known methods. | Sections 1–3 limit novelty to a falsifiable empirical application and interaction. Sections 5–6 include residual-equivalence, strong fixed objectives and HydaLearn-derived controls, with inspected-source limits in Section 13. | Scientific findings **SC01–SC06 closed**. No foundational novelty, frontier replacement or practical superiority is claimed. |
| **Scientific + experimental:** a single fixed auxiliary task is weaker than the candidate's two-horizon menu; an incomplete greedy auxiliary network would also weaken the comparison. | `P_static` includes a tuned fixed convex mixture of both horizons. `G_attn` includes the full extra Transformer plus residual FFN and readout at every early boundary; the first-boundary MTP adaptation and later heads are explicit. Parameter and arithmetic tables match that graph. | Baseline definitions, loss equations, work packages and cost tables rechecked together. Fixed-mixture and full-head defects closed. |
| **Scientific + experimental:** the lookahead control could waste an unused score, use a smaller action menu, or have undefined scaling and persistent weights. | Section 5.5 specifies source-style and safeguarded weighted-update variants, pure-objective probe scaling, stored weight vectors, a three-task extension and a same-horizon comparison. Source adaptations are labeled as such. | Baseline menu, ordinary-step policy, zero-lambda behavior and transaction tests rechecked. No invented published recipe or decision-unused scoring charge is imposed. |
| **Scientific + experimental:** selecting the best action on `Q` and reporting its gain on the same batch creates optimistic utility evidence. | Section 5.6 requires signed utility measurements on fresh independent `Q_check`, excluded from selection and fitting. Lag diagnostics use fresh proposals at the later model state. Critic admission cannot use selected-on-`Q` improvement alone. | Independent-utility, feature-access and cost boundaries rechecked in WP05/WP10. The mechanism remains contingent on measured useful and persistent decisions. |
| **Scientific + experimental:** a positive interface-by-policy interaction does not establish easier prediction; unequal suffix freezes or audit work can confound it. | H3 now concerns the incremental policy benefit of accumulation. The exact reset/accumulation quartet holds raw access, bank, ownership, freeze schedule and diagnostic proposal/scoring work fixed. Prediction and ranking lifetime require separate measurements. | Interaction estimand, four arms, matched work and permitted interpretation rechecked. A positive interaction alone cannot establish the proposed explanation. |
| **Experimental:** adaptive exploratory stopping invalidates a nominal fixed-sample variance-bound interpretation; millions of correlated token losses can create false precision. | Section 7 uses fresh fixed-size variance calibration before fixed confirmation, independent training triads, two predeclared primary comparisons, explicit development-to-confirmation population limits and separate crossed seed/article uncertainty. Protected outcomes cannot trigger sample extension. | Experimental findings **E1–E6 and M1–M5 closed** at the specification level. Required precision and the working distribution assumptions remain empirical gates. |
| **Implementation:** serial retention of the raw minimum could discard the CE-only state that the later numerical tie rule selects. A universal scored-state invariant also contradicted legitimate unscored lookahead. | Sections 5.4/5.7 use the same ordered tolerance rule for retention and selection. Discrete and safeguarded methods require scored-state parity; source-style lookahead requires parity with the exact constructed weighted update and records the combined score as unmeasured. | Near-tie retention, rollback, method-specific zero-lambda cases and WP04 acceptance requirements rechecked. Real optimizer parity tests are still required. |
| **Implementation:** an absent objective, a zero derivative and an omitted gradient have different optimizer effects; separate gradient calls could free a shared graph or understate work. | Sections 5.2–5.3 define separate horizon denominators, absent-objective skips, `None` versus numerical-zero treatment, per-parameter counters and persistent policy weights. Pseudocode explicitly requests joint CE actor/embedding and terminal-group/readout gradients; auxiliary embedding replay is charged. | Owner map, pseudocode, checkpoint state and arithmetic ledger rechecked. No cross-group derivative or omitted extra backward pass is assumed away. |
| **Implementation:** training windows, batch filling and initial calibration were insufficiently specified; full-model tests appeared before their implementation dependency. | Section 6.3 fixes window starts, EOS coverage, padding, positions, epoch permutations, whole-window batch minima, actual denominators and pending-batch state. Sections 8–9 specify a finite synthetic calibration exception, required installation/dependencies, and separate primitive WP01 from complete-model WP04 gates. | Implementation findings **IMP01–IMP12 closed**. Index arithmetic checked 24 evaluation cases and 24 training-window cases on Linux; these are not an executed data pipeline or M3 result. |
| **Experimental + implementation:** open-ended tuning, ambiguous cost scope, dirty editable source and incomplete recovery could change the comparison after the fact. | Sections 7–9 require finite registered tuning rounds and promotion/tie rules, an additive arithmetic-cost convention, complete source hashes and persistent control/sampler/admission state. Intrinsic training work, actual economic cost and genuinely external replay waste are separately recorded. | Tuning/configuration fields, accounting boundaries and replay requirements rechecked. Algorithmic failures cannot be relabeled to lower charged cost. |

The implementation reviewer independently recomputed the main parameter, optimizer-state and leading matrix/attention-cost tables and verified that the Section 12 Python block is byte-identical to the script actually executed during planning. Those checks support arithmetic and document consistency only. They do not validate an autograd implementation, actual model memory allocation or target-device performance.

**Remaining gates:** native M3 dependency and operation support; correct implemented gradient ownership, optimizer transactions, causal masks and recovery; locally verified assets and split construction; sustained complete-transaction memory/timing; informative baseline learning and independent action utility; and affordable protected confirmation. Each has an earlier work package and an explicit blocked, revise, inconclusive or stop outcome. A reviewer finding no further document-level blocker does not establish that any of these future gates will pass.

## 12. Mathematical checks actually executed during planning

These checks were run on Linux x86_64 with Python 3.12.14 and NumPy 2.3.5; PyTorch was unavailable. They use explicit small dual-number derivatives and finite differences. They verify narrow mathematical examples and cannot certify the proposed training graph, MPS compatibility, throughput, or model quality.

| Check | Observed result | What it establishes |
|---|---|---|
| Restricted residual/correction equivalence | Maximum logit difference `0`; CE difference `0`; maximum finite-difference gradient difference `0` | A named two-state description can be exactly the same residual computation in a restricted case |
| Conflicting local/final objectives at `a=0`, LR `0.2` | Local gradient `−0.5`, terminal `+0.5`, cosine `−1`; local CE `0.69314718 → 0.64439666`; terminal CE `0.69314718 → 0.74439666` | Locally successful descent can worsen the final additive prediction |
| Detach hidden only | Upstream gradient approximately `−0.45567053` remains | Detaching only `h` leaves a derivative path through predictive state |
| Detach hidden and prediction | Upstream gradient `0`; current-block gradient remains approximately `−0.09177712` | Both boundary values must be detached to get the intended local derivative |
| Disconnected versus connected hidden branch | Hidden-branch gradient `0` when disconnected; approximately `−0.33790084` when correction uses it | A side-channel representation needs an actual local training path |
| Spherical estimator, 20,000 directions/dimension | MSE/`||g||²`: `6.9329, 30.8526, 128.1508` at dimensions `8,32,128`; theoretical `7,31,127` | Finite sampling agrees with the elementary linear-case dimension-variance calculation |

All eight assertions passed. The exact executed script's SHA-256 was `16f2f13d173278dad32089a0518a7010f84191a68153faed0c4573e020aa01d2`. The code below is included so this file is self-contained. Save the block with its final newline as `mechanism_math_checks.py`; `python mechanism_math_checks.py` writes the full JSON result and prints it. Environment/hash fields change if the environment or script bytes change; floating-point results may differ slightly across NumPy builds. The spherical Monte Carlo values are diagnostic observations, not a statistical test of language-learning performance.

```python
"""Small CPU checks of proposed equations, NOT PyTorch/MPS implementation tests.

NumPy supplies linear algebra. Tiny forward-mode dual numbers explicitly model
the chain rule and detach semantics; this does not establish torch compatibility.
Run: python mechanism_math_checks.py
"""

from dataclasses import dataclass
import hashlib
import importlib.util
import json
import math
import platform
from pathlib import Path
import sys

import numpy as np


@dataclass(frozen=True)
class Dual:
    value: float
    grad: np.ndarray

    def _coerce(self, other):
        return other if isinstance(other, Dual) else Dual(float(other), np.zeros_like(self.grad))

    def __add__(self, other):
        other = self._coerce(other)
        return Dual(self.value + other.value, self.grad + other.grad)

    __radd__ = __add__

    def __mul__(self, other):
        other = self._coerce(other)
        return Dual(self.value * other.value, self.grad * other.value + other.grad * self.value)

    __rmul__ = __mul__

    def __neg__(self):
        return Dual(-self.value, -self.grad)

    def detach(self):
        return Dual(self.value, np.zeros_like(self.grad))

    def tanh(self):
        t = math.tanh(self.value)
        return Dual(t, (1.0 - t * t) * self.grad)

    def softplus(self):
        return Dual(float(np.logaddexp(0.0, self.value)), self.grad / (1.0 + math.exp(-self.value)))


def ce_correct_class_one(logit):
    return (-logit).softplus()


def equivalence_check(rng):
    h = rng.normal(size=(5, 3))
    a = rng.normal(size=(3, 3))
    w = rng.normal(size=(7, 3))
    y = np.array([0, 1, 2, 3, 4])

    def residual_logits(a):
        return (h + np.tanh(h @ a.T)) @ w.T

    def correction_logits(a):
        b = h.copy()
        c = np.tanh(h @ a.T)
        return (b + c) @ w.T

    def loss(logits):
        m = logits.max(axis=-1, keepdims=True)
        logz = m[:, 0] + np.log(np.exp(logits - m).sum(axis=-1))
        return float((logz - logits[np.arange(len(y)), y]).mean())

    eps = 1e-5
    grad_res, grad_corr = np.zeros_like(a), np.zeros_like(a)
    for i in range(3):
        for j in range(3):
            d = np.zeros_like(a)
            d[i, j] = eps
            grad_res[i, j] = (loss(residual_logits(a + d)) - loss(residual_logits(a - d))) / (2 * eps)
            grad_corr[i, j] = (loss(correction_logits(a + d)) - loss(correction_logits(a - d))) / (2 * eps)
    return {
        "max_abs_logit_difference": float(np.abs(residual_logits(a) - correction_logits(a)).max()),
        "absolute_ce_difference": abs(loss(residual_logits(a)) - loss(correction_logits(a))),
        "max_abs_finite_difference_gradient_difference": float(np.abs(grad_res - grad_corr).max()),
        "scope": "Exact restricted case: b_previous=h_previous, c equals the hidden residual, same head.",
    }


def conflict_check():
    a = Dual(0.0, np.ones(1))
    # Module 1 predicts b1=a and forwards h1=a. Frozen module 2 adds -2*h1.
    local = ce_correct_class_one(a)
    terminal = ce_correct_class_one(a + (-2.0 * a))
    step_size = 0.2
    after = a.value - step_size * local.grad[0]
    local_after = float(np.logaddexp(0.0, -after))
    terminal_after = float(np.logaddexp(0.0, after))
    return {
        "start_parameter": a.value,
        "local_gradient": float(local.grad[0]),
        "terminal_gradient": float(terminal.grad[0]),
        "gradient_cosine": float(local.grad[0] * terminal.grad[0] / abs(local.grad[0] * terminal.grad[0])),
        "learning_rate": step_size,
        "parameter_after_local_step": after,
        "local_loss_before": local.value,
        "local_loss_after": local_after,
        "terminal_loss_before": terminal.value,
        "terminal_loss_after": terminal_after,
        "scope": "A mathematical counterexample; not an empirical training result.",
    }


def detach_check(detach_prediction):
    theta1 = Dual(0.4, np.array([1.0, 0.0]))
    theta2 = Dual(0.6, np.array([0.0, 1.0]))
    h1 = (1.3 * theta1).tanh()
    b1 = 1.3 * theta1 + 0.3 * h1
    h2 = (0.7 * h1.detach()).tanh()
    c2 = theta2 * h2
    b2 = (b1.detach() if detach_prediction else b1) + c2
    loss = ce_correct_class_one(0.9 * b2)
    return {"loss": loss.value, "upstream_gradient": float(loss.grad[0]), "current_block_gradient": float(loss.grad[1])}


def dual_branch_check(connected):
    theta_h = Dual(0.4, np.array([1.0, 0.0]))
    theta_c = Dual(0.6, np.array([0.0, 1.0]))
    h = (1.3 * theta_h).tanh()
    c = h if connected else 1.3 * theta_c
    loss = ce_correct_class_one(0.2 + c)
    return {"loss": loss.value, "hidden_branch_gradient": float(loss.grad[0]), "separate_correction_gradient": float(loss.grad[1])}


def sphere_variance_check(rng):
    # Linear loss, ||g||=1, u uniform on unit sphere; ghat=d*(g dot u)*u.
    out = []
    n = 20000
    for d in (8, 32, 128):
        u = rng.normal(size=(n, d))
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        estimate = d * u[:, :1] * u
        estimate[:, 0] -= 1.0
        mse = float(np.mean(np.sum(estimate * estimate, axis=1)))
        out.append({"dimension": d, "directions": n, "measured_mse_over_gradient_squared_norm": mse, "theoretical_expectation": d - 1})
    return out


def main():
    rng = np.random.default_rng(20261011)
    result = {
        "status": "executed CPU mathematical checks, not a target hardware experiment",
        "environment": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "torch_available": importlib.util.find_spec("torch") is not None},
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "restricted_residual_equivalence": equivalence_check(rng),
        "local_terminal_descent_conflict": conflict_check(),
        "detach_hidden_only": detach_check(False),
        "detach_hidden_and_prediction": detach_check(True),
        "disconnected_hidden_output": dual_branch_check(False),
        "correction_equals_hidden_output": dual_branch_check(True),
        "random_direction_gradient_estimator_variance": sphere_variance_check(rng),
    }
    assert result["restricted_residual_equivalence"]["max_abs_logit_difference"] == 0.0
    assert result["restricted_residual_equivalence"]["max_abs_finite_difference_gradient_difference"] == 0.0
    assert result["local_terminal_descent_conflict"]["local_loss_after"] < result["local_terminal_descent_conflict"]["local_loss_before"]
    assert result["local_terminal_descent_conflict"]["terminal_loss_after"] > result["local_terminal_descent_conflict"]["terminal_loss_before"]
    assert result["detach_hidden_and_prediction"]["upstream_gradient"] == 0.0
    assert result["detach_hidden_only"]["upstream_gradient"] != 0.0
    assert result["disconnected_hidden_output"]["hidden_branch_gradient"] == 0.0
    assert result["correction_equals_hidden_output"]["hidden_branch_gradient"] != 0.0
    result["assertions"] = "all 8 passed"
    Path("mechanism_math_checks.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
```

## 13. Source ledger, research scope, and access limits

Research date: **11 October 2026**. The source records below distinguish bibliographic/abstract inspection, full method inspection, code reading and actual execution. A published claim remains an author's reported result unless independently checked here. No source is cited as proof of universal local-learning feasibility. Linked source versions/repository branches are inspection context; execution must pin the actual revisions reused.

### 13.1 Primary literature

#### R01 — Greedy local language-model pretraining

**Jihwan Moon, Sheir A. Zaheer, Jinmyoung Lee, Gunhee Kim, Chan Y. Park.** *Greedy Local Learning for Language Model Pretraining: Gaps and Objective Design*. arXiv v1, **4 October 2026**; CODEC-FM NeurIPS 2026 workshop acceptance stated in metadata. [Primary full text](https://arxiv.org/html/2610.04867v1).

**Inspection:** Full methods, main experiments, appendices B and F–H.

**Decision and limit:** Strong likelihood baseline: transformer auxiliaries and first-boundary two-token prediction. Published 125M/400M results use matched tokens and one synchronous seed; auxiliary compute is substantial and throughput unmeasured. Collapsed local loss does not alone prove information destruction.

#### R02 — DiffusionBlocks

**Makoto Shing, Masanori Koyama, Takuya Akiba.** *DiffusionBlocks: Block-wise Neural Network Training via Diffusion Interpretation*. **18 February 2026**, arXiv v3; ICLR 2026. [Primary full text](https://arxiv.org/html/2506.14202v3); [official proceedings](https://proceedings.iclr.cc/paper_files/paper/2026/hash/994c9bc68bc30954f6e5995864f1d92d-Abstract-Conference.html).

**Inspection:** Full method, autoregressive section and appendix E.4, official implementation README/license.

**Decision and limit:** Independently trained denoising blocks retain original inputs and refine target states, directly anticipating protected-context predictive correction. Language results report generated-text measures rather than ordinary held-out likelihood. Released Apache-2.0 code is image classification; a language or M3 implementation was not verified.

#### R03 — Local and overlapping Transformer training

**Michael Laskin, Luke Metz, Seth Nabarro, Mark Saroufim, Badreddine Noune, Carlo Luschi, Jascha Sohl-Dickstein, Pieter Abbeel.** *Parallel Training of Deep Networks with Local Updates*. First submitted **7 December 2020**; inspected arXiv v2, **15 June 2021**. [Primary full text](https://arxiv.org/html/2012.03837v2); [author project](https://mishalaskin.github.io/local/).

**Inspection:** Full language methods, results and compute appendix.

**Decision and limit:** Local, chunked and overlapping updates already train small Transformers. Their language compute comparison models maximal layer parallelism, and best backpropagation likelihood remains better. Small parameter count does not imply a small reproduction workload: the original data and hyperparameter sweep were substantial.

#### R04 — BoostResNet

**Furong Huang, Jordan Ash, John Langford, Robert Schapire.** *Learning Deep ResNet Blocks Sequentially using Boosting Theory*. **ICML 2018**, PMLR 80:2058–2067, conference 10–15 July. [Proceedings and paper](https://proceedings.mlr.press/v80/huang18b.html).

**Inspection:** Proceedings metadata and paper method/guarantees.

**Decision and limit:** Sequential residual blocks and telescoping feature boosting anticipate additive greedy corrections. Boosting guarantees require their weak-learning and optimization conditions; they do not imply monotonic held-out language loss or convergence for arbitrary simultaneously changing modules.

#### R05 — ResFGB

**Atsushi Nitanda, Taiji Suzuki.** *Functional Gradient Boosting based on Residual Network Perception*. **ICML 2018**, PMLR 80:3819–3828, conference 10–15 July. [Proceedings and paper](https://proceedings.mlr.press/v80/nitanda18a.html).

**Inspection:** Full predictor definition, algorithm and experimental setup.

**Decision and limit:** A linear readout of a residual feature map trained through functional boosting directly anticipates shared-readout additive representations. Classification evidence supplies no autoregressive scaling result. The linked Theano implementation is an algorithm reference; its license was not verified.

#### R06 — InfoPro

**Yulin Wang, Zanlin Ni, Shiji Song, Le Yang, Gao Huang.** *Revisiting Locally Supervised Learning: an Alternative to End-to-end Training*. First submitted **26 January 2021**; ICLR 2021. [Primary record](https://arxiv.org/abs/2101.10832).

**Inspection:** Primary abstract and metadata during this research.

**Decision and limit:** Information-preserving local supervision precedes PCN. Its auxiliary reconstruction and discrimination approach motivates protected-information controls. Vision experiments and information-theoretic motivation do not establish useful information retention or language-model likelihood parity for a new interface.

#### R07 — Context supply / ContSup

**Chengting Yu, Fengzhao Zhang, Hanzhi Ma, Aili Wang, Erping Li.** *Go beyond End-to-End Training: Boosting Greedy Local Learning with Context Supply*. First submitted **12 December 2023**; IEEE Transactions on Artificial Intelligence, **2025**. [Primary preprint](https://arxiv.org/abs/2312.07636); [publisher record](https://www.computer.org/csdl/journal/ai/2025/07/10838605/23thlA8bW2Q).

**Inspection:** Primary abstract, publication metadata and withdrawn OpenReview record.

**Decision and limit:** Explicit context supplied across gradient-isolated modules is a close protected-context antecedent. The evidence is vision-focused. Cite the preprint or journal publication; the inspected ICLR submission was withdrawn and must not be described as accepted ICLR 2024 work.

#### R08 — Dust

**Samip Dahal, Bishwas Mandal, Serdar Gülbahar, Akshay Vegesna; Q Labs.** *Dust: Pretraining Transformers Without Backpropagation*. **October 2026** research report; exact day not stated on inspected report. [Primary report](https://qlabs.sh/research/dust); [official code](https://github.com/qlabs-eng/dust).

**Inspection:** Full report methods, setup, results and caveats; repository README and MIT license.

**Decision and limit:** Token-level node perturbations, cached downstream forwards and local attention credit are direct audit-estimator antecedents. Fitted asymptotic results are extrapolations. Minimal released code requires Linux/NVIDIA bf16 and omits full-experiment optimizations; it does not demonstrate practical backpropagation replacement or laptop compatibility.

#### R09 — Decoupled neural interfaces / synthetic gradients

**Max Jaderberg, Wojciech Marian Czarnecki, Simon Osindero, Oriol Vinyals, Alex Graves, David Silver, Koray Kavukcuoglu.** *Decoupled Neural Interfaces using Synthetic Gradients*. **ICML 2017**, PMLR 70:1627–1635, conference 6–11 August. [Proceedings and paper](https://proceedings.mlr.press/v70/jaderberg17a.html).

**Inspection:** Primary abstract and metadata during this research.

**Decision and limit:** Locally predicted downstream gradients already decouple module updates. A learned utility critic needs an explicit difference in target, supervision, update geometry or sample efficiency. Synthetic-gradient prediction does not guarantee correct directions for arbitrary states or eliminate the need for feedback supervision.

#### R10 — Perturbation-trained synthetic feedback

**Benjamin James Lansdell, Prashanth Ravi Prakash, Konrad Paul Kording.** *Learning to solve the credit assignment problem*. First submitted **3 June 2019**; inspected v4, **22 April 2020**; ICLR 2020. [Primary full text](https://arxiv.org/html/1906.00889v4).

**Inspection:** Full feedback estimator, theory assumptions and experiments.

**Decision and limit:** Node-perturbation rewards train synthetic feedback subsequently used for deterministic weight updates. This anticipates learning useful local credit from counterfactual forward evaluations. The guarantees cover restricted network classes; small vision examples do not establish deep language-model efficiency.

#### R11 — Local Critic Training

**Hojung Lee, Cho-Jui Hsieh, Jong-Seok Lee.** *Local Critic Training for Model-Parallel Learning of Deep Neural Networks*. First submitted **3 February 2021**; preliminary version appeared at IJCNN 2019. [Primary full text](https://arxiv.org/html/2102.01963v1).

**Inspection:** Full critic method and convergence assumptions.

**Decision and limit:** Local critics learn loss proxies and supply block derivatives, preceding a scalar downstream-utility critic. The convergence argument assumes adequate descent alignment and regularity; fitting scalar values accurately does not establish derivative accuracy or safe module updates.

#### R12 — HydaLearn

**Sam Verboven, Muhammad Hafeez Chaudhary, Jeroen Berrevoets, Wouter Verbeke.** *HydaLearn: Highly Dynamic Task Weighting for Multi-task Learning with Auxiliary Tasks*. First submitted **26 August 2020**. [Primary paper](https://arxiv.org/pdf/2008.11643).

**Inspection:** Full fake-update algorithm, metric scoring and ablations, especially pages 2–3.

**Decision and limit:** Hypothetical task updates followed by forward scoring on the main metric already determine auxiliary weights. This is the closest shadow-update controller antecedent. Published two-task/tabular studies do not establish usefulness for gradient-isolated language modules or the proposed compute budget.

#### R13 — AutoSeM

**Han Guo, Ramakanth Pasunuru, Mohit Bansal.** *AutoSeM: Automatic Task Selection and Mixing in Multi-Task Learning*. **NAACL-HLT 2019**. [ACL record and paper](https://aclanthology.org/N19-1355/).

**Inspection:** Full task-selection algorithm and experimental setup.

**Decision and limit:** Thompson sampling chooses auxiliary tasks using primary validation improvement; posterior decay accommodates changing usefulness. Adaptive contract selection is consequently not new at the decision-rule level. Shared-representation NLP tasks differ from boundary-local contracts; direct and equally audited controls remain necessary.

#### R14 — Decoupled Greedy Learning

**Eugene Belilovsky, Michael Eickenberg, Edouard Oyallon.** *Decoupled Greedy Learning of CNNs*. **ICML 2020**, PMLR 119:736–745, conference 13–18 July. [Proceedings and paper](https://proceedings.mlr.press/v119/belilovsky20a.html).

**Inspection:** Primary abstract and metadata during this research.

**Decision and limit:** Local greedy objectives, asynchronous modules and replay for delayed communication are established. Their experiments concern CNN image classification; parallel scheduling benefits do not establish faster serial training or autoregressive likelihood performance on a unified-memory laptop.

#### R15 — AugLocal

**Chenxiang Ma, Jibin Wu, Chenyang Si, Kay Chen Tan.** *Scaling Supervised Local Learning with Augmented Auxiliary Networks*. First submitted **27 February 2024**; ICLR 2024. [Primary record](https://arxiv.org/abs/2402.17318); [proceedings](https://proceedings.iclr.cc/paper_files/paper/2024/hash/4aed772da3e13359740f345c809c2409-Abstract-Conference.html).

**Inspection:** Primary abstract and official metadata during this research.

**Decision and limit:** Auxiliary networks sampled from downstream layers address short-sighted local objectives, with auxiliary depth decreasing through the network. Richer auxiliary contracts must compete with richer auxiliary networks. Vision performance does not remove their additional training compute or establish language-model performance.

#### R16 — Periodic global guidance

**Hasnain Irshad Bhatti, Jaekyun Moon.** *Locally Supervised Learning with Periodic Global Guidance*. First submitted **1 August 2022**; ICML 2022 HAET workshop acceptance stated in metadata. [Primary record](https://arxiv.org/abs/2208.00821).

**Inspection:** Primary abstract and metadata.

**Decision and limit:** Periodic global guidance for local training predates sparse PCN correction. Any occasional global-gradient arm is a hybrid and needs its own accounting. This antecedent does not establish that rare forward-only audits have the same information or effectiveness as global gradient guidance.

#### R17 — GrAPE

**Paul Caillon, Erwan Fagnou, Blaise Delattre, Alexandre Allauzen.** *Scaling Direct Feedback Learning with Jacobian Alignment Guarantees*. **ICLR 2026**. [Proceedings](https://proceedings.iclr.cc/paper_files/paper/2026/hash/2b1c608556ce011d3a3381fe0c527136-Abstract-Conference.html); [paper](https://proceedings.iclr.cc/paper_files/paper/2026/file/2b1c608556ce011d3a3381fe0c527136-Paper-Conference.pdf).

**Inspection:** Full estimator, calibration, Transformer results and timing caveats.

**Decision and limit:** Learned feedback aligned through rank-one forward-JVP estimates plus rare backpropagation calibration anticipates sparse feedback correction. Transformer protocols retain backpropagation inside attention. Published theoretical critical-path gains and a CUDA timing prototype do not establish M3 speedups or fully backpropagation-free training.

#### R18 — Forward gradients with local losses

**Mengye Ren, Simon Kornblith, Renjie Liao, Geoffrey Hinton.** *Scaling Forward Gradient With Local Losses*. **ICLR 2023**. [Author-hosted primary paper and code links](https://mengyeren.com/research/2023/scaling-forward-gradient-with-local-losses/).

**Inspection:** Full method, theory and related work on author-hosted page.

**Decision and limit:** Activation perturbations and local losses reduce forward-gradient variance, motivating representation-level credit rather than full-weight search. Forward automatic differentiation is distinct from zeroth-order finite differences. Vision results do not establish competitive deep language-model pretraining or performance on the proposed backend.

#### R19 — Auto-Lambda

**Shikun Liu, Stephen James, Andrew J. Davison, Edward Johns.** *Auto-Lambda: Disentangling Dynamic Task Relationships*. First submitted **7 February 2022**; inspected arXiv metadata v2, **2 June 2022**; TMLR 2022. [Primary record](https://arxiv.org/abs/2202.03091).

**Inspection:** Primary abstract and metadata during this research; the method discussion was additionally inspected during this research.

**Decision and limit:** Main validation loss already meta-learns changing auxiliary-task weights. A forward-audited controller may differ in information access and cost, but adaptivity itself is established. Gradient-based bilevel optimization is not a strict-local forward-only baseline and requires separate compute accounting.

#### R20 — AuxiLearn

**Aviv Navon, Idan Achituve, Haggai Maron, Gal Chechik, Ethan Fetaya.** *Auxiliary Learning by Implicit Differentiation*. First submitted **22 June 2020** according to arXiv; inspected metadata v3, **11 May 2021**; ICLR 2021. [Primary record](https://arxiv.org/abs/2007.02693).

**Inspection:** Primary abstract and metadata during this research.

**Decision and limit:** Learning nonlinear combinations of auxiliary losses and generating auxiliary tasks both predate PCN. The proposed novelty must concern its constrained interface and measured advantage, not automatic objective design generally. Implicit differentiation introduces a distinct credit mechanism and computational burden.

#### R21 — Feature reconstruction for Transformers

**Priyank Pathak, Jingwei Zhang, Dimitris Samaras.** *Local Learning on Transformers via Feature Reconstruction*. First submitted **29 December 2022**. [Primary record](https://arxiv.org/abs/2212.14215).

**Inspection:** Primary abstract and metadata during this research.

**Decision and limit:** Reconstructing incoming features supplies a direct cheap information-preservation baseline. Its Transformer demonstrations are vision-focused, including Swin-Tiny; the broad title does not establish an autoregressive language result or make feature reconstruction a new contract mechanism.

#### R22 — Cumulative-goodness diagnostic failure

**Amirhossein Yousefiramandi.** *Cumulative-Goodness Free-Riding in Forward-Forward Networks: Real, Repairable, but Not Accuracy-Dominant*. First submitted **7 May 2026**; v2 **28 September 2026**; NeurIPS 2026 poster acceptance stated in metadata. [Primary record](https://arxiv.org/abs/2605.06240).

**Inspection:** Primary abstract and metadata.

**Decision and limit:** Repairing deeper goodness separation need not improve final accuracy. This supports separating representation-health diagnostics from task success. Forward-Forward vision findings do not directly establish an equivalent failure in PCN language training; that transfer is a testable inference.

#### R23 — Deep predictive coding / DeepPC

**Chang Qi, Matteo Forasassi, Thomas Lukasiewicz, Tommaso Salvatori.** *Towards the Training of Deeper Predictive Coding Neural Networks*. **ICML 2026**, PMLR 306:100334–100362, conference 6–11 July. [Proceedings and paper](https://proceedings.mlr.press/v306/qi26f.html); [official code](https://github.com/changqi97/DeepPCNs).

**Inspection:** Primary abstract and official repository page.

**Decision and limit:** Balanced errors, precision-weighted relaxation and auxiliary residual neurons improve deeper predictive-coding networks. This is relevant credit-assignment research but uses iterative state inference, unlike a single-pass predictive contract. The inspected evidence is vision-focused; repository licensing was not established.

#### R24 — Titans and MIRAS

**Ali Behrouz, Meisam Razaviyayn, Vahab Mirrokni; Google Research.** *Titans + MIRAS: Helping AI have long-term memory*. Official research overview, **4 December 2025**, linking both primary papers. [Official overview](https://research.google/blog/titans-miras-helping-ai-have-long-term-memory/).

**Inspection:** Official overview, including memory mechanism and linked-paper identities; not full-paper experimental or training-graph audit.

**Decision and limit:** Test-time neural memory and associative-memory frameworks provide interface ideas. Online memory adaptation and replacing global pretraining credit are different claims. This overview does not establish that outer language-model training is gradient-isolated or that the reported gains transfer to PCN.

#### R25 — Nested Learning and Hope

**Ali Behrouz, Vahab Mirrokni; Google Research.** *Introducing Nested Learning: A new ML paradigm for continual learning*. Official overview, **7 November 2025**, linking NeurIPS 2025 paper *Nested Learning: The Illusion of Deep Learning Architectures*. [Official overview](https://research.google/blog/introducing-nested-learning-a-new-ml-paradigm-for-continual-learning/).

**Inspection:** Official overview and its Hope/continuum-memory description; not full-paper training-graph audit. The overview credits underlying research to Ali Behrouz, Meisam Razaviyayn, Peilin Zhong and Vahab Mirrokni.

**Decision and limit:** Multiple update timescales and nested memory objectives are useful conceptual precedents. Hope's reported language and memory improvements do not demonstrate a replacement for end-to-end pretraining credit. An official overview is insufficient to transfer exact algorithmic or compute claims.

#### R26 — LoPT

**Hengyu Shi, Tianyang Han, Peizhe Wang, Zhiling Wang, Xu Yang, Junhao Su.** *Rethinking Local Learning: A Cheaper and Faster Recipe for LLM Post-Training*. First submitted **6 May 2026**; inspected metadata v4, **8 June 2026**. [Primary record](https://arxiv.org/abs/2605.04913).

**Inspection:** Primary abstract and metadata.

**Decision and limit:** A midpoint gradient boundary with lower feature reconstruction and upper task learning shows practical local LLM post-training relevance. It starts from pretrained models; preserving pretrained features does not establish from-scratch local pretraining capability or economic scaling.

#### R27 — Local self-supervised learning

**Wu S. Zihan, Ariane Delrocq, Wulfram Gerstner, Guillaume Bellec.** *Can local learning match self-supervised backpropagation?* **ICML 2026**, PMLR 306:168340–168361, conference 6–11 July. [Proceedings](https://proceedings.mlr.press/v306/zihan26a.html); [official code](https://github.com/zihan-wu/local-SSL).

**Inspection:** Primary abstract and official README/MIT license.

**Decision and limit:** Conditional exact-update results for linear networks and nonlinear local SSL vision comparisons show that local objectives can be competitive in defined settings. They do not establish arbitrary local-rule equivalence or autoregressive pretraining parity. Released code targets vision and an older PyTorch environment.

### 13.2 Hardware, software, data, and statistical sources

The following official pages/repositories were inspected on the research date. Moving documentation and repository branches do not establish a tested dependency pin. Organization, observed version/date context, decision and inspection limit are given for each record.

| ID | Source, organization, date/version and primary URL | Inspection and decision; limit |
|---|---|---|
| **H01** | PyTorch contributors, *MPS backend*, observed documentation 2.14, page dated 11 May 2026: [MPS notes](https://docs.pytorch.org/docs/2.14/notes/mps.html) | Availability/OS guidance read. Establishes the discovery path, not exact target-wheel compatibility. |
| **H02** | PyTorch contributors, observed 2.14: [MPS environment variables](https://docs.pytorch.org/docs/2.14/mps_environment_variables.html) | Fallback, fast-math and allocator settings read; motivates surfaced operation failures and retained safeguards. |
| **H03** | PyTorch contributors, inspected moving source and 2.14 docs: [MPS operation utilities](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/mps/OperationUtils.mm), [AMP](https://docs.pytorch.org/docs/2.14/amp.html) | Dtype mapping/Float64 rejection and autocast guidance read. Dtype recognition is not complete operator/backward coverage. |
| **H04** | PyTorch contributors, observed 2.14: [scaled dot-product attention](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html) | Mask/dropout and dispatch semantics read. No fused MPS backward or cached-decode correctness inferred. |
| **H05** | PyTorch contributors, observed 2.14: [activation checkpointing](https://docs.pytorch.org/docs/2.14/checkpoint.html) | Reentrant/non-reentrant and detached-tensor constraints read; exact graph still requires parity tests. |
| **H06** | PyTorch contributors, observed 2.14: [MPS API](https://docs.pytorch.org/docs/2.14/mps.html), [current allocation](https://docs.pytorch.org/docs/2.14/generated/torch.mps.current_allocated_memory.html), [driver allocation](https://docs.pytorch.org/docs/2.14/generated/torch.mps.driver_allocated_memory.html), [recommended memory](https://docs.pytorch.org/docs/2.14/generated/torch.mps.recommended_max_memory.html) | Synchronization/RNG/memory scopes inspected. Current readings are not historical true peaks; whole-process overlap must be considered. |
| **H07** | PyTorch contributors, observed 2.14: [MPS profiler](https://docs.pytorch.org/docs/2.14/generated/torch.mps.profiler.profile.html) | Signpost profiling guidance read; optional diagnosis, not a requirement for privileged instrumentation. |
| **H08** | Apple MLX contributors, observed docs 0.32.3: [installation](https://ml-explore.github.io/mlx/build/html/install.html), [data types](https://ml-explore.github.io/mlx/build/html/python/data_types.html) | Native environment/dtype guidance inspected; no target install performed. |
| **H09** | Apple MLX contributors, observed 0.32.3: [value and gradient](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.value_and_grad.html), [stop-gradient](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.stop_gradient.html), [checkpoint](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.checkpoint.html) | Differentiation and recomputation interfaces read. Proposed ownership implementation remains untested. |
| **H10** | Apple MLX contributors, observed 0.32.3: [precision](https://ml-explore.github.io/mlx/build/html/usage/precision.html) | Reduced internal FP32 precision and its control inspected; exact numerical effects must be measured locally. |
| **H11** | Apple MLX contributors, observed 0.32.3/moving source: [synchronize](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.synchronize.html), [active memory](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.get_active_memory.html), [peak memory](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.get_peak_memory.html), [fast SDPA](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.scaled_dot_product_attention.html), [fast-operation source](https://github.com/ml-explore/mlx/blob/main/mlx/fast.cpp), [compilation/state guide](https://github.com/ml-explore/mlx/blob/main/docs/src/usage/compile.rst) | Lazy evaluation, counters and visible gradient/VJP fallback branches inspected; source display was partial. No universal fused-backward or M3 throughput claim. |
| **H12** | PyTorch/MLX contributors, current inspected docs: [PyTorch reproducibility](https://docs.pytorch.org/docs/2.14/notes/randomness.html), [MLX saving/loading](https://ml-explore.github.io/mlx/build/html/usage/saving_and_loading.html) | Backend reproducibility/serialization limits read. Complete controller/optimizer/sampler recovery is a project responsibility. |
| **B01** | Andrej Karpathy and contributors, moving repository inspected 11 October 2026: [nanoGPT](https://github.com/karpathy/nanoGPT), [model source](https://github.com/karpathy/nanoGPT/blob/master/model.py), [MIT license](https://github.com/karpathy/nanoGPT/blob/master/LICENSE) | README/deprecation and core implementation inspected. Useful readable reference, not a sufficient strongest-current baseline recipe. |
| **B02** | Andrej Karpathy and contributors, moving repository inspected 11 October 2026: [nanochat](https://github.com/karpathy/nanochat), [CPU/MPS run path](https://github.com/karpathy/nanochat/blob/master/runs/runcpu.sh), [optimizer source](https://github.com/karpathy/nanochat/blob/master/nanochat/optim.py) | README and relevant path/dtype handling read. Author caveats retained; full code/license compatibility must be verified at the pinned revision before copying. |
| **B03** | PyTorch contributors, observed documentation 2.14: [Muon optimizer](https://docs.pytorch.org/docs/2.14/generated/torch.optim.Muon.html) | Eligible hidden-matrix roles and other-role optimizer guidance inspected. Target MPS and shadow-state parity remain a gate. |
| **D01** | Salesforce/Hugging Face dataset maintainers, moving card inspected 11 October 2026: [WikiText dataset card](https://huggingface.co/datasets/Salesforce/wikitext/blob/main/README.md) | Split/configuration and conflicting license declarations read. Preserve the exact card at the execution revision; no corpus downloaded here. |
| **D02** | Salesforce dataset repository, exact revision `b08601e04326c79dfdd32d625aee71d232d685c3`: [commit](https://huggingface.co/datasets/Salesforce/wikitext/commit/b08601e04326c79dfdd32d625aee71d232d685c3), [raw-103 files](https://huggingface.co/datasets/Salesforce/wikitext/tree/b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-103-raw-v1) | File sizes and LFS hashes read. Local byte checks remain required; published row counts are not article counts. |
| **D03** | Stephen Merity, original WikiText release description (2016), surviving author page inspected 11 October 2026: [WikiText author description](https://state.smerity.com/direct/smerity/state/01HRTBE0WN2QV8PRC6JDH22DGW) | Raw/word distinction, article counts and share-alike statement inspected. The older smerity.com URL failed; the license-version conflict remains unresolved. |
| **D05** | Salesforce repository, same immutable revision: [raw-2 files](https://huggingface.co/datasets/Salesforce/wikitext/tree/b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-2-raw-v1), [raw-103 files](https://huggingface.co/datasets/Salesforce/wikitext/tree/b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-103-raw-v1) | Matching held-out file hashes inspected. WT-2 is not an independent held-out confirmation after WT-103 evaluation. |
| **D04** | Hugging Face Hub contributors, current documentation: [download guide](https://huggingface.co/docs/huggingface_hub/en/guides/download) | Exact-file/revision download API and local directory behavior read. The plan's downloader/checksum wrapper remains to be implemented. |
| **D06** | OpenWebText maintainers, card inspected 11 October 2026: [OpenWebText dataset card](https://huggingface.co/datasets/Skylion007/openwebtext/blob/main/README.md) | Data scale/split and packaging-rights disclaimer read. Full download excluded from the initial workload. |
| **D07** | Hugging Face FineWeb team, card inspected 11 October 2026: [FineWeb-Edu card](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/blob/main/README.md) | Web provenance, curation and sample sizes inspected. Any later subset needs a new immutable shard/population manifest. |
| **D08** | Hugging Face Tokenizers contributors, current releases/API guidance: [releases](https://github.com/huggingface/tokenizers/releases), [quick tour](https://huggingface.co/docs/tokenizers/en/quicktour), [trainer API](https://huggingface.co/docs/tokenizers/api/trainers), [pre-tokenizer API](https://huggingface.co/docs/tokenizers/api/pre-tokenizers) | Byte/BPE guidance inspected; some current/versioned documentation routes were incomplete or failed. Pin an actually compatible stable wheel and validate its API rather than inventing a version. |
| **D09** | Eric Zhu and datasketch contributors, observed release 2.0.0, 5 July 2026: [official API](https://ekzhu.com/datasketch/documentation.html), [MinHash guide](https://ekzhu.com/datasketch/minhash.html), [MIT license](https://github.com/ekzhu/datasketch/blob/master/LICENSE), [release/wheel record](https://pypi.org/project/datasketch/) | Explicit CPU mode, permutation scheme/batching, fixed LSH banding and license inspected. A universal Python wheel is listed; no local install or performance probe was performed here. |
| **S01** | Xavier Bouthillier et al., *Accounting for Variance in Machine Learning Benchmarks*, MLSys 2021: [proceedings](https://proceedings.mlsys.org/paper_files/paper/2021/hash/0184b0cd3cfb185989f858a1d9f5c1eb-Abstract.html), [paper](https://proceedings.mlsys.org/paper_files/paper/2021/file/0184b0cd3cfb185989f858a1d9f5c1eb-Paper.pdf) | Training/data/search variance discussion inspected. Motivates a clearly conditional frozen-pipeline claim; does not set this project's sample size. |
| **S02** | Art B. Owen, *The pigeonhole bootstrap*, 2007: [primary preprint](https://arxiv.org/abs/0712.1111) | Crossed row/column resampling method inspected. Approximate secondary inference, not a universally exact small-sample procedure. |
| **S03** | Art B. Owen and Dean Eckles, *Bootstrapping data arrays of arbitrary order*, Annals of Applied Statistics 6(3), 895–927 (2012), [DOI](https://doi.org/10.1214/12-AOAS547), [preprint](https://arxiv.org/abs/1106.2125), [author implementation](https://github.com/deaneckles/multiway_bootstrap) | Product-factor bootstrap conditions inspected. Does not guarantee representative articles or compensate for too few independent training runs. |

Core package licenses were inspected for [PyTorch](https://github.com/pytorch/pytorch/blob/main/LICENSE), [MLX](https://github.com/ml-explore/mlx/blob/main/LICENSE) and [Tokenizers](https://github.com/huggingface/tokenizers/blob/main/LICENSE). Record the exact installed/copy-source licenses and notices at execution; upstream code licensing and source-text rights are separate. No paid API, pretrained weights or remote accelerator is a prerequisite.

### 13.3 Search coverage, patents, deployment and stopping rationale

| Search pass | What was checked | Material effect on the plan |
|---|---|---|
| Named-source verification | Exact titles, arXiv identifiers, proceedings, official reports and author repositories; October 2026 and ICML/ICLR 2026 claims checked against primary pages. | Resolved Greedy LM access, separated accepted workshop/journal records from withdrawn submissions, and prevented treating generative perplexity as likelihood. |
| Architecture ancestry | Greedy local language training, residual/feature boosting, original-input/context supply, predictive denoising, local reconstruction; followed paper references and related primary work. | BoostResNet/ResFGB remove additive refinement as an independent novelty claim; context supply and DiffusionBlocks substantially anticipate the interface. |
| Credit ancestry | Synthetic gradients, local critics, perturbation-trained feedback, forward gradients and periodic global guidance; checked equations or primary methods for nearest approaches. | Counterfactual feedback learning and sparse calibration are established families. The critic must beat direct audit use under equal information and cost. |
| Controller ancestry | Adaptive auxiliary weighting/task selection, fake update scoring, validation-guided and implicit-differentiation methods. | HydaLearn and AutoSeM invalidate broad claims of novel shadow-update selection or adaptive auxiliary objectives. |
| Final targeted follow-ups | Exact metadata, code scope/license, and closely related residual-boosting developments. | Added constraints and implementation caveats without exposing a materially different closest mechanism after the controller antecedents were found. |

**Why stop here:** The closest mechanisms now supply concrete reduction tests and executable competing controls. Further targeted searches returned members of already represented families or refinements that did not change the core novelty assessment. This is practical saturation for designing this PoC, not proof of complete literature coverage, nor a predetermined paper quota.

**Blind spots and remaining bounds:** The search emphasized English-language, publicly indexed papers, proceedings, author pages and repositories accessible on 11 October 2026. It did not audit every proof, supplementary file, code revision, historical implementation or training run. Public repository presence does not establish full experimental reproducibility; license claims apply only to repositories actually inspected and require rechecking at the pinned revision used for reuse. Some full text or code may be unavailable or unindexed; source inspection is not independent replication.

**Patents and deployed systems:** A subsequent narrow patent/deployment check is recorded in the addendum below. It found direct synthetic-gradient and adaptive-auxiliary patent disclosures, plus deployed auxiliary-prediction training in DeepSeek-V3. No legal-status, freedom-to-operate, patentability or exhaustive family analysis was performed. Proprietary and unpublished systems remain a coverage gap. No inspected primary source establishes production frontier-LM pretraining with the proposed strict-local regime; this scoped absence is not proof no such system exists.

**Boundary on evidence:** Small-model, vision, post-training, test-time adaptation, theoretical-linear and generated-text results are retained as precedents with explicit conditions. They are not collectively treated as evidence that a new language-pretraining system will reach frontier quality or improve end-to-end economic efficiency.

#### Public technical disclosures checked on 11 October 2026

##### PAT01 — Synthetic-gradient training disclosure

**Title:** *Training neural networks using synthetic gradients*. **Inventors:** Oriol Vinyals, Alexander Benjamin Graves, Wojciech Czarnecki, Koray Kavukcuoglu, Simon Osindero, Maxwell Elliot Jaderberg. **Dates:** claimed priority 20 May 2016; international publication **WO2017201506A1, 23 November 2017**; related US publication **US11715009B2, 1 August 2023**. The inspected WO record lists Google LLC as original assignee; the US record lists DeepMind Technologies Ltd.

**Primary documents:** [WO publication text](https://patents.google.com/patent/WO2017201506A1/en); [US publication text](https://patents.google.com/patent/US11715009B2/en); [US issued-document PDF](https://patentimages.storage.googleapis.com/9b/f0/f9/756ee6f4d2556f/US11715009.pdf). **Inspected:** bibliographic records, abstract/description, US claims 1–12 in reproduced publication text; no legal-status audit.

**Technical consequence:** Claim 1 describes activation-conditioned gradient prediction, subnetwork updates and predictor training against a backpropagated target; claims 3–4 and 9–10 cover synthetic downstream targets, local backpropagation and asynchronous/distributed updates. This corroborates DNI ancestry without establishing any relation between a proposed implementation and enforceable claim scope.

##### PAT02 — Validation-guided auxiliary task/data scheduling

**Title:** *Method and apparatus for auxiliary learning with joint task and data scheduling*. **Inventors:** Xin Wang, Wenwu Zhu, Yijian Qin, Ze Cheng. **Listed applicants/original assignees:** Tsinghua University and Robert Bosch GmbH. **Dates:** filed 1 July 2022; **WO2024000566A1, published 4 January 2024**; corrected publication A9 dated 14 March 2024 is linked but was not compared.

**Primary documents:** [Publication text](https://patents.google.com/patent/WO2024000566A1/en); [published-document PDF](https://patentimages.storage.googleapis.com/83/31/52/79130ddb1030d2/WO2024000566A1.pdf). **Inspected:** bibliographic record, abstract and claims 1, 4–15 in reproduced publication text.

**Technical consequence:** Claims describe weighting auxiliary task/sample contributions, alternating model/controller updates, target-only validation feedback and implicit gradients. This adds a public patent disclosure of adaptive auxiliaries. It does not establish the exact forward-only shadow-update controller; HydaLearn remains the closer inspected algorithmic precedent.

##### DEP01 — Deployed auxiliary-prediction training: DeepSeek-V3

**Organization:** DeepSeek-AI. **Sources/date:** [official introduction, 26 December 2024](https://deepseek.com/en/news/deepseek-v3/); [DeepSeek-V3 Technical Report, December 2024 v1](https://arxiv.org/html/2412.19437v1); [official repository](https://github.com/deepseek-ai/DeepSeek-V3). **Inspected:** release notice with API availability, repository model/training summary and public inference offering, report sections 2.2 and 3.2 describing MTP and the backward training pipeline.

**Technical consequence:** Sequential multi-token modules share embeddings/readout, add future-token training losses and can be removed for inference. The published model was offered through an API. HAI-LLM/DualPipe retains backward computation: this supports deployed auxiliary prediction, not strict-local pretraining or a deployed PCN critic/controller.

##### Scope and no-result statements

Searches covered exact synthetic-gradient/DNI patent phrases, local-critic training phrases, auxiliary-task validation and fake-update terms, followed by the closest patent publications and official deployment documentation. Generic local-critic results mainly described actor–critic control and were not treated as module-critic precedents. No consequential patent result for the exact forward-only shadow-update mechanism, or verified production deployment of strict-local LLM pretraining, was established in this narrow pass. The patent disclosures sharpen the provenance record but do not require different core experiments beyond the academic baselines already identified. This is a technical disclosure check, not an exhaustive patent or deployment search.

[R01]: https://arxiv.org/html/2610.04867v1
[R02]: https://arxiv.org/html/2506.14202v3
[R03]: https://arxiv.org/html/2012.03837v2
[R04]: https://proceedings.mlr.press/v80/huang18b.html
[R05]: https://proceedings.mlr.press/v80/nitanda18a.html
[R06]: https://arxiv.org/abs/2101.10832
[R07]: https://arxiv.org/abs/2312.07636
[R08]: https://qlabs.sh/research/dust
[R09]: https://proceedings.mlr.press/v70/jaderberg17a.html
[R10]: https://arxiv.org/html/1906.00889v4
[R11]: https://arxiv.org/html/2102.01963v1
[R12]: https://arxiv.org/pdf/2008.11643
[R13]: https://aclanthology.org/N19-1355/
[R14]: https://proceedings.mlr.press/v119/belilovsky20a.html
[R15]: https://arxiv.org/abs/2402.17318
[R16]: https://arxiv.org/abs/2208.00821
[R17]: https://proceedings.iclr.cc/paper_files/paper/2026/hash/2b1c608556ce011d3a3381fe0c527136-Abstract-Conference.html
[R18]: https://mengyeren.com/research/2023/scaling-forward-gradient-with-local-losses/
[R19]: https://arxiv.org/abs/2202.03091
[R20]: https://arxiv.org/abs/2007.02693
[R21]: https://arxiv.org/abs/2212.14215
[R22]: https://arxiv.org/abs/2605.06240
[R23]: https://proceedings.mlr.press/v306/qi26f.html
[R24]: https://research.google/blog/titans-miras-helping-ai-have-long-term-memory/
[R25]: https://research.google/blog/introducing-nested-learning-a-new-ml-paradigm-for-continual-learning/
[R26]: https://arxiv.org/abs/2605.04913
[R27]: https://proceedings.mlr.press/v306/zihan26a.html
[H01]: https://docs.pytorch.org/docs/2.14/notes/mps.html
[H02]: https://docs.pytorch.org/docs/2.14/mps_environment_variables.html
[H03]: https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/mps/OperationUtils.mm
[H04]: https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html
[H05]: https://docs.pytorch.org/docs/2.14/checkpoint.html
[H06]: https://docs.pytorch.org/docs/2.14/mps.html
[H07]: https://docs.pytorch.org/docs/2.14/generated/torch.mps.profiler.profile.html
[H08]: https://ml-explore.github.io/mlx/build/html/install.html
[H09]: https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.value_and_grad.html
[H10]: https://ml-explore.github.io/mlx/build/html/usage/precision.html
[H11]: https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.synchronize.html
[H12]: https://docs.pytorch.org/docs/2.14/notes/randomness.html
[B01]: https://github.com/karpathy/nanoGPT
[B02]: https://github.com/karpathy/nanochat
[B03]: https://docs.pytorch.org/docs/2.14/generated/torch.optim.Muon.html
[D01]: https://huggingface.co/datasets/Salesforce/wikitext/blob/main/README.md
[D02]: https://huggingface.co/datasets/Salesforce/wikitext/commit/b08601e04326c79dfdd32d625aee71d232d685c3
[D03]: https://state.smerity.com/direct/smerity/state/01HRTBE0WN2QV8PRC6JDH22DGW
[D04]: https://huggingface.co/docs/huggingface_hub/en/guides/download
[D05]: https://huggingface.co/datasets/Salesforce/wikitext/tree/b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-2-raw-v1
[D06]: https://huggingface.co/datasets/Skylion007/openwebtext/blob/main/README.md
[D07]: https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/blob/main/README.md
[D08]: https://github.com/huggingface/tokenizers/releases
[D09]: https://ekzhu.com/datasketch/documentation.html
[S01]: https://proceedings.mlsys.org/paper_files/paper/2021/hash/0184b0cd3cfb185989f858a1d9f5c1eb-Abstract.html
[S02]: https://arxiv.org/abs/0712.1111
[S03]: https://doi.org/10.1214/12-AOAS547
[PAT01]: https://patents.google.com/patent/WO2017201506A1/en
[PAT02]: https://patents.google.com/patent/WO2024000566A1/en
[DEP01]: https://deepseek.com/en/news/deepseek-v3/
